import Mathlib

namespace initial_bags_count_l697_697159

theorem initial_bags_count
  (points_per_bag : ℕ)
  (non_recycled_bags : ℕ)
  (total_possible_points : ℕ)
  (points_earned : ℕ)
  (B : ℕ)
  (h1 : points_per_bag = 5)
  (h2 : non_recycled_bags = 2)
  (h3 : total_possible_points = 45)
  (h4 : points_earned = 5 * (B - non_recycled_bags))
  : B = 11 :=
by {
  sorry
}

end initial_bags_count_l697_697159


namespace prime_factor_of_difference_l697_697068

theorem prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A) (hA9 : A ≤ 9) (hC : 1 ≤ C) (hC9 : C ≤ 9) (hA_ne_C : A ≠ C) :
  ∃ p : ℕ, Prime p ∧ p = 3 ∧ p ∣ 3 * (100 * A + 10 * B + C - (100 * C + 10 * B + A)) := by
  sorry

end prime_factor_of_difference_l697_697068


namespace uncle_age_when_seokjin_is_12_l697_697516

-- Definitions for the conditions
def mother_age_when_seokjin_born : ℕ := 32
def uncle_is_younger_by : ℕ := 3
def seokjin_age : ℕ := 12

-- Definition for the main hypothesis
theorem uncle_age_when_seokjin_is_12 :
  let mother_age_when_seokjin_is_12 := mother_age_when_seokjin_born + seokjin_age
  let uncle_age_when_seokjin_is_12 := mother_age_when_seokjin_is_12 - uncle_is_younger_by
  uncle_age_when_seokjin_is_12 = 41 :=
by
  sorry

end uncle_age_when_seokjin_is_12_l697_697516


namespace confidence_level_l697_697383

def k_value : ℝ := 4.073
def P_3_841 : ℝ := 0.05
def P_5_024 : ℝ := 0.025
def thresh_3_841 : ℝ := 3.841
def thresh_5_024 : ℝ := 5.024

theorem confidence_level 
  (k_value > thresh_3_841) 
  (P_3_841 ≈ 0.05)
  (P_5_024 ≈ 0.025) :
  "confidence level 95%" := sorry

end confidence_level_l697_697383


namespace hyperbola_equation_l697_697795

-- Points F1, F2 are the foci and the hyperbola C is centered at the origin
-- with axes along the coordinate axes.
variables (F1 F2 : ℝ × ℝ) (C : Type) [hyperbola C F1 F2]
variables (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) [line_through l F2 A B]

-- Given points I1 I2 are incenters of triangles AF1F2 and BF1F2, respectively,
-- eccentricity of the hyperbola is 2, |I1 I2| = 9/2, and sine of the angle
-- of inclination of the line l is 8/9.
variables (I1 I2 : ℝ × ℝ) (eccentricity : ℝ) (distance_I1I2 : ℝ)
variables (sin_angle : ℝ)

-- The statement to prove
theorem hyperbola_equation :
  eccentricity = 2 → distance_I1I2 = 9 / 2 → sin_angle = 8 / 9 →
  equation_of_hyperbola C = (λ x y, x^2 / 4 - y^2 / 12 = 1) := by
  sorry

end hyperbola_equation_l697_697795


namespace sum_of_exceptions_l697_697273

def f (x : ℝ) := 4 * x / (3 * x^2 - 9 * x + 6)

theorem sum_of_exceptions : 
  let exceptions := {x : ℝ | 3 * x^2 - 9 * x + 6 = 0} in
  ∃ s : ℝ, s = 1 + 2 ∧ exceptions = {1, 2} ∧ s = 3 := 
by 
  sorry

end sum_of_exceptions_l697_697273


namespace double_of_quarter_of_four_percent_is_correct_l697_697161

-- Define 4 percent as a decimal
def four_percent : ℝ := 0.04

-- Define a quarter of a given number
def quarter (x : ℝ) : ℝ := x / 4

-- Define the double of a given number
def double (x : ℝ) : ℝ := x * 2

-- The proof statement rewriting the math problem
theorem double_of_quarter_of_four_percent_is_correct :
  double (quarter four_percent) = 0.02 :=
by
  sorry

end double_of_quarter_of_four_percent_is_correct_l697_697161


namespace smallest_power_of_13_non_palindrome_l697_697717

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l697_697717


namespace target_cube_side_length_l697_697711

def volume_of_cube (s : ℝ) : ℝ := s^3

def surface_area_of_cube (s : ℝ) : ℝ := 6 * s^2

theorem target_cube_side_length (s_target : ℝ) (s_base : ℝ) (V_base : ℝ) (H1 : volume_of_cube s_base = V_base) (H2 : surface_area_of_cube s_target = 3 * surface_area_of_cube s_base) :
  s_target = 2 * real.sqrt 3 :=
by sorry

end target_cube_side_length_l697_697711


namespace complex_number_solution_l697_697695

open Complex

noncomputable def findComplexNumber (z : ℂ) : Prop :=
  abs (z - 2) = abs (z + 4) ∧ abs (z - 2) = abs (z - Complex.I * 2)

theorem complex_number_solution :
  ∃ z : ℂ, findComplexNumber z ∧ z = -1 + Complex.I :=
by
  sorry

end complex_number_solution_l697_697695


namespace find_vector_AC_l697_697506

-- Assume we are working in an affine space with vectors.

noncomputable def vector_AC (a b : ℝ) : ℝ :=
  b - (1 + Real.sqrt 5) / 2 * a

theorem find_vector_AC 
  {a b : ℝ} -- Given vectors a and b as real numbers
  (circum_interval : ∀ A B C D E : affine_space.point, -- This represents that the points are on a circle with equal intervals
    |A - D| = |B - D| ∧ |C - D| = |E - D|)
  (vec_a : vector a) (vec_b : vector b) : 
  -- Prove the expression for AC
  (vector_AC a b) = b - ((1 + Real.sqrt 5) / 2) * a :=
sorry -- proof will be filled in later

end find_vector_AC_l697_697506


namespace problem_solution_l697_697654

theorem problem_solution
  (y1 y2 y3 y4 y5 y6 y7 : ℝ)
  (h1 : y1 + 3*y2 + 5*y3 + 7*y4 + 9*y5 + 11*y6 + 13*y7 = 0)
  (h2 : 3*y1 + 5*y2 + 7*y3 + 9*y4 + 11*y5 + 13*y6 + 15*y7 = 10)
  (h3 : 5*y1 + 7*y2 + 9*y3 + 11*y4 + 13*y5 + 15*y6 + 17*y7 = 104) :
  7*y1 + 9*y2 + 11*y3 + 13*y4 + 15*y5 + 17*y6 + 19*y7 = 282 := by
  sorry

end problem_solution_l697_697654


namespace log_eq_implies_value_l697_697690

theorem log_eq_implies_value (x : ℝ) (h : log x 256 = -4) : x = 1 / 4 := 
sorry

end log_eq_implies_value_l697_697690


namespace sec_minus_tan_l697_697831

theorem sec_minus_tan (x : ℝ) (h : sec x + tan x = 3) : sec x - tan x = 1 / 3 :=
sorry

end sec_minus_tan_l697_697831


namespace special_ten_digit_numbers_l697_697906

theorem special_ten_digit_numbers :
  let factorial := Nat.factorial in
  414 * factorial 9 = (number of special ten-digit numbers) :=
sorry

end special_ten_digit_numbers_l697_697906


namespace math_group_question_count_l697_697205

theorem math_group_question_count (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m = 6 ∧ n = 3 := 
sorry

end math_group_question_count_l697_697205


namespace trigonometric_identity_l697_697382

variable (θ : ℝ)

-- Given conditions
def vertex_at_origin (θ : ℝ) : Prop := True
def initial_side_positive_x_axis (θ : ℝ) : Prop := True
def terminal_side_on_line_3x_minus_y_eq_zero (θ : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ 0 ∧ y = 3 * x ∧ θ = Real.arctan (3)

-- Proof statement
theorem trigonometric_identity
  (h1 : vertex_at_origin θ)
  (h2 : initial_side_positive_x_axis θ)
  (h3 : terminal_side_on_line_3x_minus_y_eq_zero θ) :
  (sin θ + cos (π - θ)) / (sin (π / 2 - θ) - sin (π + θ)) = 1 / 2 :=
sorry

end trigonometric_identity_l697_697382


namespace three_digit_square_ends_with_self_l697_697312

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l697_697312


namespace not_rain_both_rain_one_exactly_rain_at_least_one_rain_at_most_one_l697_697282

namespace RainProbability

variables (P : Set → ℝ) (A B : Set)
variables (probA : P A = 0.2) (probB : P B = 0.3)
variables (independent : P (A ∩ B) = P A * P B)

open Set

-- 1. Probability that it does not rain in both places A and B
theorem not_rain_both : P (Aᶜ ∩ Bᶜ) = 0.56 := sorry

-- 2. Probability that it rains in exactly one of places A or B
theorem rain_one_exactly : P ((A ∩ Bᶜ) ∪ (Aᶜ ∩ B)) = 0.38 := sorry

-- 3. Probability that it rains in at least one of places A or B
theorem rain_at_least_one : P (A ∪ B) = 0.44 := sorry

-- 4. Probability that it rains in at most one of places A or B
theorem rain_at_most_one : P ((A ∩ Bᶜ) ∪ (Aᶜ ∩ B) ∪ (Aᶜ ∩ Bᶜ)) = 0.94 := sorry

end RainProbability

end not_rain_both_rain_one_exactly_rain_at_least_one_rain_at_most_one_l697_697282


namespace folded_paper_perimeter_l697_697210

theorem folded_paper_perimeter (L W : ℝ) 
  (h1 : 2 * L + W = 34)         -- Condition 1
  (h2 : L * W = 140)            -- Condition 2
  : 2 * W + L = 38 :=           -- Goal
sorry

end folded_paper_perimeter_l697_697210


namespace joe_paint_initial_amount_l697_697880

theorem joe_paint_initial_amount (P : ℝ) 
    (h1 : Joe uses (1/3) * P gallons of paint during the first week) 
    (h2 : Joe uses (1/5) * (2/3) * P gallons of paint during the second week)
    (h3 : Joe has used 168 gallons of paint in total) : 
    (1/3) * P + (2/15) * P = 168 → P = 360 :=
by
  sorry

end joe_paint_initial_amount_l697_697880


namespace inequality_solution_l697_697114

theorem inequality_solution (x : ℝ) :
  ((x - 1) * (x - 3) * (x - 5)) / ((x - 2) * (x - 4) * (x - 6)) > 0 ↔
  (x ∈ Iio 1 ∨ x ∈ Ioo 2 3 ∨ x ∈ Ioo 4 5 ∨ x ∈ Ioi 6) :=
by sorry

end inequality_solution_l697_697114


namespace Alex_sandwich_varieties_l697_697242

theorem Alex_sandwich_varieties :
  let meats := 12
  let cheeses := 11
  let breads := 3
  let choose (n k : ℕ) := nat.choose n k
  (
    meats * 
    (choose cheeses 0 + choose cheeses 1 + choose cheeses 2) *
    breads = 2412
  ) := 
sorry

end Alex_sandwich_varieties_l697_697242


namespace equal_lengths_l697_697869

-- Given conditions
axiom triangle_ABC (A B C : Point) : Triangle A B C
axiom isosceles_triangle (A B C : Point) (h : triangle_ABC A B C) : side_length A B = side_length A C
axiom circle_B (A B C D : Point) (h : triangle_ABC A B C) : intersect_circle_line B (side_length B C) (line A C) = D ∧ D ≠ C
axiom circle_D (A B C D E : Point) (h : triangle_ABC A B C) (h1 : circle_B A B C D) : intersect_circle_line D (side_length B C) (line A C) = E
axiom reflection_BD_BE (A B C D E : Point) (h : triangle_ABC A B C) (h1 : circle_B A B C D) (h2 : circle_D A B C D E) : 
  reflect_over_angle_bisector B D (angle_bisector A B C) = BE_line_segment E

-- To prove: EA = EB
theorem equal_lengths (A B C D E : Point) 
  (h : triangle_ABC A B C)
  (h1 : isosceles_triangle A B C h)
  (h2 : circle_B A B C D h)
  (h3 : circle_D A B C D E h h2)
  (h4 : reflection_BD_BE A B C D E h h2 h3) 
  : length_of_segment E A = length_of_segment E B :=
sorry

end equal_lengths_l697_697869


namespace maximum_distance_sum_upper_bound_l697_697405

variable {ℝ : Type*} [InnerProductSpace ℝ (ℝ^3)]

noncomputable def max_distance_sum (a b c d : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) : ℝ :=
‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2

theorem maximum_distance_sum_upper_bound 
  (a b c d : ℝ^3) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  max_distance_sum a b c d ha hb hc hd ≤ 16 := 
sorry

end maximum_distance_sum_upper_bound_l697_697405


namespace ann_frosting_time_l697_697499

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end ann_frosting_time_l697_697499


namespace solve_for_h_l697_697524

def f (x : ℝ) : ℝ := x^4 - x^3 - 1
def g (x : ℝ) : ℝ := x^8 - x^6 - 2x^4 + 1  
def h (x : ℝ) : ℝ := x^4 + x^3 - 1

theorem solve_for_h : ∀ x : ℝ, (g x = f x * h x) :=
by
  sorry

end solve_for_h_l697_697524


namespace total_cards_in_stack_l697_697122

theorem total_cards_in_stack (n : ℕ) (ordered_stack : List ℕ)
  (h_length : ordered_stack.length = 2 * n)
  (h_consecutive : ∀ i (hi : i < 2 * n), ordered_stack.nth i = some (i + 1))
  (A B : List ℕ)
  (h_A : A = ordered_stack.take n)
  (h_B : B = ordered_stack.drop n)
  (restacked : List ℕ)
  (h_restacked : restacked = restack_alternately A B)
  (h_positions_unchanged : ∀ k (hk : k < 2 * n), restacked.nth k = ordered_stack.nth k)
  (card_84 : ∃ k, ordered_stack.nth k = some 84 ∧ ordered_stack.nth k = restacked.nth k):
  2 * n = 252 := sorry

end total_cards_in_stack_l697_697122


namespace valid_votes_computation_votes_for_candidate_a_votes_for_candidate_b_votes_for_candidate_c_votes_for_candidate_d_l697_697046

def total_votes : ℕ := 1000000
def invalid_percentage : ℝ := 0.25
def valid_percentage : ℝ := 1 - invalid_percentage

def candidate_a_percentage : ℝ := 0.45
def candidate_b_percentage : ℝ := 0.30
def candidate_c_percentage : ℝ := 0.20
def candidate_d_percentage : ℝ := 0.05

def compute_valid_votes (total_votes : ℕ) (valid_percentage : ℝ) : ℕ :=
  total_votes * valid_percentage

def votes_per_candidate (valid_votes : ℕ) (candidate_percentage : ℝ) : ℕ :=
  valid_votes * candidate_percentage

theorem valid_votes_computation :
  compute_valid_votes total_votes valid_percentage = 750000 := by
  sorry

theorem votes_for_candidate_a :
  votes_per_candidate (compute_valid_votes total_votes valid_percentage) candidate_a_percentage = 337500 := by
  sorry

theorem votes_for_candidate_b :
  votes_per_candidate (compute_valid_votes total_votes valid_percentage) candidate_b_percentage = 225000 := by
  sorry

theorem votes_for_candidate_c :
  votes_per_candidate (compute_valid_votes total_votes valid_percentage) candidate_c_percentage = 150000 := by
  sorry

theorem votes_for_candidate_d :
  votes_per_candidate (compute_valid_votes total_votes valid_percentage) candidate_d_percentage = 37500 := by
  sorry

end valid_votes_computation_votes_for_candidate_a_votes_for_candidate_b_votes_for_candidate_c_votes_for_candidate_d_l697_697046


namespace find_perimeter_l697_697782

variables {a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 l w : ℕ}

def rectangle_squares_conditions (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ) : Prop :=
  a_1 + a_2 = a_3 ∧
  a_1 + a_3 = a_4 ∧
  a_3 + a_4 = a_5 ∧
  a_4 + a_5 = a_6 ∧
  a_2 + 2 * a_3 = a_7 ∧
  a_2 + a_7 = a_8 ∧
  a_1 + a_4 + a_6 = a_9 ∧
  a_6 + a_9 = a_7 + a_8

def dimensions (a_9 a_6 a_8 a_7 : ℕ) : ℕ × ℕ :=
  (a_9, a_6 + a_8 - a_7)

theorem find_perimeter
  (h : ∃ a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9,
    rectangle_squares_conditions a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 ∧
    (is_coprime (a_6 + a_8 - a_7) a_9)) :
  ∃ l w : ℕ, 2 * (l + w) = 220 :=
begin
  obtain ⟨a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, hcond, hcoprime⟩ := h,
  let d := dimensions a_9 a_6 a_8 a_7,
  use d.fst,
  use d.snd,
  sorry
end

end find_perimeter_l697_697782


namespace isosceles_triangle_angle_l697_697793

-- Define the triangle \triangle ABC with the given conditions
def Triangle (A B C : Point) :=
  ∃ (v1 v2 v3 : Fin 3 → ℝ), 
    A = v1 ∧ B = v2 ∧ C = v3 ∧ 
    dist A B = dist A C ∧ -- Condition: AB = AC
    abs (angle v1 v2 v3) + abs (angle v2 v3 v1) + abs (angle v3 v1 v2) = 180  -- Property: sum of angles of a triangle is 180°

-- Define the proposition to prove
def Prop (A B C : Point) : Prop :=
  ∀ (t : Triangle A B C), 
  angle A B C < 90

-- Rewrite the problem in Lean statement
theorem isosceles_triangle_angle (A B C : Point) (h : Triangle A B C) :
  Prop A B C :=
by
  sorry

end isosceles_triangle_angle_l697_697793


namespace period_of_sin_x_cos_x_l697_697276

theorem period_of_sin_x_cos_x : ∀ x ∈ ℝ, (sin x * cos x).period = π :=
by
suffices : ∀ x ∈ ℝ, sin x * cos x = (1/2) * sin (2*x)
  sorry
sorry

end period_of_sin_x_cos_x_l697_697276


namespace min_S_value_l697_697439

theorem min_S_value (n : ℕ) (m : ℕ) (abs_value : ∀ (i j : ℕ), |abs_value i j| ≤ 1) :
  n = 200 ∧ m = 200 ∧ (∑ i, ∑ j, abs_value i j = 0) →
  ∃ r : ℕ, (r < n) → |∑ j, abs_value r j| ≤ 100 ∨
  ∃ c : ℕ, (c < m) → |∑ i, abs_value i c| ≤ 100 :=
sorry

end min_S_value_l697_697439


namespace number_of_rows_l697_697093

-- Definitions of conditions
def tomatoes : ℕ := 3 * 5
def cucumbers : ℕ := 5 * 4
def potatoes : ℕ := 30
def additional_vegetables : ℕ := 85
def spaces_per_row : ℕ := 15

-- Total number of vegetables already planted
def planted_vegetables : ℕ := tomatoes + cucumbers + potatoes

-- Total capacity of the garden
def garden_capacity : ℕ := planted_vegetables + additional_vegetables

-- Number of rows in the garden
def rows_in_garden : ℕ := garden_capacity / spaces_per_row

theorem number_of_rows : rows_in_garden = 10 := by
  sorry

end number_of_rows_l697_697093


namespace Claire_remaining_balance_l697_697672

/-- 
Claire's purchases on specific days:
- Monday, Wednesday, Friday: $3.75 (latte) + $3.50 (croissant) + $2.25 (bagel) = $9.50 (except Friday which has $4.50 special drink)
- Tuesday, Thursday: $3.75 (latte) + $3.50 (croissant) + $2.50 (muffin) = $9.75
- Saturday: $3.75 (latte) + $3.50 * 0.9 (10% off croissant) = $6.90
- Sunday: $3.75 * 0.8 (20% off latte) + $3.50 (croissant) = $6.50
- She also buys 5 cookies at $1.25 each, costing $6.25
The total spent amount is $68.40.
Starting with $100, Claire has $31.60 left after purchases.
-/
theorem Claire_remaining_balance :
  let latte := 3.75
  let croissant := 3.50
  let bagel := 2.25
  let muffin := 2.50
  let special_drink := 4.50
  let cookie := 1.25

  let mon_wed_fri_total (latte croissant bagel) := latte + croissant + bagel
  let tue_thu_total (latte croissant muffin) := latte + croissant + muffin
  let sat_total (latte croissant) := latte + (croissant * 0.9)
  let sun_total (latte croissant) := (latte * 0.8) + croissant
  let bag_friday_total (special_drink croissant bagel) := special_drink + croissant + bagel

  let daily_expenses := mon_wed_fri_total latte croissant bagel + tue_thu_total latte croissant muffin + mon_wed_fri_total latte croissant bagel + tue_thu_total latte croissant muffin + bag_friday_total special_drink croissant bagel + sat_total latte croissant + sun_total latte croissant + 5 * cookie
  let initial_balance := 100
  let total_spent := 68.40
  let remaining_balance := initial_balance - total_spent in
    remaining_balance = 31.60 :=
by
  let latte := 3.75
  let croissant := 3.50
  let bagel := 2.25
  let muffin := 2.50
  let special_drink := 4.50
  let cookie := 1.25

  let mon_wed_fri_total := latte + croissant + bagel
  let tue_thu_total := latte + croissant + muffin
  let sat_total := latte + (croissant * 0.9)
  let sun_total := (latte * 0.8) + croissant
  let bag_friday_total := special_drink + croissant + bagel
  let daily_expenses := mon_wed_fri_total + tue_thu_total + mon_wed_fri_total + tue_thu_total + bag_friday_total + sat_total + sun_total + 5 * cookie
  let initial_balance := 100
  let total_spent := 68.40
  let remaining_balance := initial_balance - total_spent
  sorry

end Claire_remaining_balance_l697_697672


namespace nat_total_distance_350_l697_697912

def nat_distance_total : ℕ :=
  let monday := 40
  let tuesday := 50
  let wednesday := 0.5 * tuesday
  let thursday := monday + wednesday
  let friday := thursday + 0.2 * thursday
  let saturday := 0.75 * friday
  let sunday := saturday - wednesday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem nat_total_distance_350 :
  nat_distance_total = 350 :=
by
  sorry

end nat_total_distance_350_l697_697912


namespace average_monthly_growth_rate_l697_697086

-- Define the initial conditions
def profit_march : ℝ := 5000
def profit_may : ℝ := 7200
def months_between : ℕ := 2

-- Define the equation representing the profit growth
noncomputable def growth_rate (x : ℝ) : Prop :=
  profit_march * (1 + x)^2 = profit_may

-- State the theorem to be proved
theorem average_monthly_growth_rate :
  ∃ x : ℝ, growth_rate x ∧ x = 0.2 :=
begin
  sorry
end

end average_monthly_growth_rate_l697_697086


namespace smallest_non_palindromic_power_of_13_l697_697746

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l697_697746


namespace max_value_f_on_interval_l697_697386

noncomputable def f (x a : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem max_value_f_on_interval (a : ℝ) (a_pos : 0 < a) (a_lt_two : a < 2)
  (h_min : ∀ x ∈ set.Icc (1:ℝ) 4, f x a ≥ - (16 / 3)) :
  ∃ x ∈ set.Icc (1:ℝ) 4, f x a = 10 / 3 :=
  sorry

end max_value_f_on_interval_l697_697386


namespace find_arc_points_l697_697155

variable {O D A C : Point}
variable {arc_DA : Arc}
variable {r : ℝ}

-- Definitions based on conditions:
def initial_arc_DA := arc_DA
def circle_radius := 2 * (dist C D)

-- Statement of the problem:
theorem find_arc_points 
  (D A : Point) (O : Point) (arc_DA : Arc) (r := 2 * (dist C D)) :
  ∃ (A₁ A₂ A₃ : Point), 
    (∀ n : ℕ, arc_length (arc O D Aₙ₊₁) = arc_length arc_DA 
    ∧ construct_using_angle_bisectors O D Aₙ₊₁ Aₙ ) :=
sorry

end find_arc_points_l697_697155


namespace weight_difference_l697_697473

def weight_chemistry : ℝ := 7.12
def weight_geometry : ℝ := 0.62

theorem weight_difference : weight_chemistry - weight_geometry = 6.50 :=
by
  sorry

end weight_difference_l697_697473


namespace seq2_in_W_seq4_in_W_l697_697903

-- Define sequence 2: a_n = (2n + 9) / (2n + 11)
def seq2 (n : ℕ) : ℝ := (2 * n + 9) / (2 * n + 11)

-- Define sequence 4: a_n = 1 - 1 / 2^n
def seq4 (n : ℕ) : ℝ := 1 - 1 / (2^n)

-- Define the set W of sequences with the given properties
def in_set_W (a : ℕ → ℝ) : Prop :=
  (∀ n, (a n + a (n + 2)) / 2 < a (n + 1)) ∧
  ∃ M : ℝ, ∀ n, a n ≤ M

theorem seq2_in_W : in_set_W seq2 :=
sorry

theorem seq4_in_W : in_set_W seq4 :=
sorry

end seq2_in_W_seq4_in_W_l697_697903


namespace max_rubles_can_receive_l697_697919

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l697_697919


namespace harry_spends_1920_annually_l697_697003

def geckoCount : Nat := 3
def iguanaCount : Nat := 2
def snakeCount : Nat := 4

def geckoFeedTimesPerMonth : Nat := 2
def iguanaFeedTimesPerMonth : Nat := 3
def snakeFeedTimesPerMonth : Nat := 1 / 2

def geckoFeedCostPerMeal : Nat := 8
def iguanaFeedCostPerMeal : Nat := 12
def snakeFeedCostPerMeal : Nat := 20

def annualCostHarrySpends (geckoCount guCount scCount : Nat) (geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth : Nat) (geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal : Nat) : Nat :=
  let geckoAnnualCost := geckoCount * (geckoFeedTimesPerMonth * 12 * geckoFeedCostPerMeal)
  let iguanaAnnualCost := iguanaCount * (iguanaFeedTimesPerMonth * 12 * iguanaFeedCostPerMeal)
  let snakeAnnualCost := snakeCount * ((12 / (2 : Nat)) * snakeFeedCostPerMeal)
  geckoAnnualCost + iguanaAnnualCost + snakeAnnualCost

theorem harry_spends_1920_annually : annualCostHarrySpends geckoCount iguanaCount snakeCount geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal = 1920 := 
  sorry

end harry_spends_1920_annually_l697_697003


namespace martha_weight_l697_697664

theorem martha_weight :
  ∀ (Bridget_weight : ℕ) (difference : ℕ) (Martha_weight : ℕ),
  Bridget_weight = 39 → difference = 37 →
  Bridget_weight = Martha_weight + difference →
  Martha_weight = 2 :=
by
  intros Bridget_weight difference Martha_weight hBridget hDifference hRelation
  sorry

end martha_weight_l697_697664


namespace greatest_integer_inequality_l697_697164

theorem greatest_integer_inequality : 
  ⌊ (3 ^ 100 + 2 ^ 100 : ℝ) / (3 ^ 96 + 2 ^ 96) ⌋ = 80 :=
by
  sorry

end greatest_integer_inequality_l697_697164


namespace find_S25_l697_697791

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a 1 - a 0
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

-- Condition that given S_{15} - S_{10} = 1
axiom sum_difference : S 15 - S 10 = 1

-- Theorem we need to prove
theorem find_S25 (h_arith : is_arithmetic_sequence a) (h_sum : sum_of_first_n_terms a S) : S 25 = 5 :=
by
-- Placeholder for the actual proof
sorry

end find_S25_l697_697791


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697225

open Real

theorem height_of_cylinder_inscribed_in_hemisphere
  (r_cylinder r_hemisphere : ℝ)
  (h_radius_cylinder : r_cylinder = 3)
  (h_radius_hemisphere : r_hemisphere = 8) :
  ∃ h_cylinder : ℝ, h_cylinder = sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) :=
by
  use sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  sorry

#check height_of_cylinder_inscribed_in_hemisphere

end height_of_cylinder_inscribed_in_hemisphere_l697_697225


namespace Creekview_science_fair_l697_697680

/-- Given the total number of students at Creekview High School is 1500,
    900 of these students participate in a science fair, where three-quarters
    of the girls participate and two-thirds of the boys participate,
    prove that 900 girls participate in the science fair. -/
theorem Creekview_science_fair
  (g b : ℕ)
  (h1 : g + b = 1500)
  (h2 : (3 / 4) * g + (2 / 3) * b = 900) :
  (3 / 4) * g = 900 := by
sorry

end Creekview_science_fair_l697_697680


namespace ann_frosting_cakes_l697_697500

theorem ann_frosting_cakes (normalRate sprainedRate cakes : ℕ) (H1 : normalRate = 5) (H2 : sprainedRate = 8) (H3 : cakes = 10) :
  (sprainedRate * cakes) - (normalRate * cakes) = 30 :=
by
  -- Substitute the provided values into the expression
  rw [H1, H2, H3]
  -- Evaluate the expression
  norm_num

end ann_frosting_cakes_l697_697500


namespace a_seq_gt_one_l697_697488

noncomputable def a_seq (a : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1 + a
  else (1 / a_seq a (n - 1)) + a

theorem a_seq_gt_one (a : ℝ) (h : 0 < a ∧ a < 1) : ∀ n : ℕ, 1 < a_seq a n :=
by {
  sorry
}

end a_seq_gt_one_l697_697488


namespace alcohol_percentage_correct_in_mixed_solution_l697_697190

-- Define the ratios of alcohol to water
def ratio_A : ℚ := 21 / 25
def ratio_B : ℚ := 2 / 5

-- Define the mixing ratio of solutions A and B
def mix_ratio_A : ℚ := 5 / 11
def mix_ratio_B : ℚ := 6 / 11

-- Define the function to compute the percentage of alcohol in the mixed solution
def alcohol_percentage_mixed : ℚ := 
  (mix_ratio_A * ratio_A + mix_ratio_B * ratio_B) * 100

-- The theorem to be proven
theorem alcohol_percentage_correct_in_mixed_solution : 
  alcohol_percentage_mixed = 60 :=
by
  sorry

end alcohol_percentage_correct_in_mixed_solution_l697_697190


namespace minimum_of_tan_C_minus_cot_A_l697_697851

open Real

-- Definitions of conditions
variables {A B C D : Point}
variables {AC AD: Vector}
variables (m b : ℝ)
variables [Midpoint D B C]
variables (h1 : AC ≠ 0)
variables (h2 : AD ≠ 0)
variables (h3 : (AD • AC) = 0) -- AD.dot AC = 0

-- Definitions for the problem regarding tangent and cotangent
noncomputable def tan_C := m / b
noncomputable def cot_A := b / (2 * m)

-- Statement of the minimum value of tan C - cot A
theorem minimum_of_tan_C_minus_cot_A 
  (h4 : tan_C = m / b)
  (h5 : cot_A = b / (2 * m)) :
  ∃ minimum_value, minimum_value = sqrt 2 ∧ ∀ x, tan_C - cot_A ≥ minimum_value :=
sorry

end minimum_of_tan_C_minus_cot_A_l697_697851


namespace complement_U_M_correct_l697_697817

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}
def complement_U_M : Set ℕ := {1, 2, 3}

theorem complement_U_M_correct : U \ M = complement_U_M :=
  by sorry

end complement_U_M_correct_l697_697817


namespace integral_2x_minus_5_cos_4x_l697_697256

variable (x : ℝ)

theorem integral_2x_minus_5_cos_4x :
  ∫ (2 * x - 5) * cos (4 * x) dx =
    (1 / 4) * (2 * x - 5) * sin (4 * x) + (1 / 8) * cos (4 * x) + C :=
sorry

end integral_2x_minus_5_cos_4x_l697_697256


namespace mersenne_prime_approximation_l697_697508

theorem mersenne_prime_approximation :
  let N := 2^607 - 1
  let M := 2^127 - 1
  let log2_approx := 0.3010 in
  let k := 2^480 in
  let log_k := 480 * log2_approx in
  10^144 ≈ 10^144.48 :=
begin
  sorry
end

end mersenne_prime_approximation_l697_697508


namespace combined_mixture_percentages_l697_697575

-- Definitions of the percentages and volumes given in the problem
def first_container_volume : ℤ := 40
def first_container_chemicalA_percentage : ℝ := 0.50
def first_container_chemicalB_percentage : ℝ := 0.30
def first_container_chemicalC_percentage : ℝ := 0.20

def second_container_volume : ℤ := 60
def second_container_chemicalA_percentage : ℝ := 0.40
def second_container_chemicalB_percentage : ℝ := 0.10
def second_container_chemicalC_percentage : ℝ := 0.50

-- The total amounts of chemicals in each container
def first_container_chemicalA : ℝ := first_container_volume * first_container_chemicalA_percentage
def first_container_chemicalB : ℝ := first_container_volume * first_container_chemicalB_percentage
def first_container_chemicalC : ℝ := first_container_volume * first_container_chemicalC_percentage

def second_container_chemicalA : ℝ := second_container_volume * second_container_chemicalA_percentage
def second_container_chemicalB : ℝ := second_container_volume * second_container_chemicalB_percentage
def second_container_chemicalC : ℝ := second_container_volume * second_container_chemicalC_percentage

-- The sums of chemicals when the containers are combined
def total_chemicalA : ℝ := first_container_chemicalA + second_container_chemicalA
def total_chemicalB : ℝ := first_container_chemicalB + second_container_chemicalB
def total_chemicalC : ℝ := first_container_chemicalC + second_container_chemicalC

-- The total volume when the containers are combined
def total_volume : ℤ := first_container_volume + second_container_volume

-- The resulting percentages
def resulting_percentage_chemicalA : ℝ := total_chemicalA / (total_volume:ℝ) * 100
def resulting_percentage_chemicalB : ℝ := total_chemicalB / (total_volume:ℝ) * 100
def resulting_percentage_chemicalC : ℝ := total_chemicalC / (total_volume:ℝ) * 100

-- The theorem to be proven
theorem combined_mixture_percentages :
  resulting_percentage_chemicalA = 44 ∧
  resulting_percentage_chemicalB = 18 ∧
  resulting_percentage_chemicalC = 38 :=
by {
  -- Proof goes here, using sorry for now
  sorry
}

end combined_mixture_percentages_l697_697575


namespace Sn_times_Tn_eq_third_l697_697764

def S_n (n : ℕ) : ℚ :=
  ∑ k in Finset.range(n), k.succ / (1 + k.succ^2 + k.succ^4)

def T_n (n : ℕ) : ℚ :=
  ∏ k in Finset.range(n - 1), (k + 2)^3 - 1 / ((k + 2)^3 + 1)

theorem Sn_times_Tn_eq_third (n : ℕ) (hn : 2 ≤ n) : S_n n * T_n n = 1 / 3 :=
by
  sorry

end Sn_times_Tn_eq_third_l697_697764


namespace find_m_n_l697_697066

def parabola (P : ℝ → ℝ) : Prop :=
  ∃ a b c, ∀ x, P x = a*x^2 + b*x + c ∧ P 5 = 0 ∧ P 0 = 12

def P_friendly (ℓ : ℝ → ℝ) (P : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, ℓ x₁ = 0 ∧ ℓ x₂ = 12 ∧ P x₃ = 0 ∧
  (x₁^2 + x₂^2 + x₃^2) = 0 ∧ 
  abs (x₁ - x₂) = abs (x₂ - x₃)

noncomputable def sum_of_slopes_of_P_friendly_lines (P : ℝ → ℝ) : ℤ :=
  sorry  -- Calculation steps omitted for brevity

theorem find_m_n : 
  ∃ m n : ℤ, gcd m n = 1 ∧ 
  (-sum_of_slopes_of_P_friendly_lines (parabola P) = m / n) ∧
  m + n = 437 :=
sorry  -- Proof is omitted as the focus is on the statement

end find_m_n_l697_697066


namespace mod_problem_l697_697841

theorem mod_problem (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 21 [ZMOD 25]) : (x^2 ≡ 21 [ZMOD 25]) :=
sorry

end mod_problem_l697_697841


namespace combined_flock_after_5_years_l697_697689

theorem combined_flock_after_5_years :
  let initial_flock : ℕ := 100
  let annual_net_gain : ℕ := 30 - 20
  let years : ℕ := 5
  let joined_flock : ℕ := 150
  let final_flock := initial_flock + annual_net_gain * years + joined_flock
  in final_flock = 300 :=
by
  sorry

end combined_flock_after_5_years_l697_697689


namespace continuity_sufficient_but_not_necessary_l697_697996

noncomputable def is_continuous_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε

noncomputable def is_defined_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
∃ y, f x₀ = y

theorem continuity_sufficient_but_not_necessary (f : ℝ → ℝ) (x₀ : ℝ) :
  is_continuous_at f x₀ ↔ is_defined_at f x₀ :=
begin
  sorry
end

end continuity_sufficient_but_not_necessary_l697_697996


namespace jesse_needs_more_carpet_l697_697059

def additional_carpet_needed (carpet : ℕ) (length : ℕ) (width : ℕ) : ℕ :=
  let room_area := length * width
  room_area - carpet

theorem jesse_needs_more_carpet
  (carpet : ℕ) (length : ℕ) (width : ℕ)
  (h_carpet : carpet = 18)
  (h_length : length = 4)
  (h_width : width = 20) :
  additional_carpet_needed carpet length width = 62 :=
by {
  -- the proof goes here
  sorry
}

end jesse_needs_more_carpet_l697_697059


namespace knights_at_positions_3_and_4_l697_697916

-- Define the types and constants
constant P : Type -- P represents positions

-- Define specific positions
constants P1 P2 P3 P4 P5 P6 : P

-- Define predicates for knights and liars
constant is_knight : P → Prop
constant is_liar : P → Prop

-- Establish that each person is either a knight or a liar
axiom knight_or_liar (p : P) : is_knight p ∨ is_liar p
-- Establish that there are exactly 3 knights and 3 liars
axiom three_knights_and_liars : 
  (finset.filter is_knight (finset.univ : finset P)).card = 3 ∧ 
  (finset.filter is_liar (finset.univ : finset P)).card = 3

-- Define the statements made by the islanders
axiom P1_statement : (is_knight P1 → (∃ p3, p3 = 3 ∧ is_knight p3)) ∧ (is_liar P1 → ¬(∃ p3, p3 = 3 ∧ is_knight p3))
axiom P4_statement : (is_knight P4 → (∃ p1, p1 = 1 ∧ is_knight p1)) ∧ (is_liar P4 → ¬(∃ p1, p1 = 1 ∧ is_knight p1))
axiom P5_statement : (is_knight P5 → (∃ p2, p2 = 2 ∧ is_knight p2)) ∧ (is_liar P5 → ¬(∃ p2, p2 = 2 ∧ is_knight p2))

-- Define the proof problem, which states that P3 and P4 are always knights
theorem knights_at_positions_3_and_4 : is_knight P3 ∧ is_knight P4 :=
by {
  sorry -- Proof to be filled in
}

end knights_at_positions_3_and_4_l697_697916


namespace find_A_to_make_B_max_l697_697674

theorem find_A_to_make_B_max (A B : ℕ) (q : ℕ) (h1 : q = 33) (h2 : A = 13 * q + B) (h3 : 0 ≤ B ∧ B < 13) :
  A = 441 :=
by {
  have hB_max : B = 12,
  { sorry },
  rw [h1, hB_max] at h2,
  calc
    A = 13 * 33 + 12 : by {exact h2}
    ... = 429 + 12 : by {norm_num}
    ... = 441 : by {norm_num}
}

end find_A_to_make_B_max_l697_697674


namespace divide_and_add_l697_697567

theorem divide_and_add (x : ℤ) (h1 : x = 95) : (x / 5) + 23 = 42 := by
  sorry

end divide_and_add_l697_697567


namespace three_digit_numbers_with_square_ending_in_them_l697_697302

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l697_697302


namespace probability_both_notes_counterfeit_l697_697769

axiom twenty_yuan_notes : ℕ
axiom counterfeit_notes : ℕ
axiom drawn_counterfeit_prob : ℚ

noncomputable def probability_two_counterfeit_given_one_counterfeit (total: ℕ) (counterfeit: ℕ) (one_counterfeit: ℚ) : ℚ :=
  let comb := λ (n k : ℕ), (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k)) in
  let P_A := comb counterfeit 2 / comb total 2 in
  let P_B := (comb counterfeit 2 + comb counterfeit 1 * comb (total - counterfeit) 1) / comb total 2 in
  P_A / P_B

theorem probability_both_notes_counterfeit :
  twenty_yuan_notes = 20 ∧ counterfeit_notes = 5 ∧ drawn_counterfeit_prob = (1/2) →
  probability_two_counterfeit_given_one_counterfeit 20 5 (1/2) = (2/17) := by
  intros
  sorry

end probability_both_notes_counterfeit_l697_697769


namespace broccoli_to_carrots_ratio_l697_697571

theorem broccoli_to_carrots_ratio (B : ℝ)
  (carrot_calories : ℝ) (broccoli_calories : ℝ) (total_calories : ℝ) 
  (carrots_eaten : ℝ) (broccoli_ratio_cal : broccoli_calories = carrot_calories / 3)
  (carrot_calories_per_pound : carrot_calories = 51)
  (total_calories_consumed : total_calories = 85)
  (carrots_amount : carrots_eaten = 1)
  (calories_equation : calories = carrot_calories * carrots_eaten + broccoli_calories * B) :
  broccoli_to_carrots_ratio_total: 2 = B := 
by
  sorry

end broccoli_to_carrots_ratio_l697_697571


namespace volume_of_cube_l697_697992

theorem volume_of_cube (d : ℝ) (h : d = 5 * Real.sqrt 3) : ∃ (V : ℝ), V = 125 := by
  sorry

end volume_of_cube_l697_697992


namespace find_set_B_l697_697798

set_option pp.all true

variable (A : Set ℤ) (B : Set ℤ)

theorem find_set_B (hA : A = {-2, 0, 1, 3})
                    (hB : B = {x | -x ∈ A ∧ 1 - x ∉ A}) :
  B = {-3, -1, 2} :=
by
  sorry

end find_set_B_l697_697798


namespace geom_proof_selected_topics_tangent_l697_697195

def circle (O : Type) := sorry
def point (A M P N B K O : Type) := sorry
def tangent (M A B : Type) := sorry
def perpendicular (A B P O N : Type) := sorry

theorem geom_proof_selected_topics_tangent :
  ∀ (O : Type) (M A P N B K : point) (tangent_MA : tangent M A) (perpendicular_AP_OM : perpendicular A P O M) (N_on_AP : N ∈ line A P) 
    (perpendicular_NB_ON : perpendicular N B O N) (tangent_BK_ON : tangent B K) 
    , ∠ K O M = 90 :=
by
  sorry

end geom_proof_selected_topics_tangent_l697_697195


namespace trains_crossing_time_l697_697592

noncomputable def relative_speed := (60 + 40) * (5/18 : ℚ) -- converting km/hr to m/s
noncomputable def total_distance := 150 + 160
noncomputable def crossing_time := total_distance / relative_speed

theorem trains_crossing_time :
  crossing_time ≈ 11.16 := 
sorry

end trains_crossing_time_l697_697592


namespace expression_independent_of_a_l697_697097

theorem expression_independent_of_a (a : ℝ) :
  7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 :=
by sorry

end expression_independent_of_a_l697_697097


namespace drawings_in_first_five_pages_l697_697876

theorem drawings_in_first_five_pages : ∑ i in range 5, 5 * (i + 1) = 75 :=
by
  sorry

end drawings_in_first_five_pages_l697_697876


namespace find_positive_integer_unique_positive_integer_l697_697127

theorem find_positive_integer (n : ℕ) : 20 - 5 * n > 12 → n < 2 := 
by
-- Proof steps go here
sorry

theorem unique_positive_integer : ∃! (n : ℕ), 20 - 5 * n > 12 :=
by
  have h : 20 - 5 * 1 > 12 := by simp
  use 1
  split
  { exact h }
  { intros m hm
    have h1 : 20 - 5 * m > 12 := hm
    have h2 : 20 - 12 > 5 * m := sub_pos.mp h1
    have h3 : 8 > 5 * m := h2
    have h4 : (8 : ℚ) / 5 > m := (div_lt_iff (by norm_num)).mpr h3
    have h5 : (1.6 : ℚ) > m := by conversion
    exact nat.lt_of_lt_pred_of_le (by linarith) (nat.le_of_lt_succ (nat.ceil_lt_add_one.mp h3)) }

end find_positive_integer_unique_positive_integer_l697_697127


namespace calculate_expression_l697_697669

theorem calculate_expression 
  (a1 : 84 + 4 / 19 = 1600 / 19) 
  (a2 : 105 + 5 / 19 = 2000 / 19) 
  (a3 : 1.375 = 11 / 8) 
  (a4 : 0.8 = 4 / 5) :
  84 * (4 / 19) * (11 / 8) + 105 * (5 / 19) * (4 / 5) = 200 := 
sorry

end calculate_expression_l697_697669


namespace cylinder_height_in_hemisphere_l697_697216

noncomputable def height_of_cylinder (r_hemisphere r_cylinder : ℝ) : ℝ :=
  real.sqrt (r_hemisphere^2 - r_cylinder^2)

theorem cylinder_height_in_hemisphere :
  let r_hemisphere := 8
  let r_cylinder := 3
  height_of_cylinder r_hemisphere r_cylinder = real.sqrt 55 :=
by
  sorry

end cylinder_height_in_hemisphere_l697_697216


namespace determine_location_d_l697_697645

/-- Determine specific location -/
def canDetermineSpecificLocation (desc : String) : Prop :=
  desc = "East longitude 102°, north latitude 24°"

/-- Main theorem: Determining the specific location based on given conditions -/
theorem determine_location_d (H_A : String = "The third row of Classroom 3 in Class 7")
                             (H_B : String = "People's East Road, Kunming City")
                             (H_C : String = "Southward and westward by 45°")
                             (H_D : String = "East longitude 102°, north latitude 24°")
                             : canDetermineSpecificLocation H_D :=
by {
  sorry
}

end determine_location_d_l697_697645


namespace articles_produced_l697_697014

theorem articles_produced (a b c d f p q r g : ℕ) :
  (a * b * c = d) → 
  ((p * q * r * d * g) / (a * b * c * f) = pqr * d * g / (abc * f)) :=
by
  sorry

end articles_produced_l697_697014


namespace exists_three_real_roots_l697_697478

-- Define the distinct nonzero real numbers
variables {a b c α : ℝ}
-- Assume distinct nonzero real numbers
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)

-- Define the cubic equations
def y1 (x : ℝ) : ℝ := a * x^3 + b * x + c
def y2 (x : ℝ) : ℝ := b * x^3 + c * x + a
def y3 (x : ℝ) : ℝ := c * x^3 + a * x + b

-- Assume there is a common root α
variables (hroot : y1 α = 0 ∧ y2 α = 0 ∧ y3 α = 0)

-- Define the statement to prove: at least one of the equations has 3 real roots
theorem exists_three_real_roots :
  (∃ x, 3 ≤ multiplicity (has_roots y1 x)) ∨
  (∃ x, 3 ≤ multiplicity (has_roots y2 x)) ∨
  (∃ x, 3 ≤ multiplicity (has_roots y3 x)) :=
  sorry

end exists_three_real_roots_l697_697478


namespace vector_cross_product_inequality_l697_697517

noncomputable def vector_magnitude_cross_product_inequality (a b : EuclideanSpace) : Prop :=
  ∥a × b∥^3 ≤ (3 * Real.sqrt 3 / 8) * ∥a∥^2 * ∥b∥^2 * ∥a - b∥^2

theorem vector_cross_product_inequality (a b : EuclideanSpace) : 
  vector_magnitude_cross_product_inequality a b :=
by sorry

end vector_cross_product_inequality_l697_697517


namespace remainder_poly_division_l697_697326

theorem remainder_poly_division (Q R : Polynomial ℤ) :
  (Q = Polynomial.X^101 / (Polynomial.X^2 - 5 * Polynomial.X + 6)) →
  (R = Polynomial.X^101 % (Polynomial.X^2 - 5 * Polynomial.X + 6)) →
  R = (Polynomial.C (3^101) - Polynomial.C (2^101)) * Polynomial.X + (Polynomial.C (2^101) - Polynomial.C (2 * 3^101)) :=
sorry

end remainder_poly_division_l697_697326


namespace prime_factor_of_difference_l697_697128

theorem prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A) (hA2 : A ≤ 9) (hB : 0 ≤ B) (hB2 : B ≤ 9) (hC : 1 ≤ C) (hC2 : C ≤ 9) (h : A ≠ C) :
  3 ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) :=
by
  dsimp at *
  rw [sub_sub, add_sub_assoc, sub_self, zero_add, sub_sub, sub_sub, add_sub_add_right_eq_sub]
  rw [mul_sub, mul_sub]
  exact dvd_mul_right 3 33 (A - C)

end prime_factor_of_difference_l697_697128


namespace petya_max_rubles_l697_697941

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l697_697941


namespace complex_conjugate_product_l697_697805

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number z based on the given condition
def z : ℂ := (i + i^2 + i^3) / (2 - i)

-- State the proof problem
theorem complex_conjugate_product : z * complex.conj(z) = 1 / 5 :=
by sorry

end complex_conjugate_product_l697_697805


namespace polygon_area_l697_697271

/-- Define the vertices of the polygon. --/
def vertices : List (ℝ × ℝ) := [(0, 0), (4, 0), (6, 3), (4, 6)]

/-- Shoelace Theorem formula for the area of a polygon given its vertices. --/
def shoelace_formula (V : List (ℝ × ℝ)) : ℝ :=
  let xy_pairs := V.zip (V.rotate 1)  -- Creating pairs (x_i, y_{i+1})
  abs ((xy_pairs.map (λ ⟨(x1, y1), (x2, y2)⟩ => x1 * y2 - y1 * x2)).sum) / 2

/-- The area of the given polygon is 18 square units. --/
theorem polygon_area : shoelace_formula vertices = 18 := by
  sorry

end polygon_area_l697_697271


namespace variance_of_data_set_l697_697786

theorem variance_of_data_set :
  let data_set := [2, 3, 4, 5, 6]
  let mean := (2 + 3 + 4 + 5 + 6) / 5
  let variance := (1 / 5 : Real) * ((2 - mean)^2 + (3 - mean)^2 + (4 - mean)^2 + (5 - mean)^2 + (6 - mean)^2)
  variance = 2 :=
by
  sorry

end variance_of_data_set_l697_697786


namespace sqrt_sum_inequality_l697_697067

theorem sqrt_sum_inequality (a b c : ℝ) (h1 : 0 < a)
  (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a = 1) : 
  sqrt (a^3 + a) + sqrt (b^3 + b) + sqrt (c^3 + c) ≥ 2 * sqrt (a + b + c) :=
by
  sorry

end sqrt_sum_inequality_l697_697067


namespace problem_1_problem_2_l697_697819

def vec2D (x y : ℝ) : Type := ℝ × ℝ

def a : vec2D := (3, 2)
def b : vec2D := (-1, 2)
def c : vec2D := (4, 1)

theorem problem_1 (k : ℝ) (n : ℝ) : (3 + 4 * k, 2 + k) = (4 * b.1 + k * c.1, 4 * b.2 + k * c.2) → k = -11 / 18 :=
by sorry

theorem problem_2 (m n : ℝ) : a = (m * b.1 - n * c.1, m * b.2 - n * c.2) →
    m = 5 / 9 ∧ n = -8 / 9 := 
by sorry

end problem_1_problem_2_l697_697819


namespace gcd_of_palindromes_eq_102_all_palindromes_divisible_by_3_l697_697163

-- Definitions for three-digit palindromes
def is_three_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 102 * a + 10 * b

-- GCD of all three-digit palindromes
def gcd_of_three_digit_palindromes : ℕ :=
  Nat.gcd 102 102

-- Proof statements
theorem gcd_of_palindromes_eq_102 : gcd_of_three_digit_palindromes = 102 :=
by
  unfold gcd_of_three_digit_palindromes
  exact Nat.gcd_refl 102
  -- sorry term added for simplifier, replace with actual proof

theorem all_palindromes_divisible_by_3 : ∀ n, is_three_digit_palindrome n → n % 3 = 0 :=
by
  intro n h
  obtain ⟨a, b, ha1, ha2, hb1, hb2, hn⟩ := h
  rw hn
  have h3 : 102 * a % 3 = 0 :=
    by
      rw ←Nat.mul_mod
      exact Nat.mod_eq_zero_of_dvd (dvd_mul_left 34 a)
  have h4 : 10 * b % 3 = 0 :=
    by
      rw ←Nat.mul_mod
      exact Nat.mod_eq_zero_of_dvd (dvd_mul_right 10 b)
  exact Nat.mod_eq_zero_of_dvd (show n % 3 = 0 by rw [hn, Nat.add_mod, h3, h4, zero_add, mod_zero])
  -- sorry term added for simplifier, replace with actual proof

end gcd_of_palindromes_eq_102_all_palindromes_divisible_by_3_l697_697163


namespace find_S25_l697_697790

variable (a : ℕ → ℚ) (S : ℕ → ℚ)

-- Conditions: arithmetic sequence {a_n} and sum of the first n terms S_n with S15 - S10 = 1
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a (i + 1)

axiom condition : S 15 - S 10 = 1

-- Question: Prove that S25 = 5
theorem find_S25 (h1 : arithmetic_sequence a) (h2 : sum_of_first_n_terms a S) : S 25 = 5 :=
by
  sorry

end find_S25_l697_697790


namespace three_digit_ends_with_itself_iff_l697_697306

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l697_697306


namespace sum_a2012_a2013_l697_697884

-- Define the geometric sequence and its conditions
def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n * q

-- Parameters for the problem
variable (a : ℕ → ℚ)
variable (q : ℚ)
variable (h_seq : geometric_sequence a q)
variable (h_q : 1 < q)
variable (h_eq : ∀ x : ℚ, 4 * x^2 - 8 * x + 3 = 0 → x = a 2010 ∨ x = a 2011)

-- Statement to prove
theorem sum_a2012_a2013 : a 2012 + a 2013 = 18 :=
by
  sorry

end sum_a2012_a2013_l697_697884


namespace probability_of_darkness_l697_697231

theorem probability_of_darkness (rev_per_min : ℕ) (stay_in_dark_time : ℕ) (revolution_time : ℕ) (stay_fraction : ℕ → ℚ) :
  rev_per_min = 2 →
  stay_in_dark_time = 10 →
  revolution_time = 60 / rev_per_min →
  stay_fraction stay_in_dark_time / revolution_time = 1 / 3 :=
by
  sorry

end probability_of_darkness_l697_697231


namespace _l697_697836

statement theorem sec_tan_difference (x : ℝ) (h : Real.sec x + Real.tan x = 3) : 
  Real.sec x - Real.tan x = 1 / 3 := by
  sorry

end _l697_697836


namespace total_miles_february_l697_697873

-- Defining the individual and group walking distances
def group_walk_per_week : ℕ := 3 * 6  -- 18 miles per week
def weeks_in_february : ℕ := 4
def group_walk_february : ℕ := group_walk_per_week * weeks_in_february -- 72 miles

def extra_walk_jamie_per_week : ℕ := 2 * 3  -- 6 miles per week
def total_walk_jamie : ℕ := extra_walk_jamie_per_week * weeks_in_february -- 24 miles

def extra_walk_sue_per_week : ℕ := (3 / 2) * 3  -- 4.5 miles per week
def total_walk_sue : ℕ := extra_walk_sue_per_week * weeks_in_february -- 18 miles

def extra_walk_laura_per_week : ℕ := (1 * 2) + (1.5 * 3)  -- 6.5 miles per week
def total_walk_laura : ℕ := extra_walk_laura_per_week * weeks_in_february -- 26 miles

def extra_walk_melissa_per_week : ℕ := (2 * 2) + 4  -- 8 miles per week
def total_walk_melissa : ℕ := extra_walk_melissa_per_week * weeks_in_february -- 32 miles

def extra_walk_katie_per_week : ℕ := (1 * 6) + 3  -- 9 miles per week
def total_walk_katie : ℕ := extra_walk_katie_per_week * weeks_in_february -- 36 miles

-- Total miles walked by the group and the individuals
def total_miles_walked : ℕ := group_walk_february + total_walk_jamie + total_walk_sue + total_walk_laura + total_walk_melissa + total_walk_katie

theorem total_miles_february : total_miles_walked = 208 :=
by
  -- Proof goes here
  sorry

end total_miles_february_l697_697873


namespace total_money_is_correct_l697_697011

-- Define conditions as constants
def numChocolateCookies : ℕ := 220
def pricePerChocolateCookie : ℕ := 1
def numVanillaCookies : ℕ := 70
def pricePerVanillaCookie : ℕ := 2

-- Total money made from selling chocolate cookies
def moneyFromChocolateCookies : ℕ := numChocolateCookies * pricePerChocolateCookie

-- Total money made from selling vanilla cookies
def moneyFromVanillaCookies : ℕ := numVanillaCookies * pricePerVanillaCookie

-- Total money made from selling all cookies
def totalMoneyMade : ℕ := moneyFromChocolateCookies + moneyFromVanillaCookies

-- The statement to prove, with the expected result
theorem total_money_is_correct : totalMoneyMade = 360 := by
  sorry

end total_money_is_correct_l697_697011


namespace largest_integer_value_l697_697323

theorem largest_integer_value (x : ℤ) : 
  (1/4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 2/3 ∧ (x : ℚ) < 10 → x = 3 := 
by
  sorry

end largest_integer_value_l697_697323


namespace distance_between_inc_and_exc_circles_l697_697883

theorem distance_between_inc_and_exc_circles 
  (A B C : Type) 
  [metric_space A] [metric_space B] [metric_space C]
  (side_AB : dist A B = 8) 
  (side_AC : dist A C = 17) 
  (side_BC : dist B C = 15) : 
  ∃ (I E : Type) [metric_space I] [metric_space E], dist I E = 19 := 
by
  sorry

end distance_between_inc_and_exc_circles_l697_697883


namespace combined_flock_size_l697_697687

def original_ducks := 100
def killed_per_year := 20
def born_per_year := 30
def years_passed := 5
def another_flock := 150

theorem combined_flock_size :
  original_ducks + years_passed * (born_per_year - killed_per_year) + another_flock = 300 :=
by
  sorry

end combined_flock_size_l697_697687


namespace simplify_complex_div_l697_697104

theorem simplify_complex_div (a b c d : ℝ) (i : ℂ)
  (h1 : (a = 3) ∧ (b = 5) ∧ (c = -2) ∧ (d = 7) ∧ (i = Complex.I)) :
  ((Complex.mk a b) / (Complex.mk c d) = (Complex.mk (29/53) (-31/53))) :=
by
  sorry

end simplify_complex_div_l697_697104


namespace total_board_length_l697_697198

-- Defining the lengths of the pieces of the board
def shorter_piece_length : ℕ := 23
def longer_piece_length : ℕ := 2 * shorter_piece_length

-- Stating the theorem that the total length of the board is 69 inches
theorem total_board_length : shorter_piece_length + longer_piece_length = 69 :=
by
  -- The proof is omitted for now
  sorry

end total_board_length_l697_697198


namespace inequality_solution_l697_697111

theorem inequality_solution (x : ℝ) :
  ( (x - 1) * (x - 3) * (x - 5) / ((x - 2) * (x - 4) * (x - 6)) > 0 ) ↔
  (x ∈ set.Iio 1 ∪ set.Ioo 2 3 ∪ set.Ioo 4 5 ∪ set.Ioi 6) :=
by sorry

end inequality_solution_l697_697111


namespace sec_tan_equation_l697_697830

theorem sec_tan_equation (x : ℝ) (h : Real.sec x + Real.tan x = 3) : Real.sec x - Real.tan x = 1 / 3 :=
sorry

end sec_tan_equation_l697_697830


namespace distinct_vectors_l697_697065

-- Given conditions
variable (F : Type*) [field F] (p : ℕ) [fact (nat.prime p)]
variable (n : ℕ) (v : fin n → F) (M : matrix (fin n) (fin n) F)
variable (G : (fin n → F) → (fin n → F) := λ x, v + M.mul_vec x)

def G_iter : ℕ → (fin n → F) → (fin n → F)
| 0, x := x
| (k + 1), x := G (G_iter k x)

-- Placeholder to skip the proof
theorem distinct_vectors (p n : ℕ) (v : fin n → (zmod p)) (M : matrix (fin n) (fin n) (zmod p)) :
  (∀ k : fin (p^n), ∀ l : fin (p^n), k ≠ l → G_iter G (int.of_nat 0) k ≠ G_iter G (int.of_nat 0) l) ↔ 
  ( (p = 2 ∧ n = 2) ∨ n = 1) :=
sorry

end distinct_vectors_l697_697065


namespace total_surface_area_of_prism_l697_697653

-- Define the conditions of the problem
def sphere_radius (R : ℝ) := R > 0
def prism_circumscribed_around_sphere (R : ℝ) := True  -- Placeholder as the concept assertion, actual geometry handling not needed here
def prism_height (R : ℝ) := 2 * R

-- Define the main theorem to be proved
theorem total_surface_area_of_prism (R : ℝ) (hR : sphere_radius R) (hCircumscribed : prism_circumscribed_around_sphere R) (hHeight : prism_height R = 2 * R) : 
  ∃ (S : ℝ), S = 12 * R^2 * Real.sqrt 3 :=
sorry

end total_surface_area_of_prism_l697_697653


namespace shaded_region_perimeter_l697_697630

-- Define the conditions
def length_cm : ℝ := 20
def width_cm : ℝ := 12

-- Define the theorem for the perimeter of the shaded region
theorem shaded_region_perimeter (length_cm : ℝ) (width_cm : ℝ)
  (h_length : length_cm = 20) (h_width : width_cm = 12) : 
  2 * (length_cm + width_cm) = 64 :=
begin
  -- Proof would go here, but is omitted by using sorry
  sorry
end

end shaded_region_perimeter_l697_697630


namespace prob_equation_l697_697277

def five_inv_mod_seventeen := 7
def five_inv2_mod_seventeen := 15
def modulo := 17

theorem prob_equation :
  (5^(-1) - 5^(-2)) % modulo = 9 :=
by
  have five_inv_eq := five_inv_mod_seventeen
  have five_inv2_eq := five_inv2_mod_seventeen
  rw five_inv_eq
  rw five_inv2_eq
  sorry

end prob_equation_l697_697277


namespace max_catch_up_distance_l697_697241

/-- 
Given:
  - The total length of the race is 5000 feet.
  - Alex and Max are even for the first 200 feet, so the initial distance between them is 0 feet.
  - On the uphill slope, Alex gets ahead by 300 feet.
  - On the downhill slope, Max gains a lead of 170 feet over Alex, reducing Alex's lead.
  - On the flat section, Alex pulls ahead by 440 feet.

Prove:
  - The distance left for Max to catch up to Alex is 4430 feet.
--/
theorem max_catch_up_distance :
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  total_distance - final_distance = 4430 :=
by
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  have final_distance_calc : final_distance = 570
  sorry
  show total_distance - final_distance = 4430
  sorry

end max_catch_up_distance_l697_697241


namespace range_of_h_l697_697541

theorem range_of_h 
  (y1 y2 y3 k : ℝ)
  (h : ℝ)
  (H1 : y1 = (-3 - h)^2 + k)
  (H2 : y2 = (-1 - h)^2 + k)
  (H3 : y3 = (1 - h)^2 + k)
  (H_ord : y2 < y1 ∧ y1 < y3) : 
  -2 < h ∧ h < -1 :=
sorry

end range_of_h_l697_697541


namespace expression_evaluation_l697_697105

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = -1 ^ 2023) :
  (2 * m + n) * (2 * m - n) - (2 * m - n) ^ 2 + 2 * n * (m + n) = -12 := by
  sorry

end expression_evaluation_l697_697105


namespace maximum_rubles_received_l697_697925

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l697_697925


namespace tan_ratio_sum_l697_697490

theorem tan_ratio_sum (x y : ℝ) 
(h1 : (sin x / cos y + sin y / cos x = 1))
(h2 : (cos x / sin y + cos y / sin x = 5)) :
(tan x / tan y + tan y / tan x = 310 / 39) := 
by
  sorry

end tan_ratio_sum_l697_697490


namespace A_work_days_l697_697249

theorem A_work_days {total_wages B_share : ℝ} (B_work_days : ℝ) (total_wages_eq : total_wages = 5000) 
    (B_share_eq : B_share = 3333) (B_rate : ℝ) (correct_rate : B_rate = 1 / B_work_days) :
    ∃x : ℝ, B_share / (total_wages - B_share) = B_rate / (1 / x) ∧ total_wages - B_share = 5000 - B_share ∧ B_work_days = 10 -> x = 20 :=
by
  sorry

end A_work_days_l697_697249


namespace necessary_but_not_sufficient_condition_l697_697073

open Classical

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 3) → (a + b > 4) ∧ ¬((a + b > 4) → (a > 1 ∧ b > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l697_697073


namespace radian_of_15_degree_l697_697552

/-- The degree to radian conversion -/
def degree_to_radian (d : ℚ) : ℚ :=
d * (Real.pi / 180)

/-- Problem Statement:
  We need to prove that converting 15° to radians equals π / 12 given the conditions.
-/
theorem radian_of_15_degree :
  degree_to_radian 15 = Real.pi / 12 := 
by
  sorry

end radian_of_15_degree_l697_697552


namespace henry_money_l697_697593

-- Define the conditions
def initial : ℕ := 11
def birthday : ℕ := 18
def spent : ℕ := 10

-- Define the final amount
def final_amount : ℕ := initial + birthday - spent

-- State the theorem
theorem henry_money : final_amount = 19 := by
  -- Skipping the proof
  sorry

end henry_money_l697_697593


namespace magnitude_of_z_l697_697523

-- Define the conditions as Lean statements
variable (w z : ℂ)
variable (h1 : w * z = 20 - 15 * complex.I)
variable (h2 : complex.abs w = real.sqrt 20)

-- Formalize the problem: Prove that |z| = 5 * sqrt 5 / 2
theorem magnitude_of_z : complex.abs z = (5 * real.sqrt 5) / 2 :=
by
  -- Sorry is used to indicate proof is omitted
  sorry

end magnitude_of_z_l697_697523


namespace maximum_rubles_received_l697_697952

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l697_697952


namespace combined_flock_size_l697_697686

def original_ducks := 100
def killed_per_year := 20
def born_per_year := 30
def years_passed := 5
def another_flock := 150

theorem combined_flock_size :
  original_ducks + years_passed * (born_per_year - killed_per_year) + another_flock = 300 :=
by
  sorry

end combined_flock_size_l697_697686


namespace finite_set_primitive_points_l697_697651

-- Define primitive points and their properties
def isPrimitivePoint (x y : ℤ) : Prop := Int.gcd x y = 1

-- Main theorem to prove
theorem finite_set_primitive_points (S : Finset (ℤ × ℤ)) (hS : ∀ (p ∈ S), isPrimitivePoint p.1 p.2) : 
  ∃ (n : ℕ) (a : Fin (n + 1) → ℤ), ∀ p ∈ S, 
  let x := p.1
  let y := p.2
  (Finset.univ.sum (λ i : Fin (n + 1), a i * x^(n - i.val) * y^i.val) = 1) :=
sorry

end finite_set_primitive_points_l697_697651


namespace find_y_when_x_is_2_l697_697776

theorem find_y_when_x_is_2 :
  let y_1 := (λ x : ℝ, x^2 - 7 * x + 6)
  let y_2 := (λ x : ℝ, 7 * x - 3)
  let y := y_1 2 + 2 * y_2 2
  y = 18 :=
by
  sorry

end find_y_when_x_is_2_l697_697776


namespace right_triangles_count_l697_697009

noncomputable def count_right_triangles : ℕ :=
  let triangles := {⟨a, b⟩ : ℕ × ℕ | a^2 + b^2 = (b + 2)^2 ∧ b < 100 }
  in triangles.to_finset.card

theorem right_triangles_count :
  count_right_triangles = 10 := sorry

end right_triangles_count_l697_697009


namespace angle_CAD_equals_15_l697_697871

theorem angle_CAD_equals_15
  (A B C D E : Type)
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E]
  (AB AC BC DE EB CD : ℝ)
  (h_iso : AB = AC)
  (h_rect : BC = DE ∧ DE = EB ∧ EB = CD / 2) :
  ∠CAD = 15 :=
sorry

end angle_CAD_equals_15_l697_697871


namespace at_most_one_hired_l697_697245

-- Definition of the events and their probabilities
def P_A : ℝ := 0.5
def P_B : ℝ := 0.6

-- Assumption of independence of events A and B
def independent_events : Prop := 
  ∀ (A B : Prop), (P_A * P_B) = 0.3

-- Statement of the problem
theorem at_most_one_hired : 
  (P_A + P_B - 2 * 0.3) = 0.7 :=
by
  sorry

end at_most_one_hired_l697_697245


namespace area_of_region_is_12_l697_697557

def region_area : ℝ :=
  let f1 (x : ℝ) : ℝ := |x - 2|
  let f2 (x : ℝ) : ℝ := 5 - |x + 1|
  let valid_region (x y : ℝ) : Prop := f1 x ≤ y ∧ y ≤ f2 x
  12

theorem area_of_region_is_12 :
  ∃ (area : ℝ), region_area = 12 := by
  use 12
  sorry

end area_of_region_is_12_l697_697557


namespace tteokbokki_cost_l697_697471

theorem tteokbokki_cost (P : ℝ) (h1 : P / 2 - P * (3 / 16) = 2500) : P / 2 = 4000 :=
by
  sorry

end tteokbokki_cost_l697_697471


namespace inequality_relation_l697_697353

noncomputable def a : ℝ := Real.log (1 / 3)
noncomputable def b : ℝ := 3 ^ 0.1
noncomputable def c : ℝ := Real.sin 3

theorem inequality_relation : b > c ∧ c > a := by
  sorry

end inequality_relation_l697_697353


namespace april_earnings_l697_697193

theorem april_earnings (rose_price : ℕ) (initial_roses : ℕ) (remaining_roses : ℕ) :
  rose_price = 4 → initial_roses = 13 → remaining_roses = 4 → (initial_roses - remaining_roses) * rose_price = 36 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3]
  norm_num
  sorry

end april_earnings_l697_697193


namespace smallest_possible_x2_plus_y2_l697_697419

theorem smallest_possible_x2_plus_y2 (x y : ℝ) (h : (x + 3) * (y - 3) = 0) : x^2 + y^2 = 18 :=
sorry

end smallest_possible_x2_plus_y2_l697_697419


namespace arcsin_sin_solutions_l697_697521

theorem arcsin_sin_solutions (x : ℝ) (h : -π / 2 ≤ x ∧ x ≤ π / 2) :
  (arcsin (sin (2 * x)) = x ↔ x = 0 ∨ x = π / 3 ∨ x = -π / 3) :=
by
  sorry

end arcsin_sin_solutions_l697_697521


namespace smallest_power_of_13_non_palindrome_l697_697714

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l697_697714


namespace total_money_l697_697643

variable (A B C: ℕ)
variable (h1: A + C = 200) 
variable (h2: B + C = 350)
variable (h3: C = 200)

theorem total_money : A + B + C = 350 :=
by
  sorry

end total_money_l697_697643


namespace correct_statement_l697_697406

theorem correct_statement (a b : ℚ) :
  (|a| = b → a = b) ∧ (|a| > |b| → a > b) ∧ (|a| > b → |a| > |b|) ∧ (|a| = b → a^2 = (-b)^2) ↔ 
  (true ∧ false ∧ false ∧ true) :=
by
  sorry

end correct_statement_l697_697406


namespace triangle_obtuse_PE_eq_PF_l697_697475

variables {A B C D E F P : Type*}

-- Assuming there exist obtuse triangle ABC with AB = AC
variables [triangle ABC] (h_eq_ab_ac : AB = AC)
-- Assuming the existence of a circle Gamma tangent to AB at B and to AC at C
variable (Γ : Circle)
-- Assuming D is the furthest point on Gamma from A and AD ⊥ BC
variables [on_circle D Γ] [is_furthest_from D A Γ] [perpendicular AD BC]
-- Assuming E is the intersection of AB and DC
variables [on_line E AB] [on_line E DC]
-- Assuming F lies on AB such that BC = BF and B lies on segment AF
variables [on_line F AB] (h_eq_bc_bf : BC = BF) [between B (seg AF)]
-- Assuming P is the intersection of AC and DB
variables [on_line P AC] [on_line P DB]

-- The goal is to show PE = PF
theorem triangle_obtuse_PE_eq_PF 
    (h_eq_ab_ac : AB = AC) 
    (Γ : Circle) 
    [on_circle D Γ] 
    [is_furthest_from D A Γ] 
    [perpendicular AD BC] 
    [on_line E AB] 
    [on_line E DC] 
    (h_eq_bc_bf : BC = BF) 
    [on_line F AB] 
    [between B (seg AF)] 
    [on_line P AC] 
    [on_line P DB] : PE = PF := 
sorry

end triangle_obtuse_PE_eq_PF_l697_697475


namespace largest_sum_of_watch_digits_l697_697605

theorem largest_sum_of_watch_digits : ∃ s : ℕ, s = 23 ∧ 
  (∀ h m : ℕ, h < 24 → m < 60 → s ≤ (h / 10 + h % 10 + m / 10 + m % 10)) :=
by
  sorry

end largest_sum_of_watch_digits_l697_697605


namespace sum_of_six_terms_l697_697364

theorem sum_of_six_terms (a1 : ℝ) (S4 : ℝ) (d : ℝ) (a1_eq : a1 = 1 / 2) (S4_eq : S4 = 20) :
  S4 = (4 * a1 + (4 * (4 - 1) / 2) * d) → (S4 = 20) →
  (6 * a1 + (6 * (6 - 1) / 2) * d = 48) :=
by
  intros
  sorry

end sum_of_six_terms_l697_697364


namespace three_digit_identical_divisible_by_37_l697_697096

theorem three_digit_identical_divisible_by_37 (A : ℕ) (h : A ≤ 9) : 37 ∣ (111 * A) :=
sorry

end three_digit_identical_divisible_by_37_l697_697096


namespace remaining_dimes_l697_697101

-- Conditions
def initial_pennies : Nat := 7
def initial_dimes : Nat := 8
def borrowed_dimes : Nat := 4

-- Define the theorem
theorem remaining_dimes : initial_dimes - borrowed_dimes = 4 := by
  -- Use the conditions to state the remaining dimes
  sorry

end remaining_dimes_l697_697101


namespace us_to_cny_exchange_rate_l697_697281

-- Definitions given in the conditions
def appreciation_rate : ℝ := 0.02
def initial_exchange_rate : ℝ := 6.81
def calculation_period : ℝ := 60 / 30 -- Number of 30-day periods in 60 days

-- The goal is to prove the final exchange rate after 60 days.
theorem us_to_cny_exchange_rate (H : appreciation_rate = 0.02 ∧ initial_exchange_rate = 6.81 ∧ calculation_period = 2) :
  initial_exchange_rate * (1 - appreciation_rate) ^ 2 = 6.53 :=
by
  sorry

end us_to_cny_exchange_rate_l697_697281


namespace return_trip_time_l697_697629

theorem return_trip_time (d p w : ℝ) (h1 : d = 84 * (p - w)) (h2 : d / (p + w) = d / p - 9) :
  (d / (p + w) = 63) ∨ (d / (p + w) = 12) :=
by
  sorry

end return_trip_time_l697_697629


namespace circle_C1_equation_circle_C2_equation_length_of_chord_AB_l697_697863

-- Define circle C1 in parametric form and convert to Cartesian coordinates
def circle_C1_param (α : ℝ) : ℝ × ℝ := (2 * cos α, 2 + 2 * sin α)

-- Define circle C1 in standard Cartesian form
def circle_C1_cartesian (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define circle C2 in polar coordinates and convert to Cartesian coordinates
def circle_C2_polar (θ : ℝ) : ℝ := 2 * sqrt 2 * cos (θ + π / 4)

-- Define circle C2 in Cartesian coordinates
def circle_C2_cartesian (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Prove the standard form equation of circle C1
theorem circle_C1_equation : ∀ (α : ℝ), 
  let (x, y) := circle_C1_param α in
  circle_C1_cartesian x y :=
by sorry

-- Prove the Cartesian coordinate equation of circle C2
theorem circle_C2_equation : 
  ∀ (θ : ℝ), 
  let p := circle_C2_polar θ in 
  ∃ x y : ℝ, p = sqrt (x^2 + y^2) ∧ atan (y / x) = θ ∧ circle_C2_cartesian x y :=
by sorry

-- Prove that the length of the chord AB is 4 if circles intersect at points A and B
theorem length_of_chord_AB : 
  (∃ A B : ℝ × ℝ, 
    circle_C1_cartesian A.1 A.2 ∧ 
    circle_C2_cartesian A.1 A.2 ∧ 
    circle_C1_cartesian B.1 B.2 ∧ 
    circle_C2_cartesian B.1 B.2 ∧ 
    A ≠ B) → 
  let chord_length := dist (A.1, A.2) (B.1, B.2) in 
  chord_length = 4 :=
by sorry

end circle_C1_equation_circle_C2_equation_length_of_chord_AB_l697_697863


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697751

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l697_697751


namespace total_tshirt_cost_l697_697062

theorem total_tshirt_cost (white_small_men: ℕ) (white_medium_men: ℕ) (white_large_men: ℕ)
                          (black_small_men: ℕ) (black_medium_men: ℕ) (black_large_men: ℕ)
                          (white_small_women: ℕ) (white_medium_women: ℕ) (white_large_women: ℕ)
                          (black_small_women: ℕ) (black_medium_women: ℕ) (black_large_women: ℕ)
                          (num_employees: ℕ) (small_percentage: ℚ) (medium_percentage: ℚ) (large_percentage: ℚ)
                          (num_men_women: ℕ) (small_men_women: ℕ) (medium_men_women: ℕ) (large_men_women: ℕ):
  num_employees = 40 →
  small_percentage = 0.5 →
  medium_percentage = 0.3 →
  large_percentage = 0.2 →
  num_men_women = num_employees / 2 →
  small_men_women = (num_employees * small_percentage).toNat / 2 →
  medium_men_women = (num_employees * medium_percentage).toNat / 2 →
  large_men_women = (num_employees * large_percentage).toNat / 2 →
  white_small_men = 20 →
  white_medium_men = 24 →
  white_large_men = 28 →
  black_small_men = 18 →
  black_medium_men = 22 →
  black_large_men = 26 →
  white_small_women = white_small_men - 5 →
  white_medium_women = white_medium_men - 5 →
  white_large_women = white_large_men - 5 →
  black_small_women = black_small_men - 5 →
  black_medium_women = black_medium_men - 5 →
  black_large_women = black_large_men - 5 →
  let cost_white_men := (small_men_women * white_small_men + medium_men_women * white_medium_men + large_men_women * white_large_men) in
  let cost_white_women := (small_men_women * white_small_women + medium_men_women * white_medium_women + large_men_women * white_large_women) in
  let cost_black_men := (small_men_women * black_small_men + medium_men_women * black_medium_men + large_men_women * black_large_men) in
  let cost_black_women := (small_men_women * black_small_women + medium_men_women * black_medium_women + large_men_women * black_large_women) in
  cost_white_men + cost_white_women + cost_black_men + cost_black_women = 1544 := by
  -- Proof omitted
  sorry

end total_tshirt_cost_l697_697062


namespace unique_players_count_l697_697910

-- Define the parameters and conditions
def num_groups := 25
def players_per_group := 4
def groups_with_two_repeats := 8
def groups_with_one_repeat := 5

-- Calculate total player spots and repeated player spots
def total_player_spots := num_groups * players_per_group
def repeated_spots_with_two := groups_with_two_repeats * 2
def repeated_spots_with_one := groups_with_one_repeat * 1
def total_repeated_spots := repeated_spots_with_two + repeated_spots_with_one
def unique_player_spots := total_player_spots - total_repeated_spots

-- The theorem to prove
theorem unique_players_count : unique_player_spots = 79 := by
  unfold unique_player_spots
  unfold total_player_spots
  unfold total_repeated_spots
  unfold repeated_spots_with_two
  unfold repeated_spots_with_one
  unfold num_groups
  unfold players_per_group
  unfold groups_with_two_repeats
  unfold groups_with_one_repeat
  simp
  sorry

end unique_players_count_l697_697910


namespace probability_red_purple_different_beds_l697_697667

-- Definitions based on the given conditions
def flower_colors := {'red, 'yellow, 'white, 'purple}

def combinations (n k : ℕ) : ℕ := n.choose k

-- Lean 4 statement as per the proof problem
theorem probability_red_purple_different_beds :
  let total_ways := combinations 4 2 in
  let red_purple_together := 2 in
  let red_purple_different := total_ways - red_purple_together in
  (red_purple_different : ℚ) / total_ways = 2 / 3 :=
by
  sorry

end probability_red_purple_different_beds_l697_697667


namespace Petya_rubles_maximum_l697_697953

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l697_697953


namespace handrail_length_l697_697633

theorem handrail_length :
  let full_rotation := 360
      additional_degrees := 30
      height := 12.0
      radius := 4.0
      total_degrees := full_rotation + additional_degrees
      circumference := 2 * Real.pi * radius
      arc_length := (total_degrees / full_rotation) * circumference
      handrail_length := Real.sqrt (height^2 + (arc_length^2)) in
      (handrail_length ≈ 107.9) :=
by
  sorry

end handrail_length_l697_697633


namespace sqrt_product_eq_l697_697673

theorem sqrt_product_eq :
  (Int.sqrt (2 ^ 2 * 3 ^ 4) : ℤ) = 18 :=
sorry

end sqrt_product_eq_l697_697673


namespace combined_weight_of_parcels_l697_697911

variable (x y z : ℕ)

theorem combined_weight_of_parcels : 
  (x + y = 132) ∧ (y + z = 135) ∧ (z + x = 140) → x + y + z = 204 :=
by 
  intros
  sorry

end combined_weight_of_parcels_l697_697911


namespace find_function_p_t_additional_hours_l697_697606

variable (p0 : ℝ) (t k : ℝ)

-- Given condition: initial concentration decreased by 1/5 after one hour
axiom filtration_condition_1 : (p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)))
axiom filtration_condition_2 : (p0 * ((4 : ℝ) / 5) = p0 * (Real.exp (-k)))

-- Problem 1: Find the function p(t)
theorem find_function_p_t : ∃ k, ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)) := by
  sorry

-- Problem 2: Find the additional hours of filtration needed
theorem additional_hours (h : ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t))) :
  ∀ t, p0 * ((4 : ℝ) / 5) ^ t ≤ (p0 / 1000) → t ≥ 30 := by
  sorry

end find_function_p_t_additional_hours_l697_697606


namespace distance_between_A_and_B_l697_697443

-- Definition of points A and B in Cartesian coordinate system
def pointA : ℝ × ℝ × ℝ := (1, 1, 2)
def pointB : ℝ × ℝ × ℝ := (2, 3, 4)

-- Defining the distance formula in three-dimensional space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Statement we need to prove
theorem distance_between_A_and_B : distance pointA pointB = 3 := by
  sorry

end distance_between_A_and_B_l697_697443


namespace parabola_coeff_sum_l697_697289

theorem parabola_coeff_sum (p q r : ℝ) :
  (∀ x : ℝ, y = p * x ^ 2 + q * x + r ∧ ((0 : ℝ), -2) ∧ vertex (-3 : ℝ) (4) -  p + q + r = -20 / 3= Rational)atic := 
  sorry

end parabola_coeff_sum_l697_697289


namespace quadractic_transformation_sum_l697_697414

theorem quadractic_transformation_sum :
  let a := 5
  let h := 2
  let k := -12
  a + h + k = -5 := 
by
  sorry

end quadractic_transformation_sum_l697_697414


namespace circumcircle_intersect_ext_angle_bisector_l697_697510

theorem circumcircle_intersect_ext_angle_bisector (A B C D : ℝ) 
  (h1 : AC > AB) 
  (h2 : ∃ K : Triangle, circumcircle K = Circle center radius) 
  (h3 : ∃ point_D : Point, is_ext_angle_bisector ∠A A D) 
  : AC - AB = 2 * AD * sin(A / 2) :=
sorry

end circumcircle_intersect_ext_angle_bisector_l697_697510


namespace length_of_grassy_area_excluding_path_l697_697209

-- Definitions based on conditions
def length_grassy_plot : ℕ := 110
def width_grassy_plot : ℕ := 65
def width_gravel_path : ℕ := 2.5

-- Theorem statement to prove the length of the grassy plot excluding the gravel path is 105 m
theorem length_of_grassy_area_excluding_path : 
  length_grassy_plot - 2 * width_gravel_path = 105 :=
by
  sorry

end length_of_grassy_area_excluding_path_l697_697209


namespace intersection_point_unique_intersection_l697_697063

noncomputable def g (x : ℝ) : ℝ := x^3 + 9 * x^2 + 18 * x + 38

theorem intersection_point_unique_intersection :
  ∃ c d : ℝ, (g c = c) ∧ (c = d) :=
begin
  use [-2, -2],
  split,
  {
    -- c = g(c)
    have h1 : g (-2) = (-2)^3 + 9 * (-2)^2 + 18 * (-2) + 38 := rfl,
    norm_num at h1,
    exact h1,
  },
  {
    -- c = d
    refl,
  }
end

end intersection_point_unique_intersection_l697_697063


namespace parallelogram_area_correct_l697_697254

noncomputable def parallelogram_area (a b : ℝ^3) : ℝ := (a × b).norm

-- Definitions
variables (p q : ℝ^3)
variables (a : ℝ^3 := 2 • p - q) (b : ℝ^3 := p + 3 • q)
variables (norm_p : ℝ := 3) (norm_q : ℝ := 2)
variables (angle_pq : ℝ := Real.pi / 2)

-- Conditions
axiom p_norm : ∥p∥ = norm_p
axiom q_norm : ∥q∥ = norm_q
axiom p_q_angle : Real.angle p q = angle_pq

theorem parallelogram_area_correct : parallelogram_area a b = 42 := 
by
  sorry

end parallelogram_area_correct_l697_697254


namespace max_integers_blackboard_l697_697149

def is_prime_power (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p^k

def valid_integers (S : Set ℕ) :=
  ∀ x ∈ S, ∀ y ∈ S, x ≠ y → is_prime_power (x + y)

theorem max_integers_blackboard : ∀ S : Set ℕ, valid_integers S → S.card ≤ 4 := by
  sorry

end max_integers_blackboard_l697_697149


namespace max_possible_a_l697_697392

theorem max_possible_a (a b : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, (λ f, f y = f x + y) (λ x, x ^ 2 + a * x + b)) →
  a ≤ 1 / 2 :=
sorry

end max_possible_a_l697_697392


namespace pasture_feeds_two_sheep_in_two_days_l697_697858

theorem pasture_feeds_two_sheep_in_two_days :
  let grass_consumed := (sheep : ℕ) → (days : ℕ) → ℕ := λ sheep days, sheep * days
  let daily_growth := ((grass_consumed 10 20) - (grass_consumed 14 12)) / (20 - 12)
  let grass_needed_per_sheep_per_day := 1
  let total_grass_needed := grass_needed_per_sheep_per_day * 2
  daily_growth = 2 * total_grass_needed := 
by
  let grass_consumed := (sheep : ℕ) → (days : ℕ) → ℕ := λ sheep days, sheep * days
  let total_grass_10_sheep_20_days := grass_consumed 10 20
  let total_grass_14_sheep_12_days := grass_consumed 14 12
  let daily_growth := (total_grass_10_sheep_20_days - total_grass_14_sheep_12_days) / (20 - 12)
  let grass_needed_per_sheep_per_day := 1
  let total_grass_needed := grass_needed_per_sheep_per_day * 2
  have h : (daily_growth = 4) := by
    rw [grass_consumed, grass_consumed]
    have h_grass_10 := total_grass_10_sheep_20_days = 10 * 20 := by rfl
    have h_grass_14 := total_grass_14_sheep_12_days = 14 * 12 := by rfl
    rw [h_grass_10, h_grass_14]
    have h_diff := (200 - 168) = 32 := by rfl
    have h8 := (20 - 12) = 8 := by rfl
    rw [h_diff, h8]
    exact Eq.refl 4
  have h_needed := total_grass_needed = 2 := by rfl
  rw [h]
  rw [h_needed]
  exact Eq.refl 4

end pasture_feeds_two_sheep_in_two_days_l697_697858


namespace has_one_real_root_l697_697975

noncomputable def f (x : ℝ) : ℝ := x + real.sqrt (x - 4)

theorem has_one_real_root : (∀ x : ℝ, f x = 6 → x = 5) :=
begin
  sorry
end

end has_one_real_root_l697_697975


namespace function_properties_l697_697808

def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem function_properties : 
  (∀ x, f (-x) = f x) ∧ (∀ x, 0 ≤ x → f (x + 1) ≥ f x) :=
by
-- Proof that f is even
sorry 
-- Proof that f is increasing for x ≥ 0
sorry

end function_properties_l697_697808


namespace cylindrical_coords_of_point_l697_697679

theorem cylindrical_coords_of_point :
  ∃ (r θ z : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
                 r = Real.sqrt (3^2 + 3^2) ∧
                 θ = Real.arctan (3 / 3) ∧
                 z = 4 ∧
                 (3, 3, 4) = (r * Real.cos θ, r * Real.sin θ, z) :=
by
  sorry

end cylindrical_coords_of_point_l697_697679


namespace min_weights_for_1_to_100_eq_seven_l697_697166

theorem min_weights_for_1_to_100_eq_seven : 
  ∃ (w : Finset ℕ), 
    (∀ (x : ℕ), x ∈ w ↔ x ∈ {2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6}) 
    ∧ (∀ m, 1 ≤ m ∧ m ≤ 100 → ∃ (s : Finset ℕ), s ⊆ w ∧ s.sum = m) 
    ∧ w.card = 7 := 
sorry

end min_weights_for_1_to_100_eq_seven_l697_697166


namespace calc_root_diff_l697_697253

theorem calc_root_diff : 81^(1/4) - 16^(1/2) = -1 := by
  sorry

end calc_root_diff_l697_697253


namespace final_shape_independent_of_order_l697_697566

inductive Figure
| Circle : Figure
| Square : Figure
| Triangle : Figure

open Figure

def circ : Figure → Figure → Figure
| Circle, Circle => Circle
| Circle, Square => Square
| Square, Circle => Square
| Square, Square => Triangle
| Circle, Triangle => Triangle
| Triangle, Circle => Triangle
| Triangle, Triangle => Square
| Triangle, Square => Circle
| Square, Triangle => Circle

theorem final_shape_independent_of_order (figures : List Figure) :
  ∃! final_figure : Figure, ∀ order : List (Figure × Figure), 
    last_figure (erase_figures order figures) = final_figure :=
sorry

end final_shape_independent_of_order_l697_697566


namespace felix_brother_weight_ratio_l697_697287

theorem felix_brother_weight_ratio (F B : ℕ) (hF1 : 1.5 * F = 150) (hB1 : 3 * B = 600) : (B / F) = 2 :=
by
  sorry

end felix_brother_weight_ratio_l697_697287


namespace max_rectangle_area_in_circle_l697_697189

theorem max_rectangle_area_in_circle (r : ℝ) (h : r = 5) : 
  let s := 5 * real.sqrt 2 in
  s * s = 50 :=
by
  sorry

end max_rectangle_area_in_circle_l697_697189


namespace difference_digits_in_base2_l697_697173

def binaryDigitCount (n : Nat) : Nat := Nat.log2 n + 1

theorem difference_digits_in_base2 : binaryDigitCount 1400 - binaryDigitCount 300 = 2 :=
by
  sorry

end difference_digits_in_base2_l697_697173


namespace face_areas_of_perpendicular_planes_l697_697032

theorem face_areas_of_perpendicular_planes
  {P A B C : Type} [plane PAB] [plane PBC] [plane PCA]
  (h1 : is_perpendicular PAB PBC)
  (h2 : is_perpendicular PBC PCA)
  (h3 : is_perpendicular PCA PAB):
  (S_triangle ABC)^2 = (S_triangle PAB)^2 + (S_triangle PBC)^2 + (S_triangle PCA)^2 :=
sorry

end face_areas_of_perpendicular_planes_l697_697032


namespace smallest_non_palindrome_power_of_13_l697_697720

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l697_697720


namespace trigonometric_inequality_l697_697349

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 4) :
  sin (sin x) < sin x ∧ sin x < sin (tan x) :=
by
  sorry

end trigonometric_inequality_l697_697349


namespace least_positive_integer_div_conditions_l697_697582

theorem least_positive_integer_div_conditions :
  ∃ n > 1, (n % 4 = 3) ∧ (n % 5 = 3) ∧ (n % 7 = 3) ∧ (n % 10 = 3) ∧ (n % 11 = 3) ∧ n = 1543 := 
by 
  sorry

end least_positive_integer_div_conditions_l697_697582


namespace Pablo_puzzle_completion_l697_697507

theorem Pablo_puzzle_completion :
  let pieces_per_hour := 100
  let puzzles_400 := 15
  let pieces_per_puzzle_400 := 400
  let puzzles_700 := 10
  let pieces_per_puzzle_700 := 700
  let daily_work_hours := 6
  let daily_work_400_hours := 4
  let daily_work_700_hours := 2
  let break_every_hours := 2
  let break_time := 30 / 60   -- 30 minutes break in hours

  let total_pieces_400 := puzzles_400 * pieces_per_puzzle_400
  let total_pieces_700 := puzzles_700 * pieces_per_puzzle_700
  let total_pieces := total_pieces_400 + total_pieces_700

  let effective_daily_hours := daily_work_hours - (daily_work_hours / break_every_hours * break_time)
  let pieces_400_per_day := daily_work_400_hours * pieces_per_hour
  let pieces_700_per_day := (effective_daily_hours - daily_work_400_hours) * pieces_per_hour
  let total_pieces_per_day := pieces_400_per_day + pieces_700_per_day
  
  total_pieces / total_pieces_per_day = 26 := by
sorry

end Pablo_puzzle_completion_l697_697507


namespace log_inequality_solution_set_l697_697761

theorem log_inequality_solution_set :
  {x : ℝ | log 3 (2 * x - 1) ≤ 1} = {x : ℝ | 1 / 2 < x ∧ x ≤ 2} :=
by
  sorry

end log_inequality_solution_set_l697_697761


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697734

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l697_697734


namespace diana_age_l697_697671

open Classical

theorem diana_age :
  ∃ (D : ℚ), (∃ (C E : ℚ), C = 4 * D ∧ E = D + 5 ∧ C = E) ∧ D = 5/3 :=
by
  -- Definitions and conditions are encapsulated in the existential quantifiers and the proof concludes with D = 5/3.
  sorry

end diana_age_l697_697671


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697749

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l697_697749


namespace index_card_area_l697_697001

theorem index_card_area :
  ∀ (length width : ℕ), length = 5 → width = 7 →
  (length - 2) * width = 21 →
  length * (width - 1) = 30 :=
by
  intros length width h_length h_width h_condition
  sorry

end index_card_area_l697_697001


namespace probability_f_geq_1_l697_697809

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - x - 1

theorem probability_f_geq_1 : 
  ∀ x ∈ Icc (-1 : ℝ) 2, 
  let len_total := (2 - (-1)) in -- total length is 3
  let len_sub_interval := (1 - (-2 / 3)) in -- interval length for f(x) >= 1
  let probability := len_sub_interval / len_total in
  probability = 5 / 9 :=
begin
  sorry
end

end probability_f_geq_1_l697_697809


namespace difference_digits_in_base2_l697_697174

def binaryDigitCount (n : Nat) : Nat := Nat.log2 n + 1

theorem difference_digits_in_base2 : binaryDigitCount 1400 - binaryDigitCount 300 = 2 :=
by
  sorry

end difference_digits_in_base2_l697_697174


namespace cos_A_minus_B_l697_697370

theorem cos_A_minus_B (A B : ℝ) :
  (sin A + sin B = 1/2) →
  (cos A + cos B = 1) →
  cos (A - B) = -3/8 := by
sorry

end cos_A_minus_B_l697_697370


namespace remainder_product_units_6_factorial_l697_697484

theorem remainder_product_units_6_factorial :
  let n := 6!
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x n) (Finset.range n), i
  m % n = 1 :=
by
  let n := 6!
  let m := ∏ i in Finset.filter (λ x, Nat.coprime x n) (Finset.range n), i
  sorry

end remainder_product_units_6_factorial_l697_697484


namespace smallest_possible_value_u3_v3_l697_697978

theorem smallest_possible_value_u3_v3 
  (u v : ℂ) 
  (h1 : complex.abs (u + v) = 2)
  (h2 : complex.abs (u^2 + v^2) = 17) :
  ∃ w, complex.abs (u^3 + v^3) = w ∧ w ≥ 47 :=
by
  sorry

end smallest_possible_value_u3_v3_l697_697978


namespace solve_complex_equation_l697_697334

-- Definitions for real numbers x, y and complex number z.
variables (x y : ℝ) (z : ℂ)

-- The condition where z is a complex number defined by real parts x and y.
def z_def : ℂ := x + y * complex.I

-- The main statement translating the problem into Lean
theorem solve_complex_equation : 
  (∃ (x y : ℝ) (z : ℂ), z = x + y * complex.I ∧ z^6 = -8 ∧
   (z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 5 * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 7 * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 11 * real.pi / 6) ∨ 
    z = complex.I * (2 : ℝ)^(1 / 3) ∨ 
    z = -complex.I * (2 : ℝ)^(1 / 3))) := 
sorry

end solve_complex_equation_l697_697334


namespace min_sum_of_ten_numbers_l697_697119

theorem min_sum_of_ten_numbers
  (S : Finset ℕ)
  (h_distinct : S.card = 10)
  (h_prod_even : ∀ T ⊆ S, T.card = 5 → ∃ x ∈ T, x % 2 = 0)
  (h_sum_odd : S.sum % 2 = 1) :
  S.sum = 65 :=
sorry

end min_sum_of_ten_numbers_l697_697119


namespace adam_final_score_l697_697433

-- Definitions derived from the conditions
def correct_first_half : ℕ := 15
def correct_second_half : ℕ := 12
def points_first_half : ℕ := 3
def points_second_half : ℕ := 5
def bonus_threshold : ℕ := 3
def bonus_points : ℕ := 2
def penalty_per_wrong : ℤ := -1
def total_questions : ℕ := 35

--- Main theorem statement
theorem adam_final_score :
  let correct_answers := correct_first_half + correct_second_half in
  let total_correct_points := (correct_first_half * points_first_half) + (correct_second_half * points_second_half) in
  let bonus := (correct_answers / bonus_threshold) * bonus_points in
  let incorrect_answers := total_questions - correct_answers in
  let penalty := incorrect_answers * penalty_per_wrong in
  (total_correct_points + bonus + penalty) = 115 :=
by
  let correct_answers := correct_first_half + correct_second_half
  let total_correct_points := (correct_first_half * points_first_half) + (correct_second_half * points_second_half)
  let bonus := (correct_answers / bonus_threshold) * bonus_points
  let incorrect_answers := total_questions - correct_answers
  let penalty := incorrect_answers * penalty_per_wrong
  show (total_correct_points + bonus + penalty) = 115
  sorry

end adam_final_score_l697_697433


namespace lark_locker_combinations_l697_697474

/-- Calculate the number of combinations for Lark's locker given the conditions:
    1. First number is an odd number in range [1, 20]
    2. Second number is an even number in range [1, 40]
    3. Third number is a multiple of 3 in range [1, 30]
    4. Fourth number is a multiple of 5 in range [1, 50]
-/
theorem lark_locker_combinations :
  let count_odd := 10,
      count_even := 20,
      count_multiple_3 := 10,
      count_multiple_5 := 10
  in count_odd * count_even * count_multiple_3 * count_multiple_5 = 20000 :=
by
  let count_odd := 10
  let count_even := 20
  let count_multiple_3 := 10
  let count_multiple_5 := 10
  calc
    count_odd * count_even * count_multiple_3 * count_multiple_5
        = 10 * 20 * 10 * 10 : by rfl
    ... = 20000 : by norm_num
-- sorry

end lark_locker_combinations_l697_697474


namespace white_marbles_count_l697_697197

def number_of_white_marbles (total_marbles blue_marbles red_marbles : Nat) (prob_red_or_white : ℚ) : Nat := total_marbles - blue_marbles - red_marbles

theorem white_marbles_count (total_marbles blue_marbles red_marbles : Nat) (prob_red_or_white : ℚ)
  (h_total : total_marbles = 50)
  (h_blue : blue_marbles = 5)
  (h_red : red_marbles = 9)
  (h_prob : prob_red_or_white = 0.9) :
  number_of_white_marbles total_marbles blue_marbles red_marbles prob_red_or_white = 36 :=
by
  rw [h_total, h_blue, h_red]
  have h_W := (number_of_white_marbles 50 5 9 prob_red_or_white)
  calc
    h_W : 36 = 50 - 5 - 9 := by
      calc
        50 - 5 - 9 = 36 := by norm_num
  sorry

end white_marbles_count_l697_697197


namespace min_10_T_value_is_23_l697_697882

noncomputable def min_10_T_value : ℝ :=
  let T (x y z : ℝ) := (1 / 4 * x^2 - 1 / 5 * y^2 + 1 / 6 * z^2)
  in  Sup {10 * T x y z | x y z : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 4 ∧ 1 ≤ z ∧ z ≤ 4 ∧ x - y + z = 4}

theorem min_10_T_value_is_23 :
  min_10_T_value = 23 := sorry

end min_10_T_value_is_23_l697_697882


namespace number_of_ways_to_sum_210_as_consecutive_integers_l697_697440

theorem number_of_ways_to_sum_210_as_consecutive_integers : ∃! k : ℕ, k ≥ 2 ∧ ∃ n : ℕ, 210 = k * n + k*(k - 1) / 2 ∧ k is_factor_of 210 :=
by sorry

end number_of_ways_to_sum_210_as_consecutive_integers_l697_697440


namespace number_of_poles_needed_l697_697237

theorem number_of_poles_needed :
  let parallel_side_1 := 60
  let parallel_side_2 := 80
  let non_parallel_side := 50
  let interval_parallel := 5
  let interval_non_parallel := 7
  -- Number of poles on the first parallel side
  let poles_parallel_1 := parallel_side_1 / interval_parallel + 1
  -- Number of poles on the second parallel side
  let poles_parallel_2 := parallel_side_2 / interval_parallel + 1
  -- Number of poles on one non-parallel side (ceiling function rounded)
  let poles_non_parallel := (non_parallel_side / interval_non_parallel).ceil + 1
  -- Total number of poles considering all sides
  let total_poles := poles_parallel_1 + poles_parallel_2 + 2 * poles_non_parallel
  -- Adjust for double-counted corner poles
  let poles_needed := total_poles - 4
  poles_needed = 44 :=
begin
  sorry
end

end number_of_poles_needed_l697_697237


namespace range_of_c_in_acute_triangle_l697_697435

variables {A B C a b c : ℝ}
def is_acute_triangle (A B C : ℝ) := 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

theorem range_of_c_in_acute_triangle (h1 : is_acute_triangle A B C)
  (h2 : sqrt 3 * (a * cos B + 1 * cos A) = 2 * c * sin C)
  (h3 : b = 1) :
  c ∈ Ioo (sqrt 3 / 2) sqrt 3 :=
by
  sorry

end range_of_c_in_acute_triangle_l697_697435


namespace height_of_cylinder_in_hemisphere_l697_697214

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (h_base_parallel : Prop) : ℝ :=
  if h_base_parallel then sqrt (r_hemisphere^2 - r_cylinder^2) else 0

theorem height_of_cylinder_in_hemisphere :
  cylinder_height 3 8 true = sqrt 55 :=
by
  unfold cylinder_height
  simp
  sorry

end height_of_cylinder_in_hemisphere_l697_697214


namespace solve_equation_l697_697179

def cos_not_zero (n : ℤ) (x : ℝ) : Prop := cos (2 * x) ≠ 0 ∧ cos (5 * x) ≠ 0

def equation (x : ℝ) : Prop := 1 + tan (2 * x) * tan (5 * x) - sqrt 2 * tan (2 * x) * cos (3 * x) * acos (5 * x) = 0

theorem solve_equation :
  (∀ n : ℤ, cos_not_zero n ((π / 6) * (6 * n + 1)) ∨ cos_not_zero n ((π / 6) * (6 * n - 1)) → equation ((π / 6) * (6 * n + 1)) ∨ equation ((π / 6) * (6 * n - 1)))
  ∧ 
  (∀ k : ℤ, cos_not_zero k ((-1 : ℝ) ^ k * (π / 8) + (π * k) / 2) → equation ((-1 : ℝ) ^ k * (π / 8) + (π * k) / 2)) :=
by sorry

end solve_equation_l697_697179


namespace smallest_non_palindrome_power_of_13_l697_697755

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l697_697755


namespace graphs_intersect_once_l697_697263

theorem graphs_intersect_once : 
  ∃! (x : ℝ), |3 * x + 6| = -|4 * x - 3| :=
sorry

end graphs_intersect_once_l697_697263


namespace ratio_of_length_to_width_l697_697998

theorem ratio_of_length_to_width (W : ℕ) (P : ℕ) (hW : W = 60) (hP : P = 288) :
  let L := (P - 2 * W) / 2 in L / W = 7 / 5 := by
  sorry

end ratio_of_length_to_width_l697_697998


namespace locus_of_midpoint_l697_697570

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

-- Definition stating that the two lines are perpendicular at the origin
def is_perpendicular (A B : ℝ × ℝ) : Prop :=
  let ⟨a1, b1⟩ := A in
  let ⟨a2, b2⟩ := B in
  a1 * a2 + b1 * b2 = 0

-- Definition of the midpoint of two points
def midpoint (A B P : ℝ × ℝ) : Prop :=
  let ⟨a1, b1⟩ := A in
  let ⟨a2, b2⟩ := B in
  let ⟨x, y⟩ := P in
  (2 * x = a1 + a2) ∧ (2 * y = b1 + b2)

-- The main theorem that we need to prove
theorem locus_of_midpoint {P : ℝ × ℝ} :
  (∃ A B : ℝ × ℝ, hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧
    is_perpendicular A B ∧ midpoint A B P) →
  3 * (4 * P.1^2 - P.2^2)^2 = 4 * (16 * P.1^2 + P.2) := 
sorry

end locus_of_midpoint_l697_697570


namespace sin_of_angle_passing_through_point_is_neg_sqrt_three_over_two_l697_697413

theorem sin_of_angle_passing_through_point_is_neg_sqrt_three_over_two :
  (let α : ℝ := α in
   let P := (1 : ℝ, -real.sqrt 3) in
   let x := P.1 in
   let y := P.2 in
   let r := real.sqrt (x^2 + y^2) in
   sin α = y / r) :=
begin
  let α : ℝ := α,
  let P : ℝ × ℝ := (1, -real.sqrt 3),
  let x : ℝ := P.1,
  let y : ℝ := P.2,
  let r : ℝ := real.sqrt (x^2 + y^2),
  have x_def : x = 1 := rfl,
  have y_def : y = -real.sqrt 3 := rfl,
  have r_def : r = 2 := by linarith [x_def, y_def],
  show sin α = -real.sqrt 3 / 2,
  sorry
end

end sin_of_angle_passing_through_point_is_neg_sqrt_three_over_two_l697_697413


namespace toys_in_box_time_l697_697493

theorem toys_in_box_time :
  (∀ (n m : ℕ), n = 50 → m = 2 ∧ (5 - 3 = m)
  → (∃ t : ℕ, t = 18 ∧ (n = 50 → t * 60 = (n / m) * 45))) :=
begin
  sorry
end

end toys_in_box_time_l697_697493


namespace sum_of_3_digit_numbers_l697_697169

theorem sum_of_3_digit_numbers : 
  let digits := [1, 3, 4] in
  let valid_numbers := list.permutations digits 
    |>.map (λ p, p.foldl (λ acc d, 10 * acc + d) 0) in
  valid_numbers.sum = 1776 :=
by
  have digits := [1, 3, 4]
  have valid_numbers := list.permutations digits 
    |>.map (λ p, p.foldl (λ acc d, 10 * acc + d) 0)
  have sum_of_valid_numbers := valid_numbers.sum
  show sum_of_valid_numbers = 1776
  sorry

end sum_of_3_digit_numbers_l697_697169


namespace total_tissues_brought_l697_697544

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end total_tissues_brought_l697_697544


namespace three_digit_ends_with_itself_iff_l697_697304

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l697_697304


namespace polynomial_remainder_l697_697710

theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 7) (h2 : p 5 = 11) :
  ∃ r : ℝ → ℝ, r = (λ x, (4/3) * x + 13/3) ∧ ∃ q : ℝ → ℝ, p = λ x, q x * (x - 2) * (x - 5) + r :=
by
  sorry

end polynomial_remainder_l697_697710


namespace problem_statement_l697_697355

theorem problem_statement (a b : ℤ) (h : |a + 5| + (b - 2) ^ 2 = 0) : (a + b) ^ 2010 = 3 ^ 2010 :=
by
  sorry

end problem_statement_l697_697355


namespace midpoint_locus_inclination_pi_four_l697_697612

theorem midpoint_locus_inclination_pi_four
  (h : ∀ (x y : ℝ), (y = x / 4 - a) → ∃ (xm ym : ℝ), xm = x / 2 ∧ ym = (-a / 2)):
  ∀ (A B M : ℝ × ℝ), M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    (∃ (x y : ℝ), M = (x, y) → x + 4 * y = 0) :=
begin
  sorry
end

end midpoint_locus_inclination_pi_four_l697_697612


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697736

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l697_697736


namespace volume_of_pyramid_l697_697261

-- Define conditions
variables (x h : ℝ)
axiom x_pos : x > 0
axiom h_pos : h > 0

-- Define the main theorem/problem statement
theorem volume_of_pyramid (x h : ℝ) (x_pos : x > 0) (h_pos : h > 0) : 
  ∃ (V : ℝ), V = (1 / 6) * x^2 * h :=
by sorry

end volume_of_pyramid_l697_697261


namespace total_milk_poured_is_8_l697_697118

-- Define the parameters and conditions
def milk_poured_8_cup_bottle := 5.333333333333333
def capacity_8_cup_bottle := 8
def capacity_4_cup_bottle := 4

-- Fraction f each bottle is filled
def fraction_filled := milk_poured_8_cup_bottle / capacity_8_cup_bottle

-- Amount of milk for the 4-cup bottle
def milk_poured_4_cup_bottle := capacity_4_cup_bottle * fraction_filled

-- Total amount of milk poured into both bottles
def total_milk_poured := milk_poured_8_cup_bottle + milk_poured_4_cup_bottle

-- The theorem to be proven
theorem total_milk_poured_is_8 : total_milk_poured = 8 := by
  -- Insert proof here
  sorry

end total_milk_poured_is_8_l697_697118


namespace angle_AMB_l697_697534

-- Define the angle condition at vertex A
def angle_at_vertex_A (A B M : Type) [circle (A)] [is_tangent (A)] [intersects_other_side (B)] : Prop :=
  ∠CAB = 40

-- Define the central angle condition
def central_angle_condition (A B : Type) [circle (A)] [is_tangent (A)] [intersects_other_side (B)] : Prop :=
  ∠AOB = 80

-- Define the target angle proof
theorem angle_AMB (A B M : Type) [circle (A)] [is_tangent (A)] [intersects_other_side (B)] [point_on_minor_arc (M)] :
  ∠AMB = 100 :=
by
  -- The proof goes here
  sorry

end angle_AMB_l697_697534


namespace cos_double_angle_l697_697368

theorem cos_double_angle (α : ℝ) (hα : π / 2 < α ∧ α < π) 
  (h_sincos : sin α + cos α = sqrt 3 / 3) : cos (2 * α) = -sqrt 5 / 3 := 
sorry

end cos_double_angle_l697_697368


namespace expected_value_in_10_experiments_l697_697177

noncomputable def expected_value_successful_trials : ℚ :=
  let p : ℚ := 5/9 in
  let n : ℕ := 10 in
  n * p

theorem expected_value_in_10_experiments :
  expected_value_successful_trials = 50 / 9 :=
by
  unfold expected_value_successful_trials
  have p := 5 / 9
  have n := 10
  calc
    n * p = 10 * (5 / 9)  : by rw [n, p]
       ... = 50 / 9       : by norm_num

end expected_value_in_10_experiments_l697_697177


namespace average_percentage_reduction_l697_697034

theorem average_percentage_reduction (P0 P2 P3: ℝ) (h_init: P0 = 30) (h_final: P2 = 19.2) (h_red: P3 = 15.36) :
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ ((P0 * (1 - x) * (1 - x)) = P2) ∧ ((P2 * (1 - x)) = P3) ∧ (x = 0.2) :=
by {
  use 0.2,
  sorry
}

end average_percentage_reduction_l697_697034


namespace pure_imaginary_b_l697_697025

theorem pure_imaginary_b (b : ℝ) : (1 + b * complex.I) * (2 + complex.I) = (1 + 2 * b) * complex.I → b = 2 := 
by
  -- Proof steps are omitted
  sorry

end pure_imaginary_b_l697_697025


namespace binomial_expansion_problem_l697_697050

noncomputable theory
open_locale big_operators

theorem binomial_expansion_problem (n : ℕ) :
  (nat.choose n 2) - (nat.choose n 1) = 35 ∧
  ((n = 10) → ∃ r, r = 8 ∧ nat.choose 10 r = 45) :=
by 
{
  split,
  { sorry },
  { intros hn, rw hn, use 8, split; simp [nat.choose] },
  { sorry },
}

end binomial_expansion_problem_l697_697050


namespace sum_of_divisors_prime_factors_count_l697_697008

theorem sum_of_divisors_prime_factors_count : 
  ∑ n in (finset.range (900 + 1)), (if 900 % n = 0 then n else 0) = 7 * 13 * 31 → 
  (finset.card (finset.filter (λ p, nat.prime p) 
                        (finset.image nat.prime (↑((finset.range (31 + 1)).filter (|> nat.is_prime)))))) = 3 := 
by {
  cc,
  sorry
}

end sum_of_divisors_prime_factors_count_l697_697008


namespace totalOwlsOnFence_l697_697598

-- Define the conditions given in the problem
def initialOwls : Nat := 3
def joinedOwls : Nat := 2

-- Define the total number of owls
def totalOwls : Nat := initialOwls + joinedOwls

-- State the theorem we want to prove
theorem totalOwlsOnFence : totalOwls = 5 := by
  sorry

end totalOwlsOnFence_l697_697598


namespace num_divisors_square_l697_697621

theorem num_divisors_square (n : ℕ) (h₁ : ∃ p q : ℕ, prime p ∧ prime q ∧ p ≠ q ∧ n = p * q) :
  num_divisors (n^2) = 9 :=
by
  sorry

end num_divisors_square_l697_697621


namespace problem_remainder_of_sum_of_cubics_mod_5_l697_697081

theorem problem_remainder_of_sum_of_cubics_mod_5 :
  ∀ (b : ℕ → ℕ), (∀ n, b n < b (n + 1)) →
  (∑ i in finset.range 2021, b i = 2021^4) →
  ((∑ i in finset.range 2021, (b i) ^ 3) % 5 = 1) :=
by
  intros b hb sum_eq
  sorry

end problem_remainder_of_sum_of_cubics_mod_5_l697_697081


namespace maximum_rubles_received_max_payment_possible_l697_697937

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l697_697937


namespace probability_even_first_odd_second_l697_697495

-- Definitions based on the conditions
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Finset ℕ := {2, 4, 6}
def odd_numbers : Finset ℕ := {1, 3, 5}

-- Probability calculations
def prob_even := (even_numbers.card : ℚ) / (die_sides.card : ℚ)
def prob_odd := (odd_numbers.card : ℚ) / (die_sides.card : ℚ)

-- Proof statement
theorem probability_even_first_odd_second :
  prob_even * prob_odd = 1 / 4 :=
by
  sorry

end probability_even_first_odd_second_l697_697495


namespace petya_max_rubles_l697_697939

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l697_697939


namespace find_special_three_digit_numbers_l697_697294

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l697_697294


namespace num_prime_values_l697_697765

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def f (n : ℕ) : ℕ := n^3 - 7 * n^2 + 15 * n - 12

theorem num_prime_values (h₀ : ∀ n, n ∈ {1, 2, 3, 4, 5, 6} → (is_prime (f n)) = (n = 5)) : 
  ∑ n in {1, 2, 3, 4, 5, 6}, if is_prime (f n) then 1 else 0 = 1 := by sorry

end num_prime_values_l697_697765


namespace smallest_power_of_13_non_palindrome_l697_697712

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l697_697712


namespace smallest_non_palindromic_power_of_13_l697_697731

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l697_697731


namespace carl_wins_probability_l697_697430

noncomputable def carl_heads_prob : ℚ := 2 / 7
noncomputable def dana_heads_prob : ℚ := 3 / 8
noncomputable def dana_tails_prob : ℚ := 1 - dana_heads_prob
noncomputable def carl_tails_prob : ℚ := 1 - carl_heads_prob

theorem carl_wins_probability : 
  ∃ (p : ℚ), p = 10 / 31 ∧
  (dana_tails_prob * carl_heads_prob) * (1 + (dana_tails_prob * carl_tails_prob) + (dana_tails_prob * carl_tails_prob)^2 + ...) = p :=
by
  -- Proof goes here
  sorry

end carl_wins_probability_l697_697430


namespace regular_hexagon_interior_angle_l697_697004

theorem regular_hexagon_interior_angle : ∀ (n : ℕ), n = 6 → ∀ (angle_sum : ℕ), angle_sum = (n - 2) * 180 → (∀ (angle : ℕ), angle = angle_sum / n → angle = 120) :=
by sorry

end regular_hexagon_interior_angle_l697_697004


namespace max_rubles_l697_697963

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l697_697963


namespace delivery_cost_l697_697872

theorem delivery_cost (base_fee : ℕ) (limit : ℕ) (extra_fee : ℕ) 
(item_weight : ℕ) (total_cost : ℕ) 
(h1 : base_fee = 13) (h2 : limit = 5) (h3 : extra_fee = 2) 
(h4 : item_weight = 7) (h5 : total_cost = 17) : 
  total_cost = base_fee + (item_weight - limit) * extra_fee := 
by
  sorry

end delivery_cost_l697_697872


namespace chord_eq_radius_central_angle_eq_pi_div_3_l697_697844

theorem chord_eq_radius_central_angle_eq_pi_div_3 (r : ℝ) :
  ∀ (C : Type) [metric_space C] [circle C] (o : C) (a b : C),
    dist o a = r → dist o b = r → dist a b = r → central_angle o a b = π / 3 :=
by
  intros C hms hc o a b hoa hob hab
  sorry

end chord_eq_radius_central_angle_eq_pi_div_3_l697_697844


namespace max_rubles_can_receive_l697_697920

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l697_697920


namespace smallest_non_palindrome_power_of_13_l697_697724

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l697_697724


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697747

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l697_697747


namespace smallest_non_palindrome_power_of_13_l697_697721

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l697_697721


namespace divisors_of_square_of_cube_of_prime_l697_697618

theorem divisors_of_square_of_cube_of_prime (p : ℕ) (hp : p.prime) (n : ℕ) (h : n = p^3) :
  nat.num_divisors (n^2) = 7 :=
sorry

end divisors_of_square_of_cube_of_prime_l697_697618


namespace smallest_non_palindrome_power_of_13_l697_697756

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l697_697756


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697752

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l697_697752


namespace solution_set_inequality_l697_697138

theorem solution_set_inequality (x : ℝ) : (0 < x ∧ x < 1) ↔ (1 / (x - 1) < -1) :=
by
  sorry

end solution_set_inequality_l697_697138


namespace rational_roots_of_polynomial_l697_697397

theorem rational_roots_of_polynomial (p : ℕ) (hp : Prime p) :
  {x : ℚ | (Polynomial.aeval x (Polynomial.X ^ 4 + (2 - p) * Polynomial.X ^ 3 + (2 - 2 * p) * Polynomial.X ^ 2 + (1 - 2 * p) * Polynomial.X - p) = 0)}.to_finset.card = 2 :=
sorry

end rational_roots_of_polynomial_l697_697397


namespace time_to_cross_is_correct_l697_697236

noncomputable def train_cross_bridge_time : ℝ :=
  let length_train := 130
  let speed_train_kmh := 45
  let length_bridge := 245.03
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_train_ms
  time

theorem time_to_cross_is_correct : train_cross_bridge_time = 30.0024 :=
by
  sorry

end time_to_cross_is_correct_l697_697236


namespace correct_statements_count_l697_697994

axiom h1 : ∀ n : ℤ, n < 0 → n ≤ -1

axiom h2 : ∀ a : ℤ, a ≠ 0 → (1 / (a : ℤ) = (1 : ℚ) / a)

axiom h3 : ∀ a b : ℤ, a = -b → a / b = -1

axiom h4 : (-2) ^ 3 = -2 ^ 3

axiom h5 : (2 : ℚ) / 3 ≠ 2

axiom h6 : ¬(∀ (x y : ℚ), xy^2 - xy + 24 = (c : ℚ) * x^3 + (d : ℚ) * x^2 + (e : ℚ) * x + f * y^3 + g * y^2 + h * y + i)

theorem correct_statements_count : finset.card (finset.filter (λ s, s) (finset.insert true (finset.insert false (finset.insert false (finset.insert true (finset.insert false (finset.singleton false))))))) = 2 := sorry

end correct_statements_count_l697_697994


namespace decreasing_monotonicity_phi_range_l697_697573

theorem decreasing_monotonicity_phi_range :
  ∀ (φ : ℝ), (0 < φ ∧ φ ≤ π / 2) →
    (∀ x, (π/4 < x ∧ x < π/2) → 2 * sin (2 * x + 2 * φ - π / 3) < 2 * sin (2 * (x + 1e-6) + 2 * φ - π / 3)) ↔ 
    (π / 6 ≤ φ ∧ φ ≤ 5 * π / 12) :=
by sorry

end decreasing_monotonicity_phi_range_l697_697573


namespace find_value_l697_697824

noncomputable def condition (a b θ : ℝ) : Prop :=
  (sin θ ^ 6 / a^2) + (cos θ ^ 6 / b^2) = 1 / (a + b)

theorem find_value (a b θ : ℝ) (h : condition a b θ) :
  (sin θ ^ 12 / a^5) + (cos θ ^ 12 / b^5) = 1 / (a + b)^5 :=
  sorry

end find_value_l697_697824


namespace eq_AD_l697_697463

noncomputable def find_AD (AB AC BD CD : ℝ) (BD_CD_ratio : ℝ) :=
  let h := 8 * Real.sqrt 2 in
  h

theorem eq_AD :
  ∀ (A B C D : Type) (AB AC : ℝ), 
    AB = 13 → 
    AC = 20 → 
    BD/CD = 3/4 → 
    AD = 8 * Real.sqrt 2 :=
by
  intros
  sorry

end eq_AD_l697_697463


namespace all_increased_quadratics_have_integer_roots_l697_697123

def original_quadratic (p q : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -p ∧ α * β = q

def increased_quadratic (p q n : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -(p + n) ∧ α * β = (q + n)

theorem all_increased_quadratics_have_integer_roots (p q : ℤ) :
  original_quadratic p q →
  (∀ n, 0 ≤ n ∧ n ≤ 9 → increased_quadratic p q n) :=
sorry

end all_increased_quadratics_have_integer_roots_l697_697123


namespace steering_wheel_right_l697_697270

/-
Problem Statement:
Determine on which side the steering wheel is located in the car shown in the picture given:
1. The rearview mirrors provide the driver with a view of the road behind the vehicle.
2. The mirror on the driver's side is positioned almost perpendicular to the axis of the car.
3. The mirror on the passenger side is set at an angle, approximately 45 degrees.
4. Observing the image, one mirror is perpendicular and the other is angled close to 45 degrees.
-/
def is_right_steering_wheel (mirrors_perpendicular : bool) (mirror_angle_45 : bool) : Prop :=
  mirrors_perpendicular = true ∧ mirror_angle_45 = true → true

theorem steering_wheel_right 
  (mirrors_perpendicular : bool) 
  (mirror_angle_45 : bool) 
  (h1 : mirrors_perpendicular = true)
  (h2 : mirror_angle_45 = true) : 
  is_right_steering_wheel mirrors_perpendicular mirror_angle_45 :=
by
  sorry

end steering_wheel_right_l697_697270


namespace total_tissues_brought_l697_697547

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end total_tissues_brought_l697_697547


namespace coordinates_reflection_y_axis_l697_697126

theorem coordinates_reflection_y_axis (x y : ℤ) (hx : x = 4) (hy : y = -1) :
  (∃ x' y', (x' = -4) ∧ (y' = -1)) :=
begin
  use [-4, -1],
  split;
  solve_by_elim,
  sorry,
end

end coordinates_reflection_y_axis_l697_697126


namespace angles_equal_l697_697890

-- Given statements
variable {A B O C D E F : Point}
variable {circle1 : Circle}
variable {circle2 : Circle}

-- Conditions
variable (h1 : ∠A O B = ∠C O D)
variable (h2 : circle1.inscribed_in_angle A O B)
variable (h3 : circle2.inscribed_in_angle C O D)
variable (h4 : E ∈ circle1 ∧ E ∈ circle2)
variable (h5 : F ∈ circle1 ∧ F ∈ circle2)

-- Goal
theorem angles_equal :
  ∠A O E = ∠D O F :=
by
  sorry

end angles_equal_l697_697890


namespace total_tissues_brought_l697_697546

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end total_tissues_brought_l697_697546


namespace three_distinct_sums_among_eight_cycles_l697_697778

open Polynomial

noncomputable def cubic_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d, ∀ x, f x = a * x^3 + b * x^2 + c * x + d

def is_cycle (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f a = b ∧ f b = c ∧ f c = a ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

noncomputable def eight_cycles (f : ℝ → ℝ) (cycles : fin 8 → (ℝ × ℝ × ℝ)) : Prop :=
  ∃ numbers : fin 24 → ℝ, bijective numbers ∧
  (∀ i, is_cycle f (fst (cycles i)) (fst (snd (cycles i))) (snd (snd (cycles i))) ∧
        fst (cycles i) ∈ set.range numbers ∧
        fst (snd (cycles i)) ∈ set.range numbers ∧
        snd (snd (cycles i)) ∈ set.range numbers)

theorem three_distinct_sums_among_eight_cycles
  (f : ℝ → ℝ)
  (hf : cubic_polynomial f)
  (cycles : fin 8 → (ℝ × ℝ × ℝ))
  (hcycles : eight_cycles f cycles) :
  ∃ (s₁ s₂ s₃ : ℝ), 
  s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ ∧
  (∃ i₁ i₂ i₃, s₁ = (fst (cycles i₁) + fst (snd (cycles i₁)) + snd (snd (cycles i₁))) ∧
                s₂ = (fst (cycles i₂) + fst (snd (cycles i₂)) + snd (snd (cycles i₂))) ∧
                s₃ = (fst (cycles i₃) + fst (snd (cycles i₃)) + snd (snd (cycles i₃)))) :=
sorry

end three_distinct_sums_among_eight_cycles_l697_697778


namespace exists_two_same_color_at_distance_l697_697157

-- Define the right isosceles triangle and the problem setup
def point := ℝ × ℝ

structure triangle :=
(vertices : point × point × point)
(hypotenuse_length : dist (vertices.1.2, vertices.2.2) = 1)
(isosceles_right : dist (vertices.1.1, vertices.1.2) = 1 ∧ dist (vertices.2.1, vertices.2.2) = 1)

def colored_triangle_points (T : triangle) :=
∀ p : point, p ∈ (set.univ : set point) →
  (colored p = "red" ∨ colored p = "yellow" ∨ colored p = "green" ∨ colored p = "blue")

-- Proof statement
theorem exists_two_same_color_at_distance {T : triangle}
  (hT : colored_triangle_points T) : 
  ∃ p1 p2 : point, colored p1 = colored p2 ∧ dist (p1, p2) ≥ 2 - real.sqrt 2 :=
sorry

end exists_two_same_color_at_distance_l697_697157


namespace james_total_time_l697_697058

def research_time_1 : ℕ := 6  -- months
def expedition_time_1 : ℕ := 24  -- months (2 years)

def research_time_2 : ℕ := 3 * research_time_1
def expedition_time_2 : ℕ := 2 * expedition_time_1

def research_time_3 : ℕ := research_time_2 / 2
def expedition_time_3 : ℕ := expedition_time_2

def research_time_4 : ℕ := research_time_1
def expedition_time_4 : ℕ := expedition_time_3 + expedition_time_3 / 2

def research_time_5 : ℕ := research_time_4 + research_time_4 / 4
def expedition_time_5 : ℕ := 3 * (research_time_1 + expedition_time_1) / 2

def total_time : ℕ := research_time_1 + expedition_time_1 + 
                      research_time_2 + expedition_time_2 + 
                      research_time_3 + expedition_time_3 + 
                      research_time_4 + expedition_time_4 + 
                      research_time_5 + expedition_time_5 

theorem james_total_time : total_time = 283.5 := by
  sorry

end james_total_time_l697_697058


namespace grid_points_on_parabola_l697_697239

theorem grid_points_on_parabola : 
  let side_length := 2 * 10 ^ (-4)
  let grid_points := { ⟨x, y⟩ | x = n * side_length ∧ y = m * side_length ∧ 0 ≤ n ∧ n ≤ 5000 ∧ 0 ≤ m ∧ m ≤ 5000 }
  let on_parabola := { ⟨x, y⟩ | y = x ^ 2 }
  let valid_points := grid_points ∩ on_parabola
  in valid_points.card = 49 :=
by 
  sorry

end grid_points_on_parabola_l697_697239


namespace complex_number_solution_l697_697696

open Complex

noncomputable def findComplexNumber (z : ℂ) : Prop :=
  abs (z - 2) = abs (z + 4) ∧ abs (z - 2) = abs (z - Complex.I * 2)

theorem complex_number_solution :
  ∃ z : ℂ, findComplexNumber z ∧ z = -1 + Complex.I :=
by
  sorry

end complex_number_solution_l697_697696


namespace range_of_a_l697_697843

def f (x a : ℝ) : ℝ :=
  x^2 + a * |x - 1/2|

def is_monotone_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y ∈ s, x ≤ y → f x ≤ f y

theorem range_of_a (a : ℝ) : 
  (is_monotone_on (f a) (Set.Ici 0)) → -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l697_697843


namespace tan_double_angle_l697_697380

theorem tan_double_angle (x y : ℝ) (h : (x, y) = (3 / 5, 4 / 5)) :
    let α := real.atan2 y x
    ∧ tanα : ℝ := y / x
    ∧ tan2α : ℝ := (2 * tanα) / (1 - tanα ^ 2)
    in tan2α = -24 / 7 := 
by
  -- Definitions of x, y, α, tanα and tan2α
  let α := real.atan2 y x
  let tanα := y / x
  let tan2α := (2 * tanα) / (1 - tanα ^ 2)
  -- Substitute and simplify according to the given conditions
  sorry

end tan_double_angle_l697_697380


namespace find_missing_digit_l697_697539

theorem find_missing_digit 
  (x : Nat) 
  (h : 16 + x ≡ 0 [MOD 9]) : 
  x = 2 :=
sorry

end find_missing_digit_l697_697539


namespace range_of_a_l697_697268

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the main theorem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, tensor (x - a) (x + a) < 1) → 
  (-((1 : ℝ) / 2) < a ∧ a < (3 : ℝ) / 2) :=
by
  sorry

end range_of_a_l697_697268


namespace all_integers_appear_exactly_once_l697_697901

noncomputable def sequence_of_integers (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ m : ℕ, a m > 0 ∧ ∃ m' : ℕ, a m' < 0

noncomputable def distinct_modulo_n (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, (∀ i j : ℕ, i < j ∧ j < n → a i % n ≠ a j % n)

theorem all_integers_appear_exactly_once
  (a : ℕ → ℤ)
  (h_seq : sequence_of_integers a)
  (h_distinct : distinct_modulo_n a) :
  ∀ x : ℤ, ∃! i : ℕ, a i = x := 
sorry

end all_integers_appear_exactly_once_l697_697901


namespace sec_minus_tan_l697_697833

theorem sec_minus_tan (x : ℝ) (h : sec x + tan x = 3) : sec x - tan x = 1 / 3 :=
sorry

end sec_minus_tan_l697_697833


namespace locus_of_Y_right_angled_triangle_l697_697361

-- Conditions definitions
variables {A B C : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (b c m : ℝ) -- Coordinates and slopes related to the problem
variables (x : ℝ) -- Independent variable for the locus line

-- The problem statement
theorem locus_of_Y_right_angled_triangle 
  (A_right_angle : ∀ (α β : ℝ), α * β = 0) 
  (perpendicular_lines : b ≠ m * c) 
  (no_coincide : (b^2 * m - 2 * b * c - c^2 * m) ≠ 0) :
  ∃ (y : ℝ), y = (2 * b * c * (b * m - c) - x * (b^2 + 2 * b * c * m - c^2)) / (b^2 * m - 2 * b * c - c^2 * m) := 
sorry

end locus_of_Y_right_angled_triangle_l697_697361


namespace infinite_solutions_exist_l697_697971

open Nat

noncomputable def binom : ℕ → ℕ → ℕ
| n 0 := 1
| 0 k := 0
| (n+1) (k+1) := binom n k + binom n (k+1)

theorem infinite_solutions_exist :
  (∃ (a b d : ℕ), d^2 = binom a 2 - binom b 2) ∧
  (∃ (a b d : ℕ), d^3 = binom a 3 - binom b 3) :=
by
  sorry

end infinite_solutions_exist_l697_697971


namespace sum_of_geometric_sequence_l697_697555

theorem sum_of_geometric_sequence (a : ℕ → ℝ) (n : ℕ) 
  (h_geom : ∀ k, a (k+1) = (a k) * (1/2))
  (a_2_eq_2 : a 2 = 2) (a_5_eq_1_div_4 : a 5 = 1/4) :
  ∑ i in Finset.range n, (a i) * (a (i+1)) = (32 / 3) * (1 - 4^(-n)) := by
  sorry

end sum_of_geometric_sequence_l697_697555


namespace sequence_solution_m_range_l697_697359

-- Defining the conditions for the geometric sequence
def geometric_sequence (a : ℕ → ℝ) :=
  a 2 + a 3 + a 4 = 28 ∧
  a 3 + 2 = (a 2 + a 4) / 2 ∧
  monotone a

-- Defining the problem statement
theorem sequence_solution :
  ∃ a : ℕ → ℝ, geometric_sequence a ∧
  (∀ n : ℕ, a n = 2 ^ n) :=
by
  sorry

-- Defining bn and Sn based on the given equations
def b (a : ℕ → ℝ) (n : ℕ) := a n * real.log (1 / 2) (a n)
def S (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range (n + 1), b a i

-- Range of m based on given conditions
theorem m_range (a : ℕ → ℝ) (mono_a : monotone a) (m : ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, S a n + (n + m) * a (n + 1) < 0) →
  m ≤ -1 :=
by
  sorry

end sequence_solution_m_range_l697_697359


namespace sum_of_x_that_satisfy_l697_697886

def f (x : ℝ) : ℝ := 16 * x + 3

theorem sum_of_x_that_satisfy : 
  let f_inv x := (x - 3) / 16,
      inv_2x := (λ x, (2*x)⁻¹),
      eq_finv_fx := (f_inv = f ∘ inv_2x) in
  (eq_finv_fx) → 
  let quadr_eq := x^2 - 51*x - 128 = 0 in
  ∃ sum_x : ℝ, sum_x = 51 :=
  -- sorry is used to skip the proof.
  sorry

end sum_of_x_that_satisfy_l697_697886


namespace set_intersection_and_polynomial_solution_l697_697393

theorem set_intersection_and_polynomial_solution {a b : ℝ} :
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  (A ∩ B = {x | x < -3}) ∧ ((A ∪ B = {x | x < -2 ∨ x > 1}) →
    (a = 2 ∧ b = -4)) :=
by
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  sorry

end set_intersection_and_polynomial_solution_l697_697393


namespace units_digit_2015_powers_sum_l697_697170

theorem units_digit_2015_powers_sum :
  (2015^2 % 10 + 2015^0 % 10 + 2015^1 % 10 + 2015^5 % 10) % 10 = 6 :=
by 
  -- Simplify each term's units digit
  have h1 : 2015^2 % 10 = 5, 
  { -- 2015^2 ends with the same digit as 5^2, which is 25, thus 5
    sorry },
  
  have h2 : 2015^0 % 10 = 1, 
  { -- Any number to the power of 0 is 1
    sorry },

  have h3 : 2015^1 % 10 = 5, 
  { -- 2015^1 ends with the same digit as 5^1, which is 5
    sorry },
  
  have h4 : 2015^5 % 10 = 5, 
  { -- 2015^5 ends with the same digit as 5^5, which is 5
    sorry },
  
  -- Sum the simplified units digits and take modulo 10
  calc
    (2015^2 % 10 + 2015^0 % 10 + 2015^1 % 10 + 2015^5 % 10) % 10
        = (5 + 1 + 5 + 5) % 10 : by rw [h1, h2, h3, h4]
    ... = 16 % 10              : by norm_num
    ... = 6                    : by norm_num

end units_digit_2015_powers_sum_l697_697170


namespace average_of_integers_l697_697160

theorem average_of_integers (h : ∀ N : ℤ, 20 < N ∧ N < 30):
  ∑ N in { N | 20 < N ∧ N < 30 }.toFinset / (card { N | 20 < N ∧ N < 30 }.toFinset) = 25 := 
by
  sorry

end average_of_integers_l697_697160


namespace sum_of_binary_numbers_with_adjacent_ones_equals_381_l697_697130

theorem sum_of_binary_numbers_with_adjacent_ones_equals_381 :
  let binary_numbers := [3, 6, 12, 24, 48, 96, 192] in
  (∑ x in binary_numbers, x) = 381 :=
by {
  sorry
}

end sum_of_binary_numbers_with_adjacent_ones_equals_381_l697_697130


namespace wage_payment_days_l697_697637

noncomputable theory
open_locale classical

theorem wage_payment_days (A B : ℝ) (h1 : ∀ t : ℝ, t * B = 30 * B) (h2 : ∀ t : ℝ, t * (A + B) = 12 * (A + B)) :
  (∀ D : ℝ, D * A = 30 * B → D = 20) :=
by {
  -- Suppose M is sufficient to pay A's wages for D days
  -- D * A = 30 * B
  -- Show that D = 20
  sorry
}

end wage_payment_days_l697_697637


namespace ellipse_hexagon_proof_l697_697384

noncomputable def m_value : ℝ := 3 + 2 * Real.sqrt 3

theorem ellipse_hexagon_proof (m : ℝ) (k : ℝ) 
  (hk : k ≠ 0) (hm : m > 3) :
  (∀ x y : ℝ, (x / m)^2 + (y / 3)^2 = 1 ∧ (y = k * x ∨ y = -k * x)) →
  k = Real.sqrt 3 →
  (|((4*m)/(m+1)) - (m-3)| = 0) →
  m = m_value :=
by
  sorry

end ellipse_hexagon_proof_l697_697384


namespace largest_perimeter_regular_polygons_l697_697152

theorem largest_perimeter_regular_polygons :
  ∃ (p q r : ℕ), 
    (p ≥ 3 ∧ q ≥ 3 ∧ r >= 3) ∧
    (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧
    (180 * (p - 2)/p + 180 * (q - 2)/q + 180 * (r - 2)/r = 360) ∧
    ((p + q + r - 6) = 9) :=
sorry

end largest_perimeter_regular_polygons_l697_697152


namespace sum_of_midpoints_l697_697141

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l697_697141


namespace cube_split_l697_697345

theorem cube_split (m : ℕ) (h : m > 1) (hm3_split_contains_31 : ∃ (a b : list ℕ), 
  (a.length = m ∧ b.length = m ∧ (∀ (x ∈ a), x % 2 = 1) ∧ 
  (∀ (y ∈ b), y = x ∈ a ∧ b.sum = m ^ 3 ∧ 31 ∈ b))) : m = 6 := 
by sorry

end cube_split_l697_697345


namespace angle_between_vectors_is_29_5_degrees_l697_697693

noncomputable def vec_a : ℝ × ℝ × ℝ := (3, -2, 2)
noncomputable def vec_b : ℝ × ℝ × ℝ := (1, -1, 1)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cos_theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
dot_product v1 v2 / (magnitude v1 * magnitude v2)

def theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
Real.arccos (cos_theta v1 v2) * 180 / Real.pi

theorem angle_between_vectors_is_29_5_degrees :
  theta vec_a vec_b = 29.5 :=
sorry

end angle_between_vectors_is_29_5_degrees_l697_697693


namespace trig_ineq_l697_697825

theorem trig_ineq (θ : ℝ) (h : -real.pi / 8 < θ ∧ θ < 0) : real.tan θ < real.sin θ ∧ real.sin θ < real.cos θ := 
sorry

end trig_ineq_l697_697825


namespace smallest_power_of_13_non_palindrome_l697_697718

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l697_697718


namespace tangent_line_to_curve_perpendicular_l697_697024

noncomputable def perpendicular_tangent_line (x y : ℝ) : Prop :=
  y = x^4 ∧ (4*x - y - 3 = 0)

theorem tangent_line_to_curve_perpendicular {x y : ℝ} (h : y = x^4 ∧ (4*x - y - 3 = 0)) :
  ∃ (x y : ℝ), (x+4*y-8=0) ∧ (4*x - y - 3 = 0) :=
by
  sorry

end tangent_line_to_curve_perpendicular_l697_697024


namespace number_of_arrangements_l697_697822

-- Define the concept of a digit list and its properties.
def is_six_digit_number (l : List ℕ) : Prop := l.length = 6
def contains_digits (l : List ℕ) : Prop := l.count 3 = 4 ∧ l.count 5 = 2

-- Define the divisibility condition for 11.
def divisible_by_11 (l : List ℕ) : Prop := 
  ((l.nth 0).get_or_else 0 + (l.nth 2).get_or_else 0 + (l.nth 4).get_or_else 0
    - (l.nth 1).get_or_else 0 - (l.nth 3).get_or_else 0 - (l.nth 5).get_or_else 0) % 11 = 0

-- Define the main theorem.
theorem number_of_arrangements : 
  ∃! l : List ℕ, is_six_digit_number l ∧ contains_digits l ∧ divisible_by_11 l ∧ l.permutations.length = 9 :=
sorry

end number_of_arrangements_l697_697822


namespace find_a_l697_697810

noncomputable def f (a : ℝ) (x : ℝ) := a * x * Real.log x

theorem find_a (a : ℝ) (h : (f a 1)' = 3) : a = 3 :=
by {
  unfold f,
  rw [Real.log_one, zero_mul, add_zero] at h,
  simp at h,
  exact h,
}

end find_a_l697_697810


namespace inequality_proof_l697_697095

theorem inequality_proof (x y : ℝ) : 
  -1 / 2 ≤ (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ∧
  (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ≤ 1 / 2 :=
sorry

end inequality_proof_l697_697095


namespace license_plate_palindrome_probability_l697_697907

theorem license_plate_palindrome_probability :
  let p := 155
  let q := 13520
  let probability := p / q
  p + q = 13675 := by {
  -- Definitions based on conditions
  let four_letter_palindrome_probability := 1 / 676
  let four_digit_palindrome_probability := 1 / 100
  let combined_probability := four_letter_palindrome_probability + four_digit_palindrome_probability - (four_letter_palindrome_probability * four_digit_palindrome_probability)
  have : combined_probability = 155 / 13520 := sorry
  have : p = 155 := rfl
  have : q = 13520 := rfl
  have : p + q = 155 + 13520 := rfl
  exact this
}

end license_plate_palindrome_probability_l697_697907


namespace convert_cylindrical_to_rectangular_l697_697266

theorem convert_cylindrical_to_rectangular (r θ z x y : ℝ) (h_r : r = 5) (h_θ : θ = (3 * Real.pi) / 2) (h_z : z = 4)
    (h_x : x = r * Real.cos θ) (h_y : y = r * Real.sin θ) :
    (x, y, z) = (0, -5, 4) :=
by
    sorry

end convert_cylindrical_to_rectangular_l697_697266


namespace total_tissues_brought_l697_697543

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end total_tissues_brought_l697_697543


namespace correct_propositions_are_123_l697_697341

theorem correct_propositions_are_123
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x-1) = -f x → f x = f (x-2))
  (h2 : ∀ x, f (1 - x) = f (x - 1) → f (1 - x) = -f x)
  (h3 : ∀ x, f (x) = -f (-x)) :
  (∀ x, f (x-1) = -f x → ∃ c, c * (f (1-1)) = -f x) ∧
  (∀ x, f (1 - x) = f (x - 1) → ∀ x, f x = f (-x)) ∧
  (∀ x, f (x-1) = -f x → ∀ x, f (x - 2) = f x) :=
sorry

end correct_propositions_are_123_l697_697341


namespace find_AD_l697_697450

noncomputable def AD (AB AC BD_ratio CD_ratio : ℝ) : ℝ :=
  if h = sqrt(128) then h else 0

theorem find_AD 
  (AB AC : ℝ) 
  (BD_ratio CD_ratio : ℝ)
  (h : ℝ)
  (h_eq : h = sqrt 128) :
  AB = 13 → 
  AC = 20 → 
  BD_ratio / CD_ratio = 3 / 4 → 
  AD AB AC BD_ratio CD_ratio = 8 * Real.sqrt 2 :=
by
  intro h_eq AB_eq AC_eq ratio_eq
  simp only [AD, h_eq, AB_eq, AC_eq, ratio_eq]
  sorry

end find_AD_l697_697450


namespace mode_of_goals_is_3_l697_697438

-- Definitions based on the conditions:
def num_students : ℕ := 12
def shots_per_student : ℕ := 10

-- The number of goals and the corresponding number of students
def goals_counts : List (ℕ × ℕ) :=
  [(1, 1), (2, 1), (3, 4), (4, 2), (5, 3), (7, 1)]

-- The goal of this problem is to prove that the mode of the number of goals is 3.
def mode_of_goals_correct : Prop :=
  let mode : ℕ := 3 in
  ∃ mode, (∃ count, (mode, count) ∈ goals_counts) ∧ ∀ g c, (g, c) ∈ goals_counts → c ≤ 4

theorem mode_of_goals_is_3 : mode_of_goals_correct :=
  sorry

end mode_of_goals_is_3_l697_697438


namespace sum_of_nonzero_perfect_squares_l697_697774

theorem sum_of_nonzero_perfect_squares (p n : ℕ) (hp_prime : Nat.Prime p) 
    (hn_ge_p : n ≥ p) (h_perfect_square : ∃ k : ℕ, 1 + n * p = k^2) :
    ∃ (a : ℕ) (f : Fin p → ℕ), (∀ i, 0 < f i ∧ ∃ m, f i = m^2) ∧ (n + 1 = a + (Finset.univ.sum f)) :=
sorry

end sum_of_nonzero_perfect_squares_l697_697774


namespace find_y_value_l697_697338

theorem find_y_value : (15^3 * 7^4) / 5670 = 1428.75 := by
  sorry

end find_y_value_l697_697338


namespace species_population_estimate_l697_697428

theorem species_population_estimate :
  ∃ (N_A N_B N_C : ℕ), 
    (40 / 2400 = 3 / 180) ∧
    (40 / 1440 = 5 / 180) ∧
    (40 / 3600 = 2 / 180) ∧
    N_A = 2400 ∧
    N_B = 1440 ∧
    N_C = 3600 :=
by {
  existsi (2400 : ℕ),
  existsi (1440 : ℕ),
  existsi (3600 : ℕ),
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { reflexivity },
  split,
  { reflexivity },
  { reflexivity }
}

end species_population_estimate_l697_697428


namespace inequality_proof_l697_697109

def inequality_solution (x : ℝ) : Prop :=
  (x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)

theorem inequality_proof (x : ℝ) :
  (1 < x ∨ x < 2) ∧ (2 < x ∨ x < 3) ∧ (3 < x ∨ x < 4) ∧ (4 < x ∨ x < 5) ∧ (5 < x ∨ x < 6) →
  inequality_solution x :=
begin
  -- Proof goes here
  sorry,
end

end inequality_proof_l697_697109


namespace inequality_square_l697_697352

theorem inequality_square (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end inequality_square_l697_697352


namespace least_integer_value_abs_l697_697581

theorem least_integer_value_abs (x : ℤ) : 
  (∃ x : ℤ, (abs (3 * x + 5) ≤ 20) ∧ (∀ y : ℤ, (abs (3 * y + 5) ≤ 20) → x ≤ y)) ↔ x = -8 :=
by
  sorry

end least_integer_value_abs_l697_697581


namespace choose_6_starters_with_one_triplet_l697_697092

theorem choose_6_starters_with_one_triplet :
  let num_players := 14
  let triplets := 3
  let starters := 6
  let triplet_case := 1
  let non_triplets := num_players - triplets
  let choices_for_triplet := 3
  let choices_for_non_triplets := (non_triplets.choose (starters - triplet_case))
  in choices_for_triplet * choices_for_non_triplets = 1386 :=
by
  let num_players := 14
  let triplets := 3
  let starters := 6
  let triplet_case := 1
  let non_triplets := num_players - triplets
  let choices_for_triplet := 3
  let choices_for_non_triplets := (non_triplets.choose (starters - triplet_case))
  exact calc
    choices_for_triplet * choices_for_non_triplets = 3 * 462 : by sorry -- this combines the calculation steps
    ... = 1386 : rfl -- the final result is 1386, matching the answer

end choose_6_starters_with_one_triplet_l697_697092


namespace max_angle_AFB_l697_697131

noncomputable def focus_of_parabola := (2, 0)
def parabola (x y : ℝ) := y^2 = 8 * x
def on_parabola (A B : ℝ × ℝ) := parabola A.1 A.2 ∧ parabola B.1 B.2
def condition (x1 x2 : ℝ) (AB : ℝ) := x1 + x2 + 4 = (2 * Real.sqrt 3 / 3) * AB

theorem max_angle_AFB (A B : ℝ × ℝ) (x1 x2 : ℝ) (AB : ℝ)
  (h1 : on_parabola A B)
  (h2 : condition x1 x2 AB)
  (hA : A.1 = x1)
  (hB : B.1 = x2) :
  ∃ θ, θ ≤ Real.pi * 2 / 3 := 
  sorry

end max_angle_AFB_l697_697131


namespace second_snowplow_time_needed_l697_697154

variable (rate1 rate2 timeTogether timeNeeded workTogether remainingWork : ℝ)

-- Conditions:
def rate1 := 1 -- work rate of the first snowplow (jobs per hour)
def rate2 := 4 / 3 -- work rate of the second snowplow (jobs per hour)
def timeTogether := 1 / 3 -- time both snowplows work together (hours)
def workTogether := (rate1 + rate2) * timeTogether -- work done together (jobs)
def remainingWork := 1 - workTogether -- remaining work after 20 minutes (jobs)
def timeNeeded := remainingWork / rate2 -- time needed for the second snowplow (hours)

theorem second_snowplow_time_needed : timeNeeded = 1 / 6 :=
by
  rw [rate1, rate2, timeTogether, workTogether, remainingWork, timeNeeded]
  norm_num
  sorry

end second_snowplow_time_needed_l697_697154


namespace function_fixed_point_l697_697995

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  2^x + log a (x + 1) + 3

theorem function_fixed_point (a : ℝ) : f 0 a = 4 :=
by
  unfold f
  sorry

end function_fixed_point_l697_697995


namespace blake_change_l697_697663

-- Definitions based on conditions
def n_l : ℕ := 4
def n_c : ℕ := 6
def p_l : ℕ := 2
def p_c : ℕ := 4 * p_l
def amount_given : ℕ := 6 * 10

-- Total cost calculations derived from the conditions
def total_cost_lollipops : ℕ := n_l * p_l
def total_cost_chocolates : ℕ := n_c * p_c
def total_cost : ℕ := total_cost_lollipops + total_cost_chocolates

-- Calculating the change
def change : ℕ := amount_given - total_cost

-- Theorem stating the final answer
theorem blake_change : change = 4 := sorry

end blake_change_l697_697663


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697222

open Real

theorem height_of_cylinder_inscribed_in_hemisphere
  (r_cylinder r_hemisphere : ℝ)
  (h_radius_cylinder : r_cylinder = 3)
  (h_radius_hemisphere : r_hemisphere = 8) :
  ∃ h_cylinder : ℝ, h_cylinder = sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) :=
by
  use sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  sorry

#check height_of_cylinder_inscribed_in_hemisphere

end height_of_cylinder_inscribed_in_hemisphere_l697_697222


namespace class_president_and_vice_president_count_l697_697013

/--
There are 3 candidates for class president and 5 candidates for vice president.
Prove that there are 15 ways to choose one class president and one vice president.
-/
theorem class_president_and_vice_president_count :
  let president_candidates := 3
  let vice_president_candidates := 5
  president_candidates * vice_president_candidates = 15 :=
by
  let president_candidates := 3
  let vice_president_candidates := 5
  show president_candidates * vice_president_candidates = 15
  from sorry

end class_president_and_vice_president_count_l697_697013


namespace perimeter_of_new_region_is_2_25_l697_697604

noncomputable def side_length : ℝ := 1 / Real.pi
noncomputable def radius (side_length : ℝ) : ℝ := side_length / 2

def perimeter_new_region (side_length : ℝ) : ℝ :=
  let circumference := Real.pi * side_length in
  let semicircle_perimeter := circumference / 2 in
  let quarter_circle_perimeter := circumference * 0.75 in
  3 * semicircle_perimeter + quarter_circle_perimeter

theorem perimeter_of_new_region_is_2_25 :
  perimeter_new_region (1 / Real.pi) = 2.25 := by
  sorry

end perimeter_of_new_region_is_2_25_l697_697604


namespace divisors_of_square_of_cube_of_prime_l697_697619

theorem divisors_of_square_of_cube_of_prime (p : ℕ) (hp : p.prime) (n : ℕ) (h : n = p^3) :
  nat.num_divisors (n^2) = 7 :=
sorry

end divisors_of_square_of_cube_of_prime_l697_697619


namespace solve_system_l697_697117

theorem solve_system :
  ∃ (m k : ℚ), (3 * k - 4) / (k + 7) = 2 / 5 ∧ 2 * m + 5 * k = 14 ∧ k = 34 / 13 ∧ m = 6 / 13 :=
by
  use 6 / 13, 34 / 13
  split
  { calc (3 * (34 / 13) - 4) / ((34 / 13) + 7) = (102 / 13 - 4) / (34 / 13 + 91 / 13) : by norm_num
    ... = (102 / 13 - 52 / 13) / (125 / 13) : by norm_num
    ... = 50 / 13 / (125 / 13) : by norm_num
    ... = 50 / 125 : by norm_num
    ... = 2 / 5 : by norm_num }
  split
  { calc 2 * (6 / 13) + 5 * (34 / 13) = 12 / 13 + 170 / 13 : by norm_num
    ... = 182 / 13 : by norm_num
    ... = 14 : by norm_num }
  split
  { exact (rfl : 34 / 13 = 34 / 13) }
  { exact (rfl : 6 / 13 = 6 / 13) }

end solve_system_l697_697117


namespace max_tan_B_l697_697870

-- Given
variables (A B C : Type) [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]
variable (triangle_ABC : Prop)
variables (AB BC : ℝ)
variables (B_pos : BC > 0) (AB_fixed : AB = 25) (BC_fixed : BC = 15)

-- To Prove
def largest_tan_B : Prop :=
  ∃ tanB : ℝ, tanB = 4 / 3 ∧ (∀ t : ℝ, t ≤ tanB)

theorem max_tan_B (h : triangle_ABC) (hAB : AB = 25) (hBC : BC = 15) : largest_tan_B :=
  sorry

end max_tan_B_l697_697870


namespace min_value_f_l697_697551

-- Function definition
def f (x : ℝ) : ℝ := x + 9 / (x - 2)

-- Lean theorem statement to prove the minimum value of f(x) for x > 2 is 8
theorem min_value_f : ∀ x > 2, f(x) ≥ 8 :=
by
  sorry  -- Proof goes here

end min_value_f_l697_697551


namespace find_special_three_digit_numbers_l697_697296

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l697_697296


namespace smallest_integer_solution_l697_697584

theorem smallest_integer_solution :
  ∃ y : ℤ, (5 / 8 < (y - 3) / 19) ∧ ∀ z : ℤ, (5 / 8 < (z - 3) / 19) → y ≤ z :=
sorry

end smallest_integer_solution_l697_697584


namespace area_FDBG_is_70_l697_697054

noncomputable def area_of_quadrilateral_FDBG
  (A B C D E F G : Point)
  (AB AC : ℝ)
  (area_ABC : ℝ)
  (midpoint_D : midpoint A B D)
  (midpoint_E : midpoint A C E)
  (bisector_intersection_F : angle_bisector_intersection A B C D E F)
  (bisector_intersection_G : angle_bisector_intersection A B C B C G) :
  ℝ :=
sorry

theorem area_FDBG_is_70
  {A B C D E F G : Point}
  (h1 : AB = 40)
  (h2 : AC = 20)
  (h3 : area_ABC = 160)
  (h4 : midpoint A B D)
  (h5 : midpoint A C E)
  (h6 : angle_bisector_intersection A B C D E F)
  (h7 : angle_bisector_intersection A B C B C G) :
  area_of_quadrilateral_FDBG A B C D E F G AB AC area_ABC midpoint_D midpoint_E bisector_intersection_F bisector_intersection_G = 70 :=
sorry

end area_FDBG_is_70_l697_697054


namespace maximum_rubles_received_max_payment_possible_l697_697935

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l697_697935


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697230

theorem height_of_cylinder_inscribed_in_hemisphere :
  ∀ (r_cylinder r_hemisphere : ℝ),
  r_cylinder = 3 →
  r_hemisphere = 8 →
  (∃ h : ℝ, h = real.sqrt (r_hemisphere^2 - r_cylinder^2)) :=
begin
  intros r_cylinder r_hemisphere h1 h2,
  use real.sqrt (r_hemisphere^2 - r_cylinder^2),
  rw [h1, h2],
  simp,
  sorry
end

end height_of_cylinder_inscribed_in_hemisphere_l697_697230


namespace complex_fraction_l697_697799

def imaginary_unit : ℂ := complex.I

theorem complex_fraction : 
  (3 - 4 * imaginary_unit) * (1 + imaginary_unit) ^ 3 / (4 + 3 * imaginary_unit) = (2 + 2 * imaginary_unit) :=
by
  sorry

end complex_fraction_l697_697799


namespace find_solutions_to_z6_eq_neg8_l697_697328

noncomputable def solution_set : Set ℂ :=
  {z | ∃ x y : ℝ, z = x + y * I ∧ 
  (x^6 - 15 * x^4 * y^2 + 15 * x^2 * y^4 - y^6 = -8) ∧
  (6 * x^5 * y - 20 * x^3 * y^3 + 6 * x * y^5 = 0)}

theorem find_solutions_to_z6_eq_neg8 :
  solution_set = 
  { 
    complex.sqrt 2 + complex.i * complex.sqrt 2, 
    - complex.sqrt 2 + complex.i * complex.sqrt 2, 
    - complex.sqrt 2 - complex.i * complex.sqrt 2, 
    complex.sqrt 2 - complex.i * complex.sqrt 2 
  } :=
sorry

end find_solutions_to_z6_eq_neg8_l697_697328


namespace number_of_divisors_of_n_squared_l697_697626

-- Define the conditions
def has_four_divisors (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p * q

def num_divisors (k : ℕ) : ℕ :=
  let f := k.factors in
  f.foldl (λ acc x, acc * (x.2 + 1)) 1

-- Create the proof problem
theorem number_of_divisors_of_n_squared (n : ℕ) (h : has_four_divisors n) :
  num_divisors (n * n) = 9 :=
sorry

end number_of_divisors_of_n_squared_l697_697626


namespace max_independent_set_l697_697866

/-- 
In the morning, there are 100 students who form 50 groups each with 2 students.
In the afternoon, the students form another 50 groups each with 2 students.
Groups can be arranged differently between morning and afternoon.
Prove that the maximum number of students such that no two of them study together either in the 
morning or in the afternoon is 50.
-/
theorem max_independent_set {students : Type} (morning_groups afternoon_groups : list (students × students)) :
  (∀ g ∈ morning_groups, ∃ a b : students, g = (a, b) ∧ a ≠ b) → 
  (∀ g ∈ afternoon_groups, ∃ a b : students, g = (a, b) ∧ a ≠ b) →
  (list.length morning_groups = 50) →
  (list.length afternoon_groups = 50) →
  ∃ (S : finset students), S.card ≤ 50 ∧
  (∀ a b, a ∈ S → b ∈ S → (a, b) ∉ morning_groups ∧ (a, b) ∉ afternoon_groups) :=
by sorry

end max_independent_set_l697_697866


namespace total_drawings_first_five_pages_l697_697877

theorem total_drawings_first_five_pages : 
  ∀ (drawings_per_page : ℕ → ℕ) (n : ℕ),
  (drawings_per_page 1 = 5) →
  (∀ k, drawings_per_page (k + 1) = drawings_per_page k + 5) →
  (∑ k in finset.range 5, drawings_per_page (k + 1)) = 75 :=
by
  intros drawings_per_page n h1 h2
  sorry

end total_drawings_first_five_pages_l697_697877


namespace maximum_rubles_received_l697_697948

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l697_697948


namespace multiplication_preserves_F_l697_697892

/-- Definition of the set F(n) for polynomials P(x) with specified coefficient relationships --/
def F (n : ℕ) : set (polynomial ℝ) := 
  { P | ∃ (a : fin (n+1) → ℝ), 
      (∀ i, 0 ≤ a i) ∧ 
      (a 0 = a n) ∧ 
      (∀ i, a i = a (n - i)) ∧
      (P = ∑ i in finset.range (n + 1), polynomial.C (a i) * polynomial.X ^ i) }

theorem multiplication_preserves_F {m n : ℕ} {f g : polynomial ℝ} 
  (hf : f ∈ F m) (hg : g ∈ F n) : 
  f * g ∈ F (m + n) := 
sorry

end multiplication_preserves_F_l697_697892


namespace sixth_largest_divisor_correct_l697_697168

noncomputable def sixth_largest_divisor_of_4056600000 : ℕ :=
  50707500

theorem sixth_largest_divisor_correct : sixth_largest_divisor_of_4056600000 = 50707500 :=
sorry

end sixth_largest_divisor_correct_l697_697168


namespace cos_AHB_correct_l697_697079

-- Define the triangle vertices and orthocenter
variables {A B C H : Type}
-- Define the vector relationships
variables {HA HB HC : vector_space}
-- Condition given in the problem
axiom orthocenter_condition : 3 * HA + 4 * HB + 5 * HC = 0

-- We are to find the value of cos(∠AHB)
noncomputable def cos_AHB : real :=
  - (1 / sqrt (3 + 2 * sqrt 2))

theorem cos_AHB_correct : cos_AHB = - (sqrt 6 / 6) :=
  sorry

end cos_AHB_correct_l697_697079


namespace share_of_C_is_11647_l697_697180

constant real : Type.{0}
noncomputable instance : DecidableEq real := classical.decEq real
noncomputable instance : LinearOrderedField real := {
  ..Real.field,
  ..Real.linearOrderedCommRing
}

variable (B C A : real) -- Investment amounts
variable H1 : A = 3 * B -- A invests 3 times as much as B
variable H2 : A = (2 / 3) * C -- A invests 2/3 of what C invests
variable profit : real := 22000 -- Total profit

def total_investment := A + B + C 

def C_share := (C / total_investment) * profit

theorem share_of_C_is_11647.06 : C_share = 11647.06 :=
by sorry

end share_of_C_is_11647_l697_697180


namespace am_gm_inequality_am_gm_inequality_eq_l697_697082

theorem am_gm_inequality (x : ℝ) (hx : 0 ≤ x) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 :=
begin
  sorry
end

theorem am_gm_inequality_eq (x : ℝ) (hx : 0 ≤ x) : 1 + x^2 + x^6 + x^8 = 4 * x^4 ↔ x = 1 :=
begin
  sorry
end

end am_gm_inequality_am_gm_inequality_eq_l697_697082


namespace august_answers_total_l697_697905

theorem august_answers_total 
  (x : ℕ) (y : ℕ) (z : ℕ)
  (h1 : x = 600)
  (h2 : z = 1400 - y) :
  x + (2 * x - y) + (3 * x - z) + (x + (2 * x - y) + (3 * x - z)) / 3 = 2933.33 :=
by
  sorry

end august_answers_total_l697_697905


namespace estimate_white_balls_l697_697860

-- Given conditions
variable (total_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ)

-- Assume the total number of balls is 10 and the probability of drawing a white ball is 0.3
axiom bag_condition : total_balls = 10
axiom prob_condition : prob_white = 0.3

-- The theorem to prove
theorem estimate_white_balls (h : prob_white = white_balls / total_balls) : white_balls = 3 :=
by
  rw [bag_condition, prob_condition] at h
  linarith

end estimate_white_balls_l697_697860


namespace find_S25_l697_697792

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a 1 - a 0
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

-- Condition that given S_{15} - S_{10} = 1
axiom sum_difference : S 15 - S 10 = 1

-- Theorem we need to prove
theorem find_S25 (h_arith : is_arithmetic_sequence a) (h_sum : sum_of_first_n_terms a S) : S 25 = 5 :=
by
-- Placeholder for the actual proof
sorry

end find_S25_l697_697792


namespace least_positive_angle_l697_697320

-- Let’s define the constants involved
def angle := ℝ -- using real numbers for angles
noncomputable def cos_10 : angle := Real.cos (10 * Real.pi / 180)
noncomputable def sin_20 : angle := Real.sin (20 * Real.pi / 180)
noncomputable def sin_theta (theta : angle) : angle := Real.sin (theta * Real.pi / 180)

theorem least_positive_angle (theta : angle) :
  cos_10 = sin_20 + sin_theta theta → theta = 40 := 
sorry

end least_positive_angle_l697_697320


namespace Petya_rubles_maximum_l697_697957

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l697_697957


namespace Petya_rubles_maximum_l697_697959

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l697_697959


namespace solve_for_b_l697_697026

-- Define the condition for a complex number to be pure imaginary
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Define the given complex number expression
def complex_expr (b : ℝ) : ℂ := complex.I * (2 + complex.I * b)

-- State the problem: if the given complex number expression is pure imaginary, then b = 0
theorem solve_for_b (b : ℝ) (h : is_pure_imaginary (complex_expr b)) : b = 0 :=
by
  sorry

end solve_for_b_l697_697026


namespace find_solutions_to_z6_eq_neg8_l697_697329

noncomputable def solution_set : Set ℂ :=
  {z | ∃ x y : ℝ, z = x + y * I ∧ 
  (x^6 - 15 * x^4 * y^2 + 15 * x^2 * y^4 - y^6 = -8) ∧
  (6 * x^5 * y - 20 * x^3 * y^3 + 6 * x * y^5 = 0)}

theorem find_solutions_to_z6_eq_neg8 :
  solution_set = 
  { 
    complex.sqrt 2 + complex.i * complex.sqrt 2, 
    - complex.sqrt 2 + complex.i * complex.sqrt 2, 
    - complex.sqrt 2 - complex.i * complex.sqrt 2, 
    complex.sqrt 2 - complex.i * complex.sqrt 2 
  } :=
sorry

end find_solutions_to_z6_eq_neg8_l697_697329


namespace sum_sequence_eq_24_l697_697232

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 0 := 2
| 1 := 3
| (n + 2) := (1/2) * (sequence (n + 1)) + (1/3) * (sequence n)

theorem sum_sequence_eq_24 :
  (∑ n, sequence n) = 24 := 
sorry

end sum_sequence_eq_24_l697_697232


namespace solve_problem_l697_697684

noncomputable def problem_expression : ℝ :=
  4^(1/2) + Real.log (3^2) / Real.log 3

theorem solve_problem : problem_expression = 4 := by
  sorry

end solve_problem_l697_697684


namespace giants_games_won_so_far_l697_697120

-- Define the conditions
def total_games_played := 20
def games_left := 10
def games_needed_to_win := 8
def win_ratio := 2 / 3

-- Define the question and the target number of games won
def total_games := total_games_played + games_left
def required_wins := (win_ratio * total_games : ℚ).toNat  -- converting rational to integer
def already_won_games := required_wins - games_needed_to_win

-- Lean 4 statement proving the required number of wins
theorem giants_games_won_so_far : already_won_games = 12 :=
by
  sorry

end giants_games_won_so_far_l697_697120


namespace sum_first_six_arithmetic_terms_l697_697379

variable {α : Type*} [LinearOrderedField α]

def arithmeticSeq (a₁ d : α) : ℕ → α
| 0     := a₁
| (n+1) := a₁ + n * d

def sumOfArithmeticSeq (a₁ d : α) : ℕ → α
| 0     := 0
| (n+1) := (n + 1) / 2 * (2 * a₁ + n * d)

-- conditions
variables (a₁ d : α)
#reduce (arithmeticSeq a₁ d 1 = 4)
#reduce (arithmeticSeq a₁ d 3 = 2)

def yourProblem (a₁ d : α) : Prop :=
  ∀ a₁ d,
  arithmeticSeq a₁ d 1 = 4 →
  arithmeticSeq a₁ d 3 = 2 →
  sumOfArithmeticSeq a₁ d 5 = 15

-- theorem statement
theorem sum_first_six_arithmetic_terms :
  yourProblem (5 : α) (-1 : α) :=
begin
  intros a₁ d h₁ h₂,
  sorry
end

end sum_first_six_arithmetic_terms_l697_697379


namespace length_AB_l697_697094

theorem length_AB
  (XY : ℝ)
  (O X Y Z A B : Point)
  (O_center : Center O)
  (O_circle : Circle O [X, Y, Z])
  (segment_XY : Segment XY [A, B])
  (len_OA : dist O A = 5)
  (len_AZ : dist A Z = 5)
  (len_ZB : dist Z B = 5)
  (len_BO : dist B O = 5) :
  dist A B = 2 * sqrt 13 := 
sorry

end length_AB_l697_697094


namespace solution_set_l697_697683

-- Let u be the condition for the equation

theorem solution_set (x : ℝ) : (x ≠ 0) ∧ (x ≠ 1) ∧ (x^{Real.log x / Real.log 10} = x^5 / 1000) → (x = 10 ∨ x = 1000)  :=
by
  sorry

end solution_set_l697_697683


namespace trajectory_and_line_eq_l697_697820

-- Definitions of points and their positions
def F1 : ℝ × ℝ := (-Real.sqrt 2, 0)
def F2 : ℝ × ℝ := (Real.sqrt 2, 0)

-- Definition that P satisfies the given condition |PF2| - |PF1| = 2
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  abs ((P.1 - F2.1)^2 + (P.2 - F2.2)^2).sqrt - ((P.1 - F1.1)^2 + (P.2 - F1.2)^2).sqrt = 2

-- Definition of the point through which the line passes
def point_line_passes : ℝ × ℝ := (0, -1)

-- Definition of the distance between points A and B
def distance_AB (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The problem definition: curve equation and line equation
theorem trajectory_and_line_eq :
  (∀ P : ℝ × ℝ, satisfies_condition P ↔ P.1 ^ 2 - P.2 ^ 2 = 1 ∧ P.1 < 0) ∧
  (∀ A B : ℝ × ℝ, satisfies_condition A ∧ satisfies_condition B ∧ distance_AB A B = 6 * Real.sqrt 3 →
    ∃ k : ℝ, k = -Real.sqrt 5 / 2 ∧ ∀ x y : ℝ, y = k * x - 1 → y + (Real.sqrt 5 / 2) * x + 1 = 0) :=
by sorry

end trajectory_and_line_eq_l697_697820


namespace angle_C_in_triangle_l697_697852

noncomputable def cos_rule_angle (a b c : ℝ) : ℝ :=
  real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

theorem angle_C_in_triangle (a b c : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 13) :
  cos_rule_angle a b c = 2 * real.pi / 3 := by
  sorry

end angle_C_in_triangle_l697_697852


namespace max_fills_time_l697_697909

theorem max_fills_time
  (rate_max : ℕ) (rate_zach : ℕ) (time_zach : ℕ) (balloons_popped : ℕ) (total_balloons : ℕ)
  (h_rate_max : rate_max = 2) (h_rate_zach : rate_zach = 3) (h_time_zach : time_zach = 40)
  (h_balloons_popped : balloons_popped = 10) (h_total_balloons : total_balloons = 170) :
  ∃ t : ℕ, t = 30 :=
by
  let total_balloons_filled := total_balloons + balloons_popped
  have h : 2 * t + 3 * 40 = total_balloons_filled,
    sorry
  have h_total_filled : total_balloons_filled = 180,
    rw [h_total_balloons, h_balloons_popped],
    sorry
  have h_eqn : 2 * t + 120 = 180,
    sorry
  have ht : t = 30,
    linarith,
    use 30,
    exact ht,
  sorry

end max_fills_time_l697_697909


namespace object_reaches_max_height_at_three_l697_697650

theorem object_reaches_max_height_at_three :
  ∀ (h : ℝ) (t : ℝ), h = -15 * (t - 3)^2 + 150 → t = 3 :=
by
  sorry

end object_reaches_max_height_at_three_l697_697650


namespace seokgi_milk_amount_l697_697515

theorem seokgi_milk_amount :
  ∀ (J : ℕ), (J + (J + 200) = 2100) → (J + 200 = 1150) :=
by
  intros J h
  have : 2 * J + 200 = 2100 := by
    rw [←add_assoc] at h
    exact h
  have : 2 * J = 1900 := by
    linarith
  have : J = 950 := by
    linarith
  rw this
  linarith

end seokgi_milk_amount_l697_697515


namespace cylinder_height_in_hemisphere_l697_697220

noncomputable def height_of_cylinder (r_hemisphere r_cylinder : ℝ) : ℝ :=
  real.sqrt (r_hemisphere^2 - r_cylinder^2)

theorem cylinder_height_in_hemisphere :
  let r_hemisphere := 8
  let r_cylinder := 3
  height_of_cylinder r_hemisphere r_cylinder = real.sqrt 55 :=
by
  sorry

end cylinder_height_in_hemisphere_l697_697220


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697223

open Real

theorem height_of_cylinder_inscribed_in_hemisphere
  (r_cylinder r_hemisphere : ℝ)
  (h_radius_cylinder : r_cylinder = 3)
  (h_radius_hemisphere : r_hemisphere = 8) :
  ∃ h_cylinder : ℝ, h_cylinder = sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) :=
by
  use sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  sorry

#check height_of_cylinder_inscribed_in_hemisphere

end height_of_cylinder_inscribed_in_hemisphere_l697_697223


namespace express_as_sum_of_cubes_l697_697286

variables {a b : ℝ}

theorem express_as_sum_of_cubes (a b : ℝ) : 
  2 * a * (a^2 + 3 * b^2) = (a + b)^3 + (a - b)^3 :=
by sorry

end express_as_sum_of_cubes_l697_697286


namespace triangles_in_extended_figure_l697_697010

theorem triangles_in_extended_figure : 
  ∀ (row1_tri : ℕ) (row2_tri : ℕ) (row3_tri : ℕ) (row4_tri : ℕ) 
  (row1_2_med_tri : ℕ) (row2_3_med_tri : ℕ) (row3_4_med_tri : ℕ) 
  (large_tri : ℕ), 
  row1_tri = 6 →
  row2_tri = 5 →
  row3_tri = 4 →
  row4_tri = 3 →
  row1_2_med_tri = 5 →
  row2_3_med_tri = 2 →
  row3_4_med_tri = 1 →
  large_tri = 1 →
  row1_tri + row2_tri + row3_tri + row4_tri
  + row1_2_med_tri + row2_3_med_tri + row3_4_med_tri
  + large_tri = 27 :=
by
  intro row1_tri row2_tri row3_tri row4_tri
  intro row1_2_med_tri row2_3_med_tri row3_4_med_tri
  intro large_tri
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end triangles_in_extended_figure_l697_697010


namespace find_three_digit_numbers_l697_697316

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l697_697316


namespace smallest_non_palindrome_power_of_13_l697_697723

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l697_697723


namespace x_div_11p_is_integer_l697_697418

theorem x_div_11p_is_integer (x p : ℕ) (h1 : x > 0) (h2 : Prime p) (h3 : x = 66) : ∃ k : ℤ, x / (11 * p) = k := by
  sorry

end x_div_11p_is_integer_l697_697418


namespace imaginary_part_of_i_mul_i_sub_2_l697_697997

open Complex

theorem imaginary_part_of_i_mul_i_sub_2 : (∀ i : ℂ, i = Complex.I → (Complex.imag (Complex.I * (Complex.I - 2))) = -2) :=
by
  intro i h
  sorry

end imaginary_part_of_i_mul_i_sub_2_l697_697997


namespace maximum_rubles_received_l697_697951

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l697_697951


namespace equal_ratios_l697_697090

-- Define points and their relations on a triangle
variables {A B C A1 B1 C1 A2 B2 C2 : Type*}
variables [add_comm_group A] [add_comm_group B] [add_comm_group C]
variables [module ℝ A] [module ℝ B] [module ℝ C]

-- Assume the points A1, B1, C1 lie on BC, CA, AB respectively
-- Define the intersection points A2, B2, C2 in terms of A1, B1, C1 and sides of the triangle.
noncomputable def intersection_points_condition (A1 A2 B1 B2 C1 C2 : Type*) : Prop :=
  true

-- Express the given vector sum condition
axiom given_vector_sum : (A → ℝ) → (B → ℝ) → (C → ℝ) → Prop
axiom vector_sum_zero : ∀ (f : A → ℝ) (g : B → ℝ) (h : C → ℝ),
  given_vector_sum f g h → f AA_2 + g BB_2 + h CC_2 = 0

-- Define ratios to prove
def ratios_equal (f : A → ℝ) (g : B → ℝ) (h : C → ℝ) : Prop :=
  f AB_1 / f B_1C = g CA_1 / g A_1B ∧ g CA_1 / g A_1B = h BC_1 / h C_1A

-- The theorem statement
theorem equal_ratios {A1 B1 C1 A2 B2 C2 : Type*}
(pts_cond : intersection_points_condition A1 A2 B1 B2 C1 C2) 
(conds : given_vector_sum AA_2 BB_2 CC_2) : 
ratios_equal AB_1 B_1C CA_1 A_1B BC_1 C_1A :=
by {
  sorry
}

end equal_ratios_l697_697090


namespace part1_part2_l697_697395

open Set 

def setA : Set ℝ := {x : ℝ | x^2 - 2 * x - 3 ≥ 0}
def setB (m : ℝ) : Set ℝ := {x : ℝ | m - 2 ≤ x ∧ x ≤ m + 2}

theorem part1 : (Z ∩ (setAᶜ)) = ({0, 1, 2} : Set ℝ) := 
by 
  sorry

theorem part2 (m : ℝ) (h : setB m ⊆ setA) : m ∈ (-∞, -3] ∪ [5, ∞) := 
by 
  sorry

end part1_part2_l697_697395


namespace solution_set_of_inequality_l697_697538

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_f1_zero : f 1 = 0) : 
  { x | f x > 0 } = { x | x < -1 ∨ 1 < x } := 
by
  sorry

end solution_set_of_inequality_l697_697538


namespace num_divisors_square_l697_697620

theorem num_divisors_square (n : ℕ) (h₁ : ∃ p q : ℕ, prime p ∧ prime q ∧ p ≠ q ∧ n = p * q) :
  num_divisors (n^2) = 9 :=
by
  sorry

end num_divisors_square_l697_697620


namespace solve_z6_eq_neg8_l697_697331

noncomputable def solutions : Set ℂ := 
  {z | ∃ x y : ℝ, z = x + y * I ∧ (x + y * I)^6 = -8}

theorem solve_z6_eq_neg8 :
  solutions = {z | z = 1 + I ∨ z = -1 - I ∨ z = 1 - I ∨ z = -1 + I ∨ z = complex.exp ((2 * π * I + log (-8))/6) ∨ z = complex.exp (-(2 * π * I + log (-8))/6)} :=
by
  sorry

end solve_z6_eq_neg8_l697_697331


namespace domain_of_g_l697_697376

noncomputable def f : ℝ → ℝ := sorry  -- Placeholder for the function f

theorem domain_of_g :
  {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)} = -- Expected domain of g(x)
  { x : ℝ |
    (0 ≤ x ∧ x ≤ 6) ∧ -- Domain of f is 0 ≤ x ≤ 6
    2 * x ≤ 6 ∧ -- For g(x) to be in the domain of f(2x)
    0 ≤ 2 * x ∧ -- Ensures 2x fits within the domain 0 < 2x < 6
    x ≠ 2 } -- x cannot be 2
:= sorry

end domain_of_g_l697_697376


namespace p_sufficient_not_necessary_for_q_l697_697796

variable {x : ℝ}

def p : Prop := -2 < x ∧ x < 0
def q : Prop := |x| < 2

theorem p_sufficient_not_necessary_for_q : (p → q) ∧ ¬ (q → p) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l697_697796


namespace arithmetic_sequence_ratio_proof_l697_697344

variable {a_n b_n : ℕ → ℕ}
variable {S_n T_n : ℕ → ℕ}

-- Definition for arithmetic sequences' sum
def sum_arithmetic_sequence (s : ℕ → ℕ) : ℕ → ℕ
| 0       := 0
| (n + 1) := s (n + 1) + sum_arithmetic_sequence s n

-- Hypothesis for the given sum relationship between two series
axiom given_ratio (n : ℕ) : (S_n n).toRat / (T_n n).toRat = (7 * n + 1).toRat / (4 * n + 27).toRat

-- Lean statement
theorem arithmetic_sequence_ratio_proof :
  (∑ i in Finset.range 11, a_n (i + 1)) / (∑ i in Finset.range 11, b_n (i + 1)) = (4 : ℚ) / (3 : ℚ) :=
by
  sorry

end arithmetic_sequence_ratio_proof_l697_697344


namespace estimate_correctness_l697_697426

noncomputable def total_species_estimate (A B C : ℕ) : Prop :=
  A = 2400 ∧ B = 1440 ∧ C = 3600

theorem estimate_correctness (A B C taggedA taggedB taggedC caught : ℕ) 
  (h1 : taggedA = 40) 
  (h2 : taggedB = 40) 
  (h3 : taggedC = 40)
  (h4 : caught = 180)
  (h5 : 3 * A = taggedA * caught) 
  (h6 : 5 * B = taggedB * caught) 
  (h7 : 2 * C = taggedC * caught) 
  : total_species_estimate A B C := 
by
  sorry

end estimate_correctness_l697_697426


namespace second_customer_regular_hours_l697_697602
noncomputable theory

-- Define variables representing the hourly rates and the equations given in the problem.
variables (x y r : ℝ)
-- Condition 1: One customer spent 2 hours in premium areas and 9 regular hours, charged $28.
def eq1 : Prop := 2 * y + 9 * x = 28
-- Condition 2: Another customer spent 3 hours in premium areas and spent r regular hours, charged $27.
def eq2 : Prop := 3 * y + r * x = 27

-- Statement: Prove that given these conditions, the regular hours spent by the second customer is 3.
theorem second_customer_regular_hours (h1: eq1 x y) (h2: eq2 x y r) : r = 3 :=
sorry

end second_customer_regular_hours_l697_697602


namespace wax_for_suv_l697_697057

/--
It takes 3 ounces of wax to detail Kellan's car
Kellan bought an 11-ounce bottle of vehicle wax
Kellan spilled 2 ounces before using the wax
Kellan has 2 ounces left after waxing his car and SUV

Prove that the number of ounces of wax used to detail the Kellan's SUV is 4 ounces.
-/
theorem wax_for_suv (wax_for_car wax_initial wax_spilled wax_remaining wax_left : ℕ)
  (h_car: wax_for_car = 3)
  (h_initial: wax_initial = 11)
  (h_spilled: wax_spilled = 2)
  (h_after: wax_left = 2)
  (h_total: wax_remaining = wax_initial - wax_spilled - wax_left)
  : wax_remaining - wax_for_car = 4 := 
by 
  -- expanding out the known values to conclude wax_for_suv must be 4
  rw [h_car, h_initial, h_spilled, h_after]
  sorry

end wax_for_suv_l697_697057


namespace ratio_of_areas_of_shaded_and_white_region_l697_697583

theorem ratio_of_areas_of_shaded_and_white_region
  (all_squares_have_vertices_in_middle: ∀ (n : ℕ), n ≠ 0 → (square_vertices_positioned_mid : Prop)) :
  ∃ (ratio : ℚ), ratio = 5 / 3 :=
by
  sorry

end ratio_of_areas_of_shaded_and_white_region_l697_697583


namespace equilateral_triangle_of_radii_l697_697968

theorem equilateral_triangle_of_radii (ABC : Triangle) (r R : ℝ) (h₁ : ABC.IncircleRadius = r) (h₂ : ABC.CircumcircleRadius = R) (h₃ : R = 2 * r) :
  ABC.IsEquilateral :=
sorry

end equilateral_triangle_of_radii_l697_697968


namespace smallest_non_palindrome_power_of_13_l697_697760

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l697_697760


namespace smallest_non_palindromic_power_of_13_l697_697745

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l697_697745


namespace find_three_digit_numbers_l697_697315

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l697_697315


namespace charlie_received_495_l697_697597

theorem charlie_received_495 : 
  ∃ (A B C x : ℕ), 
    A + B + C = 1105 ∧ 
    A - 10 = 11 * x ∧ 
    B - 20 = 18 * x ∧ 
    C - 15 = 24 * x ∧ 
    C = 495 := 
by
  sorry

end charlie_received_495_l697_697597


namespace car_catch_truck_l697_697238

theorem car_catch_truck (truck_speed car_speed : ℕ) (time_head_start : ℕ) (t : ℕ)
  (h1 : truck_speed = 45) (h2 : car_speed = 60) (h3 : time_head_start = 1) :
  45 * t + 45 = 60 * t → t = 3 := by
  intro h
  sorry

end car_catch_truck_l697_697238


namespace minimum_varphi_l697_697974

theorem minimum_varphi (ϕ : ℝ) (hϕ : ϕ > 0) :
  let f := λ x, 2 * sin (2 * x + (π / 4))
  let g := λ x, 2 * sin (4 * x - 2 * ϕ + (π / 4))
  (∀ x, g x = -g (-x)) → ϕ = π / 8 :=
by
  sorry

end minimum_varphi_l697_697974


namespace y_increases_when_x_decreases_l697_697826

theorem y_increases_when_x_decreases (α x : ℝ) (hα : 0 < Real.sin α ∧ Real.sin α < 1) :
  let k := Real.sin α - 1 in ∀ x1 x2 : ℝ, x1 < x2 → ((k * x1 - 6) > (k * x2 - 6)) :=
by
  sorry

end y_increases_when_x_decreases_l697_697826


namespace problem_statement_l697_697291

/-- A predicate that checks if the numbers from 1 to 2n can be split into two groups 
    such that the sum of the product of the elements of each group is divisible by 2n - 1. -/
def valid_split (n : ℕ) : Prop :=
  ∃ (a b : Finset ℕ), 
  a ∪ b = Finset.range (2 * n) ∧
  a ∩ b = ∅ ∧
  (2 * n) ∣ (a.prod id + b.prod id - 1)

theorem problem_statement : 
  ∀ n : ℕ, n > 0 → valid_split n ↔ (n = 1 ∨ ∃ a : ℕ, n = 2^a ∧ a ≥ 1) :=
by
  sorry

end problem_statement_l697_697291


namespace min_rectangles_l697_697156

def corners (fig: Type) : Type := 
  {Type1 : fin 12 // True} -- 12 Type 1 corners
  ⊕ 
  {Type2 : fin 12 // True} -- 12 Type 2 corners

def triplets (c: corners) : Prop := 
  ∀ (t : Type2), ∃ (a b : Type2), t ≠ a ∧ t ≠ b ∧ a ≠ b

theorem min_rectangles (fig : Type) (c : corners fig) (t : triplets c) : 
  let corner_covering := 12 in
  let rect_needed := 12 in
  rect_needed = 12 := 
sorry

end min_rectangles_l697_697156


namespace number_of_pupils_in_class_wrong_entry_l697_697183

theorem number_of_pupils_in_class_wrong_entry (n : ℕ) 
    (h1 : 83 - 70 = 13)
    (h2 : ∃! d, d = 13 → (∀ k, 83 - k = d))
    (h3 : (average_marks_increase_by_half : ℕ)) :
  13 = 1 / 2 * n → n = 26 := by
  -- Conditions in place, setup our statement
  sorry

end number_of_pupils_in_class_wrong_entry_l697_697183


namespace equal_angles_l697_697644

-- Define the points and their relationships as per the problem
variables (A B C D P Q M N : Type) [affine_space ℝ (affine_space.point ℝ)] 
variables [has_dist ℝ (affine_space.point ℝ)] [inner_product_space ℝ (affine_space.point ℝ)]
variables (midpoint : Π (a b : affine_space.point ℝ), affine_space.point ℝ)

-- Define the conditions
def is_parallelogram (A B C D : affine_space.point ℝ) : Prop :=
  ∃ P : affine_space.point ℝ, affine_space.convex_hull (A::B::C::D::[]) = affine_space.convex_hull (P::B::C::D::[]) ∧ 
                               affine_space.convex_hull (A::P::C::D::[]) = affine_space.convex_hull (A::B::P::D::[])

def midpoint_condition_1 (M P C : affine_space.point ℝ) : Prop :=
  dist M P = dist M C

def midpoint_condition_2 (N P A : affine_space.point ℝ) : Prop :=
  dist N P = dist N A

def midpoint_of_PB (P B : affine_space.point ℝ) : affine_space.point ℝ :=
  midpoint P B

-- Theorem to be proven
theorem equal_angles (A B C D P Q M N : affine_space.point ℝ) 
  (h1 : is_parallelogram A B C D) 
  (h2 : M = midpoint A D) 
  (h3 : N = midpoint C D) 
  (h4 : midpoint_condition_1 M P C)
  (h5 : midpoint_condition_2 N P A)
  (h6 : Q = midpoint P B) :
  ∠PAQ = ∠PCQ :=
by 
  sorry

end equal_angles_l697_697644


namespace distinct_prime_factors_sum_of_divisors_900_l697_697006

theorem distinct_prime_factors_sum_of_divisors_900 :
  ∃ n : ℕ, n = 3 ∧
  ∃ σ : ℕ → ℕ, 
    σ 900 = (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ∧
    ∀ p : ℕ, p.prime → ∀ d : ℕ, d ∣ σ 900 → ∃ k : ℕ, (p^k).count_factors = n := sorry

end distinct_prime_factors_sum_of_divisors_900_l697_697006


namespace max_rubles_l697_697961

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l697_697961


namespace profit_margin_comparison_l697_697535

theorem profit_margin_comparison
    (cost_price_A : ℝ) (selling_price_A : ℝ)
    (cost_price_B : ℝ) (selling_price_B : ℝ)
    (h1 : cost_price_A = 1600)
    (h2 : selling_price_A = 0.9 * 2000)
    (h3 : cost_price_B = 320)
    (h4 : selling_price_B = 0.8 * 460) :
    ((selling_price_B - cost_price_B) / cost_price_B) > ((selling_price_A - cost_price_A) / cost_price_A) := 
by
    sorry

end profit_margin_comparison_l697_697535


namespace triangle_AD_length_l697_697456

theorem triangle_AD_length
  (A B C D : Type)
  (h : ∀ M, euclidean_geometry.point M [M = A, M = B, M = C])
  (AB AC : ℝ)
  (p : A ≠ B)
  (q : A ≠ C)
  (r : B ≠ C)
  (AB_eq : AB = 13)
  (AC_eq : AC = 20)
  (perpendicular : euclidean_geometry.perpendicular A D B C)
  (BD CD : ℝ)
  (ratio : BD / CD = 3 / 4) :
  let AD := sqrt (128) in AD = 8 * sqrt 2 :=
by
  sorry

end triangle_AD_length_l697_697456


namespace solve_complex_equation_l697_697335

-- Definitions for real numbers x, y and complex number z.
variables (x y : ℝ) (z : ℂ)

-- The condition where z is a complex number defined by real parts x and y.
def z_def : ℂ := x + y * complex.I

-- The main statement translating the problem into Lean
theorem solve_complex_equation : 
  (∃ (x y : ℝ) (z : ℂ), z = x + y * complex.I ∧ z^6 = -8 ∧
   (z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 5 * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 7 * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 11 * real.pi / 6) ∨ 
    z = complex.I * (2 : ℝ)^(1 / 3) ∨ 
    z = -complex.I * (2 : ℝ)^(1 / 3))) := 
sorry

end solve_complex_equation_l697_697335


namespace length_of_CD_l697_697357

theorem length_of_CD :
  ∀ (A B C D : ℝ × ℝ),
  (∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ x - sqrt 3 * y + 6 = 0)) → -- line condition
  (∃ (x y : ℝ), x^2 + y^2 = 12 ∧ (x, y) ∈ l) →                   -- circle condition and intersection with line
  let ⟨A, B⟩ := 
    (classical.some (exists_pair_mem_of_ne_empty 
                      (set.ne_univ_of_nonempty_inter 
                        (by simp_rw [←set.mem_set_of_eq, set_of_eq_eq_singleton] 
                          at classical.some_spec this)))) in
  let C := (A.1, 0) in                                            -- perpendicular from A to x-axis
  let D := (B.1, 0) in                                            -- perpendicular from B to x-axis 
  abs (C.1 - D.1) = 4 := sorry

end length_of_CD_l697_697357


namespace both_not_qualified_probability_l697_697042

variable (p_a_qualified : ℝ) (p_b_qualified : ℝ)

def p_a_not_qualified : ℝ := 1 - p_a_qualified
def p_b_not_qualified : ℝ := 1 - p_b_qualified
def p_both_not_qualified : ℝ := p_a_not_qualified * p_b_not_qualified

theorem both_not_qualified_probability (h1 : p_a_qualified = 0.9) (h2 : p_b_qualified = 0.8) :
  p_both_not_qualified p_a_qualified p_b_qualified = 0.02 := by
  simp [p_both_not_qualified, p_a_not_qualified, p_b_not_qualified]
  rw [h1, h2]
  norm_num
  sorry

end both_not_qualified_probability_l697_697042


namespace smallest_non_palindrome_power_of_13_l697_697757

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l697_697757


namespace solve_z6_eq_neg8_l697_697333

noncomputable def solutions : Set ℂ := 
  {z | ∃ x y : ℝ, z = x + y * I ∧ (x + y * I)^6 = -8}

theorem solve_z6_eq_neg8 :
  solutions = {z | z = 1 + I ∨ z = -1 - I ∨ z = 1 - I ∨ z = -1 + I ∨ z = complex.exp ((2 * π * I + log (-8))/6) ∨ z = complex.exp (-(2 * π * I + log (-8))/6)} :=
by
  sorry

end solve_z6_eq_neg8_l697_697333


namespace peach_pie_slices_equal_six_l697_697656

def apple_pie_slices := 8
def total_apple_pie_slices_sold := 56
def total_peach_pie_slices_sold := 48
def total_pies_sold := 15

theorem peach_pie_slices_equal_six :
  (total_peach_pie_slices_sold.toNat / 
  (total_pies_sold - (total_apple_pie_slices_sold.toNat / apple_pie_slices.toNat).toNat).toNat) = 6 := by
  sorry

end peach_pie_slices_equal_six_l697_697656


namespace find_ellipse_equation_find_k_l697_697374

-- Definition of variables
variables {a b e x y k : ℝ}

-- Condition: Ellipse equation with a > b > 0
def ellipse_equation (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

-- Condition: Eccentricity e = √6 / 3
def eccentricity_condition (a b e : ℝ) : Prop :=
  e = sqrt 6 / 3

-- Condition: Distance from the origin to the line AB
def distance_origin_to_AB (a b : ℝ) : Prop :=
  sqrt (a^2 + b^2) = 3

-- Question (1): Prove that the ellipse has this equation
theorem find_ellipse_equation
  (ha : a > b) (hb : b > 0) (hecc : e = sqrt 6 / 3)
  (hdist : sqrt (a^2 + b^2) = 3) : ∃ x y, ellipse_equation 3 1 x y :=
sorry

-- Fixed Point E(-1, 0)
def fixed_point_E : Prop :=
  (-1, 0) = E

-- Question (2): Prove the existence of k = 7/6
theorem find_k
  (ha : a = sqrt 3) (hb: b = 1)
  (he : (-1, 0) ∈ (E : set (ℝ × ℝ)))
  (intersect_ellipse : ∃ C D : set (ℝ × ℝ),
     let k := 7 / 6
     ∧ ∀ x y, (x^2 / 3 + y^2 = 1) ∧ (y = k * x + 2))
  : ∃ k : ℝ, k = 7 / 6 :=
sorry

end find_ellipse_equation_find_k_l697_697374


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697733

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l697_697733


namespace proof_abc_gt_28_l697_697477

variable {a b c : ℝ}

def P (x : ℝ) : ℝ := a * x^3 + (b - a) * x^2 - (c + b) * x + c
def Q (x : ℝ) : ℝ := x^4 + (b - 1) * x^3 + (a - b) * x^2 - (c + a) * x + c
def roots (f : ℝ → ℝ) (xs : Set ℝ) : Prop := ∀ x ∈ xs, f x = 0

theorem proof_abc_gt_28 (a b c x0 x1 x2 : ℝ) (hx0 : P x0 = 0) (hx1 : P x1 = 0) (hx2 : P x2 = 0) (hx0Q : Q x0 = 0) (hx1Q : Q x1 = 0) (hx2Q : Q x2 = 0) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) (c_ne_zero : c ≠ 0) (b_gt_zero : b > 0) :
  abc > 28 := sorry

end proof_abc_gt_28_l697_697477


namespace first_avg_score_refers_to_matches_l697_697531

-- Definitions from the conditions
noncomputable def avg_score_10_matches := 38.9
noncomputable def avg_score_some_matches := 42
noncomputable def avg_score_last_4_matches := 34.25
noncomputable def total_matches := 10
noncomputable def last_4_matches := 4

-- Lean theorem statement
theorem first_avg_score_refers_to_matches (x : ℕ) :
  (avg_score_some_matches * x + avg_score_last_4_matches * last_4_matches = total_matches * avg_score_10_matches) →
  x = 6 :=
by
  sorry

end first_avg_score_refers_to_matches_l697_697531


namespace number_of_real_solutions_l697_697707

def f (x : ℝ) : ℝ :=
  ∑ k in finset.range 100, (2 * (k + 1)) / (x - (k + 1))

theorem number_of_real_solutions :
  (∑ k in finset.range 100, (2 * (k + 1)) / (x - (k + 1))) = x → 
  ∃ s ∈ finset.range 101, s ≠ ∅ ∧ ∀ y ∈ s, (∑ k in finset.range 100, (2 * (k + 1)) / (y - (k + 1))) = y :=
sorry

end number_of_real_solutions_l697_697707


namespace frosting_time_difference_l697_697504

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end frosting_time_difference_l697_697504


namespace trapezoid_area_l697_697607

theorem trapezoid_area (a b c h : ℝ) (ha : 2 * a = 800)
  (hb : b = 0.4 * a) (hc : c = 0.6 * a)
  (hx_eq : (c - b) / 2 = 40)
  (h_eq : a^2 = h^2 + 40^2) :
  (1/2 * (b + c) * h) = 79600 :=
begin
  have ha_400 : a = 400,
  { linarith, },

  have hb_160 : b = 160,
  { linarith [ha_400, hb], },

  have hc_240 : c = 240,
  { linarith [ha_400, hc], },

  have hx_40 : (240 - 160) / 2 = 40,
  { linarith, },

  have h_square : 400^2 = h^2 + 40^2,
  { linarith [hx_40], },

  have h_398 : h = 398 := by
  { sorry, -- We need additional real number calculations to simplify this properly },
  
  calc (1 / 2) * (b + c) * h 
      = (1 / 2) * (160 + 240) * 398 : by {
        rw [hb_160, hc_240, h_398], sorry},
      = 79600 : by sorry
end

end trapezoid_area_l697_697607


namespace sin_210_correct_l697_697146

noncomputable def sin_210_equals_minus_half : Prop :=
  sin (210 * Real.pi / 180) = -1 / 2

theorem sin_210_correct : sin_210_equals_minus_half :=
by
  sorry

end sin_210_correct_l697_697146


namespace EF_equals_FG_l697_697437

-- Definitions to represent the entities and setup
variables {O A B C D E F G : Type}
variables [circle O] [chord O A B] [chord O C D]
variables [intersect A B C D E]
variables [line_parallel EF BC]
variables [intersect_line DA EF F]
variables [tangent_line FG O G]

-- The theorem statement
theorem EF_equals_FG : EF = FG := sorry

end EF_equals_FG_l697_697437


namespace sec_tan_equation_l697_697828

theorem sec_tan_equation (x : ℝ) (h : Real.sec x + Real.tan x = 3) : Real.sec x - Real.tan x = 1 / 3 :=
sorry

end sec_tan_equation_l697_697828


namespace eq_AD_l697_697461

noncomputable def find_AD (AB AC BD CD : ℝ) (BD_CD_ratio : ℝ) :=
  let h := 8 * Real.sqrt 2 in
  h

theorem eq_AD :
  ∀ (A B C D : Type) (AB AC : ℝ), 
    AB = 13 → 
    AC = 20 → 
    BD/CD = 3/4 → 
    AD = 8 * Real.sqrt 2 :=
by
  intros
  sorry

end eq_AD_l697_697461


namespace ratio_CD_DA_l697_697464

variables (A B C D : Type) [EuclideanGeometry A B C D]
variables (angle_B : ∠ B = 120)
variables (side_AB_eq_2BC : AB = 2 * BC)
variables (perp_bisector_intersects : PerpendicularBisector AB intersects AC at D)

theorem ratio_CD_DA :
  ratio (CD) (DA) = 3 / 2 :=
sorry

end ratio_CD_DA_l697_697464


namespace exists_root_in_interval_l697_697529

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x - 2

theorem exists_root_in_interval :
  (f 1 < 0) ∧ (f 2 > 0) ∧ (∀ x > 0, ContinuousAt f x) → (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by sorry

end exists_root_in_interval_l697_697529


namespace complex_imaginary_part_eq_three_l697_697075

theorem complex_imaginary_part_eq_three (a : ℝ) (i : ℂ) (h : i^2 = -1) (h_im : (1 - i^2023) / (a * i)).im = 3 :
  a = -1 / 3 :=
by sorry

end complex_imaginary_part_eq_three_l697_697075


namespace find_k_l697_697804

theorem find_k (k : ℝ) : (1 - 1.5 * k = (k - 2.5) / 3) → k = 1 :=
by
  intro h
  sorry

end find_k_l697_697804


namespace eq_AD_l697_697460

noncomputable def find_AD (AB AC BD CD : ℝ) (BD_CD_ratio : ℝ) :=
  let h := 8 * Real.sqrt 2 in
  h

theorem eq_AD :
  ∀ (A B C D : Type) (AB AC : ℝ), 
    AB = 13 → 
    AC = 20 → 
    BD/CD = 3/4 → 
    AD = 8 * Real.sqrt 2 :=
by
  intros
  sorry

end eq_AD_l697_697460


namespace inequality_solution_l697_697110

theorem inequality_solution (x : ℝ) :
  ( (x - 1) * (x - 3) * (x - 5) / ((x - 2) * (x - 4) * (x - 6)) > 0 ) ↔
  (x ∈ set.Iio 1 ∪ set.Ioo 2 3 ∪ set.Ioo 4 5 ∪ set.Ioi 6) :=
by sorry

end inequality_solution_l697_697110


namespace no_solution_condition_l697_697806

theorem no_solution_condition (m : ℝ) : (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 :=
by
  sorry

end no_solution_condition_l697_697806


namespace convex_ngon_sides_l697_697603

theorem convex_ngon_sides (n : ℕ) (h : (n * (n - 3)) / 2 = 27) : n = 9 :=
by
  -- Proof omitted
  sorry

end convex_ngon_sides_l697_697603


namespace Petya_rubles_maximum_l697_697958

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l697_697958


namespace circle_quadrilateral_proof_l697_697768

-- Define the problem conditions and prove the required equation.
theorem circle_quadrilateral_proof (r1 r2 d : ℝ) :
  (∃ (quad : set ℝ), (∀ x ∈ quad, x ∈ set.univ) ∧ -- A quadrilateral can be inscribed in the first circle
  (∀ y ∈ quad, y ∈ set.univ)) →   -- and circumscribed around the second circle
  (1 / (r1 - d)^2 + 1 / (r1 + d)^2 = 1 / r2^2) := 
by
  sorry

end circle_quadrilateral_proof_l697_697768


namespace max_possible_a_l697_697391

theorem max_possible_a (a b : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, (λ f, f y = f x + y) (λ x, x ^ 2 + a * x + b)) →
  a ≤ 1 / 2 :=
sorry

end max_possible_a_l697_697391


namespace problem_solution_l697_697668

theorem problem_solution :
  (2:ℝ)⁻¹ - (π - 2014)⁰ + (real.cos (real.pi / 4))^2 + (real.tan (real.pi / 6)) * (real.sin (real.pi / 3)) = (1/2:ℝ) :=
by
  sorry

end problem_solution_l697_697668


namespace jacket_final_price_and_percentage_l697_697139

theorem jacket_final_price_and_percentage (SRP : ℝ) (MP : ℝ) (FSP : ℝ) (P : ℝ) :
  SRP = 120 →
  MP = 0.60 * SRP →
  FSP = 0.80 * MP →
  P = (FSP / SRP) * 100 →
  FSP = 57.6 ∧ P = 48 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- continue to automate the proof
  simp
  norm_num
  sorry


end jacket_final_price_and_percentage_l697_697139


namespace solve_z6_eq_neg8_l697_697332

noncomputable def solutions : Set ℂ := 
  {z | ∃ x y : ℝ, z = x + y * I ∧ (x + y * I)^6 = -8}

theorem solve_z6_eq_neg8 :
  solutions = {z | z = 1 + I ∨ z = -1 - I ∨ z = 1 - I ∨ z = -1 + I ∨ z = complex.exp ((2 * π * I + log (-8))/6) ∨ z = complex.exp (-(2 * π * I + log (-8))/6)} :=
by
  sorry

end solve_z6_eq_neg8_l697_697332


namespace toppings_per_pizza_l697_697002

theorem toppings_per_pizza:
  ∀ T : ℕ, 
  let large_pizza_cost := 14 in
  let topping_cost := 2 in
  let num_pizzas := 2 in
  let total_cost := 50 in
  let tip_percent := 0.25 in

  -- calculating values based on conditions
  let pizzas_cost := num_pizzas * large_pizza_cost in
  let toppings_cost := num_pizzas * T * topping_cost in
  let subtotal := pizzas_cost + toppings_cost in
  let final_cost := subtotal * (1 + tip_percent) in

  final_cost = total_cost → T = 3 := 
by 
  intros T large_pizza_cost topping_cost num_pizzas total_cost tip_percent 
  pizzas_cost toppings_cost subtotal final_cost h_final_cost_eq
  sorry

end toppings_per_pizza_l697_697002


namespace num_divisors_square_l697_697623

theorem num_divisors_square (n : ℕ) (h₁ : ∃ p q : ℕ, prime p ∧ prime q ∧ p ≠ q ∧ n = p * q) :
  num_divisors (n^2) = 9 :=
by
  sorry

end num_divisors_square_l697_697623


namespace willam_land_percentage_l697_697185

theorem willam_land_percentage (total_tax_collected willam_tax_payment : ℝ) (H1 : total_tax_collected = 4000) (H2 : willam_tax_payment = 500) :
  (willam_tax_payment / total_tax_collected) * 100 = 12.5 :=
by
  -- Introduce the given conditions
  rw [H2, H1],
  -- Solve the equation
  norm_num,
  done

end willam_land_percentage_l697_697185


namespace new_edition_geometry_book_pages_l697_697206

theorem new_edition_geometry_book_pages : 
  let P_old := 340 in
  let P_new := 2 * P_old - 230 in
  P_new = 450 := by
  let P_old := 340
  let P_new := 2 * P_old - 230
  sorry

end new_edition_geometry_book_pages_l697_697206


namespace tax_diminished_by_20_l697_697561

variable {T C x : ℝ}

-- Definitions of conditions
def original_revenue (T C : ℝ) : ℝ := T * C
def new_tax_rate (T x : ℝ) : ℝ := T - (x / 100) * T
def new_consumption (C : ℝ) : ℝ := 1.2 * C
def new_revenue (T C x : ℝ) : ℝ := new_tax_rate T x * new_consumption C

-- Proof that the tax is diminished by 20%
theorem tax_diminished_by_20 :
  0.96 * original_revenue T C = new_revenue T C 20 :=
by sorry

end tax_diminished_by_20_l697_697561


namespace eq_AD_l697_697462

noncomputable def find_AD (AB AC BD CD : ℝ) (BD_CD_ratio : ℝ) :=
  let h := 8 * Real.sqrt 2 in
  h

theorem eq_AD :
  ∀ (A B C D : Type) (AB AC : ℝ), 
    AB = 13 → 
    AC = 20 → 
    BD/CD = 3/4 → 
    AD = 8 * Real.sqrt 2 :=
by
  intros
  sorry

end eq_AD_l697_697462


namespace trigonometric_sum_of_x_l697_697337

theorem trigonometric_sum_of_x :
  (∀ x : ℝ, (0 < x ∧ x < 180) →
    (sin (2 * x * real.pi / 180))^2 + (sin (6 * x * real.pi / 180))^2 = 8 * (sin (4 * x * real.pi / 180))^2 * (sin (x * real.pi / 180))^2) →
    ∑ x in {45, 90, 135}, x = 270 :=
by
  sorry

end trigonometric_sum_of_x_l697_697337


namespace range_of_c_l697_697351

theorem range_of_c (a b c : ℝ) (h1 : 6 < a) (h2 : a < 10) (h3 : a / 2 ≤ b) (h4 : b ≤ 2 * a) (h5 : c = a + b) : 
  9 < c ∧ c < 30 :=
sorry

end range_of_c_l697_697351


namespace solve_equation_l697_697572

def f (x : ℝ) := |3 * x - 2|

theorem solve_equation 
  (x : ℝ) 
  (a : ℝ)
  (hx1 : x ≠ 3)
  (hx2 : x ≠ 0) :
  (3 * x - 2) ^ 2 = (x + a) ^ 2 ↔
  (a = -4 * x + 2) ∨ (a = 2 * x - 2) := by
  sorry

end solve_equation_l697_697572


namespace express_in_standard_form_l697_697029

theorem express_in_standard_form:
  ∃ a h k : ℝ, (∀ x : ℝ, 2 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = -3) :=
by
  use 2, 2, -7
  split
  { intro x
    calc
      2 * x^2 - 8 * x + 1
      = 2 * (x^2 - 4 * x) + 1           : by ring
  ... = 2 * ((x - 2)^2 - 4) + 1         : by
      
  ... = 2 * (x - 2)^2 - 8 + 1           : by 
  ... = 2 * (x - 2)^2 - 7               : by ring }
  { linarith }

end express_in_standard_form_l697_697029


namespace no_integer_solutions_l697_697509

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 :=
by
  sorry

end no_integer_solutions_l697_697509


namespace complex_number_solution_l697_697700

theorem complex_number_solution : 
  ∃ (z : ℂ), (|z - 2| = |z + 4| ∧ |z + 4| = |z - 2i|) ∧ z = -1 + ⅈ :=
by
  sorry

end complex_number_solution_l697_697700


namespace generated_surface_l697_697549

theorem generated_surface (L : ℝ → ℝ → ℝ → Prop)
  (H1 : ∀ x y z, L x y z → y = z) 
  (H2 : ∀ t, L (t^2 / 2) t 0) 
  (H3 : ∀ s, L (s^2 / 3) 0 s) : 
  ∀ y z, ∃ x, L x y z → x = (y - z) * (y / 2 - z / 3) :=
by
  sorry

end generated_surface_l697_697549


namespace woman_speed_in_still_water_l697_697642

noncomputable def speed_in_still_water (V_c : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  let V_downstream := d / (t / 3600)
  V_downstream - V_c

theorem woman_speed_in_still_water :
  let V_c := 60
  let t := 9.99920006399488
  let d := 0.5 -- 500 meters converted to kilometers
  speed_in_still_water V_c t d = 120.01800180018 :=
by
  unfold speed_in_still_water
  sorry

end woman_speed_in_still_water_l697_697642


namespace min_distance_sum_l697_697594

theorem min_distance_sum
  (A B C D E P : ℝ)
  (h_collinear : B = A + 2 ∧ C = B + 2 ∧ D = C + 3 ∧ E = D + 4)
  (h_bisector : P = (A + E) / 2) :
  (A - P)^2 + (B - P)^2 + (C - P)^2 + (D - P)^2 + (E - P)^2 = 77.25 :=
by
  sorry

end min_distance_sum_l697_697594


namespace arith_seq_general_formula_l697_697595

noncomputable def increasing_arith_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arith_seq_general_formula (a : ℕ → ℤ) (d : ℤ)
  (h_arith : increasing_arith_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = (a 2)^2 - 4) :
  ∀ n, a n = 3 * n - 2 :=
sorry

end arith_seq_general_formula_l697_697595


namespace individuals_grouping_l697_697479

def individuals : List Nat := [1, 2, 3, 4, 5, 6, 7, 8]

def pairs (l : List Nat) : List (List (Nat × Nat)) := 
  match l with
  | [] => [[]]
  | x :: xs => 
    xs.toFinset.powerset.filter (λ s, s.card = 1).toList.bind (λ s, 
      let p := (x, s.toList.head!)
      in pairs (xs.erase (s.toList.head!)).map (fun t => p :: t))

def groupings (l : List Nat) := 
  pairs l |>.quotientBy (Perm.homPerm Group.powRight 4)

theorem individuals_grouping : groupings individuals |> List.length = 105 := by
  -- proof to be filled in
  sorry

end individuals_grouping_l697_697479


namespace mean_median_mode_equal_l697_697550

theorem mean_median_mode_equal {y : ℕ} :
  let L := [2, 3, 4, 5, 5, 6, y]
  let mean := (L.sum : ℚ) / L.length
  let median := L.sorted.nth (L.length / 2)
  let mode := L.mode.head

  (mean = 5 ∧ median = some 5 ∧ mode = some 5) → y = 5 :=
begin
  sorry
end

end mean_median_mode_equal_l697_697550


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697229

theorem height_of_cylinder_inscribed_in_hemisphere :
  ∀ (r_cylinder r_hemisphere : ℝ),
  r_cylinder = 3 →
  r_hemisphere = 8 →
  (∃ h : ℝ, h = real.sqrt (r_hemisphere^2 - r_cylinder^2)) :=
begin
  intros r_cylinder r_hemisphere h1 h2,
  use real.sqrt (r_hemisphere^2 - r_cylinder^2),
  rw [h1, h2],
  simp,
  sorry
end

end height_of_cylinder_inscribed_in_hemisphere_l697_697229


namespace major_premise_is_statement_1_l697_697278

-- Conditions based on given problem
def rectangle_is_parallelogram (R : Type) [IsRectangle R] : IsParallelogram R := 
sorry

def triangle_is_not_parallelogram (T : Type) [IsTriangle T] : ¬ IsParallelogram T := 
sorry

def triangle_is_not_rectangle (T : Type) [IsTriangle T] (R : Type) [IsRectangle R] : ¬ (T = R) := 
sorry

-- To prove that statement ① (rectangle_is_parallelogram) is the major premise
theorem major_premise_is_statement_1
    (R : Type) [IsRectangle R]
    (T : Type) [IsTriangle T]: 
    (rectangle_is_parallelogram R) → 
    (triangle_is_not_parallelogram T) → 
    (triangle_is_not_rectangle T R) → 
    true :=
by sorry

end major_premise_is_statement_1_l697_697278


namespace find_S25_l697_697789

variable (a : ℕ → ℚ) (S : ℕ → ℚ)

-- Conditions: arithmetic sequence {a_n} and sum of the first n terms S_n with S15 - S10 = 1
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range(n), a (i + 1)

axiom condition : S 15 - S 10 = 1

-- Question: Prove that S25 = 5
theorem find_S25 (h1 : arithmetic_sequence a) (h2 : sum_of_first_n_terms a S) : S 25 = 5 :=
by
  sorry

end find_S25_l697_697789


namespace maria_earnings_l697_697908

def brushes_cost (c1 c2 c3 : ℕ) : ℕ := c1 + c2 + c3
def canvas_cost (B : ℕ) : ℕ := 3 * B + 2 * B
def paint_cost_acrylic (liters : ℕ) (cost_per_liter : ℕ) : ℕ := liters * cost_per_liter
def paint_cost_oil (liters : ℕ) (cost_per_liter : ℕ) : ℕ := liters * cost_per_liter
def total_material_cost (brush_cost canvas_cost acrylic_cost oil_cost : ℕ) : ℕ := brush_cost + canvas_cost + acrylic_cost + oil_cost
def earnings (selling_price total_cost : ℕ) : ℕ := selling_price - total_cost

theorem maria_earnings : 
  ∀ (S B A O : ℕ) (c1 c2 c3 : ℕ) (a_liters o_liters a_cost_per_liter o_cost_per_liter : ℕ),
  S = 200 →
  c1 = 20 →
  c2 = 25 →
  c3 = 30 →
  a_liters = 5 →
  o_liters = 3 →
  a_cost_per_liter = 8 →
  o_cost_per_liter = 12 →
  let B := brushes_cost c1 c2 c3 in
  let canvas := canvas_cost B in
  let acrylic := paint_cost_acrylic a_liters a_cost_per_liter in
  let oil := paint_cost_oil o_liters o_cost_per_liter in
  let total_cost := total_material_cost B canvas acrylic oil in
  earnings S total_cost = -326 :=
begin
  intros,
  sorry
end

end maria_earnings_l697_697908


namespace number_of_n_with_prime_g_l697_697898

noncomputable def g (n : ℕ) : ℕ :=
(n.divisors : finset ℕ).sum

theorem number_of_n_with_prime_g :
  { n : ℕ | 1 ≤ n ∧ n ≤ 30 ∧ Prime (g n) }.to_finset.card = 4 :=
by
  sorry

end number_of_n_with_prime_g_l697_697898


namespace smallest_non_palindrome_power_of_13_l697_697722

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l697_697722


namespace sequence_of_arrows_cycle_l697_697023

theorem sequence_of_arrows_cycle (n : ℕ) (h : n % 6 = 1) : 
  ∀ m, m % 6 = 0 → (m - n) % 6 = 5 :=
begin
  intros m hm,
  -- Proof will go here
  sorry
end

end sequence_of_arrows_cycle_l697_697023


namespace triangle_AD_length_l697_697458

theorem triangle_AD_length
  (A B C D : Type)
  (h : ∀ M, euclidean_geometry.point M [M = A, M = B, M = C])
  (AB AC : ℝ)
  (p : A ≠ B)
  (q : A ≠ C)
  (r : B ≠ C)
  (AB_eq : AB = 13)
  (AC_eq : AC = 20)
  (perpendicular : euclidean_geometry.perpendicular A D B C)
  (BD CD : ℝ)
  (ratio : BD / CD = 3 / 4) :
  let AD := sqrt (128) in AD = 8 * sqrt 2 :=
by
  sorry

end triangle_AD_length_l697_697458


namespace angle_A_eq_pi_div_3_area_triangle_ABC_l697_697853

/-- Given \(\triangle ABC\) with sides \( a, b, c \) opposite angles \( A, B, C \) respectively, and 
     \( 2a\cos A = b\cos C - c\cos B \), prove that \( A = \frac{\pi}{3} \). -/
theorem angle_A_eq_pi_div_3 {a b c : ℝ} 
    (h1 : 2 * a * real.cos A = b * real.cos C - c * real.cos B)
    (h2 : a = 6)
    (h3 : b + c = 8) : A = real.pi / 3 :=
sorry

/-- Given \(\triangle ABC\) with sides \( a, b, c \) opposite angles \( A, B, C \) respectively, 
     \( a = 6 \) and \( b + c = 8 \), prove that the area is \( \frac{7\sqrt{3}}{3} \). -/
theorem area_triangle_ABC {a b c : ℝ} 
    (h1 : a = 6)
    (h2 : b + c = 8)
    (h3 : A = real.pi / 3) : 
    let bc := (28 : ℝ) / 3
    in (1/2) * bc * (real.sin (real.pi / 3)) = (7 * real.sqrt 3) / 3 :=
sorry

end angle_A_eq_pi_div_3_area_triangle_ABC_l697_697853


namespace compound_interest_third_year_l697_697124

-- Definitions of conditions
def r : ℝ := 0.08
def CI_2 : ℝ := 1400

-- Definition of compound interest function
def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n - P

-- Definition of principal (P) calculated from CI_2
def P : ℝ := CI_2 / ((1 + r)^2 - 1)

-- Lean statement to prove CI_3 is approximately 2184.62
theorem compound_interest_third_year : 
  compound_interest P r 3 ≈ 2184.62 := 
by 
  sorry

end compound_interest_third_year_l697_697124


namespace length_of_AZ_l697_697064

theorem length_of_AZ 
  (A B C X Y Z : Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space X]
  [metric_space Y]
  [metric_space Z]
  (AB AC BC : ℝ)
  (hAB : AB = dist A B)
  (hAC : AC = dist A C)
  (hBC : BC = dist B C)
  (dist_AB : dist A B = 5)
  (dist_AC : dist A C = 8)
  (dist_BC : dist B C = 9)
  (angle_bisector_BCA_meets_ba_at_X : True) -- Placeholder for actual geometric condition
  (angle_bisector_CAB_meets_bc_at_Y : True) -- Placeholder for actual geometric condition
  (Z_intersection_of_XY_and_AC : True) -- Placeholder for actual geometric condition
  : dist A Z = 10 :=
sorry

end length_of_AZ_l697_697064


namespace dan_minimum_average_speed_l697_697537

def route1_total_distance := 180
def route2_total_distance := 210
def speed1 := 30
def distance1 := 60
def speed2 := 45
def distance2 := 120
def cara_departure_time := 0
def dan_departure_delay := 1 -- Dan leaves 60 minutes (1 hour) after Cara
def break_time := 0.25 -- Each 15-minute break in hours
def break_points := [70, 130] -- Stopping points on Route 2
def dan_required_time := 4.67 - dan_departure_delay - 2 * break_time -- Time Dan needs

theorem dan_minimum_average_speed :
  (210 / (4.67 - 1 - (2 * 0.25))) ≈ 66.25 := sorry

end dan_minimum_average_speed_l697_697537


namespace loom_weaving_approximation_l697_697588

noncomputable def loom_weaving_time (total_cloth : ℚ) (rate : ℚ) : ℚ :=
  total_cloth / rate

theorem loom_weaving_approximation :
  loom_weaving_time 15 0.126 ≈ 119 := by sorry

end loom_weaving_approximation_l697_697588


namespace inequality_solution_l697_697112

theorem inequality_solution (x : ℝ) :
  ( (x - 1) * (x - 3) * (x - 5) / ((x - 2) * (x - 4) * (x - 6)) > 0 ) ↔
  (x ∈ set.Iio 1 ∪ set.Ioo 2 3 ∪ set.Ioo 4 5 ∪ set.Ioi 6) :=
by sorry

end inequality_solution_l697_697112


namespace ann_frosting_cakes_l697_697502

theorem ann_frosting_cakes (normalRate sprainedRate cakes : ℕ) (H1 : normalRate = 5) (H2 : sprainedRate = 8) (H3 : cakes = 10) :
  (sprainedRate * cakes) - (normalRate * cakes) = 30 :=
by
  -- Substitute the provided values into the expression
  rw [H1, H2, H3]
  -- Evaluate the expression
  norm_num

end ann_frosting_cakes_l697_697502


namespace third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l697_697969

-- Define the first finite difference function
def delta (f : ℕ → ℤ) (n : ℕ) : ℤ := f (n + 1) - f n

-- Define the second finite difference using the first
def delta2 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta f) n

-- Define the third finite difference using the second
def delta3 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta2 f) n

-- Prove the third finite difference of n^3 is 6
theorem third_diff_n_cube_is_const_6 :
  delta3 (fun (n : ℕ) => (n : ℤ)^3) = fun _ => 6 := 
by
  sorry

-- Prove the third finite difference of the general form function is 6
theorem third_diff_general_form_is_6 (a b c : ℤ) :
  delta3 (fun (n : ℕ) => (n : ℤ)^3 + a * (n : ℤ)^2 + b * (n : ℤ) + c) = fun _ => 6 := 
by
  sorry

end third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l697_697969


namespace parabola_shift_l697_697987

theorem parabola_shift (x : ℝ) : 
  let y := -2 * x^2 
  let y1 := -2 * (x + 1)^2 
  let y2 := y1 - 3 
  y2 = -2 * x^2 - 4 * x - 5 := 
by 
  sorry

end parabola_shift_l697_697987


namespace arithmetic_geometric_sequence_l697_697378

theorem arithmetic_geometric_sequence
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (b : ℕ → ℕ)
  (T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : S 3 = 12)
  (h3 : (a 1 + (a 2 - a 1))^2 = a 1 * (a 1 + 2 * (a 2 - a 1) + 2))
  (h4 : ∀ n, b n = (3 ^ n) * a n) :
  (∀ n, a n = 2 * n) ∧ 
  (∀ n, T n = (2 * n - 1) * 3^(n + 1) / 2 + 3 / 2) :=
sorry

end arithmetic_geometric_sequence_l697_697378


namespace exists_unique_P_circumcircle_passes_through_fixed_point_l697_697365

-- Define the conditions
variables (A B C M P D N: Point)
variables (A' B' C': Point)
variables (circumcircle_MNP : Circle)

-- Define the equilateral triangle ABC
def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Point conditions for M being a reflection point bringing A, B, C to A', B', C'
def reflections (A B C A' B' C' M : Point) : Prop :=
  reflect A M = A' ∧ reflect B M = B' ∧ reflect C M = C'

-- Midpoint condition
def midpoint (D A B : Point) : Prop :=
  dist A D = dist B D ∧ dist A D ≠ 0

-- Intersection condition
def intersection (D M : Point) (N : Point) : Prop :=
  line_of_points D M ∩ line_of_points A P = N

-- Part I: Prove the existence of a unique P equidistant from certain points
theorem exists_unique_P (A B C A' B' C' M : Point) 
(equi_triangle : equilateral_triangle A B C)
(reflections : reflections A B C A' B' C' M) : 
∃! P, dist P A = dist P B' ∧ dist P B = dist P C' ∧ dist P C = dist P A' :=
sorry

-- Part II: Prove that the circumcircle of triangle MNP passes through a fixed point
theorem circumcircle_passes_through_fixed_point (A B C A' B' C' M D N P fixed_point: Point)
(equi_triangle : equilateral_triangle A B C)
(reflections : reflections A B C A' B' C' M)
(mid_seg : midpoint D A B)
(intersect : intersection D M N)
(M_ne_D : M ≠ D) :
circumcircle_MNP.passes_through fixed_point :=
sorry

end exists_unique_P_circumcircle_passes_through_fixed_point_l697_697365


namespace workers_new_daily_wage_l697_697240

def wage_before : ℝ := 25
def increase_percentage : ℝ := 0.40

theorem workers_new_daily_wage : wage_before * (1 + increase_percentage) = 35 :=
by
  -- sorry will be replaced by the actual proof steps
  sorry

end workers_new_daily_wage_l697_697240


namespace median_room_number_l697_697247

def median_of_rooms : Nat :=
  let rooms := List.range' 1 11 ++ List.range' 15 11
  rooms.get (rooms.length / 2)

theorem median_room_number :
  let rooms := List.range' 1 11 ++ List.range' 15 11 in
  rooms.get (rooms.length / 2) = 11 := by
  sorry

end median_room_number_l697_697247


namespace number_of_goats_l697_697423

-- Define the conditions using Lean definitions
def caravan_conditions : Prop :=
  ∃ (G : ℕ), 
  let total_heads := 50 + 8 + 15 + G in
  let total_feet := 50 * 2 + 8 * 4 + 15 * 2 + G * 4 in
  total_feet = total_heads + 224

-- The statement we want to prove
theorem number_of_goats : ∃ (G : ℕ), caravan_conditions ∧ G = 45 := 
by
  sorry

end number_of_goats_l697_697423


namespace max_rubles_can_receive_l697_697922

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l697_697922


namespace cos_pi_over_3_minus_2alpha_l697_697404

variable (α : ℝ)

def condition (α : ℝ) : Prop := cos (α + real.pi / 3) = 4 / 5

theorem cos_pi_over_3_minus_2alpha (h : condition α) : cos (real.pi / 3 - 2 * α) = -7 / 25 :=
by
  sorry

end cos_pi_over_3_minus_2alpha_l697_697404


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697735

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l697_697735


namespace three_digit_numbers_with_square_ending_in_them_l697_697299

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l697_697299


namespace triangle_two_solutions_range_of_a_l697_697056

noncomputable def range_of_a (a b : ℝ) (A : ℝ) : Prop :=
b * Real.sin A < a ∧ a < b

theorem triangle_two_solutions_range_of_a (a : ℝ) (A : ℝ := Real.pi / 6) (b : ℝ := 2) :
  range_of_a a b A ↔ 1 < a ∧ a < 2 := by
sorry

end triangle_two_solutions_range_of_a_l697_697056


namespace find_xy_pairs_l697_697290

theorem find_xy_pairs : 
  ∃ (x y : ℝ), (x + y = 2) ∧ (x^4 - y^4 = 5 * x - 3 * y) ∧ 
               (x = 3 + (1 / real.cbrt 4)) ∧ (y = -1 - (1 / real.cbrt 4)) :=
sorry

end find_xy_pairs_l697_697290


namespace graph_passes_through_fixed_point_l697_697132

theorem graph_passes_through_fixed_point (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : (∃ x y : ℝ, x = -1 ∧ y = 0 ∧ y = a^(x+1) - 1) :=
by
  use -1
  use 0
  split
  · exact rfl
  split
  · exact rfl
  · sorry

end graph_passes_through_fixed_point_l697_697132


namespace common_sales_day_l697_697601

theorem common_sales_day :
  let bookstore_sales := {4, 8, 12, 16, 20, 24, 28}
  let shoestore_sales := {2, 9, 16, 23, 30}
  let common_days := bookstore_sales ∩ shoestore_sales
  common_days = {16} :=
by
  let bookstore_sales := {4, 8, 12, 16, 20, 24, 28}
  let shoestore_sales := {2, 9, 16, 23, 30}
  let common_days := bookstore_sales ∩ shoestore_sales
  have h1 : common_days = {16} := sorry
  exact h1

end common_sales_day_l697_697601


namespace petya_max_rubles_l697_697940

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l697_697940


namespace roommate_payment_ratio_l697_697469

/-- Define the cost of the first two tickets -/
def first_two_tickets_cost (cost_per_ticket : ℝ) : ℝ :=
  2 * cost_per_ticket

/-- Define the cost of the third ticket -/
def third_ticket_cost (cost_per_ticket : ℝ) : ℝ :=
  cost_per_ticket / 3

/-- Define the total cost of all tickets -/
def total_cost (cost_per_ticket : ℝ) : ℝ :=
  first_two_tickets_cost cost_per_ticket + third_ticket_cost cost_per_ticket

/-- Define the amount paid by the roommate -/
def paid_by_roommate (cost_per_ticket : ℝ) (amount_left_by_james : ℝ) : ℝ :=
  total_cost cost_per_ticket - (total_cost cost_per_ticket - amount_left_by_james)

/-- Prove the ratio of the amount paid by the roommate to the total cost of the tickets is 13:14 -/
theorem roommate_payment_ratio (cost_per_ticket : ℝ) (amount_left_by_james : ℝ) :
  cost_per_ticket = 150 →
  amount_left_by_james = 325 →
  (paid_by_roommate cost_per_ticket amount_left_by_james) / (total_cost cost_per_ticket) = 13 / 14 :=
begin
  intros h1 h2,
  simp [first_two_tickets_cost, third_ticket_cost, total_cost, paid_by_roommate],
  rewrite h1,
  rewrite h2,
  norm_num,
end

end roommate_payment_ratio_l697_697469


namespace combined_flock_after_5_years_l697_697688

theorem combined_flock_after_5_years :
  let initial_flock : ℕ := 100
  let annual_net_gain : ℕ := 30 - 20
  let years : ℕ := 5
  let joined_flock : ℕ := 150
  let final_flock := initial_flock + annual_net_gain * years + joined_flock
  in final_flock = 300 :=
by
  sorry

end combined_flock_after_5_years_l697_697688


namespace min_value_512_l697_697074

noncomputable def min_value (a b c d e f g h : ℝ) : ℝ :=
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2

theorem min_value_512 
  (a b c d e f g h : ℝ)
  (H1 : a * b * c * d = 8)
  (H2 : e * f * g * h = 16) : 
  ∃ (min_val : ℝ), min_val = 512 ∧ min_value a b c d e f g h = min_val :=
sorry

end min_value_512_l697_697074


namespace quadratic_roots_range_l697_697385

variable (a : ℝ)

theorem quadratic_roots_range (h : ∀ b c (eq : b = -a ∧ c = a^2 - 4), ∃ x y, x ≠ y ∧ x^2 + b * x + c = 0 ∧ x > 0 ∧ y^2 + b * y + c = 0) :
  -2 ≤ a ∧ a ≤ 2 :=
by sorry

end quadratic_roots_range_l697_697385


namespace find_complex_number_l697_697704

theorem find_complex_number (z : ℂ) (h1 : complex.abs (z - 2) = complex.abs (z + 4))
  (h2 : complex.abs (z + 4) = complex.abs (z - 2 * complex.I)) : z = -1 + complex.I :=
by
  sorry

end find_complex_number_l697_697704


namespace max_min_projection_in_directions_l697_697243

noncomputable def max_projection_sum (AB : ℝ) : ℝ :=
  2 * AB

noncomputable def min_projection_sum (AB : ℝ) : ℝ :=
  AB * Real.sqrt 3

-- Assuming AB is a segment inside an equilateral triangle,
-- we'll provide the proof later to validate the correctness
theorem max_min_projection_in_directions (AB : ℝ) (h : 0 ≤ AB ∧ AB ≤ 1) :
  ∃ (θ_max θ_min : ℝ), 
    (0 ≤ θ_max ∧ θ_max ≤ π / 2 ∧ max_projection_sum AB = sum_projections θ_max AB) ∧
    (0 ≤ θ_min ∧ θ_min ≤ π / 2 ∧ min_projection_sum AB = sum_projections θ_min AB) := sorry

end max_min_projection_in_directions_l697_697243


namespace find_c_value_l697_697171

theorem find_c_value :
  (∀ x : ℝ, x ∈ Ioo (-5 / 2) 3 → x * (4 * x + 2) < 45) :=
begin
  sorry
end

end find_c_value_l697_697171


namespace triangle_AD_length_l697_697457

theorem triangle_AD_length
  (A B C D : Type)
  (h : ∀ M, euclidean_geometry.point M [M = A, M = B, M = C])
  (AB AC : ℝ)
  (p : A ≠ B)
  (q : A ≠ C)
  (r : B ≠ C)
  (AB_eq : AB = 13)
  (AC_eq : AC = 20)
  (perpendicular : euclidean_geometry.perpendicular A D B C)
  (BD CD : ℝ)
  (ratio : BD / CD = 3 / 4) :
  let AD := sqrt (128) in AD = 8 * sqrt 2 :=
by
  sorry

end triangle_AD_length_l697_697457


namespace radius_of_cone_l697_697203

theorem radius_of_cone (R r h h_cone : ℝ) (V_cylinder V_cone : ℝ)
  (h1 : R = 8)
  (h2 : r = 5)
  (h3 : h = 2)
  (h4 : h_cone = 6)
  (h5 : V_cylinder = π * h * (R^2 - r^2))
  (h6 : V_cone = (1/3) * π * (sqrt 39)^2 * h_cone):
  sqrt 39 ≈ 6.24 := 
by sorry

end radius_of_cone_l697_697203


namespace rotated_point_coordinates_l697_697367

noncomputable def A : ℝ × ℝ := (1, 2)

def rotate_90_counterclockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.snd, p.fst)

theorem rotated_point_coordinates :
  rotate_90_counterclockwise A = (-2, 1) :=
by
  -- Skipping the proof
  sorry

end rotated_point_coordinates_l697_697367


namespace opposite_rational_division_l697_697585

theorem opposite_rational_division (a : ℚ) :
  (a ≠ 0 → a / -a = -1) ∧ (a = 0 → ¬is_defined (λ x : ℚ, 0 / x)) := 
by
  sorry

end opposite_rational_division_l697_697585


namespace petya_max_rubles_l697_697942

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l697_697942


namespace isosceles_triangle_side_length_l697_697532

theorem isosceles_triangle_side_length (b : ℝ) (A : ℝ) (a : ℝ)
  (h₁ : b = 26)
  (h₂ : A = 78) :
  a = real.sqrt 205 :=
by
  sorry

end isosceles_triangle_side_length_l697_697532


namespace triangle_BDC_is_isosceles_l697_697089

-- Define the given conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC BC AD DC : ℝ)
variables (a : ℝ)
variables (α : ℝ)

-- Given conditions
def is_isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) : Prop :=
AB = AC

def angle_BAC_120 (α : ℝ) : Prop :=
α = 120

def point_D_extension (AD AB : ℝ) : Prop :=
AD = 2 * AB

-- Let triangle ABC be isosceles with AB = AC and angle BAC = 120 degrees
axiom isosceles_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB AC : ℝ) : is_isosceles_triangle A B C AB AC

axiom angle_BAC (α : ℝ) : angle_BAC_120 α

axiom point_D (AD AB : ℝ) : point_D_extension AD AB

-- Prove that triangle BDC is isosceles
theorem triangle_BDC_is_isosceles 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC BC AD DC : ℝ) 
  (α : ℝ) 
  (h1 : is_isosceles_triangle A B C AB AC)
  (h2 : angle_BAC_120 α)
  (h3 : point_D_extension AD AB) :
  BC = DC :=
sorry

end triangle_BDC_is_isosceles_l697_697089


namespace evaluate_expression_at_one_l697_697676

theorem evaluate_expression_at_one : 
  let y := λ x : ℝ, (x + 2) / (x - 2) in
  (x : ℝ → ℝ) →
  ∀ x, x = 1 → 
  (y x + 2) / (y x - 2) = 1 / 5 := 
by
  intros y x hx
  have hy : y 1 = (1 + 2) / (1 - 2) := by simp [y]
  simp [hx, hy]
  sorry

end evaluate_expression_at_one_l697_697676


namespace passes_midpoint_of_LM_l697_697137

open EuclideanGeometry

-- Definitions for the conditions
variable {A B C K L M D E : Point}

-- The sides of triangle ABC are of different lengths.
axiom hABC : ¬ collinear A B C

-- The incircle of triangle ABC touches sides BC, CA, and AB at points K, L, and M respectively.
axiom hIncircle : ∃ I : Point, inscribedCircleInTriangle I A B C K L M

-- Definitions for intersections
def isMidpoint (P Q R : Point) : Prop := distance P Q = distance P R

-- Conditions of the parallel lines intersecting at points D and E
axiom hParallelBDLM : parallel (line_through B D) (line_through L M)
axiom hParallelCELM : parallel (line_through C E) (line_through L M)

-- The line segment DE passes through the midpoint of LM
theorem passes_midpoint_of_LM (hABC : ¬ collinear A B C) 
  (hIncircle : ∃ I : Point, inscribedCircleInTriangle I A B C K L M)
  (hParallelBDLM : parallel (line_through B D) (line_through L M))
  (hParallelCELM : parallel (line_through C E) (line_through L M)) :
  ∃ M' : Point, isMidpoint M' L M ∧ collinear D E M' :=
sorry

end passes_midpoint_of_LM_l697_697137


namespace right_triangle_third_side_l697_697360

theorem right_triangle_third_side (x y : ℝ) (h : |x - 4| + real.sqrt (y - 3) = 0) :
  ∃ z : ℝ, z = 5 ∨ z = real.sqrt 7 :=
by
  sorry

end right_triangle_third_side_l697_697360


namespace complete_square_sum_l697_697417

theorem complete_square_sum (a h k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) →
  a + h + k = -5 :=
by
  intro h1
  sorry

end complete_square_sum_l697_697417


namespace number_of_n_with_prime_g_l697_697897

noncomputable def g (n : ℕ) : ℕ :=
(n.divisors : finset ℕ).sum

theorem number_of_n_with_prime_g :
  { n : ℕ | 1 ≤ n ∧ n ≤ 30 ∧ Prime (g n) }.to_finset.card = 4 :=
by
  sorry

end number_of_n_with_prime_g_l697_697897


namespace length_of_train_l697_697609

theorem length_of_train 
  (speed_kmph : ℕ)
  (platform_length_m : ℕ)
  (crossing_time_s : ℕ)
  (speed_kmph = 72)
  (platform_length_m = 260)
  (crossing_time_s = 26):
  let speed_mps := speed_kmph * 1000 / 3600 in
  let total_distance_m := speed_mps * crossing_time_s in
  let train_length_m := total_distance_m - platform_length_m in
  train_length_m = 260 :=
by 
  -- Convert speed from km/h to m/s
  let speed_mps := 72 * 1000 / 3600
  -- Calculate the total distance covered
  let total_distance_m := speed_mps * 26
  -- Calculate the train length by subtracting the platform length
  let train_length_m := total_distance_m - 260
  -- Prove the train length is 260 meters
  show train_length_m = 260 from sorry

end length_of_train_l697_697609


namespace maximum_rubles_received_l697_697926

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l697_697926


namespace volume_difference_l697_697339

-- Define the dimensions of the first bowl
def length1 : ℝ := 14
def width1 : ℝ := 16
def height1 : ℝ := 9

-- Define the dimensions of the second bowl
def length2 : ℝ := 14
def width2 : ℝ := 16
def height2 : ℝ := 4

-- Define the volumes of the two bowls assuming they are rectangular prisms
def volume1 : ℝ := length1 * width1 * height1
def volume2 : ℝ := length2 * width2 * height2

-- Statement to prove the volume difference
theorem volume_difference : volume1 - volume2 = 1120 := by
  sorry

end volume_difference_l697_697339


namespace maximum_rubles_received_l697_697947

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l697_697947


namespace infinite_n_exists_l697_697970

open Classical
noncomputable section

def sigma (n : ℕ) : ℕ := (Finset.range n.succ).filter (λ d, n % d = 0).sum id

theorem infinite_n_exists (σ : ℕ → ℕ) (h : ∀ n : ℕ, σ n = sigma n) : 
    ∃ᶠ n in at_top, ∀ k : ℕ, k < n → (σ(n):ℚ) / n > (σ(k):ℚ) / k :=
sorry

end infinite_n_exists_l697_697970


namespace sum_ceil_log2_eq_19854_l697_697480

noncomputable def log2 (x : ℝ) : ℝ := real.log x / real.log 2
noncomputable def ceil (x : ℝ) : ℝ := real.ceil x

theorem sum_ceil_log2_eq_19854 : 
  (∑ n in finset.range 1991, ceil (log2 (n + 1))) = 19854 := 
sorry

end sum_ceil_log2_eq_19854_l697_697480


namespace find_special_three_digit_numbers_l697_697297

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l697_697297


namespace right_angle_at_P_l697_697649

theorem right_angle_at_P 
  (a b c : ℝ) 
  (P A B C : ℝ × ℝ) 
  (hPA : dist P A = a) 
  (hPB : dist P B = b) 
  (hPC : dist P C = b + c) 
  (h_relation : a^2 = b^2 + c^2)
  (P_interior_square : ∃ D : ℝ × ℝ, is_square A B C D ∧ P ∈ interior (polygon A B C D)) : 
  ∠ B P C = π / 2 := 
sorry

end right_angle_at_P_l697_697649


namespace symmetric_center_of_graph_l697_697158

-- Definitions given in the problem conditions
def f (x : ℝ) : ℝ := x^3 - 6 * x^2

def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)
def is_symmetric_with_respect_to (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  is_odd (λ x, g (x + a) - b)

-- Proof problem statement
theorem symmetric_center_of_graph :
  ∃ a b : ℝ, is_symmetric_with_respect_to f a b ∧ a = 2 ∧ b = -16 :=
by
  sorry

end symmetric_center_of_graph_l697_697158


namespace quadrilateral_BD_l697_697047

theorem quadrilateral_BD :
  ∀ (BD : ℕ), (AB = 6 ∧ BC = 20 ∧ CD = 6 ∧ DA = 12) → (BD = 15 ∨ BD = 16 ∨ BD = 17) :=
by
  unfold AB BC CD DA
  -- assuming given conditions in problem statement
  assume BD h
  sorry

end quadrilateral_BD_l697_697047


namespace total_stickers_purchased_l697_697917

-- Definitions for the number of sheets and stickers per sheet for each folder
def num_sheets_per_folder := 10
def stickers_per_sheet_red := 3
def stickers_per_sheet_green := 2
def stickers_per_sheet_blue := 1

-- Theorem stating that the total number of stickers is 60
theorem total_stickers_purchased : 
  num_sheets_per_folder * (stickers_per_sheet_red + stickers_per_sheet_green + stickers_per_sheet_blue) = 60 := 
  by
  -- Skipping the proof
  sorry

end total_stickers_purchased_l697_697917


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697737

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l697_697737


namespace profit_without_discount_equals_22_55_l697_697233

-- Definitions based on given conditions
def CP : ℝ := 100
def discount_rate : ℝ := 5 / 100
def profit_rate_with_discount : ℝ := 22.55 / 100

-- Prove that the percentage of profit without discount is the same
theorem profit_without_discount_equals_22_55 :
  let SP_with_discount := CP - (discount_rate * CP),
      profit_with_discount := profit_rate_with_discount * CP,
      SP_without_discount := SP_with_discount + (discount_rate * CP),
      profit_rate_without_discount := (profit_with_discount / CP) * 100 in
  profit_rate_without_discount = 22.55 :=
by
  sorry

end profit_without_discount_equals_22_55_l697_697233


namespace find_AD_l697_697455

-- Declare the conditions as hypotheses
theorem find_AD (AB AC : ℝ) (BD CD AD : ℝ) 
  (h1 : AB = 13) (h2 : AC = 20) (h_ratio : BD / CD = 3 / 4) 
  (h1_perp : AD = (hypotenuse_sqrt (BD^2 + AD^2) = AB) )
  (h2_perp : AD = (hypotenuse_sqrt (CD^2 + AD^2) = AC)) :
  AD = 8 * Real.sqrt 2 := 
  -- Include sorry to indicate the proof is omitted
  sorry

end find_AD_l697_697455


namespace cost_per_bracelet_l697_697513

/-- Each friend and the number of their name's letters -/
def friends_letters_counts : List (String × Nat) :=
  [("Jessica", 7), ("Tori", 4), ("Lily", 4), ("Patrice", 7)]

/-- Total cost spent by Robin -/
def total_cost : Nat := 44

/-- Calculate the total number of bracelets -/
def total_bracelets : Nat :=
  friends_letters_counts.foldr (λ p acc => p.snd + acc) 0

theorem cost_per_bracelet : (total_cost / total_bracelets) = 2 :=
  by
    sorry

end cost_per_bracelet_l697_697513


namespace min_value_of_x_l697_697342

theorem min_value_of_x (x : ℝ) (h : 2 * (x + 1) ≥ x + 1) : x ≥ -1 := sorry

end min_value_of_x_l697_697342


namespace sec_minus_tan_l697_697832

theorem sec_minus_tan (x : ℝ) (h : sec x + tan x = 3) : sec x - tan x = 1 / 3 :=
sorry

end sec_minus_tan_l697_697832


namespace black_cat_detective_catches_one_ear_in_14_minutes_l697_697250

theorem black_cat_detective_catches_one_ear_in_14_minutes :
  ∀ v : ℝ, 
    let d1 := 13 * (5 * v),
        d2 := 13 * v,
        D := d1 + d2,
        new_speed := 1.5 * 5 * v,
        relative_speed := new_speed - v in
    (D / relative_speed = 1) → (14 = 13 + 1) :=
by
  intros v d1 d2 D new_speed relative_speed h1
  sorry

end black_cat_detective_catches_one_ear_in_14_minutes_l697_697250


namespace max_rubles_can_receive_l697_697923

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l697_697923


namespace correct_statements_l697_697563

namespace BallDrawing

-- Definitions for the events
def red_balls := 6
def white_balls := 3
def balls_marked_1 := 2 -- 2 red + 1 white
def balls_marked_red_1 := 2
def balls_marked_white_1 := 1
def total_balls := red_balls + white_balls

-- Event A: The first ball drawn is red
def event_A := "The first ball drawn is red"

-- Event B: The first ball drawn is marked with the number 1
def event_B := "The first ball drawn is marked with the number 1"

-- Event C: The first ball drawn is marked with the number 2
def event_C := "The first ball drawn is marked with the number 2"

-- Event D: The second ball drawn is marked with the number 1
def event_D := "The second ball drawn is marked with the number 1"

-- Statement B is correct because by definition events B and C are mutually exclusive
def statement_B_correct : Prop := 
  (λ B C, B ∧ C -> False) (event_B = "The first ball drawn is marked with the number 1") (event_C = "The first ball drawn is marked with the number 2")

-- Statement D is correct because events A and D are independent
def statement_D_correct : Prop := 
  let P_A := (red_balls : ℝ) / total_balls
  let P_D := ((balls_marked_red_1 / total_balls * balls_marked_white_1 / (total_balls - 1)) + (balls_marked_white_1 / total_balls * balls_marked_white_1 / (total_balls - 1)))
  let P_AD := P_A * P_D
  P_AD = P_A * P_D

-- Final theorem combining the correct statements
theorem correct_statements : statement_B_correct ∧ statement_D_correct := 
by {
  sorry
}

end BallDrawing

end correct_statements_l697_697563


namespace min_value_xy2z_l697_697078

theorem min_value_xy2z (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_xyz : x * y^2 * z = 64) :
  ∃ m, (∀ x y z, x > 0 → y > 0 → z > 0 → x * y^2 * z = 64 → x^2 + 8 * x * y + 8 * y^2 + 4 * z^2 ≥ m) ∧ m = 1536 :=
begin
  use 1536,
  intros x y z hx hy hz hxyz,
  -- Here the proof steps would be provided
  sorry
end

end min_value_xy2z_l697_697078


namespace min_value_of_x_plus_y_l697_697354

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y = x * y) :
  x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l697_697354


namespace volume_of_cube_l697_697993

theorem volume_of_cube (d : ℝ) (h : d = 5 * Real.sqrt 3) : ∃ (V : ℝ), V = 125 := by
  sorry

end volume_of_cube_l697_697993


namespace evaluate_expression_l697_697285

theorem evaluate_expression : (502 * 502) - (501 * 503) = 1 := sorry

end evaluate_expression_l697_697285


namespace blake_change_l697_697661

theorem blake_change :
  ∃ (change : ℕ), 
    let lollipop_cost := 2 in
    let lollipops := 4 in
    let chocolate_cost := lollipop_cost * 4 in
    let chocolates := 6 in
    let total_cost := (lollipop_cost * lollipops) + (chocolate_cost * chocolates) in
    let amount_given := 10 * 6 in
    change = amount_given - total_cost := 
by
  use 4
  sorry

end blake_change_l697_697661


namespace perpendicular_lines_a_eq_3_l697_697794

theorem perpendicular_lines_a_eq_3 (a : ℝ) :
  let l₁ := (a + 1) * x + 2 * y + 6
  let l₂ := x + (a - 5) * y + a^2 - 1
  (a ≠ 5 → -((a + 1) / 2) * (1 / (5 - a)) = -1) → a = 3 := by
  intro l₁ l₂ h
  sorry

end perpendicular_lines_a_eq_3_l697_697794


namespace total_tissues_brought_l697_697548

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end total_tissues_brought_l697_697548


namespace cos_alpha_plus_pi_over_three_tan_two_alpha_minus_pi_over_four_l697_697771

theorem cos_alpha_plus_pi_over_three (α : ℝ) (h1 : sin (π - α) = 2 * cos α) (hα : 0 < α ∧ α < π / 2) :
    cos (α + π / 3) = (√5 - 2 * √15) / 10 := by
  sorry

theorem tan_two_alpha_minus_pi_over_four (α : ℝ) (h1 : sin (π - α) = 2 * cos α) :
    tan (2 * α - π / 4) = 7 := by
  sorry

end cos_alpha_plus_pi_over_three_tan_two_alpha_minus_pi_over_four_l697_697771


namespace new_average_l697_697981

theorem new_average (n : ℕ) (average : ℝ) (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : average = 80)
  (h3 : new_average = (2 * average * n) / n) : 
  new_average = 160 := 
by 
  simp [h1, h2, h3]
  sorry

end new_average_l697_697981


namespace area_of_isosceles_trapezoid_l697_697682

-- Define the conditions
def base1 := 18 -- length of the smaller base in meters
def base2 := 24 -- length of the larger base in meters
def height := 6 -- height of the trapezoid in meters
def angle := 30 -- angle between the longer base and one of the non-parallel sides in degrees

-- Define a function to calculate the area of the trapezoid
noncomputable def calculate_area (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

-- Statement to prove the area of the trapezoid is 126 square meters
theorem area_of_isosceles_trapezoid : calculate_area base1 base2 height = 126 := by
  sorry

end area_of_isosceles_trapezoid_l697_697682


namespace austin_initial_amount_l697_697658

-- Definitions for conditions
def num_robots := 7
def cost_per_robot := 8.75
def tax := 7.22
def change_left := 11.53

-- Calculation of total spent
def total_cost_robots := num_robots * cost_per_robot
def total_cost_with_tax := total_cost_robots + tax
def total_amount := total_cost_with_tax + change_left

-- Theorem to prove the initial amount Austin started with
theorem austin_initial_amount : total_amount = 80.00 := by
  sorry

end austin_initial_amount_l697_697658


namespace total_drawings_first_five_pages_l697_697878

theorem total_drawings_first_five_pages : 
  ∀ (drawings_per_page : ℕ → ℕ) (n : ℕ),
  (drawings_per_page 1 = 5) →
  (∀ k, drawings_per_page (k + 1) = drawings_per_page k + 5) →
  (∑ k in finset.range 5, drawings_per_page (k + 1)) = 75 :=
by
  intros drawings_per_page n h1 h2
  sorry

end total_drawings_first_five_pages_l697_697878


namespace colored_pencils_most_blue_l697_697147

theorem colored_pencils_most_blue :
  let total_pencils := 24
  let red_pencils := total_pencils / 4
  let blue_pencils := red_pencils + 6
  let yellow_pencils := total_pencils - (red_pencils + blue_pencils)
  in max red_pencils (max blue_pencils yellow_pencils) = blue_pencils :=
by
  sorry

end colored_pencils_most_blue_l697_697147


namespace avg_growth_rate_production_lines_target_infeasibility_of_60000_l697_697586

-- 1. Prove that the average growth rate of sales in the first three quarters is 20%
theorem avg_growth_rate (q1_sales q3_sales : ℕ) (x : ℚ) :
  q1_sales = 20000 → q3_sales = 28800 → (1 + x) ^ 2 = q3_sales / q1_sales → x = 0.2 :=
by
  intros
  sorry

-- 2. Prove that 5 production lines are required to achieve a production target of 26,000 units per quarter
theorem production_lines_target (max_capacity per_reduction : ℕ) (target_production : ℕ) :
  max_capacity = 6000 → per_reduction = 200 → target_production = 26000 →
  ∃ m : ℕ, 30 * m - m^2 = 100 ∧ 26 ≤ (1 + m) * (6000 - 200 * m) := 
by
  intros
  sorry

-- 3. Prove that it is not possible to produce 60,000 units per quarter with any number of production lines
theorem infeasibility_of_60000 (max_capacity per_reduction : ℕ) (target_production : ℕ) :
  max_capacity = 6000 → per_reduction = 200 → target_production = 60000 →
  ¬ ∃ n : ℕ, n^2 - 30 * n + 270 = 0 ∧ 60 ≤ (1 + n) * (6000 - 200 * n) :=
by
  intros
  sorry

end avg_growth_rate_production_lines_target_infeasibility_of_60000_l697_697586


namespace three_digit_ends_with_itself_iff_l697_697308

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l697_697308


namespace train_cross_time_l697_697640

noncomputable def time_to_cross_platform (train_length : ℝ) (train_speed_kmph : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length in
  let train_speed_mps := train_speed_kmph * (1000 / 3600) in
  total_distance / train_speed_mps

theorem train_cross_time (train_length : ℝ) (train_speed_kmph : ℝ) (platform_length : ℝ)
  (h_train_length : train_length = 200) (h_train_speed : train_speed_kmph = 80) (h_platform_length : platform_length = 288.928) :
  time_to_cross_platform train_length train_speed_kmph platform_length = 22 :=
by
  -- Skipping the proof for now
  sorry

end train_cross_time_l697_697640


namespace additional_seasons_is_one_l697_697060

-- Definitions for conditions
def episodes_per_season : Nat := 22
def episodes_last_season : Nat := episodes_per_season + 4
def episodes_in_9_seasons : Nat := 9 * episodes_per_season
def hours_per_episode : Nat := 1 / 2 -- Stored as half units

-- Given conditions
def total_hours_to_watch_after_last_season: Nat := 112 * 2 -- converted to half-hours
def time_watched_in_9_seasons: Nat := episodes_in_9_seasons * hours_per_episode
def additional_hours: Nat := total_hours_to_watch_after_last_season - time_watched_in_9_seasons

-- Theorem to prove
theorem additional_seasons_is_one : additional_hours / hours_per_episode = episodes_last_season -> 
      additional_hours / hours_per_episode / episodes_per_season = 1 :=
by
  sorry

end additional_seasons_is_one_l697_697060


namespace find_complex_number_l697_697703

theorem find_complex_number (z : ℂ) (h1 : complex.abs (z - 2) = complex.abs (z + 4))
  (h2 : complex.abs (z + 4) = complex.abs (z - 2 * complex.I)) : z = -1 + complex.I :=
by
  sorry

end find_complex_number_l697_697703


namespace original_denominator_l697_697234

theorem original_denominator (d : ℕ) (h : 11 = 3 * (d + 8)) : d = 25 :=
by
  sorry

end original_denominator_l697_697234


namespace determine_y_increase_volume_l697_697553

noncomputable def volume_increase_y (r h y : ℝ) : Prop :=
  (1/3) * Real.pi * (r + y)^2 * h = (1/3) * Real.pi * r^2 * (h + y)

theorem determine_y_increase_volume (y : ℝ) :
  volume_increase_y 5 12 y ↔ y = 31 / 12 :=
by
  sorry

end determine_y_increase_volume_l697_697553


namespace smallest_power_of_13_non_palindrome_l697_697716

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l697_697716


namespace exist_parallelogram_inside_triangle_ABC_l697_697080

variable {α : Type} [LinearOrderedField α]
  {A B C P : α} -- Points A, B, C, and P
  {D E F : α}   -- Points D, E, and F where AP, BP, CP intersect sides of the triangle

theorem exist_parallelogram_inside_triangle_ABC
  (hP_inside_triangle_ABC : ∀ x y z : α, P ∈ triangle A B C)
  (hD_intersect : ∃ D, line_through A P ∩ side B C = {D})
  (hE_intersect : ∃ E, line_through B P ∩ side A C = {E})
  (hF_intersect : ∃ F, line_through C P ∩ side A B = {F}) :
  ∃ Q R, parallelogram_in_triangle (triangle_def D E F) (triangle_def A B C) Q R := 
sorry

end exist_parallelogram_inside_triangle_ABC_l697_697080


namespace maximum_rubles_received_l697_697927

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l697_697927


namespace probability_Q_eq_1_l697_697615

open Complex

theorem probability_Q_eq_1 :
  let W := {2 * I, -2 * I, (1 + I) / 2, (-1 + I) / 2, (1 - I) / 2, (-1 - I) / 2, 1, -1}
  let Q := ∏ k in Finset.range 16, Finset.choose W 16
  ∃ (c d q : ℕ), q.prime ∧ c % q ≠ 0 ∧ probability (Q = 1) = (c : ℝ) / (q : ℝ)^d ∧ c + d + q = 65 :=
begin
  sorry
end

end probability_Q_eq_1_l697_697615


namespace painting_time_for_tired_people_l697_697284

theorem painting_time_for_tired_people :
  (∀ (p h : ℕ) (e : ℝ), p = 8 → h = 2 → e = 0.8 →
  let initial_time := 4 in
  let total_efficiency := (p : ℝ) * initial_time in
  let tired_efficiency := (8 : ℝ) * e in
  let required_efficiency := total_efficiency in
  ∃ t : ℝ, (5 : ℝ) * tired_efficiency / (8 : ℝ) * t = required_efficiency / (8 : ℝ) →
  t = 8) :=
begin
  -- or it can be written succinctly to satisfy the syntactic requirement as:
  sorry
end

end painting_time_for_tired_people_l697_697284


namespace valid_outfit_combinations_l697_697012

theorem valid_outfit_combinations :
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  total_combinations - invalid_combinations = 205 :=
by
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  have h : total_combinations - invalid_combinations = 205 := sorry
  exact h

end valid_outfit_combinations_l697_697012


namespace find_AD_l697_697451

noncomputable def AD (AB AC BD_ratio CD_ratio : ℝ) : ℝ :=
  if h = sqrt(128) then h else 0

theorem find_AD 
  (AB AC : ℝ) 
  (BD_ratio CD_ratio : ℝ)
  (h : ℝ)
  (h_eq : h = sqrt 128) :
  AB = 13 → 
  AC = 20 → 
  BD_ratio / CD_ratio = 3 / 4 → 
  AD AB AC BD_ratio CD_ratio = 8 * Real.sqrt 2 :=
by
  intro h_eq AB_eq AC_eq ratio_eq
  simp only [AD, h_eq, AB_eq, AC_eq, ratio_eq]
  sorry

end find_AD_l697_697451


namespace austin_initial_amount_l697_697657

-- Definitions for conditions
def num_robots := 7
def cost_per_robot := 8.75
def tax := 7.22
def change_left := 11.53

-- Calculation of total spent
def total_cost_robots := num_robots * cost_per_robot
def total_cost_with_tax := total_cost_robots + tax
def total_amount := total_cost_with_tax + change_left

-- Theorem to prove the initial amount Austin started with
theorem austin_initial_amount : total_amount = 80.00 := by
  sorry

end austin_initial_amount_l697_697657


namespace solve_for_k_l697_697018

theorem solve_for_k (k : ℕ) (h : 16 / k = 4) : k = 4 :=
sorry

end solve_for_k_l697_697018


namespace Petya_rubles_maximum_l697_697954

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l697_697954


namespace student_D_score_l697_697043

structure StudentAnswers :=
  (q1 : Bool) (q2 : Bool) (q3 : Bool)
  (q4 : Bool) (q5 : Bool) (q6 : Bool)
  (q7 : Bool) (q8 : Bool)

def total_score (ans : StudentAnswers) (correct_ans : StudentAnswers) : ℕ :=
  (if ans.q1 = correct_ans.q1 then 5 else 0) +
  (if ans.q2 = correct_ans.q2 then 5 else 0) +
  (if ans.q3 = correct_ans.q3 then 5 else 0) +
  (if ans.q4 = correct_ans.q4 then 5 else 0) +
  (if ans.q5 = correct_ans.q5 then 5 else 0) +
  (if ans.q6 = correct_ans.q6 then 5 else 0) +
  (if ans.q7 = correct_ans.q7 then 5 else 0) +
  (if ans.q8 = correct_ans.q8 then 5 else 0)

theorem student_D_score :
  let correct_answers := ⟨false, false, false, true, true, false, true, false⟩
  let answers_D := ⟨false, true, false, true, true, false, true, true⟩
  total_score answers_D correct_answers = 30 :=
by {
  intro1 correct_answers;
  intro1 answers_D;
  sorry
}

end student_D_score_l697_697043


namespace semi_minor_axis_of_ellipse_l697_697039

-- Define the coordinates of the points involved
def center : ℝ × ℝ := (2, -1)
def focus : ℝ × ℝ := (2, -4)
def endpoint_semi_major : ℝ × ℝ := (2, 3)

-- Function to calculate Euclidean distance between two points in R^2
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Calculate c as the distance between the center and the focus
def c : ℝ := distance center focus

-- Calculate a as the distance between the center and the endpoint of the semi-major axis
def a : ℝ := distance center endpoint_semi_major

-- Calculate the semi-minor axis b using the relationship b = sqrt(a^2 - c^2)
def b : ℝ := real.sqrt (a^2 - c^2)

theorem semi_minor_axis_of_ellipse :
  b = real.sqrt 7 :=
by
  -- Skipping the proof for now
  sorry

end semi_minor_axis_of_ellipse_l697_697039


namespace train_speed_l697_697639

theorem train_speed (length : ℕ) (time : ℕ) (h_length : length = 1200) (h_time : time = 15) :
  (length / time) = 80 := by
  sorry

end train_speed_l697_697639


namespace partition_into_districts_l697_697425

open Mathlib

variables (City : Type) (Road : City → City → Prop) (n : ℕ)

-- Conditions
axiom no_three_mutual (c1 c2 c3 : City) : (Road c1 c2) → (Road c2 c3) → (Road c1 c3) → false

axiom at_least_two_roads (r : ℕ) (r > 0) (H : {c : City // ∃ (rs : finset (City × City)), rs.card = r ∧ ∀ ⟨c1, c2⟩ ∈ rs, Road c1 c2}) :
  ∃ c' : City, (∃ rs : finset (City × City), rs.card = r ∧ ∃ c2, (c', c2) ∈ rs) ∧ (∃ c3, (c', c3) ∈ rs)

-- Goal
theorem partition_into_districts : 
  ∃ (districts : City → ℕ), (∀ c1 c2 : City, Road c1 c2 → districts c1 ≠ districts c2) ∧ (finset.univ.card (finset.image districts (finset.univ)) = n) :=
sorry

end partition_into_districts_l697_697425


namespace sale_prices_correct_l697_697636

-- Define the cost prices and profit percentages
def cost_price_A : ℕ := 320
def profit_percentage_A : ℕ := 50

def cost_price_B : ℕ := 480
def profit_percentage_B : ℕ := 70

def cost_price_C : ℕ := 600
def profit_percentage_C : ℕ := 40

-- Define the expected sale prices
def sale_price_A : ℕ := 480
def sale_price_B : ℕ := 816
def sale_price_C : ℕ := 840

-- Define a function to compute sale price
def compute_sale_price (cost_price : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost_price + (profit_percentage * cost_price) / 100

-- The proof statement
theorem sale_prices_correct :
  compute_sale_price cost_price_A profit_percentage_A = sale_price_A ∧
  compute_sale_price cost_price_B profit_percentage_B = sale_price_B ∧
  compute_sale_price cost_price_C profit_percentage_C = sale_price_C :=
by {
  sorry
}

end sale_prices_correct_l697_697636


namespace analytical_expression_value_of_ff_neg3_l697_697387

-- Define the function
def f (x a b : ℝ) : ℝ := x / (a * x + b)

-- Given: a ≠ 0
axiom a_neq_zero (a b : ℝ) : a ≠ 0

-- Given: f(4) = 4/3
axiom f_of_4 (a b : ℝ) : f 4 a b = 4 / 3

-- Given: f(x) = x has a unique solution
axiom unique_solution (a b : ℝ) : (∃! x, f x a b = x)

-- Proving analytical expression 
theorem analytical_expression (a b : ℝ) (h₁ : a ≠ 0) (h₂ : f 4 a b = 4 / 3) (h₃ : ∃! x, f x a b = x) : 
  ∀ x, f x (1/2) 1 = 2 * x / (x + 2) :=
sorry

-- Proving value of f[f(-3)]
theorem value_of_ff_neg3 : f (f (-3 1/2 1)) (1/2) 1 = 3 / 2 :=
sorry

end analytical_expression_value_of_ff_neg3_l697_697387


namespace find_a_b_sum_l697_697144

theorem find_a_b_sum (a b : ℕ) (h1 : 830 - (400 + 10 * a + 7) = 300 + 10 * b + 4)
    (h2 : ∃ k : ℕ, 300 + 10 * b + 4 = 7 * k) : a + b = 2 :=
by
  sorry

end find_a_b_sum_l697_697144


namespace part1_sales_volume_part2_selling_price_for_profit_part3_maximizes_profit_l697_697199

section
-- Definitions based on conditions
def cost_per_kg := 30
def initial_selling_price := 40
def initial_sales := 400
def decrease_rate_per_increase := 10

-- Part (1) Lean 4 Statement
theorem part1_sales_volume (selling_price : ℕ) (h : selling_price = 45) : 
  let decrease_amount := (selling_price - initial_selling_price) * decrease_rate_per_increase
  let sales_volume := initial_sales - decrease_amount
  sales_volume = 350 := 
by
  sorry

-- Part (2) Lean 4 Statement
theorem part2_selling_price_for_profit 
  (monthly_profit : ℕ) (h : monthly_profit = 5250) 
  (selling_price1 selling_price2 : ℕ) 
  : let profit_per_kg (x : ℕ) := x - 30
  let sales_volume (x : ℕ) := 400 - 10 * (x - 40)
  let profit (x : ℕ) := profit_per_kg x * sales_volume x
  profit selling_price1 = monthly_profit ∧ profit selling_price2 = monthly_profit ∧ 
  (selling_price1 = 45 ∨ selling_price1 = 65) ∧ (selling_price2 = 45 ∨ selling_price2 = 65) := 
by 
  sorry

-- Part (3) Lean 4 Statement
theorem part3_maximizes_profit (selling_price : ℕ) (maximum_profit : ℕ) 
  : let profit_per_kg (m : ℕ) := m - 30
  let sales_volume (m : ℕ) := 400 - 10 * (m - 40)
  let profit (m : ℕ) := profit_per_kg m * sales_volume m
  let vertex := 55
  let max_profit := -10 * (vertex - 55)^2 + 6250
  (selling_price = vertex ∧ maximum_profit = max_profit) :=
by 
  sorry

end

end part1_sales_volume_part2_selling_price_for_profit_part3_maximizes_profit_l697_697199


namespace a_n_arithmetic_sum_c_n_l697_697779

-- Definitions based directly on the conditions
def is_arithmetic (a_n : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a_n (n + 1) - a_n n = d

def is_geometric (b_n : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, b_n (n + 1) = b_n n * q

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def a_n (b_n : ℕ → ℝ) : ℕ → ℝ := λ n, log_base_2 (b_n n) + 2

def c_n (a_n : ℕ → ℝ) : ℕ → ℝ := λ n, 1 / (a_n n * a_n (n + 1))

noncomputable def S_n (c_n : ℕ → ℝ) : ℕ → ℝ := λ n, (Finset.range n).sum c_n

-- Problem 1: Prove that a_n is an arithmetic sequence
theorem a_n_arithmetic (b_n : ℕ → ℝ) (q : ℝ) 
  (h_geometric : is_geometric b_n q)
  (h_b3b5_sum : b_n 3 + b_n 5 = 40)
  (h_b3b5_product : b_n 3 * b_n 5 = 256) :
  is_arithmetic (a_n b_n) := sorry

-- Problem 2: Prove the sum of the first n terms of c_n
theorem sum_c_n (b_n : ℕ → ℝ) (q : ℝ)
  (h_geometric : is_geometric b_n q)
  (h_b3b5_sum : b_n 3 + b_n 5 = 40)
  (h_b3b5_product : b_n 3 * b_n 5 = 256)
  (n : ℕ) :
  S_n (c_n (a_n b_n)) n = n / (3 * n + 9) := sorry

end a_n_arithmetic_sum_c_n_l697_697779


namespace chickens_egg_production_l697_697973

/--
Roberto buys 4 chickens for $20 each. The chickens cost $1 in total per week to feed.
Roberto used to buy 1 dozen eggs (12 eggs) a week, spending $2 per dozen.
After 81 weeks, the total cost of raising chickens will be cheaper than buying the eggs.
Prove that each chicken produces 3 eggs per week.
-/
theorem chickens_egg_production:
  let chicken_cost := 20
  let num_chickens := 4
  let weekly_feed_cost := 1
  let weekly_eggs_cost := 2
  let dozen_eggs := 12
  let weeks := 81

  -- Cost calculations
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weekly_feed_cost * weeks
  let total_raising_cost := total_chicken_cost + total_feed_cost
  let total_buying_cost := weekly_eggs_cost * weeks

  -- Ensure cost condition
  (total_raising_cost <= total_buying_cost) →
  
  -- Egg production calculation
  (dozen_eggs / num_chickens) = 3 :=
by
  intros
  sorry

end chickens_egg_production_l697_697973


namespace binary_digit_difference_l697_697176

theorem binary_digit_difference (n1 n2 : ℕ) (h1 : n1 = 300) (h2 : n2 = 1400) : 
  (nat.bit_length n2 - nat.bit_length n1) = 2 := by
  sorry

end binary_digit_difference_l697_697176


namespace k_value_for_passing_through_left_focus_k_range_for_common_points_l697_697814

-- Defining the line and ellipse
def line (k : ℝ) (x : ℝ) : ℝ :=
  k * x + 4

def ellipse (x y : ℝ) : Prop :=
  (x^2 / 5) + y^2 = 1

-- Condition: the line passes through the left focus of the ellipse
def passesThroughLeftFocus (k : ℝ) : Prop :=
  line k (-2) = 0

-- Question I: Given the line passes through the left focus of the ellipse, then k = 2
theorem k_value_for_passing_through_left_focus (k : ℝ) : 
  passesThroughLeftFocus k → k = 2 := by
    sorry

-- Condition: the line has common points with the ellipse
def hasCommonPoints (k : ℝ) : Prop :=
  ∀ (x y : ℝ), (ellipse x y) ∧ (y = line k x)

-- Question II: Given the line has common points with the ellipse, determine the range of k
theorem k_range_for_common_points (k : ℝ) : 
  hasCommonPoints k → (k ≤ -real.sqrt 3 ∨ k ≥ real.sqrt 3) := by
    sorry

end k_value_for_passing_through_left_focus_k_range_for_common_points_l697_697814


namespace find_ab_l697_697052

theorem find_ab (a b : ℝ) (h1 : a - b = 26) (h2 : a + b = 15) :
  a = 41 / 2 ∧ b = 11 / 2 :=
sorry

end find_ab_l697_697052


namespace wrapping_paper_area_l697_697634

theorem wrapping_paper_area (p w h : ℝ) : 0 ≤ p → 0 ≤ w → 0 ≤ h → ∃ (A : ℝ), A = 4 * w * (p + h) :=
by
  intros hp hw hh
  use 4 * w * (p + h)
  sorry

end wrapping_paper_area_l697_697634


namespace team_a_completion_rate_l697_697861

theorem team_a_completion_rate :
  ∃ x : ℝ, (9000 / x - 9000 / (1.5 * x) = 15) ∧ x = 200 :=
by {
  sorry
}

end team_a_completion_rate_l697_697861


namespace y_increase_for_x_increase_l697_697358

theorem y_increase_for_x_increase (x y : ℝ) (h : 4 * y = 9) : 12 * y = 27 :=
by
  sorry

end y_increase_for_x_increase_l697_697358


namespace common_value_of_4a_and_5b_l697_697409

theorem common_value_of_4a_and_5b (a b C : ℝ) (h1 : 4 * a = C) (h2 : 5 * b = C) (h3 : 40 * a * b = 1800) :
  C = 60 :=
sorry

end common_value_of_4a_and_5b_l697_697409


namespace cylinder_height_in_hemisphere_l697_697218

noncomputable def height_of_cylinder (r_hemisphere r_cylinder : ℝ) : ℝ :=
  real.sqrt (r_hemisphere^2 - r_cylinder^2)

theorem cylinder_height_in_hemisphere :
  let r_hemisphere := 8
  let r_cylinder := 3
  height_of_cylinder r_hemisphere r_cylinder = real.sqrt 55 :=
by
  sorry

end cylinder_height_in_hemisphere_l697_697218


namespace smallest_non_palindromic_power_of_13_l697_697730

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l697_697730


namespace total_percent_sampling_l697_697424

theorem total_percent_sampling 
  (total_customers : ℝ)
  (percent_caught : ℝ)
  (percent_not_caught_of_total_samplers : ℝ) :
  percent_caught = 22 →
  percent_not_caught_of_total_samplers = 12 →
  total_customers = 100 →
  let percent_samplers := 25 in
  (percent_caught + 0.12 * percent_samplers = percent_samplers) := 
by
  intros h_caught h_not_caught h_total
  let percent_samplers := 25
  have h1 : percent_caught + 0.12 * percent_samplers = percent_samplers, from sorry
  exact h1

end total_percent_sampling_l697_697424


namespace eccentricity_proof_l697_697813

-- Define the parameters and assumptions for the hyperbola
variables {a b c λ μ : ℝ}
variable (F : ℝ × ℝ)
variable (A B P : ℝ × ℝ)

-- Given conditions and parameters
noncomputable def hyperbola_equation : Prop := ∀ x y, (x^2) / (a^2) - (y^2) / (b^2) = 1 ∧ a > 0 ∧ b > 0
def right_focus : Prop := F = (c, 0)
def line_through_focus : Prop := ∃ A B, (A = (c, b * c / a)) ∧ (B = (c, -b * c / a))
def point_in_first_quadrant : Prop := P = (c, b^2 / a) ∧ c > 0 ∧ b > 0

-- Linear combination and λ^2 + μ^2 = 5/8
def linear_combination : Prop := ∃ λ μ, λ^2 + μ^2 = 5 / 8

-- The proof problem
theorem eccentricity_proof (h1 : hyperbola_equation) (h2 : right_focus) (h3 : line_through_focus) (h4 : point_in_first_quadrant) (h5 : linear_combination) :
  let e := c / a in e = 2 * real.sqrt 3 / 3 :=
sorry

end eccentricity_proof_l697_697813


namespace problem_solution_l697_697087

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ m, m ∣ n).card

noncomputable def solve_problem : set ℕ :=
  let non_primes := (finset.range 21).filter (λ n, ¬ is_prime n) in
  let max_divisors_count := non_primes.sup num_divisors in
  non_primes.filter (λ n, num_divisors n = max_divisors_count).to_set

theorem problem_solution : solve_problem = {12, 18, 20} :=
  sorry

end problem_solution_l697_697087


namespace sample_size_is_80_l697_697977

-- Define the given conditions
variables (x : ℕ) (numA numB numC n : ℕ)

-- Conditions in Lean
def ratio_condition (x numA numB numC : ℕ) : Prop :=
  numA = 2 * x ∧ numB = 3 * x ∧ numC = 5 * x

def sample_condition (numA : ℕ) : Prop :=
  numA = 16

-- Definition of the proof problem
theorem sample_size_is_80 (x : ℕ) (numA numB numC n : ℕ)
  (h_ratio : ratio_condition x numA numB numC)
  (h_sample : sample_condition numA) : 
  n = 80 :=
by
-- The proof is omitted, just state the theorem
sorry

end sample_size_is_80_l697_697977


namespace relationship_between_a_b_c_d_l697_697407

-- Define the constants
def a : ℝ := 10 / 7
def b : ℝ := Real.log 3
def c : ℝ := 2 * Real.sqrt 3 / 3
def d : ℝ := Real.exp 0.3

theorem relationship_between_a_b_c_d :
  a > d ∧ d > c ∧ c > b :=
by 
  sorry

end relationship_between_a_b_c_d_l697_697407


namespace divisors_of_square_of_cube_of_prime_l697_697617

theorem divisors_of_square_of_cube_of_prime (p : ℕ) (hp : p.prime) (n : ℕ) (h : n = p^3) :
  nat.num_divisors (n^2) = 7 :=
sorry

end divisors_of_square_of_cube_of_prime_l697_697617


namespace find_special_three_digit_numbers_l697_697298

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l697_697298


namespace trapezoid_to_parallelogram_l697_697267

theorem trapezoid_to_parallelogram
  (A B C D M : ℝ × ℝ)
  (h : ℝ)
  (hAB : A.1 < B.1)
  (hCD : C.1 < D.1)
  (h_height : A.2 = C.2 ∧ B.2 = D.2 ∧ A.2=h):
  ∃ (P Q R S : ℝ × ℝ), parallelogram P Q R S ∧ rearrange_to_parallelogram A B C D M P Q R S :=
sorry

end trapezoid_to_parallelogram_l697_697267


namespace octagon_side_length_l697_697048

/--
In rectangle PQRS, PQ=8 and QR=6.
Points A and B lie on PQ, points C and D lie on QR, points E and F lie on RS, and points G and H lie on SP so that AP = BQ < 4 and the convex octagon ABCDEFGH is equilateral.
The length of a side of this octagon can be expressed in the form k + m * sqrt(n), where k, m, and n are integers and n is not divisible by the square of any prime.
Prove that k + m + n = 7.
-/
theorem octagon_side_length {PQ QR : ℝ} (hPQ : PQ = 8) (hQR : QR = 6)
  (A B : ℝ → Prop) (C D : ℝ → Prop) (E F : ℝ → Prop) (G H : ℝ → Prop)
  (hAB : ∀ x, A x → B x → x < 4) (hEquilateral : ∀ x, (x ∈ {A, B, C, D, E, F, G, H}) → x = -7 + 3 * (Real.sqrt 11)) : 
  (-7) + 3 + 11 = 7 :=
by
  sorry

end octagon_side_length_l697_697048


namespace find_number_l697_697027

theorem find_number (x : ℤ) (h : 3 * x + 4 = 19) : x = 5 :=
by {
  sorry
}

end find_number_l697_697027


namespace min_detectors_correct_l697_697153

noncomputable def min_detectors (M N : ℕ) : ℕ :=
  ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊

theorem min_detectors_correct (M N : ℕ) (hM : 2 ≤ M) (hN : 2 ≤ N) :
  min_detectors M N = ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊ :=
by {
  -- The proof goes here
  sorry
}

end min_detectors_correct_l697_697153


namespace angle_between_chord_and_radius_extension_l697_697527

theorem angle_between_chord_and_radius_extension (O A B : Point) (r : ℝ)
  (h1 : is_center O)
  (h2 : on_circle A O r)
  (h3 : on_circle B O r)
  (h4 : ∠(O, A, B) = 110) :
  ∠(A, B, 180) - ∠(O, A, B) = 145 := 
sorry

end angle_between_chord_and_radius_extension_l697_697527


namespace centroid_condition_l697_697037

-- Define the 4030 x 4030 grid and the selection of lines 
def grid_size : ℕ := 4030

def horizontal_lines : Finset ℤ := {n | -2015 ≤ n ∧ n ≤ 2015}
def vertical_lines : Finset ℤ := {n | -2015 ≤ n ∧ n ≤ 2015}

def intersection_points (h_lines : Finset ℤ) (v_lines : Finset ℤ) : Finset (ℤ × ℤ) := 
  (h_lines.product v_lines)

-- Define the centroid calculation
noncomputable def centroid (points : Finset (ℤ × ℤ)) : ℤ × ℤ :=
  let x_sum := points.sum (λ p, p.1)
  let y_sum := points.sum (λ p, p.2)
  let n := points.card
  (x_sum / n, y_sum / n)

-- The main theorem statement
theorem centroid_condition :
  ∀ (h_lines : Finset ℤ) (v_lines : Finset ℤ),
  horizontal_lines ⊆ h_lines →
  vertical_lines ⊆ v_lines →
  h_lines.card = 2017 →
  v_lines.card = 2017 →
  ∃ (A B C D E F : ℤ × ℤ),
    A ∈ intersection_points(h_lines, v_lines) ∧
    B ∈ intersection_points(h_lines, v_lines) ∧
    C ∈ intersection_points(h_lines, v_lines) ∧
    D ∈ intersection_points(h_lines, v_lines) ∧
    E ∈ intersection_points(h_lines, v_lines) ∧
    F ∈ intersection_points(h_lines, v_lines) ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    centroid {A, B, C} = (0, 0) ∧
    centroid {D, E, F} = (0, 0) :=
sorry

end centroid_condition_l697_697037


namespace x_varies_as_nth_power_of_z_l697_697410

theorem x_varies_as_nth_power_of_z 
  (k j z : ℝ) 
  (h1 : ∃ y : ℝ, x = k * y^4 ∧ y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := 
 sorry

end x_varies_as_nth_power_of_z_l697_697410


namespace find_amount_spent_on_shirt_l697_697514

variable (S : ℝ)
variable (shorts : ℝ)
variable (returned : ℝ)
variable (net_spend : ℝ)

-- Given conditions
def conditions := shorts = 13.99 ∧ returned = 7.43 ∧ net_spend = 18.7

-- We need to prove that the amount spent on the shirt is 12.14
def amount_spent_on_shirt (S : ℝ) (shorts : ℝ) (returned : ℝ) (net_spend : ℝ) : Prop :=
  shorts + S - returned = net_spend

theorem find_amount_spent_on_shirt : 
  conditions → amount_spent_on_shirt S 13.99 7.43 18.7 → S = 12.14 := by
  sorry

end find_amount_spent_on_shirt_l697_697514


namespace exists_large_suitable_set_l697_697578

open Finset

noncomputable def is_suitable (S : Finset ℕ) : Prop :=
  ∀ a ∈ S, gcd a (S.sum id) ≠ 1

theorem exists_large_suitable_set (ε : ℝ) (hε : 0 < ε ∧ ε < 1) :
  ∃ N₀ : ℕ, ∀ N ≥ N₀, ∃ S : Finset ℕ,
    S ⊆ range (N + 1) ∧
    S.card ≥ ⌊ε * N⌋ ∧
    is_suitable S :=
by { sorry }

end exists_large_suitable_set_l697_697578


namespace six_digit_increasing_check_six_digit_increasing_l697_697071

theorem six_digit_increasing (N : ℕ) : 
  N = number_of_six_digit_increasing_nonstarting_with_six 1 2 3 4 5 6 := by 
  sorry

/--
Define the number of six-digit integers where digits are in increasing order 
and do not start with the digit 6.
-/
def number_of_six_digit_increasing_nonstarting_with_six
  (d1 d2 d3 d4 d5 d6 : ℕ) : ℕ :=
  let total := Nat.choose (6 + 5) 5
  let exclude6 := Nat.choose (5 + 4) 4
  total - exclude6

noncomputable def number_of_six_digit_increasing_nonstarting_with_six_value : ℕ :=
  number_of_six_digit_increasing_nonstarting_with_six 1 2 3 4 5 6

theorem check_six_digit_increasing :
    number_of_six_digit_increasing_nonstarting_with_six_value = 336 := by
  sorry

end six_digit_increasing_check_six_digit_increasing_l697_697071


namespace exist_R_inoceronte_l697_697476

open Real

noncomputable def min_R (M α β : ℝ) (hM : 0 < M) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :=
  Inf {R : ℝ | 1 < R ∧
  ∃ (C : ℕ → ℝ) (hC : ∀ n ≥ 1, (∑ i in finset.range n, R^(n-i-1) * C i) ≤ R^n * M) 
    (hdiv : ∑' n : ℕ, β^n * (C n)^α = ∞) }

theorem exist_R_inoceronte (M α β : ℝ) (hM : 0 < M) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  min_R M α β hM hα hβ = β^(-1/α) :=
sorry

end exist_R_inoceronte_l697_697476


namespace quadratic_function_exists_l697_697678

theorem quadratic_function_exists (f : ℝ → ℝ) : 
  (f(1) = 0) ∧ (f(5) = 0) ∧ (f(3) = 8) → 
  ∃ a b c, f = λ x, a * x^2 + b * x + c ∧ a = -2 ∧ b = 12 ∧ c = -10 :=
by 
  intro h,
  use [-2, 12, -10],
  split,
  { rw funext,
    intro x,
    simp },
  exact h,
sorry

end quadratic_function_exists_l697_697678


namespace find_complex_number_l697_697702

theorem find_complex_number (z : ℂ) (h1 : complex.abs (z - 2) = complex.abs (z + 4))
  (h2 : complex.abs (z + 4) = complex.abs (z - 2 * complex.I)) : z = -1 + complex.I :=
by
  sorry

end find_complex_number_l697_697702


namespace find_k_l697_697800

variables {a b : E} {k : ℝ} [inner_product_space ℝ E]

-- Conditions
variables (ha : ∥a∥ = 3) (hb : ∥b∥ = 4) (horth : inner (a + k • b) (a - k • b) = 0)
variables (hnot_collinear : ¬ collinear ℝ ({a, b} : set E))

-- Statement to prove
theorem find_k (ha : ∥a∥ = 3) (hb : ∥b∥ = 4) (horth : inner (a + k • b) (a - k • b) = 0): k = 3/4 ∨ k = -3/4 :=
sorry

end find_k_l697_697800


namespace mary_needs_10_charges_to_vacuum_house_l697_697533

theorem mary_needs_10_charges_to_vacuum_house :
  (let bedroom_time := 10
   let kitchen_time := 12
   let living_room_time := 8
   let dining_room_time := 6
   let office_time := 9
   let bathroom_time := 5
   let battery_duration := 8
   3 * bedroom_time + kitchen_time + living_room_time + dining_room_time + office_time + 2 * bathroom_time) / battery_duration = 10 :=
by sorry

end mary_needs_10_charges_to_vacuum_house_l697_697533


namespace collinear_if_and_only_if_l697_697053

open Real
open EuclideanGeometry

/-- Given a triangle ABC and points P and Q on the external bisector of ∠A such that
    B and P lie on the same side of AC. Perpendiculars from P to AB and from Q to AC 
    intersect at X. Points P' and Q' lie on PB and QC such that PX = P'X and QX = Q'X. 
    Point T is the midpoint of the arc BC (not containing A) of the circumcircle of △ABC.
    Prove that P', Q', and T are collinear if and only if ∠PBA + ∠QCA = 90°. -/
theorem collinear_if_and_only_if (A B C P Q P' Q' X T : Point) (H1 : lies_on_external_bisector A P Q)
  (H2 : same_side B P AC) (H3 : perpendicular (line_through P A) (line_through P B))
  (H4 : perpendicular (line_through Q A) (line_through Q C)) (H5 : PX = P'X) (H6 : QX = Q'X)
  (H7 : T = midpoint_arc B C (triangle_circumcircle A B C) A) :
  (P', Q', T are collinear ↔ ∠(line_through P B) (line_through Q C) = 90°) :=
sorry

end collinear_if_and_only_if_l697_697053


namespace calculate_A_plus_B_l697_697076

theorem calculate_A_plus_B (A B : ℝ) (h1 : A ≠ B) 
  (h2 : ∀ x : ℝ, (A * (B * x^2 + A * x + 1)^2 + B * (B * x^2 + A * x + 1) + 1) 
                - (B * (A * x^2 + B * x + 1)^2 + A * (A * x^2 + B * x + 1) + 1) 
                = x^4 + 5 * x^3 + x^2 - 4 * x) : A + B = 0 :=
by
  sorry

end calculate_A_plus_B_l697_697076


namespace monotonicity_intervals_f_le_g_l697_697390

section Problem1
variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (x^2 + x)
noncomputable def F (x : ℝ) : ℝ := f x - g x

theorem monotonicity_intervals (h1 : 0 < x) : 
  (0 < x ∧ x < 2 → F x > F (x - 1)) ∧ 
  (x > 2 → F x < F (x + 1)) := 
sorry
end Problem1

section Problem2
variable {x : ℝ} {a : ℝ} (h2 : 0 < x) (h3 : 1 ≤ a)

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x
noncomputable def g (x : ℝ) : ℝ := a * (x^2 + x)

theorem f_le_g : f x ≤ g x :=
sorry
end Problem2

end monotonicity_intervals_f_le_g_l697_697390


namespace blue_marbles_initial_count_l697_697400

variables (x y : ℕ)

theorem blue_marbles_initial_count (h1 : 5 * x = 8 * y) (h2 : 3 * (x - 12) = y + 21) : x = 24 :=
sorry

end blue_marbles_initial_count_l697_697400


namespace no_positive_integer_solutions_l697_697520

-- Definition of the gcd
def gcd (x y : ℕ) : ℕ := Nat.gcd x y

-- The problem statement
theorem no_positive_integer_solutions (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 → 
  gcd (a^2) (b^2) + gcd a (b * c) + gcd b (a * c) + gcd c (a * b) ≠ 199 := 
by 
  intros a_positive b_positive c_positive
  sorry

end no_positive_integer_solutions_l697_697520


namespace surface_area_hemisphere_l697_697590

theorem surface_area_hemisphere
  (r : ℝ)
  (h₁ : 4 * Real.pi * r^2 = 4 * Real.pi * r^2)
  (h₂ : Real.pi * r^2 = 3) :
  3 * Real.pi * r^2 = 9 :=
by
  sorry

end surface_area_hemisphere_l697_697590


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697739

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l697_697739


namespace sin_polar_curve_intersecting_lines_l697_697536

theorem sin_polar_curve_intersecting_lines : 
  (∃ ρ : ℝ, sin θ = sqrt 2 / 2) → 
  (∀ θ, sin θ = sqrt 2 / 2 → (tan θ = 1 ∨ tan θ = -1) → ∃ x: ℝ, ∃ y: ℝ, y = x ∨ y = -x) :=
by
  sorry

end sin_polar_curve_intersecting_lines_l697_697536


namespace find_m_range_l697_697389

def f (m : ℝ) : ℝ → ℝ :=
λ x, if x ∈ Ioo (-1 : ℝ) 1 then m * real.sqrt (1 - x^2)
     else if x ∈ Ioo 1 3 then 1 - |x - 2|
     else 0

theorem find_m_range : 
  (∃! (x : ℝ), 3 * f m x = x) ∧  5 ∈ { x : ℝ | ∃! m, (3 * f m x = x ∧ m > 0) } → 
  m ∈ set.Ioo (real.sqrt 15 / 3) (real.sqrt 7) :=
sorry

end find_m_range_l697_697389


namespace maximum_rubles_received_l697_697930

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l697_697930


namespace sum_of_squares_le_largest_l697_697140

-- Given several positive numbers whose sum is 1, prove there is a number among them
-- that is no less than the sum of the squares of all the numbers.
theorem sum_of_squares_le_largest (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) (h_sum : (Finset.univ.sum a) = 1) :
  ∃ A ∈ (Finset.univ.image a), A ≥ (Finset.univ.sum (λ i, (a i) ^ 2)) :=
by
  sorry

end sum_of_squares_le_largest_l697_697140


namespace hyperbola_center_foci_l697_697319

theorem hyperbola_center_foci (x y : ℝ) :
  (∃ (k₁ k₂ : ℝ), (k₁ = -1) ∧ (k₂ = 3/2) ∧
  {c₁ : ℝ | c₁ = k₁ + sqrt(101/45)} ∪ {c₁ : ℝ | c₁ = k₁ - sqrt(101/45)} ∧
  ((sqrt(5) * y + 5)^2 / 9) - ((6 * x - 9)^2 / 16) = 1) :=
begin
  sorry
end

end hyperbola_center_foci_l697_697319


namespace num_values_a_satisfying_conditions_l697_697848

open Classical

variable {x a : ℤ}

theorem num_values_a_satisfying_conditions :
  (∀ x : ℤ, 2 * (x + 1) < x + 3 → x < 1) →
  (∀ x : ℤ, x - a ≤ a + 5 → x < 1) →
  (∃ a : ℤ, a ≤ 0) →
  (Nat.card' { a : ℤ | a ≤ 0 ∧ x < 1 → x ≤ 2 * a + 5 ∧ 2 * (x + 1) < x + 3 } = 3) :=
sorry

end num_values_a_satisfying_conditions_l697_697848


namespace total_interest_l697_697652

/-- Angus invested $18,000, with $6000 at 5% and the remaining amount at 3%.
    Prove that the total interest at the end of the year is $660. -/
theorem total_interest (total_investment : ℝ) (part1_investment : ℝ) (rate1 : ℝ) (rate2 : ℝ)
  (h_total : total_investment = 18000) (h_part1 : part1_investment = 6000)
  (h_rate1 : rate1 = 0.05) (h_rate2 : rate2 = 0.03) :
  let part2_investment := total_investment - part1_investment in
  let interest1 := part1_investment * rate1 in
  let interest2 := part2_investment * rate2 in
  interest1 + interest2 = 660 := 
by
  sorry

end total_interest_l697_697652


namespace distance_covered_at_40_kmph_l697_697587

theorem distance_covered_at_40_kmph (x : ℝ) 
  (h₁ : x / 40 + (250 - x) / 60 = 5.5) :
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l697_697587


namespace complex_number_solution_l697_697694

open Complex

noncomputable def findComplexNumber (z : ℂ) : Prop :=
  abs (z - 2) = abs (z + 4) ∧ abs (z - 2) = abs (z - Complex.I * 2)

theorem complex_number_solution :
  ∃ z : ℂ, findComplexNumber z ∧ z = -1 + Complex.I :=
by
  sorry

end complex_number_solution_l697_697694


namespace tan_sum_l697_697466

theorem tan_sum (A C B : ℝ) (h1 : Real.cot A * Real.cot C = 1 / 3) (h2 : Real.cot B * Real.cot C = 1 / 12) (h3 : C = Real.pi / 4)
  : Real.tan A + Real.tan B = 15 := 
sorry

end tan_sum_l697_697466


namespace intersection_points_x_axis_vertex_on_line_inequality_c_l697_697049

section
variable {r : ℝ}
def quadratic_function (x m : ℝ) : ℝ := -0.5 * (x - 2*m)^2 + 3 - m

theorem intersection_points_x_axis (m : ℝ) (h : m = 2) : 
  ∃ x1 x2 : ℝ, quadratic_function x1 m = 0 ∧ quadratic_function x2 m = 0 ∧ x1 ≠ x2 :=
by
  sorry

theorem vertex_on_line (m : ℝ) (h : true) : 
  ∀ m : ℝ, (2*m, 3-m) ∈ {p : ℝ × ℝ | p.2 = -0.5 * p.1 + 3} :=
by
  sorry

theorem inequality_c (a c m : ℝ) (hP : quadratic_function (a+1) m = c) (hQ : quadratic_function ((4*m-5)+a) m = c) : 
  c ≤ 13/8 :=
by
  sorry
end

end intersection_points_x_axis_vertex_on_line_inequality_c_l697_697049


namespace find_three_digit_numbers_l697_697318

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l697_697318


namespace find_AE_length_l697_697260

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 4⟩
def B : Point := ⟨6, 0⟩
def C : Point := ⟨5, 3⟩
def D : Point := ⟨3, 0⟩

def ratio : ℝ := 5 / 8

def E : Point := ⟨(5 * 3 / 8), (3 / 2)⟩ -- Dummy placeholder, find correct intersection

def AE_length : ℝ :=
  distance (A.x, A.y) (E.x, E.y)

theorem find_AE_length : AE_length = 5*real.sqrt(13)/4 :=
  sorry

end find_AE_length_l697_697260


namespace smallest_proportional_part_l697_697638

theorem smallest_proportional_part (total parts : ℕ) (h1 : total = 120)
  (h2 : parts = [3, 5, 7]) : ∃ smallest_part, smallest_part = 24 := by
  have parts_sum : List.sum parts = 15 := by
    exact rfl
  have each_part_total := h1 ▸ have parts_sum : List.sum parts := 15 by rfl
    15 * x = 120
  have x_value : x = 8 := by
    exact rfl
  have parts_values := List.map (· * x) parts
  have parts_values = [24, 40, 56] := by
    exact rfl
  have smallest_part_val : smallest_part = min 24 (min 40 56) := by
    exact rfl
  exact 24

end smallest_proportional_part_l697_697638


namespace max_rubles_l697_697966

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l697_697966


namespace frosting_time_difference_l697_697503

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end frosting_time_difference_l697_697503


namespace angle_YZX_25_degrees_l697_697055

theorem angle_YZX_25_degrees
    (O X Y Z : Type)
    [triangle XYZ]
    (tangent_O : ∀ P : Type, is_tangent (lines_through P) O)
    (angle_XYZ_eq_75 : angle XYZ = 75)
    (angle_YXO_eq_40 : angle YXO = 40) : angle YZX = 25 := by
  sorry

end angle_YZX_25_degrees_l697_697055


namespace proof_problem_l697_697859

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

def is_acute_triangle (A B C : ℝ) : Prop := A < π / 2 ∧ B < π / 2 ∧ C < π / 2

def opposite_sides (A B C a b c : ℝ) : Prop := -- some definition to link angles with sides

def vector_m (a b : ℝ) : ℝ × ℝ := (1 / a, 1 / b)
def vector_n (a b : ℝ) : ℝ × ℝ := (b, a)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem proof_problem :
  is_acute_triangle A B C →
  opposite_sides A B C a b c →
  let m := vector_m a b
  let n := vector_n a b
  dot_product m n = 6 * Real.cos C →
  (Real.tan C / Real.tan A + Real.tan C / Real.tan B) = 4 := by
  sorry

end proof_problem_l697_697859


namespace ratio_of_sums_l697_697788

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_first_n_terms (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Given conditions
variables (a1 d : ℝ) (h_d_ne_zero : d ≠ 0)
variable (h_condition : a1 + 7 * d = 2 * (a1 + 2 * d))

-- Theorem to prove the required ratio
theorem ratio_of_sums (h_d_ne_zero : d ≠ 0) (h_condition : a1 + 7 * d = 2 * (a1 + 2 * d)) :
  (sum_of_first_n_terms a1 d 15) / (sum_of_first_n_terms a1 d 5) = 6 :=
by
  sorry

end ratio_of_sums_l697_697788


namespace length_of_room_is_5_5_l697_697999

-- The width of the room
def width : ℝ := 3.75

-- The rate for paving the floor in Rs. per sq. meter
def rate : ℝ := 600

-- The total cost for paving the floor in Rs.
def total_cost : ℝ := 12375

-- Define length of the room
def length : ℝ := (total_cost / rate) / width

-- The main theorem we want to prove
theorem length_of_room_is_5_5 : length = 5.5 :=
by
  sorry

end length_of_room_is_5_5_l697_697999


namespace values_of_a_plus_b_l697_697017

theorem values_of_a_plus_b (a b : ℝ) (h1 : abs (-a) = abs (-1)) (h2 : b^2 = 9) (h3 : abs (a - b) = b - a) : a + b = 2 ∨ a + b = 4 := 
by 
  sorry

end values_of_a_plus_b_l697_697017


namespace train_pass_tree_time_eq_19_l697_697641

/-- Define the length of the train in meters. -/
def train_length : ℝ := 285

/-- Define the speed of the train in km/hr. -/
def train_speed_km_per_hr : ℝ := 54

/-- Convert the train speed to m/s. -/
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

/-- Define the time it takes for the train to pass the tree in seconds. -/
def time_to_pass_tree : ℝ := train_length / train_speed_m_per_s

/-- Theorem: The train will pass the tree in 19 seconds. -/
theorem train_pass_tree_time_eq_19 : time_to_pass_tree = 19 := by
  sorry

end train_pass_tree_time_eq_19_l697_697641


namespace sum_of_max_min_values_of_ratio_l697_697675

theorem sum_of_max_min_values_of_ratio (x y : ℝ) :
  (3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 48 = 0) → 
  (x < 0 ∧ y > 0) →
  let m := y / x in
  let a := (maximum {m | ∃ x y, 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 48 = 0 ∧ x < 0 ∧ y > 0}) in
  let b := (minimum {m | ∃ x y, 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 48 = 0 ∧ x < 0 ∧ y > 0}) in
  a + b = 10 / 3 :=
sorry

end sum_of_max_min_values_of_ratio_l697_697675


namespace triangle_side_length_l697_697467

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h₁ : a * Real.cos B = b * Real.sin A)
  (h₂ : C = Real.pi / 6) (h₃ : c = 2) : b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l697_697467


namespace probability_same_color_is_19_over_39_l697_697181
-- Step d): Rewrite in Lean 4 statement

def probability_same_color : ℚ :=
  let total_balls := 13
  let green_balls := 5
  let white_balls := 8
  let total_ways := Nat.choose total_balls 2
  let green_ways := Nat.choose green_balls 2
  let white_ways := Nat.choose white_balls 2
  (green_ways + white_ways) / total_ways

theorem probability_same_color_is_19_over_39 :
  probability_same_color = 19 / 39 :=
by
  sorry

end probability_same_color_is_19_over_39_l697_697181


namespace midpoint_of_segment_l697_697165

theorem midpoint_of_segment :
  let p1 := (9 : ℝ, -8 : ℝ)
  let p2 := (-5 : ℝ, 6 : ℝ)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = (2, -1) :=
by
  sorry

end midpoint_of_segment_l697_697165


namespace max_difference_is_89_9375_l697_697613

def modified_iterative_average (seq : List ℕ) : Rational :=
  let first := seq.get? 0 |>.getD 0 + 1
  let second := (seq.get? 1 |>.getD 0)^2
  let third := Nat.factorial (seq.get? 2 |>.getD 0)
  let avg1 := (first + second) / 2
  let avg2 := (avg1 + third) / 2
  let avg3 := (avg2 + (seq.get? 3 |>.getD 0)) / 2
  (avg3 + (seq.get? 4 |>.getD 0)) / 2

def max_min_difference (l : List (List ℕ)) : Rational :=
  let values := l.map modified_iterative_average
  values.maximum.getD 0 - values.minimum.getD 0

def sequences := [[6, 4, 2, 3, 1], [1, 2, 6, 4, 3]]

theorem max_difference_is_89_9375 :
  max_min_difference sequences = 89.9375 := by
  sorry

end max_difference_is_89_9375_l697_697613


namespace min_value_of_expression_l697_697888

-- Define the variables x, y, z as positive real numbers.
variables (x y z : ℝ)
variables (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

-- Define the expression in question
def f (x y z : ℝ) := 8 * x^3 + 12 * y^3 + 50 * z^3 + 1 / (5 * x * y * z)

-- State the problem as a theorem
theorem min_value_of_expression (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    f x y z ≥ 4 * Real.sqrt 3 := sorry

end min_value_of_expression_l697_697888


namespace circle_equation_tangent_and_intersects_l697_697201

theorem circle_equation_tangent_and_intersects
  (C : Type)
  (x y : ℝ → ℝ → ℝ)
  (T : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (hT : T = (1, 0))
  (hAB : A.1 = 0 ∧ B.1 = 0 ∧ (A.2 - B.2).abs = 2)
  (hCT : ∀ {a b : ℝ}, C a b = (a - 1)^2 + (b - sqrt 2)^2 - 2) :
  (C T.1 T.2 = 0 ∧ A.2 > 0 ∧ B.2 > 0) →
  (∀ x' y', C x' y' = (x' - 1)^2 + (y' - sqrt 2)^2 - 2) :=
by
  intro h
  sorry  -- proof goes here

end circle_equation_tangent_and_intersects_l697_697201


namespace tim_income_percentage_less_than_juan_l697_697491

variables (M T J : ℝ)

theorem tim_income_percentage_less_than_juan 
  (h1 : M = 1.60 * T)
  (h2 : M = 0.80 * J) : 
  100 - 100 * (T / J) = 50 :=
by
  sorry

end tim_income_percentage_less_than_juan_l697_697491


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697750

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l697_697750


namespace find_cos_angle_BAC_l697_697372

noncomputable def cos_angle_BAC (A B C : ℝ³) (O : ℝ³) : ℝ :=
  let OA := A - O in
  let OB := B - O in
  let OC := C - O in
  if h : (3 • OA + 4 • OB + 5 • OC = 0) then
    let R := 1 in
    let AB := OB - OA in
    let AC := OC - OA in
    let AB_dot_AC := (OB - OA) • (OC - OA) in
    let len_AB := (OB - OA) • (OB - OA) in
    let len_AC := (OC - OA) • (OC - OA) in
    let cos_BAC := AB_dot_AC / (sqrt len_AB * sqrt len_AC) in
    cos_BAC
  else 0

theorem find_cos_angle_BAC (A B C O : ℝ³)
  (h : 3 • (A - O) + 4 • (B - O) + 5 • (C - O) = 0) :
  cos_angle_BAC A B C O = sqrt 10 / 10 :=
by sorry

end find_cos_angle_BAC_l697_697372


namespace transformed_parabola_equation_l697_697985

theorem transformed_parabola_equation :
    ∀ (x : ℝ), let f := λ x, -2 * x ^ 2 in
    (f (x + 1) - 3) = -2 * x ^ 2 - 4 * x - 5 :=
by
  intro x
  let f := λ x, -2 * x ^ 2
  sorry

end transformed_parabola_equation_l697_697985


namespace problem_remainders_l697_697084

open Int

theorem problem_remainders (x : ℤ) :
  (x + 2) % 45 = 7 →
  ((x + 2) % 20 = 7 ∧ x % 19 = 5) :=
by
  sorry

end problem_remainders_l697_697084


namespace frosting_cupcakes_l697_697252

variable (Cagney_rate : ℕ) (Lacey_rate : ℕ) (time_minutes : ℕ)

def cupcakes_frosted (Cagney_rate : ℕ) (Lacey_rate : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let combined_rate := 1 / (1 / (Cagney_rate : ℚ) + 1 / (Lacey_rate : ℚ))
  (time_seconds / combined_rate).to_nat

theorem frosting_cupcakes : 
  cupcakes_frosted 15 25 8 = 51 :=
by
  noncomputable theory  -- Required due to division
  sorry

end frosting_cupcakes_l697_697252


namespace binom_n_n_minus_1_l697_697580

theorem binom_n_n_minus_1 (n : ℕ) (h : 0 < n) : (Nat.choose n (n-1)) = n :=
  sorry

end binom_n_n_minus_1_l697_697580


namespace discount_percentage_is_10_l697_697631

-- Definition of the conditions
def wholesale_price : ℝ := 90
def profit : ℝ := 0.20 * wholesale_price
def retail_price : ℝ := 120
def selling_price : ℝ := wholesale_price + profit
def discount_amount : ℝ := retail_price - selling_price
def discount_percentage : ℝ := (discount_amount / retail_price) * 100

-- Statement to be proved
theorem discount_percentage_is_10 :
  discount_percentage = 10 := 
sorry

end discount_percentage_is_10_l697_697631


namespace find_AD_l697_697448

noncomputable def AD (AB AC BD_ratio CD_ratio : ℝ) : ℝ :=
  if h = sqrt(128) then h else 0

theorem find_AD 
  (AB AC : ℝ) 
  (BD_ratio CD_ratio : ℝ)
  (h : ℝ)
  (h_eq : h = sqrt 128) :
  AB = 13 → 
  AC = 20 → 
  BD_ratio / CD_ratio = 3 / 4 → 
  AD AB AC BD_ratio CD_ratio = 8 * Real.sqrt 2 :=
by
  intro h_eq AB_eq AC_eq ratio_eq
  simp only [AD, h_eq, AB_eq, AC_eq, ratio_eq]
  sorry

end find_AD_l697_697448


namespace max_profit_rate_profit_rate_ge_190_l697_697525

noncomputable def p (x : ℝ) : ℝ :=
  if 15 ≤ x ∧ x ≤ 36 then -1/10 * x^2 + 8 * x - 90
  else if 36 < x ∧ x ≤ 40 then 0.4 * x + 54
  else 0

def y (x : ℝ) : ℝ :=
  (p x) / x * 100

theorem max_profit_rate :
  ∀ x, 15 ≤ x ∧ x ≤ 40 → y x ≤ 200 :=
sorry

theorem profit_rate_ge_190 :
  ∀ x, 15 ≤ x ∧ x ≤ 36 → y x ≥ 190 ↔ 25 ≤ x ∧ x ≤ 36 :=
sorry

end max_profit_rate_profit_rate_ge_190_l697_697525


namespace square_lawn_side_length_l697_697038

theorem square_lawn_side_length (length width : ℕ) (h_length : length = 18) (h_width : width = 8) : 
  ∃ x : ℕ, x * x = length * width ∧ x = 12 := by
  -- Assume the necessary definitions and theorems to build the proof
  sorry

end square_lawn_side_length_l697_697038


namespace trader_profit_percentage_l697_697184

def original_price (P : ℝ) : ℝ := P

def discount_price (P : ℝ) : ℝ := 0.80 * P

def selling_price (P : ℝ) : ℝ := 1.28 * P

def profit_percentage (P : ℝ) : ℝ := (selling_price P - original_price P) / original_price P * 100

theorem trader_profit_percentage (P : ℝ) : profit_percentage P = 28 :=
by
  unfold profit_percentage selling_price discount_price original_price
  simp
  sorry

end trader_profit_percentage_l697_697184


namespace common_value_of_7a_and_2b_l697_697020

variable (a b : ℝ)

theorem common_value_of_7a_and_2b (h1 : 7 * a = 2 * b) (h2 : 42 * a * b = 674.9999999999999) :
  7 * a = 15 :=
by
  -- This place will contain the proof steps
  sorry

end common_value_of_7a_and_2b_l697_697020


namespace ellipse_major_axis_length_l697_697070

/--
Given:
- Ellipse equation: x^2 / a^2 + y^2 / b^2 = 1
- a > b > 0
- Foci of the ellipse: F1(-c, 0) and F2(c, 0)
- Line passing through F1 intersects the ellipse at points A and B
- y-intercept of line is 1
- |AF1| = 3|F1B|
- AF2 is perpendicular to the x-axis
Then Prove:
- The length of the major axis of the ellipse is 6
-/
theorem ellipse_major_axis_length (a b c : ℝ) (ha : a > b) (hb : b > 0) (hc : c = sqrt (a^2 - b^2))
  (A B : ℝ × ℝ) (hA : A = (c, b^2)) (hB : B = (-5/3 * c, -1/3 * b^2)) :
  |A.fst - (-c)| = 3 * |(-c) - B.fst| →
  let major_axis := 2 * a in 
  major_axis = 6 := sorry

end ellipse_major_axis_length_l697_697070


namespace number_of_parts_outside_range_l697_697441

theorem number_of_parts_outside_range (mu sigma : ℝ) (n : ℕ) (h_n : n = 1000) :
  let normal_parts := λ x, NormalPdf x mu sigma in
  let empirical_rule := 0.997 in
  let parts_outside_range := n * (1 - empirical_rule) in
  parts_outside_range = 3 :=
by
  have empirical_rule : 0.997 := by sorry
  have parts_outside_range : n * (1 - empirical_rule) = 3 := by sorry
  sorry

end number_of_parts_outside_range_l697_697441


namespace man_speed_l697_697235

theorem man_speed (train_length : ℝ) (time_to_cross : ℝ) (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) (h2 : time_to_cross = 6) (h3 : train_speed_kmph = 54.99520038396929) : 
  ∃ man_speed : ℝ, man_speed = 16.66666666666667 - 15.27644455165814 :=
by sorry

end man_speed_l697_697235


namespace count_n_for_g_n_prime_l697_697900

def g (n : ℕ) : ℕ := (n.divisors : finset ℕ).sum

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_n_for_g_n_prime : 
  (finset.range 31).filter (λ n, is_prime (g n)).card = 5 := 
sorry

end count_n_for_g_n_prime_l697_697900


namespace max_rubles_l697_697962

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l697_697962


namespace smallest_power_of_13_non_palindrome_l697_697713

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l697_697713


namespace new_average_production_l697_697767

theorem new_average_production (n : ℕ) (past_avg today_prod : ℕ) (new_avg : ℕ) :
  n = 4 →
  past_avg = 50 →
  today_prod = 90 →
  new_avg = (4 * 50 + 90) / (4 + 1) →
  new_avg = 58 :=
by
  intros h_n h_past_avg h_today_prod h_new_avg
  rw [h_n, h_past_avg, h_today_prod] at h_new_avg
  exact h_new_avg

end new_average_production_l697_697767


namespace number_of_real_solutions_l697_697706

def f (x : ℝ) : ℝ :=
  ∑ k in finset.range 100, (2 * (k + 1)) / (x - (k + 1))

theorem number_of_real_solutions :
  (∑ k in finset.range 100, (2 * (k + 1)) / (x - (k + 1))) = x → 
  ∃ s ∈ finset.range 101, s ≠ ∅ ∧ ∀ y ∈ s, (∑ k in finset.range 100, (2 * (k + 1)) / (y - (k + 1))) = y :=
sorry

end number_of_real_solutions_l697_697706


namespace cartesian_eq_line_l_general_eq_curve_C_max_distance_to_line_l_l697_697442

noncomputable def curve_C_parametric (alpha : ℝ) : ℝ × ℝ :=
  (3 * Real.cos alpha, Real.sqrt 3 * Real.sin alpha)

noncomputable def line_l_polar (rho theta : ℝ) : Prop :=
  rho * Real.cos (theta + Real.pi / 3) = Real.sqrt 3

theorem cartesian_eq_line_l :
  ∀ (x y : ℝ), line_l_polar (Real.sqrt (x^2 + y^2)) (Real.atan2 y x) ↔ x - Real.sqrt 3 * y - 2 * Real.sqrt 3 = 0 := 
sorry

theorem general_eq_curve_C :
  ∀ (x y : ℝ), (∃ (alpha : ℝ), (x, y) = curve_C_parametric alpha) ↔ x^2 / 9 + y^2 / 3 = 1 :=
sorry

theorem max_distance_to_line_l :
  ∀ (alpha : ℝ), let P := curve_C_parametric alpha in
  ∃ (d : ℝ), d = (abs (3 * Real.sqrt 2 * Real.cos (alpha + Real.pi / 4) - 2 * Real.sqrt 3)) / 2 ∧ 
              d = (3 * Real.sqrt 2 + 2 * Real.sqrt 3) / 2 :=
sorry

end cartesian_eq_line_l_general_eq_curve_C_max_distance_to_line_l_l697_697442


namespace distinct_values_of_z_l697_697677

def num_integers_between (a b : ℤ) := {n : ℤ | a ≤ n ∧ n ≤ b}

noncomputable def reverse_digits (n : ℤ) : ℤ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

def valid_values_x := {x : ℤ | 100 ≤ x ∧ x ≤ 999}
def valid_values_y := {y : ℤ | 100 ≤ y ∧ y ≤ 999}

def calculate_z (x y : ℤ) : ℤ := abs (x - y)

def possible_values_z (x : ℤ) :=
  {z : ℤ | ∃ (a b c : ℤ), (x = 100 * a + 10 * b + c) ∧ (reverse_digits x = 100 * c + 10 * b + a) ∧ (z = 99 * abs (a - c))}

theorem distinct_values_of_z :
  ∀ x ∈ valid_values_x, ∃ z ∈ (possible_values_z x), z = 99 * abs (x / 100 - x % 10) → finset.card (finite.to_finset (possible_values_z x)) = 10 :=
sorry

end distinct_values_of_z_l697_697677


namespace smallest_non_palindromic_power_of_13_l697_697744

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l697_697744


namespace sqrt_two_squared_l697_697666

noncomputable def sqrt_two : Real := Real.sqrt 2

theorem sqrt_two_squared : (sqrt_two) ^ 2 = 2 :=
by
  sorry

end sqrt_two_squared_l697_697666


namespace trigonometric_ineq_l697_697559

theorem trigonometric_ineq (h₁ : (Real.pi / 4) < 1.5) (h₂ : 1.5 < (Real.pi / 2)) : 
  Real.cos 1.5 < Real.sin 1.5 ∧ Real.sin 1.5 < Real.tan 1.5 := 
sorry

end trigonometric_ineq_l697_697559


namespace divide_0_24_by_0_004_l697_697258

theorem divide_0_24_by_0_004 : 0.24 / 0.004 = 60 := by
  sorry

end divide_0_24_by_0_004_l697_697258


namespace solve_for_x_l697_697106

theorem solve_for_x (x : ℝ) : 
  (∛(x^3) = 81 * ∜(81)) →
  (x = 243 * ∛(9)) :=
by
  sorry

end solve_for_x_l697_697106


namespace smallest_lcm_l697_697838

theorem smallest_lcm (k l : ℕ) (k_digits : 999 < k ∧ k < 10000) (l_digits : 999 < l ∧ l < 10000)
                     (gcd_cond : Nat.gcd k l = 3) : 
  ∃ k l, 999 < k ∧ k < 10000 ∧ 999 < l ∧ l < 10000 ∧ Nat.gcd k l = 3 ∧ Nat.lcm k l = 335670 :=
begin
  sorry
end

end smallest_lcm_l697_697838


namespace num_values_a_satisfying_conditions_l697_697849

open Classical

variable {x a : ℤ}

theorem num_values_a_satisfying_conditions :
  (∀ x : ℤ, 2 * (x + 1) < x + 3 → x < 1) →
  (∀ x : ℤ, x - a ≤ a + 5 → x < 1) →
  (∃ a : ℤ, a ≤ 0) →
  (Nat.card' { a : ℤ | a ≤ 0 ∧ x < 1 → x ≤ 2 * a + 5 ∧ 2 * (x + 1) < x + 3 } = 3) :=
sorry

end num_values_a_satisfying_conditions_l697_697849


namespace tangent_line_equation_l697_697989

def curve (x : ℝ) : ℝ := Real.exp x + 2

theorem tangent_line_equation :
  ∀ (x y : ℝ), (curve 0 = 3) → ∀ (m : ℝ), (m = (deriv curve) 0) → (y - 3 = m * (x - 0)) → (x - y + 3 = 0) :=
by
  intros x y hpoint m hderiv htangent
  sorry

end tangent_line_equation_l697_697989


namespace problem_statement_l697_697103

theorem problem_statement
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 0)
  (h₄ : a + b + c = a * b + b * c + c * a) :
  3 + real.cbrt ((a^3 + 1) / 2) + real.cbrt ((b^3 + 1) / 2) + real.cbrt ((c^3 + 1) / 2) ≤ 
  2 * (a + b + c) :=
sorry

end problem_statement_l697_697103


namespace triangle_is_isosceles_l697_697136

theorem triangle_is_isosceles 
{α : ℝ} {r R : ℝ}
(h : r = 4 * R * real.cos α * (real.sin (α / 2))^2) 
: ∃ a b c : ℝ, a = b ∨ a = c ∨ b = c :=
sorry

end triangle_is_isosceles_l697_697136


namespace piglet_gifted_balloons_l697_697763

noncomputable def piglet_balloons_gifted (piglet_balloons : ℕ) : ℕ :=
  let winnie_balloons := 3 * piglet_balloons
  let owl_balloons := 4 * piglet_balloons
  let total_balloons := piglet_balloons + winnie_balloons + owl_balloons
  let burst_balloons := total_balloons - 60
  piglet_balloons - burst_balloons / 8

-- Prove that Piglet gifted 4 balloons given the conditions
theorem piglet_gifted_balloons :
  ∃ (piglet_balloons : ℕ), piglet_balloons = 8 ∧ piglet_balloons_gifted piglet_balloons = 4 := sorry

end piglet_gifted_balloons_l697_697763


namespace three_digit_square_ends_with_self_l697_697309

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l697_697309


namespace range_of_f_area_of_triangle_ABC_l697_697388

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6) + 2 * cos x ^ 2

theorem range_of_f :
  (∀ x, 0 ≤ x ∧ x ≤ π / 2 → (1 / 2 ≤ f x ∧ f x ≤ 2)) :=
by
  sorry

theorem area_of_triangle_ABC (A a b : ℝ) (c : ℝ) (hA : f A = 3 / 2) (ha : a = sqrt 6) (hb : b = 2)
  (hc : c = 1 + sqrt 3) :
  (1 / 2 * b * c * sin A = 3 / 2 + sqrt 3 / 2) :=
by
  sorry

end range_of_f_area_of_triangle_ABC_l697_697388


namespace total_mileage_pay_l697_697085

-- Conditions
def distance_first_package : ℕ := 10
def distance_second_package : ℕ := 28
def distance_third_package : ℕ := distance_second_package / 2
def total_miles_driven : ℕ := distance_first_package + distance_second_package + distance_third_package
def pay_per_mile : ℕ := 2

-- Proof statement
theorem total_mileage_pay (X : ℕ) : 
  X + (total_miles_driven * pay_per_mile) = X + 104 := by
sorry

end total_mileage_pay_l697_697085


namespace jelly_beans_total_l697_697150

theorem jelly_beans_total (T : ℕ)
  (H1 : 3 * T / 4 = round (3 / 4 * T))
  (H2 : (1 * round (3 / 4 * T)) / 4 = round (1 / 4 * round (3 / 4 * T)))
  (H3 : round (1 / 4 * (3 / 4 * T)) = 750) :
  T = 4000 := 
by 
  sorry

end jelly_beans_total_l697_697150


namespace three_digit_square_ends_with_self_l697_697313

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l697_697313


namespace number_of_choices_at_least_2_science_l697_697864

def subjects : List String := ["Physics", "Chemistry", "Biology", "Politics", "History", "Geography"]
def science_subjects : List String := ["Physics", "Chemistry", "Biology"]

theorem number_of_choices_at_least_2_science :
  (∃ (choices : List (List String)) (h : ∀ choice ∈ choices, choice ⊆ subjects ∧ choice.length = 3 ∧
      (science_subjects.filter (λ s, s ∈ choice)).length ≥ 2), choices.length = 10) :=
sorry

end number_of_choices_at_least_2_science_l697_697864


namespace total_window_width_is_24_l697_697264

-- Define some constants according to the given problem
constant panes_count : ℕ := 6
constant ratio_height_width : ℕ × ℕ := (3, 4)
constant border_width_inch : ℕ := 3
constant panes_rows : ℕ := 2
constant panes_columns : ℕ := 3

-- Define the width of the window
noncomputable def window_width (pane_width_inch : ℕ) : ℕ :=
  let total_panes_width := panes_columns * pane_width_inch
  let total_borders_width := (panes_columns + 1) * border_width_inch
  total_panes_width + total_borders_width

-- Prove that the total width of the window is 24 inches given the assumptions
theorem total_window_width_is_24 : window_width (4 * 1) = 24 := by
  calc
    window_width (4 * 1)
      = 3 * 4 + 4 * 3 : by rfl
    ... = 12 + 12 : by rfl
    ... = 24 : by rfl

end total_window_width_is_24_l697_697264


namespace parametric_line_eq_l697_697983

theorem parametric_line_eq (t : ℝ) :
  ∃ t : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
  (x = 3 * t + 5) ∧ (y = 6 * t - 7) → y = 2 * x - 17 :=
by
  sorry

end parametric_line_eq_l697_697983


namespace gcd_64_144_l697_697162

theorem gcd_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end gcd_64_144_l697_697162


namespace distinct_values_of_expression_l697_697398

variable {u v x y z : ℝ}

theorem distinct_values_of_expression (hu : u + u⁻¹ = x) (hv : v + v⁻¹ = y)
  (hx_distinct : x ≠ y) (hx_abs : |x| ≥ 2) (hy_abs : |y| ≥ 2) :
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z = u * v + (u * v)⁻¹)) →
  ∃ n, n = 2 := by 
    sorry

end distinct_values_of_expression_l697_697398


namespace wages_of_one_man_l697_697196

variable (R : Type) [DivisionRing R] [DecidableEq R]
variable (money : R)
variable (num_men : ℕ := 5)
variable (num_women : ℕ := 8)
variable (total_wages : R := 180)
variable (wages_men : R := 36)

axiom equal_women : num_men = num_women
axiom total_earnings (wages : ℕ → R) :
  (wages num_men) + (wages num_women) + (wages 8) = total_wages

theorem wages_of_one_man :
  wages_men = total_wages / num_men := by
  sorry

end wages_of_one_man_l697_697196


namespace minimum_perimeter_of_triangle_PQF_l697_697611

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 25) + (y^2 / 16) = 1

def line_through_center (x y: ℝ) : Prop :=
  x = 0

noncomputable def focus (x : ℝ) : ℝ :=
  if x = 0 then sqrt (25 - 16) else 0 -- One of the foci of the ellipse with the center at origin

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (sqrt (25 - 16), 0) ∨ F = (-sqrt (25 - 16), 0)

theorem minimum_perimeter_of_triangle_PQF :
  ∀ (P Q F : ℝ × ℝ), ellipse_eq P.1 P.2 ∧ ellipse_eq Q.1 Q.2 ∧ line_through_center P.1 P.2 ∧ line_through_center Q.1 Q.2 ∧ is_focus F → 
  (|dist P F + dist Q F + dist P Q|) = 18 :=
by
  sorry

end minimum_perimeter_of_triangle_PQF_l697_697611


namespace species_population_estimate_l697_697429

theorem species_population_estimate :
  ∃ (N_A N_B N_C : ℕ), 
    (40 / 2400 = 3 / 180) ∧
    (40 / 1440 = 5 / 180) ∧
    (40 / 3600 = 2 / 180) ∧
    N_A = 2400 ∧
    N_B = 1440 ∧
    N_C = 3600 :=
by {
  existsi (2400 : ℕ),
  existsi (1440 : ℕ),
  existsi (3600 : ℕ),
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { reflexivity },
  split,
  { reflexivity },
  { reflexivity }
}

end species_population_estimate_l697_697429


namespace center_of_mass_case1_center_of_mass_case2_l697_697915

-- Definition of cell weights for case 1
def cell_weight_case1 (n i j : ℕ) : ℕ :=
if i = 1 ∧ j = 1 then 1
else if i = 1 then j
else if j = 1 then i
else (i + j - 1)

-- Definition of cell weights for case 2
def cell_weight_case2 (n i j : ℕ) : ℕ :=
i * j

-- Sum of weights for case 1
def sum_weights_case1 (n : ℕ) : ℕ :=
(∑ i in range n, ∑ j in range n, cell_weight_case1 n (i+1) (j+1))

-- Sum of weights for case 2
def sum_weights_case2 (n : ℕ) : ℕ :=
(∑ i in range n, ∑ j in range n, cell_weight_case2 n (i+1) (j+1))

-- Coordinate moments for case 1
def moment_x_case1 (n : ℕ) : ℕ :=
(∑ i in range n, ∑ j in range n, (cell_weight_case1 n (i+1) (j+1)) * (j+1))

def moment_y_case1 (n : ℕ) : ℕ :=
(∑ i in range n, ∑ j in range n, (cell_weight_case1 n (i+1) (j+1)) * (i+1))

-- Coordinate moments for case 2
def moment_x_case2 (n : ℕ) : ℕ :=
(∑ i in range n, ∑ j in range n, (cell_weight_case2 n (i+1) (j+1)) * (j+1))

def moment_y_case2 (n : ℕ) : ℕ :=
(∑ i in range n, ∑ j in range n, (cell_weight_case2 n (i+1) (j+1)) * (i+1))

-- Center of mass for case 1
theorem center_of_mass_case1 (n : ℕ) (h : 0 < n) : (moment_x_case1 n / sum_weights_case1 n) = ((n + 1) * (7 * n - 1) / (12 * n)) ∧
                                                      (moment_y_case1 n / sum_weights_case1 n) = ((n + 1) * (7 * n - 1) / (12 * n)) :=
sorry

-- Center of mass for case 2
theorem center_of_mass_case2 (n : ℕ) (h : 0 < n) : (moment_x_case2 n / sum_weights_case2 n) = ((2 * n + 1) / 3) ∧
                                                      (moment_y_case2 n / sum_weights_case2 n) = ((2 * n + 1) / 3) :=
sorry

end center_of_mass_case1_center_of_mass_case2_l697_697915


namespace remaining_content_at_end_of_second_day_l697_697840

-- Definitions corresponding to the mathematical conditions and question
def initial_content : ℚ := 1
def first_day_evaporated : ℚ := 2/3
def second_day_evaporated_fraction : ℚ := 1/4

-- Statement of the problem
theorem remaining_content_at_end_of_second_day :
  let first_day_remaining := initial_content - first_day_evaporated in
  let second_day_remaining := first_day_remaining * (1 - second_day_evaporated_fraction) in
  second_day_remaining = 1/4 :=
by
  sorry

end remaining_content_at_end_of_second_day_l697_697840


namespace number_of_real_solutions_l697_697709

noncomputable def f (x : ℝ) : ℝ :=
  ∑ n in (Finset.range 100).map (λ i, i + 1), 2 * (n : ℝ) / (x - n)

theorem number_of_real_solutions :
  (Finset.range 101).card = 101 :=
sorry

end number_of_real_solutions_l697_697709


namespace find_special_three_digit_numbers_l697_697295

theorem find_special_three_digit_numbers :
  {A : ℕ | 100 ≤ A ∧ A < 1000 ∧ (A^2 % 1000 = A)} = {376, 625} :=
by
  sorry

end find_special_three_digit_numbers_l697_697295


namespace collinear_points_l697_697818

-- Define the points A, B, C
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (4, 3)
def C (m : ℝ) : ℝ × ℝ := (5, m)

-- Prove that m = 6 if A, B, and C are collinear
theorem collinear_points :
  ∀ m : ℝ, 
    let slope_AB := (B.2 - A.2) / (B.1 - A.1),
        slope_BC := (C m).2 - B.2,
    slope_AB = slope_BC →
    m = 6 :=
by
  intros m slope_AB slope_BC h
  sorry

end collinear_points_l697_697818


namespace angle_BXY_l697_697444

theorem angle_BXY 
  (A B C D E F X Y : ℝ)
  (parallel_AB_CD : A = C ∧ B = D)
  (angle_AXE_CYX_relationship : ∀ x, ∠AXE = (2 * x) - 72 ∧ ∠CYX = x) :
  ∠BXY = 72 :=
by
  sorry

end angle_BXY_l697_697444


namespace total_tissues_brought_l697_697545

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end total_tissues_brought_l697_697545


namespace number_of_divisors_of_n_squared_l697_697627

-- Define the conditions
def has_four_divisors (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p * q

def num_divisors (k : ℕ) : ℕ :=
  let f := k.factors in
  f.foldl (λ acc x, acc * (x.2 + 1)) 1

-- Create the proof problem
theorem number_of_divisors_of_n_squared (n : ℕ) (h : has_four_divisors n) :
  num_divisors (n * n) = 9 :=
sorry

end number_of_divisors_of_n_squared_l697_697627


namespace fraction_ordering_l697_697839

theorem fraction_ordering (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) :=
by
  sorry

end fraction_ordering_l697_697839


namespace sum_of_distances_l697_697381

noncomputable def parabola := { P : ℝ × ℝ // P.1^2 = 6 * P.2 }
def focus : ℝ × ℝ := (0, 1.5)

def triangle_ABC (A B C : parabola) (F : ℝ × ℝ) : Prop :=
  let A_coords := (A.1, A.2)
  let B_coords := (B.1, B.2)
  let C_coords := (C.1, C.2) in
  F = (focus.1, focus.2) ∧
  vector_eq (vector_sub A_coords F) (vector_scale (1/3) (vector_add (vector_sub B_coords A_coords) (vector_sub C_coords A_coords)))

theorem sum_of_distances (A B C : parabola) (F : ℝ × ℝ)
  (h : triangle_ABC A B C F) : 
  (dist A.1 focus + dist B.focus + dist C.focus) = 9 := 
sorry

end sum_of_distances_l697_697381


namespace common_tangent_through_A_l697_697191

-- Defining the basic structure of points and circles
variables {Point : Type} [MetricSpace Point]

structure Circle :=
(center : Point)
(radius : ℝ)

-- Assumptions based on the problem conditions
variables {A B C I C1 A1 B1 : Point}

-- Define the circles ωB and ωC as described in the problem
variables (ωB : Circle) (ωC : Circle)

-- Assume the quadrilateral constraints as per problem specification
variable (quad_BA1IC1 : quadrilateral B A1 I C1)
variable (quad_CA1IB1 : quadrilateral C A1 I B1)

-- Assuming the common internal tangents other than IA1
def common_internal_tangent (ωB ωC : Circle) := sorry  -- Placeholder definition

-- The proof goal: proving the common internal tangent passes through A
theorem common_tangent_through_A
  (cond1 : quad_BA1IC1.contains ωB)
  (cond2 : quad_CA1IB1.contains ωC)
  (cond3 : tangent_condition ωB ωC) :
  ∃ P, common_internal_tangent ωB ωC P = true ∧ passes_through P A :=
sorry

end common_tangent_through_A_l697_697191


namespace minimum_distance_parabola_to_line_l697_697375

theorem minimum_distance_parabola_to_line 
  (x y : ℝ) (hx : x^2 = 4 * y) : 
  ∃ m : ℝ, m = sqrt 2 ∧ ∀ (b : ℝ), (b = y - x + 3 → (abs ((x - 3 - b) / sqrt 2)) = m) :=
sorry

end minimum_distance_parabola_to_line_l697_697375


namespace second_term_of_geometric_series_l697_697648

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem second_term_of_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) 
  (h_r : r = 1 / 4) (h_S : S = 16) (h_sum : S = geometric_series_sum a r) : 
  a * r = 3 :=
by
  have h_a : a = 12,
  { 
    -- We are given that S = 16 and r = 1/4, and S = a / (1 - r)
    -- So, 16 = a / (1 - 1/4) = a / (3/4) => 16 = 4a / 3 => 4a = 48 => a = 12
    sorry
  }
  have h_ar : a * r = 12 * (1 / 4), from by
  {
    -- Calculation of the second term
    sorry
  }
  show a * r = 3, by
  {
    -- a * r = 12 * 1/4 = 3
    sorry
  }

end second_term_of_geometric_series_l697_697648


namespace trigonometric_identities_of_7pi_over_6_l697_697288

theorem trigonometric_identities_of_7pi_over_6 :
  (real.sec (7 * real.pi / 6) = - 2 * real.sqrt 3 / 3) ∧
  (real.csc (7 * real.pi / 6) = - 2) :=
by
  sorry

end trigonometric_identities_of_7pi_over_6_l697_697288


namespace fixed_point_l697_697540

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : (1, 4) ∈ (λ x : ℝ, (3 + a^(x - 1))) :=
by
  -- These conditions guarantee the corresponding fixed point
  intro x y
  sorry

end fixed_point_l697_697540


namespace tangent_circle_equation_l697_697028

theorem tangent_circle_equation :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi →
    ∃ c : ℝ × ℝ, ∃ r : ℝ,
      (∀ (a b : ℝ), c = (a, b) →
        (|a * Real.cos θ + b * Real.sin θ - Real.cos θ - 2 * Real.sin θ - 2| = r) ∧
        (r = 2)) ∧
      (∃ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = r^2)) :=
by
  sorry

end tangent_circle_equation_l697_697028


namespace frosting_time_difference_l697_697505

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end frosting_time_difference_l697_697505


namespace smallest_number_is_33_l697_697188

theorem smallest_number_is_33 
  (x : ℕ) 
  (h1 : ∀ y z, y = 2 * x → z = 4 * x → (x + y + z) / 3 = 77) : 
  x = 33 :=
by
  sorry

end smallest_number_is_33_l697_697188


namespace complex_number_solution_l697_697698

theorem complex_number_solution : 
  ∃ (z : ℂ), (|z - 2| = |z + 4| ∧ |z + 4| = |z - 2i|) ∧ z = -1 + ⅈ :=
by
  sorry

end complex_number_solution_l697_697698


namespace find_d_l697_697030

variable (A B C P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P]
variable (AB BC AC d : ℕ)

-- Given conditions
def triangle_ABC (AB BC AC : ℕ) := AB = 425 ∧ BC = 450 ∧ AC = 510

def interior_point_P (P : Type) := ∃ A B C, MetricSpace A ∧ MetricSpace B ∧ MetricSpace C ∧ MetricSpace P

def segments_parallel_and_equal_length (d : ℕ) := ∃ P, interior_point_P P ∧ ∃ (p1 p2 p3 : A), (p1 = p2 ∧ p2 = p3 ∧ p3 = d)

-- Proving the desired result
theorem find_d 
  (h1 : triangle_ABC 425 450 510) 
  (h2 : segments_parallel_and_equal_length 306) 
  : d = 306 := 
by 
  sorry

end find_d_l697_697030


namespace smallest_possible_e_l697_697135

-- Definitions based on given conditions
def polynomial (x : ℝ) (a b c d e : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- The given polynomial has roots -3, 4, 8, and -1/4, and e is positive integer
theorem smallest_possible_e :
  ∃ (a b c d e : ℤ), polynomial x a b c d e = 4*x^4 - 32*x^3 - 23*x^2 + 104*x + 96 ∧ e > 0 ∧ e = 96 :=
by
  sorry

end smallest_possible_e_l697_697135


namespace parabola_transform_l697_697980

theorem parabola_transform :
  ∀ (x : ℝ),
    ∃ (y : ℝ),
      (y = -2 * x^2) →
      (∃ (y' : ℝ), y' = y - 1 ∧
      ∃ (x' : ℝ), x' = x - 3 ∧
      ∃ (y'' : ℝ), y'' = -2 * (x')^2 - 1) :=
by sorry

end parabola_transform_l697_697980


namespace maximum_rubles_received_l697_697931

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l697_697931


namespace least_total_nuts_l697_697151

-- Define the conditions as constants
constant n1 n2 n3 : ℤ
constant x : ℤ

-- Given conditions
axiom (conds_1 : ∃ k : ℤ, n1 = 144 * k)
axiom (conds_2 : ∃ k : ℤ, n2 = 144 * k)
axiom (conds_3 : ∃ k : ℤ, n3 = 576 * k)
axiom (conds_4 : 4 * x = (5 * n1) / 6 + n2 / 18 + (7 * n3) / 48)
axiom (conds_5 : 3 * x = n1 / 9 + (n2) / 3 + 7 * n3 / 48)
axiom (conds_6 : 2 * x = n1 / 9 + n2 / 9 + n3 / 8)

-- Goal
theorem least_total_nuts : ∃ k : ℤ, (n1 + n2 + n3 = 864) := by {
  sorry
}

end least_total_nuts_l697_697151


namespace drawings_in_first_five_pages_l697_697875

theorem drawings_in_first_five_pages : ∑ i in range 5, 5 * (i + 1) = 75 :=
by
  sorry

end drawings_in_first_five_pages_l697_697875


namespace cindy_first_to_get_five_l697_697244

def probability_of_five : ℚ := 1 / 6

def anne_turn (p: ℚ) : ℚ := 1 - p
def cindy_turn (p: ℚ) : ℚ := p
def none_get_five (p: ℚ) : ℚ := (1 - p)^3

theorem cindy_first_to_get_five : 
    (∑' n, (anne_turn probability_of_five * none_get_five probability_of_five ^ n) * 
                cindy_turn probability_of_five) = 30 / 91 := by 
    sorry

end cindy_first_to_get_five_l697_697244


namespace at_least_one_not_less_than_four_l697_697246

theorem at_least_one_not_less_than_four 
( m n t : ℝ ) 
( h_m : 0 < m ) 
( h_n : 0 < n ) 
( h_t : 0 < t ) : 
∃ a, ( a = m + 4 / n ∨ a = n + 4 / t ∨ a = t + 4 / m ) ∧ 4 ≤ a :=
sorry

end at_least_one_not_less_than_four_l697_697246


namespace problem_l697_697770

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}
def C : Set ℕ := {1, 3}

theorem problem : A ∩ (U \ B) = C := by
  sorry

end problem_l697_697770


namespace height_of_cylinder_in_hemisphere_l697_697215

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (h_base_parallel : Prop) : ℝ :=
  if h_base_parallel then sqrt (r_hemisphere^2 - r_cylinder^2) else 0

theorem height_of_cylinder_in_hemisphere :
  cylinder_height 3 8 true = sqrt 55 :=
by
  unfold cylinder_height
  simp
  sorry

end height_of_cylinder_in_hemisphere_l697_697215


namespace max_teams_advancing_l697_697040

theorem max_teams_advancing (teams : ℕ) (points_to_advance : ℕ) 
  (points_win points_draw points_loss total_games total_points : ℕ) 
  (h1 : teams = 6)
  (h2 : points_to_advance = 12)
  (h3 : points_win = 3)
  (h4 : points_draw = 1)
  (h5 : points_loss = 0)
  (h6 : total_games = (teams * (teams - 1)) / 2)
  (h7 : total_points = total_games * points_win) :
  ∀ n, n * points_to_advance ≤ total_points → n ≤ 3 :=
by
  intros n h
  have h_max_points : total_points = 45 := by
    rw [h6, h1]
    simp only [Nat.mul_sub_left_distrib, Nat.mul_one, Nat.mul_comm]
    norm_num
  have h_n : n * points_to_advance ≤ 45 := by
    rw h_max_points at h
    exact h
  have : n ≤ 3 := by
    exact (Nat.le_div_iff_mul_le _ _ _).mpr h_n
    norm_num
  exact this
  sorry

end max_teams_advancing_l697_697040


namespace flux_through_section_l697_697255

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ := (y * z, x * z, x * y)
noncomputable def plane (x y : ℝ) : ℝ := 1 - x - y

/-- The region of interest is the section of the plane in the first octant (x, y, z ≥ 0) -/
def region_of_interest (x y : ℝ) : Prop := (0 ≤ x) ∧ (0 ≤ y) ∧ (x + y ≤ 1)

/-- The normal vector to the plane x + y + z = 1 has coordinates (1, 1, 1). -/
def normal_vector : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def unit_normal_vector (nv : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let magnitude := Real.sqrt (nv.1^2 + nv.2^2 + nv.3^2)
in (nv.1 / magnitude, nv.2 / magnitude, nv.3 / magnitude)

noncomputable def flux_integral : ℝ :=
  ∫⁻ (x : ℝ) in Icc 0 1, ∫⁻ (y : ℝ) in Icc 0 (1 - x), 
    (vector_field x y (plane x y)) (unit_normal_vector normal_vector) (x, y)

/-- The flux of the vector field through the section of the plane in the first octant, 
along the normal vector of the plane, is 1/12. -/
theorem flux_through_section : flux_integral = 1 / 12 := by
  sorry

end flux_through_section_l697_697255


namespace arc_lengths_difference_l697_697803

theorem arc_lengths_difference (x y : ℝ) (h1 : x + 7 * y = 10) (h2 : x^2 + y^2 = 4) :
  |arc_length_difference (x + 7y = 10) (x^2 + y^2 = 4)| = 2 * π :=
sorry

end arc_lengths_difference_l697_697803


namespace cylinder_height_in_hemisphere_l697_697219

noncomputable def height_of_cylinder (r_hemisphere r_cylinder : ℝ) : ℝ :=
  real.sqrt (r_hemisphere^2 - r_cylinder^2)

theorem cylinder_height_in_hemisphere :
  let r_hemisphere := 8
  let r_cylinder := 3
  height_of_cylinder r_hemisphere r_cylinder = real.sqrt 55 :=
by
  sorry

end cylinder_height_in_hemisphere_l697_697219


namespace sequence_general_formula_l697_697784

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, (2 * a n - 2) = ∑ k in Finset.range n, a k

theorem sequence_general_formula (a : ℕ → ℕ) : sequence a → ∀ n : ℕ, a n = 2^n :=
begin
  intros h_seq n,
  sorry
end

end sequence_general_formula_l697_697784


namespace minimum_value_of_f_plus_f_prime_l697_697807

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + a * x^2 - 4
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := -3 * x^2 + 2 * a * x

theorem minimum_value_of_f_plus_f_prime:
  (∃ a : ℝ, f'(2, a) = 0) → 
  (∀ m n : ℝ, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 → f(m, 3) + f'(-1, 3) = -13) :=
by
  sorry

end minimum_value_of_f_plus_f_prime_l697_697807


namespace sum_of_cubes_mod_5_l697_697481

theorem sum_of_cubes_mod_5 (b : Fin 10 → ℕ) (h_inc : ∀ i j, i < j → b i < b j) (h_sum : (∑ i, b i) = 100) :
  ((∑ i, (b i)^3) % 5) = 0 :=
sorry

end sum_of_cubes_mod_5_l697_697481


namespace max_rubles_l697_697960

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l697_697960


namespace find_three_digit_numbers_l697_697314

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l697_697314


namespace tan_alpha_tan_beta_l697_697403

/-- Given the cosine values of the sum and difference of two angles, 
    find the value of the product of their tangents. -/
theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/3) 
  (h2 : Real.cos (α - β) = 1/5) : 
  Real.tan α * Real.tan β = -1/4 := sorry

end tan_alpha_tan_beta_l697_697403


namespace smallest_non_palindromic_power_of_13_l697_697729

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l697_697729


namespace petya_max_rubles_l697_697945

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l697_697945


namespace tammy_speed_l697_697979

noncomputable theory

variables
  (t v : ℝ) -- climbing time on the first day and speed on the first day
  (t2 v2 : ℝ) -- climbing time and speed on the second day

def effective_time_first_day := t - 0.5
def effective_time_second_day := t2 - 0.75

def first_day_distance := v * effective_time_first_day
def second_day_distance := v2 * effective_time_second_day

def total_distance := first_day_distance + second_day_distance

theorem tammy_speed
  (h1 : t + t2 = 14) -- total climbing time
  (h2 : t2 = t - 2) -- second day time difference
  (h3 : v2 = v + 0.5) -- speed difference
  (h4 : total_distance t v (t - 2) (v + 0.5) = 52) -- total distance
  : v + 0.5 = 4.375 :=
sorry

end tammy_speed_l697_697979


namespace maximum_rubles_received_max_payment_possible_l697_697936

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l697_697936


namespace price_of_cookie_cookie_price_verification_l697_697522

theorem price_of_cookie 
  (total_spent : ℝ) 
  (cost_per_cupcake : ℝ)
  (num_cupcakes : ℕ)
  (cost_per_doughnut : ℝ)
  (num_doughnuts : ℕ)
  (cost_per_pie_slice : ℝ)
  (num_pie_slices : ℕ)
  (num_cookies : ℕ)
  (total_cookies_cost : ℝ)
  (total_cost : ℝ) :
  (num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  + num_cookies * total_cookies_cost = total_spent) → 
  total_cookies_cost = 0.60 :=
by
  sorry

noncomputable def sophie_cookies_price : ℝ := 
  let total_cost := 33
  let num_cupcakes := 5
  let cost_per_cupcake := 2
  let num_doughnuts := 6
  let cost_per_doughnut := 1
  let num_pie_slices := 4
  let cost_per_pie_slice := 2
  let num_cookies := 15
  let total_spent_on_other_items := 
    num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  let remaining_cost := total_cost - total_spent_on_other_items 
  remaining_cost / num_cookies

theorem cookie_price_verification :
  sophie_cookies_price = 0.60 :=
by
  sorry

end price_of_cookie_cookie_price_verification_l697_697522


namespace midpoint_distance_trapezoid_l697_697045

theorem midpoint_distance_trapezoid (x : ℝ) : 
  let AD := x
  let BC := 5
  PQ = (|x - 5| / 2) :=
sorry

end midpoint_distance_trapezoid_l697_697045


namespace maximum_rubles_received_l697_697949

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l697_697949


namespace maximum_rubles_received_max_payment_possible_l697_697934

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l697_697934


namespace smallest_non_palindromic_power_of_13_l697_697741

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l697_697741


namespace find_AD_l697_697454

-- Declare the conditions as hypotheses
theorem find_AD (AB AC : ℝ) (BD CD AD : ℝ) 
  (h1 : AB = 13) (h2 : AC = 20) (h_ratio : BD / CD = 3 / 4) 
  (h1_perp : AD = (hypotenuse_sqrt (BD^2 + AD^2) = AB) )
  (h2_perp : AD = (hypotenuse_sqrt (CD^2 + AD^2) = AC)) :
  AD = 8 * Real.sqrt 2 := 
  -- Include sorry to indicate the proof is omitted
  sorry

end find_AD_l697_697454


namespace f_g_of_2_eq_4_l697_697837

def f (x : ℝ) : ℝ := x^2 - 2*x + 1
def g (x : ℝ) : ℝ := 2*x - 5

theorem f_g_of_2_eq_4 : f (g 2) = 4 := by
  sorry

end f_g_of_2_eq_4_l697_697837


namespace complex_number_solution_l697_697699

theorem complex_number_solution : 
  ∃ (z : ℂ), (|z - 2| = |z + 4| ∧ |z + 4| = |z - 2i|) ∧ z = -1 + ⅈ :=
by
  sorry

end complex_number_solution_l697_697699


namespace four_digit_numbers_count_four_digit_numbers_with_adjacent_evens_count_four_digit_numbers_with_non_adjacent_evens_count_l697_697348

theorem four_digit_numbers_count : 
  let numbers := {1, 2, 3, 4, 5, 6},
      evens := {2, 4, 6},
      odds := {1, 3, 5},
      C := λ n k, Nat.choose n k,
      A := λ n r, Nat.perm n r in
  (C evens.card 2 * C odds.card 2 * A 4 4) = 216 := 
sorry

theorem four_digit_numbers_with_adjacent_evens_count : 
  let numbers := {1, 2, 3, 4, 5, 6},
      evens := {2, 4, 6},
      odds := {1, 3, 5},
      C := λ n k, Nat.choose n k,
      A := λ n r, Nat.perm n r in
  (C evens.card 2 * C odds.card 2 * 3 * A 3 3) = 108 := 
sorry

theorem four_digit_numbers_with_non_adjacent_evens_count : 
  let numbers := {1, 2, 3, 4, 5, 6},
      evens := {2, 4, 6},
      odds := {1, 3, 5},
      C := λ n k, Nat.choose n k,
      A := λ n r, Nat.perm n r in
  (C evens.card 2 * C odds.card 2 * A 4 4 - C evens.card 2 * C odds.card 2 * 3 * A 3 3) = 108 :=
sorry

end four_digit_numbers_count_four_digit_numbers_with_adjacent_evens_count_four_digit_numbers_with_non_adjacent_evens_count_l697_697348


namespace phase_shift_of_cosine_l697_697325

theorem phase_shift_of_cosine (b c : ℝ) (h_b : b = 5) (h_c : c = 5 * real.pi / 6) :
  ∃ x : ℝ, (b * x - c = 0) ∧ (x = real.pi / 6 ∨ x = - real.pi / 6) := by
sorry

end phase_shift_of_cosine_l697_697325


namespace minimum_editors_l697_697655

theorem minimum_editors
  (writers : ℕ)
  (total_people : ℕ)
  (x : ℕ)
  (both_writers_editors : ℕ)
  (neither_writers_nor_editors : ℕ)
  (max_both_writers_editors : ℕ)
  (h1 : writers = 45)
  (h2 : total_people = 100)
  (h3 : both_writers_editors = x)
  (h4 : neither_writers_nor_editors = 2 * x)
  (h5 : max_both_writers_editors = 18)
  (h6 : x ≤ max_both_writers_editors)
  : ∃ E : ℕ, E = 73 := by
  have inclusion_exclusion := total_people - neither_writers_nor_editors = writers + x - both_writers_editors
  have E := 55 + x
  have min_editors := E = 73
  use min_editors
  sorry

end minimum_editors_l697_697655


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697224

open Real

theorem height_of_cylinder_inscribed_in_hemisphere
  (r_cylinder r_hemisphere : ℝ)
  (h_radius_cylinder : r_cylinder = 3)
  (h_radius_hemisphere : r_hemisphere = 8) :
  ∃ h_cylinder : ℝ, h_cylinder = sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) :=
by
  use sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  sorry

#check height_of_cylinder_inscribed_in_hemisphere

end height_of_cylinder_inscribed_in_hemisphere_l697_697224


namespace three_digit_numbers_with_square_ending_in_them_l697_697303

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l697_697303


namespace ratio_of_areas_l697_697797

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def ratio_of_triangle_areas : Prop :=
let A : ℝ × ℝ := (0, 0) in
let B : ℝ × ℝ := (2, 0) in
let C : ℝ × ℝ := (2, 1) in
let D : ℝ × ℝ := (0, 1) in
let E := midpoint B D in
let F : ℝ × ℝ := (0, 0.75) in
let G : ℝ × ℝ := (2, 2 / 3) in
let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2 in
let area_DFE := area_triangle D F E in
let area_BGE := area_triangle B G E in
area_DFE / area_BGE = 3 / 4

theorem ratio_of_areas : ratio_of_triangle_areas := 
by {
  -- Verification steps will go here.
  sorry
}

end ratio_of_areas_l697_697797


namespace compare_logs_l697_697350

theorem compare_logs (h1 : 4^5 < 7^4) (h2 : 11^4 < 7^5) : (log 11 / log 7) < (log 243 / log 81) ∧ (log 243 / log 81) < (log 7 / log 4) :=
by
  have a : log 7 / log 4 > 5 / 4 := sorry
  have b : log 11 / log 7 < 5 / 4 := sorry
  have c : log 243 / log 81 = 5 / 4 := sorry
  exact ⟨b, by { exact c.symm ▸ a }⟩

end compare_logs_l697_697350


namespace op_example_l697_697015

def op (c d : ℕ) : ℕ := 4 * c + 3 * d - c * d

theorem op_example :
  op 3 7 = 12 :=
begin
  -- Lean automatically understands the necessary conditions here.
  sorry
end

end op_example_l697_697015


namespace fewer_gallons_for_plants_correct_l697_697874

-- Define the initial conditions
def initial_water : ℕ := 65
def water_per_car : ℕ := 7
def total_cars : ℕ := 2
def water_for_cars : ℕ := water_per_car * total_cars
def water_remaining_after_cars : ℕ := initial_water - water_for_cars
def water_for_plates_clothes : ℕ := 24
def water_remaining_before_plates_clothes : ℕ := water_for_plates_clothes * 2
def water_for_plants : ℕ := water_remaining_after_cars - water_remaining_before_plates_clothes

-- Define the query statement
def fewer_gallons_for_plants : Prop := water_per_car - water_for_plants = 4

-- Proof skeleton
theorem fewer_gallons_for_plants_correct : fewer_gallons_for_plants :=
by sorry

end fewer_gallons_for_plants_correct_l697_697874


namespace interval_decreasing_triangle_area_l697_697000

noncomputable theory

-- Definitions for the vectors a and b
def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1 / 2)

-- Definition for the function f(x)
def f (x : ℝ) : ℝ :=
  let ⟨ax, ay⟩ := vec_a x
  let ⟨bx, by⟩ := vec_b x
  (ax + bx) * ax + (ay + by) * ay - 2

-- Proof problem 1: Interval where f(x) is monotonically decreasing
theorem interval_decreasing (k : ℤ) : 
  ∀ x ∈ Set.Icc ((Real.pi / 3) + k * Real.pi) ((5 * Real.pi / 6) + k * Real.pi), 
  f'(x) < 0 :=
sorry

-- Additional condition for problem 2
def A_condition (A : ℝ) : Prop :=
  0 < A ∧ A < Real.pi / 2 ∧ f A = 1

-- Proof problem 2: Area of triangle ABC
theorem triangle_area (A : ℝ) (a b c : ℝ) (hA : A_condition A) (ha : a = 2 * Real.sqrt 3) (hc : c = 4) :
  let b := (solve quadratic equation for b using cosine law with A, a, c)
  let S := 1 / 2 * c * b * Real.sin A
  S = 4 * Real.sqrt 3 :=
sorry

end interval_decreasing_triangle_area_l697_697000


namespace shaina_chocolate_l697_697061

theorem shaina_chocolate (total_chocolate : ℚ) (piles : ℕ) (piles_given_to_shaina : ℕ) :
  total_chocolate = 64 / 7 →
  piles = 6 →
  piles_given_to_shaina = 2 →
  (total_chocolate / piles) * piles_given_to_shaina = 64 / 21 :=
by
  intros htotal hpiles hpiles_given
  rw [htotal, hpiles, hpiles_given]
  have h_div : 64 / 7 / 6 = 64 / 42 := by
    rw div_eq_mul_inv
    norm_num
  rw [h_div]
  norm_num
  sorry

end shaina_chocolate_l697_697061


namespace inverse_function_proof_l697_697542

theorem inverse_function_proof :
  ∀ (y : ℝ), (y = 3^(x^2 - 1) ∧ -1 ≤ x ∧ x < 0) → 
  (-√(log 3 y + 1) = x ∧ (1/3 < y ∧ y ≤ 1)) :=
by
  sorry

end inverse_function_proof_l697_697542


namespace leo_speed_faster_than_mike_l697_697494

noncomputable def mike_original_speed : ℕ := 600
noncomputable def mike_hours_before_break : ℕ := 9
noncomputable def mike_total_hours_before_break : ℕ := mike_original_speed * mike_hours_before_break
noncomputable def mike_speed_after_break : ℕ := mike_original_speed / 3
noncomputable def mike_hours_after_break : ℕ := 2
noncomputable def mike_total_after_break : ℕ := mike_speed_after_break * mike_hours_after_break
noncomputable def total pamphlets_printed : ℕ := 9400

noncomputable def pamphlets_by_mike : ℕ := mike_total_hours_before_break + mike_total_after_break
noncomputable def pamphlets_by_leo : ℕ := total_pamphlets_printed - pamphlets_by_mike
noncomputable def leo_hours : ℕ := mike_hours_before_break / 3
noncomputable def leo_speed : ℕ := pamphlets_by_leo / leo_hours / mike_original_speed

theorem leo_speed_faster_than_mike :
  leo_speed = 2 := sorry

end leo_speed_faster_than_mike_l697_697494


namespace fraction_multiplication_l697_697850

-- Given fractions a and b
def a := (1 : ℚ) / 4
def b := (1 : ℚ) / 8

-- The first product result
def result1 := a * b

-- The final product result when multiplied by 4
def result2 := result1 * 4

-- The theorem to prove
theorem fraction_multiplication : result2 = (1 : ℚ) / 8 := by
  sorry

end fraction_multiplication_l697_697850


namespace like_terms_ratio_l697_697402

theorem like_terms_ratio (m n : ℕ) (h₁ : m - 2 = 2) (h₂ : 3 = 2 * n - 1) : m / n = 2 := 
by
  sorry

end like_terms_ratio_l697_697402


namespace area_APEG_l697_697576

-- Defining the vertices of the squares and points of intersection
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (8, 8)
def D : ℝ × ℝ := (0, 8)
def E : ℝ × ℝ := (14, 0)
def F : ℝ × ℝ := (14, 6)
def G : ℝ × ℝ := (8, 6)

-- Equation of line DE
def line_DE (x : ℝ) := - (4/7) * x + 8

-- Intersection point P where DE intersects BG
def P : ℝ × ℝ := (8, line_DE 8)

-- Prove that the area of quadrilateral APEG is 18 cm²
theorem area_APEG : 
  let area_triangle (p q r : ℝ × ℝ) : ℝ := 
    abs ((p.fst * (q.snd - r.snd) + q.fst * (r.snd - p.snd) + r.fst * (p.snd - q.snd)) / 2)
  in 
  area_triangle A P E + area_triangle P E G = 18 :=
by sorry

end area_APEG_l697_697576


namespace three_digit_ends_with_itself_iff_l697_697305

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l697_697305


namespace minimum_value_of_f_l697_697324

/-- Define the function f(x) = 1/x - 2x. -/
def f (x : ℝ) : ℝ := (1 / x) - 2 * x

/-- Define the interval [-2, -1/2]. -/
def interval : set ℝ := {x | -2 ≤ x ∧ x ≤ -1 / 2}

/-- Prove that the minimum value of f(x) on the interval [-2, -1/2] is -1. -/
theorem minimum_value_of_f : ∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -1 :=
by {
  sorry -- Proof will be provided here
}

end minimum_value_of_f_l697_697324


namespace function_property_l697_697894

theorem function_property (k a b : ℝ) (h : a ≠ b) (h_cond : k * a + 1 / b = k * b + 1 / a) : 
  k * (a * b) + 1 = 0 := 
by 
  sorry

end function_property_l697_697894


namespace condition_necessary_but_not_sufficient_l697_697125

-- Definition of conditions
variables {a b c : ℝ}
def Sufficient (a b c : ℝ) : Prop := a > b → ac^2 > bc^2
def Necessary (a b c : ℝ) : Prop := ac^2 > bc^2 → a > b

-- Theorem statement for the problem conclusion
theorem condition_necessary_but_not_sufficient
  (a b c : ℝ) (hab : a > b) (h : c ≠ 0) : (ac^2 > bc^2) :=
by
  -- The proof would be provided here
  sorry

end condition_necessary_but_not_sufficient_l697_697125


namespace volume_of_cube_with_diagonal_l697_697991

theorem volume_of_cube_with_diagonal (d : ℝ) (h : d = 5 * real.sqrt 3) : 
  ∃ (V : ℝ), V = 125 := 
by
  -- Definitions and conditions from the problem are used directly
  let s := d / real.sqrt 3
  sorry

end volume_of_cube_with_diagonal_l697_697991


namespace percentage_equivalence_l697_697408

theorem percentage_equivalence (x : ℝ) (h : 0.30 * 0.15 * x = 45) : 0.15 * 0.30 * x = 45 :=
sorry

end percentage_equivalence_l697_697408


namespace volume_of_solid_l697_697558

-- Define the vectors and conditions
def vector_w (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)
def given_vector : ℝ × ℝ × ℝ := (6, -30, 12)

-- Define the condition
def satisfies_condition (x y z : ℝ) : Prop :=
  (x * x + y * y + z * z) = (x * 6 + y * (-30) + z * 12)

-- Define the volume of the sphere
def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

-- Translate to Lean statement
theorem volume_of_solid :
  ∃ (x y z : ℝ), satisfies_condition x y z → sphere_volume (3 * Real.sqrt 30) = 324 * Real.sqrt 30 * Real.pi :=
sorry

end volume_of_solid_l697_697558


namespace _l697_697834

statement theorem sec_tan_difference (x : ℝ) (h : Real.sec x + Real.tan x = 3) : 
  Real.sec x - Real.tan x = 1 / 3 := by
  sorry

end _l697_697834


namespace ann_frosting_time_l697_697498

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end ann_frosting_time_l697_697498


namespace intersection_of_sets_l697_697816

-- Definitions for sets and universe
def U := Set ℝ
def A : Set ℝ := { x | 2^x > 1 }
def B : Set ℝ := { x | -4 < x ∧ x < 1 }

-- The theorem we aim to prove
theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_of_sets_l697_697816


namespace divisors_of_square_of_cube_of_prime_l697_697616

theorem divisors_of_square_of_cube_of_prime (p : ℕ) (hp : p.prime) (n : ℕ) (h : n = p^3) :
  nat.num_divisors (n^2) = 7 :=
sorry

end divisors_of_square_of_cube_of_prime_l697_697616


namespace exists_divisible_by_A_l697_697269

def A : ℕ := 2011 ^ 2012

def sequence (u : ℕ → ℕ) : Prop :=
  u 0 = 0 ∧ u 1 = 0 ∧ ∀ n, u (n + 2) = u (n + 1) + u n + 1

theorem exists_divisible_by_A (u : ℕ → ℕ) (h_seq : sequence u) :
  ∃ N ≥ 1, A ∣ u N ∧ A ∣ u (N + 1) :=
sorry

end exists_divisible_by_A_l697_697269


namespace marbles_removed_is_2_l697_697600

theorem marbles_removed_is_2 (k : ℕ) 
  (h_total_marbles : 5)
  (h_red_marbles : 2)
  (h_green_marbles : 3)
  (h_probability : (nat.choose 2 k / nat.choose 5 k : ℝ) = 0.1) :
  k = 2 := 
sorry

end marbles_removed_is_2_l697_697600


namespace first_group_size_l697_697530

/-- The average of 11 results is 42. The average of the first some results is 49, 
and the average of the last 7 results is 52. The fifth result is 147. 
Prove that the number of results in the first group is 5. -/
theorem first_group_size (n : ℕ) (h1 : (42 : ℝ) = (11:ℝ)⁻¹ * (∑ i in finset.range 11, some (results i)))
  (h2 : (49 : ℝ) = (n:ℝ)⁻¹ * (∑ i in finset.range n, some (results i)))
  (h3 : (52 : ℝ) = (7:ℝ)⁻¹ * (∑ i in finset.range(11)-n, some (results (i + n))))
  (h4 : (results 4) = 147) : n = 5 :=
by
  sorry

end first_group_size_l697_697530


namespace solve_for_x_l697_697021

noncomputable def F (a b c d e : ℝ) : ℝ :=
  a^b + (c * d) / e

theorem solve_for_x (x ≈ 9.25) :
  F 2 (x - 2) 4 11 2 = 174 :=
by
  sorry

end solve_for_x_l697_697021


namespace S_2011_eq_l697_697811

noncomputable def f (x : ℝ) (b : ℝ) := x^2 + b * x
noncomputable def f_derivative (x : ℝ) (b : ℝ) := 2 * x + b

def slope_at_A (b : ℝ): Prop := f_derivative 1 b = 3

def sequence (n : ℕ) (b : ℝ) := 1 / (f n b)

noncomputable def sum_of_sequence (n : ℕ) (b : ℝ) := (Finset.range n).sum (λ i, sequence i b)

theorem S_2011_eq : ∀ (b : ℝ), slope_at_A b → sum_of_sequence 2011 b = 2011 / 2012 := 
by 
  intros b hb
  sorry

end S_2011_eq_l697_697811


namespace find_AD_l697_697452

-- Declare the conditions as hypotheses
theorem find_AD (AB AC : ℝ) (BD CD AD : ℝ) 
  (h1 : AB = 13) (h2 : AC = 20) (h_ratio : BD / CD = 3 / 4) 
  (h1_perp : AD = (hypotenuse_sqrt (BD^2 + AD^2) = AB) )
  (h2_perp : AD = (hypotenuse_sqrt (CD^2 + AD^2) = AC)) :
  AD = 8 * Real.sqrt 2 := 
  -- Include sorry to indicate the proof is omitted
  sorry

end find_AD_l697_697452


namespace max_total_pieces_l697_697035

-- Define the size of the chessboard
def chessboard := fin 200 × fin 200

-- Define the piece type and their locations
inductive PieceColor | Red | Blue

-- Define visibility condition for pieces in the same row or column
def canSeeEachOther (p1 p2 : chessboard) : Prop :=
  p1.1 = p2.1 ∨ p1.2 = p2.2

-- Define the condition that each piece sees exactly five pieces of the opposite color
def seesExactlyFiveOppositeColor (placement: chessboard → Option PieceColor) (p : chessboard) (pc : PieceColor) : Prop :=
  (count_opposite_color (placement p) .filter (λ p2, canSeeEachOther p p2)) = 5

-- Define count_opposite_color function
def count_opposite_color (c: Option PieceColor) : list (chessboard) → nat := sorry

-- Define the maximum pieces condition
def max_pieces_condition (placement : chessboard → Option PieceColor) : Prop :=
  ∀ p : chessboard, ∀ pc : PieceColor,
  placement p = some pc → seesExactlyFiveOppositeColor placement p pc

-- The total number of pieces
def total_pieces (placement : chessboard → Option PieceColor) : nat :=
  list.length (list.filter_map placement (list.finRange 200 × list.finRange 200))

-- The main theorem statement
theorem max_total_pieces : ∃ placement : chessboard → Option PieceColor, max_pieces_condition placement ∧ total_pieces placement = 4000 := sorry

end max_total_pieces_l697_697035


namespace exists_M_on_y_axis_equal_distances_exists_M_on_y_axis_equilateral_triangle_l697_697421

-- Definitions of points in 3D space
def point := (ℝ × ℝ × ℝ)

def dist (p q : point) : ℝ :=
  let (x1, y1, z1) := p
  let (x2, y2, z2) := q
  sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

-- Conditions given in the problem
def A : point := (3, 0, 1)
def B : point := (1, 0, -3)

-- Coordinate on the y-axis
def is_on_y_axis (M : point) : Prop := M.1 = 0 ∧ M.3 = 0

-- Question 1
theorem exists_M_on_y_axis_equal_distances : 
  ∃ M : point, is_on_y_axis M ∧ dist M A = dist M B :=
by 
  sorry

-- Question 2
theorem exists_M_on_y_axis_equilateral_triangle :
  ∃ M : point, is_on_y_axis M ∧ dist M A = dist M B ∧ dist M A = dist A B :=
by 
  sorry

end exists_M_on_y_axis_equal_distances_exists_M_on_y_axis_equilateral_triangle_l697_697421


namespace right_triangle_isosceles_division_l697_697967

theorem right_triangle_isosceles_division (A B C M : Point)
(h_right_triangle : right_triangle A B C)
(h_midpoint : midpoint M A B)
(h_median : median C M) :
  is_isosceles_triangle A C M ∧ is_isosceles_triangle B C M :=
sorry

end right_triangle_isosceles_division_l697_697967


namespace num_divisors_square_l697_697622

theorem num_divisors_square (n : ℕ) (h₁ : ∃ p q : ℕ, prime p ∧ prime q ∧ p ≠ q ∧ n = p * q) :
  num_divisors (n^2) = 9 :=
by
  sorry

end num_divisors_square_l697_697622


namespace ratio_AD_DC_l697_697420

-- Definitions and given conditions
variables {A B C D : Type}
variables {a b c d : ℝ}
variables {AB BC AC BD : ℝ}
variables [h_AB : AB = 5] [h_BC : BC = 7] [h_AC : AC = 9] [h_BD : BD = 5]
variables [is_on_AC : AC = A + C] -- D is on AC

-- Ratio we want to prove
theorem ratio_AD_DC (AD DC : ℝ) (h_AD_DC : AD / DC = 19 / 8) :
  AD / DC = 19 / 8 :=
sorry -- proof not required, hence the sorry keyword.

end ratio_AD_DC_l697_697420


namespace find_AD_l697_697453

-- Declare the conditions as hypotheses
theorem find_AD (AB AC : ℝ) (BD CD AD : ℝ) 
  (h1 : AB = 13) (h2 : AC = 20) (h_ratio : BD / CD = 3 / 4) 
  (h1_perp : AD = (hypotenuse_sqrt (BD^2 + AD^2) = AB) )
  (h2_perp : AD = (hypotenuse_sqrt (CD^2 + AD^2) = AC)) :
  AD = 8 * Real.sqrt 2 := 
  -- Include sorry to indicate the proof is omitted
  sorry

end find_AD_l697_697453


namespace range_of_f_eq_l697_697554

noncomputable def f (x : ℝ) : ℝ := real.log2 (2^x + 1)

theorem range_of_f_eq :
  (∀ x : ℝ, 2^x + 1 > 1) ∧
  (∀ x y : ℝ, x < y → real.log2 x < real.log2 y) →
  set.range f = set.Ioi 0 :=
by
  sorry

end range_of_f_eq_l697_697554


namespace actual_distance_l697_697914

def map_scale := 1 / 400000  -- Define the map scale as a ratio
def map_distance := 8.5 -- The measured distance on the map in cm

theorem actual_distance :
  (map_distance / map_scale) / 100000 = 34 :=
sorry

end actual_distance_l697_697914


namespace range_of_m_l697_697356

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then
    1/2 - 2 * x^2
  else if 1 < x ∧ x < 2 then
    sorry -- definition of f(2-x)
  else
    sorry -- definition for other ranges due to quasi-periodicity

def g (x m : ℝ) : ℝ :=
  -2 * Real.log x + 1/2 * x^2 + x + m

theorem range_of_m (m : ℝ) :
  (∃ x1 ∈ Icc 6 8, ∃ x2 ∈ Ioi 0, g x2 m - f x1 ≤ 0) →
  m ≤ 13/2 :=
sorry

end range_of_m_l697_697356


namespace kaleb_cherries_left_l697_697472

theorem kaleb_cherries_left (initial_cherries eaten_cherries remaining_cherries : ℕ) (h1 : initial_cherries = 67) (h2 : eaten_cherries = 25) : remaining_cherries = initial_cherries - eaten_cherries → remaining_cherries = 42 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end kaleb_cherries_left_l697_697472


namespace value_of_a_l697_697579

theorem value_of_a (a b : ℝ) (h1 : b = 2120) (h2 : a / b = 0.5) : a = 1060 := 
by
  sorry

end value_of_a_l697_697579


namespace general_formula_sum_Tn_lt_1_l697_697446

variable {n : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ}

-- Define the common ratio and initial terms with conditions
def common_ratio (q : ℕ) : Prop := q > 1
def a_2 (a_1 q : ℕ) : Prop := a_1 * q = 2
def S_3 (a_1 q : ℕ) : Prop := a_1 * (1 + q + q^2) = 7

-- Define the general sequence formula
def a_n (n : ℕ) : ℕ := 2^(n-1)

-- Define the log and auxiliary sequences
def b_n (n : ℕ) : ℕ := n - 1
def c_n (n : ℕ) : ℕ := 1 / (b_n (n+1) * b_n (n+2))

-- Define the sum of the first n terms of c sequence
def T_n (n : ℕ) : ℕ := ∑ k in finset.range(n), c_n k

-- Theorem: General formula for the sequence
theorem general_formula (q : ℕ) (hq : common_ratio q) (a_1 : ℕ) (ha2 : a_2 a_1 q) (hs3 : S_3 a_1 q) : ∀ n, a_n n = 2^(n-1) := 
sorry

-- Theorem: Sum of sequence T_n
theorem sum_Tn_lt_1 : ∀ n, T_n n < 1 := 
sorry

end general_formula_sum_Tn_lt_1_l697_697446


namespace sin_angle_sum_leq_two_over_sqrt_three_l697_697823

open Real

variable {A B C M : Point} -- Define the points A, B, C, and M

variable [IsTriangle A B C] -- State that A, B, C form a triangle

-- Definition stating that M is the centroid of triangle ABC
def is_centroid (M : Point) (A B C : Point) : Prop :=
  M = (A + B + C) / 3

-- Final theorem stating the desired inequality
theorem sin_angle_sum_leq_two_over_sqrt_three (hM : is_centroid M A B C) :
  (sin (angle C A M) + sin (angle C B M) ≤ 2 / sqrt 3) :=
begin
  sorry
end

end sin_angle_sum_leq_two_over_sqrt_three_l697_697823


namespace smallest_non_palindromic_power_of_13_l697_697726

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l697_697726


namespace sin_B_value_triangle_area_l697_697465

variable (a b c A B C : ℝ)
variable (triangle_ABC : Triangle A B C a b c)
variable (h1 : 3 * b = 4 * c)
variable (h2 : B = 2 * C)

theorem sin_B_value : 
  sin B = 4 * sqrt 5 / 9 :=
by sorry

variable (h3 : b = 4)

theorem triangle_area : 
  area triangle_ABC = 14 * sqrt 5 / 9 :=
by sorry

end sin_B_value_triangle_area_l697_697465


namespace proof_problem_l697_697895

noncomputable def f (x y k : ℝ) : ℝ := k * x + (1 / y)

theorem proof_problem
  (a b k : ℝ) (h1 : f a b k = f b a k) (h2 : a ≠ b) :
  f (a * b) 1 k = 0 :=
sorry

end proof_problem_l697_697895


namespace prod_square_free_coeffs_8_l697_697614

def is_square_free {α : Type*} (term : Multiset α) : Prop :=
  ∀ i, term.count i ≤ 1

def sum_square_free_coeffs (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else 
    let a : ℕ → ℕ := λ n => 
      if n = 0 then 1
      else if n = 1 then 1
      else (sum_square_free_coeffs (n - 1)) + (n - 1) * (sum_square_free_coeffs (n - 2))
    a n

theorem prod_square_free_coeffs_8 : sum_square_free_coeffs 8 = 764 := 
  sorry

end prod_square_free_coeffs_8_l697_697614


namespace combined_cost_increase_l697_697577

def original_bicycle_cost : ℝ := 200
def original_skates_cost : ℝ := 50
def bike_increase_percent : ℝ := 0.06
def skates_increase_percent : ℝ := 0.15

noncomputable def new_bicycle_cost : ℝ := original_bicycle_cost * (1 + bike_increase_percent)
noncomputable def new_skates_cost : ℝ := original_skates_cost * (1 + skates_increase_percent)
noncomputable def original_total_cost : ℝ := original_bicycle_cost + original_skates_cost
noncomputable def new_total_cost : ℝ := new_bicycle_cost + new_skates_cost
noncomputable def total_increase : ℝ := new_total_cost - original_total_cost
noncomputable def percent_increase : ℝ := (total_increase / original_total_cost) * 100

theorem combined_cost_increase : percent_increase = 7.8 := by
  sorry

end combined_cost_increase_l697_697577


namespace problem1_problem2_l697_697670

-- Define the base types and expressions
variables (x m : ℝ)

-- Proofs of the given expressions
theorem problem1 : (x^7 / x^3) * x^4 = x^8 :=
by sorry

theorem problem2 : m * m^3 + ((-m^2)^3 / m^2) = 0 :=
by sorry

end problem1_problem2_l697_697670


namespace classroom_students_count_l697_697280

-- Definitions of given conditions
def total_students : ℕ := 1260

def aud_students : ℕ := (7 * total_students) / 18

def non_aud_students : ℕ := total_students - aud_students

def classroom_students : ℕ := (6 * non_aud_students) / 11

-- Theorem statement
theorem classroom_students_count : classroom_students = 420 := by
  sorry

end classroom_students_count_l697_697280


namespace segment_MH_length_l697_697445

-- Definition of the geometric properties and conditions
variables {A B C D H M : Type*}
variables [point A] [point B] [point C] [point D] [point H] [point M]
variables (AB AC AD CD BC : ℝ) (angleDAH angleHAB angleAHB : ℝ)
variables (is_altitude : ∀ {X Y Z : Type*}, point X → point Y → point Z → Prop)
variables (is_angle_bisector : ∀ {X Y Z : Type*}, point X → point Y → point Z → Prop)
variables (is_mid_segment : ∀ {X Y Z W : Type*}, point X → point Y → point Z → point W → Prop)
variables (is_mid_point : ∀ {X Y Z : Type*}, point X → point Y → point Z → Prop)

axiom tri_ABC : AB = 4 ∧ AC = 6
axiom angle_DAH_eq_HAB : angleDAH = angleHAB
axiom angle_AHB_90 : angleAHB = 90
axiom mid_M_BC : is_mid_point M B C

-- Main theorem to state the required proof
theorem segment_MH_length : ∃ (MH : ℝ), 
  AB = 4 ∧ AC = 6 ∧ 
  angleDAH = angleHAB ∧ 
  angleAHB = 90 ∧ 
  is_mid_point M B C ∧
  MH = 1 :=
by 
  sorry

end segment_MH_length_l697_697445


namespace find_covered_number_l697_697431

theorem find_covered_number (a x : ℤ) (h : (x - a) / 2 = x + 3) (hx : x = -7) : a = 1 := by
  sorry

end find_covered_number_l697_697431


namespace rachel_math_homework_pages_l697_697512

theorem rachel_math_homework_pages (M : ℕ) 
  (h1 : 23 = M + (M + 3)) : M = 10 :=
by {
  sorry
}

end rachel_math_homework_pages_l697_697512


namespace find_n_for_divisibility_l697_697412

def digit_sum_odd_positions := 8 + 4 + 5 + 6 -- The sum of the digits in odd positions
def digit_sum_even_positions (n : ℕ) := 5 + n + 2 -- The sum of the digits in even positions

def is_divisible_by_11 (n : ℕ) := (digit_sum_odd_positions - digit_sum_even_positions n) % 11 = 0

theorem find_n_for_divisibility : is_divisible_by_11 5 :=
by
  -- Proof would go here (but according to the instructions, we'll insert a placeholder)
  sorry

end find_n_for_divisibility_l697_697412


namespace value_of_s_l697_697483

theorem value_of_s (s : ℝ) : (3 * (-1)^5 + 2 * (-1)^4 - (-1)^3 + (-1)^2 - 4 * (-1) + s = 0) → (s = -5) :=
by
  intro h
  sorry

end value_of_s_l697_697483


namespace sum_of_radii_eq_radius_l697_697787

noncomputable def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) := 
  {p | (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 + (p.3 - center.3) ^ 2 = radius ^ 2}

def circle_in_plane (center : ℝ × ℝ) (radius : ℝ) := 
  {p | (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = radius ^ 2}

-- Given conditions translated to Lean 4
variables 
  (O O1 O2 : ℝ × ℝ × ℝ)
  (R r1 r2 : ℝ)
  (plane_k : ℝ × ℝ × ℝ → Prop)
  (k : set (ℝ × ℝ))
  (G G1 G2 : set (ℝ × ℝ × ℝ))
  (proj_O_on_plane_k: ℝ × ℝ)
  [is_circle: circle_in_plane proj_O_on_plane_k R = k]
  [is_sphere_G: sphere O R = G]
  [is_sphere_G1: sphere O1 r1 = G1]
  [is_sphere_G2: sphere O2 r2 = G2]
  [G1_tangent_G: ∀ p, p ∈ G1 → p ∈ G]
  [G2_tangent_G: ∀ p, p ∈ G2 → p ∈ G]

-- Proof statement
theorem sum_of_radii_eq_radius : r1 + r2 = R :=
sorry

end sum_of_radii_eq_radius_l697_697787


namespace decision_based_on_mode_l697_697200

-- Define the sales data as a list of pairs (size, quantity)
def sales_data : List (ℤ × ℕ) :=
[(22, 3), (22.5, 5), (23, 10), (23.5, 15), (24, 8), (24.5, 3), (25, 2)]

-- Define the target shoe size being restocked
def target_size : ℤ := 23.5

-- The theorem to prove that the decision is based on the mode
theorem decision_based_on_mode (data : List (ℤ × ℕ)) (size : ℤ) :
  (is_mode data size ∧ size = 23.5) := sorry

end decision_based_on_mode_l697_697200


namespace magic_square_l697_697036

-- Define a 3x3 grid with positions a, b, c and unknowns x, y, z, t, u, v
variables (a b c x y z t u v : ℝ)

-- State the theorem: there exists values for x, y, z, t, u, v
-- such that the sums in each row, column, and both diagonals are the same
theorem magic_square (h1: x = (b + 3*c - 2*a) / 2)
  (h2: y = a + b - c)
  (h3: z = (b + c) / 2)
  (h4: t = 2*c - a)
  (h5: u = b + c - a)
  (h6: v = (2*a + b - c) / 2) :
  x + a + b = y + z + t ∧
  y + z + t = u ∧
  z + t + u = b + z + c ∧
  t + u + v = a + u + c ∧
  x + t + v = u + y + c ∧
  by sorry :=
sorry

end magic_square_l697_697036


namespace complex_number_solution_l697_697701

theorem complex_number_solution : 
  ∃ (z : ℂ), (|z - 2| = |z + 4| ∧ |z + 4| = |z - 2i|) ∧ z = -1 + ⅈ :=
by
  sorry

end complex_number_solution_l697_697701


namespace arkansas_tshirts_sold_l697_697526

theorem arkansas_tshirts_sold (A T : ℕ) (h1 : A + T = 163) (h2 : 98 * A = 8722) : A = 89 := by
  -- We state the problem and add 'sorry' to skip the actual proof
  sorry

end arkansas_tshirts_sold_l697_697526


namespace line_passes_through_fixed_point_l697_697145

theorem line_passes_through_fixed_point
  (A B C O X Y : Type)
  [triangle ABC]
  (h : angle BAC = 30 * (π / 180))
  (circumcenter O ABC)
  (X_on_AC : X ∈ ray A C)
  (Y_on_BC : Y ∈ ray B C)
  (hOX_BY : distance O X = distance B Y) :
  ∃ P : Type, line_through P X Y :=
sorry

end line_passes_through_fixed_point_l697_697145


namespace height_of_cylinder_in_hemisphere_l697_697213

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (h_base_parallel : Prop) : ℝ :=
  if h_base_parallel then sqrt (r_hemisphere^2 - r_cylinder^2) else 0

theorem height_of_cylinder_in_hemisphere :
  cylinder_height 3 8 true = sqrt 55 :=
by
  unfold cylinder_height
  simp
  sorry

end height_of_cylinder_in_hemisphere_l697_697213


namespace max_rubles_can_receive_l697_697924

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l697_697924


namespace gcd_4536_8721_l697_697321

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end gcd_4536_8721_l697_697321


namespace three_digit_square_ends_with_self_l697_697311

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l697_697311


namespace determine_f_at_zero_l697_697802

theorem determine_f_at_zero (f : ℝ → ℝ) 
  (h : ∀ α : ℝ, f (sin α + cos α) = sin α * cos α) : f 0 = -1/2 := 
by 
  sorry

end determine_f_at_zero_l697_697802


namespace solve_system_l697_697116

theorem solve_system :
  ∃ (x y z : ℝ), x + y + z = 9 ∧ (1/x + 1/y + 1/z = 1) ∧ (x * y + x * z + y * z = 27) ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end solve_system_l697_697116


namespace condition_B_is_necessary_but_not_sufficient_l697_697889

-- Definitions of conditions A and B
def condition_A (x : ℝ) : Prop := 0 < x ∧ x < 5
def condition_B (x : ℝ) : Prop := abs (x - 2) < 3

-- The proof problem statement
theorem condition_B_is_necessary_but_not_sufficient : 
∀ x, condition_A x → condition_B x ∧ ¬(∀ x, condition_B x → condition_A x) := 
sorry

end condition_B_is_necessary_but_not_sufficient_l697_697889


namespace smallest_non_palindromic_power_of_13_l697_697727

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l697_697727


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697221

open Real

theorem height_of_cylinder_inscribed_in_hemisphere
  (r_cylinder r_hemisphere : ℝ)
  (h_radius_cylinder : r_cylinder = 3)
  (h_radius_hemisphere : r_hemisphere = 8) :
  ∃ h_cylinder : ℝ, h_cylinder = sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2) :=
by
  use sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  sorry

#check height_of_cylinder_inscribed_in_hemisphere

end height_of_cylinder_inscribed_in_hemisphere_l697_697221


namespace blake_change_l697_697662

-- Definitions based on conditions
def n_l : ℕ := 4
def n_c : ℕ := 6
def p_l : ℕ := 2
def p_c : ℕ := 4 * p_l
def amount_given : ℕ := 6 * 10

-- Total cost calculations derived from the conditions
def total_cost_lollipops : ℕ := n_l * p_l
def total_cost_chocolates : ℕ := n_c * p_c
def total_cost : ℕ := total_cost_lollipops + total_cost_chocolates

-- Calculating the change
def change : ℕ := amount_given - total_cost

-- Theorem stating the final answer
theorem blake_change : change = 4 := sorry

end blake_change_l697_697662


namespace fraction_yellow_surface_area_l697_697202

theorem fraction_yellow_surface_area
  (cube_edge : ℕ)
  (small_cubes : ℕ)
  (yellow_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (fraction_yellow : ℚ) :
  cube_edge = 4 ∧
  small_cubes = 64 ∧
  yellow_cubes = 15 ∧
  total_surface_area = 6 * cube_edge * cube_edge ∧
  yellow_surface_area = 16 ∧
  fraction_yellow = yellow_surface_area / total_surface_area →
  fraction_yellow = 1/6 :=
by
  sorry

end fraction_yellow_surface_area_l697_697202


namespace perpendicular_line_exists_l697_697781

theorem perpendicular_line_exists (A : Point) (l : Line) (h : ¬ (A ∈ l)) : 
  ∃ (m : Line), (A ∈ m) ∧ (is_perpendicular m l) :=
sorry

end perpendicular_line_exists_l697_697781


namespace mary_change_percentage_is_closest_to_five_percent_l697_697492

noncomputable def total_price : ℝ := 12.99 + 9.99 + 7.99 + 6.50 + 4.99 + 3.75 + 1.27

noncomputable def change (payment : ℝ) : ℝ := payment - total_price

noncomputable def percentage (part total : ℝ) : ℝ := (part / total) * 100

theorem mary_change_percentage_is_closest_to_five_percent :
  let payment := 50.00 in
  let change_amount := change payment in
  let percent_change := percentage change_amount payment in
  abs (percent_change - 5) < abs (percent_change - 10) ∧
  abs (percent_change - 5) < abs (percent_change - 3) ∧
  abs (percent_change - 5) < abs (percent_change - 7) ∧
  abs (percent_change - 5) < abs (percent_change - 12) :=
by
  sorry

end mary_change_percentage_is_closest_to_five_percent_l697_697492


namespace inequality_proof_l697_697108

def inequality_solution (x : ℝ) : Prop :=
  (x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)

theorem inequality_proof (x : ℝ) :
  (1 < x ∨ x < 2) ∧ (2 < x ∨ x < 3) ∧ (3 < x ∨ x < 4) ∧ (4 < x ∨ x < 5) ∧ (5 < x ∨ x < 6) →
  inequality_solution x :=
begin
  -- Proof goes here
  sorry,
end

end inequality_proof_l697_697108


namespace sum_real_imaginary_parts_l697_697142

noncomputable def z : ℂ := (1 + 3 * complex.I) / (1 - complex.I)

theorem sum_real_imaginary_parts : (z.re + z.im) = 1 := by
  sorry

end sum_real_imaginary_parts_l697_697142


namespace height_of_cylinder_in_hemisphere_l697_697212

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (h_base_parallel : Prop) : ℝ :=
  if h_base_parallel then sqrt (r_hemisphere^2 - r_cylinder^2) else 0

theorem height_of_cylinder_in_hemisphere :
  cylinder_height 3 8 true = sqrt 55 :=
by
  unfold cylinder_height
  simp
  sorry

end height_of_cylinder_in_hemisphere_l697_697212


namespace maximum_rubles_received_max_payment_possible_l697_697932

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l697_697932


namespace green_peaches_more_than_red_l697_697565

theorem green_peaches_more_than_red :
  let red_peaches := 5
  let green_peaches := 11
  (green_peaches - red_peaches) = 6 := by
  sorry

end green_peaches_more_than_red_l697_697565


namespace max_rubles_can_receive_l697_697918

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l697_697918


namespace kite_height_30_sqrt_43_l697_697340

theorem kite_height_30_sqrt_43
  (c d h : ℝ)
  (h1 : h^2 + c^2 = 170^2)
  (h2 : h^2 + d^2 = 150^2)
  (h3 : c^2 + d^2 = 160^2) :
  h = 30 * Real.sqrt 43 := by
  sorry

end kite_height_30_sqrt_43_l697_697340


namespace area_BNM_l697_697447

-- Define the conditions as hypotheses
def parallelogram (A B C D : Point) : Prop :=
  parallel (line A B) (line C D) ∧ parallel (line B C) (line D A) 

def angle_bisector (A B C M : Point) : Prop :=
  angle ∠BAC = 2 * angle ∠BAM

-- Create the final Lean theorem statement
theorem area_BNM (A B C D M N : Point)
  (h1 : parallelogram A B C D)
  (h2 : distance A B = 6)
  (h3 : height (point D) (line A B) = 3)
  (h4 : on_line A B M)
  (h5 : on_line B C M)
  (h6 : distance B M = 4)
  (h7 : angle_bisector A B B M)
  (h8 : intersection (line A M) (line B D) = N) :
  area (triangle B N M) = 27 / 8 := by
  sorry

end area_BNM_l697_697447


namespace binom_505_505_eq_one_l697_697259

theorem binom_505_505_eq_one : Nat.choose 505 505 = 1 := by
  have prop := Nat.choose_symm
  sorry

end binom_505_505_eq_one_l697_697259


namespace ruby_shares_with_9_friends_l697_697100

theorem ruby_shares_with_9_friends
    (total_candies : ℕ) (candies_per_friend : ℕ)
    (h1 : total_candies = 36) (h2 : candies_per_friend = 4) :
    total_candies / candies_per_friend = 9 := by
  sorry

end ruby_shares_with_9_friends_l697_697100


namespace measure_of_angle_C_maximum_area_of_triangle_l697_697856

noncomputable def triangle (A B C a b c : ℝ) : Prop :=
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B

theorem measure_of_angle_C :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 :=
by
  intros A B C a b c h
  sorry

theorem maximum_area_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 →
  1 / 2 * a * b * Real.sin C = (Real.sqrt 2 / 2 + 1 / 2) :=
by
  intros A B C a b c h hC
  sorry

end measure_of_angle_C_maximum_area_of_triangle_l697_697856


namespace arithmetic_sequence_sum_l697_697865

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h1 : a 1 + a 4 + a 7 = 39) (h2 : a 3 + a 6 + a 9 = 27) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) : (∑ i in Finset.range 9, a (i + 1)) = 99 :=
by
  sorry

end arithmetic_sequence_sum_l697_697865


namespace abs_five_minus_sqrt_two_l697_697685

theorem abs_five_minus_sqrt_two : |(5 : ℝ) - real.sqrt 2| = 5 - real.sqrt 2 :=
by
  sorry

end abs_five_minus_sqrt_two_l697_697685


namespace original_position_402_l697_697982

theorem original_position_402 (n : ℕ) (cards : Finset ℕ) (h1 : ∀ (k : ℕ), k ∈ cards ↔ 1 ≤ k ∧ k ≤ 2 * n)
  (pileA pileB : Finset ℕ) (h2 : pileA = (cards.filter (λ x, x ≤ n)))
  (h3 : pileB = (cards.filter (λ x, n < x)))
  (h4 : ∀ (x : ℕ), x ∈ pileB ∧ 2 * x - 1 ∉ pileA)
  (h5 : ∀ i, 1 ≤ i ∧ i ≤ 2 * n → (if i ≤ n then 2 * i - 1 else 2 * (i - n)) = 201) :
  201 ∈ cards → 2 * n = 402 :=
by
  sorry

end original_position_402_l697_697982


namespace find_AD_l697_697449

noncomputable def AD (AB AC BD_ratio CD_ratio : ℝ) : ℝ :=
  if h = sqrt(128) then h else 0

theorem find_AD 
  (AB AC : ℝ) 
  (BD_ratio CD_ratio : ℝ)
  (h : ℝ)
  (h_eq : h = sqrt 128) :
  AB = 13 → 
  AC = 20 → 
  BD_ratio / CD_ratio = 3 / 4 → 
  AD AB AC BD_ratio CD_ratio = 8 * Real.sqrt 2 :=
by
  intro h_eq AB_eq AC_eq ratio_eq
  simp only [AD, h_eq, AB_eq, AC_eq, ratio_eq]
  sorry

end find_AD_l697_697449


namespace proof_problem_l697_697896

noncomputable def f (x y k : ℝ) : ℝ := k * x + (1 / y)

theorem proof_problem
  (a b k : ℝ) (h1 : f a b k = f b a k) (h2 : a ≠ b) :
  f (a * b) 1 k = 0 :=
sorry

end proof_problem_l697_697896


namespace smallest_power_of_13_non_palindrome_l697_697715

-- Define a function to check if a number is a palindrome
def is_palindrome (n : ℕ) : Bool :=
  let s := n.to_string
  s = s.reverse

-- Define the main theorem stating the smallest power of 13 that is not a palindrome
theorem smallest_power_of_13_non_palindrome (n : ℕ) (hn : ∀ k : ℕ, 0 < k → 13^k = n → k = 2) : n = 169 :=
by
  have h1 : is_palindrome (13^1) = true := by sorry
  have h2 : is_palindrome (13^2) = false := by sorry
  have hm : ∀ m : ℕ, 0 < m → m < 2 → is_palindrome (13^m) = true := by sorry
  sorry

end smallest_power_of_13_non_palindrome_l697_697715


namespace area_of_triangle_KDC_is_25sqrt3_l697_697051

-- Definitions for the conditions provided in the problem
def radius_length : ℝ := 10
def chord_length : ℝ := 10
def KA_length : ℝ := 20
def area_of_triangle_KDC : ℝ := 25 * real.sqrt 3

-- Premises based on the conditions given
variables (O K A B D C X: Type)
variables [metric_space O]
variables [metric_space K]
variables [metric_space A]
variables [metric_space B]
variables [metric_space D]
variables [metric_space C]
variables [metric_space X]
variables (r : ℝ) [fact (r = radius_length)]
variables (cd : ℝ) [fact (cd = chord_length)]
variables (ka : ℝ) [fact (ka = KA_length)]
variables (l : line O) (diam : line O) (perp : line O)
variables (points_collinear : ∀ p q r s : O, p.collinear q r s)

-- The theorem stating the required proof problem
theorem area_of_triangle_KDC_is_25sqrt3 
  (collinear_OKAB : points_collinear K A O B)
  (parallel_CD_KB : ∀ cd kb : line O, cd.parallel kb)
  (radius_KA_is_20 : KA_length = 20) :
  area_of_triangle_KDC = 25 * real.sqrt 3 :=
sorry

end area_of_triangle_KDC_is_25sqrt3_l697_697051


namespace problem_statement_l697_697022

def mean_of_products (S : List ℕ) : ℚ :=
  let non_empty_subsets := List.powerset S |>.filter (λ l => l ≠ [])
  let products := non_empty_subsets.map (List.prod)
  (↑(products.sum) : ℚ) / ↑(products.length)

noncomputable def S' : List ℕ :=
  [2, 2, 7, 23]

theorem problem_statement (S S' : List ℕ) (h : ℕ)
  (h₁ : mean_of_products S = 13)
  (h₂ : S' = S ++ [h])
  (h₃ : mean_of_products S' = 49) :
  S' = [2, 2, 7, 23] :=
sorry

end problem_statement_l697_697022


namespace quadratic_roots_range_l697_697346

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (x1^2 - 2 * x1 + m - 2 = 0) ∧ 
    (x2^2 - 2 * x2 + m - 2 = 0)) → m < 3 := 
by 
  sorry

end quadratic_roots_range_l697_697346


namespace vertex_in_quadrant_I_l697_697133

variable {f : ℝ → ℝ}

/-- The graph of y = f(x) passes through the origin, and its derivative y = f'(x) is a straight line.
    Prove that the vertex of the graph of y = f(x) is located in Quadrant I. -/
theorem vertex_in_quadrant_I
  (h₁ : f 0 = 0)
  (h₂ : ∃ a b : ℝ, ∀ x, f' x = a * x + b)
  : vertex_of f ∈ quadrant_I := sorry

end vertex_in_quadrant_I_l697_697133


namespace eval_sum_of_squares_l697_697489

-- Define floor function and summation bounds
def floor : ℝ → ℤ := λ x, (Int.floor x)

-- Define the given theorem to be proved
theorem eval_sum_of_squares :
  let sum := ∑ m in Finset.range 4018, Real.sqrt (2009^2 + (m + 1))
  in floor sum = 8074171 :=
by
  -- sorry to skip the proof steps
  sorry

end eval_sum_of_squares_l697_697489


namespace squares_with_equal_black_and_white_cells_l697_697854

open Nat

/-- Given a specific coloring of cells in a 5x5 grid, prove that there are
exactly 16 squares that have an equal number of black and white cells. --/
theorem squares_with_equal_black_and_white_cells :
  let gridSize := 5
  let number_of_squares_with_equal_black_and_white_cells := 16
  true := sorry

end squares_with_equal_black_and_white_cells_l697_697854


namespace bus_A_speed_l697_697251

variable (v_A v_B : ℝ)
variable (h1 : v_A - v_B = 15)
variable (h2 : v_A + v_B = 75)

theorem bus_A_speed : v_A = 45 := sorry

end bus_A_speed_l697_697251


namespace cone_volume_ratio_l697_697208

noncomputable def ratio_of_volumes (r h : ℝ) : ℝ :=
  let S1 := r^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 12
  let S2 := r^2 * (10 * Real.pi + 3 * Real.sqrt 3) / 12
  S1 / S2

theorem cone_volume_ratio (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  ratio_of_volumes r h = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3) :=
  sorry

end cone_volume_ratio_l697_697208


namespace maximum_rubles_received_l697_697929

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l697_697929


namespace solve_for_x_l697_697519

theorem solve_for_x (x : ℝ) (hx : 16^x * 16^x * 16^x = 256^3) : x = 2 :=
sorry

end solve_for_x_l697_697519


namespace count_numbers_satisfying_conditions_l697_697399

theorem count_numbers_satisfying_conditions :
  {n : ℕ // 200 ≤ n ∧ n < 500 ∧ (3 ∈ n.digits 10) ∧ (n % 5 = 0)}.card = 24 := sorry

end count_numbers_satisfying_conditions_l697_697399


namespace smallest_n_for_triples_l697_697327

theorem smallest_n_for_triples :
  ∃ n : ℕ, n > 0 ∧ (∀ (x y z : ℕ) (i : fin n), x + y = 3 * z → 
      set.pairwise_disjoint {1, 2, ..., 3 * n} (λ {x y z}, x + y = 3 * z)) :=
sorry

end smallest_n_for_triples_l697_697327


namespace reproductive_quantity_increase_by_11_times_l697_697659

-- Definitions of the given conditions
def Q : ℝ := 6
def T : ℝ := 50
def lambda : ℝ := T / (Q - 1)
def K (n : ℝ) : ℝ := lambda * Real.log n

-- Statement of the problem to be proved
theorem reproductive_quantity_increase_by_11_times : K (12 * n) - K n = 24.8 :=
by
  -- Formal proof skipped
  sorry

end reproductive_quantity_increase_by_11_times_l697_697659


namespace perpendicular_vectors_l697_697396

section
  variables {m : ℝ}
  def a := (1 : ℝ, -2 : ℝ)
  def b := (m, 1 : ℝ)
  def a_plus_b := (a.1 + b.1, a.2 + b.2)
  def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

  theorem perpendicular_vectors : dot_product a a_plus_b = 0 → m = -3 :=
  by
    sorry
end

end perpendicular_vectors_l697_697396


namespace desired_percentage_markup_l697_697632

variable (W : ℝ)

def initial_price := 45
def increased_amount := 5

-- Condition: W + 0.80W = $45
def wholesale_price := W + 0.80 * W = initial_price

-- Condition: new price after increasing by $5
def new_price := initial_price + increased_amount

-- Desired percentage markup
def percentage_markup := ((new_price - W) / W) * 100

-- The theorem to prove
theorem desired_percentage_markup (W : ℝ) :
  W + 0.80 * W = initial_price →
  new_price = initial_price + increased_amount →
  percentage_markup = 100 :=
by
  sorry

end desired_percentage_markup_l697_697632


namespace smallest_non_palindromic_power_of_13_l697_697743

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l697_697743


namespace parabola_vertex_and_point_l697_697262

theorem parabola_vertex_and_point (a b c : ℝ) : 
  (∀ x, y = a * x^2 + b * x + c) ∧ 
  ∃ x y, (y = a * (x - 4)^2 + 3) → 
  (a * 2^2 + b * 2 + c = 5) → 
  (a = 1/2 ∧ b = -4 ∧ c = 11) :=
by
  sorry

end parabola_vertex_and_point_l697_697262


namespace probability_three_digit_divisible_by_3_l697_697347

def digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def remainder_groups : (Finset ℕ) × (Finset ℕ) × (Finset ℕ) :=
  ({1, 4, 7}, {2, 5, 8}, {0, 3, 6, 9})

noncomputable def total_combinations :=
  fintype.card (digits.erase 0).to_finset * fintype.card (digits.erase 0).to_finset * fintype.card (digits.erase 1).to_finset

noncomputable def favorable_combinations : ℕ :=
  -- placeholder calculation for valid combinations, assume we have 228 valid cases.
  228

noncomputable def probability_divisible_by_3 :=
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem probability_three_digit_divisible_by_3 :
  probability_divisible_by_3 = 19 / 54 :=
sorry

end probability_three_digit_divisible_by_3_l697_697347


namespace find_a_l697_697411

-- Definitions based on the problem conditions
def pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_a (a : ℝ) (z : ℂ): 
  (1 + complex.I) * z = 1 - a * complex.I ∧ pure_imaginary z → a = 1 :=
  by
    sorry

end find_a_l697_697411


namespace maximum_rubles_received_l697_697946

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l697_697946


namespace min_boxes_needed_to_form_cube_l697_697821

-- Definitions based on problem conditions
def width : ℕ := 18
def length : ℕ := 12
def height : ℕ := 9

-- Least common multiple of the given dimensions
def lcm_dimensions : ℕ := Nat.lcm (Nat.lcm width length) height

-- Volume of the cube whose side is the LCM of the dimensions
def volume_cube : ℕ := lcm_dimensions ^ 3

-- Volume of one cuboid-shaped box
def volume_box : ℕ := width * length * height

-- Number of boxes needed to fill the cube
def number_boxes : ℕ := volume_cube / volume_box

-- Theorem: Proving that the number of boxes required is 24
theorem min_boxes_needed_to_form_cube : number_boxes = 24 := by
  -- Placeholder for the actual proof
  sorry

end min_boxes_needed_to_form_cube_l697_697821


namespace basic_computer_price_l697_697591

theorem basic_computer_price (C P : ℝ) 
(h1 : C + P = 2500) 
(h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end basic_computer_price_l697_697591


namespace probability_of_uniform_color_l697_697599

noncomputable def probability_uniform_grid : ℚ :=
  let p_center_uniform := (2 : ℚ) * (1 / 2) ^ 4 in
  let p_edge_pairs := (1 / 2) ^ 8 in
  let p_corner_pairs := (1 / 2) ^ 4 in
  p_center_uniform * p_edge_pairs * p_corner_pairs

theorem probability_of_uniform_color :
  probability_uniform_grid = 1 / 32768 :=
by
  -- Probability calculation based on given conditions
  let p_center_uniform := (2 : ℚ) * (1 / 2) ^ 4
  let p_edge_pairs := (1 / 2) ^ 8
  let p_corner_pairs := (1 / 2) ^ 4
  have h_center_uniform : p_center_uniform = 1 / 8 := by 
    norm_cast
    -- Calculation for the probability that all center squares are uniform
    sorry
  have h_edge_pairs : p_edge_pairs = 1 / 256 := by 
    norm_cast
    -- Calculation for the probability that all edges match after rotation
    sorry
  have h_corner_pairs : p_corner_pairs = 1 / 16 := by 
    norm_cast
    -- Calculation for the probability that all corners match after rotation
    sorry
  exact h_center_uniform ▸ h_edge_pairs ▸ h_corner_pairs ▸ rfl

#check probability_of_uniform_color

end probability_of_uniform_color_l697_697599


namespace quadractic_transformation_sum_l697_697415

theorem quadractic_transformation_sum :
  let a := 5
  let h := 2
  let k := -12
  a + h + k = -5 := 
by
  sorry

end quadractic_transformation_sum_l697_697415


namespace problem_statement_l697_697482

noncomputable def g : ℤ → ℤ := sorry

theorem problem_statement :
  (∃ n s, n * s = 6 ∧
     ∀ m n : ℤ, g(m + n) + g(m * n + 1) = g(m) * g(n) + 3) := sorry

end problem_statement_l697_697482


namespace smallest_non_palindromic_power_of_13_l697_697740

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l697_697740


namespace chord_length_squared_l697_697257

-- Let radii of circles be defined
def radius₄ : ℝ := 4
def radius₈ : ℝ := 8
def radius₁₂ : ℝ := 12

-- Let the centers be points on a plane
structure Point (α : Type) := 
  (x : α)
  (y : α)

-- Define points O4, O8, and O12 as centers of the circles
def O₄ : Point ℝ := ⟨0, 0⟩
def O₈ : Point ℝ := ⟨12, 0⟩
def O₁₂ : Point ℝ := ⟨6, 0⟩

-- Define the positions of feet of perpendiculars from centers to the chord
def A₄ : Point ℝ := ⟨0, 4⟩
def A₈ : Point ℝ := ⟨12, 8⟩
def A₁₂ : Point ℝ := ⟨6, 20/3⟩

-- Proving the square of the length of the chord
theorem chord_length_squared : 
  let PQ := λ A : Point ℝ, Real.sqrt ((A₁₂.y - A.y)^2 + (A₁₂.x - A.x)^2)
  in PQ(A₄) ^ 2 + PQ(A₈) ^ 2  = 3584 / 9 := sorry

end chord_length_squared_l697_697257


namespace inequality_solution_l697_697113

theorem inequality_solution (x : ℝ) :
  ((x - 1) * (x - 3) * (x - 5)) / ((x - 2) * (x - 4) * (x - 6)) > 0 ↔
  (x ∈ Iio 1 ∨ x ∈ Ioo 2 3 ∨ x ∈ Ioo 4 5 ∨ x ∈ Ioi 6) :=
by sorry

end inequality_solution_l697_697113


namespace volume_of_cube_with_diagonal_l697_697990

theorem volume_of_cube_with_diagonal (d : ℝ) (h : d = 5 * real.sqrt 3) : 
  ∃ (V : ℝ), V = 125 := 
by
  -- Definitions and conditions from the problem are used directly
  let s := d / real.sqrt 3
  sorry

end volume_of_cube_with_diagonal_l697_697990


namespace triangle_isosceles_l697_697363

theorem triangle_isosceles (A B C P : Type)
  [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint P]
  (h_interior : InteriorPoint P (Triangle A B C))
  (h_angle_PAB : Angle P A B = 10)
  (h_angle_PBA : Angle P B A = 20)
  (h_angle_PCA : Angle P C A = 30)
  (h_angle_PAC : Angle P A C = 40)
  : IsIsoscelesTriangle A B C :=
sorry

end triangle_isosceles_l697_697363


namespace julia_kids_difference_l697_697881

theorem julia_kids_difference :
  let monday_kids := 6
  let tuesday_kids := 17
  let wednesday_kids := 4
  let thursday_kids := 12
  let sunday_kids := 9
  (tuesday_kids + thursday_kids) - (monday_kids + wednesday_kids + sunday_kids) = 10 :=
by
  unfold monday_kids tuesday_kids wednesday_kids thursday_kids sunday_kids
  show (17 + 12) - (6 + 4 + 9) = 10
  sorry

end julia_kids_difference_l697_697881


namespace Petya_rubles_maximum_l697_697956

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l697_697956


namespace maximum_rubles_received_l697_697928

def four_digit_number_of_form_20xx (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (d : ℕ) (n : ℕ) : Prop :=
  n % d = 0

theorem maximum_rubles_received :
  ∃ (n : ℕ), four_digit_number_of_form_20xx n ∧
             divisible_by 1 n ∧
             divisible_by 3 n ∧
             divisible_by 7 n ∧
             divisible_by 9 n ∧
             divisible_by 11 n ∧
             ¬ divisible_by 5 n ∧
             1 + 3 + 7 + 9 + 11 = 31 :=
sorry

end maximum_rubles_received_l697_697928


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697748

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l697_697748


namespace three_digit_ends_with_itself_iff_l697_697307

-- Define the condition for a number to be a three-digit number
def is_three_digit (A : ℕ) : Prop := 100 ≤ A ∧ A ≤ 999

-- Define the requirement of the problem in terms of modular arithmetic
def ends_with_itself (A : ℕ) : Prop := A^2 % 1000 = A

-- The main theorem stating the solution
theorem three_digit_ends_with_itself_iff : 
  ∀ (A : ℕ), is_three_digit A → ends_with_itself A ↔ A = 376 ∨ A = 625 :=
by
  intro A,
  intro h,
  split,
  { intro h1,
    sorry }, -- Proof omitted
  { intro h2,
    cases h2,
    { rw h2, exact dec_trivial },
    { rw h2, exact dec_trivial } }

end three_digit_ends_with_itself_iff_l697_697307


namespace min_sum_squares_l697_697077

theorem min_sum_squares {p q r s t u v w : ℤ}
    (hp : p ≠ q) (hpq : p ≠ r) (hpr : p ≠ s) (hps : p ≠ t) (hpt : p ≠ u) (hpu : p ≠ v) (hpv : p ≠ w)
    (hqr : q ≠ r) (hqs : q ≠ s) (hqt : q ≠ t) (hqu : q ≠ u) (hqv : q ≠ v) (hqw : q ≠ w)
    (hrs : r ≠ s) (hrt : r ≠ t) (hru : r ≠ u) (hrv : r ≠ v) (hrw : r ≠ w)
    (hst : s ≠ t) (hsu : s ≠ u) (hsv : s ≠ v) (hsw : s ≠ w)
    (htu : t ≠ u) (htv : t ≠ v) (htw : t ≠ w)
    (huv : u ≠ v) (huw : u ≠ w)
    (hvw : v ≠ w)
    (hp_set : p ∈ {-6, -4, -3, -1, 1, 3, 5, 8})
    (hq_set : q ∈ {-6, -4, -3, -1, 1, 3, 5, 8})
    (hr_set : r ∈ {-6, -4, -3, -1, 1, 3, 5, 8})
    (hs_set : s ∈ {-6, -4, -3, -1, 1, 3, 5, 8})
    (ht_set : t ∈ {-6, -4, -3, -1, 1, 3, 5, 8})
    (hu_set : u ∈ {-6, -4, -3, -1, 1, 3, 5, 8})
    (hv_set : v ∈ {-6, -4, -3, -1, 1, 3, 5, 8})
    (hw_set : w ∈ {-6, -4, -3, -1, 1, 3, 5, 8}) :
    ∃ p q r s t u v w, (p + q + r + s) ∈ {-6, -4, -3, -1, 1, 3, 5, 8} ∧
    (t + u + v + w) ∈ {-6, -4, -3, -1, 1, 3, 5, 8} ∧
    ((p + q + r + s)^2 + (t + u + v + w)^2) = 5 :=
sorry

end min_sum_squares_l697_697077


namespace _l697_697835

statement theorem sec_tan_difference (x : ℝ) (h : Real.sec x + Real.tan x = 3) : 
  Real.sec x - Real.tan x = 1 / 3 := by
  sorry

end _l697_697835


namespace problem_part1_problem_part2_l697_697369

-- Define what we need to prove
theorem problem_part1 (x : ℝ) (a b : ℤ) 
  (h : (2*x - 21)*(3*x - 7) - (3*x - 7)*(x - 13) = (3*x + a)*(x + b)): 
  a + 3*b = -31 := 
by {
  -- We know from the problem that h holds,
  -- thus the values of a and b must satisfy the condition.
  sorry
}

theorem problem_part2 (x : ℝ) : 
  (x^2 - 3*x + 2) = (x - 1)*(x - 2) := 
by {
  sorry
}

end problem_part1_problem_part2_l697_697369


namespace sum_of_divisors_prime_factors_count_l697_697007

theorem sum_of_divisors_prime_factors_count : 
  ∑ n in (finset.range (900 + 1)), (if 900 % n = 0 then n else 0) = 7 * 13 * 31 → 
  (finset.card (finset.filter (λ p, nat.prime p) 
                        (finset.image nat.prime (↑((finset.range (31 + 1)).filter (|> nat.is_prime)))))) = 3 := 
by {
  cc,
  sorry
}

end sum_of_divisors_prime_factors_count_l697_697007


namespace combined_weight_of_boxes_l697_697470

def weight_box1 : ℝ := 2
def weight_box2 : ℝ := 11
def weight_box3 : ℝ := 5

theorem combined_weight_of_boxes : weight_box1 + weight_box2 + weight_box3 = 18 := by
  sorry

end combined_weight_of_boxes_l697_697470


namespace distance_vertex_to_asymptote_eq_l697_697129

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Definition of the asymptote
def asymptote (x y : ℝ) : Prop :=
  3 * x + 4 * y = 0

-- Vertex of the hyperbola
def vertex := (4 : ℝ, 0 : ℝ)

-- Required theorem: distance from vertex to asymptote is 12/5
theorem distance_vertex_to_asymptote_eq :
  ∃ (d : ℝ), d = 12 / 5 ∧
  ∀ (x y : ℝ), hyperbola x y ∧ x = 4 ∧ y = 0 →
  ∀ (a b : ℝ), asymptote a b ∧ a = 3 * x ∧ b = - (4 * y) →
  dist (x, y) (a / 3, -a / 4) = d :=
sorry

end distance_vertex_to_asymptote_eq_l697_697129


namespace unique_records_l697_697102

variable (Samantha_records : Nat)
variable (shared_records : Nat)
variable (Lily_unique_records : Nat)

theorem unique_records (h1 : Samantha_records = 24) (h2 : shared_records = 15) (h3 : Lily_unique_records = 9) :
  let Samantha_unique_records := Samantha_records - shared_records
  Samantha_unique_records + Lily_unique_records = 18 :=
by
  sorry

end unique_records_l697_697102


namespace correct_option_l697_697178

noncomputable def f_A (x : ℝ) : ℝ := -Real.log x
noncomputable def f_B (x : ℝ) : ℝ := (1/ 2)^x
noncomputable def f_C (x : ℝ) : ℝ := -1 / x
noncomputable def f_D (x : ℝ) : ℝ := x^2 - x

def is_monotonically_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x < f y

theorem correct_option (x : ℝ) :
  (∀ x > 0, is_monotonically_increasing f_C {y : ℝ | y > 0}) ∧
  (∀ x > 0, ¬ is_monotonically_increasing f_A {y : ℝ | y > 0}) ∧
  (∀ x > 0, ¬ is_monotonically_increasing f_B {y : ℝ | y > 0}) ∧
  (∀ x > 0, ¬ is_monotonically_increasing f_D {y : ℝ | y > 0}) :=
sorry

end correct_option_l697_697178


namespace height_of_cylinder_in_hemisphere_l697_697211

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (h_base_parallel : Prop) : ℝ :=
  if h_base_parallel then sqrt (r_hemisphere^2 - r_cylinder^2) else 0

theorem height_of_cylinder_in_hemisphere :
  cylinder_height 3 8 true = sqrt 55 :=
by
  unfold cylinder_height
  simp
  sorry

end height_of_cylinder_in_hemisphere_l697_697211


namespace total_male_students_l697_697041

noncomputable def total_students : ℕ := 1200
noncomputable def sampled_students : ℕ := 200
noncomputable def sampled_females : ℕ := 85

theorem total_male_students :
  let fraction_sampled := sampled_students / total_students,
      total_females := sampled_females / fraction_sampled,
      total_males := total_students - total_females
  in total_males = 690 := by
  sorry

end total_male_students_l697_697041


namespace arctan_roots_sum_eq_pi_div_4_l697_697487

def polynomial (x : ℝ) : ℝ := x^3 - 16 * x + 17

theorem arctan_roots_sum_eq_pi_div_4 (x1 x2 x3 : ℝ) (h1 : polynomial x1 = 0) (h2 : polynomial x2 = 0) (h3 : polynomial x3 = 0) :
  real.arctan x1 + real.arctan x2 + real.arctan x3 = real.pi / 4 := 
by sorry

end arctan_roots_sum_eq_pi_div_4_l697_697487


namespace max_volume_at_6_l697_697172

noncomputable def volume (x : ℝ) : ℝ :=
  x * (36 - 2 * x)^2

theorem max_volume_at_6 :
  ∃ x : ℝ, (0 < x) ∧ (x < 18) ∧ 
  (∀ y : ℝ, (0 < y) ∧ (y < 18) → volume y ≤ volume 6) :=
by
  sorry

end max_volume_at_6_l697_697172


namespace find_f2_l697_697773

-- A condition of the problem is the specific form of the function
def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Given condition
theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by
  sorry

end find_f2_l697_697773


namespace quadratic_perfect_square_form_l697_697562

def quadratic_is_perfect_square (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

theorem quadratic_perfect_square_form (a b c : ℤ) (h : quadratic_is_perfect_square a b c) :
  ∃ d e : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (d * x + e)^2 :=
  sorry

end quadratic_perfect_square_form_l697_697562


namespace maximum_rubles_received_l697_697950

theorem maximum_rubles_received : ∃ (n : ℕ), -- There exists a natural number n such that
  (n ≥ 2000 ∧ n < 2100) ∧                -- condition 1: n is between 2000 and 2100
  ((∃ k1, n = 3^3 * 7 * 11 * k1) ∨         -- condition 2: n is of the form 3^3 * 7 * 11 * k1
   (∃ k2, n = 5 * k2)) ∧                  -- or n is of the form 5 * k2
  let payouts :=                          -- definition: distinguish the payouts for divisibility
    (if n % 1 = 0 then 1 else 0) +        -- payout for divisibility by 1
    (if n % 3 = 0 then 3 else 0) +        -- payout for divisibility by 3
    (if n % 5 = 0 then 5 else 0) +        -- payout for divisibility by 5
    (if n % 7 = 0 then 7 else 0) +        -- payout for divisibility by 7
    (if n % 9 = 0 then 9 else 0) +        -- payout for divisibility by 9
    (if n % 11 = 0 then 11 else 0) in     -- payout for divisibility by 11
  payouts = 31                            -- condition 3: sum of payouts equals 31
  ∧ n = 2079 := sorry                   -- answer: n equals 2079

end maximum_rubles_received_l697_697950


namespace proof_lcm_expression_l697_697186

-- Function to define the least common multiple (LCM)
def lcm (m n : ℕ) : ℕ :=
  m * n / Nat.gcd m n

-- Given conditions
def twelve_sixteen_lcm := lcm 12 16
def eighteen_twentyfour_lcm := lcm 18 24
def eleven_thirteen_lcm := lcm 11 13
def seventeen_nineteen_lcm := lcm 17 19

-- Define p based on the given problem
def p := (twelve_sixteen_lcm * eighteen_twentyfour_lcm + eleven_thirteen_lcm) - seventeen_nineteen_lcm

-- Statement to prove
theorem proof_lcm_expression : p = 3276 := 
by sorry

end proof_lcm_expression_l697_697186


namespace find_multiple_l697_697628

-- Definitions of the divisor, original number, and remainders given in the problem conditions.
def D : ℕ := 367
def remainder₁ : ℕ := 241
def remainder₂ : ℕ := 115

-- Statement of the problem.
theorem find_multiple (N m k l : ℕ) :
  (N = k * D + remainder₁) →
  (m * N = l * D + remainder₂) →
  ∃ m, m > 0 ∧ 241 * m - 115 % 367 = 0 ∧ m = 2 :=
by
  sorry

end find_multiple_l697_697628


namespace max_matches_per_participant_l697_697855

theorem max_matches_per_participant (participants : Finset ℕ) 
  (H₁ : participants.card = 300)
  (H₂ : ∀ (x y : ℕ), x ≠ y → has_match x y → ¬(∃ z, z ≠ x ∧ z ≠ y ∧ has_match x z ∧ has_match y z)) :
  ∃ n, (∀ p ∈ participants, (number_of_matches p ≤ n)) ∧ n = 200 :=
sorry

end max_matches_per_participant_l697_697855


namespace unique_real_solution_l697_697292

noncomputable def cubic_eq (b x : ℝ) : ℝ :=
  x^3 - b * x^2 - 3 * b * x + b^2 - 2

theorem unique_real_solution (b : ℝ) :
  (∃! x : ℝ, cubic_eq b x = 0) ↔ b = 7 / 4 :=
by
  sorry

end unique_real_solution_l697_697292


namespace fg_tangent_to_inscribed_circle_l697_697891

def square (ABCD : set (ℝ × ℝ)) : Prop :=
  ∃ (A B C D : ℝ × ℝ),
  A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2) ∧ D = (0, 2) ∧
  (ABCD = {A, B, C, D} : set (ℝ × ℝ))

def midpoint (p1 p2 m : ℝ × ℝ) : Prop :=
  m = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def is_parallel (l1 l2 : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ (p1 p2 p3 p4 : ℝ × ℝ),
  l1 p1 p2 ∧ l2 p3 p4 → (p2.2 - p1.2) * (p4.1 - p3.1) = (p4.2 - p3.2) * (p2.1 - p1.1)

def tangent_to_circle (FG : ℝ × ℝ → ℝ × ℝ → Prop)
  (circle : ℝ × ℝ → ℝ → Prop) : Prop :=
  ∀ (p q : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ),
  FG p q → circle O r → ∃ t : ℝ, (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2 ∧ p.1 * t = q.1 ∧ p.2 * t = q.2

theorem fg_tangent_to_inscribed_circle 
  (ABCD : set (ℝ × ℝ)) (circle : ℝ × ℝ → ℝ → Prop)
  (A B C D E F G : ℝ × ℝ) (r : ℝ) 
  (P Q : ℝ × ℝ) :
  square ABCD →
  midpoint A B E →
  P = (2, 1) →
  Q = (1, 2) →
  (∃ p q : ℝ × ℝ, p = G ∧ q = (2, r / G.1)) →
  is_parallel (λ p1 p2, p1 = A ∧ p2 = G) (λ p1 p2, p1 = E ∧ p2 = F) →
  tangent_to_circle (λ p q, (p.1, p.2) = F ∧ (q.1, q.2) = G) circle :=
sorry

end fg_tangent_to_inscribed_circle_l697_697891


namespace sequence_sum_l697_697785

def sequence (n : ℕ) : ℤ :=
  if n = 0 then 0
  else if n = 1 then 1
  else sequence (n - 1)^2 - 1

theorem sequence_sum : sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5 = -1 := by
  sorry

end sequence_sum_l697_697785


namespace radius_of_O2016_l697_697862

-- Define the centers and radii of circles
variable (a : ℝ) (n : ℕ) (r : ℕ → ℝ)

-- Conditions
-- Radius of the first circle
def initial_radius := r 1 = 1 / (2 * a)
-- Sequence of the radius difference based on solution step
def radius_recursive := ∀ n > 1, r (n + 1) - r n = 1 / a

-- The final statement to be proven
theorem radius_of_O2016 (h1 : initial_radius a r) (h2 : radius_recursive a r) :
  r 2016 = 4031 / (2 * a) := 
by sorry

end radius_of_O2016_l697_697862


namespace solve_complex_equation_l697_697336

-- Definitions for real numbers x, y and complex number z.
variables (x y : ℝ) (z : ℂ)

-- The condition where z is a complex number defined by real parts x and y.
def z_def : ℂ := x + y * complex.I

-- The main statement translating the problem into Lean
theorem solve_complex_equation : 
  (∃ (x y : ℝ) (z : ℂ), z = x + y * complex.I ∧ z^6 = -8 ∧
   (z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 5 * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 7 * real.pi / 6) ∨ 
    z = (-(8: ℂ)^(1/6)) * complex.exp (complex.I * 11 * real.pi / 6) ∨ 
    z = complex.I * (2 : ℝ)^(1 / 3) ∨ 
    z = -complex.I * (2 : ℝ)^(1 / 3))) := 
sorry

end solve_complex_equation_l697_697336


namespace exists_polynomial_distinct_powers_of_2_l697_697343

open Polynomial

variable (n : ℕ) (hn : n > 0)

theorem exists_polynomial_distinct_powers_of_2 :
  ∃ P : Polynomial ℤ, P.degree = n ∧ (∃ (k : Fin (n + 1) → ℕ), ∀ i j : Fin (n + 1), i ≠ j → 2 ^ k i ≠ 2 ^ k j ∧ (∀ i, P.eval i.val = 2 ^ k i)) :=
sorry

end exists_polynomial_distinct_powers_of_2_l697_697343


namespace average_speed_l697_697589

-- Definitions of conditions
def speed_first_hour : ℝ := 120
def speed_second_hour : ℝ := 60
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := 2

-- Theorem stating the equivalent proof problem
theorem average_speed : total_distance / total_time = 90 := by
  sorry

end average_speed_l697_697589


namespace amount_diff_l697_697019

variable (x : ℕ) (y : ℕ) (a : ℕ) (b : ℕ)

theorem amount_diff (h1 : x = 690)
                    (h2 : y = 1500)
                    (h3 : a = 50 * x / 100)
                    (h4 : b = 25 * y / 100) :
  b - a = 30 :=
by
  rw [h1, h2, h3, h4]
  -- Sorry for the proof part, as requested.
  sorry

end amount_diff_l697_697019


namespace triangle_AD_length_l697_697459

theorem triangle_AD_length
  (A B C D : Type)
  (h : ∀ M, euclidean_geometry.point M [M = A, M = B, M = C])
  (AB AC : ℝ)
  (p : A ≠ B)
  (q : A ≠ C)
  (r : B ≠ C)
  (AB_eq : AB = 13)
  (AC_eq : AC = 20)
  (perpendicular : euclidean_geometry.perpendicular A D B C)
  (BD CD : ℝ)
  (ratio : BD / CD = 3 / 4) :
  let AD := sqrt (128) in AD = 8 * sqrt 2 :=
by
  sorry

end triangle_AD_length_l697_697459


namespace triangle_inequality_geq_l697_697373

variable {a b c : ℝ} (S : ℝ)

theorem triangle_inequality_geq :
  (∀ (a b c : ℝ) (S : ℝ), (a^2 + b^2 + c^2) ≥ 4 * sqrt 3 * S) :=
sorry

end triangle_inequality_geq_l697_697373


namespace variety_show_arrangements_l697_697248

/-
  Given seven acts: dance, comic dialogue, sketch, singing, magic, acrobatics, and opera.
  Dance, comic dialogue, and sketch cannot be adjacent to each other.
  Prove that the number of different arrangements of these acts is 1440.
-/
theorem variety_show_arrangements : 
    let acts := ["dance", "comic dialogue", "sketch", "singing", "magic", "acrobatics", "opera"]
    ∃ (arrangements: ℕ), 
      arrangements = 1440 ∧
      (dance # acts → ¬ (adjacent_any_three acts "dance" "comic dialogue" "sketch"))
    :=
  sorry

end variety_show_arrangements_l697_697248


namespace sin_theta_value_l697_697401

theorem sin_theta_value (θ : ℝ) (h1 : 10 * tan θ = 4 * cos θ) (h2 : 0 < θ ∧ θ < π) : 
  sin θ = ( - 5 + real.sqrt 41) / 4 :=
sorry

end sin_theta_value_l697_697401


namespace maximize_correlation_coefficient_l697_697044

noncomputable def points : List (ℝ × ℝ) := [(1, 3), (2, 4), (3, 10), (4, 6), (10, 12)]

def on_line (p1 p2 p : ℝ × ℝ) : Prop :=
  (p.snd - p1.snd) * (p2.fst - p1.fst) = (p.fst - p1.fst) * (p2.snd - p1.snd)

theorem maximize_correlation_coefficient :
  ∀ {A B C D E : ℝ × ℝ},
    [A, B, C, D, E] = points →
    (on_line A B D) ∧ (on_line A B E) ∧ ¬ (on_line A B C) →
    ¬ [A, B, D, E].pairwise (≠) → R :=
sorry

end maximize_correlation_coefficient_l697_697044


namespace train_cross_time_l697_697187

-- Declare the conditions as definitions
def train_length : ℝ := 45  -- length in meters
def train_speed_km_hr : ℝ := 108  -- speed in km/hr
def conversion_factor : ℝ := 5 / 18  -- conversion factor from km/hr to m/s

-- Convert the speed to meters per second
def train_speed_m_s : ℝ := train_speed_km_hr * conversion_factor  -- speed in m/s

-- Define the time to cross the pole
def time_to_cross_pole : ℝ := train_length / train_speed_m_s  -- time in seconds

-- The theorem to prove
theorem train_cross_time :
  time_to_cross_pole = 1.5 :=
by
  sorry

end train_cross_time_l697_697187


namespace distinct_prime_factors_sum_of_divisors_900_l697_697005

theorem distinct_prime_factors_sum_of_divisors_900 :
  ∃ n : ℕ, n = 3 ∧
  ∃ σ : ℕ → ℕ, 
    σ 900 = (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5 + 5^2) ∧
    ∀ p : ℕ, p.prime → ∀ d : ℕ, d ∣ σ 900 → ∃ k : ℕ, (p^k).count_factors = n := sorry

end distinct_prime_factors_sum_of_divisors_900_l697_697005


namespace inverse_function_l697_697845

def f (a : Real) (x : Real) : Real := 1 + Real.log x / Real.log a

noncomputable def f_inverse (x : Real) : Real := 2^(x - 1)

theorem inverse_function (a : Real) (h : f a 8 = 4) : ∃ g : Real → Real, ∀ x, g (f a x) = x :=
begin
  use f_inverse,
  intro x,
  unfold f f_inverse,
  sorry
end

end inverse_function_l697_697845


namespace programs_produce_same_output_l697_697098

def sum_program_a : ℕ :=
  let S := (Finset.range 1000).sum (λ i => i + 1)
  S

def sum_program_b : ℕ :=
  let S := (Finset.range 1000).sum (λ i => 1000 - i)
  S

theorem programs_produce_same_output :
  sum_program_a = sum_program_b := by
  sorry

end programs_produce_same_output_l697_697098


namespace gaussian_integral_prob_dist_negative_binomial_prob_dist_l697_697972

-- Definition and proof of Gaussian integral equality
theorem gaussian_integral_prob_dist :
  (1 / Real.sqrt (2 * Real.pi)) * ∫ x, Real.exp (-x^2 / 2) = 1 :=
sorry

-- Definition and proof of negative binomial distribution equality
theorem negative_binomial_prob_dist (p : ℝ) (k : ℕ) (hp : 0 < p ∧ p ≤ 1) :
  ∑' n in finset.Icc k ∞, (nat.choose (n - 1) (k - 1) * p ^ k * (1 - p) ^ (n - k)) = 1 :=
sorry

end gaussian_integral_prob_dist_negative_binomial_prob_dist_l697_697972


namespace part_I_part_II_part_III_l697_697783

variable (n : ℕ)
variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Condition definitions:
def positive_sequence : Prop :=
  ∀ (k : ℕ), 0 < a k

def seq_initial_condition : Prop :=
  a 1 = 1

def seq_diff_condition : Prop :=
  ∀ (k i : ℕ), i ≤ k → k < n → a (k + 1) - a k = a i

def sum_of_first_n_terms : Prop :=
  S n = ∑ i in Finset.range n, a (i + 1)

-- Proof problems:
theorem part_I 
  (h_pos : positive_sequence a)
  (h_init : seq_initial_condition a)
  (h_diff : seq_diff_condition a n) :
  ∀ (k : ℕ), k < n → (a (k + 1) - a k) ≥ 1 := sorry

theorem part_II 
  (h_pos : positive_sequence a)
  (h_init : seq_initial_condition a)
  (h_geometric : ∀ (k : ℕ), k < n → a (k + 1) = 2 * a k) :
  ∀ (m : ℕ), m < n → a m = 2^(m - 1) := sorry

theorem part_III 
  (h_pos : positive_sequence a)
  (h_init : seq_initial_condition a)
  (h_geometric : ∀ (k : ℕ), k < n → a (k + 1) = 2 * a k)
  (h_sum : sum_of_first_n_terms S) :
  ∀ (n : ℕ), 1 / 2 * n * (n + 1) ≤ S n ∧ S n ≤ 2^n - 1 := sorry

end part_I_part_II_part_III_l697_697783


namespace max_ordered_pairs_divisibility_l697_697083

open Set

theorem max_ordered_pairs_divisibility (m n : ℕ)
  (A : Fin m → Set ℕ)
  (h_disjoint : ∀ i j : Fin m, i ≠ j → Disjoint (A i) (A j))
  (h_card : ∀ i : Fin m, (A i).card = n)
  (h_divisibility : ∀ i : Fin m, ∀ x ∈ A i, ∀ y ∈ A ((i + 1) % m), ¬(x ∣ y)) :
  ∃ p : ℕ, p = ((m - 1) ∑ k in finset.range m, n^2) :=
  sorry

end max_ordered_pairs_divisibility_l697_697083


namespace sec_tan_equation_l697_697829

theorem sec_tan_equation (x : ℝ) (h : Real.sec x + Real.tan x = 3) : Real.sec x - Real.tan x = 1 / 3 :=
sorry

end sec_tan_equation_l697_697829


namespace part_I_a1_part_I_a_n_part_II_l697_697904

open Nat

-- Step 1: Definitions based on conditions
def S_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ :=
  (4 * a_n n - (2 ^ (n + 1))) / 3 + 2 / 3

def a1 (a_n : ℕ → ℤ) : Prop := a_n 1 = 2

def a_n (a_n : ℕ → ℤ) : Prop := ∀ n : ℕ, a_n n = 4^n - 2^n

def T_n (S_n : (ℕ → ℤ) → ℕ → ℤ) (T_n : (ℕ → ℤ) → ℕ → ℚ) (a_n : ℕ → ℤ) : ℕ → ℚ :=
  λ n, (2 ^ n).toRat / S_n a_n n

def sum_T (T_n : (ℕ → ℤ) → ℕ → ℚ) (S_n : (ℕ → ℤ) → ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∑ i in range n, T_n S_n a_n i < (3 / 2).toRat

-- Step 2: Theorem statements
theorem part_I_a1 (a_n : ℕ → ℤ) : a1 a_n :=
sorry

theorem part_I_a_n (a_n : ℕ → ℤ) : a_n a_n :=
sorry

theorem part_II (S_n : (ℕ → ℤ) → ℕ → ℤ) (T_n : (ℕ → ℤ) → ℕ → ℚ) (a_n : ℕ → ℤ) : sum_T T_n S_n a_n :=
sorry

end part_I_a1_part_I_a_n_part_II_l697_697904


namespace complex_number_solution_l697_697697

open Complex

noncomputable def findComplexNumber (z : ℂ) : Prop :=
  abs (z - 2) = abs (z + 4) ∧ abs (z - 2) = abs (z - Complex.I * 2)

theorem complex_number_solution :
  ∃ z : ℂ, findComplexNumber z ∧ z = -1 + Complex.I :=
by
  sorry

end complex_number_solution_l697_697697


namespace three_digit_numbers_with_square_ending_in_them_l697_697301

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l697_697301


namespace monotonicity_of_f_l697_697275

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonicity_of_f : 
  ∀ x : ℝ, 0 < x ∧ x < 5 → 
    (x < (1 / Real.exp 1) → ∀ y : ℝ, y < (1 / Real.exp 1) ∧ 0 < y → f' y < 0) ∧ 
    (x > (1 / Real.exp 1) → ∀ y : ℝ, (1 / Real.exp 1) < y ∧ y < 5 → f' y > 0) := 
by
  sorry

end monotonicity_of_f_l697_697275


namespace simplest_quadratic_radical_l697_697647

variable (a : ℝ) (ha : a > 0)

def expr_A : ℝ := Real.sqrt 27
def expr_B : ℝ := Real.sqrt 15
def expr_C : ℝ := Real.sqrt (3 * a^2)
def expr_D : ℝ := Real.sqrt (1 / a)

theorem simplest_quadratic_radical :
  expr_B = Real.sqrt 15 ∧
  (∀ x ∈ {expr_A, expr_C, expr_D}, Simplify(x) ≠ expr_B) :=
by
  sorry

end simplest_quadratic_radical_l697_697647


namespace number_of_divisors_of_n_squared_l697_697624

-- Define the conditions
def has_four_divisors (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p * q

def num_divisors (k : ℕ) : ℕ :=
  let f := k.factors in
  f.foldl (λ acc x, acc * (x.2 + 1)) 1

-- Create the proof problem
theorem number_of_divisors_of_n_squared (n : ℕ) (h : has_four_divisors n) :
  num_divisors (n * n) = 9 :=
sorry

end number_of_divisors_of_n_squared_l697_697624


namespace binary_digit_difference_l697_697175

theorem binary_digit_difference (n1 n2 : ℕ) (h1 : n1 = 300) (h2 : n2 = 1400) : 
  (nat.bit_length n2 - nat.bit_length n1) = 2 := by
  sorry

end binary_digit_difference_l697_697175


namespace sum_of_lucky_tickets_divisible_by_13_l697_697194

-- Define a six-digit number as a structure with two parts
structure SixDigitNumber :=
  (first_part : Fin 1000)  -- First three digits (0 to 999)
  (second_part : Fin 1000) -- Last three digits (0 to 999)

-- Define what it means for a ticket to be lucky
def isLuckyTicket (n : SixDigitNumber) : Prop :=
  (n.first_part.val.digits 10).sum = (n.second_part.val.digits 10).sum

-- The sum of all lucky ticket numbers
def sumOfLuckyTickets : Nat :=
  List.sum $ (List.range 1000).bind $ λ x => (List.range 1000).filterMap $ λ y =>
    if isLuckyTicket ⟨⟨x, by linarith [Nat.lt_succ_self 999]⟩, ⟨y, by linarith [Nat.lt_succ_self 999]⟩⟩
    then some (x * 1000 + y)
    else none

-- The theorem that needs to be proved
theorem sum_of_lucky_tickets_divisible_by_13 :
  sumOfLuckyTickets % 13 = 0 := 
sorry

end sum_of_lucky_tickets_divisible_by_13_l697_697194


namespace bowling_average_decrease_l697_697182

noncomputable def number_of_wickets (avg_before avg_dec runs_last wickets_last wickets_after runs_after : ℝ) : ℝ := 
  let total_runs_before := avg_before * wickets_before
  let total_runs_after := total_runs_before + runs_last
  (total_runs_after - runs_after) / (wickets_after - wickets_last)

theorem bowling_average_decrease
  (avg_before avg_dec runs_last : ℝ)
  (wickets_last : ℕ)
  (h_avg_before : avg_before = 12.4)
  (h_avg_dec : avg_dec = 0.4)
  (h_runs_last : runs_last = 26)
  (h_wickets_last : wickets_last = 3) :
  number_of_wickets avg_before avg_dec runs_last wickets_last (wickets + wickets_last) (avg_before - avg_dec) = 25 :=
by 
  sorry

end bowling_average_decrease_l697_697182


namespace cylinder_height_in_hemisphere_l697_697217

noncomputable def height_of_cylinder (r_hemisphere r_cylinder : ℝ) : ℝ :=
  real.sqrt (r_hemisphere^2 - r_cylinder^2)

theorem cylinder_height_in_hemisphere :
  let r_hemisphere := 8
  let r_cylinder := 3
  height_of_cylinder r_hemisphere r_cylinder = real.sqrt 55 :=
by
  sorry

end cylinder_height_in_hemisphere_l697_697217


namespace z_real_z_pure_imaginary_z_second_quadrant_l697_697902

noncomputable def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2 * m - 3) (m^2 + 3 * m + 2)

theorem z_real (m : ℝ) : z m.im = 0 ↔ m = -1 ∨ m = -2 := by sorry

theorem z_pure_imaginary (m : ℝ) : z m.re = 0 ∧ z m.im ≠ 0 ↔ m = 3 := by sorry

theorem z_second_quadrant (m : ℝ) : Complex.re (z m) < 0 ∧ Complex.im (z m) > 0 ↔ -1 < m ∧ m < 3 := by sorry

end z_real_z_pure_imaginary_z_second_quadrant_l697_697902


namespace petya_max_rubles_l697_697943

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l697_697943


namespace find_solutions_to_z6_eq_neg8_l697_697330

noncomputable def solution_set : Set ℂ :=
  {z | ∃ x y : ℝ, z = x + y * I ∧ 
  (x^6 - 15 * x^4 * y^2 + 15 * x^2 * y^4 - y^6 = -8) ∧
  (6 * x^5 * y - 20 * x^3 * y^3 + 6 * x * y^5 = 0)}

theorem find_solutions_to_z6_eq_neg8 :
  solution_set = 
  { 
    complex.sqrt 2 + complex.i * complex.sqrt 2, 
    - complex.sqrt 2 + complex.i * complex.sqrt 2, 
    - complex.sqrt 2 - complex.i * complex.sqrt 2, 
    complex.sqrt 2 - complex.i * complex.sqrt 2 
  } :=
sorry

end find_solutions_to_z6_eq_neg8_l697_697330


namespace tan_alpha_second_quadrant_l697_697827

theorem tan_alpha_second_quadrant (α : ℝ) 
(h_cos : Real.cos α = -4/5) 
(h_quadrant : π/2 < α ∧ α < π) : 
  Real.tan α = -3/4 :=
by
  sorry

end tan_alpha_second_quadrant_l697_697827


namespace unique_n_for_permuted_squares_l697_697691

theorem unique_n_for_permuted_squares:
  ∀ (n : ℕ), 
    (∃ k : ℕ, n = 2 * k + 1 ∧ n > 1) → 
    (∃ (a : Fin (k+1) → Fin (k+1)), 
      (∀ (i : Fin (k+1)), a i ∈ List.ofFn (λ (x : Fin (k+1)), x)) ∧ 
      (∀ (i : Fin k), 
        (a (Fin.append 1 i)).val^2 - (a i).val^2 ≡ (a (Fin.append 1 (Fin.append 1 i))).val^2 - (a (Fin.append 1 i)).val^2 [MOD n])
    ) ↔ (n = 3 ∨ n = 5) :=
begin
  sorry
end

end unique_n_for_permuted_squares_l697_697691


namespace number_of_values_a_l697_697847

theorem number_of_values_a :
  let a_set : Set Int := {a | a ≤ 0 ∧ a ≥ -2}
  in a_set.card = 3 :=
by
  sorry

end number_of_values_a_l697_697847


namespace find_root_and_m_l697_697801

theorem find_root_and_m {x : ℝ} {m : ℝ} (h : ∃ x1 x2 : ℝ, (x1 = 1) ∧ (x1 + x2 = -m) ∧ (x1 * x2 = 3)) :
  ∃ x2 : ℝ, (x2 = 3) ∧ (m = -4) :=
by
  obtain ⟨x1, x2, h1, h_sum, h_product⟩ := h
  have hx1 : x1 = 1 := h1
  rw [hx1] at h_product
  have hx2 : x2 = 3 := by linarith [h_product]
  have hm : m = -4 := by
    rw [hx1, hx2] at h_sum
    linarith
  exact ⟨x2, hx2, hm⟩

end find_root_and_m_l697_697801


namespace minimum_area_PMN_probability_distance_from_line_l697_697366

noncomputable def line : set (ℝ × ℝ) := { p | p.1 - 2 * p.2 - 2 * real.sqrt 5 = 0 }
def circle : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = 2 }
def M : ℝ × ℝ := (2 * real.sqrt 5, 0)
def N : ℝ × ℝ := (0, -real.sqrt 5)

theorem minimum_area_PMN :
  ∃ P : ℝ × ℝ, P ∈ circle ∧ P ≠ M ∧ P ≠ N →
  (1 / 2 * abs (M.1 * N.2 - N.1 * M.2) * (2 - real.sqrt (P.1 ^ 2 + P.2 ^ 2))) = (10 - 5 * real.sqrt 2) / 2 :=
sorry

theorem probability_distance_from_line :
  ∃ P : ℝ × ℝ, P ∈ circle ∧ (2 - real.sqrt 2 ≤ (P.1 - 2 * P.2 - 2 * real.sqrt 5) / real.sqrt (1 + 4) ∧ (P.1 - 2 * P.2 - 2 * real.sqrt 5) / real.sqrt (1 + 4) < 1) →
  (real.sqrt 2 - 1) / (2 * real.sqrt 2) = (2 - real.sqrt 2) / 4 :=
sorry

end minimum_area_PMN_probability_distance_from_line_l697_697366


namespace maximum_rubles_received_max_payment_possible_l697_697938

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l697_697938


namespace smallest_non_palindrome_power_of_13_l697_697754

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l697_697754


namespace perimeter_of_original_square_not_integer_integer_perimeter_rectangles_l697_697635

theorem perimeter_of_original_square_not_integer_integer_perimeter_rectangles {a : ℚ} (hₘ : ∃ (l₁ l₂ w₁ w₂ : ℝ), l₁ + w₁ = l₂ + w₂ ∧ (2 * l₁ + 2 * w₁).denom = 1 ∧ (2 * l₂ + 2 * w₂).denom = 1 ) : ¬ (a.denom = 1) where
 P := 4 * a
-- Assuming 'a' is the side length of the square and the square is cut with a straight cut into two rectangles
-- with integer perimeters (hₘ).

end perimeter_of_original_square_not_integer_integer_perimeter_rectangles_l697_697635


namespace problem_part1_problem_part2_problem_part3_l697_697362

variable {a : ℕ → ℝ} {S : ℕ → ℝ} (λ : ℝ)

-- Conditions
def sequence_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) : Prop :=
  a 1 = 1 ∧ (∀ n, a n ≠ 0) ∧ (∀ n, λ * S n = a n * a (n + 1) + 1)

-- Prove that {a_(2n-1)} is an arithmetic sequence
theorem problem_part1 (h : sequence_conditions a S λ) :
  ∃ d, ∀ n, a (2 * n + 1) = a 1 + n * d :=
sorry

-- Check for λ such that {a_n} is an arithmetic sequence
theorem problem_part2 (h : sequence_conditions a S 4) :
  ∀ n, a (n + 1) - a n = 2 :=
sorry

-- Sum of the first n terms of the sequence b_n
noncomputable def b (n : ℕ) : ℝ :=
  (-1) ^ (n - 1) * 4 * n / (a n * a (n + 1))

theorem problem_part3 (h : sequence_conditions a S 4) :
  ∀ n, (sum (λ i, b i) (range n)) = if even n then (2 * n) / (2 * n + 1) else (2 * n + 2) / (2 * n + 1) :=
sorry

end problem_part1_problem_part2_problem_part3_l697_697362


namespace intersection_points_and_verification_l697_697274

theorem intersection_points_and_verification :
  (∃ x y : ℝ, y = -3 * x ∧ y + 3 = 9 * x ∧ x = 1 / 4 ∧ y = -3 / 4) ∧
  ¬ (y = 2 * (1 / 4) - 1 ∧ (2 * (1 / 4) - 1 = -3 / 4)) :=
by
  sorry

end intersection_points_and_verification_l697_697274


namespace opposite_angles_equal_l697_697099

theorem opposite_angles_equal {α β : ℝ} (h : OppositeAngles α β) : α = β := by
  sorry

-- Definition of opposite angles (to be defined appropriately for the specific context)
def OppositeAngles (α β : ℝ) : Prop :=
  -- Assume a proper definition relating opposite angles
  sorry

end opposite_angles_equal_l697_697099


namespace area_ratio_l697_697069

-- Define the side length of the equilateral triangle ABC.
def side_length : ℝ := 1  -- we can normalize the side length to 1 for simplicity

-- Areas of equilateral triangles with given side lengths.
def area (s : ℝ) : ℝ := (√3 / 4) * s^2

-- Define the side length of triangle A'B'C' after extension
def extended_side_length : ℝ := 5 * side_length

-- State the main theorem to state the ratio of the areas of the triangles.
theorem area_ratio :
  let ABC_area := area side_length in
  let A'B'C'_area := area extended_side_length in
  (A'B'C'_area / ABC_area) = 25 :=
by
  sorry

end area_ratio_l697_697069


namespace max_rubles_can_receive_l697_697921

-- Define the four-digit number 2079
def number := 2079

-- Check divisibility conditions
def divisible_by_1 := number % 1 = 0
def divisible_by_3 := number % 3 = 0
def divisible_by_5 := number % 5 = 0
def divisible_by_7 := number % 7 = 0
def divisible_by_9 := number % 9 = 0
def divisible_by_11 := number % 11 = 0

-- Define the sum of rubles under the conditions described
def sum_rubles := if divisible_by_1 then 1 else 0 + if divisible_by_3 then 3 else 0 + if divisible_by_7 then 7 else 0 + if divisible_by_9 then 9 else 0 + if divisible_by_11 then 11 else 0

-- The final proof statement
theorem max_rubles_can_receive : sum_rubles = 31 :=
by
  unfold sum_rubles
  -- Confirm the result by simplification, let's skip the proof as required
  sorry

end max_rubles_can_receive_l697_697921


namespace valid_match_schedule_l697_697564

noncomputable def numberOfGames (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem valid_match_schedule (n dates maxGames: ℕ) (h_n_pos : n > 1) (h_dates_pos: dates > 0) (h_maxGames_pos: maxGames > 0)
  (h_constraint: numberOfGames n / maxGames > dates):
  ∃ (total_games : ℕ), total_games = numberOfGames n ∧ (date_ways : ℕ) ∧ date_ways = 0 :=
by
  have total_games := numberOfGames n
  have date_ways := if total_games / maxGames > dates then 0 else sorry
  use [total_games, date_ways]
  exact ⟨rfl, rfl⟩

end valid_match_schedule_l697_697564


namespace intersection_M_N_l697_697815

noncomputable def M : Set ℝ := { x | x^2 + x - 2 = 0 }
def N : Set ℝ := { x | x < 0 }

theorem intersection_M_N : M ∩ N = { -2 } := by
  sorry

end intersection_M_N_l697_697815


namespace Petya_rubles_maximum_l697_697955

theorem Petya_rubles_maximum :
  ∃ n, (2000 ≤ n ∧ n < 2100) ∧ (∀ d ∈ [1, 3, 5, 7, 9, 11], n % d = 0 → true) → 
  (1 + ite (n % 1 = 0) 1 0 + ite (n % 3 = 0) 3 0 + ite (n % 5 = 0) 5 0 + ite (n % 7 = 0) 7 0 + ite (n % 9 = 0) 9 0 + ite (n % 11 = 0) 11 0 = 31) :=
begin
  sorry
end

end Petya_rubles_maximum_l697_697955


namespace smallest_non_palindromic_power_of_13_l697_697728

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l697_697728


namespace find_polynomial_P_l697_697885

-- Definitions of the roots and conditions
variables {a b c : ℝ}
variables (h_roots : ∀ x, (x ^ 3 + 4 * x ^ 2 + 7 * x + 10 = 0) → (x = a ∨ x = b ∨ x = c))
variables (h_sum : a + b + c = -4)
variables (h_p_a : P(a) = b + c)
variables (h_p_b : P(b) = a + c)
variables (h_p_c : P(c) = a + b)
variables (h_p_sum : P(-4) = -22)

-- Definition of P(x)
noncomputable def P (x : ℝ) : ℝ :=
  (8 / 9) * x^3 + (44 / 9) * x^2 + (71 / 9) * x + (2 / 3)

-- The main theorem
theorem find_polynomial_P :
  ∀ x, P(x) = (8 / 9) * x^3 + (44 / 9) * x^2 + (71 / 9) * x + (2 / 3) := by
  sorry

end find_polynomial_P_l697_697885


namespace find_c_gen_formula_l697_697868

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
a 1 = 2 ∧
(∀ n, a (n + 1) = a n + c * n) ∧
(2 + c) * (2 + c) = 2 * (2 + 3 * c)

theorem find_c (a : ℕ → ℕ) : ∃ c, seq a c :=
by
  sorry

theorem gen_formula (a : ℕ → ℕ) (c : ℕ) (h : seq a c) : (∀ n, a n = n^2 - n + 2) :=
by
  sorry

end find_c_gen_formula_l697_697868


namespace translate_line_upwards_l697_697574

theorem translate_line_upwards (x y y' : ℝ) (h : y = -2 * x) (t : y' = y + 4) : y' = -2 * x + 4 :=
by
  sorry

end translate_line_upwards_l697_697574


namespace ann_frosting_cakes_l697_697501

theorem ann_frosting_cakes (normalRate sprainedRate cakes : ℕ) (H1 : normalRate = 5) (H2 : sprainedRate = 8) (H3 : cakes = 10) :
  (sprainedRate * cakes) - (normalRate * cakes) = 30 :=
by
  -- Substitute the provided values into the expression
  rw [H1, H2, H3]
  -- Evaluate the expression
  norm_num

end ann_frosting_cakes_l697_697501


namespace find_three_digit_numbers_l697_697317

theorem find_three_digit_numbers : {A : ℕ // 100 ≤ A ∧ A ≤ 999 ∧ (A^2 % 1000 = A)} = {376, 625} :=
sorry

end find_three_digit_numbers_l697_697317


namespace max_rubles_l697_697965

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l697_697965


namespace triangle_area_is_two_l697_697031

noncomputable def triangle_area (b c : ℝ) (angle_A : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_A

theorem triangle_area_is_two
  (A B C : ℝ) (a b c : ℝ)
  (hA : A = π / 4)
  (hCondition : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B)
  (hBC : b * c = 4 * Real.sqrt 2) : 
  triangle_area b c A = 2 :=
by
  -- actual proof omitted
  sorry

end triangle_area_is_two_l697_697031


namespace order_of_a_b_c_l697_697072

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem order_of_a_b_c : b < a ∧ a < c :=
by
  sorry

end order_of_a_b_c_l697_697072


namespace siblings_of_Lily_l697_697148

-- Define the children and their attributes.
def Child : Type := { name : String, eyeColor : String, hairColor : String, ageGroup : String }

def Lily : Child := { name := "Lily", eyeColor := "Blue", hairColor := "Blonde", ageGroup := "Middle-aged" }
def Mark : Child := { name := "Mark", eyeColor := "Green", hairColor := "Blonde", ageGroup := "Old" }
def Sophie : Child := { name := "Sophie", eyeColor := "Brown", hairColor := "Black", ageGroup := "Young" }
def Oliver : Child := { name := "Oliver", eyeColor := "Blue", hairColor := "Blonde", ageGroup := "Old" }
def Emma : Child := { name := "Emma", eyeColor := "Blue", hairColor := "Brown", ageGroup := "Middle-aged" }
def Noah : Child := { name := "Noah", eyeColor := "Green", hairColor := "Black", ageGroup := "Middle-aged" }
def Mia : Child := { name := "Mia", eyeColor := "Brown", hairColor := "Blonde", ageGroup := "Young" }
def Lucas : Child := { name := "Lucas", eyeColor := "Blue", hairColor := "Blonde", ageGroup := "Middle-aged" }

-- Predicate for being siblings, requiring sharing at least two characteristics.
def areSiblings (c1 c2 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∧ c1.hairColor = c2.hairColor) ∨
  (c1.eyeColor = c2.eyeColor ∧ c1.ageGroup = c2.ageGroup) ∨
  (c1.hairColor = c2.hairColor ∧ c1.ageGroup = c2.ageGroup)

-- Main theorem stating the siblings of Lily
theorem siblings_of_Lily : {c : Child // areSiblings Lily c ∧ (c.name = "Emma" ∨ c.name = "Lucas")} := sorry

end siblings_of_Lily_l697_697148


namespace number_of_sister_pairs_l697_697394

noncomputable def f : ℝ → ℝ 
| x if x < 0  := x^2 + 2 * x
| x if x ≥ 0 := 2 / (Real.exp x)

def is_sister_pair (A B : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  (A.2 = f A.1) ∧ (B.2 = f B.1) ∧ (A.1 = -B.1) ∧ (A.2 = -B.2)

theorem number_of_sister_pairs : 
  (∑' (A B : ℝ × ℝ), if is_sister_pair A B f then 1 else 0) = 2 :=
sorry

end number_of_sister_pairs_l697_697394


namespace estimate_correctness_l697_697427

noncomputable def total_species_estimate (A B C : ℕ) : Prop :=
  A = 2400 ∧ B = 1440 ∧ C = 3600

theorem estimate_correctness (A B C taggedA taggedB taggedC caught : ℕ) 
  (h1 : taggedA = 40) 
  (h2 : taggedB = 40) 
  (h3 : taggedC = 40)
  (h4 : caught = 180)
  (h5 : 3 * A = taggedA * caught) 
  (h6 : 5 * B = taggedB * caught) 
  (h7 : 2 * C = taggedC * caught) 
  : total_species_estimate A B C := 
by
  sorry

end estimate_correctness_l697_697427


namespace water_used_is_12_l697_697857

def water_used (W : ℝ) : ℝ := (3 / 5) * W
def vinegar_used : ℝ := (5 / 6) * 18
def total_mixture (W : ℝ) : ℝ := water_used W + vinegar_used

theorem water_used_is_12 (W : ℝ) (h1 : vinegar_used = 15) (h2 : total_mixture W = 27) :
  water_used W = 12 := 
sorry

end water_used_is_12_l697_697857


namespace complete_square_sum_l697_697416

theorem complete_square_sum (a h k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) →
  a + h + k = -5 :=
by
  intro h1
  sorry

end complete_square_sum_l697_697416


namespace inequality_solution_l697_697115

theorem inequality_solution (x : ℝ) :
  ((x - 1) * (x - 3) * (x - 5)) / ((x - 2) * (x - 4) * (x - 6)) > 0 ↔
  (x ∈ Iio 1 ∨ x ∈ Ioo 2 3 ∨ x ∈ Ioo 4 5 ∨ x ∈ Ioi 6) :=
by sorry

end inequality_solution_l697_697115


namespace coeff_x2_l697_697272

def poly := 5 * (x^2 - 2 * x^3) + 3 * (x - x^2 + 2 * x^4) - (3 * x^4 - 2 * x^2)

theorem coeff_x2 :
  coeff poly 2 = 4 := by
  sorry

end coeff_x2_l697_697272


namespace triangle_bisectors_circle_l697_697468

theorem triangle_bisectors_circle :
  ∀ (A B C : ℝ) (A1 B1 C1 : ℝ),
    (A + B + C = π) →  -- Sum of angles in a triangle
    (∃ P : ℝ, P = (A A1 * cos (A / 2) + B B1 * cos (B / 2) + C C1 * cos (C / 2)) / (sin A + sin B + sin C)) →
    P = 2 := 
by
  sorry

end triangle_bisectors_circle_l697_697468


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697228

theorem height_of_cylinder_inscribed_in_hemisphere :
  ∀ (r_cylinder r_hemisphere : ℝ),
  r_cylinder = 3 →
  r_hemisphere = 8 →
  (∃ h : ℝ, h = real.sqrt (r_hemisphere^2 - r_cylinder^2)) :=
begin
  intros r_cylinder r_hemisphere h1 h2,
  use real.sqrt (r_hemisphere^2 - r_cylinder^2),
  rw [h1, h2],
  simp,
  sorry
end

end height_of_cylinder_inscribed_in_hemisphere_l697_697228


namespace number_of_real_solutions_l697_697708

noncomputable def f (x : ℝ) : ℝ :=
  ∑ n in (Finset.range 100).map (λ i, i + 1), 2 * (n : ℝ) / (x - n)

theorem number_of_real_solutions :
  (Finset.range 101).card = 101 :=
sorry

end number_of_real_solutions_l697_697708


namespace find_integer_cube_sum_l697_697322

-- Define the problem in Lean
theorem find_integer_cube_sum : ∃ n : ℤ, n^3 = (n-1)^3 + (n-2)^3 + (n-3)^3 := by
  use 6
  sorry

end find_integer_cube_sum_l697_697322


namespace min_value_of_M_l697_697485

noncomputable def f (p q x : ℝ) : ℝ := x^2 + p * x + q

theorem min_value_of_M (p q M : ℝ) :
  (M = max (|f p q 1|) (max (|f p q (-1)|) (|f p q 0|))) →
  (0 > f p q 1 → 0 > f p q (-1) → 0 > f p q 0 → M = 1 / 2) :=
sorry

end min_value_of_M_l697_697485


namespace ann_frosting_time_l697_697497

theorem ann_frosting_time (time_normal time_sprained n : ℕ) (h1 : time_normal = 5) (h2 : time_sprained = 8) (h3 : n = 10) : 
  ((time_sprained * n) - (time_normal * n)) = 30 := 
by 
  sorry

end ann_frosting_time_l697_697497


namespace problem1_problem2_problem3_l697_697812

-- Function definition
def f (m n : ℝ) (x : ℝ) : ℝ := (m * x - n) / x - Real.log x

-- Problem 1: Tangent parallel condition
theorem problem1 (m n : ℝ) : (∃ k : ℝ, (n - 4) / 4 = k ∧ k = 1) → n = 8 := by
  intro h
  obtain ⟨k, hk1, hk2⟩ := h
  have hk : k = 1 := hk2
  have hn : (n - 4) / 4 = 1 := by rw [hk1, hk]
  linarith

-- Problem 2: Maximum value analysis
theorem problem2 (m n : ℝ) (h : 1 ≤ n / 2) : (m - 2 - Real.log (n / 2)) ≤ f m n 1 := by
  sorry

-- Problem 3: Sum of zeros
theorem problem3 (m x₁ x₂ : ℝ) (hx₁_positive : 0 < x₁) (hx₂_ordered : x₁ < x₂) (hx₁_eq : m * x₁ - x₁ * Real.log x₁ = 1) (hx₂_eq : m * x₂ - x₂ * Real.log x₂ = 1) : x₁ + x₂ > 2 := by
  sorry

end problem1_problem2_problem3_l697_697812


namespace count_n_for_g_n_prime_l697_697899

def g (n : ℕ) : ℕ := (n.divisors : finset ℕ).sum

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_n_for_g_n_prime : 
  (finset.range 31).filter (λ n, is_prime (g n)).card = 5 := 
sorry

end count_n_for_g_n_prime_l697_697899


namespace solve_for_a_l697_697913

theorem solve_for_a
  (a b c : ℝ)
  (h1 : sqrt a = sqrt b + sqrt c)
  (h2 : b = 52 - 30 * sqrt 3)
  (h3 : c = a - 2) : a = 27 :=
sorry

end solve_for_a_l697_697913


namespace closest_log5_b2023_l697_697681

noncomputable def star (a b : ℝ) : ℝ := a ^ Real.logBase 5 b 
noncomputable def circ (a b : ℝ) : ℝ := a ^ Real.logBase b 5

noncomputable def b : ℕ → ℝ
| 3 := circ 5 2
| (n + 1) := star (circ n (n - 1)) (b n)

theorem closest_log5_b2023 : abs (logBase 5 (b 2023) - 2021) < 0.5 := sorry

end closest_log5_b2023_l697_697681


namespace chickens_and_cages_l697_697596

theorem chickens_and_cages (x : ℕ) (h : 4 * x + 1 = 5 * (x - 1)) : 
  let chickens := 4 * x + 1 in
  chickens = 25 ∧ x = 6 := 
by
  sorry

end chickens_and_cages_l697_697596


namespace smallest_non_palindrome_power_of_13_l697_697759

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l697_697759


namespace smallest_non_palindrome_power_of_13_l697_697758

def is_palindrome (n : ℕ) : Bool :=
  let s := (toString n).toList
  s == s.reverse

def is_power_of_13 (n : ℕ) : Bool :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindrome_power_of_13 : ∃ n : ℕ, n = 13 ∧ 
  (¬ is_palindrome n) ∧ (is_power_of_13 n) ∧ (∀ m : ℕ, m < n → ¬ is_palindrome m ∨ ¬ is_power_of_13 m) :=
    sorry

end smallest_non_palindrome_power_of_13_l697_697758


namespace mutually_exclusive_l697_697610

def male_students := {1, 2, 3, 4}
def female_students := {5, 6}

def group := male_students ∪ female_students

def event_at_least_one_male (s : Set ℕ) : Prop :=
  ∃ m ∈ male_students, m ∈ s

def event_all_female (s : Set ℕ) : Prop :=
  ∀ f ∈ s, f ∈ female_students

theorem mutually_exclusive :
  ∀ s, event_at_least_one_male s → ¬event_all_female s :=
by
  intro s h1 h2
  obtain ⟨m, hm_male, hm_s⟩ := h1
  have hf := h2 m hm_s
  exact hm_male hf

end mutually_exclusive_l697_697610


namespace solve_cubic_root_eq_l697_697293

theorem solve_cubic_root_eq (x : ℝ) : (∃ x, 3 - x / 3 = -8) -> x = 33 :=
by
  sorry

end solve_cubic_root_eq_l697_697293


namespace find_value_l697_697772

theorem find_value (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 108) : a^2 * b + a * b^2 = 108 :=
by
  sorry

end find_value_l697_697772


namespace appearance_not_definitely_different_l697_697528

-- Defining basic structure and appearance from different positions
def object := Type
def position := Type
def appearance (o : object) (p : position) := Prop

-- Statement indicating appearances observed from different positions are not necessarily different
theorem appearance_not_definitely_different (o : object) (p1 p2 : position)
  (h : ∀ (o : object) (p1 p2 : position), appearance o p1 ↔ appearance o p2) : 
  ¬ (∀ (o : object) (p1 p2 : position), appearance o p1 → ¬ appearance o p2) :=
by
  -- Proof is omitted as per instructions
  sorry

end appearance_not_definitely_different_l697_697528


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697227

theorem height_of_cylinder_inscribed_in_hemisphere :
  ∀ (r_cylinder r_hemisphere : ℝ),
  r_cylinder = 3 →
  r_hemisphere = 8 →
  (∃ h : ℝ, h = real.sqrt (r_hemisphere^2 - r_cylinder^2)) :=
begin
  intros r_cylinder r_hemisphere h1 h2,
  use real.sqrt (r_hemisphere^2 - r_cylinder^2),
  rw [h1, h2],
  simp,
  sorry
end

end height_of_cylinder_inscribed_in_hemisphere_l697_697227


namespace find_mu_l697_697766

open Complex

noncomputable def complex_z : ℂ := 3 * exp (Complex.I * θ) -- z = 3e^{iϕ}

theorem find_mu (z : ℂ) (h1 : abs z = 3)
    (h2 : ∃ μ : ℝ, μ > 2 ∧ μ * z - z = 3 * z * (9 * exp (2 * Complex.I * θ) - 1))
    (h3 : abs (z^3 - z) = abs ((μ : ℂ) * z - z)) :
    ∃ μ, μ = 1 + Real.sqrt 82 :=
begin
    existsi (1 + Real.sqrt 82),
    sorry
end

end find_mu_l697_697766


namespace salary_change_l697_697436

noncomputable def average (x : ℝ^5) : ℝ :=
  (x[0] + x[1] + x[2] + x[3] + x[4]) / 5

noncomputable def variance (x : ℝ^5) : ℝ :=
  let avg := average x in
  ((x[0] - avg)^2 + (x[1] - avg)^2 + (x[2] - avg)^2 + (x[3] - avg)^2 + (x[4] - avg)^2) / 5

theorem salary_change (x : ℝ^5) (h_avg : average x = 3500) (h_var : variance x = 45) :
  average (λ i, x i + 100) = 3600 ∧ variance (λ i, x i + 100) = 45 := by
  sorry

end salary_change_l697_697436


namespace decimal_to_binary_11_l697_697265

theorem decimal_to_binary_11 : ∃ n : ℕ, n = 11 ∧ nat.digit_to_nat [1,0,1,1] 2 = 11 :=
by
  sorry

end decimal_to_binary_11_l697_697265


namespace sauce_free_percentage_l697_697207

theorem sauce_free_percentage (total_weight : ℕ) (sauce_weight : ℕ) (h1 : total_weight = 200) (h2 : sauce_weight = 50) : 
  (total_weight - sauce_weight) / total_weight.to_rat = 75 / 100 :=
by
  sorry

end sauce_free_percentage_l697_697207


namespace car_avg_speed_l697_697560

def avg_speed_problem (d1 d2 t : ℕ) : ℕ :=
  (d1 + d2) / t

theorem car_avg_speed (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 70) (h2 : d2 = 90) (ht : t = 2) :
  avg_speed_problem d1 d2 t = 80 := by
  sorry

end car_avg_speed_l697_697560


namespace parabola_shift_l697_697988

theorem parabola_shift (x : ℝ) : 
  let y := -2 * x^2 
  let y1 := -2 * (x + 1)^2 
  let y2 := y1 - 3 
  y2 = -2 * x^2 - 4 * x - 5 := 
by 
  sorry

end parabola_shift_l697_697988


namespace trapezoid_height_l697_697984

section TrapezoidHeight

variables (a : ℝ) [fact (0 < a)] -- the area a^2 implies a is positive

-- Define the properties of the isosceles trapezoid and its diagonals
structure IsoscelesTrapezoid :=
  (A B C D : ℝ)
  (AB_eq_CD : B = C)
  (diagonals_perpendicular : A * C + B * D = 0)
  (area : ℝ)
  (area_eq : area = a^2)

-- The main theorem stating the height of the trapezoid
theorem trapezoid_height (t : IsoscelesTrapezoid a) : ∃ h : ℝ, h = a :=
by 
  sorry -- proof is omitted

end TrapezoidHeight

end trapezoid_height_l697_697984


namespace problem_statement_l697_697371

theorem problem_statement (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_condition : ∀ x, f'' x < 2 * f x) :
  exp(4034) * f (-2017) > f 0 ∧ f 2017 < exp(4034) * f 0 :=
by
  sorry

end problem_statement_l697_697371


namespace find_length_AX_l697_697432

theorem find_length_AX 
  (A B C X : Type)
  (BC BX AC : ℝ)
  (h_BC : BC = 36)
  (h_BX : BX = 30)
  (h_AC : AC = 27)
  (h_bisector : ∃ (x : ℝ), x = BX / BC ∧ x = AX / AC ) :
  ∃ AX : ℝ, AX = 22.5 := 
sorry

end find_length_AX_l697_697432


namespace blake_change_l697_697660

theorem blake_change :
  ∃ (change : ℕ), 
    let lollipop_cost := 2 in
    let lollipops := 4 in
    let chocolate_cost := lollipop_cost * 4 in
    let chocolates := 6 in
    let total_cost := (lollipop_cost * lollipops) + (chocolate_cost * chocolates) in
    let amount_given := 10 * 6 in
    change = amount_given - total_cost := 
by
  use 4
  sorry

end blake_change_l697_697660


namespace choose_student_for_competition_l697_697143

def average_score (student : Type) : ℕ
def variance_score (student : Type) : ℝ

constant A : Type
constant B : Type
constant C : Type
constant D : Type

axiom avg_A : average_score A = 92
axiom var_A : variance_score A = 3.6

axiom avg_B : average_score B = 95
axiom var_B : variance_score B = 3.6

axiom avg_C : average_score C = 95
axiom var_C : variance_score C = 7.4

axiom avg_D : average_score D = 95
axiom var_D : variance_score D = 8.1

theorem choose_student_for_competition : B = B := by sorry

end choose_student_for_competition_l697_697143


namespace find_complex_number_l697_697705

theorem find_complex_number (z : ℂ) (h1 : complex.abs (z - 2) = complex.abs (z + 4))
  (h2 : complex.abs (z + 4) = complex.abs (z - 2 * complex.I)) : z = -1 + complex.I :=
by
  sorry

end find_complex_number_l697_697705


namespace three_digit_square_ends_with_self_l697_697310

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l697_697310


namespace smallest_c_condition_l697_697279

theorem smallest_c_condition :
  ∃ (c : ℕ), c > 0 ∧ (∀ (n : ℕ), n > 0 → ∀ (digit : ℕ), digit ∈ digits 10 (c ^ n + 2014) → digit < 5) ∧
  ∀ k : ℕ, k > 0 → k < c → (∃ n : ℕ, n > 0 ∧ ∃ digit : ℕ, digit ∈ digits 10 (k ^ n + 2014) ∧ digit ≥ 5) :=
begin
  sorry
end

end smallest_c_condition_l697_697279


namespace derivative_of_f_at_pi_over_2_l697_697016

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = -5 :=
sorry

end derivative_of_f_at_pi_over_2_l697_697016


namespace problem1_problem2_problem3_l697_697518

-- Part I
theorem problem1 : 
  (sqrt 3 * sin (-20 * Real.pi / 3) / tan (11 * Real.pi / 3) - cos (13 * Real.pi / 4) * tan (-35 * Real.pi / 4)) = 
  (sqrt 3 + sqrt 2) / 2 :=
by
  sorry

-- Part II
theorem problem2 : 
  (sqrt (1 - 2 * sin (10 * Real.pi / 180) * cos (10 * Real.pi / 180)) / (cos (10 * Real.pi / 180) - sqrt (1 - cos^2 (170 * Real.pi / 180)))) = 
  1 :=
by
  sorry

-- Part III
theorem problem3 (a : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < Real.pi) (hroots : (sin θ, cos θ) = (rootOfQuadratic2 2 (-1) a)) : 
  cos θ - sin θ = - sqrt 7 / 2 :=
by
  sorry

end problem1_problem2_problem3_l697_697518


namespace equilateral_triangle_side_length_l697_697167

def perimeter := 8
theorem equilateral_triangle_side_length (P : ℝ) (hP : P = 8) : ℝ :=
  let l := P / 3
  by {

  }

end equilateral_triangle_side_length_l697_697167


namespace transformed_parabola_equation_l697_697986

theorem transformed_parabola_equation :
    ∀ (x : ℝ), let f := λ x, -2 * x ^ 2 in
    (f (x + 1) - 3) = -2 * x ^ 2 - 4 * x - 5 :=
by
  intro x
  let f := λ x, -2 * x ^ 2
  sorry

end transformed_parabola_equation_l697_697986


namespace cubs_eighth_inning_home_runs_l697_697422

theorem cubs_eighth_inning_home_runs :
  ∀ (x : ℕ), 
  2 + 1 + x = (1 + 1) + 3 →
  x = 2 :=
by
  intro x
  assume h
  sorry

end cubs_eighth_inning_home_runs_l697_697422


namespace probability_of_total_sum_of_dice_is_odd_l697_697568

-- Define the fair coin and fair dice rolls
def fair_coin : Pmf Bool :=
  Pmf.ofMultiset { True, False }.toMultiset

def fair_die : Pmf ℕ :=
  Pmf.ofMultiset { 1, 2, 3, 4, 5, 6 }.toMultiset

-- Define the problem conditions
def three_fair_coins_tossed_once : Pmf (List Bool) :=
  Pmf.replicateM 3 fair_coin

def heads_resulting_to_dice_rolls (heads : ℕ) : Pmf (List ℕ) :=
  Pmf.replicateM (2 * heads) fair_die

-- Define the problem statement and the probability function
def probability_odd_sum_of_dice : Pmf (List ℕ) → ℚ :=
  λ l, if l.sum % 2 = 1 then 1 else 0

noncomputable def total_probability_odd_sum : ℚ :=
  (∑ outcome in (three_fair_coins_tossed_once.support), 
  (three_fair_coins_tossed_once outcome) * 
  (let heads := outcome.count id in 
  ∑ rolls in (heads_resulting_to_dice_rolls heads).support,
  (heads_resulting_to_dice_rolls heads rolls) * 
  (probability_odd_sum_of_dice rolls)))

-- Proof to show the probability of odd sum is 7/16
theorem probability_of_total_sum_of_dice_is_odd : 
  total_probability_odd_sum = 7 / 16 :=
sorry

end probability_of_total_sum_of_dice_is_odd_l697_697568


namespace total_crayons_l697_697283

-- Definitions for the conditions
def crayons_per_child : Nat := 12
def number_of_children : Nat := 18

-- The statement to be proved
theorem total_crayons :
  (crayons_per_child * number_of_children = 216) := 
by
  sorry

end total_crayons_l697_697283


namespace find_sum_f_a5_a6_l697_697377

noncomputable theory

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

def sequence_a (a : ℕ → ℝ) : Prop :=
a 1 = -1 ∧ ∀ n, a (n+1) = 1 - 2^(n+1)

theorem find_sum_f_a5_a6 (f : ℝ → ℝ) (a : ℕ → ℝ)
  (S : ℕ → ℝ) (h_odd : is_odd_function f)
  (h_periodic : ∀ x, f (3 - x) = -f x)
  (h_f1 : f 1 = -3)
  (h_S : ∀ n, S n = 2 * a n + n) :
  f (a 5) + f (a 6) = 3 :=
sorry

end find_sum_f_a5_a6_l697_697377


namespace sum_inequality_l697_697486

theorem sum_inequality
  (n : ℕ) (h_n : 2 ≤ n)
  (a : ℕ → ℝ) (h_pos : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < a k)
  (h_sum : ∑ k in Finset.range (n + 1), a k = 1) :
  ∑ k in Finset.range (n + 1), (a k / (1 - a k)) * (∑ j in Finset.range k, a j)^2 < 1 / 3 :=
by
  sorry

end sum_inequality_l697_697486


namespace solution_correct_l697_697976

-- Define the equation as a predicate
def equation (x : ℝ) : Prop :=
  sqrt (x + 1) - 1 = x / (sqrt (x + 1) + 1)

-- State the main theorem
theorem solution_correct (x : ℝ) : equation x → x ≥ -1 :=
by
  intros
  sorry

end solution_correct_l697_697976


namespace solid_of_revolution_volume_l697_697204

noncomputable def volume_of_solid_of_revolution (a α : ℝ) : ℝ :=
  (π * a^3 / 3) * cot(α / 2) * cos(α / 2) * cot α

theorem solid_of_revolution_volume (a α : ℝ) :
  0 < a → 0 < α ∧ α < π → volume_of_solid_of_revolution a α = (π * a^3 / 3) * cot(α / 2) * cos(α / 2) * cot α :=
by
  -- We state that the volume_of_solid_of_revolution satisfies the desired volume formula
  sorry

end solid_of_revolution_volume_l697_697204


namespace probability_is_two_thirds_l697_697842

def is_divisible_by_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def probability_n_times_n_plus_1_divisible_by_3 : ℚ :=
  let favorable_outcomes := (finset.range 100).filter (λ n, is_divisible_by_3 (n * (n + 1))).card in
  favorable_outcomes / 99

theorem probability_is_two_thirds :
  probability_n_times_n_plus_1_divisible_by_3 = 2 / 3 :=
sorry

end probability_is_two_thirds_l697_697842


namespace correct_option_for_sentence_completion_l697_697091

-- Define the mathematical formalization of the problem
def sentence_completion_problem : String × (List String) := 
    ("One of the most important questions they had to consider was _ of public health.", 
     ["what", "this", "that", "which"])

-- Define the correct answer
def correct_answer : String := "that"

-- The formal statement of the problem in Lean 4
theorem correct_option_for_sentence_completion 
    (problem : String × (List String)) (answer : String) :
    answer = "that" :=
by
  sorry  -- Proof to be completed

end correct_option_for_sentence_completion_l697_697091


namespace max_rubles_l697_697964

theorem max_rubles (n : ℕ) (h1 : 2000 ≤ n) (h2 : n ≤ 2099) :
  (∃ k, n = 99 * k) → 
  31 ≤
    (if n % 1 = 0 then 1 else 0) +
    (if n % 3 = 0 then 3 else 0) +
    (if n % 5 = 0 then 5 else 0) +
    (if n % 7 = 0 then 7 else 0) +
    (if n % 9 = 0 then 9 else 0) +
    (if n % 11 = 0 then 11 else 0) :=
sorry

end max_rubles_l697_697964


namespace Norb_age_l697_697496

def isPrime (n : ℕ) : Prop := ∀ m : ℕ, 2 ≤ m → m ≤ Int.natAbs n → n % m ≠ 0

theorem Norb_age :
  ∃ age : ℕ,
    (∀ g ∈ [26, 30, 35, 39, 42, 43, 45, 47, 49, 52, 53], g < age → (Σ g', g' ∈ [26, 30, 35, 39, 42, 43, 45, 47, 49, 52, 53] ∧ g' < age) ≥ 6)
    ∧ ((Σ g', g' ∈ [26, 30, 35, 39, 42, 43, 45, 47, 49, 52, 53] ∧ (g' = age - 1 ∨ g' = age + 1)) = 3)
    ∧ (isPrime age)
    ∧ (age ∈ [26, 30, 35, 39, 42, 43, 45, 47, 49, 52, 53]) :=
  sorry

end Norb_age_l697_697496


namespace coefficients_sq_diff_eq_l697_697777

theorem coefficients_sq_diff_eq (a : ℕ → ℚ) :
  (∀ x : ℚ, (2 - real.sqrt 3 * x) ^ 8 = ∑ i in finset.range (9), a i * x ^ i) →
  (∑ i in finset.range 5, a (2 * i)) ^ 2 - (∑ i in finset.range 4, a (2 * i + 1)) ^ 2 = 1 :=
by
  intro h
  sorry

end coefficients_sq_diff_eq_l697_697777


namespace smallest_non_palindromic_power_of_13_l697_697732

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13
  (smallest_non_palindrome_power : ℕ)
  (h_p13 : is_power_of_13 smallest_non_palindrome_power)
  (h_palindrome : ¬ is_palindrome smallest_non_palindrome_power)
  (h_smallest : ∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → smallest_non_palindrome_power ≤ m)
  : smallest_non_palindrome_power = 2197 :=
sorry

end smallest_non_palindromic_power_of_13_l697_697732


namespace morgans_payment_l697_697646

variable (C D X : ℝ) (h : C > D)

def initial_share_alex := (2 / 3 : ℝ)
def initial_share_morgan := (1 / 3 : ℝ)

-- Given that the reimbursement needs to be split equally:
def reimbursed_share := X / 2

-- Total actual expenses
def total_expenses := C + D + X
def total_expenses_adjusted := C + D

-- Share adjusted for reimbursement:
def alex_adjusted := (2 / 3) * total_expenses_adjusted - X
def morgan_adjusted := (1 / 3) * total_expenses_adjusted + X

-- Calculate how much more must Morgan pay Alex:
def morgans_extra_payment := ((1 / 3) * total_expenses_adjusted + X - D)

theorem morgans_payment (h : C > D) :
  morgans_extra_payment C D X = (1 / 3) * C - (2 / 3) * D + X :=
sorry

end morgans_payment_l697_697646


namespace statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l697_697192

-- Definitions of conditions
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℝ)
  (angles : Fin n → ℝ)

def circumscribed (P : Polygon n) : Prop := sorry -- Definition of circumscribed
def inscribed (P : Polygon n) : Prop := sorry -- Definition of inscribed
def equal_sides (P : Polygon n) : Prop := ∀ i j, P.sides i = P.sides j
def equal_angles (P : Polygon n) : Prop := ∀ i j, P.angles i = P.angles j

-- The statements to be proved
theorem statement_I : ∀ P : Polygon n, circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_II : ∃ P : Polygon n, inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_III : ∃ P : Polygon n, circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_IV : ∀ P : Polygon n, inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_V : ∀ (P : Polygon 5), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VI : ∀ (P : Polygon 6), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VII : ∀ (P : Polygon 5), inscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VIII : ∃ (P : Polygon 6), inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_IX : ∀ (P : Polygon 5), circumscribed P → equal_angles P → equal_sides P := sorry

theorem statement_X : ∃ (P : Polygon 6), circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_XI : ∀ (P : Polygon 5), inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_XII : ∀ (P : Polygon 6), inscribed P → equal_angles P → equal_sides P := sorry

end statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l697_697192


namespace inequality_power_of_half_l697_697775

theorem inequality_power_of_half (x y : ℝ) (h : x > y) : (1 / 2) ^ x < (1 / 2) ^ y := 
  sorry

end inequality_power_of_half_l697_697775


namespace height_of_cylinder_inscribed_in_hemisphere_l697_697226

theorem height_of_cylinder_inscribed_in_hemisphere :
  ∀ (r_cylinder r_hemisphere : ℝ),
  r_cylinder = 3 →
  r_hemisphere = 8 →
  (∃ h : ℝ, h = real.sqrt (r_hemisphere^2 - r_cylinder^2)) :=
begin
  intros r_cylinder r_hemisphere h1 h2,
  use real.sqrt (r_hemisphere^2 - r_cylinder^2),
  rw [h1, h2],
  simp,
  sorry
end

end height_of_cylinder_inscribed_in_hemisphere_l697_697226


namespace grape_juice_amount_l697_697608

def drink_volume := 300
def orange_percent := 0.25
def watermelon_percent := 0.40

def grape_juice_volume (total_juice orange_juice watermelon_juice : ℝ) : ℝ :=
  total_juice - (orange_juice + watermelon_juice)

theorem grape_juice_amount :
  grape_juice_volume drink_volume (orange_percent * drink_volume) (watermelon_percent * drink_volume) = 105 :=
by
  sorry

end grape_juice_amount_l697_697608


namespace valid_zogbonian_sentences_l697_697121

def words : List String := ["zorb", "plink", "murb", "flox"]

def is_invalid_sentence (s : List String) : Bool :=
  (s = ["zorb", "plink", "murb"]) ∨
  (s = ["zorb", "plink", "flox"]) ∨
  (s = ["murb", "flox", "zorb"]) ∨
  (s = ["murb", "flox", "plink"])

def all_sentences : List (List String) :=
  List.product (List.product words words) words

def count_valid_sentences : Nat :=
  all_sentences.foldl (λ acc s => if is_invalid_sentence s then acc else acc + 1) 0

theorem valid_zogbonian_sentences : count_valid_sentences = 50 := by
  sorry

end valid_zogbonian_sentences_l697_697121


namespace three_digit_numbers_with_square_ending_in_them_l697_697300

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

theorem three_digit_numbers_with_square_ending_in_them (A : ℕ) :
  is_three_digit A → (A^2 % 1000 = A) → A = 376 ∨ A = 625 :=
by
  sorry

end three_digit_numbers_with_square_ending_in_them_l697_697300


namespace simple_state_petya_wins_complex_state_vasya_wins_l697_697434

-- Definitions

def is_tree {V : Type} (G : simple_graph V) : Prop :=
  G.connected ∧ G.acyclic

inductive game_result
| petya_wins
| vasya_wins

-- Statements

theorem simple_state_petya_wins {V : Type} (G : simple_graph V) :
  (is_tree G) → (game_result.petya_wins) :=
by
  intros tree_prop
  sorry

theorem complex_state_vasya_wins {V : Type} (G : simple_graph V) :
  (¬ is_tree G) → (game_result.vasya_wins) :=
by
  intros not_tree_prop
  sorry

end simple_state_petya_wins_complex_state_vasya_wins_l697_697434


namespace triangle_solution_l697_697033

noncomputable def triangle_proof (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∀ (a b c : ℝ),
  B = 2 * C →
  c = 2 →
  a = 1 →
  b = Real.sqrt 6 ∧ Real.sin (2 * B - Real.pi / 3) = (7 * Real.sqrt 3 - Real.sqrt 15) / 16

-- Defining the properties of our triangle
def triangle_ABC : Prop :=
  ∃ (A B C a b c : ℝ), ∀ (h : triangle_proof A B C a b c), h a b c

-- Main statement
theorem triangle_solution : triangle_ABC :=
begin
  sorry
end

end triangle_solution_l697_697033


namespace Miquel_point_lies_on_segment_l697_697511

noncomputable
def cyclic_quad (A B C D : Type) (circumcircle : (Type → Prop)) :=
  ∃ (P Q R S : A), circumcircle A ∧ circumcircle B ∧ circumcircle C ∧ circumcircle D

noncomputable
def intersect (A B : Type) (U V : A) : A → Prop :=
  ∃ (X : A), (X ∈ Line(U, V)) ∧ (X ∈ Line(A, B))

noncomputable
def Miquel_point (circumcircle_BCE circumcircle_CDF : (Type → Prop)) : Type :=
  ∃ (P : Type), P ∈ circumcircle_BCE ∧ P ∈ circumcircle_CDF

theorem Miquel_point_lies_on_segment :
  ∀ (A B C D E F : Type), 
  cyclic_quad A B C D → 
  intersect A B D C E → 
  intersect B C D A F →
  Miquel_point (circumcircle (triangle BCE)) (circumcircle (triangle CDF)) →
  lies_on_segment P E F :=
by
  -- We state the conditions here, 
  -- but the proof is omitted.
  sorry

end Miquel_point_lies_on_segment_l697_697511


namespace bruce_current_age_l697_697665

theorem bruce_current_age :
  ∃ B : ℕ, (∀ S : ℕ, S = 8 → B + 6 = 3 * (S + 6)) ∧ B = 36 :=
by
  existsi 36
  intros S hS
  rw [hS]
  norm_num
  sorry

end bruce_current_age_l697_697665


namespace function_property_l697_697893

theorem function_property (k a b : ℝ) (h : a ≠ b) (h_cond : k * a + 1 / b = k * b + 1 / a) : 
  k * (a * b) + 1 = 0 := 
by 
  sorry

end function_property_l697_697893


namespace sequence_general_term_and_sum_l697_697556

theorem sequence_general_term_and_sum (a n : ℕ) (S : ℕ → ℕ) (b T : ℕ → ℝ):
  (∀ n, S n = (3 / 2 : ℝ) * a n - (1 / 2 : ℝ) * a 1) →
  a 1 = 5 →
  ((a 3 + 5) : ℕ) + (a 4 - 15)  = 2 * (a 3 + 5) →
  (∀ n, a n = 3^n) ∧
  (∀ n, let b_n := (4 * real.log (a n) - 1) / a n in
         (T n = ∑ i in finset.range n, b (i + 1)) →
         T n = (5 / 2 : ℝ) - (2 * n + 5 / 2 : ℝ) * (1 / 3 : ℝ)^n
  ) :=
begin
  sorry
end

end sequence_general_term_and_sum_l697_697556


namespace complex_product_l697_697887

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex numbers z1 and z2
def z1 : ℂ := 1 - i
def z2 : ℂ := 3 + i

-- Statement of the problem
theorem complex_product : z1 * z2 = 4 - 2 * i := by
  sorry

end complex_product_l697_697887


namespace Kevin_finishes_first_l697_697879

variable (j r : ℝ)
-- Conditions based on the problem statements
def Jenny_lawn : ℝ := j
def Kevin_lawn : ℝ := j / 3
def Lana_lawn : ℝ := j / 4

def Jenny_rate : ℝ := r
def Lana_rate : ℝ := r / 4
def Kevin_rate : ℝ := r / 2

-- Define the mowing times
def Jenny_time : ℝ := Jenny_lawn / Jenny_rate
def Kevin_time : ℝ := Kevin_lawn / Kevin_rate
def Lana_time : ℝ := Lana_lawn / Lana_rate

-- The theorem to be proved which represents the condition Jared finishes first
theorem Kevin_finishes_first :
  Kevin_time j r < Jenny_time j r ∧ Kevin_time j r < Lana_time j r :=
by
  sorry

end Kevin_finishes_first_l697_697879


namespace most_important_measure_for_preference_l697_697088

-- Definitions of the types of zongzi.
inductive Zongzi
| red_bean
| salted_egg_yolk
| meat

-- The statistical measures considered.
inductive StatisticalMeasure
| mean
| variance
| median
| mode

-- The main proof problem statement.
theorem most_important_measure_for_preference : 
  ∀ (survey_data : List Zongzi), 
  (most_important : StatisticalMeasure) →
  most_important = StatisticalMeasure.mode :=
by
  assume survey_data
  assume most_important
  sorry

end most_important_measure_for_preference_l697_697088


namespace smallest_non_palindromic_power_of_13_l697_697742

def is_palindrome (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_power_of_13 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 13 ^ k

theorem smallest_non_palindromic_power_of_13 : ∃ n : ℕ, is_power_of_13 n ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, is_power_of_13 m ∧ ¬ is_palindrome m → n ≤ m) :=
by
  use 169
  split
  · use 2
    exact rfl
  split
  · exact dec_trivial
  · intros m ⟨k, hk⟩ hpm
    sorry

end smallest_non_palindromic_power_of_13_l697_697742


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697753

-- Definition of palindrome
def is_palindrome (n : ℕ) : Prop :=
  let s := toString n
  s = s.reverse

-- Definition of power of 13
def power_of_13 (n : ℕ) : Prop :=
  ∃ k, n = 13^k

-- Theorem statement
theorem smallest_non_palindrome_power_of_13_is_2197 :
  ∀ n, power_of_13 n → ¬is_palindrome n → n ≥ 13 → n = 2197 :=
by
  sorry

end smallest_non_palindrome_power_of_13_is_2197_l697_697753


namespace inequality_proof_l697_697107

def inequality_solution (x : ℝ) : Prop :=
  (x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)

theorem inequality_proof (x : ℝ) :
  (1 < x ∨ x < 2) ∧ (2 < x ∨ x < 3) ∧ (3 < x ∨ x < 4) ∧ (4 < x ∨ x < 5) ∧ (5 < x ∨ x < 6) →
  inequality_solution x :=
begin
  -- Proof goes here
  sorry,
end

end inequality_proof_l697_697107


namespace tangent_length_from_A_to_circle_l697_697867

noncomputable def point_A_polar : (ℝ × ℝ) := (6, Real.pi)
noncomputable def circle_eq_polar (θ : ℝ) : ℝ := -4 * Real.cos θ

theorem tangent_length_from_A_to_circle : 
  ∃ (length : ℝ), length = 2 * Real.sqrt 3 ∧ 
  (∃ (ρ θ : ℝ), point_A_polar = (6, Real.pi) ∧ ρ = circle_eq_polar θ) :=
sorry

end tangent_length_from_A_to_circle_l697_697867


namespace maximum_rubles_received_max_payment_possible_l697_697933

def is_four_digit (n : ℕ) : Prop :=
  2000 ≤ n ∧ n < 2100

def divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def payment (n : ℕ) : ℕ :=
  if divisible_by n 1 then 1 else 0 +
  (if divisible_by n 3 then 3 else 0) +
  (if divisible_by n 5 then 5 else 0) +
  (if divisible_by n 7 then 7 else 0) +
  (if divisible_by n 9 then 9 else 0) +
  (if divisible_by n 11 then 11 else 0)

theorem maximum_rubles_received :
  ∀ (n : ℕ), is_four_digit n → payment n ≤ 31 :=
sorry

theorem max_payment_possible :
  ∃ (n : ℕ), is_four_digit n ∧ payment n = 31 :=
sorry

end maximum_rubles_received_max_payment_possible_l697_697933


namespace smallest_d_l697_697134

-- Define the conditions as given in the problem
def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- The main theorem to prove
theorem smallest_d (d : ℕ) : (2021 % d = 11) → d ≥ 12 → d = 15 :=
begin
  -- proof steps are not needed
  sorry
end

end smallest_d_l697_697134


namespace number_of_values_a_l697_697846

theorem number_of_values_a :
  let a_set : Set Int := {a | a ≤ 0 ∧ a ≥ -2}
  in a_set.card = 3 :=
by
  sorry

end number_of_values_a_l697_697846


namespace max_reciprocal_sum_l697_697780

-- Definitions and conditions from the problem statement
def distinct (l : List ℕ) : Prop := l.Nodup

def nonempty_subsets_sums_neq (S : Finset ℕ) : Prop :=
  ∀ A B : Finset ℕ, A ≠ B → A ≠ ∅ → B ≠ ∅ → (A.sum id ≠ B.sum id)

-- The main theorem to prove
theorem max_reciprocal_sum (n : ℕ) (h : n ≥ 5)
  (a : Fin n → ℕ) (h_distinct : distinct (Finset.univ.image a).to_list)
  (h_sums : nonempty_subsets_sums_neq (Finset.univ.image a)) :
  ∑ i in Finset.univ.image a, (1 : ℚ) / i ≤ 2 - (1 / 2^(n-1 : ℚ)) :=
sorry

end max_reciprocal_sum_l697_697780


namespace rotated_sheets_area_l697_697569

theorem rotated_sheets_area :
  let side := 8
  let middle_rotation := 45
  let top_rotation := 90
  let required_area := 96
  ∃ (area : ℝ), 
    (area = required_area) :=
begin
  sorry
end

end rotated_sheets_area_l697_697569


namespace complex_magnitude_l697_697762

theorem complex_magnitude (z : ℂ) (h : (2+1*I) * z = 5) : complex.abs (z + 1 * I) = 2 := 
by
  sorry

end complex_magnitude_l697_697762


namespace smallest_non_palindrome_power_of_13_l697_697719

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l697_697719


namespace smallest_non_palindrome_power_of_13_is_2197_l697_697738

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_non_palindrome_power_of_13_is_2197 : ∃ n : ℕ, 13^n = 2197 ∧ ¬ is_palindrome (13^n) := 
by
  use 3
  have h1 : 13^3 = 2197 := by norm_num
  exact ⟨h1, by norm_num; sorry⟩

end smallest_non_palindrome_power_of_13_is_2197_l697_697738


namespace solve_problem_l697_697692

-- Define the conditions of the problem:
def is_solution_pair (a b : ℕ) : Prop :=
  (a > 0 ∧ b > 0) ∧ (b ∣ (a^2 + 1)) ∧ (a ∣ (b^2 + 1))

-- Define the Fibonacci sequence using a function:
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Predicate to check if a pair is of the form (F_{2n-1}, F_{2n+1})
def is_fibonacci_pair (a b : ℕ) : Prop :=
  ∃ n : ℕ, a = fibonacci (2 * n - 1) ∧ b = fibonacci (2 * n + 1)

-- Main theorem statement:
theorem solve_problem : 
  ∀ (a b : ℕ), is_solution_pair a b ↔ (a = 1 ∧ b = 1) ∨ is_fibonacci_pair a b :=
by
  sorry

end solve_problem_l697_697692


namespace petya_max_rubles_l697_697944

theorem petya_max_rubles :
  let is_valid_number := (λ n : ℕ, 2000 ≤ n ∧ n < 2100),
      is_divisible := (λ n d : ℕ, d ∣ n),
      rubles := (λ n, if is_divisible n 1 then 1 else 0 +
                       if is_divisible n 3 then 3 else 0 +
                       if is_divisible n 5 then 5 else 0 +
                       if is_divisible n 7 then 7 else 0 +
                       if is_divisible n 9 then 9 else 0 +
                       if is_divisible n 11 then 11 else 0)
  in ∃ n, is_valid_number n ∧ rubles n = 31 := 
sorry

end petya_max_rubles_l697_697944


namespace number_of_divisors_of_n_squared_l697_697625

-- Define the conditions
def has_four_divisors (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p * q

def num_divisors (k : ℕ) : ℕ :=
  let f := k.factors in
  f.foldl (λ acc x, acc * (x.2 + 1)) 1

-- Create the proof problem
theorem number_of_divisors_of_n_squared (n : ℕ) (h : has_four_divisors n) :
  num_divisors (n * n) = 9 :=
sorry

end number_of_divisors_of_n_squared_l697_697625


namespace smallest_non_palindrome_power_of_13_l697_697725

def is_smallest_non_palindrome_power_of_13 (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 13^k) ∧ ¬ nat.is_palindrome n ∧ ∀ m, (∃ k : ℕ, m = 13^k) ∧ ¬ nat.is_palindrome m → n ≤ m

theorem smallest_non_palindrome_power_of_13 : is_smallest_non_palindrome_power_of_13 169 :=
by {
  sorry
}

end smallest_non_palindrome_power_of_13_l697_697725
