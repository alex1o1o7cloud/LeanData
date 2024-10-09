import Mathlib

namespace option_a_correct_l1389_138971

-- Define the variables as real numbers
variables {a b : ℝ}

-- Define the main theorem to prove
theorem option_a_correct : (a - b) * (2 * a + 2 * b) = 2 * a^2 - 2 * b^2 := by
  -- start the proof block
  sorry

end option_a_correct_l1389_138971


namespace expand_polynomial_l1389_138982

theorem expand_polynomial (x : ℝ) :
    (5*x^2 + 3*x - 7) * (4*x^3) = 20*x^5 + 12*x^4 - 28*x^3 :=
by
  sorry

end expand_polynomial_l1389_138982


namespace solve_inequality_l1389_138967

theorem solve_inequality (a : ℝ) : 
  (a = 0 → {x : ℝ | x ≥ -1} = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
  (a ≠ 0 → 
    ((a > 0 → { x : ℝ | -1 ≤ x ∧ x ≤ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (-2 < a ∧ a < 0 → { x : ℝ | x ≤ 2 / a } ∪ { x : ℝ | -1 ≤ x }  = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a < -2 → { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a = -2 → { x : ℝ | True } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 })
)) :=
sorry

end solve_inequality_l1389_138967


namespace car_speed_l1389_138975

theorem car_speed (v t Δt : ℝ) (h1: 90 = v * t) (h2: 90 = (v + 30) * (t - Δt)) (h3: Δt = 0.5) : 
  ∃ v, 90 = v * t ∧ 90 = (v + 30) * (t - Δt) :=
by {
  sorry
}

end car_speed_l1389_138975


namespace dentist_filling_cost_l1389_138943

variable (F : ℝ)
variable (total_bill : ℝ := 5 * F)
variable (cleaning_cost : ℝ := 70)
variable (extraction_cost : ℝ := 290)
variable (two_fillings_cost : ℝ := 2 * F)

theorem dentist_filling_cost :
  total_bill = cleaning_cost + two_fillings_cost + extraction_cost → 
  F = 120 :=
by
  intros h
  sorry

end dentist_filling_cost_l1389_138943


namespace fg_of_3_l1389_138984

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem we want to prove
theorem fg_of_3 : f (g 3) = 344 := by
  sorry

end fg_of_3_l1389_138984


namespace translate_point_left_l1389_138924

def initial_point : ℝ × ℝ := (-2, -1)
def translation_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1 - units, p.2)

theorem translate_point_left :
  translation_left initial_point 2 = (-4, -1) :=
by
  -- By definition and calculation
  -- Let p = initial_point
  -- x' = p.1 - 2,
  -- y' = p.2
  -- translation_left (-2, -1) 2 = (-4, -1)
  sorry

end translate_point_left_l1389_138924


namespace two_non_coincident_planes_divide_space_l1389_138957

-- Define conditions for non-coincident planes
def non_coincident_planes (P₁ P₂ : Plane) : Prop :=
  ¬(P₁ = P₂)

-- Define the main theorem based on the conditions and the question
theorem two_non_coincident_planes_divide_space (P₁ P₂ : Plane) 
  (h : non_coincident_planes P₁ P₂) :
  ∃ n : ℕ, n = 3 ∨ n = 4 :=
by
  sorry

end two_non_coincident_planes_divide_space_l1389_138957


namespace geometric_sequence_property_l1389_138989

variable {a_n : ℕ → ℝ}

theorem geometric_sequence_property (h1 : ∀ m n p q : ℕ, m + n = p + q → a_n m * a_n n = a_n p * a_n q)
    (h2 : a_n 4 * a_n 5 * a_n 6 = 27) : a_n 1 * a_n 9 = 9 := by
  sorry

end geometric_sequence_property_l1389_138989


namespace dropped_test_score_l1389_138979

theorem dropped_test_score (A B C D : ℕ) 
  (h1 : A + B + C + D = 280) 
  (h2 : A + B + C = 225) : 
  D = 55 := 
by sorry

end dropped_test_score_l1389_138979


namespace most_likely_event_is_C_l1389_138949

open Classical

noncomputable def total_events : ℕ := 6 * 6

noncomputable def P_A : ℚ := 7 / 36
noncomputable def P_B : ℚ := 18 / 36
noncomputable def P_C : ℚ := 1
noncomputable def P_D : ℚ := 0

theorem most_likely_event_is_C :
  P_C > P_A ∧ P_C > P_B ∧ P_C > P_D := by
  sorry

end most_likely_event_is_C_l1389_138949


namespace jordan_rectangle_length_l1389_138944

def rectangle_area (length width : ℝ) : ℝ := length * width

theorem jordan_rectangle_length :
  let carol_length := 8
  let carol_width := 15
  let jordan_width := 30
  let carol_area := rectangle_area carol_length carol_width
  ∃ jordan_length, rectangle_area jordan_length jordan_width = carol_area →
  jordan_length = 4 :=
by
  sorry

end jordan_rectangle_length_l1389_138944


namespace avg_production_last_5_days_l1389_138983

theorem avg_production_last_5_days
  (avg_first_25_days : ℕ)
  (total_days : ℕ)
  (avg_entire_month : ℕ)
  (h1 : avg_first_25_days = 60)
  (h2 : total_days = 30)
  (h3 : avg_entire_month = 58) : 
  (total_days * avg_entire_month - 25 * avg_first_25_days) / 5 = 48 := 
by
  sorry

end avg_production_last_5_days_l1389_138983


namespace sum_seven_l1389_138952

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ)

axiom a2 : a 2 = 3
axiom a6 : a 6 = 11
axiom arithmetic_seq : arithmetic_sequence a
axiom sum_of_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_seven : S 7 = 49 :=
sorry

end sum_seven_l1389_138952


namespace part1_part2_l1389_138914

-- Definitions of propositions P and q
def P (t : ℝ) : Prop := (4 - t > t - 1 ∧ t - 1 > 0)
def q (a t : ℝ) : Prop := t^2 - (a+3)*t + (a+2) < 0

-- Part 1: If P is true, find the range of t.
theorem part1 (t : ℝ) (hP : P t) : 1 < t ∧ t < 5/2 :=
by sorry

-- Part 2: If P is a sufficient but not necessary condition for q, find the range of a.
theorem part2 (a : ℝ) 
  (hP_q : ∀ t, P t → q a t) 
  (hsubset : ∀ t, 1 < t ∧ t < 5/2 → q a t) 
  : a > 1/2 :=
by sorry

end part1_part2_l1389_138914


namespace area_of_gray_region_l1389_138900

theorem area_of_gray_region (r R : ℝ) (hr : r = 2) (hR : R = 3 * r) : 
  π * R ^ 2 - π * r ^ 2 = 32 * π :=
by
  have hr : r = 2 := hr
  have hR : R = 3 * r := hR
  sorry

end area_of_gray_region_l1389_138900


namespace range_of_a_l1389_138937

noncomputable def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : a ≤ -1 / 2 ∨ a ≥ 2 :=
  sorry

end range_of_a_l1389_138937


namespace chimps_moved_l1389_138997

theorem chimps_moved (total_chimps : ℕ) (chimps_staying : ℕ) (chimps_moved : ℕ) 
  (h_total : total_chimps = 45)
  (h_staying : chimps_staying = 27) :
  chimps_moved = 18 :=
by
  sorry

end chimps_moved_l1389_138997


namespace find_a_l1389_138946

variable {f : ℝ → ℝ}

-- Conditions
variables (a : ℝ) (domain : Set ℝ := Set.Ioo (3 - 2 * a) (a + 1))
variable (even_f : ∀ x, f (x + 1) = f (- (x + 1)))

-- The theorem stating the problem
theorem find_a (h : ∀ x, x ∈ domain ↔ x ∈ Set.Ioo (3 - 2 * a) (a + 1)) : a = 2 := by
  sorry

end find_a_l1389_138946


namespace probability_of_odd_number_l1389_138920

theorem probability_of_odd_number (total_outcomes : ℕ) (odd_outcomes : ℕ) (h1 : total_outcomes = 6) (h2 : odd_outcomes = 3) : (odd_outcomes / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry 

end probability_of_odd_number_l1389_138920


namespace minimal_rooms_l1389_138966

-- Definitions
def numTourists := 100

def roomsAvailable (n k : Nat) : Prop :=
  ∀ k_even : k % 2 = 0, 
    ∃ m : Nat, k = 2 * m ∧ n = 100 * (m + 1) ∨
    ∀ k_odd : k % 2 = 1, k = 2 * m + 1 ∧ n = 100 * (m + 1) + 1

-- Proof statement
theorem minimal_rooms (k n : Nat) : roomsAvailable n k :=
by 
  -- The proof is provided in the solution steps
  sorry

end minimal_rooms_l1389_138966


namespace min_value_4a_plus_b_l1389_138972

theorem min_value_4a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1/a + 1/b = 1) : 4*a + b = 9 :=
sorry

end min_value_4a_plus_b_l1389_138972


namespace find_k_n_l1389_138951

theorem find_k_n (k n : ℕ) (h_kn_pos : 0 < k ∧ 0 < n) (h_eq : k^2 - 2016 = 3^n) : k = 45 ∧ n = 2 := 
by {
  sorry
}

end find_k_n_l1389_138951


namespace find_acute_angles_right_triangle_l1389_138953

theorem find_acute_angles_right_triangle (α β : ℝ)
  (h₁ : α + β = π / 2)
  (h₂ : 0 < α ∧ α < π / 2)
  (h₃ : 0 < β ∧ β < π / 2)
  (h4 : Real.tan α + Real.tan β + Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan α ^ 3 + Real.tan β ^ 3 = 70) :
  (α = 75 * (π / 180) ∧ β = 15 * (π / 180)) 
  ∨ (α = 15 * (π / 180) ∧ β = 75 * (π / 180)) := 
sorry

end find_acute_angles_right_triangle_l1389_138953


namespace standard_equation_of_circle_l1389_138930

/-- A circle with radius 2, center in the fourth quadrant, and tangent to the lines x = 0 and x + y = 2√2 has the standard equation (x - 2)^2 + (y + 2)^2 = 4. -/
theorem standard_equation_of_circle :
  ∃ a, a > 0 ∧ (∀ x y : ℝ, ((x - a)^2 + (y + 2)^2 = 4) ∧ 
                        (a > 0) ∧ 
                        (x = 0 → a = 2) ∧
                        x + y = 2 * Real.sqrt 2 → a = 2) := 
by
  sorry

end standard_equation_of_circle_l1389_138930


namespace value_of_N_l1389_138980

theorem value_of_N (N : ℕ): 6 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 7.5 ↔ N = 25 ∨ N = 26 ∨ N = 27 ∨ N = 28 ∨ N = 29 := 
by
  sorry

end value_of_N_l1389_138980


namespace melanie_gave_3_plums_to_sam_l1389_138950

theorem melanie_gave_3_plums_to_sam 
  (initial_plums : ℕ) 
  (plums_left : ℕ) 
  (plums_given : ℕ) 
  (h1 : initial_plums = 7) 
  (h2 : plums_left = 4) 
  (h3 : plums_left + plums_given = initial_plums) : 
  plums_given = 3 :=
by 
  sorry

end melanie_gave_3_plums_to_sam_l1389_138950


namespace ivan_total_money_l1389_138991

-- Define the value of a dime in cents
def value_of_dime : ℕ := 10

-- Define the value of a penny in cents
def value_of_penny : ℕ := 1

-- Define the number of dimes per piggy bank
def dimes_per_piggy_bank : ℕ := 50

-- Define the number of pennies per piggy bank
def pennies_per_piggy_bank : ℕ := 100

-- Define the number of piggy banks
def number_of_piggy_banks : ℕ := 2

-- Define the total value in dollars
noncomputable def total_value_in_dollars : ℕ := 
  (dimes_per_piggy_bank * value_of_dime + pennies_per_piggy_bank * value_of_penny) * number_of_piggy_banks / 100

theorem ivan_total_money : total_value_in_dollars = 12 := by
  sorry

end ivan_total_money_l1389_138991


namespace replace_floor_cost_l1389_138916

-- Define the conditions
def floor_removal_cost : ℝ := 50
def new_floor_cost_per_sqft : ℝ := 1.25
def room_length : ℝ := 8
def room_width : ℝ := 7

-- Define the area of the room
def room_area : ℝ := room_length * room_width

-- Define the cost of the new floor
def new_floor_cost : ℝ := room_area * new_floor_cost_per_sqft

-- Define the total cost to replace the floor
def total_cost : ℝ := floor_removal_cost + new_floor_cost

-- State the proof problem
theorem replace_floor_cost : total_cost = 120 := by
  sorry

end replace_floor_cost_l1389_138916


namespace second_term_of_geometric_series_l1389_138912

theorem second_term_of_geometric_series (a r S term2 : ℝ) 
  (h1 : r = 1 / 4)
  (h2 : S = 40)
  (h3 : S = a / (1 - r))
  (h4 : term2 = a * r) : 
  term2 = 7.5 := 
  by
  sorry

end second_term_of_geometric_series_l1389_138912


namespace sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l1389_138956

theorem sum_of_consecutive_natural_numbers_eq_three_digit_same_digits :
  ∃ n : ℕ, (1 + n) * n / 2 = 111 * 6 ∧ n = 36 :=
by
  sorry

end sum_of_consecutive_natural_numbers_eq_three_digit_same_digits_l1389_138956


namespace third_divisor_l1389_138987

/-- 
Given that the new number after subtracting 7 from 3,381 leaves a remainder of 8 when divided by 9 
and 11, prove that the third divisor that also leaves a remainder of 8 is 17.
-/
theorem third_divisor (x : ℕ) (h1 : x = 3381 - 7)
                      (h2 : x % 9 = 8)
                      (h3 : x % 11 = 8) :
  ∃ (d : ℕ), d = 17 ∧ x % d = 8 := sorry

end third_divisor_l1389_138987


namespace find_y_l1389_138938

variables (ABC ACB BAC : ℝ)
variables (CDE ADE EAD AED DEB y : ℝ)

-- Conditions
axiom angle_ABC : ABC = 45
axiom angle_ACB : ACB = 90
axiom angle_BAC_eq : BAC = 180 - ABC - ACB
axiom angle_CDE : CDE = 72
axiom angle_ADE_eq : ADE = 180 - CDE
axiom angle_EAD : EAD = 45
axiom angle_AED_eq : AED = 180 - ADE - EAD
axiom angle_DEB_eq : DEB = 180 - AED
axiom y_eq : y = DEB

-- Goal
theorem find_y : y = 153 :=
by {
  -- Here we would proceed with the proof using the established axioms.
  sorry
}

end find_y_l1389_138938


namespace remainder_of_n_plus_3255_l1389_138969

theorem remainder_of_n_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := 
by
  sorry

end remainder_of_n_plus_3255_l1389_138969


namespace total_students_in_faculty_l1389_138988

theorem total_students_in_faculty (N A B : ℕ) (hN : N = 230) (hA : A = 423) (hB : B = 134)
  (h80_percent : (N + A - B) = 80 / 100 * T) : T = 649 := 
by
  sorry

end total_students_in_faculty_l1389_138988


namespace sin_lower_bound_lt_l1389_138970

theorem sin_lower_bound_lt (a : ℝ) (h : ∃ x : ℝ, Real.sin x < a) : a > -1 :=
sorry

end sin_lower_bound_lt_l1389_138970


namespace value_of_expression_l1389_138904

theorem value_of_expression :
  (3150 - 3030)^2 / 144 = 100 :=
by {
  -- This imported module allows us to use basic mathematical functions and properties
  sorry -- We use sorry to skip the actual proof
}

end value_of_expression_l1389_138904


namespace length_of_rect_box_l1389_138931

noncomputable def length_of_box (height : ℝ) (width : ℝ) (volume : ℝ) : ℝ :=
  volume / (width * height)

theorem length_of_rect_box :
  (length_of_box 0.5 25 (6000 / 7.48052)) = 64.1624 :=
by
  unfold length_of_box
  norm_num
  sorry

end length_of_rect_box_l1389_138931


namespace sequence_unique_l1389_138933

theorem sequence_unique (n : ℕ) (h1 : n > 1)
  (x : ℕ → ℕ)
  (hx1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j < n → x i < x j)
  (hx2 : ∀ i, 1 ≤ i ∧ i < n → x i + x (n - i) = 2 * n)
  (hx3 : ∀ i j, 1 ≤ i ∧ i < n ∧ 1 ≤ j ∧ j < n ∧ x i + x j < 2 * n →
    ∃ k, 1 ≤ k ∧ k < n ∧ x i + x j = x k) :
  ∀ k, 1 ≤ k ∧ k < n → x k = 2 * k :=
by
  sorry

end sequence_unique_l1389_138933


namespace simplify_expression_l1389_138903

variable (x : ℝ)

theorem simplify_expression :
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 + 5 * x ^ 10 + 3 * x ^ 9)) =
  (15 * x ^ 13 - x ^ 12 + 9 * x ^ 11 - x ^ 10 - 6 * x ^ 9) :=
by
  sorry

end simplify_expression_l1389_138903


namespace solve_laundry_problem_l1389_138994

def laundry_problem : Prop :=
  let total_weight := 20
  let clothes_weight := 5
  let detergent_per_scoop := 0.02
  let initial_detergent := 2 * detergent_per_scoop
  let optimal_ratio := 0.004
  let additional_detergent := 0.02
  let additional_water := 14.94
  let total_detergent := initial_detergent + additional_detergent
  let final_amount := clothes_weight + initial_detergent + additional_detergent + additional_water
  final_amount = total_weight ∧ total_detergent / (total_weight - clothes_weight) = optimal_ratio

theorem solve_laundry_problem : laundry_problem :=
by 
  -- the proof would go here
  sorry

end solve_laundry_problem_l1389_138994


namespace Sarah_copy_total_pages_l1389_138942

theorem Sarah_copy_total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ)
  (h1 : num_people = 9) (h2 : copies_per_person = 2) (h3 : pages_per_contract = 20) :
  num_people * copies_per_person * pages_per_contract = 360 :=
by
  sorry

end Sarah_copy_total_pages_l1389_138942


namespace sufficient_but_not_necessary_l1389_138926

theorem sufficient_but_not_necessary (a : ℝ) (h : a = 1/4) : 
  (∀ x : ℝ, x > 0 → x + a / x ≥ 1) ∧ ¬(∀ x : ℝ, x > 0 → x + a / x ≥ 1 ↔ a = 1/4) :=
by
  sorry

end sufficient_but_not_necessary_l1389_138926


namespace storks_count_l1389_138945

theorem storks_count (B S : ℕ) (h1 : B = 3) (h2 : B + 2 = S + 1) : S = 4 :=
by
  sorry

end storks_count_l1389_138945


namespace find_q_l1389_138925

noncomputable def q (x : ℝ) : ℝ := -2 * x^4 + 10 * x^3 - 2 * x^2 + 7 * x + 3

theorem find_q :
  ∀ x : ℝ,
  q x + (2 * x^4 - 5 * x^2 + 8 * x + 3) = (10 * x^3 - 7 * x^2 + 15 * x + 6) :=
by
  intro x
  unfold q
  sorry

end find_q_l1389_138925


namespace isosceles_triangle_height_ratio_l1389_138995

theorem isosceles_triangle_height_ratio (b1 h1 b2 h2 : ℝ) 
  (A1 : ℝ := 1/2 * b1 * h1) (A2 : ℝ := 1/2 * b2 * h2)
  (area_ratio : A1 / A2 = 16 / 49)
  (similar : b1 / b2 = h1 / h2) : 
  h1 / h2 = 4 / 7 := 
by {
  sorry
}

end isosceles_triangle_height_ratio_l1389_138995


namespace chess_team_girls_count_l1389_138922

theorem chess_team_girls_count (B G : ℕ) 
  (h1 : B + G = 26) 
  (h2 : (3 / 4 : ℝ) * B + (1 / 4 : ℝ) * G = 13) : G = 13 := 
sorry

end chess_team_girls_count_l1389_138922


namespace nba_conferences_division_l1389_138947

theorem nba_conferences_division (teams : ℕ) (games_per_team : ℕ) (E : ℕ) :
  teams = 30 ∧ games_per_team = 82 ∧
  (teams = E + (teams - E)) ∧
  (games_per_team / 2 * E) + (games_per_team / 2 * (teams - E))  ≠ teams * games_per_team / 2 :=
by
  sorry

end nba_conferences_division_l1389_138947


namespace not_p_sufficient_not_necessary_for_not_q_l1389_138905

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) (h1 : q → p) (h2 : ¬ (p → q)) : 
  (¬p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l1389_138905


namespace z_coordinate_of_point_on_line_l1389_138910

theorem z_coordinate_of_point_on_line (t : ℝ)
  (h₁ : (1 + 3 * t, 3 + 2 * t, 2 + 4 * t) = (x, 7, z))
  (h₂ : x = 1 + 3 * t) :
  z = 10 :=
sorry

end z_coordinate_of_point_on_line_l1389_138910


namespace election_winning_votes_l1389_138909

noncomputable def total_votes (x y : ℕ) (p : ℚ) : ℚ := 
  (x + y) / (1 - p)

noncomputable def winning_votes (x y : ℕ) (p : ℚ) : ℚ :=
  p * total_votes x y p

theorem election_winning_votes :
  winning_votes 2136 7636 0.54336448598130836 = 11628 := 
by
  sorry

end election_winning_votes_l1389_138909


namespace other_root_l1389_138973

theorem other_root (m : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + m * x - 5 = 0 → (x = 1 ∨ x = -5 / 3)) :=
by {
  sorry
}

end other_root_l1389_138973


namespace solve_for_x_l1389_138907

theorem solve_for_x (x : ℝ) (h₁ : (7 * x) / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) (h₂ : x ≠ -4) : x = 6 / 7 :=
by
  sorry

end solve_for_x_l1389_138907


namespace height_difference_l1389_138936

def pine_tree_height : ℚ := 12 + 1 / 4
def maple_tree_height : ℚ := 18 + 1 / 2

theorem height_difference :
  maple_tree_height - pine_tree_height = 6 + 1 / 4 :=
by sorry

end height_difference_l1389_138936


namespace area_of_triangle_l1389_138968

theorem area_of_triangle (x : ℝ) :
  let t1_area := 16
  let t2_area := 25
  let t3_area := 64
  let total_area_factor := t1_area + t2_area + t3_area
  let side_factor := 17 * 17
  ΔABC_area = side_factor * total_area_factor :=
by {
  -- Placeholder to complete the proof
  sorry
}

end area_of_triangle_l1389_138968


namespace total_cost_of_purchases_l1389_138919

def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

theorem total_cost_of_purchases : cost_cat_toy + cost_cage = 21.95 := by
  -- skipping the proof
  sorry

end total_cost_of_purchases_l1389_138919


namespace middle_box_label_l1389_138927

/--
Given a sequence of 23 boxes in a row on the table, where each box has a label indicating either
  "There is no prize here" or "The prize is in a neighboring box",
and it is known that exactly one of these statements is true.
Prove that the label on the middle box (the 12th box) says "The prize is in the adjacent box."
-/
theorem middle_box_label :
  ∃ (boxes : Fin 23 → Prop) (labels : Fin 23 → String),
    (∀ i, labels i = "There is no prize here" ∨ labels i = "The prize is in a neighboring box") ∧
    (∃! i : Fin 23, boxes i ∧ (labels i = "The prize is in a neighboring box")) →
    labels ⟨11, sorry⟩ = "The prize is in a neighboring box" :=
sorry

end middle_box_label_l1389_138927


namespace volume_of_cuboid_l1389_138963

variable (a b c : ℝ)

def is_cuboid_adjacent_faces (a b c : ℝ) := a * b = 3 ∧ a * c = 5 ∧ b * c = 15

theorem volume_of_cuboid (a b c : ℝ) (h : is_cuboid_adjacent_faces a b c) :
  a * b * c = 15 := by
  sorry

end volume_of_cuboid_l1389_138963


namespace equation_for_pears_l1389_138962

-- Define the conditions
def pearDist1 (x : ℕ) : ℕ := 4 * x + 12
def pearDist2 (x : ℕ) : ℕ := 6 * x

-- State the theorem to be proved
theorem equation_for_pears (x : ℕ) : pearDist1 x = pearDist2 x :=
by
  sorry

end equation_for_pears_l1389_138962


namespace factor_polynomial_l1389_138992

theorem factor_polynomial (z : ℝ) : (70 * z ^ 20 + 154 * z ^ 40 + 224 * z ^ 60) = 14 * z ^ 20 * (5 + 11 * z ^ 20 + 16 * z ^ 40) := 
sorry

end factor_polynomial_l1389_138992


namespace find_dividend_l1389_138954

def dividend_problem (dividend divisor : ℕ) : Prop :=
  (15 * divisor + 5 = dividend) ∧ (dividend + divisor + 15 + 5 = 2169)

theorem find_dividend : ∃ dividend, ∃ divisor, dividend_problem dividend divisor ∧ dividend = 2015 :=
sorry

end find_dividend_l1389_138954


namespace find_f_of_7_over_3_l1389_138978

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the odd function f

-- Hypothesis: f is an odd function
axiom odd_function (x : ℝ) : f (-x) = -f x

-- Hypothesis: f(1 + x) = f(-x) for all x in ℝ
axiom functional_equation (x : ℝ) : f (1 + x) = f (-x)

-- Hypothesis: f(-1/3) = 1/3
axiom initial_condition : f (-1 / 3) = 1 / 3

-- The statement we need to prove
theorem find_f_of_7_over_3 : f (7 / 3) = - (1 / 3) :=
by
  sorry -- Proof to be provided

end find_f_of_7_over_3_l1389_138978


namespace photograph_perimeter_l1389_138999

theorem photograph_perimeter (w l m : ℕ) 
  (h1 : (w + 4) * (l + 4) = m)
  (h2 : (w + 8) * (l + 8) = m + 94) :
  2 * (w + l) = 23 := 
by
  sorry

end photograph_perimeter_l1389_138999


namespace problem_1_problem_2_problem_3_l1389_138964

noncomputable def area_triangle (a b C : ℝ) : ℝ := (1/2) * a * b * Real.sin C
noncomputable def area_quadrilateral (e f φ : ℝ) : ℝ := (1/2) * e * f * Real.sin φ

theorem problem_1 (a b C : ℝ) (hC : Real.sin C ≤ 1) : 
  area_triangle a b C ≤ (a^2 + b^2) / 4 :=
sorry

theorem problem_2 (e f φ : ℝ) (hφ : Real.sin φ ≤ 1) : 
  area_quadrilateral e f φ ≤ (e^2 + f^2) / 4 :=
sorry

theorem problem_3 (a b C c d D : ℝ) 
  (hC : Real.sin C ≤ 1) 
  (hD : Real.sin D ≤ 1) :
  area_triangle a b C + area_triangle c d D ≤ (a^2 + b^2 + c^2 + d^2) / 4 :=
sorry

end problem_1_problem_2_problem_3_l1389_138964


namespace house_to_car_ratio_l1389_138974

-- Define conditions
def cost_per_night := 4000
def nights_at_hotel := 2
def cost_of_car := 30000
def total_value_of_treats := 158000

-- Prove that the ratio of the value of the house to the value of the car is 4:1
theorem house_to_car_ratio : 
  (total_value_of_treats - (nights_at_hotel * cost_per_night + cost_of_car)) / cost_of_car = 4 := by
  sorry

end house_to_car_ratio_l1389_138974


namespace bill_difference_l1389_138996

theorem bill_difference (mandy_bills : ℕ) (manny_bills : ℕ) 
  (mandy_bill_value : ℕ) (manny_bill_value : ℕ) (target_bill_value : ℕ) 
  (h_mandy : mandy_bills = 3) (h_mandy_val : mandy_bill_value = 20) 
  (h_manny : manny_bills = 2) (h_manny_val : manny_bill_value = 50)
  (h_target : target_bill_value = 10) :
  (manny_bills * manny_bill_value / target_bill_value) - (mandy_bills * mandy_bill_value / target_bill_value) = 4 :=
by
  sorry

end bill_difference_l1389_138996


namespace frank_picked_apples_l1389_138955

theorem frank_picked_apples (F : ℕ) 
  (susan_picked : ℕ := 3 * F) 
  (susan_left : ℕ := susan_picked / 2) 
  (frank_left : ℕ := 2 * F / 3) 
  (total_left : susan_left + frank_left = 78) : 
  F = 36 :=
sorry

end frank_picked_apples_l1389_138955


namespace solve_ordered_pairs_l1389_138941

theorem solve_ordered_pairs (a b : ℕ) (h : a^2 + b^2 = ab * (a + b)) : 
  (a, b) = (1, 1) ∨ (a, b) = (1, 1) :=
by 
  sorry

end solve_ordered_pairs_l1389_138941


namespace percentage_A_is_22_l1389_138981

noncomputable def percentage_A_in_mixture : ℝ :=
  (0.8 * 0.20 + 0.2 * 0.30) * 100

theorem percentage_A_is_22 :
  percentage_A_in_mixture = 22 := 
by
  sorry

end percentage_A_is_22_l1389_138981


namespace milk_needed_6_cookies_3_3_pints_l1389_138976

def gallon_to_quarts (g : ℚ) : ℚ := g * 4
def quarts_to_pints (q : ℚ) : ℚ := q * 2
def cookies_to_pints (p : ℚ) (c : ℚ) (n : ℚ) : ℚ := (p / c) * n
def measurement_error (p : ℚ) : ℚ := p * 1.1

theorem milk_needed_6_cookies_3_3_pints :
  (measurement_error (cookies_to_pints (quarts_to_pints (gallon_to_quarts 1.5)) 24 6) = 3.3) :=
by
  sorry

end milk_needed_6_cookies_3_3_pints_l1389_138976


namespace center_of_rotation_l1389_138911

noncomputable def f (z : ℂ) : ℂ := ((-1 - (Complex.I * Real.sqrt 3)) * z + (2 * Real.sqrt 3 - 12 * Complex.I)) / 2

theorem center_of_rotation :
  ∃ c : ℂ, f c = c ∧ c = -5 * Real.sqrt 3 / 2 - 7 / 2 * Complex.I :=
by
  sorry

end center_of_rotation_l1389_138911


namespace find_deleted_files_l1389_138985

def original_files : Nat := 21
def remaining_files : Nat := 7
def deleted_files : Nat := 14

theorem find_deleted_files : original_files - remaining_files = deleted_files := by
  sorry

end find_deleted_files_l1389_138985


namespace Katie_average_monthly_balance_l1389_138940

def balances : List ℕ := [120, 240, 180, 180, 240]

def average (l : List ℕ) : ℕ := l.sum / l.length

theorem Katie_average_monthly_balance : average balances = 192 :=
by
  sorry

end Katie_average_monthly_balance_l1389_138940


namespace pet_fee_is_120_l1389_138908

noncomputable def daily_rate : ℝ := 125.00
noncomputable def rental_days : ℕ := 14
noncomputable def service_fee_rate : ℝ := 0.20
noncomputable def security_deposit : ℝ := 1110.00
noncomputable def security_deposit_rate : ℝ := 0.50

theorem pet_fee_is_120 :
  let total_stay_cost := daily_rate * rental_days
  let service_fee := service_fee_rate * total_stay_cost
  let total_before_pet_fee := total_stay_cost + service_fee
  let entire_bill := security_deposit / security_deposit_rate
  let pet_fee := entire_bill - total_before_pet_fee
  pet_fee = 120 := by
  sorry

end pet_fee_is_120_l1389_138908


namespace finite_cuboid_blocks_l1389_138961

/--
Prove that there are only finitely many cuboid blocks with integer dimensions a, b, c
such that abc = 2(a - 2)(b - 2)(c - 2) and c ≤ b ≤ a.
-/
theorem finite_cuboid_blocks :
  ∃ (S : Finset (ℤ × ℤ × ℤ)), ∀ (a b c : ℤ), (abc = 2 * (a - 2) * (b - 2) * (c - 2)) → (c ≤ b) → (b ≤ a) → (a, b, c) ∈ S := 
by
  sorry

end finite_cuboid_blocks_l1389_138961


namespace cubic_sum_l1389_138958

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 :=
  sorry

end cubic_sum_l1389_138958


namespace perp_lines_implies_values_l1389_138998

variable (a : ℝ)

def line1_perpendicular (a : ℝ) : Prop :=
  (1 - a) * (2 * a + 3) + a * (a - 1) = 0

theorem perp_lines_implies_values (h : line1_perpendicular a) :
  a = 1 ∨ a = -3 :=
by {
  sorry
}

end perp_lines_implies_values_l1389_138998


namespace find_a_l1389_138934

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f : ℝ → ℝ := sorry -- The definition of f is to be handled in the proof

theorem find_a (a : ℝ) (h1 : is_odd_function f)
  (h2 : ∀ x : ℝ, 0 < x → f x = 2^(x - a) - 2 / (x + 1))
  (h3 : f (-1) = 3 / 4) : a = 3 :=
sorry

end find_a_l1389_138934


namespace intersection_A_B_l1389_138986

open Set

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 1}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := 
by {
  sorry
}

end intersection_A_B_l1389_138986


namespace sixteenth_answer_is_three_l1389_138990

theorem sixteenth_answer_is_three (total_members : ℕ)
  (answers_1 answers_2 answers_3 : ℕ) 
  (h_total : total_members = 16) 
  (h_answers_1 : answers_1 = 6) 
  (h_answers_2 : answers_2 = 6) 
  (h_answers_3 : answers_3 = 3) :
  ∃ answer : ℕ, answer = 3 ∧ (answers_1 + answers_2 + answers_3 + 1 = total_members) :=
sorry

end sixteenth_answer_is_three_l1389_138990


namespace amount_paid_by_customer_l1389_138923

theorem amount_paid_by_customer 
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (final_price : ℝ)
  (h1 : cost_price = 6681.818181818181)
  (h2 : markup_percentage = 10 / 100)
  (h3 : final_price = cost_price * (1 + markup_percentage)) :
  final_price = 7350 :=
by 
  sorry

end amount_paid_by_customer_l1389_138923


namespace find_d_l1389_138929

theorem find_d (x y d : ℕ) (h_midpoint : (1 + 5)/2 = 3 ∧ (3 + 11)/2 = 7) 
  : x + y = d ↔ d = 10 := 
sorry

end find_d_l1389_138929


namespace find_numbers_l1389_138921

theorem find_numbers (x y : ℕ) (h1 : x / y = 3) (h2 : (x^2 + y^2) / (x + y) = 5) : 
  x = 6 ∧ y = 2 := 
by
  sorry

end find_numbers_l1389_138921


namespace amount_with_r_l1389_138913

theorem amount_with_r (p q r : ℕ) (h1 : p + q + r = 7000) (h2 : r = (2 * (p + q)) / 3) : r = 2800 :=
sorry

end amount_with_r_l1389_138913


namespace total_tickets_sales_l1389_138960

theorem total_tickets_sales:
    let student_ticket_price := 6
    let adult_ticket_price := 8
    let number_of_students := 20
    let number_of_adults := 12
    number_of_students * student_ticket_price + number_of_adults * adult_ticket_price = 216 :=
by
    intros
    sorry

end total_tickets_sales_l1389_138960


namespace farm_corn_cobs_l1389_138959

theorem farm_corn_cobs (rows_field1 rows_field2 cobs_per_row : Nat) (h1 : rows_field1 = 13) (h2 : rows_field2 = 16) (h3 : cobs_per_row = 4) : rows_field1 * cobs_per_row + rows_field2 * cobs_per_row = 116 := by
  sorry

end farm_corn_cobs_l1389_138959


namespace smallest_four_digit_multiple_of_53_l1389_138965

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l1389_138965


namespace basket_weight_l1389_138977

variable (B P : ℕ)

theorem basket_weight (h1 : B + P = 62) (h2 : B + P / 2 = 34) : B = 6 :=
by
  sorry

end basket_weight_l1389_138977


namespace elder_age_is_30_l1389_138935

/-- The ages of two persons differ by 16 years, and 6 years ago, the elder one was 3 times as old as the younger one. 
Prove that the present age of the elder person is 30 years. --/
theorem elder_age_is_30 (y e: ℕ) (h₁: e = y + 16) (h₂: e - 6 = 3 * (y - 6)) : e = 30 := 
sorry

end elder_age_is_30_l1389_138935


namespace polar_equations_and_ratios_l1389_138901

open Real

theorem polar_equations_and_ratios (α β : ℝ)
    (h_line : ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ)
    (h_curve : ∀ (α : ℝ), ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2) :
    ( ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ) ∧
    ( ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2 → 
    0 < r * sin 2 * θ / (r / cos θ) ∧ r * sin 2 * θ / (r / cos θ) ≤ 1 / 2) :=
by
  sorry

end polar_equations_and_ratios_l1389_138901


namespace son_present_age_l1389_138915

theorem son_present_age (S F : ℕ) (h1 : F = S + 34) (h2 : F + 2 = 2 * (S + 2)) : S = 32 :=
by
  sorry

end son_present_age_l1389_138915


namespace circle_radius_correct_l1389_138917

noncomputable def radius_of_circle 
  (side_length : ℝ)
  (angle_tangents : ℝ)
  (sin_18 : ℝ) : ℝ := 
  sorry

theorem circle_radius_correct 
  (side_length : ℝ := 6 + 2 * Real.sqrt 5)
  (angle_tangents : ℝ := 36)
  (sin_18 : ℝ := (Real.sqrt 5 - 1) / 4) :
  radius_of_circle side_length angle_tangents sin_18 = 
  2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) :=
sorry

end circle_radius_correct_l1389_138917


namespace smallest_x_value_l1389_138948

theorem smallest_x_value : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y^2 - 5 * y - 84) / (y - 9) = 4 / (y + 6) → y >= (x)) ∧ 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) ∧ 
  x = ( - 13 - Real.sqrt 17 ) / 2 := 
sorry

end smallest_x_value_l1389_138948


namespace sum_a_b_c_l1389_138932

theorem sum_a_b_c (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 390) (h2: a * b + b * c + c * a = 5) : a + b + c = 20 ∨ a + b + c = -20 := 
by 
  sorry

end sum_a_b_c_l1389_138932


namespace area_of_smallest_square_l1389_138993

-- Define a circle with a given radius
def radius : ℝ := 7

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the side length of the smallest square that can contain the circle
def side_length : ℝ := diameter

-- Define the area of the square as the side length squared
def area_of_square : ℝ := side_length ^ 2

-- State the theorem: the area of the smallest square that contains a circle of radius 7 is 196
theorem area_of_smallest_square : area_of_square = 196 := by
    sorry

end area_of_smallest_square_l1389_138993


namespace tan_330_eq_neg_sqrt3_div_3_l1389_138928

theorem tan_330_eq_neg_sqrt3_div_3 :
  Real.tan (330 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_330_eq_neg_sqrt3_div_3_l1389_138928


namespace amy_hours_per_week_l1389_138939

theorem amy_hours_per_week {h w summer_salary school_weeks school_salary} 
  (hours_per_week_summer : h = 45)
  (weeks_summer : w = 8)
  (summer_salary_h : summer_salary = 3600)
  (school_weeks_h : school_weeks = 24)
  (school_salary_h : school_salary = 3600) :
  ∃ hours_per_week_school, hours_per_week_school = 15 :=
by
  sorry

end amy_hours_per_week_l1389_138939


namespace present_age_of_son_l1389_138918

theorem present_age_of_son
  (S M : ℕ)
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  sorry
}

end present_age_of_son_l1389_138918


namespace base3_to_base10_conversion_l1389_138902

theorem base3_to_base10_conversion : 
  (1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3^1 + 1 * 3^0 = 100) :=
by 
  sorry

end base3_to_base10_conversion_l1389_138902


namespace find_average_speed_l1389_138906

theorem find_average_speed :
  ∃ v : ℝ, (880 / v) - (880 / (v + 10)) = 2 ∧ v = 61.5 :=
by
  sorry

end find_average_speed_l1389_138906
