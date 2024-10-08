import Mathlib

namespace greatest_area_difference_l98_98748

theorem greatest_area_difference 
    (a b c d : ℕ) 
    (H1 : 2 * (a + b) = 100)
    (H2 : 2 * (c + d) = 100)
    (H3 : ∀i j : ℕ, 2 * (i + j) = 100 → i * j ≤ a * b)
    : 373 ≤ a * b - (c * d) := 
sorry

end greatest_area_difference_l98_98748


namespace clean_time_per_room_l98_98061

variable (h : ℕ)

-- Conditions
def floors := 4
def rooms_per_floor := 10
def total_rooms := floors * rooms_per_floor
def hourly_wage := 15
def total_earnings := 3600

-- Question and condition mapping to conclusion
theorem clean_time_per_room (H1 : total_rooms = 40) 
                            (H2 : total_earnings = 240 * hourly_wage) 
                            (H3 : 240 = 40 * h) :
                            h = 6 :=
by {
  sorry
}

end clean_time_per_room_l98_98061


namespace corrected_multiplication_result_l98_98283

theorem corrected_multiplication_result :
  ∃ n : ℕ, 987 * n = 559989 ∧ 987 * n ≠ 559981 ∧ 559981 % 100 = 98 :=
by
  sorry

end corrected_multiplication_result_l98_98283


namespace geometric_sequence_tenth_term_l98_98599

theorem geometric_sequence_tenth_term :
  let a : ℚ := 4
  let r : ℚ := 5/3
  let n : ℕ := 10
  a * r^(n-1) = 7812500 / 19683 :=
by sorry

end geometric_sequence_tenth_term_l98_98599


namespace range_of_m_l98_98656

theorem range_of_m (f : ℝ → ℝ) {m : ℝ} (h_dec : ∀ x y, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x ≥ f y)
  (h_ineq : f (m - 1) > f (2 * m - 1)) : 0 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l98_98656


namespace not_monotonic_in_interval_l98_98346

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a * x - 5

theorem not_monotonic_in_interval (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → f a x ≠ (1/3) * x^3 - x^2 + a * x - 5) → a ≥ 1 ∨ a ≤ -3 :=
sorry

end not_monotonic_in_interval_l98_98346


namespace tan_sum_trig_identity_l98_98779

variable {A B C : ℝ} -- Angles
variable {a b c : ℝ} -- Sides opposite to angles A, B and C

-- Acute triangle implies A, B, C are all less than π/2 and greater than 0
variable (hAcute : 0 < A ∧ A < pi / 2 ∧ 0 < B ∧ B < pi / 2 ∧ 0 < C ∧ C < pi / 2)

-- Given condition in the problem
variable (hCondition : b / a + a / b = 6 * Real.cos C)

theorem tan_sum_trig_identity : 
  Real.tan C / Real.tan A + Real.tan C / Real.tan B = 4 :=
sorry

end tan_sum_trig_identity_l98_98779


namespace find_x_in_average_l98_98830

theorem find_x_in_average (x : ℝ) :
  (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + x) / 9 = 207 → x = 217 :=
by
  intro h
  sorry

end find_x_in_average_l98_98830


namespace sqrt_23_range_l98_98722

theorem sqrt_23_range : 4.5 < Real.sqrt 23 ∧ Real.sqrt 23 < 5 := by
  sorry

end sqrt_23_range_l98_98722


namespace count_numbers_with_digit_sum_10_l98_98117

theorem count_numbers_with_digit_sum_10 : 
  ∃ n : ℕ, 
  (n = 66) ∧ ∀ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  a + b + c = 10 → 
  true :=
by
  sorry

end count_numbers_with_digit_sum_10_l98_98117


namespace total_plates_l98_98431

-- define the variables for the number of plates
def plates_lobster_rolls : Nat := 25
def plates_spicy_hot_noodles : Nat := 14
def plates_seafood_noodles : Nat := 16

-- state the problem as a theorem
theorem total_plates :
  plates_lobster_rolls + plates_spicy_hot_noodles + plates_seafood_noodles = 55 := by
  sorry

end total_plates_l98_98431


namespace doug_fires_l98_98560

theorem doug_fires (D : ℝ) (Kai_fires : ℝ) (Eli_fires : ℝ) 
    (hKai : Kai_fires = 3 * D)
    (hEli : Eli_fires = 1.5 * D)
    (hTotal : D + Kai_fires + Eli_fires = 110) : 
  D = 20 := 
by
  sorry

end doug_fires_l98_98560


namespace ordered_triples_2022_l98_98429

theorem ordered_triples_2022 :
  ∃ n : ℕ, n = 13 ∧ (∃ a c : ℕ, a ≤ c ∧ (a * c = 2022^2)) := by
  sorry

end ordered_triples_2022_l98_98429


namespace area_smaller_part_l98_98909

theorem area_smaller_part (A B : ℝ) (h₁ : A + B = 500) (h₂ : B - A = (A + B) / 10) : A = 225 :=
by sorry

end area_smaller_part_l98_98909


namespace find_height_of_cylinder_l98_98839

theorem find_height_of_cylinder (r SA : ℝ) (h : ℝ) (h_r : r = 3) (h_SA : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h → h = 2 :=
by
  sorry

end find_height_of_cylinder_l98_98839


namespace article_production_l98_98565

-- Conditions
variables (x z : ℕ) (hx : 0 < x) (hz : 0 < z)
-- The given condition: x men working x hours a day for x days produce 2x^2 articles.
def articles_produced_x (x : ℕ) : ℕ := 2 * x^2

-- The question: the number of articles produced by z men working z hours a day for z days
def articles_produced_z (x z : ℕ) : ℕ := 2 * z^3 / x

-- Prove that the number of articles produced by z men working z hours a day for z days is 2 * (z^3) / x
theorem article_production (hx : 0 < x) (hz : 0 < z) :
  articles_produced_z x z = 2 * z^3 / x :=
sorry

end article_production_l98_98565


namespace lesser_number_l98_98545

theorem lesser_number (x y : ℕ) (h1 : x + y = 58) (h2 : x - y = 6) : y = 26 :=
by
  sorry

end lesser_number_l98_98545


namespace train_length_is_300_l98_98261

-- Definitions based on the conditions
def trainCrossesPlatform (L V : ℝ) : Prop :=
  L + 400 = V * 42

def trainCrossesSignalPole (L V : ℝ) : Prop :=
  L = V * 18

-- The main theorem statement
theorem train_length_is_300 (L V : ℝ)
  (h1 : trainCrossesPlatform L V)
  (h2 : trainCrossesSignalPole L V) :
  L = 300 :=
by
  sorry

end train_length_is_300_l98_98261


namespace diamond_eight_five_l98_98329

def diamond (a b : ℕ) : ℕ := (a + b) * ((a - b) * (a - b))

theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end diamond_eight_five_l98_98329


namespace min_omega_l98_98126

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega (ω φ T : ℝ) (hω : ω > 0) (hφ1 : 0 < φ) (hφ2 : φ < Real.pi / 2)
  (hT : f ω φ T = Real.sqrt 3 / 2)
  (hx : f ω φ (Real.pi / 6) = 0) :
  ω = 4 := by
  sorry

end min_omega_l98_98126


namespace correct_cases_needed_l98_98258

noncomputable def cases_needed (boxes_sold : ℕ) (boxes_per_case : ℕ) : ℕ :=
  (boxes_sold + boxes_per_case - 1) / boxes_per_case

theorem correct_cases_needed :
  cases_needed 10 6 = 2 ∧ -- For trefoils
  cases_needed 15 5 = 3 ∧ -- For samoas
  cases_needed 20 10 = 2  -- For thin mints
:= by
  sorry

end correct_cases_needed_l98_98258


namespace find_third_circle_radius_l98_98031

-- Define the context of circles and their tangency properties
variable (A B : ℝ → ℝ → Prop) -- Centers of circles
variable (r1 r2 : ℝ) -- Radii of circles

-- Define conditions from the problem
def circles_are_tangent (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) : Prop :=
  ∀ x y : ℝ, A x y → B (x + 7) y ∧ r1 = 2 ∧ r2 = 5

def third_circle_tangent_to_others_and_tangent_line (A B : ℝ → ℝ → Prop) (r3 : ℝ) : Prop :=
  ∃ D : ℝ → ℝ → Prop, ∀ x y : ℝ, D x y →
  ((A (x + r3) y ∧ B (x - r3) y) ∧ (r3 > 0))

theorem find_third_circle_radius (A B : ℝ → ℝ → Prop) (r1 r2 : ℝ) :
  circles_are_tangent A B r1 r2 →
  (∃ r3 : ℝ, r3 = 1 ∧ third_circle_tangent_to_others_and_tangent_line A B r3) :=
by
  sorry

end find_third_circle_radius_l98_98031


namespace max_pages_within_budget_l98_98957

-- Definitions based on the problem conditions
def page_cost_in_cents : ℕ := 5
def total_budget_in_cents : ℕ := 5000
def max_expenditure_in_cents : ℕ := 4500

-- Proof problem statement
theorem max_pages_within_budget : 
  ∃ (pages : ℕ), pages = max_expenditure_in_cents / page_cost_in_cents ∧ 
                  pages * page_cost_in_cents ≤ total_budget_in_cents :=
by {
  sorry
}

end max_pages_within_budget_l98_98957


namespace count_squares_ending_in_4_l98_98932

theorem count_squares_ending_in_4 (n : ℕ) : 
  (∀ k : ℕ, (n^2 < 5000) → (n^2 % 10 = 4) → (k ≤ 70)) → 
  (∃ m : ℕ, m = 14) :=
by 
  sorry

end count_squares_ending_in_4_l98_98932


namespace third_divisor_l98_98726

theorem third_divisor (x : ℕ) (h12 : 12 ∣ (x + 3)) (h15 : 15 ∣ (x + 3)) (h40 : 40 ∣ (x + 3)) :
  ∃ d : ℕ, d ≠ 12 ∧ d ≠ 15 ∧ d ≠ 40 ∧ d ∣ (x + 3) ∧ d = 2 :=
by
  sorry

end third_divisor_l98_98726


namespace intersection_at_7_m_l98_98133

def f (x : Int) (d : Int) : Int := 4 * x + d

theorem intersection_at_7_m (d m : Int) (h₁ : f 7 d = m) (h₂ : 7 = f m d) : m = 7 := by
  sorry

end intersection_at_7_m_l98_98133


namespace find_teacher_age_l98_98222

noncomputable def age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) 
                                (avg_age_inclusive : ℕ) (num_people_inclusive : ℕ) : ℕ :=
  let total_age_students := num_students * avg_age_students
  let total_age_inclusive := num_people_inclusive * avg_age_inclusive
  total_age_inclusive - total_age_students

theorem find_teacher_age : age_of_teacher 15 10 16 11 = 26 := 
by 
  sorry

end find_teacher_age_l98_98222


namespace contractor_daily_wage_l98_98005

theorem contractor_daily_wage 
  (total_days : ℕ)
  (daily_wage : ℝ)
  (fine_per_absence : ℝ)
  (total_earned : ℝ)
  (absent_days : ℕ)
  (H1 : total_days = 30)
  (H2 : fine_per_absence = 7.5)
  (H3 : total_earned = 555.0)
  (H4 : absent_days = 6)
  (H5 : total_earned = daily_wage * (total_days - absent_days) - fine_per_absence * absent_days) :
  daily_wage = 25 := by
  sorry

end contractor_daily_wage_l98_98005


namespace area_ratio_trapezoid_abm_abcd_l98_98658

-- Definitions based on conditions
variables {A B C D M : Type} [Zero A] [Zero B] [Zero C] [Zero D] [Zero M]
variables (BC AD : ℝ)

-- Condition: ABCD is a trapezoid with BC parallel to AD and diagonals AC and BD intersect M
-- Given BC = b and AD = a

-- Theorem statement
theorem area_ratio_trapezoid_abm_abcd (a b : ℝ) (h1 : BC = b) (h2 : AD = a) : 
  ∃ S_ABM S_ABCD : ℝ,
  (S_ABM / S_ABCD = a * b / (a + b)^2) :=
sorry

end area_ratio_trapezoid_abm_abcd_l98_98658


namespace infinite_product_value_l98_98724

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, 9^(1/(3^n))

theorem infinite_product_value : infinite_product = 27 := 
  by sorry

end infinite_product_value_l98_98724


namespace consecutive_integers_divisible_by_12_l98_98991

theorem consecutive_integers_divisible_by_12 (a b c d : ℤ) 
  (h1 : b = a + 1) (h2 : c = b + 1) (h3 : d = c + 1) : 
  12 ∣ (a * b + a * c + a * d + b * c + b * d + c * d + 1) := 
sorry

end consecutive_integers_divisible_by_12_l98_98991


namespace first_quarter_spending_l98_98316

variables (spent_february_start spent_march_end spent_april_end : ℝ)

-- Given conditions
def begin_february_spent : Prop := spent_february_start = 0.5
def end_march_spent : Prop := spent_march_end = 1.5
def end_april_spent : Prop := spent_april_end = 2.0

-- Proof statement
theorem first_quarter_spending (h1 : begin_february_spent spent_february_start) 
                               (h2 : end_march_spent spent_march_end) 
                               (h3 : end_april_spent spent_april_end) : 
                                spent_march_end - spent_february_start = 1.5 :=
by sorry

end first_quarter_spending_l98_98316


namespace not_divisible_by_121_l98_98705

theorem not_divisible_by_121 (n : ℤ) : ¬ ∃ t : ℤ, (n^2 + 3*n + 5) = 121 * t ∧ (n^2 - 3*n + 5) = 121 * t := sorry

end not_divisible_by_121_l98_98705


namespace problem_l98_98648

theorem problem (a : ℕ) (b : ℚ) (c : ℤ) 
  (h1 : a = 1) 
  (h2 : b = 0) 
  (h3 : abs (c) = 6) :
  (a - b + c = (7 : ℤ)) ∨ (a - b + c = (-5 : ℤ)) := by
  sorry

end problem_l98_98648


namespace no_integer_sided_triangle_with_odd_perimeter_1995_l98_98908

theorem no_integer_sided_triangle_with_odd_perimeter_1995 :
  ¬ ∃ (a b c : ℕ), (a + b + c = 1995) ∧ (∃ (h1 h2 h3 : ℕ), true) :=
by
  sorry

end no_integer_sided_triangle_with_odd_perimeter_1995_l98_98908


namespace stick_segments_l98_98337

theorem stick_segments (L : ℕ) (L_nonzero : L > 0) :
  let red_segments := 8
  let blue_segments := 12
  let black_segments := 18
  let total_segments := (red_segments + blue_segments + black_segments) 
                       - (lcm red_segments blue_segments / blue_segments) 
                       - (lcm blue_segments black_segments / black_segments)
                       - (lcm red_segments black_segments / black_segments)
                       + (lcm red_segments (lcm blue_segments black_segments) / (lcm blue_segments black_segments))
  let shortest_segment_length := L / lcm red_segments (lcm blue_segments black_segments)
  (total_segments = 28) ∧ (shortest_segment_length = L / 72) := by
  sorry

end stick_segments_l98_98337


namespace fish_lifespan_proof_l98_98376

def hamster_lifespan : ℝ := 2.5

def dog_lifespan : ℝ := 4 * hamster_lifespan

def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_proof :
  fish_lifespan = 12 := 
  by
  sorry

end fish_lifespan_proof_l98_98376


namespace total_birds_count_l98_98832

def cage1_parrots := 9
def cage1_finches := 4
def cage1_canaries := 7

def cage2_parrots := 5
def cage2_parakeets := 8
def cage2_finches := 10

def cage3_parakeets := 15
def cage3_finches := 7
def cage3_canaries := 3

def cage4_parrots := 10
def cage4_parakeets := 5
def cage4_finches := 12

def total_birds := cage1_parrots + cage1_finches + cage1_canaries +
                   cage2_parrots + cage2_parakeets + cage2_finches +
                   cage3_parakeets + cage3_finches + cage3_canaries +
                   cage4_parrots + cage4_parakeets + cage4_finches

theorem total_birds_count : total_birds = 95 :=
by
  -- Proof is omitted here.
  sorry

end total_birds_count_l98_98832


namespace extreme_points_sum_gt_l98_98447

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem extreme_points_sum_gt (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 8)
    {x₁ x₂ : ℝ} (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) (h₄ : x₁ < x₂)
    (h₅ : 0 < x₁) (h₆ : 0 < x₂) : f x₁ a + f x₂ a > 3 - 2 * Real.log 2 := sorry

end extreme_points_sum_gt_l98_98447


namespace min_odd_integers_l98_98874

-- Definitions of the conditions
variable (a b c d e f : ℤ)

-- The mathematical theorem statement
theorem min_odd_integers 
  (h1 : a + b = 30)
  (h2 : a + b + c + d = 50) 
  (h3 : a + b + c + d + e + f = 70)
  (h4 : e + f % 2 = 1) : 
  ∃ n, n ≥ 1 ∧ n = (if a % 2 = 1 then 1 else 0) + (if b % 2 = 1 then 1 else 0) + 
                    (if c % 2 = 1 then 1 else 0) + (if d % 2 = 1 then 1 else 0) + 
                    (if e % 2 = 1 then 1 else 0) + (if f % 2 = 1 then 1 else 0) :=
sorry

end min_odd_integers_l98_98874


namespace line1_line2_line3_l98_98476

-- Line 1: Through (-1, 3), parallel to x - 2y + 3 = 0.
theorem line1 (x y : ℝ) : (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 3) →
                              (x - 2 * y + 7 = 0) :=
by sorry

-- Line 2: Through (3, 4), perpendicular to 3x - y + 2 = 0.
theorem line2 (x y : ℝ) : (3 * x - y + 2 = 0) ∧ (x = 3) ∧ (y = 4) →
                              (x + 3 * y - 15 = 0) :=
by sorry

-- Line 3: Through (1, 2), with equal intercepts on both axes.
theorem line3 (x y : ℝ) : (x = y) ∧ (x = 1) ∧ (y = 2) →
                              (x + y - 3 = 0) :=
by sorry

end line1_line2_line3_l98_98476


namespace paths_from_A_to_B_via_C_l98_98245

open Classical

-- Definitions based on conditions
variables (lattice : Type) [PartialOrder lattice]
variables (A B C : lattice)
variables (first_red first_blue second_red second_blue first_green second_green orange : lattice)

-- Conditions encoded as hypotheses
def direction_changes : Prop :=
  -- Arrow from first green to orange is now one way from orange to green
  ∀ x : lattice, x = first_green → orange < x ∧ ¬ (x < orange) ∧
  -- Additional stop at point C located directly after the first blue arrows
  (C < first_blue ∨ first_blue < C)

-- Now stating the proof problem
theorem paths_from_A_to_B_via_C :
  direction_changes lattice first_green orange first_blue C →
  -- Total number of paths from A to B via C is 12
  (2 + 2) * 3 * 1 = 12 :=
by
  sorry

end paths_from_A_to_B_via_C_l98_98245


namespace incenter_circumcenter_identity_l98_98140

noncomputable def triangle : Type := sorry
noncomputable def incenter (t : triangle) : Type := sorry
noncomputable def circumcenter (t : triangle) : Type := sorry
noncomputable def inradius (t : triangle) : ℝ := sorry
noncomputable def circumradius (t : triangle) : ℝ := sorry
noncomputable def distance (A B : Type) : ℝ := sorry

theorem incenter_circumcenter_identity (t : triangle) (I O : Type)
  (hI : I = incenter t) (hO : O = circumcenter t)
  (r : ℝ) (h_r : r = inradius t)
  (R : ℝ) (h_R : R = circumradius t) :
  distance I O ^ 2 = R ^ 2 - 2 * R * r :=
sorry

end incenter_circumcenter_identity_l98_98140


namespace count_8_digit_odd_last_l98_98510

-- Define the constraints for the digits of the 8-digit number
def first_digit_choices := 9
def next_six_digits_choices := 10 ^ 6
def last_digit_choices := 5

-- State the theorem based on the given conditions and the solution
theorem count_8_digit_odd_last : first_digit_choices * next_six_digits_choices * last_digit_choices = 45000000 :=
by
  sorry

end count_8_digit_odd_last_l98_98510


namespace contrapositive_proposition_l98_98741

theorem contrapositive_proposition (a b : ℝ) :
  (¬ ((a - b) * (a + b) = 0) → ¬ (a - b = 0)) :=
sorry

end contrapositive_proposition_l98_98741


namespace rank_siblings_l98_98438

variable (Person : Type) (Dan Elena Finn : Person)

variable (height : Person → ℝ)

-- Conditions
axiom different_heights : height Dan ≠ height Elena ∧ height Elena ≠ height Finn ∧ height Finn ≠ height Dan
axiom one_true_statement : (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn)) 
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))

theorem rank_siblings : height Finn > height Elena ∧ height Elena > height Dan := by
  sorry

end rank_siblings_l98_98438


namespace actual_cost_of_article_l98_98059

-- Define the basic conditions of the problem
variable (x : ℝ)
variable (h : x - 0.24 * x = 1064)

-- The theorem we need to prove
theorem actual_cost_of_article : x = 1400 :=
by
  -- since we are not proving anything here, we skip the proof
  sorry

end actual_cost_of_article_l98_98059


namespace athlete_distance_proof_l98_98032

-- Definition of conditions as constants
def time_seconds : ℕ := 20
def speed_kmh : ℕ := 36

-- Convert speed from km/h to m/s
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Proof statement that the distance is 200 meters
theorem athlete_distance_proof : speed_mps * time_seconds = 200 :=
by sorry

end athlete_distance_proof_l98_98032


namespace sculpture_cost_in_INR_l98_98422

def USD_per_NAD := 1 / 5
def INR_per_USD := 8
def cost_in_NAD := 200
noncomputable def cost_in_INR := (cost_in_NAD * USD_per_NAD) * INR_per_USD

theorem sculpture_cost_in_INR :
  cost_in_INR = 320 := by
  sorry

end sculpture_cost_in_INR_l98_98422


namespace impossible_even_sum_l98_98141

theorem impossible_even_sum (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 :=
sorry

end impossible_even_sum_l98_98141


namespace solution_l98_98890

def system (a b : ℝ) : Prop :=
  (2 * a + b = 3) ∧ (a - b = 1)

theorem solution (a b : ℝ) (h: system a b) : a + 2 * b = 2 :=
by
  cases h with
  | intro h1 h2 => sorry

end solution_l98_98890


namespace fraction_of_shaded_hexagons_l98_98150

-- Definitions
def total_hexagons : ℕ := 9
def shaded_hexagons : ℕ := 5

-- Theorem statement
theorem fraction_of_shaded_hexagons : 
  (shaded_hexagons: ℚ) / (total_hexagons : ℚ) = 5 / 9 := by
sorry

end fraction_of_shaded_hexagons_l98_98150


namespace wendy_candy_in_each_box_l98_98879

variable (x : ℕ)

def brother_candy : ℕ := 6
def total_candy : ℕ := 12
def wendy_boxes : ℕ := 2 * x

theorem wendy_candy_in_each_box :
  2 * x + brother_candy = total_candy → x = 3 :=
by
  intro h
  sorry

end wendy_candy_in_each_box_l98_98879


namespace ratio_a_c_l98_98793

variable (a b c d : ℕ)

/-- The given conditions -/
axiom ratio_a_b : a / b = 5 / 2
axiom ratio_c_d : c / d = 4 / 1
axiom ratio_d_b : d / b = 1 / 3

/-- The proof problem -/
theorem ratio_a_c : a / c = 15 / 8 := by
  sorry

end ratio_a_c_l98_98793


namespace work_completes_in_39_days_l98_98486

theorem work_completes_in_39_days 
  (amit_days : ℕ := 15)  -- Amit can complete work in 15 days
  (ananthu_days : ℕ := 45)  -- Ananthu can complete work in 45 days
  (amit_worked_days : ℕ := 3)  -- Amit worked for 3 days
  : (amit_worked_days + ((4 / 5) / (1 / ananthu_days))) = 39 :=
by
  sorry

end work_completes_in_39_days_l98_98486


namespace remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l98_98523

theorem remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one :
  ((x - 1) ^ 2028) % (x^2 - x + 1) = 1 :=
by
  sorry

end remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l98_98523


namespace sum_of_numbers_l98_98655

theorem sum_of_numbers (a b c : ℝ) :
  a^2 + b^2 + c^2 = 138 → ab + bc + ca = 131 → a + b + c = 20 :=
by
  sorry

end sum_of_numbers_l98_98655


namespace mechanism_parts_l98_98567

-- Definitions
def total_parts (S L : Nat) : Prop := S + L = 25
def condition1 (S L : Nat) : Prop := ∀ (A : Finset (Fin 25)), (A.card = 12) → ∃ i, i ∈ A ∧ i < S
def condition2 (S L : Nat) : Prop := ∀ (B : Finset (Fin 25)), (B.card = 15) → ∃ i, i ∈ B ∧ i >= S

-- Main statement
theorem mechanism_parts :
  ∃ (S L : Nat), 
  total_parts S L ∧ 
  condition1 S L ∧ 
  condition2 S L ∧ 
  S = 14 ∧ 
  L = 11 :=
sorry

end mechanism_parts_l98_98567


namespace m_value_if_Q_subset_P_l98_98313

noncomputable def P : Set ℝ := {x | x^2 = 1}
def Q (m : ℝ) : Set ℝ := {x | m * x = 1}
def m_values (m : ℝ) : Prop := Q m ⊆ P → m = 0 ∨ m = 1 ∨ m = -1

theorem m_value_if_Q_subset_P (m : ℝ) : m_values m :=
sorry

end m_value_if_Q_subset_P_l98_98313


namespace Rachel_drinks_correct_glasses_l98_98250

def glasses_Sunday : ℕ := 2
def glasses_Monday : ℕ := 4
def glasses_TuesdayToFriday : ℕ := 3
def days_TuesdayToFriday : ℕ := 4
def ounces_per_glass : ℕ := 10
def total_goal : ℕ := 220
def glasses_Saturday : ℕ := 4

theorem Rachel_drinks_correct_glasses :
  ounces_per_glass * (glasses_Sunday + glasses_Monday + days_TuesdayToFriday * glasses_TuesdayToFriday + glasses_Saturday) = total_goal :=
sorry

end Rachel_drinks_correct_glasses_l98_98250


namespace midpoints_distance_l98_98937

theorem midpoints_distance
  (A B C D M N : ℝ)
  (h1 : M = (A + C) / 2)
  (h2 : N = (B + D) / 2)
  (h3 : D - A = 68)
  (h4 : C - B = 26)
  : abs (M - N) = 21 := 
sorry

end midpoints_distance_l98_98937


namespace distance_to_x_axis_P_l98_98214

-- The coordinates of point P
def P : ℝ × ℝ := (3, -2)

-- The distance from point P to the x-axis
def distance_to_x_axis (point : ℝ × ℝ) : ℝ :=
  abs (point.snd)

theorem distance_to_x_axis_P : distance_to_x_axis P = 2 :=
by
  -- Use the provided point P and calculate the distance
  sorry

end distance_to_x_axis_P_l98_98214


namespace pizza_slices_left_per_person_l98_98462

def total_slices (small: Nat) (large: Nat) : Nat := small + large

def total_eaten (phil: Nat) (andre: Nat) : Nat := phil + andre

def slices_left (total: Nat) (eaten: Nat) : Nat := total - eaten

def pieces_per_person (left: Nat) (people: Nat) : Nat := left / people

theorem pizza_slices_left_per_person :
  ∀ (small large phil andre people: Nat),
  small = 8 → large = 14 → phil = 9 → andre = 9 → people = 2 →
  pieces_per_person (slices_left (total_slices small large) (total_eaten phil andre)) people = 2 :=
by
  intros small large phil andre people h_small h_large h_phil h_andre h_people
  rw [h_small, h_large, h_phil, h_andre, h_people]
  /-
  Here we conclude the proof.
  -/
  sorry

end pizza_slices_left_per_person_l98_98462


namespace days_c_worked_l98_98988

noncomputable def work_done_by_a_b := 1 / 10
noncomputable def work_done_by_b_c := 1 / 18
noncomputable def work_done_by_c_alone := 1 / 45

theorem days_c_worked
  (A B C : ℚ)
  (h1 : A + B = work_done_by_a_b)
  (h2 : B + C = work_done_by_b_c)
  (h3 : C = work_done_by_c_alone) :
  15 = (1/3) / work_done_by_c_alone :=
sorry

end days_c_worked_l98_98988


namespace smallest_positive_debt_l98_98430

theorem smallest_positive_debt :
  ∃ (D : ℕ) (p g : ℤ), 0 < D ∧ D = 350 * p + 240 * g ∧ D = 10 := sorry

end smallest_positive_debt_l98_98430


namespace triangle_perimeter_l98_98771

theorem triangle_perimeter (a b c : ℕ) (ha : a = 7) (hb : b = 10) (hc : c = 15) :
  a + b + c = 32 :=
by
  -- Given the lengths of the sides
  have H1 : a = 7 := ha
  have H2 : b = 10 := hb
  have H3 : c = 15 := hc
  
  -- Therefore, we need to prove the sum
  sorry

end triangle_perimeter_l98_98771


namespace exists_y_square_divisible_by_five_btw_50_and_120_l98_98406

theorem exists_y_square_divisible_by_five_btw_50_and_120 : ∃ y : ℕ, (∃ k : ℕ, y = k^2) ∧ (y % 5 = 0) ∧ (50 ≤ y ∧ y ≤ 120) ∧ y = 100 :=
by
  sorry

end exists_y_square_divisible_by_five_btw_50_and_120_l98_98406


namespace hyperbola_eccentricity_l98_98339

theorem hyperbola_eccentricity (a : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y = (1 / 8) * x^2 → x^2 = 8 * y) →
  (∀ y x : ℝ, y^2 / a - x^2 = 1 → a + 1 = 4) →
  e^2 = 4 / 3 →
  e = 2 * Real.sqrt 3 / 3 :=
by
  intros h1 h2 h3
  sorry

end hyperbola_eccentricity_l98_98339


namespace Claudia_solution_l98_98820

noncomputable def Claudia_coins : Prop :=
  ∃ (x y : ℕ), x + y = 12 ∧ 23 - x = 17 ∧ y = 6

theorem Claudia_solution : Claudia_coins :=
by
  existsi 6
  existsi 6
  sorry

end Claudia_solution_l98_98820


namespace fraction_irreducibility_l98_98990

theorem fraction_irreducibility (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducibility_l98_98990


namespace identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l98_98144

-- Question 1: Prove the given identity for 1/(n(n+1))
theorem identity_1_over_n_n_plus_1 (n : ℕ) (hn : n ≠ 0) : 
  (1 : ℚ) / (n * (n + 1)) = (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by
  sorry

-- Question 2: Prove the sum of series 1/k(k+1) from k=1 to k=2021
theorem sum_series_1_over_k_k_plus_1 : 
  (Finset.range 2021).sum (λ k => (1 : ℚ) / (k+1) / (k+2)) = 2021 / 2022 :=
by
  sorry

-- Question 3: Prove the sum of series 1/(3k-2)(3k+1) from k=1 to k=673
theorem sum_series_1_over_3k_minus_2_3k_plus_1 : 
  (Finset.range 673).sum (λ k => (1 : ℚ) / ((3 * k + 1 - 2) * (3 * k + 1))) = 674 / 2023 :=
by
  sorry

end identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l98_98144


namespace area_ratio_triangle_l98_98963

noncomputable def area_ratio (x y : ℝ) (n m : ℕ) : ℝ :=
(x * y) / (2 * n) / ((x * y) / (2 * m))

theorem area_ratio_triangle (x y : ℝ) (n m : ℕ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  area_ratio x y n m = (m : ℝ) / (n : ℝ) := by
  sorry

end area_ratio_triangle_l98_98963


namespace boys_running_speed_l98_98334
-- Import the necessary libraries

-- Define the input conditions:
def side_length : ℝ := 50
def time_seconds : ℝ := 80
def conversion_factor_meters_to_kilometers : ℝ := 1000
def conversion_factor_seconds_to_hours : ℝ := 3600

-- Define the theorem:
theorem boys_running_speed :
  let perimeter := 4 * side_length
  let distance_kilometers := perimeter / conversion_factor_meters_to_kilometers
  let time_hours := time_seconds / conversion_factor_seconds_to_hours
  distance_kilometers / time_hours = 9 :=
by
  sorry

end boys_running_speed_l98_98334


namespace average_speed_of_train_b_l98_98070

-- Given conditions
def distance_between_trains_initially := 13
def speed_of_train_a := 37
def time_to_overtake := 5
def distance_a_in_5_hours := speed_of_train_a * time_to_overtake
def distance_b_to_overtake := distance_between_trains_initially + distance_a_in_5_hours + 17

-- Prove: The average speed of Train B
theorem average_speed_of_train_b : 
  ∃ v_B, v_B = distance_b_to_overtake / time_to_overtake ∧ v_B = 43 :=
by
  -- The proof should go here, but we use sorry to skip it.
  sorry

end average_speed_of_train_b_l98_98070


namespace customers_left_l98_98410

theorem customers_left (initial_customers : ℝ) (first_left : ℝ) (second_left : ℝ) : initial_customers = 36.0 ∧ first_left = 19.0 ∧ second_left = 14.0 → initial_customers - first_left - second_left = 3.0 :=
by
  intros h
  sorry

end customers_left_l98_98410


namespace range_of_m_l98_98149

theorem range_of_m (a b m : ℝ) (h₀ : a > 0) (h₁ : b > 1) (h₂ : a + b = 2) (h₃ : ∀ m, (4/a + 1/(b-1)) > m^2 + 8*m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l98_98149


namespace heavier_boxes_weight_l98_98190

theorem heavier_boxes_weight
  (x y : ℤ)
  (h1 : x ≥ 0)
  (h2 : x ≤ 30)
  (h3 : 10 * x + (30 - x) * y = 540)
  (h4 : 10 * x + (15 - x) * y = 240) :
  y = 20 :=
by
  sorry

end heavier_boxes_weight_l98_98190


namespace angle_relation_l98_98848

-- Definitions for the triangle properties and angles.
variables {α : Type*} [LinearOrderedField α]
variables {A B C D E F : α}

-- Definitions stating the properties of the triangles.
def is_isosceles_triangle (a b c : α) : Prop :=
  a = b ∨ b = c ∨ c = a

def triangle_ABC_is_isosceles (AB AC : α) (ABC : α) : Prop :=
  is_isosceles_triangle AB AC ABC

def triangle_DEF_is_isosceles (DE DF : α) (DEF : α) : Prop :=
  is_isosceles_triangle DE DF DEF

-- Condition that gives the specific angle measure in triangle DEF.
def angle_DEF_is_100 (DEF : α) : Prop :=
  DEF = 100

-- The main theorem to prove.
theorem angle_relation (AB AC DE DF DEF a b c : α) :
  triangle_ABC_is_isosceles AB AC (AB + AC) →
  triangle_DEF_is_isosceles DE DF DEF →
  angle_DEF_is_100 DEF →
  a = c :=
by
  -- Assuming the conditions define the angles and state the relationship.
  sorry

end angle_relation_l98_98848


namespace evaluate_expression_l98_98898

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2
  else -n / 2

theorem evaluate_expression : S 19 * S 31 + S 48 = 136 :=
by sorry

end evaluate_expression_l98_98898


namespace factorization_identity_l98_98764

theorem factorization_identity (a b : ℝ) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 * (a + b)^2 :=
by
  sorry

end factorization_identity_l98_98764


namespace tan_alpha_plus_pi_over_4_l98_98621

noncomputable def sin_cos_identity (α : ℝ) : Prop :=
  (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : sin_cos_identity α) :
  Real.tan (α + Real.pi / 4) = -3 :=
  by
  sorry

end tan_alpha_plus_pi_over_4_l98_98621


namespace tangent_lines_passing_through_point_l98_98047

theorem tangent_lines_passing_through_point :
  ∀ (x0 y0 : ℝ) (p : ℝ × ℝ), 
  (p = (1, 1)) ∧ (y0 = x0 ^ 3) → 
  (y0 - 1 = 3 * x0 ^ 2 * (1 - x0)) → 
  (x0 = 1 ∨ x0 = -1/2) → 
  ((y - (3 * 1 - 2)) * (y - (3/4 * x0 + 1/4))) = 0 :=
sorry

end tangent_lines_passing_through_point_l98_98047


namespace fraction_product_l98_98998

theorem fraction_product (a b c d e : ℝ) (h1 : a = 1/2) (h2 : b = 1/3) (h3 : c = 1/4) (h4 : d = 1/6) (h5 : e = 144) :
  a * b * c * d * e = 1 := 
by
  -- Given the conditions h1 to h5, we aim to prove the product is 1
  sorry

end fraction_product_l98_98998


namespace fare_ratio_l98_98264

theorem fare_ratio (F1 F2 : ℕ) (h1 : F1 = 96000) (h2 : F1 + F2 = 224000) : F1 / (Nat.gcd F1 F2) = 3 ∧ F2 / (Nat.gcd F1 F2) = 4 :=
by
  sorry

end fare_ratio_l98_98264


namespace product_of_consecutive_integers_l98_98160

theorem product_of_consecutive_integers
  (a b : ℕ) (n : ℕ)
  (h1 : a = 12)
  (h2 : b = 22)
  (mean_five_numbers : (a + b + n + (n + 1) + (n + 2)) / 5 = 17) :
  (n * (n + 1) * (n + 2)) = 4896 := by
  sorry

end product_of_consecutive_integers_l98_98160


namespace intersecting_lines_angle_difference_l98_98721

-- Define the conditions
def angle_y : ℝ := 40
def straight_angle_sum : ℝ := 180

-- Define the variables representing the angles
variable (x y : ℝ)

-- Define the proof problem
theorem intersecting_lines_angle_difference : 
  ∀ x y : ℝ, 
  y = angle_y → 
  (∃ (a b : ℝ), a + b = straight_angle_sum ∧ a = y ∧ b = x) → 
  x - y = 100 :=
by
  intros x y hy h
  sorry

end intersecting_lines_angle_difference_l98_98721


namespace most_followers_is_sarah_l98_98276

def initial_followers_susy : ℕ := 100
def initial_followers_sarah : ℕ := 50

def susy_week1_new : ℕ := 40
def susy_week2_new := susy_week1_new / 2
def susy_week3_new := susy_week2_new / 2
def susy_total_new := susy_week1_new + susy_week2_new + susy_week3_new
def susy_final_followers := initial_followers_susy + susy_total_new

def sarah_week1_new : ℕ := 90
def sarah_week2_new := sarah_week1_new / 3
def sarah_week3_new := sarah_week2_new / 3
def sarah_total_new := sarah_week1_new + sarah_week2_new + sarah_week3_new
def sarah_final_followers := initial_followers_sarah + sarah_total_new

theorem most_followers_is_sarah : 
    sarah_final_followers ≥ susy_final_followers := by
  sorry

end most_followers_is_sarah_l98_98276


namespace compute_expression_l98_98399

theorem compute_expression : (-3) * 2 + 4 = -2 := 
by
  sorry

end compute_expression_l98_98399


namespace card_statements_are_false_l98_98343

theorem card_statements_are_false :
  ¬( ( (statements: ℕ) →
        (statements = 1 ↔ ¬statements = 1 ∧ ¬statements = 2 ∧ ¬statements = 3 ∧ ¬statements = 4 ∧ ¬statements = 5) ∧
        ( statements = 2 ↔ (statements = 1 ∨ statements = 3 ∨ statements = 4 ∨ statements = 5)) ∧
        (statements = 3 ↔ (statements = 1 ∧ statements = 2 ∧ (statements = 4 ∨ statements = 5) ) ) ∧
        (statements = 4 ↔ (statements = 1 ∧ statements = 2 ∧ statements = 3 ∧ statements != 5 ) ) ∧
        (statements = 5 ↔ (statements = 4 ) )
)) :=
sorry

end card_statements_are_false_l98_98343


namespace mouse_seed_hiding_l98_98417

theorem mouse_seed_hiding : 
  ∀ (h_m h_r x : ℕ), 
  4 * h_m = x →
  7 * h_r = x →
  h_m = h_r + 3 →
  x = 28 :=
by
  intros h_m h_r x H1 H2 H3
  sorry

end mouse_seed_hiding_l98_98417


namespace quadrilateral_area_lemma_l98_98369

-- Define the coordinates of the vertices
structure Point where
  x : ℤ
  y : ℤ

def A : Point := ⟨1, 3⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨2, 1⟩
def D : Point := ⟨2006, 2007⟩

-- Function to calculate the area of a quadrilateral given its vertices
def quadrilateral_area (A B C D : Point) : ℤ := 
  let triangle_area (P Q R : Point) : ℤ :=
    (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x) / 2
  triangle_area A B C + triangle_area A C D

-- The statement to be proved
theorem quadrilateral_area_lemma : quadrilateral_area A B C D = 3008 := 
  sorry

end quadrilateral_area_lemma_l98_98369


namespace AdvancedVowelSoup_l98_98433

noncomputable def AdvancedVowelSoup.sequence_count : ℕ :=
  let total_sequences := 7^7
  let vowel_only_sequences := 5^7
  let consonant_only_sequences := 2^7
  total_sequences - vowel_only_sequences - consonant_only_sequences

theorem AdvancedVowelSoup.valid_sequences : AdvancedVowelSoup.sequence_count = 745290 := by
  sorry

end AdvancedVowelSoup_l98_98433


namespace price_of_each_lemon_square_l98_98421

-- Given
def brownies_sold : Nat := 4
def price_per_brownie : Nat := 3
def lemon_squares_sold : Nat := 5
def goal_amount : Nat := 50
def cookies_sold : Nat := 7
def price_per_cookie : Nat := 4

-- Prove
theorem price_of_each_lemon_square :
  (brownies_sold * price_per_brownie + lemon_squares_sold * L + cookies_sold * price_per_cookie = goal_amount) →
  L = 2 :=
by
  sorry

end price_of_each_lemon_square_l98_98421


namespace cistern_filling_time_l98_98882

/-- Define the rates at which the cistern is filled and emptied -/
def fill_rate := (1 : ℚ) / 3
def empty_rate := (1 : ℚ) / 8

/-- Define the net rate of filling when both taps are open -/
def net_rate := fill_rate - empty_rate

/-- Define the volume of the cistern -/
def cistern_volume := (1 : ℚ)

/-- Compute the time to fill the cistern given the net rate -/
def fill_time := cistern_volume / net_rate

theorem cistern_filling_time :
  fill_time = 4.8 := by
sorry

end cistern_filling_time_l98_98882


namespace tan_frac_eq_l98_98200

theorem tan_frac_eq (x : ℝ) (h : Real.tan (x + π / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 := 
  sorry

end tan_frac_eq_l98_98200


namespace sum_of_100th_row_l98_98390

def triangularArraySum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2^(n+1) - 3*n

theorem sum_of_100th_row :
  triangularArraySum 100 = 2^100 - 297 :=
by
  sorry

end sum_of_100th_row_l98_98390


namespace find_initial_marbles_l98_98513

-- Definitions based on conditions
def loses_to_street (initial_marbles : ℕ) : ℕ := initial_marbles - (initial_marbles * 60 / 100)
def loses_to_sewer (marbles_after_street : ℕ) : ℕ := marbles_after_street / 2

-- The given number of marbles left
def remaining_marbles : ℕ := 20

-- Proof statement
theorem find_initial_marbles (initial_marbles : ℕ) : 
  loses_to_sewer (loses_to_street initial_marbles) = remaining_marbles -> 
  initial_marbles = 100 :=
by
  sorry

end find_initial_marbles_l98_98513


namespace unique_two_digit_solution_l98_98259

theorem unique_two_digit_solution :
  ∃! (u : ℕ), 9 < u ∧ u < 100 ∧ 13 * u % 100 = 52 := 
sorry

end unique_two_digit_solution_l98_98259


namespace no_integer_soln_x_y_l98_98738

theorem no_integer_soln_x_y (x y : ℤ) : x^2 + 5 ≠ y^3 := 
sorry

end no_integer_soln_x_y_l98_98738


namespace find_positive_integer_pairs_l98_98535

theorem find_positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a^2 = 3 * b^3) ↔ ∃ d : ℕ, 0 < d ∧ a = 18 * d^3 ∧ b = 6 * d^2 :=
by
  sorry

end find_positive_integer_pairs_l98_98535


namespace tan_alpha_result_l98_98207

theorem tan_alpha_result (α : ℝ) (h : Real.tan (α - Real.pi / 4) = 1 / 6) : Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_result_l98_98207


namespace evaluate_expression_l98_98916

theorem evaluate_expression :
  1 + (3 / (4 + (5 / (6 + (7 / 8))))) = 85 / 52 := 
by
  sorry

end evaluate_expression_l98_98916


namespace min_cookies_divisible_by_13_l98_98584

theorem min_cookies_divisible_by_13 (a b : ℕ) : ∃ n : ℕ, n > 0 ∧ n % 13 = 0 ∧ (∃ a b : ℕ, n = 10 * a + 21 * b) ∧ n = 52 :=
by
  sorry

end min_cookies_divisible_by_13_l98_98584


namespace quadrilateral_perimeter_l98_98425

theorem quadrilateral_perimeter
  (EF FG HG : ℝ)
  (h1 : EF = 7)
  (h2 : FG = 15)
  (h3 : HG = 3)
  (perp1 : EF * FG = 0)
  (perp2 : HG * FG = 0) :
  EF + FG + HG + Real.sqrt (4^2 + 15^2) = 25 + Real.sqrt 241 :=
by
  sorry

end quadrilateral_perimeter_l98_98425


namespace percentage_saved_on_hats_l98_98842

/-- Suppose the regular price of a hat is $60 and Maria buys four hats with progressive discounts: 
20% off the second hat, 40% off the third hat, and 50% off the fourth hat.
Prove that the percentage saved on the regular price for four hats is 27.5%. -/
theorem percentage_saved_on_hats :
  let regular_price := 60
  let discount_2 := 0.2 * regular_price
  let discount_3 := 0.4 * regular_price
  let discount_4 := 0.5 * regular_price
  let price_1 := regular_price
  let price_2 := regular_price - discount_2
  let price_3 := regular_price - discount_3
  let price_4 := regular_price - discount_4
  let total_regular := 4 * regular_price
  let total_discounted := price_1 + price_2 + price_3 + price_4
  let savings := total_regular - total_discounted
  let percentage_saved := (savings / total_regular) * 100
  percentage_saved = 27.5 :=
by
  sorry

end percentage_saved_on_hats_l98_98842


namespace david_chemistry_marks_l98_98374

theorem david_chemistry_marks :
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects
  chemistry = 97 :=
by
  -- Definition of variables
  let english := 96
  let mathematics := 95
  let physics := 82
  let biology := 95
  let average_all := 93
  let total_subjects := 5
  let total_other_subjects := english + mathematics + physics + biology
  let total_all_subjects := average_all * total_subjects
  let chemistry := total_all_subjects - total_other_subjects

  -- Assert the final value
  show chemistry = 97
  sorry

end david_chemistry_marks_l98_98374


namespace sum_of_angles_l98_98883

theorem sum_of_angles (A B C D E F : ℝ)
  (h1 : A + B + C = 180) 
  (h2 : D + E + F = 180) : 
  A + B + C + D + E + F = 360 := 
by 
  sorry

end sum_of_angles_l98_98883


namespace log_50_between_integers_l98_98768

open Real

-- Declaration of the proof problem
theorem log_50_between_integers (a b : ℤ) (h1 : log 10 = 1) (h2 : log 100 = 2) (h3 : 10 < 50) (h4 : 50 < 100) :
  a + b = 3 :=
by
  sorry

end log_50_between_integers_l98_98768


namespace find_c_l98_98096

theorem find_c (a b c : ℚ) (h_eqn : ∀ y, a * y^2 + b * y + c = y^2 / 12 + 5 * y / 6 + 145 / 12)
  (h_vertex : ∀ x, x = a * (-5)^2 + b * (-5) + c)
  (h_pass : a * (-1 + 5)^2 + 1 = 4) :
  c = 145 / 12 := by
sorry

end find_c_l98_98096


namespace smallest_positive_integer_l98_98210

theorem smallest_positive_integer :
  ∃ x : ℕ,
    x % 5 = 4 ∧
    x % 7 = 5 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    (∀ y : ℕ, (y % 5 = 4 ∧ y % 7 = 5 ∧ y % 11 = 9 ∧ y % 13 = 11) → y ≥ x) ∧ x = 999 :=
by
  sorry

end smallest_positive_integer_l98_98210


namespace find_vector_coordinates_l98_98156

structure Point3D :=
  (x y z : ℝ)

def vector_sub (a b : Point3D) : Point3D :=
  Point3D.mk (b.x - a.x) (b.y - a.y) (b.z - a.z)

theorem find_vector_coordinates (A B : Point3D)
  (hA : A = { x := 1, y := -3, z := 4 })
  (hB : B = { x := -3, y := 2, z := 1 }) :
  vector_sub A B = { x := -4, y := 5, z := -3 } :=
by
  -- Proof is omitted
  sorry

end find_vector_coordinates_l98_98156


namespace gcd_polynomial_997_l98_98872

theorem gcd_polynomial_997 (b : ℤ) (h : ∃ k : ℤ, b = 997 * k ∧ k % 2 = 1) :
  Int.gcd (3 * b ^ 2 + 17 * b + 31) (b + 7) = 1 := by
  sorry

end gcd_polynomial_997_l98_98872


namespace g_evaluation_l98_98266

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then (2 * a * b - a + 3) / (3 * a)
  else (a * b - b - 1) / (-3 * b)

theorem g_evaluation : g 2 1 + g 2 4 = 7 / 12 := 
by {
  sorry
}

end g_evaluation_l98_98266


namespace no_such_cuboid_exists_l98_98516

theorem no_such_cuboid_exists (a b c : ℝ) :
  a + b + c = 12 ∧ ab + bc + ca = 1 ∧ abc = 12 → false :=
by
  sorry

end no_such_cuboid_exists_l98_98516


namespace valid_expression_l98_98340

theorem valid_expression (x : ℝ) : 
  (x - 1 ≥ 0 ∧ x - 2 ≠ 0) ↔ (x ≥ 1 ∧ x ≠ 2) := 
by
  sorry

end valid_expression_l98_98340


namespace time_to_cross_first_platform_l98_98356

noncomputable section

def train_length : ℝ := 310
def platform_1_length : ℝ := 110
def platform_2_length : ℝ := 250
def crossing_time_platform_2 : ℝ := 20

def total_distance_2 (train_length platform_2_length : ℝ) : ℝ :=
  train_length + platform_2_length

def train_speed (total_distance_2 crossing_time_platform_2 : ℝ) : ℝ :=
  total_distance_2 / crossing_time_platform_2

def total_distance_1 (train_length platform_1_length : ℝ) : ℝ :=
  train_length + platform_1_length

def crossing_time_platform_1 (total_distance_1 train_speed : ℝ) : ℝ :=
  total_distance_1 / train_speed

theorem time_to_cross_first_platform :
  crossing_time_platform_1 (total_distance_1 train_length platform_1_length)
                           (train_speed (total_distance_2 train_length platform_2_length)
                                        crossing_time_platform_2) 
  = 15 :=
by
  -- We would prove this in a detailed proof which is omitted here.
  sorry

end time_to_cross_first_platform_l98_98356


namespace geometric_sequence_problem_l98_98309

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (h1 : a 1 + a 1 * q ^ 2 = 10) (h2 : a 1 * q + a 1 * q ^ 3 = 5) (h3 : geometric_sequence a q) :
  a 8 = 1 / 16 := sorry

end geometric_sequence_problem_l98_98309


namespace find_smallest_angle_l98_98155

theorem find_smallest_angle 
  (x y : ℝ)
  (hx : x + y = 45)
  (hy : y = x - 5)
  (hz : x > 0 ∧ y > 0 ∧ x + y < 180) :
  min x y = 20 := 
sorry

end find_smallest_angle_l98_98155


namespace awareness_not_related_to_education_level_l98_98226

def low_education : ℕ := 35 + 35 + 80 + 40 + 60 + 150
def high_education : ℕ := 55 + 64 + 6 + 110 + 140 + 25

def a : ℕ := 150
def b : ℕ := 125
def c : ℕ := 250
def d : ℕ := 275
def n : ℕ := 800

-- K^2 calculation
def K2 : ℚ := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Critical value for 95% confidence
def critical_value_95 : ℚ := 3.841

theorem awareness_not_related_to_education_level : K2 < critical_value_95 :=
by
  -- proof to be added here
  sorry

end awareness_not_related_to_education_level_l98_98226


namespace find_n_cosine_l98_98291

theorem find_n_cosine (n : ℤ) (h1 : 100 ≤ n ∧ n ≤ 300) (h2 : Real.cos (n : ℝ) = Real.cos 140) : n = 220 :=
by
  sorry

end find_n_cosine_l98_98291


namespace train_pass_jogger_in_40_seconds_l98_98330

noncomputable def time_to_pass_jogger (jogger_speed_kmh : ℝ) (train_speed_kmh : ℝ) (initial_distance_m : ℝ) (train_length_m : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - jogger_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)  -- Conversion from km/hr to m/s
  let total_distance_m := initial_distance_m + train_length_m
  total_distance_m / relative_speed_ms

theorem train_pass_jogger_in_40_seconds :
  time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end train_pass_jogger_in_40_seconds_l98_98330


namespace area_and_cost_of_path_l98_98407

-- Define the dimensions of the rectangular grass field
def length_field : ℝ := 75
def width_field : ℝ := 55

-- Define the width of the path around the field
def path_width : ℝ := 2.8

-- Define the cost per square meter for constructing the path
def cost_per_sq_m : ℝ := 2

-- Define the total length and width including the path
def total_length : ℝ := length_field + 2 * path_width
def total_width : ℝ := width_field + 2 * path_width

-- Define the area of the entire field including the path
def area_total : ℝ := total_length * total_width

-- Define the area of the grass field alone
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_total - area_field

-- Define the cost of constructing the path
def cost_path : ℝ := area_path * cost_per_sq_m

-- The statement to be proved
theorem area_and_cost_of_path :
  area_path = 759.36 ∧ cost_path = 1518.72 := by
  sorry

end area_and_cost_of_path_l98_98407


namespace triangle_coordinates_sum_l98_98713

noncomputable def coordinates_of_triangle_A (p q : ℚ) : Prop :=
  let B := (12, 19)
  let C := (23, 20)
  let area := ((B.1 * C.2 + C.1 * q + p * B.2) - (B.2 * C.1 + C.2 * p + q * B.1)) / 2 
  let M := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let median_slope := (q - M.2) / (p - M.1)
  area = 60 ∧ median_slope = 3 

theorem triangle_coordinates_sum (p q : ℚ) 
(h : coordinates_of_triangle_A p q) : p + q = 52 := 
sorry

end triangle_coordinates_sum_l98_98713


namespace fraction_of_a_eq_1_fifth_of_b_l98_98290

theorem fraction_of_a_eq_1_fifth_of_b (a b : ℝ) (x : ℝ) 
  (h1 : a + b = 100) 
  (h2 : (1/5) * b = 12)
  (h3 : b = 60) : x = 3/10 := by
  sorry

end fraction_of_a_eq_1_fifth_of_b_l98_98290


namespace solution_strategy_l98_98188

-- Defining the total counts for the groups
def total_elderly : ℕ := 28
def total_middle_aged : ℕ := 54
def total_young : ℕ := 81

-- The sample size we need
def sample_size : ℕ := 36

-- Proposing the strategy
def appropriate_sampling_method : Prop := 
  (total_elderly - 1) % sample_size.gcd (total_middle_aged.gcd total_young) = 0

theorem solution_strategy :
  appropriate_sampling_method :=
by {
  sorry
}

end solution_strategy_l98_98188


namespace find_x_l98_98378

noncomputable def h (x : ℝ) : ℝ := (2 * x^2 + 3 * x + 1)^(1 / 3) / 5^(1/3)

theorem find_x (x : ℝ) :
  h (3 * x) = 3 * h x ↔ x = -1 + (10^(1/2)) / 3 ∨ x = -1 - (10^(1/2)) / 3 := by
  sorry

end find_x_l98_98378


namespace new_pressure_of_helium_l98_98152

noncomputable def helium_pressure (p V p' V' : ℝ) (k : ℝ) : Prop :=
  p * V = k ∧ p' * V' = k

theorem new_pressure_of_helium :
  ∀ (p V p' V' k : ℝ), 
  p = 8 ∧ V = 3.5 ∧ V' = 7 ∧ k = 28 →
  helium_pressure p V p' V' k →
  p' = 4 :=
by
  intros p V p' V' k h1 h2
  sorry

end new_pressure_of_helium_l98_98152


namespace convex_polyhedron_in_inscribed_sphere_l98_98525

-- Definitions based on conditions
variables (S c r : ℝ) (S' V R : ℝ)

-- The given relationship for a convex polygon.
def poly_relationship := S = (1 / 2) * c * r

-- The desired relationship for a convex polyhedron.
def polyhedron_relationship := V = (1 / 3) * S' * R

-- Proof statement
theorem convex_polyhedron_in_inscribed_sphere (S c r S' V R : ℝ) 
  (poly : S = (1 / 2) * c * r) : V = (1 / 3) * S' * R :=
sorry

end convex_polyhedron_in_inscribed_sphere_l98_98525


namespace union_sets_example_l98_98606

theorem union_sets_example : ({0, 1} ∪ {2} : Set ℕ) = {0, 1, 2} := by 
  sorry

end union_sets_example_l98_98606


namespace work_done_in_days_l98_98260

theorem work_done_in_days (M B : ℕ) (x : ℕ) 
  (h1 : 12 * 2 * B + 16 * B = 200 * B / 5) 
  (h2 : 13 * 2 * B + 24 * B = 50 * x * B)
  (h3 : M = 2 * B) : 
  x = 4 := 
by
  sorry

end work_done_in_days_l98_98260


namespace sum_and_product_of_conjugates_l98_98355

theorem sum_and_product_of_conjugates (c d : ℚ) 
  (h1 : 2 * c = 6)
  (h2 : c^2 - 4 * d = 4) :
  c + d = 17 / 4 :=
by
  sorry

end sum_and_product_of_conjugates_l98_98355


namespace min_value_quadratic_l98_98780

noncomputable def quadratic_min (a c : ℝ) : ℝ :=
  (2 / a) + (2 / c)

theorem min_value_quadratic {a c : ℝ} (ha : a > 0) (hc : c > 0) (hac : a * c = 1/4) : 
  quadratic_min a c = 8 :=
sorry

end min_value_quadratic_l98_98780


namespace M_intersect_N_equals_M_l98_98256

-- Define the sets M and N
def M := { x : ℝ | x^2 - 3 * x + 2 = 0 }
def N := { x : ℝ | x * (x - 1) * (x - 2) = 0 }

-- The theorem we want to prove
theorem M_intersect_N_equals_M : M ∩ N = M := 
by 
  sorry

end M_intersect_N_equals_M_l98_98256


namespace value_of_expression_l98_98986

theorem value_of_expression : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 :=
by
  sorry

end value_of_expression_l98_98986


namespace area_of_octagon_l98_98287

-- Define the basic geometric elements and properties
variables {A B C D E F G H : Type}
variables (isRectangle : BDEF A B C D E F G H)
variables (AB BC : ℝ) (hAB : AB = 2) (hBC : BC = 2)
variables (isRightIsosceles : ABC A B C D E F G H)

-- Assumptions and known facts
def BDEF_is_rectangle : Prop := isRectangle
def AB_eq_2 : AB = 2 := hAB
def BC_eq_2 : BC = 2 := hBC
def ABC_is_right_isosceles : Prop := isRightIsosceles

-- Statement of the problem to be proved
theorem area_of_octagon : (exists (area : ℝ), area = 8 * Real.sqrt 2) :=
by {
  -- The proof details will go here, which we skip for now
  sorry
}

end area_of_octagon_l98_98287


namespace probability_of_non_perimeter_square_l98_98638

-- Defining the total number of squares on a 10x10 board
def total_squares : ℕ := 10 * 10

-- Defining the number of perimeter squares
def perimeter_squares : ℕ := 10 + 10 + (10 - 2) * 2

-- Defining the number of non-perimeter squares
def non_perimeter_squares : ℕ := total_squares - perimeter_squares

-- Defining the probability of selecting a non-perimeter square
def probability_non_perimeter : ℚ := non_perimeter_squares / total_squares

-- The main theorem statement to be proved
theorem probability_of_non_perimeter_square:
  probability_non_perimeter = 16 / 25 := 
sorry

end probability_of_non_perimeter_square_l98_98638


namespace equal_number_of_boys_and_girls_l98_98855

theorem equal_number_of_boys_and_girls
  (m d M D : ℕ)
  (h1 : (M / m) ≠ (D / d))
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : m = d :=
sorry

end equal_number_of_boys_and_girls_l98_98855


namespace second_storm_duration_l98_98273

theorem second_storm_duration (x y : ℕ) 
  (h1 : x + y = 45) 
  (h2 : 30 * x + 15 * y = 975) : 
  y = 25 :=
by
  sorry

end second_storm_duration_l98_98273


namespace sum_integers_30_to_50_subtract_15_l98_98299

-- Definitions and proof problem based on conditions
def sumIntSeries (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_30_to_50_subtract_15 : sumIntSeries 30 50 - 15 = 825 := by
  -- We are stating that the sum of the integers from 30 to 50 minus 15 is equal to 825
  sorry


end sum_integers_30_to_50_subtract_15_l98_98299


namespace total_cost_of_shirts_l98_98876

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l98_98876


namespace vector_normalization_condition_l98_98467

variables {a b : ℝ} -- Ensuring that Lean understands ℝ refers to real numbers and specifically vectors in ℝ before using it in the next parts.

-- Definitions of the vector variables
variables (a b : ℝ) (ab_non_zero : a ≠ 0 ∧ b ≠ 0)

-- Required statement
theorem vector_normalization_condition (a b : ℝ) 
(h₀ : a ≠ 0 ∧ b ≠ 0) :
  (a / abs a = b / abs b) ↔ (a = 2 * b) :=
sorry

end vector_normalization_condition_l98_98467


namespace books_total_l98_98432

theorem books_total (Tim_books Sam_books : ℕ) (h1 : Tim_books = 44) (h2 : Sam_books = 52) : Tim_books + Sam_books = 96 := 
by
  sorry

end books_total_l98_98432


namespace find_M_l98_98611

theorem find_M 
  (M : ℕ)
  (h : 997 + 999 + 1001 + 1003 + 1005 = 5100 - M) :
  M = 95 :=
by
  sorry

end find_M_l98_98611


namespace lara_flowers_l98_98279

theorem lara_flowers (M : ℕ) : 52 - M - (M + 6) - 16 = 0 → M = 15 :=
by
  sorry

end lara_flowers_l98_98279


namespace remainder_of_17_power_1801_mod_28_l98_98526

theorem remainder_of_17_power_1801_mod_28 : (17 ^ 1801) % 28 = 17 := 
by
  sorry

end remainder_of_17_power_1801_mod_28_l98_98526


namespace isosceles_triangle_angles_l98_98007

noncomputable 
def is_triangle_ABC_isosceles (A B C : ℝ) (alpha beta : ℝ) (AB AC : ℝ) 
  (h1 : AB = AC) (h2 : alpha = 2 * beta) : Prop :=
  180 - 3 * beta = C ∧ C / 2 = 90 - 1.5 * beta

theorem isosceles_triangle_angles (A B C C1 C2 : ℝ) (alpha beta : ℝ) (AB AC : ℝ)
  (h1 : AB = AC) (h2 : alpha = 2 * beta) :
  (180 - 3 * beta) / 2 = 90 - 1.5 * beta :=
by sorry

end isosceles_triangle_angles_l98_98007


namespace extreme_values_x_axis_l98_98888

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

theorem extreme_values_x_axis (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, f a b c x = x * (a * x^2 + b * x + c))
  (h3 : ∀ x, deriv (f a b c) x = 3 * a * x^2 + 2 * b * x + c)
  (h4 : deriv (f a b c) 1 = 0)
  (h5 : deriv (f a b c) (-1) = 0) :
  b = 0 :=
sorry

end extreme_values_x_axis_l98_98888


namespace largest_among_numbers_l98_98500

theorem largest_among_numbers :
  ∀ (a b c d e : ℝ), 
  a = 0.997 ∧ b = 0.9799 ∧ c = 0.999 ∧ d = 0.9979 ∧ e = 0.979 →
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by intros a b c d e habcde
   rcases habcde with ⟨ha, hb, hc, hd, he⟩
   simp [ha, hb, hc, hd, he]
   sorry

end largest_among_numbers_l98_98500


namespace find_a_to_satisfy_divisibility_l98_98074

theorem find_a_to_satisfy_divisibility (a : ℕ) (h₀ : 0 ≤ a) (h₁ : a < 11) (h₂ : (2 * 10^10 + a) % 11 = 0) : a = 9 :=
sorry

end find_a_to_satisfy_divisibility_l98_98074


namespace roger_has_more_candies_l98_98315

def candies_sandra_bag1 : ℕ := 6
def candies_sandra_bag2 : ℕ := 6
def candies_roger_bag1 : ℕ := 11
def candies_roger_bag2 : ℕ := 3

def total_candies_sandra := candies_sandra_bag1 + candies_sandra_bag2
def total_candies_roger := candies_roger_bag1 + candies_roger_bag2

theorem roger_has_more_candies : (total_candies_roger - total_candies_sandra) = 2 := by
  sorry

end roger_has_more_candies_l98_98315


namespace find_cost_per_batch_l98_98511

noncomputable def cost_per_tire : ℝ := 8
noncomputable def selling_price_per_tire : ℝ := 20
noncomputable def profit_per_tire : ℝ := 10.5
noncomputable def number_of_tires : ℕ := 15000

noncomputable def total_cost (C : ℝ) : ℝ := C + cost_per_tire * number_of_tires
noncomputable def total_revenue : ℝ := selling_price_per_tire * number_of_tires
noncomputable def total_profit : ℝ := profit_per_tire * number_of_tires

theorem find_cost_per_batch (C : ℝ) :
  total_profit = total_revenue - total_cost C → C = 22500 := by
  sorry

end find_cost_per_batch_l98_98511


namespace age_of_b_l98_98965

variable (a b : ℕ)
variable (h1 : a * 3 = b * 5)
variable (h2 : (a + 2) * 2 = (b + 2) * 3)

theorem age_of_b : b = 6 :=
by
  sorry

end age_of_b_l98_98965


namespace lock_combination_correct_l98_98627

noncomputable def lock_combination : ℤ := 812

theorem lock_combination_correct :
  ∀ (S T A R : ℕ), S ≠ T → S ≠ A → S ≠ R → T ≠ A → T ≠ R → A ≠ R →
  ((S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S) + 
   (T * 9^4 + A * 9^3 + R * 9^2 + T * 9 + S) + 
   (S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + T)) % 9^5 = 
  S * 9^4 + T * 9^3 + A * 9^2 + R * 9 + S →
  (S * 9^2 + T * 9^1 + A) = lock_combination := 
by
  intros S T A R hST hSA hSR hTA hTR hAR h_eq
  sorry

end lock_combination_correct_l98_98627


namespace Olivia_hours_worked_on_Monday_l98_98751

/-- Olivia works on multiple days in a week with given wages per hour and total income -/
theorem Olivia_hours_worked_on_Monday 
  (M : ℕ)  -- Hours worked on Monday
  (rate_per_hour : ℕ := 9) -- Olivia’s earning rate per hour
  (hours_Wednesday : ℕ := 3)  -- Hours worked on Wednesday
  (hours_Friday : ℕ := 6)  -- Hours worked on Friday
  (total_income : ℕ := 117)  -- Total income earned this week
  (hours_total : ℕ := hours_Wednesday + hours_Friday + M)
  (income_calc : ℕ := rate_per_hour * hours_total) :
  -- Prove that the hours worked on Monday is 4 given the conditions
  income_calc = total_income → M = 4 :=
by
  sorry

end Olivia_hours_worked_on_Monday_l98_98751


namespace pigeon_distance_l98_98755

-- Define the conditions
def pigeon_trip (d : ℝ) (v : ℝ) (wind : ℝ) (time_nowind : ℝ) (time_wind : ℝ) :=
  (2 * d / v = time_nowind) ∧
  (d / (v + wind) + d / (v - wind) = time_wind)

-- Define the theorems to be proven
theorem pigeon_distance : ∃ (d : ℝ), pigeon_trip d 40 10 3.75 4 ∧ d = 75 :=
  by {
  sorry
}

end pigeon_distance_l98_98755


namespace xiaoqiang_average_score_l98_98647

theorem xiaoqiang_average_score
    (x : ℕ)
    (prev_avg : ℝ)
    (next_score : ℝ)
    (target_avg : ℝ)
    (h_prev_avg : prev_avg = 84)
    (h_next_score : next_score = 100)
    (h_target_avg : target_avg = 86) :
    (86 * x - (84 * (x - 1)) = 100) → x = 8 := 
by
  intros h_eq
  sorry

end xiaoqiang_average_score_l98_98647


namespace num_decompositions_144_l98_98625

theorem num_decompositions_144 : ∃ D, D = 45 ∧ 
  (∀ (factors : List ℕ), 
    (∀ x, x ∈ factors → x > 1) ∧ factors.prod = 144 → 
    factors.permutations.length = D) :=
sorry

end num_decompositions_144_l98_98625


namespace student_average_grade_l98_98709

noncomputable def average_grade_two_years : ℝ :=
  let year1_courses := 6
  let year1_average_grade := 100
  let year1_total_points := year1_courses * year1_average_grade

  let year2_courses := 5
  let year2_average_grade := 40
  let year2_total_points := year2_courses * year2_average_grade

  let total_courses := year1_courses + year2_courses
  let total_points := year1_total_points + year2_total_points

  total_points / total_courses

theorem student_average_grade : average_grade_two_years = 72.7 :=
by
  sorry

end student_average_grade_l98_98709


namespace max_value_of_a_l98_98731

theorem max_value_of_a :
  ∀ (a : ℚ),
  (∀ (m : ℚ), 1/3 < m ∧ m < a →
   (∀ (x : ℤ), 0 < x ∧ x ≤ 200 →
    ¬ (∃ (y : ℤ), y = m * x + 3 ∨ y = m * x + 1))) →
  a = 68/201 :=
by
  sorry

end max_value_of_a_l98_98731


namespace moon_radius_scientific_notation_l98_98642

noncomputable def moon_radius : ℝ := 1738000

theorem moon_radius_scientific_notation :
  moon_radius = 1.738 * 10^6 :=
by
  sorry

end moon_radius_scientific_notation_l98_98642


namespace find_coefficients_l98_98450

def polynomial (a b : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 3 - 3 * x ^ 2 + b * x - 7

theorem find_coefficients (a b : ℝ) :
  polynomial a b 2 = -17 ∧ polynomial a b (-1) = -11 → a = 0 ∧ b = -1 :=
by
  sorry

end find_coefficients_l98_98450


namespace derivative_y_l98_98136

noncomputable def y (x : ℝ) : ℝ := Real.sin x - Real.exp (x * Real.log 2)

theorem derivative_y (x : ℝ) : 
  deriv y x = Real.cos x - Real.exp (x * Real.log 2) * Real.log 2 := 
by 
  sorry

end derivative_y_l98_98136


namespace plain_pancakes_l98_98186

/-- Define the given conditions -/
def total_pancakes : ℕ := 67
def blueberry_pancakes : ℕ := 20
def banana_pancakes : ℕ := 24

/-- Define a theorem stating the number of plain pancakes given the conditions -/
theorem plain_pancakes : total_pancakes - (blueberry_pancakes + banana_pancakes) = 23 := by
  -- Here we will provide a proof
  sorry

end plain_pancakes_l98_98186


namespace distance_between_parallel_lines_l98_98459

theorem distance_between_parallel_lines (r d : ℝ) 
  (h1 : ∃ p1 p2 p3 : ℝ, p1 = 40 ∧ p2 = 40 ∧ p3 = 36) 
  (h2 : ∀ θ : ℝ, ∃ A B C D : ℝ → ℝ, 
    (A θ - B θ) = 40 ∧ (C θ - D θ) = 36) : d = 6 :=
sorry

end distance_between_parallel_lines_l98_98459


namespace min_length_MN_l98_98505

theorem min_length_MN (a b : ℝ) (H h : ℝ) (MN : ℝ) (midsegment_eq_4 : (a + b) / 2 = 4)
    (area_div_eq_half : (a + MN) / 2 * h = (MN + b) / 2 * H) : MN = 4 :=
by
  sorry

end min_length_MN_l98_98505


namespace proof_A_union_B_eq_R_l98_98189

def A : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - 5) < a }

theorem proof_A_union_B_eq_R (a : ℝ) (h : a > 6) : 
  A ∪ B a = Set.univ :=
by {
  sorry
}

end proof_A_union_B_eq_R_l98_98189


namespace kaleb_books_l98_98303

-- Define the initial number of books
def initial_books : ℕ := 34

-- Define the number of books sold
def books_sold : ℕ := 17

-- Define the number of books bought
def books_bought : ℕ := 7

-- Prove that the final number of books is 24
theorem kaleb_books (h : initial_books - books_sold + books_bought = 24) : initial_books - books_sold + books_bought = 24 :=
by
  exact h

end kaleb_books_l98_98303


namespace Masha_thought_of_numbers_l98_98757

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l98_98757


namespace hyperbola_eccentricity_l98_98800

-- Definitions based on the conditions
def hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0)

def distance_from_focus_to_asymptote (a b c : ℝ) : Prop :=
  (b^2 * c) / (a^2 + b^2).sqrt = b ∧ b = 2 * Real.sqrt 3

def minimum_distance_point_to_focus (a c : ℝ) : Prop :=
  c - a = 2

def eccentricity (a c e : ℝ) : Prop :=
  e = c / a

-- Problem statement
theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_hyperbola : hyperbola a b)
  (h_dist_asymptote : distance_from_focus_to_asymptote a b c)
  (h_min_dist_focus : minimum_distance_point_to_focus a c)
  (h_eccentricity : eccentricity a c e) :
  e = 2 :=
sorry

end hyperbola_eccentricity_l98_98800


namespace set_D_cannot_form_triangle_l98_98895

-- Definition for triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given lengths
def length_1 := 1
def length_2 := 2
def length_3 := 3

-- The proof problem statement
theorem set_D_cannot_form_triangle : ¬ triangle_inequality length_1 length_2 length_3 :=
  by sorry

end set_D_cannot_form_triangle_l98_98895


namespace Jazmin_strips_width_l98_98614

theorem Jazmin_strips_width (w1 w2 g : ℕ) (h1 : w1 = 44) (h2 : w2 = 33) (hg : g = Nat.gcd w1 w2) : g = 11 := by
  -- Markdown above outlines:
  -- w1, w2 are widths of the construction paper
  -- h1: w1 = 44
  -- h2: w2 = 33
  -- hg: g = gcd(w1, w2)
  -- Prove g == 11
  sorry

end Jazmin_strips_width_l98_98614


namespace race_meeting_time_l98_98987

noncomputable def track_length : ℕ := 500
noncomputable def first_meeting_from_marie_start : ℕ := 100
noncomputable def time_until_first_meeting : ℕ := 2
noncomputable def second_meeting_time : ℕ := 12

theorem race_meeting_time
  (h1 : track_length = 500)
  (h2 : first_meeting_from_marie_start = 100)
  (h3 : time_until_first_meeting = 2)
  (h4 : ∀ t v1 v2 : ℕ, t * (v1 + v2) = track_length)
  (h5 : 12 = second_meeting_time) :
  second_meeting_time = 12 := by
  sorry

end race_meeting_time_l98_98987


namespace residue_12_2040_mod_19_l98_98490

theorem residue_12_2040_mod_19 :
  12^2040 % 19 = 7 := 
sorry

end residue_12_2040_mod_19_l98_98490


namespace grandfather_age_5_years_back_l98_98799

variable (F S G : ℕ)

-- Conditions
def father_age : Prop := F = 58
def son_current_age : Prop := S = 58 - S
def son_grandfather_age_relation : Prop := S - 5 = 1 / 2 * (G - 5)

-- Theorem: Prove the grandfather's age 5 years back given the conditions.
theorem grandfather_age_5_years_back (h1 : father_age F) (h2 : son_current_age S) (h3 : son_grandfather_age_relation S G) : G = 2 * S - 5 :=
sorry

end grandfather_age_5_years_back_l98_98799


namespace polynomial_value_at_five_l98_98195

def f (x : ℤ) : ℤ := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

theorem polynomial_value_at_five : f 5 = 2677 := by
  -- The proof goes here.
  sorry

end polynomial_value_at_five_l98_98195


namespace total_spent_correct_l98_98463

def shorts : ℝ := 13.99
def shirt : ℝ := 12.14
def jacket : ℝ := 7.43
def total_spent : ℝ := 33.56

theorem total_spent_correct : shorts + shirt + jacket = total_spent :=
by
  sorry

end total_spent_correct_l98_98463


namespace find_num_male_general_attendees_l98_98254

def num_attendees := 1000
def num_presenters := 420
def total_general_attendees := num_attendees - num_presenters

variables (M_p F_p M_g F_g : ℕ)

axiom condition1 : M_p = F_p + 20
axiom condition2 : M_p + F_p = 420
axiom condition3 : F_g = M_g + 56
axiom condition4 : M_g + F_g = total_general_attendees

theorem find_num_male_general_attendees :
  M_g = 262 :=
by
  sorry

end find_num_male_general_attendees_l98_98254


namespace factorial_comparison_l98_98395

theorem factorial_comparison :
  (Nat.factorial (Nat.factorial 100)) <
  (Nat.factorial 99)^(Nat.factorial 100) * (Nat.factorial 100)^(Nat.factorial 99) :=
  sorry

end factorial_comparison_l98_98395


namespace gcd_of_1887_and_2091_is_51_l98_98508

variable (a b : Nat)
variable (coefficient1 coefficient2 quotient1 quotient2 quotient3 remainder1 remainder2 : Nat)

def gcd_condition1 : Prop := (b = 1 * a + remainder1)
def gcd_condition2 : Prop := (a = quotient1 * remainder1 + remainder2)
def gcd_condition3 : Prop := (remainder1 = quotient2 * remainder2)

def numbers_1887_and_2091 : Prop := (a = 1887) ∧ (b = 2091)

theorem gcd_of_1887_and_2091_is_51 :
  numbers_1887_and_2091 a b ∧
  gcd_condition1 a b remainder1 ∧ 
  gcd_condition2 a remainder1 remainder2 quotient1 ∧ 
  gcd_condition3 remainder1 remainder2 quotient2 → 
  Nat.gcd 1887 2091 = 51 :=
by
  sorry

end gcd_of_1887_and_2091_is_51_l98_98508


namespace possible_values_f_zero_l98_98108

theorem possible_values_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
    f 0 = 0 ∨ f 0 = 1 / 2 := 
sorry

end possible_values_f_zero_l98_98108


namespace symmetric_circle_with_respect_to_origin_l98_98618

theorem symmetric_circle_with_respect_to_origin :
  ∀ x y : ℝ, (x + 2) ^ 2 + (y - 1) ^ 2 = 1 → (x - 2) ^ 2 + (y + 1) ^ 2 = 1 :=
by
  intros x y h
  -- Symmetric transformation and verification will be implemented here
  sorry

end symmetric_circle_with_respect_to_origin_l98_98618


namespace sin_trig_identity_l98_98512

theorem sin_trig_identity (α : ℝ) (h : Real.sin (α - π/4) = 1/2) : Real.sin ((5 * π) / 4 - α) = 1/2 := 
by 
  sorry

end sin_trig_identity_l98_98512


namespace sqrt_720_simplified_l98_98717

theorem sqrt_720_simplified : Real.sqrt 720 = 6 * Real.sqrt 5 := sorry

end sqrt_720_simplified_l98_98717


namespace discount_on_soap_l98_98944

theorem discount_on_soap :
  (let chlorine_price := 10
   let chlorine_discount := 0.20 * chlorine_price
   let discounted_chlorine_price := chlorine_price - chlorine_discount

   let soap_price := 16

   let total_savings := 26

   let chlorine_savings := 3 * chlorine_price - 3 * discounted_chlorine_price
   let soap_savings := total_savings - chlorine_savings

   let discount_per_soap := soap_savings / 5
   let discount_percentage_per_soap := (discount_per_soap / soap_price) * 100
   discount_percentage_per_soap = 25) := sorry

end discount_on_soap_l98_98944


namespace count_divisors_divisible_exactly_2007_l98_98363

-- Definitions and conditions
def prime_factors_2006 : List Nat := [2, 17, 59]

def prime_factors_2006_pow_2006 : List (Nat × Nat) := [(2, 2006), (17, 2006), (59, 2006)]

def number_of_divisors (n : Nat) : Nat :=
  prime_factors_2006_pow_2006.foldl (λ acc ⟨p, exp⟩ => acc * (exp + 1)) 1

theorem count_divisors_divisible_exactly_2007 : 
  (number_of_divisors (2^2006 * 17^2006 * 59^2006) = 3) :=
  sorry

end count_divisors_divisible_exactly_2007_l98_98363


namespace determine_coefficients_l98_98338

theorem determine_coefficients (a b c : ℝ) (x y : ℝ) :
  (x = 3/4 ∧ y = 5/8) →
  (a * (x - 1) + 2 * y = 1) →
  (b * |x - 1| + c * y = 3) →
  (a = 1 ∧ b = 2 ∧ c = 4) := 
by 
  intros 
  sorry

end determine_coefficients_l98_98338


namespace projectile_height_reaches_49_l98_98554

theorem projectile_height_reaches_49 (t : ℝ) :
  (∃ t : ℝ, 49 = -20 * t^2 + 100 * t) → t = 0.7 :=
by
  sorry

end projectile_height_reaches_49_l98_98554


namespace problem_statement_l98_98278

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x + 2) * (x - 1) > 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 0}
def C_U (B : Set ℝ) : Set ℝ := {x | x ∉ B}

theorem problem_statement : A ∪ C_U B = {x | x < -2 ∨ x ≥ 0} :=
by
  sorry

end problem_statement_l98_98278


namespace max_marks_is_400_l98_98415

theorem max_marks_is_400 :
  ∃ M : ℝ, (0.30 * M = 120) ∧ (M = 400) := 
by 
  sorry

end max_marks_is_400_l98_98415


namespace factor_expression_eq_l98_98398

-- Define the given expression
def given_expression (x : ℝ) : ℝ :=
  (12 * x^3 + 90 * x - 6) - (-3 * x^3 + 5 * x - 6)

-- Define the correct factored form
def factored_expression (x : ℝ) : ℝ :=
  5 * x * (3 * x^2 + 17)

-- The theorem stating the equality of the given expression and its factored form
theorem factor_expression_eq (x : ℝ) : given_expression x = factored_expression x :=
  by
  sorry

end factor_expression_eq_l98_98398


namespace used_computer_lifespan_l98_98712

-- Problem statement
theorem used_computer_lifespan (cost_new : ℕ) (lifespan_new : ℕ) (cost_used : ℕ) (num_used : ℕ) (savings : ℕ) :
  cost_new = 600 →
  lifespan_new = 6 →
  cost_used = 200 →
  num_used = 2 →
  savings = 200 →
  ((cost_new - savings = num_used * cost_used) → (2 * (lifespan_new / 2) = 6) → lifespan_new / 2 = 3)
:= by
  intros
  sorry

end used_computer_lifespan_l98_98712


namespace tomatoes_left_after_yesterday_correct_l98_98711

def farmer_initial_tomatoes := 160
def tomatoes_picked_yesterday := 56
def tomatoes_left_after_yesterday : ℕ := farmer_initial_tomatoes - tomatoes_picked_yesterday

theorem tomatoes_left_after_yesterday_correct :
  tomatoes_left_after_yesterday = 104 :=
by
  unfold tomatoes_left_after_yesterday
  -- Proof goes here
  sorry

end tomatoes_left_after_yesterday_correct_l98_98711


namespace Moscow1964_27th_MMO_l98_98538

theorem Moscow1964_27th_MMO {a : ℤ} (h : ∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) : 
  a = 27^1964 :=
sorry

end Moscow1964_27th_MMO_l98_98538


namespace admission_fee_for_adults_l98_98304

theorem admission_fee_for_adults (C : ℝ) (N M N_c N_a : ℕ) (A : ℝ) 
  (h1 : C = 1.50) 
  (h2 : N = 2200) 
  (h3 : M = 5050) 
  (h4 : N_c = 700) 
  (h5 : N_a = 1500) :
  A = 2.67 := 
by
  sorry

end admission_fee_for_adults_l98_98304


namespace vertex_of_quadratic_function_l98_98255

-- Define the function and constants
variables (p q : ℝ)
  (hp : p > 0)
  (hq : q > 0)

-- State the theorem
theorem vertex_of_quadratic_function : 
  ∀ p q : ℝ, p > 0 → q > 0 → 
  (∀ x : ℝ, x = - (2 * p) / (2 : ℝ) → x = -p) := 
sorry

end vertex_of_quadratic_function_l98_98255


namespace seed_mixture_ryegrass_percent_l98_98458

theorem seed_mixture_ryegrass_percent (R : ℝ) :
  let X := 0.40
  let percentage_X_in_mixture := 1 / 3
  let percentage_Y_in_mixture := 2 / 3
  let final_ryegrass := 0.30
  (final_ryegrass = percentage_X_in_mixture * X + percentage_Y_in_mixture * R) → 
  R = 0.25 :=
by
  intros X percentage_X_in_mixture percentage_Y_in_mixture final_ryegrass H
  sorry

end seed_mixture_ryegrass_percent_l98_98458


namespace malvina_correct_l98_98862
noncomputable def angle (x : ℝ) : Prop := 0 < x ∧ x < 180
noncomputable def malvina_identifies (x : ℝ) : Prop := x > 90

noncomputable def sum_of_values := (Real.sqrt 5 + Real.sqrt 2) / 2

theorem malvina_correct (x : ℝ) (h1 : angle x) (h2 : malvina_identifies x) :
  sum_of_values = (Real.sqrt 5 + Real.sqrt 2) / 2 :=
by sorry

end malvina_correct_l98_98862


namespace max_non_multiples_of_3_l98_98727

theorem max_non_multiples_of_3 (a b c d e f : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h2 : a * b * c * d * e * f % 3 = 0) : 
  ¬ ∃ (count : ℕ), count > 5 ∧ (∀ x ∈ [a, b, c, d, e, f], x % 3 ≠ 0) :=
by
  sorry

end max_non_multiples_of_3_l98_98727


namespace Megan_bought_24_eggs_l98_98834

def eggs_problem : Prop :=
  ∃ (p c b : ℕ),
    b = 3 ∧
    c = 2 * b ∧
    p - c = 9 ∧
    p + c + b = 24

theorem Megan_bought_24_eggs : eggs_problem :=
  sorry

end Megan_bought_24_eggs_l98_98834


namespace correct_option_is_optionB_l98_98870

-- Definitions based on conditions
def optionA : ℝ := 0.37 * 1.5
def optionB : ℝ := 3.7 * 1.5
def optionC : ℝ := 0.37 * 1500
def original : ℝ := 0.37 * 15

-- Statement to prove that the correct answer (optionB) yields the same result as the original expression
theorem correct_option_is_optionB : optionB = original :=
sorry

end correct_option_is_optionB_l98_98870


namespace find_f_1_0_plus_f_2_0_general_form_F_l98_98703

variable {F : ℝ → ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ a, F a a = a
axiom cond2 : ∀ (k a b : ℝ), F (k * a) (k * b) = k * F a b
axiom cond3 : ∀ (a1 a2 b1 b2 : ℝ), F (a1 + a2) (b1 + b2) = F a1 b1 + F a2 b2
axiom cond4 : ∀ (a b : ℝ), F a b = F b ((a + b) / 2)

-- Proof problem
theorem find_f_1_0_plus_f_2_0 : F 1 0 + F 2 0 = 0 :=
sorry

theorem general_form_F : ∀ (x y : ℝ), F x y = y :=
sorry

end find_f_1_0_plus_f_2_0_general_form_F_l98_98703


namespace pencils_given_out_l98_98817

-- Defining the conditions
def num_children : ℕ := 4
def pencils_per_child : ℕ := 2

-- Formulating the problem statement, with the goal to prove the total number of pencils
theorem pencils_given_out : num_children * pencils_per_child = 8 := 
by 
  sorry

end pencils_given_out_l98_98817


namespace complete_square_transform_l98_98569

theorem complete_square_transform (x : ℝ) (h : x^2 + 8*x + 7 = 0) : (x + 4)^2 = 9 :=
by sorry

end complete_square_transform_l98_98569


namespace arrange_natural_numbers_divisors_l98_98083

theorem arrange_natural_numbers_divisors :
  ∃ (seq : List ℕ), seq = [7, 1, 8, 4, 10, 6, 9, 3, 2, 5] ∧ 
  seq.length = 10 ∧
  ∀ n (h : n < seq.length), seq[n] ∣ (List.take n seq).sum := 
by
  sorry

end arrange_natural_numbers_divisors_l98_98083


namespace find_a4_l98_98454

variable {a : ℕ → ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) * a (n - 1) = a n * a n

def given_sequence_conditions (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ a 2 + a 6 = 34 ∧ a 3 * a 5 = 64

-- Statement
theorem find_a4 (a : ℕ → ℝ) (h : given_sequence_conditions a) : a 4 = 8 :=
sorry

end find_a4_l98_98454


namespace corresponding_angle_C1_of_similar_triangles_l98_98645

theorem corresponding_angle_C1_of_similar_triangles
  (α β γ : ℝ)
  (ABC_sim_A1B1C1 : true)
  (angle_A : α = 50)
  (angle_B : β = 95) :
  γ = 35 :=
by
  sorry

end corresponding_angle_C1_of_similar_triangles_l98_98645


namespace betty_cookies_and_brownies_difference_l98_98444

-- Definitions based on the conditions
def initial_cookies : ℕ := 60
def initial_brownies : ℕ := 10
def cookies_per_day : ℕ := 3
def brownies_per_day : ℕ := 1
def days : ℕ := 7

-- The proof statement
theorem betty_cookies_and_brownies_difference :
  initial_cookies - (cookies_per_day * days) - (initial_brownies - (brownies_per_day * days)) = 36 :=
by
  sorry

end betty_cookies_and_brownies_difference_l98_98444


namespace find_expression_for_a_n_l98_98934

noncomputable def seq (n : ℕ) : ℕ := sorry
def sumFirstN (n : ℕ) : ℕ := sorry

theorem find_expression_for_a_n (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∀ n, S n + 1 = 2 * a n) :
  ∀ n, a n = 2^(n-1) :=
sorry

end find_expression_for_a_n_l98_98934


namespace first_digit_after_decimal_correct_l98_98211

noncomputable def first_digit_after_decimal (n: ℕ) : ℕ :=
  if n % 2 = 0 then 9 else 4

theorem first_digit_after_decimal_correct (n : ℕ) :
  (first_digit_after_decimal n = 9 ↔ n % 2 = 0) ∧ (first_digit_after_decimal n = 4 ↔ n % 2 = 1) :=
by
  sorry

end first_digit_after_decimal_correct_l98_98211


namespace hotel_friends_count_l98_98543

theorem hotel_friends_count
  (n : ℕ)
  (friend_share extra friend_payment : ℕ)
  (h1 : 7 * 80 + friend_payment = 720)
  (h2 : friend_payment = friend_share + extra)
  (h3 : friend_payment = 160)
  (h4 : extra = 70)
  (h5 : friend_share = 90) :
  n = 8 :=
sorry

end hotel_friends_count_l98_98543


namespace evaluate_expression_l98_98609

-- Define the operation * given by the table
def op (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1,1) => 1 | (1,2) => 2 | (1,3) => 3 | (1,4) => 4
  | (2,1) => 2 | (2,2) => 4 | (2,3) => 1 | (2,4) => 3
  | (3,1) => 3 | (3,2) => 1 | (3,3) => 4 | (3,4) => 2
  | (4,1) => 4 | (4,2) => 3 | (4,3) => 2 | (4,4) => 1
  | _ => 0  -- default to handle cases outside the defined table

-- Define the theorem to prove $(2*4)*(1*3) = 4$
theorem evaluate_expression : op (op 2 4) (op 1 3) = 4 := by
  sorry

end evaluate_expression_l98_98609


namespace value_of_x_l98_98044

theorem value_of_x (a b x : ℝ) (h : x^2 + 4 * b^2 = (2 * a - x)^2) : 
  x = (a^2 - b^2) / a :=
by
  sorry

end value_of_x_l98_98044


namespace minimalYellowFraction_l98_98296

-- Definitions
def totalSurfaceArea (sideLength : ℕ) : ℕ := 6 * (sideLength * sideLength)

def minimalYellowExposedArea : ℕ := 15

theorem minimalYellowFraction (sideLength : ℕ) (totalYellow : ℕ) (totalBlue : ℕ) 
    (totalCubes : ℕ) (yellowExposed : ℕ) :
    sideLength = 4 → totalYellow = 16 → totalBlue = 48 →
    totalCubes = 64 → yellowExposed = minimalYellowExposedArea →
    (yellowExposed / (totalSurfaceArea sideLength) : ℚ) = 5 / 32 :=
by
  sorry

end minimalYellowFraction_l98_98296


namespace total_weight_of_ripe_fruits_correct_l98_98125

-- Definitions based on conditions
def total_apples : ℕ := 14
def total_pears : ℕ := 10
def total_lemons : ℕ := 5

def ripe_apple_weight : ℕ := 150
def ripe_pear_weight : ℕ := 200
def ripe_lemon_weight : ℕ := 100

def unripe_apples : ℕ := 6
def unripe_pears : ℕ := 4
def unripe_lemons : ℕ := 2

def total_weight_of_ripe_fruits : ℕ :=
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight

theorem total_weight_of_ripe_fruits_correct :
  total_weight_of_ripe_fruits = 2700 :=
by
  -- proof goes here (use sorry to skip the actual proof)
  sorry

end total_weight_of_ripe_fruits_correct_l98_98125


namespace maximum_value_of_f_inequality_holds_for_all_x_l98_98396

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

theorem maximum_value_of_f (a : ℝ) (h : 0 ≤ a) : 
  (∀ x, f a x ≤ f a 1) → f a 1 = 3 / Real.exp 1 → a = 1 := 
by 
  sorry

theorem inequality_holds_for_all_x (b : ℝ) : 
  (∀ a ≤ 0, ∀ x, 0 ≤ x → f a x ≤ b * Real.log (x + 1)) → 1 ≤ b := 
by 
  sorry

end maximum_value_of_f_inequality_holds_for_all_x_l98_98396


namespace fish_caught_l98_98532

noncomputable def total_fish_caught (chris_trips : ℕ) (chris_fish_per_trip : ℕ) (brian_trips : ℕ) (brian_fish_per_trip : ℕ) : ℕ :=
  chris_trips * chris_fish_per_trip + brian_trips * brian_fish_per_trip

theorem fish_caught (chris_trips : ℕ) (brian_factor : ℕ) (brian_fish_per_trip : ℕ) (ratio_numerator : ℕ) (ratio_denominator : ℕ) :
  chris_trips = 10 → brian_factor = 2 → brian_fish_per_trip = 400 → ratio_numerator = 3 → ratio_denominator = 5 →
  total_fish_caught chris_trips (brian_fish_per_trip * ratio_denominator / ratio_numerator) (chris_trips * brian_factor) brian_fish_per_trip = 14660 :=
by
  intros h_chris_trips h_brian_factor h_brian_fish_per_trip h_ratio_numer h_ratio_denom
  rw [h_chris_trips, h_brian_factor, h_brian_fish_per_trip, h_ratio_numer, h_ratio_denom]
  -- adding actual arithmetic would resolve the statement correctly
  sorry

end fish_caught_l98_98532


namespace original_number_of_boys_l98_98931

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 40 = (n + 1) * 36) 
  : n = 4 :=
sorry

end original_number_of_boys_l98_98931


namespace solve_inequality_system_l98_98758

theorem solve_inequality_system (x : ℝ) 
  (h1 : 3 * x - 1 > x + 1) 
  (h2 : (4 * x - 5) / 3 ≤ x) 
  : 1 < x ∧ x ≤ 5 :=
by
  sorry

end solve_inequality_system_l98_98758


namespace number_of_sheep_l98_98619

theorem number_of_sheep (legs animals : ℕ) (h1 : legs = 60) (h2 : animals = 20)
  (chickens sheep : ℕ) (hc : chickens + sheep = animals) (hl : 2 * chickens + 4 * sheep = legs) :
  sheep = 10 :=
sorry

end number_of_sheep_l98_98619


namespace sum_xyz_eq_10_l98_98301

theorem sum_xyz_eq_10 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + 2 * x * y + 3 * x * y * z = 115) : 
  x + y + z = 10 :=
sorry

end sum_xyz_eq_10_l98_98301


namespace triangle_is_right_angle_l98_98854

theorem triangle_is_right_angle (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) : 
  a^2 + b^2 = c^2 :=
by 
  sorry

end triangle_is_right_angle_l98_98854


namespace length_of_second_square_l98_98644

-- Define conditions as variables
def Area_flag := 135
def Area_square1 := 40
def Area_square3 := 25

-- Define the length variable for the second square
variable (L : ℕ)

-- Define the area of the second square in terms of L
def Area_square2 : ℕ := 7 * L

-- Lean statement to be proved
theorem length_of_second_square :
  Area_square1 + Area_square2 L + Area_square3 = Area_flag → L = 10 :=
by sorry

end length_of_second_square_l98_98644


namespace value_of_a_l98_98871

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 5*a^2 - 6*a = 0 → 
    ∃ x₁ x₂, x₁ + x₂ = 4*a ∧ x₁ * x₂ = 5*a^2 - 6*a ∧ |x₁ - x₂| = 6)) → a = 3 :=
by {
  sorry
}

end value_of_a_l98_98871


namespace min_value_at_neg7_l98_98700

noncomputable def f (x : ℝ) : ℝ := x^2 + 14 * x + 24

theorem min_value_at_neg7 : ∀ x : ℝ, f (-7) ≤ f x :=
by
  sorry

end min_value_at_neg7_l98_98700


namespace find_px_l98_98093

theorem find_px (p : ℕ → ℚ) (h1 : p 1 = 1) (h2 : p 2 = 1 / 4) (h3 : p 3 = 1 / 9) 
  (h4 : p 4 = 1 / 16) (h5 : p 5 = 1 / 25) : p 6 = 1 / 18 :=
sorry

end find_px_l98_98093


namespace exponentiated_value_l98_98674

theorem exponentiated_value (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + b) = 24 := by
  sorry

end exponentiated_value_l98_98674


namespace sum_of_digits_of_m_l98_98289

theorem sum_of_digits_of_m (k m : ℕ) : 
  1 ≤ k ∧ k ≤ 3 ∧ 10000 ≤ 11131 * k + 1203 ∧ 11131 * k + 1203 < 100000 ∧ 
  11131 * k + 1203 = m * m ∧ 3 * k < 10 → 
  (m.digits 10).sum = 15 :=
by 
  sorry

end sum_of_digits_of_m_l98_98289


namespace diamond_value_l98_98392

def diamond (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem diamond_value : diamond 6 3 = 18 :=
by
  sorry

end diamond_value_l98_98392


namespace minimum_value_expression_l98_98994

theorem minimum_value_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 :=
by
  sorry

end minimum_value_expression_l98_98994


namespace max_value_of_function_for_x_lt_0_l98_98924

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem max_value_of_function_for_x_lt_0 :
  ∀ x : ℝ, x < 0 → f x ≤ -4 ∧ (∃ y : ℝ, f y = -4 ∧ y < 0) := sorry

end max_value_of_function_for_x_lt_0_l98_98924


namespace proof_problem_l98_98075

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}
def complement (s : Set ℕ) : Set ℕ := {x | x ∉ s}

theorem proof_problem : ((complement A ∪ A) ∪ B) = U :=
by sorry

end proof_problem_l98_98075


namespace complex_z_power_l98_98518

theorem complex_z_power:
  ∀ (z : ℂ), (z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) →
  z^1000 + (1/z)^1000 = 2 * Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_z_power_l98_98518


namespace budget_percentage_for_genetically_modified_organisms_l98_98182

theorem budget_percentage_for_genetically_modified_organisms
  (microphotonics : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (industrial_lubricants : ℝ)
  (astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 15 →
  industrial_lubricants = 8 →
  astrophysics_degrees = 72 →
  (72 / 360) * 100 = 20 →
  100 - (14 + 24 + 15 + 8 + 20) = 19 :=
  sorry

end budget_percentage_for_genetically_modified_organisms_l98_98182


namespace hyperbola_asymptote_slope_l98_98333

theorem hyperbola_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 100 - y^2 / 64 = 1) → y = (4/5) * x ∨ y = -(4/5) * x) :=
by
  sorry

end hyperbola_asymptote_slope_l98_98333


namespace digit_A_of_3AA1_divisible_by_9_l98_98857

theorem digit_A_of_3AA1_divisible_by_9 (A : ℕ) (h : (3 + A + A + 1) % 9 = 0) : A = 7 :=
sorry

end digit_A_of_3AA1_divisible_by_9_l98_98857


namespace ratio_of_typing_speeds_l98_98145

-- Defining Tim's and Tom's typing speeds
variables (T M : ℝ)

-- Conditions given in the problem
def condition1 : Prop := T + M = 15
def condition2 : Prop := T + 1.6 * M = 18

-- Conclusion to be proved: the ratio of M to T is 1:2
theorem ratio_of_typing_speeds (h1 : condition1 T M) (h2 : condition2 T M) :
  M / T = 1 / 2 :=
by
  -- skip the proof
  sorry

end ratio_of_typing_speeds_l98_98145


namespace rent_percentage_l98_98284

variable (E : ℝ)

def rent_last_year (E : ℝ) : ℝ := 0.20 * E 
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 0.25 * (earnings_this_year E)

-- Prove that the rent this year is 143.75% of the rent last year
theorem rent_percentage : (rent_this_year E) = 1.4375 * (rent_last_year E) :=
by
  sorry

end rent_percentage_l98_98284


namespace n_squared_plus_n_is_even_l98_98539

theorem n_squared_plus_n_is_even (n : ℤ) : Even (n^2 + n) :=
by
  sorry

end n_squared_plus_n_is_even_l98_98539


namespace b_not_six_iff_neg_two_not_in_range_l98_98040

def g (x b : ℝ) := x^3 + x^2 + b*x + 2

theorem b_not_six_iff_neg_two_not_in_range (b : ℝ) : 
  (∀ x : ℝ, g x b ≠ -2) ↔ b ≠ 6 :=
by
  sorry

end b_not_six_iff_neg_two_not_in_range_l98_98040


namespace sum_C_D_equals_seven_l98_98006

def initial_grid : Matrix (Fin 4) (Fin 4) (Option Nat) :=
  ![ ![ some 1, none, none, none ],
     ![ none, some 2, none, none ],
     ![ none, none, none, none ],
     ![ none, none, none, some 4 ] ]

def valid_grid (grid : Matrix (Fin 4) (Fin 4) (Option Nat)) : Prop :=
  ∀ i j, grid i j ≠ none →
    (∀ k, k ≠ j → grid i k ≠ grid i j) ∧ 
    (∀ k, k ≠ i → grid k j ≠ grid i j)

theorem sum_C_D_equals_seven :
  ∃ (C D : Nat), C + D = 7 ∧ valid_grid initial_grid :=
sorry

end sum_C_D_equals_seven_l98_98006


namespace mark_reads_1750_pages_per_week_l98_98675

def initialReadingHoursPerDay := 2
def increasePercentage := 150
def initialPagesPerDay := 100

def readingHoursPerDayAfterIncrease : Nat := initialReadingHoursPerDay + (initialReadingHoursPerDay * increasePercentage) / 100
def readingSpeedPerHour := initialPagesPerDay / initialReadingHoursPerDay
def pagesPerDayNow := readingHoursPerDayAfterIncrease * readingSpeedPerHour
def pagesPerWeekNow : Nat := pagesPerDayNow * 7

theorem mark_reads_1750_pages_per_week :
  pagesPerWeekNow = 1750 :=
sorry -- Proof omitted

end mark_reads_1750_pages_per_week_l98_98675


namespace inequality_solution_1_inequality_system_solution_2_l98_98788

theorem inequality_solution_1 (x : ℝ) : 
  (2 * x - 1) / 2 ≥ 1 - (x + 1) / 3 ↔ x ≥ 7 / 8 := 
sorry

theorem inequality_system_solution_2 (x : ℝ) : 
  (-2 * x ≤ -3) ∧ (x / 2 < 2) ↔ (3 / 2 ≤ x) ∧ (x < 4) :=
sorry

end inequality_solution_1_inequality_system_solution_2_l98_98788


namespace value_of_kaftan_l98_98479

theorem value_of_kaftan (K : ℝ) (h : (7 / 12) * (12 + K) = 5 + K) : K = 4.8 :=
by
  sorry

end value_of_kaftan_l98_98479


namespace difference_received_from_parents_l98_98049

-- Define conditions
def amount_from_mom := 8
def amount_from_dad := 5

-- Question: Prove the difference between amount_from_mom and amount_from_dad is 3
theorem difference_received_from_parents : (amount_from_mom - amount_from_dad) = 3 :=
by
  sorry

end difference_received_from_parents_l98_98049


namespace mean_score_is_74_l98_98514

theorem mean_score_is_74 (M SD : ℝ) 
  (h1 : 58 = M - 2 * SD) 
  (h2 : 98 = M + 3 * SD) : 
  M = 74 := 
by 
  -- problem statement without solving steps
  sorry

end mean_score_is_74_l98_98514


namespace triangle_min_perimeter_l98_98620

-- Definitions of points A, B, and C and the conditions specified in the problem.
def pointA : ℝ × ℝ := (3, 2)
def pointB (t : ℝ) : ℝ × ℝ := (t, t)
def pointC (c : ℝ) : ℝ × ℝ := (c, 0)

-- Main theorem which states that the minimum perimeter of triangle ABC is sqrt(26).
theorem triangle_min_perimeter : 
  ∃ (B C : ℝ × ℝ), B = pointB (B.1) ∧ C = pointC (C.1) ∧ 
  ∀ (B' C' : ℝ × ℝ), B' = pointB (B'.1) ∧ C' = pointC (C'.1) →
  (dist pointA B + dist B C + dist C pointA ≥ dist (2, 3) (3, -2)) :=
by 
  sorry

end triangle_min_perimeter_l98_98620


namespace range_of_a_sqrt10_e_bounds_l98_98441

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≤ g x) ↔ a ≤ 1 :=
by
  sorry

theorem sqrt10_e_bounds : 
  (1095 / 1000 : ℝ) < Real.exp (1/10 : ℝ) ∧ Real.exp (1/10 : ℝ) < (2000 / 1791 : ℝ) :=
by
  sorry

end range_of_a_sqrt10_e_bounds_l98_98441


namespace rounding_and_scientific_notation_l98_98153

-- Define the original number
def original_number : ℕ := 1694000

-- Define the function to round to the nearest hundred thousand
def round_to_nearest_hundred_thousand (n : ℕ) : ℕ :=
  ((n + 50000) / 100000) * 100000

-- Define the function to convert to scientific notation
def to_scientific_notation (n : ℕ) : String :=
  let base := n / 1000000
  let exponent := 6
  s!"{base}.0 × 10^{exponent}"

-- Assert the equivalence
theorem rounding_and_scientific_notation :
  to_scientific_notation (round_to_nearest_hundred_thousand original_number) = "1.7 × 10^{6}" :=
by
  sorry

end rounding_and_scientific_notation_l98_98153


namespace min_liars_in_presidium_l98_98292

-- Define the conditions of the problem
def liars_and_truthlovers (grid : ℕ → ℕ → Prop) : Prop :=
  ∃ n : ℕ, n = 32 ∧ 
  (∀ i j, i < 4 ∧ j < 8 → 
    (∃ ni nj, (ni = i + 1 ∨ ni = i - 1 ∨ ni = i ∨ nj = j + 1 ∨ nj = j - 1 ∨ nj = j) ∧
      (ni < 4 ∧ nj < 8) → (grid i j ↔ ¬ grid ni nj)))

-- Define proof problem
theorem min_liars_in_presidium (grid : ℕ → ℕ → Prop) :
  liars_and_truthlovers grid → (∃ l, l = 8) := by
  sorry

end min_liars_in_presidium_l98_98292


namespace mike_oranges_l98_98945

-- Definitions and conditions
variables (O A B : ℕ)
def condition1 := A = 2 * O
def condition2 := B = O + A
def condition3 := O + A + B = 18

-- Theorem to prove that Mike received 3 oranges
theorem mike_oranges (h1 : condition1 O A) (h2 : condition2 O A B) (h3 : condition3 O A B) : 
  O = 3 := 
by 
  sorry

end mike_oranges_l98_98945


namespace emily_total_points_l98_98365

def score_round_1 : ℤ := 16
def score_round_2 : ℤ := 33
def score_round_3 : ℤ := -25
def score_round_4 : ℤ := 46
def score_round_5 : ℤ := 12
def score_round_6 : ℤ := 30 - (2 * score_round_5 / 3)

def total_score : ℤ :=
  score_round_1 + score_round_2 + score_round_3 + score_round_4 + score_round_5 + score_round_6

theorem emily_total_points : total_score = 104 := by
  sorry

end emily_total_points_l98_98365


namespace both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l98_98349

variables (p1 p2 : Prop)

theorem both_shots_hit (p1 p2 : Prop) : (p1 ∧ p2) ↔ (p1 ∧ p2) :=
by sorry

theorem both_shots_missed (p1 p2 : Prop) : (¬p1 ∧ ¬p2) ↔ (¬p1 ∧ ¬p2) :=
by sorry

theorem exactly_one_shot_hit (p1 p2 : Prop) : ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) ↔ ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) :=
by sorry

theorem at_least_one_shot_hit (p1 p2 : Prop) : (p1 ∨ p2) ↔ (p1 ∨ p2) :=
by sorry

end both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l98_98349


namespace exists_sum_of_150_consecutive_integers_l98_98783

theorem exists_sum_of_150_consecutive_integers :
  ∃ a : ℕ, 1627395075 = 150 * a + 11175 :=
by
  sorry

end exists_sum_of_150_consecutive_integers_l98_98783


namespace number_of_students_l98_98941

-- Define parameters and conditions
variables (B G : ℕ) -- number of boys and girls

-- Condition: each boy is friends with exactly two girls
axiom boys_to_girls : ∀ (B G : ℕ), 2 * B = 3 * G

-- Condition: total number of children in the class
axiom total_children : ∀ (B G : ℕ), B + G = 31

-- Define the theorem that proves the correct number of students
theorem number_of_students : (B G : ℕ) → 2 * B = 3 * G → B + G = 31 → B + G = 35 :=
by
  sorry

end number_of_students_l98_98941


namespace solve_for_x_l98_98270

theorem solve_for_x (x : ℕ) : 100^3 = 10^x → x = 6 := by
  sorry

end solve_for_x_l98_98270


namespace right_triangle_perimeter_l98_98942

-- Given conditions
variable (x y : ℕ)
def leg1 := 11
def right_triangle := (101 * 11 = 121)

-- The question and answer
theorem right_triangle_perimeter :
  (y + x = 121) ∧ (y - x = 1) → (11 + x + y = 132) :=
by
  sorry

end right_triangle_perimeter_l98_98942


namespace calc1_calc2_calc3_l98_98999

theorem calc1 : 1 - 2 + 3 - 4 + 5 = 3 := by sorry
theorem calc2 : - (4 / 7) / (8 / 49) = - (7 / 2) := by sorry
theorem calc3 : ((1 / 2) - (3 / 5) + (2 / 3)) * (-15) = - (17 / 2) := by sorry

end calc1_calc2_calc3_l98_98999


namespace B_cycling_speed_l98_98178

/--
A walks at 10 kmph. 10 hours after A starts, B cycles after him at a certain speed.
B catches up with A at a distance of 200 km from the start. Prove that B's cycling speed is 20 kmph.
-/
theorem B_cycling_speed (speed_A : ℝ) (time_A_to_start_B : ℝ) 
  (distance_at_catch : ℝ) (B_speed : ℝ)
  (h1 : speed_A = 10) 
  (h2 : time_A_to_start_B = 10)
  (h3 : distance_at_catch = 200)
  (h4 : distance_at_catch = speed_A * time_A_to_start_B + speed_A * (distance_at_catch / speed_B)) :
    B_speed = 20 := by
  sorry

end B_cycling_speed_l98_98178


namespace circle_tangent_parabola_height_difference_l98_98397

theorem circle_tangent_parabola_height_difference
  (a b r : ℝ)
  (point_of_tangency_left : a ≠ 0)
  (points_of_tangency_on_parabola : (2 * a^2) = (2 * (-a)^2))
  (center_y_coordinate : ∃ c , c = b)
  (circle_equation_tangent_parabola : ∀ x, (x^2 + (2*x^2 - b)^2 = r^2))
  (quartic_double_root : ∀ x, (x = a ∨ x = -a) → (x^2 + (4 - 2*b)*x^2 + b^2 - r^2 = 0)) :
  b - 2 * a^2 = 2 :=
by
  sorry

end circle_tangent_parabola_height_difference_l98_98397


namespace calculate_gallons_of_milk_l98_98865

-- Definitions of the given constants and conditions
def price_of_soup : Nat := 2
def price_of_bread : Nat := 5
def price_of_cereal : Nat := 3
def price_of_milk : Nat := 4
def total_amount_paid : Nat := 4 * 10

-- Calculation of total cost of non-milk items
def total_cost_non_milk : Nat :=
  (6 * price_of_soup) + (2 * price_of_bread) + (2 * price_of_cereal)

-- The function to calculate the remaining amount to be spent on milk
def remaining_amount : Nat := total_amount_paid - total_cost_non_milk

-- Statement to compute the number of gallons of milk
def gallons_of_milk (remaining : Nat) (price_per_gallon : Nat) : Nat :=
  remaining / price_per_gallon

-- Proof theorem statement (no implementation required, proof skipped)
theorem calculate_gallons_of_milk : 
  gallons_of_milk remaining_amount price_of_milk = 3 := 
by
  sorry

end calculate_gallons_of_milk_l98_98865


namespace lollipop_count_l98_98109

theorem lollipop_count (total_cost one_lollipop_cost : ℚ) (h1 : total_cost = 90) (h2 : one_lollipop_cost = 0.75) : total_cost / one_lollipop_cost = 120 :=
by
  sorry

end lollipop_count_l98_98109


namespace polar_to_rectangular_inequality_range_l98_98493

-- Part A: Transforming a polar coordinate equation to a rectangular coordinate equation
theorem polar_to_rectangular (ρ θ : ℝ) : 
  (ρ^2 * Real.cos θ - ρ = 0) ↔ ((ρ = 0 ∧ 0 = 1) ∨ (ρ ≠ 0 ∧ Real.cos θ = 1 / ρ)) := 
sorry

-- Part B: Determining range for an inequality
theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2-x| + |x+1| ≤ a) ↔ (a ≥ 9) := 
sorry

end polar_to_rectangular_inequality_range_l98_98493


namespace quadratic_value_at_sum_of_roots_is_five_l98_98016

noncomputable def quadratic_func (a b x : ℝ) : ℝ := a * x^2 + b * x + 5

theorem quadratic_value_at_sum_of_roots_is_five
  (a b x₁ x₂ : ℝ)
  (hA : quadratic_func a b x₁ = 2023)
  (hB : quadratic_func a b x₂ = 2023)
  (ha : a ≠ 0) :
  quadratic_func a b (x₁ + x₂) = 5 :=
sorry

end quadratic_value_at_sum_of_roots_is_five_l98_98016


namespace polygon_sides_l98_98426

-- Define the given condition formally
def sum_of_internal_and_external_angle (n : ℕ) : ℕ :=
  (n - 2) * 180 + (1) -- This represents the sum of internal angles plus an external angle

theorem polygon_sides (n : ℕ) : 
  sum_of_internal_and_external_angle n = 1350 → n = 9 :=
by
  sorry

end polygon_sides_l98_98426


namespace number_of_guest_cars_l98_98127

-- Definitions and conditions
def total_wheels : ℕ := 48
def mother_car_wheels : ℕ := 4
def father_jeep_wheels : ℕ := 4
def wheels_per_car : ℕ := 4

-- Theorem statement
theorem number_of_guest_cars (total_wheels mother_car_wheels father_jeep_wheels wheels_per_car : ℕ) : ℕ :=
  (total_wheels - (mother_car_wheels + father_jeep_wheels)) / wheels_per_car

-- Specific instance for the problem
example : number_of_guest_cars 48 4 4 4 = 10 := 
by
  sorry

end number_of_guest_cars_l98_98127


namespace log_ordering_l98_98762

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_ordering (P Q R : ℝ) (h₁ : P = Real.log 3 / Real.log 2)
  (h₂ : Q = Real.log 2 / Real.log 3) (h₃ : R = Real.log (Real.log 2 / Real.log 3) / Real.log 2) :
  R < Q ∧ Q < P := by
  sorry

end log_ordering_l98_98762


namespace find_larger_integer_l98_98057

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l98_98057


namespace determine_k_l98_98148

theorem determine_k (S : ℕ → ℝ) (k : ℝ)
  (hSn : ∀ n, S n = k + 2 * (1 / 3)^n)
  (a1 : ℝ := S 1)
  (a2 : ℝ := S 2 - S 1)
  (a3 : ℝ := S 3 - S 2)
  (geom_property : a2^2 = a1 * a3) :
  k = -2 := 
by
  sorry

end determine_k_l98_98148


namespace determine_guilty_defendant_l98_98213

-- Define the defendants
inductive Defendant
| A
| B
| C

open Defendant

-- Define the guilty defendant
def guilty_defendant : Defendant := C

-- Define the conditions
def condition1 (d : Defendant) : Prop :=
d ≠ A ∧ d ≠ B ∧ d ≠ C → false  -- "There were three defendants, and only one of them was guilty."

def condition2 (d : Defendant) : Prop :=
d = A → d ≠ B  -- "Defendant A accused defendant B."

def condition3 (d : Defendant) : Prop :=
d = B → d = B  -- "Defendant B admitted to being guilty."

def condition4 (d : Defendant) : Prop :=
d = C → (d = C ∨ d = A)  -- "Defendant C either admitted to being guilty or accused A."

-- The proof problem statement
theorem determine_guilty_defendant :
  (∃ d : Defendant, condition1 d ∧ condition2 d ∧ condition3 d ∧ condition4 d) → guilty_defendant = C :=
by {
  sorry
}

end determine_guilty_defendant_l98_98213


namespace correct_choice_C_l98_98360

def geometric_sequence (n : ℕ) : ℕ := 
  2^(n - 1)

def sum_geometric_sequence (n : ℕ) : ℕ := 
  2^n - 1

theorem correct_choice_C (n : ℕ) (h : 0 < n) : sum_geometric_sequence n < geometric_sequence (n + 1) := by
  sorry

end correct_choice_C_l98_98360


namespace option_c_opp_numbers_l98_98623

theorem option_c_opp_numbers : (- (2 ^ 2)) = - ((-2) ^ 2) :=
by
  sorry

end option_c_opp_numbers_l98_98623


namespace arithmetic_sequence_sum_l98_98105

theorem arithmetic_sequence_sum (a₁ d S : ℤ)
  (ha : 10 * a₁ + 24 * d = 37) :
  19 * (a₁ + 2 * d) + (a₁ + 10 * d) = 74 :=
by
  sorry

end arithmetic_sequence_sum_l98_98105


namespace find_savings_l98_98774

theorem find_savings (income expenditure : ℕ) (ratio_income_expenditure : ℕ × ℕ) (income_value : income = 40000)
    (ratio_condition : ratio_income_expenditure = (8, 7)) :
    income - expenditure = 5000 :=
by
  sorry

end find_savings_l98_98774


namespace range_of_a_l98_98416

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then x^2 + 2 * a else -x

theorem range_of_a (a : ℝ) (h : a < 0) (hf : f a (1 - a) ≥ f a (1 + a)) : -2 ≤ a ∧ a ≤ -1 :=
  sorry

end range_of_a_l98_98416


namespace solve_quadratics_and_sum_l98_98587

theorem solve_quadratics_and_sum (d e f : ℤ) 
  (h1 : ∃ d e : ℤ, d + e = 19 ∧ d * e = 88) 
  (h2 : ∃ e f : ℤ, e + f = 23 ∧ e * f = 120) : 
  d + e + f = 31 := by
  sorry

end solve_quadratics_and_sum_l98_98587


namespace characteristic_triangle_smallest_angle_l98_98816

theorem characteristic_triangle_smallest_angle 
  (α β : ℝ)
  (h1 : α = 2 * β)
  (h2 : α = 100)
  (h3 : β + α + γ = 180) : 
  min α (min β γ) = 30 := 
by 
  sorry

end characteristic_triangle_smallest_angle_l98_98816


namespace determine_a_b_l98_98746

-- Define the polynomial expression
def poly (x a b : ℝ) : ℝ := x^2 + a * x + b

-- Define the factored form
def factored_poly (x : ℝ) : ℝ := (x + 1) * (x - 3)

-- State the theorem
theorem determine_a_b (a b : ℝ) (h : ∀ x, poly x a b = factored_poly x) : a = -2 ∧ b = -3 :=
by 
  sorry

end determine_a_b_l98_98746


namespace find_principal_amount_l98_98818

theorem find_principal_amount (P r : ℝ) 
    (h1 : 815 - P = P * r * 3) 
    (h2 : 850 - P = P * r * 4) : 
    P = 710 :=
by
  -- proof steps will go here
  sorry

end find_principal_amount_l98_98818


namespace perimeter_of_regular_polygon_l98_98054

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l98_98054


namespace evalCeilingOfNegativeSqrt_l98_98958

noncomputable def ceiling_of_negative_sqrt : ℤ :=
  Int.ceil (-(Real.sqrt (36 / 9)))

theorem evalCeilingOfNegativeSqrt : ceiling_of_negative_sqrt = -2 := by
  sorry

end evalCeilingOfNegativeSqrt_l98_98958


namespace balls_drawn_ensure_single_color_ge_20_l98_98794

theorem balls_drawn_ensure_single_color_ge_20 (r g y b w bl : ℕ) (h_r : r = 34) (h_g : g = 28) (h_y : y = 23) (h_b : b = 18) (h_w : w = 12) (h_bl : bl = 11) : 
  ∃ (n : ℕ), n ≥ 20 →
    (r + g + y + b + w + bl - n) + 1 > 20 :=
by
  sorry

end balls_drawn_ensure_single_color_ge_20_l98_98794


namespace garageHasWheels_l98_98209

-- Define the conditions
def bikeWheelsPerBike : Nat := 2
def bikesInGarage : Nat := 10

-- State the theorem to be proved
theorem garageHasWheels : bikesInGarage * bikeWheelsPerBike = 20 := by
  sorry

end garageHasWheels_l98_98209


namespace percentage_decrease_correct_l98_98020

variable (O N : ℕ)
variable (percentage_decrease : ℕ)

-- Define the conditions based on the problem
def original_price := 1240
def new_price := 620
def price_effect := ((original_price - new_price) * 100) / original_price

-- Prove the percentage decrease is 50%
theorem percentage_decrease_correct :
  price_effect = 50 := by
  sorry

end percentage_decrease_correct_l98_98020


namespace range_of_b_l98_98498

theorem range_of_b (b : ℝ) : (∃ x : ℝ, |x - 2| + |x - 5| < b) → b > 3 :=
by 
-- This is where the proof would go.
sorry

end range_of_b_l98_98498


namespace mean_and_variance_l98_98087

def scores_A : List ℝ := [8, 9, 14, 15, 15, 16, 21, 22]
def scores_B : List ℝ := [7, 8, 13, 15, 15, 17, 22, 23]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)
noncomputable def variance (l : List ℝ) : ℝ := mean (l.map (λ x => (x - (mean l)) ^ 2))

theorem mean_and_variance :
  (mean scores_A = mean scores_B) ∧ (variance scores_A < variance scores_B) :=
by
  sorry

end mean_and_variance_l98_98087


namespace four_minus_x_is_five_l98_98959

theorem four_minus_x_is_five (x y : ℤ) (h1 : 4 + x = 5 - y) (h2 : 3 + y = 6 + x) : 4 - x = 5 := by
sorry

end four_minus_x_is_five_l98_98959


namespace ratio_of_areas_l98_98522
-- Define the conditions and the ratio to be proven
theorem ratio_of_areas (t r : ℝ) (h : 3 * t = 2 * π * r) : 
  (π^2 / 18) = (π^2 * r^2 / 9) / (2 * r^2) :=
by 
  sorry

end ratio_of_areas_l98_98522


namespace perp_line_eq_l98_98634

theorem perp_line_eq (x y : ℝ) (c : ℝ) (hx : x = 1) (hy : y = 2) (hline : 2 * x + y - 5 = 0) :
  x - 2 * y + c = 0 ↔ c = 3 := 
by
  sorry

end perp_line_eq_l98_98634


namespace min_xy_min_a_b_l98_98377

-- Problem 1 Lean Statement
theorem min_xy {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / (4 * y) = 1) : xy ≥ 2 := sorry

-- Problem 2 Lean Statement
theorem min_a_b {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : ab = a + 2 * b + 4) : a + b ≥ 3 + 2 * Real.sqrt 6 := sorry

end min_xy_min_a_b_l98_98377


namespace program_exists_l98_98077
open Function

-- Define the chessboard and labyrinth
namespace ChessMaze

structure Position :=
  (row : Nat)
  (col : Nat)
  (h_row : row < 8)
  (h_col : col < 8)

inductive Command
| RIGHT | LEFT | UP | DOWN

structure Labyrinth :=
  (barriers : Position → Position → Bool) -- True if there's a barrier between the two positions

def accessible (L : Labyrinth) (start : Position) (cmd : List Command) : Set Position :=
  -- The set of positions accessible after applying the commands from start in labyrinth L
  sorry

-- The main theorem we want to prove
theorem program_exists : 
  ∃ (cmd : List Command), ∀ (L : Labyrinth) (start : Position), ∀ pos ∈ accessible L start cmd, ∃ p : Position, p = pos :=
  sorry

end ChessMaze

end program_exists_l98_98077


namespace g_of_five_eq_one_l98_98265

variable (g : ℝ → ℝ)

theorem g_of_five_eq_one (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
    (h2 : ∀ x : ℝ, g x ≠ 0) : g 5 = 1 :=
sorry

end g_of_five_eq_one_l98_98265


namespace log_value_between_integers_l98_98947

theorem log_value_between_integers : (1 : ℤ) < Real.log 25 / Real.log 10 ∧ Real.log 25 / Real.log 10 < (2 : ℤ) → 1 + 2 = 3 :=
by
  sorry

end log_value_between_integers_l98_98947


namespace sinA_mul_sinC_eq_three_fourths_l98_98011
open Real

-- Definitions based on conditions
def angles_form_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

def sides_form_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- The theorem to prove
theorem sinA_mul_sinC_eq_three_fourths
  (A B C a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_angles : A + B + C = π)
  (h_angles_arithmetic : angles_form_arithmetic_sequence A B C)
  (h_sides_geometric : sides_form_geometric_sequence a b c) :
  sin A * sin C = 3 / 4 :=
sorry

end sinA_mul_sinC_eq_three_fourths_l98_98011


namespace age_ratio_l98_98576

-- Definitions as per the conditions
variable (j e x : ℕ)

-- Conditions from the problem
def condition1 : Prop := j - 4 = 2 * (e - 4)
def condition2 : Prop := j - 10 = 3 * (e - 10)

-- The statement we need to prove
theorem age_ratio (j e x : ℕ) (h1 : condition1 j e)
(h2 : condition2 j e) :
(j + x) * 2 = (e + x) * 3 ↔ x = 8 :=
sorry

end age_ratio_l98_98576


namespace angle_division_quadrant_l98_98024

variable (k : ℤ)
variable (α : ℝ)
variable (h : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi)

theorem angle_division_quadrant 
  (hα_sec_quadrant : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi) : 
  (∃ m : ℤ, (m = 0 ∧ Real.pi / 4 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
             (m = 1 ∧ Real.pi * (2 * m + 1) / 4 < α / 2 ∧ α / 2 < Real.pi * (2 * m + 1) / 2)) :=
sorry

end angle_division_quadrant_l98_98024


namespace dogwood_trees_current_l98_98570

variable (X : ℕ)
variable (trees_today : ℕ := 41)
variable (trees_tomorrow : ℕ := 20)
variable (total_trees_after : ℕ := 100)

theorem dogwood_trees_current (h : X + trees_today + trees_tomorrow = total_trees_after) : X = 39 :=
by
  sorry

end dogwood_trees_current_l98_98570


namespace planA_text_message_cost_l98_98173

def planA_cost (x : ℝ) : ℝ := 60 * x + 9
def planB_cost : ℝ := 60 * 0.40

theorem planA_text_message_cost (x : ℝ) (h : planA_cost x = planB_cost) : x = 0.25 :=
by
  -- h represents the condition that the costs are equal
  -- The proof is skipped with sorry
  sorry

end planA_text_message_cost_l98_98173


namespace impossible_to_form_triangle_l98_98404

theorem impossible_to_form_triangle 
  (a b c : ℝ)
  (h1 : a = 9) 
  (h2 : b = 4) 
  (h3 : c = 3) 
  : ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  rw [h1, h2, h3]
  simp
  sorry

end impossible_to_form_triangle_l98_98404


namespace range_of_b_l98_98050

noncomputable def f (x a b : ℝ) := (x - a)^2 * (x + b) * Real.exp x

theorem range_of_b (a b : ℝ) (h_max : ∃ δ > 0, ∀ x, |x - a| < δ → f x a b ≤ f a a b) : b < -a := sorry

end range_of_b_l98_98050


namespace forces_angle_result_l98_98681

noncomputable def forces_angle_condition (p1 p2 p : ℝ) (α : ℝ) : Prop :=
  p^2 = p1 * p2

noncomputable def angle_condition_range (p1 p2 : ℝ) : Prop :=
  (3 - Real.sqrt 5) / 2 ≤ p1 / p2 ∧ p1 / p2 ≤ (3 + Real.sqrt 5) / 2

theorem forces_angle_result (p1 p2 p α : ℝ) (h : forces_angle_condition p1 p2 p α) :
  120 * π / 180 ≤ α ∧ α ≤ 120 * π / 180 ∧ (angle_condition_range p1 p2) := 
sorry

end forces_angle_result_l98_98681


namespace gcd_7_fact_10_fact_div_4_fact_eq_5040_l98_98504

def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

noncomputable def quotient_fact (a b : ℕ) : ℕ := fact a / fact b

theorem gcd_7_fact_10_fact_div_4_fact_eq_5040 :
  Nat.gcd (fact 7) (quotient_fact 10 4) = 5040 := by
sorry

end gcd_7_fact_10_fact_div_4_fact_eq_5040_l98_98504


namespace algebraic_expression_value_l98_98663

def algebraic_expression (a b : ℤ) :=
  a + 2 * b + 2 * (a + 2 * b) + 1

theorem algebraic_expression_value :
  algebraic_expression 1 (-1) = -2 :=
by
  -- Proof skipped
  sorry

end algebraic_expression_value_l98_98663


namespace luke_total_points_l98_98137

/-- Luke gained 327 points in each round of a trivia game. 
    He played 193 rounds of the game. 
    How many points did he score in total? -/
theorem luke_total_points : 327 * 193 = 63111 :=
by
  sorry

end luke_total_points_l98_98137


namespace count_integers_in_range_l98_98527

theorem count_integers_in_range : 
  let lower_bound := -2.8
  let upper_bound := Real.pi
  let in_range (x : ℤ) := (lower_bound : ℝ) < (x : ℝ) ∧ (x : ℝ) ≤ upper_bound
  (Finset.filter in_range (Finset.Icc (Int.floor lower_bound) (Int.floor upper_bound))).card = 6 :=
by
  sorry

end count_integers_in_range_l98_98527


namespace eval_expression_l98_98580

theorem eval_expression : (2^5 - 5^2) = 7 :=
by {
  -- Proof steps will be here
  sorry
}

end eval_expression_l98_98580


namespace proof_problem_l98_98371

noncomputable def f (x : ℝ) : ℝ :=
  Real.log ((1 + Real.sqrt x) / (1 - Real.sqrt x))

theorem proof_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) :
  f ( (5 * x + 2 * x^2) / (1 + 5 * x + 3 * x^2) ) = Real.sqrt 5 * f x :=
by
  sorry

end proof_problem_l98_98371


namespace lincoln_county_houses_l98_98275

theorem lincoln_county_houses (original_houses : ℕ) (built_houses : ℕ) (total_houses : ℕ) 
(h1 : original_houses = 20817) 
(h2 : built_houses = 97741) 
(h3 : total_houses = original_houses + built_houses) : 
total_houses = 118558 :=
by
  -- proof omitted
  sorry

end lincoln_county_houses_l98_98275


namespace find_coordinates_l98_98884

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def satisfiesCondition (A B P : Point) : Prop :=
  2 * (P.x - A.x) = (B.x - P.x) ∧ 2 * (P.y - A.y) = (B.y - P.y)

theorem find_coordinates (P : Point) (h : satisfiesCondition A B P) : 
  P = ⟨6, -9⟩ :=
  sorry

end find_coordinates_l98_98884


namespace number_of_terms_added_l98_98950

theorem number_of_terms_added (k : ℕ) (h : 1 ≤ k) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k :=
by sorry

end number_of_terms_added_l98_98950


namespace older_sister_age_l98_98714

theorem older_sister_age (x : ℕ) (older_sister_age : ℕ) (h1 : older_sister_age = 3 * x)
  (h2 : older_sister_age + 2 = 2 * (x + 2)) : older_sister_age = 6 :=
by
  sorry

end older_sister_age_l98_98714


namespace find_f_of_monotonic_and_condition_l98_98082

noncomputable def monotonic (f : ℝ → ℝ) :=
  ∀ {a b : ℝ}, a < b → f a ≤ f b

theorem find_f_of_monotonic_and_condition (f : ℝ → ℝ) (h_mono : monotonic f) (h_cond : ∀ x : ℝ, 0 < x → f (f x - x^2) = 6) : f 2 = 6 :=
by
  sorry

end find_f_of_monotonic_and_condition_l98_98082


namespace find_number_l98_98488

noncomputable def number_divided_by_seven_is_five_fourteen (x : ℝ) : Prop :=
  x / 7 = 5 / 14

theorem find_number (x : ℝ) (h : number_divided_by_seven_is_five_fourteen x) : x = 2.5 :=
by
  sorry

end find_number_l98_98488


namespace transformation_correctness_l98_98129

theorem transformation_correctness :
  (∀ x : ℝ, 3 * x = -4 → x = -4 / 3) ∧
  (∀ x : ℝ, 5 = 2 - x → x = -3) ∧
  (∀ x : ℝ, (x - 1) / 6 - (2 * x + 3) / 8 = 1 → 4 * (x - 1) - 3 * (2 * x + 3) = 24) ∧
  (∀ x : ℝ, 3 * x - (2 - 4 * x) = 5 → 3 * x + 4 * x - 2 = 5) :=
by
  -- Prove the given conditions
  sorry

end transformation_correctness_l98_98129


namespace line_AB_equation_l98_98970

theorem line_AB_equation (m : ℝ) (A B : ℝ × ℝ)
  (hA : A = (0, 0)) (hA_line : ∀ (x y : ℝ), A = (x, y) → x + m * y = 0)
  (hB : B = (1, 3)) (hB_line : ∀ (x y : ℝ), B = (x, y) → m * x - y - m + 3 = 0) :
  ∃ (a b c : ℝ), a * 1 - b * 3 + c = 0 ∧ a * x + b * y + c * 0 = 0 ∧ 3 * x - y + 0 = 0 :=
by
  sorry

end line_AB_equation_l98_98970


namespace power_function_value_l98_98852

theorem power_function_value
  (α : ℝ)
  (h : 2^α = Real.sqrt 2) :
  (4 : ℝ) ^ α = 2 :=
by {
  sorry
}

end power_function_value_l98_98852


namespace avg_of_eleven_numbers_l98_98039

variable (S1 : ℕ)
variable (S2 : ℕ)
variable (sixth_num : ℕ)
variable (total_sum : ℕ)
variable (avg_eleven : ℕ)

def condition1 := S1 = 6 * 58
def condition2 := S2 = 6 * 65
def condition3 := sixth_num = 188
def condition4 := total_sum = S1 + S2 - sixth_num
def condition5 := avg_eleven = total_sum / 11

theorem avg_of_eleven_numbers : (S1 = 6 * 58) →
                                (S2 = 6 * 65) →
                                (sixth_num = 188) →
                                (total_sum = S1 + S2 - sixth_num) →
                                (avg_eleven = total_sum / 11) →
                                avg_eleven = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end avg_of_eleven_numbers_l98_98039


namespace product_of_variables_l98_98737

theorem product_of_variables (a b c d : ℚ)
  (h1 : 4 * a + 5 * b + 7 * c + 9 * d = 56)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d) :
  a * b * c * d = 58653 / 10716361 := 
sorry

end product_of_variables_l98_98737


namespace probability_of_one_black_ball_l98_98969

theorem probability_of_one_black_ball (total_balls black_balls white_balls drawn_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_black : black_balls = 2)
  (h_white : white_balls = 2)
  (h_drawn : drawn_balls = 2) :
  ((Nat.choose black_balls 1) * (Nat.choose white_balls 1) : ℚ) / (Nat.choose total_balls drawn_balls) = 2 / 3 :=
by {
  -- Insert proof here
  sorry
}

end probability_of_one_black_ball_l98_98969


namespace perimeter_difference_zero_l98_98657

theorem perimeter_difference_zero :
  let shape1_length := 4
  let shape1_width := 3
  let shape2_length := 6
  let shape2_width := 1
  let perimeter (l w : ℕ) := 2 * (l + w)
  perimeter shape1_length shape1_width = perimeter shape2_length shape2_width :=
by
  sorry

end perimeter_difference_zero_l98_98657


namespace max_bars_scenario_a_max_bars_scenario_b_l98_98948

-- Define the game conditions and the maximum bars Ivan can take in each scenario.

def max_bars_taken (initial_bars : ℕ) : ℕ :=
  if initial_bars = 14 then 13 else 13

theorem max_bars_scenario_a :
  max_bars_taken 13 = 13 :=
by sorry

theorem max_bars_scenario_b :
  max_bars_taken 14 = 13 :=
by sorry

end max_bars_scenario_a_max_bars_scenario_b_l98_98948


namespace geometric_sequence_sum_l98_98088

theorem geometric_sequence_sum :
  let a := (1/2 : ℚ)
  let r := (1/3 : ℚ)
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 243 :=
by
  sorry

end geometric_sequence_sum_l98_98088


namespace product_of_roots_is_12_l98_98489

theorem product_of_roots_is_12 :
  (81 ^ (1 / 4) * 8 ^ (1 / 3) * 4 ^ (1 / 2)) = 12 := by
  sorry

end product_of_roots_is_12_l98_98489


namespace option_b_correct_l98_98921

theorem option_b_correct (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3: a ≠ 1) (h4: b ≠ 1) (h5 : 0 < m) (h6 : m < 1) :
  m^a < m^b :=
sorry

end option_b_correct_l98_98921


namespace course_selection_l98_98269

noncomputable def number_of_ways (nA nB : ℕ) : ℕ :=
  (Nat.choose nA 2) * (Nat.choose nB 1) + (Nat.choose nA 1) * (Nat.choose nB 2)

theorem course_selection :
  (number_of_ways 3 4) = 30 :=
by
  sorry

end course_selection_l98_98269


namespace find_k_l98_98718

theorem find_k (k : ℝ) 
  (h1 : ∀ (r s : ℝ), r + s = -k ∧ r * s = 8 → (r + 3) + (s + 3) = k) : 
  k = 3 :=
by
  sorry

end find_k_l98_98718


namespace simplify_and_evaluate_l98_98351

noncomputable def expression (x : ℤ) : ℤ :=
  ( (-2 * x^3 - 6 * x) / (-2 * x) - 2 * (3 * x + 1) * (3 * x - 1) + 7 * x * (x - 1) )

theorem simplify_and_evaluate : 
  (expression (-3) = -64) := by
  sorry

end simplify_and_evaluate_l98_98351


namespace diane_faster_than_rhonda_l98_98897

theorem diane_faster_than_rhonda :
  ∀ (rhonda_time sally_time diane_time total_time : ℕ), 
  rhonda_time = 24 →
  sally_time = rhonda_time + 2 →
  total_time = 71 →
  total_time = rhonda_time + sally_time + diane_time →
  (rhonda_time - diane_time) = 3 :=
by
  intros rhonda_time sally_time diane_time total_time
  intros h_rhonda h_sally h_total h_sum
  sorry

end diane_faster_than_rhonda_l98_98897


namespace percentage_of_boys_l98_98435

theorem percentage_of_boys (total_students boys_per_group girls_per_group : ℕ)
  (ratio_condition : boys_per_group + girls_per_group = 7)
  (total_condition : total_students = 42)
  (ratio_b_condition : boys_per_group = 3)
  (ratio_g_condition : girls_per_group = 4) :
  (boys_per_group : ℚ) / (boys_per_group + girls_per_group : ℚ) * 100 = 42.86 :=
by sorry

end percentage_of_boys_l98_98435


namespace apps_addition_vs_deletion_l98_98205

-- Defining the initial conditions
def initial_apps : ℕ := 21
def added_apps : ℕ := 89
def remaining_apps : ℕ := 24

-- The proof problem statement
theorem apps_addition_vs_deletion :
  added_apps - (initial_apps + added_apps - remaining_apps) = 3 :=
by
  sorry

end apps_addition_vs_deletion_l98_98205


namespace solve_for_t_l98_98100

variable (S₁ S₂ u t : ℝ)

theorem solve_for_t 
  (h₀ : u ≠ 0) 
  (h₁ : u = (S₁ - S₂) / (t - 1)) :
  t = (S₁ - S₂ + u) / u :=
by
  sorry

end solve_for_t_l98_98100


namespace series_evaluation_l98_98750

noncomputable def series_sum : ℝ :=
  ∑' m : ℕ, (∑' n : ℕ, (m^2 * n) / (3^m * (n * 3^m + m * 3^n)))

theorem series_evaluation : series_sum = 9 / 32 :=
by
  sorry

end series_evaluation_l98_98750


namespace solution_set_of_f_greater_than_one_l98_98326

theorem solution_set_of_f_greater_than_one (f : ℝ → ℝ) (h_inv : ∀ x, f (x / (x + 3)) = x) :
  {x | f x > 1} = {x | 1 / 4 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_f_greater_than_one_l98_98326


namespace valid_values_l98_98701

noncomputable def is_defined (x : ℝ) : Prop := 
  (x^2 - 4*x + 3 > 0) ∧ (5 - x^2 > 0)

theorem valid_values (x : ℝ) : 
  is_defined x ↔ (-Real.sqrt 5 < x ∧ x < 1) ∨ (3 < x ∧ x < Real.sqrt 5) := by
  sorry

end valid_values_l98_98701


namespace evaluate_polynomial_l98_98281

theorem evaluate_polynomial : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_l98_98281


namespace value_of_expression_l98_98622

theorem value_of_expression (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) : 
  3 * a + 2 = (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end value_of_expression_l98_98622


namespace percentage_singing_l98_98394

def total_rehearsal_time : ℕ := 75
def warmup_time : ℕ := 6
def notes_time : ℕ := 30
def words_time (t : ℕ) : ℕ := t
def singing_time (t : ℕ) : ℕ := total_rehearsal_time - warmup_time - notes_time - words_time t
def singing_percentage (t : ℕ) : ℕ := (singing_time t * 100) / total_rehearsal_time

theorem percentage_singing (t : ℕ) : (singing_percentage t) = (4 * (39 - t)) / 3 :=
by
  sorry

end percentage_singing_l98_98394


namespace brother_paint_time_is_4_l98_98824

noncomputable def brother_paint_time (B : ℝ) : Prop :=
  (1 / 3) + (1 / B) = 1 / 1.714

theorem brother_paint_time_is_4 : ∃ B, brother_paint_time B ∧ abs (B - 4) < 0.001 :=
by {
  sorry -- Proof to be filled in later.
}

end brother_paint_time_is_4_l98_98824


namespace factorial_plus_one_div_prime_l98_98424

theorem factorial_plus_one_div_prime (n : ℕ) (h : (n! + 1) % (n + 1) = 0) : Nat.Prime (n + 1) := 
sorry

end factorial_plus_one_div_prime_l98_98424


namespace inequality_solution_set_l98_98660

theorem inequality_solution_set (x : ℝ) : 3 ≤ abs (5 - 2 * x) ∧ abs (5 - 2 * x) < 9 ↔ (x > -2 ∧ x ≤ 1) ∨ (x ≥ 4 ∧ x < 7) := sorry

end inequality_solution_set_l98_98660


namespace max_value_of_linear_combination_l98_98843

theorem max_value_of_linear_combination
  (x y : ℝ)
  (h : x^2 + y^2 = 16 * x + 8 * y + 10) :
  ∃ z, z = 4.58 ∧ (∀ x y, (4 * x + 3 * y) ≤ z ∧ (x^2 + y^2 = 16 * x + 8 * y + 10) → (4 * x + 3 * y) ≤ 4.58) :=
by
  sorry

end max_value_of_linear_combination_l98_98843


namespace no_solution_for_given_m_l98_98734

theorem no_solution_for_given_m (x m : ℝ) (h1 : x ≠ 5) (h2 : x ≠ 8) :
  (∀ y : ℝ, (y - 2) / (y - 5) = (y - m) / (y - 8) → false) ↔ m = 5 :=
by
  sorry

end no_solution_for_given_m_l98_98734


namespace almost_square_as_quotient_l98_98826

-- Defining what almost squares are
def isAlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

-- Statement of the theorem
theorem almost_square_as_quotient (n : ℕ) (hn : n > 0) :
  ∃ a b : ℕ, isAlmostSquare a ∧ isAlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end almost_square_as_quotient_l98_98826


namespace pond_field_ratio_l98_98579

theorem pond_field_ratio (L W : ℕ) (pond_side : ℕ) (hL : L = 24) (hLW : L = 2 * W) (hPond : pond_side = 6) :
  pond_side * pond_side / (L * W) = 1 / 8 :=
by
  sorry

end pond_field_ratio_l98_98579


namespace area_increase_percentage_area_percentage_increase_length_to_width_ratio_l98_98691

open Real

-- Part (a)
theorem area_increase_percentage (a b : ℝ) :
  (1.12 * a) * (1.15 * b) = 1.288 * (a * b) :=
  sorry

theorem area_percentage_increase (a b : ℝ) :
  ((1.12 * a) * (1.15 * b)) / (a * b) = 1.288 :=
  sorry

-- Part (b)
theorem length_to_width_ratio (a b : ℝ) (h : 2 * ((1.12 * a) + (1.15 * b)) = 1.13 * 2 * (a + b)) :
  a = 2 * b :=
  sorry

end area_increase_percentage_area_percentage_increase_length_to_width_ratio_l98_98691


namespace inequality_proof_l98_98068

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
by
  sorry

end inequality_proof_l98_98068


namespace sales_second_month_l98_98043

theorem sales_second_month 
  (sale_1 : ℕ) (sale_2 : ℕ) (sale_3 : ℕ) (sale_4 : ℕ) (sale_5 : ℕ) (sale_6 : ℕ)
  (avg_sale : ℕ)
  (h1 : sale_1 = 5400)
  (h2 : sale_3 = 6300)
  (h3 : sale_4 = 7200)
  (h4 : sale_5 = 4500)
  (h5 : sale_6 = 1200)
  (h_avg : avg_sale = 5600) :
  sale_2 = 9000 := 
by sorry

end sales_second_month_l98_98043


namespace sum_of_first_15_even_positive_integers_l98_98104

theorem sum_of_first_15_even_positive_integers :
  let a := 2
  let l := 30
  let n := 15
  let S := (a + l) / 2 * n
  S = 240 := by
  sorry

end sum_of_first_15_even_positive_integers_l98_98104


namespace year_2013_is_not_special_l98_98952

def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), month * day = year % 100 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

theorem year_2013_is_not_special : ¬ is_special_year 2013 := by
  sorry

end year_2013_is_not_special_l98_98952


namespace law_I_law_II_l98_98765

section
variable (x y z : ℝ)

def op_at (a b : ℝ) : ℝ := a + 2 * b
def op_hash (a b : ℝ) : ℝ := 2 * a - b

theorem law_I (x y z : ℝ) : op_at x (op_hash y z) = op_hash (op_at x y) (op_at x z) := 
by
  unfold op_at op_hash
  sorry

theorem law_II (x y z : ℝ) : x + op_at y z ≠ op_at (x + y) (x + z) := 
by
  unfold op_at
  sorry

end

end law_I_law_II_l98_98765


namespace scientific_notation_of_130944000000_l98_98241

theorem scientific_notation_of_130944000000 :
  130944000000 = 1.30944 * 10^11 :=
by sorry

end scientific_notation_of_130944000000_l98_98241


namespace geometric_sum_n_equals_4_l98_98379

def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def S (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))
def sum_value : ℚ := 26 / 81

theorem geometric_sum_n_equals_4 (n : ℕ) (h : S n = sum_value) : n = 4 :=
by sorry

end geometric_sum_n_equals_4_l98_98379


namespace perpendicular_k_value_parallel_k_value_l98_98331

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 2)
def u (k : ℝ) : ℝ × ℝ := (k - 1, 2 * k + 2)
def v : ℝ × ℝ := (4, -4)

noncomputable def is_perpendicular (x y : ℝ × ℝ) : Prop :=
  x.1 * y.1 + x.2 * y.2 = 0

noncomputable def is_parallel (x y : ℝ × ℝ) : Prop :=
  x.1 * y.2 = x.2 * y.1

theorem perpendicular_k_value :
  is_perpendicular (u (-3)) v :=
by sorry

theorem parallel_k_value :
  is_parallel (u (-1/3)) v :=
by sorry

end perpendicular_k_value_parallel_k_value_l98_98331


namespace sum_of_intercepts_of_line_l98_98081

theorem sum_of_intercepts_of_line (x y : ℝ) (hx : 2 * x - 3 * y + 6 = 0) :
  2 + (-3) = -1 :=
sorry

end sum_of_intercepts_of_line_l98_98081


namespace regression_line_l98_98704

theorem regression_line (x y : ℝ) (m : ℝ) (x1 y1 : ℝ)
  (h_slope : m = 6.5)
  (h_point : (x1, y1) = (2, 3)) :
  (y - y1) = m * (x - x1) ↔ y = 6.5 * x - 10 :=
by
  sorry

end regression_line_l98_98704


namespace fraction_four_or_older_l98_98336

theorem fraction_four_or_older (total_students : ℕ) (under_three : ℕ) (not_between_three_and_four : ℕ)
  (h_total : total_students = 300) (h_under_three : under_three = 20) (h_not_between_three_and_four : not_between_three_and_four = 50) :
  (not_between_three_and_four - under_three) / total_students = 1 / 10 :=
by
  sorry

end fraction_four_or_older_l98_98336


namespace lines_coinicide_l98_98115

open Real

theorem lines_coinicide (k m n : ℝ) :
  (∃ (x y : ℝ), y = k * x + m ∧ y = m * x + n ∧ y = n * x + k) →
  k = m ∧ m = n :=
by
  sorry

end lines_coinicide_l98_98115


namespace sphere_radius_l98_98641

theorem sphere_radius (x y z r : ℝ) (h1 : 2 * x * y + 2 * y * z + 2 * z * x = 384)
  (h2 : x + y + z = 28) (h3 : (2 * r)^2 = x^2 + y^2 + z^2) : r = 10 := sorry

end sphere_radius_l98_98641


namespace ordered_pair_a_82_a_28_l98_98728

-- Definitions for the conditions
def a (i j : ℕ) : ℕ :=
  if i % 2 = 1 then
    if j = 1 then i * i else i * i - (j - 1)
  else
    if j = 1 then (i-1) * i + 1 else i * i - (j - 1)

theorem ordered_pair_a_82_a_28 : (a 8 2, a 2 8) = (51, 63) := by
  sorry

end ordered_pair_a_82_a_28_l98_98728


namespace field_size_l98_98568

theorem field_size
  (cost_per_foot : ℝ)
  (total_money : ℝ)
  (cannot_fence : ℝ)
  (cost_per_foot_eq : cost_per_foot = 30)
  (total_money_eq : total_money = 120000)
  (cannot_fence_eq : cannot_fence > 1000) :
  ∃ (side_length : ℝ), side_length * side_length = 1000000 := 
by
  sorry

end field_size_l98_98568


namespace number_of_lilies_l98_98064

theorem number_of_lilies (L : ℕ) 
  (h1 : ∀ n:ℕ, n * 6 = 6 * n)
  (h2 : ∀ n:ℕ, n * 3 = 3 * n) 
  (h3 : 5 * 3 = 15)
  (h4 : 6 * L + 15 = 63) : 
  L = 8 := 
by
  -- Proof omitted 
  sorry

end number_of_lilies_l98_98064


namespace rain_in_first_hour_l98_98632

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end rain_in_first_hour_l98_98632


namespace unique_function_l98_98308

def satisfies_inequality (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x + z) + k * f x * f (y * z) ≥ k^2

theorem unique_function (k : ℤ) (h : k > 0) :
  ∃! f : ℝ → ℝ, satisfies_inequality f k :=
by
  sorry

end unique_function_l98_98308


namespace solve_system_a_solve_system_b_l98_98716

-- For problem (a):
theorem solve_system_a (x y : ℝ) :
  (x + y + x * y = 5) ∧ (x * y * (x + y) = 6) → 
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) := 
by
  sorry

-- For problem (b):
theorem solve_system_b (x y : ℝ) :
  (x^3 + y^3 + 2 * x * y = 4) ∧ (x^2 - x * y + y^2 = 1) → 
  (x = 1 ∧ y = 1) := 
by
  sorry

end solve_system_a_solve_system_b_l98_98716


namespace term_addition_k_to_kplus1_l98_98487

theorem term_addition_k_to_kplus1 (k : ℕ) : 
  (2 * k + 2) + (2 * k + 3) = 4 * k + 5 := 
sorry

end term_addition_k_to_kplus1_l98_98487


namespace number_of_indeterminate_conditions_l98_98321

noncomputable def angle_sum (A B C : ℝ) : Prop := A + B + C = 180
noncomputable def condition1 (A B C : ℝ) : Prop := A + B = C
noncomputable def condition2 (A B C : ℝ) : Prop := A = C / 6 ∧ B = 2 * (C / 6)
noncomputable def condition3 (A B : ℝ) : Prop := A = 90 - B
noncomputable def condition4 (A B C : ℝ) : Prop := A = B ∧ B = C
noncomputable def condition5 (A B C : ℝ) : Prop := 2 * A = C ∧ 2 * B = C
noncomputable def is_right_triangle (C : ℝ) : Prop := C = 90

theorem number_of_indeterminate_conditions (A B C : ℝ) :
  (angle_sum A B C) →
  (condition1 A B C → is_right_triangle C) →
  (condition2 A B C → is_right_triangle C) →
  (condition3 A B → is_right_triangle C) →
  (condition4 A B C → ¬ is_right_triangle C) →
  (condition5 A B C → is_right_triangle C) →
  ∃ n, n = 1 :=
sorry

end number_of_indeterminate_conditions_l98_98321


namespace propositions_imply_implication_l98_98004

theorem propositions_imply_implication (p q r : Prop) :
  ( ((p ∧ q ∧ ¬r) → ((p ∧ q) → r) = False) ∧ 
    ((¬p ∧ q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((p ∧ ¬q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((¬p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r) = True) ) → 
  ( (∀ (x : ℕ), x = 3) ) :=
by
  sorry

end propositions_imply_implication_l98_98004


namespace value_2_std_devs_less_than_mean_l98_98617

-- Define the arithmetic mean
def mean : ℝ := 15.5

-- Define the standard deviation
def standard_deviation : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value_2_std_less_than_mean : ℝ := mean - 2 * standard_deviation

-- The theorem we want to prove
theorem value_2_std_devs_less_than_mean : value_2_std_less_than_mean = 12.5 := by
  sorry

end value_2_std_devs_less_than_mean_l98_98617


namespace part1_part2_l98_98581

section
  variable {x a : ℝ}

  def f (x a : ℝ) := |x - a| + 3 * x

  theorem part1 (h : a = 1) : 
    (∀ x, f x a ≥ 3 * x + 2 ↔ (x ≥ 3 ∨ x ≤ -1)) :=
    sorry

  theorem part2 : 
    (∀ x, (f x a) ≤ 0 ↔ (x ≤ -1)) → a = 2 :=
    sorry
end

end part1_part2_l98_98581


namespace ratio_of_neighborhood_to_gina_l98_98651

variable (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ)

def neighborhood_to_gina_ratio (Gina_bags : ℕ) (Weight_per_bag : ℕ) (Total_weight_collected : ℕ) := 
  (Total_weight_collected - Gina_bags * Weight_per_bag) / (Gina_bags * Weight_per_bag)

theorem ratio_of_neighborhood_to_gina 
  (h₁ : Gina_bags = 2) 
  (h₂ : Weight_per_bag = 4) 
  (h₃ : Total_weight_collected = 664) :
  neighborhood_to_gina_ratio Gina_bags Weight_per_bag Total_weight_collected = 82 := 
by 
  sorry

end ratio_of_neighborhood_to_gina_l98_98651


namespace correct_value_division_l98_98814

theorem correct_value_division (x : ℕ) (h : 9 - x = 3) : 96 / x = 16 :=
by
  sorry

end correct_value_division_l98_98814


namespace largest_integer_less_85_with_remainder_3_l98_98840

theorem largest_integer_less_85_with_remainder_3 (n : ℕ) : 
  n < 85 ∧ n % 9 = 3 → n ≤ 84 :=
by
  intro h
  sorry

end largest_integer_less_85_with_remainder_3_l98_98840


namespace nonagon_perimeter_l98_98025

theorem nonagon_perimeter :
  (2 + 2 + 3 + 3 + 1 + 3 + 2 + 2 + 2 = 20) := by
  sorry

end nonagon_perimeter_l98_98025


namespace number_of_children_l98_98524

theorem number_of_children (total_people : ℕ) (num_adults num_children : ℕ)
  (h1 : total_people = 42)
  (h2 : num_children = 2 * num_adults)
  (h3 : num_adults + num_children = total_people) :
  num_children = 28 :=
by
  sorry

end number_of_children_l98_98524


namespace phil_packs_duration_l98_98234

noncomputable def total_cards_left_after_fire : ℕ := 520
noncomputable def total_cards_initially : ℕ := total_cards_left_after_fire * 2
noncomputable def cards_per_pack : ℕ := 20
noncomputable def packs_bought_weeks : ℕ := total_cards_initially / cards_per_pack

theorem phil_packs_duration : packs_bought_weeks = 52 := by
  sorry

end phil_packs_duration_l98_98234


namespace ratio_tin_copper_in_b_l98_98002

variable (L_a T_a T_b C_b : ℝ)

-- Conditions
axiom h1 : 170 + 250 = 420
axiom h2 : L_a / T_a = 1 / 3
axiom h3 : T_a + T_b = 221.25
axiom h4 : T_a + L_a = 170
axiom h5 : T_b + C_b = 250

-- Target
theorem ratio_tin_copper_in_b (h1 : 170 + 250 = 420) (h2 : L_a / T_a = 1 / 3)
  (h3 : T_a + T_b = 221.25) (h4 : T_a + L_a = 170) (h5 : T_b + C_b = 250) :
  T_b / C_b = 3 / 5 := by
  sorry

end ratio_tin_copper_in_b_l98_98002


namespace original_price_of_article_l98_98191

theorem original_price_of_article (SP : ℝ) (profit_rate : ℝ) (P : ℝ) (h1 : SP = 550) (h2 : profit_rate = 0.10) (h3 : SP = P * (1 + profit_rate)) : P = 500 :=
by
  sorry

end original_price_of_article_l98_98191


namespace dave_more_than_derek_l98_98169

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l98_98169


namespace seokjin_higher_than_jungkook_l98_98235

variable (Jungkook_yoojeong_seokjin_stairs : ℕ)

def jungkook_stair := 19
def yoojeong_stair := jungkook_stair + 8
def seokjin_stair := yoojeong_stair - 5

theorem seokjin_higher_than_jungkook : seokjin_stair - jungkook_stair = 3 :=
by sorry

end seokjin_higher_than_jungkook_l98_98235


namespace find_m_l98_98445

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

def C_UA : Set ℕ := {1, 2}

theorem find_m (m : ℝ) (hA : A m = {0, 3}) (hCUA : U \ A m = C_UA) : m = -3 := 
  sorry

end find_m_l98_98445


namespace find_constants_l98_98562

theorem find_constants :
  ∃ (A B C : ℚ), 
  (A = 1 ∧ B = 4 ∧ C = 1) ∧ 
  (∀ x, x ≠ -1 → x ≠ 3/2 → x ≠ 2 → 
    (6 * x^2 - 13 * x + 6) / (2 * x^3 + 3 * x^2 - 11 * x - 6) = 
    (A / (x + 1) + B / (2 * x - 3) + C / (x - 2))) :=
by
  sorry

end find_constants_l98_98562


namespace regular_polygon_sides_l98_98996

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l98_98996


namespace closest_approx_w_l98_98003

noncomputable def w : ℝ := ((69.28 * 123.57 * 0.004) - (42.67 * 3.12)) / (0.03 * 8.94 * 1.25)

theorem closest_approx_w : |w + 296.073| < 0.001 :=
by
  sorry

end closest_approx_w_l98_98003


namespace smallest_sum_of_three_diff_numbers_l98_98860

theorem smallest_sum_of_three_diff_numbers : 
  ∀ (s : Set ℤ), s = {8, -7, 2, -4, 20} → ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = -9) :=
by
  sorry

end smallest_sum_of_three_diff_numbers_l98_98860


namespace calculate_expression_l98_98138

theorem calculate_expression : 
  (0.25 ^ 16) * ((-4) ^ 17) = -4 := 
by
  sorry

end calculate_expression_l98_98138


namespace minimum_value_l98_98629

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a + b = 1 / 2

theorem minimum_value (a b : ℝ) (h : min_value_condition a b) :
  (4 / a) + (1 / b) ≥ 18 :=
by
  sorry

end minimum_value_l98_98629


namespace bar_graph_proportion_correct_l98_98121

def white : ℚ := 1/2
def black : ℚ := 1/4
def gray : ℚ := 1/8
def light_gray : ℚ := 1/16

theorem bar_graph_proportion_correct :
  (white = 1 / 2) ∧
  (black = white / 2) ∧
  (gray = black / 2) ∧
  (light_gray = gray / 2) →
  (white = 1 / 2) ∧
  (black = 1 / 4) ∧
  (gray = 1 / 8) ∧
  (light_gray = 1 / 16) :=
by
  intros
  sorry

end bar_graph_proportion_correct_l98_98121


namespace directrix_of_parabola_l98_98171

theorem directrix_of_parabola (y x : ℝ) : 
  (∃ a h k : ℝ, y = a * (x - h)^2 + k ∧ a = 1/8 ∧ h = 4 ∧ k = 0) → 
  y = -1/2 :=
by
  intro h
  sorry

end directrix_of_parabola_l98_98171


namespace cast_cost_l98_98542

theorem cast_cost (C : ℝ) 
  (visit_cost : ℝ := 300)
  (insurance_coverage : ℝ := 0.60)
  (out_of_pocket_cost : ℝ := 200) :
  0.40 * (visit_cost + C) = out_of_pocket_cost → 
  C = 200 := by
  sorry

end cast_cost_l98_98542


namespace first_term_of_infinite_geometric_series_l98_98915

theorem first_term_of_infinite_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := 
sorry

end first_term_of_infinite_geometric_series_l98_98915


namespace range_of_a_nonempty_intersection_range_of_a_subset_intersection_l98_98628

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) ≤ 0}

-- Define set B in terms of variable a
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Statement 1: Proving the range of a when A ∩ B ≠ ∅
theorem range_of_a_nonempty_intersection (a : ℝ) : (A ∩ B a ≠ ∅) → (-1 / 2 ≤ a ∧ a ≤ 2) :=
by
  sorry

-- Statement 2: Proving the range of a when A ∩ B = B
theorem range_of_a_subset_intersection (a : ℝ) : (A ∩ B a = B a) → (a ≥ 2 ∨ a ≤ -3) :=
by
  sorry

end range_of_a_nonempty_intersection_range_of_a_subset_intersection_l98_98628


namespace sum_of_possible_ks_l98_98229

theorem sum_of_possible_ks :
  ∃ S : Finset ℕ, (∀ (j k : ℕ), j > 0 ∧ k > 0 → (1 / j + 1 / k = 1 / 4) ↔ k ∈ S) ∧ S.sum id = 51 :=
  sorry

end sum_of_possible_ks_l98_98229


namespace correct_equation_l98_98640

-- Conditions:
def number_of_branches (x : ℕ) := x
def number_of_small_branches (x : ℕ) := x * x
def total_number (x : ℕ) := 1 + number_of_branches x + number_of_small_branches x

-- Proof Problem:
theorem correct_equation (x : ℕ) : total_number x = 43 → x^2 + x + 1 = 43 :=
by 
  sorry

end correct_equation_l98_98640


namespace determine_function_l98_98630

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem determine_function (f : ℤ → ℤ) (h : satisfies_condition f) :
  ∀ n : ℤ, f n = 0 ∨ ∃ K : ℤ, f n = 2 * n + K :=
sorry

end determine_function_l98_98630


namespace no_solution_for_x_l98_98341

theorem no_solution_for_x (m : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (mx - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) :=
by
  sorry

end no_solution_for_x_l98_98341


namespace marked_vertices_coincide_l98_98097

theorem marked_vertices_coincide :
  ∀ (P Q : Fin 16 → Prop),
  (∃ A B C D E F G : Fin 16, P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G) →
  (∃ A' B' C' D' E' F' G' : Fin 16, Q A' ∧ Q B' ∧ Q C' ∧ Q D' ∧ Q E' ∧ Q F' ∧ Q G') →
  ∃ (r : Fin 16), ∃ (A B C D : Fin 16), 
  (Q ((A + r) % 16) ∧ Q ((B + r) % 16) ∧ Q ((C + r) % 16) ∧ Q ((D + r) % 16)) :=
by
  sorry

end marked_vertices_coincide_l98_98097


namespace geometric_sequence_ratio_l98_98232

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (q : ℕ) (h1 : q = 2)
  (h2 : ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 4 / S 2 = 5 :=
by
  sorry

end geometric_sequence_ratio_l98_98232


namespace sequence_difference_l98_98864

-- Definition of sequences sums
def odd_sum (n : ℕ) : ℕ := (n * n)
def even_sum (n : ℕ) : ℕ := n * (n + 1)

-- Main property to prove
theorem sequence_difference :
  odd_sum 1013 - even_sum 1011 = 3047 :=
by
  -- Definitions and assertions here
  sorry

end sequence_difference_l98_98864


namespace hyperbola_foci_difference_l98_98080

noncomputable def hyperbola_foci_distance (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (a : ℝ) : ℝ :=
  |dist P F₁ - dist P F₂|

theorem hyperbola_foci_difference (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : 
  (P.1 ^ 2 - P.2 ^ 2 = 4) ∧ (P.1 < 0) → (hyperbola_foci_distance P F₁ F₂ 2 = -4) :=
by
  intros h
  sorry

end hyperbola_foci_difference_l98_98080


namespace cost_of_toaster_l98_98359

-- Definitions based on the conditions
def initial_spending : ℕ := 3000
def tv_return : ℕ := 700
def returned_bike_cost : ℕ := 500
def sold_bike_cost : ℕ := returned_bike_cost + (returned_bike_cost / 5)
def selling_price : ℕ := (4 * sold_bike_cost) / 5
def total_out_of_pocket : ℕ := 2020

-- Proving the cost of the toaster
theorem cost_of_toaster : initial_spending - (tv_return + returned_bike_cost) + selling_price - total_out_of_pocket = 260 := by
  sorry

end cost_of_toaster_l98_98359


namespace swimming_club_cars_l98_98162

theorem swimming_club_cars (c : ℕ) :
  let vans := 3
  let people_per_car := 5
  let people_per_van := 3
  let max_people_per_car := 6
  let max_people_per_van := 8
  let extra_people := 17
  let total_people := 5 * c + (people_per_van * vans)
  let max_capacity := max_people_per_car * c + (max_people_per_van * vans)
  (total_people + extra_people = max_capacity) → c = 2 := by
  sorry

end swimming_club_cars_l98_98162


namespace person_last_name_length_l98_98177

theorem person_last_name_length (samantha_lastname: ℕ) (bobbie_lastname: ℕ) (person_lastname: ℕ) 
  (h1: samantha_lastname + 3 = bobbie_lastname)
  (h2: bobbie_lastname - 2 = 2 * person_lastname)
  (h3: samantha_lastname = 7) :
  person_lastname = 4 :=
by 
  sorry

end person_last_name_length_l98_98177


namespace moles_of_CO2_required_l98_98009

theorem moles_of_CO2_required (n_H2O n_H2CO3 : ℕ) (h1 : n_H2O = n_H2CO3) (h2 : n_H2O = 2): 
  (n_H2O = 2) → (∃ n_CO2 : ℕ, n_CO2 = n_H2O) :=
by
  sorry

end moles_of_CO2_required_l98_98009


namespace cost_of_45_roses_l98_98810

theorem cost_of_45_roses (cost_15_roses : ℕ → ℝ) 
  (h1 : cost_15_roses 15 = 25) 
  (h2 : ∀ (n m : ℕ), cost_15_roses n / n = cost_15_roses m / m )
  (h3 : ∀ (n : ℕ), n > 30 → cost_15_roses n = (1 - 0.10) * cost_15_roses n) :
  cost_15_roses 45 = 67.5 :=
by
  sorry

end cost_of_45_roses_l98_98810


namespace one_thirds_in_eight_halves_l98_98743

theorem one_thirds_in_eight_halves : (8 / 2) / (1 / 3) = 12 := by
  sorry

end one_thirds_in_eight_halves_l98_98743


namespace range_of_4x_plus_2y_l98_98243

theorem range_of_4x_plus_2y (x y : ℝ) 
  (h₁ : 1 ≤ x + y ∧ x + y ≤ 3)
  (h₂ : -1 ≤ x - y ∧ x - y ≤ 1) : 
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 :=
sorry

end range_of_4x_plus_2y_l98_98243


namespace price_of_second_tea_l98_98559

theorem price_of_second_tea (P : ℝ) (h1 : 1 * 64 + 1 * P = 2 * 69) : P = 74 := 
by
  sorry

end price_of_second_tea_l98_98559


namespace find_inverse_sum_l98_98509

variable {R : Type*} [OrderedRing R]

-- Define the function f and its inverse
variable (f : R → R)
variable (f_inv : R → R)

-- Conditions
axiom f_inverse : ∀ y, f (f_inv y) = y
axiom f_prop : ∀ x, f x + f (1 - x) = 2

-- The theorem we need to prove
theorem find_inverse_sum (x : R) : f_inv (x - 2) + f_inv (4 - x) = 1 :=
by
  sorry

end find_inverse_sum_l98_98509


namespace solve_equations_l98_98248

theorem solve_equations :
  (∀ x : ℝ, (1 / 2) * (2 * x - 5) ^ 2 - 2 = 0 ↔ x = 7 / 2 ∨ x = 3 / 2) ∧
  (∀ x : ℝ, x ^ 2 - 4 * x - 4 = 0 ↔ x = 2 + 2 * Real.sqrt 2 ∨ x = 2 - 2 * Real.sqrt 2) :=
by
  sorry

end solve_equations_l98_98248


namespace solve_for_N_l98_98589

theorem solve_for_N : ∃ N : ℕ, 32^4 * 4^5 = 2^N ∧ N = 30 := by
  sorry

end solve_for_N_l98_98589


namespace members_not_playing_either_l98_98776

variable (total_members badminton_players tennis_players both_players : ℕ)

theorem members_not_playing_either (h1 : total_members = 40)
                                   (h2 : badminton_players = 20)
                                   (h3 : tennis_players = 18)
                                   (h4 : both_players = 3) :
  total_members - (badminton_players + tennis_players - both_players) = 5 := by
  sorry

end members_not_playing_either_l98_98776


namespace parallel_lines_slope_equal_l98_98146

theorem parallel_lines_slope_equal (k : ℝ) : (∀ x : ℝ, 2 * x = k * x + 3) → k = 2 :=
by
  intros
  sorry

end parallel_lines_slope_equal_l98_98146


namespace wire_around_field_l98_98252

theorem wire_around_field 
  (area_square : ℕ)
  (total_length_wire : ℕ)
  (h_area : area_square = 69696)
  (h_total_length : total_length_wire = 15840) :
  (total_length_wire / (4 * Int.natAbs (Int.sqrt area_square))) = 15 :=
  sorry

end wire_around_field_l98_98252


namespace x_varies_as_nth_power_of_z_l98_98503

theorem x_varies_as_nth_power_of_z 
  (k j z : ℝ) 
  (h1 : ∃ y : ℝ, x = k * y^4 ∧ y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := 
 sorry

end x_varies_as_nth_power_of_z_l98_98503


namespace rope_length_equals_120_l98_98019

theorem rope_length_equals_120 (x : ℝ) (l : ℝ)
  (h1 : x + 20 = 3 * x) 
  (h2 : l = 4 * (2 * x)) : 
  l = 120 :=
by
  -- Proof will be provided here
  sorry

end rope_length_equals_120_l98_98019


namespace circle_radius_l98_98437

theorem circle_radius (M N r : ℝ) (h1 : M = Real.pi * r^2) (h2 : N = 2 * Real.pi * r) (h3 : M / N = 25) : r = 50 :=
by
  sorry

end circle_radius_l98_98437


namespace geometric_mean_of_4_and_9_l98_98472

theorem geometric_mean_of_4_and_9 : ∃ (G : ℝ), G = 6 ∨ G = -6 :=
by
  sorry

end geometric_mean_of_4_and_9_l98_98472


namespace value_of_A_l98_98929

theorem value_of_A (h p a c k e : ℤ) 
  (H : h = 8)
  (PACK : p + a + c + k = 50)
  (PECK : p + e + c + k = 54)
  (CAKE : c + a + k + e = 40) : 
  a = 25 :=
by 
  sorry

end value_of_A_l98_98929


namespace card_probability_l98_98984

theorem card_probability :
  let totalCards := 52
  let kings := 4
  let jacks := 4
  let queens := 4
  let firstCardKing := kings / totalCards
  let secondCardJack := jacks / (totalCards - 1)
  let thirdCardQueen := queens / (totalCards - 2)
  (firstCardKing * secondCardJack * thirdCardQueen) = (8 / 16575) :=
by
  sorry

end card_probability_l98_98984


namespace determine_c_l98_98811

-- Assume we have three integers a, b, and unique x, y, z such that
variables (a b c x y z : ℕ)

-- Define the conditions
def condition1 : Prop := a = Nat.lcm y z
def condition2 : Prop := b = Nat.lcm x z
def condition3 : Prop := c = Nat.lcm x y

-- Prove that Bob can determine c based on a and b
theorem determine_c (h1 : condition1 a y z) (h2 : condition2 b x z) (h3 : ∀ u v w : ℕ, (Nat.lcm u w = a ∧ Nat.lcm v w = b ∧ Nat.lcm u v = c) → (u = x ∧ v = y ∧ w = z) ) : ∃ c, condition3 c x y :=
by sorry

end determine_c_l98_98811


namespace seth_pounds_lost_l98_98586

-- Definitions
def pounds_lost_by_Seth (S : ℝ) : Prop := 
  let total_loss := S + 3 * S + (S + 1.5)
  total_loss = 89

theorem seth_pounds_lost (S : ℝ) : pounds_lost_by_Seth S → S = 17.5 := by
  sorry

end seth_pounds_lost_l98_98586


namespace smallest_possible_N_l98_98102

theorem smallest_possible_N (N : ℕ) (h : ∀ m : ℕ, m ≤ 60 → m % 3 = 0 → ∃ i : ℕ, i < 20 ∧ m = 3 * i + 1 ∧ N = 20) :
    N = 20 :=
by 
  sorry

end smallest_possible_N_l98_98102


namespace pond_diameter_l98_98590

theorem pond_diameter 
  (h k r : ℝ)
  (H1 : (4 - h) ^ 2 + (11 - k) ^ 2 = r ^ 2)
  (H2 : (12 - h) ^ 2 + (9 - k) ^ 2 = r ^ 2)
  (H3 : (2 - h) ^ 2 + (7 - k) ^ 2 = (r - 1) ^ 2) :
  2 * r = 9.2 :=
sorry

end pond_diameter_l98_98590


namespace Walter_age_in_2003_l98_98015

-- Defining the conditions
def Walter_age_1998 (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  walter_age_1998 = grandmother_age_1998 / 3

def birth_years_sum (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = 3858

-- Defining the theorem to be proved
theorem Walter_age_in_2003 (walter_age_1998 grandmother_age_1998 : ℝ) 
  (h1 : Walter_age_1998 walter_age_1998 grandmother_age_1998) 
  (h2 : birth_years_sum walter_age_1998 grandmother_age_1998) : 
  walter_age_1998 + 5 = 39.5 :=
  sorry

end Walter_age_in_2003_l98_98015


namespace min_value_of_a_squared_plus_b_squared_l98_98668

-- Problem definition and condition
def is_on_circle (a b : ℝ) : Prop :=
  (a^2 + b^2 - 2*a + 4*b - 20) = 0

-- Theorem statement
theorem min_value_of_a_squared_plus_b_squared (a b : ℝ) (h : is_on_circle a b) :
  a^2 + b^2 = 30 - 10 * Real.sqrt 5 :=
sorry

end min_value_of_a_squared_plus_b_squared_l98_98668


namespace two_people_paint_time_l98_98418

theorem two_people_paint_time (h : 5 * 7 = 35) :
  ∃ t : ℝ, 2 * t = 35 ∧ t = 17.5 := 
sorry

end two_people_paint_time_l98_98418


namespace train_travel_distance_l98_98411

theorem train_travel_distance
  (coal_per_mile_lb : ℝ)
  (remaining_coal_lb : ℝ)
  (travel_distance_per_unit_mile : ℝ)
  (units_per_unit_lb : ℝ)
  (remaining_units : ℝ)
  (total_distance : ℝ) :
  coal_per_mile_lb = 2 →
  remaining_coal_lb = 160 →
  travel_distance_per_unit_mile = 5 →
  units_per_unit_lb = remaining_coal_lb / coal_per_mile_lb →
  remaining_units = units_per_unit_lb →
  total_distance = remaining_units * travel_distance_per_unit_mile →
  total_distance = 400 :=
by
  sorry

end train_travel_distance_l98_98411


namespace two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l98_98021

theorem two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n
  (n : ℕ) (h : 2 < n) : (2 * n - 1) ^ n + (2 * n) ^ n < (2 * n + 1) ^ n :=
sorry

end two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l98_98021


namespace kyle_delivers_daily_papers_l98_98697

theorem kyle_delivers_daily_papers (x : ℕ) (h : 6 * x + (x - 10) + 30 = 720) : x = 100 :=
by
  sorry

end kyle_delivers_daily_papers_l98_98697


namespace total_number_of_bees_is_fifteen_l98_98566

noncomputable def totalBees (B : ℝ) : Prop :=
  (1/5) * B + (1/3) * B + (2/5) * B + 1 = B

theorem total_number_of_bees_is_fifteen : ∃ B : ℝ, totalBees B ∧ B = 15 :=
by
  sorry

end total_number_of_bees_is_fifteen_l98_98566


namespace union_sets_l98_98639

def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 5} := 
by {
  sorry
}

end union_sets_l98_98639


namespace min_max_SX_SY_l98_98935

theorem min_max_SX_SY (n : ℕ) (hn : 2 ≤ n) (a : Finset ℕ) 
  (ha_sum : Finset.sum a id = 2 * n - 1) :
  ∃ (min_val max_val : ℕ), 
    (min_val = 2 * n - 2) ∧ 
    (max_val = n * (n - 1)) :=
sorry

end min_max_SX_SY_l98_98935


namespace largest_exterior_angle_l98_98218

theorem largest_exterior_angle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 180 - 3 * (180 / 12) = 135 :=
by {
  -- Sorry is a placeholder for the actual proof
  sorry
}

end largest_exterior_angle_l98_98218


namespace unknown_number_is_three_or_twenty_seven_l98_98227

theorem unknown_number_is_three_or_twenty_seven
    (x y : ℝ)
    (h1 : y - 3 = x - y)
    (h2 : (y - 6) / 3 = x / (y - 6)) :
    x = 3 ∨ x = 27 :=
by
  sorry

end unknown_number_is_three_or_twenty_seven_l98_98227


namespace anna_phone_chargers_l98_98537

theorem anna_phone_chargers (p l : ℕ) (h₁ : l = 5 * p) (h₂ : l + p = 24) : p = 4 :=
by
  sorry

end anna_phone_chargers_l98_98537


namespace subset_M_N_l98_98583

-- Definitions of M and N as per the problem statement
def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | 1 / x < 2}

-- Lean statement for the proof problem: M ⊆ N
theorem subset_M_N : M ⊆ N := by
  -- Proof will be provided here
  sorry

end subset_M_N_l98_98583


namespace average_growth_rate_income_prediction_l98_98297

-- Define the given conditions
def income2018 : ℝ := 20000
def income2020 : ℝ := 24200
def growth_rate : ℝ := 0.1
def predicted_income2021 : ℝ := 26620

-- Lean 4 statement for the first part of the problem
theorem average_growth_rate :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  growth_rate = 0.1 :=
by
  intros h
  sorry

-- Lean 4 statement for the second part of the problem
theorem income_prediction :
  (income2020 = income2018 * (1 + growth_rate)^2) →
  (growth_rate = 0.1) →
  (income2018 * (1 + growth_rate)^3 = predicted_income2021) :=
by
  intros h1 h2
  sorry

end average_growth_rate_income_prediction_l98_98297


namespace original_curve_eqn_l98_98946

-- Definitions based on conditions
def scaling_transformation_formula (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

def transformed_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

-- The proof problem to be shown in Lean
theorem original_curve_eqn {x y : ℝ} (h : transformed_curve (2 * x) (3 * y)) :
  4 * x^2 + 9 * y^2 = 1 :=
sorry

end original_curve_eqn_l98_98946


namespace car_travel_l98_98443

namespace DistanceTravel

/- Define the conditions -/
def distance_initial : ℕ := 120
def car_speed : ℕ := 80

/- Define the relationship between y and x -/
def y (x : ℝ) : ℝ := distance_initial - car_speed * x

/- Prove that y is a linear function and verify the value of y at x = 0.8 -/
theorem car_travel (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1.5) : 
  (y x = distance_initial - car_speed * x) ∧ 
  (y x = 120 - 80 * x) ∧ 
  (x = 0.8 → y x = 56) :=
sorry

end DistanceTravel

end car_travel_l98_98443


namespace cost_of_basic_calculator_l98_98667

variable (B S G : ℕ)

theorem cost_of_basic_calculator 
  (h₁ : S = 2 * B)
  (h₂ : G = 3 * S)
  (h₃ : B + S + G = 72) : 
  B = 8 :=
by
  sorry

end cost_of_basic_calculator_l98_98667


namespace percentage_increase_in_average_visibility_l98_98577

theorem percentage_increase_in_average_visibility :
  let avg_visibility_without_telescope := (100 + 110) / 2
  let avg_visibility_with_telescope := (150 + 165) / 2
  let increase_in_avg_visibility := avg_visibility_with_telescope - avg_visibility_without_telescope
  let percentage_increase := (increase_in_avg_visibility / avg_visibility_without_telescope) * 100
  percentage_increase = 50 := by
  -- calculations are omitted; proof goes here
  sorry

end percentage_increase_in_average_visibility_l98_98577


namespace general_formulas_max_b_seq_l98_98594

noncomputable def a_seq (n : ℕ) : ℕ := 4 * n - 2
noncomputable def b_seq (n : ℕ) : ℕ := 4 * n - 2 - 2^(n - 1)

-- The general formulas to be proved
theorem general_formulas :
  (∀ n : ℕ, a_seq n = 4 * n - 2) ∧ 
  (∀ n : ℕ, b_seq n = 4 * n - 2 - 2^(n - 1)) :=
by
  sorry

-- The maximum value condition to be proved
theorem max_b_seq :
  ((∀ n : ℕ, b_seq n ≤ b_seq 3) ∨ (∀ n : ℕ, b_seq n ≤ b_seq 4)) :=
by
  sorry

end general_formulas_max_b_seq_l98_98594


namespace julia_played_tag_with_4_kids_on_tuesday_l98_98573

variable (k_monday : ℕ) (k_diff : ℕ)

theorem julia_played_tag_with_4_kids_on_tuesday
  (h_monday : k_monday = 16)
  (h_diff : k_monday = k_tuesday + 12) :
  k_tuesday = 4 :=
by
  sorry

end julia_played_tag_with_4_kids_on_tuesday_l98_98573


namespace correct_multiplication_l98_98548

theorem correct_multiplication :
  ∃ (n : ℕ), 98765 * n = 888885 ∧ (98765 * n = 867559827931 → n = 9) :=
by
  sorry

end correct_multiplication_l98_98548


namespace six_box_four_div_three_eight_box_two_div_four_l98_98907

def fills_middle_zero (d : Nat) : Prop :=
  d < 3

def fills_last_zero (d : Nat) : Prop :=
  (80 + d) % 4 = 0

theorem six_box_four_div_three {d : Nat} : fills_middle_zero d → ((600 + d * 10 + 4) / 3) % 100 / 10 = 0 :=
  sorry

theorem eight_box_two_div_four {d : Nat} : fills_last_zero d → ((800 + d * 10 + 2) / 4) % 10 = 0 :=
  sorry

end six_box_four_div_three_eight_box_two_div_four_l98_98907


namespace algebra_expression_value_l98_98885

theorem algebra_expression_value (x y : ℝ) (h1 : x * y = 3) (h2 : x - y = -2) : x^2 * y - x * y^2 = -6 := 
by
  sorry

end algebra_expression_value_l98_98885


namespace jar_water_fraction_l98_98923

theorem jar_water_fraction
  (S L : ℝ)
  (h1 : S = (1 / 5) * S)
  (h2 : S = x * L)
  (h3 : (1 / 5) * S + x * L = (2 / 5) * L) :
  x = (1 / 10) :=
by
  sorry

end jar_water_fraction_l98_98923


namespace simplify_expr_l98_98496

variable (a b : ℤ)  -- assuming a and b are elements of the ring ℤ

theorem simplify_expr : 105 * a - 38 * a + 27 * b - 12 * b = 67 * a + 15 * b := 
by
  sorry

end simplify_expr_l98_98496


namespace x_investment_amount_l98_98300

variable (X : ℝ)
variable (investment_y : ℝ := 15000)
variable (total_profit : ℝ := 1600)
variable (x_share : ℝ := 400)

theorem x_investment_amount :
  (total_profit - x_share) / investment_y = x_share / X → X = 5000 :=
by
  intro ratio
  have h1: 1200 / 15000 = 400 / 5000 :=
    by sorry
  have h2: X = 5000 :=
    by sorry
  exact h2

end x_investment_amount_l98_98300


namespace compare_magnitudes_l98_98801

noncomputable def A : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))
noncomputable def B : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def C : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def D : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))

theorem compare_magnitudes : B < C ∧ C < A ∧ A < D :=
by
  sorry

end compare_magnitudes_l98_98801


namespace expression_evaluation_l98_98380

theorem expression_evaluation : 
  (50 - (2210 - 251)) + (2210 - (251 - 50)) = 100 := 
  by sorry

end expression_evaluation_l98_98380


namespace find_angle_D_l98_98183

theorem find_angle_D 
  (A B C D : ℝ) 
  (h1 : A + B = 180) 
  (h2 : C = D + 10) 
  (h3 : A = 50)
  : D = 20 := by
  sorry

end find_angle_D_l98_98183


namespace john_allowance_calculation_l98_98812

theorem john_allowance_calculation (initial_money final_money game_cost allowance: ℕ) 
(h_initial: initial_money = 5) 
(h_game_cost: game_cost = 2) 
(h_final: final_money = 29) 
(h_allowance: final_money = initial_money - game_cost + allowance) : 
  allowance = 26 :=
by
  sorry

end john_allowance_calculation_l98_98812


namespace range_of_x_l98_98653

theorem range_of_x (x : ℝ) : (∀ t : ℝ, -1 ≤ t ∧ t ≤ 3 → x^2 - (t^2 + t - 3) * x + t^2 * (t - 3) > 0) ↔ (x < -4 ∨ x > 9) :=
by
  sorry

end range_of_x_l98_98653


namespace find_first_term_geometric_sequence_l98_98038

theorem find_first_term_geometric_sequence 
  (a b c : ℚ) 
  (h₁ : b = a * 4) 
  (h₂ : 36 = a * 4^2) 
  (h₃ : c = a * 4^3) 
  (h₄ : 144 = a * 4^4) : 
  a = 9 / 4 :=
sorry

end find_first_term_geometric_sequence_l98_98038


namespace find_e_of_conditions_l98_98745

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := x^3 + d * x^2 + e * x + f

theorem find_e_of_conditions (d e f : ℝ) 
  (h1 : f = 6) 
  (h2 : -d / 3 = -f)
  (h3 : -f = d + e + f - 1) : 
  e = -30 :=
by 
  sorry

end find_e_of_conditions_l98_98745


namespace classroom_problem_l98_98203

noncomputable def classroom_problem_statement : Prop :=
  ∀ (B G : ℕ) (b g : ℝ),
    b > 0 →
    g > 0 →
    B > 0 →
    G > 0 →
    ¬ ((B * g + G * b) / (B + G) = b + g ∧ b > 0 ∧ g > 0)

theorem classroom_problem : classroom_problem_statement :=
  by
    intros B G b g hb_gt0 hg_gt0 hB_gt0 hG_gt0
    sorry

end classroom_problem_l98_98203


namespace ABC_books_sold_eq_4_l98_98913

/-- "TOP" book cost in dollars --/
def TOP_price : ℕ := 8

/-- "ABC" book cost in dollars --/
def ABC_price : ℕ := 23

/-- Number of "TOP" books sold --/
def TOP_books_sold : ℕ := 13

/-- Difference in earnings in dollars --/
def earnings_difference : ℕ := 12

/-- Prove the number of "ABC" books sold --/
theorem ABC_books_sold_eq_4 (x : ℕ) (h : TOP_books_sold * TOP_price - x * ABC_price = earnings_difference) : x = 4 :=
by
  sorry

end ABC_books_sold_eq_4_l98_98913


namespace negation_of_prop_l98_98030

-- Define the original proposition
def prop (x : ℝ) : Prop := x^2 - x + 2 ≥ 0

-- State the negation of the original proposition
theorem negation_of_prop : (¬ ∀ x : ℝ, prop x) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := 
by
  sorry

end negation_of_prop_l98_98030


namespace friend_charge_per_animal_l98_98071

-- Define the conditions.
def num_cats := 2
def num_dogs := 3
def total_payment := 65

-- Define the total number of animals.
def total_animals := num_cats + num_dogs

-- Define the charge per animal per night.
def charge_per_animal := total_payment / total_animals

-- State the theorem.
theorem friend_charge_per_animal : charge_per_animal = 13 := by
  -- Proof goes here.
  sorry

end friend_charge_per_animal_l98_98071


namespace job_candidates_excel_nights_l98_98483

theorem job_candidates_excel_nights (hasExcel : ℝ) (dayShift : ℝ) 
    (h1 : hasExcel = 0.2) (h2 : dayShift = 0.7) : 
    (1 - dayShift) * hasExcel = 0.06 :=
by
  sorry

end job_candidates_excel_nights_l98_98483


namespace line_eq1_line_eq2_l98_98835

-- Define the line equations
def l1 (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Theorem for when midpoint is at (0, 0)
theorem line_eq1 : ∀ x y : ℝ, (x + 6 * y = 0) ↔
  ∃ (a : ℝ), 
    l1 a (-(a / 6)) ∧
    l2 (-a) ((a / 6)) ∧
    (a + -a = 0) ∧ (-(a / 6) + a / 6 = 0) := 
by 
  sorry

-- Theorem for when midpoint is at (0, 1)
theorem line_eq2 : ∀ x y : ℝ, (x + 2 * y - 2 = 0) ↔
  ∃ (b : ℝ),
    l1 b (-b / 2 + 1) ∧
    l2 (-b) (1 - (-b / 2)) ∧
    (b + -b = 0) ∧ (-b / 2 + 1 + (1 - (-b / 2)) = 2) := 
by 
  sorry

end line_eq1_line_eq2_l98_98835


namespace find_normal_price_l98_98624

open Real

theorem find_normal_price (P : ℝ) (h1 : 0.612 * P = 108) : P = 176.47 := by
  sorry

end find_normal_price_l98_98624


namespace defeat_giant_enemy_crab_l98_98198

-- Definitions for the conditions of cutting legs and claws
def claws : ℕ := 2
def legs : ℕ := 6
def totalCuts : ℕ := claws + legs
def valid_sequences : ℕ :=
  (Nat.factorial legs) * (Nat.factorial claws) * Nat.choose (totalCuts - claws - 1) claws

-- Statement to prove the number of valid sequences of cuts given the conditions
theorem defeat_giant_enemy_crab : valid_sequences = 14400 := by
  sorry

end defeat_giant_enemy_crab_l98_98198


namespace gas_and_maintenance_money_l98_98347

theorem gas_and_maintenance_money
  (income : ℝ := 3200)
  (rent : ℝ := 1250)
  (utilities : ℝ := 150)
  (retirement_savings : ℝ := 400)
  (groceries : ℝ := 300)
  (insurance : ℝ := 200)
  (miscellaneous_expenses : ℝ := 200)
  (car_payment : ℝ := 350) :
  income - (rent + utilities + retirement_savings + groceries + insurance + miscellaneous_expenses + car_payment) = 350 :=
by
  sorry

end gas_and_maintenance_money_l98_98347


namespace num_of_positive_divisors_l98_98219

-- Given conditions
variables {x y z : ℕ}
variables (p1 p2 p3 : ℕ) -- primes
variables (h1 : x = p1 ^ 3) (h2 : y = p2 ^ 3) (h3 : z = p3 ^ 3)
variables (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)

-- Lean statement to prove
theorem num_of_positive_divisors (hx3 : x = p1 ^ 3) (hy3 : y = p2 ^ 3) (hz3 : z = p3 ^ 3) 
    (Hdist : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
    ∃ n : ℕ, n = 10 * 13 * 7 ∧ n = (x^3 * y^4 * z^2).factors.length :=
sorry

end num_of_positive_divisors_l98_98219


namespace intersection_of_M_and_N_l98_98066

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l98_98066


namespace sequence_perfect_square_l98_98485

theorem sequence_perfect_square (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) :
  ∃! n, ∃ k, a n = k ^ 2 :=
by
  sorry

end sequence_perfect_square_l98_98485


namespace relationship_between_k_and_a_l98_98677

theorem relationship_between_k_and_a (a k : ℝ) (h_a : 0 < a ∧ a < 1) :
  (k^2 + 1) * a^2 ≥ 1 :=
sorry

end relationship_between_k_and_a_l98_98677


namespace prove_sufficient_and_necessary_l98_98244

-- The definition of the focus of the parabola y^2 = 4x.
def focus_parabola : (ℝ × ℝ) := (1, 0)

-- The condition that the line passes through a given point.
def line_passes_through (m b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.2 = m * p.1 + b

-- Let y = x + b and the equation of the parabola be y^2 = 4x.
def sufficient_and_necessary (b : ℝ) : Prop :=
  line_passes_through 1 b focus_parabola ↔ b = -1

theorem prove_sufficient_and_necessary : sufficient_and_necessary (-1) :=
by
  sorry

end prove_sufficient_and_necessary_l98_98244


namespace shape_area_l98_98845

-- Define the conditions as Lean definitions
def side_length : ℝ := 3
def num_squares : ℕ := 4

-- Prove that the area of the shape is 36 cm² given the conditions
theorem shape_area : num_squares * (side_length * side_length) = 36 := by
    -- The proof is skipped with sorry
    sorry

end shape_area_l98_98845


namespace sqrt_four_squared_l98_98654

theorem sqrt_four_squared : (Real.sqrt 4) ^ 2 = 4 :=
  by
    sorry

end sqrt_four_squared_l98_98654


namespace cookie_distribution_l98_98123

theorem cookie_distribution:
  ∀ (initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny : ℕ),
    initial_boxes = 45 →
    brother_cookies = 12 →
    sister_cookies = 9 →
    after_siblings = initial_boxes - brother_cookies - sister_cookies →
    leftover_sonny = 17 →
    leftover = after_siblings - leftover_sonny →
    leftover = 7 :=
by
  intros initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny
  intros h1 h2 h3 h4 h5 h6
  sorry

end cookie_distribution_l98_98123


namespace average_speed_to_first_summit_l98_98759

theorem average_speed_to_first_summit 
  (time_first_summit : ℝ := 3)
  (time_descend_partially : ℝ := 1)
  (time_second_uphill : ℝ := 2)
  (time_descend_back : ℝ := 2)
  (avg_speed_whole_journey : ℝ := 3) :
  avg_speed_whole_journey = 3 →
  time_first_summit = 3 →
  avg_speed_whole_journey * (time_first_summit + time_descend_partially + time_second_uphill + time_descend_back) = 24 →
  avg_speed_whole_journey = 3 := 
by
  intros h_avg_speed h_time_first_summit h_total_distance
  sorry

end average_speed_to_first_summit_l98_98759


namespace volume_of_set_l98_98408

theorem volume_of_set (m n p : ℕ) (h_rel_prime : Nat.gcd n p = 1) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_p : 0 < p) 
  (h_volume : (m + n * Real.pi) / p = (324 + 37 * Real.pi) / 3) : 
  m + n + p = 364 := 
  sorry

end volume_of_set_l98_98408


namespace distance_travelled_downstream_l98_98154

def speed_boat_still_water : ℕ := 24
def speed_stream : ℕ := 4
def time_downstream : ℕ := 6

def effective_speed_downstream : ℕ := speed_boat_still_water + speed_stream
def distance_downstream : ℕ := effective_speed_downstream * time_downstream

theorem distance_travelled_downstream : distance_downstream = 168 := by
  sorry

end distance_travelled_downstream_l98_98154


namespace find_number_l98_98791

theorem find_number (N x : ℝ) (h1 : x = 1) (h2 : N / (4 + 1 / x) = 1) : N = 5 := 
by 
  sorry

end find_number_l98_98791


namespace one_sixth_time_l98_98702

-- Conditions
def total_kids : ℕ := 40
def kids_less_than_6_minutes : ℕ := total_kids * 10 / 100
def kids_less_than_8_minutes : ℕ := 3 * kids_less_than_6_minutes
def remaining_kids : ℕ := total_kids - (kids_less_than_6_minutes + kids_less_than_8_minutes)
def kids_more_than_certain_minutes : ℕ := 4
def one_sixth_remaining_kids : ℕ := remaining_kids / 6

-- Statement to prove the equivalence
theorem one_sixth_time :
  one_sixth_remaining_kids = kids_more_than_certain_minutes := 
sorry

end one_sixth_time_l98_98702


namespace natural_number_square_l98_98202

theorem natural_number_square (n : ℕ) : 
  (∃ x : ℕ, n^4 + 4 * n^3 + 5 * n^2 + 6 * n = x^2) ↔ n = 1 := 
by 
  sorry

end natural_number_square_l98_98202


namespace quadratic_real_roots_range_l98_98939

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9 / 4 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l98_98939


namespace country_X_tax_l98_98626

theorem country_X_tax (I T x : ℝ) (hI : I = 51999.99) (hT : T = 8000) (h : T = 0.14 * x + 0.20 * (I - x)) : 
  x = 39999.97 := sorry

end country_X_tax_l98_98626


namespace determine_m_l98_98034

noncomputable def f (m x : ℝ) := (m^2 - m - 1) * x^(-5 * m - 3)

theorem determine_m : ∃ m : ℝ, (∀ x > 0, f m x = (m^2 - m - 1) * x^(-5 * m - 3)) ∧ (∀ x > 0, (m^2 - m - 1) * x^(-(5 * m + 3)) = (m^2 - m - 1) * x^(-5 * m - 3) → -5 * m - 3 > 0) ∧ m = -1 :=
by
  sorry

end determine_m_l98_98034


namespace quadratic_min_n_l98_98412

theorem quadratic_min_n (m n : ℝ) : 
  (∃ x : ℝ, (x^2 + (m - 2023) * x + (n - 1)) = 0) ∧ 
  (m - 2023)^2 - 4 * (n - 1) = 0 → 
  n = 1 := 
sorry

end quadratic_min_n_l98_98412


namespace plan_y_cost_effective_l98_98966

theorem plan_y_cost_effective (m : ℕ) (h1 : ∀ minutes, cost_plan_x = 15 * minutes)
(h2 : ∀ minutes, cost_plan_y = 3000 + 10 * minutes) :
m ≥ 601 → 3000 + 10 * m < 15 * m :=
by
sorry

end plan_y_cost_effective_l98_98966


namespace perpendicular_lines_a_value_l98_98194

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ x y : ℝ, ax + y + 1 = 0) ∧ (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ x y : ℝ, (y = -ax)) → a = -1 := by
  sorry

end perpendicular_lines_a_value_l98_98194


namespace number_of_stanzas_is_correct_l98_98977

-- Define the total number of words in the poem
def total_words : ℕ := 1600

-- Define the number of lines per stanza
def lines_per_stanza : ℕ := 10

-- Define the number of words per line
def words_per_line : ℕ := 8

-- Calculate the number of words per stanza
def words_per_stanza : ℕ := lines_per_stanza * words_per_line

-- Define the number of stanzas
def stanzas (total_words words_per_stanza : ℕ) := total_words / words_per_stanza

-- Theorem: Prove that given the conditions, the number of stanzas is 20
theorem number_of_stanzas_is_correct : stanzas total_words words_per_stanza = 20 :=
by
  -- Insert the proof here
  sorry

end number_of_stanzas_is_correct_l98_98977


namespace find_principal_l98_98118

-- Define the conditions
variables (P R : ℝ) -- Define P and R as real numbers
variable (h : (P * 50) / 100 = 300) -- Introduce the equation obtained from the conditions

-- State the theorem
theorem find_principal (P R : ℝ) (h : (P * 50) / 100 = 300) : P = 600 :=
sorry

end find_principal_l98_98118


namespace maximize_expr_at_neg_5_l98_98696

-- Definition of the expression
def expr (x : ℝ) : ℝ := 1 - (x + 5) ^ 2

-- Prove that when x = -5, the expression has its maximum value
theorem maximize_expr_at_neg_5 : ∀ x : ℝ, expr x ≤ expr (-5) :=
by
  -- Placeholder for the proof
  sorry

end maximize_expr_at_neg_5_l98_98696


namespace range_of_m_for_hyperbola_l98_98778

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ (x y : ℝ), (m+2) ≠ 0 ∧ (m-2) ≠ 0 ∧ (x^2)/(m+2) + (y^2)/(m-2) = 1) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end range_of_m_for_hyperbola_l98_98778


namespace english_alphabet_is_set_l98_98961

-- Conditions definition: Elements of a set must have the properties of definiteness, distinctness, and unorderedness.
def is_definite (A : Type) : Prop := ∀ (a b : A), a = b ∨ a ≠ b
def is_distinct (A : Type) : Prop := ∀ (a b : A), a ≠ b → (a ≠ b)
def is_unordered (A : Type) : Prop := true  -- For simplicity, we assume unorderedness holds for any set

-- Property that verifies if the 26 letters of the English alphabet can form a set
def english_alphabet_set : Prop :=
  is_definite Char ∧ is_distinct Char ∧ is_unordered Char

theorem english_alphabet_is_set : english_alphabet_set :=
  sorry

end english_alphabet_is_set_l98_98961


namespace pentagon_PT_length_l98_98400

theorem pentagon_PT_length (QR RS ST : ℝ) (angle_T right_angle_QRS T : Prop) (length_PT := (fun (a b : ℝ) => a + 3 * Real.sqrt b)) :
  QR = 3 →
  RS = 3 →
  ST = 3 →
  angle_T →
  right_angle_QRS →
  (angle_Q angle_R angle_S : ℝ) →
  angle_Q = 135 →
  angle_R = 135 →
  angle_S = 135 →
  ∃ (a b : ℝ), length_PT a b = 6 * Real.sqrt 2 ∧ a + b = 2 :=
by
  sorry

end pentagon_PT_length_l98_98400


namespace sum_of_common_ratios_eq_three_l98_98135

variable (k p r a2 a3 b2 b3 : ℝ)

-- Conditions on the sequences:
variable (h_nz_k : k ≠ 0)  -- k is nonzero as it is scaling factor
variable (h_seq1 : a2 = k * p)
variable (h_seq2 : a3 = k * p^2)
variable (h_seq3 : b2 = k * r)
variable (h_seq4 : b3 = k * r^2)
variable (h_diff_ratios : p ≠ r)

-- The given equation:
variable (h_eq : a3^2 - b3^2 = 3 * (a2^2 - b2^2))

-- The theorem statement
theorem sum_of_common_ratios_eq_three :
  p^2 + r^2 = 3 :=
by
  -- Introduce the assumptions
  sorry

end sum_of_common_ratios_eq_three_l98_98135


namespace prime_divides_a_minus_3_l98_98850

theorem prime_divides_a_minus_3 (a p : ℤ) (hp : Prime p) (h1 : p ∣ 5 * a - 1) (h2 : p ∣ a - 10) : p ∣ a - 3 := by
  sorry

end prime_divides_a_minus_3_l98_98850


namespace length_of_MN_l98_98456

noncomputable def curve_eq (α : ℝ) : ℝ × ℝ := (2 * Real.cos α + 1, 2 * Real.sin α)

noncomputable def line_eq (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem length_of_MN : ∀ (M N : ℝ × ℝ), 
  M ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  N ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  M ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} ∧
  N ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} →
  dist M N = Real.sqrt 14 :=
by
  sorry

end length_of_MN_l98_98456


namespace opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l98_98361

theorem opposite_number_of_neg_two (a : Int) (h : a = -2) :
  -a = 2 := by
  sorry

theorem reciprocal_of_three (x y : Real) (hx : x = 3) (hy : y = 1 / 3) : 
  x * y = 1 := by
  sorry

theorem abs_val_three_eq (x : Real) (hx : abs x = 3) :
  x = -3 ∨ x = 3 := by
  sorry

end opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l98_98361


namespace highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l98_98613

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def sales_volume_decrease_per_yuan_increase : ℝ := 2000

-- Question 1
theorem highest_price_without_lowering_revenue :
  ∀ (x : ℝ), 
  25 ≤ x ∧ (8 - (x - original_price) * 0.2) * x ≥ 25 * 8 → 
  x ≤ 40 :=
sorry

-- Question 2
noncomputable def tech_reform_fee (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600)
noncomputable def fixed_promotion_fee : ℝ := 50
noncomputable def variable_promotion_fee (x : ℝ) : ℝ := (1 / 5) * x

theorem minimum_annual_sales_volume_and_price (x : ℝ) (a : ℝ) :
  x > 25 →
  (a * x ≥ 25 * 8 + fixed_promotion_fee + tech_reform_fee x + variable_promotion_fee x) →
  (a ≥ 10.2 ∧ x = 30) :=
sorry

end highest_price_without_lowering_revenue_minimum_annual_sales_volume_and_price_l98_98613


namespace problem_solution_l98_98605

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * x^(x^2) = 59052 :=
by
  rw [h]
  -- The condition is now x = 3
  let t := 3 + 3 * 3^(3^2)
  have : t = 59052 := sorry
  exact this

end problem_solution_l98_98605


namespace valid_punching_settings_l98_98917

theorem valid_punching_settings :
  let total_patterns := 2^9
  let symmetric_patterns := 2^6
  total_patterns - symmetric_patterns = 448 :=
by
  sorry

end valid_punching_settings_l98_98917


namespace Apollonius_circle_symmetry_l98_98953

theorem Apollonius_circle_symmetry (a : ℝ) (h : a > 1): 
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let locus_C := {P : ℝ × ℝ | ∃ x y, P = (x, y) ∧ (Real.sqrt ((x + 1)^2 + y^2) = a * Real.sqrt ((x - 1)^2 + y^2))}
  let symmetric_y := ∀ (P : ℝ × ℝ), P ∈ locus_C → (P.1, -P.2) ∈ locus_C
  symmetric_y := sorry

end Apollonius_circle_symmetry_l98_98953


namespace ratio_of_place_values_l98_98480

def thousands_place_value : ℝ := 1000
def tenths_place_value : ℝ := 0.1

theorem ratio_of_place_values : thousands_place_value / tenths_place_value = 10000 := by
  sorry

end ratio_of_place_values_l98_98480


namespace find_b_l98_98428

def passesThrough (b c : ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = P.1^2 + b * P.1 + c

theorem find_b (b c : ℝ)
  (H1 : passesThrough b c (1, 2))
  (H2 : passesThrough b c (5, 2)) :
  b = -6 :=
by
  sorry

end find_b_l98_98428


namespace arithmetic_identity_l98_98664

theorem arithmetic_identity : 45 * 27 + 73 * 45 = 4500 := by sorry

end arithmetic_identity_l98_98664


namespace speed_of_man_l98_98992

open Real Int

/-- 
  A train 110 m long is running with a speed of 40 km/h.
  The train passes a man who is running at a certain speed
  in the direction opposite to that in which the train is going.
  The train takes 9 seconds to pass the man.
  This theorem proves that the speed of the man is 3.992 km/h.
-/
theorem speed_of_man (T_length : ℝ) (T_speed : ℝ) (t_pass : ℝ) (M_speed : ℝ) : 
  T_length = 110 → T_speed = 40 → t_pass = 9 → M_speed = 3.992 :=
by
  intro h1 h2 h3
  sorry

end speed_of_man_l98_98992


namespace arithmetic_progression_power_of_two_l98_98596

theorem arithmetic_progression_power_of_two 
  (a d : ℤ) (n : ℕ) (k : ℕ) 
  (Sn : ℤ)
  (h_sum : Sn = 2^k)
  (h_ap : Sn = n * (2 * a + (n - 1) * d) / 2)  :
  ∃ m : ℕ, n = 2^m := 
sorry

end arithmetic_progression_power_of_two_l98_98596


namespace percentage_equivalence_l98_98495

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l98_98495


namespace daniella_lap_time_l98_98517

theorem daniella_lap_time
  (T_T : ℕ) (H_TT : T_T = 56)
  (meet_time : ℕ) (H_meet : meet_time = 24) :
  ∃ T_D : ℕ, T_D = 42 :=
by
  sorry

end daniella_lap_time_l98_98517


namespace expression_evaluation_l98_98233

theorem expression_evaluation (x y : ℝ) (h₁ : x > y) (h₂ : y > 0) : 
    (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x / y)^(y - x) :=
by
  sorry

end expression_evaluation_l98_98233


namespace evaporate_water_l98_98357

theorem evaporate_water (M : ℝ) (W_i W_f x : ℝ) (d : ℝ)
  (h_initial_mass : M = 500)
  (h_initial_water_content : W_i = 0.85 * M)
  (h_final_water_content : W_f = 0.75 * (M - x))
  (h_desired_fraction : d = 0.75) :
  x = 200 := 
  sorry

end evaporate_water_l98_98357


namespace avg_age_grandparents_is_64_l98_98058

-- Definitions of conditions
def num_grandparents : ℕ := 2
def num_parents : ℕ := 2
def num_grandchildren : ℕ := 3
def num_family_members : ℕ := num_grandparents + num_parents + num_grandchildren

def avg_age_parents : ℕ := 39
def avg_age_grandchildren : ℕ := 6
def avg_age_family : ℕ := 32

-- Total number of family members
theorem avg_age_grandparents_is_64 (G : ℕ) :
  (num_grandparents * G) + (num_parents * avg_age_parents) + (num_grandchildren * avg_age_grandchildren) = (num_family_members * avg_age_family) →
  G = 64 :=
by
  intro h
  sorry

end avg_age_grandparents_is_64_l98_98058


namespace jordan_rectangle_width_l98_98899

theorem jordan_rectangle_width (length_carol width_carol length_jordan width_jordan : ℝ)
  (h1: length_carol = 15) (h2: width_carol = 20) (h3: length_jordan = 6)
  (area_equal: length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 50 :=
by
  sorry

end jordan_rectangle_width_l98_98899


namespace incorrect_proposition_C_l98_98809

theorem incorrect_proposition_C (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a^4 + b^4 + c^4 + d^4 = 2 * (a^2 * b^2 + c^2 * d^2) → ¬ (a = b ∧ b = c ∧ c = d) := 
sorry

end incorrect_proposition_C_l98_98809


namespace sum_of_digits_of_number_of_rows_l98_98332

theorem sum_of_digits_of_number_of_rows :
  ∃ N, (3 * (N * (N + 1) / 2) = 1575) ∧ (Nat.digits 10 N).sum = 8 :=
by
  sorry

end sum_of_digits_of_number_of_rows_l98_98332


namespace count_valid_m_l98_98263

def is_divisor (a b : ℕ) : Prop := b % a = 0

def valid_m (m : ℕ) : Prop :=
  m > 1 ∧ is_divisor m 480 ∧ (480 / m) > 1

theorem count_valid_m : (∃ m, valid_m m) → Nat.card {m // valid_m m} = 22 :=
by sorry

end count_valid_m_l98_98263


namespace problem1_problem2_l98_98859

-- Problem 1: Prove the expression equals the calculated value
theorem problem1 : (-2:ℝ)^0 + (1 / Real.sqrt 2) - Real.sqrt 9 = (Real.sqrt 2) / 2 - 2 :=
by sorry

-- Problem 2: Prove the solution to the system of linear equations
theorem problem2 (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : x + y = -2) :
  x = 1/3 ∧ y = -(7/3) :=
by sorry

end problem1_problem2_l98_98859


namespace isosceles_triangle_l98_98877

theorem isosceles_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a + b = (Real.tan (C / 2)) * (a * Real.tan A + b * Real.tan B)) :
  A = B := 
sorry

end isosceles_triangle_l98_98877


namespace uniform_heights_l98_98122

theorem uniform_heights (varA varB : ℝ) (hA : varA = 0.56) (hB : varB = 2.1) : varA < varB := by
  rw [hA, hB]
  exact (by norm_num)

end uniform_heights_l98_98122


namespace distance_from_y_axis_l98_98910

theorem distance_from_y_axis (x : ℝ) : abs x = 10 :=
by
  -- Define distances
  let d_x := 5
  let d_y := abs x
  -- Given condition
  have h : d_x = (1 / 2) * d_y := sorry
  -- Use the given condition to prove the required statement
  sorry

end distance_from_y_axis_l98_98910


namespace total_games_played_l98_98385

-- Define the number of teams and games per matchup condition
def num_teams : ℕ := 10
def games_per_matchup : ℕ := 5

-- Calculate total games played during the season
theorem total_games_played : 
  5 * ((num_teams * (num_teams - 1)) / 2) = 225 := by 
  sorry

end total_games_played_l98_98385


namespace find_rhombus_acute_angle_l98_98740

-- Definitions and conditions
def rhombus_angle (V1 V2 : ℝ) (α : ℝ) : Prop :=
  V1 / V2 = 1 / (2 * Real.sqrt 5)
  
-- Theorem statement
theorem find_rhombus_acute_angle (V1 V2 a : ℝ) (α : ℝ) (h : rhombus_angle V1 V2 α) :
  α = Real.arccos (1 / 9) :=
sorry

end find_rhombus_acute_angle_l98_98740


namespace intersection_A1_B1_complement_A1_B1_union_A2_B2_l98_98151

-- Problem 1: Intersection and Complement
def setA1 : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def setB1 : Set ℕ := {1, 2, 3}

theorem intersection_A1_B1 : (setA1 ∩ setB1) = {1, 2, 3} := by
  sorry

theorem complement_A1_B1 : {x : ℕ | x ∈ setA1 ∧ x ∉ setB1} = {4, 5, 6, 7, 8} := by
  sorry

-- Problem 2: Union
def setA2 : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def setB2 : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_A2_B2 : (setA2 ∪ setB2) = {x : ℝ | (-3 < x ∧ x < 1) ∨ (2 < x ∧ x < 10)} := by
  sorry

end intersection_A1_B1_complement_A1_B1_union_A2_B2_l98_98151


namespace min_sum_of_primes_l98_98754

open Classical

theorem min_sum_of_primes (k m n p : ℕ) (h1 : 47 + m = k) (h2 : 53 + n = k) (h3 : 71 + p = k)
  (pm : Prime m) (pn : Prime n) (pp : Prime p) :
  m + n + p = 57 ↔ (k = 76 ∧ m = 29 ∧ n = 23 ∧ p = 5) :=
by {
  sorry
}

end min_sum_of_primes_l98_98754


namespace f_log_sum_l98_98827

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 2

theorem f_log_sum (x : ℝ) : f (Real.log 5) + f (Real.log (1 / 5)) = 4 :=
by
  sorry

end f_log_sum_l98_98827


namespace cutoff_score_admission_l98_98591

theorem cutoff_score_admission (x : ℝ) 
  (h1 : (2 / 5) * (x + 15) + (3 / 5) * (x - 20) = 90) : x = 96 :=
sorry

end cutoff_score_admission_l98_98591


namespace weeding_planting_support_l98_98477

-- Definitions based on conditions
def initial_weeding := 31
def initial_planting := 18
def additional_support := 20

-- Let x be the number of people sent to support weeding.
variable (x : ℕ)

-- The equation to prove.
theorem weeding_planting_support :
  initial_weeding + x = 2 * (initial_planting + (additional_support - x)) :=
sorry

end weeding_planting_support_l98_98477


namespace real_value_of_m_pure_imaginary_value_of_m_l98_98685

open Complex

-- Given condition
def z (m : ℝ) : ℂ := (m^2 - m : ℂ) - (m^2 - 1 : ℂ) * I

-- Part (I)
theorem real_value_of_m (m : ℝ) (h : im (z m) = 0) : m = 1 ∨ m = -1 := by
  sorry

-- Part (II)
theorem pure_imaginary_value_of_m (m : ℝ) (h1 : re (z m) = 0) (h2 : im (z m) ≠ 0) : m = 0 := by
  sorry

end real_value_of_m_pure_imaginary_value_of_m_l98_98685


namespace number_of_numbers_is_11_l98_98328

noncomputable def total_number_of_numbers 
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) : ℝ :=
if h : avg_all = 60 ∧ avg_first_6 = 58 ∧ avg_last_6 = 65 ∧ num_6th = 78 
then 11 else 0 

-- The theorem statement assuming the problem conditions
theorem number_of_numbers_is_11
  {n S : ℝ}
  (avg_all : ℝ) (avg_first_6 : ℝ) (avg_last_6 : ℝ) (num_6th : ℝ) 
  (h1 : avg_all = 60) 
  (h2 : avg_first_6 = 58)
  (h3 : avg_last_6 = 65)
  (h4 : num_6th = 78) 
  (h5 : S = 6 * avg_first_6 + 6 * avg_last_6 - num_6th)
  (h6 : S = avg_all * n) : 
  n = 11 := sorry

end number_of_numbers_is_11_l98_98328


namespace smallest_n_value_l98_98306

-- Define the given expression
def exp := (2^5) * (6^2) * (7^3) * (13^4)

-- Define the conditions
def condition_5_2 (n : ℕ) := ∃ k, n * exp = k * 5^2
def condition_3_3 (n : ℕ) := ∃ k, n * exp = k * 3^3
def condition_11_2 (n : ℕ) := ∃ k, n * exp = k * 11^2

-- Define the smallest possible value of n
def smallest_n (n : ℕ) : Prop :=
  condition_5_2 n ∧ condition_3_3 n ∧ condition_11_2 n ∧ ∀ m, (condition_5_2 m ∧ condition_3_3 m ∧ condition_11_2 m) → m ≥ n

-- The theorem statement
theorem smallest_n_value : smallest_n 9075 :=
  by
    sorry

end smallest_n_value_l98_98306


namespace sqrt_D_irrational_l98_98000

theorem sqrt_D_irrational (a b c : ℤ) (h : a + 1 = b) (h_c : c = a + b) : 
  Irrational (Real.sqrt ((a^2 : ℤ) + (b^2 : ℤ) + (c^2 : ℤ))) :=
  sorry

end sqrt_D_irrational_l98_98000


namespace train_speed_l98_98846

theorem train_speed (v : ℝ) :
  let speed_train1 := 80  -- speed of the first train in km/h
  let length_train1 := 150 / 1000 -- length of the first train in km
  let length_train2 := 100 / 1000 -- length of the second train in km
  let total_time := 5.999520038396928 / 3600 -- time in hours
  let total_length := length_train1 + length_train2 -- total length in km
  let relative_speed := total_length / total_time -- relative speed in km/h
  relative_speed = speed_train1 + v → v = 70 :=
by
  sorry

end train_speed_l98_98846


namespace grandfather_age_l98_98436

variable (F S G : ℕ)

theorem grandfather_age (h1 : F = 58) (h2 : F - S = S) (h3 : S - 5 = (1 / 2) * G) : G = 48 := by
  sorry

end grandfather_age_l98_98436


namespace find_parking_cost_l98_98023

theorem find_parking_cost :
  ∃ (C : ℝ), (C + 7 * 1.75) / 9 = 2.4722222222222223 ∧ C = 10 :=
sorry

end find_parking_cost_l98_98023


namespace correct_option_is_B_l98_98752

def satisfy_triangle_inequality (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem correct_option_is_B :
  satisfy_triangle_inequality 3 4 5 ∧
  ¬ satisfy_triangle_inequality 1 1 2 ∧
  ¬ satisfy_triangle_inequality 1 4 6 ∧
  ¬ satisfy_triangle_inequality 2 3 7 :=
by
  sorry

end correct_option_is_B_l98_98752


namespace ellipse_standard_eq_l98_98268

theorem ellipse_standard_eq
  (e : ℝ) (a b : ℝ) (h1 : e = 1 / 2) (h2 : 2 * a = 4) (h3 : b^2 = a^2 - (a * e)^2)
  : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) ↔
    ( ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ) :=
by
  sorry

end ellipse_standard_eq_l98_98268


namespace find_m_l98_98661

def setA (m : ℝ) : Set ℝ := {1, m - 2}
def setB : Set ℝ := {2}

theorem find_m (m : ℝ) (H : setA m ∩ setB = {2}) : m = 4 :=
by
  sorry

end find_m_l98_98661


namespace pole_length_is_5_l98_98401

theorem pole_length_is_5 (x : ℝ) (gate_width gate_height : ℝ) 
  (h_gate_wide : gate_width = 3) 
  (h_pole_taller : gate_height = x - 1) 
  (h_diagonal : x^2 = gate_height^2 + gate_width^2) : 
  x = 5 :=
by
  sorry

end pole_length_is_5_l98_98401


namespace hair_cut_length_l98_98139

theorem hair_cut_length (original_length after_haircut : ℕ) (h1 : original_length = 18) (h2 : after_haircut = 9) :
  original_length - after_haircut = 9 :=
by
  sorry

end hair_cut_length_l98_98139


namespace probability_of_winning_l98_98519

def probability_of_losing : ℚ := 3 / 7

theorem probability_of_winning (h : probability_of_losing + p = 1) : p = 4 / 7 :=
by 
  sorry

end probability_of_winning_l98_98519


namespace smallest_five_digit_number_divisibility_l98_98951

-- Define the smallest 5-digit number satisfying the conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_number_divisibility :
  ∃ (n : ℕ), isDivisibleBy n 15
          ∧ isDivisibleBy n (2^8)
          ∧ isDivisibleBy n 45
          ∧ isDivisibleBy n 54
          ∧ n >= 10000
          ∧ n < 100000
          ∧ n = 69120 :=
sorry

end smallest_five_digit_number_divisibility_l98_98951


namespace least_three_digit_multiple_l98_98739

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

theorem least_three_digit_multiple (n : ℕ) :
  (n >= 100) ∧ (n < 1000) ∧ (n % 36 = 0) ∧ (∀ m, (m >= 100) ∧ (m < 1000) ∧ (m % 36 = 0) → n <= m) ↔ n = 108 :=
sorry

end least_three_digit_multiple_l98_98739


namespace root_in_interval_l98_98730

noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

theorem root_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b ∧ b < 1) : 
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a b x = 0 :=
by {
  sorry
}

end root_in_interval_l98_98730


namespace ratio_a_c_l98_98391

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_c_l98_98391


namespace range_of_m_l98_98707

open Real

theorem range_of_m (a m y1 y2 : ℝ) (h_a_pos : a > 0)
  (hA : y1 = a * (m - 1)^2 + 4 * a * (m - 1) + 3)
  (hB : y2 = a * m^2 + 4 * a * m + 3)
  (h_y1_lt_y2 : y1 < y2) : 
  m > -3 / 2 := 
sorry

end range_of_m_l98_98707


namespace quadratic_has_one_real_solution_l98_98460

theorem quadratic_has_one_real_solution (m : ℝ) : (∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → m = 6 :=
by
  sorry

end quadratic_has_one_real_solution_l98_98460


namespace junior_score_l98_98736

theorem junior_score (total_students : ℕ) (juniors_percentage : ℝ) (seniors_percentage : ℝ)
  (class_average : ℝ) (senior_average : ℝ) (juniors_same_score : Prop) 
  (h1 : juniors_percentage = 0.2) (h2 : seniors_percentage = 0.8)
  (h3 : class_average = 85) (h4 : senior_average = 84) : 
  ∃ junior_score : ℝ, juniors_same_score → junior_score = 89 :=
by
  sorry

end junior_score_l98_98736


namespace count_multiples_of_four_between_100_and_350_l98_98756

-- Define the problem conditions
def is_multiple_of_four (n : ℕ) : Prop := n % 4 = 0
def in_range (n : ℕ) : Prop := 100 < n ∧ n < 350

-- Problem statement
theorem count_multiples_of_four_between_100_and_350 : 
  ∃ (k : ℕ), k = 62 ∧ ∀ n : ℕ, is_multiple_of_four n ∧ in_range n ↔ (100 < n ∧ n < 350 ∧ is_multiple_of_four n)
:= sorry

end count_multiples_of_four_between_100_and_350_l98_98756


namespace find_a_l98_98501

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a^2 - 1

theorem find_a (a : ℝ) (h : ∀ x ∈ (Set.Icc 1 2), f x a ≤ 16 ∧ ∃ y ∈ (Set.Icc 1 2), f y a = 16) : a = 3 ∨ a = -3 :=
by
  sorry

end find_a_l98_98501


namespace triangle_area_formed_by_lines_l98_98067

def line1 := { p : ℝ × ℝ | p.2 = p.1 - 4 }
def line2 := { p : ℝ × ℝ | p.2 = -p.1 - 4 }
def x_axis := { p : ℝ × ℝ | p.2 = 0 }

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_formed_by_lines :
  ∃ (A B C : ℝ × ℝ), A ∈ line1 ∧ A ∈ line2 ∧ B ∈ line1 ∧ B ∈ x_axis ∧ C ∈ line2 ∧ C ∈ x_axis ∧ 
  triangle_area A B C = 8 :=
by
  sorry

end triangle_area_formed_by_lines_l98_98067


namespace minimum_number_of_different_numbers_l98_98683

theorem minimum_number_of_different_numbers (total_numbers : ℕ) (frequent_count : ℕ) (frequent_occurrences : ℕ) (less_frequent_occurrences : ℕ) (h1 : total_numbers = 2019) (h2 : frequent_count = 10) (h3 : less_frequent_occurrences = 9) : ∃ k : ℕ, k ≥ 225 :=
by {
  sorry
}

end minimum_number_of_different_numbers_l98_98683


namespace find_k_l98_98267

def f (x : ℝ) := x^2 - 7 * x

theorem find_k : ∃ a h k : ℝ, f x = a * (x - h)^2 + k ∧ k = -49 / 4 := 
sorry

end find_k_l98_98267


namespace right_triangle_sqrt_l98_98686

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_sqrt: 
  (sqrt_2 ^ 2 + sqrt_3 ^ 2 = sqrt_5 ^ 2) :=
by
  sorry

end right_triangle_sqrt_l98_98686


namespace common_difference_l98_98805

def Sn (S : Nat → ℝ) (n : Nat) : ℝ := S n

theorem common_difference (S : Nat → ℝ) (H : Sn S 2016 / 2016 = Sn S 2015 / 2015 + 1) : 2 = 2 := 
by
  sorry

end common_difference_l98_98805


namespace increase_in_rectangle_area_l98_98925

theorem increase_in_rectangle_area (L B : ℝ) :
  let L' := 1.11 * L
  let B' := 1.22 * B
  let original_area := L * B
  let new_area := L' * B'
  let area_increase := new_area - original_area
  let percentage_increase := (area_increase / original_area) * 100
  percentage_increase = 35.42 :=
by
  sorry

end increase_in_rectangle_area_l98_98925


namespace average_student_headcount_l98_98216

theorem average_student_headcount (headcount_03_04 headcount_04_05 : ℕ) 
  (h1 : headcount_03_04 = 10500) 
  (h2 : headcount_04_05 = 10700) : 
  (headcount_03_04 + headcount_04_05) / 2 = 10600 := 
by
  sorry

end average_student_headcount_l98_98216


namespace hypotenuse_length_l98_98478

theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : a^2 + b^2 = c^2) : c = 13 :=
by
  -- proof
  sorry

end hypotenuse_length_l98_98478


namespace math_problem_l98_98825

theorem math_problem:
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 :=
by
  sorry

end math_problem_l98_98825


namespace rectangle_area_constant_l98_98744

theorem rectangle_area_constant (d : ℝ) (x : ℝ)
  (length width : ℝ)
  (h_length : length = 5 * x)
  (h_width : width = 4 * x)
  (h_diagonal : d = Real.sqrt (length ^ 2 + width ^ 2)) :
  (exists k : ℝ, k = 20 / 41 ∧ (length * width = k * d ^ 2)) :=
by
  use 20 / 41
  sorry

end rectangle_area_constant_l98_98744


namespace product_divisible_by_sum_l98_98076

theorem product_divisible_by_sum (m n : ℕ) (h : ∃ k : ℕ, m * n = k * (m + n)) : m + n ≤ Nat.gcd m n * Nat.gcd m n := by
  sorry

end product_divisible_by_sum_l98_98076


namespace alice_minimum_speed_exceed_l98_98386

-- Define the conditions

def distance_ab : ℕ := 30  -- Distance from city A to city B is 30 miles
def speed_bob : ℕ := 40    -- Bob's constant speed is 40 miles per hour
def bob_travel_time := distance_ab / speed_bob  -- Bob's travel time in hours
def alice_travel_time := bob_travel_time - (1 / 2)  -- Alice leaves 0.5 hours after Bob

-- Theorem stating the minimum speed Alice must exceed
theorem alice_minimum_speed_exceed : ∃ v : Real, v > 60 ∧ distance_ab / alice_travel_time ≤ v := sorry

end alice_minimum_speed_exceed_l98_98386


namespace simplify_radical_expression_l98_98095

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end simplify_radical_expression_l98_98095


namespace find_multiplier_l98_98111

theorem find_multiplier (x : ℤ) : 
  30 * x - 138 = 102 ↔ x = 8 := 
by
  sorry

end find_multiplier_l98_98111


namespace calculate_expression_l98_98457

theorem calculate_expression : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end calculate_expression_l98_98457


namespace number_of_teachers_at_Queen_Middle_School_l98_98815

-- Conditions
def num_students : ℕ := 1500
def classes_per_student : ℕ := 6
def classes_per_teacher : ℕ := 5
def students_per_class : ℕ := 25

-- Proof that the number of teachers is 72
theorem number_of_teachers_at_Queen_Middle_School :
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by sorry

end number_of_teachers_at_Queen_Middle_School_l98_98815


namespace minimum_a_plus_b_l98_98134

theorem minimum_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 :=
by sorry

end minimum_a_plus_b_l98_98134


namespace solve_for_angle_a_l98_98257

theorem solve_for_angle_a (a b c d e : ℝ) (h1 : a + b + c + d = 360) (h2 : e = 360 - (a + d)) : a = 360 - e - b - c :=
by
  sorry

end solve_for_angle_a_l98_98257


namespace square_root_calc_l98_98201

theorem square_root_calc (x : ℤ) (hx : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end square_root_calc_l98_98201


namespace radius_of_circle_l98_98936

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 81 * π) : r = 9 :=
by
  sorry

end radius_of_circle_l98_98936


namespace find_k_l98_98894

theorem find_k (t k : ℤ) (h1 : t = 35) (h2 : t = 5 * (k - 32) / 9) : k = 95 :=
sorry

end find_k_l98_98894


namespace image_of_center_after_transform_l98_98673

structure Point where
  x : ℤ
  y : ℤ

def reflect_across_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def translate_right (p : Point) (units : ℤ) : Point :=
  { x := p.x + units, y := p.y }

def transform_point (p : Point) : Point :=
  translate_right (reflect_across_x p) 5

theorem image_of_center_after_transform :
  transform_point {x := -3, y := 4} = {x := 2, y := -4} := by
  sorry

end image_of_center_after_transform_l98_98673


namespace problem1_problem2_l98_98179

-- Definitions of M and N
def setM : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def setN (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

-- Problem 1: Prove that if M ∩ N has only one element, then k = -1
theorem problem1 (h : ∀ x, x ∈ setM ∩ setN k → x = -1) : k = -1 := by 
  sorry

-- Problem 2: Given k = 2, prove the sets M ∩ N and M ∪ N
theorem problem2 (hk : k = 2) : (setM ∩ setN k = {x | -1 ≤ x ∧ x ≤ 2}) ∧ (setM ∪ setN k = {x | x ≤ 5}) := by
  sorry

end problem1_problem2_l98_98179


namespace billboards_and_road_length_l98_98593

theorem billboards_and_road_length :
  ∃ (x y : ℕ), 5 * (x + 21 - 1) = y ∧ (55 * (x - 1)) / 10 = y ∧ x = 200 ∧ y = 1100 :=
sorry

end billboards_and_road_length_l98_98593


namespace probability_sum_8_twice_l98_98688

-- Define a structure for the scenario: a 7-sided die.
structure Die7 :=
(sides : Fin 7)

-- Define a function to check if the sum of two dice equals 8.
def is_sum_8 (d1 d2 : Die7) : Prop :=
  (d1.sides.val + 1) + (d2.sides.val + 1) = 8

-- Define the probability of the event given the conditions.
def probability_event_twice (successes total_outcomes : ℕ) : ℚ :=
  (successes / total_outcomes) * (successes / total_outcomes)

-- The total number of outcomes when rolling two 7-sided dice.
def total_outcomes : ℕ := 7 * 7

-- The number of successful outcomes that yield a sum of 8 with two rolls.
def successful_outcomes : ℕ := 7

-- Main theorem statement to be proved.
theorem probability_sum_8_twice :
  probability_event_twice successful_outcomes total_outcomes = 1 / 49 :=
by
  -- Sorry to indicate that the proof is omitted.
  sorry

end probability_sum_8_twice_l98_98688


namespace product_of_sequence_l98_98180

theorem product_of_sequence : 
  (1 / 2) * (4 / 1) * (1 / 8) * (16 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) = 64 := 
by
  sorry

end product_of_sequence_l98_98180


namespace find_constants_l98_98777

theorem find_constants (A B : ℚ) : 
  (∀ x : ℚ, x ≠ 10 ∧ x ≠ -5 → (8 * x - 3) / (x^2 - 5 * x - 50) = A / (x - 10) + B / (x + 5)) 
  → (A = 77 / 15 ∧ B = 43 / 15) := by 
  sorry

end find_constants_l98_98777


namespace cats_weigh_more_than_puppies_l98_98060

noncomputable def weight_puppy_A : ℝ := 6.5
noncomputable def weight_puppy_B : ℝ := 7.2
noncomputable def weight_puppy_C : ℝ := 8
noncomputable def weight_puppy_D : ℝ := 9.5
noncomputable def weight_cat : ℝ := 2.8
noncomputable def num_cats : ℕ := 16

theorem cats_weigh_more_than_puppies :
  (num_cats * weight_cat) - (weight_puppy_A + weight_puppy_B + weight_puppy_C + weight_puppy_D) = 13.6 :=
by
  sorry

end cats_weigh_more_than_puppies_l98_98060


namespace pseudoprime_pow_minus_one_l98_98383

theorem pseudoprime_pow_minus_one (n : ℕ) (hpseudo : 2^n ≡ 2 [MOD n]) : 
  ∃ m : ℕ, 2^(2^n - 1) ≡ 1 [MOD (2^n - 1)] :=
by
  sorry

end pseudoprime_pow_minus_one_l98_98383


namespace line_problems_l98_98130

noncomputable def l1 : (ℝ → ℝ) := λ x => x - 1
noncomputable def l2 (k : ℝ) : (ℝ → ℝ) := λ x => -(k + 1) / k * x - 1

theorem line_problems (k : ℝ) :
  ∃ k, k = 0 → (l2 k 1) = 90 →      -- A
  (∀ k, (l1 1 = l2 k 1 → True)) →   -- B
  (∀ k, (l1 1 ≠ l2 k 1 → True)) →   -- C (negated conclusion from False in C)
  (∀ k, (l1 1 * l2 k 1 ≠ -1))       -- D
:=
sorry

end line_problems_l98_98130


namespace cost_price_per_meter_l98_98277

-- Define the given conditions
def selling_price : ℕ := 8925
def meters : ℕ := 85
def profit_per_meter : ℕ := 35

-- Define the statement to be proved
theorem cost_price_per_meter :
  (selling_price - profit_per_meter * meters) / meters = 70 := 
by
  sorry

end cost_price_per_meter_l98_98277


namespace petya_cannot_win_l98_98650

theorem petya_cannot_win (n : ℕ) (h : n ≥ 3) : ¬ ∃ strategy : ℕ → ℕ → Prop, 
  (∀ k, strategy k (k+1) ∧ strategy k (k-1))
  ∧ ∀ m, ¬ strategy n m :=
sorry

end petya_cannot_win_l98_98650


namespace minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l98_98933

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem minimum_value_when_a_is_1 : ∀ x : ℝ, ∃ m : ℝ, 
  (∀ y : ℝ, f y 1 ≥ f x 1) ∧ (f x 1 = m) :=
sorry

theorem range_of_a_given_fx_geq_0 : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 0 ≤ f x a) ↔ 1 ≤ a :=
sorry

end minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l98_98933


namespace problem_statement_l98_98960

variables (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def condition (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (2 - x)

theorem problem_statement (h_odd : is_odd f) (h_cond : condition f) : f 2010 = 0 := 
sorry

end problem_statement_l98_98960


namespace number_of_quartets_l98_98008

theorem number_of_quartets :
  let n := 5
  let factorial (x : Nat) := Nat.factorial x
  factorial n ^ 3 = 120 ^ 3 :=
by
  sorry

end number_of_quartets_l98_98008


namespace factory_days_worked_l98_98465

-- Define the number of refrigerators produced per hour
def refrigerators_per_hour : ℕ := 90

-- Define the number of coolers produced per hour
def coolers_per_hour : ℕ := refrigerators_per_hour + 70

-- Define the number of working hours per day
def working_hours_per_day : ℕ := 9

-- Define the total products produced per hour
def products_per_hour : ℕ := refrigerators_per_hour + coolers_per_hour

-- Define the total products produced in a day
def products_per_day : ℕ := products_per_hour * working_hours_per_day

-- Define the total number of products produced in given days
def total_products : ℕ := 11250

-- Define the number of days worked
def days_worked : ℕ := total_products / products_per_day

-- Prove that the number of days worked equals 5
theorem factory_days_worked : days_worked = 5 :=
by
  sorry

end factory_days_worked_l98_98465


namespace sequence_linear_constant_l98_98358

open Nat

theorem sequence_linear_constant (a : ℕ → ℕ) 
  (h1 : ∀ n, 1 < a 1 ∧ a (n + 1) > a n)
  (h2 : ∀ n, a (n + a n) = 2 * a n) :
  ∃ c : ℕ, ∀ n, a n = n + c := 
sorry

end sequence_linear_constant_l98_98358


namespace find_m_l98_98427

theorem find_m 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (h_f : ∀ x, f x = x^2 - 4*x + m)
  (h_g : ∀ x, g x = x^2 - 2*x + 2*m)
  (h_cond : 3 * f 3 = g 3)
  : m = 12 := 
sorry

end find_m_l98_98427


namespace max_k_inequality_l98_98785

theorem max_k_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) 
                                      (h₂ : 0 ≤ b) (h₃ : b ≤ 1) 
                                      (h₄ : 0 ≤ c) (h₅ : c ≤ 1) 
                                      (h₆ : 0 ≤ d) (h₇ : d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) :=
sorry

end max_k_inequality_l98_98785


namespace probability_exact_n_points_l98_98120

open Classical

noncomputable def probability_of_n_points (n : ℕ) : ℚ :=
  1/3 * (2 + (-1/2)^n)

theorem probability_exact_n_points (n : ℕ) :
  ∀ n : ℕ, probability_of_n_points n = 1/3 * (2 + (-1/2)^n) :=
sorry

end probability_exact_n_points_l98_98120


namespace probability_of_green_ball_l98_98985

-- Define the number of balls in each container
def number_balls_I := (10, 5)  -- (red, green)
def number_balls_II := (3, 6)  -- (red, green)
def number_balls_III := (3, 6)  -- (red, green)

-- Define the probability of selecting each container
noncomputable def probability_container_selected := (1 / 3 : ℝ)

-- Define the probability of drawing a green ball from each container
noncomputable def probability_green_I := (number_balls_I.snd : ℝ) / ((number_balls_I.fst + number_balls_I.snd) : ℝ)
noncomputable def probability_green_II := (number_balls_II.snd : ℝ) / ((number_balls_II.fst + number_balls_II.snd) : ℝ)
noncomputable def probability_green_III := (number_balls_III.snd : ℝ) / ((number_balls_III.fst + number_balls_III.snd) : ℝ)

-- Define the combined probabilities for drawing a green ball and selecting each container
noncomputable def combined_probability_I := probability_container_selected * probability_green_I
noncomputable def combined_probability_II := probability_container_selected * probability_green_II
noncomputable def combined_probability_III := probability_container_selected * probability_green_III

-- Define the total probability of drawing a green ball
noncomputable def total_probability_green := combined_probability_I + combined_probability_II + combined_probability_III

-- The theorem to be proved
theorem probability_of_green_ball : total_probability_green = (5 / 9 : ℝ) :=
by
  sorry

end probability_of_green_ball_l98_98985


namespace total_marbles_l98_98956

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3

theorem total_marbles : Mary_marbles + Joan_marbles = 12 :=
by
  -- Please provide the proof here if needed
  sorry

end total_marbles_l98_98956


namespace shaded_fraction_of_rectangle_l98_98659

theorem shaded_fraction_of_rectangle (a b : ℕ) (h_dim : a = 15 ∧ b = 24) (h_shaded : ∃ s, s = (1/3 : ℚ)) :
  ∃ f, f = (1/9 : ℚ) := 
by
  sorry

end shaded_fraction_of_rectangle_l98_98659


namespace abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l98_98471

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 :=
sorry

theorem neg_one_pow_2023_eq_neg_one : (-1 : ℤ) ^ 2023 = -1 :=
sorry

end abs_neg_2023_eq_2023_neg_one_pow_2023_eq_neg_one_l98_98471


namespace cost_price_watch_l98_98858

variable (cost_price : ℚ)

-- Conditions
def sold_at_loss (cost_price : ℚ) := 0.90 * cost_price
def sold_at_gain (cost_price : ℚ) := 1.03 * cost_price
def price_difference (cost_price : ℚ) := sold_at_gain cost_price - sold_at_loss cost_price = 140

-- Theorem
theorem cost_price_watch (h : price_difference cost_price) : cost_price = 1076.92 := by
  sorry

end cost_price_watch_l98_98858


namespace largest_consecutive_multiple_l98_98544

theorem largest_consecutive_multiple (n : ℕ) (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 117) : 3 * (n + 2) = 42 :=
sorry

end largest_consecutive_multiple_l98_98544


namespace factor_polynomials_l98_98199

theorem factor_polynomials (x : ℝ) :
  (x^2 + 4 * x + 3) * (x^2 + 9 * x + 20) + (x^2 + 6 * x - 9) = 
  (x^2 + 6 * x + 6) * (x^2 + 6 * x + 3) :=
sorry

end factor_polynomials_l98_98199


namespace man_l98_98062

theorem man's_speed_against_the_current (vm vc : ℝ) 
(h1: vm + vc = 15) 
(h2: vm - vc = 10) : 
vm - vc = 10 := 
by 
  exact h2

end man_l98_98062


namespace simplify_and_evaluate_l98_98571

theorem simplify_and_evaluate (x : Real) (h : x = Real.sqrt 2 - 1) :
  ( (1 / (x - 1) - 1 / (x + 1)) / (2 / (x - 1) ^ 2) ) = 1 - Real.sqrt 2 :=
by
  subst h
  sorry

end simplify_and_evaluate_l98_98571


namespace parallelogram_sides_eq_l98_98588

theorem parallelogram_sides_eq (x y : ℚ) :
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -1.6 := by
  sorry

end parallelogram_sides_eq_l98_98588


namespace Tim_running_hours_per_week_l98_98649

noncomputable def running_time_per_week : ℝ :=
  let MWF_morning : ℝ := (1 * 60 + 20 - 10) / 60 -- minutes to hours
  let MWF_evening : ℝ := (45 - 10) / 60 -- minutes to hours
  let TS_morning : ℝ := (1 * 60 + 5 - 10) / 60 -- minutes to hours
  let TS_evening : ℝ := (50 - 10) / 60 -- minutes to hours
  let MWF_total : ℝ := (MWF_morning + MWF_evening) * 3
  let TS_total : ℝ := (TS_morning + TS_evening) * 2
  MWF_total + TS_total

theorem Tim_running_hours_per_week : running_time_per_week = 8.42 := by
  -- Add the detailed proof here
  sorry

end Tim_running_hours_per_week_l98_98649


namespace prob_lfloor_XZ_YZ_product_eq_33_l98_98875

noncomputable def XZ_YZ_product : ℝ :=
  let AB := 15
  let BC := 14
  let CA := 13
  -- Definition of points and conditions
  -- Note: Specific geometric definitions and conditions need to be properly defined as per Lean's geometry library. This is a simplified placeholder.
  sorry

theorem prob_lfloor_XZ_YZ_product_eq_33 :
  (⌊XZ_YZ_product⌋ = 33) := sorry

end prob_lfloor_XZ_YZ_product_eq_33_l98_98875


namespace negation_of_proposition_l98_98540

theorem negation_of_proposition : 
    (¬ (∀ x : ℝ, x^2 - 2 * |x| ≥ 0)) ↔ (∃ x : ℝ, x^2 - 2 * |x| < 0) :=
by sorry

end negation_of_proposition_l98_98540


namespace negation_of_universal_proposition_l98_98353

theorem negation_of_universal_proposition :
  (∃ x : ℤ, x % 5 = 0 ∧ ¬ (x % 2 = 1)) ↔ ¬ (∀ x : ℤ, x % 5 = 0 → (x % 2 = 1)) :=
by sorry

end negation_of_universal_proposition_l98_98353


namespace literature_more_than_science_science_less_than_literature_percent_l98_98574

theorem literature_more_than_science (l s : ℕ) (h : 8 * s = 5 * l) : (l - s) / s = 3 / 5 :=
by {
  -- definition and given condition will be provided
  sorry
}

theorem science_less_than_literature_percent (l s : ℕ) (h : 8 * s = 5 * l) : ((l - s : ℚ) / l) * 100 = 37.5 :=
by {
  -- definition and given condition will be provided
  sorry
}

end literature_more_than_science_science_less_than_literature_percent_l98_98574


namespace find_FC_l98_98938

theorem find_FC 
  (DC CB AD AB ED FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 12)
  (h3 : AB = (1/5) * AD)
  (h4 : ED = (2/3) * AD)
  (h5 : AD = (5/4) * 22)  -- Derived step from solution for full transparency
  (h6 : FC = (ED * (CB + AB)) / AD) : 
  FC = 35 / 3 := 
sorry

end find_FC_l98_98938


namespace domain_of_f_l98_98320

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ↔ x ∈ {x : ℝ | f x = f x} :=
by
  sorry

end domain_of_f_l98_98320


namespace fourth_hexagon_dots_l98_98836

   -- Define the number of dots in the first, second, and third hexagons
   def hexagon_dots (n : ℕ) : ℕ :=
     match n with
     | 1 => 1
     | 2 => 8
     | 3 => 22
     | 4 => 46
     | _ => 0

   -- State the theorem to be proved
   theorem fourth_hexagon_dots : hexagon_dots 4 = 46 :=
   by
     sorry
   
end fourth_hexagon_dots_l98_98836


namespace parabola_expression_l98_98515

theorem parabola_expression 
  (a b : ℝ) 
  (h : 9 = a * (-2)^2 + b * (-2) + 5) : 
  2 * a - b + 6 = 8 :=
by
  sorry

end parabola_expression_l98_98515


namespace ratio_of_areas_of_triangles_l98_98597

noncomputable def area_ratio_triangle_GHI_JKL
  (a_GHI b_GHI c_GHI : ℕ) (a_JKL b_JKL c_JKL : ℕ) 
  (alt_ratio_GHI : ℕ × ℕ) (alt_ratio_JKL : ℕ × ℕ) : ℚ :=
  let area_GHI := (a_GHI * b_GHI) / 2
  let area_JKL := (a_JKL * b_JKL) / 2
  area_GHI / area_JKL

theorem ratio_of_areas_of_triangles :
  let GHI_sides := (7, 24, 25)
  let JKL_sides := (9, 40, 41)
  area_ratio_triangle_GHI_JKL 7 24 25 9 40 41 (2, 3) (4, 5) = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l98_98597


namespace determine_values_of_a_and_b_l98_98402

namespace MathProofProblem

variables (a b : ℤ)

theorem determine_values_of_a_and_b :
  (b + 1 = 2) ∧ (a - 1 ≠ -3) ∧ (a - 1 = -3) ∧ (b + 1 ≠ 2) ∧ (a - 1 = 2) ∧ (b + 1 = -3) →
  a = 3 ∧ b = -4 := by
  sorry

end MathProofProblem

end determine_values_of_a_and_b_l98_98402


namespace xinjiang_arable_land_increase_reason_l98_98393

theorem xinjiang_arable_land_increase_reason
  (global_climate_warm: Prop)
  (annual_rainfall_increase: Prop)
  (reserve_arable_land_development: Prop)
  (national_land_policies_adjustment: Prop)
  (arable_land_increased: Prop) :
  (arable_land_increased → reserve_arable_land_development) :=
sorry

end xinjiang_arable_land_increase_reason_l98_98393


namespace age_ratio_l98_98101

theorem age_ratio 
  (a b c : ℕ)
  (h1 : a = b + 2)
  (h2 : a + b + c = 32)
  (h3 : b = 12) :
  b = 2 * c :=
by
  sorry

end age_ratio_l98_98101


namespace min_major_axis_ellipse_l98_98853

theorem min_major_axis_ellipse (a b c : ℝ) (h1 : b * c = 1) (h2 : a^2 = b^2 + c^2) :
  2 * a ≥ 2 * Real.sqrt 2 :=
by {
  sorry
}

end min_major_axis_ellipse_l98_98853


namespace factorization_correct_l98_98546

theorem factorization_correct : ∀ y : ℝ, y^2 - 4*y + 4 = (y - 2)^2 := by
  intro y
  sorry

end factorization_correct_l98_98546


namespace initial_pairs_l98_98912

variable (p1 p2 p3 p4 p_initial : ℕ)

def week1_pairs := 12
def week2_pairs := week1_pairs + 4
def week3_pairs := (week1_pairs + week2_pairs) / 2
def week4_pairs := week3_pairs - 3
def total_pairs := 57

theorem initial_pairs :
  let p1 := week1_pairs
  let p2 := week2_pairs
  let p3 := week3_pairs
  let p4 := week4_pairs
  p1 + p2 + p3 + p4 + p_initial = 57 → p_initial = 4 :=
by
  sorry

end initial_pairs_l98_98912


namespace total_people_100_l98_98687

noncomputable def total_people (P : ℕ) : Prop :=
  (2 / 5 : ℚ) * P = 40 ∧ (1 / 4 : ℚ) * P ≤ P ∧ P ≥ 40 

theorem total_people_100 {P : ℕ} (h : total_people P) : P = 100 := 
by 
  sorry -- proof would go here

end total_people_100_l98_98687


namespace find_y_for_orthogonal_vectors_l98_98272

theorem find_y_for_orthogonal_vectors : 
  (∀ y, ((3:ℝ) * y + (-4:ℝ) * 9 = 0) → y = 12) :=
by
  sorry

end find_y_for_orthogonal_vectors_l98_98272


namespace trigonometric_identity_l98_98384

theorem trigonometric_identity (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 1) :
  Real.cos (2 * x) + Real.cos (2 * y) + 2 * Real.cos (2 * z) = 2 :=
by
  sorry

end trigonometric_identity_l98_98384


namespace diamond_value_l98_98442

def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem diamond_value : diamond 7 3 = 22 :=
by
  -- Proof skipped
  sorry

end diamond_value_l98_98442


namespace inequality_solution_l98_98240

theorem inequality_solution (x : ℝ) :
  (2 * x - 1 > 0 ∧ x + 1 ≤ 3) ↔ (1 / 2 < x ∧ x ≤ 2) :=
by
  sorry

end inequality_solution_l98_98240


namespace no_common_real_solution_l98_98132

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 - 6 * x + y + 9 = 0) ∧ (x^2 + 4 * y + 5 = 0) :=
by
  sorry

end no_common_real_solution_l98_98132


namespace sum_weights_greater_than_2p_l98_98181

variables (p x y l l' : ℝ)

-- Conditions
axiom balance1 : x * l = p * l'
axiom balance2 : y * l' = p * l

-- The statement to prove
theorem sum_weights_greater_than_2p : x + y > 2 * p :=
by
  sorry

end sum_weights_greater_than_2p_l98_98181


namespace log_eq_solution_l98_98212

theorem log_eq_solution (x : ℝ) (h : Real.log 8 / Real.log x = Real.log 5 / Real.log 125) : x = 512 := by
  sorry

end log_eq_solution_l98_98212


namespace standard_eq_of_parabola_l98_98372

-- Conditions:
-- The point (1, -2) lies on the parabola.
def point_on_parabola : Prop := ∃ p : ℝ, (1, -2).2^2 = 2 * p * (1, -2).1 ∨ (1, -2).1^2 = 2 * p * (1, -2).2

-- Question to be proved:
-- The standard equation of the parabola passing through the point (1, -2) is y^2 = 4x or x^2 = - (1/2) y.
theorem standard_eq_of_parabola : point_on_parabola → (y^2 = 4*x ∨ x^2 = -(1/(2:ℝ)) * y) :=
by
  sorry -- proof to be provided

end standard_eq_of_parabola_l98_98372


namespace minimum_distance_to_recover_cost_l98_98091

theorem minimum_distance_to_recover_cost 
  (initial_consumption : ℝ) (modification_cost : ℝ) (modified_consumption : ℝ) (gas_cost : ℝ) : 
  22000 < (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 ∧ 
  (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 < 26000 :=
by
  let initial_consumption := 8.4
  let modified_consumption := 6.3
  let modification_cost := 400.0
  let gas_cost := 0.80
  sorry

end minimum_distance_to_recover_cost_l98_98091


namespace card_distribution_l98_98974

-- Definitions of the total cards and distribution rules
def total_cards : ℕ := 363

def ratio_xiaoming_xiaohua (k : ℕ) : Prop := ∃ x y, x = 7 * k ∧ y = 6 * k
def ratio_xiaogang_xiaoming (m : ℕ) : Prop := ∃ x z, z = 8 * m ∧ x = 5 * m

-- Final values to prove
def xiaoming_cards : ℕ := 105
def xiaohua_cards : ℕ := 90
def xiaogang_cards : ℕ := 168

-- The proof statement
theorem card_distribution (x y z k m : ℕ) 
  (hk : total_cards = 7 * k + 6 * k + 8 * m)
  (hx : ratio_xiaoming_xiaohua k)
  (hz : ratio_xiaogang_xiaoming m) :
  x = xiaoming_cards ∧ y = xiaohua_cards ∧ z = xiaogang_cards :=
by
  -- Placeholder for the proof
  sorry

end card_distribution_l98_98974


namespace find_a_l98_98228

variable {x y a : ℝ}

theorem find_a (h1 : 2 * x - y + a ≥ 0) (h2 : 3 * x + y ≤ 3) (h3 : ∀ (x y : ℝ), 4 * x + 3 * y ≤ 8) : a = 2 := 
sorry

end find_a_l98_98228


namespace expression_value_l98_98364

theorem expression_value :
  ∀ (x y : ℚ), (x = -5/4) → (y = -3/2) → -2 * x - y^2 = 1/4 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end expression_value_l98_98364


namespace dodecagon_diagonals_l98_98305

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem dodecagon_diagonals : num_diagonals 12 = 54 :=
by
  -- by sorry means we skip the actual proof
  sorry

end dodecagon_diagonals_l98_98305


namespace sequence_product_l98_98665

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = q * a n) (h₄ : a 4 = 2) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l98_98665


namespace playground_area_l98_98029

theorem playground_area (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 100)
  (h_length : l = 3 * w) : l * w = 468.75 :=
by
  sorry

end playground_area_l98_98029


namespace final_population_correct_l98_98382

noncomputable def initialPopulation : ℕ := 300000
noncomputable def immigration : ℕ := 50000
noncomputable def emigration : ℕ := 30000

noncomputable def populationAfterImmigration : ℕ := initialPopulation + immigration
noncomputable def populationAfterEmigration : ℕ := populationAfterImmigration - emigration

noncomputable def pregnancies : ℕ := populationAfterEmigration / 8
noncomputable def twinPregnancies : ℕ := pregnancies / 4
noncomputable def singlePregnancies : ℕ := pregnancies - twinPregnancies

noncomputable def totalBirths : ℕ := twinPregnancies * 2 + singlePregnancies
noncomputable def finalPopulation : ℕ := populationAfterEmigration + totalBirths

theorem final_population_correct : finalPopulation = 370000 :=
by
  sorry

end final_population_correct_l98_98382


namespace picking_time_l98_98831

theorem picking_time (x : ℝ) 
  (h_wang : x * 8 - 0.25 = x * 7) : 
  x = 0.25 := 
by
  sorry

end picking_time_l98_98831


namespace remainder_50_pow_50_mod_7_l98_98679

theorem remainder_50_pow_50_mod_7 : (50^50) % 7 = 1 := by
  sorry

end remainder_50_pow_50_mod_7_l98_98679


namespace largest_additional_plates_l98_98856

theorem largest_additional_plates
  (initial_first_set_size : ℕ)
  (initial_second_set_size : ℕ)
  (initial_third_set_size : ℕ)
  (new_letters : ℕ)
  (constraint : 1 ≤ initial_second_set_size + 1 ∧ 1 ≤ initial_third_set_size + 1)
  (initial_combinations : ℕ)
  (final_combinations1 : ℕ)
  (final_combinations2 : ℕ)
  (additional_combinations : ℕ) :
  initial_first_set_size = 5 →
  initial_second_set_size = 3 →
  initial_third_set_size = 4 →
  new_letters = 4 →
  initial_combinations = initial_first_set_size * initial_second_set_size * initial_third_set_size →
  final_combinations1 = initial_first_set_size * (initial_second_set_size + 2) * (initial_third_set_size + 2) →
  final_combinations2 = (initial_first_set_size + 1) * (initial_second_set_size + 2) * (initial_third_set_size + 1) →
  additional_combinations = max (final_combinations1 - initial_combinations) (final_combinations2 - initial_combinations) →
  additional_combinations = 90 :=
by sorry

end largest_additional_plates_l98_98856


namespace strictly_increasing_arithmetic_seq_l98_98530

theorem strictly_increasing_arithmetic_seq 
  (s : ℕ → ℕ) 
  (hs_incr : ∀ n, s n < s (n + 1)) 
  (hs_seq1 : ∃ D1, ∀ n, s (s n) = s (s 0) + n * D1) 
  (hs_seq2 : ∃ D2, ∀ n, s (s n + 1) = s (s 0 + 1) + n * D2) : 
  ∃ d, ∀ n, s (n + 1) = s n + d :=
sorry

end strictly_increasing_arithmetic_seq_l98_98530


namespace probability_of_rolling_8_l98_98206

theorem probability_of_rolling_8 :
  let num_favorable := 5
  let num_total := 36
  let probability := (5 : ℚ) / 36
  probability =
    (num_favorable : ℚ) / num_total :=
by
  sorry

end probability_of_rolling_8_l98_98206


namespace branches_count_eq_6_l98_98534

theorem branches_count_eq_6 (x : ℕ) (h : 1 + x + x^2 = 43) : x = 6 :=
sorry

end branches_count_eq_6_l98_98534


namespace simplify_f_of_alpha_value_of_f_given_cos_l98_98980

variable (α : Real) (f : Real → Real)

def third_quadrant (α : Real) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

noncomputable def f_def : Real → Real := 
  λ α => (Real.sin (α - Real.pi / 2) * 
           Real.cos (3 * Real.pi / 2 + α) * 
           Real.tan (Real.pi - α)) / 
           (Real.tan (-α - Real.pi) * 
           Real.sin (-Real.pi - α))

theorem simplify_f_of_alpha (h : third_quadrant α) :
  f α = -Real.cos α := sorry

theorem value_of_f_given_cos 
  (h : third_quadrant α) 
  (cos_h : Real.cos (α - 3 / 2 * Real.pi) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := sorry

end simplify_f_of_alpha_value_of_f_given_cos_l98_98980


namespace minimum_seats_occupied_l98_98193

-- Define the conditions
def initial_seat_count : Nat := 150
def people_initially_leaving_up_to_two_empty_seats := true
def eventually_rule_changes_to_one_empty_seat := true

-- Define the function which checks the minimum number of occupied seats needed
def fewest_occupied_seats (total_seats : Nat) (initial_rule : Bool) (final_rule : Bool) : Nat :=
  if initial_rule && final_rule && total_seats = 150 then 57 else 0

-- The main theorem we need to prove
theorem minimum_seats_occupied {total_seats : Nat} : 
  total_seats = initial_seat_count → 
  people_initially_leaving_up_to_two_empty_seats → 
  eventually_rule_changes_to_one_empty_seat → 
  fewest_occupied_seats total_seats people_initially_leaving_up_to_two_empty_seats eventually_rule_changes_to_one_empty_seat = 57 :=
by
  intro h1 h2 h3
  sorry

end minimum_seats_occupied_l98_98193


namespace range_of_angle_B_l98_98238

theorem range_of_angle_B {A B C : ℝ} (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sinB : Real.sin B = Real.sqrt (Real.sin A * Real.sin C)) :
  0 < B ∧ B ≤ Real.pi / 3 :=
sorry

end range_of_angle_B_l98_98238


namespace total_kayaks_built_l98_98322

/-- Geometric sequence sum definition -/
def geom_sum (a r : ℕ) (n : ℕ) : ℕ :=
  if r = 1 then n * a
  else a * (r ^ n - 1) / (r - 1)

/-- Problem statement: Prove that the total number of kayaks built by the end of June is 726 -/
theorem total_kayaks_built : geom_sum 6 3 5 = 726 :=
  sorry

end total_kayaks_built_l98_98322


namespace jean_more_trips_than_bill_l98_98079

variable (b j : ℕ)

theorem jean_more_trips_than_bill
  (h1 : b + j = 40)
  (h2 : j = 23) :
  j - b = 6 := by
  sorry

end jean_more_trips_than_bill_l98_98079


namespace rectangle_length_is_4_l98_98676

theorem rectangle_length_is_4 (a : ℕ) (s : a = 4) (area_square : ℕ) 
(area_square_eq : area_square = a * a) 
(area_rectangle_eq : area_square = a * 4) : 
4 = a := by
  sorry

end rectangle_length_is_4_l98_98676


namespace total_students_in_middle_school_l98_98414

/-- Given that 20% of the students are in the band and there are 168 students in the band,
    prove that the total number of students in the middle school is 840. -/
theorem total_students_in_middle_school (total_students : ℕ) (band_students : ℕ) 
  (h1 : 20 ≤ 100)
  (h2 : band_students = 168)
  (h3 : band_students = 20 * total_students / 100) 
  : total_students = 840 :=
sorry

end total_students_in_middle_school_l98_98414


namespace evaluate_expression_l98_98143

theorem evaluate_expression : (5 * 3 ^ 4 + 6 * 4 ^ 3 = 789) :=
by
  sorry

end evaluate_expression_l98_98143


namespace totalMountainNumbers_l98_98078

-- Define a 4-digit mountain number based on the given conditions.
def isMountainNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    b > a ∧ b > d ∧ c > a ∧ c > d ∧
    a ≠ d

-- Define the main theorem stating that the total number of 4-digit mountain numbers is 1512.
theorem totalMountainNumbers : 
  ∃ n, (∀ m, isMountainNumber m → ∃ l, l = 1 ∧ 4 ≤ m ∧ m ≤ 9999) ∧ n = 1512 := sorry

end totalMountainNumbers_l98_98078


namespace find_n_l98_98342

theorem find_n (k : ℤ) : 
  ∃ n : ℤ, (n = 35 * k + 24) ∧ (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) :=
by
  -- Proof goes here
  sorry

end find_n_l98_98342


namespace inequality_not_holds_l98_98578

variable (x y : ℝ)

theorem inequality_not_holds (h1 : x > 1) (h2 : 1 > y) : x - 1 ≤ 1 - y :=
sorry

end inequality_not_holds_l98_98578


namespace intersection_C_U_M_N_l98_98557

open Set

-- Define U, M and N
def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

-- Define complement C_U M in U
def C_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem intersection_C_U_M_N : (C_U_M ∩ N) = {3} := by
  sorry

end intersection_C_U_M_N_l98_98557


namespace exchange_positions_l98_98027

theorem exchange_positions : ∀ (people : ℕ), people = 8 → (∃ (ways : ℕ), ways = 336) :=
by sorry

end exchange_positions_l98_98027


namespace constant_term_2x3_minus_1_over_sqrtx_pow_7_l98_98873

noncomputable def constant_term_in_expansion (n : ℕ) (x : ℝ) : ℝ :=
  (2 : ℝ) * (Nat.choose 7 6 : ℝ)

theorem constant_term_2x3_minus_1_over_sqrtx_pow_7 :
  constant_term_in_expansion 7 (2 : ℝ) = 14 :=
by
  -- proof is omitted
  sorry

end constant_term_2x3_minus_1_over_sqrtx_pow_7_l98_98873


namespace volume_ratio_of_cube_cut_l98_98787

/-
  The cube ABCDEFGH has its side length assumed to be 1.
  The points K, L, M divide the vertical edges AA', BB', CC'
  respectively, in the ratios 1:2, 1:3, 1:4. 
  We need to prove that the plane KLM cuts the cube into
  two parts such that the volume ratio of the two parts is 4:11.
-/
theorem volume_ratio_of_cube_cut (s : ℝ) (K L M : ℝ) :
  ∃ (Vbelow Vabove : ℝ), 
    s = 1 → 
    K = 1/3 → 
    L = 1/4 → 
    M = 1/5 → 
    Vbelow / Vabove = 4 / 11 :=
sorry

end volume_ratio_of_cube_cut_l98_98787


namespace alice_basketball_probability_l98_98666

/-- Alice and Bob play a game with a basketball. On each turn, if Alice has the basketball,
 there is a 5/8 chance that she will toss it to Bob and a 3/8 chance that she will keep the basketball.
 If Bob has the basketball, there is a 1/4 chance that he will toss it to Alice, and if he doesn't toss it to Alice,
 he keeps it. Alice starts with the basketball. What is the probability that Alice has the basketball again after two turns? -/
theorem alice_basketball_probability :
  (5 / 8) * (1 / 4) + (3 / 8) * (3 / 8) = 19 / 64 := 
by
  sorry

end alice_basketball_probability_l98_98666


namespace map_scale_to_yards_l98_98652

theorem map_scale_to_yards :
  (6.25 * 500) / 3 = 1041 + 2 / 3 := 
by sorry

end map_scale_to_yards_l98_98652


namespace alpha_eq_two_thirds_l98_98600

theorem alpha_eq_two_thirds (α : ℚ) (h1 : 0 < α) (h2 : α < 1) (h3 : Real.cos (3 * Real.pi * α) + 2 * Real.cos (2 * Real.pi * α) = 0) : α = 2 / 3 :=
sorry

end alpha_eq_two_thirds_l98_98600


namespace orange_slices_l98_98166

theorem orange_slices (x : ℕ) (hx1 : 5 * x = x + 8) : x + 2 * x + 5 * x = 16 :=
by {
  sorry
}

end orange_slices_l98_98166


namespace percentage_more_likely_to_lose_both_l98_98695

def first_lawsuit_win_probability : ℝ := 0.30
def first_lawsuit_lose_probability : ℝ := 0.70
def second_lawsuit_win_probability : ℝ := 0.50
def second_lawsuit_lose_probability : ℝ := 0.50

theorem percentage_more_likely_to_lose_both :
  (second_lawsuit_lose_probability * first_lawsuit_lose_probability - second_lawsuit_win_probability * first_lawsuit_win_probability) / (second_lawsuit_win_probability * first_lawsuit_win_probability) * 100 = 133.33 :=
by
  sorry

end percentage_more_likely_to_lose_both_l98_98695


namespace no_perfect_square_l98_98806

-- Define the given polynomial
def poly (n : ℕ) : ℤ := n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3

-- The theorem to prove
theorem no_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, poly n = k^2 := by
  sorry

end no_perfect_square_l98_98806


namespace bridge_length_proof_l98_98905

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_of_train_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) : ℝ :=
  let speed_of_train_m_per_s := speed_of_train_km_per_hr * (1000 / 3600)
  let total_distance := speed_of_train_m_per_s * time_to_cross_bridge
  total_distance - length_of_train

theorem bridge_length_proof : length_of_bridge 100 75 11.279097672186225 = 135 := by
  simp [length_of_bridge]
  sorry

end bridge_length_proof_l98_98905


namespace frank_problems_each_type_l98_98434

theorem frank_problems_each_type (bill_total : ℕ) (ryan_ratio bill_total_ratio : ℕ) (frank_ratio ryan_total : ℕ) (types : ℕ)
  (h1 : bill_total = 20)
  (h2 : ryan_ratio = 2)
  (h3 : bill_total_ratio = bill_total * ryan_ratio)
  (h4 : ryan_total = bill_total_ratio)
  (h5 : frank_ratio = 3)
  (h6 : ryan_total * frank_ratio = ryan_total) :
  (ryan_total * frank_ratio) / types = 30 :=
by
  sorry

end frank_problems_each_type_l98_98434


namespace pages_and_cost_calculation_l98_98251

noncomputable def copy_pages_cost (cents_per_5_pages : ℕ) (total_cents : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
if total_cents < discount_threshold * (cents_per_5_pages / 5) then
  total_cents / (cents_per_5_pages / 5)
else
  let num_pages_before_discount := discount_threshold
  let remaining_pages := total_cents / (cents_per_5_pages / 5) - num_pages_before_discount
  let cost_before_discount := num_pages_before_discount * (cents_per_5_pages / 5)
  let discounted_cost := remaining_pages * (cents_per_5_pages / 5) * (1 - discount_rate)
  cost_before_discount + discounted_cost

theorem pages_and_cost_calculation :
  let cents_per_5_pages := 10
  let total_cents := 5000
  let discount_threshold := 1000
  let discount_rate := 0.10
  let num_pages := (cents_per_5_pages * 2500) / 5
  let cost := copy_pages_cost cents_per_5_pages total_cents discount_threshold discount_rate
  (num_pages = 2500) ∧ (cost = 4700) :=
by
  sorry

end pages_and_cost_calculation_l98_98251


namespace sugar_cheaper_than_apples_l98_98607

/-- Given conditions about the prices and quantities of items that Fabian wants to buy,
    prove the price difference between one pack of sugar and one kilogram of apples. --/
theorem sugar_cheaper_than_apples
  (price_kg_apples : ℝ)
  (price_kg_walnuts : ℝ)
  (total_cost : ℝ)
  (cost_diff : ℝ)
  (num_kg_apples : ℕ := 5)
  (num_packs_sugar : ℕ := 3)
  (num_kg_walnuts : ℝ := 0.5)
  (price_kg_apples_val : price_kg_apples = 2)
  (price_kg_walnuts_val : price_kg_walnuts = 6)
  (total_cost_val : total_cost = 16) :
  cost_diff = price_kg_apples - (total_cost - (num_kg_apples * price_kg_apples + num_kg_walnuts * price_kg_walnuts))/num_packs_sugar → 
  cost_diff = 1 :=
by
  sorry

end sugar_cheaper_than_apples_l98_98607


namespace distinct_sequences_l98_98775

theorem distinct_sequences (N : ℕ) (α : ℝ) 
  (cond1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i * α) ≠ Int.floor (j * α)) 
  (cond2 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i / α) ≠ Int.floor (j / α)) : 
  (↑(N - 1) / ↑N : ℝ) ≤ α ∧ α ≤ (↑N / ↑(N - 1) : ℝ) := 
sorry

end distinct_sequences_l98_98775


namespace sqrt_fraction_expression_eq_one_l98_98962

theorem sqrt_fraction_expression_eq_one :
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + 1 / 6) = 1 := 
by
  sorry

end sqrt_fraction_expression_eq_one_l98_98962


namespace find_m_l98_98643

namespace MathProof

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 8

-- State the problem
theorem find_m (m : ℝ) (h : f 5 - g 5 m = 15) : m = -15 := by
  sorry

end MathProof

end find_m_l98_98643


namespace repair_cost_is_5000_l98_98325

-- Define the initial cost of the machine
def initial_cost : ℝ := 9000

-- Define the transportation charges
def transportation_charges : ℝ := 1000

-- Define the selling price
def selling_price : ℝ := 22500

-- Define the profit percentage as a decimal
def profit_percentage : ℝ := 0.5

-- Define the total cost including repairs
def total_cost (repair_cost : ℝ) : ℝ :=
  initial_cost + transportation_charges + repair_cost

-- Define the equation for selling price with 50% profit
def selling_price_equation (repair_cost : ℝ) : Prop :=
  selling_price = (1 + profit_percentage) * total_cost repair_cost

-- State the proof problem in Lean
theorem repair_cost_is_5000 : selling_price_equation 5000 :=
by 
  sorry

end repair_cost_is_5000_l98_98325


namespace find_value_of_fraction_l98_98026

theorem find_value_of_fraction (x y z : ℝ)
  (h1 : 3 * x - 4 * y - z = 0)
  (h2 : x + 4 * y - 15 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3 * x * y - y * z) / (y^2 + z^2) = 2.4 :=
by
  sorry

end find_value_of_fraction_l98_98026


namespace jellybeans_needed_l98_98867

theorem jellybeans_needed (n : ℕ) : (n ≥ 120 ∧ n % 15 = 14) → n = 134 :=
by sorry

end jellybeans_needed_l98_98867


namespace empty_plane_speed_l98_98362

variable (V : ℝ)

def speed_first_plane (V : ℝ) : ℝ := V - 2 * 50
def speed_second_plane (V : ℝ) : ℝ := V - 2 * 60
def speed_third_plane (V : ℝ) : ℝ := V - 2 * 40

theorem empty_plane_speed (V : ℝ) (h : (speed_first_plane V + speed_second_plane V + speed_third_plane V) / 3 = 500) : V = 600 :=
by 
  sorry

end empty_plane_speed_l98_98362


namespace female_employees_sampled_l98_98955

theorem female_employees_sampled
  (T : ℕ) -- Total number of employees
  (M : ℕ) -- Number of male employees
  (F : ℕ) -- Number of female employees
  (S_m : ℕ) -- Number of sampled male employees
  (H_T : T = 140)
  (H_M : M = 80)
  (H_F : F = 60)
  (H_Sm : S_m = 16) :
  ∃ S_f : ℕ, S_f = 12 :=
by
  sorry

end female_employees_sampled_l98_98955


namespace fraction_shaded_is_one_tenth_l98_98767

theorem fraction_shaded_is_one_tenth :
  ∀ (A L S: ℕ), A = 300 → L = 5 → S = 2 → 
  ((15 * 20 = A) → (A / L = 60) → (60 / S = 30) → (30 / A = 1 / 10)) :=
by sorry

end fraction_shaded_is_one_tenth_l98_98767


namespace combined_girls_avg_l98_98176

noncomputable def centralHS_boys_avg := 68
noncomputable def deltaHS_boys_avg := 78
noncomputable def combined_boys_avg := 74
noncomputable def centralHS_girls_avg := 72
noncomputable def deltaHS_girls_avg := 85
noncomputable def centralHS_combined_avg := 70
noncomputable def deltaHS_combined_avg := 80

theorem combined_girls_avg (C c D d : ℝ) 
  (h1 : (68 * C + 72 * c) / (C + c) = 70)
  (h2 : (78 * D + 85 * d) / (D + d) = 80)
  (h3 : (68 * C + 78 * D) / (C + D) = 74) :
  (3/7 * 72 + 4/7 * 85) = 79 := 
by 
  sorry

end combined_girls_avg_l98_98176


namespace admin_fee_percentage_l98_98531

noncomputable def percentage_deducted_for_admin_fees 
  (amt_johnson : ℕ) (amt_sutton : ℕ) (amt_rollin : ℕ)
  (amt_school : ℕ) (amt_after_deduction : ℕ) : ℚ :=
  ((amt_school - amt_after_deduction) * 100) / amt_school

theorem admin_fee_percentage : 
  ∃ (amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction : ℕ),
  amt_johnson = 2300 ∧
  amt_johnson = 2 * amt_sutton ∧
  amt_sutton * 8 = amt_rollin ∧
  amt_rollin * 3 = amt_school ∧
  amt_after_deduction = 27048 ∧
  percentage_deducted_for_admin_fees amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction = 2 :=
by
  sorry

end admin_fee_percentage_l98_98531


namespace cylinder_lateral_surface_area_l98_98735
noncomputable def lateralSurfaceArea (S : ℝ) : ℝ :=
  let l := Real.sqrt S
  let d := l
  let r := d / 2
  let h := l
  2 * Real.pi * r * h

theorem cylinder_lateral_surface_area (S : ℝ) (hS : S ≥ 0) : 
  lateralSurfaceArea S = Real.pi * S := by
  sorry

end cylinder_lateral_surface_area_l98_98735


namespace parabola_intersection_prob_l98_98561

noncomputable def prob_intersect_parabolas : ℚ :=
  57 / 64

theorem parabola_intersection_prob :
  ∀ (a b c d : ℤ), (1 ≤ a ∧ a ≤ 8) → (1 ≤ b ∧ b ≤ 8) →
  (1 ≤ c∧ c ≤ 8) → (1 ≤ d ∧ d ≤ 8) →
  prob_intersect_parabolas = 57 / 64 :=
by
  intros a b c d ha hb hc hd
  sorry

end parabola_intersection_prob_l98_98561


namespace quadratic_has_unique_solution_l98_98314

theorem quadratic_has_unique_solution (k : ℝ) :
  (∀ x : ℝ, (x + 6) * (x + 3) = k + 3 * x) → k = 9 :=
by
  intro h
  sorry

end quadratic_has_unique_solution_l98_98314


namespace minimum_boys_needed_l98_98440

theorem minimum_boys_needed (k n m : ℕ) (hn : n > 0) (hm : m > 0) (h : 100 * n + m * k = 10 * k) : n + m = 6 :=
by
  sorry

end minimum_boys_needed_l98_98440


namespace total_number_of_fish_l98_98635

-- Define the number of each type of fish
def goldfish : ℕ := 23
def blue_fish : ℕ := 15
def angelfish : ℕ := 8
def neon_tetra : ℕ := 12

-- Theorem stating the total number of fish
theorem total_number_of_fish : goldfish + blue_fish + angelfish + neon_tetra = 58 := by
  sorry

end total_number_of_fish_l98_98635


namespace problem_even_and_monotonically_increasing_l98_98891

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem problem_even_and_monotonically_increasing :
  is_even_function (fun x => Real.exp (|x|)) ∧ is_monotonically_increasing_on (fun x => Real.exp (|x|)) (Set.Ioo 0 1) :=
by
  sorry

end problem_even_and_monotonically_increasing_l98_98891


namespace starting_number_is_10_l98_98585

axiom between_nums_divisible_by_10 (n : ℕ) : 
  (∃ start : ℕ, start ≤ n ∧ n ≤ 76 ∧ 
  ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
  (¬ (76 % 10 = 0) → start = 10) ∧ 
  ((76 - (76 % 10)) / 10 = 6) )

theorem starting_number_is_10 
  (start : ℕ) 
  (h1 : ∃ n, (start ≤ n ∧ n ≤ 76 ∧ 
             ∀ m, start ≤ m ∧ m ≤ n → m % 10 = 0 ∧ 
             (n - start) / 10 = 6)):
  start = 10 :=
sorry

end starting_number_is_10_l98_98585


namespace meadow_income_is_960000_l98_98844

theorem meadow_income_is_960000 :
  let boxes := 30
  let packs_per_box := 40
  let diapers_per_pack := 160
  let price_per_diaper := 5
  (boxes * packs_per_box * diapers_per_pack * price_per_diaper) = 960000 := 
by
  sorry

end meadow_income_is_960000_l98_98844


namespace point_coordinates_l98_98065

noncomputable def parametric_curve (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) := (3 * Real.cos θ, 4 * Real.sin θ)

theorem point_coordinates (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) : 
  (Real.arcsin (4 * (Real.tan θ)) = π/4) → (3 * Real.cos θ, 4 * Real.sin θ) = (12 / 5, 12 / 5) :=
by
  sorry

end point_coordinates_l98_98065


namespace simple_interest_is_correct_l98_98819

def Principal : ℝ := 10000
def Rate : ℝ := 0.09
def Time : ℝ := 1

theorem simple_interest_is_correct :
  Principal * Rate * Time = 900 := by
  sorry

end simple_interest_is_correct_l98_98819


namespace maximize_profit_l98_98370

noncomputable def profit (x : ℕ) : ℝ :=
  if x ≤ 200 then
    (0.40 - 0.24) * 30 * x
  else if x ≤ 300 then
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * x - (0.24 - 0.08) * 10 * (x - 200)
  else
    (0.40 - 0.24) * 10 * 200 + (0.40 - 0.24) * 20 * 300 - (0.24 - 0.08) * 10 * (x - 200) - (0.24 - 0.08) * 20 * (x - 300)

theorem maximize_profit : ∀ x : ℕ, 
  profit 300 = 1120 ∧ (∀ y : ℕ, profit y ≤ 1120) :=
by
  sorry

end maximize_profit_l98_98370


namespace apple_cost_l98_98900

theorem apple_cost (x l q : ℝ) 
  (h1 : 10 * l = 3.62) 
  (h2 : x * l + (33 - x) * q = 11.67)
  (h3 : x * l + (36 - x) * q = 12.48) : 
  x = 30 :=
by
  sorry

end apple_cost_l98_98900


namespace painting_price_decrease_l98_98469

theorem painting_price_decrease (P : ℝ) (h1 : 1.10 * P - 0.935 * P = x * 1.10 * P) :
  x = 0.15 := by
  sorry

end painting_price_decrease_l98_98469


namespace flower_combinations_count_l98_98792

/-- Prove that there are exactly 3 combinations of tulips and sunflowers that sum up to $60,
    where tulips cost $4 each and sunflowers cost $3 each, and the number of sunflowers is greater than the number 
    of tulips. -/
theorem flower_combinations_count :
  ∃ n : ℕ, n = 3 ∧
    ∃ t s : ℕ, 4 * t + 3 * s = 60 ∧ s > t :=
by {
  sorry
}

end flower_combinations_count_l98_98792


namespace problem_equivalent_l98_98366

-- Define the problem conditions
def an (n : ℕ) : ℤ := -4 * n + 2

-- Arithmetic sequence: given conditions
axiom arith_seq_cond1 : an 2 + an 7 = -32
axiom arith_seq_cond2 : an 3 + an 8 = -40

-- Suppose the sequence {an + bn} is geometric with first term 1 and common ratio 2
def geom_seq (n : ℕ) : ℤ := 2 ^ (n - 1)
def bn (n : ℕ) : ℤ := geom_seq n - an n

-- To prove: sum of the first n terms of {bn}, denoted as Sn
def Sn (n : ℕ) : ℤ := (n * (2 + 4 * n - 2)) / 2 + (1 - 2 ^ n) / (1 - 2)

theorem problem_equivalent (n : ℕ) :
  an 2 + an 7 = -32 ∧
  an 3 + an 8 = -40 ∧
  (∀ n : ℕ, an n + bn n = geom_seq n) →
  Sn n = 2 * n ^ 2 + 2 ^ n - 1 :=
by
  intros h
  sorry

end problem_equivalent_l98_98366


namespace multiple_of_x_l98_98295

theorem multiple_of_x (k x y : ℤ) (hk : k * x + y = 34) (hx : 2 * x - y = 20) (hy : y^2 = 4) : k = 4 :=
sorry

end multiple_of_x_l98_98295


namespace petri_dish_count_l98_98615

theorem petri_dish_count (total_germs : ℝ) (germs_per_dish : ℝ) (h1 : total_germs = 0.036 * 10^5) (h2 : germs_per_dish = 199.99999999999997) :
  total_germs / germs_per_dish = 18 :=
by
  sorry

end petri_dish_count_l98_98615


namespace total_weight_of_carrots_and_cucumbers_is_875_l98_98239

theorem total_weight_of_carrots_and_cucumbers_is_875 :
  ∀ (carrots : ℕ) (cucumbers : ℕ),
    carrots = 250 →
    cucumbers = (5 * carrots) / 2 →
    carrots + cucumbers = 875 := 
by
  intros carrots cucumbers h_carrots h_cucumbers
  rw [h_carrots, h_cucumbers]
  sorry

end total_weight_of_carrots_and_cucumbers_is_875_l98_98239


namespace eggs_today_l98_98672

-- Condition definitions
def eggs_yesterday : ℕ := 10
def difference : ℕ := 59

-- Statement of the problem
theorem eggs_today : eggs_yesterday + difference = 69 := by
  sorry

end eggs_today_l98_98672


namespace prime_condition_composite_condition_l98_98192

theorem prime_condition (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a)
  (h_prime : Prime (2 * n - 1)) :
  ∃ i j : Fin n, i ≠ j ∧ ((a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) := 
sorry

theorem composite_condition (n : ℕ) (h_composite : ¬ Prime (2 * n - 1)) :
  ∃ a : Fin n → ℕ, Function.Injective a ∧ (∀ i j : Fin n, i ≠ j → ((a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1)) := 
sorry

end prime_condition_composite_condition_l98_98192


namespace math_problem_l98_98940

noncomputable def a (b : ℝ) : ℝ := 
  sorry -- to be derived from the conditions

noncomputable def b : ℝ := 
  sorry -- to be derived from the conditions

theorem math_problem (a b: ℝ) 
  (h1: a - b = 1)
  (h2: a^2 - b^2 = -1) : 
  a^2008 - b^2008 = -1 := 
sorry

end math_problem_l98_98940


namespace range_a_f_x_neg_l98_98847

noncomputable def f (a x : ℝ) : ℝ := x^2 + (2 * a - 1) * x - 3

theorem range_a_f_x_neg (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ f a x < 0) → a < 3 / 2 := sorry

end range_a_f_x_neg_l98_98847


namespace amount_paid_for_grapes_l98_98053

-- Definitions based on the conditions
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := 2.23

-- The statement to be proved
theorem amount_paid_for_grapes : total_spent + refund_for_cherries = 12.08 := 
by 
  -- Here the specific mathematical proof would go, but is replaced by sorry as instructed
  sorry

end amount_paid_for_grapes_l98_98053


namespace chairs_per_row_l98_98533

/-- There are 10 rows of chairs, with the first row for awardees, the second and third rows for
    administrators and teachers, the last two rows for parents, and the remaining five rows for students.
    Given that 4/5 of the student seats are occupied, and there are 15 vacant seats among the students,
    proves that the number of chairs per row is 15. --/
theorem chairs_per_row (x : ℕ) (h1 : 10 = 1 + 1 + 1 + 5 + 2)
  (h2 : 4 / 5 * (5 * x) + 1 / 5 * (5 * x) = 5 * x)
  (h3 : 1 / 5 * (5 * x) = 15) : x = 15 :=
sorry

end chairs_per_row_l98_98533


namespace shaded_region_area_l98_98165

open Real

noncomputable def area_of_shaded_region (side : ℝ) (radius : ℝ) : ℝ :=
  let area_square := side ^ 2
  let area_sector := π * radius ^ 2 / 4
  let area_triangle := (1 / 2) * (side / 2) * sqrt ((side / 2) ^ 2 - radius ^ 2)
  area_square - 8 * area_triangle - 4 * area_sector

theorem shaded_region_area (h_side : ℝ) (h_radius : ℝ)
  (h1 : h_side = 8) (h2 : h_radius = 3) :
  area_of_shaded_region h_side h_radius = 64 - 16 * sqrt 7 - 3 * π :=
by
  rw [h1, h2]
  sorry

end shaded_region_area_l98_98165


namespace number_of_games_in_complete_season_l98_98849

-- Define the number of teams in each division
def teams_in_division_A : Nat := 6
def teams_in_division_B : Nat := 7
def teams_in_division_C : Nat := 5

-- Define the number of games each team must play within their division
def games_per_team_within_division (teams : Nat) : Nat :=
  (teams - 1) * 2

-- Calculate the total number of games within a division
def total_games_within_division (teams : Nat) : Nat :=
  (games_per_team_within_division teams * teams) / 2

-- Calculate cross-division games for a team in one division
def cross_division_games_per_team (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  (teams_other_div1 + teams_other_div2) * 2

-- Calculate total cross-division games from all teams in one division
def total_cross_division_games (teams_div : Nat) (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  cross_division_games_per_team teams_other_div1 teams_other_div2 * teams_div

-- Given conditions translated to definitions
def games_in_division_A : Nat := total_games_within_division teams_in_division_A
def games_in_division_B : Nat := total_games_within_division teams_in_division_B
def games_in_division_C : Nat := total_games_within_division teams_in_division_C

def cross_division_games_A : Nat := total_cross_division_games teams_in_division_A teams_in_division_B teams_in_division_C
def cross_division_games_B : Nat := total_cross_division_games teams_in_division_B teams_in_division_A teams_in_division_C
def cross_division_games_C : Nat := total_cross_division_games teams_in_division_C teams_in_division_A teams_in_division_B

-- Total cross-division games with each game counted twice
def total_cross_division_games_in_season : Nat :=
  (cross_division_games_A + cross_division_games_B + cross_division_games_C) / 2

-- Total number of games in the season
def total_games_in_season : Nat :=
  games_in_division_A + games_in_division_B + games_in_division_C + total_cross_division_games_in_season

-- The final proof statement
theorem number_of_games_in_complete_season : total_games_in_season = 306 :=
by
  -- This is the place where the proof would go if it were required.
  sorry

end number_of_games_in_complete_season_l98_98849


namespace power_cycle_i_l98_98997

theorem power_cycle_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^23 + i^75 = -2 * i :=
by
  sorry

end power_cycle_i_l98_98997


namespace products_arrangement_count_l98_98208

/--
There are five different products: A, B, C, D, and E arranged in a row on a shelf.
- Products A and B must be adjacent.
- Products C and D must not be adjacent.
Prove that there are a total of 24 distinct valid arrangements under these conditions.
-/
theorem products_arrangement_count : 
  ∃ (n : ℕ), 
  (∀ (A B C D E : Type), n = 24 ∧
  ∀ l : List (Type), l = [A, B, C, D, E] ∧
  -- A and B must be adjacent
  (∀ p : List (Type), p = [A, B] ∨ p = [B, A]) ∧
  -- C and D must not be adjacent
  ¬ (∀ q : List (Type), q = [C, D] ∨ q = [D, C])) :=
sorry

end products_arrangement_count_l98_98208


namespace mans_rate_in_still_water_l98_98164

/-- The man's rowing speed in still water given his rowing speeds with and against the stream. -/
theorem mans_rate_in_still_water (v_with_stream v_against_stream : ℝ) (h1 : v_with_stream = 6) (h2 : v_against_stream = 2) : (v_with_stream + v_against_stream) / 2 = 4 := by
  sorry

end mans_rate_in_still_water_l98_98164


namespace roots_of_quadratic_expression_l98_98215

theorem roots_of_quadratic_expression :
    (∀ x: ℝ, (x^2 + 3 * x - 2 = 0) → ∃ x₁ x₂: ℝ, x = x₁ ∨ x = x₂) ∧ 
    (∀ x₁ x₂ : ℝ, (x₁ + x₂ = -3) ∧ (x₁ * x₂ = -2) → x₁^2 + 2 * x₁ - x₂ = 5) :=
by
  sorry

end roots_of_quadratic_expression_l98_98215


namespace aunt_angela_nieces_l98_98253

theorem aunt_angela_nieces (total_jellybeans : ℕ)
                           (jellybeans_per_child : ℕ)
                           (num_nephews : ℕ)
                           (num_nieces : ℕ) 
                           (total_children : ℕ) 
                           (h1 : total_jellybeans = 70)
                           (h2 : jellybeans_per_child = 14)
                           (h3 : num_nephews = 3)
                           (h4 : total_children = total_jellybeans / jellybeans_per_child)
                           (h5 : total_children = num_nephews + num_nieces) :
                           num_nieces = 2 :=
by
  sorry

end aunt_angela_nieces_l98_98253


namespace digit_difference_l98_98113

theorem digit_difference (X Y : ℕ) (h1 : 10 * X + Y - (10 * Y + X) = 36) : X - Y = 4 := by
  sorry

end digit_difference_l98_98113


namespace solve_for_y_l98_98690

theorem solve_for_y (y : ℝ) (h : (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1) : y = 4.5 :=
by
  -- Proof goes here
  sorry

end solve_for_y_l98_98690


namespace find_m_value_l98_98017

-- Definitions of the given lines
def l1 (x y : ℝ) (m : ℝ) : Prop := x + m * y + 6 = 0
def l2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Parallel lines condition
def parallel (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m = l2 x y m

-- Proof that the value of m for the lines to be parallel is indeed -1
theorem find_m_value : parallel (-1) :=
by
  sorry

end find_m_value_l98_98017


namespace points_collinear_sum_l98_98926

theorem points_collinear_sum (x y : ℝ) :
  ∃ k : ℝ, (x - 1 = 3 * k ∧ 1 = k * (y - 2) ∧ -1 = 2 * k) → 
  x + y = -1 / 2 :=
by
  sorry

end points_collinear_sum_l98_98926


namespace negation_of_universal_l98_98553

theorem negation_of_universal : (¬ ∀ x : ℝ, x^2 + 2 * x - 1 = 0) ↔ ∃ x : ℝ, x^2 + 2 * x - 1 ≠ 0 :=
by sorry

end negation_of_universal_l98_98553


namespace geometric_sum_of_ratios_l98_98598

theorem geometric_sum_of_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ) 
  (ha2 : a2 = k * p) (ha3 : a3 = k * p^2) 
  (hb2 : b2 = k * r) (hb3 : b3 = k * r^2) 
  (h : a3 - b3 = 5 * (a2 - b2)) :
  p + r = 5 :=
by {
  sorry
}

end geometric_sum_of_ratios_l98_98598


namespace probability_positive_ball_drawn_is_half_l98_98310

-- Definition of the problem elements
def balls : List Int := [-1, 0, 2, 3]

-- Definition for the event of drawing a positive number
def is_positive (x : Int) : Bool := x > 0

-- The proof statement
theorem probability_positive_ball_drawn_is_half : 
  (List.filter is_positive balls).length / balls.length = 1 / 2 :=
by
  sorry

end probability_positive_ball_drawn_is_half_l98_98310


namespace correct_substitution_l98_98976

theorem correct_substitution (x : ℝ) : 
    (2 * x - 7)^2 + (5 * x - 17.5)^2 = 0 → 
    x = 7 / 2 :=
by
  sorry

end correct_substitution_l98_98976


namespace option_C_cannot_form_right_triangle_l98_98902

def is_right_triangle_sides (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem option_C_cannot_form_right_triangle :
  ¬ (is_right_triangle_sides 1.5 2 3) :=
by
  -- This is intentionally left incomplete as per instructions
  sorry

end option_C_cannot_form_right_triangle_l98_98902


namespace not_perfect_power_l98_98242

theorem not_perfect_power (k : ℕ) (h : k ≥ 2) : ∀ m n : ℕ, m > 1 → n > 1 → 10^k - 1 ≠ m ^ n :=
by 
  sorry

end not_perfect_power_l98_98242


namespace sum_of_common_ratios_l98_98318

theorem sum_of_common_ratios (k p r a2 a3 b2 b3 : ℝ)
  (h1 : a3 = k * p^2) (h2 : a2 = k * p) 
  (h3 : b3 = k * r^2) (h4 : b2 = k * r)
  (h5 : p ≠ r)
  (h6 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2)) :
  p + r = 5 :=
by {
  sorry
}

end sum_of_common_ratios_l98_98318


namespace circle_area_from_circumference_l98_98142

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end circle_area_from_circumference_l98_98142


namespace students_with_equal_scores_l98_98662

theorem students_with_equal_scores 
  (n : ℕ)
  (scores : Fin n → Fin (n - 1)): 
  ∃ i j : Fin n, i ≠ j ∧ scores i = scores j := 
by 
  sorry

end students_with_equal_scores_l98_98662


namespace complement_of_A_in_U_l98_98502

noncomputable def U : Set ℤ := {-3, -1, 0, 1, 3}

noncomputable def A : Set ℤ := {x | x^2 - 2 * x - 3 = 0}

theorem complement_of_A_in_U : (U \ A) = {-3, 0, 1} :=
by sorry

end complement_of_A_in_U_l98_98502


namespace bobs_total_profit_l98_98373

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l98_98373


namespace triangle_ABC_properties_l98_98461

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem triangle_ABC_properties
  (xA xB xC : ℝ)
  (h_seq : xA < xB ∧ xB < xC ∧ 2 * xB = xA + xC)
  : (f xB + (f xA + f xC) / 2 > f ((xA + xC) / 2)) ∧ (f xA ≠ f xB ∧ f xB ≠ f xC) := 
sorry

end triangle_ABC_properties_l98_98461


namespace ratio_female_to_male_l98_98262

-- Definitions for the conditions
def average_age_female (f : ℕ) : ℕ := 40 * f
def average_age_male (m : ℕ) : ℕ := 25 * m
def average_age_total (f m : ℕ) : ℕ := (30 * (f + m))

-- Statement to prove
theorem ratio_female_to_male (f m : ℕ) 
  (h_avg_f: average_age_female f = 40 * f)
  (h_avg_m: average_age_male m = 25 * m)
  (h_avg_total: average_age_total f m = 30 * (f + m)) : 
  f / m = 1 / 2 :=
by
  sorry

end ratio_female_to_male_l98_98262


namespace question1_question2_question3_l98_98375

def f : Nat → Nat → Nat := sorry

axiom condition1 : f 1 1 = 1
axiom condition2 : ∀ m n, f m (n + 1) = f m n + 2
axiom condition3 : ∀ m, f (m + 1) 1 = 2 * f m 1

theorem question1 (n : Nat) : f 1 n = 2 * n - 1 :=
sorry

theorem question2 (m : Nat) : f m 1 = 2 ^ (m - 1) :=
sorry

theorem question3 : f 2002 9 = 2 ^ 2001 + 16 :=
sorry

end question1_question2_question3_l98_98375


namespace seq_inequality_l98_98285

noncomputable def sequence_of_nonneg_reals (a : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, a (n + m) ≤ a n + a m

theorem seq_inequality
  (a : ℕ → ℝ)
  (h : sequence_of_nonneg_reals a)
  (h_nonneg : ∀ n, 0 ≤ a n) :
  ∀ n m : ℕ, m > 0 → n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := 
by
  sorry

end seq_inequality_l98_98285


namespace inequality_proof_l98_98742

theorem inequality_proof
  (x y z : ℝ) (hxpos : 0 < x) (hypos : 0 < y) (hzpos : 0 < z)
  (hineq : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
sorry

end inequality_proof_l98_98742


namespace shaded_triangle_probability_l98_98001

noncomputable def total_triangles : ℕ := 5
noncomputable def shaded_triangles : ℕ := 2
noncomputable def probability_shaded : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : probability_shaded = 2 / 5 :=
by
  sorry

end shaded_triangle_probability_l98_98001


namespace actual_discount_is_expected_discount_l98_98168

-- Define the conditions
def promotional_discount := 20 / 100  -- 20% discount
def vip_card_discount := 10 / 100  -- 10% additional discount

-- Define the combined discount calculation
def combined_discount := (1 - promotional_discount) * (1 - vip_card_discount)

-- Define the expected discount off the original price
def expected_discount := 28 / 100  -- 28% discount

-- Theorem statement proving the combined discount is equivalent to the expected discount
theorem actual_discount_is_expected_discount :
  combined_discount = 1 - expected_discount :=
by
  -- Proof omitted.
  sorry

end actual_discount_is_expected_discount_l98_98168


namespace find_coordinates_of_P_l98_98221

/-- Let the curve C be defined by the equation y = x^3 - 10x + 3 and point P lies on this curve in the second quadrant.
We are given that the slope of the tangent line to the curve at point P is 2. We need to find the coordinates of P.
--/
theorem find_coordinates_of_P :
  ∃ (x y : ℝ), (y = x ^ 3 - 10 * x + 3) ∧ (3 * x ^ 2 - 10 = 2) ∧ (x < 0) ∧ (x = -2) ∧ (y = 15) :=
by
  sorry

end find_coordinates_of_P_l98_98221


namespace geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l98_98056

def seq_an : ℕ → ℝ := sorry
def sum_Sn : ℕ → ℝ := sorry

axiom Sn_recurrence (n : ℕ) : sum_Sn (n + 1) = (1/2) * sum_Sn n + 2
axiom a1_def : seq_an 1 = 2
axiom a2_def : seq_an 2 = 1

theorem geometric_seq (n : ℕ) : ∃ r : ℝ, ∀ (m : ℕ), sum_Sn m - 4 = (sum_Sn 1 - 4) * r^(m-1) := 
sorry

theorem an_formula (n : ℕ) : seq_an n = (1/2)^(n-2) := 
sorry

theorem inequality_proof (t n : ℕ) (t_pos : 0 < t) : 
  (seq_an t * sum_Sn (n + 1) - 1) / (seq_an t * seq_an (n + 1) - 1) < 1/2 :=
sorry

theorem find_t : ∃ (t : ℕ), t = 3 ∨ t = 4 := 
sorry

theorem sum_not_in_seq (m n k : ℕ) (distinct : k ≠ m ∧ m ≠ n ∧ k ≠ n) : 
  (seq_an m + seq_an n ≠ seq_an k) :=
sorry

end geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l98_98056


namespace symmetrical_parabola_eq_l98_98497

/-- 
  Given a parabola y = (x-1)^2 + 3, prove that its symmetrical parabola 
  about the x-axis is y = -(x-1)^2 - 3.
-/
theorem symmetrical_parabola_eq (x : ℝ) : 
  (x-1)^2 + 3 = -(x-1)^2 - 3 ↔ y = -(x-1)^2 - 3 := 
sorry

end symmetrical_parabola_eq_l98_98497


namespace initial_garrison_men_l98_98821

theorem initial_garrison_men (M : ℕ) (h1 : 62 * M = 62 * M) 
  (h2 : M * 47 = (M + 2700) * 20) : M = 2000 := by
  sorry

end initial_garrison_men_l98_98821


namespace arithmetic_mean_q_r_l98_98167

theorem arithmetic_mean_q_r (p q r : ℝ) (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) (h3 : r - p = 34) : (q + r) / 2 = 27 :=
sorry

end arithmetic_mean_q_r_l98_98167


namespace exists_cubic_polynomial_with_cubed_roots_l98_98474

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * x + 4

-- Statement that we need to prove
theorem exists_cubic_polynomial_with_cubed_roots :
  ∃ (b c d : ℝ), ∀ (x : ℝ),
  (f x = 0) → (x^3 = y → x^3^3 + b * x^3^2 + c * x^3 + d = 0) :=
sorry

end exists_cubic_polynomial_with_cubed_roots_l98_98474


namespace simplify_expression_l98_98107

theorem simplify_expression :
  ((0.3 * 0.2) / (0.4 * 0.5)) - (0.1 * 0.6) = 0.24 :=
by
  sorry

end simplify_expression_l98_98107


namespace eval_expression_l98_98507

theorem eval_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / (a * b) - (a^2 + a * b) / (a^2 + b^2) = (a^4 + b^4 + a^2 * b^2 - a^2 * b - a * b^2) / (a * b * (a^2 + b^2)) :=
by
  sorry

end eval_expression_l98_98507


namespace quadratic_roots_eq1_quadratic_roots_eq2_l98_98389

theorem quadratic_roots_eq1 :
  ∀ x : ℝ, (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
by
  intros x
  sorry

theorem quadratic_roots_eq2 :
  ∀ x : ℝ, ((x + 2)^2 = (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intros x
  sorry

end quadratic_roots_eq1_quadratic_roots_eq2_l98_98389


namespace intersection_M_P_l98_98682

variable {x a : ℝ}

def M (a : ℝ) : Set ℝ := { x | x > a ∧ a^2 - 12*a + 20 < 0 }
def P : Set ℝ := { x | x ≤ 10 }

theorem intersection_M_P (a : ℝ) (h : 2 < a ∧ a < 10) : 
  M a ∩ P = { x | a < x ∧ x ≤ 10 } :=
sorry

end intersection_M_P_l98_98682


namespace complement_M_l98_98094

open Set

-- Define the universal set U as the set of all real numbers
def U := ℝ

-- Define the set M as {x | |x| > 2}
def M : Set ℝ := {x | |x| > 2}

-- State that the complement of M (in the universal set U) is [-2, 2]
theorem complement_M : Mᶜ = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

end complement_M_l98_98094


namespace find_integer_m_l98_98886

theorem find_integer_m (m : ℤ) :
  (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2) → m = 4 :=
by
  intro h
  sorry

end find_integer_m_l98_98886


namespace exists_n_of_form_2k_l98_98018

theorem exists_n_of_form_2k (n : ℕ) (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_recip : 1/x + 1/y + 1/z = 1/(n : ℤ)) : ∃ k : ℕ, n = 2 * k :=
sorry

end exists_n_of_form_2k_l98_98018


namespace percentage_of_temporary_workers_l98_98506

theorem percentage_of_temporary_workers (total_workers technicians non_technicians permanent_technicians permanent_non_technicians : ℕ) 
  (h1 : total_workers = 100)
  (h2 : technicians = total_workers / 2) 
  (h3 : non_technicians = total_workers / 2) 
  (h4 : permanent_technicians = technicians / 2) 
  (h5 : permanent_non_technicians = non_technicians / 2) :
  ((total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers) * 100 = 50 :=
by
  sorry

end percentage_of_temporary_workers_l98_98506


namespace mary_stickers_left_l98_98595

def initial_stickers : ℕ := 50
def stickers_per_friend : ℕ := 4
def number_of_friends : ℕ := 5
def total_students_including_mary : ℕ := 17
def stickers_per_other_student : ℕ := 2

theorem mary_stickers_left :
  let friends_stickers := stickers_per_friend * number_of_friends
  let other_students := total_students_including_mary - 1 - number_of_friends
  let other_students_stickers := stickers_per_other_student * other_students
  let total_given_away := friends_stickers + other_students_stickers
  initial_stickers - total_given_away = 8 :=
by
  sorry

end mary_stickers_left_l98_98595


namespace mother_younger_than_father_l98_98106

variable (total_age : ℕ) (father_age : ℕ) (brother_age : ℕ) (sister_age : ℕ) (kaydence_age : ℕ) (mother_age : ℕ)

noncomputable def family_data : Prop :=
  total_age = 200 ∧
  father_age = 60 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  kaydence_age = 12 ∧
  mother_age = total_age - (father_age + brother_age + sister_age + kaydence_age)

theorem mother_younger_than_father :
  family_data total_age father_age brother_age sister_age kaydence_age mother_age →
  father_age - mother_age = 2 :=
sorry

end mother_younger_than_father_l98_98106


namespace tan_shifted_value_l98_98197

theorem tan_shifted_value (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = -((6 + 5 * Real.sqrt 3) / 13) :=
by
  sorry

end tan_shifted_value_l98_98197


namespace violet_prob_l98_98927

noncomputable def total_candies := 8 + 5 + 9 + 10 + 6

noncomputable def prob_green_first := (8 : ℚ) / total_candies
noncomputable def prob_yellow_second := (10 : ℚ) / (total_candies - 1)
noncomputable def prob_pink_third := (6 : ℚ) / (total_candies - 2)

noncomputable def combined_prob := prob_green_first * prob_yellow_second * prob_pink_third

theorem violet_prob :
  combined_prob = (20 : ℚ) / 2109 := by
    sorry

end violet_prob_l98_98927


namespace retailer_overhead_expenses_l98_98582

theorem retailer_overhead_expenses (purchase_price selling_price profit_percent : ℝ) (overhead_expenses : ℝ) 
  (h1 : purchase_price = 225) 
  (h2 : selling_price = 300) 
  (h3 : profit_percent = 25) 
  (h4 : selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100)) : 
  overhead_expenses = 15 := 
by
  sorry

end retailer_overhead_expenses_l98_98582


namespace total_jelly_beans_l98_98013

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end total_jelly_beans_l98_98013


namespace no_valid_k_exists_l98_98405

theorem no_valid_k_exists {k : ℕ} : ¬(∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p + q = 41 ∧ p * q = k) :=
by
  sorry

end no_valid_k_exists_l98_98405


namespace greatest_integer_b_for_no_real_roots_l98_98954

theorem greatest_integer_b_for_no_real_roots :
  ∃ (b : ℤ), (b * b < 20) ∧ (∀ (c : ℤ), (c * c < 20) → c ≤ 4) :=
by
  sorry

end greatest_integer_b_for_no_real_roots_l98_98954


namespace jack_bought_apples_l98_98271

theorem jack_bought_apples :
  ∃ n : ℕ, 
    (∃ k : ℕ, k = 10 ∧ ∃ m : ℕ, m = 5 * 9 ∧ n = k + m) ∧ n = 55 :=
by
  sorry

end jack_bought_apples_l98_98271


namespace sin_identity_proof_l98_98604

theorem sin_identity_proof (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) :
  Real.sin (5 * π / 6 - x) + Real.sin (π / 3 - x) ^ 2 = 19 / 16 :=
by
  sorry

end sin_identity_proof_l98_98604


namespace perimeter_of_playground_l98_98610

theorem perimeter_of_playground 
  (x y : ℝ) 
  (h1 : x^2 + y^2 = 900) 
  (h2 : x * y = 216) : 
  2 * (x + y) = 72 := 
by 
  sorry

end perimeter_of_playground_l98_98610


namespace sum_of_solutions_l98_98729

theorem sum_of_solutions (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (∃ x₁ x₂ : ℝ, (3 * x₁ + 2) * (x₁ - 4) = 0 ∧ (3 * x₂ + 2) * (x₂ - 4) = 0 ∧
  x₁ ≠ 1 ∧ x₁ ≠ -1 ∧ x₂ ≠ 1 ∧ x₂ ≠ -1 ∧ x₁ + x₂ = 10 / 3) :=
sorry

end sum_of_solutions_l98_98729


namespace maximize_log_power_l98_98789

theorem maximize_log_power (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (hab : a * b = 100) :
  ∃ x : ℝ, (a ^ (Real.logb 10 b)^2 = 10^x) ∧ x = 32 / 27 :=
by
  sorry

end maximize_log_power_l98_98789


namespace find_a_l98_98920

-- Define the function f(x)
def f (a : ℚ) (x : ℚ) : ℚ := x^2 + (2 * a + 3) * x + (a^2 + 1)

-- State that the discriminant of f(x) is non-negative
def discriminant_nonnegative (a : ℚ) : Prop :=
  let Δ := (2 * a + 3)^2 - 4 * (a^2 + 1)
  Δ ≥ 0

-- Final statement expressing the final condition on a and the desired result |p| + |q|
theorem find_a (a : ℚ) (p q : ℤ) (h_relprime : Int.gcd p q = 1) (h_eq : a = -5 / 12) (h_abs : p * q = -5 * 12) :
  discriminant_nonnegative a →
  |p| + |q| = 17 :=
by sorry

end find_a_l98_98920


namespace triangle_properties_l98_98012

open Real

noncomputable def vec_m (a : ℝ) : ℝ × ℝ := (2 * sin (a / 2), sqrt 3)
noncomputable def vec_n (a : ℝ) : ℝ × ℝ := (cos a, 2 * cos (a / 4)^2 - 1)
noncomputable def area_triangle := 3 * sqrt 3 / 2

theorem triangle_properties (a b c : ℝ) (A : ℝ)
  (ha : a = sqrt 7)
  (hA : (1 / 2) * b * c * sin A = area_triangle)
  (hparallel : vec_m A = vec_n A) :
  A = π / 3 ∧ b + c = 5 :=
by
  sorry

end triangle_properties_l98_98012


namespace car_speed_l98_98710

/-- Given a car covers a distance of 624 km in 2 3/5 hours,
    prove that the speed of the car is 240 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ)
  (h_distance : distance = 624)
  (h_time : time = 13 / 5) :
  (distance / time) = 240 :=
by
  sorry

end car_speed_l98_98710


namespace parabola_conditions_l98_98547

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end parabola_conditions_l98_98547


namespace gcd_12012_21021_l98_98732

-- Definitions
def factors_12012 : List ℕ := [2, 2, 3, 7, 11, 13] -- Factors of 12,012
def factors_21021 : List ℕ := [3, 7, 7, 11, 13] -- Factors of 21,021

def common_factors := [3, 7, 11, 13] -- Common factors between 12,012 and 21,021

def gcd (ls : List ℕ) : ℕ :=
ls.foldr Nat.gcd 0 -- Function to calculate gcd of list of numbers

-- Main statement
theorem gcd_12012_21021 : gcd common_factors = 1001 := by
  -- Proof is not required, so we use sorry to skip the proof.
  sorry

end gcd_12012_21021_l98_98732


namespace hyperbola_slope_of_asymptote_positive_value_l98_98901

noncomputable def hyperbola_slope_of_asymptote (x y : ℝ) : ℝ :=
  if h : (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4)
  then (Real.sqrt 5) / 2
  else 0

-- Statement of the mathematically equivalent proof problem
theorem hyperbola_slope_of_asymptote_positive_value :
  ∃ x y : ℝ, (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) ∧
  hyperbola_slope_of_asymptote x y = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_slope_of_asymptote_positive_value_l98_98901


namespace isosceles_triangle_base_length_l98_98468

theorem isosceles_triangle_base_length :
  ∃ (x y : ℝ), 
    ((x + x / 2 = 15 ∧ y + x / 2 = 6) ∨ (x + x / 2 = 6 ∧ y + x / 2 = 15)) ∧ y = 1 :=
by
  sorry

end isosceles_triangle_base_length_l98_98468


namespace district_B_high_schools_l98_98045

theorem district_B_high_schools :
  ∀ (total_schools public_schools parochial_schools private_schools districtA_schools districtB_private_schools: ℕ),
  total_schools = 50 ∧ 
  public_schools = 25 ∧ 
  parochial_schools = 16 ∧ 
  private_schools = 9 ∧ 
  districtA_schools = 18 ∧ 
  districtB_private_schools = 2 ∧ 
  (∃ districtC_schools, 
     districtC_schools = public_schools / 3 + parochial_schools / 3 + private_schools / 3) →
  ∃ districtB_schools, 
    districtB_schools = total_schools - districtA_schools - (public_schools / 3 + parochial_schools / 3 + private_schools / 3) ∧ 
    districtB_schools = 5 := by
  sorry

end district_B_high_schools_l98_98045


namespace acute_angle_at_7_20_is_100_degrees_l98_98892

theorem acute_angle_at_7_20_is_100_degrees :
  let minute_hand_angle := 4 * 30 -- angle of the minute hand (in degrees)
  let hour_hand_progress := 20 / 60 -- progress of hour hand between 7 and 8
  let hour_hand_angle := 7 * 30 + hour_hand_progress * 30 -- angle of the hour hand (in degrees)

  ∃ angle_acute : ℝ, 
  angle_acute = abs (minute_hand_angle - hour_hand_angle) ∧
  angle_acute = 100 :=
by
  sorry

end acute_angle_at_7_20_is_100_degrees_l98_98892


namespace no_function_exists_l98_98838

-- Main theorem statement
theorem no_function_exists : ¬ ∃ f : ℝ → ℝ, 
  (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * f (2 * y * f x + f y) = x^3 * f (y * f x)) ∧ 
  (∀ z : ℝ, 0 < z → f z > 0) :=
sorry

end no_function_exists_l98_98838


namespace max_oranges_donated_l98_98423

theorem max_oranges_donated (N : ℕ) : ∃ n : ℕ, n < 7 ∧ (N % 7 = n) ∧ n = 6 :=
by
  sorry

end max_oranges_donated_l98_98423


namespace line_through_points_l98_98637

theorem line_through_points (x1 y1 x2 y2 : ℝ) (m b : ℝ) 
  (h1 : x1 = -3) (h2 : y1 = 1) (h3 : x2 = 1) (h4 : y2 = 3)
  (h5 : y1 = m * x1 + b) (h6 : y2 = m * x2 + b) :
  m + b = 3 := 
sorry

end line_through_points_l98_98637


namespace horner_method_v3_value_l98_98760

theorem horner_method_v3_value :
  let f (x : ℤ) := 3 * x^6 + 5 * x^5 + 6 * x^4 + 79 * x^3 - 8 * x^2 + 35 * x + 12
  let v : ℤ := 3
  let v1 (x : ℤ) : ℤ := v * x + 5
  let v2 (x : ℤ) (v1x : ℤ) : ℤ := v1x * x + 6
  let v3 (x : ℤ) (v2x : ℤ) : ℤ := v2x * x + 79
  x = -4 →
  v3 x (v2 x (v1 x)) = -57 :=
by
  sorry

end horner_method_v3_value_l98_98760


namespace andy_questions_wrong_l98_98028

variables (a b c d : ℕ)

-- Given conditions
def condition1 : Prop := a + b = c + d
def condition2 : Prop := a + d = b + c + 6
def condition3 : Prop := c = 7

-- The theorem to prove
theorem andy_questions_wrong (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 c) : a = 10 :=
by
  sorry

end andy_questions_wrong_l98_98028


namespace intersection_of_lines_l98_98103

-- Define the conditions of the problem
def first_line (x y : ℝ) : Prop := y = -3 * x + 1
def second_line (x y : ℝ) : Prop := y + 1 = 15 * x

-- Prove the intersection point of the two lines
theorem intersection_of_lines : 
  ∃ (x y : ℝ), first_line x y ∧ second_line x y ∧ x = 1 / 9 ∧ y = 2 / 3 :=
by
  sorry

end intersection_of_lines_l98_98103


namespace degrees_to_radians_18_l98_98612

theorem degrees_to_radians_18 (degrees : ℝ) (h : degrees = 18) : 
  (degrees * (Real.pi / 180) = Real.pi / 10) :=
by
  sorry

end degrees_to_radians_18_l98_98612


namespace ned_shirts_problem_l98_98698

theorem ned_shirts_problem
  (long_sleeve_shirts : ℕ)
  (total_shirts_washed : ℕ)
  (total_shirts_had : ℕ)
  (h1 : long_sleeve_shirts = 21)
  (h2 : total_shirts_washed = 29)
  (h3 : total_shirts_had = total_shirts_washed + 1) :
  ∃ short_sleeve_shirts : ℕ, short_sleeve_shirts = total_shirts_had - total_shirts_washed - 1 :=
by
  sorry

end ned_shirts_problem_l98_98698


namespace color_stamps_sold_l98_98520

theorem color_stamps_sold :
    let total_stamps : ℕ := 1102609
    let black_and_white_stamps : ℕ := 523776
    total_stamps - black_and_white_stamps = 578833 := 
by
  sorry

end color_stamps_sold_l98_98520


namespace work_completion_days_l98_98302

variable (Paul_days Rose_days Sam_days : ℕ)

def Paul_rate := 1 / 80
def Rose_rate := 1 / 120
def Sam_rate := 1 / 150

def combined_rate := Paul_rate + Rose_rate + Sam_rate

noncomputable def days_to_complete_work := 1 / combined_rate

theorem work_completion_days :
  Paul_days = 80 →
  Rose_days = 120 →
  Sam_days = 150 →
  days_to_complete_work = 37 := 
by
  intros
  simp only [Paul_rate, Rose_rate, Sam_rate, combined_rate, days_to_complete_work]
  sorry

end work_completion_days_l98_98302


namespace chosen_number_eq_l98_98761

-- Given a number x, if (x / 2) - 100 = 4, then x = 208.
theorem chosen_number_eq (x : ℝ) (h : (x / 2) - 100 = 4) : x = 208 := 
by
  sorry

end chosen_number_eq_l98_98761


namespace triangle_area_l98_98887

theorem triangle_area (B : Real) (AB AC : Real) 
  (hB : B = Real.pi / 6) 
  (hAB : AB = 2 * Real.sqrt 3)
  (hAC : AC = 2) : 
  let area := 1 / 2 * AB * AC * Real.sin B
  area = 2 * Real.sqrt 3 := by
  sorry

end triangle_area_l98_98887


namespace sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l98_98247

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

theorem sec_225_eq_neg_sqrt2 :
  sec (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

theorem csc_225_eq_neg_sqrt2 :
  csc (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

end sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l98_98247


namespace find_y_for_orthogonality_l98_98158

theorem find_y_for_orthogonality (y : ℝ) : (3 * y + 7 * (-4) = 0) → y = 28 / 3 := by
  sorry

end find_y_for_orthogonality_l98_98158


namespace yellow_tiles_count_l98_98529

theorem yellow_tiles_count
  (total_tiles : ℕ)
  (yellow_tiles : ℕ)
  (blue_tiles : ℕ)
  (purple_tiles : ℕ)
  (white_tiles : ℕ)
  (h1 : total_tiles = 20)
  (h2 : blue_tiles = yellow_tiles + 1)
  (h3 : purple_tiles = 6)
  (h4 : white_tiles = 7)
  (h5 : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  yellow_tiles = 3 :=
by sorry

end yellow_tiles_count_l98_98529


namespace track_meet_total_people_l98_98464

theorem track_meet_total_people (B G : ℕ) (H1 : B = 30)
  (H2 : ∃ G, (3 * G) / 5 + (2 * G) / 5 = G)
  (H3 : ∀ G, 2 * G / 5 = 10) :
  B + G = 55 :=
by
  sorry

end track_meet_total_people_l98_98464


namespace triangle_inequality_l98_98348

theorem triangle_inequality (A B C : ℝ) :
  ∀ (a b c : ℝ), (a = 2 * Real.sin (A / 2) * Real.cos (A / 2)) ∧
                 (b = 2 * Real.sin (B / 2) * Real.cos (B / 2)) ∧
                 (c = Real.cos ((A + B) / 2)) ∧
                 (x = Real.sqrt (Real.tan (A / 2) * Real.tan (B / 2)))
                 → (Real.sqrt (a * b) / Real.sin (C / 2) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2)) := by {
  sorry
}

end triangle_inequality_l98_98348


namespace cost_of_gravelling_roads_l98_98022

theorem cost_of_gravelling_roads :
  let lawn_length := 70
  let lawn_breadth := 30
  let road_width := 5
  let cost_per_sqm := 4
  let area_road_length := lawn_length * road_width
  let area_road_breadth := lawn_breadth * road_width
  let area_intersection := road_width * road_width
  let total_area_to_be_graveled := (area_road_length + area_road_breadth) - area_intersection
  let total_cost := total_area_to_be_graveled * cost_per_sqm
  total_cost = 1900 :=
by
  sorry

end cost_of_gravelling_roads_l98_98022


namespace determine_A_l98_98684

theorem determine_A (x y A : ℝ) 
  (h : (x + y) ^ 3 - x * y * (x + y) = (x + y) * A) : 
  A = x^2 + x * y + y^2 := 
by
  sorry

end determine_A_l98_98684


namespace min_homework_assignments_l98_98157

variable (p1 p2 p3 : Nat)

-- Define the points and assignments
def points_first_10 : Nat := 10
def assignments_first_10 : Nat := 10 * 1

def points_second_10 : Nat := 10
def assignments_second_10 : Nat := 10 * 2

def points_third_10 : Nat := 10
def assignments_third_10 : Nat := 10 * 3

def total_points : Nat := points_first_10 + points_second_10 + points_third_10
def total_assignments : Nat := assignments_first_10 + assignments_second_10 + assignments_third_10

theorem min_homework_assignments (hp1 : points_first_10 = 10) (ha1 : assignments_first_10 = 10) 
  (hp2 : points_second_10 = 10) (ha2 : assignments_second_10 = 20)
  (hp3 : points_third_10 = 10) (ha3 : assignments_third_10 = 30)
  (tp : total_points = 30) : 
  total_assignments = 60 := 
by sorry

end min_homework_assignments_l98_98157


namespace range_of_a_l98_98949

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, ¬ (x^2 - a * x + 1 ≤ 0)) ↔ -2 < a ∧ a < 2 := 
sorry

end range_of_a_l98_98949


namespace skipping_rope_equation_correct_l98_98237

-- Definitions of constraints
variable (x : ℕ) -- Number of skips per minute by Xiao Ji
variable (H1 : 0 < x) -- The number of skips per minute by Xiao Ji is positive
variable (H2 : 100 / x * x = 100) -- Xiao Ji skips exactly 100 times

-- Xiao Fan's conditions
variable (H3 : 100 + 20 = 120) -- Xiao Fan skips 20 more times than Xiao Ji
variable (H4 : x + 30 > 0) -- Xiao Fan skips 30 more times per minute than Xiao Ji

-- Prove the equation is correct
theorem skipping_rope_equation_correct :
  100 / x = 120 / (x + 30) :=
by
  sorry

end skipping_rope_equation_correct_l98_98237


namespace students_answered_both_questions_correctly_l98_98608

theorem students_answered_both_questions_correctly (P_A P_B P_A'_B' : ℝ) (h_P_A : P_A = 0.75) (h_P_B : P_B = 0.7) (h_P_A'_B' : P_A'_B' = 0.2) :
  ∃ P_A_B : ℝ, P_A_B = 0.65 := 
by
  sorry

end students_answered_both_questions_correctly_l98_98608


namespace linear_function_result_l98_98922

variable {R : Type*} [LinearOrderedField R]

noncomputable def linear_function (g : R → R) : Prop :=
  ∃ (a b : R), ∀ x, g x = a * x + b

theorem linear_function_result (g : R → R) (h_lin : linear_function g) (h : g 5 - g 1 = 16) : g 13 - g 1 = 48 :=
  by
  sorry

end linear_function_result_l98_98922


namespace problem1_subproblem1_subproblem2_l98_98968

-- Problem 1: Prove that a² + b² = 40 given ab = 30 and a + b = 10
theorem problem1 (a b : ℝ) (h1 : a * b = 30) (h2 : a + b = 10) : a^2 + b^2 = 40 := 
sorry

-- Problem 2: Subproblem 1 - Prove that (40 - x)² + (x - 20)² = 420 given (40 - x)(x - 20) = -10
theorem subproblem1 (x : ℝ) (h : (40 - x) * (x - 20) = -10) : (40 - x)^2 + (x - 20)^2 = 420 := 
sorry

-- Problem 2: Subproblem 2 - Prove that (30 + x)² + (20 + x)² = 120 given (30 + x)(20 + x) = 10
theorem subproblem2 (x : ℝ) (h : (30 + x) * (20 + x) = 10) : (30 + x)^2 + (20 + x)^2 = 120 :=
sorry

end problem1_subproblem1_subproblem2_l98_98968


namespace intersection_of_lines_l98_98861

theorem intersection_of_lines : 
  let x := (5 : ℚ) / 9
  let y := (5 : ℚ) / 3
  (y = 3 * x ∧ y - 5 = -6 * x) ↔ (x, y) = ((5 : ℚ) / 9, (5 : ℚ) / 3) := 
by 
  sorry

end intersection_of_lines_l98_98861


namespace invalid_perimeters_l98_98449

theorem invalid_perimeters (x : ℕ) (h1 : 18 < x) (h2 : x < 42) :
  (42 + x ≠ 58) ∧ (42 + x ≠ 85) :=
by
  sorry

end invalid_perimeters_l98_98449


namespace bananas_to_oranges_l98_98335

theorem bananas_to_oranges :
  (3 / 4) * 12 * b = 9 * o →
  ((3 / 5) * 15 * b) = 9 * o := 
by
  sorry

end bananas_to_oranges_l98_98335


namespace sum_of_cubes_consecutive_divisible_by_9_l98_98323

theorem sum_of_cubes_consecutive_divisible_by_9 (n : ℤ) : 9 ∣ (n-1)^3 + n^3 + (n+1)^3 :=
  sorry

end sum_of_cubes_consecutive_divisible_by_9_l98_98323


namespace distribute_6_balls_in_3_boxes_l98_98286

def number_of_ways_to_distribute_balls (balls boxes : Nat) : Nat :=
  boxes ^ balls

theorem distribute_6_balls_in_3_boxes : number_of_ways_to_distribute_balls 6 3 = 729 := by
  sorry

end distribute_6_balls_in_3_boxes_l98_98286


namespace consecutive_integers_sum_l98_98943

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end consecutive_integers_sum_l98_98943


namespace Annie_cookies_sum_l98_98575

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end Annie_cookies_sum_l98_98575


namespace lauren_change_l98_98749

-- Define the given conditions as Lean terms.
def price_meat_per_pound : ℝ := 3.5
def pounds_meat : ℝ := 2.0
def price_buns : ℝ := 1.5
def price_lettuce : ℝ := 1.0
def pounds_tomato : ℝ := 1.5
def price_tomato_per_pound : ℝ := 2.0
def price_pickles : ℝ := 2.5
def coupon_value : ℝ := 1.0
def amount_paid : ℝ := 20.0

-- Define the total cost of each item.
def cost_meat : ℝ := pounds_meat * price_meat_per_pound
def cost_tomato : ℝ := pounds_tomato * price_tomato_per_pound
def total_cost_before_coupon : ℝ := cost_meat + price_buns + price_lettuce + cost_tomato + price_pickles

-- Define the final total cost after applying the coupon.
def final_total_cost : ℝ := total_cost_before_coupon - coupon_value

-- Define the expected change.
def expected_change : ℝ := amount_paid - final_total_cost

-- Prove that the expected change is $6.00.
theorem lauren_change : expected_change = 6.0 := by
  sorry

end lauren_change_l98_98749


namespace max_xy_l98_98829

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x / 3 + y / 4 = 1) : xy ≤ 3 :=
by {
  -- proof omitted
  sorry
}

end max_xy_l98_98829


namespace sum_of_squares_of_roots_l98_98989

theorem sum_of_squares_of_roots :
  ∀ (p q r : ℚ), (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) ∧
                 (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) ∧
                 (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
                 p^2 + q^2 + r^2 = 34 / 9 := 
by
  sorry

end sum_of_squares_of_roots_l98_98989


namespace min_guesses_correct_l98_98354

def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  (min_guesses n k = 2 ↔ n = 2 * k) ∧ (min_guesses n k = 1 ↔ n ≠ 2 * k) := by
  sorry

end min_guesses_correct_l98_98354


namespace inequality_solution_set_l98_98035

theorem inequality_solution_set {x : ℝ} : 2 * x^2 - x - 1 > 0 ↔ (x < -1 / 2 ∨ x > 1) := 
sorry

end inequality_solution_set_l98_98035


namespace probability_heads_exactly_2_times_three_tosses_uniform_coin_l98_98046

noncomputable def probability_heads_exactly_2_times (n k : ℕ) (p : ℚ) : ℚ :=
(n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_heads_exactly_2_times_three_tosses_uniform_coin :
  probability_heads_exactly_2_times 3 2 (1/2) = 3 / 8 :=
by
  sorry

end probability_heads_exactly_2_times_three_tosses_uniform_coin_l98_98046


namespace stadium_fee_difference_l98_98918

theorem stadium_fee_difference :
  let capacity := 2000
  let entry_fee := 20
  let full_fees := capacity * entry_fee
  let three_quarters_fees := (capacity * 3 / 4) * entry_fee
  full_fees - three_quarters_fees = 10000 :=
by
  sorry

end stadium_fee_difference_l98_98918


namespace rice_pounds_l98_98863

noncomputable def pounds_of_rice (r p : ℝ) : Prop :=
  r + p = 30 ∧ 1.10 * r + 0.55 * p = 23.50

theorem rice_pounds (r p : ℝ) (h : pounds_of_rice r p) : r = 12.7 :=
sorry

end rice_pounds_l98_98863


namespace football_match_goals_even_likely_l98_98387

noncomputable def probability_even_goals (p_1 : ℝ) (q_1 : ℝ) : Prop :=
  let p := p_1^2 + q_1^2
  let q := 2 * p_1 * q_1
  p >= q

theorem football_match_goals_even_likely (p_1 : ℝ) (h : p_1 >= 0 ∧ p_1 <= 1) : probability_even_goals p_1 (1 - p_1) :=
by sorry

end football_match_goals_even_likely_l98_98387


namespace xyz_inequality_l98_98680

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 :=
by
  sorry

end xyz_inequality_l98_98680


namespace Alyssa_spent_on_marbles_l98_98388

def total_spent_on_toys : ℝ := 12.30
def cost_of_football : ℝ := 5.71
def amount_spent_on_marbles : ℝ := 12.30 - 5.71

theorem Alyssa_spent_on_marbles :
  total_spent_on_toys - cost_of_football = amount_spent_on_marbles :=
by
  sorry

end Alyssa_spent_on_marbles_l98_98388


namespace final_total_cost_is_12_70_l98_98466

-- Definitions and conditions
def sandwich_count : ℕ := 2
def sandwich_cost_per_unit : ℝ := 2.45

def soda_count : ℕ := 4
def soda_cost_per_unit : ℝ := 0.87

def chips_count : ℕ := 3
def chips_cost_per_unit : ℝ := 1.29

def sandwich_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

-- Final price after discount and tax
noncomputable def total_cost : ℝ :=
  let sandwiches_total := sandwich_count * sandwich_cost_per_unit
  let discounted_sandwiches := sandwiches_total * (1 - sandwich_discount)
  let sodas_total := soda_count * soda_cost_per_unit
  let chips_total := chips_count * chips_cost_per_unit
  let subtotal := discounted_sandwiches + sodas_total + chips_total
  let final_total := subtotal * (1 + sales_tax)
  final_total

theorem final_total_cost_is_12_70 : total_cost = 12.70 :=
by 
  sorry

end final_total_cost_is_12_70_l98_98466


namespace ab_eq_neg_one_l98_98246

variable (a b : ℝ)

-- Condition for the inequality (x >= 0) -> (0 ≤ x^4 - x^3 + ax + b ≤ (x^2 - 1)^2)
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → 
    0 ≤ x^4 - x^3 + a * x + b ∧ 
    x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2

-- Main statement to prove that assuming the condition, a * b = -1
theorem ab_eq_neg_one (h : condition a b) : a * b = -1 := 
  sorry

end ab_eq_neg_one_l98_98246


namespace solution_set_of_inequality_l98_98971

-- Define the conditions and theorem
theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) : (1 / x < x) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x)) :=
by sorry

end solution_set_of_inequality_l98_98971


namespace arithmetic_sequence_a15_l98_98494

variable {α : Type*} [LinearOrderedField α]

-- Conditions for the arithmetic sequence
variable (a : ℕ → α)
variable (d : α)
variable (a1 : α)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 5)
variable (h_a10 : a 10 = 15)

-- To prove that a15 = 25
theorem arithmetic_sequence_a15 : a 15 = 25 := by
  sorry

end arithmetic_sequence_a15_l98_98494


namespace find_a_squared_l98_98784

-- Defining the conditions for the problem
structure RectangleConditions :=
  (a : ℝ) 
  (side_length : ℝ := 36)
  (hinges_vertex : Bool := true)
  (hinges_midpoint : Bool := true)
  (pressed_distance : ℝ := 24)
  (hexagon_area_equiv : Bool := true)

-- Stating the theorem
theorem find_a_squared (cond : RectangleConditions) (ha : 36 * cond.a = 
  (24 * cond.a) + 2 * 15 * Real.sqrt (cond.a^2 - 36)) : 
  cond.a^2 = 720 :=
sorry

end find_a_squared_l98_98784


namespace train_pass_bridge_in_50_seconds_l98_98288

def length_of_train : ℕ := 360
def length_of_bridge : ℕ := 140
def speed_of_train_kmh : ℕ := 36
def total_distance : ℕ := length_of_train + length_of_bridge
def speed_of_train_ms : ℚ := (speed_of_train_kmh * 1000 : ℚ) / 3600 -- we use ℚ to avoid integer division issues
def time_to_pass_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge = 50 := by
  sorry

end train_pass_bridge_in_50_seconds_l98_98288


namespace last_two_digits_7_pow_2018_l98_98699

theorem last_two_digits_7_pow_2018 : 
  (7 ^ 2018) % 100 = 49 := 
sorry

end last_two_digits_7_pow_2018_l98_98699


namespace parallelogram_area_correct_l98_98967

noncomputable def parallelogram_area (a b : ℝ) (α : ℝ) (h : a < b) : ℝ :=
  (4 * a^2 - b^2) / 4 * (Real.tan α)

theorem parallelogram_area_correct (a b α : ℝ) (h : a < b) :
  parallelogram_area a b α h = (4 * a^2 - b^2) / 4 * (Real.tan α) :=
by
  sorry

end parallelogram_area_correct_l98_98967


namespace geometric_sequence_value_l98_98671

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geo : geometric_sequence a r)
  (h_pos : ∀ n, a n > 0)
  (h_roots : ∀ (a1 a19 : ℝ), a1 = a 1 → a19 = a 19 → a1 * a19 = 16 ∧ a1 + a19 = 10) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geometric_sequence_value_l98_98671


namespace sum_remainder_l98_98452

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l98_98452


namespace arithmetic_sequence_proof_l98_98889

open Nat

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = (a 1) * (a 5)

def general_formula (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (d = 0 ∧ ∀ n, a n = 2) ∨ (d = 4 ∧ ∀ n, a n = 4 * n - 2)

def sum_seq (a : ℕ → ℤ) (S_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ((∀ n, a n = 2) ∧ (∀ n, S_n n = 2 * n)) ∨ ((∀ n, a n = 4 * n - 2) ∧ (∀ n, S_n n = 4 * n^2 - 2 * n))

theorem arithmetic_sequence_proof :
  ∃ a : ℕ → ℤ, ∃ d : ℤ, arithmetic_seq a d ∧ general_formula a d ∧ ∃ S_n : ℕ → ℤ, sum_seq a S_n d := by
  sorry

end arithmetic_sequence_proof_l98_98889


namespace exists_digit_sum_divisible_by_11_l98_98187

-- Define a function to compute the sum of the digits of a natural number
def digit_sum (n : ℕ) : ℕ := 
  Nat.digits 10 n |>.sum

-- The main theorem to be proven
theorem exists_digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k) % 11 = 0) := 
sorry

end exists_digit_sum_divisible_by_11_l98_98187


namespace intersection_A_B_find_a_b_l98_98798

noncomputable def A : Set ℝ := { x | x^2 - 5 * x + 6 > 0 }
noncomputable def B : Set ℝ := { x | Real.log (x + 1) / Real.log 2 < 2 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 2 } :=
by
  -- Proof will be provided
  sorry

theorem find_a_b :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + a * x - b < 0 ↔ -1 < x ∧ x < 2) ∧ a = -1 ∧ b = 2 :=
by
  -- Proof will be provided
  sorry

end intersection_A_B_find_a_b_l98_98798


namespace probability_not_all_same_l98_98196

theorem probability_not_all_same (n m : ℕ) (h₁ : n = 5) (h₂ : m = 8) 
  (fair_dice : ∀ (die : ℕ), 1 ≤ die ∧ die ≤ m)
  : (1 - (m / m^n) = 4095 / 4096) := by
  sorry

end probability_not_all_same_l98_98196


namespace range_of_a_l98_98706

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l98_98706


namespace congruent_rectangle_perimeter_l98_98350

theorem congruent_rectangle_perimeter (x y w l P : ℝ) 
  (h1 : x + 2 * w = 2 * y) 
  (h2 : x + 2 * l = y) 
  (hP : P = 2 * l + 2 * w) : 
  P = 3 * y - 2 * x :=
by sorry

end congruent_rectangle_perimeter_l98_98350


namespace solve_inequality_l98_98983

theorem solve_inequality :
  (4 - Real.sqrt 17 < x ∧ x < 4 - Real.sqrt 3) ∨ 
  (4 + Real.sqrt 3 < x ∧ x < 4 + Real.sqrt 17) → 
  0 < (x^2 - 8*x + 13) / (x^2 - 4*x + 7) ∧ 
  (x^2 - 8*x + 13) / (x^2 - 4*x + 7) < 2 :=
sorry

end solve_inequality_l98_98983


namespace parabola_directrix_l98_98063

variable {F P1 P2 : Point}

def is_on_parabola (F : Point) (P1 : Point) : Prop := 
  -- Definition of a point being on the parabola with focus F and a directrix (to be determined).
  sorry

def construct_circles (F P1 P2 : Point) : Circle × Circle :=
  -- Construct circles centered at P1 and P2 passing through F.
  sorry

def common_external_tangents (k1 k2 : Circle) : Nat :=
  -- Function to find the number of common external tangents between two circles.
  sorry

theorem parabola_directrix (F P1 P2 : Point) (h1 : is_on_parabola F P1) (h2 : is_on_parabola F P2) :
  ∃ (k1 k2 : Circle), construct_circles F P1 P2 = (k1, k2) → 
    common_external_tangents k1 k2 = 2 :=
by
  -- Proof that under these conditions, there are exactly 2 common external tangents.
  sorry

end parabola_directrix_l98_98063


namespace quadratic_func_max_value_l98_98555

theorem quadratic_func_max_value (b c x y : ℝ) (h1 : y = -x^2 + b * x + c)
(h1_x1 : (y = 0) → x = -1 ∨ x = 3) :
    -x^2 + 2 * x + 3 ≤ 4 :=
sorry

end quadratic_func_max_value_l98_98555


namespace log_sum_l98_98014

theorem log_sum : (Real.log 0.01 / Real.log 10) + (Real.log 16 / Real.log 2) = 2 := by
  sorry

end log_sum_l98_98014


namespace expected_value_ball_draw_l98_98475

noncomputable def E_xi : ℚ :=
  let prob_xi_2 := 3/5
  let prob_xi_3 := 3/10
  let prob_xi_4 := 1/10
  2 * prob_xi_2 + 3 * prob_xi_3 + 4 * prob_xi_4

theorem expected_value_ball_draw : E_xi = 5 / 2 := by
  sorry

end expected_value_ball_draw_l98_98475


namespace no_polyhedron_without_triangles_and_three_valent_vertices_l98_98048

-- Definitions and assumptions based on the problem's conditions
def f_3 := 0 -- no triangular faces
def p_3 := 0 -- no vertices with degree three

-- Euler's formula for convex polyhedra
def euler_formula (f p a : ℕ) : Prop := f + p - a = 2

-- Define general properties for faces and vertices in polyhedra
def polyhedron_no_triangular_no_three_valent (f p a f_4 f_5 p_4 p_5: ℕ) : Prop :=
  f_3 = 0 ∧ p_3 = 0 ∧ 2 * a ≥ 4 * (f_4 + f_5) ∧ 2 * a ≥ 4 * (p_4 + p_5) ∧ euler_formula f p a

-- Theorem to prove there does not exist such a polyhedron
theorem no_polyhedron_without_triangles_and_three_valent_vertices :
  ¬ ∃ (f p a f_4 f_5 p_4 p_5 : ℕ), polyhedron_no_triangular_no_three_valent f p a f_4 f_5 p_4 p_5 :=
by
  sorry

end no_polyhedron_without_triangles_and_three_valent_vertices_l98_98048


namespace complex_problem_l98_98558

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem complex_problem :
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 :=
by
  sorry

end complex_problem_l98_98558


namespace find_a_l98_98804

theorem find_a (a : ℝ) (U A CU: Set ℝ) (hU : U = {2, 3, a^2 - a - 1}) (hA : A = {2, 3}) (hCU : CU = {1}) (hComplement : CU = U \ A) :
  a = -1 ∨ a = 2 :=
by
  sorry

end find_a_l98_98804


namespace rope_segments_divided_l98_98282

theorem rope_segments_divided (folds1 folds2 : ℕ) (cut : ℕ) (h_folds1 : folds1 = 3) (h_folds2 : folds2 = 2) (h_cut : cut = 1) :
  (folds1 * folds2 + cut = 7) :=
by {
  -- Proof steps would go here
  sorry
}

end rope_segments_divided_l98_98282


namespace rectangle_volume_l98_98808

theorem rectangle_volume {a b c : ℕ} (h1 : a * b - c * a - b * c = 1) (h2 : c * a = b * c + 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a * b * c = 6 :=
sorry

end rectangle_volume_l98_98808


namespace C_finishes_job_in_days_l98_98409

theorem C_finishes_job_in_days :
  ∀ (A B C : ℚ),
    (A + B = 1 / 15) →
    (A + B + C = 1 / 3) →
    1 / C = 3.75 :=
by
  intros A B C hab habc
  sorry

end C_finishes_job_in_days_l98_98409


namespace part1_max_price_part2_min_sales_volume_l98_98170

noncomputable def original_price : ℝ := 25
noncomputable def original_sales_volume : ℝ := 80000
noncomputable def original_revenue : ℝ := original_price * original_sales_volume
noncomputable def max_new_price (t : ℝ) : Prop := t * (130000 - 2000 * t) ≥ original_revenue

theorem part1_max_price (t : ℝ) (ht : max_new_price t) : t ≤ 40 :=
sorry

noncomputable def investment (x : ℝ) : ℝ := (1 / 6) * (x^2 - 600) + 50 + (x / 5)
noncomputable def min_sales_volume (x : ℝ) (a : ℝ) : Prop := a * x ≥ original_revenue + investment x

theorem part2_min_sales_volume (a : ℝ) : min_sales_volume 30 a → a ≥ 10.2 :=
sorry

end part1_max_price_part2_min_sales_volume_l98_98170


namespace find_f1_l98_98795

theorem find_f1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f (3 * x + 1) = x^2 + 3*x + 2) :
  f 1 = 2 :=
by
  -- Proof is omitted
  sorry

end find_f1_l98_98795


namespace partition_no_infinite_arith_prog_l98_98896

theorem partition_no_infinite_arith_prog :
  ∃ (A B : Set ℕ), 
  (∀ n ∈ A, n ∈ B → False) ∧ 
  (∀ (a b : ℕ) (d : ℕ), (a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b) % d = 0) → False) ∧
  (∀ (a b : ℕ) (d : ℕ), (a ∈ B ∧ b ∈ B ∧ a ≠ b ∧ (a - b) % d = 0) → False) :=
sorry

end partition_no_infinite_arith_prog_l98_98896


namespace students_to_add_l98_98174

theorem students_to_add (students := 1049) (teachers := 9) : ∃ n, students + n ≡ 0 [MOD teachers] ∧ n = 4 :=
by
  use 4
  sorry

end students_to_add_l98_98174


namespace men_who_wore_glasses_l98_98563

theorem men_who_wore_glasses (total_people : ℕ) (women_ratio men_with_glasses_ratio : ℚ)  
  (h_total : total_people = 1260) 
  (h_women_ratio : women_ratio = 7 / 18)
  (h_men_with_glasses_ratio : men_with_glasses_ratio = 6 / 11)
  : ∃ (men_with_glasses : ℕ), men_with_glasses = 420 := 
by
  sorry

end men_who_wore_glasses_l98_98563


namespace lcm_fractions_l98_98481

theorem lcm_fractions (x : ℕ) (hx : x > 0) :
  lcm (1 / (2 * x)) (lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (12 * x)))) = 1 / (12 * x) :=
sorry

end lcm_fractions_l98_98481


namespace probability_diamond_then_ace_l98_98841

theorem probability_diamond_then_ace :
  let total_cards := 104
  let diamonds := 26
  let aces := 8
  let remaining_cards_after_first_draw := total_cards - 1
  let ace_of_diamonds_prob := (2 : ℚ) / total_cards
  let any_ace_after_ace_of_diamonds := (7 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_ace_of_diamonds_then_any_ace := ace_of_diamonds_prob * any_ace_after_ace_of_diamonds
  let diamond_not_ace_prob := (24 : ℚ) / total_cards
  let any_ace_after_diamond_not_ace := (8 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_diamond_not_ace_then_any_ace := diamond_not_ace_prob * any_ace_after_diamond_not_ace
  let total_prob := combined_prob_ace_of_diamonds_then_any_ace + combined_prob_diamond_not_ace_then_any_ace
  total_prob = (31 : ℚ) / 5308 :=
by
  sorry

end probability_diamond_then_ace_l98_98841


namespace mod_remainder_l98_98603

theorem mod_remainder (a b c : ℕ) : 
  (7 * 10 ^ 20 + 1 ^ 20) % 11 = 8 := by
  -- Lean proof will be written here
  sorry

end mod_remainder_l98_98603


namespace tony_quilt_square_side_length_l98_98790

theorem tony_quilt_square_side_length (length width : ℝ) (h_length : length = 6) (h_width : width = 24) : 
  ∃ s, s * s = length * width ∧ s = 12 :=
by
  sorry

end tony_quilt_square_side_length_l98_98790


namespace David_pushups_l98_98293

-- Definitions and setup conditions
def Zachary_pushups : ℕ := 7
def additional_pushups : ℕ := 30

-- Theorem statement to be proved
theorem David_pushups 
  (zachary_pushups : ℕ) 
  (additional_pushups : ℕ) 
  (Zachary_pushups_val : zachary_pushups = Zachary_pushups) 
  (additional_pushups_val : additional_pushups = additional_pushups) :
  zachary_pushups + additional_pushups = 37 :=
sorry

end David_pushups_l98_98293


namespace ratio_turkeys_to_ducks_l98_98878

theorem ratio_turkeys_to_ducks (chickens ducks turkeys total_birds : ℕ)
  (h1 : chickens = 200)
  (h2 : ducks = 2 * chickens)
  (h3 : total_birds = 1800)
  (h4 : total_birds = chickens + ducks + turkeys) :
  (turkeys : ℚ) / ducks = 3 := by
sorry

end ratio_turkeys_to_ducks_l98_98878


namespace cannot_form_right_triangle_setA_l98_98689

def is_right_triangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem cannot_form_right_triangle_setA (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  ¬ is_right_triangle a b c :=
by {
  sorry
}

end cannot_form_right_triangle_setA_l98_98689


namespace product_of_roots_l98_98833

theorem product_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -2) :
  (∀ x : ℝ, x^2 + x - 2 = 0 → (x = x1 ∨ x = x2)) → x1 * x2 = -2 :=
by
  intros h_root
  exact h

end product_of_roots_l98_98833


namespace ratio_of_girls_participated_to_total_l98_98317

noncomputable def ratio_participating_girls {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ℚ :=
    girls_participated / (girls_participated + boys_participated)

theorem ratio_of_girls_participated_to_total {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ratio_participating_girls h1 equal_boys_girls girls_participated boys_participated = 9 / 17 :=
by
    sorry

end ratio_of_girls_participated_to_total_l98_98317


namespace third_term_of_geometric_sequence_l98_98919

theorem third_term_of_geometric_sequence
  (a₁ : ℕ) (a₄ : ℕ)
  (h1 : a₁ = 5)
  (h4 : a₄ = 320) :
  ∃ a₃ : ℕ, a₃ = 80 :=
by
  sorry

end third_term_of_geometric_sequence_l98_98919


namespace scientific_notation_equivalence_l98_98678

theorem scientific_notation_equivalence : 3 * 10^(-7) = 0.0000003 :=
by
  sorry

end scientific_notation_equivalence_l98_98678


namespace y1_gt_y2_l98_98098

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) (hA : y1 = k * (-3) + 3) (hB : y2 = k * 1 + 3) (hK : k < 0) : y1 > y2 :=
by 
  sorry

end y1_gt_y2_l98_98098


namespace painting_ways_correct_l98_98549

noncomputable def num_ways_to_paint : ℕ :=
  let red := 1
  let green_or_blue := 2
  let total_ways_case1 := red
  let total_ways_case2 := (green_or_blue ^ 4)
  let total_ways_case3 := green_or_blue ^ 3
  let total_ways_case4 := green_or_blue ^ 2
  let total_ways_case5 := green_or_blue
  let total_ways_case6 := red
  total_ways_case1 + total_ways_case2 + total_ways_case3 + total_ways_case4 + total_ways_case5 + total_ways_case6

theorem painting_ways_correct : num_ways_to_paint = 32 :=
  by
  sorry

end painting_ways_correct_l98_98549


namespace find_a_l98_98719

variable (a b c : ℤ)

theorem find_a (h1 : a + b = 2) (h2 : b + c = 0) (h3 : |c| = 1) : a = 3 ∨ a = 1 := 
sorry

end find_a_l98_98719


namespace percent_of_x_is_y_minus_z_l98_98796

variable (x y z : ℝ)

axiom condition1 : 0.60 * (x - y) = 0.30 * (x + y + z)
axiom condition2 : 0.40 * (y - z) = 0.20 * (y + x - z)

theorem percent_of_x_is_y_minus_z :
  (y - z) = x := by
  sorry

end percent_of_x_is_y_minus_z_l98_98796


namespace system_of_equations_solution_l98_98881

theorem system_of_equations_solution (x y : ℝ) (h1 : 4 * x + 3 * y = 11) (h2 : 4 * x - 3 * y = 5) :
  x = 2 ∧ y = 1 :=
by {
  sorry
}

end system_of_equations_solution_l98_98881


namespace sum_series_l98_98161

theorem sum_series : (List.sum [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56, -59]) = -30 :=
by
  sorry

end sum_series_l98_98161


namespace max_shirt_price_l98_98116

theorem max_shirt_price (total_budget : ℝ) (entrance_fee : ℝ) (num_shirts : ℝ) 
  (discount_rate : ℝ) (tax_rate : ℝ) (max_price : ℝ) 
  (budget_after_fee : total_budget - entrance_fee = 195)
  (shirt_discount : num_shirts > 15 → discounted_price = num_shirts * max_price * (1 - discount_rate))
  (price_with_tax : discounted_price * (1 + tax_rate) ≤ 195) : 
  max_price ≤ 10 := 
sorry

end max_shirt_price_l98_98116


namespace solution_set_of_inequalities_l98_98975

theorem solution_set_of_inequalities (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : ∀ x, mx + n > 0 ↔ x < (1/3)) : ∀ x, nx - m < 0 ↔ x < -3 :=
by
  sorry

end solution_set_of_inequalities_l98_98975


namespace find_water_needed_l98_98051

def apple_juice := 4
def honey (A : ℕ) := 3 * A
def water (H : ℕ) := 3 * H

theorem find_water_needed : water (honey apple_juice) = 36 :=
  sorry

end find_water_needed_l98_98051


namespace perpendicular_lines_b_value_l98_98099

theorem perpendicular_lines_b_value :
  ( ∀ x y : ℝ, 2 * x + 3 * y + 4 = 0)  →
  ( ∀ x y : ℝ, b * x + 3 * y - 1 = 0) →
  ( - (2 : ℝ) / (3 : ℝ) * - b / (3 : ℝ) = -1 ) →
  b = - (9 : ℝ) / (2 : ℝ) :=
by
  intros h1 h2 h3
  sorry

end perpendicular_lines_b_value_l98_98099


namespace find_T_l98_98159

theorem find_T (T : ℝ) (h : (3/4) * (1/8) * T = (1/2) * (1/6) * 72) : T = 64 :=
by {
  -- proof goes here
  sorry
}

end find_T_l98_98159


namespace value_of_f_is_29_l98_98036

noncomputable def f (x : ℕ) : ℕ := 3 * x - 4
noncomputable def g (x : ℕ) : ℕ := x^2 + 1

theorem value_of_f_is_29 :
  f (1 + g 3) = 29 := by
  sorry

end value_of_f_is_29_l98_98036


namespace stratified_sampling_correct_l98_98403

-- Define the total number of employees
def total_employees : ℕ := 100

-- Define the number of employees in each age group
def under_30 : ℕ := 20
def between_30_and_40 : ℕ := 60
def over_40 : ℕ := 20

-- Define the number of people to be drawn
def total_drawn : ℕ := 20

-- Function to calculate number of people to be drawn from each group
def stratified_draw (group_size : ℕ) (total_size : ℕ) (drawn : ℕ) : ℕ :=
  (group_size * drawn) / total_size

-- The proof problem statement
theorem stratified_sampling_correct :
  stratified_draw under_30 total_employees total_drawn = 4 ∧
  stratified_draw between_30_and_40 total_employees total_drawn = 12 ∧
  stratified_draw over_40 total_employees total_drawn = 4 := by
  sorry

end stratified_sampling_correct_l98_98403


namespace maria_profit_disks_l98_98368

theorem maria_profit_disks (cost_price_per_5 : ℝ) (sell_price_per_4 : ℝ) (desired_profit : ℝ) : 
  (cost_price_per_5 = 6) → (sell_price_per_4 = 8) → (desired_profit = 120) →
  (150 : ℝ) = desired_profit / ((sell_price_per_4 / 4) - (cost_price_per_5 / 5)) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end maria_profit_disks_l98_98368


namespace find_k_for_circle_l98_98204

theorem find_k_for_circle (k : ℝ) : (∃ x y : ℝ, (x^2 + 8*x + y^2 + 4*y - k = 0) ∧ (x + 4)^2 + (y + 2)^2 = 25) → k = 5 := 
by 
  sorry

end find_k_for_circle_l98_98204


namespace smallest_n_l98_98646

theorem smallest_n (n : ℕ) (h1 : n ≡ 1 [MOD 3]) (h2 : n ≡ 4 [MOD 5]) (h3 : n > 20) : n = 34 := 
sorry

end smallest_n_l98_98646


namespace bert_money_left_l98_98230

theorem bert_money_left (initial_money : ℕ) (spent_hardware : ℕ) (spent_cleaners : ℕ) (spent_grocery : ℕ) :
  initial_money = 52 →
  spent_hardware = initial_money * 1 / 4 →
  spent_cleaners = 9 →
  spent_grocery = (initial_money - spent_hardware - spent_cleaners) / 2 →
  initial_money - spent_hardware - spent_cleaners - spent_grocery = 15 := 
by
  intros h_initial h_hardware h_cleaners h_grocery
  rw [h_initial, h_hardware, h_cleaners, h_grocery]
  sorry

end bert_money_left_l98_98230


namespace initial_distance_l98_98964

theorem initial_distance (speed_enrique speed_jamal : ℝ) (hours : ℝ) 
  (h_enrique : speed_enrique = 16) 
  (h_jamal : speed_jamal = 23) 
  (h_time : hours = 8) 
  (h_difference : speed_jamal = speed_enrique + 7) : 
  (speed_enrique * hours + speed_jamal * hours = 312) :=
by 
  sorry

end initial_distance_l98_98964


namespace parallel_segments_k_value_l98_98592

open Real

theorem parallel_segments_k_value :
  let A' := (-6, 0)
  let B' := (0, -6)
  let X' := (0, 12)
  ∃ k : ℝ,
  let Y' := (18, k)
  let m_ab := (B'.2 - A'.2) / (B'.1 - A'.1)
  let m_xy := (Y'.2 - X'.2) / (Y'.1 - X'.1)
  m_ab = m_xy → k = -6 :=
by
  sorry

end parallel_segments_k_value_l98_98592


namespace matrix_power_four_l98_98073

theorem matrix_power_four :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![3 * Real.sqrt 2, -3],
    ![3, 3 * Real.sqrt 2]
  ]
  (A ^ 4 = ![
    ![ -81, 0],
    ![0, -81]
  ]) :=
by
  sorry

end matrix_power_four_l98_98073


namespace sqrt_0_09_eq_0_3_l98_98911

theorem sqrt_0_09_eq_0_3 : Real.sqrt 0.09 = 0.3 := 
by 
  sorry

end sqrt_0_09_eq_0_3_l98_98911


namespace largest_divisor_even_triplet_l98_98499

theorem largest_divisor_even_triplet :
  ∀ (n : ℕ), 24 ∣ (2 * n) * (2 * n + 2) * (2 * n + 4) :=
by intros; sorry

end largest_divisor_even_triplet_l98_98499


namespace base8_subtraction_l98_98995

-- Define the base 8 notation for the given numbers
def b8_256 := 256
def b8_167 := 167
def b8_145 := 145

-- Define the sum of 256_8 and 167_8 in base 8
def sum_b8 := 435

-- Define the result of subtracting 145_8 from the sum in base 8
def result_b8 := 370

-- Prove that the result of the entire operation is 370_8
theorem base8_subtraction : sum_b8 - b8_145 = result_b8 := by
  sorry

end base8_subtraction_l98_98995


namespace num_passengers_on_second_plane_l98_98249

theorem num_passengers_on_second_plane :
  ∃ x : ℕ, 600 - (2 * 50) + 600 - (2 * x) + 600 - (2 * 40) = 1500 →
  x = 60 :=
by
  sorry

end num_passengers_on_second_plane_l98_98249


namespace find_RS_length_l98_98319

-- Define the conditions and the problem in Lean

theorem find_RS_length
  (radius : ℝ)
  (P Q R S T : ℝ)
  (center_to_T : ℝ)
  (PT : ℝ)
  (PQ : ℝ)
  (RT TS : ℝ)
  (h_radius : radius = 7)
  (h_center_to_T : center_to_T = 3)
  (h_PT : PT = 8)
  (h_bisect_PQ : PQ = 2 * PT)
  (h_intersecting_chords : PT * (PQ / 2) = RT * TS)
  (h_perfect_square : ∃ k : ℝ, k^2 = RT * TS) :
  RS = 16 :=
by
  sorry

end find_RS_length_l98_98319


namespace clock_equiv_to_square_l98_98747

theorem clock_equiv_to_square : ∃ h : ℕ, h > 5 ∧ (h^2 - h) % 24 = 0 ∧ h = 9 :=
by 
  let h := 9
  use h
  refine ⟨by decide, by decide, rfl⟩ 

end clock_equiv_to_square_l98_98747


namespace number_of_candidates_is_three_l98_98552

variable (votes : List ℕ) (totalVotes : ℕ)

def determineNumberOfCandidates (votes : List ℕ) (totalVotes : ℕ) : ℕ :=
  votes.length

theorem number_of_candidates_is_three (V : ℕ) 
  (h_votes : [2500, 5000, 20000].sum = V) 
  (h_percent : 20000 = 7273 / 10000 * V): 
  determineNumberOfCandidates [2500, 5000, 20000] V = 3 := 
by 
  sorry

end number_of_candidates_is_three_l98_98552


namespace hexagon_diagonal_length_is_twice_side_l98_98114

noncomputable def regular_hexagon_side_length : ℝ := 12

def diagonal_length_in_regular_hexagon (s : ℝ) : ℝ :=
2 * s

theorem hexagon_diagonal_length_is_twice_side :
  diagonal_length_in_regular_hexagon regular_hexagon_side_length = 2 * regular_hexagon_side_length :=
by 
  -- Simplify and check the computation according to the understanding of the properties of the hexagon
  sorry

end hexagon_diagonal_length_is_twice_side_l98_98114


namespace zero_neither_positive_nor_negative_l98_98802

def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0
def is_rational (n : ℤ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ n = p / q

theorem zero_neither_positive_nor_negative : ¬is_positive 0 ∧ ¬is_negative 0 :=
by
  sorry

end zero_neither_positive_nor_negative_l98_98802


namespace rhombus_area_l98_98175

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) : 
  1 / 2 * d1 * d2 = 15 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l98_98175


namespace Francie_remaining_money_l98_98763

theorem Francie_remaining_money :
  let weekly_allowance_8_weeks : ℕ := 5 * 8
  let weekly_allowance_6_weeks : ℕ := 6 * 6
  let cash_gift : ℕ := 20
  let initial_total_savings := weekly_allowance_8_weeks + weekly_allowance_6_weeks + cash_gift

  let investment_amount : ℕ := 10
  let expected_return_investment_1 : ℚ := 0.05 * 10
  let expected_return_investment_2 : ℚ := (0.5 * 0.10 * 10) + (0.5 * 0.02 * 10)
  let best_investment_return := max expected_return_investment_1 expected_return_investment_2
  let final_savings_after_investment : ℚ := initial_total_savings - investment_amount + best_investment_return

  let amount_for_clothes : ℚ := final_savings_after_investment / 2
  let remaining_after_clothes := final_savings_after_investment - amount_for_clothes
  let cost_of_video_game : ℕ := 35
  
  remaining_after_clothes.sub cost_of_video_game = 8.30 :=
by
  intros
  sorry

end Francie_remaining_money_l98_98763


namespace large_marshmallows_are_eight_l98_98993

-- Definition for the total number of marshmallows
def total_marshmallows : ℕ := 18

-- Definition for the number of mini marshmallows
def mini_marshmallows : ℕ := 10

-- Definition for the number of large marshmallows
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

-- Theorem stating that the number of large marshmallows is 8
theorem large_marshmallows_are_eight : large_marshmallows = 8 := by
  sorry

end large_marshmallows_are_eight_l98_98993


namespace find_fraction_l98_98930

theorem find_fraction (x f : ℝ) (h₁ : x = 140) (h₂ : 0.65 * x = f * x - 21) : f = 0.8 :=
by
  sorry

end find_fraction_l98_98930


namespace election_max_k_1002_l98_98528

/-- There are 2002 candidates initially. 
In each round, one candidate with the least number of votes is eliminated unless a candidate receives more than half the votes.
Determine the highest possible value of k if Ostap Bender is elected in the 1002nd round. -/
theorem election_max_k_1002 
  (number_of_candidates : ℕ)
  (number_of_rounds : ℕ)
  (k : ℕ)
  (h1 : number_of_candidates = 2002)
  (h2 : number_of_rounds = 1002)
  (h3 : k ≤ number_of_candidates - 1)
  (h4 : ∀ n : ℕ, n < number_of_rounds → (k + n) % (number_of_candidates - n) ≠ 0) : 
  k = 2001 := sorry

end election_max_k_1002_l98_98528


namespace proof_problem_l98_98085

theorem proof_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b ∣ c * (c ^ 2 - c + 1))
  (h5 : (c ^ 2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c ^ 2 - c + 1) ∨ (a = c ^ 2 - c + 1 ∧ b = c) :=
sorry

end proof_problem_l98_98085


namespace least_cost_planting_l98_98633

theorem least_cost_planting :
  let region1_area := 3 * 1
  let region2_area := 4 * 4
  let region3_area := 7 * 2
  let region4_area := 5 * 4
  let region5_area := 5 * 6
  let easter_lilies_cost_per_sqft := 3.25
  let dahlias_cost_per_sqft := 2.75
  let cannas_cost_per_sqft := 2.25
  let begonias_cost_per_sqft := 1.75
  let asters_cost_per_sqft := 1.25
  region1_area * easter_lilies_cost_per_sqft +
  region2_area * dahlias_cost_per_sqft +
  region3_area * cannas_cost_per_sqft +
  region4_area * begonias_cost_per_sqft +
  region5_area * asters_cost_per_sqft =
  156.75 := 
sorry

end least_cost_planting_l98_98633


namespace cost_effective_bus_choice_l98_98601

theorem cost_effective_bus_choice (x y : ℕ) (h1 : y = x - 1) (h2 : 32 < 48 * x - 64 * y ∧ 48 * x - 64 * y < 64) : 
  64 * 300 < x * 2600 → True :=
by {
  sorry
}

end cost_effective_bus_choice_l98_98601


namespace ben_minimum_test_score_l98_98163

theorem ben_minimum_test_score 
  (scores : List ℕ) 
  (current_avg : ℕ) 
  (desired_increase : ℕ) 
  (lowest_score : ℕ) 
  (required_score : ℕ) 
  (h_scores : scores = [95, 85, 75, 65, 90]) 
  (h_current_avg : current_avg = 82) 
  (h_desired_increase : desired_increase = 5) 
  (h_lowest_score : lowest_score = 65) 
  (h_required_score : required_score = 112) :
  (current_avg + desired_increase) = 87 ∧ 
  (6 * (current_avg + desired_increase)) = 522 ∧ 
  required_score = (522 - (95 + 85 + 75 + 65 + 90)) ∧ 
  (522 - (95 + 85 + 75 + 65 + 90)) > lowest_score :=
by 
  sorry

end ben_minimum_test_score_l98_98163


namespace min_right_triangle_side_l98_98797

theorem min_right_triangle_side (s : ℕ) : 
  (7^2 + 24^2 = s^2 ∧ 7 + 24 > s ∧ 24 + s > 7 ∧ 7 + s > 24) → s = 25 :=
by
  intro h
  sorry

end min_right_triangle_side_l98_98797


namespace ellipse_eccentricity_l98_98294

theorem ellipse_eccentricity (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 16 = 1) ∧ (∃ e : ℝ, e = 3 / 4) ∧ (∀ c : ℝ, c = 3 / 4)
   → a = 7 :=
by
  sorry

end ellipse_eccentricity_l98_98294


namespace div_fact_l98_98345

-- Conditions
def fact_10 : ℕ := 3628800
def fact_4 : ℕ := 4 * 3 * 2 * 1

-- Question and Correct Answer
theorem div_fact (h : fact_10 = 3628800) : fact_10 / fact_4 = 151200 :=
by
  sorry

end div_fact_l98_98345


namespace lines_symmetric_about_y_axis_l98_98324

theorem lines_symmetric_about_y_axis (m n p : ℝ) :
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0)
  ↔ (m = -n ∧ p = -5) :=
sorry

end lines_symmetric_about_y_axis_l98_98324


namespace same_sign_abc_l98_98128
open Classical

theorem same_sign_abc (a b c : ℝ) (h1 : (b / a) * (c / a) > 1) (h2 : (b / a) + (c / a) ≥ -2) : 
  (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0) :=
sorry

end same_sign_abc_l98_98128


namespace probability_grade_A_l98_98231

-- Defining probabilities
def P_B : ℝ := 0.05
def P_C : ℝ := 0.03

-- Theorem: proving the probability of Grade A
theorem probability_grade_A : 1 - P_B - P_C = 0.92 :=
by
  -- Placeholder for proof
  sorry

end probability_grade_A_l98_98231


namespace cost_formula_correct_l98_98803

def cost_of_ride (T : ℤ) : ℤ :=
  if T > 5 then 10 + 5 * T - 10 else 10 + 5 * T

theorem cost_formula_correct (T : ℤ) : cost_of_ride T = 10 + 5 * T - (if T > 5 then 10 else 0) := by
  sorry

end cost_formula_correct_l98_98803


namespace prime_factors_sum_correct_prime_factors_product_correct_l98_98972

-- The number we are considering
def n : ℕ := 172480

-- Prime factors of the number n
def prime_factors : List ℕ := [2, 3, 5, 719]

-- Sum of the prime factors
def sum_prime_factors : ℕ := 2 + 3 + 5 + 719

-- Product of the prime factors
def prod_prime_factors : ℕ := 2 * 3 * 5 * 719

theorem prime_factors_sum_correct :
  sum_prime_factors = 729 :=
by {
  -- Proof goes here
  sorry
}

theorem prime_factors_product_correct :
  prod_prime_factors = 21570 :=
by {
  -- Proof goes here
  sorry
}

end prime_factors_sum_correct_prime_factors_product_correct_l98_98972


namespace total_games_played_l98_98733

-- Define the conditions as parameters
def ratio_games_won_lost (W L : ℕ) : Prop := W / 2 = L / 3

-- Let's state the problem formally in Lean
theorem total_games_played (W L : ℕ) (h1 : ratio_games_won_lost W L) (h2 : W = 18) : W + L = 30 :=
by 
  sorry  -- The proof will be filled in


end total_games_played_l98_98733


namespace one_point_shots_count_l98_98837

-- Define the given conditions
def three_point_shots : Nat := 15
def two_point_shots : Nat := 12
def total_points : Nat := 75
def points_per_three_shot : Nat := 3
def points_per_two_shot : Nat := 2

-- Define the total points contributed by three-point and two-point shots
def three_point_total : Nat := three_point_shots * points_per_three_shot
def two_point_total : Nat := two_point_shots * points_per_two_shot
def combined_point_total : Nat := three_point_total + two_point_total

-- Formulate the theorem to prove the number of one-point shots Tyson made
theorem one_point_shots_count : combined_point_total <= total_points →
  (total_points - combined_point_total = 6) :=
by 
  -- Skip the proof
  sorry

end one_point_shots_count_l98_98837


namespace quadratic_inequality_solution_l98_98893

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : 1 + 2 = b / a)
  (h3 : 1 * 2 = c / a) :
  ∀ x : ℝ, cx^2 + bx + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1 / 2 :=
by
  sorry

end quadratic_inequality_solution_l98_98893


namespace leq_sum_l98_98807

open BigOperators

theorem leq_sum (x : Fin 3 → ℝ) (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, x i = 1) :
  (∑ i, 1 / (1 + (x i)^2)) ≤ 27 / 10 :=
sorry

end leq_sum_l98_98807


namespace correct_inequality_l98_98982

theorem correct_inequality (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > ab ∧ ab > a :=
sorry

end correct_inequality_l98_98982


namespace triangle_area_l98_98223

theorem triangle_area : 
  ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → 
  B = (4, 0) → 
  C = (2, 6) → 
  (1 / 2 : ℝ) * (4 : ℝ) * (6 : ℝ) = (12.0 : ℝ) := 
by 
  intros A B C hA hB hC
  simp [hA, hB, hC]
  norm_num

end triangle_area_l98_98223


namespace before_lunch_rush_customers_l98_98536

def original_customers_before_lunch := 29
def added_customers_during_lunch := 20
def customers_no_tip := 34
def customers_tip := 15

theorem before_lunch_rush_customers : 
  original_customers_before_lunch + added_customers_during_lunch = customers_no_tip + customers_tip → 
  original_customers_before_lunch = 29 := 
by
  sorry

end before_lunch_rush_customers_l98_98536


namespace chandler_needs_to_sell_more_rolls_l98_98311

/-- Chandler's wrapping paper selling condition. -/
def chandler_needs_to_sell : ℕ := 12

def sold_to_grandmother : ℕ := 3
def sold_to_uncle : ℕ := 4
def sold_to_neighbor : ℕ := 3

def total_sold : ℕ := sold_to_grandmother + sold_to_uncle + sold_to_neighbor

theorem chandler_needs_to_sell_more_rolls : chandler_needs_to_sell - total_sold = 2 :=
by
  sorry

end chandler_needs_to_sell_more_rolls_l98_98311


namespace Kim_total_hours_l98_98550

-- Define the initial conditions
def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_class : ℕ := 1

-- The proof problem: Given the initial conditions, prove the total hours of classes per day is 6
theorem Kim_total_hours : (initial_classes - dropped_class) * hours_per_class = 6 := by
  sorry

end Kim_total_hours_l98_98550


namespace johnny_marble_combinations_l98_98973

/-- 
Johnny has 10 different colored marbles. 
The number of ways he can choose four different marbles from his bag is 210.
-/
theorem johnny_marble_combinations : (Nat.choose 10 4) = 210 := by
  sorry

end johnny_marble_combinations_l98_98973


namespace lasagna_package_weight_l98_98037

theorem lasagna_package_weight 
  (beef : ℕ) 
  (noodles_needed_per_beef : ℕ) 
  (current_noodles : ℕ) 
  (packages_needed : ℕ) 
  (noodles_per_package : ℕ) 
  (H1 : beef = 10)
  (H2 : noodles_needed_per_beef = 2)
  (H3 : current_noodles = 4)
  (H4 : packages_needed = 8)
  (H5 : noodles_per_package = (2 * beef - current_noodles) / packages_needed) :
  noodles_per_package = 2 := 
by
  sorry

end lasagna_package_weight_l98_98037


namespace perimeter_triangle_ABC_l98_98521

-- Define the conditions and statement
theorem perimeter_triangle_ABC 
  (r : ℝ) (AP PB altitude : ℝ) 
  (h1 : r = 30) 
  (h2 : AP = 26) 
  (h3 : PB = 32) 
  (h4 : altitude = 96) :
  (2 * (58 + 34.8)) = 185.6 :=
by
  sorry

end perimeter_triangle_ABC_l98_98521


namespace cone_height_l98_98491

noncomputable def height_of_cone (r : ℝ) (n : ℕ) : ℝ :=
  let sector_circumference := (2 * Real.pi * r) / n
  let cone_base_radius := sector_circumference / (2 * Real.pi)
  Real.sqrt (r^2 - cone_base_radius^2)

theorem cone_height
  (r_original : ℝ)
  (n : ℕ)
  (h : r_original = 10)
  (hc : n = 4) :
  height_of_cone r_original n = 5 * Real.sqrt 3 := by
  sorry

end cone_height_l98_98491


namespace y_coordinate_in_fourth_quadrant_l98_98822
-- Importing the necessary libraries

-- Definition of the problem statement
theorem y_coordinate_in_fourth_quadrant (x y : ℝ) (h : x = 5 ∧ y < 0) : y < 0 :=
by 
  sorry

end y_coordinate_in_fourth_quadrant_l98_98822


namespace min_value_expr_l98_98694

noncomputable def min_value (a b c : ℝ) := 4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c)

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  min_value a b c ≥ 8 / Real.sqrt 3 :=
by
  sorry

end min_value_expr_l98_98694


namespace A_loses_240_l98_98453

def initial_house_value : ℝ := 12000
def house_value_after_A_sells : ℝ := initial_house_value * 0.85
def house_value_after_B_sells_back : ℝ := house_value_after_A_sells * 1.2

theorem A_loses_240 : house_value_after_B_sells_back - initial_house_value = 240 := by
  sorry

end A_loses_240_l98_98453


namespace quadratic_has_real_solutions_iff_l98_98551

theorem quadratic_has_real_solutions_iff (m : ℝ) :
  ∃ x y : ℝ, (y = m * x + 3) ∧ (y = (3 * m - 2) * x ^ 2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2) ∨ (m ≥ 12 + 8 * Real.sqrt 2) :=
by
  sorry

end quadratic_has_real_solutions_iff_l98_98551


namespace workers_contribution_eq_l98_98184

variable (W C : ℕ)

theorem workers_contribution_eq :
  W * C = 300000 → W * (C + 50) = 320000 → W = 400 :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end workers_contribution_eq_l98_98184


namespace no_primes_in_sequence_l98_98225

-- Definitions and conditions derived from the problem statement
variable (a : ℕ → ℕ) -- sequence of natural numbers
variable (increasing : ∀ n, a n < a (n + 1)) -- increasing sequence
variable (is_arith_or_geom : ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) ^ 2 = a n * a (n + 2))) -- arithmetic or geometric progression condition
variable (divisible_by_four : a 0 % 4 = 0 ∧ a 1 % 4 = 0) -- first two numbers divisible by 4

-- The statement to prove: no prime numbers exist in the sequence
theorem no_primes_in_sequence : ∀ n, ¬ (Nat.Prime (a n)) :=
by 
  sorry

end no_primes_in_sequence_l98_98225


namespace find_D_plus_E_plus_F_l98_98556

noncomputable def g (x : ℝ) (D E F : ℝ) : ℝ := (x^2) / (D * x^2 + E * x + F)

theorem find_D_plus_E_plus_F (D E F : ℤ) 
  (h1 : ∀ x : ℝ, x > 3 → g x D E F > 0.3)
  (h2 : ∀ x : ℝ, ¬(D * x^2 + E * x + F = 0 ↔ (x = -3 ∨ x = 2))) :
  D + E + F = -8 :=
sorry

end find_D_plus_E_plus_F_l98_98556


namespace ratio_of_incomes_l98_98439

theorem ratio_of_incomes 
  (E1 E2 I1 I2 : ℕ)
  (h1 : E1 / E2 = 3 / 2)
  (h2 : E1 = I1 - 1200)
  (h3 : E2 = I2 - 1200)
  (h4 : I1 = 3000) :
  I1 / I2 = 5 / 4 :=
sorry

end ratio_of_incomes_l98_98439


namespace chef_meals_prepared_for_dinner_l98_98782

theorem chef_meals_prepared_for_dinner (lunch_meals_prepared lunch_meals_sold dinner_meals_total : ℕ) 
  (h1 : lunch_meals_prepared = 17)
  (h2 : lunch_meals_sold = 12)
  (h3 : dinner_meals_total = 10) :
  (dinner_meals_total - (lunch_meals_prepared - lunch_meals_sold)) = 5 :=
by
  -- Lean proof code to proceed from here
  sorry

end chef_meals_prepared_for_dinner_l98_98782


namespace james_marbles_left_l98_98084

def total_initial_marbles : Nat := 28
def marbles_in_bag_A : Nat := 4
def marbles_in_bag_B : Nat := 6
def marbles_in_bag_C : Nat := 2
def marbles_in_bag_D : Nat := 8
def marbles_in_bag_E : Nat := 4
def marbles_in_bag_F : Nat := 4

theorem james_marbles_left : total_initial_marbles - marbles_in_bag_D = 20 := by
  -- James has 28 marbles initially.
  -- He gives away Bag D which has 8 marbles.
  -- 28 - 8 = 20
  sorry

end james_marbles_left_l98_98084


namespace Anne_Katherine_savings_l98_98708

theorem Anne_Katherine_savings :
  ∃ A K : ℕ, (A - 150 = K / 3) ∧ (2 * K = 3 * A) ∧ (A + K = 750) := 
sorry

end Anne_Katherine_savings_l98_98708


namespace find_smallest_n_l98_98906

open Matrix Complex

noncomputable def rotation_matrix := ![
  ![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
  ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]
]

def I_2 := (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem find_smallest_n (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = rotation_matrix) : 
  ∃ (n : ℕ), 0 < n ∧ A ^ n = I_2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → A ^ m ≠ I_2 :=
by {
  sorry
}

end find_smallest_n_l98_98906


namespace john_books_purchase_l98_98978

theorem john_books_purchase : 
  let john_money := 4575
  let book_price := 325
  john_money / book_price = 14 :=
by
  sorry

end john_books_purchase_l98_98978


namespace circle_radius_eq_two_l98_98307

theorem circle_radius_eq_two (x y : ℝ) : (x^2 + y^2 + 1 = 2 * x + 4 * y) → (∃ c : ℝ × ℝ, ∃ r : ℝ, ((x - c.1)^2 + (y - c.2)^2 = r^2) ∧ r = 2) := by
  sorry

end circle_radius_eq_two_l98_98307


namespace standard_parts_bounds_l98_98473

noncomputable def n : ℕ := 900
noncomputable def p : ℝ := 0.9
noncomputable def confidence_level : ℝ := 0.95
noncomputable def lower_bound : ℝ := 792
noncomputable def upper_bound : ℝ := 828

theorem standard_parts_bounds : 
  792 ≤ n * p - 1.96 * (n * p * (1 - p)).sqrt ∧ 
  n * p + 1.96 * (n * p * (1 - p)).sqrt ≤ 828 :=
sorry

end standard_parts_bounds_l98_98473


namespace triangle_angle_sum_l98_98147

theorem triangle_angle_sum (α β γ : ℝ) (h : α + β + γ = 180) (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) : false :=
sorry

end triangle_angle_sum_l98_98147


namespace unknown_number_is_six_l98_98828

theorem unknown_number_is_six (n : ℝ) (h : 12 * n^4 / 432 = 36) : n = 6 :=
by 
  -- This will be the placeholder for the proof
  sorry

end unknown_number_is_six_l98_98828


namespace right_triangle_area_l98_98692

-- Define the lengths of the legs of the right triangle
def leg_length : ℝ := 1

-- State the theorem
theorem right_triangle_area (a b : ℝ) (h1 : a = leg_length) (h2 : b = leg_length) : 
  (1 / 2) * a * b = 1 / 2 :=
by
  rw [h1, h2]
  -- From the substitutions above, it simplifies to:
  sorry

end right_triangle_area_l98_98692


namespace pow_mod_1110_l98_98172

theorem pow_mod_1110 (n : ℕ) (h₀ : 0 ≤ n ∧ n < 1111)
    (h₁ : 2^1110 % 11 = 1) (h₂ : 2^1110 % 101 = 14) : 
    n = 1024 := 
sorry

end pow_mod_1110_l98_98172


namespace star_comm_l98_98766

section SymmetricOperation

variable {S : Type*} 
variable (star : S → S → S)
variable (symm : ∀ a b : S, star a b = star (star b a) (star b a)) 

theorem star_comm (a b : S) : star a b = star b a := 
by 
  sorry

end SymmetricOperation

end star_comm_l98_98766


namespace articles_produced_l98_98541

theorem articles_produced (a b c p q r : Nat) (h : a * b * c = abc) : p * q * r = pqr := sorry

end articles_produced_l98_98541


namespace arithmetic_sqrt_of_4_l98_98914

theorem arithmetic_sqrt_of_4 : ∃ x : ℚ, x^2 = 4 ∧ x > 0 → x = 2 :=
by {
  sorry
}

end arithmetic_sqrt_of_4_l98_98914


namespace quadratic_roots_l98_98110

theorem quadratic_roots (c : ℝ) 
  (h : ∀ x : ℝ, (x^2 - 3*x + c = 0) ↔ (x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2)) :
  c = 9 / 5 :=
by
  sorry

end quadratic_roots_l98_98110


namespace product_of_solutions_l98_98602

theorem product_of_solutions (x : ℚ) (h : abs (12 / x + 3) = 2) :
  x = -12 ∨ x = -12 / 5 → x₁ * x₂ = 144 / 5 := by
  sorry

end product_of_solutions_l98_98602


namespace alice_commission_percentage_l98_98352

-- Definitions from the given problem
def basic_salary : ℝ := 240
def total_sales : ℝ := 2500
def savings : ℝ := 29
def savings_percentage : ℝ := 0.10

-- The target percentage we want to prove
def commission_percentage : ℝ := 0.02

-- The statement we aim to prove
theorem alice_commission_percentage :
  commission_percentage =
  (savings / savings_percentage - basic_salary) / total_sales := 
sorry

end alice_commission_percentage_l98_98352


namespace composite_numbers_characterization_l98_98670

noncomputable def is_sum_and_product_seq (n : ℕ) (seq : List ℕ) : Prop :=
  seq.sum = n ∧ seq.prod = n ∧ 2 ≤ seq.length ∧ ∀ x ∈ seq, 1 ≤ x

theorem composite_numbers_characterization (n : ℕ) :
  (∃ seq : List ℕ, is_sum_and_product_seq n seq) ↔ ¬Nat.Prime n ∧ 1 < n :=
sorry

end composite_numbers_characterization_l98_98670


namespace hyperbola_parabola_foci_l98_98090

-- Definition of the hyperbola
def hyperbola (k : ℝ) (x y : ℝ) : Prop := y^2 / 5 - x^2 / k = 1

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Condition that both curves have the same foci
def same_foci (focus : ℝ) (x y : ℝ) : Prop := focus = 3 ∧ (parabola x y → ((0, focus) : ℝ×ℝ) = (0, 3)) ∧ (∃ k : ℝ, hyperbola k x y ∧ ((0, focus) : ℝ×ℝ) = (0, 3))

theorem hyperbola_parabola_foci (k : ℝ) (x y : ℝ) : same_foci 3 x y → k = -4 := 
by {
  sorry
}

end hyperbola_parabola_foci_l98_98090


namespace eccentricity_of_hyperbola_l98_98042

open Real

-- Hyperbola parameters and conditions
variables (a b c e : ℝ)
-- Ensure a > 0, b > 0
axiom h_a_pos : a > 0
axiom h_b_pos : b > 0
-- Hyperbola equation
axiom hyperbola_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
-- Coincidence of right focus and center of circle
axiom circle_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 3 = 0 → (x, y) = (2, 0)
-- Distance from focus to asymptote is 1
axiom distance_focus_to_asymptote : b = 1

-- Prove the eccentricity e of the hyperbola is 2sqrt(3)/3
theorem eccentricity_of_hyperbola : e = 2 * sqrt 3 / 3 := sorry

end eccentricity_of_hyperbola_l98_98042


namespace vasya_numbers_l98_98616

theorem vasya_numbers (x y : ℚ) (h : x + y = xy ∧ xy = x / y) : x = 1 / 2 ∧ y = -1 :=
by {
  sorry
}

end vasya_numbers_l98_98616


namespace not_lengths_of_external_diagonals_l98_98367

theorem not_lengths_of_external_diagonals (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) :
  (¬ (a = 5 ∧ b = 6 ∧ c = 9)) :=
by
  sorry

end not_lengths_of_external_diagonals_l98_98367


namespace jeans_cost_l98_98903

theorem jeans_cost (initial_money pizza_cost soda_cost quarter_value after_quarters : ℝ) (quarters_count: ℕ) :
  initial_money = 40 ->
  pizza_cost = 2.75 ->
  soda_cost = 1.50 ->
  quarter_value = 0.25 ->
  quarters_count = 97 ->
  after_quarters = quarters_count * quarter_value ->
  initial_money - (pizza_cost + soda_cost) - after_quarters = 11.50 :=
by
  intros h_initial h_pizza h_soda h_quarter_val h_quarters h_after_quarters
  sorry

end jeans_cost_l98_98903


namespace incorrect_statement_l98_98236

def data_set : List ℤ := [10, 8, 6, 9, 8, 7, 8]

theorem incorrect_statement : 
  let mode := 8
  let median := 8
  let mean := 8
  let variance := 8
  (∃ x ∈ data_set, x ≠ 8) → -- suppose there is at least one element in the dataset not equal to 8
  (1 / 7 : ℚ) * (4 + 0 + 4 + 1 + 0 + 1 + 0) ≠ 8 := -- calculating real variance from dataset
by
  sorry

end incorrect_statement_l98_98236


namespace probability_of_sum_14_l98_98484

-- Define the set of faces on a tetrahedral die
def faces : Set ℕ := {2, 4, 6, 8}

-- Define the event where the sum of two rolls equals 14
def event_sum_14 (a b : ℕ) : Prop := a + b = 14 ∧ a ∈ faces ∧ b ∈ faces

-- Define the total number of outcomes when rolling two dice
def total_outcomes : ℕ := 16

-- Define the number of successful outcomes for the event where the sum is 14
def successful_outcomes : ℕ := 2

-- The probability of rolling a sum of 14 with two such tetrahedral dice
def probability_sum_14 : ℚ := successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_sum_14 : probability_sum_14 = 1 / 8 := 
by sorry

end probability_of_sum_14_l98_98484


namespace solve_for_X_l98_98770

theorem solve_for_X (X : ℝ) (h : (X ^ (5 / 4)) = 32 * (32 ^ (1 / 16))) :
  X =  16 * (2 ^ (1 / 4)) :=
sorry

end solve_for_X_l98_98770


namespace chocolate_bars_count_l98_98904

theorem chocolate_bars_count (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
    (h_milk : milk_chocolate = 25)
    (h_almond : almond_chocolate = 25)
    (h_white : white_chocolate = 25)
    (h_percent : milk_chocolate = almond_chocolate ∧ almond_chocolate = white_chocolate ∧ white_chocolate = dark_chocolate) :
    dark_chocolate = 25 := by
  sorry

end chocolate_bars_count_l98_98904


namespace find_number_l98_98725

/--
A number is added to 5, then multiplied by 5, then subtracted by 5, and then divided by 5. 
The result is still 5. Prove that the number is 1.
-/
theorem find_number (x : ℝ) (h : ((5 * (x + 5) - 5) / 5 = 5)) : x = 1 := 
  sorry

end find_number_l98_98725


namespace quadratic_inequality_solution_l98_98419

def range_of_k (k : ℝ) : Prop := (k ≥ 4) ∨ (k ≤ 2)

theorem quadratic_inequality_solution (k : ℝ) (x : ℝ) (h : x = 1) :
  k^2*x^2 - 6*k*x + 8 ≥ 0 → range_of_k k := 
sorry

end quadratic_inequality_solution_l98_98419


namespace students_participated_in_function_l98_98851

theorem students_participated_in_function :
  ∀ (B G : ℕ),
  B + G = 800 →
  (3 / 4 : ℚ) * G = 150 →
  (2 / 3 : ℚ) * B + 150 = 550 :=
by
  intros B G h1 h2
  sorry

end students_participated_in_function_l98_98851


namespace calculate_PC_l98_98564
noncomputable def ratio (a b : ℝ) : ℝ := a / b

theorem calculate_PC (AB BC CA PC PA : ℝ) (h1: AB = 6) (h2: BC = 10) (h3: CA = 8)
  (h4: ratio PC PA = ratio 8 6)
  (h5: ratio PA (PC + 10) = ratio 6 10) :
  PC = 40 :=
sorry

end calculate_PC_l98_98564


namespace sufficient_but_not_necessary_l98_98381

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬((x < -1 ∨ x > 1) → (x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l98_98381


namespace total_profit_Q2_is_correct_l98_98092

-- Conditions as definitions
def profit_Q1_A := 1500
def profit_Q1_B := 2000
def profit_Q1_C := 1000

def profit_Q2_A := 2500
def profit_Q2_B := 3000
def profit_Q2_C := 1500

def profit_Q3_A := 3000
def profit_Q3_B := 2500
def profit_Q3_C := 3500

def profit_Q4_A := 2000
def profit_Q4_B := 3000
def profit_Q4_C := 2000

-- The total profit calculation for the second quarter
def total_profit_Q2 := profit_Q2_A + profit_Q2_B + profit_Q2_C

-- Proof statement
theorem total_profit_Q2_is_correct : total_profit_Q2 = 7000 := by
  sorry

end total_profit_Q2_is_correct_l98_98092


namespace geometric_series_sum_l98_98089

noncomputable def T (r : ℝ) := 15 / (1 - r)

theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) (H : T b * T (-b) = 3240) : T b + T (-b) = 432 := 
by sorry

end geometric_series_sum_l98_98089


namespace min_value_sq_sum_l98_98131

theorem min_value_sq_sum (x1 x2 : ℝ) (h : x1 * x2 = 2013) : (x1 + x2)^2 ≥ 8052 :=
by
  sorry

end min_value_sq_sum_l98_98131


namespace arithmetic_mean_is_correct_l98_98769

variable (x a : ℝ)
variable (hx : x ≠ 0)

theorem arithmetic_mean_is_correct : 
  (1/2 * ((x + 2 * a) / x - 1 + (x - 3 * a) / x + 1)) = (1 - a / (2 * x)) := 
  sorry

end arithmetic_mean_is_correct_l98_98769


namespace blue_lipstick_students_l98_98072

def total_students : ℕ := 200
def students_with_lipstick : ℕ := total_students / 2
def students_with_red_lipstick : ℕ := students_with_lipstick / 4
def students_with_blue_lipstick : ℕ := students_with_red_lipstick / 5

theorem blue_lipstick_students : students_with_blue_lipstick = 5 :=
by
  sorry

end blue_lipstick_students_l98_98072


namespace eval_expr_l98_98086

theorem eval_expr : (2/5) + (3/8) - (1/10) = 27/40 :=
by
  sorry

end eval_expr_l98_98086


namespace painted_sphere_area_proportionality_l98_98720

theorem painted_sphere_area_proportionality
  (r : ℝ)
  (R_inner R_outer : ℝ)
  (A_inner : ℝ)
  (h_r : r = 1)
  (h_R_inner : R_inner = 4)
  (h_R_outer : R_outer = 6)
  (h_A_inner : A_inner = 47) :
  ∃ A_outer : ℝ, A_outer = 105.75 :=
by
  have ratio := (R_outer / R_inner)^2
  have A_outer := A_inner * ratio
  use A_outer
  sorry

end painted_sphere_area_proportionality_l98_98720


namespace shortest_side_length_l98_98572

theorem shortest_side_length (A B C : ℝ) (a b c : ℝ)
  (h_sinA : Real.sin A = 5 / 13)
  (h_cosB : Real.cos B = 3 / 5)
  (h_longest : c = 63)
  (h_angles : A < B ∧ C = π - (A + B)) :
  a = 25 := by
sorry

end shortest_side_length_l98_98572


namespace remainder_of_470521_div_5_l98_98124

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := 
by sorry

end remainder_of_470521_div_5_l98_98124


namespace min_students_l98_98217

theorem min_students (M D : ℕ) (hD : D = 5) (h_ratio : (M: ℚ) / (M + D) > 0.6) : M + D = 13 :=
by 
  sorry

end min_students_l98_98217


namespace fraction_meaningful_l98_98880

theorem fraction_meaningful (x : ℝ) : 2 * x - 1 ≠ 0 ↔ x ≠ 1 / 2 :=
by
  sorry

end fraction_meaningful_l98_98880


namespace range_of_m_l98_98112

-- Conditions:
def is_opposite_sides_of_line (p1 p2 : ℝ × ℝ) (a b m : ℝ) : Prop :=
  let l1 := a * p1.1 + b * p1.2 + m
  let l2 := a * p2.1 + b * p2.2 + m
  l1 * l2 < 0

-- Point definitions:
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-4, -2)

-- Line definition with coefficients
def a : ℝ := 2
def b : ℝ := 1

-- Proof Goal:
theorem range_of_m (m : ℝ) : is_opposite_sides_of_line point1 point2 a b m ↔ -5 < m ∧ m < 10 :=
by sorry

end range_of_m_l98_98112


namespace mod_3_pow_2040_eq_1_mod_5_l98_98455

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l98_98455


namespace find_sum_l98_98451

-- Define the prime conditions
variables (P : ℝ) (SI15 SI12 : ℝ)

-- Assume conditions for the problem
axiom h1 : SI15 = P * 15 / 100 * 2
axiom h2 : SI12 = P * 12 / 100 * 2
axiom h3 : SI15 - SI12 = 840

-- Prove that P = 14000
theorem find_sum : P = 14000 :=
sorry

end find_sum_l98_98451


namespace eventually_constant_sequence_a_floor_l98_98715

noncomputable def sequence_a (n : ℕ) : ℝ := sorry
noncomputable def sequence_b (n : ℕ) : ℝ := sorry

axiom base_conditions : 
  (sequence_a 1 = 1) ∧
  (sequence_b 1 = 2) ∧
  (∀ n, sequence_a (n + 1) * sequence_b n = 1 + sequence_a n + sequence_a n * sequence_b n) ∧
  (∀ n, sequence_b (n + 1) * sequence_a n = 1 + sequence_b n + sequence_a n * sequence_b n)

theorem eventually_constant_sequence_a_floor:
  (∃ N, ∀ n ≥ N, 4 < sequence_a n ∧ sequence_a n < 5) →
  (∃ N, ∀ n ≥ N, Int.floor (sequence_a n) = 4) :=
sorry

end eventually_constant_sequence_a_floor_l98_98715


namespace remainder_of_sum_mod_9_l98_98868

theorem remainder_of_sum_mod_9 :
  (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 :=
by
  sorry

end remainder_of_sum_mod_9_l98_98868


namespace intersection_of_A_and_B_l98_98636

def set_A : Set ℝ := {x | -x^2 - x + 6 > 0}
def set_B : Set ℝ := {x | 5 / (x - 3) ≤ -1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l98_98636


namespace line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l98_98823

-- Define the set of people
def people : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : ℕ := 1 
def B : ℕ := 2
def C : ℕ := 3

-- Proof Problem 1: Prove that there are 1800 ways to line up 5 people out of 7 given A must be included.
theorem line_up_including_A : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 2: Prove that there are 1800 ways to line up 5 people out of 7 given A, B, and C are not all included.
theorem line_up_excluding_all_ABC : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 3: Prove that there are 144 ways to line up 5 people out of 7 given A, B, and C are all included, A and B are adjacent, and C is not adjacent to A or B.
theorem line_up_adjacent_AB_not_adjacent_C : Finset ℕ → ℕ :=
by
  sorry

end line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l98_98823


namespace intersection_M_N_l98_98298

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ -3}

-- Prove the intersection of M and N is [1, 2)
theorem intersection_M_N : (M ∩ N) = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l98_98298


namespace triangle_inequality_proof_l98_98492

theorem triangle_inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
    sorry

end triangle_inequality_proof_l98_98492


namespace price_of_basic_computer_l98_98312

-- Conditions
variables (C P : ℝ)
axiom cond1 : C + P = 2500
axiom cond2 : 3 * P = C + 500

-- Prove that the price of the basic computer is $1750
theorem price_of_basic_computer : C = 1750 :=
by 
  sorry

end price_of_basic_computer_l98_98312


namespace probability_of_one_pair_one_triplet_l98_98470

-- Define the necessary conditions
def six_sided_die_rolls (n : ℕ) : ℕ := 6 ^ n

def successful_outcomes : ℕ :=
  6 * 20 * 5 * 3 * 4

def total_outcomes : ℕ :=
  six_sided_die_rolls 6

def probability_success : ℚ :=
  successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_one_pair_one_triplet :
  probability_success = 25/162 :=
sorry

end probability_of_one_pair_one_triplet_l98_98470


namespace find_original_number_l98_98772

def original_number_divide_multiply (x : ℝ) : Prop :=
  (x / 12) * 24 = x + 36

theorem find_original_number (x : ℝ) (h : original_number_divide_multiply x) : x = 36 :=
by
  sorry

end find_original_number_l98_98772


namespace minimum_throws_for_repeated_sum_l98_98928

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l98_98928


namespace solve_rational_numbers_l98_98327

theorem solve_rational_numbers:
  ∃ (a b c d : ℚ),
    8 * a^2 - 3 * b^2 + 5 * c^2 + 16 * d^2 - 10 * a * b + 42 * c * d + 18 * a + 22 * b - 2 * c - 54 * d = 42 ∧
    15 * a^2 - 3 * b^2 + 21 * c^2 - 5 * d^2 + 4 * a * b + 32 * c * d - 28 * a + 14 * b - 54 * c - 52 * d = -22 ∧
    a = 4 / 7 ∧ b = 19 / 7 ∧ c = 29 / 19 ∧ d = -6 / 19 :=
  sorry

end solve_rational_numbers_l98_98327


namespace positive_real_solution_unique_l98_98723

theorem positive_real_solution_unique :
  (∃! x : ℝ, 0 < x ∧ x^12 + 5 * x^11 - 3 * x^10 + 2000 * x^9 - 1500 * x^8 = 0) :=
sorry

end positive_real_solution_unique_l98_98723


namespace find_f_a5_a6_l98_98274

-- Define the function properties and initial conditions
variables {f : ℝ → ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions for the function f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_period : ∀ x, f (3/2 - x) = f x
axiom f_minus_2 : f (-2) = -3

-- Initial sequence condition and recursive relation
axiom a_1 : a 1 = -1
axiom S_def : ∀ n, S n = 2 * a n + n
axiom seq_recursive : ∀ n ≥ 2, S (n - 1) = 2 * a (n - 1) + (n - 1)

-- Theorem to prove
theorem find_f_a5_a6 : f (a 5) + f (a 6) = 3 := by
  sorry

end find_f_a5_a6_l98_98274


namespace x_is_integer_l98_98446

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ k1 : ℤ, x^2 - x = k1)
  (h2 : ∃ (n : ℕ) (_ : n > 2) (k2 : ℤ), x^n - x = k2) : 
  ∃ (m : ℤ), x = m := 
sorry

end x_is_integer_l98_98446


namespace oil_bill_january_l98_98033

-- Define the constants and variables
variables (F J : ℝ)

-- Define the conditions
def condition1 : Prop := F / J = 5 / 4
def condition2 : Prop := (F + 45) / J = 3 / 2

-- Define the main theorem stating the proof problem
theorem oil_bill_january 
  (h1 : condition1 F J) 
  (h2 : condition2 F J) : 
  J = 180 :=
sorry

end oil_bill_january_l98_98033


namespace minimum_boxes_to_eliminate_l98_98224

theorem minimum_boxes_to_eliminate (total_boxes remaining_boxes : ℕ) 
  (high_value_boxes : ℕ) (h1 : total_boxes = 30) (h2 : high_value_boxes = 10)
  (h3 : remaining_boxes = total_boxes - 20) :
  remaining_boxes ≥ high_value_boxes → remaining_boxes = 10 :=
by 
  sorry

end minimum_boxes_to_eliminate_l98_98224


namespace integral_sin_from_0_to_pi_div_2_l98_98869

theorem integral_sin_from_0_to_pi_div_2 :
  ∫ x in (0 : ℝ)..(Real.pi / 2), Real.sin x = 1 := by
  sorry

end integral_sin_from_0_to_pi_div_2_l98_98869


namespace calc_expression_l98_98753

noncomputable def x := (3 + Real.sqrt 5) / 2 -- chosen from one of the roots of the quadratic equation x^2 - 3x + 1

theorem calc_expression (h : x + 1 / x = 3) : 
  (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 7 + 3 * Real.sqrt 5 := 
by 
  sorry

end calc_expression_l98_98753


namespace employee_payment_l98_98631

theorem employee_payment (X Y : ℝ) 
  (h1 : X + Y = 880) 
  (h2 : X = 1.2 * Y) : Y = 400 := by
  sorry

end employee_payment_l98_98631


namespace find_prime_p_l98_98773

def is_prime (p: ℕ) : Prop := Nat.Prime p

def is_product_of_three_distinct_primes (n: ℕ) : Prop :=
  ∃ (p1 p2 p3: ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3

theorem find_prime_p (p: ℕ) (hp: is_prime p) :
  (∃ x y z: ℕ, x^p + y^p + z^p - x - y - z = 30) ↔ (p = 2 ∨ p = 3 ∨ p = 5) := 
sorry

end find_prime_p_l98_98773


namespace simplify_fraction_l98_98344

theorem simplify_fraction : (140 / 9800) * 35 = 1 / 70 := 
by
  -- Proof steps would go here.
  sorry

end simplify_fraction_l98_98344


namespace expand_polynomial_l98_98069

theorem expand_polynomial (z : ℂ) :
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 :=
by sorry

end expand_polynomial_l98_98069


namespace area_of_one_cookie_l98_98448

theorem area_of_one_cookie (L W : ℝ)
    (W_eq_15 : W = 15)
    (circumference_condition : 4 * L + 2 * W = 70) :
    L * W = 150 :=
by
  sorry

end area_of_one_cookie_l98_98448


namespace altitude_inequality_l98_98669

theorem altitude_inequality
  (a b m_a m_b : ℝ)
  (h1 : a > b)
  (h2 : a * m_a = b * m_b) :
  a^2010 + m_a^2010 ≥ b^2010 + m_b^2010 :=
sorry

end altitude_inequality_l98_98669


namespace milk_processing_days_required_l98_98420

variable (a m x : ℝ) (n : ℝ)

theorem milk_processing_days_required
  (h1 : (n - a) * (x + m) = nx)
  (h2 : ax + (10 * a / 9) * x + (5 * a / 9) * m = 2 / 3)
  (h3 : nx = 1 / 2) :
  n = 2 * a :=
by sorry

end milk_processing_days_required_l98_98420


namespace smallest_x_l98_98010

theorem smallest_x {
    x : ℤ
} : (x % 11 = 9) ∧ (x % 13 = 11) ∧ (x % 15 = 13) → x = 2143 := by
sorry

end smallest_x_l98_98010


namespace carol_first_to_roll_six_l98_98220

def probability_roll (x : ℕ) (success : ℕ) : ℚ := success / x

def first_to_roll_six_probability (a b c : ℕ) : ℚ :=
  let p_six : ℚ := probability_roll 6 1
  let p_not_six : ℚ := 1 - p_six
  let cycle_prob : ℚ := p_not_six * p_not_six * p_six
  let continue_prob : ℚ := p_not_six * p_not_six * p_not_six
  let geometric_sum : ℚ := cycle_prob / (1 - continue_prob)
  geometric_sum

theorem carol_first_to_roll_six :
  first_to_roll_six_probability 1 1 1 = 25 / 91 := 
sorry

end carol_first_to_roll_six_l98_98220


namespace chocolates_difference_l98_98981

theorem chocolates_difference (robert_chocolates : ℕ) (nickel_chocolates : ℕ) (h1 : robert_chocolates = 7) (h2 : nickel_chocolates = 5) : robert_chocolates - nickel_chocolates = 2 :=
by sorry

end chocolates_difference_l98_98981


namespace shirts_washed_total_l98_98866

theorem shirts_washed_total (short_sleeve_shirts long_sleeve_shirts : Nat) (h1 : short_sleeve_shirts = 4) (h2 : long_sleeve_shirts = 5) : short_sleeve_shirts + long_sleeve_shirts = 9 := by
  sorry

end shirts_washed_total_l98_98866


namespace compare_rat_neg_l98_98813

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l98_98813


namespace nesbitts_inequality_l98_98482

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (a + c) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end nesbitts_inequality_l98_98482


namespace find_a_l98_98693

theorem find_a (a : ℝ) (h1 : 0 < a)
  (c1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (c2 : ∀ x y : ℝ, x^2 + y^2 + 2 * a * y - 6 = 0)
  (h_chord : (2 * Real.sqrt 3) = 2 * Real.sqrt 3) :
  a = 1 := 
sorry

end find_a_l98_98693


namespace range_of_a_l98_98185

open Real

-- The quadratic expression
def quadratic (a x : ℝ) : ℝ := a*x^2 + 2*x + a

-- The condition of the problem
def quadratic_nonnegative_for_all (a : ℝ) := ∀ x : ℝ, quadratic a x ≥ 0

-- The theorem to be proven
theorem range_of_a (a : ℝ) (h : quadratic_nonnegative_for_all a) : a ≥ 1 :=
sorry

end range_of_a_l98_98185


namespace fred_likes_12_pairs_of_digits_l98_98041

theorem fred_likes_12_pairs_of_digits :
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs ↔ ∃ (n : ℕ), n < 100 ∧ n % 8 = 0 ∧ n = 10 * a + b) ∧
    pairs.card = 12) :=
by
  sorry

end fred_likes_12_pairs_of_digits_l98_98041


namespace equilateral_triangle_fixed_area_equilateral_triangle_max_area_l98_98119

theorem equilateral_triangle_fixed_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = minimized ∨ a + b + c = minimized ∨ a^2 + b^2 + c^2 = minimized ∨ R = minimized) →
    (a = b ∧ b = c) :=
by
  sorry

theorem equilateral_triangle_max_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = fixed ∨ a + b + c = fixed ∨ a^2 + b^2 + c^2 = fixed ∨ R = fixed) →
  (Δ = maximized) →
    (a = b ∧ b = c) :=
by
  sorry

end equilateral_triangle_fixed_area_equilateral_triangle_max_area_l98_98119


namespace find_f_zero_l98_98055

theorem find_f_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y - x * y) 
  (h1 : f 1 = 1) : 
  f 0 = 0 := 
sorry

end find_f_zero_l98_98055


namespace age_difference_l98_98413

theorem age_difference (A B : ℕ) (h1 : B = 34) (h2 : A + 10 = 2 * (B - 10)) : A - B = 4 :=
by
  sorry

end age_difference_l98_98413


namespace square_pyramid_intersection_area_l98_98280

theorem square_pyramid_intersection_area (a b c d e : ℝ) (h_midpoints : a = 2 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4) : 
  ∃ p : ℝ, (p = 80) :=
by
  sorry

end square_pyramid_intersection_area_l98_98280


namespace similar_polygons_perimeter_ratio_l98_98979

-- Define the main function to assert the proportional relationship
theorem similar_polygons_perimeter_ratio (x y : ℕ) (h1 : 9 * y^2 = 64 * x^2) : x * 8 = y * 3 :=
by sorry

-- noncomputable if needed (only necessary when computation is involved, otherwise omit)

end similar_polygons_perimeter_ratio_l98_98979


namespace simplest_square_root_l98_98781

theorem simplest_square_root : 
  let a1 := Real.sqrt 20
  let a2 := Real.sqrt 2
  let a3 := Real.sqrt (1 / 2)
  let a4 := Real.sqrt 0.2
  a2 = Real.sqrt 2 ∧
  (a1 ≠ Real.sqrt 2 ∧ a3 ≠ Real.sqrt 2 ∧ a4 ≠ Real.sqrt 2) :=
by {
  -- Here, we fill in the necessary proof steps, but it's omitted for now.
  sorry
}

end simplest_square_root_l98_98781


namespace slope_of_line_l98_98052

/-- 
Given points M(1, 2) and N(3, 4), prove that the slope of the line passing through these points is 1.
-/
theorem slope_of_line (x1 y1 x2 y2 : ℝ) (hM : x1 = 1 ∧ y1 = 2) (hN : x2 = 3 ∧ y2 = 4) : 
  (y2 - y1) / (x2 - x1) = 1 :=
by
  -- The proof is omitted here because only the statement is required.
  sorry

end slope_of_line_l98_98052


namespace first_worker_time_l98_98786

theorem first_worker_time
  (T : ℝ) 
  (hT : T ≠ 0)
  (h_comb : (T + 8) / (8 * T) = 1 / 3.428571428571429) :
  T = 8 / 7 :=
by
  sorry

end first_worker_time_l98_98786
