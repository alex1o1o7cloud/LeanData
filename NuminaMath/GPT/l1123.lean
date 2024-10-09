import Mathlib

namespace proof_U_eq_A_union_complement_B_l1123_112348

noncomputable def U : Set Nat := {1, 2, 3, 4, 5, 7}
noncomputable def A : Set Nat := {1, 3, 5, 7}
noncomputable def B : Set Nat := {3, 5}
noncomputable def complement_U_B := U \ B

theorem proof_U_eq_A_union_complement_B : U = A ∪ complement_U_B := by
  sorry

end proof_U_eq_A_union_complement_B_l1123_112348


namespace trigonometric_identity_proof_l1123_112369

theorem trigonometric_identity_proof
  (α : Real)
  (h1 : Real.sin (Real.pi + α) = -Real.sin α)
  (h2 : Real.cos (Real.pi + α) = -Real.cos α)
  (h3 : Real.cos (-α) = Real.cos α)
  (h4 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) :
  Real.sin (Real.pi + α) ^ 2 - Real.cos (Real.pi + α) * Real.cos (-α) + 1 = 2 := 
by
  sorry

end trigonometric_identity_proof_l1123_112369


namespace relative_magnitude_of_reciprocal_l1123_112347

theorem relative_magnitude_of_reciprocal 
  (a b : ℝ) (hab : a < 1 / b) :
  (a > 0 ∧ b > 0 ∧ 1 / a > b) ∨ (a < 0 ∧ b < 0 ∧ 1 / a > b)
   ∨ (a > 0 ∧ b < 0 ∧ 1 / a < b) ∨ (a < 0 ∧ b > 0 ∧ 1 / a < b) :=
by sorry

end relative_magnitude_of_reciprocal_l1123_112347


namespace pond_87_5_percent_algae_free_on_day_17_l1123_112326

/-- The algae in a local pond doubles every day. -/
def algae_doubles_every_day (coverage : ℕ → ℝ) : Prop :=
  ∀ n, coverage (n + 1) = 2 * coverage n

/-- The pond is completely covered in algae on day 20. -/
def pond_completely_covered_on_day_20 (coverage : ℕ → ℝ) : Prop :=
  coverage 20 = 1

/-- Determine the day on which the pond was 87.5% algae-free. -/
theorem pond_87_5_percent_algae_free_on_day_17 (coverage : ℕ → ℝ)
  (h1 : algae_doubles_every_day coverage)
  (h2 : pond_completely_covered_on_day_20 coverage) :
  coverage 17 = 0.125 :=
sorry

end pond_87_5_percent_algae_free_on_day_17_l1123_112326


namespace roots_equal_and_real_l1123_112358

theorem roots_equal_and_real:
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y = 0 ∨ y = -24 / 5)) ∧
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y ≥ 0 ∨ y ≤ -24 / 5)) :=
  by sorry

end roots_equal_and_real_l1123_112358


namespace time_to_run_round_square_field_l1123_112311

theorem time_to_run_round_square_field
  (side : ℝ) (speed_km_hr : ℝ)
  (h_side : side = 45)
  (h_speed_km_hr : speed_km_hr = 9) : 
  (4 * side / (speed_km_hr * 1000 / 3600)) = 72 := 
by 
  sorry

end time_to_run_round_square_field_l1123_112311


namespace find_y_l1123_112321

def custom_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_y (y : ℤ) (h : custom_op y 10 = 90) : y = 11 :=
by
  sorry

end find_y_l1123_112321


namespace compute_value_l1123_112335

variables {p q r : ℝ}

theorem compute_value (h1 : (p * q) / (p + r) + (q * r) / (q + p) + (r * p) / (r + q) = -7)
                      (h2 : (p * r) / (p + r) + (q * p) / (q + p) + (r * q) / (r + q) = 8) :
  (q / (p + q) + r / (q + r) + p / (r + p)) = 9 :=
sorry

end compute_value_l1123_112335


namespace simplify_expression_correct_l1123_112397

noncomputable def simplify_expression : Prop :=
  (1 / (Real.log 3 / Real.log 6 + 1) + 1 / (Real.log 7 / Real.log 15 + 1) + 1 / (Real.log 4 / Real.log 12 + 1)) = -Real.log 84 / Real.log 10

theorem simplify_expression_correct : simplify_expression :=
  by
    sorry

end simplify_expression_correct_l1123_112397


namespace find_m_l1123_112340

-- Define the pattern of splitting cubes into odd numbers
def split_cubes (m : ℕ) : List ℕ := 
  let rec odd_numbers (n : ℕ) : List ℕ :=
    if n = 0 then []
    else (2 * n - 1) :: odd_numbers (n - 1)
  odd_numbers m

-- Define the condition that 59 is part of the split numbers of m^3
def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  n ∈ (split_cubes m)

-- Prove that if 59 is part of the split numbers of m^3, then m = 8
theorem find_m (m : ℕ) (h : is_split_number m 59) : m = 8 := 
sorry

end find_m_l1123_112340


namespace original_profit_percentage_l1123_112301

theorem original_profit_percentage {P S : ℝ}
  (h1 : S = 1100)
  (h2 : P ≠ 0)
  (h3 : 1.17 * P = 1170) :
  (S - P) / P * 100 = 10 :=
by
  sorry

end original_profit_percentage_l1123_112301


namespace sean_needs_six_packs_l1123_112396

/-- 
 Sean needs to replace 2 light bulbs in the bedroom, 
 1 in the bathroom, 1 in the kitchen, and 4 in the basement. 
 He also needs to replace 1/2 of that amount in the garage. 
 The bulbs come 2 per pack. 
 -/
def bedroom_bulbs: ℕ := 2
def bathroom_bulbs: ℕ := 1
def kitchen_bulbs: ℕ := 1
def basement_bulbs: ℕ := 4
def bulbs_per_pack: ℕ := 2

noncomputable def total_bulbs_needed_including_garage: ℕ := 
  let total_rooms_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs
  let garage_bulbs := total_rooms_bulbs / 2
  total_rooms_bulbs + garage_bulbs

noncomputable def total_packs_needed: ℕ := total_bulbs_needed_including_garage / bulbs_per_pack

theorem sean_needs_six_packs : total_packs_needed = 6 :=
by
  sorry

end sean_needs_six_packs_l1123_112396


namespace percentage_of_people_with_diploma_l1123_112377

variable (P : Type) -- P is the type representing people in Country Z.

-- Given Conditions:
def no_diploma_job (population : ℝ) : ℝ := 0.18 * population
def people_with_job (population : ℝ) : ℝ := 0.40 * population
def diploma_no_job (population : ℝ) : ℝ := 0.25 * (0.60 * population)

-- To Prove:
theorem percentage_of_people_with_diploma (population : ℝ) :
  no_diploma_job population + (diploma_no_job population) + (people_with_job population - no_diploma_job population) = 0.37 * population := 
by
  sorry

end percentage_of_people_with_diploma_l1123_112377


namespace find_g_8_l1123_112318

def g (x : ℝ) : ℝ := x^2 + x + 1

theorem find_g_8 : (∀ x : ℝ, g (2*x - 4) = x^2 + x + 1) → g 8 = 43 := 
by sorry

end find_g_8_l1123_112318


namespace total_cost_of_returned_packets_l1123_112381

/--
  Martin bought 10 packets of milk with varying prices.
  The average price (arithmetic mean) of all the packets is 25¢.
  If Martin returned three packets to the retailer, and the average price of the remaining packets was 20¢,
  then the total cost, in cents, of the three returned milk packets is 110¢.
-/
theorem total_cost_of_returned_packets 
  (T10 : ℕ) (T7 : ℕ) (average_price_10 : T10 / 10 = 25)
  (average_price_7 : T7 / 7 = 20) :
  (T10 - T7 = 110) := 
sorry

end total_cost_of_returned_packets_l1123_112381


namespace total_votes_l1123_112304

theorem total_votes (A B C V : ℝ)
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V)
  : V = 60000 := 
sorry

end total_votes_l1123_112304


namespace ratio_a7_b7_l1123_112344

variable (a b : ℕ → ℕ) -- Define sequences a and b
variable (S T : ℕ → ℕ) -- Define sums S and T

-- Define conditions: arithmetic sequences and given ratio
variable (h_arith_a : ∀ n, a (n + 1) - a n = a 1)
variable (h_arith_b : ∀ n, b (n + 1) - b n = b 1)
variable (h_sum_a : ∀ n, S n = (n + 1) * a 1 + n * a n)
variable (h_sum_b : ∀ n, T n = (n + 1) * b 1 + n * b n)
variable (h_ratio : ∀ n, (S n) / (T n) = (3 * n + 2) / (2 * n))

-- Define the problem statement using the given conditions
theorem ratio_a7_b7 : (a 7) / (b 7) = 41 / 26 :=
by
  sorry

end ratio_a7_b7_l1123_112344


namespace gumballs_problem_l1123_112303

theorem gumballs_problem 
  (L x : ℕ)
  (h1 : 19 ≤ (17 + L + x) / 3 ∧ (17 + L + x) / 3 ≤ 25)
  (h2 : ∃ x_min x_max, x_max - x_min = 18 ∧ x_min = 19 ∧ x = x_min ∨ x = x_max) : 
  L = 21 :=
sorry

end gumballs_problem_l1123_112303


namespace scientific_notation_of_10760000_l1123_112391

theorem scientific_notation_of_10760000 : 
  (10760000 : ℝ) = 1.076 * 10^7 := 
sorry

end scientific_notation_of_10760000_l1123_112391


namespace triangle_area_of_tangent_circles_l1123_112346

/-- 
Given three circles with radii 1, 3, and 5, that are mutually externally tangent and all tangent to 
the same line, the area of the triangle determined by the points where each circle is tangent to the line 
is 6.
-/
theorem triangle_area_of_tangent_circles :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  ∃ (A B C : ℝ × ℝ),
    A = (0, -(r1 : ℝ)) ∧ B = (0, -(r2 : ℝ)) ∧ C = (0, -(r3 : ℝ)) ∧
    (∃ (h : ℝ), ∃ (b : ℝ), h = 4 ∧ b = 3 ∧
    (1 / 2) * h * b = 6) := 
by
  sorry

end triangle_area_of_tangent_circles_l1123_112346


namespace kittens_given_to_Jessica_is_3_l1123_112380

def kittens_initial := 18
def kittens_given_to_Sara := 6
def kittens_now := 9

def kittens_after_Sara := kittens_initial - kittens_given_to_Sara
def kittens_given_to_Jessica := kittens_after_Sara - kittens_now

theorem kittens_given_to_Jessica_is_3 : kittens_given_to_Jessica = 3 := by
  sorry

end kittens_given_to_Jessica_is_3_l1123_112380


namespace cos_seven_pi_six_l1123_112368

theorem cos_seven_pi_six : (Real.cos (7 * Real.pi / 6) = - Real.sqrt 3 / 2) :=
sorry

end cos_seven_pi_six_l1123_112368


namespace y_intercept_of_tangent_line_l1123_112384

def point (x y : ℝ) : Prop := true

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + 4*x - 2*y + 3

theorem y_intercept_of_tangent_line :
  ∃ m b : ℝ,
  (∀ x : ℝ, circle_eq x (m*x + b) = 0 → m * m = 1) ∧
  (∃ P: ℝ × ℝ, P = (-1, 0)) ∧
  ∀ b : ℝ, (∃ m : ℝ, m = 1 ∧ (∃ P: ℝ × ℝ, P = (-1, 0)) ∧ b = 1) := 
sorry

end y_intercept_of_tangent_line_l1123_112384


namespace no_four_digit_number_ending_in_47_is_divisible_by_5_l1123_112306

theorem no_four_digit_number_ending_in_47_is_divisible_by_5 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (n % 100 = 47 → n % 10 ≠ 0 ∧ n % 10 ≠ 5) := by
  intro n
  intro Hn
  intro H47
  sorry

end no_four_digit_number_ending_in_47_is_divisible_by_5_l1123_112306


namespace ball_hits_ground_time_l1123_112322

theorem ball_hits_ground_time :
  ∃ t : ℚ, -20 * t^2 + 30 * t + 50 = 0 ∧ t = 5 / 2 :=
sorry

end ball_hits_ground_time_l1123_112322


namespace triangle_area_l1123_112309

theorem triangle_area (d : ℝ) (h : d = 8 * Real.sqrt 10) (ang : ∀ {α β γ : ℝ}, α = 45 ∨ β = 45 ∨ γ = 45) :
  ∃ A : ℝ, A = 160 :=
by
  sorry

end triangle_area_l1123_112309


namespace LeanProof_l1123_112337

noncomputable def ProblemStatement : Prop :=
  let AB_parallel_YZ := True -- given condition that AB is parallel to YZ
  let AZ := 36 
  let BQ := 15
  let QY := 20
  let similarity_ratio := BQ / QY = 3 / 4
  ∃ QZ : ℝ, AZ = (3 / 4) * QZ + QZ ∧ QZ = 144 / 7

theorem LeanProof : ProblemStatement :=
sorry

end LeanProof_l1123_112337


namespace hyperbola_range_m_l1123_112390

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m - 2) ≠ 0 ∧ (m + 3) ≠ 0 ∧ (x^2 / (m - 2) + y^2 / (m + 3) = 1)) ↔ (-3 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l1123_112390


namespace trajectory_equation_l1123_112331

-- Define the condition that the distance to the coordinate axes is equal.
def equidistantToAxes (x y : ℝ) : Prop :=
  abs x = abs y

-- State the theorem that we need to prove.
theorem trajectory_equation (x y : ℝ) (h : equidistantToAxes x y) : y^2 = x^2 :=
by sorry

end trajectory_equation_l1123_112331


namespace compare_log_exp_powers_l1123_112363

variable (a b c : ℝ)

theorem compare_log_exp_powers (h1 : a = Real.log 0.3 / Real.log 2)
                               (h2 : b = Real.exp (Real.log 2 * 0.1))
                               (h3 : c = Real.exp (Real.log 0.2 * 1.3)) :
  a < c ∧ c < b :=
by
  sorry

end compare_log_exp_powers_l1123_112363


namespace determine_n_l1123_112367

theorem determine_n : ∃ n : ℤ, 0 ≤ n ∧ n < 8 ∧ -2222 % 8 = n := by
  use 2
  sorry

end determine_n_l1123_112367


namespace range_of_m_l1123_112333

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - ((Real.exp x - 1) / (Real.exp x + 1))

theorem range_of_m (m : ℝ) (h : f (4 - m) - f m ≥ 8 - 4 * m) : 2 ≤ m := by
  sorry

end range_of_m_l1123_112333


namespace team_arrangement_count_l1123_112398

-- Definitions of the problem
def veteran_players := 2
def new_players := 3
def total_players := veteran_players + new_players
def team_size := 3

-- Conditions
def condition_veteran : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → Finset.card (team ∩ (Finset.range veteran_players)) ≥ 1

def condition_new_player : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → 
    ∃ (p1 p2 : ℕ), p1 ∈ team ∧ p2 ∈ team ∧ 
    p1 ≠ p2 ∧ p1 < team_size ∧ p2 < team_size ∧
    (p1 ∈ (Finset.Ico veteran_players total_players) ∨ p2 ∈ (Finset.Ico veteran_players total_players))

-- Goal
def number_of_arrangements := 48

-- The statement to prove
theorem team_arrangement_count : condition_veteran → condition_new_player → 
  (∃ (arrangements : ℕ), arrangements = number_of_arrangements) :=
by
  sorry

end team_arrangement_count_l1123_112398


namespace square_side_length_l1123_112355

theorem square_side_length (d : ℝ) (s : ℝ) (h : d = Real.sqrt 2) (h2 : d = Real.sqrt 2 * s) : s = 1 :=
by
  sorry

end square_side_length_l1123_112355


namespace sqrt_expression_eq_l1123_112307

theorem sqrt_expression_eq :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := 
  sorry

end sqrt_expression_eq_l1123_112307


namespace range_of_m_l1123_112313

def f (x : ℝ) : ℝ := x^2 - 4 * x - 6

theorem range_of_m (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ m → -10 ≤ f x ∧ f x ≤ -6) →
  2 ≤ m ∧ m ≤ 4 := 
sorry

end range_of_m_l1123_112313


namespace conference_handshakes_l1123_112349

theorem conference_handshakes (n : ℕ) (h : n = 10) :
  (n * (n - 1)) / 2 = 45 :=
by
  sorry

end conference_handshakes_l1123_112349


namespace number_of_ways_to_divide_l1123_112334

-- Define the given shape
structure Shape :=
  (sides : Nat) -- Number of 3x1 stripes along the sides
  (centre : Nat) -- Size of the central square (3x3)

-- Define the specific problem shape
def problem_shape : Shape :=
  { sides := 4, centre := 9 } -- 3x1 stripes on all sides and a 3x3 centre

-- Theorem stating the number of ways to divide the shape into 1x3 rectangles
theorem number_of_ways_to_divide (s : Shape) (h1 : s.sides = 4) (h2 : s.centre = 9) : 
  ∃ ways, ways = 2 :=
by
  -- The proof is skipped
  sorry

end number_of_ways_to_divide_l1123_112334


namespace vacation_expense_sharing_l1123_112359

def alice_paid : ℕ := 90
def bob_paid : ℕ := 150
def charlie_paid : ℕ := 120
def donna_paid : ℕ := 240
def total_paid : ℕ := alice_paid + bob_paid + charlie_paid + donna_paid
def individual_share : ℕ := total_paid / 4

def alice_owes : ℕ := individual_share - alice_paid
def charlie_owes : ℕ := individual_share - charlie_paid
def donna_owes : ℕ := donna_paid - individual_share

def a : ℕ := charlie_owes
def b : ℕ := donna_owes - (donna_owes - charlie_owes)

theorem vacation_expense_sharing : a - b = 0 :=
by
  sorry

end vacation_expense_sharing_l1123_112359


namespace triangle_is_isosceles_l1123_112352

theorem triangle_is_isosceles (α β γ δ ε : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : α + β = δ) 
  (h3 : β + γ = ε) : 
  α = γ ∨ β = γ ∨ α = β := 
sorry

end triangle_is_isosceles_l1123_112352


namespace jean_average_mark_l1123_112336

/-
  Jean writes five tests and achieves the following marks: 80, 70, 60, 90, and 80.
  Prove that her average mark on these five tests is 76.
-/
theorem jean_average_mark : 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  average_mark = 76 :=
by 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  sorry

end jean_average_mark_l1123_112336


namespace smallest_integer_2023m_54321n_l1123_112362

theorem smallest_integer_2023m_54321n : ∃ (m n : ℤ), 2023 * m + 54321 * n = 1 :=
sorry

end smallest_integer_2023m_54321n_l1123_112362


namespace length_of_walls_l1123_112399

-- Definitions of the given conditions.
def wall_height : ℝ := 12
def third_wall_length : ℝ := 20
def third_wall_height : ℝ := 12
def total_area : ℝ := 960

-- The area of two walls with length L each and height 12 feet.
def two_walls_area (L : ℝ) : ℝ := 2 * L * wall_height

-- The area of the third wall.
def third_wall_area : ℝ := third_wall_length * third_wall_height

-- The proof statement
theorem length_of_walls (L : ℝ) (h1 : two_walls_area L + third_wall_area = total_area) : L = 30 :=
by
  sorry

end length_of_walls_l1123_112399


namespace percentage_subtraction_l1123_112389

theorem percentage_subtraction (P : ℝ) : (700 - (P / 100 * 7000) = 700) → P = 0 :=
by
  sorry

end percentage_subtraction_l1123_112389


namespace coin_difference_l1123_112383

theorem coin_difference (h : ∃ x y z : ℕ, 5*x + 10*y + 20*z = 40) : (∃ x : ℕ, 5*x = 40) → (∃ y : ℕ, 20*y = 40) → 8 - 2 = 6 :=
by
  intros h1 h2
  exact rfl

end coin_difference_l1123_112383


namespace inequality_f_x_f_a_l1123_112382

noncomputable def f (x : ℝ) : ℝ := x * x + x + 13

theorem inequality_f_x_f_a (a x : ℝ) (h : |x - a| < 1) : |f x * f a| < 2 * (|a| + 1) := 
sorry

end inequality_f_x_f_a_l1123_112382


namespace find_m_l1123_112365

def U : Set ℤ := {-1, 2, 3, 6}
def A (m : ℤ) : Set ℤ := {x | x^2 - 5 * x + m = 0}
def complement_U_A (m : ℤ) : Set ℤ := U \ A m

theorem find_m (m : ℤ) (hU : U = {-1, 2, 3, 6}) (hcomp : complement_U_A m = {2, 3}) :
  m = -6 := by
  sorry

end find_m_l1123_112365


namespace expression_values_l1123_112316

-- Define the conditions as a predicate
def conditions (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a^2 - b * c = b^2 - a * c ∧ b^2 - a * c = c^2 - a * b

-- The main theorem statement
theorem expression_values (a b c : ℝ) (h : conditions a b c) :
  (∃ x : ℝ, x = (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b)) ∧ (x = 7 / 2 ∨ x = -7)) :=
by
  sorry

end expression_values_l1123_112316


namespace variance_of_data_set_is_4_l1123_112372

/-- The data set for which we want to calculate the variance --/
def data_set : List ℝ := [2, 4, 5, 6, 8]

/-- The mean of the data set --/
noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Calculation of the variance of a list given its mean
noncomputable def variance (l : List ℝ) (μ : ℝ) : ℝ :=
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem variance_of_data_set_is_4 :
  variance data_set (mean data_set) = 4 :=
by
  sorry

end variance_of_data_set_is_4_l1123_112372


namespace negation_of_quadratic_prop_l1123_112314

theorem negation_of_quadratic_prop :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 < 0 :=
by
  sorry

end negation_of_quadratic_prop_l1123_112314


namespace mark_has_seven_butterfingers_l1123_112350

/-
  Mark has 12 candy bars in total between Mars bars, Snickers, and Butterfingers.
  He has 3 Snickers and 2 Mars bars.
  Prove that he has 7 Butterfingers.
-/

noncomputable def total_candy_bars : Nat := 12
noncomputable def snickers : Nat := 3
noncomputable def mars_bars : Nat := 2
noncomputable def butterfingers : Nat := total_candy_bars - (snickers + mars_bars)

theorem mark_has_seven_butterfingers : butterfingers = 7 := by
  sorry

end mark_has_seven_butterfingers_l1123_112350


namespace chocolate_bars_per_box_is_25_l1123_112327

-- Define the conditions
def total_chocolate_bars : Nat := 400
def total_small_boxes : Nat := 16

-- Define the statement to be proved
def chocolate_bars_per_small_box : Nat := total_chocolate_bars / total_small_boxes

theorem chocolate_bars_per_box_is_25
  (h1 : total_chocolate_bars = 400)
  (h2 : total_small_boxes = 16) :
  chocolate_bars_per_small_box = 25 :=
by
  -- proof will go here
  sorry

end chocolate_bars_per_box_is_25_l1123_112327


namespace solve_equation_l1123_112379

theorem solve_equation (x : ℝ) :
  (3 / x - (1 / x * 6 / x) = -2.5) ↔ (x = (-3 + Real.sqrt 69) / 5 ∨ x = (-3 - Real.sqrt 69) / 5) :=
by {
  sorry
}

end solve_equation_l1123_112379


namespace evaluate_expression_l1123_112378

theorem evaluate_expression : 2 - 1 / (2 + 1 / (2 - 1 / 3)) = 21 / 13 := by
  sorry

end evaluate_expression_l1123_112378


namespace veranda_area_correct_l1123_112329

-- Definitions of the room dimensions and veranda width
def room_length : ℝ := 18
def room_width : ℝ := 12
def veranda_width : ℝ := 2

-- Definition of the total length including veranda
def total_length : ℝ := room_length + 2 * veranda_width

-- Definition of the total width including veranda
def total_width : ℝ := room_width + 2 * veranda_width

-- Definition of the area of the entire space (room plus veranda)
def area_entire_space : ℝ := total_length * total_width

-- Definition of the area of the room
def area_room : ℝ := room_length * room_width

-- Definition of the area of the veranda
def area_veranda : ℝ := area_entire_space - area_room

-- Theorem statement to prove the area of the veranda
theorem veranda_area_correct : area_veranda = 136 := 
by
  sorry

end veranda_area_correct_l1123_112329


namespace correct_equation_l1123_112374

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l1123_112374


namespace kids_played_on_Wednesday_l1123_112328

def played_on_Monday : ℕ := 17
def played_on_Tuesday : ℕ := 15
def total_kids : ℕ := 34

theorem kids_played_on_Wednesday :
  total_kids - (played_on_Monday + played_on_Tuesday) = 2 :=
by sorry

end kids_played_on_Wednesday_l1123_112328


namespace find_d_given_n_eq_cda_div_a_minus_d_l1123_112373

theorem find_d_given_n_eq_cda_div_a_minus_d (a c d n : ℝ) (h : n = c * d * a / (a - d)) :
  d = n * a / (c * d + n) := 
by
  sorry

end find_d_given_n_eq_cda_div_a_minus_d_l1123_112373


namespace alcohol_percentage_new_mixture_l1123_112338

namespace AlcoholMixtureProblem

def original_volume : ℝ := 3
def alcohol_percentage : ℝ := 0.33
def additional_water_volume : ℝ := 1
def new_volume : ℝ := original_volume + additional_water_volume
def alcohol_amount : ℝ := original_volume * alcohol_percentage

theorem alcohol_percentage_new_mixture : (alcohol_amount / new_volume) * 100 = 24.75 := by
  sorry

end AlcoholMixtureProblem

end alcohol_percentage_new_mixture_l1123_112338


namespace distance_between_lamps_l1123_112343

/-- 
A rectangular classroom measures 10 meters in length. Two lamps emitting conical light beams with a 90° opening angle 
are installed on the ceiling. The first lamp is located at the center of the ceiling and illuminates a circle on the 
floor with a diameter of 6 meters. The second lamp is adjusted such that the illuminated area along the length 
of the classroom spans a 10-meter section without reaching the opposite walls. Prove that the distance between the 
two lamps is 4 meters.
-/
theorem distance_between_lamps : 
  ∀ (length width height : ℝ) (center_illum_radius illum_length : ℝ) (d_center_to_lamp1 d_center_to_lamp2 dist_lamps : ℝ),
  length = 10 ∧ d_center_to_lamp1 = 3 ∧ d_center_to_lamp2 = 1 ∧ dist_lamps = 4 → d_center_to_lamp1 - d_center_to_lamp2 = dist_lamps :=
by
  intros length width height center_illum_radius illum_length d_center_to_lamp1 d_center_to_lamp2 dist_lamps conditions
  sorry

end distance_between_lamps_l1123_112343


namespace sufficient_but_not_necessary_l1123_112357

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x^2 + 2 * x > 0) ∧ ¬(x^2 + 2 * x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l1123_112357


namespace directrix_of_parabola_l1123_112345

theorem directrix_of_parabola (x y : ℝ) (h : y = x^2) : y = -1 / 4 :=
sorry

end directrix_of_parabola_l1123_112345


namespace calculate_moment_of_inertia_l1123_112361

noncomputable def moment_of_inertia (a ρ₀ k : ℝ) : ℝ :=
  8 * (a ^ (9/2)) * ((ρ₀ / 7) + (k * a / 9))

theorem calculate_moment_of_inertia (a ρ₀ k : ℝ) 
  (h₀ : 0 ≤ a) :
  moment_of_inertia a ρ₀ k = 8 * a ^ (9/2) * ((ρ₀ / 7) + (k * a / 9)) :=
sorry

end calculate_moment_of_inertia_l1123_112361


namespace range_alpha_minus_beta_over_2_l1123_112319

theorem range_alpha_minus_beta_over_2 (α β : ℝ) (h1 : -π / 2 ≤ α) (h2 : α < β) (h3 : β ≤ π / 2) :
  Set.Ico (-π / 2) 0 = {x : ℝ | ∃ α β : ℝ, -π / 2 ≤ α ∧ α < β ∧ β ≤ π / 2 ∧ x = (α - β) / 2} :=
by
  sorry

end range_alpha_minus_beta_over_2_l1123_112319


namespace evaluate_expression_l1123_112324

theorem evaluate_expression :
  let x := 1.93
  let y := 51.3
  let z := 0.47
  Float.round (x * (y + z)) = 100 := by
sorry

end evaluate_expression_l1123_112324


namespace math_problem_l1123_112387

-- Define the individual numbers
def a : Int := 153
def b : Int := 39
def c : Int := 27
def d : Int := 21

-- Define the entire expression and its expected result
theorem math_problem : (a + b + c + d) * 2 = 480 := by
  sorry

end math_problem_l1123_112387


namespace group_total_people_l1123_112393

theorem group_total_people (k : ℕ) (h1 : k = 7) (h2 : ((n - k) / n : ℝ) - (k / n : ℝ) = 0.30000000000000004) : n = 20 :=
  sorry

end group_total_people_l1123_112393


namespace parabola_vertex_and_point_l1123_112354

/-- The vertex form of the parabola is at (7, -6) and passes through the point (1,0).
    Verify that the equation parameters a, b, c satisfy a + b + c = -43 / 6. -/
theorem parabola_vertex_and_point (a b c : ℚ)
  (h_eq : ∀ y, (a * y^2 + b * y + c) = a * (y + 6)^2 + 7)
  (h_vertex : ∃ x y, x = a * y^2 + b * y + c ∧ y = -6 ∧ x = 7)
  (h_point : ∃ x y, x = a * y^2 + b * y + c ∧ x = 1 ∧ y = 0) :
  a + b + c = -43 / 6 :=
by
  sorry

end parabola_vertex_and_point_l1123_112354


namespace SumataFamilyTotalMiles_l1123_112356

def miles_per_day := 250
def days := 5

theorem SumataFamilyTotalMiles : miles_per_day * days = 1250 :=
by
  sorry

end SumataFamilyTotalMiles_l1123_112356


namespace cube_surface_area_l1123_112325

noncomputable def volume (x : ℝ) : ℝ := x ^ 3

noncomputable def surface_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem cube_surface_area (x : ℝ) :
  surface_area x = 6 * x ^ 2 :=
by sorry

end cube_surface_area_l1123_112325


namespace bank_account_balance_l1123_112360

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l1123_112360


namespace project_completion_time_l1123_112351

def process_duration (a b c d e f : Nat) : Nat :=
  let duration_c := max a b + c
  let duration_d := duration_c + d
  let duration_e := duration_c + e
  let duration_f := max duration_d duration_e + f
  duration_f

theorem project_completion_time :
  ∀ (a b c d e f : Nat), a = 2 → b = 3 → c = 2 → d = 5 → e = 4 → f = 1 →
  process_duration a b c d e f = 11 := by
  intros
  subst_vars
  sorry

end project_completion_time_l1123_112351


namespace how_many_more_rolls_needed_l1123_112375

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end how_many_more_rolls_needed_l1123_112375


namespace maximum_volume_regular_triangular_pyramid_l1123_112371

-- Given values
def R : ℝ := 1

-- Prove the maximum volume
theorem maximum_volume_regular_triangular_pyramid : 
  ∃ (V_max : ℝ), V_max = (8 * Real.sqrt 3) / 27 := 
by 
  sorry

end maximum_volume_regular_triangular_pyramid_l1123_112371


namespace cos_A_minus_B_minus_3pi_div_2_l1123_112312

theorem cos_A_minus_B_minus_3pi_div_2 (A B : ℝ)
  (h1 : Real.tan B = 2 * Real.tan A)
  (h2 : Real.cos A * Real.sin B = 4 / 5) :
  Real.cos (A - B - 3 * Real.pi / 2) = 2 / 5 := 
sorry

end cos_A_minus_B_minus_3pi_div_2_l1123_112312


namespace probability_one_out_of_three_l1123_112385

def probability_passing_exactly_one (p : ℚ) (n k : ℕ) :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_one_out_of_three :
  probability_passing_exactly_one (1/3) 3 1 = 4/9 :=
by sorry

end probability_one_out_of_three_l1123_112385


namespace original_number_is_106_25_l1123_112386

theorem original_number_is_106_25 (x : ℝ) (h : (x + 0.375 * x) - (x - 0.425 * x) = 85) : x = 106.25 := by
  sorry

end original_number_is_106_25_l1123_112386


namespace average_minutes_correct_l1123_112302

noncomputable def average_minutes_run_per_day : ℚ :=
  let f (fifth_graders : ℕ) : ℚ := (48 * (4 * fifth_graders) + 30 * (2 * fifth_graders) + 10 * fifth_graders) / (4 * fifth_graders + 2 * fifth_graders + fifth_graders)
  f 1

theorem average_minutes_correct :
  average_minutes_run_per_day = 88 / 7 :=
by
  sorry

end average_minutes_correct_l1123_112302


namespace find_f_of_7_over_2_l1123_112310

variable (f : ℝ → ℝ)

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f (x - 2)
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = 3^x

theorem find_f_of_7_over_2 : f (7 / 2) = -Real.sqrt 3 :=
by
  sorry

end find_f_of_7_over_2_l1123_112310


namespace area_of_parallelogram_l1123_112315

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 := by
  sorry

end area_of_parallelogram_l1123_112315


namespace historical_fiction_new_releases_fraction_l1123_112353

noncomputable def HF_fraction_total_inventory : ℝ := 0.4
noncomputable def Mystery_fraction_total_inventory : ℝ := 0.3
noncomputable def SF_fraction_total_inventory : ℝ := 0.2
noncomputable def Romance_fraction_total_inventory : ℝ := 0.1

noncomputable def HF_new_release_percentage : ℝ := 0.35
noncomputable def Mystery_new_release_percentage : ℝ := 0.60
noncomputable def SF_new_release_percentage : ℝ := 0.45
noncomputable def Romance_new_release_percentage : ℝ := 0.80

noncomputable def historical_fiction_new_releases : ℝ := HF_fraction_total_inventory * HF_new_release_percentage
noncomputable def mystery_new_releases : ℝ := Mystery_fraction_total_inventory * Mystery_new_release_percentage
noncomputable def sf_new_releases : ℝ := SF_fraction_total_inventory * SF_new_release_percentage
noncomputable def romance_new_releases : ℝ := Romance_fraction_total_inventory * Romance_new_release_percentage

noncomputable def total_new_releases : ℝ :=
  historical_fiction_new_releases + mystery_new_releases + sf_new_releases + romance_new_releases

theorem historical_fiction_new_releases_fraction :
  (historical_fiction_new_releases / total_new_releases) = (2 / 7) :=
by
  sorry

end historical_fiction_new_releases_fraction_l1123_112353


namespace functions_satisfying_equation_l1123_112323

theorem functions_satisfying_equation 
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, f x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, g x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, h x = a * x + b) :=
sorry

end functions_satisfying_equation_l1123_112323


namespace winner_percentage_l1123_112332

theorem winner_percentage (W L V : ℕ) 
    (hW : W = 868) 
    (hDiff : W - L = 336)
    (hV : V = W + L) : 
    (W * 100 / V) = 62 := 
by 
    sorry

end winner_percentage_l1123_112332


namespace mark_cans_correct_l1123_112342

variable (R : ℕ) -- Rachel's cans
variable (J : ℕ) -- Jaydon's cans
variable (M : ℕ) -- Mark's cans
variable (T : ℕ) -- Total cans 

-- Conditions
def jaydon_cans (R : ℕ) : ℕ := 2 * R + 5
def mark_cans (J : ℕ) : ℕ := 4 * J
def total_cans (R : ℕ) (J : ℕ) (M : ℕ) : ℕ := R + J + M

theorem mark_cans_correct (R : ℕ) (J : ℕ) 
  (h1 : J = jaydon_cans R) 
  (h2 : M = mark_cans J) 
  (h3 : total_cans R J M = 135) : 
  M = 100 := 
sorry

end mark_cans_correct_l1123_112342


namespace total_watermelon_weight_l1123_112320

theorem total_watermelon_weight :
  let w1 := 9.91
  let w2 := 4.112
  let w3 := 6.059
  w1 + w2 + w3 = 20.081 :=
by
  sorry

end total_watermelon_weight_l1123_112320


namespace same_yield_among_squares_l1123_112341

-- Define the conditions
def rectangular_schoolyard (length : ℝ) (width : ℝ) := length = 70 ∧ width = 35

def total_harvest (harvest : ℝ) := harvest = 1470 -- in kilograms (14.7 quintals)

def smaller_square (side : ℝ) := side = 0.7

-- Define the proof problem
theorem same_yield_among_squares :
  ∃ side : ℝ, smaller_square side ∧
  ∃ length width harvest : ℝ, rectangular_schoolyard length width ∧ total_harvest harvest →
  ∃ (yield1 yield2 : ℝ), yield1 = yield2 ∧ yield1 ≠ 0 ∧ yield2 ≠ 0 :=
by sorry

end same_yield_among_squares_l1123_112341


namespace min_f_value_l1123_112370

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2

theorem min_f_value (x y z : ℝ) (hxyz_pos : 0 < x ∧ 0 < y ∧ 0 < z) (hxyz : x * y * z = 1) :
  f x y z ≥ 18 :=
sorry

end min_f_value_l1123_112370


namespace coprime_ab_and_a_plus_b_l1123_112330

theorem coprime_ab_and_a_plus_b (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a * b) (a + b) = 1 := by
  sorry

end coprime_ab_and_a_plus_b_l1123_112330


namespace cost_of_coffee_A_per_kg_l1123_112392

theorem cost_of_coffee_A_per_kg (x : ℝ) :
  (240 * x + 240 * 12 = 480 * 11) → x = 10 :=
by
  intros h
  sorry

end cost_of_coffee_A_per_kg_l1123_112392


namespace problem_R_l1123_112395

noncomputable def R (g S h : ℝ) : ℝ := g * S + h

theorem problem_R {g h : ℝ} (h_h : h = 6 - 4 * g) :
  R g 14 h = 56 :=
by
  sorry

end problem_R_l1123_112395


namespace Dave_ticket_count_l1123_112366

variable (T C total : ℕ)

theorem Dave_ticket_count
  (hT1 : T = 12)
  (hC1 : C = 7)
  (hT2 : T = C + 5) :
  total = T + C → total = 19 := by
  sorry

end Dave_ticket_count_l1123_112366


namespace simplify_sqrt_is_cos_20_l1123_112317

noncomputable def simplify_sqrt : ℝ :=
  let θ : ℝ := 160 * Real.pi / 180
  Real.sqrt (1 - Real.sin θ ^ 2)

theorem simplify_sqrt_is_cos_20 : simplify_sqrt = Real.cos (20 * Real.pi / 180) :=
  sorry

end simplify_sqrt_is_cos_20_l1123_112317


namespace number_of_nurses_l1123_112339

theorem number_of_nurses (total : ℕ) (ratio_d_to_n : ℕ → ℕ) (h1 : total = 250) (h2 : ratio_d_to_n 2 = 3) : ∃ n : ℕ, n = 150 := 
by
  sorry

end number_of_nurses_l1123_112339


namespace determinant_eval_l1123_112305

open Matrix

noncomputable def matrix_example (α γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, 3 * Real.sin γ],
    ![2 * Real.cos α, -Real.sin γ, 0]]

theorem determinant_eval (α γ : ℝ) :
  det (matrix_example α γ) = 10 * Real.sin α * Real.sin γ * Real.cos α :=
sorry

end determinant_eval_l1123_112305


namespace balance_weights_l1123_112394

def pair_sum {α : Type*} (l : List α) [Add α] : List (α × α) :=
  l.zip l.tail

theorem balance_weights (w : Fin 100 → ℝ) (h : ∀ i j, |w i - w j| ≤ 20) :
  ∃ (l r : Finset (Fin 100)), l.card = 50 ∧ r.card = 50 ∧
  |(l.sum w - r.sum w)| ≤ 20 :=
sorry

end balance_weights_l1123_112394


namespace exponent_subtraction_l1123_112300

variable {a : ℝ} {m n : ℕ}

theorem exponent_subtraction (hm : a ^ m = 12) (hn : a ^ n = 3) : a ^ (m - n) = 4 :=
by
  sorry

end exponent_subtraction_l1123_112300


namespace gcd_840_1764_gcd_98_63_l1123_112308

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by sorry

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by sorry

end gcd_840_1764_gcd_98_63_l1123_112308


namespace least_value_MX_l1123_112388

-- Definitions of points and lines
variables (A B C D M P X : ℝ × ℝ)
variables (y : ℝ)

-- Hypotheses based on the conditions
variables (h1 : A = (0, 0))
variables (h2 : B = (33, 0))
variables (h3 : C = (33, 56))
variables (h4 : D = (0, 56))
variables (h5 : M = (33 / 2, 0)) -- M is midpoint of AB
variables (h6 : P = (33, y)) -- P is on BC
variables (hy_range : 0 ≤ y ∧ y ≤ 56) -- y is within the bounds of BC

-- Additional derived hypotheses needed for the proof
variables (h7 : ∃ x, X = (x, sqrt (816.75))) -- X is intersection point on DA

-- The theorem statement
theorem least_value_MX : ∃ y, 0 ≤ y ∧ y ≤ 56 ∧ MX = 33 :=
by
  use 28
  sorry

end least_value_MX_l1123_112388


namespace min_value_expr_sum_of_squares_inequality_l1123_112364

-- Given conditions
variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Problem (1): Prove minimum value of (2 / a + 8 / b) is 9
theorem min_value_expr : ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ((2 / a) + (8 / b) = 9) := sorry

-- Problem (2): Prove a^2 + b^2 ≥ 2
theorem sum_of_squares_inequality : a^2 + b^2 ≥ 2 :=
by { sorry }

end min_value_expr_sum_of_squares_inequality_l1123_112364


namespace find_a_plus_b_l1123_112376

-- Conditions for the lines
def line_l0 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l1 (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def line_l2 (b : ℝ) (x y : ℝ) : Prop := x + b * y + 3 = 0

-- Perpendicularity condition for l1 to l0
def perpendicular (a : ℝ) : Prop := 1 * a + (-1) * (-2) = 0

-- Parallel condition for l2 to l0
def parallel (b : ℝ) : Prop := 1 * b = (-1) * 1

-- Prove the value of a + b given the conditions
theorem find_a_plus_b (a b : ℝ) 
  (h1 : perpendicular a)
  (h2 : parallel b) : a + b = -3 :=
sorry

end find_a_plus_b_l1123_112376
