import Mathlib

namespace man_l1067_106787

theorem man's_speed_against_the_current (vm vc : ℝ) 
(h1: vm + vc = 15) 
(h2: vm - vc = 10) : 
vm - vc = 10 := 
by 
  exact h2

end man_l1067_106787


namespace andy_questions_wrong_l1067_106783

variables (a b c d : ℕ)

-- Given conditions
def condition1 : Prop := a + b = c + d
def condition2 : Prop := a + d = b + c + 6
def condition3 : Prop := c = 7

-- The theorem to prove
theorem andy_questions_wrong (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 c) : a = 10 :=
by
  sorry

end andy_questions_wrong_l1067_106783


namespace avg_age_of_team_is_23_l1067_106722

-- Conditions
def captain_age := 24
def wicket_keeper_age := captain_age + 7

def remaining_players_avg_age (team_avg_age : ℝ) := team_avg_age - 1
def total_team_age (team_avg_age : ℝ) := 11 * team_avg_age
def total_remaining_players_age (team_avg_age : ℝ) := 9 * remaining_players_avg_age team_avg_age

-- Proof statement
theorem avg_age_of_team_is_23 (team_avg_age : ℝ) :
  total_team_age team_avg_age = captain_age + wicket_keeper_age + total_remaining_players_age team_avg_age → 
  team_avg_age = 23 :=
by
  sorry

end avg_age_of_team_is_23_l1067_106722


namespace range_of_x_l1067_106719

noncomputable def a (x : ℝ) : ℝ := x
def b : ℝ := 2
def B : ℝ := 60

-- State the problem: Prove the range of x given the conditions
theorem range_of_x (x : ℝ) (A : ℝ) (C : ℝ) (h1 : a x = b / (Real.sin (B * Real.pi / 180)) * (Real.sin (A * Real.pi / 180)))
  (h2 : A + C = 180 - 60) (two_solutions : (60 < A ∧ A < 120)) :
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 :=
sorry

end range_of_x_l1067_106719


namespace proof_problem_l1067_106736

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}
def complement (s : Set ℕ) : Set ℕ := {x | x ∉ s}

theorem proof_problem : ((complement A ∪ A) ∪ B) = U :=
by sorry

end proof_problem_l1067_106736


namespace LineDoesNotIntersectParabola_sum_r_s_l1067_106730

noncomputable def r : ℝ := -0.6
noncomputable def s : ℝ := 40.6
def Q : ℝ × ℝ := (10, -6)
def line_through_Q_with_slope (m : ℝ) (p : ℝ × ℝ) : ℝ := m * p.1 - 10 * m - 6
def parabola (x : ℝ) : ℝ := 2 * x^2

theorem LineDoesNotIntersectParabola (m : ℝ) :
  r < m ∧ m < s ↔ (m^2 - 4 * 2 * (10 * m + 6) < 0) :=
by sorry

theorem sum_r_s : r + s = 40 :=
by sorry

end LineDoesNotIntersectParabola_sum_r_s_l1067_106730


namespace find_m_value_l1067_106752

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

end find_m_value_l1067_106752


namespace cost_of_gravelling_roads_l1067_106753

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

end cost_of_gravelling_roads_l1067_106753


namespace b_not_six_iff_neg_two_not_in_range_l1067_106788

def g (x b : ℝ) := x^3 + x^2 + b*x + 2

theorem b_not_six_iff_neg_two_not_in_range (b : ℝ) : 
  (∀ x : ℝ, g x b ≠ -2) ↔ b ≠ 6 :=
by
  sorry

end b_not_six_iff_neg_two_not_in_range_l1067_106788


namespace eccentricity_of_hyperbola_l1067_106771

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

end eccentricity_of_hyperbola_l1067_106771


namespace octadecagon_identity_l1067_106712

theorem octadecagon_identity (a r : ℝ) (h : a = 2 * r * Real.sin (π / 18)) :
  a^3 + r^3 = 3 * r^2 * a :=
sorry

end octadecagon_identity_l1067_106712


namespace inequality_proof_l1067_106765

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2) / c + (c^2 - b^2) / a + (a^2 - c^2) / b ≥ 3 * a - 4 * b + c :=
by
  sorry

end inequality_proof_l1067_106765


namespace lasagna_package_weight_l1067_106775

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

end lasagna_package_weight_l1067_106775


namespace persons_in_office_l1067_106720

theorem persons_in_office
  (P : ℕ)
  (h1 : (P - (1/7 : ℚ)*P) = (6/7 : ℚ)*P)
  (h2 : (16.66666666666667/100 : ℚ) = 1/6) :
  P = 35 :=
sorry

end persons_in_office_l1067_106720


namespace athlete_distance_proof_l1067_106793

-- Definition of conditions as constants
def time_seconds : ℕ := 20
def speed_kmh : ℕ := 36

-- Convert speed from km/h to m/s
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Proof statement that the distance is 200 meters
theorem athlete_distance_proof : speed_mps * time_seconds = 200 :=
by sorry

end athlete_distance_proof_l1067_106793


namespace value_of_x_l1067_106744

theorem value_of_x (a b x : ℝ) (h : x^2 + 4 * b^2 = (2 * a - x)^2) : 
  x = (a^2 - b^2) / a :=
by
  sorry

end value_of_x_l1067_106744


namespace geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l1067_106757

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

end geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l1067_106757


namespace blue_lipstick_students_l1067_106784

def total_students : ℕ := 200
def students_with_lipstick : ℕ := total_students / 2
def students_with_red_lipstick : ℕ := students_with_lipstick / 4
def students_with_blue_lipstick : ℕ := students_with_red_lipstick / 5

theorem blue_lipstick_students : students_with_blue_lipstick = 5 :=
by
  sorry

end blue_lipstick_students_l1067_106784


namespace exchange_positions_l1067_106779

theorem exchange_positions : ∀ (people : ℕ), people = 8 → (∃ (ways : ℕ), ways = 336) :=
by sorry

end exchange_positions_l1067_106779


namespace math_problem_l1067_106715

open Real

variable (x : ℝ)
variable (h : x + 1 / x = sqrt 3)

theorem math_problem : x^7 - 3 * x^5 + x^2 = -5 * x + 4 * sqrt 3 :=
by sorry

end math_problem_l1067_106715


namespace find_first_term_geometric_sequence_l1067_106767

theorem find_first_term_geometric_sequence 
  (a b c : ℚ) 
  (h₁ : b = a * 4) 
  (h₂ : 36 = a * 4^2) 
  (h₃ : c = a * 4^3) 
  (h₄ : 144 = a * 4^4) : 
  a = 9 / 4 :=
sorry

end find_first_term_geometric_sequence_l1067_106767


namespace oil_bill_january_l1067_106741

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

end oil_bill_january_l1067_106741


namespace proof_subset_l1067_106717

def set_A := {x : ℝ | x ≥ 0}

theorem proof_subset (B : Set ℝ) (h : set_A ∪ B = B) : set_A ⊆ B := 
by
  sorry

end proof_subset_l1067_106717


namespace unique_students_total_l1067_106708

variables (euclid_students raman_students pythagoras_students overlap_3 : ℕ)

def total_students (E R P O : ℕ) : ℕ := E + R + P - O

theorem unique_students_total (hE : euclid_students = 12) 
                              (hR : raman_students = 10) 
                              (hP : pythagoras_students = 15) 
                              (hO : overlap_3 = 3) : 
    total_students euclid_students raman_students pythagoras_students overlap_3 = 34 :=
by
    sorry

end unique_students_total_l1067_106708


namespace find_water_needed_l1067_106760

def apple_juice := 4
def honey (A : ℕ) := 3 * A
def water (H : ℕ) := 3 * H

theorem find_water_needed : water (honey apple_juice) = 36 :=
  sorry

end find_water_needed_l1067_106760


namespace cats_weigh_more_than_puppies_l1067_106737

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

end cats_weigh_more_than_puppies_l1067_106737


namespace percentage_decrease_correct_l1067_106748

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

end percentage_decrease_correct_l1067_106748


namespace totalMountainNumbers_l1067_106768

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

end totalMountainNumbers_l1067_106768


namespace average_rounds_rounded_eq_4_l1067_106706

def rounds_distribution : List (Nat × Nat) := [(1, 4), (2, 3), (4, 4), (5, 2), (6, 6)]

def total_rounds : Nat := rounds_distribution.foldl (λ acc (rounds, golfers) => acc + rounds * golfers) 0

def total_golfers : Nat := rounds_distribution.foldl (λ acc (_, golfers) => acc + golfers) 0

def average_rounds : Float := total_rounds.toFloat / total_golfers.toFloat

theorem average_rounds_rounded_eq_4 : Float.round average_rounds = 4 := by
  sorry

end average_rounds_rounded_eq_4_l1067_106706


namespace product_divisible_by_sum_l1067_106769

theorem product_divisible_by_sum (m n : ℕ) (h : ∃ k : ℕ, m * n = k * (m + n)) : m + n ≤ Nat.gcd m n * Nat.gcd m n := by
  sorry

end product_divisible_by_sum_l1067_106769


namespace maia_daily_client_requests_l1067_106731

theorem maia_daily_client_requests (daily_requests : ℕ) (remaining_requests : ℕ) (days : ℕ) 
  (received_requests : ℕ) (total_requests : ℕ) (worked_requests : ℕ) :
  (daily_requests = 6) →
  (remaining_requests = 10) →
  (days = 5) →
  (received_requests = daily_requests * days) →
  (total_requests = received_requests - remaining_requests) →
  (worked_requests = total_requests / days) →
  worked_requests = 4 :=
by
  sorry

end maia_daily_client_requests_l1067_106731


namespace find_ordered_triple_l1067_106701

theorem find_ordered_triple (a b c : ℝ) (h1 : a > 2) (h2 : b > 2) (h3 : c > 2)
  (h4 : (a + 1)^2 / (b + c - 1) + (b + 3)^2 / (c + a - 3) + (c + 5)^2 / (a + b - 5) = 27) :
  (a, b, c) = (9, 7, 2) :=
by sorry

end find_ordered_triple_l1067_106701


namespace clean_time_per_room_l1067_106749

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

end clean_time_per_room_l1067_106749


namespace sequence_formula_correct_l1067_106728

noncomputable def S (n : ℕ) : ℕ := 2^n - 3

def a (n : ℕ) : ℤ :=
  if n = 1 then -1
  else 2^(n-1)

theorem sequence_formula_correct (n : ℕ) :
  a n = (if n = 1 then -1 else 2^(n-1)) :=
by
  sorry

end sequence_formula_correct_l1067_106728


namespace find_larger_integer_l1067_106758

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l1067_106758


namespace zoo_pandas_l1067_106718

-- Defining the conditions
variable (total_couples : ℕ)
variable (pregnant_couples : ℕ)
variable (baby_pandas : ℕ)
variable (total_pandas : ℕ)

-- Given conditions
def paired_mates : Prop := ∃ c : ℕ, c = total_couples

def pregnant_condition : Prop := pregnant_couples = (total_couples * 25) / 100

def babies_condition : Prop := baby_pandas = 2

def total_condition : Prop := total_pandas = total_couples * 2 + baby_pandas

-- The theorem to be proven
theorem zoo_pandas (h1 : paired_mates total_couples)
                   (h2 : pregnant_condition total_couples pregnant_couples)
                   (h3 : babies_condition baby_pandas)
                   (h4 : pregnant_couples = 2) :
                   total_condition total_couples baby_pandas total_pandas :=
by sorry

end zoo_pandas_l1067_106718


namespace solve_inequality_solve_system_of_inequalities_l1067_106709

-- Inequality proof problem
theorem solve_inequality (x : ℝ) (h : (2*x - 3)/3 > (3*x + 1)/6 - 1) : x > 1 := by
  sorry

-- System of inequalities proof problem
theorem solve_system_of_inequalities (x : ℝ) (h1 : x ≤ 3*x - 6) (h2 : 3*x + 1 > 2*(x - 1)) : x ≥ 3 := by
  sorry

end solve_inequality_solve_system_of_inequalities_l1067_106709


namespace negation_of_prop_l1067_106762

-- Define the original proposition
def prop (x : ℝ) : Prop := x^2 - x + 2 ≥ 0

-- State the negation of the original proposition
theorem negation_of_prop : (¬ ∀ x : ℝ, prop x) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := 
by
  sorry

end negation_of_prop_l1067_106762


namespace avg_of_eleven_numbers_l1067_106776

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

end avg_of_eleven_numbers_l1067_106776


namespace jean_more_trips_than_bill_l1067_106777

variable (b j : ℕ)

theorem jean_more_trips_than_bill
  (h1 : b + j = 40)
  (h2 : j = 23) :
  j - b = 6 := by
  sorry

end jean_more_trips_than_bill_l1067_106777


namespace program_exists_l1067_106770
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

end program_exists_l1067_106770


namespace divisible_by_primes_l1067_106727

theorem divisible_by_primes (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (100100 * x + 10010 * y + 1001 * z) % 7 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 11 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 13 = 0 := 
by
  sorry

end divisible_by_primes_l1067_106727


namespace greatest_matching_pairs_left_l1067_106710

-- Define the initial number of pairs and lost individual shoes
def initial_pairs : ℕ := 26
def lost_ind_shoes : ℕ := 9

-- The statement to be proved
theorem greatest_matching_pairs_left : 
  (initial_pairs * 2 - lost_ind_shoes) / 2 + (initial_pairs - (initial_pairs * 2 - lost_ind_shoes) / 2) / 1 = 17 := 
by 
  sorry

end greatest_matching_pairs_left_l1067_106710


namespace playground_area_l1067_106786

theorem playground_area (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 100)
  (h_length : l = 3 * w) : l * w = 468.75 :=
by
  sorry

end playground_area_l1067_106786


namespace odd_function_f_neg_9_l1067_106729

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then x^(1/2) 
else -((-x)^(1/2))

theorem odd_function_f_neg_9 : f (-9) = -3 := by
  sorry

end odd_function_f_neg_9_l1067_106729


namespace find_c_l1067_106733

variable (x y c : ℝ)

def condition1 : Prop := 2 * x + 5 * y = 3
def condition2 : Prop := c = Real.sqrt (4^(x + 1/2) * 32^y)

theorem find_c (h1 : condition1 x y) (h2 : condition2 x y c) : c = 4 := by
  sorry

end find_c_l1067_106733


namespace find_third_circle_radius_l1067_106763

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

end find_third_circle_radius_l1067_106763


namespace Darius_scored_10_points_l1067_106734

theorem Darius_scored_10_points
  (D Marius Matt : ℕ)
  (h1 : Marius = D + 3)
  (h2 : Matt = D + 5)
  (h3 : D + Marius + Matt = 38) : 
  D = 10 :=
by
  sorry

end Darius_scored_10_points_l1067_106734


namespace find_a_to_satisfy_divisibility_l1067_106739

theorem find_a_to_satisfy_divisibility (a : ℕ) (h₀ : 0 ≤ a) (h₁ : a < 11) (h₂ : (2 * 10^10 + a) % 11 = 0) : a = 9 :=
sorry

end find_a_to_satisfy_divisibility_l1067_106739


namespace difference_received_from_parents_l1067_106797

-- Define conditions
def amount_from_mom := 8
def amount_from_dad := 5

-- Question: Prove the difference between amount_from_mom and amount_from_dad is 3
theorem difference_received_from_parents : (amount_from_mom - amount_from_dad) = 3 :=
by
  sorry

end difference_received_from_parents_l1067_106797


namespace hyperbola_foci_difference_l1067_106746

noncomputable def hyperbola_foci_distance (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (a : ℝ) : ℝ :=
  |dist P F₁ - dist P F₂|

theorem hyperbola_foci_difference (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : 
  (P.1 ^ 2 - P.2 ^ 2 = 4) ∧ (P.1 < 0) → (hyperbola_foci_distance P F₁ F₂ 2 = -4) :=
by
  intros h
  sorry

end hyperbola_foci_difference_l1067_106746


namespace matrix_power_four_l1067_106785

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

end matrix_power_four_l1067_106785


namespace part1_part2_l1067_106714

noncomputable def f (m x : ℝ) : ℝ := m - |x - 1| - |x + 1|

theorem part1 (x : ℝ) : -3 / 2 < x ∧ x < 3 / 2 ↔ f 5 x > 2 := by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ y : ℝ, x^2 + 2 * x + 3 = f m y) ↔ 4 ≤ m := by
  sorry

end part1_part2_l1067_106714


namespace total_kids_at_camp_l1067_106703

-- Definition of the conditions
def kids_from_lawrence_camp : ℕ := 34044
def kids_from_outside_camp : ℕ := 424944

-- The proof statement
theorem total_kids_at_camp : kids_from_lawrence_camp + kids_from_outside_camp = 459988 := by
  sorry

end total_kids_at_camp_l1067_106703


namespace actual_cost_of_article_l1067_106738

-- Define the basic conditions of the problem
variable (x : ℝ)
variable (h : x - 0.24 * x = 1064)

-- The theorem we need to prove
theorem actual_cost_of_article : x = 1400 :=
by
  -- since we are not proving anything here, we skip the proof
  sorry

end actual_cost_of_article_l1067_106738


namespace number_of_lilies_l1067_106750

theorem number_of_lilies (L : ℕ) 
  (h1 : ∀ n:ℕ, n * 6 = 6 * n)
  (h2 : ∀ n:ℕ, n * 3 = 3 * n) 
  (h3 : 5 * 3 = 15)
  (h4 : 6 * L + 15 = 63) : 
  L = 8 := 
by
  -- Proof omitted 
  sorry

end number_of_lilies_l1067_106750


namespace find_value_of_fraction_l1067_106778

theorem find_value_of_fraction (x y z : ℝ)
  (h1 : 3 * x - 4 * y - z = 0)
  (h2 : x + 4 * y - 15 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3 * x * y - y * z) / (y^2 + z^2) = 2.4 :=
by
  sorry

end find_value_of_fraction_l1067_106778


namespace find_parking_cost_l1067_106754

theorem find_parking_cost :
  ∃ (C : ℝ), (C + 7 * 1.75) / 9 = 2.4722222222222223 ∧ C = 10 :=
sorry

end find_parking_cost_l1067_106754


namespace flower_problem_solution_l1067_106725

/-
Given the problem conditions:
1. There are 88 flowers.
2. Each flower was visited by at least one bee.
3. Each bee visited exactly 54 flowers.

Prove that bitter flowers exceed sweet flowers by 14.
-/

noncomputable def flower_problem : Prop :=
  ∃ (s g : ℕ), 
    -- Condition: The total number of flowers
    s + g + (88 - s - g) = 88 ∧ 
    -- Condition: Total number of visits by bees
    3 * 54 = 162 ∧ 
    -- Proof goal: Bitter flowers exceed sweet flowers by 14
    g - s = 14

theorem flower_problem_solution : flower_problem :=
by
  sorry

end flower_problem_solution_l1067_106725


namespace expand_polynomial_l1067_106780

theorem expand_polynomial (z : ℂ) :
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 :=
by sorry

end expand_polynomial_l1067_106780


namespace angle_division_quadrant_l1067_106798

variable (k : ℤ)
variable (α : ℝ)
variable (h : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi)

theorem angle_division_quadrant 
  (hα_sec_quadrant : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi) : 
  (∃ m : ℤ, (m = 0 ∧ Real.pi / 4 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
             (m = 1 ∧ Real.pi * (2 * m + 1) / 4 < α / 2 ∧ α / 2 < Real.pi * (2 * m + 1) / 2)) :=
sorry

end angle_division_quadrant_l1067_106798


namespace tangent_lines_passing_through_point_l1067_106743

theorem tangent_lines_passing_through_point :
  ∀ (x0 y0 : ℝ) (p : ℝ × ℝ), 
  (p = (1, 1)) ∧ (y0 = x0 ^ 3) → 
  (y0 - 1 = 3 * x0 ^ 2 * (1 - x0)) → 
  (x0 = 1 ∨ x0 = -1/2) → 
  ((y - (3 * 1 - 2)) * (y - (3/4 * x0 + 1/4))) = 0 :=
sorry

end tangent_lines_passing_through_point_l1067_106743


namespace two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l1067_106764

theorem two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n
  (n : ℕ) (h : 2 < n) : (2 * n - 1) ^ n + (2 * n) ^ n < (2 * n + 1) ^ n :=
sorry

end two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l1067_106764


namespace average_speed_of_train_b_l1067_106781

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

end average_speed_of_train_b_l1067_106781


namespace rope_length_equals_120_l1067_106790

theorem rope_length_equals_120 (x : ℝ) (l : ℝ)
  (h1 : x + 20 = 3 * x) 
  (h2 : l = 4 * (2 * x)) : 
  l = 120 :=
by
  -- Proof will be provided here
  sorry

end rope_length_equals_120_l1067_106790


namespace district_B_high_schools_l1067_106772

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

end district_B_high_schools_l1067_106772


namespace tracy_michelle_distance_ratio_l1067_106716

theorem tracy_michelle_distance_ratio :
  ∀ (T M K : ℕ), 
  (M = 294) → 
  (M = 3 * K) → 
  (T + M + K = 1000) →
  ∃ x : ℕ, (T = x * M + 20) ∧ x = 2 :=
by
  intro T M K
  intro hM hMK hDistance
  use 2
  sorry

end tracy_michelle_distance_ratio_l1067_106716


namespace sum_of_intercepts_of_line_l1067_106747

theorem sum_of_intercepts_of_line (x y : ℝ) (hx : 2 * x - 3 * y + 6 = 0) :
  2 + (-3) = -1 :=
sorry

end sum_of_intercepts_of_line_l1067_106747


namespace triangle_area_formed_by_lines_l1067_106792

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

end triangle_area_formed_by_lines_l1067_106792


namespace inequality_solution_set_l1067_106782

theorem inequality_solution_set {x : ℝ} : 2 * x^2 - x - 1 > 0 ↔ (x < -1 / 2 ∨ x > 1) := 
sorry

end inequality_solution_set_l1067_106782


namespace racing_championship_guarantee_l1067_106724

/-- 
In a racing championship consisting of five races, the points awarded are as follows: 
6 points for first place, 4 points for second place, and 2 points for third place, with no ties possible. 
What is the smallest number of points a racer must accumulate in these five races to be guaranteed of having more points than any other racer? 
-/
theorem racing_championship_guarantee :
  ∀ (points_1st : ℕ) (points_2nd : ℕ) (points_3rd : ℕ) (races : ℕ),
  points_1st = 6 → points_2nd = 4 → points_3rd = 2 → 
  races = 5 →
  (∃ min_points : ℕ, min_points = 26 ∧ 
    ∀ (possible_points : ℕ), possible_points ≠ min_points → 
    (possible_points < min_points)) :=
by
  sorry

end racing_championship_guarantee_l1067_106724


namespace total_lives_l1067_106726

-- Definitions of given conditions
def original_friends : Nat := 2
def lives_per_player : Nat := 6
def additional_players : Nat := 2

-- Proof statement to show the total number of lives
theorem total_lives :
  (original_friends * lives_per_player) + (additional_players * lives_per_player) = 24 := by
  sorry

end total_lives_l1067_106726


namespace board_cut_ratio_l1067_106713

theorem board_cut_ratio (L S : ℝ) (h1 : S + L = 20) (h2 : S = L + 4) (h3 : S = 8.0) : S / L = 1 := by
  sorry

end board_cut_ratio_l1067_106713


namespace factorization_of_x_squared_minus_nine_l1067_106702

theorem factorization_of_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by
  sorry

end factorization_of_x_squared_minus_nine_l1067_106702


namespace find_a_l1067_106711

theorem find_a (a : ℝ) (A B : ℝ × ℝ × ℝ) (hA : A = (-1, 1, -a)) (hB : B = (-a, 3, -1)) (hAB : dist A B = 2) : a = -1 := by
  sorry

end find_a_l1067_106711


namespace macy_miles_left_to_run_l1067_106707

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end macy_miles_left_to_run_l1067_106707


namespace perimeter_of_regular_polygon_l1067_106759

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l1067_106759


namespace range_of_b_l1067_106794

noncomputable def f (x a b : ℝ) := (x - a)^2 * (x + b) * Real.exp x

theorem range_of_b (a b : ℝ) (h_max : ∃ δ > 0, ∀ x, |x - a| < δ → f x a b ≤ f a a b) : b < -a := sorry

end range_of_b_l1067_106794


namespace avg_age_grandparents_is_64_l1067_106766

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

end avg_age_grandparents_is_64_l1067_106766


namespace regular_polygon_sides_l1067_106735

theorem regular_polygon_sides (n : ℕ) (h₁ : n ≥ 3) (h₂ : 120 = 180 * (n - 2) / n) : n = 6 :=
by
  sorry

end regular_polygon_sides_l1067_106735


namespace price_of_adult_ticket_l1067_106732

/--
Given:
1. The price of a child's ticket is half the price of an adult's ticket.
2. Janet buys tickets for 10 people, 4 of whom are children.
3. Janet buys a soda for $5.
4. With the soda, Janet gets a 20% discount on the total admission price.
5. Janet paid $197 in total for everything.

Prove that the price of an adult admission ticket is $30.
-/
theorem price_of_adult_ticket : 
  ∃ (A : ℝ), 
  (∀ (childPrice adultPrice total : ℝ),
    adultPrice = A →
    childPrice = A / 2 →
    total = adultPrice * 6 + childPrice * 4 →
    totalPriceWithDiscount = 192 →
    total / 0.8 = total + 5 →
    A = 30) :=
sorry

end price_of_adult_ticket_l1067_106732


namespace find_f_zero_l1067_106740

theorem find_f_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y - x * y) 
  (h1 : f 1 = 1) : 
  f 0 = 0 := 
sorry

end find_f_zero_l1067_106740


namespace value_of_f_is_29_l1067_106774

noncomputable def f (x : ℕ) : ℕ := 3 * x - 4
noncomputable def g (x : ℕ) : ℕ := x^2 + 1

theorem value_of_f_is_29 :
  f (1 + g 3) = 29 := by
  sorry

end value_of_f_is_29_l1067_106774


namespace total_plates_l1067_106700

-- Define the initial conditions
def flower_plates_initial : ℕ := 4
def checked_plates : ℕ := 8
def polka_dotted_plates := 2 * checked_plates
def flower_plates_remaining := flower_plates_initial - 1

-- Prove the total number of plates Jack has left
theorem total_plates : flower_plates_remaining + polka_dotted_plates + checked_plates = 27 :=
by
  sorry

end total_plates_l1067_106700


namespace exists_n_of_form_2k_l1067_106789

theorem exists_n_of_form_2k (n : ℕ) (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_recip : 1/x + 1/y + 1/z = 1/(n : ℤ)) : ∃ k : ℕ, n = 2 * k :=
sorry

end exists_n_of_form_2k_l1067_106789


namespace amount_paid_for_grapes_l1067_106756

-- Definitions based on the conditions
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := 2.23

-- The statement to be proved
theorem amount_paid_for_grapes : total_spent + refund_for_cherries = 12.08 := 
by 
  -- Here the specific mathematical proof would go, but is replaced by sorry as instructed
  sorry

end amount_paid_for_grapes_l1067_106756


namespace problem_solution_l1067_106721

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f x + 2 * Real.cos x ^ 2

theorem problem_solution :
  (∀ x, (∃ ω > 0, ∃ φ, |φ| < Real.pi / 2 ∧ Real.sin (ω * x - φ) = 0 ∧ 2 * ω = Real.pi)) →
  (∀ x, f x = Real.sin (2 * x - Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (g x ≤ 2 ∧ g x ≥ 1 / 2)) :=
by
  sorry

end problem_solution_l1067_106721


namespace probability_heads_exactly_2_times_three_tosses_uniform_coin_l1067_106773

noncomputable def probability_heads_exactly_2_times (n k : ℕ) (p : ℚ) : ℚ :=
(n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_heads_exactly_2_times_three_tosses_uniform_coin :
  probability_heads_exactly_2_times 3 2 (1/2) = 3 / 8 :=
by
  sorry

end probability_heads_exactly_2_times_three_tosses_uniform_coin_l1067_106773


namespace friend_charge_per_animal_l1067_106795

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

end friend_charge_per_animal_l1067_106795


namespace parabola_directrix_l1067_106755

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

end parabola_directrix_l1067_106755


namespace bobby_candy_total_l1067_106705

-- Definitions for the conditions
def initial_candy : Nat := 20
def first_candy_eaten : Nat := 34
def second_candy_eaten : Nat := 18

-- Theorem to prove the total pieces of candy Bobby ate
theorem bobby_candy_total : first_candy_eaten + second_candy_eaten = 52 := by
  sorry

end bobby_candy_total_l1067_106705


namespace general_formula_l1067_106704

-- Define the sequence term a_n
def sequence_term (n : ℕ) : ℚ :=
  if h : n = 0 then 1
  else (2 * n - 1 : ℚ) / (n * n)

-- State the theorem for the general formula of the nth term
theorem general_formula (n : ℕ) (hn : n ≠ 0) : 
  sequence_term n = (2 * n - 1 : ℚ) / (n * n) :=
by sorry

end general_formula_l1067_106704


namespace intersection_of_M_and_N_l1067_106742

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_of_M_and_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l1067_106742


namespace point_coordinates_l1067_106751

noncomputable def parametric_curve (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) := (3 * Real.cos θ, 4 * Real.sin θ)

theorem point_coordinates (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) : 
  (Real.arcsin (4 * (Real.tan θ)) = π/4) → (3 * Real.cos θ, 4 * Real.sin θ) = (12 / 5, 12 / 5) :=
by
  sorry

end point_coordinates_l1067_106751


namespace sales_second_month_l1067_106745

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

end sales_second_month_l1067_106745


namespace fred_likes_12_pairs_of_digits_l1067_106791

theorem fred_likes_12_pairs_of_digits :
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs ↔ ∃ (n : ℕ), n < 100 ∧ n % 8 = 0 ∧ n = 10 * a + b) ∧
    pairs.card = 12) :=
by
  sorry

end fred_likes_12_pairs_of_digits_l1067_106791


namespace calculate_f_of_g_l1067_106723

def g (x : ℝ) := 4 * x + 6
def f (x : ℝ) := 6 * x - 10

theorem calculate_f_of_g :
  f (g 10) = 266 := by
  sorry

end calculate_f_of_g_l1067_106723


namespace no_polyhedron_without_triangles_and_three_valent_vertices_l1067_106796

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

end no_polyhedron_without_triangles_and_three_valent_vertices_l1067_106796


namespace slope_of_line_l1067_106761

/-- 
Given points M(1, 2) and N(3, 4), prove that the slope of the line passing through these points is 1.
-/
theorem slope_of_line (x1 y1 x2 y2 : ℝ) (hM : x1 = 1 ∧ y1 = 2) (hN : x2 = 3 ∧ y2 = 4) : 
  (y2 - y1) / (x2 - x1) = 1 :=
by
  -- The proof is omitted here because only the statement is required.
  sorry

end slope_of_line_l1067_106761


namespace nonagon_perimeter_l1067_106799

theorem nonagon_perimeter :
  (2 + 2 + 3 + 3 + 1 + 3 + 2 + 2 + 2 = 20) := by
  sorry

end nonagon_perimeter_l1067_106799
