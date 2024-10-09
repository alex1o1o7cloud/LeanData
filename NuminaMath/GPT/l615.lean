import Mathlib

namespace two_digit_number_problem_l615_61582

theorem two_digit_number_problem (a b : ℕ) :
  let M := 10 * b + a
  let N := 10 * a + b
  2 * M - N = 19 * b - 8 * a := by
  sorry

end two_digit_number_problem_l615_61582


namespace derivative_odd_function_l615_61521

theorem derivative_odd_function (a b c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^3 + b * x^2 + c * x + 2) 
    (h_deriv_odd : ∀ x, deriv f (-x) = - deriv f x) : a^2 + c^2 ≠ 0 :=
by
  sorry

end derivative_odd_function_l615_61521


namespace vertical_asymptote_l615_61508

theorem vertical_asymptote (x : ℝ) : 
  (∃ x, 4 * x + 5 = 0) → x = -5/4 :=
by 
  sorry

end vertical_asymptote_l615_61508


namespace estate_value_l615_61599

theorem estate_value (x : ℕ) (E : ℕ) (cook_share : ℕ := 500) 
  (daughter_share : ℕ := 4 * x) (son_share : ℕ := 3 * x) 
  (wife_share : ℕ := 6 * x) (estate_eqn : E = 14 * x) : 
  2 * (daughter_share + son_share) = E ∧ wife_share = 2 * son_share ∧ E = 13 * x + cook_share → 
  E = 7000 :=
by
  sorry

end estate_value_l615_61599


namespace find_salary_l615_61522

-- Define the conditions
variables (S : ℝ) -- S is the man's monthly salary

def saves_25_percent (S : ℝ) : ℝ := 0.25 * S
def expenses (S : ℝ) : ℝ := 0.75 * S
def increased_expenses (S : ℝ) : ℝ := 0.75 * S + 0.10 * (0.75 * S)
def monthly_savings_after_increase (S : ℝ) : ℝ := S - increased_expenses S

-- Define the problem statement
theorem find_salary
  (h1 : saves_25_percent S = 0.25 * S)
  (h2 : increased_expenses S = 0.825 * S)
  (h3 : monthly_savings_after_increase S = 175) :
  S = 1000 :=
sorry

end find_salary_l615_61522


namespace replace_asterisk_l615_61590

theorem replace_asterisk :
  ∃ x : ℤ, (x / 21) * (63 / 189) = 1 ∧ x = 63 := sorry

end replace_asterisk_l615_61590


namespace fourth_root_eq_solution_l615_61556

theorem fourth_root_eq_solution (x : ℝ) (h : Real.sqrt (Real.sqrt x) = 16 / (8 - Real.sqrt (Real.sqrt x))) : x = 256 := by
  sorry

end fourth_root_eq_solution_l615_61556


namespace range_of_m_l615_61516

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (-2 < x ∧ x ≤ 2) → x ≤ m) → m ≥ 2 :=
by
  intro h
  -- insert necessary proof steps here
  sorry

end range_of_m_l615_61516


namespace total_lobster_pounds_l615_61543

variable (lobster_other_harbor1 : ℕ)
variable (lobster_other_harbor2 : ℕ)
variable (lobster_hooper_bay : ℕ)

-- Conditions
axiom h_eq : lobster_hooper_bay = 2 * (lobster_other_harbor1 + lobster_other_harbor2)
axiom other_harbors_eq : lobster_other_harbor1 = 80 ∧ lobster_other_harbor2 = 80

-- Proof statement
theorem total_lobster_pounds : 
  lobster_other_harbor1 + lobster_other_harbor2 + lobster_hooper_bay = 480 :=
by
  sorry

end total_lobster_pounds_l615_61543


namespace find_certain_number_l615_61519

theorem find_certain_number (mystery_number certain_number : ℕ) (h1 : mystery_number = 47) 
(h2 : mystery_number + certain_number = 92) : certain_number = 45 :=
by
  sorry

end find_certain_number_l615_61519


namespace quadratic_real_roots_range_l615_61585

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 3 * x - 9 / 4 = 0) →
  (k >= -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l615_61585


namespace circle_center_l615_61549

theorem circle_center (x y : ℝ) :
  x^2 + 4 * x + y^2 - 6 * y + 1 = 0 → (x + 2, y - 3) = (0, 0) :=
by
  sorry

end circle_center_l615_61549


namespace problem_proof_l615_61580

theorem problem_proof (c d : ℝ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 6 + d = 9 + c) : 
  5 - c = 6 := 
sorry

end problem_proof_l615_61580


namespace ratio_of_volumes_l615_61518

noncomputable def inscribedSphereVolume (s : ℝ) : ℝ := (4 / 3) * Real.pi * (s / 2) ^ 3

noncomputable def cubeVolume (s : ℝ) : ℝ := s ^ 3

theorem ratio_of_volumes (s : ℝ) (h : s > 0) :
  inscribedSphereVolume s / cubeVolume s = Real.pi / 6 :=
by
  sorry

end ratio_of_volumes_l615_61518


namespace ratio_c_d_l615_61571

theorem ratio_c_d (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
    (h1 : 8 * x - 5 * y = c) (h2 : 10 * y - 12 * x = d) 
  : c / d = 2 / 3 := by
  sorry

end ratio_c_d_l615_61571


namespace part1_solution_sets_part2_solution_set_l615_61579

-- Define the function f(x)
def f (a x : ℝ) := x^2 + (1 - a) * x - a

-- Statement for part (1)
theorem part1_solution_sets (a x : ℝ) :
  (a < -1 → f a x < 0 ↔ a < x ∧ x < -1) ∧
  (a = -1 → ¬ (f a x < 0)) ∧
  (a > -1 → f a x < 0 ↔ -1 < x ∧ x < a) :=
sorry

-- Statement for part (2)
theorem part2_solution_set (x : ℝ) :
  (f 2 x) > 0 → (x^3 * f 2 x > 0 ↔ (-1 < x ∧ x < 0) ∨ 2 < x) :=
sorry

end part1_solution_sets_part2_solution_set_l615_61579


namespace max_pN_value_l615_61525

noncomputable def max_probability_units_digit (N: ℕ) (q2 q5 q10: ℚ) : ℚ :=
  let qk (k : ℕ) := (Nat.floor (N / k) : ℚ) / N
  q10 * (2 - q10) + 2 * (q2 - q10) * (q5 - q10)

theorem max_pN_value : ∃ (a b : ℕ), (a.gcd b = 1) ∧ (∀ N q2 q5 q10, max_probability_units_digit N q2 q5 q10 ≤  27 / 100) ∧ (100 * 27 + 100 = 2800) :=
by
  sorry

end max_pN_value_l615_61525


namespace rose_share_correct_l615_61509

-- Define the conditions
def purity_share (P : ℝ) : ℝ := P
def sheila_share (P : ℝ) : ℝ := 5 * P
def rose_share (P : ℝ) : ℝ := 3 * P
def total_rent := 5400

-- The theorem to be proven
theorem rose_share_correct (P : ℝ) (h : purity_share P + sheila_share P + rose_share P = total_rent) : 
  rose_share P = 1800 :=
  sorry

end rose_share_correct_l615_61509


namespace number_of_books_from_second_shop_l615_61504

theorem number_of_books_from_second_shop (books_first_shop : ℕ) (cost_first_shop : ℕ)
    (books_second_shop : ℕ) (cost_second_shop : ℕ) (average_price : ℕ) :
    books_first_shop = 50 →
    cost_first_shop = 1000 →
    cost_second_shop = 800 →
    average_price = 20 →
    average_price * (books_first_shop + books_second_shop) = cost_first_shop + cost_second_shop →
    books_second_shop = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_books_from_second_shop_l615_61504


namespace angle_XYZ_of_excircle_circumcircle_incircle_l615_61551

theorem angle_XYZ_of_excircle_circumcircle_incircle 
  (a b c x y z : ℝ) 
  (hA : a = 50)
  (hB : b = 70)
  (hC : c = 60) 
  (triangleABC : a + b + c = 180) 
  (excircle_Omega : Prop) 
  (incircle_Gamma : Prop) 
  (circumcircle_Omega_triangleXYZ : Prop) 
  (X_on_BC : Prop)
  (Y_on_AB : Prop) 
  (Z_on_CA : Prop): 
  x = 115 := 
by 
  sorry

end angle_XYZ_of_excircle_circumcircle_incircle_l615_61551


namespace sofa_love_seat_ratio_l615_61554

theorem sofa_love_seat_ratio (L S: ℕ) (h1: L = 148) (h2: S + L = 444): S = 2 * L := by
  sorry

end sofa_love_seat_ratio_l615_61554


namespace f_at_neg_one_l615_61594

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_at_neg_one : f (-1) = 0 := by
  sorry

end f_at_neg_one_l615_61594


namespace total_trip_time_l615_61537

noncomputable def speed_coastal := 10 / 20  -- miles per minute
noncomputable def speed_highway := 4 * speed_coastal  -- miles per minute
noncomputable def time_highway := 50 / speed_highway  -- minutes
noncomputable def total_time := 20 + time_highway  -- minutes

theorem total_trip_time : total_time = 45 := 
by
  -- Proof omitted
  sorry

end total_trip_time_l615_61537


namespace sum_area_triangles_lt_total_area_l615_61553

noncomputable def G : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def A_k (k : ℕ+) : ℝ := sorry -- Assume we've defined A_k's expression correctly
noncomputable def S (S1 S2 : ℝ) : ℝ := 2 * S1 - S2

theorem sum_area_triangles_lt_total_area (k : ℕ+) (S1 S2 : ℝ) :
  (A_k k < S S1 S2) :=
sorry

end sum_area_triangles_lt_total_area_l615_61553


namespace vertical_line_divides_triangle_equal_area_l615_61564

theorem vertical_line_divides_triangle_equal_area :
  let A : (ℝ × ℝ) := (1, 2)
  let B : (ℝ × ℝ) := (1, 1)
  let C : (ℝ × ℝ) := (10, 1)
  let area_ABC := (1 / 2 : ℝ) * (C.1 - A.1) * (A.2 - B.2)
  let a : ℝ := 5.5
  let area_left_triangle := (1 / 2 : ℝ) * (a - A.1) * (A.2 - B.2)
  let area_right_triangle := (1 / 2 : ℝ) * (C.1 - a) * (A.2 - B.2)
  area_left_triangle = area_right_triangle :=
by
  sorry

end vertical_line_divides_triangle_equal_area_l615_61564


namespace multiple_for_snack_cost_l615_61570

-- Define the conditions
def kyle_time_to_work : ℕ := 2 -- Kyle bikes for 2 hours to work every day.
def cost_of_snacks (total_cost packs : ℕ) : ℕ := total_cost / packs -- Ryan will pay $2000 to buy 50 packs of snacks.

-- Ryan pays $2000 for 50 packs of snacks.
def cost_per_pack := cost_of_snacks 2000 50

-- The time for a round trip (to work and back)
def round_trip_time (h : ℕ) : ℕ := 2 * h

-- The multiple of the time taken to travel to work and back that equals the cost of a pack of snacks
def multiple (cost time : ℕ) : ℕ := cost / time

-- Statement we need to prove
theorem multiple_for_snack_cost : 
  multiple cost_per_pack (round_trip_time kyle_time_to_work) = 10 :=
  by
  sorry

end multiple_for_snack_cost_l615_61570


namespace find_years_invested_l615_61555

-- Defining the conditions and theorem
variables (P : ℕ) (r1 r2 D : ℝ) (n : ℝ)

-- Given conditions
def principal := (P : ℝ) = 7000
def rate_1 := r1 = 0.15
def rate_2 := r2 = 0.12
def interest_diff := D = 420

-- Theorem to be proven
theorem find_years_invested (h1 : principal P) (h2 : rate_1 r1) (h3 : rate_2 r2) (h4 : interest_diff D) :
  7000 * 0.15 * n - 7000 * 0.12 * n = 420 → n = 2 :=
by
  sorry

end find_years_invested_l615_61555


namespace value_of_expression_l615_61565

open Real

theorem value_of_expression (m n r t : ℝ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := 
by
  sorry

end value_of_expression_l615_61565


namespace eval_32_pow_5_div_2_l615_61573

theorem eval_32_pow_5_div_2 :
  32^(5/2) = 4096 * Real.sqrt 2 :=
by
  sorry

end eval_32_pow_5_div_2_l615_61573


namespace circle_line_intersection_points_l615_61529

theorem circle_line_intersection_points :
  let circle_eqn : ℝ × ℝ → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 16
  let line_eqn  : ℝ × ℝ → Prop := fun p => p.1 = 4
  ∃ (p₁ p₂ : ℝ × ℝ), 
    circle_eqn p₁ ∧ line_eqn p₁ ∧ circle_eqn p₂ ∧ line_eqn p₂ ∧ p₁ ≠ p₂ 
      → ∀ (p : ℝ × ℝ), circle_eqn p ∧ line_eqn p → 
        p = p₁ ∨ p = p₂ ∧ (p₁ ≠ p ∨ p₂ ≠ p)
 := sorry

end circle_line_intersection_points_l615_61529


namespace range_of_a_l615_61545

variable {x a : ℝ}

theorem range_of_a (h1 : 2 * x - a < 0)
                   (h2 : 1 - 2 * x ≥ 7)
                   (h3 : ∀ x, x ≤ -3) : ∀ a, a > -6 :=
by
  sorry

end range_of_a_l615_61545


namespace intersection_M_N_l615_61597

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | x ∣ 4 ∧ 0 < x}

theorem intersection_M_N :
  M ∩ N = {1, 2, 4} :=
sorry

end intersection_M_N_l615_61597


namespace ways_to_select_four_doctors_l615_61536

def num_ways_to_select_doctors (num_internists : ℕ) (num_surgeons : ℕ) (team_size : ℕ) : ℕ :=
  (Nat.choose num_internists 1 * Nat.choose num_surgeons (team_size - 1)) + 
  (Nat.choose num_internists 2 * Nat.choose num_surgeons (team_size - 2)) + 
  (Nat.choose num_internists 3 * Nat.choose num_surgeons (team_size - 3))

theorem ways_to_select_four_doctors : num_ways_to_select_doctors 5 6 4 = 310 := 
by
  sorry

end ways_to_select_four_doctors_l615_61536


namespace probability_mixed_doubles_l615_61535

def num_athletes : ℕ := 6
def num_males : ℕ := 3
def num_females : ℕ := 3
def num_coaches : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select athletes
def total_ways : ℕ :=
  (choose num_athletes 2) * (choose (num_athletes - 2) 2) * (choose (num_athletes - 4) 2)

-- Number of favorable ways to select mixed doubles teams
def favorable_ways : ℕ :=
  (choose num_males 1) * (choose num_females 1) *
  (choose (num_males - 1) 1) * (choose (num_females - 1) 1) *
  (choose 1 1) * (choose 1 1)

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

theorem probability_mixed_doubles :
  probability = 2/5 :=
by
  sorry

end probability_mixed_doubles_l615_61535


namespace sum_of_fractions_l615_61583

theorem sum_of_fractions : (3/7 : ℚ) + (5/14 : ℚ) = 11/14 :=
by
  sorry

end sum_of_fractions_l615_61583


namespace intersection_complement_eq_l615_61560

open Set

def U : Set Int := univ
def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 3}

theorem intersection_complement_eq :
  (U \ M) ∩ N = {3} :=
  by sorry

end intersection_complement_eq_l615_61560


namespace number_of_rectangles_on_3x3_grid_l615_61566

-- Define the grid and its properties
structure Grid3x3 where
  sides_are_2_units_apart : Bool
  diagonal_connections_allowed : Bool
  condition : sides_are_2_units_apart = true ∧ diagonal_connections_allowed = true

-- Define the number_rectangles function
def number_rectangles (g : Grid3x3) : Nat := 60

-- Define the theorem to prove the number of rectangles
theorem number_of_rectangles_on_3x3_grid : ∀ (g : Grid3x3), g.sides_are_2_units_apart = true ∧ g.diagonal_connections_allowed = true → number_rectangles g = 60 := by
  intro g
  intro h
  -- proof goes here
  sorry

end number_of_rectangles_on_3x3_grid_l615_61566


namespace no_positive_int_solutions_l615_61586

theorem no_positive_int_solutions
  (x y z t : ℕ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (ht : 0 < t)
  (h1 : x^2 + 2 * y^2 = z^2)
  (h2 : 2 * x^2 + y^2 = t^2) : false :=
by
  sorry

end no_positive_int_solutions_l615_61586


namespace passengers_taken_second_station_l615_61584

def initial_passengers : ℕ := 288
def passengers_dropped_first_station : ℕ := initial_passengers / 3
def passengers_after_first_station : ℕ := initial_passengers - passengers_dropped_first_station
def passengers_taken_first_station : ℕ := 280
def total_passengers_after_first_station : ℕ := passengers_after_first_station + passengers_taken_first_station
def passengers_dropped_second_station : ℕ := total_passengers_after_first_station / 2
def passengers_left_after_second_station : ℕ := total_passengers_after_first_station - passengers_dropped_second_station
def passengers_at_third_station : ℕ := 248

theorem passengers_taken_second_station : 
  ∃ (x : ℕ), passengers_left_after_second_station + x = passengers_at_third_station ∧ x = 12 :=
by 
  sorry

end passengers_taken_second_station_l615_61584


namespace domain_of_g_l615_61542

def f : ℝ → ℝ := sorry

theorem domain_of_g 
  (hf_dom : ∀ x, -2 ≤ x ∧ x ≤ 4 → f x = f x) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 ↔ (∃ y, y = f x + f (-x)) := 
by {
  sorry
}

end domain_of_g_l615_61542


namespace fourth_term_geometric_sequence_l615_61505

theorem fourth_term_geometric_sequence (x : ℝ) :
  ∃ r : ℝ, (r > 0) ∧ 
  x ≠ 0 ∧
  (3 * x + 3)^2 = x * (6 * x + 6) →
  x = -3 →
  6 * x + 6 ≠ 0 →
  4 * (6 * x + 6) * (3 * x + 3) = -24 :=
by
  -- Placeholder for the proof steps
  sorry

end fourth_term_geometric_sequence_l615_61505


namespace optionC_is_correct_l615_61581

theorem optionC_is_correct (x : ℝ) : (x^2)^3 = x^6 :=
by sorry

end optionC_is_correct_l615_61581


namespace sum_a5_a6_a7_l615_61591

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = q * a n

variables (a : ℕ → ℤ)
variables (h_geo : geometric_sequence a)
variables (h1 : a 2 + a 3 = 1)
variables (h2 : a 3 + a 4 = -2)

theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 24 :=
by
  sorry

end sum_a5_a6_a7_l615_61591


namespace complement_intersection_l615_61552

open Set

namespace UniversalSetProof

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {4, 5} :=
by
  sorry

end UniversalSetProof

end complement_intersection_l615_61552


namespace smallest_value_of_expression_l615_61523

theorem smallest_value_of_expression :
  ∃ (k l : ℕ), 36^k - 5^l = 11 := 
sorry

end smallest_value_of_expression_l615_61523


namespace parallelogram_area_l615_61558

theorem parallelogram_area (base height : ℕ) (h_base : base = 5) (h_height : height = 3) :
  base * height = 15 :=
by
  -- Here would be the proof, but it is omitted per instructions
  sorry

end parallelogram_area_l615_61558


namespace mono_intervals_range_of_a_l615_61530

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.exp (x - 1)

theorem mono_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x, f x a > 0) ∧ 
  (a > 0 → (∀ x, x < 1 - Real.log a → f x a > 0) ∧ (∀ x, x > 1 - Real.log a → f x a < 0)) :=
sorry

theorem range_of_a (h : ∀ x, f x a ≤ 0) : a ≥ 1 :=
sorry

end mono_intervals_range_of_a_l615_61530


namespace calculate_subtraction_l615_61515

def base9_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

def base6_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

theorem calculate_subtraction : base9_to_base10 324 - base6_to_base10 231 = 174 :=
  by sorry

end calculate_subtraction_l615_61515


namespace cistern_length_is_four_l615_61532

noncomputable def length_of_cistern (width depth total_area : ℝ) : ℝ :=
  let L := ((total_area - (2 * width * depth)) / (2 * (width + depth)))
  L

theorem cistern_length_is_four
  (width depth total_area : ℝ)
  (h_width : width = 2)
  (h_depth : depth = 1.25)
  (h_total_area : total_area = 23) :
  length_of_cistern width depth total_area = 4 :=
by 
  sorry

end cistern_length_is_four_l615_61532


namespace son_age_is_10_l615_61598

-- Define the conditions
variables (S F : ℕ)
axiom condition1 : F = S + 30
axiom condition2 : F + 5 = 3 * (S + 5)

-- State the theorem to prove the son's age
theorem son_age_is_10 : S = 10 :=
by
  sorry

end son_age_is_10_l615_61598


namespace perimeter_of_ghost_l615_61528
open Real

def radius := 2
def angle_degrees := 90
def full_circle_degrees := 360

noncomputable def missing_angle := angle_degrees
noncomputable def remaining_angle := full_circle_degrees - missing_angle
noncomputable def fraction_of_circle := remaining_angle / full_circle_degrees
noncomputable def full_circumference := 2 * π * radius
noncomputable def arc_length := fraction_of_circle * full_circumference
noncomputable def radii_length := 2 * radius

theorem perimeter_of_ghost : arc_length + radii_length = 3 * π + 4 :=
by
  sorry

end perimeter_of_ghost_l615_61528


namespace diameter_circle_inscribed_triangle_l615_61588

noncomputable def diameter_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let K := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := K / s
  2 * r

theorem diameter_circle_inscribed_triangle (XY XZ YZ : ℝ) (hXY : XY = 13) (hXZ : XZ = 8) (hYZ : YZ = 9) :
  diameter_of_inscribed_circle XY XZ YZ = 2 * Real.sqrt 210 / 5 := by
{
  rw [hXY, hXZ, hYZ]
  sorry
}

end diameter_circle_inscribed_triangle_l615_61588


namespace parabola_coefficients_sum_l615_61513

theorem parabola_coefficients_sum (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = a * (x + 3)^2 + 2) ∧
  (-6 = a * (1 + 3)^2 + 2) →
  a + b + c = -11/2 :=
by
  sorry

end parabola_coefficients_sum_l615_61513


namespace difference_of_numbers_l615_61567

theorem difference_of_numbers (x y : ℝ) (h1 : x * y = 23) (h2 : x + y = 24) : |x - y| = 22 :=
sorry

end difference_of_numbers_l615_61567


namespace max_brownie_cakes_l615_61572

theorem max_brownie_cakes (m n : ℕ) (h : (m-2)*(n-2) = (1/2)*m*n) :  m * n ≤ 60 :=
sorry

end max_brownie_cakes_l615_61572


namespace angle_bao_proof_l615_61507

noncomputable def angle_bao : ℝ := sorry -- angle BAO in degrees

theorem angle_bao_proof 
    (CD_is_diameter : true)
    (A_on_extension_DC_beyond_C : true)
    (E_on_semicircle : true)
    (B_is_intersection_AE_semicircle : B ≠ E)
    (AB_eq_OE : AB = OE)
    (angle_EOD_30_degrees : EOD = 30) : 
    angle_bao = 7.5 :=
sorry

end angle_bao_proof_l615_61507


namespace find_a_l615_61593

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x ^ 3 - 3 * x) (h1 : f (-1) = 4) : a = -1 :=
by
  sorry

end find_a_l615_61593


namespace negation_of_universal_proposition_l615_61514

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l615_61514


namespace find_a_l615_61576

variable (y : ℝ) (a : ℝ)

theorem find_a (hy : y > 0) (h_expr : (a * y / 20) + (3 * y / 10) = 0.7 * y) : a = 8 :=
by
  sorry

end find_a_l615_61576


namespace area_of_equilateral_triangle_l615_61511

theorem area_of_equilateral_triangle
  (A B C D E : Type) 
  (side_length : ℝ) 
  (medians_perpendicular : Prop) 
  (BD CE : ℝ)
  (inscribed_circle : Prop)
  (equilateral_triangle : A = B ∧ B = C) 
  (s : side_length = 18) 
  (BD_len : BD = 15) 
  (CE_len : CE = 9) 
  : ∃ area, area = 81 * Real.sqrt 3
  :=
by {
  sorry
}

end area_of_equilateral_triangle_l615_61511


namespace sandbox_length_l615_61544

theorem sandbox_length (width : ℕ) (area : ℕ) (h_width : width = 146) (h_area : area = 45552) : ∃ length : ℕ, length = 312 :=
by {
  sorry
}

end sandbox_length_l615_61544


namespace inscribed_circle_radius_squared_l615_61574

theorem inscribed_circle_radius_squared 
  (X Y Z W R S : Type) 
  (XR RY WS SZ : ℝ)
  (hXR : XR = 23) 
  (hRY : RY = 29)
  (hWS : WS = 41) 
  (hSZ : SZ = 31)
  (tangent_at_XY : true) (tangent_at_WZ : true) -- since tangents are assumed by problem
  : ∃ (r : ℝ), r^2 = 905 :=
by sorry

end inscribed_circle_radius_squared_l615_61574


namespace cosine_theorem_l615_61538

theorem cosine_theorem (a b c : ℝ) (A : ℝ) (hA : 0 < A) (hA_lt_pi : A < π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

end cosine_theorem_l615_61538


namespace arithmetic_seq_a2_l615_61589

theorem arithmetic_seq_a2 (a : ℕ → ℤ) (d : ℤ) (h1 : d = -2) 
  (h2 : (a 1 + a 5) / 2 = -1) : 
  a 2 = 1 :=
by
  sorry

end arithmetic_seq_a2_l615_61589


namespace positive_integer_base_conversion_l615_61562

theorem positive_integer_base_conversion (A B : ℕ) (h1 : A < 9) (h2 : B < 7) 
(h3 : 9 * A + B = 7 * B + A) : 9 * 3 + 4 = 31 :=
by sorry

end positive_integer_base_conversion_l615_61562


namespace least_number_to_add_l615_61547

theorem least_number_to_add (n divisor : ℕ) (h₁ : n = 27306) (h₂ : divisor = 151) : 
  ∃ k : ℕ, k = 25 ∧ (n + k) % divisor = 0 := 
by
  sorry

end least_number_to_add_l615_61547


namespace third_place_prize_is_120_l615_61561

noncomputable def prize_for_third_place (total_prize : ℕ) (first_place_prize : ℕ) (second_place_prize : ℕ) (prize_per_novel : ℕ) (num_novels_receiving_prize : ℕ) : ℕ :=
  let remaining_prize := total_prize - first_place_prize - second_place_prize
  let total_other_prizes := num_novels_receiving_prize * prize_per_novel
  remaining_prize - total_other_prizes

theorem third_place_prize_is_120 : prize_for_third_place 800 200 150 22 15 = 120 := by
  sorry

end third_place_prize_is_120_l615_61561


namespace scaled_multiplication_l615_61540

theorem scaled_multiplication
  (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 :=
by
  sorry

end scaled_multiplication_l615_61540


namespace part1_part2_l615_61569

-- Definition of points and given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Conditions for part 1
def A1 (a : ℝ) : Point := { x := -2, y := a + 1 }
def B1 (a : ℝ) : Point := { x := a - 1, y := 4 }

-- Definition for distance calculation
def distance (p1 p2 : Point) : ℝ := abs (p2.x - p1.x)

-- Problem 1 Statement
theorem part1 (a : ℝ) (h : a = 3) : distance (A1 a) (B1 a) = 4 :=
by 
  sorry

-- Conditions for part 2
def C2 (b : ℝ) : Point := { x := b - 2, y := b }

-- Problem 2 Statement
theorem part2 (b : ℝ) (h : abs b = 1) :
  (C2 b = { x := -1, y := 1 } ∨ C2 b = { x := -3, y := -1 }) :=
by
  sorry

end part1_part2_l615_61569


namespace clara_sells_total_cookies_l615_61534

theorem clara_sells_total_cookies :
  let cookies_per_box_1 := 12
  let cookies_per_box_2 := 20
  let cookies_per_box_3 := 16
  let cookies_per_box_4 := 18
  let cookies_per_box_5 := 22

  let boxes_sold_1 := 50.5
  let boxes_sold_2 := 80.25
  let boxes_sold_3 := 70.75
  let boxes_sold_4 := 65.5
  let boxes_sold_5 := 55.25

  let total_cookies_1 := cookies_per_box_1 * boxes_sold_1
  let total_cookies_2 := cookies_per_box_2 * boxes_sold_2
  let total_cookies_3 := cookies_per_box_3 * boxes_sold_3
  let total_cookies_4 := cookies_per_box_4 * boxes_sold_4
  let total_cookies_5 := cookies_per_box_5 * boxes_sold_5

  let total_cookies := total_cookies_1 + total_cookies_2 + total_cookies_3 + total_cookies_4 + total_cookies_5

  total_cookies = 5737.5 :=
by
  sorry

end clara_sells_total_cookies_l615_61534


namespace triangle_area_l615_61550

theorem triangle_area
  (area_WXYZ : ℝ)
  (side_small_squares : ℝ)
  (AB_eq_AC : ℝ)
  (A_coincides_with_O : ℝ)
  (area : ℝ) :
  area_WXYZ = 49 →  -- The area of square WXYZ is 49 cm^2
  side_small_squares = 2 → -- Sides of the smaller squares are 2 cm long
  AB_eq_AC = AB_eq_AC → -- Triangle ABC is isosceles with AB = AC
  A_coincides_with_O = A_coincides_with_O → -- A coincides with O
  area = 45 / 4 := -- The area of triangle ABC is 45/4 cm^2
by
  sorry

end triangle_area_l615_61550


namespace no_function_f_exists_l615_61502

theorem no_function_f_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2013 :=
by sorry

end no_function_f_exists_l615_61502


namespace evaluate_expression_l615_61546

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end evaluate_expression_l615_61546


namespace necessary_condition_abs_sq_necessary_and_sufficient_add_l615_61527

theorem necessary_condition_abs_sq (a b : ℝ) : a^2 > b^2 → |a| > |b| :=
sorry

theorem necessary_and_sufficient_add (a b c : ℝ) :
  (a > b) ↔ (a + c > b + c) :=
sorry

end necessary_condition_abs_sq_necessary_and_sufficient_add_l615_61527


namespace inequality_cannot_hold_l615_61563

theorem inequality_cannot_hold (a b : ℝ) (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) :=
by {
  sorry
}

end inequality_cannot_hold_l615_61563


namespace misread_system_of_equations_solutions_l615_61510

theorem misread_system_of_equations_solutions (a b : ℤ) (x₁ y₁ x₂ y₂ : ℤ)
  (h1 : x₁ = -3) (h2 : y₁ = -1) (h3 : x₂ = 5) (h4 : y₂ = 4)
  (eq1 : a * x₂ + 5 * y₂ = 15)
  (eq2 : 4 * x₁ - b * y₁ = -2) :
  a = -1 ∧ b = 10 ∧ a ^ 2023 + (- (1 / 10 : ℚ) * b) ^ 2023 = -2 := by
  -- Translate misreading conditions into theorems we need to prove (note: skipping proof).
  have hb : b = 10 := by sorry
  have ha : a = -1 := by sorry
  exact ⟨ha, hb, by simp [ha, hb]; norm_num⟩

end misread_system_of_equations_solutions_l615_61510


namespace minimum_lines_for_regions_l615_61587

theorem minimum_lines_for_regions (n : ℕ) : 1 + n * (n + 1) / 2 ≥ 1000 ↔ n ≥ 45 :=
sorry

end minimum_lines_for_regions_l615_61587


namespace function_is_increasing_l615_61578

theorem function_is_increasing : ∀ (x1 x2 : ℝ), x1 < x2 → (2 * x1 + 1) < (2 * x2 + 1) :=
by sorry

end function_is_increasing_l615_61578


namespace total_precious_stones_l615_61575

theorem total_precious_stones (agate olivine diamond : ℕ)
  (h1 : olivine = agate + 5)
  (h2 : diamond = olivine + 11)
  (h3 : agate = 30) : 
  agate + olivine + diamond = 111 :=
by
  sorry

end total_precious_stones_l615_61575


namespace max_integer_solutions_l615_61539

def quad_func (x : ℝ) : ℝ := x^2 - 6 * x + 1

theorem max_integer_solutions (p : ℝ → ℝ) : 
  (p = quad_func) →
  (∃ n1 n2 n3 n4 : ℤ, 
    ((p n1 = p (n1 ^ 2)) ∧ (p n2 = p (n2 ^ 2)) ∧ 
    (p n3 = p (n3 ^ 2)) ∧ (p n4 = p (n4 ^ 2))) ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ 
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ 
    n3 ≠ n4) :=
by
  sorry

end max_integer_solutions_l615_61539


namespace problem_1_problem_2_l615_61512

theorem problem_1 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2) → (a = 0 ∨ a = 1) :=
by sorry

theorem problem_2 (a : ℝ) : (∀ x1 x2 : ℝ, (a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) → x1 = x2 ∨ ¬ ∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a ≥ 1 ∨ a = 0) :=
by sorry

end problem_1_problem_2_l615_61512


namespace minimum_a_for_f_leq_one_range_of_a_for_max_value_l615_61520

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * log x - (1 / 3) * a * x^3 + 2 * x

theorem minimum_a_for_f_leq_one :
  ∀ {a : ℝ}, (a > 0) → (∀ x : ℝ, f a x ≤ 1) → (a ≥ 3) :=
sorry

theorem range_of_a_for_max_value :
  ∀ {a : ℝ}, (a > 0) → (∃ B : ℝ, ∀ x : ℝ, f a x ≤ B) ↔ (0 < a ∧ a ≤ (3 / 2) * exp 3) :=
sorry

end minimum_a_for_f_leq_one_range_of_a_for_max_value_l615_61520


namespace find_f_half_l615_61596

theorem find_f_half (f : ℝ → ℝ) (h : ∀ x, f (2 * x / (x + 1)) = x^2 - 1) : f (1 / 2) = -8 / 9 :=
by
  sorry

end find_f_half_l615_61596


namespace min_value_reciprocal_sum_l615_61501

theorem min_value_reciprocal_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  (∃ c, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x + 1/y) ≥ c) ∧ (1/a + 1/b = c)) 
:= 
sorry

end min_value_reciprocal_sum_l615_61501


namespace proof_problem_l615_61559

theorem proof_problem (a b : ℝ) (h : a^2 + b^2 + 2*a - 4*b + 5 = 0) : 2*a^2 + 4*b - 3 = 7 :=
sorry

end proof_problem_l615_61559


namespace loss_percentage_eq_100_div_9_l615_61595

theorem loss_percentage_eq_100_div_9 :
  ( ∀ C : ℝ,
    (11 * C > 1) ∧ 
    (8.25 * (1 + 0.20) * C = 1) →
    ((C - 1/11) / C * 100) = 100 / 9) 
  :=
by sorry

end loss_percentage_eq_100_div_9_l615_61595


namespace evaluate_expression_l615_61506

theorem evaluate_expression :
  (3 + 6 + 9) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 6 + 9) = 5 / 6 :=
by sorry

end evaluate_expression_l615_61506


namespace baseball_cards_initial_count_unkn_l615_61557

-- Definitions based on the conditions
def cardValue : ℕ := 6
def tradedCards : ℕ := 2
def receivedCardsValue : ℕ := (3 * 2) + 9   -- 3 cards worth $2 each and 1 card worth $9
def profit : ℕ := receivedCardsValue - (tradedCards * cardValue)

-- Lean 4 statement to represent the proof problem
theorem baseball_cards_initial_count_unkn (h_trade : tradedCards * cardValue = 12)
    (h_receive : receivedCardsValue = 15)
    (h_profit : profit = 3) : ∃ n : ℕ, n >= 2 ∧ n = 2 + (n - 2) :=
sorry

end baseball_cards_initial_count_unkn_l615_61557


namespace fish_caught_300_l615_61577

def fish_caught_at_dawn (F : ℕ) : Prop :=
  (3 * F / 5) = 180

theorem fish_caught_300 : ∃ F, fish_caught_at_dawn F ∧ F = 300 := 
by 
  use 300 
  have h1 : 3 * 300 / 5 = 180 := by norm_num 
  exact ⟨h1, rfl⟩

end fish_caught_300_l615_61577


namespace elder_age_is_twenty_l615_61592

-- Let e be the present age of the elder person
-- Let y be the present age of the younger person

def ages_diff_by_twelve (e y : ℕ) : Prop :=
  e = y + 12

def elder_five_years_ago (e y : ℕ) : Prop :=
  e - 5 = 5 * (y - 5)

theorem elder_age_is_twenty (e y : ℕ) (h1 : ages_diff_by_twelve e y) (h2 : elder_five_years_ago e y) :
  e = 20 :=
by
  sorry

end elder_age_is_twenty_l615_61592


namespace sum_of_values_of_z_l615_61533

def f (x : ℝ) := x^2 - 2*x + 3

theorem sum_of_values_of_z (z : ℝ) (h : f (5 * z) = 7) : z = 2 / 25 :=
sorry

end sum_of_values_of_z_l615_61533


namespace Vasechkin_result_l615_61541

-- Define the operations
def P (x : ℕ) : ℕ := (x / 2 * 7) - 1001
def V (x : ℕ) : ℕ := (x / 8) ^ 2 - 1001

-- Define the proposition
theorem Vasechkin_result (x : ℕ) (h_prime : P x = 7) : V x = 295 := 
by {
  -- Proof is omitted
  sorry
}

end Vasechkin_result_l615_61541


namespace problem_statement_l615_61500

def operation (x y z : ℕ) : ℕ := (2 * x * y + y^2) * z^3

theorem problem_statement : operation 7 (operation 4 5 3) 2 = 24844760 :=
by
  sorry

end problem_statement_l615_61500


namespace subtraction_decimal_nearest_hundredth_l615_61531

theorem subtraction_decimal_nearest_hundredth : 
  (845.59 - 249.27 : ℝ) = 596.32 :=
by
  sorry

end subtraction_decimal_nearest_hundredth_l615_61531


namespace solve_triangle_problem_l615_61524
noncomputable def triangle_problem (A B C a b c : ℝ) (area : ℝ) : Prop :=
  (2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0) ∧
  area = Real.sqrt 3 ∧ 
  b + c = 5 →
  (A = Real.pi / 3) ∧ (a = Real.sqrt 13)

-- Lean statement for the proof problem
theorem solve_triangle_problem 
  (A B C a b c : ℝ) 
  (h1 : 2 * c * Real.sin B * Real.cos A - b * Real.sin C = 0)
  (h2 : 1/2 * b * c * Real.sin A = Real.sqrt 3)
  (h3 : b + c = 5) :
  A = Real.pi / 3 ∧ a = Real.sqrt 13 :=
sorry

end solve_triangle_problem_l615_61524


namespace length_AB_l615_61517

theorem length_AB (r : ℝ) (A B : ℝ) (π : ℝ) : 
  r = 4 ∧ π = 3 ∧ (A = 8 ∧ B = 8) → (A = B ∧ A + B = 24 → AB = 6) :=
by
  intros
  sorry

end length_AB_l615_61517


namespace flower_profit_equation_l615_61548

theorem flower_profit_equation
  (initial_plants : ℕ := 3)
  (initial_profit_per_plant : ℕ := 10)
  (decrease_in_profit_per_additional_plant : ℕ := 1)
  (target_profit_per_pot : ℕ := 40)
  (x : ℕ) :
  (initial_plants + x) * (initial_profit_per_plant - x) = target_profit_per_pot :=
sorry

end flower_profit_equation_l615_61548


namespace percentage_increase_14point4_from_12_l615_61503

theorem percentage_increase_14point4_from_12 (x : ℝ) (h : x = 14.4) : 
  ((x - 12) / 12) * 100 = 20 := 
by
  sorry

end percentage_increase_14point4_from_12_l615_61503


namespace find_y_l615_61568

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (h3 : x = 1) : y = 13 := by
  sorry

end find_y_l615_61568


namespace nancy_earns_more_l615_61526

theorem nancy_earns_more (giraffe_jade : ℕ) (giraffe_price : ℕ) (elephant_jade : ℕ)
    (elephant_price : ℕ) (total_jade : ℕ) (giraffes : ℕ) (elephants : ℕ) (giraffe_total : ℕ) (elephant_total : ℕ)
    (diff : ℕ) :
    giraffe_jade = 120 →
    giraffe_price = 150 →
    elephant_jade = 240 →
    elephant_price = 350 →
    total_jade = 1920 →
    giraffes = total_jade / giraffe_jade →
    giraffe_total = giraffes * giraffe_price →
    elephants = total_jade / elephant_jade →
    elephant_total = elephants * elephant_price →
    diff = elephant_total - giraffe_total →
    diff = 400 :=
by
  intros
  sorry

end nancy_earns_more_l615_61526
