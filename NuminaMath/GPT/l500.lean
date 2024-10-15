import Mathlib

namespace NUMINAMATH_GPT_shadow_of_tree_l500_50021

open Real

theorem shadow_of_tree (height_tree height_pole shadow_pole shadow_tree : ℝ) 
(h1 : height_tree = 12) (h2 : height_pole = 150) (h3 : shadow_pole = 100) 
(h4 : height_tree / shadow_tree = height_pole / shadow_pole) : shadow_tree = 8 := 
by 
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_shadow_of_tree_l500_50021


namespace NUMINAMATH_GPT_length_of_chord_MN_l500_50063

theorem length_of_chord_MN 
  (m n : ℝ)
  (h1 : ∃ (M N : ℝ × ℝ), M ≠ N ∧ M.1 * M.1 + M.2 * M.2 + m * M.1 + n * M.2 - 4 = 0 ∧ N.1 * N.1 + N.2 * N.2 + m * N.1 + n * N.2 - 4 = 0 
    ∧ N.2 = M.1 ∧ N.1 = M.2) 
  (h2 : x + y = 0)
  : length_of_chord = 4 := sorry

end NUMINAMATH_GPT_length_of_chord_MN_l500_50063


namespace NUMINAMATH_GPT_largest_integer_l500_50059

theorem largest_integer (a b c d : ℤ) 
  (h1 : a + b + c = 210) 
  (h2 : a + b + d = 230) 
  (h3 : a + c + d = 245) 
  (h4 : b + c + d = 260) : 
  d = 105 :=
by 
  sorry

end NUMINAMATH_GPT_largest_integer_l500_50059


namespace NUMINAMATH_GPT_intersection_of_sets_l500_50040

def M : Set ℝ := { x | 3 * x - 6 ≥ 0 }
def N : Set ℝ := { x | x^2 < 16 }

theorem intersection_of_sets : M ∩ N = { x | 2 ≤ x ∧ x < 4 } :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_sets_l500_50040


namespace NUMINAMATH_GPT_range_of_m_l500_50028

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), -2 ≤ x ∧ x ≤ 3 ∧ m * x + 6 = 0) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l500_50028


namespace NUMINAMATH_GPT_smallest_number_two_reps_l500_50015

theorem smallest_number_two_reps : 
  ∃ (n : ℕ), (∀ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = n ∧ 3 * x2 + 4 * y2 = n → (x1 = x2 ∧ y1 = y2 ∨ ¬(x1 = x2 ∧ y1 = y2))) ∧ 
  ∀ m < n, (∀ x y : ℕ, ¬(3 * x + 4 * y = m ∧ ¬∃ (x1 y1 : ℕ), 3 * x1 + 4 * y1 = m) ∧ 
            (∃ x1 y1 x2 y2 : ℕ, 3 * x1 + 4 * y1 = m ∧ 3 * x2 + 4 * y2 = m ∧ ¬(x1 = x2 ∧ y1 = y2))) :=
  sorry

end NUMINAMATH_GPT_smallest_number_two_reps_l500_50015


namespace NUMINAMATH_GPT_christmas_tree_problem_l500_50031

theorem christmas_tree_problem (b t : ℕ) (h1 : t = b + 1) (h2 : 2 * b = t - 1) : b = 3 ∧ t = 4 :=
by
  sorry

end NUMINAMATH_GPT_christmas_tree_problem_l500_50031


namespace NUMINAMATH_GPT_three_students_two_groups_l500_50052

theorem three_students_two_groups : 
  (2 : ℕ) ^ 3 = 8 := 
by
  sorry

end NUMINAMATH_GPT_three_students_two_groups_l500_50052


namespace NUMINAMATH_GPT_number_of_valid_triples_l500_50042

theorem number_of_valid_triples : 
  ∃ n, n = 7 ∧ ∀ (a b c : ℕ), b = 2023 → a ≤ b → b ≤ c → a * c = 2023^2 → (n = 7) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_valid_triples_l500_50042


namespace NUMINAMATH_GPT_desired_depth_is_50_l500_50005

noncomputable def desired_depth_dig (d days : ℝ) : ℝ :=
  let initial_man_hours := 45 * 8 * d
  let additional_man_hours := 100 * 6 * d
  (initial_man_hours / additional_man_hours) * 30

theorem desired_depth_is_50 (d : ℝ) : desired_depth_dig d = 50 :=
  sorry

end NUMINAMATH_GPT_desired_depth_is_50_l500_50005


namespace NUMINAMATH_GPT_proof_problem_l500_50004

variables (p q : Prop)

-- Assuming p is true and q is false
axiom p_is_true : p
axiom q_is_false : ¬ q

-- Proving that (¬p) ∨ (¬q) is true
theorem proof_problem : (¬p) ∨ (¬q) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l500_50004


namespace NUMINAMATH_GPT_opposite_face_of_orange_is_blue_l500_50041

structure CubeOrientation :=
  (top : String)
  (front : String)
  (right : String)

def first_view : CubeOrientation := { top := "B", front := "Y", right := "S" }
def second_view : CubeOrientation := { top := "B", front := "V", right := "S" }
def third_view : CubeOrientation := { top := "B", front := "K", right := "S" }

theorem opposite_face_of_orange_is_blue
  (colors : List String)
  (c1 : CubeOrientation)
  (c2 : CubeOrientation)
  (c3 : CubeOrientation)
  (no_orange_in_views : "O" ∉ colors.erase c1.top ∧ "O" ∉ colors.erase c1.front ∧ "O" ∉ colors.erase c1.right ∧
                         "O" ∉ colors.erase c2.top ∧ "O" ∉ colors.erase c2.front ∧ "O" ∉ colors.erase c2.right ∧
                         "O" ∉ colors.erase c3.top ∧ "O" ∉ colors.erase c3.front ∧ "O" ∉ colors.erase c3.right) :
  (c1.top = "B" → c2.top = "B" → c3.top = "B" → c1.right = "S" → c2.right = "S" → c3.right = "S" → 
  ∃ opposite_color, opposite_color = "B") :=
sorry

end NUMINAMATH_GPT_opposite_face_of_orange_is_blue_l500_50041


namespace NUMINAMATH_GPT_point_B_represents_2_or_neg6_l500_50094

def A : ℤ := -2

def B (move : ℤ) : ℤ := A + move

theorem point_B_represents_2_or_neg6 (move : ℤ) (h : move = 4 ∨ move = -4) : 
  B move = 2 ∨ B move = -6 :=
by
  cases h with
  | inl h1 => 
    rw [h1]
    unfold B
    unfold A
    simp
  | inr h1 => 
    rw [h1]
    unfold B
    unfold A
    simp

end NUMINAMATH_GPT_point_B_represents_2_or_neg6_l500_50094


namespace NUMINAMATH_GPT_gcd_765432_654321_l500_50013

theorem gcd_765432_654321 : Int.gcd 765432 654321 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_765432_654321_l500_50013


namespace NUMINAMATH_GPT_range_of_m_l500_50091

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 1 > m

def proposition_q (m : ℝ) : Prop :=
  3 - m > 1

theorem range_of_m (m : ℝ) (p_false : ¬proposition_p m) (q_true : proposition_q m) (pq_false : ¬(proposition_p m ∧ proposition_q m)) (porq_true : proposition_p m ∨ proposition_q m) : 
  1 ≤ m ∧ m < 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l500_50091


namespace NUMINAMATH_GPT_probability_of_selecting_one_marble_each_color_l500_50009

theorem probability_of_selecting_one_marble_each_color
  (total_red_marbles : ℕ) (total_blue_marbles : ℕ) (total_green_marbles : ℕ) (total_selected_marbles : ℕ) 
  (total_marble_count : ℕ) : 
  total_red_marbles = 3 → total_blue_marbles = 3 → total_green_marbles = 3 → total_selected_marbles = 3 → total_marble_count = 9 →
  (27 / 84) = 9 / 28 :=
by
  intros h_red h_blue h_green h_selected h_total
  sorry

end NUMINAMATH_GPT_probability_of_selecting_one_marble_each_color_l500_50009


namespace NUMINAMATH_GPT_total_sides_is_48_l500_50051

-- Definitions based on the conditions
def num_dice_tom : Nat := 4
def num_dice_tim : Nat := 4
def sides_per_die : Nat := 6

-- The proof problem statement
theorem total_sides_is_48 : (num_dice_tom + num_dice_tim) * sides_per_die = 48 := by
  sorry

end NUMINAMATH_GPT_total_sides_is_48_l500_50051


namespace NUMINAMATH_GPT_percentage_rent_this_year_l500_50095

variables (E : ℝ)

-- Define the conditions from the problem
def rent_last_year (E : ℝ) : ℝ := 0.20 * E
def earnings_this_year (E : ℝ) : ℝ := 1.15 * E
def rent_this_year (E : ℝ) : ℝ := 1.4375 * rent_last_year E

-- The main statement to prove
theorem percentage_rent_this_year : 
  0.2875 * E = (25 / 100) * (earnings_this_year E) :=
by sorry

end NUMINAMATH_GPT_percentage_rent_this_year_l500_50095


namespace NUMINAMATH_GPT_smallest_n_divisible_by_125000_l500_50024

noncomputable def geometric_term_at (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

noncomputable def first_term : ℚ := 5 / 8
noncomputable def second_term : ℚ := 25
noncomputable def common_ratio : ℚ := second_term / first_term

theorem smallest_n_divisible_by_125000 :
  ∃ n : ℕ, n ≥ 7 ∧ geometric_term_at first_term common_ratio n % 125000 = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_125000_l500_50024


namespace NUMINAMATH_GPT_polygon_sides_l500_50060

theorem polygon_sides (sum_of_interior_angles : ℝ) (x : ℝ) (h : sum_of_interior_angles = 1080) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l500_50060


namespace NUMINAMATH_GPT_complement_of_A_relative_to_U_l500_50034

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 4, 5}

-- Define the proof statement for the complement of A with respect to U
theorem complement_of_A_relative_to_U : (U \ A) = {2} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_relative_to_U_l500_50034


namespace NUMINAMATH_GPT_mathematical_proof_l500_50065

noncomputable def proof_problem (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : Prop :=
  (1 + x) / y < 2 ∨ (1 + y) / x < 2

theorem mathematical_proof (x y : ℝ) (hx_pos : y > 0) (hxy_gt2 : x + y > 2) : proof_problem x y hx_pos hxy_gt2 :=
by {
  sorry
}

end NUMINAMATH_GPT_mathematical_proof_l500_50065


namespace NUMINAMATH_GPT_range_of_a_plus_b_l500_50083

variable {a b : ℝ}

def has_two_real_roots (a b : ℝ) : Prop :=
  let discriminant := b^2 - 4 * a * (-4)
  discriminant ≥ 0

def has_root_in_interval (a b : ℝ) : Prop :=
  (a + b - 4) * (4 * a + 2 * b - 4) < 0

theorem range_of_a_plus_b 
  (h1 : has_two_real_roots a b) 
  (h2 : has_root_in_interval a b) 
  (h3 : a > 0) : 
  a + b < 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l500_50083


namespace NUMINAMATH_GPT_decorations_total_l500_50007

def number_of_skulls : Nat := 12
def number_of_broomsticks : Nat := 4
def number_of_spiderwebs : Nat := 12
def number_of_pumpkins (spiderwebs : Nat) : Nat := 2 * spiderwebs
def number_of_cauldron : Nat := 1
def number_of_lanterns (trees : Nat) : Nat := 3 * trees
def number_of_scarecrows (trees : Nat) : Nat := 1 * (trees / 2)
def total_stickers : Nat := 30
def stickers_per_window (stickers : Nat) (windows : Nat) : Nat := (stickers / 2) / windows
def additional_decorations (bought : Nat) (used_percent : Nat) (leftover : Nat) : Nat := ((bought * used_percent) / 100) + leftover

def total_decorations : Nat :=
  number_of_skulls +
  number_of_broomsticks +
  number_of_spiderwebs +
  (number_of_pumpkins number_of_spiderwebs) +
  number_of_cauldron +
  (number_of_lanterns 5) +
  (number_of_scarecrows 4) +
  (additional_decorations 25 70 15)

theorem decorations_total : total_decorations = 102 := by
  sorry

end NUMINAMATH_GPT_decorations_total_l500_50007


namespace NUMINAMATH_GPT_percentage_error_l500_50006

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end NUMINAMATH_GPT_percentage_error_l500_50006


namespace NUMINAMATH_GPT_find_a_l500_50084

-- Define the circle equation and the line equation as conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1
def line_eq (x y a : ℝ) : Prop := y = x + a
def chord_length (l : ℝ) : Prop := l = 2

-- State the main problem
theorem find_a (a : ℝ) (h1 : ∀ x y : ℝ, circle_eq x y → ∃ y', line_eq x y' a ∧ chord_length 2) :
  a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l500_50084


namespace NUMINAMATH_GPT_find_k_l500_50023

-- Define the vectors and the condition that k · a + b is perpendicular to a
theorem find_k 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (k : ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (-2, 0))
  (h_perpendicular : ∀ (k : ℝ), (k * a.1 + b.1, k * a.2 + b.2) • a = 0 ) : k = 2 / 5 :=
sorry

end NUMINAMATH_GPT_find_k_l500_50023


namespace NUMINAMATH_GPT_spherical_segment_equals_circle_area_l500_50061

noncomputable def spherical_segment_surface_area (R H : ℝ) : ℝ := 2 * Real.pi * R * H
noncomputable def circle_area (b : ℝ) : ℝ := Real.pi * (b * b)

theorem spherical_segment_equals_circle_area
  (R H b : ℝ) 
  (hb : b^2 = 2 * R * H) 
  : spherical_segment_surface_area R H = circle_area b :=
by
  sorry

end NUMINAMATH_GPT_spherical_segment_equals_circle_area_l500_50061


namespace NUMINAMATH_GPT_ratio_of_areas_l500_50047

theorem ratio_of_areas
  (R_X R_Y : ℝ)
  (h : (60 / 360) * 2 * Real.pi * R_X = (40 / 360) * 2 * Real.pi * R_Y) :
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l500_50047


namespace NUMINAMATH_GPT_trains_distance_l500_50054

theorem trains_distance (t x : ℝ) 
  (h1 : x = 20 * t)
  (h2 : x + 50 = 25 * t) : 
  x + (x + 50) = 450 := 
by 
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_trains_distance_l500_50054


namespace NUMINAMATH_GPT_driver_spending_increase_l500_50048

theorem driver_spending_increase (P Q : ℝ) (X : ℝ) (h1 : 1.20 * P = (1 + 20 / 100) * P) (h2 : 0.90 * Q = (1 - 10 / 100) * Q) :
  (1 + X / 100) * (P * Q) = 1.20 * P * 0.90 * Q → X = 8 := 
by
  sorry

end NUMINAMATH_GPT_driver_spending_increase_l500_50048


namespace NUMINAMATH_GPT_conjunction_used_in_proposition_l500_50030

theorem conjunction_used_in_proposition (x : ℝ) (h : x^2 = 4) :
  (x = 2 ∨ x = -2) :=
sorry

end NUMINAMATH_GPT_conjunction_used_in_proposition_l500_50030


namespace NUMINAMATH_GPT_smallest_constant_for_triangle_l500_50027

theorem smallest_constant_for_triangle 
  (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)  
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (a^2 + b^2 + c^2) / (a*b + b*c + c*a) < 2 := 
  sorry

end NUMINAMATH_GPT_smallest_constant_for_triangle_l500_50027


namespace NUMINAMATH_GPT_white_pairs_coincide_l500_50020

theorem white_pairs_coincide :
  ∀ (red_triangles blue_triangles white_triangles : ℕ)
    (red_pairs blue_pairs red_blue_pairs : ℕ),
  red_triangles = 4 →
  blue_triangles = 4 →
  white_triangles = 6 →
  red_pairs = 3 →
  blue_pairs = 2 →
  red_blue_pairs = 1 →
  (2 * white_triangles - red_triangles - blue_triangles - red_blue_pairs) = white_triangles →
  6 = white_triangles :=
by
  intros red_triangles blue_triangles white_triangles
         red_pairs blue_pairs red_blue_pairs
         H_red H_blue H_white
         H_red_pairs H_blue_pairs H_red_blue_pairs
         H_pairs
  sorry

end NUMINAMATH_GPT_white_pairs_coincide_l500_50020


namespace NUMINAMATH_GPT_right_triangle_legs_sum_squares_area_l500_50082

theorem right_triangle_legs_sum_squares_area:
  ∀ (a b c : ℝ), 
  (0 < a) → (0 < b) → (0 < c) → 
  (a^2 + b^2 = c^2) → 
  (1 / 2 * a * b = 24) → 
  (a^2 + b^2 = 48) → 
  (a = 2 * Real.sqrt 6 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 * Real.sqrt 3) := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_sum_squares_area_l500_50082


namespace NUMINAMATH_GPT_find_A_find_b_and_c_l500_50049

open Real

variable {a b c A B C : ℝ}

-- Conditions for the problem
axiom triangle_sides : ∀ {A B C : ℝ}, a > 0
axiom sine_law_condition : b * sin B + c * sin C - sqrt 2 * b * sin C = a * sin A
axiom degrees_60 : B = π / 3
axiom side_a : a = 2

theorem find_A : A = π / 4 :=
by sorry

theorem find_b_and_c (h : A = π / 4) (hB : B = π / 3) (ha : a = 2) : b = sqrt 6 ∧ c = 1 + sqrt 3 :=
by sorry

end NUMINAMATH_GPT_find_A_find_b_and_c_l500_50049


namespace NUMINAMATH_GPT_number_of_cars_repaired_l500_50097

theorem number_of_cars_repaired
  (oil_change_cost repair_cost car_wash_cost : ℕ)
  (oil_changes repairs car_washes total_earnings : ℕ)
  (h₁ : oil_change_cost = 20)
  (h₂ : repair_cost = 30)
  (h₃ : car_wash_cost = 5)
  (h₄ : oil_changes = 5)
  (h₅ : car_washes = 15)
  (h₆ : total_earnings = 475)
  (h₇ : 5 * oil_change_cost + 15 * car_wash_cost + repairs * repair_cost = total_earnings) :
  repairs = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_cars_repaired_l500_50097


namespace NUMINAMATH_GPT_pounds_of_apples_needed_l500_50076

-- Define the conditions
def n : ℕ := 8
def c_p : ℕ := 1
def a_p : ℝ := 2.00
def c_crust : ℝ := 2.00
def c_lemon : ℝ := 0.50
def c_butter : ℝ := 1.50

-- Define the theorem to be proven
theorem pounds_of_apples_needed : 
  (n * c_p - (c_crust + c_lemon + c_butter)) / a_p = 2 := 
by
  sorry

end NUMINAMATH_GPT_pounds_of_apples_needed_l500_50076


namespace NUMINAMATH_GPT_enclosed_area_is_correct_l500_50053

noncomputable def area_between_curves : ℝ := 
  let circle (x y : ℝ) := x^2 + y^2 = 4
  let cubic_parabola (x : ℝ) := - 1 / 2 * x^3 + 2 * x
  let x1 : ℝ := -2
  let x2 : ℝ := Real.sqrt 2
  -- Properly calculate the area between the two curves
  sorry

theorem enclosed_area_is_correct :
  area_between_curves = 3 * ( Real.pi + 1 ) / 2 :=
sorry

end NUMINAMATH_GPT_enclosed_area_is_correct_l500_50053


namespace NUMINAMATH_GPT_f_at_zero_f_positive_f_increasing_l500_50079

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : true
axiom f_nonzero : f 0 ≠ 0
axiom f_pos_gt1 (x : ℝ) : x > 0 → f x > 1
axiom f_add (a b : ℝ) : f (a + b) = f a * f b

theorem f_at_zero : f 0 = 1 :=
sorry

theorem f_positive (x : ℝ) : f x > 0 :=
sorry

theorem f_increasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

end NUMINAMATH_GPT_f_at_zero_f_positive_f_increasing_l500_50079


namespace NUMINAMATH_GPT_total_handshakes_calculation_l500_50011

-- Define the conditions
def teams := 3
def players_per_team := 5
def total_players := teams * players_per_team
def referees := 2

def handshakes_among_players := (total_players * (players_per_team * (teams - 1))) / 2
def handshakes_with_referees := total_players * referees

def total_handshakes := handshakes_among_players + handshakes_with_referees

-- Define the theorem statement
theorem total_handshakes_calculation :
  total_handshakes = 105 :=
by
  sorry

end NUMINAMATH_GPT_total_handshakes_calculation_l500_50011


namespace NUMINAMATH_GPT_race_victory_l500_50057

variable (distance : ℕ := 200)
variable (timeA : ℕ := 18)
variable (timeA_beats_B_by : ℕ := 7)

theorem race_victory : ∃ meters_beats_B : ℕ, meters_beats_B = 56 :=
by
  let speedA := distance / timeA
  let timeB := timeA + timeA_beats_B_by
  let speedB := distance / timeB
  let distanceB := speedB * timeA
  let meters_beats_B := distance - distanceB
  use meters_beats_B
  sorry

end NUMINAMATH_GPT_race_victory_l500_50057


namespace NUMINAMATH_GPT_number_properties_l500_50032

theorem number_properties : 
    ∃ (N : ℕ), 
    35 < N ∧ N < 70 ∧ N % 6 = 3 ∧ N % 8 = 1 ∧ N = 57 :=
by 
  sorry

end NUMINAMATH_GPT_number_properties_l500_50032


namespace NUMINAMATH_GPT_smallest_N_for_equal_adults_and_children_l500_50014

theorem smallest_N_for_equal_adults_and_children :
  ∃ (N : ℕ), N > 0 ∧ (∀ a b : ℕ, 8 * N = a ∧ 12 * N = b ∧ a = b) ∧ N = 3 :=
sorry

end NUMINAMATH_GPT_smallest_N_for_equal_adults_and_children_l500_50014


namespace NUMINAMATH_GPT_number_is_7612_l500_50003

-- Definitions of the conditions
def digits_correct_wrong_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10, 
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 ≠ (guess / 1000) % 10 ∧ 
      digits_placed 1 ≠ (guess / 100) % 10 ∧ 
      digits_placed 2 ≠ (guess / 10) % 10 ∧ 
      digits_placed 3 ≠ guess % 10)))

def digits_correct_positions (guess : Nat) (num_correct : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → 
      (num_correct = 2 ∧
      (digits_placed 0 = (guess / 1000) % 10 ∨ 
      digits_placed 1 = (guess / 100) % 10 ∨ 
      digits_placed 2 = (guess / 10) % 10 ∨ 
      digits_placed 3 = guess % 10)))

def digits_not_correct (guess : Nat) : Prop :=
  ∀ digits_placed : Fin 4 → Fin 10,
    ((digits_placed 0 = (guess / 1000) % 10 ∧ 
      digits_placed 1 = (guess / 100) % 10 ∧ 
      digits_placed 2 = (guess / 10) % 10 ∧ 
      digits_placed 3 = guess % 10) → False)

-- The main theorem to prove
theorem number_is_7612 :
  digits_correct_wrong_positions 8765 2 ∧
  digits_correct_wrong_positions 1023 2 ∧
  digits_correct_positions 8642 2 ∧
  digits_not_correct 5430 →
  ∃ (num : Nat), 
    (num / 1000) % 10 = 7 ∧
    (num / 100) % 10 = 6 ∧
    (num / 10) % 10 = 1 ∧
    num % 10 = 2 ∧
    num = 7612 :=
sorry

end NUMINAMATH_GPT_number_is_7612_l500_50003


namespace NUMINAMATH_GPT_industrial_lubricants_percentage_l500_50037

noncomputable def percentage_microphotonics : ℕ := 14
noncomputable def percentage_home_electronics : ℕ := 19
noncomputable def percentage_food_additives : ℕ := 10
noncomputable def percentage_gmo : ℕ := 24
noncomputable def total_percentage : ℕ := 100
noncomputable def percentage_basic_astrophysics : ℕ := 25

theorem industrial_lubricants_percentage :
  total_percentage - (percentage_microphotonics + percentage_home_electronics + 
  percentage_food_additives + percentage_gmo + percentage_basic_astrophysics) = 8 := 
sorry

end NUMINAMATH_GPT_industrial_lubricants_percentage_l500_50037


namespace NUMINAMATH_GPT_one_third_percent_of_200_l500_50074

theorem one_third_percent_of_200 : ((1206 / 3) / 200) * 100 = 201 := by
  sorry

end NUMINAMATH_GPT_one_third_percent_of_200_l500_50074


namespace NUMINAMATH_GPT_expression_equals_1390_l500_50080

theorem expression_equals_1390 :
  (25 + 15 + 8) ^ 2 - (25 ^ 2 + 15 ^ 2 + 8 ^ 2) = 1390 := 
by
  sorry

end NUMINAMATH_GPT_expression_equals_1390_l500_50080


namespace NUMINAMATH_GPT_households_accommodated_l500_50077

theorem households_accommodated (floors_per_building : ℕ)
                                (households_per_floor : ℕ)
                                (number_of_buildings : ℕ)
                                (total_households : ℕ)
                                (h1 : floors_per_building = 16)
                                (h2 : households_per_floor = 12)
                                (h3 : number_of_buildings = 10)
                                : total_households = 1920 :=
by
  sorry

end NUMINAMATH_GPT_households_accommodated_l500_50077


namespace NUMINAMATH_GPT_train_length_l500_50008

theorem train_length (L : ℕ) (speed : ℕ) 
  (h1 : L + 1200 = speed * 45) 
  (h2 : L + 180 = speed * 15) : 
  L = 330 := 
sorry

end NUMINAMATH_GPT_train_length_l500_50008


namespace NUMINAMATH_GPT_largest_divisor_poly_l500_50043

-- Define the polynomial and the required properties
def poly (n : ℕ) : ℕ := (n+1) * (n+3) * (n+5) * (n+7) * (n+11)

-- Define the conditions and the proof statement
theorem largest_divisor_poly (n : ℕ) (h_even : n % 2 = 0) : ∃ d, d = 15 ∧ ∀ m, m ∣ poly n → m ≤ d :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_poly_l500_50043


namespace NUMINAMATH_GPT_change_received_is_zero_l500_50078

noncomputable def combined_money : ℝ := 10 + 8
noncomputable def cost_chicken_wings : ℝ := 6
noncomputable def cost_chicken_salad : ℝ := 4
noncomputable def cost_cheeseburgers : ℝ := 2 * 3.50
noncomputable def cost_fries : ℝ := 2
noncomputable def cost_sodas : ℝ := 2 * 1.00
noncomputable def total_cost_before_discount : ℝ := cost_chicken_wings + cost_chicken_salad + cost_cheeseburgers + cost_fries + cost_sodas
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08
noncomputable def discounted_total : ℝ := total_cost_before_discount * (1 - discount_rate)
noncomputable def tax_amount : ℝ := discounted_total * tax_rate
noncomputable def total_cost_after_tax : ℝ := discounted_total + tax_amount

theorem change_received_is_zero : combined_money < total_cost_after_tax → 0 = combined_money - total_cost_after_tax + combined_money := by
  intros h
  sorry

end NUMINAMATH_GPT_change_received_is_zero_l500_50078


namespace NUMINAMATH_GPT_students_in_zack_classroom_l500_50092

theorem students_in_zack_classroom 
(T M Z : ℕ)
(h1 : T = M)
(h2 : Z = (T + M) / 2)
(h3 : T + M + Z = 69) :
Z = 23 :=
by
  sorry

end NUMINAMATH_GPT_students_in_zack_classroom_l500_50092


namespace NUMINAMATH_GPT_polynomial_factorization_l500_50055

-- Define the given polynomial expression
def given_poly (x : ℤ) : ℤ :=
  3 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2

-- Define the supposed factored form
def factored_poly (x : ℤ) : ℤ :=
  x * (3 * x^3 + 117 * x^2 + 1430 * x + 14895)

-- The theorem stating the equality of the two expressions
theorem polynomial_factorization (x : ℤ) : given_poly x = factored_poly x :=
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l500_50055


namespace NUMINAMATH_GPT_number_of_ways_two_girls_together_l500_50081

theorem number_of_ways_two_girls_together
  (boys girls : ℕ)
  (total_people : ℕ)
  (ways : ℕ) :
  boys = 3 →
  girls = 3 →
  total_people = boys + girls →
  ways = 432 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_ways_two_girls_together_l500_50081


namespace NUMINAMATH_GPT_geometric_series_sum_first_four_terms_l500_50002

theorem geometric_series_sum_first_four_terms :
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  (a * (1 - r^n) / (1 - r)) = 40 / 27 := by
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  sorry

end NUMINAMATH_GPT_geometric_series_sum_first_four_terms_l500_50002


namespace NUMINAMATH_GPT_geometric_sequence_correct_l500_50085

theorem geometric_sequence_correct (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 8)
  (h2 : a 2 * a 3 = -8)
  (h_geom : ∀ (n : ℕ), a (n + 1) = a n * r) :
  a 4 = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_correct_l500_50085


namespace NUMINAMATH_GPT_shaded_area_l500_50088

-- Definition of square side lengths
def side_lengths : List ℕ := [2, 4, 6, 8, 10]

-- Definition for the area of the largest square
def largest_square_area : ℕ := 10 * 10

-- Definition for the area of the smallest non-shaded square
def smallest_square_area : ℕ := 2 * 2

-- Total area of triangular regions
def triangular_area : ℕ := 2 * (2 * 4 + 2 * 6 + 2 * 8 + 2 * 10)

-- Question to prove
theorem shaded_area : largest_square_area - smallest_square_area - triangular_area = 40 := by
  sorry

end NUMINAMATH_GPT_shaded_area_l500_50088


namespace NUMINAMATH_GPT_liars_are_C_and_D_l500_50093
open Classical 

-- We define inhabitants and their statements
inductive Inhabitant
| A | B | C | D

open Inhabitant

axiom is_liar : Inhabitant → Prop

-- Statements by the inhabitants:
-- A: "At least one of us is a liar."
-- B: "At least two of us are liars."
-- C: "At least three of us are liars."
-- D: "None of us are liars."

def statement_A : Prop := is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D
def statement_B : Prop := (is_liar A ∧ is_liar B) ∨ (is_liar A ∧ is_liar C) ∨ (is_liar A ∧ is_liar D) ∨
                          (is_liar B ∧ is_liar C) ∨ (is_liar B ∧ is_liar D) ∨ (is_liar C ∧ is_liar D)
def statement_C : Prop := (is_liar A ∧ is_liar B ∧ is_liar C) ∨ (is_liar A ∧ is_liar B ∧ is_liar D) ∨
                          (is_liar A ∧ is_liar C ∧ is_liar D) ∨ (is_liar B ∧ is_liar C ∧ is_liar D)
def statement_D : Prop := ¬(is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D)

-- Given that there are some liars
axiom some_liars_exist : ∃ x, is_liar x

-- Lean proof statement
theorem liars_are_C_and_D : is_liar C ∧ is_liar D ∧ ¬(is_liar A) ∧ ¬(is_liar B) :=
by
  sorry

end NUMINAMATH_GPT_liars_are_C_and_D_l500_50093


namespace NUMINAMATH_GPT_quotient_is_eight_l500_50050

theorem quotient_is_eight (d v r q : ℕ) (h₁ : d = 141) (h₂ : v = 17) (h₃ : r = 5) (h₄ : d = v * q + r) : q = 8 :=
by
  sorry

end NUMINAMATH_GPT_quotient_is_eight_l500_50050


namespace NUMINAMATH_GPT_shaded_fraction_is_correct_l500_50071

-- Definitions based on the identified conditions
def initial_fraction_shaded : ℚ := 4 / 9
def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)
def infinite_series_fraction_shaded : ℚ := 4 / 9 * (4 / 3)

-- The theorem stating the problem
theorem shaded_fraction_is_correct :
  infinite_series_fraction_shaded = 16 / 27 :=
by
  sorry -- proof to be provided

end NUMINAMATH_GPT_shaded_fraction_is_correct_l500_50071


namespace NUMINAMATH_GPT_intersection_correct_l500_50073

def set_A : Set ℤ := {-1, 1, 2, 4}
def set_B : Set ℤ := {x | |x - 1| ≤ 1}

theorem intersection_correct :
  set_A ∩ set_B = {1, 2} :=
  sorry

end NUMINAMATH_GPT_intersection_correct_l500_50073


namespace NUMINAMATH_GPT_find_k_l500_50033

noncomputable def f (k : ℤ) (x : ℝ) := (k^2 + k - 1) * x^(k^2 - 3 * k)

-- The conditions in the problem
variables (k : ℤ) (x : ℝ)
axiom sym_y_axis : ∀ (x : ℝ), f k (-x) = f k x
axiom decreasing_on_positive : ∀ x1 x2, 0 < x1 → x1 < x2 → f k x1 > f k x2

-- The proof problem statement
theorem find_k : k = 1 :=
sorry

end NUMINAMATH_GPT_find_k_l500_50033


namespace NUMINAMATH_GPT_blue_tickets_per_red_ticket_l500_50099

-- Definitions based on conditions
def yellow_tickets_to_win_bible : Nat := 10
def red_tickets_per_yellow_ticket : Nat := 10
def blue_tickets_needed : Nat := 163
def additional_yellow_tickets_needed (current_yellow : Nat) : Nat := yellow_tickets_to_win_bible - current_yellow
def additional_red_tickets_needed (current_red : Nat) (needed_yellow : Nat) : Nat := needed_yellow * red_tickets_per_yellow_ticket - current_red

-- Given conditions
def current_yellow_tickets : Nat := 8
def current_red_tickets : Nat := 3
def current_blue_tickets : Nat := 7
def needed_yellow_tickets : Nat := additional_yellow_tickets_needed current_yellow_tickets
def needed_red_tickets : Nat := additional_red_tickets_needed current_red_tickets needed_yellow_tickets

-- Theorem to prove
theorem blue_tickets_per_red_ticket : blue_tickets_needed / needed_red_tickets = 10 :=
by
  sorry

end NUMINAMATH_GPT_blue_tickets_per_red_ticket_l500_50099


namespace NUMINAMATH_GPT_sum_of_center_coordinates_l500_50086

theorem sum_of_center_coordinates (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7) (h2 : y1 = -6) (h3 : x2 = -5) (h4 : y2 = 4) :
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  midpoint_x + midpoint_y = 0 := by
  -- Definitions and setup
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  sorry

end NUMINAMATH_GPT_sum_of_center_coordinates_l500_50086


namespace NUMINAMATH_GPT_partition_of_sum_l500_50026

-- Define the conditions
def is_positive_integer (n : ℕ) : Prop := n > 0
def is_bounded_integer (n : ℕ) : Prop := n ≤ 10
def can_be_partitioned (S : ℕ) (integers : List ℕ) : Prop :=
  ∃ (A B : List ℕ), 
    A.sum ≤ 70 ∧ 
    B.sum ≤ 70 ∧ 
    A ++ B = integers

-- Define the theorem statement
theorem partition_of_sum (S : ℕ) (integers : List ℕ)
  (h1 : ∀ x ∈ integers, is_positive_integer x ∧ is_bounded_integer x)
  (h2 : List.sum integers = S) :
  S ≤ 133 ↔ can_be_partitioned S integers :=
sorry

end NUMINAMATH_GPT_partition_of_sum_l500_50026


namespace NUMINAMATH_GPT_rajas_salary_percentage_less_than_rams_l500_50044

-- Definitions from the problem conditions
def raja_salary : ℚ := sorry -- Placeholder, since Raja's salary doesn't need a fixed value
def ram_salary : ℚ := 1.25 * raja_salary

-- Theorem to be proved
theorem rajas_salary_percentage_less_than_rams :
  ∃ r : ℚ, (ram_salary - raja_salary) / ram_salary * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_rajas_salary_percentage_less_than_rams_l500_50044


namespace NUMINAMATH_GPT_sum_of_ages_l500_50069

theorem sum_of_ages (a b c d : ℕ) (h1 : a * b = 20) (h2 : c * d = 28) (distinct : ∀ (x y : ℕ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) : a + b + c + d = 19 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l500_50069


namespace NUMINAMATH_GPT_intersection_x_coordinate_l500_50017

-- Definitions based on conditions
def line1 (x : ℝ) : ℝ := 3 * x + 5
def line2 (x : ℝ) : ℝ := 35 - 5 * x

-- Proof statement
theorem intersection_x_coordinate : ∃ x : ℝ, line1 x = line2 x ∧ x = 15 / 4 :=
by
  use 15 / 4
  sorry

end NUMINAMATH_GPT_intersection_x_coordinate_l500_50017


namespace NUMINAMATH_GPT_parabola_transformation_l500_50098

theorem parabola_transformation :
  (∀ x : ℝ, y = 2 * x^2 → y = 2 * (x-3)^2 - 1) := by
  sorry

end NUMINAMATH_GPT_parabola_transformation_l500_50098


namespace NUMINAMATH_GPT_contractor_daily_wage_l500_50058

theorem contractor_daily_wage (total_days : ℕ) (absent_days : ℕ) (fine_per_absent_day total_amount : ℝ) (daily_wage : ℝ)
  (h_total_days : total_days = 30)
  (h_absent_days : absent_days = 8)
  (h_fine : fine_per_absent_day = 7.50)
  (h_total_amount : total_amount = 490) 
  (h_work_days : total_days - absent_days = 22)
  (h_total_fined : fine_per_absent_day * absent_days = 60)
  (h_total_earned : 22 * daily_wage - 60 = 490) :
  daily_wage = 25 := 
by 
  sorry

end NUMINAMATH_GPT_contractor_daily_wage_l500_50058


namespace NUMINAMATH_GPT_at_least_two_equal_l500_50018

theorem at_least_two_equal (x y z : ℝ) (h : x / y + y / z + z / x = z / y + y / x + x / z) : 
  x = y ∨ y = z ∨ z = x := 
  sorry

end NUMINAMATH_GPT_at_least_two_equal_l500_50018


namespace NUMINAMATH_GPT_maria_needs_nuts_l500_50045

theorem maria_needs_nuts (total_cookies nuts_per_cookie : ℕ) 
  (nuts_fraction : ℚ) (chocolate_fraction : ℚ) 
  (H1 : nuts_fraction = 1 / 4) 
  (H2 : chocolate_fraction = 0.4) 
  (H3 : total_cookies = 60) 
  (H4 : nuts_per_cookie = 2) :
  (total_cookies * nuts_fraction + (total_cookies - total_cookies * nuts_fraction - total_cookies * chocolate_fraction) * nuts_per_cookie) = 72 := 
by
  sorry

end NUMINAMATH_GPT_maria_needs_nuts_l500_50045


namespace NUMINAMATH_GPT_expand_expression_l500_50056

theorem expand_expression : ∀ (x : ℝ), (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  intro x
  sorry

end NUMINAMATH_GPT_expand_expression_l500_50056


namespace NUMINAMATH_GPT_gcd_of_462_and_330_l500_50025

theorem gcd_of_462_and_330 :
  Nat.gcd 462 330 = 66 :=
sorry

end NUMINAMATH_GPT_gcd_of_462_and_330_l500_50025


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_ratio_l500_50075

-- Given a positive geometric sequence {a_n} with a_3, a_5, a_6 forming an arithmetic sequence,
-- we need to prove that (a_3 + a_5) / (a_4 + a_6) is among specific values {1, (sqrt 5 - 1) / 2}

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos: ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_arith : 2 * a 5 = a 3 + a 6) :
  (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequence_ratio_l500_50075


namespace NUMINAMATH_GPT_unique_toy_value_l500_50016

/-- Allie has 9 toys in total. The total worth of these toys is $52. 
One toy has a certain value "x" dollars and the remaining 8 toys each have a value of $5. 
Prove that the value of the unique toy is $12. -/
theorem unique_toy_value (x : ℕ) (h1 : 1 + 8 = 9) (h2 : x + 8 * 5 = 52) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_unique_toy_value_l500_50016


namespace NUMINAMATH_GPT_days_before_reinforcement_l500_50087

/-- A garrison of 2000 men originally has provisions for 62 days.
    After some days, a reinforcement of 2700 men arrives.
    The provisions are found to last for only 20 days more after the reinforcement arrives.
    Prove that the number of days passed before the reinforcement arrived is 15. -/
theorem days_before_reinforcement 
  (x : ℕ) 
  (num_men_orig : ℕ := 2000) 
  (num_men_reinf : ℕ := 2700) 
  (days_orig : ℕ := 62) 
  (days_after_reinf : ℕ := 20) 
  (total_provisions : ℕ := num_men_orig * days_orig)
  (remaining_provisions : ℕ := num_men_orig * (days_orig - x))
  (consumption_after_reinf : ℕ := (num_men_orig + num_men_reinf) * days_after_reinf) 
  (provisions_eq : remaining_provisions = consumption_after_reinf) : 
  x = 15 := 
by 
  sorry

end NUMINAMATH_GPT_days_before_reinforcement_l500_50087


namespace NUMINAMATH_GPT_crowdfunding_total_amount_l500_50038

theorem crowdfunding_total_amount
  (backers_highest_level : ℕ := 2)
  (backers_second_level : ℕ := 3)
  (backers_lowest_level : ℕ := 10)
  (amount_highest_level : ℝ := 5000) :
  ((backers_highest_level * amount_highest_level) + 
   (backers_second_level * (amount_highest_level / 10)) + 
   (backers_lowest_level * (amount_highest_level / 100))) = 12000 :=
by
  sorry

end NUMINAMATH_GPT_crowdfunding_total_amount_l500_50038


namespace NUMINAMATH_GPT_trapezium_distance_l500_50010

theorem trapezium_distance (a b h: ℝ) (area: ℝ) (h_area: area = 300) (h_sides: a = 22) (h_sides_2: b = 18)
  (h_formula: area = (1 / 2) * (a + b) * h): h = 15 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_distance_l500_50010


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l500_50039

-- Given conditions p and q
def p_geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def q_product_equality (a b c d : ℝ) : Prop :=
  a * d = b * c

-- Theorem statement: p implies q, but q does not imply p
theorem p_sufficient_not_necessary_for_q (a b c d : ℝ) :
  (p_geometric_sequence a b c d → q_product_equality a b c d) ∧
  (¬ (q_product_equality a b c d → p_geometric_sequence a b c d)) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l500_50039


namespace NUMINAMATH_GPT_min_large_buses_proof_l500_50019

def large_bus_capacity : ℕ := 45
def small_bus_capacity : ℕ := 30
def total_students : ℕ := 523
def min_small_buses : ℕ := 5

def min_large_buses_required (large_capacity small_capacity total small_buses : ℕ) : ℕ :=
  let remaining_students := total - (small_buses * small_capacity)
  let buses_needed := remaining_students / large_capacity
  if remaining_students % large_capacity = 0 then buses_needed else buses_needed + 1

theorem min_large_buses_proof :
  min_large_buses_required large_bus_capacity small_bus_capacity total_students min_small_buses = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_large_buses_proof_l500_50019


namespace NUMINAMATH_GPT_find_y_l500_50070

theorem find_y (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 4 * y + 2 = 0)
  (h2 : 3 * x + y + 4 = 0) :
  y^2 + 17 * y - 11 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_find_y_l500_50070


namespace NUMINAMATH_GPT_daily_sacks_per_section_l500_50035

theorem daily_sacks_per_section (harvests sections : ℕ) (h_harvests : harvests = 360) (h_sections : sections = 8) : harvests / sections = 45 := by
  sorry

end NUMINAMATH_GPT_daily_sacks_per_section_l500_50035


namespace NUMINAMATH_GPT_find_value_of_a_l500_50067

theorem find_value_of_a (a : ℝ) (h: (1 + 3 + 2 + 5 + a) / 5 = 3) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l500_50067


namespace NUMINAMATH_GPT_trig_identity_example_l500_50096

open Real -- Using the Real namespace for trigonometric functions

theorem trig_identity_example :
  sin (135 * π / 180) * cos (-15 * π / 180) + cos (225 * π / 180) * sin (15 * π / 180) = 1 / 2 :=
by 
  -- sorry to skip the proof steps
  sorry

end NUMINAMATH_GPT_trig_identity_example_l500_50096


namespace NUMINAMATH_GPT_largest_root_is_sqrt6_l500_50029

theorem largest_root_is_sqrt6 (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -6) 
  (h3 : p * q * r = -18) : 
  max p (max q r) = Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_largest_root_is_sqrt6_l500_50029


namespace NUMINAMATH_GPT_pow_div_eq_l500_50046

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end NUMINAMATH_GPT_pow_div_eq_l500_50046


namespace NUMINAMATH_GPT_balanced_apple_trees_l500_50062

theorem balanced_apple_trees: 
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (x1 * y2 - x1 * y4 - x3 * y2 + x3 * y4 = 0) ∧
    (x2 * y1 - x2 * y3 - x4 * y1 + x4 * y3 = 0) ∧
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4) :=
  sorry

end NUMINAMATH_GPT_balanced_apple_trees_l500_50062


namespace NUMINAMATH_GPT_susan_typing_time_l500_50068

theorem susan_typing_time :
  let Jonathan_rate := 1 -- page per minute
  let Jack_rate := 5 / 3 -- pages per minute
  let combined_rate := 4 -- pages per minute
  ∃ S : ℝ, (1 + 1/S + 5/3 = 4) → S = 30 :=
by
  sorry

end NUMINAMATH_GPT_susan_typing_time_l500_50068


namespace NUMINAMATH_GPT_smaller_angle_measure_l500_50001

theorem smaller_angle_measure (x : ℝ) (a b : ℝ) (h_suppl : a + b = 180) (h_ratio : a = 4 * x ∧ b = x) :
  b = 36 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_measure_l500_50001


namespace NUMINAMATH_GPT_cos_sin_value_l500_50090

theorem cos_sin_value (α : ℝ) (h : Real.tan α = Real.sqrt 2) : Real.cos α * Real.sin α = Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_cos_sin_value_l500_50090


namespace NUMINAMATH_GPT_simplify_expression_l500_50000

variables {x p q r : ℝ}

theorem simplify_expression (h1 : p ≠ q) (h2 : p ≠ r) (h3 : q ≠ r) :
   ( (x + p)^4 / ((p - q) * (p - r)) + (x + q)^4 / ((q - p) * (q - r)) + (x + r)^4 / ((r - p) * (r - q)) 
   ) = p + q + r + 4 * x :=
sorry

end NUMINAMATH_GPT_simplify_expression_l500_50000


namespace NUMINAMATH_GPT_number_of_programs_correct_l500_50064

-- Conditions definition
def solo_segments := 5
def chorus_segments := 3

noncomputable def number_of_programs : ℕ :=
  let solo_permutations := Nat.factorial solo_segments
  let available_spaces := solo_segments + 1
  let chorus_placements := Nat.choose (available_spaces - 1) chorus_segments
  solo_permutations * chorus_placements

theorem number_of_programs_correct : number_of_programs = 7200 :=
  by
    -- The proof is omitted
    sorry

end NUMINAMATH_GPT_number_of_programs_correct_l500_50064


namespace NUMINAMATH_GPT_pencil_length_after_sharpening_l500_50072

def initial_length : ℕ := 50
def monday_sharpen : ℕ := 2
def tuesday_sharpen : ℕ := 3
def wednesday_sharpen : ℕ := 4
def thursday_sharpen : ℕ := 5

def total_sharpened : ℕ := monday_sharpen + tuesday_sharpen + wednesday_sharpen + thursday_sharpen

def final_length : ℕ := initial_length - total_sharpened

theorem pencil_length_after_sharpening : final_length = 36 := by
  -- Here would be the proof body
  sorry

end NUMINAMATH_GPT_pencil_length_after_sharpening_l500_50072


namespace NUMINAMATH_GPT_range_of_reciprocal_sum_l500_50036

theorem range_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : a + b = 1) :
  ∃ c > 4, ∀ x, x = (1 / a + 1 / b) → c < x :=
sorry

end NUMINAMATH_GPT_range_of_reciprocal_sum_l500_50036


namespace NUMINAMATH_GPT_jars_of_peanut_butter_l500_50022

theorem jars_of_peanut_butter (x : Nat) : 
  (16 * x + 28 * x + 40 * x + 52 * x = 2032) → 
  (4 * x = 60) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_jars_of_peanut_butter_l500_50022


namespace NUMINAMATH_GPT_remainder_91_pow_91_mod_100_l500_50089

-- Definitions
def large_power_mod (a b n : ℕ) : ℕ :=
  (a^b) % n

-- Statement
theorem remainder_91_pow_91_mod_100 : large_power_mod 91 91 100 = 91 :=
by
  sorry

end NUMINAMATH_GPT_remainder_91_pow_91_mod_100_l500_50089


namespace NUMINAMATH_GPT_zoo_problem_l500_50012

theorem zoo_problem
  (num_zebras : ℕ)
  (num_camels : ℕ)
  (num_monkeys : ℕ)
  (num_giraffes : ℕ)
  (hz : num_zebras = 12)
  (hc : num_camels = num_zebras / 2)
  (hm : num_monkeys = 4 * num_camels)
  (hg : num_giraffes = 2) :
  num_monkeys - num_giraffes = 22 := by
  sorry

end NUMINAMATH_GPT_zoo_problem_l500_50012


namespace NUMINAMATH_GPT_average_after_11th_inning_is_30_l500_50066

-- Define the conditions as Lean 4 definitions
def score_in_11th_inning : ℕ := 80
def increase_in_avg : ℕ := 5
def innings_before_11th : ℕ := 10

-- Define the average before 11th inning
def average_before (x : ℕ) : ℕ := x

-- Define the total runs before 11th inning
def total_runs_before (x : ℕ) : ℕ := innings_before_11th * (average_before x)

-- Define the total runs after 11th inning
def total_runs_after (x : ℕ) : ℕ := total_runs_before x + score_in_11th_inning

-- Define the new average after 11th inning
def new_average_after (x : ℕ) : ℕ := total_runs_after x / (innings_before_11th + 1)

-- Theorem statement
theorem average_after_11th_inning_is_30 : 
  ∃ (x : ℕ), new_average_after x = average_before x + increase_in_avg → new_average_after 25 = 30 :=
by
  sorry

end NUMINAMATH_GPT_average_after_11th_inning_is_30_l500_50066
