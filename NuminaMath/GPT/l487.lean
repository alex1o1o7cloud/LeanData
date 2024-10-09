import Mathlib

namespace determinant_of_A_l487_48747

section
  open Matrix

  -- Define the given matrix
  def A : Matrix (Fin 3) (Fin 3) ℤ :=
    ![ ![0, 2, -4], ![6, -1, 3], ![2, -3, 5] ]

  -- State the theorem for the determinant
  theorem determinant_of_A : det A = 16 :=
  sorry
end

end determinant_of_A_l487_48747


namespace raisin_weight_l487_48792

theorem raisin_weight (Wg : ℝ) (dry_grapes_fraction : ℝ) (dry_raisins_fraction : ℝ) :
  Wg = 101.99999999999999 → dry_grapes_fraction = 0.10 → dry_raisins_fraction = 0.85 → 
  Wg * dry_grapes_fraction / dry_raisins_fraction = 12 := 
by
  intros h1 h2 h3
  sorry

end raisin_weight_l487_48792


namespace probability_all_calls_same_probability_two_calls_for_A_l487_48738

theorem probability_all_calls_same (pA pB pC : ℚ) (hA : pA = 1/6) (hB : pB = 1/3) (hC : pC = 1/2) :
  (pA^3 + pB^3 + pC^3) = 1/6 :=
by
  sorry

theorem probability_two_calls_for_A (pA : ℚ) (hA : pA = 1/6) :
  (3 * (pA^2) * (5/6)) = 5/72 :=
by
  sorry

end probability_all_calls_same_probability_two_calls_for_A_l487_48738


namespace number_of_solutions_l487_48707

theorem number_of_solutions :
  ∃ n : ℕ,  (1 + ⌊(102 * n : ℚ) / 103⌋ = ⌈(101 * n : ℚ) / 102⌉) ↔ (n < 10506) := 
sorry

end number_of_solutions_l487_48707


namespace Mitzi_score_l487_48784

-- Definitions based on the conditions
def Gretchen_score : ℕ := 120
def Beth_score : ℕ := 85
def average_score (total_score : ℕ) (num_bowlers : ℕ) : ℕ := total_score / num_bowlers

-- Theorem stating that Mitzi's bowling score is 113
theorem Mitzi_score (m : ℕ) (h : average_score (Gretchen_score + m + Beth_score) 3 = 106) :
  m = 113 :=
by sorry

end Mitzi_score_l487_48784


namespace candy_problem_l487_48700

theorem candy_problem
  (G : Nat := 7) -- Gwen got 7 pounds of candy
  (C : Nat := 17) -- Combined weight of candy
  (F : Nat) -- Pounds of candy Frank got
  (h : F + G = C) -- Condition: Combined weight
  : F = 10 := 
by
  sorry

end candy_problem_l487_48700


namespace line_passes_through_vertex_of_parabola_l487_48745

theorem line_passes_through_vertex_of_parabola : 
  ∃ (a : ℝ), (∀ x y : ℝ, y = 2 * x + a ↔ y = x^2 + a^2) ↔ a = 0 ∨ a = 1 := by
  sorry

end line_passes_through_vertex_of_parabola_l487_48745


namespace ratio_of_a_and_b_l487_48787

theorem ratio_of_a_and_b (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : (a * Real.sin (Real.pi / 7) + b * Real.cos (Real.pi / 7)) / 
        (a * Real.cos (Real.pi / 7) - b * Real.sin (Real.pi / 7)) = 
        Real.tan (10 * Real.pi / 21)) :
  b / a = Real.sqrt 3 :=
sorry

end ratio_of_a_and_b_l487_48787


namespace value_of_t_for_x_equals_y_l487_48741

theorem value_of_t_for_x_equals_y (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : 
    t = 1 / 2 → x = y :=
by 
  intro ht
  rw [ht] at h1 h2
  sorry

end value_of_t_for_x_equals_y_l487_48741


namespace car_A_overtakes_car_B_l487_48702

theorem car_A_overtakes_car_B (z : ℕ) :
  let y := (5 * z) / 4
  let x := (13 * z) / 10
  10 * y / (x - y) = 250 := 
by
  sorry

end car_A_overtakes_car_B_l487_48702


namespace jason_total_games_l487_48764

theorem jason_total_games :
  let jan_games := 11
  let feb_games := 17
  let mar_games := 16
  let apr_games := 20
  let may_games := 14
  let jun_games := 14
  let jul_games := 14
  jan_games + feb_games + mar_games + apr_games + may_games + jun_games + jul_games = 106 :=
by
  sorry

end jason_total_games_l487_48764


namespace primes_infinite_l487_48727

theorem primes_infinite : ∀ (S : Set ℕ), (∀ p, p ∈ S → Nat.Prime p) → (∃ a, a ∉ S ∧ Nat.Prime a) :=
by
  sorry

end primes_infinite_l487_48727


namespace minimum_additional_coins_l487_48790

-- The conditions
def total_friends : ℕ := 15
def current_coins : ℕ := 100

-- The fact that the total coins required to give each friend a unique number of coins from 1 to 15 is 120
def total_required_coins : ℕ := (total_friends * (total_friends + 1)) / 2

-- The theorem stating the required number of additional coins
theorem minimum_additional_coins (total_friends : ℕ) (current_coins : ℕ) (total_required_coins : ℕ) : ℕ :=
  sorry

end minimum_additional_coins_l487_48790


namespace john_ate_10_chips_l487_48718

variable (c p : ℕ)

/-- Given the total calories from potato chips and the calories increment of cheezits,
prove the number of potato chips John ate. -/
theorem john_ate_10_chips (h₀ : p * c = 60)
  (h₁ : ∃ c_cheezit, (c_cheezit = (4 / 3 : ℝ) * c))
  (h₂ : ∀ c_cheezit, p * c + 6 * c_cheezit = 108) :
  p = 10 :=
by {
  sorry
}

end john_ate_10_chips_l487_48718


namespace min_c_value_l487_48759

theorem min_c_value (c : ℝ) : (-c^2 + 9 * c - 14 >= 0) → (c >= 2) :=
by {
  sorry
}

end min_c_value_l487_48759


namespace simplify_expression_l487_48783

variable (a b : ℝ)

theorem simplify_expression :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by
  sorry

end simplify_expression_l487_48783


namespace approximate_pi_value_l487_48767

theorem approximate_pi_value (r h : ℝ) (L : ℝ) (V : ℝ) (π : ℝ) 
  (hL : L = 2 * π * r)
  (hV : V = 1 / 3 * π * r^2 * h) 
  (approxV : V = 2 / 75 * L^2 * h) :
  π = 25 / 8 := 
by
  -- Proof goes here
  sorry

end approximate_pi_value_l487_48767


namespace lines_intersect_l487_48712

-- Define the coefficients of the lines
def A1 : ℝ := 3
def B1 : ℝ := -2
def C1 : ℝ := 5

def A2 : ℝ := 1
def B2 : ℝ := 3
def C2 : ℝ := 10

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := A1 * x + B1 * y + C1 = 0
def line2 (x y : ℝ) : Prop := A2 * x + B2 * y + C2 = 0

-- Mathematical problem to prove
theorem lines_intersect : ∃ (x y : ℝ), line1 x y ∧ line2 x y :=
by
  sorry

end lines_intersect_l487_48712


namespace fraction_calculation_l487_48704

theorem fraction_calculation : 
  (1 / 4 + 1 / 6 - 1 / 2) / (-1 / 24) = 2 := 
by 
  sorry

end fraction_calculation_l487_48704


namespace remainder_when_x_squared_divided_by_20_l487_48725

theorem remainder_when_x_squared_divided_by_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] :=
sorry

end remainder_when_x_squared_divided_by_20_l487_48725


namespace find_r_l487_48713

theorem find_r (a b m p r : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a * b = 4)
  (h4 : ∀ x : ℚ, x^2 - m * x + 4 = (x - a) * (x - b)) :
  (a - 1 / b) * (b - 1 / a) = 9 / 4 := by
  sorry

end find_r_l487_48713


namespace sin_A_eq_one_half_l487_48773

theorem sin_A_eq_one_half (a b : ℝ) (sin_B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : sin_B = 2/3) : 
  ∃ (sin_A : ℝ), sin_A = 1/2 := 
by
  let sin_A := a * sin_B / b
  existsi sin_A
  sorry

end sin_A_eq_one_half_l487_48773


namespace excursion_min_parents_l487_48739

theorem excursion_min_parents 
  (students : ℕ) 
  (car_capacity : ℕ)
  (h_students : students = 30)
  (h_car_capacity : car_capacity = 5) 
  : ∃ (parents_needed : ℕ), parents_needed = 8 := 
by
  sorry -- proof goes here

end excursion_min_parents_l487_48739


namespace max_g_on_interval_l487_48779

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 :=
by
  sorry

end max_g_on_interval_l487_48779


namespace probability_even_product_l487_48710

-- Define spinner A and spinner C
def SpinnerA : List ℕ := [1, 2, 3, 4]
def SpinnerC : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define even and odd number sets for Spinner A and Spinner C
def evenNumbersA : List ℕ := [2, 4]
def oddNumbersA : List ℕ := [1, 3]

def evenNumbersC : List ℕ := [2, 4, 6]
def oddNumbersC : List ℕ := [1, 3, 5]

-- Define a function to check if a product is even
def isEven (n : ℕ) : Bool := n % 2 == 0

-- Probability calculation
def evenProductProbability : ℚ :=
  let totalOutcomes := (SpinnerA.length * SpinnerC.length)
  let evenA_outcomes := (evenNumbersA.length * SpinnerC.length)
  let oddA_evenC_outcomes := (oddNumbersA.length * evenNumbersC.length)
  (evenA_outcomes + oddA_evenC_outcomes) / totalOutcomes

theorem probability_even_product :
  evenProductProbability = 3 / 4 :=
by
  sorry

end probability_even_product_l487_48710


namespace find_alpha_l487_48732

theorem find_alpha
  (α : Real)
  (h1 : α > 0)
  (h2 : α < π)
  (h3 : 1 / Real.sin α + 1 / Real.cos α = 2) :
  α = π + 1 / 2 * Real.arcsin ((1 - Real.sqrt 5) / 2) :=
sorry

end find_alpha_l487_48732


namespace square_possible_n12_square_possible_n15_l487_48742

-- Define the nature of the problem with condition n = 12
def min_sticks_to_break_for_square_n12 : ℕ :=
  let n := 12
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Define the nature of the problem with condition n = 15
def min_sticks_to_break_for_square_n15 : ℕ :=
  let n := 15
  let total_length := (n * (n + 1)) / 2
  if total_length % 4 = 0 then 0 else 2

-- Statement of the problems in Lean 4 language
theorem square_possible_n12 : min_sticks_to_break_for_square_n12 = 2 := by
  sorry

theorem square_possible_n15 : min_sticks_to_break_for_square_n15 = 0 := by
  sorry

end square_possible_n12_square_possible_n15_l487_48742


namespace perfect_square_polynomial_l487_48714

theorem perfect_square_polynomial (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = m - 10 * x + x^2) → m = 25 :=
sorry

end perfect_square_polynomial_l487_48714


namespace john_bought_3_reels_l487_48729

theorem john_bought_3_reels (reel_length section_length : ℕ) (n_sections : ℕ)
  (h1 : reel_length = 100) (h2 : section_length = 10) (h3 : n_sections = 30) :
  n_sections * section_length / reel_length = 3 :=
by
  sorry

end john_bought_3_reels_l487_48729


namespace braids_each_dancer_l487_48734

-- Define the conditions
def num_dancers := 8
def time_per_braid := 30 -- seconds per braid
def total_time := 20 * 60 -- convert 20 minutes into seconds

-- Define the total number of braids Jill makes
def total_braids := total_time / time_per_braid

-- Define the number of braids per dancer
def braids_per_dancer := total_braids / num_dancers

-- Theorem: Prove that each dancer has 5 braids
theorem braids_each_dancer : braids_per_dancer = 5 := 
by sorry

end braids_each_dancer_l487_48734


namespace tracy_initial_candies_l487_48757

theorem tracy_initial_candies 
  (x : ℕ)
  (h1 : 4 ∣ x)
  (h2 : 5 ≤ ((x / 2) - 24))
  (h3 : ((x / 2) - 24) ≤ 9) 
  : x = 68 :=
sorry

end tracy_initial_candies_l487_48757


namespace find_abc_pairs_l487_48733

theorem find_abc_pairs :
  ∀ (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧ (a-1)*(b-1)*(c-1) ∣ a*b*c - 1 → 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  -- Proof omitted
  sorry

end find_abc_pairs_l487_48733


namespace find_C_and_D_l487_48715

noncomputable def C : ℚ := 51 / 10
noncomputable def D : ℚ := 29 / 10

theorem find_C_and_D (x : ℚ) (h1 : x^2 - 4*x - 21 = (x - 7)*(x + 3))
  (h2 : (8*x - 5) / ((x - 7)*(x + 3)) = C / (x - 7) + D / (x + 3)) :
  C = 51 / 10 ∧ D = 29 / 10 :=
by
  sorry

end find_C_and_D_l487_48715


namespace ab_divisible_by_6_l487_48769

theorem ab_divisible_by_6
  (n : ℕ) (a b : ℕ)
  (h1 : 2^n = 10 * a + b)
  (h2 : n > 3)
  (h3 : b < 10) :
  (a * b) % 6 = 0 :=
sorry

end ab_divisible_by_6_l487_48769


namespace area_DEFG_l487_48778

-- Define points and the properties of the rectangle ABCD
variable (A B C D E G F : Type)
variables (area_ABCD : ℝ) (Eg_parallel_AB_CD Df_parallel_AD_BC : Prop)
variable (E_position_AD : ℝ) (G_position_CD : ℝ) (F_midpoint_BC : Prop)
variables (length_abcd width_abcd : ℝ)

-- Assumptions based on given conditions
axiom h1 : area_ABCD = 150
axiom h2 : E_position_AD = 1 / 3
axiom h3 : G_position_CD = 1 / 3
axiom h4 : Eg_parallel_AB_CD
axiom h5 : Df_parallel_AD_BC
axiom h6 : F_midpoint_BC

-- Theorem to prove the area of DEFG
theorem area_DEFG : length_abcd * width_abcd / 3 = 50 :=
    sorry

end area_DEFG_l487_48778


namespace contrapositive_mul_non_zero_l487_48705

variables (a b : ℝ)

theorem contrapositive_mul_non_zero (h : a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :
  (a = 0 ∨ b = 0) → a * b = 0 :=
by
  sorry

end contrapositive_mul_non_zero_l487_48705


namespace Sue_button_count_l487_48711

variable (K S : ℕ)

theorem Sue_button_count (H1 : 64 = 5 * K + 4) (H2 : S = K / 2) : S = 6 := 
by
sorry

end Sue_button_count_l487_48711


namespace num_ways_128_as_sum_of_four_positive_perfect_squares_l487_48763

noncomputable def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, 0 < m ∧ m * m = n

noncomputable def four_positive_perfect_squares_sum (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    is_positive_perfect_square a ∧
    is_positive_perfect_square b ∧
    is_positive_perfect_square c ∧
    is_positive_perfect_square d ∧
    a + b + c + d = n

theorem num_ways_128_as_sum_of_four_positive_perfect_squares :
  (∃! (a b c d : ℕ), four_positive_perfect_squares_sum 128) :=
sorry

end num_ways_128_as_sum_of_four_positive_perfect_squares_l487_48763


namespace minimum_distinct_numbers_l487_48748

theorem minimum_distinct_numbers (a : ℕ → ℕ) (h_pos : ∀ i, 1 ≤ i → a i > 0)
  (h_distinct_ratios : ∀ i j : ℕ, 1 ≤ i ∧ i < 2006 ∧ 1 ≤ j ∧ j < 2006 ∧ i ≠ j → a i / a (i + 1) ≠ a j / a (j + 1)) :
  ∃ (n : ℕ), n = 46 ∧ ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 2006 ∧ 1 ≤ j ∧ j ≤ i ∧ (a i = a j → i = j) :=
sorry

end minimum_distinct_numbers_l487_48748


namespace sqrt_9_eq_pm3_l487_48799

theorem sqrt_9_eq_pm3 : ∃ x : ℤ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end sqrt_9_eq_pm3_l487_48799


namespace volume_of_blue_tetrahedron_in_cube_l487_48708

theorem volume_of_blue_tetrahedron_in_cube (side_length : ℝ) (h : side_length = 8) :
  let cube_volume := side_length^3
  let tetrahedra_volume := 4 * (1/3 * (1/2 * side_length * side_length) * side_length)
  cube_volume - tetrahedra_volume = 512/3 :=
by
  sorry

end volume_of_blue_tetrahedron_in_cube_l487_48708


namespace susan_more_cats_than_bob_after_transfer_l487_48706

-- Definitions and conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def cats_transferred : ℕ := 4

-- Question statement translated to Lean
theorem susan_more_cats_than_bob_after_transfer :
  (susan_initial_cats - cats_transferred) - bob_initial_cats = 14 :=
by
  sorry

end susan_more_cats_than_bob_after_transfer_l487_48706


namespace surface_area_of_modified_structure_l487_48703

-- Define the given conditions
def initial_cube_side_length : ℕ := 12
def smaller_cube_side_length : ℕ := 2
def smaller_cubes_count : ℕ := 72
def face_center_cubes_count : ℕ := 6

-- Define the calculation of the surface area
def single_smaller_cube_surface_area : ℕ := 6 * (smaller_cube_side_length ^ 2)
def added_surface_from_removed_center_cube : ℕ := 4 * (smaller_cube_side_length ^ 2)
def modified_smaller_cube_surface_area : ℕ := single_smaller_cube_surface_area + added_surface_from_removed_center_cube
def unaffected_smaller_cubes : ℕ := smaller_cubes_count - face_center_cubes_count

-- Define the given surface area according to the problem
def correct_surface_area : ℕ := 1824

-- The equivalent proof problem statement
theorem surface_area_of_modified_structure : 
    66 * single_smaller_cube_surface_area + 6 * modified_smaller_cube_surface_area = correct_surface_area := 
by
    -- placeholders for the actual proof
    sorry

end surface_area_of_modified_structure_l487_48703


namespace time_with_family_l487_48785

theorem time_with_family : 
    let hours_in_day := 24
    let sleep_fraction := 1 / 3
    let school_fraction := 1 / 6
    let assignment_fraction := 1 / 12
    let sleep_hours := sleep_fraction * hours_in_day
    let school_hours := school_fraction * hours_in_day
    let assignment_hours := assignment_fraction * hours_in_day
    let total_hours_occupied := sleep_hours + school_hours + assignment_hours
    hours_in_day - total_hours_occupied = 10 :=
by
  sorry

end time_with_family_l487_48785


namespace train_length_l487_48755

def relative_speed (v_fast v_slow : ℕ) : ℚ :=
  v_fast - v_slow

def convert_speed (speed : ℚ) : ℚ :=
  (speed * 1000) / 3600

def covered_distance (speed : ℚ) (time_seconds : ℚ) : ℚ :=
  speed * time_seconds

theorem train_length (L : ℚ) (v_fast v_slow : ℕ) (time_seconds : ℚ)
    (hf : v_fast = 42) (hs : v_slow = 36) (ht : time_seconds = 36)
    (hc : relative_speed v_fast v_slow * 1000 / 3600 * time_seconds = 2 * L) :
    L = 30 := by
  sorry

end train_length_l487_48755


namespace find_missing_number_l487_48771

theorem find_missing_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 :=
by
  sorry

end find_missing_number_l487_48771


namespace carlos_laundry_l487_48736

theorem carlos_laundry (n : ℕ) 
  (h1 : 45 * n + 75 = 165) : n = 2 :=
by
  sorry

end carlos_laundry_l487_48736


namespace problem1_problem2_l487_48758

theorem problem1 : 6 + (-8) - (-5) = 3 := sorry

theorem problem2 : 18 / (-3) + (-2) * (-4) = 2 := sorry

end problem1_problem2_l487_48758


namespace division_minutes_per_day_l487_48768

-- Define the conditions
def total_hours : ℕ := 5
def minutes_multiplication_per_day : ℕ := 10
def days_total : ℕ := 10

-- Convert hours to minutes
def total_minutes : ℕ := total_hours * 60

-- Total minutes spent on multiplication
def total_minutes_multiplication : ℕ := minutes_multiplication_per_day * days_total

-- Total minutes spent on division
def total_minutes_division : ℕ := total_minutes - total_minutes_multiplication

-- Minutes spent on division per day
def minutes_division_per_day : ℕ := total_minutes_division / days_total

-- The theorem to prove
theorem division_minutes_per_day : minutes_division_per_day = 20 := by
  sorry

end division_minutes_per_day_l487_48768


namespace equivalent_statements_l487_48724

variable (P Q : Prop)

theorem equivalent_statements :
  (P → Q) ↔ (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalent_statements_l487_48724


namespace circles_intersect_l487_48743

-- Definitions of the circles
def circle_O1 := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
def circle_O2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 9}

-- Proving the relationship between the circles
theorem circles_intersect : ∀ (p : ℝ × ℝ),
  p ∈ circle_O1 ∧ p ∈ circle_O2 :=
sorry

end circles_intersect_l487_48743


namespace min_people_same_score_l487_48735

theorem min_people_same_score (participants : ℕ) (nA nB : ℕ) (pointsA pointsB : ℕ) (scores : Finset ℕ) :
  participants = 400 →
  nA = 8 →
  nB = 6 →
  pointsA = 4 →
  pointsB = 7 →
  scores.card = (nA + 1) * (nB + 1) - 6 →
  participants / scores.card < 8 :=
by
  intros h_participants h_nA h_nB h_pointsA h_pointsB h_scores_card
  sorry

end min_people_same_score_l487_48735


namespace new_mean_rent_l487_48726

theorem new_mean_rent (avg_rent : ℕ) (num_friends : ℕ) (rent_increase_pct : ℕ) (initial_rent : ℕ) :
  avg_rent = 800 →
  num_friends = 4 →
  rent_increase_pct = 25 →
  initial_rent = 800 →
  (avg_rent * num_friends + initial_rent * rent_increase_pct / 100) / num_friends = 850 :=
by
  intros h_avg h_num h_pct h_init
  sorry

end new_mean_rent_l487_48726


namespace expected_value_decisive_games_l487_48721

/-- According to the rules of a chess match, the winner is the one who gains two victories over the opponent. -/
def winner_conditions (a b : Nat) : Prop :=
  a = 2 ∨ b = 2

/-- A game match where the probabilities of winning for the opponents are equal.-/
def probabilities_equal : Prop :=
  true

/-- Define X as the random variable representing the number of decisive games in the match. -/
def X (a b : Nat) : Nat :=
  a + b

/-- The expected value of the number of decisive games given equal probabilities of winning. -/
theorem expected_value_decisive_games (a b : Nat) (h1 : winner_conditions a b) (h2 : probabilities_equal) : 
  (X a b) / 2 = 4 :=
sorry

end expected_value_decisive_games_l487_48721


namespace total_score_is_938_l487_48793

-- Define the average score condition
def average_score (S : ℤ) : Prop := 85.25 ≤ (S : ℚ) / 11 ∧ (S : ℚ) / 11 < 85.35

-- Define the condition that each student's score is an integer
def total_score (S : ℤ) : Prop := average_score S ∧ ∃ n : ℕ, S = n

-- Lean 4 statement for the proof problem
theorem total_score_is_938 : ∃ S : ℤ, total_score S ∧ S = 938 :=
by
  sorry

end total_score_is_938_l487_48793


namespace b_plus_c_neg_seven_l487_48746

theorem b_plus_c_neg_seven {A B : Set ℝ} (hA : A = {x : ℝ | x > 3 ∨ x < -1}) (hB : B = {x : ℝ | -1 ≤ x ∧ x ≤ 4})
  (h_union : A ∪ B = Set.univ) (h_inter : A ∩ B = {x : ℝ | 3 < x ∧ x ≤ 4}) :
  ∃ b c : ℝ, (∀ x, x^2 + b * x + c ≤ 0 ↔ x ∈ B) ∧ b + c = -7 :=
by
  sorry

end b_plus_c_neg_seven_l487_48746


namespace fraction_product_eq_l487_48777

theorem fraction_product_eq :
  (1 / 3) * (3 / 5) * (5 / 7) * (7 / 9) = 1 / 9 := by
  sorry

end fraction_product_eq_l487_48777


namespace transformed_cube_edges_l487_48716

-- Let's define the problem statement
theorem transformed_cube_edges : 
  let original_edges := 12 
  let new_edges_per_edge := 2 
  let additional_edges_per_pyramid := 1 
  let total_edges := original_edges + (original_edges * new_edges_per_edge) + (original_edges * additional_edges_per_pyramid) 
  total_edges = 48 :=
by sorry

end transformed_cube_edges_l487_48716


namespace ellipse_with_given_foci_and_point_l487_48795

noncomputable def areFociEqual (a b c₁ c₂ : ℝ) : Prop :=
  c₁ = Real.sqrt (a^2 - b^2) ∧ c₂ = Real.sqrt (a^2 - b^2)

noncomputable def isPointOnEllipse (x₀ y₀ a₂ b₂ : ℝ) : Prop :=
  (x₀^2 / a₂) + (y₀^2 / b₂) = 1

theorem ellipse_with_given_foci_and_point :
  ∃a b : ℝ, 
    areFociEqual 8 3 a b ∧
    a = Real.sqrt 5 ∧ b = Real.sqrt 5 ∧
    isPointOnEllipse 3 (-2) 15 10  :=
sorry

end ellipse_with_given_foci_and_point_l487_48795


namespace min_formula_l487_48701

theorem min_formula (a b : ℝ) : 
  min a b = (a + b - Real.sqrt ((a - b) ^ 2)) / 2 :=
by
  sorry

end min_formula_l487_48701


namespace ships_meeting_count_l487_48776

theorem ships_meeting_count :
  ∀ (n : ℕ) (east_sailing west_sailing : ℕ),
    n = 10 →
    east_sailing = 5 →
    west_sailing = 5 →
    east_sailing + west_sailing = n →
    (∀ (v : ℕ), v > 0) →
    25 = east_sailing * west_sailing :=
by
  intros n east_sailing west_sailing h1 h2 h3 h4 h5
  sorry

end ships_meeting_count_l487_48776


namespace reassemble_black_rectangles_into_1x2_rectangle_l487_48752

theorem reassemble_black_rectangles_into_1x2_rectangle
  (x y : ℝ)
  (h1 : 0 < x ∧ x < 2)
  (h2 : 0 < y ∧ y < 2)
  (black_white_equal : 2*x*y - 2*x - 2*y + 2 = 0) :
  (x = 1 ∨ y = 1) →
  ∃ (z : ℝ), z = 1 :=
by
  sorry

end reassemble_black_rectangles_into_1x2_rectangle_l487_48752


namespace total_votes_l487_48719

theorem total_votes (votes_veggies : ℕ) (votes_meat : ℕ) (H1 : votes_veggies = 337) (H2 : votes_meat = 335) : votes_veggies + votes_meat = 672 :=
by
  sorry

end total_votes_l487_48719


namespace recurrence_relation_solution_l487_48740

theorem recurrence_relation_solution (a : ℕ → ℕ) 
  (h_rec : ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2))
  (h0 : a 0 = 3)
  (h1 : a 1 = 5) :
  ∀ n, a n = 3^n + 2 :=
by
  sorry

end recurrence_relation_solution_l487_48740


namespace mathematically_equivalent_proof_l487_48744

noncomputable def proof_problem (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ a^x = 2 ∧ a^y = 3 → a^(x - 2 * y) = 2 / 9

theorem mathematically_equivalent_proof (a : ℝ) (x y : ℝ) :
  proof_problem a x y :=
by
  sorry  -- Proof steps will go here

end mathematically_equivalent_proof_l487_48744


namespace opposite_neg_half_l487_48722

theorem opposite_neg_half : -(-1/2) = 1/2 :=
by
  sorry

end opposite_neg_half_l487_48722


namespace min_value_expression_l487_48717

theorem min_value_expression {x y z w : ℝ} 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) 
  (hw : 0 ≤ w ∧ w ≤ 1) : 
  ∃ m, m = 2 ∧ ∀ x y z w, (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) →
  m ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w))) :=
by
  sorry

end min_value_expression_l487_48717


namespace construction_costs_l487_48781

theorem construction_costs 
  (land_cost_per_sq_meter : ℕ := 50)
  (bricks_cost_per_1000 : ℕ := 100)
  (roof_tile_cost_per_tile : ℕ := 10)
  (land_area : ℕ := 2000)
  (number_of_bricks : ℕ := 10000)
  (number_of_roof_tiles : ℕ := 500) :
  50 * 2000 + (100 / 1000) * 10000 + 10 * 500 = 106000 :=
by sorry

end construction_costs_l487_48781


namespace Danny_bottle_caps_l487_48720

theorem Danny_bottle_caps (r w c : ℕ) (h1 : r = 11) (h2 : c = r + 1) : c = 12 := by
  sorry

end Danny_bottle_caps_l487_48720


namespace DE_value_l487_48730

theorem DE_value {AG GF FC HJ DE : ℝ} (h1 : AG = 2) (h2 : GF = 13) 
  (h3 : FC = 1) (h4 : HJ = 7) : DE = 2 * Real.sqrt 22 :=
sorry

end DE_value_l487_48730


namespace tangent_line_eqn_l487_48762

theorem tangent_line_eqn :
  ∃ k : ℝ, 
  x^2 + y^2 - 4*x + 3 = 0 → 
  (∃ x y : ℝ, (x-2)^2 + y^2 = 1 ∧ x > 2 ∧ y < 0 ∧ y = k*x) → 
  k = - (Real.sqrt 3) / 3 := 
by
  sorry

end tangent_line_eqn_l487_48762


namespace value_of_b_l487_48756

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 :=
sorry

end value_of_b_l487_48756


namespace smallest_value_for_x_9_l487_48782

theorem smallest_value_for_x_9 :
  let x := 9
  ∃ i, i = (8 / (x + 2)) ∧ 
  (i < (8 / x) ∧ 
   i < (8 / (x - 2)) ∧ 
   i < (x / 8) ∧ 
   i < ((x + 2) / 8)) :=
by
  let x := 9
  use (8 / (x + 2))
  sorry

end smallest_value_for_x_9_l487_48782


namespace chocolates_received_per_boy_l487_48772

theorem chocolates_received_per_boy (total_chocolates : ℕ) (total_people : ℕ)
(boys : ℕ) (girls : ℕ) (chocolates_per_girl : ℕ)
(h_total_chocolates : total_chocolates = 3000)
(h_total_people : total_people = 120)
(h_boys : boys = 60)
(h_girls : girls = 60)
(h_chocolates_per_girl : chocolates_per_girl = 3) :
  (total_chocolates - (girls * chocolates_per_girl)) / boys = 47 :=
by
  sorry

end chocolates_received_per_boy_l487_48772


namespace Ursula_hours_per_day_l487_48760

theorem Ursula_hours_per_day (hourly_wage : ℝ) (days_per_month : ℕ) (annual_salary : ℝ) (months_per_year : ℕ) :
  hourly_wage = 8.5 →
  days_per_month = 20 →
  annual_salary = 16320 →
  months_per_year = 12 →
  (annual_salary / months_per_year / days_per_month / hourly_wage) = 8 :=
by
  intros
  sorry

end Ursula_hours_per_day_l487_48760


namespace smaller_number_is_neg_five_l487_48709

theorem smaller_number_is_neg_five (x y : ℤ) (h1 : x + y = 30) (h2 : x - y = 40) : y = -5 :=
by
  sorry

end smaller_number_is_neg_five_l487_48709


namespace quilt_width_l487_48751

-- Definitions according to the conditions
def quilt_length : ℕ := 16
def patch_area : ℕ := 4
def first_10_patches_cost : ℕ := 100
def total_cost : ℕ := 450
def remaining_budget : ℕ := total_cost - first_10_patches_cost
def cost_per_additional_patch : ℕ := 5
def num_additional_patches : ℕ := remaining_budget / cost_per_additional_patch
def total_patches : ℕ := 10 + num_additional_patches
def total_area : ℕ := total_patches * patch_area

-- Theorem statement
theorem quilt_width :
  (total_area / quilt_length) = 20 :=
by
  sorry

end quilt_width_l487_48751


namespace sum_of_roots_l487_48791

   theorem sum_of_roots : 
     let a := 2
     let b := 7
     let c := 3
     let roots := (-b / a : ℝ)
     roots = -3.5 :=
   by
     sorry
   
end sum_of_roots_l487_48791


namespace range_of_a_l487_48765

open Set

variable {x a : ℝ}

def p (x a : ℝ) := x^2 + 2 * a * x - 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) := x^2 + 2 * x - 8 < 0

theorem range_of_a (h : ∀ x, p x a → q x): 0 < a ∧ a ≤ 4 / 3 := 
  sorry

end range_of_a_l487_48765


namespace distinct_x_intercepts_l487_48737

-- Given conditions
def polynomial (x : ℝ) : ℝ := (x - 4) * (x^2 + 4 * x + 13)

-- Statement of the problem as a Lean theorem
theorem distinct_x_intercepts : 
  (∃ (x : ℝ), polynomial x = 0 ∧ 
    ∀ (y : ℝ), y ≠ x → polynomial y = 0 → False) :=
  sorry

end distinct_x_intercepts_l487_48737


namespace minimize_y_l487_48775

noncomputable def y (x a b k : ℝ) : ℝ :=
  (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y (a b k : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b k ≤ y x' a b k) ∧ x = (a + b - k / 2) / 2 :=
by
  have x := (a + b - k / 2) / 2
  use x
  sorry

end minimize_y_l487_48775


namespace quadratic_inequality_solution_set_l487_48789

theorem quadratic_inequality_solution_set (m t : ℝ)
  (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - m*x + t < 0) : 
  m - t = -1 := sorry

end quadratic_inequality_solution_set_l487_48789


namespace solution_l487_48788

/-- Definition of the number with 2023 ones. -/
def x_2023 : ℕ := (10^2023 - 1) / 9

/-- Definition of the polynomial equation. -/
def polynomial_eq (x : ℕ) : ℤ :=
  567 * x^3 + 171 * x^2 + 15 * x - (7 * x + 5 * 10^2023 + 3 * 10^(2*2023))

/-- The solution x_2023 satisfies the polynomial equation. -/
theorem solution : polynomial_eq x_2023 = 0 := sorry

end solution_l487_48788


namespace complement_of_angleA_is_54_l487_48794

variable (A : ℝ)

-- Condition: \(\angle A = 36^\circ\)
def angleA := 36

-- Definition of complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Proof statement
theorem complement_of_angleA_is_54 (h : angleA = 36) : complement angleA = 54 :=
sorry

end complement_of_angleA_is_54_l487_48794


namespace proof_solution_l487_48753

noncomputable def proof_problem : Prop :=
  ∀ (x y z : ℝ), 3 * x - 4 * y - 2 * z = 0 ∧ x - 2 * y - 8 * z = 0 ∧ z ≠ 0 → 
  (x^2 + 3 * x * y) / (y^2 + z^2) = 329 / 61

theorem proof_solution : proof_problem :=
by
  intros x y z h
  sorry

end proof_solution_l487_48753


namespace geometric_sequence_division_condition_l487_48796

variable {a : ℕ → ℝ}
variable {q : ℝ}

/-- a is a geometric sequence with common ratio q -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = a 1 * q ^ (n - 1)

/-- 3a₁, 1/2a₅, and 2a₃ forming an arithmetic sequence -/
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * (a 1 * q ^ 2) = 2 * (1 / 2 * (a 1 * q ^ 4))

theorem geometric_sequence_division_condition
  (h1 : is_geometric_sequence a q)
  (h2 : arithmetic_sequence_condition a q) :
  (a 9 + a 10) / (a 7 + a 8) = 3 :=
sorry

end geometric_sequence_division_condition_l487_48796


namespace vertex_farthest_from_origin_l487_48728

theorem vertex_farthest_from_origin (center : ℝ × ℝ) (area : ℝ) (top_side_horizontal : Prop) (dilation_center : ℝ × ℝ) (scale_factor : ℝ) :
  center = (10, -5) ∧ area = 16 ∧ top_side_horizontal ∧ dilation_center = (0, 0) ∧ scale_factor = 3 →
  ∃ (vertex_farthest : ℝ × ℝ), vertex_farthest = (36, -21) :=
by
  sorry

end vertex_farthest_from_origin_l487_48728


namespace verify_squaring_method_l487_48798

theorem verify_squaring_method (x : ℝ) :
  ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 :=
by
  sorry

end verify_squaring_method_l487_48798


namespace base10_representation_of_n_l487_48780

theorem base10_representation_of_n (a b c n : ℕ) (ha : a > 0)
  (h14 : n = 14^2 * a + 14 * b + c)
  (h15 : n = 15^2 * a + 15 * c + b)
  (h6 : n = 6^3 * a + 6^2 * c + 6 * a + c) : n = 925 :=
by sorry

end base10_representation_of_n_l487_48780


namespace simplify_and_evaluate_l487_48797

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = 1/25) (h2 : y = -25) :
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = -3 :=
by
  sorry

end simplify_and_evaluate_l487_48797


namespace number_of_strawberry_cakes_l487_48761

def number_of_chocolate_cakes := 3
def price_of_chocolate_cake := 12
def price_of_strawberry_cake := 22
def total_payment := 168

theorem number_of_strawberry_cakes (S : ℕ) : 
    number_of_chocolate_cakes * price_of_chocolate_cake + S * price_of_strawberry_cake = total_payment → 
    S = 6 :=
by
  sorry

end number_of_strawberry_cakes_l487_48761


namespace evaluate_64_pow_3_div_2_l487_48731

theorem evaluate_64_pow_3_div_2 : (64 : ℝ)^(3/2) = 512 := by
  -- given 64 = 2^6
  have h : (64 : ℝ) = 2^6 := by norm_num
  -- use this substitution and properties of exponents
  rw [h, ←pow_mul]
  norm_num
  sorry -- completing the proof, not needed based on the guidelines

end evaluate_64_pow_3_div_2_l487_48731


namespace sum_of_other_endpoint_l487_48723

theorem sum_of_other_endpoint (x y : ℝ) (h1 : (6 + x) / 2 = 3) (h2 : (-2 + y) / 2 = 5) : x + y = 12 := 
by {
  sorry
}

end sum_of_other_endpoint_l487_48723


namespace pm_star_eq_6_l487_48766

open Set

-- Definitions based on the conditions
def universal_set : Set ℕ := univ
def M : Set ℕ := {1, 2, 3, 4, 5}
def P : Set ℕ := {2, 3, 6}
def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The theorem to prove
theorem pm_star_eq_6 : star P M = {6} :=
sorry

end pm_star_eq_6_l487_48766


namespace find_x_l487_48750

theorem find_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) : 
  x = (36 * Real.sqrt 5)^(4/11) := 
sorry

end find_x_l487_48750


namespace mean_rest_scores_l487_48786

theorem mean_rest_scores (n : ℕ) (h : 15 < n) 
  (overall_mean : ℝ := 10)
  (mean_of_fifteen : ℝ := 12)
  (total_score : ℝ := n * overall_mean): 
  (180 + p * (n - 15) = total_score) →
  p = (10 * n - 180) / (n - 15) :=
sorry

end mean_rest_scores_l487_48786


namespace problem_statement_l487_48770

-- Define C and D as specified in the problem conditions.
def C : ℕ := 4500
def D : ℕ := 3000

-- The final statement of the problem to prove C + D = 7500.
theorem problem_statement : C + D = 7500 := by
  -- This proof can be completed by checking arithmetic.
  sorry

end problem_statement_l487_48770


namespace games_within_division_l487_48774

variables (N M : ℕ)
  (h1 : N > 2 * M)
  (h2 : M > 4)
  (h3 : 3 * N + 4 * M = 76)

theorem games_within_division :
  3 * N = 48 :=
sorry

end games_within_division_l487_48774


namespace tan_sum_sin_cos_conditions_l487_48749

theorem tan_sum_sin_cos_conditions {x y : ℝ} 
  (h1 : Real.sin x + Real.sin y = 1 / 2) 
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = -Real.sqrt 3 := 
sorry

end tan_sum_sin_cos_conditions_l487_48749


namespace tan_beta_minus_2alpha_l487_48754

open Real

-- Given definitions
def condition1 (α : ℝ) : Prop :=
  (sin α * cos α) / (1 - cos (2 * α)) = 1 / 4

def condition2 (α β : ℝ) : Prop :=
  tan (α - β) = 2

-- Proof problem statement
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : condition1 α) (h2 : condition2 α β) :
  tan (β - 2 * α) = 4 / 3 :=
sorry

end tan_beta_minus_2alpha_l487_48754
