import Mathlib

namespace number_of_blind_students_l53_53751

variable (B D : ℕ)

-- Condition 1: The deaf-student population is 3 times the blind-student population.
axiom H1 : D = 3 * B

-- Condition 2: There are 180 students in total.
axiom H2 : B + D = 180

theorem number_of_blind_students : B = 45 :=
by
  -- Sorry is used to skip the proof steps. The theorem statement is correct and complete based on the conditions.
  sorry

end number_of_blind_students_l53_53751


namespace computation_of_expression_l53_53173

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l53_53173


namespace original_number_of_sides_l53_53808

theorem original_number_of_sides (sum_of_angles : ℕ) (H : (sum_of_angles = 2160)) : 
  ∃ x : ℕ, (2 * x - 2) * 180 = 2160 := 
by
  use 7
  have : (2 * 7 - 2) * 180 = 2160 := by sorry
  exact this

end original_number_of_sides_l53_53808


namespace igors_number_l53_53424

-- Define the initial lineup of players
def initialLineup : List ℕ := [9, 7, 11, 10, 6, 8, 5, 4, 1]

-- Define the condition for a player running to the locker room
def runsToLockRoom (n : ℕ) (left : Option ℕ) (right : Option ℕ) : Prop :=
  match left, right with
  | some l, some r => n < l ∨ n < r
  | some l, none   => n < l
  | none, some r   => n < r
  | none, none     => False

-- Define the process of players running to the locker room iteratively
def runProcess : List ℕ → List ℕ := 
  sorry   -- Implementation of the run process is skipped

-- Define the remaining players after repeated commands until 3 players are left
def remainingPlayers (lineup : List ℕ) : List ℕ :=
  sorry  -- Implementation to find the remaining players is skipped

-- Statement of the theorem
theorem igors_number (afterIgorRanOff : List ℕ := remainingPlayers initialLineup)
  (finalLineup : List ℕ := [9, 11, 10]) :
  ∃ n, n ∈ initialLineup ∧ ¬(n ∈ finalLineup) ∧ afterIgorRanOff.length = 3 → n = 5 :=
  sorry

end igors_number_l53_53424


namespace assignment_count_36_l53_53905

noncomputable def studentAssignments : Nat :=
  let choose_2_from_4 := Nat.choose 4 2
  let arrange_3_across_positions := Nat.factorial 3
  choose_2_from_4 * arrange_3_across_positions

theorem assignment_count_36 :
  ∃ n : Nat, n = studentAssignments :=
by
  let answer := 36
  sorry

end assignment_count_36_l53_53905


namespace average_side_length_of_squares_l53_53359

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53359


namespace solve_xy_l53_53689

theorem solve_xy (x y : ℝ) (hx: x ≠ 0) (hxy: x + y ≠ 0) : 
  (x + y) / x = 2 * y / (x + y) + 1 → (x = y ∨ x = -3 * y) := 
by 
  intros h 
  sorry

end solve_xy_l53_53689


namespace not_all_zero_iff_at_least_one_nonzero_l53_53406

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by 
  sorry

end not_all_zero_iff_at_least_one_nonzero_l53_53406


namespace average_side_lengths_l53_53372

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l53_53372


namespace valid_domain_of_x_l53_53404

theorem valid_domain_of_x (x : ℝ) : 
  (x + 1 ≥ 0 ∧ x ≠ 0) ↔ (x ≥ -1 ∧ x ≠ 0) :=
by sorry

end valid_domain_of_x_l53_53404


namespace regular_polygon_perimeter_is_28_l53_53075

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53075


namespace difference_in_x_coordinates_is_constant_l53_53198

variable {a x₀ y₀ k : ℝ}

-- Define the conditions
def point_on_x_axis (a : ℝ) : Prop := true

def passes_through_fixed_point_and_tangent (a : ℝ) : Prop :=
  a = 1

def equation_of_curve_C (x y : ℝ) : Prop :=
  y^2 = 4 * x

def tangent_condition (a x₀ y₀ : ℝ) (k : ℝ) : Prop :=
  a > 2 ∧ y₀ > 0 ∧ y₀^2 = 4 * x₀ ∧ 
  (4 * x₀ - 2 * y₀ * y₀ + y₀^2 = 0)

-- The statement
theorem difference_in_x_coordinates_is_constant (a x₀ y₀ k : ℝ) :
  point_on_x_axis a →
  passes_through_fixed_point_and_tangent a →
  equation_of_curve_C x₀ y₀ →
  tangent_condition a x₀ y₀ k → 
  a - x₀ = 2 :=
by
  intro h1 h2 h3 h4 
  sorry

end difference_in_x_coordinates_is_constant_l53_53198


namespace average_side_lengths_l53_53364

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l53_53364


namespace calculation_l53_53446

theorem calculation (a b c d e : ℤ)
  (h1 : a = (-4)^6)
  (h2 : b = 4^4)
  (h3 : c = 2^5)
  (h4 : d = 7^2)
  (h5 : e = (a / b) + c - d) :
  e = -1 := by
  sorry

end calculation_l53_53446


namespace difference_in_amount_paid_l53_53563

variable (P Q : ℝ)

def original_price := P
def intended_quantity := Q

def new_price := P * 1.10
def new_quantity := Q * 0.80

theorem difference_in_amount_paid :
  ((new_price P * new_quantity Q) - (original_price P * intended_quantity Q)) = -0.12 * (original_price P * intended_quantity Q) :=
by
  sorry

end difference_in_amount_paid_l53_53563


namespace houses_without_garage_nor_pool_l53_53843

def total_houses : ℕ := 85
def houses_with_garage : ℕ := 50
def houses_with_pool : ℕ := 40
def houses_with_both : ℕ := 35
def neither_garage_nor_pool : ℕ := 30

theorem houses_without_garage_nor_pool :
  total_houses - (houses_with_garage + houses_with_pool - houses_with_both) = neither_garage_nor_pool :=
by
  sorry

end houses_without_garage_nor_pool_l53_53843


namespace total_maple_trees_in_park_after_planting_l53_53872

def number_of_maple_trees_in_the_park (X_M : ℕ) (Y_M : ℕ) : ℕ := 
  X_M + Y_M

theorem total_maple_trees_in_park_after_planting : 
  number_of_maple_trees_in_the_park 2 9 = 11 := 
by 
  unfold number_of_maple_trees_in_the_park
  -- provide the mathematical proof here
  sorry

end total_maple_trees_in_park_after_planting_l53_53872


namespace sum_of_reciprocals_ineq_l53_53511

theorem sum_of_reciprocals_ineq (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a ^ 2 - 4 * a + 11)) + 
  (1 / (5 * b ^ 2 - 4 * b + 11)) + 
  (1 / (5 * c ^ 2 - 4 * c + 11)) ≤ 
  (1 / 4) := 
by {
  sorry
}

end sum_of_reciprocals_ineq_l53_53511


namespace average_of_side_lengths_of_squares_l53_53388

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l53_53388


namespace speed_of_current_l53_53429

-- Define the conditions in Lean
theorem speed_of_current (c : ℝ) (r : ℝ) 
  (hu : c - r = 12 / 6) -- upstream speed equation
  (hd : c + r = 12 / 0.75) -- downstream speed equation
  : r = 7 := 
sorry

end speed_of_current_l53_53429


namespace sum_of_fractions_l53_53961

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l53_53961


namespace original_price_of_wand_l53_53488

theorem original_price_of_wand (P : ℝ) (h1 : 8 = P / 8) : P = 64 :=
by sorry

end original_price_of_wand_l53_53488


namespace average_side_length_of_squares_l53_53381

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l53_53381


namespace auspicious_numbers_count_l53_53031

open Finset

theorem auspicious_numbers_count :
  ∑ s in ({0, 1, 2, 3, 4, 5}.powerset.filter (λ s, s.sum id = 8 ∧ s.card = 4)),
    (filter (λ n, n > 2015) (permutations_of_set s)).card = 23 :=
by
  sorry

end auspicious_numbers_count_l53_53031


namespace product_of_solutions_of_abs_equation_l53_53283

theorem product_of_solutions_of_abs_equation : 
  (∃ x1 x2 : ℝ, |5 * x1| + 2 = 47 ∧ |5 * x2| + 2 = 47 ∧ x1 ≠ x2 ∧ x1 * x2 = -81) :=
sorry

end product_of_solutions_of_abs_equation_l53_53283


namespace simplify_expression_l53_53624

theorem simplify_expression : 2 - Real.sqrt 3 + 1 / (2 - Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) = 6 :=
by
  sorry

end simplify_expression_l53_53624


namespace seeds_in_first_plot_l53_53636

theorem seeds_in_first_plot (x : ℕ) (h1 : 0 < x)
  (h2 : 200 = 200)
  (h3 : 0.25 * (x : ℝ) = 0.25 * (x : ℝ))
  (h4 : 0.35 * 200 = 70)
  (h5 : (0.25 * (x : ℝ) + 70) / (x + 200) = 0.29) :
  x = 300 :=
by sorry

end seeds_in_first_plot_l53_53636


namespace Cd_sum_l53_53671

theorem Cd_sum : ∀ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 3 → (C / (x-3) + D * (x+2) = (-2 * x^2 + 8 * x + 28) / (x-3))) → 
  (C + D = 20) :=
by
  intros C D h
  sorry

end Cd_sum_l53_53671


namespace cone_new_height_l53_53748

noncomputable def new_cone_height : ℝ := 6

theorem cone_new_height (r h V : ℝ) (circumference : 2 * Real.pi * r = 24 * Real.pi)
  (original_height : h = 40) (same_base_circumference : 2 * Real.pi * r = 24 * Real.pi)
  (volume : (1 / 3) * Real.pi * (r ^ 2) * new_cone_height = 288 * Real.pi) :
    new_cone_height = 6 := 
sorry

end cone_new_height_l53_53748


namespace person_next_to_Boris_arkady_galya_l53_53587

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l53_53587


namespace max_k_pos_l53_53541

-- Define the sequences {a_n} and {b_n}
def sequence_a (n k : ℤ) : ℤ := 2 * n + k - 1
def sequence_b (n : ℤ) : ℤ := 3 * n + 2

-- Conditions and given values
def S (n k : ℤ) : ℤ := n + k
def sum_first_9_b : ℤ := 153
def b_3 : ℤ := 11

-- Given the sequence {c_n}
def sequence_c (n k : ℤ) : ℤ := sequence_a n k - k * sequence_b n

-- Define the sum of the first n terms of the sequence {c_n}
def T (n k : ℤ) : ℤ := (n * (2 * sequence_c 1 k + (n - 1) * (2 - 3 * k))) / 2

-- Proof problem statement
theorem max_k_pos (k : ℤ) : (∀ n : ℤ, n > 0 → T n k > 0) → k ≤ 1 :=
sorry

end max_k_pos_l53_53541


namespace mod_inverse_sum_l53_53715

theorem mod_inverse_sum :
  ∃ a b : ℕ, (5 * a ≡ 1 [MOD 21]) ∧ (b = (a * a) % 21) ∧ ((a + b) % 21 = 9) :=
by
  sorry

end mod_inverse_sum_l53_53715


namespace decimal_to_fraction_l53_53231

theorem decimal_to_fraction : 2.36 = 59 / 25 :=
by
  sorry

end decimal_to_fraction_l53_53231


namespace min_balls_to_ensure_20_l53_53896

theorem min_balls_to_ensure_20 (red green yellow blue purple white black : ℕ) (Hred : red = 30) (Hgreen : green = 25) (Hyellow : yellow = 18) (Hblue : blue = 15) (Hpurple : purple = 12) (Hwhite : white = 10) (Hblack : black = 7) :
  ∀ n, n ≥ 101 → (∃ r g y b p w bl, r + g + y + b + p + w + bl = n ∧ (r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ p ≥ 20 ∨ w ≥ 20 ∨ bl ≥ 20)) :=
by
  intro n hn
  sorry

end min_balls_to_ensure_20_l53_53896


namespace min_value_x2_y2_l53_53496

theorem min_value_x2_y2 (x y : ℝ) (h : x^3 + y^3 + 3 * x * y = 1) : x^2 + y^2 ≥ 1 / 2 :=
by
  -- We are required to prove the minimum value of x^2 + y^2 given the condition is 1/2
  sorry

end min_value_x2_y2_l53_53496


namespace solve_system1_solve_system2_l53_53690

section System1

variables (x y : ℤ)

def system1_sol := x = 4 ∧ y = 8

theorem solve_system1 (h1 : y = 2 * x) (h2 : x + y = 12) : system1_sol x y :=
by 
  sorry

end System1

section System2

variables (x y : ℤ)

def system2_sol := x = 2 ∧ y = 3

theorem solve_system2 (h1 : 3 * x + 5 * y = 21) (h2 : 2 * x - 5 * y = -11) : system2_sol x y :=
by 
  sorry

end System2

end solve_system1_solve_system2_l53_53690


namespace gcd_ab_is_22_l53_53234

def a : ℕ := 198
def b : ℕ := 308

theorem gcd_ab_is_22 : Nat.gcd a b = 22 := 
by { sorry }

end gcd_ab_is_22_l53_53234


namespace no_quadruples_sum_2013_l53_53771

theorem no_quadruples_sum_2013 :
  ¬ ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c + d = 2013 ∧
  2013 % a = 0 ∧ 2013 % b = 0 ∧ 2013 % c = 0 ∧ 2013 % d = 0 :=
by
  sorry

end no_quadruples_sum_2013_l53_53771


namespace not_divisor_of_44_l53_53527

theorem not_divisor_of_44 (m j : ℤ) (H1 : m = j * (j + 1) * (j + 2) * (j + 3))
  (H2 : 11 ∣ m) : ¬ (∀ j : ℤ, 44 ∣ j * (j + 1) * (j + 2) * (j + 3)) :=
by
  sorry

end not_divisor_of_44_l53_53527


namespace expression_value_l53_53959

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l53_53959


namespace minimum_jumps_to_cover_circle_l53_53709

/--
Given 2016 points arranged in a circle and the ability to jump either 2 or 3 points clockwise,
prove that the minimum number of jumps required to visit every point at least once and return to the starting 
point is 2017.
-/
theorem minimum_jumps_to_cover_circle (n : Nat) (h : n = 2016) : 
  ∃ (a b : Nat), 2 * a + 3 * b = n ∧ (a + b) = 2017 := 
sorry

end minimum_jumps_to_cover_circle_l53_53709


namespace cone_volume_half_sector_rolled_l53_53033

theorem cone_volume_half_sector_rolled {r slant_height h V : ℝ}
  (radius_given : r = 3)
  (height_calculated : h = 3 * Real.sqrt 3)
  (slant_height_given : slant_height = 6)
  (arc_length : 2 * Real.pi * r = 6 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * (r^2) * h) :
  V = 9 * Real.pi * Real.sqrt 3 :=
by {
  sorry
}

end cone_volume_half_sector_rolled_l53_53033


namespace find_a_b_and_intervals_l53_53483

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x * Real.log x + a * x + b

theorem find_a_b_and_intervals (a b : ℝ) :
  (f 1 a b = 1 / 2) ∧ (f 0 a b = b) ∧ ((1 + Real.log x) > 0 ↔ x ∈ Set.Ioi (1 / Real.exp 1)) ↔
  (a = 0 ∧ b = 1/2) :=
  sorry

end find_a_b_and_intervals_l53_53483


namespace find_quotient_l53_53825

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![4, 5]]

noncomputable def matrix_b (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![c, d]]

theorem find_quotient (a b c d : ℝ) (H1 : matrix_a * (matrix_b a b c d) = (matrix_b a b c d) * matrix_a)
  (H2 : 2*b ≠ 3*c) : ((a - d) / (c - 2*b)) = 3 / 2 :=
  sorry

end find_quotient_l53_53825


namespace dorothy_and_jemma_sales_l53_53273

theorem dorothy_and_jemma_sales :
  ∀ (frames_sold_by_jemma price_per_frame_jemma : ℕ)
  (price_per_frame_dorothy frames_sold_by_dorothy : ℚ)
  (total_sales_jemma total_sales_dorothy total_sales : ℚ),
  price_per_frame_jemma = 5 →
  frames_sold_by_jemma = 400 →
  price_per_frame_dorothy = price_per_frame_jemma / 2 →
  frames_sold_by_jemma = 2 * frames_sold_by_dorothy →
  total_sales_jemma = frames_sold_by_jemma * price_per_frame_jemma →
  total_sales_dorothy = frames_sold_by_dorothy * price_per_frame_dorothy →
  total_sales = total_sales_jemma + total_sales_dorothy →
  total_sales = 2500 := by
  sorry

end dorothy_and_jemma_sales_l53_53273


namespace multiple_solutions_no_solution_2891_l53_53123

theorem multiple_solutions (n : ℤ) (x y : ℤ) (h1 : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ (u v : ℤ), u ≠ x ∧ v ≠ y ∧ u^3 - 3 * u * v^2 + v^3 = n :=
  sorry

theorem no_solution_2891 (x y : ℤ) (h2 : x^3 - 3 * x * y^2 + y^3 = 2891) :
  false :=
  sorry

end multiple_solutions_no_solution_2891_l53_53123


namespace average_side_length_of_squares_l53_53356

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53356


namespace ordered_pairs_count_l53_53310

theorem ordered_pairs_count : ∃ (count : ℕ), count = 4 ∧
  ∀ (m n : ℕ), m > 0 → n > 0 → m ≥ n → m^2 - n^2 = 144 → (∃ (i : ℕ), i < count) := by
  sorry

end ordered_pairs_count_l53_53310


namespace correct_exp_identity_l53_53884

variable (a b : ℝ)

theorem correct_exp_identity : ((a^2 * b)^3 / (-a * b)^2 = a^4 * b) := sorry

end correct_exp_identity_l53_53884


namespace interest_difference_correct_l53_53242

noncomputable def principal : ℝ := 1000
noncomputable def rate : ℝ := 0.10
noncomputable def time : ℝ := 4

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r)^t - P

noncomputable def interest_difference (P r t : ℝ) : ℝ := 
  compound_interest P r t - simple_interest P r t

theorem interest_difference_correct :
  interest_difference principal rate time = 64.10 :=
by
  sorry

end interest_difference_correct_l53_53242


namespace male_red_ants_percentage_l53_53726

noncomputable def percentage_of_total_ant_population_that_are_red_females (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    (percent_red_are_females / 100) * red_population_pct

noncomputable def percentage_of_total_ant_population_that_are_red_males (total_population_pct red_population_pct percent_red_are_females : ℝ) : ℝ :=
    red_population_pct - percentage_of_total_ant_population_that_are_red_females total_population_pct red_population_pct percent_red_are_females

theorem male_red_ants_percentage (total_population_pct red_population_pct percent_red_are_females male_red_ants_pct : ℝ) :
    red_population_pct = 85 → percent_red_are_females = 45 → male_red_ants_pct = 46.75 →
    percentage_of_total_ant_population_that_are_red_males total_population_pct red_population_pct percent_red_are_females = male_red_ants_pct :=
by
sorry

end male_red_ants_percentage_l53_53726


namespace problem_statement_l53_53984

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l53_53984


namespace math_problem_l53_53989

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l53_53989


namespace book_arrangements_l53_53311

theorem book_arrangements :
  let math_books := 4
  let english_books := 4
  let groups := 2
  (groups.factorial) * (math_books.factorial) * (english_books.factorial) = 1152 :=
by
  sorry

end book_arrangements_l53_53311


namespace TV_height_l53_53256

theorem TV_height (area : ℝ) (width : ℝ) (height : ℝ) (h1 : area = 21) (h2 : width = 3) : height = 7 :=
  by
  sorry

end TV_height_l53_53256


namespace sum_of_two_numbers_l53_53728

-- Define the two numbers and conditions
variables {x y : ℝ}
axiom prod_eq : x * y = 120
axiom sum_squares_eq : x^2 + y^2 = 289

-- The statement we want to prove
theorem sum_of_two_numbers (x y : ℝ) (prod_eq : x * y = 120) (sum_squares_eq : x^2 + y^2 = 289) : x + y = 23 :=
sorry

end sum_of_two_numbers_l53_53728


namespace simplify_fraction_120_1800_l53_53207

theorem simplify_fraction_120_1800 :
  (120 : ℚ) / 1800 = (1 : ℚ) / 15 := by
  sorry

end simplify_fraction_120_1800_l53_53207


namespace vector_dot_product_problem_l53_53642

theorem vector_dot_product_problem :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-1, 3)
  let C : ℝ × ℝ := (2, 1)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  let dot_prod := AB.1 * (2 * AC.1 + BC.1) + AB.2 * (2 * AC.2 + BC.2)
  dot_prod = -14 :=
by
  sorry

end vector_dot_product_problem_l53_53642


namespace negation_prop_equiv_l53_53401

variable (a : ℝ)

theorem negation_prop_equiv :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2 * a * x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2 * a * x - 1 ≥ 0) :=
sorry

end negation_prop_equiv_l53_53401


namespace probability_same_color_two_dice_l53_53312

theorem probability_same_color_two_dice :
  let total_sides : ℕ := 30
  let maroon_sides : ℕ := 5
  let teal_sides : ℕ := 10
  let cyan_sides : ℕ := 12
  let sparkly_sides : ℕ := 3
  (maroon_sides / total_sides)^2 + (teal_sides / total_sides)^2 + (cyan_sides / total_sides)^2 + (sparkly_sides / total_sides)^2 = 139 / 450 :=
by
  sorry

end probability_same_color_two_dice_l53_53312


namespace regular_polygon_perimeter_l53_53091

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53091


namespace wire_length_l53_53744

variables (L M S W : ℕ)

def ratio_condition (L M S : ℕ) : Prop :=
  L * 2 = 7 * S ∧ M * 2 = 3 * S

def total_length (L M S : ℕ) : ℕ :=
  L + M + S

theorem wire_length (h : ratio_condition L M 16) : total_length L M 16 = 96 :=
by sorry

end wire_length_l53_53744


namespace max_value_of_z_l53_53946

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y ≤ 2) (h2 : x + y ≥ 0) (h3 : x ≤ 4) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≤ 11 :=
by
  sorry

end max_value_of_z_l53_53946


namespace boris_neighbors_l53_53581

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l53_53581


namespace a7_equals_21_l53_53187

-- Define the sequence {a_n} recursively
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n + 2) => seq n + seq (n + 1)

-- Statement to prove that a_7 = 21
theorem a7_equals_21 : seq 6 = 21 := 
  sorry

end a7_equals_21_l53_53187


namespace math_problem_l53_53993

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l53_53993


namespace regular_polygon_perimeter_l53_53050

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53050


namespace regular_polygon_perimeter_l53_53102

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53102


namespace sum_product_smallest_number_l53_53223

theorem sum_product_smallest_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : min x y = 8 :=
  sorry

end sum_product_smallest_number_l53_53223


namespace reservoir_solution_l53_53253

theorem reservoir_solution (x y z : ℝ) :
  8 * (1 / x - 1 / y) = 1 →
  24 * (1 / x - 1 / y - 1 / z) = 1 →
  8 * (1 / y + 1 / z) = 1 →
  x = 8 ∧ y = 24 ∧ z = 12 :=
by
  intros h1 h2 h3
  sorry

end reservoir_solution_l53_53253


namespace standing_next_to_boris_l53_53607

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l53_53607


namespace decimal_to_fraction_l53_53230

theorem decimal_to_fraction : 2.36 = 59 / 25 :=
by
  sorry

end decimal_to_fraction_l53_53230


namespace tom_ratio_is_three_fourths_l53_53713

-- Define the years for the different programs
def bs_years : ℕ := 3
def phd_years : ℕ := 5
def tom_years : ℕ := 6
def normal_years : ℕ := bs_years + phd_years

-- Define the ratio of Tom's time to the normal time
def ratio : ℚ := tom_years / normal_years

theorem tom_ratio_is_three_fourths :
  ratio = 3 / 4 :=
by
  unfold ratio normal_years bs_years phd_years tom_years
  -- continued proof steps would go here
  sorry

end tom_ratio_is_three_fourths_l53_53713


namespace mary_earns_per_home_l53_53836

theorem mary_earns_per_home :
  let total_earned := 12696
  let homes_cleaned := 276.0
  total_earned / homes_cleaned = 46 :=
by
  sorry

end mary_earns_per_home_l53_53836


namespace max_n_value_l53_53290

theorem max_n_value (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
by
  sorry

end max_n_value_l53_53290


namespace sum_of_fractions_l53_53965

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l53_53965


namespace average_side_lengths_l53_53365

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l53_53365


namespace problem_l53_53009

variable (f g h : ℕ → ℕ)

-- Define the conditions as hypotheses
axiom h1 : ∀ (n m : ℕ), n ≠ m → h n ≠ h m
axiom h2 : ∀ y, ∃ x, g x = y
axiom h3 : ∀ n, f n = g n - h n + 1

theorem problem : ∀ n, f n = 1 := 
by 
  sorry

end problem_l53_53009


namespace point_on_angle_bisector_l53_53723

-- Define the properties and theorems relevant to the problem.

theorem point_on_angle_bisector {α : Type*} [euclidean_geometry α] 
  {A B : α} (P : α) (h : on_angle_bisector P A B) : 
  is_equidistant P A B :=
begin
  -- The proof here would show that any point on the bisector is equidistant from the two sides of the angle
  sorry
end

end point_on_angle_bisector_l53_53723


namespace length_of_second_train_l53_53543

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (clear_time_seconds : ℝ)
  (relative_speed_kmph : ℝ) :
  speed_first_train_kmph + speed_second_train_kmph = relative_speed_kmph →
  relative_speed_kmph * (5 / 18) * clear_time_seconds = length_first_train + 280 :=
by
  let length_first_train := 120
  let speed_first_train_kmph := 42
  let speed_second_train_kmph := 30
  let clear_time_seconds := 20
  let relative_speed_kmph := 72
  sorry

end length_of_second_train_l53_53543


namespace intersection_eq_l53_53303

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def M : Set ℝ := { x | -1/2 < x ∧ x < 1/2 }
def N : Set ℝ := { x | 0 ≤ x ∧ x * x ≤ x }

theorem intersection_eq :
  M ∩ N = { x | 0 ≤ x ∧ x < 1/2 } := by
  sorry

end intersection_eq_l53_53303


namespace weight_of_new_student_l53_53851

theorem weight_of_new_student (W x y z : ℝ) (h : (W - x - y + z = W - 40)) : z = 40 - (x + y) :=
by
  sorry

end weight_of_new_student_l53_53851


namespace perimeter_of_polygon_l53_53038

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53038


namespace perimeter_of_ABCD_is_35_2_l53_53550

-- Definitions of geometrical properties and distances
variable (AB BC DC : ℝ)
variable (AB_perp_BC : ∃P, is_perpendicular AB BC)
variable (DC_parallel_AB : ∃Q, is_parallel DC AB)
variable (AB_length : AB = 7)
variable (BC_length : BC = 10)
variable (DC_length : DC = 6)

-- Target statement to be proved
theorem perimeter_of_ABCD_is_35_2
  (h1 : AB_perp_BC)
  (h2 : DC_parallel_AB)
  (h3 : AB_length)
  (h4 : BC_length)
  (h5 : DC_length) :
  ∃ P : ℝ, P = 35.2 :=
sorry

end perimeter_of_ABCD_is_35_2_l53_53550


namespace max_green_beads_l53_53035

theorem max_green_beads (n : ℕ) (red blue green : ℕ) 
    (total_beads : ℕ)
    (h_total : total_beads = 100)
    (h_colors : n = red + blue + green)
    (h_blue_condition : ∀ i : ℕ, i ≤ total_beads → ∃ j, j ≤ 4 ∧ (i + j) % total_beads = blue)
    (h_red_condition : ∀ i : ℕ, i ≤ total_beads → ∃ k, k ≤ 6 ∧ (i + k) % total_beads = red) :
    green ≤ 65 :=
by
  sorry

end max_green_beads_l53_53035


namespace maximize_perimeter_l53_53252

theorem maximize_perimeter 
  (l : ℝ) (c_f : ℝ) (C : ℝ) (b : ℝ)
  (hl: l = 400) (hcf: c_f = 5) (hC: C = 1500) :
  ∃ (y : ℝ), y = 180 :=
by
  sorry

end maximize_perimeter_l53_53252


namespace original_proposition_true_converse_proposition_false_l53_53147

theorem original_proposition_true (a b : ℝ) : 
  a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := 
sorry

theorem converse_proposition_false : 
  ¬ (∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end original_proposition_true_converse_proposition_false_l53_53147


namespace tangent_line_value_l53_53947

theorem tangent_line_value {k : ℝ} 
  (h1 : ∃ x y : ℝ, x^2 + y^2 - 6*y + 8 = 0) 
  (h2 : ∃ P Q : ℝ, x^2 + y^2 - 6*y + 8 = 0 ∧ Q = k * P)
  (h3 : P * k < 0 ∧ P < 0 ∧ Q > 0) : 
  k = -2 * Real.sqrt 2 :=
sorry

end tangent_line_value_l53_53947


namespace simplified_expression_evaluate_at_zero_l53_53688

noncomputable def simplify_expr (x : ℝ) : ℝ :=
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2 * x + 1))

theorem simplified_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  simplify_expr x = 1 / (x - 1) :=
by sorry

theorem evaluate_at_zero (h₁ : (0 : ℝ) ≠ -1) (h₂ : (0 : ℝ) ≠ 1) : 
  simplify_expr 0 = -1 :=
by sorry

end simplified_expression_evaluate_at_zero_l53_53688


namespace find_slope_angle_l53_53464

theorem find_slope_angle (α : ℝ) :
    (∃ x y : ℝ, x * Real.sin (2 * Real.pi / 5) + y * Real.cos (2 * Real.pi / 5) = 0) →
    α = 3 * Real.pi / 5 :=
by
  intro h
  sorry

end find_slope_angle_l53_53464


namespace average_side_length_of_squares_l53_53362

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53362


namespace initial_production_rate_l53_53113

theorem initial_production_rate 
  (x : ℝ)
  (h1 : 60 <= (60 * x) / 30 - 60 + 1800)
  (h2 : 60 <= 120)
  (h3 : 30 = (120 / (60 / x + 1))) : x = 20 := by
  sorry

end initial_production_rate_l53_53113


namespace points_on_circle_l53_53131

theorem points_on_circle (t : ℝ) : 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  x^2 + y^2 = 1 := 
by 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  sorry

end points_on_circle_l53_53131


namespace evaluate_double_sum_l53_53277

theorem evaluate_double_sum :
  ∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m + 1) ^ 2 / (n + 1) / (m + n + 3) = 1 := by
  sorry

end evaluate_double_sum_l53_53277


namespace percentage_of_money_spent_is_80_l53_53822

-- Define the cost of items
def cheeseburger_cost : ℕ := 3
def milkshake_cost : ℕ := 5
def cheese_fries_cost : ℕ := 8

-- Define the amount of money Jim and his cousin brought
def jim_money : ℕ := 20
def cousin_money : ℕ := 10

-- Define the total cost of the meal
def total_cost : ℕ :=
  2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money they brought
def combined_money : ℕ := jim_money + cousin_money

-- Define the percentage of combined money spent
def percentage_spent : ℕ :=
  (total_cost * 100) / combined_money

theorem percentage_of_money_spent_is_80 :
  percentage_spent = 80 :=
by
  -- proof goes here
  sorry

end percentage_of_money_spent_is_80_l53_53822


namespace polygon_perimeter_l53_53094

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53094


namespace special_case_m_l53_53318

theorem special_case_m (m : ℝ) :
  (∀ x : ℝ, mx^2 - 4 * x + 3 = 0 → y = mx^2 - 4 * x + 3 → (x = 0 ∧ m = 0) ∨ (x ≠ 0 ∧ m = 4/3)) :=
sorry

end special_case_m_l53_53318


namespace second_candidate_extra_marks_l53_53734

theorem second_candidate_extra_marks (T : ℝ) (marks_40_percent : ℝ) (marks_passing : ℝ) (marks_60_percent : ℝ) 
  (h1 : marks_40_percent = 0.40 * T)
  (h2 : marks_passing = 160)
  (h3 : marks_60_percent = 0.60 * T)
  (h4 : marks_passing = marks_40_percent + 40) :
  (marks_60_percent - marks_passing) = 20 :=
by
  sorry

end second_candidate_extra_marks_l53_53734


namespace number_of_senior_citizen_tickets_l53_53021

theorem number_of_senior_citizen_tickets 
    (A S : ℕ)
    (h1 : A + S = 529)
    (h2 : 25 * A + 15 * S = 9745) 
    : S = 348 := 
by
  sorry

end number_of_senior_citizen_tickets_l53_53021


namespace trader_gain_percentage_l53_53439

structure PenType :=
  (pens_sold : ℕ)
  (cost_per_pen : ℕ)

def total_cost (pen : PenType) : ℕ :=
  pen.pens_sold * pen.cost_per_pen

def gain (pen : PenType) (multiplier : ℕ) : ℕ :=
  multiplier * pen.cost_per_pen

def weighted_average_gain_percentage (penA penB penC : PenType) (gainA gainB gainC : ℕ) : ℚ :=
  (((gainA + gainB + gainC):ℚ) / ((total_cost penA + total_cost penB + total_cost penC):ℚ)) * 100

theorem trader_gain_percentage :
  ∀ (penA penB penC : PenType)
  (gainA gainB gainC : ℕ),
  penA.pens_sold = 60 →
  penA.cost_per_pen = 2 →
  penB.pens_sold = 40 →
  penB.cost_per_pen = 3 →
  penC.pens_sold = 50 →
  penC.cost_per_pen = 4 →
  gainA = 20 * penA.cost_per_pen →
  gainB = 15 * penB.cost_per_pen →
  gainC = 10 * penC.cost_per_pen →
  weighted_average_gain_percentage penA penB penC gainA gainB gainC = 28.41 := 
by
  intros
  sorry

end trader_gain_percentage_l53_53439


namespace polygon_perimeter_l53_53096

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53096


namespace points_per_touchdown_l53_53697

theorem points_per_touchdown (total_points touchdowns : ℕ) (h1 : total_points = 21) (h2 : touchdowns = 3) :
  total_points / touchdowns = 7 :=
by
  sorry

end points_per_touchdown_l53_53697


namespace average_side_lengths_l53_53369

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l53_53369


namespace bug_paths_l53_53559

-- Define the problem conditions
structure PathSetup (A B : Type) :=
  (red_arrows : ℕ) -- number of red arrows from point A
  (red_to_blue : ℕ) -- number of blue arrows reachable from each red arrow
  (blue_to_green : ℕ) -- number of green arrows reachable from each blue arrow
  (green_to_orange : ℕ) -- number of orange arrows reachable from each green arrow
  (start_arrows : ℕ) -- starting number of arrows from point A to red arrows
  (orange_arrows : ℕ) -- number of orange arrows equivalent to green arrows

-- Define the conditions for our specific problem setup
def problem_setup : PathSetup Point Point :=
  {
    red_arrows := 3,
    red_to_blue := 2,
    blue_to_green := 2,
    green_to_orange := 1,
    start_arrows := 3,
    orange_arrows := 6 * 2 * 2 -- derived from blue_to_green and red_to_blue steps
  }

-- Prove the number of unique paths from A to B
theorem bug_paths (setup : PathSetup Point Point) : 
  setup.start_arrows * setup.red_to_blue * setup.blue_to_green * setup.green_to_orange * setup.orange_arrows = 1440 :=
by
  -- Calculations are performed; exact values must hold
  sorry

end bug_paths_l53_53559


namespace appeared_candidates_l53_53814

noncomputable def number_of_candidates_that_appeared_from_each_state (X : ℝ) : Prop :=
  (8 / 100) * X + 220 = (12 / 100) * X

theorem appeared_candidates (X : ℝ) (h : number_of_candidates_that_appeared_from_each_state X) : X = 5500 :=
  sorry

end appeared_candidates_l53_53814


namespace perimeter_of_regular_polygon_l53_53066

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53066


namespace bob_weight_l53_53506

noncomputable def jim_bob_equations (j b : ℝ) : Prop :=
  j + b = 200 ∧ b - 3 * j = b / 4

theorem bob_weight (j b : ℝ) (h : jim_bob_equations j b) : b = 171.43 :=
by
  sorry

end bob_weight_l53_53506


namespace correct_systematic_sampling_method_l53_53442

inductive SamplingMethod
| A
| B
| C
| D

def most_suitable_for_systematic_sampling (A B C D : SamplingMethod) : SamplingMethod :=
SamplingMethod.C

theorem correct_systematic_sampling_method : 
    most_suitable_for_systematic_sampling SamplingMethod.A SamplingMethod.B SamplingMethod.C SamplingMethod.D = SamplingMethod.C :=
by
  sorry

end correct_systematic_sampling_method_l53_53442


namespace option_C_correct_l53_53881

theorem option_C_correct (a b : ℝ) : ((a^2 * b)^3) / ((-a * b)^2) = a^4 * b := by
  sorry

end option_C_correct_l53_53881


namespace regular_polygon_perimeter_l53_53101

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53101


namespace distance_between_vertices_hyperbola_l53_53775

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l53_53775


namespace match_piles_l53_53710

theorem match_piles (a b c : ℕ) (h : a + b + c = 96)
    (h1 : 2 * b = a + c) (h2 : 2 * c = b + a) (h3 : 2 * a = c + b) : 
    a = 44 ∧ b = 28 ∧ c = 24 :=
  sorry

end match_piles_l53_53710


namespace butterfly_flutters_total_distance_l53_53246

-- Define the conditions
def start_pos : ℤ := 0
def first_move : ℤ := 4
def second_move : ℤ := -3
def third_move : ℤ := 7

-- Define a function that calculates the total distance
def total_distance (xs : List ℤ) : ℤ :=
  List.sum (List.map (fun ⟨x, y⟩ => abs (y - x)) (xs.zip xs.tail))

-- Create the butterfly's path
def path : List ℤ := [start_pos, first_move, second_move, third_move]

-- Define the proposition that we need to prove
theorem butterfly_flutters_total_distance : total_distance path = 21 := sorry

end butterfly_flutters_total_distance_l53_53246


namespace product_B_original_price_l53_53109

variable (a b : ℝ)

theorem product_B_original_price (h1 : a = 1.2 * b) (h2 : 0.9 * a = 198) : b = 183.33 :=
by
  sorry

end product_B_original_price_l53_53109


namespace final_value_l53_53967

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l53_53967


namespace total_kayaks_built_by_April_l53_53444

theorem total_kayaks_built_by_April
    (a : Nat := 9) (r : Nat := 3) (n : Nat := 4) :
    let S := a * (r ^ n - 1) / (r - 1)
    S = 360 := by
  sorry

end total_kayaks_built_by_April_l53_53444


namespace inner_hexagon_area_l53_53007

-- Define necessary conditions in Lean 4
variable (a b c d e f : ℕ)
variable (a1 a2 a3 a4 a5 a6 : ℕ)

-- Congruent equilateral triangles conditions forming a hexagon
axiom congruent_equilateral_triangles_overlap : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16

-- We want to show that the area of the inner hexagon is 38
theorem inner_hexagon_area : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16 → a = 38 :=
by
  intro h
  sorry

end inner_hexagon_area_l53_53007


namespace calc_expr_l53_53447

theorem calc_expr : 
  (1 / (1 + 1 / (2 + 1 / 5))) = 11 / 16 :=
by
  sorry

end calc_expr_l53_53447


namespace largest_pies_without_any_ingredients_l53_53189

-- Define the conditions
def total_pies : ℕ := 60
def pies_with_strawberries : ℕ := total_pies / 4
def pies_with_bananas : ℕ := total_pies * 3 / 8
def pies_with_cherries : ℕ := total_pies / 2
def pies_with_pecans : ℕ := total_pies / 10

-- State the theorem to prove
theorem largest_pies_without_any_ingredients : (total_pies - pies_with_cherries) = 30 := by
  sorry

end largest_pies_without_any_ingredients_l53_53189


namespace six_times_eightx_plus_tenpi_eq_fourP_l53_53803

variable {x : ℝ} {π P : ℝ}

theorem six_times_eightx_plus_tenpi_eq_fourP (h : 3 * (4 * x + 5 * π) = P) : 
    6 * (8 * x + 10 * π) = 4 * P :=
sorry

end six_times_eightx_plus_tenpi_eq_fourP_l53_53803


namespace which_is_lying_l53_53711

-- Ben's statement
def ben_says (dan_truth cam_truth : Bool) : Bool :=
  (dan_truth ∧ ¬ cam_truth) ∨ (¬ dan_truth ∧ cam_truth)

-- Dan's statement
def dan_says (ben_truth cam_truth : Bool) : Bool :=
  (ben_truth ∧ ¬ cam_truth) ∨ (¬ ben_truth ∧ cam_truth)

-- Cam's statement
def cam_says (ben_truth dan_truth : Bool) : Bool :=
  ¬ ben_truth ∧ ¬ dan_truth

-- Lean statement to be proven
theorem which_is_lying :
  (∃ (ben_truth dan_truth cam_truth : Bool), 
    ben_says dan_truth cam_truth ∧ 
    dan_says ben_truth cam_truth ∧ 
    cam_says ben_truth dan_truth ∧
    ¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) ↔ (¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) :=
sorry

end which_is_lying_l53_53711


namespace perimeter_of_polygon_l53_53044

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53044


namespace max_integer_value_of_f_l53_53928

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5)

theorem max_integer_value_of_f :
  ∃ n : ℤ, n = 17 ∧ ∀ x : ℝ, f x ≤ (n : ℝ) :=
by
  sorry

end max_integer_value_of_f_l53_53928


namespace numberOfWaysToChooseLeadership_is_correct_l53_53742

noncomputable def numberOfWaysToChooseLeadership (totalMembers : ℕ) : ℕ :=
  let choicesForGovernor := totalMembers
  let remainingAfterGovernor := totalMembers - 1

  let choicesForDeputies := Nat.choose remainingAfterGovernor 3
  let remainingAfterDeputies := remainingAfterGovernor - 3

  let choicesForLieutenants1 := Nat.choose remainingAfterDeputies 3
  let remainingAfterLieutenants1 := remainingAfterDeputies - 3

  let choicesForLieutenants2 := Nat.choose remainingAfterLieutenants1 3
  let remainingAfterLieutenants2 := remainingAfterLieutenants1 - 3

  let choicesForLieutenants3 := Nat.choose remainingAfterLieutenants2 3
  let remainingAfterLieutenants3 := remainingAfterLieutenants2 - 3

  let choicesForSubordinates : List ℕ := 
    (List.range 8).map (λ i => Nat.choose (remainingAfterLieutenants3 - 2*i) 2)

  choicesForGovernor 
  * choicesForDeputies 
  * choicesForLieutenants1 
  * choicesForLieutenants2 
  * choicesForLieutenants3 
  * List.prod choicesForSubordinates

theorem numberOfWaysToChooseLeadership_is_correct : 
  numberOfWaysToChooseLeadership 35 = 
    35 * Nat.choose 34 3 * Nat.choose 31 3 * Nat.choose 28 3 * Nat.choose 25 3 *
    Nat.choose 16 2 * Nat.choose 14 2 * Nat.choose 12 2 * Nat.choose 10 2 *
    Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2 :=
by
  sorry

end numberOfWaysToChooseLeadership_is_correct_l53_53742


namespace smallest_denominator_fraction_interval_exists_l53_53633

def interval (a b c d : ℕ) : Prop :=
a = 14 ∧ b = 73 ∧ c = 5 ∧ d = 26

theorem smallest_denominator_fraction_interval_exists :
  ∃ (a b c d : ℕ), 
    a / b < 19 / 99 ∧ b < 99 ∧
    19 / 99 < c / d ∧ d < 99 ∧
    interval a b c d :=
by
  sorry

end smallest_denominator_fraction_interval_exists_l53_53633


namespace find_original_number_l53_53878

theorem find_original_number (x : ℤ) (h : (x + 5) % 23 = 0) : x = 18 :=
sorry

end find_original_number_l53_53878


namespace sum_of_fractions_l53_53964

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l53_53964


namespace population_stable_at_K_l53_53551

-- Definitions based on conditions
def follows_S_curve (population : ℕ → ℝ) : Prop := sorry
def relatively_stable_at_K (population : ℕ → ℝ) (K : ℝ) : Prop := sorry
def ecological_factors_limit (population : ℕ → ℝ) : Prop := sorry

-- The main statement to be proved
theorem population_stable_at_K (population : ℕ → ℝ) (K : ℝ) :
  follows_S_curve population ∧ relatively_stable_at_K population K ∧ ecological_factors_limit population →
  relatively_stable_at_K population K :=
by sorry

end population_stable_at_K_l53_53551


namespace guise_hot_dogs_l53_53486

theorem guise_hot_dogs (x : ℤ) (h1 : x + (x + 2) + (x + 4) = 36) : x = 10 :=
by
  sorry

end guise_hot_dogs_l53_53486


namespace find_f_function_l53_53645

def oddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem find_f_function (f : ℝ → ℝ) (h_odd : oddFunction f) (h_pos : ∀ x, 0 < x → f x = x * (1 + x)) :
  ∀ x, x < 0 → f x = -x - x^2 :=
by
  sorry

end find_f_function_l53_53645


namespace reflect_point_example_l53_53533

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflect_over_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem reflect_point_example :
  reflect_over_x_axis ⟨2, 3, 4⟩ = ⟨2, -3, -4⟩ :=
by
  -- Proof can be filled in here
  sorry

end reflect_point_example_l53_53533


namespace relationship_between_P_and_Q_l53_53477

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem relationship_between_P_and_Q : 
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
sorry

end relationship_between_P_and_Q_l53_53477


namespace larger_triangle_perimeter_is_65_l53_53578

theorem larger_triangle_perimeter_is_65 (s1 s2 s3 t1 t2 t3 : ℝ)
  (h1 : s1 = 7) (h2 : s2 = 7) (h3 : s3 = 12)
  (h4 : t3 = 30)
  (similar : t1 / s1 = t2 / s2 ∧ t2 / s2 = t3 / s3) :
  t1 + t2 + t3 = 65 := by
  sorry

end larger_triangle_perimeter_is_65_l53_53578


namespace max_integer_value_of_k_l53_53801

theorem max_integer_value_of_k :
  ∀ x y k : ℤ,
    x - 4 * y = k - 1 →
    2 * x + y = k →
    x - y ≤ 0 →
    k ≤ 0 :=
by
  intros x y k h1 h2 h3
  sorry

end max_integer_value_of_k_l53_53801


namespace range_of_f_l53_53651

def f (x : ℤ) := x + 1

theorem range_of_f : 
  (∀ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x ∈ ({0, 1, 2, 3} : Set ℤ)) ∧ 
  (∀ y ∈ ({0, 1, 2, 3} : Set ℤ), ∃ x ∈ ({-1, 0, 1, 2} : Set ℤ), f x = y) := 
by 
  sorry

end range_of_f_l53_53651


namespace ratio_fraction_4A3B_5C2A_l53_53805

def ratio (a b c : ℝ) := a / b = 3 / 2 ∧ b / c = 2 / 6 ∧ a / c = 3 / 6

theorem ratio_fraction_4A3B_5C2A (A B C : ℝ) (h : ratio A B C) : (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := 
  sorry

end ratio_fraction_4A3B_5C2A_l53_53805


namespace jacket_purchase_price_l53_53249

theorem jacket_purchase_price (P S SP : ℝ)
  (h1 : S = P + 0.40 * S)
  (h2 : SP = 0.80 * S)
  (h3 : SP - P = 18) :
  P = 54 :=
by
  sorry

end jacket_purchase_price_l53_53249


namespace frog_eggs_ratio_l53_53741

theorem frog_eggs_ratio
    (first_day : ℕ)
    (second_day : ℕ)
    (third_day : ℕ)
    (total_eggs : ℕ)
    (h1 : first_day = 50)
    (h2 : second_day = first_day * 2)
    (h3 : third_day = second_day + 20)
    (h4 : total_eggs = 810) :
    (total_eggs - (first_day + second_day + third_day)) / (first_day + second_day + third_day) = 2 :=
by
    sorry

end frog_eggs_ratio_l53_53741


namespace max_k_value_l53_53479

theorem max_k_value (m : ℝ) (h : 0 < m ∧ m < 1/2) : 
  ∃ k : ℝ, (∀ m, 0 < m ∧ m < 1/2 → (1 / m + 2 / (1 - 2 * m)) ≥ k) ∧ k = 8 :=
by sorry

end max_k_value_l53_53479


namespace fraction_to_decimal_and_add_l53_53449

theorem fraction_to_decimal_and_add (a b : ℚ) (h : a = 7 / 16) : (a + b) = 2.4375 ↔ b = 2 :=
by
   sorry

end fraction_to_decimal_and_add_l53_53449


namespace average_side_length_of_squares_l53_53352

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53352


namespace bacteria_doubling_time_l53_53859

noncomputable def doubling_time_population 
    (initial final : ℝ) 
    (time : ℝ) 
    (growth_factor : ℕ) : ℝ :=
    time / (Real.log growth_factor / Real.log 2)

theorem bacteria_doubling_time :
  doubling_time_population 1000 500000 26.897352853986263 500 = 0.903 :=
by
  sorry

end bacteria_doubling_time_l53_53859


namespace perimeter_of_regular_polygon_l53_53058

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53058


namespace intersection_M_N_l53_53952

noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {x | abs x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l53_53952


namespace problem_statement_l53_53987

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l53_53987


namespace sandy_money_taken_l53_53847

-- Condition: Let T be the total money Sandy took for shopping, and it is known that 70% * T = $224
variable (T : ℝ)
axiom h : 0.70 * T = 224

-- Theorem to prove: T is 320
theorem sandy_money_taken : T = 320 :=
by 
  sorry

end sandy_money_taken_l53_53847


namespace find_like_term_l53_53111

-- Definition of the problem conditions
def monomials : List (String × String) := 
  [("A", "-2a^2b"), 
   ("B", "a^2b^2"), 
   ("C", "ab^2"), 
   ("D", "3ab")]

-- A function to check if two terms can be combined (like terms)
def like_terms(a b : String) : Prop :=
  a = "a^2b" ∧ b = "-2a^2b"

-- The theorem we need to prove
theorem find_like_term : ∃ x, x ∈ monomials ∧ like_terms "a^2b" (x.2) ∧ x.2 = "-2a^2b" :=
  sorry

end find_like_term_l53_53111


namespace real_solution_to_abs_equation_l53_53150

theorem real_solution_to_abs_equation :
  (∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| + |x - 8|) :=
by
  sorry

end real_solution_to_abs_equation_l53_53150


namespace ratio_of_Phil_to_Bob_l53_53491

-- There exists real numbers P, J, and B such that
theorem ratio_of_Phil_to_Bob (P J B : ℝ) (h1 : J = 2 * P) (h2 : B = 60) (h3 : J = B - 20) : P / B = 1 / 3 :=
by
  sorry

end ratio_of_Phil_to_Bob_l53_53491


namespace cost_difference_l53_53553

def cost_per_copy_X : ℝ := 1.25
def cost_per_copy_Y : ℝ := 2.75
def num_copies : ℕ := 80

theorem cost_difference :
  num_copies * cost_per_copy_Y - num_copies * cost_per_copy_X = 120 := sorry

end cost_difference_l53_53553


namespace solution_system_solution_rational_l53_53849

-- Definitions for the system of equations
def sys_eq_1 (x y : ℤ) : Prop := 2 * x - y = 3
def sys_eq_2 (x y : ℤ) : Prop := x + y = -12

-- Theorem to prove the solution of the system of equations
theorem solution_system (x y : ℤ) (h1 : sys_eq_1 x y) (h2 : sys_eq_2 x y) : x = -3 ∧ y = -9 :=
by {
  sorry
}

-- Definition for the rational equation
def rational_eq (x : ℤ) : Prop := (2 / (1 - x) : ℚ) + 1 = (x / (1 + x) : ℚ)

-- Theorem to prove the solution of the rational equation
theorem solution_rational (x : ℤ) (h : rational_eq x) : x = -3 :=
by {
  sorry
}

end solution_system_solution_rational_l53_53849


namespace arithmetic_sequence_a7_l53_53817

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) (h3 : a 3 = 3) (h5 : a 5 = -3) : a 7 = -9 := 
sorry

end arithmetic_sequence_a7_l53_53817


namespace range_of_m_for_common_point_l53_53807

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ :=
  -x^2 - 2 * x + m

-- Define the condition for a common point with the x-axis (i.e., it has real roots)
def has_common_point_with_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_function x m = 0

-- The theorem statement
theorem range_of_m_for_common_point : ∀ m : ℝ, has_common_point_with_x_axis m ↔ m ≥ -1 := 
sorry

end range_of_m_for_common_point_l53_53807


namespace percentage_of_bags_not_sold_l53_53190

theorem percentage_of_bags_not_sold
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_wednesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (h_initial : initial_stock = 600)
  (h_monday : sold_monday = 25)
  (h_tuesday : sold_tuesday = 70)
  (h_wednesday : sold_wednesday = 100)
  (h_thursday : sold_thursday = 110)
  (h_friday : sold_friday = 145) : 
  (initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday)) * 100 / initial_stock = 25 :=
by
  sorry

end percentage_of_bags_not_sold_l53_53190


namespace number_of_integers_satisfying_inequality_l53_53654

theorem number_of_integers_satisfying_inequality : 
  {n : Int | (n - 3) * (n + 5) < 0}.card = 7 :=
by
  sorry

end number_of_integers_satisfying_inequality_l53_53654


namespace john_pre_lunch_drive_l53_53188

def drive_before_lunch (h : ℕ) : Prop :=
  45 * h + 45 * 3 = 225

theorem john_pre_lunch_drive : ∃ h : ℕ, drive_before_lunch h ∧ h = 2 :=
by
  sorry

end john_pre_lunch_drive_l53_53188


namespace football_club_initial_balance_l53_53030

noncomputable def initial_balance (final_balance income expense : ℕ) : ℕ :=
  final_balance + income - expense

theorem football_club_initial_balance :
  initial_balance 60 (2 * 10) (4 * 15) = 20 := by
sorry

end football_club_initial_balance_l53_53030


namespace Stu_books_l53_53455

-- Define the number of books each person has
variables (E L S : ℕ)

-- Assumptions/conditions from the problem
def condition1 := E = 3 * L
def condition2 := L = 2 * S
def condition3 := E = 24

-- Theorem statement
theorem Stu_books : E = 24 → E = 3 * L → L = 2 * S → S = 4 :=
by { intros h1 h2 h3, sorry }

end Stu_books_l53_53455


namespace difference_of_sums_1500_l53_53877

def sum_of_first_n_odd_numbers (n : ℕ) : ℕ :=
  n * n

def sum_of_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_of_sums_1500 :
  sum_of_first_n_even_numbers 1500 - sum_of_first_n_odd_numbers 1500 = 1500 :=
by
  sorry

end difference_of_sums_1500_l53_53877


namespace suraj_avg_after_10th_inning_l53_53209

theorem suraj_avg_after_10th_inning (A : ℝ) 
  (h1 : ∀ A : ℝ, (9 * A + 200) / 10 = A + 8) :
  ∀ A : ℝ, A = 120 → (A + 8 = 128) :=
by
  sorry

end suraj_avg_after_10th_inning_l53_53209


namespace average_side_length_of_squares_l53_53355

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53355


namespace largest_integer_n_exists_l53_53781

theorem largest_integer_n_exists :
  ∃ (x y z n : ℤ), (x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 10 = n^2) ∧ (n = 6) :=
by
  sorry

end largest_integer_n_exists_l53_53781


namespace percentage_increase_of_gross_l53_53565

theorem percentage_increase_of_gross
  (P R : ℝ)
  (price_drop : ℝ := 0.20)
  (quantity_increase : ℝ := 0.60)
  (original_gross : ℝ := P * R)
  (new_price : ℝ := (1 - price_drop) * P)
  (new_quantity_sold : ℝ := (1 + quantity_increase) * R)
  (new_gross : ℝ := new_price * new_quantity_sold)
  (percentage_increase : ℝ := ((new_gross - original_gross) / original_gross) * 100) :
  percentage_increase = 28 :=
by
  sorry

end percentage_increase_of_gross_l53_53565


namespace solution_set_of_inequality_l53_53144

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem solution_set_of_inequality :
  {x : ℝ | f (2 * x + 1) + f (1) ≥ 0} = {x : ℝ | -1 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l53_53144


namespace antonella_purchase_l53_53114

theorem antonella_purchase
  (total_coins : ℕ)
  (coin_value : ℕ → ℕ)
  (num_toonies : ℕ)
  (initial_loonies : ℕ)
  (initial_toonies : ℕ)
  (total_value : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (H1 : total_coins = 10)
  (H2 : coin_value 1 = 1)
  (H3 : coin_value 2 = 2)
  (H4 : initial_toonies = 4)
  (H5 : initial_loonies = total_coins - initial_toonies)
  (H6 : total_value = initial_loonies * coin_value 1 + initial_toonies * coin_value 2)
  (H7 : amount_spent = 3)
  (H8 : amount_left = total_value - amount_spent)
  (H9 : amount_left = 11) :
  ∃ (used_loonies used_toonies : ℕ), used_loonies = 1 ∧ used_toonies = 1 ∧ (used_loonies * coin_value 1 + used_toonies * coin_value 2 = amount_spent) :=
by
  sorry

end antonella_purchase_l53_53114


namespace sandbox_width_l53_53331

theorem sandbox_width (P : ℕ) (W L : ℕ) (h1 : P = 30) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : W = 5 := 
sorry

end sandbox_width_l53_53331


namespace planting_schemes_l53_53662

/--
There are 6 plots of land. Each plot can be planted with either type A or type B vegetables.
The constraint is that no two adjacent plots can both have type A vegetables.
Prove that the total number of possible planting schemes is 21.
-/
theorem planting_schemes (P : Fin 6 → Bool) (h : ∀ i, i < 5 → (P i = true → P (i + 1) = false)) :
  (∃0 ≤ t ≤ 3, (Nat.choose (6 - t) t) = 21) :=
begin
  sorry
end

end planting_schemes_l53_53662


namespace perimeter_of_regular_polygon_l53_53067

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53067


namespace a_share_in_gain_l53_53576

noncomputable def investment_share (x: ℝ) (total_gain: ℝ): ℝ := 
  let a_interest := x * 0.1
  let b_interest := (2 * x) * (7 / 100) * (1.5)
  let c_interest := (3 * x) * (10 / 100) * (1.33)
  let total_interest := a_interest + b_interest + c_interest
  a_interest

theorem a_share_in_gain (total_gain: ℝ) (a_share: ℝ) (x: ℝ)
  (hx: 0.709 * x = total_gain):
  investment_share x total_gain = a_share :=
sorry

end a_share_in_gain_l53_53576


namespace new_concentration_is_37_percent_l53_53239

-- Conditions
def capacity_vessel_1 : ℝ := 2 -- litres
def alcohol_concentration_vessel_1 : ℝ := 0.35

def capacity_vessel_2 : ℝ := 6 -- litres
def alcohol_concentration_vessel_2 : ℝ := 0.50

def total_poured_liquid : ℝ := 8 -- litres
def final_vessel_capacity : ℝ := 10 -- litres

-- Question: Prove the new concentration of the mixture
theorem new_concentration_is_37_percent :
  (alcohol_concentration_vessel_1 * capacity_vessel_1 + alcohol_concentration_vessel_2 * capacity_vessel_2) / final_vessel_capacity = 0.37 := by
  sorry

end new_concentration_is_37_percent_l53_53239


namespace quadrant_of_P_l53_53478

theorem quadrant_of_P (m n : ℝ) (h1 : m * n > 0) (h2 : m + n < 0) : (m < 0 ∧ n < 0) :=
by
  sorry

end quadrant_of_P_l53_53478


namespace working_capacity_ratio_l53_53552

theorem working_capacity_ratio (team_p_engineers : ℕ) (team_q_engineers : ℕ) (team_p_days : ℕ) (team_q_days : ℕ) :
  team_p_engineers = 20 → team_q_engineers = 16 → team_p_days = 32 → team_q_days = 30 →
  (team_p_days / team_q_days) = (16:ℤ) / (15:ℤ) :=
by
  intros h1 h2 h3 h4
  sorry

end working_capacity_ratio_l53_53552


namespace average_of_side_lengths_of_squares_l53_53390

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l53_53390


namespace expression_value_l53_53995

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l53_53995


namespace expression_value_l53_53957

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l53_53957


namespace sqrt_meaningful_value_x_l53_53804

theorem sqrt_meaningful_value_x (x : ℝ) (h : x-1 ≥ 0) : x = 2 :=
by
  sorry

end sqrt_meaningful_value_x_l53_53804


namespace rectangle_length_from_square_thread_l53_53572

theorem rectangle_length_from_square_thread (side_of_square width_of_rectangle : ℝ) (same_thread : Bool) 
  (h1 : side_of_square = 20) (h2 : width_of_rectangle = 14) (h3 : same_thread) : 
  ∃ length_of_rectangle : ℝ, length_of_rectangle = 26 := 
by
  sorry

end rectangle_length_from_square_thread_l53_53572


namespace problem_i_l53_53028

theorem problem_i (n : ℕ) (h : n ≥ 1) : n ∣ 2^n - 1 ↔ n = 1 := by
  sorry

end problem_i_l53_53028


namespace sq_in_scientific_notation_l53_53944

theorem sq_in_scientific_notation (a : Real) (h : a = 25000) (h_scientific : a = 2.5 * 10^4) : a^2 = 6.25 * 10^8 :=
sorry

end sq_in_scientific_notation_l53_53944


namespace common_remainder_is_zero_l53_53512

noncomputable def least_number := 100040

theorem common_remainder_is_zero 
  (n : ℕ) 
  (h1 : n = least_number) 
  (condition1 : 4 ∣ n)
  (condition2 : 610 ∣ n)
  (condition3 : 15 ∣ n)
  (h2 : (n.digits 10).sum = 5)
  : ∃ r : ℕ, ∀ (a : ℕ), (a ∈ [4, 610, 15] → n % a = r) ∧ r = 0 :=
by {
  sorry
}

end common_remainder_is_zero_l53_53512


namespace infinite_series_sum_l53_53265

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 5) ^ (n + 1)) = 5 / 16 :=
sorry

end infinite_series_sum_l53_53265


namespace sum_of_products_l53_53763

theorem sum_of_products : 4 * 7 + 5 * 12 + 6 * 4 + 7 * 5 = 147 := by
  sorry

end sum_of_products_l53_53763


namespace train_pass_time_l53_53575

def train_length : ℕ := 250
def train_speed_kmph : ℕ := 36
def station_length : ℕ := 200

def total_distance : ℕ := train_length + station_length

noncomputable def train_speed_mps : ℚ := (train_speed_kmph : ℚ) * 1000 / 3600

noncomputable def time_to_pass_station : ℚ := total_distance / train_speed_mps

theorem train_pass_time : time_to_pass_station = 45 := by
  sorry

end train_pass_time_l53_53575


namespace isosceles_largest_angle_eq_60_l53_53185

theorem isosceles_largest_angle_eq_60 :
  ∀ (A B C : ℝ), (
    -- Condition: A triangle is isosceles with two equal angles of 60 degrees.
    ∀ (x y : ℝ), A = x ∧ B = x ∧ C = y ∧ x = 60 →
    -- Prove that
    max A (max B C) = 60 ) :=
by
  intros A B C h
  -- Sorry denotes skipping the proof.
  sorry

end isosceles_largest_angle_eq_60_l53_53185


namespace count_valid_triples_l53_53261

theorem count_valid_triples :
  ∃! (a c : ℕ), a ≤ 101 ∧ 101 ≤ c ∧ a * c = 101^2 :=
sorry

end count_valid_triples_l53_53261


namespace count_valid_integers_1_to_999_l53_53308

-- Define a function to count the valid integers
def count_valid_integers : Nat :=
  let digits := [1, 2, 6, 7, 9]
  let one_digit_count := 5
  let two_digit_count := 5 * 5
  let three_digit_count := 5 * 5 * 5
  one_digit_count + two_digit_count + three_digit_count

-- The theorem we want to prove
theorem count_valid_integers_1_to_999 : count_valid_integers = 155 := by
  sorry

end count_valid_integers_1_to_999_l53_53308


namespace compute_expression_l53_53164

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l53_53164


namespace regular_polygon_perimeter_l53_53048

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53048


namespace vann_teeth_cleaning_l53_53875

def numDogsCleaned (D : Nat) : Prop :=
  let dogTeethCount := 42
  let catTeethCount := 30
  let pigTeethCount := 28
  let numCats := 10
  let numPigs := 7
  let totalTeeth := 706
  dogTeethCount * D + catTeethCount * numCats + pigTeethCount * numPigs = totalTeeth

theorem vann_teeth_cleaning : numDogsCleaned 5 :=
by
  sorry

end vann_teeth_cleaning_l53_53875


namespace average_side_length_of_squares_l53_53376

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53376


namespace find_subtracted_value_l53_53416

theorem find_subtracted_value (N V : ℤ) (hN : N = 12) (h : 4 * N - 3 = 9 * (N - V)) : V = 7 := 
by
  sorry

end find_subtracted_value_l53_53416


namespace bike_ride_energetic_time_l53_53262

theorem bike_ride_energetic_time :
  ∃ x : ℚ, (22 * x + 15 * (7.5 - x) = 142) ∧ x = (59 / 14) :=
by
  sorry

end bike_ride_energetic_time_l53_53262


namespace algebraic_expression_identity_l53_53177

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l53_53177


namespace intersecting_lines_l53_53219

-- Definitions for the conditions
def line1 (x y a : ℝ) : Prop := x = (1/3) * y + a
def line2 (x y b : ℝ) : Prop := y = (1/3) * x + b

-- The theorem we need to prove
theorem intersecting_lines (a b : ℝ) (h1 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line1 x y a) 
                           (h2 : ∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ line2 x y b) : 
  a + b = 10 / 3 :=
sorry

end intersecting_lines_l53_53219


namespace final_temperature_l53_53687

theorem final_temperature (initial_temp cost_per_tree spent amount temperature_drop : ℝ) 
  (h1 : initial_temp = 80) 
  (h2 : cost_per_tree = 6)
  (h3 : spent = 108) 
  (h4 : temperature_drop = 0.1) 
  (trees_planted : ℝ) 
  (h5 : trees_planted = spent / cost_per_tree) 
  (temp_reduction : ℝ) 
  (h6 : temp_reduction = trees_planted * temperature_drop) 
  (final_temp : ℝ) 
  (h7 : final_temp = initial_temp - temp_reduction) : 
  final_temp = 78.2 := 
by
  sorry

end final_temperature_l53_53687


namespace rhombus_perimeter_l53_53217

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  (4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))) = 52 := by
  sorry

end rhombus_perimeter_l53_53217


namespace Bill_has_39_dollars_l53_53467

noncomputable def Frank_initial_money : ℕ := 42
noncomputable def pizza_cost : ℕ := 11
noncomputable def num_pizzas : ℕ := 3
noncomputable def Bill_initial_money : ℕ := 30

noncomputable def Frank_spent : ℕ := pizza_cost * num_pizzas
noncomputable def Frank_remaining_money : ℕ := Frank_initial_money - Frank_spent
noncomputable def Bill_final_money : ℕ := Bill_initial_money + Frank_remaining_money

theorem Bill_has_39_dollars :
  Bill_final_money = 39 :=
by
  sorry

end Bill_has_39_dollars_l53_53467


namespace boris_neighbors_l53_53582

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l53_53582


namespace factorial_expression_equals_l53_53118

theorem factorial_expression_equals :
  7 * Nat.factorial 7 + 5 * Nat.factorial 5 - 3 * Nat.factorial 3 + 2 * Nat.factorial 2 = 35866 := by
  sorry

end factorial_expression_equals_l53_53118


namespace incorrect_reasoning_C_l53_53721

theorem incorrect_reasoning_C
  {Point : Type} {Line Plane : Type}
  (A B : Point) (l : Line) (α β : Plane)
  (in_line : Point → Line → Prop)
  (in_plane : Point → Plane → Prop)
  (line_in_plane : Line → Plane → Prop)
  (disjoint : Line → Plane → Prop) :

  ¬(line_in_plane l α) ∧ in_line A l ∧ in_plane A α :=
sorry

end incorrect_reasoning_C_l53_53721


namespace largest_possible_value_for_a_l53_53510

theorem largest_possible_value_for_a (a b c d : ℕ) 
  (h1: a < 3 * b) 
  (h2: b < 2 * c + 1) 
  (h3: c < 5 * d - 2)
  (h4: d ≤ 50) 
  (h5: d % 5 = 0) : 
  a ≤ 1481 :=
sorry

end largest_possible_value_for_a_l53_53510


namespace heart_ratio_correct_l53_53656

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_correct : (heart 3 5 : ℚ) / (heart 5 3) = 26 / 67 :=
by
  sorry

end heart_ratio_correct_l53_53656


namespace cows_milk_production_l53_53528

variable (p q r s t : ℕ)

theorem cows_milk_production
  (h : p * r > 0)  -- Assuming p and r are positive to avoid division by zero
  (produce : p * r * q ≠ 0) -- Additional assumption to ensure non-zero q
  (h_cows : q = p * r * (q / (p * r))) 
  : s * t * q / (p * r) = s * t * (q / (p * r)) :=
by
  sorry

end cows_milk_production_l53_53528


namespace distinct_sums_count_l53_53288

theorem distinct_sums_count (n : ℕ) (a : Fin n.succ → ℕ) (h_distinct : Function.Injective a) :
  ∃ (S : Finset ℕ), S.card ≥ n * (n + 1) / 2 := sorry

end distinct_sums_count_l53_53288


namespace jill_bought_5_packs_of_red_bouncy_balls_l53_53328

theorem jill_bought_5_packs_of_red_bouncy_balls
  (r : ℕ) -- number of packs of red bouncy balls
  (yellow_packs : ℕ := 4)
  (bouncy_balls_per_pack : ℕ := 18)
  (extra_red_bouncy_balls : ℕ := 18)
  (total_yellow_bouncy_balls : ℕ := yellow_packs * bouncy_balls_per_pack)
  (total_red_bouncy_balls : ℕ := total_yellow_bouncy_balls + extra_red_bouncy_balls)
  (h : r * bouncy_balls_per_pack = total_red_bouncy_balls) :
  r = 5 :=
by sorry

end jill_bought_5_packs_of_red_bouncy_balls_l53_53328


namespace range_of_a_l53_53291

-- Given definitions from the problem
def p (a : ℝ) : Prop :=
  (4 - 4 * a) > 0

def q (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

-- The theorem we want to prove
theorem range_of_a (a : ℝ) : ¬ (p a ∨ q a) ↔ a ≥ 3 := 
by sorry

end range_of_a_l53_53291


namespace total_weight_of_fish_l53_53912

-- Define the weights of fish caught by Peter, Ali, and Joey.
variables (P A J : ℕ)

-- Ali caught twice as much fish as Peter.
def condition1 := A = 2 * P

-- Joey caught 1 kg more fish than Peter.
def condition2 := J = P + 1

-- Ali caught 12 kg of fish.
def condition3 := A = 12

-- Prove the total weight of the fish caught by all three is 25 kg.
theorem total_weight_of_fish :
  condition1 P A → condition2 P J → condition3 A → P + A + J = 25 :=
by
  intros h1 h2 h3
  sorry

end total_weight_of_fish_l53_53912


namespace standing_next_to_boris_l53_53604

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l53_53604


namespace smallest_m_plus_n_l53_53700

theorem smallest_m_plus_n
  (m n : ℕ)
  (h1 : m > 1)
  (h2 : ∀ x, -1 ≤ Real.log (n * x) / Real.log m ∧ Real.log (n * x) / Real.log m ≤ 1 ↔ (1/(n*m) ≤ x ∧ x ≤ m/n))
  (h3 : Real.log m ≠ 0) :
  ∃ m n, (1009 * (m^2 - 1)) / (m * n) = 1 / 1009 ∧ m > 1 ∧ n > 0 ∧ m + n = 7007 := 
sorry

end smallest_m_plus_n_l53_53700


namespace range_neg2a_plus_3_l53_53313

theorem range_neg2a_plus_3 (a : ℝ) (h : a < 1) : -2 * a + 3 > 1 :=
sorry

end range_neg2a_plus_3_l53_53313


namespace calculate_total_amount_l53_53438

theorem calculate_total_amount
  (price1 discount1 price2 discount2 additional_discount : ℝ)
  (h1 : price1 = 76) (h2 : discount1 = 25)
  (h3 : price2 = 85) (h4 : discount2 = 15)
  (h5 : additional_discount = 10) :
  price1 - discount1 + price2 - discount2 - additional_discount = 111 :=
by {
  sorry
}

end calculate_total_amount_l53_53438


namespace original_polygon_sides_l53_53809

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
(n - 2) * 180

theorem original_polygon_sides (x : ℕ) (h1 : sum_of_interior_angles (2 * x) = 2160) : x = 7 :=
by
  sorry

end original_polygon_sides_l53_53809


namespace line_equation_45_deg_through_point_l53_53649

theorem line_equation_45_deg_through_point :
  ∀ (x y : ℝ), 
  (∃ m k: ℝ, m = 1 ∧ k = 5 ∧ y = m * x + k) ∧ (∃ p q : ℝ, p = -2 ∧ q = 3 ∧ y = q ) :=  
  sorry

end line_equation_45_deg_through_point_l53_53649


namespace addition_problem_l53_53664

theorem addition_problem (F I V N E : ℕ) (h1: F = 8) (h2: I % 2 = 0) 
  (h3: 1 ≤ F ∧ F ≤ 9) (h4: 1 ≤ I ∧ I ≤ 9) (h5: 1 ≤ V ∧ V ≤ 9) 
  (h6: 1 ≤ N ∧ N ≤ 9) (h7: 1 ≤ E ∧ E ≤ 9) 
  (h8: F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E) 
  (h9: I ≠ V ∧ I ≠ N ∧ I ≠ E ∧ V ≠ N ∧ V ≠ E ∧ N ≠ E)
  (h10: 2 * F + 2 * I + 2 * V = 1000 * N + 100 * I + 10 * N + E):
  V = 5 :=
sorry

end addition_problem_l53_53664


namespace trajectory_sum_of_distances_to_axes_l53_53008

theorem trajectory_sum_of_distances_to_axes (x y : ℝ) (h : |x| + |y| = 6) :
  |x| + |y| = 6 := 
by 
  sorry

end trajectory_sum_of_distances_to_axes_l53_53008


namespace excircle_diameter_l53_53269

noncomputable def diameter_of_excircle (a b c S : ℝ) (s : ℝ) : ℝ :=
  2 * S / (s - a)

theorem excircle_diameter (a b c S h_A : ℝ) (s : ℝ) (h_v : 2 * ((a + b + c) / 2) = a + b + c) :
    diameter_of_excircle a b c S s = 2 * S / (s - a) :=
by
  sorry

end excircle_diameter_l53_53269


namespace prove_union_l53_53791

variable (M N : Set ℕ)
variable (x : ℕ)

def M_definition := (0 ∈ M) ∧ (x ∈ M) ∧ (M = {0, x})
def N_definition := (N = {1, 2})
def intersection_condition := (M ∩ N = {2})
def union_result := (M ∪ N = {0, 1, 2})

theorem prove_union (M : Set ℕ) (N : Set ℕ) (x : ℕ) :
  M_definition M x → N_definition N → intersection_condition M N → union_result M N :=
by
  sorry

end prove_union_l53_53791


namespace cos_alpha_plus_beta_l53_53493

theorem cos_alpha_plus_beta (α β : ℝ) (hα : Complex.exp (Complex.I * α) = 4 / 5 + Complex.I * 3 / 5)
  (hβ : Complex.exp (Complex.I * β) = -5 / 13 + Complex.I * 12 / 13) : 
  Real.cos (α + β) = -7 / 13 :=
  sorry

end cos_alpha_plus_beta_l53_53493


namespace cooking_time_l53_53893

theorem cooking_time (total_potatoes cooked_potatoes potato_time : ℕ) 
    (h1 : total_potatoes = 15) 
    (h2 : cooked_potatoes = 6) 
    (h3 : potato_time = 8) : 
    total_potatoes - cooked_potatoes * potato_time = 72 :=
by
    sorry

end cooking_time_l53_53893


namespace union_sets_l53_53296

theorem union_sets :
  let A := { x : ℝ | x^2 - x - 2 < 0 }
  let B := { x : ℝ | x > -2 ∧ x < 0 }
  A ∪ B = { x : ℝ | x > -2 ∧ x < 2 } :=
by
  sorry

end union_sets_l53_53296


namespace triangle_side_identity_l53_53626

theorem triangle_side_identity
  (a b c : ℝ)
  (alpha beta gamma : ℝ)
  (h1 : alpha = 60)
  (h2 : a^2 = b^2 + c^2 - b * c) :
  a^2 = (a^3 + b^3 + c^3) / (a + b + c) := 
by
  sorry

end triangle_side_identity_l53_53626


namespace who_next_to_boris_l53_53619

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l53_53619


namespace expression_value_l53_53996

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l53_53996


namespace number_of_three_star_reviews_l53_53210

theorem number_of_three_star_reviews:
  ∀ (x : ℕ),
  (6 * 5 + 7 * 4 + 1 * 2 + x * 3) / 18 = 4 →
  x = 4 :=
by
  intros x H
  sorry  -- Placeholder for the proof

end number_of_three_star_reviews_l53_53210


namespace calculate_gcd_correct_l53_53885

theorem calculate_gcd_correct (n : ℕ) (numbers : list ℕ) (h : numbers.length = n) 
  (h_pos : ∀ x ∈ numbers, 0 < x) : 
  let a := numbers.foldl gcd 0 in
  is_gcd_of_list numbers a :=
by
  assume numbers h h_pos
  -- proof goes here
  sorry

end calculate_gcd_correct_l53_53885


namespace max_positive_integers_on_circle_l53_53133

theorem max_positive_integers_on_circle (a : ℕ → ℕ) (h: ∀ k : ℕ, 2 < k → a k > a (k-1) + a (k-2)) :
  ∃ n : ℕ, (∀ i < 2018, a i > 0 -> n ≤ 1009) :=
  sorry

end max_positive_integers_on_circle_l53_53133


namespace angle_between_a_and_b_is_2pi_over_3_l53_53138

open Real

variables (a b c : ℝ × ℝ)

-- Given conditions
def condition1 := a.1^2 + a.2^2 = 2  -- |a| = sqrt(2)
def condition2 := b = (-1, 1)        -- b = (-1, 1)
def condition3 := c = (2, -2)        -- c = (2, -2)
def condition4 := a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 1  -- a · (b + c) = 1

-- Prove the angle θ between a and b is 2π/3
theorem angle_between_a_and_b_is_2pi_over_3 :
  condition1 a → condition2 b → condition3 c → condition4 a b c →
  ∃ θ, 0 ≤ θ ∧ θ ≤ π ∧ cos θ = -(1/2) ∧ θ = 2 * π / 3 :=
by
  sorry

end angle_between_a_and_b_is_2pi_over_3_l53_53138


namespace average_of_side_lengths_of_squares_l53_53389

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l53_53389


namespace expansion_eq_coeff_sum_l53_53143

theorem expansion_eq_coeff_sum (a : ℕ → ℤ) (m : ℤ) 
  (h : (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7)
  (h_coeff : a 4 = -35) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1 ∧ a 1 + a 3 + a 5 + a 7 = 26 := 
by 
  sorry

end expansion_eq_coeff_sum_l53_53143


namespace math_problem_l53_53991

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l53_53991


namespace inequality_proof_l53_53788

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_abc : a * b * c = 1)

theorem inequality_proof :
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) 
  ≥ (3 / 2) + (1 / 4) * (a * (c - b) ^ 2 / (c + b) + b * (c - a) ^ 2 / (c + a) + c * (b - a) ^ 2 / (b + a)) :=
by
  sorry

end inequality_proof_l53_53788


namespace problem_statement_l53_53977

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l53_53977


namespace max_green_beads_l53_53036

/-- Define the problem conditions:
  1. A necklace consists of 100 beads of red, blue, and green colors.
  2. Among any five consecutive beads, there is at least one blue bead.
  3. Among any seven consecutive beads, there is at least one red bead.
  4. The beads in the necklace are arranged cyclically (the last one is adjacent to the first one).
--/
def necklace_conditions := 
  ∀ (beads : ℕ → ℕ) (n : ℕ), 
    (∀ i, 0 ≤ beads i ∧ beads i < 3) ∧
    (∀ i, 0 ≤ i ∧ i < 100 → ∃ j, i ≤ j ∧ j < i + 5 ∧ beads j = 1) ∧
    (∀ i, 0 ≤ i ∧ i < 100 → ∃ j, i ≤ j ∧ j < i + 7 ∧ beads j = 0)

/-- Prove the maximum number of green beads that can be in this necklace is 65. --/
theorem max_green_beads : ∃ beads : (ℕ → ℕ), necklace_conditions beads → (∑ p, if beads p = 2 then 1 else 0) = 65 :=
by sorry

end max_green_beads_l53_53036


namespace select_k_plus_1_nums_divisible_by_n_l53_53844

theorem select_k_plus_1_nums_divisible_by_n (n k : ℕ) (hn : n > 0) (hk : k > 0) (nums : Fin (n + k) → ℕ) :
  ∃ (indices : Finset (Fin (n + k))), indices.card ≥ k + 1 ∧ (indices.sum (nums ∘ id)) % n = 0 :=
sorry

end select_k_plus_1_nums_divisible_by_n_l53_53844


namespace integer_values_of_a_l53_53268

-- Define the polynomial P(x)
def P (a x : ℤ) : ℤ := x^3 + a * x^2 + 3 * x + 7

-- Define the main theorem
theorem integer_values_of_a (a x : ℤ) (hx : P a x = 0) (hx_is_int : x = 1 ∨ x = -1 ∨ x = 7 ∨ x = -7) :
  a = -11 ∨ a = -3 :=
by
  sorry

end integer_values_of_a_l53_53268


namespace polygon_perimeter_l53_53099

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53099


namespace perimeter_of_regular_polygon_l53_53054

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53054


namespace solution_set_inequality_system_l53_53863

theorem solution_set_inequality_system (x : ℝ) :
  (x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x) ↔ (-1 ≤ x ∧ x < 5) := by
  sorry

end solution_set_inequality_system_l53_53863


namespace subset_condition_l53_53197

variable {U : Type}
variables (P Q : Set U)

theorem subset_condition (h : P ∩ Q = P) : ∀ x : U, x ∉ Q → x ∉ P :=
by {
  sorry
}

end subset_condition_l53_53197


namespace stock_percentage_calculation_l53_53280

noncomputable def stock_percentage (investment_amount stock_price annual_income : ℝ) : ℝ :=
  (annual_income / (investment_amount / stock_price) / stock_price) * 100

theorem stock_percentage_calculation :
  stock_percentage 6800 136 1000 = 14.71 :=
by
  sorry

end stock_percentage_calculation_l53_53280


namespace calculate_milk_and_oil_l53_53874

theorem calculate_milk_and_oil (q_f div_f milk_p oil_p : ℕ) (portions q_m q_o : ℕ) :
  q_f = 1050 ∧ div_f = 350 ∧ milk_p = 70 ∧ oil_p = 30 ∧
  portions = q_f / div_f ∧
  q_m = portions * milk_p ∧
  q_o = portions * oil_p →
  q_m = 210 ∧ q_o = 90 := by
  sorry

end calculate_milk_and_oil_l53_53874


namespace find_n_l53_53141

theorem find_n {n : ℕ} (H : 2 * nat.choose n 9 = nat.choose n 8 + nat.choose n 10) :
  n = 14 ∨ n = 23 :=
by
  sorry

end find_n_l53_53141


namespace adam_clothing_ratio_l53_53910

-- Define the initial amount of clothing Adam took out
def initial_clothing_adam : ℕ := 4 + 4 + 8 + 20

-- Define the number of friends donating the same amount of clothing as Adam
def number_of_friends : ℕ := 3

-- Define the total number of clothes being donated
def total_donated_clothes : ℕ := 126

-- Define the ratio of the clothes Adam is keeping to the clothes he initially took out
def ratio_kept_to_initial (initial_clothing: ℕ) (total_donated: ℕ) (kept: ℕ) : Prop :=
  kept * initial_clothing = 0

-- Theorem statement
theorem adam_clothing_ratio :
  ratio_kept_to_initial initial_clothing_adam total_donated_clothes 0 :=
by 
  sorry

end adam_clothing_ratio_l53_53910


namespace find_value_of_n_l53_53126

theorem find_value_of_n (n : ℤ) : 
    n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 :=
by 
  intro h
  sorry

end find_value_of_n_l53_53126


namespace average_side_length_of_squares_l53_53358

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53358


namespace range_of_a_l53_53637

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 1| + |x - 2| ≤ a^2 + a + 1)) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l53_53637


namespace sandbox_width_l53_53332

theorem sandbox_width (P : ℕ) (W L : ℕ) (h1 : P = 30) (h2 : L = 2 * W) (h3 : P = 2 * L + 2 * W) : W = 5 := 
sorry

end sandbox_width_l53_53332


namespace regular_polygon_perimeter_l53_53082

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53082


namespace remainder_mul_mod_l53_53717

theorem remainder_mul_mod (a b n : ℕ) (h₁ : a ≡ 3 [MOD n]) (h₂ : b ≡ 150 [MOD n]) (n_eq : n = 400) : 
  (a * b) % n = 50 :=
by 
  sorry

end remainder_mul_mod_l53_53717


namespace handshake_max_participants_l53_53321

theorem handshake_max_participants (N : ℕ) (hN : 5 < N) (hNotAllShaken: ∃ p1 p2 : ℕ, p1 ≠ p2 ∧ p1 < N ∧ p2 < N ∧ (∀ i : ℕ, i < N → i ≠ p1 → i ≠ p2 → ∃ j : ℕ, j < N ∧ j ≠ i ∧ j ≠ p1 ∧ j ≠ p2)) :
∃ k, k = N - 2 :=
by
  sorry

end handshake_max_participants_l53_53321


namespace parts_repetition_cycle_l53_53434

noncomputable def parts_repetition_condition (t : ℕ) : Prop := sorry
def parts_initial_condition : Prop := sorry

theorem parts_repetition_cycle :
  parts_initial_condition →
  parts_repetition_condition 2 ∧
  parts_repetition_condition 4 ∧
  parts_repetition_condition 38 ∧
  parts_repetition_condition 76 :=
sorry


end parts_repetition_cycle_l53_53434


namespace sector_area_l53_53538

theorem sector_area (r α S : ℝ) (h1 : α = 2) (h2 : 2 * r + α * r = 8) : S = 4 :=
sorry

end sector_area_l53_53538


namespace question_b_l53_53418

theorem question_b (a b c : ℝ) (h : c ≠ 0) (h_eq : a / c = b / c) : a = b := 
by
  sorry

end question_b_l53_53418


namespace largest_quantity_l53_53125

theorem largest_quantity 
  (A := (2010 / 2009) + (2010 / 2011))
  (B := (2012 / 2011) + (2010 / 2011))
  (C := (2011 / 2010) + (2011 / 2012)) : C > A ∧ C > B := 
by {
  sorry
}

end largest_quantity_l53_53125


namespace sum_of_three_divisible_by_three_l53_53338

open Finset 

theorem sum_of_three_divisible_by_three (S : Finset ℕ) (h : S.card = 7) :
  ∃ a b c ∈ S, (a + b + c) % 3 = 0 :=
by
  sorry

end sum_of_three_divisible_by_three_l53_53338


namespace standing_next_to_boris_l53_53606

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l53_53606


namespace who_is_next_to_boris_l53_53593

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l53_53593


namespace infinite_series_evaluates_to_12_l53_53456

noncomputable def infinite_series : ℝ :=
  ∑' k, (k^3) / (3^k)

theorem infinite_series_evaluates_to_12 :
  infinite_series = 12 :=
by
  sorry

end infinite_series_evaluates_to_12_l53_53456


namespace regular_polygon_perimeter_is_28_l53_53072

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53072


namespace compute_expression_l53_53166

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l53_53166


namespace quadratic_interlaced_roots_l53_53729

theorem quadratic_interlaced_roots
  (p1 p2 q1 q2 : ℝ)
  (h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  ∃ (r1 r2 s1 s2 : ℝ),
    (r1^2 + p1 * r1 + q1 = 0) ∧
    (r2^2 + p1 * r2 + q1 = 0) ∧
    (s1^2 + p2 * s1 + q2 = 0) ∧
    (s2^2 + p2 * s2 + q2 = 0) ∧
    (r1 < s1 ∧ s1 < r2 ∨ s1 < r1 ∧ r1 < s2) :=
sorry

end quadratic_interlaced_roots_l53_53729


namespace math_problem_l53_53135

open Real

theorem math_problem
  (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h : x^2 + y^2 + z^2 = 3) :
  sqrt (3 - ( (x + y) / 2) ^ 2) + sqrt (3 - ( (y + z) / 2) ^ 2) + sqrt (3 - ( (z + x) / 2) ^ 2) ≥ 3 * sqrt 2 :=
by 
  sorry

end math_problem_l53_53135


namespace regular_polygon_perimeter_is_28_l53_53073

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53073


namespace total_journey_time_l53_53745

def distance_to_post_office : ℝ := 19.999999999999996
def speed_to_post_office : ℝ := 25
def speed_back : ℝ := 4

theorem total_journey_time : 
  (distance_to_post_office / speed_to_post_office) + (distance_to_post_office / speed_back) = 5.8 :=
by
  sorry

end total_journey_time_l53_53745


namespace perimeter_of_polygon_l53_53041

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53041


namespace math_problem_l53_53988

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l53_53988


namespace doug_marbles_l53_53276

theorem doug_marbles (e_0 d_0 : ℕ) (h1 : e_0 = d_0 + 12) (h2 : e_0 - 20 = 17) : d_0 = 25 :=
by
  sorry

end doug_marbles_l53_53276


namespace income_percentage_increase_l53_53886

theorem income_percentage_increase (b : ℝ) (a : ℝ) (h : a = b * 0.75) :
  (b - a) / a * 100 = 33.33 :=
by
  sorry

end income_percentage_increase_l53_53886


namespace milk_amount_at_beginning_l53_53535

theorem milk_amount_at_beginning (H: 0.69 = 0.6 * total_milk) : total_milk = 1.15 :=
sorry

end milk_amount_at_beginning_l53_53535


namespace remainders_equal_l53_53832

theorem remainders_equal (P P' D R k s s' : ℕ) (h1 : P > P') 
  (h2 : P % D = 2 * R) (h3 : P' % D = R) (h4 : R < D) :
  (k * (P + P')) % D = s → (k * (2 * R + R)) % D = s' → s = s' :=
by
  sorry

end remainders_equal_l53_53832


namespace average_side_lengths_l53_53366

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l53_53366


namespace two_point_three_six_as_fraction_l53_53233

theorem two_point_three_six_as_fraction : (236 : ℝ) / 100 = (59 : ℝ) / 25 := 
by
  sorry

end two_point_three_six_as_fraction_l53_53233


namespace log_problem_l53_53920

open Real

theorem log_problem : 2 * log 5 + log 4 = 2 := by
  sorry

end log_problem_l53_53920


namespace max_angle_position_l53_53871

-- Definitions for points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

-- Definitions for points A and B on the X-axis
def A (a : ℝ) : Point := { x := -a, y := 0 }
def B (a : ℝ) : Point := { x := a, y := 0 }

-- Definition for point C moving along the line y = 10 - x
def moves_along_line (C : Point) : Prop :=
  C.y = 10 - C.x

-- Definition for calculating the angle ACB (gamma)
def angle_ACB (A B C : Point) : ℝ := sorry -- The detailed function to calculate angle is omitted for brevity

-- Main statement to prove
theorem max_angle_position (a : ℝ) (C : Point) (ha : 0 ≤ a ∧ a ≤ 10) (hC : moves_along_line C) :
  (C = { x := 4, y := 6 } ∨ C = { x := 16, y := -6 }) ↔ (∀ C', moves_along_line C' → (angle_ACB (A a) (B a) C') ≤ angle_ACB (A a) (B a) C) :=
sorry

end max_angle_position_l53_53871


namespace intersection_complement_l53_53513

open Set

variable (U A B : Set ℕ)

-- Given conditions:
def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3}

theorem intersection_complement (U A B : Set ℕ) : 
  U = universal_set → A = set_A → B = set_B → (A ∩ (U \ B)) = {1, 5} := by
  sorry

end intersection_complement_l53_53513


namespace h_is_even_l53_53195

variable {α : Type*} [LinearOrderedCommRing α]
variable (g h : α → α)

def is_odd_function (f : α → α) : Prop :=
∀ x, f (-x) = -f x

def is_even_function (f : α → α) : Prop :=
∀ x, f (-x) = f x

def h_definition (g : α → α) (x : α) : α :=
| g (x^5) |

theorem h_is_even (hg : is_odd_function g) : is_even_function (h_definition g) :=
sorry

end h_is_even_l53_53195


namespace polygon_perimeter_l53_53097

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53097


namespace price_reduction_after_markup_l53_53110

theorem price_reduction_after_markup (p : ℝ) (x : ℝ) (h₁ : 0 < p) (h₂ : 0 ≤ x ∧ x < 1) :
  (1.25 : ℝ) * (1 - x) = 1 → x = 0.20 := by
  sorry

end price_reduction_after_markup_l53_53110


namespace math_problem_l53_53240

theorem math_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 35) : L = 1631 := 
by
  sorry

end math_problem_l53_53240


namespace option_C_correct_l53_53882

theorem option_C_correct (a b : ℝ) : ((a^2 * b)^3) / ((-a * b)^2) = a^4 * b := by
  sorry

end option_C_correct_l53_53882


namespace satisfy_inequality_l53_53124

theorem satisfy_inequality (x : ℤ) : 
  (3 * x - 5 ≤ 10 - 2 * x) ↔ (x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end satisfy_inequality_l53_53124


namespace ferries_are_divisible_by_4_l53_53000

theorem ferries_are_divisible_by_4 (t T : ℕ) (H : ∃ n : ℕ, T = n * t) :
  ∃ N : ℕ, N = 4 * (T / t) ∧ N % 4 = 0 :=
by
  sorry

end ferries_are_divisible_by_4_l53_53000


namespace final_value_l53_53973

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l53_53973


namespace who_is_next_to_boris_l53_53591

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l53_53591


namespace largest_square_side_l53_53025

theorem largest_square_side (width length : ℕ) (h_width : width = 63) (h_length : length = 42) : 
  Nat.gcd width length = 21 :=
by
  rw [h_width, h_length]
  sorry

end largest_square_side_l53_53025


namespace polygon_perimeter_l53_53100

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53100


namespace average_side_lengths_l53_53367

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l53_53367


namespace who_next_to_boris_l53_53616

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l53_53616


namespace cannot_determine_right_triangle_l53_53441

-- Definitions of conditions
variables {a b c : ℕ}
variables {angle_A angle_B angle_C : ℕ}

-- Context for the proof
def is_right_angled_triangle_via_sides (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def triangle_angle_sum_theorem (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Statements for conditions as used in the problem
def condition_A (a2 b2 c2 : ℕ) : Prop :=
  a2 = 1 ∧ b2 = 2 ∧ c2 = 3

def condition_B (a b c : ℕ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5

def condition_C (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B = angle_C

def condition_D (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A = 45 ∧ angle_B = 60 ∧ angle_C = 75

-- Proof statement
theorem cannot_determine_right_triangle (a b c angle_A angle_B angle_C : ℕ) :
  condition_D angle_A angle_B angle_C →
  ¬(is_right_angled_triangle_via_sides a b c) :=
sorry

end cannot_determine_right_triangle_l53_53441


namespace monomial_completes_square_l53_53879

variable (x : ℝ)

theorem monomial_completes_square :
  ∃ (m : ℝ), ∀ (x : ℝ), ∃ (a b : ℝ), (16 * x^2 + 1 + m) = (a * x + b)^2 :=
sorry

end monomial_completes_square_l53_53879


namespace smallest_whole_number_l53_53718

theorem smallest_whole_number :
  ∃ a : ℕ, a % 3 = 2 ∧ a % 5 = 3 ∧ a % 7 = 3 ∧ ∀ b : ℕ, (b % 3 = 2 ∧ b % 5 = 3 ∧ b % 7 = 3 → a ≤ b) :=
sorry

end smallest_whole_number_l53_53718


namespace giant_lollipop_calories_l53_53838

-- Definitions based on the conditions
def sugar_per_chocolate_bar := 10
def chocolate_bars_bought := 14
def sugar_in_giant_lollipop := 37
def total_sugar := 177
def calories_per_gram_of_sugar := 4

-- Prove that the number of calories in the giant lollipop is 148 given the conditions
theorem giant_lollipop_calories : (sugar_in_giant_lollipop * calories_per_gram_of_sugar) = 148 := by
  sorry

end giant_lollipop_calories_l53_53838


namespace total_amount_paid_l53_53870

noncomputable def cost_per_night_per_person : ℕ := 40
noncomputable def number_of_people : ℕ := 3
noncomputable def number_of_nights : ℕ := 3

theorem total_amount_paid (cost_per_night_per_person number_of_people number_of_nights : ℕ) :
  (cost_per_night_per_person * number_of_people * number_of_nights = 360) :=
by
  have h : cost_per_night_per_person * number_of_people * number_of_nights = 40 * 3 * 3 := by rfl
  rw h
  exact rfl

end total_amount_paid_l53_53870


namespace average_side_length_of_squares_l53_53361

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53361


namespace series_sum_formula_l53_53263

open BigOperators

theorem series_sum_formula (n : ℕ) :
  (∑ k in Finset.range n, k * (k + 2)^2) = (n.choose 3) * (3 * n + 2) / 2 :=
sorry

end series_sum_formula_l53_53263


namespace official_exchange_rate_l53_53724

theorem official_exchange_rate (E : ℝ)
  (h1 : 70 = 10 * (7 / 5) * E) :
  E = 5 :=
by
  sorry

end official_exchange_rate_l53_53724


namespace sum_of_number_and_square_is_306_l53_53181

theorem sum_of_number_and_square_is_306 (n : ℕ) (h : n = 17) : n + n^2 = 306 :=
by
  sorry

end sum_of_number_and_square_is_306_l53_53181


namespace fg_2_eq_9_l53_53314

def f (x: ℝ) := x^2
def g (x: ℝ) := -4 * x + 5

theorem fg_2_eq_9 : f (g 2) = 9 :=
by
  sorry

end fg_2_eq_9_l53_53314


namespace tom_tim_typing_ratio_l53_53024

variable (T M : ℝ)

theorem tom_tim_typing_ratio (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
sorry

end tom_tim_typing_ratio_l53_53024


namespace perimeter_of_regular_polygon_l53_53055

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53055


namespace final_value_l53_53968

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l53_53968


namespace balls_in_boxes_l53_53153

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end balls_in_boxes_l53_53153


namespace sum_of_fractions_l53_53962

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l53_53962


namespace find_g4_l53_53855

noncomputable def g : ℝ → ℝ := sorry

theorem find_g4 (h : ∀ x y : ℝ, x * g y = 2 * y * g x) (h₁ : g 10 = 5) : g 4 = 4 :=
sorry

end find_g4_l53_53855


namespace total_time_marco_6_laps_total_time_in_minutes_and_seconds_l53_53676

noncomputable def marco_running_time : ℕ :=
  let distance_1 := 150
  let speed_1 := 5
  let time_1 := distance_1 / speed_1

  let distance_2 := 300
  let speed_2 := 4
  let time_2 := distance_2 / speed_2

  let time_per_lap := time_1 + time_2
  let total_laps := 6
  let total_time_seconds := time_per_lap * total_laps

  total_time_seconds

theorem total_time_marco_6_laps : marco_running_time = 630 := sorry

theorem total_time_in_minutes_and_seconds : 10 * 60 + 30 = 630 := sorry

end total_time_marco_6_laps_total_time_in_minutes_and_seconds_l53_53676


namespace arithmetic_sequence_common_difference_l53_53644

theorem arithmetic_sequence_common_difference
  (a_n : ℕ → ℤ) (h_arithmetic : ∀ n, (a_n (n + 1) = a_n n + d)) 
  (h_sum1 : a_n 1 + a_n 3 + a_n 5 = 105)
  (h_sum2 : a_n 2 + a_n 4 + a_n 6 = 99) : 
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l53_53644


namespace scientific_notation_0_056_l53_53405

theorem scientific_notation_0_056 :
  (0.056 = 5.6 * 10^(-2)) :=
by
  sorry

end scientific_notation_0_056_l53_53405


namespace touchdowns_points_l53_53695

theorem touchdowns_points 
    (num_touchdowns : ℕ) (total_points : ℕ) 
    (h1 : num_touchdowns = 3) 
    (h2 : total_points = 21) : 
    total_points / num_touchdowns = 7 :=
by
    sorry

end touchdowns_points_l53_53695


namespace average_side_length_of_squares_l53_53354

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53354


namespace six_people_paint_time_l53_53665

noncomputable def time_to_paint_house_with_six_people 
    (initial_people : ℕ) (initial_time : ℝ) (less_efficient_worker_factor : ℝ) 
    (new_people : ℕ) : ℝ :=
  let initial_total_efficiency := initial_people - 1 + less_efficient_worker_factor
  let total_work := initial_total_efficiency * initial_time
  let new_total_efficiency := (new_people - 1) + less_efficient_worker_factor
  total_work / new_total_efficiency

theorem six_people_paint_time (initial_people : ℕ) (initial_time : ℝ) 
    (less_efficient_worker_factor : ℝ) (new_people : ℕ) :
    initial_people = 5 → initial_time = 10 → less_efficient_worker_factor = 0.5 → new_people = 6 →
    time_to_paint_house_with_six_people initial_people initial_time less_efficient_worker_factor new_people = 8.18 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end six_people_paint_time_l53_53665


namespace platform_length_1000_l53_53426

open Nat Real

noncomputable def length_of_platform (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) : ℝ :=
  let speed := train_length / time_pole
  let platform_length := (speed * time_platform) - train_length
  platform_length

theorem platform_length_1000 :
  length_of_platform 300 9 39 = 1000 := by
  sorry

end platform_length_1000_l53_53426


namespace sum_base10_to_base4_l53_53916

theorem sum_base10_to_base4 : 
  (31 + 22 : ℕ) = 3 * 4^2 + 1 * 4^1 + 1 * 4^0 :=
by
  sorry

end sum_base10_to_base4_l53_53916


namespace average_side_lengths_l53_53374

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l53_53374


namespace hash_op_8_4_l53_53537

def hash_op (a b : ℕ) : ℕ := a + a / b - 2

theorem hash_op_8_4 : hash_op 8 4 = 8 := 
by 
  -- The proof is left as an exercise, indicated by sorry.
  sorry

end hash_op_8_4_l53_53537


namespace brochures_per_box_l53_53557

theorem brochures_per_box (total_brochures : ℕ) (boxes : ℕ) 
  (htotal : total_brochures = 5000) (hboxes : boxes = 5) : 
  (1000 / 5000 : ℚ) = 1 / 5 := 
by sorry

end brochures_per_box_l53_53557


namespace log2_lt_prob_l53_53648

open Set

noncomputable def prob_event (x : ℝ) := x ∈ Ioc 0 2 ∧ x ∈ Icc 0 3

theorem log2_lt_prob : 
  let space := Icc (0 : ℝ) 3 in
  let event := {x | log 2 x < 1} ∩ space in
  (event.measure (volume.restrict space)) / (space.measure volume) = 2 / 3 := sorry

end log2_lt_prob_l53_53648


namespace clarinet_fraction_l53_53519

theorem clarinet_fraction 
  (total_flutes total_clarinets total_trumpets total_pianists total_band: ℕ)
  (percent_flutes : ℚ) (fraction_trumpets fraction_pianists : ℚ)
  (total_persons_in_band: ℚ)
  (flutes_got_in : total_flutes = 20)
  (clarinets_got_in : total_clarinets = 30)
  (trumpets_got_in : total_trumpets = 60)
  (pianists_got_in : total_pianists = 20)
  (band_got_in : total_band = 53)
  (percent_flutes_got_in: percent_flutes = 0.8)
  (fraction_trumpets_got_in: fraction_trumpets = 1/3)
  (fraction_pianists_got_in: fraction_pianists = 1/10)
  (persons_in_band: total_persons_in_band = 53) :
  (15 / 30 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end clarinet_fraction_l53_53519


namespace inequality_proof_l53_53490

variable {a b c d : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end inequality_proof_l53_53490


namespace wire_total_length_l53_53743

theorem wire_total_length (a b c total_length : ℕ) (h1 : a = 7) (h2 : b = 3) (h3 : c = 2) (h4 : c * 16 = 32) :
  total_length = (a + b + c) * (16 / c) :=
by
  have h5 : c = 2 := by rw [←nat.add_assoc, add_comm, h3]
  have h6 : total_length = (a + b + c) * 8 := sorry
  exact h6

end wire_total_length_l53_53743


namespace work_duration_l53_53890

/-- Definition of the work problem, showing that the work lasts for 5 days. -/
theorem work_duration (work_rate_p work_rate_q : ℝ) (total_work time_p time_q : ℝ) 
  (p_work_days q_work_days : ℝ) 
  (H1 : p_work_days = 10)
  (H2 : q_work_days = 6)
  (H3 : work_rate_p = total_work / 10)
  (H4 : work_rate_q = total_work / 6)
  (H5 : time_p = 2)
  (H6 : time_q = 4 * total_work / 5 / (total_work / 2 / 3) )
  : (time_p + time_q = 5) := 
by 
  sorry

end work_duration_l53_53890


namespace expected_value_bound_l53_53790

open Probability Theory

noncomputable section

def trimOnce (l : List ℝ) : List ℝ := l.inits.filter (λ l => l.length = 3).map (λ l => l.sorted (λ a b => a ≤ b) !! 1)

def trimRepeatedly (l : List ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => l.head' |>.get_or_else 0
  | k + 1 => trimRepeatedly (trimOnce l) k

def finalTrimmedValue (l : List ℝ) : ℝ := trimRepeatedly l 2021

def E_X_abs_diff_half : ℝ :=
  ENNReal.toReal (expectation (λ (l : List ℝ) => |finalTrimmedValue l - (1/2)|))

theorem expected_value_bound :
  ∀ (l : List ℝ), l.length = 3^2021 ∧ (∀ x, x ∈ l → x ≥ 0 ∧ x ≤ 1) →
  E_X_abs_diff_half ≥ 1/4 * (2/3)^2021 :=
sorry

end expected_value_bound_l53_53790


namespace price_of_third_variety_l53_53027

-- Define the given conditions
def price1 : ℝ := 126
def price2 : ℝ := 135
def average_price : ℝ := 153
def ratio1 : ℝ := 1
def ratio2 : ℝ := 1
def ratio3 : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := ratio1 + ratio2 + ratio3

-- Define the equation based on the given conditions
def weighted_avg_price (P : ℝ) : Prop :=
  (ratio1 * price1 + ratio2 * price2 + ratio3 * P) / total_ratio = average_price

-- Statement of the proof
theorem price_of_third_variety :
  ∃ P : ℝ, weighted_avg_price P ∧ P = 175.5 :=
by {
  -- Proof omitted
  sorry
}

end price_of_third_variety_l53_53027


namespace distance_between_hyperbola_vertices_l53_53776

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l53_53776


namespace perimeter_of_regular_polygon_l53_53061

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53061


namespace rectangle_area_unchanged_l53_53350

theorem rectangle_area_unchanged (l w : ℝ) (h : l * w = 432) : 
  0.8 * l * 1.25 * w = 432 := 
by {
  -- The proof goes here
  sorry
}

end rectangle_area_unchanged_l53_53350


namespace total_time_on_road_l53_53327

def driving_time_day1 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def driving_time_day2 (jade_time krista_time break_time krista_refuel lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + break_time + krista_refuel + lunch_break

def driving_time_day3 (jade_time krista_time krista_delay lunch_break : ℝ) : ℝ :=
  jade_time + krista_time + krista_delay + lunch_break

def total_driving_time (day1 day2 day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem total_time_on_road :
  total_driving_time 
    (driving_time_day1 8 6 1 1) 
    (driving_time_day2 7 5 0.5 (1/3) 1) 
    (driving_time_day3 6 4 1 1) 
  = 42.3333 := 
  by 
    sorry

end total_time_on_road_l53_53327


namespace exists_bisecting_line_l53_53938

theorem exists_bisecting_line {P : Point} {pentagon : Polygon} 
  (h_convex : pentagon.isConvex) (h_on_boundary : P ∈ pentagon.boundary)
  : ∃ (Q : Point), (Q ∈ pentagon.boundary) ∧ (line_through P Q).dividesAreaIntoTwoEqualParts :=
sorry

end exists_bisecting_line_l53_53938


namespace A_days_to_complete_work_l53_53560

noncomputable def work (W : ℝ) (A_work_per_day B_work_per_day : ℝ) (days_A days_B days_B_alone : ℝ) : ℝ :=
  A_work_per_day * days_A + B_work_per_day * days_B

theorem A_days_to_complete_work 
  (W : ℝ)
  (A_work_per_day B_work_per_day : ℝ)
  (days_A days_B days_B_alone : ℝ)
  (h1 : days_A = 5)
  (h2 : days_B = 12)
  (h3 : days_B_alone = 18)
  (h4 : B_work_per_day = W / days_B_alone)
  (h5 : work W A_work_per_day B_work_per_day days_A days_B days_B_alone = W) :
  W / A_work_per_day = 15 := 
sorry

end A_days_to_complete_work_l53_53560


namespace expression_value_l53_53999

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l53_53999


namespace five_minus_a_l53_53157

theorem five_minus_a (a b : ℚ) (h1 : 5 + a = 3 - b) (h2 : 3 + b = 8 + a) : 5 - a = 17/2 :=
by
  sorry

end five_minus_a_l53_53157


namespace xiaohong_test_number_l53_53236

theorem xiaohong_test_number (x : ℕ) :
  (88 * x - 85 * (x - 1) = 100) → x = 5 :=
by
  intro h
  sorry

end xiaohong_test_number_l53_53236


namespace average_side_lengths_l53_53371

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l53_53371


namespace smallest_perimeter_of_triangle_with_area_sqrt3_l53_53329

open Real

-- Define an equilateral triangle with given area
def equilateral_triangle (a : ℝ) : Prop :=
  ∃ s: ℝ, s > 0 ∧ a = (sqrt 3 / 4) * s^2

-- Problem statement: Prove the smallest perimeter of such a triangle is 6.
theorem smallest_perimeter_of_triangle_with_area_sqrt3 : 
  equilateral_triangle (sqrt 3) → ∃ s: ℝ, s > 0 ∧ 3 * s = 6 :=
by 
  sorry

end smallest_perimeter_of_triangle_with_area_sqrt3_l53_53329


namespace sum_of_ages_l53_53699

theorem sum_of_ages (a b c d e : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 9) 
  (h2 : 1 ≤ b ∧ b ≤ 9) 
  (h3 : 1 ≤ c ∧ c ≤ 9) 
  (h4 : 1 ≤ d ∧ d ≤ 9) 
  (h5 : 1 ≤ e ∧ e ≤ 9) 
  (h6 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h7 : a * b = 28 ∨ a * c = 28 ∨ a * d = 28 ∨ a * e = 28 ∨ b * c = 28 ∨ b * d = 28 ∨ b * e = 28 ∨ c * d = 28 ∨ c * e = 28 ∨ d * e = 28)
  (h8 : a * b = 20 ∨ a * c = 20 ∨ a * d = 20 ∨ a * e = 20 ∨ b * c = 20 ∨ b * d = 20 ∨ b * e = 20 ∨ c * d = 20 ∨ c * e = 20 ∨ d * e = 20)
  (h9 : a + b = 14 ∨ a + c = 14 ∨ a + d = 14 ∨ a + e = 14 ∨ b + c = 14 ∨ b + d = 14 ∨ b + e = 14 ∨ c + d = 14 ∨ c + e = 14 ∨ d + e = 14) 
  : a + b + c + d + e = 25 :=
by
  sorry

end sum_of_ages_l53_53699


namespace question1_question2_case1_question2_case2_question2_case3_l53_53299

def f (x a : ℝ) : ℝ := x^2 + (1 - a) * x - a

theorem question1 (x : ℝ) (h : (-1 < x) ∧ (x < 3)) : f x 3 < 0 := sorry

theorem question2_case1 (x : ℝ) : f x (-1) > 0 ↔ x ≠ -1 := sorry

theorem question2_case2 (x a : ℝ) (h : a > -1) : f x a > 0 ↔ (x < -1 ∨ x > a) := sorry

theorem question2_case3 (x a : ℝ) (h : a < -1) : f x a > 0 ↔ (x < a ∨ x > -1) := sorry

end question1_question2_case1_question2_case2_question2_case3_l53_53299


namespace round_sum_to_nearest_tenth_l53_53753

theorem round_sum_to_nearest_tenth (a b : ℝ) (c : ℝ) (h₀ : a = 2.72) (h₁ : b = 0.76) (h₂ : c = 3.48) :
  Real.round (a + b) = 3.5 :=
by
  rw [h₀, h₁]
  have h_sum : a + b = c, by sorry
  rw [h_sum, ←h₂]
  rw [Real.round]
  sorry

end round_sum_to_nearest_tenth_l53_53753


namespace integer_values_of_b_l53_53629

theorem integer_values_of_b (b : ℤ) :
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ 
  b = -21 ∨ b = 19 ∨ b = -17 ∨ b = -4 ∨ b = 3 :=
by
  sorry

end integer_values_of_b_l53_53629


namespace part1_part2_l53_53685

-- Define the conditions for part (1)
def nonEmptyBoxes := ∀ i j k: Nat, (i ≠ j ∧ i ≠ k ∧ j ≠ k)
def ball3inBoxB := ∀ (b3: Nat) (B: Nat), b3 = 3 ∧ B > 0

-- Define the conditions for part (2)
def ball1notInBoxA := ∀ (b1: Nat) (A: Nat), b1 ≠ 1 ∧ A > 0
def ball2notInBoxB := ∀ (b2: Nat) (B: Nat), b2 ≠ 2 ∧ B > 0

-- Theorems to be proved
theorem part1 (h1: nonEmptyBoxes) (h2: ball3inBoxB) : ∃ n, n = 12 := by sorry

theorem part2 (h3: ball1notInBoxA) (h4: ball2notInBoxB) : ∃ n, n = 36 := by sorry

end part1_part2_l53_53685


namespace algebraic_expression_identity_l53_53176

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l53_53176


namespace cost_of_500_pencils_in_dollars_l53_53213

def cost_of_pencil := 3 -- cost of 1 pencil in cents
def pencils_quantity := 500 -- number of pencils
def cents_in_dollar := 100 -- number of cents in 1 dollar

theorem cost_of_500_pencils_in_dollars :
  (pencils_quantity * cost_of_pencil) / cents_in_dollar = 15 := by
    sorry

end cost_of_500_pencils_in_dollars_l53_53213


namespace find_coordinates_of_A_l53_53293

theorem find_coordinates_of_A (x : ℝ) :
  let A := (x, 1, 2)
  let B := (2, 3, 4)
  (Real.sqrt ((x - 2)^2 + (1 - 3)^2 + (2 - 4)^2) = 2 * Real.sqrt 6) →
  (x = 6 ∨ x = -2) := 
by
  intros
  sorry

end find_coordinates_of_A_l53_53293


namespace generate_sequence_next_three_members_l53_53702

-- Define the function that generates the sequence
def f (n : ℕ) : ℕ := 2 * (n + 1) ^ 2 * (n + 2) ^ 2

-- Define the predicate that checks if a number can be expressed as the sum of squares of two positive integers
def is_sum_of_squares_of_two_positives (k : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = k

-- The problem statement to prove the equivalence
theorem generate_sequence_next_three_members :
  is_sum_of_squares_of_two_positives (f 1) ∧
  is_sum_of_squares_of_two_positives (f 2) ∧
  is_sum_of_squares_of_two_positives (f 3) ∧
  is_sum_of_squares_of_two_positives (f 4) ∧
  is_sum_of_squares_of_two_positives (f 5) ∧
  is_sum_of_squares_of_two_positives (f 6) ∧
  f 1 = 72 ∧
  f 2 = 288 ∧
  f 3 = 800 ∧
  f 4 = 1800 ∧
  f 5 = 3528 ∧
  f 6 = 6272 :=
sorry

end generate_sequence_next_three_members_l53_53702


namespace find_value_of_a_perpendicular_lines_l53_53307

theorem find_value_of_a_perpendicular_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x - 2 → y = 2 * x + 1 → 
  (a * 2 = -1)) → a = -1/2 :=
by
  sorry

end find_value_of_a_perpendicular_lines_l53_53307


namespace club_last_names_l53_53227

theorem club_last_names :
  ∃ A B C D E F : ℕ,
    A + B + C + D + E + F = 21 ∧
    A^2 + B^2 + C^2 + D^2 + E^2 + F^2 = 91 :=
by {
  sorry
}

end club_last_names_l53_53227


namespace hexagon_interior_angles_l53_53248

theorem hexagon_interior_angles
  (A B C D E F : ℝ)
  (hA : A = 90)
  (hB : B = 120)
  (hCD : C = D)
  (hE : E = 2 * C + 20)
  (hF : F = 60)
  (hsum : A + B + C + D + E + F = 720) :
  D = 107.5 := 
by
  -- formal proof required here
  sorry

end hexagon_interior_angles_l53_53248


namespace find_F_l53_53158

theorem find_F (F C : ℝ) (h1 : C = 30) (h2 : C = (5 / 9) * (F - 30)) : F = 84 := by
  sorry

end find_F_l53_53158


namespace largest_k_statement_l53_53660

noncomputable def largest_k (n : ℕ) : ℕ :=
  n - 2

theorem largest_k_statement (S : Finset ℕ) (A : Finset (Finset ℕ)) (h1 : ∀ (A_i : Finset ℕ), A_i ∈ A → 2 ≤ A_i.card ∧ A_i.card < S.card) : 
  largest_k S.card = S.card - 2 :=
by
  sorry

end largest_k_statement_l53_53660


namespace fraction_comparison_l53_53235

theorem fraction_comparison : (5555553 / 5555557 : ℚ) > (6666664 / 6666669 : ℚ) :=
  sorry

end fraction_comparison_l53_53235


namespace sum_of_valid_single_digit_z_l53_53767

theorem sum_of_valid_single_digit_z :
  let valid_z (z : ℕ) := z < 10 ∧ (16 + z) % 3 = 0
  let sum_z := (Finset.filter valid_z (Finset.range 10)).sum id
  sum_z = 15 :=
by
  -- Proof steps are omitted
  sorry

end sum_of_valid_single_digit_z_l53_53767


namespace light_travel_distance_in_km_l53_53705

-- Define the conditions
def speed_of_light_miles_per_sec : ℝ := 186282
def conversion_factor_mile_to_km : ℝ := 1.609
def time_seconds : ℕ := 500
def expected_distance_km : ℝ := 1.498 * 10^8

-- The theorem we need to prove
theorem light_travel_distance_in_km :
  (speed_of_light_miles_per_sec * time_seconds * conversion_factor_mile_to_km) = expected_distance_km :=
  sorry

end light_travel_distance_in_km_l53_53705


namespace smaller_number_l53_53395

theorem smaller_number (x y : ℝ) (h1 : y - x = (1 / 3) * y) (h2 : y = 71.99999999999999) : x = 48 :=
by
  sorry

end smaller_number_l53_53395


namespace factor_expression_l53_53770

theorem factor_expression (y : ℝ) : 84 * y ^ 13 + 210 * y ^ 26 = 42 * y ^ 13 * (2 + 5 * y ^ 13) :=
by sorry

end factor_expression_l53_53770


namespace intersection_closure_M_and_N_l53_53302

noncomputable def set_M : Set ℝ :=
  { x | 2 / x < 1 }

noncomputable def closure_M : Set ℝ :=
  Set.Icc 0 2

noncomputable def set_N : Set ℝ :=
  { y | ∃ x, y = Real.sqrt (x - 1) }

theorem intersection_closure_M_and_N :
  (closure_M ∩ set_N) = Set.Icc 0 2 :=
by
  sorry

end intersection_closure_M_and_N_l53_53302


namespace inequality_l53_53305

theorem inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) : 
  (a / b) + (b / c) + (c / a) + (b / a) + (a / c) + (c / b) + 6 ≥ 
  2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
sorry

end inequality_l53_53305


namespace bob_initial_pennies_l53_53657

-- Definitions of conditions
variables (a b : ℕ)
def condition1 : Prop := b + 2 = 4 * (a - 2)
def condition2 : Prop := b - 2 = 3 * (a + 2)

-- Goal: Proving that b = 62
theorem bob_initial_pennies (h1 : condition1 a b) (h2 : condition2 a b) : b = 62 :=
by {
  sorry
}

end bob_initial_pennies_l53_53657


namespace average_side_length_of_squares_l53_53351

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53351


namespace sum_of_fractions_l53_53963

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l53_53963


namespace math_enthusiast_gender_relation_female_success_probability_l53_53897

-- Constants and probabilities
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 12
def d : ℕ := 28
def n : ℕ := 100
def P_male_success : ℚ := 3 / 4
def P_female_success : ℚ := 2 / 3
def K_threshold : ℚ := 6.635

-- Computation of K^2
def K_square : ℚ := n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- The first part of the proof comparing K^2 with threshold
theorem math_enthusiast_gender_relation : K_square < K_threshold := sorry

-- The second part calculating given conditions for probability calculation
def P_A : ℚ := (P_male_success ^ 2 * (1 - P_female_success)) + (2 * (1 - P_male_success) * P_male_success * P_female_success)
def P_AB : ℚ := 2 * (1 - P_male_success) * P_male_success * P_female_success
def P_B_given_A : ℚ := P_AB / P_A

theorem female_success_probability : P_B_given_A = 4 / 7 := sorry

end math_enthusiast_gender_relation_female_success_probability_l53_53897


namespace odd_natural_of_form_l53_53450

/-- 
  Prove that the only odd natural number n in the form (p + q) / (p - q)
  where p and q are prime numbers and p > q is 5.
-/
theorem odd_natural_of_form (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p > q) 
  (h2 : ∃ n : ℕ, n = (p + q) / (p - q) ∧ n % 2 = 1) : ∃ n : ℕ, n = 5 :=
sorry

end odd_natural_of_form_l53_53450


namespace monkey_bananas_max_l53_53901

noncomputable def max_bananas_home : ℕ :=
  let total_bananas := 100
  let distance := 50
  let carry_capacity := 50
  let consumption_rate := 1
  let distance_each_way := distance / 2
  let bananas_eaten_each_way := distance_each_way * consumption_rate
  let bananas_left_midway := total_bananas / 2 - bananas_eaten_each_way
  let bananas_picked_midway := bananas_left_midway * 2
  let bananas_left_home := bananas_picked_midway - distance_each_way * consumption_rate
  bananas_left_home

theorem monkey_bananas_max : max_bananas_home = 25 :=
  sorry

end monkey_bananas_max_l53_53901


namespace claire_photos_eq_10_l53_53554

variable (C L R : Nat)

theorem claire_photos_eq_10
  (h1: L = 3 * C)
  (h2: R = C + 20)
  (h3: L = R)
  : C = 10 := by
  sorry

end claire_photos_eq_10_l53_53554


namespace sequence_general_term_l53_53300

noncomputable def a (n : ℕ) : ℝ :=
if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n ≠ 0) : 
  a n = if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1)) :=
by
  sorry

end sequence_general_term_l53_53300


namespace value_of_x_l53_53923

theorem value_of_x (x : ℝ) : abs (4 * x - 8) ≤ 0 ↔ x = 2 :=
by {
  sorry
}

end value_of_x_l53_53923


namespace expIConjugate_l53_53315

open Complex

-- Define the given condition
def expICondition (θ φ : ℝ) : Prop :=
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I

-- The theorem we want to prove
theorem expIConjugate (θ φ : ℝ) (h : expICondition θ φ) : 
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
sorry

end expIConjugate_l53_53315


namespace polynomial_remainder_l53_53826

theorem polynomial_remainder (a : ℝ) (h : ∀ x : ℝ, x^3 + a * x^2 + 1 = (x^2 - 1) * (x + 2) + (x + 3)) : a = 2 :=
sorry

end polynomial_remainder_l53_53826


namespace final_value_l53_53969

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l53_53969


namespace regular_polygon_perimeter_l53_53084

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53084


namespace probability_two_digit_between_15_25_l53_53337

-- Define a type for standard six-sided dice rolls
def is_standard_six_sided_die (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 6

-- Define the set of valid two-digit numbers
def valid_two_digit_number (n : ℕ) : Prop := n ≥ 15 ∧ n ≤ 25

-- Function to form a two-digit number from two dice rolls
def form_two_digit_number (d1 d2 : ℕ) : ℕ := 10 * d1 + d2

-- The main statement of the problem
theorem probability_two_digit_between_15_25 :
  (∃ (n : ℚ), n = 5/9) ∧
  (∀ (d1 d2 : ℕ), is_standard_six_sided_die d1 → is_standard_six_sided_die d2 →
  valid_two_digit_number (form_two_digit_number d1 d2)) :=
sorry

end probability_two_digit_between_15_25_l53_53337


namespace minimum_sum_l53_53860

theorem minimum_sum (a b c : ℕ) (h : a * b * c = 3006) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ 105 :=
sorry

end minimum_sum_l53_53860


namespace circle_radius_condition_l53_53797

theorem circle_radius_condition (c: ℝ):
  (∃ x y : ℝ, (x^2 + y^2 + 4 * x - 2 * y - 5 * c = 0)) → c > -1 :=
by
  sorry

end circle_radius_condition_l53_53797


namespace who_is_next_to_Boris_l53_53614

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l53_53614


namespace solution_set_abs_inequality_l53_53894

theorem solution_set_abs_inequality (x : ℝ) : |3 - x| + |x - 7| ≤ 8 ↔ 1 ≤ x ∧ x ≤ 9 :=
sorry

end solution_set_abs_inequality_l53_53894


namespace a_and_b_solution_l53_53514

noncomputable def solve_for_a_b (a b : ℕ) : Prop :=
  a > 0 ∧ (∀ b : ℤ, b > 0) ∧ (2 * a^b + 16 + 3 * a^b - 8) / 2 = 84 → a = 2 ∧ b = 5

theorem a_and_b_solution (a b : ℕ) (h : solve_for_a_b a b) : a = 2 ∧ b = 5 :=
sorry

end a_and_b_solution_l53_53514


namespace prob_teamY_wins_first_given_conditions_l53_53347

namespace TeamSeries

noncomputable def probability_teamY_wins_first_game
  (Y_wins_third : Prop)
  (X_wins_series : Prop)
  : ℝ := 5 / 12

theorem prob_teamY_wins_first_given_conditions :
  ∀ (first_to_four : ∀ (winsX winsY: ℕ), winsX = 4 ∨ winsY = 4)
    (equal_likely : ∀ (team : string), team = "X" ∨ team = "Y" → ℙ (win_single_game := 0.5))
    (no_ties : ∀ (team : string), team = "X" ∨ team = "Y" → win_single_game ≠ lose_single_game)
    (independent : ∀ (a b : Prop), indep a b)
    (Y_wins_third : Prop)
    (X_wins_series : Prop),
    probability_teamY_wins_first_game Y_wins_third X_wins_series = 5 / 12 :=
by sorry

end TeamSeries

end prob_teamY_wins_first_given_conditions_l53_53347


namespace lloyd_total_hours_worked_l53_53200

noncomputable def total_hours_worked (daily_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier: ℝ) (total_earnings : ℝ) : ℝ :=
  let regular_hours := 7.5
  let regular_pay := regular_hours * regular_rate
  if total_earnings <= regular_pay then daily_hours else
  let overtime_pay := total_earnings - regular_pay
  let overtime_hours := overtime_pay / (regular_rate * overtime_multiplier)
  regular_hours + overtime_hours

theorem lloyd_total_hours_worked :
  total_hours_worked 7.5 5.50 1.5 66 = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l53_53200


namespace area_of_shaded_region_l53_53322

theorem area_of_shaded_region 
  (ABCD : Type) 
  (BC : ℝ)
  (height : ℝ)
  (BE : ℝ)
  (CF : ℝ)
  (BC_length : BC = 12)
  (height_length : height = 10)
  (BE_length : BE = 5)
  (CF_length : CF = 3) :
  (BC * height - (1 / 2 * BE * height) - (1 / 2 * CF * height)) = 80 :=
by
  sorry

end area_of_shaded_region_l53_53322


namespace domain_of_f_intervals_of_increase_and_decrease_max_and_min_values_on_interval_l53_53673

noncomputable def f (x : ℝ) := real.sqrt (-x^2 + 5*x + 6)

theorem domain_of_f :
  {x | f x ≥ 0} = set.Icc (-1) 6 :=
begin
  sorry
end

theorem intervals_of_increase_and_decrease :
  {x | ∃ (l u : ℝ), l ≤ x ∧ x ≤ u ∧ (∀ y, l < y ∧ y < x → f y < f x) ∧ (∀ z, x < z ∧ z < u → f z > f x)} =
  set.Icc (-1) (5 / 2) ∪ set.Icc (5 / 2) 6 :=
begin
  sorry
end

theorem max_and_min_values_on_interval :
  ∃ (max_x min_x : ℝ),
    (1 ≤ max_x ∧ max_x ≤ 5 ∧ ∀ y, 1 ≤ y ∧ y ≤ 5 → f y ≤ f max_x)
    ∧
    (1 ≤ min_x ∧ min_x ≤ 5 ∧ ∀ z, 1 ≤ z ∧ z ≤ 5 → f z ≥ f min_x)
    ∧
    f max_x = 7 / 2
    ∧
    f min_x = real.sqrt 6 :=
begin
  sorry
end

end domain_of_f_intervals_of_increase_and_decrease_max_and_min_values_on_interval_l53_53673


namespace product_of_repeating_decimals_l53_53459

noncomputable def repeating_decimal_038 : ℚ := 38 / 999
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem product_of_repeating_decimals :
  repeating_decimal_038 * repeating_decimal_4 = 152 / 8991 :=
by
  sorry

end product_of_repeating_decimals_l53_53459


namespace regular_polygon_perimeter_l53_53085

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53085


namespace math_problem_l53_53994

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l53_53994


namespace problem_solution_l53_53764

noncomputable def given_problem : ℝ := (Real.pi - 3)^0 - Real.sqrt 8 + 2 * Real.sin (45 * Real.pi / 180) + (1 / 2)⁻¹

theorem problem_solution : given_problem = 3 - Real.sqrt 2 := by
  sorry

end problem_solution_l53_53764


namespace average_side_length_of_squares_l53_53384

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l53_53384


namespace quadratic_function_a_equals_one_l53_53939

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_function_a_equals_one
  (a b c : ℝ)
  (h1 : 1 < x)
  (h2 : x < c)
  (h_neg : ∀ x, 1 < x → x < c → quadratic_function a b c x < 0):
  a = 1 := by
  sorry

end quadratic_function_a_equals_one_l53_53939


namespace regular_polygon_perimeter_is_28_l53_53069

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53069


namespace Amy_initial_cupcakes_l53_53258

def initialCupcakes (packages : ℕ) (cupcakesPerPackage : ℕ) (eaten : ℕ) : ℕ :=
  packages * cupcakesPerPackage + eaten

theorem Amy_initial_cupcakes :
  let packages := 9
  let cupcakesPerPackage := 5
  let eaten := 5
  initialCupcakes packages cupcakesPerPackage eaten = 50 :=
by
  sorry

end Amy_initial_cupcakes_l53_53258


namespace variance_of_planted_trees_l53_53681

def number_of_groups := 10

def planted_trees : List ℕ := [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / (xs.length : ℚ)

noncomputable def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / (xs.length : ℚ)

theorem variance_of_planted_trees :
  variance planted_trees = 0.6 := sorry

end variance_of_planted_trees_l53_53681


namespace closest_time_to_1600_mirror_l53_53621

noncomputable def clock_in_mirror_time (hour_hand_minute: ℕ) (minute_hand_minute: ℕ) : (ℕ × ℕ) :=
  let hour_in_mirror := (12 - hour_hand_minute) % 12
  let minute_in_mirror := minute_hand_minute
  (hour_in_mirror, minute_in_mirror)

theorem closest_time_to_1600_mirror (A B C D : (ℕ × ℕ)) :
  clock_in_mirror_time 4 0 = D → D = (8, 0) :=
by
  -- Introduction of hypothesis that clock closest to 16:00 (4:00) is represented by D
  intro h
  -- State the conclusion based on the given hypothesis
  sorry

end closest_time_to_1600_mirror_l53_53621


namespace cupric_cyanide_formed_l53_53309

-- Definition of the problem
def formonitrile : ℕ := 6
def copper_sulfate : ℕ := 3
def sulfuric_acid : ℕ := 3

-- Stoichiometry from the balanced equation
def stoichiometry (hcn mol_multiplier: ℕ): ℕ := 
  (hcn / mol_multiplier)

theorem cupric_cyanide_formed :
  stoichiometry formonitrile 2 = 3 := 
sorry

end cupric_cyanide_formed_l53_53309


namespace expression_value_l53_53955

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l53_53955


namespace geom_prog_common_ratio_unique_l53_53577

theorem geom_prog_common_ratio_unique (b q : ℝ) (hb : b > 0) (hq : q > 1) :
  (∃ b : ℝ, (q = (1 + Real.sqrt 5) / 2) ∧ 
    (0 < b ∧ b * q ≠ b ∧ b * q^2 ≠ b ∧ b * q^3 ≠ b) ∧ 
    ((2 * b * q = b + b * q^2) ∨ (2 * b * q = b + b * q^3) ∨ (2 * b * q^2 = b + b * q^3))) := 
sorry

end geom_prog_common_ratio_unique_l53_53577


namespace problem_statement_l53_53983

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l53_53983


namespace fuel_consumption_l53_53841

def initial_volume : ℕ := 3000
def volume_jan_1 : ℕ := 180
def volume_may_1 : ℕ := 1238
def refill_volume : ℕ := 3000

theorem fuel_consumption :
  (initial_volume - volume_jan_1) + (refill_volume - volume_may_1) = 4582 := by
  sorry

end fuel_consumption_l53_53841


namespace convex_polygon_with_arith_prog_angles_l53_53400

theorem convex_polygon_with_arith_prog_angles 
  (n : ℕ) 
  (angles : Fin n → ℝ)
  (is_convex : ∀ i, angles i < 180)
  (arithmetic_progression : ∃ a d, d = 3 ∧ ∀ i, angles i = a + i * d)
  (largest_angle : ∃ i, angles i = 150)
  : n = 24 :=
sorry

end convex_polygon_with_arith_prog_angles_l53_53400


namespace eval_series_l53_53457

theorem eval_series : ∑ k in (Set.Ici 1), (k ^ 3) / (3^k : ℝ) = (39 / 8 : ℝ) :=
by
  sorry

end eval_series_l53_53457


namespace packets_of_gum_is_eight_l53_53201

-- Given conditions
def pieces_left : ℕ := 2
def pieces_chewed : ℕ := 54
def pieces_per_packet : ℕ := 7

-- Given he chews all the gum except for pieces_left pieces, and chews pieces_chewed pieces at once
def total_pieces_of_gum (pieces_chewed pieces_left : ℕ) : ℕ :=
  pieces_chewed + pieces_left

-- Calculate the number of packets
def number_of_packets (total_pieces pieces_per_packet : ℕ) : ℕ :=
  total_pieces / pieces_per_packet

-- The final theorem asserting the number of packets is 8
theorem packets_of_gum_is_eight : number_of_packets (total_pieces_of_gum pieces_chewed pieces_left) pieces_per_packet = 8 :=
  sorry

end packets_of_gum_is_eight_l53_53201


namespace log_expression_value_l53_53556

theorem log_expression_value :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 :=
by
  sorry

end log_expression_value_l53_53556


namespace overall_loss_amount_l53_53435

theorem overall_loss_amount 
    (S : ℝ)
    (hS : S = 12499.99)
    (profit_percent : ℝ)
    (loss_percent : ℝ)
    (sold_at_profit : ℝ)
    (sold_at_loss : ℝ) 
    (condition1 : profit_percent = 0.2)
    (condition2 : loss_percent = -0.1)
    (condition3 : sold_at_profit = 0.2 * S * (1 + profit_percent))
    (condition4 : sold_at_loss = 0.8 * S * (1 + loss_percent))
    :
    S - (sold_at_profit + sold_at_loss) = 500 := 
by 
  sorry

end overall_loss_amount_l53_53435


namespace find_b_c_find_a_range_l53_53247

noncomputable def f (a b c x : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + b * x + c
noncomputable def g (a b c x : ℝ) : ℝ := f a b c x + 2 * x
noncomputable def f_prime (a b x : ℝ) : ℝ := x^2 - a * x + b
noncomputable def g_prime (a b x : ℝ) : ℝ := f_prime a b x + 2

theorem find_b_c (a c : ℝ) (h_f0 : f a 0 c 0 = c) (h_tangent_y_eq_1 : 1 = c) : 
  b = 0 ∧ c = 1 :=
by
  sorry

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, g_prime a 0 x ≥ 0) ↔ a ≤ 2 * Real.sqrt 2 :=
by
  sorry

end find_b_c_find_a_range_l53_53247


namespace problem_statement_l53_53978

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l53_53978


namespace expression_value_l53_53956

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l53_53956


namespace regular_polygon_perimeter_l53_53080

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53080


namespace product_of_two_numbers_l53_53546

theorem product_of_two_numbers (a b : ℕ) (H1 : Nat.gcd a b = 20) (H2 : Nat.lcm a b = 128) : a * b = 2560 :=
by
  sorry

end product_of_two_numbers_l53_53546


namespace geometric_sequence_a3_eq_sqrt_5_l53_53292

theorem geometric_sequence_a3_eq_sqrt_5 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * r)
  (h_a1 : a 1 = 1) (h_a5 : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end geometric_sequence_a3_eq_sqrt_5_l53_53292


namespace tables_in_conference_hall_l53_53815

theorem tables_in_conference_hall (c t : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : 4 * c + 4 * t = 648) : 
  t = 18 :=
by sorry

end tables_in_conference_hall_l53_53815


namespace minimum_value_expression_l53_53640

noncomputable def minimum_value (a b : ℝ) := (1 / (2 * |a|)) + (|a| / b)

theorem minimum_value_expression
  (a : ℝ) (b : ℝ) (h1 : a + b = 2) (h2 : b > 0) :
  ∃ (min_val : ℝ), min_val = 3 / 4 ∧ ∀ (a b : ℝ), a + b = 2 → b > 0 → minimum_value a b ≥ min_val :=
sorry

end minimum_value_expression_l53_53640


namespace percent_increase_from_may_to_june_l53_53403

noncomputable def profit_increase_from_march_to_april (P : ℝ) : ℝ := 1.30 * P
noncomputable def profit_decrease_from_april_to_may (P : ℝ) : ℝ := 1.04 * P
noncomputable def profit_increase_from_march_to_june (P : ℝ) : ℝ := 1.56 * P

theorem percent_increase_from_may_to_june (P : ℝ) :
  (1.04 * P * (1 + 0.50)) = 1.56 * P :=
by
  sorry

end percent_increase_from_may_to_june_l53_53403


namespace integer_solution_count_l53_53857

theorem integer_solution_count (x : ℤ) : (12 * x - 1) * (6 * x - 1) * (4 * x - 1) * (3 * x - 1) = 330 ↔ x = 1 :=
by
  sorry

end integer_solution_count_l53_53857


namespace persons_next_to_Boris_l53_53601

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l53_53601


namespace find_fixed_tangent_circle_l53_53306

open Real

def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def orthogonal (O P Q : Point) : Prop := (P.x * Q.x + P.y * Q.y = 0)

def tangentCircleExists (r : ℝ) (a b : Point) (M uₜ l_m k: Prop) : Prop :=
∃ (c : Circle), c.radius^2 = r*12 / 7 ∧ ∀ (P Q: Point), c tangent l_m

theorem find_fixed_tangent_circle :
  (ellipse 2 (sqrt 3) x y) ∧ (∃ r:ℝ, r > 0) →
  tangentCircleExists (sqrt (12 / 7)) A B M l_m k :=
sorry -- No proof needed

end find_fixed_tangent_circle_l53_53306


namespace time_A_problems_60_l53_53182

variable (t : ℕ) -- time in minutes per type B problem

def time_per_A_problem := 2 * t
def time_per_C_problem := t / 2
def total_time_for_A_problems := 20 * time_per_A_problem

theorem time_A_problems_60 (hC : 80 * time_per_C_problem = 60) : total_time_for_A_problems = 60 := by
  sorry

end time_A_problems_60_l53_53182


namespace sum_first_3000_terms_l53_53864

variable {α : Type*}

noncomputable def geometric_sum_1000 (a r : α) [Field α] : α := a * (r ^ 1000 - 1) / (r - 1)
noncomputable def geometric_sum_2000 (a r : α) [Field α] : α := a * (r ^ 2000 - 1) / (r - 1)
noncomputable def geometric_sum_3000 (a r : α) [Field α] : α := a * (r ^ 3000 - 1) / (r - 1)

theorem sum_first_3000_terms 
  {a r : ℝ}
  (h1 : geometric_sum_1000 a r = 1024)
  (h2 : geometric_sum_2000 a r = 2040) :
  geometric_sum_3000 a r = 3048 := 
  sorry

end sum_first_3000_terms_l53_53864


namespace computation_of_expression_l53_53175

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l53_53175


namespace product_increase_l53_53714

theorem product_increase (a b : ℝ) (h : (a + 1) * (b + 1) = 2 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a^2 - 1) * (b^2 - 1) = 4 * a * b) :=
sorry

end product_increase_l53_53714


namespace average_side_length_of_squares_l53_53379

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53379


namespace triangle_inequality_part_a_l53_53420

theorem triangle_inequality_part_a (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  a^2 + b^2 + c^2 + a * b * c < 8 :=
sorry

end triangle_inequality_part_a_l53_53420


namespace total_amount_paid_l53_53869

noncomputable def cost_per_night_per_person : ℕ := 40
noncomputable def number_of_people : ℕ := 3
noncomputable def number_of_nights : ℕ := 3

theorem total_amount_paid (cost_per_night_per_person number_of_people number_of_nights : ℕ) :
  (cost_per_night_per_person * number_of_people * number_of_nights = 360) :=
by
  have h : cost_per_night_per_person * number_of_people * number_of_nights = 40 * 3 * 3 := by rfl
  rw h
  exact rfl

end total_amount_paid_l53_53869


namespace range_of_y_given_x_l53_53140

theorem range_of_y_given_x (x : ℝ) (h₁ : x > 3) : 0 < (6 / x) ∧ (6 / x) < 2 :=
by 
  sorry

end range_of_y_given_x_l53_53140


namespace perpendicular_pair_is_14_l53_53941

variable (x y : ℝ)

def equation1 := 4 * y - 3 * x = 16
def equation2 := -3 * x - 4 * y = 15
def equation3 := 4 * y + 3 * x = 16
def equation4 := 3 * y + 4 * x = 15

theorem perpendicular_pair_is_14 : (∃ y1 y2 x1 x2 : ℝ,
  4 * y1 - 3 * x1 = 16 ∧ 3 * y2 + 4 * x2 = 15 ∧ (3 / 4) * (-4 / 3) = -1) :=
sorry

end perpendicular_pair_is_14_l53_53941


namespace solutionToSystemOfEquations_solutionToSystemOfInequalities_l53_53244

open Classical

noncomputable def solveSystemOfEquations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solutionToSystemOfEquations : ∃ (x y : ℝ), solveSystemOfEquations x y ∧ x = 4 ∧ y = 5 := by
  sorry

def solveSystemOfInequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solutionToSystemOfInequalities : ∃ x : ℝ, solveSystemOfInequalities x ∧ 1 ≤ x ∧ x < 2 := by
  sorry

end solutionToSystemOfEquations_solutionToSystemOfInequalities_l53_53244


namespace find_a_for_odd_function_l53_53334

theorem find_a_for_odd_function (f : ℝ → ℝ) (a : ℝ) (h₀ : ∀ x, f (-x) = -f x) (h₁ : ∀ x, x < 0 → f x = x^2 + a * x) (h₂ : f 3 = 6) : a = 5 :=
by
  sorry

end find_a_for_odd_function_l53_53334


namespace expected_value_of_shorter_gentlemen_correct_l53_53759
noncomputable def expected_value_of_shorter_gentlemen (n : ℕ) : ℝ :=
  ∑ j in Finset.range n, (j : ℝ) / n

theorem expected_value_of_shorter_gentlemen_correct (n : ℕ) :
  expected_value_of_shorter_gentlemen (n + 1) = n / 2 :=
by
  sorry

end expected_value_of_shorter_gentlemen_correct_l53_53759


namespace cannot_divide_1980_into_four_groups_l53_53892

theorem cannot_divide_1980_into_four_groups :
  ¬∃ (S₁ S₂ S₃ S₄ : ℕ),
    S₂ = S₁ + 10 ∧
    S₃ = S₂ + 10 ∧
    S₄ = S₃ + 10 ∧
    (1 + 1980) * 1980 / 2 = S₁ + S₂ + S₃ + S₄ := 
sorry

end cannot_divide_1980_into_four_groups_l53_53892


namespace regular_polygon_perimeter_l53_53086

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53086


namespace sqrt_one_sixty_four_l53_53531

theorem sqrt_one_sixty_four : Real.sqrt (1 / 64) = 1 / 8 :=
sorry

end sqrt_one_sixty_four_l53_53531


namespace gcd_cubic_l53_53634

theorem gcd_cubic (n : ℕ) (h1 : n > 9) :
  let k := gcd (n^3 + 25) (n + 3)
  in if (n + 3) % 2 = 1 then k = 1 else k = 2 :=
by
  sorry

end gcd_cubic_l53_53634


namespace average_of_side_lengths_of_squares_l53_53392

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l53_53392


namespace can_form_isosceles_triangle_with_given_sides_l53_53722

-- Define a structure for the sides of a triangle
structure Triangle (α : Type _) :=
  (a b c : α)

-- Define the predicate for the triangle inequality
def triangle_inequality {α : Type _} [LinearOrder α] [Add α] (t : Triangle α) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

-- Define the predicate for an isosceles triangle
def is_isosceles {α : Type _} [DecidableEq α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the main theorem which checks if the given sides can form an isosceles triangle
theorem can_form_isosceles_triangle_with_given_sides
  (t : Triangle ℕ)
  (h_tri : triangle_inequality t)
  (h_iso : is_isosceles t) :
  t = ⟨2, 2, 1⟩ :=
  sorry

end can_form_isosceles_triangle_with_given_sides_l53_53722


namespace ratio_shorter_to_longer_l53_53326

theorem ratio_shorter_to_longer (x y : ℝ) (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = y / 3) : x / y = 5 / 12 :=
sorry

end ratio_shorter_to_longer_l53_53326


namespace regular_polygon_perimeter_l53_53046

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53046


namespace difference_q_r_l53_53259

theorem difference_q_r (x : ℝ) (p q r : ℝ) 
  (h1 : 7 * x - 3 * x = 3600) 
  (h2 : q = 7 * x) 
  (h3 : r = 12 * x) :
  r - q = 4500 := 
sorry

end difference_q_r_l53_53259


namespace tens_digit_of_6_pow_4_is_9_l53_53019

theorem tens_digit_of_6_pow_4_is_9 : (6 ^ 4 / 10) % 10 = 9 :=
by
  sorry

end tens_digit_of_6_pow_4_is_9_l53_53019


namespace fraction_black_part_l53_53251

theorem fraction_black_part (L : ℝ) (blue_part : ℝ) (white_part_fraction : ℝ) 
  (h1 : L = 8) (h2 : blue_part = 3.5) (h3 : white_part_fraction = 0.5) : 
  (8 - (3.5 + 0.5 * (8 - 3.5))) / 8 = 9 / 32 :=
by
  sorry

end fraction_black_part_l53_53251


namespace dorothy_and_jemma_sales_l53_53274

theorem dorothy_and_jemma_sales :
  ∀ (frames_sold_by_jemma price_per_frame_jemma : ℕ)
  (price_per_frame_dorothy frames_sold_by_dorothy : ℚ)
  (total_sales_jemma total_sales_dorothy total_sales : ℚ),
  price_per_frame_jemma = 5 →
  frames_sold_by_jemma = 400 →
  price_per_frame_dorothy = price_per_frame_jemma / 2 →
  frames_sold_by_jemma = 2 * frames_sold_by_dorothy →
  total_sales_jemma = frames_sold_by_jemma * price_per_frame_jemma →
  total_sales_dorothy = frames_sold_by_dorothy * price_per_frame_dorothy →
  total_sales = total_sales_jemma + total_sales_dorothy →
  total_sales = 2500 := by
  sorry

end dorothy_and_jemma_sales_l53_53274


namespace cause_of_polarization_by_electronegativity_l53_53344

-- Definition of the problem conditions as hypotheses
def strong_polarization_of_CH_bond (C_H_bond : Prop) (electronegativity : Prop) : Prop 
  := C_H_bond ∧ electronegativity

-- Given conditions: Carbon atom is in sp hybridization and C-H bond shows strong polarization
axiom carbon_sp_hybridized : Prop
axiom CH_bond_strong_polarization : Prop

-- Question: The cause of strong polarization of the C-H bond at the carbon atom in sp hybridization in alkynes
def cause_of_strong_polarization (sp_hybridization : Prop) : Prop 
  := true  -- This definition will hold as a placeholder, to indicate there is a causal connection

-- Correct answer: high electronegativity of the carbon atom in sp-hybrid state causes strong polarization
theorem cause_of_polarization_by_electronegativity 
  (high_electronegativity : Prop) 
  (sp_hybridized : Prop) 
  (polarized : Prop) 
  (H : strong_polarization_of_CH_bond polarized high_electronegativity) 
  : sp_hybridized ∧ polarized := 
  sorry

end cause_of_polarization_by_electronegativity_l53_53344


namespace probability_A_not_lose_l53_53500

-- Define the probabilities
def P_A_wins : ℝ := 0.30
def P_draw : ℝ := 0.25
def P_A_not_lose : ℝ := 0.55

-- Statement to prove
theorem probability_A_not_lose : P_A_wins + P_draw = P_A_not_lose :=
by 
  sorry

end probability_A_not_lose_l53_53500


namespace boris_neighbors_l53_53583

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l53_53583


namespace last_two_digits_of_large_exponent_l53_53549

theorem last_two_digits_of_large_exponent :
  (9 ^ (8 ^ (7 ^ (6 ^ (5 ^ (4 ^ (3 ^ 2))))))) % 100 = 21 :=
by
  sorry

end last_two_digits_of_large_exponent_l53_53549


namespace distance_between_hyperbola_vertices_l53_53773

theorem distance_between_hyperbola_vertices :
  (∀ x y : ℝ, (x^2 / 121) - (y^2 / 49) = 1) → 
  (∃ d : ℝ, d = 22) :=
by
  -- Assume the equation of the hyperbola
  intro hyp_eq,
  -- Use the provided information and conditions
  let a := Float.sqrt 121,
  -- The distance between the vertices is 2a
  have dist := 2 * a,
  -- Simplify a as sqrt(121) = 11
  have a_eq_11 : a = 11,
  -- Thus, distance is 2 * 11 = 22
  have dist_22 : dist = 22,
  use dist_22,
  sorry

end distance_between_hyperbola_vertices_l53_53773


namespace cistern_total_wet_surface_area_l53_53739

/-- Given a cistern with length 6 meters, width 4 meters, and water depth 1.25 meters,
    the total area of the wet surface is 49 square meters. -/
theorem cistern_total_wet_surface_area
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 6) (h_width : width = 4) (h_depth : depth = 1.25) :
  (length * width) + 2 * (length * depth) + 2 * (width * depth) = 49 :=
by {
  -- Proof goes here
  sorry
}

end cistern_total_wet_surface_area_l53_53739


namespace base_conversion_problem_l53_53004

variable (A C : ℕ)
variable (h1 : 0 ≤ A ∧ A < 8)
variable (h2 : 0 ≤ C ∧ C < 5)

theorem base_conversion_problem (h : 8 * A + C = 5 * C + A) : 8 * A + C = 39 := 
sorry

end base_conversion_problem_l53_53004


namespace maria_needs_green_beans_l53_53835

theorem maria_needs_green_beans :
  ∀ (potatoes carrots onions green_beans : ℕ), 
  (carrots = 6 * potatoes) →
  (onions = 2 * carrots) →
  (green_beans = onions / 3) →
  (potatoes = 2) →
  green_beans = 8 :=
by
  intros potatoes carrots onions green_beans h1 h2 h3 h4
  rw [h4, Nat.mul_comm 6 2] at h1
  rw [h1, Nat.mul_comm 2 12] at h2
  rw [h2] at h3
  sorry

end maria_needs_green_beans_l53_53835


namespace distance_between_hyperbola_vertices_l53_53772

theorem distance_between_hyperbola_vertices :
  (∀ x y : ℝ, (x^2 / 121) - (y^2 / 49) = 1) → 
  (∃ d : ℝ, d = 22) :=
by
  -- Assume the equation of the hyperbola
  intro hyp_eq,
  -- Use the provided information and conditions
  let a := Float.sqrt 121,
  -- The distance between the vertices is 2a
  have dist := 2 * a,
  -- Simplify a as sqrt(121) = 11
  have a_eq_11 : a = 11,
  -- Thus, distance is 2 * 11 = 22
  have dist_22 : dist = 22,
  use dist_22,
  sorry

end distance_between_hyperbola_vertices_l53_53772


namespace total_yarn_length_is_1252_l53_53856

/-- Defining the lengths of the yarns according to the conditions --/
def green_yarn : ℕ := 156
def red_yarn : ℕ := 3 * green_yarn + 8
def blue_yarn : ℕ := (green_yarn + red_yarn) / 2
def average_yarn_length : ℕ := (green_yarn + red_yarn + blue_yarn) / 3
def yellow_yarn : ℕ := average_yarn_length - 12

/-- Proving the total length of the four pieces of yarn is 1252 cm --/
theorem total_yarn_length_is_1252 :
  green_yarn + red_yarn + blue_yarn + yellow_yarn = 1252 := by
  sorry

end total_yarn_length_is_1252_l53_53856


namespace geom_seq_log_eqn_l53_53297

theorem geom_seq_log_eqn {a : ℕ → ℝ} {b : ℕ → ℝ}
    (geom_seq : ∃ (r : ℝ) (a1 : ℝ), ∀ n : ℕ, a (n + 1) = a1 * r^n)
    (log_seq : ∀ n : ℕ, b n = Real.log (a (n + 1)) / Real.log 2)
    (b_eqn : b 1 + b 3 = 4) : a 2 = 4 :=
by
  sorry

end geom_seq_log_eqn_l53_53297


namespace option_a_is_correct_l53_53137

variable (a b : ℝ)
variable (ha : a < 0)
variable (hb : b < 0)
variable (hab : a < b)

theorem option_a_is_correct : (a < abs (3 * a + 2 * b) / 5) ∧ (abs (3 * a + 2 * b) / 5 < b) :=
by
  sorry

end option_a_is_correct_l53_53137


namespace weaving_increase_l53_53324

theorem weaving_increase (a₁ : ℕ) (S₃₀ : ℕ) (d : ℚ) (hₐ₁ : a₁ = 5) (hₛ₃₀ : S₃₀ = 390)
  (h_sum : S₃₀ = 30 * (a₁ + (a₁ + 29 * d)) / 2) : d = 16 / 29 :=
by {
  sorry
}

end weaving_increase_l53_53324


namespace stone_reaches_bottom_l53_53573

structure StoneInWater where
  σ : ℝ   -- Density of stone in g/cm³
  d : ℝ   -- Depth of lake in cm
  g : ℝ   -- Acceleration due to gravity in cm/sec²
  σ₁ : ℝ  -- Density of water in g/cm³

noncomputable def time_and_velocity (siw : StoneInWater) : ℝ × ℝ :=
  let g₁ := ((siw.σ - siw.σ₁) / siw.σ) * siw.g
  let t := Real.sqrt ((2 * siw.d) / g₁)
  let v := g₁ * t
  (t, v)

theorem stone_reaches_bottom (siw : StoneInWater)
  (hσ : siw.σ = 2.1)
  (hd : siw.d = 850)
  (hg : siw.g = 980.8)
  (hσ₁ : siw.σ₁ = 1.0) :
  time_and_velocity siw = (1.82, 935) :=
by
  sorry

end stone_reaches_bottom_l53_53573


namespace range_of_a_l53_53810

theorem range_of_a (a x : ℝ) (h_eq : 2 * x - 1 = x + a) (h_pos : x > 0) : a > -1 :=
sorry

end range_of_a_l53_53810


namespace number_of_lines_through_focus_intersecting_hyperbola_l53_53950

open Set

noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 2) - y^2 = 1

-- The coordinates of the focuses of the hyperbola
def right_focus : ℝ × ℝ := (2, 0)

-- Definition to express that a line passes through the right focus
def line_through_focus (l : ℝ → ℝ) : Prop := l 2 = 0

-- Definition for the length of segment AB being 4
def length_AB_is_4 (A B : ℝ × ℝ) : Prop := dist A B = 4

-- The statement asserting the number of lines satisfying the given condition
theorem number_of_lines_through_focus_intersecting_hyperbola:
  ∃ (n : ℕ), n = 3 ∧ ∀ (l : ℝ → ℝ),
  line_through_focus l →
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ length_AB_is_4 A B :=
sorry

end number_of_lines_through_focus_intersecting_hyperbola_l53_53950


namespace average_salary_is_8000_l53_53891

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

def average_salary : ℕ := total_salary / num_people

theorem average_salary_is_8000 : average_salary = 8000 := by
  sorry

end average_salary_is_8000_l53_53891


namespace min_value_condition_l53_53010

variable (a b : ℝ)

theorem min_value_condition (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 2) :
    (1 / a^2) + (1 / b^2) = 9 / 2 :=
sorry

end min_value_condition_l53_53010


namespace loss_percentage_initially_l53_53908

theorem loss_percentage_initially 
  (SP : ℝ) 
  (CP : ℝ := 400) 
  (h1 : SP + 100 = 1.05 * CP) : 
  (1 - SP / CP) * 100 = 20 := 
by 
  sorry

end loss_percentage_initially_l53_53908


namespace find_N_l53_53542

theorem find_N (a b c N : ℚ) (h_sum : a + b + c = 84)
    (h_a : a - 7 = N) (h_b : b + 7 = N) (h_c : c / 7 = N) : 
    N = 28 / 3 :=
sorry

end find_N_l53_53542


namespace jeans_price_increase_l53_53740

theorem jeans_price_increase
  (C R P : ℝ)
  (h1 : P = 1.15 * R)
  (h2 : P = 1.6100000000000001 * C) :
  R = 1.4 * C :=
by
  sorry

end jeans_price_increase_l53_53740


namespace boris_neighbors_l53_53579

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l53_53579


namespace coconut_to_almond_ratio_l53_53668

-- Conditions
def number_of_coconut_candles (C : ℕ) : Prop :=
  ∃ L A : ℕ, L = 2 * C ∧ A = 10

-- Question
theorem coconut_to_almond_ratio (C : ℕ) (h : number_of_coconut_candles C) :
  ∃ r : ℚ, r = C / 10 := by
  sorry

end coconut_to_almond_ratio_l53_53668


namespace solution_set_f_x_minus_2_pos_l53_53827

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then 2 * x - 4 else 2 * (-x) - 4

theorem solution_set_f_x_minus_2_pos :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_x_minus_2_pos_l53_53827


namespace average_side_length_of_squares_l53_53377

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53377


namespace min_value_ab_l53_53951

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a / 2) + b = 1) :
  (1 / a) + (1 / b) = (3 / 2) + Real.sqrt 2 :=
by sorry

end min_value_ab_l53_53951


namespace calculate_expression_l53_53120

theorem calculate_expression : 12 * (1 / (2 / 3 - 1 / 4 + 1 / 6)) = 144 / 7 :=
by
  sorry

end calculate_expression_l53_53120


namespace tuition_fee_l53_53015

theorem tuition_fee (R T : ℝ) (h1 : T + R = 2584) (h2 : T = R + 704) : T = 1644 := by sorry

end tuition_fee_l53_53015


namespace solve_equation_l53_53848

theorem solve_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) :
  (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 :=
sorry

end solve_equation_l53_53848


namespace alice_bob_not_both_l53_53638

-- Define the group of 8 students
def total_students : ℕ := 8

-- Define the committee size
def committee_size : ℕ := 5

-- Calculate the total number of unrestricted committees
def total_committees : ℕ := Nat.choose total_students committee_size

-- Calculate the number of committees where both Alice and Bob are included
def alice_bob_committees : ℕ := Nat.choose (total_students - 2) (committee_size - 2)

-- Calculate the number of committees where Alice and Bob are not both included
def not_both_alice_bob : ℕ := total_committees - alice_bob_committees

-- Now state the theorem we want to prove
theorem alice_bob_not_both : not_both_alice_bob = 36 :=
by
  sorry

end alice_bob_not_both_l53_53638


namespace sum_of_series_l53_53876

theorem sum_of_series (h1 : 2 + 4 + 6 + 8 + 10 = 30) (h2 : 1 + 3 + 5 + 7 + 9 = 25) : 
  ((2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9)) + ((1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10)) = 61 / 30 := by
  sorry

end sum_of_series_l53_53876


namespace standing_next_to_boris_l53_53605

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l53_53605


namespace regular_polygon_perimeter_l53_53104

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53104


namespace percent_of_ac_is_db_l53_53495

variable (a b c d : ℝ)

-- Given conditions
variable (h1 : c = 0.25 * a)
variable (h2 : c = 0.10 * b)
variable (h3 : d = 0.50 * b)

-- Theorem statement: Prove the final percentage
theorem percent_of_ac_is_db : (d * b) / (a * c) * 100 = 1250 :=
by
  sorry

end percent_of_ac_is_db_l53_53495


namespace derivative_of_gx_eq_3x2_l53_53394

theorem derivative_of_gx_eq_3x2 (f : ℝ → ℝ) : (∀ x : ℝ, f x = (x + 1) * (x^2 - x + 1)) → (∀ x : ℝ, deriv f x = 3 * x^2) :=
by
  intro h
  sorry

end derivative_of_gx_eq_3x2_l53_53394


namespace line_equation_l53_53463

theorem line_equation {m : ℤ} :
  (∀ x y : ℤ, 2 * x + y + m = 0) →
  (∀ x y : ℤ, 2 * x + y - 10 = 0) →
  (2 * 1 + 0 + m = 0) →
  m = -2 :=
by
  sorry

end line_equation_l53_53463


namespace initial_oranges_l53_53122

variable (x : ℕ)
variable (total_oranges : ℕ := 8)
variable (oranges_from_joyce : ℕ := 3)

theorem initial_oranges (h : total_oranges = x + oranges_from_joyce) : x = 5 := by
  sorry

end initial_oranges_l53_53122


namespace fill_tank_with_only_C_l53_53900

noncomputable def time_to_fill_with_only_C (x y z : ℝ) : ℝ := 
  let eq1 := (1 / z - 1 / x) * 2 = 1
  let eq2 := (1 / z - 1 / y) * 4 = 1
  let eq3 := 1 / z * 5 - (1 / x + 1 / y) * 8 = 0
  z

theorem fill_tank_with_only_C (x y z : ℝ) (h1 : (1 / z - 1 / x) * 2 = 1) 
  (h2 : (1 / z - 1 / y) * 4 = 1) (h3 : 1 / z * 5 - (1 / x + 1 / y) * 8 = 0) : 
  time_to_fill_with_only_C x y z = 11 / 6 :=
by
  sorry

end fill_tank_with_only_C_l53_53900


namespace expression_value_l53_53954

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l53_53954


namespace solve_weight_of_bowling_ball_l53_53452

-- Conditions: Eight bowling balls equal the weight of five canoes
-- and four canoes weigh 120 pounds.

def weight_of_bowling_ball : Prop :=
  ∃ (b c : ℝ), (8 * b = 5 * c) ∧ (4 * c = 120) ∧ (b = 18.75)

theorem solve_weight_of_bowling_ball : weight_of_bowling_ball :=
  sorry

end solve_weight_of_bowling_ball_l53_53452


namespace find_x_parallel_l53_53472

theorem find_x_parallel (x : ℝ) 
  (a : ℝ × ℝ := (x, 2)) 
  (b : ℝ × ℝ := (2, 4)) 
  (h : a.1 * b.2 = a.2 * b.1) :
  x = 1 := 
by
  sorry

end find_x_parallel_l53_53472


namespace evaluate_poly_at_2_l53_53833

def my_op (x y : ℕ) : ℕ := (x + 1) * (y + 1)
def star2 (x : ℕ) : ℕ := my_op x x

theorem evaluate_poly_at_2 :
  3 * (star2 2) - 2 * 2 + 1 = 24 :=
by
  sorry

end evaluate_poly_at_2_l53_53833


namespace expression_value_l53_53998

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l53_53998


namespace area_of_inscribed_square_l53_53738

noncomputable def circle_eq (x y : ℝ) : Prop := 
  3*x^2 + 3*y^2 - 15*x + 9*y + 27 = 0

theorem area_of_inscribed_square :
  (∃ x y : ℝ, circle_eq x y) →
  ∃ s : ℝ, s^2 = 25 :=
by
  sorry

end area_of_inscribed_square_l53_53738


namespace true_propositions_count_l53_53755

theorem true_propositions_count :
  (∃ x₀ : ℤ, x₀^3 < 0) ∧
  ((∀ a : ℝ, (∃ x : ℝ, a*x^2 + 2*x + 1 = 0 ∧ x < 0) ↔ a ≤ 1) → false) ∧ 
  (¬ (∀ x : ℝ, x^2 = 1/4 * x^2 → y = 1 → false)) →
  true_prop_count = 1 := 
sorry

end true_propositions_count_l53_53755


namespace remainder_of_x_div_9_l53_53880

theorem remainder_of_x_div_9 (x : ℕ) (hx_pos : 0 < x) (h : (6 * x) % 9 = 3) : x % 9 = 5 :=
by {
  sorry
}

end remainder_of_x_div_9_l53_53880


namespace part1_part2_l53_53289

def p (x : ℝ) : Prop := x^2 - 10*x + 16 ≤ 0
def q (x m : ℝ) : Prop := m > 0 ∧ x^2 - 4*m*x + 3*m^2 ≤ 0

theorem part1 (x : ℝ) : 
  (∃ (m : ℝ), m = 1 ∧ (p x ∨ q x m)) → 1 ≤ x ∧ x ≤ 8 :=
by
  intros
  sorry

theorem part2 (m : ℝ) :
  (∀ x, q x m → p x) ∧ ∃ x, ¬ q x m ∧ p x → 2 ≤ m ∧ m ≤ 8/3 :=
by
  intros
  sorry

end part1_part2_l53_53289


namespace geom_seq_seventh_term_l53_53504

theorem geom_seq_seventh_term (a r : ℝ) (n : ℕ) (h1 : a = 2) (h2 : r^8 * a = 32) :
  a * r^6 = 128 :=
by
  sorry

end geom_seq_seventh_term_l53_53504


namespace simplify_and_evaluate_l53_53206

theorem simplify_and_evaluate (a b : ℝ) (h1 : a = -1) (h2 : b = 1) :
  (4/5 * a * b - (2 * a * b^2 - 4 * (-1/5 * a * b + 3 * a^2 * b)) + 2 * a * b^2) = 12 :=
by
  have ha : a = -1 := h1
  have hb : b = 1 := h2
  sorry

end simplify_and_evaluate_l53_53206


namespace derivative_at_zero_l53_53914

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem derivative_at_zero : deriv f 0 = 2 := by
  sorry

end derivative_at_zero_l53_53914


namespace cyclists_equal_distance_l53_53018

theorem cyclists_equal_distance (v1 v2 v3 : ℝ) (t1 t2 t3 : ℝ) (d : ℝ)
  (h_v1 : v1 = 12) (h_v2 : v2 = 16) (h_v3 : v3 = 24)
  (h_one_riding : t1 + t2 + t3 = 3) 
  (h_dist_equal : v1 * t1 = v2 * t2 ∧ v2 * t2 = v3 * t3 ∧ v1 * t1 = d) :
  d = 16 :=
by
  sorry

end cyclists_equal_distance_l53_53018


namespace sqrt_mul_example_complex_expression_example_l53_53918

theorem sqrt_mul_example : Real.sqrt 3 * Real.sqrt 27 = 9 :=
by sorry

theorem complex_expression_example : 
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6 :=
by sorry

end sqrt_mul_example_complex_expression_example_l53_53918


namespace who_next_to_boris_l53_53618

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l53_53618


namespace clean_time_per_room_l53_53733

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

end clean_time_per_room_l53_53733


namespace geometric_sequence_sum_l53_53692

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q a2005 a2006 : ℝ), 
    (∀ n, a (n + 1) = a n * q) ∧
    q > 1 ∧
    a2005 + a2006 = 2 ∧ 
    a2005 * a2006 = 3 / 4 ∧ 
    a (2005) = a2005 ∧ 
    a (2006) = a2006 → 
    a (2007) + a (2008) = 18 := 
by
  sorry

end geometric_sequence_sum_l53_53692


namespace find_e_l53_53858

def P (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

-- Conditions
variables (d e f : ℝ)
-- Mean of zeros, twice product of zeros, and sum of coefficients are equal
variables (mean_of_zeros equals twice_product_of_zeros equals sum_of_coefficients equals: ℝ)
-- y-intercept is 9
axiom intercept_eq_nine : f = 9

-- Vieta's formulas for cubic polynomial
axiom product_of_zeros : twice_product_of_zeros = 2 * (- (f / 3))
axiom mean_of_zeros_sum : mean_of_zeros = -18/3  -- 3 times the mean of the zeros
axiom sum_of_coef : 3 + d + e + f = sum_of_coefficients

-- All these quantities are equal to the same value
axiom triple_equality : mean_of_zeros = twice_product_of_zeros
axiom triple_equality_coefs : mean_of_zeros = sum_of_coefficients

-- Lean statement we need to prove
theorem find_e : e = -72 :=
by
  sorry

end find_e_l53_53858


namespace second_train_catches_first_l53_53752

-- Define the starting times and speeds
def t1_start_time := 14 -- 2:00 pm in 24-hour format
def t1_speed := 70 -- km/h
def t2_start_time := 15 -- 3:00 pm in 24-hour format
def t2_speed := 80 -- km/h

-- Define the time at which the second train catches the first train
def catch_time := 22 -- 10:00 pm in 24-hour format

theorem second_train_catches_first :
  ∃ t : ℕ, t = catch_time ∧
    t1_speed * ((t - t1_start_time) + 1) = t2_speed * (t - t2_start_time) := by
  sorry

end second_train_catches_first_l53_53752


namespace regular_polygon_perimeter_l53_53051

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53051


namespace sequence_sum_l53_53750

-- Definition of the sequence b_n
def b : ℕ → ℝ
| 0     := 0 -- We define b_0 to be 0 to make b_1 be the first element at index 1
| 1     := 2
| 2     := 3
| (n+3) := (1 / 2) * b (n + 2) + (1 / 3) * b (n + 1)

-- The statement to prove that the infinite sum of the sequence is 4.2
theorem sequence_sum : (∑' n, b n) = 4.2 := by
  sorry

end sequence_sum_l53_53750


namespace arithmetic_sequence_a5_zero_l53_53482

variable {a : ℕ → ℤ}
variable {d : ℤ}

theorem arithmetic_sequence_a5_zero 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : d ≠ 0)
  (h3 : a 3 + a 9 = a 10 - a 8) : 
  a 5 = 0 := sorry

end arithmetic_sequence_a5_zero_l53_53482


namespace regular_polygon_perimeter_l53_53077

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53077


namespace units_digit_of_m_squared_plus_two_to_the_m_is_seven_l53_53829

def m := 2016^2 + 2^2016

theorem units_digit_of_m_squared_plus_two_to_the_m_is_seven :
  (m^2 + 2^m) % 10 = 7 := by
sorry

end units_digit_of_m_squared_plus_two_to_the_m_is_seven_l53_53829


namespace who_next_to_boris_l53_53615

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l53_53615


namespace tangent_normal_equations_l53_53243

open Real

noncomputable def tangent_normal_lines (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a * (sin t) ^ 3, a * (cos t) ^ 3)

theorem tangent_normal_equations (a t_0 : ℝ) (h : t_0 = π / 3) :
  let x_0 := a * (sin t_0) ^ 3,
      y_0 := a * (cos t_0) ^ 3,
      m_tangent := -cot t_0,
      m_normal := tan t_0 in
  -- equation of tangent line
  (∀ x y, y - y_0 = m_tangent * (x - x_0) → 
           y = - 1 / (√3) * x + a / 2) ∧
  -- equation of normal line
  (∀ x y, y - y_0 = -1 / m_tangent * (x - x_0) → 
           y = √3 * x - a) :=
by
  have h0 : t_0 = π / 3 := h
  sorry

end tangent_normal_equations_l53_53243


namespace average_side_length_of_squares_l53_53385

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l53_53385


namespace min_value_of_expression_l53_53933

theorem min_value_of_expression : 
  ∃ x y : ℝ, (z = x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3) ∧ z = 1 ∧ x = 0 ∧ y = -1 :=
by
  sorry

end min_value_of_expression_l53_53933


namespace regular_polygon_perimeter_is_28_l53_53076

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53076


namespace standing_next_to_boris_l53_53608

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l53_53608


namespace quadrilateral_smallest_angle_l53_53746

theorem quadrilateral_smallest_angle
  (a d : ℝ)
  (h1 : a + (a + 2 * d) = 160)
  (h2 : a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) :
  a = 60 :=
by
  sorry

end quadrilateral_smallest_angle_l53_53746


namespace probability_at_least_one_multiple_of_4_l53_53516

/-- Definition for the total number of integers in the range -/
def total_numbers : ℕ := 60

/-- Definition for the number of multiples of 4 within the range -/
def multiples_of_4 : ℕ := 15

/-- Probability that a single number chosen is not a multiple of 4 -/
def prob_not_multiple_of_4 : ℚ := (total_numbers - multiples_of_4) / total_numbers

/-- Probability that none of the three chosen numbers is a multiple of 4 -/
def prob_none_multiple_of_4 : ℚ := prob_not_multiple_of_4 ^ 3

/-- Given condition that Linda choose three times -/
axiom linda_chooses_thrice (x y z : ℕ) : 
1 ≤ x ∧ x ≤ 60 ∧ 
1 ≤ y ∧ y ≤ 60 ∧ 
1 ≤ z ∧ z ≤ 60

/-- Theorem stating the desired probability -/
theorem probability_at_least_one_multiple_of_4 : 
1 - prob_none_multiple_of_4 = 37 / 64 := by
  sorry

end probability_at_least_one_multiple_of_4_l53_53516


namespace regular_polygon_perimeter_l53_53103

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53103


namespace exists_solution_in_interval_l53_53451

theorem exists_solution_in_interval : ∃ x ∈ (Set.Ioo (3: ℝ) (4: ℝ)), Real.log x / Real.log 2 + x - 5 = 0 :=
by
  sorry

end exists_solution_in_interval_l53_53451


namespace average_side_length_of_squares_l53_53375

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53375


namespace optimal_strategy_l53_53737

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6
noncomputable def score_A_correct : ℝ := 20
noncomputable def score_B_correct : ℝ := 80

def X_distribution : (ℝ → ℝ) :=
λ x, if x = 0 then 1 - prob_A_correct
     else if x = 20 then prob_A_correct * (1 - prob_B_correct)
     else if x = 100 then prob_A_correct * prob_B_correct
     else 0

noncomputable def E_X : ℝ :=
(0 * (1 - prob_A_correct)) + (20 * (prob_A_correct * (1 - prob_B_correct))) + (100 * (prob_A_correct * prob_B_correct))

noncomputable def E_Y : ℝ :=
(0 * (1 - prob_B_correct)) + (80 * (prob_B_correct * (1 - prob_A_correct))) + (100 * (prob_B_correct * prob_A_correct))

theorem optimal_strategy : E_X = 54.4 ∧ E_Y = 57.6 → (57.6 > 54.4) :=
by {
  sorry 
}

end optimal_strategy_l53_53737


namespace tenth_term_is_19_over_4_l53_53784

def nth_term_arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

theorem tenth_term_is_19_over_4 :
  nth_term_arithmetic_sequence (1/4) (1/2) 10 = 19/4 :=
by
  sorry

end tenth_term_is_19_over_4_l53_53784


namespace find_value_l53_53180

theorem find_value (x : ℝ) (h : 0.20 * x = 80) : 0.40 * x = 160 := 
by
  sorry

end find_value_l53_53180


namespace intersection_of_sets_l53_53199

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { y | ∃ x : ℝ, y = 2^x - 1 }
def C : Set ℝ := { m | -1 < m ∧ m < 2 }

theorem intersection_of_sets : A ∩ B = C := 
by sorry

end intersection_of_sets_l53_53199


namespace blown_out_sand_dunes_l53_53521

theorem blown_out_sand_dunes (p_remain p_lucky p_both : ℝ) (h_rem: p_remain = 1 / 3) (h_luck: p_lucky = 2 / 3)
(h_both: p_both = 0.08888888888888889) : 
  ∃ N : ℕ, N = 8 :=
by
  sorry

end blown_out_sand_dunes_l53_53521


namespace correct_exp_identity_l53_53883

variable (a b : ℝ)

theorem correct_exp_identity : ((a^2 * b)^3 / (-a * b)^2 = a^4 * b) := sorry

end correct_exp_identity_l53_53883


namespace bill_has_correct_final_amount_l53_53469

def initial_amount : ℕ := 42
def pizza_cost : ℕ := 11
def pizzas_bought : ℕ := 3
def bill_initial_amount : ℕ := 30
def amount_spent := pizzas_bought * pizza_cost
def frank_remaining_amount := initial_amount - amount_spent
def bill_final_amount := bill_initial_amount + frank_remaining_amount

theorem bill_has_correct_final_amount : bill_final_amount = 39 := by
  sorry

end bill_has_correct_final_amount_l53_53469


namespace find_sum_s_u_l53_53873

theorem find_sum_s_u (p r s u : ℝ) (q t : ℝ) 
  (h_q : q = 5) 
  (h_t : t = -p - r) 
  (h_sum_imaginary : q + s + u = 4) :
  s + u = -1 := 
sorry

end find_sum_s_u_l53_53873


namespace regular_polygon_perimeter_l53_53107

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53107


namespace math_problem_l53_53990

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l53_53990


namespace isosceles_triangle_base_length_l53_53220

theorem isosceles_triangle_base_length (P B : ℕ) (hP : P = 13) (hB : B = 3) :
    ∃ S : ℕ, S ≠ 3 ∧ S = 3 :=
by
    sorry

end isosceles_triangle_base_length_l53_53220


namespace perimeter_of_regular_polygon_l53_53053

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53053


namespace construct_quadratic_l53_53921

-- Definitions from the problem's conditions
def quadratic_has_zeros (f : ℝ → ℝ) (r1 r2 : ℝ) : Prop :=
  f r1 = 0 ∧ f r2 = 0

def quadratic_value_at (f : ℝ → ℝ) (x_val value : ℝ) : Prop :=
  f x_val = value

-- Construct the Lean theorem statement
theorem construct_quadratic :
  ∃ f : ℝ → ℝ, quadratic_has_zeros f 1 5 ∧ quadratic_value_at f 3 10 ∧
  ∀ x, f x = (-5/2 : ℝ) * x^2 + 15 * x - 25 / 2 :=
sorry

end construct_quadratic_l53_53921


namespace tangent_line_b_value_l53_53270

noncomputable def b_value : ℝ := Real.log 2 - 1

theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x > 0, (fun x => Real.log x) x = (1/2) * x + b → ∃ c : ℝ, c = b) → b = Real.log 2 - 1 :=
by
  sorry

end tangent_line_b_value_l53_53270


namespace range_of_x_l53_53284

theorem range_of_x (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : x^2 + a * x > 4 * x + a - 3 ↔ (x > 3 ∨ x < -1) := by
  sorry

end range_of_x_l53_53284


namespace fractional_eq_solution_range_l53_53498

theorem fractional_eq_solution_range (x m : ℝ) (h : (2 * x - m) / (x + 1) = 1) (hx : x < 0) : 
  m < -1 ∧ m ≠ -2 := 
by 
  sorry

end fractional_eq_solution_range_l53_53498


namespace variance_of_planted_trees_l53_53680

def number_of_groups := 10

def planted_trees : List ℕ := [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]

noncomputable def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / (xs.length : ℚ)

noncomputable def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / (xs.length : ℚ)

theorem variance_of_planted_trees :
  variance planted_trees = 0.6 := sorry

end variance_of_planted_trees_l53_53680


namespace arrangement_proof_l53_53768

def arrangement_problem : Prop :=
  ∃ (countries : Set ℕ) (hotels : Set ℕ) (f : ℕ → ℕ),
    countries = {(1, 2, 3, 4, 5)} ∧
    hotels = {(1, 2, 3)} ∧
    (∀ country ∈ countries, f country ∈ hotels) ∧
    (∀ hotel ∈ hotels, ∃ country ∈ countries, f country = hotel) ∧
    fintype.card {g : ℕ → ℕ // g = f} = 150

theorem arrangement_proof : arrangement_problem :=
begin
  -- proof would go here
  sorry,
end

end arrangement_proof_l53_53768


namespace additional_teddies_per_bunny_l53_53507

theorem additional_teddies_per_bunny (teddies bunnies koala total_mascots: ℕ) 
  (h1 : teddies = 5) 
  (h2 : bunnies = 3 * teddies) 
  (h3 : koala = 1) 
  (h4 : total_mascots = 51): 
  (total_mascots - (teddies + bunnies + koala)) / bunnies = 2 := 
by 
  sorry

end additional_teddies_per_bunny_l53_53507


namespace range_of_m_l53_53398

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < |m - 2|) ↔ m < 0 ∨ m > 4 := 
sorry

end range_of_m_l53_53398


namespace gcd_factorial_7_8_l53_53632

theorem gcd_factorial_7_8 : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := 
by
  sorry

end gcd_factorial_7_8_l53_53632


namespace circle_square_area_difference_l53_53436

theorem circle_square_area_difference :
  let d_square := 8
      d_circle := 8
      s := d_square / (Real.sqrt 2)
      r := d_circle / 2
      area_square := s * s
      area_circle := Real.pi * r * r
      difference := area_circle - area_square
  in Real.abs (difference - 18.2) < 0.1 :=
by
  sorry

end circle_square_area_difference_l53_53436


namespace value_of_expression_l53_53414

theorem value_of_expression (x y : ℕ) (h₁ : x = 12) (h₂ : y = 7) : (x - y) * (x + y) = 95 := by
  -- Here we assume all necessary conditions as given:
  -- x = 12 and y = 7
  -- and we prove that (x - y)(x + y) = 95
  sorry

end value_of_expression_l53_53414


namespace perimeter_of_regular_polygon_l53_53062

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53062


namespace perimeter_of_polygon_l53_53042

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53042


namespace balls_in_boxes_ways_l53_53156

theorem balls_in_boxes_ways : 
  (∃ (ways : ℕ), ways = 21 ∧
    ∀ {k : ℕ}, (5 + k - 1).choose (5) = ways) := 
begin
  sorry,
end

end balls_in_boxes_ways_l53_53156


namespace problem_statement_l53_53986

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l53_53986


namespace sum_of_solutions_l53_53783

theorem sum_of_solutions (a : ℝ) (h : 0 < a ∧ a < 1) :
  let x1 := 3 + a
  let x2 := 3 - a
  let x3 := 1 + a
  let x4 := 1 - a
  x1 + x2 + x3 + x4 = 8 :=
by
  intros
  sorry

end sum_of_solutions_l53_53783


namespace triangle_angle_contradiction_l53_53719

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ α > 60 ∧ β > 60 ∧ γ > 60 → False :=
by
  sorry

end triangle_angle_contradiction_l53_53719


namespace regular_polygon_perimeter_l53_53092

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53092


namespace new_ratio_after_2_years_l53_53861

-- Definitions based on conditions
variable (A : ℕ) -- Current age of a
variable (B : ℕ) -- Current age of b

-- Conditions
def ratio_a_b := A / B = 5 / 3
def current_age_b := B = 6

-- Theorem: New ratio after 2 years is 3:2
theorem new_ratio_after_2_years (h1 : ratio_a_b A B) (h2 : current_age_b B) : (A + 2) / (B + 2) = 3 / 2 := by
  sorry

end new_ratio_after_2_years_l53_53861


namespace who_is_next_to_boris_l53_53596

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l53_53596


namespace a_range_l53_53481

theorem a_range (x y z a : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1)
  (h_eq : a / (x * y * z) = (1 / x) + (1 / y) + (1 / z) - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
sorry

end a_range_l53_53481


namespace greatest_a_l53_53929

theorem greatest_a (a : ℝ) : a^2 - 14*a + 45 ≤ 0 → a ≤ 9 :=
by
  -- placeholder for the actual proof
  sorry

end greatest_a_l53_53929


namespace average_side_lengths_l53_53368

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l53_53368


namespace average_side_lengths_l53_53370

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l53_53370


namespace boris_neighbors_l53_53580

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l53_53580


namespace value_of_b_l53_53887

theorem value_of_b (a c : ℝ) (b : ℝ) (h1 : a = 105) (h2 : c = 70) (h3 : a^4 = 21 * 25 * 15 * b * c^3) : b = 0.045 :=
by
  sorry

end value_of_b_l53_53887


namespace pet_store_has_70_birds_l53_53568

-- Define the given conditions
def num_cages : ℕ := 7
def parrots_per_cage : ℕ := 4
def parakeets_per_cage : ℕ := 3
def cockatiels_per_cage : ℕ := 2
def canaries_per_cage : ℕ := 1

-- Total number of birds in one cage
def birds_per_cage : ℕ := parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage

-- Total number of birds in all cages
def total_birds := birds_per_cage * num_cages

-- Prove that the total number of birds is 70
theorem pet_store_has_70_birds : total_birds = 70 :=
sorry

end pet_store_has_70_birds_l53_53568


namespace sum_of_odd_integers_13_to_41_l53_53412

theorem sum_of_odd_integers_13_to_41 : 
  (∑ k in Finset.filter (λ n, n % 2 = 1) (Finset.range 42), k) - ∑ k in Finset.filter (λ n, n % 2 = 1) (Finset.range 13), k = 405 :=
by
  sorry

end sum_of_odd_integers_13_to_41_l53_53412


namespace f_odd_function_no_parallel_lines_l53_53650

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - (1 / a^x))

theorem f_odd_function {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x : ℝ, f a (-x) = -f a x := 
by
  sorry

theorem no_parallel_lines {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f a x1 ≠ f a x2 :=
by
  sorry

end f_odd_function_no_parallel_lines_l53_53650


namespace shortest_side_of_triangle_l53_53684

noncomputable def triangle_shortest_side_length (a b r : ℝ) (shortest : ℝ) : Prop :=
a = 8 ∧ b = 6 ∧ r = 4 ∧ shortest = 12

theorem shortest_side_of_triangle 
  (a b r shortest : ℝ) 
  (h : triangle_shortest_side_length a b r shortest) : shortest = 12 :=
sorry

end shortest_side_of_triangle_l53_53684


namespace who_is_next_to_Boris_l53_53612

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l53_53612


namespace required_additional_amount_l53_53903

noncomputable def ryan_order_total : ℝ := 15.80 + 8.20 + 10.50 + 6.25 + 9.15
def minimum_free_delivery : ℝ := 50
def discount_threshold : ℝ := 30
def discount_rate : ℝ := 0.10

theorem required_additional_amount : 
  ∃ X : ℝ, ryan_order_total + X - discount_rate * (ryan_order_total + X) = minimum_free_delivery :=
sorry

end required_additional_amount_l53_53903


namespace quadratic_decreasing_then_increasing_l53_53534

-- Define the given quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 6 * x + 10

-- Define the interval of interest
def interval (x : ℝ) : Prop := 2 < x ∧ x < 4

-- The main theorem to prove: the function is first decreasing on (2, 3] and then increasing on [3, 4)
theorem quadratic_decreasing_then_increasing :
  (∀ (x : ℝ), 2 < x ∧ x ≤ 3 → quadratic_function x > quadratic_function (x + ε) ∧ ε > 0) ∧
  (∀ (x : ℝ), 3 ≤ x ∧ x < 4 → quadratic_function x < quadratic_function (x + ε) ∧ ε > 0) :=
sorry

end quadratic_decreasing_then_increasing_l53_53534


namespace carly_butterfly_days_l53_53623

-- Define the conditions
variable (x : ℕ) -- number of days Carly practices her butterfly stroke
def butterfly_hours_per_day := 3  -- hours per day for butterfly stroke
def backstroke_hours_per_day := 2  -- hours per day for backstroke stroke
def backstroke_days_per_week := 6  -- days per week for backstroke stroke
def total_hours_per_month := 96  -- total hours practicing swimming in a month
def weeks_in_month := 4  -- number of weeks in a month

-- The proof problem
theorem carly_butterfly_days :
  (butterfly_hours_per_day * x + backstroke_hours_per_day * backstroke_days_per_week) * weeks_in_month = total_hours_per_month
  → x = 4 := 
by
  sorry

end carly_butterfly_days_l53_53623


namespace problem_proof_l53_53162

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l53_53162


namespace complete_square_l53_53765

theorem complete_square :
  (∀ x: ℝ, 2 * x^2 - 4 * x + 1 = 2 * (x - 1)^2 - 1) := 
by
  intro x
  sorry

end complete_square_l53_53765


namespace parabola_circle_intercept_l53_53250

theorem parabola_circle_intercept (p : ℝ) (h_pos : p > 0) :
  (∃ (x y : ℝ), y^2 = 2 * p * x ∧ x^2 + y^2 + 2 * x - 3 = 0) ∧
  (∃ (y1 y2 : ℝ), (y1 - y2)^2 + (-(p / 2) + 1)^2 = 4^2) → p = 2 :=
by sorry

end parabola_circle_intercept_l53_53250


namespace find_amount_with_r_l53_53555

variable (p q r s : ℝ) (total : ℝ := 9000)

-- Condition 1: Total amount is 9000 Rs
def total_amount_condition := p + q + r + s = total

-- Condition 2: r has three-quarters of the combined amount of p, q, and s
def r_amount_condition := r = (3/4) * (p + q + s)

-- The goal is to prove that r = 10800
theorem find_amount_with_r (h1 : total_amount_condition p q r s) (h2 : r_amount_condition p q r s) :
  r = 10800 :=
sorry

end find_amount_with_r_l53_53555


namespace developer_lots_l53_53564

theorem developer_lots (acres : ℕ) (cost_per_acre : ℕ) (lot_price : ℕ) 
  (h1 : acres = 4) 
  (h2 : cost_per_acre = 1863) 
  (h3 : lot_price = 828) : 
  ((acres * cost_per_acre) / lot_price) = 9 := 
  by
    sorry

end developer_lots_l53_53564


namespace who_is_next_to_boris_l53_53592

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l53_53592


namespace problem_proof_l53_53161

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l53_53161


namespace regular_polygon_perimeter_l53_53078

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53078


namespace calculate_area_correct_l53_53934

-- Define the side length of the square
def side_length : ℝ := 5

-- Define the rotation angles in degrees
def rotation_angles : List ℝ := [0, 30, 45, 60]

-- Define the area calculation function (to be implemented)
def calculate_overlap_area (s : ℝ) (angles : List ℝ) : ℝ := sorry

-- Define the proof that the calculated area is equal to 123.475
theorem calculate_area_correct : calculate_overlap_area side_length rotation_angles = 123.475 :=
by
  sorry

end calculate_area_correct_l53_53934


namespace regular_polygon_perimeter_is_28_l53_53071

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53071


namespace find_value_of_c_l53_53228

-- Given: The transformed linear regression equation and the definition of z
theorem find_value_of_c (z : ℝ) (y : ℝ) (x : ℝ) (c : ℝ) (k : ℝ) (h1 : z = 0.4 * x + 2) (h2 : z = Real.log y) (h3 : y = c * Real.exp (k * x)) : 
  c = Real.exp 2 :=
by
  sorry

end find_value_of_c_l53_53228


namespace expression_simplification_l53_53026

theorem expression_simplification (a : ℝ) (h : a ≠ 1) (h_beta : 1 = 1):
  (2^(Real.log (a) / Real.log (Real.sqrt 2)) - 
   3^((Real.log (a^2+1)) / (Real.log 27)) - 
   2 * a) / 
  (7^(4 * (Real.log (a) / Real.log 49)) - 
   5^((0.5 * Real.log (a)) / (Real.log (Real.sqrt 5))) - 1) = a^2 + a + 1 :=
by
  sorry

end expression_simplification_l53_53026


namespace regular_polygon_perimeter_l53_53108

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53108


namespace range_of_sum_l53_53295

theorem range_of_sum (a b c : ℝ) (h1: a > b) (h2 : b > c) (h3 : a + b + c = 1) (h4 : a^2 + b^2 + c^2 = 3) :
-2/3 < b + c ∧ b + c < 0 := 
by 
  sorry

end range_of_sum_l53_53295


namespace average_speed_is_correct_l53_53112
noncomputable def average_speed_trip : ℝ :=
  let distance_AB := 240 * 5
  let distance_BC := 300 * 3
  let distance_CD := 400 * 4
  let total_distance := distance_AB + distance_BC + distance_CD
  let flight_time_AB := 5
  let layover_B := 2
  let flight_time_BC := 3
  let layover_C := 1
  let flight_time_CD := 4
  let total_time := (flight_time_AB + flight_time_BC + flight_time_CD) + (layover_B + layover_C)
  total_distance / total_time

theorem average_speed_is_correct :
  average_speed_trip = 246.67 := sorry

end average_speed_is_correct_l53_53112


namespace persons_next_to_Boris_l53_53602

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l53_53602


namespace initial_investment_l53_53913

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_investment (A : ℝ) (r : ℝ) (n t : ℕ) (P : ℝ) :
  A = 3630.0000000000005 → r = 0.10 → n = 1 → t = 2 → P = 3000 →
  A = compound_interest P r n t :=
by
  intros hA hr hn ht hP
  rw [compound_interest, hA, hr, hP]
  sorry

end initial_investment_l53_53913


namespace polynomial_roots_problem_l53_53943

theorem polynomial_roots_problem (γ δ : ℝ) (h₁ : γ^2 - 3*γ + 2 = 0) (h₂ : δ^2 - 3*δ + 2 = 0) :
  8*γ^3 - 6*δ^2 = 48 :=
by
  sorry

end polynomial_roots_problem_l53_53943


namespace some_number_value_l53_53567

theorem some_number_value (some_number : ℝ): 
  (∀ n : ℝ, (n / some_number) * (n / 80) = 1 → n = 40) → some_number = 80 :=
by
  sorry

end some_number_value_l53_53567


namespace probability_square_product_l53_53407

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_favorable_outcomes : ℕ :=
  List.length [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (4, 4), (2, 8), (8, 2), (5, 5), (4, 9), (6, 6), (7, 7), (8, 8), (9, 9)]

def total_outcomes : ℕ := 12 * 8

theorem probability_square_product :
  (count_favorable_outcomes : ℚ) / (total_outcomes : ℚ) = (7 : ℚ) / (48 : ℚ) := 
by 
  sorry

end probability_square_product_l53_53407


namespace arithmetic_geom_seq_S5_l53_53643

theorem arithmetic_geom_seq_S5 (a_n : ℕ → ℚ) (S_n : ℕ → ℚ)
  (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * (1/2))
  (h_sum : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) / 2) * (1/2))
  (h_geom_seq : (a_n 2) * (a_n 14) = (a_n 6) ^ 2) :
  S_n 5 = 25 / 2 :=
by
  sorry

end arithmetic_geom_seq_S5_l53_53643


namespace find_max_value_l53_53145

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x + a

theorem find_max_value (a x : ℝ) (h_min : f 1 a = 1) : 
  ∃ x : ℝ, f (-1/3) 2 = 59/27 :=
by {
  sorry
}

end find_max_value_l53_53145


namespace eldest_child_age_l53_53566

variables (y m e : ℕ)

theorem eldest_child_age (h1 : m = y + 3)
                        (h2 : e = 3 * y)
                        (h3 : e = y + m + 2) : e = 15 :=
by
  sorry

end eldest_child_age_l53_53566


namespace ratio_of_segments_l53_53267

variable (F S T : ℕ)

theorem ratio_of_segments : T = 10 → F = 2 * (S + T) → F + S + T = 90 → (T / S = 1 / 2) :=
by
  intros hT hF hSum
  sorry

end ratio_of_segments_l53_53267


namespace person_next_to_Boris_arkady_galya_l53_53590

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l53_53590


namespace final_value_l53_53971

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l53_53971


namespace find_cookbooks_stashed_in_kitchen_l53_53149

-- Definitions of the conditions
def total_books := 99
def books_in_boxes := 3 * 15
def books_in_room := 21
def books_on_table := 4
def books_picked_up := 12
def current_books := 23

-- Main statement
theorem find_cookbooks_stashed_in_kitchen :
  let books_donated := books_in_boxes + books_in_room + books_on_table
  let books_left_initial := total_books - books_donated
  let books_left_before_pickup := current_books - books_picked_up
  books_left_initial - books_left_before_pickup = 18 := by
  sorry

end find_cookbooks_stashed_in_kitchen_l53_53149


namespace cuboid_face_areas_l53_53529

-- Conditions
variables (a b c S : ℝ)
-- Surface area of the sphere condition
theorem cuboid_face_areas 
  (h1 : a * b = 6) 
  (h2 : b * c = 10) 
  (h3 : a^2 + b^2 + c^2 = 76) 
  (h4 : 4 * π * 38 = 152 * π) :
  a * c = 15 :=
by 
  -- Prove that the solution matches the conclusion
  sorry

end cuboid_face_areas_l53_53529


namespace Sarah_pool_depth_l53_53823

theorem Sarah_pool_depth (S J : ℝ) (h1 : J = 2 * S + 5) (h2 : J = 15) : S = 5 := by
  sorry

end Sarah_pool_depth_l53_53823


namespace expected_value_shorter_gentlemen_l53_53760

-- Definitions based on the problem conditions
def expected_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

-- The main theorem statement based on the problem translation
theorem expected_value_shorter_gentlemen (n : ℕ) : 
  expected_shorter_gentlemen n = (n - 1) / 2 :=
by
  sorry

end expected_value_shorter_gentlemen_l53_53760


namespace jeremy_is_40_l53_53221

-- Definitions for Jeremy (J), Sebastian (S), and Sophia (So)
def JeremyCurrentAge : ℕ := 40
def SebastianCurrentAge : ℕ := JeremyCurrentAge + 4
def SophiaCurrentAge : ℕ := 60 - 3

-- Assertion properties
axiom age_sum_in_3_years : (JeremyCurrentAge + 3) + (SebastianCurrentAge + 3) + (SophiaCurrentAge + 3) = 150
axiom sebastian_older_by_4 : SebastianCurrentAge = JeremyCurrentAge + 4
axiom sophia_age_in_3_years : SophiaCurrentAge + 3 = 60

-- The theorem to prove that Jeremy is currently 40 years old
theorem jeremy_is_40 : JeremyCurrentAge = 40 := by
  sorry

end jeremy_is_40_l53_53221


namespace nonneg_or_nonpos_l53_53846

theorem nonneg_or_nonpos (n : ℕ) (h : n ≥ 2) (c : Fin n → ℝ)
  (h_eq : (n - 1) * (Finset.univ.sum (fun i => c i ^ 2)) = (Finset.univ.sum c) ^ 2) :
  (∀ i, c i ≥ 0) ∨ (∀ i, c i ≤ 0) := 
  sorry

end nonneg_or_nonpos_l53_53846


namespace angle_C_measure_l53_53658

-- We define angles and the specific conditions given in the problem.
def measure_angle_A : ℝ := 80
def external_angle_C : ℝ := 100

theorem angle_C_measure :
  ∃ (C : ℝ) (A B : ℝ), (A + B = measure_angle_A) ∧
                       (C + external_angle_C = 180) ∧
                       (external_angle_C = measure_angle_A) →
                       C = 100 :=
by {
  -- skipping proof
  sorry
}

end angle_C_measure_l53_53658


namespace find_x_value_l53_53816

theorem find_x_value (X : ℕ) 
  (top_left : ℕ := 2)
  (top_second : ℕ := 3)
  (top_last : ℕ := 4)
  (bottom_left : ℕ := 3)
  (bottom_middle : ℕ := 5) 
  (top_sum_eq: 2 + 3 + X + 4 = 9 + X)
  (bottom_sum_eq: 3 + 5 + (X + 1) = 9 + X) : 
  X = 1 := by 
  sorry

end find_x_value_l53_53816


namespace standing_next_to_boris_l53_53603

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end standing_next_to_boris_l53_53603


namespace algebraic_expression_identity_l53_53179

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l53_53179


namespace number_of_good_sets_l53_53515

open Finset

def is_good_set (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ ∀ (x y ∈ s), x ≠ y → x + y ≠ 8

def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

theorem number_of_good_sets :
  (A.powerset.filter is_good_set).card = 8 :=
  sorry

end number_of_good_sets_l53_53515


namespace percent_to_decimal_l53_53422

theorem percent_to_decimal : (2 : ℝ) / 100 = 0.02 :=
by
  -- Proof would go here
  sorry

end percent_to_decimal_l53_53422


namespace evaluate_expression_l53_53927

def a := 3 + 6 + 9
def b := 2 + 5 + 8
def c := 3 + 6 + 9
def d := 2 + 5 + 8

theorem evaluate_expression : (a / b) - (d / c) = 11 / 30 :=
by
  sorry

end evaluate_expression_l53_53927


namespace persons_next_to_Boris_l53_53599

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l53_53599


namespace find_a5_l53_53212

-- Define the geometric sequence and the given conditions
def geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Define the conditions for our problem
def conditions (a : ℕ → ℝ) :=
  geom_sequence a 2 ∧ (∀ n, 0 < a n) ∧ a 3 * a 11 = 16

-- Our goal is to prove that a_5 = 1
theorem find_a5 (a : ℕ → ℝ) (h : conditions a) : a 5 = 1 := 
by 
  sorry

end find_a5_l53_53212


namespace range_of_m_l53_53301

noncomputable def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem range_of_m (m : ℝ) (h : ∀ x, x ∈ B m → x ∈ A) : m = 3 ∨ -2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_m_l53_53301


namespace initial_apps_l53_53266

-- Define the initial condition stating the number of files Dave had initially
def files_initial : ℕ := 21

-- Define the condition after deletion
def apps_after_deletion : ℕ := 3
def files_after_deletion : ℕ := 7

-- Define the number of files deleted
def files_deleted : ℕ := 14

-- Prove that the initial number of apps Dave had was 3
theorem initial_apps (a : ℕ) (h1 : files_initial = 21) 
(h2 : files_after_deletion = 7) 
(h3 : files_deleted = 14) 
(h4 : a - 3 = 0) : a = 3 :=
by sorry

end initial_apps_l53_53266


namespace remainder_102_104_plus_6_div_9_l53_53716

theorem remainder_102_104_plus_6_div_9 :
  ((102 * 104 + 6) % 9) = 3 :=
by
  sorry

end remainder_102_104_plus_6_div_9_l53_53716


namespace smallest_number_divisible_by_20_and_36_l53_53022

-- Define the conditions that x must be divisible by both 20 and 36
def divisible_by (x n : ℕ) : Prop := ∃ m : ℕ, x = n * m

-- Define the problem statement
theorem smallest_number_divisible_by_20_and_36 : 
  ∃ x : ℕ, divisible_by x 20 ∧ divisible_by x 36 ∧ 
  (∀ y : ℕ, (divisible_by y 20 ∧ divisible_by y 36) → y ≥ x) ∧ x = 180 := 
by
  sorry

end smallest_number_divisible_by_20_and_36_l53_53022


namespace infimum_of_function_l53_53932

open Real

-- Definitions given in the conditions:
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (1 - x) = f (1 + x)
def function_on_interval (f : ℝ → ℝ) := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = -3 * x ^ 2 + 2

-- Proof problem statement:
theorem infimum_of_function (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_periodic : periodic_function f) 
  (h_interval : function_on_interval f) : 
  ∃ M : ℝ, (∀ x : ℝ, f x ≥ M) ∧ M = -1 :=
by
  sorry

end infimum_of_function_l53_53932


namespace range_of_a_plus_b_l53_53492

theorem range_of_a_plus_b 
  (a b : ℝ)
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) ≥ -1) : 
  -1 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end range_of_a_plus_b_l53_53492


namespace proof_equilateral_inscribed_circle_l53_53820

variables {A B C : Type*}
variables (r : ℝ) (D : ℝ)

def is_equilateral_triangle (A B C : Type*) : Prop := 
  -- Define the equilateral condition, where all sides are equal
  true

def is_inscribed_circle_radius (D r : ℝ) : Prop := 
  -- Define the property that D is the center and r is the radius 
  true

def distance_center_to_vertex (D r x : ℝ) : Prop := 
  x = 3 * r

theorem proof_equilateral_inscribed_circle 
  (A B C : Type*) 
  (r D : ℝ) 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_inscribed_circle_radius D r) : 
  distance_center_to_vertex D r (1 / 16) :=
by sorry

end proof_equilateral_inscribed_circle_l53_53820


namespace two_point_three_six_as_fraction_l53_53232

theorem two_point_three_six_as_fraction : (236 : ℝ) / 100 = (59 : ℝ) / 25 := 
by
  sorry

end two_point_three_six_as_fraction_l53_53232


namespace average_side_length_of_squares_l53_53382

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l53_53382


namespace cube_volume_fourth_power_l53_53023

theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 :=
sorry

end cube_volume_fourth_power_l53_53023


namespace average_side_length_of_squares_l53_53360

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53360


namespace regular_polygon_perimeter_l53_53089

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53089


namespace discarded_second_number_l53_53532

-- Define the conditions
def avg_original_50 : ℝ := 38
def total_sum_50_numbers : ℝ := 50 * avg_original_50
def discarded_first : ℝ := 45
def avg_remaining_48 : ℝ := 37.5
def total_sum_remaining_48 : ℝ := 48 * avg_remaining_48
def sum_discarded := total_sum_50_numbers - total_sum_remaining_48

-- Define the proof statement
theorem discarded_second_number (x : ℝ) (h : discarded_first + x = sum_discarded) : x = 55 :=
by
  sorry

end discarded_second_number_l53_53532


namespace cyclic_triples_l53_53907

theorem cyclic_triples (n : Nat) (wins losses : Fin n → Nat)
  (h1 : ∀ t, wins t = 6)
  (h2 : ∀ t, losses t = 9)
  (h3 : ∀ t u, t ≠ u → (wins t + losses t) = n - 1)
  (h4 : ∀ t u, t ≠ u → ((wins t + losses t)  + (wins u + losses u)) = n - 1)
: ∃ count, count = 320 := 
by
  use 320
  sorry

end cyclic_triples_l53_53907


namespace truncated_pyramid_distance_l53_53539

noncomputable def distance_from_plane_to_base
  (a b : ℝ) (α : ℝ) : ℝ :=
  (a * (a - b) * Real.tan α) / (3 * a - b)

theorem truncated_pyramid_distance
  (a b : ℝ) (α : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_α : 0 < α) :
  (a * (a - b) * Real.tan α) / (3 * a - b) = distance_from_plane_to_base a b α :=
by
  sorry

end truncated_pyramid_distance_l53_53539


namespace Marilyn_has_40_bananas_l53_53518

-- Definitions of the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- Statement of the proof problem
theorem Marilyn_has_40_bananas : (boxes * bananas_per_box) = 40 := by
  sorry

end Marilyn_has_40_bananas_l53_53518


namespace perimeter_of_regular_polygon_l53_53059

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53059


namespace cost_of_500_pencils_is_15_dollars_l53_53216

-- Defining the given conditions
def cost_per_pencil_cents : ℕ := 3
def pencils_count : ℕ := 500
def cents_to_dollars : ℕ := 100

-- The proof problem: statement only, no proof provided
theorem cost_of_500_pencils_is_15_dollars :
  (cost_per_pencil_cents * pencils_count) / cents_to_dollars = 15 :=
by
  sorry

end cost_of_500_pencils_is_15_dollars_l53_53216


namespace who_is_next_to_Boris_l53_53611

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l53_53611


namespace find_smaller_number_l53_53225

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : y = 8 :=
by sorry

end find_smaller_number_l53_53225


namespace multiple_of_3_l53_53001

theorem multiple_of_3 (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 3 ∣ n :=
sorry

end multiple_of_3_l53_53001


namespace hyperbola_vertices_distance_l53_53778

/--
For the hyperbola given by the equation
(x^2 / 121) - (y^2 / 49) = 1,
the distance between its vertices is 22.
-/
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ,
  (x^2 / 121) - (y^2 / 49) = 1 →
  ∃ a : ℝ, a = 11 ∧ 2 * a = 22 :=
by
  intros x y h
  use 11
  split
  · refl
  · norm_num

end hyperbola_vertices_distance_l53_53778


namespace regular_polygon_perimeter_l53_53088

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53088


namespace alphaMoreAdvantageousRegular_betaMoreAdvantageousMood_l53_53229

/-
  Definitions based on conditions
-/
def alphaCostPerMonth : ℕ := 999
def betaCostPerMonth : ℕ := 1299
def monthsInYear : ℕ := 12
def weeksInMonth : ℕ := 4
def regularVisitsPerWeek : ℕ := 2
def moodPatternVisitsPerYear : ℕ := 56  -- Derived from mood pattern calculation

/-
  Hypotheses based on question interpretation
-/
def alphaYearlyCost : ℕ := alphaCostPerMonth * monthsInYear
def betaYearlyCost : ℕ := betaCostPerMonth * monthsInYear

def regularVisitsPerYear : ℕ := regularVisitsPerWeek * weeksInMonth * monthsInYear
def alphaCostPerVisitRegular : ℕ := alphaYearlyCost / regularVisitsPerYear
def betaCostPerVisitRegular : ℕ := betaYearlyCost / regularVisitsPerYear

def alphaCostPerVisitMood : ℕ := alphaYearlyCost / moodPatternVisitsPerYear
def betaCostDuringMood : ℕ := betaCostPerMonth * 8  -- Only 8 months visited
def betaCostPerVisitMood : ℕ := betaCostDuringMood / moodPatternVisitsPerYear

/-
  Theorems to be proven.
-/
theorem alphaMoreAdvantageousRegular : alphaCostPerVisitRegular < betaCostPerVisitRegular := 
sorry

theorem betaMoreAdvantageousMood : betaCostPerVisitMood < alphaCostPerVisitMood := 
sorry

end alphaMoreAdvantageousRegular_betaMoreAdvantageousMood_l53_53229


namespace max_distance_from_circle_to_line_l53_53476

theorem max_distance_from_circle_to_line :
  ∀ (P : ℝ × ℝ), (P.1 - 1)^2 + P.2^2 = 9 →
  ∀ (x y : ℝ), 5 * x + 12 * y + 8 = 0 →
  ∃ (d : ℝ), d = 4 :=
by
  -- Proof is omitted as instructed.
  sorry

end max_distance_from_circle_to_line_l53_53476


namespace sum_of_center_coordinates_l53_53462

def center_of_circle_sum (x y : ℝ) : Prop :=
  (x - 6)^2 + (y + 5)^2 = 101

theorem sum_of_center_coordinates : center_of_circle_sum x y → x + y = 1 :=
sorry

end sum_of_center_coordinates_l53_53462


namespace min_value_sum_l53_53646

def positive_real (x : ℝ) : Prop := x > 0

theorem min_value_sum (x y : ℝ) (hx : positive_real x) (hy : positive_real y)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : x + y ≥ 20 :=
sorry

end min_value_sum_l53_53646


namespace average_side_length_of_squares_l53_53383

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l53_53383


namespace negation_of_no_slow_learners_attend_school_l53_53536

variable {α : Type}
variable (SlowLearner : α → Prop) (AttendsSchool : α → Prop)

-- The original statement
def original_statement : Prop := ∀ x, SlowLearner x → ¬ AttendsSchool x

-- The corresponding negation
def negation_statement : Prop := ∃ x, SlowLearner x ∧ AttendsSchool x

-- The proof problem statement
theorem negation_of_no_slow_learners_attend_school : 
  ¬ original_statement SlowLearner AttendsSchool ↔ negation_statement SlowLearner AttendsSchool := by
  sorry

end negation_of_no_slow_learners_attend_school_l53_53536


namespace find_pairs_l53_53278

theorem find_pairs (n p : ℕ) (hp : Prime p) (hnp : n ≤ 2 * p) (hdiv : (p - 1) * n + 1 % n^(p-1) = 0) :
  (n = 1 ∧ Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end find_pairs_l53_53278


namespace radius_of_convergence_zeta_eq_e_l53_53193

noncomputable def xi (n : ℕ) : ℝ → ℝ := sorry -- Define the sequence of random variables

noncomputable def zeta (n : ℕ) : ℝ := ∏ i in finset.range (n+1), xi i 0

def radius_of_convergence (a : ℕ → ℝ) : ℝ :=
  1 / real.limsup (λ n, (real.abs (a n)) ^ (1 / (n:ℝ)))

theorem radius_of_convergence_zeta_eq_e :
  ∀ (ξ : ℕ → ℝ) (h_indep : ∀ i j, i ≠ j → probabilistic_independence (ξ i) (ξ j)) 
  (h_uniform : ∀ n, uniform ξ n),
  radius_of_convergence (λ n, ∏ i in finset.range (n + 1), ξ i) = real.exp 1 :=
by sorry

end radius_of_convergence_zeta_eq_e_l53_53193


namespace find_ordered_pair_l53_53346

theorem find_ordered_pair (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroot : ∀ x : ℝ, 2 * x^2 + a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1 / 2, -3 / 4) := 
  sorry

end find_ordered_pair_l53_53346


namespace perimeter_of_regular_polygon_l53_53056

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53056


namespace angle_mul_add_proof_solve_equation_proof_l53_53119

-- For (1)
def angle_mul_add_example : Prop :=
  let a := 34 * 3600 + 25 * 60 + 20 -- 34°25'20'' to seconds
  let b := 35 * 60 + 42 * 60        -- 35°42' to total minutes
  let result := a * 3 + b * 60      -- Multiply a by 3 and convert b to seconds
  let final_result := result / 3600 -- Convert back to degrees
  final_result = 138 + (58 / 60)

-- For (2)
def solve_equation_example : Prop :=
  ∀ x : ℚ, (x + 1) / 2 - 1 = (2 - 3 * x) / 3 → x = 1 / 9

theorem angle_mul_add_proof : angle_mul_add_example := sorry

theorem solve_equation_proof : solve_equation_example := sorry

end angle_mul_add_proof_solve_equation_proof_l53_53119


namespace jane_cycling_time_difference_l53_53821

theorem jane_cycling_time_difference :
  (3 * 5 / 6.5 - (5 / 10 + 5 / 5 + 5 / 8)) * 60 = 11 :=
by sorry

end jane_cycling_time_difference_l53_53821


namespace find_expression_l53_53136

variables {x y : ℝ}

theorem find_expression
  (h1: 3 * x + y = 5)
  (h2: x + 3 * y = 6)
  : 10 * x^2 + 13 * x * y + 10 * y^2 = 97 :=
by
  sorry

end find_expression_l53_53136


namespace spherical_to_rectangular_coords_l53_53922

noncomputable def sphericalToRectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * (Real.sin phi) * (Real.cos theta), 
   rho * (Real.sin phi) * (Real.sin theta), 
   rho * (Real.cos phi))

theorem spherical_to_rectangular_coords :
  sphericalToRectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_coords_l53_53922


namespace triangle_proof_l53_53940

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions
axiom cos_rule_1 : a / cos A = c / (2 - cos C)
axiom b_value : b = 4
axiom c_value : c = 3
axiom area_equation : (1 / 2) * a * b * sin C = 3

-- The theorem statement
theorem triangle_proof : 3 * sin C + 4 * cos C = 5 := sorry

end triangle_proof_l53_53940


namespace quadratic_root_range_l53_53148

noncomputable def quadratic_function (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range (a : ℝ) (h : a ≠ 0) (h_distinct_roots : ∃ x1 x2 : ℝ, quadratic_function a x1 = 0 ∧ quadratic_function a x2 = 0 ∧ x1 ≠ x2 ∧ x1 < 1 ∧ x2 > 1) :
    -(2 / 11) < a ∧ a < 0 :=
sorry

end quadratic_root_range_l53_53148


namespace remainder_of_13_pow_a_mod_37_l53_53831

theorem remainder_of_13_pow_a_mod_37 (a : ℕ) (h_pos : a > 0) (h_mult : ∃ k : ℕ, a = 3 * k) : (13^a) % 37 = 1 := 
sorry

end remainder_of_13_pow_a_mod_37_l53_53831


namespace second_cart_travel_distance_l53_53408

-- Given definitions:
def first_cart_first_term : ℕ := 6
def first_cart_common_difference : ℕ := 8
def second_cart_first_term : ℕ := 7
def second_cart_common_difference : ℕ := 9

-- Given times:
def time_first_cart : ℕ := 35
def time_second_cart : ℕ := 33

-- Arithmetic series sum formula
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Total distance traveled by the second cart
noncomputable def distance_second_cart : ℕ :=
  arithmetic_series_sum second_cart_first_term second_cart_common_difference time_second_cart

-- Theorem to prove the distance traveled by the second cart
theorem second_cart_travel_distance : distance_second_cart = 4983 :=
  sorry

end second_cart_travel_distance_l53_53408


namespace gcd_n_cube_plus_25_n_plus_3_l53_53635

theorem gcd_n_cube_plus_25_n_plus_3 (n : ℕ) (h : n > 3^2) : 
  Int.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 :=
by
  sorry

end gcd_n_cube_plus_25_n_plus_3_l53_53635


namespace person_next_to_Boris_arkady_galya_l53_53585

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l53_53585


namespace power_sum_is_99_l53_53915

theorem power_sum_is_99 : 3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 :=
by sorry

end power_sum_is_99_l53_53915


namespace total_texts_received_l53_53625

structure TextMessageScenario :=
  (textsBeforeNoon : Nat)
  (textsAtNoon : Nat)
  (textsAfterNoonDoubling : (Nat → Nat) → Nat)
  (textsAfter6pm : (Nat → Nat) → Nat)

def textsBeforeNoon := 21
def textsAtNoon := 2

-- Calculation for texts received from noon to 6 pm
def noonTo6pmTexts (textsAtNoon : Nat) : Nat :=
  let rec doubling (n : Nat) : Nat := match n with
    | 0 => textsAtNoon
    | n + 1 => 2 * (doubling n)
  (doubling 0) + (doubling 1) + (doubling 2) + (doubling 3) + (doubling 4) + (doubling 5)

def textsAfterNoonDoubling : (Nat → Nat) → Nat := λ doubling => noonTo6pmTexts 2

-- Calculation for texts received from 6 pm to midnight
def after6pmTexts (textsAt6pm : Nat) : Nat :=
  let rec decrease (n : Nat) : Nat := match n with
    | 0 => textsAt6pm
    | n + 1 => (decrease n) - 5
  (decrease 0) + (decrease 1) + (decrease 2) + (decrease 3) + (decrease 4) + (decrease 5) + (decrease 6)

def textsAfter6pm : (Nat → Nat) → Nat := λ decrease => after6pmTexts 64

theorem total_texts_received : textsBeforeNoon + (textsAfterNoonDoubling (λ x => x)) + (textsAfter6pm (λ x => x)) = 490 := by
  sorry
 
end total_texts_received_l53_53625


namespace greatest_value_inequality_l53_53282

theorem greatest_value_inequality (x : ℝ) :
  x^2 - 6 * x + 8 ≤ 0 → x ≤ 4 := 
sorry

end greatest_value_inequality_l53_53282


namespace fgf_of_3_l53_53194

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 4 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

-- The statement we need to prove
theorem fgf_of_3 : f (g (f 3)) = 332 := by
  sorry

end fgf_of_3_l53_53194


namespace count_valid_ys_l53_53238

theorem count_valid_ys : 
  ∃ ys : Finset ℤ, ys.card = 4 ∧ ∀ y ∈ ys, (y - 3 > 0) ∧ ((y + 3) * (y - 3) * (y^2 + 9) < 2000) :=
by
  sorry

end count_valid_ys_l53_53238


namespace determinant_scaled_matrix_l53_53285

-- Definitions based on the conditions given in the problem.
def determinant2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

variable (a b c d : ℝ)
variable (h : determinant2x2 a b c d = 5)

-- The proof statement to be filled, proving the correct answer.
theorem determinant_scaled_matrix :
  determinant2x2 (2 * a) (2 * b) (2 * c) (2 * d) = 20 :=
by
  sorry

end determinant_scaled_matrix_l53_53285


namespace compute_expression_l53_53165

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l53_53165


namespace regular_polygon_perimeter_l53_53047

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53047


namespace find_prime_and_integer_l53_53279

theorem find_prime_and_integer (p x : ℕ) (hp : Nat.Prime p) 
  (hx1 : 1 ≤ x) (hx2 : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (p, x) = (2, 1) ∨ (p, x) = (2, 2) ∨ (p, x) = (3, 1) ∨ (p, x) = (3, 3) ∨ ((p ≥ 5) ∧ (x = 1)) :=
by
  sorry

end find_prime_and_integer_l53_53279


namespace divisible_by_42_l53_53845

theorem divisible_by_42 (n : ℕ) : 42 ∣ (n^3 * (n^6 - 1)) :=
sorry

end divisible_by_42_l53_53845


namespace perimeter_of_regular_polygon_l53_53068

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53068


namespace pepper_remaining_l53_53117

/-- Brennan initially had 0.25 grams of pepper. He used 0.16 grams for scrambling eggs. 
His friend added x grams of pepper to another dish. Given y grams are remaining, 
prove that y = 0.09 + x . --/
theorem pepper_remaining (x y : ℝ) (h1 : 0.25 - 0.16 = 0.09) (h2 : y = 0.09 + x) : y = 0.09 + x := 
by
  sorry

end pepper_remaining_l53_53117


namespace vote_percentage_for_candidate_A_l53_53889

noncomputable def percent_democrats : ℝ := 0.60
noncomputable def percent_republicans : ℝ := 0.40
noncomputable def percent_voting_a_democrats : ℝ := 0.70
noncomputable def percent_voting_a_republicans : ℝ := 0.20

theorem vote_percentage_for_candidate_A :
    (percent_democrats * percent_voting_a_democrats + percent_republicans * percent_voting_a_republicans) * 100 = 50 := by
  sorry

end vote_percentage_for_candidate_A_l53_53889


namespace collinear_points_l53_53115

variable (α β γ δ E : Type)
variables {A B C D K L P Q : α}
variables (convex : α → α → α → α → Prop)
variables (not_parallel : α → α → Prop)
variables (internal_bisector : α → α → α → Prop)
variables (external_bisector : α → α → α → Prop)
variables (collinear : α → α → α → α → Prop)

axiom convex_quad : convex A B C D
axiom AD_not_parallel_BC : not_parallel A D ∧ not_parallel B C

axiom internal_bisectors :
  internal_bisector A B K ∧ internal_bisector B A K ∧ internal_bisector C D P ∧ internal_bisector D C P

axiom external_bisectors :
  external_bisector A B L ∧ external_bisector B A L ∧ external_bisector C D Q ∧ external_bisector D C Q

theorem collinear_points : collinear K L P Q := 
sorry

end collinear_points_l53_53115


namespace who_is_next_to_Boris_l53_53610

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l53_53610


namespace calculate_sum_l53_53622

theorem calculate_sum :
  (1 : ℚ) + 3 / 6 + 5 / 12 + 7 / 20 + 9 / 30 + 11 / 42 + 13 / 56 + 15 / 72 + 17 / 90 = 81 + 2 / 5 :=
sorry

end calculate_sum_l53_53622


namespace average_book_width_l53_53336

noncomputable def bookWidths : List ℝ := [5, 0.75, 1.5, 3, 12, 2, 7.5]

theorem average_book_width :
  (bookWidths.sum / bookWidths.length = 4.54) :=
by
  sorry

end average_book_width_l53_53336


namespace part1_part2_l53_53287

noncomputable def f (x : ℝ) (a : ℝ) := (a + 2 * (Real.cos (x / 2)) ^ 2) * Real.cos (x + Real.pi / 2)

-- Prove that a = -1 given the conditions
theorem part1 (h1 : f (Real.pi / 2) a = 0) : a = -1 := sorry

-- Prove the value of cos(π/6 - 2α) given the conditions
theorem part2 (h1 : f (Real.pi / 2) a = 0) (h2 : f (α / 2) (-1) = -2 / 5) (h3 : α ∈ set.Ioo (Real.pi / 2) Real.pi) :
  Real.cos (Real.pi / 6 - 2 * α) = (-7 * Real.sqrt 3 - 24) / 50 := sorry

end part1_part2_l53_53287


namespace acquaintances_condition_l53_53499

theorem acquaintances_condition (n : ℕ) (hn : n > 1) (acquainted : ℕ → ℕ → Prop) :
  (∀ X Y, acquainted X Y → acquainted Y X) ∧
  (∀ X, ¬acquainted X X) →
  (∀ n, n ≠ 2 → n ≠ 4 → ∃ (A B : ℕ), (∃ (C : ℕ), acquainted C A ∧ acquainted C B) ∨ (∃ (D : ℕ), ¬acquainted D A ∧ ¬acquainted D B)) :=
by
  intros
  sorry

end acquaintances_condition_l53_53499


namespace minimum_value_fraction_l53_53218

theorem minimum_value_fraction (m n : ℝ) (h1 : m + 4 * n = 1) (h2 : m > 0) (h3 : n > 0): 
  (1 / m + 4 / n) ≥ 25 :=
sorry

end minimum_value_fraction_l53_53218


namespace arithmetic_sqrt_of_frac_l53_53530

theorem arithmetic_sqrt_of_frac (a b : ℝ) (h : a = 1) (h' : b = 64) :
  Real.sqrt (a / b) = 1 / 8 :=
by
  rw [h, h']
  rw [Real.sqrt_div, Real.sqrt_one, Real.sqrt_eq_rpow, Real.rpow_nat_cast]
  norm_num
  exact zero_le_one
  exact zero_le_of_real (show b > 0 by norm_num)

end arithmetic_sqrt_of_frac_l53_53530


namespace number_of_tables_cost_price_l53_53854

theorem number_of_tables_cost_price
  (C S : ℝ)
  (N : ℝ)
  (h1 : N * C = 20 * S)
  (h2 : S = 0.75 * C) :
  N = 15 := by
  -- insert proof here
  sorry

end number_of_tables_cost_price_l53_53854


namespace average_alligators_l53_53523

theorem average_alligators (t s n : ℕ) (h1 : t = 50) (h2 : s = 20) (h3 : n = 3) :
  (t - s) / n = 10 :=
by 
  sorry

end average_alligators_l53_53523


namespace who_next_to_boris_l53_53617

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l53_53617


namespace classes_after_drop_remaining_hours_of_classes_per_day_l53_53824

def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_classes : ℕ := 1

theorem classes_after_drop 
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ) :
  initial_classes - dropped_classes = 3 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

theorem remaining_hours_of_classes_per_day
  (initial_classes : ℕ)
  (hours_per_class : ℕ)
  (dropped_classes : ℕ)
  (h : initial_classes - dropped_classes = 3) :
  hours_per_class * (initial_classes - dropped_classes) = 6 :=
by
  -- We are skipping the proof and using sorry for now.
  sorry

end classes_after_drop_remaining_hours_of_classes_per_day_l53_53824


namespace clerical_percentage_l53_53679

theorem clerical_percentage (total_employees clerical_fraction reduce_fraction: ℕ) 
  (h1 : total_employees = 3600) 
  (h2 : clerical_fraction = 1 / 3)
  (h3 : reduce_fraction = 1 / 2) : 
  ( (reduce_fraction * (clerical_fraction * total_employees)) / 
    (total_employees - reduce_fraction * (clerical_fraction * total_employees))) * 100 = 20 :=
by
  sorry

end clerical_percentage_l53_53679


namespace probability_first_head_second_tail_l53_53415

-- Conditions
def fair_coin := true
def prob_heads := 1 / 2
def prob_tails := 1 / 2
def independent_events (A B : Prop) := true

-- Statement
theorem probability_first_head_second_tail :
  fair_coin →
  independent_events (prob_heads = 1/2) (prob_tails = 1/2) →
  (prob_heads * prob_tails) = 1/4 :=
by
  sorry

end probability_first_head_second_tail_l53_53415


namespace cost_of_500_pencils_in_dollars_l53_53214

def cost_of_pencil := 3 -- cost of 1 pencil in cents
def pencils_quantity := 500 -- number of pencils
def cents_in_dollar := 100 -- number of cents in 1 dollar

theorem cost_of_500_pencils_in_dollars :
  (pencils_quantity * cost_of_pencil) / cents_in_dollar = 15 := by
    sorry

end cost_of_500_pencils_in_dollars_l53_53214


namespace lcm_24_36_40_l53_53421

-- Define the natural numbers 24, 36, and 40
def n1 : ℕ := 24
def n2 : ℕ := 36
def n3 : ℕ := 40

-- Define the prime factorization of each number
def factors_n1 := [2^3, 3^1] -- 24 = 2^3 * 3^1
def factors_n2 := [2^2, 3^2] -- 36 = 2^2 * 3^2
def factors_n3 := [2^3, 5^1] -- 40 = 2^3 * 5^1

-- Prove that the LCM of n1, n2, n3 is 360
theorem lcm_24_36_40 : Nat.lcm (Nat.lcm n1 n2) n3 = 360 := sorry

end lcm_24_36_40_l53_53421


namespace volume_calculation_l53_53448

-- Define the dimensions of the rectangular parallelepiped
def a : ℕ := 2
def b : ℕ := 3
def c : ℕ := 4

-- Define the radius for spheres and cylinders
def r : ℝ := 2

theorem volume_calculation : 
  let l := 384
  let o := 140
  let q := 3
  (l + o + q = 527) :=
by
  sorry

end volume_calculation_l53_53448


namespace farmer_field_m_value_l53_53431

theorem farmer_field_m_value (m : ℝ) 
    (h_length : ∀ m, m > -4 → 2 * m + 9 > 0) 
    (h_breadth : ∀ m, m > -4 → m - 4 > 0)
    (h_area : (2 * m + 9) * (m - 4) = 88) : 
    m = 7.5 :=
by
  sorry

end farmer_field_m_value_l53_53431


namespace _l53_53203

variables (a b c : ℝ)
-- Conditionally define the theorem giving the constraints in the context.
example (h1 : a < 0) (h2 : b < 0) (h3 : c > 0) : 
  abs a - abs (a + b) + abs (c - a) + abs (b - c) = 2 * c - a := by 
sorry

end _l53_53203


namespace team_selection_l53_53202

-- Define the number of boys and girls in the club
def boys : Nat := 10
def girls : Nat := 12

-- Define the number of boys and girls to be selected for the team
def boys_team : Nat := 4
def girls_team : Nat := 4

-- Calculate the number of combinations using Nat.choose
noncomputable def choosing_boys : Nat := Nat.choose boys boys_team
noncomputable def choosing_girls : Nat := Nat.choose girls girls_team

-- Calculate the total number of ways to form the team
noncomputable def total_combinations : Nat := choosing_boys * choosing_girls

-- Theorem stating the total number of combinations equals the correct answer
theorem team_selection :
  total_combinations = 103950 := by
  sorry

end team_selection_l53_53202


namespace total_income_percentage_l53_53834

-- Define the base income of Juan
def juan_base_income (J : ℝ) := J

-- Define Tim's base income
def tim_base_income (J : ℝ) := 0.70 * J

-- Define Mary's total income
def mary_total_income (J : ℝ) := 1.232 * J

-- Define Lisa's total income
def lisa_total_income (J : ℝ) := 0.6489 * J

-- Define Nina's total income
def nina_total_income (J : ℝ) := 1.3375 * J

-- Define the sum of the total incomes of Mary, Lisa, and Nina
def sum_income (J : ℝ) := mary_total_income J + lisa_total_income J + nina_total_income J

-- Define the statement we need to prove: the percentage of Juan's total income
theorem total_income_percentage (J : ℝ) (hJ : J ≠ 0) :
  ((sum_income J / juan_base_income J) * 100) = 321.84 :=
by
  unfold juan_base_income sum_income mary_total_income lisa_total_income nina_total_income
  sorry

end total_income_percentage_l53_53834


namespace min_value_expr_l53_53294

open Real

theorem min_value_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ c, c = 4 * sqrt 3 - 6 ∧ ∀ (z w : ℝ), z = x ∧ w = y → (3 * z) / (3 * z + 2 * w) + w / (2 * z + w) ≥ c :=
by
  sorry

end min_value_expr_l53_53294


namespace final_value_l53_53970

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l53_53970


namespace problem_statement_l53_53982

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l53_53982


namespace kelly_games_giveaway_l53_53730

theorem kelly_games_giveaway (n m g : ℕ) (h_current: n = 50) (h_left: m = 35) : g = n - m :=
by
  sorry

end kelly_games_giveaway_l53_53730


namespace count_points_in_intersection_is_7_l53_53304

def isPointInSetA (x y : ℤ) : Prop :=
  (x - 3)^2 + (y - 4)^2 ≤ (5 / 2)^2

def isPointInSetB (x y : ℤ) : Prop :=
  (x - 4)^2 + (y - 5)^2 > (5 / 2)^2

def isPointInIntersection (x y : ℤ) : Prop :=
  isPointInSetA x y ∧ isPointInSetB x y

def pointsInIntersection : List (ℤ × ℤ) :=
  [(1, 5), (1, 4), (1, 3), (2, 3), (3, 2), (3, 3), (3, 4)]

theorem count_points_in_intersection_is_7 :
  (List.length pointsInIntersection = 7)
  ∧ (∀ (p : ℤ × ℤ), p ∈ pointsInIntersection → isPointInIntersection p.fst p.snd) :=
by
  sorry

end count_points_in_intersection_is_7_l53_53304


namespace min_value_a_b_c_l53_53130

def A_n (a : ℕ) (n : ℕ) : ℕ := a * ((10^n - 1) / 9)
def B_n (b : ℕ) (n : ℕ) : ℕ := b * ((10^n - 1) / 9)
def C_n (c : ℕ) (n : ℕ) : ℕ := c * ((10^(2*n) - 1) / 9)

theorem min_value_a_b_c (a b c : ℕ) (Ha : 0 < a ∧ a < 10) (Hb : 0 < b ∧ b < 10) (Hc : 0 < c ∧ c < 10) :
  (∃ n1 n2 : ℕ, (n1 ≠ n2) ∧ (C_n c n1 - A_n a n1 = B_n b n1 ^ 2) ∧ (C_n c n2 - A_n a n2 = B_n b n2 ^ 2)) →
  a + b + c = 5 :=
by
  sorry

end min_value_a_b_c_l53_53130


namespace peter_twice_as_old_in_years_l53_53323

def mother_age : ℕ := 60
def harriet_current_age : ℕ := 13
def peter_current_age : ℕ := mother_age / 2
def years_later : ℕ := 4

theorem peter_twice_as_old_in_years : 
  peter_current_age + years_later = 2 * (harriet_current_age + years_later) :=
by
  -- using given conditions 
  -- Peter's current age is 30
  -- Harriet's current age is 13
  -- years_later is 4
  sorry

end peter_twice_as_old_in_years_l53_53323


namespace persons_next_to_Boris_l53_53598

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l53_53598


namespace line_passes_point_a_ne_zero_l53_53399

theorem line_passes_point_a_ne_zero (a : ℝ) (h1 : ∀ (x y : ℝ), (y = 5 * x + a) → (x = a ∧ y = a^2)) (h2 : a ≠ 0) : a = 6 :=
sorry

end line_passes_point_a_ne_zero_l53_53399


namespace person_next_to_Boris_arkady_galya_l53_53589

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l53_53589


namespace problem_proof_l53_53160

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l53_53160


namespace cakes_donated_l53_53628
-- Import necessary libraries for arithmetic operations and proofs

-- Define the conditions and required proof in Lean
theorem cakes_donated (c : ℕ) (h : 8 * c + 4 * c + 2 * c = 140) : c = 10 :=
by
  sorry

end cakes_donated_l53_53628


namespace value_of_h_h_2_is_353_l53_53828

def h (x : ℕ) : ℕ := 3 * x^2 - x + 1

theorem value_of_h_h_2_is_353 : h (h 2) = 353 := 
by
  sorry

end value_of_h_h_2_is_353_l53_53828


namespace values_of_a_and_b_l53_53320

theorem values_of_a_and_b (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2 * x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) → (a = 2 ∧ b = 1) :=
by 
  sorry

end values_of_a_and_b_l53_53320


namespace arithmetic_progression_even_terms_l53_53661

theorem arithmetic_progression_even_terms (a d n : ℕ) (h_even : n % 2 = 0)
  (h_last_first_diff : (n - 1) * d = 16)
  (h_sum_odd : n * (a + (n - 2) * d / 2) = 81)
  (h_sum_even : n * (a + d + (n - 2) * d / 2) = 75) :
  n = 8 :=
by sorry

end arithmetic_progression_even_terms_l53_53661


namespace parallelogram_base_length_l53_53005

theorem parallelogram_base_length (A H : ℝ) (base : ℝ) 
    (hA : A = 72) (hH : H = 6) (h_area : A = base * H) : base = 12 := 
by 
  sorry

end parallelogram_base_length_l53_53005


namespace find_smaller_number_l53_53226

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : y = 8 :=
by sorry

end find_smaller_number_l53_53226


namespace bill_has_correct_final_amount_l53_53470

def initial_amount : ℕ := 42
def pizza_cost : ℕ := 11
def pizzas_bought : ℕ := 3
def bill_initial_amount : ℕ := 30
def amount_spent := pizzas_bought * pizza_cost
def frank_remaining_amount := initial_amount - amount_spent
def bill_final_amount := bill_initial_amount + frank_remaining_amount

theorem bill_has_correct_final_amount : bill_final_amount = 39 := by
  sorry

end bill_has_correct_final_amount_l53_53470


namespace regular_polygon_perimeter_is_28_l53_53070

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53070


namespace domain_sqrt_tan_x_sub_sqrt3_l53_53780

open Real

noncomputable def domain := {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2}

theorem domain_sqrt_tan_x_sub_sqrt3 :
  {x | ∃ y : ℝ, y = sqrt (tan x - sqrt 3)} = domain :=
by
  sorry

end domain_sqrt_tan_x_sub_sqrt3_l53_53780


namespace domain_of_ln_function_l53_53731

-- We need to define the function and its domain condition.
def function_domain (x : ℝ) : Prop := x^2 - x > 0

theorem domain_of_ln_function : {x : ℝ | function_domain x} = set.Iio 0 ∪ set.Ioi 1 :=
by
  sorry

end domain_of_ln_function_l53_53731


namespace Bill_has_39_dollars_l53_53468

noncomputable def Frank_initial_money : ℕ := 42
noncomputable def pizza_cost : ℕ := 11
noncomputable def num_pizzas : ℕ := 3
noncomputable def Bill_initial_money : ℕ := 30

noncomputable def Frank_spent : ℕ := pizza_cost * num_pizzas
noncomputable def Frank_remaining_money : ℕ := Frank_initial_money - Frank_spent
noncomputable def Bill_final_money : ℕ := Bill_initial_money + Frank_remaining_money

theorem Bill_has_39_dollars :
  Bill_final_money = 39 :=
by
  sorry

end Bill_has_39_dollars_l53_53468


namespace solution_to_diameter_area_problem_l53_53502

def diameter_area_problem : Prop :=
  let radius := 4
  let area_of_shaded_region := 16 + 8 * Real.pi
  -- Definitions derived directly from conditions
  let circle_radius := radius
  let diameter1_perpendicular_to_diameter2 := True
  -- Conclusively prove the area of the shaded region
  ∀ (PQ RS : ℝ) (h1 : PQ = 2 * circle_radius) (h2 : RS = 2 * circle_radius) (h3 : diameter1_perpendicular_to_diameter2),
  ∃ (area : ℝ), area = area_of_shaded_region

-- This is just the statement, the proof part is omitted.
theorem solution_to_diameter_area_problem : diameter_area_problem :=
  sorry

end solution_to_diameter_area_problem_l53_53502


namespace average_price_of_5_baskets_l53_53423

/-- Saleem bought 4 baskets with an average cost of $4 each. --/
def average_cost_first_4_baskets : ℝ := 4

/-- Saleem buys the fifth basket with the price of $8. --/
def price_fifth_basket : ℝ := 8

/-- Prove that the average price of the 5 baskets is $4.80. --/
theorem average_price_of_5_baskets :
  (4 * average_cost_first_4_baskets + price_fifth_basket) / 5 = 4.80 := 
by
  sorry

end average_price_of_5_baskets_l53_53423


namespace expected_defective_chips_in_60000_l53_53727

def shipmentS1 := (2, 5000)
def shipmentS2 := (4, 12000)
def shipmentS3 := (2, 15000)
def shipmentS4 := (4, 16000)

def total_defective_chips := shipmentS1.1 + shipmentS2.1 + shipmentS3.1 + shipmentS4.1
def total_chips := shipmentS1.2 + shipmentS2.2 + shipmentS3.2 + shipmentS4.2

def defective_ratio := total_defective_chips / total_chips
def shipment60000 := 60000

def expected_defectives (ratio : ℝ) (total_chips : ℝ) := ratio * total_chips

theorem expected_defective_chips_in_60000 :
  expected_defectives defective_ratio shipment60000 = 15 :=
by
  sorry

end expected_defective_chips_in_60000_l53_53727


namespace fuel_consumption_l53_53842

def initial_volume : ℕ := 3000
def volume_jan_1 : ℕ := 180
def volume_may_1 : ℕ := 1238
def refill_volume : ℕ := 3000

theorem fuel_consumption :
  (initial_volume - volume_jan_1) + (refill_volume - volume_may_1) = 4582 := by
  sorry

end fuel_consumption_l53_53842


namespace a_ge_zero_of_set_nonempty_l53_53473

theorem a_ge_zero_of_set_nonempty {a : ℝ} (h : ∃ x : ℝ, x^2 = a) : a ≥ 0 :=
sorry

end a_ge_zero_of_set_nonempty_l53_53473


namespace factorize_equivalence_l53_53461

-- declaring that the following definition may not be computable
noncomputable def factorize_expression (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 = x * y * (x + y)

-- theorem to state the proof problem
theorem factorize_equivalence (x y : ℝ) : factorize_expression x y :=
sorry

end factorize_equivalence_l53_53461


namespace points_PQST_colinear_l53_53430

variables {A B C K L S T P Q : Point} {ω ℓ : Line} 

-- Definitions and conditions
def tangent_circle_tangency_points (ω : Line) (A B C : Point) : Prop :=
  ω.TangentAt B ∧ ω.TangentAt C

def line_intersects_segments (ℓ : Line) (K L : Point) (AB AC : Segment) : Prop :=
  ℓ ∩ AB = {K} ∧ ℓ ∩ AC = {L}

def circle_intersects_line (ω : Line) (ℓ : Line) (P Q : Point) : Prop :=
  ω ∩ ℓ = {P, Q}

def points_on_segment_with_parallel_segments
  (K S L T : Point) (B C : Segment) : Prop :=
  S ∈ B ∧ T ∈ C ∧ (KS ∥ AC) ∧ (LT ∥ AB)

-- The theorem to prove
theorem points_PQST_colinear 
  (h1 : tangent_circle_tangency_points ω A B C)
  (h2 : line_intersects_segments ℓ K L AB AC)
  (h3 : circle_intersects_line ω ℓ P Q)
  (h4 : points_on_segment_with_parallel_segments K S L T BC) : 
  CyclicQuad P Q S T :=
by
  sorry

end points_PQST_colinear_l53_53430


namespace arithmetic_sequence_sum_condition_l53_53501

variable (a : ℕ → ℤ)

theorem arithmetic_sequence_sum_condition (h1 : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) : 
  a 2 + a 10 = 120 :=
sorry

end arithmetic_sequence_sum_condition_l53_53501


namespace cost_of_500_pencils_is_15_dollars_l53_53215

-- Defining the given conditions
def cost_per_pencil_cents : ℕ := 3
def pencils_count : ℕ := 500
def cents_to_dollars : ℕ := 100

-- The proof problem: statement only, no proof provided
theorem cost_of_500_pencils_is_15_dollars :
  (cost_per_pencil_cents * pencils_count) / cents_to_dollars = 15 :=
by
  sorry

end cost_of_500_pencils_is_15_dollars_l53_53215


namespace balls_in_boxes_l53_53154

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end balls_in_boxes_l53_53154


namespace div_gt_sum_div_sq_l53_53520

theorem div_gt_sum_div_sq (n d d' : ℕ) (h₁ : d' > d) (h₂ : d ∣ n) (h₃ : d' ∣ n) : 
  d' > d + d * d / n :=
by 
  sorry

end div_gt_sum_div_sq_l53_53520


namespace perimeter_of_regular_polygon_l53_53060

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53060


namespace haley_stickers_l53_53487

theorem haley_stickers (friends : ℕ) (stickers_per_friend : ℕ) (total_stickers : ℕ) :
  friends = 9 → stickers_per_friend = 8 → total_stickers = friends * stickers_per_friend → total_stickers = 72 :=
by
  intros h_friends h_stickers_per_friend h_total_stickers
  rw [h_friends, h_stickers_per_friend] at h_total_stickers
  exact h_total_stickers

end haley_stickers_l53_53487


namespace expected_value_of_X_l53_53761

noncomputable def expected_value_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

theorem expected_value_of_X (n : ℕ) (h : n > 0) :
  let X : ℕ → ℕ → ℚ := λ i n, (i - 1 : ℚ) / n
  let E_X : ℚ := ∑ i in Finset.range n, X (i + 1) n
  E_X = expected_value_shorter_gentlemen n := by
{
  sorry
}

end expected_value_of_X_l53_53761


namespace domain_of_f_l53_53396

def condition1 (x : ℝ) : Prop := 4 - |x| ≥ 0
def condition2 (x : ℝ) : Prop := (x^2 - 5 * x + 6) / (x - 3) > 0

theorem domain_of_f (x : ℝ) :
  (condition1 x) ∧ (condition2 x) ↔ ((2 < x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_l53_53396


namespace range_of_a_l53_53794

variable (f : ℝ → ℝ)

-- f is an odd function
def odd_function : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 1: f is an odd function
axiom h_odd : odd_function f

-- Condition 2: f(x) + f(x + 3 / 2) = 0 for any real number x
axiom h_periodicity : ∀ x : ℝ, f x + f (x + 3 / 2) = 0

-- Condition 3: f(1) > 1
axiom h_f1 : f 1 > 1

-- Condition 4: f(2) = a for some real number a
variable (a : ℝ)
axiom h_f2 : f 2 = a

-- Goal: Prove that a < -1
theorem range_of_a : a < -1 :=
  sorry

end range_of_a_l53_53794


namespace line_always_passes_fixed_point_l53_53034

theorem line_always_passes_fixed_point (m : ℝ) :
  m * 1 + (1 - m) * 2 + m - 2 = 0 :=
by
  sorry

end line_always_passes_fixed_point_l53_53034


namespace largest_value_l53_53720

theorem largest_value :
  max (max (max (max (4^2) (4 * 2)) (4 - 2)) (4 / 2)) (4 + 2) = 4^2 :=
by sorry

end largest_value_l53_53720


namespace lucy_lovely_age_ratio_l53_53785

theorem lucy_lovely_age_ratio (L l : ℕ) (x : ℕ) (h1 : L = 50) (h2 : 45 = x * (l - 5)) (h3 : 60 = 2 * (l + 10)) :
  (45 / (l - 5)) = 3 :=
by
  sorry

end lucy_lovely_age_ratio_l53_53785


namespace no_all_perfect_squares_l53_53333

theorem no_all_perfect_squares (x : ℤ) 
  (h1 : ∃ a : ℤ, 2 * x - 1 = a^2) 
  (h2 : ∃ b : ℤ, 5 * x - 1 = b^2) 
  (h3 : ∃ c : ℤ, 13 * x - 1 = c^2) : 
  False :=
sorry

end no_all_perfect_squares_l53_53333


namespace sum_of_fractions_l53_53966

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l53_53966


namespace average_side_lengths_l53_53363

open Real

theorem average_side_lengths (A1 A2 A3 : ℝ) (h1 : A1 = 25) (h2 : A2 = 64) (h3 : A3 = 144) :
  ((sqrt A1) + (sqrt A2) + (sqrt A3)) / 3 = 25 / 3 :=
by 
  -- To be filled in the proof later
  sorry

end average_side_lengths_l53_53363


namespace min_value_of_expression_l53_53930

theorem min_value_of_expression (n : ℕ) (h_pos : n > 0) : n = 8 → (n / 2 + 32 / n) = 8 :=
by sorry

end min_value_of_expression_l53_53930


namespace primes_with_large_gap_exists_l53_53339

noncomputable def exists_primes_with_large_gap_and_composites_between : Prop :=
  ∃ p q : ℕ, p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p > 2015 ∧ (∀ n : ℕ, p < n ∧ n < q → ¬Nat.Prime n)

theorem primes_with_large_gap_exists : exists_primes_with_large_gap_and_composites_between := sorry

end primes_with_large_gap_exists_l53_53339


namespace trees_variance_l53_53682

theorem trees_variance :
  let groups := [3, 4, 3]
  let trees := [5, 6, 7]
  let n := 10
  let mean := (5 * 3 + 6 * 4 + 7 * 3) / n
  let variance := (3 * (5 - mean)^2 + 4 * (6 - mean)^2 + 3 * (7 - mean)^2) / n
  variance = 0.6 := 
by
  sorry

end trees_variance_l53_53682


namespace perimeter_of_polygon_l53_53039

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53039


namespace polygon_perimeter_l53_53098

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53098


namespace units_digit_of_a_l53_53016

theorem units_digit_of_a :
  (2003^2004 - 2004^2003) % 10 = 7 :=
by
  sorry

end units_digit_of_a_l53_53016


namespace perimeter_of_polygon_l53_53043

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53043


namespace tangent_line_to_circle_l53_53397

-- Definitions derived directly from the conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0
def passes_through_point (l : ℝ → ℝ → Prop) : Prop := l (-1) 6

-- The statement to be proven
theorem tangent_line_to_circle :
  ∃ (l : ℝ → ℝ → Prop), passes_through_point l ∧ 
    ((∀ x y, l x y ↔ 3*x - 4*y + 27 = 0) ∨ 
     (∀ x y, l x y ↔ x + 1 = 0)) :=
sorry

end tangent_line_to_circle_l53_53397


namespace maria_bought_9_hardcover_volumes_l53_53924

def total_volumes (h p : ℕ) : Prop := h + p = 15
def total_cost (h p : ℕ) : Prop := 10 * p + 30 * h = 330

theorem maria_bought_9_hardcover_volumes (h p : ℕ) (h_vol : total_volumes h p) (h_cost : total_cost h p) : h = 9 :=
by
  sorry

end maria_bought_9_hardcover_volumes_l53_53924


namespace perimeter_of_regular_polygon_l53_53057

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l53_53057


namespace expression_value_l53_53997

variables (p q r : ℝ)

theorem expression_value (h1 : p + q + r = 5) (h2 : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
sorry

end expression_value_l53_53997


namespace inequality_solutions_l53_53465

theorem inequality_solutions :
  (∀ x : ℝ, 2 * x / (x + 1) < 1 ↔ -1 < x ∧ x < 1) ∧
  (∀ a x : ℝ,
    (x^2 + (2 - a) * x - 2 * a ≥ 0 ↔
      (a = -2 → True) ∧
      (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧
      (a < -2 → (x ≤ a ∨ x ≥ -2)))) :=
by
  sorry

end inequality_solutions_l53_53465


namespace det_scaled_matrix_l53_53639

variable (a b c d : ℝ)
variable (h : Matrix.det ![![a, b], ![c, d]] = 5)

theorem det_scaled_matrix : Matrix.det ![![3 * a, 3 * b], ![4 * c, 4 * d]] = 60 := by
  sorry

end det_scaled_matrix_l53_53639


namespace total_length_of_board_l53_53732

theorem total_length_of_board (x y : ℝ) (h1 : y = 2 * x) (h2 : y = 46) : x + y = 69 :=
by
  sorry

end total_length_of_board_l53_53732


namespace boris_neighbors_l53_53584

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l53_53584


namespace value_of_2_Z_6_l53_53319

def Z (a b : ℝ) : ℝ := b + 10 * a - a^2

theorem value_of_2_Z_6 : Z 2 6 = 22 :=
by
  sorry

end value_of_2_Z_6_l53_53319


namespace expression_value_l53_53953

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l53_53953


namespace persons_next_to_Boris_l53_53600

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l53_53600


namespace fred_seashells_l53_53471

-- Define the initial number of seashells Fred found.
def initial_seashells : ℕ := 47

-- Define the number of seashells Fred gave to Jessica.
def seashells_given : ℕ := 25

-- Prove that Fred now has 22 seashells.
theorem fred_seashells : initial_seashells - seashells_given = 22 :=
by
  sorry

end fred_seashells_l53_53471


namespace regular_polygon_perimeter_l53_53083

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53083


namespace toys_produced_each_day_l53_53725

def toys_produced_per_week : ℕ := 6000
def work_days_per_week : ℕ := 4

theorem toys_produced_each_day :
  (toys_produced_per_week / work_days_per_week) = 1500 := 
by
  -- The details of the proof are omitted
  -- The correct answer given the conditions is 1500 toys
  sorry

end toys_produced_each_day_l53_53725


namespace problem_statement_l53_53981

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l53_53981


namespace regular_polygon_perimeter_l53_53090

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53090


namespace problem_statement_l53_53641

variable {x y : Real}

theorem problem_statement (hx : x * y < 0) (hxy : x > |y|) : x + y > 0 := by
  sorry

end problem_statement_l53_53641


namespace problem_statement_l53_53974

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l53_53974


namespace middle_number_in_consecutive_nat_sum_squares_equals_2030_l53_53466

theorem middle_number_in_consecutive_nat_sum_squares_equals_2030 
  (n : ℕ)
  (h1 : (n - 1)^2 + n^2 + (n + 1)^2 = 2030)
  (h2 : (n^3 - n^2) % 7 = 0)
  : n = 26 := 
sorry

end middle_number_in_consecutive_nat_sum_squares_equals_2030_l53_53466


namespace average_side_length_of_squares_l53_53357

theorem average_side_length_of_squares (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) : 
  (real.sqrt a1 + real.sqrt a2 + real.sqrt a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53357


namespace MrKishoreSavings_l53_53911

noncomputable def TotalExpenses : ℕ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

noncomputable def MonthlySalary : ℕ :=
  (TotalExpenses * 10) / 9

noncomputable def Savings : ℕ :=
  (MonthlySalary * 1) / 10

theorem MrKishoreSavings :
  Savings = 2300 :=
by
  sorry

end MrKishoreSavings_l53_53911


namespace check_ratio_l53_53237

theorem check_ratio (initial_balance check_amount new_balance : ℕ) 
  (h1 : initial_balance = 150) (h2 : check_amount = 50) (h3 : new_balance = initial_balance + check_amount) :
  (check_amount : ℚ) / new_balance = 1 / 4 := 
by { 
  sorry 
}

end check_ratio_l53_53237


namespace distance_between_vertices_hyperbola_l53_53774

theorem distance_between_vertices_hyperbola : 
  ∀ {x y : ℝ}, (x^2 / 121 - y^2 / 49 = 1) → (11 * 2 = 22) :=
by
  sorry

end distance_between_vertices_hyperbola_l53_53774


namespace sum_product_smallest_number_l53_53224

theorem sum_product_smallest_number (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 80) : min x y = 8 :=
  sorry

end sum_product_smallest_number_l53_53224


namespace ellipse_slope_condition_l53_53325

theorem ellipse_slope_condition (a b x y x₀ y₀ : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h_ellipse1 : x^2 / a^2 + y^2 / b^2 = 1) 
  (h_ellipse2 : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (hA : x ≠ x₀ ∨ y ≠ y₀) 
  (hB : x ≠ -x₀ ∨ y ≠ -y₀) :
  ((y - y₀) / (x - x₀)) * ((y + y₀) / (x + x₀)) = -b^2 / a^2 := 
sorry

end ellipse_slope_condition_l53_53325


namespace problem1_problem2_problem3_problem4_problem5_problem6_l53_53919

-- Problem 1
theorem problem1 : (-20 + 3 - (-5) - 7 : Int) = -19 := sorry

-- Problem 2
theorem problem2 : (-2.4 - 3.7 - 4.6 + 5.7 : Real) = -5 := sorry

-- Problem 3
theorem problem3 : (-0.25 + ((-3 / 7) * (4 / 5)) : Real) = (-83 / 140) := sorry

-- Problem 4
theorem problem4 : ((-1 / 2) * (-8) + (-6)^2 : Real) = 40 := sorry

-- Problem 5
theorem problem5 : ((-1 / 12 - 1 / 36 + 1 / 6) * (-36) : Real) = -2 := sorry

-- Problem 6
theorem problem6 : (-1^4 + (-2) + (-1 / 3) - abs (-9) : Real) = -37 / 3 := sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l53_53919


namespace find_certain_number_l53_53316

theorem find_certain_number 
  (num : ℝ)
  (h1 : num / 14.5 = 177)
  (h2 : 29.94 / 1.45 = 17.7) : 
  num = 2566.5 := 
by 
  sorry

end find_certain_number_l53_53316


namespace average_side_length_of_squares_l53_53380

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53380


namespace inequality_x_y_z_squares_l53_53937

theorem inequality_x_y_z_squares (x y z m : ℝ) (h : x + y + z = m) : x^2 + y^2 + z^2 ≥ (m^2) / 3 := by
  sorry

end inequality_x_y_z_squares_l53_53937


namespace three_digit_2C4_not_multiple_of_5_l53_53787

theorem three_digit_2C4_not_multiple_of_5 : ∀ C : ℕ, C < 10 → ¬(∃ n : ℕ, 2 * 100 + C * 10 + 4 = 5 * n) :=
by
  sorry

end three_digit_2C4_not_multiple_of_5_l53_53787


namespace regular_polygon_perimeter_l53_53052

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53052


namespace who_is_next_to_boris_l53_53595

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l53_53595


namespace smallest_x_of_quadratic_eqn_l53_53003

theorem smallest_x_of_quadratic_eqn : ∃ x : ℝ, (12*x^2 - 44*x + 40 = 0) ∧ x = 5 / 3 :=
by
  sorry

end smallest_x_of_quadratic_eqn_l53_53003


namespace division_remainder_l53_53917

theorem division_remainder : 1234567 % 112 = 0 := 
by 
  sorry

end division_remainder_l53_53917


namespace computation_of_expression_l53_53172

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l53_53172


namespace sam_hourly_rate_l53_53204

theorem sam_hourly_rate
  (first_month_earnings : ℕ)
  (second_month_earnings : ℕ)
  (total_hours : ℕ)
  (h1 : first_month_earnings = 200)
  (h2 : second_month_earnings = first_month_earnings + 150)
  (h3 : total_hours = 55) :
  (first_month_earnings + second_month_earnings) / total_hours = 10 := 
  by
  sorry

end sam_hourly_rate_l53_53204


namespace number_of_people_l53_53127

-- Definitions based on conditions
def per_person_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 100 else max 72 (100 - 2 * (x - 30))

def total_cost (x : ℕ) : ℕ :=
  x * per_person_cost x

-- Main theorem statement
theorem number_of_people (x : ℕ) (h1 : total_cost x = 3150) (h2 : x > 30) : x = 35 :=
by {
  sorry
}

end number_of_people_l53_53127


namespace minimum_value_of_fraction_l53_53132

theorem minimum_value_of_fraction (a b : ℝ) (h1 : a > 2 * b) (h2 : 2 * b > 0) :
  (a^4 + 1) / (b * (a - 2 * b)) >= 16 :=
sorry

end minimum_value_of_fraction_l53_53132


namespace tip_is_24_l53_53694

-- Definitions based on conditions
def women's_haircut_cost : ℕ := 48
def children's_haircut_cost : ℕ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℚ := 0.20

-- Calculating total cost and tip amount
def total_cost : ℕ := women's_haircut_cost + (number_of_children * children's_haircut_cost)
def tip_amount : ℚ := tip_percentage * total_cost

-- Lean theorem statement based on the problem
theorem tip_is_24 : tip_amount = 24 := by
  sorry

end tip_is_24_l53_53694


namespace regular_decagon_interior_angle_l53_53411

-- Define the number of sides in a regular decagon
def n : ℕ := 10

-- Define the formula for the sum of the interior angles of an n-sided polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the measure of one interior angle of a regular decagon
def one_interior_angle_of_regular_polygon (sum_of_angles : ℕ) (n : ℕ) : ℕ :=
  sum_of_angles / n

-- Prove that the measure of one interior angle of a regular decagon is 144 degrees
theorem regular_decagon_interior_angle : one_interior_angle_of_regular_polygon (sum_of_interior_angles 10) 10 = 144 := by
  sorry

end regular_decagon_interior_angle_l53_53411


namespace touchdowns_points_l53_53696

theorem touchdowns_points 
    (num_touchdowns : ℕ) (total_points : ℕ) 
    (h1 : num_touchdowns = 3) 
    (h2 : total_points = 21) : 
    total_points / num_touchdowns = 7 :=
by
    sorry

end touchdowns_points_l53_53696


namespace contradiction_assumption_l53_53853

variable (x y z : ℝ)

/-- The negation of "at least one is positive" for proof by contradiction is 
    "all three numbers are non-positive". -/
theorem contradiction_assumption (h : ¬ (x > 0 ∨ y > 0 ∨ z > 0)) : 
  (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) :=
by
  sorry

end contradiction_assumption_l53_53853


namespace expenses_each_month_l53_53271
noncomputable def total_expenses (worked_hours1 worked_hours2 worked_hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (total_left : ℕ) : ℕ :=
  (worked_hours1 * rate1) + (worked_hours2 * rate2) + (worked_hours3 * rate3) - total_left

theorem expenses_each_month (hours1 : ℕ)
  (hours2 : ℕ)
  (hours3 : ℕ)
  (rate1 : ℕ)
  (rate2 : ℕ)
  (rate3 : ℕ)
  (left_over : ℕ) :
  hours1 = 20 → 
  rate1 = 10 →
  hours2 = 30 →
  rate2 = 20 →
  hours3 = 5 →
  rate3 = 40 →
  left_over = 500 → 
  total_expenses hours1 hours2 hours3 rate1 rate2 rate3 left_over = 500 := by
  intros h1 r1 h2 r2 h3 r3 l
  sorry

end expenses_each_month_l53_53271


namespace add_55_result_l53_53526

theorem add_55_result (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 :=
sorry

end add_55_result_l53_53526


namespace eight_x_plus_y_l53_53806

theorem eight_x_plus_y (x y z : ℝ) (h1 : x + 2 * y - 3 * z = 7) (h2 : 2 * x - y + 2 * z = 6) : 
  8 * x + y = 32 :=
sorry

end eight_x_plus_y_l53_53806


namespace value_of_x_l53_53811

theorem value_of_x (x : ℝ) (h1 : (x^2 - 4) / (x + 2) = 0) : x = 2 := by
  sorry

end value_of_x_l53_53811


namespace probability_of_B_given_A_l53_53142

-- Conditions
def total_bottles : ℕ := 6
def qualified_bottles : ℕ := 4
def unqualified_bottles : ℕ := 2
def draw_without_replacement : Prop := true

-- Event A: Drawing an unqualified disinfectant the first time
def event_A : Prop := true
-- Event B: Drawing an unqualified disinfectant the second time
def event_B : Prop := true

-- The probability of drawing an unqualified disinfectant the second time given that an unqualified disinfectant was drawn the first time
def P_B_given_A : ℚ := 1 / 5

theorem probability_of_B_given_A (htotal : total_bottles = 6)
                                 (hqual : qualified_bottles = 4)
                                 (hunqual : unqualified_bottles = 2)
                                 (hdraw : draw_without_replacement) 
                                 (hA : event_A) 
                                 (hB : event_B) : 
  P_B_given_A = 1 / 5 :=
sorry

end probability_of_B_given_A_l53_53142


namespace problem_proof_l53_53163

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l53_53163


namespace tip_calculation_l53_53693

def women's_haircut_cost : ℝ := 48
def children's_haircut_cost : ℝ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℝ := 0.20

theorem tip_calculation : 
  let total_cost := (children's_haircut_cost * number_of_children) + women's_haircut_cost in
  let tip := total_cost * tip_percentage in
  tip = 24 :=
by 
  sorry

end tip_calculation_l53_53693


namespace sum_unchanged_difference_changes_l53_53014

-- Definitions from conditions
def original_sum (a b c : ℤ) := a + b + c
def new_first (a : ℤ) := a - 329
def new_second (b : ℤ) := b + 401

-- Problem statement for sum unchanged
theorem sum_unchanged (a b c : ℤ) (h : original_sum a b c = 1281) :
  original_sum (new_first a) (new_second b) (c - 72) = 1281 := by
  sorry

-- Definitions for difference condition
def abs_diff (x y : ℤ) := abs (x - y)
def alter_difference (a b c : ℤ) :=
  abs_diff (new_first a) (new_second b) + abs_diff (new_first a) c + abs_diff b c

-- Problem statement addressing the difference
theorem difference_changes (a b c : ℤ) (h : original_sum a b c = 1281) :
  alter_difference a b c = abs_diff (new_first a) (new_second b) + abs_diff (c - 730) (new_first a) + abs_diff (c - 730) (new_first a) := by
  sorry

end sum_unchanged_difference_changes_l53_53014


namespace intersection_points_l53_53011

theorem intersection_points (k : ℝ) : ∃ (P : ℝ × ℝ), P = (1, 0) ∧ ∀ x y : ℝ, (kx - y - k = 0) → (x^2 + y^2 = 2) → ∃ y1 y2 : ℝ, (y = y1 ∨ y = y2) :=
by
  sorry

end intersection_points_l53_53011


namespace solution_set_of_inequality_l53_53704

theorem solution_set_of_inequality :
  { x : ℝ | (2 * x - 1) / (x + 1) ≤ 1 } = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end solution_set_of_inequality_l53_53704


namespace mr_smith_children_l53_53678

noncomputable def gender_probability (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let equal_gender_ways := Nat.choose n (n / 2)
  let favourable_outcomes := total_outcomes - equal_gender_ways
  favourable_outcomes / total_outcomes

theorem mr_smith_children (n : ℕ) (h : n = 8) : 
  gender_probability n = 93 / 128 :=
by
  rw [h]
  sorry

end mr_smith_children_l53_53678


namespace root_constraints_between_zero_and_twoR_l53_53343

variable (R l a : ℝ)
variable (hR : R > 0) (hl : l > 0) (ha_nonzero : a ≠ 0)

theorem root_constraints_between_zero_and_twoR :
  ∀ (x : ℝ), (2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2 = 0) →
  (0 < x ∧ x < 2 * R) ↔
  (a > 0 ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (a < 0 ∧ -2 * R < a ∧ l^2 < (2 * R - a)^2) :=
sorry

end root_constraints_between_zero_and_twoR_l53_53343


namespace falcons_win_probability_l53_53349

noncomputable def probability_falcons_win_at_least_five_games (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), if i ≥ k then (@Nat.choose n i : ℝ) * p^i * (1 - p)^(n - i) else 0

theorem falcons_win_probability 
  : probability_falcons_win_at_least_five_games 9 (1/2) 5 = 1/2 :=
sorry

end falcons_win_probability_l53_53349


namespace find_values_of_p_l53_53631

def geometric_progression (p : ℝ) : Prop :=
  (2 * p)^2 = (4 * p + 5) * |p - 3|

theorem find_values_of_p :
  {p : ℝ | geometric_progression p} = {-1, 15 / 8} :=
by
  sorry

end find_values_of_p_l53_53631


namespace find_m_l53_53475

open Classical

variable {d : ℤ} (h₁ : d ≠ 0) (a : ℕ → ℤ)

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∃ a₀ : ℤ, ∀ n, a n = a₀ + n * d

theorem find_m 
  (h_seq : arithmetic_sequence a d)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : ∃ m, a m = 8) :
  ∃ m, m = 8 :=
sorry

end find_m_l53_53475


namespace no_equal_partition_product_l53_53686

theorem no_equal_partition_product (n : ℕ) (h : n > 1) : 
  ¬ ∃ A B : Finset ℕ, 
    (A ∪ B = (Finset.range n).erase 0 ∧ A ∩ B = ∅ ∧ (A ≠ ∅) ∧ (B ≠ ∅) 
    ∧ A.prod id = B.prod id) := 
sorry

end no_equal_partition_product_l53_53686


namespace caterer_cheapest_option_l53_53850

theorem caterer_cheapest_option :
  ∃ x : ℕ, x ≥ 42 ∧ (∀ y : ℕ, y ≥ x → (20 * y < 120 + 18 * y) ∧ (20 * y < 250 + 14 * y)) := 
by
  sorry

end caterer_cheapest_option_l53_53850


namespace t_range_l53_53630

noncomputable def exists_nonneg_real_numbers_satisfying_conditions (t : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
  (3 * x^2 + 3 * z * x + z^2 = 1) ∧ 
  (3 * y^2 + 3 * y * z + z^2 = 4) ∧ 
  (x^2 - x * y + y^2 = t)

theorem t_range : ∀ t : ℝ, exists_nonneg_real_numbers_satisfying_conditions t → 
  (t ≥ (3 - Real.sqrt 5) / 2 ∧ t ≤ 1) :=
sorry

end t_range_l53_53630


namespace no_integer_solutions_l53_53330

theorem no_integer_solutions (w l : ℕ) (hw_pos : 0 < w) (hl_pos : 0 < l) : 
  (w * l = 24 ∧ (w = l ∨ 2 * l = w)) → false :=
by 
  sorry

end no_integer_solutions_l53_53330


namespace associate_professor_charts_l53_53116

theorem associate_professor_charts (A B C : ℕ) : 
  A + B = 8 → 
  2 * A + B = 10 → 
  C * A + 2 * B = 14 → 
  C = 1 := 
by 
  intros h1 h2 h3 
  sorry

end associate_professor_charts_l53_53116


namespace trip_length_l53_53029

theorem trip_length 
  (total_time : ℝ) (canoe_speed : ℝ) (hike_speed : ℝ) (hike_distance : ℝ)
  (hike_time_eq : hike_distance / hike_speed = 5.4) 
  (canoe_time_eq : total_time - hike_distance / hike_speed = 0.1)
  (canoe_distance_eq : canoe_speed * (total_time - hike_distance / hike_speed) = 1.2)
  (total_time_val : total_time = 5.5)
  (canoe_speed_val : canoe_speed = 12)
  (hike_speed_val : hike_speed = 5)
  (hike_distance_val : hike_distance = 27) :
  total_time = 5.5 → canoe_speed = 12 → hike_speed = 5 → hike_distance = 27 → hike_distance + canoe_speed * (total_time - hike_distance / hike_speed) = 28.2 := 
by
  intro h_total_time h_canoe_speed h_hike_speed h_hike_distance
  rw [h_total_time, h_canoe_speed, h_hike_speed, h_hike_distance]
  sorry

end trip_length_l53_53029


namespace perimeter_of_polygon_l53_53040

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53040


namespace gain_percentage_l53_53906

variables (C S : ℝ) (hC : C > 0)
variables (hS : S > 0)

def cost_price := 25 * C
def selling_price := 25 * S
def gain := 10 * S 

theorem gain_percentage (h_eq : 25 * S = 25 * C + 10 * S):
  (S = C) → 
  ((gain / cost_price) * 100 = 40) :=
by
  sorry

end gain_percentage_l53_53906


namespace factorize_equivalence_l53_53460

-- declaring that the following definition may not be computable
noncomputable def factorize_expression (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 = x * y * (x + y)

-- theorem to state the proof problem
theorem factorize_equivalence (x y : ℝ) : factorize_expression x y :=
sorry

end factorize_equivalence_l53_53460


namespace regular_polygon_perimeter_l53_53081

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53081


namespace probability_of_exactly_three_blue_marbles_l53_53925

-- Define the conditions
def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def total_selections : ℕ := 6
def blue_selections : ℕ := 3
def blue_probability : ℚ := 8 / 15
def red_probability : ℚ := 7 / 15
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability formula calculation
def binomial_probability : ℚ :=
  binomial_coefficient total_selections blue_selections * (blue_probability ^ blue_selections) * (red_probability ^ (total_selections - blue_selections))

-- The hypothesis (conditions) and conclusion (the solution)
theorem probability_of_exactly_three_blue_marbles :
  binomial_probability = (3512320 / 11390625) :=
by sorry

end probability_of_exactly_three_blue_marbles_l53_53925


namespace total_amount_paid_is_correct_l53_53867

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end total_amount_paid_is_correct_l53_53867


namespace only_one_passes_prob_l53_53345

variable (P_A P_B P_C : ℚ)
variable (only_one_passes : ℚ)

def prob_A := 4 / 5 
def prob_B := 3 / 5
def prob_C := 7 / 10

def prob_only_A := prob_A * (1 - prob_B) * (1 - prob_C)
def prob_only_B := (1 - prob_A) * prob_B * (1 - prob_C)
def prob_only_C := (1 - prob_A) * (1 - prob_B) * prob_C

def prob_sum : ℚ := prob_only_A + prob_only_B + prob_only_C

theorem only_one_passes_prob : prob_sum = 47 / 250 := 
by sorry

end only_one_passes_prob_l53_53345


namespace total_pens_count_l53_53017

def total_pens (red black blue : ℕ) : ℕ :=
  red + black + blue

theorem total_pens_count :
  let red := 8
  let black := red + 10
  let blue := red + 7
  total_pens red black blue = 41 :=
by
  sorry

end total_pens_count_l53_53017


namespace true_proposition_is_b_l53_53948

open Real

theorem true_proposition_is_b :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (¬ ∀ n : ℝ, n^2 ≥ n) ∧
  (¬ ∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧
  (¬ ∀ n : ℝ, n^2 < n) :=
  by
    sorry

end true_proposition_is_b_l53_53948


namespace donuts_distribution_l53_53762

theorem donuts_distribution (kinds total min_each : ℕ) (h_kinds : kinds = 4) (h_total : total = 7) (h_min_each : min_each = 1) :
  ∃ n : ℕ, n = 20 := by
  sorry

end donuts_distribution_l53_53762


namespace polynomial_evaluation_l53_53769

theorem polynomial_evaluation 
  (x : ℝ) (h : x^2 - 3*x - 10 = 0 ∧ x > 0) :
  x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 :=
sorry

end polynomial_evaluation_l53_53769


namespace union_A_B_intersection_A_CI_B_l53_53802

-- Define the sets
def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 5, 6, 7}

-- Define the complement of B in the universal set I
def C_I (I : Set ℕ) (B : Set ℕ) : Set ℕ := {x ∈ I | x ∉ B}

-- The theorem for the union of A and B
theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6, 7} := sorry

-- The theorem for the intersection of A and the complement of B in I
theorem intersection_A_CI_B : A ∩ (C_I I B) = {1, 2, 4} := sorry

end union_A_B_intersection_A_CI_B_l53_53802


namespace tangent_line_at_0_l53_53281

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem tangent_line_at_0 : 
  let m := (deriv f 0)
  let y₀ := f 0
  in ∀ x, (y = -x - 2) := 
by 
  sorry

end tangent_line_at_0_l53_53281


namespace pencil_price_l53_53222

variable (P N : ℕ) -- This assumes the price of a pencil (P) and the price of a notebook (N) are natural numbers (non-negative integers).

-- Define the conditions
def conditions : Prop :=
  (P + N = 950) ∧ (N = P + 150)

-- The theorem to prove
theorem pencil_price (h : conditions P N) : P = 400 :=
by
  sorry

end pencil_price_l53_53222


namespace impossible_four_teams_tie_possible_three_teams_tie_l53_53935

-- Definitions for the conditions
def num_teams : ℕ := 4
def num_matches : ℕ := (num_teams * (num_teams - 1)) / 2
def total_possible_outcomes : ℕ := 2^num_matches
def winning_rate : ℚ := 1 / 2

-- Problem 1: It is impossible for exactly four teams to tie for first place.
theorem impossible_four_teams_tie :
  ¬ ∃ (score : ℕ), (∀ (team : ℕ) (h : team < num_teams), team = score ∧
                     (num_teams * score = num_matches / 2 ∧
                      num_teams * score + num_matches / 2 = num_matches)) := sorry

-- Problem 2: It is possible for exactly three teams to tie for first place.
theorem possible_three_teams_tie :
  ∃ (score : ℕ), (∃ (teamA teamB teamC teamD : ℕ),
  (teamA < num_teams ∧ teamB < num_teams ∧ teamC < num_teams ∧ teamD <num_teams ∧ teamA ≠ teamB ∧ teamA ≠ teamC ∧ teamA ≠ teamD ∧ 
  teamB ≠ teamC ∧ teamB ≠ teamD ∧ teamC ≠ teamD)) ∧
  (teamA = score ∧ teamB = score ∧ teamC = score ∧ teamD = 0) := sorry

end impossible_four_teams_tie_possible_three_teams_tie_l53_53935


namespace algebraic_expression_value_l53_53168

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l53_53168


namespace initial_money_jennifer_l53_53667

theorem initial_money_jennifer (M : ℝ) (h1 : (1/5) * M + (1/6) * M + (1/2) * M + 12 = M) : M = 90 :=
sorry

end initial_money_jennifer_l53_53667


namespace fuel_oil_used_l53_53840

theorem fuel_oil_used (V_initial : ℕ) (V_jan : ℕ) (V_may : ℕ) : 
  (V_initial - V_jan) + (V_initial - V_may) = 4582 :=
by
  let V_initial := 3000
  let V_jan := 180
  let V_may := 1238
  sorry

end fuel_oil_used_l53_53840


namespace find_ratio_a6_b6_l53_53508

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def T (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def b (n : ℕ) : ℕ := sorry

theorem find_ratio_a6_b6 
  (H1 : ∀ n: ℕ, n > 0 → (S n / T n : ℚ) = n / (2 * n + 1)) :
  (a 6 / b 6 : ℚ) = 11 / 23 :=
sorry

end find_ratio_a6_b6_l53_53508


namespace vector_collinearity_l53_53485

variables (a b : ℝ × ℝ)

def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vector_collinearity : collinear (-1, 2) (1, -2) :=
by
  sorry

end vector_collinearity_l53_53485


namespace description_of_S_l53_53672

noncomputable def S := {p : ℝ × ℝ | (3 = (p.1 + 2) ∧ p.2 - 5 ≤ 3) ∨ 
                                      (3 = (p.2 - 5) ∧ p.1 + 2 ≤ 3) ∨ 
                                      (p.1 + 2 = p.2 - 5 ∧ 3 ≤ p.1 + 2 ∧ 3 ≤ p.2 - 5)}

theorem description_of_S :
  S = {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 ≤ 8) ∨ 
                    (p.2 = 8 ∧ p.1 ≤ 1) ∨ 
                    (p.2 = p.1 + 7 ∧ p.1 ≥ 1 ∧ p.2 ≥ 8)} :=
sorry

end description_of_S_l53_53672


namespace sarah_stamp_collection_value_l53_53205

theorem sarah_stamp_collection_value :
  ∀ (stamps_owned total_value_for_4_stamps : ℝ) (num_stamps_single_series : ℕ), 
  stamps_owned = 20 → 
  total_value_for_4_stamps = 10 → 
  num_stamps_single_series = 4 → 
  (stamps_owned / num_stamps_single_series) * (total_value_for_4_stamps / num_stamps_single_series) = 50 :=
by
  intros stamps_owned total_value_for_4_stamps num_stamps_single_series 
  intro h_stamps_owned
  intro h_total_value_for_4_stamps
  intro h_num_stamps_single_series
  rw [h_stamps_owned, h_total_value_for_4_stamps, h_num_stamps_single_series]
  sorry

end sarah_stamp_collection_value_l53_53205


namespace distance_to_other_asymptote_is_8_l53_53474

-- Define the hyperbola and the properties
def hyperbola (x y : ℝ) : Prop := (x^2) / 2 - (y^2) / 8 = 1

-- Define the asymptotes
def asymptote_1 (x y : ℝ) : Prop := y = 2 * x
def asymptote_2 (x y : ℝ) : Prop := y = -2 * x

-- Given conditions
variables (P : ℝ × ℝ)
variable (distance_to_one_asymptote : ℝ)
variable (distance_to_other_asymptote : ℝ)

axiom point_on_hyperbola : hyperbola P.1 P.2
axiom distance_to_one_asymptote_is_1_over_5 : distance_to_one_asymptote = 1 / 5

-- The proof statement
theorem distance_to_other_asymptote_is_8 :
  distance_to_other_asymptote = 8 := sorry

end distance_to_other_asymptote_is_8_l53_53474


namespace noncongruent_triangles_count_l53_53655

/-- Prove the number of noncongruent integer-sided triangles with positive area,
    perimeter less than 20, that are neither equilateral, isosceles, nor right triangles
    is 17 -/
theorem noncongruent_triangles_count:
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ s → a + b + c < 20 ∧ a + b > c ∧ a < b ∧ b < c ∧ 
         ¬(a = b ∨ b = c ∨ a = c) ∧ ¬(a * a + b * b = c * c)) ∧ 
    s.card = 17 := 
sorry

end noncongruent_triangles_count_l53_53655


namespace points_A_B_D_collinear_l53_53286

variable (a b : ℝ)

theorem points_A_B_D_collinear
  (AB : ℝ × ℝ := (a, 5 * b))
  (BC : ℝ × ℝ := (-2 * a, 8 * b))
  (CD : ℝ × ℝ := (3 * a, -3 * b)) :
  AB = (BC.1 + CD.1, BC.2 + CD.2) := 
by
  sorry

end points_A_B_D_collinear_l53_53286


namespace distance_closer_to_R_after_meeting_l53_53544

def distance_between_R_and_S : ℕ := 80
def rate_of_man_from_R : ℕ := 5
def initial_rate_of_man_from_S : ℕ := 4

theorem distance_closer_to_R_after_meeting 
  (t : ℕ) 
  (x : ℕ) 
  (h1 : t ≠ 0) 
  (h2 : distance_between_R_and_S = 80) 
  (h3 : rate_of_man_from_R = 5) 
  (h4 : initial_rate_of_man_from_S = 4) 
  (h5 : (rate_of_man_from_R * t) 
        + (t * initial_rate_of_man_from_S 
        + ((t - 1) * t / 2)) = distance_between_R_and_S) :
  x = 20 :=
sorry

end distance_closer_to_R_after_meeting_l53_53544


namespace regular_polygon_perimeter_l53_53079

theorem regular_polygon_perimeter (side_length : ℕ) (exterior_angle : ℕ) (h1 : side_length = 7) (h2 : exterior_angle = 90) :
  ∃ (n : ℕ), (360 / n = exterior_angle) ∧ (n = 4) ∧ (perimeter = 4 * side_length) ∧ (perimeter = 28) :=
by
  sorry

end regular_polygon_perimeter_l53_53079


namespace problem_statement_l53_53159

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) : b < 0 ∧ |b| > |a| :=
by
  sorry

end problem_statement_l53_53159


namespace smallest_sum_ab_l53_53792

theorem smallest_sum_ab (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 2^10 * 3^6 = a^b) : a + b = 866 :=
sorry

end smallest_sum_ab_l53_53792


namespace simon_gift_bags_l53_53525

theorem simon_gift_bags (rate_per_day : ℕ) (days : ℕ) (total_bags : ℕ) :
  rate_per_day = 42 → days = 13 → total_bags = rate_per_day * days → total_bags = 546 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end simon_gift_bags_l53_53525


namespace perimeter_of_polygon_l53_53037

theorem perimeter_of_polygon : 
  ∀ (side_length : ℝ) (exterior_angle : ℝ), 
  side_length = 7 → exterior_angle = 90 → 
  let n := 360 / exterior_angle in
  let P := n * side_length in 
  P = 28 := 
by 
  intros side_length exterior_angle h1 h2 
  dsimp [n, P]
  have h3 : n = 360 / exterior_angle := rfl 
  rw [h2] at h3 
  simp at h3 
  have h4 : P = n * side_length := rfl 
  rw [h1, h3] at h4 
  simp at h4 
  exact h4 
  sorry 

end perimeter_of_polygon_l53_53037


namespace polygon_perimeter_l53_53095

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53095


namespace regular_polygon_perimeter_is_28_l53_53074

-- Given conditions
def side_length := 7
def exterior_angle := 90

-- Mathematically equivalent proof problem
theorem regular_polygon_perimeter_is_28 :
  ∀ n : ℕ, (2 * n + 2) * side_length = 28 :=
by intros n; sorry

end regular_polygon_perimeter_is_28_l53_53074


namespace negative_integer_solution_l53_53707

theorem negative_integer_solution (N : ℤ) (h1 : N < 0) (h2 : N^2 + N = 6) : N = -3 := 
by 
  sorry

end negative_integer_solution_l53_53707


namespace container_capacity_l53_53895

theorem container_capacity (C : ℝ) (h1 : 0.30 * C + 36 = 0.75 * C) : C = 80 :=
by
  sorry

end container_capacity_l53_53895


namespace number_of_meetings_l53_53340

noncomputable def selena_radius : ℝ := 70
noncomputable def bashar_radius : ℝ := 80
noncomputable def selena_speed : ℝ := 200
noncomputable def bashar_speed : ℝ := 240
noncomputable def active_time_together : ℝ := 30

noncomputable def selena_circumference : ℝ := 2 * Real.pi * selena_radius
noncomputable def bashar_circumference : ℝ := 2 * Real.pi * bashar_radius

noncomputable def selena_angular_speed : ℝ := (selena_speed / selena_circumference) * (2 * Real.pi)
noncomputable def bashar_angular_speed : ℝ := (bashar_speed / bashar_circumference) * (2 * Real.pi)

noncomputable def relative_angular_speed : ℝ := selena_angular_speed + bashar_angular_speed
noncomputable def time_to_meet_once : ℝ := (2 * Real.pi) / relative_angular_speed

theorem number_of_meetings : Int := 
    ⌊active_time_together / time_to_meet_once⌋

example : number_of_meetings = 21 := by
  sorry

end number_of_meetings_l53_53340


namespace regular_polygon_perimeter_l53_53087

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ)
  (h1 : side_length = 7)  (h2 : exterior_angle = 90) 
  (h3 : exterior_angle = 360 / n) : 
  (side_length * n = 28) := by
  sorry

end regular_polygon_perimeter_l53_53087


namespace quadrilateral_area_is_6_l53_53569

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨3, 1⟩
def D : Point := ⟨5, 5⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

def quadrilateral_area (A B C D : Point) : ℝ :=
  area_triangle A B C + area_triangle A C D

theorem quadrilateral_area_is_6 : quadrilateral_area A B C D = 6 :=
  sorry

end quadrilateral_area_is_6_l53_53569


namespace average_side_lengths_l53_53373

theorem average_side_lengths (a1 a2 a3 : ℝ) (h1 : a1 = 25) (h2 : a2 = 64) (h3 : a3 = 144) :
  (√a1 + √a2 + √a3) / 3 = 25 / 3 :=
by
  sorry

end average_side_lengths_l53_53373


namespace dig_site_date_l53_53260

theorem dig_site_date (S F T Fourth : ℤ) 
  (h₁ : F = S - 352)
  (h₂ : T = F + 3700)
  (h₃ : Fourth = 2 * T)
  (h₄ : Fourth = 8400) : S = 852 := 
by 
  sorry

end dig_site_date_l53_53260


namespace bake_sale_earnings_eq_400_l53_53756

/-
  The problem statement derived from the given bake sale problem.
  We are to verify that the bake sale earned 400 dollars.
-/

def total_donation (bake_sale_earnings : ℕ) :=
  ((bake_sale_earnings - 100) / 2) + 10

theorem bake_sale_earnings_eq_400 (X : ℕ) (h : total_donation X = 160) : X = 400 :=
by
  sorry

end bake_sale_earnings_eq_400_l53_53756


namespace average_side_length_of_squares_l53_53378

theorem average_side_length_of_squares :
  let a₁ := 25
  let a₂ := 64
  let a₃ := 144
  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃
  (s₁ + s₂ + s₃) / 3 = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53378


namespace person_next_to_Boris_arkady_galya_l53_53586

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l53_53586


namespace ratio_of_areas_l53_53184

theorem ratio_of_areas (h a b R : ℝ) (h_triangle : a^2 + b^2 = h^2) (h_circumradius : R = h / 2) :
  (π * R^2) / (1/2 * a * b) = π * h / (4 * R) :=
by sorry

end ratio_of_areas_l53_53184


namespace graph_intersect_points_l53_53691

-- Define f as a function defined on all real numbers and invertible
variable (f : ℝ → ℝ) (hf : Function.Injective f)

-- Define the theorem to find the number of intersection points
theorem graph_intersect_points : 
  ∃ (n : ℕ), n = 3 ∧ ∃ (x : ℝ), (f (x^2) = f (x^6)) :=
  by
    -- Outline sketch: We aim to show there are 3 real solutions satisfying the equation
    -- The proof here is skipped, hence we put sorry
    sorry

end graph_intersect_points_l53_53691


namespace red_apples_count_l53_53926

theorem red_apples_count
  (r y g : ℕ)
  (h1 : r = y)
  (h2 : g = 2 * r)
  (h3 : r + y + g = 28) : r = 7 :=
sorry

end red_apples_count_l53_53926


namespace computation_of_expression_l53_53174

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end computation_of_expression_l53_53174


namespace y_intercept_of_parallel_line_l53_53517

-- Define the conditions for the problem
def line_parallel (m1 m2 : ℝ) : Prop := 
  m1 = m2

def point_on_line (m : ℝ) (b x1 y1 : ℝ) : Prop := 
  y1 = m * x1 + b

-- Define the main problem statement
theorem y_intercept_of_parallel_line (m b1 b2 x1 y1 : ℝ) 
  (h1 : line_parallel m 3) 
  (h2 : point_on_line m b1 x1 y1) 
  (h3 : x1 = 1) 
  (h4 : y1 = 2) 
  : b1 = -1 :=
sorry

end y_intercept_of_parallel_line_l53_53517


namespace cobs_count_l53_53428

theorem cobs_count (bushel_weight : ℝ) (ear_weight : ℝ) (num_bushels : ℕ)
  (h1 : bushel_weight = 56) (h2 : ear_weight = 0.5) (h3 : num_bushels = 2) : 
  ((num_bushels * bushel_weight) / ear_weight) = 224 :=
by 
  sorry

end cobs_count_l53_53428


namespace base7_addition_sum_l53_53257

theorem base7_addition_sum :
  let n1 := 256
  let n2 := 463
  let n3 := 132
  n1 + n2 + n3 = 1214 := sorry

end base7_addition_sum_l53_53257


namespace h_even_if_g_odd_l53_53196

structure odd_function (g : ℝ → ℝ) : Prop :=
(odd : ∀ x : ℝ, g (-x) = -g x)

def h (g : ℝ → ℝ) (x : ℝ) : ℝ := abs (g (x^5))

theorem h_even_if_g_odd (g : ℝ → ℝ) (hg : odd_function g) : ∀ x : ℝ, h g x = h g (-x) :=
by
  sorry

end h_even_if_g_odd_l53_53196


namespace quadratic_distinct_roots_example_l53_53789

theorem quadratic_distinct_roots_example {b c : ℝ} (hb : b = 1) (hc : c = 0) :
    (b^2 - 4 * c) > 0 := by
  sorry

end quadratic_distinct_roots_example_l53_53789


namespace liquid_X_percentage_36_l53_53208

noncomputable def liquid_X_percentage (m : ℕ) (pX : ℕ) (m_evaporate : ℕ) (m_add : ℕ) (p_add : ℕ) : ℕ :=
  let m_X_initial := (pX * m / 100)
  let m_water_initial := ((100 - pX) * m / 100)
  let m_X_after_evaporation := m_X_initial
  let m_water_after_evaporation := m_water_initial - m_evaporate
  let m_X_additional := (p_add * m_add / 100)
  let m_water_additional := ((100 - p_add) * m_add / 100)
  let m_X_new := m_X_after_evaporation + m_X_additional
  let m_water_new := m_water_after_evaporation + m_water_additional
  let m_total_new := m_X_new + m_water_new
  (m_X_new * 100 / m_total_new)

theorem liquid_X_percentage_36 :
  liquid_X_percentage 10 30 2 2 30 = 36 := by
  sorry

end liquid_X_percentage_36_l53_53208


namespace total_amount_paid_is_correct_l53_53868

-- Define constants based on conditions
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The proof problem statement
theorem total_amount_paid_is_correct :
  total_cost = 360 :=
by
  sorry

end total_amount_paid_is_correct_l53_53868


namespace solve_weight_of_bowling_ball_l53_53453

-- Conditions: Eight bowling balls equal the weight of five canoes
-- and four canoes weigh 120 pounds.

def weight_of_bowling_ball : Prop :=
  ∃ (b c : ℝ), (8 * b = 5 * c) ∧ (4 * c = 120) ∧ (b = 18.75)

theorem solve_weight_of_bowling_ball : weight_of_bowling_ball :=
  sorry

end solve_weight_of_bowling_ball_l53_53453


namespace who_is_next_to_Boris_l53_53609

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l53_53609


namespace product_of_two_numbers_l53_53545

theorem product_of_two_numbers (a b : ℝ) 
  (h1 : a - b = 2 * k)
  (h2 : a + b = 8 * k)
  (h3 : 2 * a * b = 30 * k) : a * b = 15 :=
by
  sorry

end product_of_two_numbers_l53_53545


namespace geometric_sequence_common_ratio_l53_53786

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 3 = 2 * S 2 + 1) (h2 : a 4 = 2 * S 3 + 1) :
  ∃ q : ℝ, (q = 3) :=
by
  -- Proof will go here.
  sorry

end geometric_sequence_common_ratio_l53_53786


namespace person_next_to_Boris_arkady_galya_l53_53588

-- Define the people involved
inductive Person
| Arkady
| Boris
| Vera
| Galya
| Danya
| Egor

open Person

def next_to (p1 p2 : Person) (standing_next : Person → Person → Prop) : Prop :=
standing_next p1 p2 ∨ standing_next p2 p1

noncomputable def position_relationships : Prop :=
  ∃ (standing_next : Person → Person → Prop),
    -- Danya is next to Vera, on Vera's right side
    standing_next Danya Vera ∧
    -- Galya stood opposite Egor
    (∀ p, next_to p Galya standing_next → next_to p Egor standing_next) ∧
    -- Egor is next to Danya
    standing_next Egor Danya ∧
    -- Arkady and Galya did not want to stand next to each other
    ¬ next_to Arkady Galya standing_next ∧
    -- Conclusion: Arkady and Galya are standing next to Boris
    next_to Arkady Boris standing_next ∧ next_to Galya Boris standing_next

theorem person_next_to_Boris_arkady_galya : position_relationships :=
    sorry

end person_next_to_Boris_arkady_galya_l53_53588


namespace cost_price_percentage_l53_53393

theorem cost_price_percentage (MP CP : ℝ) (h_discount : 0.75 * MP = CP * 1.171875) :
  ((CP / MP) * 100) = 64 :=
by
  sorry

end cost_price_percentage_l53_53393


namespace smallest_value_N_l53_53255

theorem smallest_value_N (N : ℕ) (a b c : ℕ) (h1 : N = a * b * c) (h2 : (a - 1) * (b - 1) * (c - 1) = 252) : N = 392 :=
sorry

end smallest_value_N_l53_53255


namespace quadrant_conditions_l53_53317

-- Formalizing function and conditions in Lean specifics
variable {a b : ℝ}

theorem quadrant_conditions 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 0 < a ∧ a < 1)
  (h4 : ∀ x < 0, a^x + b - 1 > 0)
  (h5 : ∀ x > 0, a^x + b - 1 > 0) :
  0 < b ∧ b < 1 := 
sorry

end quadrant_conditions_l53_53317


namespace correct_statement_l53_53417

theorem correct_statement (a b c : ℝ) (h1 : ac = bc) (h2 : a = b) (h3 : a^2 = b^2) : 
  (∀ (c ≠ 0), (ac = bc → a = b)) ∧ 
  (∀ (c ≠ 0), (a / c = b / c → a = b)) ∧
  (a = b → a + 3 = b + 3) ∧ 
  (a^2 = b^2 → a = b) :=
by 
  sorry

end correct_statement_l53_53417


namespace problem_statement_l53_53975

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l53_53975


namespace probability_winning_game_show_l53_53032

open ProbabilityTheory
open Finset

noncomputable def probability_of_winning : ℚ :=
  let p_correct_one := (1 / 4) in
  let p_incorrect_one := (3 / 4) in
  let cases_4_correct := (p_correct_one ^ 4) in
  let cases_3_correct := (4.choose 3) * (p_correct_one ^ 3) * p_incorrect_one in
  cases_4_correct + cases_3_correct

theorem probability_winning_game_show :
  probability_of_winning = 13 / 256 :=
by
  simp [probability_of_winning]
  sorry

end probability_winning_game_show_l53_53032


namespace total_course_selection_schemes_l53_53898

theorem total_course_selection_schemes : 
    let total_courses := 8 in 
    let pe_courses := 4 in
    let art_courses := 4 in
    (∃ k, k = 2 ∨ k = 3) → 
    (∀ k, k = 2 → ∃ n1 n2, n1 = 1 ∧ n2 = 1 ∧ choose pe_courses n1 * choose art_courses n2 = 4 * 4) →
    (∀ k, k = 3 → 
        (∃ n1 n2, n1 = 2 ∧ n2 = 1 ∧ choose pe_courses n1 * choose art_courses n2 = 6 * 4) ∧ 
        (∃ n1 n2, n1 = 1 ∧ n2 = 2 ∧ choose pe_courses n1 * choose art_courses n2 = 4 * 6)) →
    (16 + 48 = 64) :=
begin
    sorry
end

end total_course_selection_schemes_l53_53898


namespace percentage_decrease_l53_53012

theorem percentage_decrease (original_salary new_salary decreased_salary : ℝ) (p : ℝ) (D : ℝ) : 
  original_salary = 4000.0000000000005 →
  p = 10 →
  new_salary = original_salary * (1 + p/100) →
  decreased_salary = 4180 →
  decreased_salary = new_salary * (1 - D / 100) →
  D = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_decrease_l53_53012


namespace average_of_side_lengths_of_squares_l53_53387

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l53_53387


namespace pyramid_volume_l53_53904

theorem pyramid_volume
  (s : ℝ) (h : ℝ) (base_area : ℝ) (triangular_face_area : ℝ) (surface_area : ℝ)
  (h_base_area : base_area = s * s)
  (h_triangular_face_area : triangular_face_area = (1 / 3) * base_area)
  (h_surface_area : surface_area = base_area + 4 * triangular_face_area)
  (h_surface_area_value : surface_area = 768)
  (h_vol : h = 7.78) :
  (1 / 3) * base_area * h = 853.56 :=
by
  sorry

end pyramid_volume_l53_53904


namespace perimeter_is_36_l53_53503

-- Define an equilateral triangle with a given side length
def equilateral_triangle_perimeter (side_length : ℝ) : ℝ :=
  3 * side_length

-- Given: The base of the equilateral triangle is 12 m
def base_length : ℝ := 12

-- Theorem: The perimeter of the equilateral triangle is 36 m
theorem perimeter_is_36 : equilateral_triangle_perimeter base_length = 36 :=
by
  -- Placeholder for the proof
  sorry

end perimeter_is_36_l53_53503


namespace arrangement_count_l53_53757

def basil_tomato_arrangements : ℕ :=
  let basil_positions := 4
  let total_positions := 6 -- 4 basil plants & 2 positions for tomato groups
  let choose_2_slots := Nat.choose total_positions 2
  let permutations := 2! * 2! -- permutations of two groups of 2 tomato plants
  choose_2_slots * permutations

theorem arrangement_count (basil tomato : Type) [Fintype basil] [Fintype tomato] [DecidableEq basil] [DecidableEq tomato]
  (h_basil : Fintype.card basil = 4) (h_tomato : Fintype.card tomato = 4) :
  basil_tomato_arrangements = 40 :=
by
  have : Fintype.card (Finset.univ : Finset basil) = 4 := h_basil
  have : Fintype.card (Finset.univ : Finset tomato) = 4 := h_tomato
  exact sorry

end arrangement_count_l53_53757


namespace sum_of_digits_base10_representation_l53_53663

def digit_sum (n : ℕ) : ℕ := sorry  -- Define a function to calculate the sum of digits

noncomputable def a : ℕ := 7 * (10 ^ 1234 - 1) / 9
noncomputable def b : ℕ := 2 * (10 ^ 1234 - 1) / 9
noncomputable def product : ℕ := 7 * a * b

theorem sum_of_digits_base10_representation : digit_sum product = 11100 := 
by sorry

end sum_of_digits_base10_representation_l53_53663


namespace leo_third_part_time_l53_53443

theorem leo_third_part_time :
  ∃ (T3 : ℕ), 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 3 → T = 25 * k) →
  T1 = 25 →
  T2 = 50 →
  Break1 = 10 →
  Break2 = 15 →
  TotalTime = 2 * 60 + 30 →
  (TotalTime - (T1 + Break1 + T2 + Break2) = T3) →
  T3 = 50 := 
sorry

end leo_third_part_time_l53_53443


namespace Barbara_Mike_ratio_is_one_half_l53_53677

-- Define the conditions
def Mike_age_current : ℕ := 16
def Mike_age_future : ℕ := 24
def Barbara_age_future : ℕ := 16

-- Define Barbara's current age based on the conditions
def Barbara_age_current : ℕ := Mike_age_current - (Mike_age_future - Barbara_age_future)

-- Define the ratio of Barbara's age to Mike's age
def ratio_Barbara_Mike : ℚ := Barbara_age_current / Mike_age_current

-- Prove that the ratio is 1:2
theorem Barbara_Mike_ratio_is_one_half : ratio_Barbara_Mike = 1 / 2 := by
  sorry

end Barbara_Mike_ratio_is_one_half_l53_53677


namespace farmer_apples_after_giving_l53_53701

-- Define the initial number of apples and the number of apples given to the neighbor
def initial_apples : ℕ := 127
def given_apples : ℕ := 88

-- Define the expected number of apples after giving some away
def remaining_apples : ℕ := 39

-- Formulate the proof problem
theorem farmer_apples_after_giving : initial_apples - given_apples = remaining_apples := by
  sorry

end farmer_apples_after_giving_l53_53701


namespace total_nephews_l53_53440

noncomputable def Alden_past_nephews : ℕ := 50
noncomputable def Alden_current_nephews : ℕ := 2 * Alden_past_nephews
noncomputable def Vihaan_current_nephews : ℕ := Alden_current_nephews + 60

theorem total_nephews :
  Alden_current_nephews + Vihaan_current_nephews = 260 := 
by
  sorry

end total_nephews_l53_53440


namespace Yankees_to_Mets_ratio_l53_53659

theorem Yankees_to_Mets_ratio : 
  ∀ (Y M R : ℕ), M = 88 → (M + R + Y = 330) → (4 * R = 5 * M) → (Y : ℚ) / M = 3 / 2 :=
by
  intros Y M R hm htotal hratio
  sorry

end Yankees_to_Mets_ratio_l53_53659


namespace ral_current_age_l53_53522

-- Definitions according to the conditions
def ral_three_times_suri (ral suri : ℕ) : Prop := ral = 3 * suri
def suri_in_6_years (suri : ℕ) : Prop := suri + 6 = 25

-- The proof problem statement
theorem ral_current_age (ral suri : ℕ) (h1 : ral_three_times_suri ral suri) (h2 : suri_in_6_years suri) : ral = 57 :=
by sorry

end ral_current_age_l53_53522


namespace cost_of_one_pie_l53_53837

theorem cost_of_one_pie (x c2 c5 : ℕ) 
  (h1: 4 * x = c2 + 60)
  (h2: 5 * x = c5 + 60) 
  (h3: 6 * x = c2 + c5 + 60) : 
  x = 20 :=
by
  sorry

end cost_of_one_pie_l53_53837


namespace problem_solution_l53_53480

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -5) :
  (1 / a) + (1 / b) = -3 / 5 :=
by
  sorry

end problem_solution_l53_53480


namespace third_divisor_l53_53862

theorem third_divisor (x : ℕ) (h1 : x - 16 = 136) (h2 : ∃ y, y = x - 16) (h3 : 4 ∣ x) (h4 : 6 ∣ x) (h5 : 10 ∣ x) : 19 ∣ x := 
by
  sorry

end third_divisor_l53_53862


namespace total_weight_correct_weight_difference_correct_l53_53432

variables (baskets_of_apples baskets_of_pears : ℕ) (kg_per_basket_of_apples kg_per_basket_of_pears : ℕ)

def total_weight_apples_ppears (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_apples * kg_per_basket_of_apples) + (baskets_of_pears * kg_per_basket_of_pears)

def weight_difference_pears_apples (baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears : ℕ) : ℕ :=
  (baskets_of_pears * kg_per_basket_of_pears) - (baskets_of_apples * kg_per_basket_of_apples)

theorem total_weight_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  total_weight_apples_ppears baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 11300 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

theorem weight_difference_correct (h_apples: baskets_of_apples = 120) (h_pears: baskets_of_pears = 130) (h_kg_apples: kg_per_basket_of_apples = 40) (h_kg_pears: kg_per_basket_of_pears = 50) : 
  weight_difference_pears_apples baskets_of_apples baskets_of_pears kg_per_basket_of_apples kg_per_basket_of_pears = 1700 :=
by
  rw [h_apples, h_pears, h_kg_apples, h_kg_pears]
  sorry

end total_weight_correct_weight_difference_correct_l53_53432


namespace toys_in_row_l53_53708

theorem toys_in_row (n_left n_right : ℕ) (hy : 10 = n_left + 1) (hy' : 7 = n_right + 1) :
  n_left + n_right + 1 = 16 :=
by
  -- Fill in the proof here
  sorry

end toys_in_row_l53_53708


namespace persons_next_to_Boris_l53_53597

-- Definitions of names for convenience
inductive Person
| Arkady : Person
| Boris : Person
| Vera : Person
| Galya : Person
| Danya : Person
| Egor : Person
deriving DecidableEq, Repr

-- Define the circle arrangement
structure CircleArrangement where
  next : Person → Person
  left : Danya ≠ next Vera ∧ next Vera = Danya ∧ next Danya = Egor
  right : ∀ p : Person, p ≠ Danya → p ≠ Vera → next (next p) = p

namespace CircleArrangement
-- Given conditions
abbreviation cond_1 (ca : CircleArrangement) := ca.next Danya = Vera ∧ ca.next Vera = Danya
abbreviation cond_2 (ca : CircleArrangement) := ca.next Egor = Galya ∧ ca.next Galya = Egor
abbreviation cond_3 (ca : CircleArrangement) := (ca.next Egor = Danya ∧ ca.next Danya = Egor)
abbreviation cond_4 (ca : CircleArrangement) := ¬ (ca.next Arkady = Galya ∨ ca.next Galya = Arkady)

-- Goal to prove
theorem persons_next_to_Boris (ca : CircleArrangement) (h1: cond_1 ca) (h2: cond_2 ca)
    (h3: cond_3 ca) (h4: cond_4 ca) :
    (ca.next Boris = Arkady ∧ ca.next Arkady = Galya) ∨
    (ca.next Galya = Boris ∧ ca.next Boris = Arkady) := by
  sorry
end CircleArrangement

end persons_next_to_Boris_l53_53597


namespace last_four_digits_of_3_power_24000_l53_53489

theorem last_four_digits_of_3_power_24000 (h : 3^800 ≡ 1 [MOD 2000]) : 3^24000 ≡ 1 [MOD 2000] :=
  by sorry

end last_four_digits_of_3_power_24000_l53_53489


namespace find_soma_cubes_for_shape_l53_53427

def SomaCubes (n : ℕ) : Type := 
  if n = 1 
  then Fin 3 
  else if 2 ≤ n ∧ n ≤ 7 
       then Fin 4 
       else Fin 0

theorem find_soma_cubes_for_shape :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  SomaCubes a = Fin 3 ∧ SomaCubes b = Fin 4 ∧ SomaCubes c = Fin 4 ∧ 
  a + b + c = 11 ∧ ((a, b, c) = (1, 3, 5) ∨ (a, b, c) = (1, 3, 6)) := 
by
  sorry

end find_soma_cubes_for_shape_l53_53427


namespace part1_positive_root_part2_negative_solution_l53_53798

theorem part1_positive_root (x k : ℝ) (hx1 : x > 0)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k = 6 ∨ k = -8 := 
sorry

theorem part2_negative_solution (x k : ℝ) (hx2 : x < 0)
  (hx_ne1 : x ≠ 1) (hx_ne_neg1 : x ≠ -1)
  (h : 4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)) : 
  k < -1 ∧ k ≠ -8 := 
sorry

end part1_positive_root_part2_negative_solution_l53_53798


namespace final_value_l53_53972

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l53_53972


namespace expand_polynomial_l53_53458

variable {x y z : ℝ}

theorem expand_polynomial : (x + 10 * z + 5) * (2 * y + 15) = 2 * x * y + 20 * y * z + 15 * x + 10 * y + 150 * z + 75 :=
  sorry

end expand_polynomial_l53_53458


namespace hyperbola_vertices_distance_l53_53779

/--
For the hyperbola given by the equation
(x^2 / 121) - (y^2 / 49) = 1,
the distance between its vertices is 22.
-/
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ,
  (x^2 / 121) - (y^2 / 49) = 1 →
  ∃ a : ℝ, a = 11 ∧ 2 * a = 22 :=
by
  intros x y h
  use 11
  split
  · refl
  · norm_num

end hyperbola_vertices_distance_l53_53779


namespace total_cupcakes_baked_l53_53666

-- Conditions
def morning_cupcakes : ℕ := 20
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

-- Goal
theorem total_cupcakes_baked :
  (morning_cupcakes + afternoon_cupcakes) = 55 :=
by
  sorry

end total_cupcakes_baked_l53_53666


namespace math_problem_l53_53992

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l53_53992


namespace tan_alpha_value_l53_53139

variables (α β : ℝ)

theorem tan_alpha_value
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) :
  Real.tan α = 13 / 16 :=
sorry

end tan_alpha_value_l53_53139


namespace arith_seq_sum_geom_mean_proof_l53_53509

theorem arith_seq_sum_geom_mean_proof (a_1 : ℝ) (a_n : ℕ → ℝ)
(common_difference : ℝ) (s_n : ℕ → ℝ)
(h_sequence : ∀ n, a_n n = a_1 + (n - 1) * common_difference)
(h_sum : ∀ n, s_n n = n / 2 * (2 * a_1 + (n - 1) * common_difference))
(h_geom_mean : (s_n 2) ^ 2 = s_n 1 * s_n 4)
(h_common_diff : common_difference = -1) :
a_1 = -1 / 2 :=
sorry

end arith_seq_sum_geom_mean_proof_l53_53509


namespace find_range_a_l53_53782

noncomputable def sincos_inequality (x a θ : ℝ) : Prop :=
  (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1 / 8

theorem find_range_a :
  (∀ (x : ℝ) (θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → sincos_inequality x a θ)
  ↔ a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end find_range_a_l53_53782


namespace javier_initial_games_l53_53505

/--
Javier plays 2 baseball games a week. In each of his first some games, 
he averaged 2 hits. If he has 10 games left, he has to average 5 hits 
a game to bring his average for the season up to 3 hits a game. 
Prove that the number of games Javier initially played is 20.
-/
theorem javier_initial_games (x : ℕ) :
  (2 * x + 5 * 10) / (x + 10) = 3 → x = 20 :=
by
  sorry

end javier_initial_games_l53_53505


namespace distributing_balls_into_boxes_l53_53152

-- Define the parameters for the problem
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Statement of the problem in Lean
theorem distributing_balls_into_boxes :
  (finset.card (finset.univ.filter (λ (f : fin num_boxes → ℕ), fin.sum univ f = num_balls))) = 21 := 
sorry

end distributing_balls_into_boxes_l53_53152


namespace value_of_b_l53_53275

-- Defining the number sum in circles and overlap
def circle_sum := 21
def num_circles := 5
def total_sum := 69

-- Overlapping numbers
def overlap_1 := 2
def overlap_2 := 8
def overlap_3 := 9
variable (b d : ℕ)

-- Circle equation containing d
def circle_with_d := d + 5 + 9

-- Prove b = 10 given the conditions
theorem value_of_b (h₁ : num_circles * circle_sum = 105)
    (h₂ : 105 - (overlap_1 + overlap_2 + overlap_3 + b + d) = total_sum)
    (h₃ : circle_with_d d = 21) : b = 10 :=
by sorry

end value_of_b_l53_53275


namespace problem_p_3_l53_53669

theorem problem_p_3 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) (hn : n = (2^(2*p) - 1) / 3) : n ∣ 2^n - 2 := by
  sorry

end problem_p_3_l53_53669


namespace find_possible_sets_C_l53_53942

open Set

def A : Set ℕ := {3, 4}
def B : Set ℕ := {0, 1, 2, 3, 4}
def possible_C_sets : Set (Set ℕ) :=
  { {3, 4}, {3, 4, 0}, {3, 4, 1}, {3, 4, 2}, {3, 4, 0, 1},
    {3, 4, 0, 2}, {3, 4, 1, 2}, {0, 1, 2, 3, 4} }

theorem find_possible_sets_C :
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B} = possible_C_sets :=
by
  sorry

end find_possible_sets_C_l53_53942


namespace speed_ratio_l53_53245

-- Definitions of the conditions in the problem
variables (v_A v_B : ℝ) -- speeds of A and B

-- Condition 1: positions after 3 minutes are equidistant from O
def equidistant_3min : Prop := 3 * v_A = |(-300 + 3 * v_B)|

-- Condition 2: positions after 12 minutes are equidistant from O
def equidistant_12min : Prop := 12 * v_A = |(-300 + 12 * v_B)|

-- Statement to prove
theorem speed_ratio (h1 : equidistant_3min v_A v_B) (h2 : equidistant_12min v_A v_B) :
  v_A / v_B = 4 / 5 := sorry

end speed_ratio_l53_53245


namespace highest_score_of_batsman_l53_53006

theorem highest_score_of_batsman
  (avg : ℕ)
  (inn : ℕ)
  (diff_high_low : ℕ)
  (sum_high_low : ℕ)
  (avg_excl : ℕ)
  (inn_excl : ℕ)
  (h_l_avg : avg = 60)
  (h_l_inn : inn = 46)
  (h_l_diff : diff_high_low = 140)
  (h_l_sum : sum_high_low = 208)
  (h_l_avg_excl : avg_excl = 58)
  (h_l_inn_excl : inn_excl = 44) :
  ∃ H L : ℕ, H = 174 :=
by
  sorry

end highest_score_of_batsman_l53_53006


namespace graphs_relative_position_and_intersection_l53_53264

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 5

theorem graphs_relative_position_and_intersection :
  (1 > -1.5) ∧ ( ∃ y, f 0 = y ∧ g 0 = y ) ∧ f 0 = 5 :=
by
  -- sorry to skip the proof
  sorry

end graphs_relative_position_and_intersection_l53_53264


namespace no_integer_solutions_l53_53425

theorem no_integer_solutions (m n : ℤ) : ¬ (m ^ 3 + 6 * m ^ 2 + 5 * m = 27 * n ^ 3 + 9 * n ^ 2 + 9 * n + 1) :=
sorry

end no_integer_solutions_l53_53425


namespace average_side_length_of_squares_l53_53353

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end average_side_length_of_squares_l53_53353


namespace profit_percentage_B_l53_53749

theorem profit_percentage_B (cost_price_A : ℝ) (sell_price_C : ℝ) 
  (profit_A_percent : ℝ) (profit_B_percent : ℝ) 
  (cost_price_A_eq : cost_price_A = 148) 
  (sell_price_C_eq : sell_price_C = 222) 
  (profit_A_percent_eq : profit_A_percent = 0.2) :
  profit_B_percent = 0.25 := 
by
  have cost_price_B := cost_price_A * (1 + profit_A_percent)
  have profit_B := sell_price_C - cost_price_B
  have profit_B_percent := (profit_B / cost_price_B) * 100 
  sorry

end profit_percentage_B_l53_53749


namespace initial_value_subtract_perfect_square_l53_53902

theorem initial_value_subtract_perfect_square :
  ∃ n : ℕ, n^2 = 308 - 139 :=
by
  sorry

end initial_value_subtract_perfect_square_l53_53902


namespace original_price_l53_53445

noncomputable def original_selling_price (CP : ℝ) : ℝ := CP * 1.25
noncomputable def selling_price_at_loss (CP : ℝ) : ℝ := CP * 0.5

theorem original_price (CP : ℝ) (h : selling_price_at_loss CP = 320) : original_selling_price CP = 800 :=
by
  sorry

end original_price_l53_53445


namespace money_given_to_each_friend_l53_53675

-- Define the conditions
def initial_amount : ℝ := 20.10
def money_spent_on_sweets : ℝ := 1.05
def amount_left : ℝ := 17.05
def number_of_friends : ℝ := 2.0

-- Theorem statement
theorem money_given_to_each_friend :
  (initial_amount - amount_left - money_spent_on_sweets) / number_of_friends = 1.00 :=
by
  sorry

end money_given_to_each_friend_l53_53675


namespace game_remaining_sprite_color_l53_53813

theorem game_remaining_sprite_color (m n : ℕ) : 
  (∀ m n : ℕ, ∃ sprite : String, sprite = if n % 2 = 0 then "Red" else "Blue") :=
by sorry

end game_remaining_sprite_color_l53_53813


namespace find_f_0_plus_f_neg_1_l53_53433

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - x^2 else
if x < 0 then -(2^(-x) - (-x)^2) else 0

theorem find_f_0_plus_f_neg_1 : f 0 + f (-1) = -1 := by
  sorry

end find_f_0_plus_f_neg_1_l53_53433


namespace problem_statement_l53_53980

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l53_53980


namespace peanuts_in_box_l53_53888

   theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (total_peanuts : ℕ) 
     (h1 : initial_peanuts = 4) (h2 : added_peanuts = 6) : total_peanuts = initial_peanuts + added_peanuts :=
   by
     sorry

   example : peanuts_in_box 4 6 10 rfl rfl = rfl :=
   by
     sorry
   
end peanuts_in_box_l53_53888


namespace perfect_square_tens_digits_l53_53494

theorem perfect_square_tens_digits
  (a b : ℕ)
  (is_square_a : ∃ k : ℕ, a = k * k)
  (is_square_b : ∃ k : ℕ, b = k * k)
  (units_digit_a : a % 10 = 1)
  (tens_digit_a : ∃ x : ℕ, a / 10 % 10 = x)
  (units_digit_b : b % 10 = 6)
  (tens_digit_b : ∃ y : ℕ, b / 10 % 10 = y) :
  ∃ x y : ℕ, (a / 10 % 10 = x) ∧ (b / 10 % 10 = y) ∧ (x % 2 = 0) ∧ (y % 2 = 1) :=
sorry

end perfect_square_tens_digits_l53_53494


namespace average_side_length_of_squares_l53_53386

theorem average_side_length_of_squares :
  let a1 := 25
  let a2 := 64
  let a3 := 144
  let s1 := Real.sqrt a1
  let s2 := Real.sqrt a2
  let s3 := Real.sqrt a3
  (s1 + s2 + s3) / 3 = 25 / 3 := 
by
  sorry

end average_side_length_of_squares_l53_53386


namespace domain_of_function_is_all_real_l53_53410

def domain_function : Prop :=
  ∀ t : ℝ, (t - 3)^2 + (t + 3)^2 + 6 ≠ 0

theorem domain_of_function_is_all_real :
  domain_function :=
by
  intros t
  sorry

end domain_of_function_is_all_real_l53_53410


namespace difference_cubed_divisible_by_27_l53_53341

theorem difference_cubed_divisible_by_27 (a b : ℤ) :
    ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3) % 27 = 0 := 
by
  sorry

end difference_cubed_divisible_by_27_l53_53341


namespace problem_statement_l53_53979

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l53_53979


namespace total_cakes_served_l53_53747

-- Defining the values for cakes served during lunch and dinner
def lunch_cakes : ℤ := 6
def dinner_cakes : ℤ := 9

-- Stating the theorem that the total number of cakes served today is 15
theorem total_cakes_served : lunch_cakes + dinner_cakes = 15 :=
by
  sorry

end total_cakes_served_l53_53747


namespace train_speed_l53_53574

noncomputable def distance : ℝ := 45  -- 45 km
noncomputable def time_minutes : ℝ := 30  -- 30 minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert minutes to hours

theorem train_speed (d : ℝ) (t_m : ℝ) : d = 45 → t_m = 30 → d / (t_m / 60) = 90 :=
by
  intros h₁ h₂
  sorry

end train_speed_l53_53574


namespace balls_in_boxes_ways_l53_53155

theorem balls_in_boxes_ways : 
  (∃ (ways : ℕ), ways = 21 ∧
    ∀ {k : ℕ}, (5 + k - 1).choose (5) = ways) := 
begin
  sorry,
end

end balls_in_boxes_ways_l53_53155


namespace total_cost_of_long_distance_bill_l53_53931

theorem total_cost_of_long_distance_bill
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (minutes_billed : ℝ := 28.08) :
  monthly_fee + cost_per_minute * minutes_billed = 12.02 := by
  sorry

end total_cost_of_long_distance_bill_l53_53931


namespace parabola_symmetric_points_l53_53146

theorem parabola_symmetric_points (a : ℝ) (x1 y1 x2 y2 m : ℝ) 
  (h_parabola : ∀ x, y = a * x^2)
  (h_a_pos : a > 0)
  (h_focus_directrix : 1 / (2 * a) = 1 / 4)
  (h_symmetric : y1 = a * x1^2 ∧ y2 = a * x2^2 ∧ ∃ m, y1 = m + (x1 - m))
  (h_product : x1 * x2 = -1 / 2) :
  m = 3 / 2 := 
sorry

end parabola_symmetric_points_l53_53146


namespace average_reading_time_l53_53128

theorem average_reading_time (t_Emery t_Serena : ℕ) (h1 : t_Emery = 20) (h2 : t_Serena = 5 * t_Emery) : 
  (t_Emery + t_Serena) / 2 = 60 := 
by
  sorry

end average_reading_time_l53_53128


namespace who_next_to_boris_l53_53620

-- Define the individuals
inductive Person : Type
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq, Inhabited

open Person

-- Define the standing arrangement in a circle
structure CircleArrangement :=
(stands_next_to : Person → Person → Bool)
(opposite : Person → Person → Bool)

variables (arr : CircleArrangement)

-- Given conditions
axiom danya_next_to_vera : arr.stands_next_to Danya Vera ∧ ¬ arr.stands_next_to Vera Danya
axiom galya_opposite_egor : arr.opposite Galya Egor
axiom egor_next_to_danya : arr.stands_next_to Egor Danya ∧ arr.stands_next_to Danya Egor
axiom arkady_not_next_to_galya : ¬ arr.stands_next_to Arkady Galya ∧ ¬ arr.stands_next_to Galya Arkady

-- Conclude who stands next to Boris
theorem who_next_to_boris : (arr.stands_next_to Boris Arkady ∧ arr.stands_next_to Arkady Boris) ∨
                            (arr.stands_next_to Boris Galya ∧ arr.stands_next_to Galya Boris) :=
sorry

end who_next_to_boris_l53_53620


namespace algebraic_expression_identity_l53_53178

noncomputable theory

/-- Proof of the given problem condition. -/
theorem algebraic_expression_identity (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 24 := 
sorry

end algebraic_expression_identity_l53_53178


namespace sum_odd_integers_correct_l53_53413

def sum_odd_integers_from_13_to_41 : ℕ := 
  let a := 13
  let l := 41
  let n := 15
  n * (a + l) / 2

theorem sum_odd_integers_correct : sum_odd_integers_from_13_to_41 = 405 :=
  by sorry

end sum_odd_integers_correct_l53_53413


namespace max_x_on_circle_l53_53670

theorem max_x_on_circle : 
  ∀ x y : ℝ,
  (x - 10)^2 + (y - 30)^2 = 100 → x ≤ 20 :=
by
  intros x y h
  sorry

end max_x_on_circle_l53_53670


namespace minimum_employees_for_identical_training_l53_53909

def languages : Finset String := {"English", "French", "Spanish", "German"}

noncomputable def choose_pairings_count (n k : ℕ) : ℕ :=
Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem minimum_employees_for_identical_training 
  (num_languages : ℕ := 4) 
  (employees_per_pairing : ℕ := 4)
  (pairings : ℕ := choose_pairings_count num_languages 2) 
  (total_employees : ℕ := employees_per_pairing * pairings)
  (minimum_employees : ℕ := total_employees + 1):
  minimum_employees = 25 :=
by
  -- We skip the proof details as per the instructions
  sorry

end minimum_employees_for_identical_training_l53_53909


namespace cumulative_distribution_X_maximized_expected_score_l53_53736

noncomputable def distribution_X (p_A : ℝ) (p_B : ℝ) : (ℝ × ℝ × ℝ) :=
(1 - p_A, p_A * (1 - p_B), p_A * p_B)

def expected_score (p_A : ℝ) (p_B : ℝ) (s_A : ℝ) (s_B : ℝ) : ℝ :=
0 * (1 - p_A) + s_A * (p_A * (1 - p_B)) + (s_A + s_B) * (p_A * p_B)

theorem cumulative_distribution_X :
  distribution_X 0.8 0.6 = (0.2, 0.32, 0.48) :=
sorry

theorem maximized_expected_score :
  expected_score 0.8 0.6 20 80 < expected_score 0.6 0.8 80 20 :=
sorry

end cumulative_distribution_X_maximized_expected_score_l53_53736


namespace num_integers_satisfy_l53_53653

theorem num_integers_satisfy : 
  ∃ n : ℕ, (n = 7 ∧ ∀ k : ℤ, (k > -5 ∧ k < 3) → (k = -4 ∨ k = -3 ∨ k = -2 ∨ k = -1 ∨ k = 0 ∨ k = 1 ∨ k = 2)) := 
sorry

end num_integers_satisfy_l53_53653


namespace points_per_touchdown_l53_53698

theorem points_per_touchdown (total_points touchdowns : ℕ) (h1 : total_points = 21) (h2 : touchdowns = 3) :
  total_points / touchdowns = 7 :=
by
  sorry

end points_per_touchdown_l53_53698


namespace empty_solution_set_of_inequalities_l53_53540

theorem empty_solution_set_of_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ ((2 * x < 5 - 3 * x) ∧ ((x - 1) / 2 > a))) ↔ (0 ≤ a) := 
by
  sorry

end empty_solution_set_of_inequalities_l53_53540


namespace arithmetic_sequence_common_difference_l53_53945

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℕ)
  (d : ℚ)
  (h_arith_seq : ∀ (n m : ℕ), (n > 0) → (m > 0) → (a n) / n - (a m) / m = (n - m) * d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1 / 9 ∧ a 12 = 20 :=
by 
  sorry

end arithmetic_sequence_common_difference_l53_53945


namespace rbcmul_div7_div89_l53_53402

theorem rbcmul_div7_div89 {r b c : ℕ} (h : (523000 + 100 * r + 10 * b + c) % 7 = 0 ∧ (523000 + 100 * r + 10 * b + c) % 89 = 0) :
  r * b * c = 36 :=
by
  sorry

end rbcmul_div7_div89_l53_53402


namespace arrange_decimals_in_order_l53_53758

theorem arrange_decimals_in_order 
  (a b c d : ℚ) 
  (h₀ : a = 6 / 10) 
  (h₁ : b = 676 / 1000) 
  (h₂ : c = 677 / 1000) 
  (h₃ : d = 67 / 100) : 
  a < d ∧ d < b ∧ b < c := 
by
  sorry

end arrange_decimals_in_order_l53_53758


namespace regular_polygon_perimeter_l53_53106

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53106


namespace simplify_and_evaluate_l53_53002

theorem simplify_and_evaluate (a : ℚ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5 / 3 :=
by
  sorry

end simplify_and_evaluate_l53_53002


namespace algebraic_expression_value_l53_53170

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l53_53170


namespace distance_between_hyperbola_vertices_l53_53777

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l53_53777


namespace sum_of_fractions_l53_53960

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l53_53960


namespace find_second_expression_l53_53211

theorem find_second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 84) (h₂ : a = 32) : x = 88 :=
  sorry

end find_second_expression_l53_53211


namespace expression_value_l53_53958

theorem expression_value (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
begin
  sorry
end

end expression_value_l53_53958


namespace tessellation_solutions_l53_53547

theorem tessellation_solutions (m n : ℕ) (h : 60 * m + 90 * n = 360) : m = 3 ∧ n = 2 :=
by
  sorry

end tessellation_solutions_l53_53547


namespace base_of_isosceles_triangle_l53_53241

theorem base_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : 3 * a = 45) 
  (h₂ : 2 * b + c = 40) 
  (h₃ : b = a ∨ b = a) : c = 10 := 
sorry

end base_of_isosceles_triangle_l53_53241


namespace freight_train_distance_l53_53899

variable (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) 

def total_distance_traveled (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) : ℕ :=
  let traveled_distance := (time_minutes / travel_rate) 
  traveled_distance + initial_distance

theorem freight_train_distance :
  total_distance_traveled 2 5 90 = 50 :=
by
  sorry

end freight_train_distance_l53_53899


namespace probability_blue_face_up_l53_53409

def cube_probability_blue : ℚ := 
  let total_faces := 6
  let blue_faces := 4
  blue_faces / total_faces

theorem probability_blue_face_up :
  cube_probability_blue = 2 / 3 :=
by
  sorry

end probability_blue_face_up_l53_53409


namespace monotonically_decreasing_interval_range_of_f_l53_53703

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (abs (x - 1))

theorem monotonically_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ > f x₂ := by sorry

theorem range_of_f :
  Set.range f = {y : ℝ | 0 < y ∧ y ≤ 1 } := by sorry

end monotonically_decreasing_interval_range_of_f_l53_53703


namespace sin_A_value_l53_53812

variables {A B C a b c : ℝ}
variables {sin cos : ℝ → ℝ}

-- Conditions
axiom triangle_sides : ∀ (A B C: ℝ), ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0
axiom sin_cos_conditions : 3 * b * sin A = c * cos A + a * cos C

-- Proof statement
theorem sin_A_value (h : 3 * b * sin A = c * cos A + a * cos C) : sin A = 1 / 3 :=
by 
  sorry

end sin_A_value_l53_53812


namespace regular_polygon_perimeter_l53_53045

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53045


namespace angle_bisector_triangle_inequality_l53_53524

theorem angle_bisector_triangle_inequality (AB AC D BD CD x : ℝ) (hAB : AB = 10) (hCD : CD = 3) (h_angle_bisector : BD = 30 / x)
  (h_triangle_inequality_1 : x + (BD + CD) > AB)
  (h_triangle_inequality_2 : AB + (BD + CD) > x)
  (h_triangle_inequality_3 : AB + x > BD + CD) :
  (3 < x) ∧ (x < 15) ∧ (3 + 15 = (18 : ℝ)) :=
by
  sorry

end angle_bisector_triangle_inequality_l53_53524


namespace area_of_set_R_is_1006point5_l53_53192

-- Define the set of points R as described in the problem
def isPointInSetR (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + y ≤ 2013 ∧ ⌈x⌉ * ⌊y⌋ = ⌊x⌋ * ⌈y⌉

noncomputable def computeAreaOfSetR : ℝ :=
  1006.5

theorem area_of_set_R_is_1006point5 :
  (∃ x y : ℝ, isPointInSetR x y) → computeAreaOfSetR = 1006.5 := by
  sorry

end area_of_set_R_is_1006point5_l53_53192


namespace regular_polygon_perimeter_l53_53105

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l53_53105


namespace total_amount_paid_l53_53865

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end total_amount_paid_l53_53865


namespace inequality_proof_l53_53830

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
by
  sorry

end inequality_proof_l53_53830


namespace line_intersects_curve_l53_53706

theorem line_intersects_curve (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ax₁ + 16 = x₁^3 ∧ ax₂ + 16 = x₂^3) →
  a = 12 :=
by
  sorry

end line_intersects_curve_l53_53706


namespace who_is_next_to_boris_l53_53594

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end who_is_next_to_boris_l53_53594


namespace area_diff_circle_square_l53_53437

theorem area_diff_circle_square (s r : ℝ) (A_square A_circle : ℝ) (d : ℝ) (pi : ℝ) 
  (h1 : d = 8) -- diagonal of the square
  (h2 : d = 2 * r) -- diameter of the circle is 8, so radius is 4
  (h3 : s^2 + s^2 = d^2) -- Pythagorean Theorem for the square
  (h4 : A_square = s^2) -- area of the square
  (h5 : A_circle = pi * r^2) -- area of the circle
  (h6 : pi = 3.14159) -- approximation for π
  : abs (A_circle - A_square) - 18.3 < 0.1 := sorry

end area_diff_circle_square_l53_53437


namespace number_of_trapezoids_l53_53819

def reg_pent_midpoints := set (fin 5 → ℝ × ℝ)  -- Representation of the midpoints of a regular pentagon

theorem number_of_trapezoids (P : reg_pent_midpoints) : ∃ t : set (set (ℝ × ℝ)), t.card = 15 ∧ (∀ x ∈ t, is_trapezoid x) := 
sorry

end number_of_trapezoids_l53_53819


namespace one_python_can_eat_per_week_l53_53561

-- Definitions based on the given conditions
def burmese_pythons := 5
def alligators_eaten := 15
def weeks := 3

-- Theorem statement to prove the number of alligators one python can eat per week
theorem one_python_can_eat_per_week : (alligators_eaten / burmese_pythons) / weeks = 1 := 
by 
-- sorry is used to skip the actual proof
sorry

end one_python_can_eat_per_week_l53_53561


namespace perimeter_of_regular_polygon_l53_53065

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53065


namespace sequence_geometric_l53_53134

theorem sequence_geometric {a_n : ℕ → ℕ} (S : ℕ → ℕ) (a1 a2 a3 : ℕ) 
(hS : ∀ n, S n = 2 * a_n n - a_n 1) 
(h_arith : 2 * (a_n 2 + 1) = a_n 3 + a_n 1) : 
  ∀ n, a_n n = 2 ^ n :=
sorry

end sequence_geometric_l53_53134


namespace perimeter_of_regular_polygon_l53_53063

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53063


namespace correct_proposition_l53_53298

def curve_is_ellipse (k : ℝ) : Prop :=
  9 < k ∧ k < 25

def curve_is_hyperbola_on_x_axis (k : ℝ) : Prop :=
  k < 9

theorem correct_proposition (k : ℝ) :
  (curve_is_ellipse k ∨ ¬ curve_is_ellipse k) ∧ 
  (curve_is_hyperbola_on_x_axis k ∨ ¬ curve_is_hyperbola_on_x_axis k) →
  (9 < k ∧ k < 25 → curve_is_ellipse k) ∧ 
  (curve_is_ellipse k ↔ (9 < k ∧ k < 25)) ∧ 
  (curve_is_hyperbola_on_x_axis k ↔ k < 9) → 
  (curve_is_ellipse k ∧ curve_is_hyperbola_on_x_axis k) :=
by
  sorry

end correct_proposition_l53_53298


namespace range_a_l53_53652

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 2))

def domain_A : Set ℝ := { x | x < -1 ∨ x > 2 }

def solution_set_B (a : ℝ) : Set ℝ := { x | x < a ∨ x > a + 1 }

theorem range_a (a : ℝ)
  (h : (domain_A ∪ solution_set_B a) = solution_set_B a) :
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l53_53652


namespace min_val_x_2y_l53_53674

noncomputable def min_x_2y (x y : ℝ) : ℝ :=
  x + 2 * y

theorem min_val_x_2y : 
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / (x + 2) + 1 / (y + 2) = 1 / 3) → 
  min_x_2y x y ≥ 3 + 6 * Real.sqrt 2 :=
by
  intros x y x_pos y_pos eqn
  sorry

end min_val_x_2y_l53_53674


namespace total_people_at_zoo_l53_53562

theorem total_people_at_zoo (A K : ℕ) (ticket_price_adult : ℕ := 28) (ticket_price_kid : ℕ := 12) (total_sales : ℕ := 3864) (number_of_kids : ℕ := 203) :
  (ticket_price_adult * A + ticket_price_kid * number_of_kids = total_sales) → 
  (A + number_of_kids = 254) :=
by
  sorry

end total_people_at_zoo_l53_53562


namespace misread_weight_l53_53852

theorem misread_weight (avg_initial : ℝ) (avg_correct : ℝ) (n : ℕ) (actual_weight : ℝ) (x : ℝ) : 
  avg_initial = 58.4 → avg_correct = 58.7 → n = 20 → actual_weight = 62 → 
  (n * avg_correct - n * avg_initial = actual_weight - x) → x = 56 :=
by
  intros
  sorry

end misread_weight_l53_53852


namespace option_d_l53_53121

variable {R : Type*} [LinearOrderedField R]

theorem option_d (a b c d : R) (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by 
  sorry

end option_d_l53_53121


namespace difference_largest_smallest_l53_53712

noncomputable def ratio_2_3_5 := 2 / 3
noncomputable def ratio_3_5 := 3 / 5
noncomputable def int_sum := 90

theorem difference_largest_smallest :
  ∃ (a b c : ℝ), 
    a + b + c = int_sum ∧
    b / a = ratio_2_3_5 ∧
    c / a = 5 / 2 ∧
    b / a = 3 / 2 ∧
    c - a = 12.846 := 
by
  sorry

end difference_largest_smallest_l53_53712


namespace sin_cos_alpha_eq_fifth_l53_53936

variable {α : ℝ}
variable (h : Real.sin α = 2 * Real.cos α)

theorem sin_cos_alpha_eq_fifth : Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end sin_cos_alpha_eq_fifth_l53_53936


namespace polygon_perimeter_l53_53093

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l53_53093


namespace BDD1H_is_Spatial_in_Cube_l53_53818

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Cube :=
(A B C D A1 B1 C1 D1 : Point3D)
(midpoint_B1C1 : Point3D)
(middle_B1C1 : midpoint_B1C1 = ⟨(B1.x + C1.x) / 2, (B1.y + C1.y) / 2, (B1.z + C1.z) / 2⟩)

def is_not_planar (a b c d : Point3D) : Prop :=
¬ ∃ α β γ δ : ℝ, α * a.x + β * a.y + γ * a.z + δ = 0 ∧ 
                α * b.x + β * b.y + γ * b.z + δ = 0 ∧ 
                α * c.x + β * c.y + γ * c.z + δ = 0 ∧ 
                α * d.x + β * d.y + γ * d.z + δ = 0

def BDD1H_is_spatial (cube : Cube) : Prop :=
is_not_planar cube.B cube.D cube.D1 cube.midpoint_B1C1

theorem BDD1H_is_Spatial_in_Cube (cube : Cube) : BDD1H_is_spatial cube :=
sorry

end BDD1H_is_Spatial_in_Cube_l53_53818


namespace regular_polygon_perimeter_l53_53049

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l53_53049


namespace range_of_x_l53_53647

noncomputable def f : ℝ → ℝ := sorry  -- f is an even function and decreasing on [0, +∞)

theorem range_of_x (x : ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y) 
  (h_condition : f (Real.log x) > f 1) : 
  1 / 10 < x ∧ x < 10 := 
sorry

end range_of_x_l53_53647


namespace S_40_eq_150_l53_53796

variable {R : Type*} [Field R]

-- Define the sum function for geometric sequences.
noncomputable def geom_sum (a q : R) (n : ℕ) : R :=
  a * (1 - q^n) / (1 - q)

-- Given conditions from the problem.
axiom S_10_eq : ∀ {a q : R}, geom_sum a q 10 = 10
axiom S_30_eq : ∀ {a q : R}, geom_sum a q 30 = 70

-- The main theorem stating S40 = 150 under the given conditions.
theorem S_40_eq_150 {a q : R} (h10 : geom_sum a q 10 = 10) (h30 : geom_sum a q 30 = 70) :
  geom_sum a q 40 = 150 :=
sorry

end S_40_eq_150_l53_53796


namespace tan_lt_neg_one_implies_range_l53_53627

theorem tan_lt_neg_one_implies_range {x : ℝ} (h1 : 0 < x) (h2 : x < π) (h3 : Real.tan x < -1) :
  (π / 2 < x) ∧ (x < 3 * π / 4) :=
sorry

end tan_lt_neg_one_implies_range_l53_53627


namespace algebraic_expression_value_l53_53169

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l53_53169


namespace rectangle_circumference_15pi_l53_53570

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ := 
  Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def circumference_of_circle (d : ℝ) : ℝ := 
  Real.pi * d
  
theorem rectangle_circumference_15pi :
  let a := 9
  let b := 12
  let diagonal := rectangle_diagonal a b
  circumference_of_circle diagonal = 15 * Real.pi :=
by 
  sorry

end rectangle_circumference_15pi_l53_53570


namespace tan_ratio_l53_53793

theorem tan_ratio (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.tan x / Real.tan (2 * x) = 4 / 9 :=
sorry

end tan_ratio_l53_53793


namespace words_per_page_l53_53558

theorem words_per_page 
    (p : ℕ) 
    (h1 : 150 > 0) 
    (h2 : 150 * p ≡ 200 [MOD 221]) :
    p = 118 := 
by sorry

end words_per_page_l53_53558


namespace temperature_difference_l53_53795

def h : ℤ := 10
def l : ℤ := -5
def d : ℤ := 15

theorem temperature_difference : h - l = d :=
by
  rw [h, l, d]
  sorry

end temperature_difference_l53_53795


namespace trees_variance_l53_53683

theorem trees_variance :
  let groups := [3, 4, 3]
  let trees := [5, 6, 7]
  let n := 10
  let mean := (5 * 3 + 6 * 4 + 7 * 3) / n
  let variance := (3 * (5 - mean)^2 + 4 * (6 - mean)^2 + 3 * (7 - mean)^2) / n
  variance = 0.6 := 
by
  sorry

end trees_variance_l53_53683


namespace probability_of_grade_A_l53_53183

open ProbabilityTheory

noncomputable def prob_grade_A (X : MeasureTheory.MeasureSpace Real) := 
  sorry

theorem probability_of_grade_A :
  prob_grade_A (MeasureTheory.gaussian 80 (real.sqrt 25)) = 0.15865 :=
by
  sorry

end probability_of_grade_A_l53_53183


namespace dave_trips_l53_53766

theorem dave_trips :
  let trays_at_a_time := 12
  let trays_table_1 := 26
  let trays_table_2 := 49
  let trays_table_3 := 65
  let trays_table_4 := 38
  let total_trays := trays_table_1 + trays_table_2 + trays_table_3 + trays_table_4
  let trips := (total_trays + trays_at_a_time - 1) / trays_at_a_time
  trips = 15 := by
    repeat { sorry }

end dave_trips_l53_53766


namespace exponential_function_range_l53_53799

noncomputable def exponential_function (a x : ℝ) : ℝ := a^x

theorem exponential_function_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : exponential_function a (-2) < exponential_function a (-3)) : 
  0 < a ∧ a < 1 :=
by
  sorry

end exponential_function_range_l53_53799


namespace fuel_oil_used_l53_53839

theorem fuel_oil_used (V_initial : ℕ) (V_jan : ℕ) (V_may : ℕ) : 
  (V_initial - V_jan) + (V_initial - V_may) = 4582 :=
by
  let V_initial := 3000
  let V_jan := 180
  let V_may := 1238
  sorry

end fuel_oil_used_l53_53839


namespace gcd_m_n_l53_53335

def m : ℕ := 3333333
def n : ℕ := 66666666

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end gcd_m_n_l53_53335


namespace problem_statement_l53_53976

variable {p q r : ℝ}

theorem problem_statement (h1 : p + q + r = 5)
                          (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
    r / (p + q) + p / (q + r) + q / (p + r) = 42 :=
  sorry

end problem_statement_l53_53976


namespace problem_statement_l53_53985

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end problem_statement_l53_53985


namespace average_speed_of_journey_is_24_l53_53419

noncomputable def average_speed (D : ℝ) (speed_to_office speed_to_home : ℝ) : ℝ :=
  let time_to_office := D / speed_to_office
  let time_to_home := D / speed_to_home
  let total_distance := 2 * D
  let total_time := time_to_office + time_to_home
  total_distance / total_time

theorem average_speed_of_journey_is_24 (D : ℝ) : average_speed D 20 30 = 24 := by
  -- nonconstructive proof to fulfill theorem definition
  sorry

end average_speed_of_journey_is_24_l53_53419


namespace simplify_expression_l53_53342

theorem simplify_expression (x y : ℝ) : 2 - (3 - (2 + (5 - (3 * y - x)))) = 6 - 3 * y + x :=
by
  sorry

end simplify_expression_l53_53342


namespace f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l53_53949

open Real

noncomputable def f : ℝ → ℝ :=
sorry

axiom func_prop : ∀ x y : ℝ, f (x + y) = f x + f y - 1
axiom pos_x_gt_1 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_1 : f 1 = 2

-- Prove that f(0) = 1
theorem f_0_eq_1 : f 0 = 1 :=
sorry

-- Prove that f(-1) ≠ 1 (and direct derivation showing f(-1) = 0)
theorem f_neg_1_ne_1 : f (-1) ≠ 1 ∧ f (-1) = 0 :=
sorry

-- Prove that f(x) is increasing
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ > f x₁ :=
sorry

-- Prove minimum value of f on [-3, 3] is -2
theorem min_f_neg3_3 : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -2 :=
sorry

end f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l53_53949


namespace exists_nat_pair_l53_53129

theorem exists_nat_pair 
  (k : ℕ) : 
  let a := 2 * k
  let b := 2 * k * k + 2 * k + 1
  (b - 1) % (a + 1) = 0 ∧ (a * a + a + 2) % b = 0 := by
  sorry

end exists_nat_pair_l53_53129


namespace largest_integer_k_l53_53254

def sequence_term (n : ℕ) : ℚ :=
  (1 / n - 1 / (n + 2))

def sum_sequence (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), sequence_term (i + 1)

theorem largest_integer_k (k : ℕ):
  (sum_sequence k < 1.499) ↔ k = 1998 :=
sorry

end largest_integer_k_l53_53254


namespace who_is_next_to_Boris_l53_53613

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l53_53613


namespace compute_expression_l53_53167

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end compute_expression_l53_53167


namespace average_of_side_lengths_of_squares_l53_53391

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l53_53391


namespace cost_to_open_store_l53_53191

-- Define the conditions as constants
def revenue_per_month : ℕ := 4000
def expenses_per_month : ℕ := 1500
def months_to_payback : ℕ := 10

-- Theorem stating the cost to open the store
theorem cost_to_open_store : (revenue_per_month - expenses_per_month) * months_to_payback = 25000 :=
by
  sorry

end cost_to_open_store_l53_53191


namespace part1_part2_part3_l53_53497

-- Part 1
theorem part1 (x : ℝ) :
  (2 * x - 5 > 3 * x - 8 ∧ -4 * x + 3 < x - 4) ↔ x = 2 :=
sorry

-- Part 2
theorem part2 (x : ℤ) :
  (x - 1 / 4 < 1 ∧ 4 + 2 * x > -7 * x + 5) ↔ x = 1 :=
sorry

-- Part 3
theorem part3 (m : ℝ) :
  (∀ x, m < x ∧ x <= m + 2 → (x = 3 ∨ x = 2)) ↔ 1 ≤ m ∧ m < 2 :=
sorry

end part1_part2_part3_l53_53497


namespace total_cost_of_coat_l53_53571

def original_price : ℝ := 150
def sale_discount : ℝ := 0.25
def additional_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_cost_of_coat :
  let sale_price := original_price * (1 - sale_discount)
  let price_after_discount := sale_price - additional_discount
  let final_price := price_after_discount * (1 + sales_tax)
  final_price = 112.75 :=
by
  -- sorry for the actual proof
  sorry

end total_cost_of_coat_l53_53571


namespace probability_of_team_A_winning_is_11_over_16_l53_53348

noncomputable def prob_A_wins_series : ℚ :=
  let total_games := 5
  let wins_needed_A := 2
  let wins_needed_B := 3
  -- Assuming equal probability for each game being won by either team
  let equal_chance_of_winning := 0.5
  -- Calculation would follow similar steps omitted for brevity
  -- Assuming the problem statement proven by external logical steps
  11 / 16

theorem probability_of_team_A_winning_is_11_over_16 :
  prob_A_wins_series = 11 / 16 := 
  sorry

end probability_of_team_A_winning_is_11_over_16_l53_53348


namespace algebraic_expression_value_l53_53171

open Real

theorem algebraic_expression_value (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
sorry

end algebraic_expression_value_l53_53171


namespace V3_is_correct_l53_53548

-- Definitions of the polynomial and Horner's method applied at x = -4
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def V_3_value : ℤ := 
  let v0 := -4
  let v1 := v0 * 3 + 5
  let v2 := v0 * v1 + 6
  v0 * v2 + 79

theorem V3_is_correct : V_3_value = -57 := 
  by sorry

end V3_is_correct_l53_53548


namespace no_two_digit_number_divisible_l53_53272

theorem no_two_digit_number_divisible (a b : ℕ) (distinct : a ≠ b)
  (h₁ : 1 ≤ a ∧ a ≤ 9) (h₂ : 1 ≤ b ∧ b ≤ 9)
  : ¬ ∃ k : ℕ, (1 < k ∧ k ≤ 9) ∧ (10 * a + b = k * (10 * b + a)) :=
by
  sorry

end no_two_digit_number_divisible_l53_53272


namespace total_amount_paid_l53_53866

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end total_amount_paid_l53_53866


namespace transformed_equation_correct_l53_53020
-- Import the necessary library

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation functions for the transformations
def translate_right (x : ℝ) : ℝ := x - 1
def translate_down (y : ℝ) : ℝ := y - 3

-- Define the transformed parabola equation
def transformed_parabola (x : ℝ) : ℝ := -2 * (translate_right x)^2 |> translate_down

-- The theorem stating the transformed equation
theorem transformed_equation_correct :
  ∀ x, transformed_parabola x = -2 * (x - 1)^2 - 3 :=
by { sorry }

end transformed_equation_correct_l53_53020


namespace XiaoMing_strategy_l53_53735

noncomputable def prob_A_correct : ℝ := 0.8
noncomputable def prob_B_correct : ℝ := 0.6

def points_A_correct : ℝ := 20
def points_B_correct : ℝ := 80

def prob_XA_0 : ℝ := 1 - prob_A_correct
def prob_XA_20 : ℝ := prob_A_correct * (1 - prob_B_correct)
def prob_XA_100 : ℝ := prob_A_correct * prob_B_correct

def expected_XA : ℝ := 0 * prob_XA_0 + points_A_correct * prob_XA_20 + (points_A_correct + points_B_correct) * prob_XA_100

def prob_YB_0 : ℝ := 1 - prob_B_correct
def prob_YB_80 : ℝ := prob_B_correct * (1 - prob_A_correct)
def prob_YB_100 : ℝ := prob_B_correct * prob_A_correct

def expected_YB : ℝ := 0 * prob_YB_0 + points_B_correct * prob_YB_80 + (points_A_correct + points_B_correct) * prob_YB_100

def distribution_A_is_correct : Prop :=
  prob_XA_0 = 0.2 ∧ prob_XA_20 = 0.32 ∧ prob_XA_100 = 0.48

def choose_B_first : Prop :=
  expected_YB > expected_XA

theorem XiaoMing_strategy :
  distribution_A_is_correct ∧ choose_B_first :=
by
  sorry

end XiaoMing_strategy_l53_53735


namespace arithmetic_sequence_a7_l53_53186

/--
In an arithmetic sequence {a_n}, it is known that a_1 = 2 and a_3 + a_5 = 10.
Then, we need to prove that a_7 = 8.
-/
theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 3 + a 5 = 10) 
  (h3 : ∀ n, a n = 2 + (n - 1) * d) : 
  a 7 = 8 := by
  sorry

end arithmetic_sequence_a7_l53_53186


namespace max_a_condition_range_a_condition_l53_53800

-- Definitions of the functions f and g
def f (x a : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Problem (I)
theorem max_a_condition (a : ℝ) :
  (∀ x, g x ≤ 5 → f x a ≤ 6) → a ≤ 1 :=
sorry

-- Problem (II)
theorem range_a_condition (a : ℝ) :
  (∀ x, f x a + g x ≥ 3) → a ≥ 2 :=
sorry

end max_a_condition_range_a_condition_l53_53800


namespace distributing_balls_into_boxes_l53_53151

-- Define the parameters for the problem
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Statement of the problem in Lean
theorem distributing_balls_into_boxes :
  (finset.card (finset.univ.filter (λ (f : fin num_boxes → ℕ), fin.sum univ f = num_balls))) = 21 := 
sorry

end distributing_balls_into_boxes_l53_53151


namespace middle_number_divisible_by_4_l53_53013

noncomputable def three_consecutive_cubes_is_cube (x y : ℕ) : Prop :=
  (x-1)^3 + x^3 + (x+1)^3 = y^3

theorem middle_number_divisible_by_4 (x y : ℕ) (h : three_consecutive_cubes_is_cube x y) : 4 ∣ x :=
sorry

end middle_number_divisible_by_4_l53_53013


namespace perimeter_of_regular_polygon_l53_53064

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : exterior_angle = 360 / n) : 
  4 * side_length = 28 := 
by 
  sorry

end perimeter_of_regular_polygon_l53_53064


namespace stu_books_count_l53_53454

noncomputable def elmo_books : ℕ := 24
noncomputable def laura_books : ℕ := elmo_books / 3
noncomputable def stu_books : ℕ := laura_books / 2

theorem stu_books_count :
  stu_books = 4 :=
by
  sorry

end stu_books_count_l53_53454


namespace intersection_complement_l53_53484

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}

theorem intersection_complement :
  A ∩ (U \ B) = {4, 5} := by
  sorry

end intersection_complement_l53_53484


namespace sunset_time_range_l53_53754

theorem sunset_time_range (h : ℝ) :
  ¬(h ≥ 7) ∧ ¬(h ≤ 8) ∧ ¬(h ≤ 6) ↔ h ∈ Set.Ioi 8 :=
by
  sorry

end sunset_time_range_l53_53754
