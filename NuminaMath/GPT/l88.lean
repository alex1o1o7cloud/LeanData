import Mathlib

namespace actual_diameter_of_tissue_l88_88807

theorem actual_diameter_of_tissue (magnification_factor : ℝ) (magnified_diameter : ℝ) (image_magnified : magnification_factor = 1000 ∧ magnified_diameter = 2) : (1 / magnification_factor) * magnified_diameter = 0.002 :=
by
  sorry

end actual_diameter_of_tissue_l88_88807


namespace ninth_graders_only_math_l88_88567

theorem ninth_graders_only_math 
  (total_students : ℕ)
  (math_students : ℕ)
  (foreign_language_students : ℕ)
  (science_only_students : ℕ)
  (math_and_foreign_language_no_science : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 85)
  (h3 : foreign_language_students = 75)
  (h4 : science_only_students = 20)
  (h5 : math_and_foreign_language_no_science = 40) :
  math_students - math_and_foreign_language_no_science = 45 :=
by 
  sorry

end ninth_graders_only_math_l88_88567


namespace intersection_M_N_l88_88505

def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {x | x^2 - 4 * x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} :=
by sorry

end intersection_M_N_l88_88505


namespace toucan_count_correct_l88_88469

def initial_toucans : ℕ := 2
def toucans_joined : ℕ := 1
def total_toucans : ℕ := initial_toucans + toucans_joined

theorem toucan_count_correct : total_toucans = 3 := by
  sorry

end toucan_count_correct_l88_88469


namespace prime_squared_difference_divisible_by_24_l88_88467

theorem prime_squared_difference_divisible_by_24 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) :
  24 ∣ (p^2 - q^2) :=
sorry

end prime_squared_difference_divisible_by_24_l88_88467


namespace sum_of_distinct_integers_l88_88875

noncomputable def distinct_integers (p q r s t : ℤ) : Prop :=
  (p ≠ q) ∧ (p ≠ r) ∧ (p ≠ s) ∧ (p ≠ t) ∧ 
  (q ≠ r) ∧ (q ≠ s) ∧ (q ≠ t) ∧ 
  (r ≠ s) ∧ (r ≠ t) ∧ 
  (s ≠ t)

theorem sum_of_distinct_integers
  (p q r s t : ℤ)
  (h_distinct : distinct_integers p q r s t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -120) :
  p + q + r + s + t = 22 :=
  sorry

end sum_of_distinct_integers_l88_88875


namespace mean_of_three_l88_88179

theorem mean_of_three (a b c : ℝ) (h : (a + b + c + 105) / 4 = 92) : (a + b + c) / 3 = 87.7 :=
by
  sorry

end mean_of_three_l88_88179


namespace f_divisible_by_27_l88_88842

theorem f_divisible_by_27 (n : ℕ) : 27 ∣ (2^(2*n - 1) - 9 * n^2 + 21 * n - 14) :=
sorry

end f_divisible_by_27_l88_88842


namespace exist_indices_eq_l88_88459

theorem exist_indices_eq (p q n : ℕ) (x : ℕ → ℤ) 
    (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_n : 0 < n) 
    (h_pq_n : p + q < n) 
    (h_x0 : x 0 = 0) 
    (h_xn : x n = 0) 
    (h_step : ∀ i, 1 ≤ i ∧ i ≤ n → (x i - x (i - 1) = p ∨ x i - x (i - 1) = -q)) :
    ∃ (i j : ℕ), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
sorry

end exist_indices_eq_l88_88459


namespace explicit_expression_l88_88617

variable {α : Type*} [LinearOrder α] {f : α → α}

/-- Given that the function satisfies a specific condition, prove the function's explicit expression. -/
theorem explicit_expression (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : 
  ∀ x, f x = 3 * x + 2 :=
by
  sorry

end explicit_expression_l88_88617


namespace jerry_time_proof_l88_88779

noncomputable def tom_walk_speed (step_length_tom : ℕ) (pace_tom : ℕ) : ℕ := 
  step_length_tom * pace_tom

noncomputable def tom_distance_to_office (walk_speed_tom : ℕ) (time_tom : ℕ) : ℕ :=
  walk_speed_tom * time_tom

noncomputable def jerry_walk_speed (step_length_jerry : ℕ) (pace_jerry : ℕ) : ℕ :=
  step_length_jerry * pace_jerry

noncomputable def jerry_time_to_office (distance_to_office : ℕ) (walk_speed_jerry : ℕ) : ℚ :=
  distance_to_office / walk_speed_jerry

theorem jerry_time_proof :
  let step_length_tom := 80
  let pace_tom := 85
  let time_tom := 20
  let step_length_jerry := 70
  let pace_jerry := 110
  let office_distance := tom_distance_to_office (tom_walk_speed step_length_tom pace_tom) time_tom
  let jerry_speed := jerry_walk_speed step_length_jerry pace_jerry
  jerry_time_to_office office_distance jerry_speed = 53/3 := 
by
  sorry

end jerry_time_proof_l88_88779


namespace nineteen_times_eight_pow_n_plus_seventeen_is_composite_l88_88491

theorem nineteen_times_eight_pow_n_plus_seventeen_is_composite 
  (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
sorry

end nineteen_times_eight_pow_n_plus_seventeen_is_composite_l88_88491


namespace part1_part2_l88_88662

-- Lean 4 statement for proving A == 2B
theorem part1 (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) 
    (h₃ : 0 < B) (h₄ : B < π / 2) (h₅ : 0 < C) (h₆ : C < π / 2) (h₇ : A + B + C = π)
    (h₈ : c = 2 * b * Real.cos A + b) : A = 2 * B :=
by sorry

-- Lean 4 statement for finding range of area of ∆ABD
theorem part2 (B : ℝ) (c : ℝ) (h₁ : 0 < B) (h₂ : B < π / 2) 
    (h₃ : A = 2 * B) (h₄ : c = 2) : 
    (Real.tan (π / 6) < (1 / 2) * c * (1 / Real.cos B) * Real.sin B) ∧ 
    ((1 / 2) * c * (1 / Real.cos B) * Real.sin B < 1) :=
by sorry

end part1_part2_l88_88662


namespace ice_cream_sales_l88_88783

theorem ice_cream_sales : 
  let tuesday_sales := 12000
  let wednesday_sales := 2 * tuesday_sales
  let total_sales := tuesday_sales + wednesday_sales
  total_sales = 36000 := 
by 
  sorry

end ice_cream_sales_l88_88783


namespace list_price_of_article_l88_88805

theorem list_price_of_article (P : ℝ) (h : 0.882 * P = 57.33) : P = 65 :=
by
  sorry

end list_price_of_article_l88_88805


namespace find_k_l88_88234

theorem find_k (k : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) ∧ 
  (∀ x : ℝ, y = k * x + 4) ∧ 
  (1, 5) ∈ { p | ∃ x, p = (x, 2 * x + 3) } ∧ 
  (1, 5) ∈ { q | ∃ x, q = (x, k * x + 4) } → 
  k = 1 :=
by
  sorry

end find_k_l88_88234


namespace chemical_transformations_correct_l88_88533

def ethylbenzene : String := "C6H5CH2CH3"
def brominate (A : String) : String := "C6H5CH(Br)CH3"
def hydrolyze (B : String) : String := "C6H5CH(OH)CH3"
def dehydrate (C : String) : String := "C6H5CH=CH2"
def oxidize (D : String) : String := "C6H5COOH"
def brominate_with_catalyst (E : String) : String := "m-C6H4(Br)COOH"

def sequence_of_transformations : Prop :=
  ethylbenzene = "C6H5CH2CH3" ∧
  brominate ethylbenzene = "C6H5CH(Br)CH3" ∧
  hydrolyze (brominate ethylbenzene) = "C6H5CH(OH)CH3" ∧
  dehydrate (hydrolyze (brominate ethylbenzene)) = "C6H5CH=CH2" ∧
  oxidize (dehydrate (hydrolyze (brominate ethylbenzene))) = "C6H5COOH" ∧
  brominate_with_catalyst (oxidize (dehydrate (hydrolyze (brominate ethylbenzene)))) = "m-C6H4(Br)COOH"

theorem chemical_transformations_correct : sequence_of_transformations :=
by
  -- proof would go here
  sorry

end chemical_transformations_correct_l88_88533


namespace jack_needs_more_money_l88_88236

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end jack_needs_more_money_l88_88236


namespace intersection_complement_eq_three_l88_88071

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_eq_three : N ∩ (U \ M) = {3} := by
  sorry

end intersection_complement_eq_three_l88_88071


namespace hundredth_number_is_201_l88_88483

-- Mathematical definition of the sequence
def counting_sequence (n : ℕ) : ℕ :=
  3 + (n - 1) * 2

-- Statement to prove
theorem hundredth_number_is_201 : counting_sequence 100 = 201 :=
by
  sorry

end hundredth_number_is_201_l88_88483


namespace solution_set_I_range_of_m_II_l88_88380

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set_I : {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x ≤ 3} :=
sorry

theorem range_of_m_II (x : ℝ) (hx : x > 0) : ∃ m : ℝ, ∀ (x : ℝ), f x ≤ m - x - 4 / x → m ≥ 5 :=
sorry

end solution_set_I_range_of_m_II_l88_88380


namespace find_e_l88_88302

theorem find_e (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + b = 32)
  (h6 : a + c = 36)
  (h7 : b + c = 37)
  (h8 : c + e = 48)
  (h9 : d + e = 51) : e = 55 / 2 :=
  sorry

end find_e_l88_88302


namespace polygon_sides_l88_88265

theorem polygon_sides (n : ℕ) (hn : 3 ≤ n) (H : (n * (n - 3)) / 2 = 15) : n = 7 :=
by
  sorry

end polygon_sides_l88_88265


namespace vector_BC_l88_88112

def vector_subtraction (v1 v2 : ℤ × ℤ) : ℤ × ℤ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem vector_BC (BA CA BC : ℤ × ℤ) (hBA : BA = (2, 3)) (hCA : CA = (4, 7)) :
  BC = vector_subtraction BA CA → BC = (-2, -4) :=
by
  intro hBC
  rw [vector_subtraction, hBA, hCA] at hBC
  simpa using hBC

end vector_BC_l88_88112


namespace part_one_l88_88415

theorem part_one (m : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = m * Real.exp x - x - 2) :
  (∀ x : ℝ, f x > 0) → m > Real.exp 1 :=
sorry

end part_one_l88_88415


namespace beads_taken_out_l88_88037

/--
There is 1 green bead, 2 brown beads, and 3 red beads in a container.
Tom took some beads out of the container and left 4 in.
Prove that Tom took out 2 beads.
-/
theorem beads_taken_out : 
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  initial_beads - beads_left = 2 :=
by
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  show initial_beads - beads_left = 2
  sorry

end beads_taken_out_l88_88037


namespace pages_needed_l88_88551

theorem pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) (total_packs : packs = 60) (cards_in_pack : cards_per_pack = 7) (capacity_per_page : cards_per_page = 10) : (packs * cards_per_pack) / cards_per_page = 42 := 
by
  -- Utilize the conditions
  have H1 : packs = 60 := total_packs
  have H2 : cards_per_pack = 7 := cards_in_pack
  have H3 : cards_per_page = 10 := capacity_per_page
  -- Use these to simplify and prove the target expression 
  sorry

end pages_needed_l88_88551


namespace lcm_of_lap_times_l88_88547

theorem lcm_of_lap_times :
  Nat.lcm (Nat.lcm 5 8) 10 = 40 := by
  sorry

end lcm_of_lap_times_l88_88547


namespace negation_equiv_l88_88871

noncomputable def negate_existential : Prop :=
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0

noncomputable def universal_negation : Prop :=
  ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0

theorem negation_equiv : negate_existential = universal_negation :=
by
  -- Proof to be filled in
  sorry

end negation_equiv_l88_88871


namespace minimize_distance_sum_l88_88917

open Real

noncomputable def distance_squared (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem minimize_distance_sum : 
  ∀ P : ℝ × ℝ, (P.1 = P.2) → 
    let A : ℝ × ℝ := (1, -1)
    let B : ℝ × ℝ := (2, 2)
    (distance_squared P A + distance_squared P B) ≥ 
    (distance_squared (1, 1) A + distance_squared (1, 1) B) := by
  intro P hP
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (2, 2)
  sorry

end minimize_distance_sum_l88_88917


namespace river_depth_in_mid_may_l88_88186

variable (D : ℕ)
variable (h1 : D + 10 - 5 + 8 = 45)

theorem river_depth_in_mid_may (h1 : D + 13 = 45) : D = 32 := by
  sorry

end river_depth_in_mid_may_l88_88186


namespace value_of_n_l88_88036

theorem value_of_n (n : ℝ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : n = 3.5 :=
by sorry

end value_of_n_l88_88036


namespace positive_integer_solutions_l88_88183

theorem positive_integer_solutions :
  ∀ m n : ℕ, 0 < m ∧ 0 < n ∧ 3^m - 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by
  sorry

end positive_integer_solutions_l88_88183


namespace alberto_more_than_bjorn_and_charlie_l88_88049

theorem alberto_more_than_bjorn_and_charlie (time : ℕ) 
  (alberto_speed bjorn_speed charlie_speed: ℕ) 
  (alberto_distance bjorn_distance charlie_distance : ℕ) :
  time = 6 ∧ alberto_speed = 10 ∧ bjorn_speed = 8 ∧ charlie_speed = 9
  ∧ alberto_distance = alberto_speed * time
  ∧ bjorn_distance = bjorn_speed * time
  ∧ charlie_distance = charlie_speed * time
  → (alberto_distance - bjorn_distance = 12) ∧ (alberto_distance - charlie_distance = 6) :=
by
  sorry

end alberto_more_than_bjorn_and_charlie_l88_88049


namespace energy_equivalence_l88_88751

def solar_energy_per_sqm := 1.3 * 10^8
def china_land_area := 9.6 * 10^6
def expected_coal_energy := 1.248 * 10^15

theorem energy_equivalence : 
  solar_energy_per_sqm * china_land_area = expected_coal_energy := 
by
  sorry

end energy_equivalence_l88_88751


namespace shift_parabola_5_units_right_l88_88377

def original_parabola (x : ℝ) : ℝ := x^2 + 3
def shifted_parabola (x : ℝ) : ℝ := (x-5)^2 + 3

theorem shift_parabola_5_units_right : ∀ x : ℝ, shifted_parabola x = original_parabola (x - 5) :=
by {
  -- This is the mathematical equivalence that we're proving
  sorry
}

end shift_parabola_5_units_right_l88_88377


namespace remainder_div2_l88_88052

   theorem remainder_div2 :
     ∀ z x : ℕ, (∃ k : ℕ, z = 4 * k) → (∃ n : ℕ, x = 2 * n) → (z + x + 4 + z + 3) % 2 = 1 :=
   by
     intros z x h1 h2
     sorry
   
end remainder_div2_l88_88052


namespace abs_eq_iff_mul_nonpos_l88_88417

theorem abs_eq_iff_mul_nonpos (a b : ℝ) : |a - b| = |a| + |b| ↔ a * b ≤ 0 :=
sorry

end abs_eq_iff_mul_nonpos_l88_88417


namespace exist_nat_nums_l88_88977

theorem exist_nat_nums :
  ∃ (a b c d : ℕ), (a / (b : ℚ) + c / (d : ℚ) = 1) ∧ (a / (d : ℚ) + c / (b : ℚ) = 2008) :=
sorry

end exist_nat_nums_l88_88977


namespace num_marked_cells_at_least_num_cells_in_one_square_l88_88569

-- Defining the total number of squares
def num_squares : ℕ := 2009

-- A square covers a cell if it is within its bounds.
-- A cell is marked if it is covered by an odd number of squares.
-- We have to show that the number of marked cells is at least the number of cells in one square.
theorem num_marked_cells_at_least_num_cells_in_one_square (side_length : ℕ) : 
  side_length * side_length ≤ (num_squares : ℕ) :=
sorry

end num_marked_cells_at_least_num_cells_in_one_square_l88_88569


namespace frustum_volume_fraction_l88_88771

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * height

noncomputable def fraction_of_frustum (base_edge height : ℝ) : ℝ :=
  let original_volume := volume_pyramid base_edge height
  let smaller_volume := volume_pyramid (base_edge / 5) (height / 5)
  let frustum_volume := original_volume - smaller_volume
  frustum_volume / original_volume

theorem frustum_volume_fraction :
  fraction_of_frustum 40 20 = 63 / 64 :=
by sorry

end frustum_volume_fraction_l88_88771


namespace num_administrative_personnel_l88_88905

noncomputable def total_employees : ℕ := 280
noncomputable def sample_size : ℕ := 56
noncomputable def ordinary_staff_sample : ℕ := 49

theorem num_administrative_personnel (n : ℕ) (h1 : total_employees = 280) 
(h2 : sample_size = 56) (h3 : ordinary_staff_sample = 49) : 
n = 35 := 
by
  have h_proportion : (sample_size - ordinary_staff_sample) / sample_size = n / total_employees := by sorry
  have h_sol : n = (sample_size - ordinary_staff_sample) * (total_employees / sample_size) := by sorry
  have h_n : n = 35 := by sorry
  exact h_n

end num_administrative_personnel_l88_88905


namespace interior_angle_of_regular_polygon_l88_88388

theorem interior_angle_of_regular_polygon (n : ℕ) (h_diagonals : n * (n - 3) / 2 = n) :
    n = 5 ∧ (5 - 2) * 180 / 5 = 108 := by
  sorry

end interior_angle_of_regular_polygon_l88_88388


namespace intersection_of_A_and_B_l88_88426

def A : Set (ℝ × ℝ) := {p | p.snd = 3 * p.fst - 2}
def B : Set (ℝ × ℝ) := {p | p.snd = p.fst ^ 2}

theorem intersection_of_A_and_B :
  {p : ℝ × ℝ | p ∈ A ∧ p ∈ B} = {(1, 1), (2, 4)} :=
by
  sorry

end intersection_of_A_and_B_l88_88426


namespace union_complement_U_B_l88_88541

def U : Set ℤ := { x | -3 < x ∧ x < 3 }
def A : Set ℤ := { 1, 2 }
def B : Set ℤ := { -2, -1, 2 }

theorem union_complement_U_B : A ∪ (U \ B) = { 0, 1, 2 } := by
  sorry

end union_complement_U_B_l88_88541


namespace lee_can_make_36_cookies_l88_88382

-- Conditions
def initial_cups_of_flour : ℕ := 2
def initial_cookies_made : ℕ := 18
def initial_total_flour : ℕ := 5
def spilled_flour : ℕ := 1

-- Define the remaining cups of flour after spilling
def remaining_flour := initial_total_flour - spilled_flour

-- Define the proportion to solve for the number of cookies made with remaining_flour
def cookies_with_remaining_flour (c : ℕ) : Prop :=
  (initial_cookies_made / initial_cups_of_flour) = (c / remaining_flour)

-- The statement to prove
theorem lee_can_make_36_cookies : cookies_with_remaining_flour 36 :=
  sorry

end lee_can_make_36_cookies_l88_88382


namespace right_triangle_area_l88_88282

theorem right_triangle_area (h : Real) (a : Real) (b : Real) (c : Real) (h_is_hypotenuse : h = 13) (a_is_leg : a = 5) (pythagorean_theorem : a^2 + b^2 = h^2) : (1 / 2) * a * b = 30 := 
by 
  sorry

end right_triangle_area_l88_88282


namespace smallest_other_divisor_of_40_l88_88032

theorem smallest_other_divisor_of_40 (n : ℕ) (h₁ : n > 1) (h₂ : 40 % n = 0) (h₃ : n ≠ 8) :
  (∀ m : ℕ, m > 1 → 40 % m = 0 → m ≠ 8 → n ≤ m) → n = 5 :=
by 
  sorry

end smallest_other_divisor_of_40_l88_88032


namespace problem_1_problem_2_l88_88540

noncomputable def a (k : ℝ) : ℝ × ℝ := (2, k)
noncomputable def b : ℝ × ℝ := (1, 1)
noncomputable def a_minus_3b (k : ℝ) : ℝ × ℝ := (2 - 3 * 1, k - 3 * 1)

-- First problem: Prove that k = 4 given vectors a and b, and the condition that b is perpendicular to (a - 3b)
theorem problem_1 (k : ℝ) (h : b.1 * (a_minus_3b k).1 + b.2 * (a_minus_3b k).2 = 0) : k = 4 :=
sorry

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def cosine (v w : ℝ × ℝ) : ℝ := dot_product v w / (magnitude v * magnitude w)

-- Second problem: Prove that the cosine value of the angle between a and b is 3√10/10 when k is 4
theorem problem_2 (k : ℝ) (hk : k = 4) : cosine (a k) b = 3 * Real.sqrt 10 / 10 :=
sorry

end problem_1_problem_2_l88_88540


namespace min_rubles_reaching_50_points_l88_88168

-- Define conditions and prove the required rubles amount
def min_rubles_needed : ℕ := 11

theorem min_rubles_reaching_50_points (points : ℕ) (rubles : ℕ) : points = 50 ∧ rubles = min_rubles_needed → rubles = 11 :=
by
  intro h
  sorry

end min_rubles_reaching_50_points_l88_88168


namespace complex_number_C_l88_88514

-- Define the complex numbers corresponding to points A and B
def A : ℂ := 1 + 2 * Complex.I
def B : ℂ := 3 - 5 * Complex.I

-- Prove the complex number corresponding to point C
theorem complex_number_C :
  ∃ C : ℂ, (C = 10 - 3 * Complex.I) ∧ 
           (A = 1 + 2 * Complex.I) ∧ 
           (B = 3 - 5 * Complex.I) ∧ 
           -- Square with vertices in counterclockwise order
           True := 
sorry

end complex_number_C_l88_88514


namespace line_through_PQ_l88_88210

theorem line_through_PQ (x y : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (3, 2)) (hQ : Q = (1, 4))
  (h_line : ∀ t, (x, y) = (1 - t) • P + t • Q):
  y = x - 2 :=
by
  have h1 : P = ((3 : ℝ), (2 : ℝ)) := hP
  have h2 : Q = ((1 : ℝ), (4 : ℝ)) := hQ
  sorry

end line_through_PQ_l88_88210


namespace product_of_midpoint_coordinates_l88_88876

def x1 := 10
def y1 := -3
def x2 := 4
def y2 := 7

def midpoint_x := (x1 + x2) / 2
def midpoint_y := (y1 + y2) / 2

theorem product_of_midpoint_coordinates : 
  midpoint_x * midpoint_y = 14 :=
by
  sorry

end product_of_midpoint_coordinates_l88_88876


namespace max_cos_x_l88_88641

theorem max_cos_x (x y : ℝ) (h : Real.cos (x - y) = Real.cos x - Real.cos y) : 
  ∃ M, (∀ x, Real.cos x <= M) ∧ M = 1 := 
sorry

end max_cos_x_l88_88641


namespace green_fish_count_l88_88742

theorem green_fish_count (B O G : ℕ) (H1 : B = 40) (H2 : O = B - 15) (H3 : 80 = B + O + G) : G = 15 := 
by 
  sorry

end green_fish_count_l88_88742


namespace problem_l88_88365

def f (x : ℤ) : ℤ := 3 * x - 1
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem (h : ℤ) :
  (g (f (g (3))) : ℚ) / f (g (f (3))) = 69 / 206 :=
by
  sorry

end problem_l88_88365


namespace prime_power_sum_l88_88688

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem prime_power_sum (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  is_perfect_square (p^q + p^r) →
  (p = 2 ∧ ((q = 2 ∧ r = 5) ∨ (q = 5 ∧ r = 2) ∨ (q ≥ 3 ∧ is_prime q ∧ q = r)))
  ∨
  (p = 3 ∧ ((q = 2 ∧ r = 3) ∨ (q = 3 ∧ r = 2))) :=
sorry

end prime_power_sum_l88_88688


namespace population_increase_l88_88731

theorem population_increase (initial_population final_population: ℝ) (r: ℝ) : 
  initial_population = 14000 →
  final_population = 16940 →
  final_population = initial_population * (1 + r) ^ 2 →
  r = 0.1 :=
by
  intros h_initial h_final h_eq
  sorry

end population_increase_l88_88731


namespace max_value_of_expression_l88_88550

theorem max_value_of_expression {a x1 x2 : ℝ}
  (h1 : x1^2 + a * x1 + a = 2)
  (h2 : x2^2 + a * x2 + a = 2)
  (h1_ne_x2 : x1 ≠ x2) :
  ∃ a : ℝ, (x1 - 2 * x2) * (x2 - 2 * x1) = -63 / 8 :=
by
  sorry

end max_value_of_expression_l88_88550


namespace cube_roots_not_arithmetic_progression_l88_88594

theorem cube_roots_not_arithmetic_progression
  (p q r : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (h_distinct: p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ¬ ∃ (d : ℝ) (m n : ℤ), (n ≠ m) ∧ (↑q)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (m : ℝ) * d ∧ (↑r)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (n : ℝ) * d :=
by sorry

end cube_roots_not_arithmetic_progression_l88_88594


namespace original_price_l88_88675

variable (P SP : ℝ)

axiom condition1 : SP = 0.8 * P
axiom condition2 : SP = 480

theorem original_price : P = 600 :=
by
  sorry

end original_price_l88_88675


namespace find_x_l88_88671

theorem find_x (m n k : ℝ) (x z : ℝ) (h1 : x = m * (n / (Real.sqrt z))^3)
  (h2 : x = 3 ∧ z = 12 ∧ 3 * 12 * Real.sqrt 12 = k) :
  (z = 75) → x = 24 / 125 :=
by
  -- Placeholder for proof, these assumptions and conditions would form the basis of the proof.
  sorry

end find_x_l88_88671


namespace y_share_is_correct_l88_88493

noncomputable def share_of_y (a : ℝ) := 0.45 * a

theorem y_share_is_correct :
  ∃ a : ℝ, (1 * a + 0.45 * a + 0.30 * a = 245) ∧ (share_of_y a = 63) :=
by
  sorry

end y_share_is_correct_l88_88493


namespace elmer_saves_14_3_percent_l88_88319

-- Define the problem statement conditions and goal
theorem elmer_saves_14_3_percent (old_efficiency new_efficiency : ℝ) (old_cost new_cost : ℝ) :
  new_efficiency = 1.75 * old_efficiency →
  new_cost = 1.5 * old_cost →
  (500 / old_efficiency * old_cost - 500 / new_efficiency * new_cost) / (500 / old_efficiency * old_cost) * 100 = 14.3 := by
  -- sorry to skip the actual proof
  sorry

end elmer_saves_14_3_percent_l88_88319


namespace sum_of_squares_of_two_numbers_l88_88327

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) :
  x^2 + y^2 = 289 := 
  sorry

end sum_of_squares_of_two_numbers_l88_88327


namespace money_increase_factor_two_years_l88_88677

theorem money_increase_factor_two_years (P : ℝ) (rate : ℝ) (n : ℕ)
  (h_rate : rate = 0.50) (h_n : n = 2) :
  (P * (1 + rate) ^ n) = 2.25 * P :=
by
  -- proof goes here
  sorry

end money_increase_factor_two_years_l88_88677


namespace mechanism_parts_l88_88097

theorem mechanism_parts (L S : ℕ) (h1 : L + S = 30) (h2 : L ≤ 11) (h3 : S ≤ 19) :
  L = 11 ∧ S = 19 :=
by
  sorry

end mechanism_parts_l88_88097


namespace number_of_zeros_g_l88_88211

variable (f : ℝ → ℝ)
variable (hf_cont : continuous f)
variable (hf_diff : differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, x * (deriv f x) + f x > 0)

theorem number_of_zeros_g (hg : ∀ x : ℝ, x > 0 → x * f x + 1 = 0 → false) : 
    ∀ x : ℝ , x > 0 → ¬ (x * f x + 1 = 0) :=
by
  sorry

end number_of_zeros_g_l88_88211


namespace S_eq_Z_l88_88490

noncomputable def set_satisfies_conditions (S : Set ℤ) (a : Fin n → ℤ) :=
  (∀ i : Fin n, a i ∈ S) ∧
  (∀ i j : Fin n, (a i - a j) ∈ S) ∧
  (∀ x y : ℤ, x ∈ S → y ∈ S → x + y ∈ S → x - y ∈ S) ∧
  (Nat.gcd (List.foldr Nat.gcd 0 (Fin.val <$> List.finRange n)) = 1)

theorem S_eq_Z (S : Set ℤ) (a : Fin n → ℤ) (h_cond : set_satisfies_conditions S a) : S = Set.univ :=
  sorry

end S_eq_Z_l88_88490


namespace red_to_blue_ratio_l88_88057

theorem red_to_blue_ratio
    (total_balls : ℕ)
    (num_white_balls : ℕ)
    (num_blue_balls : ℕ)
    (num_red_balls : ℕ) :
    total_balls = 100 →
    num_white_balls = 16 →
    num_blue_balls = num_white_balls + 12 →
    num_red_balls = total_balls - (num_white_balls + num_blue_balls) →
    (num_red_balls / num_blue_balls : ℚ) = 2 :=
by
  intro h1 h2 h3 h4
  -- Proof is omitted
  sorry

end red_to_blue_ratio_l88_88057


namespace greatest_savings_option2_l88_88309

-- Define the initial price
def initial_price : ℝ := 15000

-- Define the discounts for each option
def discounts_option1 : List ℝ := [0.75, 0.85, 0.95]
def discounts_option2 : List ℝ := [0.65, 0.90, 0.95]
def discounts_option3 : List ℝ := [0.70, 0.90, 0.90]

-- Define a function to compute the final price after successive discounts
def final_price (initial : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * d) initial

-- Define the savings for each option
def savings_option1 : ℝ := initial_price - (final_price initial_price discounts_option1)
def savings_option2 : ℝ := initial_price - (final_price initial_price discounts_option2)
def savings_option3 : ℝ := initial_price - (final_price initial_price discounts_option3)

-- Formulate the proof
theorem greatest_savings_option2 :
  max (max savings_option1 savings_option2) savings_option3 = savings_option2 :=
by
  sorry

end greatest_savings_option2_l88_88309


namespace simplify_polynomial_l88_88645

def p (x : ℝ) : ℝ := 3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7
def q (x : ℝ) : ℝ := -x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4
def r (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2

theorem simplify_polynomial (x : ℝ) :
  (p x) + (q x) - (r x) = 6 * x^4 - x^3 + 3 * x + 1 :=
by sorry

end simplify_polynomial_l88_88645


namespace test_end_time_l88_88048

def start_time := 12 * 60 + 35  -- 12 hours 35 minutes in minutes
def duration := 4 * 60 + 50     -- 4 hours 50 minutes in minutes

theorem test_end_time : (start_time + duration) = 17 * 60 + 25 := by
  sorry

end test_end_time_l88_88048


namespace pork_price_increase_l88_88418

variable (x : ℝ)
variable (P_aug P_oct : ℝ)
variable (P_aug := 32)
variable (P_oct := 64)

theorem pork_price_increase :
  P_aug * (1 + x) ^ 2 = P_oct :=
sorry

end pork_price_increase_l88_88418


namespace magnitude_of_2a_minus_b_l88_88216

/-- Definition of the vectors a and b --/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

/-- Proposition stating the magnitude of 2a - b --/
theorem magnitude_of_2a_minus_b : 
  (Real.sqrt ((2 * a.1 - b.1) ^ 2 + (2 * a.2 - b.2) ^ 2)) = Real.sqrt 10 :=
by
  sorry

end magnitude_of_2a_minus_b_l88_88216


namespace smallest_base_for_100_l88_88002

theorem smallest_base_for_100 :
  ∃ (b : ℕ), (b^2 ≤ 100) ∧ (100 < b^3) ∧ ∀ (b' : ℕ), (b'^2 ≤ 100) ∧ (100 < b'^3) → b ≤ b' :=
by
  sorry

end smallest_base_for_100_l88_88002


namespace rice_on_8th_day_l88_88202

variable (a1 : ℕ) (d : ℕ) (n : ℕ)
variable (rice_per_laborer : ℕ)

def is_arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem rice_on_8th_day (ha1 : a1 = 64) (hd : d = 7) (hr : rice_per_laborer = 3) :
  let a8 := is_arithmetic_sequence a1 d 8
  (a8 * rice_per_laborer = 339) :=
by
  sorry

end rice_on_8th_day_l88_88202


namespace minimum_value_2x_4y_l88_88362

theorem minimum_value_2x_4y (x y : ℝ) (h : x + 2 * y = 3) : 
  ∃ (min_val : ℝ), min_val = 2 ^ (5/2) ∧ (2 ^ x + 4 ^ y = min_val) :=
by
  sorry

end minimum_value_2x_4y_l88_88362


namespace polynomial_remainder_division_l88_88995

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (x^4 + 2 * x^2 - 3) % (x^2 + 3 * x + 2) = -21 * x - 21 := 
by
  sorry

end polynomial_remainder_division_l88_88995


namespace range_of_a_l88_88047

def A := {x : ℝ | |x| >= 3}
def B (a : ℝ) := {x : ℝ | x >= a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a <= -3 :=
sorry

end range_of_a_l88_88047


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l88_88698

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l88_88698


namespace darnell_saves_money_l88_88521

-- Define conditions
def current_plan_cost := 12
def text_cost := 1
def call_cost := 3
def texts_per_month := 60
def calls_per_month := 60
def texts_per_unit := 30
def calls_per_unit := 20

-- Define the costs for the alternative plan
def alternative_texting_cost := (text_cost * (texts_per_month / texts_per_unit))
def alternative_calling_cost := (call_cost * (calls_per_month / calls_per_unit))
def alternative_plan_cost := alternative_texting_cost + alternative_calling_cost

-- Define the problem to prove
theorem darnell_saves_money :
  current_plan_cost - alternative_plan_cost = 1 :=
by
  sorry

end darnell_saves_money_l88_88521


namespace arc_length_of_pentagon_side_l88_88485

theorem arc_length_of_pentagon_side 
  (r : ℝ) (h : r = 4) :
  (2 * r * Real.pi * (72 / 360)) = (8 * Real.pi / 5) :=
by
  sorry

end arc_length_of_pentagon_side_l88_88485


namespace farmer_total_cows_l88_88502

theorem farmer_total_cows (cows : ℕ) 
  (h1 : 1 / 3 + 1 / 6 + 1 / 8 = 5 / 8) 
  (h2 : (3 / 8) * cows = 15) : 
  cows = 40 := by
  -- Given conditions:
  -- h1: The first three sons receive a total of 5/8 of the cows.
  -- h2: The fourth son receives 3/8 of the cows, which is 15 cows.
  sorry

end farmer_total_cows_l88_88502


namespace find_integers_l88_88130

theorem find_integers (x y : ℤ) 
  (h1 : x * y + (x + y) = 95) 
  (h2 : x * y - (x + y) = 59) : 
  (x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11) :=
by
  sorry

end find_integers_l88_88130


namespace part_a_cube_edge_length_part_b_cube_edge_length_l88_88284

-- Part (a)
theorem part_a_cube_edge_length (small_cubes : ℕ) (edge_length_original : ℤ) :
  small_cubes = 512 → edge_length_original^3 = small_cubes → edge_length_original = 8 :=
by
  intros h1 h2
  sorry

-- Part (b)
theorem part_b_cube_edge_length (small_cubes_internal : ℕ) (edge_length_inner : ℤ) (edge_length_original : ℤ) :
  small_cubes_internal = 512 →
  edge_length_inner^3 = small_cubes_internal → 
  edge_length_original = edge_length_inner + 2 →
  edge_length_original = 10 :=
by
  intros h1 h2 h3
  sorry

end part_a_cube_edge_length_part_b_cube_edge_length_l88_88284


namespace marbles_leftover_l88_88724

theorem marbles_leftover (r p : ℤ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 := by
  sorry

end marbles_leftover_l88_88724


namespace reasoning_is_inductive_l88_88489

-- Define conditions
def conducts_electricity (metal : String) : Prop :=
  metal = "copper" ∨ metal = "iron" ∨ metal = "aluminum" ∨ metal = "gold" ∨ metal = "silver"

-- Define the inductive reasoning type
def is_inductive_reasoning : Prop := 
  ∀ metals, conducts_electricity metals → (∀ m : String, conducts_electricity m → conducts_electricity m)

-- The theorem to prove
theorem reasoning_is_inductive : is_inductive_reasoning :=
by
  sorry

end reasoning_is_inductive_l88_88489


namespace red_balls_count_after_game_l88_88572

structure BagState :=
  (red : Nat)         -- Number of red balls
  (green : Nat)       -- Number of green balls
  (blue : Nat)        -- Number of blue balls
  (yellow : Nat)      -- Number of yellow balls
  (black : Nat)       -- Number of black balls
  (white : Nat)       -- Number of white balls)

def initialBallCount (totalBalls : Nat) : BagState :=
  let totalRatio := 15 + 13 + 17 + 9 + 7 + 23
  { red := totalBalls * 15 / totalRatio
  , green := totalBalls * 13 / totalRatio
  , blue := totalBalls * 17 / totalRatio
  , yellow := totalBalls * 9 / totalRatio
  , black := totalBalls * 7 / totalRatio
  , white := totalBalls * 23 / totalRatio
  }

def finalBallCount (initialState : BagState) : BagState :=
  { red := initialState.red + 400
  , green := initialState.green - 250
  , blue := initialState.blue
  , yellow := initialState.yellow - 100
  , black := initialState.black + 200
  , white := initialState.white - 500
  }

theorem red_balls_count_after_game :
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  final.red = 2185 :=
by
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  sorry

end red_balls_count_after_game_l88_88572


namespace necessary_condition_l88_88024

theorem necessary_condition (m : ℝ) : 
  (∀ x > 0, (x / 2) + (1 / (2 * x)) - (3 / 2) > m) → (m ≤ -1 / 2) :=
by
  -- Proof omitted
  sorry

end necessary_condition_l88_88024


namespace trig_proof_l88_88626

theorem trig_proof (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_proof_l88_88626


namespace exists_fixed_point_subset_l88_88095

-- Definitions of set and function f with the required properties
variable {α : Type} [DecidableEq α]
variable (H : Finset α)
variable (f : Finset α → Finset α)

-- Conditions
axiom increasing_mapping (X Y : Finset α) : X ⊆ Y → f X ⊆ f Y
axiom range_in_H (X : Finset α) : f X ⊆ H

-- Statement to prove
theorem exists_fixed_point_subset : ∃ H₀ ⊆ H, f H₀ = H₀ :=
sorry

end exists_fixed_point_subset_l88_88095


namespace total_preparation_and_cooking_time_l88_88954

def time_to_chop_pepper : Nat := 3
def time_to_chop_onion : Nat := 4
def time_to_grate_cheese_per_omelet : Nat := 1
def time_to_cook_omelet : Nat := 5
def num_peppers : Nat := 4
def num_onions : Nat := 2
def num_omelets : Nat := 5

theorem total_preparation_and_cooking_time :
  num_peppers * time_to_chop_pepper +
  num_onions * time_to_chop_onion +
  num_omelets * (time_to_grate_cheese_per_omelet + time_to_cook_omelet) = 50 := 
by
  sorry

end total_preparation_and_cooking_time_l88_88954


namespace total_marbles_l88_88681

-- Define the given conditions 
def bags : ℕ := 20
def marbles_per_bag : ℕ := 156

-- The theorem stating that the total number of marbles is 3120
theorem total_marbles : bags * marbles_per_bag = 3120 := by
  sorry

end total_marbles_l88_88681


namespace polynomial_exists_int_coeff_l88_88373

theorem polynomial_exists_int_coeff (n : ℕ) (hn : n > 1) : 
  ∃ P : Polynomial ℤ × Polynomial ℤ × Polynomial ℤ → Polynomial ℤ, 
  ∀ x : Polynomial ℤ, P ⟨x^n, x^(n+1), x + x^(n+2)⟩ = x :=
by sorry

end polynomial_exists_int_coeff_l88_88373


namespace can_form_triangle_8_6_4_l88_88817

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle_8_6_4 : can_form_triangle 8 6 4 :=
by
  unfold can_form_triangle
  simp
  exact ⟨by linarith, by linarith, by linarith⟩

end can_form_triangle_8_6_4_l88_88817


namespace line_slope_is_neg_half_l88_88526

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- The main theorem to be proved
theorem line_slope_is_neg_half : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) := by
  sorry

end line_slope_is_neg_half_l88_88526


namespace equal_playtime_l88_88565

theorem equal_playtime (children : ℕ) (total_minutes : ℕ) (simultaneous_players : ℕ) (equal_playtime_per_child : ℕ)
  (h1 : children = 12) (h2 : total_minutes = 120) (h3 : simultaneous_players = 2) (h4 : equal_playtime_per_child = (simultaneous_players * total_minutes) / children) :
  equal_playtime_per_child = 20 := 
by sorry

end equal_playtime_l88_88565


namespace quadratic_equal_roots_l88_88250

theorem quadratic_equal_roots :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 → (0 ≤ 0) ∧ 
  (∀ a b : ℝ, 0 = b^2 - 4 * a * 1 → (x = -b / (2 * a))) :=
by
  sorry

end quadratic_equal_roots_l88_88250


namespace value_of_a_minus_b_l88_88497

theorem value_of_a_minus_b (a b : ℝ) 
  (h₁ : (a-4)*(a+4) = 28*a - 112) 
  (h₂ : (b-4)*(b+4) = 28*b - 112) 
  (h₃ : a ≠ b)
  (h₄ : a > b) :
  a - b = 20 :=
sorry

end value_of_a_minus_b_l88_88497


namespace tan_15_degree_identity_l88_88628

theorem tan_15_degree_identity : (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end tan_15_degree_identity_l88_88628


namespace weight_of_D_l88_88892

open Int

def weights (A B C D : Int) : Prop :=
  A < B ∧ B < C ∧ C < D ∧ 
  A + B = 45 ∧ A + C = 49 ∧ A + D = 55 ∧ 
  B + C = 54 ∧ B + D = 60 ∧ C + D = 64

theorem weight_of_D {A B C D : Int} (h : weights A B C D) : D = 35 := 
  by
    sorry

end weight_of_D_l88_88892


namespace sum_cyc_geq_one_l88_88589

theorem sum_cyc_geq_one (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hcond : a * b + b * c + c * a = a * b * c) :
  (a^4 / (b * (b^4 + c^3)) + b^4 / (c * (c^3 + a^4)) + c^4 / (a * (a^4 + b^3))) ≥ 1 :=
sorry

end sum_cyc_geq_one_l88_88589


namespace parts_in_batch_l88_88005

theorem parts_in_batch (a : ℕ) (h₁ : 20 * (a / 20) + 13 = a) (h₂ : 27 * (a / 27) + 20 = a) 
  (h₃ : 500 ≤ a) (h₄ : a ≤ 600) : a = 533 :=
by sorry

end parts_in_batch_l88_88005


namespace sum_of_corners_9x9_grid_l88_88808

theorem sum_of_corners_9x9_grid : 
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  show topLeft + topRight + bottomLeft + bottomRight = 164
  sorry
}

end sum_of_corners_9x9_grid_l88_88808


namespace ratio_grass_area_weeded_l88_88818

/-- Lucille earns six cents for every weed she pulls. -/
def earnings_per_weed : ℕ := 6

/-- There are eleven weeds in the flower bed. -/
def weeds_flower_bed : ℕ := 11

/-- There are fourteen weeds in the vegetable patch. -/
def weeds_vegetable_patch : ℕ := 14

/-- There are thirty-two weeds in the grass around the fruit trees. -/
def weeds_grass_total : ℕ := 32

/-- Lucille bought a soda for 99 cents on her break. -/
def soda_cost : ℕ := 99

/-- Lucille has 147 cents left after the break. -/
def cents_left : ℕ := 147

/-- Statement to prove: The ratio of the grass area Lucille weeded to the total grass area around the fruit trees is 1:2. -/
theorem ratio_grass_area_weeded :
  (earnings_per_weed * (weeds_flower_bed + weeds_vegetable_patch) + earnings_per_weed * (weeds_flower_bed + (weeds_grass_total - (earnings_per_weed + soda_cost)) / earnings_per_weed) = soda_cost + cents_left)
→ ((earnings_per_weed  * (32 - (147 + 99) / earnings_per_weed)) / weeds_grass_total) = 1 / 2 :=
by
  sorry

end ratio_grass_area_weeded_l88_88818


namespace finite_negatives_condition_l88_88700

-- Define the sequence terms
def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + n * d

-- Define the condition for finite negative terms
def has_finite_negatives (a1 d : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, arithmetic_seq a1 d n ≥ 0

-- Theorem that proves the desired statement
theorem finite_negatives_condition (a1 d : ℝ) (h1 : a1 < 0) (h2 : d > 0) :
  has_finite_negatives a1 d :=
sorry

end finite_negatives_condition_l88_88700


namespace max_value_expr_l88_88324

theorem max_value_expr (x : ℝ) : 
  ( x ^ 6 / (x ^ 12 + 3 * x ^ 8 - 6 * x ^ 6 + 12 * x ^ 4 + 36) <= 1/18 ) :=
by
  sorry

end max_value_expr_l88_88324


namespace brit_age_after_vacation_l88_88799

-- Define the given conditions and the final proof question

-- Rebecca's age is 25 years
def rebecca_age : ℕ := 25

-- Brittany is older than Rebecca by 3 years
def brit_age_before_vacation (rebecca_age : ℕ) : ℕ := rebecca_age + 3

-- Brittany goes on a 4-year vacation
def vacation_duration : ℕ := 4

-- Prove that Brittany’s age when she returns from her vacation is 32
theorem brit_age_after_vacation (rebecca_age vacation_duration : ℕ) : brit_age_before_vacation rebecca_age + vacation_duration = 32 :=
by
  sorry

end brit_age_after_vacation_l88_88799


namespace car_speed_l88_88992

-- Definitions based on the conditions
def distance : ℕ := 375
def time : ℕ := 5

-- Mathematically equivalent proof statement
theorem car_speed : distance / time = 75 := 
  by
  -- The actual proof will be placed here, but we'll skip it for now.
  sorry

end car_speed_l88_88992


namespace locus_of_center_of_circle_l88_88013

theorem locus_of_center_of_circle (x y a : ℝ)
  (hC : x^2 + y^2 - (2 * a^2 - 4) * x - 4 * a^2 * y + 5 * a^4 - 4 = 0) :
  2 * x - y + 4 = 0 ∧ -2 ≤ x ∧ x < 0 :=
sorry

end locus_of_center_of_circle_l88_88013


namespace seed_mixture_x_percentage_l88_88478

theorem seed_mixture_x_percentage (x y : ℝ) (h : 0.40 * x + 0.25 * y = 0.30 * (x + y)) : 
  (x / (x + y)) * 100 = 33.33 := sorry

end seed_mixture_x_percentage_l88_88478


namespace evaluate_expression_l88_88401

theorem evaluate_expression : 
  ((-4 : ℤ) ^ 6) / (4 ^ 4) + (2 ^ 5) * (5 : ℤ) - (7 ^ 2) = 127 :=
by sorry

end evaluate_expression_l88_88401


namespace problem_statement_l88_88930

-- Define what it means to be a quadratic equation
def is_quadratic (eqn : String) : Prop :=
  -- In the context of this solution, we'll define a quadratic equation as one
  -- that fits the form ax^2 + bx + c = 0 where a, b, c are constants and a ≠ 0.
  eqn = "x^2 - 2 = 0"

-- We need to formulate a theorem that checks the validity of which equation is quadratic.
theorem problem_statement :
  is_quadratic "x^2 - 2 = 0" :=
sorry

end problem_statement_l88_88930


namespace total_number_of_elementary_events_is_16_l88_88722

def num_events_three_dice : ℕ := 6 * 6 * 6

theorem total_number_of_elementary_events_is_16 :
  num_events_three_dice = 16 := 
sorry

end total_number_of_elementary_events_is_16_l88_88722


namespace geometric_sequence_a5_l88_88509

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (hq : q = 2) (h_a2a6 : a 2 * a 6 = 16) :
  a 5 = 8 :=
sorry

end geometric_sequence_a5_l88_88509


namespace extremum_of_f_unique_solution_of_equation_l88_88398

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x

theorem extremum_of_f (m : ℝ) (h_pos : 0 < m) :
  ∃ x_min : ℝ, x_min = Real.sqrt m ∧
  ∀ x : ℝ, 0 < x → f x m ≥ f (Real.sqrt m) m :=
sorry

theorem unique_solution_of_equation (m : ℝ) (h_ge_one : 1 ≤ m) :
  ∃! x : ℝ, 0 < x ∧ f x m = x^2 - (m + 1) * x :=
sorry

#check extremum_of_f -- Ensure it can be checked
#check unique_solution_of_equation -- Ensure it can be checked

end extremum_of_f_unique_solution_of_equation_l88_88398


namespace correct_fill_l88_88877

/- Define the conditions and the statement in Lean 4 -/
def sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ ТРИДЦАТЬ ДВЕ БУКВЫ"

/- The condition is that the phrase without the number has 21 characters -/
def initial_length : ℕ := 21

/- Define the term "тридцать две" as the correct number to fill the blank -/
def correct_number := "тридцать две"

/- The target phrase with the correct number filled in -/
def target_sentence := "В ЭТОМ ПРЕДЛОЖЕНИИ " ++ correct_number ++ " БУКВЫ"

/- Prove that the correct number fills the blank correctly -/
theorem correct_fill :
  (String.length target_sentence = 38) :=
by
  /- Convert everything to string length and verify -/
  sorry

end correct_fill_l88_88877


namespace sum_of_squares_of_roots_l88_88152

theorem sum_of_squares_of_roots :
  ∃ x1 x2 : ℝ, (10 * x1 ^ 2 + 15 * x1 - 20 = 0) ∧ (10 * x2 ^ 2 + 15 * x2 - 20 = 0) ∧ (x1 ≠ x2) ∧ x1^2 + x2^2 = 25/4 :=
sorry

end sum_of_squares_of_roots_l88_88152


namespace number_of_sheets_is_9_l88_88680

-- Define the conditions in Lean
variable (n : ℕ) -- Total number of pages

-- The stack is folded and renumbered
axiom folded_and_renumbered : True

-- The sum of numbers on one of the sheets is 74
axiom sum_sheet_is_74 : 2 * n + 2 = 74

-- The number of sheets is the number of pages divided by 4
def number_of_sheets (n : ℕ) : ℕ := n / 4

-- Prove that the number of sheets is 9 given the conditions
theorem number_of_sheets_is_9 : number_of_sheets 36 = 9 :=
by
  sorry

end number_of_sheets_is_9_l88_88680


namespace quotient_is_six_l88_88776

def larger_number (L : ℕ) : Prop := L = 1620
def difference (L S : ℕ) : Prop := L - S = 1365
def division_remainder (L S Q : ℕ) : Prop := L = S * Q + 15

theorem quotient_is_six (L S Q : ℕ) 
  (hL : larger_number L) 
  (hdiff : difference L S) 
  (hdiv : division_remainder L S Q) : Q = 6 :=
sorry

end quotient_is_six_l88_88776


namespace average_salary_l88_88887

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 14000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

theorem average_salary : (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8200 := 
  by 
    sorry

end average_salary_l88_88887


namespace min_value_of_squares_l88_88177

theorem min_value_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := 
by
  -- Proof omitted
  sorry

end min_value_of_squares_l88_88177


namespace combine_expr_l88_88920

variable (a b : ℝ)

theorem combine_expr : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end combine_expr_l88_88920


namespace total_money_9pennies_4nickels_3dimes_l88_88375

def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05
def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10

def total_value (pennies nickels dimes : ℕ) : ℝ :=
  value_of_pennies pennies + value_of_nickels nickels + value_of_dimes dimes

theorem total_money_9pennies_4nickels_3dimes :
  total_value 9 4 3 = 0.59 :=
by 
  sorry

end total_money_9pennies_4nickels_3dimes_l88_88375


namespace min_distance_l88_88123

open Complex

theorem min_distance (z : ℂ) (hz : abs (z + 2 - 2*I) = 1) : abs (z - 2 - 2*I) = 3 :=
sorry

end min_distance_l88_88123


namespace first_person_days_l88_88090

theorem first_person_days (x : ℝ) (hp : 30 ≥ 0) (ht : 10 ≥ 0) (h_work : 1/x + 1/30 = 1/10) : x = 15 :=
by
  -- Begin by acknowledging the assumptions: hp, ht, and h_work
  sorry

end first_person_days_l88_88090


namespace age_problem_contradiction_l88_88348

theorem age_problem_contradiction (C1 C2 : ℕ) (k : ℕ)
  (h1 : 15 = k * (C1 + C2))
  (h2 : 20 = 2 * (C1 + 5 + C2 + 5)) : false :=
by
  sorry

end age_problem_contradiction_l88_88348


namespace joan_total_seashells_l88_88554

-- Definitions of the conditions
def joan_initial_seashells : ℕ := 79
def mike_additional_seashells : ℕ := 63

-- Definition of the proof problem statement
theorem joan_total_seashells : joan_initial_seashells + mike_additional_seashells = 142 :=
by
  -- Proof would go here
  sorry

end joan_total_seashells_l88_88554


namespace multiples_of_3_or_5_but_not_6_l88_88194

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 150) :
  (∃ m : ℕ, m ≤ 150 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ m % 6 ≠ 0)) ↔ n = 45 :=
by {
  sorry
}

end multiples_of_3_or_5_but_not_6_l88_88194


namespace inequality_abc_l88_88638

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by 
  sorry

end inequality_abc_l88_88638


namespace monkey2_peach_count_l88_88162

noncomputable def total_peaches : ℕ := 81
def monkey1_share (p : ℕ) : ℕ := (5 * p) / 6
def remaining_after_monkey1 (p : ℕ) : ℕ := p - monkey1_share p
def monkey2_share (p : ℕ) : ℕ := (5 * remaining_after_monkey1 p) / 9
def remaining_after_monkey2 (p : ℕ) : ℕ := remaining_after_monkey1 p - monkey2_share p
def monkey3_share (p : ℕ) : ℕ := remaining_after_monkey2 p

theorem monkey2_peach_count : monkey2_share total_peaches = 20 :=
by
  sorry

end monkey2_peach_count_l88_88162


namespace functional_equation_solution_l88_88747

theorem functional_equation_solution:
  (∀ f : ℝ → ℝ, (∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x * y * z = 1 →
  f x ^ 2 - f y * f z = x * (x + y + z) * (f x + f y + f z)) →
  (∀ x : ℝ, x ≠ 0 → ( (f x = x^2 - 1/x) ∨ (f x = 0)))) :=
by
  sorry

end functional_equation_solution_l88_88747


namespace abs_sin_diff_le_abs_sin_sub_l88_88446

theorem abs_sin_diff_le_abs_sin_sub (A B : ℝ) (hA : 0 ≤ A) (hA' : A ≤ π) (hB : 0 ≤ B) (hB' : B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| :=
by
  -- Proof would go here
  sorry

end abs_sin_diff_le_abs_sin_sub_l88_88446


namespace chloe_total_books_l88_88009

noncomputable def total_books (average_books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (science_fiction_shelves : ℕ) (history_shelves : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + science_fiction_shelves + history_shelves) * average_books_per_shelf

theorem chloe_total_books : 
  total_books 85 7 5 3 2 = 14500 / 100 :=
  by
  sorry

end chloe_total_books_l88_88009


namespace probability_first_player_takes_card_l88_88685

variable (n : ℕ) (i : ℕ)

-- Conditions
def even_n : Prop := ∃ k, n = 2 * k
def valid_i : Prop := 1 ≤ i ∧ i ≤ n

-- The key function (probability) and theorem to prove
def P (i n : ℕ) : ℚ := (i - 1) / (n - 1)

theorem probability_first_player_takes_card :
  even_n n → valid_i n i → P i n = (i - 1) / (n - 1) :=
by
  intro h1 h2
  sorry

end probability_first_player_takes_card_l88_88685


namespace fred_sheets_left_l88_88976

def sheets_fred_had_initially : ℕ := 212
def sheets_jane_given : ℕ := 307
def planned_percentage_more : ℕ := 50
def given_percentage : ℕ := 25

-- Prove that after all transactions, Fred has 389 sheets left
theorem fred_sheets_left :
  let planned_sheets := (sheets_jane_given * 100) / (planned_percentage_more + 100)
  let sheets_jane_actual := planned_sheets + (planned_sheets * planned_percentage_more) / 100
  let total_sheets := sheets_fred_had_initially + sheets_jane_actual
  let charles_given := (total_sheets * given_percentage) / 100
  let fred_sheets_final := total_sheets - charles_given
  fred_sheets_final = 389 := 
by
  sorry

end fred_sheets_left_l88_88976


namespace last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l88_88578

noncomputable def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_1989_1989:
  last_digit (1989 ^ 1989) = 9 := 
sorry

theorem last_digit_1989_1992:
  last_digit (1989 ^ 1992) = 1 := 
sorry

theorem last_digit_1992_1989:
  last_digit (1992 ^ 1989) = 2 := 
sorry

theorem last_digit_1992_1992:
  last_digit (1992 ^ 1992) = 6 := 
sorry

end last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l88_88578


namespace find_value_of_a_squared_b_plus_ab_squared_l88_88939

theorem find_value_of_a_squared_b_plus_ab_squared 
  (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : ab = 2) : 
  a^2 * b + a * b^2 = -6 :=
by 
  sorry

end find_value_of_a_squared_b_plus_ab_squared_l88_88939


namespace find_integer_pair_l88_88713

theorem find_integer_pair (x y : ℤ) :
  (x + 2)^4 - x^4 = y^3 → (x = -1 ∧ y = 0) :=
by
  intro h
  sorry

end find_integer_pair_l88_88713


namespace kite_area_is_28_l88_88494

noncomputable def area_of_kite : ℝ :=
  let base_upper := 8
  let height_upper := 2
  let base_lower := 8
  let height_lower := 5
  let area_upper := (1 / 2 : ℝ) * base_upper * height_upper
  let area_lower := (1 / 2 : ℝ) * base_lower * height_lower
  area_upper + area_lower

theorem kite_area_is_28 :
  area_of_kite = 28 :=
by
  simp [area_of_kite]
  sorry

end kite_area_is_28_l88_88494


namespace geometric_sequence_general_formula_no_arithmetic_sequence_l88_88711

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Condition: Sum of the first n terms of the sequence {a_n} is S_n
-- and S_n = 2a_n - n for n \in \mathbb{N}^*.
axiom sum_condition (n : ℕ) (h : n > 0) : S n = 2 * a n - n

-- Question 1: Prove that the sequence {a_n + 1} forms a geometric sequence.
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r, r ≠ 0 ∧ ∀ m, m > 0 → a (m + 1) + 1 = r * (a m + 1) := 
sorry

-- Question 2: Find the general formula for the sequence {a_n}.
theorem general_formula (n : ℕ) (h : n > 0) : a n = 2 ^ n - 1 := 
sorry

-- Question 3: Prove that there do not exist three consecutive terms in the sequence {a_n} that can form an arithmetic sequence.
theorem no_arithmetic_sequence (k : ℕ) (h : k > 0) : ¬ ∃ k, k > 0 ∧ a k = (a (k + 1) + a (k + 2)) / 2 := 
sorry

end geometric_sequence_general_formula_no_arithmetic_sequence_l88_88711


namespace find_divisor_l88_88121

theorem find_divisor (D : ℕ) : 
  (242 % D = 15) ∧ 
  (698 % D = 27) ∧ 
  ((242 + 698) % D = 5) → 
  D = 42 := 
by 
  sorry

end find_divisor_l88_88121


namespace frequency_of_zero_in_3021004201_l88_88246

def digit_frequency (n : Nat) (d : Nat) :  Rat :=
  let digits := n.digits 10
  let count_d := digits.count d
  (count_d : Rat) / digits.length

theorem frequency_of_zero_in_3021004201 : 
  digit_frequency 3021004201 0 = 0.4 := 
by 
  sorry

end frequency_of_zero_in_3021004201_l88_88246


namespace find_number_of_observations_l88_88971

theorem find_number_of_observations 
  (n : ℕ) 
  (mean_before_correction : ℝ)
  (incorrect_observation : ℝ)
  (correct_observation : ℝ)
  (mean_after_correction : ℝ) 
  (h0 : mean_before_correction = 36)
  (h1 : incorrect_observation = 23)
  (h2 : correct_observation = 45)
  (h3 : mean_after_correction = 36.5) 
  (h4 : (n * mean_before_correction + (correct_observation - incorrect_observation)) / n = mean_after_correction) : 
  n = 44 := 
by
  sorry

end find_number_of_observations_l88_88971


namespace nonagon_diagonals_l88_88542

-- Define nonagon and its properties
def is_nonagon (n : ℕ) : Prop := n = 9
def has_parallel_sides (n : ℕ) : Prop := n = 9 ∧ true

-- Define the formula for calculating diagonals in a convex polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The main theorem statement
theorem nonagon_diagonals :
  ∀ (n : ℕ), is_nonagon n → has_parallel_sides n → diagonals n = 27 :=  by 
  intros n hn _ 
  rw [is_nonagon] at hn
  rw [hn]
  sorry

end nonagon_diagonals_l88_88542


namespace fraction_zero_solution_l88_88581

theorem fraction_zero_solution (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := 
by sorry

end fraction_zero_solution_l88_88581


namespace find_pq_l88_88975

noncomputable def area_of_triangle (p q : ℝ) : ℝ := 1/2 * (12 / p) * (12 / q)

theorem find_pq (p q : ℝ) (hp : p > 0) (hq : q > 0) (harea : area_of_triangle p q = 12) : p * q = 6 := 
by
  sorry

end find_pq_l88_88975


namespace range_of_expression_l88_88203

theorem range_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  0 < (x * y + y * z + z * x - 2 * x * y * z) ∧ (x * y + y * z + z * x - 2 * x * y * z) ≤ 7 / 27 := by
  sorry

end range_of_expression_l88_88203


namespace calculate_sum_of_triangles_l88_88649

def operation_triangle (a b c : Int) : Int :=
  a * b - c 

theorem calculate_sum_of_triangles :
  operation_triangle 3 4 5 + operation_triangle 1 2 4 + operation_triangle 2 5 6 = 9 :=
by 
  sorry

end calculate_sum_of_triangles_l88_88649


namespace evaluate_expression_l88_88556

theorem evaluate_expression (x : ℤ) (z : ℤ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 :=
by
  sorry

end evaluate_expression_l88_88556


namespace average_home_runs_correct_l88_88560

-- Define the number of players hitting specific home runs
def players_5_hr : ℕ := 3
def players_7_hr : ℕ := 2
def players_9_hr : ℕ := 1
def players_11_hr : ℕ := 2
def players_13_hr : ℕ := 1

-- Calculate the total number of home runs and total number of players
def total_hr : ℕ := 5 * players_5_hr + 7 * players_7_hr + 9 * players_9_hr + 11 * players_11_hr + 13 * players_13_hr
def total_players : ℕ := players_5_hr + players_7_hr + players_9_hr + players_11_hr + players_13_hr

-- Calculate the average number of home runs
def average_home_runs : ℚ := total_hr / total_players

-- The theorem we need to prove
theorem average_home_runs_correct : average_home_runs = 73 / 9 :=
by
  sorry

end average_home_runs_correct_l88_88560


namespace total_pieces_of_junk_mail_l88_88428

def pieces_per_block : ℕ := 48
def num_blocks : ℕ := 4

theorem total_pieces_of_junk_mail : (pieces_per_block * num_blocks) = 192 := by
  sorry

end total_pieces_of_junk_mail_l88_88428


namespace correct_division_result_l88_88379

theorem correct_division_result {x : ℕ} (h : 3 * x = 90) : x / 3 = 10 :=
by
  -- placeholder for the actual proof
  sorry

end correct_division_result_l88_88379


namespace min_value_f_l88_88372

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + 4 * x + 20) + Real.sqrt (x^2 + 2 * x + 10)

theorem min_value_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 :=
by
  sorry

end min_value_f_l88_88372


namespace negation_of_positive_l88_88963

def is_positive (x : ℝ) : Prop := x > 0
def is_non_positive (x : ℝ) : Prop := x ≤ 0

theorem negation_of_positive (a b c : ℝ) :
  (¬ (is_positive a ∨ is_positive b ∨ is_positive c)) ↔ (is_non_positive a ∧ is_non_positive b ∧ is_non_positive c) :=
by
  sorry

end negation_of_positive_l88_88963


namespace base7_perfect_square_xy5z_l88_88222

theorem base7_perfect_square_xy5z (n : ℕ) (x y z : ℕ) (hx : x ≠ 0) (hn : n = 343 * x + 49 * y + 35 + z) (hsq : ∃ m : ℕ, n = m * m) : z = 1 ∨ z = 6 :=
sorry

end base7_perfect_square_xy5z_l88_88222


namespace sin_cos_identity_l88_88361

theorem sin_cos_identity {x : Real} 
    (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
    Real.sin x ^ 12 + Real.cos x ^ 12 = 5 / 18 :=
sorry

end sin_cos_identity_l88_88361


namespace minimum_breaks_l88_88706

-- Definitions based on conditions given in the problem statement
def longitudinal_grooves : ℕ := 2
def transverse_grooves : ℕ := 3

-- The problem statement to be proved
theorem minimum_breaks (l t : ℕ) (hl : l = longitudinal_grooves) (ht : t = transverse_grooves) :
  l + t = 4 :=
by
  sorry

end minimum_breaks_l88_88706


namespace apples_per_box_l88_88902

-- Defining the given conditions
variable (apples_per_crate : ℤ)
variable (number_of_crates : ℤ)
variable (rotten_apples : ℤ)
variable (number_of_boxes : ℤ)

-- Stating the facts based on given conditions
def total_apples := apples_per_crate * number_of_crates
def remaining_apples := total_apples - rotten_apples

-- The statement to prove
theorem apples_per_box 
    (hc1 : apples_per_crate = 180)
    (hc2 : number_of_crates = 12)
    (hc3 : rotten_apples = 160)
    (hc4 : number_of_boxes = 100) :
    (remaining_apples apples_per_crate number_of_crates rotten_apples) / number_of_boxes = 20 := 
sorry

end apples_per_box_l88_88902


namespace find_divisor_l88_88040

variable {N : ℤ} (k q : ℤ) {D : ℤ}

theorem find_divisor (h1 : N = 158 * k + 50) (h2 : N = D * q + 13) (h3 : D > 13) (h4 : D < 158) :
  D = 37 :=
by 
  sorry

end find_divisor_l88_88040


namespace parallel_vectors_l88_88753

theorem parallel_vectors {m : ℝ} 
  (h : (2 * m + 1) / 2 = 3 / m): m = 3 / 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l88_88753


namespace roots_quadratic_l88_88943

theorem roots_quadratic (m x₁ x₂ : ℝ) (h : m < 0) (h₁ : x₁ < x₂) (hx : ∀ x, (x^2 - x - 6 = m) ↔ (x = x₁ ∨ x = x₂)) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by {
  sorry
}

end roots_quadratic_l88_88943


namespace andrews_age_l88_88421

-- Define Andrew's age
variable (a g : ℚ)

-- Problem conditions
axiom condition1 : g = 10 * a
axiom condition2 : g - (a + 2) = 57

theorem andrews_age : a = 59 / 9 := 
by
  -- Set the proof steps aside for now
  sorry

end andrews_age_l88_88421


namespace smallest_possible_value_l88_88053

theorem smallest_possible_value (x : ℝ) (hx : 11 = x^2 + 1 / x^2) :
  x + 1 / x = -Real.sqrt 13 :=
by
  sorry

end smallest_possible_value_l88_88053


namespace sequence_term_l88_88345

theorem sequence_term (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) (hn : n > 0)
  (hSn : ∀ n, S n = n^2)
  (hrec : ∀ n, n > 1 → a n = S n - S (n-1)) :
  a n = 2 * n - 1 := by
  -- Base case
  cases n with
  | zero => contradiction  -- n > 0 implies n ≠ 0
  | succ n' =>
    cases n' with
    | zero => sorry  -- When n = 0 + 1 = 1, we need to show a 1 = 2 * 1 - 1 = 1 based on given conditions
    | succ k => sorry -- When n = k + 1, we use the provided recursive relation to prove the statement

end sequence_term_l88_88345


namespace john_recreation_percent_l88_88648

theorem john_recreation_percent (W : ℝ) (P : ℝ) (H1 : 0 ≤ P ∧ P ≤ 1) (H2 : 0 ≤ W) (H3 : 0.15 * W = 0.50 * (P * W)) :
  P = 0.30 :=
by
  sorry

end john_recreation_percent_l88_88648


namespace max_value_f_l88_88848

def f (a x y : ℝ) : ℝ := a * x + y

theorem max_value_f (a : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : |x| + |y| ≤ 1) :
    f a x y ≤ 1 :=
by
  sorry

end max_value_f_l88_88848


namespace average_homework_time_decrease_l88_88669

theorem average_homework_time_decrease (x : ℝ) :
  (100 * (1 - x)^2 = 70) :=
sorry

end average_homework_time_decrease_l88_88669


namespace necessary_but_not_sufficient_l88_88462

-- Definitions used in the conditions
variable (a b : ℝ)

-- The Lean 4 theorem statement for the proof problem
theorem necessary_but_not_sufficient : (a > b - 1) ∧ ¬ (a > b) ↔ a > b := 
sorry

end necessary_but_not_sufficient_l88_88462


namespace factorize1_factorize2_factorize3_l88_88237

-- Proof problem 1: Prove m^2 + 4m + 4 = (m + 2)^2
theorem factorize1 (m : ℝ) : m^2 + 4 * m + 4 = (m + 2)^2 :=
sorry

-- Proof problem 2: Prove a^2 b - 4ab^2 + 3b^3 = b(a-b)(a-3b)
theorem factorize2 (a b : ℝ) : a^2 * b - 4 * a * b^2 + 3 * b^3 = b * (a - b) * (a - 3 * b) :=
sorry

-- Proof problem 3: Prove (x^2 + y^2)^2 - 4x^2 y^2 = (x + y)^2 (x - y)^2
theorem factorize3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

end factorize1_factorize2_factorize3_l88_88237


namespace compound_interest_rate_l88_88397

theorem compound_interest_rate (SI CI : ℝ) (P1 P2 : ℝ) (T1 T2 : ℝ) (R1 : ℝ) (R : ℝ) 
    (H1 : SI = (P1 * R1 * T1) / 100)
    (H2 : CI = 2 * SI)
    (H3 : CI = P2 * ((1 + R/100)^2 - 1))
    (H4 : P1 = 1272)
    (H5 : P2 = 5000)
    (H6 : T1 = 5)
    (H7 : T2 = 2)
    (H8 : R1 = 10) :
  R = 12 :=
by
  sorry

end compound_interest_rate_l88_88397


namespace inequality_proof_l88_88693

variable (a b : ℝ)

theorem inequality_proof (h : a < b) : 1 - a > 1 - b :=
sorry

end inequality_proof_l88_88693


namespace part_a_part_b_l88_88733

def good (p q n : ℕ) : Prop :=
  ∃ x y : ℕ, n = p * x + q * y

def bad (p q n : ℕ) : Prop := 
  ¬ good p q n

theorem part_a (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ A, A = p * q - p - q ∧ ∀ x y, x + y = A → (good p q x ∧ bad p q y) ∨ (bad p q x ∧ good p q y) := by
  sorry

theorem part_b (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ N, N = (p - 1) * (q - 1) / 2 ∧ ∀ n, n < p * q - p - q → bad p q n :=
  sorry

end part_a_part_b_l88_88733


namespace union_of_A_and_B_l88_88637

def A : Set ℤ := {-1, 0, 2}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l88_88637


namespace min_distance_between_M_and_N_l88_88844

noncomputable def f (x : ℝ) := Real.sin x + (1 / 6) * x^3
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_between_M_and_N :
  ∃ (x1 x2 : ℝ), x1 ≥ 0 ∧ x2 ≥ 0 ∧ f x1 = g x2 ∧ (x2 - x1 = 1) :=
sorry

end min_distance_between_M_and_N_l88_88844


namespace sin_angle_calculation_l88_88041

theorem sin_angle_calculation (α : ℝ) (h : α = 240) : Real.sin (150 - α) = -1 :=
by
  rw [h]
  norm_num
  sorry

end sin_angle_calculation_l88_88041


namespace jill_age_l88_88728

theorem jill_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : J = 16 :=
by
  sorry

end jill_age_l88_88728


namespace find_a_b_l88_88546

theorem find_a_b (a b : ℝ)
  (h1 : (0 - a)^2 + (-12 - b)^2 = 36)
  (h2 : (0 - a)^2 + (0 - b)^2 = 36) :
  a = 0 ∧ b = -6 :=
by
  sorry

end find_a_b_l88_88546


namespace smallest_positive_period_and_axis_of_symmetry_l88_88344

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem smallest_positive_period_and_axis_of_symmetry :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ k : ℤ, ∀ x, 2 * x - Real.pi / 4 = k * Real.pi + Real.pi / 2 → x = k * Real.pi / 2 - Real.pi / 8) :=
  sorry

end smallest_positive_period_and_axis_of_symmetry_l88_88344


namespace min_value_a_l88_88219

noncomputable def equation_has_real_solutions (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 9 * x1 - (4 + a) * 3 * x1 + 4 = 0 ∧ 9 * x2 - (4 + a) * 3 * x2 + 4 = 0

theorem min_value_a : ∀ a : ℝ, 
  equation_has_real_solutions a → 
  a ≥ 2 :=
sorry

end min_value_a_l88_88219


namespace fathers_age_after_further_8_years_l88_88015

variable (R F : ℕ)

def age_relation_1 : Prop := F = 4 * R
def age_relation_2 : Prop := F + 8 = (5 * (R + 8)) / 2

theorem fathers_age_after_further_8_years (h1 : age_relation_1 R F) (h2 : age_relation_2 R F) : (F + 16) = 2 * (R + 16) :=
by 
  sorry

end fathers_age_after_further_8_years_l88_88015


namespace volume_of_right_square_prism_l88_88595

theorem volume_of_right_square_prism (length width : ℕ) (H1 : length = 12) (H2 : width = 8) :
    ∃ V, (V = 72 ∨ V = 48) :=
by
  sorry

end volume_of_right_square_prism_l88_88595


namespace integer_solutions_pxy_eq_xy_l88_88286

theorem integer_solutions_pxy_eq_xy (p : ℤ) (hp : Prime p) :
  ∃ x y : ℤ, p * (x + y) = x * y ∧ 
  ((x, y) = (2 * p, 2 * p) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (p + 1, p + p^2) ∨ 
  (x, y) = (p - 1, p - p^2) ∨ 
  (x, y) = (p + p^2, p + 1) ∨ 
  (x, y) = (p - p^2, p - 1)) :=
by
  sorry

end integer_solutions_pxy_eq_xy_l88_88286


namespace function_increasing_iff_m_eq_1_l88_88864

theorem function_increasing_iff_m_eq_1 (m : ℝ) : 
  (m^2 - 4 * m + 4 = 1) ∧ (m^2 - 6 * m + 8 > 0) ↔ m = 1 :=
by {
  sorry
}

end function_increasing_iff_m_eq_1_l88_88864


namespace opposite_of_neg_3_is_3_l88_88296

theorem opposite_of_neg_3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg_3_is_3_l88_88296


namespace sum_dihedral_angles_gt_360_l88_88315

-- Define the structure Tetrahedron
structure Tetrahedron (α : Type*) :=
  (A B C D : α)

-- Define the dihedral angles function
noncomputable def sum_dihedral_angles {α : Type*} (T : Tetrahedron α) : ℝ := 
  -- Placeholder for the actual sum of dihedral angles of T
  sorry

-- Statement of the problem
theorem sum_dihedral_angles_gt_360 {α : Type*} (T : Tetrahedron α) :
  sum_dihedral_angles T > 360 := 
sorry

end sum_dihedral_angles_gt_360_l88_88315


namespace quadratic_inequality_solution_l88_88523

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 2 * x + 1 > 0) ↔ (a > 1) :=
by
  sorry

end quadratic_inequality_solution_l88_88523


namespace arithmetic_mean_q_r_l88_88014

theorem arithmetic_mean_q_r (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) 
  (h3 : r - p = 24) : 
  (q + r) / 2 = 22 := 
by
  sorry

end arithmetic_mean_q_r_l88_88014


namespace bob_more_than_ken_l88_88472

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := by
  -- proof steps to be filled in
  sorry

end bob_more_than_ken_l88_88472


namespace arithmetic_sequence_third_term_l88_88118

theorem arithmetic_sequence_third_term (a d : ℤ) 
  (h20 : a + 19 * d = 17) (h21 : a + 20 * d = 20) : a + 2 * d = -34 := 
sorry

end arithmetic_sequence_third_term_l88_88118


namespace probability_no_correct_letter_for_7_envelopes_l88_88178

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

noncomputable def probability_no_correct_letter (n : ℕ) : ℚ :=
  derangement n / factorial n

theorem probability_no_correct_letter_for_7_envelopes :
  probability_no_correct_letter 7 = 427 / 1160 :=
by sorry

end probability_no_correct_letter_for_7_envelopes_l88_88178


namespace cyclic_quadrilateral_angles_l88_88932

theorem cyclic_quadrilateral_angles (ABCD_cyclic : True) (P_interior : True)
  (x y z t : ℝ) (h1 : x + y + z + t = 360)
  (h2 : x + t = 180) :
  x = 180 - y - z :=
by
  sorry

end cyclic_quadrilateral_angles_l88_88932


namespace bahs_equivalent_to_1500_yahs_l88_88651

-- Definitions from conditions
def bahs := ℕ
def rahs := ℕ
def yahs := ℕ

-- Conversion ratios given in conditions
def ratio_bah_rah : ℚ := 10 / 16
def ratio_rah_yah : ℚ := 9 / 15

-- Given the conditions
def condition1 (b r : ℚ) : Prop := b / r = ratio_bah_rah
def condition2 (r y : ℚ) : Prop := r / y = ratio_rah_yah

-- Goal: proving the question
theorem bahs_equivalent_to_1500_yahs (b : ℚ) (r : ℚ) (y : ℚ)
  (h1 : condition1 b r) (h2 : condition2 r y) : b * (1500 / y) = 562.5
:=
sorry

end bahs_equivalent_to_1500_yahs_l88_88651


namespace triple_f_of_3_l88_88610

def f (x : ℤ) : ℤ := -3 * x + 5

theorem triple_f_of_3 : f (f (f 3)) = -46 := by
  sorry

end triple_f_of_3_l88_88610


namespace y_expression_value_l88_88694

theorem y_expression_value
  (y : ℝ)
  (h : y + 2 / y = 2) :
  y^6 + 3 * y^4 - 4 * y^2 + 2 = 2 := sorry

end y_expression_value_l88_88694


namespace smallest_number_of_three_l88_88916

theorem smallest_number_of_three (a b c : ℕ) (h1 : a + b + c = 78) (h2 : b = 27) (h3 : c = b + 5) :
  a = 19 :=
by
  sorry

end smallest_number_of_three_l88_88916


namespace binomial_coeffs_not_arith_seq_l88_88268

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def are_pos_integer (n : ℕ) : Prop := n > 0

def is_arith_seq (a b c d : ℕ) : Prop := 
  2 * b = a + c ∧ 2 * c = b + d 

theorem binomial_coeffs_not_arith_seq (n r : ℕ) : 
  are_pos_integer n → are_pos_integer r → n ≥ r + 3 → ¬ is_arith_seq (binomial n r) (binomial n (r+1)) (binomial n (r+2)) (binomial n (r+3)) :=
by
  sorry

end binomial_coeffs_not_arith_seq_l88_88268


namespace find_a_and_b_l88_88982

theorem find_a_and_b (a b : ℕ) :
  42 = a * 6 ∧ 72 = 6 * b ∧ 504 = 42 * 12 → (a, b) = (7, 12) :=
by
  sorry

end find_a_and_b_l88_88982


namespace log_base_4_of_8_l88_88390

noncomputable def log_base_change (b a c : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem log_base_4_of_8 : log_base_change 4 8 10 = 3 / 2 :=
by
  have h1 : Real.log 8 = 3 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 8 = 2^3
  have h2 : Real.log 4 = 2 * Real.log 2 := by
    sorry  -- Use properties of logarithms: 4 = 2^2
  have h3 : log_base_change 4 8 10 = (3 * Real.log 2) / (2 * Real.log 2) := by
    rw [log_base_change, h1, h2]
  have h4 : (3 * Real.log 2) / (2 * Real.log 2) = 3 / 2 := by
    sorry  -- Simplify the fraction
  rw [h3, h4]

end log_base_4_of_8_l88_88390


namespace brianna_books_gift_l88_88413

theorem brianna_books_gift (books_per_month : ℕ) (months_per_year : ℕ) (books_bought : ℕ) 
  (borrow_difference : ℕ) (books_reread : ℕ) (total_books_needed : ℕ) : 
  (books_per_month * months_per_year = total_books_needed) →
  ((books_per_month * months_per_year) - books_reread - 
  (books_bought + (books_bought - borrow_difference)) = 
  books_given) →
  books_given = 6 := 
by
  intro h1 h2
  sorry

end brianna_books_gift_l88_88413


namespace calc_neg_half_times_neg_two_pow_l88_88960

theorem calc_neg_half_times_neg_two_pow :
  - (0.5 ^ 20) * ((-2) ^ 26) = -64 := by
  sorry

end calc_neg_half_times_neg_two_pow_l88_88960


namespace international_call_cost_per_minute_l88_88468

theorem international_call_cost_per_minute 
  (local_call_minutes : Nat)
  (international_call_minutes : Nat)
  (local_rate : Nat)
  (total_cost_cents : Nat) 
  (spent_dollars : Nat) 
  (spent_cents : Nat)
  (local_call_cost : Nat)
  (international_call_total_cost : Nat) : 
  local_call_minutes = 45 → 
  international_call_minutes = 31 → 
  local_rate = 5 → 
  total_cost_cents = spent_dollars * 100 → 
  spent_dollars = 10 → 
  local_call_cost = local_call_minutes * local_rate → 
  spent_cents = spent_dollars * 100 → 
  total_cost_cents = spent_cents →  
  international_call_total_cost = total_cost_cents - local_call_cost → 
  international_call_total_cost / international_call_minutes = 25 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end international_call_cost_per_minute_l88_88468


namespace segments_either_disjoint_or_common_point_l88_88066

theorem segments_either_disjoint_or_common_point (n : ℕ) (segments : List (ℝ × ℝ)) 
  (h_len : segments.length = n^2 + 1) : 
  (∃ (disjoint_segments : List (ℝ × ℝ)), disjoint_segments.length ≥ n + 1 ∧ 
    (∀ (s1 s2 : (ℝ × ℝ)), s1 ∈ disjoint_segments → s2 ∈ disjoint_segments 
    → s1 ≠ s2 → ¬ (s1.1 ≤ s2.2 ∧ s2.1 ≤ s1.2))) 
  ∨ 
  (∃ (common_point_segments : List (ℝ × ℝ)), common_point_segments.length ≥ n + 1 ∧ 
    (∃ (p : ℝ), ∀ (s : (ℝ × ℝ)), s ∈ common_point_segments → s.1 ≤ p ∧ p ≤ s.2)) :=
sorry

end segments_either_disjoint_or_common_point_l88_88066


namespace real_part_of_complex_l88_88758

theorem real_part_of_complex (z : ℂ) (h : i * (z + 1) = -3 + 2 * i) : z.re = 1 :=
sorry

end real_part_of_complex_l88_88758


namespace smallest_prime_factor_2379_l88_88872

-- Define the given number
def n : ℕ := 2379

-- Define the condition that 3 is a prime number.
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define the smallest prime factor
def smallest_prime_factor (n p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q, is_prime q → q ∣ n → p ≤ q)

-- The statement that 3 is the smallest prime factor of 2379
theorem smallest_prime_factor_2379 : smallest_prime_factor n 3 :=
sorry

end smallest_prime_factor_2379_l88_88872


namespace least_number_to_multiply_for_multiple_of_112_l88_88273

theorem least_number_to_multiply_for_multiple_of_112 (n : ℕ) : 
  (Nat.lcm 72 112) / 72 = 14 := 
sorry

end least_number_to_multiply_for_multiple_of_112_l88_88273


namespace percent_increase_twice_eq_44_percent_l88_88867

variable (P : ℝ) (x : ℝ)

theorem percent_increase_twice_eq_44_percent (h : P * (1 + x)^2 = P * 1.44) : x = 0.2 :=
by sorry

end percent_increase_twice_eq_44_percent_l88_88867


namespace union_M_N_l88_88271

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_M_N : M ∪ N = {x | -1 < x ∧ x < 3} := 
by 
  sorry

end union_M_N_l88_88271


namespace find_width_of_metallic_sheet_l88_88028

noncomputable def width_of_metallic_sheet (w : ℝ) : Prop :=
  let length := 48
  let square_side := 8
  let new_length := length - 2 * square_side
  let new_width := w - 2 * square_side
  let height := square_side
  let volume := new_length * new_width * height
  volume = 5120

theorem find_width_of_metallic_sheet (w : ℝ) :
  width_of_metallic_sheet w -> w = 36 := 
sorry

end find_width_of_metallic_sheet_l88_88028


namespace divisible_by_a_minus_one_squared_l88_88159

theorem divisible_by_a_minus_one_squared (a n : ℕ) (h : n > 0) :
  (a^(n+1) - n * (a - 1) - a) % (a - 1)^2 = 0 :=
by
  sorry

end divisible_by_a_minus_one_squared_l88_88159


namespace equal_real_roots_of_quadratic_eq_l88_88349

theorem equal_real_roots_of_quadratic_eq {k : ℝ} (h : ∃ x : ℝ, (x^2 + 3 * x - k = 0) ∧ ∀ y : ℝ, (y^2 + 3 * y - k = 0) → y = x) : k = -9 / 4 := 
by 
  sorry

end equal_real_roots_of_quadratic_eq_l88_88349


namespace nonagon_area_l88_88530

noncomputable def area_of_nonagon (r : ℝ) : ℝ :=
  (9 / 2) * r^2 * Real.sin (Real.pi * 40 / 180)

theorem nonagon_area (r : ℝ) : 
  area_of_nonagon r = 2.891 * r^2 :=
by
  sorry

end nonagon_area_l88_88530


namespace jason_tattoos_on_each_leg_l88_88341

-- Define the basic setup
variable (x : ℕ)

-- Define the number of tattoos Jason has on each leg
def tattoos_on_each_leg := x

-- Define the total number of tattoos Jason has
def total_tattoos_jason := 2 + 2 + 2 * x

-- Define the total number of tattoos Adam has
def total_tattoos_adam := 23

-- Define the relation between Adam's and Jason's tattoos
def relation := 2 * total_tattoos_jason + 3 = total_tattoos_adam

-- The proof statement we need to show
theorem jason_tattoos_on_each_leg : tattoos_on_each_leg = 3  :=
by
  sorry

end jason_tattoos_on_each_leg_l88_88341


namespace find_number_l88_88570

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
by
  sorry

end find_number_l88_88570


namespace quadrilateral_perimeter_proof_l88_88258

noncomputable def perimeter_quadrilateral (AB BC CD AD : ℝ) : ℝ :=
  AB + BC + CD + AD

theorem quadrilateral_perimeter_proof
  (AB BC CD AD : ℝ)
  (h1 : AB = 15)
  (h2 : BC = 10)
  (h3 : CD = 6)
  (h4 : AB = AD)
  (h5 : AD = Real.sqrt 181)
  : perimeter_quadrilateral AB BC CD AD = 31 + Real.sqrt 181 := by
  unfold perimeter_quadrilateral
  rw [h1, h2, h3, h5]
  sorry

end quadrilateral_perimeter_proof_l88_88258


namespace number_of_neither_l88_88682

def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def both_drinkers : ℕ := 6

theorem number_of_neither (total_businessmen coffee_drinkers tea_drinkers both_drinkers : ℕ) : 
  coffee_drinkers = 15 ∧ 
  tea_drinkers = 12 ∧ 
  both_drinkers = 6 ∧ 
  total_businessmen = 30 → 
  total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers) = 9 :=
by
  sorry

end number_of_neither_l88_88682


namespace p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l88_88880

variable (x : ℝ)

def p := |x| = x
def q := x^2 + x ≥ 0

theorem p_implies_q : p x → q x :=
by sorry

theorem q_not_implies_p : q x → ¬p x :=
by sorry

theorem p_sufficient_but_not_necessary : (p x → q x) ∧ ¬(q x → p x) :=
by sorry

end p_implies_q_q_not_implies_p_p_sufficient_but_not_necessary_l88_88880


namespace johns_ratio_l88_88136

-- Definitions for initial counts
def initial_pink := 26
def initial_green := 15
def initial_yellow := 24
def initial_total := initial_pink + initial_green + initial_yellow

-- Definitions for Carl's and John's actions
def carl_pink_taken := 4
def john_pink_taken := 6
def remaining_pink := initial_pink - carl_pink_taken - john_pink_taken

-- Definition for remaining hard hats
def total_remaining := 43

-- Compute John's green hat withdrawal
def john_green_taken := (initial_total - carl_pink_taken - john_pink_taken) - total_remaining
def ratio := john_green_taken / john_pink_taken

theorem johns_ratio : ratio = 2 :=
by
  -- Proof details omitted
  sorry

end johns_ratio_l88_88136


namespace perfect_squares_50_to_200_l88_88046

theorem perfect_squares_50_to_200 : 
  ∃ (k : ℕ), k = 7 ∧ ∀ n : ℤ, 50 < n^2 ∧ n^2 < 200 -> (8 ≤ n ∧ n ≤ 14) := 
by
  sorry

end perfect_squares_50_to_200_l88_88046


namespace f_2016_eq_neg1_l88_88882

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_property : ∀ x y : ℝ, f x * f y = f (x + y) + f (x - y)

theorem f_2016_eq_neg1 : f 2016 = -1 := 
by 
  sorry

end f_2016_eq_neg1_l88_88882


namespace rectangle_area_difference_l88_88888

theorem rectangle_area_difference :
  let area (l w : ℝ) := l * w
  let combined_area (l w : ℝ) := 2 * area l w
  combined_area 11 19 - combined_area 9.5 11 = 209 :=
by
  sorry

end rectangle_area_difference_l88_88888


namespace line_through_origin_l88_88734

theorem line_through_origin (x y : ℝ) :
  (∃ x0 y0 : ℝ, 4 * x0 + y0 + 6 = 0 ∧ 3 * (-x0) + (- 5) * y0 + 6 = 0)
  → (x + 6 * y = 0) :=
by
  sorry

end line_through_origin_l88_88734


namespace perimeter_of_monster_is_correct_l88_88618

/-
  The problem is to prove that the perimeter of a shaded sector of a circle
  with radius 2 cm and a central angle of 120 degrees (where the mouth is a chord)
  is equal to (8 * π / 3 + 2 * sqrt 3) cm.
-/

noncomputable def perimeter_of_monster (r : ℝ) (theta_deg : ℝ) : ℝ :=
  let theta_rad := theta_deg * Real.pi / 180
  let chord_length := 2 * r * Real.sin (theta_rad / 2)
  let arc_length := (2 * (2 * Real.pi) * (240 / 360))
  arc_length + chord_length

theorem perimeter_of_monster_is_correct : perimeter_of_monster 2 120 = (8 * Real.pi / 3 + 2 * Real.sqrt 3) :=
by
  sorry

end perimeter_of_monster_is_correct_l88_88618


namespace cost_of_one_book_l88_88790

theorem cost_of_one_book (x : ℝ) : 
  (9 * x = 11) ∧ (13 * x = 15) → x = 1.23 :=
by sorry

end cost_of_one_book_l88_88790


namespace monotonicity_of_f_inequality_f_l88_88346

section
variables {f : ℝ → ℝ}
variables (h_dom : ∀ x, x > 0 → f x > 0)
variables (h_f2 : f 2 = 1)
variables (h_fxy : ∀ x y, f (x * y) = f x + f y)
variables (h_pos : ∀ x, 1 < x → f x > 0)

-- Monotonicity of f(x)
theorem monotonicity_of_f :
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

-- Inequality f(x) + f(x-2) ≤ 3 
theorem inequality_f (x : ℝ) :
  2 < x ∧ x ≤ 4 → f x + f (x - 2) ≤ 3 :=
sorry

end

end monotonicity_of_f_inequality_f_l88_88346


namespace total_shaded_area_l88_88156

/-- 
Given a 6-foot by 12-foot floor tiled with 1-foot by 1-foot tiles,
where each tile has four white quarter circles of radius 1/3 foot at its corners,
prove that the total shaded area of the floor is 72 - 8π square feet.
-/
theorem total_shaded_area :
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  total_shaded_area = 72 - 8 * Real.pi :=
by
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  sorry

end total_shaded_area_l88_88156


namespace min_value_of_expression_l88_88144

theorem min_value_of_expression : 
  ∃ x y : ℝ, (z = x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3) ∧ z = 1 ∧ x = 0 ∧ y = -1 :=
by
  sorry

end min_value_of_expression_l88_88144


namespace mean_temperature_is_correct_l88_88438

-- Defining the list of temperatures
def temperatures : List ℝ := [75, 74, 76, 77, 80, 81, 83, 85, 83, 85]

-- Lean statement asserting the mean temperature is 79.9
theorem mean_temperature_is_correct : temperatures.sum / (temperatures.length: ℝ) = 79.9 := 
by
  sorry

end mean_temperature_is_correct_l88_88438


namespace runway_show_duration_l88_88914

theorem runway_show_duration
  (evening_wear_time : ℝ) (bathing_suits_time : ℝ) (formal_wear_time : ℝ) (casual_wear_time : ℝ)
  (evening_wear_sets : ℕ) (bathing_suits_sets : ℕ) (formal_wear_sets : ℕ) (casual_wear_sets : ℕ)
  (num_models : ℕ) :
  evening_wear_time = 4 → bathing_suits_time = 2 → formal_wear_time = 3 → casual_wear_time = 2.5 →
  evening_wear_sets = 4 → bathing_suits_sets = 2 → formal_wear_sets = 3 → casual_wear_sets = 5 →
  num_models = 10 →
  (evening_wear_time * evening_wear_sets + bathing_suits_time * bathing_suits_sets
   + formal_wear_time * formal_wear_sets + casual_wear_time * casual_wear_sets) * num_models = 415 :=
by
  intros
  sorry

end runway_show_duration_l88_88914


namespace gumballs_difference_l88_88997

variable (x y : ℕ)

def total_gumballs := 16 + 12 + 20 + x + y
def avg_gumballs (T : ℕ) := T / 5

theorem gumballs_difference (h1 : 18 <= avg_gumballs (total_gumballs x y)) 
                            (h2 : avg_gumballs (total_gumballs x y) <= 27) : (87 - 42) = 45 := by
  sorry

end gumballs_difference_l88_88997


namespace find_m_l88_88850

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b : ℝ × ℝ := (-1, -1)
noncomputable def a_minus_b : ℝ × ℝ := (2, 3)
noncomputable def m_a_plus_b (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m - 1)

theorem find_m (m : ℝ) : (a_minus_b.1 * (m_a_plus_b m).1 + a_minus_b.2 * (m_a_plus_b m).2) = 0 → m = 5 / 8 := 
by
  sorry

end find_m_l88_88850


namespace min_value_quadratic_function_l88_88094

def f (a b c x : ℝ) : ℝ := a * (x - b) * (x - c)

theorem min_value_quadratic_function :
  ∃ a b c : ℝ, 
    (1 ≤ a ∧ a < 10) ∧
    (1 ≤ b ∧ b < 10) ∧
    (1 ≤ c ∧ c < 10) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (∀ x : ℝ, f a b c x ≥ -128) :=
sorry

end min_value_quadratic_function_l88_88094


namespace downloaded_data_l88_88557

/-- 
  Mason is trying to download a 880 MB game to his phone. After downloading some amount, his Internet
  connection slows to 3 MB/minute. It will take him 190 more minutes to download the game. Prove that 
  Mason has downloaded 310 MB before his connection slowed down. 
-/
theorem downloaded_data (total_size : ℕ) (speed : ℕ) (time_remaining : ℕ) (remaining_data : ℕ) (downloaded : ℕ) :
  total_size = 880 ∧
  speed = 3 ∧
  time_remaining = 190 ∧
  remaining_data = speed * time_remaining ∧
  downloaded = total_size - remaining_data →
  downloaded = 310 := 
by 
  sorry

end downloaded_data_l88_88557


namespace find_max_value_l88_88113

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x + a

theorem find_max_value (a x : ℝ) (h_min : f 1 a = 1) : 
  ∃ x : ℝ, f (-1/3) 2 = 59/27 :=
by {
  sorry
}

end find_max_value_l88_88113


namespace length_of_second_train_l88_88529

/-- 
  Given:
  * Speed of train 1 is 60 km/hr.
  * Speed of train 2 is 40 km/hr.
  * Length of train 1 is 500 meters.
  * Time to cross each other is 44.99640028797697 seconds.

  Then the length of train 2 is 750 meters.
-/
theorem length_of_second_train (v1 v2 t : ℝ) (d1 L : ℝ) : 
  v1 = 60 ∧
  v2 = 40 ∧
  t = 44.99640028797697 ∧
  d1 = 500 ∧
  L = ((v1 + v2) * (1000 / 3600) * t - d1) →
  L = 750 :=
by sorry

end length_of_second_train_l88_88529


namespace pave_hall_with_stones_l88_88500

def hall_length_m : ℕ := 36
def hall_breadth_m : ℕ := 15
def stone_length_dm : ℕ := 4
def stone_breadth_dm : ℕ := 5

def to_decimeters (m : ℕ) : ℕ := m * 10

def hall_length_dm : ℕ := to_decimeters hall_length_m
def hall_breadth_dm : ℕ := to_decimeters hall_breadth_m

def hall_area_dm2 : ℕ := hall_length_dm * hall_breadth_dm
def stone_area_dm2 : ℕ := stone_length_dm * stone_breadth_dm

def number_of_stones_required : ℕ := hall_area_dm2 / stone_area_dm2

theorem pave_hall_with_stones :
  number_of_stones_required = 2700 :=
sorry

end pave_hall_with_stones_l88_88500


namespace lcm_18_30_is_90_l88_88762

theorem lcm_18_30_is_90 : Nat.lcm 18 30 = 90 := 
by 
  unfold Nat.lcm
  sorry

end lcm_18_30_is_90_l88_88762


namespace find_reciprocal_l88_88294

open Real

theorem find_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^3 + y^3 + 1 / 27 = x * y) : 1 / x = 3 := 
sorry

end find_reciprocal_l88_88294


namespace b_n_geometric_a_n_formula_T_n_sum_less_than_2_l88_88343

section problem

variable {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} {C_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- Given conditions
axiom seq_a (n : ℕ) : a_n 1 = 1
axiom recurrence (n : ℕ) : 2 * a_n (n + 1) - a_n n = (n - 2) / (n * (n + 1) * (n + 2))
axiom seq_b (n : ℕ) : b_n n = a_n n - 1 / (n * (n + 1))

-- Required proofs
theorem b_n_geometric : ∀ n : ℕ, b_n n = (1 / 2) ^ n := sorry
theorem a_n_formula : ∀ n : ℕ, a_n n = (1 / 2) ^ n + 1 / (n * (n + 1)) := sorry
theorem T_n_sum_less_than_2 : ∀ n : ℕ, T_n n < 2 := sorry

end problem

end b_n_geometric_a_n_formula_T_n_sum_less_than_2_l88_88343


namespace find_A_l88_88605

def clubsuit (A B : ℤ) : ℤ := 4 * A + 2 * B + 6

theorem find_A : ∃ A : ℤ, clubsuit A 6 = 70 → A = 13 := 
by
  sorry

end find_A_l88_88605


namespace carl_typing_hours_per_day_l88_88821

theorem carl_typing_hours_per_day (words_per_minute : ℕ) (total_words : ℕ) (days : ℕ) (hours_per_day : ℕ) :
  words_per_minute = 50 →
  total_words = 84000 →
  days = 7 →
  hours_per_day = (total_words / days) / (words_per_minute * 60) →
  hours_per_day = 4 :=
by
  intros h_word_rate h_total_words h_days h_hrs_formula
  rewrite [h_word_rate, h_total_words, h_days] at h_hrs_formula
  exact h_hrs_formula

end carl_typing_hours_per_day_l88_88821


namespace license_plate_palindrome_probability_find_m_plus_n_l88_88800

noncomputable section

open Nat

def is_palindrome {α : Type} (seq : List α) : Prop :=
  seq = seq.reverse

def number_of_three_digit_palindromes : ℕ :=
  10 * 10  -- explanation: 10 choices for the first and last digits, 10 for the middle digit

def total_three_digit_numbers : ℕ :=
  10^3  -- 1000

def prob_three_digit_palindrome : ℚ :=
  number_of_three_digit_palindromes / total_three_digit_numbers

def number_of_three_letter_palindromes : ℕ :=
  26 * 26  -- 26 choices for the first and last letters, 26 for the middle letter

def total_three_letter_combinations : ℕ :=
  26^3  -- 26^3

def prob_three_letter_palindrome : ℚ :=
  number_of_three_letter_palindromes / total_three_letter_combinations

def prob_either_palindrome : ℚ :=
  prob_three_digit_palindrome + prob_three_letter_palindrome - (prob_three_digit_palindrome * prob_three_letter_palindrome)

def m : ℕ := 7
def n : ℕ := 52

theorem license_plate_palindrome_probability :
  prob_either_palindrome = 7 / 52 := sorry

theorem find_m_plus_n :
  m + n = 59 := rfl

end license_plate_palindrome_probability_find_m_plus_n_l88_88800


namespace find_m_l88_88107

theorem find_m (m : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (2, -4) ∧ b = (-3, m) ∧ (‖a‖ * ‖b‖ + (a.1 * b.1 + a.2 * b.2)) = 0) : m = 6 := 
by 
  sorry

end find_m_l88_88107


namespace shaded_fraction_is_four_fifteenths_l88_88528

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ)
  let r := (1/16 : ℚ)
  a / (1 - r)

theorem shaded_fraction_is_four_fifteenths :
  shaded_fraction = (4 / 15 : ℚ) := sorry

end shaded_fraction_is_four_fifteenths_l88_88528


namespace fraction_of_satisfactory_grades_is_3_4_l88_88370

def num_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D" + grades "F"

def satisfactory_grades (grades : String → ℕ) : ℕ := 
  grades "A" + grades "B" + grades "C" + grades "D"

def fraction_satisfactory (grades : String → ℕ) : ℚ := 
  satisfactory_grades grades / num_grades grades

theorem fraction_of_satisfactory_grades_is_3_4 
  (grades : String → ℕ)
  (hA : grades "A" = 5)
  (hB : grades "B" = 4)
  (hC : grades "C" = 3)
  (hD : grades "D" = 3)
  (hF : grades "F" = 5) : 
  fraction_satisfactory grades = (3 : ℚ) / 4 := by
{
  sorry
}

end fraction_of_satisfactory_grades_is_3_4_l88_88370


namespace linear_regression_solution_l88_88069

theorem linear_regression_solution :
  let barx := 5
  let bary := 50
  let sum_xi_squared := 145
  let sum_xiyi := 1380
  let n := 5
  let b := (sum_xiyi - barx * bary) / (sum_xi_squared - n * barx^2)
  let a := bary - b * barx
  let predicted_y := 6.5 * 10 + 17.5
  b = 6.5 ∧ a = 17.5 ∧ predicted_y = 82.5 := 
by
  intros
  sorry

end linear_regression_solution_l88_88069


namespace billy_ate_72_cherries_l88_88473

-- Definitions based on conditions:
def initial_cherries : Nat := 74
def remaining_cherries : Nat := 2

-- Problem: How many cherries did Billy eat?
def cherries_eaten := initial_cherries - remaining_cherries

theorem billy_ate_72_cherries : cherries_eaten = 72 :=
by
  -- proof here
  sorry

end billy_ate_72_cherries_l88_88473


namespace circumference_of_base_of_cone_l88_88051

theorem circumference_of_base_of_cone (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) 
  (h1 : V = 24 * Real.pi) (h2 : h = 6) (h3 : V = (1/3) * Real.pi * r^2 * h) 
  (h4 : r = Real.sqrt 12) : C = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end circumference_of_base_of_cone_l88_88051


namespace difference_of_distances_l88_88601

-- Definition of John's walking distance to school
def John_distance : ℝ := 0.7

-- Definition of Nina's walking distance to school
def Nina_distance : ℝ := 0.4

-- Assertion that the difference in walking distance is 0.3 miles
theorem difference_of_distances : (John_distance - Nina_distance) = 0.3 := 
by 
  sorry

end difference_of_distances_l88_88601


namespace sum_of_coefficients_l88_88065

def P (x : ℝ) : ℝ := 3 * (x^8 - 2 * x^5 + x^3 - 7) - 5 * (x^6 + 3 * x^2 - 6) + 2 * (x^4 - 5)

theorem sum_of_coefficients : P 1 = -19 := by
  sorry

end sum_of_coefficients_l88_88065


namespace find_ratio_l88_88714

variable {x y k x1 x2 y1 y2 : ℝ}

-- Inverse proportionality
def inverse_proportional (x y k : ℝ) : Prop := x * y = k

-- Given conditions
axiom h1 : inverse_proportional x1 y1 k
axiom h2 : inverse_proportional x2 y2 k
axiom h3 : x1 ≠ 0
axiom h4 : x2 ≠ 0
axiom h5 : y1 ≠ 0
axiom h6 : y2 ≠ 0
axiom h7 : x1 / x2 = 3 / 4

theorem find_ratio : y1 / y2 = 4 / 3 :=
by
  sorry

end find_ratio_l88_88714


namespace subset_condition_l88_88103

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x | x + a > 0}

theorem subset_condition (a : ℝ) (h : A ⊆ B a) : a > 0 :=
sorry

end subset_condition_l88_88103


namespace vacation_months_away_l88_88212

theorem vacation_months_away (total_savings : ℕ) (pay_per_check : ℕ) (checks_per_month : ℕ) :
  total_savings = 3000 → pay_per_check = 100 → checks_per_month = 2 → 
  total_savings / pay_per_check / checks_per_month = 15 :=
by 
  intros h1 h2 h3
  sorry

end vacation_months_away_l88_88212


namespace reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l88_88200

theorem reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs
  (a b c h : Real)
  (area_legs : ℝ := (1 / 2) * a * b)
  (area_hypotenuse : ℝ := (1 / 2) * c * h)
  (eq_areas : a * b = c * h)
  (height_eq : h = a * b / c)
  (pythagorean_theorem : c ^ 2 = a ^ 2 + b ^ 2) :
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 := 
by
  sorry

end reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l88_88200


namespace Jacob_fill_tank_in_206_days_l88_88679

noncomputable def tank_capacity : ℕ := 350 * 1000
def rain_collection : ℕ := 500
def river_collection : ℕ := 1200
def daily_collection : ℕ := rain_collection + river_collection
def required_days (C R r : ℕ) : ℕ := (C + (R + r) - 1) / (R + r)

theorem Jacob_fill_tank_in_206_days :
  required_days tank_capacity rain_collection river_collection = 206 :=
by 
  sorry

end Jacob_fill_tank_in_206_days_l88_88679


namespace only_n_equal_1_l88_88659

theorem only_n_equal_1 (n : ℕ) (h : n ≥ 1) : Nat.Prime (9^n - 2^n) ↔ n = 1 := by
  sorry

end only_n_equal_1_l88_88659


namespace arithmetic_sequence_problem_l88_88248

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ A, ∀ n : ℕ, a n = A * (q ^ (n - 1))

theorem arithmetic_sequence_problem
  (q : ℝ) 
  (h1 : q > 1)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h_seq : is_arithmetic_sequence a q) : 
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := 
by 
  sorry

end arithmetic_sequence_problem_l88_88248


namespace triangle_with_angle_ratios_l88_88020

theorem triangle_with_angle_ratios {α β γ : ℝ} (h : α + β + γ = 180 ∧ (α / 2 = β / 3) ∧ (α / 2 = γ / 5)) : (α = 90 ∨ β = 90 ∨ γ = 90) :=
by
  sorry

end triangle_with_angle_ratios_l88_88020


namespace rocky_miles_total_l88_88598

-- Defining the conditions
def m1 : ℕ := 4
def m2 : ℕ := 2 * m1
def m3 : ℕ := 3 * m2

-- The statement to be proven
theorem rocky_miles_total : m1 + m2 + m3 = 36 := by
  sorry

end rocky_miles_total_l88_88598


namespace joey_route_length_l88_88045

-- Definitions
def time_one_way : ℝ := 1
def avg_speed : ℝ := 8
def return_speed : ℝ := 12

-- Theorem to prove
theorem joey_route_length : (∃ D : ℝ, D = 6 ∧ (D / avg_speed = time_one_way + D / return_speed)) :=
sorry

end joey_route_length_l88_88045


namespace complement_of_A_l88_88986

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

theorem complement_of_A :
  U \ A = {x | -3 < x ∧ x < 0} :=
sorry

end complement_of_A_l88_88986


namespace vehicle_A_must_pass_B_before_B_collides_with_C_l88_88825

theorem vehicle_A_must_pass_B_before_B_collides_with_C
  (V_A : ℝ) -- speed of vehicle A in mph
  (V_B : ℝ := 40) -- speed of vehicle B in mph
  (V_C : ℝ := 65) -- speed of vehicle C in mph
  (distance_AB : ℝ := 100) -- distance between A and B in ft
  (distance_BC : ℝ := 250) -- initial distance between B and C in ft
  : (V_A > (100 * 65 - 150 * 40) / 250) :=
by {
  sorry
}

end vehicle_A_must_pass_B_before_B_collides_with_C_l88_88825


namespace volume_Q4_l88_88851

noncomputable def tetrahedron_sequence (n : ℕ) : ℝ :=
  -- Define the sequence recursively
  match n with
  | 0       => 1
  | (n + 1) => tetrahedron_sequence n + (4^n * (1 / 27)^(n + 1))

theorem volume_Q4 : tetrahedron_sequence 4 = 1.173832 :=
by
  sorry

end volume_Q4_l88_88851


namespace root_expression_value_l88_88574

theorem root_expression_value (p m n : ℝ) 
  (h1 : m^2 + (p - 2) * m + 1 = 0) 
  (h2 : n^2 + (p - 2) * n + 1 = 0) : 
  (m^2 + p * m + 1) * (n^2 + p * n + 1) - 2 = 2 :=
by
  sorry

end root_expression_value_l88_88574


namespace median_of_consecutive_integers_l88_88447

theorem median_of_consecutive_integers (n : ℕ) (S : ℤ) (h1 : n = 35) (h2 : S = 1225) : 
  n % 2 = 1 → S / n = 35 := 
sorry

end median_of_consecutive_integers_l88_88447


namespace max_k_for_3_pow_11_as_sum_of_consec_integers_l88_88841

theorem max_k_for_3_pow_11_as_sum_of_consec_integers :
  ∃ k n : ℕ, (3^11 = k * (2 * n + k + 1) / 2) ∧ (k = 486) :=
by
  sorry

end max_k_for_3_pow_11_as_sum_of_consec_integers_l88_88841


namespace rectangle_length_l88_88269

theorem rectangle_length {b l : ℝ} (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) : l = 18 := by
    sorry

end rectangle_length_l88_88269


namespace units_digit_of_expression_l88_88006

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_expression : units_digit (7 * 18 * 1978 - 7^4) = 7 := by
  sorry

end units_digit_of_expression_l88_88006


namespace distance_between_first_and_last_tree_l88_88290

theorem distance_between_first_and_last_tree
  (n : ℕ)
  (trees : ℕ)
  (dist_between_first_and_fourth : ℕ)
  (eq_dist : ℕ):
  trees = 6 ∧ dist_between_first_and_fourth = 60 ∧ eq_dist = dist_between_first_and_fourth / 3 ∧ n = (trees - 1) * eq_dist → n = 100 :=
by
  intro h
  sorry

end distance_between_first_and_last_tree_l88_88290


namespace hyungjun_initial_paint_count_l88_88774

theorem hyungjun_initial_paint_count (X : ℝ) (h1 : X / 2 - (X / 6 + 5) = 5) : X = 30 :=
sorry

end hyungjun_initial_paint_count_l88_88774


namespace range_of_p_l88_88295

-- Definitions of A and B
def A (p : ℝ) := {x : ℝ | x^2 + (p + 2) * x + 1 = 0}
def B := {x : ℝ | x > 0}

-- Condition of the problem: A ∩ B = ∅
def condition (p : ℝ) := ∀ x ∈ A p, x ∉ B

-- The statement to prove: p > -4
theorem range_of_p (p : ℝ) : condition p → p > -4 :=
by
  intro h
  sorry

end range_of_p_l88_88295


namespace A_inter_complement_RB_eq_l88_88525

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x^2)}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

def complement_RB : Set ℝ := {x | x ≥ 1}

theorem A_inter_complement_RB_eq : A ∩ complement_RB = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end A_inter_complement_RB_eq_l88_88525


namespace line_divides_circle_1_3_l88_88117

noncomputable def circle_equidistant_from_origin : Prop := 
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, ((x-1)^2 + (y-1)^2 = 2) → 
                     (l 0 = 0 ∧ (l x = l y) ∧ 
                     ((x = 0) ∨ (y = 0)))

theorem line_divides_circle_1_3 (x y : ℝ) : 
  (x - 1)^2 + (y - 1)^2 = 2 → 
  (x = 0 ∨ y = 0) :=
by
  sorry

end line_divides_circle_1_3_l88_88117


namespace divide_rope_length_l88_88725

-- Definitions of variables based on the problem conditions
def rope_length : ℚ := 8 / 15
def num_parts : ℕ := 3

-- Theorem statement
theorem divide_rope_length :
  (1 / num_parts = (1 : ℚ) / 3) ∧ (rope_length * (1 / num_parts) = 8 / 45) :=
by
  sorry

end divide_rope_length_l88_88725


namespace train_B_departure_time_l88_88646

def distance : ℕ := 65
def speed_A : ℕ := 20
def speed_B : ℕ := 25
def departure_A := 7
def meeting_time := 9

theorem train_B_departure_time : ∀ (d : ℕ) (vA : ℕ) (vB : ℕ) (tA : ℕ) (m : ℕ), 
  d = 65 → vA = 20 → vB = 25 → tA = 7 → m = 9 → ((9 - (m - tA + (d - (2 * vA)) / vB)) = 1) → 
  8 = ((9 - (meeting_time - departure_A + (distance - (2 * speed_A)) / speed_B))) := 
  by {
    sorry
  }

end train_B_departure_time_l88_88646


namespace find_a_if_odd_l88_88321

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 1) * (x + a)

theorem find_a_if_odd (a : ℝ) : (∀ x : ℝ, f (-x) a = -f x a) → a = 0 := by
  intro h
  have h0 : f 0 a = 0 := by
    simp [f]
    specialize h 0
    simp [f] at h
    exact h
  sorry

end find_a_if_odd_l88_88321


namespace m_eq_n_is_necessary_but_not_sufficient_l88_88507

noncomputable def circle_condition (m n : ℝ) : Prop :=
  m = n ∧ m > 0

theorem m_eq_n_is_necessary_but_not_sufficient 
  (m n : ℝ) :
  (circle_condition m n → mx^2 + ny^2 = 3 → False) ∧
  (mx^2 + ny^2 = 3 → circle_condition m n) :=
by 
  sorry

end m_eq_n_is_necessary_but_not_sufficient_l88_88507


namespace choose_5_from_12_l88_88858

theorem choose_5_from_12 : Nat.choose 12 5 = 792 := by
  sorry

end choose_5_from_12_l88_88858


namespace slow_speed_distance_l88_88741

theorem slow_speed_distance (D : ℝ) (h : (D + 20) / 14 = D / 10) : D = 50 := by
  sorry

end slow_speed_distance_l88_88741


namespace ratio_D_E_equal_l88_88149

variable (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ)

def mary_story_conditions (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ) : Prop :=
  total_characters = 60 ∧
  initial_A = 1 / 2 * total_characters ∧
  initial_C = 1 / 2 * initial_A ∧
  initial_D + initial_E = total_characters - (initial_A + initial_C)

theorem ratio_D_E_equal (total_characters initial_A initial_C initial_D initial_E : ℕ) :
  mary_story_conditions total_characters initial_A initial_C initial_D initial_E →
  initial_D = initial_E :=
sorry

end ratio_D_E_equal_l88_88149


namespace total_amount_spent_l88_88635

theorem total_amount_spent (T : ℝ) (h1 : 5000 + 200 + 0.30 * T = T) : 
  T = 7428.57 :=
by
  sorry

end total_amount_spent_l88_88635


namespace brad_siblings_product_l88_88974

theorem brad_siblings_product (S B : ℕ) (hS : S = 5) (hB : B = 7) : S * B = 35 :=
by
  have : S = 5 := hS
  have : B = 7 := hB
  sorry

end brad_siblings_product_l88_88974


namespace range_of_a_l88_88683

theorem range_of_a {a : ℝ} 
  (hA : ∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))
  (hB : ∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))
  (hC : (a^2 + 1) / (2 * a) > 0)
  (hOnlyOneFalse : (¬(∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬(∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬((a^2 + 1) / (2 * a) > 0))):
  0 < a ∧ a < 1 := 
sorry

end range_of_a_l88_88683


namespace curve_C2_equation_l88_88437

theorem curve_C2_equation (x y : ℝ) :
  (∀ x, y = 2 * Real.sin (2 * x + π / 3) → 
    y = 2 * Real.sin (4 * (( x - π / 6) / 2))) := 
  sorry

end curve_C2_equation_l88_88437


namespace intersection_points_count_l88_88334

theorem intersection_points_count : 
  ∃ n : ℕ, n = 2 ∧
  (∀ x ∈ (Set.Icc 0 (2 * Real.pi)), (1 + Real.sin x = 3 / 2) → n = 2) :=
sorry

end intersection_points_count_l88_88334


namespace value_of_P_2017_l88_88658

theorem value_of_P_2017 (a b c : ℝ) (h_distinct: a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c)
    (p : ℝ → ℝ) :
    (∀ x, p x = (c * (x - a) * (x - b) / ((c - a) * (c - b))) + (a * (x - b) * (x - c) / ((a - b) * (a - c))) + (b * (x - c) * (x - a) / ((b - c) * (b - a))) + 1) →
    p 2017 = 2 :=
sorry

end value_of_P_2017_l88_88658


namespace remainder_of_polynomial_division_is_88_l88_88555

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem remainder_of_polynomial_division_is_88 :
  p 2 = 88 :=
by
  sorry

end remainder_of_polynomial_division_is_88_l88_88555


namespace inequality_problem_l88_88150

-- Define the conditions and the problem statement
theorem inequality_problem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_problem_l88_88150


namespace linear_function_quadrants_l88_88339

theorem linear_function_quadrants (k b : ℝ) 
  (h1 : k < 0)
  (h2 : b < 0) 
  : k * b > 0 := 
sorry

end linear_function_quadrants_l88_88339


namespace third_factorial_is_7_l88_88587

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Problem conditions
def b : ℕ := 9
def factorial_b_minus_2 : ℕ := factorial (b - 2)
def factorial_b_plus_1 : ℕ := factorial (b + 1)
def GCD_value : ℕ := Nat.gcd (Nat.gcd factorial_b_minus_2 factorial_b_plus_1) (factorial 7)

-- Theorem statement
theorem third_factorial_is_7 :
  Nat.gcd (Nat.gcd (factorial (b - 2)) (factorial (b + 1))) (factorial 7) = 5040 →
  ∃ k : ℕ, factorial k = 5040 ∧ k = 7 :=
by
  sorry

end third_factorial_is_7_l88_88587


namespace inf_solutions_integers_l88_88636

theorem inf_solutions_integers (x y z : ℕ) : ∃ (n : ℕ), ∀ n > 0, (x = 2^(32 + 72 * n)) ∧ (y = 2^(28 + 63 * n)) ∧ (z = 2^(25 + 56 * n)) → x^7 + y^8 = z^9 :=
by {
  sorry
}

end inf_solutions_integers_l88_88636


namespace cars_equilibrium_l88_88387

variable (days : ℕ) -- number of days after which we need the condition to hold
variable (carsA_init carsB_init carsA_to_B carsB_to_A : ℕ) -- initial conditions and parameters

theorem cars_equilibrium :
  let cars_total := 192 + 48
  let carsA := carsA_init + (carsB_to_A - carsA_to_B) * days
  let carsB := carsB_init + (carsA_to_B - carsB_to_A) * days
  carsA_init = 192 -> carsB_init = 48 ->
  carsA_to_B = 21 -> carsB_to_A = 24 ->
  cars_total = 192 + 48 ->
  days = 6 ->
  cars_total = carsA + carsB -> carsA = 7 * carsB :=
by
  intros
  sorry

end cars_equilibrium_l88_88387


namespace mark_score_is_46_l88_88745

theorem mark_score_is_46 (highest_score : ℕ) (range: ℕ) (mark_score : ℕ) :
  highest_score = 98 →
  range = 75 →
  (mark_score = 2 * (highest_score - range)) →
  mark_score = 46 := by
  intros
  sorry

end mark_score_is_46_l88_88745


namespace find_mangoes_l88_88613

def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 1165
def cost_per_kg_of_mangoes : ℕ := 55

theorem find_mangoes (m : ℕ) : cost_of_grapes + m * cost_per_kg_of_mangoes = total_amount_paid → m = 11 :=
by
  sorry

end find_mangoes_l88_88613


namespace total_area_of_figure_l88_88859

theorem total_area_of_figure :
  let h := 7
  let w1 := 6
  let h1 := 2
  let h2 := 3
  let h3 := 1
  let w2 := 5
  let a1 := h * w1
  let a2 := (h - h1) * (11 - 7)
  let a3 := (h - h1 - h2) * (11 - 7)
  let a4 := (15 - 11) * h3
  a1 + a2 + a3 + a4 = 74 :=
by
  sorry

end total_area_of_figure_l88_88859


namespace sorting_five_rounds_l88_88719

def direct_sorting_method (l : List ℕ) : List ℕ := sorry

theorem sorting_five_rounds (initial_seq : List ℕ) :
  initial_seq = [49, 38, 65, 97, 76, 13, 27] →
  (direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method) initial_seq = [97, 76, 65, 49, 38, 13, 27] :=
by
  intros h
  sorry

end sorting_five_rounds_l88_88719


namespace find_value_of_expression_l88_88086

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry

def condition1 : Prop := x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 + 11 * x6 = 2
def condition2 : Prop := 3 * x1 + 5 * x2 + 7 * x3 + 9 * x4 + 11 * x5 + 13 * x6 = 15
def condition3 : Prop := 5 * x1 + 7 * x2 + 9 * x3 + 11 * x4 + 13 * x5 + 15 * x6 = 52

theorem find_value_of_expression : condition1 → condition2 → condition3 → (7 * x1 + 9 * x2 + 11 * x3 + 13 * x4 + 15 * x5 + 17 * x6 = 65) :=
by
  intros h1 h2 h3
  sorry

end find_value_of_expression_l88_88086


namespace min_value_frac_sum_l88_88031

open Real

theorem min_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
    ∃ (z : ℝ), z = 1 + (sqrt 3) / 2 ∧ 
    (∀ t, (t > 0 → ∃ (u : ℝ), u > 0 ∧ t + u = 4 → ∀ t' (h : t' = (1 / t) + (3 / u)), t' ≥ z)) :=
by sorry

end min_value_frac_sum_l88_88031


namespace josephine_milk_containers_l88_88291

theorem josephine_milk_containers :
  3 * 2 + 2 * 0.75 + 5 * x = 10 → x = 0.5 :=
by
  intro h
  sorry

end josephine_milk_containers_l88_88291


namespace find_intersection_find_range_of_a_l88_88881

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | x < -2 ∨ (3 < x ∧ x < 4) }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 5 }

-- Proof Problem 1: Prove the intersection A ∩ B
theorem find_intersection : (A ∩ B) = { x : ℝ | 3 < x ∧ x ≤ 5 } := by
  sorry

-- Define the set C and the condition B ∩ C = B
def C (a : ℝ) : Set ℝ := { x : ℝ | x ≥ a }
def condition (a : ℝ) : Prop := B ∩ C a = B

-- Proof Problem 2: Find the range of a
theorem find_range_of_a : ∀ a : ℝ, condition a → a ≤ -3 := by
  sorry

end find_intersection_find_range_of_a_l88_88881


namespace x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l88_88325

variable {x y : ℝ}

theorem x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 - y^3 = 176 * Real.sqrt 13 := 
by
  sorry

end x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l88_88325


namespace find_B_squared_l88_88622

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 85 / x

theorem find_B_squared :
  let x1 := (Real.sqrt 31 + Real.sqrt 371) / 2
  let x2 := (Real.sqrt 31 - Real.sqrt 371) / 2
  let B := |x1| + |x2|
  B^2 = 371 :=
by
  sorry

end find_B_squared_l88_88622


namespace quadratic_eq_has_nonzero_root_l88_88906

theorem quadratic_eq_has_nonzero_root (b c : ℝ) (h : c ≠ 0) (h_eq : c^2 + b * c + c = 0) : b + c = -1 :=
sorry

end quadratic_eq_has_nonzero_root_l88_88906


namespace blue_length_is_2_l88_88132

-- Define the lengths of the parts
def total_length : ℝ := 4
def purple_length : ℝ := 1.5
def black_length : ℝ := 0.5

-- Define the length of the blue part with the given conditions
def blue_length : ℝ := total_length - (purple_length + black_length)

-- State the theorem we need to prove
theorem blue_length_is_2 : blue_length = 2 :=
by 
  sorry

end blue_length_is_2_l88_88132


namespace complement_union_A_B_l88_88522

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 5}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def U : Set ℝ := A ∪ B
def R : Set ℝ := univ

theorem complement_union_A_B : (R \ U) = {x | -2 < x ∧ x ≤ -1} :=
by
  sorry

end complement_union_A_B_l88_88522


namespace power_sum_l88_88900

theorem power_sum
: (-2)^(2005) + (-2)^(2006) = 2^(2005) := by
  sorry

end power_sum_l88_88900


namespace max_crystalline_polyhedron_volume_l88_88125

theorem max_crystalline_polyhedron_volume (n : ℕ) (R : ℝ) (h_n : n > 1) :
  ∃ V : ℝ, 
    V = (32 / 81) * (n - 1) * (R ^ 3) * Real.sin (2 * Real.pi / (n - 1)) :=
sorry

end max_crystalline_polyhedron_volume_l88_88125


namespace plan_y_cheaper_than_plan_x_l88_88232

def cost_plan_x (z : ℕ) : ℕ := 15 * z

def cost_plan_y (z : ℕ) : ℕ :=
  if z > 500 then 3000 + 7 * z - 1000 else 3000 + 7 * z

theorem plan_y_cheaper_than_plan_x (z : ℕ) (h : z > 500) : cost_plan_y z < cost_plan_x z :=
by
  sorry

end plan_y_cheaper_than_plan_x_l88_88232


namespace length_of_third_side_l88_88696

-- Define the properties and setup for the problem
variables {a b : ℝ} (h1 : a = 4) (h2 : b = 8)

-- Define the condition for an isosceles triangle
def isosceles_triangle (x y z : ℝ) : Prop :=
  (x = y ∧ x ≠ z) ∨ (x = z ∧ x ≠ y) ∨ (y = z ∧ y ≠ x)

-- Define the condition for a valid triangle
def valid_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- State the theorem to be proved
theorem length_of_third_side (c : ℝ) (h : isosceles_triangle a b c ∧ valid_triangle a b c) : c = 8 :=
sorry

end length_of_third_side_l88_88696


namespace inequality_solution_l88_88656

theorem inequality_solution (x : ℚ) (hx : x = 3 ∨ x = 2 ∨ x = 1 ∨ x = 0) : 
  (1 / 3) - (x / 3) < -(1 / 2) → x = 3 :=
by
  sorry

end inequality_solution_l88_88656


namespace necessarily_negative_l88_88444

theorem necessarily_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := 
sorry

end necessarily_negative_l88_88444


namespace find_x2_plus_y2_l88_88423

open Real

theorem find_x2_plus_y2 (x y : ℝ) 
  (h1 : (x + y) ^ 4 + (x - y) ^ 4 = 4112)
  (h2 : x ^ 2 - y ^ 2 = 16) :
  x ^ 2 + y ^ 2 = 34 := 
sorry

end find_x2_plus_y2_l88_88423


namespace find_largest_integer_l88_88374

theorem find_largest_integer (x : ℤ) (hx1 : x < 100) (hx2 : x % 7 = 4) : x = 95 :=
sorry

end find_largest_integer_l88_88374


namespace B_completes_remaining_work_in_2_days_l88_88263

theorem B_completes_remaining_work_in_2_days 
  (A_work_rate : ℝ) (B_work_rate : ℝ) (total_work : ℝ) 
  (A_days_to_complete : A_work_rate = 1 / 2) 
  (B_days_to_complete : B_work_rate = 1 / 6) 
  (combined_work_1_day : A_work_rate + B_work_rate = 2 / 3) : 
  (total_work - (A_work_rate + B_work_rate)) / B_work_rate = 2 := 
by
  sorry

end B_completes_remaining_work_in_2_days_l88_88263


namespace domain_of_sqrt_fun_l88_88316

theorem domain_of_sqrt_fun : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 7 → 7 + 6 * x - x^2 ≥ 0) :=
sorry

end domain_of_sqrt_fun_l88_88316


namespace number_of_cats_l88_88743

def cats_on_ship (C S : ℕ) : Prop :=
  (C + S + 2 = 16) ∧ (4 * C + 2 * S + 3 = 45)

theorem number_of_cats (C S : ℕ) (h : cats_on_ship C S) : C = 7 :=
by
  sorry

end number_of_cats_l88_88743


namespace phil_baseball_cards_left_l88_88816

-- Step a): Define the conditions
def packs_week := 20
def weeks_year := 52
def lost_factor := 1 / 2

-- Step c): Establish the theorem statement
theorem phil_baseball_cards_left : 
  (packs_week * weeks_year * (1 - lost_factor) = 520) := 
  by
    -- proof steps will come here
    sorry

end phil_baseball_cards_left_l88_88816


namespace complement_of_intersection_l88_88721

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {y | ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ y = -x^2}

theorem complement_of_intersection :
  (Set.compl (A ∩ B) = {x | x < -2 ∨ x > 0 }) :=
by
  sorry

end complement_of_intersection_l88_88721


namespace num_divisible_by_33_l88_88501

theorem num_divisible_by_33 : ∀ (x y : ℕ), 
  (0 ≤ x ∧ x ≤ 9) → (0 ≤ y ∧ y ≤ 9) →
  (19 + x + y) % 3 = 0 →
  (x - y + 1) % 11 = 0 →
  ∃! (n : ℕ), (20070002008 * 100 + x * 10 + y) = n ∧ n % 33 = 0 :=
by
  intros x y hx hy h3 h11
  sorry

end num_divisible_by_33_l88_88501


namespace area_diminished_by_64_percent_l88_88820

/-- Given a rectangular field where both the length and width are diminished by 40%, 
    prove that the area is diminished by 64%. -/
theorem area_diminished_by_64_percent (L W : ℝ) :
  let L' := 0.6 * L
  let W' := 0.6 * W
  let A := L * W
  let A' := L' * W'
  (A - A') / A * 100 = 64 :=
by
  sorry

end area_diminished_by_64_percent_l88_88820


namespace cubic_sum_l88_88080

theorem cubic_sum (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 :=
by 
  sorry

end cubic_sum_l88_88080


namespace initial_amount_A_correct_l88_88545

noncomputable def initial_amount_A :=
  let a := 21
  let b := 5
  let c := 9

  -- After A gives B and C
  let b_after_A := b + 5
  let c_after_A := c + 9
  let a_after_A := a - (5 + 9)

  -- After B gives A and C
  let a_after_B := a_after_A + (a_after_A / 2)
  let c_after_B := c_after_A + (c_after_A / 2)
  let b_after_B := b_after_A - (a_after_A / 2 + c_after_A / 2)

  -- After C gives A and B
  let a_final := a_after_B + 3 * a_after_B
  let b_final := b_after_B + 3 * b_after_B
  let c_final := c_after_B - (3 * a_final + b_final)

  (a_final = 24) ∧ (b_final = 16) ∧ (c_final = 8)

theorem initial_amount_A_correct : initial_amount_A := 
by
  -- Skipping proof details
  sorry

end initial_amount_A_correct_l88_88545


namespace find_value_of_a_l88_88631

theorem find_value_of_a (a : ℝ) (h : a^3 = 21 * 25 * 315 * 7) : a = 105 := by
  sorry

end find_value_of_a_l88_88631


namespace find_total_photos_l88_88261

noncomputable def total_photos (T : ℕ) (Paul Tim Tom : ℕ) : Prop :=
  Tim = T - 100 ∧ Paul = Tim + 10 ∧ Tom = 38 ∧ Tom + Tim + Paul = T

theorem find_total_photos : ∃ T, total_photos T (T - 90) (T - 100) 38 :=
sorry

end find_total_photos_l88_88261


namespace measure_weights_l88_88563

theorem measure_weights (w1 w3 w7 : Nat) (h1 : w1 = 1) (h3 : w3 = 3) (h7 : w7 = 7) :
  ∃ s : Finset Nat, s.card = 7 ∧ 
    (1 ∈ s) ∧ (3 ∈ s) ∧ (7 ∈ s) ∧
    (4 ∈ s) ∧ (8 ∈ s) ∧ (10 ∈ s) ∧ 
    (11 ∈ s) := 
by
  sorry

end measure_weights_l88_88563


namespace rightmost_four_digits_of_7_pow_2045_l88_88538

theorem rightmost_four_digits_of_7_pow_2045 : (7^2045 % 10000) = 6807 :=
by
  sorry

end rightmost_four_digits_of_7_pow_2045_l88_88538


namespace find_8th_result_l88_88893

theorem find_8th_result 
  (S_17 : ℕ := 17 * 24) 
  (S_7 : ℕ := 7 * 18) 
  (S_5_1 : ℕ := 5 * 23) 
  (S_5_2 : ℕ := 5 * 32) : 
  S_17 - S_7 - S_5_1 - S_5_2 = 7 := 
by
  sorry

end find_8th_result_l88_88893


namespace harmony_numbers_with_first_digit_2_count_l88_88607

def is_harmony_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (1000 ≤ n ∧ n < 10000) ∧ (a + b + c + d = 6)

noncomputable def count_harmony_numbers_with_first_digit_2 : ℕ :=
  Nat.card { n : ℕ // is_harmony_number n ∧ n / 1000 = 2 }

theorem harmony_numbers_with_first_digit_2_count :
  count_harmony_numbers_with_first_digit_2 = 15 :=
sorry

end harmony_numbers_with_first_digit_2_count_l88_88607


namespace relationship_correct_l88_88792

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem relationship_correct (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) :
  log_base a b < a^b ∧ a^b < b^a :=
by sorry

end relationship_correct_l88_88792


namespace tory_needs_to_raise_more_l88_88408

variable (goal : ℕ) (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
variable (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ)

def remainingAmount (goal : ℕ) 
                    (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
                    (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ) : ℕ :=
  let profitFromChocolateChip := soldChocolateChip * pricePerChocolateChip
  let profitFromOatmealRaisin := soldOatmealRaisin * pricePerOatmealRaisin
  let profitFromSugarCookie := soldSugarCookie * pricePerSugarCookie
  let totalProfit := profitFromChocolateChip + profitFromOatmealRaisin + profitFromSugarCookie
  goal - totalProfit

theorem tory_needs_to_raise_more : 
  remainingAmount 250 6 5 4 5 10 15 = 110 :=
by
  -- Proof omitted 
  sorry

end tory_needs_to_raise_more_l88_88408


namespace fraction_in_classroom_l88_88431

theorem fraction_in_classroom (total_students absent_fraction canteen_students present_students class_students : ℕ) 
  (h_total : total_students = 40)
  (h_absent_fraction : absent_fraction = 1 / 10)
  (h_canteen_students : canteen_students = 9)
  (h_absent_students : absent_fraction * total_students = 4)
  (h_present_students : present_students = total_students - absent_fraction * total_students)
  (h_class_students : class_students = present_students - canteen_students) :
  class_students / present_students = 3 / 4 := 
by {
  sorry
}

end fraction_in_classroom_l88_88431


namespace triangle_inequality_a2_a3_a4_l88_88332

variables {a1 a2 a3 a4 d : ℝ}

def is_arithmetic_sequence (a1 a2 a3 a4 : ℝ) (d : ℝ) : Prop :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℝ) : Prop :=
  0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4

theorem triangle_inequality_a2_a3_a4 (h1: positive_terms a1 a2 a3 a4)
  (h2: is_arithmetic_sequence a1 a2 a3 a4 d) (h3: d > 0) :
  (a2 + a3 > a4) ∧ (a2 + a4 > a3) ∧ (a3 + a4 > a2) :=
sorry

end triangle_inequality_a2_a3_a4_l88_88332


namespace find_a_l88_88087

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l88_88087


namespace clients_number_l88_88614

theorem clients_number (C : ℕ) (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ)
  (h1 : total_cars = 12)
  (h2 : cars_per_client = 4)
  (h3 : selections_per_car = 3)
  (h4 : C * cars_per_client = total_cars * selections_per_car) : C = 9 :=
by sorry

end clients_number_l88_88614


namespace outfit_choices_l88_88425

theorem outfit_choices (tops pants : ℕ) (TopsCount : tops = 4) (PantsCount : pants = 3) :
  tops * pants = 12 := by
  sorry

end outfit_choices_l88_88425


namespace derivative_at_neg_one_l88_88355

def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

theorem derivative_at_neg_one : deriv f (-1) = -1 :=
by
  -- definition of the function
  -- proof of the statement
  sorry

end derivative_at_neg_one_l88_88355


namespace problem_l88_88936

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x + 2
noncomputable def f' (a x : ℝ) : ℝ := a * (Real.log x + 1) + 1
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x - x^2 - (a + 2) * x + a

theorem problem (a x : ℝ) (h : 1 ≤ x) (ha : 0 < a) : f' a x < x^2 + (a + 2) * x + 1 :=
by
  sorry

end problem_l88_88936


namespace jars_left_when_boxes_full_l88_88913

-- Conditions
def jars_in_first_set_of_boxes : Nat := 12 * 10
def jars_in_second_set_of_boxes : Nat := 10 * 30
def total_jars : Nat := 500

-- Question (equivalent proof problem)
theorem jars_left_when_boxes_full : total_jars - (jars_in_first_set_of_boxes + jars_in_second_set_of_boxes) = 80 := 
by
  sorry

end jars_left_when_boxes_full_l88_88913


namespace jessica_needs_stamps_l88_88432

-- Define the weights and conditions
def weight_of_paper := 1 / 5
def total_papers := 8
def weight_of_envelope := 2 / 5
def stamps_per_ounce := 1

-- Calculate the total weight and determine the number of stamps needed
theorem jessica_needs_stamps : 
  total_papers * weight_of_paper + weight_of_envelope = 2 :=
by
  sorry

end jessica_needs_stamps_l88_88432


namespace fraction_never_simplifiable_l88_88634

theorem fraction_never_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_never_simplifiable_l88_88634


namespace solve_y_determinant_l88_88904

theorem solve_y_determinant (b y : ℝ) (hb : b ≠ 0) :
  Matrix.det ![
    ![y + b, y, y], 
    ![y, y + b, y], 
    ![y, y, y + b]
  ] = 0 ↔ y = -b / 3 :=
by
  sorry

end solve_y_determinant_l88_88904


namespace arithmetic_sequence_sum_is_18_l88_88135

variable (a : ℕ → ℕ)

theorem arithmetic_sequence_sum_is_18
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 18 := 
sorry

end arithmetic_sequence_sum_is_18_l88_88135


namespace hadley_total_distance_l88_88254

def distance_to_grocery := 2
def distance_to_pet_store := 2 - 1
def distance_back_home := 4 - 1

theorem hadley_total_distance : distance_to_grocery + distance_to_pet_store + distance_back_home = 6 :=
by
  -- Proof is omitted.
  sorry

end hadley_total_distance_l88_88254


namespace length_AE_l88_88718

structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 4⟩
def B : Point := ⟨7, 0⟩
def C : Point := ⟨5, 3⟩
def D : Point := ⟨3, 0⟩

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt (((Q.x - P.x : ℝ) ^ 2) + ((Q.y - P.y : ℝ) ^ 2))

noncomputable def AE_length : ℝ :=
  (5 * (dist A B)) / 9

theorem length_AE :
  ∃ E : Point, AE_length = (5 * Real.sqrt 65) / 9 := by
  sorry

end length_AE_l88_88718


namespace average_percent_score_l88_88019

def num_students : ℕ := 180

def score_distrib : List (ℕ × ℕ) :=
[(95, 12), (85, 30), (75, 50), (65, 45), (55, 30), (45, 13)]

noncomputable def total_score : ℕ :=
(95 * 12) + (85 * 30) + (75 * 50) + (65 * 45) + (55 * 30) + (45 * 13)

noncomputable def average_score : ℕ :=
total_score / num_students

theorem average_percent_score : average_score = 70 :=
by 
  -- Here you would provide the proof, but for now we will leave it as:
  sorry

end average_percent_score_l88_88019


namespace misha_problem_l88_88424

theorem misha_problem (N : ℕ) (h : ∀ a, a ∈ {a | a > 1 → ∃ b > 0, b ∈ {b' | b' < a ∧ a % b' = 0}}) :
  (∀ t : ℕ, (t > 1) → (1 / t ^ 2) < (1 / t * (t - 1))) →
  (∃ (n : ℕ), n = 1) → (N = 1 ↔ ∃ (k : ℕ), k = N^2) :=
by
  sorry

end misha_problem_l88_88424


namespace greatest_k_dividing_n_l88_88815

noncomputable def num_divisors (n : ℕ) : ℕ :=
  n.divisors.card

theorem greatest_k_dividing_n (n : ℕ) (h_pos : n > 0)
  (h_n_divisors : num_divisors n = 120)
  (h_5n_divisors : num_divisors (5 * n) = 144) :
  ∃ k : ℕ, 5^k ∣ n ∧ (∀ m : ℕ, 5^m ∣ n → m ≤ k) ∧ k = 4 :=
by sorry

end greatest_k_dividing_n_l88_88815


namespace robert_books_l88_88106

/-- Given that Robert reads at a speed of 75 pages per hour, books have 300 pages, and Robert reads for 9 hours,
    he can read 2 complete 300-page books in that time. -/
theorem robert_books (reading_speed : ℤ) (pages_per_book : ℤ) (hours_available : ℤ) 
(h1 : reading_speed = 75) 
(h2 : pages_per_book = 300) 
(h3 : hours_available = 9) : 
  hours_available / (pages_per_book / reading_speed) = 2 := 
by {
  -- adding placeholder for proof
  sorry
}

end robert_books_l88_88106


namespace remainder_when_divided_l88_88223

/-- Given integers T, E, N, S, E', N', S'. When T is divided by E, 
the quotient is N and the remainder is S. When N is divided by E', 
the quotient is N' and the remainder is S'. Prove that the remainder 
when T is divided by E + E' is ES' + S. -/
theorem remainder_when_divided (T E N S E' N' S' : ℤ) (h1 : T = N * E + S) (h2 : N = N' * E' + S') :
  (T % (E + E')) = (E * S' + S) :=
by
  sorry

end remainder_when_divided_l88_88223


namespace original_population_multiple_of_3_l88_88399

theorem original_population_multiple_of_3 (x y z : ℕ) (h1 : x^2 + 121 = y^2) (h2 : y^2 + 121 = z^2) :
  3 ∣ x^2 :=
sorry

end original_population_multiple_of_3_l88_88399


namespace angle_B_in_triangle_l88_88748

theorem angle_B_in_triangle (a b c : ℝ) (B : ℝ) (h : (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c) :
  B = 60 ∨ B = 120 := 
sorry

end angle_B_in_triangle_l88_88748


namespace alex_guarantees_victory_with_52_bullseyes_l88_88769

variable (m : ℕ) -- total score of Alex after the first half
variable (opponent_score : ℕ) -- total score of opponent after the first half
variable (remaining_shots : ℕ := 60) -- shots remaining for both players

-- Assume Alex always scores at least 3 points per shot and a bullseye earns 10 points
def min_bullseyes_to_guarantee_victory (m opponent_score : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 52 ∧
  (m + 7 * n + 180) > (opponent_score + 540)

-- Statement: Prove that if Alex leads by 60 points halfway through, then the minimum number of bullseyes he needs to guarantee a win is 52.
theorem alex_guarantees_victory_with_52_bullseyes (m opponent_score : ℕ) :
  m >= opponent_score + 60 → min_bullseyes_to_guarantee_victory m opponent_score :=
  sorry

end alex_guarantees_victory_with_52_bullseyes_l88_88769


namespace num_of_solutions_eq_28_l88_88298

def num_solutions : Nat :=
  sorry

theorem num_of_solutions_eq_28 : num_solutions = 28 :=
  sorry

end num_of_solutions_eq_28_l88_88298


namespace find_n_l88_88109

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ 100 * n % 103 = 65 % 103 ∧ n = 68 :=
by
  sorry

end find_n_l88_88109


namespace total_people_in_tour_group_l88_88763

noncomputable def tour_group_total_people (θ : ℝ) (N : ℕ) (children_percentage young_adults_percentage older_people_percentage : ℝ) : Prop :=
  (older_people_percentage = (θ + 9) / 3.6) ∧
  (young_adults_percentage = (θ + 27) / 3.6) ∧
  (N * young_adults_percentage / 100 = N * children_percentage / 100 + 9) ∧
  (children_percentage = θ / 3.6) →
  N = 120

theorem total_people_in_tour_group (θ : ℝ) (N : ℕ) (children_percentage young_adults_percentage older_people_percentage : ℝ) :
  tour_group_total_people θ N children_percentage young_adults_percentage older_people_percentage :=
sorry

end total_people_in_tour_group_l88_88763


namespace remaining_sweet_potatoes_l88_88160

def harvested_sweet_potatoes : ℕ := 80
def sold_sweet_potatoes_mrs_adams : ℕ := 20
def sold_sweet_potatoes_mr_lenon : ℕ := 15
def traded_sweet_potatoes : ℕ := 10
def donated_sweet_potatoes : ℕ := 5

theorem remaining_sweet_potatoes :
  harvested_sweet_potatoes - (sold_sweet_potatoes_mrs_adams + sold_sweet_potatoes_mr_lenon + traded_sweet_potatoes + donated_sweet_potatoes) = 30 :=
by
  sorry

end remaining_sweet_potatoes_l88_88160


namespace max_value_of_y_l88_88575

theorem max_value_of_y (x : ℝ) (h₁ : 0 < x) (h₂ : x < 4) : 
  ∃ y : ℝ, (y = x * (8 - 2 * x)) ∧ (∀ z : ℝ, z = x * (8 - 2 * x) → z ≤ 8) :=
sorry

end max_value_of_y_l88_88575


namespace smallest_points_2016_l88_88461

theorem smallest_points_2016 (n : ℕ) :
  n = 28225 →
  ∀ (points : Fin n → (ℤ × ℤ)),
  ∃ i j : Fin n, i ≠ j ∧
    let dist_sq := (points i).fst - (points j).fst ^ 2 + (points i).snd - (points j).snd ^ 2 
    ∃ k : ℤ, dist_sq = 2016 * k :=
by
  intro h points
  sorry

end smallest_points_2016_l88_88461


namespace three_digit_numbers_excluding_adjacent_same_digits_is_correct_l88_88672

def num_valid_three_digit_numbers_exclude_adjacent_same_digits : Nat :=
  let total_numbers := 900
  let excluded_numbers_AAB := 81
  let excluded_numbers_BAA := 81
  total_numbers - (excluded_numbers_AAB + excluded_numbers_BAA)

theorem three_digit_numbers_excluding_adjacent_same_digits_is_correct :
  num_valid_three_digit_numbers_exclude_adjacent_same_digits = 738 := by
  sorry

end three_digit_numbers_excluding_adjacent_same_digits_is_correct_l88_88672


namespace trigonometric_identity_l88_88687

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π - α) = -2) :
  (Real.cos (2 * π - α) + 2 * Real.cos (3 * π / 2 - α)) / (Real.sin (π - α) - Real.sin (-π / 2 - α)) = -1 :=
by
  sorry

end trigonometric_identity_l88_88687


namespace max_profit_300_l88_88796

noncomputable def total_cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def total_revenue (x : ℝ) : ℝ :=
if x ≤ 400 then (400 * x - (1 / 2) * x^2)
else 80000

noncomputable def total_profit (x : ℝ) : ℝ :=
total_revenue x - total_cost x

theorem max_profit_300 :
    ∃ x : ℝ, (total_profit x = (total_revenue 300 - total_cost 300)) := sorry

end max_profit_300_l88_88796


namespace find_f3_l88_88767

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 4)
  (h2 : f 2 = 10)
  (h3 : ∀ x, f x = a * x^2 + b * x + 2) :
  f 3 = 20 :=
by
  sorry

end find_f3_l88_88767


namespace complement_union_of_sets_l88_88208

variable {U M N : Set ℕ}

theorem complement_union_of_sets (h₁ : M ⊆ N) (h₂ : N ⊆ U) :
  (U \ M) ∪ (U \ N) = U \ M :=
by
  sorry

end complement_union_of_sets_l88_88208


namespace balls_in_boxes_l88_88746

theorem balls_in_boxes : 
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  total_ways - exclude_one_empty = 537 := 
by
  let total_ways := 3^6
  let exclude_one_empty := 3 * 2^6
  have h : total_ways - exclude_one_empty = 537 := sorry
  exact h

end balls_in_boxes_l88_88746


namespace max_composite_rel_prime_set_l88_88823

theorem max_composite_rel_prime_set : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 10 ≤ n ∧ n ≤ 99 ∧ ¬Nat.Prime n) ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1) ∧ 
  S.card = 4 := by
sorry

end max_composite_rel_prime_set_l88_88823


namespace brian_fewer_seashells_l88_88701

-- Define the conditions
def cb_ratio (Craig Brian : ℕ) : Prop := 9 * Brian = 7 * Craig
def craig_seashells (Craig : ℕ) : Prop := Craig = 54

-- Define the main theorem to be proven
theorem brian_fewer_seashells (Craig Brian : ℕ) (h1 : cb_ratio Craig Brian) (h2 : craig_seashells Craig) : Craig - Brian = 12 :=
by
  sorry

end brian_fewer_seashells_l88_88701


namespace total_grains_in_grey_parts_l88_88369

theorem total_grains_in_grey_parts 
  (total_grains_each_circle : ℕ)
  (white_grains_first_circle : ℕ)
  (white_grains_second_circle : ℕ)
  (common_white_grains : ℕ) 
  (h1 : white_grains_first_circle = 87)
  (h2 : white_grains_second_circle = 110)
  (h3 : common_white_grains = 68) :
  (white_grains_first_circle - common_white_grains) +
  (white_grains_second_circle - common_white_grains) = 61 :=
by
  sorry

end total_grains_in_grey_parts_l88_88369


namespace power_of_two_expression_l88_88707

theorem power_of_two_expression :
  2^2010 - 2^2009 - 2^2008 + 2^2007 - 2^2006 = 5 * 2^2006 :=
by
  sorry

end power_of_two_expression_l88_88707


namespace hundred_times_reciprocal_l88_88668

theorem hundred_times_reciprocal (x : ℝ) (h : 5 * x = 2) : 100 * (1 / x) = 250 := 
by 
  sorry

end hundred_times_reciprocal_l88_88668


namespace country_of_second_se_asian_fields_medal_recipient_l88_88008

-- Given conditions as definitions
def is_highest_recognition (award : String) : Prop :=
  award = "Fields Medal"

def fields_medal_freq (years : Nat) : Prop :=
  years = 4 -- Fields Medal is awarded every four years

def second_se_asian_recipient (name : String) : Prop :=
  name = "Ngo Bao Chau"

-- The main theorem to prove
theorem country_of_second_se_asian_fields_medal_recipient :
  ∀ (award : String) (years : Nat) (name : String),
    is_highest_recognition award ∧ fields_medal_freq years ∧ second_se_asian_recipient name →
    (name = "Ngo Bao Chau" → ∃ (country : String), country = "Vietnam") :=
by
  intros award years name h
  sorry

end country_of_second_se_asian_fields_medal_recipient_l88_88008


namespace reduced_price_is_25_l88_88879

def original_price (P : ℝ) (X : ℝ) (R : ℝ) : Prop :=
  R = 0.85 * P ∧ 
  500 = X * P ∧ 
  500 = (X + 3) * R

theorem reduced_price_is_25 (P X R : ℝ) (h : original_price P X R) :
  R = 25 :=
by
  sorry

end reduced_price_is_25_l88_88879


namespace slope_divides_polygon_area_l88_88308

structure Point where
  x : ℝ
  y : ℝ

noncomputable def polygon_vertices : List Point :=
  [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩]

-- Define the area calculation and conditions needed 
noncomputable def area_of_polygon (vertices : List Point) : ℝ :=
  -- Assuming here that a function exists to calculate the area given the vertices
  sorry

def line_through_origin (slope : ℝ) (x : ℝ) : Point :=
  ⟨x, slope * x⟩

theorem slope_divides_polygon_area :
  let line := line_through_origin (2 / 7)
  ∀ x : ℝ, ∃ (G : Point), 
  polygon_vertices = [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩] →
  area_of_polygon polygon_vertices / 2 = 
  area_of_polygon [⟨0, 0⟩, line x, G] :=
sorry

end slope_divides_polygon_area_l88_88308


namespace jellybean_count_l88_88182

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end jellybean_count_l88_88182


namespace relationship_y1_y2_l88_88456

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem relationship_y1_y2 :
  ∀ (a b c x₀ x₁ x₂ : ℝ),
    (quadratic_function a b c 0 = 4) →
    (quadratic_function a b c 1 = 1) →
    (quadratic_function a b c 2 = 0) →
    1 < x₁ → 
    x₁ < 2 → 
    3 < x₂ → 
    x₂ < 4 → 
    (quadratic_function a b c x₁ < quadratic_function a b c x₂) :=
by 
  sorry

end relationship_y1_y2_l88_88456


namespace factor_difference_of_squares_l88_88765

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l88_88765


namespace total_colored_hangers_l88_88038

theorem total_colored_hangers (pink_hangers green_hangers : ℕ) (h1 : pink_hangers = 7) (h2 : green_hangers = 4)
  (blue_hangers yellow_hangers : ℕ) (h3 : blue_hangers = green_hangers - 1) (h4 : yellow_hangers = blue_hangers - 1) :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = 16 :=
by
  sorry

end total_colored_hangers_l88_88038


namespace four_digit_number_l88_88022

-- Defining the cards and their holders
def cards : List ℕ := [2, 0, 1, 5]
def A : ℕ := 5
def B : ℕ := 1
def C : ℕ := 2
def D : ℕ := 0

-- Conditions based on statements
def A_statement (a b c d : ℕ) : Prop := 
  ¬ ((b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1))

def B_statement (a b c d : ℕ) : Prop := 
  (b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1)

def C_statement (c : ℕ) : Prop := ¬ (c = 1 ∨ c = 2 ∨ c = 5)
def D_statement (d : ℕ) : Prop := d ≠ 0

-- Truth conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

def tells_truth (n : ℕ) : Prop := is_odd n
def lies (n : ℕ) : Prop := is_even n

-- Proof statement
theorem four_digit_number (a b c d : ℕ) 
  (ha : a ∈ cards) (hb : b ∈ cards) (hc : c ∈ cards) (hd : d ∈ cards) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (truth_A : tells_truth a → A_statement a b c d)
  (lie_A : lies a → ¬ A_statement a b c d)
  (truth_B : tells_truth b → B_statement a b c d)
  (lie_B : lies b → ¬ B_statement a b c d)
  (truth_C : tells_truth c → C_statement c)
  (lie_C : lies c → ¬ C_statement c)
  (truth_D : tells_truth d → D_statement d)
  (lie_D : lies d → ¬ D_statement d) :
  a * 1000 + b * 100 + c * 10 + d = 5120 := 
  by
    sorry

end four_digit_number_l88_88022


namespace total_feet_is_140_l88_88503

def total_heads : ℕ := 48
def number_of_hens : ℕ := 26
def number_of_cows : ℕ := total_heads - number_of_hens
def feet_per_hen : ℕ := 2
def feet_per_cow : ℕ := 4

theorem total_feet_is_140 : ((number_of_hens * feet_per_hen) + (number_of_cows * feet_per_cow)) = 140 := by
  sorry

end total_feet_is_140_l88_88503


namespace num_distinct_prime_factors_330_l88_88925

theorem num_distinct_prime_factors_330 : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, Nat.Prime x ∧ 330 % x = 0 := 
sorry

end num_distinct_prime_factors_330_l88_88925


namespace parabola_sum_l88_88108

theorem parabola_sum (a b c : ℝ)
  (h1 : 4 = a * 1^2 + b * 1 + c)
  (h2 : -1 = a * (-2)^2 + b * (-2) + c)
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c = a * (x + 1)^2 - 2)
  : a + b + c = 5 := by
  sorry

end parabola_sum_l88_88108


namespace sum_of_coeffs_eq_one_l88_88262

theorem sum_of_coeffs_eq_one (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) (x : ℝ) :
  (1 - 2 * x) ^ 10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + 
                    a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1 :=
  sorry

end sum_of_coeffs_eq_one_l88_88262


namespace find_k_l88_88187

theorem find_k (α β k : ℝ) (h₁ : α^2 - α + k - 1 = 0) (h₂ : β^2 - β + k - 1 = 0) (h₃ : α^2 - 2*α - β = 4) :
  k = -4 :=
sorry

end find_k_l88_88187


namespace quadratic_equation_solutions_l88_88918

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 + 7 * x = 0 ↔ (x = 0 ∨ x = -7) := 
by 
  intro x
  sorry

end quadratic_equation_solutions_l88_88918


namespace sum_of_three_largest_ge_50_l88_88909

theorem sum_of_three_largest_ge_50 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) :
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
  a₆ ≠ a₇ ∧
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0 ∧ a₅ > 0 ∧ a₆ > 0 ∧ a₇ > 0 ∧
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 →
  ∃ (x y z : ℕ), (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧ (x > 0 ∧ y > 0 ∧ z > 0) ∧ (x + y + z ≥ 50) :=
by sorry

end sum_of_three_largest_ge_50_l88_88909


namespace smallest_positive_period_l88_88197

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem smallest_positive_period (ω : ℝ) (hω : ω > 0)
  (H : ∀ x1 x2 : ℝ, abs (f ω x1 - f ω x2) = 2 → abs (x1 - x2) = Real.pi / 2) :
  ∃ T > 0, T = Real.pi ∧ (∀ x : ℝ, f ω (x + T) = f ω x) := 
sorry

end smallest_positive_period_l88_88197


namespace closest_point_on_line_l88_88360

theorem closest_point_on_line (x y: ℚ) (h1: y = -4 * x + 3) (h2: ∀ p q: ℚ, y = -4 * p + 3 ∧ y = q * (-4 * p) - q * (-4 * 1 + 0)): (x, y) = (-1 / 17, 55 / 17) :=
sorry

end closest_point_on_line_l88_88360


namespace weekly_diesel_spending_l88_88657

-- Conditions
def cost_per_gallon : ℝ := 3
def fuel_used_in_two_weeks : ℝ := 24

-- Question: Prove that Mr. Alvarez spends $36 on diesel fuel each week.
theorem weekly_diesel_spending : (fuel_used_in_two_weeks / 2) * cost_per_gallon = 36 := by
  sorry

end weekly_diesel_spending_l88_88657


namespace find_incorrect_statements_l88_88710

-- Definitions of the statements based on their mathematical meanings
def is_regular_tetrahedron (shape : Type) : Prop := 
  -- assume some definition for regular tetrahedron
  sorry 

def is_cube (shape : Type) : Prop :=
  -- assume some definition for cube
  sorry

def is_generatrix_parallel (cylinder : Type) : Prop :=
  -- assume definition stating that generatrix of a cylinder is parallel to its axis
  sorry

def is_lateral_faces_isosceles (pyramid : Type) : Prop :=
  -- assume definition that in a regular pyramid, lateral faces are congruent isosceles triangles
  sorry

def forms_cone_on_rotation (triangle : Type) (axis : Type) : Prop :=
  -- assume definition that a right triangle forms a cone when rotated around one of its legs (other than hypotenuse)
  sorry

-- Given conditions as definitions
def statement_A : Prop := ∀ (shape : Type), is_regular_tetrahedron shape → is_cube shape = false
def statement_B : Prop := ∀ (cylinder : Type), is_generatrix_parallel cylinder = true
def statement_C : Prop := ∀ (pyramid : Type), is_lateral_faces_isosceles pyramid = true
def statement_D : Prop := ∀ (triangle : Type) (axis : Type), forms_cone_on_rotation triangle axis = false

-- The proof problem equivalent to incorrectness of statements A, B, and D
theorem find_incorrect_statements : 
  (statement_A = true) ∧ -- statement A is indeed incorrect
  (statement_B = true) ∧ -- statement B is indeed incorrect
  (statement_C = false) ∧ -- statement C is correct
  (statement_D = true)    -- statement D is indeed incorrect
:= 
sorry

end find_incorrect_statements_l88_88710


namespace franks_earnings_l88_88511

/-- Frank's earnings problem statement -/
theorem franks_earnings 
  (total_hours : ℕ) (days : ℕ) (regular_pay_rate : ℝ) (overtime_pay_rate : ℝ)
  (hours_first_day : ℕ) (overtime_first_day : ℕ)
  (hours_second_day : ℕ) (hours_third_day : ℕ)
  (hours_fourth_day : ℕ) (overtime_fourth_day : ℕ)
  (regular_hours_per_day : ℕ) :
  total_hours = 32 →
  days = 4 →
  regular_pay_rate = 15 →
  overtime_pay_rate = 22.50 →
  hours_first_day = 12 →
  overtime_first_day = 4 →
  hours_second_day = 8 →
  hours_third_day = 8 →
  hours_fourth_day = 12 →
  overtime_fourth_day = 4 →
  regular_hours_per_day = 8 →
  (32 * regular_pay_rate + 8 * overtime_pay_rate) = 660 := 
by 
  intros 
  sorry

end franks_earnings_l88_88511


namespace number_of_unit_fraction_pairs_l88_88548

/-- 
 The number of ways that 1/2007 can be expressed as the sum of two distinct positive unit fractions is 7.
-/
theorem number_of_unit_fraction_pairs : 
  ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 ≠ p.2 ∧ (1 : ℚ) / 2007 = 1 / ↑p.1 + 1 / ↑p.2) ∧ 
    pairs.card = 7 :=
sorry

end number_of_unit_fraction_pairs_l88_88548


namespace sum_of_other_endpoint_coordinates_l88_88242

theorem sum_of_other_endpoint_coordinates (x y : ℤ) :
  (7 + x) / 2 = 5 ∧ (4 + y) / 2 = -8 → x + y = -17 :=
by 
  sorry

end sum_of_other_endpoint_coordinates_l88_88242


namespace sequence_formula_sequence_inequality_l88_88000

open Nat

-- Definition of the sequence based on the given conditions
noncomputable def a : ℕ → ℚ
| 0     => 1                -- 0-indexed for Lean handling convenience, a_1 = 1 is a(0) in Lean
| (n+1) => 2 - 1 / (a n)    -- recurrence relation

-- Proof for part (I) that a_n = (n + 1) / n
theorem sequence_formula (n : ℕ) : a (n + 1) = (n + 2) / (n + 1) := sorry

-- Proof for part (II)
theorem sequence_inequality (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  (1 + a (n + 1)) / a (k + 1) < 2 ∨ (1 + a (k + 1)) / a (n + 1) < 2 := sorry

end sequence_formula_sequence_inequality_l88_88000


namespace product_of_roots_l88_88410

theorem product_of_roots (a b c : ℤ) (h_eq : a = 24 ∧ b = 60 ∧ c = -600) :
  ∀ x : ℂ, (a * x^2 + b * x + c = 0) → (x * (-b - x) = -25) := sorry

end product_of_roots_l88_88410


namespace geometric_sequence_sum_l88_88035

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (n : ℕ) 
    (S_n : ℝ) (S_3n : ℝ) (S_4n : ℝ)
    (h1 : S_n = 2) 
    (h2 : S_3n = 14) 
    (h3 : ∀ m : ℕ, S_m = a_n 1 * (1 - q^m) / (1 - q)) :
    S_4n = 30 :=
by
  sorry

end geometric_sequence_sum_l88_88035


namespace hyperbola_slope_condition_l88_88376

-- Define the setup
variables (a b : ℝ) (P F1 F2 : ℝ × ℝ)
variables (h : a > 0) (k : b > 0)
variables (hyperbola : (∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1)))

-- Define the condition
variables (cond : ∃ (P : ℝ × ℝ), 3 * abs (dist P F1 + dist P F2) ≤ 2 * dist F1 F2)

-- The proof goal
theorem hyperbola_slope_condition : (b / a) ≥ (Real.sqrt 5 / 2) :=
sorry

end hyperbola_slope_condition_l88_88376


namespace digits_product_l88_88231

-- Define the conditions
variables (A B : ℕ)

-- Define the main problem statement using the conditions and expected answer
theorem digits_product (h1 : A + B = 12) (h2 : (10 * A + B) % 3 = 0) : A * B = 35 := 
by
  sorry

end digits_product_l88_88231


namespace factorization_correct_l88_88597

-- Defining the expressions
def expr1 (x : ℝ) : ℝ := 4 * x^2 + 4 * x
def expr2 (x : ℝ) : ℝ := 4 * x * (x + 1)

-- Theorem statement: Prove that expr1 and expr2 are equivalent
theorem factorization_correct (x : ℝ) : expr1 x = expr2 x :=
by 
  sorry

end factorization_correct_l88_88597


namespace find_constant_a_l88_88524

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (Real.exp x - 1)

theorem find_constant_a (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = - f a x) : a = -1 := 
by
  sorry

end find_constant_a_l88_88524


namespace geometric_sequence_problem_l88_88359

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given condition for the geometric sequence
variables {a : ℕ → ℝ} (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27)

-- Theorem to be proven
theorem geometric_sequence_problem (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27) : a 1 * a 9 = 9 :=
sorry

end geometric_sequence_problem_l88_88359


namespace largest_divisor_8_l88_88229

theorem largest_divisor_8 (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) : 
  8 ∣ (p^2 - q^2 + 2*p - 2*q) := 
sorry

end largest_divisor_8_l88_88229


namespace arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l88_88299

theorem arrangement_two_rows :
  ∃ (ways : ℕ), ways = 5040 := by
  sorry

theorem arrangement_no_head_tail (A : ℕ):
  ∃ (ways : ℕ), ways = 3600 := by
  sorry

theorem arrangement_girls_together :
  ∃ (ways : ℕ), ways = 576 := by
  sorry

theorem arrangement_no_boys_next :
  ∃ (ways : ℕ), ways = 1440 := by
  sorry

end arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l88_88299


namespace total_toothpicks_correct_l88_88448

noncomputable def total_toothpicks_in_grid 
  (height : ℕ) (width : ℕ) (partition_interval : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  let num_partitions := height / partition_interval
  (horizontal_lines * width) + (vertical_lines * height) + (num_partitions * width)

theorem total_toothpicks_correct :
  total_toothpicks_in_grid 25 15 5 = 850 := 
by 
  sorry

end total_toothpicks_correct_l88_88448


namespace packs_per_box_l88_88749

theorem packs_per_box (total_cost : ℝ) (num_boxes : ℕ) (cost_per_pack : ℝ) 
  (num_tissues_per_pack : ℕ) (cost_per_tissue : ℝ) (total_packs : ℕ) :
  total_cost = 1000 ∧ num_boxes = 10 ∧ cost_per_pack = num_tissues_per_pack * cost_per_tissue ∧ 
  num_tissues_per_pack = 100 ∧ cost_per_tissue = 0.05 ∧ total_packs * cost_per_pack = total_cost / num_boxes →
  total_packs = 20 :=
by
  sorry

end packs_per_box_l88_88749


namespace money_distribution_l88_88966

-- Conditions
variable (A B x y : ℝ)
variable (h1 : x + 1/2 * y = 50)
variable (h2 : 2/3 * x + y = 50)

-- Problem statement
theorem money_distribution : x = A → y = B → (x + 1/2 * y = 50 ∧ 2/3 * x + y = 50) :=
by
  intro hx hy
  rw [hx, hy]
  exfalso -- using exfalso to skip proof body
  sorry

end money_distribution_l88_88966


namespace truncated_quadrilateral_pyramid_exists_l88_88279

theorem truncated_quadrilateral_pyramid_exists :
  ∃ (x y z u r s t : ℤ),
    x = 4 * r * t ∧
    y = 4 * s * t ∧
    z = (r - s)^2 - 2 * t^2 ∧
    u = (r - s)^2 + 2 * t^2 ∧
    (x - y)^2 + 2 * z^2 = 2 * u^2 :=
by
  sorry

end truncated_quadrilateral_pyramid_exists_l88_88279


namespace sam_last_30_minutes_speed_l88_88276

/-- 
Given the total distance of 96 miles driven in 1.5 hours, 
with the first 30 minutes at an average speed of 60 mph, 
and the second 30 minutes at an average speed of 65 mph,
we need to show that the average speed during the last 30 minutes was 67 mph.
-/
theorem sam_last_30_minutes_speed (total_distance : ℤ) (time1 time2 : ℤ) (speed1 speed2 speed_last segment_time : ℤ)
  (h_total_distance : total_distance = 96)
  (h_total_time : time1 + time2 + segment_time = 90)
  (h_segment_time : segment_time = 30)
  (convert_time1 : time1 = 30)
  (convert_time2 : time2 = 30)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 65)
  (h_average_speed : ((60 + 65 + speed_last) / 3) = 64) :
  speed_last = 67 := 
sorry

end sam_last_30_minutes_speed_l88_88276


namespace geometric_sequence_a3_a5_l88_88794

-- Define the geometric sequence condition using a function
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the given conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h1 : is_geometric_seq a)
variable (h2 : a 1 > 0)
variable (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

-- The main goal is to prove: a 3 + a 5 = 5
theorem geometric_sequence_a3_a5 : a 3 + a 5 = 5 :=
by
  simp [is_geometric_seq] at h1
  obtain ⟨q, ⟨hq_pos, hq⟩⟩ := h1
  sorry

end geometric_sequence_a3_a5_l88_88794


namespace age_problem_l88_88863

theorem age_problem
  (D M : ℕ)
  (h1 : M = D + 45)
  (h2 : M - 5 = 6 * (D - 5)) :
  D = 14 ∧ M = 59 := by
  sorry

end age_problem_l88_88863


namespace Allan_more_balloons_l88_88386

-- Define the number of balloons that Allan and Jake brought
def Allan_balloons := 5
def Jake_balloons := 3

-- Prove that the number of more balloons that Allan had than Jake is 2
theorem Allan_more_balloons : (Allan_balloons - Jake_balloons) = 2 := by sorry

end Allan_more_balloons_l88_88386


namespace function_even_and_monotonically_increasing_l88_88176

-- Definition: Even Function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Definition: Monotonically Increasing on (0, ∞)
def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- Given Function
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem to prove
theorem function_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on_pos f := by
  sorry

end function_even_and_monotonically_increasing_l88_88176


namespace saturday_earnings_l88_88253

-- Lean 4 Statement

theorem saturday_earnings 
  (S Wednesday_earnings : ℝ)
  (h1 : S + Wednesday_earnings = 5182.50)
  (h2 : Wednesday_earnings = S - 142.50) 
  : S = 2662.50 := 
by
  sorry

end saturday_earnings_l88_88253


namespace find_equation_with_new_roots_l88_88929

variable {p q r s : ℝ}

theorem find_equation_with_new_roots 
  (h_eq : ∀ x, x^2 - p * x + q = 0 ↔ (x = r ∧ x = s))
  (h_r_nonzero : r ≠ 0)
  (h_s_nonzero : s ≠ 0)
  : 
  ∀ x, (x^2 - ((q^2 + 1) * (p^2 - 2 * q) / q^2) * x + (q + 1/q)^2) = 0 ↔ 
       (x = r^2 + 1/(s^2) ∧ x = s^2 + 1/(r^2)) := 
sorry

end find_equation_with_new_roots_l88_88929


namespace original_work_days_l88_88120

-- Definitions based on conditions
noncomputable def L : ℕ := 7  -- Number of laborers originally employed
noncomputable def A : ℕ := 3  -- Number of absent laborers
noncomputable def t : ℕ := 14 -- Number of days it took the remaining laborers to finish the work

-- Theorem statement to prove
theorem original_work_days : (L - A) * t = L * 8 := by
  sorry

end original_work_days_l88_88120


namespace emails_left_in_inbox_l88_88642

-- Define the initial conditions and operations
def initial_emails : ℕ := 600

def move_half_to_trash (emails : ℕ) : ℕ := emails / 2
def move_40_percent_to_work (emails : ℕ) : ℕ := emails - (emails * 40 / 100)
def move_25_percent_to_personal (emails : ℕ) : ℕ := emails - (emails * 25 / 100)
def move_10_percent_to_miscellaneous (emails : ℕ) : ℕ := emails - (emails * 10 / 100)
def filter_30_percent_to_subfolders (emails : ℕ) : ℕ := emails - (emails * 30 / 100)
def archive_20_percent (emails : ℕ) : ℕ := emails - (emails * 20 / 100)

-- Statement we need to prove
theorem emails_left_in_inbox : 
  archive_20_percent
    (filter_30_percent_to_subfolders
      (move_10_percent_to_miscellaneous
        (move_25_percent_to_personal
          (move_40_percent_to_work
            (move_half_to_trash initial_emails))))) = 69 := 
by sorry

end emails_left_in_inbox_l88_88642


namespace Conor_can_chop_116_vegetables_in_a_week_l88_88044

-- Define the conditions
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8
def work_days_per_week : ℕ := 4

-- Define the total vegetables per day
def vegetables_per_day : ℕ := eggplants_per_day + carrots_per_day + potatoes_per_day

-- Define the total vegetables per week
def vegetables_per_week : ℕ := vegetables_per_day * work_days_per_week

-- The proof statement
theorem Conor_can_chop_116_vegetables_in_a_week : vegetables_per_week = 116 :=
by
  sorry  -- The proof step is omitted with sorry

end Conor_can_chop_116_vegetables_in_a_week_l88_88044


namespace sum_of_discounts_l88_88333

theorem sum_of_discounts
  (price_fox : ℝ)
  (price_pony : ℝ)
  (savings : ℝ)
  (discount_pony : ℝ) :
  (3 * price_fox * (F / 100) + 2 * price_pony * (discount_pony / 100) = savings) →
  (F + discount_pony = 22) :=
sorry


end sum_of_discounts_l88_88333


namespace combined_weight_l88_88972

noncomputable def Jake_weight : ℕ := 196
noncomputable def Kendra_weight : ℕ := 94

-- Condition: If Jake loses 8 pounds, he will weigh twice as much as Kendra
axiom lose_8_pounds (j k : ℕ) : (j - 8 = 2 * k) → j = Jake_weight → k = Kendra_weight

-- To Prove: The combined weight of Jake and Kendra is 290 pounds
theorem combined_weight (j k : ℕ) (h₁ : j = Jake_weight) (h₂ : k = Kendra_weight) : j + k = 290 := 
by  sorry

end combined_weight_l88_88972


namespace percentage_of_men_l88_88260

variables {M W : ℝ}
variables (h1 : M + W = 100)
variables (h2 : 0.20 * M + 0.40 * W = 34)

theorem percentage_of_men :
  M = 30 :=
by
  sorry

end percentage_of_men_l88_88260


namespace intersection_of_sets_l88_88429

noncomputable def A : Set ℝ := {x | -1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 5}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_sets : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_of_sets_l88_88429


namespace yacht_actual_cost_l88_88411

theorem yacht_actual_cost
  (discount_percentage : ℝ)
  (amount_paid : ℝ)
  (original_cost : ℝ)
  (h1 : discount_percentage = 0.72)
  (h2 : amount_paid = 3200000)
  (h3 : amount_paid = (1 - discount_percentage) * original_cost) :
  original_cost = 11428571.43 :=
by
  sorry

end yacht_actual_cost_l88_88411


namespace solution_set_x_l88_88499

theorem solution_set_x (x : ℝ) (h₁ : 33 * 32 ≤ x)
  (h₂ : ⌊x⌋ + ⌈x⌉ = 5) : 2 < x ∧ x < 3 :=
by
  sorry

end solution_set_x_l88_88499


namespace maximum_revenue_l88_88133

def ticket_price (x : ℕ) (y : ℤ) : Prop :=
  (6 ≤ x ∧ x ≤ 10 ∧ y = 1000 * x - 5750) ∨
  (10 < x ∧ x ≤ 38 ∧ y = -30 * x^2 + 1300 * x - 5750)

theorem maximum_revenue :
  ∃ x y, ticket_price x y ∧ y = 8830 ∧ x = 22 :=
by {
  sorry
}

end maximum_revenue_l88_88133


namespace not_super_lucky_years_l88_88756

def sum_of_month_and_day (m d : ℕ) : ℕ := m + d
def product_of_month_and_day (m d : ℕ) : ℕ := m * d
def sum_of_last_two_digits (y : ℕ) : ℕ :=
  let d1 := y / 10 % 10
  let d2 := y % 10
  d1 + d2

def is_super_lucky_year (y : ℕ) : Prop :=
  ∃ (m d : ℕ), sum_of_month_and_day m d = 24 ∧
               product_of_month_and_day m d = 2 * sum_of_last_two_digits y

theorem not_super_lucky_years :
  ¬ is_super_lucky_year 2070 ∧
  ¬ is_super_lucky_year 2081 ∧
  ¬ is_super_lucky_year 2092 :=
by {
  sorry
}

end not_super_lucky_years_l88_88756


namespace missing_coins_l88_88420

-- Definition representing the total number of coins Charlie received
variable (y : ℚ)

-- Conditions
def initial_lost_coins (y : ℚ) := (1 / 3) * y
def recovered_coins (y : ℚ) := (2 / 9) * y

-- Main Theorem
theorem missing_coins (y : ℚ) :
  y - (y * (8 / 9)) = y * (1 / 9) :=
by
  sorry

end missing_coins_l88_88420


namespace binary_representation_of_23_l88_88305

theorem binary_representation_of_23 : 23 = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end binary_representation_of_23_l88_88305


namespace equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l88_88218

-- Definition of a peculiar triangle.
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2

-- Problem 1: Proving an equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a : ℝ) : is_peculiar_triangle a a a :=
sorry

-- Problem 2: Proving the case when b is the hypotenuse in Rt△ABC makes it peculiar
theorem rt_triangle_is_peculiar (a b c : ℝ) (ha : a = 5 * Real.sqrt 2) (hc : c = 10) : 
  is_peculiar_triangle a b c ↔ b = Real.sqrt (c^2 + a^2) :=
sorry

-- Problem 3: Proving the ratio of the sides in a peculiar right triangle is 1 : √2 : √3
theorem peculiar_rt_triangle_ratio (a b c : ℝ) (hc : c^2 = a^2 + b^2) (hpeculiar : is_peculiar_triangle a c b) :
  (b = Real.sqrt 2 * a) ∧ (c = Real.sqrt 3 * a) :=
sorry

end equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l88_88218


namespace arithmetic_seq_a3_a9_zero_l88_88640

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_11_zero (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 0

theorem arithmetic_seq_a3_a9_zero (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_11_zero a) :
  a 3 + a 9 = 0 :=
sorry

end arithmetic_seq_a3_a9_zero_l88_88640


namespace solution_to_equation_l88_88981

theorem solution_to_equation :
  ∃ x : ℝ, x = (11 - 3 * Real.sqrt 5) / 2 ∧ x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 31 :=
by
  sorry

end solution_to_equation_l88_88981


namespace sum_of_variables_l88_88705

theorem sum_of_variables (a b c d : ℝ) (h₁ : a * c + a * d + b * c + b * d = 68) (h₂ : c + d = 4) : a + b + c + d = 21 :=
sorry

end sum_of_variables_l88_88705


namespace max_sqrt_expression_l88_88973

open Real

theorem max_sqrt_expression (x y z : ℝ) (h_sum : x + y + z = 3)
  (hx : x ≥ -1) (hy : y ≥ -(2/3)) (hz : z ≥ -2) :
  sqrt (3 * x + 3) + sqrt (3 * y + 2) + sqrt (3 * z + 6) ≤ 2 * sqrt 15 := by
  sorry

end max_sqrt_expression_l88_88973


namespace find_m_value_l88_88430

def vectors_parallel (a1 a2 b1 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

theorem find_m_value (m : ℝ) :
  let a := (6, 3)
  let b := (m, 2)
  vectors_parallel a.1 a.2 b.1 b.2 ↔ m = 4 :=
by
  intro H
  obtain ⟨_, _⟩ := H
  sorry

end find_m_value_l88_88430


namespace frog_jump_problem_l88_88252

theorem frog_jump_problem (A B C : ℝ) (PA PB PC : ℝ) 
  (H1: PA' = (PB + PC) / 2)
  (H2: jump_distance_B = 60)
  (H3: jump_distance_B = 2 * abs ((PB - (PB + PC) / 2))) :
  third_jump_distance = 30 := sorry

end frog_jump_problem_l88_88252


namespace highest_value_meter_l88_88993

theorem highest_value_meter (A B C : ℝ) 
  (h_avg : (A + B + C) / 3 = 6)
  (h_A_min : A = 2)
  (h_B_min : B = 2) : C = 14 :=
by {
  sorry
}

end highest_value_meter_l88_88993


namespace polygon_sides_l88_88846

theorem polygon_sides (n : ℕ) (h : n ≥ 3) (sum_angles : (n - 2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
sorry

end polygon_sides_l88_88846


namespace bike_helmet_cost_increase_l88_88933

open Real

theorem bike_helmet_cost_increase :
  let old_bike_cost := 150
  let old_helmet_cost := 50
  let new_bike_cost := old_bike_cost + 0.10 * old_bike_cost
  let new_helmet_cost := old_helmet_cost + 0.20 * old_helmet_cost
  let old_total_cost := old_bike_cost + old_helmet_cost
  let new_total_cost := new_bike_cost + new_helmet_cost
  let total_increase := new_total_cost - old_total_cost
  let percent_increase := (total_increase / old_total_cost) * 100
  percent_increase = 12.5 :=
by
  sorry

end bike_helmet_cost_increase_l88_88933


namespace carla_total_students_l88_88340

-- Defining the conditions
def students_in_restroom : Nat := 2
def absent_students : Nat := (3 * students_in_restroom) - 1
def total_desks : Nat := 4 * 6
def occupied_desks : Nat := total_desks * 2 / 3
def students_present : Nat := occupied_desks

-- The target is to prove the total number of students Carla teaches
theorem carla_total_students : students_in_restroom + absent_students + students_present = 23 := by
  sorry

end carla_total_students_l88_88340


namespace first_discount_percentage_l88_88165

-- Given conditions
def initial_price : ℝ := 390
def final_price : ℝ := 285.09
def second_discount : ℝ := 0.15

-- Definition for the first discount percentage
noncomputable def first_discount (D : ℝ) : ℝ :=
initial_price * (1 - D / 100) * (1 - second_discount)

-- Theorem statement
theorem first_discount_percentage : ∃ D : ℝ, first_discount D = final_price ∧ D = 13.99 :=
by
  sorry

end first_discount_percentage_l88_88165


namespace complement_B_in_U_l88_88138

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x = 1}
def U : Set ℕ := A ∪ B

theorem complement_B_in_U : (U \ B) = {2, 3} := by
  sorry

end complement_B_in_U_l88_88138


namespace desired_average_l88_88465

theorem desired_average (P1 P2 P3 : ℝ) (A : ℝ) 
  (hP1 : P1 = 74) 
  (hP2 : P2 = 84) 
  (hP3 : P3 = 67) 
  (hA : A = (P1 + P2 + P3) / 3) : 
  A = 75 :=
  sorry

end desired_average_l88_88465


namespace hypotenuse_length_l88_88151

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l88_88151


namespace same_remainder_division_l88_88699

theorem same_remainder_division (k r a b c d : ℕ) 
  (h_k_pos : 0 < k)
  (h_nonzero_r : 0 < r)
  (h_r_lt_k : r < k)
  (a_def : a = 2613)
  (b_def : b = 2243)
  (c_def : c = 1503)
  (d_def : d = 985)
  (h_a : a % k = r)
  (h_b : b % k = r)
  (h_c : c % k = r)
  (h_d : d % k = r) : 
  k = 74 ∧ r = 23 := 
by
  sorry

end same_remainder_division_l88_88699


namespace group_membership_l88_88852

theorem group_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 11 = 6) (h3 : 100 ≤ n ∧ n ≤ 200) :
  n = 116 ∨ n = 193 :=
sorry

end group_membership_l88_88852


namespace no_negatives_l88_88385

theorem no_negatives (x y : ℝ) (h : |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4|) : 
  ¬ (x < 0) ∧ ¬ (y < 0) :=
by
  sorry

end no_negatives_l88_88385


namespace complement_intersection_l88_88311

noncomputable def M : Set ℝ := {x | |x| > 2}
noncomputable def N : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | 1 < x ∧ x ≤ 2} :=
sorry

end complement_intersection_l88_88311


namespace moles_CO2_is_one_l88_88241

noncomputable def moles_CO2_formed (moles_HNO3 moles_NaHCO3 : ℕ) : ℕ :=
  if moles_HNO3 = 1 ∧ moles_NaHCO3 = 1 then 1 else 0

theorem moles_CO2_is_one :
  moles_CO2_formed 1 1 = 1 :=
by
  sorry

end moles_CO2_is_one_l88_88241


namespace poly_constant_or_sum_constant_l88_88007

-- definitions of the polynomials as real-coefficient polynomials
variables (P Q R : Polynomial ℝ)

-- conditions
#check ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ) -- Considering 'constant' as 1 for simplicity

-- target
theorem poly_constant_or_sum_constant 
  (h : ∀ x, P.eval (Q.eval x) + P.eval (R.eval x) = (1 : ℝ)) :
  (∃ c : ℝ, ∀ x, P.eval x = c) ∨ (∃ c : ℝ, ∀ x, Q.eval x + R.eval x = c) :=
sorry

end poly_constant_or_sum_constant_l88_88007


namespace quadratic_roots_real_find_m_value_l88_88101

theorem quadratic_roots_real (m : ℝ) (h_roots : ∃ x1 x2 : ℝ, x1 * x1 + 4 * x1 + (m - 1) = 0 ∧ x2 * x2 + 4 * x2 + (m - 1) = 0) :
  m ≤ 5 :=
by {
  sorry
}

theorem find_m_value (m : ℝ) (x1 x2 : ℝ) (h_eq1 : x1 * x1 + 4 * x1 + (m - 1) = 0) (h_eq2 : x2 * x2 + 4 * x2 + (m - 1) = 0) (h_cond : 2 * (x1 + x2) + x1 * x2 + 10 = 0) :
  m = -1 :=
by {
  sorry
}

end quadratic_roots_real_find_m_value_l88_88101


namespace math_proof_problem_l88_88301

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

def parallel (x : Line) (y : Plane) : Prop := sorry
def contained_in (x : Line) (y : Plane) : Prop := sorry
def perpendicular (x : Plane) (y : Plane) : Prop := sorry
def perpendicular_line_plane (x : Line) (y : Plane) : Prop := sorry

theorem math_proof_problem :
  (perpendicular α β) ∧ (perpendicular_line_plane m β) ∧ ¬(contained_in m α) → parallel m α :=
by
  sorry

end math_proof_problem_l88_88301


namespace total_pieces_eq_21_l88_88021

-- Definitions based on conditions
def red_pieces : Nat := 5
def yellow_pieces : Nat := 7
def green_pieces : Nat := 11

-- Derived definitions from conditions
def red_cuts : Nat := red_pieces - 1
def yellow_cuts : Nat := yellow_pieces - 1
def green_cuts : Nat := green_pieces - 1

-- Total cuts and the resulting total pieces
def total_cuts : Nat := red_cuts + yellow_cuts + green_cuts
def total_pieces : Nat := total_cuts + 1

-- Prove the total number of pieces is 21
theorem total_pieces_eq_21 : total_pieces = 21 := by
  sorry

end total_pieces_eq_21_l88_88021


namespace combined_reach_l88_88455

theorem combined_reach (barry_reach : ℝ) (larry_height : ℝ) (shoulder_ratio : ℝ) :
  barry_reach = 5 → larry_height = 5 → shoulder_ratio = 0.80 → 
  (larry_height * shoulder_ratio + barry_reach) = 9 :=
by
  intros h1 h2 h3
  sorry

end combined_reach_l88_88455


namespace interest_group_selections_l88_88116

-- Define the number of students and the number of interest groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem statement: The total number of different possible selections of interest groups is 81.
theorem interest_group_selections : num_groups ^ num_students = 81 := by
  sorry

end interest_group_selections_l88_88116


namespace slope_of_given_line_eq_l88_88163

theorem slope_of_given_line_eq : (∀ x y : ℝ, (4 / x + 5 / y = 0) → (x ≠ 0 ∧ y ≠ 0) → ∀ y x : ℝ, y = - (5 * x / 4) → ∃ m, m = -5/4) :=
by
  sorry

end slope_of_given_line_eq_l88_88163


namespace intersection_eq_one_l88_88585

def M : Set ℕ := {0, 1}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_eq_one : M ∩ N = {1} := 
by
  sorry

end intersection_eq_one_l88_88585


namespace range_of_a_plus_b_at_least_one_nonnegative_l88_88824

-- Conditions
variable (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2)

-- Proof Problem 1: Prove that the range of a + b is [0, +∞)
theorem range_of_a_plus_b : (a + b) ≥ 0 :=
by sorry

-- Proof Problem 2: Prove by contradiction that at least one of a or b is greater than or equal to 0
theorem at_least_one_nonnegative : ¬(a < 0 ∧ b < 0) :=
by sorry

end range_of_a_plus_b_at_least_one_nonnegative_l88_88824


namespace black_cars_in_parking_lot_l88_88828

theorem black_cars_in_parking_lot :
  let total_cars := 3000
  let blue_percent := 0.40
  let red_percent := 0.25
  let green_percent := 0.15
  let yellow_percent := 0.10
  let black_percent := 1 - (blue_percent + red_percent + green_percent + yellow_percent)
  let number_of_black_cars := total_cars * black_percent
  number_of_black_cars = 300 :=
by
  sorry

end black_cars_in_parking_lot_l88_88828


namespace perimeter_of_similar_triangle_l88_88785

theorem perimeter_of_similar_triangle (a b c d : ℕ) (h_iso : (a = 12) ∧ (b = 24) ∧ (c = 24)) (h_sim : d = 30) 
  : (d + 2 * b) = 150 := by
  sorry

end perimeter_of_similar_triangle_l88_88785


namespace passengers_off_in_texas_l88_88504

variable (x : ℕ) -- number of passengers who got off in Texas
variable (initial_passengers : ℕ := 124)
variable (texas_boarding : ℕ := 24)
variable (nc_off : ℕ := 47)
variable (nc_boarding : ℕ := 14)
variable (virginia_passengers : ℕ := 67)

theorem passengers_off_in_texas {x : ℕ} :
  (initial_passengers - x + texas_boarding - nc_off + nc_boarding) = virginia_passengers → 
  x = 48 :=
by
  sorry

end passengers_off_in_texas_l88_88504


namespace root_of_linear_equation_l88_88281

theorem root_of_linear_equation (b c : ℝ) (hb : b ≠ 0) :
  ∃ x : ℝ, 0 * x^2 + b * x + c = 0 → x = -c / b :=
by
  -- The proof steps would typically go here
  sorry

end root_of_linear_equation_l88_88281


namespace intersection_point_exists_l88_88709

def h : ℝ → ℝ := sorry  -- placeholder for the function h
def j : ℝ → ℝ := sorry  -- placeholder for the function j

-- Conditions
axiom h_3_eq : h 3 = 3
axiom j_3_eq : j 3 = 3
axiom h_6_eq : h 6 = 9
axiom j_6_eq : j 6 = 9
axiom h_9_eq : h 9 = 18
axiom j_9_eq : j 9 = 18

-- Theorem
theorem intersection_point_exists :
  ∃ a b : ℝ, a = 2 ∧ h (3 * a) = 3 * j (a) ∧ h (3 * a) = b ∧ 3 * j (a) = b ∧ a + b = 11 :=
  sorry

end intersection_point_exists_l88_88709


namespace total_problems_completed_l88_88450

variables (p t : ℕ)
variables (hp_pos : 15 < p) (ht_pos : 0 < t)
variables (eq1 : (3 * p - 6) * (t - 3) = p * t)

theorem total_problems_completed : p * t = 120 :=
by sorry

end total_problems_completed_l88_88450


namespace sum_of_reversed_integers_l88_88055

-- Definitions of properties and conditions
def reverse_digits (m n : ℕ) : Prop :=
  let to_digits (x : ℕ) : List ℕ := x.digits 10
  to_digits m = (to_digits n).reverse

-- The main theorem statement
theorem sum_of_reversed_integers
  (m n : ℕ)
  (h_rev: reverse_digits m n)
  (h_prod: m * n = 1446921630) :
  m + n = 79497 :=
sorry

end sum_of_reversed_integers_l88_88055


namespace ball_count_in_box_eq_57_l88_88476

theorem ball_count_in_box_eq_57 (N : ℕ) (h : N - 44 = 70 - N) : N = 57 :=
sorry

end ball_count_in_box_eq_57_l88_88476


namespace solve_system_eqns_l88_88510

theorem solve_system_eqns (x y z : ℝ) :
  x^2 - 23 * y + 66 * z + 612 = 0 ∧
  y^2 + 62 * x - 20 * z + 296 = 0 ∧
  z^2 - 22 * x + 67 * y + 505 = 0 ↔
  x = -20 ∧ y = -22 ∧ z = -23 :=
by
  sorry

end solve_system_eqns_l88_88510


namespace remainder_when_divided_82_l88_88729

theorem remainder_when_divided_82 (x : ℤ) (k m : ℤ) (R : ℤ) (h1 : 0 ≤ R) (h2 : R < 82)
    (h3 : x = 82 * k + R) (h4 : x + 7 = 41 * m + 12) : R = 5 :=
by
  sorry

end remainder_when_divided_82_l88_88729


namespace number_of_intersection_points_l88_88759

theorem number_of_intersection_points (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃ x : Finset ℝ, (∀ y ∈ x, f ((y:ℝ)^2) = f ((y:ℝ)^6)) ∧ x.card = 3 :=
by
  sorry

end number_of_intersection_points_l88_88759


namespace fraction_of_boxes_loaded_by_day_crew_l88_88811

-- Definitions based on the conditions
variables (D W : ℕ)  -- Day crew per worker boxes (D) and number of workers (W)

-- Helper Definitions
def boxes_day_crew : ℕ := D * W  -- Total boxes by day crew
def boxes_night_crew : ℕ := (3 * D / 4) * (3 * W / 4)  -- Total boxes by night crew
def total_boxes : ℕ := boxes_day_crew D W + boxes_night_crew D W  -- Total boxes by both crews

-- The main theorem
theorem fraction_of_boxes_loaded_by_day_crew :
  (boxes_day_crew D W : ℚ) / (total_boxes D W : ℚ) = 16/25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l88_88811


namespace run_to_cafe_time_l88_88632

theorem run_to_cafe_time (h_speed_const : ∀ t1 t2 d1 d2 : ℝ, (t1 / d1) = (t2 / d2))
  (h_store_time : 24 = 3 * (24 / 3))
  (h_cafe_halfway : ∀ d : ℝ, d = 1.5) :
  ∃ t : ℝ, t = 12 :=
by
  sorry

end run_to_cafe_time_l88_88632


namespace expand_and_simplify_product_l88_88903

variable (x : ℝ)

theorem expand_and_simplify_product :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 :=
by
  sorry

end expand_and_simplify_product_l88_88903


namespace probability_even_sum_of_spins_l88_88304

theorem probability_even_sum_of_spins :
  let prob_even_first := 3 / 6
  let prob_odd_first := 3 / 6
  let prob_even_second := 2 / 5
  let prob_odd_second := 3 / 5
  let prob_both_even := prob_even_first * prob_even_second
  let prob_both_odd := prob_odd_first * prob_odd_second
  prob_both_even + prob_both_odd = 1 / 2 := 
by 
  sorry

end probability_even_sum_of_spins_l88_88304


namespace monotonicity_of_f_on_interval_l88_88663

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x - 2

theorem monotonicity_of_f_on_interval (a b : ℝ) (h1 : a = -3) (h2 : b = 0) :
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 2 → f x1 a b ≥ f x2 a b := 
by
  sorry

end monotonicity_of_f_on_interval_l88_88663


namespace total_earnings_from_peaches_l88_88257

-- Definitions of the conditions
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def price_per_peach_friends : ℝ := 2
def peaches_sold_to_relatives : ℕ :=  4
def price_per_peach_relatives : ℝ := 1.25
def peaches_for_self : ℕ := 1

-- We aim to prove the following statement
theorem total_earnings_from_peaches :
  (peaches_sold_to_friends * price_per_peach_friends) +
  (peaches_sold_to_relatives * price_per_peach_relatives) = 25 := by
  -- proof goes here
  sorry

end total_earnings_from_peaches_l88_88257


namespace final_amount_is_75139_84_l88_88512

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : ℝ :=
  P * (1 + r/n)^(n * t)

theorem final_amount_is_75139_84 (P : ℝ) (r : ℝ) (t : ℝ) (n : ℕ) :
  P = 64000 → r = 1/12 → t = 2 → n = 12 → compoundInterest P r t n = 75139.84 :=
by
  intros hP hr ht hn
  sorry

end final_amount_is_75139_84_l88_88512


namespace determine_k_l88_88148

theorem determine_k (k : ℝ) : (1 - 3 * k * (-2/3) = 7 * 3) → k = 10 :=
by
  intro h
  sorry

end determine_k_l88_88148


namespace expr_value_l88_88915

theorem expr_value : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 :=
by
  sorry

end expr_value_l88_88915


namespace abs_diff_of_two_numbers_l88_88140

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 216) : |x - y| = 6 := 
sorry

end abs_diff_of_two_numbers_l88_88140


namespace div_1988_form_1989_div_1989_form_1988_l88_88185

/-- There exists a number of the form 1989...19890... (1989 repeated several times followed by several zeros), which is divisible by 1988. -/
theorem div_1988_form_1989 (k : ℕ) : ∃ n : ℕ, (n = 1989 * 10^(4*k) ∧ n % 1988 = 0) := sorry

/-- There exists a number of the form 1988...1988 (1988 repeated several times), which is divisible by 1989. -/
theorem div_1989_form_1988 (k : ℕ) : ∃ n : ℕ, (n = 1988 * ((10^(4*k)) - 1) ∧ n % 1989 = 0) := sorry

end div_1988_form_1989_div_1989_form_1988_l88_88185


namespace determine_percentage_of_second_mixture_l88_88383

-- Define the given conditions and question
def mixture_problem (P : ℝ) : Prop :=
  ∃ (V1 V2 : ℝ) (A1 A2 A_final : ℝ),
  V1 = 2.5 ∧ A1 = 0.30 ∧
  V2 = 7.5 ∧ A2 = P / 100 ∧
  A_final = 0.45 ∧
  (V1 * A1 + V2 * A2) / (V1 + V2) = A_final

-- State the theorem
theorem determine_percentage_of_second_mixture : mixture_problem 50 := sorry

end determine_percentage_of_second_mixture_l88_88383


namespace change_calculation_l88_88947

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end change_calculation_l88_88947


namespace smallest_positive_angle_l88_88945

theorem smallest_positive_angle (deg : ℤ) (k : ℤ) (h : deg = -2012) : ∃ m : ℤ, m = 148 ∧ 0 ≤ m ∧ m < 360 ∧ (∃ n : ℤ, deg + 360 * n = m) :=
by
  sorry

end smallest_positive_angle_l88_88945


namespace sqrt_of_square_neg7_l88_88206

theorem sqrt_of_square_neg7 : Real.sqrt ((-7:ℝ)^2) = 7 := by
  sorry

end sqrt_of_square_neg7_l88_88206


namespace total_points_seven_players_l88_88829

theorem total_points_seven_players (S : ℕ) (x : ℕ) 
  (hAlex : Alex_scored = S / 4)
  (hBen : Ben_scored = 2 * S / 7)
  (hCharlie : Charlie_scored = 15)
  (hTotal : S / 4 + 2 * S / 7 + 15 + x = S)
  (hMultiple : S = 56) : 
  x = 11 := 
sorry

end total_points_seven_players_l88_88829


namespace solution_is_permutations_of_2_neg2_4_l88_88702

-- Definitions of the conditions
def cond1 (x y z : ℤ) : Prop := x * y + y * z + z * x = -4
def cond2 (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 24
def cond3 (x y z : ℤ) : Prop := x^3 + y^3 + z^3 + 3 * x * y * z = 16

-- The set of all integer solutions as permutations of (2, -2, 4)
def is_solution (x y z : ℤ) : Prop :=
  (x = 2 ∧ y = -2 ∧ z = 4) ∨ (x = 2 ∧ y = 4 ∧ z = -2) ∨
  (x = -2 ∧ y = 2 ∧ z = 4) ∨ (x = -2 ∧ y = 4 ∧ z = 2) ∨
  (x = 4 ∧ y = 2 ∧ z = -2) ∨ (x = 4 ∧ y = -2 ∧ z = 2)

-- Lean statement for the proof problem
theorem solution_is_permutations_of_2_neg2_4 (x y z : ℤ) :
  cond1 x y z → cond2 x y z → cond3 x y z → is_solution x y z :=
by
  -- sorry, the proof goes here
  sorry

end solution_is_permutations_of_2_neg2_4_l88_88702


namespace ax5_plus_by5_l88_88207

-- Declare real numbers a, b, x, y
variables (a b x y : ℝ)

theorem ax5_plus_by5 (h1 : a * x + b * y = 3)
                     (h2 : a * x^2 + b * y^2 = 7)
                     (h3 : a * x^3 + b * y^3 = 6)
                     (h4 : a * x^4 + b * y^4 = 42) :
                     a * x^5 + b * y^5 = 20 := 
sorry

end ax5_plus_by5_l88_88207


namespace area_of_defined_region_l88_88985

theorem area_of_defined_region : 
  ∃ (A : ℝ), (∀ x y : ℝ, |4 * x - 20| + |3 * y + 9| ≤ 6 → A = 9) :=
sorry

end area_of_defined_region_l88_88985


namespace range_of_a_for_integer_solutions_l88_88716

theorem range_of_a_for_integer_solutions (a : ℝ) :
  (∃ x : ℤ, (a - 2 < x ∧ x ≤ 3)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_for_integer_solutions_l88_88716


namespace parallelogram_is_central_not_axis_symmetric_l88_88935

-- Definitions for the shapes discussed in the problem
def is_central_symmetric (shape : Type) : Prop := sorry
def is_axis_symmetric (shape : Type) : Prop := sorry

-- Specific shapes being used in the problem
def rhombus : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry

-- Example additional assumptions about shapes can be added here if needed

-- The problem assertion
theorem parallelogram_is_central_not_axis_symmetric :
  is_central_symmetric parallelogram ∧ ¬ is_axis_symmetric parallelogram :=
sorry

end parallelogram_is_central_not_axis_symmetric_l88_88935


namespace people_got_rid_of_some_snails_l88_88644

namespace SnailProblem

def originalSnails : ℕ := 11760
def remainingSnails : ℕ := 8278
def snailsGotRidOf : ℕ := 3482

theorem people_got_rid_of_some_snails :
  originalSnails - remainingSnails = snailsGotRidOf :=
by 
  sorry

end SnailProblem

end people_got_rid_of_some_snails_l88_88644


namespace unique_zero_of_f_l88_88938

theorem unique_zero_of_f (f : ℝ → ℝ) (h1 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 16) 
  (h2 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 8) (h3 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 4) 
  (h4 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 2) : ¬ ∃ x, f x = 0 ∧ 2 ≤ x ∧ x < 16 := 
by
  sorry

end unique_zero_of_f_l88_88938


namespace rhombus_area_l88_88059

theorem rhombus_area (side d1 : ℝ) (h_side : side = 28) (h_d1 : d1 = 12) : 
  (side = 28 ∧ d1 = 12) →
  ∃ area : ℝ, area = 328.32 := 
by 
  sorry

end rhombus_area_l88_88059


namespace remainder_product_mod_eq_l88_88192

theorem remainder_product_mod_eq (n : ℤ) :
  ((12 - 2 * n) * (n + 5)) % 11 = (-2 * n^2 + 2 * n + 5) % 11 := by
  sorry

end remainder_product_mod_eq_l88_88192


namespace percentage_of_number_l88_88650

/-- 
  Given a certain percentage \( P \) of 600 is 90.
  If 30% of 50% of a number 4000 is 90,
  Then P equals to 15%.
-/
theorem percentage_of_number (P : ℝ) (h1 : (0.30 : ℝ) * (0.50 : ℝ) * 4000 = 600) (h2 : P * 600 = 90) :
  P = 0.15 :=
  sorry

end percentage_of_number_l88_88650


namespace molecular_weight_of_3_moles_HBrO3_l88_88310

-- Definitions from the conditions
def mol_weight_H : ℝ := 1.01  -- atomic weight of H
def mol_weight_Br : ℝ := 79.90  -- atomic weight of Br
def mol_weight_O : ℝ := 16.00  -- atomic weight of O

-- Definition of molecular weight of HBrO3
def mol_weight_HBrO3 : ℝ := mol_weight_H + mol_weight_Br + 3 * mol_weight_O

-- The goal: The molecular weight of 3 moles of HBrO3 is 386.73 grams
theorem molecular_weight_of_3_moles_HBrO3 : 3 * mol_weight_HBrO3 = 386.73 :=
by
  -- We will insert the proof here later
  sorry

end molecular_weight_of_3_moles_HBrO3_l88_88310


namespace copy_pages_15_dollars_l88_88951

theorem copy_pages_15_dollars (cpp : ℕ) (budget : ℕ) (pages : ℕ) (h1 : cpp = 3) (h2 : budget = 1500) (h3 : pages = budget / cpp) : pages = 500 :=
by
  sorry

end copy_pages_15_dollars_l88_88951


namespace arithmetic_sequence_sum_l88_88313

theorem arithmetic_sequence_sum :
  ∀ {a : ℕ → ℕ} {S : ℕ → ℕ},
  (∀ n, a (n + 1) - a n = a 1 - a 0) →
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 1 + a 9 = 18 →
  a 4 = 7 →
  S 8 = 64 :=
by
  intros a S h_arith_seq h_sum_formula h_a1_a9 h_a4
  sorry

end arithmetic_sequence_sum_l88_88313


namespace part1_solution_set_part2_no_real_x_l88_88768

-- Condition and problem definitions
def f (x a : ℝ) : ℝ := a^2 * x^2 + 2 * a * x - a^2 + 1

theorem part1_solution_set :
  (∀ x : ℝ, f x 2 ≤ 0 ↔ -3 / 2 ≤ x ∧ x ≤ 1 / 2) := sorry

theorem part2_no_real_x :
  ¬ ∃ x : ℝ, ∀ a : ℝ, -2 ≤ a ∧ a ≤ 2 → f x a ≥ 0 := sorry

end part1_solution_set_part2_no_real_x_l88_88768


namespace find_p_from_binomial_distribution_l88_88987

theorem find_p_from_binomial_distribution (p : ℝ) (h₁ : 0 ≤ p ∧ p ≤ 1) 
    (h₂ : ∀ n k : ℕ, k ≤ n → 0 ≤ p^(k:ℝ) * (1-p)^((n-k):ℝ)) 
    (h₃ : (1 - (1 - p)^2 = 5 / 9)) : p = 1 / 3 :=
by sorry

end find_p_from_binomial_distribution_l88_88987


namespace x_days_worked_l88_88803

theorem x_days_worked (W : ℝ) :
  let x_work_rate := W / 20
  let y_work_rate := W / 24
  let y_days := 12
  let y_work_done := y_work_rate * y_days
  let total_work := W
  let work_done_by_x := (W - y_work_done) / x_work_rate
  work_done_by_x = 10 := 
by
  sorry

end x_days_worked_l88_88803


namespace find_ccb_l88_88814

theorem find_ccb (a b c : ℕ) 
  (h1: a ≠ b) 
  (h2: a ≠ c) 
  (h3: b ≠ c) 
  (h4: b = 1) 
  (h5: (10 * a + b) ^ 2 = 100 * c + 10 * c + b) 
  (h6: 100 * c + 10 * c + b > 300) : 
  100 * c + 10 * c + b = 441 :=
sorry

end find_ccb_l88_88814


namespace choose_president_and_vice_president_l88_88255

theorem choose_president_and_vice_president :
  let total_members := 24
  let boys := 8
  let girls := 16
  let senior_members := 4
  let senior_boys := 2
  let senior_girls := 2
  let president_choices := senior_members
  let vice_president_choices_boy_pres := girls
  let vice_president_choices_girl_pres := boys - senior_boys
  let total_ways :=
    (senior_boys * vice_president_choices_boy_pres) + 
    (senior_girls * vice_president_choices_girl_pres)
  total_ways = 44 := 
by
  sorry

end choose_president_and_vice_president_l88_88255


namespace friend_spent_seven_l88_88661

/-- You and your friend spent a total of $11 for lunch.
    Your friend spent $3 more than you.
    Prove that your friend spent $7 on their lunch. -/
theorem friend_spent_seven (you friend : ℝ) 
  (h1: you + friend = 11) 
  (h2: friend = you + 3) : 
  friend = 7 := 
by 
  sorry

end friend_spent_seven_l88_88661


namespace delta_delta_delta_l88_88896

-- Define the function Δ
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Mathematical statement to be proved
theorem delta_delta_delta (x : ℝ) : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end delta_delta_delta_l88_88896


namespace constant_sum_l88_88535

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem constant_sum (a1 d : ℝ) (h : 3 * arithmetic_sequence a1 d 8 = k) :
  ∃ k : ℝ, sum_arithmetic_sequence a1 d 15 = k :=
sorry

end constant_sum_l88_88535


namespace smallest_circle_radius_eq_l88_88853

open Real

-- Declaring the problem's conditions
def largestCircleRadius : ℝ := 10
def smallestCirclesCount : ℕ := 6
def congruentSmallerCirclesFitWithinLargerCircle (r : ℝ) : Prop :=
  3 * (2 * r) = 2 * largestCircleRadius

-- Stating the theorem to prove
theorem smallest_circle_radius_eq :
  ∃ r : ℝ, congruentSmallerCirclesFitWithinLargerCircle r ∧ r = 10 / 3 :=
by
  sorry

end smallest_circle_radius_eq_l88_88853


namespace Ferris_break_length_l88_88172

noncomputable def Audrey_rate_per_hour := (1:ℝ) / 4
noncomputable def Ferris_rate_per_hour := (1:ℝ) / 3
noncomputable def total_completion_time := (2:ℝ)
noncomputable def number_of_breaks := (6:ℝ)
noncomputable def job_completion_audrey := total_completion_time * Audrey_rate_per_hour
noncomputable def job_completion_ferris := 1 - job_completion_audrey
noncomputable def working_time_ferris := job_completion_ferris / Ferris_rate_per_hour
noncomputable def total_break_time := total_completion_time - working_time_ferris
noncomputable def break_length := total_break_time / number_of_breaks

theorem Ferris_break_length :
  break_length = (5:ℝ) / 60 := 
sorry

end Ferris_break_length_l88_88172


namespace variance_of_white_balls_l88_88833

section
variable (n : ℕ := 7) 
variable (p : ℚ := 3/7)

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_white_balls : binomial_variance n p = 12/7 :=
by
  sorry
end

end variance_of_white_balls_l88_88833


namespace sin_range_l88_88703

theorem sin_range :
  ∀ x, (-Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4) → (∃ y, y = Real.sin x ∧ -Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1) := by
  sorry

end sin_range_l88_88703


namespace circle_radius_l88_88953

theorem circle_radius (r x y : ℝ) (hx : x = π * r^2) (hy : y = 2 * π * r) (h : x + y = 90 * π) : r = 9 := by
  sorry

end circle_radius_l88_88953


namespace product_of_real_values_l88_88416

theorem product_of_real_values (r : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x)) = (r - x) / 8 → (3 * x * x - 3 * r * x + 8 = 0)) →
  r = 4 * Real.sqrt 6 / 3 ∨ r = -(4 * Real.sqrt 6 / 3) →
  r * -r = -32 / 3 :=
by
  intro h_x
  intro h_r
  sorry

end product_of_real_values_l88_88416


namespace divisible_by_n_sequence_l88_88083

theorem divisible_by_n_sequence (n : ℕ) (h1 : n > 1) (h2 : n % 2 = 1) : 
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 ∧ n ∣ (2^k - 1) :=
by {
  sorry
}

end divisible_by_n_sequence_l88_88083


namespace maria_traveled_portion_of_distance_l88_88686

theorem maria_traveled_portion_of_distance (total_distance first_stop remaining_distance_to_destination : ℝ) 
  (h1 : total_distance = 560) 
  (h2 : first_stop = total_distance / 2) 
  (h3 : remaining_distance_to_destination = 210) : 
  ((first_stop - (first_stop - (remaining_distance_to_destination + (first_stop - total_distance / 2)))) / (total_distance - first_stop)) = 1 / 4 :=
by
  sorry

end maria_traveled_portion_of_distance_l88_88686


namespace find_k_l88_88576

def condition (k : ℝ) : Prop := 24 / k = 4

theorem find_k (k : ℝ) (h : condition k) : k = 6 :=
sorry

end find_k_l88_88576


namespace locus_of_midpoint_l88_88937

theorem locus_of_midpoint (x y : ℝ) (h : y ≠ 0) :
  (∃ P : ℝ × ℝ, P = (2*x, 2*y) ∧ ((P.1^2 + (P.2-3)^2 = 9))) →
  (x^2 + (y - 3/2)^2 = 9/4) :=
by
  sorry

end locus_of_midpoint_l88_88937


namespace eval_at_3_l88_88773

theorem eval_at_3 : (3^3)^(3^3) = 27^27 :=
by sorry

end eval_at_3_l88_88773


namespace profit_percentage_correct_l88_88170

-- Statement of the problem in Lean
theorem profit_percentage_correct (SP CP : ℝ) (hSP : SP = 400) (hCP : CP = 320) : 
  ((SP - CP) / CP) * 100 = 25 := by
  -- Proof goes here
  sorry

end profit_percentage_correct_l88_88170


namespace find_blue_balloons_l88_88061

theorem find_blue_balloons (purple_balloons : ℕ) (left_balloons : ℕ) (total_balloons : ℕ) (blue_balloons : ℕ) :
  purple_balloons = 453 →
  left_balloons = 378 →
  total_balloons = left_balloons * 2 →
  total_balloons = purple_balloons + blue_balloons →
  blue_balloons = 303 := by
  intros h1 h2 h3 h4
  sorry

end find_blue_balloons_l88_88061


namespace lara_additional_miles_needed_l88_88111

theorem lara_additional_miles_needed :
  ∀ (d1 d2 d_total t1 speed1 speed2 avg_speed : ℝ),
    d1 = 20 →
    speed1 = 25 →
    speed2 = 40 →
    avg_speed = 35 →
    t1 = d1 / speed1 →
    d_total = d1 + d2 →
    avg_speed = (d_total) / (t1 + d2 / speed2) →
    d2 = 64 :=
by sorry

end lara_additional_miles_needed_l88_88111


namespace right_triangle_conditions_l88_88898

theorem right_triangle_conditions (x y z h α β : ℝ) : 
  x - y = α → 
  z - h = β → 
  x^2 + y^2 = z^2 → 
  x * y = h * z → 
  β > α :=
by 
sorry

end right_triangle_conditions_l88_88898


namespace functional_equation_solution_l88_88760

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equation_solution_l88_88760


namespace find_angle_l88_88147

variable (a b : ℝ × ℝ) (α : ℝ)
variable (θ : ℝ)

-- Conditions provided in the problem
def condition1 := (a.1^2 + a.2^2 = 4)
def condition2 := (b = (4 * Real.cos α, -4 * Real.sin α))
def condition3 := (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)

-- Desired result
theorem find_angle (h1 : condition1 a) (h2 : condition2 b α) (h3 : condition3 a b) :
  θ = Real.pi / 3 :=
sorry

end find_angle_l88_88147


namespace possible_values_of_expr_l88_88039

-- Define conditions
variables (x y : ℝ)
axiom h1 : x + y = 2
axiom h2 : y > 0
axiom h3 : x ≠ 0

-- Define the expression we're investigating
noncomputable def expr : ℝ := (1 / (abs x)) + (abs x / (y + 2))

-- The statement of the problem
theorem possible_values_of_expr :
  expr x y = 3 / 4 ∨ expr x y = 5 / 4 :=
sorry

end possible_values_of_expr_l88_88039


namespace vector_subtraction_scalar_mul_l88_88274

theorem vector_subtraction_scalar_mul :
  let v₁ := (3, -8) 
  let scalar := -5 
  let v₂ := (4, 6)
  v₁.1 - scalar * v₂.1 = 23 ∧ v₁.2 - scalar * v₂.2 = 22 := by
    sorry

end vector_subtraction_scalar_mul_l88_88274


namespace susan_correct_guess_probability_l88_88860

theorem susan_correct_guess_probability :
  (1 - (5/6)^6) = 31031/46656 := 
sorry

end susan_correct_guess_probability_l88_88860


namespace a_2015_eq_neg6_l88_88238

noncomputable def a : ℕ → ℤ
| 0 => 3
| 1 => 6
| (n+2) => a (n+1) - a n

theorem a_2015_eq_neg6 : a 2015 = -6 := 
by 
  sorry

end a_2015_eq_neg6_l88_88238


namespace green_apples_more_than_red_apples_l88_88910

noncomputable def num_original_green_apples : ℕ := 32
noncomputable def num_more_red_apples_than_green : ℕ := 200
noncomputable def num_delivered_green_apples : ℕ := 340
noncomputable def num_original_red_apples : ℕ :=
  num_original_green_apples + num_more_red_apples_than_green
noncomputable def num_new_green_apples : ℕ :=
  num_original_green_apples + num_delivered_green_apples

theorem green_apples_more_than_red_apples :
  num_new_green_apples - num_original_red_apples = 140 :=
by {
  sorry
}

end green_apples_more_than_red_apples_l88_88910


namespace mass_percentage_Al_aluminum_carbonate_l88_88962

theorem mass_percentage_Al_aluminum_carbonate :
  let m_Al := 26.98  -- molar mass of Al in g/mol
  let m_C := 12.01  -- molar mass of C in g/mol
  let m_O := 16.00  -- molar mass of O in g/mol
  let molar_mass_CO3 := m_C + 3 * m_O  -- molar mass of CO3 in g/mol
  let molar_mass_Al2CO33 := 2 * m_Al + 3 * molar_mass_CO3  -- molar mass of Al2(CO3)3 in g/mol
  let mass_Al_in_Al2CO33 := 2 * m_Al  -- mass of Al in Al2(CO3)3 in g/mol
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  -- Proof goes here
  sorry

end mass_percentage_Al_aluminum_carbonate_l88_88962


namespace parabola_y_values_order_l88_88624

theorem parabola_y_values_order :
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  -- The proof is omitted
  sorry

end parabola_y_values_order_l88_88624


namespace max_value_l88_88443

variable (a b c d : ℝ)

theorem max_value 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) 
  (h5 : b ≠ d) (h6 : c ≠ d)
  (cond1 : a / b + b / c + c / d + d / a = 4)
  (cond2 : a * c = b * d) :
  (a / c + b / d + c / a + d / b) ≤ -12 :=
sorry

end max_value_l88_88443


namespace manager_salary_proof_l88_88173

noncomputable def manager_salary 
    (avg_salary_without_manager : ℝ) 
    (num_employees_without_manager : ℕ) 
    (increase_in_avg_salary : ℝ) 
    (new_total_salary : ℝ) : ℝ :=
    new_total_salary - (num_employees_without_manager * avg_salary_without_manager)

theorem manager_salary_proof :
    manager_salary 3500 100 800 (101 * (3500 + 800)) = 84300 :=
by
    sorry

end manager_salary_proof_l88_88173


namespace student_b_speed_l88_88329

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end student_b_speed_l88_88329


namespace no_real_roots_ffx_l88_88558

noncomputable def quadratic_f (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem no_real_roots_ffx (a b c : ℝ) (h : (b - 1)^2 < 4 * a * c) :
  ∀ x : ℝ, quadratic_f a b c (quadratic_f a b c x) ≠ x :=
by
  sorry

end no_real_roots_ffx_l88_88558


namespace eval_infinite_series_eq_4_l88_88612

open BigOperators

noncomputable def infinite_series_sum : ℝ :=
  ∑' k, (k^2) / (3^k)

theorem eval_infinite_series_eq_4 : infinite_series_sum = 4 := 
  sorry

end eval_infinite_series_eq_4_l88_88612


namespace fuchsia_to_mauve_l88_88454

theorem fuchsia_to_mauve (F : ℝ) :
  (5 / 8) * F + (3 * 26.67 : ℝ) = (3 / 8) * F + (5 / 8) * F →
  F = 106.68 :=
by
  intro h
  -- Step to implement the solution would go here
  sorry

end fuchsia_to_mauve_l88_88454


namespace sum_of_series_l88_88948

theorem sum_of_series : 
  (6 + 16 + 26 + 36 + 46) + (14 + 24 + 34 + 44 + 54) = 300 :=
by
  sorry

end sum_of_series_l88_88948


namespace team_a_daily_work_rate_l88_88961

theorem team_a_daily_work_rate
  (L : ℕ) (D1 : ℕ) (D2 : ℕ) (w : ℕ → ℕ)
  (hL : L = 8250)
  (hD1 : D1 = 4)
  (hD2 : D2 = 7)
  (hwB : ∀ (x : ℕ), w x = x + 150)
  (hwork : ∀ (x : ℕ), D1 * x + D2 * (x + (w x)) = L) :
  ∃ x : ℕ, x = 400 :=
by
  sorry

end team_a_daily_work_rate_l88_88961


namespace paolo_sevilla_birthday_l88_88181

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l88_88181


namespace minimum_value_of_a_l88_88357

theorem minimum_value_of_a :
  (∀ x : ℝ, x > 0 → (a : ℝ) * x * Real.exp x - x - Real.log x ≥ 0) → a ≥ 1 / Real.exp 1 :=
by
  sorry

end minimum_value_of_a_l88_88357


namespace living_room_area_l88_88984

-- Define the conditions
def carpet_area (length width : ℕ) : ℕ :=
  length * width

def percentage_coverage (carpet_area living_room_area : ℕ) : ℕ :=
  (carpet_area * 100) / living_room_area

-- State the problem
theorem living_room_area (A : ℕ) (carpet_len carpet_wid : ℕ) (carpet_coverage : ℕ) :
  carpet_len = 4 → carpet_wid = 9 → carpet_coverage = 20 →
  20 * A = 36 * 100 → A = 180 :=
by
  intros h_len h_wid h_coverage h_proportion
  sorry

end living_room_area_l88_88984


namespace fuel_relationship_l88_88381

theorem fuel_relationship (y : ℕ → ℕ) (h₀ : y 0 = 80) (h₁ : y 1 = 70) (h₂ : y 2 = 60) (h₃ : y 3 = 50) :
  ∀ x : ℕ, y x = 80 - 10 * x :=
by
  sorry

end fuel_relationship_l88_88381


namespace change_received_proof_l88_88331

-- Define the costs and amounts
def regular_ticket_cost : ℕ := 9
def children_ticket_discount : ℕ := 2
def amount_given : ℕ := 2 * 20

-- Define the number of people
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 3

-- Define the costs calculations
def child_ticket_cost := regular_ticket_cost - children_ticket_discount
def total_adults_cost := number_of_adults * regular_ticket_cost
def total_children_cost := number_of_children * child_ticket_cost
def total_cost := total_adults_cost + total_children_cost
def change_received := amount_given - total_cost

-- Lean statement to prove the change received
theorem change_received_proof : change_received = 1 := by
  sorry

end change_received_proof_l88_88331


namespace compute_f5_l88_88967

-- Definitions of the logical operations used in the conditions
axiom x1 : Prop
axiom x2 : Prop
axiom x3 : Prop
axiom x4 : Prop
axiom x5 : Prop

noncomputable def x6 : Prop := x1 ∨ x3
noncomputable def x7 : Prop := x2 ∧ x6
noncomputable def x8 : Prop := x3 ∨ x5
noncomputable def x9 : Prop := x4 ∧ x8
noncomputable def f5 : Prop := x7 ∨ x9

-- Proof statement to be proven
theorem compute_f5 : f5 = (x7 ∨ x9) :=
by sorry

end compute_f5_l88_88967


namespace sum_of_segments_l88_88517

noncomputable def segment_sum (AB_len CB_len FG_len : ℕ) : ℝ :=
  199 * (Real.sqrt (AB_len * AB_len + CB_len * CB_len) +
         Real.sqrt (AB_len * AB_len + FG_len * FG_len))

theorem sum_of_segments : segment_sum 5 6 8 = 199 * (Real.sqrt 61 + Real.sqrt 89) :=
by
  sorry

end sum_of_segments_l88_88517


namespace base_length_l88_88519

-- Definition: Isosceles triangle
structure IsoscelesTriangle :=
  (perimeter : ℝ)
  (side : ℝ)

-- Conditions: Perimeter and one side of the isosceles triangle
def given_triangle : IsoscelesTriangle := {
  perimeter := 26,
  side := 11
}

-- The problem to solve: length of the base given the perimeter and one side
theorem base_length : 
  (given_triangle.perimeter = 26 ∧ given_triangle.side = 11) →
  (∃ b : ℝ, b = 11 ∨ b = 7.5) :=
by 
  sorry

end base_length_l88_88519


namespace valid_k_for_triangle_l88_88091

theorem valid_k_for_triangle (k : ℕ) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ b + c > a ∧ c + a > b)) → k ≥ 6 :=
by
  sorry

end valid_k_for_triangle_l88_88091


namespace roll_contains_25_coins_l88_88337

variable (coins_per_roll : ℕ)

def rolls_per_teller := 10
def number_of_tellers := 4
def total_coins := 1000

theorem roll_contains_25_coins : 
  (number_of_tellers * rolls_per_teller * coins_per_roll = total_coins) → 
  (coins_per_roll = 25) :=
by
  sorry

end roll_contains_25_coins_l88_88337


namespace maximum_ratio_x_over_y_l88_88994

theorem maximum_ratio_x_over_y {x y : ℕ} (hx : x > 9 ∧ x < 100) (hy : y > 9 ∧ y < 100)
  (hmean : x + y = 110) (hsquare : ∃ z : ℕ, z^2 = x * y) : x = 99 ∧ y = 11 := 
by
  -- mathematical proof
  sorry

end maximum_ratio_x_over_y_l88_88994


namespace melted_mixture_weight_l88_88084

theorem melted_mixture_weight
    (Z C : ℝ)
    (ratio_eq : Z / C = 9 / 11)
    (zinc_weight : Z = 33.3) :
    Z + C = 74 :=
by
  sorry

end melted_mixture_weight_l88_88084


namespace sin_neg_pi_l88_88486

theorem sin_neg_pi : Real.sin (-Real.pi) = 0 := by
  sorry

end sin_neg_pi_l88_88486


namespace solve_for_a_and_b_l88_88287

theorem solve_for_a_and_b (a b : ℤ) (h1 : 5 + a = 6 - b) (h2 : 6 + b = 9 + a) : 5 - a = 6 := 
sorry

end solve_for_a_and_b_l88_88287


namespace positive_solution_y_l88_88335

theorem positive_solution_y (x y z : ℝ) 
  (h1 : x * y = 8 - 3 * x - 2 * y) 
  (h2 : y * z = 15 - 5 * y - 3 * z) 
  (h3 : x * z = 40 - 5 * x - 4 * z) : 
  y = 4 := 
sorry

end positive_solution_y_l88_88335


namespace prime_power_implies_one_l88_88073

theorem prime_power_implies_one (p : ℕ) (a : ℤ) (n : ℕ) (h_prime : Nat.Prime p) (h_eq : 2^p + 3^p = a^n) :
  n = 1 :=
sorry

end prime_power_implies_one_l88_88073


namespace correct_conclusions_l88_88434

variable (f : ℝ → ℝ)

def condition_1 := ∀ x : ℝ, f (x + 2) = f (2 - (x + 2))
def condition_2 := ∀ x : ℝ, f (-2*x - 1) = -f (2*x + 1)

theorem correct_conclusions 
  (h1 : condition_1 f) 
  (h2 : condition_2 f) : 
  f 1 = f 3 ∧ 
  f 2 + f 4 = 0 ∧ 
  f (-1 / 2) * f (11 / 2) ≤ 0 := 
by 
  sorry

end correct_conclusions_l88_88434


namespace supremum_of_function_l88_88849

theorem supremum_of_function : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 
  (∃ M : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ M) ∧
    (∀ K : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ K) → M ≤ K) → M = -9 / 2) := 
sorry

end supremum_of_function_l88_88849


namespace find_number_of_women_in_first_group_l88_88584

variables (W : ℕ)

-- Conditions
def women_coloring_rate := 10
def total_cloth_colored_in_3_days := 180
def women_in_first_group := total_cloth_colored_in_3_days / 3

theorem find_number_of_women_in_first_group
  (h1 : 5 * women_coloring_rate * 4 = 200)
  (h2 : W * women_coloring_rate = women_in_first_group) :
  W = 6 :=
by
  sorry

end find_number_of_women_in_first_group_l88_88584


namespace no_int_coords_equilateral_l88_88378

--- Define a structure for points with integer coordinates
structure Point :=
(x : ℤ)
(y : ℤ)

--- Definition of the distance squared between two points
def dist_squared (P Q : Point) : ℤ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

--- Statement that given three points with integer coordinates, they cannot form an equilateral triangle
theorem no_int_coords_equilateral (A B C : Point) :
  ¬ (dist_squared A B = dist_squared B C ∧ dist_squared B C = dist_squared C A ∧ dist_squared C A = dist_squared A B) :=
sorry

end no_int_coords_equilateral_l88_88378


namespace jessica_initial_money_l88_88616

def amount_spent : ℝ := 10.22
def amount_left : ℝ := 1.51
def initial_amount : ℝ := 11.73

theorem jessica_initial_money :
  amount_spent + amount_left = initial_amount := 
  by
    sorry

end jessica_initial_money_l88_88616


namespace cone_prism_volume_ratio_l88_88184

-- Define the volumes and the ratio proof problem
theorem cone_prism_volume_ratio (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    (V_cone / V_prism) = (π / 36) :=
by
    -- Here we define the volumes of the cone and prism as given in the problem
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    -- We then assert the ratio condition based on the solution
    sorry

end cone_prism_volume_ratio_l88_88184


namespace intersection_S_T_l88_88854

def S : Set ℝ := {x | x > -2}

def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_S_T : S ∩ T = {x | -2 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_S_T_l88_88854


namespace scientific_notation_600_million_l88_88314

theorem scientific_notation_600_million : (600000000 : ℝ) = 6 * 10^8 := 
by 
  -- Insert the proof here
  sorry

end scientific_notation_600_million_l88_88314


namespace total_money_given_to_children_l88_88886

theorem total_money_given_to_children (B : ℕ) (x : ℕ) (total : ℕ) 
  (h1 : B = 300) 
  (h2 : x = B / 3) 
  (h3 : total = (2 * x) + (3 * x) + (4 * x)) : 
  total = 900 := 
by 
  sorry

end total_money_given_to_children_l88_88886


namespace rectangle_area_l88_88199

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l88_88199


namespace find_second_number_l88_88400

-- Definitions for the conditions
def ratio_condition (x : ℕ) : Prop := 5 * x = 40

-- The theorem we need to prove, i.e., the second number is 8 given the conditions
theorem find_second_number (x : ℕ) (h : ratio_condition x) : x = 8 :=
by sorry

end find_second_number_l88_88400


namespace solve_for_x_l88_88866

theorem solve_for_x :
  (16^x * 16^x * 16^x * 4^(3 * x) = 64^(4 * x)) → x = 0 := by
  sorry

end solve_for_x_l88_88866


namespace polynomial_coeff_sum_l88_88868

theorem polynomial_coeff_sum :
  let p := ((Polynomial.C 1 + Polynomial.X)^3 * (Polynomial.C 2 + Polynomial.X)^2)
  let a0 := p.coeff 0
  let a2 := p.coeff 2
  let a4 := p.coeff 4
  a4 + a2 + a0 = 36 := by 
  sorry

end polynomial_coeff_sum_l88_88868


namespace find_value_of_expression_l88_88782

variable {a b c d x : ℝ}

-- Conditions
def opposites (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def abs_three (x : ℝ) : Prop := |x| = 3

-- Proof
theorem find_value_of_expression (h1 : opposites a b) (h2 : reciprocals c d) 
  (h3 : abs_three x) : ∃ res : ℝ, (res = 3 ∨ res = -3) ∧ res = 10 * a + 10 * b + c * d * x :=
by
  sorry

end find_value_of_expression_l88_88782


namespace line_intersects_x_axis_at_point_l88_88470

theorem line_intersects_x_axis_at_point : 
  let x1 := 3
  let y1 := 7
  let x2 := -1
  let y2 := 3
  let m := (y2 - y1) / (x2 - x1) -- slope formula
  let b := y1 - m * x1        -- y-intercept formula
  let x_intersect := -b / m  -- x-coordinate where the line intersects x-axis
  (x_intersect, 0) = (-4, 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l88_88470


namespace prime_power_value_l88_88070

theorem prime_power_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h1 : Nat.Prime (7 * p + q)) (h2 : Nat.Prime (p * q + 11)) : 
  p ^ q = 8 ∨ p ^ q = 9 := 
sorry

end prime_power_value_l88_88070


namespace increase_in_cost_l88_88970

def initial_lumber_cost : ℝ := 450
def initial_nails_cost : ℝ := 30
def initial_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

def initial_total_cost : ℝ := initial_lumber_cost + initial_nails_cost + initial_fabric_cost

def new_lumber_cost : ℝ := initial_lumber_cost * (1 + lumber_inflation_rate)
def new_nails_cost : ℝ := initial_nails_cost * (1 + nails_inflation_rate)
def new_fabric_cost : ℝ := initial_fabric_cost * (1 + fabric_inflation_rate)

def new_total_cost : ℝ := new_lumber_cost + new_nails_cost + new_fabric_cost

theorem increase_in_cost :
  new_total_cost - initial_total_cost = 97 := 
sorry

end increase_in_cost_l88_88970


namespace possible_days_l88_88471

namespace AnyaVanyaProblem

-- Conditions
def AnyaLiesOn (d : String) : Prop := d = "Tuesday" ∨ d = "Wednesday" ∨ d = "Thursday"
def AnyaTellsTruthOn (d : String) : Prop := ¬AnyaLiesOn d

def VanyaLiesOn (d : String) : Prop := d = "Thursday" ∨ d = "Friday" ∨ d = "Saturday"
def VanyaTellsTruthOn (d : String) : Prop := ¬VanyaLiesOn d

-- Statements
def AnyaStatement (d : String) : Prop := d = "Friday"
def VanyaStatement (d : String) : Prop := d = "Tuesday"

-- Proof problem
theorem possible_days (d : String) : 
  (AnyaTellsTruthOn d ↔ AnyaStatement d) ∧ (VanyaTellsTruthOn d ↔ VanyaStatement d)
  → d = "Tuesday" ∨ d = "Thursday" ∨ d = "Friday" := 
sorry

end AnyaVanyaProblem

end possible_days_l88_88471


namespace initial_number_of_nurses_l88_88884

theorem initial_number_of_nurses (N : ℕ) (initial_doctors : ℕ) (remaining_staff : ℕ) 
  (h1 : initial_doctors = 11) 
  (h2 : remaining_staff = 22) 
  (h3 : initial_doctors - 5 + N - 2 = remaining_staff) : N = 18 :=
by
  rw [h1, h2] at h3
  sorry

end initial_number_of_nurses_l88_88884


namespace total_employees_l88_88549

theorem total_employees (female_employees managers male_associates female_managers : ℕ)
  (h_female_employees : female_employees = 90)
  (h_managers : managers = 40)
  (h_male_associates : male_associates = 160)
  (h_female_managers : female_managers = 40) :
  female_employees - female_managers + male_associates + managers = 250 :=
by {
  sorry
}

end total_employees_l88_88549


namespace multiple_of_one_third_l88_88957

theorem multiple_of_one_third (x : ℚ) (h : x * (1 / 3) = 2 / 9) : x = 2 / 3 :=
sorry

end multiple_of_one_third_l88_88957


namespace average_marks_l88_88690

theorem average_marks (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) 
                      (chemistry_marks : ℕ) (biology_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 →
  chemistry_marks = 87 → biology_marks = 81 → 
  (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / 5 = 85 :=
by
  intros
  sorry

end average_marks_l88_88690


namespace original_number_divisible_by_3_l88_88836

theorem original_number_divisible_by_3:
  ∃ (a b c d e f g h : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h) ∧
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h) ∧
  (c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h) ∧
  (d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h) ∧
  (e ≠ f ∧ e ≠ g ∧ e ≠ h) ∧
  (f ≠ g ∧ f ≠ h) ∧
  (g ≠ h) ∧ 
  (a + b + c + b + d + e + f + e + g + d + h) % 3 = 0 :=
sorry

end original_number_divisible_by_3_l88_88836


namespace cos_triple_angle_l88_88011

theorem cos_triple_angle
  (θ : ℝ)
  (h : Real.cos θ = 1/3) :
  Real.cos (3 * θ) = -23 / 27 :=
by
  sorry

end cos_triple_angle_l88_88011


namespace inequality_always_holds_l88_88689

theorem inequality_always_holds (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) →
  (a > 3) ∧ (∀ x : ℝ, x = a + 9 / (a - 1) → x ≥ 7) :=
by
  sorry

end inequality_always_holds_l88_88689


namespace systematic_sampling_correct_l88_88272

-- Definitions for the conditions
def total_products := 60
def group_count := 5
def products_per_group := total_products / group_count

-- systematic sampling condition: numbers are in increments of products_per_group
def systematic_sample (start : ℕ) (count : ℕ) : List ℕ := List.range' start products_per_group count

-- Given sequences
def A : List ℕ := [5, 10, 15, 20, 25]
def B : List ℕ := [5, 12, 31, 39, 57]
def C : List ℕ := [5, 17, 29, 41, 53]
def D : List ℕ := [5, 15, 25, 35, 45]

-- Correct solution defined
def correct_solution := [5, 17, 29, 41, 53]

-- Problem Statement
theorem systematic_sampling_correct :
  systematic_sample 5 group_count = correct_solution :=
by
  sorry

end systematic_sampling_correct_l88_88272


namespace bus_stops_per_hour_l88_88593

theorem bus_stops_per_hour 
  (bus_speed_without_stoppages : Float)
  (bus_speed_with_stoppages : Float)
  (bus_stops_per_hour_in_minutes : Float) :
  bus_speed_without_stoppages = 60 ∧ 
  bus_speed_with_stoppages = 45 → 
  bus_stops_per_hour_in_minutes = 15 := by
  sorry

end bus_stops_per_hour_l88_88593


namespace exists_infinitely_many_n_with_increasing_ω_l88_88952

open Nat

/--
  Let ω(n) represent the number of distinct prime factors of a natural number n (where n > 1).
  Prove that there exist infinitely many n such that ω(n) < ω(n + 1) < ω(n + 2).
-/
theorem exists_infinitely_many_n_with_increasing_ω (ω : ℕ → ℕ) (hω : ∀ (n : ℕ), n > 1 → ∃ k, ω k < ω (k + 1) ∧ ω (k + 1) < ω (k + 2)) :
  ∃ (infinitely_many : ℕ → Prop), ∀ N : ℕ, ∃ n : ℕ, N < n ∧ infinitely_many n :=
by
  sorry

end exists_infinitely_many_n_with_increasing_ω_l88_88952


namespace volleyball_team_selection_l88_88064

noncomputable def numberOfWaysToChooseStarters : ℕ :=
  (Nat.choose 13 4 * 3) + (Nat.choose 14 4 * 1)

theorem volleyball_team_selection :
  numberOfWaysToChooseStarters = 3146 := by
  sorry

end volleyball_team_selection_l88_88064


namespace isosceles_triangle_height_l88_88312

theorem isosceles_triangle_height (s h : ℝ) (eq_areas : (2 * s * s) = (1/2 * s * h)) : h = 4 * s :=
by
  sorry

end isosceles_triangle_height_l88_88312


namespace compare_fractions_l88_88928

theorem compare_fractions : (-2 / 7) > (-3 / 10) :=
sorry

end compare_fractions_l88_88928


namespace range_of_a_l88_88076

variable {x a : ℝ}

def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := |x| > a

theorem range_of_a (h : ¬p x → ¬q x a) : a ≤ 1 :=
sorry

end range_of_a_l88_88076


namespace prob_four_vertical_faces_same_color_l88_88599

noncomputable def painted_cube_probability : ℚ :=
  let total_arrangements := 3^6
  let suitable_arrangements := 3 + 18 + 6
  suitable_arrangements / total_arrangements

theorem prob_four_vertical_faces_same_color : 
  painted_cube_probability = 1 / 27 := by
  sorry

end prob_four_vertical_faces_same_color_l88_88599


namespace value_of_x_y_squared_l88_88245

theorem value_of_x_y_squared (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : (x - y)^2 = 16 :=
by
  sorry

end value_of_x_y_squared_l88_88245


namespace find_a_minimum_value_at_x_2_l88_88739

def f (x a : ℝ) := x^3 - a * x

theorem find_a_minimum_value_at_x_2 (a : ℝ) :
  (∃ x : ℝ, x = 2 ∧ ∀ y ≠ 2, f y a ≥ f 2 a) → a = 12 :=
by 
  -- Here we should include the proof steps
  sorry

end find_a_minimum_value_at_x_2_l88_88739


namespace gumballs_per_box_l88_88389

-- Given conditions
def total_gumballs : ℕ := 20
def total_boxes : ℕ := 4

-- Mathematically equivalent proof problem
theorem gumballs_per_box:
  total_gumballs / total_boxes = 5 := by
  sorry

end gumballs_per_box_l88_88389


namespace die_total_dots_l88_88155

theorem die_total_dots :
  ∀ (face1 face2 face3 face4 face5 face6 : ℕ),
    face1 < face2 ∧ face2 < face3 ∧ face3 < face4 ∧ face4 < face5 ∧ face5 < face6 ∧
    (face2 - face1 ≥ 2) ∧ (face3 - face2 ≥ 2) ∧ (face4 - face3 ≥ 2) ∧ (face5 - face4 ≥ 2) ∧ (face6 - face5 ≥ 2) ∧
    (face3 ≠ face1 + 2) ∧ (face4 ≠ face2 + 2) ∧ (face5 ≠ face3 + 2) ∧ (face6 ≠ face4 + 2)
    → face1 + face2 + face3 + face4 + face5 + face6 = 27 :=
by {
  sorry
}

end die_total_dots_l88_88155


namespace necessary_but_not_sufficient_condition_l88_88730

-- Given conditions and translated inequalities
variable {x : ℝ}
variable (h_pos : 0 < x) (h_bound : x < π / 2)
variable (h_sin_pos : 0 < Real.sin x) (h_sin_bound : Real.sin x < 1)

-- Define the inequalities we are dealing with
def ineq_1 (x : ℝ) := Real.sqrt x - 1 / Real.sin x < 0
def ineq_2 (x : ℝ) := 1 / Real.sin x - x > 0

-- The main proof statement
theorem necessary_but_not_sufficient_condition 
  (h1 : ineq_1 x) 
  (hx : 0 < x) (hπ : x < π/2) : 
  ineq_2 x → False := by
  sorry

end necessary_but_not_sufficient_condition_l88_88730


namespace basketball_students_l88_88924

variable (C B_inter_C B_union_C B : ℕ)

theorem basketball_students (hC : C = 5) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 9) (hInclusionExclusion : B_union_C = B + C - B_inter_C) : B = 7 := by
  sorry

end basketball_students_l88_88924


namespace chess_club_members_l88_88422

theorem chess_club_members {n : ℤ} (h10 : n % 10 = 6) (h11 : n % 11 = 6) (rng : 300 ≤ n ∧ n ≤ 400) : n = 336 :=
  sorry

end chess_club_members_l88_88422


namespace find_value_of_a_l88_88158

-- Definitions based on the conditions
def x (k : ℕ) : ℕ := 3 * k
def y (k : ℕ) : ℕ := 4 * k
def z (k : ℕ) : ℕ := 6 * k

-- Setting up the sum equation
def sum_eq_52 (k : ℕ) : Prop := x k + y k + z k = 52

-- Defining the y equation
def y_eq (a : ℚ) (k : ℕ) : Prop := y k = 15 * a + 5

-- Stating the main problem
theorem find_value_of_a (a : ℚ) (k : ℕ) : sum_eq_52 k → y_eq a k → a = 11 / 15 := by
  sorry

end find_value_of_a_l88_88158


namespace initial_money_l88_88068

theorem initial_money (cost_of_candy_bar : ℕ) (change_received : ℕ) (initial_money : ℕ) 
  (h_cost : cost_of_candy_bar = 45) (h_change : change_received = 5) :
  initial_money = cost_of_candy_bar + change_received :=
by
  -- here is the place for the proof which is not needed
  sorry

end initial_money_l88_88068


namespace sunil_total_amount_back_l88_88180

theorem sunil_total_amount_back 
  (CI : ℝ) (P : ℝ) (r : ℝ) (t : ℕ) (total_amount : ℝ) 
  (h1 : CI = 2828.80) 
  (h2 : r = 8) 
  (h3 : t = 2) 
  (h4 : CI = P * ((1 + r / 100) ^ t - 1)) : 
  total_amount = P + CI → 
  total_amount = 19828.80 :=
by
  sorry

end sunil_total_amount_back_l88_88180


namespace fraction_of_yard_occupied_l88_88075

/-
Proof Problem: Given a rectangular yard that measures 30 meters by 8 meters and contains
an isosceles trapezoid-shaped flower bed with parallel sides measuring 14 meters and 24 meters,
and a height of 6 meters, prove that the fraction of the yard occupied by the flower bed is 19/40.
-/

theorem fraction_of_yard_occupied (length_yard width_yard b1 b2 h area_trapezoid area_yard : ℝ) 
  (h_length_yard : length_yard = 30) 
  (h_width_yard : width_yard = 8) 
  (h_b1 : b1 = 14) 
  (h_b2 : b2 = 24) 
  (h_height_trapezoid : h = 6) 
  (h_area_trapezoid : area_trapezoid = (1/2) * (b1 + b2) * h) 
  (h_area_yard : area_yard = length_yard * width_yard) : 
  area_trapezoid / area_yard = 19 / 40 := 
by {
  -- Follow-up steps to prove the statement would go here
  sorry
}

end fraction_of_yard_occupied_l88_88075


namespace find_M_l88_88457

theorem find_M :
  (∃ M: ℕ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M) → M = 551 :=
by
  sorry

end find_M_l88_88457


namespace parabola_relationship_l88_88085

noncomputable def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem parabola_relationship (a b m n t : ℝ) (ha : a ≠ 0)
  (h1 : 3 * a + b > 0) (h2 : a + b < 0)
  (hm : parabola a b (-3) = m)
  (hn : parabola a b 2 = n)
  (ht : parabola a b 4 = t) :
  n < t ∧ t < m :=
by
  sorry

end parabola_relationship_l88_88085


namespace sum_of_c_and_d_l88_88789

theorem sum_of_c_and_d (c d : ℝ) 
  (h1 : ∀ x, x ≠ 2 ∧ x ≠ -1 → x^2 + c * x + d ≠ 0)
  (h_asymp_2 : 2^2 + c * 2 + d = 0)
  (h_asymp_neg1 : (-1)^2 + c * (-1) + d = 0) :
  c + d = -3 :=
by 
  -- Proof placeholder
  sorry

end sum_of_c_and_d_l88_88789


namespace no_tangent_line_l88_88566

-- Define the function f(x) = x^3 - 3ax
def f (a x : ℝ) : ℝ := x^3 - 3 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

-- Proposition stating no b exists in ℝ such that y = -x + b is tangent to f
theorem no_tangent_line (a : ℝ) (H : ∀ b : ℝ, ¬ ∃ x : ℝ, f' a x = -1) : a < 1 / 3 :=
by
  sorry

end no_tangent_line_l88_88566


namespace no_valid_pair_for_tangential_quadrilateral_l88_88067

theorem no_valid_pair_for_tangential_quadrilateral (a d : ℝ) (h : d > 0) :
  ¬((∃ a d, a + (a + 2 * d) = (a + d) + (a + 3 * d))) :=
by
  sorry

end no_valid_pair_for_tangential_quadrilateral_l88_88067


namespace problem_solution_l88_88479

theorem problem_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := 
  sorry

end problem_solution_l88_88479


namespace num_members_in_league_l88_88153

theorem num_members_in_league :
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  num_members = 150 :=
by
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  sorry

end num_members_in_league_l88_88153


namespace expression_for_A_l88_88288

theorem expression_for_A (A k : ℝ)
  (h : ∀ k : ℝ, Ax^2 + 6 * k * x + 2 = 0 → k = 0.4444444444444444 → (6 * k)^2 - 4 * A * 2 = 0) :
  A = 9 * k^2 / 2 := 
sorry

end expression_for_A_l88_88288


namespace find_f_l88_88931

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sqrt x + 4) = x + 8 * Real.sqrt x) :
  ∀ (x : ℝ), x ≥ 4 → f x = x^2 - 16 :=
by
  sorry

end find_f_l88_88931


namespace largest_x_quadratic_inequality_l88_88959

theorem largest_x_quadratic_inequality : 
  ∃ (x : ℝ), (x^2 - 10 * x + 24 ≤ 0) ∧ (∀ y, (y^2 - 10 * y + 24 ≤ 0) → y ≤ x) :=
sorry

end largest_x_quadratic_inequality_l88_88959


namespace compound_interest_calculation_l88_88280

noncomputable def compoundInterest (P r t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simpleInterest (P r t : ℝ) : ℝ :=
  P * r * t

theorem compound_interest_calculation :
  ∃ P : ℝ, simpleInterest P 0.10 2 = 600 ∧ compoundInterest P 0.10 2 = 630 :=
by
  sorry

end compound_interest_calculation_l88_88280


namespace smaller_number_of_ratio_4_5_lcm_180_l88_88691

theorem smaller_number_of_ratio_4_5_lcm_180 {a b : ℕ} (h_ratio : 4 * b = 5 * a) (h_lcm : Nat.lcm a b = 180) : a = 144 :=
by
  sorry

end smaller_number_of_ratio_4_5_lcm_180_l88_88691


namespace M_intersection_N_eq_M_l88_88980

def is_element_of_M (y : ℝ) : Prop := ∃ x : ℝ, y = 2^x
def is_element_of_N (y : ℝ) : Prop := ∃ x : ℝ, y = x^2

theorem M_intersection_N_eq_M : {y | is_element_of_M y} ∩ {y | is_element_of_N y} = {y | is_element_of_M y} :=
by
  sorry

end M_intersection_N_eq_M_l88_88980


namespace ab_cd_eq_one_l88_88463

theorem ab_cd_eq_one (a b c d : ℕ) (p : ℕ) 
  (h_div_a : a % p = 0)
  (h_div_b : b % p = 0)
  (h_div_c : c % p = 0)
  (h_div_d : d % p = 0)
  (h_div_ab_cd : (a * b - c * d) % p = 0) : 
  (a * b - c * d) = 1 :=
sorry

end ab_cd_eq_one_l88_88463


namespace fraction_sum_is_five_l88_88940

noncomputable def solve_fraction_sum (x y z : ℝ) : Prop :=
  (x + 1/y = 5) ∧ (y + 1/z = 2) ∧ (z + 1/x = 3) ∧ 0 < x ∧ 0 < y ∧ 0 < z → 
  (x / y + y / z + z / x = 5)
    
theorem fraction_sum_is_five (x y z : ℝ) : solve_fraction_sum x y z :=
  sorry

end fraction_sum_is_five_l88_88940


namespace john_has_hours_to_spare_l88_88708

def total_wall_area (num_walls : ℕ) (wall_width wall_height : ℕ) : ℕ :=
  num_walls * wall_width * wall_height

def time_to_paint_area (area : ℕ) (rate_per_square_meter_in_minutes : ℕ) : ℕ :=
  area * rate_per_square_meter_in_minutes

def minutes_to_hours (minutes : ℕ) : ℕ :=
  minutes / 60

theorem john_has_hours_to_spare 
  (num_walls : ℕ) (wall_width wall_height : ℕ)
  (rate_per_square_meter_in_minutes : ℕ) (total_available_hours : ℕ)
  (to_spare_hours : ℕ)
  (h : total_wall_area num_walls wall_width wall_height = num_walls * wall_width * wall_height)
  (h1 : time_to_paint_area (num_walls * wall_width * wall_height) rate_per_square_meter_in_minutes = num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes)
  (h2 : minutes_to_hours (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) = (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes) / 60)
  (h3 : total_available_hours = 10) 
  (h4 : to_spare_hours = total_available_hours - (num_walls * wall_width * wall_height * rate_per_square_meter_in_minutes / 60)) : 
  to_spare_hours = 5 := 
sorry

end john_has_hours_to_spare_l88_88708


namespace sqrt_continued_fraction_l88_88506

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l88_88506


namespace solve_system_of_inequalities_l88_88487

theorem solve_system_of_inequalities (x : ℝ) : 
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2) → -2 < x ∧ x < -1 :=
by {
  sorry
}

end solve_system_of_inequalities_l88_88487


namespace correct_eq_count_l88_88093

-- Define the correctness of each expression
def eq1 := (∀ x : ℤ, (-2 * x)^3 = 2 * x^3 = false)
def eq2 := (∀ a : ℤ, a^2 * a^3 = a^3 = false)
def eq3 := (∀ x : ℤ, (-x)^9 / (-x)^3 = x^6 = true)
def eq4 := (∀ a : ℤ, (-3 * a^2)^3 = -9 * a^6 = false)

-- Define the condition that there are exactly one correct equation
def num_correct_eqs := (1 = 1)

-- The theorem statement, proving the count of correct equations is 1
theorem correct_eq_count : eq1 → eq2 → eq3 → eq4 → num_correct_eqs :=
  by intros; sorry

end correct_eq_count_l88_88093


namespace area_of_rectangle_l88_88307

variables {group_interval rate : ℝ}

theorem area_of_rectangle (length_of_small_rectangle : ℝ) (height_of_small_rectangle : ℝ) :
  (length_of_small_rectangle = group_interval) → (height_of_small_rectangle = rate / group_interval) →
  length_of_small_rectangle * height_of_small_rectangle = rate :=
by
  intros h_length h_height
  rw [h_length, h_height]
  exact mul_div_cancel' rate (by sorry)

end area_of_rectangle_l88_88307


namespace monotonicity_and_extrema_of_f_l88_88285

noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

theorem monotonicity_and_extrema_of_f :
  (∀ (x_1 x_2 : ℝ), x_1 ∈ Set.Icc (-1 : ℝ) 2 → x_2 ∈ Set.Icc (-1 : ℝ) 2 → x_1 < x_2 → f x_1 < f x_2) ∧ 
  (f (-1) = -1) ∧ 
  (f 2 = 8) :=
by
  sorry

end monotonicity_and_extrema_of_f_l88_88285


namespace sale_in_fifth_month_l88_88427

theorem sale_in_fifth_month (a1 a2 a3 a4 a5 a6 avg : ℝ)
  (h1 : a1 = 5420) (h2 : a2 = 5660) (h3 : a3 = 6200) (h4 : a4 = 6350) (h6 : a6 = 6470) (h_avg : avg = 6100) :
  a5 = 6500 :=
by
  sorry

end sale_in_fifth_month_l88_88427


namespace jacks_remaining_capacity_l88_88278

noncomputable def jacks_basket_full_capacity : ℕ := 12
noncomputable def jills_basket_full_capacity : ℕ := 2 * jacks_basket_full_capacity
noncomputable def jacks_current_apples (x : ℕ) : Prop := 3 * x = jills_basket_full_capacity

theorem jacks_remaining_capacity {x : ℕ} (hx : jacks_current_apples x) :
  jacks_basket_full_capacity - x = 4 :=
by sorry

end jacks_remaining_capacity_l88_88278


namespace impossible_to_load_two_coins_l88_88990

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l88_88990


namespace initial_ratio_l88_88553

variable (A B : ℕ) (a b : ℕ)
variable (h1 : B = 6)
variable (h2 : (A + 2) / (B + 2) = 3 / 2)

theorem initial_ratio (A B : ℕ) (h1 : B = 6) (h2 : (A + 2) / (B + 2) = 3 / 2) : A / B = 5 / 3 := 
by 
    sorry

end initial_ratio_l88_88553


namespace perimeter_of_new_rectangle_l88_88201

-- Definitions based on conditions
def side_of_square : ℕ := 8
def length_of_rectangle : ℕ := 8
def breadth_of_rectangle : ℕ := 4

-- Perimeter calculation
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the problem
theorem perimeter_of_new_rectangle :
  perimeter (side_of_square + length_of_rectangle) side_of_square = 48 :=
  by sorry

end perimeter_of_new_rectangle_l88_88201


namespace find_smaller_angle_l88_88129

theorem find_smaller_angle (x : ℝ) (h1 : (x + (x + 18) = 180)) : x = 81 := 
by 
  sorry

end find_smaller_angle_l88_88129


namespace sum_of_reciprocals_l88_88030

theorem sum_of_reciprocals (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 31 / 21) :=
sorry

end sum_of_reciprocals_l88_88030


namespace shaded_rectangle_area_l88_88394

theorem shaded_rectangle_area (side_length : ℝ) (x y : ℝ) 
  (h1 : side_length = 42) 
  (h2 : 4 * x + 2 * y = 168 - 4 * x) 
  (h3 : 2 * (side_length - y) + 2 * x = 168 - 4 * x)
  (h4 : 2 * (2 * x + y) = 168 - 4 * x) 
  (h5 : x = 18) :
  (2 * x) * (4 * x - (side_length - y)) = 540 := 
by
  sorry

end shaded_rectangle_area_l88_88394


namespace upstream_speed_l88_88949

variable (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)

def speed_of_man_in_still_water := V_m = 35
def speed_of_man_downstream := V_downstream = 45
def speed_of_man_upstream := V_upstream = 25

theorem upstream_speed
  (h1: speed_of_man_in_still_water V_m)
  (h2: speed_of_man_downstream V_downstream)
  : speed_of_man_upstream V_upstream :=
by
  -- Placeholder for the proof
  sorry

end upstream_speed_l88_88949


namespace length_of_field_l88_88072

theorem length_of_field (width : ℕ) (distance_covered : ℕ) (n : ℕ) (L : ℕ) 
  (h1 : width = 15) 
  (h2 : distance_covered = 540) 
  (h3 : n = 3) 
  (h4 : 2 * (L + width) = perimeter)
  (h5 : n * perimeter = distance_covered) : 
  L = 75 :=
by 
  sorry

end length_of_field_l88_88072


namespace problem_l88_88527

def polynomial (x : ℝ) : ℝ := 9 * x ^ 3 - 27 * x + 54

theorem problem (a b c : ℝ) 
  (h_roots : polynomial a = 0 ∧ polynomial b = 0 ∧ polynomial c = 0) :
  (a + b) ^ 3 + (b + c) ^ 3 + (c + a) ^ 3 = 18 :=
by
  sorry

end problem_l88_88527


namespace necessary_without_sufficient_for_parallel_lines_l88_88764

noncomputable def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 2 = 0
noncomputable def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 1 = 0

theorem necessary_without_sufficient_for_parallel_lines :
  (∀ (a : ℝ), a = 2 → (∀ (x y : ℝ), line1 a x y → line2 a x y)) ∧ 
  ¬ (∀ (a : ℝ), (∀ (x y : ℝ), line1 a x y → line2 a x y) → a = 2) :=
sorry

end necessary_without_sufficient_for_parallel_lines_l88_88764


namespace brush_length_percentage_increase_l88_88537

-- Define the length of Carla's brush in inches
def carla_brush_length_in_inches : ℝ := 12

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the length of Carmen's brush in centimeters
def carmen_brush_length_in_cm : ℝ := 45

-- Noncomputable definition to calculate the percentage increase
noncomputable def percentage_increase : ℝ :=
  let carla_brush_length_in_cm := carla_brush_length_in_inches * inch_to_cm
  (carmen_brush_length_in_cm - carla_brush_length_in_cm) / carla_brush_length_in_cm * 100

-- Statement to prove the percentage increase is 47.6%
theorem brush_length_percentage_increase :
  percentage_increase = 47.6 :=
sorry

end brush_length_percentage_increase_l88_88537


namespace find_x_l88_88835

theorem find_x (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 :=
sorry

end find_x_l88_88835


namespace square_implies_increasing_l88_88726

def seq (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n > 1, 
    ((a n - 2 > 0 ∧ ¬(∃ m < n, a m = a n - 2)) → a (n + 1) = a n - 2) ∧
    ((a n - 2 ≤ 0 ∨ ∃ m < n, a m = a n - 2) → a (n + 1) = a n + 3)

theorem square_implies_increasing (a : ℕ → ℤ) (n : ℕ) (h_seq : seq a) 
  (h_square : ∃ k, a n = k^2) (h_n_pos : n > 1) : 
  a n > a (n - 1) :=
sorry

end square_implies_increasing_l88_88726


namespace equal_cost_number_of_minutes_l88_88678

theorem equal_cost_number_of_minutes :
  ∃ m : ℝ, (8 + 0.25 * m = 12 + 0.20 * m) ∧ m = 80 :=
by
  sorry

end equal_cost_number_of_minutes_l88_88678


namespace total_distance_l88_88169

noncomputable def total_distance_covered 
  (radius1 radius2 radius3 : ℝ) 
  (rev1 rev2 rev3 : ℕ) : ℝ :=
  let π := Real.pi
  let circumference r := 2 * π * r
  let distance r rev := circumference r * rev
  distance radius1 rev1 + distance radius2 rev2 + distance radius3 rev3

theorem total_distance
  (h1 : radius1 = 20.4) 
  (h2 : radius2 = 15.3) 
  (h3 : radius3 = 25.6) 
  (h4 : rev1 = 400) 
  (h5 : rev2 = 320) 
  (h6 : rev3 = 500) :
  total_distance_covered 20.4 15.3 25.6 400 320 500 = 162436.6848 := 
sorry

end total_distance_l88_88169


namespace sum_of_smallest_two_consecutive_numbers_l88_88062

theorem sum_of_smallest_two_consecutive_numbers (n : ℕ) (h : n * (n + 1) * (n + 2) = 210) : n + (n + 1) = 11 :=
sorry

end sum_of_smallest_two_consecutive_numbers_l88_88062


namespace chord_length_on_parabola_eq_five_l88_88592

theorem chord_length_on_parabola_eq_five
  (A B : ℝ × ℝ)
  (hA : A.snd ^ 2 = 4 * A.fst)
  (hB : B.snd ^ 2 = 4 * B.fst)
  (hM : A.fst + B.fst = 3 ∧ A.snd + B.snd = 2 
     ∧ A.fst - B.fst = 0 ∧ A.snd - B.snd = 0) :
  dist A B = 5 :=
by
  -- Proof goes here
  sorry

end chord_length_on_parabola_eq_five_l88_88592


namespace wall_building_l88_88460

-- Definitions based on conditions
def total_work (m d : ℕ) : ℕ := m * d

-- Prove that if 30 men including 10 twice as efficient men work for 3 days, they can build the wall
theorem wall_building (m₁ m₂ d₁ d₂ : ℕ) (h₁ : total_work m₁ d₁ = total_work m₂ d₂) (m₁_eq : m₁ = 20) (d₁_eq : d₁ = 6) 
(h₂ : m₂ = 40) : d₂ = 3 :=
  sorry

end wall_building_l88_88460


namespace prob_ending_game_after_five_distribution_and_expectation_l88_88171

-- Define the conditions
def shooting_accuracy_rate : ℚ := 2 / 3
def game_clear_coupon : ℕ := 9
def game_fail_coupon : ℕ := 3
def game_no_clear_no_fail_coupon : ℕ := 6

-- Define the probabilities for ending the game after 5 shots
def ending_game_after_five : ℚ := (shooting_accuracy_rate^2 * (1 - shooting_accuracy_rate)^3 * 2) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate))

-- Define the distribution table
def P_clear : ℚ := (shooting_accuracy_rate^3) + (shooting_accuracy_rate^3 * (1 - shooting_accuracy_rate)) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate) * 2)
def P_fail : ℚ := ((1 - shooting_accuracy_rate)^2) + ((1 - shooting_accuracy_rate)^2 * shooting_accuracy_rate * 2) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^2 * 3) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^3)
def P_neither : ℚ := 1 - P_clear - P_fail

-- Expected value calculation
def expectation : ℚ := (P_fail * game_fail_coupon) + (P_neither * game_no_clear_no_fail_coupon) + (P_clear * game_clear_coupon)

-- The Part I proof statement
theorem prob_ending_game_after_five : ending_game_after_five = 8 / 81 :=
by
  sorry

-- The Part II proof statement
theorem distribution_and_expectation (X : ℕ → ℚ) :
  (X game_fail_coupon = 233 / 729) ∧
  (X game_no_clear_no_fail_coupon = 112 / 729) ∧
  (X game_clear_coupon = 128 / 243) ∧
  (expectation = 1609 / 243) :=
by
  sorry

end prob_ending_game_after_five_distribution_and_expectation_l88_88171


namespace original_bill_amount_l88_88795

/-- 
If 8 people decided to split the restaurant bill evenly and each paid $314.15 after rounding
up to the nearest cent, then the original bill amount was $2513.20.
-/
theorem original_bill_amount (n : ℕ) (individual_share : ℝ) (total_amount : ℝ) 
  (h1 : n = 8) (h2 : individual_share = 314.15) 
  (h3 : total_amount = n * individual_share) : 
  total_amount = 2513.20 :=
by
  sorry

end original_bill_amount_l88_88795


namespace task_completion_time_l88_88029

variable (x : Real) (y : Real)

theorem task_completion_time :
  (1 / 16) * y + (1 / 12) * x = 1 ∧ y + 5 = 8 → x = 3 ∧ y = 3 :=
  by {
    sorry 
  }

end task_completion_time_l88_88029


namespace diamond_expression_evaluation_l88_88536

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem diamond_expression_evaluation :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 :=
by {
    sorry
}

end diamond_expression_evaluation_l88_88536


namespace win_lottery_amount_l88_88847

theorem win_lottery_amount (W : ℝ) (cond1 : W * 0.20 + 5 = 35) : W = 50 := by
  sorry

end win_lottery_amount_l88_88847


namespace arithmetic_sequence_probability_l88_88513

def favorable_sequences : List (List ℕ) :=
  [[1, 2, 3], [1, 3, 5], [2, 3, 4], [2, 4, 6], [3, 4, 5], [4, 5, 6], 
   [3, 2, 1], [5, 3, 1], [4, 3, 2], [6, 4, 2], [5, 4, 3], [6, 5, 4], 
   [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]

def total_outcomes : ℕ := 216
def favorable_outcomes : ℕ := favorable_sequences.length

theorem arithmetic_sequence_probability : (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by
  sorry

end arithmetic_sequence_probability_l88_88513


namespace dawn_annual_salary_l88_88855

variable (M : ℝ)

theorem dawn_annual_salary (h1 : 0.10 * M = 400) : M * 12 = 48000 := by
  sorry

end dawn_annual_salary_l88_88855


namespace sum_of_endpoints_l88_88979

noncomputable def triangle_side_length (PQ QR PR QS PS : ℝ) (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = (PS / QS)) : ℝ :=
  if 4 < PR ∧ PR < 18 then 4 + 18 else 0

theorem sum_of_endpoints {PQ PR QS PS : ℝ} (h1 : PQ = 12) (h2 : QS = 4)
  (h3 : (PQ / PR) = ( PS / QS)) :
  triangle_side_length PQ 0 PR QS PS h1 h2 h3 = 22 := by
  sorry

end sum_of_endpoints_l88_88979


namespace circle_area_l88_88128

theorem circle_area : 
    (∃ x y : ℝ, 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
    (∃ A : ℝ, A = (7 / 4) * Real.pi) :=
by
  sorry

end circle_area_l88_88128


namespace sum_of_roots_l88_88798

open Real

theorem sum_of_roots (r s : ℝ) (P : ℝ → ℝ) (Q : ℝ × ℝ) (m : ℝ) :
  (∀ (x : ℝ), P x = x^2) → 
  Q = (20, 14) → 
  (∀ m : ℝ, (m^2 - 80 * m + 56 < 0) ↔ (r < m ∧ m < s)) →
  r + s = 80 :=
by {
  -- sketched proof goes here
  sorry
}

end sum_of_roots_l88_88798


namespace tan_six_theta_eq_l88_88174

theorem tan_six_theta_eq (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (6 * θ) = 21 / 8 :=
by
  sorry

end tan_six_theta_eq_l88_88174


namespace center_of_circle_l88_88826

theorem center_of_circle (x y : ℝ) : 
  (x^2 + y^2 = 6 * x - 10 * y + 9) → 
  (∃ c : ℝ × ℝ, c = (3, -5) ∧ c.1 + c.2 = -2) :=
by
  sorry

end center_of_circle_l88_88826


namespace line_equation_l88_88667

theorem line_equation (m b : ℝ) (h_slope : m = 3) (h_intercept : b = 4) :
  3 * x - y + 4 = 0 :=
by
  sorry

end line_equation_l88_88667


namespace sum_squares_of_roots_l88_88508

def a := 8
def b := 12
def c := -14

theorem sum_squares_of_roots : (b^2 - 2 * a * c)/(a^2) = 23/4 := by
  sorry

end sum_squares_of_roots_l88_88508


namespace exists_two_digit_pair_product_l88_88017

theorem exists_two_digit_pair_product (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (hprod : a * b = 8670) : a * b = 8670 :=
by
  exact hprod

end exists_two_digit_pair_product_l88_88017


namespace collinear_vectors_value_m_l88_88697

theorem collinear_vectors_value_m (m : ℝ) : 
  (∃ k : ℝ, (2*m = k * (m - 1)) ∧ (3 = k)) → m = 3 :=
by
  sorry

end collinear_vectors_value_m_l88_88697


namespace min_value_f_l88_88396

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y

theorem min_value_f : ∃ x y : ℝ, f x y = -9 / 5 :=
sorry

end min_value_f_l88_88396


namespace total_donation_l88_88189

theorem total_donation {carwash_proceeds bake_sale_proceeds mowing_lawn_proceeds : ℝ}
    (hc : carwash_proceeds = 100)
    (hb : bake_sale_proceeds = 80)
    (hl : mowing_lawn_proceeds = 50)
    (carwash_donation : ℝ := 0.9 * carwash_proceeds)
    (bake_sale_donation : ℝ := 0.75 * bake_sale_proceeds)
    (mowing_lawn_donation : ℝ := 1.0 * mowing_lawn_proceeds) :
    carwash_donation + bake_sale_donation + mowing_lawn_donation = 200 := by
  sorry

end total_donation_l88_88189


namespace totalPoundsOfFoodConsumed_l88_88615

def maxConsumptionPerGuest : ℝ := 2.5
def minNumberOfGuests : ℕ := 165

theorem totalPoundsOfFoodConsumed : 
    maxConsumptionPerGuest * (minNumberOfGuests : ℝ) = 412.5 := by
  sorry

end totalPoundsOfFoodConsumed_l88_88615


namespace four_integers_product_sum_l88_88275

theorem four_integers_product_sum (a b c d : ℕ) (h1 : a * b * c * d = 2002) (h2 : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l88_88275


namespace required_tiles_0_4m_l88_88830

-- Defining given conditions
def num_tiles_0_3m : ℕ := 720
def side_length_0_3m : ℝ := 0.3
def side_length_0_4m : ℝ := 0.4

-- The problem statement translated to Lean 4
theorem required_tiles_0_4m : (side_length_0_4m ^ 2) * (405 : ℝ) = (side_length_0_3m ^ 2) * (num_tiles_0_3m : ℝ) := 
by
  -- Skipping the proof
  sorry

end required_tiles_0_4m_l88_88830


namespace min_value_of_expression_l88_88003

theorem min_value_of_expression (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ min_value, min_value = 9 / 2 ∧ ∀ z, z = (1 / (x + 1) + 4 / y) → z ≥ min_value :=
sorry

end min_value_of_expression_l88_88003


namespace polynomial_coeff_divisible_by_5_l88_88870

theorem polynomial_coeff_divisible_by_5
  (a b c : ℤ)
  (h : ∀ k : ℤ, (a * k^2 + b * k + c) % 5 = 0) :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 :=
by
  sorry

end polynomial_coeff_divisible_by_5_l88_88870


namespace jack_bill_age_difference_l88_88016

def jack_bill_ages_and_difference (a b : ℕ) :=
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  (a + b = 2) ∧ (7 * a - 29 * b = 14) → jack_age - bill_age = 18

theorem jack_bill_age_difference (a b : ℕ) (h₀ : a + b = 2) (h₁ : 7 * a - 29 * b = 14) : 
  let jack_age := 10 * a + b
  let bill_age := 10 * b + a
  jack_age - bill_age = 18 :=
by {
  sorry
}

end jack_bill_age_difference_l88_88016


namespace sqrt_7_minus_a_l88_88198

theorem sqrt_7_minus_a (a : ℝ) (h : a = -1) : Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_7_minus_a_l88_88198


namespace max_area_of_cone_l88_88543

noncomputable def max_cross_sectional_area (l θ : ℝ) : ℝ := (1/2) * l^2 * Real.sin θ

theorem max_area_of_cone :
  (∀ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) → max_cross_sectional_area 3 θ ≤ (9 / 2))
  ∧ (∃ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) ∧ max_cross_sectional_area 3 θ = (9 / 2)) := 
by
  sorry

end max_area_of_cone_l88_88543


namespace percentage_increase_in_surface_area_l88_88215

variable (a : ℝ)

theorem percentage_increase_in_surface_area (ha : a > 0) :
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  percentage_increase = 125 := 
by 
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  sorry

end percentage_increase_in_surface_area_l88_88215


namespace polygon_sides_sum_l88_88363

theorem polygon_sides_sum (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
by
  sorry

end polygon_sides_sum_l88_88363


namespace rectangle_length_l88_88137

theorem rectangle_length (L W : ℝ) (h1 : L = 4 * W) (h2 : L * W = 100) : L = 20 :=
by
  sorry

end rectangle_length_l88_88137


namespace total_cost_l88_88058

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 4
def num_sodas : ℕ := 5

theorem total_cost : (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 31 := by
  sorry

end total_cost_l88_88058


namespace function_relationship_l88_88033

variable {A B : Type} [Nonempty A] [Nonempty B]
variable (f : A → B) 

def domain (f : A → B) : Set A := {a | ∃ b, f a = b}
def range (f : A → B) : Set B := {b | ∃ a, f a = b}

theorem function_relationship (M : Set A) (N : Set B) (hM : M = Set.univ)
                              (hN : N = range f) : M = Set.univ ∧ N ⊆ Set.univ :=
  sorry

end function_relationship_l88_88033


namespace squirrel_acorns_l88_88591

theorem squirrel_acorns :
  ∃ (c s r : ℕ), (4 * c = 5 * s) ∧ (3 * r = 4 * c) ∧ (r = s + 3) ∧ (5 * s = 40) :=
by
  sorry

end squirrel_acorns_l88_88591


namespace find_x_plus_y_l88_88582

theorem find_x_plus_y (x y : ℝ) (hx : abs x - x + y = 6) (hy : x + abs y + y = 16) : x + y = 10 :=
sorry

end find_x_plus_y_l88_88582


namespace intersection_complement_l88_88392

open Set

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Theorem
theorem intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end intersection_complement_l88_88392


namespace triangle_with_positive_area_l88_88351

noncomputable def num_triangles_with_A (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) : ℕ :=
  let points_excluding_A := total_points.erase A
  let total_pairs := points_excluding_A.card.choose 2
  let collinear_pairs := 20  -- Derived from the problem; in practice this would be calculated
  total_pairs - collinear_pairs

theorem triangle_with_positive_area (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) (h : total_points.card = 25):
  num_triangles_with_A total_points A = 256 :=
by
  sorry

end triangle_with_positive_area_l88_88351


namespace parrots_false_statements_l88_88878

theorem parrots_false_statements (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 140 ∧ 
    (∀ statements : ℕ → Prop, 
      (statements 0 = false) ∧ 
      (∀ i : ℕ, 1 ≤ i → i < n → 
          (statements i = true → 
            (∃ fp : ℕ, fp < i ∧ 7 * (fp + 1) > 10 * i)))) := 
by
  sorry

end parrots_false_statements_l88_88878


namespace find_b_for_parallel_lines_l88_88283

theorem find_b_for_parallel_lines :
  (∀ (b : ℝ), (∃ (f g : ℝ → ℝ),
  (∀ x, f x = 3 * x + b) ∧
  (∀ x, g x = (b + 9) * x - 2) ∧
  (∀ x, f x = g x → False)) →
  b = -6) :=
sorry

end find_b_for_parallel_lines_l88_88283


namespace greatest_pq_plus_r_l88_88827

theorem greatest_pq_plus_r (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h : p * q + q * r + r * p = 2016) : 
  pq + r ≤ 1008 :=
sorry

end greatest_pq_plus_r_l88_88827


namespace remainder_of_sum_mod_13_l88_88755

theorem remainder_of_sum_mod_13 {a b c d e : ℕ} 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7) 
  (h4 : d % 13 = 9) 
  (h5 : e % 13 = 11) : 
  (a + b + c + d + e) % 13 = 9 :=
by
  sorry

end remainder_of_sum_mod_13_l88_88755


namespace area_of_rectangle_l88_88991

-- Given conditions
def shadedSquareArea : ℝ := 4
def nonShadedSquareArea : ℝ := shadedSquareArea
def largerSquareArea : ℝ := 4 * 4  -- Since the side length is twice the previous squares

-- Problem statement
theorem area_of_rectangle (shadedSquareArea nonShadedSquareArea largerSquareArea : ℝ) :
  shadedSquareArea + nonShadedSquareArea + largerSquareArea = 24 :=
sorry

end area_of_rectangle_l88_88991


namespace find_x_l88_88737

noncomputable def inv_cubicroot (y x : ℝ) : ℝ := y * x^(1/3)

theorem find_x (x y : ℝ) (h1 : ∃ k, inv_cubicroot 2 8 = k) (h2 : y = 8) : x = 1 / 8 :=
by
  sorry

end find_x_l88_88737


namespace percentage_distance_l88_88127

theorem percentage_distance (start : ℝ) (end_point : ℝ) (point : ℝ) (total_distance : ℝ)
  (distance_from_start : ℝ) :
  start = -55 → end_point = 55 → point = 5.5 → total_distance = end_point - start →
  distance_from_start = point - start →
  (distance_from_start / total_distance) * 100 = 55 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_distance_l88_88127


namespace bicycle_owners_no_car_l88_88922

-- Definitions based on the conditions in (a)
def total_adults : ℕ := 500
def bicycle_owners : ℕ := 450
def car_owners : ℕ := 120
def both_owners : ℕ := bicycle_owners + car_owners - total_adults

-- Proof problem statement
theorem bicycle_owners_no_car : (bicycle_owners - both_owners = 380) :=
by
  -- Placeholder proof
  sorry

end bicycle_owners_no_car_l88_88922


namespace calculate_r_l88_88404

def a := 0.24 * 450
def b := 0.62 * 250
def c := 0.37 * 720
def d := 0.38 * 100
def sum_bc := b + c
def diff := sum_bc - a
def r := diff / d

theorem calculate_r : r = 8.25 := by
  sorry

end calculate_r_l88_88404


namespace sum_of_zeros_l88_88190

-- Defining the conditions and the result
theorem sum_of_zeros (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) (a b c : ℝ)
  (h1 : f a = 0) (h2 : f b = 0) (h3 : f c = 0) : 
  a + b + c = 3 := 
by 
  sorry

end sum_of_zeros_l88_88190


namespace problem_statement_l88_88063

theorem problem_statement : 2017 - (1 / 2017) = (2018 * 2016) / 2017 :=
by
  sorry

end problem_statement_l88_88063


namespace trivia_team_total_points_l88_88164

def totalPoints : Nat := 182

def points_member_A : Nat := 3 * 2
def points_member_B : Nat := 5 * 4 + 1 * 6
def points_member_C : Nat := 2 * 6
def points_member_D : Nat := 4 * 2 + 2 * 4
def points_member_E : Nat := 1 * 2 + 3 * 4
def points_member_F : Nat := 5 * 6
def points_member_G : Nat := 2 * 4 + 1 * 2
def points_member_H : Nat := 3 * 6 + 2 * 2
def points_member_I : Nat := 1 * 4 + 4 * 6
def points_member_J : Nat := 7 * 2 + 1 * 4

theorem trivia_team_total_points : 
  points_member_A + points_member_B + points_member_C + points_member_D + points_member_E + 
  points_member_F + points_member_G + points_member_H + points_member_I + points_member_J = totalPoints := 
by
  repeat { sorry }

end trivia_team_total_points_l88_88164


namespace area_increase_of_square_garden_l88_88323

theorem area_increase_of_square_garden
  (length : ℝ) (width : ℝ)
  (h_length : length = 60)
  (h_width : width = 20) :
  let perimeter := 2 * (length + width)
  let side_length := perimeter / 4
  let initial_area := length * width
  let square_area := side_length ^ 2
  square_area - initial_area = 400 :=
by
  sorry

end area_increase_of_square_garden_l88_88323


namespace hands_per_student_l88_88969

theorem hands_per_student (hands_without_peter : ℕ) (total_students : ℕ) (hands_peter : ℕ) 
  (h1 : hands_without_peter = 20) 
  (h2 : total_students = 11) 
  (h3 : hands_peter = 2) : 
  (hands_without_peter + hands_peter) / total_students = 2 :=
by
  sorry

end hands_per_student_l88_88969


namespace john_bought_six_bagels_l88_88998

theorem john_bought_six_bagels (b m : ℕ) (expenditure_in_dollars_whole : (90 * b + 60 * m) % 100 = 0) (total_items : b + m = 7) : 
b = 6 :=
by
  -- The proof goes here. For now, we skip it with sorry.
  sorry

end john_bought_six_bagels_l88_88998


namespace devin_teaching_years_l88_88676

theorem devin_teaching_years (total_years : ℕ) (tom_years : ℕ) (devin_years : ℕ) 
  (half_tom_years : ℕ)
  (h1 : total_years = 70) 
  (h2 : tom_years = 50)
  (h3 : total_years = tom_years + devin_years) 
  (h4 : half_tom_years = tom_years / 2) : 
  half_tom_years - devin_years = 5 :=
by
  sorry

end devin_teaching_years_l88_88676


namespace initial_total_quantity_l88_88134

theorem initial_total_quantity
  (x : ℝ)
  (milk_water_ratio : 5 / 9 = 5 * x / (3 * x + 12))
  (milk_juice_ratio : 5 / 8 = 5 * x / (4 * x + 6)) :
  5 * x + 3 * x + 4 * x = 24 :=
by
  sorry

end initial_total_quantity_l88_88134


namespace nested_fraction_eval_l88_88320

theorem nested_fraction_eval : (1 / (1 + (1 / (2 + (1 / (1 + (1 / 4))))))) = (14 / 19) :=
by
  sorry

end nested_fraction_eval_l88_88320


namespace base_seven_sum_of_digits_of_product_l88_88862

theorem base_seven_sum_of_digits_of_product :
  let a := 24
  let b := 30
  let product := a * b
  let base_seven_product := 105 -- The product in base seven notation
  let sum_of_digits (n : ℕ) : ℕ := n.digits 7 |> List.sum
  sum_of_digits base_seven_product = 6 :=
by
  sorry

end base_seven_sum_of_digits_of_product_l88_88862


namespace third_price_reduction_l88_88146

theorem third_price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : (original_price * (1 - x)^2 = final_price))
  (h2 : final_price = 100)
  (h3 : original_price = 100 / (1 - 0.19)) :
  (original_price * (1 - x)^3 = 90) :=
by
  sorry

end third_price_reduction_l88_88146


namespace smallest_k_for_mutual_criticism_l88_88395

-- Define a predicate that checks if a given configuration of criticisms lead to mutual criticism
def mutual_criticism_exists (deputies : ℕ) (k : ℕ) : Prop :=
  k ≥ 8 -- This is derived from the problem where k = 8 is the smallest k ensuring a mutual criticism

theorem smallest_k_for_mutual_criticism:
  mutual_criticism_exists 15 8 :=
by
  -- This is the theorem statement with the conditions and correct answer. The proof is omitted.
  sorry

end smallest_k_for_mutual_criticism_l88_88395


namespace total_paint_area_eq_1060_l88_88531

/-- Define the dimensions of the stable and chimney -/
def stable_width := 12
def stable_length := 15
def stable_height := 6
def chimney_width := 2
def chimney_length := 2
def chimney_height := 2

/-- Define the area to be painted computation -/

def wall_area (width length height : ℕ) : ℕ :=
  (width * height * 2) * 2 + (length * height * 2) * 2

def roof_area (width length : ℕ) : ℕ :=
  width * length

def ceiling_area (width length : ℕ) : ℕ :=
  width * length

def chimney_area (width length height : ℕ) : ℕ :=
  (4 * (width * height)) + (width * length)

def total_paint_area : ℕ :=
  wall_area stable_width stable_length stable_height +
  roof_area stable_width stable_length +
  ceiling_area stable_width stable_length +
  chimney_area chimney_width chimney_length chimney_height

/-- Goal: Prove that the total paint area is 1060 sq. yd -/
theorem total_paint_area_eq_1060 : total_paint_area = 1060 := by
  sorry

end total_paint_area_eq_1060_l88_88531


namespace principal_amount_borrowed_l88_88358

theorem principal_amount_borrowed
  (R : ℝ) (T : ℝ) (SI : ℝ) (P : ℝ) 
  (hR : R = 12) 
  (hT : T = 20) 
  (hSI : SI = 2100) 
  (hFormula : SI = (P * R * T) / 100) : 
  P = 875 := 
by 
  -- Assuming the initial steps 
  sorry

end principal_amount_borrowed_l88_88358


namespace star_result_l88_88890

-- Define the operation star
def star (m n p q : ℚ) := (m * p) * (n / q)

-- Given values
def a := (5 : ℚ) / 9
def b := (10 : ℚ) / 6

-- Condition to check
theorem star_result : star 5 9 10 6 = 75 := by
  sorry

end star_result_l88_88890


namespace pencils_per_student_l88_88477

-- Define the number of pens
def numberOfPens : ℕ := 1001

-- Define the number of pencils
def numberOfPencils : ℕ := 910

-- Define the maximum number of students
def maxNumberOfStudents : ℕ := 91

-- Using the given conditions, prove that each student gets 10 pencils
theorem pencils_per_student :
  (numberOfPencils / maxNumberOfStudents) = 10 :=
by sorry

end pencils_per_student_l88_88477


namespace steve_total_money_l88_88750

theorem steve_total_money
    (nickels : ℕ)
    (dimes : ℕ)
    (nickel_value : ℕ := 5)
    (dime_value : ℕ := 10)
    (cond1 : nickels = 2)
    (cond2 : dimes = nickels + 4) 
    : (nickels * nickel_value + dimes * dime_value) = 70 := by
  sorry

end steve_total_money_l88_88750


namespace matrix_pow_three_l88_88934

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_pow_three :
  A^3 = !![-4, 2; -2, 1] := by
  sorry

end matrix_pow_three_l88_88934


namespace people_in_rooms_l88_88078

theorem people_in_rooms (x y : ℕ) (h1 : x + y = 76) (h2 : x - 30 = y - 40) : x = 33 ∧ y = 43 := by
  sorry

end people_in_rooms_l88_88078


namespace percentage_of_knives_l88_88810

def initial_knives : Nat := 6
def initial_forks : Nat := 12
def initial_spoons : Nat := 3 * initial_knives
def traded_knives : Nat := 10
def traded_spoons : Nat := 6

theorem percentage_of_knives :
  100 * (initial_knives + traded_knives) / (initial_knives + initial_forks + initial_spoons - traded_spoons + traded_knives) = 40 := by
  sorry

end percentage_of_knives_l88_88810


namespace race_distance_l88_88034

theorem race_distance (D : ℝ)
  (A_time : D / 36 * 45 = D + 20) : 
  D = 80 :=
by
  sorry

end race_distance_l88_88034


namespace J_of_given_values_l88_88451

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_of_given_values : J 3 (-15) 10 = 49 / 30 := 
by 
  sorry

end J_of_given_values_l88_88451


namespace fraction_computation_l88_88188

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l88_88188


namespace minimum_additional_squares_needed_to_achieve_symmetry_l88_88777

def initial_grid : List (ℕ × ℕ) := [(1, 4), (4, 1)] -- Initial shaded squares

def is_symmetric (grid : List (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ), x ∈ grid → y ∈ grid →
    ((x.1 = 2 * 2 - y.1 ∧ x.2 = y.2) ∨
     (x.1 = y.1 ∧ x.2 = 5 - y.2) ∨
     (x.1 = 2 * 2 - y.1 ∧ x.2 = 5 - y.2))

def additional_squares_needed : ℕ :=
  6 -- As derived in the solution steps, 6 additional squares are needed to achieve symmetry

theorem minimum_additional_squares_needed_to_achieve_symmetry :
  ∀ (initial_shades : List (ℕ × ℕ)),
    initial_shades = initial_grid →
    ∃ (additional : List (ℕ × ℕ)),
      initial_shades ++ additional = symmetric_grid ∧
      additional.length = additional_squares_needed :=
by 
-- skip the proof
sorry

end minimum_additional_squares_needed_to_achieve_symmetry_l88_88777


namespace sum_of_divisors_of_11_squared_l88_88230

theorem sum_of_divisors_of_11_squared (a b c : ℕ) (h1 : a ∣ 11^2) (h2 : b ∣ 11^2) (h3 : c ∣ 11^2) (h4 : a * b * c = 11^2) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) :
  a + b + c = 23 :=
sorry

end sum_of_divisors_of_11_squared_l88_88230


namespace cookies_milk_conversion_l88_88393

theorem cookies_milk_conversion :
  (18 : ℕ) / (3 * 2 : ℕ) / (18 : ℕ) * (9 : ℕ) = (3 : ℕ) :=
by
  sorry

end cookies_milk_conversion_l88_88393


namespace prime_expression_integer_value_l88_88025

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_expression_integer_value (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  ∃ n, (p * q + p^p + q^q) % (p + q) = 0 → n = 3 :=
by
  sorry

end prime_expression_integer_value_l88_88025


namespace find_blue_sea_glass_pieces_l88_88704

-- Define all required conditions and the proof problem.
theorem find_blue_sea_glass_pieces (B : ℕ) : 
  let BlancheRed := 3
  let RoseRed := 9
  let DorothyRed := 2 * (BlancheRed + RoseRed)
  let DorothyBlue := 3 * B
  let DorothyTotal := 57
  DorothyTotal = DorothyRed + DorothyBlue → B = 11 :=
by {
  sorry
}

end find_blue_sea_glass_pieces_l88_88704


namespace ben_chairs_in_10_days_l88_88787

noncomputable def chairs_built_per_day (hours_per_shift : ℕ) (hours_per_chair : ℕ) : ℕ :=
  hours_per_shift / hours_per_chair

theorem ben_chairs_in_10_days 
  (hours_per_shift : ℕ)
  (hours_per_chair : ℕ)
  (days: ℕ)
  (h_shift: hours_per_shift = 8)
  (h_chair: hours_per_chair = 5)
  (h_days: days = 10) : 
  chairs_built_per_day hours_per_shift hours_per_chair * days = 10 :=
by 
  -- We insert a placeholder 'sorry' to be replaced by an actual proof.
  sorry

end ben_chairs_in_10_days_l88_88787


namespace frank_can_buy_seven_candies_l88_88466

def tickets_won_whackamole := 33
def tickets_won_skeeball := 9
def cost_per_candy := 6

theorem frank_can_buy_seven_candies : (tickets_won_whackamole + tickets_won_skeeball) / cost_per_candy = 7 :=
by
  sorry

end frank_can_buy_seven_candies_l88_88466


namespace trapezoid_area_equal_l88_88919

namespace Geometry

-- Define the areas of the outer and inner equilateral triangles.
def outer_triangle_area : ℝ := 25
def inner_triangle_area : ℝ := 4

-- The number of congruent trapezoids formed between the triangles.
def number_of_trapezoids : ℕ := 4

-- Prove that the area of one trapezoid is 5.25 square units.
theorem trapezoid_area_equal :
  (outer_triangle_area - inner_triangle_area) / number_of_trapezoids = 5.25 := by
  sorry

end Geometry

end trapezoid_area_equal_l88_88919


namespace tangent_properties_l88_88018

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

-- Given conditions
axiom differentiable_f : Differentiable ℝ f
axiom func_eq : ∀ x, f (x - 2) = f (-x)
axiom tangent_eq_at_1 : ∀ x, (x = 1 → f x = 2 * x + 1)

-- Prove the required results
theorem tangent_properties :
  (deriv f 1 = 2) ∧ (∃ B C, (∀ x, (x = -3) → f x = B -2 * (x + 3)) ∧ (B = 3) ∧ (C = -3)) :=
by
  sorry

end tangent_properties_l88_88018


namespace least_positive_integer_satifies_congruences_l88_88099

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end least_positive_integer_satifies_congruences_l88_88099


namespace bronson_yellow_leaves_l88_88580

-- Bronson collects 12 leaves on Thursday
def leaves_thursday : ℕ := 12

-- Bronson collects 13 leaves on Friday
def leaves_friday : ℕ := 13

-- 20% of the leaves are Brown (as a fraction)
def percent_brown : ℚ := 0.2

-- 20% of the leaves are Green (as a fraction)
def percent_green : ℚ := 0.2

theorem bronson_yellow_leaves : 
  (leaves_thursday + leaves_friday) * (1 - percent_brown - percent_green) = 15 := by
sorry

end bronson_yellow_leaves_l88_88580


namespace speed_of_B_is_three_l88_88941

noncomputable def speed_of_B (rounds_per_hour : ℕ) : Prop :=
  let A_speed : ℕ := 2
  let crossings : ℕ := 5
  let time_hours : ℕ := 1
  rounds_per_hour = (crossings - A_speed)

theorem speed_of_B_is_three : speed_of_B 3 :=
  sorry

end speed_of_B_is_three_l88_88941


namespace kittens_given_away_l88_88674

-- Conditions
def initial_kittens : ℕ := 8
def remaining_kittens : ℕ := 4

-- Statement to prove
theorem kittens_given_away : initial_kittens - remaining_kittens = 4 :=
by
  sorry

end kittens_given_away_l88_88674


namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l88_88102

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l88_88102


namespace largest_perimeter_l88_88204

noncomputable def interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

noncomputable def condition (n1 n2 n3 n4 : ℕ) : Prop :=
  2 * interior_angle n1 + interior_angle n2 + interior_angle n3 = 360

theorem largest_perimeter
  {n1 n2 n3 n4 : ℕ}
  (h : n1 = n4)
  (h_condition : condition n1 n2 n3 n4) :
  4 * n1 + 2 * n2 + 2 * n3 - 8 ≤ 22 :=
sorry

end largest_perimeter_l88_88204


namespace ratio_of_voters_l88_88367

theorem ratio_of_voters (V_X V_Y : ℝ) 
  (h1 : 0.62 * V_X + 0.38 * V_Y = 0.54 * (V_X + V_Y)) : V_X / V_Y = 2 :=
by
  sorry

end ratio_of_voters_l88_88367


namespace exponents_multiplication_l88_88715

variable (a : ℝ)

theorem exponents_multiplication : a^3 * a = a^4 := by
  sorry

end exponents_multiplication_l88_88715


namespace right_triangle_side_lengths_l88_88414

theorem right_triangle_side_lengths :
  ¬ (4^2 + 5^2 = 6^2) ∧
  (12^2 + 16^2 = 20^2) ∧
  ¬ (5^2 + 10^2 = 13^2) ∧
  ¬ (8^2 + 40^2 = 41^2) := by
  sorry

end right_triangle_side_lengths_l88_88414


namespace no_such_polynomial_exists_l88_88364

theorem no_such_polynomial_exists :
  ∀ (P : ℤ → ℤ), (∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
                  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4) → false :=
by
  sorry

end no_such_polynomial_exists_l88_88364


namespace average_weight_of_class_l88_88220

variable (SectionA_students : ℕ := 26)
variable (SectionB_students : ℕ := 34)
variable (SectionA_avg_weight : ℝ := 50)
variable (SectionB_avg_weight : ℝ := 30)

theorem average_weight_of_class :
  (SectionA_students * SectionA_avg_weight + SectionB_students * SectionB_avg_weight) / (SectionA_students + SectionB_students) = 38.67 := by
  sorry

end average_weight_of_class_l88_88220


namespace external_angle_theorem_proof_l88_88735

theorem external_angle_theorem_proof
    (x : ℝ)
    (FAB : ℝ)
    (BCA : ℝ)
    (ABC : ℝ)
    (h1 : FAB = 70)
    (h2 : BCA = 20 + x)
    (h3 : ABC = x + 20)
    (h4 : FAB = ABC + BCA) : 
    x = 15 :=
  by
  sorry

end external_angle_theorem_proof_l88_88735


namespace range_of_a_l88_88564

open Real

noncomputable def A (x : ℝ) : Prop := (x + 1) / (x - 2) ≥ 0
noncomputable def B (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≥ 0

theorem range_of_a :
  (∀ x, A x → B x a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l88_88564


namespace maximum_reflections_l88_88300

theorem maximum_reflections (θ : ℕ) (h : θ = 10) (max_angle : ℕ) (h_max : max_angle = 180) : 
∃ n : ℕ, n ≤ max_angle / θ ∧ n = 18 := by
  sorry

end maximum_reflections_l88_88300


namespace admission_methods_correct_l88_88482

-- Define the number of famous schools.
def famous_schools : ℕ := 8

-- Define the number of students.
def students : ℕ := 3

-- Define the total number of different admission methods:
def admission_methods (schools : ℕ) (students : ℕ) : ℕ :=
  Nat.choose schools 2 * 3

-- The theorem stating the desired result.
theorem admission_methods_correct :
  admission_methods famous_schools students = 84 :=
by
  sorry

end admission_methods_correct_l88_88482


namespace find_multiple_l88_88822

theorem find_multiple (m : ℤ) : 38 + m * 43 = 124 → m = 2 := by
  intro h
  sorry

end find_multiple_l88_88822


namespace possible_apple_counts_l88_88809

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l88_88809


namespace t_le_s_l88_88559

theorem t_le_s (a b : ℝ) (t s : ℝ) (h1 : t = a + 2 * b) (h2 : s = a + b^2 + 1) : t ≤ s :=
by
  sorry

end t_le_s_l88_88559


namespace perimeter_of_triangle_is_13_l88_88732

-- Conditions
noncomputable def perimeter_of_triangle_with_two_sides_and_third_root_of_eq : ℝ :=
  let a := 3
  let b := 6
  let c1 := 2 -- One root of the equation x^2 - 6x + 8 = 0
  let c2 := 4 -- Another root of the equation x^2 - 6x + 8 = 0
  if a + b > c2 ∧ a + c2 > b ∧ b + c2 > a then
    a + b + c2
  else
    0 -- not possible to form a triangle with these sides

-- Assertion
theorem perimeter_of_triangle_is_13 :
  perimeter_of_triangle_with_two_sides_and_third_root_of_eq = 13 := 
sorry

end perimeter_of_triangle_is_13_l88_88732


namespace exists_right_triangle_area_twice_hypotenuse_l88_88221

theorem exists_right_triangle_area_twice_hypotenuse : 
  ∃ (a : ℝ), a ≠ 0 ∧ (a^2 / 2 = 2 * a * Real.sqrt 2) ∧ (a = 4 * Real.sqrt 2) :=
by
  sorry

end exists_right_triangle_area_twice_hypotenuse_l88_88221


namespace unique_function_solution_l88_88244

variable (f : ℝ → ℝ)

theorem unique_function_solution :
  (∀ x y : ℝ, f (f x - y^2) = f x ^ 2 - 2 * f x * y^2 + f (f y))
  → (∀ x : ℝ, f x = x^2) :=
by
  sorry

end unique_function_solution_l88_88244


namespace limo_gas_price_l88_88196

theorem limo_gas_price
  (hourly_wage : ℕ := 15)
  (ride_payment : ℕ := 5)
  (review_bonus : ℕ := 20)
  (hours_worked : ℕ := 8)
  (rides_given : ℕ := 3)
  (gallons_gas : ℕ := 17)
  (good_reviews : ℕ := 2)
  (total_owed : ℕ := 226) :
  total_owed = (hours_worked * hourly_wage) + (rides_given * ride_payment) + (good_reviews * review_bonus) + (gallons_gas * 3) :=
by
  sorry

end limo_gas_price_l88_88196


namespace Qing_Dynasty_Problem_l88_88433

variable {x y : ℕ}

theorem Qing_Dynasty_Problem (h1 : 4 * x + 6 * y = 48) (h2 : 2 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (2 * x + 5 * y = 38) := by
  exact ⟨h1, h2⟩

end Qing_Dynasty_Problem_l88_88433


namespace largest_value_among_given_numbers_l88_88330

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem largest_value_among_given_numbers :
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20 
  b > a ∧ b > c ∧ b > d :=
by
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20
  -- Add the necessary steps to show that b is the largest value
  sorry

end largest_value_among_given_numbers_l88_88330


namespace popcorn_probability_l88_88692

theorem popcorn_probability {w y b : ℝ} (hw : w = 3/5) (hy : y = 1/5) (hb : b = 1/5)
  {pw py pb : ℝ} (hpw : pw = 1/3) (hpy : py = 3/4) (hpb : pb = 1/2) :
  (y * py) / (w * pw + y * py + b * pb) = 1/3 := 
sorry

end popcorn_probability_l88_88692


namespace sphere_touches_pyramid_edges_l88_88583

theorem sphere_touches_pyramid_edges :
  ∃ (KL : ℝ), 
  ∃ (K L M N : ℝ) (MN LN NK : ℝ) (AC: ℝ) (BC: ℝ), 
  MN = 7 ∧ 
  NK = 5 ∧ 
  LN = 2 * Real.sqrt 29 ∧ 
  KL = L ∧ 
  KL = M ∧ 
  KL = 9 :=
sorry

end sphere_touches_pyramid_edges_l88_88583


namespace find_circle_center_l88_88861

def circle_center (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 16 = 0

theorem find_circle_center (x y : ℝ) :
  circle_center x y ↔ (x, y) = (3, 4) :=
by
  sorry

end find_circle_center_l88_88861


namespace new_percentage_of_girls_is_5_l88_88832

theorem new_percentage_of_girls_is_5
  (initial_children : ℕ)
  (percentage_boys : ℕ)
  (added_boys : ℕ)
  (initial_total_boys : ℕ)
  (initial_total_girls : ℕ)
  (new_total_boys : ℕ)
  (new_total_children : ℕ)
  (new_percentage_girls : ℕ)
  (h1 : initial_children = 60)
  (h2 : percentage_boys = 90)
  (h3 : added_boys = 60)
  (h4 : initial_total_boys = (percentage_boys * initial_children / 100))
  (h5 : initial_total_girls = initial_children - initial_total_boys)
  (h6 : new_total_boys = initial_total_boys + added_boys)
  (h7 : new_total_children = initial_children + added_boys)
  (h8 : new_percentage_girls = (initial_total_girls * 100 / new_total_children)) :
  new_percentage_girls = 5 :=
by sorry

end new_percentage_of_girls_is_5_l88_88832


namespace total_polled_votes_proof_l88_88804

-- Define the conditions
variables (V : ℕ) -- total number of valid votes
variables (invalid_votes : ℕ) -- number of invalid votes
variables (total_polled_votes : ℕ) -- total polled votes
variables (candidateA_votes candidateB_votes : ℕ) -- votes for candidate A and B respectively

-- Assume the known conditions
variable (h1 : candidateA_votes = 45 * V / 100) -- candidate A got 45% of valid votes
variable (h2 : candidateB_votes = 55 * V / 100) -- candidate B got 55% of valid votes
variable (h3 : candidateB_votes - candidateA_votes = 9000) -- candidate A was defeated by 9000 votes
variable (h4 : invalid_votes = 83) -- there are 83 invalid votes
variable (h5 : total_polled_votes = V + invalid_votes) -- total polled votes is sum of valid and invalid votes

-- Define the theorem to prove
theorem total_polled_votes_proof : total_polled_votes = 90083 :=
by 
  -- Placeholder for the proof
  sorry

end total_polled_votes_proof_l88_88804


namespace quadratic_has_two_real_roots_find_m_for_roots_difference_l88_88193

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1^2 + (2 - m) * x1 + (1 - m) = 0 ∧
                 x2^2 + (2 - m) * x2 + (1 - m) = 0 :=
by sorry

theorem find_m_for_roots_difference (m x1 x2 : ℝ) (h1 : x1^2 + (2 - m) * x1 + (1 - m) = 0) 
  (h2 : x2^2 + (2 - m) * x2 + (1 - m) = 0) (hm : m < 0) (hd : x1 - x2 = 3) : 
  m = -3 :=
by sorry

end quadratic_has_two_real_roots_find_m_for_roots_difference_l88_88193


namespace total_rides_correct_l88_88831

-- Definitions based on the conditions:
def billy_rides : ℕ := 17
def john_rides : ℕ := 2 * billy_rides
def mother_rides : ℕ := john_rides + 10
def total_rides : ℕ := billy_rides + john_rides + mother_rides

-- The theorem to prove their total bike rides.
theorem total_rides_correct : total_rides = 95 := by
  sorry

end total_rides_correct_l88_88831


namespace volume_of_rectangular_solid_l88_88968

theorem volume_of_rectangular_solid 
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : c * a = 6) :
  a * b * c = 30 := 
by
  -- sorry placeholder for the proof
  sorry

end volume_of_rectangular_solid_l88_88968


namespace student_A_final_score_l88_88889

theorem student_A_final_score (total_questions : ℕ) (correct_responses : ℕ) 
  (h1 : total_questions = 100) (h2 : correct_responses = 93) : 
  correct_responses - 2 * (total_questions - correct_responses) = 79 :=
by
  rw [h1, h2]
  -- sorry

end student_A_final_score_l88_88889


namespace concentration_proof_l88_88105

noncomputable def newConcentration (vol1 vol2 vol3 : ℝ) (perc1 perc2 perc3 : ℝ) (totalVol : ℝ) (finalVol : ℝ) :=
  (vol1 * perc1 + vol2 * perc2 + vol3 * perc3) / finalVol

theorem concentration_proof : 
  newConcentration 2 6 4 0.2 0.55 0.35 (12 : ℝ) (15 : ℝ) = 0.34 := 
by 
  sorry

end concentration_proof_l88_88105


namespace calories_burned_per_week_l88_88571

-- Definitions from conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℚ := 1.5
def calories_per_minute : ℕ := 7

-- Prove the total calories burned per week
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * 60) * calories_per_minute) = 1890 := by
    sorry

end calories_burned_per_week_l88_88571


namespace exam_correct_answers_count_l88_88926

theorem exam_correct_answers_count (x y : ℕ) (h1 : x + y = 80) (h2 : 4 * x - y = 130) : x = 42 :=
by {
  -- (proof to be completed later)
  sorry
}

end exam_correct_answers_count_l88_88926


namespace tan_identity_l88_88856

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l88_88856


namespace highest_possible_rubidium_concentration_l88_88723

noncomputable def max_rubidium_concentration (R C F : ℝ) : Prop :=
  (R + C + F > 0) →
  (0.10 * R + 0.08 * C + 0.05 * F) / (R + C + F) = 0.07 ∧
  (0.05 * F) / (R + C + F) ≤ 0.02 →
  (0.10 * R) / (R + C + F) = 0.01

theorem highest_possible_rubidium_concentration :
  ∃ R C F : ℝ, max_rubidium_concentration R C F :=
sorry

end highest_possible_rubidium_concentration_l88_88723


namespace simplify_fraction_l88_88079

variables {x y : ℝ}

theorem simplify_fraction (h : x / y = 2 / 5) : (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 :=
by
  sorry

end simplify_fraction_l88_88079


namespace large_cube_side_length_painted_blue_l88_88251

   theorem large_cube_side_length_painted_blue (n : ℕ) (h : 6 * n^2 = (1 / 3) * 6 * n^3) : n = 3 :=
   by
     sorry
   
end large_cube_side_length_painted_blue_l88_88251


namespace greatest_possible_gcd_value_l88_88092

noncomputable def sn (n : ℕ) := n ^ 2
noncomputable def expression (n : ℕ) := 2 * sn n + 10 * n
noncomputable def gcd_value (a b : ℕ) := Nat.gcd a b 

theorem greatest_possible_gcd_value :
  ∃ n : ℕ, gcd_value (expression n) (n - 3) = 42 :=
sorry

end greatest_possible_gcd_value_l88_88092


namespace functional_equation_solution_l88_88712

theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) : 
    ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l88_88712


namespace travel_distance_proof_l88_88775

-- Definitions based on conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Calculate distances traveled
def amoli_distance : ℕ := amoli_speed * amoli_time
def anayet_distance : ℕ := anayet_speed * anayet_time

-- Calculate total distance covered
def total_distance_covered : ℕ := amoli_distance + anayet_distance

-- Define remaining distance to travel
def remaining_distance (total : ℕ) (covered : ℕ) : ℕ := total - covered

-- The theorem to prove
theorem travel_distance_proof : remaining_distance total_distance total_distance_covered = 121 := by
  -- Placeholder for the actual proof
  sorry

end travel_distance_proof_l88_88775


namespace equation_in_terms_of_y_l88_88289

theorem equation_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : y = 5 - 2 * x :=
sorry

end equation_in_terms_of_y_l88_88289


namespace Kayla_score_fifth_level_l88_88175

theorem Kayla_score_fifth_level :
  ∃ (a b c d e f : ℕ),
  a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 8 ∧ f = 17 ∧
  (b - a = 1) ∧ (c - b = 2) ∧ (d - c = 3) ∧ (e - d = 4) ∧ (f - e = 5) ∧ e = 12 :=
sorry

end Kayla_score_fifth_level_l88_88175


namespace last_third_speed_l88_88772

-- Definitions based on the conditions in the problem statement
def first_third_speed : ℝ := 80
def second_third_speed : ℝ := 30
def average_speed : ℝ := 45

-- Definition of the distance covered variable (non-zero to avoid division by zero)
variable (D : ℝ) (hD : D ≠ 0)

-- The unknown speed during the last third of the distance
noncomputable def V : ℝ := 
  D / ((D / 3 / first_third_speed) + (D / 3 / second_third_speed) + (D / 3 / average_speed))

-- The theorem to prove
theorem last_third_speed : V = 48 :=
by
  sorry

end last_third_speed_l88_88772


namespace find_seating_capacity_l88_88911

theorem find_seating_capacity (x : ℕ) :
  (4 * x + 30 = 5 * x - 10) → (x = 40) :=
by
  intros h
  sorry

end find_seating_capacity_l88_88911


namespace puppies_per_cage_l88_88247

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (remaining_puppies : ℕ)
  (cages : ℕ)
  (puppies_per_cage : ℕ)
  (h1 : initial_puppies = 78)
  (h2 : sold_puppies = 30)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 6)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 8 := by
  sorry

end puppies_per_cage_l88_88247


namespace find_p_l88_88484

-- Conditions: Consider the quadratic equation 2x^2 + px + q = 0 where p and q are integers.
-- Roots of the equation differ by 2.
-- q = 4

theorem find_p (p : ℤ) (q : ℤ) (h1 : q = 4) (h2 : ∃ x₁ x₂ : ℝ, 2 * x₁^2 + p * x₁ + q = 0 ∧ 2 * x₂^2 + p * x₂ + q = 0 ∧ |x₁ - x₂| = 2) :
  p = 7 ∨ p = -7 :=
by
  sorry

end find_p_l88_88484


namespace required_extra_money_l88_88297

theorem required_extra_money 
(Patricia_money Lisa_money Charlotte_money : ℕ) 
(hP : Patricia_money = 6) 
(hL : Lisa_money = 5 * Patricia_money) 
(hC : Lisa_money = 2 * Charlotte_money) 
(cost : ℕ) 
(hCost : cost = 100) : 
  cost - (Patricia_money + Lisa_money + Charlotte_money) = 49 := 
by 
  sorry

end required_extra_money_l88_88297


namespace largest_digit_A_l88_88784

theorem largest_digit_A (A : ℕ) (h1 : (31 + A) % 3 = 0) (h2 : 96 % 4 = 0) : 
  A ≤ 7 ∧ (∀ a, a > 7 → ¬((31 + a) % 3 = 0 ∧ 96 % 4 = 0)) :=
by
  sorry

end largest_digit_A_l88_88784


namespace bill_has_six_times_more_nuts_l88_88894

-- Definitions for the conditions
def sue_has_nuts : ℕ := 48
def harry_has_nuts (sueNuts : ℕ) : ℕ := 2 * sueNuts
def combined_nuts (harryNuts : ℕ) (billNuts : ℕ) : ℕ := harryNuts + billNuts
def bill_has_nuts (totalNuts : ℕ) (harryNuts : ℕ) : ℕ := totalNuts - harryNuts

-- Statement to prove
theorem bill_has_six_times_more_nuts :
  ∀ sueNuts billNuts harryNuts totalNuts,
    sueNuts = sue_has_nuts →
    harryNuts = harry_has_nuts sueNuts →
    totalNuts = 672 →
    combined_nuts harryNuts billNuts = totalNuts →
    billNuts = bill_has_nuts totalNuts harryNuts →
    billNuts = 6 * harryNuts :=
by
  intros sueNuts billNuts harryNuts totalNuts hsueNuts hharryNuts htotalNuts hcombinedNuts hbillNuts
  sorry

end bill_has_six_times_more_nuts_l88_88894


namespace range_of_a_l88_88235

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l88_88235


namespace factor_congruence_l88_88277

theorem factor_congruence (n : ℕ) (hn : n ≠ 0) :
  ∀ p : ℕ, p ∣ (2 * n)^(2^n) + 1 → p ≡ 1 [MOD 2^(n+1)] :=
sorry

end factor_congruence_l88_88277


namespace expand_and_simplify_l88_88843

theorem expand_and_simplify (x : ℝ) : 
  -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 :=
sorry

end expand_and_simplify_l88_88843


namespace courtyard_width_l88_88119

theorem courtyard_width (length : ℕ) (brick_length brick_width : ℕ) (num_bricks : ℕ) (W : ℕ)
  (H1 : length = 25)
  (H2 : brick_length = 20)
  (H3 : brick_width = 10)
  (H4 : num_bricks = 18750)
  (H5 : 2500 * (W * 100) = num_bricks * (brick_length * brick_width)) :
  W = 15 :=
by sorry

end courtyard_width_l88_88119


namespace find_sum_of_numbers_l88_88897

-- Define the problem using the given conditions
def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

-- The main theorem we want to prove
theorem find_sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) :
  sum_of_three_numbers a b c = 22 :=
by
  sorry

end find_sum_of_numbers_l88_88897


namespace custom_op_example_l88_88371

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := 4 * a + 5 * b - a * b

-- The proof statement
theorem custom_op_example : custom_op 7 3 = 22 := by
  sorry

end custom_op_example_l88_88371


namespace max_value_of_E_l88_88664

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  ∀ (a b c d : ℝ),
    (-8.5 ≤ a ∧ a ≤ 8.5) →
    (-8.5 ≤ b ∧ b ≤ 8.5) →
    (-8.5 ≤ c ∧ c ≤ 8.5) →
    (-8.5 ≤ d ∧ d ≤ 8.5) →
    E a b c d ≤ 306 := sorry

end max_value_of_E_l88_88664


namespace ratio_of_distances_l88_88322

-- Define the given conditions
variables (w x y : ℕ)
variables (h1 : w > 0) -- walking speed must be positive
variables (h2 : x > 0) -- distance from home must be positive
variables (h3 : y > 0) -- distance to stadium must be positive

-- Define the two times:
-- Time taken to walk directly to the stadium
def time_walk (w y : ℕ) := y / w

-- Time taken to walk home, then bike to the stadium
def time_walk_bike (w x y : ℕ) := x / w + (x + y) / (5 * w)

-- Given that both times are equal
def times_equal (w x y : ℕ) := time_walk w y = time_walk_bike w x y

-- We want to prove that the ratio of x to y is 2/3
theorem ratio_of_distances (w x y : ℕ) (h_time_eq : times_equal w x y) : x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l88_88322


namespace cyclist_speed_l88_88621

theorem cyclist_speed:
  ∀ (c : ℝ), 
  ∀ (hiker_speed : ℝ), 
  (hiker_speed = 4) → 
  (4 * (5 / 60) + 4 * (25 / 60) = c * (5 / 60)) → 
  c = 24 := 
by
  intros c hiker_speed hiker_speed_def distance_eq
  sorry

end cyclist_speed_l88_88621


namespace pen_sales_average_l88_88620

theorem pen_sales_average :
  ∃ d : ℕ, (48 = (96 + 44 * d) / (d + 1)) → d = 12 :=
by
  sorry

end pen_sales_average_l88_88620


namespace annika_current_age_l88_88895

-- Define the conditions
def hans_age_current : ℕ := 8
def hans_age_in_4_years : ℕ := hans_age_current + 4
def annika_age_in_4_years : ℕ := 3 * hans_age_in_4_years

-- lean statement to prove Annika's current age
theorem annika_current_age (A : ℕ) (hyp : A + 4 = annika_age_in_4_years) : A = 32 :=
by
  -- Skipping the proof
  sorry

end annika_current_age_l88_88895


namespace repayment_correct_l88_88161

noncomputable def repayment_amount (a γ : ℝ) : ℝ :=
  a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1)

theorem repayment_correct (a γ : ℝ) (γ_pos : γ > 0) : 
  repayment_amount a γ = a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1) :=
by
   sorry

end repayment_correct_l88_88161


namespace total_waiting_time_l88_88627

def t1 : ℕ := 20
def t2 : ℕ := 4 * t1 + 14
def T : ℕ := t1 + t2

theorem total_waiting_time : T = 114 :=
by {
  -- Preliminary calculations and justification would go here
  sorry
}

end total_waiting_time_l88_88627


namespace break_even_price_correct_l88_88786

-- Conditions
def variable_cost_per_handle : ℝ := 0.60
def fixed_cost_per_week : ℝ := 7640
def handles_per_week : ℝ := 1910

-- Define the correct answer for the price per handle to break even
def break_even_price_per_handle : ℝ := 4.60

-- The statement to prove
theorem break_even_price_correct :
  fixed_cost_per_week + (variable_cost_per_handle * handles_per_week) / handles_per_week = break_even_price_per_handle :=
by
  -- The proof is omitted
  sorry

end break_even_price_correct_l88_88786


namespace Isabella_redeem_day_l88_88655

def is_coupon_day_closed_sunday (start_day : ℕ) (num_coupons : ℕ) (cycle_days : ℕ) : Prop :=
  ∃ n, n < num_coupons ∧ (start_day + n * cycle_days) % 7 = 0

theorem Isabella_redeem_day: 
  ∀ (day : ℕ), day ≡ 1 [MOD 7]
  → ¬ is_coupon_day_closed_sunday day 6 11 :=
by
  intro day h_mod
  simp [is_coupon_day_closed_sunday]
  sorry

end Isabella_redeem_day_l88_88655


namespace sin_double_angle_subst_l88_88227

open Real

theorem sin_double_angle_subst 
  (α : ℝ)
  (h : sin (α + π / 6) = -1 / 3) :
  sin (2 * α - π / 6) = -7 / 9 := 
by
  sorry

end sin_double_angle_subst_l88_88227


namespace dasha_strip_problem_l88_88043

theorem dasha_strip_problem (a b c : ℕ) (h : a * (2 * b + 2 * c - a) = 43) :
  a = 1 ∧ b + c = 22 :=
by {
  sorry
}

end dasha_strip_problem_l88_88043


namespace fgh_deriv_at_0_l88_88518

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry

-- Function Values at x = 0
axiom f_zero : f 0 = 1
axiom g_zero : g 0 = 2
axiom h_zero : h 0 = 3

-- Derivatives of the pairwise products at x = 0
axiom d_gh_zero : (deriv (λ x => g x * h x)) 0 = 4
axiom d_hf_zero : (deriv (λ x => h x * f x)) 0 = 5
axiom d_fg_zero : (deriv (λ x => f x * g x)) 0 = 6

-- We need to prove that the derivative of the product of f, g, h at x = 0 is 16
theorem fgh_deriv_at_0 : (deriv (λ x => f x * g x * h x)) 0 = 16 := by
  sorry

end fgh_deriv_at_0_l88_88518


namespace value_of_star_15_25_l88_88978

noncomputable def star (x y : ℝ) : ℝ := Real.log x / Real.log y

axiom condition1 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star (star (x^2) y) y = star x y
axiom condition2 (x y : ℝ) (hxy : x > 0 ∧ y > 0) : star x (star y y) = star (star x y) (star x 1)
axiom condition3 (h : 1 > 0) : star 1 1 = 0

theorem value_of_star_15_25 : star 15 25 = (Real.log 3 / (2 * Real.log 5)) + 1 / 2 := 
by 
  sorry

end value_of_star_15_25_l88_88978


namespace sum_of_first_three_terms_is_zero_l88_88619

variable (a d : ℤ) 

-- Definitions from the conditions
def a₄ := a + 3 * d
def a₅ := a + 4 * d
def a₆ := a + 5 * d

-- Theorem statement
theorem sum_of_first_three_terms_is_zero 
  (h₁ : a₄ = 8) 
  (h₂ : a₅ = 12) 
  (h₃ : a₆ = 16) : 
  a + (a + d) + (a + 2 * d) = 0 := 
by 
  sorry

end sum_of_first_three_terms_is_zero_l88_88619


namespace find_larger_number_l88_88885

theorem find_larger_number (x y : ℕ) (h1 : x * y = 56) (h2 : x + y = 15) : max x y = 8 :=
sorry

end find_larger_number_l88_88885


namespace correct_value_l88_88368

theorem correct_value : ∀ (x : ℕ),  (x / 6 = 12) → (x * 7 = 504) :=
  sorry

end correct_value_l88_88368


namespace statement_3_correct_l88_88652

-- Definitions based on the conditions
def DeductiveReasoningGeneralToSpecific := True
def SyllogismForm := True
def ConclusionDependsOnPremisesAndForm := True

-- Proof problem statement
theorem statement_3_correct : SyllogismForm := by
  exact True.intro

end statement_3_correct_l88_88652


namespace common_ratio_of_gp_l88_88098

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem common_ratio_of_gp (a : ℝ) (r : ℝ) (h : geometric_sum a r 6 / geometric_sum a r 3 = 28) : r = 3 :=
by
  sorry

end common_ratio_of_gp_l88_88098


namespace sum_of_coefficients_l88_88781

theorem sum_of_coefficients : 
  ∃ (a b c d e f g h j k : ℤ), 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) → 
    (a + b + c + d + e + f + g + h + j + k = 92) :=
sorry

end sum_of_coefficients_l88_88781


namespace hyperbola_vertex_distance_l88_88660

open Real

/-- The distance between the vertices of the hyperbola represented by the equation
    (y-4)^2 / 32 - (x+3)^2 / 18 = 1 is 8√2. -/
theorem hyperbola_vertex_distance :
  let a := sqrt 32
  2 * a = 8 * sqrt 2 :=
by
  sorry

end hyperbola_vertex_distance_l88_88660


namespace length_of_segment_AB_l88_88819

noncomputable def speed_relation_first (x v1 v2 : ℝ) : Prop :=
  300 / v1 = (x - 300) / v2

noncomputable def speed_relation_second (x v1 v2 : ℝ) : Prop :=
  (x + 100) / v1 = (x - 100) / v2

theorem length_of_segment_AB :
  (∃ (x v1 v2 : ℝ),
    x > 0 ∧
    v1 > 0 ∧
    v2 > 0 ∧
    speed_relation_first x v1 v2 ∧
    speed_relation_second x v1 v2) →
  ∃ x : ℝ, x = 500 :=
by
  sorry

end length_of_segment_AB_l88_88819


namespace prob_qualified_bulb_factory_a_l88_88115

-- Define the given probability of a light bulb being produced by Factory A
def prob_factory_a : ℝ := 0.7

-- Define the given pass rate (conditional probability) of Factory A's light bulbs
def pass_rate_factory_a : ℝ := 0.95

-- The goal is to prove that the probability of getting a qualified light bulb produced by Factory A is 0.665
theorem prob_qualified_bulb_factory_a : prob_factory_a * pass_rate_factory_a = 0.665 :=
by
  -- This is where the proof would be, but we'll use sorry to skip the proof
  sorry

end prob_qualified_bulb_factory_a_l88_88115


namespace total_distance_correct_l88_88608

def day1_distance : ℕ := (5 * 4) + (3 * 2) + (4 * 3)
def day2_distance : ℕ := (6 * 3) + (2 * 1) + (6 * 3) + (3 * 4)
def day3_distance : ℕ := (4 * 2) + (2 * 1) + (7 * 3) + (5 * 2)

def total_distance : ℕ := day1_distance + day2_distance + day3_distance

theorem total_distance_correct :
  total_distance = 129 := by
  sorry

end total_distance_correct_l88_88608


namespace find_third_number_l88_88801

theorem find_third_number (x : ℝ) : 3 + 33 + x + 3.33 = 369.63 → x = 330.30 :=
by
  intros h
  sorry

end find_third_number_l88_88801


namespace total_time_for_12000_dolls_l88_88639

noncomputable def total_combined_machine_operation_time (num_dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) (time_per_doll time_per_accessory : ℕ) : ℕ :=
  let total_accessories_per_doll := shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll
  let total_accessories := num_dolls * total_accessories_per_doll
  let time_for_dolls := num_dolls * time_per_doll
  let time_for_accessories := total_accessories * time_per_accessory
  time_for_dolls + time_for_accessories

theorem total_time_for_12000_dolls (h1 : ∀ (x : ℕ), x = 12000) (h2 : ∀ (x : ℕ), x = 2) (h3 : ∀ (x : ℕ), x = 3) (h4 : ∀ (x : ℕ), x = 1) (h5 : ∀ (x : ℕ), x = 5) (h6 : ∀ (x : ℕ), x = 45) (h7 : ∀ (x : ℕ), x = 10) :
  total_combined_machine_operation_time 12000 2 3 1 5 45 10 = 1860000 := by 
  sorry

end total_time_for_12000_dolls_l88_88639


namespace degree_measure_cherry_pie_l88_88736

theorem degree_measure_cherry_pie 
  (total_students : ℕ) 
  (chocolate_pie : ℕ) 
  (apple_pie : ℕ) 
  (blueberry_pie : ℕ) 
  (remaining_students : ℕ)
  (remaining_students_eq_div : remaining_students = (total_students - (chocolate_pie + apple_pie + blueberry_pie))) 
  (equal_division : remaining_students / 2 = 5) 
  : (remaining_students / 2 * 360 / total_students = 45) := 
by 
  sorry

end degree_measure_cherry_pie_l88_88736


namespace problem_1_problem_2_l88_88965

-- Proof Problem 1
theorem problem_1 (x : ℝ) : (x^2 + 2 > |x - 4| - |x - 1|) ↔ (x > 1 ∨ x ≤ -1) :=
sorry

-- Proof Problem 2
theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂, x₁^2 + 2 ≥ |x₂ - a| - |x₂ - 1|) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end problem_1_problem_2_l88_88965


namespace increasing_function_range_l88_88839

theorem increasing_function_range (k : ℝ) :
  (∀ x y : ℝ, x < y → (k + 2) * x + 1 < (k + 2) * y + 1) ↔ k > -2 :=
by
  sorry

end increasing_function_range_l88_88839


namespace kylie_gave_21_coins_to_Laura_l88_88727

def coins_from_piggy_bank : ℕ := 15
def coins_from_brother : ℕ := 13
def coins_from_father : ℕ := 8
def coins_left : ℕ := 15

def total_coins_collected : ℕ := coins_from_piggy_bank + coins_from_brother + coins_from_father
def coins_given_to_Laura : ℕ := total_coins_collected - coins_left

theorem kylie_gave_21_coins_to_Laura :
  coins_given_to_Laura = 21 :=
by
  sorry

end kylie_gave_21_coins_to_Laura_l88_88727


namespace initial_amount_l88_88891

theorem initial_amount (bread_price : ℝ) (bread_qty : ℝ) (pb_price : ℝ) (leftover : ℝ) :
  bread_price = 2.25 → bread_qty = 3 → pb_price = 2 → leftover = 5.25 →
  bread_qty * bread_price + pb_price + leftover = 14 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num


end initial_amount_l88_88891


namespace ed_money_left_l88_88623

theorem ed_money_left
  (cost_per_hour_night : ℝ := 1.5)
  (cost_per_hour_morning : ℝ := 2)
  (initial_money : ℝ := 80)
  (hours_night : ℝ := 6)
  (hours_morning : ℝ := 4) :
  initial_money - (cost_per_hour_night * hours_night + cost_per_hour_morning * hours_morning) = 63 := 
  by
  sorry

end ed_money_left_l88_88623


namespace operation_1_and_2004_l88_88214

def operation (m n : ℕ) : ℕ :=
  if m = 1 ∧ n = 1 then 2
  else if m = 1 ∧ n > 1 then 2 + 3 * (n - 1)
  else 0 -- handle other cases generically, although specifics are not given

theorem operation_1_and_2004 : operation 1 2004 = 6011 :=
by
  unfold operation
  sorry

end operation_1_and_2004_l88_88214


namespace original_number_of_people_l88_88328

-- Defining the conditions
variable (n : ℕ) -- number of people originally
variable (total_cost : ℕ := 375)
variable (equal_cost_split : n > 0 ∧ total_cost = 375) -- total cost is $375 and n > 0
variable (cost_condition : 375 / n + 50 = 375 / 5)

-- The proof statement
theorem original_number_of_people (h1 : total_cost = 375) (h2 : 375 / n + 50 = 375 / 5) : n = 15 :=
by
  sorry

end original_number_of_people_l88_88328


namespace rationalize_denominator_l88_88001

noncomputable def sqrt_12 := Real.sqrt 12
noncomputable def sqrt_5 := Real.sqrt 5
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_15 := Real.sqrt 15

theorem rationalize_denominator :
  (sqrt_12 + sqrt_5) / (sqrt_3 + sqrt_5) = (-1 / 2) + (sqrt_15 / 2) :=
sorry

end rationalize_denominator_l88_88001


namespace symmetric_point_coordinates_l88_88317

theorem symmetric_point_coordinates (Q : ℝ × ℝ × ℝ) 
  (P : ℝ × ℝ × ℝ := (-6, 7, -9)) 
  (A : ℝ × ℝ × ℝ := (1, 3, -1)) 
  (B : ℝ × ℝ × ℝ := (6, 5, -2)) 
  (C : ℝ × ℝ × ℝ := (0, -3, -5)) : Q = (2, -5, 7) :=
sorry

end symmetric_point_coordinates_l88_88317


namespace ratio_two_to_three_nights_ago_l88_88042

def question (x : ℕ) (k : ℕ) : (ℕ × ℕ) := (x, k)

def pages_three_nights_ago := 15
def additional_pages_last_night (x : ℕ) := x + 5
def total_pages := 100
def pages_tonight := 20

theorem ratio_two_to_three_nights_ago :
  ∃ (x : ℕ), 
    (x + additional_pages_last_night x = total_pages - (pages_three_nights_ago + pages_tonight)) 
    ∧ (x / pages_three_nights_ago = 2 / 1) :=
by
  sorry

end ratio_two_to_three_nights_ago_l88_88042


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l88_88958

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l88_88958


namespace find_AB_l88_88391

-- Definitions based on conditions
variables (AB CD : ℝ)

-- Given conditions
def area_ratio_condition : Prop :=
  AB / CD = 5 / 3

def sum_condition : Prop :=
  AB + CD = 160

-- The main statement to be proven
theorem find_AB (h_ratio : area_ratio_condition AB CD) (h_sum : sum_condition AB CD) :
  AB = 100 :=
by
  sorry

end find_AB_l88_88391


namespace sum_of_g_31_values_l88_88453

def f (x : ℝ) : ℝ := 4 * x^2 - 3
def g (y : ℝ) : ℝ := y ^ 2 - y + 2

theorem sum_of_g_31_values :
  g 31 + g 31 = 21 := sorry

end sum_of_g_31_values_l88_88453


namespace length_of_other_train_is_correct_l88_88027

noncomputable def length_of_other_train
  (l1 : ℝ) -- length of the first train in meters
  (s1 : ℝ) -- speed of the first train in km/hr
  (s2 : ℝ) -- speed of the second train in km/hr
  (t : ℝ)  -- time in seconds
  (h1 : l1 = 500)
  (h2 : s1 = 240)
  (h3 : s2 = 180)
  (h4 : t = 12) :
  ℝ :=
  let s1_m_s := s1 * 1000 / 3600
  let s2_m_s := s2 * 1000 / 3600
  let relative_speed := s1_m_s + s2_m_s
  let total_distance := relative_speed * t
  total_distance - l1

theorem length_of_other_train_is_correct :
  length_of_other_train 500 240 180 12 rfl rfl rfl rfl = 900 := sorry

end length_of_other_train_is_correct_l88_88027


namespace total_money_shared_l88_88845

theorem total_money_shared (rA rB rC : ℕ) (pA : ℕ) (total : ℕ) 
  (h_ratio : rA = 1 ∧ rB = 2 ∧ rC = 7) 
  (h_A_money : pA = 20) 
  (h_total : total = pA * rA + pA * rB + pA * rC) : 
  total = 200 := by 
  sorry

end total_money_shared_l88_88845


namespace unique_function_satisfying_conditions_l88_88209

theorem unique_function_satisfying_conditions :
  ∀ (f : ℝ → ℝ), 
    (∀ x : ℝ, f x ≥ 0) → 
    (∀ x : ℝ, f (x^2) = f x ^ 2 - 2 * x * f x) →
    (∀ x : ℝ, f (-x) = f (x - 1)) → 
    (∀ x y : ℝ, 1 < x → x < y → f x < f y) →
    (∀ x : ℝ, f x = x^2 + x + 1) :=
by
  -- formal proof would go here
  sorry

end unique_function_satisfying_conditions_l88_88209


namespace find_BG_l88_88673

-- Define given lengths and the required proof
def BC : ℝ := 5
def BF : ℝ := 12

theorem find_BG : BG = 13 := by
  -- Formal proof would go here
  sorry

end find_BG_l88_88673


namespace saree_blue_stripes_l88_88406

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    brown_stripes = 4 →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_gold h_blue h_brown
  sorry

end saree_blue_stripes_l88_88406


namespace isosceles_triangle_vertex_angle_l88_88793

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) (h_triangle : α + β + γ = 180)
  (h_isosceles : α = β ∨ β = α ∨ α = γ ∨ γ = α ∨ β = γ ∨ γ = β)
  (h_ratio : α / γ = 1 / 4 ∨ γ / α = 1 / 4) :
  (γ = 20 ∨ γ = 120) :=
sorry

end isosceles_triangle_vertex_angle_l88_88793


namespace division_quotient_difference_l88_88988

theorem division_quotient_difference :
  (32.5 / 1.3) - (60.8 / 7.6) = 17 :=
by
  sorry

end division_quotient_difference_l88_88988


namespace candle_height_l88_88899

variable (h d a b x : ℝ)

theorem candle_height (h d a b : ℝ) : x = h * (1 + d / (a + b)) :=
by
  sorry

end candle_height_l88_88899


namespace proof_problem_l88_88629

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 2 * x + 1

theorem proof_problem : f (g 3) - g (f 3) = -5 := by
  sorry

end proof_problem_l88_88629


namespace remainder_2abc_mod_7_l88_88520

theorem remainder_2abc_mod_7
  (a b c : ℕ)
  (h₀ : 2 * a + 3 * b + c ≡ 1 [MOD 7])
  (h₁ : 3 * a + b + 2 * c ≡ 2 [MOD 7])
  (h₂ : a + b + c ≡ 3 [MOD 7])
  (ha : a < 7)
  (hb : b < 7)
  (hc : c < 7) :
  2 * a * b * c ≡ 0 [MOD 7] :=
sorry

end remainder_2abc_mod_7_l88_88520


namespace ratio_siblings_l88_88124

theorem ratio_siblings (M J C : ℕ) 
  (hM : M = 60)
  (hJ : J = 4 * M - 60)
  (hJ_C : J = C + 135) :
  (C : ℚ) / M = 3 / 4 :=
by
  sorry

end ratio_siblings_l88_88124


namespace min_value_expression_l88_88122

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ a b, a > 0 ∧ b > 0 ∧ (∀ x y, x > 0 ∧ y > 0 → (1 / x + x / y^2 + y ≥ 2 * Real.sqrt 2)) := 
sorry

end min_value_expression_l88_88122


namespace final_position_correct_total_distance_correct_l88_88010

def movements : List Int := [15, -25, 20, -35]

-- Final Position: 
def final_position (moves : List Int) : Int := moves.sum

-- Total Distance Traveled calculated by taking the absolutes and summing:
def total_distance (moves : List Int) : Nat :=
  moves.map (λ x => Int.natAbs x) |>.sum

theorem final_position_correct : final_position movements = -25 :=
by
  sorry

theorem total_distance_correct : total_distance movements = 95 :=
by
  sorry

end final_position_correct_total_distance_correct_l88_88010


namespace isosceles_triangle_perimeter_l88_88292

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l88_88292


namespace factorization_problem_l88_88166

theorem factorization_problem :
  ∃ (a b : ℤ), (25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) ∧ (a + 3 * b = -86) := by
  sorry

end factorization_problem_l88_88166


namespace total_surface_area_excluding_bases_l88_88440

def lower_base_radius : ℝ := 8
def upper_base_radius : ℝ := 5
def frustum_height : ℝ := 6
def cylinder_section_height : ℝ := 2
def cylinder_section_radius : ℝ := 5

theorem total_surface_area_excluding_bases :
  let l := Real.sqrt (frustum_height ^ 2 + (lower_base_radius - upper_base_radius) ^ 2)
  let lateral_surface_area_frustum := π * (lower_base_radius + upper_base_radius) * l
  let lateral_surface_area_cylinder := 2 * π * cylinder_section_radius * cylinder_section_height
  lateral_surface_area_frustum + lateral_surface_area_cylinder = 39 * π * Real.sqrt 5 + 20 * π :=
by
  sorry

end total_surface_area_excluding_bases_l88_88440


namespace arctan_sum_pi_over_two_l88_88757

theorem arctan_sum_pi_over_two : 
  Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
by sorry

end arctan_sum_pi_over_two_l88_88757


namespace solve_for_y_l88_88950

theorem solve_for_y : (12^3 * 6^2) / 432 = 144 := 
by 
  sorry

end solve_for_y_l88_88950


namespace total_amount_to_be_divided_l88_88326

theorem total_amount_to_be_divided
  (k m x : ℕ)
  (h1 : 18 * k = x)
  (h2 : 20 * m = x)
  (h3 : 13 * m = 11 * k + 1400) :
  x = 36000 := 
sorry

end total_amount_to_be_divided_l88_88326


namespace invitations_per_package_l88_88056

theorem invitations_per_package (total_friends : ℕ) (total_packs : ℕ) (invitations_per_pack : ℕ) 
  (h1 : total_friends = 10) (h2 : total_packs = 5)
  (h3 : invitations_per_pack * total_packs = total_friends) : 
  invitations_per_pack = 2 :=
by
  sorry

end invitations_per_package_l88_88056


namespace determine_k_l88_88143

theorem determine_k (k : ℚ) (h_collinear : ∃ (f : ℚ → ℚ), 
  f 0 = 3 ∧ f 7 = k ∧ f 21 = 2) : k = 8 / 3 :=
by
  sorry

end determine_k_l88_88143


namespace count_perfect_squares_between_l88_88761

theorem count_perfect_squares_between :
  let n := 8
  let m := 70
  (m - n + 1) = 64 :=
by
  -- Definitions and step-by-step proof would go here.
  sorry

end count_perfect_squares_between_l88_88761


namespace map_to_actual_distance_ratio_l88_88738

def distance_in_meters : ℝ := 250
def distance_on_map_cm : ℝ := 5
def cm_per_meter : ℝ := 100

theorem map_to_actual_distance_ratio :
  distance_on_map_cm / (distance_in_meters * cm_per_meter) = 1 / 5000 :=
by
  sorry

end map_to_actual_distance_ratio_l88_88738


namespace find_value_l88_88964

noncomputable def S2013 (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) : ℂ :=
  (x / (x + y))^2013 + (y / (x + y))^2013

theorem find_value (x y : ℂ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : x^2 + x * y + y^2 = 0) :
  S2013 x y h h_eq = -2 :=
sorry

end find_value_l88_88964


namespace intersection_eq_l88_88213

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_eq : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_eq_l88_88213


namespace households_with_car_l88_88562

theorem households_with_car {H_total H_neither H_both H_bike_only : ℕ} 
    (cond1 : H_total = 90)
    (cond2 : H_neither = 11)
    (cond3 : H_both = 22)
    (cond4 : H_bike_only = 35) : 
    H_total - H_neither - (H_bike_only + H_both - H_both) + H_both = 44 := by
  sorry

end households_with_car_l88_88562


namespace isosceles_triangle_side_length_l88_88927

theorem isosceles_triangle_side_length (a b : ℝ) (h : a < b) : 
  ∃ l : ℝ, l = (b - a) / 2 := 
sorry

end isosceles_triangle_side_length_l88_88927


namespace simplify_fraction_l88_88081

theorem simplify_fraction :
  (30 / 35) * (21 / 45) * (70 / 63) - (2 / 3) = - (8 / 15) :=
by
  sorry

end simplify_fraction_l88_88081


namespace rectangular_to_polar_coordinates_l88_88766

theorem rectangular_to_polar_coordinates :
  ∀ (x y : ℝ) (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 →
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) →
  (x, y) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) →
  (r, θ) = (4, Real.pi / 4) :=
by
  intros x y r θ h1 h2 h3
  sorry

end rectangular_to_polar_coordinates_l88_88766


namespace height_of_smaller_cone_removed_l88_88074

noncomputable def frustum_area_lower_base : ℝ := 196 * Real.pi
noncomputable def frustum_area_upper_base : ℝ := 16 * Real.pi
def frustum_height : ℝ := 30

theorem height_of_smaller_cone_removed (r1 r2 H : ℝ)
  (h1 : r1 = Real.sqrt (frustum_area_lower_base / Real.pi))
  (h2 : r2 = Real.sqrt (frustum_area_upper_base / Real.pi))
  (h3 : r2 / r1 = 2 / 7)
  (h4 : frustum_height = (5 / 7) * H) :
  H - frustum_height = 12 :=
by 
  sorry

end height_of_smaller_cone_removed_l88_88074


namespace find_divisor_l88_88752

theorem find_divisor (x d : ℤ) (h1 : ∃ k : ℤ, x = k * d + 5)
                     (h2 : ∃ n : ℤ, x + 17 = n * 41 + 22) :
    d = 1 :=
by
  sorry

end find_divisor_l88_88752


namespace amn_div_l88_88611

theorem amn_div (a m n : ℕ) (a_pos : a > 1) (h : a > 1 ∧ (a^m + 1) ∣ (a^n + 1)) : m ∣ n :=
by sorry

end amn_div_l88_88611


namespace pregnant_dogs_count_l88_88435

-- Definitions as conditions stated in the problem
def total_puppies (P : ℕ) : ℕ := 4 * P
def total_shots (P : ℕ) : ℕ := 2 * total_puppies P
def total_cost (P : ℕ) : ℕ := total_shots P * 5

-- Proof statement without proof
theorem pregnant_dogs_count : ∃ P : ℕ, total_cost P = 120 → P = 3 :=
by sorry

end pregnant_dogs_count_l88_88435


namespace binomials_product_evaluation_l88_88989

-- Define the binomials and the resulting polynomial
def binomial_one (x : ℝ) := 4 * x + 3
def binomial_two (x : ℝ) := 2 * x - 6
def resulting_polynomial (x : ℝ) := 8 * x^2 - 18 * x - 18

-- Define the proof problem
theorem binomials_product_evaluation :
  ∀ (x : ℝ), (binomial_one x) * (binomial_two x) = resulting_polynomial x ∧ 
  resulting_polynomial (-1) = 8 := 
by 
  intro x
  have h1 : (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 := sorry
  have h2 : resulting_polynomial (-1) = 8 := sorry
  exact ⟨h1, h2⟩

end binomials_product_evaluation_l88_88989


namespace total_goals_is_15_l88_88653

-- Define the conditions as variables
def KickersFirstPeriodGoals : ℕ := 2
def KickersSecondPeriodGoals : ℕ := 2 * KickersFirstPeriodGoals
def SpidersFirstPeriodGoals : ℕ := KickersFirstPeriodGoals / 2
def SpidersSecondPeriodGoals : ℕ := 2 * KickersSecondPeriodGoals

-- Define total goals by each team
def TotalKickersGoals : ℕ := KickersFirstPeriodGoals + KickersSecondPeriodGoals
def TotalSpidersGoals : ℕ := SpidersFirstPeriodGoals + SpidersSecondPeriodGoals

-- Define total goals by both teams
def TotalGoals : ℕ := TotalKickersGoals + TotalSpidersGoals

-- Prove the statement
theorem total_goals_is_15 : TotalGoals = 15 :=
by
  sorry

end total_goals_is_15_l88_88653


namespace compute_expression_l88_88498

theorem compute_expression : 1013^2 - 991^2 - 1007^2 + 997^2 = 24048 := by
  sorry

end compute_expression_l88_88498


namespace baker_cakes_left_l88_88336

theorem baker_cakes_left (cakes_made cakes_bought : ℕ) (h1 : cakes_made = 155) (h2 : cakes_bought = 140) : cakes_made - cakes_bought = 15 := by
  sorry

end baker_cakes_left_l88_88336


namespace largest_integer_solution_l88_88060

theorem largest_integer_solution :
  ∀ (x : ℤ), x - 5 > 3 * x - 1 → x ≤ -3 := by
  sorry

end largest_integer_solution_l88_88060


namespace range_of_a_l88_88293

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * x + 3 ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end range_of_a_l88_88293


namespace tan_add_pi_over_six_l88_88838

theorem tan_add_pi_over_six (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + Real.pi / 6) = 5 + 2 * Real.sqrt 3 :=
sorry

end tan_add_pi_over_six_l88_88838


namespace complement_intersection_l88_88141

def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection : (M ∩ N)ᶜ = { x : ℝ | x < 1 ∨ x > 3 } :=
  sorry

end complement_intersection_l88_88141


namespace equation_of_line_l88_88539

theorem equation_of_line (P : ℝ × ℝ) (m : ℝ) : 
  P = (3, 3) → m = 2 * 1 → ∃ b : ℝ, ∀ x : ℝ, P.2 = m * (x - P.1) + b ↔ y = 2 * x - 3 := 
by {
  sorry
}

end equation_of_line_l88_88539


namespace probability_within_three_units_from_origin_l88_88955

-- Define the properties of the square Q is selected from
def isInSquare (Q : ℝ × ℝ) : Prop :=
  Q.1 ≥ -2 ∧ Q.1 ≤ 2 ∧ Q.2 ≥ -2 ∧ Q.2 ≤ 2

-- Define the condition of being within 3 units from the origin
def withinThreeUnits (Q: ℝ × ℝ) : Prop :=
  (Q.1)^2 + (Q.2)^2 ≤ 9

-- State the problem: Proving the probability is 1
theorem probability_within_three_units_from_origin : 
  ∀ (Q : ℝ × ℝ), isInSquare Q → withinThreeUnits Q := 
by 
  sorry

end probability_within_three_units_from_origin_l88_88955


namespace parallelogram_area_l88_88403

theorem parallelogram_area (base height : ℝ) (h_base : base = 22) (h_height : height = 14) :
  base * height = 308 := by
  sorry

end parallelogram_area_l88_88403


namespace fraction_relation_l88_88670

theorem fraction_relation 
  (m n p q : ℚ)
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 14) : 
  m / q = 3 / 14 :=
by
  sorry

end fraction_relation_l88_88670


namespace water_consumption_l88_88195

theorem water_consumption (num_cows num_goats num_pigs num_sheep : ℕ)
  (water_per_cow water_per_goat water_per_pig water_per_sheep daily_total weekly_total : ℕ)
  (h1 : num_cows = 40)
  (h2 : num_goats = 25)
  (h3 : num_pigs = 30)
  (h4 : water_per_cow = 80)
  (h5 : water_per_goat = water_per_cow / 2)
  (h6 : water_per_pig = water_per_cow / 3)
  (h7 : num_sheep = 10 * num_cows)
  (h8 : water_per_sheep = water_per_cow / 4)
  (h9 : daily_total = num_cows * water_per_cow + num_goats * water_per_goat + num_pigs * water_per_pig + num_sheep * water_per_sheep)
  (h10 : weekly_total = daily_total * 7) :
  weekly_total = 91000 := by
  sorry

end water_consumption_l88_88195


namespace total_cost_eq_16000_l88_88802

theorem total_cost_eq_16000 (F M T : ℕ) (n : ℕ) (hF : F = 12000) (hM : M = 200) (hT : T = 16000) :
  T = F + M * n → n = 20 :=
by
  sorry

end total_cost_eq_16000_l88_88802


namespace combined_collectors_edition_dolls_l88_88516

-- Definitions based on given conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def luna_dolls : ℕ := ivy_dolls - 10

-- Additional constraints based on the problem statement
def total_dolls : ℕ := dina_dolls + ivy_dolls + luna_dolls
def ivy_collectors_edition_dolls : ℕ := 2/3 * ivy_dolls
def luna_collectors_edition_dolls : ℕ := 1/2 * luna_dolls

-- Proof statement
theorem combined_collectors_edition_dolls :
  ivy_collectors_edition_dolls + luna_collectors_edition_dolls = 30 :=
sorry

end combined_collectors_edition_dolls_l88_88516


namespace trajectory_midpoint_l88_88082

-- Define the hyperbola equation
def hyperbola (x y : ℝ) := x^2 - (y^2 / 4) = 1

-- Define the condition that a line passes through the point (0, 1)
def line_through_fixed_point (k x y : ℝ) := y = k * x + 1

-- Define the theorem to prove the trajectory of the midpoint of the chord
theorem trajectory_midpoint (x y k : ℝ) (h : ∃ x y, hyperbola x y ∧ line_through_fixed_point k x y) : 
    4 * x^2 - y^2 + y = 0 := 
sorry

end trajectory_midpoint_l88_88082


namespace selection_methods_count_l88_88609

theorem selection_methods_count
  (multiple_choice_questions : ℕ)
  (fill_in_the_blank_questions : ℕ)
  (h1 : multiple_choice_questions = 9)
  (h2 : fill_in_the_blank_questions = 3) :
  multiple_choice_questions + fill_in_the_blank_questions = 12 := by
  sorry

end selection_methods_count_l88_88609


namespace problem_inequality_l88_88088

variable {α : Type*} [LinearOrder α]

def M (x y : α) : α := max x y
def m (x y : α) : α := min x y

theorem problem_inequality (a b c d e : α) (h : a < b) (h1 : b < c) (h2 : c < d) (h3 : d < e) : 
  M (M a (m b c)) (m d (m a e)) = b := sorry

end problem_inequality_l88_88088


namespace find_integer_pairs_l88_88436

theorem find_integer_pairs (x y : ℕ) (h : x ^ 5 = y ^ 5 + 10 * y ^ 2 + 20 * y + 1) : (x, y) = (1, 0) :=
  sorry

end find_integer_pairs_l88_88436


namespace time_to_drain_tank_l88_88604

theorem time_to_drain_tank (P L: ℝ) (hP : P = 1/3) (h_combined : P - L = 2/7) : 1 / L = 21 :=
by
  -- Proof omitted. Use the conditions given to show that 1 / L = 21.
  sorry

end time_to_drain_tank_l88_88604


namespace simplify_expression_l88_88126

theorem simplify_expression (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 :=
by
  sorry

end simplify_expression_l88_88126


namespace largest_inscribed_circle_radius_l88_88131

theorem largest_inscribed_circle_radius (k : ℝ) (h_perimeter : 0 < k) :
  ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2) :=
by
  have h_r : ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2)
  exact ⟨(k / 2) * (3 - 2 * Real.sqrt 2), rfl⟩
  exact h_r

end largest_inscribed_circle_radius_l88_88131


namespace candy_comparison_l88_88552

variable (skittles_bryan : ℕ)
variable (gummy_bears_bryan : ℕ)
variable (chocolate_bars_bryan : ℕ)
variable (mms_ben : ℕ)
variable (jelly_beans_ben : ℕ)
variable (lollipops_ben : ℕ)

def bryan_total_candies := skittles_bryan + gummy_bears_bryan + chocolate_bars_bryan
def ben_total_candies := mms_ben + jelly_beans_ben + lollipops_ben

def difference_skittles_mms := skittles_bryan - mms_ben
def difference_gummy_jelly := jelly_beans_ben - gummy_bears_bryan
def difference_choco_lollipops := chocolate_bars_bryan - lollipops_ben

def sum_of_differences := difference_skittles_mms + difference_gummy_jelly + difference_choco_lollipops

theorem candy_comparison
  (h_bryan_skittles : skittles_bryan = 50)
  (h_bryan_gummy_bears : gummy_bears_bryan = 25)
  (h_bryan_choco_bars : chocolate_bars_bryan = 15)
  (h_ben_mms : mms_ben = 20)
  (h_ben_jelly_beans : jelly_beans_ben = 30)
  (h_ben_lollipops : lollipops_ben = 10) :
  bryan_total_candies = 90 ∧
  ben_total_candies = 60 ∧
  bryan_total_candies > ben_total_candies ∧
  difference_skittles_mms = 30 ∧
  difference_gummy_jelly = 5 ∧
  difference_choco_lollipops = 5 ∧
  sum_of_differences = 40 := by
  sorry

end candy_comparison_l88_88552


namespace isosceles_triangle_angles_l88_88606

theorem isosceles_triangle_angles (y : ℝ) (h : y > 0) :
  let P := y
  let R := 5 * y
  let Q := R
  P + Q + R = 180 → Q = 81.82 :=
by
  sorry

end isosceles_triangle_angles_l88_88606


namespace sticks_problem_solution_l88_88630

theorem sticks_problem_solution :
  ∃ n : ℕ, n > 0 ∧ 1012 = 2 * n * (n + 1) ∧ 1012 > 1000 ∧ 
           1012 % 3 = 1 ∧ 1012 % 5 = 2 :=
by
  sorry

end sticks_problem_solution_l88_88630


namespace find_q_l88_88249

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 6) : q = 3 + Real.sqrt 3 :=
by
  sorry

end find_q_l88_88249


namespace average_of_remaining_numbers_l88_88239

theorem average_of_remaining_numbers (s : ℝ) (a b c d e f : ℝ)
  (h1: (a + b + c + d + e + f) / 6 = 3.95)
  (h2: (a + b) / 2 = 4.4)
  (h3: (c + d) / 2 = 3.85) :
  ((e + f) / 2 = 3.6) :=
by
  sorry

end average_of_remaining_numbers_l88_88239


namespace jessica_walks_distance_l88_88104

theorem jessica_walks_distance (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 :=
by 
  rw [h_rate, h_time]
  norm_num

end jessica_walks_distance_l88_88104


namespace distance_traveled_l88_88666

-- Let T be the time in hours taken to travel the actual distance D at 10 km/hr.
-- Let D be the actual distance traveled by the person.
-- Given: D = 10 * T and D + 40 = 20 * T prove that D = 40.

theorem distance_traveled (T : ℝ) (D : ℝ) 
  (h1 : D = 10 * T)
  (h2 : D + 40 = 20 * T) : 
  D = 40 := by
  sorry

end distance_traveled_l88_88666


namespace smallest_lcm_value_l88_88873

def is_five_digit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000

theorem smallest_lcm_value :
  ∃ (m n : ℕ), is_five_digit m ∧ is_five_digit n ∧ Nat.gcd m n = 5 ∧ Nat.lcm m n = 20030010 :=
by
  sorry

end smallest_lcm_value_l88_88873


namespace base8_subtraction_l88_88812

theorem base8_subtraction : (52 - 27 : ℕ) = 23 := by sorry

end base8_subtraction_l88_88812


namespace min_a2_b2_l88_88004

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + 2 * x^2 + b * x + 1 = 0) : a^2 + b^2 ≥ 8 :=
sorry

end min_a2_b2_l88_88004


namespace arithmetic_sequence_geometric_subsequence_l88_88409

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) = a n + 1)
  (h2 : (a 3)^2 = a 1 * a 7) :
  a 5 = 6 :=
sorry

end arithmetic_sequence_geometric_subsequence_l88_88409


namespace point_B_third_quadrant_l88_88205

theorem point_B_third_quadrant (m n : ℝ) (hm : m < 0) (hn : n < 0) :
  (-m * n < 0) ∧ (m < 0) :=
by
  sorry

end point_B_third_quadrant_l88_88205


namespace blue_books_count_l88_88110

def number_of_blue_books (R B : ℕ) (p : ℚ) : Prop :=
  R = 4 ∧ p = 3/14 → B^2 + 7 * B - 44 = 0

theorem blue_books_count :
  ∃ B : ℕ, number_of_blue_books 4 B (3/14) ∧ B = 4 :=
by
  sorry

end blue_books_count_l88_88110


namespace car_p_less_hours_l88_88744

theorem car_p_less_hours (distance : ℕ) (speed_r : ℕ) (speed_p : ℕ) (time_r : ℕ) (time_p : ℕ) (h1 : distance = 600) (h2 : speed_r = 50) (h3 : speed_p = speed_r + 10) (h4 : time_r = distance / speed_r) (h5 : time_p = distance / speed_p) : time_r - time_p = 2 := 
by
  sorry

end car_p_less_hours_l88_88744


namespace percentage_difference_l88_88857

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.60 * x) (h2 : z = 0.60 * y) :
  abs ((z - x) / z * 100) = 4.17 :=
by
  sorry

end percentage_difference_l88_88857


namespace cost_of_one_pie_l88_88921

theorem cost_of_one_pie (x c2 c5 : ℕ) 
  (h1: 4 * x = c2 + 60)
  (h2: 5 * x = c5 + 60) 
  (h3: 6 * x = c2 + c5 + 60) : 
  x = 20 :=
by
  sorry

end cost_of_one_pie_l88_88921


namespace ratio_child_to_jane_babysit_l88_88225

-- Definitions of the conditions
def jane_current_age : ℕ := 32
def years_since_jane_stopped_babysitting : ℕ := 10
def oldest_person_current_age : ℕ := 24

-- Derived definitions
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped_babysitting
def oldest_person_age_when_jane_stopped : ℕ := oldest_person_current_age - years_since_jane_stopped_babysitting

-- Statement of the problem to be proven in Lean 4
theorem ratio_child_to_jane_babysit :
  (oldest_person_age_when_jane_stopped : ℚ) / (jane_age_when_stopped : ℚ) = 7 / 11 :=
by
  sorry

end ratio_child_to_jane_babysit_l88_88225


namespace cheesecake_factory_working_days_l88_88259

-- Define the savings rates
def robby_saves := 2 / 5
def jaylen_saves := 3 / 5
def miranda_saves := 1 / 2

-- Define their hourly rate and daily working hours
def hourly_rate := 10 -- dollars per hour
def work_hours_per_day := 10 -- hours per day

-- Define their combined savings after four weeks and the combined savings target
def four_weeks := 4 * 7
def combined_savings_target := 3000 -- dollars

-- Question: Prove that the number of days they work per week is 7
theorem cheesecake_factory_working_days (d : ℕ) (h : d * 400 = combined_savings_target / 4) : d = 7 := sorry

end cheesecake_factory_working_days_l88_88259


namespace fraction_product_l88_88350

theorem fraction_product :
  ((1: ℚ) / 2) * (3 / 5) * (7 / 11) = 21 / 110 :=
by {
  sorry
}

end fraction_product_l88_88350


namespace S_when_R_is_16_and_T_is_1_div_4_l88_88665

theorem S_when_R_is_16_and_T_is_1_div_4 :
  ∃ (S : ℝ), (∀ (R S T : ℝ) (c : ℝ), (R = c * S / T) →
  (2 = c * 8 / (1/2)) → c = 1 / 8) ∧
  (16 = (1/8) * S / (1/4)) → S = 32 :=
sorry

end S_when_R_is_16_and_T_is_1_div_4_l88_88665


namespace Jan_older_than_Cindy_l88_88923

noncomputable def Cindy_age : ℕ := 5
noncomputable def Greg_age : ℕ := 16

variables (Marcia_age Jan_age : ℕ)

axiom Greg_and_Marcia : Greg_age = Marcia_age + 2
axiom Marcia_and_Jan : Marcia_age = 2 * Jan_age

theorem Jan_older_than_Cindy : (Jan_age - Cindy_age) = 2 :=
by
  -- Insert proof here
  sorry

end Jan_older_than_Cindy_l88_88923


namespace time_period_for_investment_l88_88534

variable (P R₁₅ R₁₀ I₁₅ I₁₀ : ℝ)
variable (T : ℝ)

noncomputable def principal := 8400
noncomputable def rate15 := 15
noncomputable def rate10 := 10
noncomputable def interestDifference := 840

theorem time_period_for_investment :
  ∀ (T : ℝ),
    P = principal →
    R₁₅ = rate15 →
    R₁₀ = rate10 →
    I₁₅ = P * (R₁₅ / 100) * T →
    I₁₀ = P * (R₁₀ / 100) * T →
    (I₁₅ - I₁₀) = interestDifference →
    T = 2 :=
  sorry

end time_period_for_investment_l88_88534


namespace probability_closer_to_center_radius6_eq_1_4_l88_88449

noncomputable def probability_closer_to_center (radius : ℝ) (r_inner : ℝ) :=
    let area_outer := Real.pi * radius ^ 2
    let area_inner := Real.pi * r_inner ^ 2
    area_inner / area_outer

theorem probability_closer_to_center_radius6_eq_1_4 :
    probability_closer_to_center 6 3 = 1 / 4 := by
    sorry

end probability_closer_to_center_radius6_eq_1_4_l88_88449


namespace range_of_k_for_real_roots_l88_88217

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 = x2 ∧ x^2 - 2*x + k = 0) ↔ k ≤ 1 := 
by
  sorry

end range_of_k_for_real_roots_l88_88217


namespace probability_of_picking_letter_in_mathematics_l88_88412

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_picking_letter_in_mathematics :
  (unique_letters_in_mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l88_88412


namespace shirts_per_minute_l88_88089

theorem shirts_per_minute (shirts_in_6_minutes : ℕ) (time_minutes : ℕ) (h1 : shirts_in_6_minutes = 36) (h2 : time_minutes = 6) : 
  ((shirts_in_6_minutes / time_minutes) = 6) :=
by
  sorry

end shirts_per_minute_l88_88089


namespace part1_part2_l88_88233

open Complex

-- Define the first proposition p
def p (m : ℝ) : Prop :=
  (m - 1 < 0) ∧ (m + 3 > 0)

-- Define the second proposition q
def q (m : ℝ) : Prop :=
  abs (Complex.mk 1 (m - 2)) ≤ Real.sqrt 10

-- Prove the first part of the problem
theorem part1 (m : ℝ) (hp : p m) : -3 < m ∧ m < 1 :=
sorry

-- Prove the second part of the problem
theorem part2 (m : ℝ) (h : ¬ (p m ∧ q m) ∧ (p m ∨ q m)) : (-3 < m ∧ m < -1) ∨ (1 ≤ m ∧ m ≤ 5) :=
sorry

end part1_part2_l88_88233


namespace length_of_courtyard_l88_88573

-- Define the dimensions and properties of the courtyard and paving stones
def width := 33 / 2
def numPavingStones := 132
def pavingStoneLength := 5 / 2
def pavingStoneWidth := 2

-- Total area covered by paving stones
def totalArea := numPavingStones * (pavingStoneLength * pavingStoneWidth)

-- To prove: Length of the courtyard
theorem length_of_courtyard : totalArea / width = 40 := by
  sorry

end length_of_courtyard_l88_88573


namespace hens_count_l88_88480

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 144) 
  (h3 : H ≥ 10) (h4 : C ≥ 5) : H = 24 :=
by
  sorry

end hens_count_l88_88480


namespace number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l88_88999

theorem number_of_whole_numbers_between_sqrt_18_and_sqrt_120 : 
  ∀ (n : ℕ), 
  (5 ≤ n ∧ n ≤ 10) ↔ (6 = 6) :=
sorry

end number_of_whole_numbers_between_sqrt_18_and_sqrt_120_l88_88999


namespace total_handshakes_at_convention_l88_88054

theorem total_handshakes_at_convention :
  let gremlins := 25
  let imps := 18
  let specific_gremlins := 5
  let friendly_gremlins := gremlins - specific_gremlins
  let handshakes_among_gremlins := (friendly_gremlins * (friendly_gremlins - 1)) / 2
  let handshakes_between_imps_and_gremlins := imps * gremlins
  handshakes_among_gremlins + handshakes_between_imps_and_gremlins = 640 := by
  sorry

end total_handshakes_at_convention_l88_88054


namespace original_price_of_car_l88_88100

theorem original_price_of_car (P : ℝ) 
  (h₁ : 0.561 * P + 200 = 7500) : 
  P = 13012.48 := 
sorry

end original_price_of_car_l88_88100


namespace cost_of_first_ring_is_10000_l88_88012

theorem cost_of_first_ring_is_10000 (x : ℝ) (h₁ : x + 2*x - x/2 = 25000) : x = 10000 :=
sorry

end cost_of_first_ring_is_10000_l88_88012


namespace y_intercept_of_line_l88_88907

theorem y_intercept_of_line : ∀ (x y : ℝ), (5 * x - 2 * y - 10 = 0) → (x = 0) → (y = -5) :=
by
  intros x y h1 h2
  sorry

end y_intercept_of_line_l88_88907


namespace max_value_expression_l88_88419

variable (a b : ℝ)

theorem max_value_expression (h : a^2 + b^2 = 3 + a * b) : 
  ∃ a b : ℝ, (2 * a - 3 * b)^2 + (a + 2 * b) * (a - 2 * b) = 22 :=
by
  -- This is a placeholder for the actual proof
  sorry

end max_value_expression_l88_88419


namespace jack_jill_next_in_step_l88_88495

theorem jack_jill_next_in_step (stride_jack : ℕ) (stride_jill : ℕ) : 
  stride_jack = 64 → stride_jill = 56 → Nat.lcm stride_jack stride_jill = 448 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end jack_jill_next_in_step_l88_88495


namespace expand_product_l88_88788

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 :=
by
  sorry

end expand_product_l88_88788


namespace additional_treetags_l88_88496

noncomputable def initial_numerals : Finset ℕ := {1, 2, 3, 4}
noncomputable def initial_letters : Finset Char := {'A', 'E', 'I'}
noncomputable def initial_symbols : Finset Char := {'!', '@', '#', '$'}
noncomputable def added_numeral : Finset ℕ := {5}
noncomputable def added_symbols : Finset Char := {'&'}

theorem additional_treetags : 
  let initial_treetags := initial_numerals.card * initial_letters.card * initial_symbols.card
  let new_numerals := initial_numerals ∪ added_numeral
  let new_symbols := initial_symbols ∪ added_symbols
  let new_treetags := new_numerals.card * initial_letters.card * new_symbols.card
  new_treetags - initial_treetags = 27 := 
by 
  sorry

end additional_treetags_l88_88496


namespace find_x_plus_y_l88_88780

variable (x y : ℝ)

theorem find_x_plus_y (h1 : |x| + x + y = 8) (h2 : x + |y| - y = 10) : x + y = 14 / 5 := 
by
  sorry

end find_x_plus_y_l88_88780


namespace negative_y_implies_negative_y_is_positive_l88_88834

theorem negative_y_implies_negative_y_is_positive (y : ℝ) (h : y < 0) : -y > 0 :=
sorry

end negative_y_implies_negative_y_is_positive_l88_88834


namespace max_cone_cross_section_area_l88_88077

theorem max_cone_cross_section_area
  (V A B : Type)
  (E : Type)
  (l : ℝ)
  (α : ℝ) :
  0 < l ∧ 0 < α ∧ α < 180 → 
  ∃ (area : ℝ), area = (1 / 2) * l^2 :=
by
  sorry

end max_cone_cross_section_area_l88_88077


namespace find_integers_l88_88407

theorem find_integers (A B C : ℤ) (hA : A = 500) (hB : B = -1) (hC : C = -500) : 
  (A : ℚ) / 999 + (B : ℚ) / 1000 + (C : ℚ) / 1001 = 1 / (999 * 1000 * 1001) :=
by 
  rw [hA, hB, hC]
  sorry

end find_integers_l88_88407


namespace insurance_covers_90_percent_l88_88352

-- We firstly define the variables according to the conditions.
def adoption_fee : ℕ := 150
def training_cost_per_week : ℕ := 250
def training_weeks : ℕ := 12
def certification_cost : ℕ := 3000
def total_out_of_pocket_cost : ℕ := 3450

-- We now compute intermediate results based on the conditions provided.
def total_training_cost : ℕ := training_cost_per_week * training_weeks
def out_of_pocket_cert_cost : ℕ := total_out_of_pocket_cost - adoption_fee - total_training_cost
def insurance_coverage_amount : ℕ := certification_cost - out_of_pocket_cert_cost
def insurance_coverage_percentage : ℕ := (insurance_coverage_amount * 100) / certification_cost

-- Now, we state the theorem that needs to be proven.
theorem insurance_covers_90_percent : insurance_coverage_percentage = 90 := by
  sorry

end insurance_covers_90_percent_l88_88352


namespace devin_teaching_years_l88_88353

section DevinTeaching
variable (Calculus Algebra Statistics Geometry DiscreteMathematics : ℕ)

theorem devin_teaching_years :
  Calculus = 4 ∧
  Algebra = 2 * Calculus ∧
  Statistics = 5 * Algebra ∧
  Geometry = 3 * Statistics ∧
  DiscreteMathematics = Geometry / 2 ∧
  (Calculus + Algebra + Statistics + Geometry + DiscreteMathematics) = 232 :=
by
  sorry
end DevinTeaching

end devin_teaching_years_l88_88353


namespace sale_prices_correct_l88_88996

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

end sale_prices_correct_l88_88996


namespace sin_neg_600_eq_sqrt_3_div_2_l88_88458

theorem sin_neg_600_eq_sqrt_3_div_2 :
  Real.sin (-(600 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
sorry

end sin_neg_600_eq_sqrt_3_div_2_l88_88458


namespace largest_band_members_l88_88654

theorem largest_band_members 
  (r x m : ℕ) 
  (h1 : (r * x + 3 = m)) 
  (h2 : ((r - 3) * (x + 1) = m))
  (h3 : m < 100) : 
  m = 75 :=
sorry

end largest_band_members_l88_88654


namespace height_of_each_step_l88_88791

-- Define the number of steps in each staircase
def first_staircase_steps : ℕ := 20
def second_staircase_steps : ℕ := 2 * first_staircase_steps
def third_staircase_steps : ℕ := second_staircase_steps - 10

-- Define the total steps climbed
def total_steps_climbed : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

-- Define the total height climbed
def total_height_climbed : ℝ := 45

-- Prove the height of each step
theorem height_of_each_step : (total_height_climbed / total_steps_climbed) = 0.5 := by
  sorry

end height_of_each_step_l88_88791


namespace smaller_mold_radius_l88_88224

theorem smaller_mold_radius (R : ℝ) (third_volume_sharing : ℝ) (molds_count : ℝ) (r : ℝ) 
  (hR : R = 3) 
  (h_third_volume_sharing : third_volume_sharing = 1/3) 
  (h_molds_count : molds_count = 9) 
  (h_r : (2/3) * Real.pi * r^3 = (2/3) * Real.pi / molds_count) : 
  r = 1 := 
by
  sorry

end smaller_mold_radius_l88_88224


namespace positive_difference_sum_of_squares_l88_88647

-- Given definitions
def sum_of_squares_even (n : ℕ) : ℕ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6

def sum_of_squares_odd (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- The explicit values for this problem
def sum_of_squares_first_25_even := sum_of_squares_even 25
def sum_of_squares_first_20_odd := sum_of_squares_odd 20

-- The required proof statement
theorem positive_difference_sum_of_squares : 
  (sum_of_squares_first_25_even - sum_of_squares_first_20_odd) = 19230 := by
  sorry

end positive_difference_sum_of_squares_l88_88647


namespace cos_sum_identity_cosine_30_deg_l88_88167

theorem cos_sum_identity : 
  (Real.cos (Real.pi * 43 / 180) * Real.cos (Real.pi * 13 / 180) + 
   Real.sin (Real.pi * 43 / 180) * Real.sin (Real.pi * 13 / 180)) = 
   (Real.cos (Real.pi * 30 / 180)) :=
sorry

theorem cosine_30_deg : 
  Real.cos (Real.pi * 30 / 180) = (Real.sqrt 3 / 2) :=
sorry

end cos_sum_identity_cosine_30_deg_l88_88167


namespace batsman_average_increase_l88_88778

theorem batsman_average_increase (A : ℝ) (X : ℝ) (runs_11th_inning : ℝ) (average_11th_inning : ℝ) 
  (h_runs_11th_inning : runs_11th_inning = 85) 
  (h_average_11th_inning : average_11th_inning = 35) 
  (h_eq : (10 * A + runs_11th_inning) / 11 = average_11th_inning) :
  X = 5 := 
by 
  sorry

end batsman_average_increase_l88_88778


namespace total_trapezoid_area_l88_88142

def large_trapezoid_area (AB CD altitude_L : ℝ) : ℝ :=
  0.5 * (AB + CD) * altitude_L

def small_trapezoid_area (EF GH altitude_S : ℝ) : ℝ :=
  0.5 * (EF + GH) * altitude_S

def total_area (large_area small_area : ℝ) : ℝ :=
  large_area + small_area

theorem total_trapezoid_area :
  large_trapezoid_area 60 30 15 + small_trapezoid_area 25 10 5 = 762.5 :=
by
  -- proof goes here
  sorry

end total_trapezoid_area_l88_88142


namespace stream_speed_l88_88586

theorem stream_speed (v : ℝ) : 
  (∀ (speed_boat_in_still_water distance time : ℝ), 
    speed_boat_in_still_water = 25 ∧ distance = 90 ∧ time = 3 →
    distance = (speed_boat_in_still_water + v) * time) →
  v = 5 :=
by
  intro h
  have h1 := h 25 90 3 ⟨rfl, rfl, rfl⟩
  sorry

end stream_speed_l88_88586


namespace steve_halfway_time_longer_l88_88050

theorem steve_halfway_time_longer :
  ∀ (Td: ℝ) (Ts: ℝ),
  Td = 33 →
  Ts = 2 * Td →
  (Ts / 2) - (Td / 2) = 16.5 :=
by
  intros Td Ts hTd hTs
  rw [hTd, hTs]
  sorry

end steve_halfway_time_longer_l88_88050


namespace total_airflow_in_one_week_l88_88347

-- Define the conditions
def airflow_rate : ℕ := 10 -- liters per second
def working_time_per_day : ℕ := 10 -- minutes per day
def days_per_week : ℕ := 7

-- Define the conversion factors
def minutes_to_seconds : ℕ := 60

-- Define the total working time in seconds
def total_working_time_per_week : ℕ := working_time_per_day * days_per_week * minutes_to_seconds

-- Define the expected total airflow in one week
def expected_total_airflow : ℕ := airflow_rate * total_working_time_per_week

-- Prove that the expected total airflow is 42000 liters
theorem total_airflow_in_one_week : expected_total_airflow = 42000 := 
by
  -- assertion is correct given the conditions above 
  -- skip the proof
  sorry

end total_airflow_in_one_week_l88_88347


namespace trig_identity_example_l88_88754

theorem trig_identity_example :
  (Real.cos (47 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) - 
   Real.sin (47 * Real.pi / 180) * Real.sin (13 * Real.pi / 180)) = 
  (Real.cos (60 * Real.pi / 180)) := by
  sorry

end trig_identity_example_l88_88754


namespace no_ordered_triples_l88_88625

theorem no_ordered_triples (x y z : ℕ)
  (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) :
  x * y * z + 2 * (x * y + y * z + z * x) ≠ 2 * (2 * (x * y + y * z + z * x)) + 12 :=
by {
  sorry
}

end no_ordered_triples_l88_88625


namespace smallest_of_five_consecutive_l88_88228

theorem smallest_of_five_consecutive (n : ℤ) (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 2015) : n - 2 = 401 :=
by sorry

end smallest_of_five_consecutive_l88_88228


namespace range_of_f_l88_88717

noncomputable def f (x : ℝ) : ℝ :=
  x + Real.sqrt (x - 2)

theorem range_of_f : Set.range f = {y : ℝ | 2 ≤ y} :=
by
  sorry

end range_of_f_l88_88717


namespace increasing_interval_l88_88145

noncomputable def f (x : ℝ) := Real.log x / Real.log (1 / 2)

def is_monotonically_increasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def h (x : ℝ) : ℝ := x^2 + x - 2

theorem increasing_interval :
  is_monotonically_increasing (f ∘ h) {x : ℝ | x < -2} :=
sorry

end increasing_interval_l88_88145


namespace brass_weight_l88_88026

theorem brass_weight (copper zinc brass : ℝ) (h_ratio : copper / zinc = 3 / 7) (h_zinc : zinc = 70) : brass = 100 :=
by
  sorry

end brass_weight_l88_88026


namespace probability_after_5_rounds_l88_88481

def initial_coins : ℕ := 5
def rounds : ℕ := 5
def final_probability : ℚ := 1 / 2430000

structure Player :=
  (name : String)
  (initial_coins : ℕ)
  (final_coins : ℕ)

def Abby : Player := ⟨"Abby", 5, 5⟩
def Bernardo : Player := ⟨"Bernardo", 4, 3⟩
def Carl : Player := ⟨"Carl", 3, 3⟩
def Debra : Player := ⟨"Debra", 4, 5⟩

def check_final_state (players : List Player) : Prop :=
  ∀ (p : Player), p ∈ players →
  (p.name = "Abby" ∧ p.final_coins = 5 ∨
   p.name = "Bernardo" ∧ p.final_coins = 3 ∨
   p.name = "Carl" ∧ p.final_coins = 3 ∨
   p.name = "Debra" ∧ p.final_coins = 5)

theorem probability_after_5_rounds :
  ∃ prob : ℚ, prob = final_probability ∧ check_final_state [Abby, Bernardo, Carl, Debra] :=
sorry

end probability_after_5_rounds_l88_88481


namespace least_possible_xy_l88_88441

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l88_88441


namespace gain_per_year_is_correct_l88_88270

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem gain_per_year_is_correct :
  let borrowed_amount := 7000
  let borrowed_rate := 0.04
  let borrowed_time := 2
  let borrowed_compound_freq := 1 -- annually
  
  let lent_amount := 7000
  let lent_rate := 0.06
  let lent_time := 2
  let lent_compound_freq := 2 -- semi-annually
  
  let amount_owed := compound_interest borrowed_amount borrowed_rate borrowed_compound_freq borrowed_time
  let amount_received := compound_interest lent_amount lent_rate lent_compound_freq lent_time
  let total_gain := amount_received - amount_owed
  let gain_per_year := total_gain / lent_time
  
  gain_per_year = 153.65 :=
by
  sorry

end gain_per_year_is_correct_l88_88270


namespace islands_not_connected_by_bridges_for_infinitely_many_primes_l88_88366

open Nat

theorem islands_not_connected_by_bridges_for_infinitely_many_primes :
  ∃ᶠ p in at_top, ∃ n m : ℕ, n ≠ m ∧ ¬(p ∣ (n^2 - m + 1) * (m^2 - n + 1)) :=
sorry

end islands_not_connected_by_bridges_for_infinitely_many_primes_l88_88366


namespace rational_solution_system_l88_88720

theorem rational_solution_system (x y z t w : ℚ) :
  (t^2 - w^2 + z^2 = 2 * x * y) →
  (t^2 - y^2 + w^2 = 2 * x * z) →
  (t^2 - w^2 + x^2 = 2 * y * z) →
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros h1 h2 h3
  sorry

end rational_solution_system_l88_88720


namespace sequence_diff_l88_88240

theorem sequence_diff (x : ℕ → ℕ)
  (h1 : ∀ n, x n < x (n + 1))
  (h2 : ∀ n, 2 * n + 1 ≤ x (2 * n + 1)) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_l88_88240


namespace sum_last_two_digits_9_pow_23_plus_11_pow_23_l88_88568

theorem sum_last_two_digits_9_pow_23_plus_11_pow_23 :
  (9^23 + 11^23) % 100 = 60 :=
by
  sorry

end sum_last_two_digits_9_pow_23_plus_11_pow_23_l88_88568


namespace first_pack_weight_l88_88544

variable (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
variable (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ)

theorem first_pack_weight (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
    (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ) :
    hiking_rate = 2.5 →
    hours_per_day = 9 →
    days = 7 →
    pounds_per_mile = 0.6 →
    first_resupply_percentage = 0.30 →
    second_resupply_percentage = 0.20 →
    ∃ first_pack : ℝ, first_pack = 47.25 :=
by
  intro h1 h2 h3 h4 h5 h6
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := pounds_per_mile * total_distance
  let first_resupply := total_supplies * first_resupply_percentage
  let second_resupply := total_supplies * second_resupply_percentage
  let first_pack := total_supplies - (first_resupply + second_resupply)
  use first_pack
  sorry

end first_pack_weight_l88_88544


namespace unique_H_value_l88_88442

theorem unique_H_value :
  ∀ (T H R E F I V S : ℕ),
    T = 8 →
    E % 2 = 1 →
    E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ F ∧ E ≠ I ∧ E ≠ V ∧ E ≠ S ∧ 
    H ≠ T ∧ H ≠ R ∧ H ≠ F ∧ H ≠ I ∧ H ≠ V ∧ H ≠ S ∧
    F ≠ T ∧ F ≠ I ∧ F ≠ V ∧ F ≠ S ∧
    I ≠ T ∧ I ≠ V ∧ I ≠ S ∧
    V ≠ T ∧ V ≠ S ∧
    S ≠ T ∧
    (8 + 8) = 10 + F ∧
    (E + E) % 10 = 6 →
    H + H = 10 + 4 →
    H = 7 := 
sorry

end unique_H_value_l88_88442


namespace sum_of_ages_five_years_ago_l88_88633

-- Definitions from the conditions
variables (A B : ℕ) -- Angela's current age and Beth's current age

-- Conditions
def angela_is_four_times_as_old_as_beth := A = 4 * B
def angela_will_be_44_in_five_years := A + 5 = 44

-- Theorem statement to prove the sum of their ages five years ago
theorem sum_of_ages_five_years_ago (h1 : angela_is_four_times_as_old_as_beth A B) (h2 : angela_will_be_44_in_five_years A) : 
  (A - 5) + (B - 5) = 39 :=
by sorry

end sum_of_ages_five_years_ago_l88_88633


namespace circle_numbers_exist_l88_88266

theorem circle_numbers_exist :
  ∃ (a b c d e f : ℚ),
    a = 2 ∧
    b = 3 ∧
    c = 3 / 2 ∧
    d = 1 / 2 ∧
    e = 1 / 3 ∧
    f = 2 / 3 ∧
    a = b * f ∧
    b = a * c ∧
    c = b * d ∧
    d = c * e ∧
    e = d * f ∧
    f = e * a := by
  sorry

end circle_numbers_exist_l88_88266


namespace original_cost_of_horse_l88_88139

theorem original_cost_of_horse (x : ℝ) (h : x - x^2 / 100 = 24) : x = 40 ∨ x = 60 := 
by 
  sorry

end original_cost_of_horse_l88_88139


namespace sum_of_acute_angles_l88_88865

theorem sum_of_acute_angles (α β γ : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : β > 0 ∧ β < π / 2) (h3: γ > 0 ∧ γ < π / 2) (h4 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) :
  (3 * π / 4) < α + β + γ ∧ α + β + γ < π :=
by
  sorry

end sum_of_acute_angles_l88_88865


namespace volume_of_soil_extracted_l88_88256

-- Definition of the conditions
def Length : ℝ := 20
def Width : ℝ := 10
def Depth : ℝ := 8

-- Statement of the proof problem
theorem volume_of_soil_extracted : Length * Width * Depth = 1600 := by
  -- Proof skipped
  sorry

end volume_of_soil_extracted_l88_88256


namespace total_charge_for_3_hours_l88_88643

namespace TherapyCharges

-- Conditions
variables (A F : ℝ)
variable (h1 : F = A + 20)
variable (h2 : F + 4 * A = 300)

-- Prove that the total charge for 3 hours of therapy is 188
theorem total_charge_for_3_hours : F + 2 * A = 188 :=
by
  sorry

end TherapyCharges

end total_charge_for_3_hours_l88_88643


namespace sum_of_powers_of_i_l88_88445

-- Define the imaginary unit and its property
def i : ℂ := Complex.I -- ℂ represents the complex numbers, Complex.I is the imaginary unit

-- The statement we need to prove
theorem sum_of_powers_of_i : i + i^2 + i^3 + i^4 = 0 := 
by {
  -- Lean requires the proof, but we will use sorry to skip it.
  -- Define the properties of i directly or use in-built properties
  sorry
}

end sum_of_powers_of_i_l88_88445


namespace subtract_digits_value_l88_88813

theorem subtract_digits_value (A B : ℕ) (h1 : A ≠ B) (h2 : 2 * 1000 + A * 100 + 3 * 10 + 2 - (B * 100 + B * 10 + B) = 1 * 1000 + B * 100 + B * 10 + B) :
  B - A = 3 :=
by
  sorry

end subtract_digits_value_l88_88813


namespace three_cards_different_suits_probability_l88_88684

-- Define the conditions and problem
noncomputable def prob_three_cards_diff_suits : ℚ :=
  have first_card_options := 52
  have second_card_options := 39
  have third_card_options := 26
  have total_ways_to_pick := (52 : ℕ) * (51 : ℕ) * (50 : ℕ)
  (39 / 51) * (26 / 50)

-- State our proof problem
theorem three_cards_different_suits_probability :
  prob_three_cards_diff_suits = 169 / 425 :=
sorry

end three_cards_different_suits_probability_l88_88684


namespace acute_angle_condition_l88_88577

theorem acute_angle_condition 
  (m : ℝ) 
  (a : ℝ × ℝ := (2,1))
  (b : ℝ × ℝ := (m,6)) 
  (dot_product := a.1 * b.1 + a.2 * b.2)
  (magnitude_a := Real.sqrt (a.1 * a.1 + a.2 * a.2))
  (magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2))
  (cos_angle := dot_product / (magnitude_a * magnitude_b))
  (acute_angle : cos_angle > 0) : -3 < m ∧ m ≠ 12 :=
sorry

end acute_angle_condition_l88_88577


namespace max_edges_in_8_points_graph_no_square_l88_88342

open Finset

-- Define what a graph is and the properties needed for the problem
structure Graph (V : Type*) :=
  (edges : Finset (V × V))
  (sym : ∀ {x y : V}, (x, y) ∈ edges ↔ (y, x) ∈ edges)
  (irrefl : ∀ {x : V}, ¬ (x, x) ∈ edges)

-- Define the conditions of the problem
def no_square {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c d : V), 
    (a, b) ∈ G.edges → (b, c) ∈ G.edges → (c, d) ∈ G.edges → (d, a) ∈ G.edges →
    (a, c) ∈ G.edges → (b, d) ∈ G.edges → False

-- Define 8 vertices
inductive Vertices
| A | B | C | D | E | F | G | H

-- Define the number of edges
noncomputable def max_edges_no_square : ℕ :=
  11

-- Define the final theorem
theorem max_edges_in_8_points_graph_no_square :
  ∃ (G : Graph Vertices), 
    no_square G ∧ (G.edges.card = max_edges_no_square) :=
sorry

end max_edges_in_8_points_graph_no_square_l88_88342


namespace correct_operation_l88_88532

noncomputable def valid_operation (n : ℕ) (a b : ℕ) (c d : ℤ) (x : ℚ) : Prop :=
  match n with
  | 0 => (x ^ a / x ^ b = x ^ (a - b))
  | 1 => (x ^ a * x ^ b = x ^ (a + b))
  | 2 => (c * x ^ a + d * x ^ a = (c + d) * x ^ a)
  | 3 => ((c * x ^ a) ^ b = c ^ b * x ^ (a * b))
  | _ => False

theorem correct_operation (x : ℚ) : valid_operation 1 2 3 0 0 x :=
by sorry

end correct_operation_l88_88532


namespace bert_puzzle_days_l88_88492

noncomputable def words_per_pencil : ℕ := 1050
noncomputable def words_per_puzzle : ℕ := 75

theorem bert_puzzle_days : words_per_pencil / words_per_puzzle = 14 := by
  sorry

end bert_puzzle_days_l88_88492


namespace smaller_cuboid_length_l88_88603

theorem smaller_cuboid_length
  (width_sm : ℝ)
  (height_sm : ℝ)
  (length_lg : ℝ)
  (width_lg : ℝ)
  (height_lg : ℝ)
  (num_sm : ℝ)
  (h1 : width_sm = 2)
  (h2 : height_sm = 3)
  (h3 : length_lg = 18)
  (h4 : width_lg = 15)
  (h5 : height_lg = 2)
  (h6 : num_sm = 18) :
  ∃ (length_sm : ℝ), (108 * length_sm = 540) ∧ (length_sm = 5) :=
by
  -- proof logic will be here
  sorry

end smaller_cuboid_length_l88_88603


namespace problem_statement_l88_88354

def f (x : ℝ) : ℝ := x^6 + x^2 + 7 * x

theorem problem_statement : f 3 - f (-3) = 42 := by
  sorry

end problem_statement_l88_88354


namespace rectangle_short_side_l88_88740

theorem rectangle_short_side
  (r : ℝ) (a_circle : ℝ) (a_rect : ℝ) (d : ℝ) (other_side : ℝ) :
  r = 6 →
  a_circle = Real.pi * r^2 →
  a_rect = 3 * a_circle →
  d = 2 * r →
  a_rect = d * other_side →
  other_side = 9 * Real.pi :=
by
  sorry

end rectangle_short_side_l88_88740


namespace inequality_solution_set_l88_88356

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := 
by 
  sorry

end inequality_solution_set_l88_88356


namespace total_oranges_is_correct_l88_88590

-- Definitions based on the problem's conditions
def layer_count : ℕ := 6
def base_length : ℕ := 9
def base_width : ℕ := 6

-- Function to compute the number of oranges in a layer given the current dimensions
def oranges_in_layer (length width : ℕ) : ℕ :=
  length * width

-- Function to compute the total number of oranges in the stack
def total_oranges_in_stack (base_length base_width : ℕ) : ℕ :=
  oranges_in_layer base_length base_width +
  oranges_in_layer (base_length - 1) (base_width - 1) +
  oranges_in_layer (base_length - 2) (base_width - 2) +
  oranges_in_layer (base_length - 3) (base_width - 3) +
  oranges_in_layer (base_length - 4) (base_width - 4) +
  oranges_in_layer (base_length - 5) (base_width - 5)

-- The theorem to be proved
theorem total_oranges_is_correct : total_oranges_in_stack 9 6 = 154 := by
  sorry

end total_oranges_is_correct_l88_88590


namespace student_count_l88_88806

theorem student_count (ratio : ℝ) (teachers : ℕ) (students : ℕ)
  (h1 : ratio = 27.5)
  (h2 : teachers = 42)
  (h3 : ratio * (teachers : ℝ) = students) :
  students = 1155 :=
sorry

end student_count_l88_88806


namespace least_number_subtracted_l88_88023

/-- The least number that must be subtracted from 50248 so that the 
remaining number is divisible by both 20 and 37 is 668. -/
theorem least_number_subtracted (n : ℕ) (x : ℕ ) (y : ℕ ) (a : ℕ) (b : ℕ) :
  n = 50248 → x = 20 → y = 37 → (a = 20 * 37) →
  (50248 - b) % a = 0 → 50248 - b < a → b = 668 :=
by
  sorry

end least_number_subtracted_l88_88023


namespace flour_baking_soda_ratio_l88_88474

theorem flour_baking_soda_ratio 
  (sugar flour baking_soda : ℕ)
  (h1 : sugar = 2000)
  (h2 : 5 * flour = 6 * sugar)
  (h3 : 8 * (baking_soda + 60) = flour) :
  flour / baking_soda = 10 := by
  sorry

end flour_baking_soda_ratio_l88_88474


namespace open_parking_spots_fourth_level_l88_88579

theorem open_parking_spots_fourth_level :
  ∀ (n_first n_total : ℕ)
    (n_second_diff n_third_diff : ℕ),
    n_first = 4 →
    n_second_diff = 7 →
    n_third_diff = 6 →
    n_total = 46 →
    ∃ (n_first n_second n_third n_fourth : ℕ),
      n_second = n_first + n_second_diff ∧
      n_third = n_second + n_third_diff ∧
      n_first + n_second + n_third + n_fourth = n_total ∧
      n_fourth = 14 := by
  sorry

end open_parking_spots_fourth_level_l88_88579


namespace cost_price_computer_table_l88_88874

variable (CP SP : ℝ)

theorem cost_price_computer_table (h1 : SP = 2 * CP) (h2 : SP = 1000) : CP = 500 := by
  sorry

end cost_price_computer_table_l88_88874


namespace base_difference_is_correct_l88_88475

-- Definitions of given conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 324 => 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Lean statement to prove the equivalence
theorem base_difference_is_correct : base9_to_base10 324 - base6_to_base10 231 = 174 :=
by
  sorry

end base_difference_is_correct_l88_88475


namespace max_popsicles_l88_88488

def popsicles : ℕ := 1
def box_3 : ℕ := 3
def box_5 : ℕ := 5
def box_10 : ℕ := 10
def cost_popsicle : ℕ := 1
def cost_box_3 : ℕ := 2
def cost_box_5 : ℕ := 3
def cost_box_10 : ℕ := 4
def budget : ℕ := 10

theorem max_popsicles : 
  ∀ (popsicle_count : ℕ) (b3_count : ℕ) (b5_count : ℕ) (b10_count : ℕ),
    popsicle_count * cost_popsicle + b3_count * cost_box_3 + b5_count * cost_box_5 + b10_count * cost_box_10 ≤ budget →
    popsicle_count * popsicles + b3_count * box_3 + b5_count * box_5 + b10_count * box_10 ≤ 23 →
    ∃ p b3 b5 b10, popsicle_count = p ∧ b3_count = b3 ∧ b5_count = b5 ∧ b10_count = b10 ∧
    (p * cost_popsicle + b3 * cost_box_3 + b5 * cost_box_5 + b10 * cost_box_10 ≤ budget) ∧
    (p * popsicles + b3 * box_3 + b5 * box_5 + b10 * box_10 = 23) :=
by sorry

end max_popsicles_l88_88488


namespace speed_of_jakes_dad_second_half_l88_88912

theorem speed_of_jakes_dad_second_half :
  let distance_to_park := 22
  let total_time := 0.5
  let time_half_journey := total_time / 2
  let speed_first_half := 28
  let distance_first_half := speed_first_half * time_half_journey
  let remaining_distance := distance_to_park - distance_first_half
  let time_second_half := time_half_journey
  let speed_second_half := remaining_distance / time_second_half
  speed_second_half = 60 :=
by
  sorry

end speed_of_jakes_dad_second_half_l88_88912


namespace joeys_votes_l88_88561

theorem joeys_votes
  (M B J : ℕ) 
  (h1 : M = 66) 
  (h2 : M = 3 * B) 
  (h3 : B = 2 * (J + 3)) : 
  J = 8 := 
by 
  sorry

end joeys_votes_l88_88561


namespace john_spent_30_l88_88114

/-- At a supermarket, John spent 1/5 of his money on fresh fruits and vegetables, 1/3 on meat products, and 1/10 on bakery products. If he spent the remaining $11 on candy, how much did John spend at the supermarket? -/
theorem john_spent_30 (X : ℝ) (h1 : X * (1/5) + X * (1/3) + X * (1/10) + 11 = X) : X = 30 := 
by 
  sorry

end john_spent_30_l88_88114


namespace max_total_balls_l88_88588

theorem max_total_balls
  (r₁ : ℕ := 89)
  (t₁ : ℕ := 90)
  (r₂ : ℕ := 8)
  (t₂ : ℕ := 9)
  (y : ℕ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (h₃ : 92 ≤ (r₁ + r₂ * y) * 100 / (t₁ + t₂ * y))
  : y ≤ 22 → 90 + 9 * y = 288 :=
by sorry

end max_total_balls_l88_88588


namespace sum_of_two_integers_l88_88402

noncomputable def sum_of_integers (a b : ℕ) : ℕ :=
a + b

theorem sum_of_two_integers (a b : ℕ) (h1 : a - b = 14) (h2 : a * b = 120) : sum_of_integers a b = 26 := 
by
  sorry

end sum_of_two_integers_l88_88402


namespace not_necessarily_divisible_by_20_l88_88452

theorem not_necessarily_divisible_by_20 (k : ℤ) (h : ∃ k : ℤ, 5 ∣ k * (k+1) * (k+2)) : ¬ ∀ k : ℤ, 20 ∣ k * (k+1) * (k+2) :=
by
  sorry

end not_necessarily_divisible_by_20_l88_88452


namespace tank_capacity_l88_88318

theorem tank_capacity :
  ∀ (T : ℚ), (3 / 4) * T + 4 = (7 / 8) * T → T = 32 :=
by
  intros T h
  sorry

end tank_capacity_l88_88318


namespace delta_value_l88_88946

theorem delta_value (Δ : ℝ) (h : 4 * 3 = Δ - 6) : Δ = 18 :=
sorry

end delta_value_l88_88946


namespace total_time_in_range_l88_88464

-- Definitions for the problem conditions
def section1 := 240 -- km
def section2 := 300 -- km
def section3 := 400 -- km

def speed1 := 40 -- km/h
def speed2 := 75 -- km/h
def speed3 := 80 -- km/h

-- The time it takes to cover a section at a certain speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Total time to cover all sections with different speed assignments
def total_time (s1 s2 s3 v1 v2 v3 : ℕ) : ℕ :=
  time s1 v1 + time s2 v2 + time s3 v3

-- Prove that the total time is within the range [15, 17]
theorem total_time_in_range :
  (total_time section1 section2 section3 speed3 speed2 speed1 = 15) ∧
  (total_time section1 section2 section3 speed1 speed2 speed3 = 17) →
  ∃ (T : ℕ), 15 ≤ T ∧ T ≤ 17 :=
by
  intro h
  sorry

end total_time_in_range_l88_88464


namespace correct_quadratic_equation_l88_88303

-- The main statement to prove.
theorem correct_quadratic_equation :
  (∀ (x y a : ℝ), (3 * x + 2 * y - 1 ≠ 0) ∧ (5 * x^2 - 6 * y - 3 ≠ 0) ∧ (a * x^2 - x + 2 ≠ 0) ∧ (x^2 - 1 = 0) → (x^2 - 1 = 0)) :=
by
  sorry

end correct_quadratic_equation_l88_88303


namespace sufficient_condition_l88_88267

theorem sufficient_condition (a b : ℝ) (h : a > b ∧ b > 0) : a + a^2 > b + b^2 :=
by
  sorry

end sufficient_condition_l88_88267


namespace solve_for_x_l88_88695

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (7 * x) ^ 5 = (14 * x) ^ 4 → x = 16 / 7 :=
by
  sorry

end solve_for_x_l88_88695


namespace max_smart_winners_min_total_prize_l88_88096

-- Define relevant constants and conditions
def total_winners := 25
def prize_smart : ℕ := 15
def prize_comprehensive : ℕ := 30

-- Problem 1: Maximum number of winners in "Smartest Brain" competition
theorem max_smart_winners (x : ℕ) (h1 : total_winners = 25)
  (h2 : total_winners - x ≥ 5 * x) : x ≤ 4 :=
sorry

-- Problem 2: Minimum total prize amount
theorem min_total_prize (y : ℕ) (h1 : y ≤ 4)
  (h2 : total_winners = 25)
  (h3 : (total_winners - y) ≥ 5 * y)
  (h4 : prize_smart = 15)
  (h5 : prize_comprehensive = 30) :
  15 * y + 30 * (25 - y) = 690 :=
sorry

end max_smart_winners_min_total_prize_l88_88096


namespace distance_is_20_sqrt_6_l88_88942

-- Definitions for problem setup
def distance_between_parallel_lines (r d : ℝ) : Prop :=
  ∃ O C D E F P Q : ℝ, 
  40^2 * 40 + (d / 2)^2 * 40 = 40 * r^2 ∧ 
  15^2 * 30 + (d / 2)^2 * 30 = 30 * r^2

-- The main statement to be proved
theorem distance_is_20_sqrt_6 :
  ∀ r d : ℝ,
  distance_between_parallel_lines r d →
  d = 20 * Real.sqrt 6 :=
sorry

end distance_is_20_sqrt_6_l88_88942


namespace ceil_of_fractional_square_l88_88602

theorem ceil_of_fractional_square :
  (Int.ceil ((- (7/4) + 1/4) ^ 2) = 3) :=
by
  sorry

end ceil_of_fractional_square_l88_88602


namespace number_of_valid_six_tuples_l88_88191

def is_valid_six_tuple (p : ℕ) (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 3 * p ∧
  (a + b) % (c + d) = 0 ∧
  (b + c) % (d + e) = 0 ∧
  (c + d) % (e + f) = 0 ∧
  (d + e) % (f + a) = 0 ∧
  (e + f) % (a + b) = 0

theorem number_of_valid_six_tuples (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : 
  ∃! n, n = p + 2 ∧ ∀ (a b c d e f : ℕ), is_valid_six_tuple p a b c d e f → n = p + 2 :=
sorry

end number_of_valid_six_tuples_l88_88191


namespace find_y_l88_88840

noncomputable def angle_ABC := 75
noncomputable def angle_BAC := 70
noncomputable def angle_CDE := 90
noncomputable def angle_BCA : ℝ := 180 - (angle_ABC + angle_BAC)
noncomputable def y : ℝ := 90 - angle_BCA

theorem find_y : y = 55 :=
by
  have h1: angle_BCA = 180 - (75 + 70) := rfl
  have h2: y = 90 - angle_BCA := rfl
  rw [h1] at h2
  exact h2.trans (by norm_num)

end find_y_l88_88840


namespace vertices_integer_assignment_zero_l88_88956

theorem vertices_integer_assignment_zero (f : ℕ → ℤ) (h100 : ∀ i, i < 100 → (i + 3) % 100 < 100) 
  (h : ∀ i, (i < 97 → f i + f (i + 2) = f (i + 1)) 
            ∨ (i < 97 → f (i + 1) + f (i + 3) = f (i + 2)) 
            ∨ (i < 97 → f i + f (i + 1) = f (i + 2))): 
  ∀ i, i < 100 → f i = 0 :=
by
  sorry

end vertices_integer_assignment_zero_l88_88956


namespace solution_set_of_inequality_l88_88983

theorem solution_set_of_inequality (x : ℝ) :  (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
by
  sorry

end solution_set_of_inequality_l88_88983


namespace f_f_neg1_l88_88154

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_f_neg1 : f (f (-1)) = 5 :=
  by
    sorry

end f_f_neg1_l88_88154


namespace sqrt_sum_gt_l88_88515

theorem sqrt_sum_gt (a b : ℝ) (ha : a = 2) (hb : b = 3) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by 
  sorry

end sqrt_sum_gt_l88_88515


namespace greater_number_l88_88600

theorem greater_number (a b : ℕ) (h1 : a + b = 36) (h2 : a - b = 8) : a = 22 :=
by
  sorry

end greater_number_l88_88600


namespace find_a_even_function_l88_88264

theorem find_a_even_function (a : ℝ) :
  (∀ x : ℝ, (x ^ 2 + a * x - 4) = ((-x) ^ 2 + a * (-x) - 4)) → a = 0 :=
by
  intro h
  sorry

end find_a_even_function_l88_88264


namespace geometric_sequence_n_value_l88_88226

theorem geometric_sequence_n_value
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 * a 2 * a 3 = 4)
  (h2 : a 4 * a 5 * a 6 = 12)
  (h3 : a (n-1) * a n * a (n+1) = 324)
  (h_geometric : ∃ r > 0, ∀ i, a (i+1) = a i * r) :
  n = 14 :=
sorry

end geometric_sequence_n_value_l88_88226


namespace fraction_spent_on_museum_ticket_l88_88901

theorem fraction_spent_on_museum_ticket (initial_money : ℝ) (sandwich_fraction : ℝ) (book_fraction : ℝ) (remaining_money : ℝ) (h1 : initial_money = 90) (h2 : sandwich_fraction = 1/5) (h3 : book_fraction = 1/2) (h4 : remaining_money = 12) : (initial_money - remaining_money) / initial_money - (sandwich_fraction * initial_money + book_fraction * initial_money) / initial_money = 1/6 :=
by
  sorry

end fraction_spent_on_museum_ticket_l88_88901


namespace max_isosceles_triangles_l88_88306

theorem max_isosceles_triangles 
  {A B C D P : ℝ} 
  (h_collinear: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D)
  (h_non_collinear: P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D)
  : (∀ a b c : ℝ, (a = P ∨ a = A ∨ a = B ∨ a = C ∨ a = D) ∧ (b = P ∨ b = A ∨ b = B ∨ b = C ∨ b = D) ∧ (c = P ∨ c = A ∨ c = B ∨ c = C ∨ c = D) 
    ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ((a - b)^2 + (b - c)^2 = (a - c)^2 ∨ (a - c)^2 + (b - c)^2 = (a - b)^2 ∨ (a - b)^2 + (a - c)^2 = (b - c)^2)) → 
    isosceles_triangle_count = 6 :=
sorry

end max_isosceles_triangles_l88_88306


namespace dan_spent_amount_l88_88243

-- Defining the prices of items
def candy_bar_price : ℝ := 7
def chocolate_price : ℝ := 6
def gum_price : ℝ := 3
def chips_price : ℝ := 4

-- Defining the discount and tax rates
def candy_bar_discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Defining the steps to calculate the total price including discount and tax
def total_before_discount_and_tax := candy_bar_price + chocolate_price + gum_price + chips_price
def candy_bar_discount := candy_bar_discount_rate * candy_bar_price
def candy_bar_after_discount := candy_bar_price - candy_bar_discount
def total_after_discount := candy_bar_after_discount + chocolate_price + gum_price + chips_price
def tax := tax_rate * total_after_discount
def total_with_discount_and_tax := total_after_discount + tax

theorem dan_spent_amount : total_with_discount_and_tax = 20.27 :=
by sorry

end dan_spent_amount_l88_88243


namespace popsicles_consumed_l88_88157

def total_minutes (hours : ℕ) (additional_minutes : ℕ) : ℕ :=
  hours * 60 + additional_minutes

def popsicles_in_time (total_time : ℕ) (interval : ℕ) : ℕ :=
  total_time / interval

theorem popsicles_consumed : popsicles_in_time (total_minutes 4 30) 15 = 18 :=
by
  -- The proof is omitted
  sorry

end popsicles_consumed_l88_88157


namespace possible_values_of_expression_l88_88596

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ v : ℝ, v = (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|) ∧ 
            (v = 5 ∨ v = 1 ∨ v = -3 ∨ v = -5)) :=
by
  sorry

end possible_values_of_expression_l88_88596


namespace bowling_ball_weight_l88_88908

variable {b c : ℝ}

theorem bowling_ball_weight :
  (10 * b = 4 * c) ∧ (3 * c = 108) → b = 14.4 :=
by
  sorry

end bowling_ball_weight_l88_88908


namespace original_price_doubled_l88_88944

variable (P : ℝ)

-- Given condition: Original price plus 20% equals 351
def price_increased (P : ℝ) : Prop :=
  P + 0.20 * P = 351

-- The goal is to prove that 2 times the original price is 585
theorem original_price_doubled (P : ℝ) (h : price_increased P) : 2 * P = 585 :=
sorry

end original_price_doubled_l88_88944


namespace Kenny_played_basketball_for_10_hours_l88_88883

theorem Kenny_played_basketball_for_10_hours
  (played_basketball ran practiced_trumpet : ℕ)
  (H1 : practiced_trumpet = 40)
  (H2 : ran = 2 * played_basketball)
  (H3 : practiced_trumpet = 2 * ran) :
  played_basketball = 10 :=
by
  sorry

end Kenny_played_basketball_for_10_hours_l88_88883


namespace prop_converse_inverse_contrapositive_correct_statements_l88_88797

-- Defining the proposition and its types
def prop (x : ℕ) : Prop := x > 0 → x^2 ≥ 0
def converse (x : ℕ) : Prop := x^2 ≥ 0 → x > 0
def inverse (x : ℕ) : Prop := ¬ (x > 0) → x^2 < 0
def contrapositive (x : ℕ) : Prop := x^2 < 0 → ¬ (x > 0)

-- The proof problem
theorem prop_converse_inverse_contrapositive_correct_statements :
  (∃! (p : Prop), p = (∀ x : ℕ, converse x) ∨ p = (∀ x : ℕ, inverse x) ∨ p = (∀ x : ℕ, contrapositive x) ∧ p = True) :=
sorry

end prop_converse_inverse_contrapositive_correct_statements_l88_88797


namespace sin_double_angle_l88_88770

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.sin (2 * θ) = 3 / 5 := 
by 
sorry

end sin_double_angle_l88_88770


namespace y_coord_of_equidistant_point_on_y_axis_l88_88405

/-!
  # Goal
  Prove that the $y$-coordinate of the point P on the $y$-axis that is equidistant from points $A(5, 0)$ and $B(3, 6)$ is \( \frac{5}{3} \).
  Conditions:
  - Point A has coordinates (5, 0).
  - Point B has coordinates (3, 6).
-/

theorem y_coord_of_equidistant_point_on_y_axis :
  ∃ y : ℝ, y = 5 / 3 ∧ (dist (⟨0, y⟩ : ℝ × ℝ) (⟨5, 0⟩ : ℝ × ℝ) = dist (⟨0, y⟩ : ℝ × ℝ) (⟨3, 6⟩ : ℝ × ℝ)) :=
by
  sorry -- Proof omitted

end y_coord_of_equidistant_point_on_y_axis_l88_88405


namespace correct_calculation_result_l88_88338

theorem correct_calculation_result (x : ℤ) (h : x + 63 = 8) : x * 36 = -1980 := by
  sorry

end correct_calculation_result_l88_88338


namespace determine_k_values_l88_88837

theorem determine_k_values (k : ℝ) :
  (∃ a b : ℝ, 3 * a ^ 2 + 6 * a + k = 0 ∧ 3 * b ^ 2 + 6 * b + k = 0 ∧ |a - b| = 1 / 2 * (a ^ 2 + b ^ 2)) → (k = 0 ∨ k = 12) :=
by
  sorry

end determine_k_values_l88_88837


namespace monotonic_increasing_m_ge_neg4_l88_88439

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ a → y > x → f y ≥ f x

def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 2

theorem monotonic_increasing_m_ge_neg4 (m : ℝ) :
  is_monotonic_increasing (f m) 2 → m ≥ -4 :=
by
  sorry

end monotonic_increasing_m_ge_neg4_l88_88439


namespace find_angle_A_l88_88869

theorem find_angle_A 
  (a b : ℝ) (A B : ℝ) 
  (h1 : b = 2 * a)
  (h2 : B = A + 60) : 
  A = 30 :=
  sorry

end find_angle_A_l88_88869


namespace find_sum_l88_88384

theorem find_sum (x y : ℝ) (h₁ : 3 * |x| + 2 * x + y = 20) (h₂ : 2 * x + 3 * |y| - y = 30) : x + y = 15 :=
sorry

end find_sum_l88_88384
