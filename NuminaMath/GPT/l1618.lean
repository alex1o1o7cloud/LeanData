import Mathlib

namespace NUMINAMATH_GPT_range_of_m_l1618_161840

-- Definitions for the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Prove that m ≥ 2 given the condition A ∪ B = A 
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1618_161840


namespace NUMINAMATH_GPT_fraction_of_bones_in_foot_is_approx_one_eighth_l1618_161874

def number_bones_human_body : ℕ := 206
def number_bones_one_foot : ℕ := 26
def fraction_bones_one_foot (total_bones foot_bones : ℕ) : ℚ := foot_bones / total_bones

theorem fraction_of_bones_in_foot_is_approx_one_eighth :
  fraction_bones_one_foot number_bones_human_body number_bones_one_foot = 13 / 103 ∧ 
  (abs ((13 / 103 : ℚ) - (1 / 8)) < 1 / 103) := 
sorry

end NUMINAMATH_GPT_fraction_of_bones_in_foot_is_approx_one_eighth_l1618_161874


namespace NUMINAMATH_GPT_surface_area_increase_l1618_161807

theorem surface_area_increase (r h : ℝ) (cs : Bool) : -- cs is a condition switch, True for circular cut, False for rectangular cut
  0 < r ∧ 0 < h →
  let inc_area := if cs then 2 * π * r^2 else 2 * h * r 
  inc_area > 0 :=
by 
  sorry

end NUMINAMATH_GPT_surface_area_increase_l1618_161807


namespace NUMINAMATH_GPT_y_mul_k_is_perfect_square_l1618_161858

-- Defining y as given in the problem with its prime factorization
def y : Nat := 3^4 * (2^2)^5 * 5^6 * (2 * 3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Since the question asks for an integer k (in this case 75) such that y * k is a perfect square
def k : Nat := 75

-- The statement that needs to be proved
theorem y_mul_k_is_perfect_square : ∃ n : Nat, (y * k) = n^2 := 
by
  sorry

end NUMINAMATH_GPT_y_mul_k_is_perfect_square_l1618_161858


namespace NUMINAMATH_GPT_ribbon_problem_l1618_161829

variable (Ribbon1 Ribbon2 : ℕ)
variable (L : ℕ)

theorem ribbon_problem
    (h1 : Ribbon1 = 8)
    (h2 : ∀ L, L > 0 → Ribbon1 % L = 0 → Ribbon2 % L = 0)
    (h3 : ∀ k, (k > 0 ∧ Ribbon1 % k = 0 ∧ Ribbon2 % k = 0) → k ≤ 8) :
    Ribbon2 = 8 := by
  sorry

end NUMINAMATH_GPT_ribbon_problem_l1618_161829


namespace NUMINAMATH_GPT_find_m_n_l1618_161808

theorem find_m_n (m n : ℤ) (h : |m - 2| + (n^2 - 8 * n + 16) = 0) : m = 2 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_l1618_161808


namespace NUMINAMATH_GPT_staircase_toothpicks_l1618_161803

theorem staircase_toothpicks (a : ℕ) (r : ℕ) (n : ℕ) :
  a = 9 ∧ r = 3 ∧ n = 3 + 4 
  → (a * r ^ 3 + a * r ^ 2 + a * r + a) + (a * r ^ 2 + a * r + a) + (a * r + a) + a = 351 :=
by
  sorry

end NUMINAMATH_GPT_staircase_toothpicks_l1618_161803


namespace NUMINAMATH_GPT_find_polynomials_l1618_161886

-- Definition of polynomials in Lean
noncomputable def polynomials : Type := Polynomial ℝ

-- Main theorem statement
theorem find_polynomials : 
  ∀ p : polynomials, 
    (∀ x : ℝ, p.eval (5 * x) ^ 2 - 3 = p.eval (5 * x^2 + 1)) → 
    (p.eval 0 ≠ 0 → (∃ c : ℝ, (p = Polynomial.C c) ∧ (c = (1 + Real.sqrt 13) / 2 ∨ c = (1 - Real.sqrt 13) / 2))) ∧ 
    (p.eval 0 = 0 → ∀ x : ℝ, p.eval x = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_polynomials_l1618_161886


namespace NUMINAMATH_GPT_lower_amount_rent_l1618_161889

theorem lower_amount_rent (L : ℚ) (total_rent : ℚ) (reduction : ℚ)
  (h1 : total_rent = 2000)
  (h2 : reduction = 200)
  (h3 : 10 * (60 - L) = reduction) :
  L = 40 := by
  sorry

end NUMINAMATH_GPT_lower_amount_rent_l1618_161889


namespace NUMINAMATH_GPT_height_of_sarah_building_l1618_161868

-- Define the conditions
def shadow_length_building : ℝ := 75
def height_pole : ℝ := 15
def shadow_length_pole : ℝ := 30

-- Define the height of the building
def height_building : ℝ := 38

-- Height of Sarah's building given the conditions
theorem height_of_sarah_building (h : ℝ) (H1 : shadow_length_building = 75)
    (H2 : height_pole = 15) (H3 : shadow_length_pole = 30) :
    h = height_building :=
by
  -- State the ratio of the height of the pole to its shadow
  have ratio_pole : ℝ := height_pole / shadow_length_pole

  -- Set up the ratio for Sarah's building and solve for h
  have h_eq : ℝ := ratio_pole * shadow_length_building

  -- Provide the proof (skipped here)
  sorry

end NUMINAMATH_GPT_height_of_sarah_building_l1618_161868


namespace NUMINAMATH_GPT_smallest_d_value_l1618_161853

theorem smallest_d_value : 
  ∃ d : ℝ, (d ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d + 5) = 4 * d) ∧ ∀ d' : ℝ, (d' ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d' + 5) = 4 * d') → (3 ≤ d') → d = 3 := 
by
  sorry

end NUMINAMATH_GPT_smallest_d_value_l1618_161853


namespace NUMINAMATH_GPT_candidates_count_l1618_161855

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 90) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_candidates_count_l1618_161855


namespace NUMINAMATH_GPT_solve_inequality_system_l1618_161879

theorem solve_inequality_system (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1618_161879


namespace NUMINAMATH_GPT_triangle_area_zero_vertex_l1618_161875

theorem triangle_area_zero_vertex (x1 y1 x2 y2 : ℝ) :
  (1 / 2) * |x1 * y2 - x2 * y1| = 
    abs (1 / 2 * (x1 * y2 - x2 * y1)) := 
sorry

end NUMINAMATH_GPT_triangle_area_zero_vertex_l1618_161875


namespace NUMINAMATH_GPT_num_races_necessary_l1618_161870

/-- There are 300 sprinters registered for a 200-meter dash at a local track meet,
where the track has only 8 lanes. In each race, 3 of the competitors advance to the
next round, while the rest are eliminated immediately. Determine how many races are
needed to identify the champion sprinter. -/
def num_races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (advance_per_race : ℕ) : ℕ :=
  if h : advance_per_race < lanes ∧ lanes > 0 then
    let eliminations_per_race := lanes - advance_per_race
    let total_eliminations := total_sprinters - 1
    Nat.ceil (total_eliminations / eliminations_per_race)
  else
    0

theorem num_races_necessary
  (total_sprinters : ℕ)
  (lanes : ℕ)
  (advance_per_race : ℕ)
  (h_total_sprinters : total_sprinters = 300)
  (h_lanes : lanes = 8)
  (h_advance_per_race : advance_per_race = 3) :
  num_races_to_champion total_sprinters lanes advance_per_race = 60 := by
  sorry

end NUMINAMATH_GPT_num_races_necessary_l1618_161870


namespace NUMINAMATH_GPT_part_a_part_b_l1618_161899

-- Define the tower of exponents function for convenience
def tower (base : ℕ) (height : ℕ) : ℕ :=
  if height = 0 then 1 else base^(tower base (height - 1))

-- Part a: Tower of 3s with height 99 is greater than Tower of 2s with height 100
theorem part_a : tower 3 99 > tower 2 100 := sorry

-- Part b: Tower of 3s with height 100 is greater than Tower of 3s with height 99
theorem part_b : tower 3 100 > tower 3 99 := sorry

end NUMINAMATH_GPT_part_a_part_b_l1618_161899


namespace NUMINAMATH_GPT_ground_beef_lean_beef_difference_l1618_161887

theorem ground_beef_lean_beef_difference (x y z : ℕ) 
  (h1 : x + y + z = 20) 
  (h2 : y + 2 * z = 18) :
  x - z = 2 :=
sorry

end NUMINAMATH_GPT_ground_beef_lean_beef_difference_l1618_161887


namespace NUMINAMATH_GPT_find_a1_l1618_161831

-- Definitions stemming from the conditions in the problem
def arithmetic_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

def is_geometric (a₁ a₃ a₆ : ℕ) : Prop :=
  ∃ r : ℕ, a₃ = r * a₁ ∧ a₆ = r^2 * a₁

theorem find_a1 :
  ∀ a₁ : ℕ,
    (arithmetic_seq a₁ 3 1 = a₁) ∧
    (arithmetic_seq a₁ 3 3 = a₁ + 6) ∧
    (arithmetic_seq a₁ 3 6 = a₁ + 15) ∧
    is_geometric a₁ (a₁ + 6) (a₁ + 15) →
    a₁ = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_a1_l1618_161831


namespace NUMINAMATH_GPT_smallest_k_divides_l1618_161891

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_divides_l1618_161891


namespace NUMINAMATH_GPT_negation_is_correct_l1618_161859

-- Define the original proposition as a predicate on real numbers.
def original_prop : Prop := ∀ x : ℝ, 4*x^2 - 3*x + 2 < 0

-- State the negation of the original proposition
def negation_of_original_prop : Prop := ∃ x : ℝ, 4*x^2 - 3*x + 2 ≥ 0

-- The theorem to prove the correctness of the negation of the original proposition
theorem negation_is_correct : ¬original_prop ↔ negation_of_original_prop := by
  sorry

end NUMINAMATH_GPT_negation_is_correct_l1618_161859


namespace NUMINAMATH_GPT_cost_of_ox_and_sheep_l1618_161821

variable (x y : ℚ)

theorem cost_of_ox_and_sheep :
  (5 * x + 2 * y = 10) ∧ (2 * x + 8 * y = 8) → (x = 16 / 9 ∧ y = 5 / 9) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_ox_and_sheep_l1618_161821


namespace NUMINAMATH_GPT_difference_of_squares_evaluation_l1618_161881

theorem difference_of_squares_evaluation :
  49^2 - 16^2 = 2145 :=
by sorry

end NUMINAMATH_GPT_difference_of_squares_evaluation_l1618_161881


namespace NUMINAMATH_GPT_handshakes_min_l1618_161812

-- Define the number of people and the number of handshakes each person performs
def numPeople : ℕ := 35
def handshakesPerPerson : ℕ := 3

-- Define the minimum possible number of unique handshakes
theorem handshakes_min : (numPeople * handshakesPerPerson) / 2 = 105 := by
  sorry

end NUMINAMATH_GPT_handshakes_min_l1618_161812


namespace NUMINAMATH_GPT_tyre_punctures_deflation_time_l1618_161815

theorem tyre_punctures_deflation_time :
  (1 / (1 / 9 + 1 / 6)) = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_tyre_punctures_deflation_time_l1618_161815


namespace NUMINAMATH_GPT_limit_one_minus_reciprocal_l1618_161882

theorem limit_one_minus_reciprocal (h : Filter.Tendsto (fun (n : ℕ) => 1 / n) Filter.atTop (nhds 0)) :
  Filter.Tendsto (fun (n : ℕ) => 1 - 1 / n) Filter.atTop (nhds 1) :=
sorry

end NUMINAMATH_GPT_limit_one_minus_reciprocal_l1618_161882


namespace NUMINAMATH_GPT_bonus_points_amount_l1618_161842

def points_per_10_dollars : ℕ := 50

def beef_price : ℕ := 11
def beef_quantity : ℕ := 3

def fruits_vegetables_price : ℕ := 4
def fruits_vegetables_quantity : ℕ := 8

def spices_price : ℕ := 6
def spices_quantity : ℕ := 3

def other_groceries_total : ℕ := 37

def total_points : ℕ := 850

def total_spent : ℕ :=
  (beef_price * beef_quantity) +
  (fruits_vegetables_price * fruits_vegetables_quantity) +
  (spices_price * spices_quantity) +
  other_groceries_total

def points_from_spending : ℕ :=
  (total_spent / 10) * points_per_10_dollars

theorem bonus_points_amount :
  total_spent > 100 → total_points - points_from_spending = 250 :=
by
  sorry

end NUMINAMATH_GPT_bonus_points_amount_l1618_161842


namespace NUMINAMATH_GPT_marbles_in_jar_is_144_l1618_161814

noncomputable def marbleCount (M : ℕ) : Prop :=
  M / 16 - M / 18 = 1

theorem marbles_in_jar_is_144 : ∃ M : ℕ, marbleCount M ∧ M = 144 :=
by
  use 144
  unfold marbleCount
  sorry

end NUMINAMATH_GPT_marbles_in_jar_is_144_l1618_161814


namespace NUMINAMATH_GPT_largest_subset_size_l1618_161841

theorem largest_subset_size (T : Finset ℕ) (h : ∀ x ∈ T, ∀ y ∈ T, x ≠ y → (x - y) % 2021 ≠ 5 ∧ (x - y) % 2021 ≠ 8) :
  T.card ≤ 918 := sorry

end NUMINAMATH_GPT_largest_subset_size_l1618_161841


namespace NUMINAMATH_GPT_triangle_area_x_value_l1618_161856

theorem triangle_area_x_value (x : ℝ) (h1 : x > 0) (h2 : 1 / 2 * x * (2 * x) = 64) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_x_value_l1618_161856


namespace NUMINAMATH_GPT_direct_proportion_k_l1618_161860

theorem direct_proportion_k (k x : ℝ) : ((k-1) * x + k^2 - 1 = 0) ∧ (k ≠ 1) ↔ k = -1 := 
sorry

end NUMINAMATH_GPT_direct_proportion_k_l1618_161860


namespace NUMINAMATH_GPT_total_volume_of_water_in_container_l1618_161854

def volume_each_hemisphere : ℝ := 4
def number_of_hemispheres : ℝ := 2735

theorem total_volume_of_water_in_container :
  (volume_each_hemisphere * number_of_hemispheres) = 10940 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_water_in_container_l1618_161854


namespace NUMINAMATH_GPT_opposite_direction_of_vectors_l1618_161832

theorem opposite_direction_of_vectors
  (x : ℝ)
  (a : ℝ × ℝ := (x, 1))
  (b : ℝ × ℝ := (4, x)) :
  (∃ k : ℝ, k ≠ 0 ∧ a = -k • b) → x = -2 := 
sorry

end NUMINAMATH_GPT_opposite_direction_of_vectors_l1618_161832


namespace NUMINAMATH_GPT_boy_speed_l1618_161872

theorem boy_speed (d : ℝ) (v₁ v₂ : ℝ) (t₁ t₂ l e : ℝ) :
  d = 2 ∧ v₂ = 8 ∧ l = 7 / 60 ∧ e = 8 / 60 ∧ t₁ = d / v₁ ∧ t₂ = d / v₂ ∧ t₁ - t₂ = l + e → v₁ = 4 :=
by
  sorry

end NUMINAMATH_GPT_boy_speed_l1618_161872


namespace NUMINAMATH_GPT_lisa_interest_l1618_161871

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem lisa_interest (hP : ℝ := 1500) (hr : ℝ := 0.02) (hn : ℕ := 10) :
  (compound_interest hP hr hn - hP) = 328.49 :=
by
  sorry

end NUMINAMATH_GPT_lisa_interest_l1618_161871


namespace NUMINAMATH_GPT_probability_of_square_product_is_17_over_96_l1618_161884

def num_tiles : Nat := 12
def num_die_faces : Nat := 8

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def favorable_outcomes_count : Nat :=
  -- Valid pairs where tile's number and die's number product is a perfect square
  List.length [ (1, 1), (1, 4), (2, 2), (4, 1),
                (1, 9), (3, 3), (9, 1), (4, 4),
                (2, 8), (8, 2), (5, 5), (6, 6),
                (4, 9), (9, 4), (7, 7), (8, 8),
                (9, 9) ] -- Equals 17 pairs

def total_outcomes_count : Nat :=
  num_tiles * num_die_faces

def probability_square_product : ℚ :=
  favorable_outcomes_count / total_outcomes_count

theorem probability_of_square_product_is_17_over_96 :
  probability_square_product = (17 : ℚ) / 96 := 
  by sorry

end NUMINAMATH_GPT_probability_of_square_product_is_17_over_96_l1618_161884


namespace NUMINAMATH_GPT_which_is_linear_l1618_161846

-- Define what it means to be a linear equation in two variables
def is_linear_equation_in_two_vars (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y = (a * x + b * y = c)

-- Define each of the given equations
def equation_A (x y : ℝ) : Prop := x / 2 + 3 * y = 2
def equation_B (x y : ℝ) : Prop := x / 2 + 1 = 3 * x * y
def equation_C (x y : ℝ) : Prop := 2 * x + 1 = 3 * x
def equation_D (x y : ℝ) : Prop := 3 * x + 2 * y^2 = 1

-- Theorem stating which equation is linear in two variables
theorem which_is_linear : 
  is_linear_equation_in_two_vars equation_A ∧ 
  ¬ is_linear_equation_in_two_vars equation_B ∧ 
  ¬ is_linear_equation_in_two_vars equation_C ∧ 
  ¬ is_linear_equation_in_two_vars equation_D := 
by 
  sorry

end NUMINAMATH_GPT_which_is_linear_l1618_161846


namespace NUMINAMATH_GPT_pirate_treasure_division_l1618_161873

theorem pirate_treasure_division (initial_treasure : ℕ) (p1_share p2_share p3_share p4_share p5_share remaining : ℕ)
  (h_initial : initial_treasure = 3000)
  (h_p1_share : p1_share = initial_treasure / 10)
  (h_p1_rem : remaining = initial_treasure - p1_share)
  (h_p2_share : p2_share = 2 * remaining / 10)
  (h_p2_rem : remaining = remaining - p2_share)
  (h_p3_share : p3_share = 3 * remaining / 10)
  (h_p3_rem : remaining = remaining - p3_share)
  (h_p4_share : p4_share = 4 * remaining / 10)
  (h_p4_rem : remaining = remaining - p4_share)
  (h_p5_share : p5_share = 5 * remaining / 10)
  (h_p5_rem : remaining = remaining - p5_share)
  (p6_p9_total : ℕ)
  (h_p6_p9_total : p6_p9_total = 20 * 4)
  (final_remaining : ℕ)
  (h_final_remaining : final_remaining = remaining - p6_p9_total) :
  final_remaining = 376 :=
by sorry

end NUMINAMATH_GPT_pirate_treasure_division_l1618_161873


namespace NUMINAMATH_GPT_unique_solution_for_system_l1618_161801

theorem unique_solution_for_system (a : ℝ) :
  (∃! (x y : ℝ), x^2 + 4 * y^2 = 1 ∧ x + 2 * y = a) ↔ a = -1.41 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_for_system_l1618_161801


namespace NUMINAMATH_GPT_sequence_x_y_sum_l1618_161824

theorem sequence_x_y_sum :
  ∃ (r x y : ℝ), 
    (r * 3125 = 625) ∧ 
    (r * 625 = 125) ∧ 
    (r * 125 = x) ∧ 
    (r * x = y) ∧ 
    (r * y = 1) ∧
    (r * 1 = 1/5) ∧ 
    (r * (1/5) = 1/25) ∧ 
    x + y = 30 := 
by
  -- A placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sequence_x_y_sum_l1618_161824


namespace NUMINAMATH_GPT_arrange_books_l1618_161865

-- Definition of the problem
def total_books : ℕ := 5 + 3

-- Definition of the combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Prove that arranging 5 copies of Introduction to Geometry and 
-- 3 copies of Introduction to Number Theory into total_books positions can be done in 56 ways.
theorem arrange_books : combination total_books 5 = 56 := by
  sorry

end NUMINAMATH_GPT_arrange_books_l1618_161865


namespace NUMINAMATH_GPT_mike_earnings_l1618_161828

def prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]

theorem mike_earnings :
  List.sum prices = 75 :=
by
  sorry

end NUMINAMATH_GPT_mike_earnings_l1618_161828


namespace NUMINAMATH_GPT_shorter_side_of_rectangle_l1618_161850

theorem shorter_side_of_rectangle (a b : ℕ) (h_perimeter : 2 * a + 2 * b = 62) (h_area : a * b = 240) : b = 15 :=
by
  sorry

end NUMINAMATH_GPT_shorter_side_of_rectangle_l1618_161850


namespace NUMINAMATH_GPT_proof_part1_proof_part2_l1618_161897

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2 + 3 * x

def condition1 (a : ℝ) : Prop := ∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x + 3 ≥ 0

def condition2 (a : ℝ) : Prop := 3 * 3^2 - 2 * a * 3 + 3 = 0

theorem proof_part1 (a : ℝ) : condition1 a → a ≤ 3 := 
sorry

theorem proof_part2 (a : ℝ) (ha : a = 5) : 
  f 1 a = -1 ∧ f 3 a = -9 ∧ f 5 a = 15 :=
sorry

end NUMINAMATH_GPT_proof_part1_proof_part2_l1618_161897


namespace NUMINAMATH_GPT_configuration_count_l1618_161804

theorem configuration_count :
  (∃ (w h s : ℕ), 2 * (w + h + 2 * s) = 120 ∧ w < h ∧ s % 2 = 0) →
  ∃ n, n = 196 := 
sorry

end NUMINAMATH_GPT_configuration_count_l1618_161804


namespace NUMINAMATH_GPT_least_number_conditioned_l1618_161869

theorem least_number_conditioned (n : ℕ) :
  n % 56 = 3 ∧ n % 78 = 3 ∧ n % 9 = 0 ↔ n = 2187 := 
sorry

end NUMINAMATH_GPT_least_number_conditioned_l1618_161869


namespace NUMINAMATH_GPT_geometric_sequences_l1618_161892

variable (a_n b_n : ℕ → ℕ) -- Geometric sequences
variable (S_n T_n : ℕ → ℕ) -- Sums of first n terms
variable (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4)

theorem geometric_sequences (n : ℕ) (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 := 
sorry

end NUMINAMATH_GPT_geometric_sequences_l1618_161892


namespace NUMINAMATH_GPT_three_sum_xyz_l1618_161893

theorem three_sum_xyz (x y z : ℝ) 
  (h1 : y + z = 18 - 4 * x) 
  (h2 : x + z = 22 - 4 * y) 
  (h3 : x + y = 15 - 4 * z) : 
  3 * x + 3 * y + 3 * z = 55 / 2 := 
  sorry

end NUMINAMATH_GPT_three_sum_xyz_l1618_161893


namespace NUMINAMATH_GPT_DE_minimal_length_in_triangle_l1618_161844

noncomputable def min_length_DE (BC AC : ℝ) (angle_B : ℝ) : ℝ :=
  if BC = 5 ∧ AC = 12 ∧ angle_B = 13 then 2 * Real.sqrt 3 else sorry

theorem DE_minimal_length_in_triangle :
  min_length_DE 5 12 13 = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_DE_minimal_length_in_triangle_l1618_161844


namespace NUMINAMATH_GPT_parallelogram_area_l1618_161851

/-- The area of a parallelogram is given by the product of its base and height. 
Given a parallelogram ABCD with base BC of 4 units and height of 2 units, 
prove its area is 8 square units. --/
theorem parallelogram_area (base height : ℝ) (h_base : base = 4) (h_height : height = 2) : 
  base * height = 8 :=
by
  rw [h_base, h_height]
  norm_num
  done

end NUMINAMATH_GPT_parallelogram_area_l1618_161851


namespace NUMINAMATH_GPT_find_ellipse_and_hyperbola_equations_l1618_161805

-- Define the conditions
def eccentricity (e : ℝ) (a b : ℝ) : Prop :=
  e = (Real.sqrt (a ^ 2 - b ^ 2)) / a

def focal_distance (f : ℝ) (a b : ℝ) : Prop :=
  f = 2 * Real.sqrt (a ^ 2 + b ^ 2)

-- Define the problem to prove the equations of the ellipse and hyperbola
theorem find_ellipse_and_hyperbola_equations (a b : ℝ) (e : ℝ) (f : ℝ)
  (h1 : eccentricity e a b) (h2 : focal_distance f a b) 
  (h3 : e = 4 / 5) (h4 : f = 2 * Real.sqrt 34) 
  (h5 : a > b) (h6 : 0 < b) :
  (a^2 = 25 ∧ b^2 = 9) → 
  (∀ x y, (x^2 / 25 + y^2 / 9 = 1) ∧ (x^2 / 25 - y^2 / 9 = 1)) :=
sorry

end NUMINAMATH_GPT_find_ellipse_and_hyperbola_equations_l1618_161805


namespace NUMINAMATH_GPT_bob_monthly_hours_l1618_161896

noncomputable def total_hours_in_month : ℝ :=
  let daily_hours := 10
  let weekly_days := 5
  let weeks_in_month := 4.33
  daily_hours * weekly_days * weeks_in_month

theorem bob_monthly_hours :
  total_hours_in_month = 216.5 :=
by
  sorry

end NUMINAMATH_GPT_bob_monthly_hours_l1618_161896


namespace NUMINAMATH_GPT_renata_final_money_l1618_161864

-- Defining the initial condition and the sequence of financial transactions.
def initial_money := 10
def donation := 4
def prize := 90
def slot_loss1 := 50
def slot_loss2 := 10
def slot_loss3 := 5
def water_cost := 1
def lottery_ticket_cost := 1
def lottery_prize := 65

-- Prove that given all these transactions, the final amount of money is $94.
theorem renata_final_money :
  initial_money 
  - donation 
  + prize 
  - slot_loss1 
  - slot_loss2 
  - slot_loss3 
  - water_cost 
  - lottery_ticket_cost 
  + lottery_prize 
  = 94 := 
by
  sorry

end NUMINAMATH_GPT_renata_final_money_l1618_161864


namespace NUMINAMATH_GPT_equation_of_latus_rectum_l1618_161890

theorem equation_of_latus_rectum (p : ℝ) (h1 : p = 6) :
  (∀ x y : ℝ, y ^ 2 = -12 * x → x = 3) :=
sorry

end NUMINAMATH_GPT_equation_of_latus_rectum_l1618_161890


namespace NUMINAMATH_GPT_calculate_income_l1618_161833

theorem calculate_income (I : ℝ) (T : ℝ) (a b c d : ℝ) (h1 : a = 0.15) (h2 : b = 40000) (h3 : c = 0.20) (h4 : T = 8000) (h5 : T = a * b + c * (I - b)) : I = 50000 :=
by
  sorry

end NUMINAMATH_GPT_calculate_income_l1618_161833


namespace NUMINAMATH_GPT_circle_packing_line_equation_l1618_161847

theorem circle_packing_line_equation
  (d : ℝ) (n1 n2 n3 : ℕ) (slope : ℝ)
  (l_intersects_tangencies : ℝ → ℝ → Prop)
  (l_divides_R : Prop)
  (gcd_condition : ℕ → ℕ → ℕ → ℕ)
  (a b c : ℕ)
  (a_pos : 0 < a) (b_neg : b < 0) (c_pos : 0 < c)
  (gcd_abc : gcd_condition a b c = 1)
  (correct_equation_format : Prop) :
  n1 = 4 ∧ n2 = 4 ∧ n3 = 2 →
  d = 2 →
  slope = 5 →
  l_divides_R →
  l_intersects_tangencies 1 1 →
  l_intersects_tangencies 4 6 → 
  correct_equation_format → 
  a^2 + b^2 + c^2 = 42 :=
by sorry

end NUMINAMATH_GPT_circle_packing_line_equation_l1618_161847


namespace NUMINAMATH_GPT_coefficient_of_x_eq_2_l1618_161876

variable (a : ℝ)

theorem coefficient_of_x_eq_2 (h : (5 * (-2)) + (4 * a) = 2) : a = 3 :=
sorry

end NUMINAMATH_GPT_coefficient_of_x_eq_2_l1618_161876


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1618_161861

def M := {n : ℕ | 0 < n ∧ n < 1000}

def circ (a b : ℕ) : ℕ :=
  if a * b < 1000 then a * b
  else 
    let k := (a * b) / 1000
    let r := (a * b) % 1000
    if k + r < 1000 then k + r
    else (k + r) % 1000 + 1

theorem problem_1 : circ 559 758 = 146 := 
by
  sorry

theorem problem_2 : ∃ (x : ℕ) (h : x ∈ M), circ 559 x = 1 ∧ x = 361 :=
by
  sorry

theorem problem_3 : ∀ (a b c : ℕ) (h₁ : a ∈ M) (h₂ : b ∈ M) (h₃ : c ∈ M), circ a (circ b c) = circ (circ a b) c :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1618_161861


namespace NUMINAMATH_GPT_difference_of_roots_l1618_161830

theorem difference_of_roots 
  (a b c : ℝ)
  (h : ∀ x, x^2 - 2 * (a^2 + b^2 + c^2 - 2 * a * c) * x + (b^2 - a^2 - c^2 + 2 * a * c)^2 = 0) :
  ∃ (x1 x2 : ℝ), (x1 - x2 = 4 * b * (a - c)) ∨ (x1 - x2 = -4 * b * (a - c)) :=
sorry

end NUMINAMATH_GPT_difference_of_roots_l1618_161830


namespace NUMINAMATH_GPT_min_f_of_shangmei_number_l1618_161843

def is_shangmei_number (a b c d : ℕ) : Prop :=
  a + c = 11 ∧ b + d = 11

def f (a b : ℕ) : ℚ :=
  (b - (11 - b) : ℚ) / (a - (11 - a))

def G (a b : ℕ) : ℤ :=
  20 * a + 2 * b - 121

def is_multiple_of_7 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 7 * k

theorem min_f_of_shangmei_number :
  ∃ (a b c d : ℕ), a < b ∧ is_shangmei_number a b c d ∧ is_multiple_of_7 (G a b) ∧ f a b = -3 :=
sorry

end NUMINAMATH_GPT_min_f_of_shangmei_number_l1618_161843


namespace NUMINAMATH_GPT_total_number_of_coins_l1618_161878

-- Define conditions
def pennies : Nat := 38
def nickels : Nat := 27
def dimes : Nat := 19
def quarters : Nat := 24
def half_dollars : Nat := 13
def one_dollar_coins : Nat := 17
def two_dollar_coins : Nat := 5
def australian_fifty_cent_coins : Nat := 4
def mexican_one_peso_coins : Nat := 12

-- Define the problem as a theorem
theorem total_number_of_coins : 
  pennies + nickels + dimes + quarters + half_dollars + one_dollar_coins + two_dollar_coins + australian_fifty_cent_coins + mexican_one_peso_coins = 159 := by
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l1618_161878


namespace NUMINAMATH_GPT_exists_unique_c_l1618_161898

theorem exists_unique_c (a : ℝ) (h₁ : 1 < a) :
  (∃ (c : ℝ), ∀ (x : ℝ), x ∈ Set.Icc a (2 * a) → ∃ (y : ℝ), y ∈ Set.Icc a (a ^ 2) ∧ (Real.log x / Real.log a + Real.log y / Real.log a = c)) ↔ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_unique_c_l1618_161898


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1618_161819

theorem eccentricity_of_ellipse :
  ∀ (A B : ℝ × ℝ) (has_axes_intersection : A.2 = 0 ∧ B.2 = 0) 
    (product_of_slopes : ∀ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B → (P.2 / (P.1 - A.1)) * (P.2 / (P.1 + B.1)) = -1/2),
  ∃ (e : ℝ), e = 1 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1618_161819


namespace NUMINAMATH_GPT_find_cookies_per_tray_l1618_161827

def trays_baked_per_day := 2
def days_of_baking := 6
def cookies_eaten_by_frank := 1
def cookies_eaten_by_ted := 4
def cookies_left := 134

theorem find_cookies_per_tray (x : ℕ) (h : 12 * x - 10 = 134) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_cookies_per_tray_l1618_161827


namespace NUMINAMATH_GPT_add_pure_alcohol_to_achieve_percentage_l1618_161895

-- Define the initial conditions
def initial_solution_volume : ℝ := 6
def initial_alcohol_percentage : ℝ := 0.30
def initial_pure_alcohol : ℝ := initial_solution_volume * initial_alcohol_percentage

-- Define the final conditions
def final_alcohol_percentage : ℝ := 0.50

-- Define the unknown to prove
def amount_of_alcohol_to_add : ℝ := 2.4

-- The target statement to prove
theorem add_pure_alcohol_to_achieve_percentage :
  (initial_pure_alcohol + amount_of_alcohol_to_add) / (initial_solution_volume + amount_of_alcohol_to_add) = final_alcohol_percentage :=
by
  sorry

end NUMINAMATH_GPT_add_pure_alcohol_to_achieve_percentage_l1618_161895


namespace NUMINAMATH_GPT_profit_percent_l1618_161852

variable (C S : ℝ)
variable (h : (1 / 3) * S = 0.8 * C)

theorem profit_percent (h : (1 / 3) * S = 0.8 * C) : 
  ((S - C) / C) * 100 = 140 := 
by
  sorry

end NUMINAMATH_GPT_profit_percent_l1618_161852


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1618_161885

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 1}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def intersection : Set ℝ := {z | -1 ≤ z ∧ z ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {z | -1 ≤ z ∧ z ≤ 3} := 
sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1618_161885


namespace NUMINAMATH_GPT_uma_income_l1618_161836

theorem uma_income
  (x y : ℝ)
  (h1 : 8 * x - 7 * y = 2000)
  (h2 : 7 * x - 6 * y = 2000) :
  8 * x = 16000 := by
  sorry

end NUMINAMATH_GPT_uma_income_l1618_161836


namespace NUMINAMATH_GPT_recipe_sugar_amount_l1618_161883

-- Definitions from A)
def cups_of_salt : ℕ := 9
def additional_cups_of_sugar (sugar salt : ℕ) : Prop := sugar = salt + 2

-- Statement to prove
theorem recipe_sugar_amount (salt : ℕ) (h : salt = cups_of_salt) : ∃ sugar : ℕ, additional_cups_of_sugar sugar salt ∧ sugar = 11 :=
by
  sorry

end NUMINAMATH_GPT_recipe_sugar_amount_l1618_161883


namespace NUMINAMATH_GPT_sum_sequence_l1618_161811

theorem sum_sequence (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = -2/3)
  (h2 : ∀ n, n ≥ 2 → S n = -1 / (S (n - 1) + 2)) :
  ∀ n, S n = -(n + 1) / (n + 2) := 
by 
  sorry

end NUMINAMATH_GPT_sum_sequence_l1618_161811


namespace NUMINAMATH_GPT_neg_P_l1618_161877

def P := ∃ x : ℝ, (0 < x) ∧ (3^x < x^3)

theorem neg_P : ¬P ↔ ∀ x : ℝ, (0 < x) → (3^x ≥ x^3) :=
by
  sorry

end NUMINAMATH_GPT_neg_P_l1618_161877


namespace NUMINAMATH_GPT_sum_of_digits_133131_l1618_161894

noncomputable def extract_digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldl (· + ·) 0

theorem sum_of_digits_133131 :
  let ABCDEF := 665655 / 5
  extract_digits_sum ABCDEF = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_133131_l1618_161894


namespace NUMINAMATH_GPT_solve_inequality_l1618_161867

theorem solve_inequality (x : ℝ) : (x + 1) / (x - 2) + (x - 3) / (3 * x) ≥ 4 ↔ x ∈ Set.Ico (-1/4) 0 ∪ Set.Ioc 2 3 := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1618_161867


namespace NUMINAMATH_GPT_area_proof_l1618_161880

def square_side_length : ℕ := 2
def triangle_leg_length : ℕ := 2

-- Definition of the initial square area
def square_area (side_length : ℕ) : ℕ := side_length * side_length

-- Definition of the area for one isosceles right triangle
def triangle_area (leg_length : ℕ) : ℕ := (leg_length * leg_length) / 2

-- Area of the initial square
def R_square_area : ℕ := square_area square_side_length

-- Area of the 12 isosceles right triangles
def total_triangle_area : ℕ := 12 * triangle_area triangle_leg_length

-- Total area of region R
def R_area : ℕ := R_square_area + total_triangle_area

-- Smallest convex polygon S is a larger square with side length 8
def S_area : ℕ := square_area (4 * square_side_length)

-- Area inside S but outside R
def area_inside_S_outside_R : ℕ := S_area - R_area

theorem area_proof : area_inside_S_outside_R = 36 :=
by
  sorry

end NUMINAMATH_GPT_area_proof_l1618_161880


namespace NUMINAMATH_GPT_evaluate_expression_l1618_161816

theorem evaluate_expression : 
  (Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3)) = Real.sqrt 3 + 3 + 5 / 6 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1618_161816


namespace NUMINAMATH_GPT_coins_count_l1618_161862

variable (x : ℕ)

def total_value : ℕ → ℕ := λ x => x + (x * 50) / 100 + (x * 25) / 100

theorem coins_count (h : total_value x = 140) : x = 80 :=
sorry

end NUMINAMATH_GPT_coins_count_l1618_161862


namespace NUMINAMATH_GPT_intersection_eq_union_eq_l1618_161849

noncomputable def A := {x : ℝ | -2 < x ∧ x <= 3}
noncomputable def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_eq : A ∩ B = {x : ℝ | -2 < x ∧ x < -1} := by
  sorry

theorem union_eq : A ∪ B = {x : ℝ | x <= 3 ∨ x > 4} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_union_eq_l1618_161849


namespace NUMINAMATH_GPT_greatest_s_property_l1618_161817

noncomputable def find_greatest_s (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] : ℕ :=
if h : m > 0 ∧ n > 0 then m else 0

theorem greatest_s_property (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] (H : 0 < m) (H1 : 0 < n)  :
  ∃ s, (s = find_greatest_s m n p) ∧ s * n * p ≤ m * n * p :=
by 
  sorry

end NUMINAMATH_GPT_greatest_s_property_l1618_161817


namespace NUMINAMATH_GPT_probability_of_diamond_or_ace_at_least_one_l1618_161837

noncomputable def prob_at_least_one_diamond_or_ace : ℚ := 
  1 - (9 / 13) ^ 2

theorem probability_of_diamond_or_ace_at_least_one :
  prob_at_least_one_diamond_or_ace = 88 / 169 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_diamond_or_ace_at_least_one_l1618_161837


namespace NUMINAMATH_GPT_gcd_three_numbers_l1618_161839

def a : ℕ := 13680
def b : ℕ := 20400
def c : ℕ := 47600

theorem gcd_three_numbers (a b c : ℕ) : Nat.gcd (Nat.gcd a b) c = 80 :=
by
  sorry

end NUMINAMATH_GPT_gcd_three_numbers_l1618_161839


namespace NUMINAMATH_GPT_range_f_l1618_161888

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.sqrt (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) + 
  Real.sqrt (a * Real.sin x ^ 2 + b * Real.cos x ^ 2)

theorem range_f (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Set.range (f a b) = Set.Icc (Real.sqrt a + Real.sqrt b) (Real.sqrt (2 * (a + b))) :=
sorry

end NUMINAMATH_GPT_range_f_l1618_161888


namespace NUMINAMATH_GPT_simple_interest_rate_l1618_161866

-- Define the entities and conditions
variables (P A T : ℝ) (R : ℝ)

-- Conditions given in the problem
def principal := P = 12500
def amount := A = 16750
def time := T = 8

-- Result that needs to be proved
def correct_rate := R = 4.25

-- Main statement to be proven: Given the conditions, the rate is 4.25%
theorem simple_interest_rate :
  principal P → amount A → time T → (A - P = (P * R * T) / 100) → correct_rate R :=
by
  intros hP hA hT hSI
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1618_161866


namespace NUMINAMATH_GPT_eleven_hash_five_l1618_161826

def my_op (r s : ℝ) : ℝ := sorry

axiom op_cond1 : ∀ r : ℝ, my_op r 0 = r
axiom op_cond2 : ∀ r s : ℝ, my_op r s = my_op s r
axiom op_cond3 : ∀ r s : ℝ, my_op (r + 1) s = (my_op r s) + s + 1

theorem eleven_hash_five : my_op 11 5 = 71 :=
by {
    sorry
}

end NUMINAMATH_GPT_eleven_hash_five_l1618_161826


namespace NUMINAMATH_GPT_least_sub_to_make_div_by_10_l1618_161838

theorem least_sub_to_make_div_by_10 : 
  ∃ n, n = 8 ∧ ∀ k, 427398 - k = 10 * m → k ≥ n ∧ k = 8 :=
sorry

end NUMINAMATH_GPT_least_sub_to_make_div_by_10_l1618_161838


namespace NUMINAMATH_GPT_least_positive_integer_solution_l1618_161818

theorem least_positive_integer_solution :
  ∃ x : ℤ, x > 0 ∧ ∃ n : ℤ, (3 * x + 29)^2 = 43 * n ∧ x = 19 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_solution_l1618_161818


namespace NUMINAMATH_GPT_indoor_tables_count_l1618_161863

theorem indoor_tables_count
  (I : ℕ)  -- the number of indoor tables
  (O : ℕ)  -- the number of outdoor tables
  (H1 : O = 12)  -- Condition 1: O = 12
  (H2 : 3 * I + 3 * O = 60)  -- Condition 2: Total number of chairs
  : I = 8 :=
by
  -- Insert the actual proof here
  sorry

end NUMINAMATH_GPT_indoor_tables_count_l1618_161863


namespace NUMINAMATH_GPT_inequality_5positives_l1618_161825

variable {x1 x2 x3 x4 x5 : ℝ}

theorem inequality_5positives (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_5positives_l1618_161825


namespace NUMINAMATH_GPT_seat_39_l1618_161802

-- Defining the main structure of the problem
def circle_seating_arrangement (n k : ℕ) : ℕ :=
  if k = 1 then 1
  else sorry -- The pattern-based implementation goes here

-- The theorem to state the problem
theorem seat_39 (n k : ℕ) (h_n : n = 128) (h_k : k = 39) :
  circle_seating_arrangement n k = 51 :=
sorry

end NUMINAMATH_GPT_seat_39_l1618_161802


namespace NUMINAMATH_GPT_intersection_coordinates_l1618_161835

theorem intersection_coordinates (x y : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : y = x + 1) : 
  x = 2 ∧ y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_intersection_coordinates_l1618_161835


namespace NUMINAMATH_GPT_find_f_1988_l1618_161823

def f : ℕ+ → ℕ+ := sorry

axiom functional_equation (m n : ℕ+) : f (f m + f n) = m + n

theorem find_f_1988 : f 1988 = 1988 :=
by sorry

end NUMINAMATH_GPT_find_f_1988_l1618_161823


namespace NUMINAMATH_GPT_probability_circle_containment_l1618_161834

theorem probability_circle_containment :
  let a_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
  let circle_C_contained (a : ℕ) : Prop := a > 3
  let m : ℕ := (a_set.filter circle_C_contained).card
  let n : ℕ := a_set.card
  let p : ℚ := m / n
  p = 4 / 7 := 
by
  sorry

end NUMINAMATH_GPT_probability_circle_containment_l1618_161834


namespace NUMINAMATH_GPT_new_ratio_l1618_161848

def milk_to_water_initial_ratio (M W : ℕ) : Prop := 4 * W = M

def total_volume (V M W : ℕ) : Prop := V = M + W

def new_water_volume (W_new W A : ℕ) : Prop := W_new = W + A

theorem new_ratio (V M W W_new A : ℕ) 
  (h1: milk_to_water_initial_ratio M W) 
  (h2: total_volume V M W) 
  (h3: A = 23) 
  (h4: new_water_volume W_new W A) 
  (h5: V = 45) 
  : 9 * W_new = 8 * M :=
by 
  sorry

end NUMINAMATH_GPT_new_ratio_l1618_161848


namespace NUMINAMATH_GPT_rational_solutions_k_l1618_161813

theorem rational_solutions_k (k : ℕ) (hpos : k > 0) : (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 :=
by
  sorry

end NUMINAMATH_GPT_rational_solutions_k_l1618_161813


namespace NUMINAMATH_GPT_triangle_constructibility_l1618_161820

variables (a b c γ : ℝ)

-- definition of the problem conditions
def valid_triangle_constructibility_conditions (a b_c_diff γ : ℝ) : Prop :=
  γ < 90 ∧ b_c_diff < a * Real.cos γ

-- constructibility condition
def is_constructible (a b c γ : ℝ) : Prop :=
  b - c < a * Real.cos γ

-- final theorem statement
theorem triangle_constructibility (a b c γ : ℝ) (h1 : γ < 90) (h2 : b > c) :
  (b - c < a * Real.cos γ) ↔ valid_triangle_constructibility_conditions a (b - c) γ :=
by sorry

end NUMINAMATH_GPT_triangle_constructibility_l1618_161820


namespace NUMINAMATH_GPT_benny_added_march_l1618_161857

theorem benny_added_march :
  let january := 19 
  let february := 19
  let march_total := 46
  (march_total - (january + february) = 8) :=
by
  let january := 19
  let february := 19
  let march_total := 46
  sorry

end NUMINAMATH_GPT_benny_added_march_l1618_161857


namespace NUMINAMATH_GPT_quadratic_trinomial_m_eq_2_l1618_161810

theorem quadratic_trinomial_m_eq_2 (m : ℤ) (P : |m| = 2 ∧ m + 2 ≠ 0) : m = 2 :=
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_m_eq_2_l1618_161810


namespace NUMINAMATH_GPT_compound_interest_rate_l1618_161806

theorem compound_interest_rate (P A : ℝ) (t n : ℝ)
  (hP : P = 5000) 
  (hA : A = 7850)
  (ht : t = 8)
  (hn : n = 1) : 
  ∃ r : ℝ, 0.057373 ≤ (r * 100) ∧ (r * 100) ≤ 5.7373 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1618_161806


namespace NUMINAMATH_GPT_find_divisor_l1618_161809

theorem find_divisor (dividend remainder quotient : ℕ) (h1 : dividend = 76) (h2 : remainder = 8) (h3 : quotient = 4) : ∃ d : ℕ, dividend = (d * quotient) + remainder ∧ d = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1618_161809


namespace NUMINAMATH_GPT_shaded_l_shaped_area_l1618_161822

def square (side : ℕ) : ℕ := side * side
def rectangle (length width : ℕ) : ℕ := length * width

theorem shaded_l_shaped_area :
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  areaABCD - total_area_small_shapes = 20 :=
by
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  have h : areaABCD - total_area_small_shapes = 20 := sorry
  exact h

end NUMINAMATH_GPT_shaded_l_shaped_area_l1618_161822


namespace NUMINAMATH_GPT_tip_per_person_l1618_161800

-- Define the necessary conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def total_amount_made : ℝ := 37

-- Define the problem statement
theorem tip_per_person : (total_amount_made - hourly_wage) / people_served = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_tip_per_person_l1618_161800


namespace NUMINAMATH_GPT_new_students_joined_l1618_161845

theorem new_students_joined (orig_avg_age new_avg_age : ℕ) (decrease_in_avg_age : ℕ) (orig_strength : ℕ) (new_students_avg_age : ℕ) :
  orig_avg_age = 40 ∧ new_avg_age = 36 ∧ decrease_in_avg_age = 4 ∧ orig_strength = 18 ∧ new_students_avg_age = 32 →
  ∃ x : ℕ, ((orig_strength * orig_avg_age) + (x * new_students_avg_age) = new_avg_age * (orig_strength + x)) ∧ x = 18 :=
by
  sorry

end NUMINAMATH_GPT_new_students_joined_l1618_161845
