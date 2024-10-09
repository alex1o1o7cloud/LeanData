import Mathlib

namespace number_of_people_on_boats_l1895_189540

def boats := 5
def people_per_boat := 3

theorem number_of_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end number_of_people_on_boats_l1895_189540


namespace max_bricks_truck_can_carry_l1895_189556

-- Define the truck's capacity in terms of bags of sand and bricks
def max_sand_bags := 50
def max_bricks := 400
def sand_to_bricks_ratio := 8

-- Define the current number of sand bags already on the truck
def current_sand_bags := 32

-- Define the number of bricks equivalent to a given number of sand bags
def equivalent_bricks (sand_bags: ℕ) := sand_bags * sand_to_bricks_ratio

-- Define the remaining capacity in terms of bags of sand
def remaining_sand_bags := max_sand_bags - current_sand_bags

-- Define the maximum number of additional bricks the truck can carry
def max_additional_bricks := equivalent_bricks remaining_sand_bags

-- Prove the number of additional bricks the truck can carry is 144
theorem max_bricks_truck_can_carry : max_additional_bricks = 144 := by
  sorry

end max_bricks_truck_can_carry_l1895_189556


namespace expected_value_of_winnings_l1895_189501

noncomputable def expected_value : ℝ :=
  (1 / 8) * (1 / 2) + (1 / 8) * (3 / 2) + (1 / 8) * (5 / 2) + (1 / 8) * (7 / 2) +
  (1 / 8) * 2 + (1 / 8) * 4 + (1 / 8) * 6 + (1 / 8) * 8

theorem expected_value_of_winnings : expected_value = 3.5 :=
by
  -- the proof steps will go here
  sorry

end expected_value_of_winnings_l1895_189501


namespace additional_lollipops_needed_l1895_189595

theorem additional_lollipops_needed
  (kids : ℕ) (initial_lollipops : ℕ) (min_lollipops : ℕ) (max_lollipops : ℕ)
  (total_kid_with_lollipops : ∀ k, ∃ n, min_lollipops ≤ n ∧ n ≤ max_lollipops ∧ k = n ∨ k = n + 1 )
  (divisible_by_kids : (min_lollipops + max_lollipops) % kids = 0)
  (min_lollipops_eq : min_lollipops = 42)
  (kids_eq : kids = 42)
  (initial_lollipops_eq : initial_lollipops = 650)
  : ∃ additional_lollipops, (n : ℕ) = 42 → additional_lollipops = 1975 := 
by sorry

end additional_lollipops_needed_l1895_189595


namespace max_value_f_on_0_4_l1895_189554

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_f_on_0_4 : ∃ (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (4 : ℝ)), ∀ (y : ℝ), y ∈ Set.Icc (0 : ℝ) (4 : ℝ) → f y ≤ f x ∧ f x = 1 / Real.exp 1 :=
by
  sorry

end max_value_f_on_0_4_l1895_189554


namespace problem_proof_l1895_189508

theorem problem_proof:
  (∃ n : ℕ, 25 = n ^ 2) ∧
  (Prime 31) ∧
  (¬ ∀ p : ℕ, Prime p → p >= 3 → p = 2) ∧
  (∃ m : ℕ, 8 = m ^ 3) ∧
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ 15 = a * b) :=
by
  sorry

end problem_proof_l1895_189508


namespace percentage_gain_second_week_l1895_189575

variables (initial_investment final_value after_first_week_value gain_percentage first_week_gain second_week_gain second_week_gain_percentage : ℝ)

def pima_investment (initial_investment: ℝ) (first_week_gain_percentage: ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage)

def second_week_investment (initial_investment first_week_gain_percentage second_week_gain_percentage : ℝ) : ℝ :=
  initial_investment * (1 + first_week_gain_percentage) * (1 + second_week_gain_percentage)

theorem percentage_gain_second_week
  (initial_investment : ℝ)
  (first_week_gain_percentage : ℝ)
  (final_value : ℝ)
  (h1: initial_investment = 400)
  (h2: first_week_gain_percentage = 0.25)
  (h3: final_value = 750) :
  second_week_gain_percentage = 0.5 :=
by
  let after_first_week_value := pima_investment initial_investment first_week_gain_percentage
  let second_week_gain := final_value - after_first_week_value
  let second_week_gain_percentage := second_week_gain / after_first_week_value * 100
  sorry

end percentage_gain_second_week_l1895_189575


namespace paige_folders_l1895_189586

-- Definitions derived from the conditions
def initial_files : Nat := 27
def deleted_files : Nat := 9
def files_per_folder : Nat := 6

-- Define the remaining files after deletion
def remaining_files : Nat := initial_files - deleted_files

-- The theorem: Prove that the number of folders is 3
theorem paige_folders : remaining_files / files_per_folder = 3 := by
  sorry

end paige_folders_l1895_189586


namespace solve_linear_equation_l1895_189526

theorem solve_linear_equation (x : ℝ) (h : 2 * x - 1 = 1) : x = 1 :=
sorry

end solve_linear_equation_l1895_189526


namespace max_difference_is_62_l1895_189584

open Real

noncomputable def max_difference_of_integers : ℝ :=
  let a (k : ℝ) := 2 * k + 1 + sqrt (8 * k)
  let b (k : ℝ) := 2 * k + 1 - sqrt (8 * k)
  let diff (k : ℝ) := a k - b k
  let max_k := 120 -- Maximum integer value k such that 2k + 1 + sqrt(8k) < 1000
  diff max_k

theorem max_difference_is_62 :
  max_difference_of_integers = 62 :=
sorry

end max_difference_is_62_l1895_189584


namespace boys_or_girls_rink_l1895_189519

variables (Class : Type) (is_boy : Class → Prop) (is_girl : Class → Prop) (visited_rink : Class → Prop) (met_at_rink : Class → Class → Prop)

-- Every student in the class visited the rink at least once.
axiom all_students_visited : ∀ (s : Class), visited_rink s

-- Every boy met every girl at the rink.
axiom boys_meet_girls : ∀ (b g : Class), is_boy b → is_girl g → met_at_rink b g

-- Prove that there exists a time when all the boys, or all the girls were simultaneously on the rink.
theorem boys_or_girls_rink : ∃ (t : Prop), (∀ b, is_boy b → visited_rink b) ∨ (∀ g, is_girl g → visited_rink g) :=
sorry

end boys_or_girls_rink_l1895_189519


namespace quad_eq_diagonals_theorem_l1895_189544

noncomputable def quad_eq_diagonals (a b c d m n : ℝ) (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem quad_eq_diagonals_theorem (a b c d m n A C : ℝ) :
  quad_eq_diagonals a b c d m n A C :=
by
  sorry

end quad_eq_diagonals_theorem_l1895_189544


namespace cos_two_pi_over_three_plus_two_alpha_l1895_189539

theorem cos_two_pi_over_three_plus_two_alpha 
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := 
by
  sorry

end cos_two_pi_over_three_plus_two_alpha_l1895_189539


namespace lattice_midpoint_l1895_189529

theorem lattice_midpoint (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ 
  let (x1, y1) := points i 
  let (x2, y2) := points j
  (x1 + x2) % 2 = 0 ∧ (y1 + y2) % 2 = 0 := 
sorry

end lattice_midpoint_l1895_189529


namespace perpendicular_slope_l1895_189542

theorem perpendicular_slope (x y : ℝ) (h : 3 * x - 4 * y = 8) :
  ∃ m : ℝ, m = - (4 / 3) :=
by
  sorry

end perpendicular_slope_l1895_189542


namespace parabola_focus_l1895_189546

theorem parabola_focus (a b c : ℝ) (h k : ℝ) (p : ℝ) :
  (a = 4) →
  (b = -4) →
  (c = -3) →
  (h = -b / (2 * a)) →
  (k = a * h ^ 2 + b * h + c) →
  (p = 1 / (4 * a)) →
  (k + p = -4 + 1 / 16) →
  (h, k + p) = (1 / 2, -63 / 16) :=
by
  intros a_eq b_eq c_eq h_eq k_eq p_eq focus_eq
  rw [a_eq, b_eq, c_eq] at *
  sorry

end parabola_focus_l1895_189546


namespace find_N_l1895_189548

theorem find_N (
    A B : ℝ) (N : ℕ) (r : ℝ) (hA : A = N * π * r^2 / 2) 
    (hB : B = (π * r^2 / 2) * (N^2 - N)) 
    (ratio : A / B = 1 / 18) : 
    N = 19 :=
by
  sorry

end find_N_l1895_189548


namespace bin_sum_sub_eq_l1895_189565

-- Define binary numbers
def b1 := 0b101110  -- binary 101110_2
def b2 := 0b10101   -- binary 10101_2
def b3 := 0b111000  -- binary 111000_2
def b4 := 0b110101  -- binary 110101_2
def b5 := 0b11101   -- binary 11101_2

-- Define the theorem
theorem bin_sum_sub_eq : ((b1 + b2) - (b3 - b4) + b5) = 0b1011101 := by
  sorry

end bin_sum_sub_eq_l1895_189565


namespace oliver_remaining_dishes_l1895_189531

def num_dishes := 36
def dishes_with_mango_salsa := 3
def dishes_with_fresh_mango := num_dishes / 6
def dishes_with_mango_jelly := 1
def oliver_picks_two := 2

theorem oliver_remaining_dishes : 
  num_dishes - (dishes_with_mango_salsa + dishes_with_fresh_mango + dishes_with_mango_jelly - oliver_picks_two) = 28 := by
  sorry

end oliver_remaining_dishes_l1895_189531


namespace problem_l1895_189571

noncomputable def f (x : ℝ) : ℝ := Real.sin x + x - Real.pi / 4
noncomputable def g (x : ℝ) : ℝ := Real.cos x - x + Real.pi / 4

theorem problem (x1 x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < Real.pi / 2) (hx2 : 0 < x2 ∧ x2 < Real.pi / 2) :
  (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ f x = 0) ∧ (∃! x, 0 < x ∧ x < Real.pi / 2 ∧ g x = 0) →
  x1 + x2 = Real.pi / 2 :=
by
  sorry -- Proof goes here

end problem_l1895_189571


namespace final_surface_area_l1895_189560

noncomputable def surface_area (total_cubes remaining_cubes cube_surface removed_internal_surface : ℕ) : ℕ :=
  (remaining_cubes * cube_surface) + (remaining_cubes * removed_internal_surface)

theorem final_surface_area :
  surface_area 64 55 54 6 = 3300 :=
by
  sorry

end final_surface_area_l1895_189560


namespace simplify_fraction_fraction_c_over_d_l1895_189533

-- Define necessary constants and variables
variable (k : ℤ)

/-- Original expression -/
def original_expr := (6 * k + 12 + 3 : ℤ)

/-- Simplified expression -/
def simplified_expr := (2 * k + 5 : ℤ)

/-- The main theorem to prove the equivalent mathematical proof problem -/
theorem simplify_fraction : (original_expr / 3) = simplified_expr :=
by
  sorry

-- The final fraction to prove the answer
theorem fraction_c_over_d : (2 / 5 : ℚ) = 2 / 5 :=
by
  sorry

end simplify_fraction_fraction_c_over_d_l1895_189533


namespace passing_percentage_l1895_189591

theorem passing_percentage
  (marks_obtained : ℕ)
  (marks_failed_by : ℕ)
  (max_marks : ℕ)
  (h_marks_obtained : marks_obtained = 92)
  (h_marks_failed_by : marks_failed_by = 40)
  (h_max_marks : max_marks = 400) :
  (marks_obtained + marks_failed_by) / max_marks * 100 = 33 := 
by
  sorry

end passing_percentage_l1895_189591


namespace cyclic_sum_inequality_l1895_189583

theorem cyclic_sum_inequality (n : ℕ) (a : Fin n.succ -> ℕ) (h : ∀ i, a i > 0) : 
  (Finset.univ.sum fun i => a i / a ((i + 1) % n)) ≥ n :=
by
  sorry

end cyclic_sum_inequality_l1895_189583


namespace integer_solution_l1895_189506

theorem integer_solution (x : ℕ) (h : (4 * x)^2 - 2 * x = 3178) : x = 226 :=
by
  sorry

end integer_solution_l1895_189506


namespace digit_x_for_divisibility_by_29_l1895_189597

-- Define the base 7 number 34x1_7 in decimal form
def base7_to_decimal (x : ℕ) : ℕ := 3 * 7^3 + 4 * 7^2 + x * 7 + 1

-- State the proof problem
theorem digit_x_for_divisibility_by_29 (x : ℕ) (h : base7_to_decimal x % 29 = 0) : x = 3 :=
by
  sorry

end digit_x_for_divisibility_by_29_l1895_189597


namespace missing_number_geometric_sequence_l1895_189545

theorem missing_number_geometric_sequence : 
  ∃ (x : ℤ), (x = 162) ∧ 
  (x = 54 * 3 ∧ 
  486 = x * 3 ∧ 
  ∀ a b : ℤ, (b = 2 * 3) ∧ 
              (a = 2 * 3) ∧ 
              (18 = b * 3) ∧ 
              (54 = 18 * 3) ∧ 
              (54 * 3 = x)) := 
by sorry

end missing_number_geometric_sequence_l1895_189545


namespace Douglas_weight_correct_l1895_189524

def Anne_weight : ℕ := 67
def weight_diff : ℕ := 15
def Douglas_weight : ℕ := 52

theorem Douglas_weight_correct : Douglas_weight = Anne_weight - weight_diff := by
  sorry

end Douglas_weight_correct_l1895_189524


namespace cube_face_area_l1895_189555

-- Definition for the condition of the cube's surface area
def cube_surface_area (s : ℝ) : Prop := s = 36

-- Definition stating a cube has 6 faces
def cube_faces : ℝ := 6

-- The target proposition to prove
theorem cube_face_area (s : ℝ) (area_of_one_face : ℝ) (h1 : cube_surface_area s) (h2 : cube_faces = 6) : area_of_one_face = s / 6 :=
by
  sorry

end cube_face_area_l1895_189555


namespace edges_after_truncation_l1895_189581

-- Define a regular tetrahedron with 4 vertices and 6 edges
structure Tetrahedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Initial regular tetrahedron
def initial_tetrahedron : Tetrahedron :=
  { vertices := 4, edges := 6 }

-- Function to calculate the number of edges after truncating vertices
def truncated_edges (t : Tetrahedron) (vertex_truncations : ℕ) (new_edges_per_vertex : ℕ) : ℕ :=
  vertex_truncations * new_edges_per_vertex

-- Given a regular tetrahedron and the truncation process
def resulting_edges (t : Tetrahedron) (vertex_truncations : ℕ) :=
  truncated_edges t vertex_truncations 3

-- Problem statement: Proving the resulting figure has 12 edges
theorem edges_after_truncation :
  resulting_edges initial_tetrahedron 4 = 12 :=
  sorry

end edges_after_truncation_l1895_189581


namespace part_I_part_II_l1895_189514

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := Real.exp x + a * x
noncomputable def g (a : ℝ) (x : ℝ) := x * Real.exp (a * x - 1) - 2 * a * x + f a x

def monotonicity_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x < f y

theorem part_I (a : ℝ) : 
  monotonicity_in_interval (f a) 0 (Real.log 3) = monotonicity_in_interval (F a) 0 (Real.log 3) ↔ a ≤ -3 :=
sorry

theorem part_II (a : ℝ) (ha : a ∈ Set.Iic (-1 / Real.exp 2)) : 
  (∃ x, x > 0 ∧ g a x = M) → M ≥ 0 :=
sorry

end part_I_part_II_l1895_189514


namespace simplify_rationalize_denominator_l1895_189598

theorem simplify_rationalize_denominator : 
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by
  sorry

end simplify_rationalize_denominator_l1895_189598


namespace households_soap_usage_l1895_189503

theorem households_soap_usage
  (total_households : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (only_B_ratio : ℕ)
  (B := only_B_ratio * both) :
  total_households = 200 →
  neither = 80 →
  both = 40 →
  only_B_ratio = 3 →
  (total_households - neither - both - B = 40) :=
by
  intros
  sorry

end households_soap_usage_l1895_189503


namespace max_sum_of_diagonals_l1895_189579

theorem max_sum_of_diagonals (a b : ℝ) (h_side : a^2 + b^2 = 25) (h_bounds1 : 2 * a ≤ 6) (h_bounds2 : 2 * b ≥ 6) : 2 * (a + b) = 14 :=
sorry

end max_sum_of_diagonals_l1895_189579


namespace pqrsum_l1895_189587

-- Given constants and conditions:
variables {p q r : ℝ} -- p, q, r are real numbers
axiom Hpq : p < q -- given condition p < q
axiom Hineq : ∀ x : ℝ, (x > 5 ∨ 7 ≤ x ∧ x ≤ 15) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0) -- given inequality condition

-- Values from the solution:
axiom Hp : p = 7
axiom Hq : q = 15
axiom Hr : r = 5

-- Proof statement:
theorem pqrsum : p + 2 * q + 3 * r = 52 :=
sorry 

end pqrsum_l1895_189587


namespace base10_to_base8_440_l1895_189552

theorem base10_to_base8_440 :
  ∃ k1 k2 k3,
    k1 = 6 ∧
    k2 = 7 ∧
    k3 = 0 ∧
    (440 = k1 * 64 + k2 * 8 + k3) ∧
    (64 = 8^2) ∧
    (8^3 > 440) :=
sorry

end base10_to_base8_440_l1895_189552


namespace equation_of_line_bisecting_chord_l1895_189561

theorem equation_of_line_bisecting_chord
  (P : ℝ × ℝ) 
  (A B : ℝ × ℝ)
  (P_bisects_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (P_on_ellipse : 3 * P.1^2 + 4 * P.2^2 = 24)
  (A_on_ellipse : 3 * A.1^2 + 4 * A.2^2 = 24)
  (B_on_ellipse : 3 * B.1^2 + 4 * B.2^2 = 24) :
  ∃ (a b c : ℝ), a * P.2 + b * P.1 + c = 0 ∧ a = 2 ∧ b = -3 ∧ c = 7 :=
by 
  sorry

end equation_of_line_bisecting_chord_l1895_189561


namespace solve_indeterminate_equation_l1895_189504

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solve_indeterminate_equation (x y : ℕ) (hx : is_prime x) (hy : is_prime y) :
  x^2 - y^2 = x * y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) :=
by
  sorry

end solve_indeterminate_equation_l1895_189504


namespace layla_more_than_nahima_l1895_189535

-- Definitions for the conditions
def layla_points : ℕ := 70
def total_points : ℕ := 112
def nahima_points : ℕ := total_points - layla_points

-- Statement of the theorem
theorem layla_more_than_nahima : layla_points - nahima_points = 28 :=
by
  sorry

end layla_more_than_nahima_l1895_189535


namespace max_difference_l1895_189596

theorem max_difference (U V W X Y Z : ℕ) (hUVW : U ≠ V ∧ V ≠ W ∧ U ≠ W)
    (hXYZ : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) (digits_UVW : 1 ≤ U ∧ U ≤ 9 ∧ 1 ≤ V ∧ V ≤ 9 ∧ 1 ≤ W ∧ W ≤ 9)
    (digits_XYZ : 1 ≤ X ∧ X ≤ 9 ∧ 1 ≤ Y ∧ Y ≤ 9 ∧ 1 ≤ Z ∧ Z ≤ 9) :
    U * 100 + V * 10 + W = 987 → X * 100 + Y * 10 + Z = 123 → (U * 100 + V * 10 + W) - (X * 100 + Y * 10 + Z) = 864 :=
by
  sorry

end max_difference_l1895_189596


namespace absent_children_count_l1895_189547

theorem absent_children_count (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ)
    (absent_children : ℕ) (total_bananas : ℕ) (present_children : ℕ) :
    total_children = 640 →
    bananas_per_child = 2 →
    extra_bananas_per_child = 2 →
    total_bananas = (total_children * bananas_per_child) →
    present_children = (total_children - absent_children) →
    total_bananas = (present_children * (bananas_per_child + extra_bananas_per_child)) →
    absent_children = 320 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end absent_children_count_l1895_189547


namespace min_x2_plus_y2_l1895_189532

noncomputable def min_val (x y : ℝ) : ℝ :=
  if h : (x + 1)^2 + y^2 = 1/4 then x^2 + y^2 else 0

theorem min_x2_plus_y2 : 
  ∃ x y : ℝ, (x + 1)^2 + y^2 = 1/4 ∧ x^2 + y^2 = 1/4 :=
by
  sorry

end min_x2_plus_y2_l1895_189532


namespace tautology_a_tautology_b_tautology_c_tautology_d_l1895_189505

variable (p q : Prop)

theorem tautology_a : p ∨ ¬ p := by
  sorry

theorem tautology_b : ¬ ¬ p ↔ p := by
  sorry

theorem tautology_c : ((p → q) → p) → p := by
  sorry

theorem tautology_d : ¬ (p ∧ ¬ p) := by
  sorry

end tautology_a_tautology_b_tautology_c_tautology_d_l1895_189505


namespace number_of_rel_prime_to_21_in_range_l1895_189569

def is_rel_prime (a b : ℕ) : Prop := gcd a b = 1

noncomputable def count_rel_prime_in_range (a b g : ℕ) : ℕ :=
  ((b - a + 1) : ℕ) - ((b / 3 - (a - 1) / 3) + (b / 7 - (a - 1) / 7) - (b / 21 - (a - 1) / 21))

theorem number_of_rel_prime_to_21_in_range :
  count_rel_prime_in_range 11 99 21 = 51 :=
by 
  sorry

end number_of_rel_prime_to_21_in_range_l1895_189569


namespace simplify_expression_l1895_189543

theorem simplify_expression (w : ℝ) :
  3 * w + 4 - 2 * w - 5 + 6 * w + 7 - 3 * w - 9 = 4 * w - 3 :=
by 
  sorry

end simplify_expression_l1895_189543


namespace intersection_S_T_eq_T_l1895_189564

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l1895_189564


namespace probability_YW_correct_l1895_189527

noncomputable def probability_YW_greater_than_six_sqrt_three (XY YZ XZ YW : ℝ) : ℝ :=
  if H : XY = 12 ∧ YZ = 6 ∧ XZ = 6 * Real.sqrt 3 then 
    if YW > 6 * Real.sqrt 3 then (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3
    else 0
  else 0

theorem probability_YW_correct : probability_YW_greater_than_six_sqrt_three 12 6 (6 * Real.sqrt 3) (6 * Real.sqrt 3) = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
sorry

end probability_YW_correct_l1895_189527


namespace problem_1_l1895_189551

theorem problem_1 (f : ℝ → ℝ) (hf_mul : ∀ x y : ℝ, f (x * y) = f x + f y) (hf_4 : f 4 = 2) : f (Real.sqrt 2) = 1 / 2 :=
sorry

end problem_1_l1895_189551


namespace max_points_of_intersection_l1895_189592

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l1895_189592


namespace largest_k_exists_l1895_189562

theorem largest_k_exists (n : ℕ) (h : n ≥ 4) : 
  ∃ k : ℕ, (∀ (a b c : ℕ), 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ n → (c - b) ≥ k ∧ (b - a) ≥ k ∧ (a + b ≥ c + 1)) ∧ 
  (k = (n - 1) / 3) :=
  sorry

end largest_k_exists_l1895_189562


namespace car_speed_ratio_l1895_189589

noncomputable def speed_ratio (t_round_trip t_leaves t_returns t_walk_start t_walk_end : ℕ) (meet_time : ℕ) : ℕ :=
  let one_way_time_car := t_round_trip / 2
  let total_car_time := t_returns - t_leaves
  let meeting_time_car := total_car_time / 2
  let remaining_time_to_factory := one_way_time_car - meeting_time_car
  let total_walk_time := t_walk_end - t_walk_start
  total_walk_time / remaining_time_to_factory

theorem car_speed_ratio :
  speed_ratio 60 120 160 60 140 80 = 8 :=
by
  sorry

end car_speed_ratio_l1895_189589


namespace inverse_variation_l1895_189563

theorem inverse_variation (a : ℕ) (b : ℝ) (h : a * b = 400) (h₀ : a = 3200) : b = 0.125 :=
by sorry

end inverse_variation_l1895_189563


namespace brian_breath_proof_l1895_189574

def breath_holding_time (initial_time: ℕ) (week1_factor: ℝ) (week2_factor: ℝ) 
  (missed_days: ℕ) (missed_decrease: ℝ) (week3_factor: ℝ): ℝ := by
  let week1_time := initial_time * week1_factor
  let hypothetical_week2_time := week1_time * (1 + week2_factor)
  let missed_decrease_total := week1_time * missed_decrease * missed_days
  let effective_week2_time := hypothetical_week2_time - missed_decrease_total
  let final_time := effective_week2_time * (1 + week3_factor)
  exact final_time

theorem brian_breath_proof :
  breath_holding_time 10 2 0.75 2 0.1 0.5 = 46.5 := 
by
  sorry

end brian_breath_proof_l1895_189574


namespace base_seven_sum_of_product_l1895_189534

def base_seven_to_decimal (d1 d0 : ℕ) : ℕ :=
  7 * d1 + d0

def decimal_to_base_seven (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d3 := n / (7 ^ 3)
  let r3 := n % (7 ^ 3)
  let d2 := r3 / (7 ^ 2)
  let r2 := r3 % (7 ^ 2)
  let d1 := r2 / 7
  let d0 := r2 % 7
  (d3, d2, d1, d0)

def sum_of_base_seven_digits (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 + d2 + d1 + d0

theorem base_seven_sum_of_product :
  let n1 := base_seven_to_decimal 3 5
  let n2 := base_seven_to_decimal 4 2
  let product := n1 * n2
  let (d3, d2, d1, d0) := decimal_to_base_seven product
  sum_of_base_seven_digits d3 d2 d1 d0 = 18 :=
  by
    sorry

end base_seven_sum_of_product_l1895_189534


namespace algebraic_expression_simplification_l1895_189558

theorem algebraic_expression_simplification (k x : ℝ) (h : (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :
  k = 3 ∨ k = -3 :=
by {
  sorry
}

end algebraic_expression_simplification_l1895_189558


namespace no_equal_refereed_matches_l1895_189521

theorem no_equal_refereed_matches {k : ℕ} (h1 : ∀ {n : ℕ}, n > k → n = 2 * k) 
    (h2 : ∀ {n : ℕ}, n > k → ∃ m, m = k * (2 * k - 1))
    (h3 : ∀ {n : ℕ}, n > k → ∃ r, r = (2 * k - 1) / 2): 
    False := 
by
  sorry

end no_equal_refereed_matches_l1895_189521


namespace average_weight_increase_l1895_189520

theorem average_weight_increase
  (initial_weight replaced_weight : ℝ)
  (num_persons : ℕ)
  (avg_increase : ℝ)
  (h₁ : num_persons = 5)
  (h₂ : replaced_weight = 65)
  (h₃ : avg_increase = 1.5)
  (total_increase : ℝ)
  (new_weight : ℝ)
  (h₄ : total_increase = num_persons * avg_increase)
  (h₅ : total_increase = new_weight - replaced_weight) :
  new_weight = 72.5 :=
by
  sorry

end average_weight_increase_l1895_189520


namespace books_finished_l1895_189582

theorem books_finished (miles_traveled : ℕ) (miles_per_book : ℕ) (h_travel : miles_traveled = 6760) (h_rate : miles_per_book = 450) : (miles_traveled / miles_per_book) = 15 :=
by {
  -- Proof will be inserted here
  sorry
}

end books_finished_l1895_189582


namespace candy_count_after_giving_l1895_189541

def numKitKats : ℕ := 5
def numHersheyKisses : ℕ := 3 * numKitKats
def numNerds : ℕ := 8
def numLollipops : ℕ := 11
def numBabyRuths : ℕ := 10
def numReeseCups : ℕ := numBabyRuths / 2
def numLollipopsGivenAway : ℕ := 5

def totalCandyBefore : ℕ := numKitKats + numHersheyKisses + numNerds + numLollipops + numBabyRuths + numReeseCups
def totalCandyAfter : ℕ := totalCandyBefore - numLollipopsGivenAway

theorem candy_count_after_giving : totalCandyAfter = 49 := by
  sorry

end candy_count_after_giving_l1895_189541


namespace cent_piece_value_l1895_189517

theorem cent_piece_value (Q P : ℕ) 
  (h1 : Q + P = 29)
  (h2 : 25 * Q + P = 545)
  (h3 : Q = 17) : 
  P = 120 := by
  sorry

end cent_piece_value_l1895_189517


namespace number_of_friends_l1895_189553

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def envelopes_left : ℕ := 22

theorem number_of_friends :
  ((total_envelopes - envelopes_left) / envelopes_per_friend) = 5 := by
  sorry

end number_of_friends_l1895_189553


namespace angle_C_in_triangle_l1895_189576

theorem angle_C_in_triangle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A + B = 115) : C = 65 := 
by 
  sorry

end angle_C_in_triangle_l1895_189576


namespace p_necessary_not_sufficient_for_q_l1895_189536

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def p (a : ℝ) : Prop :=
  collinear (vec a (a^2)) (vec 1 2)

def q (a : ℝ) : Prop := a = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ ¬(∀ a : ℝ, p a → q a) :=
sorry

end p_necessary_not_sufficient_for_q_l1895_189536


namespace translated_line_eqn_l1895_189538

theorem translated_line_eqn
  (c : ℝ) :
  ∀ (y_eqn : ℝ → ℝ), 
    (∀ x, y_eqn x = 2 * x + 1) →
    (∀ x, (y_eqn (x - 2) - 3) = (2 * x - 6)) :=
by
  sorry

end translated_line_eqn_l1895_189538


namespace no_geometric_progression_l1895_189550

theorem no_geometric_progression (r s t : ℕ) (h1 : r < s) (h2 : s < t) :
  ¬ ∃ (b : ℂ), (3^r - 2^r) * b^(s - r) = 3^s - 2^s ∧ (3^s - 2^s) * b^(t - s) = 3^t - 2^t := by
  sorry

end no_geometric_progression_l1895_189550


namespace watch_cost_price_l1895_189502

theorem watch_cost_price (CP : ℝ) (H1 : 0.90 * CP = CP - 0.10 * CP)
(H2 : 1.04 * CP = CP + 0.04 * CP)
(H3 : 1.04 * CP - 0.90 * CP = 168) : CP = 1200 := by
sorry

end watch_cost_price_l1895_189502


namespace find_polygon_sides_l1895_189515

theorem find_polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by
  sorry

end find_polygon_sides_l1895_189515


namespace cistern_filled_in_12_hours_l1895_189511

def fill_rate := 1 / 6
def empty_rate := 1 / 12
def net_rate := fill_rate - empty_rate

theorem cistern_filled_in_12_hours :
  (1 / net_rate) = 12 :=
by
  -- Proof omitted for clarity
  sorry

end cistern_filled_in_12_hours_l1895_189511


namespace gray_eyed_black_haired_students_l1895_189518

theorem gray_eyed_black_haired_students :
  ∀ (students : ℕ)
    (green_eyed_red_haired : ℕ)
    (black_haired : ℕ)
    (gray_eyed : ℕ),
    students = 60 →
    green_eyed_red_haired = 20 →
    black_haired = 40 →
    gray_eyed = 25 →
    (gray_eyed - (students - black_haired - green_eyed_red_haired)) = 25 := by
  intros students green_eyed_red_haired black_haired gray_eyed
  intros h_students h_green h_black h_gray
  sorry

end gray_eyed_black_haired_students_l1895_189518


namespace f_at_8_l1895_189510

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem f_at_8 : f 8 = -1 := 
by
-- The following will be filled with the proof, hence sorry for now.
sorry

end f_at_8_l1895_189510


namespace change_in_y_when_x_increases_l1895_189516

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- State the theorem
theorem change_in_y_when_x_increases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -5 :=
by
  sorry

end change_in_y_when_x_increases_l1895_189516


namespace chickens_and_rabbits_l1895_189513

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (rabbits : ℕ) 
    (h1 : total_animals = 40) 
    (h2 : total_legs = 108) 
    (h3 : total_animals = chickens + rabbits) 
    (h4 : total_legs = 2 * chickens + 4 * rabbits) : 
    chickens = 26 ∧ rabbits = 14 :=
by
  sorry

end chickens_and_rabbits_l1895_189513


namespace total_cost_toys_l1895_189537

variable (c_e_actionfigs : ℕ := 60) -- number of action figures for elder son
variable (cost_e_actionfig : ℕ := 5) -- cost per action figure for elder son
variable (c_y_actionfigs : ℕ := 3 * c_e_actionfigs) -- number of action figures for younger son
variable (cost_y_actionfig : ℕ := 4) -- cost per action figure for younger son
variable (c_y_cars : ℕ := 20) -- number of cars for younger son
variable (cost_car : ℕ := 3) -- cost per car
variable (c_y_animals : ℕ := 10) -- number of stuffed animals for younger son
variable (cost_animal : ℕ := 7) -- cost per stuffed animal

theorem total_cost_toys (c_e_actionfigs c_y_actionfigs c_y_cars c_y_animals : ℕ)
                         (cost_e_actionfig cost_y_actionfig cost_car cost_animal : ℕ) :
  (c_e_actionfigs * cost_e_actionfig + c_y_actionfigs * cost_y_actionfig + 
  c_y_cars * cost_car + c_y_animals * cost_animal) = 1150 := by
  sorry

end total_cost_toys_l1895_189537


namespace russia_is_one_third_bigger_l1895_189590

theorem russia_is_one_third_bigger (U : ℝ) (Canada Russia : ℝ) 
  (h1 : Canada = 1.5 * U) (h2 : Russia = 2 * U) : 
  (Russia - Canada) / Canada = 1 / 3 :=
by
  sorry

end russia_is_one_third_bigger_l1895_189590


namespace real_root_if_and_only_if_l1895_189523

theorem real_root_if_and_only_if (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end real_root_if_and_only_if_l1895_189523


namespace arithmetic_sequence_sum_l1895_189507

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h0 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h1 : S 10 = 12)
  (h2 : S 20 = 17) :
  S 30 = 15 := by
  sorry

end arithmetic_sequence_sum_l1895_189507


namespace min_distance_eq_sqrt2_l1895_189568

open Real

variables {P Q : ℝ × ℝ}
variables {x y : ℝ}

/-- Given that point P is on the curve y = e^x and point Q is on the curve y = ln x, prove that the minimum value of the distance |PQ| is sqrt(2). -/
theorem min_distance_eq_sqrt2 : 
  (P.2 = exp P.1) ∧ (Q.2 = log Q.1) → (dist P Q) = sqrt 2 :=
by
  sorry

end min_distance_eq_sqrt2_l1895_189568


namespace percentage_of_males_l1895_189567

theorem percentage_of_males (total_employees males_below_50 males_percentage : ℕ) (h1 : total_employees = 800) (h2 : males_below_50 = 120) (h3 : 40 * males_percentage / 100 = 60 * males_below_50):
  males_percentage = 25 :=
by
  sorry

end percentage_of_males_l1895_189567


namespace added_amount_correct_l1895_189566

theorem added_amount_correct (n x : ℕ) (h1 : n = 20) (h2 : 1/2 * n + x = 15) :
  x = 5 :=
by
  sorry

end added_amount_correct_l1895_189566


namespace find_a_minus_b_l1895_189578

theorem find_a_minus_b (a b : ℚ)
  (h1 : 2 = a + b / 2)
  (h2 : 7 = a - b / 2)
  : a - b = 19 / 2 := 
  sorry

end find_a_minus_b_l1895_189578


namespace twenty_five_billion_scientific_notation_l1895_189512

theorem twenty_five_billion_scientific_notation :
  (25 * 10^9 : ℝ) = 2.5 * 10^10 := 
by simp only [←mul_assoc, ←@pow_add ℝ, pow_one, two_mul];
   norm_num

end twenty_five_billion_scientific_notation_l1895_189512


namespace chair_cost_l1895_189599

theorem chair_cost (T P n : ℕ) (hT : T = 135) (hP : P = 55) (hn : n = 4) : 
  (T - P) / n = 20 := by
  sorry

end chair_cost_l1895_189599


namespace simplify_expression_l1895_189594

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : 
  (2 / y^2 - y⁻¹) = (2 - y) / y^2 :=
by sorry

end simplify_expression_l1895_189594


namespace range_of_m_l1895_189588

theorem range_of_m (m : ℝ) (h : (m^2 + m) ^ (3 / 5) ≤ (3 - m) ^ (3 / 5)) : 
  -3 ≤ m ∧ m ≤ 1 :=
by { sorry }

end range_of_m_l1895_189588


namespace total_salmons_caught_l1895_189528

theorem total_salmons_caught :
  let hazel_salmons := 24
  let dad_salmons := 27
  hazel_salmons + dad_salmons = 51 :=
by
  sorry

end total_salmons_caught_l1895_189528


namespace blue_paint_gallons_l1895_189580

-- Define the total gallons of paint used
def total_paint_gallons : ℕ := 6689

-- Define the gallons of white paint used
def white_paint_gallons : ℕ := 660

-- Define the corresponding proof problem
theorem blue_paint_gallons : 
  ∀ total white blue : ℕ, total = 6689 → white = 660 → blue = total - white → blue = 6029 := by
  sorry

end blue_paint_gallons_l1895_189580


namespace midpoint_coordinates_l1895_189577

theorem midpoint_coordinates (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 2) (hy1 : y1 = 10) (hx2 : x2 = 6) (hy2 : y2 = 2) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx = 4 ∧ my = 6 :=
by
  sorry

end midpoint_coordinates_l1895_189577


namespace race_total_people_l1895_189549

theorem race_total_people (b t : ℕ) 
(h1 : b = t + 15) 
(h2 : 3 * t = 2 * b + 15) : 
b + t = 105 := 
sorry

end race_total_people_l1895_189549


namespace initial_population_is_10000_l1895_189525

def population_growth (P : ℝ) : Prop :=
  let growth_rate := 0.20
  let final_population := 12000
  final_population = P * (1 + growth_rate)

theorem initial_population_is_10000 : population_growth 10000 :=
by
  unfold population_growth
  sorry

end initial_population_is_10000_l1895_189525


namespace wrapping_paper_area_l1895_189557

variable {l w h : ℝ}

theorem wrapping_paper_area (hl : 0 < l) (hw : 0 < w) (hh : 0 < h) :
  (4 * l * h + 2 * l * h + 2 * w * h) = 6 * l * h + 2 * w * h :=
  sorry

end wrapping_paper_area_l1895_189557


namespace total_time_to_complete_work_l1895_189559

-- Definitions based on conditions
variable (W : ℝ) -- W is the total work
variable (Mahesh_days : ℝ := 35) -- Mahesh can complete the work in 35 days
variable (Mahesh_working_days : ℝ := 20) -- Mahesh works for 20 days
variable (Rajesh_days : ℝ := 30) -- Rajesh finishes the remaining work in 30 days

-- Proof statement
theorem total_time_to_complete_work : Mahesh_working_days + Rajesh_days = 50 :=
by
  sorry

end total_time_to_complete_work_l1895_189559


namespace tower_total_surface_area_l1895_189509

/-- Given seven cubes with volumes 1, 8, 27, 64, 125, 216, and 343 cubic units each, stacked vertically
    with volumes decreasing from bottom to top, compute their total surface area including the bottom. -/
theorem tower_total_surface_area :
  let volumes := [1, 8, 27, 64, 125, 216, 343]
  let side_lengths := volumes.map (fun v => v ^ (1 / 3))
  let surface_area (n : ℝ) (visible_faces : ℕ) := visible_faces * (n ^ 2)
  let total_surface_area := surface_area 7 5 + surface_area 6 4 + surface_area 5 4 + surface_area 4 4
                            + surface_area 3 4 + surface_area 2 4 + surface_area 1 5
  total_surface_area = 610 := sorry

end tower_total_surface_area_l1895_189509


namespace product_is_58_l1895_189500

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def p := 2
def q := 29

-- Conditions based on the problem
axiom prime_p : is_prime p
axiom prime_q : is_prime q
axiom sum_eq_31 : p + q = 31

-- Theorem to be proven
theorem product_is_58 : p * q = 58 :=
by
  sorry

end product_is_58_l1895_189500


namespace sum_of_interior_angles_l1895_189585

theorem sum_of_interior_angles (n : ℕ) (h1 : 180 * (n - 2) = 1800) (h2 : n = 12) : 
  180 * ((n + 4) - 2) = 2520 := 
by 
  { sorry }

end sum_of_interior_angles_l1895_189585


namespace find_k_l1895_189570

theorem find_k (k : ℝ) : 
  (∃ c1 c2 : ℝ, (2 * c1^2 + 5 * c1 = k) ∧ 
                (2 * c2^2 + 5 * c2 = k) ∧ 
                (c1 > c2) ∧ 
                (c1 - c2 = 5.5)) → 
  k = 12 := 
by
  intros h
  obtain ⟨c1, c2, h1, h2, h3, h4⟩ := h
  sorry

end find_k_l1895_189570


namespace sequence_general_term_l1895_189573

noncomputable def a (n : ℕ) : ℝ :=
if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n ≠ 0) : 
  a n = if n = 1 then 1 else (n : ℝ) / (2 ^ (n - 1)) :=
by
  sorry

end sequence_general_term_l1895_189573


namespace f_value_plus_deriv_l1895_189530

noncomputable def f : ℝ → ℝ := sorry

-- Define the function f and its derivative at x = 1
axiom f_deriv_at_1 : deriv f 1 = 1 / 2

-- Define the value of the function f at x = 1
axiom f_value_at_1 : f 1 = 5 / 2

-- Prove that f(1) + f'(1) = 3
theorem f_value_plus_deriv : f 1 + deriv f 1 = 3 :=
by
  rw [f_value_at_1, f_deriv_at_1]
  norm_num

end f_value_plus_deriv_l1895_189530


namespace equation_of_line_AB_l1895_189593

theorem equation_of_line_AB 
  (x y : ℝ)
  (passes_through_P : (4 - 1)^2 + (1 - 0)^2 = 1)     
  (circle_eq : (x - 1)^2 + y^2 = 1) :
  3 * x + y - 4 = 0 :=
sorry

end equation_of_line_AB_l1895_189593


namespace range_of_a_l1895_189522

-- Definition of sets A and B
def set_A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def set_B (a : ℝ) := {x : ℝ | 0 < x ∧ x < a}

-- Statement that if A ⊆ B, then a > 3
theorem range_of_a (a : ℝ) (h : set_A ⊆ set_B a) : 3 < a :=
by sorry

end range_of_a_l1895_189522


namespace division_decimal_l1895_189572

theorem division_decimal (x : ℝ) (h : x = 0.3333): 12 / x = 36 :=
  by
    sorry

end division_decimal_l1895_189572
