import Mathlib

namespace geometric_sequence_sum_l405_40517

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r)
    (h2 : r = 2) (h3 : a 1 * 2 + a 3 * 8 + a 5 * 32 = 3) :
    a 4 * 16 + a 6 * 64 + a 8 * 256 = 24 :=
sorry

end geometric_sequence_sum_l405_40517


namespace quadratic_inequality_solution_l405_40523

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end quadratic_inequality_solution_l405_40523


namespace solve_arctan_equation_l405_40546

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan (1 / x) + Real.arctan (1 / (x^3))

theorem solve_arctan_equation (x : ℝ) (hx : x = (1 + Real.sqrt 5) / 2) :
  f x = Real.pi / 4 :=
by
  rw [hx]
  sorry

end solve_arctan_equation_l405_40546


namespace teachers_students_relationship_l405_40573

variables (m n k l : ℕ)

theorem teachers_students_relationship
  (teachers_count : m > 0)
  (students_count : n > 0)
  (students_per_teacher : k > 0)
  (teachers_per_student : l > 0)
  (h1 : ∀ p ∈ (Finset.range m), (Finset.card (Finset.range k)) = k)
  (h2 : ∀ s ∈ (Finset.range n), (Finset.card (Finset.range l)) = l) :
  m * k = n * l :=
sorry

end teachers_students_relationship_l405_40573


namespace subtract_largest_unit_fraction_l405_40506

theorem subtract_largest_unit_fraction
  (a b n : ℕ) (ha : a > 0) (hb : b > a) (hn : 1 ≤ b * n ∧ b * n <= a * n + b): 
  (a * n - b < a) := by
  sorry

end subtract_largest_unit_fraction_l405_40506


namespace problem_l405_40528

variable {a b c : ℝ}

theorem problem (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 := 
sorry

end problem_l405_40528


namespace max_C_usage_l405_40556

-- Definition of variables (concentration percentages and weights)
def A_conc := 3 / 100
def B_conc := 8 / 100
def C_conc := 11 / 100

def target_conc := 7 / 100
def total_weight := 100

def max_A := 50
def max_B := 70
def max_C := 60

-- Equation to satisfy
def conc_equation (x y : ℝ) : Prop :=
  C_conc * x + B_conc * y + A_conc * (total_weight - x - y) = target_conc * total_weight

-- Definition with given constraints
def within_constraints (x y : ℝ) : Prop :=
  x ≤ max_C ∧ y ≤ max_B ∧ (total_weight - x - y) ≤ max_A

-- The theorem that needs to be proved
theorem max_C_usage (x y : ℝ) : within_constraints x y ∧ conc_equation x y → x ≤ 50 :=
by
  sorry

end max_C_usage_l405_40556


namespace root_in_interval_2_3_l405_40518

noncomputable def f (x : ℝ) : ℝ := -|x - 5| + 2^(x - 1)

theorem root_in_interval_2_3 :
  (f 2) * (f 3) < 0 → ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 := by sorry

end root_in_interval_2_3_l405_40518


namespace complement_intersection_range_of_a_l405_40568

open Set

variable {α : Type*} [TopologicalSpace α]

def U : Set ℝ := univ

def A : Set ℝ := { x | -1 < x ∧ x < 1 }

def B : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 3/2 }

def C (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x ≤ 2 * a - 7 }

-- Question 1
theorem complement_intersection (x : ℝ) :
  x ∈ (U \ A) ∩ B ↔ 1 ≤ x ∧ x ≤ 3 / 2 := sorry

-- Question 2
theorem range_of_a {a : ℝ} (h : A ∩ C a = C a) : a < 4 := sorry

end complement_intersection_range_of_a_l405_40568


namespace max_water_bottles_one_athlete_l405_40529

-- Define variables and key conditions
variable (total_bottles : Nat := 40)
variable (total_athletes : Nat := 25)
variable (at_least_one : ∀ i, i < total_athletes → Nat.succ i ≥ 1)

-- Define the problem as a theorem
theorem max_water_bottles_one_athlete (h_distribution : total_bottles = 40) :
  ∃ max_bottles, max_bottles = 16 :=
by
  sorry

end max_water_bottles_one_athlete_l405_40529


namespace maximum_a_value_condition_l405_40526

theorem maximum_a_value_condition (x a : ℝ) :
  (∀ x, (x^2 - 2 * x - 3 > 0 → x < a)) ↔ a ≤ -1 :=
by sorry

end maximum_a_value_condition_l405_40526


namespace circle_equations_l405_40582

-- Given conditions: the circle passes through points O(0,0), A(1,1), B(4,2)
-- Prove the general equation of the circle and the standard equation 

theorem circle_equations : 
  ∃ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ 
                      (x, y) = (0, 0) ∨ (x, y) = (1, 1) ∨ (x, y) = (4, 2)) ∧
  (D = -8) ∧ (E = 6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y = 0 ↔ (x - 4)^2 + (y + 3)^2 = 25) :=
sorry

end circle_equations_l405_40582


namespace gcd_2000_7700_l405_40572

theorem gcd_2000_7700 : Nat.gcd 2000 7700 = 100 := by
  -- Prime factorizations of 2000 and 7700
  have fact_2000 : 2000 = 2^4 * 5^3 := sorry
  have fact_7700 : 7700 = 2^2 * 5^2 * 7 * 11 := sorry
  -- Proof of gcd
  sorry

end gcd_2000_7700_l405_40572


namespace exists_x_y_with_specific_difference_l405_40574

theorem exists_x_y_with_specific_difference :
  ∃ x y : ℤ, (2 * x^2 + 8 * y = 26) ∧ (x - y = 26) := 
sorry

end exists_x_y_with_specific_difference_l405_40574


namespace problem_statement_l405_40522

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

def set_otimes (A B : Set ℝ) : Set ℝ := {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

theorem problem_statement : set_otimes M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end problem_statement_l405_40522


namespace election_threshold_l405_40533

theorem election_threshold (total_votes geoff_percent_more_votes : ℕ) (geoff_vote_percent : ℚ) (geoff_votes_needed extra_votes_needed : ℕ) (threshold_percent : ℚ) :
  total_votes = 6000 → 
  geoff_vote_percent = 0.5 → 
  geoff_votes_needed = (geoff_vote_percent / 100) * total_votes →
  extra_votes_needed = 3000 → 
  (geoff_votes_needed + extra_votes_needed) / total_votes * 100 = threshold_percent →
  threshold_percent = 50.5 := 
by
  intros total_votes_eq geoff_vote_percent_eq geoff_votes_needed_eq extra_votes_needed_eq threshold_eq
  sorry

end election_threshold_l405_40533


namespace max_min_f_l405_40575

noncomputable def f (x : ℝ) : ℝ := 
  5 * Real.cos x ^ 2 - 6 * Real.sin (2 * x) + 20 * Real.sin x - 30 * Real.cos x + 7

theorem max_min_f :
  (∃ x : ℝ, f x = 16 + 10 * Real.sqrt 13) ∧
  (∃ x : ℝ, f x = 16 - 10 * Real.sqrt 13) :=
sorry

end max_min_f_l405_40575


namespace min_val_xy_l405_40510

theorem min_val_xy (x y : ℝ) 
  (h : 2 * (Real.cos (x + y - 1))^2 = ((x + 1)^2 + (y - 1)^2 - 2 * x * y) / (x - y + 1)) : 
  xy ≥ (1 / 4) :=
sorry

end min_val_xy_l405_40510


namespace sum_of_squares_of_roots_l405_40589

theorem sum_of_squares_of_roots (α β : ℝ)
  (h_root1 : 10 * α^2 - 14 * α - 24 = 0)
  (h_root2 : 10 * β^2 - 14 * β - 24 = 0)
  (h_distinct : α ≠ β) :
  α^2 + β^2 = 169 / 25 :=
sorry

end sum_of_squares_of_roots_l405_40589


namespace bales_stored_in_barn_l405_40500

-- Defining the conditions
def bales_initial : Nat := 28
def bales_stacked : Nat := 28
def bales_already_there : Nat := 54

-- Formulate the proof statement
theorem bales_stored_in_barn : bales_already_there + bales_stacked = 82 := by
  sorry

end bales_stored_in_barn_l405_40500


namespace total_handshakes_tournament_l405_40593

/-- 
In a women's doubles tennis tournament, four teams of two women competed. After the tournament, 
each woman shook hands only once with each of the other players, except with her own partner.
Prove that the total number of unique handshakes is 24.
-/
theorem total_handshakes_tournament : 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  total_handshakes = 24 :=
by 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  have : total_handshakes = 24 := sorry
  exact this

end total_handshakes_tournament_l405_40593


namespace max_log2_x_2log2_y_l405_40599

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem max_log2_x_2log2_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y^2 = 2) :
  log2 x + 2 * log2 y ≤ 0 :=
sorry

end max_log2_x_2log2_y_l405_40599


namespace percent_students_at_trip_l405_40550

variable (total_students : ℕ)
variable (students_taking_more_than_100 : ℕ := (14 * total_students) / 100)
variable (students_not_taking_more_than_100 : ℕ := (75 * total_students) / 100)
variable (students_who_went_to_trip := (students_taking_more_than_100 * 100) / 25)

/--
  If 14 percent of the students at a school went to a camping trip and took more than $100,
  and 75 percent of the students who went to the camping trip did not take more than $100,
  then 56 percent of the students at the school went to the camping trip.
-/
theorem percent_students_at_trip :
    (students_who_went_to_trip * 100) / total_students = 56 :=
sorry

end percent_students_at_trip_l405_40550


namespace mrs_hilt_candy_l405_40583

theorem mrs_hilt_candy : 2 * 9 + 3 * 9 + 1 * 9 = 54 :=
by
  sorry

end mrs_hilt_candy_l405_40583


namespace spadesuit_evaluation_l405_40535

-- Define the operation
def spadesuit (x y : ℚ) : ℚ := x - (1 / y)

-- Prove the main statement
theorem spadesuit_evaluation : spadesuit 3 (spadesuit 3 (3 / 2)) = 18 / 7 :=
by
  sorry

end spadesuit_evaluation_l405_40535


namespace carpet_area_proof_l405_40548

noncomputable def carpet_area (main_room_length_ft : ℕ) (main_room_width_ft : ℕ)
  (corridor_length_ft : ℕ) (corridor_width_ft : ℕ) (feet_per_yard : ℕ) : ℚ :=
  let main_room_length_yd := main_room_length_ft / feet_per_yard
  let main_room_width_yd := main_room_width_ft / feet_per_yard
  let corridor_length_yd := corridor_length_ft / feet_per_yard
  let corridor_width_yd := corridor_width_ft / feet_per_yard
  let main_room_area_yd2 := main_room_length_yd * main_room_width_yd
  let corridor_area_yd2 := corridor_length_yd * corridor_width_yd
  main_room_area_yd2 + corridor_area_yd2

theorem carpet_area_proof : carpet_area 15 12 10 3 3 = 23.33 :=
by
  -- Proof steps go here
  sorry

end carpet_area_proof_l405_40548


namespace vlad_taller_than_sister_l405_40542

def height_vlad_meters : ℝ := 1.905
def height_sister_cm : ℝ := 86.36

theorem vlad_taller_than_sister :
  (height_vlad_meters * 100 - height_sister_cm = 104.14) :=
by 
  sorry

end vlad_taller_than_sister_l405_40542


namespace bus_children_l405_40547

theorem bus_children (X : ℕ) (initial_children : ℕ) (got_on : ℕ) (total_children_after : ℕ) 
  (h1 : initial_children = 28) 
  (h2 : got_on = 82) 
  (h3 : total_children_after = 30) 
  (h4 : initial_children + got_on - X = total_children_after) : 
  got_on - X = 2 :=
by 
  -- h1, h2, h3, and h4 are conditions from the problem
  sorry

end bus_children_l405_40547


namespace original_cost_before_changes_l405_40595

variable (C : ℝ)

theorem original_cost_before_changes (h : 2 * C * 1.20 = 480) : C = 200 :=
by
  -- proof goes here
  sorry

end original_cost_before_changes_l405_40595


namespace problem_l405_40557

variables {b1 b2 b3 a1 a2 : ℤ}

-- Condition: five numbers -9, b1, b2, b3, -1 form a geometric sequence.
def is_geometric_seq (b1 b2 b3 : ℤ) : Prop :=
b1^2 = -9 * b2 ∧ b2^2 = b1 * b3 ∧ b1 * b3 = 9

-- Condition: four numbers -9, a1, a2, -3 form an arithmetic sequence.
def is_arithmetic_seq (a1 a2 : ℤ) : Prop :=
2 * a1 = -9 + a2 ∧ 2 * a2 = a1 - 3

-- Proof problem: prove that b2(a2 - a1) = -6
theorem problem (h_geom : is_geometric_seq b1 b2 b3) (h_arith : is_arithmetic_seq a1 a2) : 
  b2 * (a2 - a1) = -6 :=
by sorry

end problem_l405_40557


namespace unique_solution_l405_40537

theorem unique_solution : ∀ (x y z : ℕ), 
  x > 0 → y > 0 → z > 0 → 
  x^2 = 2 * (y + z) → 
  x^6 = y^6 + z^6 + 31 * (y^2 + z^2) → 
  (x, y, z) = (2, 1, 1) :=
by sorry

end unique_solution_l405_40537


namespace quadratic_inequality_solution_l405_40541

theorem quadratic_inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 1/3 → ax^2 + bx + 2 > 0) :
  a + b = -14 :=
sorry

end quadratic_inequality_solution_l405_40541


namespace area_not_covered_correct_l405_40536

-- Define the dimensions of the rectangle
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8

-- Define the side length of the square
def square_side_length : ℕ := 5

-- The area of the rectangle
def rectangle_area : ℕ := rectangle_length * rectangle_width

-- The area of the square
def square_area : ℕ := square_side_length * square_side_length

-- The area of the region not covered by the square
def area_not_covered : ℕ := rectangle_area - square_area

-- The theorem statement asserting the required area
theorem area_not_covered_correct : area_not_covered = 55 :=
by
  -- Proof is omitted
  sorry

end area_not_covered_correct_l405_40536


namespace find_value_l405_40531

theorem find_value (a b : ℝ) (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) : 2 * a^100 - 3 * b⁻¹ = 3 :=
by sorry

end find_value_l405_40531


namespace miranda_saved_per_month_l405_40501

-- Definition of the conditions and calculation in the problem
def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def months : ℕ := 3
def miranda_savings : ℕ := total_cost - sister_contribution
def saved_per_month : ℕ := miranda_savings / months

-- Theorem statement with the expected answer
theorem miranda_saved_per_month : saved_per_month = 70 :=
by
  sorry

end miranda_saved_per_month_l405_40501


namespace number_of_girls_l405_40512

theorem number_of_girls (boys girls : ℕ) (h1 : boys = 337) (h2 : girls = boys + 402) : girls = 739 := by
  sorry

end number_of_girls_l405_40512


namespace problem1_problem2_l405_40590

-- Problem 1: Prove that (-11) + 8 + (-4) = -7
theorem problem1 : (-11) + 8 + (-4) = -7 := by
  sorry

-- Problem 2: Prove that -1^2023 - |1 - 1/3| * (-3/2)^2 = -(5/2)
theorem problem2 : (-1 : ℚ)^2023 - abs (1 - 1/3) * (-3/2)^2 = -(5/2) := by
  sorry

end problem1_problem2_l405_40590


namespace marts_income_percentage_l405_40520

variable (J T M : ℝ)

theorem marts_income_percentage (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end marts_income_percentage_l405_40520


namespace chord_ratio_l405_40577

theorem chord_ratio {FQ HQ : ℝ} (h : EQ * FQ = GQ * HQ) (h_eq : EQ = 5) (h_gq : GQ = 12) : 
  FQ / HQ = 12 / 5 :=
by
  rw [h_eq, h_gq] at h
  sorry

end chord_ratio_l405_40577


namespace cubes_painted_on_one_side_l405_40576

def is_cube_painted_on_one_side (l w h : ℕ) (cube_size : ℕ) : ℕ :=
  let top_bottom := (l - 2) * (w - 2) * 2
  let front_back := (l - 2) * (h - 2) * 2
  let left_right := (w - 2) * (h - 2) * 2
  top_bottom + front_back + left_right

theorem cubes_painted_on_one_side (l w h cube_size : ℕ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) (h_cube_size : cube_size = 1) :
  is_cube_painted_on_one_side l w h cube_size = 22 :=
by
  sorry

end cubes_painted_on_one_side_l405_40576


namespace order_of_means_l405_40588

variables (a b : ℝ)
-- a and b are positive and unequal
axiom h1 : 0 < a
axiom h2 : 0 < b
axiom h3 : a ≠ b

-- Definitions of the means
noncomputable def AM : ℝ := (a + b) / 2
noncomputable def GM : ℝ := Real.sqrt (a * b)
noncomputable def HM : ℝ := (2 * a * b) / (a + b)
noncomputable def QM : ℝ := Real.sqrt ((a^2 + b^2) / 2)

-- The theorem to prove the order of the means
theorem order_of_means (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  QM a b > AM a b ∧ AM a b > GM a b ∧ GM a b > HM a b :=
sorry

end order_of_means_l405_40588


namespace value_of_k_l405_40562

theorem value_of_k : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := 
by 
  sorry

end value_of_k_l405_40562


namespace ones_digit_of_6_pow_52_l405_40543

theorem ones_digit_of_6_pow_52 : (6 ^ 52) % 10 = 6 := by
  -- we'll put the proof here
  sorry

end ones_digit_of_6_pow_52_l405_40543


namespace sets_satisfying_union_l405_40580

open Set

theorem sets_satisfying_union :
  {B : Set ℕ | {1, 2} ∪ B = {1, 2, 3}} = { {3}, {1, 3}, {2, 3}, {1, 2, 3} } :=
by
  sorry

end sets_satisfying_union_l405_40580


namespace max_distinct_dance_counts_l405_40569

theorem max_distinct_dance_counts (B G : ℕ) (hB : B = 29) (hG : G = 15) 
  (dance_with : ℕ → ℕ → Prop)
  (h_dance_limit : ∀ b g, dance_with b g → b ≤ B ∧ g ≤ G) :
  ∃ max_counts : ℕ, max_counts = 29 :=
by
  -- The statement of the theorem. Proof is omitted.
  sorry

end max_distinct_dance_counts_l405_40569


namespace Robinson_age_l405_40502

theorem Robinson_age (R : ℕ)
    (brother : ℕ := R + 2)
    (sister : ℕ := R + 6)
    (mother : ℕ := R + 20)
    (avg_age_yesterday : ℕ := 39)
    (total_age_yesterday : ℕ := 156)
    (eq : (R - 1) + (brother - 1) + (sister - 1) + (mother - 1) = total_age_yesterday) :
  R = 33 :=
by
  sorry

end Robinson_age_l405_40502


namespace addition_in_sets_l405_40596

theorem addition_in_sets (a b : ℤ) (hA : ∃ k : ℤ, a = 2 * k) (hB : ∃ k : ℤ, b = 2 * k + 1) : ∃ k : ℤ, a + b = 2 * k + 1 :=
by
  sorry

end addition_in_sets_l405_40596


namespace ratio_of_sizes_l405_40507

-- Defining Anna's size
def anna_size : ℕ := 2

-- Defining Becky's size as three times Anna's size
def becky_size : ℕ := 3 * anna_size

-- Defining Ginger's size
def ginger_size : ℕ := 8

-- Defining the goal statement
theorem ratio_of_sizes : (ginger_size : ℕ) / (becky_size : ℕ) = 4 / 3 :=
by
  sorry

end ratio_of_sizes_l405_40507


namespace expected_value_of_difference_l405_40553

noncomputable def expected_difference (num_days : ℕ) : ℝ :=
  let p_prime := 3 / 4
  let p_composite := 1 / 4
  let p_no_reroll := 2 / 3
  let expected_unsweetened_days := p_prime * p_no_reroll * num_days
  let expected_sweetened_days := p_composite * p_no_reroll * num_days
  expected_unsweetened_days - expected_sweetened_days

theorem expected_value_of_difference :
  expected_difference 365 = 121.667 := by
  sorry

end expected_value_of_difference_l405_40553


namespace integral_of_2x_minus_1_over_x_sq_l405_40516

theorem integral_of_2x_minus_1_over_x_sq:
  ∫ x in (1 : ℝ)..3, (2 * x - (1 / x^2)) = 26 / 3 := by
  sorry

end integral_of_2x_minus_1_over_x_sq_l405_40516


namespace problem_l405_40513

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log (1/8) / Real.log 2
noncomputable def c := Real.sqrt 2

theorem problem : c > a ∧ a > b := 
by
  sorry

end problem_l405_40513


namespace proposition_incorrect_l405_40551

theorem proposition_incorrect :
  ¬(∀ x : ℝ, x^2 + 3 * x + 1 > 0) :=
by
  sorry

end proposition_incorrect_l405_40551


namespace number_of_donut_selections_l405_40558

-- Definitions for the problem
def g : ℕ := sorry
def c : ℕ := sorry
def p : ℕ := sorry

-- Condition: Pat wants to buy four donuts from three types
def equation : Prop := g + c + p = 4

-- Question: Prove the number of different selections possible
theorem number_of_donut_selections : (∃ n, n = 15) := 
by 
  -- Use combinatorial method to establish this
  sorry

end number_of_donut_selections_l405_40558


namespace cone_curved_surface_area_at_5_seconds_l405_40571

theorem cone_curved_surface_area_at_5_seconds :
  let l := λ t : ℝ => 10 + 2 * t
  let r := λ t : ℝ => 5 + 1 * t
  let CSA := λ t : ℝ => Real.pi * r t * l t
  CSA 5 = 160 * Real.pi :=
by
  -- Definitions and calculations in the problem ensure this statement
  sorry

end cone_curved_surface_area_at_5_seconds_l405_40571


namespace symmetry_about_origin_l405_40584

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Define the function v based on f and g
def v (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x * |g x|

-- The theorem statement
theorem symmetry_about_origin (f g : ℝ → ℝ) (h_odd : is_odd f) (h_even : is_even g) : 
  ∀ x : ℝ, v f g (-x) = -v f g x := 
by
  sorry

end symmetry_about_origin_l405_40584


namespace complement_M_in_U_l405_40566

def M (x : ℝ) : Prop := 0 < x ∧ x < 2

def complement_M (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2

theorem complement_M_in_U (x : ℝ) : ¬ M x ↔ complement_M x :=
by sorry

end complement_M_in_U_l405_40566


namespace minimum_sum_dimensions_l405_40544

def is_product (a b c : ℕ) (v : ℕ) : Prop :=
  a * b * c = v

def sum (a b c : ℕ) : ℕ :=
  a + b + c

theorem minimum_sum_dimensions : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ is_product a b c 3003 ∧ sum a b c = 45 :=
by
  sorry

end minimum_sum_dimensions_l405_40544


namespace range_of_m_l405_40539

theorem range_of_m {m : ℝ} : 
  (¬ ∃ x : ℝ, (1 / 2 ≤ x ∧ x ≤ 2 ∧ x^2 - 2 * x - m ≤ 0)) → m < -1 :=
by
  sorry

end range_of_m_l405_40539


namespace cannot_be_sum_of_two_or_more_consecutive_integers_l405_40561

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

theorem cannot_be_sum_of_two_or_more_consecutive_integers (n : ℕ) :
  (¬∃ k m : ℕ, k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ is_power_of_two n :=
by
  sorry

end cannot_be_sum_of_two_or_more_consecutive_integers_l405_40561


namespace house_selling_price_l405_40565

theorem house_selling_price
  (original_price : ℝ := 80000)
  (profit_rate : ℝ := 0.20)
  (commission_rate : ℝ := 0.05):
  original_price + (original_price * profit_rate) + (original_price * commission_rate) = 100000 := by
  sorry

end house_selling_price_l405_40565


namespace value_of_g_at_2_l405_40508

def g (x : ℝ) : ℝ := x^2 - 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  -- proof goes here
  sorry

end value_of_g_at_2_l405_40508


namespace proof_problem_l405_40570

-- Given condition
variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b)
variable (h3 : Real.log a + Real.log (b ^ 2) ≥ 2 * a + (b ^ 2) / 2 - 2)

-- Proof statement
theorem proof_problem : a - 2 * b = 1/2 - 2 * Real.sqrt 2 :=
by
  sorry

end proof_problem_l405_40570


namespace guiding_normal_vector_l405_40591

noncomputable def ellipsoid (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 - 6

def point_M0 : ℝ × ℝ × ℝ := (1, -1, 1)

def normal_vector (x y z : ℝ) : ℝ × ℝ × ℝ := (
  2 * x,
  4 * y,
  6 * z
)

theorem guiding_normal_vector : normal_vector 1 (-1) 1 = (2, -4, 6) :=
by
  sorry

end guiding_normal_vector_l405_40591


namespace bricks_needed_for_courtyard_l405_40578

noncomputable def total_bricks_required (courtyard_length courtyard_width : ℝ)
  (brick_length_cm brick_width_cm : ℝ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width
  let brick_length := brick_length_cm / 100
  let brick_width := brick_width_cm / 100
  let brick_area := brick_length * brick_width
  courtyard_area / brick_area

theorem bricks_needed_for_courtyard :
  total_bricks_required 35 24 15 8 = 70000 := by
  sorry

end bricks_needed_for_courtyard_l405_40578


namespace angle_C_in_triangle_l405_40563

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end angle_C_in_triangle_l405_40563


namespace percentage_y_more_than_z_l405_40564

theorem percentage_y_more_than_z :
  ∀ (P y x k : ℕ),
    P = 200 →
    740 = x + y + P →
    x = (5 / 4) * y →
    y = P * (1 + k / 100) →
    k = 20 :=
by
  sorry

end percentage_y_more_than_z_l405_40564


namespace Peggy_needs_to_add_stamps_l405_40585

theorem Peggy_needs_to_add_stamps :
  ∀ (Peggy_stamps Bert_stamps Ernie_stamps : ℕ),
  Peggy_stamps = 75 →
  Ernie_stamps = 3 * Peggy_stamps →
  Bert_stamps = 4 * Ernie_stamps →
  Bert_stamps - Peggy_stamps = 825 :=
by
  intros Peggy_stamps Bert_stamps Ernie_stamps hPeggy hErnie hBert
  sorry

end Peggy_needs_to_add_stamps_l405_40585


namespace converse_statement_l405_40527

theorem converse_statement (a : ℝ) : (a > 2018 → a > 2017) ↔ (a > 2017 → a > 2018) :=
by
  sorry

end converse_statement_l405_40527


namespace bus_stop_time_l405_40549

noncomputable def time_stopped_per_hour (excl_speed incl_speed : ℕ) : ℕ :=
  60 * (excl_speed - incl_speed) / excl_speed

theorem bus_stop_time:
  time_stopped_per_hour 54 36 = 20 :=
by
  sorry

end bus_stop_time_l405_40549


namespace quadratic_two_distinct_real_roots_l405_40504

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (k - 1 ≠ 0 ∧ 8 - 4 * k > 0) ↔ (k < 2 ∧ k ≠ 1) := 
by
  sorry

end quadratic_two_distinct_real_roots_l405_40504


namespace jerry_feathers_count_l405_40509

noncomputable def hawk_feathers : ℕ := 6
noncomputable def eagle_feathers : ℕ := 17 * hawk_feathers
noncomputable def total_feathers : ℕ := hawk_feathers + eagle_feathers
noncomputable def remaining_feathers_after_sister : ℕ := total_feathers - 10
noncomputable def jerry_feathers_left : ℕ := remaining_feathers_after_sister / 2

theorem jerry_feathers_count : jerry_feathers_left = 49 :=
  by
  sorry

end jerry_feathers_count_l405_40509


namespace students_like_both_l405_40545

theorem students_like_both {total students_apple_pie students_chocolate_cake students_none students_at_least_one students_both : ℕ} 
  (h_total : total = 50)
  (h_apple : students_apple_pie = 22)
  (h_chocolate : students_chocolate_cake = 20)
  (h_none : students_none = 17)
  (h_least_one : students_at_least_one = total - students_none)
  (h_union : students_at_least_one = students_apple_pie + students_chocolate_cake - students_both) :
  students_both = 9 :=
by
  sorry

end students_like_both_l405_40545


namespace complex_fraction_value_l405_40505

theorem complex_fraction_value (a b : ℝ) (h : (i - 2) / (1 + i) = a + b * i) : a + b = 1 :=
by
  sorry

end complex_fraction_value_l405_40505


namespace triangle_solid_revolution_correct_l405_40581

noncomputable def triangle_solid_revolution (t : ℝ) (alpha beta gamma : ℝ) (longest_side : string) : ℝ × ℝ :=
  let pi := Real.pi;
  let sin := Real.sin;
  let cos := Real.cos;
  let sqrt := Real.sqrt;
  let to_rad (x : ℝ) : ℝ := x * pi / 180;
  let alpha_rad := to_rad alpha;
  let beta_rad := to_rad beta;
  let gamma_rad := to_rad gamma;
  let a := sqrt (2 * t * sin alpha_rad / (sin beta_rad * sin gamma_rad));
  let b := sqrt (2 * t * sin beta_rad / (sin gamma_rad * sin alpha_rad));
  let m_c := sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  let F := 2 * pi * t * cos ((alpha_rad - beta_rad) / 2) / sin (gamma_rad / 2);
  let K := 2 * pi / 3 * t * sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  (F, K)

theorem triangle_solid_revolution_correct :
  triangle_solid_revolution 80.362 (39 + 34/60 + 30/3600) (60 : ℝ) (80 + 25/60 + 30/3600) "c" = (769.3, 1595.3) :=
sorry

end triangle_solid_revolution_correct_l405_40581


namespace minutes_before_noon_l405_40598

theorem minutes_before_noon
    (x : ℕ)
    (h1 : 20 <= x)
    (h2 : 180 - (x - 20) = 3 * (x - 20)) :
    x = 65 := by
  sorry

end minutes_before_noon_l405_40598


namespace original_weight_calculation_l405_40525

-- Conditions
variable (postProcessingWeight : ℝ) (originalWeight : ℝ)
variable (lostPercentage : ℝ)

-- Problem Statement
theorem original_weight_calculation
  (h1 : postProcessingWeight = 240)
  (h2 : lostPercentage = 0.40) :
  originalWeight = 400 :=
sorry

end original_weight_calculation_l405_40525


namespace cannot_determine_orange_groups_l405_40586

-- Definitions of the conditions
def oranges := 87
def bananas := 290
def bananaGroups := 2
def bananasPerGroup := 145

-- Lean statement asserting that the number of groups of oranges 
-- cannot be determined from the given conditions
theorem cannot_determine_orange_groups:
  ∀ (number_of_oranges_per_group : ℕ), 
  (bananasPerGroup * bananaGroups = bananas) ∧ (oranges = 87) → 
  ¬(∃ (number_of_orange_groups : ℕ), oranges = number_of_oranges_per_group * number_of_orange_groups) :=
by
  sorry -- Since we are not required to provide the proof here

end cannot_determine_orange_groups_l405_40586


namespace positive_integer_conditions_l405_40559

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) :
  (∃ q : ℕ, q > 0 ∧ (5 * p + 36) = q * (2 * p - 9)) ↔ (p = 5 ∨ p = 6 ∨ p = 9 ∨ p = 18) :=
by sorry

end positive_integer_conditions_l405_40559


namespace smallest_b_l405_40597

theorem smallest_b (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : a * b * c = 360) : b = 3 :=
sorry

end smallest_b_l405_40597


namespace fixed_point_l405_40532

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 2

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = 3 :=
by
  unfold f
  sorry

end fixed_point_l405_40532


namespace weight_of_fourth_dog_l405_40524

theorem weight_of_fourth_dog (y x : ℝ) : 
  (25 + 31 + 35 + x) / 4 = (25 + 31 + 35 + x + y) / 5 → 
  x = -91 - 5 * y :=
by
  sorry

end weight_of_fourth_dog_l405_40524


namespace johns_donation_l405_40540

theorem johns_donation (A : ℝ) (T : ℝ) (J : ℝ) (h1 : A + 0.5 * A = 75) (h2 : T = 3 * A) 
                       (h3 : (T + J) / 4 = 75) : J = 150 := by
  sorry

end johns_donation_l405_40540


namespace find_MN_sum_l405_40521

noncomputable def M : ℝ := sorry -- Placeholder for the actual non-zero solution M
noncomputable def N : ℝ := M ^ 2

theorem find_MN_sum :
  (M^2 = N) ∧ (Real.log N / Real.log M = Real.log M / Real.log N) ∧ (M ≠ N) ∧ (M ≠ 1) ∧ (N ≠ 1) → (M + N = 6) :=
by
  intros h
  exact sorry -- Will be replaced by the actual proof


end find_MN_sum_l405_40521


namespace product_of_solutions_l405_40567

theorem product_of_solutions (x : ℝ) (h : |(18 / x) - 6| = 3) : 2 * 6 = 12 :=
by
  sorry

end product_of_solutions_l405_40567


namespace new_boarders_joined_l405_40552

theorem new_boarders_joined (initial_boarders new_boarders initial_day_students total_boarders total_day_students: ℕ)
  (h1: initial_boarders = 60)
  (h2: initial_day_students = 150)
  (h3: total_boarders = initial_boarders + new_boarders)
  (h4: total_day_students = initial_day_students)
  (h5: 2 * initial_day_students = 5 * initial_boarders)
  (h6: 2 * total_boarders = total_day_students) :
  new_boarders = 15 :=
by
  sorry

end new_boarders_joined_l405_40552


namespace maximize_GDP_investment_l405_40554

def invest_A_B_max_GDP : Prop :=
  ∃ (A B : ℝ), 
  A + B ≤ 30 ∧
  20000 * A + 40000 * B ≤ 1000000 ∧
  24 * A + 32 * B ≥ 800 ∧
  A = 20 ∧ B = 10

theorem maximize_GDP_investment : invest_A_B_max_GDP :=
by
  sorry

end maximize_GDP_investment_l405_40554


namespace parabola_relationship_l405_40503

theorem parabola_relationship (a : ℝ) (h : a < 0) :
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  y1 < y3 ∧ y3 < y2 :=
by
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  sorry

end parabola_relationship_l405_40503


namespace maximum_value_of_z_l405_40511

theorem maximum_value_of_z :
  ∃ x y : ℝ, (x - y ≥ 0) ∧ (x + y ≤ 2) ∧ (y ≥ 0) ∧ (∀ u v : ℝ, (u - v ≥ 0) ∧ (u + v ≤ 2) ∧ (v ≥ 0) → 3 * u - v ≤ 6) :=
by
  sorry

end maximum_value_of_z_l405_40511


namespace completing_the_square_solution_correct_l405_40538

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l405_40538


namespace circle_radius_five_iff_l405_40530

noncomputable def circle_eq_radius (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

def is_circle_with_radius (r : ℝ) (x y : ℝ) (k : ℝ) : Prop :=
  circle_eq_radius x y k ↔ r = 5 ∧ k = 5

theorem circle_radius_five_iff (k : ℝ) :
  (∃ x y : ℝ, circle_eq_radius x y k) ↔ k = 5 :=
sorry

end circle_radius_five_iff_l405_40530


namespace mutually_exclusive_pairs_l405_40514

/-- Define the events for shooting rings and drawing balls. -/
inductive ShootEvent
| hits_7th_ring : ShootEvent
| hits_8th_ring : ShootEvent

inductive PersonEvent
| at_least_one_hits : PersonEvent
| A_hits_B_does_not : PersonEvent

inductive BallEvent
| at_least_one_black : BallEvent
| both_red : BallEvent
| no_black : BallEvent
| one_red : BallEvent

/-- Define mutually exclusive events. -/
def mutually_exclusive (e1 e2 : Prop) : Prop := e1 ∧ e2 → False

/-- Prove the pairs of events that are mutually exclusive. -/
theorem mutually_exclusive_pairs :
  mutually_exclusive (ShootEvent.hits_7th_ring = ShootEvent.hits_7th_ring) (ShootEvent.hits_8th_ring = ShootEvent.hits_8th_ring) ∧
  ¬mutually_exclusive (PersonEvent.at_least_one_hits = PersonEvent.at_least_one_hits) (PersonEvent.A_hits_B_does_not = PersonEvent.A_hits_B_does_not) ∧
  mutually_exclusive (BallEvent.at_least_one_black = BallEvent.at_least_one_black) (BallEvent.both_red = BallEvent.both_red) ∧
  mutually_exclusive (BallEvent.no_black = BallEvent.no_black) (BallEvent.one_red = BallEvent.one_red) :=
by {
  sorry
}

end mutually_exclusive_pairs_l405_40514


namespace convert_to_cylindrical_l405_40519

noncomputable def cylindricalCoordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / r)
  if y / r < 0 then (r, 2 * Real.pi - θ, z) else (r, θ, z)

theorem convert_to_cylindrical :
  cylindricalCoordinates 3 (-3 * Real.sqrt 3) 4 = (6, 5 * Real.pi / 3, 4) :=
by
  sorry

end convert_to_cylindrical_l405_40519


namespace age_of_other_man_l405_40592

-- Definitions of the given conditions
def average_age_increase (avg_men : ℕ → ℝ) (men_removed women_avg : ℕ) : Prop :=
  avg_men 8 + 2 = avg_men 6 + women_avg / 2

def one_man_age : ℕ := 24
def women_avg : ℕ := 30

-- Statement of the problem to prove
theorem age_of_other_man (avg_men : ℕ → ℝ) (other_man : ℕ) :
  average_age_increase avg_men 24 women_avg →
  other_man = 20 :=
sorry

end age_of_other_man_l405_40592


namespace mean_score_is_76_l405_40534

noncomputable def mean_stddev_problem := 
  ∃ (M SD : ℝ), (M - 2 * SD = 60) ∧ (M + 3 * SD = 100) ∧ (M = 76)

theorem mean_score_is_76 : mean_stddev_problem :=
sorry

end mean_score_is_76_l405_40534


namespace files_deleted_l405_40515

theorem files_deleted 
  (orig_files : ℕ) (final_files : ℕ) (deleted_files : ℕ) 
  (h_orig : orig_files = 24) 
  (h_final : final_files = 21) : 
  deleted_files = orig_files - final_files :=
by
  rw [h_orig, h_final]
  sorry

end files_deleted_l405_40515


namespace annual_income_A_l405_40594

variable (A B C : ℝ)
variable (monthly_income_C : C = 17000)
variable (monthly_income_B : B = C + 0.12 * C)
variable (ratio_A_to_B : A / B = 5 / 2)

theorem annual_income_A (A B C : ℝ) 
    (hC : C = 17000) 
    (hB : B = C + 0.12 * C) 
    (hR : A / B = 5 / 2) : 
    A * 12 = 571200 :=
by
  sorry

end annual_income_A_l405_40594


namespace find_some_number_l405_40579

theorem find_some_number (some_number : ℕ) : 
  ( ∃ n:ℕ, n = 54 ∧ (n / 18) * (n / some_number) = 1 ) ∧ some_number = 162 :=
by {
  sorry
}

end find_some_number_l405_40579


namespace least_positive_integer_l405_40555
  
theorem least_positive_integer 
  (x : ℕ) (d n : ℕ) (p : ℕ) 
  (h_eq : x = 10^p * d + n) 
  (h_ratio : n = x / 17) 
  (h_cond1 : 1 ≤ d) 
  (h_cond2 : d ≤ 9)
  (h_nonzero : n > 0) : 
  x = 10625 :=
by
  sorry

end least_positive_integer_l405_40555


namespace speed_of_train_l405_40587

-- Conditions
def train_length : ℝ := 180
def total_length : ℝ := 195
def time_cross_bridge : ℝ := 30

-- Conversion factor for units (1 m/s = 3.6 km/hr)
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem speed_of_train : 
  (total_length - train_length) / time_cross_bridge * conversion_factor = 23.4 :=
sorry

end speed_of_train_l405_40587


namespace intersection_of_A_and_B_l405_40560

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end intersection_of_A_and_B_l405_40560
