import Mathlib

namespace NUMINAMATH_GPT_total_course_selection_schemes_l1929_192973

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end NUMINAMATH_GPT_total_course_selection_schemes_l1929_192973


namespace NUMINAMATH_GPT_john_saves_money_l1929_192989

def original_spending (coffees_per_day: ℕ) (price_per_coffee: ℕ) : ℕ :=
  coffees_per_day * price_per_coffee

def new_price (original_price: ℕ) (increase_percentage: ℕ) : ℕ :=
  original_price + (original_price * increase_percentage / 100)

def new_coffees_per_day (original_coffees_per_day: ℕ) (reduction_fraction: ℕ) : ℕ :=
  original_coffees_per_day / reduction_fraction

def current_spending (new_coffees_per_day: ℕ) (new_price_per_coffee: ℕ) : ℕ :=
  new_coffees_per_day * new_price_per_coffee

theorem john_saves_money
  (coffees_per_day : ℕ := 4)
  (price_per_coffee : ℕ := 2)
  (increase_percentage : ℕ := 50)
  (reduction_fraction : ℕ := 2) :
  original_spending coffees_per_day price_per_coffee
  - current_spending (new_coffees_per_day coffees_per_day reduction_fraction)
                     (new_price price_per_coffee increase_percentage)
  = 2 := by
{
  sorry
}

end NUMINAMATH_GPT_john_saves_money_l1929_192989


namespace NUMINAMATH_GPT_triple_root_possible_values_l1929_192987

-- Definitions and conditions
def polynomial (x : ℤ) (b3 b2 b1 : ℤ) := x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 24

-- The proof problem
theorem triple_root_possible_values 
  (r b3 b2 b1 : ℤ)
  (h_triple_root : polynomial r b3 b2 b1 = (x * (x - 1) * (x - 2)) * (x - r) ) :
  r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 :=
by
  sorry

end NUMINAMATH_GPT_triple_root_possible_values_l1929_192987


namespace NUMINAMATH_GPT_smallest_t_for_given_roots_l1929_192998

-- Define the polynomial with integer coefficients and specific roots
def poly (x : ℝ) : ℝ := (x + 3) * (x - 4) * (x - 6) * (2 * x - 1)

-- Define the main theorem statement
theorem smallest_t_for_given_roots :
  ∃ (t : ℤ), 0 < t ∧ t = 72 := by
  -- polynomial expansion skipped, proof will come here
  sorry

end NUMINAMATH_GPT_smallest_t_for_given_roots_l1929_192998


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1929_192995

theorem necessary_and_sufficient_condition (x : ℝ) (h : x > 0) : (x + 1/x ≥ 2) ↔ (x > 0) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1929_192995


namespace NUMINAMATH_GPT_chosen_number_l1929_192922

theorem chosen_number (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 :=
sorry

end NUMINAMATH_GPT_chosen_number_l1929_192922


namespace NUMINAMATH_GPT_central_angle_l1929_192986

-- Definition: percentage corresponds to central angle
def percentage_equal_ratio (P : ℝ) (θ : ℝ) : Prop :=
  P = θ / 360

-- Theorem statement: Given that P = θ / 360, we want to prove θ = 360 * P
theorem central_angle (P θ : ℝ) (h : percentage_equal_ratio P θ) : θ = 360 * P :=
sorry

end NUMINAMATH_GPT_central_angle_l1929_192986


namespace NUMINAMATH_GPT_only_B_is_like_terms_l1929_192912

def is_like_terms (terms : List (String × String)) : List Bool :=
  let like_term_checker := fun (term1 term2 : String) =>
    -- The function to check if two terms are like terms
    sorry
  terms.map (fun (term1, term2) => like_term_checker term1 term2)

theorem only_B_is_like_terms :
  is_like_terms [("−2x^3", "−3x^2"), ("−(1/4)ab", "18ba"), ("a^2b", "−ab^2"), ("4m", "6mn")] =
  [false, true, false, false] :=
by
  sorry

end NUMINAMATH_GPT_only_B_is_like_terms_l1929_192912


namespace NUMINAMATH_GPT_points_satisfy_equation_l1929_192908

theorem points_satisfy_equation (x y : ℝ) : 
  (2 * x^2 + 3 * x * y + y^2 + x = 1) ↔ (y = -x - 1) ∨ (y = -2 * x + 1) := by
  sorry

end NUMINAMATH_GPT_points_satisfy_equation_l1929_192908


namespace NUMINAMATH_GPT_marbles_end_of_day_l1929_192969

theorem marbles_end_of_day :
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54 :=
by
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  show initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54
  sorry

end NUMINAMATH_GPT_marbles_end_of_day_l1929_192969


namespace NUMINAMATH_GPT_find_intersection_point_l1929_192970

theorem find_intersection_point :
  ∃ (x y z : ℝ), 
    ((∃ t : ℝ, x = 1 + 2 * t ∧ y = 1 - t ∧ z = -2 + 3 * t) ∧ 
    (4 * x + 2 * y - z - 11 = 0)) ∧ 
    (x = 3 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_point_l1929_192970


namespace NUMINAMATH_GPT_ratio_of_container_volumes_l1929_192917

-- Define the volumes of the first and second containers.
variables (A B : ℝ )

-- Hypotheses based on the problem conditions
-- First container is 4/5 full
variable (h1 : A * 4 / 5 = B * 2 / 3)

-- The statement to prove
theorem ratio_of_container_volumes : A / B = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_container_volumes_l1929_192917


namespace NUMINAMATH_GPT_find_m_l1929_192951

theorem find_m (x y m : ℤ) (h1 : x = 3) (h2 : y = 1) (h3 : x - m * y = 1) : m = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_m_l1929_192951


namespace NUMINAMATH_GPT_original_number_l1929_192905

variable (n : ℝ)

theorem original_number :
  (2 * (n + 3)^2 - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l1929_192905


namespace NUMINAMATH_GPT_min_value_of_expression_l1929_192985

noncomputable def min_expression_value (y : ℝ) (hy : y > 2) : ℝ :=
  (y^2 + y + 1) / Real.sqrt (y - 2)

theorem min_value_of_expression (y : ℝ) (hy : y > 2) :
  min_expression_value y hy = 3 * Real.sqrt 35 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1929_192985


namespace NUMINAMATH_GPT_stream_speed_l1929_192959

variable (v : ℝ)

def effective_speed_downstream (v : ℝ) : ℝ := 7.5 + v
def effective_speed_upstream (v : ℝ) : ℝ := 7.5 - v 

theorem stream_speed : (7.5 - v) / (7.5 + v) = 1 / 2 → v = 2.5 :=
by
  intro h
  -- Proof will be resolved here
  sorry

end NUMINAMATH_GPT_stream_speed_l1929_192959


namespace NUMINAMATH_GPT_carriages_people_equation_l1929_192988

theorem carriages_people_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end NUMINAMATH_GPT_carriages_people_equation_l1929_192988


namespace NUMINAMATH_GPT_max_subjects_per_teacher_l1929_192940

theorem max_subjects_per_teacher (maths physics chemistry : ℕ) (min_teachers : ℕ)
  (h_math : maths = 6) (h_physics : physics = 5) (h_chemistry : chemistry = 5) (h_min_teachers : min_teachers = 4) :
  (maths + physics + chemistry) / min_teachers = 4 :=
by
  -- the proof will be here
  sorry

end NUMINAMATH_GPT_max_subjects_per_teacher_l1929_192940


namespace NUMINAMATH_GPT_jenny_hours_left_l1929_192904

theorem jenny_hours_left 
    (h_research : ℕ := 10)
    (h_proposal : ℕ := 2)
    (h_visual_aids : ℕ := 5)
    (h_editing : ℕ := 3)
    (h_total : ℕ := 25) :
    h_total - (h_research + h_proposal + h_visual_aids + h_editing) = 5 := by
  sorry

end NUMINAMATH_GPT_jenny_hours_left_l1929_192904


namespace NUMINAMATH_GPT_solve_for_a_l1929_192992

-- Define the sets M and N as given in the problem
def M : Set ℝ := {x : ℝ | x^2 + 6 * x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x : ℝ | x * a - 3 = 0}

-- Define the proof statement
theorem solve_for_a (a : ℝ) : (N a ⊆ M) ↔ (a = 0 ∨ a = 3/2 ∨ a = -3/8) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_solve_for_a_l1929_192992


namespace NUMINAMATH_GPT_solve_for_x_l1929_192976

noncomputable def solve_equation (x : ℝ) : Prop := 
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = 3 * x / (3 * x - 2) ∧ x ≠ 2 / 3

theorem solve_for_x (x : ℝ) (h : solve_equation x) : x = (Real.sqrt 6) / 3 ∨ x = - (Real.sqrt 6) / 3 := 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1929_192976


namespace NUMINAMATH_GPT_inner_hexagon_area_l1929_192999

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

end NUMINAMATH_GPT_inner_hexagon_area_l1929_192999


namespace NUMINAMATH_GPT_mean_books_read_l1929_192909

theorem mean_books_read :
  let readers1 := 4
  let books1 := 3
  let readers2 := 5
  let books2 := 5
  let readers3 := 2
  let books3 := 7
  let readers4 := 1
  let books4 := 10
  let total_readers := readers1 + readers2 + readers3 + readers4
  let total_books := (readers1 * books1) + (readers2 * books2) + (readers3 * books3) + (readers4 * books4)
  let mean_books := total_books / total_readers
  mean_books = 5.0833 :=
by
  sorry

end NUMINAMATH_GPT_mean_books_read_l1929_192909


namespace NUMINAMATH_GPT_abs_sum_sequence_l1929_192925

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem abs_sum_sequence (h : ∀ n, S n = n^2 - 4 * n) :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_sequence_l1929_192925


namespace NUMINAMATH_GPT_largest_of_w_l1929_192991

variable {x y z w : ℝ}

namespace MathProof

theorem largest_of_w
  (h1 : x + 3 = y - 1)
  (h2 : x + 3 = z + 5)
  (h3 : x + 3 = w - 2) :  
  w > y ∧ w > x ∧ w > z :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_largest_of_w_l1929_192991


namespace NUMINAMATH_GPT_trig_identity_proof_l1929_192965

noncomputable def value_expr : ℝ :=
  (2 * Real.cos (10 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.sin (70 * Real.pi / 180)

theorem trig_identity_proof : value_expr = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l1929_192965


namespace NUMINAMATH_GPT_seq_period_3_l1929_192945

def seq (a : ℕ → ℚ) := ∀ n, 
  (0 ≤ a n ∧ a n < 1) ∧ (
  (0 ≤ a n ∧ a n < 1/2 → a (n+1) = 2 * a n) ∧ 
  (1/2 ≤ a n ∧ a n < 1 → a (n+1) = 2 * a n - 1))

theorem seq_period_3 (a : ℕ → ℚ) (h : seq a) (h1 : a 1 = 6 / 7) : 
  a 2016 = 3 / 7 := 
sorry

end NUMINAMATH_GPT_seq_period_3_l1929_192945


namespace NUMINAMATH_GPT_bathroom_visits_time_l1929_192931

variable (t_8 : ℕ) (n8 : ℕ) (n6 : ℕ)

theorem bathroom_visits_time (h1 : t_8 = 20) (h2 : n8 = 8) (h3 : n6 = 6) :
  (t_8 / n8) * n6 = 15 := by
  sorry

end NUMINAMATH_GPT_bathroom_visits_time_l1929_192931


namespace NUMINAMATH_GPT_find_a7_a8_l1929_192903

variable {R : Type*} [LinearOrderedField R]
variable {a : ℕ → R}

-- Conditions
def cond1 : a 1 + a 2 = 40 := sorry
def cond2 : a 3 + a 4 = 60 := sorry

-- Goal 
theorem find_a7_a8 : a 7 + a 8 = 135 := 
by 
  -- provide the actual proof here
  sorry

end NUMINAMATH_GPT_find_a7_a8_l1929_192903


namespace NUMINAMATH_GPT_cos_beta_l1929_192977

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = 3 / 5)
variable (h2 : Real.cos (α + β) = 5 / 13)

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 3 / 5) (h2 : Real.cos (α + β) = 5 / 13) : 
  Real.cos β = 56 / 65 := by
  sorry

end NUMINAMATH_GPT_cos_beta_l1929_192977


namespace NUMINAMATH_GPT_inequality_proof_l1929_192979

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a + b + c ≤ (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ∧ 
    (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ≤ (a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1929_192979


namespace NUMINAMATH_GPT_solve_for_a_l1929_192930

theorem solve_for_a (a : ℝ) : 
  (2 * a + 16 + 3 * a - 8) / 2 = 69 → a = 26 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1929_192930


namespace NUMINAMATH_GPT_total_cost_of_digging_well_l1929_192947

noncomputable def cost_of_digging (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := Real.pi * (radius^2) * depth
  volume * cost_per_cubic_meter

theorem total_cost_of_digging_well :
  cost_of_digging 14 3 15 = 1484.4 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_digging_well_l1929_192947


namespace NUMINAMATH_GPT_correct_operation_l1929_192915

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1929_192915


namespace NUMINAMATH_GPT_B_pow_2024_l1929_192993

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), 0, -Real.sin (Real.pi / 4)],
    ![0, 1, 0],
    ![Real.sin (Real.pi / 4), 0, Real.cos (Real.pi / 4)]
  ]

theorem B_pow_2024 :
  B ^ 2024 = ![
    ![-1, 0, 0],
    ![0, 1, 0],
    ![0, 0, -1]
  ] :=
by
  sorry

end NUMINAMATH_GPT_B_pow_2024_l1929_192993


namespace NUMINAMATH_GPT_parabola_coefficients_l1929_192927

theorem parabola_coefficients
    (vertex : (ℝ × ℝ))
    (passes_through : (ℝ × ℝ))
    (vertical_axis_of_symmetry : Prop)
    (hv : vertex = (2, -3))
    (hp : passes_through = (0, 1))
    (has_vertical_axis : vertical_axis_of_symmetry) :
    ∃ a b c : ℝ, ∀ x : ℝ, (x = 0 → (a * x^2 + b * x + c = 1)) ∧ (x = 2 → (a * x^2 + b * x + c = -3)) := sorry

end NUMINAMATH_GPT_parabola_coefficients_l1929_192927


namespace NUMINAMATH_GPT_length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l1929_192906

noncomputable def spring_length (x : ℝ) : ℝ :=
  2 * x + 18

-- Problem (1)
theorem length_at_4kg : (spring_length 4) = 26 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (2)
theorem length_increases_by_2 : ∀ (x y : ℝ), y = x + 1 → (spring_length y) = (spring_length x) + 2 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (3)
theorem relationship_linear : ∃ (k b : ℝ), (∀ x, spring_length x = k * x + b) ∧ k = 2 ∧ b = 18 :=
  by
    -- The complete proof is omitted.
    sorry

-- Problem (4)
theorem length_at_12kg : (spring_length 12) = 42 :=
  by
    -- The complete proof is omitted.
    sorry

end NUMINAMATH_GPT_length_at_4kg_length_increases_by_2_relationship_linear_length_at_12kg_l1929_192906


namespace NUMINAMATH_GPT_ladder_geometric_sequence_solution_l1929_192907

-- A sequence {aₙ} is a 3rd-order ladder geometric sequence given by a_{n+3}^2 = a_n * a_{n+6} for any positive integer n
def ladder_geometric_3rd_order (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 3) ^ 2 = a n * a (n + 6)

-- Initial conditions
def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 4 = 2

-- Main theorem to be proven in Lean 4
theorem ladder_geometric_sequence_solution :
  ∃ a : ℕ → ℝ, ladder_geometric_3rd_order a ∧ initial_conditions a ∧ a 10 = 8 :=
by
  sorry

end NUMINAMATH_GPT_ladder_geometric_sequence_solution_l1929_192907


namespace NUMINAMATH_GPT_roots_relationship_l1929_192935

variable {a b c : ℝ} (h : a ≠ 0)

theorem roots_relationship (x y : ℝ) :
  (x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) →
  (y = (-b + Real.sqrt (b^2 - 4*a*c)) / 2 ∨ y = (-b - Real.sqrt (b^2 - 4*a*c)) / 2) →
  (x = y / a) :=
by
  sorry

end NUMINAMATH_GPT_roots_relationship_l1929_192935


namespace NUMINAMATH_GPT_clock_correction_time_l1929_192974

theorem clock_correction_time :
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  correction = 138.75 :=
by
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  have : correction = 138.75 := sorry
  exact this

end NUMINAMATH_GPT_clock_correction_time_l1929_192974


namespace NUMINAMATH_GPT_capacity_of_first_bucket_is_3_l1929_192972

variable (C : ℝ)

theorem capacity_of_first_bucket_is_3 
  (h1 : 48 / C = 48 / 3 - 4) : 
  C = 3 := 
  sorry

end NUMINAMATH_GPT_capacity_of_first_bucket_is_3_l1929_192972


namespace NUMINAMATH_GPT_total_people_l1929_192913

-- Definitions of the given conditions
variable (I N B Ne T : ℕ)

-- These variables represent the given conditions
axiom h1 : I = 25
axiom h2 : N = 23
axiom h3 : B = 21
axiom h4 : Ne = 23

-- The theorem we want to prove
theorem total_people : T = 50 :=
by {
  sorry -- We denote the skipping of proof details.
}

end NUMINAMATH_GPT_total_people_l1929_192913


namespace NUMINAMATH_GPT_percentage_of_original_solution_l1929_192957

-- Define the problem and conditions
variable (P : ℝ)
variable (h1 : (0.5 * P + 0.5 * 60) = 55)

-- The theorem to prove
theorem percentage_of_original_solution : P = 50 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_percentage_of_original_solution_l1929_192957


namespace NUMINAMATH_GPT_solve_for_x_and_y_l1929_192968

theorem solve_for_x_and_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_and_y_l1929_192968


namespace NUMINAMATH_GPT_kamals_salary_change_l1929_192939

theorem kamals_salary_change : 
  ∀ (S : ℝ), ((S * 0.5 * 1.3 * 0.8 - S) / S) * 100 = -48 :=
by
  intro S
  sorry

end NUMINAMATH_GPT_kamals_salary_change_l1929_192939


namespace NUMINAMATH_GPT_shaded_total_area_l1929_192980

theorem shaded_total_area:
  ∀ (r₁ r₂ r₃ : ℝ),
  π * r₁ ^ 2 = 100 * π →
  r₂ = r₁ / 2 →
  r₃ = r₂ / 2 →
  (1 / 2) * (π * r₁ ^ 2) + (1 / 2) * (π * r₂ ^ 2) + (1 / 2) * (π * r₃ ^ 2) = 65.625 * π :=
by
  intro r₁ r₂ r₃ h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_shaded_total_area_l1929_192980


namespace NUMINAMATH_GPT_option_a_is_correct_l1929_192990

theorem option_a_is_correct (a b : ℝ) :
  (a - b) * (-a - b) = b^2 - a^2 :=
sorry

end NUMINAMATH_GPT_option_a_is_correct_l1929_192990


namespace NUMINAMATH_GPT_cube_volume_l1929_192954

theorem cube_volume (A : ℝ) (h : A = 24) : 
  ∃ V : ℝ, V = 8 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_l1929_192954


namespace NUMINAMATH_GPT_Mr_A_Mrs_A_are_normal_l1929_192942

def is_knight (person : Type) : Prop := sorry
def is_liar (person : Type) : Prop := sorry
def is_normal (person : Type) : Prop := sorry

variable (Mr_A Mrs_A : Type)

axiom Mr_A_statement : is_normal Mrs_A → False
axiom Mrs_A_statement : is_normal Mr_A → False

theorem Mr_A_Mrs_A_are_normal :
  is_normal Mr_A ∧ is_normal Mrs_A :=
sorry

end NUMINAMATH_GPT_Mr_A_Mrs_A_are_normal_l1929_192942


namespace NUMINAMATH_GPT_expected_value_of_biased_die_l1929_192996

-- Define the probabilities
def P1 : ℚ := 1/10
def P2 : ℚ := 1/10
def P3 : ℚ := 2/10
def P4 : ℚ := 2/10
def P5 : ℚ := 2/10
def P6 : ℚ := 2/10

-- Define the outcomes
def X1 : ℚ := 1
def X2 : ℚ := 2
def X3 : ℚ := 3
def X4 : ℚ := 4
def X5 : ℚ := 5
def X6 : ℚ := 6

-- Define the expected value calculation according to the probabilities and outcomes
def expected_value : ℚ := P1 * X1 + P2 * X2 + P3 * X3 + P4 * X4 + P5 * X5 + P6 * X6

-- The theorem we want to prove
theorem expected_value_of_biased_die : expected_value = 3.9 := by
  -- We skip the proof here with sorry for now
  sorry

end NUMINAMATH_GPT_expected_value_of_biased_die_l1929_192996


namespace NUMINAMATH_GPT_expressions_not_equal_l1929_192932

theorem expressions_not_equal (x : ℝ) (hx : x > 0) : 
  3 * x^x ≠ 2 * x^x + x^(2 * x) ∧ 
  x^(3 * x) ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^x ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^(3 * x) ≠ 2 * x^x + x^(2 * x) :=
by 
  sorry

end NUMINAMATH_GPT_expressions_not_equal_l1929_192932


namespace NUMINAMATH_GPT_inequality_solution_value_l1929_192953

theorem inequality_solution_value 
  (a : ℝ)
  (h : ∀ x, (1 < x ∧ x < 2) ↔ (ax / (x - 1) > 1)) :
  a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_value_l1929_192953


namespace NUMINAMATH_GPT_perfect_cube_divisor_l1929_192983

theorem perfect_cube_divisor (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a^2 + 3*a*b + 3*b^2 - 1 ∣ a + b^3) :
  ∃ k > 1, ∃ m : ℕ, a^2 + 3*a*b + 3*b^2 - 1 = k^3 * m := 
sorry

end NUMINAMATH_GPT_perfect_cube_divisor_l1929_192983


namespace NUMINAMATH_GPT_find_n_l1929_192971

theorem find_n (x y n : ℝ) (h1 : 2 * x - 5 * y = 3 * n + 7) (h2 : x - 3 * y = 4) 
  (h3 : x = y):
  n = -1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_l1929_192971


namespace NUMINAMATH_GPT_rectangular_prism_diagonals_l1929_192949

theorem rectangular_prism_diagonals :
  let l := 3
  let w := 4
  let h := 5
  let face_diagonals := 6 * 2
  let space_diagonals := 4
  face_diagonals + space_diagonals = 16 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_diagonals_l1929_192949


namespace NUMINAMATH_GPT_water_loss_per_jump_l1929_192981

def pool_capacity : ℕ := 2000 -- in liters
def jump_limit : ℕ := 1000
def clean_threshold : ℝ := 0.80

theorem water_loss_per_jump :
  (pool_capacity * (1 - clean_threshold)) * 1000 / jump_limit = 400 :=
by
  -- We prove that the water lost per jump in mL is 400
  sorry

end NUMINAMATH_GPT_water_loss_per_jump_l1929_192981


namespace NUMINAMATH_GPT_roots_of_eq_l1929_192916

theorem roots_of_eq (x : ℝ) : (x - 1) * (x - 2) = 0 ↔ (x = 1 ∨ x = 2) := by
  sorry

end NUMINAMATH_GPT_roots_of_eq_l1929_192916


namespace NUMINAMATH_GPT_complement_complement_l1929_192911

theorem complement_complement (alpha : ℝ) (h : alpha = 35) : (90 - (90 - alpha)) = 35 := by
  -- proof goes here, but we write sorry to skip it
  sorry

end NUMINAMATH_GPT_complement_complement_l1929_192911


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l1929_192960

variable {x : ℝ}

theorem parallel_vectors_x_value (h : (1 / x) = (2 / -6)) : x = -3 := sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l1929_192960


namespace NUMINAMATH_GPT_clarinet_fraction_l1929_192943

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

end NUMINAMATH_GPT_clarinet_fraction_l1929_192943


namespace NUMINAMATH_GPT_puffy_muffy_total_weight_l1929_192914

theorem puffy_muffy_total_weight (scruffy_weight muffy_weight puffy_weight : ℕ)
  (h1 : scruffy_weight = 12)
  (h2 : muffy_weight = scruffy_weight - 3)
  (h3 : puffy_weight = muffy_weight + 5) :
  puffy_weight + muffy_weight = 23 := by
  sorry

end NUMINAMATH_GPT_puffy_muffy_total_weight_l1929_192914


namespace NUMINAMATH_GPT_find_a_if_g_even_l1929_192978

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then x - 1 else if -2 ≤ x ∧ x ≤ 0 then -1 else 0

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x) + a * x

theorem find_a_if_g_even (a : ℝ) : (∀ x : ℝ, f x + a * x = f (-x) + a * (-x)) → a = -1/2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_if_g_even_l1929_192978


namespace NUMINAMATH_GPT_sum_is_24_l1929_192921

-- Define the conditions
def A := 3
def B := 7 * A

-- Define the theorem to prove that the sum is 24
theorem sum_is_24 : A + B = 24 :=
by
  -- Adding sorry here since we're not required to provide the proof
  sorry

end NUMINAMATH_GPT_sum_is_24_l1929_192921


namespace NUMINAMATH_GPT_total_students_in_Lansing_l1929_192982

theorem total_students_in_Lansing :
  let num_schools_300 := 20
  let num_schools_350 := 30
  let num_schools_400 := 15
  let students_per_school_300 := 300
  let students_per_school_350 := 350
  let students_per_school_400 := 400
  (num_schools_300 * students_per_school_300 + num_schools_350 * students_per_school_350 + num_schools_400 * students_per_school_400 = 22500) := 
  sorry

end NUMINAMATH_GPT_total_students_in_Lansing_l1929_192982


namespace NUMINAMATH_GPT_picture_area_l1929_192944

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 5) * (y + 4) - x * y = 84) : x * y = 15 :=
by
  sorry

end NUMINAMATH_GPT_picture_area_l1929_192944


namespace NUMINAMATH_GPT_karl_savings_proof_l1929_192938

-- Definitions based on the conditions
def original_price_per_notebook : ℝ := 3.00
def sale_discount : ℝ := 0.25
def extra_discount_threshold : ℝ := 10
def extra_discount_rate : ℝ := 0.05

-- The number of notebooks Karl could have purchased instead
def notebooks_purchased : ℝ := 12

-- The total savings calculation
noncomputable def total_savings : ℝ := 
  let original_total := notebooks_purchased * original_price_per_notebook
  let discounted_price_per_notebook := original_price_per_notebook * (1 - sale_discount)
  let extra_discount := if notebooks_purchased > extra_discount_threshold then discounted_price_per_notebook * extra_discount_rate else 0
  let total_price_after_discounts := notebooks_purchased * discounted_price_per_notebook - notebooks_purchased * extra_discount
  original_total - total_price_after_discounts

-- Formal statement to prove
theorem karl_savings_proof : total_savings = 10.35 := 
  sorry

end NUMINAMATH_GPT_karl_savings_proof_l1929_192938


namespace NUMINAMATH_GPT_infinite_double_perfect_squares_l1929_192946

-- Definition of a double number
def is_double_number (n : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (d : ℕ), d ≠ 0 ∧ 10^k * d + d = n ∧ 10^k ≤ d ∧ d < 10^(k+1)

-- The theorem statement
theorem infinite_double_perfect_squares :
  ∃ (S : Set ℕ), (∀ n ∈ S, is_double_number n ∧ ∃ m, m * m = n) ∧
  Set.Infinite S :=
sorry

end NUMINAMATH_GPT_infinite_double_perfect_squares_l1929_192946


namespace NUMINAMATH_GPT_power_product_l1929_192958

theorem power_product (m n : ℕ) (hm : 2 < m) (hn : 0 < n) : 
  (2^m - 1) * (2^n + 1) > 0 :=
by 
  sorry

end NUMINAMATH_GPT_power_product_l1929_192958


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1929_192994

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : 
  ∃ h : ℝ, h = Real.sqrt (a^2 + b^2) ∧ h = Real.sqrt 34 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1929_192994


namespace NUMINAMATH_GPT_m_range_and_simplification_l1929_192950

theorem m_range_and_simplification (x y m : ℝ)
  (h1 : (3 * (x + 1) / 2) + y = 2)
  (h2 : 3 * x - m = 2 * y)
  (hx : x ≤ 1)
  (hy : y ≤ 1) :
  (-3 ≤ m) ∧ (m ≤ 5) ∧ (|x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8) := 
by sorry

end NUMINAMATH_GPT_m_range_and_simplification_l1929_192950


namespace NUMINAMATH_GPT_softball_team_total_players_l1929_192900

theorem softball_team_total_players 
  (M W : ℕ) 
  (h1 : W = M + 4)
  (h2 : (M : ℚ) / (W : ℚ) = 0.6666666666666666) :
  M + W = 20 :=
by sorry

end NUMINAMATH_GPT_softball_team_total_players_l1929_192900


namespace NUMINAMATH_GPT_area_of_square_field_l1929_192966

theorem area_of_square_field (side_length : ℕ) (h : side_length = 25) :
  side_length * side_length = 625 := by
  sorry

end NUMINAMATH_GPT_area_of_square_field_l1929_192966


namespace NUMINAMATH_GPT_correct_division_result_l1929_192910

theorem correct_division_result (x : ℝ) 
  (h : (x - 14) / 5 = 11) : (x - 5) / 7 = 64 / 7 :=
by
  sorry

end NUMINAMATH_GPT_correct_division_result_l1929_192910


namespace NUMINAMATH_GPT_problem_solution_l1929_192941

theorem problem_solution (m n p : ℝ) 
  (h1 : 1 * m + 4 * p - 2 = 0) 
  (h2 : 2 * 1 - 5 * p + n = 0) 
  (h3 : (m / (-4)) * (2 / 5) = -1) :
  n = -12 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1929_192941


namespace NUMINAMATH_GPT_sub_neg_four_l1929_192928

theorem sub_neg_four : -3 - 1 = -4 :=
by
  sorry

end NUMINAMATH_GPT_sub_neg_four_l1929_192928


namespace NUMINAMATH_GPT_parallel_resistor_problem_l1929_192975

theorem parallel_resistor_problem
  (x : ℝ)
  (r : ℝ := 2.2222222222222223)
  (y : ℝ := 5) : 
  (1 / r = 1 / x + 1 / y) → x = 4 :=
by sorry

end NUMINAMATH_GPT_parallel_resistor_problem_l1929_192975


namespace NUMINAMATH_GPT_chosen_number_l1929_192936

theorem chosen_number (x : ℕ) (h : 5 * x - 138 = 102) : x = 48 :=
sorry

end NUMINAMATH_GPT_chosen_number_l1929_192936


namespace NUMINAMATH_GPT_probability_of_blue_ball_l1929_192984

theorem probability_of_blue_ball 
(P_red P_yellow P_blue : ℝ) 
(h_red : P_red = 0.48)
(h_yellow : P_yellow = 0.35) 
(h_prob : P_red + P_yellow + P_blue = 1) 
: P_blue = 0.17 := 
sorry

end NUMINAMATH_GPT_probability_of_blue_ball_l1929_192984


namespace NUMINAMATH_GPT_sample_size_is_150_l1929_192926

theorem sample_size_is_150 
  (classes : ℕ) (students_per_class : ℕ) (selected_students : ℕ)
  (h1 : classes = 40) (h2 : students_per_class = 50) (h3 : selected_students = 150)
  : selected_students = 150 :=
sorry

end NUMINAMATH_GPT_sample_size_is_150_l1929_192926


namespace NUMINAMATH_GPT_Amelia_wins_probability_correct_l1929_192923

-- Define the probabilities
def probability_Amelia_heads := 1 / 3
def probability_Blaine_heads := 2 / 5

-- The infinite geometric series sum calculation for Amelia to win
def probability_Amelia_wins :=
  probability_Amelia_heads * (1 / (1 - (1 - probability_Amelia_heads) * (1 - probability_Blaine_heads)))

-- Given values p and q from the conditions
def p := 5
def q := 9

-- The correct answer $\frac{5}{9}$
def Amelia_wins_correct := 5 / 9

-- Prove that the probability calculation matches the given $\frac{5}{9}$, and find q - p
theorem Amelia_wins_probability_correct :
  probability_Amelia_wins = Amelia_wins_correct ∧ q - p = 4 := by
  sorry

end NUMINAMATH_GPT_Amelia_wins_probability_correct_l1929_192923


namespace NUMINAMATH_GPT_width_of_wall_l1929_192920

-- Define the dimensions of a single brick.
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of bricks.
def num_bricks : ℝ := 6800

-- Define the dimensions of the wall (length and height).
def wall_length : ℝ := 850
def wall_height : ℝ := 600

-- Prove that the width of the wall is 22.5 cm.
theorem width_of_wall : 
  (wall_length * wall_height * 22.5 = num_bricks * (brick_length * brick_width * brick_height)) :=
by
  sorry

end NUMINAMATH_GPT_width_of_wall_l1929_192920


namespace NUMINAMATH_GPT_simplify_expression_l1929_192937

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_simplify_expression_l1929_192937


namespace NUMINAMATH_GPT_base3_to_base10_conversion_l1929_192929

theorem base3_to_base10_conversion : ∀ n : ℕ, n = 120102 → (1 * 3^5 + 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = 416 :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_base3_to_base10_conversion_l1929_192929


namespace NUMINAMATH_GPT_min_lines_to_separate_points_l1929_192963

theorem min_lines_to_separate_points (m n : ℕ) (h_m : m = 8) (h_n : n = 8) : 
  (m - 1) + (n - 1) = 14 := by
  sorry

end NUMINAMATH_GPT_min_lines_to_separate_points_l1929_192963


namespace NUMINAMATH_GPT_father_age_l1929_192952

theorem father_age (F D : ℕ) (h1 : F = 4 * D) (h2 : (F + 5) + (D + 5) = 50) : F = 32 :=
by
  sorry

end NUMINAMATH_GPT_father_age_l1929_192952


namespace NUMINAMATH_GPT_impossible_to_save_one_minute_for_60kmh_l1929_192955

theorem impossible_to_save_one_minute_for_60kmh (v : ℝ) (h : v = 60) :
  ¬ ∃ (new_v : ℝ), 1 / new_v = (1 / 60) - 1 :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_save_one_minute_for_60kmh_l1929_192955


namespace NUMINAMATH_GPT_range_of_a_extrema_of_y_l1929_192962

variable {a b c : ℝ}

def setA (a b c : ℝ) : Prop := a^2 - b * c - 8 * a + 7 = 0
def setB (a b c : ℝ) : Prop := b^2 + c^2 + b * c - b * a + b = 0

theorem range_of_a (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) : 1 ≤ a ∧ a ≤ 9 :=
sorry

theorem extrema_of_y (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) 
  (y : ℝ) 
  (hy1 : y = a * b + b * c + a * c)
  (hy2 : ∀ x y z : ℝ, setA x y z → setB x y z → y = x * y + y * z + x * z) : 
  y = 88 ∨ y = -56 :=
sorry

end NUMINAMATH_GPT_range_of_a_extrema_of_y_l1929_192962


namespace NUMINAMATH_GPT_alyssa_total_spent_l1929_192901

theorem alyssa_total_spent :
  let grapes := 12.08
  let cherries := 9.85
  grapes + cherries = 21.93 := by
  sorry

end NUMINAMATH_GPT_alyssa_total_spent_l1929_192901


namespace NUMINAMATH_GPT_ashton_sheets_l1929_192997
-- Import the entire Mathlib to bring in the necessary library

-- Defining the conditions and proving the statement
theorem ashton_sheets (t j a : ℕ) (h1 : t = j + 10) (h2 : j = 32) (h3 : j + a = t + 30) : a = 40 := by
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_ashton_sheets_l1929_192997


namespace NUMINAMATH_GPT_not_in_range_g_zero_l1929_192918

noncomputable def g (x: ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else 0 -- g(x) is not defined at x = -3, this is a placeholder

theorem not_in_range_g_zero :
  ¬ (∃ x : ℝ, g x = 0) :=
sorry

end NUMINAMATH_GPT_not_in_range_g_zero_l1929_192918


namespace NUMINAMATH_GPT_certain_number_N_l1929_192919

theorem certain_number_N (G N : ℕ) (hG : G = 127)
  (h₁ : ∃ k : ℕ, N = G * k + 10)
  (h₂ : ∃ m : ℕ, 2045 = G * m + 13) :
  N = 2042 :=
sorry

end NUMINAMATH_GPT_certain_number_N_l1929_192919


namespace NUMINAMATH_GPT_hexagon_label_count_l1929_192956

def hexagon_label (s : Finset ℕ) (a b c d e f g : ℕ) : Prop :=
  s = Finset.range 8 ∧ 
  (a ∈ s) ∧ (b ∈ s) ∧ (c ∈ s) ∧ (d ∈ s) ∧ (e ∈ s) ∧ (f ∈ s) ∧ (g ∈ s) ∧
  a + b + c + d + e + f + g = 28 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ 
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  a + g + d = b + g + e ∧ b + g + e = c + g + f

theorem hexagon_label_count : ∃ s a b c d e f g, hexagon_label s a b c d e f g ∧ 
  (s.card = 8) ∧ (a + g + d = 10) ∧ (b + g + e = 10) ∧ (c + g + f = 10) ∧ 
  144 = 3 * 48 :=
sorry

end NUMINAMATH_GPT_hexagon_label_count_l1929_192956


namespace NUMINAMATH_GPT_least_subtraction_for_divisibility_l1929_192964

def original_number : ℕ := 5474827

def required_subtraction : ℕ := 7

theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, (original_number - required_subtraction) = 12 * k :=
sorry

end NUMINAMATH_GPT_least_subtraction_for_divisibility_l1929_192964


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l1929_192933

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n, a (n+1) - a n = a 2 - a 1)
  (h_a1 : a 1 = 5) (h_a5 : a 5 = 1) : a 6 = 0 :=
by
  -- Definitions derived from conditions in the problem:
  -- 1. a : ℕ → ℤ : Sequence defined on ℕ with integer values.
  -- 2. h_arith : ∀ n, a (n+1) - a n = a 2 - a 1 : Arithmetic sequence property
  -- 3. h_a1 : a 1 = 5 : First term of the sequence is 5.
  -- 4. h_a5 : a 5 = 1 : Fifth term of the sequence is 1.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l1929_192933


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1929_192967

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 0 + a 1 + a 2 = 8)
  (h2 : a 3 + a 4 + a 5 = -4) :
  a 6 + a 7 + a 8 = 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1929_192967


namespace NUMINAMATH_GPT_profit_percentage_previous_year_l1929_192902

-- Declaring variables
variables (R P : ℝ) -- revenues and profits in the previous year
variable (revenues_1999 := 0.8 * R) -- revenues in 1999
variable (profits_1999 := 0.14 * revenues_1999) -- profits in 1999

-- Given condition: profits in 1999 were 112.00000000000001 percent of the profits in the previous year
axiom profits_ratio : 0.112 * R = 1.1200000000000001 * P

-- Prove the profit as a percentage of revenues in the previous year was 10%
theorem profit_percentage_previous_year : (P / R) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_previous_year_l1929_192902


namespace NUMINAMATH_GPT_circle_equation_has_valid_k_l1929_192924

theorem circle_equation_has_valid_k (k : ℝ) : (∃ a b r : ℝ, r > 0 ∧ ∀ x y : ℝ, (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ k < 5 / 4 := by
  sorry

end NUMINAMATH_GPT_circle_equation_has_valid_k_l1929_192924


namespace NUMINAMATH_GPT_remainder_of_hx10_divided_by_hx_is_6_l1929_192961

noncomputable def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_hx10_divided_by_hx_is_6 : 
  let q := h (x ^ 10);
  q % h (x) = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_of_hx10_divided_by_hx_is_6_l1929_192961


namespace NUMINAMATH_GPT_dilation_image_l1929_192934

theorem dilation_image :
  let z_0 := (1 : ℂ) + 2 * I
  let k := (2 : ℂ)
  let z_1 := (3 : ℂ) + I
  let z := z_0 + k * (z_1 - z_0)
  z = 5 :=
by
  sorry

end NUMINAMATH_GPT_dilation_image_l1929_192934


namespace NUMINAMATH_GPT_cell_phone_plan_cost_l1929_192948

theorem cell_phone_plan_cost:
  let base_cost : ℕ := 25
  let text_cost : ℕ := 8
  let extra_min_cost : ℕ := 12
  let texts_sent : ℕ := 150
  let hours_talked : ℕ := 27
  let extra_minutes := (hours_talked - 25) * 60
  let total_cost := (base_cost * 100) + (texts_sent * text_cost) + (extra_minutes * extra_min_cost)
  (total_cost = 5140) :=
by
  sorry

end NUMINAMATH_GPT_cell_phone_plan_cost_l1929_192948
