import Mathlib

namespace NUMINAMATH_GPT_equation_of_line_AB_l905_90506

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨2, -1⟩

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the center C
def C : Point := ⟨1, 0⟩

-- The equation of line AB we want to verify
def line_AB (P : Point) := P.x - P.y - 3 = 0

-- The theorem to prove
theorem equation_of_line_AB :
  (circle_eq P.x P.y ∧ P = ⟨2, -1⟩ ∧ C = ⟨1, 0⟩) → line_AB P :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_AB_l905_90506


namespace NUMINAMATH_GPT_find_b50_l905_90500

noncomputable def T (n : ℕ) : ℝ := if n = 1 then 2 else 2 / (6 * n - 5)

noncomputable def b (n : ℕ) : ℝ :=
  if n = 1 then 2 else T n - T (n - 1)

theorem find_b50 : b 50 = -6 / 42677.5 := by sorry

end NUMINAMATH_GPT_find_b50_l905_90500


namespace NUMINAMATH_GPT_min_knights_proof_l905_90548

-- Noncomputable theory as we are dealing with existence proofs
noncomputable def min_knights (n : ℕ) : ℕ :=
  -- Given the table contains 1001 people
  if n = 1001 then 502 else 0

-- The proof problem statement, we need to ensure that minimum number of knights is 502
theorem min_knights_proof : min_knights 1001 = 502 := 
  by
    -- Sketch of proof: Deriving that the minimum number of knights must be 502 based on the problem constraints
    sorry

end NUMINAMATH_GPT_min_knights_proof_l905_90548


namespace NUMINAMATH_GPT_find_missing_score_l905_90526

theorem find_missing_score
  (scores : List ℕ)
  (h_scores : scores = [73, 83, 86, 73, x])
  (mean : ℚ)
  (h_mean : mean = 79.2)
  (h_length : scores.length = 5)
  : x = 81 := by
  sorry

end NUMINAMATH_GPT_find_missing_score_l905_90526


namespace NUMINAMATH_GPT_work_together_days_l905_90540

theorem work_together_days (A_rate B_rate : ℝ) (x B_alone_days : ℝ)
  (hA : A_rate = 1 / 5)
  (hB : B_rate = 1 / 15)
  (h_total_work : (A_rate + B_rate) * x + B_rate * B_alone_days = 1) :
  x = 2 :=
by
  -- Set up the equation based on given rates and solving for x.
  sorry

end NUMINAMATH_GPT_work_together_days_l905_90540


namespace NUMINAMATH_GPT_least_subtraction_divisible_by13_l905_90517

theorem least_subtraction_divisible_by13 (n : ℕ) (h : n = 427398) : ∃ k : ℕ, k = 2 ∧ (n - k) % 13 = 0 := by
  sorry

end NUMINAMATH_GPT_least_subtraction_divisible_by13_l905_90517


namespace NUMINAMATH_GPT_types_of_problems_l905_90558

def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def problems_per_type : ℕ := 30

theorem types_of_problems : (frank_problems / problems_per_type) = 4 := by
  sorry

end NUMINAMATH_GPT_types_of_problems_l905_90558


namespace NUMINAMATH_GPT_speed_of_faster_train_l905_90511

noncomputable def speed_of_slower_train : ℝ := 36
noncomputable def length_of_each_train : ℝ := 70
noncomputable def time_to_pass : ℝ := 36

theorem speed_of_faster_train : 
  ∃ (V_f : ℝ), 
    (V_f - speed_of_slower_train) * (1000 / 3600) = 140 / time_to_pass ∧ 
    V_f = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_speed_of_faster_train_l905_90511


namespace NUMINAMATH_GPT_compare_sqrt_l905_90556

noncomputable def a : ℝ := 3 * Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 15

theorem compare_sqrt : a > b :=
by
  sorry

end NUMINAMATH_GPT_compare_sqrt_l905_90556


namespace NUMINAMATH_GPT_variance_of_yield_l905_90544

/-- Given a data set representing annual average yields,
    prove that the variance of this data set is approximately 171. --/
theorem variance_of_yield {yields : List ℝ} 
  (h_yields : yields = [450, 430, 460, 440, 450, 440, 470, 460]) :
  let mean := (yields.sum / yields.length : ℝ)
  let squared_diffs := (yields.map (fun x => (x - mean)^2))
  let variance := (squared_diffs.sum / (yields.length - 1 : ℝ))
  abs (variance - 171) < 1 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_yield_l905_90544


namespace NUMINAMATH_GPT_tan_of_angle_l905_90529

theorem tan_of_angle (α : ℝ) (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h₂ : Real.sin α = 3 / 5) : 
  Real.tan α = -3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_of_angle_l905_90529


namespace NUMINAMATH_GPT_line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l905_90577

theorem line_eq_45_deg_y_intercept_2 :
  (∃ l : ℝ → ℝ, (l 0 = 2) ∧ (∀ x, l x = x + 2)) := sorry

theorem circle_eq_center_neg2_3_tangent_yaxis :
  (∃ c : ℝ × ℝ → ℝ, (c (-2, 3) = 0) ∧ (∀ x y, c (x, y) = (x + 2)^2 + (y - 3)^2 - 4)) := sorry

end NUMINAMATH_GPT_line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l905_90577


namespace NUMINAMATH_GPT_total_bills_inserted_l905_90550

theorem total_bills_inserted (x y : ℕ) (h1 : x = 175) (h2 : x + 5 * y = 300) : 
  x + y = 200 :=
by {
  -- Since we focus strictly on the statement per instruction, the proof is omitted
  sorry 
}

end NUMINAMATH_GPT_total_bills_inserted_l905_90550


namespace NUMINAMATH_GPT_smallest_a_l905_90571

-- Define the conditions and the proof goal
theorem smallest_a (a b : ℝ) (h₁ : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (15 * (x : ℝ))) (h₂ : 0 ≤ a) (h₃ : 0 ≤ b) :
  a = 15 :=
sorry

end NUMINAMATH_GPT_smallest_a_l905_90571


namespace NUMINAMATH_GPT_range_of_a_l905_90527

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 * a + 1)^x > (2 * a + 1)^y) → (-1/2 < a ∧ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l905_90527


namespace NUMINAMATH_GPT_total_donation_l905_90551

-- Define the conditions in the problem
def Barbara_stuffed_animals : ℕ := 9
def Trish_stuffed_animals : ℕ := 2 * Barbara_stuffed_animals
def Barbara_sale_price : ℝ := 2
def Trish_sale_price : ℝ := 1.5

-- Define the goal as a theorem to be proven
theorem total_donation : Barbara_sale_price * Barbara_stuffed_animals + Trish_sale_price * Trish_stuffed_animals = 45 := by
  sorry

end NUMINAMATH_GPT_total_donation_l905_90551


namespace NUMINAMATH_GPT_reserve_bird_percentage_l905_90557

theorem reserve_bird_percentage (total_birds hawks paddyfield_warbler_percentage kingfisher_percentage woodpecker_percentage owl_percentage : ℕ) 
  (h1 : total_birds = 5000)
  (h2 : hawks = 30 * total_birds / 100)
  (h3 : paddyfield_warbler_percentage = 40)
  (h4 : kingfisher_percentage = 25)
  (h5 : woodpecker_percentage = 15)
  (h6 : owl_percentage = 15) :
  let non_hawks := total_birds - hawks
  let paddyfield_warblers := paddyfield_warbler_percentage * non_hawks / 100
  let kingfishers := kingfisher_percentage * paddyfield_warblers / 100
  let woodpeckers := woodpecker_percentage * non_hawks / 100
  let owls := owl_percentage * non_hawks / 100
  let specified_non_hawks := paddyfield_warblers + kingfishers + woodpeckers + owls
  let unspecified_non_hawks := non_hawks - specified_non_hawks
  let percentage_unspecified := unspecified_non_hawks * 100 / total_birds
  percentage_unspecified = 14 := by
  sorry

end NUMINAMATH_GPT_reserve_bird_percentage_l905_90557


namespace NUMINAMATH_GPT_find_passing_marks_l905_90566

-- Defining the conditions as Lean statements
def condition1 (T P : ℝ) : Prop := 0.30 * T = P - 50
def condition2 (T P : ℝ) : Prop := 0.45 * T = P + 25

-- The theorem to prove
theorem find_passing_marks (T P : ℝ) (h1 : condition1 T P) (h2 : condition2 T P) : P = 200 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_find_passing_marks_l905_90566


namespace NUMINAMATH_GPT_max_profit_l905_90549

noncomputable def total_cost (Q : ℝ) : ℝ := 5 * Q^2

noncomputable def demand_non_slytherin (P : ℝ) : ℝ := 26 - 2 * P

noncomputable def demand_slytherin (P : ℝ) : ℝ := 10 - P

noncomputable def combined_demand (P : ℝ) : ℝ :=
  if P >= 13 then demand_non_slytherin P else demand_non_slytherin P + demand_slytherin P

noncomputable def inverse_demand (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q / 2 else 12 - Q / 3

noncomputable def revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then Q * (13 - Q / 2) else Q * (12 - Q / 3)

noncomputable def marginal_revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q else 12 - 2 * Q / 3

noncomputable def marginal_cost (Q : ℝ) : ℝ := 10 * Q

theorem max_profit :
  ∃ Q P TR TC π,
    P = inverse_demand Q ∧
    TR = P * Q ∧
    TC = total_cost Q ∧
    π = TR - TC ∧
    π = 7.69 :=
sorry

end NUMINAMATH_GPT_max_profit_l905_90549


namespace NUMINAMATH_GPT_find_c_l905_90573

theorem find_c (x : ℝ) (c : ℝ) (h : x = 0.3)
  (equ : (10 * x + 2) / c - (3 * x - 6) / 18 = (2 * x + 4) / 3) :
  c = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l905_90573


namespace NUMINAMATH_GPT_radius_of_smaller_molds_l905_90518

noncomputable def volumeOfHemisphere (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

theorem radius_of_smaller_molds (r : ℝ) :
  volumeOfHemisphere 2 = 64 * volumeOfHemisphere r → r = 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_radius_of_smaller_molds_l905_90518


namespace NUMINAMATH_GPT_equation_solution_l905_90519

noncomputable def solve_equation (x : ℝ) : Prop :=
  (1/4) * x^(1/2 * Real.log x / Real.log 2) = 2^(1/4 * (Real.log x / Real.log 2)^2)

theorem equation_solution (x : ℝ) (hx : 0 < x) : solve_equation x → (x = 2^(2*Real.sqrt 2) ∨ x = 2^(-2*Real.sqrt 2)) :=
  by
  intro h
  sorry

end NUMINAMATH_GPT_equation_solution_l905_90519


namespace NUMINAMATH_GPT_surface_area_of_tunneled_cube_l905_90524

-- Definition of the initial cube and its properties.
def cube (side_length : ℕ) := side_length * side_length * side_length

-- Initial side length of the large cube
def large_cube_side : ℕ := 12

-- Each small cube side length
def small_cube_side : ℕ := 3

-- Number of small cubes that fit into the large cube
def num_small_cubes : ℕ := (cube large_cube_side) / (cube small_cube_side)

-- Number of cubes removed initially
def removed_cubes : ℕ := 27

-- Number of remaining cubes after initial removal
def remaining_cubes : ℕ := num_small_cubes - removed_cubes

-- Surface area of each unmodified small cube
def small_cube_surface : ℕ := 54

-- Additional surface area due to removal of center units
def additional_surface : ℕ := 24

-- Surface area of each modified small cube
def modified_cube_surface : ℕ := small_cube_surface + additional_surface

-- Total surface area before adjustment for shared faces
def total_surface_before_adjustment : ℕ := remaining_cubes * modified_cube_surface

-- Shared surface area to be subtracted
def shared_surface : ℕ := 432

-- Final surface area of the resulting figure
def final_surface_area : ℕ := total_surface_before_adjustment - shared_surface

-- Theorem statement
theorem surface_area_of_tunneled_cube : final_surface_area = 2454 :=
by {
  -- Proof required here
  sorry
}

end NUMINAMATH_GPT_surface_area_of_tunneled_cube_l905_90524


namespace NUMINAMATH_GPT_coast_guard_overtake_smuggler_l905_90595

noncomputable def time_of_overtake (initial_distance : ℝ) (initial_time : ℝ) 
                                   (smuggler_speed1 coast_guard_speed : ℝ) 
                                   (duration1 new_smuggler_speed : ℝ) : ℝ :=
  let distance_after_duration1 := initial_distance + (smuggler_speed1 * duration1) - (coast_guard_speed * duration1)
  let relative_speed_new := coast_guard_speed - new_smuggler_speed
  duration1 + (distance_after_duration1 / relative_speed_new)

theorem coast_guard_overtake_smuggler : 
  time_of_overtake 15 0 18 20 1 16 = 4.25 := by
  sorry

end NUMINAMATH_GPT_coast_guard_overtake_smuggler_l905_90595


namespace NUMINAMATH_GPT_Mina_stops_in_D_or_A_l905_90530

-- Define the relevant conditions and problem statement
def circumference := 60
def total_distance := 6000
def quarters := ["A", "B", "C", "D"]
def start_position := "S"
def stop_position := if (total_distance % circumference) == 0 then "S" else ""

theorem Mina_stops_in_D_or_A : stop_position = start_position → start_position = "D" ∨ start_position = "A" :=
by
  sorry

end NUMINAMATH_GPT_Mina_stops_in_D_or_A_l905_90530


namespace NUMINAMATH_GPT_negation_of_exists_is_forall_l905_90545

theorem negation_of_exists_is_forall :
  (¬ ∃ x : ℝ, x^3 + 1 = 0) ↔ ∀ x : ℝ, x^3 + 1 ≠ 0 :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_exists_is_forall_l905_90545


namespace NUMINAMATH_GPT_inequalities_hold_l905_90583

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_inequalities_hold_l905_90583


namespace NUMINAMATH_GPT_volume_and_surface_area_of_convex_body_l905_90591

noncomputable def volume_of_convex_body (a b c : ℝ) : ℝ := 
  (a^2 + b^2 + c^2)^3 / (6 * a * b * c)

noncomputable def surface_area_of_convex_body (a b c : ℝ) : ℝ :=
  (a^2 + b^2 + c^2)^(5/2) / (a * b * c)

theorem volume_and_surface_area_of_convex_body (a b c d : ℝ)
  (h : d^2 = a^2 + b^2 + c^2) :
  volume_of_convex_body a b c = (a^2 + b^2 + c^2)^3 / (6 * a * b * c) ∧
  surface_area_of_convex_body a b c = (a^2 + b^2 + c^2)^(5/2) / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_volume_and_surface_area_of_convex_body_l905_90591


namespace NUMINAMATH_GPT_finite_set_elements_at_least_half_m_l905_90520

theorem finite_set_elements_at_least_half_m (m : ℕ) (A : Finset ℤ) (B : ℕ → Finset ℤ) 
  (hm : 2 ≤ m) 
  (hB : ∀ k : ℕ, 1 ≤ k → k ≤ m → (B k).sum id = (m : ℤ) ^ k) : 
  ∃ n : ℕ, (A.card ≥ n) ∧ (n ≥ m / 2) :=
by
  sorry

end NUMINAMATH_GPT_finite_set_elements_at_least_half_m_l905_90520


namespace NUMINAMATH_GPT_sum_of_digits_0_to_2012_l905_90568

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

def sum_of_digits_in_range (a b : Nat) : Nat :=
  ((List.range (b + 1)).drop a).map sum_of_digits |>.sum

theorem sum_of_digits_0_to_2012 : 
  sum_of_digits_in_range 0 2012 = 28077 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_0_to_2012_l905_90568


namespace NUMINAMATH_GPT_part1_part2_l905_90501

-- Part (1): Proving the range of x when a = 1
theorem part1 (x : ℝ) : (x^2 - 6 * 1 * x + 8 < 0) ∧ (x^2 - 4 * x + 3 ≤ 0) ↔ 2 < x ∧ x ≤ 3 := 
by sorry

-- Part (2): Proving the range of a when p is a sufficient but not necessary condition for q
theorem part2 (a : ℝ) : (∀ x : ℝ, (x^2 - 6 * a * x + 8 * a^2 < 0 → x^2 - 4 * x + 3 ≤ 0) 
  ∧ (∃ x : ℝ, x^2 - 4 * x + 3 ≤ 0 ∧ x^2 - 6 * a * x + 8 * a^2 ≥ 0)) ↔ (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l905_90501


namespace NUMINAMATH_GPT_previous_job_salary_is_correct_l905_90523

-- Define the base salary and commission structure.
def base_salary_new_job : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750
def minimum_sales : ℝ := 266.67

-- Define the total salary from the new job with the minimum sales.
def new_job_total_salary : ℝ :=
  base_salary_new_job + (commission_rate * sale_amount * minimum_sales)

-- Define Tom's previous job's salary.
def previous_job_salary : ℝ := 75000

-- Prove that Tom's previous job salary matches the new job total salary with the minimum sales.
theorem previous_job_salary_is_correct :
  (new_job_total_salary = previous_job_salary) :=
by
  -- This is where you would include the proof steps, but it's sufficient to put 'sorry' for now.
  sorry

end NUMINAMATH_GPT_previous_job_salary_is_correct_l905_90523


namespace NUMINAMATH_GPT_radhika_christmas_games_l905_90593

variable (C B : ℕ)

def games_on_birthday := 8
def total_games (C : ℕ) (B : ℕ) := C + B + (C + B) / 2

theorem radhika_christmas_games : 
  total_games C games_on_birthday = 30 → C = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_radhika_christmas_games_l905_90593


namespace NUMINAMATH_GPT_irrational_number_line_representation_l905_90597

theorem irrational_number_line_representation :
  ∀ (x : ℝ), ¬ (∃ r s : ℚ, x = r / s ∧ r ≠ 0 ∧ s ≠ 0) → ∃ p : ℝ, x = p := 
by
  sorry

end NUMINAMATH_GPT_irrational_number_line_representation_l905_90597


namespace NUMINAMATH_GPT_salary_increase_l905_90534

theorem salary_increase (S P : ℝ) (h1 : 0.70 * S + P * (0.70 * S) = 0.91 * S) : P = 0.30 :=
by
  have eq1 : 0.70 * S * (1 + P) = 0.91 * S := by sorry
  have eq2 : S * (0.70 + 0.70 * P) = 0.91 * S := by sorry
  have eq3 : 0.70 + 0.70 * P = 0.91 := by sorry
  have eq4 : 0.70 * P = 0.21 := by sorry
  have eq5 : P = 0.21 / 0.70 := by sorry
  have eq6 : P = 0.30 := by sorry
  exact eq6

end NUMINAMATH_GPT_salary_increase_l905_90534


namespace NUMINAMATH_GPT_average_price_of_fruit_l905_90538

theorem average_price_of_fruit 
  (price_apple price_orange : ℝ)
  (total_fruits initial_fruits kept_oranges kept_fruits : ℕ)
  (average_price_kept average_price_initial : ℝ)
  (h1 : price_apple = 40)
  (h2 : price_orange = 60)
  (h3 : initial_fruits = 10)
  (h4 : kept_oranges = initial_fruits - 6)
  (h5 : average_price_kept = 50) :
  average_price_initial = 56 := 
sorry

end NUMINAMATH_GPT_average_price_of_fruit_l905_90538


namespace NUMINAMATH_GPT_eq_curveE_eq_lineCD_l905_90579

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def curveE (x y : ℝ) : Prop :=
  distance (x, y) (-1, 0) = Real.sqrt 3 * distance (x, y) (1, 0)

theorem eq_curveE (x y : ℝ) : curveE x y ↔ (x - 2)^2 + y^2 = 3 :=
by sorry

variables (m : ℝ)
variables (m_nonzero : m ≠ 0)
variables (A C B D : ℝ × ℝ)
variables (line1_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = A ∨ p = C) → p.1 - m * p.2 - 1 = 0)
variables (line2_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = B ∨ p = D) → m * p.1 + p.2 - m = 0)
variables (CD_slope : (D.2 - C.2) / (D.1 - C.1) = -1)

theorem eq_lineCD (x y : ℝ) : 
  (y = -x ∨ y = -x + 3) :=
by sorry

end NUMINAMATH_GPT_eq_curveE_eq_lineCD_l905_90579


namespace NUMINAMATH_GPT_isosceles_obtuse_triangle_l905_90564

theorem isosceles_obtuse_triangle (A B C : ℝ) (h_isosceles: A = B)
  (h_obtuse: A + B + C = 180) 
  (h_max_angle: C = 157.5): A = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_obtuse_triangle_l905_90564


namespace NUMINAMATH_GPT_find_number_l905_90575

theorem find_number (x : ℝ) (h1 : 0.35 * x = 0.2 * 700) (h2 : 0.2 * 700 = 140) (h3 : 0.35 * x = 140) : x = 400 :=
by sorry

end NUMINAMATH_GPT_find_number_l905_90575


namespace NUMINAMATH_GPT_fraction_identity_l905_90562

theorem fraction_identity (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l905_90562


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l905_90502

-- Definition of symmetry in the Cartesian coordinate system
def is_symmetrical_about_origin (A A' : ℝ × ℝ) : Prop :=
  A'.1 = -A.1 ∧ A'.2 = -A.2

-- Given point A and its symmetric property to find point A'
theorem symmetric_point_coordinates (A A' : ℝ × ℝ)
  (hA : A = (1, -2))
  (h_symm : is_symmetrical_about_origin A A') :
  A' = (-1, 2) :=
by
  sorry -- Proof to be filled in (not required as per the instructions)

end NUMINAMATH_GPT_symmetric_point_coordinates_l905_90502


namespace NUMINAMATH_GPT_negative_represents_backward_l905_90589

-- Definitions based on conditions
def forward (distance : Int) : Int := distance
def backward (distance : Int) : Int := -distance

-- The mathematical equivalent proof problem
theorem negative_represents_backward
  (distance : Int)
  (h : forward distance = 5) :
  backward distance = -5 :=
sorry

end NUMINAMATH_GPT_negative_represents_backward_l905_90589


namespace NUMINAMATH_GPT_circle_tangent_to_line_iff_m_eq_zero_l905_90539

theorem circle_tangent_to_line_iff_m_eq_zero (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m^2 ∧ x - y = m) ↔ m = 0 :=
by 
  sorry

end NUMINAMATH_GPT_circle_tangent_to_line_iff_m_eq_zero_l905_90539


namespace NUMINAMATH_GPT_find_base_of_log_equation_l905_90592

theorem find_base_of_log_equation :
  ∃ b : ℝ, (∀ x : ℝ, (9 : ℝ)^(x + 5) = (5 : ℝ)^x → x = Real.logb b ((9 : ℝ)^5)) ∧ b = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_base_of_log_equation_l905_90592


namespace NUMINAMATH_GPT_find_t2_l905_90533

variable {P A1 A2 t1 r t2 : ℝ}
def conditions (P A1 A2 t1 r t2 : ℝ) :=
  P = 650 ∧
  A1 = 815 ∧
  A2 = 870 ∧
  t1 = 3 ∧
  A1 = P + (P * r * t1) / 100 ∧
  A2 = P + (P * r * t2) / 100

theorem find_t2
  (P A1 A2 t1 r t2 : ℝ)
  (hc : conditions P A1 A2 t1 r t2) :
  t2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_t2_l905_90533


namespace NUMINAMATH_GPT_determine_ABC_l905_90535

theorem determine_ABC : 
  ∀ (A B C : ℝ), 
    A = 2 * B - 3 * C ∧ 
    B = 2 * C - 5 ∧ 
    A + B + C = 100 → 
    A = 18.75 ∧ B = 52.5 ∧ C = 28.75 :=
by
  intro A B C h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_determine_ABC_l905_90535


namespace NUMINAMATH_GPT_number_of_red_balls_l905_90546

theorem number_of_red_balls
    (black_balls : ℕ)
    (frequency : ℝ)
    (total_balls : ℕ)
    (red_balls : ℕ) 
    (h_black : black_balls = 5)
    (h_frequency : frequency = 0.25)
    (h_total : total_balls = black_balls / frequency) :
    red_balls = total_balls - black_balls → red_balls = 15 :=
by
  intros h_red
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l905_90546


namespace NUMINAMATH_GPT_container_fullness_calc_l905_90516

theorem container_fullness_calc (initial_percent : ℝ) (added_water : ℝ) (total_capacity : ℝ) (result_fraction : ℝ) :
  initial_percent = 0.3 →
  added_water = 27 →
  total_capacity = 60 →
  result_fraction = 3/4 →
  ((initial_percent * total_capacity + added_water) / total_capacity) = result_fraction :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_container_fullness_calc_l905_90516


namespace NUMINAMATH_GPT_a_share_is_6300_l905_90522

noncomputable def investment_split (x : ℝ) :  ℝ × ℝ × ℝ :=
  let a_share := x * 12
  let b_share := 2 * x * 6
  let c_share := 3 * x * 4
  (a_share, b_share, c_share)

noncomputable def total_gain : ℝ := 18900

noncomputable def a_share_calculation : ℝ :=
  let (a_share, b_share, c_share) := investment_split 1
  total_gain / (a_share + b_share + c_share) * a_share

theorem a_share_is_6300 : a_share_calculation = 6300 := by
  -- Here, you would provide the proof, but for now we skip it.
  sorry

end NUMINAMATH_GPT_a_share_is_6300_l905_90522


namespace NUMINAMATH_GPT_average_score_girls_l905_90521

theorem average_score_girls (num_boys num_girls : ℕ) (avg_boys avg_class : ℕ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  avg_boys = 84 → 
  avg_class = 86 → 
  ∃ avg_girls : ℕ, avg_girls = 92 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_score_girls_l905_90521


namespace NUMINAMATH_GPT_f_is_periodic_l905_90572

noncomputable def f : ℝ → ℝ := sorry

def a : ℝ := sorry

axiom exists_a_gt_zero : a > 0

axiom functional_eq (x : ℝ) : f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)

theorem f_is_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end NUMINAMATH_GPT_f_is_periodic_l905_90572


namespace NUMINAMATH_GPT_max_value_range_of_t_l905_90514

theorem max_value_range_of_t (t x : ℝ) (h : t ≤ x ∧ x ≤ t + 2) 
: ∃ y : ℝ, y = -x^2 + 6 * x - 7 ∧ y = -(t - 3)^2 + 2 ↔ t ≥ 3 := 
by {
    sorry
}

end NUMINAMATH_GPT_max_value_range_of_t_l905_90514


namespace NUMINAMATH_GPT_gift_cost_l905_90565

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 2 * half_cost = 28 :=
by
  sorry

end NUMINAMATH_GPT_gift_cost_l905_90565


namespace NUMINAMATH_GPT_gcd_210_162_l905_90504

-- Define the numbers
def a := 210
def b := 162

-- The proposition we need to prove: The GCD of 210 and 162 is 6
theorem gcd_210_162 : Nat.gcd a b = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_210_162_l905_90504


namespace NUMINAMATH_GPT_evaluate_expression_l905_90559

theorem evaluate_expression (m n : ℤ) (hm : m = 2) (hn : n = -3) : (m + n) ^ 2 - 2 * m * (m + n) = 5 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_evaluate_expression_l905_90559


namespace NUMINAMATH_GPT_total_oranges_picked_l905_90542

-- Defining the number of oranges picked by Mary, Jason, and Sarah
def maryOranges := 122
def jasonOranges := 105
def sarahOranges := 137

-- The theorem to prove that the total number of oranges picked is 364
theorem total_oranges_picked : maryOranges + jasonOranges + sarahOranges = 364 := by
  sorry

end NUMINAMATH_GPT_total_oranges_picked_l905_90542


namespace NUMINAMATH_GPT_functional_eq_solution_l905_90590

noncomputable def functional_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

theorem functional_eq_solution (f : ℝ → ℝ) (h : functional_solution f) :
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 :=
sorry

end NUMINAMATH_GPT_functional_eq_solution_l905_90590


namespace NUMINAMATH_GPT_songs_performed_l905_90581

variable (R L S M : ℕ)
variable (songs_total : ℕ)

def conditions := 
  R = 9 ∧ L = 6 ∧ (6 ≤ S ∧ S ≤ 9) ∧ (6 ≤ M ∧ M ≤ 9) ∧ songs_total = (R + L + S + M) / 3

theorem songs_performed (h : conditions R L S M songs_total) :
  songs_total = 9 ∨ songs_total = 10 ∨ songs_total = 11 :=
sorry

end NUMINAMATH_GPT_songs_performed_l905_90581


namespace NUMINAMATH_GPT_determine_h_l905_90554

noncomputable def h (x : ℝ) : ℝ :=
  -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3

theorem determine_h :
  (12*x^4 + 4*x^3 - 2*x + 3 + h x = 6*x^3 + 8*x^2 - 10*x + 6) ↔
  (h x = -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3) :=
by 
  sorry

end NUMINAMATH_GPT_determine_h_l905_90554


namespace NUMINAMATH_GPT_trapezoid_area_l905_90541

noncomputable def area_trapezoid : ℝ :=
  let x1 := 10
  let x2 := -10
  let y1 := 10
  let h := 10
  let a := 20  -- length of top side at y = 10
  let b := 10  -- length of lower side
  (a + b) * h / 2

theorem trapezoid_area : area_trapezoid = 150 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l905_90541


namespace NUMINAMATH_GPT_geometric_sequence_a9_l905_90515

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℝ)
variable (q : ℝ)

theorem geometric_sequence_a9
  (h_seq : geometric_sequence a q)
  (h2 : a 1 * a 4 = -32)
  (h3 : a 2 + a 3 = 4)
  (hq : ∃ n : ℤ, q = ↑n) :
  a 8 = -256 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a9_l905_90515


namespace NUMINAMATH_GPT_find_m_value_l905_90567

theorem find_m_value (m : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x y : ℝ), x + m * y + 3 - 2 * m = 0) →
  (∃ (y : ℝ), x = 0 ∧ y = -1) →
  m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l905_90567


namespace NUMINAMATH_GPT_numbers_represented_3_units_from_A_l905_90599

theorem numbers_represented_3_units_from_A (A : ℝ) (x : ℝ) (h : A = -2) : 
  abs (x + 2) = 3 ↔ x = 1 ∨ x = -5 := by
  sorry

end NUMINAMATH_GPT_numbers_represented_3_units_from_A_l905_90599


namespace NUMINAMATH_GPT_base_of_second_fraction_l905_90586

theorem base_of_second_fraction (base : ℝ) (h1 : (1/2) ^ 16 * (1/base) ^ 8 = 1 / (18 ^ 16)): base = 81 :=
sorry

end NUMINAMATH_GPT_base_of_second_fraction_l905_90586


namespace NUMINAMATH_GPT_black_dogs_count_l905_90580

def number_of_brown_dogs := 20
def number_of_white_dogs := 10
def total_number_of_dogs := 45
def number_of_black_dogs := total_number_of_dogs - (number_of_brown_dogs + number_of_white_dogs)

theorem black_dogs_count : number_of_black_dogs = 15 := by
  sorry

end NUMINAMATH_GPT_black_dogs_count_l905_90580


namespace NUMINAMATH_GPT_part1_part2_part3_l905_90588

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x + a / x + Real.log x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  1 - a / x^2 + 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  f' x a - x

theorem part1 (a : ℝ) (h : f' 1 a = 0) : a = 2 :=
  sorry

theorem part2 {a : ℝ} (h : ∀ x, 1 < x → x < 2 → f' x a ≥ 0) : a ≤ 2 :=
  sorry

theorem part3 (a : ℝ) :
  ((a > 1 → ∀ x, g x a ≠ 0) ∧ 
  (a = 1 ∨ a ≤ 0 → ∃ x, g x a = 0 ∧ ∀ y, g y a = 0 → y = x) ∧ 
  (0 < a ∧ a < 1 → ∃ x y, x ≠ y ∧ g x a = 0 ∧ g y a = 0)) :=
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l905_90588


namespace NUMINAMATH_GPT_distance_between_nails_l905_90594

theorem distance_between_nails (banner_length : ℕ) (num_nails : ℕ) (end_distance : ℕ) :
  banner_length = 20 → num_nails = 7 → end_distance = 1 → 
  (banner_length - 2 * end_distance) / (num_nails - 1) = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_distance_between_nails_l905_90594


namespace NUMINAMATH_GPT_clowns_attended_l905_90537

-- Definition of the problem's conditions
def num_children : ℕ := 30
def initial_candies : ℕ := 700
def candies_sold_per_person : ℕ := 20
def remaining_candies : ℕ := 20

-- Main theorem stating that 4 clowns attended the carousel
theorem clowns_attended (num_clowns : ℕ) (candies_left: num_clowns * candies_sold_per_person + num_children * candies_sold_per_person = initial_candies - remaining_candies) : num_clowns = 4 := by
  sorry

end NUMINAMATH_GPT_clowns_attended_l905_90537


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l905_90532

theorem solution_set_abs_inequality (x : ℝ) :
  (|2 - x| ≥ 1) ↔ (x ≤ 1 ∨ x ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l905_90532


namespace NUMINAMATH_GPT_triangle_area_r_l905_90507

theorem triangle_area_r (r : ℝ) (h₁ : 12 ≤ (r - 3) ^ (3 / 2)) (h₂ : (r - 3) ^ (3 / 2) ≤ 48) : 15 ≤ r ∧ r ≤ 19 := by
  sorry

end NUMINAMATH_GPT_triangle_area_r_l905_90507


namespace NUMINAMATH_GPT_compute_permutation_eq_4_l905_90512

def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem compute_permutation_eq_4 :
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_compute_permutation_eq_4_l905_90512


namespace NUMINAMATH_GPT_solve_inequality_l905_90510

theorem solve_inequality (x : ℝ) :
  |(3 * x - 2) / (x ^ 2 - x - 2)| > 3 ↔ (x ∈ Set.Ioo (-1) (-2 / 3) ∪ Set.Ioo (1 / 3) 4) :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l905_90510


namespace NUMINAMATH_GPT_odd_function_condition_l905_90552

-- Definitions for real numbers and absolute value function
def f (x a b : ℝ) : ℝ := (x + a) * |x + b|

-- Theorem statement
theorem odd_function_condition (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = (x + a) * |x + b|) :
  (∀ x : ℝ, f x a b = -f (-x) a b) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_condition_l905_90552


namespace NUMINAMATH_GPT_xyz_value_l905_90598

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) : 
  x * y * z = 10 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l905_90598


namespace NUMINAMATH_GPT_horse_food_per_day_l905_90576

theorem horse_food_per_day (ratio_sh : ℕ) (ratio_h : ℕ) (sheep : ℕ) (total_food : ℕ) (sheep_count : sheep = 32) (ratio : ratio_sh = 4) (ratio_horses : ratio_h = 7) (total_food_need : total_food = 12880) :
  total_food / (sheep * ratio_h / ratio_sh) = 230 :=
by
  sorry

end NUMINAMATH_GPT_horse_food_per_day_l905_90576


namespace NUMINAMATH_GPT_find_f_half_l905_90578

noncomputable def g (x : ℝ) : ℝ := 1 - 2 * x
noncomputable def f (y : ℝ) : ℝ := if y ≠ 0 then (1 - y^2) / y^2 else 0

theorem find_f_half :
  f (g (1 / 4)) = 15 :=
by
  have g_eq : g (1 / 4) = 1 / 2 := sorry
  rw [g_eq]
  have f_eq : f (1 / 2) = 15 := sorry
  exact f_eq

end NUMINAMATH_GPT_find_f_half_l905_90578


namespace NUMINAMATH_GPT_no_integer_solutions_for_inequality_l905_90569

open Int

theorem no_integer_solutions_for_inequality : ∀ x : ℤ, (x - 4) * (x - 5) < 0 → False :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solutions_for_inequality_l905_90569


namespace NUMINAMATH_GPT_eq_squares_diff_l905_90561

theorem eq_squares_diff {x y z : ℝ} :
  x = (y - z)^2 ∧ y = (x - z)^2 ∧ z = (x - y)^2 →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_eq_squares_diff_l905_90561


namespace NUMINAMATH_GPT_range_of_a_l905_90536

theorem range_of_a (a : ℝ) (h : 2 * a - 1 ≤ 11) : a < 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l905_90536


namespace NUMINAMATH_GPT_James_uses_150_sheets_of_paper_l905_90543

-- Define the conditions
def number_of_books := 2
def pages_per_book := 600
def pages_per_side := 4
def sides_per_sheet := 2

-- Statement to prove
theorem James_uses_150_sheets_of_paper :
  number_of_books * pages_per_book / (pages_per_side * sides_per_sheet) = 150 :=
by sorry

end NUMINAMATH_GPT_James_uses_150_sheets_of_paper_l905_90543


namespace NUMINAMATH_GPT_hyewon_painted_colors_l905_90531

def pentagonal_prism := 
  let num_rectangular_faces := 5 
  let num_pentagonal_faces := 2
  num_rectangular_faces + num_pentagonal_faces

theorem hyewon_painted_colors : pentagonal_prism = 7 := 
by
  sorry

end NUMINAMATH_GPT_hyewon_painted_colors_l905_90531


namespace NUMINAMATH_GPT_least_number_to_subtract_l905_90547

theorem least_number_to_subtract (n : ℕ) : (n = 5) → (5000 - n) % 37 = 0 :=
by sorry

end NUMINAMATH_GPT_least_number_to_subtract_l905_90547


namespace NUMINAMATH_GPT_number_of_turns_to_wind_tape_l905_90503

theorem number_of_turns_to_wind_tape (D δ L : ℝ) 
(hD : D = 22) 
(hδ : δ = 0.018) 
(hL : L = 90000) : 
∃ n : ℕ, n = 791 := 
sorry

end NUMINAMATH_GPT_number_of_turns_to_wind_tape_l905_90503


namespace NUMINAMATH_GPT_gain_percent_is_150_l905_90505

variable (C S : ℝ)
variable (h : 50 * C = 20 * S)

theorem gain_percent_is_150 (h : 50 * C = 20 * S) : ((S - C) / C) * 100 = 150 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_is_150_l905_90505


namespace NUMINAMATH_GPT_gcd_119_34_l905_90570

theorem gcd_119_34 : Nat.gcd 119 34 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_119_34_l905_90570


namespace NUMINAMATH_GPT_cafeteria_pies_l905_90582

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h1 : initial_apples = 96)
  (h2 : handed_out_apples = 42)
  (h3 : apples_per_pie = 6) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := by
  sorry

end NUMINAMATH_GPT_cafeteria_pies_l905_90582


namespace NUMINAMATH_GPT_arithmetic_expression_equality_l905_90585

theorem arithmetic_expression_equality :
  15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_expression_equality_l905_90585


namespace NUMINAMATH_GPT_remainder_hx10_div_hx_l905_90574

noncomputable def h (x : ℕ) := x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_hx10_div_hx (x : ℕ) : (h x ^ 10) % (h x) = 7 := by
  sorry

end NUMINAMATH_GPT_remainder_hx10_div_hx_l905_90574


namespace NUMINAMATH_GPT_intersection_is_singleton_zero_l905_90553

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

-- Define the theorem to be proved
theorem intersection_is_singleton_zero : M ∩ N = {0} :=
by
  -- Proof is provided by the steps above but not needed here
  sorry

end NUMINAMATH_GPT_intersection_is_singleton_zero_l905_90553


namespace NUMINAMATH_GPT_rectangle_width_l905_90525

-- Definitions and Conditions
variables (L W : ℕ)

-- Condition 1: The perimeter of the rectangle is 16 cm
def perimeter_eq : Prop := 2 * (L + W) = 16

-- Condition 2: The width is 2 cm longer than the length
def width_eq : Prop := W = L + 2

-- Proof Statement: Given the above conditions, the width of the rectangle is 5 cm
theorem rectangle_width (h1 : perimeter_eq L W) (h2 : width_eq L W) : W = 5 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l905_90525


namespace NUMINAMATH_GPT_jack_time_to_school_l905_90563

noncomputable def dave_speed : ℚ := 8000 -- cm/min
noncomputable def distance_to_school : ℚ := 160000 -- cm
noncomputable def jack_speed : ℚ := 7650 -- cm/min
noncomputable def jack_start_delay : ℚ := 10 -- min

theorem jack_time_to_school : (distance_to_school / jack_speed) - jack_start_delay = 10.92 :=
by
  sorry

end NUMINAMATH_GPT_jack_time_to_school_l905_90563


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l905_90528

def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := -2
  let c := -7
  discriminant a b c > 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l905_90528


namespace NUMINAMATH_GPT_at_least_one_nonzero_l905_90596

theorem at_least_one_nonzero (a b : ℝ) : a^2 + b^2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end NUMINAMATH_GPT_at_least_one_nonzero_l905_90596


namespace NUMINAMATH_GPT_taimour_paint_time_l905_90508

theorem taimour_paint_time (T : ℝ) :
  (1 / T + 2 / T) * 7 = 1 → T = 21 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_taimour_paint_time_l905_90508


namespace NUMINAMATH_GPT_find_hours_l905_90513

theorem find_hours (x : ℕ) (h : (14 + 10 + 13 + 9 + 12 + 11 + x) / 7 = 12) : x = 15 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_find_hours_l905_90513


namespace NUMINAMATH_GPT_multiply_5915581_7907_l905_90587

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := 
by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_multiply_5915581_7907_l905_90587


namespace NUMINAMATH_GPT_jack_jill_meet_distance_l905_90555

theorem jack_jill_meet_distance : 
  ∀ (total_distance : ℝ) (uphill_distance : ℝ) (headstart : ℝ) 
  (jack_speed_up : ℝ) (jack_speed_down : ℝ)
  (jill_speed_up : ℝ) (jill_speed_down : ℝ), 
  total_distance = 12 → 
  uphill_distance = 6 → 
  headstart = 1 / 4 → 
  jack_speed_up = 12 → 
  jack_speed_down = 18 → 
  jill_speed_up = 14 → 
  jill_speed_down = 20 → 
  ∃ meet_position : ℝ, meet_position = 15.75 :=
by
  sorry

end NUMINAMATH_GPT_jack_jill_meet_distance_l905_90555


namespace NUMINAMATH_GPT_decimal_to_fraction_l905_90509

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end NUMINAMATH_GPT_decimal_to_fraction_l905_90509


namespace NUMINAMATH_GPT_find_x_value_l905_90584

variable {x : ℝ}

def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = (k * b.1, k * b.2)

theorem find_x_value (h : opposite_directions (x, 1) (4, x)) : x = -2 :=
sorry

end NUMINAMATH_GPT_find_x_value_l905_90584


namespace NUMINAMATH_GPT_emily_purchased_9_wall_prints_l905_90560

/-
  Given the following conditions:
  - cost_of_each_pair_of_curtains = 30
  - num_of_pairs_of_curtains = 2
  - installation_cost = 50
  - cost_of_each_wall_print = 15
  - total_order_cost = 245

  Prove that Emily purchased 9 wall prints
-/
noncomputable def num_wall_prints_purchased 
  (cost_of_each_pair_of_curtains : ℝ) 
  (num_of_pairs_of_curtains : ℝ) 
  (installation_cost : ℝ) 
  (cost_of_each_wall_print : ℝ) 
  (total_order_cost : ℝ) 
  : ℝ :=
  (total_order_cost - (num_of_pairs_of_curtains * cost_of_each_pair_of_curtains + installation_cost)) / cost_of_each_wall_print

theorem emily_purchased_9_wall_prints
  (cost_of_each_pair_of_curtains : ℝ := 30) 
  (num_of_pairs_of_curtains : ℝ := 2) 
  (installation_cost : ℝ := 50) 
  (cost_of_each_wall_print : ℝ := 15) 
  (total_order_cost : ℝ := 245) :
  num_wall_prints_purchased cost_of_each_pair_of_curtains num_of_pairs_of_curtains installation_cost cost_of_each_wall_print total_order_cost = 9 :=
sorry

end NUMINAMATH_GPT_emily_purchased_9_wall_prints_l905_90560
