import Mathlib

namespace sqrt_cos_sin_relation_l1065_106578

variable {a b c θ : ℝ}

theorem sqrt_cos_sin_relation 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * (Real.cos θ) ^ 2 + b * (Real.sin θ) ^ 2 < c) :
  Real.sqrt a * (Real.cos θ) ^ 2 + Real.sqrt b * (Real.sin θ) ^ 2 < Real.sqrt c :=
sorry

end sqrt_cos_sin_relation_l1065_106578


namespace jack_walked_distance_l1065_106571

theorem jack_walked_distance (time_in_hours : ℝ) (rate : ℝ) (expected_distance : ℝ) : 
  time_in_hours = 1 + 15 / 60 ∧ 
  rate = 6.4 →
  expected_distance = 8 → 
  rate * time_in_hours = expected_distance :=
by 
  intros h
  sorry

end jack_walked_distance_l1065_106571


namespace division_addition_problem_l1065_106539

theorem division_addition_problem :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end division_addition_problem_l1065_106539


namespace find_blue_balls_l1065_106543

/-- 
Given the conditions that a bag contains:
- 5 red balls
- B blue balls
- 2 green balls
And the probability of picking 2 red balls at random is 0.1282051282051282,
prove that the number of blue balls (B) is 6.
--/

theorem find_blue_balls (B : ℕ) (h : 0.1282051282051282 = (10 : ℚ) / (↑((7 + B) * (6 + B)) / 2)) : B = 6 := 
by sorry

end find_blue_balls_l1065_106543


namespace volume_of_prism_l1065_106592

-- Given dimensions a, b, and c, with the following conditions:
variables (a b c : ℝ)
axiom ab_eq_30 : a * b = 30
axiom ac_eq_40 : a * c = 40
axiom bc_eq_60 : b * c = 60

-- The volume of the prism is given by:
theorem volume_of_prism : a * b * c = 120 * Real.sqrt 5 :=
by
  sorry

end volume_of_prism_l1065_106592


namespace combined_resistance_parallel_l1065_106535

theorem combined_resistance_parallel (R1 R2 R3 : ℝ) (r : ℝ) (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6) :
  (1 / r) = (1 / R1) + (1 / R2) + (1 / R3) → r = 15 / 13 :=
by
  sorry

end combined_resistance_parallel_l1065_106535


namespace find_m_l1065_106593

-- Define the given equations of the lines
def line1 (m : ℝ) : ℝ × ℝ → Prop := fun p => (3 + m) * p.1 - 4 * p.2 = 5 - 3 * m
def line2 : ℝ × ℝ → Prop := fun p => 2 * p.1 - p.2 = 8

-- Define the condition for parallel lines based on the given equations
def are_parallel (m : ℝ) : Prop := (3 + m) / 4 = 2

-- The main theorem stating the value of m
theorem find_m (m : ℝ) (h1 : ∀ p : ℝ × ℝ, line1 m p) (h2 : ∀ p : ℝ × ℝ, line2 p) (h_parallel : are_parallel m) : m = 5 :=
sorry

end find_m_l1065_106593


namespace find_initial_amount_l1065_106540

theorem find_initial_amount
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1050)
  (hR : R = 8)
  (hT : T = 5) :
  P = 750 :=
by
  have hSI : P * R * T / 100 = 1050 - P := sorry
  have hFormulaSimplified : P * 0.4 = 1050 - P := sorry
  have hFinal : P * 1.4 = 1050 := sorry
  exact sorry

end find_initial_amount_l1065_106540


namespace trigonometric_expression_value_l1065_106583

theorem trigonometric_expression_value :
  4 * Real.cos (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) -
  Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 3 / 4 := sorry

end trigonometric_expression_value_l1065_106583


namespace mean_of_combined_set_is_52_over_3_l1065_106551

noncomputable def mean_combined_set : ℚ := 
  let mean_set1 := 10
  let size_set1 := 4
  let mean_set2 := 21
  let size_set2 := 8
  let sum_set1 := mean_set1 * size_set1
  let sum_set2 := mean_set2 * size_set2
  let total_sum := sum_set1 + sum_set2
  let combined_size := size_set1 + size_set2
  let combined_mean := total_sum / combined_size
  combined_mean

theorem mean_of_combined_set_is_52_over_3 :
  mean_combined_set = 52 / 3 :=
by
  sorry

end mean_of_combined_set_is_52_over_3_l1065_106551


namespace find_missing_number_l1065_106588

theorem find_missing_number (x : ℕ) (h : (1 + x + 23 + 24 + 25 + 26 + 27 + 2) / 8 = 20) : x = 32 := 
by sorry

end find_missing_number_l1065_106588


namespace largest_multiple_of_12_neg_gt_neg_150_l1065_106526

theorem largest_multiple_of_12_neg_gt_neg_150 : ∃ m : ℤ, (m % 12 = 0) ∧ (-m > -150) ∧ ∀ n : ℤ, (n % 12 = 0) ∧ (-n > -150) → n ≤ m := sorry

end largest_multiple_of_12_neg_gt_neg_150_l1065_106526


namespace converse_of_prop1_true_l1065_106504

theorem converse_of_prop1_true
  (h1 : ∀ {x : ℝ}, x^2 - 3 * x + 2 = 0 → x = 1 ∨ x = 2)
  (h2 : ∀ {x : ℝ}, -2 ≤ x ∧ x < 3 → (x - 2) * (x - 3) ≤ 0)
  (h3 : ∀ {x y : ℝ}, x = 0 ∧ y = 0 → x^2 + y^2 = 0)
  (h4 : ∀ {x y : ℕ}, x > 0 ∧ y > 0 ∧ (x + y) % 2 = 1 → (x % 2 = 1 ∧ y % 2 = 0) ∨ (x % 2 = 0 ∧ y % 2 = 1)) :
  (∀ {x : ℝ}, x = 1 ∨ x = 2 → x^2 - 3 * x + 2 = 0) :=
by
  sorry

end converse_of_prop1_true_l1065_106504


namespace solve_for_C_and_D_l1065_106590

theorem solve_for_C_and_D (C D : ℚ) (h1 : 2 * C + 3 * D + 4 = 31) (h2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 :=
by
  sorry

end solve_for_C_and_D_l1065_106590


namespace complement_of_A_in_reals_l1065_106573

open Set

theorem complement_of_A_in_reals :
  (compl {x : ℝ | (x - 1) / (x - 2) ≥ 0}) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end complement_of_A_in_reals_l1065_106573


namespace vertical_asymptotes_sum_l1065_106506

theorem vertical_asymptotes_sum : 
  let f (x : ℝ) := (6 * x^2 + 1) / (4 * x^2 + 6 * x + 3)
  let den := 4 * x^2 + 6 * x + 3
  let p := -(3 / 2)
  let q := -(1 / 2)
  (den = 0) → (p + q = -2) :=
by
  sorry

end vertical_asymptotes_sum_l1065_106506


namespace angle_of_inclination_vert_line_l1065_106550

theorem angle_of_inclination_vert_line (x : ℝ) (h : x = -1) : 
  ∃ θ : ℝ, θ = 90 := 
by
  sorry

end angle_of_inclination_vert_line_l1065_106550


namespace parabola_line_unique_eq_l1065_106508

noncomputable def parabola_line_equation : Prop :=
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 4 * A.1) ∧ (B.2^2 = 4 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) ∧ ((A.2 + B.2) / 2 = 2) ∧
    ∀ x y, (y - 2 = 1 * (x - 2)) → (x - y = 0)

theorem parabola_line_unique_eq : parabola_line_equation :=
  sorry

end parabola_line_unique_eq_l1065_106508


namespace contractor_daily_wage_l1065_106568

theorem contractor_daily_wage 
  (total_days : ℕ)
  (daily_wage : ℝ)
  (fine_per_absence : ℝ)
  (total_earned : ℝ)
  (absent_days : ℕ)
  (H1 : total_days = 30)
  (H2 : fine_per_absence = 7.5)
  (H3 : total_earned = 555.0)
  (H4 : absent_days = 6)
  (H5 : total_earned = daily_wage * (total_days - absent_days) - fine_per_absence * absent_days) :
  daily_wage = 25 := by
  sorry

end contractor_daily_wage_l1065_106568


namespace inequality_proof_l1065_106533

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) :=
by
  sorry

end inequality_proof_l1065_106533


namespace jar_filling_fraction_l1065_106514

theorem jar_filling_fraction (C1 C2 C3 W : ℝ)
  (h1 : W = (1/7) * C1)
  (h2 : W = (2/9) * C2)
  (h3 : W = (3/11) * C3)
  (h4 : C3 > C1 ∧ C3 > C2) :
  (3 * W) = (9 / 11) * C3 :=
by sorry

end jar_filling_fraction_l1065_106514


namespace sum_C_D_equals_seven_l1065_106569

def initial_grid : Matrix (Fin 4) (Fin 4) (Option Nat) :=
  ![ ![ some 1, none, none, none ],
     ![ none, some 2, none, none ],
     ![ none, none, none, none ],
     ![ none, none, none, some 4 ] ]

def valid_grid (grid : Matrix (Fin 4) (Fin 4) (Option Nat)) : Prop :=
  ∀ i j, grid i j ≠ none →
    (∀ k, k ≠ j → grid i k ≠ grid i j) ∧ 
    (∀ k, k ≠ i → grid k j ≠ grid i j)

theorem sum_C_D_equals_seven :
  ∃ (C D : Nat), C + D = 7 ∧ valid_grid initial_grid :=
sorry

end sum_C_D_equals_seven_l1065_106569


namespace sasha_skated_distance_l1065_106524

theorem sasha_skated_distance (d total_distance v : ℝ)
  (h1 : total_distance = 3300)
  (h2 : v > 0)
  (h3 : d = 3 * v * (total_distance / (3 * v + 2 * v))) :
  d = 1100 :=
by
  sorry

end sasha_skated_distance_l1065_106524


namespace geometric_sequence_second_term_l1065_106530

theorem geometric_sequence_second_term (a r : ℝ) (h1 : a * r ^ 2 = 5) (h2 : a * r ^ 4 = 45) :
  a * r = 5 / 3 :=
by
  sorry

end geometric_sequence_second_term_l1065_106530


namespace kathleen_allowance_l1065_106553

theorem kathleen_allowance (x : ℝ) :
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  percentage_increase = 150 → x = 2 :=
by
  -- Definitions and conditions setup
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  intros h
  -- Skipping the proof
  sorry

end kathleen_allowance_l1065_106553


namespace extra_interest_amount_l1065_106528

def principal : ℝ := 15000
def rate1 : ℝ := 0.15
def rate2 : ℝ := 0.12
def time : ℕ := 2

theorem extra_interest_amount :
  principal * (rate1 - rate2) * time = 900 := by
  sorry

end extra_interest_amount_l1065_106528


namespace sqrt_eq_l1065_106591

noncomputable def sqrt_22500 := 150

theorem sqrt_eq (h : sqrt_22500 = 150) : Real.sqrt 0.0225 = 0.15 :=
sorry

end sqrt_eq_l1065_106591


namespace max_colors_4x4_grid_l1065_106548

def cell := (Fin 4) × (Fin 4)
def color := Fin 8

def valid_coloring (f : cell → color) : Prop :=
∀ c1 c2 : color, (c1 ≠ c2) →
(∃ i : Fin 4, ∃ j1 j2 : Fin 4, j1 ≠ j2 ∧ f (i, j1) = c1 ∧ f (i, j2) = c2) ∨ 
(∃ j : Fin 4, ∃ i1 i2 : Fin 4, i1 ≠ i2 ∧ f (i1, j) = c1 ∧ f (i2, j) = c2)

theorem max_colors_4x4_grid : ∃ (f : cell → color), valid_coloring f :=
sorry

end max_colors_4x4_grid_l1065_106548


namespace required_range_of_a_l1065_106574

variable (a : ℝ) (f : ℝ → ℝ)
def function_increasing_on (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, DifferentiableAt ℝ f x ∧ (deriv f x) ≥ 0

theorem required_range_of_a (h : function_increasing_on (fun x => a * Real.log x + x) a (Set.Icc 2 3)) :
  a ≥ -2 :=
sorry

end required_range_of_a_l1065_106574


namespace smallest_int_square_eq_3x_plus_72_l1065_106541

theorem smallest_int_square_eq_3x_plus_72 :
  ∃ x : ℤ, x^2 = 3 * x + 72 ∧ (∀ y : ℤ, y^2 = 3 * y + 72 → x ≤ y) :=
sorry

end smallest_int_square_eq_3x_plus_72_l1065_106541


namespace problem_solution_l1065_106519

theorem problem_solution
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2007)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2007)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2007)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1003 := 
sorry

end problem_solution_l1065_106519


namespace integer_solution_of_inequality_l1065_106595

theorem integer_solution_of_inequality :
  ∀ (x : ℤ), 0 < (x - 1 : ℚ) * (x - 1) / (x + 1) ∧ (x - 1) * (x - 1) / (x + 1) < 1 →
  x > -1 ∧ x ≠ 1 ∧ x < 3 → 
  x = 2 :=
by
  sorry

end integer_solution_of_inequality_l1065_106595


namespace shaded_triangle_probability_l1065_106561

noncomputable def total_triangles : ℕ := 5
noncomputable def shaded_triangles : ℕ := 2
noncomputable def probability_shaded : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : probability_shaded = 2 / 5 :=
by
  sorry

end shaded_triangle_probability_l1065_106561


namespace largest_angle_of_triangle_l1065_106522

theorem largest_angle_of_triangle 
  (α β γ : ℝ) 
  (h1 : α = 60) 
  (h2 : β = 70) 
  (h3 : α + β + γ = 180) : 
  max α (max β γ) = 70 := 
by 
  sorry

end largest_angle_of_triangle_l1065_106522


namespace tank_capacity_l1065_106564

theorem tank_capacity :
  (∃ (C : ℕ), ∀ (leak_rate inlet_rate net_rate : ℕ),
    leak_rate = C / 6 ∧
    inlet_rate = 6 * 60 ∧
    net_rate = C / 12 ∧
    inlet_rate - leak_rate = net_rate → C = 1440) :=
sorry

end tank_capacity_l1065_106564


namespace closest_approx_w_l1065_106576

noncomputable def w : ℝ := ((69.28 * 123.57 * 0.004) - (42.67 * 3.12)) / (0.03 * 8.94 * 1.25)

theorem closest_approx_w : |w + 296.073| < 0.001 :=
by
  sorry

end closest_approx_w_l1065_106576


namespace solve_inequality_l1065_106537

theorem solve_inequality (a x : ℝ) :
  (a = 0 → x < 1) ∧
  (a ≠ 0 → ((a > 0 → (a-1)/a < x ∧ x < 1) ∧
            (a < 0 → (x < 1 ∨ x > (a-1)/a)))) :=
by
  sorry

end solve_inequality_l1065_106537


namespace permutations_with_k_in_first_position_l1065_106511

noncomputable def numberOfPermutationsWithKInFirstPosition (N k : ℕ) (h : k < N) : ℕ :=
  (2 : ℕ)^(N-1)

theorem permutations_with_k_in_first_position (N k : ℕ) (h : k < N) :
  numberOfPermutationsWithKInFirstPosition N k h = (2 : ℕ)^(N-1) :=
sorry

end permutations_with_k_in_first_position_l1065_106511


namespace cone_base_radius_l1065_106520

/-- Given a semicircular piece of paper with a diameter of 2 cm is used to construct the 
  lateral surface of a cone, prove that the radius of the base of the cone is 0.5 cm. --/
theorem cone_base_radius (d : ℝ) (arc_length : ℝ) (circumference : ℝ) (r : ℝ)
  (h₀ : d = 2)
  (h₁ : arc_length = (1 / 2) * d * Real.pi)
  (h₂ : circumference = arc_length)
  (h₃ : r = circumference / (2 * Real.pi)) :
  r = 0.5 :=
by
  sorry

end cone_base_radius_l1065_106520


namespace isosceles_triangle_angles_l1065_106558

noncomputable 
def is_triangle_ABC_isosceles (A B C : ℝ) (alpha beta : ℝ) (AB AC : ℝ) 
  (h1 : AB = AC) (h2 : alpha = 2 * beta) : Prop :=
  180 - 3 * beta = C ∧ C / 2 = 90 - 1.5 * beta

theorem isosceles_triangle_angles (A B C C1 C2 : ℝ) (alpha beta : ℝ) (AB AC : ℝ)
  (h1 : AB = AC) (h2 : alpha = 2 * beta) :
  (180 - 3 * beta) / 2 = 90 - 1.5 * beta :=
by sorry

end isosceles_triangle_angles_l1065_106558


namespace number_of_correct_conclusions_is_two_l1065_106567

section AnalogicalReasoning
  variable (a b c : ℝ) (x y : ℂ)

  -- Condition 1: The analogy for distributive property over addition in ℝ and division
  def analogy1 : (c ≠ 0) → ((a + b) * c = a * c + b * c) → (a + b) / c = a / c + b / c := by
    sorry

  -- Condition 2: The analogy for equality of real and imaginary parts in ℂ
  def analogy2 : (x - y = 0) → x = y := by
    sorry

  -- Theorem stating that the number of correct conclusions is 2
  theorem number_of_correct_conclusions_is_two : 2 = 2 := by
    -- which implies that analogy1 and analogy2 are valid, and the other two analogies are not
    sorry

end AnalogicalReasoning

end number_of_correct_conclusions_is_two_l1065_106567


namespace lines_parallel_l1065_106529

-- Definitions based on conditions
variable (line1 line2 : ℝ → ℝ → Prop) -- Assuming lines as relations for simplicity
variable (plane : ℝ → ℝ → ℝ → Prop) -- Assuming plane as a relation for simplicity

-- Condition: Both lines are perpendicular to the same plane
def perpendicular_to_plane (line : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ (x y z : ℝ), plane x y z → line x y

axiom line1_perpendicular : perpendicular_to_plane line1 plane
axiom line2_perpendicular : perpendicular_to_plane line2 plane

-- Theorem: Both lines are parallel
theorem lines_parallel : ∀ (line1 line2 : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop),
  (perpendicular_to_plane line1 plane) →
  (perpendicular_to_plane line2 plane) →
  (∀ x y : ℝ, line1 x y → line2 x y) := sorry

end lines_parallel_l1065_106529


namespace odd_function_behavior_l1065_106572

-- Define that f is odd
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Define f for x > 0
def f_pos (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → (f x = (Real.log x / Real.log 2) - 2 * x)

-- Prove that for x < 0, f(x) == -log₂(-x) - 2x
theorem odd_function_behavior (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_pos : f_pos f) :
  ∀ x, x < 0 → f x = -((Real.log (-x)) / (Real.log 2)) - 2 * x := 
by
  sorry -- proof goes here

end odd_function_behavior_l1065_106572


namespace student_score_l1065_106597

theorem student_score
    (total_questions : ℕ)
    (correct_responses : ℕ)
    (grading_method : ℕ → ℕ → ℕ)
    (h1 : total_questions = 100)
    (h2 : correct_responses = 92)
    (h3 : grading_method = λ correct incorrect => correct - 2 * incorrect) :
  grading_method correct_responses (total_questions - correct_responses) = 76 :=
by
  -- proof would be here, but is skipped
  sorry

end student_score_l1065_106597


namespace simultaneous_eq_solvable_l1065_106570

theorem simultaneous_eq_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = m * x + 4 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_eq_solvable_l1065_106570


namespace rahul_deepak_present_ages_l1065_106547

theorem rahul_deepak_present_ages (R D : ℕ) 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26)
  (h3 : D + 6 = 1/2 * (R + (R + 6)))
  (h4 : (R + 11) + (D + 11) = 59) 
  : R = 20 ∧ D = 17 :=
sorry

end rahul_deepak_present_ages_l1065_106547


namespace find_a5_a6_l1065_106566

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

-- Given conditions
axiom h1 : a 1 + a 2 = 5
axiom h2 : a 3 + a 4 = 7

-- Arithmetic sequence property
axiom arith_seq : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)

-- The statement we want to prove
theorem find_a5_a6 : a 5 + a 6 = 9 :=
sorry

end ArithmeticSequence

end find_a5_a6_l1065_106566


namespace toys_left_after_two_weeks_l1065_106510

theorem toys_left_after_two_weeks
  (initial_stock : ℕ)
  (sold_first_week : ℕ)
  (sold_second_week : ℕ)
  (total_stock : initial_stock = 83)
  (first_week_sales : sold_first_week = 38)
  (second_week_sales : sold_second_week = 26) :
  initial_stock - (sold_first_week + sold_second_week) = 19 :=
by
  sorry

end toys_left_after_two_weeks_l1065_106510


namespace chess_tournament_distribution_l1065_106545

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l1065_106545


namespace eight_row_triangle_pieces_l1065_106586

def unit_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

def connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem eight_row_triangle_pieces : unit_rods 8 + connectors 9 = 153 :=
by
  sorry

end eight_row_triangle_pieces_l1065_106586


namespace complex_quadrant_l1065_106560

theorem complex_quadrant (i : ℂ) (h_imag : i = Complex.I) :
  let z := (1 + i)⁻¹
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l1065_106560


namespace total_clients_l1065_106565

theorem total_clients (V K B N : Nat) (hV : V = 7) (hK : K = 8) (hB : B = 3) (hN : N = 18) :
    V + K - B + N = 30 := by
  sorry

end total_clients_l1065_106565


namespace propositions_imply_implication_l1065_106577

theorem propositions_imply_implication (p q r : Prop) :
  ( ((p ∧ q ∧ ¬r) → ((p ∧ q) → r) = False) ∧ 
    ((¬p ∧ q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((p ∧ ¬q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((¬p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r) = True) ) → 
  ( (∀ (x : ℕ), x = 3) ) :=
by
  sorry

end propositions_imply_implication_l1065_106577


namespace min_value_fraction_solve_inequality_l1065_106500

-- Part 1
theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (f : ℝ → ℝ)
  (h3 : f 1 = 2) (h4 : ∀ x, f x = a * x^2 + b * x + 1) :
  (a + b = 1) → (∃ z, z = (1 / a + 4 / b) ∧ z = 9) := 
by {
  sorry
}

-- Part 2
theorem solve_inequality (a : ℝ) (x : ℝ) (h1 : b = -a - 1) (f : ℝ → ℝ)
  (h2 : ∀ x, f x = a * x^2 + b * x + 1) :
  (f x ≤ 0) → 
  (if a = 0 then 
      {x | x ≥ 1}
  else if a > 0 then
      if a = 1 then 
          {x | x = 1}
      else if 0 < a ∧ a < 1 then 
          {x | 1 ≤ x ∧ x ≤ 1 / a}
      else 
          {x | 1 / a ≤ x ∧ x ≤ 1}
  else 
      {x | x ≥ 1 ∨ x ≤ 1 / a}) :=
by {
  sorry
}

end min_value_fraction_solve_inequality_l1065_106500


namespace length_of_BC_l1065_106579

def triangle_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 20

def triangle_area (a b : ℝ) : Prop :=
  (1/2) * a * b * (Real.sqrt 3 / 2) = 10

theorem length_of_BC (a b c : ℝ) (h1 : triangle_perimeter a b c) (h2 : triangle_area a b) : c = 7 :=
  sorry

end length_of_BC_l1065_106579


namespace train_length_equals_750_l1065_106503

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end train_length_equals_750_l1065_106503


namespace x_lt_2_necessary_not_sufficient_x_sq_lt_4_l1065_106552

theorem x_lt_2_necessary_not_sufficient_x_sq_lt_4 (x : ℝ) :
  (x < 2) → (x^2 < 4) ∧ ¬((x^2 < 4) → (x < 2)) :=
by
  sorry

end x_lt_2_necessary_not_sufficient_x_sq_lt_4_l1065_106552


namespace lights_on_fourth_tier_l1065_106507

def number_lights_topmost_tier (total_lights : ℕ) : ℕ :=
  total_lights / 127

def number_lights_tier (tier : ℕ) (lights_topmost : ℕ) : ℕ :=
  2^(tier - 1) * lights_topmost

theorem lights_on_fourth_tier (total_lights : ℕ) (H : total_lights = 381) : number_lights_tier 4 (number_lights_topmost_tier total_lights) = 24 :=
by
  rw [H]
  sorry

end lights_on_fourth_tier_l1065_106507


namespace log_sum_l1065_106556

theorem log_sum : (Real.log 0.01 / Real.log 10) + (Real.log 16 / Real.log 2) = 2 := by
  sorry

end log_sum_l1065_106556


namespace moles_of_CO2_required_l1065_106598

theorem moles_of_CO2_required (n_H2O n_H2CO3 : ℕ) (h1 : n_H2O = n_H2CO3) (h2 : n_H2O = 2): 
  (n_H2O = 2) → (∃ n_CO2 : ℕ, n_CO2 = n_H2O) :=
by
  sorry

end moles_of_CO2_required_l1065_106598


namespace quadratic_value_at_sum_of_roots_is_five_l1065_106599

noncomputable def quadratic_func (a b x : ℝ) : ℝ := a * x^2 + b * x + 5

theorem quadratic_value_at_sum_of_roots_is_five
  (a b x₁ x₂ : ℝ)
  (hA : quadratic_func a b x₁ = 2023)
  (hB : quadratic_func a b x₂ = 2023)
  (ha : a ≠ 0) :
  quadratic_func a b (x₁ + x₂) = 5 :=
sorry

end quadratic_value_at_sum_of_roots_is_five_l1065_106599


namespace sin_300_eq_neg_sqrt3_div_2_l1065_106502

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l1065_106502


namespace minimize_slope_at_one_l1065_106581

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * x^2 - (1 / (a * x))

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  4 * a * x - (1 / (a * x^2))

noncomputable def slope_at_one (a : ℝ) : ℝ :=
  f_deriv a 1

theorem minimize_slope_at_one : ∀ a : ℝ, a > 0 → slope_at_one a ≥ 4 ∧ (slope_at_one a = 4 ↔ a = 1 / 2) :=
by 
  sorry

end minimize_slope_at_one_l1065_106581


namespace sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l1065_106517

variable (α β : ℝ)
variable (hα : α < π/2) (hβ : β < π/2) -- acute angles
variable (h1 : Real.cos (α + π/6) = 3/5)
variable (h2 : Real.cos (α + β) = -Real.sqrt 5 / 5)

theorem sin_2alpha_plus_pi_over_3 :
  Real.sin (2 * α + π/3) = 24 / 25 :=
by
  sorry

theorem cos_beta_minus_pi_over_6 :
  Real.cos (β - π/6) = Real.sqrt 5 / 5 :=
by
  sorry

end sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l1065_106517


namespace calculate_speed_l1065_106513

variable (time : ℝ) (distance : ℝ)

theorem calculate_speed (h_time : time = 5) (h_distance : distance = 500) : 
  distance / time = 100 := 
by 
  sorry

end calculate_speed_l1065_106513


namespace solution_set_empty_iff_a_in_range_l1065_106549

theorem solution_set_empty_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬ (2 * x^2 + a * x + 2 < 0)) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end solution_set_empty_iff_a_in_range_l1065_106549


namespace trapezoid_base_ratio_l1065_106596

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end trapezoid_base_ratio_l1065_106596


namespace min_questions_any_three_cards_min_questions_consecutive_three_cards_l1065_106546

-- Definitions for numbers on cards and necessary questions
variables (n : ℕ) (h_n : n > 3)
  (cards : Fin n → ℤ)
  (h_cards_range : ∀ i, cards i = 1 ∨ cards i = -1)

-- Case (a): Product of any three cards
theorem min_questions_any_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (∃ (k : ℕ), n = 3 * k + 1 ∧ p = k + 1) ∨
  (∃ (k : ℕ), n = 3 * k + 2 ∧ p = k + 2) :=
sorry
  
-- Case (b): Product of any three consecutive cards
theorem min_questions_consecutive_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (¬(∃ (k : ℕ), n = 3 * k) ∧ p = n) :=
sorry

end min_questions_any_three_cards_min_questions_consecutive_three_cards_l1065_106546


namespace Z_evaluation_l1065_106538

def Z (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem Z_evaluation : Z 5 3 = 19 := by
  sorry

end Z_evaluation_l1065_106538


namespace farmer_initial_days_l1065_106589

theorem farmer_initial_days 
  (x : ℕ) 
  (plan_daily : ℕ) 
  (actual_daily : ℕ) 
  (extra_days : ℕ) 
  (left_area : ℕ) 
  (total_area : ℕ)
  (h1 : plan_daily = 120) 
  (h2 : actual_daily = 85) 
  (h3 : extra_days = 2) 
  (h4 : left_area = 40) 
  (h5 : total_area = 720): 
  85 * (x + extra_days) + left_area = total_area → x = 6 :=
by
  intros h
  sorry

end farmer_initial_days_l1065_106589


namespace third_number_is_60_l1065_106523

theorem third_number_is_60 (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 80 + 15) / 3 + 5 → x = 60 :=
by
  intro h
  sorry

end third_number_is_60_l1065_106523


namespace ratio_tin_copper_in_b_l1065_106575

variable (L_a T_a T_b C_b : ℝ)

-- Conditions
axiom h1 : 170 + 250 = 420
axiom h2 : L_a / T_a = 1 / 3
axiom h3 : T_a + T_b = 221.25
axiom h4 : T_a + L_a = 170
axiom h5 : T_b + C_b = 250

-- Target
theorem ratio_tin_copper_in_b (h1 : 170 + 250 = 420) (h2 : L_a / T_a = 1 / 3)
  (h3 : T_a + T_b = 221.25) (h4 : T_a + L_a = 170) (h5 : T_b + C_b = 250) :
  T_b / C_b = 3 / 5 := by
  sorry

end ratio_tin_copper_in_b_l1065_106575


namespace circle_radius_tangent_to_ellipse_l1065_106518

theorem circle_radius_tangent_to_ellipse (r : ℝ) :
  (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 → x^2 + 4*y^2 = 8) ↔ r = (Real.sqrt 6) / 2 :=
by
  sorry

end circle_radius_tangent_to_ellipse_l1065_106518


namespace David_fewer_crunches_l1065_106512

-- Definitions as per conditions.
def Zachary_crunches := 62
def David_crunches := 45

-- Proof statement for how many fewer crunches David did compared to Zachary.
theorem David_fewer_crunches : Zachary_crunches - David_crunches = 17 := by
  -- Proof details would go here, but we skip them with 'sorry'.
  sorry

end David_fewer_crunches_l1065_106512


namespace father_current_age_is_85_l1065_106587

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l1065_106587


namespace remainder_when_dividing_25197631_by_17_l1065_106557

theorem remainder_when_dividing_25197631_by_17 :
  25197631 % 17 = 10 :=
by
  sorry

end remainder_when_dividing_25197631_by_17_l1065_106557


namespace compare_logs_l1065_106542

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem compare_logs : a > b ∧ b > c := by
  -- Proof will be written here, currently placeholder
  sorry

end compare_logs_l1065_106542


namespace sum_first_8_even_numbers_is_72_l1065_106554

theorem sum_first_8_even_numbers_is_72 : (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16) = 72 :=
by
  sorry

end sum_first_8_even_numbers_is_72_l1065_106554


namespace sinA_mul_sinC_eq_three_fourths_l1065_106585
open Real

-- Definitions based on conditions
def angles_form_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

def sides_form_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- The theorem to prove
theorem sinA_mul_sinC_eq_three_fourths
  (A B C a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_angles : A + B + C = π)
  (h_angles_arithmetic : angles_form_arithmetic_sequence A B C)
  (h_sides_geometric : sides_form_geometric_sequence a b c) :
  sin A * sin C = 3 / 4 :=
sorry

end sinA_mul_sinC_eq_three_fourths_l1065_106585


namespace slices_per_person_l1065_106544

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end slices_per_person_l1065_106544


namespace algebraic_expression_value_l1065_106527

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = 9) : a^2 - 3 * a * b + b^2 = 19 :=
sorry

end algebraic_expression_value_l1065_106527


namespace quadratic_roots_relation_l1065_106584

theorem quadratic_roots_relation (a b s p : ℝ) (h : a^2 + b^2 = 15) (h1 : s = a + b) (h2 : p = a * b) : s^2 - 2 * p = 15 :=
by sorry

end quadratic_roots_relation_l1065_106584


namespace find_k_l1065_106525

theorem find_k (σ μ : ℝ) (hσ : σ = 2) (hμ : μ = 55) :
  ∃ k : ℝ, μ - k * σ > 48 ∧ k = 3 :=
by
  sorry

end find_k_l1065_106525


namespace tan_theta_neq_2sqrt2_l1065_106516

theorem tan_theta_neq_2sqrt2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < Real.pi) (h₁ : Real.sin θ + Real.cos θ = (2 * Real.sqrt 2 - 1) / 3) : Real.tan θ = -2 * Real.sqrt 2 := by
  sorry

end tan_theta_neq_2sqrt2_l1065_106516


namespace inequality_product_lt_zero_l1065_106536

theorem inequality_product_lt_zero (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 :=
  sorry

end inequality_product_lt_zero_l1065_106536


namespace vector_definition_l1065_106505

-- Definition of a vector's characteristics
def hasCharacteristics (vector : Type) := ∃ (magnitude : ℝ) (direction : ℂ), true

-- The statement to prove: a vector is defined by having both magnitude and direction
theorem vector_definition (vector : Type) : hasCharacteristics vector := 
sorry

end vector_definition_l1065_106505


namespace candy_last_days_l1065_106531

variable (candy_from_neighbors candy_from_sister candy_per_day : ℕ)

theorem candy_last_days
  (h_candy_from_neighbors : candy_from_neighbors = 66)
  (h_candy_from_sister : candy_from_sister = 15)
  (h_candy_per_day : candy_per_day = 9) :
  let total_candy := candy_from_neighbors + candy_from_sister  
  (total_candy / candy_per_day) = 9 := by
  sorry

end candy_last_days_l1065_106531


namespace avg_first_six_results_l1065_106563

theorem avg_first_six_results (average_11 : ℕ := 52) (average_last_6 : ℕ := 52) (sixth_result : ℕ := 34) :
  ∃ A : ℕ, (6 * A + 6 * average_last_6 - sixth_result = 11 * average_11) ∧ A = 49 :=
by
  sorry

end avg_first_six_results_l1065_106563


namespace symmetric_about_origin_implies_odd_l1065_106501

variable {F : Type} [Field F] (f : F → F)
variable (x : F)

theorem symmetric_about_origin_implies_odd (H : ∀ x, f (-x) = -f x) : f x + f (-x) = 0 := 
by 
  sorry

end symmetric_about_origin_implies_odd_l1065_106501


namespace area_PQR_l1065_106521

-- Define the point P
def P : ℝ × ℝ := (1, 6)

-- Define the functions for lines passing through P with slopes 1 and 3
def line1 (x : ℝ) : ℝ := x + 5
def line2 (x : ℝ) : ℝ := 3 * x + 3

-- Define the x-intercepts of the lines
def Q : ℝ × ℝ := (-5, 0)
def R : ℝ × ℝ := (-1, 0)

-- Calculate the distance QR
def distance_QR : ℝ := abs (-1 - (-5))

-- Calculate the height from P to the x-axis
def height_P : ℝ := 6

-- State and prove the area of the triangle PQR
theorem area_PQR : 1 / 2 * distance_QR * height_P = 12 := by
  sorry -- The actual proof would be provided here

end area_PQR_l1065_106521


namespace max_value_of_sum_l1065_106594

open Real

theorem max_value_of_sum (x y z : ℝ)
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
    (h2 : (1 / x) + (1 / y) + (1 / z) + x + y + z = 0)
    (h3 : (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1) ∧ (z ≤ -1 ∨ z ≥ 1)) :
    x + y + z ≤ 0 := 
sorry

end max_value_of_sum_l1065_106594


namespace solution_in_quadrant_II_l1065_106534

theorem solution_in_quadrant_II (k x y : ℝ) (h1 : 2 * x + y = 6) (h2 : k * x - y = 4) : x < 0 ∧ y > 0 ↔ k < -2 :=
by
  sorry

end solution_in_quadrant_II_l1065_106534


namespace concatenated_number_divisible_by_37_l1065_106555

theorem concatenated_number_divisible_by_37
  (a b : ℕ) (ha : 100 ≤ a ∧ a ≤ 999) (hb : 100 ≤ b ∧ b ≤ 999)
  (h₁ : a % 37 ≠ 0) (h₂ : b % 37 ≠ 0) (h₃ : (a + b) % 37 = 0) :
  (1000 * a + b) % 37 = 0 :=
sorry

end concatenated_number_divisible_by_37_l1065_106555


namespace ellipse_standard_equation_l1065_106532

theorem ellipse_standard_equation
  (a b : ℝ) (P : ℝ × ℝ) (h_center : P = (3, 0))
  (h_a_eq_3b : a = 3 * b) 
  (h1 : a = 3) 
  (h2 : b = 1) : 
  (∀ (x y : ℝ), (x = 3 → y = 0) → (x = 0 → y = 3)) → 
  ((x^2 / a^2) + y^2 = 1 ∨ (x^2 / b^2) + (y^2 / a^2) = 1) := 
by sorry

end ellipse_standard_equation_l1065_106532


namespace apple_picking_ratio_l1065_106509

theorem apple_picking_ratio (a b c : ℕ) 
  (h1 : a = 66) 
  (h2 : b = 2 * 66) 
  (h3 : a + b + c = 220) :
  c = 22 → a = 66 → c / a = 1 / 3 := by
    intros
    sorry

end apple_picking_ratio_l1065_106509


namespace cos_pi_over_3_plus_2alpha_correct_l1065_106580

noncomputable def cos_pi_over_3_plus_2alpha (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) : Real :=
  Real.cos (Real.pi / 3 + 2 * α)

theorem cos_pi_over_3_plus_2alpha_correct (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) :
  cos_pi_over_3_plus_2alpha α h = -7 / 8 :=
by
  sorry

end cos_pi_over_3_plus_2alpha_correct_l1065_106580


namespace complex_problem_l1065_106582

open Complex

theorem complex_problem (a b : ℝ) (h : (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I) : a + b = 4 := by
  sorry

end complex_problem_l1065_106582


namespace number_of_quartets_l1065_106559

theorem number_of_quartets :
  let n := 5
  let factorial (x : Nat) := Nat.factorial x
  factorial n ^ 3 = 120 ^ 3 :=
by
  sorry

end number_of_quartets_l1065_106559


namespace six_digit_numbers_with_zero_l1065_106515

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l1065_106515


namespace correct_propositions_l1065_106562

noncomputable def proposition1 : Prop :=
  (∀ x : ℝ, x^2 - 3 * x + 2 = 0 -> x = 1) ->
  (∀ x : ℝ, x ≠ 1 -> x^2 - 3 * x + 2 ≠ 0)

noncomputable def proposition2 : Prop :=
  (∀ p q : Prop, p ∨ q -> p ∧ q) ->
  (∀ p q : Prop, p ∧ q -> p ∨ q)

noncomputable def proposition3 : Prop :=
  (∀ p q : Prop, ¬(p ∧ q) -> ¬p ∧ ¬q)

noncomputable def proposition4 : Prop :=
  (∃ x : ℝ, x^2 + x + 1 < 0) ->
  (∀ x : ℝ, x^2 + x + 1 ≥ 0)

theorem correct_propositions :
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l1065_106562
