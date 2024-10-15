import Mathlib

namespace NUMINAMATH_GPT_sum_of_numbers_l194_19495

theorem sum_of_numbers (a b c : ℕ) (h_order: a ≤ b ∧ b ≤ c) (h_median: b = 10) 
    (h_mean_least: (a + b + c) / 3 = a + 15) (h_mean_greatest: (a + b + c) / 3 = c - 20) :
    a + b + c = 45 :=
  by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l194_19495


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l194_19491

-- Problem 1: Prove (1 * -6) + -13 = -19
theorem problem1 : (1 * -6) + -13 = -19 := by 
  sorry

-- Problem 2: Prove (3/5) + (-3/4) = -3/20
theorem problem2 : (3/5 : ℚ) + (-3/4) = -3/20 := by 
  sorry

-- Problem 3: Prove 4.7 + (-0.8) + 5.3 + (-8.2) = 1
theorem problem3 : (4.7 + (-0.8) + 5.3 + (-8.2) : ℝ) = 1 := by 
  sorry

-- Problem 4: Prove (-1/6) + (1/3) + (-1/12) = 1/12
theorem problem4 : (-1/6 : ℚ) + (1/3) + (-1/12) = 1/12 := by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l194_19491


namespace NUMINAMATH_GPT_algebraic_expression_value_l194_19418

theorem algebraic_expression_value (x y : ℝ) (h : x = 2 * y + 3) : 4 * x - 8 * y + 9 = 21 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l194_19418


namespace NUMINAMATH_GPT_total_jokes_sum_l194_19444

theorem total_jokes_sum :
  let jessy_week1 := 11
  let alan_week1 := 7
  let tom_week1 := 5
  let emily_week1 := 3
  let jessy_week4 := 11 * 3 ^ 3
  let alan_week4 := 7 * 2 ^ 3
  let tom_week4 := 5 * 4 ^ 3
  let emily_week4 := 3 * 4 ^ 3
  let jessy_total := 11 + 11 * 3 + 11 * 3 ^ 2 + jessy_week4
  let alan_total := 7 + 7 * 2 + 7 * 2 ^ 2 + alan_week4
  let tom_total := 5 + 5 * 4 + 5 * 4 ^ 2 + tom_week4
  let emily_total := 3 + 3 * 4 + 3 * 4 ^ 2 + emily_week4
  jessy_total + alan_total + tom_total + emily_total = 1225 :=
by 
  sorry

end NUMINAMATH_GPT_total_jokes_sum_l194_19444


namespace NUMINAMATH_GPT_denis_neighbors_l194_19405

-- Define positions
inductive Position
| P1 | P2 | P3 | P4 | P5

open Position

-- Declare the children
inductive Child
| Anya | Borya | Vera | Gena | Denis

open Child

def next_to (p1 p2 : Position) : Prop := 
  (p1 = P1 ∧ p2 = P2) ∨ (p1 = P2 ∧ p2 = P1) ∨
  (p1 = P2 ∧ p2 = P3) ∨ (p1 = P3 ∧ p2 = P2) ∨
  (p1 = P3 ∧ p2 = P4) ∨ (p1 = P4 ∧ p2 = P3) ∨
  (p1 = P4 ∧ p2 = P5) ∨ (p1 = P5 ∧ p2 = P4)

variables (pos : Child → Position)

-- Given conditions
axiom borya_beginning : pos Borya = P1
axiom vera_next_to_anya : next_to (pos Vera) (pos Anya)
axiom vera_not_next_to_gena : ¬ next_to (pos Vera) (pos Gena)
axiom no_two_next_to : ∀ (c1 c2 : Child), 
  c1 ∈ [Anya, Borya, Gena] → c2 ∈ [Anya, Borya, Gena] → c1 ≠ c2 → ¬ next_to (pos c1) (pos c2)

-- Prove the result
theorem denis_neighbors : next_to (pos Denis) (pos Anya) ∧ next_to (pos Denis) (pos Gena) :=
sorry

end NUMINAMATH_GPT_denis_neighbors_l194_19405


namespace NUMINAMATH_GPT_find_distance_PF2_l194_19425

-- Define the properties of the hyperbola
def is_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- Define the property that P lies on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  is_hyperbola P.1 P.2

-- Define foci of the hyperbola
structure foci (F1 F2 : ℝ × ℝ) : Prop :=
(F1_prop : F1 = (2, 0))
(F2_prop : F2 = (-2, 0))

-- Given distance from P to F1
def distance_PF1 (P F1 : ℝ × ℝ) (d : ℝ) : Prop :=
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = d^2

-- The goal is to find the distance |PF2|
theorem find_distance_PF2 (P F1 F2 : ℝ × ℝ) (D1 D2 : ℝ) :
  point_on_hyperbola P →
  foci F1 F2 →
  distance_PF1 P F1 3 →
  D2 - 3 = 4 →
  D2 = 7 :=
by
  intros hP hFoci hDIST hEQ
  -- Proof can be provided here
  sorry

end NUMINAMATH_GPT_find_distance_PF2_l194_19425


namespace NUMINAMATH_GPT_find_f7_l194_19442

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f (x)

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = 2 * x^2

theorem find_f7 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : periodic_function f 4)
  (h3 : specific_values f) :
  f 7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f7_l194_19442


namespace NUMINAMATH_GPT_total_ladybugs_l194_19454

theorem total_ladybugs (leaves : Nat) (ladybugs_per_leaf : Nat) (total_ladybugs : Nat) : 
  leaves = 84 → 
  ladybugs_per_leaf = 139 → 
  total_ladybugs = leaves * ladybugs_per_leaf → 
  total_ladybugs = 11676 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end NUMINAMATH_GPT_total_ladybugs_l194_19454


namespace NUMINAMATH_GPT_total_seats_l194_19432

theorem total_seats (s : ℝ) : 
  let first_class := 36
  let business_class := 0.30 * s
  let economy_class := (3/5:ℝ) * s
  let premium_economy := s - (first_class + business_class + economy_class)
  first_class + business_class + economy_class + premium_economy = s := by 
  sorry

end NUMINAMATH_GPT_total_seats_l194_19432


namespace NUMINAMATH_GPT_total_daily_cost_correct_l194_19420

/-- Definition of the daily wages of each type of worker -/
def daily_wage_worker : ℕ := 100
def daily_wage_electrician : ℕ := 2 * daily_wage_worker
def daily_wage_plumber : ℕ := (5 * daily_wage_worker) / 2 -- 2.5 times daily_wage_worker
def daily_wage_architect : ℕ := 7 * daily_wage_worker / 2 -- 3.5 times daily_wage_worker

/-- Definition of the total daily cost for one project -/
def daily_cost_one_project : ℕ :=
  2 * daily_wage_worker +
  daily_wage_electrician +
  daily_wage_plumber +
  daily_wage_architect

/-- Definition of the total daily cost for three projects -/
def total_daily_cost_three_projects : ℕ :=
  3 * daily_cost_one_project

/-- Theorem stating the overall labor costs for one day for all three projects -/
theorem total_daily_cost_correct :
  total_daily_cost_three_projects = 3000 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_daily_cost_correct_l194_19420


namespace NUMINAMATH_GPT_problem_statement_l194_19419

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def geom_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

theorem problem_statement (a₁ q : ℝ) (h : geom_seq a₁ q 6 = 8 * geom_seq a₁ q 3) :
  geom_sum a₁ q 6 / geom_sum a₁ q 3 = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_statement_l194_19419


namespace NUMINAMATH_GPT_correct_answer_A_correct_answer_C_correct_answer_D_l194_19497

variable (f g : ℝ → ℝ)

namespace ProofProblem

-- Assume the given conditions
axiom f_eq : ∀ x, f x = 6 - deriv g x
axiom f_compl : ∀ x, f (1 - x) = 6 + deriv g (1 + x)
axiom g_odd : ∀ x, g x - 2 = -(g (-x) - 2)

-- Proving the correct answers
theorem correct_answer_A : g 0 = 2 :=
sorry

theorem correct_answer_C : ∀ x, g (x + 4) = g x :=
sorry

theorem correct_answer_D : f 1 * g 1 + f 3 * g 3 = 24 :=
sorry

end ProofProblem

end NUMINAMATH_GPT_correct_answer_A_correct_answer_C_correct_answer_D_l194_19497


namespace NUMINAMATH_GPT_flower_nectar_water_content_l194_19413

/-- Given that to yield 1 kg of honey, 1.6 kg of flower-nectar must be processed,
    and the honey obtained from this nectar contains 20% water,
    prove that the flower-nectar contains 50% water. --/
theorem flower_nectar_water_content :
  (1.6 : ℝ) * (0.2 / 1) = (50 / 100) * (1.6 : ℝ) := by
  sorry

end NUMINAMATH_GPT_flower_nectar_water_content_l194_19413


namespace NUMINAMATH_GPT_cost_of_fencing_per_meter_l194_19435

theorem cost_of_fencing_per_meter (x : ℝ) (length width : ℝ) (area : ℝ) (total_cost : ℝ) :
  length = 3 * x ∧ width = 2 * x ∧ area = 3750 ∧ area = length * width ∧ total_cost = 125 →
  (total_cost / (2 * (length + width)) = 0.5) :=
by
  sorry

end NUMINAMATH_GPT_cost_of_fencing_per_meter_l194_19435


namespace NUMINAMATH_GPT_algebraic_expression_value_l194_19458

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m * n = 3) 
  (h2 : n = m + 1) : 
  (m - n) ^ 2 * ((1 / n) - (1 / m)) = -1 / 3 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_value_l194_19458


namespace NUMINAMATH_GPT_yang_tricks_modulo_l194_19470

noncomputable def number_of_tricks_result : Nat :=
  let N := 20000
  let modulo := 100000
  N % modulo

theorem yang_tricks_modulo :
  number_of_tricks_result = 20000 :=
by
  sorry

end NUMINAMATH_GPT_yang_tricks_modulo_l194_19470


namespace NUMINAMATH_GPT_Andrew_spent_1395_dollars_l194_19485

-- Define the conditions
def cookies_per_day := 3
def cost_per_cookie := 15
def days_in_may := 31

-- Define the calculation
def total_spent := cookies_per_day * cost_per_cookie * days_in_may

-- State the theorem
theorem Andrew_spent_1395_dollars :
  total_spent = 1395 := 
by
  sorry

end NUMINAMATH_GPT_Andrew_spent_1395_dollars_l194_19485


namespace NUMINAMATH_GPT_sheets_of_paper_l194_19436

theorem sheets_of_paper (x : ℕ) (sheets : ℕ) 
  (h1 : sheets = 3 * x + 31)
  (h2 : sheets = 4 * x + 8) : 
  sheets = 100 := by
  sorry

end NUMINAMATH_GPT_sheets_of_paper_l194_19436


namespace NUMINAMATH_GPT_proof_problem_l194_19456

noncomputable def aₙ (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d
noncomputable def Sₙ (a₁ d : ℝ) (n : ℕ) := n * a₁ + (n * (n - 1) / 2) * d

def given_conditions (Sₙ : ℕ → ℝ) : Prop :=
  Sₙ 10 = 0 ∧ Sₙ 15 = 25

theorem proof_problem (a₁ d : ℝ) (Sₙ : ℕ → ℝ)
  (h₁ : Sₙ 10 = 0) (h₂ : Sₙ 15 = 25) :
  (aₙ a₁ d 5 = -1/3) ∧
  (∀ n, Sₙ n = (1 / 3) * (n ^ 2 - 10 * n) → n = 5) ∧
  (∀ n, n * Sₙ n = (n ^ 3 / 3) - (10 * n ^ 2 / 3) → min (n * Sₙ n) = -49) ∧
  (¬ ∃ n, (Sₙ n / n) > 0) :=
sorry

end NUMINAMATH_GPT_proof_problem_l194_19456


namespace NUMINAMATH_GPT_num_valid_pairs_l194_19487

theorem num_valid_pairs (a b : ℕ) (hb : b > a) (h_unpainted_area : ab = 3 * (a - 4) * (b - 4)) :
  (∃ (a b : ℕ), b > a ∧ ab = 3 * (a-4) * (b-4) ∧ (a-6) * (b-6) = 12 ∧ ((a, b) = (7, 18) ∨ (a, b) = (8, 12))) ∧
  (2 = 2) :=
by sorry

end NUMINAMATH_GPT_num_valid_pairs_l194_19487


namespace NUMINAMATH_GPT_hiker_distance_l194_19465

noncomputable def distance_from_start (north south east west : ℕ) : ℝ :=
  let north_south := north - south
  let east_west := east - west
  Real.sqrt (north_south ^ 2 + east_west ^ 2)

theorem hiker_distance :
  distance_from_start 24 8 15 9 = 2 * Real.sqrt 73 := by
  sorry

end NUMINAMATH_GPT_hiker_distance_l194_19465


namespace NUMINAMATH_GPT_sin_beta_equals_sqrt3_div_2_l194_19479

noncomputable def angles_acute (α β : ℝ) : Prop :=
0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2

theorem sin_beta_equals_sqrt3_div_2 
  (α β : ℝ) 
  (h_acute: angles_acute α β) 
  (h_sin_alpha: Real.sin α = (4/7) * Real.sqrt 3) 
  (h_cos_alpha_plus_beta: Real.cos (α + β) = -(11/14)) 
  : Real.sin β = (Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_sin_beta_equals_sqrt3_div_2_l194_19479


namespace NUMINAMATH_GPT_custom_op_identity_l194_19407

def custom_op (x y : ℕ) : ℕ := x * y + 3 * x - 4 * y

theorem custom_op_identity : custom_op 7 5 - custom_op 5 7 = 14 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_identity_l194_19407


namespace NUMINAMATH_GPT_inequalities_always_true_l194_19494

theorem inequalities_always_true (x y a b : ℝ) (hx : 0 < x) (hy : 0 < y) (ha : 0 < a) (hb : 0 < b) 
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ (x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b) := by
  sorry

end NUMINAMATH_GPT_inequalities_always_true_l194_19494


namespace NUMINAMATH_GPT_seq_arithmetic_l194_19421

def seq (n : ℕ) : ℤ := 2 * n + 5

theorem seq_arithmetic :
  ∀ n : ℕ, seq (n + 1) - seq n = 2 :=
by
  intro n
  have h1 : seq (n + 1) = 2 * (n + 1) + 5 := rfl
  have h2 : seq n = 2 * n + 5 := rfl
  rw [h1, h2]
  linarith

end NUMINAMATH_GPT_seq_arithmetic_l194_19421


namespace NUMINAMATH_GPT_real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l194_19445

-- Define z as a complex number with components dependent on m
def z (m : ℝ) : ℂ := ⟨m^2 - m, m - 1⟩

-- Statement 1: z is real iff m = 1
theorem real_z_iff (m : ℝ) : (∃ r : ℝ, z m = ⟨r, 0⟩) ↔ m = 1 := 
    sorry

-- Statement 2: z is purely imaginary iff m = 0
theorem imaginary_z_iff (m : ℝ) : (∃ i : ℝ, z m = ⟨0, i⟩ ∧ i ≠ 0) ↔ m = 0 := 
    sorry

-- Statement 3: z is in the first quadrant iff m > 1
theorem first_quadrant_z_iff (m : ℝ) : (z m).re > 0 ∧ (z m).im > 0 ↔ m > 1 := 
    sorry

end NUMINAMATH_GPT_real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l194_19445


namespace NUMINAMATH_GPT_axes_positioning_l194_19437

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem axes_positioning (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c < 0) :
  ∃ x_vertex y_intercept, x_vertex < 0 ∧ y_intercept < 0 ∧ (∀ x, f a b c x > f a b c x) :=
by
  sorry

end NUMINAMATH_GPT_axes_positioning_l194_19437


namespace NUMINAMATH_GPT_points_on_same_side_of_line_l194_19447

theorem points_on_same_side_of_line (m : ℝ) :
  (2 * 0 + 0 + m > 0 ∧ 2 * -1 + 1 + m > 0) ∨ 
  (2 * 0 + 0 + m < 0 ∧ 2 * -1 + 1 + m < 0) ↔ 
  (m < 0 ∨ m > 1) :=
by
  sorry

end NUMINAMATH_GPT_points_on_same_side_of_line_l194_19447


namespace NUMINAMATH_GPT_range_of_m_l194_19412

open Real

noncomputable def f (x m : ℝ) : ℝ := log x / log 2 + x - m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x m = 0) → 1 < m ∧ m < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l194_19412


namespace NUMINAMATH_GPT_roots_identity_l194_19474

theorem roots_identity (p q r : ℝ) (h₁ : p + q + r = 15) (h₂ : p * q + q * r + r * p = 25) (h₃ : p * q * r = 10) :
  (1 + p) * (1 + q) * (1 + r) = 51 :=
by sorry

end NUMINAMATH_GPT_roots_identity_l194_19474


namespace NUMINAMATH_GPT_sum_exterior_angles_triangle_and_dodecagon_l194_19401

-- Definitions derived from conditions
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Conditions
def is_polygon (n : ℕ) : Prop := n ≥ 3

-- Proof problem statement
theorem sum_exterior_angles_triangle_and_dodecagon :
  is_polygon 3 ∧ is_polygon 12 → sum_exterior_angles 3 + sum_exterior_angles 12 = 720 :=
by
  sorry

end NUMINAMATH_GPT_sum_exterior_angles_triangle_and_dodecagon_l194_19401


namespace NUMINAMATH_GPT_question_statement_l194_19484

def line := Type
def plane := Type

-- Definitions for line lying in plane and planes being parallel 
def isIn (a : line) (α : plane) : Prop := sorry
def isParallel (α β : plane) : Prop := sorry
def isParallelLinePlane (a : line) (β : plane) : Prop := sorry

-- Conditions 
variables (a b : line) (α β : plane) 
variable (distinct_lines : a ≠ b)
variable (distinct_planes : α ≠ β)

-- Main statement to prove
theorem question_statement (h_parallel_planes : isParallel α β) (h_line_in_plane : isIn a α) : isParallelLinePlane a β := 
sorry

end NUMINAMATH_GPT_question_statement_l194_19484


namespace NUMINAMATH_GPT_floor_x_plus_x_eq_13_div_3_l194_19472

-- Statement representing the mathematical problem
theorem floor_x_plus_x_eq_13_div_3 (x : ℚ) (h : ⌊x⌋ + x = 13/3) : x = 7/3 := 
sorry

end NUMINAMATH_GPT_floor_x_plus_x_eq_13_div_3_l194_19472


namespace NUMINAMATH_GPT_cos_double_angle_l194_19455

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2 * n) = 12) : Real.cos (2 * θ) = 5 / 6 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l194_19455


namespace NUMINAMATH_GPT_total_snakes_l194_19463

def People (n : ℕ) : Prop := n = 59
def OnlyDogs (n : ℕ) : Prop := n = 15
def OnlyCats (n : ℕ) : Prop := n = 10
def OnlyCatsAndDogs (n : ℕ) : Prop := n = 5
def CatsDogsSnakes (n : ℕ) : Prop := n = 3

theorem total_snakes (n_people n_dogs n_cats n_catsdogs n_catdogsnsnakes : ℕ)
  (h_people : People n_people) 
  (h_onlyDogs : OnlyDogs n_dogs)
  (h_onlyCats : OnlyCats n_cats)
  (h_onlyCatsAndDogs : OnlyCatsAndDogs n_catsdogs)
  (h_catsDogsSnakes : CatsDogsSnakes n_catdogsnsnakes) :
  n_catdogsnsnakes >= 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_snakes_l194_19463


namespace NUMINAMATH_GPT_worksheets_already_graded_eq_5_l194_19439

def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 9
def remaining_problems : ℕ := 16

def total_problems := total_worksheets * problems_per_worksheet
def graded_problems := total_problems - remaining_problems
def graded_worksheets := graded_problems / problems_per_worksheet

theorem worksheets_already_graded_eq_5 :
  graded_worksheets = 5 :=
by 
  sorry

end NUMINAMATH_GPT_worksheets_already_graded_eq_5_l194_19439


namespace NUMINAMATH_GPT_range_of_e_l194_19498

theorem range_of_e (a b c d e : ℝ) (h₁ : a + b + c + d + e = 8) (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_e_l194_19498


namespace NUMINAMATH_GPT_plywood_problem_exists_squares_l194_19496

theorem plywood_problem_exists_squares :
  ∃ (a b : ℕ), a^2 + b^2 = 625 ∧ a ≠ 20 ∧ b ≠ 20 ∧ a ≠ 15 ∧ b ≠ 15 := by
  sorry

end NUMINAMATH_GPT_plywood_problem_exists_squares_l194_19496


namespace NUMINAMATH_GPT_volume_ratio_l194_19468

-- Define the edge lengths
def edge_length_cube1 : ℝ := 4 -- in inches
def edge_length_cube2 : ℝ := 2 * 12 -- 2 feet converted to inches

-- Define the volumes
def volume_cube (a : ℝ) : ℝ := a ^ 3

-- Statement asserting the ratio of the volumes is 1/216
theorem volume_ratio : volume_cube edge_length_cube1 / volume_cube edge_length_cube2 = 1 / 216 :=
by
  -- This is the placeholder to skip the proof
  sorry

end NUMINAMATH_GPT_volume_ratio_l194_19468


namespace NUMINAMATH_GPT_lcm_24_36_42_l194_19493

-- Definitions of the numbers involved
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 42

-- Statement for the lowest common multiple
theorem lcm_24_36_42 : Nat.lcm (Nat.lcm a b) c = 504 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_lcm_24_36_42_l194_19493


namespace NUMINAMATH_GPT_race_distance_l194_19422

theorem race_distance (a b c : ℝ) (s_A s_B s_C : ℝ) :
  s_A * a = 100 → 
  s_B * a = 95 → 
  s_C * a = 90 → 
  s_B = s_A - 5 → 
  s_C = s_A - 10 → 
  s_C * (s_B / s_A) = 100 → 
  (100 - s_C) = 5 * (5 / 19) :=
sorry

end NUMINAMATH_GPT_race_distance_l194_19422


namespace NUMINAMATH_GPT_arithmetic_geometric_sum_l194_19426

theorem arithmetic_geometric_sum (S : ℕ → ℕ) (n : ℕ) 
  (h1 : S n = 48) 
  (h2 : S (2 * n) = 60)
  (h3 : (S (2 * n) - S n) ^ 2 = S n * (S (3 * n) - S (2 * n))) : 
  S (3 * n) = 63 := by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sum_l194_19426


namespace NUMINAMATH_GPT_median_of_circumscribed_trapezoid_l194_19460

theorem median_of_circumscribed_trapezoid (a b c d : ℝ) (h1 : a + b + c + d = 12) (h2 : a + b = c + d) : (a + b) / 2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_median_of_circumscribed_trapezoid_l194_19460


namespace NUMINAMATH_GPT_probability_both_selected_l194_19400

-- Given conditions
def jamie_probability : ℚ := 2 / 3
def tom_probability : ℚ := 5 / 7

-- Statement to prove
theorem probability_both_selected :
  jamie_probability * tom_probability = 10 / 21 :=
by
  sorry

end NUMINAMATH_GPT_probability_both_selected_l194_19400


namespace NUMINAMATH_GPT_min_value_expression_l194_19429

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 20)^2 ≥ 100 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l194_19429


namespace NUMINAMATH_GPT_max_sum_marks_l194_19469

theorem max_sum_marks (a b c : ℕ) (h1 : a + b + c = 2019) (h2 : a ≤ c + 2) : 
  2 * a + b ≤ 2021 :=
by {
  -- We'll skip the proof but formulate the statement following conditions strictly.
  sorry
}

end NUMINAMATH_GPT_max_sum_marks_l194_19469


namespace NUMINAMATH_GPT_paint_gallons_needed_l194_19433

theorem paint_gallons_needed (n : ℕ) (h : n = 16) (h_col_height : ℝ) (h_col_height_val : h_col_height = 24)
  (h_col_diameter : ℝ) (h_col_diameter_val : h_col_diameter = 8) (cover_area : ℝ) 
  (cover_area_val : cover_area = 350) : 
  ∃ (gallons : ℤ), gallons = 33 := 
by
  sorry

end NUMINAMATH_GPT_paint_gallons_needed_l194_19433


namespace NUMINAMATH_GPT_distance_from_point_to_line_l194_19467

-- Definition of the conditions
def point := (3, 0)
def line_y := 1

-- Problem statement: Prove that the distance between the point (3,0) and the line y=1 is 1.
theorem distance_from_point_to_line (point : ℝ × ℝ) (line_y : ℝ) : abs (point.snd - line_y) = 1 :=
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_distance_from_point_to_line_l194_19467


namespace NUMINAMATH_GPT_bilion_wins_1000000_dollars_l194_19464

theorem bilion_wins_1000000_dollars :
  ∃ (p : ℕ), (p = 1000000) ∧ (p % 3 = 1) → p = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_bilion_wins_1000000_dollars_l194_19464


namespace NUMINAMATH_GPT_chickens_problem_l194_19402

theorem chickens_problem 
    (john_took_more_mary : ∀ (john mary : ℕ), john = mary + 5)
    (ray_took : ℕ := 10)
    (john_took_more_ray : ∀ (john ray : ℕ), john = ray + 11) :
    ∃ mary : ℕ, ray = mary - 6 :=
by
    sorry

end NUMINAMATH_GPT_chickens_problem_l194_19402


namespace NUMINAMATH_GPT_numerical_form_463001_l194_19461

theorem numerical_form_463001 : 463001 = 463001 := by
  rfl

end NUMINAMATH_GPT_numerical_form_463001_l194_19461


namespace NUMINAMATH_GPT_cos_double_angle_of_tangent_is_2_l194_19483

theorem cos_double_angle_of_tangent_is_2
  (θ : ℝ)
  (h_tan : Real.tan θ = 2) :
  Real.cos (2 * θ) = -3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_of_tangent_is_2_l194_19483


namespace NUMINAMATH_GPT_average_water_drunk_l194_19476

theorem average_water_drunk (d1 d2 d3 : ℕ) (h1 : d1 = 215) (h2 : d2 = d1 + 76) (h3 : d3 = d2 - 53) :
  (d1 + d2 + d3) / 3 = 248 :=
by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_average_water_drunk_l194_19476


namespace NUMINAMATH_GPT_Paul_seashells_l194_19406

namespace SeashellProblem

variables (P L : ℕ)

def initial_total_seashells (H P L : ℕ) : Prop := H + P + L = 59

def final_total_seashells (H P L : ℕ) : Prop := H + P + L - L / 4 = 53

theorem Paul_seashells : 
  (initial_total_seashells 11 P L) → (final_total_seashells 11 P L) → P = 24 :=
by
  intros h_initial h_final
  sorry

end SeashellProblem

end NUMINAMATH_GPT_Paul_seashells_l194_19406


namespace NUMINAMATH_GPT_minimum_Q_l194_19453

def is_special (m : ℕ) : Prop :=
  let d1 := m / 10 
  let d2 := m % 10
  d1 ≠ d2 ∧ d1 ≠ 0 ∧ d2 ≠ 0

def F (m : ℕ) : ℤ :=
  let d1 := m / 10
  let d2 := m % 10
  (d1 * 100 + d2 * 10 + d1) - (d2 * 100 + d1 * 10 + d2) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s) / s

variables (a b x y : ℕ)
variables (h1 : 1 ≤ b ∧ b < a ∧ a ≤ 7)
variables (h2 : 1 ≤ x ∧ x ≤ 8)
variables (h3 : 1 ≤ y ∧ y ≤ 8)
variables (hs_is_special : is_special (10 * a + b))
variables (ht_is_special : is_special (10 * x + y))
variables (s := 10 * a + b)
variables (t := 10 * x + y)
variables (h4 : (F s % 5) = 1)
variables (h5 : F t - F s + 18 * x = 36)

theorem minimum_Q : Q s t = -42 / 73 := sorry

end NUMINAMATH_GPT_minimum_Q_l194_19453


namespace NUMINAMATH_GPT_range_of_a_l194_19410

def valid_real_a (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a

theorem range_of_a :
  (∀ a : ℝ, (¬ valid_real_a a)) ↔ (a < 1 ∨ a > 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l194_19410


namespace NUMINAMATH_GPT_max_product_l194_19446

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end NUMINAMATH_GPT_max_product_l194_19446


namespace NUMINAMATH_GPT_probability_sum_15_l194_19451

/-- If three standard 6-faced dice are rolled, the probability that the sum of the face-up integers is 15 is 5/72. -/
theorem probability_sum_15 : (1 / 6 : ℚ) ^ 3 * 3 + (1 / 6 : ℚ) ^ 3 * 6 = 5 / 72 := by 
  sorry

end NUMINAMATH_GPT_probability_sum_15_l194_19451


namespace NUMINAMATH_GPT_father_son_fish_problem_l194_19492

variables {F S x : ℕ}

theorem father_son_fish_problem (h1 : F - x = S + x) (h2 : F + x = 2 * (S - x)) : 
  (F - S) / S = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_father_son_fish_problem_l194_19492


namespace NUMINAMATH_GPT_compute_binom_12_6_eq_1848_l194_19415

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end NUMINAMATH_GPT_compute_binom_12_6_eq_1848_l194_19415


namespace NUMINAMATH_GPT_yaw_yaw_age_in_2016_l194_19403

def is_lucky_double_year (y : Nat) : Prop :=
  let d₁ := y / 1000 % 10
  let d₂ := y / 100 % 10
  let d₃ := y / 10 % 10
  let last_digit := y % 10
  last_digit = 2 * (d₁ + d₂ + d₃)

theorem yaw_yaw_age_in_2016 (next_lucky_year : Nat) (yaw_yaw_age_in_next_lucky_year : Nat)
  (h1 : is_lucky_double_year 2016)
  (h2 : ∀ y, y > 2016 → is_lucky_double_year y → y = next_lucky_year)
  (h3 : yaw_yaw_age_in_next_lucky_year = 17) :
  (17 - (next_lucky_year - 2016)) = 5 := sorry

end NUMINAMATH_GPT_yaw_yaw_age_in_2016_l194_19403


namespace NUMINAMATH_GPT_trajectory_of_P_is_line_l194_19427

noncomputable def P_trajectory_is_line (a m : ℝ) (P : ℝ × ℝ) : Prop :=
  let A := (-a, 0)
  let B := (a, 0)
  let PA := (P.1 + a) ^ 2 + P.2 ^ 2
  let PB := (P.1 - a) ^ 2 + P.2 ^ 2
  PA - PB = m → P.1 = m / (4 * a)

theorem trajectory_of_P_is_line (a m : ℝ) (h : a ≠ 0) :
  ∀ (P : ℝ × ℝ), (P_trajectory_is_line a m P) := sorry

end NUMINAMATH_GPT_trajectory_of_P_is_line_l194_19427


namespace NUMINAMATH_GPT_point_in_second_quadrant_l194_19423

def point := (ℝ × ℝ)

def second_quadrant (p : point) : Prop := p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : second_quadrant (-1, 2) :=
sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l194_19423


namespace NUMINAMATH_GPT_parallel_vectors_xy_l194_19471

theorem parallel_vectors_xy {x y : ℝ} (h : ∃ k : ℝ, (1, y, -3) = (k * x, k * (-2), k * 5)) : x * y = -2 :=
by sorry

end NUMINAMATH_GPT_parallel_vectors_xy_l194_19471


namespace NUMINAMATH_GPT_circle_equation_a_value_l194_19481

theorem circle_equation_a_value (a : ℝ) : (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
sorry

end NUMINAMATH_GPT_circle_equation_a_value_l194_19481


namespace NUMINAMATH_GPT_calculate_perimeter_of_staircase_region_l194_19431

-- Define the properties and dimensions of the staircase-shaped region
def is_right_angle (angle : ℝ) : Prop := angle = 90

def congruent_side_length : ℝ := 1

def bottom_base_length : ℝ := 12

def total_area : ℝ := 78

def perimeter_region : ℝ := 34.5

theorem calculate_perimeter_of_staircase_region
  (is_right_angle : ∀ angle, is_right_angle angle)
  (congruent_sides_count : ℕ := 12)
  (total_congruent_side_length : ℝ := congruent_sides_count * congruent_side_length)
  (bottom_base_length : ℝ)
  (total_area : ℝ)
  : bottom_base_length = 12 ∧ total_area = 78 → 
    ∃ perimeter : ℝ, perimeter = 34.5 :=
by
  admit -- Proof goes here

end NUMINAMATH_GPT_calculate_perimeter_of_staircase_region_l194_19431


namespace NUMINAMATH_GPT_books_number_in_series_l194_19430

-- Definitions and conditions from the problem
def number_books (B : ℕ) := B
def number_movies (M : ℕ) := M
def movies_watched := 61
def books_read := 19
def diff_movies_books := 2

-- The main statement to prove
theorem books_number_in_series (B M: ℕ) 
  (h1 : M = movies_watched)
  (h2 : M - B = diff_movies_books) :
  B = 59 :=
by
  sorry

end NUMINAMATH_GPT_books_number_in_series_l194_19430


namespace NUMINAMATH_GPT_range_of_a_l194_19404

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l194_19404


namespace NUMINAMATH_GPT_eval_expression_l194_19408

theorem eval_expression :
  (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_l194_19408


namespace NUMINAMATH_GPT_edward_initial_money_l194_19475

theorem edward_initial_money (initial_cost_books : ℝ) (discount_percent : ℝ) (num_pens : ℕ) 
  (cost_per_pen : ℝ) (money_left : ℝ) : 
  initial_cost_books = 40 → discount_percent = 0.25 → num_pens = 3 → cost_per_pen = 2 → money_left = 6 → 
  (initial_cost_books * (1 - discount_percent) + num_pens * cost_per_pen + money_left) = 42 :=
by
  sorry

end NUMINAMATH_GPT_edward_initial_money_l194_19475


namespace NUMINAMATH_GPT_pi_sub_alpha_in_first_quadrant_l194_19411

theorem pi_sub_alpha_in_first_quadrant (α : ℝ) (h : π / 2 < α ∧ α < π) : 0 < π - α ∧ π - α < π / 2 :=
by
  sorry

end NUMINAMATH_GPT_pi_sub_alpha_in_first_quadrant_l194_19411


namespace NUMINAMATH_GPT_min_xy_sum_is_7_l194_19466

noncomputable def min_xy_sum (x y : ℝ) : ℝ := 
x + y

theorem min_xy_sum_is_7 (x y : ℝ) (h1 : x > 1) (h2 : y > 2) (h3 : (x - 1) * (y - 2) = 4) : 
  min_xy_sum x y = 7 := by 
  sorry

end NUMINAMATH_GPT_min_xy_sum_is_7_l194_19466


namespace NUMINAMATH_GPT_molecular_weight_of_NH4Br_l194_19473

def atomic_weight (element : String) : Real :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | _ => 0.0

def molecular_weight (composition : List (String × Nat)) : Real :=
  composition.foldl (λ acc (elem, count) => acc + count * atomic_weight elem) 0

theorem molecular_weight_of_NH4Br :
  molecular_weight [("N", 1), ("H", 4), ("Br", 1)] = 97.95 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_NH4Br_l194_19473


namespace NUMINAMATH_GPT_correct_average_l194_19488

theorem correct_average (avg: ℕ) (n: ℕ) (incorrect: ℕ) (correct: ℕ) 
  (h_avg : avg = 16) (h_n : n = 10) (h_incorrect : incorrect = 25) (h_correct : correct = 35) :
  (avg * n + (correct - incorrect)) / n = 17 := 
by
  sorry

end NUMINAMATH_GPT_correct_average_l194_19488


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l194_19482

theorem molecular_weight_of_compound :
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  molecular_weight = 156.22615 :=
by
  -- conditions
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  -- prove statement
  have h1 : average_atomic_weight_c = 12.05015 := by sorry
  have h2 : molecular_weight = 156.22615 := by sorry
  exact h2

end NUMINAMATH_GPT_molecular_weight_of_compound_l194_19482


namespace NUMINAMATH_GPT_find_coefficient_of_x_l194_19449

theorem find_coefficient_of_x :
  ∃ a : ℚ, ∀ (x y : ℚ),
  (x + y = 19) ∧ (x + 3 * y = 1) ∧ (2 * x + y = 5) →
  (a * x + y = 19) ∧ (a = 7) :=
by
  sorry

end NUMINAMATH_GPT_find_coefficient_of_x_l194_19449


namespace NUMINAMATH_GPT_range_of_a_l194_19478

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 5 * a) ↔ (4 ≤ a ∨ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l194_19478


namespace NUMINAMATH_GPT_inscribed_circle_radius_in_sector_l194_19417

theorem inscribed_circle_radius_in_sector
  (radius : ℝ)
  (sector_fraction : ℝ)
  (r : ℝ) :
  radius = 4 →
  sector_fraction = 1/3 →
  r = 2 * Real.sqrt 3 - 2 →
  true := by
sorry

end NUMINAMATH_GPT_inscribed_circle_radius_in_sector_l194_19417


namespace NUMINAMATH_GPT_determine_m_value_l194_19462

-- Define the condition that the roots of the quadratic are given
def quadratic_equation_has_given_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, (8 * x^2 + 4 * x + m = 0) → (x = (-2 + (Complex.I * Real.sqrt 88)) / 8) ∨ (x = (-2 - (Complex.I * Real.sqrt 88)) / 8)

-- The main statement to be proven
theorem determine_m_value (m : ℝ) (h : quadratic_equation_has_given_roots m) : m = 13 / 4 :=
sorry

end NUMINAMATH_GPT_determine_m_value_l194_19462


namespace NUMINAMATH_GPT_hockey_league_num_games_l194_19441

theorem hockey_league_num_games :
  ∃ (num_teams : ℕ) (num_times : ℕ), 
    num_teams = 16 ∧ num_times = 10 ∧ 
    (num_teams * (num_teams - 1) / 2) * num_times = 2400 := by
  sorry

end NUMINAMATH_GPT_hockey_league_num_games_l194_19441


namespace NUMINAMATH_GPT_Xiaoyong_age_solution_l194_19490

theorem Xiaoyong_age_solution :
  ∃ (x y : ℕ), 1 ≤ y ∧ y < x ∧ x < 20 ∧ 2 * x + 5 * y = 97 ∧ x = 16 ∧ y = 13 :=
by
  -- You should provide a suitable proof here
  sorry

end NUMINAMATH_GPT_Xiaoyong_age_solution_l194_19490


namespace NUMINAMATH_GPT_oatmeal_cookies_l194_19424

theorem oatmeal_cookies (total_cookies chocolate_chip_cookies : ℕ)
  (h1 : total_cookies = 6 * 9)
  (h2 : chocolate_chip_cookies = 13) :
  total_cookies - chocolate_chip_cookies = 41 := by
  sorry

end NUMINAMATH_GPT_oatmeal_cookies_l194_19424


namespace NUMINAMATH_GPT_find_c_l194_19434

theorem find_c (c : ℝ) :
  (∀ x y : ℝ, 2*x^2 - 4*c*x*y + (2*c^2 + 1)*y^2 - 2*x - 6*y + 9 ≥ 0) ↔ c = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l194_19434


namespace NUMINAMATH_GPT_pairs_of_integers_solution_l194_19477

-- Define the main theorem
theorem pairs_of_integers_solution :
  ∃ (x y : ℤ), 9 * x * y - x^2 - 8 * y^2 = 2005 ∧ 
               ((x = 63 ∧ y = 58) ∨
               (x = -63 ∧ y = -58) ∨
               (x = 459 ∧ y = 58) ∨
               (x = -459 ∧ y = -58)) :=
by
  sorry

end NUMINAMATH_GPT_pairs_of_integers_solution_l194_19477


namespace NUMINAMATH_GPT_find_a_for_quadratic_roots_l194_19416

theorem find_a_for_quadratic_roots :
  ∀ (a x₁ x₂ : ℝ), 
    (x₁ ≠ x₂) →
    (x₁ * x₁ + a * x₁ + 6 = 0) →
    (x₂ * x₂ + a * x₂ + 6 = 0) →
    (x₁ - (72 / (25 * x₂^3)) = x₂ - (72 / (25 * x₁^3))) →
    (a = 9 ∨ a = -9) :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_quadratic_roots_l194_19416


namespace NUMINAMATH_GPT_sum_of_fractions_l194_19459

theorem sum_of_fractions : 
  (2 / 5 : ℚ) + (4 / 50 : ℚ) + (3 / 500 : ℚ) + (8 / 5000 : ℚ) = 4876 / 10000 :=
by
  -- The proof can be completed by converting fractions and summing them accurately.
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l194_19459


namespace NUMINAMATH_GPT_both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l194_19438

variable (p q : Prop)

-- 1. Both shots were unsuccessful
theorem both_shots_unsuccessful : ¬p ∧ ¬q := sorry

-- 2. Both shots were successful
theorem both_shots_successful : p ∧ q := sorry

-- 3. Exactly one shot was successful
theorem exactly_one_shot_successful : (¬p ∧ q) ∨ (p ∧ ¬q) := sorry

-- 4. At least one shot was successful
theorem at_least_one_shot_successful : p ∨ q := sorry

-- 5. At most one shot was successful
theorem at_most_one_shot_successful : ¬(p ∧ q) := sorry

end NUMINAMATH_GPT_both_shots_unsuccessful_both_shots_successful_exactly_one_shot_successful_at_least_one_shot_successful_at_most_one_shot_successful_l194_19438


namespace NUMINAMATH_GPT_triangle_area_l194_19452

variables {A B C a b c : ℝ}

/-- In triangle ABC, the sides opposite to angles A, B, and C are denoted as a, b, and c, respectively.
It is given that b * sin C + c * sin B = 4 * a * sin B * sin C and b^2 + c^2 - a^2 = 8.
Prove that the area of triangle ABC is 4 * sqrt 3 / 3. -/
theorem triangle_area (h1 : b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C)
  (h2 : b^2 + c^2 - a^2 = 8) :
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_triangle_area_l194_19452


namespace NUMINAMATH_GPT_find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l194_19450

-- Define the coordinate functions for point P
def coord_x (m : ℚ) : ℚ := 3 * m + 6
def coord_y (m : ℚ) : ℚ := m - 3

-- Definitions for each condition
def condition1 (m : ℚ) : Prop := coord_x m = coord_y m
def condition2 (m : ℚ) : Prop := coord_y m = coord_x m + 5
def condition3 (m : ℚ) : Prop := coord_x m = 3

-- Proof statements for the coordinates based on each condition
theorem find_coordinates_condition1 : 
  ∃ m, condition1 m ∧ coord_x m = -7.5 ∧ coord_y m = -7.5 :=
by sorry

theorem find_coordinates_condition2 : 
  ∃ m, condition2 m ∧ coord_x m = -15 ∧ coord_y m = -10 :=
by sorry

theorem find_coordinates_condition3 : 
  ∃ m, condition3 m ∧ coord_x m = 3 ∧ coord_y m = -4 :=
by sorry

end NUMINAMATH_GPT_find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l194_19450


namespace NUMINAMATH_GPT_draw_probability_l194_19457

theorem draw_probability (P_A_win : ℝ) (P_A_not_lose : ℝ) (h1 : P_A_win = 0.3) (h2 : P_A_not_lose = 0.8) : 
  ∃ P_draw : ℝ, P_draw = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_draw_probability_l194_19457


namespace NUMINAMATH_GPT_kenny_pieces_used_l194_19480

-- Definitions based on conditions
def mushrooms_cut := 22
def pieces_per_mushroom := 4
def karla_pieces := 42
def remaining_pieces := 8
def total_pieces := mushrooms_cut * pieces_per_mushroom

-- Theorem to be proved
theorem kenny_pieces_used :
  total_pieces - (karla_pieces + remaining_pieces) = 38 := 
by 
  sorry

end NUMINAMATH_GPT_kenny_pieces_used_l194_19480


namespace NUMINAMATH_GPT_solve_numRedBalls_l194_19440

-- Condition (1): There are a total of 10 balls in the bag
def totalBalls : ℕ := 10

-- Condition (2): The probability of drawing a black ball is 2/5
-- This means the number of black balls is 4
def numBlackBalls : ℕ := 4

-- Condition (3): The probability of drawing at least 1 white ball when drawing 2 balls is 7/9
def probAtLeastOneWhiteBall : ℚ := 7 / 9

-- The number of red balls in the bag is calculated based on the given conditions
def numRedBalls (totalBalls numBlackBalls : ℕ) (probAtLeastOneWhiteBall : ℚ) : ℕ := 
  let totalWhiteAndRedBalls := totalBalls - numBlackBalls
  let probTwoNonWhiteBalls := 1 - probAtLeastOneWhiteBall
  let comb (n k : ℕ) := Nat.choose n k
  let equation := comb totalWhiteAndRedBalls 2 * comb (totalBalls - 2) 0 / comb totalBalls 2
  if equation = probTwoNonWhiteBalls then totalWhiteAndRedBalls else 0

theorem solve_numRedBalls : numRedBalls totalBalls numBlackBalls probAtLeastOneWhiteBall = 1 := by
  sorry

end NUMINAMATH_GPT_solve_numRedBalls_l194_19440


namespace NUMINAMATH_GPT_island_inhabitants_even_l194_19443

theorem island_inhabitants_even 
  (total : ℕ) 
  (knights liars : ℕ)
  (H : total = knights + liars)
  (H1 : ∃ (knk : Prop), (knk → (knights % 2 = 0)) ∧ (¬knk → (knights % 2 = 1)))
  (H2 : ∃ (lkr : Prop), (lkr → (liars % 2 = 1)) ∧ (¬lkr → (liars % 2 = 0)))
  : (total % 2 = 0) := sorry

end NUMINAMATH_GPT_island_inhabitants_even_l194_19443


namespace NUMINAMATH_GPT_Diane_age_l194_19489

variable (C D E : ℝ)

def Carla_age_is_four_times_Diane_age : Prop := C = 4 * D
def Emma_is_eight_years_older_than_Diane : Prop := E = D + 8
def Carla_and_Emma_are_twins : Prop := C = E

theorem Diane_age : Carla_age_is_four_times_Diane_age C D → 
                    Emma_is_eight_years_older_than_Diane D E → 
                    Carla_and_Emma_are_twins C E → 
                    D = 8 / 3 :=
by
  intros hC hE hTwins
  have h1 : C = 4 * D := hC
  have h2 : E = D + 8 := hE
  have h3 : C = E := hTwins
  sorry

end NUMINAMATH_GPT_Diane_age_l194_19489


namespace NUMINAMATH_GPT_polynomial_divisibility_l194_19486

theorem polynomial_divisibility (n : ℕ) : (¬ n % 3 = 0) → (x ^ (2 * n) + x ^ n + 1) % (x ^ 2 + x + 1) = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l194_19486


namespace NUMINAMATH_GPT_jack_total_cost_l194_19499

def cost_of_tires (n : ℕ) (price_per_tire : ℕ) : ℕ := n * price_per_tire
def cost_of_window (price_per_window : ℕ) : ℕ := price_per_window

theorem jack_total_cost :
  cost_of_tires 3 250 + cost_of_window 700 = 1450 :=
by
  sorry

end NUMINAMATH_GPT_jack_total_cost_l194_19499


namespace NUMINAMATH_GPT_solution_set_of_inequality_l194_19428

theorem solution_set_of_inequality :
  ∀ x : ℝ, |2 * x^2 - 1| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l194_19428


namespace NUMINAMATH_GPT_quadratic_roots_p_eq_l194_19448

theorem quadratic_roots_p_eq (b c p q r s : ℝ)
  (h1 : r + s = -b)
  (h2 : r * s = c)
  (h3 : r^2 + s^2 = -p)
  (h4 : r^2 * s^2 = q):
  p = 2 * c - b^2 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_p_eq_l194_19448


namespace NUMINAMATH_GPT_value_of_a1_a3_a5_l194_19409

theorem value_of_a1_a3_a5 (a a1 a2 a3 a4 a5 : ℤ) (h : (2 * x + 1) ^ 5 = a + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5) :
  a1 + a3 + a5 = 122 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a1_a3_a5_l194_19409


namespace NUMINAMATH_GPT_find_books_second_shop_l194_19414

def total_books (books_first_shop books_second_shop : ℕ) : ℕ :=
  books_first_shop + books_second_shop

def total_cost (cost_first_shop cost_second_shop : ℕ) : ℕ :=
  cost_first_shop + cost_second_shop

def average_price (total_cost total_books : ℕ) : ℕ :=
  total_cost / total_books

theorem find_books_second_shop : 
  ∀ (books_first_shop cost_first_shop cost_second_shop : ℕ),
    books_first_shop = 65 →
    cost_first_shop = 1480 →
    cost_second_shop = 920 →
    average_price (total_cost cost_first_shop cost_second_shop) (total_books books_first_shop (2400 / 20 - 65)) = 20 →
    2400 / 20 - 65 = 55 := 
by sorry

end NUMINAMATH_GPT_find_books_second_shop_l194_19414
