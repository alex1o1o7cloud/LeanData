import Mathlib

namespace equation_of_line_l727_72703

theorem equation_of_line {M : ℝ × ℝ} {a b : ℝ} (hM : M = (4,2)) 
  (hAB : ∃ A B : ℝ × ℝ, M = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ 
    A ≠ B ∧ ∀ x y : ℝ, 
    (x^2 + 4 * y^2 = 36 → (∃ k : ℝ, y - 2 = k * (x - 4) ) )):
  (x + 2 * y - 8 = 0) :=
sorry

end equation_of_line_l727_72703


namespace probability_15th_roll_last_is_approximately_l727_72736

noncomputable def probability_15th_roll_last : ℝ :=
  (7 / 8) ^ 13 * (1 / 8)

theorem probability_15th_roll_last_is_approximately :
  abs (probability_15th_roll_last - 0.022) < 0.001 :=
by sorry

end probability_15th_roll_last_is_approximately_l727_72736


namespace smallest_AAAB_value_l727_72763

theorem smallest_AAAB_value : ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 111 * A + B = 7 * (10 * A + B) ∧ 111 * A + B = 667 :=
by sorry

end smallest_AAAB_value_l727_72763


namespace first_three_digits_of_x_are_571_l727_72773

noncomputable def x : ℝ := (10^2003 + 1)^(11/7)

theorem first_three_digits_of_x_are_571 : 
  ∃ d₁ d₂ d₃ : ℕ, 
  (d₁, d₂, d₃) = (5, 7, 1) ∧ 
  ∃ k : ℤ, 
  (x - k : ℝ) * 1000 = d₁ * 100 + d₂ * 10 + d₃ := 
by
  sorry

end first_three_digits_of_x_are_571_l727_72773


namespace inequality_proof_l727_72785

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9 / 16 :=
sorry

end inequality_proof_l727_72785


namespace smaller_circle_radius_l727_72769

theorem smaller_circle_radius (r R : ℝ) (hR : R = 10) (h : 2 * r = 2 * R) : r = 10 :=
by
  sorry

end smaller_circle_radius_l727_72769


namespace b_over_a_squared_eq_seven_l727_72720

theorem b_over_a_squared_eq_seven (a b k : ℕ) (ha : a > 1) (hb : b = a * (10^k + 1)) (hdiv : a^2 ∣ b) :
  b / a^2 = 7 :=
sorry

end b_over_a_squared_eq_seven_l727_72720


namespace find_pairs_l727_72768

theorem find_pairs (a b : ℕ) (h : a + b + a * b = 1000) : 
  (a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
  (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
  (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12) :=
by sorry

end find_pairs_l727_72768


namespace num_lines_satisfying_conditions_l727_72716

-- Define the entities line, angle, and perpendicularity in a geometric framework
variable (Point Line : Type)
variable (P : Point)
variable (a b l : Line)

-- Define geometrical predicates
variable (Perpendicular : Line → Line → Prop)
variable (Passes_Through : Line → Point → Prop)
variable (Forms_Angle : Line → Line → ℝ → Prop)

-- Given conditions
axiom perp_ab : Perpendicular a b
axiom passes_through_P : Passes_Through l P
axiom angle_la_30 : Forms_Angle l a (30 : ℝ)
axiom angle_lb_90 : Forms_Angle l b (90 : ℝ)

-- The statement to prove
theorem num_lines_satisfying_conditions : ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
  Passes_Through l1 P ∧ Forms_Angle l1 a (30 : ℝ) ∧ Forms_Angle l1 b (90 : ℝ) ∧
  Passes_Through l2 P ∧ Forms_Angle l2 a (30 : ℝ) ∧ Forms_Angle l2 b (90 : ℝ) ∧
  (∀ l', Passes_Through l' P ∧ Forms_Angle l' a (30 : ℝ) ∧ Forms_Angle l' b (90 : ℝ) → l' = l1 ∨ l' = l2) := sorry

end num_lines_satisfying_conditions_l727_72716


namespace larger_integer_21_l727_72718

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l727_72718


namespace total_visible_surface_area_l727_72715

-- Define the cubes by their volumes
def volumes : List ℝ := [1, 8, 27, 125, 216, 343, 512, 729]

-- Define the arrangement information as specified
def arrangement_conditions : Prop :=
  ∃ (s8 s7 s6 s5 s4 s3 s2 s1 : ℝ),
    s8^3 = 729 ∧ s7^3 = 512 ∧ s6^3 = 343 ∧ s5^3 = 216 ∧
    s4^3 = 125 ∧ s3^3 = 27 ∧ s2^3 = 8 ∧ s1^3 = 1 ∧
    5 * s8^2 + (5 * s7^2 + 4 * s6^2 + 4 * s5^2) + 
    (5 * s4^2 + 4 * s3^2 + 5 * s2^2 + 4 * s1^2) = 1250

-- The proof statement
theorem total_visible_surface_area : arrangement_conditions → 1250 = 1250 := by
  intro _ -- this stands for not proving the condition, taking it as assumption
  exact rfl


end total_visible_surface_area_l727_72715


namespace total_homework_problems_l727_72704

-- Define the conditions as Lean facts
def finished_problems : ℕ := 45
def ratio_finished_to_left := (9, 4)
def problems_left (L : ℕ) := finished_problems * ratio_finished_to_left.2 = L * ratio_finished_to_left.1 

-- State the theorem
theorem total_homework_problems (L : ℕ) (h : problems_left L) : finished_problems + L = 65 :=
sorry

end total_homework_problems_l727_72704


namespace coin_change_problem_l727_72794

theorem coin_change_problem (d q h : ℕ) (n : ℕ) 
  (h1 : 2 * d + 5 * q + 10 * h = 240)
  (h2 : d ≥ 1)
  (h3 : q ≥ 1)
  (h4 : h ≥ 1) :
  n = 275 := 
sorry

end coin_change_problem_l727_72794


namespace cone_prism_ratio_l727_72778

theorem cone_prism_ratio 
  (a b h_c h_p : ℝ) (hb_lt_a : b < a) : 
  (π * b * h_c) / (12 * a * h_p) = (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) :=
by
  sorry

end cone_prism_ratio_l727_72778


namespace rebecca_swimming_problem_l727_72776

theorem rebecca_swimming_problem :
  ∃ D : ℕ, (D / 4 - D / 5) = 6 → D = 120 :=
sorry

end rebecca_swimming_problem_l727_72776


namespace percentage_spent_on_hats_l727_72787

def total_money : ℕ := 90
def cost_per_scarf : ℕ := 2
def number_of_scarves : ℕ := 18
def cost_of_scarves : ℕ := number_of_scarves * cost_per_scarf
def money_left_for_hats : ℕ := total_money - cost_of_scarves
def number_of_hats : ℕ := 2 * number_of_scarves

theorem percentage_spent_on_hats : 
  (money_left_for_hats : ℝ) / (total_money : ℝ) * 100 = 60 :=
by
  sorry

end percentage_spent_on_hats_l727_72787


namespace arithmetic_sequence_S9_l727_72700

noncomputable def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_S9 (a : ℕ → ℕ)
    (h1 : 2 * a 6 = 6 + a 7) :
    Sn a 9 = 54 := 
sorry

end arithmetic_sequence_S9_l727_72700


namespace brown_dog_count_l727_72709

theorem brown_dog_count:
  ∀ (T L N : ℕ), T = 45 → L = 36 → N = 8 → (T - N - (T - L - N) = 37) :=
by
  intros T L N hT hL hN
  sorry

end brown_dog_count_l727_72709


namespace circle_equation_l727_72708

/-
  Prove that the standard equation for the circle passing through points
  A(-6, 0), B(0, 2), and the origin O(0, 0) is (x+3)^2 + (y-1)^2 = 10.
-/
theorem circle_equation :
  ∃ (x y : ℝ), x = -6 ∨ x = 0 ∨ x = 0 ∧ y = 0 ∨ y = 2 ∨ y = 0 → (∀ P : ℝ × ℝ, P = (-6, 0) ∨ P = (0, 2) ∨ P = (0, 0) → (P.1 + 3)^2 + (P.2 - 1)^2 = 10) := 
sorry

end circle_equation_l727_72708


namespace total_students_l727_72711

-- Define the conditions
def rank_from_right := 17
def rank_from_left := 5

-- The proof statement
theorem total_students : rank_from_right + rank_from_left - 1 = 21 := 
by 
  -- Assuming the conditions represented by the definitions
  -- Without loss of generality the proof would be derived from these, but it is skipped
  sorry

end total_students_l727_72711


namespace gcd_lcm_product_l727_72705

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end gcd_lcm_product_l727_72705


namespace mixed_number_solution_l727_72737

noncomputable def mixed_number_problem : Prop :=
  let a := 4 + 2 / 7
  let b := 5 + 1 / 2
  let c := 3 + 1 / 3
  let d := 2 + 1 / 6
  (a * b) - (c + d) = 18 + 1 / 14

theorem mixed_number_solution : mixed_number_problem := by 
  sorry

end mixed_number_solution_l727_72737


namespace shaded_area_eq_l727_72767

noncomputable def diameter_AB : ℝ := 6
noncomputable def diameter_BC : ℝ := 6
noncomputable def diameter_CD : ℝ := 6
noncomputable def diameter_DE : ℝ := 6
noncomputable def diameter_EF : ℝ := 6
noncomputable def diameter_FG : ℝ := 6
noncomputable def diameter_AG : ℝ := 6 * 6 -- 36

noncomputable def area_small_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

noncomputable def area_large_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

theorem shaded_area_eq :
  area_large_semicircle diameter_AG + area_small_semicircle diameter_AB = 166.5 * Real.pi :=
  sorry

end shaded_area_eq_l727_72767


namespace initial_amount_invested_l727_72754

-- Definition of the conditions as Lean definitions
def initial_amount_interest_condition (A r : ℝ) : Prop := 25000 = A * r
def interest_rate_condition (r : ℝ) : Prop := r = 5

-- The main theorem we want to prove
theorem initial_amount_invested (A r : ℝ) (h1 : initial_amount_interest_condition A r) (h2 : interest_rate_condition r) : A = 5000 :=
by {
  sorry
}

end initial_amount_invested_l727_72754


namespace six_digit_number_theorem_l727_72735

-- Define the problem conditions
def six_digit_number_condition (N : ℕ) (x : ℕ) : Prop :=
  N = 200000 + x ∧ N < 1000000 ∧ (10 * x + 2 = 3 * N)

-- Define the value of x
def value_of_x : ℕ := 85714

-- Main theorem to prove
theorem six_digit_number_theorem (N : ℕ) (x : ℕ) (h1 : x = value_of_x) :
  six_digit_number_condition N x → N = 285714 :=
by
  intros h
  sorry

end six_digit_number_theorem_l727_72735


namespace fixed_point_of_line_l727_72732

theorem fixed_point_of_line (a : ℝ) : 
  (a + 3) * (-2) + (2 * a - 1) * 1 + 7 = 0 := 
by 
  sorry

end fixed_point_of_line_l727_72732


namespace functional_inequality_solution_l727_72701

theorem functional_inequality_solution (f : ℝ → ℝ) (h : ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f (a) + b) * (a - f (b))) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := 
sorry

end functional_inequality_solution_l727_72701


namespace fraction_of_defective_engines_l727_72726

theorem fraction_of_defective_engines
  (total_batches : ℕ)
  (engines_per_batch : ℕ)
  (non_defective_engines : ℕ)
  (H1 : total_batches = 5)
  (H2 : engines_per_batch = 80)
  (H3 : non_defective_engines = 300)
  : (total_batches * engines_per_batch - non_defective_engines) / (total_batches * engines_per_batch) = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end fraction_of_defective_engines_l727_72726


namespace greatest_possible_value_of_y_l727_72717

theorem greatest_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 :=
sorry

end greatest_possible_value_of_y_l727_72717


namespace greatest_m_value_l727_72725

noncomputable def find_greatest_m : ℝ := sorry

theorem greatest_m_value :
  ∃ m : ℝ, 
    (∀ x, x^2 - m * x + 8 = 0 → x ∈ {x | ∃ y, y^2 = 116}) ∧ 
    m = 2 * Real.sqrt 29 :=
sorry

end greatest_m_value_l727_72725


namespace sequence_general_term_l727_72722

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (n / (n + 1 : ℝ)) * a n) : 
  ∀ n, a n = 1 / n := by
  sorry

end sequence_general_term_l727_72722


namespace smallest_n_for_constant_term_l727_72740

theorem smallest_n_for_constant_term :
  ∃ (n : ℕ), (n > 0) ∧ ((∃ (r : ℕ), 2 * n = 5 * r) ∧ (∀ (m : ℕ), m > 0 → (∃ (r' : ℕ), 2 * m = 5 * r') → n ≤ m)) ∧ n = 5 :=
by
  sorry

end smallest_n_for_constant_term_l727_72740


namespace emily_weight_l727_72743

theorem emily_weight (h_weight : 87 = 78 + e_weight) : e_weight = 9 := by
  sorry

end emily_weight_l727_72743


namespace route_comparison_l727_72782

noncomputable def t_X : ℝ := (8 / 40) * 60 -- time in minutes for Route X
noncomputable def t_Y1 : ℝ := (5.5 / 50) * 60 -- time in minutes for the normal speed segment of Route Y
noncomputable def t_Y2 : ℝ := (1 / 25) * 60 -- time in minutes for the construction zone segment of Route Y
noncomputable def t_Y3 : ℝ := (0.5 / 20) * 60 -- time in minutes for the park zone segment of Route Y
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3 -- total time in minutes for Route Y

theorem route_comparison : t_X - t_Y = 1.5 :=
by {
  -- Proof is skipped using sorry
  sorry
}

end route_comparison_l727_72782


namespace green_notebook_cost_l727_72721

def total_cost : ℕ := 45
def black_cost : ℕ := 15
def pink_cost : ℕ := 10
def num_green_notebooks : ℕ := 2

theorem green_notebook_cost :
  (total_cost - (black_cost + pink_cost)) / num_green_notebooks = 10 :=
by
  sorry

end green_notebook_cost_l727_72721


namespace david_reading_time_l727_72702

def total_time : ℕ := 180
def math_homework : ℕ := 25
def spelling_homework : ℕ := 30
def history_assignment : ℕ := 20
def science_project : ℕ := 15
def piano_practice : ℕ := 30
def study_breaks : ℕ := 2 * 10

def time_other_activities : ℕ := math_homework + spelling_homework + history_assignment + science_project + piano_practice + study_breaks

theorem david_reading_time : total_time - time_other_activities = 40 :=
by
  -- Calculation steps would go here, not provided for the theorem statement.
  sorry

end david_reading_time_l727_72702


namespace Woojin_harvested_weight_l727_72751

-- Definitions based on conditions
def younger_brother_harvest : Float := 3.8
def older_sister_harvest : Float := younger_brother_harvest + 8.4
def one_tenth_older_sister : Float := older_sister_harvest / 10
def woojin_extra_g : Float := 3720

-- Convert grams to kilograms
def grams_to_kg (g : Float) : Float := g / 1000

-- Theorem to be proven
theorem Woojin_harvested_weight :
  grams_to_kg (one_tenth_older_sister * 1000 + woojin_extra_g) = 4.94 :=
by
  sorry

end Woojin_harvested_weight_l727_72751


namespace sequence_property_l727_72742

variable (a : ℕ → ℝ)

theorem sequence_property (h : ∀ n : ℕ, 0 < a n) 
  (h_property : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / n :=
by
  sorry

end sequence_property_l727_72742


namespace cylindrical_to_rectangular_l727_72799

structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

structure RectangularCoord where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def convertCylindricalToRectangular (c : CylindricalCoord) : RectangularCoord :=
  { x := c.r * Real.cos c.θ,
    y := c.r * Real.sin c.θ,
    z := c.z }

theorem cylindrical_to_rectangular :
  convertCylindricalToRectangular ⟨7, Real.pi / 3, -3⟩ = ⟨3.5, 7 * Real.sqrt 3 / 2, -3⟩ :=
by sorry

end cylindrical_to_rectangular_l727_72799


namespace fourth_root_sq_eq_sixteen_l727_72730

theorem fourth_root_sq_eq_sixteen (x : ℝ) (h : (x^(1/4))^2 = 16) : x = 256 :=
sorry

end fourth_root_sq_eq_sixteen_l727_72730


namespace domain_of_c_l727_72795

theorem domain_of_c (m : ℝ) :
  (∀ x : ℝ, 7*x^2 - 6*x + m ≠ 0) ↔ (m > (9 / 7)) :=
by
  -- you would typically put the proof here, but we use sorry to skip it
  sorry

end domain_of_c_l727_72795


namespace sally_picked_peaches_l727_72734

-- Definitions from the conditions
def originalPeaches : ℕ := 13
def totalPeaches : ℕ := 55

-- The proof statement
theorem sally_picked_peaches : totalPeaches - originalPeaches = 42 := by
  sorry

end sally_picked_peaches_l727_72734


namespace probability_of_green_ball_l727_72712

def container_X := (5, 7)  -- (red balls, green balls)
def container_Y := (7, 5)  -- (red balls, green balls)
def container_Z := (7, 5)  -- (red balls, green balls)

def total_balls (container : ℕ × ℕ) : ℕ := container.1 + container.2

def probability_green (container : ℕ × ℕ) : ℚ := 
  (container.2 : ℚ) / total_balls container

noncomputable def probability_green_from_random_selection : ℚ :=
  (1 / 3) * probability_green container_X +
  (1 / 3) * probability_green container_Y +
  (1 / 3) * probability_green container_Z

theorem probability_of_green_ball :
  probability_green_from_random_selection = 17 / 36 :=
sorry

end probability_of_green_ball_l727_72712


namespace average_rainfall_correct_l727_72750

/-- In July 1861, 366 inches of rain fell in Cherrapunji, India. -/
def total_rainfall : ℤ := 366

/-- July has 31 days. -/
def days_in_july : ℤ := 31

/-- Each day has 24 hours. -/
def hours_per_day : ℤ := 24

/-- The total number of hours in July -/
def total_hours_in_july : ℤ := days_in_july * hours_per_day

/-- The average rainfall in inches per hour during July 1861 in Cherrapunji, India -/
def average_rainfall_per_hour : ℤ := total_rainfall / total_hours_in_july

/-- Proof that the average rainfall in inches per hour is 366 / (31 * 24) -/
theorem average_rainfall_correct : average_rainfall_per_hour = 366 / (31 * 24) :=
by
  /- We skip the proof as it is not required. -/
  sorry

end average_rainfall_correct_l727_72750


namespace max_profit_at_grade_5_l727_72739

-- Defining the conditions
def profit_per_item (x : ℕ) : ℕ :=
  4 * (x - 1) + 8

def production_count (x : ℕ) : ℕ := 
  60 - 6 * (x - 1)

def daily_profit (x : ℕ) : ℕ :=
  profit_per_item x * production_count x

-- The grade range
def grade_range (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

-- Prove that the grade that maximizes the profit is 5
theorem max_profit_at_grade_5 : (1 ≤ x ∧ x ≤ 10) → daily_profit x ≤ daily_profit 5 :=
sorry

end max_profit_at_grade_5_l727_72739


namespace paul_score_higher_by_26_l727_72765

variable {R : Type} [LinearOrderedField R]

variables (A1 A2 A3 P1 P2 P3 : R)

-- hypotheses
variable (h1 : A1 = P1 + 10)
variable (h2 : A2 = P2 + 4)
variable (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)

-- goal
theorem paul_score_higher_by_26 : P3 - A3 = 26 := by
  sorry

end paul_score_higher_by_26_l727_72765


namespace sum_of_edges_rectangular_solid_l727_72780

theorem sum_of_edges_rectangular_solid
  (a r : ℝ)
  (hr : r ≠ 0)
  (volume_eq : (a / r) * a * (a * r) = 512)
  (surface_area_eq : 2 * ((a ^ 2) / r + a ^ 2 + (a ^ 2) * r) = 384)
  (geo_progression : true) : -- This is implicitly understood in the construction
  4 * ((a / r) + a + (a * r)) = 112 :=
by
  -- The proof will be placed here
  sorry

end sum_of_edges_rectangular_solid_l727_72780


namespace quadratic_real_roots_range_l727_72729

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l727_72729


namespace bus_seats_needed_l727_72770

def members_playing_instruments : Prop :=
  let flute := 5
  let trumpet := 3 * flute
  let trombone := trumpet - 8
  let drum := trombone + 11
  let clarinet := 2 * flute
  let french_horn := trombone + 3
  let saxophone := (trumpet + trombone) / 2
  let piano := drum + 2
  let violin := french_horn - clarinet
  let guitar := 3 * flute
  let total_members := flute + trumpet + trombone + drum + clarinet + french_horn + saxophone + piano + violin + guitar
  total_members = 111

theorem bus_seats_needed : members_playing_instruments :=
by
  sorry

end bus_seats_needed_l727_72770


namespace four_angles_for_shapes_l727_72727

-- Definitions for the shapes
def is_rectangle (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_square (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_parallelogram (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

-- Main proposition
theorem four_angles_for_shapes {fig : Type} :
  (is_rectangle fig) ∧ (is_square fig) ∧ (is_parallelogram fig) →
  ∀ shape : fig, ∃ angles : ℕ, angles = 4 := by
  sorry

end four_angles_for_shapes_l727_72727


namespace sum_of_n_and_k_l727_72753

open Nat

theorem sum_of_n_and_k (n k : ℕ)
  (h1 : 2 = n - 3 * k)
  (h2 : 8 = 2 * n - 5 * k) :
  n + k = 18 :=
sorry

end sum_of_n_and_k_l727_72753


namespace license_plate_increase_factor_l727_72772

def old_license_plates := 26^2 * 10^3
def new_license_plates := 26^3 * 10^4

theorem license_plate_increase_factor : (new_license_plates / old_license_plates) = 260 := by
  sorry

end license_plate_increase_factor_l727_72772


namespace units_digit_G1000_is_3_l727_72706

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 1

theorem units_digit_G1000_is_3 : (G 1000) % 10 = 3 := sorry

end units_digit_G1000_is_3_l727_72706


namespace no_real_ordered_triples_l727_72771

theorem no_real_ordered_triples (x y z : ℝ) (h1 : x + y = 3) (h2 : xy - z^2 = 4) : false :=
sorry

end no_real_ordered_triples_l727_72771


namespace three_irreducible_fractions_prod_eq_one_l727_72797

-- Define the set of numbers available for use
def available_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a structure for an irreducible fraction
structure irreducible_fraction :=
(num : ℕ)
(denom : ℕ)
(h_coprime : Nat.gcd num denom = 1)
(h_in_set : num ∈ available_numbers ∧ denom ∈ available_numbers)

-- Definition of the main proof problem
theorem three_irreducible_fractions_prod_eq_one :
  ∃ (f1 f2 f3 : irreducible_fraction), 
    f1.num * f2.num * f3.num = f1.denom * f2.denom * f3.denom ∧ 
    f1.num ≠ f2.num ∧ f1.num ≠ f3.num ∧ f2.num ≠ f3.num ∧ 
    f1.denom ≠ f2.denom ∧ f1.denom ≠ f3.denom ∧ f2.denom ≠ f3.denom := 
by
  sorry

end three_irreducible_fractions_prod_eq_one_l727_72797


namespace min_value_of_f_l727_72752

-- Define the problem domain: positive real numbers
variables (a b c x y z : ℝ)
variables (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0)
variables (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)

-- Define the given equations
variables (h1 : c * y + b * z = a)
variables (h2 : a * z + c * x = b)
variables (h3 : b * x + a * y = c)

-- Define the function f(x, y, z)
noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

-- The theorem statement: under the given conditions the minimum value of f(x, y, z) is 1/2
theorem min_value_of_f :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    c * y + b * z = a →
    a * z + c * x = b →
    b * x + a * y = c →
    f x y z = 1 / 2) :=
sorry

end min_value_of_f_l727_72752


namespace triangle_is_right_l727_72777

theorem triangle_is_right (A B C a b c : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
    (h₄ : A + B + C = π) (h_eq : a * (Real.cos C) + c * (Real.cos A) = b * (Real.sin B)) : B = π / 2 :=
by
  sorry

end triangle_is_right_l727_72777


namespace total_pages_read_is_785_l727_72792

-- Definitions based on the conditions in the problem
def pages_read_first_five_days : ℕ := 5 * 52
def pages_read_next_five_days : ℕ := 5 * 63
def pages_read_last_three_days : ℕ := 3 * 70

-- The main statement to prove
theorem total_pages_read_is_785 :
  pages_read_first_five_days + pages_read_next_five_days + pages_read_last_three_days = 785 :=
by
  sorry

end total_pages_read_is_785_l727_72792


namespace find_n_of_permut_comb_eq_l727_72779

open Nat

theorem find_n_of_permut_comb_eq (n : Nat) (h : (n! / (n - 3)!) = 6 * (n! / (4! * (n - 4)!))) : n = 7 := by
  sorry

end find_n_of_permut_comb_eq_l727_72779


namespace candy_bars_per_bag_l727_72757

/-
Define the total number of candy bars and the number of bags
-/
def totalCandyBars : ℕ := 75
def numberOfBags : ℚ := 15.0

/-
Prove that the number of candy bars per bag is 5
-/
theorem candy_bars_per_bag : totalCandyBars / numberOfBags = 5 := by
  sorry

end candy_bars_per_bag_l727_72757


namespace arithmetic_seq_sum_div_fifth_term_l727_72793

open Int

/-- The sequence {a_n} is an arithmetic sequence with a non-zero common difference,
    given that a₂ + a₆ = a₈, prove that S₅ / a₅ = 3. -/
theorem arithmetic_seq_sum_div_fifth_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0)
  (h_condition : a 2 + a 6 = a 8) :
  ((5 * a 1 + 10 * d) / (a 1 + 4 * d) : ℚ) = 3 := 
by
  sorry

end arithmetic_seq_sum_div_fifth_term_l727_72793


namespace min_value_y_l727_72719

noncomputable def y (x : ℝ) := (2 - Real.cos x) / Real.sin x

theorem min_value_y (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) : 
  ∃ c ≥ 0, ∀ x, 0 < x ∧ x < Real.pi → y x ≥ c ∧ c = Real.sqrt 3 := 
sorry

end min_value_y_l727_72719


namespace triangle_height_in_terms_of_s_l727_72746

theorem triangle_height_in_terms_of_s (s h : ℝ)
  (rectangle_area : 2 * s * s = 2 * s^2)
  (base_of_triangle : base = s)
  (areas_equal : (1 / 2) * s * h = 2 * s^2) :
  h = 4 * s :=
by
  sorry

end triangle_height_in_terms_of_s_l727_72746


namespace combined_capacity_l727_72714

theorem combined_capacity (A B : ℝ) : 3 * A + B = A + 2 * A + B :=
by
  sorry

end combined_capacity_l727_72714


namespace parametric_line_eq_l727_72791

theorem parametric_line_eq (t : ℝ) :
  ∃ t : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
  (x = 3 * t + 5) ∧ (y = 6 * t - 7) → y = 2 * x - 17 :=
by
  sorry

end parametric_line_eq_l727_72791


namespace remainder_7n_mod_4_l727_72760

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l727_72760


namespace inhabitants_number_even_l727_72759

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem inhabitants_number_even
  (K L : ℕ)
  (hK : is_even K)
  (hL : is_even L) :
  ¬ is_even (K + L + 1) :=
by
  sorry

end inhabitants_number_even_l727_72759


namespace distance_is_absolute_value_l727_72781

noncomputable def distance_to_origin (x : ℝ) : ℝ := |x|

theorem distance_is_absolute_value (x : ℝ) : distance_to_origin x = |x| :=
by
  sorry

end distance_is_absolute_value_l727_72781


namespace determine_x_l727_72783

theorem determine_x (x y : ℝ) (h : x / (x - 1) = (y^3 + 2 * y^2 - 1) / (y^3 + 2 * y^2 - 2)) : 
  x = y^3 + 2 * y^2 - 1 :=
by
  sorry

end determine_x_l727_72783


namespace intersection_with_y_axis_l727_72761

theorem intersection_with_y_axis :
  ∀ (y : ℝ), (∃ x : ℝ, y = 2 * x + 2 ∧ x = 0) → y = 2 :=
by
  sorry

end intersection_with_y_axis_l727_72761


namespace simplify_sqrt_expression_eq_l727_72756

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  let sqrt_45x := Real.sqrt (45 * x)
  let sqrt_20x := Real.sqrt (20 * x)
  let sqrt_30x := Real.sqrt (30 * x)
  sqrt_45x * sqrt_20x * sqrt_30x

theorem simplify_sqrt_expression_eq (x : ℝ) :
  simplify_sqrt_expression x = 30 * x * Real.sqrt 30 := by
  sorry

end simplify_sqrt_expression_eq_l727_72756


namespace hyperbola_asymptotes_slope_l727_72788

open Real

theorem hyperbola_asymptotes_slope (m : ℝ) : 
  (∀ x y : ℝ, (y ^ 2 / 16) - (x ^ 2 / 9) = 1 → (y = m * x ∨ y = -m * x)) → 
  m = 4 / 3 := 
by 
  sorry

end hyperbola_asymptotes_slope_l727_72788


namespace cost_of_items_l727_72741

theorem cost_of_items {x y z : ℕ} (h1 : x + 3 * y + 2 * z = 98)
                      (h2 : 3 * x + y = 5 * z - 36)
                      (even_x : x % 2 = 0) :
  x = 4 ∧ y = 22 ∧ z = 14 := 
by
  sorry

end cost_of_items_l727_72741


namespace unique_n_for_prime_p_l727_72707

theorem unique_n_for_prime_p (p : ℕ) (hp1 : p > 2) (hp2 : Nat.Prime p) :
  ∃! (n : ℕ), (∃ (k : ℕ), n^2 + n * p = k^2) ∧ n = (p - 1) / 2 ^ 2 :=
sorry

end unique_n_for_prime_p_l727_72707


namespace infinite_sum_equals_two_l727_72786

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_equals_two_l727_72786


namespace water_volume_correct_l727_72724

def total_initial_solution : ℚ := 0.08 + 0.04 + 0.02
def fraction_water_in_initial : ℚ := 0.04 / total_initial_solution
def desired_total_volume : ℚ := 0.84
def required_water_volume : ℚ := desired_total_volume * fraction_water_in_initial

theorem water_volume_correct : 
  required_water_volume = 0.24 :=
by
  -- The proof is omitted
  sorry

end water_volume_correct_l727_72724


namespace sin_10pi_over_3_l727_72745

theorem sin_10pi_over_3 : Real.sin (10 * Real.pi / 3) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_10pi_over_3_l727_72745


namespace max_trains_ratio_l727_72747

theorem max_trains_ratio (years : ℕ) 
    (birthday_trains : ℕ) 
    (christmas_trains : ℕ) 
    (total_trains : ℕ)
    (parents_multiple : ℕ) 
    (h_years : years = 5)
    (h_birthday_trains : birthday_trains = 1)
    (h_christmas_trains : christmas_trains = 2)
    (h_total_trains : total_trains = 45)
    (h_parents_multiple : parents_multiple = 2) :
  let trains_received_in_years := years * (birthday_trains + 2 * christmas_trains)
  let trains_given_by_parents := total_trains - trains_received_in_years
  let trains_before_gift := total_trains - trains_given_by_parents
  trains_given_by_parents / trains_before_gift = parents_multiple := by
  sorry

end max_trains_ratio_l727_72747


namespace range_of_k_l727_72713

noncomputable def f (x : ℝ) : ℝ := Real.log x + x

def is_ktimes_value_function (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  0 < k ∧ a < b ∧ f a = k * a ∧ f b = k * b

theorem range_of_k (k : ℝ) : (∃ a b : ℝ, is_ktimes_value_function f k a b) ↔ 1 < k ∧ k < 1 + 1 / Real.exp 1 := by
  sorry

end range_of_k_l727_72713


namespace president_and_committee_combination_l727_72775

theorem president_and_committee_combination : 
  (∃ (n : ℕ), n = 10 * (Nat.choose 9 3)) := 
by
  use 840
  sorry

end president_and_committee_combination_l727_72775


namespace total_profit_is_35000_l727_72798

-- Definitions based on the conditions
variables (IB TB : ℝ) -- IB: Investment of B, TB: Time period of B's investment
def IB_times_TB := IB * TB
def IA := 3 * IB
def TA := 2 * TB
def profit_share_B := IB_times_TB
def profit_share_A := 6 * IB_times_TB
variable (profit_B : ℝ)
def profit_B_val := 5000

-- Ensure these definitions are used
def total_profit := profit_share_A + profit_share_B

-- Lean 4 statement showing that the total profit is Rs 35000
theorem total_profit_is_35000 : total_profit = 35000 := by
  sorry

end total_profit_is_35000_l727_72798


namespace fraction_doubled_l727_72744

variable (x y : ℝ)

theorem fraction_doubled (x y : ℝ) : 
  (x + y) ≠ 0 → (2 * x * 2 * y) / (2 * x + 2 * y) = 2 * (x * y / (x + y)) := 
by
  intro h
  sorry

end fraction_doubled_l727_72744


namespace value_of_a_b_c_l727_72774

theorem value_of_a_b_c (a b c : ℚ) (h₁ : |a| = 2) (h₂ : |b| = 2) (h₃ : |c| = 3) (h₄ : b < 0) (h₅ : 0 < a) :
  a + b + c = 3 ∨ a + b + c = -3 :=
by
  sorry

end value_of_a_b_c_l727_72774


namespace shoe_price_monday_final_price_l727_72748

theorem shoe_price_monday_final_price : 
  let thursday_price := 50
  let friday_markup_rate := 0.15
  let monday_discount_rate := 0.12
  let friday_price := thursday_price * (1 + friday_markup_rate)
  let monday_price := friday_price * (1 - monday_discount_rate)
  monday_price = 50.6 := by
  sorry

end shoe_price_monday_final_price_l727_72748


namespace positive_b_3b_sq_l727_72731

variable (a b c : ℝ)

theorem positive_b_3b_sq (h1 : 0 < a ∧ a < 0.5) (h2 : -0.5 < b ∧ b < 0) (h3 : 1 < c ∧ c < 3) : b + 3 * b^2 > 0 :=
sorry

end positive_b_3b_sq_l727_72731


namespace infinite_series_sum_l727_72784

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) * (1 / 1000)^n = 3000000 / 998001 :=
by sorry

end infinite_series_sum_l727_72784


namespace not_all_perfect_squares_l727_72758

theorem not_all_perfect_squares (d : ℕ) (hd : 0 < d) :
  ¬ (∃ (x y z : ℕ), 2 * d - 1 = x^2 ∧ 5 * d - 1 = y^2 ∧ 13 * d - 1 = z^2) :=
by
  sorry

end not_all_perfect_squares_l727_72758


namespace condition_nonzero_neither_zero_l727_72710

theorem condition_nonzero_neither_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end condition_nonzero_neither_zero_l727_72710


namespace boxes_in_case_correct_l727_72764

-- Given conditions
def total_boxes : Nat := 2
def blocks_per_box : Nat := 6
def total_blocks : Nat := 12

-- Define the number of boxes in a case as a result of total_blocks divided by blocks_per_box
def boxes_in_case : Nat := total_blocks / blocks_per_box

-- Prove the number of boxes in a case is 2
theorem boxes_in_case_correct : boxes_in_case = 2 := by
  -- Place the actual proof here
  sorry

end boxes_in_case_correct_l727_72764


namespace area_of_paper_l727_72790

theorem area_of_paper (L W : ℕ) (h1 : L + 2 * W = 34) (h2 : 2 * L + W = 38) : L * W = 140 := by
  sorry

end area_of_paper_l727_72790


namespace correct_product_of_a_and_b_l727_72762

-- Define reversal function for two-digit numbers
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- State the main problem
theorem correct_product_of_a_and_b (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 0 < b) 
  (h : (reverse_digits a) * b = 284) : a * b = 68 :=
sorry

end correct_product_of_a_and_b_l727_72762


namespace DF_length_l727_72723

-- Definitions for the given problem.
variable (AB DC EB DE : ℝ)
variable (parallelogram_ABCD : Prop)
variable (DE_altitude_AB : Prop)
variable (DF_altitude_BC : Prop)

-- Conditions
axiom AB_eq_DC : AB = DC
axiom EB_eq_5 : EB = 5
axiom DE_eq_8 : DE = 8

-- The main theorem to prove
theorem DF_length (hAB : AB = 15) (hDC : DC = 15) (hEB : EB = 5) (hDE : DE = 8)
  (hPar : parallelogram_ABCD)
  (hAltAB : DE_altitude_AB)
  (hAltBC : DF_altitude_BC) :
  ∃ DF : ℝ, DF = 8 := 
sorry

end DF_length_l727_72723


namespace manufacturing_department_degrees_l727_72789

def percentage_of_circle (percentage : ℕ) (total_degrees : ℕ) : ℕ :=
  (percentage * total_degrees) / 100

theorem manufacturing_department_degrees :
  percentage_of_circle 30 360 = 108 :=
by
  sorry

end manufacturing_department_degrees_l727_72789


namespace parities_of_E_10_11_12_l727_72796

noncomputable def E : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| (n + 3) => 2 * (E (n + 2)) + (E n)

theorem parities_of_E_10_11_12 :
  (E 10 % 2 = 0) ∧ (E 11 % 2 = 1) ∧ (E 12 % 2 = 1) := 
  by
  sorry

end parities_of_E_10_11_12_l727_72796


namespace order_DABC_l727_72728

-- Definitions of the variables given in the problem
def A : ℕ := 77^7
def B : ℕ := 7^77
def C : ℕ := 7^7^7
def D : ℕ := Nat.factorial 7

-- The theorem stating the required ascending order
theorem order_DABC : D < A ∧ A < B ∧ B < C :=
by sorry

end order_DABC_l727_72728


namespace proposition_false_n5_l727_72755

variable (P : ℕ → Prop)

-- Declaring the conditions as definitions:
def condition1 (k : ℕ) (hk : k > 0) : Prop := P k → P (k + 1)
def condition2 : Prop := ¬ P 6

-- Theorem statement which leverages the conditions to prove the desired result.
theorem proposition_false_n5 (h1: ∀ k (hk : k > 0), condition1 P k hk) (h2: condition2 P) : ¬ P 5 :=
sorry

end proposition_false_n5_l727_72755


namespace find_r_over_s_at_0_l727_72766

noncomputable def r (x : ℝ) : ℝ := -3 * (x + 1) * (x - 2)
noncomputable def s (x : ℝ) : ℝ := (x + 1) * (x - 3)

theorem find_r_over_s_at_0 : (r 0) / (s 0) = 2 := by
  sorry

end find_r_over_s_at_0_l727_72766


namespace min_sum_intercepts_l727_72733

theorem min_sum_intercepts (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (1 : ℝ) * a + (1 : ℝ) * b = a * b) : a + b = 4 :=
by
  sorry

end min_sum_intercepts_l727_72733


namespace range_of_a_l727_72738

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2 - 2 * x

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x - a * x - 1

theorem range_of_a
  (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0) ↔
  0 < a ∧ a < Real.exp (-2) :=
sorry

end range_of_a_l727_72738


namespace sequence_term_formula_l727_72749

theorem sequence_term_formula 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h : ∀ n, S n = n^2 + 3 * n)
  (h₁ : a 1 = 4)
  (h₂ : ∀ n, 1 < n → a n = S n - S (n - 1)) :
  ∀ n, a n = 2 * n + 2 :=
by
  sorry

end sequence_term_formula_l727_72749
