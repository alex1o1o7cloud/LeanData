import Mathlib

namespace NUMINAMATH_GPT_omega_range_for_monotonically_decreasing_l207_20779

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem omega_range_for_monotonically_decreasing
  (ω : ℝ)
  (hω : ω > 0)
  (h_decreasing : ∀ x ∈ Set.Ioo (Real.pi / 2) Real.pi, f ω x < f ω (x + 1e-6)) :
  1/2 ≤ ω ∧ ω ≤ 5/4 :=
by
  sorry

end NUMINAMATH_GPT_omega_range_for_monotonically_decreasing_l207_20779


namespace NUMINAMATH_GPT_triangle_largest_angle_l207_20766

theorem triangle_largest_angle (x : ℝ) (AB : ℝ) (AC : ℝ) (BC : ℝ) (h1 : AB = x + 5) 
                               (h2 : AC = 2 * x + 3) (h3 : BC = x + 10)
                               (h_angle_A_largest : BC > AB ∧ BC > AC)
                               (triangle_inequality_1 : AB + AC > BC)
                               (triangle_inequality_2 : AB + BC > AC)
                               (triangle_inequality_3 : AC + BC > AB) :
  1 < x ∧ x < 7 ∧ 6 = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_triangle_largest_angle_l207_20766


namespace NUMINAMATH_GPT_square_division_l207_20724

theorem square_division (n : Nat) : (n > 5 → ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) ∧ (n = 2 ∨ n = 3 → ¬ ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) := 
by
  sorry

end NUMINAMATH_GPT_square_division_l207_20724


namespace NUMINAMATH_GPT_range_of_a_max_value_of_z_l207_20743

variable (a b : ℝ)

-- Definition of the assumptions
def condition1 := (2 * a + b = 9)
def condition2 := (|9 - b| + |a| < 3)
def condition3 := (a > 0)
def condition4 := (b > 0)
def z := a^2 * b

-- Statement for problem (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : -1 < a ∧ a < 1 := sorry

-- Statement for problem (ii)
theorem max_value_of_z (h1 : condition1 a b) (h2 : condition3 a) (h3 : condition4 b) : 
  z a b = 27 := sorry

end NUMINAMATH_GPT_range_of_a_max_value_of_z_l207_20743


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l207_20771

theorem reciprocal_of_neg_2023 : (1 : ℝ) / (-2023) = -(1 / 2023) :=
by 
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l207_20771


namespace NUMINAMATH_GPT_geometric_seq_increasing_l207_20751

theorem geometric_seq_increasing (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) → 
  (a 1 > a 0) = (∃ a1, (a1 > 0 ∧ q > 1) ∨ (a1 < 0 ∧ 0 < q ∧ q < 1)) :=
sorry

end NUMINAMATH_GPT_geometric_seq_increasing_l207_20751


namespace NUMINAMATH_GPT_number_of_bars_in_box_l207_20787

variable (x : ℕ)
variable (cost_per_bar : ℕ := 6)
variable (remaining_bars : ℕ := 6)
variable (total_money_made : ℕ := 42)

theorem number_of_bars_in_box :
  cost_per_bar * (x - remaining_bars) = total_money_made → x = 13 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_bars_in_box_l207_20787


namespace NUMINAMATH_GPT_inequality_proof_l207_20736

variable {α β γ : ℝ}

theorem inequality_proof (h1 : β * γ ≠ 0) (h2 : (1 - γ^2) / (β * γ) ≥ 0) :
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ :=
sorry

end NUMINAMATH_GPT_inequality_proof_l207_20736


namespace NUMINAMATH_GPT_jack_flyers_count_l207_20769

-- Definitions based on the given conditions
def total_flyers : ℕ := 1236
def rose_flyers : ℕ := 320
def flyers_left : ℕ := 796

-- Statement to prove
theorem jack_flyers_count : total_flyers - (rose_flyers + flyers_left) = 120 := by
  sorry

end NUMINAMATH_GPT_jack_flyers_count_l207_20769


namespace NUMINAMATH_GPT_number_of_pupils_in_class_l207_20796

theorem number_of_pupils_in_class
(U V : ℕ) (increase : ℕ) (avg_increase : ℕ) (n : ℕ) 
(h1 : U = 85) (h2 : V = 45) (h3 : increase = U - V) (h4 : avg_increase = 1 / 2) (h5 : increase / avg_increase = n) :
n = 80 := by
sorry

end NUMINAMATH_GPT_number_of_pupils_in_class_l207_20796


namespace NUMINAMATH_GPT_angle_A_measure_find_a_l207_20704

theorem angle_A_measure (a b c : ℝ) (A B C : ℝ) (h1 : (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  A = π / 3 :=
by
  -- proof steps are omitted
  sorry

theorem find_a (a b c : ℝ) (A : ℝ) (h2 : 2 * c = 3 * b) (area : ℝ) (h3 : area = 6 * Real.sqrt 3)
  (h4 : A = π / 3) :
  a = 2 * Real.sqrt 21 / 3 :=
by
  -- proof steps are omitted
  sorry

end NUMINAMATH_GPT_angle_A_measure_find_a_l207_20704


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l207_20778

variable {M N P : Set α}

theorem necessary_but_not_sufficient_condition (h : M ∩ P = N ∩ P) : 
  (M = N) → (M ∩ P = N ∩ P) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l207_20778


namespace NUMINAMATH_GPT_quadratic_binomial_plus_int_l207_20709

theorem quadratic_binomial_plus_int (y : ℝ) : y^2 + 14*y + 60 = (y + 7)^2 + 11 :=
by sorry

end NUMINAMATH_GPT_quadratic_binomial_plus_int_l207_20709


namespace NUMINAMATH_GPT_geometry_biology_overlap_diff_l207_20773

theorem geometry_biology_overlap_diff :
  ∀ (total_students geometry_students biology_students : ℕ),
  total_students = 232 →
  geometry_students = 144 →
  biology_students = 119 →
  (max geometry_students biology_students - max 0 (geometry_students + biology_students - total_students)) = 88 :=
by
  intros total_students geometry_students biology_students
  sorry

end NUMINAMATH_GPT_geometry_biology_overlap_diff_l207_20773


namespace NUMINAMATH_GPT_work_completion_times_l207_20706

variable {M P S : ℝ} -- Let M, P, and S be work rates for Matt, Peter, and Sarah.

theorem work_completion_times (h1 : M + P + S = 1 / 15)
                             (h2 : 10 * (P + S) = 7 / 15) :
                             (1 / M = 50) ∧ (1 / (P + S) = 150 / 7) :=
by
  -- Proof comes here
  -- Calculation skipped
  sorry

end NUMINAMATH_GPT_work_completion_times_l207_20706


namespace NUMINAMATH_GPT_simplify_expression_l207_20733

theorem simplify_expression (a : ℝ) (h : a / 2 - 2 / a = 3) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l207_20733


namespace NUMINAMATH_GPT_fraction_value_l207_20777

theorem fraction_value : (2 * 0.24) / (20 * 2.4) = 0.01 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l207_20777


namespace NUMINAMATH_GPT_solve_equation_l207_20717

theorem solve_equation : ∀ x : ℝ, 4 * x + 4 - x - 2 * x + 2 - 2 - x + 2 + 6 = 0 → x = 0 :=
by 
  intro x h
  sorry

end NUMINAMATH_GPT_solve_equation_l207_20717


namespace NUMINAMATH_GPT_alloy_mixture_l207_20789

theorem alloy_mixture (x y : ℝ) 
  (h1 : x + y = 1000)
  (h2 : 0.25 * x + 0.50 * y = 450) : 
  x = 200 ∧ y = 800 :=
by
  -- Proof will follow here
  sorry

end NUMINAMATH_GPT_alloy_mixture_l207_20789


namespace NUMINAMATH_GPT_tables_count_is_correct_l207_20785

-- Definitions based on conditions
def invited_people : ℕ := 18
def people_didnt_show_up : ℕ := 12
def people_per_table : ℕ := 3

-- Calculation based on definitions
def people_attended : ℕ := invited_people - people_didnt_show_up
def tables_needed : ℕ := people_attended / people_per_table

-- The main theorem statement
theorem tables_count_is_correct : tables_needed = 2 := by
  unfold tables_needed
  unfold people_attended
  unfold invited_people
  unfold people_didnt_show_up
  unfold people_per_table
  sorry

end NUMINAMATH_GPT_tables_count_is_correct_l207_20785


namespace NUMINAMATH_GPT_solution_l207_20718

noncomputable def given_conditions (θ : ℝ) : Prop := 
  let a := (3, 1)
  let b := (Real.sin θ, Real.cos θ)
  (a.1 : ℝ) / b.1 = a.2 / b.2 

theorem solution (θ : ℝ) (h: given_conditions θ) :
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_l207_20718


namespace NUMINAMATH_GPT_tim_weekly_payment_l207_20703

-- Define the given conditions
def hourly_rate_bodyguard : ℕ := 20
def number_bodyguards : ℕ := 2
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7

-- Define the total weekly payment calculation
def weekly_payment : ℕ := (hourly_rate_bodyguard * number_bodyguards) * hours_per_day * days_per_week

-- The proof statement
theorem tim_weekly_payment : weekly_payment = 2240 := by
  sorry

end NUMINAMATH_GPT_tim_weekly_payment_l207_20703


namespace NUMINAMATH_GPT_jack_marathon_time_l207_20781

noncomputable def marathon_distance : ℝ := 42
noncomputable def jill_time : ℝ := 4.2
noncomputable def speed_ratio : ℝ := 0.7636363636363637

noncomputable def jill_speed : ℝ := marathon_distance / jill_time
noncomputable def jack_speed : ℝ := speed_ratio * jill_speed
noncomputable def jack_time : ℝ := marathon_distance / jack_speed

theorem jack_marathon_time : jack_time = 5.5 := sorry

end NUMINAMATH_GPT_jack_marathon_time_l207_20781


namespace NUMINAMATH_GPT_probability_exactly_two_sunny_days_l207_20742

-- Define the conditions
def rain_probability : ℝ := 0.8
def sun_probability : ℝ := 1 - rain_probability
def days : ℕ := 5
def sunny_days : ℕ := 2
def rainy_days : ℕ := days - sunny_days

-- Define the combinatorial and probability calculations
def comb (n k : ℕ) : ℕ := Nat.choose n k
def probability_sunny_days : ℝ := comb days sunny_days * (sun_probability ^ sunny_days) * (rain_probability ^ rainy_days)

theorem probability_exactly_two_sunny_days : probability_sunny_days = 51 / 250 := by
  sorry

end NUMINAMATH_GPT_probability_exactly_two_sunny_days_l207_20742


namespace NUMINAMATH_GPT_jack_salt_amount_l207_20720

noncomputable def amount_of_salt (volume_salt_1 : ℝ) (volume_salt_2 : ℝ) : ℝ :=
  volume_salt_1 + volume_salt_2

noncomputable def total_salt_ml (total_salt_l : ℝ) : ℝ :=
  total_salt_l * 1000

theorem jack_salt_amount :
  let day1_water_l := 4.0
  let day2_water_l := 4.0
  let day1_salt_percentage := 0.18
  let day2_salt_percentage := 0.22
  let total_salt_before_evaporation := amount_of_salt (day1_water_l * day1_salt_percentage) (day2_water_l * day2_salt_percentage)
  let final_salt_ml := total_salt_ml total_salt_before_evaporation
  final_salt_ml = 1600 :=
by
  sorry

end NUMINAMATH_GPT_jack_salt_amount_l207_20720


namespace NUMINAMATH_GPT_eval_g_l207_20705

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem eval_g : 3 * g 2 + 4 * g (-4) = 327 := 
by
  sorry

end NUMINAMATH_GPT_eval_g_l207_20705


namespace NUMINAMATH_GPT_linear_system_solution_l207_20794

/-- Given a system of three linear equations:
      x + y + z = 1
      a x + b y + c z = h
      a² x + b² y + c² z = h²
    Prove that the solution x, y, z is given by:
    x = (h - b)(h - c) / (a - b)(a - c)
    y = (h - a)(h - c) / (b - a)(b - c)
    z = (h - a)(h - b) / (c - a)(c - b) -/
theorem linear_system_solution (a b c h : ℝ) (x y z : ℝ) :
  x + y + z = 1 →
  a * x + b * y + c * z = h →
  a^2 * x + b^2 * y + c^2 * z = h^2 →
  x = (h - b) * (h - c) / ((a - b) * (a - c)) ∧
  y = (h - a) * (h - c) / ((b - a) * (b - c)) ∧
  z = (h - a) * (h - b) / ((c - a) * (c - b)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_linear_system_solution_l207_20794


namespace NUMINAMATH_GPT_vartan_recreation_l207_20741

noncomputable def vartan_recreation_percent (W : ℝ) (P : ℝ) : Prop := 
  let W_this_week := 0.9 * W
  let recreation_last_week := (P / 100) * W
  let recreation_this_week := 0.3 * W_this_week
  recreation_this_week = 1.8 * recreation_last_week

theorem vartan_recreation (W : ℝ) : ∀ P : ℝ, vartan_recreation_percent W P → P = 15 := 
by
  intro P h
  unfold vartan_recreation_percent at h
  sorry

end NUMINAMATH_GPT_vartan_recreation_l207_20741


namespace NUMINAMATH_GPT_correct_factorization_l207_20707

theorem correct_factorization :
  ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_GPT_correct_factorization_l207_20707


namespace NUMINAMATH_GPT_checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l207_20784

-- Define the conditions
def is_checkered_rectangle (S : ℕ) : Prop :=
  (∃ (a b : ℕ), a * b = S) ∧
  (∀ x y k l : ℕ, x * 13 + y * 1 = S) ∧
  (S % 39 = 0)

-- Define that S is minimal satisfying the conditions
def minimal_area_checkered_rectangle (S : ℕ) : Prop :=
  is_checkered_rectangle S ∧
  (∀ (S' : ℕ), S' < S → ¬ is_checkered_rectangle S')

-- Prove that S = 78 is the minimal area
theorem checkered_rectangle_minimal_area : minimal_area_checkered_rectangle 78 :=
  sorry

-- Define the condition for possible perimeters
def possible_perimeters (S : ℕ) (p : ℕ) : Prop :=
  (∀ (a b : ℕ), a * b = S → 2 * (a + b) = p)

-- Prove the possible perimeters for area 78
theorem checkered_rectangle_possible_perimeters :
  ∀ p, p = 38 ∨ p = 58 ∨ p = 82 ↔ possible_perimeters 78 p :=
  sorry

end NUMINAMATH_GPT_checkered_rectangle_minimal_area_checkered_rectangle_possible_perimeters_l207_20784


namespace NUMINAMATH_GPT_correct_option_l207_20728

-- Define the conditions
def c1 (a : ℝ) : Prop := (2 * a^2)^3 ≠ 6 * a^6
def c2 (a : ℝ) : Prop := (a^8) / (a^2) ≠ a^4
def c3 (x y : ℝ) : Prop := (4 * x^2 * y) / (-2 * x * y) ≠ -2
def c4 : Prop := Real.sqrt ((-2)^2) = 2

-- The main statement to be proved
theorem correct_option (a x y : ℝ) (h1 : c1 a) (h2 : c2 a) (h3 : c3 x y) (h4 : c4) : c4 :=
by
  apply h4

end NUMINAMATH_GPT_correct_option_l207_20728


namespace NUMINAMATH_GPT_relationship_abc_l207_20714

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c := by 
  sorry

end NUMINAMATH_GPT_relationship_abc_l207_20714


namespace NUMINAMATH_GPT_inverse_function_less_than_zero_l207_20731

theorem inverse_function_less_than_zero (x : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = 2^x + 1) (h₂ : ∀ y, f (f⁻¹ y) = y) (h₃ : ∀ y, f⁻¹ (f y) = y) :
  {x | f⁻¹ x < 0} = {x | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_less_than_zero_l207_20731


namespace NUMINAMATH_GPT_projection_equal_p_l207_20729

open Real EuclideanSpace

noncomputable def vector1 : ℝ × ℝ := (-3, 4)
noncomputable def vector2 : ℝ × ℝ := (1, 6)
noncomputable def v : ℝ × ℝ := (4, 2)
noncomputable def p : ℝ × ℝ := (-2.2, 4.4)

theorem projection_equal_p (p_ortho : (p.1 * v.1 + p.2 * v.2) = 0) : p = (4 * (1 / 5) - 3, 2 * (1 / 5) + 4) :=
by
  sorry

end NUMINAMATH_GPT_projection_equal_p_l207_20729


namespace NUMINAMATH_GPT_largest_number_l207_20775

theorem largest_number (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 0.9791) 
  (h₂ : x₂ = 0.97019)
  (h₃ : x₃ = 0.97909)
  (h₄ : x₄ = 0.971)
  (h₅ : x₅ = 0.97109)
  : max x₁ (max x₂ (max x₃ (max x₄ x₅))) = 0.9791 :=
  sorry

end NUMINAMATH_GPT_largest_number_l207_20775


namespace NUMINAMATH_GPT_average_eq_instantaneous_velocity_at_t_eq_3_l207_20732

theorem average_eq_instantaneous_velocity_at_t_eq_3
  (S : ℝ → ℝ) (hS : ∀ t, S t = 24 * t - 3 * t^2) :
  (1 / 6) * (S 6 - S 0) = 24 - 6 * 3 :=
by 
  sorry

end NUMINAMATH_GPT_average_eq_instantaneous_velocity_at_t_eq_3_l207_20732


namespace NUMINAMATH_GPT_amount_on_table_A_l207_20757

-- Definitions based on conditions
variables (A B C : ℝ)
variables (h1 : B = 2 * C)
variables (h2 : C = A + 20)
variables (h3 : A + B + C = 220)

-- Theorem statement
theorem amount_on_table_A : A = 40 :=
by
  -- This is expected to be filled in with the proof steps, but we skip it with 'sorry'
  sorry

end NUMINAMATH_GPT_amount_on_table_A_l207_20757


namespace NUMINAMATH_GPT_polygon_sides_l207_20759

theorem polygon_sides (n : ℕ) 
  (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l207_20759


namespace NUMINAMATH_GPT_find_a_maximize_profit_sets_sold_after_increase_l207_20701

variable (a x m : ℕ)

-- Condition for finding 'a'
def condition_for_a (a : ℕ) : Prop :=
  600 * (a - 110) = 160 * a

-- The equation after solving
def solution_for_a (a : ℕ) : Prop :=
  a = 150

theorem find_a : condition_for_a a → solution_for_a a :=
sorry

-- Profit maximization constraints
def condition_for_max_profit (x : ℕ) : Prop :=
  x + 5 * x + 20 ≤ 200

-- Total number of items purchased
def total_items_purchased (x : ℕ) : ℕ :=
  x + 5 * x + 20

-- Profit expression
def profit (x : ℕ) : ℕ :=
  215 * x + 600

-- Maximized profit
def maximum_profit (W : ℕ) : Prop :=
  W = 7050

theorem maximize_profit (x : ℕ) (W : ℕ) :
  condition_for_max_profit x → x ≤ 30 → total_items_purchased x ≤ 200 → maximum_profit W → x = 30 :=
sorry

-- Condition for sets sold after increase
def condition_for_sets_sold (a m : ℕ) : Prop :=
  let new_table_price := 160
  let new_chair_price := 50
  let profit_m_after_increase := (500 - new_table_price - 4 * new_chair_price) * m +
                                (30 - m) * (270 - new_table_price) +
                                (170 - 4 * m) * (70 - new_chair_price)
  profit_m_after_increase + 2250 = 7050 - 2250

-- Solved for 'm'
def quantity_of_sets_sold (m : ℕ) : Prop :=
  m = 20

theorem sets_sold_after_increase (a m : ℕ) :
  condition_for_sets_sold a m → quantity_of_sets_sold m :=
sorry

end NUMINAMATH_GPT_find_a_maximize_profit_sets_sold_after_increase_l207_20701


namespace NUMINAMATH_GPT_extreme_points_exactly_one_zero_in_positive_interval_l207_20735

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1 / 3) * a * x^3

theorem extreme_points (a : ℝ) (h : a > Real.exp 1) :
  ∃ (x1 x2 x3 : ℝ), (0 < x1) ∧ (x1 < x2) ∧ (x2 < x3) ∧ (deriv (f x) = 0) := sorry

theorem exactly_one_zero_in_positive_interval (a : ℝ) (h : a > Real.exp 1) :
  ∃! x : ℝ, (0 < x) ∧ (f x a = 0) := sorry

end NUMINAMATH_GPT_extreme_points_exactly_one_zero_in_positive_interval_l207_20735


namespace NUMINAMATH_GPT_original_number_l207_20710

theorem original_number (y : ℚ) (h : 1 - (1 / y) = 5 / 4) : y = -4 :=
sorry

end NUMINAMATH_GPT_original_number_l207_20710


namespace NUMINAMATH_GPT_seven_thirteenths_of_3940_percent_25000_l207_20738

noncomputable def seven_thirteenths (x : ℝ) : ℝ := (7 / 13) * x

noncomputable def percent (part whole : ℝ) : ℝ := (part / whole) * 100

theorem seven_thirteenths_of_3940_percent_25000 :
  percent (seven_thirteenths 3940) 25000 = 8.484 :=
by
  sorry

end NUMINAMATH_GPT_seven_thirteenths_of_3940_percent_25000_l207_20738


namespace NUMINAMATH_GPT_factorial_trailing_digits_l207_20792

theorem factorial_trailing_digits (n : ℕ) :
  ¬ ∃ k : ℕ, (n! / 10^k) % 10000 = 1976 ∧ k > 0 := 
sorry

end NUMINAMATH_GPT_factorial_trailing_digits_l207_20792


namespace NUMINAMATH_GPT_final_number_of_cards_l207_20760

def initial_cards : ℕ := 26
def cards_given_to_mary : ℕ := 18
def cards_found_in_box : ℕ := 40
def cards_given_to_john : ℕ := 12
def cards_purchased_at_fleamarket : ℕ := 25

theorem final_number_of_cards :
  (initial_cards - cards_given_to_mary) + (cards_found_in_box - cards_given_to_john) + cards_purchased_at_fleamarket = 61 :=
by sorry

end NUMINAMATH_GPT_final_number_of_cards_l207_20760


namespace NUMINAMATH_GPT_eliminate_y_by_subtraction_l207_20713

theorem eliminate_y_by_subtraction (m n : ℝ) :
  (6 * x + m * y = 3) ∧ (2 * x - n * y = -6) →
  (∀ x y : ℝ, 4 * x + (m + n) * y = 9) → (m + n = 0) :=
by
  intros h eq_subtracted
  sorry

end NUMINAMATH_GPT_eliminate_y_by_subtraction_l207_20713


namespace NUMINAMATH_GPT_find_repeating_digits_l207_20723

-- Specify given conditions
def incorrect_result (a : ℚ) (b : ℚ) : ℚ := 54 * b - 1.8
noncomputable def correct_multiplication_value (d: ℚ) := 2 + d
noncomputable def repeating_decimal_value : ℚ := 2 + 35 / 99

-- Define what needs to be proved
theorem find_repeating_digits : ∃ (x : ℕ), x * 100 = 35 := by
  sorry

end NUMINAMATH_GPT_find_repeating_digits_l207_20723


namespace NUMINAMATH_GPT_divides_trans_l207_20770

theorem divides_trans (m n : ℤ) (h : n ∣ m * (n + 1)) : n ∣ m :=
by
  sorry

end NUMINAMATH_GPT_divides_trans_l207_20770


namespace NUMINAMATH_GPT_value_of_B_l207_20730

theorem value_of_B (B : ℚ) (h : 3 * B - 5 = 23) : B = 28 / 3 :=
by
  sorry

-- Explanation:
-- B is declared as a rational number (ℚ) because the answer involves a fraction.
-- h is the condition 3 * B - 5 = 23.
-- The theorem states that given h, B equals 28 / 3.

end NUMINAMATH_GPT_value_of_B_l207_20730


namespace NUMINAMATH_GPT_fraction_of_two_bedroom_l207_20767

theorem fraction_of_two_bedroom {x : ℝ} 
    (h1 : 0.17 + x = 0.5) : x = 0.33 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_two_bedroom_l207_20767


namespace NUMINAMATH_GPT_walk_time_to_LakePark_restaurant_l207_20711

/-
  It takes 15 minutes for Dante to go to Hidden Lake.
  From Hidden Lake, it takes him 7 minutes to walk back to the Park Office.
  Dante will have been gone from the Park Office for a total of 32 minutes.
  Prove that the walk from the Park Office to the Lake Park restaurant is 10 minutes.
-/

def T_HiddenLake_to : ℕ := 15
def T_HiddenLake_from : ℕ := 7
def T_total : ℕ := 32
def T_LakePark_restaurant : ℕ := T_total - (T_HiddenLake_to + T_HiddenLake_from)

theorem walk_time_to_LakePark_restaurant : 
  T_LakePark_restaurant = 10 :=
by
  unfold T_LakePark_restaurant T_HiddenLake_to T_HiddenLake_from T_total
  sorry

end NUMINAMATH_GPT_walk_time_to_LakePark_restaurant_l207_20711


namespace NUMINAMATH_GPT_tank_fill_rate_l207_20744

theorem tank_fill_rate
  (length width depth : ℝ)
  (time_to_fill : ℝ)
  (h_length : length = 10)
  (h_width : width = 6)
  (h_depth : depth = 5)
  (h_time : time_to_fill = 60) : 
  (length * width * depth) / time_to_fill = 5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_tank_fill_rate_l207_20744


namespace NUMINAMATH_GPT_jamie_collects_oysters_l207_20700

theorem jamie_collects_oysters (d : ℕ) (p : ℕ) (r : ℕ) (x : ℕ)
  (h1 : d = 14)
  (h2 : p = 56)
  (h3 : r = 25)
  (h4 : x = p / d * 100 / r) :
  x = 16 :=
by
  sorry

end NUMINAMATH_GPT_jamie_collects_oysters_l207_20700


namespace NUMINAMATH_GPT_max_area_of_pen_l207_20719

theorem max_area_of_pen (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (x : ℝ), (3 * x + x = 60) ∧ (2 * x * x = 450) :=
by
  -- This theorem states that there exists an x such that
  -- the total perimeter with internal divider equals 60,
  -- and the total area of the two squares equals 450.
  use 15
  sorry

end NUMINAMATH_GPT_max_area_of_pen_l207_20719


namespace NUMINAMATH_GPT_fraction_work_left_l207_20726

theorem fraction_work_left (A_days B_days : ℕ) (together_days : ℕ) 
  (H_A : A_days = 20) (H_B : B_days = 30) (H_t : together_days = 4) : 
  (1 : ℚ) - (together_days * ((1 : ℚ) / A_days + (1 : ℚ) / B_days)) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_work_left_l207_20726


namespace NUMINAMATH_GPT_find_max_marks_l207_20727

variable (M : ℝ)
variable (pass_mark : ℝ := 60 / 100)
variable (obtained_marks : ℝ := 200)
variable (additional_marks_needed : ℝ := 80)

theorem find_max_marks (h1 : pass_mark * M = obtained_marks + additional_marks_needed) : M = 467 := 
by
  sorry

end NUMINAMATH_GPT_find_max_marks_l207_20727


namespace NUMINAMATH_GPT_maximize_profit_at_14_yuan_and_720_l207_20748

def initial_cost : ℝ := 8
def initial_price : ℝ := 10
def initial_units_sold : ℝ := 200
def decrease_units_per_half_yuan_increase : ℝ := 10
def increase_price_per_step : ℝ := 0.5

noncomputable def profit (x : ℝ) : ℝ := 
  let selling_price := initial_price + increase_price_per_step * x
  let units_sold := initial_units_sold - decrease_units_per_half_yuan_increase * x
  (selling_price - initial_cost) * units_sold

theorem maximize_profit_at_14_yuan_and_720 :
  profit 8 = 720 ∧ (initial_price + increase_price_per_step * 8 = 14) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_at_14_yuan_and_720_l207_20748


namespace NUMINAMATH_GPT_diane_coins_in_third_piggy_bank_l207_20737

theorem diane_coins_in_third_piggy_bank :
  ∀ n1 n2 n4 n5 n6 : ℕ, n1 = 72 → n2 = 81 → n4 = 99 → n5 = 108 → n6 = 117 → (n4 - (n4 - 9)) = 90 :=
by
  -- sorry is needed to avoid an incomplete proof, as only the statement is required.
  sorry

end NUMINAMATH_GPT_diane_coins_in_third_piggy_bank_l207_20737


namespace NUMINAMATH_GPT_sum_geometric_sequence_l207_20725

theorem sum_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : ∀ n, 2 * a n - 2 = S n) : 
  S n = 2^(n+1) - 2 :=
sorry

end NUMINAMATH_GPT_sum_geometric_sequence_l207_20725


namespace NUMINAMATH_GPT_solve_system_of_equations_l207_20755

theorem solve_system_of_equations : ∃ (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l207_20755


namespace NUMINAMATH_GPT_max_additional_spheres_in_cone_l207_20715

-- Definition of spheres O_{1} and O_{2} properties
def O₁_radius : ℝ := 2
def O₂_radius : ℝ := 3
def height_cone : ℝ := 8

-- Conditions:
def O₁_on_axis (h : ℝ) := height_cone > 0 ∧ h = O₁_radius
def O₁_tangent_top_base := height_cone = O₁_radius + O₁_radius
def O₂_tangent_O₁ := O₁_radius + O₂_radius = 5
def O₂_on_base := O₂_radius = 3

-- Lean theorem stating mathematically equivalent proof problem
theorem max_additional_spheres_in_cone (h : ℝ) :
  O₁_on_axis h → O₁_tangent_top_base →
  O₂_tangent_O₁ → O₂_on_base →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_additional_spheres_in_cone_l207_20715


namespace NUMINAMATH_GPT_min_value_at_x_zero_l207_20761

noncomputable def f (x : ℝ) := Real.sqrt (x^2 + (x + 1)^2) + Real.sqrt (x^2 + (x - 1)^2)

theorem min_value_at_x_zero : ∀ x : ℝ, f x ≥ f 0 := by
  sorry

end NUMINAMATH_GPT_min_value_at_x_zero_l207_20761


namespace NUMINAMATH_GPT_hashN_of_25_l207_20747

def hashN (N : ℝ) : ℝ := 0.6 * N + 2

theorem hashN_of_25 : hashN (hashN (hashN (hashN 25))) = 7.592 :=
by
  sorry

end NUMINAMATH_GPT_hashN_of_25_l207_20747


namespace NUMINAMATH_GPT_gcd_polynomial_primes_l207_20739

theorem gcd_polynomial_primes (a : ℤ) (k : ℤ) (ha : a = 2 * 947 * k) : 
  Int.gcd (3 * a^2 + 47 * a + 101) (a + 19) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_primes_l207_20739


namespace NUMINAMATH_GPT_correctly_calculated_value_l207_20756

theorem correctly_calculated_value (x : ℕ) (h : 5 * x = 40) : 2 * x = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_correctly_calculated_value_l207_20756


namespace NUMINAMATH_GPT_orange_beads_in_necklace_l207_20786

theorem orange_beads_in_necklace (O : ℕ) : 
    (∀ g w o : ℕ, g = 9 ∧ w = 6 ∧ ∃ t : ℕ, t = 45 ∧ 5 * (g + w + O) = 5 * (9 + 6 + O) ∧ 
    ∃ n : ℕ, n = 5 ∧ n * (45) =
    n * (5 * O)) → O = 9 :=
by
  sorry

end NUMINAMATH_GPT_orange_beads_in_necklace_l207_20786


namespace NUMINAMATH_GPT_min_value_PF_PA_l207_20753

open Classical

noncomputable section

def parabola_eq (x y : ℝ) : Prop := y^2 = 16 * x

def point_A : ℝ × ℝ := (1, 2)

def focus_F : ℝ × ℝ := (4, 0)  -- Focus of the given parabola y^2 = 16x

def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def PF_PA (P : ℝ × ℝ) : ℝ :=
  distance P focus_F + distance P point_A

theorem min_value_PF_PA :
  ∃ P : ℝ × ℝ, parabola_eq P.1 P.2 ∧ PF_PA P = 5 :=
sorry

end NUMINAMATH_GPT_min_value_PF_PA_l207_20753


namespace NUMINAMATH_GPT_math_proof_problem_l207_20708

noncomputable def problem_statement : Prop :=
  ∀ (x a b : ℕ), 
  (x + 2 = 5 ∧ x=3) ∧
  (60 / (x + 2) = 36 / x) ∧ 
  (a + b = 90) ∧ 
  (b ≥ 3 * a) ∧ 
  ( ∃ a_max : ℕ, (a_max ≤ a) ∧ (110*a_max + (30*b) = 10520))
  
theorem math_proof_problem : problem_statement := 
  by sorry

end NUMINAMATH_GPT_math_proof_problem_l207_20708


namespace NUMINAMATH_GPT_f_2011_equals_1_l207_20782

-- Define odd function property
def is_odd_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define function with period property
def has_period_3 (f : ℤ → ℤ) : Prop :=
  ∀ x, f (x + 3) = f (x)

-- Define main problem statement
theorem f_2011_equals_1 
  (f : ℤ → ℤ)
  (h1 : is_odd_function f)
  (h2 : has_period_3 f)
  (h3 : f (-1) = -1) 
  : f 2011 = 1 :=
sorry

end NUMINAMATH_GPT_f_2011_equals_1_l207_20782


namespace NUMINAMATH_GPT_january_salary_l207_20797

variable (J F M A My : ℕ)

axiom average_salary_1 : (J + F + M + A) / 4 = 8000
axiom average_salary_2 : (F + M + A + My) / 4 = 8400
axiom may_salary : My = 6500

theorem january_salary : J = 4900 :=
by
  /- To be filled with the proof steps applying the given conditions -/
  sorry

end NUMINAMATH_GPT_january_salary_l207_20797


namespace NUMINAMATH_GPT_middle_digit_base8_l207_20772

theorem middle_digit_base8 (M : ℕ) (e : ℕ) (d f : Fin 8) 
  (M_base8 : M = 64 * d + 8 * e + f)
  (M_base10 : M = 100 * f + 10 * e + d) :
  e = 6 :=
by sorry

end NUMINAMATH_GPT_middle_digit_base8_l207_20772


namespace NUMINAMATH_GPT_long_sleeve_shirts_l207_20783

variable (short_sleeve long_sleeve : Nat)
variable (total_shirts washed_shirts : Nat)
variable (not_washed_shirts : Nat)

-- Given conditions
axiom h1 : short_sleeve = 9
axiom h2 : total_shirts = 29
axiom h3 : not_washed_shirts = 1
axiom h4 : washed_shirts = total_shirts - not_washed_shirts

-- The question to be proved
theorem long_sleeve_shirts : long_sleeve = washed_shirts - short_sleeve := by
  sorry

end NUMINAMATH_GPT_long_sleeve_shirts_l207_20783


namespace NUMINAMATH_GPT_score_difference_l207_20764

noncomputable def mean_score (scores pcts : List ℕ) : ℚ := 
  (List.zipWith (· * ·) scores pcts).sum / 100

def median_score (scores pcts : List ℕ) : ℚ := 75

theorem score_difference :
  let scores := [60, 75, 85, 95]
  let pcts := [20, 50, 15, 15]
  abs (median_score scores pcts - mean_score scores pcts) = 1.5 := by
  sorry

end NUMINAMATH_GPT_score_difference_l207_20764


namespace NUMINAMATH_GPT_eight_faucets_fill_time_in_seconds_l207_20750

open Nat

-- Definitions under the conditions
def four_faucets_rate (gallons : ℕ) (minutes : ℕ) : ℕ := gallons / minutes

def one_faucet_rate (four_faucets_rate : ℕ) : ℕ := four_faucets_rate / 4

def eight_faucets_rate (one_faucet_rate : ℕ) : ℕ := one_faucet_rate * 8

def time_to_fill (rate : ℕ) (gallons : ℕ) : ℕ := gallons / rate

-- Main theorem to prove 
theorem eight_faucets_fill_time_in_seconds (gallons_tub : ℕ) (four_faucets_time : ℕ) :
    let four_faucets_rate := four_faucets_rate 200 8
    let one_faucet_rate := one_faucet_rate four_faucets_rate
    let rate_eight_faucets := eight_faucets_rate one_faucet_rate
    let time_fill := time_to_fill rate_eight_faucets 50
    gallons_tub = 50 ∧ four_faucets_time = 8 ∧ rate_eight_faucets = 50 -> time_fill * 60 = 60 :=
by
    intros
    sorry

end NUMINAMATH_GPT_eight_faucets_fill_time_in_seconds_l207_20750


namespace NUMINAMATH_GPT_no_way_to_write_as_sum_l207_20780

def can_be_written_as_sum (S : ℕ → ℕ) (n : ℕ) (k : ℕ) : Prop :=
  n + k - 1 + (n - 1) * (k - 1) / 2 = 528 ∧ n > 0 ∧ 2 ∣ n ∧ k > 1

theorem no_way_to_write_as_sum : 
  ∀ (S : ℕ → ℕ) (n k : ℕ), can_be_written_as_sum S n k →
    0 = 0 :=
by
  -- Problem states that there are 0 valid ways to write 528 as the sum
  -- of an increasing sequence of two or more consecutive positive integers
  sorry

end NUMINAMATH_GPT_no_way_to_write_as_sum_l207_20780


namespace NUMINAMATH_GPT_find_n_value_l207_20788

theorem find_n_value :
  ∃ m n : ℝ, (4 * x^2 + 8 * x - 448 = 0 → (x + m)^2 = n) ∧ n = 113 :=
by
  sorry

end NUMINAMATH_GPT_find_n_value_l207_20788


namespace NUMINAMATH_GPT_average_growth_rate_inequality_l207_20740

theorem average_growth_rate_inequality (p q x : ℝ) (h₁ : (1+x)^2 = (1+p)*(1+q)) (h₂ : p ≠ q) :
  x < (p + q) / 2 :=
sorry

end NUMINAMATH_GPT_average_growth_rate_inequality_l207_20740


namespace NUMINAMATH_GPT_value_of_last_installment_l207_20749

noncomputable def total_amount_paid_without_processing_fee : ℝ :=
  36 * 2300

noncomputable def total_interest_paid : ℝ :=
  total_amount_paid_without_processing_fee - 35000

noncomputable def last_installment_value : ℝ :=
  2300 + 1000

theorem value_of_last_installment :
  last_installment_value = 3300 :=
  by
    sorry

end NUMINAMATH_GPT_value_of_last_installment_l207_20749


namespace NUMINAMATH_GPT_derivative_of_f_l207_20722

noncomputable def f (x : ℝ) : ℝ := (Real.sin (1 / x)) ^ 3

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : 
  deriv f x = - (3 / x ^ 2) * (Real.sin (1 / x)) ^ 2 * Real.cos (1 / x) :=
by
  sorry 

end NUMINAMATH_GPT_derivative_of_f_l207_20722


namespace NUMINAMATH_GPT_sum_of_angles_around_point_l207_20790

theorem sum_of_angles_around_point (x : ℝ) (h : 6 * x + 3 * x + 4 * x + x + 2 * x = 360) : x = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_angles_around_point_l207_20790


namespace NUMINAMATH_GPT_hyperbola_asymptotes_iff_l207_20798

def hyperbola_asymptotes_orthogonal (a b c d e f : ℝ) : Prop :=
  a + c = 0

theorem hyperbola_asymptotes_iff (a b c d e f : ℝ) :
  (∃ x y : ℝ, a * x^2 + 2 * b * x * y + c * y^2 + d * x + e * y + f = 0) →
  hyperbola_asymptotes_orthogonal a b c d e f ↔ a + c = 0 :=
by sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_iff_l207_20798


namespace NUMINAMATH_GPT_number_of_participants_2005_l207_20734

variable (participants : ℕ → ℕ)
variable (n : ℕ)

-- Conditions
def initial_participants := participants 2001 = 1000
def increase_till_2003 := ∀ n, 2001 ≤ n ∧ n ≤ 2003 → participants (n + 1) = 2 * participants n
def increase_from_2004 := ∀ n, n ≥ 2004 → participants (n + 1) = 2 * participants n + 500

-- Proof problem
theorem number_of_participants_2005 :
    initial_participants participants →
    increase_till_2003 participants →
    increase_from_2004 participants →
    participants 2005 = 17500 :=
by sorry

end NUMINAMATH_GPT_number_of_participants_2005_l207_20734


namespace NUMINAMATH_GPT_amount_paid_is_51_l207_20799

def original_price : ℕ := 204
def discount_fraction : ℚ := 0.75
def paid_fraction : ℚ := 1 - discount_fraction

theorem amount_paid_is_51 : paid_fraction * original_price = 51 := by
  sorry

end NUMINAMATH_GPT_amount_paid_is_51_l207_20799


namespace NUMINAMATH_GPT_pairs_satisfying_x2_minus_y2_eq_45_l207_20716

theorem pairs_satisfying_x2_minus_y2_eq_45 :
  (∃ p : Finset (ℕ × ℕ), (∀ (x y : ℕ), ((x, y) ∈ p → x^2 - y^2 = 45) ∧ (∀ (x y : ℕ), (x, y) ∈ p → 0 < x ∧ 0 < y)) ∧ p.card = 3) :=
by
  sorry

end NUMINAMATH_GPT_pairs_satisfying_x2_minus_y2_eq_45_l207_20716


namespace NUMINAMATH_GPT_number_of_distinct_triangle_areas_l207_20745

noncomputable def distinct_triangle_area_counts : ℕ :=
sorry  -- Placeholder for the proof to derive the correct answer

theorem number_of_distinct_triangle_areas
  (G H I J K L : ℝ × ℝ)
  (h₁ : G.2 = H.2)
  (h₂ : G.2 = I.2)
  (h₃ : G.2 = J.2)
  (h₄ : H.2 = I.2)
  (h₅ : H.2 = J.2)
  (h₆ : I.2 = J.2)
  (h₇ : dist G H = 2)
  (h₈ : dist H I = 2)
  (h₉ : dist I J = 2)
  (h₁₀ : K.2 = L.2 - 2)  -- Assuming constant perpendicular distance between parallel lines
  (h₁₁ : dist K L = 2) : 
  distinct_triangle_area_counts = 3 :=
sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_number_of_distinct_triangle_areas_l207_20745


namespace NUMINAMATH_GPT_intersecting_line_circle_condition_l207_20752

theorem intersecting_line_circle_condition {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x / a + y / b = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) ≥ 1 :=
sorry

end NUMINAMATH_GPT_intersecting_line_circle_condition_l207_20752


namespace NUMINAMATH_GPT_num_valid_pairs_l207_20774

theorem num_valid_pairs : ∃ (n : ℕ), n = 8 ∧ (∀ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b ≤ 150 ∧ ((a + 1 / b) / (1 / a + b) = 17) ↔ (a = 17 * b) ∧ b ≤ 8) :=
by
  sorry

end NUMINAMATH_GPT_num_valid_pairs_l207_20774


namespace NUMINAMATH_GPT_slope_angle_AB_l207_20791

noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (1, 0)

theorem slope_angle_AB :
  let θ := Real.arctan (↑(B.2 - A.2) / ↑(B.1 - A.1))
  θ = 3 * Real.pi / 4 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_slope_angle_AB_l207_20791


namespace NUMINAMATH_GPT_family_members_count_l207_20795

variable (F : ℕ) -- Number of other family members

def annual_cost_per_person : ℕ := 4000 + 12 * 1000
def john_total_cost_for_family (F : ℕ) : ℕ := (F + 1) * annual_cost_per_person / 2

theorem family_members_count :
  john_total_cost_for_family F = 32000 → F = 3 := by
  sorry

end NUMINAMATH_GPT_family_members_count_l207_20795


namespace NUMINAMATH_GPT_probability_two_units_of_origin_l207_20746

def square_vertices (x_min x_max y_min y_max : ℝ) :=
  { p : ℝ × ℝ // x_min ≤ p.1 ∧ p.1 ≤ x_max ∧ y_min ≤ p.2 ∧ p.2 ≤ y_max }

def within_radius (r : ℝ) (origin : ℝ × ℝ) (p : ℝ × ℝ) :=
  (p.1 - origin.1)^2 + (p.2 - origin.2)^2 ≤ r^2

noncomputable def probability_within_radius (x_min x_max y_min y_max r : ℝ) : ℝ :=
  let square_area := (x_max - x_min) * (y_max - y_min)
  let circle_area := r^2 * Real.pi
  circle_area / square_area

theorem probability_two_units_of_origin :
  probability_within_radius (-3) 3 (-3) 3 2 = Real.pi / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_units_of_origin_l207_20746


namespace NUMINAMATH_GPT_ratio_of_line_cutting_median_lines_l207_20765

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem ratio_of_line_cutting_median_lines (A B C P Q : ℝ × ℝ) 
    (hA : A = (1, 0)) (hB : B = (0, 1)) (hC : C = (0, 0)) 
    (h_mid_AB : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) 
    (h_mid_BC : Q = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) 
    (h_ratio : (Real.sqrt (P.1^2 + P.2^2) / Real.sqrt (Q.1^2 + Q.2^2)) = (Real.sqrt (Q.1^2 + Q.2^2) / Real.sqrt (P.1^2 + P.2^2))) :
  (P.1 / Q.1) = golden_ratio :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_line_cutting_median_lines_l207_20765


namespace NUMINAMATH_GPT_averageSpeed_l207_20754

-- Define the total distance driven by Jane
def totalDistance : ℕ := 200

-- Define the total time duration from 6 a.m. to 11 a.m.
def totalTime : ℕ := 5

-- Theorem stating that the average speed is 40 miles per hour
theorem averageSpeed (h1 : totalDistance = 200) (h2 : totalTime = 5) : totalDistance / totalTime = 40 := 
by
  sorry

end NUMINAMATH_GPT_averageSpeed_l207_20754


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l207_20776

def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant :
  in_fourth_quadrant (1, -2) ∧
  ¬ in_fourth_quadrant (2, 1) ∧
  ¬ in_fourth_quadrant (-2, 1) ∧
  ¬ in_fourth_quadrant (-1, -3) :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l207_20776


namespace NUMINAMATH_GPT_smallest_positive_integer_with_18_divisors_l207_20768

theorem smallest_positive_integer_with_18_divisors : ∃ n : ℕ, 0 < n ∧ (∀ d : ℕ, d ∣ n → 0 < d → d ≠ n → (∀ m : ℕ, m ∣ d → m = 1) → n = 180) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_with_18_divisors_l207_20768


namespace NUMINAMATH_GPT_part1_equation_solution_part2_inequality_solution_l207_20762

theorem part1_equation_solution (x : ℝ) (h : x / (x - 1) = (x - 1) / (2 * (x - 1))) : 
  x = -1 :=
sorry

theorem part2_inequality_solution (x : ℝ) (h₁ : 5 * x - 1 > 3 * x - 4) (h₂ : - (1 / 3) * x ≤ 2 / 3 - x) : 
  -3 / 2 < x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_GPT_part1_equation_solution_part2_inequality_solution_l207_20762


namespace NUMINAMATH_GPT_f_def_pos_l207_20721

-- Define f to be an odd function
variable (f : ℝ → ℝ)
-- Define f as an odd function
axiom odd_f (x : ℝ) : f (-x) = -f x

-- Define f when x < 0
axiom f_def_neg (x : ℝ) (h : x < 0) : f x = (Real.cos (3 * x)) + (Real.sin (2 * x))

-- State the theorem to be proven:
theorem f_def_pos (x : ℝ) (h : 0 < x) : f x = - (Real.cos (3 * x)) + (Real.sin (2 * x)) :=
sorry

end NUMINAMATH_GPT_f_def_pos_l207_20721


namespace NUMINAMATH_GPT_no_such_triples_l207_20712

theorem no_such_triples : ¬ ∃ (x y z : ℤ), (xy + yz + zx ≠ 0) ∧ (x^2 + y^2 + z^2) / (xy + yz + zx) = 2016 :=
by
  sorry

end NUMINAMATH_GPT_no_such_triples_l207_20712


namespace NUMINAMATH_GPT_evie_l207_20758

variable (Evie_current_age : ℕ) 

theorem evie's_age_in_one_year
  (h : Evie_current_age + 4 = 3 * (Evie_current_age - 2)) : 
  Evie_current_age + 1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_evie_l207_20758


namespace NUMINAMATH_GPT_volume_of_dug_earth_l207_20702

theorem volume_of_dug_earth :
  let r := 2
  let h := 14
  ∃ V : ℝ, V = Real.pi * r^2 * h ∧ V = 56 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_volume_of_dug_earth_l207_20702


namespace NUMINAMATH_GPT_larger_inscribed_angle_corresponds_to_larger_chord_l207_20763

theorem larger_inscribed_angle_corresponds_to_larger_chord
  (R : ℝ) (α β : ℝ) (hα : α < 90) (hβ : β < 90) (h : α < β)
  (BC LM : ℝ) (hBC : BC = 2 * R * Real.sin α) (hLM : LM = 2 * R * Real.sin β) :
  BC < LM :=
sorry

end NUMINAMATH_GPT_larger_inscribed_angle_corresponds_to_larger_chord_l207_20763


namespace NUMINAMATH_GPT_min_pencils_to_ensure_18_l207_20793

theorem min_pencils_to_ensure_18 :
  ∀ (total red green yellow blue brown black : ℕ),
  total = 120 → red = 35 → green = 23 → yellow = 14 → blue = 26 → brown = 11 → black = 11 →
  ∃ (n : ℕ), n = 88 ∧
  (∀ (picked_pencils : ℕ → ℕ), (
    (picked_pencils 0 + picked_pencils 1 + picked_pencils 2 + picked_pencils 3 + picked_pencils 4 + picked_pencils 5 = n) →
    (picked_pencils 0 ≤ red) → (picked_pencils 1 ≤ green) → (picked_pencils 2 ≤ yellow) →
    (picked_pencils 3 ≤ blue) → (picked_pencils 4 ≤ brown) → (picked_pencils 5 ≤ black) →
    (picked_pencils 0 ≥ 18 ∨ picked_pencils 1 ≥ 18 ∨ picked_pencils 2 ≥ 18 ∨ picked_pencils 3 ≥ 18 ∨ picked_pencils 4 ≥ 18 ∨ picked_pencils 5 ≥ 18)
  )) := 
sorry

end NUMINAMATH_GPT_min_pencils_to_ensure_18_l207_20793
