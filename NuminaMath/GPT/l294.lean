import Mathlib

namespace NUMINAMATH_GPT_sin_cos_value_l294_29405

noncomputable def tan_plus_pi_div_two_eq_two (θ : ℝ) : Prop :=
  Real.tan (θ + Real.pi / 2) = 2

theorem sin_cos_value (θ : ℝ) (h : tan_plus_pi_div_two_eq_two θ) :
  Real.sin θ * Real.cos θ = -2 / 5 :=
sorry

end NUMINAMATH_GPT_sin_cos_value_l294_29405


namespace NUMINAMATH_GPT_reduced_price_per_kg_l294_29478

variable (P : ℝ)
variable (R : ℝ)
variable (Q : ℝ)

theorem reduced_price_per_kg
  (h1 : R = 0.75 * P)
  (h2 : 500 = Q * P)
  (h3 : 500 = (Q + 5) * R)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_per_kg_l294_29478


namespace NUMINAMATH_GPT_decision_has_two_exit_paths_l294_29415

-- Define types representing different flowchart symbols
inductive FlowchartSymbol
| Terminal
| InputOutput
| Process
| Decision

-- Define a function that states the number of exit paths given a flowchart symbol
def exit_paths (s : FlowchartSymbol) : Nat :=
  match s with
  | FlowchartSymbol.Terminal   => 1
  | FlowchartSymbol.InputOutput => 1
  | FlowchartSymbol.Process    => 1
  | FlowchartSymbol.Decision   => 2

-- State the theorem that Decision has two exit paths
theorem decision_has_two_exit_paths : exit_paths FlowchartSymbol.Decision = 2 := by
  sorry

end NUMINAMATH_GPT_decision_has_two_exit_paths_l294_29415


namespace NUMINAMATH_GPT_binomial_n_choose_n_sub_2_l294_29475

theorem binomial_n_choose_n_sub_2 (n : ℕ) (h : 2 ≤ n) : Nat.choose n (n - 2) = n * (n - 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_binomial_n_choose_n_sub_2_l294_29475


namespace NUMINAMATH_GPT_tap_filling_time_l294_29469

theorem tap_filling_time (T : ℝ) 
  (h_total : (1 / 3) = (1 / T + 1 / 15 + 1 / 6)) : T = 10 := 
sorry

end NUMINAMATH_GPT_tap_filling_time_l294_29469


namespace NUMINAMATH_GPT_depth_of_channel_l294_29422

theorem depth_of_channel (top_width bottom_width : ℝ) (area : ℝ) (h : ℝ) 
  (h_top : top_width = 14) (h_bottom : bottom_width = 8) (h_area : area = 770) :
  (1 / 2) * (top_width + bottom_width) * h = area → h = 70 :=
by
  intros h_trapezoid
  sorry

end NUMINAMATH_GPT_depth_of_channel_l294_29422


namespace NUMINAMATH_GPT_manuscript_pages_count_l294_29459

theorem manuscript_pages_count
  (P : ℕ)
  (cost_first_time : ℕ := 5 * P)
  (cost_once_revised : ℕ := 4 * 30)
  (cost_twice_revised : ℕ := 8 * 20)
  (total_cost : ℕ := 780)
  (h : cost_first_time + cost_once_revised + cost_twice_revised = total_cost) :
  P = 100 :=
sorry

end NUMINAMATH_GPT_manuscript_pages_count_l294_29459


namespace NUMINAMATH_GPT_lollipops_left_for_becky_l294_29429
-- Import the Mathlib library

-- Define the conditions as given in the problem
def lemon_lollipops : ℕ := 75
def peppermint_lollipops : ℕ := 210
def watermelon_lollipops : ℕ := 6
def marshmallow_lollipops : ℕ := 504
def friends : ℕ := 13

-- Total number of lollipops
def total_lollipops : ℕ := lemon_lollipops + peppermint_lollipops + watermelon_lollipops + marshmallow_lollipops

-- Statement to prove that the remainder after distributing the total lollipops among friends is 2
theorem lollipops_left_for_becky : total_lollipops % friends = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_lollipops_left_for_becky_l294_29429


namespace NUMINAMATH_GPT_speed_of_A_l294_29431

theorem speed_of_A :
  ∀ (v_A : ℝ), 
    (v_A * 2 + 7 * 2 = 24) → 
    v_A = 5 :=
by
  intro v_A
  intro h
  have h1 : v_A * 2 = 10 := by linarith
  have h2 : v_A = 5 := by linarith
  exact h2

end NUMINAMATH_GPT_speed_of_A_l294_29431


namespace NUMINAMATH_GPT_new_average_is_minus_one_l294_29457

noncomputable def new_average_of_deducted_sequence : ℤ :=
  let n := 15
  let avg := 20
  let seq_sum := n * avg
  let x := (seq_sum - (n * (n-1) / 2)) / n
  let deductions := (n-1) * n * 3 / 2
  let new_sum := seq_sum - deductions
  new_sum / n

theorem new_average_is_minus_one : new_average_of_deducted_sequence = -1 := 
  sorry

end NUMINAMATH_GPT_new_average_is_minus_one_l294_29457


namespace NUMINAMATH_GPT_labor_union_tree_equation_l294_29490

theorem labor_union_tree_equation (x : ℕ) : 2 * x + 21 = 3 * x - 24 := 
sorry

end NUMINAMATH_GPT_labor_union_tree_equation_l294_29490


namespace NUMINAMATH_GPT_sqrt_43_between_6_and_7_l294_29418

theorem sqrt_43_between_6_and_7 : 6 < Real.sqrt 43 ∧ Real.sqrt 43 < 7 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_43_between_6_and_7_l294_29418


namespace NUMINAMATH_GPT_CatsFavoriteNumber_l294_29461

theorem CatsFavoriteNumber :
  ∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧ 
    (∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n = p1 * p2 * p3) ∧ 
    (∀ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      n ≠ a ∧ n ≠ b ∧ n ≠ c ∧ n ≠ d ∧
      a + b - c = d ∨ b + c - d = a ∨ c + d - a = b ∨ d + a - b = c →
      (a = 30 ∧ b = 42 ∧ c = 66 ∧ d = 78)) ∧
    (n = 70) := by
  sorry

end NUMINAMATH_GPT_CatsFavoriteNumber_l294_29461


namespace NUMINAMATH_GPT_percentage_increase_l294_29448

theorem percentage_increase (use_per_six_months : ℝ) (new_annual_use : ℝ) : 
  use_per_six_months = 90 →
  new_annual_use = 216 →
  ((new_annual_use - 2 * use_per_six_months) / (2 * use_per_six_months)) * 100 = 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_percentage_increase_l294_29448


namespace NUMINAMATH_GPT_sequence_count_even_odd_l294_29447

/-- The number of 8-digit sequences such that no two adjacent digits have the same parity
    and the sequence starts with an even number. -/
theorem sequence_count_even_odd : 
  let choices_for_even := 5
  let choices_for_odd := 5
  let total_positions := 8
  (choices_for_even * (choices_for_odd * choices_for_even) ^ (total_positions / 2 - 1)) = 390625 :=
by
  sorry

end NUMINAMATH_GPT_sequence_count_even_odd_l294_29447


namespace NUMINAMATH_GPT_range_of_a_l294_29452

noncomputable def A : Set ℝ := {x | -2 ≤ x ∧ x < 4 }

noncomputable def B (a : ℝ) : Set ℝ := { x | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (a : ℝ) : (B a ⊆ A) ↔ (0 ≤ a ∧ a < 3) := sorry

end NUMINAMATH_GPT_range_of_a_l294_29452


namespace NUMINAMATH_GPT_digits_sum_l294_29425

theorem digits_sum (P Q R : ℕ) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10)
  (h_eq : 100 * P + 10 * Q + R + 10 * Q + R = 1012) :
  P + Q + R = 20 :=
by {
  -- Implementation of the proof will go here
  sorry
}

end NUMINAMATH_GPT_digits_sum_l294_29425


namespace NUMINAMATH_GPT_total_prize_amount_l294_29407

theorem total_prize_amount:
  ∃ P : ℝ, 
  (∃ n m : ℝ, n = 15 ∧ m = 15 ∧ ((2 / 5) * P = (3 / 5) * n * 285) ∧ P = 2565 * 2.5 + 6 * 15 ∧ ∀ i : ℕ, i < m → i ≥ 0 → P ≥ 15)
  ∧ P = 6502.5 :=
sorry

end NUMINAMATH_GPT_total_prize_amount_l294_29407


namespace NUMINAMATH_GPT_bus_passengers_total_l294_29408

theorem bus_passengers_total (children_percent : ℝ) (adults_number : ℝ) (H1 : children_percent = 0.25) (H2 : adults_number = 45) :
  ∃ T : ℝ, T = 60 :=
by
  sorry

end NUMINAMATH_GPT_bus_passengers_total_l294_29408


namespace NUMINAMATH_GPT_percentage_of_invalid_papers_l294_29400

theorem percentage_of_invalid_papers (total_papers : ℕ) (valid_papers : ℕ) (invalid_papers : ℕ) (percentage_invalid : ℚ) 
  (h1 : total_papers = 400) 
  (h2 : valid_papers = 240) 
  (h3 : invalid_papers = total_papers - valid_ppapers)
  (h4 : percentage_invalid = (invalid_papers : ℚ) / total_papers * 100) : 
  percentage_invalid = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_invalid_papers_l294_29400


namespace NUMINAMATH_GPT_concave_quadrilateral_area_l294_29432

noncomputable def area_of_concave_quadrilateral (AB BC CD AD : ℝ) (angle_BCD : ℝ) : ℝ :=
  let BD := Real.sqrt (BC * BC + CD * CD)
  let area_ABD := 0.5 * AB * BD
  let area_BCD := 0.5 * BC * CD
  area_ABD - area_BCD

theorem concave_quadrilateral_area :
  ∀ (AB BC CD AD : ℝ) (angle_BCD : ℝ),
    angle_BCD = Real.pi / 2 ∧ AB = 12 ∧ BC = 4 ∧ CD = 3 ∧ AD = 13 → 
    area_of_concave_quadrilateral AB BC CD AD angle_BCD = 24 :=
by
  intros AB BC CD AD angle_BCD h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end NUMINAMATH_GPT_concave_quadrilateral_area_l294_29432


namespace NUMINAMATH_GPT_calculate_floor_100_p_l294_29438

noncomputable def max_prob_sum_7 : ℝ := 
  let p1 := 0.2
  let p6 := 0.1
  let p2_p5_p3_p4 := 0.7 - p1 - p6
  2 * (p1 * p6 + p2_p5_p3_p4 / 2 ^ 2)

theorem calculate_floor_100_p : ∃ p : ℝ, (⌊100 * max_prob_sum_7⌋ = 28) :=
  by
  sorry

end NUMINAMATH_GPT_calculate_floor_100_p_l294_29438


namespace NUMINAMATH_GPT_hostel_provisions_l294_29487

theorem hostel_provisions (x : ℕ) (h1 : 250 * x = 200 * 40) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_hostel_provisions_l294_29487


namespace NUMINAMATH_GPT_n_squared_plus_d_not_square_l294_29427

theorem n_squared_plus_d_not_square 
  (n : ℕ) (d : ℕ)
  (h_pos_n : n > 0) 
  (h_pos_d : d > 0) 
  (h_div : d ∣ 2 * n^2) : 
  ¬ ∃ m : ℕ, n^2 + d = m^2 := 
sorry

end NUMINAMATH_GPT_n_squared_plus_d_not_square_l294_29427


namespace NUMINAMATH_GPT_inequality_solution_l294_29434

noncomputable def solve_inequality (a : ℝ) : Set ℝ :=
  if a = 0 then 
    {x : ℝ | 1 < x}
  else if 0 < a ∧ a < 2 then 
    {x : ℝ | 1 < x ∧ x < (2 / a)}
  else if a = 2 then 
    ∅
  else if a > 2 then 
    {x : ℝ | (2 / a) < x ∧ x < 1}
  else 
    {x : ℝ | x < (2 / a)} ∪ {x : ℝ | 1 < x}

theorem inequality_solution (a : ℝ) :
  ∀ x : ℝ, (ax^2 - (a + 2) * x + 2 < 0) ↔ (x ∈ solve_inequality a) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l294_29434


namespace NUMINAMATH_GPT_parabola_transform_l294_29486

theorem parabola_transform (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + c = (x - 4)^2 - 3) → 
  b = 4 ∧ c = 6 := 
by
  sorry

end NUMINAMATH_GPT_parabola_transform_l294_29486


namespace NUMINAMATH_GPT_greatest_common_multiple_less_than_bound_l294_29470

-- Define the numbers and the bound
def num1 : ℕ := 15
def num2 : ℕ := 10
def bound : ℕ := 150

-- Define the LCM of num1 and num2
def lcm_num1_num2 : ℕ := Nat.lcm num1 num2

-- Define the greatest multiple of LCM less than bound
def greatest_multiple_less_than_bound (lcm : ℕ) (b : ℕ) : ℕ :=
  (b / lcm) * lcm

-- Main theorem
theorem greatest_common_multiple_less_than_bound :
  greatest_multiple_less_than_bound lcm_num1_num2 bound = 120 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_multiple_less_than_bound_l294_29470


namespace NUMINAMATH_GPT_area_of_highest_points_l294_29439

noncomputable def highest_point_area (u g : ℝ) : ℝ :=
  let x₁ := u^2 / (2 * g)
  let x₂ := 2 * u^2 / g
  (1/4) * ((x₂^2) - (x₁^2))

theorem area_of_highest_points (u g : ℝ) : highest_point_area u g = 3 * u^4 / (4 * g^2) :=
by
  sorry

end NUMINAMATH_GPT_area_of_highest_points_l294_29439


namespace NUMINAMATH_GPT_cistern_emptying_l294_29467

theorem cistern_emptying (h: (3 / 4) / 12 = 1 / 16) : (8 * (1 / 16) = 1 / 2) :=
by sorry

end NUMINAMATH_GPT_cistern_emptying_l294_29467


namespace NUMINAMATH_GPT_min_green_beads_l294_29401

theorem min_green_beads (B R G : ℕ)
  (h_total : B + R + G = 80)
  (h_red_blue : ∀ i j, B ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < R)
  (h_green_red : ∀ i j, R ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < G)
  : G = 27 := 
sorry

end NUMINAMATH_GPT_min_green_beads_l294_29401


namespace NUMINAMATH_GPT_maximum_area_of_rectangle_l294_29492

theorem maximum_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : ∃ A, A = 100 ∧ ∀ x' y', 2 * x' + 2 * y' = 40 → x' * y' ≤ A := by
  sorry

end NUMINAMATH_GPT_maximum_area_of_rectangle_l294_29492


namespace NUMINAMATH_GPT_seven_pow_k_eq_two_l294_29449

theorem seven_pow_k_eq_two {k : ℕ} (h : 7 ^ (4 * k + 2) = 784) : 7 ^ k = 2 := 
by 
  sorry

end NUMINAMATH_GPT_seven_pow_k_eq_two_l294_29449


namespace NUMINAMATH_GPT_tangent_division_l294_29443

theorem tangent_division (a b c d e : ℝ) (h0 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) :
  ∃ t1 t5 : ℝ, t1 = (a + b - c - d + e) / 2 ∧ t5 = (a - b - c + d + e) / 2 ∧ t1 + t5 = a :=
by
  sorry

end NUMINAMATH_GPT_tangent_division_l294_29443


namespace NUMINAMATH_GPT_isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l294_29472

-- Definitions for number of valence electrons
def valence_electrons (atom : String) : ℕ :=
  if atom = "C" then 4
  else if atom = "N" then 5
  else if atom = "O" then 6
  else if atom = "F" then 7
  else if atom = "S" then 6
  else 0

-- Definitions for molecular valence count
def molecule_valence_electrons (molecule : List String) : ℕ :=
  molecule.foldr (λ x acc => acc + valence_electrons x) 0

-- Definitions for specific molecules
def N2_molecule := ["N", "N"]
def CO_molecule := ["C", "O"]
def N2O_molecule := ["N", "N", "O"]
def CO2_molecule := ["C", "O", "O"]
def NO2_minus_molecule := ["N", "O", "O"]
def SO2_molecule := ["S", "O", "O"]
def O3_molecule := ["O", "O", "O"]

-- Isoelectronic property definition
def isoelectronic (mol1 mol2 : List String) : Prop :=
  molecule_valence_electrons mol1 = molecule_valence_electrons mol2

theorem isoelectronic_problem_1_part_1 :
  isoelectronic N2_molecule CO_molecule := sorry

theorem isoelectronic_problem_1_part_2 :
  isoelectronic N2O_molecule CO2_molecule := sorry

theorem isoelectronic_problem_2 :
  isoelectronic NO2_minus_molecule SO2_molecule ∧
  isoelectronic NO2_minus_molecule O3_molecule := sorry

end NUMINAMATH_GPT_isoelectronic_problem_1_part_1_isoelectronic_problem_1_part_2_isoelectronic_problem_2_l294_29472


namespace NUMINAMATH_GPT_cubes_with_one_painted_side_l294_29463

theorem cubes_with_one_painted_side (side_length : ℕ) (one_cm_cubes : ℕ) : 
  side_length = 5 → one_cm_cubes = 54 :=
by 
  intro h 
  sorry

end NUMINAMATH_GPT_cubes_with_one_painted_side_l294_29463


namespace NUMINAMATH_GPT_geometric_series_sum_l294_29460

theorem geometric_series_sum : 
  ∑' n : ℕ, (5 / 3) * (-1 / 3) ^ n = (5 / 4) := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l294_29460


namespace NUMINAMATH_GPT_total_time_to_complete_project_l294_29458

-- Define the initial conditions
def initial_people : ℕ := 6
def initial_days : ℕ := 35
def fraction_completed : ℚ := 1 / 3

-- Define the additional conditions after more people joined
def additional_people : ℕ := initial_people
def total_people : ℕ := initial_people + additional_people
def remaining_fraction : ℚ := 1 - fraction_completed

-- Total time taken to complete the project
theorem total_time_to_complete_project (initial_people initial_days additional_people : ℕ) (fraction_completed remaining_fraction : ℚ)
  (h1 : initial_people * initial_days * fraction_completed = 1/3) 
  (h2 : additional_people = initial_people) 
  (h3 : total_people = initial_people + additional_people)
  (h4 : remaining_fraction = 1 - fraction_completed) : 
  (initial_days + (remaining_fraction / (total_people * (fraction_completed / (initial_people * initial_days)))) = 70) :=
sorry

end NUMINAMATH_GPT_total_time_to_complete_project_l294_29458


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l294_29483

def P (x : ℝ) : Prop := 2 < x ∧ x < 4
def Q (x : ℝ) : Prop := Real.log x < Real.exp 1

theorem sufficient_but_not_necessary (x : ℝ) : P x → Q x ∧ (¬ ∀ x, Q x → P x) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l294_29483


namespace NUMINAMATH_GPT_geometric_sequence_sum_l294_29489

variable {a : ℕ → ℕ}

-- Defining the geometric sequence and the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 (a : ℕ → ℕ) : Prop :=
  a 1 = 3

def condition2 (a : ℕ → ℕ) : Prop :=
  a 1 + a 3 + a 5 = 21

-- The main theorem
theorem geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) 
  (h1 : condition1 a) (h2: condition2 a) (hq : is_geometric_sequence a q) : 
  a 3 + a 5 + a 7 = 42 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l294_29489


namespace NUMINAMATH_GPT_smallest_m_satisfying_condition_l294_29404

def D (n : ℕ) : Finset ℕ := (n.divisors : Finset ℕ)

def F (n i : ℕ) : Finset ℕ :=
  (D n).filter (λ a => a % 4 = i)

def f (n i : ℕ) : ℕ :=
  (F n i).card

theorem smallest_m_satisfying_condition :
  ∃ m : ℕ, f m 0 + f m 1 - f m 2 - f m 3 = 2017 ∧
           m = 2^34 * 3^6 * 7^2 * 11^2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_satisfying_condition_l294_29404


namespace NUMINAMATH_GPT_probability_sum_of_two_draws_is_three_l294_29471

theorem probability_sum_of_two_draws_is_three :
  let outcomes := [(1, 1), (1, 2), (2, 1), (2, 2)]
  let favorable := [(1, 2), (2, 1)]
  (favorable.length : ℚ) / (outcomes.length : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_sum_of_two_draws_is_three_l294_29471


namespace NUMINAMATH_GPT_mod_congruence_zero_iff_l294_29426

theorem mod_congruence_zero_iff
  (a b c d n : ℕ)
  (h1 : a * c ≡ 0 [MOD n])
  (h2 : b * c + a * d ≡ 0 [MOD n]) :
  b * c ≡ 0 [MOD n] ∧ a * d ≡ 0 [MOD n] :=
by
  sorry

end NUMINAMATH_GPT_mod_congruence_zero_iff_l294_29426


namespace NUMINAMATH_GPT_tomatoes_ready_for_sale_l294_29430

-- Define all conditions
def initial_shipment := 1000 -- kg of tomatoes on Friday
def sold_on_saturday := 300 -- kg of tomatoes sold on Saturday
def rotten_on_sunday := 200 -- kg of tomatoes rotted on Sunday
def additional_shipment := 2 * initial_shipment -- kg of tomatoes arrived on Monday

-- Define the final calculation to prove
theorem tomatoes_ready_for_sale : 
  initial_shipment - sold_on_saturday - rotten_on_sunday + additional_shipment = 2500 := 
by
  sorry

end NUMINAMATH_GPT_tomatoes_ready_for_sale_l294_29430


namespace NUMINAMATH_GPT_total_weight_apples_l294_29473

variable (Minjae_weight : ℝ) (Father_weight : ℝ)

theorem total_weight_apples (h1 : Minjae_weight = 2.6) (h2 : Father_weight = 5.98) :
  Minjae_weight + Father_weight = 8.58 :=
by 
  sorry

end NUMINAMATH_GPT_total_weight_apples_l294_29473


namespace NUMINAMATH_GPT_exists_four_distinct_indices_l294_29414

theorem exists_four_distinct_indices
  (a : Fin 5 → ℝ)
  (h : ∀ i, 0 < a i) :
  ∃ i j k l : (Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_four_distinct_indices_l294_29414


namespace NUMINAMATH_GPT_problem_solution_l294_29497

noncomputable def given_problem : ℝ := (Real.pi - 3)^0 - Real.sqrt 8 + 2 * Real.sin (45 * Real.pi / 180) + (1 / 2)⁻¹

theorem problem_solution : given_problem = 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l294_29497


namespace NUMINAMATH_GPT_area_of_square_KLMN_is_25_l294_29466

-- Given a square ABCD with area 25
def ABCD_area_is_25 : Prop :=
  ∃ s : ℝ, (s * s = 25)

-- Given points K, L, M, and N forming isosceles right triangles with the sides of the square
def isosceles_right_triangles_at_vertices (A B C D K L M N : ℝ) : Prop :=
  ∃ (a b c d : ℝ),
    (a = b) ∧ (c = d) ∧
    (K - A)^2 + (B - K)^2 = (A - B)^2 ∧  -- AKB
    (L - B)^2 + (C - L)^2 = (B - C)^2 ∧  -- BLC
    (M - C)^2 + (D - M)^2 = (C - D)^2 ∧  -- CMD
    (N - D)^2 + (A - N)^2 = (D - A)^2    -- DNA

-- Given that KLMN is a square
def KLMN_is_square (K L M N : ℝ) : Prop :=
  (K - L)^2 + (L - M)^2 = (M - N)^2 + (N - K)^2

-- Proving that the area of square KLMN is 25 given the conditions
theorem area_of_square_KLMN_is_25 (A B C D K L M N : ℝ) :
  ABCD_area_is_25 → isosceles_right_triangles_at_vertices A B C D K L M N → KLMN_is_square K L M N → ∃s, s * s = 25 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_area_of_square_KLMN_is_25_l294_29466


namespace NUMINAMATH_GPT_greatest_multiple_of_5_l294_29423

theorem greatest_multiple_of_5 (y : ℕ) (h1 : y > 0) (h2 : y % 5 = 0) (h3 : y^3 < 8000) : y ≤ 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_multiple_of_5_l294_29423


namespace NUMINAMATH_GPT_wheel_distance_travelled_l294_29411

noncomputable def radius : ℝ := 3
noncomputable def num_revolutions : ℝ := 3
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def total_distance (r : ℝ) (n : ℝ) : ℝ := n * circumference r

theorem wheel_distance_travelled :
  total_distance radius num_revolutions = 18 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_wheel_distance_travelled_l294_29411


namespace NUMINAMATH_GPT_problem_statement_l294_29417

noncomputable def two_arccos_equals_arcsin : Prop :=
  2 * Real.arccos (3 / 5) = Real.arcsin (24 / 25)

theorem problem_statement : two_arccos_equals_arcsin :=
  sorry

end NUMINAMATH_GPT_problem_statement_l294_29417


namespace NUMINAMATH_GPT_lcm_subtract100_correct_l294_29454

noncomputable def lcm1364_884_subtract_100 : ℕ :=
  let a := 1364
  let b := 884
  let lcm_ab := Nat.lcm a b
  lcm_ab - 100

theorem lcm_subtract100_correct : lcm1364_884_subtract_100 = 1509692 := by
  sorry

end NUMINAMATH_GPT_lcm_subtract100_correct_l294_29454


namespace NUMINAMATH_GPT_solution_set_of_inequality_l294_29468

theorem solution_set_of_inequality (x : ℝ) : |5 * x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l294_29468


namespace NUMINAMATH_GPT_given_conditions_l294_29465

theorem given_conditions :
  ∀ (t : ℝ), t > 0 → t ≠ 1 → 
  let x := t^(2/(t-1))
  let y := t^((t+1)/(t-1))
  ¬ ((y * x^(1/y) = x * y^(1/x)) ∨ (y * x^y = x * y^x) ∨ (y^x = x^y) ∨ (x^(x+y) = y^(x+y))) :=
by
  intros t ht_pos ht_ne_1 x_def y_def
  let x := x_def
  let y := y_def
  sorry

end NUMINAMATH_GPT_given_conditions_l294_29465


namespace NUMINAMATH_GPT_sequence_sum_general_term_l294_29412

theorem sequence_sum_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + n) : ∀ n, a n = 2 * n :=
by 
  sorry

end NUMINAMATH_GPT_sequence_sum_general_term_l294_29412


namespace NUMINAMATH_GPT_polygon_sides_l294_29455

-- Definitions based on the conditions provided
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

def sum_exterior_angles : ℝ := 360 

def condition (n : ℕ) : Prop :=
  sum_interior_angles n = 2 * sum_exterior_angles + 180

-- Main theorem based on the correct answer
theorem polygon_sides (n : ℕ) (h : condition n) : n = 7 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l294_29455


namespace NUMINAMATH_GPT_true_statements_proved_l294_29499

-- Conditions
def A : Prop := ∃ n : ℕ, 25 = 5 * n
def B : Prop := (∃ m1 : ℕ, 209 = 19 * m1) ∧ (¬ ∃ m2 : ℕ, 63 = 19 * m2)
def C : Prop := (¬ ∃ k1 : ℕ, 90 = 30 * k1) ∧ (¬ ∃ k2 : ℕ, 49 = 30 * k2)
def D : Prop := (∃ l1 : ℕ, 34 = 17 * l1) ∧ (¬ ∃ l2 : ℕ, 68 = 17 * l2)
def E : Prop := ∃ q : ℕ, 140 = 7 * q

-- Correct statements
def TrueStatements : Prop := A ∧ B ∧ E ∧ ¬C ∧ ¬D

-- Lean statement to prove
theorem true_statements_proved : TrueStatements := 
by
  sorry

end NUMINAMATH_GPT_true_statements_proved_l294_29499


namespace NUMINAMATH_GPT_square_side_length_l294_29428

theorem square_side_length (s : ℝ) (h : s^2 + s - 4 * s = 4) : s = 4 :=
sorry

end NUMINAMATH_GPT_square_side_length_l294_29428


namespace NUMINAMATH_GPT_polynomial_division_l294_29416

variable (x : ℝ)

theorem polynomial_division :
  ((3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 - x^2 + 6 * x - 8)) / (x + 1) 
  = (x^2 + 4 * x - 15 + 25 / (x+1)) :=
by sorry

end NUMINAMATH_GPT_polynomial_division_l294_29416


namespace NUMINAMATH_GPT_a_2009_eq_1_a_2014_eq_0_l294_29450

section
variable (a : ℕ → ℕ)
variable (n : ℕ)

-- Condition 1: a_{4n-3} = 1
axiom cond1 : ∀ n : ℕ, a (4 * n - 3) = 1

-- Condition 2: a_{4n-1} = 0
axiom cond2 : ∀ n : ℕ, a (4 * n - 1) = 0

-- Condition 3: a_{2n} = a_n
axiom cond3 : ∀ n : ℕ, a (2 * n) = a n

-- Theorem: a_{2009} = 1
theorem a_2009_eq_1 : a 2009 = 1 := by
  sorry

-- Theorem: a_{2014} = 0
theorem a_2014_eq_0 : a 2014 = 0 := by
  sorry

end

end NUMINAMATH_GPT_a_2009_eq_1_a_2014_eq_0_l294_29450


namespace NUMINAMATH_GPT_sum_of_integers_l294_29402

theorem sum_of_integers (x y : ℕ) (h1 : x = y + 3) (h2 : x^3 - y^3 = 63) : x + y = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l294_29402


namespace NUMINAMATH_GPT_min_max_f_l294_29436

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - Real.cos x

theorem min_max_f :
  (∀ x, 2 * (Real.sin (x / 2))^2 = 1 - Real.cos x) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 5 / 4) :=
by 
  intros h x
  sorry

end NUMINAMATH_GPT_min_max_f_l294_29436


namespace NUMINAMATH_GPT_dr_reeds_statement_l294_29496

variables (P Q : Prop)

theorem dr_reeds_statement (h : P → Q) : ¬Q → ¬P :=
by sorry

end NUMINAMATH_GPT_dr_reeds_statement_l294_29496


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l294_29494

theorem relationship_between_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h₁ : a = (10 ^ 1988 + 1) / (10 ^ 1989 + 1))
  (h₂ : b = (10 ^ 1987 + 1) / (10 ^ 1988 + 1))
  (h₃ : c = (10 ^ 1987 + 9) / (10 ^ 1988 + 9)) :
  a < b ∧ b < c := 
sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l294_29494


namespace NUMINAMATH_GPT_hundred_chicken_problem_l294_29476

theorem hundred_chicken_problem :
  ∃ (x y : ℕ), x + y + 81 = 100 ∧ 5 * x + 3 * y + 81 / 3 = 100 := 
by
  sorry

end NUMINAMATH_GPT_hundred_chicken_problem_l294_29476


namespace NUMINAMATH_GPT_range_sum_of_h_l294_29444

noncomputable def h (x : ℝ) : ℝ := 5 / (5 + 3 * x^2)

theorem range_sum_of_h : 
  (∃ a b : ℝ, (∀ x : ℝ, 0 < h x ∧ h x ≤ 1) ∧ a = 0 ∧ b = 1 ∧ a + b = 1) :=
sorry

end NUMINAMATH_GPT_range_sum_of_h_l294_29444


namespace NUMINAMATH_GPT_tangent_points_l294_29477

noncomputable def curve (x : ℝ) : ℝ := x^3 - x - 1

theorem tangent_points (x y : ℝ) (h : y = curve x) (slope_line : ℝ) (h_slope : slope_line = -1/2)
  (tangent_perpendicular : (3 * x^2 - 1) = 2) :
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := sorry

end NUMINAMATH_GPT_tangent_points_l294_29477


namespace NUMINAMATH_GPT_average_percentage_reduction_equation_l294_29406

theorem average_percentage_reduction_equation (x : ℝ) : 200 * (1 - x)^2 = 162 :=
by 
  sorry

end NUMINAMATH_GPT_average_percentage_reduction_equation_l294_29406


namespace NUMINAMATH_GPT_exists_linear_function_l294_29445

-- Define the properties of the function f
def is_contraction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f x - f y| ≤ |x - y|

-- Define the property of an arithmetic progression
def is_arith_seq (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ d : ℝ, ∀ n : ℕ, (f^[n] x) = x + n * d

-- Main theorem to prove
theorem exists_linear_function (f : ℝ → ℝ) (h1 : is_contraction f) (h2 : is_arith_seq f) : ∃ a : ℝ, ∀ x : ℝ, f x = x + a :=
sorry

end NUMINAMATH_GPT_exists_linear_function_l294_29445


namespace NUMINAMATH_GPT_greatest_b_not_in_range_l294_29419

theorem greatest_b_not_in_range : ∃ b : ℤ, b = 10 ∧ ∀ x : ℝ, x^2 + (b:ℝ) * x + 20 ≠ -7 := sorry

end NUMINAMATH_GPT_greatest_b_not_in_range_l294_29419


namespace NUMINAMATH_GPT_intersection_A_B_l294_29485

open Set

def A : Set ℝ := Icc 1 2

def B : Set ℤ := {x : ℤ | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B :
  (A ∩ (coe '' B) : Set ℝ) = {1, 2} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l294_29485


namespace NUMINAMATH_GPT_range_of_b_l294_29456

theorem range_of_b (a b c : ℝ) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 24) : 
  1 ≤ b ∧ b ≤ 5 := 
sorry

end NUMINAMATH_GPT_range_of_b_l294_29456


namespace NUMINAMATH_GPT_value_of_k_l294_29482

theorem value_of_k :
  ∃ k, k = 2 ∧ (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 5 ∧
                ∀ (s t : ℕ), (s, t) ∈ pairs → s = k * t) :=
by 
sorry

end NUMINAMATH_GPT_value_of_k_l294_29482


namespace NUMINAMATH_GPT_average_speed_without_stoppages_l294_29442

variables (d : ℝ) (t : ℝ) (v_no_stop : ℝ)

-- The train stops for 12 minutes per hour
def stoppage_per_hour := 12 / 60
def moving_fraction := 1 - stoppage_per_hour

-- Given speed with stoppages is 160 km/h
def speed_with_stoppage := 160

-- Average speed of the train without stoppages
def speed_without_stoppage := speed_with_stoppage / moving_fraction

-- The average speed without stoppages should equal 200 km/h
theorem average_speed_without_stoppages : speed_without_stoppage = 200 :=
by
  unfold speed_without_stoppage
  unfold moving_fraction
  unfold stoppage_per_hour
  norm_num
  sorry

end NUMINAMATH_GPT_average_speed_without_stoppages_l294_29442


namespace NUMINAMATH_GPT_sausages_fried_l294_29480

def num_eggs : ℕ := 6
def time_per_sausage : ℕ := 5
def time_per_egg : ℕ := 4
def total_time : ℕ := 39
def time_per_sauteurs (S : ℕ) : ℕ := S * time_per_sausage

theorem sausages_fried (S : ℕ) (h : num_eggs * time_per_egg + S * time_per_sausage = total_time) : S = 3 :=
by
  sorry

end NUMINAMATH_GPT_sausages_fried_l294_29480


namespace NUMINAMATH_GPT_product_of_terms_l294_29440

variable (a : ℕ → ℝ)

-- Conditions: the sequence is geometric, a_1 = 1, a_10 = 3.
axiom geometric_sequence : ∀ n m : ℕ, a n * a m = a 1 * a (n + m - 1)

axiom a_1_eq_one : a 1 = 1
axiom a_10_eq_three : a 10 = 3

-- We need to prove that the product a_2a_3a_4a_5a_6a_7a_8a_9 = 81.
theorem product_of_terms : a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end NUMINAMATH_GPT_product_of_terms_l294_29440


namespace NUMINAMATH_GPT_find_a_for_odd_function_l294_29441

noncomputable def f (a x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem find_a_for_odd_function (a : ℝ) :
  (∀ x : ℝ, f a x + f a (-x) = 0) ↔ a = -1 := sorry

end NUMINAMATH_GPT_find_a_for_odd_function_l294_29441


namespace NUMINAMATH_GPT_simplify_sub_polynomials_l294_29420

def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 5 * r - 4
def g (r : ℝ) : ℝ := r^3 + 3 * r^2 + 7 * r - 2

theorem simplify_sub_polynomials (r : ℝ) : f r - g r = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end NUMINAMATH_GPT_simplify_sub_polynomials_l294_29420


namespace NUMINAMATH_GPT_train_speed_l294_29493

theorem train_speed (v : ℝ) (d : ℝ) : 
  (v > 0) →
  (d > 0) →
  (d + (d - 55) = 495) →
  (d / v = (d - 55) / 25) →
  v = 31.25 := 
by
  intros hv hd hdist heqn
  -- We can leave the proof part out because we only need the statement
  sorry

end NUMINAMATH_GPT_train_speed_l294_29493


namespace NUMINAMATH_GPT_rational_m_abs_nonneg_l294_29481

theorem rational_m_abs_nonneg (m : ℚ) : m + |m| ≥ 0 :=
by sorry

end NUMINAMATH_GPT_rational_m_abs_nonneg_l294_29481


namespace NUMINAMATH_GPT_find_a_l294_29488

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := 
sorry

end NUMINAMATH_GPT_find_a_l294_29488


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_l294_29424

theorem arithmetic_sequence_30th_term :
  let a₁ := 4
  let d₁ := 6
  let n := 30
  (a₁ + (n - 1) * d₁) = 178 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_l294_29424


namespace NUMINAMATH_GPT_condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l294_29498

variables {a b : ℝ}

theorem condition_3_implies_at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

theorem condition_5_implies_at_least_one_gt_one (h : ab > 1) : a > 1 ∨ b > 1 :=
sorry

end NUMINAMATH_GPT_condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l294_29498


namespace NUMINAMATH_GPT_min_value_a_2b_l294_29433

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 3 / b = 1) :
  a + 2 * b = 7 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_min_value_a_2b_l294_29433


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l294_29484

theorem solve_quadratic_inequality :
  { x : ℝ | -3 * x^2 + 8 * x + 5 < 0 } = { x : ℝ | x < -1 ∨ x > 5 / 3 } :=
sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l294_29484


namespace NUMINAMATH_GPT_xiaoming_original_phone_number_l294_29474

variable (d1 d2 d3 d4 d5 d6 : Nat)

def original_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

def upgraded_phone_number (d1 d2 d3 d4 d5 d6 : Nat) : Nat :=
  20000000 + 1000000 * d1 + 80000 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem xiaoming_original_phone_number :
  let x := original_phone_number d1 d2 d3 d4 d5 d6
  let x' := upgraded_phone_number d1 d2 d3 d4 d5 d6
  (x' = 81 * x) → (x = 282500) :=
by
  sorry

end NUMINAMATH_GPT_xiaoming_original_phone_number_l294_29474


namespace NUMINAMATH_GPT_num_people_in_group_l294_29437

-- Define constants and conditions
def cost_per_set : ℕ := 3  -- $3 to make 4 S'mores
def smores_per_set : ℕ := 4
def total_cost : ℕ := 18   -- $18 total cost
def smores_per_person : ℕ := 3

-- Calculate total S'mores that can be made
def total_sets : ℕ := total_cost / cost_per_set
def total_smores : ℕ := total_sets * smores_per_set

-- Proof problem statement
theorem num_people_in_group : (total_smores / smores_per_person) = 8 :=
by
  sorry

end NUMINAMATH_GPT_num_people_in_group_l294_29437


namespace NUMINAMATH_GPT_dice_sum_probability_l294_29453

theorem dice_sum_probability (n : ℕ) (h : ∃ k : ℕ, (8 : ℕ) * k + k = 12) : n = 330 :=
sorry

end NUMINAMATH_GPT_dice_sum_probability_l294_29453


namespace NUMINAMATH_GPT_f_at_3_l294_29495

variable {R : Type} [LinearOrderedField R]

-- Define odd function
def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

-- Define the given function f and its properties
variables (f : R → R)
  (h_odd : is_odd_function f)
  (h_domain : ∀ x : R, true) -- domain is R implicitly
  (h_eq : ∀ x : R, f x + f (2 - x) = 4)

-- Prove that f(3) = 6
theorem f_at_3 : f 3 = 6 :=
  sorry

end NUMINAMATH_GPT_f_at_3_l294_29495


namespace NUMINAMATH_GPT_infinite_series_sum_l294_29479

theorem infinite_series_sum :
  ∑' n : ℕ, n / (8 : ℝ) ^ n = (8 / 49 : ℝ) :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l294_29479


namespace NUMINAMATH_GPT_time_for_a_to_complete_one_round_l294_29435

theorem time_for_a_to_complete_one_round (T_a T_b : ℝ) 
  (h1 : 4 * T_a = 3 * T_b)
  (h2 : T_b = T_a + 10) : 
  T_a = 30 := by
  sorry

end NUMINAMATH_GPT_time_for_a_to_complete_one_round_l294_29435


namespace NUMINAMATH_GPT_greatest_4_digit_number_l294_29446

theorem greatest_4_digit_number
  (n : ℕ)
  (h1 : n % 5 = 3)
  (h2 : n % 9 = 2)
  (h3 : 1000 ≤ n)
  (h4 : n < 10000) :
  n = 9962 := 
sorry

end NUMINAMATH_GPT_greatest_4_digit_number_l294_29446


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l294_29464

theorem sufficient_and_necessary_condition (x : ℝ) :
  x^2 - 4 * x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l294_29464


namespace NUMINAMATH_GPT_expression_value_l294_29462

def a : ℝ := 0.96
def b : ℝ := 0.1

theorem expression_value : (a^3 - (b^3 / a^2) + 0.096 + b^2) = 0.989651 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l294_29462


namespace NUMINAMATH_GPT_area_of_triangle_l294_29413

theorem area_of_triangle (s1 s2 s3 : ℕ) (h1 : s1^2 = 36) (h2 : s2^2 = 64) (h3 : s3^2 = 100) (h4 : s1^2 + s2^2 = s3^2) :
  (1 / 2 : ℚ) * s1 * s2 = 24 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l294_29413


namespace NUMINAMATH_GPT_count_obtuse_triangle_values_k_l294_29410

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  if a ≥ b ∧ a ≥ c then a * a > b * b + c * c 
  else if b ≥ a ∧ b ≥ c then b * b > a * a + c * c
  else c * c > a * a + b * b

theorem count_obtuse_triangle_values_k :
  ∃! (k : ℕ), is_triangle 8 18 k ∧ is_obtuse_triangle 8 18 k :=
sorry

end NUMINAMATH_GPT_count_obtuse_triangle_values_k_l294_29410


namespace NUMINAMATH_GPT_students_play_neither_sport_l294_29421

def total_students : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def both_players : ℕ := 10

theorem students_play_neither_sport :
  total_students - (hockey_players + basketball_players - both_players) = 4 :=
by
  sorry

end NUMINAMATH_GPT_students_play_neither_sport_l294_29421


namespace NUMINAMATH_GPT_relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l294_29491

theorem relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the sufficiency part
theorem sufficiency_x_lt_1 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the necessity part
theorem necessity_x_lt_1 (x : ℝ) :
  (x^2 - 4 * x + 3 > 0) → (x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_GPT_relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l294_29491


namespace NUMINAMATH_GPT_union_M_N_equals_0_1_5_l294_29451

def M : Set ℝ := { x | x^2 - 6 * x + 5 = 0 }
def N : Set ℝ := { x | x^2 - 5 * x = 0 }

theorem union_M_N_equals_0_1_5 : M ∪ N = {0, 1, 5} := by
  sorry

end NUMINAMATH_GPT_union_M_N_equals_0_1_5_l294_29451


namespace NUMINAMATH_GPT_martin_total_distance_l294_29409

noncomputable def calculate_distance_traveled : ℕ :=
  let segment1 := 70 * 3 -- 210 km
  let segment2 := 80 * 4 -- 320 km
  let segment3 := 65 * 3 -- 195 km
  let segment4 := 50 * 2 -- 100 km
  let segment5 := 90 * 4 -- 360 km
  segment1 + segment2 + segment3 + segment4 + segment5

theorem martin_total_distance : calculate_distance_traveled = 1185 :=
by
  sorry

end NUMINAMATH_GPT_martin_total_distance_l294_29409


namespace NUMINAMATH_GPT_number_of_paintings_l294_29403

def is_valid_painting (grid : Matrix (Fin 3) (Fin 3) Bool) : Prop :=
  ∀ i j, grid i j = true → 
    (∀ k, k.succ < 3 → grid k j = true → ¬ grid (k.succ) j = false) ∧
    (∀ l, l.succ < 3 → grid i l = true → ¬ grid i (l.succ) = false)

theorem number_of_paintings : 
  ∃ n, n = 50 ∧ 
       ∃ f : Finset (Matrix (Fin 3) (Fin 3) Bool), 
         (∀ grid ∈ f, is_valid_painting grid) ∧ 
         Finset.card f = n :=
sorry

end NUMINAMATH_GPT_number_of_paintings_l294_29403
