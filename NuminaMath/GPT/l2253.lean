import Mathlib

namespace NUMINAMATH_GPT_discount_coupon_value_l2253_225370

theorem discount_coupon_value :
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  total_cost - amount_paid = 4 := by
  intros
  let hamburger_cost := 2 * 5
  let cola_cost := 3 * 2
  let total_cost := hamburger_cost + cola_cost
  let amount_paid := 12
  show total_cost - amount_paid = 4
  sorry

end NUMINAMATH_GPT_discount_coupon_value_l2253_225370


namespace NUMINAMATH_GPT_find_S_11_l2253_225328

variables (a : ℕ → ℤ)
variables (d : ℤ) (n : ℕ)

def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

noncomputable def a_3 := a 3
noncomputable def a_6 := a 6
noncomputable def a_9 := a 9

theorem find_S_11
  (h1 : is_arithmetic_sequence a d)
  (h2 : a_3 + a_9 = 18 - a_6) :
  sum_first_n_terms a 11 = 66 :=
sorry

end NUMINAMATH_GPT_find_S_11_l2253_225328


namespace NUMINAMATH_GPT_find_x_in_interval_l2253_225358

theorem find_x_in_interval (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
  abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2 → 
  Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_in_interval_l2253_225358


namespace NUMINAMATH_GPT_find_a_exactly_two_solutions_l2253_225302

theorem find_a_exactly_two_solutions :
  (∀ x y : ℝ, |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ↔ (a = 4 ∨ a = 100) :=
sorry

end NUMINAMATH_GPT_find_a_exactly_two_solutions_l2253_225302


namespace NUMINAMATH_GPT_arcsin_neg_one_eq_neg_pi_div_two_l2253_225366

theorem arcsin_neg_one_eq_neg_pi_div_two : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_neg_one_eq_neg_pi_div_two_l2253_225366


namespace NUMINAMATH_GPT_sandy_has_32_fish_l2253_225346

-- Define the initial number of pet fish Sandy has
def initial_fish : Nat := 26

-- Define the number of fish Sandy bought
def fish_bought : Nat := 6

-- Define the total number of pet fish Sandy has now
def total_fish : Nat := initial_fish + fish_bought

-- Prove that Sandy now has 32 pet fish
theorem sandy_has_32_fish : total_fish = 32 :=
by
  sorry

end NUMINAMATH_GPT_sandy_has_32_fish_l2253_225346


namespace NUMINAMATH_GPT_appropriate_sampling_methods_l2253_225379

structure Region :=
  (total_households : ℕ)
  (farmer_households : ℕ)
  (worker_households : ℕ)
  (sample_size : ℕ)

theorem appropriate_sampling_methods (r : Region) 
  (h_total: r.total_households = 2004)
  (h_farmers: r.farmer_households = 1600)
  (h_workers: r.worker_households = 303)
  (h_sample: r.sample_size = 40) :
  ("Simple random sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Systematic sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) ∧
  ("Stratified sampling" ∈ ["Simple random sampling", "Systematic sampling", "Stratified sampling"]) :=
by
  sorry

end NUMINAMATH_GPT_appropriate_sampling_methods_l2253_225379


namespace NUMINAMATH_GPT_range_of_a_l2253_225380

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → ax^2 - 2 * x + 2 > 0) ↔ (a > 1/2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2253_225380


namespace NUMINAMATH_GPT_initial_lives_emily_l2253_225314

theorem initial_lives_emily (L : ℕ) (h1 : L - 25 + 24 = 41) : L = 42 :=
by
  sorry

end NUMINAMATH_GPT_initial_lives_emily_l2253_225314


namespace NUMINAMATH_GPT_leftover_yarn_after_square_l2253_225368

theorem leftover_yarn_after_square (total_yarn : ℕ) (side_length : ℕ) (left_yarn : ℕ) :
  total_yarn = 35 →
  (4 * side_length ≤ total_yarn ∧ (∀ s : ℕ, s > side_length → 4 * s > total_yarn)) →
  left_yarn = total_yarn - 4 * side_length →
  left_yarn = 3 :=
by
  sorry

end NUMINAMATH_GPT_leftover_yarn_after_square_l2253_225368


namespace NUMINAMATH_GPT_bert_same_kangaroos_as_kameron_in_40_days_l2253_225390

theorem bert_same_kangaroos_as_kameron_in_40_days
  (k : ℕ := 100)
  (b : ℕ := 20)
  (r : ℕ := 2) :
  ∃ t : ℕ, t = 40 ∧ b + t * r = k := by
  sorry

end NUMINAMATH_GPT_bert_same_kangaroos_as_kameron_in_40_days_l2253_225390


namespace NUMINAMATH_GPT_subcommittee_ways_l2253_225322

theorem subcommittee_ways :
  ∃ (n : ℕ), n = Nat.choose 10 4 * Nat.choose 7 2 ∧ n = 4410 :=
by
  use 4410
  sorry

end NUMINAMATH_GPT_subcommittee_ways_l2253_225322


namespace NUMINAMATH_GPT_Robert_salary_loss_l2253_225344

theorem Robert_salary_loss (S : ℝ) (x : ℝ) (h : x ≠ 0) (h1 : (S - (x/100) * S + (x/100) * (S - (x/100) * S) = (96/100) * S)) : x = 20 :=
by sorry

end NUMINAMATH_GPT_Robert_salary_loss_l2253_225344


namespace NUMINAMATH_GPT_solve_for_x_l2253_225371

theorem solve_for_x (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 6)
  (h : (x + 10) / (x - 4) = (x - 3) / (x + 6)) : x = -48 / 23 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2253_225371


namespace NUMINAMATH_GPT_lulu_cash_left_l2253_225360

theorem lulu_cash_left :
  ∀ (initial money spentIceCream spentTshirt deposited finalCash: ℝ),
    initial = 65 →
    spentIceCream = 5 →
    spentTshirt = 0.5 * (initial - spentIceCream) →
    deposited = (initial - spentIceCream - spentTshirt) / 5 →
    finalCash = initial - spentIceCream - spentTshirt - deposited →
    finalCash = 24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lulu_cash_left_l2253_225360


namespace NUMINAMATH_GPT_probability_of_rain_l2253_225384

variable (P_R P_B0 : ℝ)
variable (H1 : 0 ≤ P_R ∧ P_R ≤ 1)
variable (H2 : 0 ≤ P_B0 ∧ P_B0 ≤ 1)
variable (H : P_R + P_B0 - P_R * P_B0 = 0.2)

theorem probability_of_rain : 
  P_R = 1/9 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rain_l2253_225384


namespace NUMINAMATH_GPT_total_amount_leaked_l2253_225351

def amount_leaked_before_start : ℕ := 2475
def amount_leaked_while_fixing : ℕ := 3731

theorem total_amount_leaked : amount_leaked_before_start + amount_leaked_while_fixing = 6206 := by
  sorry

end NUMINAMATH_GPT_total_amount_leaked_l2253_225351


namespace NUMINAMATH_GPT_find_age_l2253_225395

theorem find_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 := 
by 
  sorry

end NUMINAMATH_GPT_find_age_l2253_225395


namespace NUMINAMATH_GPT_simplify_expression_l2253_225362

theorem simplify_expression :
  (1 / 2^2 + (2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107 / 84 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_simplify_expression_l2253_225362


namespace NUMINAMATH_GPT_weight_computation_requires_initial_weight_l2253_225398

-- Let's define the conditions
variable (initial_weight : ℕ) -- The initial weight of the pet; needs to be provided
def yearly_gain := 11  -- The pet gains 11 pounds each year
def age := 8  -- The pet is 8 years old

-- Define the goal to be proved
def current_weight_computable : Prop :=
  initial_weight ≠ 0 → initial_weight + (yearly_gain * age) ≠ 0

-- State the theorem
theorem weight_computation_requires_initial_weight : ¬ ∃ current_weight, initial_weight + (yearly_gain * age) = current_weight :=
by {
  sorry
}

end NUMINAMATH_GPT_weight_computation_requires_initial_weight_l2253_225398


namespace NUMINAMATH_GPT_calculation_l2253_225347

theorem calculation : 8 - (7.14 * (1 / 3) - (20 / 9) / (5 / 2)) + 0.1 = 6.62 :=
by
  sorry

end NUMINAMATH_GPT_calculation_l2253_225347


namespace NUMINAMATH_GPT_solve_system_l2253_225301

theorem solve_system (a b c x y z : ℝ) (h₀ : a = (a * x + c * y) / (b * z + 1))
  (h₁ : b = (b * x + y) / (b * z + 1)) 
  (h₂ : c = (a * z + c) / (b * z + 1)) 
  (h₃ : ¬ a = b * c) :
  x = 1 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_solve_system_l2253_225301


namespace NUMINAMATH_GPT_exists_list_with_all_players_l2253_225300

-- Definitions and assumptions
variable {Player : Type} 

-- Each player plays against every other player exactly once, and there are no ties.
-- Defining defeats relationship
def defeats (p1 p2 : Player) : Prop :=
  sorry -- Assume some ordering or wins relationship

-- Defining the list of defeats
def list_of_defeats (p : Player) : Set Player :=
  { q | defeats p q ∨ (∃ r, defeats p r ∧ defeats r q) }

-- Main theorem to be proven
theorem exists_list_with_all_players (players : Set Player) :
  (∀ p q : Player, p ∈ players → q ∈ players → p ≠ q → (defeats p q ∨ defeats q p)) →
  ∃ p : Player, (list_of_defeats p) = players \ {p} :=
by
  sorry

end NUMINAMATH_GPT_exists_list_with_all_players_l2253_225300


namespace NUMINAMATH_GPT_num_isosceles_triangles_is_24_l2253_225393

-- Define the structure of the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)
  (num_vertices : ℕ)

-- Define the specific hexagonal prism from the problem
def prism := HexagonalPrism.mk 2 1 12

-- Function to count the number of isosceles triangles in a given hexagonal prism
noncomputable def count_isosceles_triangles (hp : HexagonalPrism) : ℕ := sorry

-- The theorem that needs to be proved
theorem num_isosceles_triangles_is_24 :
  count_isosceles_triangles prism = 24 :=
sorry

end NUMINAMATH_GPT_num_isosceles_triangles_is_24_l2253_225393


namespace NUMINAMATH_GPT_brian_total_distance_l2253_225374

noncomputable def miles_per_gallon : ℝ := 20
noncomputable def tank_capacity : ℝ := 15
noncomputable def tank_fraction_remaining : ℝ := 3 / 7

noncomputable def total_miles_traveled (miles_per_gallon tank_capacity tank_fraction_remaining : ℝ) : ℝ :=
  let total_miles := miles_per_gallon * tank_capacity
  let fuel_used := tank_capacity * (1 - tank_fraction_remaining)
  let miles_traveled := fuel_used * miles_per_gallon
  miles_traveled

theorem brian_total_distance : 
  total_miles_traveled miles_per_gallon tank_capacity tank_fraction_remaining = 171.4 := 
by
  sorry

end NUMINAMATH_GPT_brian_total_distance_l2253_225374


namespace NUMINAMATH_GPT_jim_net_paycheck_l2253_225355

-- Let’s state the problem conditions:
def biweekly_gross_pay : ℝ := 1120
def retirement_percentage : ℝ := 0.25
def tax_deduction : ℝ := 100

-- Define the amount deduction for the retirement account
def retirement_deduction (gross : ℝ) (percentage : ℝ) : ℝ := gross * percentage

-- Define the remaining paycheck after all deductions
def net_paycheck (gross : ℝ) (retirement : ℝ) (tax : ℝ) : ℝ :=
  gross - retirement - tax

-- The theorem to prove:
theorem jim_net_paycheck :
  net_paycheck biweekly_gross_pay (retirement_deduction biweekly_gross_pay retirement_percentage) tax_deduction = 740 :=
by
  sorry

end NUMINAMATH_GPT_jim_net_paycheck_l2253_225355


namespace NUMINAMATH_GPT_dusting_days_l2253_225391

theorem dusting_days 
    (vacuuming_minutes_per_day : ℕ) 
    (vacuuming_days_per_week : ℕ)
    (dusting_minutes_per_day : ℕ)
    (total_cleaning_minutes_per_week : ℕ)
    (x : ℕ) :
    vacuuming_minutes_per_day = 30 →
    vacuuming_days_per_week = 3 →
    dusting_minutes_per_day = 20 →
    total_cleaning_minutes_per_week = 130 →
    (vacuuming_minutes_per_day * vacuuming_days_per_week + dusting_minutes_per_day * x = total_cleaning_minutes_per_week) →
    x = 2 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_dusting_days_l2253_225391


namespace NUMINAMATH_GPT_commute_time_late_l2253_225334

theorem commute_time_late (S : ℝ) (T : ℝ) (T' : ℝ) (H1 : T = 1) (H2 : T' = (4/3)) :
  T' - T = 20 / 60 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_late_l2253_225334


namespace NUMINAMATH_GPT_polygon_sides_sum_l2253_225369

theorem polygon_sides_sum
  (area_ABCDEF : ℕ) (AB BC FA DE EF : ℕ)
  (h1 : area_ABCDEF = 78)
  (h2 : AB = 10)
  (h3 : BC = 11)
  (h4 : FA = 7)
  (h5 : DE = 4)
  (h6 : EF = 8) :
  DE + EF = 12 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_sum_l2253_225369


namespace NUMINAMATH_GPT_joan_seashells_l2253_225388

variable (initialSeashells seashellsGiven remainingSeashells : ℕ)

theorem joan_seashells : initialSeashells = 79 ∧ seashellsGiven = 63 ∧ remainingSeashells = initialSeashells - seashellsGiven → remainingSeashells = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_joan_seashells_l2253_225388


namespace NUMINAMATH_GPT_max_marks_l2253_225375

theorem max_marks (M : ℕ) (h1 : M * 33 / 100 = 175 + 56) : M = 700 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l2253_225375


namespace NUMINAMATH_GPT_notebooks_difference_l2253_225325

theorem notebooks_difference :
  ∀ (Jac_left Jac_Paula Jac_Mike Ger_not Jac_init : ℕ),
  Ger_not = 8 →
  Jac_left = 10 →
  Jac_Paula = 5 →
  Jac_Mike = 6 →
  Jac_init = Jac_left + Jac_Paula + Jac_Mike →
  Jac_init - Ger_not = 13 := 
by
  intros Jac_left Jac_Paula Jac_Mike Ger_not Jac_init
  intros Ger_not_8 Jac_left_10 Jac_Paula_5 Jac_Mike_6 Jac_init_def
  sorry

end NUMINAMATH_GPT_notebooks_difference_l2253_225325


namespace NUMINAMATH_GPT_expression_evaluation_l2253_225335

theorem expression_evaluation : 
  ( ((2 + 2)^2 / 2^2) * ((3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3) * ((6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6) = 108 ) := 
by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2253_225335


namespace NUMINAMATH_GPT_cherry_tree_leaves_l2253_225339

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end NUMINAMATH_GPT_cherry_tree_leaves_l2253_225339


namespace NUMINAMATH_GPT_speed_of_man_proof_l2253_225377

noncomputable def speed_of_man (train_length : ℝ) (crossing_time : ℝ) (train_speed_kph : ℝ) : ℝ :=
  let train_speed_mps := (train_speed_kph * 1000) / 3600
  let relative_speed := train_length / crossing_time
  train_speed_mps - relative_speed

theorem speed_of_man_proof 
  (train_length : ℝ := 600) 
  (crossing_time : ℝ := 35.99712023038157) 
  (train_speed_kph : ℝ := 64) :
  speed_of_man train_length crossing_time train_speed_kph = 1.10977777777778 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_speed_of_man_proof_l2253_225377


namespace NUMINAMATH_GPT_coordinates_of_P_l2253_225348

-- Definitions of conditions
def inFourthQuadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def absEqSeven (x : ℝ) : Prop := |x| = 7
def ysquareEqNine (y : ℝ) : Prop := y^2 = 9

-- Main theorem
theorem coordinates_of_P (x y : ℝ) (hx : absEqSeven x) (hy : ysquareEqNine y) (hq : inFourthQuadrant x y) :
  (x, y) = (7, -3) :=
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l2253_225348


namespace NUMINAMATH_GPT_ellipse_hyperbola_tangent_l2253_225342

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m) → m = 2 :=
by sorry

end NUMINAMATH_GPT_ellipse_hyperbola_tangent_l2253_225342


namespace NUMINAMATH_GPT_geometric_progression_difference_l2253_225315

variable {n : ℕ}
variable {a : ℕ → ℝ} -- assuming the sequence is indexed by natural numbers
variable {a₁ : ℝ}
variable {r : ℝ} (hr : r = (1 + Real.sqrt 5) / 2)

def geometric_progression (a : ℕ → ℝ) (a₁ : ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a₁ * (r ^ n)

theorem geometric_progression_difference
  (a₁ : ℝ)
  (hr : r = (1 + Real.sqrt 5) / 2)
  (hg : geometric_progression a a₁ r) :
  ∀ n, n ≥ 2 → a n = a (n-1) - a (n-2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_difference_l2253_225315


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l2253_225323

theorem spherical_to_rectangular_coordinates :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 5 / 2
:= by
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  have hx : x = (5 * Real.sqrt 6) / 4 := sorry
  have hy : y = (5 * Real.sqrt 6) / 4 := sorry
  have hz : z = 5 / 2 := sorry
  exact ⟨hx, hy, hz⟩

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l2253_225323


namespace NUMINAMATH_GPT_profit_distribution_l2253_225372

theorem profit_distribution (x : ℕ) (hx : 2 * x = 4000) :
  let A := 2 * x
  let B := 3 * x
  let C := 5 * x
  A + B + C = 20000 := by
  sorry

end NUMINAMATH_GPT_profit_distribution_l2253_225372


namespace NUMINAMATH_GPT_phone_sales_total_amount_l2253_225373

theorem phone_sales_total_amount
  (vivienne_phones : ℕ)
  (aliyah_more_phones : ℕ)
  (price_per_phone : ℕ)
  (aliyah_phones : ℕ := vivienne_phones + aliyah_more_phones)
  (total_phones : ℕ := vivienne_phones + aliyah_phones)
  (total_amount : ℕ := total_phones * price_per_phone) :
  vivienne_phones = 40 → aliyah_more_phones = 10 → price_per_phone = 400 → total_amount = 36000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_phone_sales_total_amount_l2253_225373


namespace NUMINAMATH_GPT_math_problem_l2253_225365

theorem math_problem (a : ℝ) (h : a = 1/3) : (3 * a⁻¹ + 2 / 3 * a⁻¹) / a = 33 := by
  sorry

end NUMINAMATH_GPT_math_problem_l2253_225365


namespace NUMINAMATH_GPT_sequence_general_term_l2253_225327

theorem sequence_general_term (a : ℕ → ℕ) :
  (a 1 = 1 * 2) ∧ (a 2 = 2 * 3) ∧ (a 3 = 3 * 4) ∧ (a 4 = 4 * 5) ↔ 
    (∀ n, a n = n^2 + n) := sorry

end NUMINAMATH_GPT_sequence_general_term_l2253_225327


namespace NUMINAMATH_GPT_non_congruent_squares_6x6_grid_l2253_225381

theorem non_congruent_squares_6x6_grid : 
  let count_squares (n: ℕ) : ℕ := 
    let horizontal_or_vertical := (6 - n) * (6 - n)
    let diagonal := if n * n <= 6 * 6 then (6 - n + 1) * (6 - n + 1) else 0
    horizontal_or_vertical + diagonal
  (count_squares 1) + (count_squares 2) + (count_squares 3) + (count_squares 4) + (count_squares 5) = 141 :=
by
  sorry

end NUMINAMATH_GPT_non_congruent_squares_6x6_grid_l2253_225381


namespace NUMINAMATH_GPT_songs_owned_initially_l2253_225350

theorem songs_owned_initially (a b c : ℕ) (hc : c = a + b) (hb : b = 7) (hc_total : c = 13) :
  a = 6 :=
by
  -- Direct usage of the given conditions to conclude the proof goes here.
  sorry

end NUMINAMATH_GPT_songs_owned_initially_l2253_225350


namespace NUMINAMATH_GPT_natural_pair_prime_ratio_l2253_225394

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_pair_prime_ratio :
  ∃ (x y : ℕ), (x = 14 ∧ y = 2) ∧ is_prime (x * y^3 / (x + y)) :=
by
  use 14
  use 2
  sorry

end NUMINAMATH_GPT_natural_pair_prime_ratio_l2253_225394


namespace NUMINAMATH_GPT_number_of_balls_sold_l2253_225340

theorem number_of_balls_sold 
  (selling_price : ℤ) (loss_per_5_balls : ℤ) (cost_price_per_ball : ℤ) (n : ℤ) 
  (h1 : selling_price = 720)
  (h2 : loss_per_5_balls = 5 * cost_price_per_ball)
  (h3 : cost_price_per_ball = 48)
  (h4 : (n * cost_price_per_ball) - selling_price = loss_per_5_balls) :
  n = 20 := 
by
  sorry

end NUMINAMATH_GPT_number_of_balls_sold_l2253_225340


namespace NUMINAMATH_GPT_find_sets_l2253_225317

variable (A X Y : Set ℕ) -- Mimicking sets of natural numbers for generality.

theorem find_sets (h1 : X ∪ Y = A) (h2 : X ∩ A = Y) : X = A ∧ Y = A := by
  -- This would need a proof, which shows that: X = A and Y = A
  sorry

end NUMINAMATH_GPT_find_sets_l2253_225317


namespace NUMINAMATH_GPT_no_2021_residents_possible_l2253_225396

-- Definition: Each islander is either a knight or a liar
def is_knight_or_liar (i : ℕ) : Prop := true -- Placeholder definition for either being a knight or a liar

-- Definition: Knights always tell the truth
def knight_tells_truth (i : ℕ) : Prop := true -- Placeholder definition for knights telling the truth

-- Definition: Liars always lie
def liar_always_lies (i : ℕ) : Prop := true -- Placeholder definition for liars always lying

-- Definition: Even number of knights claimed by some islanders
def even_number_of_knights : Prop := true -- Placeholder definition for the claim of even number of knights

-- Definition: Odd number of liars claimed by remaining islanders
def odd_number_of_liars : Prop := true -- Placeholder definition for the claim of odd number of liars

-- Question and proof problem
theorem no_2021_residents_possible (K L : ℕ) (h1 : K + L = 2021) (h2 : ∀ i, is_knight_or_liar i) 
(h3 : ∀ k, knight_tells_truth k → even_number_of_knights) 
(h4 : ∀ l, liar_always_lies l → odd_number_of_liars) : 
  false := sorry

end NUMINAMATH_GPT_no_2021_residents_possible_l2253_225396


namespace NUMINAMATH_GPT_complex_imaginary_unit_theorem_l2253_225359

def complex_imaginary_unit_equality : Prop :=
  let i := Complex.I
  i * (i + 1) = -1 + i

theorem complex_imaginary_unit_theorem : complex_imaginary_unit_equality :=
by
  sorry

end NUMINAMATH_GPT_complex_imaginary_unit_theorem_l2253_225359


namespace NUMINAMATH_GPT_miles_left_to_drive_l2253_225357

theorem miles_left_to_drive 
  (total_distance : ℕ) 
  (distance_covered : ℕ) 
  (remaining_distance : ℕ) 
  (h1 : total_distance = 78) 
  (h2 : distance_covered = 32) 
  : remaining_distance = total_distance - distance_covered -> remaining_distance = 46 :=
by
  sorry

end NUMINAMATH_GPT_miles_left_to_drive_l2253_225357


namespace NUMINAMATH_GPT_total_money_is_correct_l2253_225341

-- Define conditions as constants
def numChocolateCookies : ℕ := 220
def pricePerChocolateCookie : ℕ := 1
def numVanillaCookies : ℕ := 70
def pricePerVanillaCookie : ℕ := 2

-- Total money made from selling chocolate cookies
def moneyFromChocolateCookies : ℕ := numChocolateCookies * pricePerChocolateCookie

-- Total money made from selling vanilla cookies
def moneyFromVanillaCookies : ℕ := numVanillaCookies * pricePerVanillaCookie

-- Total money made from selling all cookies
def totalMoneyMade : ℕ := moneyFromChocolateCookies + moneyFromVanillaCookies

-- The statement to prove, with the expected result
theorem total_money_is_correct : totalMoneyMade = 360 := by
  sorry

end NUMINAMATH_GPT_total_money_is_correct_l2253_225341


namespace NUMINAMATH_GPT_find_other_number_l2253_225354

theorem find_other_number (x y : ℤ) (h1 : 3 * x + 2 * y = 145) (h2 : x = 35 ∨ y = 35) : y = 20 :=
sorry

end NUMINAMATH_GPT_find_other_number_l2253_225354


namespace NUMINAMATH_GPT_stars_per_classmate_is_correct_l2253_225343

-- Define the given conditions
def total_stars : ℕ := 45
def num_classmates : ℕ := 9

-- Define the expected number of stars per classmate
def stars_per_classmate : ℕ := 5

-- Prove that the number of stars per classmate is 5 given the conditions
theorem stars_per_classmate_is_correct :
  total_stars / num_classmates = stars_per_classmate :=
sorry

end NUMINAMATH_GPT_stars_per_classmate_is_correct_l2253_225343


namespace NUMINAMATH_GPT_smallest_part_in_ratio_l2253_225383

variable (b : ℝ)

theorem smallest_part_in_ratio (h : b = -2620) : 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  smallest_part = 100 :=
by 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  sorry

end NUMINAMATH_GPT_smallest_part_in_ratio_l2253_225383


namespace NUMINAMATH_GPT_johns_raw_squat_weight_l2253_225363

variable (R : ℝ)

def sleeves_lift := R + 30
def wraps_lift := 1.25 * R
def wraps_more_than_sleeves := wraps_lift R - sleeves_lift R = 120

theorem johns_raw_squat_weight : wraps_more_than_sleeves R → R = 600 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_johns_raw_squat_weight_l2253_225363


namespace NUMINAMATH_GPT_solve_for_x_l2253_225326

theorem solve_for_x : ∃ x : ℚ, 24 - 4 = 3 * (1 + x) ∧ x = 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2253_225326


namespace NUMINAMATH_GPT_integer_solutions_l2253_225333

theorem integer_solutions (x y : ℤ) : 2 * (x + y) = x * y + 7 ↔ (x, y) = (3, -1) ∨ (x, y) = (5, 1) ∨ (x, y) = (1, 5) ∨ (x, y) = (-1, 3) := by
  sorry

end NUMINAMATH_GPT_integer_solutions_l2253_225333


namespace NUMINAMATH_GPT_quadratic_roots_product_sum_l2253_225389

theorem quadratic_roots_product_sum :
  ∀ (f g : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 4 * x + 2 = 0 → x = f ∨ x = g) → 
  (f + g = 4 / 3) → 
  (f * g = 2 / 3) → 
  (f + 2) * (g + 2) = 22 / 3 :=
by
  intro f g roots_eq sum_eq product_eq
  sorry

end NUMINAMATH_GPT_quadratic_roots_product_sum_l2253_225389


namespace NUMINAMATH_GPT_Geli_pushups_and_runs_l2253_225382

def initial_pushups : ℕ := 10
def increment_pushups : ℕ := 5
def workouts_per_week : ℕ := 3
def weeks_in_a_month : ℕ := 4
def pushups_per_mile_run : ℕ := 30

def workout_days_in_month : ℕ := workouts_per_week * weeks_in_a_month

def pushups_on_day (day : ℕ) : ℕ := initial_pushups + (day - 1) * increment_pushups

def total_pushups : ℕ := (workout_days_in_month / 2) * (initial_pushups + pushups_on_day workout_days_in_month)

def one_mile_runs (total_pushups : ℕ) : ℕ := total_pushups / pushups_per_mile_run

theorem Geli_pushups_and_runs :
  total_pushups = 450 ∧ one_mile_runs total_pushups = 15 :=
by
  -- Here, we should prove total_pushups = 450 and one_mile_runs total_pushups = 15.
  sorry

end NUMINAMATH_GPT_Geli_pushups_and_runs_l2253_225382


namespace NUMINAMATH_GPT_min_days_to_triple_loan_l2253_225331

theorem min_days_to_triple_loan (amount_borrowed : ℕ) (interest_rate : ℝ) :
  ∀ x : ℕ, x ≥ 20 ↔ amount_borrowed + (amount_borrowed * (interest_rate / 10)) * x ≥ 3 * amount_borrowed :=
sorry

end NUMINAMATH_GPT_min_days_to_triple_loan_l2253_225331


namespace NUMINAMATH_GPT_total_doughnuts_l2253_225324

-- Definitions used in the conditions
def boxes : ℕ := 4
def doughnuts_per_box : ℕ := 12

theorem total_doughnuts : boxes * doughnuts_per_box = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_doughnuts_l2253_225324


namespace NUMINAMATH_GPT_area_of_rectangle_is_108_l2253_225307

theorem area_of_rectangle_is_108 (s w l : ℕ) (h₁ : s * s = 36) (h₂ : w = s) (h₃ : l = 3 * w) : w * l = 108 :=
by
  -- This is a placeholder for a detailed proof.
  sorry

end NUMINAMATH_GPT_area_of_rectangle_is_108_l2253_225307


namespace NUMINAMATH_GPT_find_chemistry_marks_l2253_225387

theorem find_chemistry_marks :
  (let E := 96
   let M := 95
   let P := 82
   let B := 95
   let avg := 93
   let n := 5
   let Total := avg * n
   let Chemistry_marks := Total - (E + M + P + B)
   Chemistry_marks = 97) :=
by
  let E := 96
  let M := 95
  let P := 82
  let B := 95
  let avg := 93
  let n := 5
  let Total := avg * n
  have h_total : Total = 465 := by norm_num
  let Chemistry_marks := Total - (E + M + P + B)
  have h_chemistry_marks : Chemistry_marks = 97 := by norm_num
  exact h_chemistry_marks

end NUMINAMATH_GPT_find_chemistry_marks_l2253_225387


namespace NUMINAMATH_GPT_triangle_area_l2253_225367

theorem triangle_area :
  let A := (2, -3)
  let B := (2, 4)
  let C := (8, 0) 
  let base := (4 - (-3))
  let height := (8 - 2)
  let area := (1 / 2) * base * height
  area = 21 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_l2253_225367


namespace NUMINAMATH_GPT_trisha_spent_on_eggs_l2253_225306

def totalSpent (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ) : ℕ :=
  initialAmount - (meat + chicken + veggies + dogFood + amountLeft)

theorem trisha_spent_on_eggs :
  ∀ (meat chicken veggies eggs dogFood amountLeft initialAmount : ℕ),
    meat = 17 →
    chicken = 22 →
    veggies = 43 →
    dogFood = 45 →
    amountLeft = 35 →
    initialAmount = 167 →
    totalSpent meat chicken veggies eggs dogFood amountLeft initialAmount = 5 :=
by
  intros meat chicken veggies eggs dogFood amountLeft initialAmount
  sorry

end NUMINAMATH_GPT_trisha_spent_on_eggs_l2253_225306


namespace NUMINAMATH_GPT_eighth_term_sum_of_first_15_terms_l2253_225313

-- Given definitions from the conditions
def a1 : ℚ := 5
def a30 : ℚ := 100
def n8 : ℕ := 8
def n15 : ℕ := 15
def n30 : ℕ := 30

-- Formulate the arithmetic sequence properties
def common_difference : ℚ := (a30 - a1) / (n30 - 1)

def nth_term (n : ℕ) : ℚ :=
  a1 + (n - 1) * common_difference

def sum_of_first_n_terms (n : ℕ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * common_difference)

-- Statements to be proven
theorem eighth_term :
  nth_term n8 = 25 + 1/29 := by sorry

theorem sum_of_first_15_terms :
  sum_of_first_n_terms n15 = 393 + 2/29 := by sorry

end NUMINAMATH_GPT_eighth_term_sum_of_first_15_terms_l2253_225313


namespace NUMINAMATH_GPT_remainder_division_l2253_225310

variable (P D K Q R R'_q R'_r : ℕ)

theorem remainder_division (h1 : P = Q * D + R) (h2 : R = R'_q * K + R'_r) (h3 : K < D) : 
  P % (D * K) = R'_r :=
sorry

end NUMINAMATH_GPT_remainder_division_l2253_225310


namespace NUMINAMATH_GPT_range_of_p_l2253_225329

theorem range_of_p (p : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + p * x + 1 > 2 * x + p) → p > -1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_p_l2253_225329


namespace NUMINAMATH_GPT_maximum_value_of_x2y3z_l2253_225338

theorem maximum_value_of_x2y3z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 5) : 
  x + 2 * y + 3 * z ≤ Real.sqrt 70 :=
by 
  sorry

end NUMINAMATH_GPT_maximum_value_of_x2y3z_l2253_225338


namespace NUMINAMATH_GPT_episodes_per_monday_l2253_225352

theorem episodes_per_monday (M : ℕ) (h : 67 * (M + 2) = 201) : M = 1 :=
sorry

end NUMINAMATH_GPT_episodes_per_monday_l2253_225352


namespace NUMINAMATH_GPT_graph_passes_through_point_l2253_225332

theorem graph_passes_through_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  ∃ x y, (x, y) = (0, 3) ∧ (∀ f : ℝ → ℝ, (∀ y, (f y = a ^ y) → (0, f 0 + 2) = (0, 3))) :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l2253_225332


namespace NUMINAMATH_GPT_Gerald_needs_to_average_5_chores_per_month_l2253_225397

def spending_per_month := 100
def season_length := 4
def cost_per_chore := 10
def total_spending := spending_per_month * season_length
def months_not_playing := 12 - season_length
def amount_to_save_per_month := total_spending / months_not_playing
def chores_per_month := amount_to_save_per_month / cost_per_chore

theorem Gerald_needs_to_average_5_chores_per_month :
  chores_per_month = 5 := by
  sorry

end NUMINAMATH_GPT_Gerald_needs_to_average_5_chores_per_month_l2253_225397


namespace NUMINAMATH_GPT_frustum_slant_height_l2253_225320

theorem frustum_slant_height
  (ratio_area : ℝ)
  (slant_height_removed : ℝ)
  (sf_ratio : ratio_area = 1/16)
  (shr : slant_height_removed = 3) :
  ∃ (slant_height_frustum : ℝ), slant_height_frustum = 9 :=
by
  sorry

end NUMINAMATH_GPT_frustum_slant_height_l2253_225320


namespace NUMINAMATH_GPT_total_population_is_3311_l2253_225321

-- Definitions based on the problem's conditions
def fewer_than_6000_inhabitants (L : ℕ) : Prop :=
  L < 6000

def more_girls_than_boys (girls boys : ℕ) : Prop :=
  girls = (11 * boys) / 10

def more_men_than_women (men women : ℕ) : Prop :=
  men = (23 * women) / 20

def more_children_than_adults (children adults : ℕ) : Prop :=
  children = (6 * adults) / 5

-- Prove that the total population is 3311 given the described conditions
theorem total_population_is_3311 {L n men women children boys girls : ℕ}
  (hc : more_children_than_adults children (n + men))
  (hm : more_men_than_women men n)
  (hg : more_girls_than_boys girls boys)
  (hL : L = n + men + boys + girls)
  (hL_lt : fewer_than_6000_inhabitants L) :
  L = 3311 :=
sorry

end NUMINAMATH_GPT_total_population_is_3311_l2253_225321


namespace NUMINAMATH_GPT_largest_integer_n_neg_quad_expr_l2253_225311

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_n_neg_quad_expr_l2253_225311


namespace NUMINAMATH_GPT_relationship_between_x_and_y_l2253_225392

theorem relationship_between_x_and_y (x y : ℝ) (h1 : 2 * x - y > 3 * x) (h2 : x + 2 * y < 2 * y) :
  x < 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_relationship_between_x_and_y_l2253_225392


namespace NUMINAMATH_GPT_range_of_m_l2253_225319

theorem range_of_m (a m : ℝ) (h_a_neg : a < 0) (y1 y2 : ℝ)
  (hA : y1 = a * m^2 - 4 * a * m)
  (hB : y2 = 4 * a * m^2 - 8 * a * m)
  (hA_above : y1 > -3 * a)
  (hB_above : y2 > -3 * a)
  (hy1_gt_y2 : y1 > y2) :
  4 / 3 < m ∧ m < 3 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2253_225319


namespace NUMINAMATH_GPT_total_papers_delivered_l2253_225364

-- Definitions based on given conditions
def papers_saturday : ℕ := 45
def papers_sunday : ℕ := 65
def total_papers := papers_saturday + papers_sunday

-- The statement we need to prove
theorem total_papers_delivered : total_papers = 110 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_papers_delivered_l2253_225364


namespace NUMINAMATH_GPT_find_range_of_a_l2253_225330

variable (x a : ℝ)

/-- Given p: 2 * x^2 - 9 * x + a < 0 and q: the negation of p is sufficient 
condition for the negation of q,
prove to find the range of the real number a. -/
theorem find_range_of_a (hp: 2 * x^2 - 9 * x + a < 0) (hq: ¬ (2 * x^2 - 9 * x + a < 0) → ¬ q) :
  ∃ a : ℝ, sorry := sorry

end NUMINAMATH_GPT_find_range_of_a_l2253_225330


namespace NUMINAMATH_GPT_polynomial_expression_value_l2253_225337

theorem polynomial_expression_value
  (p q r s : ℂ)
  (h1 : p + q + r + s = 0)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = -1)
  (h3 : p*q*r + p*q*s + p*r*s + q*r*s = -1)
  (h4 : p*q*r*s = 2) :
  p*(q - r)^2 + q*(r - s)^2 + r*(s - p)^2 + s*(p - q)^2 = -6 :=
by sorry

end NUMINAMATH_GPT_polynomial_expression_value_l2253_225337


namespace NUMINAMATH_GPT_rationalize_denominator_l2253_225336

theorem rationalize_denominator (a b c : ℝ) (h : b ≠ 0) (h0 : 98 = c * c) (h1 : 7 = c) :
  (7 / (Real.sqrt 98) = (Real.sqrt 2) / 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2253_225336


namespace NUMINAMATH_GPT_jason_spent_on_shorts_l2253_225308

def total_spent : ℝ := 14.28
def jacket_spent : ℝ := 4.74
def shorts_spent : ℝ := total_spent - jacket_spent

theorem jason_spent_on_shorts :
  shorts_spent = 9.54 :=
by
  -- Placeholder for the proof. The statement is correct as it matches the given problem data.
  sorry

end NUMINAMATH_GPT_jason_spent_on_shorts_l2253_225308


namespace NUMINAMATH_GPT_percentage_relationship_l2253_225353

variable {x y z : ℝ}

theorem percentage_relationship (h1 : x = 1.30 * y) (h2 : y = 0.50 * z) : x = 0.65 * z :=
by
  sorry

end NUMINAMATH_GPT_percentage_relationship_l2253_225353


namespace NUMINAMATH_GPT_expected_coins_100_rounds_l2253_225303

noncomputable def expectedCoinsAfterGame (rounds : ℕ) (initialCoins : ℕ) : ℝ :=
  initialCoins * (101 / 100) ^ rounds

theorem expected_coins_100_rounds :
  expectedCoinsAfterGame 100 1 = (101 / 100 : ℝ) ^ 100 :=
by
  sorry

end NUMINAMATH_GPT_expected_coins_100_rounds_l2253_225303


namespace NUMINAMATH_GPT_find_m_when_z_is_real_l2253_225305

theorem find_m_when_z_is_real (m : ℝ) (h : (m ^ 2 + 2 * m - 15 = 0)) : m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_when_z_is_real_l2253_225305


namespace NUMINAMATH_GPT_coefficients_identity_l2253_225361

def coefficients_of_quadratic (a b c : ℤ) (x : ℤ) : Prop :=
  a * x^2 + b * x + c = 0

theorem coefficients_identity : ∀ x : ℤ,
  coefficients_of_quadratic 3 (-4) 1 x :=
by
  sorry

end NUMINAMATH_GPT_coefficients_identity_l2253_225361


namespace NUMINAMATH_GPT_prove_m_equals_9_given_split_l2253_225316

theorem prove_m_equals_9_given_split (m : ℕ) (h : 1 < m) (h1 : m^3 = 73) : m = 9 :=
sorry

end NUMINAMATH_GPT_prove_m_equals_9_given_split_l2253_225316


namespace NUMINAMATH_GPT_distinct_integer_values_of_a_l2253_225386

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end NUMINAMATH_GPT_distinct_integer_values_of_a_l2253_225386


namespace NUMINAMATH_GPT_average_of_last_three_numbers_l2253_225378

theorem average_of_last_three_numbers (A B C D E F : ℕ) 
  (h1 : (A + B + C + D + E + F) / 6 = 30)
  (h2 : (A + B + C + D) / 4 = 25)
  (h3 : D = 25) :
  (D + E + F) / 3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_of_last_three_numbers_l2253_225378


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2253_225349

variable (a m : ℝ)

def prop_p (a m : ℝ) : Prop := (m - a) * (m - 3 * a) ≤ 0
def prop_q (m : ℝ) : Prop := (m + 2) * (m + 1) < 0

theorem problem_part1 (h₁ : a = -1) (h₂ : prop_p a m ∨ prop_q m) : -3 ≤ m ∧ m ≤ -1 :=
sorry

theorem problem_part2 (h₁ : ∀ m, prop_p a m → ¬prop_q m) :
  -1 / 3 ≤ a ∧ a < 0 ∨ a ≤ -2 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2253_225349


namespace NUMINAMATH_GPT_number_of_BMWs_sold_l2253_225356

theorem number_of_BMWs_sold (total_cars : ℕ) (ford_percentage nissan_percentage volkswagen_percentage : ℝ) 
    (h1 : total_cars = 300)
    (h2 : ford_percentage = 0.2)
    (h3 : nissan_percentage = 0.25)
    (h4 : volkswagen_percentage = 0.1) :
    ∃ (bmw_percentage : ℝ) (bmw_cars : ℕ), bmw_percentage = 0.45 ∧ bmw_cars = 135 :=
by 
    sorry

end NUMINAMATH_GPT_number_of_BMWs_sold_l2253_225356


namespace NUMINAMATH_GPT_clothes_add_percentage_l2253_225376

theorem clothes_add_percentage (W : ℝ) (C : ℝ) (h1 : W > 0) 
  (h2 : C = 0.0174 * W) : 
  ((C / (0.87 * W)) * 100) = 2 :=
by
  sorry

end NUMINAMATH_GPT_clothes_add_percentage_l2253_225376


namespace NUMINAMATH_GPT_safe_trip_possible_l2253_225345

-- Define the time intervals and eruption cycles
def total_round_trip_time := 16
def trail_time := 8
def crater1_cycle := 18
def crater2_cycle := 10
def crater1_erupt := 1
def crater1_quiet := 17
def crater2_erupt := 1
def crater2_quiet := 9

-- Ivan wants to safely reach the summit and return
theorem safe_trip_possible : ∃ t, 
  -- t is a valid start time where both craters are quiet
  ((t % crater1_cycle) ≥ crater1_erupt ∧ (t % crater2_cycle) ≥ crater2_erupt) ∧
  -- t + total_round_trip_time is also safe for both craters
  (((t + total_round_trip_time) % crater1_cycle) ≥ crater1_erupt ∧ ((t + total_round_trip_time) % crater2_cycle) ≥ crater2_erupt) :=
sorry

end NUMINAMATH_GPT_safe_trip_possible_l2253_225345


namespace NUMINAMATH_GPT_tangent_lines_parallel_l2253_225399

-- Definitions and conditions
def curve (x : ℝ) : ℝ := x^3 + x - 2
def line (x : ℝ) : ℝ := 4 * x - 1
def tangent_line_eq (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Proof statement
theorem tangent_lines_parallel (tangent_line : ℝ → ℝ) :
  (∃ x : ℝ, tangent_line_eq 4 (-1) 0 x (curve x)) ∧ 
  (∃ x : ℝ, tangent_line_eq 4 (-1) (-4) x (curve x)) :=
sorry

end NUMINAMATH_GPT_tangent_lines_parallel_l2253_225399


namespace NUMINAMATH_GPT_mass_percentage_of_C_in_CCl4_l2253_225304

theorem mass_percentage_of_C_in_CCl4 :
  let mass_carbon : ℝ := 12.01
  let mass_chlorine : ℝ := 35.45
  let molar_mass_CCl4 : ℝ := mass_carbon + 4 * mass_chlorine
  let mass_percentage_C : ℝ := (mass_carbon / molar_mass_CCl4) * 100
  mass_percentage_C = 7.81 := 
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_C_in_CCl4_l2253_225304


namespace NUMINAMATH_GPT_children_vehicle_wheels_l2253_225312

theorem children_vehicle_wheels:
  ∀ (x : ℕ),
    (6 * 2) + (15 * x) = 57 →
    x = 3 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_children_vehicle_wheels_l2253_225312


namespace NUMINAMATH_GPT_fruit_prob_l2253_225318

variable (O A B S : ℕ) 

-- Define the conditions
variables (H1 : O + A + B + S = 32)
variables (H2 : O - 5 = 3)
variables (H3 : A - 3 = 7)
variables (H4 : S - 2 = 4)
variables (H5 : 3 + 7 + 4 + B = 20)

-- Define the proof problem
theorem fruit_prob :
  (O = 8) ∧ (A = 10) ∧ (B = 6) ∧ (S = 6) → (O + S) / (O + A + B + S) = 7 / 16 := 
by
  sorry

end NUMINAMATH_GPT_fruit_prob_l2253_225318


namespace NUMINAMATH_GPT_cube_volume_is_27_l2253_225385

noncomputable def original_cube_edge (a : ℝ) : ℝ := a

noncomputable def original_cube_volume (a : ℝ) : ℝ := a^3

noncomputable def new_rectangular_solid_volume (a : ℝ) : ℝ := (a-2) * a * (a+2)

theorem cube_volume_is_27 (a : ℝ) (h : original_cube_volume a - new_rectangular_solid_volume a = 14) : original_cube_volume a = 27 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_is_27_l2253_225385


namespace NUMINAMATH_GPT_unique_divisor_of_2_pow_n_minus_1_l2253_225309

theorem unique_divisor_of_2_pow_n_minus_1 : ∀ (n : ℕ), n ≥ 1 → n ∣ (2^n - 1) → n = 1 := 
by
  intro n h1 h2
  sorry

end NUMINAMATH_GPT_unique_divisor_of_2_pow_n_minus_1_l2253_225309
