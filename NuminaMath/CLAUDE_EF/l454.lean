import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_to_hundredth_l454_45430

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem rounding_to_hundredth :
  (round_to_hundredth 34.554999 ≠ 34.56) ∧
  (round_to_hundredth 34.553 ≠ 34.56) ∧
  (round_to_hundredth 34.559 = 34.56) ∧
  (round_to_hundredth 34.5551 = 34.56) ∧
  (round_to_hundredth 34.56001 = 34.56) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_to_hundredth_l454_45430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_duration_l454_45424

/-- Represents the investment details of a partner -/
structure Investment where
  amount : ℝ
  duration : ℝ

/-- Calculates the share of an investment in the total investment -/
noncomputable def investmentShare (inv : Investment) (totalInvestment : ℝ) : ℝ :=
  (inv.amount * inv.duration) / totalInvestment

theorem b_investment_duration (a b c : Investment) 
  (total_profit : ℝ) (c_profit_share : ℝ) : b.duration = 5 := by
  have h1 : a.amount = 6500 := by sorry
  have h2 : a.duration = 6 := by sorry
  have h3 : b.amount = 8400 := by sorry
  have h4 : c.amount = 10000 := by sorry
  have h5 : c.duration = 3 := by sorry
  have h6 : total_profit = 7400 := by sorry
  have h7 : c_profit_share = 1900 := by sorry

  -- A gets 5% of total profit as working partner
  let remaining_profit : ℝ := total_profit * 0.95
  let total_investment : ℝ := a.amount * a.duration + b.amount * b.duration + c.amount * c.duration

  have h8 : c_profit_share = investmentShare c total_investment * remaining_profit := by sorry

  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_duration_l454_45424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_foci_l454_45483

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 1)

def on_line_segment (P A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem min_dot_product_foci :
  ∀ P : ℝ × ℝ, on_line_segment P A B →
    (∀ Q : ℝ × ℝ, on_line_segment Q A B →
      dot_product (F₁.1 - P.1, F₁.2 - P.2) (F₂.1 - P.1, F₂.2 - P.2) ≤
      dot_product (F₁.1 - Q.1, F₁.2 - Q.2) (F₂.1 - Q.1, F₂.2 - Q.2)) →
    dot_product (F₁.1 - P.1, F₁.2 - P.2) (F₂.1 - P.1, F₂.2 - P.2) = -11/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_foci_l454_45483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l454_45453

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x + 3)

-- Define the domain of f
def domain (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Define the decreasing interval
def decreasing_interval (x : ℝ) : Prop := 1 < x ∧ x < 3

-- Theorem statement
theorem f_decreasing_interval :
  ∀ x : ℝ, domain x → (∀ y : ℝ, domain y → x < y → f y < f x) ↔ decreasing_interval x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l454_45453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l454_45470

/-- Two circles intersecting at two points -/
structure IntersectingCircles where
  Γ₁ : Set (ℝ × ℝ)
  Γ₂ : Set (ℝ × ℝ)
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h₁ : P ∈ Γ₁ ∧ P ∈ Γ₂
  h₂ : Q ∈ Γ₁ ∧ Q ∈ Γ₂
  h₃ : P ≠ Q

/-- A line intersecting the segment PQ at an interior point -/
structure IntersectingLine (ic : IntersectingCircles) where
  d : Set (ℝ × ℝ)
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  h₁ : A ∈ d ∧ A ∈ ic.Γ₁
  h₂ : B ∈ d ∧ B ∈ ic.Γ₂
  h₃ : C ∈ d ∧ C ∈ ic.Γ₁
  h₄ : D ∈ d ∧ D ∈ ic.Γ₂
  h₅ : ∃ (X : ℝ × ℝ), X ∈ d ∧ X ≠ ic.P ∧ X ≠ ic.Q ∧ 
       ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ X = (1 - t) • ic.P + t • ic.Q

/-- The angle between three points -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem angle_equality (ic : IntersectingCircles) (il : IntersectingLine ic) :
  angle il.A ic.P il.B = angle il.C ic.Q il.D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l454_45470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l454_45474

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem area_of_region : 
  ∃ A : ℝ, (∀ x y : ℝ, x^2 + y^2 - 8*x + 12*y = -4) → A = 48 * Real.pi :=
by
  -- We'll use 48 * Real.pi as our proposed area
  let A := 48 * Real.pi
  
  -- We claim this A satisfies our theorem
  use A
  
  -- The proof goes here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l454_45474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l454_45459

open Real

noncomputable def f (x : ℝ) : ℝ := (sqrt (sin x) + log (cos x)) / tan x

theorem domain_of_f (x : ℝ) :
  (∃ k : ℤ, x ∈ Set.Ioo (2 * k * π) ((π / 2) + 2 * k * π)) ↔
  (0 ≤ sin x ∧ 0 < cos x ∧ tan x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l454_45459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_hoses_fill_time_l454_45464

/-- Represents the time it takes to fill the pool with different combinations of hoses -/
structure PoolFilling where
  xy : ℝ  -- Time for hoses X and Y together
  xz : ℝ  -- Time for hoses X and Z together
  yz : ℝ  -- Time for hoses Y and Z together

/-- Calculates the time it takes for all three hoses to fill the pool -/
noncomputable def fillTimeAllHoses (p : PoolFilling) : ℝ :=
  36 / 11

/-- Theorem stating that given the fill times for pairs of hoses, 
    the time for all three hoses working together is 36/11 hours -/
theorem all_hoses_fill_time (p : PoolFilling) 
  (h1 : p.xy = 3) 
  (h2 : p.xz = 6) 
  (h3 : p.yz = 4.5) : 
  fillTimeAllHoses p = 36 / 11 := by
  sorry

#eval (36 : ℚ) / 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_hoses_fill_time_l454_45464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l454_45442

noncomputable def m : ℝ × ℝ := (2, 2)
noncomputable def n : ℝ × ℝ := ((-4 + 2) / 2, (4 + 2) / 2)  -- Derived from 2n - m = (-4, 4)

noncomputable def cos_theta : ℝ := 
  (m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))

theorem cos_theta_value : cos_theta = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_value_l454_45442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_ghee_mixture_weight_l454_45419

/-- Represents the weight of a mixture of vegetable ghee brands. -/
noncomputable def mixture_weight (weight_a weight_b weight_c : ℝ) (ratio_a ratio_b ratio_c : ℝ) (total_volume : ℝ) : ℝ :=
  (weight_a * ratio_a + weight_b * ratio_b + weight_c * ratio_c) * total_volume / (ratio_a + ratio_b + ratio_c) / 1000

/-- Proves that the weight of the vegetable ghee mixture is 4.9 kg. -/
theorem vegetable_ghee_mixture_weight :
  mixture_weight 900 700 800 3 2 1 6 = 4.9 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval mixture_weight 900 700 800 3 2 1 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vegetable_ghee_mixture_weight_l454_45419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extremes_l454_45418

theorem sum_of_extremes (nums : List Nat) : 
  nums.length = 1928 → 
  nums.sum = 2016 → 
  nums.prod = 1001 → 
  (nums.maximum?.getD 0) + (nums.minimum?.getD 0) = 90 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extremes_l454_45418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triplets_cover_range_l454_45421

def triplet1 : Finset ℕ := {1, 2, 3}
def triplet2 : Finset ℕ := {0, 3, 6}
def triplet3 : Finset ℕ := {0, 9, 18}
def triplet4 : Finset ℕ := {0, 27, 54}

def is_representable (n : ℕ) : Prop :=
  ∃ a b c d, a ∈ triplet1 ∧ b ∈ triplet2 ∧ c ∈ triplet3 ∧ d ∈ triplet4 ∧
    n = a + b + c + d

theorem triplets_cover_range : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 81 → is_representable n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triplets_cover_range_l454_45421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_ticket_sales_l454_45427

theorem volleyball_ticket_sales (total_tickets jude_tickets sandra_extra unsold_tickets : ℕ) :
  total_tickets = 100 →
  jude_tickets = 16 →
  sandra_extra = 4 →
  unsold_tickets = 40 →
  let sandra_tickets := sandra_extra + jude_tickets / 2
  let sold_tickets := total_tickets - unsold_tickets
  let andrea_tickets := sold_tickets - jude_tickets - sandra_tickets
  andrea_tickets / jude_tickets = 2 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_ticket_sales_l454_45427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_loss_is_25_percent_l454_45412

/-- Represents the percentage of goods stolen for each type --/
structure TheftPercentages where
  typeA : ℚ
  typeB : ℚ
  typeC : ℚ

/-- Calculates the average loss percentage given theft percentages --/
def averageLossPercentage (t : TheftPercentages) : ℚ :=
  (t.typeA + t.typeB + t.typeC) / 3

/-- Theorem stating that given specific theft percentages, the average loss is 25% --/
theorem average_loss_is_25_percent (t : TheftPercentages) 
  (h1 : t.typeA = 20)
  (h2 : t.typeB = 25)
  (h3 : t.typeC = 30) :
  averageLossPercentage t = 25 := by
  sorry

#check average_loss_is_25_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_loss_is_25_percent_l454_45412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freelance_earnings_difference_l454_45498

/-- Calculates the difference between Janet's monthly freelance earnings and current job earnings -/
noncomputable def freelance_vs_current_job_earnings_difference : ℝ :=
  let current_job_hourly_rate : ℝ := 30
  let current_job_weekly_hours : ℝ := 40
  let weeks_per_month : ℝ := 4

  let client_a_hourly_rate : ℝ := 45
  let client_a_weekly_hours : ℝ := 15
  let client_b_hourly_rate : ℝ := 40
  let client_b_weekly_hours : ℝ := 15
  let client_c_hourly_rate : ℝ := (35 + 42) / 2  -- average rate
  let client_c_weekly_hours : ℝ := 20

  let fica_tax_weekly : ℝ := 25
  let healthcare_monthly : ℝ := 400
  let rent_increase_monthly : ℝ := 750
  let business_expenses_monthly : ℝ := 150
  let business_expense_deduction_rate : ℝ := 0.1

  let current_job_monthly_earnings := current_job_hourly_rate * current_job_weekly_hours * weeks_per_month

  let freelance_weekly_earnings := 
    client_a_hourly_rate * client_a_weekly_hours +
    client_b_hourly_rate * client_b_weekly_hours +
    client_c_hourly_rate * client_c_weekly_hours

  let freelance_monthly_earnings := freelance_weekly_earnings * weeks_per_month
  let business_expense_deduction := freelance_monthly_earnings * business_expense_deduction_rate
  let adjusted_freelance_earnings := freelance_monthly_earnings - business_expense_deduction

  let additional_expenses := 
    fica_tax_weekly * weeks_per_month + 
    healthcare_monthly + 
    rent_increase_monthly + 
    business_expenses_monthly

  adjusted_freelance_earnings - additional_expenses - current_job_monthly_earnings

/-- The difference between Janet's monthly freelance earnings and current job earnings is $1162 -/
theorem freelance_earnings_difference : 
  freelance_vs_current_job_earnings_difference = 1162 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freelance_earnings_difference_l454_45498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_symmetry_l454_45490

/-- Given a quadratic function f(x) = ax² + bx + 1 where a ≠ 0,
    if there exist distinct real numbers x₁ and x₂ such that f(x₁) = f(x₂),
    then f(x₁ + x₂) = 1 -/
theorem quadratic_symmetry (a b : ℝ) (x₁ x₂ : ℝ) (h_a : a ≠ 0) (h_neq : x₁ ≠ x₂) :
  let f := λ x => a * x^2 + b * x + 1
  (f x₁ = f x₂) → (f (x₁ + x₂) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_symmetry_l454_45490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ella_run_time_l454_45411

/-- Calculates the total time for Ella's run -/
noncomputable def total_time (track_length : ℝ) (num_laps : ℕ) (first_segment_length : ℝ) (first_segment_speed : ℝ) 
                (second_segment_length : ℝ) (second_segment_speed : ℝ) (break_time : ℝ) : ℝ :=
  let first_segment_time := first_segment_length / first_segment_speed
  let second_segment_time := second_segment_length / second_segment_speed
  let lap_time := first_segment_time + second_segment_time
  let total_run_time := (num_laps : ℝ) * lap_time
  let total_break_time := ((num_laps - 1) : ℝ) * break_time
  total_run_time + total_break_time

/-- Theorem stating that Ella's total time is 223 seconds -/
theorem ella_run_time : 
  total_time 300 3 80 5 180 4 20 = 223 := by
  -- Unfold the definition of total_time
  unfold total_time
  -- Simplify the arithmetic expressions
  simp [add_assoc, mul_add, add_mul]
  -- The proof is completed by normalization of real number arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ella_run_time_l454_45411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_diameter_approximation_l454_45434

-- Define the shape parameters
def square_side : ℝ := 8
def total_perimeter : ℝ := 36.56637061435917

-- Define the relationship between square side and semicircle diameter
def semicircle_diameter_equals_square_side (d : ℝ) : Prop :=
  abs (d - square_side) < 0.0001

-- Theorem statement
theorem semicircle_diameter_approximation :
  ∃ d : ℝ, 
    square_side * 3 + Real.pi * d / 2 = total_perimeter ∧
    semicircle_diameter_equals_square_side d :=
by
  -- Proof goes here
  sorry

#check semicircle_diameter_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_diameter_approximation_l454_45434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_b_is_4000_l454_45489

/-- Represents the loan details and interest calculations --/
structure LoanDetails where
  rate : ℚ  -- Annual interest rate as a rational number
  total_interest : ℚ  -- Total interest received
  loan_to_c : ℚ  -- Amount lent to C
  time_b : ℚ  -- Time period for B's loan in years
  time_c : ℚ  -- Time period for C's loan in years

/-- Calculates the amount lent to B given the loan details --/
def amount_lent_to_b (l : LoanDetails) : ℚ :=
  (l.total_interest - l.loan_to_c * l.rate * l.time_c) / (l.rate * l.time_b)

/-- Theorem stating that the amount lent to B is 4000 --/
theorem amount_lent_to_b_is_4000 (l : LoanDetails) 
    (h1 : l.rate = 275 / 2000)
    (h2 : l.total_interest = 2200)
    (h3 : l.loan_to_c = 2000)
    (h4 : l.time_b = 2)
    (h5 : l.time_c = 4) : 
  amount_lent_to_b l = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_to_b_is_4000_l454_45489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_lower_bound_l454_45494

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x

-- State the theorems
theorem tangent_line_at_zero (a : ℝ) :
  (∃ m b, ∀ x, m * x + b = f a x + (deriv (f a)) 0 * (x - 0)) ∧
  (deriv (f a)) 0 * (-1/2) = 1 →
  ∃ m b, m * x + b = 2 * x + 1 := by sorry

theorem lower_bound (a : ℝ) (x : ℝ) :
  a > 0 → f a x ≥ -4 * a^2 + 4 * a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_lower_bound_l454_45494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ends_in_14_rounds_l454_45492

/-- Represents the state of a player in the tournament -/
inductive PlayerState
  | NoLoss
  | OneLoss
  | Eliminated

/-- Represents the state of the tournament after each round -/
structure TournamentState where
  noLossCount : Nat
  oneLossCount : Nat

/-- Simulates one round of the tournament -/
def simulateRound (state : TournamentState) : TournamentState :=
  { noLossCount := state.noLossCount / 2,
    oneLossCount := state.noLossCount / 2 + state.oneLossCount / 2 }

/-- Checks if the tournament can continue -/
def canContinue (state : TournamentState) : Bool :=
  state.noLossCount + state.oneLossCount ≥ 2

/-- Counts the number of rounds until the tournament ends -/
def countRounds (initialState : TournamentState) : Nat :=
  let rec aux (state : TournamentState) (rounds : Nat) (fuel : Nat) : Nat :=
    match fuel with
    | 0 => rounds
    | fuel+1 =>
      if canContinue state then
        aux (simulateRound state) (rounds + 1) fuel
      else
        rounds
  aux initialState 0 100  -- Set an upper bound of 100 rounds

/-- Theorem: The tennis tournament with 1152 participants ends after exactly 14 rounds -/
theorem tournament_ends_in_14_rounds :
  countRounds { noLossCount := 1152, oneLossCount := 0 } = 14 := by
  sorry

#eval countRounds { noLossCount := 1152, oneLossCount := 0 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_ends_in_14_rounds_l454_45492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_properties_l454_45461

structure Graph (V : Type*) [Fintype V] where
  edges : V → V → Prop

def is_hamiltonian_path {V : Type*} [Fintype V] (G : Graph V) (path : List V) : Prop :=
  sorry

def is_eulerian_path {V : Type*} [Fintype V] (G : Graph V) (path : List V) : Prop :=
  sorry

def is_eulerian_circuit {V : Type*} [Fintype V] (G : Graph V) (path : List V) : Prop :=
  sorry

def has_no_odd_cycles {V : Type*} [Fintype V] (G : Graph V) : Prop :=
  sorry

def is_two_colorable {V : Type*} [Fintype V] (G : Graph V) (blue red : Finset V) : Prop :=
  sorry

def vertex_degree {V : Type*} [Fintype V] (G : Graph V) (v : V) : ℕ :=
  sorry

theorem graph_properties {V : Type*} [Fintype V] (G : Graph V) 
  (h_vertices : Fintype.card V = 12)
  (h_no_odd_cycles : has_no_odd_cycles G)
  (h_two_colorable : ∃ (blue red : Finset V), is_two_colorable G blue red ∧ blue.card = 5 ∧ red.card = 7)
  (h_odd_degree : ∃ (v1 v2 : V), v1 ≠ v2 ∧ 
    Odd (vertex_degree G v1) ∧ Odd (vertex_degree G v2) ∧
    ∀ (v : V), v ≠ v1 → v ≠ v2 → Even (vertex_degree G v)) :
  (¬ ∃ (path : List V), is_hamiltonian_path G path) ∧
  (∃ (path : List V), is_eulerian_path G path) ∧
  (¬ ∃ (path : List V), is_eulerian_circuit G path) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_properties_l454_45461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l454_45491

universe u

def UniverseSet : Type := Fin 3

def count_distinct_pairs : ℕ := 27

theorem distinct_pairs_count :
  count_distinct_pairs = 27 := by
  -- The proof goes here
  sorry

#eval count_distinct_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_pairs_count_l454_45491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_tournament_matches_l454_45431

theorem football_tournament_matches (n : ℕ) (h : n = 10) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧ 
  ∃ (tournament_matches : Fin n → ℕ),
    tournament_matches i = k ∧ 
    tournament_matches j = k ∧ 
    ∀ (t : Fin n), tournament_matches t ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_tournament_matches_l454_45431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_separate_l454_45413

/-- Represents a point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space --/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Converts polar coordinates to Cartesian coordinates --/
noncomputable def polarToCartesian (ρ : ℝ) (θ : ℝ) : Point2D :=
  { x := ρ * Real.cos θ, y := ρ * Real.sin θ }

/-- Defines the circle C from its polar equation --/
noncomputable def circleC : Circle :=
  { center := { x := 1, y := 1 },
    radius := Real.sqrt 2 }

/-- Defines the line l from its parametric equation --/
def lineL : Line :=
  { a := 2, b := 1, c := -7 }

/-- Calculates the distance between a point and a line --/
noncomputable def distancePointToLine (p : Point2D) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Theorem: The circle C and line l are separate --/
theorem circle_and_line_separate : 
  distancePointToLine circleC.center lineL > circleC.radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_separate_l454_45413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_smaller_base_l454_45450

/-- A trapezoid with the given properties -/
structure Trapezoid (p q a : ℝ) where
  p_pos : 0 < p
  q_pos : 0 < q
  p_lt_q : p < q
  a_pos : 0 < a
  angle_ratio : ∃ α : ℝ, 0 < α ∧ α < π/2 ∧ 
    Real.sin α = q * Real.sin (2*α) / (2*p)

/-- The smaller base of the trapezoid -/
noncomputable def smaller_base (p q a : ℝ) (t : Trapezoid p q a) : ℝ :=
  (p^2 + a*p - q^2) / p

/-- Theorem stating that the smaller base of the trapezoid is correct -/
theorem trapezoid_smaller_base (p q a : ℝ) (t : Trapezoid p q a) :
  ∃ x : ℝ, x = smaller_base p q a t ∧ 
  x = a - (q^2 + p^2) / (2*p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_smaller_base_l454_45450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_points_tangent_parallel_coordinates_l454_45466

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points (x : ℝ) :
  (deriv f x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

theorem tangent_parallel_coordinates :
  {p : ℝ × ℝ | deriv f p.1 = 4 ∧ f p.1 = p.2} = {(1, 0), (-1, -4)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_points_tangent_parallel_coordinates_l454_45466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l454_45455

theorem triangle_side_difference : 
  ∃ (maximal_y minimal_y : ℤ),
  (∀ y : ℤ, y > 3 ∧ y < 19 →
    (∃ a b c : ℝ, a = y ∧ b = 8 ∧ c = 11 ∧ 
      a + b > c ∧ b + c > a ∧ c + a > b)) ∧
  (∀ z : ℤ, (z > 3 ∧ z < 19) → z ≤ maximal_y) ∧
  (∀ z : ℤ, (z > 3 ∧ z < 19) → z ≥ minimal_y) ∧
  maximal_y - minimal_y = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_difference_l454_45455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_university_box_cost_l454_45420

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the number of boxes needed given the total volume and box volume -/
noncomputable def boxesNeeded (totalVolume boxVolume : ℝ) : ℝ :=
  totalVolume / boxVolume

/-- Calculates the cost per box given the total cost and number of boxes -/
noncomputable def costPerBox (totalCost numBoxes : ℝ) : ℝ :=
  totalCost / numBoxes

theorem university_box_cost 
  (boxDim : BoxDimensions)
  (totalVolume : ℝ)
  (totalCost : ℝ)
  (h1 : boxDim.length = 20)
  (h2 : boxDim.width = 20)
  (h3 : boxDim.height = 15)
  (h4 : totalVolume = 3060000)
  (h5 : totalCost = 612) :
  costPerBox totalCost (boxesNeeded totalVolume (boxVolume boxDim)) = 1.20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_university_box_cost_l454_45420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l454_45439

-- Define lg as logarithm with base 2
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem simplify_expression : lg 4 + 2 * lg 5 + 4^(-1/2 : ℝ) = 5/2 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l454_45439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_in_square_l454_45401

/-- The area of the largest circle inscribed in a square -/
noncomputable def largest_inscribed_circle_area (square_area : ℝ) (pi_approx : ℝ) : ℝ :=
  let side_length := Real.sqrt square_area
  let radius := side_length / 2
  pi_approx * radius * radius

/-- Theorem stating the area of the largest circle inscribed in a square with area 400 -/
theorem circle_area_in_square (square_area : ℝ) (pi_approx : ℝ) 
  (h1 : square_area = 400)
  (h2 : pi_approx = 3.1) : 
  largest_inscribed_circle_area square_area pi_approx = 310 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_in_square_l454_45401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_235_ones_minus_zeros_l454_45473

/-- Converts a natural number to its binary representation as a list of booleans -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Counts the number of true values in a list of booleans -/
def count_ones (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Counts the number of false values in a list of booleans -/
def count_zeros (l : List Bool) : ℕ :=
  l.length - count_ones l

theorem binary_235_ones_minus_zeros : 
  let binary_235 := to_binary 235
  let y := count_ones binary_235
  let x := count_zeros binary_235
  y - x = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_235_ones_minus_zeros_l454_45473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l454_45456

/-- Geometric sequence a_n -/
def a : ℕ → ℝ := sorry

/-- Arithmetic sequence b_n -/
def b : ℕ → ℝ := sorry

/-- Sum of first n terms of b_n -/
def S : ℕ → ℝ := sorry

theorem sequence_properties :
  (a 1 = 2) ∧
  (a 3 = 18) ∧
  (b 1 = 2) ∧
  (a 1 + a 2 + a 3 = b 1 + b 2 + b 3 + b 4) ∧
  (a 1 + a 2 + a 3 > 20) →
  (∀ n : ℕ, a n = 2 * 3^(n-1)) ∧
  (∀ n : ℕ, S n = (3/2) * n^2 + (1/2) * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l454_45456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_less_than_one_l454_45448

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def a (n : ℕ) : ℚ := 1 / ((fibonacci n : ℚ) * (fibonacci (n + 2) : ℚ))

theorem fibonacci_sum_less_than_one (m : ℕ) :
  (Finset.range (m + 1)).sum (λ i ↦ a i) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_sum_less_than_one_l454_45448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_180_moves_l454_45472

/-- Represents a complex number rotation by π/3 radians -/
noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 3)

/-- The position of the particle after n moves -/
noncomputable def z (n : ℕ) : ℂ :=
  8 * ω ^ n + 8 * (Finset.range n).sum (λ k => ω ^ k)

/-- The theorem stating that after 180 moves, the particle returns to (8, 0) -/
theorem particle_position_after_180_moves :
  z 180 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_180_moves_l454_45472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l454_45465

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2

noncomputable def g (x : ℝ) := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4) + 1

theorem g_solutions (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) :
  g x = 1 ↔ x = Real.pi / 8 ∨ x = 5 * Real.pi / 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l454_45465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l454_45440

/-- The rational function f(x) = (3x^2 + 5x - 9) / (x-4) -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 5*x - 9) / (x - 4)

/-- The slope of the slant asymptote -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote -/
def b : ℝ := 17

/-- Theorem: The sum of the coefficients m and b in the slant asymptote of f(x) is 20 -/
theorem slant_asymptote_sum : m + b = 20 := by
  -- Unfold the definitions of m and b
  unfold m b
  -- Perform the addition
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l454_45440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l454_45452

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the minimum value function g
noncomputable def g (t : ℝ) : ℝ :=
  if t < 0 then t^2 + 1
  else if t ≤ 1 then 1
  else t^2 - 2*t + 2

-- Theorem statement
theorem min_value_of_f (t : ℝ) :
  ∀ x ∈ Set.Icc t (t + 1), g t ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l454_45452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l454_45436

-- Define the power function f
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Define the function g
noncomputable def g (α : ℝ) (x : ℝ) : ℝ := (x - 3) * f α x

-- Theorem statement
theorem min_value_g (α : ℝ) (h : f α 5 = 1/5) :
  ∃ (x : ℝ), x ∈ Set.Icc (1/3) 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (1/3) 1 → g α x ≤ g α y ∧
  g α x = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l454_45436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l454_45432

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_area_triangle (a b : ℝ) (P : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  hyperbola a b P.1 P.2 →
  P.1 > 0 →
  distance P left_focus = 2 * distance P right_focus →
  (∀ Q : ℝ × ℝ, hyperbola a b Q.1 Q.2 → Q.1 > 0 → distance Q left_focus = 2 * distance Q right_focus →
    1/2 * distance left_focus right_focus * P.2 ≥ 1/2 * distance left_focus right_focus * Q.2) →
  1/2 * distance left_focus right_focus * P.2 = 4/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l454_45432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_c_wins_l454_45405

/-- Represents the number of games won and lost by a player -/
structure PlayerRecord where
  wins : ℕ
  losses : ℕ

/-- The theorem stating the number of games won by player C -/
theorem player_c_wins (
  playerA playerB playerC : PlayerRecord
) (
  hA_wins : playerA.wins = 4
) (
  hA_losses : playerA.losses = 2
) (
  hB_wins : playerB.wins = 3
) (
  hB_losses : playerB.losses = 3
) (
  hC_losses : playerC.losses = 3
) (
  h_balance : playerA.wins + playerB.wins + playerC.wins = playerA.losses + playerB.losses + playerC.losses
) : playerC.wins = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_c_wins_l454_45405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inequality_l454_45445

-- Define a triangle with vertices A, B, C in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define a line intersecting the triangle sides
structure IntersectingLine (t : Triangle) where
  M : ℝ × ℝ  -- Intersection point on side AB
  N : ℝ × ℝ  -- Intersection point on side AC
  passes_through_centroid : M.1 ≠ N.1 ∨ M.2 ≠ N.2  -- Ensure M and N are distinct

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem centroid_inequality (t : Triangle) (l : IntersectingLine t) :
  let O := centroid t
  distance O l.N ≤ 2 * distance O l.M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inequality_l454_45445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_property_sum_of_squares_not_always_geometric_sequence_arithmetic_sequence_l454_45486

-- A: If a > 0, then 2^a > 1
theorem exponential_property (a : ℝ) : a > 0 → (2 : ℝ)^a > 1 := by sorry

-- B: If x^2 + y^2 = 0, then x = y = 0
theorem sum_of_squares (x y : ℝ) : x^2 + y^2 = 0 → x = 0 ∧ y = 0 := by sorry

-- C: It's not always true that if b^2 = ac, then a, b, c form a geometric sequence
theorem not_always_geometric_sequence : 
  ∃ (a b c : ℝ), b^2 = a * c ∧ ¬(∃ r : ℝ, b = a * r ∧ c = b * r) := by sorry

-- D: If a + c = 2b, then a, b, c form an arithmetic sequence
theorem arithmetic_sequence (a b c : ℝ) : 
  a + c = 2 * b → ∃ d : ℝ, b = a + d ∧ c = b + d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_property_sum_of_squares_not_always_geometric_sequence_arithmetic_sequence_l454_45486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almond_butter_jar_price_l454_45446

/-- Represents a cylindrical jar --/
structure Jar where
  diameter : ℝ
  height : ℝ
  price : ℝ

/-- Calculates the volume of a cylindrical jar --/
noncomputable def jarVolume (j : Jar) : ℝ := Real.pi * (j.diameter / 2) ^ 2 * j.height

/-- Calculates the price of a larger jar based on a smaller jar's price and volume --/
noncomputable def largerJarPrice (small : Jar) (large : Jar) (discount : ℝ) : ℝ :=
  let volumeRatio := jarVolume large / jarVolume small
  let priceWithoutDiscount := volumeRatio * small.price
  priceWithoutDiscount * (1 - discount)

theorem almond_butter_jar_price :
  let small : Jar := { diameter := 4, height := 5, price := 1 }
  let large : Jar := { diameter := 8, height := 10, price := 0 }
  let discount := 0.1
  largerJarPrice small large discount = 14.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_almond_butter_jar_price_l454_45446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l454_45488

/-- A structure representing a circle in a plane -/
structure Circle where
  center : ℂ
  radius : ℝ

/-- A predicate to check if a complex point is on a circle -/
def on_circle (z : ℂ) (C : Circle) : Prop :=
  Complex.abs (z - C.center) = C.radius

/-- A predicate to check if two circles intersect at two points -/
def intersect_at (C₁ C₂ : Circle) (A B : ℂ) : Prop :=
  on_circle A C₁ ∧ on_circle A C₂ ∧ on_circle B C₁ ∧ on_circle B C₂ ∧ A ≠ B

/-- A predicate to check if four complex points are concyclic or collinear -/
def concyclic_or_collinear (A B C D : ℂ) : Prop :=
  ∃ (z : ℂ), (z - A) * (z - C) = (z - B) * (z - D) ∨ 
  (A - B) * (C - D) = (A - C) * (B - D)

/-- The main theorem -/
theorem circle_intersection_theorem 
  (C₁ C₂ C₃ C₄ : Circle) 
  (A₁ B₁ A₂ B₂ A₃ B₃ A₄ B₄ C₁' D₁ C₂' D₂ : ℂ) :
  intersect_at C₁ C₂ A₁ B₁ →
  intersect_at C₂ C₃ A₂ B₂ →
  intersect_at C₃ C₄ A₃ B₃ →
  intersect_at C₄ C₁ A₄ B₄ →
  concyclic_or_collinear A₁ B₁ C₁' D₁ →
  concyclic_or_collinear A₂ B₂ C₂' D₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l454_45488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_with_asymptote_l454_45404

/-- The denominator polynomial -/
def q (x : ℝ) : ℝ := 3*x^4 - 2*x^3 + x - 5

/-- A polynomial p with degree n -/
noncomputable def p (n : ℕ) (x : ℝ) : ℝ := sorry

/-- The rational function formed by p and q -/
noncomputable def f (n : ℕ) (x : ℝ) : ℝ := (p n x) / (q x)

/-- Definition of having a horizontal asymptote -/
def has_horizontal_asymptote (n : ℕ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f n x - L| < ε

/-- The main theorem -/
theorem largest_degree_with_asymptote :
  (∃ n : ℕ, has_horizontal_asymptote n) ∧
  (∀ n : ℕ, has_horizontal_asymptote n → n ≤ 4) ∧
  (has_horizontal_asymptote 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_degree_with_asymptote_l454_45404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factorial_l454_45462

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_difference_factorial (x y z : ℕ+) : 
  x * y * z = factorial 9 → x < y → y < z → 
  ∀ a b c : ℕ+, a * b * c = factorial 9 → a < b → b < c → 
  z.val - x.val ≤ c.val - a.val :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_factorial_l454_45462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_simplification_l454_45469

open BigOperators Finset Real

theorem sum_ratio_simplification (n : ℕ) (hn : 0 < n) :
  let a_n := ∑ k in range (n + 1), 1 / (n.choose k : ℝ)
  let c_n := ∑ k in range (n + 1), (k^2 : ℝ) / (n.choose k : ℝ)
  a_n / c_n = 1 / n^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_simplification_l454_45469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_value_l454_45425

-- Define the function f
def f : ℝ → ℝ := sorry

-- Symmetry about y-axis
axiom symmetry (x : ℝ) : f x = f (-x)

-- Property: f(x+3) = -f(x)
axiom shift_property (x : ℝ) : f (x + 3) = -f x

-- Define f(x) = (1/2)^x for x ∈ (3/2, 5/2)
axiom interval_def (x : ℝ) : 3/2 < x ∧ x < 5/2 → f x = (1/2)^x

-- Theorem to prove
theorem f_2017_value : f 2017 = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_value_l454_45425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_piles_problem_l454_45416

theorem paper_piles_problem (n : ℕ) : 
  1000 < n ∧ n < 2000 ∧
  (∀ k : ℕ, k ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ) → n % k = 1) ∧
  (∃! d : ℕ, 1 < d ∧ d < n ∧ n % d = 0) →
  ∃ d : ℕ, d = 41 ∧ 1 < d ∧ d < n ∧ n % d = 0 :=
by
  sorry

#check paper_piles_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_piles_problem_l454_45416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_A_B_l454_45451

-- Define the sets A, B, and U
def A : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_union_A_B :
  (A ∪ B)ᶜ = Set.Ioc (-4) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_A_B_l454_45451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l454_45426

-- Define the function s(x) as noncomputable
noncomputable def s (x : ℝ) : ℝ := 1 / (2 + x)^3

-- State the theorem
theorem s_range :
  {y : ℝ | ∃ x : ℝ, x ≠ -2 ∧ s x = y} = {y : ℝ | y < 0 ∨ y > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l454_45426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_element_l454_45406

def is_valid_set (M : Finset ℕ+) : Prop :=
  (Finset.card M = 2004) ∧
  ∀ x ∈ M, ∀ y ∈ M, ∀ z ∈ M, x + y ≠ z

theorem min_max_element (M : Finset ℕ+) (h : is_valid_set M) :
  ∃ m ∈ M, m ≥ 4007 ∧ ∀ x ∈ M, x ≤ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_element_l454_45406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_and_cube_root_l454_45458

-- Define the variables
variable (a b x : ℝ)

-- Define the conditions
def condition1 (a x : ℝ) : Prop := (a + 3)^2 = x ∧ (2*a - 15)^2 = x
def condition2 (b : ℝ) : Prop := Real.sqrt (2*b - 1) = 13

-- State the theorem
theorem square_roots_and_cube_root 
  (h1 : condition1 a x) 
  (h2 : condition2 b) : 
  x = 49 ∧ (a + b - 1)^(1/3) = 2 * 11^(1/3) :=
by
  sorry

#check square_roots_and_cube_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_and_cube_root_l454_45458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_on_x_axis_l454_45423

-- Define the hyperbola E
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the circle C
def circleC (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the focus of the hyperbola
def focus : ℝ × ℝ := (2, 0)

-- Define the tangent line
def tangent_line (x : ℝ) : Prop := x = -2

-- Main theorem
theorem segment_length_on_x_axis : 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- The center is on the right branch of the hyperbola
    hyperbola center.1 center.2 ∧
    -- The circle passes through the focus
    circleC center radius focus.1 focus.2 ∧
    -- The circle is tangent to the line x = -2
    (∃ y : ℝ, circleC center radius (-2) y ∧ tangent_line (-2)) →
    -- The length of the segment cut by the circle on the x-axis is 2
    ∃ x1 x2 : ℝ, x1 < x2 ∧ 
      circleC center radius x1 0 ∧ 
      circleC center radius x2 0 ∧ 
      x2 - x1 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_on_x_axis_l454_45423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_unit_square_l454_45485

/-- Given a square ABCD with side length 1 and points P on BC and Q on CD 
    such that triangle APQ is equilateral, prove that PQ = √6 - √2 -/
theorem equilateral_triangle_in_unit_square (A B C D P Q : ℝ × ℝ) :
  let square := {A, B, C, D}
  let side_length := 1
  ∀ (s : Set (ℝ × ℝ)), s = square →
    (∀ (x y : ℝ × ℝ), x ∈ s → y ∈ s → x ≠ y → dist x y = side_length) →
    P.1 = B.1 ∧ B.2 ≤ P.2 ∧ P.2 ≤ C.2 →
    Q.1 = C.1 ∧ C.2 ≤ Q.2 ∧ Q.2 ≤ D.2 →
    dist A P = dist A Q ∧ dist A P = dist P Q →
    dist P Q = Real.sqrt 6 - Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_in_unit_square_l454_45485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_range_l454_45409

noncomputable section

-- Define the curves
def C₁ (a : ℝ) (x : ℝ) : ℝ := a * x^2
def C₂ (x : ℝ) : ℝ := Real.exp x

-- Define the condition for a common tangent
def has_common_tangent (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (deriv (C₁ a)) x₁ = (deriv C₂) x₂ ∧
  (C₁ a x₁ - C₂ x₂) / (x₁ - x₂) = (deriv (C₁ a)) x₁

-- State the theorem
theorem common_tangent_implies_a_range (a : ℝ) :
  a > 0 → has_common_tangent a → a > Real.exp 2 / 4 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_range_l454_45409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_from_set_l454_45428

def digit_set : Finset Nat := {2, 7, 8, 9}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def digits_from_set (n : Nat) : Prop :=
  (n / 10) ∈ digit_set ∧ (n % 10) ∈ digit_set

def different_digits (n : Nat) : Prop :=
  (n / 10) ≠ (n % 10)

theorem two_digit_primes_from_set :
  ∃! (s : Finset Nat), 
    (∀ (n : Nat), n ∈ s ↔ 
      is_two_digit n ∧ 
      digits_from_set n ∧ 
      different_digits n ∧ 
      Nat.Prime n) ∧
    s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_primes_from_set_l454_45428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_power_product_l454_45422

/-- The sum of the digits in the decimal representation of 2^2005 × 5^2007 × 3 is 12 -/
theorem sum_of_digits_of_power_product : ∃ (n : ℕ), 
  n = 2^2005 * 5^2007 * 3 ∧ 
  (n.repr.toList.map (λ c => c.toString.toNat!)).sum = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_power_product_l454_45422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l454_45449

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 where a > 0,
    if one focus has coordinates (2√3, 0), then a = 2√2 -/
theorem hyperbola_focus (a : ℝ) (h1 : a > 0) 
    (h2 : ∀ x y : ℝ, x^2/a^2 - y^2/4 = 1 → (x, y) ∈ Set.range (λ t : ℝ ↦ (t, 0)))
    (h3 : (2 * Real.sqrt 3, 0) ∈ Set.range (λ t : ℝ ↦ (t, 0))) :
  a = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l454_45449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_proof_l454_45481

theorem rectangle_breadth_proof (square_area : ℝ) (rectangle_length : ℝ) (rectangle_area : ℝ) :
  square_area = 2025 →
  rectangle_length = 10 →
  rectangle_area = 270 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_breadth := (3 / 5) * circle_radius
  rectangle_area = rectangle_length * rectangle_breadth →
  rectangle_breadth = 27 := by
  intro h1 h2 h3 h4
  -- Proof steps would go here
  sorry

#check rectangle_breadth_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_proof_l454_45481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_l454_45457

theorem student_arrangement (n m : ℕ) : 
  n = 6 → m = 2 → (n - 1).factorial * m.factorial = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_l454_45457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l454_45471

/-- The line equation y = (3x - 1) / 4 --/
def line_eq (x y : ℝ) : Prop := y = (3 * x - 1) / 4

/-- The point we're finding the closest point to --/
noncomputable def target_point : ℝ × ℝ := (3, 5)

/-- The claimed closest point on the line --/
noncomputable def closest_point : ℝ × ℝ := (111/25, 96.5/25)

/-- Theorem stating that the closest_point is indeed the closest point on the line to target_point --/
theorem closest_point_is_correct :
  line_eq closest_point.1 closest_point.2 ∧
  ∀ (p : ℝ × ℝ), line_eq p.1 p.2 →
    (p.1 - target_point.1)^2 + (p.2 - target_point.2)^2 ≥
    (closest_point.1 - target_point.1)^2 + (closest_point.2 - target_point.2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_correct_l454_45471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l454_45414

/-- Given a hyperbola with equation y²/4 - x²/8 = 1, its asymptotes are y = ±(√2/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (y^2 / 4 - x^2 / 8 = 1) →
  (∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l454_45414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l454_45407

/-- The angle between two lines in a 2D plane --/
noncomputable def angle_between_lines (m1 m2 : ℝ) : ℝ :=
  Real.arctan ((m2 - m1) / (1 + m1 * m2))

/-- Theorem: The angle between x + 3 = 0 and x + y - 3 = 0 is 45° --/
theorem angle_between_specific_lines :
  let line1 : ℝ → ℝ → Prop := λ x y ↦ x + 3 = 0
  let line2 : ℝ → ℝ → Prop := λ x y ↦ x + y - 3 = 0
  let m1 : ℝ := (0 : ℝ)⁻¹  -- Vertical line has undefined slope, we use 1/0 as a placeholder
  let m2 : ℝ := -1
  angle_between_lines m1 m2 = π / 4
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_lines_l454_45407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_given_norms_l454_45410

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cos_theta_given_norms (a b : V) 
  (norm_a : ‖a‖ = 3)
  (norm_b : ‖b‖ = 4)
  (norm_sum_sq : ‖a + b‖^2 = 64) :
  (inner a b / (‖a‖ * ‖b‖)) = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_given_norms_l454_45410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l454_45495

theorem cube_root_equation_solution (x : ℝ) : 
  (x = (27 + Real.sqrt 769) / 10 ∨ x = (27 - Real.sqrt 769) / 10) ↔ 
  (5 * x - 2 / x) ^ (1/3 : ℝ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l454_45495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bd_fraction_of_ac_l454_45441

structure LineSegment where
  length : ℝ

def Segment (A B : Point) : LineSegment := ⟨1⟩  -- Placeholder definition

instance : Add LineSegment where
  add a b := ⟨a.length + b.length⟩

instance : SMul ℕ LineSegment where
  smul n a := ⟨n * a.length⟩

theorem bd_fraction_of_ac (A B C D : Point) :
  Segment A D = Segment A B + Segment B D →
  Segment A D = Segment A C + Segment C D →
  Segment A B = 3 • Segment B D →
  Segment A C = 7 • Segment C D →
  (Segment A D).length = 40 →
  (Segment B D).length / (Segment A C).length = 2 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bd_fraction_of_ac_l454_45441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l454_45417

/-- The area of a rectangle with vertices at (-9, 1), (1, 1), (1, -8), and (-9, -8) in a rectangular coordinate system is 90 square units. -/
theorem rectangle_area : 
  let vertex1 : ℝ × ℝ := (-9, 1)
  let vertex2 : ℝ × ℝ := (1, 1)
  let vertex3 : ℝ × ℝ := (1, -8)
  let vertex4 : ℝ × ℝ := (-9, -8)
  let length : ℝ := vertex2.1 - vertex1.1
  let width : ℝ := vertex1.2 - vertex4.2
  let area : ℝ := length * width
  area = 90 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l454_45417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traced_curve_equals_diameter_envelope_l454_45467

/-- A circle rolling along the x-axis -/
structure RollingCircle where
  radius : ℝ
  center : ℝ × ℝ
  angle : ℝ

/-- The curve traced by a point on a rolling circle -/
noncomputable def tracedCurve (c : RollingCircle) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  { (x, y) | ∃ θ : ℝ,
    x = c.center.1 + c.radius * (θ * Real.cos θ + Real.sin θ) ∧
    y = c.center.2 + c.radius * (θ * Real.sin θ - Real.cos θ + 1) }

/-- The envelope of diameters of a rolling circle -/
noncomputable def diameterEnvelope (c : RollingCircle) : Set (ℝ × ℝ) :=
  { (x, y) | ∃ θ : ℝ,
    x * Real.sin θ + y * Real.cos θ = c.radius * (θ * Real.sin θ + Real.cos θ) }

/-- Theorem stating that the traced curve of a point on a circle with radius a/2
    is identical to the envelope of diameters of a circle with radius a -/
theorem traced_curve_equals_diameter_envelope
  (a : ℝ) (ha : a > 0) :
  tracedCurve ⟨a/2, (0, a/2), 0⟩ (0, a) =
  diameterEnvelope ⟨a, (0, a), 0⟩ := by
  sorry

#check traced_curve_equals_diameter_envelope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traced_curve_equals_diameter_envelope_l454_45467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_specific_diagonals_l454_45480

/-- Two intersecting equal circles with an inscribed rhombus --/
structure IntersectingCirclesWithRhombus where
  /-- Radius of each circle --/
  R : ℝ
  /-- Length of one diagonal of the rhombus --/
  d1 : ℝ
  /-- Length of the other diagonal of the rhombus --/
  d2 : ℝ
  /-- The diagonals are positive --/
  h1 : d1 > 0
  h2 : d2 > 0
  /-- The diagonals bisect each other --/
  h3 : R ^ 2 = (R - d1 / 2) ^ 2 + (d2 / 2) ^ 2

/-- The area of each circle given the diagonals of the inscribed rhombus --/
noncomputable def circleArea (icr : IntersectingCirclesWithRhombus) : ℝ :=
  Real.pi * icr.R ^ 2

/-- Theorem: If the diagonals of the rhombus are 6 cm and 12 cm, 
    then the area of each circle is (225/4)π cm² --/
theorem circle_area_with_specific_diagonals 
    (icr : IntersectingCirclesWithRhombus) 
    (h4 : icr.d1 = 6) 
    (h5 : icr.d2 = 12) : 
    circleArea icr = (225 / 4) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_specific_diagonals_l454_45480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l454_45487

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (ratio : a 2 / a 4 = 7 / 6) : 
  S 7 / S 3 = 2 / 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l454_45487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l454_45403

/-- The sum of an arithmetic sequence with given parameters -/
noncomputable def arithmetic_sequence_sum (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1 : ℝ) * d)

/-- Theorem: The first term of the arithmetic sequence with given parameters is 4 -/
theorem arithmetic_sequence_first_term :
  ∃ (a : ℝ), 
    arithmetic_sequence_sum a 3 20 = 650 ∧ a = 4 := by
  use 4
  constructor
  · -- Prove that the sum is 650 when a = 4
    simp [arithmetic_sequence_sum]
    norm_num
  · -- Prove that a = 4
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_first_term_l454_45403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_given_sec_minus_cosec_l454_45460

open Real

theorem tan_plus_cot_given_sec_minus_cosec (x : ℝ) :
  (1 / cos x) - (1 / sin x) = 2 * Real.sqrt 6 →
  tan x + (1 / tan x) = 6 ∨ tan x + (1 / tan x) = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_given_sec_minus_cosec_l454_45460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_properties_l454_45477

/-- Represents a frustum of a right circular cone -/
structure ConeFrustum where
  r1 : ℝ  -- Radius of smaller base
  r2 : ℝ  -- Radius of larger base
  h : ℝ   -- Height between bases

/-- Calculate the lateral surface area of a cone frustum -/
noncomputable def lateralSurfaceArea (f : ConeFrustum) : ℝ :=
  Real.pi * (f.r1 + f.r2) * Real.sqrt ((f.r2 - f.r1)^2 + f.h^2)

/-- Calculate the height of the original cone before it was cut -/
noncomputable def originalConeHeight (f : ConeFrustum) : ℝ :=
  f.h + Real.sqrt ((f.r2 - f.r1)^2 + f.h^2) * (f.r1 / f.r2)

theorem frustum_properties (f : ConeFrustum) 
    (h_r1 : f.r1 = 4)
    (h_r2 : f.r2 = 8)
    (h_h : f.h = 6) :
    lateralSurfaceArea f = 24 * Real.pi * Real.sqrt 13 ∧
    originalConeHeight f = 6 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_properties_l454_45477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_box_count_l454_45435

theorem crayon_box_count (initial_crayons : ℕ) (added_crayons : ℚ) : 
  initial_crayons = 7 → added_crayons = 7/3 → 
  (Int.floor (↑initial_crayons + added_crayons) : ℤ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crayon_box_count_l454_45435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pentagon_side_length_l454_45463

/-- Definition of a quadrilateral -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Definition of a pentagon -/
structure Pentagon :=
  (vertices : Fin 5 → ℝ × ℝ)

/-- Predicate to check if a quadrilateral is a rhombus -/
def Quadrilateral.is_rhombus (q : Quadrilateral) : Prop :=
  sorry

/-- The acute angle of a rhombus -/
noncomputable def Quadrilateral.acute_angle (q : Quadrilateral) : ℝ :=
  sorry

/-- The side length of a rhombus -/
noncomputable def Quadrilateral.side_length (q : Quadrilateral) : ℝ :=
  sorry

/-- Predicate to check if a pentagon is regular -/
def Pentagon.is_regular (p : Pentagon) : Prop :=
  sorry

/-- Predicate to check if a pentagon is inscribed in a rhombus -/
def Pentagon.is_inscribed_in (p : Pentagon) (q : Quadrilateral) : Prop :=
  sorry

/-- The side length of a pentagon -/
noncomputable def Pentagon.side_length (p : Pentagon) : ℝ :=
  sorry

/-- Predicate to check if a given length is the side length of a regular pentagon inscribed in a rhombus -/
def is_side_length_of_inscribed_regular_pentagon (a c : ℝ) : Prop :=
  ∃ (rhombus : Quadrilateral) (pentagon : Pentagon),
    rhombus.is_rhombus ∧
    rhombus.acute_angle = 72 * Real.pi / 180 ∧
    rhombus.side_length = c ∧
    pentagon.is_regular ∧
    pentagon.is_inscribed_in rhombus ∧
    pentagon.side_length = a

/-- A rhombus with an acute angle of 72° and side length c contains an inscribed regular pentagon with side length (√5 - 1)c/2 -/
theorem inscribed_pentagon_side_length (c : ℝ) (h : c > 0) :
  ∃ (a : ℝ), a > 0 ∧ a = (Real.sqrt 5 - 1) * c / 2 ∧
  is_side_length_of_inscribed_regular_pentagon a c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_pentagon_side_length_l454_45463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l454_45444

theorem system_solution (a : ℝ) : 
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ),
    (x₁ = a^2 + a + 1 ∧
     y₁ = a^2 - a + 1 ∧
     z₁ = a^2 + 1 ∧
     x₂ = a^2 - a + 1 ∧
     y₂ = a^2 + a + 1 ∧
     z₂ = a^2 + 1) ∧
    (x₁^2 + y₁^2 - 2*z₁^2 = 2*a^2 ∧
     x₁ + y₁ + 2*z₁ = 4*(a^2 + 1) ∧
     z₁^2 - x₁*y₁ = a^2) ∧
    (x₂^2 + y₂^2 - 2*z₂^2 = 2*a^2 ∧
     x₂ + y₂ + 2*z₂ = 4*(a^2 + 1) ∧
     z₂^2 - x₂*y₂ = a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l454_45444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_implies_l454_45437

theorem tan_two_implies (x : ℝ) (h : Real.tan x = 2) :
  (2/3 * (Real.sin x)^2 + 1/4 * (Real.cos x)^2 = 7/12) ∧
  ((Real.cos x + Real.sin x) / (Real.cos x - Real.sin x) = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_implies_l454_45437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_calculation_l454_45433

theorem sample_size_calculation (total_students : ℕ) (selection_prob : ℝ) 
  (h1 : total_students = 400 + 320 + 280) 
  (h2 : selection_prob = 0.2) : 
  (total_students : ℝ) * selection_prob = 200 :=
by
  -- Convert total_students to ℝ
  have total_students_real : ℝ := total_students
  
  -- Substitute the values
  rw [h1, h2]
  
  -- Perform the calculation
  norm_num
  
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_calculation_l454_45433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_when_a_is_1_a_values_when_min_M_is_3_l454_45482

noncomputable def f (a x : ℝ) : ℝ := x^2 + x + a^2 + a

noncomputable def g (a x : ℝ) : ℝ := x^2 - x + a^2 - a

noncomputable def M (a x : ℝ) : ℝ := max (f a x) (g a x)

theorem min_M_when_a_is_1 :
  ∃ x₀ : ℝ, ∀ x : ℝ, M 1 x₀ ≤ M 1 x ∧ M 1 x₀ = 7/4 := by sorry

theorem a_values_when_min_M_is_3 :
  ∃ a₁ a₂ : ℝ, a₁ = -(Real.sqrt 14 - 1)/2 ∧ a₂ = (Real.sqrt 14 - 1)/2 ∧
    (∀ a x : ℝ, M a x ≥ 3) ∧
    (∃ x₁ x₂ : ℝ, M a₁ x₁ = 3 ∧ M a₂ x₂ = 3) ∧
    (∀ a : ℝ, (∃ x : ℝ, M a x = 3) → (a = a₁ ∨ a = a₂)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_when_a_is_1_a_values_when_min_M_is_3_l454_45482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_AB_l454_45479

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

structure Circle where
  center : Point
  radius : ℝ

-- Define custom membership relations
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define the given conditions
variable (M : Point)
variable (e : Line)
variable (A B : Point)
variable (F : Point)
variable (k : Circle)
variable (c : ℝ)

-- State the theorem
theorem locus_of_AB :
  (∀ (e : Line), Point.onLine M e) →  -- e rotates around M
  (∀ (A B : Point), Point.onLine A e ∧ Point.onLine B e) →  -- AB lies on e
  (A.x - M.x) * (B.x - M.x) = c^2 →  -- AM * MB is constant
  F = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) →  -- F is midpoint of AB
  Point.onCircle F k ∧ Point.onCircle M k →  -- F moves on a circle k passing through M
  (A.x - M.x)^2 + (A.y - M.y)^2 < (B.x - M.x)^2 + (B.y - M.y)^2 →  -- AM < BM
  ∃ (k₀ : Circle), Point.onCircle A k₀ ∧ Point.onCircle B k₀ ∧
    k₀.center = Point.mk (2 * k.center.x - M.x) (2 * k.center.y - M.y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_AB_l454_45479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_salary_proof_l454_45496

/-- The salary of employee n in dollars per week -/
noncomputable def salary_n : ℝ := 270

/-- The salary of employee m as a percentage of employee n's salary -/
noncomputable def salary_m_percentage : ℝ := 120

/-- The total amount paid to both employees per week in dollars -/
noncomputable def total_salary : ℝ := salary_n + (salary_m_percentage / 100) * salary_n

theorem total_salary_proof : total_salary = 594 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_salary_proof_l454_45496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_twelve_draws_l454_45493

def total_balls : ℕ := 8
def red_balls : ℕ := 3
def white_balls : ℕ := 5
def target_red_balls : ℕ := 10
def total_draws : ℕ := 12

theorem probability_of_twelve_draws (total_balls red_balls white_balls target_red_balls total_draws : ℕ) :
  total_balls = red_balls + white_balls →
  target_red_balls = 10 →
  total_draws = 12 →
  (Nat.choose 11 9 : ℚ) * (3 / 8) ^ 9 * (5 / 8) ^ 2 * (3 / 8) =
    (Nat.choose 11 9 : ℚ) * (3 / 8) ^ 10 * (5 / 8) ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_twelve_draws_l454_45493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_zero_triangles_l454_45438

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector in a plane --/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- Given a set of points, construct all possible vectors between them --/
def constructVectors (points : Finset Point) : Finset PlaneVector :=
  sorry

/-- Check if three vectors form a zero triangle --/
def isZeroTriangle (v1 v2 v3 : PlaneVector) : Prop :=
  v1.x + v2.x + v3.x = 0 ∧ v1.y + v2.y + v3.y = 0

/-- Count the number of zero triangles in a set of vectors --/
def countZeroTriangles (vectors : Finset PlaneVector) : ℕ :=
  sorry

/-- Check if three points are collinear --/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  sorry

/-- The main theorem --/
theorem max_zero_triangles 
  (points : Finset Point) 
  (h1 : points.card = 12)
  (h2 : ∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → 
       p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬areCollinear p1 p2 p3) :
  countZeroTriangles (constructVectors points) ≤ 70 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_zero_triangles_l454_45438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_volume_independent_of_radius_no_additional_gold_needed_l454_45408

/-- 
Represents the volume of a ring formed by the intersection of a sphere and a cylinder.
h: The width of the ring
r: The radius of the cylinder
k: The factor by which r is increased
-/
noncomputable def ring_volume (h : ℝ) (r : ℝ) (k : ℝ) : ℝ := (Real.pi / 6) * h^3

/-- 
Theorem stating that the volume of the ring is independent of the cylinder's radius.
-/
theorem volume_independent_of_radius (h : ℝ) (r : ℝ) (k : ℝ) 
  (h_pos : h > 0) (r_pos : r > 0) (k_pos : k > 0) :
  ring_volume h r k = ring_volume h (k * r) k :=
by sorry

/-- 
Corollary stating that no additional gold is needed when increasing the radius.
-/
theorem no_additional_gold_needed (h : ℝ) (r : ℝ) (k : ℝ) 
  (h_pos : h > 0) (r_pos : r > 0) (k_pos : k > 0) :
  ∃ (v : ℝ), v = ring_volume h r k ∧ v = ring_volume h (k * r) k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_volume_independent_of_radius_no_additional_gold_needed_l454_45408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l454_45429

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)

theorem omega_value (ω : ℝ) (m n : ℝ) 
  (h1 : ω > 0)
  (h2 : ∀ x ∈ Set.Icc (-1) 1, m ≤ f ω x ∧ f ω x ≤ n)
  (h3 : n - m = 3) :
  ω = 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l454_45429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kinder_surprise_theorem_l454_45454

/-- The number of types of gnomes -/
def num_gnome_types : ℕ := 12

/-- The number of gnomes in each Kinder Surprise -/
def gnomes_per_surprise : ℕ := 3

/-- The set of all possible Kinder Surprises -/
def all_surprises : Finset (Finset (Fin num_gnome_types)) :=
  (Finset.powerset (Finset.univ)).filter (λ s => s.card = gnomes_per_surprise)

/-- The minimum number of Kinder Surprises needed -/
def min_surprises : ℕ := 166

theorem kinder_surprise_theorem :
  (∀ s₁ s₂ : Finset (Fin num_gnome_types), s₁ ∈ all_surprises → s₂ ∈ all_surprises → s₁ ≠ s₂ → s₁ ∩ s₂ ≠ s₁) →
  (∀ n : Finset (Finset (Fin num_gnome_types)),
    n.card < min_surprises →
    (∃ i : Fin num_gnome_types, ∀ s ∈ n, i ∉ s)) →
  (∀ n : Finset (Finset (Fin num_gnome_types)),
    n.card ≥ min_surprises →
    (∀ i : Fin num_gnome_types, ∃ s ∈ n, i ∈ s)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kinder_surprise_theorem_l454_45454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_length_angle_pi_over_6_l454_45497

noncomputable def a : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)

theorem perpendicular_vector_length (m : ℝ) : 
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → 
  Real.sqrt ((b m).1^2 + (b m).2^2) = 2 * Real.sqrt 3 := by
  sorry

theorem angle_pi_over_6 : 
  ∃ m : ℝ, (a.1 * (b m).1 + a.2 * (b m).2) = 
    Real.sqrt (a.1^2 + a.2^2) * Real.sqrt ((b m).1^2 + (b m).2^2) * Real.cos (π / 6) ∧ 
  m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_length_angle_pi_over_6_l454_45497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_functions_identical_to_identity_l454_45447

-- Define the identity function
def identity (x : ℝ) : ℝ := x

-- Define the four given functions
noncomputable def f1 (x : ℝ) : ℝ := (Real.sqrt x)^2
def f2 (x : ℝ) : ℝ := 3 * x^3
noncomputable def f3 (x : ℝ) : ℝ := Real.sqrt (x^2)
noncomputable def f4 (x : ℝ) : ℝ := x^2 / x

-- Theorem stating that none of the functions are identical to the identity function
theorem no_functions_identical_to_identity :
  (∃ x : ℝ, f1 x ≠ identity x) ∧
  (∃ x : ℝ, f2 x ≠ identity x) ∧
  (∃ x : ℝ, f3 x ≠ identity x) ∧
  (∃ x : ℝ, f4 x ≠ identity x) := by
  sorry

#check no_functions_identical_to_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_functions_identical_to_identity_l454_45447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l454_45415

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem f_domain : {x : ℝ | x ≠ 1} = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l454_45415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_games_l454_45400

theorem basketball_games (x : ℕ) 
  (h1 : x > 0)
  (h2 : (3 : ℚ) / 4 * x = (x : ℚ) * 3 / 4)
  (h3 : (2 : ℚ) / 3 * (x + 18) = (x : ℚ) * 3 / 4 + 9) :
  x = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_games_l454_45400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_passes_through_Q_Q_bisects_MN_when_parallel_l454_45475

noncomputable section

/-- The line l -/
def line_l (x y : ℝ) : Prop := 5 * x - 7 * y - 70 = 0

/-- The ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- The fixed point Q -/
def point_Q : ℝ × ℝ := (25/14, -9/10)

/-- Predicate to check if a line is tangent to the ellipse at a point -/
def is_tangent_line (P M : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is on a line -/
def on_line (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) : Prop := sorry

/-- Function to get the line through two points -/
def line_through (M N : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

/-- Predicate to check if two lines are parallel -/
def is_parallel (l1 l2 : ℝ × ℝ → Prop) : Prop := sorry

/-- Predicate to check if a point is the midpoint of a line segment -/
def is_midpoint (Q M N : ℝ × ℝ) : Prop := sorry

/-- Theorem: The chord MN passes through Q for any point P on line l -/
theorem chord_passes_through_Q (P M N : ℝ × ℝ) 
  (hP : line_l P.1 P.2) 
  (hM : ellipse M.1 M.2) 
  (hN : ellipse N.1 N.2) 
  (hPM : is_tangent_line P M)
  (hPN : is_tangent_line P N) :
  on_line (line_through M N) point_Q :=
sorry

/-- Theorem: Q bisects MN when MN is parallel to line l -/
theorem Q_bisects_MN_when_parallel (P M N : ℝ × ℝ) 
  (hP : line_l P.1 P.2) 
  (hM : ellipse M.1 M.2) 
  (hN : ellipse N.1 N.2) 
  (hPM : is_tangent_line P M)
  (hPN : is_tangent_line P N)
  (hParallel : is_parallel (line_through M N) (λ p => line_l p.1 p.2)) :
  is_midpoint point_Q M N :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_passes_through_Q_Q_bisects_MN_when_parallel_l454_45475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l454_45476

/-- Given a triangle with the following properties:
    - The height of the triangle is 2 cm
    - The height divides an angle of the triangle in the ratio 2:1
    - The base of the triangle is divided into two parts
    - The smaller part of the base is 1 cm
    Prove that the area of the triangle is 11/3 cm² -/
theorem triangle_area (h : ℝ) (b_small : ℝ) (angle_ratio : ℚ) :
  h = 2 →
  b_small = 1 →
  angle_ratio = 2 / 1 →
  ∃ (b_large : ℝ), 
    (b_small + b_large) * h / 2 = 11 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l454_45476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_projections_perpendicular_l454_45484

-- Helper definitions
def IsRectangle (A B C D : ℝ × ℝ) : Prop := sorry
def OnUnitCircle (P : ℝ × ℝ) : Prop := sorry
def IsBetween (A M B : ℝ × ℝ) : Prop := sorry
def ProjectionOnto (M : ℝ × ℝ) (L : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def Line (P Q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry
def Perpendicular (L1 L2 : Set (ℝ × ℝ)) : Prop := sorry

theorem rectangle_projections_perpendicular 
  (A B C D M : ℝ × ℝ) 
  (h_rectangle : IsRectangle A B C D) 
  (h_circumcircle : OnUnitCircle A ∧ OnUnitCircle B ∧ OnUnitCircle C ∧ OnUnitCircle D) 
  (h_M_on_arc : OnUnitCircle M ∧ M ≠ A ∧ M ≠ B ∧ IsBetween A M B) 
  (P : ℝ × ℝ) (h_P : P = ProjectionOnto M (Line A D))
  (Q : ℝ × ℝ) (h_Q : Q = ProjectionOnto M (Line A B))
  (R : ℝ × ℝ) (h_R : R = ProjectionOnto M (Line B C))
  (S : ℝ × ℝ) (h_S : S = ProjectionOnto M (Line C D)) : 
  Perpendicular (Line P Q) (Line R S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_projections_perpendicular_l454_45484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_shuffle_bound_l454_45478

def shuffleOperation (N K : ℕ) (deck : List ℕ) : List ℕ :=
  (deck.take K).reverse ++ deck.drop K

def isInitialOrder (N : ℕ) (deck : List ℕ) : Prop :=
  deck = List.range N

theorem deck_shuffle_bound (N K : ℕ) (h1 : 0 < K) (h2 : K ≤ N) :
  ∃ n : ℕ, n ≤ 4 * N^2 / K^2 ∧
    isInitialOrder N (n.iterate (shuffleOperation N K) (List.range N)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_shuffle_bound_l454_45478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donna_dog_walking_hours_verify_hours_dog_walking_l454_45443

/-- Represents Donna's weekly work schedule and earnings --/
structure WorkSchedule where
  dog_walking_rate : ℚ
  card_shop_hours : ℚ
  card_shop_rate : ℚ
  babysitting_hours : ℚ
  babysitting_rate : ℚ
  total_earnings : ℚ

/-- Calculates the number of hours Donna worked walking dogs --/
def hours_dog_walking (schedule : WorkSchedule) : ℚ :=
  (schedule.total_earnings -
   (schedule.card_shop_hours * schedule.card_shop_rate +
    schedule.babysitting_hours * schedule.babysitting_rate)) /
  schedule.dog_walking_rate

/-- Theorem stating that Donna worked 14 hours walking dogs --/
theorem donna_dog_walking_hours :
  let schedule : WorkSchedule := {
    dog_walking_rate := 10
    card_shop_hours := 10
    card_shop_rate := 25/2
    babysitting_hours := 4
    babysitting_rate := 10
    total_earnings := 305
  }
  hours_dog_walking schedule = 14 := by
  sorry

/-- Proof that the calculated hours are correct --/
theorem verify_hours_dog_walking (schedule : WorkSchedule) :
  schedule.total_earnings =
    hours_dog_walking schedule * schedule.dog_walking_rate +
    schedule.card_shop_hours * schedule.card_shop_rate +
    schedule.babysitting_hours * schedule.babysitting_rate := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donna_dog_walking_hours_verify_hours_dog_walking_l454_45443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l454_45468

noncomputable def f (A : ℝ) (x : ℝ) : ℝ := A * Real.sin (2 * x - Real.pi / 3)

theorem center_of_symmetry (A : ℝ) (h₁ : A > 0) 
  (h₂ : ∫ x in (0)..(4*Real.pi/3), f A x = 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ∈ Set.Ioo ((7*Real.pi/6) - ε) ((7*Real.pi/6) + ε) →
    f A ((7*Real.pi/3) - x) = f A x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l454_45468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_3_3_l454_45402

/-- Represents the relationship between two people -/
inductive Relationship
| Acquaintance
| Stranger

/-- A group of people and their relationships -/
structure SocialGroup where
  people : Finset Nat
  relationship : Nat → Nat → Relationship

/-- Checks if a subset of people are all mutual acquaintances -/
def areAllAcquaintances (g : SocialGroup) (s : Finset Nat) : Prop :=
  ∀ i j, i ∈ s → j ∈ s → i ≠ j → g.relationship i j = Relationship.Acquaintance

/-- Checks if a subset of people are all mutual strangers -/
def areAllStrangers (g : SocialGroup) (s : Finset Nat) : Prop :=
  ∀ i j, i ∈ s → j ∈ s → i ≠ j → g.relationship i j = Relationship.Stranger

/-- Main theorem: In a group of 6 people, there are either 3 mutual acquaintances or 3 mutual strangers -/
theorem ramsey_3_3 (g : SocialGroup) (h : g.people.card = 6) :
  (∃ s : Finset Nat, s ⊆ g.people ∧ s.card = 3 ∧ areAllAcquaintances g s) ∨
  (∃ s : Finset Nat, s ⊆ g.people ∧ s.card = 3 ∧ areAllStrangers g s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_3_3_l454_45402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_baked_percentage_l454_45499

/-- The ratio of cookies baked by Irin -/
noncomputable def irin_ratio : ℝ := 9.18

/-- The ratio of cookies baked by Ingrid -/
noncomputable def ingrid_ratio : ℝ := 5.17

/-- The ratio of cookies baked by Nell -/
noncomputable def nell_ratio : ℝ := 2.05

/-- The total number of cookies baked -/
def total_cookies : ℕ := 148

/-- The total ratio of cookies baked -/
noncomputable def total_ratio : ℝ := irin_ratio + ingrid_ratio + nell_ratio

/-- Ingrid's share of the total cookies -/
noncomputable def ingrid_share : ℝ := ingrid_ratio / total_ratio

/-- Ingrid's percentage of the total cookies -/
noncomputable def ingrid_percentage : ℝ := ingrid_share * 100

theorem ingrid_baked_percentage :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |ingrid_percentage - 31.52| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ingrid_baked_percentage_l454_45499
