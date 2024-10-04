import Mathlib

namespace algebraic_identity_l235_235655

theorem algebraic_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

example : (2011 : ℝ)^2 - (2010 : ℝ)^2 = 4021 := 
by
  have h := algebraic_identity 2011 2010
  rw [h]
  norm_num

end algebraic_identity_l235_235655


namespace cos_angle_between_vectors_l235_235814

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (1, 1, -2)

def vector_u : ℝ × ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2, vector_a.3 - vector_b.3)
def vector_v : ℝ × ℝ × ℝ := (2 * vector_a.1 + vector_b.1, 2 * vector_a.2 + vector_b.2, 2 * vector_a.3 + vector_b.3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def cos_theta (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (norm u * norm v)

theorem cos_angle_between_vectors :
  cos_theta vector_u vector_v = 19 / Real.sqrt 1036 :=
by
  sorry

end cos_angle_between_vectors_l235_235814


namespace lowest_temperature_l235_235269

theorem lowest_temperature
  (average_temp : ℝ)
  (max_range : ℝ)
  (sum_of_temps : ℝ)
  (temps_are_in_range : ∀ L H : ℝ, L + 15 = H) :
  ∃ L : ℝ, L = 33 :=
by
  let five_days_avg := 45
  let range := 15
  have sum_temps : sum_of_temps = 5 * five_days_avg,
  {
    exact 225
  }
  sorry

end lowest_temperature_l235_235269


namespace volume_of_parallelepiped_l235_235928

variables {ℝ : Type*} [inner_product_space ℝ ℝ³]
variables (a b : ℝ³)
open_real_inner_product_space

def unit_vector (v : ℝ³) : Prop := ∥v∥ = 1

def angle_between (v₁ v₂ : ℝ³) (θ : ℝ) : Prop :=
  (v₁ ≠ 0 ∧ v₂ ≠ 0) → real.angle (v₁, 0) (v₂, 0) = θ

def scalar_triple_product (v₁ v₂ v₃ : ℝ³) : ℝ :=
  (v₁.dot (v₂.cross v₃)).abs

theorem volume_of_parallelepiped :
  unit_vector a →
  unit_vector b →
  angle_between a b (real.pi / 4) →
  scalar_triple_product a (a + b.cross a) b = 1 / 2 :=
by sorry

end volume_of_parallelepiped_l235_235928


namespace number_of_diagonals_in_nonagon_l235_235824

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235824


namespace binomial_60_3_l235_235031

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235031


namespace urn_problem_l235_235729

theorem urn_problem :
  ∃ x : ℕ, (120 - (48 + x) = ⟨60, by sorry⟩) ∧ (↑48 / (120 - x) = 0.80) := by
  sorry

end urn_problem_l235_235729


namespace police_catch_fantomas_l235_235273

def City := ℕ
def AirlineRoute := City × City

structure Dodecaedria :=
  (cities : Finset City)
  (routes : Finset AirlineRoute)
  (pop_size : cities.card = 20) -- 20 cities
  (route_size : routes.card = 30) -- 30 airline routes

structure State :=
  (dodecaedria: Dodecaedria)
  (fantomas_city: City)
  (valid_route : valid_route dodecaedria.routes)

-- Initial state where valid routes conform to the description,
-- Fantomas can move and police can add/remove routes
def initial_state : Prop :=
  ∃ s : State, 
    -- Check Fantomas can move
    (∃ r ∈ s.dodecaedria.routes, r.1 = s.fantomas_city)
    -- Check police can shift routes
    ∧ ∀ r ∈ s.dodecaedria.routes, ∃ c₁ c₂ : City, c₁ ≠ c₂ ∧ (c₁, c₂) ∉ s.dodecaedria.routes

-- The proposition to prove: the police will eventually catch Fantomas
theorem police_catch_fantomas :
  ∃ catch_state : State, initial_state catch_state ∧ ∀ s₁ s₂ : State, 
    (s₁.fantomas_city = s₂.fantomas_city) → 
    (¬ ∃ r ∈ s₁.dodecaedria.routes, r.1 = s₁.fantomas_city) :=
sorry

end police_catch_fantomas_l235_235273


namespace find_four_consecutive_odd_numbers_l235_235723

noncomputable def four_consecutive_odd_numbers (a b c d : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  (a = b + 2 ∨ a = b - 2) ∧ (b = c + 2 ∨ b = c - 2) ∧ (c = d + 2 ∨ c = d - 2)

def numbers_sum_to_26879 (a b c d : ℤ) : Prop :=
  1 + (a + b + c + d) +
  (a * b + a * c + a * d + b * c + b * d + c * d) +
  (a * b * c + a * b * d + a * c * d + b * c * d) +
  (a * b * c * d) = 26879

theorem find_four_consecutive_odd_numbers (a b c d : ℤ) :
  four_consecutive_odd_numbers a b c d ∧ numbers_sum_to_26879 a b c d →
  ((a, b, c, d) = (9, 11, 13, 15) ∨ (a, b, c, d) = (-17, -15, -13, -11)) :=
by {
  sorry
}

end find_four_consecutive_odd_numbers_l235_235723


namespace nonagon_diagonals_count_l235_235831

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235831


namespace years_passed_l235_235700

-- Let PV be the present value of the machine, FV be the final value of the machine, r be the depletion rate, and t be the time in years.
def PV : ℝ := 900
def FV : ℝ := 729
def r : ℝ := 0.10

-- The formula for exponential decay is FV = PV * (1 - r)^t.
-- Given FV = 729, PV = 900, and r = 0.10, we want to prove that t = 2.

theorem years_passed (t : ℕ) : FV = PV * (1 - r)^t → t = 2 := 
by 
  intro h
  sorry

end years_passed_l235_235700


namespace salary_after_increase_is_correct_l235_235253

namespace SalaryIncrease

-- Define necessary constants
def salary : ℕ := 30000
def increase_percentage : ℝ := 0.10

-- Theorem statement
theorem salary_after_increase_is_correct :
  let new_salary := ℕ := (salary : ℝ) * (1 + increase_percentage) in
  new_salary = 33000 :=
by
  sorry

end SalaryIncrease

end salary_after_increase_is_correct_l235_235253


namespace each_wolf_needs_one_deer_l235_235300

-- Definitions used directly from conditions
variable (w_hunting : ℕ) -- wolves out hunting
variable (w_pack : ℕ) -- additional wolves in the pack
variable (m_per_wolf_per_day : ℕ) -- meat requirement per wolf per day
variable (d : ℕ) -- days until next hunt
variable (m_per_deer : ℕ) -- meat provided by one deer

-- Setting up conditions
def total_wolves : ℕ := w_hunting + w_pack
def daily_meat_requirement : ℕ := total_wolves * m_per_wolf_per_day
def total_meat_needed : ℕ := d * daily_meat_requirement
def deer_needed : ℕ := total_meat_needed / m_per_deer
def deer_per_wolf : ℕ := deer_needed / w_hunting

-- The statement to be proved
theorem each_wolf_needs_one_deer
  (h_w_hunting: w_hunting = 4)
  (h_w_pack: w_pack = 16)
  (h_m_per_wolf_per_day: m_per_wolf_per_day = 8)
  (h_d: d = 5)
  (h_m_per_deer: m_per_deer = 200) :
  deer_per_wolf = 1 :=
by
  unfold total_wolves daily_meat_requirement total_meat_needed deer_needed deer_per_wolf
  rw [h_w_hunting, h_w_pack, h_m_per_wolf_per_day, h_d, h_m_per_deer] -- Rewrite with given conditions
  sorry -- Proof steps are omitted

-- Assign actual values to the variables to ensure correctness
#eval deer_per_wolf 4 16 8 5 200 -- Expected evaluation result: 1

end each_wolf_needs_one_deer_l235_235300


namespace binom_60_3_eq_34220_l235_235041

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235041


namespace totalAmountAfterDebtsPaid_l235_235074

-- Define initial amounts and debts
def initialAmounts : List (String × Int) :=
  [("Earl", 90), ("Fred", 48), ("Greg", 36), ("Hannah", 72), ("Isabella", 60)]

def debts : List (String × (String × Int)) :=
  [ ("Earl", ("Fred", 28)), ("Earl", ("Hannah", 30)), ("Earl", ("Isabella", 15)),
    ("Fred", ("Greg", 32)), ("Fred", ("Hannah", 10)), ("Fred", ("Isabella", 20)),
    ("Greg", ("Earl", 40)), ("Greg", ("Hannah", 20)), ("Greg", ("Isabella", 8)),
    ("Hannah", ("Greg", 15)), ("Hannah", ("Earl", 25)), ("Hannah", ("Fred", 5)), ("Hannah", ("Isabella", 10)),
    ("Isabella", ("Earl", 18)), ("Isabella", ("Greg", 4)), ("Isabella", ("Hannah", 12)) ]

-- Define result for persons
def finalAmountsAfterDebtsPaid : List (String × Int) :=
  [("Earl", 100), ("Fred", 14), ("Greg", 15), ("Hannah", 77), ("Isabella", 69)]

-- Prove the combined final amounts
theorem totalAmountAfterDebtsPaid (combined_amount: Int) :
  combined_amount = 
    finalAmountsAfterDebtsPaid.lookup "Greg".getOrElse 0 +
    finalAmountsAfterDebtsPaid.lookup "Earl".getOrElse 0 +
    finalAmountsAfterDebtsPaid.lookup "Hannah".getOrElse 0 +
    finalAmountsAfterDebtsPaid.lookup "Isabella".getOrElse 0 := by
  have h1: combined_amount = 15 + 100 + 77 + 69 := by sorry
  have h2: 15 + 100 + 77 + 69 = 261 := by sorry
  exact eq.trans h1 h2

end totalAmountAfterDebtsPaid_l235_235074


namespace each_wolf_needs_one_deer_l235_235299

-- Definitions used directly from conditions
variable (w_hunting : ℕ) -- wolves out hunting
variable (w_pack : ℕ) -- additional wolves in the pack
variable (m_per_wolf_per_day : ℕ) -- meat requirement per wolf per day
variable (d : ℕ) -- days until next hunt
variable (m_per_deer : ℕ) -- meat provided by one deer

-- Setting up conditions
def total_wolves : ℕ := w_hunting + w_pack
def daily_meat_requirement : ℕ := total_wolves * m_per_wolf_per_day
def total_meat_needed : ℕ := d * daily_meat_requirement
def deer_needed : ℕ := total_meat_needed / m_per_deer
def deer_per_wolf : ℕ := deer_needed / w_hunting

-- The statement to be proved
theorem each_wolf_needs_one_deer
  (h_w_hunting: w_hunting = 4)
  (h_w_pack: w_pack = 16)
  (h_m_per_wolf_per_day: m_per_wolf_per_day = 8)
  (h_d: d = 5)
  (h_m_per_deer: m_per_deer = 200) :
  deer_per_wolf = 1 :=
by
  unfold total_wolves daily_meat_requirement total_meat_needed deer_needed deer_per_wolf
  rw [h_w_hunting, h_w_pack, h_m_per_wolf_per_day, h_d, h_m_per_deer] -- Rewrite with given conditions
  sorry -- Proof steps are omitted

-- Assign actual values to the variables to ensure correctness
#eval deer_per_wolf 4 16 8 5 200 -- Expected evaluation result: 1

end each_wolf_needs_one_deer_l235_235299


namespace triangle_area_correct_l235_235381

/-- Define the points of the triangle -/
def x1 : ℝ := -4
def y1 : ℝ := 2
def x2 : ℝ := 2
def y2 : ℝ := 8
def x3 : ℝ := -2
def y3 : ℝ := -2

/-- Define the area calculation function -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Define the area of the given triangle -/
def given_triangle_area : ℝ :=
  triangle_area x1 y1 x2 y2 x3 y3

/-- The goal is to prove that the area of the given triangle is 22 square units -/
theorem triangle_area_correct : given_triangle_area = 22 := by
  sorry

end triangle_area_correct_l235_235381


namespace problem_statement_l235_235943

noncomputable def percent_of_y (y : ℝ) (z : ℂ) : ℝ :=
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10).re

theorem problem_statement (y : ℝ) (z : ℂ) (hy : y > 0) : percent_of_y y z = 0.6 * y :=
by
  sorry

end problem_statement_l235_235943


namespace solve_equal_complex_l235_235259

theorem solve_equal_complex :
  ∀ y : ℂ, (y^2 - 5 * y + 6 = -(y + 1) * (y + 7)) ↔
  (y = (-3 + Complex.I * Complex.sqrt 95) / 4 ∨ y = (-3 - Complex.I * Complex.sqrt 95) / 4) :=
by 
  intro y
  sorry

end solve_equal_complex_l235_235259


namespace sum_odds_200_600_l235_235323

-- Define the bounds 200 and 600 for our range
def lower_bound := 200
def upper_bound := 600

-- Define first and last odd integers in the range
def first_odd := 201
def last_odd := 599

-- Define the common difference in our arithmetic sequence
def common_diff := 2

-- Number of terms in the sequence
def n := ((last_odd - first_odd) / common_diff) + 1

-- Sum of the arithmetic sequence formula
def sum_arithmetic_seq (n : ℕ) (a l : ℕ) : ℕ :=
  n * (a + l) / 2

-- Specifically, the sum of odd integers between 200 and 600
def sum_odd_integers : ℕ := sum_arithmetic_seq n first_odd last_odd

-- Theorem stating the sum is equal to 80000
theorem sum_odds_200_600 : sum_odd_integers = 80000 :=
by sorry

end sum_odds_200_600_l235_235323


namespace average_speed_needed_l235_235688

theorem average_speed_needed (distance : ℝ) (late_time : ℝ) (late_speed : ℝ) : 
  distance = 70 → late_time = 0.25 → late_speed = 35 → 
  (distance / (distance / late_speed - late_time) = 40) :=
begin
  intros,
  have t35 : distance / late_speed = 2,
  { rw [h, h1, h2], norm_num },
  have to_on_time : distance / late_speed - late_time = 1.75,
  { rw [<- t35, h1], norm_num },
  rw [<- to_on_time, h],
  norm_num,
end

end average_speed_needed_l235_235688


namespace donut_selection_count_l235_235977

-- Define the conditions formally in Lean
def donuts_count_sum (g c p j : ℕ) : Prop := g + c + p + j = 6

-- Define the question as a theorem statement in Lean
theorem donut_selection_count : 
  ∃ (count : ℕ), (count = 84) ∧ 
  (∃ (g c p j : ℕ), donuts_count_sum g c p j) :=
begin
  sorry -- Proof to be provided
end

end donut_selection_count_l235_235977


namespace sine_cosine_solution_set_l235_235401

theorem sine_cosine_solution_set (x : ℝ) (k : ℤ) :
  (sin (x / 2) - cos (x / 2) = 1) ↔ (∃ k : ℤ, (x = π * (1 + 4 * k)) ∨ (x = 2 * π * (1 + 2 * k))) := 
sorry

end sine_cosine_solution_set_l235_235401


namespace probability_of_sum_4_twice_l235_235651

def probability_two_dice_sum_4_twice : ℚ := 
  let outcomes := ∑ (d1 d2 : ℕ) in if 1 ≤ d1 ∧ d1 ≤ 3 ∧ 1 ≤ d2 ∧ d2 ≤ 3 then 1 else 0
  let favorable_outcomes := ∑ (d1 d2 : ℕ) in if 1 ≤ d1 ∧ d1 ≤ 3 ∧ 1 ≤ d2 ∧ d2 ≤ 3 ∧ d1 + d2 = 4 then 1 else 0
  let probability_sum_4 := (favorable_outcomes : ℚ) / (outcomes : ℚ)
  probability_sum_4 * probability_sum_4 

theorem probability_of_sum_4_twice : probability_two_dice_sum_4_twice = 1 / 9 := by
  sorry

end probability_of_sum_4_twice_l235_235651


namespace number_of_diagonals_in_nonagon_l235_235826

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235826


namespace rectangle_cut_possible_l235_235706

-- Define the dimensions of the original and new rectangle
def original_length : ℕ := 14
def original_width : ℕ := 6
def original_area : ℕ := original_length * original_width

def new_length : ℕ := 21
def new_width : ℕ := 4
def new_area : ℕ := new_length * new_width

-- Define the theorem
theorem rectangle_cut_possible :
  (original_area = new_area) ∧ ∃ (cut: ℕ → ℕ → Prop), 
  ∀ (x y : ℕ), 
  (cut x y → x * y = original_area) ∧ 
  (cut x y → (x * y = new_length * new_width)) :=
by
  -- Equate the areas first
  have h_area_eq : original_area = new_area,
  -- Steps and logic to show the areas are equal
  sorry,
  -- Existence of a cut such that the rearrangement forms the new dimensions
  use λ x y, (x = 7 ∧ y = 6),
  intros x y cut_def,
  split,
  { -- Prove that the cut forms pieces equal in original area
    sorry
  },
  { -- Prove the reconfiguration matches the target dimensions
    sorry
  }

end rectangle_cut_possible_l235_235706


namespace trapezoid_rectangle_ratio_limit_l235_235197

noncomputable def radius : ℝ := 10
noncomputable def OP (t : ℝ) : ℝ := 10 - 3 * t

def area_WZXY (t : ℝ) : ℝ := 
  let XW := real.sqrt (radius ^ 2 - (OP t) ^ 2)
  (XW) * 3

def area_ABXY (t : ℝ) : ℝ := 
  let XW := real.sqrt (radius ^ 2 - (OP t) ^ 2)
  let AY := real.sqrt (radius ^ 2 - (OP (t + 1)) ^ 2)
  ((AY + XW) / 2) * 3

theorem trapezoid_rectangle_ratio_limit :
  tendsto (λ t, (area_ABXY t) / (area_WZXY t)) (nhds 3) (nhds (real.sqrt 2)) :=
sorry

end trapezoid_rectangle_ratio_limit_l235_235197


namespace distinct_diagonals_in_convex_nonagon_l235_235881

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235881


namespace range_of_a_l235_235294

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1 > 2 * x - 2) → (x < a)) → (a ≥ 3) :=
by
  sorry

end range_of_a_l235_235294


namespace PQ_composition_l235_235556

def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem PQ_composition : P (Q (P (Q (P (Q 2))))) = 54 := 
by sorry

end PQ_composition_l235_235556


namespace maximum_students_on_playground_l235_235663

theorem maximum_students_on_playground (S P N E : ℕ) (h1 : 170 = 8 + S * P) (h2 : 268 = -2 + S * N) (h3 : 120 = 12 + S * E) : S = 54 :=
by {
  -- The proof steps are omitted.
  sorry
}

end maximum_students_on_playground_l235_235663


namespace solution_inequality_l235_235098

open Set

theorem solution_inequality (x : ℝ) : (x > 3 ∨ x < -3) ↔ (x > 9 / x) := by
  sorry

end solution_inequality_l235_235098


namespace binom_60_3_eq_34220_l235_235049

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235049


namespace nonagon_diagonals_count_l235_235913

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235913


namespace problem1_problem2_l235_235809

theorem problem1 (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 2 → 3*x^2*exp(3*a*x)*(1 + a*x) ≥ 0) →
  a ≥ -1/2 :=
sorry

theorem problem2 (a : ℝ) :
  ((∀ x : ℝ, 0 < x ∧ x ≤ 2 → 3*x^2*exp(3*a*x)*(1 + a*x) ≥ 0) ∨
  (∃ x : ℝ, x > 0 ∧ a*x - a/x + 2*log x has an extreme value)) ∧
  ¬((∀ x : ℝ, 0 < x ∧ x ≤ 2 → 3*x^2*exp(3*a*x)*(1 + a*x) ≥ 0) ∧
  (∃ x : ℝ, x > 0 ∧ a*x - a/x + 2*log x has an extreme value)) →
  a ∈ Set.Ioo (-1 : ℝ) (-1/2 : ℝ) ∪ Set.Ici (0 : ℝ) :=
sorry

end problem1_problem2_l235_235809


namespace sum_of_solutions_eq_seven_l235_235763

theorem sum_of_solutions_eq_seven : 
  ∃ x : ℝ, x + 49/x = 14 ∧ (∀ y : ℝ, y + 49 / y = 14 → y = x) → x = 7 :=
by {
  sorry
}

end sum_of_solutions_eq_seven_l235_235763


namespace part_I5_1_part_I5_2_part_I5_3_part_I5_4_l235_235495

-- Part I5.1: 1 + 2 + 3 + ... + t = 36 -> t = 8
theorem part_I5_1 (t : ℕ) (h : (1 + 2 + 3 + ... + t) = 36) : t = 8 := sorry

-- Part I5.2: sin u° = 2 / sqrt(t), 90° < u < 180° -> u = 135°
theorem part_I5_2 (u : ℝ) (t : ℕ) (h1 : real.sin (u * π / 180) = 2 / real.sqrt (t : ℝ))
 (h2 : 90 < u ∧ u < 180) (ht : t = 8) : u = 135 := sorry

-- Part I5.3: ∠ABC = 30°, AC = (u - 90) cm -> AC = 45 cm
theorem part_I5_3 (u AC : ℝ) (h1 : u = 135) (h2 : AC = (u - 90)) : AC = 45 := sorry

-- Part I5.4: ∠APB = 40°, APB = (v-5)° -> w = 70°
theorem part_I5_4 (w v : ℝ) (h1 : 40 = (v - 5)) : w = 70 := sorry

end part_I5_1_part_I5_2_part_I5_3_part_I5_4_l235_235495


namespace prove_partial_fractions_identity_l235_235761

def partial_fraction_identity (x : ℚ) (A B C a b c : ℚ) : Prop :=
  a = 0 ∧ b = 1 ∧ c = -1 ∧
  (A / (x - a) + B / (x - b) + C / (x - c) = 4*x - 2 ∧ x^3 - x ≠ 0)

theorem prove_partial_fractions_identity :
  (partial_fraction_identity x 2 1 (-3) 0 1 (-1)) :=
by {
  sorry
}

end prove_partial_fractions_identity_l235_235761


namespace nathan_tokens_used_is_18_l235_235583

-- We define the conditions as variables and constants
variables (airHockeyGames basketballGames tokensPerGame : ℕ)

-- State the values for the conditions
def Nathan_plays : Prop :=
  airHockeyGames = 2 ∧ basketballGames = 4 ∧ tokensPerGame = 3

-- Calculate the total tokens used
def totalTokensUsed (airHockeyGames basketballGames tokensPerGame : ℕ) : ℕ :=
  (airHockeyGames * tokensPerGame) + (basketballGames * tokensPerGame)

-- Proof statement 
theorem nathan_tokens_used_is_18 : Nathan_plays airHockeyGames basketballGames tokensPerGame → totalTokensUsed airHockeyGames basketballGames tokensPerGame = 18 :=
by 
  sorry

end nathan_tokens_used_is_18_l235_235583


namespace solution_set_inequality_l235_235069

theorem solution_set_inequality (x : ℝ) : 
  (abs (x + 3) - abs (x - 2) ≥ 3) ↔ (x ≥ 1) := 
by {
  sorry
}

end solution_set_inequality_l235_235069


namespace nonagon_diagonals_count_l235_235833

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235833


namespace sequence_k_value_l235_235119

theorem sequence_k_value (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, n ≥ 2 → a n = a 1 + ∑ i in finset.range (n - 1), (1 / (i + 1)) * a (i + 1))
  (hk : a 2017 = 2017) : k = 2017 :=
sorry

end sequence_k_value_l235_235119


namespace two_digit_solutions_count_l235_235492

theorem two_digit_solutions_count :
  card {x : ℕ | (10 ≤ x ∧ x ≤ 99) ∧ (3269 * x + 532) % 17 = 875 % 17} = 5 :=
by
  sorry

end two_digit_solutions_count_l235_235492


namespace integral_pairs_l235_235060

theorem integral_pairs (a b : ℝ) (h : ∀ n : ℕ, 0 < n → a * ⌊b * n⌋ = b * ⌊a * n⌋) : a ∈ ℤ ∧ b ∈ ℤ :=
sorry

end integral_pairs_l235_235060


namespace problem_proof_l235_235966

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def irrational_count (lst : List ℝ) : ℕ :=
  lst.countp is_irrational

theorem problem_proof :
  irrational_count [Real.sqrt 4, 22 / 7, -1 / 3, 0, 0.301, 
                    Real.pi, Real.cbrt 9, (0.301300130001 : ∑ i in Finset.range ∞, 10^(-i!))] = 3 :=
by
  sorry

end problem_proof_l235_235966


namespace broomstick_race_order_l235_235159

noncomputable def valid_finish_orders : ℕ :=
  let total_permutations := 4! in
  let forbidden_permutations := 2 * (3! / 2) in
  total_permutations - forbidden_permutations

theorem broomstick_race_order :
  valid_finish_orders = 12 := sorry

end broomstick_race_order_l235_235159


namespace math_problem_l235_235967

noncomputable def line_equation : Prop :=
  ∀ (t: ℝ), (∃ (x y: ℝ), x = 2 + 1/2 * t ∧ y = √3 / 2 * t → y = √3 * (x - 2))

noncomputable def curve_equation : Prop :=
  ∀ (ρ θ: ℝ), (∃ (x y: ℝ), ρ * sin θ^2 - 4 * cos θ = 0 → y^2 = 4 * x)

noncomputable def ma_mb_value : Prop :=
  ∀ (A B: ℝ × ℝ), (∃ (t1 t2: ℝ) (M: ℝ × ℝ), 
    M = (2,0) ∧ (3 * t1^2 - 8 * t1 - 32 = 0) ∧ (3 * t2^2 - 8 * t2 - 32 = 0) ∧ 
    A = (2 + 1/2 * t1, √3 / 2 * t1) ∧ B = (2 + 1/2 * t2, √3 / 2 * t2) → 
    | (1 / dist M A ) - (1 / dist M B) | = 1/4)

theorem math_problem :
  line_equation ∧ curve_equation ∧ ma_mb_value 
    := sorry

end math_problem_l235_235967


namespace sum_of_odd_integers_l235_235322

theorem sum_of_odd_integers (a₁ aₙ d n : ℕ) (h₁ : a₁ = 201) (h₂ : aₙ = 599) (h₃ : d = 2) (h₄ : aₙ = a₁ + (n - 1) * d) :
  (∑ i in finset.range(n), a₁ + i * d) = 80000 :=
by
  sorry

end sum_of_odd_integers_l235_235322


namespace distinct_diagonals_in_convex_nonagon_l235_235889

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235889


namespace binom_60_3_l235_235016

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235016


namespace minimum_value_AP_PB_l235_235354

noncomputable def point_A := (-3, 3) : ℝ × ℝ
noncomputable def point_A' := (-3, -3) : ℝ × ℝ
noncomputable def circle_center := (1, 1) : ℝ × ℝ
noncomputable def circle_radius := Real.sqrt 2

def distance (a b : ℝ × ℝ) : ℝ := 
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Define the minimum value of |AP| + |PB| when line A'P passes through the circle center
theorem minimum_value_AP_PB :
  ∃ (P : ℝ × ℝ), P.2 = 0 ∧
    let B := sorry in -- Point B where the reflected ray intersects the circle, to be specified properly
    distance point_A P + distance P B = 3 * Real.sqrt 2 := sorry

end minimum_value_AP_PB_l235_235354


namespace equation_of_ellipse_fixed_midpoint_of_MN_l235_235125

variables {a b : ℝ} (h_a_gt_b : a > b) (h_b_gt_zero : b > 0)
variables (ellipse : (a > b > 0) ∧ (b^2 = 4) ∧ (a^2 = 4 + (5/9) * a^2) ∧ (eccentricity : Real.sqrt 5 / 3))

def point_on_ellipse (x y : ℝ) (hx : (x, y) = (-2, 0)) : Prop :=
  ((y^2 / a^2) + (x^2 / b^2) = 1)

theorem equation_of_ellipse :
  (∃ a b : ℝ, (a > b > 0) ∧ (b^2 = 4) ∧ (a^2 = 9)) → (∀ x y : ℝ, (point_on_ellipse x y ⟨hx⟩) → 
  (x^2 / b^2 + y^2 / a^2 = 1)) :=
sorry

variables (point : (-2, 3)) (p q m n : ℝ × ℝ)
variables (h_intersect_pq : ∃ p q, (line_through point p ∧ line_through point q))
variables (h_intersect_yaxis_mn : ∃ m n, (intersect_yaxis m ∧ intersect_yaxis n))

def midpoint (x y : ℝ × ℝ) : ℝ × ℝ :=
  ((x.1 + y.1) / 2, (x.2 + y.2) / 2)

theorem fixed_midpoint_of_MN (x y : ℝ × ℝ) (h : ((x.1 = 0) ∨ (y.1 = 0)) ∧ ((x.2 = 3) ∧ (y.2 = 3))) :
  midpoint x y = (0, 3) :=
sorry

end equation_of_ellipse_fixed_midpoint_of_MN_l235_235125


namespace area_of_region_bounded_by_absx_and_circle_l235_235414

theorem area_of_region_bounded_by_absx_and_circle :
  let circle_eq (x y : ℝ) := x^2 + y^2 = 9
  let abs_line_eq (x y : ℝ) := y = real.abs x
  ∃ (area : ℝ), area = 9 * real.pi / 4 ∧
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (y = x ∨ y = -x) → 0 ≤ x * y) →
  ∀ (s : set (ℝ × ℝ)), 
    (∀ x y, (x, y) ∈ s ↔ (circle_eq x y ∧ abs_line_eq x y)) → 
    area = (measure_theory.measure μ s).to_real :=
by
  /- Proof goes here. -/
  sorry

end area_of_region_bounded_by_absx_and_circle_l235_235414


namespace integral_f_l235_235430

def f (x : ℝ) : ℝ :=
if -1 ≤ x ∧ x ≤ 0 then x^2 else
if 0 < x ∧ x < 1 then 1 else 
0

theorem integral_f : ∫ x in -1..1, f x = 4/3 :=
by
  sorry

end integral_f_l235_235430


namespace angle_bisector_eq_l235_235617

theorem angle_bisector_eq (k b : ℝ) : 
  k = (1 + Real.sqrt 6) / 3 ∧ b = (6 - 2 * Real.sqrt 6) / 9 ↔ 
  ∃ (x : ℝ), x = 1 → 
  let y₁ := 2 * x,
      y₂ := -x + 2,
      m₁ := 2,
      m₂ := -1 in
  (k = (m₁ + m₂ + Real.sqrt (1 + m₁ ^ 2 + m₂ ^ 2)) / (1 - m₁ * m₂) ∨ 
  k = (m₁ + m₂ - Real.sqrt (1 + m₁ ^ 2 + m₂ ^ 2)) / (1 - m₁ * m₂)) ∧ 
  (y₁ = (k : ℝ) * x + b ∨ y₂ = (k : ℝ) * x + b) :=
by {
  sorry
}

end angle_bisector_eq_l235_235617


namespace cone_angle_generatrix_height_l235_235413

theorem cone_angle_generatrix_height
  (R : ℝ) (l : ℝ) (x : ℝ)
  (base_area : ℝ := π * R^2)
  (lateral_surface_area : ℝ := π * R * l)
  (total_surface_area : ℝ := π * R^2 + π * R * l)
  (geometric_mean_condition : lateral_surface_area^2 = base_area * total_surface_area) :
  x = arcsin ((√5 - 1) / 2) := 
sorry

end cone_angle_generatrix_height_l235_235413


namespace locus_of_C_l235_235114

theorem locus_of_C {O A B C D : Point} (R : ℝ) 
  (circle : Circle O R) 
  (hA : A ∈ interior(circle)) 
  (hB : B ∈ circle) 
  (hD : D ∈ circle) 
  (hRect : is_rectangle ABCD) :
  locus_of_C.C.circle = Circle O (Real.sqrt (2 * R^2 - (OA dist O A)^2)) :=
sorry

end locus_of_C_l235_235114


namespace dot_product_PA_PB_eq_one_l235_235960

noncomputable def origin := (0, 0)
noncomputable def pointP := (-1, 1)
noncomputable def inclination_angle := 5 * Real.pi / 6
noncomputable def polar_equation_curveC (theta : ℝ) : ℝ := 4 * Real.sin theta

-- Parametric equations of line l
noncomputable def parametric_x (t : ℝ) : ℝ := -1 - Real.sqrt 3 / 2 * t
noncomputable def parametric_y (t : ℝ) : ℝ := 1 + 1 / 2 * t

-- Cartesian equation of curve C
theorem dot_product_PA_PB_eq_one :
  ∀ (t1 t2 : ℝ),
    let PA := (t1 * (-Real.sqrt 3 / 2), t1 * (1 / 2))
    let PB := (t2 * (-Real.sqrt 3 / 2), t2 * (1 / 2))
    in
    ((parametric_x t1) ^ 2 + (parametric_y t1) ^ 2 = 4 * (parametric_y t1)) →
    ((parametric_x t2) ^ 2 + (parametric_y t2) ^ 2 = 4 * (parametric_y t2)) →
    PA.1 * PB.1 + PA.2 * PB.2 = 1 :=
by
  intros t1 t2 h1 h2
  let e := (-Real.sqrt 3 / 2, 1 / 2)
  let PA := (t1 * e.1, t1 * e.2)
  let PB := (t2 * e.1, t2 * e.2)
  sorry

end dot_product_PA_PB_eq_one_l235_235960


namespace monotonic_solution_l235_235673

noncomputable def solution_correct (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x > 0 → y > 0 → f(x + y) = (f x * f y) / (f x + f y)) →
  (∀ x : ℝ, x > 0 → f x ≠ 0) →
  (∀ x y : ℝ, x > 0 → y > 0 → (x < y → f x < f y) ∨ (x > y → f x > f y)) →
  ∃ c : ℝ, c ≠ 0 ∧ (∀ x : ℝ, x > 0 → f x = c / x)

-- We state the problem but do not provide the proof
theorem monotonic_solution (f : ℝ → ℝ) :
  solution_correct f :=
sorry

end monotonic_solution_l235_235673


namespace no_constant_term_expansion_l235_235320

theorem no_constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∀ k : ℤ, (↑12.choose (nat_abs k)) * (x ^ ((12 - k)/2) * ((-3) ^ k) * x ^ (-2 * k)) ≠ x ^ 0 :=
by sorry

end no_constant_term_expansion_l235_235320


namespace total_grouping_schemes_l235_235695

theorem total_grouping_schemes (m f : ℕ) (total : ℕ) (max_group : ℕ) 
    (no_female_alone : ∀ g1 g2 : Set ℕ, g1 ∪ g2 = Set.range (m + f) → 
        (g1 ∩ Set.filter (fun x => x ≥ m) (Set.range (m + f)) ≠ ∅ ∧ 
         g2 ∩ Set.filter (fun x => x ≥ m) (Set.range (m + f)) ≠ ∅)) :
    m = 4 → f = 3 → total = 7 → max_group = 5 → 
    (Σ (g1 g2 : Set ℕ), g1 ∪ g2 = Set.range total ∧
        g1.card ≤ max_group ∧ g2.card ≤ max_group ∧ 
        (g1 ∩ Set.filter (fun x => x ≥ m) (Set.range total) ≠ ∅ ∧ 
         g2 ∩ Set.filter (fun x => x ≥ m) (Set.range total))).card = 104 :=
begin
  sorry
end

end total_grouping_schemes_l235_235695


namespace max_M_value_l235_235101

noncomputable section

def J_k (k : ℕ) : ℕ := 10 ^ (k + 2) + 128

def M (k : ℕ) : ℕ :=
  let n := J_k k
  let factors := n.factorization 
  factors.findWithDefault 2 0

theorem max_M_value : ∃ k > 0, M k = 8 := by
  sorry

end max_M_value_l235_235101


namespace number_of_factors_and_sum_of_even_factors_l235_235164

theorem number_of_factors_and_sum_of_even_factors (n : ℕ) (h : n = 945) :
  (∃ k, k = 16 ∧ ∀ d : ℕ, d ∣ n → ∃ l, l = ∑ d' in (finset.filter (λ x, even x) (finset.filter (λ x, x ∣ n) (finset.Icc 1 n))), d' = 2 * (∑ o in (finset.filter (λ x, odd x) (finset.filter (λ x, x ∣ n) (finset.Icc 1 n))), o)) ∧ l = 7650) :=
sorry

end number_of_factors_and_sum_of_even_factors_l235_235164


namespace book_has_50_pages_l235_235158

noncomputable def sentences_per_hour : ℕ := 200
noncomputable def hours_to_read : ℕ := 50
noncomputable def sentences_per_paragraph : ℕ := 10
noncomputable def paragraphs_per_page : ℕ := 20

theorem book_has_50_pages :
  (sentences_per_hour * hours_to_read) / sentences_per_paragraph / paragraphs_per_page = 50 :=
by
  sorry

end book_has_50_pages_l235_235158


namespace sum_sequence_value_l235_235435

noncomputable def S (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n-1)

theorem sum_sequence_value (n : ℕ) : 
  (∑ k in Finset.range (n-1), S k * S (k+1)) = (2^(2*n-1) - 2)/3 := 
  sorry

end sum_sequence_value_l235_235435


namespace binom_inv_sum_eq_rhs_l235_235545

-- Define binomial coefficient inverse sum
def binom_inv_sum (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), 1 / Nat.choose n k

-- Define the right-hand side of the equation
def rhs (n : ℕ) : ℚ := (n + 1 : ℚ) / (2 ^ (n + 1)) * ∑ k in Finset.range (n + 1).succ, 2 ^ k / k

-- The main theorem stating the equality
theorem binom_inv_sum_eq_rhs (n : ℕ) (hn : n > 0) : binom_inv_sum n = rhs n :=
by
  sorry

end binom_inv_sum_eq_rhs_l235_235545


namespace smallest_positive_period_range_on_interval_l235_235145

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 2 * sin (x / 2) * cos (x / 2) - sqrt 2 * sin (x / 2) ^ 2

theorem smallest_positive_period
  : ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem range_on_interval : 
  ∀ x ∈ set.Icc (-π) 0, f x ∈ set.Icc (-1 - sqrt 2 / 2) 0 :=
sorry

end smallest_positive_period_range_on_interval_l235_235145


namespace maximum_value_M_l235_235100

-- Define the function J_k as described in the problem
def J (k : ℕ) : ℕ := 10^(k + 3) + 128

-- Define the function M(k), which counts the number of factors of 2 in the factorization of J_k
def M (k : ℕ) : ℕ := (J k).factor 2

-- Theorem statement proving the maximum value of M(k)
theorem maximum_value_M (k : ℕ) (h : k > 0) : M k ≤ 8 :=
sorry

end maximum_value_M_l235_235100


namespace trigonometric_expression_equals_one_l235_235393

noncomputable def trigonometric_expression : ℝ :=
  (1 - 1 / (Real.cos (π / 6))) *
  (1 + 1 / (Real.sin (π / 3))) *
  (1 - 1 / (Real.sin (π / 6))) *
  (1 + 1 / (Real.cos (π / 3)))

theorem trigonometric_expression_equals_one : trigonometric_expression = 1 := 
by
  sorry

end trigonometric_expression_equals_one_l235_235393


namespace jenny_distance_diff_l235_235541

theorem jenny_distance_diff (r w : ℝ) (hr : r = 0.6) (hw : w = 0.4) : r - w = 0.2 :=
by {
  rw [hr, hw],
  exact sub_eq_add_neg(0.6, 0.4),
  have : 0.6 - 0.4 = 0.2 := by norm_num,
  exact this,
  sorry
}

end jenny_distance_diff_l235_235541


namespace cube_surface_area_l235_235308

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  ((q.1 - p.1)^2 + (q.2 - p.2)^2 + (q.3 - p.3)^2).sqrt

variable (A B C : ℝ × ℝ × ℝ)
variable (side_length : ℝ)

-- Given points
def A := (2, 4, 6)
def B := (3, 0, -3)
def C := (6, -5, 5)

-- Side length of the cube
def side_length := (distance A B) / (Math.sqrt 2)

theorem cube_surface_area : 6 * side_length^2 = 294 :=
by
  sorry

end cube_surface_area_l235_235308


namespace perpendicular_bisectors_concurrent_l235_235440

universe u
variables (Γ₁ Γ₂ Γ₃ : Type u) [circle Γ₁] [circle Γ₂] [circle Γ₃]
variables {P P₁₂ P₁₃ P₂₁ P₂₃ P₃₁ P₃₂ : Point}
variable  (Tangent : Point → circle → Line)

noncomputable def PerpendicularBisector (A B : Point) : Line := sorry

theorem perpendicular_bisectors_concurrent :
  (OnCircle P Γ₁) ∧ (OnCircle P Γ₂) ∧ (OnCircle P Γ₃) ∧
  (OnCircle P₁₂ Γ₂) ∧ (OnCircle P₁₃ Γ₃) ∧
  (OnCircle P₂₁ Γ₁) ∧ (OnCircle P₂₃ Γ₃) ∧
  (OnCircle P₃₁ Γ₁) ∧ (OnCircle P₃₂ Γ₂) ∧
  (Tangent P Γ₁ = Line.through P P₁₂) ∧ (Tangent P Γ₁ = Line.through P P₁₃) ∧
  (Tangent P Γ₂ = Line.through P P₂₁) ∧ (Tangent P Γ₂ = Line.through P P₂₃) ∧
  (Tangent P Γ₃ = Line.through P P₃₁) ∧ (Tangent P Γ₃ = Line.through P P₃₂) →
  ∃ O : Point, OnLine O (PerpendicularBisector P₁₂ P₁₃) ∧
               OnLine O (PerpendicularBisector P₂₁ P₂₃) ∧
               OnLine O (PerpendicularBisector P₃₁ P₃₂) :=
sorry

end perpendicular_bisectors_concurrent_l235_235440


namespace number_of_liars_l235_235965

theorem number_of_liars (k : ℕ) (n : ℕ) (statements : ℕ → Prop) :
  n = 12 →
  (∀ i, (i ≥ 1 ∧ i ≤ 12) →
    (if statements i then k = i - 1 else k > i - 1)) →
  (k = 6) →
  (n - k = 6) :=
by
  intros h_n h_statements h_k
  rw h_n
  rw h_k
  norm_num
  sorry

end number_of_liars_l235_235965


namespace virginia_eggs_l235_235316

def initial_eggs := 96
def eggs_taken_away := 3
def final_eggs := initial_eggs - eggs_taken_away

theorem virginia_eggs : final_eggs = 93 := by
  have h : 96 - 3 = 93 := rfl
  exact h

end virginia_eggs_l235_235316


namespace sector_area_max_sector_area_l235_235139

-- Definitions based on the given conditions
def perimeter : ℝ := 8
def central_angle (α : ℝ) : Prop := α = 2

-- Question 1: Find the area of the sector given the central angle is 2 rad
theorem sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) (h2 : l = 2 * r) : 
  (1/2) * r * l = 4 := 
by sorry

-- Question 2: Find the maximum area of the sector and the corresponding central angle
theorem max_sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) : 
  ∃ r, 0 < r ∧ r < 4 ∧ l = 8 - 2 * r ∧ 
  (1/2) * r * l = 4 ∧ l = 2 * r := 
by sorry

end sector_area_max_sector_area_l235_235139


namespace distinct_diagonals_nonagon_l235_235868

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235868


namespace correct_choices_l235_235727

-- Definitions of the conditions based on the problem statement
def condition1 := ∀ (θ : ℝ), true -- Radian angles correspond to real numbers
def condition2 := ∀ (θ1 θ2 : ℝ), true -- Placeholder, actual statement would be defined based on equal terminal sides
def condition3 := ∀ (θ : ℝ), (0 < θ ∧ θ < π / 2) → true -- Acute angles are first quadrant angles
def condition4 := ∀ (θ : ℝ), true -- Placeholder, less than 90 implies acute
def condition5 := ∀ (θ1 θ2 : ℝ), true -- Placeholder, sizes of angles in quadrants

-- Correct propositions among them
def correct_propositions := [condition1, condition3]

-- Statement to prove
theorem correct_choices : correct_propositions = [condition1, condition3] := by
  sorry

end correct_choices_l235_235727


namespace eval_complex_expression_l235_235077

theorem eval_complex_expression (ω : ℂ) (hω : ω = 8 + 3 * Complex.i) :
  abs (ω^2 + 6 * ω + 73) = 181 := by
  rw [hω]
  sorry

end eval_complex_expression_l235_235077


namespace sequence_periodic_value_l235_235507

noncomputable def a : ℕ → ℝ
| 0 := 2         -- This handles a_1 since Lean indexes from 0
| 1 := 1 / 3     -- This handles a_2
| (n + 2) := a (n + 1) / a n  -- Recursive relation

theorem sequence_periodic_value :
  a 2015 = 6 :=     -- Note: a_2016 in Lean is indexed by 2015
by {
  sorry
}

end sequence_periodic_value_l235_235507


namespace nonagon_diagonals_count_l235_235891

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235891


namespace probability_five_consecutive_math_majors_l235_235643

theorem probability_five_consecutive_math_majors 
  (people : Fin 12 → ℕ)
  (mMajors : Fin 5 → Fin 12) 
  (pMajors : Fin 4 → Fin 12)
  (bMajors : Fin 3 → Fin 12)
  (distinct_majors : Function.Injective mMajors)
  (majors_disjoint1 : ∀ i, ¬ (mMajors i ∈ pMajors '' Finset.univ))
  (majors_disjoint2 : ∀ i, ¬ (pMajors i ∈ bMajors '' Finset.univ))
  : ((∑ (i : Fin 12), if ∀ j : Fin 5, mMajors j = i + j % 12 then 1 else 0).toRat / (792).toRat = 1 / 66) :=
by sorry

end probability_five_consecutive_math_majors_l235_235643


namespace geometry_test_pass_condition_l235_235373

theorem geometry_test_pass_condition (total_problems : ℕ) (passing_score_percent : ℝ) (problems_missed : ℕ) :
  total_problems = 50 →
  passing_score_percent = 75 →
  (problems_missed : ℝ) ≤ total_problems * (1 - (passing_score_percent / 100)) →
  problems_missed = 12 :=
begin
  intros h_total h_pass,
  sorry
end

end geometry_test_pass_condition_l235_235373


namespace binomial_60_3_eq_34220_l235_235034

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235034


namespace tangency_points_concyclic_and_concentric_l235_235813

theorem tangency_points_concyclic_and_concentric 
    (O1 O2 : Point) 
    (R1 R2 a : ℝ) 
    (non_intersecting : a > R1 + R2) :
    ∃ (circle1 circle2 circle3 : Circle),
    (tangency_points circle1 ∧ tangency_points circle2 ∧ intersection_points circle3) ∧
    are_concentric circle1 circle2 circle3 :=
by 
  sorry

end tangency_points_concyclic_and_concentric_l235_235813


namespace sector_area_is_24pi_l235_235458

-- Definitions used in Lean 4 statement
def central_angle_deg := 240
def central_angle_rad := (4 * Real.pi) / 3
def radius := 6
def arc_length := central_angle_rad * radius
def sector_area := (1 / 2) * arc_length * radius

-- Lean 4 statement for the proof problem
theorem sector_area_is_24pi :
  sector_area = 24 * Real.pi :=
by
  sorry

end sector_area_is_24pi_l235_235458


namespace greatest_ab_sum_l235_235067

theorem greatest_ab_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) :
  a + b = Real.sqrt 220 ∨ a + b = -Real.sqrt 220 :=
sorry

end greatest_ab_sum_l235_235067


namespace monotonic_intervals_max_min_interval_l235_235147

def f (x : ℝ) : ℝ := (x - 2) * (x + 4)

theorem monotonic_intervals :
  (∀ x, x < -1 → ∃ y, y < x ∧ f y > f x) ∧
  (∀ x, -1 ≤ x → ∃ y, y > x ∧ f y > f x) :=
sorry

theorem max_min_interval :
  ∃ x y, x ∈ set.Icc (-2) 2 ∧ y ∈ set.Icc (-2) 2 ∧ (f x = -9) ∧ (f y = 0) :=
sorry

end monotonic_intervals_max_min_interval_l235_235147


namespace part1_part2_l235_235148

noncomputable def f (a b x : ℝ) := x^3 - a * x^2 + b * x + 1

theorem part1:
  (∀ (a b : ℝ), f a b 3 = -26 ∧ (derivative (f a b)) 3 = 0 → a = 3 ∧ b = -9) :=
by
  sorry

theorem part2:
  (∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → (f 3 (-9) x ≤ 6)) ∧ 
  ( ∃ x : ℝ, -4 ≤ x ∧ x ≤ 4 ∧ (f 3 (-9) x = 6)) :=
by
  sorry

end part1_part2_l235_235148


namespace binom_60_3_l235_235014

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235014


namespace incorrect_expression_l235_235927

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 3) : x / (y - x) ≠ 5 / 2 := 
by
  sorry

end incorrect_expression_l235_235927


namespace binom_60_3_eq_34220_l235_235048

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235048


namespace measure_segment_PB_l235_235189

variable (O : Type) [metric_space O]
variable (C : circle O)
variable (M A B P : O)
variable (MP AB AP PB : ℝ)
variable (x : ℝ)

-- Conditions:
-- M is the midpoint of arc CAB
-- MP is perpendicular to chord AB at P
-- The measure of chord AC is 2x
-- The measure of segment AP is 3x + 2
variable (h1 : is_midpoint_arc_CAB M A B)
variable (h2 : perpendicular MP AB)
variable (h3 : measure_chord AC = 2 * x)
variable (h4 : measure_segment AP = 3 * x + 2)
variable (h5 : AP = PB)

-- Proof statement:
theorem measure_segment_PB : measure_segment PB = 3 * x + 2 :=
by
  sorry

end measure_segment_PB_l235_235189


namespace equal_pairs_l235_235589

-- Definitions based on conditions
def cards := {n : ℕ | 1 ≤ n ∧ n ≤ 300}
def divisible_by_25 (a b : ℕ) := (a - b) % 25 = 0
def equally_split (A B : set ℕ) := A ∪ B = cards ∧ A ∩ B = ∅ ∧ A.card = 150 ∧ B.card = 150

-- Theorem statement
theorem equal_pairs (A B : set ℕ) (h : equally_split A B) :
  ∃ A_pairs B_pairs : set (ℕ × ℕ), 
    (∀ x ∈ A_pairs, divisible_by_25 x.1 x.2) ∧ 
    (∀ x ∈ B_pairs, divisible_by_25 x.1 x.2) ∧
    A_pairs.card = B_pairs.card :=
sorry

end equal_pairs_l235_235589


namespace interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l235_235429

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem interval_monotonic_increase (k : ℤ) :
  ∀ x : ℝ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi →
    ∃ I : Set ℝ, I = Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) ∧
      (∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I → x1 ≤ x2 → f x1 ≤ f x2) := sorry

theorem axis_of_symmetry (k : ℤ) :
  ∃ x : ℝ, x = Real.pi / 3 + k * (Real.pi / 2) := sorry

theorem max_and_min_values :
  ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      ((f x = 2 ∧ x = Real.pi / 3) ∨ (f x = -1 ∧ x = 0))) := sorry

end interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l235_235429


namespace cos_alpha_value_l235_235424

theorem cos_alpha_value (α : ℝ) (h1 : sin (α + π / 3) = -4 / 5) (h2 : -π / 2 < α) (h3 : α < 0) :
  cos α = (3 - 4 * real.sqrt 3) / 10 :=
by
  sorry

end cos_alpha_value_l235_235424


namespace nonagon_diagonals_l235_235845

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235845


namespace nancy_packs_l235_235236

theorem nancy_packs (total_bars packs_bars : ℕ) (h_total : total_bars = 30) (h_packs : packs_bars = 5) :
  total_bars / packs_bars = 6 :=
by
  sorry

end nancy_packs_l235_235236


namespace integral_solutions_count_l235_235819

theorem integral_solutions_count :
  (∃ (x y : ℤ), 3 * |x| + 5 * |y| = 100) → 
  (finset.univ.filter (λ (pair : ℤ × ℤ), 3 * |pair.1| + 5 * |pair.2| = 100)).card = 26 :=
by
  sorry

end integral_solutions_count_l235_235819


namespace nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l235_235469

theorem nat_no_solution_x3_plus_5y_eq_y3_plus_5x (x y : ℕ) (h₁ : x ≠ y) : 
  x^3 + 5 * y ≠ y^3 + 5 * x :=
sorry

theorem positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5 * y = y^3 + 5 * x :=
sorry

end nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l235_235469


namespace find_line_through_center_l235_235141

theorem find_line_through_center
  {P S : ℝ → ℝ → Prop}
  (hP : ∀ x y, P x y ↔ x^2 + y^2 = 2 * x)
  (hS : ∀ x y, S x y ↔ y^2 = 4 * x)
  (A B C D : ℝ × ℝ)
  (hAB : P A.1 A.2)
  (hBC : S B.1 B.2)
  (hCD : P C.1 C.2)
  (h_intersect : ∃ n : ℝ, ∀ y, 
    A = (n * y + 1, y) ∧ 
    B = (n * (y + 1) + 1, y + 1) ∧ 
    C = (n * (y + 2) + 1, y + 2) ∧ 
    D = (n * (y + 3) + 1, y + 3))
  (h_arith : ∃ m k l : ℝ, 
    k - m = l - k ∧
    ∃ y
      (hA : A = (n * y + 1, y))
      (hB : B = (n * (y + 1) + 1, y + 1))
      (hC : C = (n * (y + 2) + 1, y + 2))
      (hD : D = (n * (y + 3) + 1, y + 3))
      (h_m : m = distance A B)
      (h_k : k = distance B C)
      (h_l : l = distance C D)) :
  ∃ n : ℝ, n = (√2 / 2) ∨ n = -(√2 / 2) :=
sorry

end find_line_through_center_l235_235141


namespace diagonal_of_rectangular_prism_l235_235506

theorem diagonal_of_rectangular_prism (L W H : ℝ) (hL : L = 3) (hW : W = 4) (hH : H = 5) : 
  sqrt (L^2 + W^2 + H^2) = 5 * sqrt 2 :=
by
  -- The assumptions
  have hL := hL
  have hW := hW
  have hH := hH
  -- Substitute the assumptions in the required statement to bring it into the proved form
  rw [hL, hW, hH]
  sorry -- proof goes here

end diagonal_of_rectangular_prism_l235_235506


namespace grasshopper_jump_distance_l235_235623

-- Definitions based on conditions
def frog_jump : ℤ := 39
def higher_jump_distance : ℤ := 22
def grasshopper_jump : ℤ := frog_jump - higher_jump_distance

-- The statement we need to prove
theorem grasshopper_jump_distance :
  grasshopper_jump = 17 :=
by
  -- Here, proof would be provided but we skip with sorry
  sorry

end grasshopper_jump_distance_l235_235623


namespace maximum_value_of_g_l235_235749

def g : ℕ → ℕ
| n := if n < 15 then n + 15 else g (n - 7)

theorem maximum_value_of_g : ∃ m, ∀ n, g n ≤ 29 ∧ g m = 29 :=
by sorry

end maximum_value_of_g_l235_235749


namespace alex_buys_15_pounds_of_rice_l235_235612

theorem alex_buys_15_pounds_of_rice (r b : ℝ) 
  (h1 : r + b = 30)
  (h2 : 75 * r + 35 * b = 1650) : 
  r = 15.0 := sorry

end alex_buys_15_pounds_of_rice_l235_235612


namespace number_of_diagonals_in_nonagon_l235_235820

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235820


namespace all_numbers_are_2007_l235_235741

noncomputable def sequence_five_numbers (a b c d e : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ 
  (a = 2007 ∨ b = 2007 ∨ c = 2007 ∨ d = 2007 ∨ e = 2007) ∧ 
  (∃ r1, b = r1 * a ∧ c = r1 * b ∧ d = r1 * c ∧ e = r1 * d) ∧
  (∃ r2, a = r2 * b ∧ c = r2 * a ∧ d = r2 * c ∧ e = r2 * d) ∧
  (∃ r3, a = r3 * c ∧ b = r3 * a ∧ d = r3 * b ∧ e = r3 * d) ∧
  (∃ r4, a = r4 * d ∧ b = r4 * a ∧ c = r4 * b ∧ e = r4 * d) ∧
  (∃ r5, a = r5 * e ∧ b = r5 * a ∧ c = r5 * b ∧ d = r5 * c)

theorem all_numbers_are_2007 (a b c d e : ℤ) 
  (h : sequence_five_numbers a b c d e) : 
  a = 2007 ∧ b = 2007 ∧ c = 2007 ∧ d = 2007 ∧ e = 2007 :=
sorry

end all_numbers_are_2007_l235_235741


namespace find_c_find_h_max_l235_235510

-- Define the conditions
variables (A B C a b c h : ℝ) (l : ℝ)
noncomputable def conditions :=
  ∆ABC a b c A B C ∧
  (A = 5 * π / 12) ∧ 
  (b = 1) ∧ 
  (sqrt 3 * sin B - cos B = 1)

-- Define the proof for part I
theorem find_c (h : conditions A B C a b c) : c = sqrt 6 / 3 := sorry

-- Define the proof for part II
theorem find_h_max (h : conditions A B C a b c) : h = sqrt 3 / 2 := sorry

end find_c_find_h_max_l235_235510


namespace dot_product_necessity_l235_235171

variables (a b : ℝ → ℝ → ℝ)

def dot_product (a b : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ :=
  a x y * b x y

def angle_is_acute (a b : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  0 < a x y

theorem dot_product_necessity (a b : ℝ → ℝ → ℝ) (x y : ℝ) :
  dot_product a b x y > 0 ↔ angle_is_acute a b x y :=
sorry

end dot_product_necessity_l235_235171


namespace purely_imaginary_zero_real_part_l235_235432

open Complex

theorem purely_imaginary_zero_real_part (m : ℝ) (h : (\<sup>\frac{m(m+2)}{m-1}\<sup> + (m^2 + m - 2) • I).re = 0) : m = 0 := by
  sorry

end purely_imaginary_zero_real_part_l235_235432


namespace harriet_siblings_product_l235_235817
-- Import necessary libraries

-- Define the problem statement
theorem harriet_siblings_product (H_sisters : ℕ) (H_brothers : ℕ)
  (Harry_sisters : H_sisters = 4) (Harry_brothers : H_brothers = 6) :
  let S := H_sisters
  let B := H_brothers
  S * B = 24 := by
  -- We state the assumptions
  have Harry_sisters_equal_4 : H_sisters = 4 := Harry_sisters
  have Harry_brothers_equal_6 : H_brothers = 6 := Harry_brothers
  
  -- Substituting the values of H_sisters and H_brothers
  let S := 4
  let B := 6
  
  -- Compute the product
  calc S * B = 4 * 6 : by rfl
            ... = 24  : by norm_num

end harriet_siblings_product_l235_235817


namespace binom_60_3_l235_235019

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235019


namespace power_equality_l235_235497

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l235_235497


namespace split_square_into_equal_polygons_l235_235973

noncomputable def can_split_square (square : set ℝ) (pentagon1 pentagon2 : set ℝ) (hexagon1 hexagon2 : set ℝ) :=
  (∃ square : set ℝ, square = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}) ∧
  (∃ pentagon1 pentagon2 : set ℝ, (pentagon1 ∪ pentagon2 = square ∧ pentagon1 ∩ pentagon2 = ∅) ∧ area pentagon1 = area pentagon2) ∧
  (∃ hexagon1 hexagon2 : set ℝ, (hexagon1 ∪ hexagon2 = square ∧ hexagon1 ∩ hexagon2 = ∅) ∧ area hexagon1 = area hexagon2) ∧
  (concave pentagon1 ∧ concave pentagon2 ∧ concave hexagon1 ∧ concave hexagon2)

theorem split_square_into_equal_polygons : 
  ∃ square pentagon1 pentagon2 hexagon1 hexagon2, can_split_square square pentagon1 pentagon2 hexagon1 hexagon2 :=
sorry

end split_square_into_equal_polygons_l235_235973


namespace missing_fraction_l235_235076

theorem missing_fraction (x : ℕ) (h1 : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 9 :=
by
  sorry

end missing_fraction_l235_235076


namespace selling_price_of_cycle_l235_235697

theorem selling_price_of_cycle (cost_price : ℕ) (gain_percent : ℕ) (cost_price_eq : cost_price = 1500) (gain_percent_eq : gain_percent = 8) :
  ∃ selling_price : ℕ, selling_price = 1620 := 
by
  sorry

end selling_price_of_cycle_l235_235697


namespace acyclic_graph_l235_235730

-- Definitions to represent the conditions
def stripes : Type := sorry
def desk : Type := sorry
def parallel_to_side_of_desk : stripes → desk → Prop := sorry
def under : stripes → stripes → Prop := sorry

-- Digraph representation with conditions
def digraph (V : Type) := sorry
def vertices (G : digraph stripes) : set stripes := sorry
def edges (G : digraph stripes) : stripes → stripes → Prop := sorry
def complete_bipartite_graph (G : digraph stripes) : Prop := sorry

-- Prove that such a digraph is acyclic
theorem acyclic_graph {desk : Type} {stripes : Type} (G : digraph stripes) 
  (parallel_to_side_of_desk : stripes → desk → Prop)
  (complete_bipartite_graph : Prop)
  (under : stripes → stripes → Prop) 
  (condition : ∀ (s1 s2 s3 s4 : stripes), 
    parallel_to_side_of_desk s1 desk → parallel_to_side_of_desk s2 desk → 
    parallel_to_side_of_desk s3 desk → parallel_to_side_of_desk s4 desk → 
    ((under s1 s3 ∧ ¬under s1 s4) ∨ (under s2 s4 ∧ ¬under s2 s3) ∨
     (under s3 s1 ∧ ¬under s3 s2) ∨ (under s4 s2 ∧ ¬under s4 s1))) :
  ∀ cycles : list stripes, ¬ (∃ (n : ℕ), n ≥ 3 ∧ is_cycle G cycles) :=
sorry

end acyclic_graph_l235_235730


namespace train_length_l235_235719

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end train_length_l235_235719


namespace incorrect_statements_l235_235367

theorem incorrect_statements (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, f x ∈ set.Ioo -2 2) ∧
  (∀ x y, f (x + y) + f (x - y) = 2 * f x * g y) ∧
  f 0 = 0 ∧
  (∀ x ≠ 0, f x * g x ≠ 0) →
  [¬ ∀ x, f (x + 1) = x ^ 2 → f (real.exp 1) = (real.exp 1 - 1) ^ 2,
   ∀ x, f (x + 2) ∈ set.Ioo -2 2,
   ∃ x, ¬ ∃ n : ℕ, f x = 2 * n,
  ¬(∃ x, ∃ y, f (x + y) + f (x - y) = 2 * f x * g y)] :=
by {
  sorry
}

end incorrect_statements_l235_235367


namespace work_efficiency_ratio_l235_235347

theorem work_efficiency_ratio
  (A B : ℝ)
  (h1 : A + B = 1 / 18)
  (h2 : B = 1 / 27) :
  A / B = 1 / 2 := 
by
  sorry

end work_efficiency_ratio_l235_235347


namespace suzanna_bike_ride_total_distance_l235_235613

theorem suzanna_bike_ride_total_distance
  (time_uphill : ℕ)
  (time_flat : ℕ)
  (rate_uphill : ℝ)
  (rate_flat : ℝ)
  (total_time_uphill : ℕ := 20) -- Given Suzanna rides for 20 minutes uphill
  (total_time_flat : ℕ := 40)   -- Given Suzanna rides for 40 minutes on flat road)
  (rate_uphill := 1.5)          -- Given rate during 20 minutes uphill
  (rate_flat := 3)              -- Given rate after 20 minutes flat
  (segments_uphill := total_time_uphill / 10)   -- Number of 10-minute segments uphill
  (segments_flat := total_time_flat / 10)       -- Number of 10-minute segments flat
  (distance_uphill := segments_uphill * rate_uphill)
  (distance_flat := segments_flat * rate_flat)
  (total_distance := distance_uphill + distance_flat) : 
  total_distance = 15 := 
by {
  sorry -- Proof to be completed
}

end suzanna_bike_ride_total_distance_l235_235613


namespace value_of_a6_l235_235123

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
def a1 : Prop := a 1 = 1
def S5 : Prop := S 5 = 15
def arithmetic_sequence : Prop := ∀ n, a (n + 1) = a n + a 2 - a 1

-- Question rewritten as a proof problem
theorem value_of_a6 (h1 : a1) (h2 : S5) (h3 : arithmetic_sequence) : a 6 = 6 :=
by
  sorry

end value_of_a6_l235_235123


namespace percentage_students_entered_l235_235636

theorem percentage_students_entered (students_before : ℕ) (students_after : ℕ) :
  students_before = 28 → students_after = 58 → 0.4 * (students_after - students_before) = 12 :=
by
  intros h_before h_after
  rw [h_before, h_after]
  norm_num


end percentage_students_entered_l235_235636


namespace area_triangle_ACM_eq_trapezoid_ABCD_l235_235281

variables {A B C D M : Type} [Geometry A B C D M]
  (h : Trapezoid ABCD)
  (h1 : Line C parallel_to Diagonal BD)
  (h2 : Meets (extension_of_base AD) M)
  (h3 : Parallelogram BCMD)

theorem area_triangle_ACM_eq_trapezoid_ABCD :
  area (△ACM) = area (ABCD) := by
  sorry

end area_triangle_ACM_eq_trapezoid_ABCD_l235_235281


namespace committee_selection_l235_235956

   theorem committee_selection (total_people women : ℕ) : total_people = 15 ∧ women = 5 → 
     (∑ n in finset.range 3, n.choose 2 * (total_people - women).choose (4 - n)) + 
     women.choose 3 * (total_people - women).choose 1 + women.choose 4 = 555 :=
   by 
   sorry
   
end committee_selection_l235_235956


namespace sum_of_sixes_formula_l235_235419

def sum_of_sixes (n : ℕ) : ℕ :=
  ∑ k in Finset.range n, 6 * (10^k - 1) / 9

theorem sum_of_sixes_formula (n : ℕ) :
  sum_of_sixes n = 2 / 3 * ((10^(n+1) - 10) / 9 - n) :=
by
  sorry

end sum_of_sixes_formula_l235_235419


namespace find_angle_B_find_area_l235_235178

-- Given conditions:
variables {a b c : Real}
variables {B C : Real}
hypothesis (h1 : a^2 + 11 * b^2 = 2 * Real.sqrt 3 * a * b)
hypothesis (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B)

-- Prove that B = π/3
theorem find_angle_B (h1 : a^2 + 11 * b^2 = 2 * Real.sqrt 3 * a * b) 
                     (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B) : 
  B = Real.pi / 3 := 
begin
  sorry -- Proof steps are not required
end

-- Given condition for area when B = π/3
theorem find_area (h1 : a^2 + 11 * b^2 = 2 * Real.sqrt 3 * a * b) 
                  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B) 
                  (h3 : ∀ {ax bx cx}, ax • bx = Real.tan B) 
                  (hB : B = Real.pi / 3) : 
  1/2 * a * c * Real.sin B = 3/2 := 
begin
  sorry -- Proof steps are not required
end

end find_angle_B_find_area_l235_235178


namespace geometric_sequence_sum_inequality_l235_235131

open Classical

variable (a_1 q : ℝ) (h1 : a_1 > 0) (h2 : q > 0) (h3 : q ≠ 1)

theorem geometric_sequence_sum_inequality :
  a_1 + a_1 * q^3 > a_1 * q + a_1 * q^2 :=
by
  sorry

end geometric_sequence_sum_inequality_l235_235131


namespace volume_of_solid_rotation_l235_235735

noncomputable def volume_of_solid : ℝ :=
  let f1 := (λ x : ℝ, x^2)
  let f2 := (λ x : ℝ, x^4)
  π * (∫ x in 0..1, x) - π * (∫ x in 0..1, x^4)

theorem volume_of_solid_rotation :
  volume_of_solid = (3 * π / 10) :=
sorry

end volume_of_solid_rotation_l235_235735


namespace distinct_diagonals_in_convex_nonagon_l235_235884

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235884


namespace five_algorithmic_statements_l235_235408

-- Define the five types of algorithmic statements in programming languages
inductive AlgorithmicStatement : Type
| input : AlgorithmicStatement
| output : AlgorithmicStatement
| assignment : AlgorithmicStatement
| conditional : AlgorithmicStatement
| loop : AlgorithmicStatement

-- Theorem: Every programming language contains these five basic types of algorithmic statements
theorem five_algorithmic_statements : 
  ∃ (s : List AlgorithmicStatement), 
    (s.length = 5) ∧ 
    ∀ x, x ∈ s ↔
    x = AlgorithmicStatement.input ∨
    x = AlgorithmicStatement.output ∨
    x = AlgorithmicStatement.assignment ∨
    x = AlgorithmicStatement.conditional ∨
    x = AlgorithmicStatement.loop :=
by
  sorry

end five_algorithmic_statements_l235_235408


namespace tan_C_value_l235_235428

noncomputable def tan_C (a b c : ℝ) (h : a^2 + b^2 - c^2 = - (2/3) * a * b) : ℝ :=
  if 0 < (1 - ((-1/3)^2 : ℝ)) then -2 * real.sqrt(2) else 0

theorem tan_C_value (a b c : ℝ) (h : a^2 + b^2 - c^2 = - (2/3) * a * b) : 
  tan_C a b c h = -2 * real.sqrt(2) :=
sorry

end tan_C_value_l235_235428


namespace nonagon_diagonals_count_l235_235839

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235839


namespace length_of_each_train_is_62_5_l235_235670

noncomputable def length_of_each_train : ℝ :=
  let speed_faster := 46 * (1000 / 3600) in  -- converting km/hr to m/s
  let speed_slower := 36 * (1000 / 3600) in  -- converting km/hr to m/s
  let relative_speed := speed_faster - speed_slower in
  let time := 45 in -- time in seconds
  let distance_covered := relative_speed * time in
  distance_covered / 2

theorem length_of_each_train_is_62_5 :
  length_of_each_train = 62.5 :=
by
  -- the proof will be filled in here
  sorry

end length_of_each_train_is_62_5_l235_235670


namespace triangle_YZ_length_l235_235201

variable (X Y Z M : Type)
variable [Side : TriangleSides X Y Z]

def sideXY : Real := 6
def sideXZ : Real := 9
def medianXM : Real := 4

theorem triangle_YZ_length :
  ∃ YZ : Real, YZ = Real.sqrt(170) := by
    sorry

end triangle_YZ_length_l235_235201


namespace distinct_diagonals_convex_nonagon_l235_235850

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235850


namespace Laplace_odd_Laplace_limit_l235_235598

-- Define the Laplace function
def Laplace (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * (∫ t in 0..x, Real.exp (-t^2 / 2))

-- Prove that the Laplace function is odd
theorem Laplace_odd (x : ℝ) : Laplace (-x) = -Laplace (x) := sorry

-- Prove the limit of the Laplace function as x approaches +∞
theorem Laplace_limit : Tendsto Laplace at_top (𝓝 (1 / 2)) := sorry

end Laplace_odd_Laplace_limit_l235_235598


namespace part1_geom_seq_part2_sum_bn_part3_sum_cn_l235_235805

noncomputable def a : ℕ → ℝ
| 0     := 1/4
| (n+1) := a n / ((-1 : ℝ)^(n+1) * a n - 2)

noncomputable def b : ℕ → ℝ
| n := (3 * 2^n + 1)^2

noncomputable def c : ℕ → ℝ
| n := a n * Real.sin ((2 * n + 1) * Real.pi / 2)

noncomputable def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b (i+1)

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, c (i+1)

theorem part1_geom_seq : 
  ∃ (r : ℝ), (∀ n : ℕ, (1 / a n + (-1 : ℝ)^n = 3 * r^n)) →
  (1 / a 0 + (-1) = 3) ∧ (∀ n : ℕ, (1 / a (n+1) + (-1)^(n+1) = -2 * (1 / a n + (-1)^n))) := 
by sorry

theorem part2_sum_bn (n : ℕ) : 
  S n = 3 * 4^n + 6 * 2^n + n - 9 := 
by sorry

theorem part3_sum_cn (n : ℕ) (h : n > 0) : 
  T n < 4 / 7 := 
by sorry

end part1_geom_seq_part2_sum_bn_part3_sum_cn_l235_235805


namespace quadratic_complex_roots_l235_235288

noncomputable def hasComplexRoots (a b c : ℂ) : Prop :=
  ∃ x : ℂ, a * x^2 + b * x + c = 0 ∧ ∃ y : ℂ, y ≠ x ∧ a * y^2 + b * y + c = 0

theorem quadratic_complex_roots (λ : ℝ) : ¬(λ = 2) ↔ hasComplexRoots (1 - complex.i) (λ + complex.i) (1 + complex.i * λ) := by
  sorry

end quadratic_complex_roots_l235_235288


namespace binary_arithmetic_example_l235_235391

theorem binary_arithmetic_example :
  nat.binary_to_nat 1101 + nat.binary_to_nat 1011 - nat.binary_to_nat 101 + nat.binary_to_nat 111 = nat.binary_to_nat 11010 :=
by
  sorry

end binary_arithmetic_example_l235_235391


namespace parallel_lines_determine_planes_l235_235307

-- Definitions
def plane_determined_by (l1 l2 : Line) : Plane := sorry
def are_in_same_plane (l1 l2 l3 : Line) : Prop := sorry

-- Problem Statement
theorem parallel_lines_determine_planes (l1 l2 l3 : Line) 
  (h1 : are_parallel l1 l2) (h2 : are_parallel l2 l3) (h3 : are_parallel l1 l3) :
  ∃ n : ℕ, (n = 1 ∨ n = 3) ∧
  (n = 1 → are_in_same_plane l1 l2 l3) ∧
  (n = 3 → ¬ are_in_same_plane l1 l2 l3) := 
begin
  sorry
end

end parallel_lines_determine_planes_l235_235307


namespace probability_of_four_draws_l235_235181

noncomputable def probability_four_good_bulbs : ℚ := 
  let good_bulbs := 8
  let defective_bulbs := 2
  let total_bulbs := 10
  let probability_good_first_two := (8 / total_bulbs) * ((8 - 1) / (total_bulbs - 1))
  let probability_good_after_two_defective := 1 - probability_good_first_two - (8 / total_bulbs * (defective_bulbs / (total_bulbs - 1)) * (7 / (total_bulbs - 2))) - ((2 / total_bulbs) * (8 / (total_bulbs - 1)) * (7 / (total_bulbs - 2)))
  probability_good_after_two_defective

theorem probability_of_four_draws : 
  let ξ := 4
  P(ξ = ξ | good_bulbs, defective_bulbs, total_bulbs) = probability_four_good_bulbs := by
    sorry

end probability_of_four_draws_l235_235181


namespace solution_set_of_inequality_l235_235418

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 12 < 0 } = { x : ℝ | -4 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l235_235418


namespace min_area_of_triangle_l235_235212

noncomputable def area_of_triangle (A B C : ℝ × ℝ × ℝ) :=
  0.5 * (Real.sqrt (((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))^2 +
                    ((B.2 - A.2) * (C.3 - A.3) - (B.3 - A.3) * (C.2 - A.2))^2 +
                    ((B.3 - A.3) * (C.1 - A.1) - (B.1 - A.1) * (C.3 - A.3))^2))

theorem min_area_of_triangle : ∃ s : ℝ, area_of_triangle (-1, 1, 0) (1, 3, 2) (s, 1, 1) = 1 :=
begin
  use 0,
  simp [area_of_triangle],
  sorry
end

end min_area_of_triangle_l235_235212


namespace base8_addition_l235_235734

theorem base8_addition :
  ∀ (a b c : ℕ), nat_digits 8 a = [6, 5, 3] ∧ nat_digits 8 b = [2, 7, 4] ∧ nat_digits 8 c = [1, 6, 7] →
  nat_digits 8 (a + b + c) = [1, 3, 5, 6] :=
by
  intros a b c h
  rcases h with ⟨ha, hb, hc⟩
  sorry

end base8_addition_l235_235734


namespace union_of_A_and_B_l235_235936

-- Define set A
def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define set B
def B := {x : ℝ | x < 1}

-- The proof problem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} :=
by sorry

end union_of_A_and_B_l235_235936


namespace isosceles_triangle_count_l235_235196

theorem isosceles_triangle_count :
  let points := [(i, j) | i <- [1, 2, 3, 4, 5, 6, 7], j <- [1, 2, 3, 4, 5, 6, 7], 
                         not ((i = 2 ∧ j = 2) ∨ (i = 5 ∧ j = 2))] in
  let AB := ((2, 2), (5, 2)) in
  let midpoint_AB := (3.5, 2) in
  (count (λ C, isosceles (AB.1) (AB.2) C) points) = 12 :=
begin
  -- Proof goes here
  sorry
end

end isosceles_triangle_count_l235_235196


namespace binom_60_3_l235_235012

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235012


namespace linear_regression_coefficient_l235_235151

theorem linear_regression_coefficient :
  let x_values := [6, 8, 10, 12]
  let y_values := [6, 5, 3, 2]
  let n := 4
  let x_mean := (x_values.sum / n.toFloat)
  let y_mean := (y_values.sum / n.toFloat)
  y_mean = x_mean * (-0.7) + 10.3 := by
    sorry

end linear_regression_coefficient_l235_235151


namespace least_four_digit_number_l235_235649

def is_prime_digit (d: ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def all_different_digits (n: ℕ) : Prop :=
  let ds := Int.toStrings (Int.digits 10 n) in
  ds = List.eraseDups ds

def is_divisible_by_digits (n: ℕ) : Prop :=
  let ds := Int.digits 10 n in
  ∀ d ∈ ds, d ≠ 0 → n % d = 0

def contains_prime_digit (n: ℕ) : Prop :=
  ∃ d ∈ Int.digits 10 n, is_prime_digit d

theorem least_four_digit_number :
  ∃ n: ℕ, 1000 ≤ n ∧ n < 10000 ∧ all_different_digits n ∧ contains_prime_digit n ∧ is_divisible_by_digits n ∧ n = 1236 := 
begin
  sorry
end

end least_four_digit_number_l235_235649


namespace trajectory_is_one_branch_of_hyperbola_l235_235198

variable (P F1 F2 : Point) -- Define points P, F1, and F2
variable (d1 d2 : ℝ) -- Define distances d1 and d2
variable (h1 : abs (dist P F1 - dist P F2) = 6) -- Condition: |PF1 - PF2| = 6
variable (h2 : dist F1 F2 = 8) -- Condition: |F1F2| = 8

theorem trajectory_is_one_branch_of_hyperbola (P F1 F2 : Point) (h1 : abs (dist P F1 - dist P F2) = 6) (h2 : dist F1 F2 = 8) :
  (trajectory P F1 F2) = hyperbola_branch :=
sorry

end trajectory_is_one_branch_of_hyperbola_l235_235198


namespace main_roads_exist_l235_235948

structure City := (id : Nat)

structure Road := (city1 city2 : City)

structure Country := (cities : List City) (roads : List Road)

def connected (c : Country) : Prop :=
  -- Assumption: roads form a connected graph
  sorry

def has_odd_degree_main_roads (c : Country) (main_roads : List Road) : Prop :=
  ∀ city ∈ c.cities, 
    let degree := c.roads.count (λ r => r.city1 = city ∨ r.city2 = city)
    degree % 2 = 1

theorem main_roads_exist (c : Country) (h1 : c.cities.length = 100) (h2 : connected c) :
  ∃ main_roads : List Road, has_odd_degree_main_roads c main_roads :=
sorry

end main_roads_exist_l235_235948


namespace distinct_diagonals_in_convex_nonagon_l235_235886

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235886


namespace greatest_integer_value_l235_235087

theorem greatest_integer_value (x : ℤ) : 3 * |x - 2| + 9 ≤ 24 → x ≤ 7 :=
by sorry

end greatest_integer_value_l235_235087


namespace visible_unit_cubes_l235_235682

theorem visible_unit_cubes (n : ℕ) (h : n = 12) : 
  let total_cubes := n * n * n,
      visible_through_one_face := n * n,
      visible_surface_on_two_faces := 2 * n * (n - (n - 2)),
      double_counted_edges := 3 * n,
      visible_cubes := visible_through_one_face + visible_surface_on_two_faces - double_counted_edges + 1 in
  visible_cubes = 181 :=
by
  sorry

end visible_unit_cubes_l235_235682


namespace max_value_expression_l235_235997

theorem max_value_expression (x y : ℝ) : 
  (2 * x + real.sqrt 2 * y) / (2 * x ^ 4 + 4 * y ^ 4 + 9) ≤ 1 / 4 := sorry

end max_value_expression_l235_235997


namespace binomial_60_3_eq_34220_l235_235038

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235038


namespace coach_votes_l235_235289

theorem coach_votes :
  ∀ (total_rowers : ℕ) (votes_per_rower : ℕ) (total_coaches : ℕ)
    (total_votes : ℕ)
    (votes_per_coach : ℕ),
    total_rowers = 60 → votes_per_rower = 3 →
    total_coaches = 36 →
    total_votes = total_rowers * votes_per_rower →
    votes_per_coach = total_votes / total_coaches →
    votes_per_coach = 5 :=
by
  intros total_rowers votes_per_rower total_coaches total_votes votes_per_coach  
  intros rowers_eq votes_per_rower_eq coaches_eq total_votes_eq votes_per_coach_eq
  rw [rowers_eq, votes_per_rower_eq, coaches_eq] at *
  simp only [mul_comm] at total_votes_eq
  rw [total_votes_eq]
  exact votes_per_coach_eq

end coach_votes_l235_235289


namespace least_positive_difference_between_A_and_B_l235_235250

-- Definitions based on the conditions
def sequenceA : List ℕ := [3, 9, 27, 81, 243]
def sequenceB : List ℕ := [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]

-- The proof problem statement
theorem least_positive_difference_between_A_and_B :
  ∃ (d : ℕ), (d > 0) ∧ (∀ (a ∈ sequenceA) (b ∈ sequenceB), abs (a - b) ≠ d) ∧ d = 3 := 
sorry

end least_positive_difference_between_A_and_B_l235_235250


namespace num_diagonals_convex_nonagon_l235_235875

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235875


namespace part1_part2_l235_235127

section

-- Definitions of the given functions
def f (x : ℝ) := x * Real.cos x
def g (x : ℝ) (a : ℝ) := a * Real.sin x

-- Part (1) statement
theorem part1 (a : ℝ) (h1 : a = 1) (x : ℝ) (hx : 0 < x ∧ x < Real.pi) :
  x > g x a ∧ g x a > f x := sorry

-- Part (2) statement
theorem part2 (x : ℝ) (hx : -Real.pi < x ∧ x < 0 ∨ 0 < x ∧ x < Real.pi)
  (h2 : f x / g x a < Real.sin x / x) : 
  1 ≤ a := sorry

end

end part1_part2_l235_235127


namespace solve_inequality_l235_235097

theorem solve_inequality (x : ℝ) : (|x - 3| + |x - 5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solve_inequality_l235_235097


namespace find_relation_l235_235124

noncomputable theory

def ellipse_eqn (x y : ℝ) : Prop :=
  (x^2 / 3 + y^2 = 1)

def slopes_relation (m n : ℝ) : Prop :=
  3 * n = 2 * m

theorem find_relation (m n : ℝ) (h₀ : m ≠ 3) 
    (h₁ : ellipse_eqn 1 (sqrt (6) / 3)) (h₂ : ellipse_eqn 1 (- (sqrt (6) / 3))) 
    (k₁ k₂ k₃ : ℝ) (hk : k₁ + k₃ = 3 * k₂) (hk₂ : k₂ = 2 / 3) :
    slopes_relation m n :=
sorry

end find_relation_l235_235124


namespace distinct_diagonals_convex_nonagon_l235_235858

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235858


namespace angle_is_right_l235_235570

-- Define the vectors p, q, r
def p : ℝ × ℝ × ℝ := (2, -3, 4)
def q : ℝ × ℝ × ℝ := (-1, 5, -2)
def r : ℝ × ℝ × ℝ := (8, -1, 6)

-- Define the dot product for 3D vectors
def dot (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Define the scalar multiplication of a vector
def smul (k : ℝ) (a : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * a.1, k * a.2, k * a.3)

-- Define vector subtraction
def vsub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

-- Construct the vector (\mathbf{p} \cdot \mathbf{r}) \mathbf{q} - (\mathbf{p} \cdot \mathbf{q}) \mathbf{r}
def vec : ℝ × ℝ × ℝ :=
  vsub (smul (dot p r) q) (smul (dot p q) r)

-- The theorem to prove the angle between p and the resulting vector vec is 90 degrees
theorem angle_is_right : dot p vec = 0 := by
  sorry

end angle_is_right_l235_235570


namespace main_roads_exist_l235_235949

structure City := (id : Nat)

structure Road := (city1 city2 : City)

structure Country := (cities : List City) (roads : List Road)

def connected (c : Country) : Prop :=
  -- Assumption: roads form a connected graph
  sorry

def has_odd_degree_main_roads (c : Country) (main_roads : List Road) : Prop :=
  ∀ city ∈ c.cities, 
    let degree := c.roads.count (λ r => r.city1 = city ∨ r.city2 = city)
    degree % 2 = 1

theorem main_roads_exist (c : Country) (h1 : c.cities.length = 100) (h2 : connected c) :
  ∃ main_roads : List Road, has_odd_degree_main_roads c main_roads :=
sorry

end main_roads_exist_l235_235949


namespace find_k_equals_one_l235_235453

variables (a b : V)
variables [inner_product_space ℝ V] [normed_group V] [normed_space ℝ V]

variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (h₁ : (a + b) ⬝ (k • a - b) = 0) (hnb : a ≠ b)

theorem find_k_equals_one (k : ℝ) : k = 1 :=
sorry

end find_k_equals_one_l235_235453


namespace cost_of_chicken_katsu_l235_235614

open Real

variables (C : ℝ)
variables (smoky_salmon black_burger total_bill : ℝ)

-- The price of the smoky salmon
def smoky_salmon := 40

-- The price of the black burger
def black_burger := 15

-- The total bill Mr. Arevalo paid
def total_bill := 92

-- The total cost before charges
def total_cost_before_charges := smoky_salmon + black_burger + C

-- Service charge is 10% of the total cost before charges
def service_charge := 0.10 * total_cost_before_charges

-- Tip is 5% of the total cost before charges
def tip := 0.05 * total_cost_before_charges

-- Total bill is the sum of the total cost before charges, the service charge, and the tip
def calculated_total_bill := total_cost_before_charges + service_charge + tip

theorem cost_of_chicken_katsu : 
  calculated_total_bill = 92 → C = 25 :=
begin
  sorry
end

end cost_of_chicken_katsu_l235_235614


namespace even_and_monotonically_decreasing_l235_235368

noncomputable def f1 (x : ℝ) : ℝ := 1 / |x|
noncomputable def f2 (x : ℝ) : ℝ := (1 / 3) ^ x
noncomputable def f3 (x : ℝ) : ℝ := x ^ 2 + 1
noncomputable def f4 (x : ℝ) : ℝ := Real.log (|x|)

theorem even_and_monotonically_decreasing (h : (0 : ℝ) < ×) : 
  (∀ f x, f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4 → 
    (∀ x, f (- x) = f x) →
      is_decreasing_on f (λ x, 0 < x) → 
      f = f1) := by sorry

end even_and_monotonically_decreasing_l235_235368


namespace radius_of_circumcircle_of_right_triangle_l235_235650

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def circumcircle_radius (a b c : ℝ) [Decidable (a^2 + b^2 = c^2)] :  ℝ :=
  if h : a^2 + b^2 = c^2 then c / 2 else 0

theorem radius_of_circumcircle_of_right_triangle : 
  is_right_triangle 6 8 10 → circumcircle_radius 6 8 10 = 5 :=
by
  intros h
  simp [is_right_triangle, circumcircle_radius, h]
  sorry

end radius_of_circumcircle_of_right_triangle_l235_235650


namespace triangle_angle_sum_l235_235564

variables {α : Type*} [Euclidean_space α]

structure Triangle (α : Type*) [Euclidean_space α] :=
(A B C P O : α)
(acute : ∀ {A B C : α}, A ≠ B ∧ B ≠ C ∧ C ≠ A → 
  angle A B C < 90 ∧ angle B C A < 90 ∧ angle C A B < 90)
(altitude_foot : ∀ {A P BC : α}, is_foot_of_altitude A P BC)
(circumcenter : ∀ {A B C O : α}, is_circumcenter_of O A B C)
(angle_inequality : ∀ {A B C : α}, angle C A B ≥ angle A B C + 30)

theorem triangle_angle_sum
  (T : Triangle α) :
  ∠(T.A) (T.B) (T.C) + ∠(T.C) (T.O) (T.P) < 90 :=
sorry

end triangle_angle_sum_l235_235564


namespace ratio_KM_BD_l235_235200

def trapezoid (A B C D K M : Type) := Prop
def midpoint (P Q R : Type) := Prop
def equilateral (P Q R S: Type) := Prop

theorem ratio_KM_BD {A B C D K M : Type}
  (h1 : trapezoid A B C D)
  (h2 : midpoint K D A)
  (h3 : midpoint M B C)
  (h4 : A = B)
  : KM / BD = 1 / 2 :=
by 
  sorry

end ratio_KM_BD_l235_235200


namespace tan_sum_l235_235167

theorem tan_sum (x y : ℝ) 
  (h1 : sin x + sin y = 33 / 65) 
  (h2 : cos x + cos y = 56 / 65) :
  tan x + tan y = 33 / 28 := 
sorry

end tan_sum_l235_235167


namespace sum_of_solutions_l235_235991

def g (x : ℝ) : ℝ := 20 * x + 4

theorem sum_of_solutions :
  let g_inv := (g⁻¹ : ℝ → ℝ) in
  let eq := ∀ x, g_inv x = g ((3 * x)⁻¹) in
  -- Solution for x from the above equation
  let roots := {x | g_inv x = g ((3 * x)⁻¹)} in
  ∑ x in roots.to_finset, x = 92 / 3 :=
begin
  sorry
end

end sum_of_solutions_l235_235991


namespace square_cookie_cutters_count_l235_235075

def triangles_sides : ℕ := 6 * 3
def hexagons_sides : ℕ := 2 * 6
def total_sides : ℕ := 46
def sides_from_squares (S : ℕ) : ℕ := S * 4

theorem square_cookie_cutters_count (S : ℕ) :
  triangles_sides + hexagons_sides + sides_from_squares S = total_sides → S = 4 :=
by
  sorry

end square_cookie_cutters_count_l235_235075


namespace g_properties_l235_235218

theorem g_properties {g : ℝ → ℝ}
  (h : ∀ x y : ℝ, g ((x - y)^2 + 1) = g x^2 - 2 * x * g y + (y + 1)^2) :
  ∃ m t : ℝ, (m = 2 ∧ t = 3 ∧ m * t = 6) :=
begin
  sorry
end

end g_properties_l235_235218


namespace area_of_triangle_PQR_l235_235382

-- Define the coordinates of points P, Q, and R
def P := (-3, 4)
def Q := (1, 7)
def R := (5, -3)

-- Define a function to compute the area of a triangle given its vertices
def triangle_area (A B C : ℤ × ℤ) : ℚ :=
  let x1 := A.1; let y1 := A.2;
  let x2 := B.1; let y2 := B.2;
  let x3 := C.1; let y3 := C.2 in
  (1 / 2 : ℚ) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- State the theorem
theorem area_of_triangle_PQR : triangle_area P Q R = 26 := by
  sorry

end area_of_triangle_PQR_l235_235382


namespace four_digit_even_between_odds_count_l235_235315

theorem four_digit_even_between_odds_count :
  ∃ n : ℕ, n = 8 ∧ (∀ (num : List ℕ), 
    num.perm.mk [1, 2, 3, 4] ∧ 
    num.nodup ∧ 
    (∃ i : ℕ, (num.nth i = some 1 ∨ num.nth i = some 3) ∧ 
               (num.nth (i + 1) = some 2 ∨ num.nth (i + 1) = some 4) ∧ 
               (num.nth (i + 2) = some 1 ∨ num.nth (i + 2) = some 3))) :=
begin
  sorry
end

end four_digit_even_between_odds_count_l235_235315


namespace compare_magnitudes_l235_235066

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem compare_magnitudes :
  let a := (0.6 : ℝ)
  let b := (5 : ℝ)
  1 < b^{0.6} ∧ a^5 ∈ (0, 1) ∧ log_base a b < 0 →
  log_base a b < a^5 ∧ a^5 < b^{0.6} :=
by 
  intros a b h
  sorry

end compare_magnitudes_l235_235066


namespace base3_to_decimal_11111_l235_235746

def convert_base3_to_decimal (n : ℕ) : ℕ :=
  let digits := [1, 1, 1, 1, 1]
  let bases := [3^4, 3^3, 3^2, 3^1, 3^0]
  (List.zip digits bases).sum (λ ⟨d, b⟩ => d * b)

theorem base3_to_decimal_11111 : convert_base3_to_decimal 11111 = 121 :=
by
  sorry

end base3_to_decimal_11111_l235_235746


namespace binom_60_3_eq_34220_l235_235045

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235045


namespace value_of_a_7_l235_235463

theorem value_of_a_7 (a : ℕ → ℝ) (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h_roots : (Polynomial.C 9 + Polynomial.C 2016 * Polynomial.X + Polynomial.X^2).roots = {a 5, a 9}) :
  a 7 = -3 :=
by sorry

end value_of_a_7_l235_235463


namespace num_diagonals_convex_nonagon_l235_235878

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235878


namespace height_ratio_of_tetrahedron_tangency_l235_235726

noncomputable def height_ratio (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  -- This would be formally defined based on the geometric properties and calculations
  sorry

theorem height_ratio_of_tetrahedron_tangency
  (A B C D : ℝ × ℝ × ℝ)
  (sphere_exists : ∃ (S : ℝ × ℝ × ℝ) (r : ℝ), ∀ (pt : ℝ × ℝ × ℝ), (pt = A ∨ pt = B ∨ pt = C ∨ pt = D) → 
                    sqrt ((pt.1 - S.1)^2 + (pt.2 - S.2)^2 + (pt.3 - S.3)^2) = r)
  (equal_midpoint_segments : let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2),
                                 M2 := ((C.1 + D.1) / 2, (C.2 + D.2) / 2, (C.3 + D.3) / 2),
                                 M3 := ((A.1 + C.1) / 2, (A.2 + C.2) / 2, (A.3 + C.3) / 2),
                                 M4 := ((B.1 + D.1) / 2, (B.2 + D.2) / 2, (B.3 + D.3) / 2),
                                 M5 := ((A.1 + D.1) / 2, (A.2 + D.2) / 2, (A.3 + D.3) / 2),
                                 M6 := ((B.1 + C.1) / 2, (B.2 + C.2) / 2, (B.3 + C.3) / 2)
                             in
                             dist M1 M2 = dist M3 M4 ∧ dist M3 M4 = dist M5 M6 ∧ dist M1 M2 = dist M5 M6)
  (angle_ABC : ∠ (A - B) (C - B) = 100)
  : height_ratio A B C D = sqrt 3 * tan (50) :=
sorry

end height_ratio_of_tetrahedron_tangency_l235_235726


namespace number_of_integers_l235_235921

theorem number_of_integers (n : ℤ) : 
    (100 < n ∧ n < 300) ∧ (n % 7 = n % 9) → 
    (∃ count: ℕ, count = 21) := by
  sorry

end number_of_integers_l235_235921


namespace pet_store_initial_puppies_l235_235357

def initially_had_puppies (sold: ℕ) (puppies_per_cage: ℕ) (number_of_cages: ℕ) : ℕ :=
  sold + puppies_per_cage * number_of_cages

theorem pet_store_initial_puppies (sold: ℕ) (puppies_per_cage: ℕ) (number_of_cages: ℕ)
  (h_sold: sold = 3)
  (h_per_cage: puppies_per_cage = 5)
  (h_cages: number_of_cages = 3) :
  initially_had_puppies sold puppies_per_cage number_of_cages = 18 :=
by
  rw [h_sold, h_per_cage, h_cages]
  simp [initially_had_puppies]
  norm_num
  done


end pet_store_initial_puppies_l235_235357


namespace zero_of_h_in_interval_l235_235228

def f (x : ℝ) : ℝ := 0.8 ^ x - 1
def g (x : ℝ) : ℝ := Real.log x
def h (x : ℝ) : ℝ := f x - g x

theorem zero_of_h_in_interval : ∃ x ∈ Ioo 0 1, h x = 0 := by
  sorry

end zero_of_h_in_interval_l235_235228


namespace length_AM_l235_235104

def length_AB := 12 -- Given the length of AB is 12 cm

def is_midpoint (A B M : Point) : Prop :=
  distance A M = distance M B

theorem length_AM 
  (A B M : Point)
  (h1 : distance A B = 12)
  (h2 : is_midpoint A B M) : distance A M = 6 :=
by sorry

end length_AM_l235_235104


namespace find_x1_l235_235130

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
    (h5 : (1 - x1)^3 + (x1 - x2)^3 + (x2 - x3)^3 + x3^3 = 1 / 8) : x1 = 3 / 4 := 
by 
  sorry

end find_x1_l235_235130


namespace calculate_product_l235_235957

variable (EF FG GH HE : ℚ)
variable (x y : ℚ)

-- Conditions
axiom h1 : EF = 110
axiom h2 : FG = 16 * y^3
axiom h3 : GH = 6 * x + 2
axiom h4 : HE = 64
-- Parallelogram properties
axiom h5 : EF = GH
axiom h6 : FG = HE

theorem calculate_product (EF FG GH HE : ℚ) (x y : ℚ)
  (h1 : EF = 110) (h2 : FG = 16 * y ^ 3) (h3 : GH = 6 * x + 2) (h4 : HE = 64) (h5 : EF = GH) (h6 : FG = HE) :
  x * y = 18 * (4) ^ (1/3) := by
  sorry

end calculate_product_l235_235957


namespace cos_product_identity_l235_235602

open Real

theorem cos_product_identity :
  (∏ k in Finset.range 24, cos (2^k * (56 * (π / 180)))) = 1 / 2^24 :=
by
  sorry

end cos_product_identity_l235_235602


namespace typing_time_together_l235_235668

theorem typing_time_together 
  (jonathan_time : ℝ)
  (susan_time : ℝ)
  (jack_time : ℝ)
  (document_pages : ℝ)
  (combined_time : ℝ) :
  jonathan_time = 40 →
  susan_time = 30 →
  jack_time = 24 →
  document_pages = 10 →
  combined_time = document_pages / ((document_pages / jonathan_time) + (document_pages / susan_time) + (document_pages / jack_time)) →
  combined_time = 10 :=
by sorry

end typing_time_together_l235_235668


namespace Jack_has_18_dimes_l235_235266

theorem Jack_has_18_dimes :
  ∃ d q : ℕ, (d = q + 3 ∧ 10 * d + 25 * q = 555) ∧ d = 18 :=
by
  sorry

end Jack_has_18_dimes_l235_235266


namespace find_d_value_l235_235126

theorem find_d_value 
  (x y d : ℝ)
  (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = 49^x * d^y)
  (h2 : x + y = 4) :
  d = 27 :=
by 
  sorry

end find_d_value_l235_235126


namespace probability_math_majors_consecutive_l235_235640

theorem probability_math_majors_consecutive :
  let total_people := 12
  let math_majors := 5
  let physics_majors := 4
  let biology_majors := 3
  let total_ways := choose 11 4 * fact 4
  let consecutive_ways := 12
  probability (total_ways : ℚ) (consecutive_ways : ℚ) = (1 / 660 : ℚ) :=
by
  let total_people := 12
  let math_majors := 5
  let physics_majors := 4
  let biology_majors := 3
  let total_ways := choose 11 4 * fact 4
  let consecutive_ways := 12
  have total_ways_nonzero : (total_ways : ℚ) ≠ 0 := by sorry
  have h : probability (total_ways : ℚ) (consecutive_ways : ℚ) = consecutive_ways / total_ways := by sorry
  rw [h]
  norm_num
  sorry  -- Justify the final step and calculation

end probability_math_majors_consecutive_l235_235640


namespace sum_f_2014_l235_235051

noncomputable def f : ℝ → ℝ := sorry

lemma f_props (x : ℝ) : 
  (∀ x, f(x) = -f(x + 3/2)) ∧ 
  f(-1) = 1 ∧ 
  f(0) = -2 := sorry

theorem sum_f_2014 : 
  f(1) + f(2) + ∑ i in Ico 3 2014, f(i) = 1 :=
by
  sorry

end sum_f_2014_l235_235051


namespace quadratic_equation_m_l235_235505

theorem quadratic_equation_m (m b : ℝ) (h : (m - 2) * x ^ |m| - b * x - 1 = 0) : m = -2 :=
by
  sorry

end quadratic_equation_m_l235_235505


namespace problem1_correct_problem2_correct_l235_235736

noncomputable def problem1 : ℝ :=
  let sin_30 := 1 / 2
  let cos_45 := sqrt 2 / 2
  let sin_60 := sqrt 3 / 2
  let tan_45 := 1
  sin_30 ^ 2 + cos_45 ^ 2 + sqrt 2 * sin_60 * tan_45

theorem problem1_correct : problem1 = (3 / 4 + sqrt 6 / 2) :=
by
  sorry

noncomputable def problem2 : ℝ :=
  let sin_45 := sqrt 2 / 2
  let tan_60 := sqrt 3
  let cos_30 := sqrt 3 / 2
  sqrt ((sin_45 - 1 / 2) ^ 2) - abs (tan_60 - cos_30)

theorem problem2_correct : problem2 = (sqrt 2 / 2 - 1 / 2 - sqrt 3 / 2) :=
by
  sorry

end problem1_correct_problem2_correct_l235_235736


namespace minimum_value_of_eccentricity_sum_l235_235142

variable {a b m n c : ℝ} (ha : a > b) (hb : b > 0) (hm : m > 0) (hn : n > 0)
variable {e1 e2 : ℝ}

theorem minimum_value_of_eccentricity_sum 
  (h_equiv : a^2 + m^2 = 2 * c^2) 
  (e1_def : e1 = c / a) 
  (e2_def : e2 = c / m) : 
  (2 * e1^2 + (e2^2) / 2) = (9 / 4) :=
sorry

end minimum_value_of_eccentricity_sum_l235_235142


namespace draw_probability_l235_235265

-- Definition of the problem conditions
def stamps : set (string) := {"Spring Begins", "Summer Begins", "Autumn Equinox", "Great Cold"}
def draw_two (s : set string) : set (set string) :=
  {t | set.card t = 2 ∧ t ⊆ s}

-- Problem statement: Prove the probability of drawing "Summer Begins" and "Autumn Equinox" is 1/6
theorem draw_probability : 
  ∃ n d : ℕ, n / d = 1 / 6 ∧ 
    let outcomes := draw_two stamps in
    let favorable := {(x, y) | x = "Summer Begins" ∧ y = "Autumn Equinox" ∨
                                x = "Autumn Equinox" ∧ y = "Summer Begins"} in
    set.card favorable / set.card outcomes = n / d :=
  sorry

end draw_probability_l235_235265


namespace rectangle_area_l235_235267

theorem rectangle_area (x : ℤ) :
  let length := 5 * x + 3 in
  let width := x - 7 in
  (length * width) = 5 * x^2 - 32 * x - 21 :=
by
  sorry

end rectangle_area_l235_235267


namespace distinct_diagonals_nonagon_l235_235863

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235863


namespace distinct_diagonals_convex_nonagon_l235_235853

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235853


namespace binary_division_correct_l235_235409

def b1100101 := 0b1100101
def b1101 := 0b1101
def b101 := 0b101
def expected_result := 0b11111010

theorem binary_division_correct : ((b1100101 * b1101) / b101) = expected_result :=
by {
  sorry
}

end binary_division_correct_l235_235409


namespace number_of_diagonals_in_nonagon_l235_235827

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235827


namespace thirtieth_triangular_number_is_465_l235_235326

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem thirtieth_triangular_number_is_465 : triangular_number 30 = 465 :=
by
  sorry

end thirtieth_triangular_number_is_465_l235_235326


namespace num_diagonals_convex_nonagon_l235_235874

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235874


namespace ben_initial_marbles_l235_235733

theorem ben_initial_marbles (B : ℕ) (John_initial_marbles : ℕ) (H1 : John_initial_marbles = 17) (H2 : John_initial_marbles + B / 2 = B / 2 + B / 2 + 17) : B = 34 := by
  sorry

end ben_initial_marbles_l235_235733


namespace circle_properties_and_distances_l235_235203

-- Given condition: Circle A passes through point P = (sqrt(2), sqrt(2))
def point_P : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)

-- Circle A is symmetric to circle B: (x + 2)^2 + (y - 2)^2 = r^2 with respect to the line x - y + 2 = 0
def circle_B_eq (x y : ℝ) (r : ℝ) : Prop :=
  (x + 2)^2 + (y - 2)^2 = r^2

def symmetry_line_eq (x y : ℝ) : Prop :=
  x - y + 2 = 0

/--
Conditions:
1. The circle A is symmetric to the circle B: (x + 2)^2 + (y - 2)^2 = 4 with respect to the line x - y + 2 = 0.
2. Circle A passes through the point P = (sqrt 2, sqrt 2).

Prove:
1. Circle A has the equation x^2 + y^2 = 4.
2. The length of the common chord of the two circles is 2 sqrt 2.
3. For any point Q = (x_0, y_0) on the plane, there exists a fixed point M such that the distance from Q to M is constant and find this constant value.
-/
theorem circle_properties_and_distances:
  (∀ (x y : ℝ), circle_B_eq x y 2) ∧ symmetry_line_eq 1 1 ∧ ∃ (P : ℝ × ℝ), P = point_P
  → (∀ (x y : ℝ), x^2 + y^2 = 4) ∧ length_common_chord 2 sqrt(2) ∧ (∃ (M : ℝ × ℝ), M = (2/3, -2/3) ∧ distance_to_M_const 2 sqrt (17) / 3) := 
by
  sorry

end circle_properties_and_distances_l235_235203


namespace tetrahedron_circumscribed_sphere_surface_area_l235_235968

-- Define the lengths PA, PB, BC, AC
def PA : ℝ := 5
def PB : ℝ := 5
def BC : ℝ := 5
def AC : ℝ := 5

-- Define the lengths PC and AB
def PC : ℝ := 4 * Real.sqrt 2
def AB : ℝ := 4 * Real.sqrt 2

-- Surface area of the circumscribed sphere of tetrahedron P-ABC
def surface_area_of_circumscribed_sphere (PA PB BC AC PC AB : ℝ) : ℝ :=
  4 * Real.pi * (PA^2 / 2 + (AB / (2 * Real.sqrt 2))^2 + BC^2 / 2)

theorem tetrahedron_circumscribed_sphere_surface_area :
  PA = 5 → PB = 5 → BC = 5 → AC = 5 → PC = 4 * Real.sqrt 2 → AB = 4 * Real.sqrt 2 →
  surface_area_of_circumscribed_sphere PA PB BC AC PC AB = 41 * Real.pi :=
by
  intros
  sorry

end tetrahedron_circumscribed_sphere_surface_area_l235_235968


namespace find_star_value_l235_235154

noncomputable def star (a b : ℝ) : ℝ := sorry

axiom star_property1 (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : (a * b) ∎ b = a * (b ∎ b)
axiom star_property2 (a : ℝ) (h : 0 < a) : (a ∎ 1) ∎ a = a ∎ 1
axiom star_property3 : 1 ∎ 1 = 2

theorem find_star_value : 5 ∎ 10 = 50 := sorry

end find_star_value_l235_235154


namespace sum_p_q_l235_235168

theorem sum_p_q (p q : ℚ) (g : ℚ → ℚ) (h : g = λ x => (x + 2) / (x^2 + p * x + q))
  (h_asymp1 : ∀ {x}, x = -1 → (x^2 + p * x + q) = 0)
  (h_asymp2 : ∀ {x}, x = 3 → (x^2 + p * x + q) = 0) :
  p + q = -5 := by
  sorry

end sum_p_q_l235_235168


namespace ab_squared_ab_cubed_ab_power_n_l235_235002

-- Definitions of a and b as real numbers, and n as a natural number
variables (a b : ℝ) (n : ℕ)

theorem ab_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by 
  sorry

theorem ab_cubed (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by 
  sorry

theorem ab_power_n (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by 
  sorry

end ab_squared_ab_cubed_ab_power_n_l235_235002


namespace minimum_single_discount_l235_235109

theorem minimum_single_discount (n : ℕ) :
  (∀ x : ℝ, 0 < x → 
    ((1 - n / 100) * x < (1 - 0.18) * (1 - 0.18) * x) ∧
    ((1 - n / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x) ∧
    ((1 - n / 100) * x < (1 - 0.28) * (1 - 0.07) * x))
  ↔ n = 34 :=
by
  sorry

end minimum_single_discount_l235_235109


namespace trig_identity_equality_l235_235245

theorem trig_identity_equality :
  (cos (28 * Real.pi / 180) * cos (56 * Real.pi / 180) / sin (2 * Real.pi / 180) +
   cos (2 * Real.pi / 180) * cos (4 * Real.pi / 180) / sin (28 * Real.pi / 180)) =
  (sqrt 3 * sin (38 * Real.pi / 180) / (4 * sin (2 * Real.pi / 180) * sin (28 * Real.pi / 180))) :=
by
  -- Proof goes here
  sorry

end trig_identity_equality_l235_235245


namespace terminating_decimal_l235_235080

theorem terminating_decimal : (45 / (2^2 * 5^3) : ℚ) = 0.090 :=
by
  sorry

end terminating_decimal_l235_235080


namespace casey_nail_decorating_time_l235_235386

theorem casey_nail_decorating_time 
  (n_toenails n_fingernails : ℕ)
  (t_apply t_dry : ℕ)
  (coats : ℕ)
  (h1 : n_toenails = 10)
  (h2 : n_fingernails = 10)
  (h3 : t_apply = 20)
  (h4 : t_dry = 20)
  (h5 : coats = 3) :
  20 * (t_apply + t_dry) * coats = 120 :=
by
  -- skipping the proof
  sorry

end casey_nail_decorating_time_l235_235386


namespace expressSolutionSetAsInterval_l235_235757

theorem expressSolutionSetAsInterval : {x : ℝ | x ≤ 1} = Iic 1 :=
by sorry

end expressSolutionSetAsInterval_l235_235757


namespace vector_addition_parallel_l235_235511

theorem vector_addition_parallel :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-4, -2)
  (a.2 / a.1) = (b.2 / b.1) →
  (a.1 + b.1, a.2 + b.2) = (-2, -1) := by
  intro h
  have ha : a.2 / a.1 = 1 / 2 := by
    rw [a]; norm_num
  have hb : b.2 / b.1 = -1 / 2 := by
    rw [b]; norm_num
  rw [ha] at h
  rw [hb] at h
  injection h
  simp [a, b]
  apply prod.mk.inj_iff.mpr
  simp
  sorry

end vector_addition_parallel_l235_235511


namespace sally_total_expense_l235_235601

-- Definitions based on the problem conditions
def peaches_price_after_coupon : ℝ := 12.32
def peaches_coupon : ℝ := 3.00
def cherries_weight : ℝ := 2.00
def cherries_price_per_kg : ℝ := 11.54
def apples_weight : ℝ := 4.00
def apples_price_per_kg : ℝ := 5.00
def apples_discount_percentage : ℝ := 0.15
def oranges_count : ℝ := 6.00
def oranges_price_per_unit : ℝ := 1.25
def oranges_promotion : ℝ := 3.00 -- Buy 2, get 1 free means she pays for 4 out of 6

-- Calculation of the total expense
def total_expense : ℝ :=
  (peaches_price_after_coupon + peaches_coupon) + 
  (cherries_weight * cherries_price_per_kg) + 
  ((apples_weight * apples_price_per_kg) * (1 - apples_discount_percentage)) +
  (4 * oranges_price_per_unit)

-- Statement to verify total expense
theorem sally_total_expense : total_expense = 60.40 := by
  sorry

end sally_total_expense_l235_235601


namespace sin_alpha_beta_l235_235789

-- Define the conditions
variables (α β : ℝ)
hypothesis cos_condition : cos (π / 4 - α) = 3 / 5
hypothesis sin_condition : sin (5 * π / 4 + β) = -12 / 13
hypothesis alpha_interval : α ∈ (π / 4, 3 * π / 4)
hypothesis beta_interval : β ∈ (0, π / 4)

-- State the theorem to be proven
theorem sin_alpha_beta :
  sin (α + β) = 56 / 65 :=
sorry

end sin_alpha_beta_l235_235789


namespace final_movie_length_l235_235684

-- Definitions based on conditions
def original_movie_length : ℕ := 60
def cut_scene_length : ℕ := 3

-- Theorem statement proving the final length of the movie
theorem final_movie_length : original_movie_length - cut_scene_length = 57 :=
by
  -- The proof will go here
  sorry

end final_movie_length_l235_235684


namespace hexagon_chord_length_eq_18_l235_235353

theorem hexagon_chord_length_eq_18
    (a b : ℕ)
    (h₁ : is_inscribed_hexagon (mk_hexagon_cyclic 6 4))
    (h₂ : is_partitioned_by_chord_through_diagonal_intersection (mk_hexagon_cyclic 6 4)) :
    let PQ := chord_length (partition_hexagon_by_diagonal (mk_hexagon_cyclic 6 4)) in
    let (m, n) := euclidean_gcd_decomposition (PQ) in
    coprime m n → m + n = 18 := 
by
  -- Insert proof steps here, skipped by the 'sorry' placeholder
  sorry

end hexagon_chord_length_eq_18_l235_235353


namespace gross_salary_after_increase_and_tax_l235_235252

noncomputable def current_salary : ℝ := 30000
noncomputable def increase_rate : ℝ := 0.10
noncomputable def tax_rate : ℝ := 0.13

theorem gross_salary_after_increase_and_tax :
  let new_salary := current_salary * (1 + increase_rate)
  let gross_salary := new_salary / (1 - tax_rate)
  gross_salary ≈ 37931 := by
  sorry

end gross_salary_after_increase_and_tax_l235_235252


namespace sum_odds_200_600_l235_235324

-- Define the bounds 200 and 600 for our range
def lower_bound := 200
def upper_bound := 600

-- Define first and last odd integers in the range
def first_odd := 201
def last_odd := 599

-- Define the common difference in our arithmetic sequence
def common_diff := 2

-- Number of terms in the sequence
def n := ((last_odd - first_odd) / common_diff) + 1

-- Sum of the arithmetic sequence formula
def sum_arithmetic_seq (n : ℕ) (a l : ℕ) : ℕ :=
  n * (a + l) / 2

-- Specifically, the sum of odd integers between 200 and 600
def sum_odd_integers : ℕ := sum_arithmetic_seq n first_odd last_odd

-- Theorem stating the sum is equal to 80000
theorem sum_odds_200_600 : sum_odd_integers = 80000 :=
by sorry

end sum_odds_200_600_l235_235324


namespace Oscar_height_correct_l235_235770

-- Definitions of the given conditions
def Tobias_height : ℕ := 184
def avg_height : ℕ := 178

def heights_valid (Victor Peter Oscar Tobias : ℕ) : Prop :=
  Tobias = 184 ∧ (Tobias + Victor + Peter + Oscar) / 4 = 178 ∧ 
  Victor = Tobias + (Tobias - Peter) ∧ 
  Oscar = Peter - (Tobias - Peter)

theorem Oscar_height_correct :
  ∃ (k : ℕ), ∀ (Victor Peter Oscar : ℕ), heights_valid Victor Peter Oscar Tobias_height →
  Oscar = 160 :=
by
  sorry

end Oscar_height_correct_l235_235770


namespace sin_distinct_inf_values_l235_235163

theorem sin_distinct_inf_values :
  ∀ (k : ℕ), ∃ (m : ℕ) (h1 : m > k), ∀ n ≤ m, ¬ ∃ i j ∈ (finset.range (m + 1)), i ≠ j ∧ (real.sin (i * real.pi / real.sqrt 3) = real.sin (j * real.pi / real.sqrt 3)) :=
by
  sorry

end sin_distinct_inf_values_l235_235163


namespace find_f2_l235_235776

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l235_235776


namespace power_expression_simplify_l235_235318

theorem power_expression_simplify :
  (1 / (-5^2)^3) * (-5)^8 * Real.sqrt 5 = 5^(5/2) :=
by
  sorry

end power_expression_simplify_l235_235318


namespace triangle_problem_l235_235525

theorem triangle_problem 
  (A B C D : Point)
  (s : ℝ)
  (h_eq_triangle_ABC : equiTriangle A B C)
  (h_isosceles_triangle_BCD : isoscelesTriangle B C D s s (s * Real.sqrt 2))
  (h_AD : height A B C)
  (h_CD : height B C D) :
  AD = (s * Real.sqrt 3) → AD / BC = Real.sqrt 3 :=
by
  sorry

end triangle_problem_l235_235525


namespace find_possible_value_of_m_l235_235808

open Nat

def M (m : ℕ) := { x : ℕ | ∃ n : ℕ, n > 0 ∧ x = m / n }

def has_eight_subsets (m : ℕ) : Prop :=
  2^(Set.card (M m)) = 8

def is_possible_value_of_m (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 10 ∧ m ∈ \{ x : ℕ | x > 0 \}

theorem find_possible_value_of_m (m : ℕ) (h : is_possible_value_of_m m) (h_subsets : has_eight_subsets m) : m = 4 :=
sorry

end find_possible_value_of_m_l235_235808


namespace difference_of_distinct_members_l235_235486

theorem difference_of_distinct_members :
  let s := {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
  ∃ n : ℕ, n = 15 ∧ (∀ d ∈ {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}, 
    ∃ a b ∈ s, a ≠ b ∧ a - b = d) :=
by
  let s := {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
  sorry

end difference_of_distinct_members_l235_235486


namespace kramer_votes_percentage_l235_235186

noncomputable def votes_received : ℕ := 942568
noncomputable def percentage_remaining_needed : ℚ := 23.076923076923077 / 100

def kramer_percentage_received (V : ℚ) : ℚ :=
  let K := percentage_remaining_needed * (V - votes_received) in
  (votes_received / V) * 100

theorem kramer_votes_percentage : ∃ V : ℚ, kramer_percentage_received V = 34.99 :=
by
  sorry

end kramer_votes_percentage_l235_235186


namespace sector_area_l235_235782

theorem sector_area (theta l : ℝ) (h_theta : theta = 2) (h_l : l = 2) :
    let r := l / theta
    let S := 1 / 2 * l * r
    S = 1 := by
  sorry

end sector_area_l235_235782


namespace angle_CMN_values_l235_235241

-- Definitions of points and angles
variables {A B C D M N : Type*}
variable [circle : Type*] -- Given original circle
variable [circle' : Type*] -- Any arbitrary circle passing through A and B

-- Conditions
variable (fixed_points : ∀ {circle}, (A B : Type*) → \overrightarrow{AB} = α)
variable (circle_pass : ∀ {circle}, (circle' : Type*) → 
                ∃ (C D : Type*), 
                C is on original circle ∧
                (∃ l : Type*, l intersects circle at A and circle' at C and D))
variable (tangents_meet_M : ∀ (C D M : Type*), 
                tangents to circle at C and D intersect at M)
variable (point_N_line_l : ∀ (N : Type*), 
                ∃ l : Type*, N is on l ∧ 
                |CN| = |AD| ∧ 
                |DN| = |CA|)

-- Theorem statement
theorem angle_CMN_values (α : ℝ) 
  (h1 : fixed_points A B)
  (h2 : ∃ circle', circle_pass circle circle') 
  (h3 : tangents_meet_M C D M)
  (h4 : ∃ N l, point_N_line_l N ∧ N is on line l):
  ∃ θ : ℝ, θ = \frac{α}{2} ∨ θ = 180 - \frac{α}{2} :=
sorry

end angle_CMN_values_l235_235241


namespace sum_of_solutions_l235_235990

def g (x : ℝ) : ℝ := 20 * x + 4

theorem sum_of_solutions :
  let g_inv := (g⁻¹ : ℝ → ℝ) in
  let eq := ∀ x, g_inv x = g ((3 * x)⁻¹) in
  -- Solution for x from the above equation
  let roots := {x | g_inv x = g ((3 * x)⁻¹)} in
  ∑ x in roots.to_finset, x = 92 / 3 :=
begin
  sorry
end

end sum_of_solutions_l235_235990


namespace find_a4_l235_235788

variable {α : Type*}
variables (a : ℕ → α) [AddCommGroup α] [Module ℚ α]

def is_arithmetic_sequence (a : ℕ → α) :=
∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem find_a4
  (h_arith : is_arithmetic_sequence a)
  (h1 : a 2 + a 5 + a 8 = 27)
  (h2 : a 3 + a 6 + a 9 = 33) :
  a 4 = 7 :=
sorry

end find_a4_l235_235788


namespace lambda_range_l235_235467

noncomputable def vector_a (λ : ℝ) : ℝ × ℝ := (λ, 4)
def vector_b : ℝ × ℝ := (-3, 5)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def is_obtuse_angle (u v : ℝ × ℝ) : Prop :=
  dot_product u v < 0 ∧ ¬ (u.1 / v.1 = u.2 / v.2)

theorem lambda_range (λ : ℝ) (h_obtuse : is_obtuse_angle (vector_a λ) vector_b) : λ > 20 / 3 :=
by
  sorry

end lambda_range_l235_235467


namespace c_plus_d_eq_neg_two_l235_235930

variable {R : Type} [Field R]

def g (c d : R) (x : R) : R :=
  c * x + d

def g_inv (c d : R) (x : R) : R :=
  d * x + c

theorem c_plus_d_eq_neg_two (c d : R) (h_inv : ∀ x : R, g c d (g_inv c d x) = x ∧ g_inv c d (g c d x) = x) :
  c + d = -2 := by
have h_cd : c * d = 1 := by
  sorry
have h_d : d = -c^2 := by
  sorry
have h_cube : c^3 = -1 := by
  sorry
have h_c : c = -1 := by
  sorry
have h_d_resolved : d = -1 := by
  sorry
show c + d = -2 := by
  rw [h_c, h_d_resolved]
  norm_num

end c_plus_d_eq_neg_two_l235_235930


namespace distinct_diagonals_nonagon_l235_235860

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235860


namespace exists_prime_divisor_one_of_six_consecutive_l235_235129

theorem exists_prime_divisor_one_of_six_consecutive (n : ℕ) (h : 0 < n) :
  ∃ p : ℕ, prime p ∧ (∃! k, k ∈ finset.range 6 ∧ p ∣ (n + k)) :=
  sorry

end exists_prime_divisor_one_of_six_consecutive_l235_235129


namespace count_irrational_is_one_l235_235619

-- Define the numbers according to the conditions
def num1 : ℝ := (-real.sqrt 2)^2
def num2 : ℝ := 22 / 7
def num3 : ℝ := 0.51
def num4 : ℝ := -real.pi
def num5 : ℝ := real.cbrt 8
def num6 : ℝ := -234.101010101010 -- Represents repeating decimal with one zero between adjacent ones

-- The theorem to prove: list of the given numbers contains exactly one irrational number
theorem count_irrational_is_one :
  list.countp (λ x : ℝ, irrational x) [num1, num2, num3, num4, num5, num6] = 1 := sorry

end count_irrational_is_one_l235_235619


namespace maximum_velocity_time_period_l235_235711

-- Define the time intervals for acceleration and deceleration
def acceleration_interval := (2 : ℝ) ≤ t ∧ t ≤ (4 : ℝ)
def deceleration_interval := (22 : ℝ) ≤ t ∧ t ≤ (24 : ℝ)

-- Define the property of maximum downward velocity
def maximum_downward_velocity_interval := (4 : ℝ) ≤ t ∧ t ≤ (22 : ℝ)

-- Final statement with the intervals of acceleration and deceleration as conditions
theorem maximum_velocity_time_period (t : ℝ) :
  (acceleration_interval ∨ deceleration_interval) → maximum_downward_velocity_interval := 
sorry

end maximum_velocity_time_period_l235_235711


namespace ball_never_returns_l235_235671

-- Define the problem conditions in a mathematically rigorous way

structure BilliardTable :=
  (vertices : Set Point)
  (sides : Set (Point × Point))
  (angle_90 : ∃ (A : vertices), angle_between_sides A = 90)
  (perpendicular_sides : ∀ (A B C : vertices), A ≠ B ∧ B ≠ C ∧ A ≠ C → angle_between_sides B = 90 ∨ angle_between_sides C = 90)

open Set

-- Define the trajectory and reflection properties
def trajectory_path (table : BilliardTable) (start : table.vertices) : sequence (Point × Point) :=
  sorry -- assuming the reflection law is implicitly defined

-- Prove the main theorem
theorem ball_never_returns (table : BilliardTable) (A : table.vertices)
  (Hangle : angle_between_sides A = 90)
  (start_direction : Vector) :
  (∀ trajectory : ℕ → Point, trajectory_path table A = trajectory → (∀ n, trajectory n ≠ A)) :=
sorry

end ball_never_returns_l235_235671


namespace line_AH_bisects_angle_EHF_l235_235395

-- Definitions of the points and lines in the problem
variables (A B C E F H X Y : Type) -- Types representing points

-- Assumptions based on the problem conditions
variables [Triangle ABC]
variables [Circle ω]
variables (center_on_BC : ω.Center ∈ Line B C)
variables (tangent_AC : ω.Tangent E ∈ Line A C)
variables (tangent_AB : ω.Tangent F ∈ Line A B)
variables (altitude_foot_H : H ∈ Line A (Line B C))

-- The final proof goal
theorem line_AH_bisects_angle_EHF :
  bisects (Line A H) (∠ E H F) :=
sorry -- proof goes here

end line_AH_bisects_angle_EHF_l235_235395


namespace concyclicity_condition_l235_235565

variables {A B C H N O' D : Type}
variables {a b c R : ℝ}

-- Definitions based on problem conditions
def orthocenter (A B C H : Type) : Prop := sorry
def circumcenter (B H C O' : Type) : Prop := sorry
def midpoint (A O' N : Type) : Prop := sorry
def reflection (N BC D : Type) : Prop := sorry
def concyclic (A B D C : Type) : Prop := sorry
def side_lengths (A B C : Type) (a b c : ℝ) := sorry
def circumradius (A B C : Type) (R : ℝ) := sorry

theorem concyclicity_condition 
  (h_orthocenter : orthocenter A B C H)
  (h_circumcenter : circumcenter B H C O')
  (h_midpoint : midpoint A O' N)
  (h_reflection : reflection N (BC : Type) D)
  (h_side_lengths : side_lengths A B C a b c)
  (h_circumradius : circumradius A B C R) :
  (concyclic A B D C) ↔ (b^2 + c^2 - a^2 = 3 * R^2) := 
sorry

end concyclicity_condition_l235_235565


namespace find_angleRTU_l235_235962
noncomputable theory

variables (angleSTQ anglePTU angleRTU : ℝ)

def angles_given : Prop :=
  angleSTQ = 140 ∧ anglePTU = 90

theorem find_angleRTU (h : angles_given) : angleRTU = 50 :=
sorry

end find_angleRTU_l235_235962


namespace find_a3_l235_235199

def a : ℕ → ℕ
| 0     := 2
| (n+1) := 2 * a n - 1

theorem find_a3 : a 2 = 5 :=
sorry

end find_a3_l235_235199


namespace max_M_value_l235_235102

noncomputable section

def J_k (k : ℕ) : ℕ := 10 ^ (k + 2) + 128

def M (k : ℕ) : ℕ :=
  let n := J_k k
  let factors := n.factorization 
  factors.findWithDefault 2 0

theorem max_M_value : ∃ k > 0, M k = 8 := by
  sorry

end max_M_value_l235_235102


namespace trajectory_length_l235_235945
noncomputable theory

open Real

def rectangular_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

theorem trajectory_length : 
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 → rectangular_distance (x, y) (1, 3) = rectangular_distance (x, y) (6, 9)) →
  ∑ (p : set (ℝ × ℝ)), rectangular_distance p (1,3) + rectangular_distance p (6,9) = 5 * (sqrt 2 + 1) :=
sorry

end trajectory_length_l235_235945


namespace part_1_part_2_l235_235474

-- Define the functions f, g and F
def f (x : ℝ) : ℝ := Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := a / x
def F (a : ℝ) (x : ℝ) : ℝ := f x + g a x

-- Part (Ⅰ)
theorem part_1 (a : ℝ) (x : ℝ) (h : a = 1) : 
  (∀ x, x > 1 → (F a)' x > 0) ∧ (∀ x, x > 0 ∧ x < 1 → (F a)' x < 0) :=
sorry

-- Part (Ⅱ)
theorem part_2 (x_0 : ℝ) (a : ℝ): 
  (∀ x_0, 0 < x_0 ∧ x_0 ≤ 3 → (x_0 - a) / (x_0^2) ≤ 1/2) → a ≥ 1/2 :=
sorry

end part_1_part_2_l235_235474


namespace sum_of_integers_l235_235766

theorem sum_of_integers (n : ℕ) (h1 : 1.5 * n - 6.3 < 7.5) : 
  ∑ k in Finset.filter (λ k, 1.5 * k - 6.3 < 7.5) (Finset.range 10) = 45 :=
sorry

end sum_of_integers_l235_235766


namespace maximum_value_M_l235_235099

-- Define the function J_k as described in the problem
def J (k : ℕ) : ℕ := 10^(k + 3) + 128

-- Define the function M(k), which counts the number of factors of 2 in the factorization of J_k
def M (k : ℕ) : ℕ := (J k).factor 2

-- Theorem statement proving the maximum value of M(k)
theorem maximum_value_M (k : ℕ) (h : k > 0) : M k ≤ 8 :=
sorry

end maximum_value_M_l235_235099


namespace apply_trig_funcs_to_get_2010_l235_235536

theorem apply_trig_funcs_to_get_2010 : 
  ∃ (f : ℝ → ℝ), 
  (∀ (x : ℝ), 
    f x = sin x ∨ f x = cos x ∨ f x = tan x ∨ f x = cot x ∨ f x = arcsin x ∨ f x = arccos x ∨ f x = arctan x ∨ f x = arccot x) 
    ∧ f 2 = 2010 :=
sorry

end apply_trig_funcs_to_get_2010_l235_235536


namespace number_of_zeros_at_end_l235_235926

def N (n : Nat) := 10^(n+1) + 1

theorem number_of_zeros_at_end (n : Nat) (h : n = 2017) : 
  (N n)^(n + 1) - 1 ≡ 0 [MOD 10^(n + 1)] :=
sorry

end number_of_zeros_at_end_l235_235926


namespace car_B_speed_90_l235_235311

def car_speed_problem (distance : ℝ) (ratio_A : ℕ) (ratio_B : ℕ) (time_minutes : ℝ) : Prop :=
  let x := distance / (ratio_A + ratio_B) * (60 / time_minutes)
  (ratio_B * x = 90)

theorem car_B_speed_90 
  (distance : ℝ := 88)
  (ratio_A : ℕ := 5)
  (ratio_B : ℕ := 6)
  (time_minutes : ℝ := 32)
  : car_speed_problem distance ratio_A ratio_B time_minutes :=
by
  sorry

end car_B_speed_90_l235_235311


namespace binomial_60_3_eq_34220_l235_235039

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235039


namespace total_kids_from_2004_to_2009_l235_235179

def kids_2004 := 60
def kids_2005 := kids_2004 / 2
def kids_2006 := (2 / 3) * kids_2005
def kids_2007 := (3 / 4) * kids_2006
def kids_2008 := Int.round (1.25 * kids_2007) -- rounding to the nearest integer
def kids_2009 := Int.round (kids_2008 - (1 / 5) * kids_2008) -- rounding to the nearest integer

def total_kids := kids_2004 + kids_2005 + kids_2006 + kids_2007 + kids_2008 + kids_2009

theorem total_kids_from_2004_to_2009 : total_kids = 159 := by
  calc
    total_kids = kids_2004 + kids_2005 + kids_2006 + kids_2007 + kids_2008 + kids_2009 := by rfl
    ... = 60 + kids_2005 + kids_2006 + kids_2007 + kids_2008 + kids_2009 := by rfl
    ... = 60 + (60 / 2) + kids_2006 + kids_2007 + kids_2008 + kids_2009 := by rfl
    ... = 60 + 30 + (2 / 3) * 30 + kids_2007 + kids_2008 + kids_2009 := by rfl
    ... = 60 + 30 + 20 + (3 / 4) * 20 + Int.round (1.25 * ((3 / 4) * 20)) + Int.round (Int.round (1.25 * ((3 / 4) * 20)) - (1/5) * Int.round (1.25 * ((3 / 4) * 20))) := by rfl
    ... = 60 + 30 + 20 + 15 + 19 + 15 := by sorry
    ... = 159 := by rfl

end total_kids_from_2004_to_2009_l235_235179


namespace distinct_diagonals_nonagon_l235_235862

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235862


namespace polygon_not_necessarily_symmetrical_l235_235358

theorem polygon_not_necessarily_symmetrical (P : Type) [polygon_like P] 
  (H : ∃ (l₁ l₂ l₃ : line), divides_into_equal_parts P l₁ ∧ divides_into_equal_parts P l₂ ∧ divides_into_equal_parts P l₃) 
  : ¬ (has_center_of_symmetry P ∨ has_axis_of_symmetry P) := 
by
  sorry

end polygon_not_necessarily_symmetrical_l235_235358


namespace ages_sum_is_71_l235_235376

def Beckett_age : ℕ := 12
def Olaf_age : ℕ := Beckett_age + 3
def Shannen_age : ℕ := Olaf_age - 2
def Jack_age : ℕ := 2 * Shannen_age + 5
def sum_of_ages : ℕ := Beckett_age + Olaf_age + Shannen_age + Jack_age

theorem ages_sum_is_71 : sum_of_ages = 71 := by
  unfold sum_of_ages Beckett_age Olaf_age Shannen_age Jack_age
  calc
    12 + (12 + 3) + (12 + 3 - 2) + (2 * (12 + 3 - 2) + 5)
      = 12 + 15 + 13 + 31 := by rfl
      ... = 71 := by rfl

end ages_sum_is_71_l235_235376


namespace intermediate_root_exists_l235_235247

open Polynomial

theorem intermediate_root_exists
  (a b c x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : -a * x2^2 + b * x2 + c = 0) :
  ∃ x3 : ℝ, (a / 2) * x3^2 + b * x3 + c = 0 ∧ (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) :=
sorry

end intermediate_root_exists_l235_235247


namespace log_sum_solution_l235_235604

theorem log_sum_solution (x : ℝ) (h₀ : 0 < x) (h₁ : Real.log 2 x + Real.log 4 x + Real.log 8 x = 9) : x = 2^(54 / 11) :=
by
  sorry

end log_sum_solution_l235_235604


namespace casey_nail_decorating_time_l235_235387

/-- Given the conditions:
1. Casey wants to apply three coats: a base coat, a coat of paint, and a coat of glitter.
2. Each coat takes 20 minutes to apply.
3. Each coat requires 20 minutes of drying time before the next one can be applied.

Prove that the total time taken by Casey to finish decorating her fingernails and toenails is 120 minutes.
-/
theorem casey_nail_decorating_time
  (application_time : ℕ)
  (drying_time : ℕ)
  (num_coats : ℕ)
  (total_time : ℕ)
  (h_app_time : application_time = 20) 
  (h_dry_time : drying_time = 20)
  (h_num_coats : num_coats = 3)
  (h_total_time_eq : total_time = num_coats * (application_time + drying_time)) :
  total_time = 120 :=
sorry

end casey_nail_decorating_time_l235_235387


namespace tan_cos_identity_l235_235344

theorem tan_cos_identity : 
  ∀(θ:ℝ), θ = 70 * real.pi / 180 → 
    (real.tan (70 * real.pi / 180) * real.cos (10 * real.pi / 180) * (real.tan (20 * real.pi / 180) - 1)) = -1 :=
by
  intros θ hθ
  sorry

end tan_cos_identity_l235_235344


namespace find_n_l235_235083

theorem find_n (n : ℕ) :
  (2^n - 1) % 3 = 0 ∧ (∃ m : ℤ, (2^n - 1) / 3 ∣ 4 * m^2 + 1) →
  ∃ j : ℕ, n = 2^j :=
by
  sorry

end find_n_l235_235083


namespace functional_equation_zero_l235_235410

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end functional_equation_zero_l235_235410


namespace intersection_of_A_and_B_union_of_A_and_B_l235_235479

variables {x : ℝ}

def A : set ℝ := { x | -1 ≤ x ∧ x < 3 }
def B : set ℝ := { x | 2x - 4 ≥ x - 2 }
def B_simplified : set ℝ := { x | x ≥ 2 }

theorem intersection_of_A_and_B : A ∩ B = { x | 2 ≤ x ∧ x < 3 } :=
by
  sorry

theorem union_of_A_and_B : A ∪ B = { x | x ≥ -1 } :=
by
  sorry

end intersection_of_A_and_B_union_of_A_and_B_l235_235479


namespace pentagon_rectangle_ratio_l235_235361

theorem pentagon_rectangle_ratio :
  let p : ℝ := 60  -- Perimeter of both the pentagon and the rectangle
  let length_side_pentagon : ℝ := 12
  let w : ℝ := 10
  p / 5 = length_side_pentagon ∧ p/6 = w ∧ length_side_pentagon / w = 6/5 :=
sorry

end pentagon_rectangle_ratio_l235_235361


namespace find_m_n_d_l235_235182

noncomputable def circle_chord_area (r : ℝ) (l : ℝ) (d : ℝ) : ℝ :=
  let sector_area := (1/4) * Real.pi * r^2
  let triangle_area := (1/2) * 84 * 24
  sector_area + triangle_area

theorem find_m_n_d 
  (r : ℝ) (chord_length : ℝ) (distance_from_center : ℝ) 
  (m n d : ℕ)
  (h_radius : r = 48)
  (h_chord_length : chord_length = 84)
  (h_distance : distance_from_center = 24)
  (h_correct_area : circle_chord_area r chord_length distance_from_center = 576 * Real.pi - 528 * Real.sqrt 198)
  (h_m : m = 576)
  (h_n : n = 528)
  (h_d : d = 198) :
  m + n + d = 1302 :=
by
  simp only
  sorry

end find_m_n_d_l235_235182


namespace function_even_periodic_l235_235703

theorem function_even_periodic (f : ℝ → ℝ) :
  (∀ x : ℝ, f (10 + x) = f (10 - x)) ∧ (∀ x : ℝ, f (5 - x) = f (5 + x)) →
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + 10) = f x) :=
by
  sorry

end function_even_periodic_l235_235703


namespace distinct_diagonals_in_nonagon_l235_235906

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235906


namespace nonagon_diagonals_l235_235841

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235841


namespace complex_sum_theorem_l235_235222

noncomputable def complex_sum_property (x : ℂ) (h₁ : x ^ 2027 = 1) (h₂ : x ≠ 1) : Prop :=
  (∑ k in Finset.range 12, x ^ (3 * 2 ^ k) / (x ^ (2 ^ k) - 1)) = 33 / 2

-- Now define a theorem that asserts the property
theorem complex_sum_theorem (x : ℂ) (h₁ : x ^ 2027 = 1) (h₂ : x ≠ 1) :
  complex_sum_property x h₁ h₂ :=
by sorry

end complex_sum_theorem_l235_235222


namespace total_pencils_l235_235365

theorem total_pencils (initial_additional1 initial_additional2 : ℕ) (h₁ : initial_additional1 = 37) (h₂ : initial_additional2 = 17) : (initial_additional1 + initial_additional2) = 54 :=
by sorry

end total_pencils_l235_235365


namespace min_value_of_g_l235_235799

-- Given conditions and definitions
def f (a x : ℝ) : ℝ := a * (x - 1) / (x ^ 2)
def g (a x : ℝ) : ℝ := x * Real.log x - a * (x - 1)
def interval := set.Icc (1 : ℝ) (Real.exp 1)

-- Statement asserting the minimum value of g over the given interval for different ranges of a
theorem min_value_of_g (a : ℝ) (h : 0 < a) : 
  (h1 : (0 < a ∧ a ≤ 1) → ∃ x ∈ interval, g a x = 0) ∧
  (h2 : (a ≥ 2) → ∃ x ∈ interval, g a x = Real.exp 1 + a - a * Real.exp 1) ∧
  (h3 : (1 < a ∧ a < 2) → ∃ x ∈ interval, g a x = a - Real.exp (a - 1)) :=
by sorry

end min_value_of_g_l235_235799


namespace sum_of_triangle_ops_l235_235175

def triangle_op (a b c : ℤ) : ℤ := 2 * a + 3 * b - 4 * c

theorem sum_of_triangle_ops : 
  triangle_op 2 3 5 + triangle_op 4 6 1 = 15 := 
by {
  rw [triangle_op, triangle_op],
  norm_num,
  sorry
}

end sum_of_triangle_ops_l235_235175


namespace floor_root_product_l235_235740

theorem floor_root_product :
  (∏ n in (1..2045).filter (λ n, n % 2 = 1), ⌊real.root 4 n⌋) / 
  (∏ n in (1..2046).filter (λ n, n % 2 = 0), ⌊real.root 4 n⌋) = (5 : ℝ) / 16 :=
by
  sorry

end floor_root_product_l235_235740


namespace angle_bisector_theorem_l235_235675

noncomputable def angle_bisector_length {a b : ℝ} {γ : ℝ} (hab : a > 0) (hbb : b > 0) (hγ : 0 < γ ∧ γ < real.pi) : ℝ :=
  (2 * a * b * real.cos (γ / 2)) / (a + b)

theorem angle_bisector_theorem {a b : ℝ} {γ : ℝ} (hab : a > 0) (hbb : b > 0) (hγ : 0 < γ ∧ γ < real.pi)
  (l : ℝ) (hl : l = angle_bisector_length hab hbb hγ) :
  l = (2 * a * b * real.cos (γ / 2)) / (a + b) := by
  sorry

end angle_bisector_theorem_l235_235675


namespace initial_people_count_l235_235609

theorem initial_people_count (C : ℝ) (n : ℕ) (h : n > 1) :
  ((C / (n - 1)) - (C / n) = 0.125) →
  n = 8 := by
  sorry

end initial_people_count_l235_235609


namespace find_TU_square_l235_235610

-- Definitions
variables (P Q R S T U : ℝ × ℝ)
variable (side : ℝ)
variable (QT RU PT SU PQ : ℝ)

-- Setting the conditions
variables (side_eq_10 : side = 10)
variables (QT_eq_7 : QT = 7)
variables (RU_eq_7 : RU = 7)
variables (PT_eq_24 : PT = 24)
variables (SU_eq_24 : SU = 24)
variables (PQ_eq_10 : PQ = 10)

-- The theorem statement
theorem find_TU_square : TU^2 = 1150 :=
by
  -- Proof to be done here.
  sorry

end find_TU_square_l235_235610


namespace smallest_positive_x_undefined_l235_235653

theorem smallest_positive_x_undefined :
  ∃ x > 0, (12 * x^2 - 50 * x + 12 = 0) ∧ (∀ y > 0, (12 * y^2 - 50 * y + 12 = 0) → x ≤ y) :=
begin
  use 1/6,
  split,
  { norm_num, exact zero_lt_one.div (by norm_num) },
  split,
  { norm_num, ring },
  { intros y hy hy_eq,
    sorry }
end

end smallest_positive_x_undefined_l235_235653


namespace cards_in_center_pile_l235_235689

/-- Represents the number of cards in each pile initially. -/
def initial_cards (x : ℕ) : Prop := x ≥ 2

/-- Represents the state of the piles after step 2. -/
def step2 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 2 ∧ right = x

/-- Represents the state of the piles after step 3. -/
def step3 (x : ℕ) (left center right : ℕ) : Prop :=
  left = x - 2 ∧ center = x + 3 ∧ right = x - 1

/-- Represents the state of the piles after step 4. -/
def step4 (x : ℕ) (left center : ℕ) : Prop :=
  left = 2 * x - 4 ∧ center = 5

/-- Prove that after performing all steps, the number of cards in the center pile is 5. -/
theorem cards_in_center_pile (x : ℕ) :
  initial_cards x →
  (∃ l₁ c₁ r₁, step2 x l₁ c₁ r₁) →
  (∃ l₂ c₂ r₂, step3 x l₂ c₂ r₂) →
  (∃ l₃ c₃, step4 x l₃ c₃) →
  ∃ (center_final : ℕ), center_final = 5 :=
by
  sorry

end cards_in_center_pile_l235_235689


namespace haman_eggs_sold_l235_235485

-- Define the initial number of trays
def initial_trays : ℕ := 10

-- Define the number of trays dropped
def dropped_trays : ℕ := 2

-- Define the additional trays added after the accident
def additional_trays : ℕ := 7

-- Define the number of eggs per tray
def eggs_per_tray : ℕ := 30

-- The proof problem
theorem haman_eggs_sold : 
  let remaining_trays := initial_trays - dropped_trays,
      total_trays := remaining_trays + additional_trays
  in total_trays * eggs_per_tray = 450 :=
by
  sorry

end haman_eggs_sold_l235_235485


namespace train_length_l235_235717

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end train_length_l235_235717


namespace locus_of_midpoints_of_chords_through_P_is_circle_l235_235548

theorem locus_of_midpoints_of_chords_through_P_is_circle
  (K : Type) [metric_space K] [normed_group K] [normed_space ℝ K]
  (O P : K) (r : ℝ) (hP : P ≠ O) (hP_in : dist O P < r) :
  ∃ (M : K) (R : ℝ), ∀ (A B : K), (dist O A = r) → (dist O B = r) → (P ∈ line_through A B) →
  (dist M ((A + B) / 2) = R) :=
sorry

end locus_of_midpoints_of_chords_through_P_is_circle_l235_235548


namespace equality_of_remainders_l235_235345

theorem equality_of_remainders (a b : ℕ) (h : a > 0 ∧ b > 0) 
  (h_prime_rem : ∀ p : ℕ, p.prime → (a % p) ≤ (b % p)) : a = b :=
begin
  sorry
end

end equality_of_remainders_l235_235345


namespace exists_4_clique_not_necessarily_exists_5_clique_l235_235306

-- Define the structure of a graph and the problem conditions
structure Graph (V : Type) :=
(adjacency : V → V → Prop)
(symm : ∀ v u, adjacency v u → adjacency u v)
(loopfree : ∀ v, ¬ adjacency v v)

noncomputable def astronauts := {x // x < 20}

-- Given conditions
axiom at_least_15_friends (G : Graph astronauts) (v : astronauts) : 
  ∃ (S : finset astronauts), S.card >= 15 ∧ ∀ u ∈ S, G.adjacency v u

-- Problem (a) - Formation of a crew of four people who are all friends with each other
theorem exists_4_clique (G : Graph astronauts) : 
  ∃ (S : finset astronauts), S.card = 4 ∧ ∀ v ∈ S, ∀ u ∈ S, v ≠ u → G.adjacency v u :=
sorry

-- Problem (b) - Formation of a crew of five people who are all friends with each other
theorem not_necessarily_exists_5_clique (G : Graph astronauts) : 
  ¬∀ (S : finset astronauts), S.card = 5 → ∀ v ∈ S, ∀ u ∈ S, v ≠ u → G.adjacency v u := 
sorry

end exists_4_clique_not_necessarily_exists_5_clique_l235_235306


namespace power_equality_l235_235499

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l235_235499


namespace max_value_m_inequality_l235_235091

theorem max_value_m_inequality :
  ∃ x : ℝ, ∀ m : ℝ, 
    m > 0 →
    m * real.sqrt m * (x^2 - 6 * x + 9) + real.sqrt m / (x^2 - 6 * x + 9) 
    ≤ real.sqrt (real.sqrt (m^3)) * abs (real.cos (real.pi * x / 5)) 
    ↔ m ≤ 1/16 :=
sorry

end max_value_m_inequality_l235_235091


namespace equilateral_triangle_exists_l235_235420

noncomputable def exists_equilateral_triangle (L : ℝ) (ε : ℝ) (hε : 0 < ε) : Prop :=
∀ ℓ : ℝ, ℓ > L → ∃ (A B C : ℝ × ℝ), 
  dist A B = ℓ ∧
  dist B C = ℓ ∧
  dist C A = ℓ ∧
  dist_to_nearest_lattice_point A < ε ∧
  dist_to_nearest_lattice_point B < ε ∧
  dist_to_nearest_lattice_point C < ε

theorem equilateral_triangle_exists (ε : ℝ) (hε : 0 < ε) : 
  ∃ L > 0, exists_equilateral_triangle L ε hε := 
sorry

end equilateral_triangle_exists_l235_235420


namespace circumcircle_tangent_to_lines_l235_235569

-- Definitions of the circles, points, and tangents as per the conditions
variables
  (Γ₁ Γ₂ : Type)
  (P Q A B C R : Type)
  [circle Γ₁] [circle Γ₂]
  [point P] [point Q] [point A] [point B] [point C] [point R]
  [tangent Γ₁ Γ₂ P A B] -- The common tangent closer to P
  [tangent Γ₁ P C] -- Tangent of Γ₁ at P meets Γ₂ at C
  (h₁ : P ≠ Q)
  (h₂ : ¬ collinear P Q A)
  (h₃ : ∃ R : Type, line_eq (line_through A P) (line_eq (extension A P)) (line_through B C)) -- R is where AP meets BC

-- Circumcircle
def circumcircle (Δ : Type) [triangle Δ] :=
  ∃ γ : circle Δ, ∀ P Q R : Δ, on_circle γ P ∧ on_circle γ Q ∧ on_circle γ R

-- Tangency to lines BP and BR
def tangent_to_lines (circ : circumcircle (Type)) :=
  ∃ (γ : circle circ), is_tangent γ (line_through B P) ∧ is_tangent γ (line_through B R)

theorem circumcircle_tangent_to_lines :
  ∃ (circ : circumcircle (Type)), tangent_to_lines circ :=
  sorry

end circumcircle_tangent_to_lines_l235_235569


namespace area_of_T_l235_235985

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 4)

def T : set ℂ := {z | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 2 ∧ z = a + b*omega + c*omega^2}

theorem area_of_T : MeasureTheory.measure_univ T = 2 :=
by
  sorry

end area_of_T_l235_235985


namespace probability_both_counterfeits_given_one_is_counterfeit_l235_235243

open ProbabilityTheory

def total_banknotes : ℕ := 20
def counterfeit_banknotes : ℕ := 5
def total_pairs := Nat.choose total_banknotes 2
def counterfeit_pairs := Nat.choose counterfeit_banknotes 2
def one_counterfeit_one_real_pairs := counterfeit_banknotes * (total_banknotes - counterfeit_banknotes)
def prob_A := (counterfeit_pairs : ℚ) / (total_pairs : ℚ)
def prob_B := ((counterfeit_pairs : ℚ) + (one_counterfeit_one_real_pairs : ℚ)) / (total_pairs : ℚ)

theorem probability_both_counterfeits_given_one_is_counterfeit 
  (hA : A ⊆ B) :
  (prob_A / prob_B) = (2 / 17) := by
sorry

end probability_both_counterfeits_given_one_is_counterfeit_l235_235243


namespace first_1000_integers_represented_l235_235924

def floor_sum (x : ℝ) : ℤ :=
  (⌊3 * x⌋ : ℤ) + (⌊6 * x⌋ : ℤ) + (⌊9 * x⌋ : ℤ) + (⌊12 * x⌋ : ℤ)

theorem first_1000_integers_represented : 
  ∃ n : ℕ, n = 880 ∧ ∀ i : ℤ, 1 ≤ i ∧ i ≤ 1000 → ∃ x : ℝ, floor_sum x = i :=
sorry

end first_1000_integers_represented_l235_235924


namespace volume_space_inside_sphere_outside_cylinder_l235_235363

noncomputable def sphere_radius : ℝ := 7
noncomputable def cylinder_radius : ℝ := 4
noncomputable def volume_inside_sphere_outside_cylinder : ℝ := (1372/3) * Real.pi - 32 * Real.pi * Real.sqrt 33

theorem volume_space_inside_sphere_outside_cylinder : 
    ∀ (r_sphere r_cylinder : ℝ),
    r_sphere = sphere_radius → 
    r_cylinder = cylinder_radius → 
    volume_inside_sphere_outside_cylinder = (1372/3) * Real.pi - 32 * Real.pi * Real.sqrt 33 :=
by
    intros
    rw [‹r_sphere = sphere_radius›, ‹r_cylinder = cylinder_radius›]
    exact eq.refl _

end volume_space_inside_sphere_outside_cylinder_l235_235363


namespace term_a3_eq_1_l235_235135

variable {a : ℕ → ℝ} -- the geometric sequence
variable T : ℕ → ℝ -- product of the first n terms

-- All the terms of the geometric sequence are positive
axiom positive_terms : ∀ n, a n > 0

-- The product of the first n terms is T n
axiom product_terms : ∀ n, T (n + 1) = T n * a (n + 1)

-- Given condition: T_5 = 1
axiom T_5_eq_1 : T 5 = 1

-- Using the property of geometric sequences
axiom geometric_property : ∀ {a : ℕ → ℝ} {T : ℕ → ℝ},
    (∀ n, T (n + 1) = T n * a (n + 1)) →
    (∀ n, T n = a 0 ^ n) -- simplified property for positive sequences 

theorem term_a3_eq_1 (h_geom : ∀ n, T (n + 1) = T n * a (n + 1)) :
  a 2 = 1 := -- Here, a_3 is represented as a 2 due to zero-based indexing
by
  -- proof will go here
  sorry -- placeholder for the proof

end term_a3_eq_1_l235_235135


namespace distinct_diagonals_in_nonagon_l235_235908

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235908


namespace range_of_a_l235_235448

variable (x a : ℝ)

-- Define proposition p
def prop_p : Prop := x^2 + 2 * x - 3 ≤ 0

-- Define proposition q
def prop_q : Prop := x ≤ a

-- Define the sufficient but not necessary condition
def sufficient_but_not_necessary (p q : Prop) : Prop := q → p ∧ ¬ (p → q)

theorem range_of_a :
  (prop_q x a → prop_p x ∧ ¬ (prop_p x → prop_q x)) →
  a ≥ 1 :=
sorry

end range_of_a_l235_235448


namespace binomial_60_3_l235_235028

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235028


namespace solve_x4_plus_81_eq_zero_l235_235412

open Complex

theorem solve_x4_plus_81_eq_zero :
  {x : ℂ | x^4 + 81 = 0} =
  { ⟨3 * Complex.sqrt 2 / 2 + 3 * Complex.sqrt 2 / 2 * Complex.I,
     -3 * Complex.sqrt 2 / 2 - 3 * Complex.sqrt 2 / 2 * Complex.I,
     -3 * Complex.sqrt 2 / 2 + 3 * Complex.sqrt 2 / 2 * Complex.I,
     3 * Complex.sqrt 2 / 2 - 3 * Complex.sqrt 2 / 2 * Complex.I⟩ } :=
by
  sorry

end solve_x4_plus_81_eq_zero_l235_235412


namespace find_missing_digit_divisibility_by_4_l235_235093

theorem find_missing_digit_divisibility_by_4 (x : ℕ) (h : x < 10) :
  (3280 + x) % 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 :=
by
  sorry

end find_missing_digit_divisibility_by_4_l235_235093


namespace distinct_diagonals_convex_nonagon_l235_235857

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235857


namespace find_expression_value_l235_235426

theorem find_expression_value (a b : ℝ) (h : 5 * a - 3 * b + 2 = 0) : 10 * a - 6 * b - 3 = -7 :=
by
sorrry

end find_expression_value_l235_235426


namespace nonagon_diagonals_count_l235_235890

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235890


namespace average_speed_round_trip_l235_235336

variable (D : ℝ) -- Assuming the distance between Silver Town and Gold Town is positive real number

theorem average_speed_round_trip (h1 : D > 0) :
  let time_upstream := D / 6
  let time_downstream := D / 3
  let total_distance := 2 * D
  let total_time := time_upstream + time_downstream
  total_distance / total_time = 4 :=
by
  have time_upstream_def : time_upstream = D / 6 := rfl
  have time_downstream_def : time_downstream = D / 3 := rfl
  have total_distance_def : total_distance = 2 * D := rfl
  have total_time_def : total_time = time_upstream + time_downstream := rfl
  rw [time_upstream_def, time_downstream_def, total_distance_def, total_time_def]
  sorry

end average_speed_round_trip_l235_235336


namespace find_point_B_l235_235128

noncomputable def point (x y z : ℝ) := (x, y, z : ℝ × ℝ × ℝ)

-- Define given points
def P := point 1 2 3
def A := point (-1) 3 (-3)
def B' := point (3.1) (1+2) (5 +3)

-- Define symmetric conditions
def midpoint (p q r : ℝ × ℝ × ℝ) : Prop :=
  (p.1 + q.1) / 2 = r.1 ∧
  (p.2 + q.2) / 2 = r.2 ∧
  (p.3 + q.3) / 2 = r.3

def vectorSub (p q : ℝ × ℝ × ℝ) := 
  (p.1 - q.1, p.2 - q.2, p.3 - q.3)

-- A'B' = vectorSub B' A'
def A'B' := (3, 1, 5 : ℝ × ℝ × ℝ)

-- Prove that B = (-4,2,-8)
theorem find_point_B (B : ℝ × ℝ × ℝ) :
  midpoint A A' P →
  midpoint B B' P →
  vectorSub B A = (-3, -1, -5) →
  B = (-4, 2, -8) :=
by
  intro h_mid_A h_mid_B h_AB
  have : B = (-4, 2, -8) := sorry
  exact this

end find_point_B_l235_235128


namespace algorithm_output_l235_235961

-- Define the algorithm as a function
def algorithm (A B : Nat) : Nat :=
  if B = 0 then A
  else algorithm B (A % B)

-- Define the main theorem that we need to prove
theorem algorithm_output (A B : Nat) (hA : A = 138) (hB : B = 22) : algorithm A B = 2 :=
by
  rw [hA, hB]
  sorry  -- This is where the proof would go

end algorithm_output_l235_235961


namespace min_square_area_l235_235683

-- Definitions to introduce the conditions
def rectangle (width height : ℕ) : Prop := width > 0 ∧ height > 0

def no_overlap (r1 r2 : (ℕ × ℕ)) (sq_side : ℕ) : Prop := 
  r1.fst + r2.fst <= sq_side ∧ r1.snd ≤ sq_side ∧ r2.snd ≤ sq_side

-- Statement of the theorem
theorem min_square_area :
  ∃ (sq_side : ℕ), sq_side^2 = 81 ∧ no_overlap (4, 3) (5, 4) sq_side :=
begin
  sorry
end

end min_square_area_l235_235683


namespace binomial_60_3_eq_34220_l235_235032

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235032


namespace trigonometric_identity_l235_235774

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := 
sorry

end trigonometric_identity_l235_235774


namespace problem_solution_l235_235240

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem problem_solution : a^10 + b^10 = 123 := sorry

end problem_solution_l235_235240


namespace binomial_60_3_l235_235023

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235023


namespace area_ratio_of_similar_polygons_incorrect_area_ratio_l235_235333

theorem area_ratio_of_similar_polygons (r : ℝ) (h_r_nonneg : 0 ≤ r) :
  ∀ (A B : Type) [metric_space A] [metric_space B] [is_similar A B r],
  (area A / area B) = r^2 :=
sorry

theorem incorrect_area_ratio :
  ¬∀ (A B : Type) [metric_space A] [metric_space B] [is_similar A B (r : ℝ)], 
  (area A / area B) = r :=
by 
  intro h 
  apply area_ratio_of_similar_polygons
  sorry

end area_ratio_of_similar_polygons_incorrect_area_ratio_l235_235333


namespace rsvp_percentage_l235_235537

def invitations := 200
def rsvp_rate := 0.9
def no_gift := 10
def thank_you_cards := 134

theorem rsvp_percentage :
  let rsvps := invitations * rsvp_rate in
  let total_who_brought_gift := thank_you_cards + no_gift in
  (total_who_brought_gift / rsvps) * 100 = 80 :=
by
  sorry

end rsvp_percentage_l235_235537


namespace encoded_value_decimal_l235_235352

-- Define the encoding and key conditions
def base_5_encoding : Type := list (fin 5)
def encoded_numbers : list (base_5_encoding → char) := [encode_VYZ, encode_VYX, encode_VVW]

-- Assume the encoding function from base-5 digits to characters in {V, W, X, Y, Z} is known.
constant encode : (fin 5) → char
constant encode_VYZ : base_5_encoding → char
constant encode_VYX : base_5_encoding → char
constant encode_VVW : base_5_encoding → char

-- Definitions of the encoded values for use in axioms
def VYZ : base_5_encoding := [(fin.of_nat 4), (fin.of_nat 1), (fin.of_nat 3)]
def VYX : base_5_encoding := [(fin.of_nat 4), (fin.of_nat 1), (fin.of_nat 4)]
def VVW : base_5_encoding := [(fin.of_nat 4), (fin.of_nat 4), (fin.of_nat 0)]

-- Prove the final encoded value
theorem encoded_value_decimal : (convert_to_decimal (encode [X, Y, Z]) ) = 108 :=
by
  sorry

-- Function to convert base-5 number (list of fin 5) to a decimal number (nat)
noncomputable def convert_to_decimal : (list (fin 5) → nat) :=
by 
  sorry

end encoded_value_decimal_l235_235352


namespace new_seq_is_arithmetic_l235_235702

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def new_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = a n + a (n + 3)

theorem new_seq_is_arithmetic (a : ℕ → ℝ) (d : ℝ) (b : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a d)
  (h_new_seq : new_sequence a b) :
  arithmetic_sequence b (2 * d) :=
sorry

end new_seq_is_arithmetic_l235_235702


namespace spell_AMC10_paths_l235_235183

theorem spell_AMC10_paths : 
  (∀ (A M C X : Type),
    (∃ (x : X), 
      (∀ (m1 m2 m3 m4 : M), 
        (∀ (c1 c2 c3 c4 : C), 
          (∀ (t1 t2 t3 t4 t5 : Nat), true)
        ) 
      )
    )
  ) → 
  ∃ (n : Nat), n = 80 :=
begin
  assume H,
  use 80,
  sorry
end

end spell_AMC10_paths_l235_235183


namespace markup_percentage_l235_235356

theorem markup_percentage 
  (CP : ℝ) (x : ℝ) (MP : ℝ) (SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (x / 100) * CP)
  (h3 : SP = MP - (10 / 100) * MP)
  (h4 : SP = CP + (35 / 100) * CP) :
  x = 50 :=
by sorry

end markup_percentage_l235_235356


namespace range_of_m_for_avg_function_l235_235751

def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) := ∃ x0, a < x0 ∧ x0 < b ∧ f x0 = (f b - f a) / (b - a)

theorem range_of_m_for_avg_function :
  is_average_value_function (λ x : ℝ, x^2 - m * x - 1) (-1) (1) → 0 < m ∧ m < 2 :=
begin
  sorry
end

end range_of_m_for_avg_function_l235_235751


namespace calc_a1_l235_235120

open Real

noncomputable def a_seq (a₁ : ℝ) : ℕ → ℝ
| 0     := a₁
| (n+1) := 2 ^ (logBase 2 a₁ - (n+1) + 1)

noncomputable def S_n (a₁ : ℝ) (n : ℕ) : ℝ :=
(∑ k in Finset.range n, a_seq a₁ k)

theorem calc_a1 (a₁ : ℝ) (h1 : ∀ n, 0 < a_seq a₁ n) (h2 : S_n a₁ 6 = 3 / 8) : a₁ = 4 / 21 :=
by sorry

end calc_a1_l235_235120


namespace distinct_triples_divisible_by_3_l235_235255

theorem distinct_triples_divisible_by_3 (n : ℕ) (hn : n ≥ 5) (nums : Fin n → ℕ) (distinct : ∀ i j, i ≠ j → nums i ≠ nums j) : 
  ∃ (triples : Fin (n*(n-1)*(n-2)/6) → (Fin n × Fin n × Fin n)), 
    (∀ t, let ⟨i, j, k⟩ := triples t in i < j ∧ j < k ∧ nums i + nums j + nums k ≡ 0 [MOD 3]) ∧ 
    (fintype.card {t | let ⟨i, j, k⟩ := triples t in i < j ∧ j < k}) ≥ n * (n - 1) * (n - 2) / 60 := 
sorry

end distinct_triples_divisible_by_3_l235_235255


namespace number_of_divisors_of_64n3_l235_235105

theorem number_of_divisors_of_64n3 (n : ℕ) (hn : n > 0) (h_divisors_150n2 : (150 * n^2).factorization.prod (\(p, e) => e + 1) = 150) : 
  (((64 * n^3).factorization.prod (\(p, e) => e + 1)) = 160) :=
sorry

end number_of_divisors_of_64n3_l235_235105


namespace train_length_200_04_l235_235716

-- Define the constants
def speed_kmh : ℝ := 60     -- speed in km/h
def time_seconds : ℕ := 12  -- time in seconds

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Convert speed to m/s
def speed_ms : ℝ := (speed_kmh * km_to_m) / hr_to_s

-- Define the length of the train in meters
def length_of_train : ℝ := speed_ms * time_seconds

-- The theorem to prove
theorem train_length_200_04 : length_of_train = 200.04 := by
  sorry

end train_length_200_04_l235_235716


namespace angle_AHB_112_l235_235530

-- Define the given conditions as constants or variables
variables (A B C H D E : Point) 
variable (α : Real)
variable (β : Real)
variable (ACB : Real)

-- Assumptions based on the conditions
variables (h1 : α = 58)
variables (h2 : β = 54)
variables (h3 : ACB = 180 - α - β)

-- Definition of orthocenter property
variable (h4 : ∃! H, IsOrthocenter H A B C)

theorem angle_AHB_112 :
  ∠AHB = 112 :=
by
  -- Use the given angles and orthocenter property
  have ACB_angle : ACB = 68 := by sorry
  show ∠AHB = 112 from calc
    ∠AHB = 180 - ACB : by sorry
    ... = 180 - 68   : by rw [ACB_angle]
    ... = 112        : by norm_num

end angle_AHB_112_l235_235530


namespace find_omega_l235_235800

open Real

def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x - π / 3)

theorem find_omega (m n : ℝ) (ω : ℝ) :
  f ω m = n →
  f ω (m + π) = n →
  abs n ≠ 1 →
  (∃ k : ℕ, 5 = k * round ((π / 2) / (2 * π / ω))) →
  ω = 4 :=
by sorry

end find_omega_l235_235800


namespace cube_surface_area_increase_l235_235327

theorem cube_surface_area_increase (L : ℝ) (hL : L > 0) : 
  let SA_original := 6 * L^2,
      SA_new := 6 * (1.30 * L)^2
  in
    (SA_new - SA_original) / SA_original * 100 = 69 :=
by
  -- This is where the actual proof would go, but it is omitted per instructions.
  sorry

end cube_surface_area_increase_l235_235327


namespace minimum_value_f_l235_235475

noncomputable def f (x a : ℝ) : ℝ := x^2 + (a + 8) * x + a^2 + a - 12

theorem minimum_value_f (a : ℝ) (n : ℕ) (h_neg : a < 0) (h_eq : f (a^2 - 4) a = f (2 * a - 8) a) :
  (n > 0) → ∃ n : ℕ, (frac (f n a - 4 * a) (n + 1) = 37 / 4) :=
by
  sorry

end minimum_value_f_l235_235475


namespace chocolate_bar_m_n_sum_l235_235349

theorem chocolate_bar_m_n_sum (n m : ℕ) 
  (checkerboard : ∀ (i j : ℕ), {i, j} ∈ (finset.range n).product (finset.range m) → (i + j) % 2 = 0 ∨ (i + j) % 2 = 1)
  (ian_more_max : ∀ (B W : ℕ), (B = W + 1) ∧ (B + W = n * m) → B = (13 * W) / 12) :
  m + n = 10 :=
by
  sorry

end chocolate_bar_m_n_sum_l235_235349


namespace baylor_final_amount_l235_235372

def CDA := 4000
def FCP := (1 / 2) * CDA
def SCP := FCP + (2 / 5) * FCP
def TCP := 2 * (FCP + SCP)
def FDA := CDA + FCP + SCP + TCP

theorem baylor_final_amount : FDA = 18400 := by
  sorry

end baylor_final_amount_l235_235372


namespace nicky_running_time_l235_235585

variable (v_C : ℕ) (v_N : ℕ) (head_start : ℕ) (d_N : ℕ)

theorem nicky_running_time (h1 : v_C = 5) 
    (h2 : v_N = 3) 
    (h3 : head_start = 20) 
    (h4 : d_N = 60) :
  let t := 30 in
  head_start + t = 50 :=
by
  -- Definitions previously proven or assumed as conditions
  have h_vc : v_C = 5 := h1
  have h_vn : v_N = 3 := h2
  have h_head : head_start = 20 := h3
  have h_dn : d_N = 60 := h4
  -- Direct calculation based on conditions
  let t := 30
  calc 
    head_start + t = 20 + 30 := by rw [h_head]
                ... = 50 := by norm_num

end nicky_running_time_l235_235585


namespace max_comic_books_l235_235975

namespace JasmineComicBooks

-- Conditions
def total_money : ℝ := 12.50
def comic_book_cost : ℝ := 1.15

-- Statement of the theorem
theorem max_comic_books (n : ℕ) (h : n * comic_book_cost ≤ total_money) : n ≤ 10 := by
  sorry

end JasmineComicBooks

end max_comic_books_l235_235975


namespace distinct_diagonals_in_nonagon_l235_235907

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235907


namespace sum_of_powers_sequence_l235_235237

theorem sum_of_powers_sequence (a b : ℝ) 
  (h₁ : a + b = 1)
  (h₂ : a^2 + b^2 = 3)
  (h₃ : a^3 + b^3 = 4)
  (h₄ : a^4 + b^4 = 7)
  (h₅ : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sum_of_powers_sequence_l235_235237


namespace work_completion_time_l235_235664

theorem work_completion_time 
  (a_b_together : ℚ := 12) 
  (a_alone : ℚ := 20) 
  (c_alone : ℚ := 30) :
  let b_alone := 1 / a_b_together - 1 / a_alone in
  let b_half_day := b_alone / 2 in
  let c_partial_day := 1 / c_alone / 3 in
  let a_one_day_work := 1 / a_alone in
  let total_work_per_day := a_one_day_work + b_half_day + c_partial_day in
  total_work_per_day * 12.857 ≈ 1 :=
by
  sorry

end work_completion_time_l235_235664


namespace binom_60_3_l235_235010

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235010


namespace train_speed_excluding_stoppages_l235_235756

theorem train_speed_excluding_stoppages 
    (speed_including_stoppages : ℕ)
    (stoppage_time_per_hour : ℕ)
    (running_time_per_hour : ℚ)
    (h1 : speed_including_stoppages = 36)
    (h2 : stoppage_time_per_hour = 20)
    (h3 : running_time_per_hour = 2 / 3) :
    ∃ S : ℕ, S = 54 :=
by 
  sorry

end train_speed_excluding_stoppages_l235_235756


namespace finite_solutions_iff_coprime_l235_235246

noncomputable def areCoprime (f g : ℂ[X][X]) : Prop := 
  ∀ d : ℂ[X][X], d ∣ f ∧ d ∣ g → d = 1

theorem finite_solutions_iff_coprime 
  {f g : ℝ[X][X]} (hf : f ≠ 0) (hg : g ≠ 0) : 
  (∃ S : set ℂ × ℂ, (S.finite ∧ ∀ ⟨x, y⟩ ∈ S, f.eval x y = 0 ∧ g.eval x y = 0)) 
  ↔ areCoprime f g :=
sorry

end finite_solutions_iff_coprime_l235_235246


namespace distinct_pairs_of_socks_l235_235298

/-- There are 3 pairs of socks, denoted as A, B, and C.
    Each pair consists of two socks, labeled as A1, A2, B1, B2, C1, and C2.
    We are to prove that the number of distinct pairs of socks that can be 
    formed when wearing socks from different pairs equals 3. -/
theorem distinct_pairs_of_socks : 
  let pairs := [(A1, A2), (B1, B2), (C1, C2)] in
  let socks := [A1, A2, B1, B2, C1, C2] in
  (∃ s1 s2 ∈ socks, (s1 ≠ s2) ∧ (∀ p ∈ pairs, s1 ∉ p ∨ s2 ∉ p)) → 3 := sorry

end distinct_pairs_of_socks_l235_235298


namespace chalk_breaking_probability_l235_235335

/-- Given you start with a single piece of chalk of length 1,
    and every second you choose a piece of chalk uniformly at random and break it in half,
    until you have 8 pieces of chalk,
    prove that the probability of all pieces having length 1/8 is 1/63. -/
theorem chalk_breaking_probability :
  let initial_pieces := 1
  let final_pieces := 8
  let total_breaks := final_pieces - initial_pieces
  let favorable_sequences := 20 * 4
  let total_sequences := Nat.factorial total_breaks
  (initial_pieces = 1) →
  (final_pieces = 8) →
  (total_breaks = 7) →
  (favorable_sequences = 80) →
  (total_sequences = 5040) →
  (favorable_sequences / total_sequences = 1 / 63) :=
by
  intros
  sorry

end chalk_breaking_probability_l235_235335


namespace graph_is_two_lines_l235_235399

theorem graph_is_two_lines : ∀ (x y : ℝ), (x ^ 2 - 25 * y ^ 2 - 20 * x + 100 = 0) ↔ (x = 10 + 5 * y ∨ x = 10 - 5 * y) := 
by 
  intro x y
  sorry

end graph_is_two_lines_l235_235399


namespace leak_empty_time_l235_235340

theorem leak_empty_time :
  let A := (1:ℝ)/6
  let AL := A - L
  ∀ L: ℝ, (A - L = (1:ℝ)/8) → (1 / L = 24) :=
by
  intros A AL L h
  sorry

end leak_empty_time_l235_235340


namespace binom_60_3_l235_235009

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235009


namespace rectangle_perimeter_l235_235705

variable (x : ℝ) (y : ℝ)

-- Definitions based on conditions
def area_of_rectangle : Prop := x * (x + 5) = 500
def side_length_relation : Prop := y = x + 5

-- The theorem we want to prove
theorem rectangle_perimeter (h_area : area_of_rectangle x) (h_side_length : side_length_relation x y) : 2 * (x + y) = 90 := by
  sorry

end rectangle_perimeter_l235_235705


namespace distinct_diagonals_in_convex_nonagon_l235_235883

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235883


namespace binom_60_3_eq_34220_l235_235042

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235042


namespace arithmetic_seq_middle_term_l235_235193

theorem arithmetic_seq_middle_term (d e g h : ℤ) (h_arith_seq : ∀ (a b c : ℤ), c - b = b - a) :
  let f := (11 + 61) / 2 in
  f = 36 :=
by
  let f := (11 + 61) / 2
  have h_arith_f : f = 36
  sorry

end arithmetic_seq_middle_term_l235_235193


namespace max_area_equilateral_triangle_in_rectangle_proof_l235_235291

-- Define the problem conditions
noncomputable def rectangle_EFGH := (12 : ℝ, 13 : ℝ)

-- Define the function to calculate the maximum area of an equilateral triangle inside the rectangle
noncomputable def max_area_equilateral_triangle_in_rectangle (width height : ℝ) : ℝ :=
  205 * Real.sqrt 3 - 468

-- The theorem states that, given the dimensions of the rectangle, the maximum possible area is as calculated
theorem max_area_equilateral_triangle_in_rectangle_proof :
  max_area_equilateral_triangle_in_rectangle 12 13 = 205 * Real.sqrt 3 - 468 :=
by
  sorry

end max_area_equilateral_triangle_in_rectangle_proof_l235_235291


namespace evaluate_triangle_l235_235750

def triangle_op (a b : Int) : Int :=
  a * b - a - b + 1

theorem evaluate_triangle :
  triangle_op (-3) 4 = -12 :=
by
  sorry

end evaluate_triangle_l235_235750


namespace area_ratio_none_of_these_l235_235278

theorem area_ratio_none_of_these (h r a : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) (a_pos : 0 < a) (h_square_a_square : h^2 > a^2) :
  ¬ (∃ ratio, ratio = (π * r / (h + r)) ∨
               ratio = (π * r^2 / (a + h)) ∨
               ratio = (π * a * r / (h + 2 * r)) ∨
               ratio = (π * r / (a + r))) :=
by sorry

end area_ratio_none_of_these_l235_235278


namespace incorrect_statement_l235_235330

-- Defining the conditions
def sum_of_interior_angles_of_quadrilateral (q : Type) [quadrilateral q] : angle_sum q = 360 := sorry
def sum_of_exterior_angles_of_quadrilateral (q : Type) [quadrilateral q] : exterior_angle_sum q = 360 := sorry
def area_ratio_of_similar_polygons {P Q : Type} [polygon P] [polygon Q] (h : similar P Q) : area P / area Q = similarity_ratio P Q := sorry
def symmetric_point_coordinates (P : Point) (origin : Point) : sym_point P origin = (-P.x, -P.y) := sorry
def median_of_triangle (T : Type) [triangle T] (M : median T) : parallel M.base M.third_side ∧ length M = 1/2 * length M.third_side := sorry

-- Theorem that needs to be proven
theorem incorrect_statement (q : Type) [quadrilateral q] (P Q : Type) [polygon P] [polygon Q] (T : Type) [triangle T] 
  (h : similar P Q) (M : median T) : 
  ¬ (area P / area Q = similarity_ratio P Q) := 
begin
  -- The actual proof was skipped
  sorry
end

end incorrect_statement_l235_235330


namespace sum_positive_k_values_l235_235275

theorem sum_positive_k_values :
  let is_integer_solution (x α β : ℤ) := (α * β = -24) ∧ (α + β = x)
  let k_values := {x : ℤ | ∃ α β : ℤ, is_integer_solution x α β}
  (∑ x in k_values ∩ (set_of (λ y, y > 0)), x) = 40 :=
by sorry

end sum_positive_k_values_l235_235275


namespace geometric_sequence_an_l235_235464

theorem geometric_sequence_an 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : a 1 * a 2 * a 3 = 64) 
  (h2 : ∀ n : ℕ, n > 0 → S (2 * n) = 5 * (∑ i in Finset.range n, a (2 * i + 1))) : 
  ∀ n : ℕ, n > 0 → a n = 4^(n - 1) := 
  by 
    -- The proof goes here
    sorry

end geometric_sequence_an_l235_235464


namespace sufficient_not_necessary_condition_l235_235454

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ (a : ℝ), a = 2 → (-(a) * (a / 4) = -1)) ∧ ∀ (a : ℝ), (-(a) * (a / 4) = -1 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_not_necessary_condition_l235_235454


namespace rent_increase_l235_235540

theorem rent_increase (monthly_rent_first_3_years : ℕ) (months_first_3_years : ℕ) 
  (total_paid : ℕ) (total_years : ℕ) (months_in_a_year : ℕ) (new_monthly_rent : ℕ) :
  monthly_rent_first_3_years * (months_in_a_year * 3) + new_monthly_rent * (months_in_a_year * (total_years - 3)) = total_paid →
  new_monthly_rent = 350 :=
by
  intros h
  -- proof development
  sorry

end rent_increase_l235_235540


namespace tangent_line_m_value_l235_235460

noncomputable def is_tangent (m : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (x^2 - 3 * real.log x) = -x + m ∧ (2 * x - 3 / x) = -1

theorem tangent_line_m_value :
  is_tangent 2 :=
by
  sorry

end tangent_line_m_value_l235_235460


namespace omega_range_monotonic_increasing_cos_l235_235779

theorem omega_range_monotonic_increasing_cos :
  ∀ (ω : ℝ), (ω > 0) ∧ (∀ x1 x2 : ℝ, (π/2 < x1 ∧ x1 < x2 ∧ x2 < π → cos (ω * x1 + π / 4) < cos (ω * x2 + π / 4)))
  → (3 / 2 ≤ ω ∧ ω ≤ 7 / 4) :=
by
  intros ω h
  sorry

end omega_range_monotonic_increasing_cos_l235_235779


namespace max_area_equilateral_triangle_in_rectangle_l235_235293

theorem max_area_equilateral_triangle_in_rectangle (a b : ℝ) (h_a : a = 12) (h_b : b = 13) :
  ∃ (T : ℝ), T = (117 * Real.sqrt 3) - 108 :=
by
  simp [h_a, h_b]
  use (117 * Real.sqrt 3) - 108
  sorry

end max_area_equilateral_triangle_in_rectangle_l235_235293


namespace main_roads_example_l235_235951

-- Condition declarations
def City := ℕ
def Road := City × City

-- Given a set of 100 cities
def cities : Finset City := {i ∈ Finset.range 100 | true}

-- Assume we have a set of roads connecting these cities without intersections 
-- Note that the verification of roads not intersecting is abstracted away
def roads (r : Finset Road) : Prop :=
  ∀ (c1 c2 : City), c1 ≠ c2 → c1 ∈ cities → c2 ∈ cities → 
  (∃ (r ∈ r), r = (c1, c2) ∧ ¬(r = (c2, c1))) ∧ (roads c1 c2 → ∃ (r1 r2 : Road), r1 ≠ r2 ∧ r1 ∈ r ∧ r2 ∈ r ∧ r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2)

-- You can travel from any city to any other city using the roads
-- Ensuring connectivity of the graph
def is_connected (r : Finset Road) : Prop := 
  ∀ (c1 c2 : City), c1 ∈ cities → c2 ∈ cities → reachable r c1 c2

-- Now formalize the main statement
theorem main_roads_example :
  ∃ (main_roads : Finset Road), 
    roads main_roads ∧ is_connected main_roads ∧ 
    ∀ (c : City), c ∈ cities → Odd (Finset.card (main_roads.filter (λ r, r.1 = c ∨ r.2 = c))) := sorry

end main_roads_example_l235_235951


namespace find_length_PQ_l235_235481

noncomputable def length_of_PQ (PQ PR : ℝ) (ST SU : ℝ) (angle_PQPR angle_STSU : ℝ) : ℝ :=
if (angle_PQPR = 120 ∧ angle_STSU = 120 ∧ PR / SU = 8 / 9) then 
  2 
else 
  0

theorem find_length_PQ :
  let PQ := 4 
  let PR := 8
  let ST := 9
  let SU := 18
  let PQ_crop := 2
  let angle_PQPR := 120
  let angle_STSU := 120
  length_of_PQ PQ PR ST SU angle_PQPR angle_STSU = PQ_crop :=
by
  sorry

end find_length_PQ_l235_235481


namespace process_box_shape_l235_235660

-- Define the type representing possible shapes.
inductive Shape
| Diamond
| Parallelogram
| Rectangle
| Triangle

open Shape

-- State the theorem
theorem process_box_shape : (∃ shape : Shape, shape = Rectangle) := 
by
  exists Rectangle
  rfl

end process_box_shape_l235_235660


namespace nonagon_diagonals_l235_235842

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235842


namespace digit_B_is_4_l235_235631

theorem digit_B_is_4 : ∃ B : ℕ, B = 4 ∧ let n := 40430 * 10 + B in
  (∃ p : ℕ, p.prime ∧ p < 10 ∧ n % p = 0) ∧
  (∀ p₁ p₂ : ℕ, p₁.prime ∧ p₂.prime ∧ p₁ < 10 ∧ p₂ < 10 → n % p₁ = 0 → n % p₂ = 0 → p₁ = p₂) :=
by
  sorry

end digit_B_is_4_l235_235631


namespace distinct_diagonals_convex_nonagon_l235_235855

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235855


namespace rank_friends_l235_235057

-- Definitions for the conditions
def EmmaNotShortest (h : (Int -> Int) -> Prop) : Prop := h Emma > h Fiona ∨ h Emma > h David
def FionaIsTallest (h : (Int -> Int) -> Prop) : Prop := h Fiona > h David ∧ h Fiona > h Emma
def DavidNotTallest (h : (Int -> Int) -> Prop) : Prop := ¬ (h David > h Fiona ∧ h David > h Emma)

-- Definitions for ranking friends
def friendsRanking (h : Int → Int) : List String :=
  if h David > h Emma ∧ h David > h Fiona then 
    if h Emma > h Fiona then ["David", "Emma", "Fiona"] else ["David", "Fiona", "Emma"]
  else if h Fiona > h David ∧ h Fiona > h Emma then 
    if h David > h Emma then ["Fiona", "David", "Emma"] else ["Fiona", "Emma", "David"]
  else ["Emma", "David", "Fiona"]

-- Main theorem to be proven
theorem rank_friends (h : Int → Int) : 
  (EmmaNotShortest h ∨ FionaIsTallest h ∨ DavidNotTallest h) ∧ 
  (EmmaNotShortest h ∨ FionaIsTallest h ∨ DavidNotTallest h → 
    friendsRanking h = ["David", "Emma", "Fiona"]) :=
sorry

end rank_friends_l235_235057


namespace math_problem_l235_235550

def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem math_problem :
  P (Q (P (Q (P (Q 2))))) = 1944 * real.sqrt 6 ^ (1/4) :=
by
  sorry

end math_problem_l235_235550


namespace palindromes_between_200_and_800_l235_235061

theorem palindromes_between_200_and_800 : 
  let p := λ n, 200 ≤ n ∧ n < 800 ∧ (Nat.digits 10 n = Nat.digits 10 n.reverse)
  in (∃ l : List ℕ, l.length = 60 ∧ ∀ n, n ∈ l → p n) :=
sorry

end palindromes_between_200_and_800_l235_235061


namespace ratio_arithmetic_sequence_last_digit_l235_235059

def is_ratio_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, n > 0 → (a (n + 2) * a n) = (a (n + 1) ^ 2) * d

theorem ratio_arithmetic_sequence_last_digit :
  ∃ a : ℕ → ℕ, is_ratio_arithmetic_sequence a 1 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (a 2009 / a 2006) % 10 = 6 :=
sorry

end ratio_arithmetic_sequence_last_digit_l235_235059


namespace min_possible_range_l235_235680

theorem min_possible_range (A B C : ℤ) : 
  (A + 15 ≤ C ∧ B + 25 ≤ C ∧ C ≤ A + 45) → C - A ≤ 45 :=
by
  intros h
  have h1 : A + 15 ≤ C := h.1
  have h2 : B + 25 ≤ C := h.2.1
  have h3 : C ≤ A + 45 := h.2.2
  sorry

end min_possible_range_l235_235680


namespace infinitely_many_in_desired_form_l235_235217

noncomputable def sequence_of_pos_ints (a : Nat → Nat) : Prop :=
  ∀ k, a k > 0 ∧ a k < a (k + 1)

theorem infinitely_many_in_desired_form
  (a : Nat → Nat) 
  (h_seq : sequence_of_pos_ints a) :
  ∃ x y : Nat, ∀ m : Nat, ∃ p q : Nat, p ≠ q ∧ a m = x * a p + y * a q :=
begin
  sorry
end

end infinitely_many_in_desired_form_l235_235217


namespace altitudes_sum_half_square_sides_l235_235533

theorem altitudes_sum_half_square_sides
  {A B C H D E F : Type}
  [incidence_geometry A B C D E F]
  (hAD : Altitude A D B C)
  (hBE : Altitude B E A C)
  (hCF : Altitude C F A B)
  (hH : Orthocenter H A B C)
  (a b c : ℝ)
  (hAH : Segment H A = a)
  (hBH : Segment H B = b)
  (hCH : Segment H C = c)
  (hAD_len : Segment A D = a)
  (hBE_len : Segment B E = b)
  (hCF_len : Segment C F = c) :
  (Segment H A * Segment A D) + 
  (Segment H B * Segment B E) + 
  (Segment H C * Segment C F) = 
  (1/2) * (a^2 + b^2 + c^2) := sorry

end altitudes_sum_half_square_sides_l235_235533


namespace train_length_is_correct_l235_235720

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end train_length_is_correct_l235_235720


namespace a_plus_b_l235_235169

theorem a_plus_b (x : ℝ) (a b : ℝ) (h1 : x^2 + 4 * x + 4 / x + 1 / x^2 = 35)
  (h2 : a ∈ ℤ ∧ b ∈ ℤ  ∧ 0 < a ∧ 0 < b ∧ x = a + real.sqrt b) : a + b = 23 :=
sorry

end a_plus_b_l235_235169


namespace min_value_of_f_range_of_a_inequality_l235_235777

section
variable (x : ℝ) (a : ℝ)
def f (x : ℝ) : ℝ := 2 * x * Real.log x
def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3
def u (x : ℝ) : ℝ := 2 * (x / Real.exp x - 2 / Real.exp 1)

-- Proof problem for part (I)
theorem min_value_of_f : ∀ x ∈ Set.Ioi 0, f x ≥ -2 / Real.exp 1 :=
begin
  sorry, -- insert proof here
end

-- Proof problem for part (II)
theorem range_of_a : ∃ x ∈ Set.Ioi 0, ∀ a, f x ≤ g x a → a ≥ 4 :=
begin
  sorry, -- insert proof here
end

-- Proof problem for part (III)
theorem inequality : ∀ x ∈ Set.Ioi 0, f x > u x :=
begin
  sorry, -- insert proof here
end
end

end min_value_of_f_range_of_a_inequality_l235_235777


namespace difference_of_integers_l235_235592

theorem difference_of_integers :
  ∀ (x y : ℤ), (x = 32) → (y = 5*x + 2) → (y - x = 130) :=
by
  intros x y hx hy
  sorry

end difference_of_integers_l235_235592


namespace math_problem_l235_235551

def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem math_problem :
  P (Q (P (Q (P (Q 2))))) = 1944 * real.sqrt 6 ^ (1/4) :=
by
  sorry

end math_problem_l235_235551


namespace sequence_integer_proof_l235_235626

theorem sequence_integer_proof (a : ℕ+ → ℤ) (h : ∀ n : ℕ+, a (n + 1) = (n + 2 : ℕ) / (n : ℕ) * (a n - 1)) 
  (h0 : a 1 ∈ ℤ) : ∀ n : ℕ+, a n ∈ ℤ := 
sorry

end sequence_integer_proof_l235_235626


namespace blend_pieces_eq_two_l235_235232

variable (n_silk n_cashmere total_pieces : ℕ)

def luther_line := n_silk = 10 ∧ n_cashmere = n_silk / 2 ∧ total_pieces = 13

theorem blend_pieces_eq_two : luther_line n_silk n_cashmere total_pieces → (n_cashmere - (total_pieces - n_silk) = 2) :=
by
  intros
  sorry

end blend_pieces_eq_two_l235_235232


namespace sammy_problems_left_l235_235248

theorem sammy_problems_left (total_problems : ℕ) (completed_problems : ℕ) (remaining_problems : ℕ) : 
  total_problems = 9 ∧ completed_problems = 2 → remaining_problems = 7 :=
by
  intro h
  cases h with h_total h_completed
  have total_eq : total_problems = 9 := h_total
  have completed_eq : completed_problems = 2 := h_completed
  have remaining_eq : remaining_problems = total_problems - completed_problems := sorry
  have remaining_eq_simplified : remaining_problems = 7 := by
    rw [total_eq, completed_eq]
    simp
  exact remaining_eq_simplified

end sammy_problems_left_l235_235248


namespace seq_periodic_if_angle_condition_l235_235480

-- Define the points A, B, and C
variables (A B C : Type) [MetricSpace A]

-- Assume C is on the perpendicular bisector of AB
def on_perpendicular_bisector (A B C : Type) [MetricSpace A] : Prop :=
  dist A C = dist B C

-- Define the sequence C_n
inductive seq_Cn : Type
| base : C → seq_Cn
| next : seq_Cn → seq_Cn

-- Define the conditions for periodicity
def periodic (C : Type) [MetricSpace C] (C_seq : seq_Cn) : Prop :=
  ∃ r s : ℕ, gcd r s = 1 ∧ ¬ (∃ k : ℕ, s = 2^k) ∧
  ∀ n : ℕ, angle A C_seq B = (180 * r / s : ℝ) → C_seq (n + 1) = C_seq n

-- The actual theorem statement
theorem seq_periodic_if_angle_condition
  (A B C : Type) [MetricSpace A] (C_seq : seq_Cn)
  (h_bisector : on_perpendicular_bisector A B C)
  (h_angle_cond : ∃ r s : ℕ, gcd r s = 1 ∧ ¬ (∃ k : ℕ, s = 2^k) ∧ angle A C B = 180 * r / s) :
  periodic C C_seq :=
sorry

end seq_periodic_if_angle_condition_l235_235480


namespace abs_diff_of_distances_is_zero_l235_235280

noncomputable def dist (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem abs_diff_of_distances_is_zero :
  let Q : ℝ × ℝ := (2, 1)
  let C : ℝ × ℝ := ((21 - real.sqrt 57) / 8, (13 - real.sqrt 57) / 4)
  let D : ℝ × ℝ := ((21 + real.sqrt 57) / 8, (13 + real.sqrt 57) / 4) in
  |dist Q C - dist Q D| = 0 :=
by
  sorry

end abs_diff_of_distances_is_zero_l235_235280


namespace total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l235_235687

-- Define the conditions
def number_of_bags : ℕ := 9
def vitamins_per_bag : ℚ := 0.2

-- Define the total vitamins in the box
def total_vitamins_in_box : ℚ := number_of_bags * vitamins_per_bag

-- Define the vitamins intake by drinking half a bag
def vitamins_per_half_bag : ℚ := vitamins_per_bag / 2

-- Prove that the total grams of vitamins in the box is 1.8 grams
theorem total_vitamins_in_box_correct : total_vitamins_in_box = 1.8 := by
  sorry

-- Prove that the vitamins intake by drinking half a bag is 0.1 grams
theorem vitamins_per_half_bag_correct : vitamins_per_half_bag = 0.1 := by
  sorry

end total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l235_235687


namespace distinct_diagonals_in_nonagon_l235_235900

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235900


namespace part_one_part_two_part_three_l235_235523

variables {R : Type*} [Real R]

def vector2 := R × R

def origin : vector2 := (0, 0)
def A : vector2 := (1, 1)
def B : vector2 := (2, 0)

def magnitude (v : vector2) : R := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : vector2) : R := v.1 * w.1 + v.2 * w.2

def angle_between (v w : vector2) : R :=
arccos (dot_product v w / (magnitude v * magnitude w))

def orthogonal (v w : vector2) : Prop := dot_product v w = 0

def distance_of_sum OC : R :=
magnitude ( (3 + OC.1, 1 + OC.2) )

theorem part_one :
  angle_between A B = π / 4 :=
sorry

theorem part_two (OC : vector2) (h₁ : magnitude OC = 1) (h₂ : orthogonal OC A) :
  OC = (real.sqrt 2 / 2, real.sqrt 2 / 2) ∨ 
  OC = (-real.sqrt 2 / 2, -real.sqrt 2 / 2) :=
sorry

theorem part_three (OC : vector2) (h₁ : magnitude OC = 1) (h₂ : orthogonal OC A) :
  ∃ m M : R, distance_of_sum OC = M ∧ distance_of_sum OC = m ∧
  m = real.sqrt 10 - 1 ∧ M = real.sqrt 10 + 1 :=
sorry

end part_one_part_two_part_three_l235_235523


namespace density_ratio_of_large_cube_l235_235696

theorem density_ratio_of_large_cube 
  (V0 m0 : ℝ) (initial_density replacement_density: ℝ)
  (initial_mass final_mass : ℝ) (V_total : ℝ) 
  (h1 : initial_density = m0 / V0)
  (h2 : replacement_density = 2 * initial_density)
  (h3 : initial_mass = 8 * m0)
  (h4 : final_mass = 6 * m0 + 2 * (2 * m0))
  (h5 : V_total = 8 * V0) :
  initial_density / (final_mass / V_total) = 0.8 :=
sorry

end density_ratio_of_large_cube_l235_235696


namespace volume_P3_is_17_over_8_volume_P3_m_plus_n_is_27_l235_235783

theorem volume_P3_is_17_over_8 :
  let V : ℕ → ℚ := λ n =>
    Nat.recOn n 1 (λ i Vi, Vi + 4 ^ i / 8 ^ (i + 1))
  V 3 = 17 / 8 := by
  sorry

theorem volume_P3_m_plus_n_is_27 :
  let V : ℕ → ℚ := λ n =>
    Nat.recOn n 1 (λ i Vi, Vi + 4 ^ i / 8 ^ (i + 1))
  let m := 17
  let n := 8
  m + n = 25 := by
  sorry

end volume_P3_is_17_over_8_volume_P3_m_plus_n_is_27_l235_235783


namespace loaves_on_friday_l235_235615

theorem loaves_on_friday
  (bread_wed : ℕ)
  (bread_thu : ℕ)
  (bread_sat : ℕ)
  (bread_sun : ℕ)
  (bread_mon : ℕ)
  (inc_wed_thu : bread_thu - bread_wed = 2)
  (inc_sat_sun : bread_sun - bread_sat = 5)
  (inc_sun_mon : bread_mon - bread_sun = 6)
  (pattern : ∀ n : ℕ, bread_wed + (2 + n) + n = bread_thu + n)
  : bread_thu + 3 = 10 := 
sorry

end loaves_on_friday_l235_235615


namespace min_dot_product_proof_l235_235190

def rectangle (A B C D : Point ℝ) :=
  A.x = 0 ∧ A.y = 0 ∧ B.x = 2 ∧ B.y = 0 ∧ D.x = 0 ∧ D.y = 1

noncomputable def min_dot_product (t : ℝ) : ℝ :=
  let PA := (t, 1 : ℝ)
  let PQ := (2 - t, -t - 1 : ℝ)
  (PA.1 * PQ.1 + PA.2 * PQ.2)

theorem min_dot_product_proof
  (A B C D P Q : Point ℝ)
  (P_on_DC : ∀ t, 0 ≤ t ∧ t ≤ 2 → P = mk_point t 1)
  (Q_on_extended_CB : ∀ t, Q = mk_point 2 (- t))
  (DP_BQ_eq : ∀ t, |P - D| = |Q - B|):
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧ min_dot_product t = 3 / 4 := 
sorry

end min_dot_product_proof_l235_235190


namespace initial_candies_tracy_l235_235639

theorem initial_candies_tracy (x : ℕ) (hx1 : ∃ k : ℕ, k * 4 = x)
  (hx2 : 7 ≤ x / 2 - 24 ∧ x / 2 - 24 ≤ 11) : 
  x = 72 ∨ x = 76 :=
by
  have div_by_4 := hx1,
  have inequality_bounds := hx2,
  sorry

end initial_candies_tracy_l235_235639


namespace max_four_prime_numbers_sum_prime_l235_235818

/-- 
  The maximum number of different (positive) prime numbers such that 
  the sum of any three of them is also a prime number is 4.
-/
theorem max_four_prime_numbers_sum_prime :
  ∃ (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p) ∧
    (∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → Nat.Prime (a + b + c)) ∧
    (Finset.card S ≤ 4)
:=
sorry

end max_four_prime_numbers_sum_prime_l235_235818


namespace smallest_a_for_polynomial_roots_in_interval_l235_235134

theorem smallest_a_for_polynomial_roots_in_interval
    (a : ℕ)
    (b c : ℤ)
    (f : ℝ → ℝ)
    (h_poly : ∀ x : ℝ, f x = a * x^2 + b * x + c)
    (x1 x2 : ℝ)
    (h_roots : f x1 = 0 ∧ f x2 = 0)
    (h_distinct : x1 ≠ x2)
    (h_interval : 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1)
    : a = 5 :=
begin
  sorry
end

end smallest_a_for_polynomial_roots_in_interval_l235_235134


namespace no_all_nine_odd_l235_235952

theorem no_all_nine_odd
  (a1 a2 a3 a4 a5 b1 b2 b3 b4 : ℤ)
  (h1 : a1 % 2 = 1) (h2 : a2 % 2 = 1) (h3 : a3 % 2 = 1)
  (h4 : a4 % 2 = 1) (h5 : a5 % 2 = 1) (h6 : b1 % 2 = 1)
  (h7 : b2 % 2 = 1) (h8 : b3 % 2 = 1) (h9 : b4 % 2 = 1)
  (sum_eq : a1 + a2 + a3 + a4 + a5 = b1 + b2 + b3 + b4) : 
  false :=
sorry

end no_all_nine_odd_l235_235952


namespace tangent_point_x_coordinate_l235_235942

-- Define the function representing the curve.
def curve (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of the curve.
def derivative (x : ℝ) : ℝ := 2 * x

-- The statement to be proved.
theorem tangent_point_x_coordinate (x : ℝ) (h : derivative x = 4) : x = 2 :=
sorry

end tangent_point_x_coordinate_l235_235942


namespace largest_possible_number_of_pencils_in_a_box_l235_235588

/-- Olivia bought 48 pencils -/
def olivia_pencils : ℕ := 48
/-- Noah bought 60 pencils -/
def noah_pencils : ℕ := 60
/-- Liam bought 72 pencils -/
def liam_pencils : ℕ := 72

/-- The GCD of the number of pencils bought by Olivia, Noah, and Liam is 12 -/
theorem largest_possible_number_of_pencils_in_a_box :
  gcd olivia_pencils (gcd noah_pencils liam_pencils) = 12 :=
by {
  sorry
}

end largest_possible_number_of_pencils_in_a_box_l235_235588


namespace inscribed_circle_radius_correct_l235_235637

noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let recip_r := (1 / a) + (1 / b) + (1 / c) + 2 * real.sqrt((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c)))
  1 / recip_r

theorem inscribed_circle_radius_correct : inscribed_circle_radius 3 6 18 = 18 / 19 :=
by
  sorry

end inscribed_circle_radius_correct_l235_235637


namespace account_balance_after_one_year_l235_235934

-- define constants and conditions
def principal : ℝ := 800
def annual_rate : ℝ := 0.10
def compounding_per_year : ℕ := 2
def years : ℕ := 1

-- define the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- state the theorem
theorem account_balance_after_one_year : 
  compound_interest principal annual_rate compounding_per_year years = 882 :=
by 
  sorry

end account_balance_after_one_year_l235_235934


namespace player_B_should_choose_6_to_win_l235_235314

-- Define the set of numbers and initial conditions
def numbers_set : set ℕ := {n | n ∈ set.Icc 1 17}

-- Define the initial state where Player A has chosen the number 8
def initial_picks : set ℕ := {8}

-- Define the conditions for a valid pick
def is_valid_pick (pick : ℕ) (picks : set ℕ) : Prop :=
  pick ∈ numbers_set ∧ 
  (∀ p ∈ picks, pick ≠ p ∧ pick ≠ 2 * p ∧ 2 * pick ≠ p)

-- Define a helper function to get the prohibited picks from a set of picks
def prohibited_picks (picks : set ℕ) : set ℕ :=
  {p | ∃ q ∈ picks, p = q ∨ p = 2 * q ∨ q = 2 * p}

-- Define the remaining numbers after the initial picks
def remaining_numbers (initial : set ℕ) : set ℕ :=
  numbers_set \ prohibited_picks initial

theorem player_B_should_choose_6_to_win :
  is_valid_pick 6 (initial_picks) ∧
  ∀ n,
  n ∈ remaining_numbers (initial_picks ∪ {6}) →
  ∃ m, m ∈ remaining_numbers (initial_picks ∪ {6} ∪ {n}) ∧ ¬ is_valid_pick m (initial_picks ∪ {6} ∪ {n}) :=
begin
  sorry
end

end player_B_should_choose_6_to_win_l235_235314


namespace not_prime_3999991_l235_235679

   theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
   by
     -- Provide the factorization proof
     sorry
   
end not_prime_3999991_l235_235679


namespace cautioned_players_received_one_yellow_card_l235_235004

theorem cautioned_players_received_one_yellow_card :
  ∀ (total_players cautioned_players red_cards yellow_cards_per_red_card : ℕ),
    total_players = 11 →
    (total_players - cautioned_players) = 5 →
    red_cards = 3 →
    yellow_cards_per_red_card = 2 →
    ∃ yellow_cards_per_cautioned_player : ℕ,
      yellow_cards_per_cautioned_player = (red_cards * yellow_cards_per_red_card) / cautioned_players ∧
      yellow_cards_per_cautioned_player = 1 :=
begin
  intros total_players cautioned_players red_cards yellow_cards_per_red_card
         h_total_players h_non_cautioned h_red_cards h_yellow_cards_per_red_card,
  use (red_cards * yellow_cards_per_red_card) / cautioned_players,
  split,
  { simp [h_red_cards, h_yellow_cards_per_red_card, h_non_cautioned],
    norm_num,
    simp [h_total_players],
    norm_num },
  { norm_num },
end

end cautioned_players_received_one_yellow_card_l235_235004


namespace find_k_l235_235803

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 + k * x - 7

-- Define the given condition f(5) - g(5) = 20
def condition (k : ℝ) : Prop := f 5 - g 5 k = 20

-- The theorem to prove that k = 16.4
theorem find_k : ∃ k : ℝ, condition k ∧ k = 16.4 :=
by
  sorry

end find_k_l235_235803


namespace smaller_circle_radius_l235_235350

noncomputable def radius_of_smaller_circle (A1 A2 : ℝ) (r2 : ℝ) (radius_larger : ℝ) :=
  A1 = radius_larger * radius_larger * Real.pi ∧
  A1 + A2 = r2 * r2 * Real.pi ∧
  (r2 * r2 * Real.pi - A1) = (A1 + (r2 * r2 * Real.pi)) / 2

theorem smaller_circle_radius :
  ∀ (A1 A2 : ℝ),
    radius_of_smaller_circle A1 A2 16 Real.pi (4) →
    sqrt (A1 / Real.pi) = (4 * sqrt 3 / 3) :=
by
  sorry

end smaller_circle_radius_l235_235350


namespace medians_concurrent_l235_235257

/--
For any triangle ABC, there exists a point G, known as the centroid, such that
the sum of the vectors from G to each of the vertices A, B, and C is the zero vector.
-/
theorem medians_concurrent 
  (A B C : ℝ×ℝ) : 
  ∃ G : ℝ×ℝ, (G -ᵥ A) + (G -ᵥ B) + (G -ᵥ C) = (0, 0) := 
by 
  -- proof will go here
  sorry 

end medians_concurrent_l235_235257


namespace smallest_integer_N_l235_235225

theorem smallest_integer_N (k : ℕ) (hk : 1 ≤ k) :
  ∃ (a : Fin (2*k + 1) → ℕ), 
    (∀ i, 1 ≤ a i) ∧ 
    Finset.sum Finset.univ (λ i, a i) ≥ 2 * k^3 + 3 * k^2 + k ∧
    ∀ (s : Finset (Fin (2*k + 1))), s.card = k → Finset.sum s (λ i, a i) ≤ (2 * k^3 + 3 * k^2 + k) / 2 :=
sorry

end smallest_integer_N_l235_235225


namespace incircle_radius_l235_235216

-- Definitions for the lengths and properties of triangle DEF
variables (DE EF DF JF : ℝ)
variables (is_isosceles : DE = EF)
variables (DE_length : DE = 40)
variables (EF_length : EF = 40)
variables (DF_length : DF = 50)
variables (JF_length : JF = 26)

-- Statement of the theorem
theorem incircle_radius (r : ℝ) : r = 26 :=
by {
  -- Conditions printing for clarity in theorem environment
  have h1 : DE = EF := is_isosceles,
  have h2 : DE = 40 := DE_length,
  have h3 : EF = 40 := EF_length,
  have h4 : DF = 50 := DF_length,
  have h5 : JF = 26 := JF_length,

  -- State the goal: proving r = 26
  -- Based on given conditions
  sorry,
}

end incircle_radius_l235_235216


namespace correct_calculation_l235_235656

theorem correct_calculation : sqrt 27 / sqrt 3 = 3 :=
by
  sorry

end correct_calculation_l235_235656


namespace g_neg2_g_pos2_l235_235572

def g (x : ℝ) : ℝ :=
if x ≤ 1 then -3 * x + 4 else 4 * x - 6

theorem g_neg2 : g (-2) = 10 :=
by 
  unfold g
  simp
  sorry

theorem g_pos2 : g (2) = 2 :=
by 
  unfold g
  simp
  sorry

end g_neg2_g_pos2_l235_235572


namespace nonagon_diagonals_count_l235_235895

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235895


namespace h_odd_at_minus_one_range_of_a_for_two_distinct_roots_l235_235470

noncomputable def f (x : ℝ) : ℝ := log_base 3 ((x - 1) / (x + 1))
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -2*a*x + a + 1
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f(x) + g(x, a)

-- Question (I)
theorem h_odd_at_minus_one : ∀ (x : ℝ), h x (-1) = - h (-x) (-1) := by
sorry

-- Question (II)
theorem range_of_a_for_two_distinct_roots : ∀ (a : ℝ), (0 < a ∧ a < 1) ↔ ∀ y, (f y = log_base 3 (g y a)) → y ≠ x := 
by
sorry

end h_odd_at_minus_one_range_of_a_for_two_distinct_roots_l235_235470


namespace charles_travel_time_l235_235494

variable (D S : ℝ)
variable (hD : D = 6) (hS : S = 3)

theorem charles_travel_time : (D / S) = 2 :=
by
  rw [hD, hS]
  norm_num
  sorry

end charles_travel_time_l235_235494


namespace mary_reg_rate_l235_235234

theorem mary_reg_rate:
  ∃ (R : ℝ), 
    (∀ O, 20 * R + 1.25 * R * O = 410 → O ≤ 25) ∧
    (20 * R + 1.25 * R * 25 = 410) ∧
    R = 8 :=
begin
  sorry
end

end mary_reg_rate_l235_235234


namespace complex_modulus_l235_235780

theorem complex_modulus
  (z : ℂ)
  (h : (z * complex.I) / (z - complex.I) = 1) :
  complex.abs z = real.sqrt (2) / 2 :=
sorry

end complex_modulus_l235_235780


namespace find_integer_n_l235_235398

noncomputable def cubic_expr_is_pure_integer (n : ℤ) : Prop :=
  (729 * n ^ 6 - 540 * n ^ 4 + 240 * n ^ 2 - 64 : ℂ).im = 0

theorem find_integer_n :
  ∃! n : ℤ, cubic_expr_is_pure_integer n := 
sorry

end find_integer_n_l235_235398


namespace matrix_pow_2018_l235_235392

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![2^2017, 2^2017], ![2^2017, 2^2017]]

theorem matrix_pow_2018 : 
  ∀ n ≥ 2, (A^n = (fun k => ![![2^(k-1), 2^(k-1)], ![2^(k-1), 2^(k-1)]]) n) → A^2018 = B :=
by
  intro n hn H
  have H : A^2018 = ![![2^2017, 2^2017], ![2^2017, 2^2017]], from H 2018 (by norm_num)
  rw H
  sorry

end matrix_pow_2018_l235_235392


namespace number_of_diagonals_in_nonagon_l235_235822

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235822


namespace quadratic_polynomial_roots_l235_235287

theorem quadratic_polynomial_roots (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (h_roots: ∀ x, x^2 + 4 * x + 8 = 0 → (x = a ∨ x = b)) :
  ∃ p : ℝ[X], p = 8 * X^2 + 4 * X + 1 ∧ (p.eval (1 / a) = 0 ∧ p.eval (1 / b) = 0) :=
by
  sorry

end quadratic_polynomial_roots_l235_235287


namespace total_protest_days_l235_235208

-- Definitions for the problem conditions
def first_protest_days : ℕ := 4
def second_protest_days : ℕ := first_protest_days + (first_protest_days / 4)

-- The proof statement
theorem total_protest_days : first_protest_days + second_protest_days = 9 := sorry

end total_protest_days_l235_235208


namespace finite_set_of_nonpositive_reals_l235_235411

-- Definition of the problem's conditions and question
def satisfies_property (X : Finset ℝ) : Prop :=
  ∀ x ∈ X, x + abs x ∈ X

-- The theorem statement of the mathematically equivalent proof problem
theorem finite_set_of_nonpositive_reals (X : Finset ℝ) (h_nonempty : X.nonempty)
  (h_property: satisfies_property X) : 
  ∃ Y ⊆ X, Y.nonempty ∧ ∀ y ∈ Y, y ≤ 0 ∧ (0 ∈ Y) :=
sorry

end finite_set_of_nonpositive_reals_l235_235411


namespace hyperbola_eccentricity_range_l235_235699

theorem hyperbola_eccentricity_range
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b = Real.sqrt (c^2 - a^2))
  (slope1 : slope = 1 → intersects_both_branches = True)
  (slope3 : slope = 3 → intersects_right_branch_twice = True) :
  ∃ e : ℝ, e = c / a ∧ sqrt 2 < e ∧ e < sqrt 10 := 
sorry

end hyperbola_eccentricity_range_l235_235699


namespace find_area_by_median_l235_235172

-- Define the conditions
def median_set := [1, 2, 0, a, 8, 7, 6, 5]
def median (s : List ℝ) := (s.nth (s.length / 2 - 1) + s.nth (s.length / 2)) / 2

-- Define a function to find the intersection points
def intersection_points (a : ℝ) := 
  if a = 3 then [(0,0), (3,9)] else []

-- Define a function to calculate the area using definite integral
def area (a : ℝ) : ℝ :=
  ∫ x in 0..3, (a*x - x^2)

-- The Lean 4 statement
theorem find_area_by_median (a : ℝ) (h : median [1, 2, 0, a, 8, 7, 6, 5] = 4) : area 3 = 9 / 2 := by 
  sorry

end find_area_by_median_l235_235172


namespace find_x_l235_235482

variables (x : ℝ)

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (2, x)
def sum_vectors : ℝ × ℝ := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
def scalar_mult_4_vectors_b : ℝ × ℝ := (4 * vector_b.1, 4 * vector_b.2)
def scalar_mult_2_vectors_a : ℝ × ℝ := (2 * vector_a.1, 2 * vector_a.2)
def vector_operation : ℝ × ℝ := (scalar_mult_4_vectors_b.1 - scalar_mult_2_vectors_a.1, scalar_mult_4_vectors_b.2 - scalar_mult_2_vectors_a.2)

theorem find_x (h : sum_vectors = vector_operation) : x = 2 := 
sorry

end find_x_l235_235482


namespace largest_x_fraction_l235_235760

theorem largest_x_fraction (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 := by
  sorry

end largest_x_fraction_l235_235760


namespace joan_books_l235_235207

theorem joan_books : 
  (33 - 26 = 7) :=
by
  sorry

end joan_books_l235_235207


namespace inscribed_polygon_property_proof_l235_235285

noncomputable def inscribed_polygon_property (n : ℕ) : Prop :=
  if odd n then 
    (∀ (A : Fin 2n → ℝ × ℝ), 
      (is_cyclic_polygon A) ∧
      (∀ i j : Fin 2n, i ≠ j → (is_parallel (A i) (A j) ∨ (i, j) = remaining_parallel_pair)) →
    (is_parallel (A remaining_parallel_pair.fst) (A remaining_parallel_pair.snd)))
  else 
    (∀ (A : Fin 2n → ℝ × ℝ), 
      (is_cyclic_polygon A) ∧
      (∀ i j : Fin 2n, i ≠ j → (is_parallel (A i) (A j) ∨ (i, j) = remaining_equal_length_pair)) →
    (is_equal_length (A remaining_equal_length_pair.fst) (A remaining_equal_length_pair.snd)))

axiom base_case (n : ℕ) : inscribed_polygon_property (2 * n)

theorem inscribed_polygon_property_proof (n : ℕ) : inscribed_polygon_property n :=
begin
  induction n with n ih,
  sorry,
  sorry
end

end inscribed_polygon_property_proof_l235_235285


namespace braking_performance_l235_235404

-- Definition of the relationship between braking distance and speed
def braking_distance (v : ℕ) : ℕ := 25 * v / 100

-- Theorem statement proving the braking distance at 60 km/h and the speed at 32 meters
theorem braking_performance :
  ∀ (v : ℕ),
  (v ≤ 140) →
  (braking_distance v = 25 * v / 100) ∧
  (braking_distance 60 = 15) ∧
  (braking_distance v = 32 → v = 128) ∧
  (128 > 120) :=
by
  intros,
  sorry

end braking_performance_l235_235404


namespace binomial_sum_to_220_l235_235070

open Nat

def binomial_coeff (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binomial_sum_to_220 :
  binomial_coeff 2 2 + binomial_coeff 3 2 + binomial_coeff 4 2 + binomial_coeff 5 2 +
  binomial_coeff 6 2 + binomial_coeff 7 2 + binomial_coeff 8 2 + binomial_coeff 9 2 +
  binomial_coeff 10 2 + binomial_coeff 11 2 = 220 :=
by
  /- Proof goes here, use the computed value of combinations -/
  sorry

end binomial_sum_to_220_l235_235070


namespace set_5_7_10_not_prism_diagonals_l235_235657

-- Definitions required
def is_external_diagonal_set_of_prism (s : set ℝ) : Prop :=
  ∃ a b c : ℝ,
  s = {real.sqrt (a^2 + b^2), real.sqrt (b^2 + c^2), real.sqrt (a^2 + c^2)}

-- The main theorem
theorem set_5_7_10_not_prism_diagonals :
  ¬ is_external_diagonal_set_of_prism {5, 7, 10} :=
by {
  -- This part is intentionally left as a placeholder for proof
  sorry
}

end set_5_7_10_not_prism_diagonals_l235_235657


namespace valid_ordered_pairs_count_l235_235204

variable {a b d n : ℕ}
variable {Jane_current_age : ℕ}
variable {Dick_current_age : ℕ}

-- Setting conditions based on the problem statement
axiom Jane_age_current : Jane_current_age = 30
axiom Dick_older_than_Jane : Dick_current_age > Jane_current_age
axiom positive_integer_n : n > 0
axiom Jane_age_future : ∀ (n : ℕ), n > 0 → (Jane_current_age + n).natDigits.length = 2
axiom Dick_age_future : ∀ (n : ℕ), n > 0 → (Dick_current_age + n).natDigits.length = 2
axiom age_interchange_condition : ∀ (n : ℕ), n > 0 → (Jane_current_age + n = 10 * (Dick_current_age + n) % 10 + (Dick_current_age + n) / 10)

-- Prove that the number of valid (d, n) pairs such that Jane's age obtained by interchanging the
-- digits of Dick's age in n years from now is 26.
theorem valid_ordered_pairs_count : ∃ (pairs : set (ℕ × ℕ)), pairs.size = 26 :=
begin
  sorry
end

end valid_ordered_pairs_count_l235_235204


namespace salary_after_increase_is_correct_l235_235254

namespace SalaryIncrease

-- Define necessary constants
def salary : ℕ := 30000
def increase_percentage : ℝ := 0.10

-- Theorem statement
theorem salary_after_increase_is_correct :
  let new_salary := ℕ := (salary : ℝ) * (1 + increase_percentage) in
  new_salary = 33000 :=
by
  sorry

end SalaryIncrease

end salary_after_increase_is_correct_l235_235254


namespace find_AQ_AO_AR_l235_235343

section proof_problem

variables {A B C D E F O P Q R : Type}
variables [regular_hexagon ABCDEF]
variables [perpendicular AP A (line_through E extends F)]
variables [perpendicular AQ A (line_through E extends D)]
variables [perpendicular AR A (line_through B extends C)]
variables (center : is_center O ABCDEF)
variables (OP_length : distance O P = 2)

theorem find_AQ_AO_AR : AO + AQ + AR = (12 * real.sqrt 3) - 2 := sorry
end proof_problem

end find_AQ_AO_AR_l235_235343


namespace solve_system_of_equations_l235_235261

theorem solve_system_of_equations :
  ∃ x y : ℝ, (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ x = 0.5 ∧ y = 0.6 :=
by
  sorry -- Proof to be completed

end solve_system_of_equations_l235_235261


namespace student_score_l235_235522

theorem student_score (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 150) : c = 42 :=
by
-- Proof steps here, we skip by using sorry for now
sorry

end student_score_l235_235522


namespace sequence_not_neccessarily_periodic_l235_235438

-- Define the conditions
def sequence (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, ∃ t : ℕ, ∀ m : ℕ, a (k + m * t) = a k

-- State the problem
theorem sequence_not_neccessarily_periodic :
  ∃ (a : ℕ → ℕ), sequence a ∧ ¬ ∃ T : ℕ, ∀ k : ℕ, a (k + T) = a k := 
sorry

end sequence_not_neccessarily_periodic_l235_235438


namespace sum_of_zeros_of_comp_range_of_a_l235_235801

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x - b / x - 2 * Real.log x

theorem sum_of_zeros_of_comp (a b : ℝ) :
  (∀ x > 0, f a b x = -f a b (1 / x)) →
  (∃ t : ℝ → ℝ, (∀ x : ℝ, t x = f a a (Real.exp x)) ∧
  (∀ x : ℝ, t x = 0 → x + (x → ∀ y : ℝ, t y = 0) = 0)) := sorry

theorem range_of_a (a : ℝ) :
  (∀ b x > 0, (f a b x = -f a b (1 / x) → f a a x ≥ 0) → x ≥ 1) →
  1 ≤ a := sorry

end sum_of_zeros_of_comp_range_of_a_l235_235801


namespace rectangle_longer_side_l235_235692

theorem rectangle_longer_side (r : ℝ) (π : ℝ) (A_circle : ℝ) (A_rectangle : ℝ) (shorter_side : ℝ) (longer_side : ℝ) :
  r = 3 → A_circle = π * r^2 → A_rectangle = 3 * A_circle → shorter_side = 2 * r → longer_side = A_rectangle / shorter_side → longer_side = 4.5 * π :=
begin
  intros hr hA_circle hA_rectangle hshorter_side hlonger_side,
  rw hr at *,
  rw hA_circle at *,
  rw hA_rectangle at *,
  rw hshorter_side at *,
  rw hlonger_side at *,
  exact sorry
end

end rectangle_longer_side_l235_235692


namespace machine_bottle_caps_l235_235303

variable (A_rate : ℕ)
variable (A_time : ℕ)
variable (B_rate : ℕ)
variable (B_time : ℕ)
variable (C_rate : ℕ)
variable (C_time : ℕ)
variable (D_rate : ℕ)
variable (D_time : ℕ)
variable (E_rate : ℕ)
variable (E_time : ℕ)

def A_bottles := A_rate * A_time
def B_bottles := B_rate * B_time
def C_bottles := C_rate * C_time
def D_bottles := D_rate * D_time
def E_bottles := E_rate * E_time

theorem machine_bottle_caps (hA_rate : A_rate = 24)
                            (hA_time : A_time = 10)
                            (hB_rate : B_rate = A_rate - 3)
                            (hB_time : B_time = 12)
                            (hC_rate : C_rate = B_rate + 6)
                            (hC_time : C_time = 15)
                            (hD_rate : D_rate = C_rate - 4)
                            (hD_time : D_time = 8)
                            (hE_rate : E_rate = D_rate + 5)
                            (hE_time : E_time = 5) :
  A_bottles A_rate A_time = 240 ∧ 
  B_bottles B_rate B_time = 252 ∧ 
  C_bottles C_rate C_time = 405 ∧ 
  D_bottles D_rate D_time = 184 ∧ 
  E_bottles E_rate E_time = 140 := by
    sorry

end machine_bottle_caps_l235_235303


namespace part1_part2_l235_235797

noncomputable def f (x : Real) : Real :=
  (Real.sqrt 3 * Real.tan x + 1) * (Real.cos x) ^ 2

theorem part1 (α : Real) (h1 : α ∈ set.Ioo (Real.pi / 2) Real.pi)
  (h2 : Real.cos α = - (Real.sqrt 5) / 5) : 
  f α = (1 - 2 * Real.sqrt 3) / 5 := by
  sorry

theorem part2 (x : Real) (hx1 : x ∈ set.Ici (Real.pi / 4))
  (hx2 : x ∈ set.Iic (3 * Real.pi / 4)) : 
  (f x) is_strict_anti_on 
    (set.Ico (Real.pi / 4) (Real.pi / 2) ∪ set.Ioc (Real.pi / 2) (2 * Real.pi / 3)) ∧ 
  (f x) is_strict_mono_on 
    (set.Ioc (2 * Real.pi / 3) (3 * Real.pi / 4)) := by
  sorry

end part1_part2_l235_235797


namespace find_y_l235_235493

variables (x y : ℝ)

theorem find_y (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 :=
by
  sorry

end find_y_l235_235493


namespace polar_equation_line_parallel_to_polar_axis_l235_235284

def line_parallel_to_polar_axis (c : ℝ) : Prop :=
  ∀ θ : ℝ, ∃ρ : ℝ, ρ * sin θ = c

-- Now, we state the theorem as:
theorem polar_equation_line_parallel_to_polar_axis (c : ℝ) :
  line_parallel_to_polar_axis c :=
by
  sorry

end polar_equation_line_parallel_to_polar_axis_l235_235284


namespace integral_values_l235_235264

noncomputable def P1 (x : ℝ) : ℝ := 2 * x
noncomputable def P2 (x : ℝ) : ℝ := 12 * x^2 - 4
noncomputable def P3 (x : ℝ) : ℝ := 120 * x^3 - 72 * x

theorem integral_values :
  (∃ k l : ℕ, k ∈ {1, 2, 3} ∧ l ∈ {1, 2, 3} ∧ ∫ x in -1..1, P_k(x) * P_l(x), dx ∈ {0, 8 / 3, 128 / 5, 336, 691.2})
where
  P_k : ℕ → ℝ → ℝ
  | 1 := P1
  | 2 := P2
  | 3 := P3 := sorry

end integral_values_l235_235264


namespace train_length_200_04_l235_235714

-- Define the constants
def speed_kmh : ℝ := 60     -- speed in km/h
def time_seconds : ℕ := 12  -- time in seconds

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Convert speed to m/s
def speed_ms : ℝ := (speed_kmh * km_to_m) / hr_to_s

-- Define the length of the train in meters
def length_of_train : ℝ := speed_ms * time_seconds

-- The theorem to prove
theorem train_length_200_04 : length_of_train = 200.04 := by
  sorry

end train_length_200_04_l235_235714


namespace minimum_bail_rate_l235_235638

theorem minimum_bail_rate 
  (distance : ℝ)
  (leak_rate : ℝ)
  (max_water : ℝ)
  (rowing_speed : ℝ)
  (bail_rate : ℝ)
  (time_to_shore : ℝ) :
  distance = 2 ∧
  leak_rate = 15 ∧
  max_water = 60 ∧
  rowing_speed = 3 ∧
  time_to_shore = distance / rowing_speed * 60 →
  bail_rate = (leak_rate * time_to_shore - max_water) / time_to_shore →
  bail_rate = 13.5 :=
by
  intros
  sorry

end minimum_bail_rate_l235_235638


namespace purely_imaginary_iff_m_eq_0_or_1_l235_235937

theorem purely_imaginary_iff_m_eq_0_or_1 (m : ℝ) :
  (m^2 - m + 0 * real.i + 3 * complex.i).re = 0 ↔ (m = 0 ∨ m = 1) :=
by {
  sorry
}

end purely_imaginary_iff_m_eq_0_or_1_l235_235937


namespace modulus_z_eq_sqrt_10_l235_235112

noncomputable def z := (10 * Complex.I) / (3 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_z_eq_sqrt_10_l235_235112


namespace unique_number_not_in_range_of_g_l235_235219

noncomputable def g (p q r s : ℝ) (x : ℝ) := (p * x + q) / (r * x + s)

theorem unique_number_not_in_range_of_g :
  ∀ (p q r s : ℝ),
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (g p q r s 31 = 31) →
  (g p q r s 41 = 41) →
  (∀ x : ℝ, x ≠ -s / r → g p q r s (g p q r s x) = x) →
  ∃! y : ℝ, ¬ ∃ x : ℝ, g p q r s x = y := 36 :=
begin
  sorry
end

end unique_number_not_in_range_of_g_l235_235219


namespace bisect_arc_l235_235211

variable {A B C D X I J M P : Type}
variable [CyclicQuadrilateral A B C D]
variable [IntersectsAt AC BD X]
variable [Incenter I X B C]
variable [Excenter J P B C]

-- Define assumptions
axiom cyclic_ABCD : CyclicQuadrilateral A B C D
axiom intersection_X : IntersectsAt AC BD X
axiom incenter_I : Incenter I (Triangle X B C)
axiom excenter_J : Excenter J (Triangle P B C)
axiom P_intersection : IntersectsAt AB CD P

-- Define to prove
theorem bisect_arc : BisectsArc IJ (Circumcircle A B C D) B C (Exclusive A D) :=
  sorry

end bisect_arc_l235_235211


namespace polar_to_rectangular_coordinates_l235_235519

theorem polar_to_rectangular_coordinates :
  let r := 2
  let θ := Real.pi / 3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (1, Real.sqrt 3) :=
by
  sorry

end polar_to_rectangular_coordinates_l235_235519


namespace binom_60_3_l235_235008

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235008


namespace mountain_number_proof_l235_235648

noncomputable def mountain_numbers_count : ℕ :=
  let case1 := ∑ x in Finset.range 1 10, (9 - x)
  let case2 := ∑ y in Finset.range 2 10, (y - 1) * (y - 1)
  case1 + case2

theorem mountain_number_proof : mountain_numbers_count = 176 := by
  sorry

end mountain_number_proof_l235_235648


namespace coefficient_x2_expansion_l235_235963

def binom : ℕ → ℕ → ℕ 
| n, k := nat.choose n k

def sum_binom (a b k : ℕ) : ℕ :=
list.sum (list.map (λ n, binom n k) (list.range (b - a + 1)).map (λ x, x + a))

theorem coefficient_x2_expansion :
  sum_binom 3 10 2 = 164 := 
begin
  sorry
end

end coefficient_x2_expansion_l235_235963


namespace hyperbola_equation_exists_l235_235933

theorem hyperbola_equation_exists :
  let ellipse_eq := (x : ℝ) (y : ℝ), (x^2 / 27) + (y^2 / 36) = 1
  let hyperbola_passing_point := (x, y) = (real.sqrt 15, 4)
  let hyperbola_eq := (x : ℝ) (y : ℝ), (y^2 / 4) - (x^2 / 5) = 1
  ellipse_foci := (0, real.sqrt 36)
in ellipse_foci = hyperbola_foci ∧ hyperbola_passing_point → hyperbola_eq := sorry

end hyperbola_equation_exists_l235_235933


namespace ratio_AC_CE_eq_sin_alpha_l235_235436

-- Define the problem conditions
variables (A B C D E : Type) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D] [inner_product_space ℝ E]
variables (alpha : ℝ) 
variables (triangle_ABC : true)
variables (angle_ABC_eq_alpha : α ∠ (A, B, C) = alpha)
variables (D_on_extension_BC : B = C + D)
variables (tangent_AD : tangent AD (circumcircle_triangle_ABC))
variables (intersect_circumcircle_ABD_E : ∃ E, line AC ∩ (circumcircle_triangle_ABD) = { E })
variables (angle_bisector_tangent : tangent (angle_bisector ADE) (circumcircle_triangle_ABC))

-- Define the proof goal
theorem ratio_AC_CE_eq_sin_alpha :
  (AC / CE) = sin alpha :=
sorry

end ratio_AC_CE_eq_sin_alpha_l235_235436


namespace math_garden_value_l235_235192

-- Definitions representing the conditions
def diff_digits : Prop := ∀ (x y : ℕ), x ≠ y → 二 ≠ 零 ∧ 一 ≠ 五 -- Different Chinese characters represent different digits

def 二零一五_eq_2015 : Prop := (二 * 1000 + 零 * 100 + 一 * 10 + 五) = 2015 -- 二零一五 = 2015

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n -- Uniqueness of prime numbers

-- Math proof problem from the conditions
theorem math_garden_value
    (diff_digits_condition : diff_digits)
    (二零一五_equals_2015 : 二零一五_eq_2015)
    (数学_is_prime : ∃ n, (数学 = n) ∧ is_prime n) :
    数学花园 = 8369 :=
    sorry

end math_garden_value_l235_235192


namespace repaved_before_today_correct_l235_235694

variable (total_repaved_so_far repaved_today repaved_before_today : ℕ)

axiom given_conditions : total_repaved_so_far = 4938 ∧ repaved_today = 805 

theorem repaved_before_today_correct :
  total_repaved_so_far = 4938 →
  repaved_today = 805 →
  repaved_before_today = total_repaved_so_far - repaved_today →
  repaved_before_today = 4133 :=
by
  intros
  sorry

end repaved_before_today_correct_l235_235694


namespace limit_f_lt_1_plus_pi_div_4_l235_235995

noncomputable def f : ℝ → ℝ := sorry

theorem limit_f_lt_1_plus_pi_div_4 (f : ℝ → ℝ) 
  (hf_diff : ∀ x ∈ Set.Ici 1, DifferentiableAt ℝ f x)
  (hf_deriv : ∀ x ∈ Set.Ici 1, deriv f x = 1 / (x^2 + (f x)^2))
  (hf_initial : f 1 = 1) :
  ∃ L : ℝ, Tendsto f atTop (𝓝 L) ∧ L < 1 + (Real.pi / 4) :=
sorry

end limit_f_lt_1_plus_pi_div_4_l235_235995


namespace gcd_80_36_l235_235310

theorem gcd_80_36 : Nat.gcd 80 36 = 4 := by
  -- Using the method of successive subtraction algorithm
  sorry

end gcd_80_36_l235_235310


namespace diagonals_inequality_l235_235534

theorem diagonals_inequality
  (Q : Type)
  [IsConvexQuadrilateral Q] -- Custom typeclass for convex quadrilateral
  (d : ℝ)
  (innerQ : Q → Type)
  [IsConvexQuadrilateral (innerQ Q)] -- Inner quadrilateral
  (d' : ℝ)
  (hQ : sum_of_diagonals_length Q = d)
  (h_innerQ : sum_of_diagonals_length (innerQ Q) = d') :
  d' < 2 * d := 
sorry -- proof to be filled in

end diagonals_inequality_l235_235534


namespace distinct_diagonals_nonagon_l235_235869

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235869


namespace num_diagonals_convex_nonagon_l235_235876

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235876


namespace intersection_empty_condition_l235_235778

-- Define the sets M and N under the given conditions
def M : Set (ℝ × ℝ) := { p | p.1^2 + 2 * p.2^2 = 3 }

def N (m b : ℝ) : Set (ℝ × ℝ) := { p | p.2 = m * p.1 + b }

-- The theorem that we need to prove based on the problem statement
theorem intersection_empty_condition (b : ℝ) :
  (∀ m : ℝ, M ∩ N m b = ∅) ↔ (b^2 > 6 * m^2 + 2) := sorry

end intersection_empty_condition_l235_235778


namespace robyn_sold_packs_l235_235242

variable (L : ℕ) (T : ℕ) (R : ℕ)

theorem robyn_sold_packs (hL : L = 19) (hT : T = 35) : R = T - L → R = 16 :=
by { intro h, rw [hL, hT] at h, exact h }

end robyn_sold_packs_l235_235242


namespace distinct_diagonals_nonagon_l235_235861

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235861


namespace largest_circumradius_of_triangle_l235_235422

theorem largest_circumradius_of_triangle (Tetrahedron : Type) (Triangle : Type)
  [AcuteScaleneTriangle Tetrahedron Triangle]
  (side_length_3 : ∃ t : Triangle, t.side = 3)
  (volume_4 : Tetrahedron.volume = 4)
  (surface_area_24 : Tetrahedron.surface_area = 24) :
  Tetrahedron.circumradius = √(4 + √3) :=
by
  sorry

end largest_circumradius_of_triangle_l235_235422


namespace circumradius_inradius_relation_l235_235256

variable {r ι b a c : ℝ}

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem circumradius_inradius_relation (b : ℝ) (h_pos : b > 2)
  (h_sides : ∀ (a b c : ℝ), a = b - 1 ∧ c = b + 1)
  (h_valid : is_valid_triangle (b - 1) b (b + 1))
  (r : ℝ) (ρ : ℝ) (h_circumradius : r = (b * (b^2 - 1)) / (sqrt (3 * (b^2 - 4))))
  (h_inradius : ρ = (sqrt (3 * (b^2 - 4))) / 6) :
  r = 2 * ρ + (1 / (2 * ρ)) := by
  sorry

end circumradius_inradius_relation_l235_235256


namespace saly_needs_10_eggs_per_week_l235_235580

theorem saly_needs_10_eggs_per_week :
  let Saly_needs_per_week := S
  let Ben_needs_per_week := 14
  let Ked_needs_per_week := Ben_needs_per_week / 2
  let total_eggs_in_month := 124
  let weeks_per_month := 4
  let Ben_needs_per_month := Ben_needs_per_week * weeks_per_month
  let Ked_needs_per_month := Ked_needs_per_week * weeks_per_month
  let Saly_needs_per_month := total_eggs_in_month - (Ben_needs_per_month + Ked_needs_per_month)
  let S := Saly_needs_per_month / weeks_per_month
  Saly_needs_per_week = 10 :=
by
  sorry

end saly_needs_10_eggs_per_week_l235_235580


namespace range_of_x_l235_235658

theorem range_of_x (a : Fin 15 → ℝ) (ha : ∀ i, a i = 0 ∨ a i = 1) :
  0 ≤ ∑ i, a i / 4^(i + 1) ∧ ∑ i, a i / 4^(i + 1) < 1 :=
sorry

end range_of_x_l235_235658


namespace max_triplets_correct_l235_235110

def maximum_triplets (n : ℕ) : ℕ :=
  if even n then
    (n * (n - 2)) / 4
  else
    ((n - 1) / 2) ^ 2

theorem max_triplets_correct (n : ℕ) : maximum_triplets n = 
  if even n then
    (n * (n - 2)) / 4
  else
    ((n - 1) / 2) ^ 2 :=
by
  sorry

end max_triplets_correct_l235_235110


namespace number_of_diagonals_in_nonagon_l235_235821

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235821


namespace determine_m_l235_235397

-- Define the polynomial p(x)
def p (x : ℝ) := 4 * x^2 - 3 * x + m

-- State the condition that p(x) is divisible by (x-2)
def condition (m : ℝ) := ∀ x : ℝ, x - 2 = 0 → p x = 0

theorem determine_m (m : ℝ) : condition m → m = -10 :=
by
  sorry

end determine_m_l235_235397


namespace non_deg_ellipse_b_l235_235402

theorem non_deg_ellipse_b (b : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = b ∧ (∀ x y : ℝ, (x - 3)^2 + 9*(y + 3/2)^2 = b + 145/4)) → b > -145/4 :=
sorry

end non_deg_ellipse_b_l235_235402


namespace number_of_subsets_of_A_l235_235427

variable (a : ℝ)

def setA : set ℝ := {x | abs (x - 1) ≤ 2 * a - a^2 - 2}

theorem number_of_subsets_of_A :
  setA a = ∅ → (number_of_subsets (setA a) = 1) := by
  sorry

end number_of_subsets_of_A_l235_235427


namespace vector_perpendicular_to_plane_l235_235283

theorem vector_perpendicular_to_plane
  (a b c d : ℝ)
  (x1 y1 z1 x2 y2 z2 : ℝ)
  (h1 : a * x1 + b * y1 + c * z1 + d = 0)
  (h2 : a * x2 + b * y2 + c * z2 + d = 0) :
  a * (x1 - x2) + b * (y1 - y2) + c * (z1 - z2) = 0 :=
sorry

end vector_perpendicular_to_plane_l235_235283


namespace nonagon_diagonals_count_l235_235838

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235838


namespace assign_positive_numbers_equally_l235_235676

theorem assign_positive_numbers_equally (lines : Finset (Set (ℝ × ℝ))) 
  (h1 : ∀ l1 l2 ∈ lines, l1 ≠ l2 → (l1 ∩ l2).Finite) 
  (h2 : ∀ l1 l2 l3 ∈ lines, l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3 → (l1 ∩ l2 ∩ l3).Nonempty → False) :
  ∃ (f : (ℝ × ℝ) → ℝ), (∀ (r : Set (ℝ × ℝ)), r ∈ regions lines → 0 < f r)
  ∧ (∀ (l ∈ lines), 
    let (left, right) := partition_regions line l in 
    sum_region_numbers f left = sum_region_numbers f right) :=
begin
  -- Sorry, proof not provided
  sorry
end

end assign_positive_numbers_equally_l235_235676


namespace oranges_to_apples_total_apple_weight_l235_235976

section
variable (num_oranges num_apples weight_apple : ℕ)
variable (weight_equiv : 9 * weight_apple = 6 * weight_apple)
variable (per_apple_weight : 120)

theorem oranges_to_apples (o a : ℕ) (h1 : o = 45) (h2 : a = 30) : 
  9 * weight_apple = 6 * weight_apple → a = (2 * o) / 3 :=
by sorry

theorem total_apple_weight (a_total_weight : ℕ) (h3 : a_total_weight = 3600) : 
  a = 30 ∧ per_apple_weight = 120 → a_total_weight = a * per_apple_weight :=
by sorry

end

end oranges_to_apples_total_apple_weight_l235_235976


namespace length_of_bridge_l235_235669

theorem length_of_bridge 
    (length_of_train : ℕ)
    (speed_of_train_km_per_hr : ℕ)
    (time_to_cross_seconds : ℕ)
    (bridge_length : ℕ) 
    (h_train_length : length_of_train = 130)
    (h_speed_train : speed_of_train_km_per_hr = 54)
    (h_time_cross : time_to_cross_seconds = 30)
    (h_bridge_length : bridge_length = 320) : 
    bridge_length = 320 :=
by sorry

end length_of_bridge_l235_235669


namespace exists_c_degree_3_l235_235744

noncomputable def f (x : ℚ) : ℚ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4
noncomputable def g (x : ℚ) : ℚ := 2 - 3*x - 7*x^3 + 12*x^4

theorem exists_c_degree_3 : ∃ c : ℚ, degree (λ x, f x + c * g x) = 3 :=
sorry

end exists_c_degree_3_l235_235744


namespace palindromes_between_200_and_800_l235_235064

/-
  Define a palindrome as a number where the hundreds digit equals the units digit.
-/
def is_palindrome (n : ℕ) : Prop :=
  let h := n / 100
  let t := (n % 100) / 10
  let u := n % 10
  h = u

/-
  The actual problem: Prove that there are exactly 60 integer palindromes between 200 and 800.
-/
theorem palindromes_between_200_and_800 : 
  ∃ (count : ℕ), count = 60 ∧ (count = (set.univ.filter (λ n, 200 ≤ n ∧ n < 800 ∧ is_palindrome n)).card) :=
begin
  sorry
end

end palindromes_between_200_and_800_l235_235064


namespace binom_60_3_eq_34220_l235_235046

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235046


namespace num_diagonals_convex_nonagon_l235_235872

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235872


namespace probability_value_l235_235461

-- Define the probability mass function as given in the conditions
noncomputable def pmf (C : ℝ) (k : ℕ) : ℝ :=
  if k ∈ {1, 2, 3, 4, 5} then (C * k) / 15 else 0

-- Given that the sum of probabilities equals 1
axiom pmf_sum_one {C : ℝ} :
  (∑ k in {1, 2, 3, 4, 5}, pmf C k) = 1

-- Define the probability to be proven
def probability_between_half_and_fivehalf (C : ℝ) :=
  pmf C 1 + pmf C 2

-- Final theorem to prove the probability value
theorem probability_value {C : ℝ} (hC : pmf_sum_one C) :
  probability_between_half_and_fivehalf C = 1 / 3 :=
sorry

end probability_value_l235_235461


namespace binom_60_3_l235_235020

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235020


namespace count_perfect_cubes_l235_235488

theorem count_perfect_cubes (a b : ℕ) (ha : a = 2^9 + 1) (hb : b = 2^17 + 1) :
  (∃ c : ℕ, c*c*c = a) ∧ (∃ d : ℕ, d*d*d = b) ∧ 
  (∃ n, ∀ (k : ℕ), (a ≤ k) ∧ (k ≤ b) → k = n*n*n) →
  42 :=
by 
  have : a = 513 := by rw [ha]
  have : b = 131073 := by rw [hb]
  sorry

end count_perfect_cubes_l235_235488


namespace find_tangent_line_l235_235941

noncomputable def tangent_line_equation (l : ℝ → ℝ) : Prop :=
  (∃ x0 : ℝ, l x0 = Real.exp x0 ∧ l = λ x, Real.exp x0 * (x - x0) + Real.exp x0) ∧
  (∃ x1 : ℝ, l x1 = -x1^2 / 4 ∧ l = λ x, -x1 / 2 * x + x1^2 / 4)

theorem find_tangent_line : ∃ l : ℝ → ℝ, tangent_line_equation l ∧ (∀ x, l x = x + 1) :=
  sorry

end find_tangent_line_l235_235941


namespace simplify_expression_l235_235654

theorem simplify_expression :
  ( (50^2 - 9^2) / (40^2 - 8^2) * ( (40 - 8) * (40 + 8) / ( (50 - 9) * (50 + 9) ) ) = 1 := 
by 
  sorry

end simplify_expression_l235_235654


namespace smallest_angle_condition_l235_235739

theorem smallest_angle_condition (y : ℝ) (deg_to_rad := (Real.pi / 180)) :
  9 * sin (y * deg_to_rad) * cos (y * deg_to_rad)^3 - 9 * sin (y * deg_to_rad)^3 * cos (y * deg_to_rad) = 3 * Real.sqrt 2 
  → y = 22.5 :=
by
  intro h
  sorry

end smallest_angle_condition_l235_235739


namespace centroid_moves_on_straight_line_l235_235527

-- Define the point
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A B C : Point

-- Define the centroid of a triangle
def centroid (t : Triangle) : Point :=
  ⟨(t.A.x + t.B.x + t.C.x) / 3, (t.A.y + t.B.y + t.C.y) / 3⟩

-- Define a line
def on_line (p : Point) (slope : ℝ) (c : ℝ) : Prop :=
  p.y = slope * p.x + c

-- Given conditions
variable (A B C : Point)
variable (b : ℝ)
variable (h_line : C.y = (1/√3) * C.x)

-- Fixed base condition
variable (base_fixed : A = ⟨0,0⟩ ∧ B = ⟨b,0⟩)

-- Prove centroid moves on a straight line
theorem centroid_moves_on_straight_line (t : Triangle) (C_line : t.C.y = (1/√3) * t.C.x) : 
  ∃ m, ∀ G, centroid t = G → on_line G (1 / √3) 0 :=
sorry

end centroid_moves_on_straight_line_l235_235527


namespace combined_average_marks_l235_235666

theorem combined_average_marks 
  (n1 n2 : ℕ) (avg1 avg2 : ℕ) 
  (h1 : n1 = 55) (h2 : avg1 = 60) 
  (h3 : n2 = 48) (h4 : avg2 = 58) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 59.07 :=
by
  sorry

end combined_average_marks_l235_235666


namespace nonagon_diagonals_count_l235_235912

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235912


namespace distinct_diagonals_in_nonagon_l235_235903

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235903


namespace percentagesProof_l235_235738

-- Mathematics definitions and conditions
def percentages := {58, 42, 150, 124, 100, 22.8} : Set ℝ

theorem percentagesProof :
  (∃ x y ∈ percentages, x + y = 100) ∧
  (124 ∈ percentages) ∧
  (150 ∈ percentages) ∧
  (100 ∈ percentages) :=
by {
  -- Placeholder for proof steps
  sorry
}

end percentagesProof_l235_235738


namespace discriminant_eq_13_l235_235504

theorem discriminant_eq_13 (m : ℝ) (h : (3)^2 - 4*1*(-m) = 13) : m = 1 :=
sorry

end discriminant_eq_13_l235_235504


namespace complex_div_conjugates_l235_235812

theorem complex_div_conjugates (z1 z2 : ℂ) (h_conj : z2 = conj z1) (h_z1 : z1 = 1 - 2 * I) :
    z1 / z2 = -3/5 - 4/5 * I := by
  sorry

end complex_div_conjugates_l235_235812


namespace hidden_values_l235_235508

def system_equations (x y : ℝ) :=
  (x + y = ★) ∧ (2 * x + y = 16)

def solution (x y : ℝ) := 
  (x = 6) ∧ (y = 4)

theorem hidden_values (★ : ℝ) (■ : ℝ) 
  (h1 : solution 6 4) 
  (h2 : system_equations 6 4) : 
  (★ = 10) ∧ (■ = 4) := by
  sorry

end hidden_values_l235_235508


namespace seating_arrangement_l235_235586

-- Define the problem in Lean
theorem seating_arrangement :
  let n := 9   -- Total number of people
  let r := 7   -- Number of seats at the circular table
  let combinations := Nat.choose n 2  -- Ways to select 2 people not seated
  let factorial (k : ℕ) := Nat.recOn k 1 (λ k' acc => (k' + 1) * acc)
  let arrangements := factorial (r - 1)  -- Ways to seat 7 people around a circular table
  combinations * arrangements = 25920 :=
by
  -- In Lean, sorry is used to indicate that we skip the proof for now.
  sorry

end seating_arrangement_l235_235586


namespace integer_solutions_3x2_plus_5y2_eq_453_l235_235587

theorem integer_solutions_3x2_plus_5y2_eq_453 :
  ∃ t : ℤ, y = 3 * t ∧ (3 * x^2 + 5 * y^2 = 453) ↔ 
    (x = 4 ∨ x = -4) ∧ (y = 9 ∨ y = -9) :=
begin
  sorry
end

end integer_solutions_3x2_plus_5y2_eq_453_l235_235587


namespace candice_bakery_expense_l235_235000

def weekly_expense (white_bread_price : ℕ → ℚ) (baguette_price : ℚ) (sourdough_bread_price : ℕ → ℚ) (croissant_price : ℚ) : ℚ :=
  white_bread_price 2 + baguette_price + sourdough_bread_price 2 + croissant_price

def four_weeks_expense (weekly_expense : ℚ) : ℚ :=
  weekly_expense * 4

theorem candice_bakery_expense :
  weekly_expense (λ n, 3.50 * n) 1.50 (λ n, 4.50 * n) 2.00 * 4 = 78.00 := by
  sorry

end candice_bakery_expense_l235_235000


namespace james_profit_correct_l235_235538

noncomputable def jamesProfit : ℝ :=
  let tickets_bought := 200
  let cost_per_ticket := 2
  let winning_ticket_percentage := 0.20
  let percentage_one_dollar := 0.50
  let percentage_three_dollars := 0.30
  let percentage_four_dollars := 0.20
  let percentage_five_dollars := 0.80
  let grand_prize_ticket_count := 1
  let average_remaining_winner := 15
  let tax_percentage := 0.10
  let total_cost := tickets_bought * cost_per_ticket
  let winning_tickets := tickets_bought * winning_ticket_percentage
  let tickets_five_dollars := winning_tickets * percentage_five_dollars
  let other_winning_tickets := winning_tickets - tickets_five_dollars - grand_prize_ticket_count
  let total_winnings_before_tax := (tickets_five_dollars * 5) + (grand_prize_ticket_count * 5000) + (other_winning_tickets * average_remaining_winner)
  let total_tax := total_winnings_before_tax * tax_percentage
  let total_winnings_after_tax := total_winnings_before_tax - total_tax
  total_winnings_after_tax - total_cost

theorem james_profit_correct : jamesProfit = 4338.50 := by
  sorry

end james_profit_correct_l235_235538


namespace find_a11_l235_235514

def geometric_seq (a n : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n * q

theorem find_a11 (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : geometric_seq a q)
  (h4 : a 4 = 6) 
  (h7 : a 7 = 48) : 
  a 11 = 768 := 
sorry

end find_a11_l235_235514


namespace num_divisors_pq_num_divisors_p2q_num_divisors_p2q2_num_divisors_pm_qn_l235_235996

-- Definitions for the conditions
variables (p q : ℕ) (m n : ℕ)
variables (hp : Nat.Prime p) (hq : Nat.Prime q) (hpdq : p ≠ q)

-- Prove the number of divisors for each case
theorem num_divisors_pq : Nat.divisors (p * q).card = 4 :=
sorry

theorem num_divisors_p2q : Nat.divisors (p^2 * q).card = 6 :=
sorry

theorem num_divisors_p2q2 : Nat.divisors (p^2 * q^2).card = 9 :=
sorry

theorem num_divisors_pm_qn : Nat.divisors (p^m * q^n).card = (m + 1) * (n + 1) :=
sorry

end num_divisors_pq_num_divisors_p2q_num_divisors_p2q2_num_divisors_pm_qn_l235_235996


namespace solution_count_l235_235742

noncomputable def equation_has_one_solution : Prop :=
∀ x : ℝ, (x - (8 / (x - 2))) = (4 - (8 / (x - 2))) → x = 4

theorem solution_count : equation_has_one_solution :=
by
  sorry

end solution_count_l235_235742


namespace minimum_value_expression_l235_235573

theorem minimum_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 6) :
  (a - 2)^2 + 2 * ((b / a) - 1)^2 + 3 * ((c / b) - 1)^2 + 4 * ((6 / c) - 1)^2 = 10 * (2^(0.65) - 1)^2 :=
sorry

end minimum_value_expression_l235_235573


namespace right_angled_triangle_area_l235_235117

theorem right_angled_triangle_area 
  (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 18) (h3 : a^2 + b^2 + c^2 = 128) : 
  (1/2) * a * b = 9 :=
by
  -- Proof will be added here
  sorry

end right_angled_triangle_area_l235_235117


namespace lattice_point_interior_segment_l235_235577

/-- If there are nine lattice points in ℤ^3, 
    there exists a lattice point on the interior 
    of one of the segments joining two of these points. -/
theorem lattice_point_interior_segment (points : Fin 9 → ℤ × ℤ × ℤ) :
  ∃ (i j : Fin 9), i ≠ j ∧ 
    (∃ (k : ℤ × ℤ × ℤ), 
     k = ((points i).1.1 + (points j).1.1) / 2,
        ((points i).1.2 + (points j).1.2) / 2,
        ((points i).2.1 + (points j).2.1) / 2 ) := 
  sorry

end lattice_point_interior_segment_l235_235577


namespace part_1_part_2_part_3_l235_235132

/-
  f is defined as an odd function over the interval [-1, 1]
  f(1) = 1
  The following condition holds for f: ∀ (m n : ℝ), m, n ∈ [-1, 1] -> (f(m) + f(n))/(m + n) > 0
-/
variables {f : ℝ → ℝ}

axiom odd_function : ∀ x : ℝ, x ∈ [-1, 1] → f(-x) = -f(x)
axiom f_1 : f 1 = 1
axiom positive_ratio : ∀ (m n : ℝ), m ∈ [-1, 1] → n ∈ [-1, 1] → (m ≠ n) → (f(m) + f(n)) / (m + n) > 0

theorem part_1 (x : ℝ) (h : f (x + 1/2) + f (x - 1) < 0) : 0 ≤ x ∧ x < 1/4 :=
sorry

theorem part_2 (a t : ℝ) (h : ∀ x, x ∈ [-1, 1] → a ∈ [-1, 1] → f(x) ≤ t^2 - 2*a*t + 1) : t ≤ -2 ∨ t = 0 ∨ t ≥ 2 :=
sorry

theorem part_3 (x t : ℝ) (h : ∀ x, x ∈ [-1, 1] → t ∈ [-1, 1] → f(x) ≤ t^2 - 2*a*t + 2) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end part_1_part_2_part_3_l235_235132


namespace solve_quadratic_eq_l235_235325

theorem solve_quadratic_eq : ∀ x : ℝ, (12 - 3 * x)^2 = x^2 ↔ x = 6 ∨ x = 3 :=
by
  intro x
  sorry

end solve_quadratic_eq_l235_235325


namespace percent_increase_may_to_june_l235_235628

theorem percent_increase_may_to_june (P : ℝ) :
  (let x := 50.000000000000025 in
  (1.2 * P + (x / 100) * 1.2 * P = 1.8000000000000003 * P)) :=
by
  let x := 50.000000000000025
  sorry

end percent_increase_may_to_june_l235_235628


namespace binom_60_3_l235_235005

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235005


namespace finite_solutions_x4_y3_z_factorial_l235_235071

theorem finite_solutions_x4_y3_z_factorial (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ N : ℕ, ∀ x y z : ℕ, (x, y, z >= N) → x^4 + y^3 ≠ z! + 7 :=
sorry

end finite_solutions_x4_y3_z_factorial_l235_235071


namespace binomial_60_3_l235_235025

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235025


namespace prod_lcm_gcd_eq_216_l235_235762

theorem prod_lcm_gcd_eq_216 (a b : ℕ) (h1 : a = 12) (h2 : b = 18) :
  (Nat.gcd a b) * (Nat.lcm a b) = 216 := by
  sorry

end prod_lcm_gcd_eq_216_l235_235762


namespace part1_part2_l235_235798

noncomputable def f (a : ℝ) (x : ℝ) := 2 * x + (2 / x) - a * Real.log x
noncomputable def f_prime (a : ℝ) (x : ℝ) := D[Real.c_derive] (λ x, f a x)
noncomputable def g (a : ℝ) (x : ℝ) := x^2 * (f_prime a x + 2 * x - 2)

theorem part1 (a : ℝ) (h1 : ∀ x : ℝ, 1 ≤ x → 2 - (2 / x^2) - (a / x) ≥ 0) : a ≤ 0 := sorry

theorem part2 (a : ℝ) (h2 : ∃ x : ℝ, x > 0 ∧ g a x = -6) : a = 6 := sorry

end part1_part2_l235_235798


namespace circle_symmetry_ratio_l235_235312

-- Define the conditions
structure Circle :=
  (center : Point)
  (radius : ℝ)

variables {A B C D : Point}
variables {R1 R2 : ℝ}

-- Define intersections and tangency conditions
def circles_intersect (circle1 circle2 : Circle) (A B : Point) : Prop :=
  -- Intersect at points A and B
  sorry

def tangent_chord (circle : Circle) (A C : Point) : Prop :=
  -- AC is tangent to the circle at point A
  sorry

-- Problem statement
theorem circle_symmetry_ratio (circle1 circle2 : Circle)
  (h1 : circles_intersect circle1 circle2 A B)
  (h2 : tangent_chord circle1 A C)
  (h3 : tangent_chord circle2 A D) :
  (symmetric_lines BC BD AB) ∧ (BC / BD = circle2.radius / circle1.radius) :=
by
    -- skipping the proof
    sorry

end circle_symmetry_ratio_l235_235312


namespace sequence_100th_term_l235_235804

theorem sequence_100th_term :
  (∑ k in Finset.range 100, (k + 1) / (101 : ℝ)) = 50 :=
by sorry

end sequence_100th_term_l235_235804


namespace distinct_diagonals_in_convex_nonagon_l235_235888

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235888


namespace nonagon_diagonals_count_l235_235914

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235914


namespace properties_of_int_floor_l235_235563

noncomputable def int_floor := λ (x : ℝ), ⌊x⌋

theorem properties_of_int_floor (x : ℝ) (n : ℤ) :
  (int_floor x ∈ ℤ) ∧
  (x_1 x_2 : ℝ) (h₁ : x₁ ≤ x₂) → int_floor x₁ ≤ int_floor x₂ ∧
  (∀ x, ∃ n ∈ ℤ, int_floor (n + x) = n + int_floor x) ∧
  (∀ x, int_floor x ≤ x ∧ x < int_floor x + 1) ∧
  (∀ x, (x ∈ ℤ → int_floor (-x) = -int_floor x) ∧ (x ∉ ℤ → int_floor (-x) = -int_floor x - 1))
:=
begin
  sorry
end

end properties_of_int_floor_l235_235563


namespace middle_segments_equal_l235_235953

theorem middle_segments_equal (
  ABC : Triangle,
  C : RightAngle,
  O : InscribedCircleCenter,
  A1 A2 B1 B2 C1 C2 : Point
) (h1 : line_through_center_inscribed_circle_parallel_to_sides O) 
  (h2 : divides_sides_into_segments A1 A2 B1 B2 C1 C2) : 
  segment_length C1 C2 = segment_length A1 A2 + segment_length B1 B2 := 
sorry

end middle_segments_equal_l235_235953


namespace sum_of_n_for_continuity_l235_235571

def f (x n : ℝ) : ℝ := 
  if x < n then x^2 + 3 else 2*x + 7

theorem sum_of_n_for_continuity :
  (∑ n in {n : ℝ | n^2 - 2*n - 4 = 0}, n) = 2 :=
by
  sorry

end sum_of_n_for_continuity_l235_235571


namespace nonagon_diagonals_count_l235_235899

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235899


namespace simplify_f_eval_f_at_8π3_l235_235989

open Real

def f (x : ℝ) := (sin (π + x) * cos (π - x) * sin (2 * π - x)) / 
                 (sin (π / 2 + x) * cos (x - π / 2) * cos (-x))

theorem simplify_f : ∀ x : ℝ, f x = -tan x := 
by
  intro x
  sorry

theorem eval_f_at_8π3 : f (8 * π / 3) = -√3 := 
by
  rw [simplify_f]
  -- Use tan periodicity and trigonometric properties here
  sorry

end simplify_f_eval_f_at_8π3_l235_235989


namespace perp_line_through_point_l235_235417

variable (x y c : ℝ)

def line_perpendicular (x y : ℝ) : Prop :=
  x - 2*y + 1 = 0

def perpendicular_line (x y c : ℝ) : Prop :=
  2*x + y + c = 0

theorem perp_line_through_point :
  (line_perpendicular x y) ∧ (perpendicular_line (-2) 3 1) :=
by
  -- The first part asserts that the given line equation holds
  have h1 : line_perpendicular x y := sorry
  -- The second part asserts that our calculated line passes through the point (-2, 3) and is perpendicular
  have h2 : perpendicular_line (-2) 3 1 := sorry
  exact ⟨h1, h2⟩

end perp_line_through_point_l235_235417


namespace composition_value_l235_235552

noncomputable def P (x : ℝ) : ℝ := 3 * real.sqrt x
noncomputable def Q (x : ℝ) : ℝ := x ^ 3

theorem composition_value :
  P (Q (P (Q (P (Q 2))))) = 846 * real.sqrt 2 :=
by sorry

end composition_value_l235_235552


namespace coefficient_x2_in_expansion_l235_235415

theorem coefficient_x2_in_expansion :
  (∃ (c : ℤ), c = 108 ∧ (∀ x, polynomial.coeff ((polynomial.expand_coeff _ 4 (polynomial.X ^ 2 - 2 * polynomial.X - 3)) ^ 4) 2 = c)) :=
sorry

end coefficient_x2_in_expansion_l235_235415


namespace binom_60_3_l235_235007

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235007


namespace minimum_sum_min_l235_235982

theorem minimum_sum_min (n : ℕ) (a : Finₓ n → ℕ) (h_perm : ∀ x, 1 ≤ a x ∧ a x ≤ n) (distinct : (∃ a' : Finₓ n → ℕ, ∀ (i j : Finₓ n), i ≠ j → a' i ≠ a' j ∧ ∀ x, 1 ≤ a' x ∧ a' x ≤ n)) :
  ∑ i in Finₓ.range n, min (a i) (2 * i.val + 1) = ∑ i in Finₓ.range (floor ((n + 2) / 3)), (2 * i.val + 1) +
    ∑ i in Finₓ.range (n - (ceil ((n + 2) / 3))) + 1, (n + 1 - (i.val + ceil ((n + 2) / 3))) :=
by
  sorry

end minimum_sum_min_l235_235982


namespace fixed_point_parabola_l235_235221

theorem fixed_point_parabola (t : ℝ) : ∀ (t : ℝ), (5 * 3 ^ 2 + t * 3 - 3 * t) = 45 :=
by
  intro t
  calc
    5 * 3 ^ 2 + t * 3 - 3 * t = 5 * 9 + t * 3 - 3 * t := by rfl
                            ... = 45                 := by ring

end fixed_point_parabola_l235_235221


namespace eq1_solution_eq2_solution_eq3_solution_eq4_solution_l235_235389

-- Equation 1: 3x^2 - 2x - 1 = 0
theorem eq1_solution (x : ℝ) : 3 * x ^ 2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) :=
by sorry

-- Equation 2: (y + 1)^2 - 4 = 0
theorem eq2_solution (y : ℝ) : (y + 1) ^ 2 - 4 = 0 ↔ (y = 1 ∨ y = -3) :=
by sorry

-- Equation 3: t^2 - 6t - 7 = 0
theorem eq3_solution (t : ℝ) : t ^ 2 - 6 * t - 7 = 0 ↔ (t = 7 ∨ t = -1) :=
by sorry

-- Equation 4: m(m + 3) - 2m = 0
theorem eq4_solution (m : ℝ) : m * (m + 3) - 2 * m = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

end eq1_solution_eq2_solution_eq3_solution_eq4_solution_l235_235389


namespace snowman_initial_volume_snowman_adjusted_volume_l235_235978

noncomputable def initial_volume_sum (r1 r2 r3 : ℕ) : ℝ :=
  (4 / 3) * real.pi * (r1^3 + r2^3 + r3^3)

noncomputable def adjusted_volume_sum (r1 r2 r3 : ℕ) : ℝ :=
  (4 / 3) * real.pi * ((r1 + 1)^3 + (r2 + 1)^3 + (r3 + 1)^3)

theorem snowman_initial_volume : initial_volume_sum 4 6 8 = 1056 * real.pi :=
  sorry

theorem snowman_adjusted_volume : adjusted_volume_sum 4 6 8 = 1596 * real.pi :=
  sorry

end snowman_initial_volume_snowman_adjusted_volume_l235_235978


namespace bounded_area_l235_235594

def C : Point := sorry
def O : Point := sorry
def OC : Length := Real.sqrt 2
def radius : Real := 2
def r_figure_area : Real := (π / 3) - Real.sqrt(6) + Real.sqrt(2)

theorem bounded_area :
  ∃ (C O : Point) (OC radius : Real) (r_figure_area : Real),
    (OC = Real.sqrt 2) ∧ (radius = 2) ∧ (r_figure_area = (π / 3) - Real.sqrt(6) + Real.sqrt(2)) :=
  sorry

end bounded_area_l235_235594


namespace distinct_factors_count_l235_235920

theorem distinct_factors_count (n : ℕ) (h : n = 4^3 * 5^4 * 6^2) : nat.count_divisors n = 135 :=
by
  have h1 : n = 2^8 * 3^2 * 5^4 := by sorry
  have h2 : nat.count_divisors (2^8 * 3^2 * 5^4) = 135 := by sorry
  rw [h1, h2]
  exact h2

end distinct_factors_count_l235_235920


namespace probability_at_least_one_unqualified_l235_235152

theorem probability_at_least_one_unqualified :
  let total_products := 6
  let qualified_products := 4
  let unqualified_products := 2
  let products_selected := 2
  (1 - (Nat.choose qualified_products 2 / Nat.choose total_products 2)) = 3/5 :=
by
  sorry

end probability_at_least_one_unqualified_l235_235152


namespace distinct_diagonals_nonagon_l235_235867

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235867


namespace answer_is_C_l235_235328

def is_set (S : Set α) : Prop := 
  -- Elements of S must be definite
  -- Elements of S must be distinct
  -- Elements of S must be unordered
  sorry

def option_A : Prop := 
  ∃ (S : Set α), ¬ is_set S -- "Students with higher basketball skills in the school"

def option_B : Prop := 
  ∃ (S : Set α), ¬ is_set S -- "Tall trees in the campus"

def option_C : Prop := 
  ∃ (S : Set α), is_set S -- "All the EU countries in 2012"

def option_D : Prop := 
  ∃ (S : Set α), ¬ is_set S -- "Economically developed cities in China"

theorem answer_is_C : option_C := 
  sorry

end answer_is_C_l235_235328


namespace find_angle_DPB_prove_P_on_AC_l235_235272

variables {A B C D P : Point}
variables (α β γ δ : ℝ)

-- Given conditions
def convex_quadrilateral (A B C D : Point) : Prop := sorry
def angle_A_60 (A B C D : Point) : Prop := ∠A = 60
def angle_bisector_AC (A B C D : Point) : Prop := is_angle_bisector A C
def angle_ACD_40 (A B C D : Point) : Prop := ∠ACD = 40
def angle_ACB_120 (A B C D : Point) : Prop := ∠ACB = 120
def angle_PDA_40 (P A D : Point) : Prop := ∠PDA = 40
def angle_PBA_10 (P A B : Point) : Prop := ∠PBA = 10

-- To prove angle DPB = 10
theorem find_angle_DPB (A B C D P : Point) (h1 : convex_quadrilateral A B C D) (h2 : angle_A_60 A B C D) (h3 : angle_bisector_AC A B C D) (h4 : angle_ACD_40 A B C D) (h5 : angle_ACB_120 A B C D) (h6 : angle_PDA_40 P A D) (h7 : angle_PBA_10 P A B) : ∠DPB = 10 := 
sorry

-- To prove P lies on diagonal AC
theorem prove_P_on_AC (A B C D P : Point) (h1 : convex_quadrilateral A B C D) (h2 : angle_A_60 A B C D) (h3 : angle_bisector_AC A B C D) (h4 : angle_ACD_40 A B C D) (h5 : angle_ACB_120 A B C D) (h6 : angle_PDA_40 P A D) (h7 : angle_PBA_10 P A B) : collinear (C :: P :: A :: []) := 
sorry

end find_angle_DPB_prove_P_on_AC_l235_235272


namespace max_value_m_inequality_l235_235092

theorem max_value_m_inequality :
  ∃ x : ℝ, ∀ m : ℝ, 
    m > 0 →
    m * real.sqrt m * (x^2 - 6 * x + 9) + real.sqrt m / (x^2 - 6 * x + 9) 
    ≤ real.sqrt (real.sqrt (m^3)) * abs (real.cos (real.pi * x / 5)) 
    ↔ m ≤ 1/16 :=
sorry

end max_value_m_inequality_l235_235092


namespace max_tickets_within_13_matches_l235_235263

theorem max_tickets_within_13_matches :
    ∃ k : ℕ, k = 4 ∧ ∀ m : ℕ, m < 13 → 
             (∑ i in finset.range (nat.succ m), nat.choose 13 i * (2:ℤ)^(13 - i) ≤ 
              nat.choose 13 4 * (2:ℤ)^(13 - 4)) :=
by {
  sorry,
}

end max_tickets_within_13_matches_l235_235263


namespace binomial_60_3_eq_34220_l235_235036

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235036


namespace nonagon_diagonals_l235_235848

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235848


namespace sum_of_reciprocals_is_3_over_8_l235_235632

theorem sum_of_reciprocals_is_3_over_8 (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  (1 / x + 1 / y) = 3 / 8 := 
by 
  sorry

end sum_of_reciprocals_is_3_over_8_l235_235632


namespace minimum_gumballs_needed_l235_235366

/-- Alex wants to buy at least 150 gumballs,
    and have exactly 14 gumballs left after dividing evenly among 17 people.
    Determine the minimum number of gumballs Alex should buy. -/
theorem minimum_gumballs_needed (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 14) : n = 150 :=
sorry

end minimum_gumballs_needed_l235_235366


namespace train_speed_l235_235313

theorem train_speed (v t : ℝ) (h1 : 16 * t + v * t = 444) (h2 : v * t = 16 * t + 60) : v = 21 := 
sorry

end train_speed_l235_235313


namespace trains_cross_time_l235_235646

noncomputable def time_to_cross : ℝ :=
  let length_train1 := 210 -- in meters
  let length_train2 := 260 -- in meters
  let init_speed_train1 := 60 * (1000 / 3600) -- km/hr to m/s
  let init_speed_train2 := 40 * (1000 / 3600) -- km/hr to m/s
  let accel_train1 := 3 -- m/s²
  let decel_train2 := 2 -- m/s²
  let time_accel_train1 := 20 -- seconds
  let final_speed_train1 := init_speed_train1 + accel_train1 * time_accel_train1
  let final_speed_train2 := 20 * (1000 / 3600) -- final speed of train 2 in m/s after deceleration
  let wind_resistance := 0.05
  let track_reduction := 0.03
  let corrected_speed_train1 := final_speed_train1 * (1 - wind_resistance - track_reduction)
  let corrected_speed_train2 := final_speed_train2 * (1 - wind_resistance - track_reduction)
  let relative_speed := corrected_speed_train1 + corrected_speed_train2
  let total_length := length_train1 + length_train2
  total_length / relative_speed

theorem trains_cross_time :
  let time := time_to_cross in
  abs (time - 6.21) < 0.01 :=
by sorry

end trains_cross_time_l235_235646


namespace isosceles_triangle_legs_length_l235_235526

theorem isosceles_triangle_legs_length (A B C D : Point) (AC AB : ℝ) 
  (h1 : is_isosceles_triangle A B C AB AC)
  (h2 : divides_by_median BD (A B C) 18 21) : 
  AC = 12 ∨ AC = 14 := 
sorry

end isosceles_triangle_legs_length_l235_235526


namespace steamed_buns_inequality_l235_235662

variables (x y : ℝ)

-- Define the conditions based on the given problem
def xiaohua_spent := 5 * x + 6 * y
def xiaoming_spent := 7 * x + 3 * y

-- Define the expressions to prove
def problem_eq := xiaohua_spent + 1 = xiaoming_spent
def question_eq := 4 * x - 6 * y = 2

theorem steamed_buns_inequality (h : problem_eq x y) : question_eq x y :=
by
  sorry

end steamed_buns_inequality_l235_235662


namespace linear_system_sum_l235_235437

theorem linear_system_sum (x y : ℝ) 
  (h1: x - y = 2) 
  (h2: y = 2): 
  x + y = 6 := 
sorry

end linear_system_sum_l235_235437


namespace smallest_m_n_sum_l235_235226

theorem smallest_m_n_sum (m n : ℕ) (hmn : m > n) (div_condition : 4900 ∣ (2023 ^ m - 2023 ^ n)) : m + n = 24 :=
by
  sorry

end smallest_m_n_sum_l235_235226


namespace n_is_power_of_p_l235_235456

-- Given conditions as definitions
variables {x y p n k l : ℕ}
variables (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < p) (h4 : 0 < n) (h5 : 0 < k)
variables (h6 : x^n + y^n = p^k) (h7 : odd n) (h8 : n > 1) (h9 : prime p) (h10 : odd p)

-- The theorem to be proved
theorem n_is_power_of_p : ∃ l : ℕ, n = p^l :=
  sorry

end n_is_power_of_p_l235_235456


namespace nonagon_diagonals_l235_235843

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235843


namespace distinct_diagonals_nonagon_l235_235865

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235865


namespace distinct_diagonals_in_nonagon_l235_235905

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235905


namespace distinct_diagonals_in_nonagon_l235_235909

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235909


namespace tom_age_ratio_l235_235309

variable (T M : ℕ)
variable (h1 : T = T) -- Tom's age is equal to the sum of the ages of his four children
variable (h2 : T - M = 3 * (T - 4 * M)) -- M years ago, Tom's age was three times the sum of his children's ages then

theorem tom_age_ratio : (T / M) = 11 / 2 := 
by
  sorry

end tom_age_ratio_l235_235309


namespace maximum_balloons_l235_235593

theorem maximum_balloons (p q m : ℕ) (hb : p = 4) (hc : m = 40 * p) : 
  let sale_price := p + p / 2,
      pairs := m / sale_price in 
  2 * pairs = 52 :=
by
  rw [hb, mul_comm 40, mul_assoc] at hc
  let sale_price := 4 + 4 / 2
  let pairs := 160 / sale_price
  have hs : sale_price = 6 := by norm_num
  have hp : pairs = 26 := by norm_num[mul_div_cancel_left 2 (ne_of_gt (nat.succ_pos 2)).symm, hc, hs, div_eq_mul_inv, div_eq_mul_inv (6 : ℚ)]
  unfold 52
  exact calc
    2 * pairs = 2 * 26 : by rw hp
         ... = 52     : by norm_num

end maximum_balloons_l235_235593


namespace binom_60_3_l235_235022

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235022


namespace max_red_socks_l235_235304

theorem max_red_socks (x y : ℕ) 
  (h1 : x + y ≤ 2017) 
  (h2 : (x * (x - 1) + y * (y - 1)) = (x + y) * (x + y - 1) / 2) : 
  x ≤ 990 := 
sorry

end max_red_socks_l235_235304


namespace shiny_iff_not_special_l235_235359

def sum_of_digits (n : ℕ) : ℕ := Nat.digits 10 n |>.sum

def is_shiny (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a + b ∧ sum_of_digits a = sum_of_digits b

def is_special (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits.tail.all (λ d => d = 9) ∧ digits.head.odd

theorem shiny_iff_not_special (n : ℕ) : is_shiny n ↔ ¬ is_special n :=
sorry

end shiny_iff_not_special_l235_235359


namespace nonagon_diagonals_count_l235_235915

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235915


namespace trigonometric_sum_evaluation_l235_235407

theorem trigonometric_sum_evaluation :
  ∑ x in (finset.range 41).image (λ x, x + 3), 
    2 * sin (x:ℝ) * sin 2 * (1 + sec (x-2) * sec (x+2)) = 90 := 
sorry

end trigonometric_sum_evaluation_l235_235407


namespace shift_sin_2x_left_by_pi_over_3_l235_235140

-- Given conditions
def sin_alpha : ℝ := √3 / 2
def cos_alpha : ℝ := -1 / 2

-- Definition of the given function
def f (x : ℝ) : ℝ := sin_alpha * Real.cos (2 * x) + cos_alpha * Real.cos (2 * x - π / 2)

-- Representation of the statement to prove
theorem shift_sin_2x_left_by_pi_over_3 :
  (∀ x, f (x) = Real.sin (2 * (x + π / 3))) :=
sorry

end shift_sin_2x_left_by_pi_over_3_l235_235140


namespace alice_distance_from_start_l235_235725

def walk_distance_in_feet (distance_in_meters : ℝ) : ℝ := distance_in_meters * 3.28084

def distance_from_starting_point (north_meters south_meters east_feet south_additional_feet : ℝ) : ℝ :=
  let total_south_feet := (walk_distance_in_feet south_meters + south_additional_feet) - walk_distance_in_feet north_meters
  real.sqrt (east_feet^2 + total_south_feet^2)

theorem alice_distance_from_start (north_meters_value east_feet south_meters_value south_additional_feet_value : ℝ) :
  north_meters_value = 12 → east_feet = 40 → south_meters_value = 12 → south_additional_feet_value = 18 →
  distance_from_starting_point north_meters_value south_meters_value east_feet south_additional_feet_value = 43.874 :=
begin
  sorry
end

end alice_distance_from_start_l235_235725


namespace time_spent_cleaning_trees_l235_235974

def trees_in_grove := 4 * 5
def time_per_tree_without_help := 6 -- in minutes
def time_per_tree_with_help := time_per_tree_without_help / 2 -- because it takes half as long
def total_time := trees_in_grove * time_per_tree_with_help -- total time in minutes

theorem time_spent_cleaning_trees : total_time / 60 = 1 := by
  have h1 : trees_in_grove = 20 := rfl
  have h2 : time_per_tree_with_help = 3 := by norm_num
  have h3 : total_time = 60 := by norm_num [h1, h2]
  have h4 : total_time / 60 = 1 := by norm_num [h3]
  exact h4

end time_spent_cleaning_trees_l235_235974


namespace system1_solution_system2_solution_l235_235607

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 4 * x + 8 * y = 12) (h2 : 3 * x - 2 * y = 5) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : (1/2) * x - (y + 1) / 3 = 1) (h2 : 6 * x + 2 * y = 10) :
  x = 2 ∧ y = -1 := by
  sorry

end system1_solution_system2_solution_l235_235607


namespace exponent_equality_l235_235500

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l235_235500


namespace nonagon_diagonals_count_l235_235917

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235917


namespace nonagon_diagonals_count_l235_235918

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235918


namespace ages_sum_l235_235379

theorem ages_sum (Beckett_age Olaf_age Shannen_age Jack_age : ℕ) 
  (h1 : Beckett_age = 12) 
  (h2 : Olaf_age = Beckett_age + 3) 
  (h3 : Shannen_age = Olaf_age - 2) 
  (h4 : Jack_age = 2 * Shannen_age + 5) : 
  Beckett_age + Olaf_age + Shannen_age + Jack_age = 71 := 
by
  sorry

end ages_sum_l235_235379


namespace line_equations_l235_235758

theorem line_equations : 
  ∀ (x y : ℝ), (∃ a b c : ℝ, 2 * x + y - 12 = 0 ∨ 2 * x - 5 * y = 0 ∧ (x, y) = (5, 2) ∧ b = 2 * a) :=
by
  sorry

end line_equations_l235_235758


namespace integral_solution_integral_final_result_l235_235050

noncomputable def integral_expression := 
  ∫ x, (x^3 + 6 * x^2 + 4 * x + 24) / ((x - 2) * (x + 2)^3)

theorem integral_solution : 
  ∫ x, (x^3 + 6 * x^2 + 4 * x + 24) / ((x - 2) * (x + 2)^3) = 
    ∫ x, (1 / (x - 2) - 8 / (x + 2)^3) := by
  sorry

theorem integral_final_result : 
  ∫ x, (1 / (x - 2) - 8 / (x + 2)^3) = 
    (λ x, Real.log (abs (x - 2)) + 4 / (x + 2)^2) + C := by
  sorry

end integral_solution_integral_final_result_l235_235050


namespace nonagon_diagonals_count_l235_235897

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235897


namespace maria_soap_cost_l235_235233

theorem maria_soap_cost:
  ∀ (months_per_bar : ℕ) (cost_per_bar : ℝ) (discount_4_bars : ℝ) (discount_6_bars : ℝ) (discount_8_bars : ℝ)
   (months_per_year : ℕ),
   months_per_bar = 2 →
   cost_per_bar = 8 →
   discount_4_bars = 0.05 →
   discount_6_bars = 0.10 →
   discount_8_bars = 0.15 →
   months_per_year = 12 →
   let bars_needed := months_per_year / months_per_bar in
   let total_cost := bars_needed * cost_per_bar in
   let discount := if bars_needed ≥ 8 then discount_8_bars else if bars_needed ≥ 6 then discount_6_bars else if bars_needed ≥ 4 then discount_4_bars else 0 in
   let final_cost := total_cost * (1 - discount) in
   final_cost = 43.20 :=
by 
  intros months_per_bar cost_per_bar discount_4_bars discount_6_bars discount_8_bars months_per_year;
  intros h1 h2 h3 h4 h5 h6;
  let bars_needed := months_per_year / months_per_bar;
  let total_cost := bars_needed * cost_per_bar;
  let discount := if bars_needed ≥ 8 then discount_8_bars else if bars_needed ≥ 6 then discount_6_bars else if bars_needed ≥ 4 then discount_4_bars else 0;
  let final_cost := total_cost * (1 - discount);
  exact sorry

end maria_soap_cost_l235_235233


namespace increasing_function_odd_function_l235_235144

variable (a : ℝ) (h : a > 1)

def f (x : ℝ) : ℝ := (a ^ x - 1) / (a ^ x + 1)

theorem increasing_function :
  ∀ x y : ℝ, x < y → f a x < f a y :=
sorry

theorem odd_function :
  ∀ x : ℝ, f a (-x) = -f a x :=
sorry

end increasing_function_odd_function_l235_235144


namespace triangle_cosine_l235_235611

/-- 
In a right triangle PQR, with ∠PQR = 90° and given cos(Q) = 3/5 and PR = 5,
prove that PQ = 3.
-/
theorem triangle_cosine (P Q R : Point)
  (h_right : angle P Q R = 90)
  (h_cos : cos (angle R P Q) = 3/5)
  (h_PR : dist P R = 5) :
  dist P Q = 3 :=
sorry

end triangle_cosine_l235_235611


namespace denomination_other_currency_notes_l235_235979

noncomputable def denomination_proof : Prop :=
  ∃ D x y : ℕ, 
  (x + y = 85) ∧
  (100 * x + D * y = 5000) ∧
  (D * y = 3500) ∧
  (D = 50)

theorem denomination_other_currency_notes :
  denomination_proof :=
sorry

end denomination_other_currency_notes_l235_235979


namespace nonagon_diagonals_l235_235844

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235844


namespace candice_bakery_expense_l235_235001

def weekly_expense (white_bread_price : ℕ → ℚ) (baguette_price : ℚ) (sourdough_bread_price : ℕ → ℚ) (croissant_price : ℚ) : ℚ :=
  white_bread_price 2 + baguette_price + sourdough_bread_price 2 + croissant_price

def four_weeks_expense (weekly_expense : ℚ) : ℚ :=
  weekly_expense * 4

theorem candice_bakery_expense :
  weekly_expense (λ n, 3.50 * n) 1.50 (λ n, 4.50 * n) 2.00 * 4 = 78.00 := by
  sorry

end candice_bakery_expense_l235_235001


namespace sum_p_eq_16_over_27_l235_235305

noncomputable def p (n : ℕ) : ℝ :=
  (n * (1 + (-1 : ℝ) ^ n)) / (2 ^ (n + 2)) + 
  ((n - 1) * (n - 3) * (1 - (-1 : ℝ) ^ n)) / (2 ^ (n + 3))

theorem sum_p_eq_16_over_27 : Sum (λ n, p n) = 16 / 27 :=
  sorry

end sum_p_eq_16_over_27_l235_235305


namespace max_area_quadrilateral_l235_235999

theorem max_area_quadrilateral (A B C D E F G H : Point)
  (h1 : midpoint A B = E)
  (h2 : midpoint B C = F)
  (h3 : midpoint C D = G)
  (h4 : midpoint D A = H)
  (h5 : dist E G = 12)
  (h6 : dist F H = 15) : 
  area_quadrilateral A B C D ≤ 360 := sorry

end max_area_quadrilateral_l235_235999


namespace solve_for_y_l235_235403

theorem solve_for_y (y : ℤ) : (4 + y) / (6 + y) = (2 + y) / (3 + y) → y = 0 := by 
  sorry

end solve_for_y_l235_235403


namespace capacity_of_larger_bottle_l235_235056

/-- 
  Dana normally drinks a 500 ml bottle of soda each day. Since the 500 ml bottles 
  are currently out of stock at the store, she buys a larger bottle of soda instead. 
  If Dana continues to drink 500 ml of soda each day, this larger bottle of soda will last for 4 days. 
  There are 1,000 ml in 1 liter. What is the capacity of the larger bottle of soda in liters?
-/
theorem capacity_of_larger_bottle :
  ∀ (daily_intake : ℕ) (days : ℕ) (ml_per_liter : ℕ),
    daily_intake = 500 →
    days = 4 →
    ml_per_liter = 1000 →
    (daily_intake * days) / ml_per_liter = 2 :=
by
  intros daily_intake days ml_per_liter hypo1 hypo2 hypo3
  rw [hypo1, hypo2, hypo3]
  norm_num
  sorry

end capacity_of_larger_bottle_l235_235056


namespace upstream_travel_time_l235_235686

noncomputable def boatSpeedInStillWater := 12 -- km/h
noncomputable def distanceDownstream := 54 -- km
noncomputable def timeDownstream := 3 -- hours
noncomputable def downstreamSpeed := distanceDownstream / timeDownstream -- km/h
noncomputable def currentSpeed := downstreamSpeed - boatSpeedInStillWater -- km/h
noncomputable def increasedCurrentSpeed := currentSpeed + 0.5 * currentSpeed -- km/h
noncomputable def upstreamSpeed := boatSpeedInStillWater - increasedCurrentSpeed -- km/h
noncomputable def distanceUpstream := distanceDownstream -- km

theorem upstream_travel_time:
  distanceUpstream / upstreamSpeed = 18 := by 
  sorry

end upstream_travel_time_l235_235686


namespace path_count_from_E_to_G_passing_through_F_l235_235162

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem path_count_from_E_to_G_passing_through_F :
  let E := (0, 0)
  let F := (5, 2)
  let G := (6, 5)
  ∃ (paths_EF paths_FG total_paths : ℕ),
  paths_EF = binom (5 + 2) 5 ∧
  paths_FG = binom (1 + 3) 1 ∧
  total_paths = paths_EF * paths_FG ∧
  total_paths = 84 := 
by
  sorry

end path_count_from_E_to_G_passing_through_F_l235_235162


namespace variance_is_five_l235_235434

-- Given the sample set with specific terms
def sample_set : List ℝ := [1, 3, 5, 7]

-- Define the second term and the fourth term of the sequence {2^(n-2)}
def a : ℝ := 2^(2-2)
def b : ℝ := 2^(4-2)

-- Let μ be the mean of the sample set
def μ : ℝ := (a + 3 + 5 + 7) / 4

-- Variance calculation formula (S^2)
def variance (sample : List ℝ) : ℝ :=
  let n := sample.length
  let mean := (sample.foldl (+) 0) / n
  (sample.map (λ x => (x - mean) * (x - mean))).sum / n

-- Prove that the variance of the given sample set is 5
theorem variance_is_five : variance sample_set = 5 := by
  sorry

end variance_is_five_l235_235434


namespace triangle_circumcircles_ratio_l235_235969

theorem triangle_circumcircles_ratio
  (ABC : Triangle) (P Q R : Point) (BC CA AB : Segment)
  (omegaA omegaB omegaC : Circle)
  (X Y Z : Point)
  (hP_on_BC : P ∈ BC) (hQ_on_CA : Q ∈ CA) (hR_on_AB : R ∈ AB)
  (homegaA : ωA = circumcircle A Q R) (homegaB : ωB = circumcircle B R P) (homegaC : ωC = circumcircle C P Q)
  (hX_on_omegaA : X ∈ ωA) (hY_on_omegaB : Y ∈ ωB) (hZ_on_omegaC : Z ∈ ωC)
  (hX_on_AP : X ∈ (AP ∩ ωA - {A}))
  (hY_on_AP : Y ∈ (AP ∩ ωB - {A}))
  (hZ_on_AP : Z ∈ (AP ∩ ωC - {A})) :
  (distance Y X) / (distance X Z) = (distance B P) / (distance P C) :=
sorry

end triangle_circumcircles_ratio_l235_235969


namespace find_M_N_l235_235677

-- Define positive integers less than 10
def is_pos_int_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

-- Main theorem to prove M = 5 and N = 6 given the conditions
theorem find_M_N (M N : ℕ) (hM : is_pos_int_lt_10 M) (hN : is_pos_int_lt_10 N) 
  (h : 8 * (10 ^ 7) * M + 420852 * 9 = N * (10 ^ 7) * 9889788 * 11) : 
  M = 5 ∧ N = 6 :=
by {
  sorry
}

end find_M_N_l235_235677


namespace num_diagonals_convex_nonagon_l235_235873

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235873


namespace largest_intersection_l235_235477

-- Define the polynomial
def poly (x a : ℝ) : ℝ := x^6 - 8 * x^5 + 22 * x^4 + 6 * x^3 + a * x^2

-- Define the line
def line (x c : ℝ) : ℝ := 2 * x + c

-- Conditions
-- There are exactly four intersection points including a triple root.
def intersection_condition (a c : ℝ) : Prop := 
  ∃ p q : ℝ, (p = 2) ∧ (q = 3) ∧
  (\lamda x, poly x a = line x c).roots.count = 4 ∧ (\lamda x, poly x a = line x c).roots.contains p ∧ 
  (\lamda x, poly x a = line x c).roots.contains q

-- Statement to be proven
theorem largest_intersection (a c : ℝ) (h : intersection_condition a c) : 
  ∃ x : ℝ, eq ((λ x, x) (x^1-x > 2 → x)) 7 :=
sorry

end largest_intersection_l235_235477


namespace average_mpg_correct_l235_235375

def initial_odometer_reading : ℝ := 35420
def first_top_up_gallons : ℝ := 5
def first_refill_odometer_reading : ℝ := 35700
def first_refill_gallons : ℝ := 10
def second_refill_odometer_reading : ℝ := 36050
def second_refill_gallons : ℝ := 15
def final_odometer_reading : ℝ := 36600
def final_refill_gallons : ℝ := 25

def total_miles : ℝ := final_odometer_reading - initial_odometer_reading
def total_gallons_used : ℝ := first_top_up_gallons + first_refill_gallons + second_refill_gallons + final_refill_gallons
def average_mpg : ℝ := total_miles / total_gallons_used

theorem average_mpg_correct : average_mpg = 23.6 := by
  sorry

end average_mpg_correct_l235_235375


namespace quadratic_two_distinct_real_roots_for_all_m_value_of_expression_given_one_root_as_3_l235_235468

open Real

-- Part (1) proof statement
theorem quadratic_two_distinct_real_roots_for_all_m (m : ℝ) :
    let a := 1
    let b := 2 * m
    let c := m^2 - 2
    let Δ := b^2 - 4 * a * c
    Δ > 0 := by
  let a := 1
  let b := 2 * m
  let c := m^2 - 2
  let Δ := b^2 - 4 * a * c
  show Δ > 0
  sorry

-- Part (2) proof statement
theorem value_of_expression_given_one_root_as_3 (m : ℝ) :
    (3:ℝ)^2 + 2 * m * 3 + m^2 - 2 = 0 →
    2 * m^2 + 12 * m + 2043 = 2029 := by
  intro h
  sorry

end quadratic_two_distinct_real_roots_for_all_m_value_of_expression_given_one_root_as_3_l235_235468


namespace basketball_subs_remainder_l235_235685

theorem basketball_subs_remainder:
  let a_0 := 1 in
  let a_1 := 12 * 12 * a_0 in
  let a_2 := 12 * 11 * a_1 in
  let a_3 := 12 * 10 * a_2 in
  let a_4 := 12 * 9 * a_3 in
  let n := a_0 + a_1 + a_2 + a_3 + a_4 in
  n % 1000 = 953 :=
by
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let n := a_0 + a_1 + a_2 + a_3 + a_4
  have h0 : n = a_0 + a_1 + a_2 + a_3 + a_4 := rfl
  have h1 : a_0 = 1 := rfl
  have h2 : a_1 = 12 * 12 := rfl
  have h3 : a_2 = 12 * 11 * a_1 := rfl
  have h4 : a_3 = 12 * 10 * a_2 := rfl
  have h5 : a_4 = 12 * 9 * a_3 := rfl
  have hsum := calc
    n = 1 + 12 * 12 + 12 * 11 * (12 * 12) + 12 * 10 * (12 * 11 * (12 * 12)) + 12 * 9 * (12 * 10 * (12 * 11 * (12 * 12))) : by 
      simp [hsum]
  have hfinal : n % 1000 = 953 := by sorry
  exact hfinal

end basketball_subs_remainder_l235_235685


namespace sum_of_x_coordinates_in_region_l235_235369

noncomputable def points : List (ℕ × ℕ) := [(4, 15), (7, 25), (13, 40), (19, 45), (21, 55)]

def aboveLine1 (p : ℕ × ℕ) : Prop := p.2 > 2.5 * p.1 + 5
def belowLine2 (p : ℕ × ℕ) : Prop := p.2 < 3 * p.1 + 3

def inRegion (p : ℕ × ℕ) : Prop := aboveLine1 p ∧ belowLine2 p

def xCoordinatesInRegion (points : List (ℕ × ℕ)) : List ℕ :=
  points.filter inRegion |>.map Prod.fst

theorem sum_of_x_coordinates_in_region :
  xCoordinatesInRegion points = [13, 19] ∧ (List.sum (xCoordinatesInRegion points) = 32) :=
by
  sorry

end sum_of_x_coordinates_in_region_l235_235369


namespace centroid_trajectory_is_parabola_l235_235153

variables {a b c : ℝ} {d : ℝ}

noncomputable def midpoint : ℝ × ℝ := (0, 0)

noncomputable def centroid (x : ℝ) : ℝ × ℝ :=
  let A := (-d, 0)
  let B := (d, 0)
  let C := (x, a * x^2 + b * x + c)
  ((-d + d + x) / 3, (0 + 0 + (a * x^2 + b * x + c)) / 3)

theorem centroid_trajectory_is_parabola :
  ∀ x : ℝ, ∃ p q r : ℝ, (centroid x).snd = p * (x / 3)^2 + q * (x / 3) + r :=
sorry

end centroid_trajectory_is_parabola_l235_235153


namespace team_c_score_l235_235334

theorem team_c_score (points_A points_B total_points : ℕ) (hA : points_A = 2) (hB : points_B = 9) (hTotal : total_points = 15) :
  total_points - (points_A + points_B) = 4 :=
by
  sorry

end team_c_score_l235_235334


namespace second_time_apart_l235_235816

theorem second_time_apart 
  (glen_speed : ℕ) 
  (hannah_speed : ℕ)
  (initial_distance : ℕ) 
  (initial_time : ℕ)
  (relative_speed : ℕ)
  (hours_later : ℕ) :
  glen_speed = 37 →
  hannah_speed = 15 →
  initial_distance = 130 →
  initial_time = 6 →
  relative_speed = glen_speed + hannah_speed →
  hours_later = initial_distance / relative_speed →
  initial_time + hours_later = 8 + 30 / 60 :=
by
  intros
  sorry

end second_time_apart_l235_235816


namespace part1_tangent_line_part2_monotonicity_l235_235794

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (x ^ 2 - 2 * a * x) * Real.log x - x ^ 2 + 4 * a * x + 1

theorem part1_tangent_line (a : ℝ) (h : a = 0) :
  let e := Real.exp 1
  let f_x := f e 0
  let tangent_line := 4 * e - 3 * e ^ 2 + 1
  tangent_line = 4 * e * (x - e) + f_x :=
sorry

theorem part2_monotonicity (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → f (x) a > 0 ↔ a ≤ 0) ∧
  (∀ x : ℝ, 0 < x → x < a → f (x) a > 0 ↔ 0 < a ∧ a < 1) ∧
  (∀ x : ℝ, 1 < x → x < a → f (x) a < 0 ↔ a > 1) ∧
  (∀ x : ℝ, 0 < x → 1 < x → x < a → f (x) a < 0 ↔ (a > 1)) ∧
  (∀ x : ℝ, x > 1 → f (x) a > 0 ↔ (a < 1)) :=
sorry

end part1_tangent_line_part2_monotonicity_l235_235794


namespace train_length_l235_235718

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end train_length_l235_235718


namespace sum_of_integers_l235_235764

theorem sum_of_integers (S : Finset ℕ) (h : ∀ n ∈ S, 1.5 * n - 6.3 < 7.5) :
  S.sum id = 45 :=
sorry

end sum_of_integers_l235_235764


namespace sum_of_odd_integers_l235_235321

theorem sum_of_odd_integers (a₁ aₙ d n : ℕ) (h₁ : a₁ = 201) (h₂ : aₙ = 599) (h₃ : d = 2) (h₄ : aₙ = a₁ + (n - 1) * d) :
  (∑ i in finset.range(n), a₁ + i * d) = 80000 :=
by
  sorry

end sum_of_odd_integers_l235_235321


namespace triangle_ABC_angles_l235_235342

theorem triangle_ABC_angles
  (A B C D O : Point)
  (h1 : is_triangle A B C)
  (h2 : is_internal_bisector A C D B)
  (h3 : is_incenter O B C D)
  (h4 : is_circumcenter O A B C) :
  ∠A = 72 ∧ ∠B = 36 ∧ ∠C = 72 := by
  sorry

end triangle_ABC_angles_l235_235342


namespace simplest_square_root_l235_235329

theorem simplest_square_root :
  (∀ (x : ℝ), x = sqrt 9 → x = 3) ∧
  (∀ (y : ℝ), y = sqrt (1 / 3) → y = sqrt 3 / 3) ∧
  (∀ (z : ℝ), z = sqrt 5 → ¬ ∃ (w : ℝ), w * w = 5 ∧ w ≠ sqrt 5) ∧
  (∀ (k : ℝ), k = sqrt 12 → ∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ k = a * b) →
  (∀ (c : ℝ), c = sqrt 5 → ∀ (d : ℝ), (d = sqrt 9 ∧ d = sqrt (1 / 3) / sqrt 3 ∧ d = sqrt 12) → c = sqrt 5). 
by 
  sorry

end simplest_square_root_l235_235329


namespace distinct_six_digit_ababab_count_l235_235490

theorem distinct_six_digit_ababab_count : 
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  let valid_numbers := [10, 22, 34, 38, 46, 58, 62, 82, 86, 94, 55, 85] in
  (∀ n, n ∈ valid_numbers → ∃ p1 p2, p1 ∈ primes ∧ p2 ∈ primes ∧ p1 ≠ p2 ∧ n = p1 * p2) → 
  list.length valid_numbers = 12 :=
by
  sorry

end distinct_six_digit_ababab_count_l235_235490


namespace minimum_value_of_function_l235_235625

theorem minimum_value_of_function (x : ℝ) (hx : x > 0) :
  (∃ c : ℝ, ∀ y : ℝ, y > 0 → (y + 1/y) ≥ c ∧ ∀ x : ℝ, x > 0 → x + 1/x = c) := 
begin
  use 2,
  intros y hy,
  split,
  {
    apply Real.add_le_add,
    { exact (Real.one_div_pos.mpr hy), },
    { left, apply Real.one_div_pos.mpr,
      exact hy, }
  },
  {
    intros hx,
    suffices : x = 1,
    { simp [this] },
    field_simp at *,
    linarith,
  }
end

end minimum_value_of_function_l235_235625


namespace train_length_200_04_l235_235715

-- Define the constants
def speed_kmh : ℝ := 60     -- speed in km/h
def time_seconds : ℕ := 12  -- time in seconds

-- Define conversion factors
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Convert speed to m/s
def speed_ms : ℝ := (speed_kmh * km_to_m) / hr_to_s

-- Define the length of the train in meters
def length_of_train : ℝ := speed_ms * time_seconds

-- The theorem to prove
theorem train_length_200_04 : length_of_train = 200.04 := by
  sorry

end train_length_200_04_l235_235715


namespace parabola_equation_l235_235277

theorem parabola_equation (a b c d e f: ℤ) (ha: a = 2) (hb: b = 0) (hc: c = 0) (hd: d = -16) (he: e = -1) (hf: f = 32) :
  ∃ x y : ℝ, 2 * x ^ 2 - 16 * x + 32 - y = 0 ∧ gcd (abs a) (gcd (abs b) (gcd (abs c) (gcd (abs d) (gcd (abs e) (abs f))))) = 1 :=
by
  sorry

end parabola_equation_l235_235277


namespace binom_60_3_l235_235021

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235021


namespace nonagon_diagonals_count_l235_235832

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235832


namespace day_crew_fraction_l235_235731

theorem day_crew_fraction (D W : ℕ) (h1 : ∀ n, n = D / 4) (h2 : ∀ w, w = 4 * W / 5) :
  (D * W) / ((D * W) + ((D / 4) * (4 * W / 5))) = 5 / 6 :=
by 
  sorry

end day_crew_fraction_l235_235731


namespace number_of_incorrect_propositions_l235_235276

open Classical

-- Define the propositions
def p1 : Prop := (p ∨ q) → (p ∧ q)
def p2 : Prop := (x > 5) → (x^2 - 4 * x - 5 > 0)
def p3 : Prop := ¬(∀ x : ℝ, 2^x > x^2) = (∃ x : ℝ, 2^x ≤ x^2)
def p4 : Prop := ∃ x : ℝ, exp x = 1 + x

-- Define the truth values of each proposition
def is_incorrect (p : Prop) : Prop := ¬p

-- Main theorem to verify the number of incorrect propositions
theorem number_of_incorrect_propositions 
(h1: is_incorrect p1)
(h2: ¬is_incorrect p2)
(h3: is_incorrect p3)
(h4: ¬is_incorrect p4) :
  1 + 1 = 2 :=
by
  -- rest of the proof steps can be filled in, this is the theorem statement front
  sorry

end number_of_incorrect_propositions_l235_235276


namespace triangle_equilateral_of_repeat_rotation_l235_235122

noncomputable def point_seq (A : ℕ → Point) (P : ℕ → Point) : ℕ → Point
| 0       := P 0
| (n + 1) := rotate (point_seq n) (A (n + 1)) 120

theorem triangle_equilateral_of_repeat_rotation
  (A : ℕ → Point) (P : ℕ → Point)
  (hA_def : ∀ n ≥ 3, A (n + 1) = A (n - 2))
  (hP0 : P 1986 = P 0) :
  equilateral_triangle (A 1) (A 2) (A 3) :=
sorry

end triangle_equilateral_of_repeat_rotation_l235_235122


namespace nonagon_diagonals_count_l235_235898

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235898


namespace a_power_2018_plus_b_power_2018_eq_2_l235_235472

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

theorem a_power_2018_plus_b_power_2018_eq_2 (a b : ℝ) :
  (∀ x : ℝ, f x a b + f (1 / x) a b = 0) → a^2018 + b^2018 = 2 :=
by 
  sorry

end a_power_2018_plus_b_power_2018_eq_2_l235_235472


namespace problem_solution_l235_235239

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem problem_solution : a^10 + b^10 = 123 := sorry

end problem_solution_l235_235239


namespace initial_position_of_flea_l235_235374

theorem initial_position_of_flea : 
  (∃ K0 : ℝ, K0 + ((∑ i in finset.range 25, (2 * i + 1)) - (∑ i in finset.range 25, (2 * i + 2))) = -26.5) → 
  (K0 = -1.5) :=
begin
  intro h,
  rcases h with ⟨K0, h⟩,
  have : ∑ i in finset.range 25, (2 * i + 1) = 25 * 25,
  { sorry },
  have : ∑ i in finset.range 25, (2 * i + 2) = 50 * 25,
  { sorry },
  linarith,
end

end initial_position_of_flea_l235_235374


namespace nonagon_diagonals_count_l235_235910

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235910


namespace example_triangle_proof_l235_235531

noncomputable def triangle_proof_problem (A B C a b c : ℝ) (r : ℝ) : Prop :=
  (2 * Real.sin B = Real.sin A + Real.cos A * Real.tan C) →
  r = (Real.sqrt 3) / 2 →
  b = 4 →
  C = Real.pi / 3 ∧ a - c = -1

-- Example of how you would use the proof problem in a theorem
theorem example_triangle_proof(A B C a b c : ℝ) (r : ℝ) :
  triangle_proof_problem A B C a b c r :=
by
  intros h1 h2 h3
  split
  -- Proof for C = π / 3
  -- Proof for a - c = -1
  sorry

end example_triangle_proof_l235_235531


namespace nonagon_diagonals_count_l235_235911

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235911


namespace solution_set_l235_235562

noncomputable def f (x : ℝ) : ℝ := sorry  -- Define f according to the conditions

lemma odd_function (x : ℝ) : f (-x) = -f(x) := sorry  -- f is an odd function
lemma periodicity (x : ℝ) : f (x + 2) = -f(x) := sorry  -- f(x+2) = -f(x)
lemma initial_segment (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f(x) = (1/2) * x := sorry  -- f(x) = 1/2 * x for 0 ≤ x ≤ 1

theorem solution_set : {x : ℝ | f(x) = -1/2} = {x : ℝ | ∃ k : ℤ, x = 4 * k - 1} :=
by
  sorry

end solution_set_l235_235562


namespace intersection_point_exists_l235_235674

variable {t : ℝ}
def line_parametric (t: ℝ) := (x = 5 - 2 * t, y = 2, z = -4 - t)
def plane (x y z : ℝ) := 2 * x - 5 * y + 4 * z + 24 = 0

theorem intersection_point_exists (t : ℝ) :
  ∃ x y z, line_parametric t ∧ plane x y z ∧ x = 3 ∧ y = 2 ∧ z = -5 := 
by
  sorry

end intersection_point_exists_l235_235674


namespace binom_60_3_eq_34220_l235_235044

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235044


namespace average_age_of_dance_group_l235_235513

theorem average_age_of_dance_group
  (avg_age_children : ℕ)
  (avg_age_adults : ℕ)
  (num_children : ℕ)
  (num_adults : ℕ)
  (total_num_members : ℕ)
  (total_sum_ages : ℕ)
  (average_age : ℚ)
  (h_children : avg_age_children = 12)
  (h_adults : avg_age_adults = 40)
  (h_num_children : num_children = 8)
  (h_num_adults : num_adults = 12)
  (h_total_members : total_num_members = 20)
  (h_total_ages : total_sum_ages = 576)
  (h_average_age : average_age = 28.8) :
  average_age = (total_sum_ages : ℚ) / total_num_members :=
by
  sorry

end average_age_of_dance_group_l235_235513


namespace manufacturer_break_even_l235_235693

theorem manufacturer_break_even :
  let cost_per_component := 80
  let shipping_cost := 5
  let fixed_monthly_cost := 16500
  let min_price_per_component := 195
  let components_to_break_even := 150
  in min_price_per_component * components_to_break_even ≥ fixed_monthly_cost + (cost_per_component + shipping_cost) * components_to_break_even :=
  sorry

end manufacturer_break_even_l235_235693


namespace camera_guarantee_l235_235297

def battery_trials (b : Fin 22 → Bool) : Prop :=
  let charged := Finset.filter (λ i => b i) (Finset.univ : Finset (Fin 22))
  -- Ensuring there are exactly 15 charged batteries
  (charged.card = 15) ∧
  -- The camera works if any set of three batteries are charged
  (∀ (trials : Finset (Finset (Fin 22))),
   trials.card = 10 →
   ∃ t ∈ trials, (t.card = 3 ∧ t ⊆ charged))

theorem camera_guarantee :
  ∃ (b : Fin 22 → Bool), battery_trials b := by
  sorry

end camera_guarantee_l235_235297


namespace triangle_area_is_13_l235_235984

def vector := (ℚ × ℚ)

def a : vector := (4, -2)
def b : vector := (1, 6)

def area_of_parallelogram (v1 v2 : vector) : ℚ :=
  (v1.1 * v2.2 - v1.2 * v2.1).abs

def area_of_triangle (v1 v2 : vector) : ℚ :=
  1 / 2 * area_of_parallelogram v1 v2

theorem triangle_area_is_13 : area_of_triangle a b = 13 :=
  sorry

end triangle_area_is_13_l235_235984


namespace kickers_goals_in_first_period_l235_235754

theorem kickers_goals_in_first_period (K : ℕ) 
  (h1 : ∀ n : ℕ, n = K) 
  (h2 : ∀ n : ℕ, n = 2 * K) 
  (h3 : ∀ n : ℕ, n = K / 2) 
  (h4 : ∀ n : ℕ, n = 4 * K) 
  (h5 : K + 2 * K + (K / 2) + 4 * K = 15) : 
  K = 2 := 
by
  sorry

end kickers_goals_in_first_period_l235_235754


namespace binomial_60_3_eq_34220_l235_235035

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235035


namespace ratio_of_maxima_l235_235431

-- Definitions for the functions f and g
def f (x : ℝ) : ℝ := sqrt (1 - x) + sqrt (x + 3)
def g (x : ℝ) : ℝ := sqrt (x - 1) - sqrt (x - 3)

-- Definition of the maximum values M and N
def M : ℝ := 2 * sqrt 2  -- Maximum of f
def N : ℝ := sqrt 2      -- Maximum of g

-- The proof goal
theorem ratio_of_maxima : M / N = 2 := 
by sorry

end ratio_of_maxima_l235_235431


namespace binomial_60_3_l235_235030

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235030


namespace rectangle_area_l235_235194

-- Define the conditions as hypotheses in Lean 4
variable (x : ℤ)
variable (area : ℤ := 864)
variable (width : ℤ := x - 12)

-- State the theorem to prove the relation between length and area
theorem rectangle_area (h : x * width = area) : x * (x - 12) = 864 :=
by 
  sorry

end rectangle_area_l235_235194


namespace division_remainder_l235_235339

theorem division_remainder (n : ℕ) (h : n = 8 * 8 + 0) : n % 5 = 4 := by
  sorry

end division_remainder_l235_235339


namespace sum_of_altitudes_of_triangle_l235_235279

theorem sum_of_altitudes_of_triangle :
  let line_eq : ℝ → ℝ → Prop := λ x y, 8 * x + 10 * y = 80 in
  let intercepts : set (ℝ × ℝ) := { (10, 0), (0, 8) } in
  let vertices : set (ℝ × ℝ) := { (0, 0), (10, 0), (0, 8) } in
  let altitude1 : ℝ := 10 in
  let altitude2 : ℝ := 8 in
  let altitude3 : ℝ := 40 / Real.sqrt 41 in
  altitude1 + altitude2 + altitude3 = (40 / Real.sqrt 41) + 18 :=
by
  let line_eq : ℝ → ℝ → Prop := λ x y, 8 * x + 10 * y = 80
  let intercepts : set (ℝ × ℝ) := { (10, 0), (0, 8) }
  let vertices : set (ℝ × ℝ) := { (0, 0), (10, 0), (0, 8) }
  let altitude1 : ℝ := 10
  let altitude2 : ℝ := 8
  let altitude3 : ℝ := 40 / Real.sqrt 41
  show altitude1 + altitude2 + altitude3 = (40 / Real.sqrt 41) + 18
  sorry

end sum_of_altitudes_of_triangle_l235_235279


namespace palindromes_between_200_and_800_l235_235062

theorem palindromes_between_200_and_800 : 
  let p := λ n, 200 ≤ n ∧ n < 800 ∧ (Nat.digits 10 n = Nat.digits 10 n.reverse)
  in (∃ l : List ℕ, l.length = 60 ∧ ∀ n, n ∈ l → p n) :=
sorry

end palindromes_between_200_and_800_l235_235062


namespace range_of_a_l235_235476

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (x^2 - x - 2) / x^3

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1 → f a x1 ≥ g x2) ↔ a ≥ -2 :=
by
  sorry

end range_of_a_l235_235476


namespace fixed_point_and_minimum_distance_l235_235786

variables {P : Point} {A B Q : Point}
def line_l (p : Point) := p.x - p.y + 4 = 0
def circle_O (p : Point) := p.x ^ 2 + p.y ^ 2 = 4
def midpoint (A B Q : Point) := Q.x = (A.x + B.x) / 2 ∧ Q.y = (A.y + B.y) / 2
def tangent (a b : Point) := a ≠ b ∧ circle_O a = true ∧ circle_O b = true ∧ (line_through a b).intersects circle_O = {a, b}

theorem fixed_point_and_minimum_distance
  (hP : line_l P)
  (hAB : tangent A B)
  (hQ : midpoint A B Q) :
  ∃ (F : Point), F = (-1, 1) ∧ distance_from Q line_l = sqrt 2 := 
sorry

end fixed_point_and_minimum_distance_l235_235786


namespace ratio_rounded_to_34_l235_235503

noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)

theorem ratio_rounded_to_34 (a b : ℕ) (h1 : a > b) (h2 : arithmetic_mean a b = 3 * geometric_mean a b) : 
  Int.near (a.to_real / b.to_real) 34 :=
begin
  -- Sorry here implies the proof is omitted
  sorry
end

end ratio_rounded_to_34_l235_235503


namespace factorial_base_a5_zero_l235_235747

/-- 
  Prove that the coefficient a_5 in the factorial base representation
  of the number 801 is 0.
-/
theorem factorial_base_a5_zero : 
  ∃ (a_1 a_2 a_3 a_4 a_5 a_6 : ℕ), 
    801 = a_1 + a_2 * 2! + a_3 * 3! + a_4 * 4! + a_5 * 5! + a_6 * 6! ∧ 
    0 ≤ a_1 ∧ a_1 ≤ 1 ∧ 
    0 ≤ a_2 ∧ a_2 ≤ 2 ∧ 
    0 ≤ a_3 ∧ a_3 ≤ 3 ∧ 
    0 ≤ a_4 ∧ a_4 ≤ 4 ∧ 
    0 ≤ a_5 ∧ a_5 ≤ 5 ∧ 
    0 ≤ a_6 ∧ a_6 ≤ 6 ∧
    a_5 = 0 :=
by
  sorry

end factorial_base_a5_zero_l235_235747


namespace total_sweet_potatoes_l235_235581

theorem total_sweet_potatoes (sold_to_adams sold_to_lenon remaining : ℕ)
  (h_sold_to_adams : sold_to_adams = 20)
  (h_sold_to_lenon : sold_to_lenon = 15)
  (h_remaining : remaining = 45) :
  sold_to_adams + sold_to_lenon + remaining = 80 :=
by
  rw [h_sold_to_adams, h_sold_to_lenon, h_remaining]
  norm_num

end total_sweet_potatoes_l235_235581


namespace negation_necessary_but_not_sufficient_l235_235678

def P (x : ℝ) : Prop := |x - 2| ≥ 1
def Q (x : ℝ) : Prop := x^2 - 3 * x + 2 ≥ 0

theorem negation_necessary_but_not_sufficient (x : ℝ) :
  (¬ P x → ¬ Q x) ∧ ¬ (¬ Q x → ¬ P x) :=
by
  sorry

end negation_necessary_but_not_sufficient_l235_235678


namespace christmas_tree_partition_l235_235517

theorem christmas_tree_partition : 
  ∃ (partition_count : ℕ), 
  partition_count = 50 ∧
  (∃ (children : Finset ℕ) (trees : Finset ℕ),
  trees.card = 2 ∧ children.card = 5 ∧
  ∀ (partition : Finset (Finset ℕ)), 
    ∀ g1 g2 ∈ partition, g1 ≠ g2 ∧ g1.card ≠ 0 ∧ g2.card ≠ 0 ∧
    partition = {g1, g2} → 
    let swapped := {g2, g1} in 
    let rotated1 := g1.to_list.permutations.to_finset.map multiset.perm in
    let rotated2 := g2.to_list.permutations.to_finset.map multiset.perm in
    (partition.singleton.g1 ∈ rotated1 ∨ partition.singleton.g2 ∈ rotated1) ∧
    (partition.singleton.g1 ∈ rotated2 ∨ partition.singleton.g2 ∈ rotated2) ∧
    (swapped.singleton.g1 ∈ rotated1 ∨ swapped.singleton.g2 ∈ rotated1) ∧
    (swapped.singleton.g1 ∈ rotated2 ∨ swapped.singleton.g2 ∈ rotated2))
:= sorry

end christmas_tree_partition_l235_235517


namespace binomial_60_3_l235_235027

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235027


namespace nonagon_diagonals_l235_235840

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235840


namespace relationship_abc_l235_235439

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f (-x) = -f (x)
axiom periodic_f_4 : ∀ x, f (x - 4) = -f (x)
axiom increasing_f : ∀ x y, (0 ≤ x ∧ x ≤ 2) → (0 ≤ y ∧ y ≤ 2) → (x < y) → f (x) < f(y)

def a := f(6)
def b := f(161)
def c := f(45)

theorem relationship_abc : a < c ∧ c < b :=
by
  sorry

end relationship_abc_l235_235439


namespace b_dot_c_l235_235214

open Real

variable (a b c : ℝ^3)
variable (a_norm : ∥a∥ = 2)
variable (b_norm : ∥b∥ = 3)
variable (ab_norm : ∥a + b∥ = 5)
variable (c_def : c = 2 • a + 3 • b + 4 • (a × b))

theorem b_dot_c : b ⋅ c = 39 := by
  sorry

end b_dot_c_l235_235214


namespace binom_60_3_l235_235013

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235013


namespace no_power_of_two_l235_235406

theorem no_power_of_two (cards : Fin 88889 → ℕ)
  (h_range : ∀ i, 11111 ≤ cards i ∧ cards i ≤ 99999) :
  let A := ∑ i, cards i * 10^(5 * (88889 - 1 - i))
  in ¬ ∃ k : ℕ, A = 2^k :=
by
  sorry

end no_power_of_two_l235_235406


namespace incorrect_statement_l235_235331

-- Defining the conditions
def sum_of_interior_angles_of_quadrilateral (q : Type) [quadrilateral q] : angle_sum q = 360 := sorry
def sum_of_exterior_angles_of_quadrilateral (q : Type) [quadrilateral q] : exterior_angle_sum q = 360 := sorry
def area_ratio_of_similar_polygons {P Q : Type} [polygon P] [polygon Q] (h : similar P Q) : area P / area Q = similarity_ratio P Q := sorry
def symmetric_point_coordinates (P : Point) (origin : Point) : sym_point P origin = (-P.x, -P.y) := sorry
def median_of_triangle (T : Type) [triangle T] (M : median T) : parallel M.base M.third_side ∧ length M = 1/2 * length M.third_side := sorry

-- Theorem that needs to be proven
theorem incorrect_statement (q : Type) [quadrilateral q] (P Q : Type) [polygon P] [polygon Q] (T : Type) [triangle T] 
  (h : similar P Q) (M : median T) : 
  ¬ (area P / area Q = similarity_ratio P Q) := 
begin
  -- The actual proof was skipped
  sorry
end

end incorrect_statement_l235_235331


namespace cylinder_volume_difference_l235_235370

theorem cylinder_volume_difference :
  let heightA := 12
  let circumferenceA := 10
  let radiusA := circumferenceA / (2 * Real.pi)
  let volumeA := Real.pi * radiusA^2 * heightA
  let heightB := 10
  let circumferenceB := 12
  let radiusB := circumferenceB / (2 * Real.pi)
  let volumeB := Real.pi * radiusB^2 * heightB
  Real.pi * ((volumeB - volumeA).abs / Real.pi) = 60 := by
  sorry

end cylinder_volume_difference_l235_235370


namespace find_triangle_angles_l235_235065

noncomputable def angles_of_triangle (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
  let θ := real.acos (1 + 2 * real.sqrt 6 / 6)
  let α := (180 - θ) / 2
  (θ, α, α)

theorem find_triangle_angles :
  let a : ℝ := 3
  let b : ℝ := 3
  let c : ℝ := real.sqrt 8 - real.sqrt 3
  angles_of_triangle a b c = (11.53, 84.235, 84.235) :=
by
  sorry

end find_triangle_angles_l235_235065


namespace ratio_AB_AD_eq_4_l235_235600

variables {AB AD EFGH : Type} [Mul EFGH] [Div EFGH] [OfNat EFGH 4.5] [OfNat EFGH 0.3] [OfNat EFGH 0.6]
variable (area : Type → EFGH) [HasZero EFGH] [HasMul EFGH]

def fraction_area_sharing (frac1 frac2 : EFGH) (area1 area2 : EFGH) : Prop :=
  (frac1 * area1 = frac2 * area2)

def ratio_length (ratio : EFGH) (len1 len2 : EFGH) : Prop :=
  (len1 = ratio * len2)

theorem ratio_AB_AD_eq_4.5 {s : EFGH} 
  (overlap_area : EFGH)
  (rectangle_area : EFGH)
  (square_area : EFGH)
  (h1 : fraction_area_sharing 0.6 0.3 rectangle_area square_area)
  (h2 : overlap_area = 0.5 * square_area)
  (x y : EFGH)
  (h3 : rectangle_area = x * y)
  (h4 : y = s / 3)
  (h5 : x = 1.5 * s) :
  ratio_length 4.5 x y := sorry

end ratio_AB_AD_eq_4_l235_235600


namespace circum_BOD_passes_through_M_l235_235629

noncomputable def midpoint (P Q R S: Type*) [Affine P Q R S] := sorry

variable (P Q R S O : Type*) [MetricSpace P] [Affine P] [Circle P]
  (inscribed : isInscribedQuadrilateral P Q R S)
  (center : Circle.center O)
  (no_diagonals : ¬ (isOnDiagonal O P R ∨ isOnDiagonal O Q S))
  (N : Point)
  (midpoint_N : N = midpoint P R Q S)
  (circ_AOC : isCircumcircle O P R N)

theorem circum_BOD_passes_through_M :
  ∃ M, (M = midpoint P Q) ∧ isCircumcircle O Q S M := sorry

end circum_BOD_passes_through_M_l235_235629


namespace tan_alpha_beta_analytical_expression_range_of_f_l235_235773

-- Let's define the conditions based on the problem description.
variable {α β x y : ℝ}
variable {f : ℝ → ℝ}

-- Problem transformations and assertions.
def condition1 : Prop := sin (2 * α + β) = 3 * sin β
def condition2 : Prop := tan α = x
def condition3 : Prop := tan β = y
def condition4 : Prop := y = f x

-- Questions to prove.
theorem tan_alpha_beta (h1 : condition1) (h2 : condition2) (h3 : condition3) : tan (α + β) = 2 * tan α := sorry

theorem analytical_expression (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : f x = x / (1 + 2*x^2) := sorry

theorem range_of_f (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : 0 < α) (h6 : α < π / 3) : 
  (0 : ℝ) < f x ∧ (f x ≤ sqrt 2 / 4) := sorry

end tan_alpha_beta_analytical_expression_range_of_f_l235_235773


namespace total_money_at_least_108_l235_235946

-- Definitions for the problem
def tram_ticket_cost : ℕ := 1
def passenger_coins (n : ℕ) : Prop := n = 2 ∨ n = 5

-- Condition that conductor had no change initially
def initial_conductor_money : ℕ := 0

-- Condition that each passenger can pay exactly 1 Ft and receive change
def can_pay_ticket_with_change (coins : List ℕ) : Prop := 
  ∀ c ∈ coins, passenger_coins c → 
    ∃ change : List ℕ, (change.sum = c - tram_ticket_cost) ∧ 
      (∀ x ∈ change, passenger_coins x)

-- Assume we have 20 passengers with only 2 Ft and 5 Ft coins
def passengers_coins : List (List ℕ) :=
  -- Simplified representation
  List.replicate 20 [2, 5]

noncomputable def total_passenger_money : ℕ :=
  (passengers_coins.map List.sum).sum

-- Lean statement for the proof problem
theorem total_money_at_least_108 : total_passenger_money ≥ 108 :=
sorry

end total_money_at_least_108_l235_235946


namespace minimal_abs_diff_l235_235929

theorem minimal_abs_diff (a b : ℕ) (h : a * b - 5 * a + 6 * b = 522) : abs (a - b) = 12 :=
sorry

end minimal_abs_diff_l235_235929


namespace distinct_diagonals_in_nonagon_l235_235901

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235901


namespace nonagon_diagonals_l235_235846

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235846


namespace cost_of_book_l235_235206

-- Definitions based on the conditions
def cost_pen : ℕ := 4
def cost_ruler : ℕ := 1
def fifty_dollar_bill : ℕ := 50
def change_received : ℕ := 20
def total_spent : ℕ := fifty_dollar_bill - change_received

-- Problem Statement: Prove the cost of the book
theorem cost_of_book : ∀ (cost_pen cost_ruler total_spent : ℕ), 
  total_spent = 50 - 20 → cost_pen = 4 → cost_ruler = 1 →
  (total_spent - (cost_pen + cost_ruler) = 25) :=
by
  intros cost_pen cost_ruler total_spent h1 h2 h3
  sorry

end cost_of_book_l235_235206


namespace tetrahedron_regular_if_five_spheres_five_spheres_exist_if_tetrahedron_regular_l235_235295

-- Definition for a tetrahedron being regular
def is_regular_tetrahedron (S A B C : Point) : Prop :=
  dist S A = dist S B ∧
  dist S A = dist S C ∧
  dist S A = dist A B ∧
  dist S A = dist A C ∧
  dist S A = dist B C

-- Problem (a): Given that there exist five spheres each tangent to the edges of tetrahedron SABC or their extensions, prove that SABC is a regular tetrahedron
theorem tetrahedron_regular_if_five_spheres (S A B C : Point)
  (h : ∃ (spheres : list Sphere), 
          list.length spheres = 5 ∧ 
          (∀ s ∈ spheres, tangent_to_edges_or_extensions s [S, A, B, C])) : 
  is_regular_tetrahedron S A B C :=
sorry

-- Problem (b): Prove that for every regular tetrahedron, five such spheres exist
theorem five_spheres_exist_if_tetrahedron_regular (S A B C : Point)
  (h : is_regular_tetrahedron S A B C) : 
  ∃ (spheres : list Sphere), 
    list.length spheres = 5 ∧ 
    (∀ s ∈ spheres, tangent_to_edges_or_extensions s [S, A, B, C]) :=
sorry

end tetrahedron_regular_if_five_spheres_five_spheres_exist_if_tetrahedron_regular_l235_235295


namespace evaluate_f_l235_235111

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem evaluate_f : f (f (f (-1))) = Real.pi + 1 :=
by
  -- Proof goes here
  sorry

end evaluate_f_l235_235111


namespace ratio_HQ_HR_l235_235954

-- Given an acute scalene triangle ABC
variables (A B C D E F H P Q R : Type) 
-- Assume the necessary properties of the points and orthocenter
variables (BC CA AB EF : Set (Set (A))) 
variables [noncomputable] [DecidableEq A]

-- Conditions as per problem
axiom acute_scalene_triangle : is_acute_scalene_triangle A B C
axiom point_on_side_D : D ∈ BC
axiom point_on_side_E : E ∈ CA
axiom point_on_side_F : F ∈ AB
axiom altitude_AD : ⊥ A D BC
axiom altitude_BE : ⊥ B E CA
axiom altitude_CF : ⊥ C F AB
axiom orthocenter : orthocenter A B C = H
axiom P_on_EF : P ∈ EF
axiom Q_on_EF : Q ∈ EF
axiom AP_perp_EF : ⊥ A P EF
axiom HQ_perp_EF : ⊥ H Q EF
axiom DP_intersect_QH = R

-- Need to prove
theorem ratio_HQ_HR : HQ / HR = 1 :=
sorry

end ratio_HQ_HR_l235_235954


namespace max_interesting_pairs_7x7_14_l235_235590

def is_neighboring (cell1 cell2 : ℕ × ℕ) : Prop :=
  (abs (cell2.1 - cell1.1) = 1 ∧ cell2.2 = cell1.2) ∨
  (abs (cell2.2 - cell1.2) = 1 ∧ cell2.1 = cell1.1)

def is_marked (grid : ℕ × ℕ → bool) (cell : ℕ × ℕ) : Prop :=
  grid cell

def interesting_pair (grid : ℕ × ℕ → bool) (cell1 cell2 : ℕ × ℕ) : Prop :=
  is_neighboring cell1 cell2 ∧ (is_marked grid cell1 ∨ is_marked grid cell2)

def max_interesting_pairs (grid : ℕ × ℕ → bool) : ℕ :=
  (∑ i in finset.range 7, ∑ j in finset.range 7, 
    if i < 6 ∧ interesting_pair grid (i, j) (i + 1, j) then 1 else 0) +
  (∑ i in finset.range 7, ∑ j in finset.range 7,
    if j < 6 ∧ interesting_pair grid (i, j) (i, j + 1) then 1 else 0)

theorem max_interesting_pairs_7x7_14 : 
  ∀ (grid : ℕ × ℕ → bool), (∑ i in finset.range 7, ∑ j in finset.range 7, if grid (i, j) then 1 else 0) = 14 →
  max_interesting_pairs grid ≤ 55 :=
begin
  sorry
end

end max_interesting_pairs_7x7_14_l235_235590


namespace max_apartments_l235_235608

theorem max_apartments (num_buildings : ℕ) (num_floors : ℕ) (apartments_per_floor : ℕ) : 
  1 ≤ num_buildings ∧ num_buildings ≤ 22 ∧ num_floors = 6 ∧ apartments_per_floor = 5 → 
  (num_buildings * num_floors * apartments_per_floor) = 660 :=
by
  assume h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  simp [h1, h3, h5, h6]
  sorry -- proof steps are not required

end max_apartments_l235_235608


namespace nonagon_diagonals_l235_235849

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235849


namespace sin_D_eq_5sqrt41_over_41_l235_235191

noncomputable def sin_of_d (D E F : Triangle) (h : ∠ E = 90) (h₁ : 4 * Real.sin (angle D E F) = 5 * Real.cos (angle D E F)) : Real :=
  Real.sin (angle D E F)

theorem sin_D_eq_5sqrt41_over_41 (D E F : Triangle) (h : ∠ E = 90) (h₁ : 4 * Real.sin (angle D E F) = 5 * Real.cos (angle D E F)) :
  sin_of_d D E F h h₁ = 5 * Real.sqrt 41 / 41 :=
sorry

end sin_D_eq_5sqrt41_over_41_l235_235191


namespace ellipse_properties_l235_235959

namespace EllipseProof

-- Define the two points
def point1 : ℝ × ℝ := (0, -Real.sqrt 3)
def point2 : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the condition for the sum of distances
def sum_of_distances (P : ℝ × ℝ) : Prop :=
  Real.dist P point1 + Real.dist P point2 = 4

-- Define the equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- The following theorem statement conforms to the conditions and proves the correct answers:
theorem ellipse_properties :
  (∀ P : ℝ × ℝ, sum_of_distances P → ellipse_equation P.1 P.2) ∧
  (∀ P : ℝ × ℝ, sum_of_distances P →
    (P = (2, 0) ∨ P = (-2, 0) ∨ P = (0, 1) ∨ P = (0, -1)) ∧
    P.1 * P.1 / 4 + P.2 * P.2 = 1 ∧
    (Real.dist point1 point2 / 2 = 2 ∧
    Real.dist (0, 0) point1 = Real.sqrt 3 ∧
    Real.dist (0, 0) point2 = Real.sqrt 3 ∧
    Real.sqrt (4 - 3) = 1 → True) →
    (Real.dist point1 point2 = Real.sqrt 3 * 2) ∧
    (∀ P : ℝ × ℝ, sum_of_distances P →
      (P.1 * P.1 / 4 + P.2 * P.2 = 1))): Prop :=
sorry

end EllipseProof

end ellipse_properties_l235_235959


namespace number_of_diagonals_in_nonagon_l235_235823

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235823


namespace total_cost_of_basketballs_and_volleyballs_l235_235932

variable {m n : ℝ}

theorem total_cost_of_basketballs_and_volleyballs (m n : ℝ) : 3 * m + 7 * n = 3m + 7n :=
by
  sorry

end total_cost_of_basketballs_and_volleyballs_l235_235932


namespace PQ_composition_l235_235555

def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem PQ_composition : P (Q (P (Q (P (Q 2))))) = 54 := 
by sorry

end PQ_composition_l235_235555


namespace maximum_value_f_g_larger_than_2x_squared_minus_2x_for_x_ge_2_l235_235441

-- Definition of the function f
def f (x : ℝ) : ℝ := Real.log x - x + 1

-- Definition of the function g
def g (x : ℝ) : ℝ := Real.exp x - 1

-- Assertion 1: Find the maximum value of f(x)
theorem maximum_value_f : (∃ x, f x = 0) ∧ (∀ x, x ≠ 1 → f x ≤ 0) := sorry

-- Assertion 2: Prove that for all x in [2, +∞), g(x) > 2x(x-1)
theorem g_larger_than_2x_squared_minus_2x_for_x_ge_2 (x : ℝ) (h : 2 ≤ x) : g x > 2 * x * (x - 1) := sorry

end maximum_value_f_g_larger_than_2x_squared_minus_2x_for_x_ge_2_l235_235441


namespace nonagon_diagonals_count_l235_235896

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235896


namespace largest_prime_divisor_base8_l235_235759

theorem largest_prime_divisor_base8 (n : ℕ) (h : n = 2 * 8^8 + 1 * 8^7 + 2 * 8^5 + 1 * 8^4 + 2 * 8^2 + 2 * 8^1 + 2 * 8^0) :
  largest_prime_divisor n = 17830531 :=
sorry

end largest_prime_divisor_base8_l235_235759


namespace lunch_meeting_probability_l235_235205

theorem lunch_meeting_probability :
  let x_range := 0 .. 60;
  let y_range := 0 .. 60;
  let total_area := (60 : ℝ) * 60;
  let meet_area := (10 : ℝ) * 60;
  (meet_area / total_area) = (1 / 6) :=
by sorry

end lunch_meeting_probability_l235_235205


namespace fraction_calculation_l235_235384

theorem fraction_calculation : (4 / 9 + 1 / 9) / (5 / 8 - 1 / 8) = 10 / 9 := by
  sorry

end fraction_calculation_l235_235384


namespace range_a_plus_2b_l235_235471

-- Defining the function f(x) = |lg x|
def f (x : ℝ) : ℝ := abs (Real.log10 x)

-- State the problem: if 0 < a < b and f(a) = f(b),
-- then the range of a + 2b is (3, +∞).
theorem range_a_plus_2b (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : f(a) = f(b)) :
  ∀ y, (∃ a b, 0 < a ∧ a < b ∧ f(a) = f(b) ∧ y = a + 2 * b) → 3 < y :=
sorry

end range_a_plus_2b_l235_235471


namespace triangle_is_isosceles_l235_235176

theorem triangle_is_isosceles 
  (A B C : ℝ)
  (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_triangle : A + B + C = π)
  (h_condition : (Real.sin B) * (Real.sin C) = (Real.cos (A / 2)) ^ 2) :
  (B = C) :=
sorry

end triangle_is_isosceles_l235_235176


namespace total_path_length_A_l235_235341

-- Define the necessary variables and conditions
variables {α : ℝ} (hα : 0 < α ∧ α < π / 3)

-- Define the unit circle and the vertices A, B, C positioning
def unit_circle : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def A : ℝ × ℝ := sorry  -- Given on the circumference of the unit circle
def B : ℝ × ℝ := (0, 0)  -- Center of the unit circle
def C : ℝ × ℝ := sorry  -- Given on the circumference of the unit circle

def angle_ABC : ℝ := 2 * α
def is_on_circumference (p : ℝ × ℝ) : Prop := p ∈ unit_circle

-- Define the problem statement
theorem total_path_length_A  :
  ∑ i in (range 33), (2 * π / 3 - 2 * α) + ∑ i in (range 33), (2 * sin α * π / 3) = 
  22 * π * (1 + sin α) - 66 * α :=
sorry

end total_path_length_A_l235_235341


namespace exponent_equality_l235_235501

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l235_235501


namespace sum_of_powers_sequence_l235_235238

theorem sum_of_powers_sequence (a b : ℝ) 
  (h₁ : a + b = 1)
  (h₂ : a^2 + b^2 = 3)
  (h₃ : a^3 + b^3 = 4)
  (h₄ : a^4 + b^4 = 7)
  (h₅ : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sum_of_powers_sequence_l235_235238


namespace find_prob_B_l235_235457

variable {Ω : Type*}
variable [ProbabilitySpace Ω]
variable (A B : Event Ω)

axiom independent_events : Indep A B
axiom P_AB : ℙ[A ∩ B] = 0.36
axiom P_not_A : ℙ[Aᶜ] = 0.6

theorem find_prob_B : ℙ[B] = 0.9 :=
by sorry

end find_prob_B_l235_235457


namespace value_of_a_8_l235_235576

noncomputable def S (n : ℕ) : ℕ := n^2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then S n else S n - S (n - 1)

theorem value_of_a_8 : a 8 = 15 := 
by
  sorry

end value_of_a_8_l235_235576


namespace area_ratio_of_similar_polygons_incorrect_area_ratio_l235_235332

theorem area_ratio_of_similar_polygons (r : ℝ) (h_r_nonneg : 0 ≤ r) :
  ∀ (A B : Type) [metric_space A] [metric_space B] [is_similar A B r],
  (area A / area B) = r^2 :=
sorry

theorem incorrect_area_ratio :
  ¬∀ (A B : Type) [metric_space A] [metric_space B] [is_similar A B (r : ℝ)], 
  (area A / area B) = r :=
by 
  intro h 
  apply area_ratio_of_similar_polygons
  sorry

end area_ratio_of_similar_polygons_incorrect_area_ratio_l235_235332


namespace max_min_f_tan_theta_l235_235815

noncomputable theory

variables {x θ : ℝ}
variables (a b : ℝ)

def vector_m (a : ℝ) (x : ℝ) : ℝ × ℝ := (a * real.cos x, real.cos x)
def vector_n (b : ℝ) (x : ℝ) : ℝ × ℝ := (2 * real.cos x, b * real.sin x)
def f (a b x : ℝ) : ℝ := (vector_m a x).1 * (vector_n b x).1 + (vector_m a x).2 * (vector_n b x).2

axiom a_condition : f a b 0 = 2
axiom b_condition : f a b (real.pi / 3) = 1 / 2 + real.sqrt 3 / 2

theorem max_min_f :
  ∃ (a b : ℝ), f a b (real.pi / 2) = sqrt 2 + 1 ∧ f a b 0 = 0 :=
sorry

theorem tan_theta :
  ∃ (θ : ℝ), (f 1 2 (θ / 2) = 3 / 2) → (real.tan θ = - (4 + real.sqrt 7) / 3) :=
sorry

end max_min_f_tan_theta_l235_235815


namespace binom_60_3_l235_235011

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235011


namespace minimum_distance_origin_to_line_l235_235116

theorem minimum_distance_origin_to_line :
  let O := (0 : ℝ, 0 : ℝ),
      A := 2,
      B := -1,
      C := 1,
      line := λ (x y: ℝ) => A * x + B * y + C = 0,
      distance : ℝ := |A * (O.1) + B * (O.2) + C| / sqrt (A^2 + B^2)
  in distance = sqrt 5 / 5 :=
by
  sorry

end minimum_distance_origin_to_line_l235_235116


namespace election_result_l235_235521

noncomputable theory
def totalVotes : ℕ := 1000000
def invalidPercentage : ℚ := 0.20
def candidateAPercentage : ℚ := 0.35
def candidateBPercentage : ℚ := 0.30
def candidateCPercentage : ℚ := 1 - candidateAPercentage - candidateBPercentage
def recountPercentage : ℚ := 0.10

def validVotes := totalVotes * (1 - invalidPercentage)
def candidateAInitialVotes := validVotes * candidateAPercentage
def candidateBInitialVotes := validVotes * candidateBPercentage
def candidateCInitialVotes := validVotes * candidateCPercentage
def votesTransferred := candidateAInitialVotes * recountPercentage

def candidateAFinalVotes := candidateAInitialVotes - votesTransferred
def candidateBFinalVotes := candidateBInitialVotes + votesTransferred
def candidateCFinalVotes := candidateCInitialVotes

theorem election_result :
  candidateAFinalVotes = 252000 ∧
  candidateBFinalVotes = 268000 ∧
  candidateCFinalVotes = 280000 :=
by sorry

end election_result_l235_235521


namespace tangent_line_eq_monotonicity_intervals_range_of_a_l235_235473

-- Definitions needed for the problems
def f (x a : ℝ) := 2 * Real.exp x + 2 * a * x - a^2

-- Problem 1: Tangent line at (0, f(0)) when a = 1
theorem tangent_line_eq (a : ℝ) (h : a = 1) :
  ∀ x y : ℝ,  y - f 0 1 = 4 * x ↔ 4 * x - y + 1 = 0 :=
sorry

-- Problem 2: Monotonicity of f(x)
theorem monotonicity_intervals (a : ℝ) :
  (∀ x : ℝ, a ≥ 0 → (f x a)' > 0) ∧
  (∀ x : ℝ, a < 0 → ((f x a)' > 0 ↔ x > Real.log (-a))) ∧
  (∀ x : ℝ, a < 0 → ((f x a)' < 0 ↔ x < Real.log (-a))) :=
sorry

-- Problem 3: Range of a given f(x) ≥ x^2 - 3 for x ≥ 0
theorem range_of_a (a : ℝ) :
  (∀ x, x ≥ 0 → f x a ≥ x^2 - 3) → ln 3 - 3 ≤ a ∧ a ≤ Real.sqrt 5 :=
sorry

end tangent_line_eq_monotonicity_intervals_range_of_a_l235_235473


namespace equal_number_of_problems_l235_235073

variable {P : Type}
variable [fintype P] (problems_first_day problems_second_day : P → ℕ)

noncomputable def total_problems_second_day : ℕ :=
  ∑ p in fintype.elems P, problems_second_day p

axiom condition (p : P) : problems_first_day p = total_problems_second_day - problems_second_day p

theorem equal_number_of_problems (p : P) :
  problems_first_day p + problems_second_day p = total_problems_second_day := by
  have total_sum : 
    problems_first_day p + problems_second_day p = 
    (total_problems_second_day - problems_second_day p) + problems_second_day p := by
    rw[condition p]
  rw[add_sub_cancel'_right] at total_sum
  exact total_sum

end equal_number_of_problems_l235_235073


namespace volume_regular_quadrangular_pyramid_l235_235270

theorem volume_regular_quadrangular_pyramid (Q S V : ℝ) 
  (h1 : Q > 0) 
  (h2 : S > 0) 
  (h3 : V = 1 / 6 * sqrt (Q * (S ^ 2 - Q ^ 2))) : 
  V = 1 / 6 * sqrt (Q * (S ^ 2 - Q ^ 2)) := 
by {
  sorry
}

end volume_regular_quadrangular_pyramid_l235_235270


namespace cos_A_value_l235_235987

theorem cos_A_value (a b c : ℝ) (A B C : ℝ) 
  (cos_C : ℝ) (h_cosC : cos C = 2 / 3) (h_a_eq_3b : a = 3 * b)
  (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c : c = Real.sqrt 6 * b) 
  (h_triangle_ineq : b + c > a ∧ a + c > b ∧ a + b > c) : 
  cos A = -Real.sqrt 6 / 6 :=
by
  sorry

end cos_A_value_l235_235987


namespace sequence_eventually_periodic_l235_235768

def S (x y z : ℤ) : ℤ × ℤ × ℤ := (x * y - x * z, y * z - y * x, z * x - z * y)

theorem sequence_eventually_periodic {a b c : ℤ} (h_abc : a * b * c > 1) 
  (S_m : ℕ → ℤ × ℤ × ℤ) (S_m_def : ∀ m, S_m (m + 1) = S (S_m m).1 (S_m m).2.1 (S_m m).2.2) 
  (n n₀ : ℕ) (h_n : n ≥ n₀) :
  ∃ n₀ k : ℕ, 0 < k ∧ k ≤ a * b * c ∧ S_m (n + k) ≡ S_m n [MOD a * b * c] :=
sorry

end sequence_eventually_periodic_l235_235768


namespace distinct_diagonals_convex_nonagon_l235_235851

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235851


namespace hundreds_digit_of_factorial_diff_is_zero_l235_235088

theorem hundreds_digit_of_factorial_diff_is_zero :
  (∃ k : ℤ, 30! - 25! = 1000 * k) → 
  ((30! - 25!) / 100 % 10 = 0) := 
by
  sorry

end hundreds_digit_of_factorial_diff_is_zero_l235_235088


namespace not_possible_to_form_tetrahedron_l235_235603

/-- Given six line segments such that any three of them can form a triangle,
    prove that these line segments cannot form a tetrahedron. -/
theorem not_possible_to_form_tetrahedron (a b c d e f : ℝ)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (h4 : a + d > e) (h5 : a + e > d) (h6 : d + e > a)
  (h7 : b + d > f) (h8 : b + f > d) (h9 : d + f > b)
  (h10 : c + e > f) (h11 : c + f > e) (h12 : e + f > c) :
  ¬ (∃ (A B C D : Type) 
       (dist : A → A → ℝ)
       (h13 : dist A B = a) (h14 : dist A C = b) (h15 : dist A D = c)
       (h16 : dist B C = d) (h17 : dist B D = e) (h18 : dist C D = f), true) :=
sorry

end not_possible_to_form_tetrahedron_l235_235603


namespace solve1_solve2_solve3_solve4_l235_235260

theorem solve1 (x : ℝ) : x^2 = 49 → (x = 7 ∨ x = -7) := by
  sorry

theorem solve2 (x : ℝ) : (2 * x + 3)^2 = 4 * (2 * x + 3) →
  (x = -3/2 ∨ x = 1/2) := by
  sorry

theorem solve3 (x : ℝ) : 2 * x^2 + 4 * x - 3 = 0 →
  (x = (-2 + sqrt 10) / 2 ∨ x = (-2 - sqrt 10) / 2) := by
  sorry

theorem solve4 (x : ℝ) : (x + 8) * (x + 1) = -12 →
  (x = -4 ∨ x = -5) := by
  sorry

end solve1_solve2_solve3_solve4_l235_235260


namespace probability_math_majors_consecutive_l235_235641

theorem probability_math_majors_consecutive :
  let total_people := 12
  let math_majors := 5
  let physics_majors := 4
  let biology_majors := 3
  let total_ways := choose 11 4 * fact 4
  let consecutive_ways := 12
  probability (total_ways : ℚ) (consecutive_ways : ℚ) = (1 / 660 : ℚ) :=
by
  let total_people := 12
  let math_majors := 5
  let physics_majors := 4
  let biology_majors := 3
  let total_ways := choose 11 4 * fact 4
  let consecutive_ways := 12
  have total_ways_nonzero : (total_ways : ℚ) ≠ 0 := by sorry
  have h : probability (total_ways : ℚ) (consecutive_ways : ℚ) = consecutive_ways / total_ways := by sorry
  rw [h]
  norm_num
  sorry  -- Justify the final step and calculation

end probability_math_majors_consecutive_l235_235641


namespace prob_X_leq_2_l235_235136

-- Define a discrete random variable X with given distributions
def X_distribution (i : ℕ) : ℚ :=
  if i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 then 1 / 4 else 0

theorem prob_X_leq_2 :
  (X_distribution 1 + X_distribution 2 = 1 / 2) :=
by
  unfold X_distribution
  simp
  norm_num
  sorry

end prob_X_leq_2_l235_235136


namespace differentiable_function_zero_l235_235681

noncomputable def f : ℝ → ℝ := sorry

theorem differentiable_function_zero (f : ℝ → ℝ) (h_diff : ∀ x ≥ 0, DifferentiableAt ℝ f x)
  (h_f0 : f 0 = 0) (h_fun : ∀ x ≥ 0, ∀ y ≥ 0, (x = y^2) → deriv f x = f y) : 
  ∀ x ≥ 0, f x = 0 :=
by
  sorry

end differentiable_function_zero_l235_235681


namespace binomial_60_3_eq_34220_l235_235033

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235033


namespace tan_2α_value_β_value_l235_235791

-- Given conditions
variables (α β : ℝ)
axiom sin_α : sin α = 4 * sqrt 3 / 7
axiom cos_β_minus_α : cos (β - α) = 13 / 14
axiom α_interval : 0 < α ∧ α < π / 2
axiom β_interval : 0 < β ∧ β < α

-- Proof Problem 1: prove tan 2α = - (8 * sqrt 3) / 47
theorem tan_2α_value : tan (2 * α) = - (8 * sqrt 3) / 47 :=
by {
  sorry
}

-- Proof Problem 2: prove β = π / 3
theorem β_value : β = π / 3 :=
by {
  sorry
}

end tan_2α_value_β_value_l235_235791


namespace find_seventh_term_l235_235970

theorem find_seventh_term :
  ∃ r : ℚ, ∃ (a₁ a₇ a₁₀ : ℚ), 
    a₁ = 12 ∧ 
    a₁₀ = 78732 ∧ 
    a₇ = a₁ * r^6 ∧ 
    a₁₀ = a₁ * r^9 ∧ 
    a₇ = 8748 :=
by
  sorry

end find_seventh_term_l235_235970


namespace triangle_angle_bisector_l235_235118

noncomputable def intersect_and_bisect (A B: ℝ×ℝ) (MN: ℝ -> ℝ) (C : ℝ×ℝ) : Prop :=
  let midpoint_AD := λ A D : ℝ×ℝ, (A.1 + D.1)/2 = A.1 ∧ (A.2 + D.2)/2 = D.2 in
  let is_perpendicular := λ A D MN, (MN (A.1 + (A.2 - D.2)/(MN A.1))) = (A.2 + (D.2 - A.2)) in
  let extend_line := λ A D B1, A.1 = D.1 ∨ (A.2 = (MN (A.1 + ((D.2 - A.2)/(MN A.1)))) + 2 * (D.2 - A.2)) in
  let segment_connect := λ A B1 B MN C, (MN((B1.1 + B.1) / 2)) = ((B1.2 + B.2) / 2) in
  ∃ A D B1, midpoint_AD A D ∧ is_perpendicular A D MN ∧ extend_line A D B1 ∧ ∃ B, segment_connect A B1 B MN C ∧ B.1 > A.1 ∧ B.2 = A.2

theorem triangle_angle_bisector (A B: ℝ×ℝ) (MN: ℝ -> ℝ) (C : ℝ×ℝ) :
  ∃ (A D B1: ℝ×ℝ), intersect_and_bisect A B MN C :=
sorry

end triangle_angle_bisector_l235_235118


namespace tan_half_angle_inequality_l235_235532

variables {A B C : ℝ} 

-- Define the angles A, B, and C such that they make up a triangle
-- i.e., 0 < A, B, C < π and A + B + C = π.
def is_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π

-- Define the given inequality to be proven.
theorem tan_half_angle_inequality (h : is_triangle A B C) :
  (Real.tan (B / 2)) * (Real.tan (C / 2)) ^ 2 ≥ 4 * (Real.tan (A / 2)) * ((Real.tan (A / 2)) * (Real.tan (C / 2)) - 1) :=
by sorry

end tan_half_angle_inequality_l235_235532


namespace range_of_a_l235_235170

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), x > 0 → x / (x ^ 2 + 3 * x + 1) ≤ a) → a ≥ 1 / 5 :=
by
  sorry

end range_of_a_l235_235170


namespace inheritance_amount_l235_235582

theorem inheritance_amount (tax_paid : ℝ) (total_taxes : ℝ) :
  tax_paid = 18000 →
  total_taxes = 0.3625 →
  let x := tax_paid / total_taxes in
  x = 49655 :=
by
  intros
  let x := tax_paid / total_taxes
  have h : x = 49655 := sorry
  exact h

end inheritance_amount_l235_235582


namespace probability_five_consecutive_math_majors_l235_235642

theorem probability_five_consecutive_math_majors 
  (people : Fin 12 → ℕ)
  (mMajors : Fin 5 → Fin 12) 
  (pMajors : Fin 4 → Fin 12)
  (bMajors : Fin 3 → Fin 12)
  (distinct_majors : Function.Injective mMajors)
  (majors_disjoint1 : ∀ i, ¬ (mMajors i ∈ pMajors '' Finset.univ))
  (majors_disjoint2 : ∀ i, ¬ (pMajors i ∈ bMajors '' Finset.univ))
  : ((∑ (i : Fin 12), if ∀ j : Fin 5, mMajors j = i + j % 12 then 1 else 0).toRat / (792).toRat = 1 / 66) :=
by sorry

end probability_five_consecutive_math_majors_l235_235642


namespace smallest_positive_angle_equiv_l235_235068

theorem smallest_positive_angle_equiv (deg : ℝ) (radians : ℝ) (pi : ℝ) 
    (h1: deg = -600) 
    (h2: pi = real.pi) 
    (h3: radians = deg / 180 * pi) : radians = 2 * pi / 3 := 
by 
    sorry

end smallest_positive_angle_equiv_l235_235068


namespace t_12_eq_1705_l235_235215

def B1 (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | _ => T (n - 1)

def B2 (n : ℕ) : ℕ := 
  match n with
  | 1 => 0
  | 2 => 1
  | 3 => 2
  | _ => B1 (n - 1)

def B3 (n : ℕ) : ℕ := 
  match n with
  | 1 => 0
  | 2 => 0
  | 3 => 1
  | _ => B2 (n - 1)

def B4 (n : ℕ) : ℕ := 0

def T (n : ℕ) : ℕ := 
  match n with
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | n + 1 => T n + T (n - 1) + T (n - 2)

theorem t_12_eq_1705 : (T 12) = 1705 := by
  sorry

end t_12_eq_1705_l235_235215


namespace find_nat_numbers_for_divisibility_l235_235086

theorem find_nat_numbers_for_divisibility :
  ∃ (a b : ℕ), (7^3 ∣ a^2 + a * b + b^2) ∧ (¬ 7 ∣ a) ∧ (¬ 7 ∣ b) ∧ (a = 1) ∧ (b = 18) := by
  sorry

end find_nat_numbers_for_divisibility_l235_235086


namespace nonagon_diagonals_count_l235_235830

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235830


namespace min_value_of_expression_l235_235787

theorem min_value_of_expression (x y : ℝ) (h : x^2 + x * y + y^2 = 3) : x^2 - x * y + y^2 ≥ 1 :=
by 
sorry

end min_value_of_expression_l235_235787


namespace maximum_spherical_triangle_sum_l235_235591

-- Define the sphere, points and conditions
variable (A B C D : Point)
variable (R : ℝ) (hR : R = 4)
variable (hAB_AC : ⟪AB, AC⟫ = 0)
variable (hAC_AD : ⟪AC, AD⟫ = 0)
variable (hAD_AB : ⟪AD, AB⟫ = 0)

-- The resulting proof statement
theorem maximum_spherical_triangle_sum :
  let S_ABC := spherical_triangle_area A B C R
  let S_ACD := spherical_triangle_area A C D R
  let S_ADB := spherical_triangle_area A D B R
  S_ABC + S_ACD + S_ADB = 32 :=
sorry

end maximum_spherical_triangle_sum_l235_235591


namespace each_wolf_needs_to_kill_one_deer_l235_235302

-- Conditions
def wolves_out_hunting : ℕ := 4
def additional_wolves : ℕ := 16
def wolves_total : ℕ := wolves_out_hunting + additional_wolves
def meat_per_wolf_per_day : ℕ := 8
def days_no_hunt : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat needed for all wolves over five days.
def total_meat_needed : ℕ := wolves_total * meat_per_wolf_per_day * days_no_hunt
-- Calculate total number of deer needed to meet the meat requirement.
def deer_needed : ℕ := total_meat_needed / meat_per_deer
-- Calculate number of deer each hunting wolf needs to kill.
def deer_per_wolf : ℕ := deer_needed / wolves_out_hunting

-- The proof statement
theorem each_wolf_needs_to_kill_one_deer : deer_per_wolf = 1 := 
by { sorry }

end each_wolf_needs_to_kill_one_deer_l235_235302


namespace min_abs_diff_l235_235115

def f (x : ℝ) : ℝ := 2 * Real.sin ((π / 2) * x + π / 5)

theorem min_abs_diff {x x1 x2 : ℝ} (h : ∀ x, f x1 ≤ f x ∧ f x ≤ f x2) : |x1 - x2| = 2 :=
sorry

end min_abs_diff_l235_235115


namespace cersei_cotton_candies_left_l235_235737

theorem cersei_cotton_candies_left : 
  (total_purchased: ℕ) 
  (brother_sister_each: ℕ) 
  (fraction_for_cousin: ℚ) 
  (eaten: ℕ) : 
  total_purchased = 50 → 
  brother_sister_each = 5 → 
  fraction_for_cousin = 1/4 → 
  eaten = 12 → 
  ( 
    let remaining1 := total_purchased - 2 * brother_sister_each in 
    let remaining2 := remaining1 - remaining1 * fraction_for_cousin in 
    remaining2 - eaten
  ) = 18 := 
by {
  intros total_purchased brother_sister_each fraction_for_cousin eaten;
  intros h1 h2 h3 h4;

  -- Substitute known values
  let remaining1 := total_purchased - 2 * brother_sister_each;
  let remaining2 := remaining1 - remaining1 * fraction_for_cousin;

  -- Calculate explicit values
  have remaining1_val : remaining1 = 50 - 2 * 5 := by rw [h1, h2];
  have remaining2_val : remaining2 = 40 - 40 * (1 / 4) := by rw [remaining1_val, h3];
  have remaining2_val_simplified : remaining2 = 30 := by norm_num [remaining2_val];

  -- Final remaining candies
  show remaining2 - eaten = 18;
  rw [remaining2_val_simplified, h4];
  norm_num;
  };

end cersei_cotton_candies_left_l235_235737


namespace evaluate_expression_l235_235078

theorem evaluate_expression :
    let odd_sum := (1012 : ℕ) ^ 2
    let even_sum := (1010 : ℕ) * (1011 : ℕ)
    odd_sum - even_sum ^ 2 = -104271921956 :=
by
  let odd_sum := 1012 ^ 2
  let even_sum := 1010 * 1011
  show odd_sum - even_sum ^ 2 = -104271921956
  sorry

end evaluate_expression_l235_235078


namespace sum_of_digits_of_factorials_of_fibs_l235_235383

-- Define the Fibonacci sequence used in the problem.
def fibs : List Nat :=
  [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

-- Function to calculate the factorial of a given number.
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Function to calculate the sum of the digits of a given number.
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n
  else sumOfDigits (n / 10) + (n % 10)

-- Calculate the sum of the digits of the factorial values for the Fibonacci sequence.
def sumOfDigitsOfFactorials (seq : List Nat) : Nat :=
  seq.map (λ n => sumOfDigits (factorial n)).sum

theorem sum_of_digits_of_factorials_of_fibs :
  sumOfDigitsOfFactorials fibs = 240 :=
by
  sorry

end sum_of_digits_of_factorials_of_fibs_l235_235383


namespace equation_of_parabola_exists_vertical_line_l235_235433

-- Define the conditions
def parabola_vertex_origin : Prop := ∀ (x y : ℝ), y^2 = 4 * x ↔ ...

def point_on_parabola_distance (m : ℝ) : Prop := let P := (4, m) in ...

-- Define theorems to prove the answers
theorem equation_of_parabola (p m : ℝ) (h_focus_x : p > 0) 
  (h1 : parabola_vertex_origin) (h2 : point_on_parabola_distance m) : y^2 = 4 * x := 
sorry

theorem exists_vertical_line (m : ℝ) (p : ℝ) (A := (4, 0)) 
  (M : ℝ × ℝ) (h1 : parabola_vertex_origin) (h2 : point_on_parabola_distance m) 
  (h3 : M ≠ (0, 0)) : ∃ (a : ℝ), a = 3 :=
sorry

end equation_of_parabola_exists_vertical_line_l235_235433


namespace polynomial_functional_equation_l235_235084

theorem polynomial_functional_equation (P : ℝ → ℝ) :
  (∀ x y : ℝ, P (x^2 - y^2) = P (x + y) * P (x - y)) →
  (P = (λ t, 0) ∨ P = (λ t, 1) ∨ ∃ (n : ℕ), P = (λ t, t ^ n)) :=
by
  -- proof goes here
  sorry

end polynomial_functional_equation_l235_235084


namespace find_length_AD_l235_235202

noncomputable def AB : ℝ := 15
noncomputable def CD : ℝ := 8
noncomputable def angle_BAD : ℝ := 60
noncomputable def angle_ABC : ℝ := 30
noncomputable def angle_BCD : ℝ := 30

theorem find_length_AD (A B C D : Type) (hAB : AB = 15) (hCD : CD = 8) 
    (hBAD : angle_BAD = 60) (hABC : angle_ABC = 30) (hBCD : angle_BCD = 30) :
    ∃ (AD : ℝ), AD = 3.5 :=
by
  use 3.5
  sorry

end find_length_AD_l235_235202


namespace vector_problem_l235_235166

open real

variables (A B C P : E)
variables [inner_product_space ℝ E]

def vector_AB := B - A
def vector_AC := C - A
def vector_AP := P - A
def vector_CA := A - C
def vector_CP := C - P

theorem vector_problem 
  (h1: inner_product_space.dot_product (vector_AB A B) (vector_AC A C) = 4)
  (h2: inner_product_space.norm (vector_AP A P) = 1) :
  inner_product_space.norm (vector_AB A B) = 2 
  ∧ 
  (exists angle : ℝ, (∀ theta, cos θ ≤ 1) → 
    let dot_product_CP_AB := inner_product_space.dot_product (vector_CP C P) (vector_AB A B) 
    in max dot_product_CP_AB = -2) := 
sorry

end vector_problem_l235_235166


namespace find_possible_value_of_m_l235_235807

open Nat

def M (m : ℕ) := { x : ℕ | ∃ n : ℕ, n > 0 ∧ x = m / n }

def has_eight_subsets (m : ℕ) : Prop :=
  2^(Set.card (M m)) = 8

def is_possible_value_of_m (m : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 10 ∧ m ∈ \{ x : ℕ | x > 0 \}

theorem find_possible_value_of_m (m : ℕ) (h : is_possible_value_of_m m) (h_subsets : has_eight_subsets m) : m = 4 :=
sorry

end find_possible_value_of_m_l235_235807


namespace genetic_variation_problem_l235_235728

-- Define each condition A through D as logical statements
def A_condition : Prop := 
  seedless_trait_of_seedless_tomatoes_cannot_be_inherited ∧ 
  seedless_watermelons_are_sterile ∧ 
  seedless_trait_of_watermelons_can_be_inherited

def B_condition : Prop := 
  ¬(homologous_chromosomes_in_somatic_cells_of_haploids) ∧ 
  haploid_plants_are_weaker_than_normal_plants

def C_condition : Prop := 
  transgenic_technology_involves_directly_introducing_exogenous_genes

def D_condition : Prop := 
  (∀ disease, appears_in_only_one_generation → ¬genetic_disease) ∧ 
  (∃ disease, appears_in_several_generations → genetic_disease)

-- The problem statement: Prove that A_condition is true and others are false
theorem genetic_variation_problem : 
  A_condition ∧ ¬B_condition ∧ ¬C_condition ∧ ¬D_condition :=
by
  -- The proof would be filled here
  sorry

end genetic_variation_problem_l235_235728


namespace smallest_even_abundant_gt_12_l235_235652

-- Define proper divisors of a number
def proper_divisors (n : ℕ) : List ℕ := (List.range (n - 1)).filter (λ d, d > 0 ∧ n % d = 0)

-- Check if a number is abundant
def is_abundant (n : ℕ) : Prop := (proper_divisors n).sum > n

-- Smallest even number that is greater than 12
def smallest_even_number_gt (m : ℕ) : ℕ := Nat.find (λ n, n > m ∧ n % 2 = 0)

-- Problem statement
theorem smallest_even_abundant_gt_12 : smallest_even_number_gt 12 = 18 :=
by
  sorry

end smallest_even_abundant_gt_12_l235_235652


namespace casey_nail_decorating_time_l235_235388

/-- Given the conditions:
1. Casey wants to apply three coats: a base coat, a coat of paint, and a coat of glitter.
2. Each coat takes 20 minutes to apply.
3. Each coat requires 20 minutes of drying time before the next one can be applied.

Prove that the total time taken by Casey to finish decorating her fingernails and toenails is 120 minutes.
-/
theorem casey_nail_decorating_time
  (application_time : ℕ)
  (drying_time : ℕ)
  (num_coats : ℕ)
  (total_time : ℕ)
  (h_app_time : application_time = 20) 
  (h_dry_time : drying_time = 20)
  (h_num_coats : num_coats = 3)
  (h_total_time_eq : total_time = num_coats * (application_time + drying_time)) :
  total_time = 120 :=
sorry

end casey_nail_decorating_time_l235_235388


namespace name_tag_area_l235_235282

-- Define the side length of the square
def side_length : ℕ := 11

-- Define the area calculation for a square
def square_area (side : ℕ) : ℕ := side * side

-- State the theorem: the area of a square with side length of 11 cm is 121 cm²
theorem name_tag_area : square_area side_length = 121 :=
by
  sorry

end name_tag_area_l235_235282


namespace distinct_diagonals_convex_nonagon_l235_235852

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235852


namespace pyramid_cone_volume_ratio_l235_235271

noncomputable def volume_ratio (R H : ℝ) (α : ℝ) : ℝ :=
  (2 * Real.sin α) / π

theorem pyramid_cone_volume_ratio (R H : ℝ) (α : ℝ) :
  -- Conditions
  α ∈ Icc 0 (π/2) → -- α is between 0 and π/2 inclusive 
  R > 0 →
  H > 0 →
  -- Question: Prove that the volume ratio is
  volume_ratio R H α = (2 * Real.sin α) / π :=
by
  intros hα hR hH
  sorry

end pyramid_cone_volume_ratio_l235_235271


namespace binomial_symmetry_binomial_alternating_signs_l235_235661

noncomputable theory

-- Problem 1
theorem binomial_symmetry (n k : ℕ) (hk : k ≤ n): 
  binom n k = binom n (n - k) :=
sorry

-- Problem 2
theorem binomial_alternating_signs (a b : ℝ) (n : ℕ) : 
  (a - b)^n = ∑ k in finset.range (n + 1), (-1) ^ k * binom n k * (a^(n - k)) * (b^k) :=
sorry

end binomial_symmetry_binomial_alternating_signs_l235_235661


namespace area_AEKS_is_correct_l235_235053

def small_triangle_area (s : ℝ) : ℝ := (real.sqrt 3 / 4) * s^2

def area_polygon_AEKS : ℝ := 
  let s := 1
  let area_small_triangle := small_triangle_area s
  let area_AES := 16 * area_small_triangle
  let area_EKJ := area_small_triangle
  let area_KJST := 6 * area_small_triangle
  let area_KJS := (1 / 2) * area_KJST
  area_AES + area_EKJ + area_KJS

theorem area_AEKS_is_correct : area_polygon_AEKS = 5 * real.sqrt 3 := by
  sorry

end area_AEKS_is_correct_l235_235053


namespace Jazmin_strips_width_l235_235539

theorem Jazmin_strips_width (w1 w2 g : ℕ) (h1 : w1 = 44) (h2 : w2 = 33) (hg : g = Nat.gcd w1 w2) : g = 11 := by
  -- Markdown above outlines:
  -- w1, w2 are widths of the construction paper
  -- h1: w1 = 44
  -- h2: w2 = 33
  -- hg: g = gcd(w1, w2)
  -- Prove g == 11
  sorry

end Jazmin_strips_width_l235_235539


namespace eccentricity_of_hyperbola_l235_235149

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := sqrt (a^2 + b^2)
  let E := c / a
  in E

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : let l := λ x, - (a / b) * (x - c) in
       let M := (a^2 / c, ab / c) in
       let N := (a^2 c / (a^2 - b^2), abc / (b^2 - a^2)) in
       ∥N∥ = 2 * ∥M∥) :
  hyperbola_eccentricity a b ha hb = 2 * sqrt 3 / 3 :=
sorry

end eccentricity_of_hyperbola_l235_235149


namespace points_in_rectangle_l235_235466

-- Definition of a rectangle and the condition
structure Rectangle where
  length : ℝ
  width : ℝ

-- Definition of a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

-- The 3 x 4 rectangle and 6 points within it
def rect : Rectangle := ⟨3, 4⟩
def points_in_rect (pts : Fin 6 → Point) : Prop :=
  ∀ i, (0 ≤ pts i).x ∧ (pts i).x ≤ rect.width ∧ (0 ≤ pts i).y ∧ (pts i).y ≤ rect.length

-- The main theorem statement
theorem points_in_rectangle :
  ∀ (pts : Fin 6 → Point), points_in_rect pts → 
  ∃ i j, i ≠ j ∧ distance (pts i) (pts j) ≤ real.sqrt 5 :=
by
  sorry

end points_in_rectangle_l235_235466


namespace binary_last_digit_of_77_is_1_l235_235745

theorem binary_last_digit_of_77_is_1 : 
  (Nat.toDigits 2 77).head = 1 :=
by
  sorry

end binary_last_digit_of_77_is_1_l235_235745


namespace intersection_of_A_and_B_l235_235450

variable (x y : ℝ)

def A := {y : ℝ | ∃ x > 1, y = Real.log x / Real.log 2}
def B := {y : ℝ | ∃ x > 1, y = (1 / 2) ^ x}

theorem intersection_of_A_and_B :
  (A ∩ B) = {y : ℝ | 0 < y ∧ y < 1 / 2} :=
by sorry

end intersection_of_A_and_B_l235_235450


namespace min_expression_value_l235_235400

theorem min_expression_value (a b c : ℝ) (h_sum : a + b + c = -1) (h_abc : a * b * c ≤ -3) :
  3 ≤ (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) :=
sorry

end min_expression_value_l235_235400


namespace inv_38_mod_53_l235_235451

theorem inv_38_mod_53 (h : 15 * 31 % 53 = 1) : ∃ x : ℤ, 38 * x % 53 = 1 ∧ (x % 53 = 22) :=
by
  sorry

end inv_38_mod_53_l235_235451


namespace min_trams_spy_sees_l235_235647

/-- 
   Vasya stood at a bus stop for some time and saw 1 bus and 2 trams.
   Buses run every hour.
   After Vasya left, a spy stood at the bus stop for 10 hours and saw 10 buses.
   Given these conditions, the minimum number of trams that the spy could have seen is 5.
-/
theorem min_trams_spy_sees (bus_interval tram_interval : ℕ) 
  (vasya_buses vasya_trams spy_buses spy_hours min_trams : ℕ) 
  (h1 : bus_interval = 1)
  (h2 : vasya_buses = 1)
  (h3 : vasya_trams = 2)
  (h4 : spy_buses = spy_hours)
  (h5 : spy_buses = 10)
  (h6 : spy_hours = 10)
  (h7 : ∀ t : ℕ, t * tram_interval ≤ 2 → 2 * bus_interval ≤ 2)
  (h8 : min_trams = 5) :
  min_trams = 5 := 
sorry

end min_trams_spy_sees_l235_235647


namespace portia_high_school_students_l235_235596

theorem portia_high_school_students (P L : ℕ) (h1 : P = 4 * L) (h2 : P + L = 2500) : P = 2000 := by
  sorry

end portia_high_school_students_l235_235596


namespace math_problem_l235_235549

def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem math_problem :
  P (Q (P (Q (P (Q 2))))) = 1944 * real.sqrt 6 ^ (1/4) :=
by
  sorry

end math_problem_l235_235549


namespace lcm_of_48_and_14_is_56_l235_235645

theorem lcm_of_48_and_14_is_56 :
  ∀ n : ℕ, (n = 48 ∧ Nat.gcd n 14 = 12) → Nat.lcm n 14 = 56 :=
by
  intro n h
  sorry

end lcm_of_48_and_14_is_56_l235_235645


namespace count_3_digit_numbers_with_digit_product_36_l235_235161

theorem count_3_digit_numbers_with_digit_product_36 : 
  ∃ n, n = 21 ∧ ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (a * b * c = 36) → 
  nat.perm_count_3_digit_numbers_with_digit_product_36 a b c = n :=
by
  sorry

noncomputable def nat.perm_count_3_digit_numbers_with_digit_product_36 (a b c : ℕ) : ℕ :=
  if h : a * b * c = 36 then
    match finset.card (finset.permutations {a, b, c}) with
    | 0 => 0
    | k => k
  else 0

end count_3_digit_numbers_with_digit_product_36_l235_235161


namespace finite_non_friends_iff_l235_235103

def isFriend (u n : ℕ) : Prop :=
  ∃ N : ℕ, N % n = 0 ∧ (N.digits 10).sum = u

theorem finite_non_friends_iff (n : ℕ) : (∃ᶠ u in at_top, ¬ isFriend u n) ↔ ¬ (3 ∣ n) := 
by
  sorry

end finite_non_friends_iff_l235_235103


namespace correct_statements_l235_235229

-- Definitions
noncomputable def f (x b c : ℝ) : ℝ := abs x * x + b * x + c

-- Proof statements
theorem correct_statements (b c : ℝ) :
  (b > 0 → ∀ x y : ℝ, x ≤ y → f x b c ≤ f y b c) ∧
  (b < 0 → ¬ (∀ x : ℝ, ∃ m : ℝ, f x b c = m)) ∧
  (b = 0 → ∀ x : ℝ, f (x) b c = f (-x) b c) ∧
  (∃ x1 x2 x3 : ℝ, f x1 b c = 0 ∧ f x2 b c = 0 ∧ f x3 b c = 0) :=
sorry

end correct_statements_l235_235229


namespace new_members_combined_weight_l235_235515

variables (avg_weight_original : ℕ) (num_people : ℕ) (increase_avg_weight : ℕ)
          (weight_replaced1 weight_replaced2 weight_replaced3 : ℕ)
          (new_avg_weight : ℕ) (new_total_weight : ℕ) (weight_remaining_original : ℕ) (weight_new_members : ℕ)

def combined_weight_of_new_members :=
  avg_weight_original = 70 ∧ 
  num_people = 8 ∧ 
  increase_avg_weight = 6 ∧ 
  weight_replaced1 = 50 ∧ 
  weight_replaced2 = 65 ∧ 
  weight_replaced3 = 75 ∧ 
  new_avg_weight = 76 ∧ 
  new_total_weight = 608 ∧ 
  weight_remaining_original = 370 ∧ 
  weight_new_members = 238
 
theorem new_members_combined_weight 
  (h : combined_weight_of_new_members) : weight_new_members = 238 :=
sorry

end new_members_combined_weight_l235_235515


namespace simplify_expression_l235_235743

variable (x : ℝ)

theorem simplify_expression (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ( ((x + 2)^3 * (x^2 - 2 * x + 4)^3 / (x^3 + 8)^3)^4 * 
    ((x - 2)^3 * (x^2 + 2 * x + 4)^3 / (x^3 - 8)^3)^4 ) = 1 := 
begin
  sorry
end

end simplify_expression_l235_235743


namespace domain_of_log_function_l235_235274

theorem domain_of_log_function : 
  (∀ x : ℝ, (∃ y : ℝ, y = log 2 (x + 1)) ↔ x > -1) :=
begin
  sorry
end

end domain_of_log_function_l235_235274


namespace technician_round_trip_completion_l235_235712

theorem technician_round_trip_completion (D : ℝ) (h0 : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center := 0.30 * D
  let traveled := to_center + from_center
  traveled / round_trip * 100 = 65 := 
by
  sorry

end technician_round_trip_completion_l235_235712


namespace cone_section_max_area_l235_235940

noncomputable def max_area_of_cone_section 
    (radius_of_sector : ℝ) (central_angle : ℝ) : ℝ :=
begin
  let r := (radius_of_sector * central_angle) / (2 * π),
  let max_area := ( r * radius_of_sector ) / 2,
  exact max_area
end

theorem cone_section_max_area 
    (radius_of_sector : ℝ) (central_angle : ℝ) (h₁ : radius_of_sector = 2)
    (h₂ : central_angle = 5 * π / 3) :
  max_area_of_cone_section radius_of_sector central_angle = 2 :=
by sorry

end cone_section_max_area_l235_235940


namespace calculation_equality_l235_235380

theorem calculation_equality : ((8^5 / 8^2) * 4^4) = 2^17 := by
  sorry

end calculation_equality_l235_235380


namespace problem_solution_l235_235177

noncomputable def b_from_sine_and_cosines (a : ℝ) (angle_a : ℝ) (angle_c : ℝ) : ℝ :=
  by sorry

noncomputable def max_area (a : ℝ) (angle_a : ℝ) : ℝ :=
  by sorry

theorem problem_solution :
  ∀ (A B C : Type*)
    (a b c : ℝ)
    (angle_A angle_C : ℝ),
    (angle_A = 45 * (Real.pi / 180)) →
    (a = 6) →
    (angle_C = 105 * (Real.pi / 180)) →
    (b = b_from_sine_and_cosines a angle_A angle_C) ∧
    (max_area a angle_A = 9 * (1 + Real.sqrt 2)) :=
by
  intros A B C a b c angle_A angle_C h1 h2 h3
  split
  {
    apply eq_of_heq
    sorry,
  }
  {
    apply eq_of_heq
    sorry,
  }

end problem_solution_l235_235177


namespace div_count_64n3_l235_235108

-- We assume n is a positive integer such that 150n^2 has exactly 150 divisors.
def meets_conditions (n : ℕ) : Prop :=
  ∃ counts : List ℕ, 
    150 = List.foldl (*) 1 counts ∧
    List.foldl (*) 1 (List.map (λ x, x + 1) (counts.map (λ k, if k > 0 then k else 1))) = 150 ∧
    counts.length = 3

theorem div_count_64n3 {n : ℕ} (h : meets_conditions n) : 
  let counts := List.map (λ k, if k > 0 then k else 1) [3, 3, 1] in
  List.foldl (*) 1 (List.map (λ x, x + 1) (counts.map (λ k, k * 3))) = 70 :=
by
  sorry

end div_count_64n3_l235_235108


namespace part1_part2_l235_235806

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x^2 - 3 * x + 2 = 0 }

theorem part1 (a : ℝ) : (A a = ∅) ↔ (a > 9/8) := sorry

theorem part2 (a : ℝ) : 
  (∃ x, A a = {x}) ↔ 
  (a = 0 ∧ A a = {2 / 3})
  ∨ (a = 9 / 8 ∧ A a = {4 / 3}) := sorry

end part1_part2_l235_235806


namespace probability_blue_marbles_l235_235072

open Classical

variable (x y r1 r2 : ℕ)
variable (h1 : x + y = 30)
variable (h2 : r1 * r2 = 96 ∨ r1 * r2 = 64)
variable (bx := x - r1)
variable (by := y - r2)

theorem probability_blue_marbles (h : h1 ∧ h2) : ∃ (p q : ℕ), p / q = bx * by / (x * y) ∧ Nat.gcd p q = 1 ∧ p + q = 28 := by
  have h15_15 : x = 15 ∧ y = 15 := sorry
  have h20_10 : x = 20 ∧ y = 10 := sorry
  cases h2 with
  | inl h96 =>
    have : r1 = 12 ∧ r2 = 8 ∨ r1 = 8 ∧ r2 = 12 := sorry
    use [3, 25]
    split
    apply ratio_eqn -- Assuming 'ratio_eqn' is the condition validation helper
    apply gcd_prime -- Assuming 'gcd_prime' is the helper for relatively prime check
    done
  | inr h64 =>
    have : r1 = 16 ∧ r2 = 4 ∨ r1 = 8 ∧ r2 = 8 := sorry
    use [3, 25]
    split
    apply ratio_eqn
    apply gcd_prime
    done

end probability_blue_marbles_l235_235072


namespace probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l235_235443

noncomputable def p_a : ℝ := 0.18
noncomputable def p_b : ℝ := 0.5
noncomputable def p_b_given_a : ℝ := 0.2
noncomputable def p_c : ℝ := 0.3
noncomputable def p_c_given_a : ℝ := 0.4
noncomputable def p_c_given_b : ℝ := 0.6

noncomputable def p_a_and_b : ℝ := p_a * p_b_given_a
noncomputable def p_a_and_b_and_c : ℝ := p_c_given_a * p_a_and_b
noncomputable def p_a_and_b_given_c : ℝ := p_a_and_b_and_c / p_c
noncomputable def p_a_and_c_given_b : ℝ := p_a_and_b_and_c / p_b

theorem probability_a_and_b_and_c : p_a_and_b_and_c = 0.0144 := by
  sorry

theorem probability_a_and_b_given_c : p_a_and_b_given_c = 0.048 := by
  sorry

theorem probability_a_and_c_given_b : p_a_and_c_given_b = 0.0288 := by
  sorry

end probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l235_235443


namespace max_value_of_f_l235_235227

variable (x : ℝ)

-- Define the function to be maximized
def f (x : ℝ) := (x^2 + 2 - (x^4 + 4 * x^2).sqrt) / x

-- Define the maximum possible value we seek
def max_value := 2 / (2 * real.sqrt 2 + 1)

-- Main theorem statement
theorem max_value_of_f (hx : 0 < x) : 
  ∃ x : ℝ, f x = max_value :=
sorry

end max_value_of_f_l235_235227


namespace rectangle_tiling_even_stars_possible_l235_235535

theorem rectangle_tiling_even_stars_possible (n : ℕ) (m : ℕ) (k : ℕ) (r : ℕ) (c : ℕ) :
  n = 1 → m = 2 →
  k = 500 →
  r = 5 → c = 200 →
  (exists (tiles : fin k → fin r × fin c × fin 2),
    ∀ i : fin r, (∑ j : fin c, (∑ t in tiles, (t.1 = (i, j))) * 2) % 2 = 0) ∧
    ∀ j : fin c, (∑ i : fin r, (∑ t in tiles, (t.1 = (i, j))) * 2) % 2 = 0 :=
begin
  intros,
  sorry
end

end rectangle_tiling_even_stars_possible_l235_235535


namespace fraction_of_yard_occupied_l235_235707

theorem fraction_of_yard_occupied (yard_length yard_width triangle_leg length_of_short_parallel length_of_long_parallel : ℕ) 
  (h1 : yard_length = 30) 
  (h2 : yard_width = 18) 
  (h3 : triangle_leg = 6) 
  (h4 : length_of_short_parallel = yard_width) 
  (h5 : length_of_long_parallel = yard_length) 
  : (2 * (1/2 * (triangle_leg * triangle_leg))) / (yard_length * yard_width) = 1 / 15 := by 
sorrry

end fraction_of_yard_occupied_l235_235707


namespace sum_S_2013_l235_235622

-- Define the general term of the sequence
def a_n (n : ℕ) : ℤ := n * int.cos (n * real.pi / 2)

-- Define the sum of the first n terms
def S_n (n : ℕ) : ℤ := (finset.range (n + 1)).sum a_n

-- State the theorem
theorem sum_S_2013 : S_n 2013 = 1006 := 
sorry

end sum_S_2013_l235_235622


namespace arithmetic_geometric_sum_problem_l235_235524

theorem arithmetic_geometric_sum_problem (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ) :
  (a 2 = 2) →
  (S 5 = 15) →
  (b 2 + b 4 = 60) →
  (∀ n, S n = n.a.sum) →
  (∀ n, a n = n) →
  (∀ n, b n = 3 * 2^n) →
  (∀ n, c n = 2 * (a n) / (b n)) →
  (∀ n, T n = 2/3 * (2 - (n + 2)/2^n)) :=
begin
  sorry
end

end arithmetic_geometric_sum_problem_l235_235524


namespace percent_of_students_with_B_is_15_l235_235180

def is_B_grade (score : ℕ) : Prop :=
  80 ≤ score ∧ score ≤ 84

def scores : List ℕ :=
  [92, 81, 68, 88, 82, 63, 79, 70, 85, 99, 59, 67, 84, 90, 75, 61, 87, 65, 86]

def B_grade_scores : List ℕ :=
  scores.filter is_B_grade

def B_grade_count : ℕ := B_grade_scores.length

def total_students : ℕ := scores.length

def B_grade_percentage : ℝ :=
  (B_grade_count.toReal / total_students.toReal) * 100

theorem percent_of_students_with_B_is_15 :
  B_grade_percentage = 15 :=
by
  sorry

end percent_of_students_with_B_is_15_l235_235180


namespace binom_60_3_l235_235017

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235017


namespace diag_AC_gt_diag_BD_l235_235597

namespace QuadrilateralProof

-- Define the quadrilateral type with vertices and internal angles
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h_sum : angle_A + angle_B + angle_C + angle_D = 2 * π)

-- Define the conditions for the problem
variables {q : Quadrilateral}
(h_A_acute : 0 < q.angle_A ∧ q.angle_A < π / 2) 
(h_B_obtuse : π / 2 < q.angle_B ∧ q.angle_B < π)
(h_C_obtuse : π / 2 < q.angle_C ∧ q.angle_C < π)
(h_D_obtuse : π / 2 < q.angle_D ∧ q.angle_D < π)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the problem to prove AC > BD
theorem diag_AC_gt_diag_BD : 
  distance q.A q.C > distance q.B q.D :=
sorry

end QuadrilateralProof

end diag_AC_gt_diag_BD_l235_235597


namespace speed_of_man_rowing_upstream_l235_235355

-- Define conditions
def V_m : ℝ := 20 -- speed of the man in still water (kmph)
def V_downstream : ℝ := 25 -- speed of the man rowing downstream (kmph)
def V_s : ℝ := V_downstream - V_m -- calculate the speed of the stream

-- Define the theorem to prove the speed of the man rowing upstream
theorem speed_of_man_rowing_upstream 
  (V_m : ℝ) (V_downstream : ℝ) (V_s : ℝ := V_downstream - V_m) : 
  V_upstream = V_m - V_s :=
by
  sorry

end speed_of_man_rowing_upstream_l235_235355


namespace composition_value_l235_235554

noncomputable def P (x : ℝ) : ℝ := 3 * real.sqrt x
noncomputable def Q (x : ℝ) : ℝ := x ^ 3

theorem composition_value :
  P (Q (P (Q (P (Q 2))))) = 846 * real.sqrt 2 :=
by sorry

end composition_value_l235_235554


namespace set_equiv1_set_equiv2_set_equiv3_l235_235081

-- Each proof statement is a theorem that states the equivalence of sets based on the given conditions.

theorem set_equiv1 : {x | -1 ≤ x ∧ x < 5 ∧ even x} = {0, 2, 4} :=
sorry

theorem set_equiv2 : {0, 1, 2, 3, 4, 5} = {x | 0 ≤ x ∧ x ≤ 5 ∧ x ∈ ℤ} :=
sorry

theorem set_equiv3 : {x | |x| = 1} = {-1, 1} :=
sorry

end set_equiv1_set_equiv2_set_equiv3_l235_235081


namespace no_common_points_iff_parallel_or_skew_l235_235659

-- Definitions (conditions):
def line (α : Type*) := set α
def plane (α : Type*) := set α

-- Proposed conditions
variable {α : Type*} [nonempty α]

-- Proposition D
theorem no_common_points_iff_parallel_or_skew
  (l1 l2 : line α) :
  (∀ p : α, p ∉ l1 ∨ p ∉ l2) ↔ (parallel l1 l2 ∨ skew l1 l2) :=
  sorry

end no_common_points_iff_parallel_or_skew_l235_235659


namespace gopi_gives_turbans_l235_235157

noncomputable def calculate_turbans (annual_cash : ℝ) (months_worked : ℝ) (cash_received : ℝ) (turban_price : ℝ) : ℝ :=
let proportion_worked := months_worked / 12 in
let total_proportion_salary := proportion_worked * annual_cash in
let cash_shortfall := total_proportion_salary - cash_received in
cash_shortfall / turban_price

theorem gopi_gives_turbans (annual_cash : ℝ) (months_worked : ℝ) (cash_received : ℝ) (turban_given : ℝ) (turban_price : ℝ) :
  (annual_cash = 90) → (months_worked = 9) → (cash_received = 45) → (turban_given = 1) → (turban_price = 90) →
  calculate_turbans annual_cash months_worked cash_received turban_price = turban_given :=
by
  intros h1 h2 h3 h4 h5
  subst h1
  subst h2
  subst h3
  subst h4
  subst h5
  simp [calculate_turbans]
  sorry

end gopi_gives_turbans_l235_235157


namespace min_value_of_sum_l235_235560

theorem min_value_of_sum (a b : ℤ) (h : a * b = 150) : a + b = -151 :=
  sorry

end min_value_of_sum_l235_235560


namespace matrix_count_l235_235567

-- Define the matrix type
def matrix (n : ℕ) : Type := Array (Array ℤ)

-- conditions
def valid_element (a : ℤ) : Prop := a = -1 ∨ a = 0 ∨ a = 1

def valid_matrix (A : matrix n) : Prop :=
  ∀ i j, valid_element (A[i][j])

def constant_sum_property (A : matrix n) : Prop :=
  ∀ (perm1 perm2 : Fin n → Fin n), 
    (∑ i, A[i][perm1 i] = ∑ i, A[i][perm2 i])

def r (n : ℕ) : ℤ := 4^n + 2 * 3^n - 4 * 2^n + 1

-- Lean theorem statement to prove r(n) given conditions 
theorem matrix_count (n : ℕ) (A : matrix n) :
  valid_matrix A →
  constant_sum_property A →
  r n = 4^n + 2 * 3^n - 4 * 2^n + 1 := 
by
  intros,
  sorry

end matrix_count_l235_235567


namespace sunscreen_fraction_limit_l235_235708

theorem sunscreen_fraction_limit (S : ℝ) (hS : S > 0) :
  let sequence := "CACBCACB...".cycle in
  let transfer_fraction := 1 / 10 in
  let ongoing_fraction := 9 / 10 in
  (S * transfer_fraction) * ∑ n in (range ∞), (ongoing_fraction ^ (2 * n)) = S * (10 / 19) :=
by
  sorry

end sunscreen_fraction_limit_l235_235708


namespace probability_approx_defective_l235_235665

noncomputable def probability_two_defective (total : ℕ) (defective : ℕ) : ℝ :=
  let pA := (defective : ℝ) / (total : ℝ)
  let pB_given_A := (defective - 1 : ℝ) / (total - 1 : ℝ)
  pA * pB_given_A

theorem probability_approx_defective :
  probability_two_defective 230 84 ≈ 0.1324 :=
sorry

end probability_approx_defective_l235_235665


namespace gross_salary_after_increase_and_tax_l235_235251

noncomputable def current_salary : ℝ := 30000
noncomputable def increase_rate : ℝ := 0.10
noncomputable def tax_rate : ℝ := 0.13

theorem gross_salary_after_increase_and_tax :
  let new_salary := current_salary * (1 + increase_rate)
  let gross_salary := new_salary / (1 - tax_rate)
  gross_salary ≈ 37931 := by
  sorry

end gross_salary_after_increase_and_tax_l235_235251


namespace nonagon_diagonals_l235_235847

/-- 
  The number of distinct diagonals in a convex nonagon (9-sided polygon) 
  can be calculated using the formula for a polygon with n sides: (n * (n - 3)) / 2.
-/
theorem nonagon_diagonals : 
  let n := 9 in (n * (n - 3)) / 2 = 27 := 
by
  -- Using the formula (n * (n - 3)) / 2
  let n := 9
  show (n * (n - 3)) / 2 = 27
  sorry

end nonagon_diagonals_l235_235847


namespace num_diagonals_convex_nonagon_l235_235877

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235877


namespace dinner_duration_l235_235209

def time_for_homework := 30
def time_for_cleaning_room := 30
def time_for_taking_trash := 5
def time_for_emptying_dishwasher := 10
def available_time := 120
def total_chore_time := time_for_homework + time_for_cleaning_room + time_for_taking_trash + time_for_emptying_dishwasher
def dinner_time := available_time - total_chore_time

theorem dinner_duration : dinner_time = 45 :=
by {
  simp [dinner_time, available_time, total_chore_time, time_for_homework, time_for_cleaning_room, time_for_taking_trash, time_for_emptying_dishwasher],
  sorry
}

end dinner_duration_l235_235209


namespace find_a_l235_235512

theorem find_a
  (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop)
  (intersects : (ℝ × ℝ) → Prop)
  (a : ℝ) (positivity : a > 0)
  (polar_eq : ∀ ρ θ, C (ρ * cos θ) (ρ * sin θ) ↔ ρ * (sin θ)^2 = 2 * a * cos θ)
  (line_eq : ∀ t, l (-2 + (sqrt 2 / 2) * t) (-4 + (sqrt 2 / 2) * t))
  (intersect_points : ∃ A B, intersects A ∧ intersects B ∧ |A.1 - B.1|^2 + |A.2 - B.2|^2 = (2 * sqrt 10)^2) :
  a = 1 :=
by
  sorry

end find_a_l235_235512


namespace num_diagonals_convex_nonagon_l235_235871

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235871


namespace little_john_spent_on_sweets_l235_235578

theorem little_john_spent_on_sweets
  (initial_amount : ℝ)
  (amount_per_friend : ℝ)
  (friends_count : ℕ)
  (amount_left : ℝ)
  (spent_on_sweets : ℝ) :
  initial_amount = 10.50 →
  amount_per_friend = 2.20 →
  friends_count = 2 →
  amount_left = 3.85 →
  spent_on_sweets = initial_amount - (amount_per_friend * friends_count) - amount_left →
  spent_on_sweets = 2.25 :=
by
  intros h_initial h_per_friend h_friends_count h_left h_spent
  sorry

end little_john_spent_on_sweets_l235_235578


namespace length_of_XY_l235_235981

theorem length_of_XY (a b : ℝ) (M N : ℝ → ℝ) 
  (OX OY XN YM : ℝ) :
  ∃ OX OY a b M N, angle OX OY = 90 ∧ 
  XN = 22 ∧ YM = 31 ∧
  midpoint M OX a = (OX + a) / 2 ∧
  midpoint N OY b = (OY + b) / 2 ∧
  length XY = 34 : ℝ :=
begin
  sorry
end

end length_of_XY_l235_235981


namespace exists_same_num_plus_signs_l235_235188

theorem exists_same_num_plus_signs (grid : Fin 8 × Fin 8 → Prop) :
  ∃ (sq1 sq2 : Fin 5 × Fin 5)
    (cont1 cont2 : Fin 4 × Fin 4 → Prop),
    (sq1 ≠ sq2) ∧
    (∑ x in Finset.univ.product Finset.univ, if grid (sq1.1 + x.1, sq1.2 + x.2) then 1 else 0) = 
    (∑ x in Finset.univ.product Finset.univ, if grid (sq2.1 + x.1, sq2.2 + x.2) then 1 else 0) :=
sorry

end exists_same_num_plus_signs_l235_235188


namespace smallest_N_l235_235096

noncomputable def i : ℂ := Complex.I

structure fieldData (K : Type*) [Field K] :=
( sqrt_N : K )
( sqrt_1pI : K )

def myField := { K : Type* [Field K] // K = ℚ(Complex.{0}-Finite.(sqrt 2), Complex.sqrt (1 + Complex.I)) }

theorem smallest_N (N : ℕ) (K : myField) (h1 : K = ℚ(sqrt N, sqrt (1 + i))) (h2 : i = Complex.I) :
  (N = 2) ∧ (Gal(K/ℚ) = (Zmod 2) × (Zmod 2) × (Zmod 2)) := sorry

end smallest_N_l235_235096


namespace count_qualifying_integers_between_200_and_250_l235_235487

theorem count_qualifying_integers_between_200_and_250 :
  let qualifying_integers := {n : ℕ | 200 ≤ n ∧ n < 250 ∧
                                        (∀ i j k : ℕ, 
                                          (n = i * 100 + j * 10 + k) ∧
                                          i < j ∧ j < k ∧
                                          k % 2 = 0) ∧
                                        n.digits ≠ List.dispatch_on_fun n.digits id} in
  qualifying_integers.card = 5 :=
begin
  sorry
end

end count_qualifying_integers_between_200_and_250_l235_235487


namespace cos_alpha_l235_235465

theorem cos_alpha (x y : ℝ) (h1 : x = 2 * Real.sqrt 5 / 5) (h2 : y = - Real.sqrt 5 / 5) (r : ℝ)
  (h3 : r = Real.sqrt (x^2 + y^2)) : (r = 1) -> (Real.cos α = x / r) -> (Real.cos α = 2 * Real.sqrt 5 / 5) :=
by
  intro hr
  intro hcos
  rw [hr, hcos]
  trivial

end cos_alpha_l235_235465


namespace casey_nail_decorating_time_l235_235385

theorem casey_nail_decorating_time 
  (n_toenails n_fingernails : ℕ)
  (t_apply t_dry : ℕ)
  (coats : ℕ)
  (h1 : n_toenails = 10)
  (h2 : n_fingernails = 10)
  (h3 : t_apply = 20)
  (h4 : t_dry = 20)
  (h5 : coats = 3) :
  20 * (t_apply + t_dry) * coats = 120 :=
by
  -- skipping the proof
  sorry

end casey_nail_decorating_time_l235_235385


namespace exists_quadrilateral_with_equal_tangents_l235_235972

theorem exists_quadrilateral_with_equal_tangents :
  ∃ α β γ δ : ℝ, α + β + γ + δ = 2 * π ∧ 
  tan α = tan β ∧ tan β = tan γ ∧ tan γ = tan δ :=
by
  sorry

end exists_quadrilateral_with_equal_tangents_l235_235972


namespace daleyza_contracted_units_l235_235748

variable (units_building1 : ℕ)
variable (units_building2 : ℕ)
variable (units_building3 : ℕ)

def total_units (units_building1 units_building2 units_building3 : ℕ) : ℕ :=
  units_building1 + units_building2 + units_building3

theorem daleyza_contracted_units :
  units_building1 = 4000 →
  units_building2 = 2 * units_building1 / 5 →
  units_building3 = 120 * units_building2 / 100 →
  total_units units_building1 units_building2 units_building3 = 7520 :=
by
  intros h1 h2 h3
  unfold total_units
  rw [h1, h2, h3]
  sorry

end daleyza_contracted_units_l235_235748


namespace rational_root_even_denominator_l235_235184

theorem rational_root_even_denominator
  (a b c : ℤ)
  (sum_ab_even : (a + b) % 2 = 0)
  (c_odd : c % 2 = 1) :
  ∀ (p q : ℤ), (q ≠ 0) → (IsRationalRoot : a * (p * p) + b * p * q + c * (q * q) = 0) →
    gcd p q = 1 → q % 2 = 0 :=
by
  sorry

end rational_root_even_denominator_l235_235184


namespace teal_is_kinda_green_l235_235346

theorem teal_is_kinda_green
    (total_people : ℕ)
    (kinda_blue : ℕ)
    (both_blue_green : ℕ)
    (neither_blue_green : ℕ)
    (other_people : ℕ)
    (total_people = 150) 
    (kinda_blue = 90) 
    (both_blue_green = 45) 
    (neither_blue_green = 25)
    (other_people = total_people - (kinda_blue + neither_blue_green - both_blue_green)) :
    (kinda_blue + other_people - both_blue_green = 80) :=
by
  -- insert steps here to demonstrate the Lean proof
  sorry

end teal_is_kinda_green_l235_235346


namespace passengers_fit_l235_235690

theorem passengers_fit (passengers_in_5_buses : ℕ) (buses2 buses1 : ℕ) 
    (h1 : passengers_in_5_buses = 110) 
    (h2: buses2 = 5) 
    (h3: buses1 = 9) : 
    (passengers_in_5_buses / buses2) * buses1 = 198 := 
by 
    rw [h1, h2, h3]
    sorry

end passengers_fit_l235_235690


namespace radius_of_circle_l235_235790

theorem radius_of_circle (P O : Point) (r : ℝ) (h1 : dist P O = 5) (h2 : dist P O < r) : r = 6 :=
sorry

end radius_of_circle_l235_235790


namespace sphere_touches_AC_implies_touches_BD_l235_235709

noncomputable theory

-- Definitions for the tetrahedron and sphere touching the edges
variables {A B C D O : Type} -- A, B, C, and D are points of the tetrahedron, O is the center of the sphere
variable (K L M N : Type) -- K, L, M, N are the points of tangency forming a square
variables (AC BD : Type) -- Edges AC and BD

-- The statement to prove
theorem sphere_touches_AC_implies_touches_BD
    (h1 : True) -- The sphere touches the edges AB, BC, CD, DA at points K, L, M, N forming a square.
    (h2 : True) -- The sphere also touches the edge AC.
    : True := -- The sphere also touches the edge BD.
sorry

end sphere_touches_AC_implies_touches_BD_l235_235709


namespace palindromes_between_200_and_800_l235_235063

/-
  Define a palindrome as a number where the hundreds digit equals the units digit.
-/
def is_palindrome (n : ℕ) : Prop :=
  let h := n / 100
  let t := (n % 100) / 10
  let u := n % 10
  h = u

/-
  The actual problem: Prove that there are exactly 60 integer palindromes between 200 and 800.
-/
theorem palindromes_between_200_and_800 : 
  ∃ (count : ℕ), count = 60 ∧ (count = (set.univ.filter (λ n, 200 ≤ n ∧ n < 800 ∧ is_palindrome n)).card) :=
begin
  sorry
end

end palindromes_between_200_and_800_l235_235063


namespace magnitude_of_2a_minus_b_l235_235811

section VectorProof

open Real

variables (a b : ℝ × ℝ)
variable y : ℝ

-- Conditions
def a := (1, 2)
def b := (-2, y)

-- Condition: a is parallel to b, which implies y = -4
def parallel : Prop := (-2 / 1 = y / 2)

-- Given 'parallel' condition, we compute the magnitude of (2a - b) and prove it equals 4√5
theorem magnitude_of_2a_minus_b : parallel y → sqrt ((2 * 1 + 2)^2 + (2 * 2 + 4)^2) = 4 * sqrt 5 := by
  sorry

end VectorProof

end magnitude_of_2a_minus_b_l235_235811


namespace polynomial_value_bound_l235_235442

open BigOperators

theorem polynomial_value_bound (n : ℕ) (x : Fin (n + 1) → ℤ) (h_incr : ∀ i j : Fin (n + 1), i < j → x i < x j) 
  (a : Fin (n + 1) → ℤ) :
  ∃ k, |(x k : ℤ) ^ n + ∑ i in Finset.range n, a i * (x k : ℤ)^(n - i - 1) + a n| ≥ nat.factorial n / 2^n := 
sorry

end polynomial_value_bound_l235_235442


namespace problem_l235_235447

noncomputable def p : Prop :=
  ∃ x : ℝ, x < 0 ∧ 2^x < 3^x

noncomputable def q : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < real.pi / 2 → sin x < x

theorem problem (hp : ¬ p) (hq : q) : (¬ p) ∧ q := by
  exact ⟨hp, hq⟩

end problem_l235_235447


namespace sum_of_first_12_terms_geometric_sequence_l235_235462

variable {α : Type*} [Field α]

def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (Finset.range n).sum a

theorem sum_of_first_12_terms_geometric_sequence
  (a : ℕ → α)
  (h_geo : geometric_sequence a)
  (h_sum1 : sum_first_n_terms a 3 = 4)
  (h_sum2 : sum_first_n_terms a 6 - sum_first_n_terms a 3 = 8) :
  sum_first_n_terms a 12 = 60 := 
sorry

end sum_of_first_12_terms_geometric_sequence_l235_235462


namespace area_of_triangle_BCF_l235_235133

noncomputable def area (triangle : Type) : ℝ := sorry

variables {A B C D E F : Type}

def is_isosceles_right_triangle (triangle : Type) : Prop := sorry
def is_right_angle (A : Type) (B : Type) (C : Type) : Prop := sorry
def is_midpoint (F : Type) (D : Type) (E : Type) : Prop := sorry 

axiom ABD_is_isosceles_right_triangle : is_isosceles_right_triangle (Type)
axiom ACE_is_isosceles_right_triangle : is_isosceles_right_triangle (Type)
axiom CAE_is_right_angle : is_right_angle A C E
axiom F_is_midpoint : is_midpoint F D E

theorem area_of_triangle_BCF :
  area (B F C) = 1/2 * (area A B D + area A C E + 3 * area A D E) :=
by sorry

end area_of_triangle_BCF_l235_235133


namespace polynomial_sum_of_coefficients_l235_235574

def sequence (u : ℕ → ℕ) : Prop :=
  u 1 = 7 ∧ ∀ n, u (n + 1) - u n = 5 + 3 * (n - 1)

theorem polynomial_sum_of_coefficients :
  ∃ (a b c : ℚ), (∀ n, u n = a * n^2 + b * n + c) ∧ a + b + c = 7
where
  u : ℕ → ℚ := sorry

end polynomial_sum_of_coefficients_l235_235574


namespace nonagon_diagonals_count_l235_235834

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235834


namespace Derek_is_42_l235_235732

def Aunt_Anne_age : ℕ := 36

def Brianna_age : ℕ := (2 * Aunt_Anne_age) / 3

def Caitlin_age : ℕ := Brianna_age - 3

def Derek_age : ℕ := 2 * Caitlin_age

theorem Derek_is_42 : Derek_age = 42 := by
  sorry

end Derek_is_42_l235_235732


namespace binomial_60_3_l235_235029

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235029


namespace part1_part1_period_part1_symmetry_points_part2_interval_of_monotonic_increase_l235_235483

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos (x + Real.pi / 8), Real.sin (x + Real.pi / 8) ^ 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sin (x + Real.pi / 8), 1)
noncomputable def f (x : ℝ) : ℝ := 2 * (vector_a x).1 * (vector_b x).1 + 2 * (vector_a x).2 * (vector_b x).2 - 1

theorem part1 (x : ℝ) : f(x) = Real.sqrt 2 * Real.sin (2 * x) :=
  sorry

theorem part1_period : ∃ T, T = Real.pi ∧ ∀ x, f(x + T) = f(x) :=
  sorry

theorem part1_symmetry_points : ∀ k : ℤ, f(Real.ofInt k * Real.pi / 2) = 0 :=
  sorry

noncomputable def g (x : ℝ) : ℝ := f(-1 / 2 * x)

theorem part2_interval_of_monotonic_increase :
  ∀ k : ℤ, ∃ a b, a = 2 * Real.ofInt k * Real.pi + Real.pi / 2 ∧
            b = 2 * Real.ofInt k * Real.pi + 3 * Real.pi / 2 ∧
            ∀ x ∈ Set.Icc a b, 
            ∀ u v, u ∈ Set.Icc a b → v ∈ Set.Icc a b → u < v → g(u) < g(v) :=
  sorry

end part1_part1_period_part1_symmetry_points_part2_interval_of_monotonic_increase_l235_235483


namespace each_wolf_needs_to_kill_one_deer_l235_235301

-- Conditions
def wolves_out_hunting : ℕ := 4
def additional_wolves : ℕ := 16
def wolves_total : ℕ := wolves_out_hunting + additional_wolves
def meat_per_wolf_per_day : ℕ := 8
def days_no_hunt : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat needed for all wolves over five days.
def total_meat_needed : ℕ := wolves_total * meat_per_wolf_per_day * days_no_hunt
-- Calculate total number of deer needed to meet the meat requirement.
def deer_needed : ℕ := total_meat_needed / meat_per_deer
-- Calculate number of deer each hunting wolf needs to kill.
def deer_per_wolf : ℕ := deer_needed / wolves_out_hunting

-- The proof statement
theorem each_wolf_needs_to_kill_one_deer : deer_per_wolf = 1 := 
by { sorry }

end each_wolf_needs_to_kill_one_deer_l235_235301


namespace triangle_median_bisector_ratio_l235_235772

theorem triangle_median_bisector_ratio (A B C : Point) (M : Point) (D : Point) (median_AD : Line) :
  is_triangle A B C →
  is_midpoint B C D →
  is_midpoint A D M →
  is_line_through C M (median_AD) →
  ∃ X : Point, on_line X A B ∧ divides_segment X A B 2 1 :=
sorry

end triangle_median_bisector_ratio_l235_235772


namespace prove_inequality_l235_235775

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (55 * Real.pi / 180)

theorem prove_inequality : c > b ∧ b > a :=
by
  -- Proof goes here
  sorry

end prove_inequality_l235_235775


namespace exists_even_function_shifting_graph_l235_235793

variables (φ : ℝ) (f : ℝ → ℝ)
noncomputable def f := λ x, Real.sin (x / 2 + φ)

-- Statement: There exists a constant φ such that f(x) is an even function
theorem exists_even_function : ∃ φ : ℝ, ∀ x : ℝ, f φ x = f φ (-x) := sorry

-- Statement: If φ < 0, the graph of f(x) can be obtained by shifting y = sin(x / 2) to the right by |2φ| units
theorem shifting_graph (h : φ < 0) : (λ x, f φ x) = (λ x, Real.sin ((x + 2 * φ) / 2)) := sorry

end exists_even_function_shifting_graph_l235_235793


namespace trigonometric_identity_l235_235156

theorem trigonometric_identity (α : ℝ) (h : Real.cos α + Real.sin α = 2 / 3) :
  (Real.sqrt 2 * Real.sin (2 * α - Real.pi / 4) + 1) / (1 + Real.tan α) = - 5 / 9 :=
sorry

end trigonometric_identity_l235_235156


namespace probability_BC_same_activity_is_one_third_l235_235771

-- Definitions of students and activities
inductive Student : Type
| A | B | C | D
  deriving DecidableEq

inductive Activity : Type
| Act1 | Act2
  deriving DecidableEq

-- Given that there are four students and we divide them into two groups (activities)
def activities : List (Student × Activity) := 
  [(Student.A, Activity.Act1), (Student.B, Activity.Act2), (Student.C, Activity.Act1), (Student.D, Activity.Act2)]

-- Define the condition of students being in the same activity
def in_same_activity (s1 s2 : Student) : Prop :=
  ∃ a, (s1, a) ∈ activities ∧ (s2, a) ∈ activities

-- Define the probability of students B and C being in the same activity
noncomputable def probability_BC_same_activity : ℚ :=
  let total_events := (finset.powerset_len 2 (finset.of_list activities)).card
  let favorable_events := (finset.filter (λ s, in_same_activity Student.B Student.C) (finset.powerset_len 2 (finset.of_list activities))).card
  favorable_events / total_events

-- The theorem to be proven
theorem probability_BC_same_activity_is_one_third : probability_BC_same_activity = 1 / 3 :=
  sorry

end probability_BC_same_activity_is_one_third_l235_235771


namespace red_ball_count_l235_235955

theorem red_ball_count :
  (∃ N : ℕ, N = 20) →
  (∃ p : ℝ, p = 0.4) →
  (∃ r : ℝ, r = 0.4 * 20) →
  ∃ n : ℕ, n = 8 :=
by
  intros,
  use 8,
  sorry

end red_ball_count_l235_235955


namespace triangle_area_hypotenuse_l235_235195

-- Definitions of the conditions
def DE : ℝ := 40
def DF : ℝ := 30
def angleD : ℝ := 90

-- Proof statement
theorem triangle_area_hypotenuse :
  let Area : ℝ := 1 / 2 * DE * DF
  let EF : ℝ := Real.sqrt (DE^2 + DF^2)
  Area = 600 ∧ EF = 50 := by
  sorry

end triangle_area_hypotenuse_l235_235195


namespace max_m_value_l235_235090

-- The condition given by the inequality
def inequality_holds (m x : ℝ) : Prop :=
  m * sqrt(m) * (x^2 - 6*x + 9) + sqrt(m) / (x^2 - 6*x + 9) ≤ real.sqrt((m^3)^(1 / 4)) * abs (real.cos (real.pi * x / 5))

-- The statement to prove
theorem max_m_value : ∃ (m : ℝ), (∀ x : ℝ, inequality_holds m x) ∧ m = 1 / 16 := sorry

end max_m_value_l235_235090


namespace distinct_diagonals_convex_nonagon_l235_235859

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235859


namespace smallest_6_digit_multiple_1379_l235_235095

theorem smallest_6_digit_multiple_1379 : 
  ∃ (x : ℕ), 100000 ≤ x ∧ x < 1000000 ∧ x % 1379 = 0 ∧ 
    ∀ (y : ℕ), 100000 ≤ y ∧ y < 1000000 ∧ y % 1379 = 0 → y ≥ x := 
by
  use 100657
  split ; try {linarith}
  split ; try {exact rfl}
  intros y hy
  linarith

end smallest_6_digit_multiple_1379_l235_235095


namespace number_of_diagonals_in_nonagon_l235_235828

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235828


namespace cookies_difference_l235_235595

-- Define the initial conditions
def initial_cookies : ℝ := 57
def cookies_eaten : ℝ := 8.5
def cookies_bought : ℝ := 125.75

-- Problem statement
theorem cookies_difference (initial_cookies cookies_eaten cookies_bought : ℝ) : 
  cookies_bought - cookies_eaten = 117.25 := 
sorry

end cookies_difference_l235_235595


namespace cylinder_cone_sphere_volume_relation_l235_235362

theorem cylinder_cone_sphere_volume_relation (r : ℝ) (M C : ℝ) :
  let h := 4 * r,
      V_cone := (1 / 3) * π * r^2 * h,
      V_cylinder := π * r^2 * h in
  (M = 4 * π * r^3) →
  (C = V_cylinder - V_cone) →
  C = (8 / 3) * π * r^3 :=
by
  sorry

end cylinder_cone_sphere_volume_relation_l235_235362


namespace angle_EFD_of_incircumcircle_l235_235390

open EuclideanGeometry

theorem angle_EFD_of_incircumcircle
    {A B C D E F : Point}
    (Γ : Circle)
    (h₁ : Γ.isIncircleTriangle A B C)
    (h₂ : Γ.isCircumcircleTriangle D E F)
    (hD : D ∈ Line B C)
    (hE : E ∈ Line A B)
    (hF : F ∈ Line A C)
    (angle_A : ∠ A = 50)
    (angle_B : ∠ B = 70)
    (angle_C : ∠ C = 60) :
    ∠ E F D = 70 :=
sorry

end angle_EFD_of_incircumcircle_l235_235390


namespace min_value_of_expression_l235_235935

theorem min_value_of_expression 
  (a b : ℝ) 
  (h : a > 0) 
  (h₀ : b > 0) 
  (h₁ : 2*a + b = 2) : 
  ∃ c : ℝ, c = (8*a + b) / (a*b) ∧ c = 9 :=
sorry

end min_value_of_expression_l235_235935


namespace tangent_range_l235_235444

theorem tangent_range (P : ℝ × ℝ) (C : ℝ × ℝ → ℝ) (k : ℝ) :
  P = (1, 2) →
  (∀ (x y : ℝ), C (x, y) = x^2 + y^2 + k * x + 2 * y + k^2) →
  (∃ (t1 t2 : ℝ × ℝ), (t1 ≠ t2) ∧ C t1 = 0 ∧ C t2 = 0) ↔ -2 * sqrt 3 / 3 < k ∧ k < 2 * sqrt 3 / 3 :=
by
  sorry


end tangent_range_l235_235444


namespace max_value_trig_correct_l235_235988

noncomputable def max_value_trig (a b ϕ : ℝ) : ℝ :=
  sqrt (a^2 + b^2)

theorem max_value_trig_correct (a b ϕ : ℝ) :
  ∀ θ : ℝ, a * cos (θ + ϕ) + b * sin (θ + ϕ) ≤ max_value_trig a b ϕ := sorry

end max_value_trig_correct_l235_235988


namespace find_number_l235_235082

theorem find_number (x : ℝ) (h : 0.6667 * x + 1 = 0.75 * x) : x = 12 := 
by
  sorry

end find_number_l235_235082


namespace length_of_train_l235_235338

-- Definition of conditions
def speed_km_hr : ℝ := 180
def time_sec : ℝ := 9
def speed_m_s : ℝ := (speed_km_hr * 1000) / 3600 -- Conversion from km/hr to m/s

-- Statement of the problem to be proved
theorem length_of_train : speed_m_s * time_sec = 450 := by
  -- The proof would go here
  sorry

end length_of_train_l235_235338


namespace range_of_approximate_3_4_l235_235630

theorem range_of_approximate_3_4 (a : ℝ) (h : round (a * 10) / 10 = 3.4) : 3.35 ≤ a ∧ a < 3.45 :=
sorry

end range_of_approximate_3_4_l235_235630


namespace base_area_cone_is_108_l235_235296

-- Defining the conditions
variables (V: ℝ) (h: ℝ) 

-- Assuming cylinder and cone have equal volume and height
-- Cylinder volume V_cylinder = base_area_cylinder * height
-- Cone volume V_cone = (1/3) * base_area_cone * height
-- Given: base area of cylinder is 36
-- Calculate: base area of cone to be 108

def base_area_cylinder : ℝ := 36
def height : ℝ := h
def volume_cylinder : ℝ := base_area_cylinder * height
def volume_cone : ℝ := (1/3) * base_area_cone * height

theorem base_area_cone_is_108 (base_area_cone: ℝ) (h: ℝ) 
    (vol_eq: volume_cylinder = volume_cone) 
    (h_eq: height = h) 
    (base_cylinder: base_area_cylinder = 36) 
    (h_cylinder: height = h):
    base_area_cone = 108 := 
by 
  sorry

end base_area_cone_is_108_l235_235296


namespace train_length_is_correct_l235_235721

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end train_length_is_correct_l235_235721


namespace cost_of_plastering_l235_235337

theorem cost_of_plastering
  (length width height : ℝ)
  (cost_per_sqm: ℝ)
  (h_length: length = 25)
  (h_width: width = 12)
  (h_height: height = 6)
  (h_cost_per_sqm: cost_per_sqm = 0.75) :
  let area_walls := 2 * (length * height) + 2 * (width * height) in
  let area_bottom := length * width in
  let total_area := area_walls + area_bottom in
  let total_cost := total_area * cost_per_sqm in
  total_cost = 558 :=
by
  sorry

end cost_of_plastering_l235_235337


namespace exponent_equality_l235_235502

theorem exponent_equality (x : ℕ) (hx : (1 / 8 : ℝ) * (2 : ℝ) ^ 40 = (2 : ℝ) ^ x) : x = 37 :=
sorry

end exponent_equality_l235_235502


namespace find_g_function_l235_235231

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_function (g_def : ∀ x : ℝ, ∀ y : ℝ, g (x + y) = 4^y * g x + 3^x * g y)
                        (g_one : g 1 = 1) :
   g = λ x, 4^x - 3^x :=
by
  -- proof will be completed later
  sorry

end find_g_function_l235_235231


namespace sequence_formulas_l235_235121

theorem sequence_formulas (S : ℕ → ℕ) (a b : ℕ → ℕ) (c : ℕ → ℝ) (λ : ℝ) :
  -- Conditions
  (∀ n, S(n) = 3 * n^2 + 8 * n) →
  (∀ n, a(n) = S(n) - S(n - 1)) →
  ∀ b, is_arithmetic_seq b →
  (∀ n, a(n) = b(n) + b(n + 1)) →

  -- Formulas
  (∀ n, a(n) = 6 * n + 5) ∧
  (∀ n, b(n) = 3 * n + 1) ∧

  -- Range for λ
  (∀ n, c(n) = (a(n) + 1)^(n + 1) / (b(n) + 2)^n) →
  (∀ n, λ > c(n + 1) / c(n)) →
  λ > 3 :=
by
  intros S a b c λ SN aN arith_b a_eq_b_sum c_def λ_ineq
  sorry

end sequence_formulas_l235_235121


namespace rhombus_area_l235_235624

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 24 :=
by {
  sorry
}

end rhombus_area_l235_235624


namespace cos_A_value_l235_235986

theorem cos_A_value (a b c : ℝ) (A B C : ℝ) 
  (cos_C : ℝ) (h_cosC : cos C = 2 / 3) (h_a_eq_3b : a = 3 * b)
  (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c : c = Real.sqrt 6 * b) 
  (h_triangle_ineq : b + c > a ∧ a + c > b ∧ a + b > c) : 
  cos A = -Real.sqrt 6 / 6 :=
by
  sorry

end cos_A_value_l235_235986


namespace sum_of_solutions_eq_11_l235_235753

theorem sum_of_solutions_eq_11 :
  Σ' x : ℝ, (x^2 - 5 * x + 3) ^ (x^2 - 6 * x + 3) = 1 = 11 :=
sorry

end sum_of_solutions_eq_11_l235_235753


namespace proof_problem_l235_235478

variables {m x : ℝ}

def p (m : ℝ) := ∀ x : ℝ, 4 * m * x^2 + x + m ≤ 0
def q (m : ℝ) := ∃ x : ℝ, (2 ≤ x) ∧ (x ≤ 8) ∧ (m * log 2 x ≥ 1)

theorem proof_problem (h1 : ¬ p m ∧ ¬ q m = false) (h2 : ¬ q m = true) : m ≤ -1 / 4 := 
sorry

end proof_problem_l235_235478


namespace train_length_is_correct_l235_235722

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end train_length_is_correct_l235_235722


namespace even_function_a_equals_1_l235_235939

/-- 
  If the function f(x) = x * ln (x + sqrt(a + x^2)) is an even function, then a = 1. 
--/
theorem even_function_a_equals_1 (a : ℝ) (f : ℝ → ℝ):
  (∀ x : ℝ, f(x) = x * log (x + sqrt (a + x^2))) →
  (∀ x : ℝ, f(-x) = f(x)) →
  a = 1 :=
by
  intros h1 h2
  sorry

end even_function_a_equals_1_l235_235939


namespace no_three_digit_guminumber_representing_multiple_values_l235_235944

theorem no_three_digit_guminumber_representing_multiple_values:
  ∀ (x : ℕ), 0 < x < 10000 → 
  ¬ ∃ (n m : ℕ), 1 ≤ n ∧ n < 10 ∧ 1 ≤ m ∧ m < 10 ∧ n ≠ m ∧ 
  n^2 ≤ x ∧ x < n^3 ∧ (n - 1)^4 < x ∧ x < n^4 ∧
  m^2 ≤ x ∧ x < m^3 ∧ (m - 1)^4 < x ∧ x < m^4 := 
by 
  intro x h
  rintro ⟨n, m, hn1, hn2, hm1, hm2, hnm, hnrange, hngumirange, hmrange, hmgumirange⟩
  -- insert proof here
  sorry

end no_three_digit_guminumber_representing_multiple_values_l235_235944


namespace angle_D_is_20_degrees_l235_235185

theorem angle_D_is_20_degrees (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 160) : D = 20 :=
by
  sorry

end angle_D_is_20_degrees_l235_235185


namespace correct_number_of_statements_l235_235220

variables (m n : Type) (α β : Type)
variables [NonCoincidentLines m n] [NonCoincidentPlanes α β]

-- Definitions of the statements
def statement1 : Prop := m ∥ n ∧ m ⊥ α → n ⊥ α
def statement2 : Prop := m ∥ n ∧ m ∥ α → n ∥ α
def statement3 : Prop := m ⊥ α ∧ n ⊆ α → m ⊥ n
def statement4 : Prop := m ⊥ α ∧ m ⊆ β → α ⊥ β

def count_correct_statements : ℕ :=
if statement1 m n α ∧ ¬statement2 m n α ∧ statement3 m n α ∧ statement4 m n α then 3 else 0

-- Main theorem to be proven
theorem correct_number_of_statements : count_correct_statements m n α β = 3 := 
by sorry

end correct_number_of_statements_l235_235220


namespace nonagon_diagonals_count_l235_235837

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235837


namespace Namjoon_books_l235_235235

theorem Namjoon_books (a b c : ℕ)
  (h1 : a = 35)
  (h2 : b = a - 16)
  (h3 : c = b + 35) :
  a + b + c = 108 :=
by
  have h4 : b = 35 - 16 := by rw [h1, Nat.sub]
  have h5 : c = (35 - 16) + 35 := by rw [←h4, h3]
  calc (35 : ℕ) + (35 - 16) + ((35 - 16) + 35)
    = 35 + 19 + 54 := by nth_rewrite 0 h4; nth_rewrite 1 h5; sorry

end Namjoon_books_l235_235235


namespace platform_length_l235_235713

theorem platform_length
  (train_length : ℝ := 360) -- The train is 360 meters long
  (train_speed_kmh : ℝ := 45) -- The train runs at a speed of 45 km/hr
  (time_to_pass_platform : ℝ := 60) -- It takes 60 seconds to pass the platform
  (platform_length : ℝ) : platform_length = 390 :=
by
  sorry

end platform_length_l235_235713


namespace find_k_l235_235792

-- Definitions
def equation (x : ℝ) : ℝ := x^3 + x - 4
def k (k_val : ℝ) : Prop := ∃ (k : ℝ), k_val = k ∧ (floor (2 * k) : ℝ) / 2 = k ∧ 
                              equation k < 0 ∧ equation (k + 1 / 2) > 0 

-- Statement
theorem find_k : k 1 :=
by 
 -- Assume the problem conditions and prove that k = 1 satisfies these conditions
 sorry

end find_k_l235_235792


namespace ages_sum_l235_235378

theorem ages_sum (Beckett_age Olaf_age Shannen_age Jack_age : ℕ) 
  (h1 : Beckett_age = 12) 
  (h2 : Olaf_age = Beckett_age + 3) 
  (h3 : Shannen_age = Olaf_age - 2) 
  (h4 : Jack_age = 2 * Shannen_age + 5) : 
  Beckett_age + Olaf_age + Shannen_age + Jack_age = 71 := 
by
  sorry

end ages_sum_l235_235378


namespace root_reciprocals_identity_l235_235394

noncomputable def cubic_roots (a b c : ℝ) : Prop :=
  (a + b + c = 12) ∧ (a * b + b * c + c * a = 20) ∧ (a * b * c = -5)

theorem root_reciprocals_identity (a b c : ℝ) (h : cubic_roots a b c) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 20.8 :=
by
  sorry

end root_reciprocals_identity_l235_235394


namespace cos_alpha_minus_pi_over_6_l235_235452

theorem cos_alpha_minus_pi_over_6 (α : Real) 
  (h1 : Real.pi / 2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.sin (α + Real.pi / 6) = 3 / 5) : 
  Real.cos (α - Real.pi / 6) = (3 * Real.sqrt 3 - 4) / 10 := 
by 
  sorry

end cos_alpha_minus_pi_over_6_l235_235452


namespace find_x_plus_y_l235_235931

variables (x y : ℝ)

def condition1 := abs x - x + y = 8
def condition2 := x + abs y + y = 16

theorem find_x_plus_y (hx : condition1) (hy : condition2) : x + y = 8 := 
  sorry

end find_x_plus_y_l235_235931


namespace num_diagonals_convex_nonagon_l235_235879

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235879


namespace isosceles_triangle_base_length_l235_235785

noncomputable def length_of_base (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : ℝ :=
  (12 - 2 * a) / 2

theorem isosceles_triangle_base_length (a b : ℝ) (h_isosceles : a = b) (h_side : a = 3) (h_perimeter : a + b + x = 12) : length_of_base a b h_isosceles h_side h_perimeter = 4.5 :=
sorry

end isosceles_triangle_base_length_l235_235785


namespace sum_of_two_numbers_with_conditions_l235_235644

/-- Two distinct natural numbers that end with 7 zeros and have exactly 72 
divisors exist. The sum of these two numbers is 70000000. -/
theorem sum_of_two_numbers_with_conditions :
  ∃ (N1 N2 : ℕ),
    N1 ≠ N2 ∧ 
    (∃ k1 k2 : ℕ, N1 = 10^7 * k1 ∧ N2 = 10^7 * k2) ∧ 
    (∀ n ∈ [N1, N2], (number_of_divisors n = 72)) ∧
    N1 + N2 = 70000000 :=
sorry

end sum_of_two_numbers_with_conditions_l235_235644


namespace main_theorem_l235_235150

-- Definitions based on conditions in the problem
def line_system (θ : ℝ) (x y: ℝ) : Prop := (x * Real.cos θ + (y - 1) * Real.sin θ = 1)

def valid_range (θ : ℝ) : Prop := (0 ≤ θ) ∧ (θ ≤ 2 * Real.pi)

-- Correct statements to be proven
def statement2 : Prop := 
  ∃ (r : ℝ), (0 < r) ∧ (r < 1) ∧ (∀ (θ : ℝ), valid_range θ → ¬line_system θ 0 r)

def statement3 : Prop := 
  ∀ (n : ℕ), (n ≥ 3) → ∃ (polygon : Fin n → (ℝ × ℝ)), 
  (polygon_has_edges_on_lines polygon line_system)

-- Main theorem
theorem main_theorem : 
  ∀ (θ : ℝ), valid_range θ →
  (statement2 ∧ statement3) :=
by
  sorry

end main_theorem_l235_235150


namespace residue_zero_l235_235094

noncomputable def residue_of_f_at_0 : ℂ :=
  let f : ℂ → ℂ := λ z, z^3 * complex.sin (1 / z^2)
  complex.residue f 0

theorem residue_zero : residue_of_f_at_0 = 0 := by
  sorry

end residue_zero_l235_235094


namespace students_history_or_statistics_or_both_l235_235516

variable (Total Students H S H_only B S_only : ℕ)

theorem students_history_or_statistics_or_both :
  H = 36 →
  S = 32 →
  H_only = 27 →
  B = H - H_only →
  S_only = S - B →
  Total = H_only + S_only + B →
  Total = 59 :=
by
  intros h_eq s_eq h_only_eq b_eq s_only_eq total_eq
  rw [h_eq, s_eq, h_only_eq] at *
  have b_val : B = 9 := by rw [b_eq]; exact rfl
  have s_only_val : S_only = 23 := by rw [b_eq, s_only_eq]; exact rfl
  rw [b_val, s_only_val] at *
  exact total_eq
  sorry   -- The proof steps are omitted

end students_history_or_statistics_or_both_l235_235516


namespace distinct_diagonals_nonagon_l235_235866

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235866


namespace max_value_t_triangle_l235_235529

theorem max_value_t_triangle 
  (A C : ℝ) 
  (h : sin A + sin C = 3 / 2) 
  (t : ℝ) 
  (ht : t = 2 * sin A * sin C) : 
  ∃ t_max : ℝ, t_max = (27 * real.sqrt 7) / 64 ∧ 
    (∀ (t' : ℝ), t' = t → t * real.sqrt ((9 / 4 - t) * (t - 1 / 4)) ≤ t_max ) := 
sorry

end max_value_t_triangle_l235_235529


namespace distinct_diagonals_in_convex_nonagon_l235_235880

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235880


namespace equilateral_triangle_intersection_impossible_l235_235971

noncomputable def trihedral_angle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ β = 90 ∧ γ = 90 ∧ α > 0

theorem equilateral_triangle_intersection_impossible :
  ¬ ∀ (α : ℝ), ∀ (β γ : ℝ), trihedral_angle α β γ → 
    ∃ (plane : ℝ → ℝ → ℝ), 
      ∀ (x y z : ℝ), plane x y = z → x = y ∧ y = z ∧ z = x ∧ 
                      x + y + z = 60 :=
sorry

end equilateral_triangle_intersection_impossible_l235_235971


namespace power_equality_l235_235498

theorem power_equality (x : ℕ) (h : (1 / 8) * (2^40) = 2^x) : x = 37 := by
  sorry

end power_equality_l235_235498


namespace distinct_diagonals_in_convex_nonagon_l235_235887

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235887


namespace measure_of_angle_Q_l235_235518

-- Given conditions
variables (α β γ δ : ℝ)
axiom h1 : α = 130
axiom h2 : β = 95
axiom h3 : γ = 110
axiom h4 : δ = 104

-- Statement of the problem
theorem measure_of_angle_Q (Q : ℝ) (h5 : Q + α + β + γ + δ = 540) : Q = 101 := 
sorry

end measure_of_angle_Q_l235_235518


namespace distinct_diagonals_convex_nonagon_l235_235856

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235856


namespace distribution_ways_proof_l235_235701

open Nat

def card_number (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 100

def card_sum_identifies_missing_box (distribution : Finset (ℕ × ℕ)) : Prop :=
  ∀ (s₁ s₂ : ℕ),
    s₁ ≠ s₂ →
    (∃ x y z : ℕ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
                  (card_number x ∧ card_number y ∧ card_number z) ∧
                  (distribution.contains (x, y) ∧ distribution.contains (x, z) ∧ distribution.contains (y, z)) ∧
                  (s₁ = x + y ∨ s₁ = x + z ∨ s₁ = y + z) ∧
                  (s₂ = x + y ∨ s₂ = x + z ∨ s₂ = y + z))

theorem distribution_ways_proof :
  ∃! (distribution : Finset (ℕ × ℕ)),
    (∀ c, card_number c → ∃ b, distribution.contains (b, c)) ∧ 
    card_sum_identifies_missing_box distribution :=
sorry

end distribution_ways_proof_l235_235701


namespace total_points_combined_l235_235520

-- Definitions of the conditions
def Jack_points : ℕ := 8972
def Alex_Bella_points : ℕ := 21955

-- The problem statement to be proven
theorem total_points_combined : Jack_points + Alex_Bella_points = 30927 :=
by sorry

end total_points_combined_l235_235520


namespace intercepts_and_sum_l235_235054

-- Define the quadratic function representing the parabola
def parabola (x : ℝ) : ℝ := 3 * x ^ 2 - 9 * x + 5

theorem intercepts_and_sum :
  let d := parabola 0 
  let e := parabola 0
  let f := (9 - Real.sqrt 21) / 6 in
  d = 5 ∧ e = 5 ∧ (d + e + f) = (69 - Real.sqrt 21) / 6 :=
by
  unfold parabola
  sorry

end intercepts_and_sum_l235_235054


namespace students_divided_into_groups_l235_235633

theorem students_divided_into_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) (n_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : not_picked = 36) 
  (h3 : students_per_group = 7) 
  (h4 : total_students - not_picked = 28) 
  (h5 : 28 / students_per_group = 4) :
  n_groups = 4 :=
by
  sorry

end students_divided_into_groups_l235_235633


namespace count_non_foldable_positions_l235_235052

structure Shape :=
(squares : ℕ)
(T_shape : squares = 4)

def non_foldable_positions (s : Shape) : Prop :=
∀ p : ℕ, p = 9 → p ∈ {positions : ℕ // positions make the T-shape non-foldable into a cube with one face missing}

theorem count_non_foldable_positions : ∀ (s : Shape), s.T_shape → non_foldable_positions s := by
sorry

end count_non_foldable_positions_l235_235052


namespace probability_at_least_one_two_l235_235698

open ProbabilityTheory

noncomputable def prob_at_least_one_two :=
  let outcomes := Finset.univ : Finset (Fin 8 × Fin 8 × Fin 8)
  let valid_outcomes := outcomes.filter (λ xyz, xyz.1.val + xyz.2.val = 2 * xyz.3.val)
  let favorable_outcomes := valid_outcomes.filter (λ xyz, xyz.1.val = 1 ∨ xyz.2.val = 1 ∨ xyz.3.val = 1)
  (favorable_outcomes.card : ℚ) / (valid_outcomes.card : ℚ)

theorem probability_at_least_one_two : prob_at_least_one_two = 1 / 8 :=
by
  sorry -- Skip the proof

end probability_at_least_one_two_l235_235698


namespace range_of_a_l235_235143

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ 2 * a * x + 4 = 0) ↔ (-2 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l235_235143


namespace count_expressible_integers_l235_235923

open Real

noncomputable def g (x : ℝ) : ℤ := ⌊3 * x⌋ + ⌊6 * x⌋ + ⌊9 * x⌋ + ⌊12 * x⌋

theorem count_expressible_integers :
  (Finset.filter (λ n, ∃ x : ℝ, n = g x) (Finset.range 1000).toFinset).card = 633 := sorry

end count_expressible_integers_l235_235923


namespace min_value_sum_inverse_l235_235561

noncomputable def min_sum_inverse_b (b : Fin 15 → ℝ) : ℝ :=
  ∑ i, 1 / b i

theorem min_value_sum_inverse (b : Fin 15 → ℝ) (hpos : ∀ i, b i > 0) (hsum : ∑ i, b i = 1) :
  min_sum_inverse_b b = 225 := 
  sorry

end min_value_sum_inverse_l235_235561


namespace distinct_diagonals_nonagon_l235_235864

def n : ℕ := 9

def diagonals_nonagon (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem distinct_diagonals_nonagon : diagonals_nonagon n = 27 :=
by
  unfold diagonals_nonagon
  norm_num
  sorry

end distinct_diagonals_nonagon_l235_235864


namespace prob_diff_colors_correct_l235_235137

noncomputable def total_outcomes : ℕ :=
  let balls_pocket1 := 2 + 3 + 5
  let balls_pocket2 := 2 + 4 + 4
  balls_pocket1 * balls_pocket2

noncomputable def favorable_outcomes_same_color : ℕ :=
  let white_balls := 2 * 2
  let red_balls := 3 * 4
  let yellow_balls := 5 * 4
  white_balls + red_balls + yellow_balls

noncomputable def prob_same_color : ℚ :=
  favorable_outcomes_same_color / total_outcomes

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_diff_colors_correct :
  prob_different_color = 16 / 25 :=
by sorry

end prob_diff_colors_correct_l235_235137


namespace polynomial_identity_l235_235769

def S (k : ℕ) : ℕ := k.digits.sum

theorem polynomial_identity
  (P : ℕ → ℕ)
  (h₀ : ∀ n, P n = ∑ i in range (n + 1), (P 0 * P i))
  (h₁ : ∀ n ≥ 2016, 0 < P n)
  (h₂ : ∀ n ≥ 2016, S (P n) = P (S n))
  : ∀ x, P x = x :=
sorry

end polynomial_identity_l235_235769


namespace a_2n_perfect_square_l235_235994

-- Definition for the number of natural numbers with sum of digits n and each digit in {1, 3, 4}
def a_n (n : ℕ) : ℕ where
  -- Recurrence relationship for a_n when n > 4
  | 0     => 1
  | 1     => 1
  | 2     => 1
  | 3     => 2
  | 4     => 4
  | n + 1 => a_n n + a_n (n - 2) + a_n (n - 3)

theorem a_2n_perfect_square (n : ℕ) : ∃ m : ℕ, m * m = a_n (2 * n) := 
  sorry

end a_2n_perfect_square_l235_235994


namespace binomial_60_3_eq_34220_l235_235037

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235037


namespace nonagon_diagonals_count_l235_235836

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235836


namespace area_probability_expected_area_l235_235244

open_locale classical

noncomputable def area_ratio_probability : ℝ :=
  9 / 16

noncomputable def expected_area_ratio : ℝ :=
  4 / 15

theorem area_probability 
  (A B C P Q M : Point) 
  (h1 : A ≠ B) 
  (h2 : A ≠ C) 
  (h3 : B ≠ C) 
  (h4 : P ∈ LineSegment A B) 
  (h5 : Q ∈ LineSegment A C)
  (h6 : M ∈ LineSegment B C)
  (h_ratio1 : ratio A P B = 2 / 1) 
  (h_ratio2 : ratio A Q C = 1 / 3) :
  probability (area (triangle P Q M) ≤ (1 / 3) * area (triangle A B C)) = area_ratio_probability := sorry

theorem expected_area 
  (A B C P Q M : Point) 
  (h1 : A ≠ B) 
  (h2 : A ≠ C) 
  (h3 : B ≠ C) 
  (h4 : P ∈ LineSegment A B) 
  (h5 : Q ∈ LineSegment A C)
  (h6 : M ∈ LineSegment B C)
  (h_ratio1 : ratio A P B = 2 / 1) 
  (h_ratio2 : ratio A Q C = 1 / 3) :
  expected_value (ratio (area (triangle P Q M)) (area (triangle A B C))) = expected_area_ratio := sorry

end area_probability_expected_area_l235_235244


namespace intervals_of_monotonicity_f_inequality_l235_235230

noncomputable def f (x : ℝ) : ℝ := x - log x + (2*x - 1) / (x^2)

-- Question 1: Intervals of monotonicity
theorem intervals_of_monotonicity :
  (∀ x, (0 < x ∧ x < 1) ∨ (sqrt 2 < x)) → f.derivative x > 0 →
  (∀ x, 1 < x ∧ x < sqrt 2) → f.derivative x < 0 := sorry

-- Question 2: Prove the inequality
theorem f_inequality (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) : 
  f x > f.derivative x + 3 / 4 := sorry

end intervals_of_monotonicity_f_inequality_l235_235230


namespace exists_zero_in_interval_l235_235796

noncomputable def f (x : ℝ) : ℝ := (6 / x) - Real.log2 x

theorem exists_zero_in_interval : ∃ c ∈ Ioo 3 4, f c = 0 :=
by
  sorry

end exists_zero_in_interval_l235_235796


namespace distinct_diagonals_in_nonagon_l235_235904

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235904


namespace find_n_l235_235223

theorem find_n (n : ℕ) :
  let x := (1 + 2) * (1 + 2 ^ 2) * (1 + 2 ^ 4) * (1 + 2 ^ 8) * (1 + 2 ^ n) in
  x + 1 = 2 ^ 128 →
  n = 64 :=
by
  intros x h
  -- placeholder for the proof steps
  sorry

end find_n_l235_235223


namespace unique_triple_l235_235085

open Nat

variable {k m n p : ℕ}

/-- Proving the only triple (k, m, n) of positive integers satisfying the given conditions is (28, 5, 2023) -/
theorem unique_triple (k m n : ℕ) (prime_m : Prime m) (h1 : (k * n).is_square)
  (h2 : ((k * (k - 1)) / 2 + n).is_prime_pow 4)
  (h3 : ∃ (p : ℕ), Prime p ∧ k = m^2 + p)
  (h4 : ∃ (p : ℕ), Prime p ∧ (n + 2) / m^2 = p^4) : 
  (k, m, n) = (28, 5, 2023) := 
  sorry

end unique_triple_l235_235085


namespace base_edge_length_l235_235635

theorem base_edge_length (x : ℕ) :
  (∃ (x : ℕ), 
    (∀ (sum_edges : ℕ), sum_edges = 6 * x + 48 → sum_edges = 120) →
    x = 12) := 
sorry

end base_edge_length_l235_235635


namespace complement_intersection_l235_235810

open Set

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (M_def : M = {2, 3})
variable (N_def : N = {1, 4})

theorem complement_intersection (U M N : Set ℕ) (U_def : U = {1, 2, 3, 4, 5, 6}) (M_def : M = {2, 3}) (N_def : N = {1, 4}) :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  sorry

end complement_intersection_l235_235810


namespace area_inner_square_l235_235964

theorem area_inner_square (ABCD_side : ℝ) (BE : ℝ) (EFGH_area : ℝ) 
  (h1 : ABCD_side = Real.sqrt 50) 
  (h2 : BE = 1) :
  EFGH_area = 36 :=
by
  sorry

end area_inner_square_l235_235964


namespace triangle_side_lengths_l235_235160

variable {c z m : ℕ}

axiom condition1 : 3 * c + z + m = 43
axiom condition2 : c + z + 3 * m = 35
axiom condition3 : 2 * (c + z + m) = 46

theorem triangle_side_lengths : c = 10 ∧ z = 7 ∧ m = 6 := 
by 
  sorry

end triangle_side_lengths_l235_235160


namespace max_m_value_l235_235089

-- The condition given by the inequality
def inequality_holds (m x : ℝ) : Prop :=
  m * sqrt(m) * (x^2 - 6*x + 9) + sqrt(m) / (x^2 - 6*x + 9) ≤ real.sqrt((m^3)^(1 / 4)) * abs (real.cos (real.pi * x / 5))

-- The statement to prove
theorem max_m_value : ∃ (m : ℝ), (∀ x : ℝ, inequality_holds m x) ∧ m = 1 / 16 := sorry

end max_m_value_l235_235089


namespace minimum_value_l235_235455

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2 * y = 3) :
  (1 / x + 1 / y) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end minimum_value_l235_235455


namespace composition_value_l235_235553

noncomputable def P (x : ℝ) : ℝ := 3 * real.sqrt x
noncomputable def Q (x : ℝ) : ℝ := x ^ 3

theorem composition_value :
  P (Q (P (Q (P (Q 2))))) = 846 * real.sqrt 2 :=
by sorry

end composition_value_l235_235553


namespace distance_foci_of_hyperbola_l235_235416

noncomputable def distance_between_foci : ℝ :=
  8 * Real.sqrt 5

theorem distance_foci_of_hyperbola :
  ∃ A B : ℝ, (9 * A^2 - 36 * A - B^2 + 4 * B = 40) → distance_between_foci = 8 * Real.sqrt 5 :=
sorry

end distance_foci_of_hyperbola_l235_235416


namespace cost_of_6_bottle_caps_l235_235405

-- Define the cost of each bottle cap
def cost_per_bottle_cap : ℕ := 2

-- Define how many bottle caps we are buying
def number_of_bottle_caps : ℕ := 6

-- Define the total cost of the bottle caps
def total_cost : ℕ := 12

-- The proof statement to prove that the total cost is as expected
theorem cost_of_6_bottle_caps :
  cost_per_bottle_cap * number_of_bottle_caps = total_cost :=
by
  sorry

end cost_of_6_bottle_caps_l235_235405


namespace num_diagonals_convex_nonagon_l235_235870

-- We need to define what a convex nonagon is, and then prove the number of diagonals.
-- Note: We provide the necessary condition about the problem without referring to the solution process directly.

def is_convex_nonagon (P : Type) [fintype P] : Prop :=
  fintype.card P = 9 ∧ ∀ a b c : P, ¬(a = b ∨ b = c ∨ c = a)

theorem num_diagonals_convex_nonagon (P : Type) [fintype P] (h : is_convex_nonagon P) : 
  ∃ (n : ℕ), n = 27 :=
sorry

end num_diagonals_convex_nonagon_l235_235870


namespace count_expressible_integers_l235_235922

open Real

noncomputable def g (x : ℝ) : ℤ := ⌊3 * x⌋ + ⌊6 * x⌋ + ⌊9 * x⌋ + ⌊12 * x⌋

theorem count_expressible_integers :
  (Finset.filter (λ n, ∃ x : ℝ, n = g x) (Finset.range 1000).toFinset).card = 633 := sorry

end count_expressible_integers_l235_235922


namespace sum_of_angles_is_180_l235_235360

-- Define a quadrilateral and a circumscribed circle scenario
variable (A B C D O : Type) -- Points in geometric space
variable (is_quadrilateral : Prop) -- A quadrilateral condition
variable (circumscribed : Prop) -- Circumscribed circle condition

-- The conditions given
axiom quadrilateral (h: is_quadrilateral)
axiom circumscribed_circle (h: circumscribed)

-- The proof statement
theorem sum_of_angles_is_180
  (AOB COD : ℝ)
  (h1 : quadrilateral ABCD)
  (h2 : circumscribed_circle ABCD O) :
  AOB + COD = 180 :=
sorry

end sum_of_angles_is_180_l235_235360


namespace binary_110011_is_51_l235_235396

def binary := [1, 1, 0, 0, 1, 1]

noncomputable def binary_to_decimal (b : List Nat) : Nat :=
  b.reverse.enum_from 0
    |>.map (λ (p : Nat × Nat) => p.1 * 2^(p.2))
    |>.sum

theorem binary_110011_is_51 :
  binary_to_decimal binary = 51 :=
by sorry

end binary_110011_is_51_l235_235396


namespace sum_of_integers_l235_235765

theorem sum_of_integers (S : Finset ℕ) (h : ∀ n ∈ S, 1.5 * n - 6.3 < 7.5) :
  S.sum id = 45 :=
sorry

end sum_of_integers_l235_235765


namespace first_1000_integers_represented_l235_235925

def floor_sum (x : ℝ) : ℤ :=
  (⌊3 * x⌋ : ℤ) + (⌊6 * x⌋ : ℤ) + (⌊9 * x⌋ : ℤ) + (⌊12 * x⌋ : ℤ)

theorem first_1000_integers_represented : 
  ∃ n : ℕ, n = 880 ∧ ∀ i : ℤ, 1 ≤ i ∧ i ≤ 1000 → ∃ x : ℝ, floor_sum x = i :=
sorry

end first_1000_integers_represented_l235_235925


namespace polynomial_inequality_cases_of_equality_l235_235993

noncomputable def polynomial_real_roots (P : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, Polynomial.eval x P = 0 → ∃ y : ℝ, x = y

theorem polynomial_inequality (P : Polynomial ℝ)
  (h1 : ∀ x : ℝ, Polynomial.eval x P = 0 → ∃ y : ℝ, x = y) :
  (P.degree - 1) * (Polynomial.deriv P)^2 ≥ P.degree * P * (Polynomial.deriv (Polynomial.deriv P)) :=
sorry

theorem cases_of_equality (P : Polynomial ℝ)
  (h1 : ∀ x : ℝ, Polynomial.eval x P = 0 → ∃ y : ℝ, x = y) :
  (P.degree - 1) * (Polynomial.deriv P)^2 = P.degree * P * (Polynomial.deriv (Polynomial.deriv P)) ↔ 
  ∃ (c : ℝ) (a : ℝ), P = Polynomial.C c * (Polynomial.X - Polynomial.C a) ^ P.degree :=
sorry

end polynomial_inequality_cases_of_equality_l235_235993


namespace smallest_positive_period_range_lambda_l235_235146

-- Definitions and conditions
def f (x : ℝ) (ω : ℝ) (λ : ℝ) : ℝ :=
  sin (ω * x) ^ 2 + (2 * sqrt 3 * sin (ω * x) - cos (ω * x)) * cos (ω * x) - λ

theorem smallest_positive_period (ω : ℝ) : (ω ∈ (1 / 2 : ℝ), 1)) ∧ (∀ x, f x π ω λ = f (π - x) π ω λ) → T = (6 * π) / 5 :=
by sorry

theorem range_lambda (ω : ℝ) (λ : ℝ) : (ω ∈ (1 / 2 : ℝ), 1)) ∧ (∃ x_0 ∈ [0, 3 * π / 5], f x_0 ω λ = 0) → λ ∈ [-1, 2] :=
by sorry

end smallest_positive_period_range_lambda_l235_235146


namespace taylor_correct_answers_percentage_l235_235724

theorem taylor_correct_answers_percentage 
  (N : ℕ := 30)
  (alex_correct_alone_percentage : ℝ := 0.85)
  (alex_overall_percentage : ℝ := 0.83)
  (taylor_correct_alone_percentage : ℝ := 0.95)
  (alex_correct_alone : ℕ := 13)
  (alex_correct_total : ℕ := 25)
  (together_correct : ℕ := 12)
  (taylor_correct_alone : ℕ := 14)
  (taylor_correct_total : ℕ := 26) :
  ((taylor_correct_total : ℝ) / (N : ℝ)) * 100 = 87 :=
by
  sorry

end taylor_correct_answers_percentage_l235_235724


namespace katies_games_is_81_l235_235210

-- Define friends' games and excess games
constant friends_games : ℕ
constant excess_games : ℕ

-- Assign known values to friends_games and excess_games
axiom h1 : friends_games = 59
axiom h2 : excess_games = 22

-- Define Katie's games as sum of friends' games and excess_games
def katies_games : ℕ := friends_games + excess_games

-- State the theorem that Katie has 81 DS games
theorem katies_games_is_81 : katies_games = 81 :=
by 
  rw [katies_games, h1, h2]
  -- This is where you'd apply the solution step, but add sorry to skip the proof
  sorry

end katies_games_is_81_l235_235210


namespace jose_gave_rebecca_two_caps_l235_235542

variables (initial : ℝ) (left : ℝ) (given : ℝ)

axiom initial_bottle_caps : initial = 7.0
axiom remaining_bottle_caps : left = 5.0
axiom given_bottle_caps : given = initial - left

theorem jose_gave_rebecca_two_caps (initial left : ℝ) : given = 2.0 :=
by
  rw [initial_bottle_caps, remaining_bottle_caps, given_bottle_caps]
  norm_num
  exact sorry

end jose_gave_rebecca_two_caps_l235_235542


namespace inequality_of_ab_l235_235544

theorem inequality_of_ab (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 :=
by
  sorry

end inequality_of_ab_l235_235544


namespace nonagon_diagonals_count_l235_235893

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235893


namespace number_of_divisors_of_64n3_l235_235106

theorem number_of_divisors_of_64n3 (n : ℕ) (hn : n > 0) (h_divisors_150n2 : (150 * n^2).factorization.prod (\(p, e) => e + 1) = 150) : 
  (((64 * n^3).factorization.prod (\(p, e) => e + 1)) = 160) :=
sorry

end number_of_divisors_of_64n3_l235_235106


namespace min_value_expr_l235_235568

open Real

noncomputable def expr (y : ℝ) : ℝ := 9 * y^3 + 4 * y^(-6)

theorem min_value_expr : ∀ y : ℝ, 0 < y → expr y ≥ 13 :=
by
  intro y hy
  sorry

end min_value_expr_l235_235568


namespace binomial_60_3_l235_235026

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235026


namespace decreasing_function_condition_l235_235138

-- Define the function and its derivative
def f (a b x : ℝ) : ℝ := -x^3 + a*x^2 + b*x - 7
def f' (a b x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

-- Define the condition
def discriminant (a b : ℝ) : ℝ := 4*a^2 + 12*b

-- The proof statement
theorem decreasing_function_condition (a b : ℝ) :
  (∀ x : ℝ, f' a b x ≤ 0) → a^2 + 3b ≤ 0 :=
by
  sorry

end decreasing_function_condition_l235_235138


namespace total_word_count_is_5000_l235_235003

def introduction : ℕ := 450
def conclusion : ℕ := 3 * introduction
def body_sections : ℕ := 4 * 800

def total_word_count : ℕ := introduction + conclusion + body_sections

theorem total_word_count_is_5000 : total_word_count = 5000 := 
by
  -- Lean proof code will go here.
  sorry

end total_word_count_is_5000_l235_235003


namespace area_triangle_APB_l235_235710

-- Definitions for the problem
def square_side_length := 8
def A := (0, 0)
def B := (square_side_length, 0)
def C := (square_side_length, square_side_length)
def D := (0, square_side_length)

structure Point := (x : ℝ) (y : ℝ)

-- Conditions
axiom PA_eq_PB_eq_PC (P : Point) : 
  dist (P.x, P.y) A = dist (P.x, P.y) B ∧ 
  dist (P.x, P.y) B = dist (P.x, P.y) C

axiom PD_perpendicular_AB (P : Point) : 
  ∃ E : Point, 
    E.x = square_side_length / 2 ∧ 
    E.y = 0 ∧ 
    dist P.x 0 = dist P.y E.y

-- Theorem that needs to be proven
theorem area_triangle_APB (P : Point) 
  (H1 : PA_eq_PB_eq_PC P) 
  (H2 : PD_perpendicular_AB P) : 
  (1 / 2) * square_side_length * (square_side_length - dist P.y 0) = 12 := 
by
  sorry

end area_triangle_APB_l235_235710


namespace probability_of_waiting_time_less_than_10_minutes_l235_235351

-- Define the scenario
def bus_departure_times : List ℕ := [420, 480, 510] -- Depatures in minutes from 00:00, i.e., 7:00, 8:00, 8:30.
def arrival_window_start : ℕ := 470 -- in minutes, i.e., 7:50
def arrival_window_end : ℕ := 510 -- in minutes, i.e., 8:30
def favorable_intervals : List (ℕ × ℕ) := [(470, 480), (500, 510)] -- (start, end) intervals in minutes
def total_window_duration : ℕ := arrival_window_end - arrival_window_start -- 40 minutes

-- Function to calculate the combined length of favorable intervals
def combined_favorable_duration (intervals : List (ℕ × ℕ)) : ℕ :=
  intervals.foldr (λ (interval : ℕ × ℕ) acc, acc + (interval.snd - interval.fst)) 0

def favorable_duration : ℕ := combined_favorable_duration favorable_intervals -- 20 minutes

-- Probability calculation
def probability : ℚ := favorable_duration / total_window_duration

-- The proof statement
theorem probability_of_waiting_time_less_than_10_minutes :
  probability = 1 / 2 :=
by
  sorry

end probability_of_waiting_time_less_than_10_minutes_l235_235351


namespace sum_binom_3_pow_eq_l235_235558

open Nat

def binom : ℕ → ℕ → ℕ
| n, k => Nat.choose n k

noncomputable def S (n : ℕ) : ℕ :=
  ∑ k in (Finset.range (n / 2 + 1)), binom n (2 * k) * 3 ^ (n - 2 * k)

theorem sum_binom_3_pow_eq:
  ∀ (n : ℕ), 0 < n → S n = 2 * 4^(n-1) + 2^(n-1) := by
  sorry

end sum_binom_3_pow_eq_l235_235558


namespace gold_bars_lost_l235_235262

-- Define the problem constants
def initial_bars : ℕ := 100
def friends : ℕ := 4
def bars_per_friend : ℕ := 20

-- Define the total distributed gold bars
def total_distributed : ℕ := friends * bars_per_friend

-- Define the number of lost gold bars
def lost_bars : ℕ := initial_bars - total_distributed

-- Theorem: Prove that the number of lost gold bars is 20
theorem gold_bars_lost : lost_bars = 20 := by
  sorry

end gold_bars_lost_l235_235262


namespace right_triangle_area_l235_235616

theorem right_triangle_area (ABC : Type)
  (A B C : ABC)
  (a : ℝ) (h : ℝ) (area : ℝ)
  (h_angle_1 : ∠BAC = π / 2)
  (h_angle_2 : ∠ABC = π / 3)
  (h_angle_3 : ∠BCA = π / 6)
  (h_altitude : h = 4) :
  area = (16 * real.sqrt 3) / 3 :=
by 
  sorry

end right_triangle_area_l235_235616


namespace PQ_composition_l235_235557

def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem PQ_composition : P (Q (P (Q (P (Q 2))))) = 54 := 
by sorry

end PQ_composition_l235_235557


namespace nonagon_diagonals_count_l235_235835

theorem nonagon_diagonals_count (n : ℕ) (h : n = 9) : (n * (n - 3)) / 2 = 27 := by
  rw [h]
  norm_num
  sorry

end nonagon_diagonals_count_l235_235835


namespace distinct_diagonals_convex_nonagon_l235_235854

theorem distinct_diagonals_convex_nonagon :
  let n := 9 in
  let k := (n - 3) in
  let total_diagonals := (n * k) / 2 in
  total_diagonals = 27 :=
by
  sorry

end distinct_diagonals_convex_nonagon_l235_235854


namespace nonagon_diagonals_count_l235_235919

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235919


namespace angle_equality_l235_235980

-- Definitions based on the conditions
variables (ω ω1 ω2 : Type*) -- Circles
variables (O O1 O2 : Type*) -- Centers of the circles
variable P : Type* -- Tangency point of ω1 and ω2
variable A : Type* -- Point of internal tangency of ω1 and ω
variable B : Type* -- Point of internal tangency of ω and ω2
variable X : Type* -- Foot of the perpendicular from P to AB 

-- Given conditions
variable (mutual_tangent : ω1 ∩ ω2 ≠ ∅ ∧ P ∈ ω1 ∩ ω2)
variable (int_tangent1 : A ∈ ω1 ∩ ω)
variable (int_tangent2 : B ∈ ω2 ∩ ω)
variable (centers : O1 = center(ω1) ∧ O2 = center(ω2) ∧ O = center(ω))
variable (perpendicular_foot : foot P (line(A, B)) = X)

-- The statement to prove
theorem angle_equality : ∠O1XP = ∠O2XP :=
sorry

end angle_equality_l235_235980


namespace total_yards_run_l235_235579

-- Define the yardages and games for each athlete
def Malik_yards_per_game : ℕ := 18
def Malik_games : ℕ := 5

def Josiah_yards_per_game : ℕ := 22
def Josiah_games : ℕ := 7

def Darnell_yards_per_game : ℕ := 11
def Darnell_games : ℕ := 4

def Kade_yards_per_game : ℕ := 15
def Kade_games : ℕ := 6

-- Prove that the total yards run by the four athletes is 378
theorem total_yards_run :
  (Malik_yards_per_game * Malik_games) +
  (Josiah_yards_per_game * Josiah_games) +
  (Darnell_yards_per_game * Darnell_games) +
  (Kade_yards_per_game * Kade_games) = 378 :=
by
  sorry

end total_yards_run_l235_235579


namespace balls_into_boxes_l235_235165

-- Define the conditions
def balls : ℕ := 7
def boxes : ℕ := 4

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the equivalent proof problem
theorem balls_into_boxes :
    (binom (balls - 1) (boxes - 1) = 20) ∧ (binom (balls + (boxes - 1)) (boxes - 1) = 120) := by
  sorry

end balls_into_boxes_l235_235165


namespace volume_of_solid_of_revolution_l235_235672

/-- Calculate the volumes of the solids obtained by rotating the figures bounded by the function graphs
    y^2 = x - 2, y = x^3, y = 0, y = 1 around the y-axis. -/
theorem volume_of_solid_of_revolution :
  let f₁ y := y^2 + 2,
      f₂ y := y^(3 : ℝ),
      a := 0,
      b := 1 in
  (π * ∫ y in a..b, (f₁ y)^2 - π * ∫ y in a..b, (f₂ y)^2) = (24 / 5) * π :=
by sorry

end volume_of_solid_of_revolution_l235_235672


namespace binom_60_3_l235_235006

theorem binom_60_3 : nat.choose 60 3 = 34220 :=
by
  -- Proof goes here.
  sorry

end binom_60_3_l235_235006


namespace smallest_gamma_l235_235566

variable {n : ℕ} (x y : Finₙ → ℝ)

theorem smallest_gamma 
  (hn : n ≥ 2) 
  (hx : ∀ i, x i > 0) 
  (hy : ∀ i, 0 ≤ y i ∧ y i ≤ 0.5) 
  (hx_sum : (Finₙ.sum x) = 1) 
  (hy_sum : (Finₙ.sum y) = 1) : 
  let γ := (1/2) * ((1/(n-1))^(n-1)) in
  (Finₙ.prod x) ≤ γ * (Finₙ.sum (λ i, x i * y i)) :=
sorry

end smallest_gamma_l235_235566


namespace ages_sum_is_71_l235_235377

def Beckett_age : ℕ := 12
def Olaf_age : ℕ := Beckett_age + 3
def Shannen_age : ℕ := Olaf_age - 2
def Jack_age : ℕ := 2 * Shannen_age + 5
def sum_of_ages : ℕ := Beckett_age + Olaf_age + Shannen_age + Jack_age

theorem ages_sum_is_71 : sum_of_ages = 71 := by
  unfold sum_of_ages Beckett_age Olaf_age Shannen_age Jack_age
  calc
    12 + (12 + 3) + (12 + 3 - 2) + (2 * (12 + 3 - 2) + 5)
      = 12 + 15 + 13 + 31 := by rfl
      ... = 71 := by rfl

end ages_sum_is_71_l235_235377


namespace sum_of_q_p_values_is_neg29_l235_235575

def p(x : ℤ) : ℤ := |x| - 3

def q(x : ℤ) : ℤ := -(x * x)

def values := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def sum_q_p_values : ℤ := values.map (λ x => q (p x)).sum

theorem sum_of_q_p_values_is_neg29 : 
  sum_q_p_values = -29 :=
  sorry

end sum_of_q_p_values_is_neg29_l235_235575


namespace inscribed_semicircle_radius_l235_235958

theorem inscribed_semicircle_radius (XY YZ : ℝ) (hXY : XY = 15) (hYZ : YZ = 8) (h_right_angle : ∠Z = 90) :
  let XZ := real.sqrt (XY^2 + YZ^2),
      area := 1/2 * XY * YZ,
      s := (XY + YZ + XZ) / 2,
      r := area / s
  in r = 3 :=
by
  sorry

end inscribed_semicircle_radius_l235_235958


namespace distinct_diagonals_in_convex_nonagon_l235_235882

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235882


namespace main_roads_example_l235_235950

-- Condition declarations
def City := ℕ
def Road := City × City

-- Given a set of 100 cities
def cities : Finset City := {i ∈ Finset.range 100 | true}

-- Assume we have a set of roads connecting these cities without intersections 
-- Note that the verification of roads not intersecting is abstracted away
def roads (r : Finset Road) : Prop :=
  ∀ (c1 c2 : City), c1 ≠ c2 → c1 ∈ cities → c2 ∈ cities → 
  (∃ (r ∈ r), r = (c1, c2) ∧ ¬(r = (c2, c1))) ∧ (roads c1 c2 → ∃ (r1 r2 : Road), r1 ≠ r2 ∧ r1 ∈ r ∧ r2 ∈ r ∧ r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2)

-- You can travel from any city to any other city using the roads
-- Ensuring connectivity of the graph
def is_connected (r : Finset Road) : Prop := 
  ∀ (c1 c2 : City), c1 ∈ cities → c2 ∈ cities → reachable r c1 c2

-- Now formalize the main statement
theorem main_roads_example :
  ∃ (main_roads : Finset Road), 
    roads main_roads ∧ is_connected main_roads ∧ 
    ∀ (c : City), c ∈ cities → Odd (Finset.card (main_roads.filter (λ r, r.1 = c ∨ r.2 = c))) := sorry

end main_roads_example_l235_235950


namespace mean_squared_sum_l235_235268

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

theorem mean_squared_sum :
  (x + y + z = 30) ∧ 
  (xyz = 125) ∧ 
  ((1 / x + 1 / y + 1 / z) = 3 / 4) 
  → x^2 + y^2 + z^2 = 712.5 :=
by
  intros h
  have h₁ : x + y + z = 30 := h.1
  have h₂ : xyz = 125 := h.2.1
  have h₃ : (1 / x + 1 / y + 1 / z) = 3 / 4 := h.2.2
  sorry

end mean_squared_sum_l235_235268


namespace binom_60_3_eq_34220_l235_235043

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235043


namespace range_of_x_minus_y_times_z_l235_235425

variables {x y z : ℝ}

theorem range_of_x_minus_y_times_z (h1 : -3 < x) (h2 : x < y) (h3 : y < 1) (h4 : -4 < z) (h5 : z < 0) : 
  0 < (x - y) * z ∧ (x - y) * z < 16 :=
begin
  sorry
end

end range_of_x_minus_y_times_z_l235_235425


namespace distinct_digit_sum_impossible_l235_235113

/-- Given 111 distinct natural numbers, each not exceeding 500, we prove that it is not possible 
that each of these numbers has its last digit coinciding with the last digit of the sum of the 
other 110 numbers. -/
theorem distinct_digit_sum_impossible (a : ℕ → ℕ) (h1 : ∀ i j, i ≠ j → a i ≠ a j) (h2 : ∀ i, a i ≤ 500) :
  ¬ (∀ k, (a k % 10) = (∑ i, if i ≠ k then a i % 10 else 0) % 10) :=
by
  sorry

end distinct_digit_sum_impossible_l235_235113


namespace odd_function_expression_l235_235459

theorem odd_function_expression (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_nonneg : ∀ x : ℝ, 0 ≤ x → f x = x * (1 + 3 * x)) :
  ∀ x : ℝ, x < 0 → f x = x * (1 - 3 * x) :=
begin
  intros x hx,
  have h_neg : 0 ≤ -x := by linarith,
  rw [← h_odd, h_nonneg (-x) h_neg],
  ring,
end

end odd_function_expression_l235_235459


namespace vector_t_solution_l235_235528

theorem vector_t_solution (t : ℝ) :
  ∃ t, (∃ (AB AC BC : ℝ × ℝ), 
         AB = (t, 1) ∧ AC = (2, 2) ∧ BC = (2 - t, 1) ∧ 
         (AC.1 - AB.1) * AC.1 + (AC.2 - AB.2) * AC.2 = 0 ) → 
         t = 3 :=
by {
  sorry -- proof content omitted as per instructions
}

end vector_t_solution_l235_235528


namespace find_a_tangent_line_l235_235802

noncomputable def quadratic_function (a x : ℝ) : ℝ :=
a * x^2 + (a + 2) * x + 1

noncomputable def given_function (x : ℝ) : ℝ :=
x + 1 + Real.log x

theorem find_a_tangent_line 
  (h_tangent_given: ∀ x: ℝ, x = 1 → has_deriv_at given_function 2 x)
  (h_eq_tangent: ∃ (x: ℝ), quadratic_function 4 x = 2 * x) :
  a = 4 := 
sorry

end find_a_tangent_line_l235_235802


namespace nathan_tokens_used_is_18_l235_235584

-- We define the conditions as variables and constants
variables (airHockeyGames basketballGames tokensPerGame : ℕ)

-- State the values for the conditions
def Nathan_plays : Prop :=
  airHockeyGames = 2 ∧ basketballGames = 4 ∧ tokensPerGame = 3

-- Calculate the total tokens used
def totalTokensUsed (airHockeyGames basketballGames tokensPerGame : ℕ) : ℕ :=
  (airHockeyGames * tokensPerGame) + (basketballGames * tokensPerGame)

-- Proof statement 
theorem nathan_tokens_used_is_18 : Nathan_plays airHockeyGames basketballGames tokensPerGame → totalTokensUsed airHockeyGames basketballGames tokensPerGame = 18 :=
by 
  sorry

end nathan_tokens_used_is_18_l235_235584


namespace div_count_64n3_l235_235107

-- We assume n is a positive integer such that 150n^2 has exactly 150 divisors.
def meets_conditions (n : ℕ) : Prop :=
  ∃ counts : List ℕ, 
    150 = List.foldl (*) 1 counts ∧
    List.foldl (*) 1 (List.map (λ x, x + 1) (counts.map (λ k, if k > 0 then k else 1))) = 150 ∧
    counts.length = 3

theorem div_count_64n3 {n : ℕ} (h : meets_conditions n) : 
  let counts := List.map (λ k, if k > 0 then k else 1) [3, 3, 1] in
  List.foldl (*) 1 (List.map (λ x, x + 1) (counts.map (λ k, k * 3))) = 70 :=
by
  sorry

end div_count_64n3_l235_235107


namespace binom_60_3_l235_235015

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235015


namespace sum_f_values_l235_235620

section

-- Define the function f
variable (f : ℝ → ℝ)

-- Property 1: f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (x) = f (-x)

-- Property 2: f is symmetric about x = 4
def is_symmetric_about_4 (f : ℝ → ℝ) : Prop := ∀ x, f (4 + x) = f (4 - x)

-- Property 3: f(x) in the interval [-4, 0] is given by x + 2
def f_interval (f : ℝ → ℝ) : Prop := ∀ x, x ∈ set.Icc (-4 : ℝ) 0 → f x = x + 2

-- The goal to prove
theorem sum_f_values (h_even: is_even f) (h_symm: is_symmetric_about_4 f) (h_interval: f_interval f) :
  (f 0) + (f 1) + (f 2) + (f 3) + (f 4) + (f 5) + (f 6) + (f 7) + (f 8) + (f 9) = 3 :=
sorry

end

end sum_f_values_l235_235620


namespace geometric_sequence_q_value_l235_235784

theorem geometric_sequence_q_value 
  (d : ℝ) (q : ℝ) (h_d_ne_0 : d ≠ 0) (h_q_lt_1 : q < 1) 
  (h_S3_T3_pos_int : (14 / (1 + q + q^2)) ∈ ℤ) : 
  q = 1 / 2 := 
sorry

end geometric_sequence_q_value_l235_235784


namespace find_unknown_rate_l235_235364

theorem find_unknown_rate :
  ∃ x : ℝ, (300 + 750 + 2 * x) / 10 = 170 ↔ x = 325 :=
by
    sorry

end find_unknown_rate_l235_235364


namespace sum_of_possible_values_of_N_l235_235496

variable (N S : ℝ) (hN : N ≠ 0)

theorem sum_of_possible_values_of_N : 
  (3 * N + 5 / N = S) → 
  ∀ N1 N2 : ℝ, (3 * N1^2 - S * N1 + 5 = 0) ∧ (3 * N2^2 - S * N2 + 5 = 0) → 
  N1 + N2 = S / 3 :=
by 
  intro hS hRoots
  sorry

end sum_of_possible_values_of_N_l235_235496


namespace inradius_of_scalene_triangle_l235_235559

open Real

variables {A B C I : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace I]

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inradius (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  heron_area a b c / s

theorem inradius_of_scalene_triangle (a b c ic : ℝ) : 
  a = 30 → b = 36 → c = 34 → ic = 18 → inradius 30 36 34 = 0.8 * Real.sqrt 14 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  unfold inradius semiperimeter heron_area
  -- we can show the internal computation but skip the rest of the proof
  sorry

end inradius_of_scalene_triangle_l235_235559


namespace number_of_diagonals_in_nonagon_l235_235829

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235829


namespace last_remaining_number_l235_235543

def evens_upto (n : Nat) : List Nat :=
  List.filter (λ k, k % 2 = 0) (List.range (n + 1))

def elimination_step (l : List Nat) : List Nat :=
  List.filter (λ x, ¬ (List.indexOf x l) % 2 = 0) l

def final_number (l : List Nat) : Nat :=
  if l.length = 1 then l.head! else final_number (elimination_step l)

theorem last_remaining_number :
  final_number (evens_upto 200) = 128 :=
by
  sorry

end last_remaining_number_l235_235543


namespace equalize_glasses_l235_235634

noncomputable def equalize_water (n : ℕ) (A : fin n → ℕ) (ops : list (fin n × fin n)) : (fin n → ℕ) :=
sorry

theorem equalize_glasses :
  ∃ ops : list (fin 8 × fin 8), ∀ A : fin 8 → ℕ, 
  (equalize_water 8 A ops) = λ _, A 0 :=
sorry

end equalize_glasses_l235_235634


namespace min_value_of_reciprocals_l235_235446

open Real

theorem min_value_of_reciprocals (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) :
  (1 / a) + (1 / (b + 1)) ≥ 2 :=
sorry

end min_value_of_reciprocals_l235_235446


namespace equation_of_circle_l235_235618

-- Definitions based on the conditions provided
def center_on_y_axis (h : ℝ) : Prop := ∃ a : ℝ, a = 0
def radius (r : ℝ) := r = 1
def passes_through_point (x y : ℝ) := (x = 1 ∧ y = 2)

-- The statement to prove
theorem equation_of_circle {h r : ℝ} (center : center_on_y_axis h) (radius1 : radius r) (point : passes_through_point 1 2) :
    ∃ b : ℝ, (1^2 + (2 - b)^2 = 1) ∧ (b = 2) :=
begin
  sorry -- Proof is omitted
end

end equation_of_circle_l235_235618


namespace max_area_equilateral_triangle_in_rectangle_l235_235292

theorem max_area_equilateral_triangle_in_rectangle (a b : ℝ) (h_a : a = 12) (h_b : b = 13) :
  ∃ (T : ℝ), T = (117 * Real.sqrt 3) - 108 :=
by
  simp [h_a, h_b]
  use (117 * Real.sqrt 3) - 108
  sorry

end max_area_equilateral_triangle_in_rectangle_l235_235292


namespace book_price_decrease_l235_235173

theorem book_price_decrease (P : ℝ) (hP : 0 ≤ P) :
  let new_price_after_decrease := P * 0.75 in
  let final_price := new_price_after_decrease + new_price_after_decrease * 0.20 in
  P - final_price = P * 0.10 :=
by
  sorry

end book_price_decrease_l235_235173


namespace perpendicular_vectors_k_value_l235_235155

theorem perpendicular_vectors_k_value : 
  let a := (2, -1, 3)
  let b := (4, -2, k)
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0 →
  k = -10 / 3 :=
by
  sorry

end perpendicular_vectors_k_value_l235_235155


namespace num_diff_squares_2008_l235_235079

theorem num_diff_squares_2008 :
  (∃ n : ℕ, finset.card (finset.filter (λ (p : ℤ × ℤ), p.1^2 - p.2^2 = 2008) (finset.product (finset.range 2009) (finset.range 2009))) = n) ∧
  n = 8 :=
begin
  sorry
end

end num_diff_squares_2008_l235_235079


namespace distinct_diagonals_in_nonagon_l235_235902

/-- 
The number of distinct diagonals in a convex nonagon (9-sided polygon) is 27.
-/
theorem distinct_diagonals_in_nonagon : 
  (∑ i in (finset.range 9), 9 - 3) / 2 = 27 := 
by
  -- This theorem asserts that the number of distinct diagonals in a convex nonagon is 27.
  sorry

end distinct_diagonals_in_nonagon_l235_235902


namespace number_of_possible_values_l235_235213

def S := {x : ℝ // 0 < x}

def f : S → S := sorry

axiom functional_eq (x y : S) (h : x.val + y.val ≠ 0) : 
  f x + f y = f ⟨x.val * x.val * y.val * f ⟨x.val + y.val, sorry⟩, sorry⟩

theorem number_of_possible_values {n s : ℝ} (h1 : f ⟨3, by norm_num⟩ = ⟨1/3, by norm_num⟩) 
  (h2 : n = 1) (h3 : s = 1/3) : n * s = 1/3 := 
begin
  rw [h2, h3],
  norm_num,
end

end number_of_possible_values_l235_235213


namespace sandy_paid_cost_shop2_l235_235249

-- Define the conditions
def books_shop1 : ℕ := 65
def cost_shop1 : ℕ := 1380
def books_shop2 : ℕ := 55
def avg_price_per_book : ℕ := 19

-- Calculation of the total amount Sandy paid for the books from the second shop
def cost_shop2 (total_books: ℕ) (avg_price: ℕ) (cost1: ℕ) : ℕ :=
  (total_books * avg_price) - cost1

-- Define the theorem we want to prove
theorem sandy_paid_cost_shop2 : cost_shop2 (books_shop1 + books_shop2) avg_price_per_book cost_shop1 = 900 :=
sorry

end sandy_paid_cost_shop2_l235_235249


namespace treewidth_bound_l235_235224

theorem treewidth_bound (G: Graph) (h k : ℕ) (hk : h ≥ k) (k1 : k ≥ 1) 
    (no_k_book : ∀ (subG : k-book_with_h_pages), ¬ contains G subG) : 
    treewidth G < h + k - 1 :=
sorry

end treewidth_bound_l235_235224


namespace solve_for_x_l235_235421

-- Definition of the operation ⊕
def op (a b : ℝ) : ℝ := (1 / b) - (1 / a)

-- Theorem stating that x = 5/6 satisfies the given equation
theorem solve_for_x : ∃ x : ℝ, 2 ⊗ (2 * x - 1) = 1 ∧ x = 5/6 := 
by {
  sorry
}

end solve_for_x_l235_235421


namespace bakery_rolls_combinations_l235_235348

theorem bakery_rolls_combinations : ∃ n, n = 15 ∧ (∃ x y z : ℕ, x + y + z = 7 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ binomial (4 + 3 - 1) (3 - 1) = n) :=
by
  use 15
  constructor
  {
    refl
  }
  {
    use 1
    use 1
    use 1
    sorry
  }

end bakery_rolls_combinations_l235_235348


namespace max_area_triangle_l235_235781

/-- The coordinates of points A and B that lie on a parabola defined by y^2 = 4x. -/
def parabola : set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

/-- Points A and B. -/
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (4, -4)

/-- The focus of the parabola y^2 = 4x. -/
def F : ℝ × ℝ := (1, 0)

/-- Prove that the area of triangle PAB is maximized when P has coordinates (1/4, -1). -/
theorem max_area_triangle :
  (∃ (P : ℝ × ℝ) (hp : P ∈ parabola), 0 ≤ P.1 ∧ P.1 ≤ 4 ∧ -4 ≤ P.2 ∧ P.2 ≤ 2 ∧ 
    (let d := abs (2 * P.1 + P.2 - 4) / (sqrt 5) in
     d = 9 / 10 * sqrt 5 ∧ 
     ∀ (Q : ℝ × ℝ) (hq : Q ∈ parabola), 0 ≤ Q.1 ∧ Q.1 ≤ 4 ∧ -4 ≤ Q.2 ∧ Q.2 ≤ 2 → 
     abs (2 * Q.1 + Q.2 - 4) / (sqrt 5) ≤ d) ∧
    (1 / 2 * 3 * sqrt 5 * abs (2 * P.1 + P.2 - 4) / (sqrt 5) = 27 / 4)) :=
sorry

end max_area_triangle_l235_235781


namespace nonagon_diagonals_count_l235_235916

-- Defining a convex nonagon
structure Nonagon :=
  (vertices : Fin 9) -- Each vertex is represented by an element of Fin 9

-- Hypothesize a diagonal counting function
def diagonal_count (nonagon : Nonagon) : Nat :=
  9 * 6 / 2

-- Theorem stating the number of distinct diagonals in a convex nonagon
theorem nonagon_diagonals_count (n : Nonagon) : diagonal_count n = 27 :=
by
  -- skipping the proof
  sorry

end nonagon_diagonals_count_l235_235916


namespace binom_60_3_l235_235018

theorem binom_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_l235_235018


namespace leo_weight_l235_235667

theorem leo_weight (L K : ℕ) (h1 : L + 10 = 1.5 * K) (h2 : L + K = 170) : L = 98 :=
sorry

end leo_weight_l235_235667


namespace binom_60_3_eq_34220_l235_235047

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l235_235047


namespace probability_rain_all_days_l235_235627

noncomputable def probability_rain_each_day := 
  ((2 / 5) : ℚ, (1 / 2) : ℚ, (3 / 10) : ℚ)

theorem probability_rain_all_days (p_fri p_sat p_sun : ℚ) (h_fri : p_fri = 2 / 5) (h_sat : p_sat = 1 / 2) (h_sun : p_sun = 3 / 10) :
  (p_fri * p_sat * p_sun * 100 : ℚ) = 6 :=
by 
  calc
    (p_fri * p_sat * p_sun * 100 : ℚ) 
        = (2 / 5 * 1 / 2 * 3 / 10 * 100 : ℚ) : by rw [h_fri, h_sat, h_sun]
    ... = 6 : by norm_num

end probability_rain_all_days_l235_235627


namespace max_value_of_xs_l235_235998

theorem max_value_of_xs (n : ℕ) (h : n ≥ 2) (x : ℕ → ℕ) (h1 : (∑ i in finset.range n, x i) = (∏ i in finset.range n, x i)) :
  finset.max' (finset.range n) (λ i : ℕ, x i) = n := 
sorry

end max_value_of_xs_l235_235998


namespace problem_statement_l235_235423

noncomputable theory

def complex_norm (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem problem_statement (a b : ℝ) (i : ℂ) (h : i = complex.I) (h1 : a / (1 - i) = 1 - b * i) : 
  complex_norm a b = real.sqrt 5 :=
by sorry

end problem_statement_l235_235423


namespace binomial_60_3_l235_235024

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l235_235024


namespace solve_inequality_l235_235795

theorem solve_inequality (a x : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 1 > 0) : 
  (-2 < a ∧ a < 1 → a < x ∧ x < 2 - a) ∧ 
  (a = 1 → False) ∧ 
  (1 < a ∧ a < 2 → 2 - a < x ∧ x < a) :=
by
  sorry

end solve_inequality_l235_235795


namespace number_of_valid_sequences_l235_235489

def isValidSequence (seq : List Bool) : Prop :=
(seq.length = 20) ∧
(seq.head = some false) ∧
(seq.getLast! = some false) ∧
(∀ (n : ℕ), n < 19 → seq.get! n = false → seq.get! (n + 1) = true) ∧
(∀ (i : ℕ), i + 3 < 20 → seq.get! i = true → seq.get! (i + 1) = true → seq.get! (i + 2) = true → seq.get! (i + 3) = true → False)

theorem number_of_valid_sequences : 
  (Finset.filter isValidSequence (Finset.univ : Finset (Vector Bool 20))).card = 86 :=
sorry

end number_of_valid_sequences_l235_235489


namespace count_two_digit_numbers_with_digit_four_l235_235491

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def has_digit_four (n : ℕ) : Prop :=
  (n / 10 = 4) ∨ (n % 10 = 4)

theorem count_two_digit_numbers_with_digit_four :
  ([(n : ℕ) | is_two_digit n ∧ has_digit_four n]).card = 18 :=
by
  sorry

end count_two_digit_numbers_with_digit_four_l235_235491


namespace measure_angle_B_l235_235509

noncomputable def find_angle_B (A B C : ℝ) (sin_A sin_B sin_C : ℝ) : Prop :=
sin_B ^ 2 = sin_A ^ 2 + sqrt 3 * sin_A * sin_C + sin_C ^ 2

theorem measure_angle_B (A B C : ℝ) (sin_A sin_B sin_C : ℝ) (h : find_angle_B A B C sin_A sin_B sin_C) :
  B = 5 * Real.pi / 6 :=
sorry

end measure_angle_B_l235_235509


namespace intersection_A_B_is_correct_l235_235449

open Set

def SetA : Set ℝ := { x | x ≤ 1 }
def SetB : Set ℝ := { -2, -1, 0, 1, 2 }

theorem intersection_A_B_is_correct : SetA ∩ SetB = { -2, -1, 0, 1 } :=
by
  sorry

end intersection_A_B_is_correct_l235_235449


namespace nonagon_diagonals_count_l235_235892

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235892


namespace circumcenter_on_angle_bisector_l235_235691

-- Definitions based on conditions
variable (O A B C : Point)
variable (circle : Circle O)
variable (angle : Angle)

-- Given conditions
def reflection (O A : Point) (line : Line) : Prop := 
  isReflectionAcross line O A

def tangents (A : Point) (circle : Circle) (B C : Point) : Prop :=
  isTangent A circle B ∧ isTangent A circle C

-- The proof problem
theorem circumcenter_on_angle_bisector
  (O A B C : Point)
  (circle : Circle O)
  (angle : Angle)
  (h1 : angle.hasInscribedCircle circle)
  (h2 : reflection O A angle.side1)
  (h3 : tangents A circle B C) : 
  liesOn (circumcenter ⟨A, B, C⟩) (angle.bisector) := 
sorry

end circumcenter_on_angle_bisector_l235_235691


namespace OQ_length_l235_235547

variables {A B C P Q O : Type}
variables [geometry A B C P Q O]

-- Definitions based on the conditions
def is_median (A C Q : Type) := midpoint A Q = midpoint C Q
def is_centroid (O A B C : Type) := intersection_point (median A B C) = O

axiom CO_eq_4 : length (segment C O) = 4
axiom CQ_is_median : is_median C A Q
axiom O_is_centroid : is_centroid O A B C

-- Proof problem: Show that the length of OQ is 2 inches based on the given conditions
theorem OQ_length :
  length (segment O Q) = 2 :=
sorry

end OQ_length_l235_235547


namespace range_of_a_l235_235621

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≤ 1 → f x ≤ f 1) : 1 ≤ a :=
by
  let f := λ x : ℝ, x^2 - 2 * a * x + 2
  sorry

end range_of_a_l235_235621


namespace smallest_D_for_trig_inequality_l235_235752

theorem smallest_D_for_trig_inequality :
  (∀ θ : ℝ, 1 + 1 ≥ (sqrt (2 : ℝ)) * (Real.sin θ + Real.cos θ)) →
  ∀ D : ℝ, (∀ θ : ℝ, 1 + 1 ≥ D * (Real.sin θ + Real.cos θ)) → D ≥ sqrt (2 : ℝ) := 
by
  -- skipped proof
  sorry

end smallest_D_for_trig_inequality_l235_235752


namespace solve_equation_l235_235258

theorem solve_equation (n : ℝ) :
  (3 - 2 * n) / (n + 2) + (3 * n - 9) / (3 - 2 * n) = 2 ↔ 
  n = (25 + Real.sqrt 13) / 18 ∨ n = (25 - Real.sqrt 13) / 18 :=
by
  sorry

end solve_equation_l235_235258


namespace floor_mod_eq_zero_l235_235317

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_mod_eq_zero : (floor (5^2017015 / (5^2015 + 7)) % 1000 = 0) :=
sorry

end floor_mod_eq_zero_l235_235317


namespace fifth_employee_selected_is_23_l235_235599

/-- 
Given 45 employees and a method to select 5 employees using a provided random number table,
starting from the 5th column and 6th digit in the first row, and selecting sequential numbers
within the range 1 to 45 while skipping duplicates and numbers exceeding 45,
prove that the 5th employee selected has the number 23.
-/
theorem fifth_employee_selected_is_23 :
  let employee_numbers := [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43,
                           84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25] in
  -- Given condition to start at 5th and 6th digit of the first row, the selected numbers:
  let start_index := 4 in -- Note: 0-based index implies 5th element is index 4
  let selected_numbers := [39, 43, 17, 37, 23] in
  selected_numbers.length = 5 ∧
  selected_numbers.nth 4 = some 23 := 
  -- Proof omitted
  sorry

end fifth_employee_selected_is_23_l235_235599


namespace minimize_distance_to_median_point_l235_235445

theorem minimize_distance_to_median_point :
  ∀ (P₁ P₂ P₃ P₄ P₅ P₆ P₇ P₈ P₉ P : ℝ),
  (s = (abs (P - P₁)) + (abs (P - P₂)) + (abs (P - P₃)) + (abs (P - P₄)) + 
        (abs (P - P₅)) + (abs (P - P₆)) + (abs (P - P₇)) + (abs (P - P₈)) + (abs (P - P₉))) →
  (P₁ ≤ P₂ ∧ P₂ ≤ P₃ ∧ P₃ ≤ P₄ ∧ P₄ ≤ P₅ ∧ P₅ ≤ P₆ ∧ P₆ ≤ P₇ ∧ P₇ ≤ P₈ ∧ P₈ ≤ P₉) →
  minimizes (s) when (P = P₅).

end minimize_distance_to_median_point_l235_235445


namespace number_of_correct_statements_l235_235058

open Real

variables {f : ℝ → ℝ} {a b : ℝ} (h1 : a < b) (h2 : continuous f) (h3 : ∀ x, continuous (deriv f x))
  (h4 : deriv f a > 0) (h5 : deriv f b < 0)

theorem number_of_correct_statements : 2 =
          (if ∃ x ∈ set.Icc a b, f x = 0 then 1 else 0) +
          (if ∃ x ∈ set.Icc a b, f x > f b then 1 else 0) +
          (if ∀ x ∈ set.Icc a b, f x ≥ f a then 1 else 0) +
          (if ∃ x ∈ set.Icc a b, f a - f b > deriv f x * (a - b) then 1 else 0) := sorry

end number_of_correct_statements_l235_235058


namespace floor_inequality_l235_235605

def floor (x : ℝ) : ℤ := x.to_int_floor

noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

theorem floor_inequality (x : ℝ) :
  floor (5 * x) ≥ floor x + (floor (2 * x) / 2) + (floor (3 * x) / 3) + (floor (4 * x) / 4) + (floor (5 * x) / 5) := 
  sorry

end floor_inequality_l235_235605


namespace min_students_same_score_l235_235187

open Nat

theorem min_students_same_score :
  ∃ s : ℕ, ∃ (H : s ∈ (Finset.range 25).image (λ n, 6 + 4 * n - 1 * (6 - n))),
    (Finset.filter (λ student_score, student_score = s) (Finset.range 51).image (λ student, 
    6 + 4 * (student % 7) - 1 * ((6 - student % 7)))) .card ≥ 3 := by
  sorry

end min_students_same_score_l235_235187


namespace number_of_diagonals_in_nonagon_l235_235825

theorem number_of_diagonals_in_nonagon : 
  ∀ (n : ℕ) (h : n = 9), 
  (n * (n - 3)) / 2 = 27 :=
begin
  intro n,
  intro h,
  rw h,
  norm_num,
end

end number_of_diagonals_in_nonagon_l235_235825


namespace min_y_squared_isosceles_trapezoid_l235_235983

theorem min_y_squared_isosceles_trapezoid:
  ∀ (EF GH y : ℝ) (circle_center : ℝ)
    (isosceles_trapezoid : Prop)
    (tangent_EH : Prop)
    (tangent_FG : Prop),
  isosceles_trapezoid ∧ EF = 72 ∧ GH = 45 ∧ EH = y ∧ FG = y ∧
  (∃ (circle : ℝ), circle_center = (EF / 2) ∧ tangent_EH ∧ tangent_FG)
  → y^2 = 486 :=
by sorry

end min_y_squared_isosceles_trapezoid_l235_235983


namespace binomial_60_3_eq_34220_l235_235040

open Nat

theorem binomial_60_3_eq_34220 : choose 60 3 = 34220 := 
by
  sorry

end binomial_60_3_eq_34220_l235_235040


namespace compute_bc_over_ad_l235_235546

noncomputable def volume_rectangular_prism : ℝ := 30
noncomputable def surface_area_rectangular_prism : ℝ := 62
noncomputable def volume_quarter_cylinder_contrib : ℝ := 10 * π
noncomputable def volume_eighth_sphere_contrib : ℝ := 4 * π / 3

theorem compute_bc_over_ad : (volume_quarter_cylinder_contrib * surface_area_rectangular_prism) / (volume_eighth_sphere_contrib * volume_rectangular_prism) = 15.5 :=
by
  sorry

end compute_bc_over_ad_l235_235546


namespace max_area_equilateral_triangle_in_rectangle_proof_l235_235290

-- Define the problem conditions
noncomputable def rectangle_EFGH := (12 : ℝ, 13 : ℝ)

-- Define the function to calculate the maximum area of an equilateral triangle inside the rectangle
noncomputable def max_area_equilateral_triangle_in_rectangle (width height : ℝ) : ℝ :=
  205 * Real.sqrt 3 - 468

-- The theorem states that, given the dimensions of the rectangle, the maximum possible area is as calculated
theorem max_area_equilateral_triangle_in_rectangle_proof :
  max_area_equilateral_triangle_in_rectangle 12 13 = 205 * Real.sqrt 3 - 468 :=
by
  sorry

end max_area_equilateral_triangle_in_rectangle_proof_l235_235290


namespace intersection_point_solution_l235_235606

open Real

noncomputable def solution_point : ℝ × ℝ :=
  (155 / 67, 5 / 67)

theorem intersection_point_solution :
  ∃ (x y : ℝ), (11 * x - 5 * y = 40) ∧ (9 * x + 2 * y = 15) ∧ (x = 155 / 67) ∧ (y = 5 / 67) := 
by 
  use 155 / 67
  use 5 / 67
  split
  { norm_num }
  split
  { norm_num }
  split
  { reflexivity }
  { reflexivity }

end intersection_point_solution_l235_235606


namespace propositions_l235_235992

variable (L M : Type) [Line L] [Line M]
variable (P Q R : Type) [Plane P] [Plane Q] [Plane R]

-- Definitions for parallel and perpendicular planes
def isParallel (α β : Plane) : Prop := sorry  -- Define parallelism for planes
def isPerpendicular (α β : Plane) : Prop := sorry  -- Define perpendicularity for planes

-- Definitions for parallel and perpendicular lines to planes
def lineParallelToPlane (m : Line) (α : Plane) : Prop := sorry  -- Define parallelism of a line to a plane
def linePerpendicularToPlane (m : Line) (α : Plane) : Prop := sorry  -- Define perpendicularity of a line to a plane

theorem propositions (m n : Line) (α β γ : Plane) :
  ((isParallel α β) ∧ (isParallel α γ) → (isParallel γ β)) ∧
  ¬ ((isPerpendicular α β) ∧ (lineParallelToPlane m α) → (linePerpendicularToPlane m β)) ∧
  ((linePerpendicularToPlane m α) ∧ (lineParallelToPlane m β) → (isPerpendicular α β)) ∧
  ¬ ((lineParallelToPlane m n) ∧ (lineParallelToPlane n α) → (linePerpendicularToPlane m α)) := sorry

end propositions_l235_235992


namespace smaller_number_l235_235174

theorem smaller_number {a b : ℕ} (h_ratio : b = 5 * a / 2) (h_lcm : Nat.lcm a b = 160) : a = 64 := 
by
  sorry

end smaller_number_l235_235174


namespace systematic_sampling_seventh_group_l235_235947

theorem systematic_sampling_seventh_group :
  ∀ (num_students total_selected : ℕ) (student_num : ℕ),
    num_students = 50 →
    total_selected = 10 →
    student_num = 46 →
    let group_size := num_students / total_selected in
    let seventh_group_number := student_num - (3 * group_size) in
    seventh_group_number = 31 :=
begin
  intros,
  sorry
end

end systematic_sampling_seventh_group_l235_235947


namespace nonagon_diagonals_count_l235_235894

/-
Given: A convex nonagon, define the number of distinct diagonals
to connect non-adjacent vertices.
Prove: The number of distinct diagonals is 27.
-/

theorem nonagon_diagonals_count :
  ∀ (n : ℕ), n = 9 → ∃ d : ℕ, d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  intro n
  intro h
  use (n * (n - 3)) / 2
  rw [h]
  split
  . refl
  . exact 27
  sorry

end nonagon_diagonals_count_l235_235894


namespace species_below_threshold_in_year_2019_l235_235755

-- Definitions based on conditions in the problem.
def initial_species (N : ℝ) : ℝ := N
def yearly_decay_rate : ℝ := 0.70
def threshold : ℝ := 0.05

-- The problem statement to prove.
theorem species_below_threshold_in_year_2019 (N : ℝ) (hN : N > 0):
  ∃ k : ℕ, k ≥ 9 ∧ yearly_decay_rate ^ k * initial_species N < threshold * initial_species N :=
sorry

end species_below_threshold_in_year_2019_l235_235755


namespace sum_of_integers_l235_235767

theorem sum_of_integers (n : ℕ) (h1 : 1.5 * n - 6.3 < 7.5) : 
  ∑ k in Finset.filter (λ k, 1.5 * k - 6.3 < 7.5) (Finset.range 10) = 45 :=
sorry

end sum_of_integers_l235_235767


namespace prove_dneq1_prove_aseq_prove_frac_prove_sum_l235_235371

variable {a b d : ℕ} 

-- Define the arithmetic sequence condition
def arithmetic_seq (aₙ : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, aₙ (n + 1) = aₙ n + d

-- Define specific conditions
def condition_1 (aₙ : ℕ → ℕ) := aₙ 2 = 5
def condition_2 (aₙ : ℕ → ℕ) := aₙ 6 + aₙ 8 = 30

-- Define the specific sequence found from conditions
def defined_seq (n : ℕ) := 2 * n + 1

-- Statements to prove
theorem prove_dneq1 (aₙ : ℕ → ℕ) (h1 : condition_1 aₙ) (h2 : condition_2 aₙ) : 
  (∀ d, arithmetic_seq aₙ d → d ≠ 1) := 
sorry

theorem prove_aseq (aₙ : ℕ → ℕ) (h1 : condition_1 aₙ) (h2 : condition_2 aₙ) : 
  aₙ = defined_seq := 
sorry

theorem prove_frac (aₙ : ℕ → ℕ) (h : aₙ = defined_seq) (n : ℕ) : 
  1 / (aₙ n ^ 2 - 1) = 1 / 4 * (1 / n - 1 / (n + 1)) := 
sorry

theorem prove_sum (aₙ : ℕ → ℕ) (h : aₙ = defined_seq) (n : ℕ) : 
  (∑ k in range n, 1 / (aₙ k ^ 2 - 1)) ≠ n / (4 * n + 1) := 
sorry


end prove_dneq1_prove_aseq_prove_frac_prove_sum_l235_235371


namespace total_cost_l235_235704

def copper_pipe_length := 10
def plastic_pipe_length := 15
def copper_pipe_cost_per_meter := 5
def plastic_pipe_cost_per_meter := 3

theorem total_cost (h₁ : copper_pipe_length = 10)
                   (h₂ : plastic_pipe_length = 15)
                   (h₃ : copper_pipe_cost_per_meter = 5)
                   (h₄ : plastic_pipe_cost_per_meter = 3) :
  10 * 5 + 15 * 3 = 95 :=
by sorry

end total_cost_l235_235704


namespace max_sum_of_positive_integers_with_product_144_l235_235286

theorem max_sum_of_positive_integers_with_product_144 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 144 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 75 := 
by
  sorry

end max_sum_of_positive_integers_with_product_144_l235_235286


namespace correct_angle_between_a_and_b_l235_235484

-- Given vectors
def a : ℝ × ℝ := (-4, 3)
def b : ℝ × ℝ := (7, 1)

-- Define the dot product function for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the magnitude function for 2D vectors
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)

-- Define the angle calculation function between two 2D vectors
def angle_between (u v : ℝ × ℝ) : ℝ := Real.arccos (dot_product u v / (magnitude u * magnitude v))

-- The theorem we need to prove
theorem correct_angle_between_a_and_b (a b : ℝ × ℝ) : angle_between a b = 3 * Real.pi / 4 := by
  sorry

end correct_angle_between_a_and_b_l235_235484


namespace convert_119_to_binary_l235_235055

theorem convert_119_to_binary :
  Nat.toDigits 2 119 = [1, 1, 1, 0, 1, 1, 1] :=
by
  sorry

end convert_119_to_binary_l235_235055


namespace distinct_diagonals_in_convex_nonagon_l235_235885

-- Definitions
def is_convex_nonagon (n : ℕ) : Prop := n = 9

-- Main theorem stating the number of distinct diagonals in a convex nonagon
theorem distinct_diagonals_in_convex_nonagon (n : ℕ) (h : is_convex_nonagon n) : 
  ∃ d : ℕ, d = 27 :=
by
  -- Use Lean constructs to formalize the proof
  sorry

end distinct_diagonals_in_convex_nonagon_l235_235885


namespace trajectory_of_P_l235_235938

theorem trajectory_of_P (a b : ℝ) (h : real.sqrt(a^2 + b^2) ≤ 1) : a^2 + b^2 ≤ 1 ∧ a^2 + b^2 ≤ 1 := by
  sorry

end trajectory_of_P_l235_235938


namespace area_of_region_bounded_by_lines_and_y_axis_l235_235319

noncomputable def area_of_triangle_bounded_by_lines : ℝ :=
  let y1 (x : ℝ) := 3 * x - 6
  let y2 (x : ℝ) := -2 * x + 18
  let intersection_x := 24 / 5
  let intersection_y := y1 intersection_x
  let base := 18 + 6
  let height := intersection_x
  1 / 2 * base * height

theorem area_of_region_bounded_by_lines_and_y_axis :
  area_of_triangle_bounded_by_lines = 57.6 :=
by
  sorry

end area_of_region_bounded_by_lines_and_y_axis_l235_235319
