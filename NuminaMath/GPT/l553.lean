import Mathlib

namespace toby_sharing_proof_l553_553864

theorem toby_sharing_proof (initial_amt amount_left num_brothers : ℕ) 
(h_init : initial_amt = 343)
(h_left : amount_left = 245)
(h_bros : num_brothers = 2) : 
(initial_amt - amount_left) / (initial_amt * num_brothers) = 1 / 7 := 
sorry

end toby_sharing_proof_l553_553864


namespace lcm_18_35_l553_553999

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end lcm_18_35_l553_553999


namespace james_monthly_earnings_l553_553377

theorem james_monthly_earnings (initial_subscribers gifted_subscribers earnings_per_subscriber : ℕ)
  (initial_subscribers_eq : initial_subscribers = 150)
  (gifted_subscribers_eq : gifted_subscribers = 50)
  (earnings_per_subscriber_eq : earnings_per_subscriber = 9) :
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber = 1800 := by
  sorry

end james_monthly_earnings_l553_553377


namespace num_arithmetic_sequences_l553_553268

theorem num_arithmetic_sequences (a d : ℕ) (n : ℕ) (h1 : n >= 3) (h2 : n * (2 * a + (n - 1) * d) = 2 * 97^2) :
  ∃ seqs : ℕ, seqs = 4 :=
by sorry

end num_arithmetic_sequences_l553_553268


namespace parallel_TD_AC_l553_553111

-- Define the conditions of the problem

variables {A B C T D : Type} [IncidenceStructure ABC]
-- Assume that there is an IncidenceStructure defined on the type ABC

-- Assume that A and B are on circumcircle of an acute triangle ABC
variable (circumcircle : Incircle ABC)

-- Assume that tangents at A and B intersect at T
axiom tangents_intersect (tangentA tangentB : Line) (h_tangentA: tangents circumcircle A tangentA) 
(h_tangentB: tangents circumcircle B tangentB) : intersects tangentA tangentB T

-- Assume D is on line BC such that DA = DC
axiom isosceles_DA_DC (lineBC : Line) [lies_on D lineBC] (h_D_point : lies_on D A ∪ B ∪ C) :
DA = DC

-- Lean 4 statement for the problem
theorem parallel_TD_AC : TD ∥ AC :=
sorry 

end parallel_TD_AC_l553_553111


namespace decreasing_interval_of_f_even_l553_553908

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k-2)*x^2 + (k-1)*x + 3

def is_even_function (k : ℝ) : Prop := 
  ∀ x : ℝ, f(k, -x) = f(k, x)

theorem decreasing_interval_of_f_even (k : ℝ) (h : is_even_function k) : 
  (0 : Set ℝ, Set.Ioi 0) = {x : ℝ | ∀ y : ℝ, f(1, y) > f(1, y + x)} :=
sorry

end decreasing_interval_of_f_even_l553_553908


namespace stephanie_oranges_l553_553076

theorem stephanie_oranges (times_at_store : ℕ) (oranges_per_time : ℕ) (total_oranges : ℕ) 
  (h1 : times_at_store = 8) (h2 : oranges_per_time = 2) :
  total_oranges = 16 :=
by
  sorry

end stephanie_oranges_l553_553076


namespace smallest_n_for_T_n_integer_l553_553003

noncomputable def K : ℚ := ∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n - 1)) * K + 1

theorem smallest_n_for_T_n_integer : ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m : ℕ, T_n m ∈ ℤ → n ≤ m :=
  ⟨504, sorry⟩

end smallest_n_for_T_n_integer_l553_553003


namespace inequality_solution_l553_553058

theorem inequality_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * sqrt b + b * sqrt a :=
by
  sorry

end inequality_solution_l553_553058


namespace exponent_on_right_side_l553_553334

theorem exponent_on_right_side (n : ℕ) (h : n = 17) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 :=
by
  sorry

end exponent_on_right_side_l553_553334


namespace ratio_of_segments_l553_553207

variables (A B C D E F G M : Type) [InCircle A B C D] [Intersection E A B C D] 
  [PointOnSegment M E B] (t : ℝ) (h_t : t = AM / AB)

theorem ratio_of_segments {EF EG : ℝ} (h_EF : EF = E - F) (h_EG : EG = E - G) :
  EF / EG = t / (1 - t) :=
sorry

end ratio_of_segments_l553_553207


namespace onewaynia_road_closure_l553_553755

variable {V : Type} -- Denoting the type of cities
variable (G : V → V → Prop) -- G represents the directed graph

-- Conditions
variables (outdegree : V → Nat) (indegree : V → Nat)
variables (two_ways : ∀ (u v : V), u ≠ v → ¬(G u v ∧ G v u))
variables (two_out : ∀ v : V, outdegree v = 2)
variables (two_in : ∀ v : V, indegree v = 2)

theorem onewaynia_road_closure:
  ∃ n : Nat, n ≥ 1 ∧ (number_of_closures : Nat) = 2 ^ n :=
by
  sorry

end onewaynia_road_closure_l553_553755


namespace problem_statement_l553_553899

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1/a) + Real.sqrt (b + 1/b) + Real.sqrt (c + 1/c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
sorry

end problem_statement_l553_553899


namespace valid_situation_exists_l553_553647

variable (V : Type) [Fintype V] [DecidableEq V] -- V is the set of vertices (beads), finite and decidable equality
variable (E : V → V → Bool) -- E is the adjacency relation (elastic bands)
variable (height : V → ℕ) -- height of each bead
variable [DecidablePred (λ v, 1 ≤ height v ∧ height v ≤ 2017)] -- height is between 1 and 2017

-- Conditions given in the problem:
-- 1. Each stick has a bead, heights are in integer intervals [1, 2017]
-- 2. Adjacency matrix E defines connections via elastic bands
-- 3. Young ant can traverse any elastic band.
-- 4. If ∃ (u v ∈ V), height(u) ≠ height(v), i.e., at least one valid situation

def valid_situation := ∃ (u v : V), E u v ∧ height u ≠ height v

theorem valid_situation_exists 
  (h : valid_situation V E height) :
  ∃ height' : V → ℕ, 
    (∀ u v, E u v → |height' u - height' v| ≤ 1) ∧ 
    (∀ w, ∃ path : List V, path.head = w ∧ path.last ∈ (path.nth 1) ∧ ∀ (i : ℕ), i < (path.length - 1) → E (path.nth i) (path.nth (i+1))) :=
sorry

end valid_situation_exists_l553_553647


namespace minimum_shaded_triangles_l553_553196

noncomputable def number_of_shaded_triangles : ℕ :=
  15

theorem minimum_shaded_triangles (n : ℕ) (h : n = 15) :
  ∀ (triangle : Type) (side_length_8 : ℕ) (side_length_1 : ℕ),
  triangle = equilateral_triangle side_length_8
  → number_of_divisions triangle = (side_length_8 / side_length_1) ^ 2
  → n = minimum_shaded side_length_8 side_length_1 := 
  sorry

end minimum_shaded_triangles_l553_553196


namespace tulip_count_l553_553115

theorem tulip_count (T : ℕ) (roses tulips daisies : ℕ)
  (h1 : roses = 25)
  (h2 : daisies = 35)
  (h3 : 75 * T / 100 = T - roses):
  tulips = T - roses - daisies :=
begin
  -- substituting the known values and conditions, solving for tulips is straightforward
  -- sorry is used here to skip the actual steps of the proof.
  sorry
end

end tulip_count_l553_553115


namespace g_is_odd_l553_553766

def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1 / 3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intros x
  sorry

end g_is_odd_l553_553766


namespace exponential_decreasing_range_l553_553247

theorem exponential_decreasing_range {a : ℝ} :
  (∀ x y : ℝ, x < y → (a - 2) ^ x > (a - 2) ^ y) ↔ 2 < a ∧ a < 3 :=
begin
  split,
  { intros h,
    have ha : a - 2 > 0, from sorry, -- Assume intermediate steps
    have hab : a - 2 < 1, from sorry, -- Assume intermediate steps
    split; linarith, },
  { rintros ⟨ha, hab⟩ x y hxy,
    calc (a - 2) ^ x > (a - 2) ^ y : sorry, -- Assume intermediate steps
  },
end

end exponential_decreasing_range_l553_553247


namespace sum_of_adjacent_to_15_l553_553486

/-- Given the set of positive divisors of 360 arranged in a circle, and the requirement that each pair of adjacent divisors has a common factor greater than 1, we need to prove that the sum of the two integers adjacent to 15 in this arrangement is 75. --/
theorem sum_of_adjacent_to_15 (divisors : List ℕ) (h : divisors = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360])
  (adjacent_property : ∀ i, (divisors.nth i).get_or_else 1 ≠ 1 → (gcd ((divisors.nth i).get_or_else 1) ((divisors.nth (i + 1) % divisors.length).get_or_else 1) > 1)) :
  (divisors.nth (divisors.indexOf 15 + 1) % divisors.length).get_or_else 0 + 
  (divisors.nth (divisors.indexOf 15 - 1 + divisors.length) % divisors.length).get_or_else 0 = 75 := sorry

end sum_of_adjacent_to_15_l553_553486


namespace moe_mowing_time_l553_553420

-- Define the conditions given in the problem
def lawn_length : ℝ := 120 -- feet
def lawn_width : ℝ := 180 -- feet
def mower_swath_inches : ℝ := 30 -- inches
def mower_overlap_inches : ℝ := 6 -- inches
def mowing_speed : ℝ := 4000 -- feet per hour

-- Convert dimensions from inches to feet
def swath_width_feet : ℝ := (mower_swath_inches - mower_overlap_inches) / 12 -- 12 inches in a foot

-- Calculate the effective swath after overlapping
def effective_swath_width : ℝ := swath_width_feet

-- Calculate the number of strips needed and total distance
def number_of_strips : ℝ := lawn_width / effective_swath_width
def total_distance_mowed : ℝ := number_of_strips * lawn_length

-- Calculate the time required to mow the lawn
def hours_needed : ℝ := total_distance_mowed / mowing_speed

-- The main theorem to prove
theorem moe_mowing_time : hours_needed = 2.7 := by
  sorry

end moe_mowing_time_l553_553420


namespace rooks_arrangement_count_l553_553910

theorem rooks_arrangement_count :
  let rooks := 4
  let board_size := 4
  let total_arrangements := (Finset.perm (Finset.range board_size)).card
  let invalid_arrangements := 15
  (total_arrangements - invalid_arrangements) = 9 :=
by
  sorry

end rooks_arrangement_count_l553_553910


namespace sum_of_three_consecutive_even_numbers_l553_553137

theorem sum_of_three_consecutive_even_numbers (a : ℤ) (h : a * (a + 2) * (a + 4) = 960) : a + (a + 2) + (a + 4) = 30 := by
  sorry

end sum_of_three_consecutive_even_numbers_l553_553137


namespace equation1_solution_equation2_solutions_l553_553073

theorem equation1_solution (x : ℝ) : (x - 2) * (x - 3) = x - 2 → (x = 2 ∨ x = 4) :=
by
  intro h
  have h1 : (x - 2) * (x - 3) - (x - 2) = 0 := by sorry
  have h2 : (x - 2) * (x - 4) = 0 := by sorry
  have h3 : x - 2 = 0 ∨ x - 4 = 0 := by sorry
  cases h3 with
  | inl h4 => left; exact eq_of_sub_eq_zero h4
  | inr h5 => right; exact eq_of_sub_eq_zero h5

theorem equation2_solutions (x : ℝ) : 2 * x^2 - 5 * x + 1 = 0 → (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by
  intro h
  have h1 : (-5)^2 - 4 * 2 * 1 = 17 := by sorry
  have h2 : 2 * x^2 - 5 * x + 1 = 2 * ((x - (5 + Real.sqrt 17) / 4) * (x - (5 - Real.sqrt 17) / 4)) := by sorry
  have h3 : (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) := by sorry
  exact h3

end equation1_solution_equation2_solutions_l553_553073


namespace min_ab_bound_l553_553803

theorem min_ab_bound (a b n : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_n : 0 < n) 
                      (h : ∀ i j, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) :
  ∃ c > 0, min a b > c^n * n^(n/2) :=
sorry

end min_ab_bound_l553_553803


namespace find_diameter_C_l553_553607

-- Define variables and radii
variables {rC rD : ℝ}

-- Definition of circles and conditions
def circle_in_circle (rC rD : ℝ) : Prop := rC < rD

def diameter_D (rD : ℝ) : Prop := rD = 10 -- since diameter is 20 cm

def shaded_area_ratio (rC rD : ℝ) : Prop := 
  (π * (rD^2 - rC^2)) / (π * rC^2) = 7

-- The theorem stating the problem
theorem find_diameter_C (hC_in_D : circle_in_circle rC rD)
    (hD : diameter_D rD)
    (h_ratio : shaded_area_ratio rC rD) :
    2 * rC = 5 * Real.sqrt 5 := 
  sorry

end find_diameter_C_l553_553607


namespace sum_of_midpoint_xcoords_l553_553493

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l553_553493


namespace rebecca_tent_stakes_l553_553065

variables (T D W : ℕ)

-- Conditions
def drink_mix_eq : Prop := D = 3 * T
def water_eq : Prop := W = T + 2
def total_items_eq : Prop := T + D + W = 22

-- Problem statement
theorem rebecca_tent_stakes 
  (h1 : drink_mix_eq T D)
  (h2 : water_eq T W)
  (h3 : total_items_eq T D W) : 
  T = 4 := 
sorry

end rebecca_tent_stakes_l553_553065


namespace monotonicity_and_distinct_positives_l553_553699

noncomputable def f (x a : ℝ) := log x + x^2 + (a + 2) * x
noncomputable def f_deriv (x a : ℝ) := (2 * x + a) * (x + 1) / x

theorem monotonicity_and_distinct_positives
    (a : ℝ)
    (h_a : a < 0)
    (x1 x2 : ℝ)
    (h_x1 : 0 < x1)
    (h_x2 : 0 < x2)
    (h_distinct : x1 ≠ x2)
    (h_feq : f x1 a = f x2 a) :
    f_deriv ((x1 + x2) / 2) a > 0 := sorry

end monotonicity_and_distinct_positives_l553_553699


namespace probability_john_david_chosen_l553_553772

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem probability_john_david_chosen :
  let total_workers := 6
  let choose_two := choose total_workers 2
  let favorable_outcomes := 1
  choose_two = 15 → (favorable_outcomes / choose_two : ℝ) = 1 / 15 :=
by
  intros
  sorry

end probability_john_david_chosen_l553_553772


namespace fraction_12_equiv_fraction_32_equiv_percentage_equiv_chinese_percentage_equiv_l553_553555
namespace ProofProblem

theorem fraction_12_equiv (n : ℕ) : (n : ℝ) / 12 = 0.25 ↔ n = 3 :=
by
  sorry

theorem fraction_32_equiv (m : ℕ) : 0.25 * 32 = m ↔ m = 8 :=
by
  sorry

theorem percentage_equiv (p : ℕ) : 0.25 * 100 = p ↔ p = 25 :=
by
  sorry

theorem chinese_percentage_equiv (s : String) : (0.25 : ℝ) = 0.25 ↔ s = "二五折" :=
by
  sorry

end ProofProblem

end fraction_12_equiv_fraction_32_equiv_percentage_equiv_chinese_percentage_equiv_l553_553555


namespace randy_current_age_l553_553806

-- Variables and definitions based on conditions
def expert_hours : ℕ := 10000
def practice_hours_per_day : ℕ := 5
def practice_days_per_week : ℕ := 5
def weeks_in_a_year : ℕ := 52
def vacation_weeks_per_year : ℕ := 2
def practice_weeks_per_year : ℕ := weeks_in_a_year - vacation_weeks_per_year
def practice_hours_per_year : ℕ := practice_hours_per_day * practice_days_per_week * practice_weeks_per_year
def max_age_to_become_expert : ℕ := 20

-- Lean statement to prove Randy's current age
theorem randy_current_age (current_years : ℕ) : current_years = 12 :=
by
  have total_hours : ℕ := expert_hours
  have hours_per_year : ℕ := practice_hours_per_year
  have years_to_expert : ℕ := total_hours / hours_per_year
  have randy_age : ℕ := max_age_to_become_expert - years_to_expert
  exact congr_arg _ (eq_symm (by simp [randy_age]))
  sorry -- Proof steps are not required, so we add sorry to skip the proof.

end randy_current_age_l553_553806


namespace volume_proofs_l553_553566

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def sphere_volume (h : ℝ) : ℝ := (4 / 3) * π * (h / 2)^3

theorem volume_proofs (r h : ℝ) (hcyl : cylinder_volume r h = 72 * π) :
  cone_volume r h = 24 * π ∧ sphere_volume h = 12 * r * π :=
by
  sorry

end volume_proofs_l553_553566


namespace simplify_expression_l553_553812

theorem simplify_expression : 4 * (15 / 5) * (25 / -75) = -4 := by
  have h1: 15 / 5 = 3 := by norm_num
  have h2: 25 / -75 = -1 / 3 := by norm_num
  calc
    4 * (15 / 5) * (25 / -75)
        = 4 * 3 * -1 / 3 := by rw [h1, h2]
    ... = 4 * -1 := by norm_num
    ... = -4 := by norm_num

end simplify_expression_l553_553812


namespace imaginary_part_conjugate_l553_553664

theorem imaginary_part_conjugate (z : ℂ) (h : z = (1 + 3 * Complex.i) / (1 - 2 * Complex.i)) :
  (Complex.conj z).im = -1 :=
by 
  sorry

end imaginary_part_conjugate_l553_553664


namespace distance_between_trees_l553_553541

theorem distance_between_trees (n : ℕ) (L : ℝ) (d : ℝ) (h1 : n = 26) (h2 : L = 700) (h3 : d = L / (n - 1)) : d = 28 :=
sorry

end distance_between_trees_l553_553541


namespace polynomial_product_l553_553534

noncomputable def g (x : ℝ) : ℝ := x^4 + 3*x^3 + 4*x^2 + 2*x + 1

noncomputable def bar_g (x : ℝ) : ℝ := x^4 - 3*x^3 + 4*x^2 - 2*x + 1

noncomputable def g_star (x : ℝ) : ℝ := x^4 - 2*x^3 + 4*x^2 - 3*x + 1

noncomputable def bar_g_star (x : ℝ) : ℝ := x^4 + 2*x^3 + 4*x^2 + 3*x + 1

theorem polynomial_product :
  ∃ (f : ℝ → ℝ) (P1 P2 P3 P4 : ℝ → ℝ) (H : ℝ → ℝ),
    (P1 = g) ∧
    (P2 = bar_g) ∧
    (P3 = g_star) ∧
    (P4 = bar_g_star) ∧
    (H = λ x, x^8 + 4*x^6 + 6*x^4 + 4*x^2 + 1) ∧
    (f = λ x, (P1 x) * (P2 x) * (P3 x) * (P4 x) * (H x)) :=
  sorry

end polynomial_product_l553_553534


namespace ellipse_focal_length_l553_553957

theorem ellipse_focal_length (m : ℝ) :
  (∀ x y : ℝ, (x^2 / m + y^2 / 5 = 1)) ∧
  (∃ c : ℝ, 2 * c = 2) ∧
  (c = 1) ∧
  (foci_on_y_axis : True) --Assumes foci are on y-axis without introducing extra variables
  → m = 4 :=
begin
  sorry
end

end ellipse_focal_length_l553_553957


namespace max_value_of_y_l553_553387

open Real

def y (x : ℝ) : ℝ := x - π - ⌊x / π⌋ - abs (sin x)

theorem max_value_of_y :
  ∀ x : ℝ, y x < 1 := 
by
  sorry

end max_value_of_y_l553_553387


namespace ellipse_domain_l553_553293

theorem ellipse_domain (m : ℝ) :
  (-1 < m ∧ m < 2 ∧ m ≠ 1 / 2) -> 
  ∃ a b : ℝ, (a = 2 - m) ∧ (b = m + 1) ∧ a > 0 ∧ b > 0 ∧ a ≠ b :=
by
  sorry

end ellipse_domain_l553_553293


namespace club_truncator_probability_proof_l553_553208

noncomputable def club_truncator_probability_condition := 
  ∀ (win_prob : ℚ) (lose_prob : ℚ) (tie_prob : ℚ) (total_matches : ℕ), 
    win_prob = 2/5 ∧ lose_prob = 1/5 ∧ tie_prob = 2/5 ∧ total_matches = 8

noncomputable def club_truncator_probability_question := 
  (prob : ℚ) (m n : ℕ),
    club_truncator_probability_condition 2/5 1/5 2/5 8 →
    prob = 144/625 → m + n = 769 → prob.proper_fraction → (prob = 144/625) ∧ (m + n = 769)

theorem club_truncator_probability_proof : 
  club_truncator_probability_question :=
  begin
    intro h,
    sorry
  end

end club_truncator_probability_proof_l553_553208


namespace distance_polar_to_cartesian_l553_553756

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def line_equation_in_cartesian (x y : ℝ) : ℝ :=
  x - Real.sqrt 3 * y + 2

theorem distance_polar_to_cartesian {r θ : ℝ} 
  (h_r: r = 2) (h_θ: θ = Real.pi / 6) :
  let (x, y) := polar_to_cartesian r θ in
  let dist := abs (line_equation_in_cartesian x y) / Real.sqrt 1 in
  dist = 1 := 
by
  -- Import statements and initial definitions required above
  sorry

end distance_polar_to_cartesian_l553_553756


namespace right_triangles_count_l553_553349

def point : Type := ℝ × ℝ

structure rectangle :=
(A B C D : point)
(AB_CD_parallel : A.1 = B.1 ∧ C.1 = D.1) -- AB || CD
(AD_BC_parallel : A.2 = D.2 ∧ B.2 = C.2) -- AD || BC

def segment (p1 p2 : point) : set point :=
{ p | (p.1 - p1.1) * (p2.2 - p1.2) = (p.2 - p1.2) * (p2.1 - p1.1) }

structure divided_rectangle :=
(RS : segment)
(divides_congruently : ∃ R S, segment R S ∧ R.1 = S.1 ∧ R.2 ≠ S.2)

def points := {A R B C S D : point}

theorem right_triangles_count (rect : rectangle) (div_rect: divided_rectangle)
  (points_set : points) : 
  ∃ tr : set (set point), |{T | (T ⊆ points_set) ∧ (right_triangle T)}| = 8 :=
sorry

end right_triangles_count_l553_553349


namespace count_integers_satisfying_sqrt_condition_l553_553847

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l553_553847


namespace range_of_m_l553_553490

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, x > 4 ↔ x > m) : m ≤ 4 :=
by {
  -- here we state the necessary assumptions and conclude the theorem
  -- detailed proof steps are not needed, hence sorry is used to skip the proof
  sorry
}

end range_of_m_l553_553490


namespace sum_of_integers_ending_in_2_between_100_and_600_l553_553606

theorem sum_of_integers_ending_in_2_between_100_and_600 :
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  ∃ S : ℤ, S = n * (a + l) / 2 ∧ S = 17350 := 
by
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  use n * (a + l) / 2
  sorry

end sum_of_integers_ending_in_2_between_100_and_600_l553_553606


namespace salesman_remittance_l553_553177

noncomputable def commission (sales : ℝ) : ℝ :=
  if sales ≤ 5000 then
    0.10 * sales
  else
    0.10 * 5000 + 0.05 * (sales - 5000)

noncomputable def amount_remitted (sales : ℝ) : ℝ :=
  sales - commission sales

theorem salesman_remittance :
  amount_remitted 15885.42 = 14841.149 :=
by
  have h_commission_first := 0.10 * 5000
  have h_remain_sales := 15885.42 - 5000
  have h_commission_remain := 0.05 * h_remain_sales
  have h_total_commission := h_commission_first + h_commission_remain
  have h_sales := 15885.42
  have h_amount_remitted := h_sales - h_total_commission
  norm_num at h_commission_first h_remain_sales h_commission_remain h_total_commission h_amount_remitted
  exact h_amount_remitted

end salesman_remittance_l553_553177


namespace average_weight_decrease_l553_553824

theorem average_weight_decrease (A : ℝ) :
  let original_total_weight := 8 * A in
  let new_total_weight := original_total_weight - 86 + 46 in
  let new_average := new_total_weight / 8 in
  A - new_average = 5 :=
by
  sorry

end average_weight_decrease_l553_553824


namespace max_cos2_sinx_l553_553098

noncomputable def cos2_sinx (x : ℝ) : ℝ := (Real.cos x) ^ 2 - Real.sin x

theorem max_cos2_sinx : ∃ x : ℝ, cos2_sinx x = 5 / 4 := 
by
  existsi (Real.arcsin (-1 / 2))
  rw [cos2_sinx]
  -- We need further steps to complete the proof
  sorry

end max_cos2_sinx_l553_553098


namespace pairs_count_l553_553440

noncomputable def count_pairs (n : ℕ) : ℕ :=
  3^n

theorem pairs_count (A : Finset ℕ) (h : A.card = n) :
  ∃ f : Finset ℕ × Finset ℕ → Finset ℕ, ∀ B C, (B ≠ ∅ ∧ B ⊆ C ∧ C ⊆ A) → (f (B, C)).card = count_pairs n :=
sorry

end pairs_count_l553_553440


namespace line_through_M_intercepts_eq_l553_553829

theorem line_through_M_intercepts_eq (M : ℝ × ℝ) (hM : M = (1, 1)) :
  ∃ a b : ℝ, ((x + y = a ∨ y = x) ∧ (M = (1,1))) :=
begin
  use [2, 1],
  split,
  { left,
    exact a,
    sorry,
  },
end

end line_through_M_intercepts_eq_l553_553829


namespace hours_difference_l553_553467

-- Define appropriate constants and data
constants (total_hours : ℕ)
constants (ratio : List ℕ)
constants (days_per_week : ℕ)
constants (work_hours_per_day : ℕ)
constants (work_days_hardest : List ℕ)
constants (work_days_least : List ℕ)

-- Initial conditions from the problem
axiom total_hours_def : total_hours = 1800
axiom ratio_def : ratio = [3, 4, 5, 6, 7]
axiom days_per_week_def : days_per_week = 5
axiom work_hours_per_day_def : work_hours_per_day = 8
axiom work_days_hardest_def : work_days_hardest = [1, 2, 3]  -- Assume Mon, Tue, Thu (3 days)
axiom work_days_least_def : work_days_least = [2, 3, 5]      -- Assume Tue, Wed, Fri (3 days)

-- Function to calculate the total work hours adjusting for days not worked
def adjust_hours (total_hours : ℕ) (work_days : List ℕ) (days_per_week : ℕ) (work_hours_per_day : ℕ) : ℕ := 
  let weeks := total_hours / (days_per_week * work_hours_per_day)
  let days_worked := weeks * work_days.length
  days_worked * work_hours_per_day

-- Difference in actual working hours between the hardest-working and least-working persons
theorem hours_difference : 
  let x := total_hours / 25 in
  let hard_hours := ratio.get 4 * x in
  let least_hours := ratio.get 0 * x in
  adjust_hours (hard_hours) work_days_hardest days_per_week work_hours_per_day - 
  adjust_hours (least_hours) work_days_least days_per_week work_hours_per_day = 168 :=
by
  sorry

end hours_difference_l553_553467


namespace equation_of_line_l553_553170

theorem equation_of_line (a b : ℝ) :
  (∃ a b : ℝ, (-2 / a) + (2 / b) = 1 ∧ |a * b| = 2) →
  ( ∃ (x y : ℝ), (x + 2 * y - 2 = 0) ∨ (2 * x + y + 2 = 0) ) :=
begin
  intros h,
  obtain ⟨a, b, h₁, h₂⟩ := h,
  sorry
end

end equation_of_line_l553_553170


namespace solve_for_s_l553_553218

/-- The function F(a, b, c) is defined as a * b^c. -/
def F (a b c : ℕ) : ℕ := a * b ^ c

/-- Prove that the positive value of s that satisfies F(s,s,2) = 216 is 6. -/
theorem solve_for_s : ∃ s : ℕ, F(s, s, 2) = 216 ∧ s > 0 := by
  use 6
  have hF : F(6, 6, 2) = 6 * 6 ^ 2 := rfl
  rw [hF]
  norm_num
  exact ⟨rfl, trivial⟩

end solve_for_s_l553_553218


namespace simplify_fraction_l553_553455

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l553_553455


namespace number_of_proper_subsets_l553_553308

def setA : Set (ℝ × ℝ) := { p | ∃ x, p = (x, x^2) }
def setB : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 1 - |x|) }
def set_intersection : Set (ℝ × ℝ) := setA ∩ setB

theorem number_of_proper_subsets (A B : Set (ℝ × ℝ)) (hA : A = setA) (hB : B = setB) :
  Fintype.card (Set (set_intersection)) - 1 = 3 :=
by
  sorry

end number_of_proper_subsets_l553_553308


namespace total_money_shared_l553_553422

-- Conditions
def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

-- Question and proof to be demonstrated
theorem total_money_shared : ken_share + tony_share = 5250 :=
by sorry

end total_money_shared_l553_553422


namespace pencil_length_l553_553575

theorem pencil_length (L : ℝ) 
  (h1 : 1 / 8 * L = b) 
  (h2 : 1 / 2 * (L - 1 / 8 * L) = w) 
  (h3 : (L - 1 / 8 * L - 1 / 2 * (L - 1 / 8 * L)) = 7 / 2) :
  L = 8 :=
sorry

end pencil_length_l553_553575


namespace expression_one_eval_expression_two_eval_l553_553203

-- Problem (1)
theorem expression_one_eval : 
  1 * (-0.1)^0 + 32 * 2^(2/3) + (1/4)^(-1/2) = 3 + 32 * (2^(1/3)) := 
  sorry

-- Problem (2)
theorem expression_two_eval : 
  log 10 500 + log 10 (8/5) - 1/2 * log 10 64 + 50 * (log 10 2 + log 10 5)^2 = 49.9031 := 
  sorry

end expression_one_eval_expression_two_eval_l553_553203


namespace find_y_l553_553661

variables (a b c x y p q r : ℝ)

-- Conditions
axiom log_condition : (log a / p = log b / q) ∧ (log b / q = log c / r) ∧ (log c / r = log x) ∧ (x ≠ 1)
axiom equation_condition : b^2 / (a * c) = x^y

-- Theorem to be proved
theorem find_y : y = 2 * q - p - r :=
sorry

end find_y_l553_553661


namespace find_total_amount_l553_553535

noncomputable def total_amount (a b c : ℕ) : Prop :=
  a = 3 * b ∧ b = c + 25 ∧ b = 134 ∧ a + b + c = 645

theorem find_total_amount : ∃ a b c, total_amount a b c :=
by
  sorry

end find_total_amount_l553_553535


namespace number_of_asymptotes_holes_l553_553616

def f (x : ℝ) : ℝ := (x^2 + 4 * x + 3) / (x^3 - 2 * x^2 - x + 2)

theorem number_of_asymptotes_holes :
  let a := 1     -- number of holes
  let b := 2     -- number of vertical asymptotes
  let c := 1     -- number of horizontal asymptotes
  let d := 0     -- number of oblique asymptotes
  a + 2 * b + 3 * c + 4 * d = 8 :=
by
  -- This is where the proof would go
  sorry

end number_of_asymptotes_holes_l553_553616


namespace rendering_completion_time_l553_553571

theorem rendering_completion_time :
  (∀ (start_time : ℕ) (one_fourth_completion_time : ℕ),
    (start_time = 9) →
    (one_fourth_completion_time = 9 + 2 + 3/4) →
    4 * (one_fourth_completion_time - start_time) = 11 →
    ∃ (completion_time : ℕ), completion_time = 9 + 11) :=
by
  intro start_time one_fourth_completion_time
  intro h1 h2 h3
  use 20 -- 8:00 PM, given that 9 + 11 = 20
  sorry

end rendering_completion_time_l553_553571


namespace part_I_part_II_l553_553302

-- Define the absolute value function
def abs (x : ℝ) := if x ≥ 0 then x else -x

-- Definition of f(x) and g(x)
def f (x k : ℝ) := abs (3 * x - 1) + abs (3 * x + k)
def g (x : ℝ) := x + 4

-- First part of the proof problem
theorem part_I (x : ℝ) : 
  (k = -3) → (f x k ≥ 4 ↔ (x ≤ 0 ∨ x ≥ 4 / 3)) := 
  by sorry

-- Second part of the proof problem
theorem part_II (k : ℝ) : 
  (k > -1) → (∀ x, x ∈ Icc (-(k / 3)) ((1 : ℝ) / 3) → f x k ≤ g x) → 
  (-1 < k ∧ k ≤ 9 / 4) := 
  by sorry

end part_I_part_II_l553_553302


namespace stephanie_oranges_l553_553079

theorem stephanie_oranges (visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) 
  (h_visits : visits = 8) (h_oranges_per_visit : oranges_per_visit = 2) : 
  total_oranges = oranges_per_visit * visits := 
by
  sorry

end stephanie_oranges_l553_553079


namespace range_of_b_l553_553296

noncomputable def f (x b : ℝ) : ℝ := Real.exp x * (x - b)
noncomputable def f' (x b : ℝ) : ℝ := Real.exp x * (x - b + 1)
noncomputable def g (x : ℝ) : ℝ := (x ^ 2 + 2 * x) / (x + 1)

theorem range_of_b : (∃ x ∈ set.Icc (1/2 : ℝ) 2, f x b + x * f' x b > 0) ↔ b < 8 / 3 :=
by
  sorry

end range_of_b_l553_553296


namespace cost_of_toys_l553_553796

theorem cost_of_toys (x y : ℝ) (h1 : x + y = 40) (h2 : 90 / x = 150 / y) :
  x = 15 ∧ y = 25 :=
sorry

end cost_of_toys_l553_553796


namespace find_sum_l553_553400

variable {x y z w : ℤ}

-- Conditions: Consecutive integers and their sum condition
def consecutive_integers (x y z : ℤ) : Prop := y = x + 1 ∧ z = x + 2
def sum_is_150 (x y z : ℤ) : Prop := x + y + z = 150
def w_definition (w z x : ℤ) : Prop := w = 2 * z - x

-- Theorem statement
theorem find_sum (h1 : consecutive_integers x y z) (h2 : sum_is_150 x y z) (h3 : w_definition w z x) :
  x + y + z + w = 203 :=
sorry

end find_sum_l553_553400


namespace student_arrangement_count_l553_553250

theorem student_arrangement_count :
  let males := 4
  let females := 5
  let select_males := 2
  let select_females := 3
  let total_selected := select_males + select_females
  (Nat.choose males select_males) * (Nat.choose females select_females) * (Nat.factorial total_selected) = 7200 := 
by
  sorry

end student_arrangement_count_l553_553250


namespace initial_men_count_l553_553074

theorem initial_men_count (M : ℕ) (P : ℕ) :
  P = M * 20 →
  P = (M + 650) * 109 / 9 →
  M = 1000 :=
by
  sorry

end initial_men_count_l553_553074


namespace distance_from_point_P_to_plane_l553_553758

-- Define the normal vector of the plane
def normal_vector := (2 : ℝ, 2 : ℝ, -1 : ℝ)

-- Define the point P
def point_P := (-1 : ℝ, 3 : ℝ, 2 : ℝ)

-- Function to calculate the distance from a point to a plane given a normal vector and a point
def distance_to_plane (n : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) : ℝ :=
  let dot_product := n.1 * P.1 + n.2 * P.2 + n.3 * P.3
  let magnitude_n := Real.sqrt (n.1 * n.1 + n.2 * n.2 + n.3 * n.3)
  abs dot_product / magnitude_n

-- Prove that the distance from point P to the plane with the given normal vector is 2/3
theorem distance_from_point_P_to_plane :
  distance_to_plane normal_vector point_P = (2 : ℝ) / (3 : ℝ) :=
by
  -- The proof will be added here
  sorry

end distance_from_point_P_to_plane_l553_553758


namespace intersection_is_empty_l553_553410

open Finset

namespace ComplementIntersection

-- Define the universal set U, sets M and N
def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {2, 4, 5}

-- The complement of M with respect to U
def complement_U_M : Finset ℕ := U \ M

-- The complement of N with respect to U
def complement_U_N : Finset ℕ := U \ N

-- The intersection of the complements
def intersection_complements : Finset ℕ := complement_U_M ∩ complement_U_N

-- The proof statement
theorem intersection_is_empty : intersection_complements = ∅ :=
by sorry

end ComplementIntersection

end intersection_is_empty_l553_553410


namespace max_terms_geometric_sequence_l553_553262

theorem max_terms_geometric_sequence :
  ∀ (a : ℕ → ℤ) (r : ℤ), (∀ n, 100 ≤ a n ∧ a n ≤ 1000) →
  (∀ n, a (n + 1) = a n * r) →
  (1 < r) →
  (∀ n, a n ∈ ℤ) →
  ∃ n₀, n₀ ≤ 6 ∧ (∀ i < n₀, a i = (128 * (3/2)^i)) :=
by
  sorry

end max_terms_geometric_sequence_l553_553262


namespace determine_Ians_age_l553_553894

noncomputable theory

variables (p : ℤ → ℤ) (a b : ℤ)
-- Conditions
def zero_of_polynomial : Prop := p a = 0
def polynomial_at_7 : Prop := p 7 = 77
def polynomial_at_b : Prop := p b = 85
def age_greater_than_7 : Prop := a > 7
def age_greater_than_b : Prop := a > b

-- Proof problem
theorem determine_Ians_age :
  zero_of_polynomial p a ∧ polynomial_at_7 p ∧ polynomial_at_b p b ∧ age_greater_than_7 a ∧ age_greater_than_b a b
  → a = 14 :=
by
  sorry

end determine_Ians_age_l553_553894


namespace survey_most_suitable_for_census_l553_553532

-- Definitions of each option as a proposition
def A_survey : Prop := "Investigating the brand awareness of a certain product"
def B_survey : Prop := "Investigating the national viewership rating of CCTV's Spring Festival Gala"
def C_survey : Prop := "Testing the explosive power of a batch of ammunition"
def D_survey : Prop := "Investigating the monthly average electricity usage of 10 households in a residential building"

-- Define which survey method each option is suitable for
def suitable_for_census (survey: Prop) : Prop :=
  match survey with
  | A_survey => false
  | B_survey => false
  | C_survey => false
  | D_survey => true
  | _ => false

-- The proof problem statement
theorem survey_most_suitable_for_census : suitable_for_census D_survey := 
  by
  sorry

end survey_most_suitable_for_census_l553_553532


namespace count_non_integer_angle_measures_l553_553782

def interior_angle_measure (n : ℕ) : ℚ := 180 * (n - 2) / n

theorem count_non_integer_angle_measures : 
  ({ n : ℕ | 3 ≤ n ∧ n < 12 ∧ ¬ (interior_angle_measure n).den = 1 }).to_finset.card = 2 := by
sorry

end count_non_integer_angle_measures_l553_553782


namespace decrease_percent_revenue_l553_553855

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.68 * T
  let new_consumption := 1.12 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 23.84 := by {
    sorry
  }

end decrease_percent_revenue_l553_553855


namespace complex_square_eq_ints_l553_553463

theorem complex_square_eq_ints (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : (⟨a, b⟩ : ℤ) - I = ((⟨15, -8⟩ : ℤ))) :
  (a = 4 ∧ b = 1) :=
by
  have h_re : a^2 - b^2 = 15 := sorry
  have h_im : a * b = 4 := sorry
  cases h_eq,
  {
    exact ⟨4, 1⟩,
    left,
    exact ⟨1, 4⟩
  }
  univ

end complex_square_eq_ints_l553_553463


namespace roots_of_quadratic_eq_l553_553391

theorem roots_of_quadratic_eq (u v : ℝ) (h1 : u + v = 3 * real.sqrt 3) 
    (h2 : u * v = 3) 
    (h3 : ∀ x : ℝ, x^2 - (3 * real.sqrt 3) * x + 3 = 0 → (x = u ∨ x = v)) :
  u^6 + v^6 = 178767 :=
  sorry

end roots_of_quadratic_eq_l553_553391


namespace perfect_square_trinomial_l553_553212

theorem perfect_square_trinomial :
  120^2 - 40 * 120 + 20^2 = 10000 := sorry

end perfect_square_trinomial_l553_553212


namespace range_of_lambda_l553_553405

noncomputable def f (x λ : ℝ) : ℝ :=
if x < 1 then -x + λ else 2^x

theorem range_of_lambda (λ : ℝ) :
  (∀ a : ℝ, f (f a λ) λ = 2^(f a λ)) →
  λ ∈ Ici 2 :=
by
  intro h
  sorry

end range_of_lambda_l553_553405


namespace exponent_problem_l553_553511

theorem exponent_problem : (5 ^ 6 * 5 ^ 9 * 5) / 5 ^ 3 = 5 ^ 13 := 
by
  sorry

end exponent_problem_l553_553511


namespace percent_increase_lines_l553_553529

theorem percent_increase_lines (final_lines increase : ℕ) (h1 : final_lines = 5600) (h2 : increase = 1600) :
  (increase * 100) / (final_lines - increase) = 40 := 
sorry

end percent_increase_lines_l553_553529


namespace divisibility_condition_l553_553380

theorem divisibility_condition (n : ℕ) : 
  13 ∣ (4 * 3^(2^n) + 3 * 4^(2^n)) ↔ Even n := 
sorry

end divisibility_condition_l553_553380


namespace smallest_n_for_T_n_is_integer_l553_553025

def L : ℚ := ∑ i in Finset.range 9, i.succ⁻¹  -- sum of reciprocals of non-zero digits

def D : ℕ := 2^3 * 3^2 * 5 * 7  -- denominator of L in simplified form

def T (n : ℕ) : ℚ := (n * 5^(n-1)) * L + 1  -- expression for T_n

theorem smallest_n_for_T_n_is_integer : ∃ n : ℕ, 0 < n ∧ T n ∈ ℤ ∧ n = 504 :=
by
  use 504
  -- It remains to prove the conditions
  sorry

end smallest_n_for_T_n_is_integer_l553_553025


namespace smallest_n_for_T_integer_l553_553020

noncomputable def J := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def T (n : ℕ) : ℚ := ∑ x in finset.range (5^n + 1), 
  ∑ d in {
    digit | digit.to_nat ≠ 0 ∧ digit.to_nat < 10 
  }, (1 : ℚ) / (digit.to_nat : ℚ)

theorem smallest_n_for_T_integer : ∃ n : ℕ, T n ∈ ℤ ∧ ∀ m : ℕ, T m ∈ ℤ → 63 ≤ n :=
by {
  sorry
}

end smallest_n_for_T_integer_l553_553020


namespace num_valid_spellings_OLYMPIADS_l553_553197

def is_valid_spelling (s: String) (typed: String) : Prop :=
  ∀ (i : Nat) (c : Char), i < String.length s →
    c = s.get ⟨i, by simp [lt_trans (c.get_length_typed)]⟩ →
    (∀ j, j < String.length typed → 
      (typed.get ⟨j, by simp [lt_trans (c.get_length_typed)]⟩ = c →
        (j ≤ i + 1) ∧ ∀ k, k < i → s.get ⟨k, by simp [<]⟩ ≠ c))

theorem num_valid_spellings_OLYMPIADS :
  ∃ count : ℕ, count = 256 ∧ 
  (∀ (typed : String), is_valid_spelling "OLYMPIADS" typed → ∃ n, n = String.count typed) := 
sorry

end num_valid_spellings_OLYMPIADS_l553_553197


namespace constant_term_of_expansion_l553_553089

theorem constant_term_of_expansion : 
  (∃ T, (T = (\sqrt x - 1 / (2 * x))^9) → T == -21 / 2

end constant_term_of_expansion_l553_553089


namespace batsman_average_increase_l553_553886

theorem batsman_average_increase 
  (A : ℝ) 
  (H1 : 11 * A + 65 = 12 * (A + 3)) : 
  A + 3 = 32 :=
by
  have H : 11 * A + 65 = 12 * A + 36 := by rw [H1]
  linarith

end batsman_average_increase_l553_553886


namespace geom_series_sum_eq_l553_553646

theorem geom_series_sum_eq :
  let a := 1 / 3,
      r := 1 / 3
  in sum_geometric_series first_term a common_ratio r num_terms 10 = 29524 / 59049 :=
by
  -- Substitute values into the geometric series sum formula
  let a := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 10
  have h1 : sum_geometric_series a r n = (a * (1 - r^n)) / (1 - r), sorry
  -- Simplify and calculate
  -- have h2 : (a * (1 - r^n)) / (1 - r) = (1 / 2 * (1 - 1/3^10)) = 29524 / 59049, sorry
  sorry

noncomputable def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

end geom_series_sum_eq_l553_553646


namespace max_value_of_sum_cubes_l553_553683

theorem max_value_of_sum_cubes
  (x : Fin 2022 → ℝ)
  (h : ∑ i, x i ^ 2 = 1) :
  ∑ i in Finset.Ico 1 2021, (x i) ^ 3 * (x ⟨i+1, by linarith⟩) ^ 3 ≤ 1 / 8 :=
sorry

end max_value_of_sum_cubes_l553_553683


namespace particle_speed_correct_l553_553574

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 9)

noncomputable def particle_speed : ℝ :=
  Real.sqrt (3 ^ 2 + 5 ^ 2)

theorem particle_speed_correct : particle_speed = Real.sqrt 34 := by
  sorry

end particle_speed_correct_l553_553574


namespace solve_fraction_equation_l553_553983

theorem solve_fraction_equation: 
  ∃ (x : ℝ), (3 / (x - 3) = 1 / (x - 1)) ↔ (x = 0) :=
begin
  sorry
end

end solve_fraction_equation_l553_553983


namespace smallest_n_for_Tn_integer_l553_553029

noncomputable def K : ℚ := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n-1)) * K

theorem smallest_n_for_Tn_integer :
  ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m < n, T_n m ∉ ℤ :=
  sorry

end smallest_n_for_Tn_integer_l553_553029


namespace problem1_problem2_l553_553700

-- Problem 1 Statement
theorem problem1 (f : ℝ → ℝ) (h_f : ∀ x, f(x) = |x - 1|) : 
  { x : ℝ | (f x) ^ 2 ≤ 2 } = { x : ℝ | 1 - Real.sqrt 2 ≤ x ∧ x ≤ 1 + Real.sqrt 2 } :=
sorry

-- Problem 2 Statement
theorem problem2 (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) (h_f : ∀ x, f(x) = |x - a|)
  (h_g : ∀ x, g(x) = |2x| + 2 * |x - a|) (h_min : ∀ x, ∃ y, y = g(x) ∧ y = 4) : 
  a = 2 :=
sorry

end problem1_problem2_l553_553700


namespace sum_of_midpoints_x_coordinates_l553_553500

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l553_553500


namespace two_sectors_area_l553_553524

theorem two_sectors_area {r : ℝ} {θ : ℝ} (h_radius : r = 15) (h_angle : θ = 45) : 
  2 * (θ / 360) * (π * r^2) = 56.25 * π := 
by
  rw [h_radius, h_angle]
  norm_num
  sorry

end two_sectors_area_l553_553524


namespace circle_has_infinite_symmetry_lines_l553_553165

theorem circle_has_infinite_symmetry_lines (C : Type) [MetricSpace C] (circ : C) (h : IsSymmetric circ) : ∃ lines : ℕ, lines = 0 ∨ lines = ∞ :=
sorry

end circle_has_infinite_symmetry_lines_l553_553165


namespace total_original_cost_of_books_l553_553159

noncomputable def original_cost_price_in_eur (selling_prices : List ℝ) (profit_margin : ℝ) (exchange_rate : ℝ) : ℝ :=
  let original_cost_prices := selling_prices.map (λ price => price / (1 + profit_margin))
  let total_original_cost_usd := original_cost_prices.sum
  total_original_cost_usd * exchange_rate

theorem total_original_cost_of_books : original_cost_price_in_eur [240, 260, 280, 300, 320] 0.20 0.85 = 991.67 :=
  sorry

end total_original_cost_of_books_l553_553159


namespace parabola_coefficient_c_l553_553933

def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem parabola_coefficient_c (b c : ℝ) (h1 : parabola b c 1 = -1) (h2 : parabola b c 3 = 9) : 
  c = -3 := 
by
  sorry

end parabola_coefficient_c_l553_553933


namespace N_not_fifth_power_of_integer_l553_553038

def N : ℤ := ∑ k in finset.range 60, ((if (k % 2 = 0) then 1 else -1) * k^(k^k))

theorem N_not_fifth_power_of_integer : ¬ ∃ m : ℤ, N = m^5 :=
sorry

end N_not_fifth_power_of_integer_l553_553038


namespace inequality_holds_for_m_l553_553879

theorem inequality_holds_for_m (x m : ℝ) (hx : x ∈ set.Iic (-1)) 
  (h : (m^2 - m) * 4^x - 2^x < 0) : m ∈ set.Ioo (-1) 2 :=
sorry

end inequality_holds_for_m_l553_553879


namespace factor_poly_l553_553727

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l553_553727


namespace average_salary_proof_l553_553822

noncomputable def average_salary_of_all_workers (tech_workers : ℕ) (tech_avg_sal : ℕ) (total_workers : ℕ) (non_tech_avg_sal : ℕ) : ℕ :=
  let non_tech_workers := total_workers - tech_workers
  let total_tech_salary := tech_workers * tech_avg_sal
  let total_non_tech_salary := non_tech_workers * non_tech_avg_sal
  let total_salary := total_tech_salary + total_non_tech_salary
  total_salary / total_workers

theorem average_salary_proof : average_salary_of_all_workers 7 14000 28 6000 = 8000 := by
  sorry

end average_salary_proof_l553_553822


namespace net_gain_loss_eq_196_l553_553171

-- Define the variables and conditions
variable (x : ℝ)
def gain (x : ℝ) : ℝ := 0.14 * x
def loss (x : ℝ) : ℝ := 0.14 * x

theorem net_gain_loss_eq_196 (h₁ : gain x - loss x = 1.96) : false :=
by
  -- Since gain and loss are equal, lhs results in 0, which contradicts 1.96.
  have h₂ : gain x - loss x = 0 := by
    rw [gain, loss]
    linarith
  linarith

end net_gain_loss_eq_196_l553_553171


namespace charge_of_second_sphere_l553_553129

variables {Q1 Q2 q2 C1 C : ℝ}

theorem charge_of_second_sphere
  (Q1 Q2 q2 : ℝ) (C1 C : ℝ)
  (q1 : ℝ := Q1 / (1 + C1 / C)) :
  ∃ Q2' : ℝ, Q2' = (Q2 / 2) - q2 + sqrt (Q2^2 / 4 + Q1 * q2) ∨
              Q2' = (Q2 / 2) - q2 - sqrt (Q2^2 / 4 + Q1 * q2) := 
by 
  sorry

end charge_of_second_sphere_l553_553129


namespace tank_capacity_l553_553556

theorem tank_capacity (liters_cost : ℕ) (liters_amount : ℕ) (full_tank_cost : ℕ) (h₁ : liters_cost = 18) (h₂ : liters_amount = 36) (h₃ : full_tank_cost = 32) : 
  (full_tank_cost * liters_amount / liters_cost) = 64 :=
by 
  sorry

end tank_capacity_l553_553556


namespace log10_bounds_sum_l553_553510

theorem log10_bounds_sum
  (c d : ℤ)
  (h1 : 10 ^ 4 = 10000)
  (h2 : 10 ^ 5 = 100000)
  (h3 : 10000 < 56342)
  (h4 : 56342 < 100000)
  (h5 : (c : ℝ) ≤ real.log10 56342)
  (h6 : real.log10 56342 < (d : ℝ)) : 
  c + d = 9 := sorry

end log10_bounds_sum_l553_553510


namespace line_through_center_eq_line_bisects_chord_eq_l553_553663

section Geometry

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define the point P
def P := (2, 2)

-- Define when line l passes through the center of the circle
def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define when line l bisects chord AB by point P
def line_bisects_chord (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- Prove the equation of line l passing through the center
theorem line_through_center_eq : 
  (∀ (x y : ℝ), line_through_center x y → circleC x y → (x, y) = (1, 0)) →
  2 * (2:ℝ) - 2 - 2 = 0 := sorry

-- Prove the equation of line l bisects chord AB by point P
theorem line_bisects_chord_eq:
  (∀ (x y : ℝ), line_bisects_chord x y → circleC x y → (2, 2) = P) →
  (2 + 2 * 2 - 6 = 0) := sorry

end Geometry

end line_through_center_eq_line_bisects_chord_eq_l553_553663


namespace squares_end_with_76_l553_553977

noncomputable def validNumbers : List ℕ := [24, 26, 74, 76]

theorem squares_end_with_76 (x : ℕ) (h₁ : x % 10 = 4 ∨ x % 10 = 6) 
    (h₂ : (x * x) % 100 = 76) : x ∈ validNumbers := by
  sorry

end squares_end_with_76_l553_553977


namespace permutation_sum_l553_553905

theorem permutation_sum (n : ℕ) (h1 : n + 3 ≤ 2 * n) (h2 : n + 1 ≤ 4) (h3 : n > 0) :
  Nat.factorial (2 * n) / Nat.factorial (2 * n - (n + 3)) + Nat.factorial 4 / Nat.factorial (4 - (n + 1)) = 744 :=
by
  sorry

end permutation_sum_l553_553905


namespace min_value_of_a_l553_553723

theorem min_value_of_a (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x + y) * (1/x + a/y) ≥ 16) : a ≥ 9 :=
sorry

end min_value_of_a_l553_553723


namespace find_missing_number_l553_553742

theorem find_missing_number :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 :=
by
  intros h1 h2
  sorry

end find_missing_number_l553_553742


namespace find_m_from_min_value_l553_553249

noncomputable theory

def quadratic_min_value (a b c : ℝ) : ℝ := (4 * a * c - b^2) / (4 * a)

theorem find_m_from_min_value (m : ℝ) (h : quadratic_min_value 1 (-4) m = 4) : m = 8 :=
by
  sorry

end find_m_from_min_value_l553_553249


namespace math_problem_correctness_l553_553306

variable {t α : ℝ}
variable {x y ρ θ : ℝ}

-- Conditions
def line_l (t α : ℝ) : Prop :=
  x = -1 + t * cos α ∧ y = 1 + t * sin α

def curve_C (ρ θ : ℝ) : Prop :=
  ρ = ρ * cos θ + 2

-- Equivalents to prove
def fixed_point : Prop :=
  -1 + 0 * cos α = -1 ∧ 1 + 0 * sin α = 1

def cartesian_curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 = (x + 2)^2

def polar_line_l_for_alpha (θ : ℝ) : Prop :=
  ρ * sin θ = ρ * cos θ + 2

def intersection_point (ρ θ : ℝ) : Prop :=
  θ = π / 2 ∧ ρ = 2

-- Main theorem to be proven
theorem math_problem_correctness :
  (line_l t α ∧ curve_C ρ θ) →
  fixed_point ∧ cartesian_curve_C x y ∧ (α = π / 4 → polar_line_l_for_alpha θ)
  ∧ (α = π / 4 → intersection_point ρ θ) :=
by
  sorry

end math_problem_correctness_l553_553306


namespace coefficient_x5y3_in_expansion_l553_553854

theorem coefficient_x5y3_in_expansion :
  let f (x y : ℤ) (n : ℕ) := (x^2 - x + 2 * y)^n
  f 1 1 6 = 64 →
  (binom 6 3) * (2^3) * (6 - 3) * (-3) = -480 :=
by
  intros f h
  have h1 : (1^2 - 1 + 2 * 1)^6 = 64 := h
  have h2 := binom 6 3 * 2^3 * (6 - 3) * (-3)
  sorry

end coefficient_x5y3_in_expansion_l553_553854


namespace arithmetic_sequences_integer_ratio_count_l553_553313

theorem arithmetic_sequences_integer_ratio_count 
  (a_n b_n : ℕ → ℕ)
  (A_n B_n : ℕ → ℕ)
  (h₁ : ∀ n, A_n n = n * (a_n 1 + a_n (2 * n - 1)) / 2)
  (h₂ : ∀ n, B_n n = n * (b_n 1 + b_n (2 * n - 1)) / 2)
  (h₃ : ∀ n, A_n n / B_n n = (7 * n + 41) / (n + 3)) :
  ∃ (cnt : ℕ), cnt = 3 ∧ ∀ n, (∃ k, n = 1 + 3 * k) → (a_n n) / (b_n n) = 7 + (10 / (n + 1)) :=
by
  sorry

end arithmetic_sequences_integer_ratio_count_l553_553313


namespace f_of_f_3_eq_3_l553_553404

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 1 - Real.logb 2 (2 - x) else 2^(1 - x) + 3 / 2

theorem f_of_f_3_eq_3 : f (f 3) = 3 := by
  sorry

end f_of_f_3_eq_3_l553_553404


namespace line_intersects_ellipse_slopes_l553_553929

theorem line_intersects_ellipse_slopes (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ 
  m ∈ Set.Iic (-Real.sqrt (1/5)) ∨ m ∈ Set.Ici (Real.sqrt (1/5)) :=
by
  sorry

end line_intersects_ellipse_slopes_l553_553929


namespace area_of_closed_figure_l553_553996

open Real

theorem area_of_closed_figure :
  let C := fun x => exp x
  let T := fun x => x + 1
  ∫ x in 0..2, (C x - T x) = exp 2 - 5 :=
sorry

end area_of_closed_figure_l553_553996


namespace percentage_of_september_authors_l553_553479

def total_authors : ℕ := 120
def september_authors : ℕ := 15

theorem percentage_of_september_authors : 
  (september_authors / total_authors : ℚ) * 100 = 12.5 :=
by
  sorry

end percentage_of_september_authors_l553_553479


namespace probability_of_units_digit_6_l553_553932

theorem probability_of_units_digit_6 :
  (probability (m^n ∈ (units_digit 6)) (m ∈ {12, 14, 16, 18, 20} ∧ n ∈ {2005, 2006, ..., 2024})) = 2 / 5 :=
by
  sorry

end probability_of_units_digit_6_l553_553932


namespace tangent_line_equation_l553_553703

noncomputable def f (x : ℝ) := 3 * x + real.cos (2 * x) + real.sin (2 * x)
noncomputable def f_prime (x : ℝ) := (deriv f) x
noncomputable def a := f_prime (real.pi / 4)
def curve_y (x : ℝ) := x ^ 3
noncomputable def P := (a, curve_y a)

theorem tangent_line_equation 
  (a : ℝ)
  (h1 : f_prime (real.pi / 4) = a) 
  (h2 : a = 1) 
  (x y : ℝ) : 
  (a = 1 → 3 * x - y - 2 = 0) ∨ (a ≠ 1 → 3 * x - 4 * y + 1 = 0) :=
by
  sorry

end tangent_line_equation_l553_553703


namespace sum_f_1_to_2013_l553_553219

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -(x + 2) ^ 2
  else if -1 ≤ x ∧ x < 3 then x
  else if 3 ≤ x ∧ x < 6 then f (x - 6)
  else f (x + 6)

theorem sum_f_1_to_2013 : (∑ k in finset.range 2013, f k.succ) = 337 :=
sorry

end sum_f_1_to_2013_l553_553219


namespace find_x_l553_553318

theorem find_x (x : ℝ) (a b c : ℝ × ℝ × ℝ)
  (h_a : a = (1, 1, x))
  (h_b : b = (1, 2, 1))
  (h_c : c = (1, 1, 1))
  (h_cond : let diff := (c.1 - a.1, c.2 - a.2, c.3 - a.3)
            let b_scaled := (2 * b.1, 2 * b.2, 2 * b.3)
            diff.1 * b_scaled.1 + diff.2 * b_scaled.2 + diff.3 * b_scaled.3 = -2) :
  x = 2 :=
by
  sorry

end find_x_l553_553318


namespace union_of_sets_l553_553403

-- Define the sets A and B with the given conditions
def A (a : ℤ) : Set ℤ := {abs (a + 1), 3, 5}
def B (a : ℤ) : Set ℤ := {2 * a + 1, a * a + 2 * a, a * a + 2 * a - 1}

-- Define the conditions as hypotheses
theorem union_of_sets (a : ℤ) (h_inter : A a ∩ B a = {2, 3}) : A a ∪ B a = {-5, 2, 3, 5} :=
by
  -- Proof body will go here
  sorry

end union_of_sets_l553_553403


namespace sum_of_midpoints_l553_553497

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l553_553497


namespace ticket_difference_l553_553943

-- Definitions representing the number of VIP and general admission tickets
def numTickets (V G : Nat) : Prop :=
  V + G = 320

def totalCost (V G : Nat) : Prop :=
  40 * V + 15 * G = 7500

-- Theorem stating that the difference between general admission and VIP tickets is 104
theorem ticket_difference (V G : Nat) (h1 : numTickets V G) (h2 : totalCost V G) : G - V = 104 := by
  sorry

end ticket_difference_l553_553943


namespace quadratic_with_sum_and_abs_diff_l553_553523

theorem quadratic_with_sum_and_abs_diff (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 6) :
  (x - 8) * (x - 2) = 0 → (∀ (a b : ℝ), (a - 8) * (a - 2) = 0 → y = a ∨ x = a → a + b = 10 → a * b = 16) :=
by intros; ring_exp_eq; try {subst_vars};
  sorry

end quadratic_with_sum_and_abs_diff_l553_553523


namespace polar_coordinates_of_point_l553_553971

open Real

theorem polar_coordinates_of_point :
  ∃ r θ : ℝ, r = 4 ∧ θ = 5 * π / 3 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
           (∃ x y : ℝ, x = 2 ∧ y = -2 * sqrt 3 ∧ x = r * cos θ ∧ y = r * sin θ) :=
sorry

end polar_coordinates_of_point_l553_553971


namespace smallest_n_for_T_n_integer_l553_553005

noncomputable def K : ℚ := ∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n - 1)) * K + 1

theorem smallest_n_for_T_n_integer : ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m : ℕ, T_n m ∈ ℤ → n ≤ m :=
  ⟨504, sorry⟩

end smallest_n_for_T_n_integer_l553_553005


namespace sum_of_midpoints_x_coordinates_l553_553501

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l553_553501


namespace find_x_l553_553315

variable (x : ℝ)

def a : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (-1, 3)

theorem find_x (h : (a x).1 + 2 * (b x).1 = 3 * (-1) ∧ (a x).2 + 2 * (b x).2 = 5) :
  x = -11 / 3 :=
by
  sorry

end find_x_l553_553315


namespace A_more_than_B_l553_553945

noncomputable def proportion := (5, 3, 2, 3)
def C_share := 1000
def parts := 2
noncomputable def part_value := C_share / parts
noncomputable def A_share := part_value * 5
noncomputable def B_share := part_value * 3

theorem A_more_than_B : A_share - B_share = 1000 := by
  sorry

end A_more_than_B_l553_553945


namespace triangle_proof_l553_553355

noncomputable def triangle_values := 
  ∃ (D E F R S T: Type) [Point D] [Point E] [Point F] [Point R] [Point S] [Point T]
  (TR TS ER ES DR DS : ℝ),
  TR = 8 ∧ 
  TS = 3 ∧ 
  (ER * ES - DR * DS = 55)

theorem triangle_proof : triangle_values :=
by sorry

end triangle_proof_l553_553355


namespace sequence_strictly_monotonic_increasing_l553_553442

noncomputable def a (n : ℕ) : ℝ := ((n + 1) ^ n * n ^ (2 - n)) / (7 * n ^ 2 + 1)

theorem sequence_strictly_monotonic_increasing :
  ∀ n : ℕ, a n < a (n + 1) := 
by {
  sorry
}

end sequence_strictly_monotonic_increasing_l553_553442


namespace count_integers_satisfying_sqrt_condition_l553_553846

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l553_553846


namespace count_of_valid_prime_numbers_l553_553185

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_valid_number (n : ℕ) : Prop := 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 
  10 * y + x = n ∧ 10 * x + y = n + 9 ∧ is_prime n

theorem count_of_valid_prime_numbers : 
  finset.card (finset.filter is_valid_number (finset.filter is_prime (finset.range 100))) = 3 :=
sorry

end count_of_valid_prime_numbers_l553_553185


namespace number_subtracted_eq_l553_553909

theorem number_subtracted_eq (x n : ℤ) (h1 : x + 1315 + 9211 - n = 11901) (h2 : x = 88320) : n = 86945 :=
by
  sorry

end number_subtracted_eq_l553_553909


namespace pencils_multiple_of_71_l553_553097

theorem pencils_multiple_of_71 (P : ℕ) : 
  (781 % 71 = 0) → (∃ x : ℕ, P = 71 * x) :=
by
  -- Given that the maximum number of students among whom 781 pens can be distributed is 71
  assume h : 781 % 71 = 0
  -- We need to prove that the number of pencils (P) is a multiple of 71
  sorry

end pencils_multiple_of_71_l553_553097


namespace switches_in_A_after_729_steps_l553_553121

-- Definition for the switches and their labels in the problem
def switch_label (x y z : ℕ) : ℕ := (2 ^ x) * (3 ^ y) * (7 ^ z)

-- Definition for the maximum switch label
def max_label : ℕ := switch_label 5 5 5

-- Definition for the divisors of a switch label
def divisors (d_i : ℕ) : ℕ :=
  let N_div_d_i := max_label / d_i
  finset.card (finset.divisors N_div_d_i)

-- Definition to check if a switch is in position A after 729 steps
def in_position_A (x y z : ℕ) : Prop :=
  (6 - x) * (6 - y) * (6 - z) % 4 = 0

-- Count of total switches given the range of x, y, and z
def total_switches : ℕ := 6 * 6 * 6 -- 729

-- Count of switches remaining in position A
def switches_in_position_A : ℕ :=
  total_switches - (finset.sum (finset.range 6) (λ x,
                    finset.sum (finset.range 6) (λ y,
                    finset.sum (finset.range 6) (λ z,
                      if ¬ in_position_A x y z then 1 else 0))))

theorem switches_in_A_after_729_steps : switches_in_position_A = 675 :=
by {
  -- Since it's a placeholder for the actual proof, we use sorry
  sorry
}

end switches_in_A_after_729_steps_l553_553121


namespace final_price_chocolate_l553_553473

-- Conditions
def original_cost : ℝ := 2.00
def discount : ℝ := 0.57

-- Question and answer
theorem final_price_chocolate : original_cost - discount = 1.43 :=
by
  sorry

end final_price_chocolate_l553_553473


namespace maximum_unit_vectors_l553_553278

-- Define unit vectors in a plane
noncomputable def unit_vectors (n : ℕ) : Prop :=
  ∃ (a : Fin n → ℝ × ℝ), (∀ i, (a i).fst ^ 2 + (a i).snd ^ 2 = 1) ∧
  (∀ i j, i < j → (a i).fst * (a j).fst + (a i).snd * (a j).snd < 1 / 2)

theorem maximum_unit_vectors (n : ℕ) : n ≤ 5 :=
begin
  sorry
end

end maximum_unit_vectors_l553_553278


namespace range_of_f_value_of_a_l553_553406

noncomputable def f (x : ℝ) : ℝ :=
  cos (x + 2 * Real.pi / 3) + 2 * cos (x / 2) ^ 2

theorem range_of_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 0 ≤ f(x) ∧ f(x) ≤ 2 :=
sorry

theorem value_of_a (A B C a b c : ℝ)
  (hB : f B = 1) (hb : b = 1) (hc : c = sqrt 3)
  (h_sum_angles : A + B + C = Real.pi) (h_pos_angles : ∀ (α : ℝ), α ∈ [A, B, C] → 0 < α < Real.pi) 
  (h_law_of_cosines : b * b = a * a + c * c - 2 * a * c * cos B) :
  a = 1 ∨ a = 2 :=
sorry

end range_of_f_value_of_a_l553_553406


namespace number_of_binders_l553_553042

-- Definitions of given conditions
def book_cost : Nat := 16
def binder_cost : Nat := 2
def notebooks_cost : Nat := 6
def total_cost : Nat := 28

-- Variable for the number of binders
variable (b : Nat)

-- Proposition that the number of binders Léa bought is 3
theorem number_of_binders (h : book_cost + binder_cost * b + notebooks_cost = total_cost) : b = 3 :=
by
  sorry

end number_of_binders_l553_553042


namespace intersection_of_diagonals_l553_553841

open EuclideanGeometry Real

noncomputable def is_cyclic (P Q R S : Points) : Prop := sorry
noncomputable def distance_to_side (S : Point) (l : Line) : Real := sorry

-- Assume we have points M, N, P, Q which form a cyclic quadrilateral, and a point S inside.
variables (M N P Q S : Point)

-- Definitions of the sides as lines
def side_MN := Line.mk M N
def side_NP := Line.mk N P
def side_PQ := Line.mk P Q
def side_QM := Line.mk Q M

-- Condition for cyclic quadrilateral
axiom cyclic_MNPQ : is_cyclic M N P Q

-- Condition for distances being proportional to sides
axiom distances_proportional :
  ∃ k : ℝ, k > 0 ∧
    distance_to_side S side_MN / distance_to_side S side_QM = k ∧
    distance_to_side S side_NP / distance_to_side S side_MN = k ∧
    distance_to_side S side_PQ / distance_to_side S side_NP = k

-- Definition of intersection point of the diagonals
def diagonals_intersect (M N P Q : Point) : Point := sorry

theorem intersection_of_diagonals:
  distances_proportional M N P Q S →
  cyclic_MNPQ M N P Q →
  S = diagonals_intersect M N P Q :=
sorry

end intersection_of_diagonals_l553_553841


namespace minimum_people_with_all_luxuries_l553_553354

variable {U : Type} [Fintype U]

-- Define the subsets representing people having each luxury.
variables (Refrigerator : Set U) (Television : Set U) (Computer : Set U) (AirConditioner : Set U)

-- Define the corresponding percentages as finite sets.
variables (hR : Fintype.card Refrigerator = 90 * Fintype.card U / 100)
variables (hT : Fintype.card Television = 86 * Fintype.card U / 100)
variables (hC : Fintype.card Computer = 80 * Fintype.card U / 100)
variables (hA : Fintype.card AirConditioner = 80 * Fintype.card U / 100)

-- Prove that at least 80% of the people in U have all four luxuries.
theorem minimum_people_with_all_luxuries :
  Fintype.card (Refrigerator ∩ Television ∩ Computer ∩ AirConditioner) ≥ 80 * Fintype.card U / 100 :=
sorry

end minimum_people_with_all_luxuries_l553_553354


namespace isosceles_triangle_base_length_l553_553601

theorem isosceles_triangle_base_length (a b c : ℕ) (h_isosceles : a = b ∨ b = c ∨ c = a)
  (h_perimeter : a + b + c = 16) (h_side_length : a = 6 ∨ b = 6 ∨ c = 6) :
  (a = 4 ∨ b = 4 ∨ c = 4) ∨ (a = 6 ∨ b = 6 ∨ c = 6) :=
sorry

end isosceles_triangle_base_length_l553_553601


namespace common_difference_is_four_l553_553508

theorem common_difference_is_four
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (d : ℤ)
  (h1 : ∀n, S_n n = n * a_n(1) + n * (n - 1) / 2 * d)
  (h2 : S_n 2016 / 2016 = S_n 2015 / 2015 + 2) :
  d = 4 := 
sorry

end common_difference_is_four_l553_553508


namespace set_a_when_a_is_2_range_of_a_l553_553710

theorem set_a_when_a_is_2 (a : ℝ) (A : set ℝ) (log : ℝ → ℝ) :
  (∀ x, log (x - a) < 2 ↔ 2 < x ∧ x < 6) ∧ a = 2 → A = { x | 2 < x ∧ x < 6 } :=
by
  intro h
  sorry

theorem range_of_a (A : set ℝ) (a : ℝ) :
  (2 ∉ A ∧ 3 ∈ A) → A = { x | 2 < x ∧ x < 6 } →
  (∀ a : ℝ, (a ≥ 2 ∨ a ≤ -2) ∧ (-1 < a ∧ a < 3)) →
  a ∈ set.Icc 2 3 :=
by
  intro h1 h2 h3
  sorry

end set_a_when_a_is_2_range_of_a_l553_553710


namespace log_identity_l553_553256

theorem log_identity (a b : ℝ) (h1 : log 14 7 = a) (h2 : log 14 5 = b) : log 35 14 = 1 / (a + b) :=
by 
  sorry

end log_identity_l553_553256


namespace total_money_shared_l553_553424

theorem total_money_shared (k t : ℕ) (h1 : k = 1750) (h2 : t = 2 * k) : k + t = 5250 :=
by
  sorry

end total_money_shared_l553_553424


namespace monotonicity_and_range_of_a_l553_553702

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) * Real.exp x
noncomputable def g (a x : ℝ) : ℝ := a * x * Real.exp x - (x^2 + 2 * x - 1)

theorem monotonicity_and_range_of_a 
  (a b : ℝ) 
  (h_extremum: (a * -1 + b + a) * Real.exp (-1) = 0)
  (h_condition: ∀ x : ℝ, x ≥ -1 → f a b x ≥ x^2 + 2 * x - 1) :
  (∀ x : ℝ, x < -1 → derivative (f a b) x < 0 ∧ x > -1 → derivative (f a b) x > 0 ∨
   ∀ x : ℝ, x < -1 → derivative (f a b) x > 0 ∧ x > -1 → derivative (f a b) x < 0) ∧
  (∀ x : ℝ, x ≥ -1 → g a x ≥ 0 → (real.log (2 / a))^2 ≤ 1) →
  (Real.exp 2) / (Real.exp 1) ≤ a ∧ a ≤ 2 * Real.exp 1 :=
  by
    sorry

end monotonicity_and_range_of_a_l553_553702


namespace sum_of_perpendiculars_to_sides_l553_553804

variable {A B C A1 B1 C1 O: Type}
variable {OA1 OB1 OC1 R ρ : ℝ}
variable (circumcenter : O)
variable (a b c : ℝ)

/-- 
  Proving the geometric property that the sum of the perpendiculars from the circumcenter 
  to the sides of the triangle is equal to the sum of the circumradius and the inradius
-/
theorem sum_of_perpendiculars_to_sides (h1 : OA1 = x)
                                      (h2 : OB1 = y) 
                                      (h3 : OC1 = z) 
                                      (circumradius : R) 
                                      (inradius : ρ)
                                      (sides_sum : a + b + c = x + y + z):
                                      x + y + z = R + ρ :=
by
  sorry

end sum_of_perpendiculars_to_sides_l553_553804


namespace total_money_shared_l553_553423

theorem total_money_shared (k t : ℕ) (h1 : k = 1750) (h2 : t = 2 * k) : k + t = 5250 :=
by
  sorry

end total_money_shared_l553_553423


namespace solve_for_n_l553_553724

theorem solve_for_n (n : ℕ) (h : 3 * sqrt (8 + n) = 15) : n = 17 := by
  sorry

end solve_for_n_l553_553724


namespace sum_same_probability_l553_553093

-- Definition for standard dice probability problem
def dice_problem (n : ℕ) (target_sum : ℕ) (target_sum_of_faces : ℕ) : Prop :=
  let faces := [1, 2, 3, 4, 5, 6]
  let min_sum := n * 1
  let max_sum := n * 6
  let average_sum := (min_sum + max_sum) / 2
  let symmetric_sum := 2 * average_sum - target_sum
  symmetric_sum = target_sum_of_faces

-- The proof statement (no proof included, just the declaration)
theorem sum_same_probability : dice_problem 8 12 44 :=
by sorry

end sum_same_probability_l553_553093


namespace product_of_total_points_l553_553193

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def Allie_rolls : List ℕ := [3, 5, 6, 2, 4]
def Betty_rolls : List ℕ := [3, 2, 1, 6, 4]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem product_of_total_points :
  total_points Allie_rolls * total_points Betty_rolls = 256 :=
by
  sorry

end product_of_total_points_l553_553193


namespace find_n_l553_553873

theorem find_n :
  ∃ n : ℤ, 0 ≤ n ∧ n < 13 ∧ (-2222 ≡ n [MOD 13]) ∧ n = 12 :=
by
  sorry

end find_n_l553_553873


namespace f_2017_equals_neg_sin_x_l553_553658

def f (x : ℝ) : ℝ := Real.cos x

def f_n : ℕ → (ℝ → ℝ)
| 0     := f
| (n+1) := (λ x, (f_n n).derivative.derivative x)

theorem f_2017_equals_neg_sin_x : ∀ x : ℝ, f_n 2017 x = -Real.sin x := 
sorry

end f_2017_equals_neg_sin_x_l553_553658


namespace intersection_AC_OB_l553_553273

/-- Point A has coordinates (4, 0). -/
def A : ℝ × ℝ := (4, 0)

/-- Point B has coordinates (4, 4). -/
def B : ℝ × ℝ := (4, 4)

/-- Point C has coordinates (2, 6). -/
def C : ℝ × ℝ := (2, 6)

/-- Point O is the coordinate origin (0, 0). -/
def O : ℝ × ℝ := (0, 0)

/-- Point P, the intersection of line segments AC and OB, has coordinates (3, 3). -/
theorem intersection_AC_OB :
  ∃ (P : ℝ × ℝ), P = (3, 3) ∧ 
    ∃ λ τ : ℝ, 
      (P = (λ * B.1, λ * B.2)) ∧ 
      (P = ((1 - τ) * A.1 + τ * C.1, (1 - τ) * A.2 + τ * C.2)) :=
  sorry

end intersection_AC_OB_l553_553273


namespace remainder_149_pow_151_plus_151_pow_149_div_22499_mod_10000_l553_553240

theorem remainder_149_pow_151_plus_151_pow_149_div_22499_mod_10000 :
  let p := 149
  let q := 151
  let n := 22499
  (⟦(p^q + q^p) / n⟧ % 10000) = 7800 :=
by
  let p := 149
  let q := 151
  let n := 22499
  have h1 : p^q % q = p := sorry
  have h2 : q^p % p = q := sorry
  have h3 : p * q = n := rfl
  sorry

end remainder_149_pow_151_plus_151_pow_149_div_22499_mod_10000_l553_553240


namespace min_y_squared_l553_553037

noncomputable def isosceles_trapezoid_bases (EF GH : ℝ) := EF = 102 ∧ GH = 26

noncomputable def trapezoid_sides (EG FH y : ℝ) := EG = y ∧ FH = y

noncomputable def tangent_circle (center_on_EF tangent_to_EG_FH : Prop) := 
  ∃ P : ℝ × ℝ, true -- center P exists somewhere and lies on EF

theorem min_y_squared (EF GH EG FH y : ℝ) (center_on_EF tangent_to_EG_FH : Prop) 
  (h1 : isosceles_trapezoid_bases EF GH)
  (h2 : trapezoid_sides EG FH y)
  (h3 : tangent_circle center_on_EF tangent_to_EG_FH) : 
  ∃ n : ℝ, n^2 = 1938 :=
sorry

end min_y_squared_l553_553037


namespace distance_focus_directrix_l553_553092

theorem distance_focus_directrix (p : ℝ) : 
  let parabola_equation := ∀ y x: ℝ, y^2 = 8*x in
  let focus := (2 : ℝ, 0 : ℝ) in
  let directrix := -2 in
  dist focus (directrix, 0) = 4 := 
by 
  sorry

end distance_focus_directrix_l553_553092


namespace find_lower_grades_students_l553_553516

/-- Given conditions about car and motorcycle ownership among students -/
variables 
  (seniors : ℕ) (seniors_with_cars seniors_with_motorcycles seniors_with_vehicle : ℕ)
  (lower_grades : ℝ) (lower_grades_with_cars lower_grades_with_motorcycles lower_grades_with_vehicle : ℝ)
  (total_students students_with_vehicle : ℝ)

def MorseHighSchool (seniors : ℕ) (lower_grades : ℝ) : Prop :=
  seniors = 300 ∧
  seniors_with_cars = 0.40 * seniors ∧
  seniors_with_motorcycles = 0.05 * seniors ∧
  seniors_with_vehicle = seniors_with_cars + seniors_with_motorcycles ∧
  lower_grades_with_cars = 0.10 * lower_grades ∧
  lower_grades_with_motorcycles = 0.03 * lower_grades ∧
  lower_grades_with_vehicle = lower_grades_with_cars + lower_grades_with_motorcycles ∧
  students_with_vehicle = seniors_with_vehicle + lower_grades_with_vehicle ∧
  total_students = seniors + lower_grades ∧
  0.20 * total_students = students_with_vehicle

theorem find_lower_grades_students
  (h : MorseHighSchool 300 X) :
  X = 1071 :=
sorry

end find_lower_grades_students_l553_553516


namespace minimum_number_of_handshakes_l553_553911

-- Definitions and conditions
constant n : ℕ := 30
constant at_least_handshakes : ℕ := 3

-- The statement that we need to prove:
theorem minimum_number_of_handshakes (n : ℕ) (at_least_handshakes : ℕ) (h : n = 30) (h2 : at_least_handshakes = 3) : ∃ (k : ℕ), k = 45 :=
by
  sorry

end minimum_number_of_handshakes_l553_553911


namespace smallest_n_for_T_n_integer_l553_553001

def L : ℚ := ∑ i in {1, 2, 3, 4}, 1 / i

theorem smallest_n_for_T_n_integer : ∃ n ∈ ℕ, n > 0 ∧ (n * 5^(n-1) * L).denom = 1 ∧ n = 12 :=
by
  have hL : L = 25 / 12 := by sorry
  existsi 12
  split
  exact Nat.succ_pos'
  split
  suffices (12 * 5^(12-1) * 25 / 12).denom = 1 by sorry
  sorry
  rfl

end smallest_n_for_T_n_integer_l553_553001


namespace time_to_build_wall_l553_553919

theorem time_to_build_wall (t_A t_B t_C : ℝ) 
  (h1 : 1 / t_A + 1 / t_B = 1 / 25)
  (h2 : 1 / t_C = 1 / 35)
  (h3 : 1 / t_A = 1 / t_B + 1 / t_C) : t_B = 87.5 :=
by
  sorry

end time_to_build_wall_l553_553919


namespace circle_eq_vals_l553_553475

theorem circle_eq_vals (D E F : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔
    (x + 2)^2 + (y - 3)^2 = 4^2) →
  (D = 4 ∧ E = -6 ∧ F = -3) :=
by {
  sorry, -- Proof is omitted.
}

end circle_eq_vals_l553_553475


namespace range_of_a_l553_553340

def point_A (a : ℝ) : ℝ × ℝ := (1, a)

def on_same_side (A B : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  l A * l B > 0

noncomputable def line_l : ℝ × ℝ → ℝ :=
  λ p, p.1 + p.2 - 1

theorem range_of_a (a : ℝ) (h : on_same_side (point_A a) (0, 0) line_l) : a < 0 :=
  sorry

end range_of_a_l553_553340


namespace triangle_area_lt_3_l553_553761

theorem triangle_area_lt_3 
  (A B C : Type)
  (AC : A ≠ B)
  (BAC : A ≠ C)
  (circ_radius : ℝ)
  (h_radius : circ_radius = 2)
  (angle_BAC : ℝ)
  (h_angle_BAC : angle_BAC = 30)
  (side_AC : ℝ)
  (h_side_AC : side_AC = 3)
  : triangle_area A B C < 3 := 
sorry

end triangle_area_lt_3_l553_553761


namespace right_triangles_product_squared_l553_553807

theorem right_triangles_product_squared
  (T1 T2 : Triangle)
  (area_T1 : T1.area = 4)
  (area_T2 : T2.area = 9)
  (congruent_side : ∃ (x : ℝ), x ∈ T1.sides ∧ x ∈ T2.sides)
  (isosceles_right_T1 : T1.isosceles ∧ T1.right_angle):
  (square_of_product_of_third_sides_of_T1_and_T2 : ℝ) :=
sorry

end right_triangles_product_squared_l553_553807


namespace find_ordered_pair_l553_553238

def cosine_30_eq : Real := Real.cos (Real.pi / 6) -- \cos 30°
def secant_30_eq : Real := 1 / (Real.cos (Real.pi / 6)) -- \sec 30°

theorem find_ordered_pair :
  ∃ (a b : Int), 
    (√(16 - 12 * cosine_30_eq) = (a:Real) + (b : Real) * secant_30_eq) ∧ 
    (a = 4) ∧ (b = -1) := by
  sorry

end find_ordered_pair_l553_553238


namespace length_60_more_than_breadth_l553_553480

noncomputable def length_more_than_breadth (cost_per_meter : ℝ) (total_cost : ℝ) (length : ℝ) : Prop :=
  ∃ (breadth : ℝ) (x : ℝ), 
    length = breadth + x ∧
    2 * length + 2 * breadth = total_cost / cost_per_meter ∧
    x = length - breadth ∧
    x = 60

theorem length_60_more_than_breadth : length_more_than_breadth 26.5 5300 80 :=
by
  sorry

end length_60_more_than_breadth_l553_553480


namespace chocolates_150_satisfies_l553_553189

def chocolates_required (chocolates : ℕ) : Prop :=
  chocolates ≥ 150 ∧ chocolates % 19 = 17

theorem chocolates_150_satisfies : chocolates_required 150 :=
by
  -- We need to show that 150 satisfies the conditions:
  -- 1. 150 ≥ 150
  -- 2. 150 % 19 = 17
  unfold chocolates_required
  -- Both conditions hold:
  exact And.intro (by linarith) (by norm_num)

end chocolates_150_satisfies_l553_553189


namespace women_in_room_l553_553543

theorem women_in_room (x q : ℕ) (h1 : 4 * x + 2 = 14) (h2 : q = 2 * (5 * x - 3)) : q = 24 :=
by sorry

end women_in_room_l553_553543


namespace area_DEF_l553_553624

-- Define the coordinates of the vertices
def D : ℝ × ℝ := (3, 4)
def E : ℝ × ℝ := (-1, 3)
def F : ℝ × ℝ := (2, -5)

-- Define the function to calculate the area of the triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  0.5 * | x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) |

-- The theorem stating the area of triangle DEF
theorem area_DEF : triangle_area D E F = 17.5 := 
by
  -- sorry is used to skip the actual proof
  sorry

end area_DEF_l553_553624


namespace general_formula_a_n_sum_of_first_n_terms_l553_553674

noncomputable def a_n (n : ℕ) (h : 0 < n) : ℕ :=
if n = 1 then 2 else 2 * n

-- assuming S_n has been given as a precondition S_n = n * (n + 1)
def S_n (n : ℕ) : ℕ := n * (n + 1)

theorem general_formula_a_n (n : ℕ) (h : 0 < n) :
  ∃ f : ℕ → ℕ, (f 1 = 2) ∧ (∀ k, k ≥ 2 → f k = 2 * k) :=
begin
  use a_n,
  split,
  { refl, }, -- for n = 1
  { intros k hk,
    simp [a_n, if_neg (ne_of_gt hk)] }, -- for n ≥ 2
end

theorem sum_of_first_n_terms (n : ℕ) :
  (∑ i in finset.range n, (λ i, ((1 / ((i + 1 : ℕ) * a_n(i + 1) (nat.succ_pos i))))) = (n : ℕ) / (2 * (n + 1))) :=
sorry -- proof omitted

end general_formula_a_n_sum_of_first_n_terms_l553_553674


namespace vector_perpendicular_solution_l553_553317

-- Define the vectors a and b
def vec_a := (-3, 1)
def vec_b (x : ℝ) := (x, 6)

-- Define the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the condition that vectors are perpendicular
def is_perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

-- Given the condition, prove that x = 2
theorem vector_perpendicular_solution : ∃ x : ℝ, is_perpendicular vec_a (vec_b x) ∧ x = 2 :=
by
  sorry

end vector_perpendicular_solution_l553_553317


namespace distinct_numbers_6x6_impossible_l553_553372

theorem distinct_numbers_6x6_impossible :
  ∀(f : ℕ × ℕ → ℕ),
  (∀ x y : ℕ, x < 6 → y < 6 → f (x, y) < 36) →
  (∀ x y₁ y₂ y₃ y₄ y₅, y₁ < 6 → y₂ < 6 → y₃ < 6 → y₄ < 6 → y₅ < 6 →
   list.nodup [f (x, y₁), f (x, y₂), f (x, y₃), f (x, y₄), f (x, y₅)] →
   f (x, y₁) + f (x, y₂) + f (x, y₃) + f (x, y₄) + f (x, y₅) = 2022 ∨
   f (x, y₁) + f (x, y₂) + f (x, y₃) + f (x, y₄) + f (x, y₅) = 2023) →
  (∀ y x₁ x₂ x₃ x₄ x₅, x₁ < 6 → x₂ < 6 → x₃ < 6 → x₄ < 6 → x₅ < 6 →
   list.nodup [f (x₁, y), f (x₂, y), f (x₃, y), f (x₄, y), f (x₅, y)] →
   f (x₁, y) + f (x₂, y) + f (x₃, y) + f (x₄, y) + f (x₅, y) = 2022 ∨
   f (x₁, y) + f (x₂, y) + f (x₃, y) + f (x₄, y) + f (x₅, y) = 2023) →
  false :=
by
  sorry

end distinct_numbers_6x6_impossible_l553_553372


namespace binom_1300_2_l553_553611

theorem binom_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_l553_553611


namespace prob_xi_greater_than_2_l553_553407

noncomputable theory

open MeasureTheory

variable (ξ : MeasureTheory.ProbabilityTheory.RealRandomVar)
variable (σ : ℝ)

axiom normal_distribution_ξ : normal ξ 1 σ
axiom σ_positive : σ > 0
axiom prob_0_to_1 : ProbabilityTheory.Probability (ξ > 0 ∧ ξ < 1) = 0.4

theorem prob_xi_greater_than_2 : ProbabilityTheory.Probability (ξ > 2) = 0.2 :=
by
  sorry

end prob_xi_greater_than_2_l553_553407


namespace nat_solutions_l553_553976

open Nat

theorem nat_solutions (a b c : ℕ) :
  (a ≤ b ∧ b ≤ c ∧ ab + bc + ca = 2 * (a + b + c)) ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 4) :=
by sorry

end nat_solutions_l553_553976


namespace range_of_a_l553_553801

theorem range_of_a (a : ℝ) (p q : Prop) :
  (∀ x : ℝ, ax^2 + ax + 1 ≥ 0) ∧ (∀ y : ℝ, ∃ x : ℝ, log (1 / 2) (ax^2 + 4x + 2) = y) → (0 ≤ a ∧ a ≤ 2) :=
by
  intro h
  sorry

end range_of_a_l553_553801


namespace mathilda_original_amount_l553_553045

theorem mathilda_original_amount (O : ℝ) (initial_installment : ℝ) (remaining_percentage : ℝ) 
  (h_initial_installment : initial_installment = 125) 
  (h_remaining : remaining_percentage = 75) 
  (h_equation : 25 / 100 * O = initial_installment) : 
  O = 500 :=
begin
  -- placeholders for conditions
  sorry
end

end mathilda_original_amount_l553_553045


namespace find_valid_a_l553_553637

noncomputable def solve_system (a b : ℝ) :=
  ∃ x y : ℝ, 
    (arccos ((4 - y) / 4) = arccos ((a + x) / 2)) ∧ 
    (x^2 + y^2 + 2*x - 8*y = b) ∧ 
    (0 ≤ y ∧ y ≤ 8) 

theorem find_valid_a (a : ℝ) : 
  (∀ b : ℝ, ¬ ∃ x1 y1 x2 y2 : ℝ,
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
    solve_system a b x1 y1 ∧ 
    solve_system a b x2 y2) ↔ 
  (a ∈ Iic (-9) ∨ a ∈ Ici 11) := sorry

end find_valid_a_l553_553637


namespace abby_wins_when_N_eq_2011_number_of_brian_winning_strategies_N_le_2011_l553_553584

-- Definition of the game mechanics and winning strategy
def can_win_abby (N : ℕ) : Prop :=
  ∃ strategy : ℕ → ℕ,
    (∀ n, strategy (2 * (n - 1)) = N)
  ∧ strategy 1 = 2011

def count_brian_winning_strategies (N : ℕ) : ℕ :=
  ∑ k in (range (N+1)).filter (λ n, ∃ k ∈ (1 : ℕ) .. (N+1), (2 ^ k = n) ∨ (2 ^ k + 2 = n)), 1

theorem abby_wins_when_N_eq_2011 : can_win_abby 2011 :=
sorry

theorem number_of_brian_winning_strategies_N_le_2011 : count_brian_winning_strategies 2011 = 31 :=
sorry

end abby_wins_when_N_eq_2011_number_of_brian_winning_strategies_N_le_2011_l553_553584


namespace max_value_trig_expr_exists_angle_for_max_value_l553_553236

theorem max_value_trig_expr : ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 :=
sorry

theorem exists_angle_for_max_value : ∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5 :=
sorry

end max_value_trig_expr_exists_angle_for_max_value_l553_553236


namespace find_alpha_l553_553253

-- Definitions based on the conditions in the problem
def vector_a (α : ℝ) : ℝ × ℝ := (1 / 3, 2 * Real.sin α)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.cos α, 3)

-- Parallel condition which must hold
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

-- The main statement close to the mathematical problem
theorem find_alpha (α : ℝ) (h₁ : 0 ≤ α ∧ α ≤ 2 * Real.pi) 
    (h₂ : parallel (vector_a α) (vector_b α)) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
by
  sorry

end find_alpha_l553_553253


namespace cost_per_box_is_070_l553_553880

-- Definition of the box dimensions
def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15

-- Given total volume and total cost
def total_volume : ℕ := 3060000
def total_cost : ℕ := 357

-- Theorem statement
theorem cost_per_box_is_070 :
  let volume_of_box := box_length * box_width * box_height,
      number_of_boxes := total_volume / volume_of_box,
      cost_per_box := total_cost / number_of_boxes in
  cost_per_box = 0.70 := by
  sorry

end cost_per_box_is_070_l553_553880


namespace count_perfect_cubes_l553_553720

theorem count_perfect_cubes (a b : ℤ) (h₁ : 100 < a) (h₂ : b < 1000) : 
  ∃ n m : ℤ, (n^3 > 100 ∧ m^3 < 1000) ∧ m - n + 1 = 5 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end count_perfect_cubes_l553_553720


namespace box_volume_in_cubic_meters_l553_553936

theorem box_volume_in_cubic_meters
  (V_feet : ℝ)
  (feet_per_yard : ℝ)
  (meters_per_yard : ℝ)
  (V_meters : ℝ) :
  V_feet = 216 →
  feet_per_yard = 3 →
  meters_per_yard = 0.9144 →
  V_meters = (216 / (feet_per_yard ^ 3)) * (meters_per_yard ^ 3) →
  V_meters = 6.1168 :=
by
  intros hV hfp hy hm
  rw [hV, hfp, hy] at hm
  exact hm

end box_volume_in_cubic_meters_l553_553936


namespace points_circles_inequality_l553_553869

noncomputable def general_position (n : ℕ) : Prop :=
  ∀ (P : fin n → ℝ × ℝ), ∀ (i j k : fin n), i ≠ j → j ≠ k → i ≠ k → 
  collinear (P i) (P j) (P k) = false

theorem points_circles_inequality (n : ℕ) (k : ℕ)
  (h1 : n > 0)
  (h2 : general_position n)
  (h3 : ∀ (P : fin n → ℝ × ℝ), ∀ (i : fin n), 
        circle_centered_at P i containing_at_least_k_points_of P (k)) :
  k < (1 / 2) + real.sqrt (2 * n) :=
sorry

end points_circles_inequality_l553_553869


namespace round_5632_49999999_l553_553067

def round_to_nearest_integer (x : ℝ) : ℤ :=
  if x - x.floor < 0.5 then x.floor.toInt else x.ceil.toInt

theorem round_5632_49999999 :
  round_to_nearest_integer 5632.49999999 = 5632 :=
by
  sorry

end round_5632_49999999_l553_553067


namespace smallest_n_for_Tn_integer_l553_553032

noncomputable def K : ℚ := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n-1)) * K

theorem smallest_n_for_Tn_integer :
  ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m < n, T_n m ∉ ℤ :=
  sorry

end smallest_n_for_Tn_integer_l553_553032


namespace same_functions_l553_553598

open Function

theorem same_functions :
  (∀ x : ℝ, (λ x, x) x = (λ x, real.cbrt (x ^ 3)) x) ∧
  (∀ x, x ≠ 0 → (λ x, ∣x∣ / x) x = (λ x, if x > 0 then 1 else -1) x) ∧
  (∀ x : ℝ, (λ t, ∣t - 1∣) x = (λ x, ∣x - 1∣) x) :=
  by
    simp
    sorry

end same_functions_l553_553598


namespace simplify_fraction_l553_553443

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l553_553443


namespace sum_of_midpoints_x_coordinates_l553_553502

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l553_553502


namespace sum_of_angles_B_and_D_l553_553673

-- Define the quadrilateral and the conditions
structure Quadrilateral (A B C D F G : Type) :=
(angle_A : ℝ)
(triangle_AFG_is_isosceles : ∠AFG = ∠AGF)
(triangle_BFD_is_isosceles : ∠BFD = 2 * ∠B)

-- Given the conditions on angles
variables {A B C D F G : Type}
variables [Quadrilateral A B C D F G]
variables (h1 : ∠BCD = 40)
variables (h2 : ∠AFG = ∠AGF = 70)
variables (h3 : ∠BFD = 110)

-- Prove that the sum of the angles B and D is 70 degrees
theorem sum_of_angles_B_and_D : ∠B + ∠D = 70 := by sorry

end sum_of_angles_B_and_D_l553_553673


namespace fraction_simplification_l553_553454

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l553_553454


namespace smallest_n_for_T_n_is_integer_l553_553024

def L : ℚ := ∑ i in Finset.range 9, i.succ⁻¹  -- sum of reciprocals of non-zero digits

def D : ℕ := 2^3 * 3^2 * 5 * 7  -- denominator of L in simplified form

def T (n : ℕ) : ℚ := (n * 5^(n-1)) * L + 1  -- expression for T_n

theorem smallest_n_for_T_n_is_integer : ∃ n : ℕ, 0 < n ∧ T n ∈ ℤ ∧ n = 504 :=
by
  use 504
  -- It remains to prove the conditions
  sorry

end smallest_n_for_T_n_is_integer_l553_553024


namespace absolute_value_solution_l553_553333

theorem absolute_value_solution (m : ℤ) (h : abs m = abs (-7)) : m = 7 ∨ m = -7 := by
  sorry

end absolute_value_solution_l553_553333


namespace coefficient_x2_l553_553036

theorem coefficient_x2 (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (hcoeff_x : 2 * m + 3 * n = 13) :
  let f (x : ℝ) := (1 + 2*x)^m + (1 + 3*x)^n in
  (binom m 2 * (2^2) + binom n 2 * (3^2) = 31) ∨ 
  (binom m 5 * (2^2) = 40) :=
sorry

end coefficient_x2_l553_553036


namespace sqrt_sum_identity_l553_553138

theorem sqrt_sum_identity : 
  sqrt ((5 / 2 - (3 * sqrt 3) / 2) ^ 2) + sqrt ((5 / 2 + (3 * sqrt 3) / 2) ^ 2) = 5 := 
by
  sorry

end sqrt_sum_identity_l553_553138


namespace tangent_line_at_neg2_monotonicity_and_extremum_range_of_a_l553_553297

noncomputable def f (a x : ℝ) := - (1 / 3) * x^3 + 2 * a * x^2 - 3 * a^2 * x
noncomputable def f' (a x : ℝ) := - x^2 + 4 * a * x - 3 * a^2

theorem tangent_line_at_neg2 (a : ℝ) (ha : a = -1) : 
  let line_eq := 3*x - 3*y + 8 = 0 in 
  ∀ x y : ℝ, y = f a x → (x = -2 ∧ y = f a (-2)) → line_eq :=
begin
  sorry
end

theorem monotonicity_and_extremum (a : ℝ) (ha : 0 < a) :
  (∀ x : ℝ, (a < x ∧ x < 3 * a) → 0 < f' a x) ∧
  (∀ x : ℝ, (x < a ∨ 3 * a < x) → f' a x < 0) ∧
  (f a (3 * a) = 0) ∧
  (f a a = - (4 / 3) * a^3) :=
begin
  sorry
end

theorem range_of_a (a : ℝ) :
  (1 ≤ a ∧ a ≤ 3) ↔ (∀ x : ℝ, (2*a ≤ x ∧ x ≤ 2*a + 2) → |f' a x| ≤ 3*a) :=
begin
  sorry
end

end tangent_line_at_neg2_monotonicity_and_extremum_range_of_a_l553_553297


namespace count_numbers_with_9_in_hundreds_place_divisible_by_7_l553_553985

theorem count_numbers_with_9_in_hundreds_place_divisible_by_7 :
  let count := (range 900 1000).filter (λ n, n % 7 = 0).length
  in count = 14 :=
by
  sorry

end count_numbers_with_9_in_hundreds_place_divisible_by_7_l553_553985


namespace simplify_fraction_l553_553449

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l553_553449


namespace angle_C_is_90_l553_553087

theorem angle_C_is_90 (A B C : ℝ) (hABC : A + B + C = π)
    (h1 : sin A + cos B = sqrt 2)
    (h2 : cos A + sin B = sqrt 2) : C = π / 2 :=
  sorry

end angle_C_is_90_l553_553087


namespace log_geom_seq_l553_553680

noncomputable theory
open_locale big_operators

variable {α : Type*} [linear_ordered_field α] [algebra ℚ α]

def geom_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = r * a n

theorem log_geom_seq (a : ℕ → ℝ) (r : ℝ) (hf : geom_seq a r) (pos_a : ∀ n, 0 < a n)
  (h : a 4 * a 5 = 81) :
  ∑ i in finset.range 10, log (1/3 : ℝ) (a i) = -20 :=
sorry

end log_geom_seq_l553_553680


namespace z_in_second_quadrant_l553_553614

-- Define the complex number z as given in the problem
noncomputable def z : ℂ := Complex.sin (-π / 7) + Complex.i * Complex.cos (-π / 7)

-- Define the necessary conditions
def conditions : Prop :=
  Complex.sin (-π / 7) < 0 ∧ Complex.cos (-π / 7) > 0

-- The theorem to prove that z is in the second quadrant
theorem z_in_second_quadrant : conditions → (Complex.re z < 0 ∧ Complex.im z > 0) := by
  sorry

end z_in_second_quadrant_l553_553614


namespace fraction_evaluation_l553_553631

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by
  sorry

end fraction_evaluation_l553_553631


namespace double_line_chart_analysis_l553_553925

theorem double_line_chart_analysis :
  (∀ (chart : Type), (shows_changes_in_quantities chart) 
  → (shows_and_analyzes_differences chart)) :=
by
  intros chart h
  sorry

-- Definitions of the properties (these should correspond directly to the problem's conditions)
def shows_changes_in_quantities (chart : Type) : Prop :=
  -- definition based on the problem's conditions
  chart = double_line_chart -- This is a simplified placeholder

def shows_and_analyzes_differences (chart : Type) : Prop :=
  -- definition based on the correct answer
  chart = double_line_chart -- This is a simplified placeholder

end double_line_chart_analysis_l553_553925


namespace integer_count_of_sqrt_x_l553_553849

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end integer_count_of_sqrt_x_l553_553849


namespace find_q_l553_553779

open Polynomial

-- Define the conditions for the roots of the first polynomial
def roots_of_first_eq (a b m : ℝ) (h : a * b = 3) : Prop := 
  ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)

-- Define the problem statement
theorem find_q (a b m p q : ℝ) 
  (h1 : a * b = 3) 
  (h2 : ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)) 
  (h3 : ∀ x, (x^2 - p*x + q) = (x - (a + 2/b)) * (x - (b + 2/a))) :
  q = 25 / 3 :=
sorry

end find_q_l553_553779


namespace prob_AB_diff_homes_l553_553602

-- Define the volunteers
inductive Volunteer : Type
| A | B | C | D | E

open Volunteer

-- Define the homes
inductive Home : Type
| home1 | home2

open Home

-- Total number of ways to distribute the volunteers
def total_ways : ℕ := 2^5  -- Each volunteer has independently 2 choices

-- Number of ways in which A and B are in different homes
def diff_ways : ℕ := 2 * 4 * 2^3  -- Split the problem down by cases for simplicity

-- Calculate the probability
def probability : ℚ := diff_ways / total_ways

-- The final statement to prove
theorem prob_AB_diff_homes : probability = 8 / 15 := sorry

end prob_AB_diff_homes_l553_553602


namespace arithmetic_sequence_sum_l553_553195

theorem arithmetic_sequence_sum :
  ∀ (a d : ℝ), 
    (a + 2017 * d = 100) →
    let t_2000 := a + 1999 * d in
    let t_2015 := a + 2014 * d in
    let t_2021 := a + 2020 * d in
    let t_2036 := a + 2035 * d in
    t_2000 + 5 * t_2015 + 5 * t_2021 + t_2036 = 1200 :=
begin
  sorry
end

end arithmetic_sequence_sum_l553_553195


namespace picture_books_count_l553_553517

theorem picture_books_count :
  ∀ (total_books fiction_books non_fiction_books autobiographies picture_books: ℕ),
  total_books = 35 →
  fiction_books = 5 →
  non_fiction_books = fiction_books + 4 →
  autobiographies = 2 * fiction_books →
  picture_books = total_books - (fiction_books + non_fiction_books + autobiographies) →
  picture_books = 11 :=
by
  intro total_books fiction_books non_fiction_books autobiographies picture_books
  intro h1 h2 h3 h4 h5
  rw [h1, h2] at h5
  rw h3 at h5
  rw h4 at h5
  norm_num at h5
  assumption

end picture_books_count_l553_553517


namespace gcd_of_36_and_12_is_8_l553_553865

-- Given conditions
def n := 36
def m := 12
def lcm_val := 54

-- Definition using given conditions
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Theorem statement to prove
theorem gcd_of_36_and_12_is_8 (h1 : n = 36) (h2 : lcm n m = lcm_val) : gcd n m = 8 := by
  sorry

end gcd_of_36_and_12_is_8_l553_553865


namespace moving_circle_trajectory_l553_553738

theorem moving_circle_trajectory (x y : ℝ) 
  (fixed_circle : x^2 + y^2 = 4): 
  (x^2 + y^2 = 9) ∨ (x^2 + y^2 = 1) :=
sorry

end moving_circle_trajectory_l553_553738


namespace scientific_notation_239000000_l553_553961

theorem scientific_notation_239000000 :
  239000000 = 2.39 * 10^8 :=
begin
  sorry
end

end scientific_notation_239000000_l553_553961


namespace simplify_fraction_l553_553445

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l553_553445


namespace yeast_population_growth_no_control_experiment_needed_even_distribution_shaking_test_tube_yeast_count_in_culture_correct_student_operations_l553_553438

-- Step 1: Experimental hypothesis about yeast population growth
theorem yeast_population_growth (sufficient_nutrients : Prop) (suitable_temperature : Prop) 
(no_enemies : Prop) (no_competition : Prop) : 
  yeast_population_growth initially_J_type → yeast_population_growth later_S_type :=
sorry

-- Step 2: No need for a control experiment due to self-comparison
theorem no_control_experiment_needed (yeast_population_changes : Prop): 
  yeast_population_comparison_adequate :=
sorry

-- Step 3: Purpose of shaking test tube for even yeast distribution
theorem even_distribution_shaking_test_tube 
(uneven_yeast_distribution : Prop): 
  shaking_test_tube → even_yeast_distribution :=
sorry

-- Step 4: Calculation of total number of yeast in 1mL culture medium
theorem yeast_count_in_culture (a : ℕ) 
(volume_of_small_square : ℝ := 0.4) : 
  yeast_count_1ml a volume_of_small_square = 2.5 * 10^3 :=
sorry

-- Step 5: Correct operations during student experiment
theorem correct_student_operations 
(operation_1 : Prop) (operation_2 : Prop) 
(operation_3 : Prop) (operation_4 : Prop) 
(operation_5 : Prop) : 
  valid_operations_hypothesis → 
  (operation_1 ∧ operation_4 ∧ operation_5) :=
sorry

end yeast_population_growth_no_control_experiment_needed_even_distribution_shaking_test_tube_yeast_count_in_culture_correct_student_operations_l553_553438


namespace integer_count_of_sqrt_x_l553_553850

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end integer_count_of_sqrt_x_l553_553850


namespace af_leq_bf_l553_553688

-- Given conditions
variables {f : ℝ → ℝ} {a b : ℝ}
hypothesis f_differentiable : ∀ x, 0 < x → differentiable_at ℝ f x
hypothesis f_nonnegative : ∀ x, 0 < x → 0 ≤ f x
hypothesis f_double_deriv_leq : ∀ x, 0 < x → x * (deriv (deriv f x)) + f x ≤ 0
hypothesis ab_pos : 0 < a ∧ a < b

-- Statement to prove
theorem af_leq_bf : a * f b ≤ b * f a :=
sorry

end af_leq_bf_l553_553688


namespace sum_of_midpoints_l553_553495

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l553_553495


namespace derivative_f_l553_553997

noncomputable def f (x : Real) : Real :=
  cot (cos 2) + (1 / 6) * (sin (6 * x)) ^ 2 / (cos (12 * x))

theorem derivative_f (x : Real) : deriv f x = (tan (12 * x)) / (cos (12 * x)) :=
by
  sorry

end derivative_f_l553_553997


namespace Buratino_can_solve_l553_553414

theorem Buratino_can_solve :
  ∃ (MA TE TI KA : ℕ), MA ≠ TE ∧ MA ≠ TI ∧ MA ≠ KA ∧ TE ≠ TI ∧ TE ≠ KA ∧ TI ≠ KA ∧
  MA * TE * MA * TI * KA = 2016000 :=
by
  -- skip the proof using sorry
  sorry

end Buratino_can_solve_l553_553414


namespace area_ratio_of_inscribed_circles_l553_553181

theorem area_ratio_of_inscribed_circles (s : ℝ) (h : s > 0) :
  let R := s * Real.sqrt 2 / 2,
      r := s / 2,
      A_original := π * R ^ 2,
      A_second := π * r ^ 2
  in A_second / A_original = 1 / 2 := by
  sorry

end area_ratio_of_inscribed_circles_l553_553181


namespace Fred_candy_bars_l553_553653

theorem Fred_candy_bars (F : ℕ) (H1 : ∀ b : ℕ, b = F + 6)
                         (H2 : ∀ t : ℕ, t = 2 * F + 6)
                         (H3 : ∀ j : ℕ, j = 10 * (2 * F + 6))
                         (H4 : 0.40 * (10 * (2 * F + 6)) = 120) : 
  F = 12 := 
sorry

end Fred_candy_bars_l553_553653


namespace polygon_sides_l553_553509

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
sorry

end polygon_sides_l553_553509


namespace sea_hidden_by_cloud_l553_553251

-- Define the conditions
def cloud_occupies := (1 : ℝ) / 2
def cloud_hides_island := (1 : ℝ) / 4
def visible_island := (1 : ℝ) / 4

-- Theorem statement to prove the portion of the sea hidden by the cloud.
theorem sea_hidden_by_cloud : 
  (visible_island / (3 / 4) - visible_island + (1 - cloud_occupies) / 2 : ℝ) = (5 / 12 : ℝ) :=
sorry

end sea_hidden_by_cloud_l553_553251


namespace monotonic_interval_f_f_l553_553660

noncomputable def f (x a : ℝ) := Real.exp x + 2 * a * x
noncomputable def g (x : ℝ) := abs (Real.e / x - Real.log x) + Real.log x

theorem monotonic_interval_f (a : ℝ) (h : a = -Real.exp 2 / 2) :
  ∃ x (h1 : 1 < x) (h2 : x < 2), deriv (f x a) < 0 ∧ ∃ x (h3 : 2 < x) (h4 : x <orial), deriv (f x a) > 0 :=
sorry

theorem f'_gt_g_plus_a (a x : ℝ) (ha : a ∈ Ioi 2) (hx : x ∈ Ioi 1) : 
  deriv (f (x - 1) a) > g x + a :=
sorry

end monotonic_interval_f_f_l553_553660


namespace slope_of_line_l553_553707

noncomputable def parabola (p : ℝ) (p_pos : p > 0) : set (ℝ × ℝ) :=
  {P : ℝ × ℝ | P.snd^2 = 2 * p * P.fst}

def intersects (p : ℝ) (p_pos : p > 0) (F : ℝ × ℝ) (l : set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ∈ parabola p p_pos ∧ B ∈ parabola p p_pos ∧ A ≠ B ∧
    ∃ k : ℝ, l = {P : ℝ × ℝ | P.snd = k * (P.fst - F.fst) + F.snd}

def ratio_AF_BF (A B F : ℝ × ℝ) : Prop :=
  real.dist (A.fst, A.snd) F = 3 * real.dist B F

theorem slope_of_line 
  (p : ℝ) (p_pos : p > 0) (F : ℝ × ℝ) (l : set (ℝ × ℝ))
  (h1 : intersects p p_pos F l) 
  (h2 : ∃ A B : ℝ × ℝ, ratio_AF_BF A B F) :
  ∃ m : ℝ, m = sqrt 3 ∨ m = - sqrt 3 :=
by
  sorry

end slope_of_line_l553_553707


namespace binom_1300_2_eq_844350_l553_553613

theorem binom_1300_2_eq_844350 : (Nat.choose 1300 2) = 844350 := by
  sorry

end binom_1300_2_eq_844350_l553_553613


namespace non_vectors_correct_l553_553951

-- Define physical quantities
inductive PhysicalQuantity
| mass
| velocity
| displacement
| force
| acceleration
| distance
| density
| work

open PhysicalQuantity

-- Define non-vectors
def non_vectors : set PhysicalQuantity := {mass, distance, density, work}

-- Define a function to check if a quantity is in the set of non-vectors
def is_non_vector (q : PhysicalQuantity) : bool :=
  q ∈ non_vectors

-- The theorem that states the solution
theorem non_vectors_correct :
  ∀ q, q ∈ non_vectors → ¬is_vector q :=
by
  intros q h 
  cases q
  sorry

end non_vectors_correct_l553_553951


namespace sum_of_midpoints_l553_553496

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l553_553496


namespace sign_of_k_l553_553366

variable (k x y : ℝ)
variable (A B : ℝ × ℝ)
variable (y₁ y₂ : ℝ)
variable (h₁ : A = (-2, y₁))
variable (h₂ : B = (5, y₂))
variable (h₃ : y₁ = k / -2)
variable (h₄ : y₂ = k / 5)
variable (h₅ : y₁ > y₂)
variable (h₀ : k ≠ 0)

-- We need to prove that k < 0
theorem sign_of_k (A B : ℝ × ℝ) (y₁ y₂ k : ℝ) 
  (h₁ : A = (-2, y₁)) 
  (h₂ : B = (5, y₂)) 
  (h₃ : y₁ = k / -2) 
  (h₄ : y₂ = k / 5) 
  (h₅ : y₁ > y₂) 
  (h₀ : k ≠ 0) : k < 0 := 
by
  sorry

end sign_of_k_l553_553366


namespace intersection_M_N_l553_553309

-- Define the set M
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the condition for set N
def N : Set ℤ := {x | x + 2 ≥ x^2}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l553_553309


namespace root_conditions_l553_553708

-- Given conditions and definitions:
def quadratic_eq (m x : ℝ) : ℝ := x^2 + (m - 3) * x + m

-- The proof problem statement
theorem root_conditions (m : ℝ) (h1 : ∃ x y : ℝ, quadratic_eq m x = 0 ∧ quadratic_eq m y = 0 ∧ x > 1 ∧ y < 1) : m < 1 :=
sorry

end root_conditions_l553_553708


namespace tan_pos_iff_sin2x_pos_l553_553687

theorem tan_pos_iff_sin2x_pos {x : ℝ} : tan x > 0 ↔ sin (2 * x) > 0 :=
by
  -- Proof will go here
  sorry

end tan_pos_iff_sin2x_pos_l553_553687


namespace largest_angle_in_pentagon_l553_553357

theorem largest_angle_in_pentagon {R S : ℝ} (h₁: R = S) 
  (h₂: (75 : ℝ) + 110 + R + S + (3 * R - 20) = 540) : 
  (3 * R - 20) = 217 :=
by {
  -- Given conditions are assigned and now we need to prove the theorem, the proof is omitted
  sorry
}

end largest_angle_in_pentagon_l553_553357


namespace find_angle_l553_553599

-- Given definitions:
def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

-- Condition:
def condition (α : ℝ) : Prop :=
  supplement α = 3 * complement α + 10

-- Statement to prove:
theorem find_angle (α : ℝ) (h : condition α) : α = 50 :=
sorry

end find_angle_l553_553599


namespace three_digit_increasing_or_decreasing_l553_553952

theorem three_digit_increasing_or_decreasing :
  ∑ k in (finset.range 10).powerset, if (k.card = 3 ∧ (∀ i j ∈ k, i < j ∨ i > j)) then 1 else 0 = 204 := 
by
  -- The proof goes here
  sorry

end three_digit_increasing_or_decreasing_l553_553952


namespace sum_of_powers_of_i_is_negative_one_l553_553210

axiom i : ℂ -- Complex number i
axiom h_i : i^2 = -1

theorem sum_of_powers_of_i_is_negative_one :
  (∑ k in Finset.range 604, i ^ k) = -1 :=
by sorry

end sum_of_powers_of_i_is_negative_one_l553_553210


namespace area_of_region_inside_D_outside_circles_l553_553988

-- Definitions of condition in Lean
noncomputable def radius_D : ℝ := 40
noncomputable def radius_circle (s : ℝ) (tangent_to_D : s + s * Real.sqrt 2 = radius_D) : ℝ := s
noncomputable def M (s : ℝ) (tangent_to_D : s + s * Real.sqrt 2 = radius_D) : ℝ :=
  π * (radius_D ^ 2 - 8 * (s ^ 2))

-- The formulated proof problem statement
theorem area_of_region_inside_D_outside_circles :
  (∀ s : ℝ, s + s * Real.sqrt 2 = radius_D → ⌊M s (by assumption)⌋ = -26175) :=
by
  sorry

end area_of_region_inside_D_outside_circles_l553_553988


namespace problem_statement_l553_553327

noncomputable def value_of_expression (θ : ℝ) : ℝ :=
  sqrt((1 + Real.cos θ) / (1 - Real.sin (Real.pi / 2 - θ))) - 
  sqrt((1 - Real.cos θ) / (1 + Real.sin (θ - 3 * Real.pi / 2)))

theorem problem_statement (θ : ℝ) 
  (h1 : Real.pi / 2 < θ ∧ θ < Real.pi) 
  (h2 : Real.tan (θ - Real.pi) = -1 / 2) : 
  value_of_expression θ = -4 := 
by
  sorry  -- Proof is omitted.

end problem_statement_l553_553327


namespace x_intercepts_are_irrational_l553_553061

theorem x_intercepts_are_irrational {p q : ℤ} (hp : odd p) (hq : odd q) :
  ∀ x : ℝ, x ∈ {r : ℝ | r^2 - 2 * p * r + 2 * q = 0} → irrational x :=
begin
  intros x hx,
  -- Proof omitted
  sorry
end

end x_intercepts_are_irrational_l553_553061


namespace max_bench_weight_support_l553_553774

/-- Definitions for the given problem conditions -/
def john_weight : ℝ := 250
def bar_weight : ℝ := 550
def total_weight : ℝ := john_weight + bar_weight
def safety_percentage : ℝ := 0.80

/-- Theorem stating the maximum weight the bench can support given the conditions -/
theorem max_bench_weight_support :
  ∀ (W : ℝ), safety_percentage * W = total_weight → W = 1000 :=
by
  sorry

end max_bench_weight_support_l553_553774


namespace baker_sold_158_cakes_l553_553963

variable (C P : ℕ)

theorem baker_sold_158_cakes (h1 : P = 147) (h2 : C = P + 11) : C = 158 := by
  rw [h1] at h2
  rw [h2]
  exact rfl

end baker_sold_158_cakes_l553_553963


namespace exist_78_lines_with_1992_intersections_l553_553060

theorem exist_78_lines_with_1992_intersections :
  ∃ (lines : Finset (Set (ℝ × ℝ))),
  lines.card = 78 ∧ 
  let I := { p : ℝ × ℝ | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ p ∈ l1 ∧ p ∈ l2 } in
  I.card = 1992 := by
  sorry

end exist_78_lines_with_1992_intersections_l553_553060


namespace calculate_income_l553_553216

theorem calculate_income (I : ℝ) (T : ℝ) (a b c d : ℝ) (h1 : a = 0.15) (h2 : b = 40000) (h3 : c = 0.20) (h4 : T = 8000) (h5 : T = a * b + c * (I - b)) : I = 50000 :=
by
  sorry

end calculate_income_l553_553216


namespace math_problem_l553_553035

noncomputable def g (x : ℝ) : ℝ := sorry

theorem math_problem (g : ℝ → ℝ)
    (h : ∀ (x y z : ℝ), g (x^2 + y * g(z)) = x * g(x) + z^2 * g(y)) :
    let n := 2 in
    let s := 9 in
    n * s = 18 := by
    sorry

end math_problem_l553_553035


namespace simplify_fraction_l553_553448

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l553_553448


namespace integral_evaluation_l553_553547

noncomputable def integral_problem : ℝ :=
  ∫ x in 0..π, 16 * (Real.sin x)^6 * (Real.cos x)^2

theorem integral_evaluation : integral_problem = (5 * π) / 8 :=
by sorry

end integral_evaluation_l553_553547


namespace find_amount_with_R_l553_553893

variable (P_amount Q_amount R_amount : ℝ)
variable (total_amount : ℝ) (r_has_twothirds : Prop)

noncomputable def amount_with_R (total_amount : ℝ) : ℝ :=
  let R_amount := 2 / 3 * (total_amount - R_amount)
  R_amount

theorem find_amount_with_R (P_amount Q_amount R_amount : ℝ) (total_amount : ℝ)
  (h_total : total_amount = 5000)
  (h_two_thirds : R_amount = 2 / 3 * (P_amount + Q_amount)) :
  R_amount = 2000 := by sorry

end find_amount_with_R_l553_553893


namespace time_left_to_room_l553_553049

theorem time_left_to_room (total_time minutes_to_gate minutes_to_building : ℕ) 
  (h1 : total_time = 30) 
  (h2 : minutes_to_gate = 15) 
  (h3 : minutes_to_building = 6) : 
  total_time - (minutes_to_gate + minutes_to_building) = 9 :=
by 
  sorry

end time_left_to_room_l553_553049


namespace student_count_l553_553825

noncomputable def numberOfStudents (decreaseInAverageWeight totalWeightDecrease : ℕ) : ℕ :=
  totalWeightDecrease / decreaseInAverageWeight

theorem student_count 
  (decreaseInAverageWeight : ℕ)
  (totalWeightDecrease : ℕ)
  (condition_avg_weight_decrease : decreaseInAverageWeight = 4)
  (condition_weight_difference : totalWeightDecrease = 92 - 72) :
  numberOfStudents decreaseInAverageWeight totalWeightDecrease = 5 := by 
  -- We are not providing the proof details as per the instruction
  sorry

end student_count_l553_553825


namespace simplify_expression_l553_553071

variable (x y : ℝ)

theorem simplify_expression
  (hx : x ≠ y)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y) :
  ( (x - y) / (x^(3/4) + x^(1/2) * y^(1/4)) * 
    (x^(1/2) * y^(1/4) + x^(1/4) * y^(1/2)) / (x^(1/2) + y^(1/2)) * 
    (x^(1/4) * y^(-1/4)) / (x^(1/2) - 2 * x^(1/4) * y^(1/4) + y^(1/2)) 
  ) = ((x^(1/4) + y^(1/4)) / (x^(1/4) - y^(1/4))) :=
  sorry

end simplify_expression_l553_553071


namespace max_abs_value_inequality_l553_553397

theorem max_abs_value_inequality (a b : ℝ)
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ (a b : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) ∧ |20 * a + 14 * b| + |20 * a - 14 * b| = 80 := 
sorry

end max_abs_value_inequality_l553_553397


namespace opposite_of_neg_five_l553_553839

theorem opposite_of_neg_five : ∃ y, (-5) + y = 0 ∧ y = 5 :=
by {
  use 5,
  split,
  { linarith, },
  { refl, }
}

end opposite_of_neg_five_l553_553839


namespace necessary_but_not_sufficient_l553_553271

section proof_problem

variables {α β : Type} [plane α] [plane β] {l : Type} [line l] 

-- Assuming there is a subset relation l ⊆ α
axiom line_subset_plane (l : line) (α : plane) : l ⊆ α

-- Definition for parallel relation between line and plane
@[class] def line_parallel_plane (l : line) (β : plane) : Prop := sorry 

-- Definition for parallel relation between two planes
@[class] def plane_parallel_plane (α : plane) (β : plane) : Prop := sorry

-- Main statement to be proven
theorem necessary_but_not_sufficient (l : line) (α β : plane) 
  (H1 : line_subset_plane l α)
  (H2 : line_parallel_plane l β) :
  (plane_parallel_plane α β ∧ ¬plane_parallel_plane β α) :=
sorry

end proof_problem

end necessary_but_not_sufficient_l553_553271


namespace repeating_decimal_as_fraction_l553_553633

theorem repeating_decimal_as_fraction :
  ∃ x : ℚ, x = 6 / 10 + 7 / 90 ∧ x = 61 / 90 :=
by
  sorry

end repeating_decimal_as_fraction_l553_553633


namespace smallest_n_for_T_integer_l553_553016

noncomputable def J := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def T (n : ℕ) : ℚ := ∑ x in finset.range (5^n + 1), 
  ∑ d in {
    digit | digit.to_nat ≠ 0 ∧ digit.to_nat < 10 
  }, (1 : ℚ) / (digit.to_nat : ℚ)

theorem smallest_n_for_T_integer : ∃ n : ℕ, T n ∈ ℤ ∧ ∀ m : ℕ, T m ∈ ℤ → 63 ≤ n :=
by {
  sorry
}

end smallest_n_for_T_integer_l553_553016


namespace find_LN_l553_553085

noncomputable def LM : ℝ := 25
noncomputable def sinN : ℝ := 4 / 5

theorem find_LN (LN : ℝ) (h_sin : sinN = LM / LN) : LN = 125 / 4 :=
by
  sorry

end find_LN_l553_553085


namespace coefficient_x2_expansion_l553_553472

theorem coefficient_x2_expansion :
  let p := (1 + 2 * x) ^ 3
  let q := (1 - x) ^ 4
  ∑ (i j : ℕ) in finset.Icc 0 3, 
    finset.Icc 0 4, 
    if i + j = 2 then (p.coeff i) * (q.coeff j) else 0 
  = -6 :=
by
  sorry

end coefficient_x2_expansion_l553_553472


namespace smallest_common_multiple_of_5_and_13_l553_553053

theorem smallest_common_multiple_of_5_and_13 : ∃ n : ℕ, n % 5 = 0 ∧ n % 13 = 0 ∧ ∀ m : ℕ, (m % 5 = 0 ∧ m % 13 = 0) → n ≤ m :=
by
  use 65
  split
  show 65 % 5 = 0 by sorry
  split
  show 65 % 13 = 0 by sorry
  intros m hm
  sorry

end smallest_common_multiple_of_5_and_13_l553_553053


namespace correct_choice_D_l553_553681

variables {Ω : Type*} {P : ProbabilityMassFunction Ω}

def A : Set Ω := sorry -- define event A
def B : Set Ω := sorry -- define event B

theorem correct_choice_D (hA : P(A) = 0.2) (hB : P(B) = 0.8) (hInd : P(A ∩ B) = P(A) * P(B)) :
    P(A ∪ B) = 0.84 ∧ P(A ∩ B) = 0.16 :=
by
  rw hInd
  have hAB : P(A ∩ B) = 0.2 * 0.8, by rw [hA, hB]; ring
  have hUnion : P(A ∪ B) = P(A) + P(B) - P(A ∩ B), by sorry -- Add calculation steps as needed
  rw [hAB, hA, hB] at hUnion
  norm_num at hUnion
  split
  · exact hUnion
  · exact hAB

end correct_choice_D_l553_553681


namespace find_omega_l553_553298

noncomputable def f (x : ℝ) (A ω φ : ℝ) : ℝ := A * Real.cos (ω * x + φ)

theorem find_omega (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → Real.monotone (f x A ω φ)) ∧
  f (-Real.pi / 3) A ω φ = f 0 A ω φ ∧
  f 0 A ω φ = -f (2 * Real.pi / 3) A ω φ → ω = 6 / 7 :=
begin
  sorry
end

end find_omega_l553_553298


namespace max_satiated_pikes_l553_553519

-- Define the total number of pikes
def total_pikes : ℕ := 30

-- Define the condition for satiation
def satiated_condition (eats : ℕ) : Prop := eats ≥ 3

-- Define the number of pikes eaten by each satiated pike
def eaten_by_satiated_pike : ℕ := 3

-- Define the theorem to find the maximum number of satiated pikes
theorem max_satiated_pikes (s : ℕ) : 
  (s * eaten_by_satiated_pike < total_pikes) → s ≤ 9 :=
by
  sorry

end max_satiated_pikes_l553_553519


namespace probability_sum_two_numbers_l553_553522

noncomputable def probability_sum_less_than_six_fifths : ℝ :=
  let omega := set.prod (set.Ioo 0 1) (set.Ioo 0 1)
  let A := {p : ℝ × ℝ | p.1 + p.2 < 6 / 5} ∩ omega
  classical.some (measure_theory.measure_space.volume A / measure_theory.measure_space.volume omega)

theorem probability_sum_two_numbers :
  probability_sum_less_than_six_fifths = 23 / 25 :=
by
  sorry

end probability_sum_two_numbers_l553_553522


namespace ice_cream_cones_l553_553425

theorem ice_cream_cones (cost_per_cone total_cost : ℕ) (h_cost : cost_per_cone = 99) (h_total : total_cost = 198) : total_cost / cost_per_cone = 2 :=
by
  rw [h_cost, h_total]
  norm_num

end ice_cream_cones_l553_553425


namespace smallest_number_l553_553194

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1/2) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d := by
  sorry

end smallest_number_l553_553194


namespace stratified_sampling_total_count_l553_553119

-- Definition encoding the conditions of the problem
def stratified_sampling_condition (a : ℕ) : Prop :=
  a > 0 ∧
  let solutions := (35 * (a + 500) = 45 * (a + 300)) in -- Proportion constraint from solution steps
  solutions

theorem stratified_sampling_total_count :
  ∀ (a : ℕ), 
    stratified_sampling_condition a → 
    let N := a + 300 + 200
    N = 900 :=
by
  intros a h
  let N := a + 300 + 200
  sorry

end stratified_sampling_total_count_l553_553119


namespace range_abs_sub_four_l553_553105

theorem range_abs_sub_four : 
  (∀ x : ℝ, -4 ≤ |x| - 4 ∧ ∃ y : ℝ, y = |x| - 4) → 
  (set.range (λ x : ℝ, |x| - 4) = set.Ici (-4)) :=
by { sorry }

end range_abs_sub_four_l553_553105


namespace problem_statement_l553_553907

theorem problem_statement :
  ∀ x a k n : ℤ, 
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n → a - n + k = 3 :=
by  
  sorry

end problem_statement_l553_553907


namespace largest_value_l553_553882

noncomputable def a : ℕ := 2 ^ 6
noncomputable def b : ℕ := 3 ^ 5
noncomputable def c : ℕ := 4 ^ 4
noncomputable def d : ℕ := 5 ^ 3
noncomputable def e : ℕ := 6 ^ 2

theorem largest_value : c > a ∧ c > b ∧ c > d ∧ c > e := by
  sorry

end largest_value_l553_553882


namespace largest_of_three_numbers_l553_553862

noncomputable def largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -20) : ℝ :=
  max p (max q r)

theorem largest_of_three_numbers (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -8) 
  (h3 : p * q * r = -20) :
  largest_root p q r h1 h2 h3 = ( -1 + Real.sqrt 21 ) / 2 :=
by
  sorry

end largest_of_three_numbers_l553_553862


namespace color_pairings_correct_l553_553206

noncomputable def num_color_pairings (bowls : ℕ) (glasses : ℕ) : ℕ :=
  bowls * glasses

theorem color_pairings_correct : 
  num_color_pairings 4 5 = 20 :=
by 
  -- proof omitted
  sorry

end color_pairings_correct_l553_553206


namespace total_fruit_in_buckets_l553_553125

theorem total_fruit_in_buckets (A B C : ℕ) 
  (h1 : A = B + 4)
  (h2 : B = C + 3)
  (h3 : C = 9) :
  A + B + C = 37 := by
  sorry

end total_fruit_in_buckets_l553_553125


namespace pairwise_function_equivalence_l553_553596

noncomputable def f_A (x : ℝ) : ℝ := x
noncomputable def g_A (x : ℝ) : ℝ := real.cbrt (x ^ 3)

noncomputable def f_B (x : ℝ) : ℝ := x + 1
noncomputable def g_B (x : ℝ) : ℝ := if x = 1 then 0 else (x ^ 2 - 1) / (x - 1)

noncomputable def f_C (x : ℝ) : ℝ := abs x / x
noncomputable def g_C (x : ℝ) : ℝ := if x > 0 then 1 else if x < 0 then -1 else 0

noncomputable def f_D (t : ℝ) : ℝ := abs (t - 1)
noncomputable def g_D (x : ℝ) : ℝ := abs (x - 1)

theorem pairwise_function_equivalence :
  (∀ x, f_A x = g_A x) ∧
  (∀ x, f_B x = g_B x → x ≠ 1) ∧
  (∀ x, f_C x = g_C x) ∧
  (∀ t, f_D t = g_D t) :=
by
  sorry

end pairwise_function_equivalence_l553_553596


namespace problem_equiv_statements_l553_553305

variable {R : Type} [LinearOrderedField R]

-- Define the line equation
def line (a b : R) (x y : R) := a * (2 * x + 3 * y + 2) + b * (x - 2 * y - 6)

-- Condition: a ≠ 0, b ≠ 0
variable {a b : R} (hne : a ≠ 0 ∧ b ≠ 0)

-- Problem statement
theorem problem_equiv_statements :
  -- Condition 1: if l passes through origin
  (line a b 0 0 = 0 → a = -3 * b) ∧
  -- Condition 2: if the sum of intercepts on the two axes is 0 and a ≠ 3b
  ((∃ m n : R, m + n = 0) → a ≠ 3 * b → b = 5 * a) ∧
  -- Condition 3: if l is tangent to the circle centered at (3, 0) with radius √5
  (∀ d : R, abs (3 * (line a b 3 0)) = sqrt 5 * d → (2 * a + b) ^ 2 + (3 * a - 2 * b) ^ 2 = 5 → a = -4 * b) :=
sorry

end problem_equiv_statements_l553_553305


namespace percent_decrease_apr_to_may_l553_553488

variables (P x : ℝ)
axioms (h1 : ∀ P, 1.50 * (1.40 * P - x / 100 * 1.40 * P) = 1.68 * P)

theorem percent_decrease_apr_to_may : x = 20 :=
by
  have h : 1.50 * (1.40 * P - x / 100 * 1.40 * P) = 1.68 * P := h1 P
  -- Apply algebraic manipulation steps here (omitted with sorry for brevity)
  sorry

end percent_decrease_apr_to_may_l553_553488


namespace range_of_m_l553_553685

-- Define the conditions:

/-- Proposition p: the equation represents an ellipse with foci on y-axis -/
def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 9 ∧ 9 - m > 2 * m ∧ 2 * m > 0

/-- Proposition q: the eccentricity of the hyperbola is in the interval (\sqrt(3)/2, \sqrt(2)) -/
def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ (5 / 2 < m ∧ m < 5)

def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def p_and_q (m : ℝ) : Prop := proposition_p m ∧ proposition_q m

-- Mathematically equivalent proof problem in Lean 4:

theorem range_of_m (m : ℝ) : (p_or_q m ∧ ¬p_and_q m) ↔ (m ∈ Set.Ioc 0 (5 / 2) ∪ Set.Icc 3 5) := sorry

end range_of_m_l553_553685


namespace age_difference_l553_553654

def JobAge := 5
def StephanieAge := 4 * JobAge
def FreddyAge := 18

theorem age_difference : StephanieAge - FreddyAge = 2 := by
  sorry

end age_difference_l553_553654


namespace compute_expression_l553_553968

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 :=
by
  sorry

end compute_expression_l553_553968


namespace sum_squares_symmetric_l553_553436

theorem sum_squares_symmetric (n : ℕ) (h : n > 0) :
  (∑ i in Finset.range (n+1), i^2) + (∑ i in Finset.range n, i^2) = n * (2 * n^2 + 1) / 3 :=
by
  sorry

end sum_squares_symmetric_l553_553436


namespace smallest_n_for_T_n_is_integer_l553_553023

def L : ℚ := ∑ i in Finset.range 9, i.succ⁻¹  -- sum of reciprocals of non-zero digits

def D : ℕ := 2^3 * 3^2 * 5 * 7  -- denominator of L in simplified form

def T (n : ℕ) : ℚ := (n * 5^(n-1)) * L + 1  -- expression for T_n

theorem smallest_n_for_T_n_is_integer : ∃ n : ℕ, 0 < n ∧ T n ∈ ℤ ∧ n = 504 :=
by
  use 504
  -- It remains to prove the conditions
  sorry

end smallest_n_for_T_n_is_integer_l553_553023


namespace num_three_digit_numbers_eq_48_l553_553126

-- Define the conditions based on the problem
def card_numbers : Set ℕ := {1, 2, 3, 4, 5, 6}
def num_cards : ℕ := 3

-- Prove that the number of different three-digit numbers formed is 48
theorem num_three_digit_numbers_eq_48 :
  (num_cards = 3) →
  (card_numbers = {1, 2, 3, 4, 5, 6}) →
  ∃ n : ℕ, n = 48 :=
by
  intro h1 h2
  use 48
  sorry

end num_three_digit_numbers_eq_48_l553_553126


namespace find_p_q_l553_553712

variable (R : Set ℝ)

def A (p : ℝ) : Set ℝ := {x | x^2 + p * x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5 * x + q = 0}

theorem find_p_q 
  (h : (R \ (A p)) ∩ (B q) = {2}) : p + q = -1 :=
by
  sorry

end find_p_q_l553_553712


namespace count_non_integer_angle_measures_l553_553783

def interior_angle_measure (n : ℕ) : ℚ := 180 * (n - 2) / n

theorem count_non_integer_angle_measures : 
  ({ n : ℕ | 3 ≤ n ∧ n < 12 ∧ ¬ (interior_angle_measure n).den = 1 }).to_finset.card = 2 := by
sorry

end count_non_integer_angle_measures_l553_553783


namespace equal_perimeters_of_cross_sections_l553_553435

-- Define a regular tetrahedron
structure RegularTetrahedron (a : ℝ) :=
  (edge_length : ℝ)
  (is_regular : edge_length = a)

-- Define the types for points, planes, and intersections
structure Plane (P : Type) := 
  parallel_to_edges : P → Prop

structure CrossSection :=
  (perimeter : ℝ)

-- Define the problem
theorem equal_perimeters_of_cross_sections (a : ℝ) 
  (T : RegularTetrahedron a) 
  (planeAB planeCD : Plane CrossSection) 
  (parallelAB : planeAB.parallel_to_edges) 
  (parallelCD : planeCD.parallel_to_edges): 
  (∀ figure1 figure2 : CrossSection, planeAB.parallel_to_edges figure1 → planeCD.parallel_to_edges figure2 → 
    figure1.perimeter = 2 * a ∧ figure2.perimeter = 2 * a ∧ figure1.perimeter = figure2.perimeter) :=
by
  sorry

end equal_perimeters_of_cross_sections_l553_553435


namespace integer_count_of_sqrt_x_l553_553848

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end integer_count_of_sqrt_x_l553_553848


namespace percent_democrats_voting_for_candidate_a_l553_553745

-- Defining the variables and conditions
def total_voters : ℝ := V
def democrat_fraction : ℝ := 0.60
def republican_fraction : ℝ := 0.40
def candidate_a_expected_votes_fraction : ℝ := 0.50
def republican_vote_fraction_a : ℝ := 0.20

-- Define the question as a Lean 4 statement with verification
theorem percent_democrats_voting_for_candidate_a 
  (V > 0) :
  let total_voters := V,
      democrats := democrat_fraction * total_voters,
      republicans := republican_fraction * total_voters,
      votes_for_a := candidate_a_expected_votes_fraction * total_voters,
      votes_from_republicans_for_a := republican_vote_fraction_a * republican_fraction * total_voters
  in
  let D := ((votes_for_a - votes_from_republicans_for_a) / democrats)
  in D = 0.7 :=
begin
  sorry
end

end percent_democrats_voting_for_candidate_a_l553_553745


namespace solve_for_x_l553_553813

theorem solve_for_x (x : ℝ) (h : log x 64 = 3) : x = 4 :=
by
  -- Proof omitted for now
  sorry

end solve_for_x_l553_553813


namespace incorrect_statement_in_geometry_l553_553148

theorem incorrect_statement_in_geometry : 
  ∃ D: Prop, 
    (∀ A: Prop, (A = "a quadrilateral with a pair of parallel and equal opposite sides is definitely a parallelogram") → ¬incorrect A) ∧
    (∀ B: Prop, (B = "two perpendicular lines in the same plane are definitely coplanar") → ¬incorrect B) ∧
    (∀ C: Prop, (C = "through a point on a line, an infinite number of lines can be drawn perpendicular to this line, and these lines are all in the same plane") → ¬incorrect C) ∧
    (D = "there is only one plane that can be perpendicular to a given plane through a line") ∧ (incorrect D) :=
sorry

end incorrect_statement_in_geometry_l553_553148


namespace isosceles_triangle_perimeter_l553_553103

theorem isosceles_triangle_perimeter (perimeter_eq_tri : ℕ) (side_eq_tri : ℕ) (base_iso_tri : ℕ) (perimeter_iso_tri : ℕ) 
  (h1 : perimeter_eq_tri = 60) 
  (h2 : side_eq_tri = perimeter_eq_tri / 3) 
  (h3 : base_iso_tri = 5)
  (h4 : perimeter_iso_tri = 2 * side_eq_tri + base_iso_tri) : 
  perimeter_iso_tri = 45 := by
  sorry

end isosceles_triangle_perimeter_l553_553103


namespace exists_rectangle_in_colored_gon_l553_553192

theorem exists_rectangle_in_colored_gon (V : Fin 100 → Fin 10) :
  ∃ (A B C D : Fin 100), 
    (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A) ∧
    (distance A C = 50) ∧ (distance B D = 50) ∧ 
    (V A = V C ∨ V A = V B) ∧ 
    (V B = V D ∨ V B = V A) ∧ 
    (V C = V D ∨ V C = V A ∨ V C = V B) :=
  sorry

end exists_rectangle_in_colored_gon_l553_553192


namespace divisor_value_l553_553546

theorem divisor_value (D : ℕ) (k m : ℤ) (h1 : 242 % D = 8) (h2 : 698 % D = 9) (h3 : (242 + 698) % D = 4) : D = 13 := by
  sorry

end divisor_value_l553_553546


namespace log8_f4_eq_one_third_l553_553303

theorem log8_f4_eq_one_third (f : ℝ → ℝ) (α : ℝ) (h₁ : f = λ x, x ^ α) (h₂ : f (1 / 2) = (√2) / 2) :
  Real.logb 8 (f 4) = 1 / 3 :=
  sorry

end log8_f4_eq_one_third_l553_553303


namespace acute_angle_89_l553_553592

def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

theorem acute_angle_89 :
  is_acute_angle 89 :=
by {
  -- proof details would go here, since only the statement is required
  sorry
}

end acute_angle_89_l553_553592


namespace rectangular_park_length_l553_553151

noncomputable def length_of_rectangular_park
  (P : ℕ) (B : ℕ) (L : ℕ) : Prop :=
  (P = 1000) ∧ (B = 200) ∧ (P = 2 * (L + B)) → (L = 300)

theorem rectangular_park_length : length_of_rectangular_park 1000 200 300 :=
by {
  sorry
}

end rectangular_park_length_l553_553151


namespace socks_ratio_l553_553047

-- Definitions for conditions
def original_black_socks : ℕ := 6
def price_ratio (blue_price black_price : ℝ) : Prop := black_price = 3 * blue_price
def cost_increase_rate : ℝ := 1.6

-- Variables
variable (b : ℕ) -- number of pairs of blue socks
variable (x : ℝ) -- price per pair of blue socks
variable (total_original_cost total_interchanged_cost : ℝ)

-- Calculate costs based on conditions
def original_cost := 6 * 3 * x + b * x
def interchanged_cost := b * 3 * x + 6 * x

-- Theorem statement
theorem socks_ratio (h1 : price_ratio x (3 * x))
                    (h2 : total_interchanged_cost = cost_increase_rate * total_original_cost)
                    (h3 : total_original_cost = original_cost)
                    (h4 : total_interchanged_cost = interchanged_cost) :
  6 / b = 3 / 8 :=
by 
-- You can add more details here if needed regarding the calculations
sorry

end socks_ratio_l553_553047


namespace combinations_of_blocks_l553_553587

theorem combinations_of_blocks (X Y : ℕ) (h : X + Y = 30) : 35 + X + Y = 65 :=
begin
  -- This is a simple arithmetic manipulation derived from the condition and its simplification
  calc
    35 + X + Y = 35 + (X + Y) : by rw add_assoc
    ...       = 35 + 30       : by rw h
    ...       = 65            : by norm_num,
end

end combinations_of_blocks_l553_553587


namespace trig_expression_l553_553694

noncomputable def point := ( -4 : ℝ, 3 : ℝ)
def r := real.sqrt (point.1 * point.1 + point.2 * point.2)

theorem trig_expression : 
  let x := point.1
  let y := point.2
  let sin_theta := y / r
  let cos_theta := x / r
  3 * sin_theta + cos_theta = 1 := by
    let x := point.1
    let y := point.2
    let r := real.sqrt(x^2 + y^2)
    let sin_theta := y / r
    let cos_theta := x / r
    sorry

end trig_expression_l553_553694


namespace sum_of_positions_l553_553204

-- Define the sequence length and erasures
def initial_sequence : List Nat := [1, 2, 3, 4, 5, 6]

def erase_every_nth (n : Nat) (seq : List Nat) : List Nat :=
  seq.filter (λ x, (seq.indexOf x + 1) % n ≠ 0)

def resulting_sequence (n : Nat) (initial_length : Nat) : List Nat :=
  let full_seq := List.replicate initial_length initial_sequence
  let flat_seq := full_seq.bind id
  let after_third_erase := erase_every_nth 3 flat_seq
  let after_fifth_erase := erase_every_nth 5 after_third_erase
  erase_every_nth 4 after_fifth_erase

def nth_cycle (n : Nat) (seq : List Nat) : Nat :=
  seq.nth! ((n - 1) % seq.length)

theorem sum_of_positions (s : List Nat) :
  let pos1 := 1019
  let pos2 := 1020
  let pos3 := 1021
  nth_cycle pos1 s + nth_cycle pos2 s + nth_cycle pos3 s = 10 :=
by
  let seq := resulting_sequence 12000 initial_length
  let sum_positions := nth_cycle 1019 seq + nth_cycle 1020 seq + nth_cycle 1021 seq
  show sum_positions = 10
  sorry

#eval sum_of_positions  ⟨initial_length, 12000, n_erase_conditions⟩

end sum_of_positions_l553_553204


namespace find_other_number_l553_553110

theorem find_other_number (x : ℕ) (h1 : 10 + x = 30) : x = 20 := by
  sorry

end find_other_number_l553_553110


namespace tan_half_angle_eq_neg2_l553_553780

-- Given conditions
variable (a : ℝ) (ha : a ∈ Set.Ioo (π / 2) π) (hcos : Real.cos a = -3 / 5)

-- Theorem statement
theorem tan_half_angle_eq_neg2 : Real.tan (a / 2) = -2 :=
sorry

end tan_half_angle_eq_neg2_l553_553780


namespace quadratic_inequality_l553_553650

noncomputable def quadratic_inequality_solution : Set ℝ :=
  {x | x < 2} ∪ {x | x > 4}

theorem quadratic_inequality (x : ℝ) : (x^2 - 6 * x + 8 > 0) ↔ (x ∈ quadratic_inequality_solution) :=
by
  sorry

end quadratic_inequality_l553_553650


namespace probability_neither_abc_l553_553104

theorem probability_neither_abc : 
  (P_A P_B P_C P_AB P_AC P_BC P_ABC: ℝ)
  (h1: P_A = 0.20) (h2: P_B = 0.40) (h3: P_C = 0.35) 
  (h4: P_AB = 0.15) (h5: P_AC = 0.10) (h6: P_BC = 0.20) 
  (h7: P_ABC = 0.05) : 
  let P_A_union_B_union_C := P_A + P_B + P_C - P_AB - P_AC - P_BC + P_ABC
  in 1 - P_A_union_B_union_C = 0.50 := 
by {
  -- apply the inclusion-exclusion principle
  let P_A_union_B_union_C := P_A + P_B + P_C - P_AB - P_AC - P_BC + P_ABC,
  show 1 - P_A_union_B_union_C = 0.50, from sorry
}

end probability_neither_abc_l553_553104


namespace find_P_and_Q_l553_553732

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l553_553732


namespace incorrect_statement_b_l553_553530

-- Definitions based on problem conditions
def is_false_proposition (p q : Prop) : Prop := 
  ¬ (p ∨ q)

def vectors_parallel (a b : ℝ × ℝ) : Prop := 
  a.1 * b.2 = a.2 * b.1

def converse_negative (H : Prop) (p q : Prop) : Prop := 
  H → (¬ q → ¬ p)

def negation_of_prop (H : ℝ → Prop) : Prop := 
  ∃ x ∈ set.Ioi 0, ¬ (H x)

-- Main statement to prove
theorem incorrect_statement_b (m : ℝ) :
  let a := (1, m+1),
      b := (m, 2) in
  vectors_parallel a b → m^2 + m - 2 = 0 → m = 1 → False :=
by sorry

end incorrect_statement_b_l553_553530


namespace compute_d_l553_553956

def ellipse_foci (d : ℝ) : Prop :=
  let F1 := (5 : ℝ, 10 : ℝ)
  let F2 := (d, 10)
  let Cx := (d + 5) / 2
  let major_axis := d + 5
  (2 * Real.sqrt ((Cx - 5) ^ 2 + 10 ^ 2) = major_axis)

theorem compute_d :
  ∀ d : ℝ, ellipse_foci d → d = 20 := by
  sorry

end compute_d_l553_553956


namespace max_expr_value_l553_553979

-- Define the function
def expr (x : ℝ) : ℝ :=
  real.sqrt (x + 64) + real.sqrt (25 - x) + 2 * real.sqrt x

-- Define the condition that x is between 0 and 25
def valid_x (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 25

-- Prove that the maximum value is 19
theorem max_expr_value : ∃ x, valid_x x ∧ expr x = 19 :=
begin
  sorry
end

end max_expr_value_l553_553979


namespace derivative_sqrt_eval_derivative_at_zero_l553_553255

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (1 + x^2)) ^ 10

-- Derivative of f
noncomputable def fp (x : ℝ) : ℝ := 
  10 * (x + Real.sqrt (1 + x^2)) ^ 9 * (1 + x / Real.sqrt (1 + x^2))

theorem derivative_sqrt (x : ℝ) : 
  (Real.sqrt (1 + x^2))' = x / Real.sqrt (1 + x^2) := sorry

theorem eval_derivative_at_zero :
  ∀ (f f' : ℝ → ℝ), 
  (∀ x, f' x = 10 * (x + Real.sqrt (1 + x^2)) ^ 9 * (1 + x / Real.sqrt (1 + x^2))) →
  f 0 = 1 →
  (f' 0) / (f 0) = 10 :=
by
  intro f f' h_deriv h_f0
  have h_fp0 : f' 0 = 10 := by sorry
  rw [h_f0]
  exact h_fp0

end derivative_sqrt_eval_derivative_at_zero_l553_553255


namespace minimize_distance_sum_l553_553648

noncomputable def triangle := Π (A B C : Type), Type

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

theorem minimize_distance_sum (ABC : triangle A B C) (α β : ℝ)
  (hα : α > 120) (hβ : β = 180 - α) :
  ∃ p : A, ∀ x : A, dist x A + dist x B + dist x C ≥ dist A B + dist A C :=
sorry

end minimize_distance_sum_l553_553648


namespace range_of_b_l553_553741

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3^x + b

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, f x b ≥ 0) ↔ b ≤ -1 :=
by sorry

end range_of_b_l553_553741


namespace valid_three_digit_numbers_count_l553_553321

/-- A proof problem to determine the valid number of three-digit numbers
    given the specific conditions. --/
theorem valid_three_digit_numbers_count : 
  let is_prime (n : ℕ) := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 in
  let total_three_digit_numbers := 999 - 100 + 1 in
  let invalid_digit (d : ℕ) := d = 0 ∨ is_prime d in
  let count_A_valid := 5 in
  let count_B := 9 in
  let count_C := 8 in
  let excluded_numbers := count_A_valid * count_B * count_C in
  let valid_numbers := total_three_digit_numbers - excluded_numbers in
  valid_numbers = 540 :=
by
  -- We declare the assumptions and definitions
  let is_prime (n : ℕ) := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
  let total_three_digit_numbers := 999 - 100 + 1
  let invalid_digit (d : ℕ) := d = 0 ∨ is_prime d
  let count_A_valid := 5
  let count_B := 9
  let count_C := 8
  let excluded_numbers := count_A_valid * count_B * count_C
  let valid_numbers := total_three_digit_numbers - excluded_numbers
  
  -- The statement to be proven
  show valid_numbers = 540 from sorry

end valid_three_digit_numbers_count_l553_553321


namespace ellipse_major_axis_length_l553_553955

noncomputable def length_of_major_axis (F1 F2 : ℝ × ℝ) (y_tangent : ℝ) : ℝ :=
  let F1_reflected := (F1.1, 2 * y_tangent - F1.2)
  real.sqrt ((F2.1 - F1_reflected.1)^2 + (F2.2 - F1_reflected.2)^2)

theorem ellipse_major_axis_length :
  let F1 := (4 : ℝ, 10 : ℝ)
  let F2 := (34 : ℝ, 40 : ℝ)
  let y_tangent := -5 : ℝ
  length_of_major_axis F1 F2 y_tangent = 30 * real.sqrt 5 :=
by
  sorry

end ellipse_major_axis_length_l553_553955


namespace travel_west_3_km_l553_553337

-- Define the condition
def east_travel (km: ℕ) : ℤ := km

-- Define the function for westward travel
def west_travel (km: ℕ) : ℤ := - (km)

-- Specify the theorem we want to prove
theorem travel_west_3_km :
  west_travel 3 = -3 :=
by {
  apply rfl,
  sorry
}

end travel_west_3_km_l553_553337


namespace simplify_fraction_l553_553447

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l553_553447


namespace total_amount_divided_l553_553163

theorem total_amount_divided (A B C : ℝ) (h1 : A / B = 3 / 4) (h2 : B / C = 5 / 6) (h3 : A = 29491.525423728814) :
  A + B + C = 116000 := 
sorry

end total_amount_divided_l553_553163


namespace coloringBooks_shelves_l553_553944

variables (initialStock soldBooks shelves : ℕ)

-- Given conditions
def initialBooks : initialStock = 87 := sorry
def booksSold : soldBooks = 33 := sorry
def numberOfShelves : shelves = 9 := sorry

-- Number of coloring books per shelf
def coloringBooksPerShelf (remainingBooksResult : ℕ) (booksPerShelfResult : ℕ) : Prop :=
  remainingBooksResult = initialStock - soldBooks ∧ booksPerShelfResult = remainingBooksResult / shelves

-- Prove the number of coloring books per shelf is 6
theorem coloringBooks_shelves (remainingBooksResult booksPerShelfResult : ℕ) : 
  coloringBooksPerShelf initialStock soldBooks shelves remainingBooksResult booksPerShelfResult →
  booksPerShelfResult = 6 :=
sorry

end coloringBooks_shelves_l553_553944


namespace number_of_equilateral_triangles_for_hexagon_l553_553429

-- Define the total number of small triangles needed to form a hexagon with given side length.
def number_of_triangles (n : ℕ) : ℕ :=
  6 * (n * (n + 1) / 2)

-- The main proof statement
theorem number_of_equilateral_triangles_for_hexagon (n : ℕ) (h : n = 6) :
  number_of_triangles n = 126 :=
by
  rw [h]
  norm_num
  sorry

end number_of_equilateral_triangles_for_hexagon_l553_553429


namespace sum_digits_3times_l553_553157

-- Define the sum of digits function
noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the 2006-th power of 2
noncomputable def power_2006 := 2 ^ 2006

-- State the theorem
theorem sum_digits_3times (n : ℕ) (h : n = power_2006) : 
  digit_sum (digit_sum (digit_sum n)) = 4 := by
  -- Add the proof steps here
  sorry

end sum_digits_3times_l553_553157


namespace five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l553_553966

theorem five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one (n : ℕ) (hn : n > 0) : ¬ (4 ^ n - 1 ∣ 5 ^ n - 1) :=
sorry

end five_pow_n_minus_one_not_divisible_by_four_pow_n_minus_one_l553_553966


namespace truck_tank_radius_l553_553169

-- Definitions based on conditions
def stationary_tank_radius : ℝ := 100
def stationary_tank_height : ℝ := 25
def truck_tank_height : ℝ := 10
def oil_level_drop : ℝ := 0.049

-- Main theorem based on the question and correct answer
theorem truck_tank_radius :
  (∃ (r_truck : ℝ), r_truck = 7) :=
begin
  -- Definition of volumes
  let volume_pumped := (π * stationary_tank_radius^2 * oil_level_drop : ℝ),
  let volume_truck := (π * r_truck^2 * truck_tank_height : ℝ),

  -- Assertion about the volumes being equal
  have h1 : volume_pumped = volume_truck,
  sorry, -- This is where the actual proof steps would go

  -- Solving for the truck tank radius
  have h2 : r_truck = 7,
  sorry, -- This is where the actual computation of the radius would go

  -- Conclusion: radius of the truck's tank is 7 feet
  use r_truck,
  exact h2,
end

end truck_tank_radius_l553_553169


namespace obtuse_angle_alpha_l553_553326

theorem obtuse_angle_alpha (alpha : ℝ) (h : Real.sin(alpha) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1) (h2 : 90 < alpha ∧ alpha < 180) : alpha = 140 * Real.pi / 180 :=
by sorry

end obtuse_angle_alpha_l553_553326


namespace smallest_n_for_T_n_is_integer_l553_553021

def L : ℚ := ∑ i in Finset.range 9, i.succ⁻¹  -- sum of reciprocals of non-zero digits

def D : ℕ := 2^3 * 3^2 * 5 * 7  -- denominator of L in simplified form

def T (n : ℕ) : ℚ := (n * 5^(n-1)) * L + 1  -- expression for T_n

theorem smallest_n_for_T_n_is_integer : ∃ n : ℕ, 0 < n ∧ T n ∈ ℤ ∧ n = 504 :=
by
  use 504
  -- It remains to prove the conditions
  sorry

end smallest_n_for_T_n_is_integer_l553_553021


namespace original_employees_229_l553_553888

noncomputable def original_number_of_employees (reduced_employees : ℕ) (reduction_percentage : ℝ) : ℝ := 
  reduced_employees / (1 - reduction_percentage)

theorem original_employees_229 : original_number_of_employees 195 0.15 = 229 := 
by
  sorry

end original_employees_229_l553_553888


namespace simplify_fraction_l553_553458

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l553_553458


namespace certain_person_current_age_l553_553439

-- Define Sandys's current age and the certain person's current age
variable (S P : ℤ)

-- Conditions from the problem
def sandy_phone_bill_condition := 10 * S = 340
def sandy_age_relation := S + 2 = 3 * P

theorem certain_person_current_age (h1 : sandy_phone_bill_condition S) (h2 : sandy_age_relation S P) : P - 2 = 10 :=
by
  sorry

end certain_person_current_age_l553_553439


namespace old_machine_rate_l553_553572

theorem old_machine_rate (R : ℝ) (new_machine_rate : ℝ) (total_bolts : ℝ) (time_hours : ℝ) :
  new_machine_rate = 150 →
  total_bolts = 500 →
  time_hours = 2 →
  2 * R + 2 * new_machine_rate = total_bolts →
  R = 100 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3] at h4,
  linarith,
end

end old_machine_rate_l553_553572


namespace rebecca_tent_stakes_l553_553064

-- Given conditions
variable (x : ℕ) -- number of tent stakes

axiom h1 : x + 3 * x + (x + 2) = 22 -- Total number of items equals 22

-- Proof objective
theorem rebecca_tent_stakes : x = 4 :=
by 
  -- Place for the proof. Using sorry to indicate it.
  sorry

end rebecca_tent_stakes_l553_553064


namespace collinearity_of_R_B_R_l553_553923

-- Define the conditions of the problem
variables {O O' A B T T' S S' R R' : Point}
variable Circle1 : Circle O -- Circle centered at O
variable Circle2 : Circle O' -- Circle centered at O'
variable LineTT' : Line -- Common tangent line TT'
variable LineOO' : Line -- Line connecting O and O'
variable RayAS : Ray A -- Ray extending through point S
variable RayAS' : Ray A -- Ray extending through point S'

-- Assumptions
axiom Circle1ContainsA : Circle1.contains A
axiom Circle1ContainsB : Circle1.contains B
axiom Circle1TangentT : LineTT'.touches Circle1 T
axiom Circle2ContainsA : Circle2.contains A
axiom Circle2ContainsB : Circle2.contains B
axiom Circle2TangentT' : LineTT'.touches Circle2 T'
axiom SOnLineOO' : S ∈ LineOO'
axiom S'OnLineOO' : S' ∈ LineOO'
axiom TPerpendicularToLineOO' : IsPerpendicular (LineThrough T S) LineOO'
axiom T'PerpendicularToLineOO' : IsPerpendicular (LineThrough T' S') LineOO'
axiom ROnRayAS : R ∈ RayAS
axiom R'OnRayAS' : R' ∈ RayAS'
axiom RSOnCircle1 : R ∈ Circle1
axiom R'S'OnCircle2 : R' ∈ Circle2

-- Theorem to be proved
theorem collinearity_of_R_B_R' : 
  AreCollinear R B R' :=
sorry

end collinearity_of_R_B_R_l553_553923


namespace find_expression_of_f_find_range_of_m_l553_553285

-- Lean 4 statement for the equivalent problem

section problem1

variable {a b c : ℝ}
variable (f : ℝ → ℝ)
variable (theta : ℝ)
variable (m : ℝ)

-- Conditions for the quadratic function f(x)
def f_is_quadratic : Prop := ∃ a b c, f(x) = a * x^2 + b * x + c

def f_zero : Prop := f(0) = -2

def f_negative_interval : Prop :=
  ∃ a b c, f(x) = a * x^2 + b * x + c ∧ ∀ (x : ℝ), -2 < x ∧ x < 1 → f(x) < 0

-- Expression of f(x)
theorem find_expression_of_f :
  f_is_quadratic f ∧ f_zero f ∧ f_negative_interval f → f = (fun x => x^2 + x - 2) :=
sorry

-- Inequality condition
def inequality_condition : Prop :=
  ∀ θ : ℝ, (f (Real.cos θ) ≤ Real.sqrt 2 * Real.sin (θ + Real.pi / 4) + m * Real.sin θ)

-- Range of real number m
theorem find_range_of_m :
  (f = (fun x => x^2 + x - 2)) ∧ inequality_condition f m → -3 ≤ m ∧ m ≤ 1 :=
sorry

end problem1

end find_expression_of_f_find_range_of_m_l553_553285


namespace complement_intersection_empty_l553_553386

variable (U : Set ℕ) 
variable (A B: U → Prop) 

def complement (A : Set ℕ) := { x : ℕ | x ∈ U ∧ x ∉ A }

theorem complement_intersection_empty (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) (U_def : U = {1, 2, 3, 4, 5}) (A_def : A = {1, 3, 5}) (B_def : B = {2, 4, 5}) :
(complement U A) ∩ (complement U B) = ∅ :=
by 
  have CU_A : complement U A = {2, 4} := sorry
  have CU_B : complement U B = {1, 3} := sorry
  show (complement U A) ∩ (complement U B) = ∅, from sorry

end complement_intersection_empty_l553_553386


namespace factorize_x4_minus_5x2_plus_4_l553_553228

theorem factorize_x4_minus_5x2_plus_4 (x : ℝ) :
  x^4 - 5 * x^2 + 4 = (x + 1) * (x - 1) * (x + 2) * (x - 2) :=
by
  sorry

end factorize_x4_minus_5x2_plus_4_l553_553228


namespace field_area_constraint_l553_553548

noncomputable def field_area_restriction :=
  let S := 10 / 3
  let field_area (a : ℤ) : Prop := 10 * 300 * S ≤ 10000
  ∧ (a >= -4.5)
  ∧ (a <= 4.5)
  ∧ (a ∈ {-4, -3, -2})

theorem field_area_constraint : ∀ S a, field_area a -> S <= 10 / 3 := 
by
  intros S a h
  sorry

end field_area_constraint_l553_553548


namespace smallest_n_for_Tn_integer_l553_553027

noncomputable def K : ℚ := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n-1)) * K

theorem smallest_n_for_Tn_integer :
  ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m < n, T_n m ∉ ℤ :=
  sorry

end smallest_n_for_Tn_integer_l553_553027


namespace jacket_final_price_l553_553941

theorem jacket_final_price :
    let initial_price := 150
    let first_discount := 0.30
    let second_discount := 0.10
    let coupon := 10
    let tax := 0.05
    let price_after_first_discount := initial_price * (1 - first_discount)
    let price_after_second_discount := price_after_first_discount * (1 - second_discount)
    let price_after_coupon := price_after_second_discount - coupon
    let final_price := price_after_coupon * (1 + tax)
    final_price = 88.725 :=
by
  sorry

end jacket_final_price_l553_553941


namespace projection_a_on_b_correct_l553_553717

-- Define the vectors
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 1)

-- Function to calculate the dot product of two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

-- Function to calculate the magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1^2 + v.2^2)

-- The actual projection calculation function
def projection (u v : ℝ × ℝ) : ℝ := 
  (dot_product u v) / (magnitude v)

-- Statement asserting the equality
theorem projection_a_on_b_correct : 
  projection a b = - (5 * Real.sqrt 17) / 17 := 
sorry

end projection_a_on_b_correct_l553_553717


namespace ratio_mark_to_jenna_l553_553959

-- Definitions based on the given conditions
def total_problems : ℕ := 20

def problems_angela : ℕ := 9
def problems_martha : ℕ := 2
def problems_jenna : ℕ := 4 * problems_martha - 2

def problems_completed : ℕ := problems_angela + problems_martha + problems_jenna
def problems_mark : ℕ := total_problems - problems_completed

-- The proof statement based on the question and conditions
theorem ratio_mark_to_jenna :
  (problems_mark : ℚ) / problems_jenna = 1 / 2 :=
by
  sorry

end ratio_mark_to_jenna_l553_553959


namespace trapezoid_perimeter_l553_553958

-- Definitions and the proof problem statement
theorem trapezoid_perimeter (R α : ℝ) (hα : α < real.pi / 2) 
    (isIsosceles : ∃ (A B C D : ℝ), true) 
    (inscribedCircle : ∃ (O : ℝ), true) 
    (OE_perp_AD : ∃ (OE AD : ℝ), OE = R) 
    (angleBAD: α = 0) : 
    P = 8 * R / real.sin α :=
sorry

end trapezoid_perimeter_l553_553958


namespace sign_of_k_l553_553365

variable (k x y : ℝ)
variable (A B : ℝ × ℝ)
variable (y₁ y₂ : ℝ)
variable (h₁ : A = (-2, y₁))
variable (h₂ : B = (5, y₂))
variable (h₃ : y₁ = k / -2)
variable (h₄ : y₂ = k / 5)
variable (h₅ : y₁ > y₂)
variable (h₀ : k ≠ 0)

-- We need to prove that k < 0
theorem sign_of_k (A B : ℝ × ℝ) (y₁ y₂ k : ℝ) 
  (h₁ : A = (-2, y₁)) 
  (h₂ : B = (5, y₂)) 
  (h₃ : y₁ = k / -2) 
  (h₄ : y₂ = k / 5) 
  (h₅ : y₁ > y₂) 
  (h₀ : k ≠ 0) : k < 0 := 
by
  sorry

end sign_of_k_l553_553365


namespace coefficients_quadratic_linear_l553_553589

theorem coefficients_quadratic_linear (x : ℝ) :
  let eqn := 3 * x^2 - 1 - 6 * x in
  ∃ (a b c : ℝ), eqn = a * x^2 + b * x + c ∧ a = 3 ∧ b = -6 :=
by
  sorry

end coefficients_quadratic_linear_l553_553589


namespace planes_parallel_if_perpendicular_to_line_l553_553671

variables {l : Line} {α β : Plane}

theorem planes_parallel_if_perpendicular_to_line (h1 : l ⊥ α) (h2 : l ⊥ β) : α ‖ β :=
sorry

end planes_parallel_if_perpendicular_to_line_l553_553671


namespace order_of_magnitudes_l553_553311

theorem order_of_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) : x < x^(x^x) ∧ x^(x^x) < x^x :=
by
  -- Definitions for y and z.
  let y := x^x
  let z := x^(x^x)
  have h1 : x < y := sorry
  have h2 : z < y := sorry
  have h3 : x < z := sorry
  exact ⟨h3, h2⟩

end order_of_magnitudes_l553_553311


namespace smallest_n_for_integer_Tn_l553_553012

/-- Define the sum of the reciprocals of the non-zero digits from 1 to 9. --/
def J : ℚ :=
  (∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 / (i : ℚ)))

/-- Define T_n as n * 5^(n-1) * J  --/
def T_n (n : ℕ) : ℚ :=
  n * 5^(n - 1) * J

theorem smallest_n_for_integer_Tn : 
  ∃ n : ℕ, ( ∀ m : ℕ, 0 < m → T_n m ∈ ℤ → n = 63 ) :=
sorry

end smallest_n_for_integer_Tn_l553_553012


namespace repeating_decimal_fraction_l553_553993

noncomputable def repeating_decimal := 4 + 36 / 99

theorem repeating_decimal_fraction : 
  repeating_decimal = 144 / 33 := 
sorry

end repeating_decimal_fraction_l553_553993


namespace mathilda_original_amount_l553_553046

theorem mathilda_original_amount (O : ℝ) (initial_installment : ℝ) (remaining_percentage : ℝ) 
  (h_initial_installment : initial_installment = 125) 
  (h_remaining : remaining_percentage = 75) 
  (h_equation : 25 / 100 * O = initial_installment) : 
  O = 500 :=
begin
  -- placeholders for conditions
  sorry
end

end mathilda_original_amount_l553_553046


namespace rice_left_after_cooking_l553_553808

theorem rice_left_after_cooking (initial_rice : ℕ) (morning_fraction : ℚ) (evening_fraction : ℚ) : 
  initial_rice = 10 ∧ morning_fraction = 9/10 ∧ evening_fraction = 1/4 →
  let morning_cooked := morning_fraction * initial_rice in
  let remaining_after_morning := initial_rice - morning_cooked in
  let evening_cooked := evening_fraction * remaining_after_morning in
  let remaining_after_evening := remaining_after_morning - evening_cooked in 
  let remaining_grams := remaining_after_evening * 1000 in
  remaining_grams = 750 :=
  by {
    intro h,
    cases h with h1 h_all,
    cases h_all with h2 h3,
    have morning_cooked := h2 * h1,
    have remaining_after_morning := h1 - morning_cooked,
    have evening_cooked := h3 * remaining_after_morning,
    have remaining_after_evening := remaining_after_morning - evening_cooked,
    have remaining_grams := remaining_after_evening * 1000,
    rw [h1, h2, h3] at remaining_grams,
    sorry
  }

end rice_left_after_cooking_l553_553808


namespace sum_digits_n_cube_l553_553507

theorem sum_digits_n_cube (n : ℕ) (digits_sum_100 : ∑ i in (digits n).to_finset, digits n i = 100) :
  ∃ m : ℕ, (∑ i in (digits (n ^ 3)).to_finset, digits (n ^ 3) i = 1000000) := sorry

end sum_digits_n_cube_l553_553507


namespace complex_number_problems_l553_553144

variable (z1 z2 : Complex)

theorem complex_number_problems :
  (|z1 - z2| = 0 ∧ z1 = conj z2 ∧ |z1| = |z2|) →
  (conj z1 = conj z2) ∧ (conj z1 = z2) ∧ (z1 * conj z1 = z2 * conj z2) :=
by
  assume h
  have h1 : |z1 - z2| = 0 := h.1
  have h2 : z1 = conj z2 := h.2.1
  have h3 : |z1| = |z2| := h.2.2
  sorry -- Proof omitted

end complex_number_problems_l553_553144


namespace ball_hits_ground_approx_time_l553_553828

noncomputable def ball_hits_ground_time (t : ℝ) : ℝ :=
-6 * t^2 - 12 * t + 60

theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, |t - 2.32| < 0.01 ∧ ball_hits_ground_time t = 0 :=
sorry

end ball_hits_ground_approx_time_l553_553828


namespace probability_sum_less_than_10_given_first_die_is_6_l553_553520

-- Definitions
noncomputable def dice : finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def sum_of_dice {x y : ℕ} (hx : x ∈ dice) (hy : y ∈ dice) : ℕ := x + y

-- Probability calculation
def probability_space := finset (ℕ × ℕ)
noncomputable def fair_dice_probability : probability_space :=
  finset.product dice dice

noncomputable def conditioned_prob (event : set (ℕ × ℕ)) (cond : set (ℕ × ℕ)) : ℝ :=
  (cond ∩ event).card.to_real / cond.card.to_real

-- Condition where sum of dice is less than 10 given first die is 6
def event_sum_less_than_10 : set (ℕ × ℕ) :=
  { p | sum_of_dice (finset.mem_univ p.1) (finset.mem_univ p.2) < 10 }

def condition_first_die_6 : set (ℕ × ℕ) :=
  { p | p.1 = 6 }

-- Theorem statement
theorem probability_sum_less_than_10_given_first_die_is_6 :
  conditioned_prob event_sum_less_than_10 condition_first_die_6 = 1 / 2 :=
by sorry

end probability_sum_less_than_10_given_first_die_is_6_l553_553520


namespace smallest_n_for_T_n_integer_l553_553007

noncomputable def K : ℚ := ∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n - 1)) * K + 1

theorem smallest_n_for_T_n_integer : ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m : ℕ, T_n m ∈ ℤ → n ≤ m :=
  ⟨504, sorry⟩

end smallest_n_for_T_n_integer_l553_553007


namespace circle_has_greatest_symmetry_lines_l553_553881

def num_of_symmetry_lines (fig : Type) [Figure fig] : ℕ := sorry
def Circle : Type := sorry
def Semicircle : Type := sorry
def EquilateralTriangle : Type := sorry
def RegularPentagon : Type := sorry
def Ellipse : Type := sorry

theorem circle_has_greatest_symmetry_lines :
  num_of_symmetry_lines Circle > num_of_symmetry_lines Semicircle ∧
  num_of_symmetry_lines Circle > num_of_symmetry_lines EquilateralTriangle ∧
  num_of_symmetry_lines Circle > num_of_symmetry_lines RegularPentagon ∧
  num_of_symmetry_lines Circle > num_of_symmetry_lines Ellipse :=
sorry

end circle_has_greatest_symmetry_lines_l553_553881


namespace independent_dependent_variables_max_acceptance_at_13_acceptance_increasing_decreasing_l553_553585

-- Definitions of the given variables and conditions
def time := ℝ
def acceptance := ℝ

def acceptance_ability (x : time) : acceptance :=
  match x with
  | 2 := 47.8
  | 5 := 53.5
  | 7 := 56.3
  | 10 := 59
  | 12 := 59.8
  | 13 := 59.9
  | 14 := 59.8
  | 17 := 58.3
  | 20 := 55
  | _ := 0

-- Statement 1: Independent and dependent variables
theorem independent_dependent_variables :
  ∃ (x : time → acceptance), 
  ∃ (y : time → acceptance), 
  (∀ t : time, 0 ≤ t ∧ t ≤ 30 → 
    (x t = t) ∧ (y t = acceptance_ability t)) := sorry

-- Statement 2: Maximum acceptance ability at x = 13
theorem max_acceptance_at_13 :
  ∃ t : time, (t = 13) ∧ (acceptance_ability t = 59.9) := sorry

-- Statement 3: Increasing and decreasing ranges of acceptance ability
theorem acceptance_increasing_decreasing :
  (∀ t : time, 0 < t ∧ t < 13 → acceptance_ability t < acceptance_ability (t + 1)) ∧ 
  (∀ t : time, 13 < t ∧ t < 20 → acceptance_ability t > acceptance_ability (t + 1)) := sorry

end independent_dependent_variables_max_acceptance_at_13_acceptance_increasing_decreasing_l553_553585


namespace volume_tetrahedron_P_DEF_l553_553350

theorem volume_tetrahedron_P_DEF 
  (x y z : ℝ) (P A B C D E F : Point)
  (edges_PABC : Edge)
  (edges_PA : Edge)
  (edges_PB : Edge)
  (edges_PC : Edge)
  (vol_P_ABC : Real) 
  (cond1 : RegularTetrahedron P A B C )
  (cond2 : PointsOnEdges P D E F A B C PA PB PC)
  (cond3 : PE ≠ PF)
  (cond4 : DE = DF = sqrt 7)
  (cond5 : EF = 2) :
  volume_tetrahedron P D E F = sqrt 17 / 8 :=
sorry

end volume_tetrahedron_P_DEF_l553_553350


namespace find_value_of_a_l553_553224

-- Given conditions
def equation1 (x y : ℝ) : Prop := 4 * y + x + 5 = 0
def equation2 (x y : ℝ) (a : ℝ) : Prop := 3 * y + a * x + 4 = 0

-- The proof problem statement
theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, equation1 x y ∧ equation2 x y a → a = -12) :=
sorry

end find_value_of_a_l553_553224


namespace find_b_l553_553120

variables {R : Type*} [Field R]
variables (a b : R^3)

def is_parallel (v1 v2 : R^3) : Prop := ∃ (k : R), v1 = k • v2
def is_orthogonal (v1 v2 : R^3) : Prop := v1 ⬝ v2 = 0

theorem find_b
  (h_ab : a + b = ![10, -5, 0])
  (h_a_parallel : is_parallel a (![2, 1, -1] : R^3))
  (h_b_orthogonal : is_orthogonal b (![2, 1, -1] : R^3)) :
  b = ![5, -7.5, 2.5] :=
sorry

end find_b_l553_553120


namespace periodic_function_proof_l553_553974

variable (f : ℝ → ℝ)

theorem periodic_function_proof (h1 : ∀ x : ℝ, f x * f (x + 2) = 1)
  (h2 : f 1 = 3)
  (h3 : f 2 = 2) :
  f 2014 = 2 :=
begin
  sorry -- proof not provided
end

end periodic_function_proof_l553_553974


namespace rectangle_width_l553_553468

theorem rectangle_width (side_length square_len rect_len : ℝ) (h1 : side_length = 4) (h2 : rect_len = 4) (h3 : square_len = side_length * side_length) (h4 : square_len = rect_len * some_width) :
  some_width = 4 :=
by
  sorry

end rectangle_width_l553_553468


namespace wise_men_hat_strategy_l553_553857

theorem wise_men_hat_strategy (n : ℕ) : 
  ∃ N : ℕ, N = 2^(n-1) ∧ 
    ∀ hat_layouts : List (Fin n → Fin 2), 
    hat_layouts.length = 2^n → 
    (∀ strategy : Fin n → (Fin n → Fin 2) → Fin 2, 
      (∀ i, strategy i (λ j, if j ≠ i then hat_layouts j else 0) = hat_layouts i) → 
      N = 2^(n-1)) :=
by
  sorry

end wise_men_hat_strategy_l553_553857


namespace infinite_set_not_of_form_nsquared_plus_p_l553_553986

open Nat

theorem infinite_set_not_of_form_nsquared_plus_p :
  ∃ (S : Set ℕ), S.infinite ∧ ∀ n ∈ S, ¬∃ (m p : ℕ), Prime p ∧ n = m^2 + p :=
by
  sorry

end infinite_set_not_of_form_nsquared_plus_p_l553_553986


namespace amount_of_bill_l553_553154

theorem amount_of_bill (TD R FV T : ℝ) (hTD : TD = 270) (hR : R = 16) (hT : T = 9/12) 
(h_formula : TD = (R * T * FV) / (100 + (R * T))) : FV = 2520 :=
by
  sorry

end amount_of_bill_l553_553154


namespace area_veranda_is_196_l553_553830

noncomputable def room_length : ℕ := 17
noncomputable def room_width : ℕ := 12
noncomputable def veranda_short_side : ℕ := 2
noncomputable def veranda_long_side : ℕ := 4

theorem area_veranda_is_196 :
  let length_entire := room_length + 2 * veranda_long_side,
      width_entire := room_width + 2 * veranda_short_side,
      area_entire := length_entire * width_entire,
      area_room := room_length * room_width
  in area_entire - area_room = 196 := by
    sorry

end area_veranda_is_196_l553_553830


namespace chord_bisected_by_point_eq_line_l553_553736

theorem chord_bisected_by_point_eq_line :
  ∀ (x y : ℝ), (x = 1) ∧ (y = 1) ∧ (∃ (A B : ℝ × ℝ), A.1 ≠ B.1 ∧ A.2 ≠ B.2 ∧ (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1 ∧ (A.1^2 / 4 + A.2^2 / 2 = 1) ∧ (B.1^2 / 4 + B.2^2 / 2 = 1)) → x + 2*y - 3 = 0 :=
by
  intros x y hp
  cases hp with hx py
  sorry

end chord_bisected_by_point_eq_line_l553_553736


namespace sqrt_D_is_odd_integer_l553_553383

theorem sqrt_D_is_odd_integer (x : ℤ) :
  let a := x
  let b := x + 1
  let c := x * (x + 1)
  let D := a^2 + b^2 + c^2 in
  ∃ n : ℤ, D = n^2 ∧ n % 2 = 1 :=
by
  let a := x
  let b := x + 1
  let c := x * (x + 1)
  let D := a^2 + b^2 + c^2
  have h1 : D = (x^2 + x + 1)^2 := sorry
  use (x^2 + x + 1)
  have h2 : (x^2 + x + 1) % 2 = 1 := sorry
  dsimp at *
  exact ⟨h1, h2⟩

end sqrt_D_is_odd_integer_l553_553383


namespace add_to_make_divisible_l553_553877

theorem add_to_make_divisible :
  ∃ n, n = 34 ∧ ∃ k : ℕ, 758492136547 + n = 51 * k := by
  sorry

end add_to_make_divisible_l553_553877


namespace final_color_is_yellow_l553_553051

def color : Type := ℕ  -- We represent the three colors as natural numbers, modulo 3 perhaps 

def blue : color := 1  -- Blue = 1
def red : color := 2   -- Red = 2
def yellow : color := 3 -- Yellow = 3

variable (initial_blues : ℕ := 7)
variable (initial_reds : ℕ := 10)
variable (initial_yellows : ℕ := 17)

-- This defines a function that describes the transformation rule
def transform (x y : color) : color :=
  match x, y with
  | 1, 2 | 2, 1 => 3
  | 2, 3 | 3, 2 => 1
  | 3, 1 | 1, 3 => 2
  | _, _ => x    -- If same color meet, no change

open Nat

theorem final_color_is_yellow (total_elves : Nat := 34) 
  (initial_blues = 7) (initial_reds = 10) (initial_yellows = 17)
  : all_same_color total_elves initial_blues initial_reds initial_yellows = yellow :=
sorry

end final_color_is_yellow_l553_553051


namespace perpendicular_planes_l553_553041

open Set

variables (m n : Line) (α β : Plane)

-- Definitions for conditions
axiom distinct_lines : m ≠ n
axiom non_coincident_planes : α ≠ β
axiom m_perp_α : m ⊥ α
axiom m_parallel_β : m ∥ β

-- Statement to prove
theorem perpendicular_planes : α ⊥ β :=
by sorry

end perpendicular_planes_l553_553041


namespace factor_poly_l553_553729

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l553_553729


namespace ant_distance_l553_553818

theorem ant_distance (n : ℕ) (a1 : ℝ) (d' : ℝ) (b1 : ℝ) (d : ℝ)
  (hlength : n = 30) (ha1 : a1 = 0.5) (hd' : d' = 0.03) (hb1 : b1 = 1.3) (hd : d = -0.013) :
  let S_n := n * a1 + (n * (n - 1)) / 2 * d',
      C_n := n * b1 + (n * (n - 1)) / 2 * d
  in S_n + C_n = 61.395 :=
by
  sorry

end ant_distance_l553_553818


namespace solve_sqrt_equation_l553_553644

theorem solve_sqrt_equation :
  ∀ (x : ℝ), (3 * Real.sqrt x + 3 * x⁻¹/2 = 7) →
  (x = (49 + 14 * Real.sqrt 13 + 13) / 36 ∨ x = (49 - 14 * Real.sqrt 13 + 13) / 36) :=
by
  intro x hx
  sorry

end solve_sqrt_equation_l553_553644


namespace time_to_pass_bridge_l553_553947

-- Define the conditions
def length_of_train : ℝ := 360 -- in meters
def speed_of_train_kmh : ℝ := 54 -- in km/hour
def length_of_bridge : ℝ := 140 -- in meters

-- Calculate required quantities
def total_distance : ℝ := length_of_train + length_of_bridge
def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

-- State the theorem to prove the time required
theorem time_to_pass_bridge : total_distance / speed_of_train_ms = 33.33 :=
by 
  sorry

end time_to_pass_bridge_l553_553947


namespace leak_emptying_time_l553_553054

theorem leak_emptying_time (A_rate L_rate : ℚ) 
  (hA : A_rate = 1 / 4)
  (hCombined : A_rate - L_rate = 1 / 8) :
  1 / L_rate = 8 := 
by
  sorry

end leak_emptying_time_l553_553054


namespace condition_sufficient_not_necessary_monotonicity_l553_553669

theorem condition_sufficient_not_necessary_monotonicity
  (f : ℝ → ℝ) (a : ℝ) (h_def : ∀ x, f x = 2^(abs (x - a))) :
  (∀ x > 1, x - a ≥ 0) → (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y) ∧
  (∃ a, a ≤ 1 ∧ (∀ x > 1, x - a ≥ 0) ∧ (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y)) :=
by
  sorry

end condition_sufficient_not_necessary_monotonicity_l553_553669


namespace relationship_between_numbers_l553_553628

theorem relationship_between_numbers :
  log 2 0.6 < 0.99^3 ∧ 0.99^3 < log 3 real.pi :=
by
  sorry

end relationship_between_numbers_l553_553628


namespace set_operation_result_l553_553791

def M : Set ℕ := {2, 3}

def bin_op (A : Set ℕ) : Set ℕ :=
  {x | ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem set_operation_result : bin_op M = {4, 5, 6} :=
by
  sorry

end set_operation_result_l553_553791


namespace circle_has_infinite_symmetry_lines_l553_553164

theorem circle_has_infinite_symmetry_lines (C : Type) [MetricSpace C] (circ : C) (h : IsSymmetric circ) : ∃ lines : ℕ, lines = 0 ∨ lines = ∞ :=
sorry

end circle_has_infinite_symmetry_lines_l553_553164


namespace value_of_x_minus_one_third_l553_553325

theorem value_of_x_minus_one_third (x : ℝ) (h: log 5 (log 4 (log 2 x)) = 1) : x ^ (-1 / 3) = 2 ^ (-1024 / 3) :=
by
  sorry

end value_of_x_minus_one_third_l553_553325


namespace find_f_inv_l553_553260

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

noncomputable def f_inv (a : ℝ) (y : ℝ) : ℝ := a ^ y

theorem find_f_inv (a : ℝ) (h_a1 : a > 0) (h_a2 : a ≠ 1) (h : f a 2 = -1) :
  f_inv a = fun x => (1/2) ^ x :=
by
  have h1 : f a = (Real.log 2 / Real.log a) := by exact h
  have h2 : Real.log 2 = -Real.log a := by sorry
  have h3 : a = 1/2 := by sorry
  have h4 : f_inv a = fun x => (1 / 2) ^ x := by sorry
  exact h4

end find_f_inv_l553_553260


namespace product_of_0_5_and_0_8_l553_553640

theorem product_of_0_5_and_0_8 : (0.5 * 0.8) = 0.4 := by
  sorry

end product_of_0_5_and_0_8_l553_553640


namespace cassidy_grades_below_B_l553_553205

theorem cassidy_grades_below_B (x : ℕ) (h1 : 26 = 14 + 3 * x) : x = 4 := 
by 
  sorry

end cassidy_grades_below_B_l553_553205


namespace max_colors_l553_553843

theorem max_colors (n : ℕ) (color : ℕ → ℕ → ℕ)
  (h_color_property : ∀ i j : ℕ, i < 2^n → j < 2^n → color i j = color j ((i + j) % 2^n)) :
  ∃ (c : ℕ), c ≤ 2^n ∧ (∀ i j : ℕ, i < 2^n → j < 2^n → color i j < c) :=
sorry

end max_colors_l553_553843


namespace marys_next_birthday_age_l553_553417

variable {m s d : ℝ}

-- Mary is 30% older than Sally
def mary_older_than_sally := m = 1.3 * s

-- Sally is 25% younger than Danielle
def sally_younger_than_danielle := s = 0.75 * d

-- The sum of their ages is 30 years
def sum_of_ages := m + s + d = 30

theorem marys_next_birthday_age (mary_old: mary_older_than_sally)
                                (sally_young: sally_younger_than_danielle)
                                (sum_age: sum_of_ages) : 
                                m + 1 = 11 :=
by
  sorry

end marys_next_birthday_age_l553_553417


namespace whitewashing_cost_l553_553474

def dimensions := (length : ℕ) (width : ℕ) (height : ℕ)
def cost_per_sqft_wall := 2 -- Rs. per square foot
def cost_per_sqft_ceiling := 3 -- Rs. per square foot
def door_dimensions := (7, 4) -- Each door has dimensions 7 feet * 4 feet
def window_dimensions_large := (5, 4) -- Three windows have dimensions 5 feet * 4 feet
def window_dimensions_small := (4, 3) -- Two windows have dimensions 4 feet * 3 feet

def total_cost (hall_dim : dimensions) (num_doors : ℕ) (num_windows_large : ℕ) (num_windows_small : ℕ) : ℕ :=
  let (l, w, h) := hall_dim
  let hall_walls_area := 2 * (l * h) + 2 * (w * h)
  let ceiling_area := l * w
  let doors_area := num_doors * (door_dimensions.1 * door_dimensions.2)
  let windows_large_area := num_windows_large * (window_dimensions_large.1 * window_dimensions_large.2)
  let windows_small_area := num_windows_small * (window_dimensions_small.1 * window_dimensions_small.2)
  let total_walls_area := hall_walls_area - (doors_area + windows_large_area + windows_small_area)
  let walls_cost := total_walls_area * cost_per_sqft_wall
  let ceiling_cost := ceiling_area * cost_per_sqft_ceiling
  walls_cost + ceiling_cost

theorem whitewashing_cost : total_cost (40, 30, 20) 3 3 2 = 8864 := by
  sorry

end whitewashing_cost_l553_553474


namespace can_measure_all_weights_l553_553263

def weights : List ℕ := [1, 3, 9, 27]

theorem can_measure_all_weights :
  (∀ n, 1 ≤ n ∧ n ≤ 40 → ∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = n) ∧ 
  (∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = 40) :=
by
  sorry

end can_measure_all_weights_l553_553263


namespace maximize_expression_l553_553709

theorem maximize_expression (a c : ℝ) (h₀ : a > 0) (h₁ : a * c = 1) :
  (∃ x ∈ [0, +∞), ∀ x ∈ [0, +∞), f x) ∧ 
  (∑ (x : ℝ) a = \sum (y : ℝ) b ) :
  let e := (1 / (c + 1) + 4 / (a + 4))
  e = 4 / 3 := 
sorry

end maximize_expression_l553_553709


namespace candy_left_l553_553242

-- Define the given conditions
def KatieCandy : ℕ := 8
def SisterCandy : ℕ := 23
def AteCandy : ℕ := 8

-- The theorem stating the total number of candy left
theorem candy_left (k : ℕ) (s : ℕ) (e : ℕ) (hk : k = KatieCandy) (hs : s = SisterCandy) (he : e = AteCandy) : 
  (k + s) - e = 23 :=
by
  -- (Proof will be inserted here, but we include a placeholder "sorry" for now)
  sorry

end candy_left_l553_553242


namespace evaluate_sum_l553_553226

open_locale big_operators

noncomputable def T : ℤ := ∑ k in finset.range 75, (-1)^k * nat.choose 149 (2*k)

theorem evaluate_sum : T = -2^74 :=
by
  sorry

end evaluate_sum_l553_553226


namespace candy_group_size_l553_553515

-- Define the given conditions
def num_candies : ℕ := 30
def num_groups : ℕ := 10

-- Define the statement that needs to be proven
theorem candy_group_size : num_candies / num_groups = 3 := 
by 
  sorry

end candy_group_size_l553_553515


namespace seashells_count_l553_553622

theorem seashells_count (seashells_given : ℕ) (seashells_left : ℕ) (total_seashells : ℕ) : 
  seashells_given = 34 ∧ seashells_left = 22 → total_seashells = 56 := 
by 
  intros h,
  cases h with hg hl,
  rw [hg, hl],
  show total_seashells = 56,
  exact (nat.add_comm 34 22 ▸ rfl)

end seashells_count_l553_553622


namespace butterfat_percentage_of_final_mixture_l553_553320

-- Definitions for the conditions given
def gallons_of_50_percent_butterfat_milk := 8
def butterfat_percentage_of_50_percent_milk := 0.5
def gallons_of_10_percent_butterfat_milk := 24
def butterfat_percentage_of_10_percent_milk := 0.1
def total_gallons_of_mixture := gallons_of_50_percent_butterfat_milk + gallons_of_10_percent_butterfat_milk
def total_butterfat := (gallons_of_50_percent_butterfat_milk * butterfat_percentage_of_50_percent_milk) +
                       (gallons_of_10_percent_butterfat_milk * butterfat_percentage_of_10_percent_milk)
def final_butterfat_percentage := (total_butterfat / total_gallons_of_mixture) * 100

-- Statement to prove
theorem butterfat_percentage_of_final_mixture :
  final_butterfat_percentage = 20 :=
by
  sorry

end butterfat_percentage_of_final_mixture_l553_553320


namespace initial_strawberry_plants_l553_553795

theorem initial_strawberry_plants (P : ℕ) (h1 : 24 * P - 4 = 500) : P = 21 := 
by
  sorry

end initial_strawberry_plants_l553_553795


namespace distance_travelled_l553_553561

def actual_speed : ℝ := 50
def additional_speed : ℝ := 25
def time_difference : ℝ := 0.5

theorem distance_travelled (D : ℝ) : 0.5 = (D / actual_speed) - (D / (actual_speed + additional_speed)) → D = 75 :=
by sorry

end distance_travelled_l553_553561


namespace range_of_a_l553_553689

theorem range_of_a (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_mono_inc : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_ineq : f (a - 3) < f 4) : -1 < a ∧ a < 7 :=
by
  sorry

end range_of_a_l553_553689


namespace f_2015_l553_553667

-- Definition of the function f according to the given conditions
noncomputable def f : ℝ → ℝ
| x => if hx : 0 < x ∧ x < 2 then real.log (1 + 3 * x) / real.log 2 else sorry

lemma f_periodic (n : ℤ) (x : ℝ) : f (x + 4 * n) = f x := sorry

lemma f_symmetric (x : ℝ) : f (-x) = -f(x) := sorry

lemma f_definition_in_interval (x : ℝ) (h : 0 < x ∧ x < 2) : f(x) = real.log (1 + 3 * x) / real.log 2 :=
by
  -- Applying the definition for the given interval
  rw f
  split_ifs
  case pos => exact h
  case neg => exact sorry

theorem f_2015 : f 2015 = -2 := by
  -- Using periodicity
  have h := f_periodic 503 3
  rw [←h]
  -- Using symmetry
  have sym := f_symmetric 1
  rw [sym]
  -- Applying the definition
  have def1 := f_definition_in_interval 1 ⟨by norm_num, by norm_num⟩
  rw [def1]
  -- Calculating logarithm
  have log2_4 : real.log 4 / real.log 2 = 2 := by sorry
  rw [log2_4]
  norm_num

#eval f 2015 -- This should return -2 as described in the problem statement.

end f_2015_l553_553667


namespace Triangle_multiple_bases_l553_553184

-- Define structures and properties related to triangle, altitudes, and bases
structure Triangle :=
  (A B C : Point)
  (nondegenerate : A ≠ B ∧ B ≠ C ∧ C ≠ A)

def has_three_altitudes (T : Triangle) : Prop :=
  ∃ (h1 h2 h3 : Line), is_altitude T h1 ∧ is_altitude T h2 ∧ is_altitude T h3

def base_corresponds_to_altitude (T : Triangle) : Prop :=
  ∀ (h : Line), is_altitude T h → ∃! (b : Line), b ⊥ h

-- Given our conditions, define the statement to prove
theorem Triangle_multiple_bases (T : Triangle) :
  has_three_altitudes T → ¬ ∃! (b : Line), ∃ (h : Line), is_altitude T h ∧ b ⊥ h := 
by
  sorry

end Triangle_multiple_bases_l553_553184


namespace exists_irrational_greater_than_neg3_l553_553884

theorem exists_irrational_greater_than_neg3 : ∃ x : ℝ, irrational x ∧ x > -3 := 
by {
  use Real.pi,
  split,
  {
    exact Real.irrational_pi,
  },
  {
    norm_num,
  },
}

end exists_irrational_greater_than_neg3_l553_553884


namespace smallest_n_for_Tn_integer_l553_553031

noncomputable def K : ℚ := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n-1)) * K

theorem smallest_n_for_Tn_integer :
  ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m < n, T_n m ∉ ℤ :=
  sorry

end smallest_n_for_Tn_integer_l553_553031


namespace range_of_x_l553_553289

theorem range_of_x (x : ℝ) : (abs (x + 1) + abs (x - 5) = 6) ↔ (-1 ≤ x ∧ x ≤ 5) :=
by sorry

end range_of_x_l553_553289


namespace find_element_atomic_mass_l553_553604

-- Define the atomic mass of bromine
def atomic_mass_br : ℝ := 79.904

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 267

-- Define the number of bromine atoms in the compound (assuming n = 1)
def n : ℕ := 1

-- Define the atomic mass of the unknown element X
def atomic_mass_x : ℝ := molecular_weight - n * atomic_mass_br

-- State the theorem to prove
theorem find_element_atomic_mass : atomic_mass_x = 187.096 :=
by
  -- placeholder for the proof
  sorry

end find_element_atomic_mass_l553_553604


namespace find_w_l553_553735

theorem find_w (a w : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * w) : w = 49 :=
by
  sorry

end find_w_l553_553735


namespace minimum_g_exists_x_minimum_g_l553_553639

noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 1) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem minimum_g (x : ℝ) (hx : x > 0) : g x ≥ 6 :=
by
  sorry

theorem exists_x_minimum_g : ∃ x > 0, g x = 6 :=
by
  sorry

end minimum_g_exists_x_minimum_g_l553_553639


namespace elena_savings_fraction_l553_553630

variable (s p n : ℚ)

def elena_fraction_left (s p n : ℚ) : ℚ :=
if h : s ≠ 0 ∧ p ≠ 0 ∧ ∃ (n : ℚ), (1/4) * s = (1/2) * n * p
then (1 / 2)
else 0

theorem elena_savings_fraction (h : (1/4) * s = (1/2) * n * p) : elena_fraction_left s p n = 1 / 2 := 
by 
  sorry

end elena_savings_fraction_l553_553630


namespace range_of_a_l553_553343

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, (2 ≤ x ∧ x ≤ 4) ∧ (2 ≤ y ∧ y ≤ 3) → x * y ≤ a * x^2 + 2 * y^2) → a ≥ 0 := 
sorry

end range_of_a_l553_553343


namespace exponent_identity_l553_553733

variable (x : ℝ) (m n : ℝ)
axiom h1 : x^m = 6
axiom h2 : x^n = 9

theorem exponent_identity : x^(2 * m - n) = 4 :=
by
  sorry

end exponent_identity_l553_553733


namespace proof_solution_l553_553202

noncomputable def proof_problem : Prop :=
  (1 / 2) ^ (-2) + (Real.sqrt 3 - 2) ^ 0 + 3 * Real.tan (Real.pi / 6) = 5 + Real.sqrt 3

theorem proof_solution : proof_problem := by
  sorry

end proof_solution_l553_553202


namespace min_a2_b2_c2_l553_553781

theorem min_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 5 * c = 100) : 
  a^2 + b^2 + c^2 ≥ (5000 / 19) :=
by
  sorry

end min_a2_b2_c2_l553_553781


namespace intersection_P_Q_eq_P_l553_553220

def set_P : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}
def set_Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem intersection_P_Q_eq_P : (set.univ.filter (λ x : ℤ, x ∈ set_P) ∩ set.univ.filter (λ y : ℝ, y ∈ set_Q)) = set.univ.filter (λ x : ℤ, x ∈ set_P) :=
by
  sorry

end intersection_P_Q_eq_P_l553_553220


namespace geometric_mean_of_expressions_l553_553625

variable {F : Type*} [Field F]

theorem geometric_mean_of_expressions (a b : F) (h : a ≠ b) :
  Real.sqrt ((2 * (a^2 - a * b) / (35 * b)) * (10 * a / (7 * (a * b - b^2)))) = 2 * a / (7 * b) := by
  sorry

end geometric_mean_of_expressions_l553_553625


namespace descending_number_count_l553_553684

theorem descending_number_count : 
  (∑ (s : {s : Finset (Fin 6) // s.card = 4}), 
    if s.val.to_list.chain' (>) then 1 else 0) = 15 :=
sorry

end descending_number_count_l553_553684


namespace solve_eq1_solve_system_l553_553062

theorem solve_eq1 : ∃ x y : ℝ, (3 / x) + (2 / y) = 4 :=
by
  use 1
  use 2
  sorry

theorem solve_system :
  ∃ x y : ℝ,
    (3 / x + 2 / y = 4) ∧ (5 / x - 6 / y = 2) ∧ (x = 1) ∧ (y = 2) :=
by
  use 1
  use 2
  sorry

end solve_eq1_solve_system_l553_553062


namespace smallest_n_for_T_n_integer_l553_553008

noncomputable def K : ℚ := ∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n - 1)) * K + 1

theorem smallest_n_for_T_n_integer : ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m : ℕ, T_n m ∈ ℤ → n ≤ m :=
  ⟨504, sorry⟩

end smallest_n_for_T_n_integer_l553_553008


namespace solve_for_a_and_b_l553_553487

noncomputable def volume_triangular_prism (a h : ℝ) : ℝ :=
  (1 / 3) * h * (√3 / 4) * a^2

noncomputable def volume_square_prism (b h : ℝ) : ℝ :=
  (1 / 3) * h * (√3 / 2) * b^2

theorem solve_for_a_and_b (V h : ℝ) (h_pos : h > 0) (V_pos : V > 0) :
  ∃ (a b : ℝ), a = sqrt (4 * √3 * V / h) ∧ b = sqrt (2 * √3 * V / h) :=
by
  use [sqrt (4 * √3 * V / h), sqrt (2 * √3 * V / h)]
  split
  sorry
  sorry

end solve_for_a_and_b_l553_553487


namespace probability_of_winning_in_7_games_l553_553158

noncomputable def prob_of_winning_in_7_games : ℝ :=
  let p := 2 / 3 in
  let prob_mathletes_win := (Nat.choose 6 4 * p ^ 4 * (1 - p) ^ 2 * p) in
  let prob_other_win := (Nat.choose 6 4 * (1 - p) ^ 4 * p ^ 2 * (1 - p)) in
  prob_mathletes_win + prob_other_win

theorem probability_of_winning_in_7_games :
  prob_of_winning_in_7_games = 20 / 27 :=
sorry

end probability_of_winning_in_7_games_l553_553158


namespace sqrt_div_fraction_proof_l553_553634

noncomputable def sqrt_div_as_fraction (a b : ℝ) : Prop :=
  (√a / √b = 50 / 19)

theorem sqrt_div_fraction_proof (a b : ℝ) 
  (h : ((1 / 3) ^ 2 + (1 / 4) ^ 2) / ((1 / 5) ^ 2 + (1 / 6) ^ 2) = 37 * a / (100 * b)) :
  sqrt_div_as_fraction a b := 
sorry

end sqrt_div_fraction_proof_l553_553634


namespace largest_pack_size_of_markers_l553_553771

theorem largest_pack_size_of_markers (markers_John markers_Alex : ℕ) (h_John : markers_John = 36) (h_Alex : markers_Alex = 60) : 
  ∃ (n : ℕ), (∀ (x : ℕ), (∀ (y : ℕ), (x * n = markers_John ∧ y * n = markers_Alex) → n ≤ 12) ∧ (12 * x = markers_John ∨ 12 * y = markers_Alex)) :=
by 
  sorry

end largest_pack_size_of_markers_l553_553771


namespace measure_angle_Z_l553_553794

-- Define the conditions
axiom parallel_lines (p q s : ℝ → ℝ → Prop) : 
  (∀ x y, p x y ↔ q x y) ∧ (∀ x y, p x y ↔ s x y)

axiom intersects (r : ℝ → ℝ → Prop) (p q : ℝ → ℝ → Prop) 
  (X Y : ℝ × ℝ) : r X.1 X.2 ∧ r Y.1 Y.2 ∧ p X.1 X.2 ∧ q Y.1 Y.2 ∧ X ≠ Y

axiom angle_at_point (r : ℝ → ℝ → Prop) (X Y Z : ℝ × ℝ) 
  (a X_val Y_val Z_val : ℝ) :
   r X.1 X.2 ∧ r Y.1 Y.2 ∧ r Z.1 Z.2 ∧ X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X_val = 100 ∧ Y_val = 110

theorem measure_angle_Z (p q s r : ℝ → ℝ → Prop) 
  (X Y Z : ℝ × ℝ) (m_X m_Y : ℝ) :
  parallel_lines p q s → intersects r p q X Y → 
  angle_at_point r X Y Z m_X m_Y (m_X + m_Y - 360) :=
by
  sorry

end measure_angle_Z_l553_553794


namespace num_integers_between_sqrt_range_l553_553852

theorem num_integers_between_sqrt_range :
  {x : ℕ | 5 > Real.sqrt x ∧ Real.sqrt x > 3}.card = 15 :=
by sorry

end num_integers_between_sqrt_range_l553_553852


namespace square_area_l553_553430

theorem square_area (x : ℝ) (h_side : PM = 40) (h_mid : MN = 40) (h_other : NQ = 40) (h_cube : x = 3 * 40) : 
  let area := x ^ 2 in
  area = 14400 := by
sorry

end square_area_l553_553430


namespace target_heart_rate_of_athlete_l553_553953

theorem target_heart_rate_of_athlete (age : ℕ) (h : age = 30) : 
  let max_heart_rate := 225 - age in
  let target_heart_rate := 0.85 * max_heart_rate in
  Int.round target_heart_rate = 166 :=
by
  sorry

end target_heart_rate_of_athlete_l553_553953


namespace exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l553_553662

theorem exists_n_such_that_an_is_cube_and_bn_is_fifth_power
  (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (n : ℕ), n ≥ 1 ∧ (∃ k : ℤ, a * n = k^3) ∧ (∃ l : ℤ, b * n = l^5) := 
by
  sorry

end exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l553_553662


namespace derivative1_derivative2_derivative3_l553_553638

-- Problem 1
theorem derivative1 (x : ℝ) : 
  HasDerivAt (λ x, (x + 1)^2 * (x - 1)) (3 * x^2 + 2 * x - 1) x := 
by sorry

-- Problem 2
theorem derivative2 (x : ℝ) : 
  HasDerivAt (λ x, x^2 * sin x) (2 * x * sin x + x^2 * cos x) x := 
by sorry

-- Problem 3
theorem derivative3 (x : ℝ) : 
  HasDerivAt (λ x, (exp x + 1) / (exp x - 1)) (-2 * exp x / (exp x - 1)^2) x := 
by sorry

end derivative1_derivative2_derivative3_l553_553638


namespace fewest_reciprocal_keypresses_l553_553324

theorem fewest_reciprocal_keypresses (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0) 
  (h1 : f 50 = 1 / 50) (h2 : f (1 / 50) = 50) : 
  ∃ n : ℕ, n = 2 ∧ (∀ m : ℕ, (m < n) → (f^[m] 50 ≠ 50)) :=
by
  sorry

end fewest_reciprocal_keypresses_l553_553324


namespace sqrt_inequality_l553_553896

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
by
  sorry

end sqrt_inequality_l553_553896


namespace min_a_for_f_le_zero_range_for_a_to_keep_g_max_lt_zero_l553_553670

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.log x - (1/2) * a * x + 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * real.log x - (1/2) * a * x^2 + x

theorem min_a_for_f_le_zero : ∃ a : ℝ, (∀ x > 0, f x a ≤ 0) ↔ a ≥ 2 :=
by
  sorry

theorem range_for_a_to_keep_g_max_lt_zero :
  (∃ x > 0, (∃ a : ℝ, g x a < 0 ∧ ∀ y > 0, g » x » a ≤ g y a)) ↔ 2 < a ∧ a < real.exp 1 :=
by
  sorry

end min_a_for_f_le_zero_range_for_a_to_keep_g_max_lt_zero_l553_553670


namespace sets_contain_triangle_sides_l553_553427

theorem sets_contain_triangle_sides :
  ∀ (s : fin 50 → finset ℕ), (∀ i, ∀ n ∈ s i, 1 ≤ n ∧ n ≤ 200) → 
    ∃ i (a b c : ℕ), a ∈ s i ∧ b ∈ s i ∧ c ∈ s i ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a := 
by
  sorry

end sets_contain_triangle_sides_l553_553427


namespace part1_part2_l553_553682

def f (a x : ℝ) : ℝ := Real.log x + 2 * a * x + 1
def g (x : ℝ) : ℝ := x * (Real.exp x + 1)

-- Part (Ⅰ): Prove that if the maximum value of f(x) is 0, then a = -1/2
theorem part1 (a : ℝ) : (∀ x : ℝ, x > 0 → f a x ≤ 0) ∧ (∃ x : ℝ, x > 0 ∧ f a x = 0) → a = -1/2 :=
sorry

-- Part (Ⅱ): Prove that if f(x) ≤ g(x) for any positive x, then a ∈ (-∞, 1]
theorem part2 (a : ℝ) : (∀ x : ℝ, x > 0 → f a x ≤ g x) → a ≤ 1 :=
sorry

end part1_part2_l553_553682


namespace tangent_at_P_passes_through_O_l553_553379

theorem tangent_at_P_passes_through_O
  (A B C D P O : Point)
  (l : Line)
  (B0 C0 : Point)
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : P ∈ arc A D \ { B, C })
  (h3 : l ⊥ Line B C)
  (h4 : B0 ∈ Line B P)
  (h5 : C0 ∈ Line C P)
  (h6 : B0 ∈ l)
  (h7 : C0 ∈ l)
  (h8 : circumcenter ABCD = O) :
  TangentAt P (Circumcircle P B0 C0) O :=
sorry

end tangent_at_P_passes_through_O_l553_553379


namespace C_remains_constant_l553_553294

-- Defining the parameters
variables (e m S s k : ℝ)
variables (h_e_pos : 0 < e) (h_m_pos : 0 < m) (h_S_pos : 0 < S) (h_s_pos : 0 < s) (h_k_pos : 0 < k)
variables (h_S_eq_km : S = k * m)

theorem C_remains_constant :
  ∀ m, ∃ C, (C = e / (k + s)) → (C = e * m / (k * m + m * s)) :=
by
  intros m,
  use e / (k + s),
  intro h,
  rw h_S_eq_km,
  simp,
  sorry

end C_remains_constant_l553_553294


namespace smallest_n_for_integer_Tn_l553_553013

/-- Define the sum of the reciprocals of the non-zero digits from 1 to 9. --/
def J : ℚ :=
  (∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 / (i : ℚ)))

/-- Define T_n as n * 5^(n-1) * J  --/
def T_n (n : ℕ) : ℚ :=
  n * 5^(n - 1) * J

theorem smallest_n_for_integer_Tn : 
  ∃ n : ℕ, ( ∀ m : ℕ, 0 < m → T_n m ∈ ℤ → n = 63 ) :=
sorry

end smallest_n_for_integer_Tn_l553_553013


namespace intersection_complement_l553_553711

open Set

variable (x : ℝ)

def M : Set ℝ := { x | -1 < x ∧ x < 2 }
def N : Set ℝ := { x | 1 ≤ x }

theorem intersection_complement :
  M ∩ (univ \ N) = { x | -1 < x ∧ x < 1 } := by
  sorry

end intersection_complement_l553_553711


namespace smallest_n_for_integer_Tn_l553_553010

/-- Define the sum of the reciprocals of the non-zero digits from 1 to 9. --/
def J : ℚ :=
  (∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 / (i : ℚ)))

/-- Define T_n as n * 5^(n-1) * J  --/
def T_n (n : ℕ) : ℚ :=
  n * 5^(n - 1) * J

theorem smallest_n_for_integer_Tn : 
  ∃ n : ℕ, ( ∀ m : ℕ, 0 < m → T_n m ∈ ℤ → n = 63 ) :=
sorry

end smallest_n_for_integer_Tn_l553_553010


namespace log_of_5_eq_3_l553_553991

theorem log_of_5_eq_3 : log 5 125 = 3 :=
by
  have h1 : 5 ^ 3 = 125 := by norm_num
  sorry

end log_of_5_eq_3_l553_553991


namespace pizza_consumption_order_l553_553747

noncomputable def amount_eaten (fraction: ℚ) (total: ℚ) := fraction * total

theorem pizza_consumption_order :
  let total := 1
  let samuel := (1 / 6 : ℚ)
  let teresa := (2 / 5 : ℚ)
  let uma := (1 / 4 : ℚ)
  let victor := total - (samuel + teresa + uma)
  let samuel_eaten := amount_eaten samuel 60
  let teresa_eaten := amount_eaten teresa 60
  let uma_eaten := amount_eaten uma 60
  let victor_eaten := amount_eaten victor 60
  (teresa_eaten > uma_eaten) 
  ∧ (uma_eaten > victor_eaten) 
  ∧ (victor_eaten > samuel_eaten) := 
by
  sorry

end pizza_consumption_order_l553_553747


namespace number_of_solutions_f_iter_l553_553549

def f (x : ℝ) : ℝ :=
if x < 1 then 0 else 2 * x - 2

theorem number_of_solutions_f_iter (h : ∀ x, f(f(f(f(x)))) = x) : 
  {x : ℝ | f(f(f(f(x)))) = x}.finite.count = 2 :=
by
  sorry

end number_of_solutions_f_iter_l553_553549


namespace football_teams_rounds_l553_553346

theorem football_teams_rounds (n : ℕ) (h_even : n % 2 = 0) :
  ∃ rounds : fin (n - 1) → list (fin n × fin n), 
  (∀ i : fin (n - 1), (rounds i).length = n / 2) ∧
  (∀ (i : fin n), ∀ (j k : fin (n - 1)), i ∈ (rounds j).flatten → i ∈ (rounds k).flatten → j = k) :=
sorry

end football_teams_rounds_l553_553346


namespace bisector_length_correct_l553_553368

noncomputable def solve_bisector_length (PQ PR : ℝ) (cos_angle_P : ℝ) : ℝ :=
  let QR := Real.sqrt (PQ^2 + PR^2 - 2 * PQ * PR * cos_angle_P) in
  let PS := (8 / 13) * QR in
  PS

theorem bisector_length_correct :
  solve_bisector_length 5 8 (1 / 5) = (8 / 13) * Real.sqrt 73 :=
by {
  sorry
}

end bisector_length_correct_l553_553368


namespace sum_of_squares_diagonals_cyclic_quadrilateral_l553_553811

theorem sum_of_squares_diagonals_cyclic_quadrilateral 
(a b c d : ℝ) (α : ℝ) 
(hc : c^2 = a^2 + b^2 + 2 * a * b * Real.cos α)
(hd : d^2 = a^2 + b^2 - 2 * a * b * Real.cos α) :
  c^2 + d^2 = 2 * a^2 + 2 * b^2 :=
by
  sorry

end sum_of_squares_diagonals_cyclic_quadrilateral_l553_553811


namespace stocks_higher_price_l553_553799

theorem stocks_higher_price (total_stocks lower_price higher_price: ℝ)
  (h_total: total_stocks = 8000)
  (h_ratio: higher_price = 1.5 * lower_price)
  (h_sum: lower_price + higher_price = total_stocks) :
  higher_price = 4800 :=
by
  sorry

end stocks_higher_price_l553_553799


namespace odd_function_expression_l553_553270

noncomputable def f : ℝ → ℝ
| x := if x > 0 then x * (1 - x) else -f (-x)

theorem odd_function_expression (x : ℝ) (hx : x < 0) : f x = x * (1 + x) :=
by sorry

end odd_function_expression_l553_553270


namespace max_value_of_trig_expr_l553_553233

variable (x : ℝ)

theorem max_value_of_trig_expr : 
  (∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5) :=
sorry

end max_value_of_trig_expr_l553_553233


namespace range_of_a_l553_553156

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - a| > 5) ↔ (a ∈ set.Ioi 8 ∪ set.Iio (-2)) :=
by
  sorry

end range_of_a_l553_553156


namespace midpoint_chord_parabola_l553_553434

theorem midpoint_chord_parabola {p x1 x2 y1 y2 : ℝ}
  (h_parabola1 : y1^2 = 2 * p * x1)
  (h_parabola2 : y2^2 = 2 * p * x2) :
  let M := (⟨ (x1 + x2) / 2, (y1 + y2) / 2 ⟩ : ℝ × ℝ) in
  let T := ⟨p * (x1 - x2) / (y1 - y2), p⟩ in
  (M.2 = (y1 + y2) / 2) := 
sorry

end midpoint_chord_parabola_l553_553434


namespace arithmetic_to_geometric_progression_l553_553940

theorem arithmetic_to_geometric_progression (d : ℝ) (h : ∀ d, (4 + d) * (4 + d) = 7 * (22 + 2 * d)) :
  ∃ d, 7 + 2 * d = 3.752 :=
sorry

end arithmetic_to_geometric_progression_l553_553940


namespace avg_sqft_per_person_approx_320000_l553_553485

theorem avg_sqft_per_person_approx_320000 :
  let sqft_per_sqmile := 5280 ^ 2 in
  let total_sqft := 3_796_742 * sqft_per_sqmile in
  let population := 331_000_000 in
  let avg_sqft_per_person := total_sqft / population in
  abs (avg_sqft_per_person - 320_000) < 1 :=
by
  sorry

end avg_sqft_per_person_approx_320000_l553_553485


namespace circle_equation_l553_553288

-- Given conditions
variables {M : Type} [metric_space M] [normed_space ℝ M]
variables (x y : ℝ)

-- Define the radius and tangency conditions
def radius : ℝ := 1
def tangent_to_x_axis (c : M) : Prop := dist c (1 : M) = radius
def tangent_to_line (c : M) : Prop := dist c (y = √3 * x) = radius

-- Define the center of the circle as a variable satisfying the conditions
variables (center : M)

-- Define the equation of the circle
def circle_eq (center : ℝ × ℝ) : Prop := 
  (center.1 - sqrt 3)^2 + (center.2 - 1)^2 = radius^2

-- The theorem to be proven
theorem circle_equation (center : M) : 
  (tangent_to_x_axis center) ∧ (tangent_to_line center) → 
  circle_eq (sqrt 3, 1) :=
by 
  sorry

end circle_equation_l553_553288


namespace nancy_pensils_total_l553_553426

theorem nancy_pensils_total
  (initial: ℕ) 
  (mult_factor: ℕ) 
  (add_pencils: ℕ) 
  (final_total: ℕ) 
  (h1: initial = 27)
  (h2: mult_factor = 4)
  (h3: add_pencils = 45):
  final_total = initial * mult_factor + add_pencils := 
by
  sorry

end nancy_pensils_total_l553_553426


namespace smallest_n_for_T_n_integer_l553_553002

def L : ℚ := ∑ i in {1, 2, 3, 4}, 1 / i

theorem smallest_n_for_T_n_integer : ∃ n ∈ ℕ, n > 0 ∧ (n * 5^(n-1) * L).denom = 1 ∧ n = 12 :=
by
  have hL : L = 25 / 12 := by sorry
  existsi 12
  split
  exact Nat.succ_pos'
  split
  suffices (12 * 5^(12-1) * 25 / 12).denom = 1 by sorry
  sorry
  rfl

end smallest_n_for_T_n_integer_l553_553002


namespace pentagonal_tiles_count_l553_553917

theorem pentagonal_tiles_count (a b : ℕ) (h1 : a + b = 30) (h2 : 3 * a + 5 * b = 120) : b = 15 :=
by
  sorry

end pentagonal_tiles_count_l553_553917


namespace sum_fifth_powers_l553_553295

theorem sum_fifth_powers (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^5 + b^5 + c^5 = 98 / 6 := 
by 
  sorry

end sum_fifth_powers_l553_553295


namespace homework_duration_decrease_l553_553132

variable (a b x : ℝ)

theorem homework_duration_decrease (h: a * (1 - x)^2 = b) :
  a * (1 - x)^2 = b := 
by
  sorry

end homework_duration_decrease_l553_553132


namespace length_of_wire_correct_l553_553538

-- Define the area of the square field.
def area : ℕ := 27889

-- The number of times the wire goes around the field.
def times : ℕ := 11

-- The correct length of the wire required to go around the square field 11 times.
def correct_length_of_wire : ℕ := 7348

-- Prove that the length of the wire required to go 11 times around the square field with area 27889 m² is 7348 meters.
theorem length_of_wire_correct :
  let side_length := Int.sqrt area in
  let perimeter := 4 * side_length in
  let total_length_of_wire := perimeter * times in
  total_length_of_wire = correct_length_of_wire := by
  sorry

end length_of_wire_correct_l553_553538


namespace regular_polygon_interior_angle_not_integer_count_l553_553785

theorem regular_polygon_interior_angle_not_integer_count :
  let is_non_integer_angle (n : ℕ) : Prop := ¬ (180 * (n - 2) / n).den = 1
  let valid_n := {n | 3 ≤ n ∧ n < 12}
  (finset.univ.filter (λ (n : ℕ), n ∈ valid_n ∧ is_non_integer_angle n)).card = 2 :=
by
  sorry

end regular_polygon_interior_angle_not_integer_count_l553_553785


namespace sequence_b4_l553_553180

theorem sequence_b4:
  let b : ℕ → ℚ :=
    λ n, nat.rec_on n 1 (nat.rec_on n.pred 2 (λ pn ir, if pn = 0 then 1 else (ir (pn-1)^2 + ir pn^2) / (ir (pn-1) + 2 * ir pn))) in
  let b1 := 1 in
  let b2 := 2 in
  let b_rec (b: ℕ → ℚ) (n: ℕ) : ℚ := 
    if n = 1 then b 1 
    else if n = 2 then b 2 
    else (b (n-2))^2 + (b (n-1))^2 / (b (n-2) + 2 * b (n-1)) in
  b 4 = 5/4 →
  let p := 5 in
  let q := 4 in
  p + q = 9 :=
sorry

end sequence_b4_l553_553180


namespace sakshi_work_done_in_days_l553_553068

noncomputable def sakshi_days (Tanya_days : ℕ) (efficiency_factor : ℝ) : ℝ := 
  (Tanya_days * efficiency_factor) / 6

theorem sakshi_work_done_in_days :
  (Tanya_days = 10) → 
  (efficiency_factor = 1.2) → 
  sakshi_days 10 1.2 = 12 :=
by
  intro htanya hefficiency
  rw [sakshi_days, htanya, hefficiency]
  norm_num

end sakshi_work_done_in_days_l553_553068


namespace train_length_is_correct_l553_553946

-- Given conditions
def train_speed_kmh : ℝ := 60
def time_to_pass_pole_s : ℝ := 7.5

-- Conversion factor from km/hr to m/s
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (5 / 18)

-- Speed in m/s
def train_speed_mps : ℝ := kmh_to_mps train_speed_kmh

-- Distance formula
def train_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem to prove
theorem train_length_is_correct :
  train_length train_speed_mps time_to_pass_pole_s = 125.025 :=
by
  sorry

end train_length_is_correct_l553_553946


namespace student_assignment_count_l553_553198

theorem student_assignment_count :
  let students := {A, B, C, D}
  let classes := {Class1, Class2, Class3}
  ∃ (f : students → classes),
  (∀ c, ∃ s, f s = c) ∧
  f A ≠ f B →
  (count_assignments f students classes H ≠ 30) := sorry

end student_assignment_count_l553_553198


namespace value_of_e_l553_553970

-- Define the polynomial coefficients d, e, f
variable (d e f : ℝ)

-- Given conditions
def Q (x : ℝ) := 3 * x^3 + d * x^2 + e * x + f
def zero_mean_eq_product := (-(f / 3) = -9 : Prop)
def y_intercept := (Q 0 = 27 : Prop)
def sum_coefficients_eq := (3 + d + e + f = -9 : Prop)

theorem value_of_e (zero_mean_eq_product : zero_mean_eq_product) 
                   (y_intercept : Q 0 = 27) 
                   (sum_coefficients_eq : sum_coefficients_eq) : 
  e = -120 := 
sorry

end value_of_e_l553_553970


namespace minimum_value_of_function_l553_553221

theorem minimum_value_of_function : ∀ x : ℝ, x ≥ 0 → (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8 / 3 := by
  sorry

end minimum_value_of_function_l553_553221


namespace roots_relation_l553_553244

/-- Definitions of the conditions -/
def polynomial (a b c : ℝ) := λ x : ℝ, a * x^2 + b * x + c

def roots (a b c : ℝ) (α β : ℝ) :=
  β = 3 * α ∧ α + β = -b / a ∧ α * β = c / a

/-- Lean 4 theorem statement -/
theorem roots_relation (a b c α β : ℝ) :
  roots a b c α β → 3 * b^2 = 16 * a * c :=
begin
  intro h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  sorry
end

end roots_relation_l553_553244


namespace find_borrowed_interest_rate_l553_553174

-- Define the conditions
variables (loanAmount : ℕ) (lentRate : ℕ) (gainPerYear : ℕ)

-- Assign the given values from the problem
def loanAmount : ℕ := 6000
def lentRate : ℕ := 6
def gainPerYear : ℕ := 120

-- Define the function to find the borrowed interest rate
noncomputable def borrowedInterestRate := 
  let interestFromLending := lentRate * loanAmount / 100 in
  let interestPaidPerYear := interestFromLending - gainPerYear in
  (interestPaidPerYear * 100) / loanAmount

-- State the theorem to prove the borrowed interest rate is 4%
theorem find_borrowed_interest_rate : borrowedInterestRate = 4 := 
  by sorry

end find_borrowed_interest_rate_l553_553174


namespace total_sales_amount_l553_553916

theorem total_sales_amount :
  ∃ (calc1 calc2 : ℕ) (price1 price2 : ℕ), 
    calc1 = 35 ∧ 
    price1 = 15 ∧ 
    calc2 = 85 - calc1 ∧ 
    price2 = 67 ∧ 
    (calc1 * price1 + calc2 * price2 = 3875) :=
by
  exists 35
  exists 50
  exists 15
  exists 67
  simp
  sorry

end total_sales_amount_l553_553916


namespace card_toss_sum_one_probability_l553_553861

noncomputable def cardTossProbability : ℂ :=
  let outcomes := [{0, 0}, {0, 1}, {1, 0}, {1, 1}] in
  let favorable := [0, 1] ∪ [1, 0] in
  (favorable.length / outcomes.length : ℂ)

theorem card_toss_sum_one_probability :
  cardTossProbability = 1/2 :=
sorry

end card_toss_sum_one_probability_l553_553861


namespace sum_vectors_zero_l553_553895

variable {n : ℕ}

-- Define a magic square condition type
structure MagicSquare (M : ℕ → ℕ → ℕ) : Prop :=
(complete : ∀ i j, 1 ≤ M i j ∧ M i j ≤ n^2)
(sum_row_eq : ∀ i, ∑ k in finset.range n, M i k = (n * (n^2 + 1)) / 2)
(sum_col_eq : ∀ j, ∑ k in finset.range n, M k j = (n * (n^2 + 1)) / 2)
(sum_diag1_eq : ∑ k in finset.range n, M k k = (n * (n^2 + 1)) / 2)
(sum_diag2_eq : ∑ k in finset.range n, M k (n - 1 - k) = (n * (n^2 + 1)) / 2)

theorem sum_vectors_zero (M : ℕ → ℕ → ℕ) [MagicSquare M] : 
  ∑ i in finset.range n, ∑ j in finset.range n, M i j = 0 := 
by
  sorry

end sum_vectors_zero_l553_553895


namespace smallest_number_including_prime_factors_l553_553190

theorem smallest_number_including_prime_factors (a : ℕ) (h_a : a = 30) (b : ℕ) 
  (h_b : ∀ p : ℕ, prime p ∧ p ∣ a → p ∣ b) : b = 30 :=
by {
  sorry
}

end smallest_number_including_prime_factors_l553_553190


namespace find_ordered_pair_l553_553239

def cosine_30_eq : Real := Real.cos (Real.pi / 6) -- \cos 30°
def secant_30_eq : Real := 1 / (Real.cos (Real.pi / 6)) -- \sec 30°

theorem find_ordered_pair :
  ∃ (a b : Int), 
    (√(16 - 12 * cosine_30_eq) = (a:Real) + (b : Real) * secant_30_eq) ∧ 
    (a = 4) ∧ (b = -1) := by
  sorry

end find_ordered_pair_l553_553239


namespace sum_of_squares_of_roots_l553_553215

theorem sum_of_squares_of_roots
  (x1 x2 : ℝ) (h : 5 * x1^2 + 6 * x1 - 15 = 0) (h' : 5 * x2^2 + 6 * x2 - 15 = 0) :
  x1^2 + x2^2 = 186 / 25 :=
sorry

end sum_of_squares_of_roots_l553_553215


namespace quadratic_equation_single_solution_l553_553149

theorem quadratic_equation_single_solution (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 = 0) ∧ (∀ x1 x2 : ℝ, a * x1^2 + a * x1 + 1 = 0 → a * x2^2 + a * x2 + 1 = 0 → x1 = x2) → a = 4 :=
by sorry

end quadratic_equation_single_solution_l553_553149


namespace elective_schemes_count_l553_553939

-- Define the number of different elective schemes available
theorem elective_schemes_count : 
  let total_courses := 10
  let excluded_courses := 3
  let remaining_courses := total_courses - excluded_courses
  let choose_3_from_7 := Nat.choose remaining_courses 3
  let choose_1_from_3 := Nat.choose excluded_courses 1
  let choose_2_from_7 := Nat.choose remaining_courses 2
  in (choose_3_from_7 + choose_1_from_3 * choose_2_from_7) = 98 :=
by {
  sorry
}

end elective_schemes_count_l553_553939


namespace count_zeros_eq_11_l553_553389

-- Define the sequence a_i, where each term is either -1, 0, or 1.
variable {a : Fin 50 → Int}
-- Assuming the sequence is in the set {-1, 0, 1}.
axiom a_range : ∀ i, a i ∈ {-1, 0, 1}

-- Define the conditions of the problem.
axiom cond1 : (∑ i in Finset.univ, a i) = 9
axiom cond2 : (∑ i in Finset.univ, (a i + 1)^2) = 107

-- Define the statement we want to prove: The number of zeros in the sequence is 11.
theorem count_zeros_eq_11 : (Finset.card (Finset.filter (λ i, a i = 0) Finset.univ)) = 11 := by
  sorry

end count_zeros_eq_11_l553_553389


namespace mutually_exclusive_not_opposite_l553_553757

-- Define the given conditions
def boys := 6
def girls := 5
def total_students := boys + girls
def selection := 3

-- Define the mutually exclusive and not opposite events
def event_at_least_2_boys := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (b ≥ 2) ∧ (g ≤ (selection - b))
def event_at_least_2_girls := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (g ≥ 2) ∧ (b ≤ (selection - g))

-- Statement that these events are mutually exclusive but not opposite
theorem mutually_exclusive_not_opposite :
  (event_at_least_2_boys ∧ event_at_least_2_girls) → 
  (¬ ((∃ (b: ℕ) (g: ℕ), b + g = selection ∧ b ≥ 2 ∧ g ≥ 2) ∧ ¬(event_at_least_2_boys))) :=
sorry

end mutually_exclusive_not_opposite_l553_553757


namespace men_in_first_scenario_l553_553740

theorem men_in_first_scenario 
  (M : ℕ) 
  (daily_hours_first weekly_earning_first daily_hours_second weekly_earning_second : ℝ) 
  (number_of_men_second : ℕ)
  (days_per_week : ℕ := 7) 
  (h1 : M * daily_hours_first * days_per_week = weekly_earning_first)
  (h2 : number_of_men_second * daily_hours_second * days_per_week = weekly_earning_second) 
  (h1_value : daily_hours_first = 10) 
  (w1_value : weekly_earning_first = 1400) 
  (h2_value : daily_hours_second = 6) 
  (w2_value : weekly_earning_second = 1890)
  (second_scenario_men : number_of_men_second = 9) : 
  M = 4 :=
by
  sorry

end men_in_first_scenario_l553_553740


namespace multiple_of_6_and_factor_of_72_l553_553573

open Nat

theorem multiple_of_6_and_factor_of_72 (n : ℕ) :
  (∃ k₁ : ℕ, n = 6 * k₁) ∧ (∃ k₂ : ℕ, 72 = n * k₂) ↔ n ∈ {6, 12, 18, 24, 36, 72} :=
by
  sorry

end multiple_of_6_and_factor_of_72_l553_553573


namespace geometric_sequence_fourth_term_l553_553347

/-- In a geometric sequence with common ratio 2, where the sequence is denoted as {a_n},
and it is given that a_1 * a_3 = 6 * a_2, prove that a_4 = 24. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n)
  (h1 : a 1 * a 3 = 6 * a 2) : a 4 = 24 :=
sorry

end geometric_sequence_fourth_term_l553_553347


namespace f_of_f_neg_one_l553_553704

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 1 - 2 ^ x else sqrt x

theorem f_of_f_neg_one : f (f (-1)) = (sqrt 2) / 2 := by
  sorry

end f_of_f_neg_one_l553_553704


namespace time_for_tom_to_finish_wall_l553_553891

theorem time_for_tom_to_finish_wall (avery_rate tom_rate : ℝ) (combined_duration : ℝ) (remaining_wall : ℝ) :
  avery_rate = 1 / 2 ∧ tom_rate = 1 / 4 ∧ combined_duration = 1 ∧ remaining_wall = 1 / 4 →
  (remaining_wall / tom_rate) = 1 :=
by
  intros h
  -- Definitions from conditions
  let avery_rate := 1 / 2
  let tom_rate := 1 / 4
  let combined_duration := 1
  let remaining_wall := 1 / 4
  -- Question to be proven
  sorry

end time_for_tom_to_finish_wall_l553_553891


namespace cot_inv_sum_l553_553964

theorem cot_inv_sum :
  Real.cot (Real.arccot 4 + Real.arccot 9 + Real.arccot 17 + Real.arccot 33) = 305 / 179 := 
by 
  sorry

end cot_inv_sum_l553_553964


namespace smallest_multiple_36_45_not_11_l553_553136

theorem smallest_multiple_36_45_not_11 (n : ℕ) :
  (n = 180) ↔ (n > 0 ∧ (36 ∣ n) ∧ (45 ∣ n) ∧ ¬ (11 ∣ n)) :=
by
  sorry

end smallest_multiple_36_45_not_11_l553_553136


namespace sum_first_n_terms_arith_geo_seq_l553_553690

theorem sum_first_n_terms_arith_geo_seq (a b : ℕ → ℕ) : 
  (a 1 = 1) ∧ (b 1 = 1) ∧
  (∀ n, S n = ∑ i in range (n+1), a i) ∧
  (S 3 = b 3 + 2) ∧
  (S 5 = b 5 - 1) ∧
  (∀ n, a (n + 1) = a n + d) ∧
  (∀ n, b n = 2^(n-1)) -> 
  (∀ n, T n = (1 - 2^n + n * 2^n)) :=
sorry

end sum_first_n_terms_arith_geo_seq_l553_553690


namespace evaluate_f_at_log3_half_l553_553401

open Real

noncomputable def f : ℝ → ℝ :=
  λ x, if hx : x > 0 then 3^(x + 1) else -3^(-x + 1)

theorem evaluate_f_at_log3_half :
  f (log 3 (1 / 2)) = -6 :=
by
  sorry

end evaluate_f_at_log3_half_l553_553401


namespace stephanie_oranges_l553_553078

theorem stephanie_oranges (visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) 
  (h_visits : visits = 8) (h_oranges_per_visit : oranges_per_visit = 2) : 
  total_oranges = oranges_per_visit * visits := 
by
  sorry

end stephanie_oranges_l553_553078


namespace cotangent_composition_l553_553241

theorem cotangent_composition :
  cot (arccot 4 - arccot 9 + arccot 14) = -523 / 33 :=
by
  -- Definitions and basic properties of cot and arccot
  sorry -- Proof is not required

end cotangent_composition_l553_553241


namespace general_formula_summation_inequality_l553_553672

variable {a : ℕ → ℝ}

-- Given Conditions
axiom a1 : a 1 = 1
axiom a4 : a 4 = 2
axiom recurrence : ∀ n : ℕ, 0 < n → 2 * (a (n + 1))^2 - (a n)^2 = (a (n + 2))^2

-- Problem Statement (I): General formula for a_n
theorem general_formula (n : ℕ) (h : 0 < n) : a n = Real.sqrt n := sorry

-- Problem Statement (II): Summation inequality
theorem summation_inequality (n : ℕ) (h : 0 < n) 
  (h_gen_form : ∀ (k : ℕ), 0 < k → a k = Real.sqrt k) : 
  ∑ i in Finset.range n, (1 / a (i + 1)) < 2 * a n := sorry

end general_formula_summation_inequality_l553_553672


namespace polynomial_characterization_l553_553396

-- Define the property in terms of sequences and polynomials
def satisfies_property (P : ℤ → ℤ) : Prop :=
  ∀ (a : ℕ → ℤ), (∀ n, ∃ m, a m = n) → ∃ i j k : ℤ, i < j ∧ (finset.Icc i j).sum a = P k

-- Main theorem statement
theorem polynomial_characterization :
  ∀ P : polynomial ℤ, (satisfies_property P.to_fun ↔ (∃ c d : ℤ, P = polynomial.C d + polynomial.X * polynomial.C c)) :=
begin
  sorry,
end

end polynomial_characterization_l553_553396


namespace find_j_l553_553094

noncomputable theory

def p (j k : ℝ) : Polynomial ℝ := Polynomial.C 100 + Polynomial.monomial 1 k + Polynomial.monomial 2 j + Polynomial.C 1

theorem find_j (a d : ℝ) (h_roots : (p (-100) k).roots = Multiset.of_list [a, a + 2*d, a + 4*d, a + 6*d])
  (h_poly : p j k = Polynomial.C 100 + Polynomial.monomial 2 j + Polynomial.monomial 4 1) :
  j = -100 :=
sorry

end find_j_l553_553094


namespace simplify_exp1_simplify_exp2_l553_553460

noncomputable def exp1_simplified : ℝ :=
(1) * (0.25) ^ (-2) + 8 ^ (2/3) - (1/16) ^ (-0.75)

noncomputable def exp2_simplified : ℝ :=
(Real.log 2) ^ 2 + Real.log 5 * Real.log 20 + Real.log 100

theorem simplify_exp1 : exp1_simplified = 12 := sorry

theorem simplify_exp2 : exp2_simplified = 3 := sorry

end simplify_exp1_simplify_exp2_l553_553460


namespace fraction_simplification_l553_553452

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l553_553452


namespace fraction_simplification_l553_553451

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l553_553451


namespace math_proof_problem_l553_553281

noncomputable def proof_problem : Prop :=
  ∀ (a b : ℝ), a > b ∧ b > 0 ∧ a + b = 1 → 
  let x := (1 / a)^b in
  let y := Real.log (ab := 1/(ab)) (ab) in
  let z := Real.log (ab := 1/b) (a) in
  y < z ∧ z < x

theorem math_proof_problem : proof_problem :=
begin
  sorry
end

end math_proof_problem_l553_553281


namespace intersection_A_B_l553_553341

-- Given definitions
def A : Set ℝ := { x : ℝ | -1 ≤ x ∧ x < 2 }
def B : Set ℤ := Set.univ

-- The theorem to be proven
theorem intersection_A_B :
  A ∩ (↑ B : Set ℝ) = { x : ℝ | x = -1 ∨ x = 0 ∨ x = 1 } :=
by
  sorry

end intersection_A_B_l553_553341


namespace full_recipes_needed_l553_553199

noncomputable def cookies_required (students : ℕ) (attendance_rate : ℚ) (cookies_per_student : ℕ)  (spare_cookies : ℕ) : ℕ :=
  let attendees : ℕ := (students * attendance_rate).to_nat
  let total_cookies : ℕ := attendees * cookies_per_student + spare_cookies
  let full_batches : ℕ := (total_cookies / 12) + if total_cookies % 12 == 0 then 0 else 1
  full_batches

def main : IO Unit :=
  IO.println s!"Required full recipes: {cookies_required 150 (60/100) 3 20}"

theorem full_recipes_needed :
  cookies_required 150 (60/100) 3 20 = 25 := by
  sorry

end full_recipes_needed_l553_553199


namespace factor_expression_l553_553967

theorem factor_expression (b : ℝ) : 
  (8 * b^4 - 100 * b^3 + 18) - (3 * b^4 - 11 * b^3 + 18) = b^3 * (5 * b - 89) :=
by
  sorry

end factor_expression_l553_553967


namespace correct_propositions_l553_553594

-- Conditions:
variable {x : ℝ} -- Universal quantifier needs to be handled in Lean

-- Definitions:
def proposition1 (x : ℝ) : Prop := ¬ (∀ x : ℝ, x^2 ≥ 0)
def proposition2 (r : ℝ) : Prop := ∃ r : ℝ, abs r = 1
def proposition3 (n a m : Set ℝ) : Prop := ¬(n ⊆ a ∧ ∃ n ⊆ m ∧ n ⊆ a)
def proposition4 (a : ℝ) : Prop := a = 2 / 5

-- The sequences of the true propositions are 2 and 4
def true_propositions (r : ℝ) : Bool :=
  proposition2 r ∧ proposition4 (2 / 5)

-- The goal is to prove that these propositions are true
theorem correct_propositions (r : ℝ) : true_propositions r :=
by
  sorry -- Proof omitted

end correct_propositions_l553_553594


namespace polynomial_identity_l553_553722

theorem polynomial_identity 
    (a b c d e f : ℤ) :
    (∀ x : ℤ, (3 * x + 1)^5 = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f) →
    a - b + c - d + e - f = 32 :=
begin
  sorry
end

end polynomial_identity_l553_553722


namespace find_a1_minus_2a2_plus_3a3_minus_4a4_l553_553323
noncomputable theory

-- Definitions and conditions
def poly_expansion (x : ℝ) : ℝ := (1 - 2 * x)^4
def a := (1 - 2 * x)^4.evaluate 0
def a1 := (1 - 2 * x)^4.derivative.evaluate 0
def a2 := (1 - 2 * x)^4.derivative.derivative.evaluate 0 / 2
def a3 := (1 - 2 * x)^4.derivative.derivative.derivative.evaluate 0 / 6
def a4 := (1 - 2 * x)^4.derivative.derivative.derivative.derivative.evaluate 0 / 24

-- Theorem statement
theorem find_a1_minus_2a2_plus_3a3_minus_4a4 : a1 - 2 * a2 + 3 * a3 - 4 * a4 = -216 := by
  sorry

end find_a1_minus_2a2_plus_3a3_minus_4a4_l553_553323


namespace find_c_ge_2_l553_553229

open Nat

def lcm_upto (n : ℕ) : ℕ :=
  list.foldr lcm 1 (list.range (n + 1))

theorem find_c_ge_2 : ∀ c : ℕ, c ≥ 2 → ∃ T : ℝ, ∀ n : ℕ, n > 0 → (lcm_upto n : ℝ) ≤ T * 2 ^ (c * n) :=
by
  sorry

end find_c_ge_2_l553_553229


namespace number_of_solutions_l553_553836

def abs_val (x : ℝ) : ℝ := if x < 0 then -x else x

def equation (x : ℝ) : Prop :=
  abs_val (x + 1) + abs_val (x + 9) + abs_val (x + 2) = 1992

theorem number_of_solutions : set.countable { x : ℝ | equation x } ∧ 2 ≤ set.count { x : ℝ | equation x } sorry

end number_of_solutions_l553_553836


namespace smallest_n_for_T_integer_l553_553019

noncomputable def J := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def T (n : ℕ) : ℚ := ∑ x in finset.range (5^n + 1), 
  ∑ d in {
    digit | digit.to_nat ≠ 0 ∧ digit.to_nat < 10 
  }, (1 : ℚ) / (digit.to_nat : ℚ)

theorem smallest_n_for_T_integer : ∃ n : ℕ, T n ∈ ℤ ∧ ∀ m : ℕ, T m ∈ ℤ → 63 ≤ n :=
by {
  sorry
}

end smallest_n_for_T_integer_l553_553019


namespace condition_for_y_exists_l553_553872

theorem condition_for_y_exists (n : ℕ) (hn : n ≥ 2) (x y : Fin (n + 1) → ℝ)
  (z : Fin (n + 1) → ℂ)
  (hz : ∀ k, z k = x k + Complex.I * y k)
  (heq : z 0 ^ 2 = ∑ k in Finset.range n, z (k + 1) ^ 2) :
  x 0 ^ 2 ≤ ∑ k in Finset.range n, x (k + 1) ^ 2 :=
by
  sorry

end condition_for_y_exists_l553_553872


namespace prove_ellipse_and_line_equations_l553_553696

noncomputable def ellipse_equation : Prop :=
  ∃ (a b c : ℝ), a > b ∧ b > 0 ∧ 
    (a - c = sqrt 3 - 1) ∧ 
    (b = sqrt 2) ∧ 
    (a^2 = b^2 + c^2) ∧ 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 3 + y^2 / 2 = 1)

noncomputable def line_through_left_focus : Prop :=
  ∃ (k : ℝ), ∀ x y : ℝ,
    (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1 → 
      (x + 1) * (y = k * (x + 1)) ∧ 
      ½ * abs k * (4 * sqrt 3 * (k^2 + 1) / (2 + 3 * k^2)) / sqrt (1 + k^2) = 3 * sqrt 2 / 4 →
      (y = sqrt 2 * (x + 1) ∨ y = -sqrt 2 * (x + 1)) ∧ 
      (√2 * x - y + √2 = 0 ∨ √2 * x + y + √2 = 0))

theorem prove_ellipse_and_line_equations : ellipse_equation ∧ line_through_left_focus :=
  by
    apply and.intro
    { sorry } -- proof for ellipse_equation
    { sorry } -- proof for line_through_left_focus

end prove_ellipse_and_line_equations_l553_553696


namespace problem1_part1_problem1_part2_l553_553402

theorem problem1_part1 : 
  ∀ (P : ℝ × ℝ) (F : ℝ × ℝ) (C : set (ℝ × ℝ))
  (h1 : ∀ (x y : ℝ), C (x, y) ↔ P.dist (x, y) F + 1 = abs x)
  (h2 : F = (1, 0)) (h3 : ∀ x, x ≥ 0 ∧ C (x, _)),
  C = { p : ℝ × ℝ | ∃ y, p = (4 * y ^ 2, y) } :=
sorry

theorem problem1_part2 : 
  ∀ (C : set (ℝ × ℝ)) (hC : C = { p : ℝ × ℝ | ∃ y, p = (y ^ 2 / 4, y) })
  (D : ℝ × ℝ) (hD : D = (1, 2)),
  ∀ (k : ℝ) (M N : ℝ × ℝ) 
  (l1 : ℝ → ℝ) (l2 : ℝ → ℝ)
  (hl1 : ∀ x, l1 x = k * (x - 1) + 2)
  (hl2 : ∀ x, l2 x = -k * (x - 1) + 2)
  (hM : C (fst M, snd M)) (hN : C (fst N, snd N))
  (hMl1 : M ∈ { p : ℝ × ℝ | p.2 = l1 p.1})
  (hNl2 : N ∈ { p : ℝ × ℝ | p.2 = l2 p.1}),
  slope M N = -1 :=
sorry

end problem1_part1_problem1_part2_l553_553402


namespace odd_naturals_in_memory_l553_553124

noncomputable def memory_contains (x : ℝ) (n : ℕ) : Prop :=
  ∃ (m : ℕ), m = n ∧ x^m is in memory

theorem odd_naturals_in_memory (x : ℝ) : ∀ (n : ℕ), odd n → memory_contains x n :=
sorry

end odd_naturals_in_memory_l553_553124


namespace median_divides_equal_area_l553_553481
-- Broader import to bring in necessary libraries

-- Definitions for points A, B, C, D, E, and F based on given conditions
variable {A B C D E F : Type}

-- Assume that A, B, and C are points forming a triangle
-- Assume D is the midpoint of BC, E is the midpoint of AB, and F is the midpoint of AC
axiom is_triangle (tri: Prop) (A B C: Prop) : tri → (A ∧ B ∧ C)

axiom is_midpoint {P Q R : Type} (D : P) (P Q R : Prop) : (D ∈ line P Q) ∧ (distance P D = distance Q D)
axiom is_half_point {P Q : Type} (E : P) (F : Q) (P Q : Prop) : (distance P E = distance Q F)

-- The theorem to prove that triangles created by the median are equal in area
theorem median_divides_equal_area 
  (h₁ : is_triangle tri A B C)
  (h₂ : is_midpoint D B C)
  (h₃ : is_half_point E A B)
  (h₄ : is_half_point F A C)
  : area (triangle A B D) = area (triangle A C D) :=
sorry

end median_divides_equal_area_l553_553481


namespace rebecca_tent_stakes_l553_553063

-- Given conditions
variable (x : ℕ) -- number of tent stakes

axiom h1 : x + 3 * x + (x + 2) = 22 -- Total number of items equals 22

-- Proof objective
theorem rebecca_tent_stakes : x = 4 :=
by 
  -- Place for the proof. Using sorry to indicate it.
  sorry

end rebecca_tent_stakes_l553_553063


namespace area_under_curve_eq_37_over_12_l553_553819

noncomputable def curve (x : ℝ) : ℝ := -x^3 + x^2 + 2 * x

theorem area_under_curve_eq_37_over_12 : 
  ∫ x in -1..0, (0 - curve x) + ∫ x in 0..2, curve x = 37 / 12 :=
by
  sorry

end area_under_curve_eq_37_over_12_l553_553819


namespace max_T_n_l553_553276

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (a₁ : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (a 1 * (1 - (a (n + 1)/a 1)))

def max_product_geometric_sequence (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, T n = ∏ i in finset.range n, a i

theorem max_T_n 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)
  (a₁ : a 1 = 30)
  (h1 : ∀ q, geometric_sequence a q 30)
  (h2 : sum_geometric_sequence S a)
  (h3 : 8 * S 6 = 9 * S 3)
  (h4 : max_product_geometric_sequence T a):
  ∃ n, T n = 5 := by
  sorry

end max_T_n_l553_553276


namespace max_y_value_l553_553335

noncomputable def max_possible_y : ℝ :=
  3 * real.sqrt 2 ^ (1 / 3) / 16

theorem max_y_value (x y : ℝ) (h : (x + y)^4 = x - y) : y ≤ max_possible_y :=
by
  sorry

end max_y_value_l553_553335


namespace problem_statement_l553_553900

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1/a) + Real.sqrt (b + 1/b) + Real.sqrt (c + 1/c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
sorry

end problem_statement_l553_553900


namespace total_customers_l553_553902

-- Define the initial number of customers
def initial_customers : ℕ := 14

-- Define the number of customers that left
def customers_left : ℕ := 3

-- Define the number of new customers gained
def new_customers : ℕ := 39

-- Prove that the total number of customers is 50
theorem total_customers : initial_customers - customers_left + new_customers = 50 := 
by
  sorry

end total_customers_l553_553902


namespace add_fifteen_sub_fifteen_l553_553837

theorem add_fifteen (n : ℕ) (m : ℕ) : n + m = 195 :=
by {
  sorry  -- placeholder for the actual proof
}

theorem sub_fifteen (n : ℕ) (m : ℕ) : n - m = 165 :=
by {
  sorry  -- placeholder for the actual proof
}

-- Let's instantiate these theorems with the specific values from the problem:
noncomputable def verify_addition : 180 + 15 = 195 :=
by exact add_fifteen 180 15

noncomputable def verify_subtraction : 180 - 15 = 165 :=
by exact sub_fifteen 180 15

end add_fifteen_sub_fifteen_l553_553837


namespace nearest_vertex_to_origin_after_dilation_l553_553816

theorem nearest_vertex_to_origin_after_dilation :
  let center : ℝ × ℝ := (10, -10)
  let area : ℝ := 16
  let center_of_dilation : ℝ × ℝ := (1, -1)
  let scale_factor : ℝ := 3
  ∃ vertex : ℝ × ℝ, 
    (vertex = (22, -22)) :=
by
  let center := (10, -10)
  let area := 16
  let center_of_dilation := (1, -1)
  let scale_factor := 3
  have vertex := (22, -22)
  existsi vertex
  sorry
  
end nearest_vertex_to_origin_after_dilation_l553_553816


namespace game_winner_l553_553118

-- Definitions and conditions
def number_of_baskets : List ℕ := [6 * n, 6 * n + 1, 6 * n + 2, 6 * n + 3, 6 * n + 4, 6 * n + 5]
def positive_integer (n : ℕ) : Prop := n > 0
def valid_move (k : ℕ) : Prop := k = 1 ∨ k = 2
def take_last_lose : Prop := True -- Placeholder, this represents the losing condition when taking the last ball

-- Yangxia's strategy function
def winning_strategy (basket : ℕ) : Prop :=
  ((basket % 6 = 1) ∨ (basket % 6 = 4)) ∧ ∀ t : List ℕ, (∀ k ∈ t, valid_move k) → 
  ∃ m ∈ t, take_last_lose

theorem game_winner (n : ℕ) (h : positive_integer n) :
  ∃ b ∈ number_of_baskets, winning_strategy b := by
  sorry

end game_winner_l553_553118


namespace range_of_g_l553_553980

theorem range_of_g (A : ℝ) (h : A ≠ n * π / 2 ∀ n : ℤ) :
  let g := λ A,
    (sin A * (4 * cos^2 A + 2 * cos^4 A + 2 * sin^2 A + sin^2 A * cos^2 A)) /
    (tan A * (sec A - 2 * sin A * tan A))
  in ∃ y ∈ Ioo 4 6, ∀ A, g A = y :=
sorry

end range_of_g_l553_553980


namespace jackson_earned_on_monday_l553_553767

-- Definitions
def goal := 1000
def tuesday_earnings := 40
def avg_rate := 10
def houses := 88
def days_remaining := 3
def total_collected_remaining_days := days_remaining * (houses / 4) * avg_rate

-- The proof problem statement
theorem jackson_earned_on_monday (m : ℕ) :
  m + tuesday_earnings + total_collected_remaining_days = goal → m = 300 :=
by
  -- We will eventually provide the proof here
  sorry

end jackson_earned_on_monday_l553_553767


namespace length_of_FN_l553_553686

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  (c / (4*a), b)

theorem length_of_FN
(Focus : ℝ × ℝ)
(M : ℝ × ℝ)
(N : ℝ × ℝ)
(Focus_is_focus_of_C : Focus = (3, 0))
(M_is_on_C : M.1 = (M.2 ^ 2) / 12)
(N_is_on_y_axis : N.1 = 0)
(M_is_midpoint_of_FN : ∀ (x y : ℝ), M = ((Focus.1 + 0) / 2, (Focus.2 + y) / 2) → N = (0, 2 * M.2)) :
  dist Focus N = 9 := by sorry

end length_of_FN_l553_553686


namespace parallelogram_angle_is_right_angle_l553_553969

open EuclideanGeometry

-- Definition of a parallelogram based on points
structure Parallelogram (A B C D : Point) : Prop :=
  (AB_parallel_CD : parallel (line A B) (line C D))
  (AD_parallel_BC : parallel (line A D) (line B C))
  (AB_eq_CD : dist A B = dist C D)
  (AD_eq_BC : dist A D = dist B C)

-- Definition of midpoint
def is_midpoint (M : Point) (A B : Point) : Prop :=
  dist M A = dist M B

-- Problem Statement
theorem parallelogram_angle_is_right_angle
  (A B C D M : Point)
  (h1 : Parallelogram A B C D)
  (h2 : is_midpoint M C D)
  (h3 : on_angle_bisector M (angle A B D)) :
  angle A M B = 90 :=
sorry

end parallelogram_angle_is_right_angle_l553_553969


namespace radius_of_inscribed_sphere_l553_553935

theorem radius_of_inscribed_sphere (a b c s : ℝ)
  (h1: 2 * (a * b + a * c + b * c) = 616)
  (h2: a + b + c = 40)
  : s = Real.sqrt 246 ↔ (2 * s) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 :=
by
  sorry

end radius_of_inscribed_sphere_l553_553935


namespace bug_at_target_probability_l553_553918

/-- Define the point (0,0) as the starting point of the bug. -/
def start_point : (Int × Int) := (0, 0)

/-- Define the point (2,2) as the target point of the bug. -/
def target_point : (Int × Int) := (2, 2)

/-- Define the probability of the bug being at (2,2) after four moves. -/
def bug_probability_at_target : ℚ := 1 / 54

/-- The problem statement to prove that the probability of the bug being at (2,2) after four moves 
    given the conditions is 1/54. -/
theorem bug_at_target_probability :
    (prob_bag_reaches_target : (start_point ->  [Int × Int] -> Int -> Prob -> ℚ)) -> 
    (prob_bag_reaches_target (0, 0) [(1, 0), (1, 0), (0, 1), (0, 1)] 4 bug_probability_at_target =
    1 / 54) : sorry

end bug_at_target_probability_l553_553918


namespace num_of_solutions_eq_28_l553_553101

def num_solutions : Nat :=
  sorry

theorem num_of_solutions_eq_28 : num_solutions = 28 :=
  sorry

end num_of_solutions_eq_28_l553_553101


namespace average_difference_problem_l553_553178

theorem average_difference_problem
    (students : ℕ)
    (teachers_first_period : ℕ)
    (teachers_second_period : ℕ)
    (enrollments_first_period : list ℕ)
    (enrollments_second_period : list ℕ) :
    students = 120 ∧ 
    teachers_first_period = 4 ∧
    teachers_second_period = 6 ∧
    enrollments_first_period = [60, 30, 20, 10] ∧
    enrollments_second_period = [40, 30, 20, 10, 10, 10] →
    let t1 := (enrollments_first_period.sum) / teachers_first_period,
        t2 := (enrollments_second_period.sum) / teachers_second_period,
        s := (60 * (60 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 10 * (10 / 120)) in
    t1 - s = -11.66 ∧ t2 - s = -21.66 :=
sorry

end average_difference_problem_l553_553178


namespace xyz_zero_if_equation_zero_l553_553777

theorem xyz_zero_if_equation_zero (x y z : ℚ) 
  (h : x^3 + 3 * y^3 + 9 * z^3 - 9 * x * y * z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := 
by 
  sorry

end xyz_zero_if_equation_zero_l553_553777


namespace work_in_one_day_l553_553887

theorem work_in_one_day (A_days B_days : ℕ) (hA : A_days = 18) (hB : B_days = A_days / 2) :
  (1 / A_days + 1 / B_days) = 1 / 6 := 
by
  sorry

end work_in_one_day_l553_553887


namespace parabola_equation_and_triangle_area_l553_553695

open Real

theorem parabola_equation_and_triangle_area :
  (∃ p : ℕ, vertex_at_origin (λ x y, y^2 = p*x) 
    ∧ focus_at_center (λ x y, y^2 = p*x) (circle_center (x - 2)^2 + y^2 = 4))
  ∧ triangle_area_OAB (parabola y^2 = 8x) line_with_slope_2 (focus_of_parabola) = 4 * sqrt 5 := 
by
  sorry

end parabola_equation_and_triangle_area_l553_553695


namespace problem_l553_553388

theorem problem (a : ℝ) (h : (a + complex.i)^2 * complex.i).im = 0 ∧ (a + complex.i)^2 * complex.i).re > 0 : a = -1 :=
by
  sorry

end problem_l553_553388


namespace total_sum_of_quartic_numbers_is_120_l553_553739

def is_quartic_number (n : ℕ) : Prop :=
  n = 4 * n.digits.sum

theorem total_sum_of_quartic_numbers_is_120 :
  Finset.sum (Finset.filter is_quartic_number (Finset.range 1000)) id = 120 :=
sorry

end total_sum_of_quartic_numbers_is_120_l553_553739


namespace total_workers_approximation_l553_553746

theorem total_workers_approximation (p percentage_present : ℝ) (total_present : ℕ) :
  total_present = 72 →
  percentage_present = 0.837 →
  p = total_present / percentage_present →
  p ≈ 86 := 
by {
  -- Proof to be completed
  sorry
}

end total_workers_approximation_l553_553746


namespace find_second_radius_l553_553512

noncomputable def radius_of_second_sphere (r1 : ℝ) (w1 : ℝ) (w2 : ℝ) : ℝ :=
  sqrt ((w2 / w1) * (r1^2))

theorem find_second_radius :
  radius_of_second_sphere 0.15 8 32 = 0.3 :=
by 
  sorry

end find_second_radius_l553_553512


namespace system_of_equations_solution_l553_553310

theorem system_of_equations_solution :
  ∀ (x y z : ℝ),
  4 * x + 2 * y + z = 20 →
  x + 4 * y + 2 * z = 26 →
  2 * x + y + 4 * z = 28 →
  20 * x^2 + 24 * x * y + 20 * y^2 + 12 * z^2 = 500 :=
by
  intros x y z h1 h2 h3
  sorry

end system_of_equations_solution_l553_553310


namespace estimate_total_number_of_fish_l553_553521

theorem estimate_total_number_of_fish
  (marked_released : ℕ)
  (second_catch : ℕ)
  (marked_second_catch : ℕ)
  (proportion_marked_second_catch : marked_second_catch = 10)
  (total_second_catch : second_catch = 100)
  (total_marked_released : marked_released = 200) :
  ∃ total_fish : ℕ, total_fish = 2000 :=
by
  -- We are given that 10 marked fish were found in the second catch of 100 fish.
  have h1 : marked_second_catch = 10 := proportion_marked_second_catch,
  have h2 : second_catch = 100 := total_second_catch,
  have h3 : marked_released = 200 := total_marked_released,
  -- We connect these data to solve for the total number of fish, which we are given to be 2000.
  -- Insert proof steps here to properly validate:
  -- total_fish = (marked_released * total_second_catch) / marked_second_catch
  use 2000,
  sorry

end estimate_total_number_of_fish_l553_553521


namespace find_constants_l553_553378

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 2, 1], ![2, 0, 2], ![1, 2, 0]]

def I : Matrix (Fin 3) (Fin 3) ℚ :=
  1

theorem find_constants (s t u : ℚ) (h : B^3 + s • B^2 + t • B + u • I = 0) :
  (s, t, u) = (-8, -12, -28) :=
begin
  -- proof goes here
  sorry
end

end find_constants_l553_553378


namespace sum_of_h_integer_values_l553_553645

theorem sum_of_h_integer_values :
  (∑ h in finset.filter (λ h : ℤ, ∀ r : ℝ, (|| r + h | - r | - 4 * r | = 9 * | r - 3 |) → function.count (λ r : ℝ, || r + h | - r | - 4 * r | = 9 * | r - 3 |) 1) (finset.Icc (-18) 12)) = -93 :=
sorry

end sum_of_h_integer_values_l553_553645


namespace directrix_of_parabola_l553_553090

theorem directrix_of_parabola :
  ∀ (a h k : ℝ), (a < 0) → (∀ x, y = a * (x - h) ^ 2 + k) → (h = 0) → (k = 0) → 
  (directrix = 1 / (4 * a)) → (directrix = 1 / 4) :=
by
  sorry

end directrix_of_parabola_l553_553090


namespace kite_of_inscribed_right_angles_l553_553554

variables {A B C M N P : Type*}
variables [add_group A] [add_group B] [add_group C] [add_group M] [add_group N] [add_group P]
variables [module A N] [module B P] [module M C]
variables [has_inner A] [has_inner B] [has_inner C]

theorem kite_of_inscribed_right_angles 
  (H1: M ∈ line_segment B C) 
  (H2: N ∈ line_segment A C)
  (H3: P ∈ line_segment A B)
  (H4: angle M N A = 90)
  (H5: angle M P A = 90)
  (H6: exists_circle_inscribed (quadrilateral M N A P)) :
  is_kite (quadrilateral M N A P) :=
sorry

end kite_of_inscribed_right_angles_l553_553554


namespace g_f_not_square_l553_553084

noncomputable def f : Polynomials ℝ := sorry
noncomputable def g : Polynomials ℝ := sorry

lemma non_constant_f : ¬ constant f := sorry
lemma non_constant_g : ¬ constant g := sorry

lemma non_square_f : ¬ ∃ (p : Polynomials ℝ), f = p^2 := sorry
lemma non_square_g : ¬ ∃ (p : Polynomials ℝ), g = p^2 := sorry

lemma f_g_square (s : Polynomials ℝ) : f (g x) = s^2 := sorry

theorem g_f_not_square : ¬ ∃ (t : Polynomials ℝ), g (f x) = t^2 := 
by 
  intro h
  cases h with T hT
  -- Proceed with proof as per the steps outlined in the solution
  sorry

end g_f_not_square_l553_553084


namespace cube_sphere_volume_relation_l553_553582

theorem cube_sphere_volume_relation (n : ℕ) (h : 2 < n)
  (h_volume : n^3 - (n^3 * pi / 6) = (n^3 * pi / 3)) : n = 8 :=
sorry

end cube_sphere_volume_relation_l553_553582


namespace train_length_l553_553183

theorem train_length
  (train_speed_kmph : ℝ)
  (person_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (h_train_speed : train_speed_kmph = 80)
  (h_person_speed : person_speed_kmph = 16)
  (h_time : time_seconds = 15)
  : (train_speed_kmph - person_speed_kmph) * (5/18) * time_seconds = 266.67 := 
by
  rw [h_train_speed, h_person_speed, h_time]
  norm_num
  sorry

end train_length_l553_553183


namespace sequence_limit_l553_553551

noncomputable def a_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), 1 / (k * (k + 1) * (k + 2))

theorem sequence_limit : ∀ (a_n : ℕ → ℝ), (∀ n, a_n n = ∑ k in Finset.range (n + 1), 1 / (k * (k + 1) * (k + 2))) → 
                                   tendsto a_n at_top (𝓝 (1 / 4)) :=
begin
  assume a_n h,
  sorry
end

end sequence_limit_l553_553551


namespace abs_sum_example_l553_553906

theorem abs_sum_example : |(-8 : ℤ)| + |(-4 : ℤ)| = 12 := by
  sorry

end abs_sum_example_l553_553906


namespace maximize_product_l553_553266

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def is_arithmetic_progression : Prop := 
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (n : ℕ) : ℝ := 
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

-- Given Conditions
axiom sum_condition : S 6 = 252
axiom term_condition : 8 * a 7 = a 4

-- To Prove
theorem maximize_product : 7 ≤ n ∧ n ≤ 8 :=
by
  sorry

end maximize_product_l553_553266


namespace monthly_cost_medicine_l553_553075

-- Definitions based on conditions
def initial_price : ℝ := 150.00
def coupon_discount : ℝ := 0.15
def shipping_fee : ℝ := 12.00
def sales_tax_rate : ℝ := 0.05
def mail_in_rebate : ℝ := 25.00
def cashback_rate : ℝ := 0.10
def num_months : ℕ := 6

-- Statement to be proved
theorem monthly_cost_medicine : 
  let price_after_coupon := initial_price * (1 - coupon_discount)
  let price_after_shipping := price_after_coupon + shipping_fee
  let price_after_tax := price_after_shipping * (1 + sales_tax_rate)
  let price_after_rebate := price_after_tax - mail_in_rebate
  let price_after_cashback := price_after_rebate - (initial_price * cashback_rate) in
  price_after_cashback / num_months = 17.75 :=
by
  sorry

end monthly_cost_medicine_l553_553075


namespace three_digit_multiple_l553_553182

open Classical

theorem three_digit_multiple (n : ℕ) (h₁ : n % 2 = 0) (h₂ : n % 5 = 0) (h₃ : n % 3 = 0) (h₄ : 100 ≤ n) (h₅ : n < 1000) :
  120 ≤ n ∧ n ≤ 990 :=
by
  sorry

end three_digit_multiple_l553_553182


namespace percentage_heavier_l553_553773

variables (J M : ℝ)

theorem percentage_heavier (hM : M ≠ 0) : 
  100 * ((J + 3) - M) / M = 100 * ((J + 3) - M) / M := 
sorry

end percentage_heavier_l553_553773


namespace nine_digit_palindromes_count_l553_553319

-- Define the sequence of digits and its constraints
def digits : List Nat := [1, 1, 2, 2, 2, 3, 3, 4, 4]

-- Define the property of a 9-digit palindrome
def is_palindrome (l : List Nat) : Prop :=
  l.length = 9 ∧ l = l.reverse

-- Define the specific 9-digit palindromes using the given digits
def nine_digit_palindromes (l : List Nat) : Prop :=
  is_palindrome l ∧ (∀ x, x ∈ l → x ∈ digits)

-- Theorem to prove the number of such palindromes
theorem nine_digit_palindromes_count :
  (∃ l : List Nat, nine_digit_palindromes l) -> (card (set_of (nine_digit_palindromes)) = 1260) :=
sorry

end nine_digit_palindromes_count_l553_553319


namespace isosceles_triangles_with_same_color_infinitely_l553_553949

noncomputable def color := bool

def colored_circle (S : Set ℂ) (colors : ℂ → color) : Prop :=
  ∀ pt ∈ S, ∃ c : color, colors pt = c

def exists_isosceles_triangle_same_color (S : Set ℂ) (colors : ℂ → color) : Prop :=
  ∃ (A B C ∈ S), isIsoscelesTriangle A B C ∧ colors A = colors B ∧ colors B = colors C

theorem isosceles_triangles_with_same_color_infinitely 
  (S : Set ℂ) (colors : ℂ → color)
  (h_colored : colored_circle S colors)
  (h_inf : Infinite S) : 
  ∃∞ (A B C ∈ S), isIsoscelesTriangle A B C ∧ colors A = colors B ∧ colors B = colors C := by
  sorry

end isosceles_triangles_with_same_color_infinitely_l553_553949


namespace determinant_equilateral_triangle_l553_553786

theorem determinant_equilateral_triangle :
  let A := Real.pi / 3
  let B := Real.pi / 3
  let C := Real.pi / 3
  matrix.det ![
    ![Real.sin A, 1, 1],
    ![1, Real.sin B, 1],
    ![1, 1, Real.sin C]
  ] = -Real.sqrt 3 + 3 := by
  sorry

end determinant_equilateral_triangle_l553_553786


namespace find_f_value_l553_553659

variable (a b c m : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * Real.sin x + c / x + 2

theorem find_f_value (h : f a b c (-5) = m) : f a b c 5 = 4 - m := by
  sorry

end find_f_value_l553_553659


namespace even_triangles_l553_553665

-- Definitions: vertices and convexity
def is_convex_polygon (vertices : list (ℝ × ℝ)) : Prop :=
  -- Definition of a convex polygon using vertices (skipped implementation)
  sorry

-- Definitions: internal point not on diagonals
def is_interior_not_on_diagonals (p : ℝ × ℝ) (vertices : list (ℝ × ℝ)) : Prop :=
  -- Check if p is an interior point not lying on any diagonals (skipped implementation)
  sorry

theorem even_triangles (vertices : list (ℝ × ℝ)) (h_convex : is_convex_polygon vertices)
    (p : ℝ × ℝ) (h_interior : is_interior_not_on_diagonals p vertices) :
  ∃ k : ℕ, (0 < k) ∧ (k % 2 = 0) ∧ (number_of_triangles_containing_p vertices p = k) :=
sorry

end even_triangles_l553_553665


namespace avg_rate_of_change_x2_plus_x_l553_553471

def average_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

theorem avg_rate_of_change_x2_plus_x (Δx : ℝ) :
  average_rate_of_change (λ x => x^2 + x) 1 (1 + Δx) = Δx + 3 :=
by
  sorry

end avg_rate_of_change_x2_plus_x_l553_553471


namespace max_value_of_trig_expr_l553_553234

variable (x : ℝ)

theorem max_value_of_trig_expr : 
  (∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5) :=
sorry

end max_value_of_trig_expr_l553_553234


namespace largest_quotient_l553_553875

theorem largest_quotient (s : Set ℤ) (H : s = {-24, -3, -2, 1, 2, 8}) : 
  ∃ a b ∈ s, a / b = 12 ∧ (∀ c d ∈ s, c / d ≤ 12) := 
sorry

end largest_quotient_l553_553875


namespace find_m_l553_553314

noncomputable def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_m (m : ℝ) :
  let a := (1, m)
  let b := (3, -2)
  are_parallel (vector_sum a b) b → m = -2 / 3 :=
by
  sorry

end find_m_l553_553314


namespace candy_difference_l553_553201

-- Defining the conditions as Lean hypotheses
variable (R K B M : ℕ)

-- Given conditions
axiom h1 : K = 4
axiom h2 : B = M - 6
axiom h3 : M = R + 2
axiom h4 : K = B + 2

-- Prove that Robert gets 2 more pieces of candy than Kate
theorem candy_difference : R - K = 2 :=
by {
  sorry
}

end candy_difference_l553_553201


namespace range_of_a_l553_553316

noncomputable def vector_m (x : ℝ) : ℝ × ℝ := (Real.exp x + x^2/2, x)
noncomputable def vector_n (a : ℝ) : ℝ × ℝ := (2, a)
noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def f (x a : ℝ) : ℝ := dot_product (vector_m x) (vector_n a)

theorem range_of_a (h : ∀ x ∈ Ioo (-1 : ℝ) 0, 0 < ∂ (f x a) / ∂ x) : 
  ∃ (a : ℝ), ∀ x ∈ Ioo (-1 : ℝ) 0, a ≥ -2 :=
sorry

end range_of_a_l553_553316


namespace general_term_arithmetic_sequence_l553_553265

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n, a n = 2 * n - 1 := 
by
  intros h1 h2 n
  sorry

end general_term_arithmetic_sequence_l553_553265


namespace count_perfect_cubes_l553_553721

theorem count_perfect_cubes (a b : ℤ) (h₁ : 100 < a) (h₂ : b < 1000) : 
  ∃ n m : ℤ, (n^3 > 100 ∧ m^3 < 1000) ∧ m - n + 1 = 5 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end count_perfect_cubes_l553_553721


namespace intersection_of_A_and_B_l553_553713

open Set

variable (U : Set ℝ := Real)

def setA (x : ℝ) : Prop := x^2 - 2 * x - 3 ≤ 0
def setB (x : ℝ) : Prop := abs (x - 2) < 2

theorem intersection_of_A_and_B :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | 0 < x ∧ x ≤ 3} := by
  sorry

end intersection_of_A_and_B_l553_553713


namespace remainder_when_divided_by_14_l553_553173

theorem remainder_when_divided_by_14 (A : ℕ) (h1 : A % 1981 = 35) (h2 : A % 1982 = 35) : A % 14 = 7 :=
sorry

end remainder_when_divided_by_14_l553_553173


namespace liked_new_menu_l553_553588

theorem liked_new_menu (total_students : ℕ) (students_not_liked : ℕ) : 
  total_students = 400 → students_not_liked = 165 → (total_students - students_not_liked) = 235 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end liked_new_menu_l553_553588


namespace correct_function_satisfies_condition_l553_553593

noncomputable def f_A (x : ℝ) : ℝ := -x + 1
noncomputable def f_B (x : ℝ) : ℝ := 2x
noncomputable def f_C (x : ℝ) : ℝ := x^2 - 1
noncomputable def f_D (x : ℝ) : ℝ := Real.log (-x)

theorem correct_function_satisfies_condition :
  ∀ (x1 x2 : ℝ), x1 < 0 ∧ x2 < 0 ∧ x1 < x2 → f_B x1 < f_B x2 := by
  sorry

end correct_function_satisfies_condition_l553_553593


namespace locus_circumcenter_l553_553750

-- Given: a triangle ABC
variables {A B C P Q H R : Type}

-- feet of the altitudes from B and A
variable [IsAltitude (B) (C) P]
variable [IsAltitude (A) (C) Q]

-- condition on angle \(\angle ACB = 60^\circ\)
variable [AngleACB60 (A) (C) (B)]

-- Prove: the locus of the circumcenter R of triangle PQC is a circle with center at the midpoint of AB and radius equal to the radius of the circumcircle of triangle ABC
theorem locus_circumcenter (A B C P Q H R : Point)
  (h1 : orthocenter A B C H)
  (h2 : cyclic (C P H Q))
  (h3 : ∠ ACB = 60°) :
  locus_circumcenter (triangle_circumcenter (P Q C)) = 
  Circle (midpoint (A, B)) (triangle_circumradius (A B C)) := 
sorry

end locus_circumcenter_l553_553750


namespace evaluate_at_10_l553_553705

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem evaluate_at_10 : f 10 = 756 := by
  -- the proof is omitted
  sorry

end evaluate_at_10_l553_553705


namespace birth_year_of_person_l553_553576

theorem birth_year_of_person (x : ℕ) (hx : x ∈ { n : ℕ | 42 < n ∧ n < 44 }) :
  (1849 - x = 1806) :=
by
  have hx_eq : x = 43 := sorry,
  calc
  1849 - x = 1849 - 43 : by rw hx_eq
    ... = 1806 : by norm_num

end birth_year_of_person_l553_553576


namespace bo_dot_ac_l553_553272

theorem bo_dot_ac (O A B C : Point) (BA BC AC BO : Vector) 
  (circumcenter : is_circumcenter O A B C)
  (h1 : |BA| = 2)
  (h2 : |BC| = 6)
  (h3 : BO = get_vector O B)
  (h4 : AC = get_vector A C) :
  BO ⬝ AC = 16 :=
by
  sorry

end bo_dot_ac_l553_553272


namespace tangent_line_to_circle_l553_553344

theorem tangent_line_to_circle (a : ℝ) (h : ∀ x y : ℝ, (x-1)^2 + y^2 = 1 → (1+a)*x + y + 1 = 0 → |1 + a + 1| / sqrt((1 + a)^2 + 1) = 1) : 
  a = -1 :=
by
  sorry

end tangent_line_to_circle_l553_553344


namespace convert_to_base13_l553_553618

theorem convert_to_base13 (n : ℕ) (h : n = 157) : 
  ∃ (a b : ℕ), a = 12 ∧ b = 1 ∧ n = a * 13 + b :=
by
  use 12
  use 1
  split
  . exact rfl
  split
  . exact rfl
  . rw [h, Nat.mul_comm 13 12]
    norm_num

end convert_to_base13_l553_553618


namespace circles_touch_at_X_l553_553112

variables {A B C P Q R X : Type}
variables [EuclideanGeometry A B C P Q R X]

-- Given conditions
axiom angle_A_gt_angle_C : ∠A > ∠C
axiom point_P_angle_PAC_eq_angle_C : ∠PAC = ∠C
axiom point_Q_outside_triangle_BQ_parallel_AC : (outside_triangle Q ∧ BQ ↔ AC)
axiom PQ_parallel_AB : PQ ↔ AB
axiom point_R_on_AC_angle_PRQ_eq_angle_C : (on_line AC R ∧ ∠PRQ = ∠C)
axiom BQ_intersects_AP_at_X : intersects BQ AP X

-- Problem: Show that the circles ABC and PQR touch at point X
theorem circles_touch_at_X
  (hAB : ¬ collinear A B C)
  (hQR : ¬ collinear P Q R) :
  Circle.touch (circumcircle A B C) (circumcircle P Q R) X :=
sorry

end circles_touch_at_X_l553_553112


namespace ellipse_foci_cond_l553_553088

theorem ellipse_foci_cond (m n : ℝ) (h_cond : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → (m > n ∧ n > 0)) ∧ ((m > n ∧ n > 0) → ∀ x y : ℝ, mx^2 + ny^2 = 1) :=
sorry

end ellipse_foci_cond_l553_553088


namespace find_x_plus_y_l553_553348

noncomputable def segment_AB := 10
noncomputable def segment_A'B' := 5
noncomputable def D := segment_AB / 2
noncomputable def D' := segment_A'B' / 2
noncomputable def x := 3
noncomputable def proportion := segment_A'B' / segment_AB
noncomputable def y := x * proportion

theorem find_x_plus_y :
  x + y = 4.5 := by
  sorry

end find_x_plus_y_l553_553348


namespace decrease_in_average_age_l553_553470

-- Definition of initial conditions
def original_average_age (A : ℝ) := A
def total_age_before_replacement (A : ℝ) := 10 * A
def age_of_person_replaced : ℝ := 40
def age_of_new_person : ℝ := 10
def total_age_after_replacement (A : ℝ) := total_age_before_replacement A - (age_of_person_replaced - age_of_new_person)
def new_average_age (A : ℝ) := total_age_after_replacement A / 10

-- Lean 4 statement of the proof problem
theorem decrease_in_average_age (A : ℝ) :
  original_average_age A - new_average_age A = 3 := 
by
  sorry

end decrease_in_average_age_l553_553470


namespace fred_initial_dimes_l553_553652

theorem fred_initial_dimes (current_dimes borrowed_dimes initial_dimes : ℕ)
  (hc : current_dimes = 4)
  (hb : borrowed_dimes = 3)
  (hi : current_dimes + borrowed_dimes = initial_dimes) :
  initial_dimes = 7 := 
by
  sorry

end fred_initial_dimes_l553_553652


namespace polynomial_count_is_four_l553_553591

theorem polynomial_count_is_four :
  let expressions := [λ x : ℝ, 1 / x, λ (x y : ℝ), 2 * x + y,
                      λ (a b : ℝ), (1 / 3) * a^2 * b, λ (x y : ℝ), (x - y) / Real.pi,
                      λ (x y : ℝ), (5 * y) / (4 * x), λ x : ℝ, 0]
  in (count_polynomials expressions) = 4 :=
by
  sorry

def count_polynomials (expressions : List (ℝ → ℝ)) : Nat :=
  expressions.filter (λ f, is_polynomial f).length

def is_polynomial (f : ℝ → ℝ) : Prop :=
  -- Assumes implementation of polynomial check logic that matches
  -- the mathematical definition described in the problem.
  sorry

end polynomial_count_is_four_l553_553591


namespace find_pairs_l553_553858

theorem find_pairs:
∃ (m1 n1 m2 n2 m3 n3: ℕ), 
    m1 + (m1 + 1) + (m1 + 2) + ... + n1 = m1 * n1 ∧
    m2 + (m2 + 1) + (m2 + 2) + ... + n2 = m2 * n2 ∧
    m3 + (m3 + 1) + (m3 + 2) + ... + n3 = m3 * n3 ∧
    (m1, n1), (m2, n2), (m3, n3) = (493, 1189), (2871, 6930), (16731, 40391) := 
  sorry

end find_pairs_l553_553858


namespace sum_square_formula_l553_553433

theorem sum_square_formula (n : ℕ) (hn : n > 0) :
    ∑ i in Finset.range n, (i + 1) ^ 2 = n * (n + 1) * (2 * n + 1) / 6 :=
sorry

end sum_square_formula_l553_553433


namespace absolute_value_inequality_solution_l553_553109

theorem absolute_value_inequality_solution (x : ℝ) : abs (x - 3) < 2 ↔ 1 < x ∧ x < 5 :=
by
  sorry

end absolute_value_inequality_solution_l553_553109


namespace infinite_divisors_sum_powers_l553_553776

theorem infinite_divisors_sum_powers (k : ℕ) (h : k > 1) :
  ∃ᶠ n in filter.at_top, n ∣ (finset.range (k + 1)).sum (λ i, i ^ n) :=
sorry

end infinite_divisors_sum_powers_l553_553776


namespace necessary_but_not_sufficient_condition_l553_553033

theorem necessary_but_not_sufficient_condition
  (A : set ℝ) (B : set ℝ) 
  (hA : A = set.Icc 0 real.pi)
  (hB : B = set.Icc 0 (real.pi / 2))
  (a : ℝ) :
  a ∈ A → a ∈ B ∧ ¬ (a ∈ B → a ∈ A) :=
by
  sorry

end necessary_but_not_sufficient_condition_l553_553033


namespace max_consecutive_interesting_numbers_l553_553411

-- Define what it means for a number to be interesting
def is_interesting (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧ n = p1 * p2

-- Define the property we want to prove
theorem max_consecutive_interesting_numbers :
  ∀ (s : List ℕ), (∀ n ∈ s, is_interesting n) → (∀ n ∈ s, List.length s ≤ 3 ∨ (∃ k, s = [k, k+1, k+2, k+3] ∧ (∃ p, k + 1 = 4 * p))) :=
begin
  sorry,
end

end max_consecutive_interesting_numbers_l553_553411


namespace total_outfits_l553_553464

theorem total_outfits (n_shirts n_pants n_ties : ℕ) (h_shirts : n_shirts = 5) (h_pants : n_pants = 4) (h_ties : n_ties = 4) : 
  n_shirts * n_pants * n_ties = 80 :=
by {
  rw [h_shirts, h_pants, h_ties],
  norm_num,
  sorry
}

end total_outfits_l553_553464


namespace find_number_l553_553885

theorem find_number (x : ℤ) (h : x - 7 = 9) : x * 3 = 48 :=
by sorry

end find_number_l553_553885


namespace total_cost_textbooks_l553_553416

theorem total_cost_textbooks :
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  sale_books + online_books + bookstore_books = 210 :=
by
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  show sale_books + online_books + bookstore_books = 210
  sorry

end total_cost_textbooks_l553_553416


namespace complement_set_U_A_l553_553714

-- Definitions of U and A
def U : Set ℝ := { x : ℝ | x^2 ≤ 4 }
def A : Set ℝ := { x : ℝ | |x - 1| ≤ 1 }

-- Theorem statement
theorem complement_set_U_A : (U \ A) = { x : ℝ | -2 ≤ x ∧ x < 0 } := 
by
  sorry

end complement_set_U_A_l553_553714


namespace polynomial_with_root_l553_553237

noncomputable def poly : ℂ[X] := X^2 - 4*X + 13

theorem polynomial_with_root (x : ℂ) :
  (monic poly) ∧ 
  (∀ a b : ℂ, poly.coeff 0 = a + b*I → a ∈ ℝ ∧ b ∈ ℝ) ∧
  (poly.eval (2 + 3*I) = 0) :=
begin
  sorry
end

end polynomial_with_root_l553_553237


namespace carol_wins_3_games_l553_553590

theorem carol_wins_3_games (alice_wins : ℕ) (alice_losses : ℕ) (bob_wins : ℕ) (bob_losses : ℕ) (carol_losses : ℕ) :
  alice_wins = 5 → alice_losses = 3 → bob_wins = 4 → bob_losses = 4 → carol_losses = 5 → 
  ∃ (carol_wins : ℕ), carol_wins = 3 :=
by {
  intros,
  sorry
}

end carol_wins_3_games_l553_553590


namespace f_satisfies_condition_l553_553726

noncomputable def f (x : ℝ) : ℝ := 2^x

-- Prove that f(x + 1) = 2 * f(x) for the defined function f.
theorem f_satisfies_condition (x : ℝ) : f (x + 1) = 2 * f x := by
  show 2^(x + 1) = 2 * 2^x
  sorry

end f_satisfies_condition_l553_553726


namespace percentage_of_customers_purchased_l553_553376

theorem percentage_of_customers_purchased (ad_cost : ℕ) (customers : ℕ) (price_per_sale : ℕ) (profit : ℕ)
  (h1 : ad_cost = 1000)
  (h2 : customers = 100)
  (h3 : price_per_sale = 25)
  (h4 : profit = 1000) :
  (profit / price_per_sale / customers) * 100 = 40 :=
by
  sorry

end percentage_of_customers_purchased_l553_553376


namespace quadratic_roots_relation_l553_553489

theorem quadratic_roots_relation (m p q : ℝ) (h_m_ne_zero : m ≠ 0) (h_p_ne_zero : p ≠ 0) (h_q_ne_zero : q ≠ 0) :
  (∀ r1 r2 : ℝ, (r1 + r2 = -q ∧ r1 * r2 = m) → (3 * r1 + 3 * r2 = -m ∧ (3 * r1) * (3 * r2) = p)) →
  p / q = 27 :=
by
  intros h
  sorry

end quadratic_roots_relation_l553_553489


namespace heaviest_vs_lightest_total_excess_shortfall_total_profit_l553_553114

-- Definitions for the conditions
def standard_weight : ℕ := 25
def num_baskets : ℕ := 20

def weight_differences : List (ℤ × ℕ) :=
  [(-3, 1), (-2, 4), (-1, 2), (0, 3), (1, 2), (2, 8)]

def cost_price_per_kg : ℝ := 1.6
def selling_price_per_kg : ℝ := 2.0

-- Proof problem statements as Lean theorems
theorem heaviest_vs_lightest (h_lightest : -3) (h_heaviest : 2) :
  10 * h_heaviest - (-10 * h_lightest) = 5 :=
by 
  sorry

theorem total_excess_shortfall :
  (weight_differences.map (fun (d, n) => d * n)).sum = 5 :=
by 
  sorry

theorem total_profit :
  let total_weight := (num_baskets * standard_weight : ℝ) + 5
  let profit_per_kg := selling_price_per_kg - cost_price_per_kg
  profit_per_kg * total_weight = 202 :=
by
  sorry

end heaviest_vs_lightest_total_excess_shortfall_total_profit_l553_553114


namespace travel_west_3_km_l553_553336

-- Define the condition
def east_travel (km: ℕ) : ℤ := km

-- Define the function for westward travel
def west_travel (km: ℕ) : ℤ := - (km)

-- Specify the theorem we want to prove
theorem travel_west_3_km :
  west_travel 3 = -3 :=
by {
  apply rfl,
  sorry
}

end travel_west_3_km_l553_553336


namespace probability_calculation_l553_553287

noncomputable def probability_distribution (n : ℕ) (a : ℚ) : ℚ :=
  a / (n * (n + 1))

theorem probability_calculation : 
  (∀ n ∈ {1, 2, 3, 4}, probability_distribution n (5/4 : ℚ) = ∑ n in {1, 2, 3, 4}, probability_distribution n (5/4 : ℚ)) ∧ 
  (probability_distribution 1 (5/4) + probability_distribution 2 (5/4) = 5/6) → 
  ( ∑ n in {1, 2, 3, 4}, probability_distribution n (5/4 : ℚ) = 1 ) := 
by
  sorry

end probability_calculation_l553_553287


namespace election_votes_l553_553542

theorem election_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 200) : V = 500 :=
sorry

end election_votes_l553_553542


namespace twice_diff_function_sol_l553_553995

noncomputable def twice_differentiable_solutions (f : ℝ → ℝ) := 
  ∀ x y : ℝ, f(x)^2 - f(y)^2 = f(x + y) * f(x - y)

theorem twice_diff_function_sol (f : ℝ → ℝ) (hf : twice_differentiable_solutions f) :
  (∃ k : ℝ, ∀ x : ℝ, f(x) = k * x) ∨ 
  (∃ a c : ℝ, ∀ x : ℝ, f(x) = a * sin(c * x)) ∨ 
  (∃ a c : ℝ, ∀ x : ℝ, f(x) = a * sinh(c * x)) :=
sorry

end twice_diff_function_sol_l553_553995


namespace find_length_of_stone_slab_l553_553912

noncomputable def length_of_square_slab (total_area : ℝ) (num_slabs : ℕ) : ℝ :=
  let area_per_slab := total_area / num_slabs
  let area_per_slab_cm := area_per_slab * 10000
  real.sqrt area_per_slab_cm

theorem find_length_of_stone_slab :
  length_of_square_slab 58.8 30 = 140 :=
by
  sorry

end find_length_of_stone_slab_l553_553912


namespace sequence_a4_value_l553_553107

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = 2 * a n + 1) ∧ a 4 = 15 :=
by
  sorry

end sequence_a4_value_l553_553107


namespace largest_of_three_numbers_l553_553863

noncomputable def largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -20) : ℝ :=
  max p (max q r)

theorem largest_of_three_numbers (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -8) 
  (h3 : p * q * r = -20) :
  largest_root p q r h1 h2 h3 = ( -1 + Real.sqrt 21 ) / 2 :=
by
  sorry

end largest_of_three_numbers_l553_553863


namespace village_population_l553_553913

theorem village_population (P : ℕ) (h : 0.80 * P = 23040) : P = 28800 := 
by 
  sorry

end village_population_l553_553913


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l553_553282

variable {a b m : ℝ}

theorem sufficient_but_not_necessary_condition (h : a * m^2 < b * m^2) : a < b := by
  sorry

-- Additional statements to express the sufficiency and not necessity nature:
theorem not_necessary_condition (h : a < b) (hm : m = 0) : ¬ (a * m^2 < b * m^2) := by
  sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l553_553282


namespace correct_statements_l553_553145

namespace ComplexProof

open Complex

noncomputable def statementA (z₁ z₂ : ℂ) : Prop :=
  (|z₁ - z₂| = 0) → (conj z₁ = conj z₂)

noncomputable def statementB (z₁ z₂ : ℂ) : Prop :=
  (z₁ = conj z₂) → (conj z₁ = z₂)

noncomputable def statementC (z₁ z₂ : ℂ) : Prop :=
  (|z₁| = |z₂|) → (z₁ * conj z₁ = z₂ * conj z₂)

noncomputable def statementD (z₁ z₂ : ℂ) : Prop :=
  (∃ z₁ z₂ : ℂ, |z₁| = |z₂| ∧ z₁^2 ≠ z₂^2)

theorem correct_statements (z₁ z₂ : ℂ) :
  statementA z₁ z₂ ∧ statementB z₁ z₂ ∧ statementC z₁ z₂ ∧ statementD z₁ z₂ :=
by
sory

end ComplexProof

end correct_statements_l553_553145


namespace probability_at_least_one_red_ball_l553_553513

-- Define the problem conditions
def balls_in_box_A := {red_ball_1, red_ball_2, white_ball}
def balls_in_box_B := {red_ball_3, red_ball_4, white_ball}

-- Define the random drawing event from each box
def event (A B : Set ball) : Set (ball × ball) := { (a, b) | a ∈ A ∧ b ∈ B }

-- Total probability space
def total_outcomes := event balls_in_box_A balls_in_box_B

-- Define event where no red balls are drawn: {white_ball from box A, white_ball from box B}
def no_red_event := {(white_ball, white_ball)}

-- Probability calculation
def P (E : Set (ball × ball)) : ℝ := (E.card / total_outcomes.card : ℝ)

-- Theorem statement for the problem
theorem probability_at_least_one_red_ball :
  P (total_outcomes \ no_red_event) = 8 / 9 :=
by
  sorry

end probability_at_least_one_red_ball_l553_553513


namespace scientific_notation_of_coronavirus_diameter_l553_553827

theorem scientific_notation_of_coronavirus_diameter :
  (0.00000011 : ℝ) = 1.1 * 10^(-7) :=
  sorry

end scientific_notation_of_coronavirus_diameter_l553_553827


namespace find_x_l553_553139

-- Let \( x \) be a real number.
variable (x : ℝ)

-- Condition given in the problem.
def condition : Prop := x = (3 / 7) * x + 200

-- The main statement to be proved.
theorem find_x (h : condition x) : x = 350 :=
  sorry

end find_x_l553_553139


namespace gcd_eq_gcd_of_eq_add_mul_l553_553802

theorem gcd_eq_gcd_of_eq_add_mul (a b q r : Int) (h_q : b > 0) (h_r : 0 ≤ r) (h_ar : a = b * q + r) : Int.gcd a b = Int.gcd b r :=
by
  -- Conditions: constraints and assertion
  exact sorry

end gcd_eq_gcd_of_eq_add_mul_l553_553802


namespace simplify_fraction_l553_553456

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l553_553456


namespace zoo_total_income_l553_553428

/-- The number of children visiting the zoo on Monday is 7. -/
def children_monday : ℕ := 7
/-- The number of adults visiting the zoo on Monday is 5. -/
def adults_monday : ℕ := 5
/-- The number of seniors visiting the zoo on Monday is 3. -/
def seniors_monday : ℕ := 3
/-- The number of student groups visiting the zoo on Monday is 2. -/
def student_groups_monday : ℕ := 2
/-- The number of children visiting the zoo on Tuesday is 9. -/
def children_tuesday : ℕ := 9
/-- The number of adults visiting the zoo on Tuesday is 6. -/
def adults_tuesday : ℕ := 6
/-- The number of seniors visiting the zoo on Tuesday is 2. -/
def seniors_tuesday : ℕ := 2
/-- The number of student groups visiting the zoo on Tuesday is 1. -/
def student_groups_tuesday : ℕ := 1

/-- Ticket prices on Monday:
- Children: $3
- Adults: $4
- Seniors: $3
- Student groups: $25
-/
def ticket_price_child_monday : ℕ := 3
def ticket_price_adult_monday : ℕ := 4
def ticket_price_senior_monday : ℕ := 3
def ticket_price_student_group_monday : ℕ := 25
/-- Special promotion on Monday: children pay $2 if combined with an adult limit to 3 discounts. -/
def discounted_child_ticket_monday : ℕ := 2
def limit_combined_discount : ℕ := 3

/-- Ticket prices on Tuesday:
- Children: $4
- Adults: $5
- Seniors: $3
- Student groups: $30
-/
def ticket_price_child_tuesday : ℕ := 4
def ticket_price_adult_tuesday : ℕ := 5
def ticket_price_senior_tuesday : ℕ := 3
def ticket_price_student_group_tuesday : ℕ := 30
/-- Special promotion on Tuesday: 10% discount on total cost excluding student groups. -/
def discount_tuesday_percent : ℕ := 10

/-- Compute the total amount of money made by the zoo on Monday and Tuesday. -/
def total_monday := 
  (min limit_combined_discount adults_monday * discounted_child_ticket_monday + 
  (children_monday - min limit_combined_discount adults_monday) * ticket_price_child_monday) +
  (adults_monday * ticket_price_adult_monday) +
  (seniors_monday * ticket_price_senior_monday) +
  (student_groups_monday * ticket_price_student_group_monday)

def total_tuesday := 
  ((children_tuesday * ticket_price_child_tuesday) +
  (adults_tuesday * ticket_price_adult_tuesday) +
  (seniors_tuesday * ticket_price_senior_tuesday) * 
  (100 - discount_tuesday_percent) / 100) +
  (student_groups_tuesday * ticket_price_student_group_tuesday)

def total := total_monday + total_tuesday

theorem zoo_total_income :
  total = 191.8 := sorry

end zoo_total_income_l553_553428


namespace exponential_inequality_l553_553258

-- Define the conditions for the problem
variables {x y a : ℝ}
axiom h1 : x > y
axiom h2 : y > 1
axiom h3 : 0 < a
axiom h4 : a < 1

-- State the problem to be proved
theorem exponential_inequality (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : a ^ x < a ^ y :=
sorry

end exponential_inequality_l553_553258


namespace new_average_daily_production_l553_553248

theorem new_average_daily_production (n : ℕ) (avg_past_n_days : ℕ) (today_production : ℕ) (h1 : avg_past_n_days = 50) (h2 : today_production = 90) (h3 : n = 9) : 
  (avg_past_n_days * n + today_production) / (n + 1) = 54 := 
by
  sorry

end new_average_daily_production_l553_553248


namespace find_a_4_l553_553676

variable {a : ℕ → ℝ}

-- Condition: a_2 = 2
def a_2 : Prop := a 2 = 2

-- Condition: a_6 = 0
def a_6 : Prop := a 6 = 0

-- Condition: {1 / (a_n + 1)} forms an arithmetic sequence
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
∀ n m : ℕ, m < n → ∃ d, s n = s m + d * (n - m)

def seq_arith : Prop := is_arithmetic_sequence (λ n, 1 / (a n + 1))

-- Goal: Prove that a_4 = 1 / 2
theorem find_a_4 (h1 : a_2) (h2 : a_6) (h3 : seq_arith) : a 4 = 1 / 2 := 
sorry

end find_a_4_l553_553676


namespace possible_values_of_a_l553_553412

variables {a b k : ℤ}

def sum_distances (a : ℤ) (k : ℤ) : ℤ :=
  (a - k).natAbs + (a - (k + 1)).natAbs + (a - (k + 2)).natAbs +
  (a - (k + 3)).natAbs + (a - (k + 4)).natAbs + (a - (k + 5)).natAbs +
  (a - (k + 6)).natAbs + (a - (k + 7)).natAbs + (a - (k + 8)).natAbs +
  (a - (k + 9)).natAbs + (a - (k + 10)).natAbs

theorem possible_values_of_a :
  sum_distances a k = 902 →
  sum_distances b k = 374 →
  a + b = 98 →
  a = 25 ∨ a = 107 ∨ a = -9 :=
sorry

end possible_values_of_a_l553_553412


namespace distinct_numbers_6x6_impossible_l553_553371

theorem distinct_numbers_6x6_impossible :
  ∀(f : ℕ × ℕ → ℕ),
  (∀ x y : ℕ, x < 6 → y < 6 → f (x, y) < 36) →
  (∀ x y₁ y₂ y₃ y₄ y₅, y₁ < 6 → y₂ < 6 → y₃ < 6 → y₄ < 6 → y₅ < 6 →
   list.nodup [f (x, y₁), f (x, y₂), f (x, y₃), f (x, y₄), f (x, y₅)] →
   f (x, y₁) + f (x, y₂) + f (x, y₃) + f (x, y₄) + f (x, y₅) = 2022 ∨
   f (x, y₁) + f (x, y₂) + f (x, y₃) + f (x, y₄) + f (x, y₅) = 2023) →
  (∀ y x₁ x₂ x₃ x₄ x₅, x₁ < 6 → x₂ < 6 → x₃ < 6 → x₄ < 6 → x₅ < 6 →
   list.nodup [f (x₁, y), f (x₂, y), f (x₃, y), f (x₄, y), f (x₅, y)] →
   f (x₁, y) + f (x₂, y) + f (x₃, y) + f (x₄, y) + f (x₅, y) = 2022 ∨
   f (x₁, y) + f (x₂, y) + f (x₃, y) + f (x₄, y) + f (x₅, y) = 2023) →
  false :=
by
  sorry

end distinct_numbers_6x6_impossible_l553_553371


namespace total_profit_is_27_l553_553798

noncomputable def total_profit : ℕ :=
  let natasha_money := 60
  let carla_money := natasha_money / 3
  let cosima_money := carla_money / 2
  let sergio_money := 3 * cosima_money / 2

  let natasha_spent := 4 * 15
  let carla_spent := 6 * 10
  let cosima_spent := 5 * 8
  let sergio_spent := 3 * 12

  let natasha_profit := natasha_spent * 10 / 100
  let carla_profit := carla_spent * 15 / 100
  let cosima_profit := cosima_spent * 12 / 100
  let sergio_profit := sergio_spent * 20 / 100

  natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_is_27 : total_profit = 27 := by
  sorry

end total_profit_is_27_l553_553798


namespace special_op_eight_four_l553_553838

def special_op (a b : ℕ) : ℕ := 2 * a + a / b

theorem special_op_eight_four : special_op 8 4 = 18 := by
  sorry

end special_op_eight_four_l553_553838


namespace ratio_of_areas_ADB_CDB_l553_553432

-- Define the setup of the equilateral triangle and point D on AC
variables {A B C D : Type*} [DecidableEq A] [DecidableEq B] [DecidableEq C]
    [DecidableEq D] [MetricSpace A] [MetricSpace B]
    [MetricSpace C] [MetricSpace D]

-- Define the equilateral triangle ABC with side length s
structure EquilateralTriangle (s : ℝ) :=
  (A B C : Point)
  (is_equilateral : dist A B = s ∧ dist B C = s ∧ dist C A = s)

-- Define a point D on AC and the measure of angle DBC
structure PointOnAC (T : EquilateralTriangle s) :=
  (D : Point)
  (on_AC : D ∈ segment T.A T.C)
  (angle_DBC_30 : ∠ T.D T.B T.C = π / 6)

-- The goal is to formally state the ratio problem and the required proof.
theorem ratio_of_areas_ADB_CDB
  (T : EquilateralTriangle s)
  (hD : PointOnAC T) :
  (area (triangle T.A T.D T.B)) / (area (triangle T.C T.D T.B)) = (3 - real.sqrt 3) / (2 * real.sqrt 3) :=
sorry

end ratio_of_areas_ADB_CDB_l553_553432


namespace total_money_shared_l553_553421

-- Conditions
def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

-- Question and proof to be demonstrated
theorem total_money_shared : ken_share + tony_share = 5250 :=
by sorry

end total_money_shared_l553_553421


namespace deanna_initial_speed_l553_553246

namespace TripSpeed

variables (v : ℝ) (h : v > 0)

def speed_equation (v : ℝ) : Prop :=
  (1/2 * v) + (1/2 * (v + 20)) = 100

theorem deanna_initial_speed (v : ℝ) (h : speed_equation v) : v = 90 := sorry

end TripSpeed

end deanna_initial_speed_l553_553246


namespace interval_of_monotonic_decrease_range_of_m_l553_553656

open Real

-- Define vectors and function f(x)
def vec_a (x : ℝ) : ℝ × ℝ := (2 * sqrt (3) * sin x, sin x + cos x)
def vec_b (x : ℝ) : ℝ × ℝ := (cos x, sin x - cos x)
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

-- Define the conditions and their implications as Lean statements
theorem interval_of_monotonic_decrease (k : ℤ) :
  k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6 → (∃ k : ℤ, k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6) :=
sorry

theorem range_of_m (a b c A m : ℝ) (h1 : b^2 + a^2 - c^2 = ab) (h2 : f(A) - m > 0)
  (h3 : 0 < A) (h4 : A < 2 * π / 3) :
  m ≤ -1 :=
sorry

end interval_of_monotonic_decrease_range_of_m_l553_553656


namespace periodic_and_even_l553_553950

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def y1 := fun x : ℝ => 10 ^ x
def y2 := fun x : ℝ => Real.tan x
def y3 := fun x : ℝ => Real.sin (2 * x)
def y4 := fun x : ℝ => abs (Real.cos x)

theorem periodic_and_even : is_periodic y4 Real.pi ∧ is_even y4 :=
by
  sorry

end periodic_and_even_l553_553950


namespace average_salary_l553_553821

theorem average_salary
  (num_technicians : ℕ) (avg_salary_technicians : ℝ)
  (num_other_workers : ℕ) (avg_salary_other_workers : ℝ)
  (total_num_workers : ℕ) (avg_salary_all_workers : ℝ) :
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  num_other_workers = total_num_workers - num_technicians →
  avg_salary_other_workers = 6000 →
  total_num_workers = 28 →
  avg_salary_all_workers = (num_technicians * avg_salary_technicians + num_other_workers * avg_salary_other_workers) / total_num_workers →
  avg_salary_all_workers = 8000 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l553_553821


namespace otimes_4_8_l553_553102

-- Define the operation ⊗
def otimes (a b : ℝ) : ℝ := a / b + b / a

-- State the theorem to be proved
theorem otimes_4_8 : otimes 4 8 = 5 / 2 := by
  -- Provide the necessary proof steps or include 'sorry' to skip proof
  sorry

end otimes_4_8_l553_553102


namespace area_of_regions_l553_553552

variables {X Y X1 Y1 X2 Y2 : ℝ × ℝ}
variables {C : set (ℝ × ℝ)}
variables {θ : ℝ}

-- Conditions of the problem
def is_on_circle (P : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2

def X_on_circle : Prop := is_on_circle X (0, 0) 1
def Y_on_circle : Prop := is_on_circle Y (0, 0) 1
def X1_on_x_axis : Prop := X1 = (X.1, 0)
def Y1_on_x_axis : Prop := Y1 = (Y.1, 0)
def X2_on_y_axis : Prop := X2 = (0, X.2)
def Y2_on_y_axis : Prop := Y2 = (0, Y.2)

-- Target conclusion
theorem area_of_regions (X Y X1 Y1 X2 Y2 : ℝ × ℝ) (C : set (ℝ × ℝ)) (θ : ℝ):
  X_on_circle ∧ Y_on_circle ∧ X1_on_x_axis ∧ Y1_on_x_axis ∧ X2_on_y_axis ∧ Y2_on_y_axis →
  (area_XYY1X1 X Y X1 Y1 + area_XYY2X2 X Y X2 Y2 = θ) :=
begin
  sorry
end

end area_of_regions_l553_553552


namespace regular_polygon_sum_is_const_l553_553393

noncomputable def regular_polygon_sum_const {n : ℕ} (r : ℝ) (P : ℂ) (θ : ℝ) : Prop :=
  let vertices := λ k : ℕ, r * complex.exp (2 * complex.I * real.pi * (k / n))
  let P := r * complex.exp (complex.I * θ)
  ∑ k in finset.range n, complex.abs(P - vertices k)^4 = 6 * n * r^4

theorem regular_polygon_sum_is_const (n : ℕ) (r : ℝ) (P : ℂ) (θ : ℝ) :
  regular_polygon_sum_const n r P θ := sorry

end regular_polygon_sum_is_const_l553_553393


namespace scientific_notation_14nm_l553_553621

theorem scientific_notation_14nm :
  (0.000000014 : ℝ) = 1.4 * 10^(-8) := 
sorry

end scientific_notation_14nm_l553_553621


namespace plant_arrangement_count_l553_553960

theorem plant_arrangement_count : 
  let basil_plants := 3
  let tomato_plants := 3
  let pepper_plants := 2
  let basil_arrangements := fact basil_plants
  let choose_tomato_groups := Nat.choose tomato_plants 2
  let arrange_two_tomato := fact 2
  let arrange_one_tomato := fact 1
  let slots_for_tomato_groups := 4
  let choose_slots := Nat.choose slots_for_tomato_groups 2
  let pepper_arrangements := fact pepper_plants
  (basil_arrangements * choose_tomato_groups * arrange_two_tomato * arrange_one_tomato * choose_slots * pepper_arrangements) = 432 :=
by
  sorry

end plant_arrangement_count_l553_553960


namespace diana_hourly_wage_l553_553537

theorem diana_hourly_wage :
  (∃ (hours_monday : ℕ) (hours_tuesday : ℕ) (hours_wednesday : ℕ) (hours_thursday : ℕ) (hours_friday : ℕ) (weekly_earnings : ℝ),
    hours_monday = 10 ∧
    hours_tuesday = 15 ∧
    hours_wednesday = 10 ∧
    hours_thursday = 15 ∧
    hours_friday = 10 ∧
    weekly_earnings = 1800 ∧
    (weekly_earnings / (hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday) = 30)) :=
sorry

end diana_hourly_wage_l553_553537


namespace product_of_metapolynomials_l553_553871

def is_metapolynomial (f : ℝ^k → ℝ) : Prop :=
  ∃ (m n : ℕ) (P : fin m → fin n → (ℝ^k) → ℝ), 
    f = λ x, max (λ i, min (λ j, P i j x))

variables {k : ℕ} (f g : ℝ^k → ℝ)

theorem product_of_metapolynomials (hf : is_metapolynomial f) (hg : is_metapolynomial g) : 
  is_metapolynomial (λ x, f x * g x) :=
sorry

end product_of_metapolynomials_l553_553871


namespace regular_polygon_interior_angle_not_integer_count_l553_553784

theorem regular_polygon_interior_angle_not_integer_count :
  let is_non_integer_angle (n : ℕ) : Prop := ¬ (180 * (n - 2) / n).den = 1
  let valid_n := {n | 3 ≤ n ∧ n < 12}
  (finset.univ.filter (λ (n : ℕ), n ∈ valid_n ∧ is_non_integer_angle n)).card = 2 :=
by
  sorry

end regular_polygon_interior_angle_not_integer_count_l553_553784


namespace floor_area_ring_l553_553990

noncomputable def circle_radius_40 := (40 : ℝ)

def inside_radius (s : ℝ) := s * (1 + Real.sqrt 2)

def area (s : ℝ) : ℝ :=
  let larger_circle_area := Real.pi * circle_radius_40^2
  let small_circle_area := 8 * Real.pi * s^2
  larger_circle_area - small_circle_area

theorem floor_area_ring (s : ℝ) (h_s : s * (1 + Real.sqrt 2) = circle_radius_40) : 
  Int.floor (area 40 (Real.sqrt 2 - 1)) = 1161 :=
sorry

end floor_area_ring_l553_553990


namespace transform_to_all_zeros_l553_553914

/-- A 4x4 board filled with 0s and 1s can be transformed to all 0s 
    using row, column, and diagonal flips if and only if the number 
    of 1s in the initial configuration is even. -/
theorem transform_to_all_zeros (board : Matrix (Fin 4) (Fin 4) ℕ) : 
    (∀ i j, board i j = 0 ∨ board i j = 1) → 
    (∃ flips : List (Fin 4 × Fin 4), 
        (∀ (i j : Fin 4), (List.foldr (fun ⟨r, c⟩ b => 
            if r = i ∨ c = j ∨ (i + j = r + c) ∨ (i - j = r - c) 
            then 1 - b else b) (board i j) flips) = 0)) ↔ 
    (board.entries.count 1 % 2 = 0) :=
by
  intros hboard
  sorry

end transform_to_all_zeros_l553_553914


namespace pairwise_function_equivalence_l553_553595

noncomputable def f_A (x : ℝ) : ℝ := x
noncomputable def g_A (x : ℝ) : ℝ := real.cbrt (x ^ 3)

noncomputable def f_B (x : ℝ) : ℝ := x + 1
noncomputable def g_B (x : ℝ) : ℝ := if x = 1 then 0 else (x ^ 2 - 1) / (x - 1)

noncomputable def f_C (x : ℝ) : ℝ := abs x / x
noncomputable def g_C (x : ℝ) : ℝ := if x > 0 then 1 else if x < 0 then -1 else 0

noncomputable def f_D (t : ℝ) : ℝ := abs (t - 1)
noncomputable def g_D (x : ℝ) : ℝ := abs (x - 1)

theorem pairwise_function_equivalence :
  (∀ x, f_A x = g_A x) ∧
  (∀ x, f_B x = g_B x → x ≠ 1) ∧
  (∀ x, f_C x = g_C x) ∧
  (∀ t, f_D t = g_D t) :=
by
  sorry

end pairwise_function_equivalence_l553_553595


namespace find_b_l553_553831

def point := ℝ × ℝ

def dir_vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def scale_vector (v : point) (s : ℝ) : point := (s * v.1, s * v.2)

theorem find_b (p1 p2 : point) (b : ℝ) :
  p1 = (-5, 0) → p2 = (-2, 2) →
  dir_vector p1 p2 = (3, 2) →
  scale_vector (3, 2) (2 / 3) = (2, b) →
  b = 4 / 3 :=
by
  intros h1 h2 h3 h4
  sorry

end find_b_l553_553831


namespace ratio_right_to_left_l553_553579

theorem ratio_right_to_left (L C R : ℕ) (hL : L = 12) (hC : C = L + 2) (hTotal : L + C + R = 50) :
  R / L = 2 :=
by
  sorry

end ratio_right_to_left_l553_553579


namespace total_snakes_in_park_l553_553859

theorem total_snakes_in_park :
  ∀ (pythons boa_constrictors rattlesnakes total_snakes : ℕ),
    boa_constrictors = 40 →
    pythons = 3 * boa_constrictors →
    rattlesnakes = 40 →
    total_snakes = boa_constrictors + pythons + rattlesnakes →
    total_snakes = 200 :=
by
  intros pythons boa_constrictors rattlesnakes total_snakes h1 h2 h3 h4
  rw [h1, h3] at h4
  rw [h2] at h4
  sorry

end total_snakes_in_park_l553_553859


namespace smallest_n_for_T_n_integer_l553_553006

noncomputable def K : ℚ := ∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n - 1)) * K + 1

theorem smallest_n_for_T_n_integer : ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m : ℕ, T_n m ∈ ℤ → n ≤ m :=
  ⟨504, sorry⟩

end smallest_n_for_T_n_integer_l553_553006


namespace point_A_coords_l553_553231

theorem point_A_coords (x y : ℝ) (h : ∀ t : ℝ, (t + 1) * x - (2 * t + 5) * y - 6 = 0) : x = -4 ∧ y = -2 := by
  sorry

end point_A_coords_l553_553231


namespace general_term_sequence_a_sum_of_sequence_b_l553_553291

-- Define the sequence {a_n} and state the given condition.
def sequence_a (n : ℕ) : ℕ := 3^(n-1)

-- We are to prove that the general term of the sequence {a_n} is 3^(n-1)
theorem general_term_sequence_a (n : ℕ) : sequence_a n = 3^(n-1) :=
sorry

-- Define the sequence {b_n} with the given condition
def sequence_b (n : ℕ) : ℝ := 3 / (n*(n+1))

-- Define the sum of the first n terms of {b_n}
def sum_b (n : ℕ) : ℝ := 3 * (1 - 1/(n+1))

-- We are to prove that the sum of the first n terms of {b_n} is 3n / (n+1)
theorem sum_of_sequence_b (n : ℕ) : sum_b n = 3 * n / (n + 1) :=
sorry

end general_term_sequence_a_sum_of_sequence_b_l553_553291


namespace max_value_trig_function_l553_553834

theorem max_value_trig_function :
  ∀ x ∈ set.Icc 0 (Real.pi / 2), 
  let f := fun x => (Real.sin x) ^ 2 + Real.sqrt 3 * (Real.cos x) - 3 / 4 
  in f x ≤ 1 :=
by
  -- This is where the proof would go
  sorry

end max_value_trig_function_l553_553834


namespace difference_between_M_and_m_l553_553225

-- Define the total student population
def total_students := 2500

-- Define the bounds for students studying Physics
def P_min := Int.ceil (0.70 * total_students)
def P_max := Int.floor (0.75 * total_students)

-- Define the bounds for students studying Chemistry
def C_min := Int.ceil (0.35 * total_students)
def C_max := Int.floor (0.45 * total_students)

-- Define the Principle of Inclusion-Exclusion
def inclusion_exclusion (P C Pc : Int) : Prop := P + C - Pc = total_students

-- Prove the difference between maximum M and minimum m for students studying both Physics and Chemistry is 375
theorem difference_between_M_and_m : 
  ∃ (m M : Int), 
    (inclusion_exclusion P_min C_min m) ∧ (inclusion_exclusion P_max C_max M) ∧ (M - m = 375) :=
  sorry 

end difference_between_M_and_m_l553_553225


namespace g_is_odd_l553_553764

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  sorry

end g_is_odd_l553_553764


namespace distance_range_l553_553091

theorem distance_range (A_school_distance : ℝ) (B_school_distance : ℝ) (x : ℝ)
  (hA : A_school_distance = 3) (hB : B_school_distance = 2) :
  1 ≤ x ∧ x ≤ 5 :=
sorry

end distance_range_l553_553091


namespace solution_set_of_inverse_inequality_l553_553666

open Function

variable {f : ℝ → ℝ}

theorem solution_set_of_inverse_inequality 
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_A : f (-2) = 2)
  (h_B : f 2 = -2)
  : { x : ℝ | |(invFun f (x + 1))| ≤ 2 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
sorry

end solution_set_of_inverse_inequality_l553_553666


namespace period_of_f_monotonic_intervals_of_f_l553_553701

def f (x : ℝ) : ℝ :=
  sqrt 3 * (sin x ^ 2 - cos x ^ 2) - 2 * sin x * cos x

theorem period_of_f : ∀ x, f (x + π) = f x := sorry

theorem monotonic_intervals_of_f:
  (∀ x y, -π/3 ≤ x ∧ x ≤ y ∧ y ≤ π/12 → f x ≥ f y) ∧
  (∀ x y, π/12 ≤ x ∧ x ≤ y ∧ y ≤ π/3 → f x ≤ f y) := sorry

end period_of_f_monotonic_intervals_of_f_l553_553701


namespace polynomial_form_l553_553623

theorem polynomial_form (P : ℝ → ℤ) 
  (hP : ∀ s t : ℝ, P s ∈ ℤ → P t ∈ ℤ → P (s * t) ∈ ℤ) :
  ∃ n : ℕ, ∃ k : ℤ, ∀ x : ℝ, P x = x^n + k :=
by
  sorry

end polynomial_form_l553_553623


namespace fourth_number_in_sequence_is_neg_97_sq_l553_553992

def alternating_sequence : List Int :=
  List.range' 1 100 |>.reverse.map (λ n, if n % 2 = 0 then n ^ 2 else -n ^ 2)

theorem fourth_number_in_sequence_is_neg_97_sq :
  alternating_sequence[3] = -97^2 :=
by
  sorry

end fourth_number_in_sequence_is_neg_97_sq_l553_553992


namespace lateral_surface_area_cone_l553_553842

theorem lateral_surface_area_cone (r l : ℝ) (h₀ : r = 6) (h₁ : l = 10) : π * r * l = 60 * π := by 
  sorry

end lateral_surface_area_cone_l553_553842


namespace exists_tangent_sphere_l553_553367

-- Definition of the types for points, lines, spheres, and polyhedra
-- Note: These are abstract definitions to facilitate the problem setup
structure Point := (x y z : ℝ)
structure Line := (A B : Point)
structure Sphere := (center : Point) (radius : ℝ)
structure Polyhedron := (edges : List Line)

-- Definition of the intersection segment length condition
def segment_intersect_condition (S : Sphere) (P : Polyhedron) : Prop :=
  ∀ edge ∈ P.edges, -- For every edge AB of P
    ∃ X Y : Point, -- There exist points X and Y such that
      segment_length (Line.mk A X) = segment_length (Line.mk X Y) ∧
      segment_length (Line.mk Y B) = (1 / 2) * segment_length edge ∧
      segment_length (Line.mk A B) = 2 * segment_length (Line.mk A X)

-- Definition of the key proof statement
theorem exists_tangent_sphere (S : Sphere) (P : Polyhedron) :
  segment_intersect_condition S P →
  ∃ T : Sphere, ∀ edge ∈ P.edges, tangent_to_sphere edge T :=
sorry

end exists_tangent_sphere_l553_553367


namespace triangle_third_side_l553_553353

/-- Given the lengths of two sides of a triangle as 3.14 and 0.67,
    prove that the third side, which is an integer, is equal to 3. -/
theorem triangle_third_side 
  (n : ℤ) 
  (h1 : real.gt 2.47 n) 
  (h2 : real.lt n 3.81) :
  n = 3 := 
sorry

end triangle_third_side_l553_553353


namespace proof_b_greater_a_greater_c_l553_553328

def a : ℤ := -2 * 3^2
def b : ℤ := (-2 * 3)^2
def c : ℤ := - (2 * 3)^2

theorem proof_b_greater_a_greater_c (ha : a = -18) (hb : b = 36) (hc : c = -36) : b > a ∧ a > c := 
by
  rw [ha, hb, hc]
  exact And.intro (by norm_num) (by norm_num)

end proof_b_greater_a_greater_c_l553_553328


namespace height_of_fourth_person_l553_553518

theorem height_of_fourth_person
  (h : ℝ)
  (H1 : h + (h + 2) + (h + 4) + (h + 10) = 4 * 79) :
  h + 10 = 85 :=
by
  have H2 : h + 4 = 79 := by linarith
  linarith


end height_of_fourth_person_l553_553518


namespace lollipop_cases_l553_553086

theorem lollipop_cases (total_cases : ℕ) (chocolate_cases : ℕ) (lollipop_cases : ℕ) 
  (h1 : total_cases = 80) (h2 : chocolate_cases = 25) : lollipop_cases = 55 :=
by
  sorry

end lollipop_cases_l553_553086


namespace pineapple_cost_l553_553578

variables (P W : ℕ)

theorem pineapple_cost (h1 : 2 * P + 5 * W = 38) : P = 14 :=
sorry

end pineapple_cost_l553_553578


namespace find_P_and_Q_l553_553731

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l553_553731


namespace find_m_l553_553275

-- Define the vectors a and b
def vector_a : ℝ × ℝ × ℝ := (-2, 1, 5)
def vector_b (m : ℝ) : ℝ × ℝ × ℝ := (6, m, -15)

-- Define the scalar multiple condition
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b = (t * a.1, t * a.2, t * a.3)

-- State the problem
theorem find_m (m : ℝ) : 
  parallel vector_a (vector_b m) → m = -3 := 
by
  sorry

end find_m_l553_553275


namespace num_integers_between_sqrt_range_l553_553851

theorem num_integers_between_sqrt_range :
  {x : ℕ | 5 > Real.sqrt x ∧ Real.sqrt x > 3}.card = 15 :=
by sorry

end num_integers_between_sqrt_range_l553_553851


namespace problem_statement_l553_553351

variables {A B C D : Type}
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D]
variables {CA CB AB m AD DB : ℝ}

-- Conditions
def right_angled_triangle (A B C : Type) : Prop :=
  ∃ (a b c : ℝ), (a = CA) ∧ (b = CB) ∧ (c = AB) ∧ (a^2 + b^2 = c^2)

def altitude_from_C (m : ℝ) (A C B : Type) : Prop :=
  ∃ (a b c : ℝ), (a = CA) ∧ (b = CB) ∧ (c = AB) ∧ (m = (a * b) / c)

def angle_bisector_D (AD DB : ℝ) (CA CB AB : ℝ) : Prop :=
  AD = (CA * AB) / (CA + CB) ∧ DB = (CB * AB) / (CA + CB)

def CA_less_CB (CA CB : ℝ) : Prop :=
  CA < CB

-- Theorem to prove
theorem problem_statement (A B C D : Type) (CA CB AB m AD DB : ℝ) :
  right_angled_triangle A B C →
  altitude_from_C m A C B →
  angle_bisector_D AD DB CA CB AB →
  CA_less_CB CA CB →
  AD < m ∧ m < DB :=
sorry

end problem_statement_l553_553351


namespace triangle_area_example_l553_553150

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * | x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) |

theorem triangle_area_example : area_of_triangle (-2) (-3) 4 (-3) 28 7 = 30 :=
by
  sorry

end triangle_area_example_l553_553150


namespace uber_cost_22_l553_553127

variable (T : ℝ) (Lux : ℝ) (Urb : ℝ) (TipRate : ℝ := 0.20)
variable (TotalTaxiCost : ℝ := 18)
variable (LyftPriceDifference : ℝ := 4)
variable (UberPriceDifference : ℝ := 3)

def taxi_cost (T : ℝ) : Prop :=
  T + TipRate * T = TotalTaxiCost

def lyft_cost (T : ℝ) (Lux : ℝ) : Prop :=
  Lux = T + LyftPriceDifference

def uber_cost (Lux : ℝ) (Urb : ℝ) : Prop :=
  Urb = Lux + UberPriceDifference

theorem uber_cost_22 (T : ℝ) (Lux : ℝ) (Urb : ℝ)
  (h_taxi : taxi_cost T) (h_lyft : lyft_cost T Lux) (h_uber : uber_cost Lux Urb) :
  Urb = 22 :=
by
  sorry

end uber_cost_22_l553_553127


namespace jill_sales_goal_l553_553770

def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def boxes_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer
def boxes_left : ℕ := 75
def sales_goal : ℕ := boxes_sold + boxes_left

theorem jill_sales_goal : sales_goal = 150 := by
  sorry

end jill_sales_goal_l553_553770


namespace sum_first_9_terms_arithmetic_sequence_l553_553267

theorem sum_first_9_terms_arithmetic_sequence :
  ∀ (a : ℕ → ℤ), 
  (∃ l_1, ∀ n ∈ (set.Ioi 0 : set ℕ), (n, a n) ∈ l_1) → 
  ((5, a 5) = (5, 3)) →
  ∑ i in range 1 10, a i = 27 :=
by 
  intro a h_line h_five_three
  sorry

end sum_first_9_terms_arithmetic_sequence_l553_553267


namespace solution_set_inequality_l553_553301

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 3) - abs (x + 1)

theorem solution_set (x : ℝ) : f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 10 :=
by sorry

def M : set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2 / 3}

theorem inequality (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : a^2 + b^2 + 2 * a - 2 * b < 5 :=
by sorry

end solution_set_inequality_l553_553301


namespace find_P_and_Q_l553_553730

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l553_553730


namespace problem_divides_area_l553_553484

noncomputable def vertical_line_divides_area (p q : ℝ) : Prop :=
  let f := λ x : ℝ, -2 * x ^ 2
  let g := λ x : ℝ, x ^ 2 + p * x + q
  let h := λ x : ℝ, f x - g x
  let x1 := (p - Real.sqrt (p ^ 2 - 12 * q)) / -6
  let x2 := (p + Real.sqrt (p ^ 2 - 12 * q)) / -6
  let x0 := -(p / 6)
  x0 = (x1 + x2) / 2

theorem problem_divides_area (p q : ℝ) :
  vertical_line_divides_area p q :=
sorry

end problem_divides_area_l553_553484


namespace proof_ratio_QP_over_EF_l553_553437

noncomputable def rectangle_theorem : Prop :=
  ∃ (A B C D E F G P Q : ℝ × ℝ),
    -- Coordinates of the rectangle vertices
    A = (0, 4) ∧ B = (5, 4) ∧ C = (5, 0) ∧ D = (0, 0) ∧
    -- Coordinates of points E, F, and G on the sides of the rectangle
    E = (4, 4) ∧ F = (2, 0) ∧ G = (5, 1) ∧
    -- Coordinates of intersection points P and Q
    P = (20 / 7, 12 / 7) ∧ Q = (40 / 13, 28 / 13) ∧
    -- Ratio of distances PQ and EF
    (dist P Q)/(dist E F) = 10 / 91

theorem proof_ratio_QP_over_EF : rectangle_theorem :=
sorry

end proof_ratio_QP_over_EF_l553_553437


namespace denote_west_travel_l553_553338

theorem denote_west_travel (east_pos : 2 = +2) : (-3) = -3 :=
by
  sorry

end denote_west_travel_l553_553338


namespace n_gon_path_segments_l553_553269

theorem n_gon_path_segments (n : ℕ) (h : n ≥ 3) : 
  ∃ k : ℕ, (∀ (P Q : (fix n : Type) (pt : pt ∈ n_gon ∨ pt On its boundary)),
    ∃ (path : path under the restriction mentioned in question), path is in k segments) ∧ k = n - 1 :=
by
  intros
  sorry

end n_gon_path_segments_l553_553269


namespace find_a7_l553_553362

def arithmetic_seq (a₁ d : ℤ) (n : ℤ) : ℤ := a₁ + (n-1) * d

theorem find_a7 (a₁ d : ℤ)
  (h₁ : arithmetic_seq a₁ d 3 + arithmetic_seq a₁ d 7 - arithmetic_seq a₁ d 10 = -1)
  (h₂ : arithmetic_seq a₁ d 11 - arithmetic_seq a₁ d 4 = 21) :
  arithmetic_seq a₁ d 7 = 20 :=
by
  sorry

end find_a7_l553_553362


namespace intersection_M_N_l553_553792

open Set

noncomputable def M : Set ℕ := {x | x < 6}
noncomputable def N : Set ℕ := {x | x^2 - 11 * x + 18 < 0}

theorem intersection_M_N : M ∩ N = {3, 4, 5} := by
  sorry

end intersection_M_N_l553_553792


namespace area_of_region_inside_D_outside_circles_l553_553987

-- Definitions of condition in Lean
noncomputable def radius_D : ℝ := 40
noncomputable def radius_circle (s : ℝ) (tangent_to_D : s + s * Real.sqrt 2 = radius_D) : ℝ := s
noncomputable def M (s : ℝ) (tangent_to_D : s + s * Real.sqrt 2 = radius_D) : ℝ :=
  π * (radius_D ^ 2 - 8 * (s ^ 2))

-- The formulated proof problem statement
theorem area_of_region_inside_D_outside_circles :
  (∀ s : ℝ, s + s * Real.sqrt 2 = radius_D → ⌊M s (by assumption)⌋ = -26175) :=
by
  sorry

end area_of_region_inside_D_outside_circles_l553_553987


namespace dealer_cash_discount_percentage_l553_553168

-- Definitions of the given conditions
variable (C : ℝ) (n m : ℕ) (profit_p list_ratio : ℝ)
variable (h_n : n = 25) (h_m : m = 20) (h_profit : profit_p = 1.36) (h_list_ratio : list_ratio = 2)

-- The statement we need to prove
theorem dealer_cash_discount_percentage 
  (h_eff_selling_price : (m : ℝ) / n * C = profit_p * C)
  : ((list_ratio * C - (m / n * C)) / (list_ratio * C) * 100 = 60) :=
by
  sorry

end dealer_cash_discount_percentage_l553_553168


namespace solve_for_x_l553_553461

noncomputable def x_solution (x : ℚ) : Prop :=
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0

theorem solve_for_x :
  ∃ x : ℚ, x_solution x ∧ x = 4 / 3 :=
by
  sorry

end solve_for_x_l553_553461


namespace income_to_expenditure_ratio_l553_553096

theorem income_to_expenditure_ratio (I E S : ℕ) (hI : I = 15000) (hS : S = 7000) (hSavings : S = I - E) :
  I / E = 15 / 8 := by
  -- Lean proof goes here
  sorry

end income_to_expenditure_ratio_l553_553096


namespace number_of_solutions_eq_six_l553_553719

theorem number_of_solutions_eq_six :
  {p : ℤ × ℤ | p.1 + p.2 = p.1 * p.2 - 2}.to_finset.card = 6 :=
sorry

end number_of_solutions_eq_six_l553_553719


namespace identical_digits_has_37_factor_l553_553057

theorem identical_digits_has_37_factor (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 37 ∣ (100 * a + 10 * a + a) :=
by
  sorry

end identical_digits_has_37_factor_l553_553057


namespace remainder_when_sum_div_6_l553_553876

-- Define the series summation modulo 6.
def sum_powers_5_mod_6 : ℕ := (List.range 1510).sum (λ n, 5^n % 6) % 6

-- Theorem to prove that the remainder of the series when divided by 6 is 5.
theorem remainder_when_sum_div_6 : sum_powers_5_mod_6 = 5 :=
by
  sorry

end remainder_when_sum_div_6_l553_553876


namespace smallest_number_of_coins_l553_553924

theorem smallest_number_of_coins (p n d q : ℕ) (total : ℕ) :
  (total < 100) →
  (total = p * 1 + n * 5 + d * 10 + q * 25) →
  (∀ k < 100, ∃ (p n d q : ℕ), k = p * 1 + n * 5 + d * 10 + q * 25) →
  p + n + d + q = 10 :=
sorry

end smallest_number_of_coins_l553_553924


namespace propositions_count_is_three_l553_553697

def proposition1 := ∀ r : ℝ, (|r| is the absolute value of the coefficient of linear correlation)
def proposition2 := ∀ (x y : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ), l = (fun x => b * x + a) → P = (⟨mean x, mean y⟩)
def proposition3 := ∀ (sampling_interval : ℝ), sampling_interval = 10 → (is_stratified_sampling = False ∧ is_systematic_sampling = True)
def proposition4 := ∀ (residuals : list ℝ), (smaller_sum_of_squares residuals → better_fit_of_model residuals)
def proposition5 := ∀ (x y : ℝ), y = 0.1 * x + 10 → (increase_in_x := 1) → (increase_in_y := 0.1)

def true_propositions_count : ℕ :=
  [proposition1, proposition2, proposition3, proposition4, proposition5].count true

theorem propositions_count_is_three : true_propositions_count = 3 :=
by sorry

end propositions_count_is_three_l553_553697


namespace average_salary_l553_553820

theorem average_salary
  (num_technicians : ℕ) (avg_salary_technicians : ℝ)
  (num_other_workers : ℕ) (avg_salary_other_workers : ℝ)
  (total_num_workers : ℕ) (avg_salary_all_workers : ℝ) :
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  num_other_workers = total_num_workers - num_technicians →
  avg_salary_other_workers = 6000 →
  total_num_workers = 28 →
  avg_salary_all_workers = (num_technicians * avg_salary_technicians + num_other_workers * avg_salary_other_workers) / total_num_workers →
  avg_salary_all_workers = 8000 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l553_553820


namespace steven_needs_more_seeds_l553_553080

def apple_seeds : Nat := 6
def pear_seeds : Nat := 2
def grape_seeds : Nat := 3
def apples_set_aside : Nat := 4
def pears_set_aside : Nat := 3
def grapes_set_aside : Nat := 9
def seeds_required : Nat := 60

theorem steven_needs_more_seeds : 
  seeds_required - (apples_set_aside * apple_seeds + pears_set_aside * pear_seeds + grapes_set_aside * grape_seeds) = 3 := by
  sorry

end steven_needs_more_seeds_l553_553080


namespace min_workers_needed_for_profit_l553_553926

noncomputable def profit_condition (n : ℕ) : Prop :=
  let daily_cost := 600 + 9 * 20 * n in
  let daily_revenue := 9 * 4 * 6 * n in
  daily_revenue > daily_cost

theorem min_workers_needed_for_profit : ∃ n : ℕ, n ≥ 17 ∧ profit_condition n :=
by
  -- Proof steps would go here, but we use sorry to skip it.
  sorry

end min_workers_needed_for_profit_l553_553926


namespace jamal_bought_4_half_dozens_l553_553768

/-- Given that each crayon costs $2, the total cost is $48, and a half dozen is 6 crayons,
    prove that Jamal bought 4 half dozens of crayons. -/
theorem jamal_bought_4_half_dozens (cost_per_crayon : ℕ) (total_cost : ℕ) (half_dozen : ℕ) 
  (h1 : cost_per_crayon = 2) (h2 : total_cost = 48) (h3 : half_dozen = 6) : 
  (total_cost / cost_per_crayon) / half_dozen = 4 := 
by 
  sorry

end jamal_bought_4_half_dozens_l553_553768


namespace Poincare_inequality_characterizes_normal_l553_553810

section PoincareInequality

-- Definitions of the conditions
variables {Ω : Type*} [ProbabilitySpace Ω]

/-- Random variable X with expectation 0 and variance 1 -/
variable {X : Ω → ℝ}
variable (hX_mean : E[X] = 0) (hX_var : E[X^2] = 1)

/-- Inequality condition for all smooth functions f -/
variable (ineq : ∀ (f : ℝ → ℝ), SmoothFunction f → E[(f' X)^2] < ∞ → var f(X) ≤ E[f'(X)]^2)

-- The theorem to prove
theorem Poincare_inequality_characterizes_normal (X : Ω → ℝ) (hX_mean : E[X] = 0)
    (hX_var : E[X^2] = 1)
    (ineq : ∀ (f : ℝ → ℝ), SmoothFunction f → E[(f' X)^2] < ∞ → var f(X) ≤ E[f'(X)]^2) :
    isNormal 0 1 X :=
sorry

end PoincareInequality

end Poincare_inequality_characterizes_normal_l553_553810


namespace range_of_g_l553_553642

noncomputable def g (t : ℝ) : ℝ :=
  (2 * t^2 - 5/2 * t + 1) / (t^2 + 3 * t + 2)

theorem range_of_g :
  set.range g = set.Icc (-19.41) (-15.59) :=
sorry

end range_of_g_l553_553642


namespace annual_fixed_costs_l553_553948

theorem annual_fixed_costs
  (profit : ℝ := 30500000)
  (selling_price : ℝ := 9035)
  (variable_cost : ℝ := 5000)
  (units_sold : ℕ := 20000) :
  ∃ (fixed_costs : ℝ), profit = (selling_price * units_sold) - (variable_cost * units_sold) - fixed_costs :=
sorry

end annual_fixed_costs_l553_553948


namespace inconsistent_coordinates_l553_553759

theorem inconsistent_coordinates
  (m n : ℝ) 
  (h1 : m - (5/2)*n + 1 = 0) 
  (h2 : (m + 1/2) - (5/2)*(n + 1) + 1 = 0) :
  false :=
by
  sorry

end inconsistent_coordinates_l553_553759


namespace compound_interest_amount_l553_553342

theorem compound_interest_amount (P r t SI : ℝ) (h1 : t = 3) (h2 : r = 0.10) (h3 : SI = 900) :
  SI = P * r * t → P = 900 / (0.10 * 3) → (P * (1 + r)^t - P = 993) :=
by
  intros hSI hP
  sorry

end compound_interest_amount_l553_553342


namespace largest_product_l553_553526

theorem largest_product :
  let S := {-20, -4, -1, 3, 5, 9}
  in ∃ a b ∈ S, (∀ c d ∈ S, c * d ≤ a * b) ∧ a * b = 80 := 
by
  let S := {-20, -4, -1, 3, 5, 9}
  use [-20, -4]
  split
  { use _, -- Side condition
    intro,
    fin_cases,
    all_goals{
        fin_cases,
        all_goals{
            norm_num,
            norm_cast,
            omega }},
    },
  try rfl, -- Optional side condition
  sorry

end largest_product_l553_553526


namespace problem_statement_l553_553901

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1/a) + Real.sqrt (b + 1/b) + Real.sqrt (c + 1/c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
sorry

end problem_statement_l553_553901


namespace f_1988_eq_1988_l553_553395

noncomputable def f (n : ℕ) : ℕ := sorry

axiom f_f_eq_add (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem f_1988_eq_1988 : f 1988 = 1988 := 
by
  sorry

end f_1988_eq_1988_l553_553395


namespace max_cake_pieces_l553_553135

theorem max_cake_pieces (cake_length cake_width piece_size : ℕ) (h_cake_length : cake_length = 20) (h_cake_width : cake_width = 24) (h_piece_size : piece_size = 4) : 
  let pieces_vertical := (cake_length / piece_size) * (cake_width / piece_size) 
  in pieces_vertical = 30 :=
by 
  have h1 : cake_length / piece_size = 5 := by { rw [h_cake_length, h_piece_size], exact Nat.div_eq_of_eq_mul_left (by decide) rfl }
  have h2 : cake_width / piece_size = 6 := by { rw [h_cake_width, h_piece_size], exact Nat.div_eq_of_eq_mul_left (by decide) rfl }
  rw [←Nat.mul_eq_pieces, h1, h2]
  exact rfl

end max_cake_pieces_l553_553135


namespace Tyson_scored_75_points_l553_553867

theorem Tyson_scored_75_points :
  let three_pointers := 15
  let two_pointers := 12
  let one_pointers := 6
  (3 * three_pointers + 2 * two_pointers + 1 * one_pointers) = 75 :=
by
  let three_pointers := 15
  let two_pointers := 12
  let one_pointers := 6
  have h1 : 3 * three_pointers = 45 := by sorry
  have h2 : 2 * two_pointers = 24 := by sorry
  have h3 : 1 * one_pointers = 6 := by sorry
  show 3 * three_pointers + 2 * two_pointers + 1 * one_pointers = 75 from sorry

end Tyson_scored_75_points_l553_553867


namespace complex_number_problems_l553_553143

variable (z1 z2 : Complex)

theorem complex_number_problems :
  (|z1 - z2| = 0 ∧ z1 = conj z2 ∧ |z1| = |z2|) →
  (conj z1 = conj z2) ∧ (conj z1 = z2) ∧ (z1 * conj z1 = z2 * conj z2) :=
by
  assume h
  have h1 : |z1 - z2| = 0 := h.1
  have h2 : z1 = conj z2 := h.2.1
  have h3 : |z1| = |z2| := h.2.2
  sorry -- Proof omitted

end complex_number_problems_l553_553143


namespace expressions_equal_l553_553984

theorem expressions_equal (a b c : ℝ) : 2 * a + 3 * b * c = (a + b) * (2 * a + c) ↔ a + b + c = 2 :=
by
  dsimp
  intros
  sorry

end expressions_equal_l553_553984


namespace evaluate_b3_l553_553632

variable (b1 q : ℤ)
variable (b1_cond : b1 = 5 ∨ b1 = -5)
variable (q_cond : q = 3 ∨ q = -3)
def b3 : ℤ := b1 * q^2

theorem evaluate_b3 (h : b1^2 * (1 + q^2 + q^4) = 2275) : b3 = 45 ∨ b3 = -45 :=
by sorry

end evaluate_b3_l553_553632


namespace angle_CED_is_60_degrees_l553_553382

theorem angle_CED_is_60_degrees
  (O A B E F C D : Type)
  [circle O A B]
  (not_diametrically_opposite_EA : ¬ (E = diametrically_opposite_to A))
  (not_diametrically_opposite_EB : ¬ (E = diametrically_opposite_to B))
  (not_diametrically_opposite_FA : ¬ (F = diametrically_opposite_to A))
  (not_diametrically_opposite_FB : ¬ (F = diametrically_opposite_to B))
  (distinct_EF : E ≠ F)
  (tangent_point_C : tangent B ∩ tangent E = C)
  (tangent_point_D : tangent F ∩ line_from A E = D)
  (angle_BAE : angle B A E = 30)
  (angle_BAF : angle B A F = 60) 
  : angle C E D = 60 :=
by
  sorry

end angle_CED_is_60_degrees_l553_553382


namespace Horner_Polynomial_Value_l553_553868

-- Define the conditions as hypotheses
theorem Horner_Polynomial_Value :
  let v₀ := 7 in
  let v₁ := v₀ * 2 + 5 in
  let v₂ := v₁ * 2 + 3 in
  let v₃ := v₂ * 2 + 1 in
  v₃ = 83 :=
by
  let v₀ := 7
  let v₁ := v₀ * 2 + 5
  let v₂ := v₁ * 2 + 3
  let v₃ := v₂ * 2 + 1
  have h₀ : v₀ = 7 := rfl
  have h₁ : v₁ = v₀ * 2 + 5 := rfl
  have h₂ : v₂ = v₁ * 2 + 3 := rfl
  have h₃ : v₃ = v₂ * 2 + 1 := rfl
  show v₃ = 83
  simp [h₀, h₁, h₂, h₃]
  sorry

end Horner_Polynomial_Value_l553_553868


namespace remainder_x1000_l553_553643

open Polynomial

variables {R : Type*} [CommRing R]

theorem remainder_x1000 :
  let f := (X^1000 : R[X])
  let g := (X^2 + 1) * (X + 1)
  ∃ q r : R[X], f = q * g + r ∧ r.degree < g.degree :=
begin
  let f := (X^1000 : R[X]),
  let g := (X^2 + 1) * (X + 1),
  use (0 : R[X]),
  use (1 : R[X]),
  sorry,
end

end remainder_x1000_l553_553643


namespace evaluate_expression_l553_553904

theorem evaluate_expression :
  (2 * 10^3)^3 = 8 * 10^9 :=
by
  sorry

end evaluate_expression_l553_553904


namespace number_of_neither_l553_553200

def total_businessmen : ℕ := 30
def coffee_drinkers : ℕ := 15
def tea_drinkers : ℕ := 12
def both_drinkers : ℕ := 6

theorem number_of_neither (total_businessmen coffee_drinkers tea_drinkers both_drinkers : ℕ) : 
  coffee_drinkers = 15 ∧ 
  tea_drinkers = 12 ∧ 
  both_drinkers = 6 ∧ 
  total_businessmen = 30 → 
  total_businessmen - (coffee_drinkers + tea_drinkers - both_drinkers) = 9 :=
by
  sorry

end number_of_neither_l553_553200


namespace exists_epsilon_l553_553034

theorem exists_epsilon {n : ℕ} (hn : n ≥ 2) (a : Fin n → ℝ) :
  ∃ ε : Fin n → ℝ, (∀ i, ε i = -1 ∨ ε i = 1) ∧ 
  ((∑ i, a i)^2 + (∑ i, ε i * a i)^2 ≤ (n + 1) * ∑ i, (a i)^2) :=
by
  sorry

end exists_epsilon_l553_553034


namespace tank_capacity_l553_553567

theorem tank_capacity
  (w c : ℝ)
  (h1 : w / c = 1 / 3)
  (h2 : (w + 5) / c = 2 / 5) :
  c = 75 :=
by
  sorry

end tank_capacity_l553_553567


namespace find_number_of_male_students_l553_553056

/- Conditions: 
 1. n ≡ 2 [MOD 4]
 2. n ≡ 1 [MOD 5]
 3. n > 15
 4. There are 15 female students
 5. There are more female students than male students
-/
theorem find_number_of_male_students (n : ℕ) (females : ℕ) (h1 : n % 4 = 2) (h2 : n % 5 = 1) (h3 : n > 15) (h4 : females = 15) (h5 : females > n - females) : (n - females) = 11 :=
by
  sorry

end find_number_of_male_students_l553_553056


namespace binom_1300_2_eq_844350_l553_553612

theorem binom_1300_2_eq_844350 : (Nat.choose 1300 2) = 844350 := by
  sorry

end binom_1300_2_eq_844350_l553_553612


namespace sum_of_midpoint_xcoords_l553_553494

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l553_553494


namespace time_to_cross_l553_553131

def first_train_length : ℝ := 140 -- meters
def second_train_length : ℝ := 200 -- meters
def first_train_speed : ℝ := 60 * (5 / 18) -- converting km/hr to m/s
def second_train_speed : ℝ := 40 * (5 / 18) -- converting km/hr to m/s
def relative_speed : ℝ := first_train_speed + second_train_speed
def total_distance : ℝ := first_train_length + second_train_length

theorem time_to_cross : (total_distance / relative_speed) ≈ 12.23 :=
by
  -- Approximation near 12.23 seconds
  sorry

end time_to_cross_l553_553131


namespace calc_det_of_cot_tri_l553_553778

variables {A B C : ℝ}

theorem calc_det_of_cot_tri
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hSum : A + B + C = π)
  : det ![
      ![Real.cot A, 1, 1],
      ![1, Real.cot B, 1],
      ![1, 1, Real.cot C]
    ] = 2 :=
  sorry

end calc_det_of_cot_tri_l553_553778


namespace capacity_of_other_bottle_l553_553921

theorem capacity_of_other_bottle (x : ℝ) :
  (16 / 3) * (x / 8) + (16 / 3) = 8 → x = 4 := by
  -- the proof will go here
  sorry

end capacity_of_other_bottle_l553_553921


namespace sum_x_midpoints_of_triangle_l553_553505

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l553_553505


namespace no_distinct_numbers_grid_l553_553369

theorem no_distinct_numbers_grid : ¬ ∃ f : Fin 6 → Fin 6 → ℕ,
  (∀ i j, f i j ≠ f i (j+1) ∧ f i j ≠ f (i+1) j) ∧
  (∀ i, ∃ sum1 sum2 : ℕ, sum1 ∈ {2022, 2023} ∧ sum2 ∈ {2022, 2023} ∧
    (finset.range 5).sum (λ k, f i k) = sum1 ∧ 
    (finset.range 6).sum (λ k, f k i) = sum2) → false :=
by {
  sorry
}

end no_distinct_numbers_grid_l553_553369


namespace smallest_n_for_T_integer_l553_553017

noncomputable def J := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def T (n : ℕ) : ℚ := ∑ x in finset.range (5^n + 1), 
  ∑ d in {
    digit | digit.to_nat ≠ 0 ∧ digit.to_nat < 10 
  }, (1 : ℚ) / (digit.to_nat : ℚ)

theorem smallest_n_for_T_integer : ∃ n : ℕ, T n ∈ ℤ ∧ ∀ m : ℕ, T m ∈ ℤ → 63 ≤ n :=
by {
  sorry
}

end smallest_n_for_T_integer_l553_553017


namespace rectangle_integral_side_l553_553106

theorem rectangle_integral_side (R : Set (ℝ × ℝ)) (R_i : ℕ → Set (ℝ × ℝ))
  (h_cover : ∀ x ∈ R, ∃ i, x ∈ R_i i)
  (h_subset : ∀ i, R_i i ⊆ R)
  (h_parallel : ∀ i, is_parallel (R_i i) (R))
  (h_disjoint : ∀ i j, i ≠ j → disjoint (interior (R_i i)) (interior (R_i j)))
  (h_integral_side : ∀ i, ∃ a b c d : ℤ, 
      ((a, b), (c, d)) ∈ vertices (R_i i) ∨ ((a, b), (c, d)) ∈ vertices (R_i i))
  : ∃ a b c d : ℤ, ((a, b), (c, d)) ∈ vertices (R) ∨ ((a, b), (c, d)) ∈ vertices (R) :=
sorry

end rectangle_integral_side_l553_553106


namespace find_m_l553_553292

-- Given: the ellipse equation with major axis along y-axis and focal length 4
def ellipse_equation (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (10 - m)) + (y^2 / (m - 2)) = 1

def major_axis_along_y (m : ℝ) : Prop := true  -- tautology to mention axis orientation

def focal_length (m : ℝ) : Prop :=
  let a := real.sqrt (m - 2) in
  let b := real.sqrt (10 - m) in
  let c := real.sqrt (a^2 - b^2) in
  c = 2

-- Prove that given these conditions, m = 8
theorem find_m (m : ℝ) (h1 : ellipse_equation m) (h2 : major_axis_along_y m) (h3 : focal_length m) : 
  m = 8 := 
sorry

end find_m_l553_553292


namespace total_surface_area_of_modified_cube_l553_553962

-- Define the side length of the original cube
def side_length_cube := 3

-- Define the side length of the holes
def side_length_hole := 1

-- Define the condition of the surface area calculation
def total_surface_area_including_internal (side_length_cube side_length_hole : ℕ) : ℕ :=
  let original_surface_area := 6 * (side_length_cube * side_length_cube)
  let reduction_area := 6 * (side_length_hole * side_length_hole)
  let remaining_surface_area := original_surface_area - reduction_area
  let interior_surface_area := 6 * (4 * side_length_hole * side_length_cube)
  remaining_surface_area + interior_surface_area

-- Statement for the proof
theorem total_surface_area_of_modified_cube : total_surface_area_including_internal 3 1 = 72 :=
by
  -- This is the statement; the proof is omitted as "sorry"
  sorry

end total_surface_area_of_modified_cube_l553_553962


namespace even_digits_count_1998_l553_553528

-- Define the function for counting the total number of digits used in the first n positive even integers
def totalDigitsEvenIntegers (n : ℕ) : ℕ :=
  let totalSingleDigit := 4 -- 2, 4, 6, 8
  let numDoubleDigit := 45 -- 10 to 98
  let digitsDoubleDigit := numDoubleDigit * 2
  let numTripleDigit := 450 -- 100 to 998
  let digitsTripleDigit := numTripleDigit * 3
  let numFourDigit := 1499 -- 1000 to 3996
  let digitsFourDigit := numFourDigit * 4
  totalSingleDigit + digitsDoubleDigit + digitsTripleDigit + digitsFourDigit

-- Theorem: The total number of digits used when the first 1998 positive even integers are written is 7440.
theorem even_digits_count_1998 : totalDigitsEvenIntegers 1998 = 7440 :=
  sorry

end even_digits_count_1998_l553_553528


namespace stephanie_quarters_fraction_l553_553817

/-- Stephanie has a collection containing exactly one of the first 25 U.S. state quarters. 
    The quarters are in the order the states joined the union.
    Suppose 8 states joined the union between 1800 and 1809. -/
theorem stephanie_quarters_fraction :
  (8 / 25 : ℚ) = (8 / 25) :=
by
  sorry

end stephanie_quarters_fraction_l553_553817


namespace simplify_sqrt_fraction_l553_553072

theorem simplify_sqrt_fraction :
  sqrt (9 / 2) = 3 * sqrt 2 / 2 :=
by 
  sorry

end simplify_sqrt_fraction_l553_553072


namespace tangent_line_at_point_l553_553476

noncomputable def f (x : ℝ) := x^3 - x

theorem tangent_line_at_point :
  let p : ℝ × ℝ := (1, 0)
  in ∃ (a b : ℝ), (∀ x y, y = a * x + b ↔ y = 2 * x - 2) ∧ (f (p.1) = p.2) :=
by
  sorry

end tangent_line_at_point_l553_553476


namespace flower_bed_distance_approximately_10_42_l553_553870

def flower_bed_distance (total_length : ℝ) (num_beds : ℕ) : ℝ :=
  total_length / (num_beds - 1)

theorem flower_bed_distance_approximately_10_42 :
  flower_bed_distance 239.66 24 ≈ 10.42 :=
by
  sorry

end flower_bed_distance_approximately_10_42_l553_553870


namespace sum_40_l553_553675

-- Define the sequence and the sum of the first n terms
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Define the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  a (n + 1) + (-1)^n * a n = n

-- Define the sum of the first n terms
def sum_of_terms (n : ℕ) : Prop :=
  S n = ∑ i in range (n + 1), a i

-- The theorem we want to prove: S_40 = 440
theorem sum_40 : (∀ n, recurrence_relation a n) → sum_of_terms a S 40 → S 40 = 440 :=
sorry

end sum_40_l553_553675


namespace trains_cross_each_other_in_5_76_seconds_l553_553155

noncomputable def trains_crossing_time (l1 l2 v1_kmh v2_kmh : ℕ) : ℚ :=
  let v1 := (v1_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let v2 := (v2_kmh : ℚ) * 5 / 18  -- convert speed from km/h to m/s
  let total_distance := (l1 : ℚ) + (l2 : ℚ)
  let relative_velocity := v1 + v2
  total_distance / relative_velocity

theorem trains_cross_each_other_in_5_76_seconds :
  trains_crossing_time 100 60 60 40 = 160 / 27.78 := by
  sorry

end trains_cross_each_other_in_5_76_seconds_l553_553155


namespace num_ways_to_place_2006_balls_l553_553514

noncomputable def numWaysToPlaceBallsIntoBoxes : ℕ :=
  let n := 2006
  nat.factorial n - 
  (derangements n + 
  (2006 * derangements (n - 1)) + 
  (nat.choose n 2 * derangements (n - 2)) + 
  (nat.choose n 3 * derangements (n - 3)) + 
  (nat.choose n 4 * derangements (n - 4)))

theorem num_ways_to_place_2006_balls : numWaysToPlaceBallsIntoBoxes = 2006! - D_{2006} - 2006 D_{2005} - nat.choose 2006 2 * D_{2004} - nat.choose 2006 3 * D_{2003} - nat.choose 2006 4 * D_{2002} :=
sorry

end num_ways_to_place_2006_balls_l553_553514


namespace smallest_positive_integer_n_l553_553211

theorem smallest_positive_integer_n 
  : ∃ (n : ℕ), (∀ m : ℕ, m < n → Logarithm.SumExpression m < 1 + log₁₀ (511 / 512)) 
               ∧ Logarithm.SumExpression n ≥ 1 + log₁₀ (511 / 512) 
  ∧ n = 3 :=
by
  sorry

namespace Logarithm

noncomputable def SumExpression (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n+1), log₁₀ (1 + (1 / 2^(2^k)))

end Logarithm

end smallest_positive_integer_n_l553_553211


namespace minimum_value_inequality_l553_553331

theorem minimum_value_inequality {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x * y * z * (x + y + z) = 1) : (x + y) * (y + z) ≥ 2 := 
sorry

end minimum_value_inequality_l553_553331


namespace space_experiment_arrangements_l553_553352

/-- 
Let there be six procedures to implement in sequence.
Procedure A can only appear in the first or last step.
Procedures B and C must be adjacent.
Prove that the total number of possible arrangements is 96.
-/
theorem space_experiment_arrangements :
  let procedures := ["A", "B", "C", "D", "E", "F"]
  let valid_arrangements := 
    (λ x : List String, x.head = "A" ∨ x.last = "A") ∧ 
    (λ x : List String, ∃ i, x[i] = "B" ∧ x[i+1] = "C" ∨ x[i] = "C" ∧ x[i+1] = "B")
  List.filter valid_arrangements (List.permutations procedures)).length = 96 := 
  sorry

end space_experiment_arrangements_l553_553352


namespace baseball_team_games_l553_553559

theorem baseball_team_games (P Q : ℕ) (hP : P > 3 * Q) (hQ : Q > 3) (hTotal : 2 * P + 6 * Q = 78) :
  2 * P = 54 :=
by
  -- placeholder for the actual proof
  sorry

end baseball_team_games_l553_553559


namespace backyard_area_l553_553044

constant total_distance : ℝ
constant laps : ℕ
constant is_square : Prop

axiom h1 : total_distance = 2000
axiom h2 : laps = 8
axiom h3 : is_square = true

theorem backyard_area :
  is_square → total_distance / laps / 4 ^ 2  = 3906.25 :=
by
  intros
  sorry

end backyard_area_l553_553044


namespace eccentricity_is_half_find_ellipse_equation_l553_553789

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h : a > b > 0) (k : 8 / 5 = (AP_length / PQ_length)) : ℝ :=
  let c := Real.sqrt (a^2 - b^2) in
  (c / a)

theorem eccentricity_is_half 
  (a b : ℝ) (h : a > b > 0)
  (AP_length PQ_length : ℝ)
  (k : 8 / 5 = (AP_length / PQ_length))
  : eccentricity_of_ellipse a b h k = 1 / 2 := sorry

noncomputable def equation_of_ellipse 
  (a Mx My : ℝ)
  (h1 : a > 0)
  (angle : ℝ)
  (h2 : angle = π / 6)
  (h3 : Mx = -3)
  (h4 : My = 0)
  : String := 
  "x^2 / 4 + y^2 / 3 = 1"

theorem find_ellipse_equation
  (a b : ℝ)
  (M : ℝ × ℝ)
  (h1 : a = 2)
  (h2 : b = Real.sqrt 3)
  (h3 : M = (-3, 0))
  : equation_of_ellipse a (M.1) (M.2) (by apply a > 0) (π / 6) (by assumption) (by assumption) = "x^2 / 4 + y^2 / 3 = 1" := sorry

end eccentricity_is_half_find_ellipse_equation_l553_553789


namespace mike_seed_problem_l553_553419

theorem mike_seed_problem :
  ∃ x : ℕ, (x + 2 * x + 30 + 30 = 120) ∧ (x = 20) :=
begin
  sorry,
end

end mike_seed_problem_l553_553419


namespace necessary_not_sufficient_condition_l553_553261

theorem necessary_not_sufficient_condition {a : ℝ} :
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) →
  (¬ (∀ x : ℝ, x ≥ a → |x - 1| < 1)) →
  a ≤ 0 :=
by
  intro h1 h2
  sorry

end necessary_not_sufficient_condition_l553_553261


namespace distance_run_l553_553920

theorem distance_run (D : ℝ) (A_time : ℝ) (B_time : ℝ) (A_beats_B : ℝ) : 
  A_time = 90 ∧ B_time = 180 ∧ A_beats_B = 2250 → D = 2250 :=
by
  sorry

end distance_run_l553_553920


namespace factorization_result_l553_553635

theorem factorization_result :
  (∃ (C D : ℤ), (6 * (y : ℤ)^2 - 31 * y + 35 = (C * y - 5) * (D * y - 7)) ∧ (CD + C = 9)) :=
begin
  sorry
end

end factorization_result_l553_553635


namespace find_second_number_l553_553562

theorem find_second_number
  (first_number : ℕ)
  (second_number : ℕ)
  (h1 : first_number = 45)
  (h2 : first_number / second_number = 5) : second_number = 9 :=
by
  -- Proof goes here
  sorry

end find_second_number_l553_553562


namespace value_range_l553_553290

variable {x a1 a2 y b1 b2 : ℝ}

-- Conditions
def is_arithmetic_sequence : Prop := 
  ∃ d : ℝ, a1 = x + d ∧ a2 = x + 2 * d ∧ y = x + 3 * d

def is_geometric_sequence : Prop := 
  ∃ r : ℝ, b1 = x * r ∧ b2 = x * r^2 ∧ y = x * r^3

-- Main statement to be proven
theorem value_range (h1 : is_arithmetic_sequence) (h2 : is_geometric_sequence) :
  let t := y / x in
  (x + y)^2 / (x * y) ∈ (-∞, 0] ∪ [4, ∞) := by
  sorry

end value_range_l553_553290


namespace throwers_count_l553_553050

variable (totalPlayers : ℕ) (rightHandedPlayers : ℕ) (nonThrowerLeftHandedFraction nonThrowerRightHandedFraction : ℚ)

theorem throwers_count
  (h1 : totalPlayers = 70)
  (h2 : rightHandedPlayers = 64)
  (h3 : nonThrowerLeftHandedFraction = 1 / 3)
  (h4 : nonThrowerRightHandedFraction = 2 / 3)
  (h5 : nonThrowerLeftHandedFraction + nonThrowerRightHandedFraction = 1) : 
  ∃ T : ℕ, T = 52 := by
  sorry

end throwers_count_l553_553050


namespace part1_part2_l553_553716

section Part1
variables (x : ℝ)

def vec_m : ℝ × ℝ := (real.sqrt 3 * real.sin (x / 4), 1)
def vec_n : ℝ × ℝ := (real.cos (x / 4), real.cos (x / 4) ^ 2)
def f (x : ℝ) : ℝ := vec_m.1 * vec_n.1 + vec_m.2 * vec_n.2

theorem part1 (h : f x = 1) : real.cos (2 * real.pi / 3 - x) = -1 / 2 :=
sorry
end Part1

section Part2
variables (A B C a b c : ℝ) (triangle_ABC : a * real.cos C + 1 / 2 * c = b)

def f (B : ℝ) : ℝ := real.sin (B / 2 + real.pi / 6) + 1 / 2

theorem part2 : 1 < f B ∧ f B < 3 / 2 :=
sorry
end Part2

end part1_part2_l553_553716


namespace smallest_n_for_integer_Tn_l553_553011

/-- Define the sum of the reciprocals of the non-zero digits from 1 to 9. --/
def J : ℚ :=
  (∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 / (i : ℚ)))

/-- Define T_n as n * 5^(n-1) * J  --/
def T_n (n : ℕ) : ℚ :=
  n * 5^(n - 1) * J

theorem smallest_n_for_integer_Tn : 
  ∃ n : ℕ, ( ∀ m : ℕ, 0 < m → T_n m ∈ ℤ → n = 63 ) :=
sorry

end smallest_n_for_integer_Tn_l553_553011


namespace find_unknown_number_l553_553527

-- Define the problem parameters
def some_operation (x : ℝ) : ℝ := 40 + 5 * 12 / (x / 3)

-- State the proof problem
theorem find_unknown_number (x : ℝ) (h : some_operation x = 41) : x = 180 := 
sorry

end find_unknown_number_l553_553527


namespace product_of_solutions_abs_x_eq_3_abs_x_minus_1_l553_553641

theorem product_of_solutions_abs_x_eq_3_abs_x_minus_1 :
  let x1 := (3 / 2)
  let x2 := -(3 / 2)
  let product := x1 * x2
  product = -(9 / 4) :=
by
  have h1 : x1 = 3 / 2 := by rfl
  have h2 : x2 = -(3 / 2) := by rfl
  have product_eq : product = x1 * x2 := by rfl
  rw [h1, h2, product_eq]
  calc
    (3 / 2) * -(3 / 2) = -((3 / 2) * (3 / 2)) : by ring
                    ... = - (9 / 4) : by ring
#align product_of_solutions_abs_x_eq_3_abs_x_minus_1
sorry

end product_of_solutions_abs_x_eq_3_abs_x_minus_1_l553_553641


namespace find_higher_rate_l553_553563

-- Definition of the conditions provided in the problem
def principal_amount : ℝ := 5000
def time_period : ℝ := 2
def additional_interest : ℝ := 600
def standard_rate : ℝ := 12

-- Calculating the interest received from the standard rate
def standard_interest : ℝ := principal_amount * (standard_rate / 100) * time_period

-- Defining the higher rate of interest to be determined
noncomputable def higher_rate (R : ℝ) : Prop :=
  principal_amount * (R / 100) * time_period - standard_interest = additional_interest

-- Statement for proving that the higher rate is 18%
theorem find_higher_rate :
  ∃ R : ℝ, higher_rate R ∧ R = 18 :=
by
  -- Proof is omitted 
  sorry

end find_higher_rate_l553_553563


namespace Karthik_upper_limit_l553_553345

noncomputable def KarthikWeight (weight : ℝ) := 
  weight > 55 ∧ weight < 58

theorem Karthik_upper_limit :
  ∀ weight : ℝ,
  (weight > 55 ∧ weight < 60) ∧ weight ≤ 58 ∧ (weight + 55) / 2 = 56.5 → KarthikWeight 58 :=
by {
  intros weight,
  sorry
}

end Karthik_upper_limit_l553_553345


namespace change_received_proof_l553_553569

-- Define the costs and amounts
def regular_ticket_cost : ℕ := 9
def children_ticket_discount : ℕ := 2
def amount_given : ℕ := 2 * 20

-- Define the number of people
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 3

-- Define the costs calculations
def child_ticket_cost := regular_ticket_cost - children_ticket_discount
def total_adults_cost := number_of_adults * regular_ticket_cost
def total_children_cost := number_of_children * child_ticket_cost
def total_cost := total_adults_cost + total_children_cost
def change_received := amount_given - total_cost

-- Lean statement to prove the change received
theorem change_received_proof : change_received = 1 := by
  sorry

end change_received_proof_l553_553569


namespace find_f_2005_1000_l553_553394

-- Define the real-valued function and its properties
def f (x y : ℝ) : ℝ := sorry

-- The condition given in the problem
axiom condition :
  ∀ x y z : ℝ, f x y = f x z - 2 * f y z - 2 * z

-- The target we need to prove
theorem find_f_2005_1000 : f 2005 1000 = 5 := 
by 
  -- all necessary logical steps (detailed in solution) would go here
  sorry

end find_f_2005_1000_l553_553394


namespace rebecca_tent_stakes_l553_553066

variables (T D W : ℕ)

-- Conditions
def drink_mix_eq : Prop := D = 3 * T
def water_eq : Prop := W = T + 2
def total_items_eq : Prop := T + D + W = 22

-- Problem statement
theorem rebecca_tent_stakes 
  (h1 : drink_mix_eq T D)
  (h2 : water_eq T W)
  (h3 : total_items_eq T D W) : 
  T = 4 := 
sorry

end rebecca_tent_stakes_l553_553066


namespace cannot_be_n_plus_2_l553_553166

theorem cannot_be_n_plus_2 (n : ℕ) : 
  ¬(∃ Y, (Y = n + 2) ∧ 
         ((Y = n - 3) ∨ (Y = n - 1) ∨ (Y = n + 5))) := 
by {
  sorry
}

end cannot_be_n_plus_2_l553_553166


namespace function_expression_correct_maximum_value_l553_553478

-- The function and conditions
def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) + 1
def T : ℝ := Real.pi
def ω_condition : ℝ := 2 * Real.pi / T

-- Condition 1: ω = 2
theorem function_expression_correct (A ω : ℝ) (hA : A > 0) (hω : ω > 0) 
  (h_symmetry : 1/2 * T = π/2) : 
  (f x ω = 2 * Real.sin (2 * x - π / 6) + 1) :=
sorry

-- Condition 2: Maximum value of f(x) in the interval [0, π/2]
theorem maximum_value (A : ℝ) (hA : A > 0) : f x 2 ≤ 3 :=
sorry

end function_expression_correct_maximum_value_l553_553478


namespace sum_of_midpoint_xcoords_l553_553492

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l553_553492


namespace raspberry_pies_l553_553413

theorem raspberry_pies (total_pies : ℕ) (r_peach : ℕ) (r_strawberry : ℕ) (r_raspberry : ℕ) (r_sum : ℕ) :
    total_pies = 36 → r_peach = 2 → r_strawberry = 5 → r_raspberry = 3 → r_sum = (r_peach + r_strawberry + r_raspberry) →
    (total_pies : ℝ) / (r_sum : ℝ) * (r_raspberry : ℝ) = 10.8 :=
by
    -- This theorem is intended to state the problem.
    sorry

end raspberry_pies_l553_553413


namespace smallest_n_l553_553232

theorem smallest_n (n : ℕ) : 
  (2^n + 5^n - n) % 1000 = 0 ↔ n = 797 :=
sorry

end smallest_n_l553_553232


namespace simplify_fraction_l553_553450

theorem simplify_fraction : 
  (1:ℚ) / 462 + 17 / 42 = 94 / 231 := 
by
  sorry

end simplify_fraction_l553_553450


namespace area_of_hexagon_l553_553525

-- Defining the problem
variables (A1 B1 C1 A2 B2 C2 : Type) [AffineSpace A1 B1 C1 A2 B2 C2]

-- Assume given areas T1 and T2
variables (T1 T2 : ℝ)

-- Auxiliary function declaration to describe the areas of the two triangles
axiom area_triangle : Π {A B C : Type} [AffineSpace A B C], ℝ

-- Hypotheses for the problem
-- Hypothesis that: The sides are parallel and the area of the triangles are known
axiom parallel_sides :
  (A1 B1 C1 A2 B2 C2 : Type)
  [AffineSpace A1 B1 C1 A2 B2 C2] →
  ((∥(A1 B1) = ∥(A2 B2)) ∧ (∥(B1 C1) = ∥(B2 C2)) ∧ (∥(A1 C1) = ∥(A2 C2)))

-- Stating the theorem
theorem area_of_hexagon :
  (A1 B1 C1 A2 B2 C2 : Type) [AffineSpace A1 B1 C1 A2 B2 C2] →
  (parallel_sides A1 B1 C1 A2 B2 C2) →
  (T1 T2 : ℝ) →
  area_hexagon_form_midpoints = (T1 + T2)/4 + sqrt (T1 * T2) := 
sorry

end area_of_hexagon_l553_553525


namespace circumscribed_circle_radius_l553_553179

/-- Given a sector with obtuse central angle θ taken from a circle of radius 12,
    the radius of the circumscribed circle about the sector is 12 * sec(θ / 2). -/
theorem circumscribed_circle_radius (θ : ℝ) (h1 : θ > π / 2) :
  let R := 12 * (1 / Real.cos (θ / 2)) in R = 12 * Real.sec (θ / 2) :=
by
  sorry

end circumscribed_circle_radius_l553_553179


namespace sum_of_midpoints_l553_553498

theorem sum_of_midpoints (p q r : ℝ) (h : p + q + r = 15) :
  (p + q) / 2 + (p + r) / 2 + (q + r) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l553_553498


namespace correct_statements_l553_553531

def Frequency := (m n : ℕ) → ℚ
def Probability := ℚ
def isFrequency (m n : ℕ) : Frequency m n := m / n
def isProbability (p : Probability) := p

def statement1 : Prop := 
  ∀ (m n : ℕ), Frequency m n = Probability

def statement2 : Prop :=
  ∀ (m n : ℕ), isFrequency m n = isProbability (m / n)

def statement3 : Prop :=
  ∀ (m n : ℕ), (isFrequency m n ∧ m ≠ n) → ¬ (isProbability (m / n))

def statement4 : Prop :=
  ∀ (m n : ℕ), n > 1000 → isFrequency m n ≈ isProbability (m / n)

theorem correct_statements :
  statement1 ∧ statement3 ∧ statement4 :=
by
  sorry

end correct_statements_l553_553531


namespace walking_distance_l553_553930

theorem walking_distance (x : ℝ) :
  ∃ x : ℝ, (∃ (y : ℝ), y = 6 ∧ (x - 3 * Real.sqrt 3)^2 + (3)^2 = (2 * Real.sqrt 3)^2) ↔ 
    (x = 3 * Real.sqrt 3 + Real.sqrt 3 ∨ x = 3 * Real.sqrt 3 - Real.sqrt 3) :=
begin
  sorry
end

end walking_distance_l553_553930


namespace arith_seq_sum_signs_l553_553752

variable {α : Type*} [LinearOrderedField α]
variable {a : ℕ → α} {S : ℕ → α} {d : α}

noncomputable def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  n * (a 1 + a n) / 2

-- Given conditions
variable (a_8_neg : a 8 < 0)
variable (a_9_pos : a 9 > 0)
variable (a_9_greater_abs_a_8 : a 9 > abs (a 8))

-- The theorem to prove
theorem arith_seq_sum_signs (h : is_arith_seq a) :
  (∀ n, n ≤ 15 → sum_first_n_terms a n < 0) ∧ (∀ n, n ≥ 16 → sum_first_n_terms a n > 0) :=
sorry

end arith_seq_sum_signs_l553_553752


namespace possible_values_of_a_l553_553465

def data_set_without_a := [60, 70, 70, 72, 73, 74, 74, 75, 76, 77, 79, 80, 83, 85, 87, 93, 100]

def is_upper_quartile (a : ℕ) : Prop :=
  ∃ a, a ∈ (a :: data_set_without_a) ∧ 
  let sorted := List.sort (compare) (a :: data_set_without_a) in
  sorted.getD 13 0 = a ∨ sorted.getD 14 0 = a

theorem possible_values_of_a :
  ∃ a, is_upper_quartile a ∧ 83 ≤ a ∧ a ≤ 85 :=
sorry

end possible_values_of_a_l553_553465


namespace count_valid_six_digit_numbers_l553_553844

def is_divisible_by (n divisor : ℕ) : Prop := divisor ∣ n

def validates_form (n : ℕ) (a b : ℕ) : Prop :=
  n = 100000 * a + 99910 + b

def satisfies_divisibility_conditions (a b : ℕ) : Prop :=
  (is_divisible_by (100000 * a + 9991 * 10 + b) 12)

theorem count_valid_six_digit_numbers : 
  ∃! (count : ℕ), count = 8 ∧
  count = (Finset.card (Finset.filter (λ n, ∃ a b, validates_form n a b ∧ satisfies_divisibility_conditions a b) (Finset.range 1000000))) :=
sorry

end count_valid_six_digit_numbers_l553_553844


namespace solve_problem_l553_553245

theorem solve_problem :
  ∃ (x y : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y = 5 :=
by
  sorry

end solve_problem_l553_553245


namespace find_value_of_expression_l553_553398

theorem find_value_of_expression (a b c : ℝ) (h : (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0) : a + 2*b + 3*c = 18 := 
sorry

end find_value_of_expression_l553_553398


namespace reassemble_chessboard_l553_553972

/-- Given a piece of checkered linoleum that can be cut into two parts, we aim to show
  that these parts can be reassembled into an 8x8 chessboard without repainting.
  We assume:
  - The piece can be divided into exactly two parts.
  - No squares need to be repainted.
  - The reassembled pieces form the correct alternating pattern of a chessboard.
--/
theorem reassemble_chessboard (A B : Set (ℕ × ℕ)) :
  (exists P Q : Set (ℕ × ℕ), is_valid_cut A B P Q ∧ is_valid_rotation P Q A B) →
  can_be_reassembled P Q into_chessboard :=
sorry

/-- Define what it means for the cutting to be valid --/
def is_valid_cut (A B P Q : Set (ℕ × ℕ)) : Prop :=
-- Definition of valid cut, ensuring exactly two parts are obtained.
sorry

/-- Define what it means for the rotation to be valid --/
def is_valid_rotation (P Q A B : Set (ℕ × ℕ)) : Prop :=
-- Definition of valid rotation, ensuring the pattern remains intact.
sorry

/-- Define what it means for the pieces to be reassembled into a chessboard --/
def can_be_reassembled (P Q : Set (ℕ × ℕ)) (into_chessboard : Set (ℕ × ℕ)) : Prop :=
-- Definition of reassembled chessboard verification, ensuring correct formation.
sorry

end reassemble_chessboard_l553_553972


namespace no_intersection_of_asymptotes_l553_553998

noncomputable def given_function (x : ℝ) : ℝ :=
  (x^2 - 9 * x + 20) / (x^2 - 9 * x + 18)

theorem no_intersection_of_asymptotes : 
  (∀ x, x = 3 → ¬ ∃ y, y = given_function x) ∧ 
  (∀ x, x = 6 → ¬ ∃ y, y = given_function x) ∧ 
  ¬ ∃ x, (x = 3 ∨ x = 6) ∧ given_function x = 1 := 
by
  sorry

end no_intersection_of_asymptotes_l553_553998


namespace minimum_positive_numbers_l553_553564

noncomputable def min_positive_numbers (n : ℕ) (numbers : list ℤ) (adj_products : list ℤ) : ℕ :=
  sorry

theorem minimum_positive_numbers {numbers : list ℤ}
  (h1 : numbers.length = 100)
  (h2 : ∀ i, numbers.nth i ≠ 0)
  (adj_products : list ℤ := list.map (λ i, numbers.nth i * numbers.nth ((i + 1) % numbers.length)) (list.range numbers.length))
  (h3 : list.filter (λ x, 0 < x) adj_products = list.filter (λ x, 0 < x) numbers) :
  min_positive_numbers 100 numbers adj_products = 34 :=
sorry

end minimum_positive_numbers_l553_553564


namespace plane_speeds_l553_553130

-- Define the speeds of the planes
def speed_slower (x : ℕ) := x
def speed_faster (x : ℕ) := 2 * x

-- Define the distances each plane travels in 3 hours
def distance_slower (x : ℕ) := 3 * speed_slower x
def distance_faster (x : ℕ) := 3 * speed_faster x

-- Define the total distance
def total_distance (x : ℕ) := distance_slower x + distance_faster x

-- Prove the speeds given the total distance
theorem plane_speeds (x : ℕ) (h : total_distance x = 2700) : speed_slower x = 300 ∧ speed_faster x = 600 :=
by {
  sorry
}

end plane_speeds_l553_553130


namespace jen_lisa_spent_l553_553243

theorem jen_lisa_spent (J L : ℝ) 
  (h1 : L = 0.8 * J) 
  (h2 : J = L + 15) : 
  J + L = 135 := 
by
  sorry

end jen_lisa_spent_l553_553243


namespace always_take_umbrella_is_optimal_l553_553356

variable (P_R : ℚ) (P_not_R : ℚ) (P_F_given_R : ℚ) (P_not_F_given_R : ℚ) 
          (P_F_given_not_R : ℚ) (P_not_F_given_not_R : ℚ)

axiom prob_of_rain: P_R = 1/3
axiom prob_of_no_rain: P_not_R = 1 - P_R
axiom forecast_given_rain_correct: P_F_given_R = 0.8
axiom forecast_given_rain_wrong: P_not_F_given_R = 1 - P_F_given_R
axiom forecast_given_no_rain_wrong: P_F_given_not_R = 0.5
axiom forecast_given_no_rain_correct: P_not_F_given_not_R = 1 - P_F_given_not_R

theorem always_take_umbrella_is_optimal :
  let EU_always := P_not_R * 1,
      EU_never := P_R * 2,
      P_F := P_F_given_R * P_R + P_F_given_not_R * P_not_R,
      P_not_F := P_not_F_given_R * P_R + P_not_F_given_not_R * P_not_R,
      P_R_given_F := (P_F_given_R * P_R) / P_F,
      P_not_R_given_F := 1 - P_R_given_F,
      P_R_given_not_F := (P_not_F_given_R * P_R) / P_not_F,
      P_not_R_given_not_F := 1 - P_R_given_not_F,
      EU_forecast := (P_R_given_F * P_F) * 1 + (P_not_R_given_F * P_F) * 1 + (P_R_given_not_F * P_not_F) * 2 in
  EU_always ≤ min EU_never EU_forecast :=
by 
  unfold EU_always EU_never P_F P_not_F P_R_given_F P_not_R_given_F P_R_given_not_F P_not_R_given_not_F EU_forecast;
  have h_always: EU_always = 2/3 := by sorry;
  have h_never: EU_never = 2/3 := by sorry;
  have h_forecast: EU_forecast = 7/9 := by sorry;
  exact le_min (le_refl _) (by linarith [h_always, h_forecast])


end always_take_umbrella_is_optimal_l553_553356


namespace hyperbola_equation_l553_553284

theorem hyperbola_equation (a b : ℝ)
  (hyp_center_origin : (0, 0) = (0, 0))
  (hyp_focus : ∃ F : ℝ × ℝ, F = (sqrt 7, 0))
  (line_eq : ∀ x y : ℝ, y = x - 1 → ∃M N : ℝ × ℝ, fib_eq : (M.1 + N.1) / 2 = -2 / 3 → True) :
  (∃ a b : ℝ, c = sqrt (a^2 + b^2) ∧ c = sqrt 7)
  → (∃ a b : ℝ, a^2 = 2 ∧ b^2 = 5 ∧ (by refl : a ≠ 0) :
  ∃ x y : ℝ, ((x^2 / 2) - (y^2 / 5)) = 1 := sorry

end hyperbola_equation_l553_553284


namespace sum_of_first_ten_interesting_numbers_l553_553975

-- Define a predicate for interesting numbers
def interesting_numbers (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p^2 * q

-- Define the first ten interesting numbers
def first_ten_interesting_numbers : list ℕ := [12, 20, 28, 18, 45, 63, 50, 75, 44, 52]

-- Prove that the sum of the first ten interesting numbers is 407
theorem sum_of_first_ten_interesting_numbers : 
  list.sum first_ten_interesting_numbers = 407 :=
by
  sorry

end sum_of_first_ten_interesting_numbers_l553_553975


namespace find_solutions_l553_553994

-- A predicate for the given equation
def satisfies_equation (x y : ℕ) : Prop := 
  (1 / (x : ℚ) + 1 / (y : ℚ)) = 1 / 4

-- Define the set of solutions
def solutions : Set (ℕ × ℕ) := 
  {(5, 20), (6, 12), (8, 8), (12, 6), (20, 5)}

-- The goal is to prove that these solutions are the only ones
theorem find_solutions : 
  {p : ℕ × ℕ | satisfies_equation p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0} = solutions := 
by 
  sorry

end find_solutions_l553_553994


namespace max_value_huabeichusai_l553_553361

-- Define the digits for the characters, ensuring they are different
variables (Hua: ℕ) (Mu: ℕ) (Jie: ℕ) (Gong: ℕ) (Chu: ℕ) (Sai: ℕ) (Tu: ℕ) (Nian: ℕ)
variables (ten: ℕ)
variables (x: ℕ → ℕ → ℕ → ℕ)

-- Conditions of the problem
axiom diff_digits : different [Hua, Mu, Jie, Gong, Chu, Sai, Tu, Nian]
axiom digit_range : ∀ {a}, a ∈ [Hua, Mu, Jie, Gong, Chu, Sai, Tu, Nian] → a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
axiom equation : ten * Hua + ((Hua + Chu) % 10 * 10 + (Mu + Jie) % 10 * 10 + (Gong + Chu) % 10 * 10 + Tu * 10) = 2011

-- Problem to prove
theorem max_value_huabeichusai : Hua * 1000 + Mu * 100 + Jie * 10 + Chu = 1769 :=
by sorry

end max_value_huabeichusai_l553_553361


namespace neither_sufficient_nor_necessary_l553_553274

variable (a b : ℝ)

def p : Prop := a + b > 0
def q : Prop := ab > 0

theorem neither_sufficient_nor_necessary :
  ¬(p a b → q a b) ∧ ¬(q a b → p a b) :=
  by sorry

end neither_sufficient_nor_necessary_l553_553274


namespace ticket_distribution_l553_553116

theorem ticket_distribution:
  let tickets := {1, 2, 3, 4, 5}
  let people := {A, B, C, D}
  ∃ (f : tickets → people),
    (∀ p ∈ people, 1 ≤ (f ⁻¹' {p}).to_finset.card ∧ (f ⁻¹' {p}).to_finset.card ≤ 2) ∧
    (∀ p ∈ people, (f ⁻¹' {p}).to_finset.card = 2 → ∃ n, (f ⁻¹' {p}).to_finset = {n, n+1}) →
    fintype.card {f | ∀ p ∈ people, 1 ≤ (f ⁻¹' {p}).to_finset.card ∧ (f ⁻¹' {p}).to_finset.card ≤ 2 ∧ (∀ p ∈ people, (f ⁻¹' {p}).to_finset.card = 2 → ∃ n, (f ⁻¹' {p}).to_finset = {n, n+1})} = 96 :=
by
  sorry

end ticket_distribution_l553_553116


namespace find_n_l553_553399

theorem find_n 
  (n : ℕ) 
  (b : Fin (n + 1) → ℝ) 
  (h0 : b 0 = 41) 
  (h1 : b 1 = 76) 
  (hn : b n = 0)
  (hrec : ∀ k : Fin n, 1 ≤ k → b (k + 1) = b (k - 1) - 4 / b k)
  : n = 777 := sorry

end find_n_l553_553399


namespace max_intersections_ellipse_three_lines_l553_553954

open Set

theorem max_intersections_ellipse_three_lines :
  ∃ (ellipse : Set ℝ × ℝ) (l1 l2 l3 : Set ℝ × ℝ),
    (∀ p1 p2 : ℝ × ℝ, p1 ∈ ellipse → p2 ∈ ellipse → p1 ≠ p2 → (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 > 0) ∧
    (∀ (l p : Set ℝ × ℝ), l ≠ ellipse → p = (λ x : ℝ, Set.Icc 0 1) → p ≠ (0,0) ∧ l ∩ ellipse = Set.empty → 
     (∃ x y : ℝ, (x, y) ∈ ellipse ∧ (x, y) ∈ l ∧ l ∩ ellipse = { (x, y) })) ∧
    (∀ l1 l2 l3 : Set ℝ × ℝ, l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → 
     (∃ p1 p2 p3 : ℝ × ℝ, l1 ∩ ellipse = { p1, p2 } ∧ l2 ∩ ellipse = { p1, p2 } ∧ l3 ∩ ellipse = { p1, p2 } ∧ 
      l1 ∩ l2 ∩ l3 = { p1, p2, p3 })) ∧
    (∃ p1 p2 p3 : ℝ × ℝ, l1 ∩ l2 = {p1} ∧ l2 ∩ l3 = {p2} ∧ l1 ∩ l3 = {p3} →
    ∃ n : ℕ, n = 9) 
: sorry

end max_intersections_ellipse_three_lines_l553_553954


namespace part1_part2_l553_553715

open Real

-- Define the vectors a and b
def a : (ℝ × ℝ) := ⟨5, -12⟩
def b : (ℝ × ℝ) := ⟨-3, 4⟩

-- Definition for part 1: cosine value of the angle θ
def cos_angle_between (u v : (ℝ × ℝ)) :=
  (u.1 * v.1 + u.2 * v.2) / (sqrt (u.1^2 + u.2^2) * sqrt (v.1^2 + v.2^2))

theorem part1 : cos_angle_between a b = -63 / 65 :=
  sorry

-- Definition for part 2: perpendicular vectors and solve for t
def perpendicular (u v : (ℝ × ℝ)) := u.1 * v.1 + u.2 * v.2 = 0

theorem part2 (t : ℝ) : perpendicular (⟨5 - 3 * t, -12 + 4 * t⟩) (⟨8, -16⟩) → t = 29 / 11 :=
  sorry

end part1_part2_l553_553715


namespace smallest_n_for_integer_Tn_l553_553009

/-- Define the sum of the reciprocals of the non-zero digits from 1 to 9. --/
def J : ℚ :=
  (∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 / (i : ℚ)))

/-- Define T_n as n * 5^(n-1) * J  --/
def T_n (n : ℕ) : ℚ :=
  n * 5^(n - 1) * J

theorem smallest_n_for_integer_Tn : 
  ∃ n : ℕ, ( ∀ m : ℕ, 0 < m → T_n m ∈ ℤ → n = 63 ) :=
sorry

end smallest_n_for_integer_Tn_l553_553009


namespace sqrt_inequality_l553_553897

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
by
  sorry

end sqrt_inequality_l553_553897


namespace find_bc_l553_553254

theorem find_bc (b c : ℤ) (h : ∀ x : ℝ, x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = 1 ∨ x = 2) :
  b = -3 ∧ c = 2 := by
  sorry

end find_bc_l553_553254


namespace correct_statements_l553_553146

namespace ComplexProof

open Complex

noncomputable def statementA (z₁ z₂ : ℂ) : Prop :=
  (|z₁ - z₂| = 0) → (conj z₁ = conj z₂)

noncomputable def statementB (z₁ z₂ : ℂ) : Prop :=
  (z₁ = conj z₂) → (conj z₁ = z₂)

noncomputable def statementC (z₁ z₂ : ℂ) : Prop :=
  (|z₁| = |z₂|) → (z₁ * conj z₁ = z₂ * conj z₂)

noncomputable def statementD (z₁ z₂ : ℂ) : Prop :=
  (∃ z₁ z₂ : ℂ, |z₁| = |z₂| ∧ z₁^2 ≠ z₂^2)

theorem correct_statements (z₁ z₂ : ℂ) :
  statementA z₁ z₂ ∧ statementB z₁ z₂ ∧ statementC z₁ z₂ ∧ statementD z₁ z₂ :=
by
sory

end ComplexProof

end correct_statements_l553_553146


namespace length_XY_is_64_l553_553800

open Real

variables (X Y G H I J : Point)
variables (dXY dXG dXH dXI dXJ : ℝ)

-- Definitions based on the conditions
def midpoint (A B C : Point) : Prop := dist A C = dist B C

axiom midpoint_XY : midpoint X G Y
axiom midpoint_XG : midpoint X H G
axiom midpoint_XH : midpoint X I H
axiom midpoint_XI : midpoint X J I

axiom XJ_eq_4 : dist X J = 4

-- Prove XY == 64 given the conditions
theorem length_XY_is_64 : dist X Y = 64 :=
by sorry

end length_XY_is_64_l553_553800


namespace smallest_n_for_Tn_integer_l553_553028

noncomputable def K : ℚ := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n-1)) * K

theorem smallest_n_for_Tn_integer :
  ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m < n, T_n m ∉ ℤ :=
  sorry

end smallest_n_for_Tn_integer_l553_553028


namespace arrangements_count_l553_553117

noncomputable def count_arrangements : ℕ := sorry

-- Definition of the problem conditions
def seven_students := {xiao_ming, xiao_li, xiao_zhang, student_1, student_2, student_3, student_4} 
def is_center (arrangement : list string) := arrangement.nth 3 = some "xiao_ming"
def together (arrangement : list string) := 
  let idx_li := arrangement.index_of "xiao_li"
  let idx_zhang := arrangement.index_of "xiao_zhang"
  (idx_li + 1 = idx_zhang ∨ idx_zhang + 1 = idx_li)

-- Proof statement
theorem arrangements_count : ∀ (arrangement : list string), 
  arrangement.length = 7 →
  is_center arrangement →
  together arrangement →
  count_arrangements = 192 :=
  sorry

end arrangements_count_l553_553117


namespace share_of_gold_NWF_l553_553223

theorem share_of_gold_NWF
  (total_NWF : ℝ)
  (deductions : list ℝ)
  (total_gold : ℝ)
  (initial_share_percentage : ℝ)
  : total_NWF = 1388.01 ∧
    deductions = [41.89, 2.77, 478.48, 309.72, 0.24] ∧
    total_gold = 554.91 ∧
    initial_share_percentage = 31.8 → 
    (total_gold / total_NWF) * 100 = 39.98 := 
begin
  intros,
  sorry,
end

end share_of_gold_NWF_l553_553223


namespace sum_of_midpoints_x_coordinates_l553_553499

theorem sum_of_midpoints_x_coordinates (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
  sorry

end sum_of_midpoints_x_coordinates_l553_553499


namespace probability_sum_is_9_l553_553152

open Finset

def s : Finset ℕ := {2, 3, 4, 5}
def b : Finset ℕ := {4, 5, 6, 7, 8}

def count_pairs_with_sum_9 (s b : Finset ℕ) : ℕ :=
  (s.product b).count (λ p, p.1 + p.2 = 9)

def total_pairs (s b : Finset ℕ) : ℕ :=
  s.card * b.card

theorem probability_sum_is_9 : 
  (count_pairs_with_sum_9 s b : ℝ) / total_pairs s b = 3 / 20 :=
by
  sorry

end probability_sum_is_9_l553_553152


namespace equal_initial_values_l553_553259

theorem equal_initial_values (n : ℕ) (a : ℕ → ℤ)
  (h_odd : n % 2 = 1)
  (h_int : ∀ b : ℕ → ℤ, (∀ i : ℕ, b i = (a i + a (i+1)) / 2) → (∀ i : ℕ, b i ∈ ℤ)) :
  (∀ i j : ℕ, i < n → j < n → a i = a j) := 
sorry

end equal_initial_values_l553_553259


namespace regular_hexagon_side_length_and_perimeter_l553_553431

-- Side length of the regular hexagon
def side_length (d : ℝ) : ℝ := (2 * d) / (Real.sqrt 3)

-- Perimeter of the regular hexagon
def perimeter (s : ℝ) : ℝ := 6 * s

theorem regular_hexagon_side_length_and_perimeter :
  ∀ (d : ℝ), 
  d = 10 → 
  side_length d = (20 / 3) * Real.sqrt 3 ∧
  perimeter (side_length d) = 40 * Real.sqrt 3 := by
  intros d h
  have h1 : side_length d = (20 / 3) * Real.sqrt 3 := by
    rw [side_length, h]
    sorry
  have h2 : perimeter (side_length d) = 40 * Real.sqrt 3 := by
    rw [perimeter, h1]
    sorry
  exact ⟨h1, h2⟩

end regular_hexagon_side_length_and_perimeter_l553_553431


namespace polygon_square_segments_perpendicular_and_ratio_l553_553052

-- Define the polygon and its properties
variable (n : ℕ) (h : n ≥ 3)

-- Define the statements about the segments B_j C_j and O A_j
theorem polygon_square_segments_perpendicular_and_ratio (A : Fin n → ℂ) (O : ℂ)
  (h_reg : ∀ i j, abs (A i - A j) = abs (A 0 - A 1))
  (B C : Fin n → ℂ)
  (h_squares : ∀ i, B i - A i = -I * (A ((i + 1) % n) - A i) ∧ C ((i + 1) % n) - A ((i + 1) % n) = -I * (A i - A ((i + 1) % n)))
  (h_center : O = 0) :
  (∀ j, isPerpendicular (line_Through O (A j)) (line_Through (B j) (C j))) ∧
  (∀ j, abs (B j - C ((j + 1) % n)) / abs (O - A j) = 2 * (1 - Real.cos(2 * Real.pi / n))) :=
sorry

end polygon_square_segments_perpendicular_and_ratio_l553_553052


namespace probability_xi_gt_6_equals_0_4_l553_553856

noncomputable def xi (μ σ : ℝ) (x : ℝ) : ℝ :=
  ∫ t in -∞ .. x, exp (- (t - μ)^2 / (2 * σ^2)) / (σ * sqrt (2 * π))

-- Given that xi follows a normal distribution N(4, σ^2)
def follows_normal_distribution (μ σ : ℝ) (x : ℝ) : Prop :=
  xi μ σ x ∈ measure_theory.measure_space.volume

-- Given P(ξ > 2) = 0.6
axiom P_xi_gt_2 : xi 4 σ 2 = 0.6

theorem probability_xi_gt_6_equals_0_4 (μ := 4) (σ : ℝ) : xi μ σ 6 = 0.4 := by
  sorry

end probability_xi_gt_6_equals_0_4_l553_553856


namespace distance_between_foci_l553_553617

def point := (Int × Int)

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def eq1 (x y : ℝ) : Prop :=
  Real.sqrt((x - 4)^2 + (y - 3)^2) + Real.sqrt((x + 6)^2 + (y - 5)^2) = 24

noncomputable def F1 : point := (4, 3)
noncomputable def F2 : point := (-6, 5)

theorem distance_between_foci :
  distance F1 F2 = 2 * Real.sqrt 26 := by
  simp only [distance, F1, F2]
  sorry

end distance_between_foci_l553_553617


namespace compound_interest_time_period_l553_553069

theorem compound_interest_time_period :
  ∃ t : ℝ, 
    (16537.5 = 15000 * (1 + 0.10 / 2)^(2 * t)) ∧ t = 1 :=
by {
  use 1,
  simp,
  rw [mul_assoc, mul_comm (1 + 0.10 / 2)],
  norm_num,
  sorry
}

end compound_interest_time_period_l553_553069


namespace school_student_ratio_l553_553123

theorem school_student_ratio :
  ∀ (F S T : ℕ), (T = 200) → (S = T + 40) → (F + S + T = 920) → (F : ℚ) / (S : ℚ) = 2 / 1 :=
by
  intros F S T hT hS hSum
  sorry

end school_student_ratio_l553_553123


namespace solve_system_of_equations_l553_553815

theorem solve_system_of_equations :
  ∃ x y : ℚ, (4 * x - 3 * y = -7) ∧ (5 * x + 4 * y = -2) ∧ x = -34/31 ∧ y = 27/31 :=
by
  use (-34/31)
  use (27/31)
  split
  { norm_num }
  split
  { norm_num }
  split
  { rfl }
  { rfl }
  sorry -- This is where the proof steps would go to show the solution is correct.

end solve_system_of_equations_l553_553815


namespace equations_neither_directly_nor_inversely_proportional_l553_553629

-- Definitions for equations
def equation1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def equation2 (x y : ℝ) : Prop := 4 * x * y = 12
def equation3 (x y : ℝ) : Prop := y = 1/2 * x
def equation4 (x y : ℝ) : Prop := 5 * x - 2 * y = 20
def equation5 (x y : ℝ) : Prop := x / y = 5

-- Theorem stating that y is neither directly nor inversely proportional to x for the given equations
theorem equations_neither_directly_nor_inversely_proportional (x y : ℝ) :
  (¬∃ k : ℝ, x = k * y) ∧ (¬∃ k : ℝ, x * y = k) ↔ 
  (equation1 x y ∨ equation4 x y) :=
sorry

end equations_neither_directly_nor_inversely_proportional_l553_553629


namespace analogous_to_parallelepiped_l553_553883

-- Define the geometric properties of the figures and the parallelepiped.
def parallelepiped (shape : Type) : Prop := 
  ∃ (f1 f2 f3 f4 f5 f6 : shape), 
    (parallel f1 f2) ∧ (parallel f3 f4) ∧ (parallel f5 f6)

def triangle (shape : Type) : Prop := -- Define a triangle, typically no parallel sides.
  ∃ (a b c : shape), ¬ (parallel a b ∨ parallel b c ∨ parallel c a)

def trapezoid (shape : Type) : Prop := -- Define a trapezoid, one pair of parallel sides.
  ∃ (a b c d : shape), (parallel a b ∧ ¬ (parallel c d)) ∨ (¬ (parallel a b) ∧ (parallel c d))

def parallelogram (shape : Type) : Prop := -- Define a parallelogram, two pairs of parallel sides.
  ∃ (a b c d : shape), (parallel a b) ∧ (parallel c d)

def rectangle (shape : Type) : Prop := -- Define a rectangle, a specific case of parallelogram.
  ∃ (a b c d : shape), (equal_length a b) ∧ (equal_length c d) ∧ (right_angle a b c) ∧ parallelogram shape

-- Prove that the parallelogram is the most analogous figure to a parallelepiped.
theorem analogous_to_parallelepiped (shape : Type) :
  parallelepiped shape → parallelogram shape :=
sorry -- Proof is omitted

end analogous_to_parallelepiped_l553_553883


namespace g_is_odd_l553_553762

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l553_553762


namespace mth_order_arithmetic_progression_l553_553553

-- Define the sequence a_n = (m+n)! / n!
def seq (m n : ℕ) : ℕ := (m + n)! / n!

-- Define the m-th differences
def m_diff (m n : ℕ) (a : ℕ → ℕ) : ℕ := (finset.range n).sum (λ k, (-1)^k * nat.choose m k * a (n - k))

-- Theorem stating the desired property
theorem mth_order_arithmetic_progression (m : ℕ) :
  ∀ n : ℕ, m_diff m n (seq m) = m! := 
sorry

end mth_order_arithmetic_progression_l553_553553


namespace number_of_bricks_required_l553_553161

def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.10
def brick_height : ℝ := 0.075

def wall_length : ℝ := 25.0
def wall_width : ℝ := 2.0
def wall_height : ℝ := 0.75

def brick_volume := brick_length * brick_width * brick_height
def wall_volume := wall_length * wall_width * wall_height

theorem number_of_bricks_required :
  wall_volume / brick_volume = 25000 := by
  sorry

end number_of_bricks_required_l553_553161


namespace fraction_simplification_l553_553453

-- Definitions based on conditions and question
def lcm_462_42 : ℕ := 462
def prime_factors_462 : List ℕ := [2, 3, 7, 11]
def prime_factors_42 : List ℕ := [2, 3, 7]

-- Main theorem statement
theorem fraction_simplification :
  (1 / 462) + (17 / 42) = 94 / 231 :=
by
  sorry

end fraction_simplification_l553_553453


namespace factor_polynomial_l553_553615

-- Statement of the proof problem
theorem factor_polynomial (x y z : ℝ) :
    x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 =
    (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) :=
by
  sorry

end factor_polynomial_l553_553615


namespace triangle_m_P_Q_R_perimeter_l553_553128

theorem triangle_m_P_Q_R_perimeter :
  let PQ := 150
  let QR := 270
  let PR := 210
  let m_P_length := 65
  let m_Q_length := 55
  let m_R_length := 25
  let lambda := m_P_length / QR
  let XZ := QR * lambda
  let YZ := PQ * lambda
  let XY := PR * lambda
  perimeter := XZ + YZ + XY
  perimeter = 151.652 :=
by
  sorry

end triangle_m_P_Q_R_perimeter_l553_553128


namespace smallest_n_for_integer_Tn_l553_553014

/-- Define the sum of the reciprocals of the non-zero digits from 1 to 9. --/
def J : ℚ :=
  (∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 / (i : ℚ)))

/-- Define T_n as n * 5^(n-1) * J  --/
def T_n (n : ℕ) : ℚ :=
  n * 5^(n - 1) * J

theorem smallest_n_for_integer_Tn : 
  ∃ n : ℕ, ( ∀ m : ℕ, 0 < m → T_n m ∈ ℤ → n = 63 ) :=
sorry

end smallest_n_for_integer_Tn_l553_553014


namespace age_difference_l553_553373

theorem age_difference (J B : ℕ) (h1 : J = 3 * B) (h2 : J + 3 = 2 * (B + 3)) : J - B = 6 :=
by
sor


end age_difference_l553_553373


namespace quadratic_sum_of_squares_l553_553264

theorem quadratic_sum_of_squares (α β : ℝ) (h1 : α * β = 3) (h2 : α + β = 7) : α^2 + β^2 = 43 := 
by
  sorry

end quadratic_sum_of_squares_l553_553264


namespace travel_time_person_travel_time_l553_553934

def distance : ℝ := 125 -- The distance to the destination in km
def speed : ℝ := 25 -- The speed of travel in km/hr

theorem travel_time (d : ℝ) (s : ℝ) (t : ℝ) (h_d : d = distance) (h_s : s = speed) : t = d / s :=
by
  rw [h_d, h_s]
  exact (eq.refl (t = d / s))
  sorry

theorem person_travel_time : ∃ t : ℝ, t = 5 ∧ t = distance / speed :=
by
  use 5
  split
  . rfl
  . exact travel_time distance speed 5 rfl rfl
  sorry

end travel_time_person_travel_time_l553_553934


namespace average_speed_l553_553095

theorem average_speed (total_distance : ℕ) (total_time : ℕ) : total_distance = 160 → total_time = 5 → total_distance / total_time = 32 :=
begin
  sorry
end

end average_speed_l553_553095


namespace sum_x_midpoints_of_triangle_l553_553506

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l553_553506


namespace total_savings_1990_percent_less_jane_1990_l553_553743

-- Definitions based on conditions
def DicksSavings1989 := 5000
def JanesSavings1989 := 3000
def InterestRate := 0.03
def AdditionalSavingsPercent := 0.10

-- Definitions calculating required values
def DicksSavings1990 := DicksSavings1989 * (1 + AdditionalSavingsPercent)
def InterestOnDicksSavings1989 := DicksSavings1989 * InterestRate
def InterestOnJanesSavings1989 := JanesSavings1989 * InterestRate

-- Calculating the total savings
def TotalSavings1990 := DicksSavings1990 + InterestOnDicksSavings1989 + JanesSavings1989 + InterestOnJanesSavings1989

-- Proof statements
theorem total_savings_1990 :
  TotalSavings1990 = 8740 := sorry

theorem percent_less_jane_1990 :
  0% = 0 := sorry

end total_savings_1990_percent_less_jane_1990_l553_553743


namespace problem_condition_problem1_problem2_l553_553760

-- Given Condition
theorem problem_condition (A B C : ℝ) (h : 4 * sin A * sin B * cos C = sin A ^ 2 + sin B ^ 2) : Prop := 
sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2

-- Problem 1: Prove the initial condition
theorem problem1 (A B C : ℝ) (h : 4 * sin A * sin B * cos C = sin A ^ 2 + sin B ^ 2) : 
  sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2 := 
sorry

-- Problem 2: Find the measure of angle A, given C = π/3
theorem problem2 (A B : ℝ) (h1 : sin A ^ 2 + sin B ^ 2 = 2 * (sin (π / 3)) ^ 2) (h2 : B = A + π / 3) : 
  A = π / 3 := 
sorry

end problem_condition_problem1_problem2_l553_553760


namespace simplify_trig_expression_l553_553459

theorem simplify_trig_expression (α : ℝ) :
  (sin (2 * α) + cos (2 * α) - cos (6 * α) - sin (6 * α)) / 
  (sin (4 * α) + 2 * sin (2 * α) ^ 2 - 1) = 
  2 * sin (2 * α) := by
  sorry

end simplify_trig_expression_l553_553459


namespace count_integers_satisfying_sqrt_condition_l553_553845

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l553_553845


namespace find_missing_number_l553_553482

theorem find_missing_number
  (x y : ℕ)
  (h1 : 30 = 6 * 5)
  (h2 : 600 = 30 * x)
  (h3 : x = 5 * y) :
  y = 4 :=
by
  sorry

end find_missing_number_l553_553482


namespace find_x_l553_553586

theorem find_x :
  ∃ X : ℝ, 0.25 * X + 0.20 * 40 = 23 ∧ X = 60 :=
by
  sorry

end find_x_l553_553586


namespace negation_of_universal_proposition_l553_553100

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) :=
by sorry

end negation_of_universal_proposition_l553_553100


namespace complement_intersection_l553_553409

open Set

-- Define the universal set U
def U : Set ℕ := {x | 0 < x ∧ x < 7}

-- Define Set A
def A : Set ℕ := {2, 3, 5}

-- Define Set B
def B : Set ℕ := {1, 4}

-- Define the complement of A in U
def CU_A : Set ℕ := U \ A

-- Define the complement of B in U
def CU_B : Set ℕ := U \ B

-- Define the intersection of CU_A and CU_B
def intersection_CU_A_CU_B : Set ℕ := CU_A ∩ CU_B

-- The theorem statement
theorem complement_intersection :
  intersection_CU_A_CU_B = {6} := by
  sorry

end complement_intersection_l553_553409


namespace T_product_sum_equiv_l553_553787

noncomputable def T (t : ℝ) (i : ℕ) : ℝ := t^i + t^(-i)

theorem T_product_sum_equiv (t : ℝ) (h : t > 0) (k : ℕ) (hk : k > 0) :
  (∏ j in Finset.range k, T t (2^j)) = (Finset.range k).sum (λ n, if n % 2 = 1 then T t n else 0) :=
sorry

end T_product_sum_equiv_l553_553787


namespace rods_in_mile_l553_553279

theorem rods_in_mile (mile_to_furlongs : 1 = 12) (furlong_to_rods : 1 = 50) : 1 * 12 * 50 = 600 :=
by
  sorry

end rods_in_mile_l553_553279


namespace probability_of_log2_condition_l553_553175

noncomputable def probability_log_condition : ℝ :=
  let a := 0
  let b := 9
  let log_lower_bound := 1
  let log_upper_bound := 2
  let exp_lower_bound := 2^log_lower_bound
  let exp_upper_bound := 2^log_upper_bound
  (exp_upper_bound - exp_lower_bound) / (b - a)

theorem probability_of_log2_condition :
  probability_log_condition = 2 / 9 :=
by
  sorry

end probability_of_log2_condition_l553_553175


namespace a_n_general_formula_constant_c_b_n_arithmetic_sequence_l553_553677

open Real

noncomputable def sequence_a (n : ℕ) : ℝ := 2^(2 - n)

theorem a_n_general_formula (S : ℕ → ℝ) (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n > 0 → S n + a n = 4) :
  ∀ n : ℕ, n > 0 → a n = 2^(2 - n) :=
sorry

theorem constant_c (c : ℕ → ℝ) (d : ℝ → ℝ) 
  (h_c : ∀ n : ℕ, n > 0 → c n = 2 * n + 3)
  (h_d : ∀ n : ℕ, n > 0 → ∃ C : ℝ, C > 0 ∧ C ≠ 1 ∧ d C = c n + log C (sequence_a n)) :
  ∃ C : ℝ, C = sqrt 2 :=
sorry

theorem b_n_arithmetic_sequence (b : ℕ → ℝ) (a : ℕ → ℝ)
  (h_a : ∀ n : ℕ, n > 0 → a n = sequence_a n)
  (h_b : ∀ n : ℕ, n > 0 → b 1 * a n + b 2 * a (n-1) + ∑ i in finset.range (n-1), b (i+1) * a (n-i) = (1/2)^n - (n + 2) / 2) :
  ∀ n : ℕ, n > 0 → b n = (-n - 3) / 8 :=
sorry

end a_n_general_formula_constant_c_b_n_arithmetic_sequence_l553_553677


namespace mixture_weight_l553_553545

theorem mixture_weight
  (weight_a : ℝ)
  (weight_b : ℝ)
  (volume_ratio_a : ℝ)
  (volume_ratio_b : ℝ)
  (total_volume : ℝ)
  (weight_a_per_liter : ℝ := 0.9) -- in kg
  (weight_b_per_liter : ℝ := 0.8) -- in kg) 
  (total_volume_l : ℝ := 4) :

  weight_a_per_liter = 0.9 → 
  weight_b_per_liter = 0.8 →
  total_volume_l = 4 →
  
  ((3 / (3 + 2)) * total_volume_l * weight_a_per_liter +
   (2 / (3 + 2)) * total_volume_l * weight_b_per_liter) = 3.44 :=

begin
  intros,
  sorry,
end

end mixture_weight_l553_553545


namespace time_to_gather_materials_l553_553927

-- Given conditions
def time_to_create_battery : ℕ := 9
def number_of_robots : ℕ := 10
def total_batteries : ℕ := 200
def total_hours : ℕ := 5
def total_minutes : ℕ := total_hours * 60

-- Statement to prove
theorem time_to_gather_materials (x : ℕ) : 
  (number_of_robots * x + number_of_robots * time_to_create_battery) * total_batteries / number_of_robots = total_minutes :=
begin
  -- Total time one robot needs to produce a battery
  let one_robot_battery_minutes := x + time_to_create_battery,

  -- Total number of batteries one robot can produce in total_minutes
  let one_robot_batteries := total_batteries / number_of_robots,

  -- The total production time for one robot
  let total_production_time := one_robot_batteries * one_robot_battery_minutes,

  exact total_minutes
end

end time_to_gather_materials_l553_553927


namespace arccos_one_half_eq_pi_over_three_l553_553609

theorem arccos_one_half_eq_pi_over_three : arccos (1 / 2) = π / 3 :=
sorry

end arccos_one_half_eq_pi_over_three_l553_553609


namespace stephanie_oranges_l553_553077

theorem stephanie_oranges (times_at_store : ℕ) (oranges_per_time : ℕ) (total_oranges : ℕ) 
  (h1 : times_at_store = 8) (h2 : oranges_per_time = 2) :
  total_oranges = 16 :=
by
  sorry

end stephanie_oranges_l553_553077


namespace commuting_days_l553_553583

theorem commuting_days 
  (morning_car_trips : ℕ)
  (afternoon_subway_trips : ℕ)
  (total_subway_commutes : ℕ)
  (hy1 : morning_car_trips = 12)
  (hy2 : afternoon_subway_trips = 20)
  (hy3 : total_subway_commutes = 15) :
  ∃ y, y = 15 :=
by
  -- Define helper variables
  let a := total_subway_commutes - morning_car_trips + afternoon_subway_trips 
  let b := morning_car_trips - a
  let y := a + b
  have ha : a = 15 - 12 := by sorry
  have hb : b = 12 := by sorry
  have hy : y = 15 := by sorry
  exists hy
  exact ⟨hy⟩
  sorry

end commuting_days_l553_553583


namespace only_correct_statement_is_B_l553_553147

-- We'll define the statements first as properties
def is_acute_angle (θ : ℝ) : Prop := θ > 0 ∧ θ < π/2
def is_obtuse_angle (θ : ℝ) : Prop := π/2 < θ ∧ θ < π
def in_first_quadrant (θ : ℝ) : Prop := 0 ≤ Real.mod (θ) (2 * π) ∧ Real.mod (θ) (2 * π) < π / 2
def in_second_quadrant (θ : ℝ) : Prop := π / 2 ≤ Real.mod (θ) (2 * π) ∧ Real.mod (θ) (2 * π) < π

-- Now we define each statement
def statement_A : Prop := ∀ θ : ℝ, θ < 90 * (π / 180) → is_acute_angle θ
def statement_B : Prop := ∀ θ : ℝ, is_obtuse_angle θ → in_second_quadrant θ
def statement_C : Prop := ∀ θ₁ θ₂ : ℝ, in_second_quadrant θ₁ → in_first_quadrant θ₂ → θ₁ > θ₂
def statement_D : Prop := ∀ (α β : ℝ), Real.mod (α) (2 * π) = Real.mod (β) (2 * π) → α = β

-- The single correct statement we need to prove
theorem only_correct_statement_is_B : ¬statement_A ∧ statement_B ∧ ¬statement_C ∧ ¬statement_D :=
  by
    sorry

end only_correct_statement_is_B_l553_553147


namespace find_legs_of_right_triangle_l553_553466

-- Define the problem conditions in Lean 4
variables (AC BC AB AD BD CD Q q : ℝ)
variables (right_triangle : ∃ C : ℝ, 0 ≤ C ∧ C ∈ Interval (0, 1))
variables (triangle_division : ∃ D : ℝ, D = hypotenuse_proj AD BD AB)

-- Define the main theorem statement: lengths of the legs of the given right triangle
theorem find_legs_of_right_triangle
  (h : ∀ AD BD CD, AD * BD = AB) -- AD * BD can be replaced based on the solution step
  (Q_given : ∃ Q, Q_area : Q = (1/2) * AD * CD)
  (q_given : ∃ q, q_area : q = (1/2) * BD * CD)
  (find_lengths : AD * BD * CD = AB) :
  AC = sqrt (2 * (q + Q) * sqrt (q / Q)) ∧ BC = sqrt (2 * (q + Q) * sqrt (Q / q)) := by
  sorry

end find_legs_of_right_triangle_l553_553466


namespace range_of_a_l553_553299

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then x^2 - a * x + 5 else a / x

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ 2 ≤ a ∧ a ≤ 3 :=
begin
  sorry
end

end range_of_a_l553_553299


namespace squirrel_burrow_has_44_walnuts_l553_553160

def boy_squirrel_initial := 30
def boy_squirrel_gathered := 20
def boy_squirrel_dropped := 4
def boy_squirrel_hid := 8
-- "Forgets where he hid 3 of them" does not affect the main burrow

def girl_squirrel_brought := 15
def girl_squirrel_ate := 5
def girl_squirrel_gave := 4
def girl_squirrel_lost_playing := 3
def girl_squirrel_knocked := 2

def third_squirrel_gathered := 10
def third_squirrel_dropped := 1
def third_squirrel_hid := 3
def third_squirrel_returned := 6 -- Given directly instead of as a formula step; 9-3=6
def third_squirrel_gave := 1 -- Given directly as a friend

def final_walnuts := boy_squirrel_initial + boy_squirrel_gathered
                    - boy_squirrel_dropped - boy_squirrel_hid
                    + girl_squirrel_brought - girl_squirrel_ate
                    - girl_squirrel_gave - girl_squirrel_lost_playing
                    - girl_squirrel_knocked + third_squirrel_returned

theorem squirrel_burrow_has_44_walnuts :
  final_walnuts = 44 :=
by
  sorry

end squirrel_burrow_has_44_walnuts_l553_553160


namespace johns_percentage_increase_l553_553892

theorem johns_percentage_increase (original_amount new_amount : ℕ) (h₀ : original_amount = 30) (h₁ : new_amount = 40) :
  (new_amount - original_amount) * 100 / original_amount = 33 :=
by
  sorry

end johns_percentage_increase_l553_553892


namespace sum_x_midpoints_of_triangle_l553_553503

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l553_553503


namespace bees_dance_paths_l553_553751

-- Define the honeycomb structure and movement rules
inductive CellContent
| zero : CellContent
| one : CellContent
| two : CellContent

-- Define the structure for hexagonal cells
structure HexCell :=
(content : CellContent)
(adj : list HexCell)

-- The main theorem stating the number of distinct paths that form the sequence '2010'
theorem bees_dance_paths (honeycomb : list HexCell) :
  ∃ (paths : ℕ), paths = 30 :=
by sorry

end bees_dance_paths_l553_553751


namespace match_graph_l553_553142

theorem match_graph (x : ℝ) (h : x > 1) : 
  (10^(Real.log10 (x - 1)) = ( (x-1) / Real.sqrt (x-1) )^2) :=
sorry

end match_graph_l553_553142


namespace CD_bisects_QH_l553_553655

open EuclideanGeometry

theorem CD_bisects_QH
  (A B Q H C D : Point ℝ)
  (h1 : A ≠ B)
  (h2 : Circle (A, B) Q)
  (h3 : Q ≠ A ∧ Q ≠ B)
  (h4 : Perpendicular (Line(Q, H)) (Line(A, B)))
  (h5 : OnLine H (Line(A, B)))
  (h6 : CircleIntersection (Circle (A, B)) (Circle(Q, (dist Q H))) C)
  (h7 : CircleIntersection (Circle (A, B)) (Circle(Q, (dist Q H))) D) :
  Midpoint (Segment(C, D)) (H) :=
sorry

end CD_bisects_QH_l553_553655


namespace largest_degree_of_p_proof_l553_553978

noncomputable def largest_degree_of_p (p : ℕ → ℕ) : Prop :=
  ∀ p : polynomial ℚ, (∃ k, polynomial.degree p = k)
    → (∃ n : ℕ, n ≤ 6 ∧ (∀ x : ℚ, (3*x^6 - 2*x^3 + x^2 - 8 ≠ 0)
    → ∀ L : ℚ, (tendsto (λ x, p x / (3*x^6 - 2*x^3 + x^2 - 8)) at_top (nhds L))
    → L = 1 / 3))

theorem largest_degree_of_p_proof : largest_degree_of_p :=
by
  sorry

end largest_degree_of_p_proof_l553_553978


namespace smallest_n_for_T_n_integer_l553_553004

noncomputable def K : ℚ := ∑ i in (Finset.range 10).filter (λ x, x ≠ 0), (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n - 1)) * K + 1

theorem smallest_n_for_T_n_integer : ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m : ℕ, T_n m ∈ ℤ → n ≤ m :=
  ⟨504, sorry⟩

end smallest_n_for_T_n_integer_l553_553004


namespace smallest_n_for_T_n_integer_l553_553000

def L : ℚ := ∑ i in {1, 2, 3, 4}, 1 / i

theorem smallest_n_for_T_n_integer : ∃ n ∈ ℕ, n > 0 ∧ (n * 5^(n-1) * L).denom = 1 ∧ n = 12 :=
by
  have hL : L = 25 / 12 := by sorry
  existsi 12
  split
  exact Nat.succ_pos'
  split
  suffices (12 * 5^(12-1) * 25 / 12).denom = 1 by sorry
  sorry
  rfl

end smallest_n_for_T_n_integer_l553_553000


namespace max_corner_odd_rectangles_5_is_60_l553_553915

noncomputable def max_corner_odd_rectangles (n : ℕ) : ℕ :=
  let pairs := (Finset.range n).choose 2 in
  let max_per_pair := n / 2 * (n - n / 2) in
  pairs.card * max_per_pair

def max_corner_odd_rectangles_5 := max_corner_odd_rectangles 5

theorem max_corner_odd_rectangles_5_is_60 : max_corner_odd_rectangles_5 = 60 := by
  -- specific definitions and constraints
  have pairs := (Finset.range 5).choose 2
  have max_per_pair := 2 * 3
  have total := pairs.card * max_per_pair
  -- assert the condition on total number of rectangle pairs
  have : total = 60
  exact this
  sorry

end max_corner_odd_rectangles_5_is_60_l553_553915


namespace edge_length_RS_correct_l553_553108

noncomputable def length_edge_RS (PQ RS: ℕ) (edges: Set ℕ) : ℕ :=
if h : PQ ∈ edges ∧ RS ∈ edges ∧ (PQ = 45) ∧ {8, 15, 22, 30, 39, 45} = edges then 15 else 0

theorem edge_length_RS_correct (PQ RS: ℕ) (edges: Set ℕ) 
  (hPQ: PQ = 45) 
  (hEdges: edges = {8, 15, 22, 30, 39, 45}) :
  length_edge_RS PQ RS edges = 15 :=
sorry

end edge_length_RS_correct_l553_553108


namespace cupcake_packages_l553_553903

theorem cupcake_packages (total_cupcakes eaten_cupcakes cupcakes_per_package number_of_packages : ℕ) 
  (h1 : total_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : cupcakes_per_package = 2)
  (h4 : number_of_packages = (total_cupcakes - eaten_cupcakes) / cupcakes_per_package) :
  number_of_packages = 5 :=
by
  -- The proof goes here, we'll use sorry to indicate it's not needed for now.
  sorry

end cupcake_packages_l553_553903


namespace max_value_trig_expr_exists_angle_for_max_value_l553_553235

theorem max_value_trig_expr : ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 :=
sorry

theorem exists_angle_for_max_value : ∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5 :=
sorry

end max_value_trig_expr_exists_angle_for_max_value_l553_553235


namespace subset_with_distance_l553_553408

open Set

/-- A set of points S with at least n points and mutual distances of at least 1 unit, 
    contains a subset T with at least ⌊n / 7⌋ points where the mutual distances are 
    at least √3 units. -/
theorem subset_with_distance {n : ℕ} (S : Finset (ℝ × ℝ)) 
  (hS : S.card = n)
  (h_dist : ∀ (x y ∈ S), x ≠ y → dist x y ≥ 1) :
  ∃ T : Finset (ℝ × ℝ), T ⊆ S ∧ T.card ≥ n / 7 ∧ 
    ∀ (x y ∈ T), x ≠ y → dist x y ≥ sqrt 3 := 
sorry

end subset_with_distance_l553_553408


namespace find_f_zero_function_decreasing_find_range_x_l553_553300

noncomputable def f : ℝ → ℝ := sorry

-- Define the main conditions as hypotheses
axiom additivity : ∀ x1 x2 : ℝ, f (x1 + x2) = f x1 + f x2
axiom negativity : ∀ x : ℝ, x > 0 → f x < 0

-- First theorem: proving f(0) = 0
theorem find_f_zero : f 0 = 0 := sorry

-- Second theorem: proving the function is decreasing over (-∞, ∞)
theorem function_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := sorry

-- Third theorem: finding the range of x such that f(x) + f(2-3x) < 0
theorem find_range_x (x : ℝ) : f x + f (2 - 3 * x) < 0 → x < 1 := sorry

end find_f_zero_function_decreasing_find_range_x_l553_553300


namespace same_functions_l553_553597

open Function

theorem same_functions :
  (∀ x : ℝ, (λ x, x) x = (λ x, real.cbrt (x ^ 3)) x) ∧
  (∀ x, x ≠ 0 → (λ x, ∣x∣ / x) x = (λ x, if x > 0 then 1 else -1) x) ∧
  (∀ x : ℝ, (λ t, ∣t - 1∣) x = (λ x, ∣x - 1∣) x) :=
  by
    simp
    sorry

end same_functions_l553_553597


namespace angles_in_arithmetic_progression_of_triangle_and_incircle_l553_553381

open EuclideanGeometry

-- Define conditions: A triangle and an incircle tangent to its sides at points D, E, F.
variables (A B C D E F : Point)
variable [InCircle (Triangle.mk A B C) D E F]
variables (∠CAB ∠ABC ∠BCA ∠FDE ∠DEF ∠EFD : Angle)

-- Main theorem statement
theorem angles_in_arithmetic_progression_of_triangle_and_incircle
  (h1 : (∠ CAB, ∠ ABC, ∠ BCA) ∈ ArithmeticProgression) :
  (∠ FDE, ∠ DEF, ∠ EFD) ∈ ArithmeticProgression :=
by
  sorry

end angles_in_arithmetic_progression_of_triangle_and_incircle_l553_553381


namespace floor_area_ring_l553_553989

noncomputable def circle_radius_40 := (40 : ℝ)

def inside_radius (s : ℝ) := s * (1 + Real.sqrt 2)

def area (s : ℝ) : ℝ :=
  let larger_circle_area := Real.pi * circle_radius_40^2
  let small_circle_area := 8 * Real.pi * s^2
  larger_circle_area - small_circle_area

theorem floor_area_ring (s : ℝ) (h_s : s * (1 + Real.sqrt 2) = circle_radius_40) : 
  Int.floor (area 40 (Real.sqrt 2 - 1)) = 1161 :=
sorry

end floor_area_ring_l553_553989


namespace find_f_when_fa_equals_negative_three_l553_553698

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2^x - 2 else -Real.log 2 (x + 1)

theorem find_f_when_fa_equals_negative_three :
  (∃ a : ℝ, f a = -3) → f (5 - 7) = -7 / 4 :=
by
  intro h
  -- Since this is a skeleton, we skip the detailed proof steps
  existsi 7
  simp [f]
  sorry

end find_f_when_fa_equals_negative_three_l553_553698


namespace fish_tank_problem_l553_553769

def number_of_fish_in_first_tank
  (F : ℕ)          -- Let F represent the number of fish in the first tank
  (twoF : ℕ)       -- Let twoF represent twice the number of fish in the first tank
  (total : ℕ) :    -- Let total represent the total number of fish
  Prop :=
  (2 * F = twoF)  -- The other two tanks each have twice as many fish as the first
  ∧ (F + twoF + twoF = total)  -- The sum of the fish in all three tanks equals the total number of fish

theorem fish_tank_problem
  (F : ℕ)
  (H : number_of_fish_in_first_tank F (2 * F) 100) : F = 20 :=
by
  sorry

end fish_tank_problem_l553_553769


namespace simplify_fraction_l553_553457

theorem simplify_fraction :
  (1 / 462) + (17 / 42) = 94 / 231 := sorry

end simplify_fraction_l553_553457


namespace factor_poly_l553_553728

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l553_553728


namespace man_speed_still_water_l553_553172

noncomputable def speed_in_still_water (U D : ℝ) : ℝ := (U + D) / 2

theorem man_speed_still_water :
  let U := 45
  let D := 55
  speed_in_still_water U D = 50 := by
  sorry

end man_speed_still_water_l553_553172


namespace exists_point_with_120_deg_angles_l553_553191

theorem exists_point_with_120_deg_angles (ABC : Type) [Inhabited ABC] 
  (A B C : ABC) (angle_A angle_B angle_C : ℝ) 
  (hA : angle_A < 120) (hB : angle_B < 120) (hC : angle_C < 120) : 
  ∃ P : ABC, ∠APB = 120 ∧ ∠APC = 120 ∧ ∠BPC = 120 := 
sorry

end exists_point_with_120_deg_angles_l553_553191


namespace ratio_of_terms_l553_553679

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, S n = n * (2 * a 0 + (n - 1) * d) / 2

theorem ratio_of_terms
  (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ)
  (h_seq : arithmetic_sequence a d)
  (h_sum : sum_of_first_n_terms S a d)
  (h_ratio : S 5 = 2 * S 3) :
  a 5 * 3 = a 3 * 4 :=
begin
  sorry
end

end ratio_of_terms_l553_553679


namespace steven_needs_more_seeds_l553_553082

theorem steven_needs_more_seeds :
  let total_seeds_needed := 60
  let seeds_per_apple := 6
  let seeds_per_pear := 2
  let seeds_per_grape := 3
  let apples_collected := 4
  let pears_collected := 3
  let grapes_collected := 9
  total_seeds_needed - (apples_collected * seeds_per_apple + pears_collected * seeds_per_pear + grapes_collected * seeds_per_grape) = 3 :=
by
  sorry

end steven_needs_more_seeds_l553_553082


namespace alcohol_percentage_after_adding_water_l553_553558

def original_mixture_volume : ℝ := 28
def alcohol_percentage : ℝ := 35 / 100
def water_added : ℝ := 6

def amount_of_alcohol (volume : ℝ) (percentage : ℝ) : ℝ := volume * percentage
def amount_of_water (volume : ℝ) (alcohol_volume : ℝ) : ℝ := volume - alcohol_volume
def new_water_amount (original_water : ℝ) (added_water : ℝ) : ℝ := original_water + added_water
def new_mixture_volume (alcohol_volume : ℝ) (new_water : ℝ) : ℝ := alcohol_volume + new_water
def alcohol_percentage_in_new_mixture (alcohol_volume : ℝ) (new_volume : ℝ) : ℝ := (alcohol_volume / new_volume) * 100

theorem alcohol_percentage_after_adding_water :
  let alcohol_volume := amount_of_alcohol original_mixture_volume alcohol_percentage in
  let water_volume := amount_of_water original_mixture_volume alcohol_volume in
  let new_water := new_water_amount water_volume water_added in
  let total_volume := new_mixture_volume alcohol_volume new_water in
  alcohol_percentage_in_new_mixture alcohol_volume total_volume = 28.82 :=
by
  sorry

end alcohol_percentage_after_adding_water_l553_553558


namespace hamiltonian_circuit_theorem_l553_553737

variables {G : Type*} [graph G]

def vertex_count (G : graph G) : ℕ := sorry  -- Assumed to be a function that returns the number of vertices in G
def independence_number (G : graph G) : ℕ := sorry  -- Assumed to be a function that returns the independence number of G
def vertex_connectivity (G : graph G) : ℕ := sorry  -- Assumed to be a function that returns the vertex connectivity of G
def has_hamiltonian_circuit (G : graph G) : Prop := sorry  -- Predicate stating whether G has a Hamiltonian circuit

theorem hamiltonian_circuit_theorem 
  (h1 : vertex_count G ≥ 3)
  (h2 : independence_number G ≤ vertex_connectivity G) : 
  has_hamiltonian_circuit G := 
sorry

end hamiltonian_circuit_theorem_l553_553737


namespace wealth_change_proof_l553_553797

-- Define the initial conditions
def initial_cash_A := 15000
def car_value := 5000
def initial_cash_B := 20000
def initial_house_value := 15000

-- Define the transactions
def car_sale_price := 6000
def house_sale_price_A_to_B := 18000
def house_sale_price_B_to_A := 20000

-- Define the appreciation rate
def appreciation_rate := 0.10

-- Calculate appreciation
def appreciated_house_value := initial_house_value * (1 + appreciation_rate)

-- Calculate final amounts after transactions
def final_cash_A := (initial_cash_A + car_sale_price - house_sale_price_A_to_B) + house_sale_price_B_to_A
def final_cash_B := (initial_cash_B - car_sale_price + house_sale_price_A_to_B) - house_sale_price_B_to_A
def final_assets_B := 12000 + car_value + appreciated_house_value

-- Net change in wealth
def net_change_A := final_cash_A - (initial_cash_A + car_value)
def net_change_B := final_assets_B - initial_cash_B

-- Lean statement to prove
theorem wealth_change_proof :
  net_change_A = 3000 ∧ net_change_B = 17000 :=
by
  sorry  -- proof goes here

end wealth_change_proof_l553_553797


namespace distinct_weights_handshakes_l553_553651

theorem distinct_weights_handshakes (w : Fin 4 → ℝ) (h_distinct : Function.Injective w) :
        ∑ i, i = 6 :=
by
  sorry

end distinct_weights_handshakes_l553_553651


namespace katie_needs_more_sugar_l553_553176

-- Let total_cups be the total cups of sugar required according to the recipe
def total_cups : ℝ := 3

-- Let already_put_in be the cups of sugar Katie has already put in
def already_put_in : ℝ := 0.5

-- Define the amount of sugar Katie still needs to put in
def remaining_cups : ℝ := total_cups - already_put_in 

-- Prove that remaining_cups is 2.5
theorem katie_needs_more_sugar : remaining_cups = 2.5 := 
by 
  -- substitute total_cups and already_put_in
  dsimp [remaining_cups, total_cups, already_put_in]
  -- calculate the difference
  norm_num

end katie_needs_more_sugar_l553_553176


namespace wire_ratio_l553_553186

theorem wire_ratio (a b : ℝ) (h_eq_area : (a / 4)^2 = 2 * (b / 8)^2 * (1 + Real.sqrt 2)) :
  a / b = Real.sqrt (2 + Real.sqrt 2) / 2 :=
by
  sorry

end wire_ratio_l553_553186


namespace vector_and_magnitude_correct_l553_553360

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 3)

-- Define the vector AB
def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of the vector AB
def magnitude_vector_AB : ℝ := real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2)

-- The theorem that encapsulates the problem setup and the expected result
theorem vector_and_magnitude_correct :
  vector_AB = (1, 1) ∧ magnitude_vector_AB = real.sqrt 2 :=
by
  sorry

end vector_and_magnitude_correct_l553_553360


namespace tan_420_eq_sqrt3_l553_553878

theorem tan_420_eq_sqrt3 : Real.tan (420 * Real.pi / 180) = Real.sqrt 3 := 
by 
  -- Additional mathematical justification can go here.
  sorry

end tan_420_eq_sqrt3_l553_553878


namespace frog_climbing_time_l553_553570

-- Definition of the problem conditions
def well_depth := 12  -- meters
def climb_up := 3     -- meters
def slip_back := 1    -- meters
def slip_time_fraction := 1 / 3 -- fraction of climbing time required to slip

-- Define the proposition to prove that the frog reaches the top in 22 minutes
theorem frog_climbing_time
  (well_depth : ℕ) 
  (climb_up : ℕ) 
  (slip_back : ℕ) 
  (slip_time_fraction : ℚ)
  (second_reach : ℕ := 17) 
  (second_reach_depth : ℕ := 9) : 
  (frog_time_to_top well_depth climb_up slip_back slip_time_fraction = 22) :=
by
  sorry

end frog_climbing_time_l553_553570


namespace cube_volume_of_surface_area_l553_553565

theorem cube_volume_of_surface_area (S : ℝ) (V : ℝ) (a : ℝ) (h1 : S = 150) (h2 : S = 6 * a^2) (h3 : V = a^3) : V = 125 := by
  sorry

end cube_volume_of_surface_area_l553_553565


namespace aaron_final_position_l553_553187

def initial_position : ℕ × ℕ := (0, 0)

def movement_rules (dir : ℕ) (p: ℕ × ℕ) (steps: ℕ) : ℕ × ℕ :=
  if dir == 0 then (p.1 + steps, p.2)
  else if dir == 1 then (p.1, p.2 + steps)
  else if dir == 2 then (p.1 - steps, p.2)
  else (p.1, p.2 - steps)
  
def update_position (n : ℕ) : ℕ × ℕ :=
  let rec move (steps_taken steps dir: ℕ) (pos: ℕ × ℕ) : ℕ × ℕ :=
    if steps_taken == 100 then pos
    else
      let steps_to_move := if steps_taken % 4 == 0 then steps + 2 else steps
      let new_pos := movement_rules dir pos steps_to_move
      move (steps_taken + 1) steps_to_move ((dir + 1) % 4) new_pos
  move 0 0 0 initial_position

theorem aaron_final_position : update_position 100 = (10, 0) := sorry

end aaron_final_position_l553_553187


namespace geometry_theorem_1_geometry_theorem_2_l553_553889

theorem geometry_theorem_1
  (ABC : Triangle)
  (A1 B1 C1 : Point) 
  (S : Point)
  (Brocard_Triangle : is_Brocard_Triangle ABC A1 B1 C1)
  (LineThroughA : Line) (LineThroughB : Line) (LineThroughC : Line)
  (ParallelA : are_parallel LineThroughA (line_through B1 C1))
  (ParallelB : are_parallel LineThroughB (line_through A1 C1))
  (ParallelC : are_parallel LineThroughC (line_through A1 B1))
  (IntersectS : are_concurrent LineThroughA LineThroughB LineThroughC S)
  :
  lies_on_circumcircle S ABC :=
by
  sorry

theorem geometry_theorem_2
  (ABC : Triangle)
  (S : Point)
  (Steiner_Point : is_Steiner_Point ABC S)
  :
  is_parallel_to_Simson_Line_S_Brocard_Diameter ABC S :=
by
  sorry

end geometry_theorem_1_geometry_theorem_2_l553_553889


namespace frac_y_over_x_plus_y_eq_one_third_l553_553725

theorem frac_y_over_x_plus_y_eq_one_third (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end frac_y_over_x_plus_y_eq_one_third_l553_553725


namespace triangle_inequality_l553_553392

variable (G : Type) [SimpleGraph G]
variable (n m T : ℕ)
variable [Fintype G]
variable [DecidableRel G.Adj]

-- We introduce the assumptions as hypothesis
hypothesis (hv : Fintype.card (Vertex G) = n)
hypothesis (he : Fintype.card (Edge G) = m)
hypothesis (ht : triangle G = T)

-- Now we state the theorem to prove the inequality
theorem triangle_inequality (hv : Fintype.card (Vertex G) = n)
                             (he : Fintype.card (Edge G) = m)
                             (ht : triangle G = T) :
                             T ≥ (m * (4 * m - n^2)) / (3 * n) := 
by sorry

end triangle_inequality_l553_553392


namespace ice_cream_cost_correct_l553_553890

variable (cost_chapati cost_rice cost_mixed_veg cost_ice_cream : ℕ)
variables (num_chapatis num_plates_rice num_plates_mixed_veg num_ice_creams : ℕ)
variable (total_paid : ℕ)

-- defining the given constants and conditions
def chapati_cost : ℕ := 6
def rice_cost : ℕ := 45
def mixed_veg_cost : ℕ := 70
def total_payment : ℕ := 961

noncomputable def cost_per_ice_cream : ℕ :=
  (total_payment - (num_chapatis * chapati_cost + num_plates_rice * rice_cost + num_plates_mixed_veg * mixed_veg_cost)) / num_ice_creams

theorem ice_cream_cost_correct :
  ∀ (cost_chapati cost_rice cost_mixed_veg cost_ice_cream : ℕ)
    (num_chapatis num_plates_rice num_plates_mixed_veg num_ice_creams : ℕ)
    (total_paid : ℕ),
  cost_chapati = 6 →
  cost_rice = 45 →
  cost_mixed_veg = 70 →
  num_chapatis = 16 →
  num_plates_rice = 5 →
  num_plates_mixed_veg = 7 →
  num_ice_creams = 6 →
  total_paid = 961 →
  cost_per_ice_cream total_paid num_chapatis num_plates_rice num_plates_mixed_veg num_ice_creams = 25 :=
by {
  intros cost_chapati cost_rice cost_mixed_veg cost_ice_cream
        num_chapatis num_plates_rice num_plates_mixed_veg num_ice_creams total_paid
        h1 h2 h3 h4 h5 h6 h7 h8,
  dsimp [chapati_cost, rice_cost, mixed_veg_cost, total_payment, cost_per_ice_cream],
  rw [h1, h2, h3, h4, h5, h6, h7, h8],
  exact rfl,
}

end ice_cream_cost_correct_l553_553890


namespace twice_x_minus_3_gt_4_l553_553227

theorem twice_x_minus_3_gt_4 (x : ℝ) : 2 * x - 3 > 4 :=
sorry

end twice_x_minus_3_gt_4_l553_553227


namespace min_omega_shift_left_l553_553790

theorem min_omega_shift_left (ω : ℝ) (hω : ω > 0)
  (h_shift : ∀ x : ℝ, sin (ω * x) = sin (ω * (x + π / 3))) : ω = 6 :=
by
  sorry

end min_omega_shift_left_l553_553790


namespace equivalence_sum_l553_553605

theorem equivalence_sum :
  2^4 ≡ -1 [MOD 17] →
  (2^(-2 : ℤ) + 2^(-3 : ℤ) + 2^(-4 : ℤ) + 2^(-5 : ℤ) + 2^(-6 : ℤ) + 2^(-7 : ℤ)) % 17 = 10 :=
by
  intro h
  sorry

end equivalence_sum_l553_553605


namespace arithmetic_sequence_difference_l553_553600

noncomputable def M := 60 - 75 * (40 / 149)
noncomputable def N := 60 + 75 * (40 / 149)
noncomputable def seq_sum : ℕ → ℚ → ℚ := λ n d,∑ i in range (2 * n - 1), (60 : ℚ) + (i - (n - 1)) * d

theorem arithmetic_sequence_difference 
  (n : ℕ)
  (d : ℚ)
  (h1 : n = 150)
  (h2 : seq_sum n d = 9000)
  (h3 : ∀ (i : ℕ), i < 2 * n - 1 → 20 ≤ 60 + (i - (n - 1)) * d ∧ 60 + (i - (n - 1)) * d ≤ 80) :
  N - M = 6000 / 149 :=
by sorry

end arithmetic_sequence_difference_l553_553600


namespace a2018_is_4035_l553_553286

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) : ℝ := sorry

axiom domain : ∀ x : ℝ, true 
axiom condition_2 : ∀ x : ℝ, x < 0 → f x > 1
axiom condition_3 : ∀ x y : ℝ, f x * f y = f (x + y)
axiom sequence_def : ∀ n : ℕ, n > 0 → a 1 = f 0 ∧ f (a (n + 1)) = 1 / f (-2 - a n)

theorem a2018_is_4035 : a 2018 = 4035 :=
sorry

end a2018_is_4035_l553_553286


namespace number_of_valid_integers_l553_553718

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_between_300_and_900 (n : ℕ) : Prop :=
  300 ≤ n ∧ n < 900

def all_digits_different (n : ℕ) : Prop :=
  let digits := (to_string n).to_list in
  digits.nodup

def digits_from_set (n : ℕ) (s : set ℕ) : Prop :=
  (to_string n).to_list.all (λ c, c.to_nat ∈ s)

noncomputable def count_valid_integers (s : set ℕ) : ℕ :=
  (finset.range 900).filter (λ n,
    is_even n ∧
    is_between_300_and_900 n ∧
    all_digits_different n ∧
    digits_from_set n s).card

theorem number_of_valid_integers :
  count_valid_integers {1, 3, 4, 6, 7, 9} = 16 :=
begin
  sorry
end

end number_of_valid_integers_l553_553718


namespace percentNonUnionWomen_proof_l553_553540

variable (totalEmployees : ℕ)
variable (percentMen : ℕ) (percentUnionized : ℕ) (percentUnionizedMen : ℕ)
variable (percentNonUnionWomen : ℕ)

def calcPercentNonUnionWomen 
  (totalEmployees : ℕ) 
  (percentMen : ℕ) 
  (percentUnionized : ℕ) 
  (percentUnionizedMen : ℕ) : ℕ :=
let totalMen := (percentMen * totalEmployees) / 100 in
let unionizedEmployees := (percentUnionized * totalEmployees) / 100 in
let unionizedMen := (percentUnionizedMen * unionizedEmployees) / 100 in
let nonUnionMen := totalMen - unionizedMen in
let nonUnionEmployees := totalEmployees - unionizedEmployees in
let nonUnionWomen := nonUnionEmployees - nonUnionMen in
(nonUnionWomen * 100) / nonUnionEmployees

theorem percentNonUnionWomen_proof :
  percentMen = 48 → 
  percentUnionized = 60 →
  percentUnionizedMen = 70 →
  totalEmployees = 100 →
  calcPercentNonUnionWomen totalEmployees percentMen percentUnionized percentUnionizedMen = 85 :=
by
  intros hMen hUnion hUnionMen hTotal
  unfold calcPercentNonUnionWomen
  sorry

end percentNonUnionWomen_proof_l553_553540


namespace find_equation_C_prove_collinearity_l553_553691

-- Condition: Distance from P to M(4,0) is twice the distance to N(1,0)
def point (a b : ℝ) := (a, b)

def P_distance_M (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - 4)^2 + P.2^2)

def P_distance_N (P : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - 1)^2 + P.2^2)

def distance_condition (P : ℝ × ℝ) : Prop := P_distance_M P = 2 * P_distance_N P

-- Condition: Intersection points A and B of curve C with x-axis
def A := point (-2 : ℝ) 0
def B := point (2 : ℝ) 0

-- Condition: Q is any point on the line x = 1
def on_line_l (Q : ℝ × ℝ) : Prop := Q.1 = 1

-- Endpoints intersection with C
def intersects_c (Q D E: ℝ × ℝ) : Prop := 
  (D.1 + 2) * (Q.2 - D.2) = Q.1 * (D.2 - Q.2) ∧
  (E.1 - 2) * (Q.2 - E.2) = Q.1 * (E.2 - Q.2)

-- Straight line 
def collinear (M D E: ℝ × ℝ) : Prop :=
(M.2 - D.2)*(E.1 - D.1) =  (M.1 - D.1)*(E.2 - D.2)

-- Correct answer for the equation of curve C
theorem find_equation_C (P : ℝ × ℝ) (h: distance_condition P) : P.1^2 + P.2^2 = 4 :=
sorry

-- Prove collinearity of points M, D, and E under given conditions
theorem prove_collinearity (D E Q : ℝ × ℝ) (hQ: on_line_l Q)
(hC: (D.1 + 2) * (Q.2 - D.2) = Q.1 * (D.2 - Q.2) ∧ (E.1 - 2) * (Q.2 - E.2) = Q.1 * (E.2 - Q.2)) : collinear (point 4 0) D E := 
sorry

end find_equation_C_prove_collinearity_l553_553691


namespace total_cost_textbooks_l553_553415

theorem total_cost_textbooks :
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  sale_books + online_books + bookstore_books = 210 :=
by
  let sale_books := 5 * 10
  let online_books := 40
  let bookstore_books := 3 * 40
  show sale_books + online_books + bookstore_books = 210
  sorry

end total_cost_textbooks_l553_553415


namespace distance_between_closest_points_l553_553608

-- Declare the centers and conditions
def center1 := (1 : ℝ, 1 : ℝ)
def center2 := (20 : ℝ, 5 : ℝ)

-- Declare the radius conditions
def radius1 := center1.2  -- y-coordinate of center1
def radius2 := center2.2  -- y-coordinate of center2

-- Distance formula between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The main statement to prove
theorem distance_between_closest_points :
  distance center1 center2 - (radius1 + radius2) = Real.sqrt 377 - 6 := 
by sorry

end distance_between_closest_points_l553_553608


namespace lin_reg_proof_l553_553580

variable (x y : List ℝ)
variable (n : ℝ := 10)
variable (sum_x : ℝ := 80)
variable (sum_y : ℝ := 20)
variable (sum_xy : ℝ := 184)
variable (sum_x2 : ℝ := 720)

noncomputable def mean (lst: List ℝ) (n: ℝ) : ℝ := (List.sum lst) / n

noncomputable def lin_reg_slope (n sum_x sum_y sum_xy sum_x2 : ℝ) : ℝ :=
  (sum_xy - n * (sum_x / n) * (sum_y / n)) / (sum_x2 - n * (sum_x / n) ^ 2)

noncomputable def lin_reg_intercept (sum_x sum_y : ℝ) (slope : ℝ) (n : ℝ) : ℝ :=
  (sum_y / n) - slope * (sum_x / n)

theorem lin_reg_proof :
  lin_reg_slope n sum_x sum_y sum_xy sum_x2 = 0.3 ∧ 
  lin_reg_intercept sum_x sum_y 0.3 n = -0.4 ∧ 
  (0.3 * 7 - 0.4 = 1.7) :=
by
  sorry

end lin_reg_proof_l553_553580


namespace angle_60_degrees_l553_553312

def vec : Type := (ℤ × ℤ × ℤ)

def a : vec := (1, 0, -1)
def b : vec := (1, -1, 0)

noncomputable def dot_product (v1 v2 : vec) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def norm (v : vec) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem angle_60_degrees (a b : vec) :
  dot_product a b = (norm a) * (norm b) * (1/2) :=
sorry

end angle_60_degrees_l553_553312


namespace distance_focus_to_asymptote_l553_553359

noncomputable def focus_of_parabola : (ℝ × ℝ) :=
  (0, 2)

noncomputable def asymptotes_of_hyperbola : ℝ → (ℝ × ℝ) :=
  λ x, (x, 3 * x)

noncomputable def distance_to_asymptote_from_focus : ℝ :=
  abs 2 / real.sqrt (3^2 + 1^2)

theorem distance_focus_to_asymptote :
  distance_to_asymptote_from_focus = real.sqrt 10 / 5 :=
by sorry

end distance_focus_to_asymptote_l553_553359


namespace monotone_decreasing_interval_of_ln4plus3x_minus_x2_l553_553222

noncomputable def f (x : ℝ) : ℝ := real.log (4 + 3 * x - x^2)

def u (x : ℝ) : ℝ := -x^2 + 3 * x + 4

theorem monotone_decreasing_interval_of_ln4plus3x_minus_x2 :
  ∀ x ∈ Icc (3 / 2) 4, ∀ y ∈ Icc (3 / 2) 4, x < y → f y ≤ f x :=
by
  intros x hx y hy hxy
  have hu : ∀ x ∈ Icc (3 / 2) 4, ∀ y ∈ Icc (3 / 2) 4, x < y → u y < u x := sorry
  have hln : ∀ x ∈ Icc (3 / 2) 4, ∀ y ∈ Icc (3 / 2) 4, x < y → f y ≤ f x := 
    by intros x hx y hy hxy
       rw [f, real.log_le_log_of_le (lt_of_le_of_lt hx (by norm_num)) (lt_of_le_of_lt hy (by norm_num))]
       exact hu x hx y hy hxy
  exact hln x hx y hy hxy

end monotone_decreasing_interval_of_ln4plus3x_minus_x2_l553_553222


namespace lambda_range_l553_553307

def a_n (n : ℕ) (λ : ℝ) : ℝ :=
  n^2 - (6 + 2 * λ) * n + 2016

theorem lambda_range (λ : ℝ) :
  (Is_Minimum_Term : ∃ n, a_n n λ ≤ a_n m λ ∀ m) →
  \(\frac{5}{2} < λ < \frac{9}{2}\) :=
by
  sorry

end lambda_range_l553_553307


namespace baby_polar_bear_playing_hours_l553_553753

-- Define the conditions
def total_hours_in_a_day : ℕ := 24
def total_central_angle : ℕ := 360
def angle_sleeping : ℕ := 130
def angle_eating : ℕ := 110

-- Main theorem statement
theorem baby_polar_bear_playing_hours :
  let angle_playing := total_central_angle - angle_sleeping - angle_eating
  let fraction_playing := angle_playing / total_central_angle
  let hours_playing := fraction_playing * total_hours_in_a_day
  hours_playing = 8 := by
  sorry

end baby_polar_bear_playing_hours_l553_553753


namespace largest_circle_radius_l553_553214

-- Define the sides of the quadrilateral
def side_a := 13
def side_b := 10
def side_c := 8
def side_d := 11

-- Lean theorem statement to be proved
theorem largest_circle_radius (a b c d : ℕ) (h_a : a = side_a) (h_b : b = side_b) (h_c : c = side_c) (h_d : d = side_d) : 
  (2 * Real.sqrt 6) = radius_of_largest_circle a b c d := 
by 
  sorry

-- radius_of_largest_circle definition placeholder
noncomputable def radius_of_largest_circle (a b c d : ℕ) : ℝ := 
  if (a = side_a ∧ b = side_b ∧ c = side_c ∧ d = side_d) then 2 * Real.sqrt 6 else 0

end largest_circle_radius_l553_553214


namespace mutually_exclusive_opposite_AC_BD_l553_553748

-- Definitions for events and their probabilities
variable (Ω : Type) [fintype Ω] [decidable_eq Ω]

noncomputable def P (s : set Ω) : ℝ := sorry  -- A placeholder for probability measure

-- Events A, B, C, D
variable (A B C D : set Ω)
variable (P_A : P A = 0.2)
variable (P_B : P B = 0.2)
variable (P_C : P C = 0.3)
variable (P_D : P D = 0.3)

-- Definition of mutually exclusive events
def mutually_exclusive (X Y : set Ω) : Prop := X ∩ Y = ∅

-- Definition of opposite events
def opposite (X Y : set Ω) : Prop := X ∪ Y = univ

-- Theorem to prove
theorem mutually_exclusive_opposite_AC_BD
  (h1 : mutually_exclusive A B) 
  (h2 : mutually_exclusive A C)
  (h3 : mutually_exclusive A D)
  (h4 : mutually_exclusive B C)
  (h5 : mutually_exclusive B D)
  (h6 : mutually_exclusive C D)
  : mutually_exclusive (A ∪ C) (B ∪ D) ∧ opposite (A ∪ C) (B ∪ D) :=
sorry

end mutually_exclusive_opposite_AC_BD_l553_553748


namespace arctan_gt_arcsin_l553_553636

noncomputable theory

open Real

theorem arctan_gt_arcsin (x : ℝ) : (arctan x > arcsin x) ↔ x ∈ Ioo 0 1 := 
sorry

end arctan_gt_arcsin_l553_553636


namespace sin_cos_mul_sin_minus_cos_l553_553252

variable {x : ℝ}
axiom h1 : -π / 2 < x
axiom h2 : x < 0
axiom h3 : sin x + cos x = 1 / 5

theorem sin_cos_mul (h1 : -π / 2 < x) (h2 : x < 0) (h3 : sin x + cos x = 1 / 5) : 
  sin x * cos x = -12 / 25 := 
sorry

theorem sin_minus_cos (h1 : -π / 2 < x) (h2 : x < 0) (h3 : sin x + cos x = 1 / 5) : 
  sin x - cos x = -7 / 5 := 
sorry

end sin_cos_mul_sin_minus_cos_l553_553252


namespace systematic_sampling_method_l553_553942

theorem systematic_sampling_method (k : ℕ) (n : ℕ) 
  (invoice_stubs : ℕ → ℕ) : 
  (k > 0) → 
  (n > 0) → 
  (invoice_stubs 15 = k) → 
  (∀ i : ℕ, invoice_stubs (15 + i * 50) = k + i * 50)
  → (sampling_method = "systematic") :=
by 
  intro h1 h2 h3 h4
  sorry

end systematic_sampling_method_l553_553942


namespace rectangular_plot_area_l553_553833

theorem rectangular_plot_area (breadth length : ℕ) (h1 : breadth = 14) (h2 : length = 3 * breadth) : (length * breadth) = 588 := 
by 
  -- imports, noncomputable keyword, and placeholder proof for compilation
  sorry

end rectangular_plot_area_l553_553833


namespace measure_angle_CAB_in_hexagon_l553_553749

-- Define the problem conditions and goal.
theorem measure_angle_CAB_in_hexagon 
  (ABCDEF_regular: ∀ (AB BC CD DE EF FA: ℝ), AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB)
  (interior_angle_120: ∀ (x y z w v u: ℝ), x+y+z+w+v+u = 720)
  (interior_angle_A: ∀ (angle: ℝ), angle = 120 )
  : ∃ (CAB:ℝ), CAB = 30 := 
sorry

end measure_angle_CAB_in_hexagon_l553_553749


namespace find_m_l553_553277

noncomputable def log10 := Real.log10

theorem find_m (x m : ℝ)
  (h1 : log10 (Real.sin x) + log10 (Real.cos x) = -2)
  (h2 : log10 (Real.sin x + Real.cos x) = (1/2) * (log10 m - 2)) :
  m = 102 :=
sorry

end find_m_l553_553277


namespace average_salary_proof_l553_553823

noncomputable def average_salary_of_all_workers (tech_workers : ℕ) (tech_avg_sal : ℕ) (total_workers : ℕ) (non_tech_avg_sal : ℕ) : ℕ :=
  let non_tech_workers := total_workers - tech_workers
  let total_tech_salary := tech_workers * tech_avg_sal
  let total_non_tech_salary := non_tech_workers * non_tech_avg_sal
  let total_salary := total_tech_salary + total_non_tech_salary
  total_salary / total_workers

theorem average_salary_proof : average_salary_of_all_workers 7 14000 28 6000 = 8000 := by
  sorry

end average_salary_proof_l553_553823


namespace analytical_expression_of_odd_function_l553_553692

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem analytical_expression_of_odd_function {a b : ℝ} :
  (∀ x : ℝ, f x a b = -f (-x) a b) ∧ 
  (∀ x : ℝ, x ∈ set.Ioo (-1) 1) ∧ 
  (f 0 a b = 0) → 
  (f = λ x : ℝ, x) :=
by
  sorry

end analytical_expression_of_odd_function_l553_553692


namespace total_apples_remaining_l553_553043

def tree_A_initial := 200
def tree_B_initial := 250
def tree_C_initial := 300

def day1_tree_A_picked := tree_A_initial / 5
def day1_tree_B_picked := tree_B_initial / 10

def day2_tree_B_picked := 2 * day1_tree_A_picked
def day2_tree_C_picked := (tree_C_initial / 8).ceil

def day3_tree_A_picked := (20 + day1_tree_A_picked)
def day3_tree_C_picked := tree_A_initial - day1_tree_A_picked
def day3_tree_C_picked_final := day3_tree_C_picked / 4

theorem total_apples_remaining : 
  let remaining_A := tree_A_initial - (day1_tree_A_picked + day3_tree_A_picked)
  let remaining_B := tree_B_initial - (day1_tree_B_picked + day2_tree_B_picked)
  let remaining_C := tree_C_initial - (day2_tree_C_picked + day3_tree_C_picked_final)
  remaining_A + remaining_B + remaining_C = 467 :=
by
  sorry

end total_apples_remaining_l553_553043


namespace points_symmetric_about_y_eq_x_l553_553040

theorem points_symmetric_about_y_eq_x (x y r : ℝ) :
  (x^2 + y^2 ≤ r^2 ∧ x + y > 0) →
  (∃ p q : ℝ, (q = p ∧ p + q = 0) ∨ (p = q ∧ q = -p)) :=
sorry

end points_symmetric_about_y_eq_x_l553_553040


namespace rhombus_perimeter_l553_553826

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  4 * side = 20 :=
by
  sorry

end rhombus_perimeter_l553_553826


namespace strips_overlap_area_l553_553937

theorem strips_overlap_area (L1 L2 AL AR S : ℝ) (hL1 : L1 = 9) (hL2 : L2 = 7) (hAL : AL = 27) (hAR : AR = 18) 
    (hrel : (AL + S) / (AR + S) = L1 / L2) : S = 13.5 := 
by
  sorry

end strips_overlap_area_l553_553937


namespace shaded_triangle_area_l553_553230

theorem shaded_triangle_area :
  let line1 := (λ x : ℝ, -3 / 10 * x + 5)
  let line2 := (λ x : ℝ, if x < 2 then 6 else -6 / 7 * x + 54 / 7)
  let intersection := (665 / 56, 455 / 56)
  let base := 665 / 56
  let height := 455 / 56
  (1 / 2) * base * height = 151425 / 3136 :=
by {
  sorry
}

end shaded_triangle_area_l553_553230


namespace apps_minus_files_eq_seven_l553_553973

-- Definitions based on conditions
def initial_apps := 24
def initial_files := 9
def deleted_apps := initial_apps - 12
def deleted_files := initial_files - 5

-- Definitions based on the question and correct answer
def apps_left := 12
def files_left := 5

theorem apps_minus_files_eq_seven : apps_left - files_left = 7 := by
  sorry

end apps_minus_files_eq_seven_l553_553973


namespace knitting_five_pairs_l553_553560

noncomputable def knitting_team (d_a d_b d_c: ℕ) : ℚ :=
  let knit_rate : ℚ := (1/d_a + 1/d_b + 1/d_c) in
  5 / knit_rate

theorem knitting_five_pairs :
  knitting_team 3 6 9 = 90 / 11 :=
by
  sorry

end knitting_five_pairs_l553_553560


namespace simplify_fraction_l553_553444

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l553_553444


namespace base_8_to_base_10_2671_to_1465_l553_553568

theorem base_8_to_base_10_2671_to_1465 :
  (2 * 8^3 + 6 * 8^2 + 7 * 8^1 + 1 * 8^0) = 1465 := by
  sorry

end base_8_to_base_10_2671_to_1465_l553_553568


namespace scientific_notation_of_125000_l553_553188

theorem scientific_notation_of_125000 :
  125000 = 1.25 * 10^5 := sorry

end scientific_notation_of_125000_l553_553188


namespace negation_proposition_l553_553835

open Classical

theorem negation_proposition :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 > 0 :=
by
  sorry

end negation_proposition_l553_553835


namespace exists_infinitely_many_n_l553_553385

def digit_sum (m : ℕ) : ℕ := sorry  -- Define the digit sum function

theorem exists_infinitely_many_n (S : ℕ → ℕ)
  (hS : ∀ m : ℕ, S m = digit_sum m) :
  ∃ᶠ n in at_top, S (3^n) ≥ S (3^(n + 1)) := 
sorry

end exists_infinitely_many_n_l553_553385


namespace length_BP_eq_CQ_l553_553099

-- Definitions of the geometric setup
variables {A B C M K L X Y P Q : Type*} [Incircle A B C K L X Y] 

-- Given conditions
axiom median_intersects_incircle (A B C M K L : Type*) 
  (h1 : is_median AM B C) 
  (h2 : intersects_incircle AM K L) : 
  true

axiom lines_through_parallel (K L X Y : Type*) 
  (h3 : are_parallel BC (line_through K)) 
  (h4 : are_parallel BC (line_through L)) 
  (h5 : intersects_other_points K X) 
  (h6 : intersects_other_points L Y) : 
  true

axiom lines_intersect_BC (A X Y P Q : Type*) 
  (h7 : intersects BC AX P) 
  (h8 : intersects BC AY Q) : 
  true

-- To prove: BP = CQ
theorem length_BP_eq_CQ : 
  ∀ {A B C M K L X Y P Q : Type*} 
  (h1 : is_median AM B C) 
  (h2 : intersects_incircle AM K L) 
  (h3 : are_parallel BC (line_through K)) 
  (h4 : are_parallel BC (line_through L)) 
  (h5 : intersects_other_points K X)  
  (h6 : intersects_other_points L Y)  
  (h7 : intersects BC AX P)  
  (h8 : intersects BC AY Q), 
  length B P = length C Q := 
sorry

end length_BP_eq_CQ_l553_553099


namespace remainder_div_9_l553_553257

theorem remainder_div_9 (x y : ℤ) (h : 9 ∣ (x + 2 * y)) : (2 * (5 * x - 8 * y - 4)) % 9 = -8 ∨ (2 * (5 * x - 8 * y - 4)) % 9 = 1 :=
by
  sorry

end remainder_div_9_l553_553257


namespace max_length_MN_l553_553840

theorem max_length_MN (p : ℝ) (h a b c r : ℝ)
  (h_perimeter : a + b + c = 2 * p)
  (h_tangent : r = (a * h) / (2 * p))
  (h_parallel : ∀ h r : ℝ, ∃ k : ℝ, MN = k * (1 - 2 * r / h)) :
  ∀ k : ℝ, MN = (p / 4) :=
sorry

end max_length_MN_l553_553840


namespace all_equal_l553_553133

theorem all_equal 
  (x : Fin 2007 → ℝ)
  (h : ∀ (I : Finset (Fin 2007)), I.card = 7 → 
       ∃ (J : Finset (Fin 2007)), J.card = 11 ∧ 
       (∑ i in I, x i) / 7 = (∑ j in J, x j) / 11) :
  ∃ c : ℝ, ∀ i, x i = c :=
by
  sorry

end all_equal_l553_553133


namespace num_8_digit_integers_l553_553626

theorem num_8_digit_integers (N : ℕ) (hN : 9999999 < N ∧ N < 100000000)
    (hdiv : ∀ (i : Fin 8), (N.remove_digit i).digits 7) : 
  ∃ T, T = 64 :=
by
  sorry

end num_8_digit_integers_l553_553626


namespace mean_greater_than_median_l553_553539

theorem mean_greater_than_median (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5 
  let median := x + 4 
  mean - median = 4 :=
by 
  sorry

end mean_greater_than_median_l553_553539


namespace prove_expression_l553_553657

noncomputable def a : ℝ := 2 / (real.sqrt 3 - 1)

theorem prove_expression : (a^2 - 2 * a + 9 = 11) :=
by 
  -- Insert proof here
  sorry

end prove_expression_l553_553657


namespace a_minus_b_value_l553_553304

theorem a_minus_b_value (a b : ℝ) :
  (∀ x : ℝ, (ax^2 - bx + 1 < 0) ↔ (x < -1/2 ∨ x > 2)) →
  let r1 := -1/2
  let r2 := 2
  (ax^2 - bx + 1 = 0) with
  root_r1 : (a * r1^2 - b * r1 + 1 = 0),
  root_r2 : (a * r2^2 - b * r2 + 1 = 0) :
  a - b = 1/2 :=
by {
  sorry,
}

end a_minus_b_value_l553_553304


namespace solve_eq1_solve_eq2_l553_553462

theorem solve_eq1 : {x : ℚ | x^2 - 4 = 0} = {2, -2} := by
  sorry

theorem solve_eq2 : {x : ℚ | (x + 3)^2 = (2x - 1) * (x + 3)} = {-3, 4} := by
  sorry

end solve_eq1_solve_eq2_l553_553462


namespace graph_f_intersects_x_eq_1_at_most_once_l553_553483

-- Define a function f from ℝ to ℝ
def f : ℝ → ℝ := sorry  -- Placeholder for the actual function

-- Define the domain of the function f (it's a generic function on ℝ for simplicity)
axiom f_unique : ∀ x y : ℝ, f x = f y → x = y  -- If f(x) = f(y), then x must equal y

-- Prove that the graph of y = f(x) intersects the line x = 1 at most once
theorem graph_f_intersects_x_eq_1_at_most_once : ∃ y : ℝ, (f 1 = y) ∨ (¬∃ y : ℝ, f 1 = y) :=
by
  -- Proof goes here
  sorry

end graph_f_intersects_x_eq_1_at_most_once_l553_553483


namespace sum_of_midpoint_xcoords_l553_553491

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l553_553491


namespace inverse_proportion_function_neg_k_l553_553364

variable {k : ℝ}
variable {y1 y2 : ℝ}

theorem inverse_proportion_function_neg_k
  (h1 : k ≠ 0)
  (h2 : y1 > y2)
  (hA : y1 = k / (-2))
  (hB : y2 = k / 5) :
  k < 0 :=
sorry

end inverse_proportion_function_neg_k_l553_553364


namespace binom_1300_2_l553_553610

theorem binom_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_l553_553610


namespace problem_statement_l553_553329

variable (a b c : ℤ) -- Declare variables as integers

-- Define conditions based on the problem
def smallest_natural_number (a : ℤ) := a = 1
def largest_negative_integer (b : ℤ) := b = -1
def number_equal_to_its_opposite (c : ℤ) := c = 0

-- State the theorem
theorem problem_statement (h1 : smallest_natural_number a) 
                         (h2 : largest_negative_integer b) 
                         (h3 : number_equal_to_its_opposite c) : 
  a + b + c = 0 := 
  by 
    rw [h1, h2, h3] 
    simp

end problem_statement_l553_553329


namespace forces_through_centroid_l553_553213

noncomputable def resultant_forces (A B C M G : Point) : Vector :=
let MA := force_vector M A in
let MB := force_vector M B in
let MC := force_vector M C in
MA + MB + MC

noncomputable def centroid (A B C : Point) : Point :=
(Point (⅓ * (A.x + B.x + C.x)) (⅓ * (A.y + B.y + C.y)))

theorem forces_through_centroid (A B C M : Point) (G : Point) :
  M ∈ triangle A B C →
  G = centroid A B C →
  resultant_forces A B C M G = 3 * (force_vector M G) :=
by
  sorry

end forces_through_centroid_l553_553213


namespace dimes_difference_l553_553217

theorem dimes_difference (a b c : ℕ) :
  a + b + c = 120 →
  5 * a + 10 * b + 25 * c = 1265 →
  c ≥ 10 →
  (max (b) - min (b)) = 92 :=
sorry

end dimes_difference_l553_553217


namespace line_slope_obtuse_angle_l553_553706

theorem line_slope_obtuse_angle (a : ℝ) : (2a - 1 < 0) → a < (1/2) :=
by
  intros h
  exact h

end line_slope_obtuse_angle_l553_553706


namespace smallest_n_for_Tn_integer_l553_553030

noncomputable def K : ℚ := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def D : ℕ := 2^3 * 3^2 * 5 * 7

def T_n (n : ℕ) : ℚ := (n * 5^(n-1)) * K

theorem smallest_n_for_Tn_integer :
  ∃ n : ℕ, T_n n ∈ ℤ ∧ ∀ m < n, T_n m ∉ ℤ :=
  sorry

end smallest_n_for_Tn_integer_l553_553030


namespace math_problems_l553_553693

-- Conditions
variable {f : ℝ → ℝ}
variable (hf1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y))
variable (hf_pos : ∀ x : ℝ, x > 0 → f(x) > 0)
variable (hf1_val : f(1) = 2)

-- Goals to prove
theorem math_problems :
  (f(0) = 0) ∧
  (f(3) = 6) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → f(x1) < f(x2)) ∧
  (∀ (a : ℝ), (∀ x : ℝ, f(4^x - a) + f(6 + 2^(x + 1)) > 6) → a ≤ 3) :=
by
  -- Explicitly introduce assumptions
  intros
  
  -- TODO: The actual proof steps would follow here, but we skip them with sorry
  sorry

end math_problems_l553_553693


namespace smallest_n_for_T_integer_l553_553018

noncomputable def J := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def T (n : ℕ) : ℚ := ∑ x in finset.range (5^n + 1), 
  ∑ d in {
    digit | digit.to_nat ≠ 0 ∧ digit.to_nat < 10 
  }, (1 : ℚ) / (digit.to_nat : ℚ)

theorem smallest_n_for_T_integer : ∃ n : ℕ, T n ∈ ℤ ∧ ∀ m : ℕ, T m ∈ ℤ → 63 ≤ n :=
by {
  sorry
}

end smallest_n_for_T_integer_l553_553018


namespace inverse_proportion_function_neg_k_l553_553363

variable {k : ℝ}
variable {y1 y2 : ℝ}

theorem inverse_proportion_function_neg_k
  (h1 : k ≠ 0)
  (h2 : y1 > y2)
  (hA : y1 = k / (-2))
  (hB : y2 = k / 5) :
  k < 0 :=
sorry

end inverse_proportion_function_neg_k_l553_553363


namespace distance_from_point_to_x_axis_l553_553874

theorem distance_from_point_to_x_axis (x y z : ℝ) (h : (x, y, z) = (1, 3, 4)) : sqrt (y^2 + z^2) = 5 :=
by
  have h1 : y = 3 := by cases h; refl
  have h2 : z = 4 := by cases h; refl
  rw [h1, h2]
  have : sqrt (3^2 + 4^2) = sqrt 25 := by congr; ring
  rw [this]
  exact sqrt_eq_rfl (by norm_num)

end distance_from_point_to_x_axis_l553_553874


namespace probability_blue_then_yellow_l553_553744

variable (total_chips : ℕ := 15)
variable (blue_chips : ℕ := 10)
variable (yellow_chips : ℕ := 5)
variable (initial_blue_prob : ℚ := blue_chips / total_chips)
variable (remaining_chips : ℕ := total_chips - 1)
variable (next_yellow_prob : ℚ := yellow_chips / remaining_chips)
variable (final_prob : ℚ := initial_blue_prob * next_yellow_prob)

theorem probability_blue_then_yellow :
  final_prob = 5 / 21 :=
sorry

end probability_blue_then_yellow_l553_553744


namespace num_integers_between_sqrt_range_l553_553853

theorem num_integers_between_sqrt_range :
  {x : ℕ | 5 > Real.sqrt x ∧ Real.sqrt x > 3}.card = 15 :=
by sorry

end num_integers_between_sqrt_range_l553_553853


namespace distance_to_circle_center_l553_553134

open Real

noncomputable def circle_center (a b c d: ℝ) : (ℝ × ℝ) :=
  let x0 := -a / 2
  let y0 := -b / 2
  (x0, y0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_to_circle_center :
  let circle_eq : ℝ × ℝ × ℝ × ℝ := (6, 8, 9, 0)
  let center := circle_center 6 8 9 0
  let point := (11, 5)
  distance center point = sqrt 65 :=
by
  let (a, b, c, d) := (6, 8, 9, 0)
  let center := circle_center a b c d
  let point := (11, 5)
  let dist := distance center point
  have h1 : center = (-a / 2, -b / 2) := rfl
  have h2 : dist = sqrt ((11 - (-3))^2 + (5 - (-4))^2) := by sorry
  have h3 : sqrt ((11 - (-3))^2 + (5 - (-4))^2) = sqrt 65 := by sorry
  exact calc
    dist = sqrt ((11 - (-3))^2 + (5 - (-4))^2) : h2
    ... = sqrt 65 : h3

end distance_to_circle_center_l553_553134


namespace jakes_money_ratio_l553_553374

-- Definitions based on the given conditions
def candy_cost : ℝ := 0.20  -- 20 cents
def feeding_allowance : ℝ := 4 -- $4
def candies_purchased : ℝ := 5 -- 5 candies

-- Total money given to friend
def money_given_to_friend : ℝ := candy_cost * candies_purchased

-- Ratio of money given to friend to feeding allowance
def ratio : ℝ := money_given_to_friend / feeding_allowance

-- The theorem to be proven
theorem jakes_money_ratio : ratio = 1 / 4 :=
by
  -- Here will go the proof, currently omitted
  sorry

end jakes_money_ratio_l553_553374


namespace average_mpg_l553_553603

-- Definitions for conditions
def initial_odometer_reading : Int := 48500
def initial_gas_fill : Int := 8
def second_gas_fill : Int := 10
def second_odometer_reading : Int := 48800
def third_gas_fill : Int := 15
def final_odometer_reading : Int := 49350

-- Theorem statement
theorem average_mpg : 
  let total_distance := final_odometer_reading - initial_odometer_reading,
      total_gas := second_gas_fill + third_gas_fill in
  (total_distance.toRational / total_gas.toRational).toReal ≈ 34.0 :=
by
  sorry

end average_mpg_l553_553603


namespace eq_has_unique_solution_l553_553627

theorem eq_has_unique_solution : 
  ∃! x : ℝ, (x ≠ 0)
    ∧ ((x < 0 → false) ∧ 
      (x > 0 → (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9)) :=
by sorry

end eq_has_unique_solution_l553_553627


namespace no_distinct_numbers_grid_l553_553370

theorem no_distinct_numbers_grid : ¬ ∃ f : Fin 6 → Fin 6 → ℕ,
  (∀ i j, f i j ≠ f i (j+1) ∧ f i j ≠ f (i+1) j) ∧
  (∀ i, ∃ sum1 sum2 : ℕ, sum1 ∈ {2022, 2023} ∧ sum2 ∈ {2022, 2023} ∧
    (finset.range 5).sum (λ k, f i k) = sum1 ∧ 
    (finset.range 6).sum (λ k, f k i) = sum2) → false :=
by {
  sorry
}

end no_distinct_numbers_grid_l553_553370


namespace benny_stored_bales_l553_553122

theorem benny_stored_bales (initial_bales current_bales : ℕ) (h₁ : initial_bales = 47) (h₂ : current_bales = 82) : 
  current_bales - initial_bales = 35 :=
by
  rw [h₁, h₂]
  exact Nat.sub_self 47

end benny_stored_bales_l553_553122


namespace shaniqua_styles_count_l553_553441

variable (S : ℕ)

def shaniqua_haircuts (haircuts : ℕ) : ℕ := 12 * haircuts
def shaniqua_styles (styles : ℕ) : ℕ := 25 * styles

theorem shaniqua_styles_count (total_money haircuts : ℕ) (styles : ℕ) :
  total_money = shaniqua_haircuts haircuts + shaniqua_styles styles → haircuts = 8 → total_money = 221 → S = 5 :=
by
  sorry

end shaniqua_styles_count_l553_553441


namespace blue_length_is_2_l553_553734

-- Define the lengths of the parts
def total_length : ℝ := 4
def purple_length : ℝ := 1.5
def black_length : ℝ := 0.5

-- Define the length of the blue part with the given conditions
def blue_length : ℝ := total_length - (purple_length + black_length)

-- State the theorem we need to prove
theorem blue_length_is_2 : blue_length = 2 :=
by 
  sorry

end blue_length_is_2_l553_553734


namespace variance_of_data_set_l553_553678

theorem variance_of_data_set (a : ℝ) (ha : (1 + a + 3 + 6 + 7) / 5 = 4) : 
  (1 / 5) * ((1 - 4)^2 + (a - 4)^2 + (3 - 4)^2 + (6 - 4)^2 + (7 - 4)^2) = 24 / 5 :=
by
  sorry

end variance_of_data_set_l553_553678


namespace polynomial_remainder_theorem_l553_553982

noncomputable def polynomial_remainder : Polynomial ℤ :=
  (Polynomial.X ^ 4 - 1) * (Polynomial.X ^ 3 - 1) % (Polynomial.X ^ 2 + 1)

theorem polynomial_remainder_theorem : polynomial_remainder = (Polynomial.C 2 + Polynomial.X) :=
  sorry

end polynomial_remainder_theorem_l553_553982


namespace students_in_class_l553_553469

theorem students_in_class
  (N : ℕ)
  (average_age : ℕ → ℕ → ℕ)
  (h1 : average_age (⋆ ⋆ ⋆) 15 = 15)
  (h2 : average_age 5 13 = 13)
  (h3 : average_age 9 16 = 16)
  (h4 : age_last_student = 16):
  N = 15 := by
  -- The proof will be filled in here
  sorry

end students_in_class_l553_553469


namespace part1_part2_l553_553668

def f (x : ℝ) : ℝ :=
  if x ≠ 1 then 1 / (x - 1) else 0

theorem part1 (x : ℝ) (h : x ≠ 1) : f (2 - x) = 1 / (1 - x) :=
  sorry

theorem part2 : 
  f (1/20) + f (3/20) + f (5/20) + f (7/20) + f (9/20) + f (11/20) + 
  f (13/20) + f (15/20) + f (17/20) + f (19/20) + f (21/20) + f (23/20) + 
  f (25/20) + f (27/20) + f (29/20) + f (31/20) + f (33/20) + f (35/20) + 
  f (37/20) + f (39/20) = 0 := 
  sorry

end part1_part2_l553_553668


namespace range_of_m_l553_553477

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - 3 * x ^ 2 - 9 * x + 3

def has_three_zeros (g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0

theorem range_of_m :
  (∃ m : ℝ, has_three_zeros (λ x, f x - m)) ↔ ∃ m : ℝ, -24 < m ∧ m < 8 := 
sorry

end range_of_m_l553_553477


namespace no_fixed_point_range_of_a_fixed_point_in_interval_l553_553332

-- Problem (1)
theorem no_fixed_point_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + a ≠ x) →
  3 - 2 * Real.sqrt 2 < a ∧ a < 3 + 2 * Real.sqrt 2 :=
by
  sorry

-- Problem (2)
theorem fixed_point_in_interval (f : ℝ → ℝ) (n : ℤ) :
  (∀ x : ℝ, f x = -Real.log x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ n ≤ x₀ ∧ x₀ < n + 1) →
  n = 2 :=
by
  sorry

end no_fixed_point_range_of_a_fixed_point_in_interval_l553_553332


namespace smallest_n_for_T_n_is_integer_l553_553026

def L : ℚ := ∑ i in Finset.range 9, i.succ⁻¹  -- sum of reciprocals of non-zero digits

def D : ℕ := 2^3 * 3^2 * 5 * 7  -- denominator of L in simplified form

def T (n : ℕ) : ℚ := (n * 5^(n-1)) * L + 1  -- expression for T_n

theorem smallest_n_for_T_n_is_integer : ∃ n : ℕ, 0 < n ∧ T n ∈ ℤ ∧ n = 504 :=
by
  use 504
  -- It remains to prove the conditions
  sorry

end smallest_n_for_T_n_is_integer_l553_553026


namespace cafeteria_problem_l553_553938

-- Define the initial conditions and transition relations
variable (a b : ℕ → ℝ)
variable (h_initial : a 1 = 300)
variable (h_sum : ∀ n, a n + b n = 500)
variable (h_transition : ∀ n, a (n + 1) = 0.8 * a n + 0.3 * b n)

theorem cafeteria_problem : 
    (a 2 = 300) ∧ (∀ n, a n = 300) := 
  by
    -- Hypotheses and required results
    have h2 : a (1 + 1) = 0.8 * a 1 + 0.3 * b 1 := h_transition 1
    have h_b1 : b 1 = 200 := by linarith [h_sum 1, h_initial]
    have h_a2 : a 2 = 0.8 * a 1 + 0.3 * b 1 := by rw [add_comm, h2]
    have h_aeq300 : a 2 = 300 := by linarith [h_a2, h_initial, h_b1]
    split
    · exact h_aeq300
    · intro n
      have hrec : ∀ (m : ℕ), a (m + 1) = 300 := sorry
      exact hrec (n - 1)

end cafeteria_problem_l553_553938


namespace distance_between_stations_is_correct_l553_553866

noncomputable def distance_between_stations : ℕ := 200

theorem distance_between_stations_is_correct 
  (start_hour_p : ℕ := 7) 
  (speed_p : ℕ := 20) 
  (start_hour_q : ℕ := 8) 
  (speed_q : ℕ := 25) 
  (meeting_hour : ℕ := 12)
  (time_travel_p := meeting_hour - start_hour_p) -- Time traveled by train from P
  (time_travel_q := meeting_hour - start_hour_q) -- Time traveled by train from Q 
  (distance_travel_p := speed_p * time_travel_p) 
  (distance_travel_q := speed_q * time_travel_q) : 
  distance_travel_p + distance_travel_q = distance_between_stations :=
by 
  sorry

end distance_between_stations_is_correct_l553_553866


namespace min_value_of_f_on_interval_l553_553832

def f (x ϕ : ℝ) : ℝ := 2 * Real.sin (2 * x + ϕ)

theorem min_value_of_f_on_interval :
  ∀ ϕ : ℝ, (0 < ϕ ∧ ϕ < Real.pi / 2) →
  (∀ x : ℝ, f x ϕ = f (Real.pi / 12 * 2 - x) ϕ) →
  (∃ xmin, (Real.pi / 2 ≤ xmin ∧ xmin ≤ Real.pi) ∧ 
           ∀ x ∈ Icc (Real.pi / 2) Real.pi, f x ϕ ≥ f xmin ϕ) →
  ∃ xmin, xmin ∈ Icc (Real.pi / 2) Real.pi ∧ f xmin ϕ = -2 := by
  sorry

end min_value_of_f_on_interval_l553_553832


namespace card_M_is_13_l553_553649

-- Define the operation "⊙"
def super_operation (m n : ℕ) : ℕ :=
  if (m % 2 = n % 2) then m + n else m * n

-- Define the set M
def M : Set (ℕ × ℕ) :=
  { p_pair | let (p, q) := p_pair in super_operation p q = 10 ∧ p > 0 ∧ q > 0 }

-- Define the statement
theorem card_M_is_13 : ∃ (M_cardinality : ℕ), M_cardinality = 13 := by
  let M_cardinality := Set.card M
  exact ⟨M_cardinality, sorry⟩

end card_M_is_13_l553_553649


namespace james_change_and_new_cost_l553_553375

theorem james_change_and_new_cost
  (total_price : ℝ) (discount_rate : ℝ) (payment_amount : ℝ) (num_packs : ℕ)
  (h1 : total_price = 12)
  (h2 : discount_rate = 0.15)
  (h3 : payment_amount = 20)
  (h4 : num_packs = 3) :
  let discount_amount := total_price * discount_rate,
      discounted_price := total_price - discount_amount,
      change := payment_amount - discounted_price,
      new_cost_per_pack := discounted_price / num_packs
  in change = 9.80 ∧ new_cost_per_pack = 3.40 :=
by
  let discount_amount := total_price * discount_rate
  let discounted_price := total_price - discount_amount
  let change := payment_amount - discounted_price
  let new_cost_per_pack := discounted_price / num_packs
  have h5 : discount_amount = 1.80, by sorry
  have h6 : discounted_price = 10.20, by sorry
  have h7 : change = 9.80, by sorry
  have h8 : new_cost_per_pack = 3.40, by sorry
  exact ⟨h7, h8⟩

end james_change_and_new_cost_l553_553375


namespace billy_initial_balloons_l553_553048

theorem billy_initial_balloons :
    let total_packs := 12,
    let balloons_per_pack := 8,
    let total_balloons := total_packs * balloons_per_pack,
    let milly_kept := 10,
    let floretta_kept := 12,
    let tamara_kept := 8,
    let billy_kept := 6,
    let total_kept := milly_kept + floretta_kept + tamara_kept + billy_kept,
    let remaining_balloons := total_balloons - total_kept,
    let evenly_distributed := remaining_balloons / 4,
    let billy_initial := billy_kept + evenly_distributed
    in billy_initial = 21 := by
  sorry

end billy_initial_balloons_l553_553048


namespace find_xy_l553_553550

theorem find_xy
    (x y : ℝ)
    (hx : 0 < x)
    (hy : 0 < y)
    (h1 : cos (π * x) ^ 2 + 2 * sin (π * y) = 1)
    (h2 : sin (π * x) + sin (π * y) = 0)
    (h3 : x ^ 2 - y ^ 2 = 12) :
    x = 4 ∧ y = 2 :=
  by
  sorry

end find_xy_l553_553550


namespace solve_log_equation_l553_553814

theorem solve_log_equation (x : ℝ) 
  (h1 : 7 * x + 3 > 0)
  (h2 : 4 * x + 5 > 0) :
  (log (sqrt (7 * x + 3)) + log (sqrt (4 * x + 5)) = 1 / 2 + log 3) ↔ x = 1 := 
begin
  sorry
end

end solve_log_equation_l553_553814


namespace smallest_w_l553_553793

theorem smallest_w (x y w : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 ^ x) ∣ (3125 * w)) (h4 : (3 ^ y) ∣ (3125 * w)) 
  (h5 : (5 ^ (x + y)) ∣ (3125 * w)) (h6 : (7 ^ (x - y)) ∣ (3125 * w))
  (h7 : (13 ^ 4) ∣ (3125 * w))
  (h8 : x + y ≤ 10) (h9 : x - y ≥ 2) :
  w = 33592336 :=
by
  sorry

end smallest_w_l553_553793


namespace concyclicity_of_AIMN_l553_553788

noncomputable def Triangle_ABC : Type := sorry -- Definition of triangle ABC
noncomputable def D : Type := sorry -- Intersection of internal angle bisector from A with segment BC
noncomputable def M : Type := sorry -- Intersection of perpendicular bisector of AD with angle bisector from B
noncomputable def N : Type := sorry -- Intersection of perpendicular bisector of AD with angle bisector from C
noncomputable def I : Type := sorry -- Center of the inscribed circle of triangle ABC

theorem concyclicity_of_AIMN (A B C D M N I : Type) 
  (D_intersect : D = intersection (angle_bisector A B C) (segment B C)) 
  (M_intersect : M = intersection (perpendicular_bisector (segment A D)) (angle_bisector B A C)) 
  (N_intersect : N = intersection (perpendicular_bisector (segment A D)) (angle_bisector C A B))
  (I_center : I = incenter A B C) : 
  cyclic A I M N :=
sorry

end concyclicity_of_AIMN_l553_553788


namespace geometric_sequence_properties_l553_553754

noncomputable def geometric_sequence (a2 a5 : ℕ) (n : ℕ) : ℕ :=
  3 ^ (n - 1)

noncomputable def sum_first_n_terms (n : ℕ) : ℕ :=
  (3^n - 1) / 2

def T10_sum_of_sequence : ℚ := 10/11

theorem geometric_sequence_properties :
  (geometric_sequence 3 81 2 = 3) ∧
  (geometric_sequence 3 81 5 = 81) ∧
  (sum_first_n_terms 2 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2)) ∧
  (sum_first_n_terms 5 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2 + geometric_sequence 3 81 3 + geometric_sequence 3 81 4 + geometric_sequence 3 81 5)) ∧
  T10_sum_of_sequence = 10/11 :=
by
  sorry

end geometric_sequence_properties_l553_553754


namespace nearest_edge_correct_l553_553577

-- Define the problem conditions
def wall_width : ℝ := 25
def picture_width : ℝ := 4
def shift : ℝ := 1

-- Calculate the required value
def nearest_edge_from_left (wall_width picture_width shift : ℝ) : ℝ :=
  let center_of_wall := wall_width / 2
  let center_of_picture := center_of_wall + shift
  center_of_picture - (picture_width / 2)

-- The goal is to prove that the nearest edge from the left is 11.5 feet
theorem nearest_edge_correct :
  nearest_edge_from_left wall_width picture_width shift = 11.5 :=
by
  dsimp [nearest_edge_from_left, wall_width, picture_width, shift]
  norm_num
  sorry

end nearest_edge_correct_l553_553577


namespace g_is_odd_l553_553765

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  sorry

end g_is_odd_l553_553765


namespace arithmetic_sequence_sum_l553_553280

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h1 : a 1 + a 3 + a 5 = 9) (h2 : a 2 + a 4 + a 6 = 15) : a 3 + a 4 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l553_553280


namespace range_of_a_l553_553981

noncomputable def f (a x : ℝ) : ℝ := log a (8 - a * x)

theorem range_of_a (a : ℝ) (h1: a > 0) (h2: a ≠ 1)
  (h3: ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 2 → x < y → f a y < f a x) :
  1 < a ∧ a < 4 :=
by sorry

end range_of_a_l553_553981


namespace triangle_inequality_l553_553283

-- Define the conditions as Lean hypotheses
variables {a b c : ℝ}

-- Lean statement for the problem
theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 :=
sorry

end triangle_inequality_l553_553283


namespace line_equation_l553_553928

theorem line_equation :
  ∃ (m b : ℚ), (m = 3 / 4 ∧ b = 13 / 2) ∧
  (∀ x y : ℚ, (pmul (3, -4) (psub (x, y) (2, 8)) = 0) → (y = m * x + b)) :=
by
  sorry

def pmul (a b : ℚ × ℚ) : ℚ :=
  a.1 * b.1 + a.2 * b.2

def psub (a b : ℚ × ℚ) : ℚ × ℚ :=
  (a.1 - b.1, a.2 - b.2)

end line_equation_l553_553928


namespace smallest_n_for_T_n_is_integer_l553_553022

def L : ℚ := ∑ i in Finset.range 9, i.succ⁻¹  -- sum of reciprocals of non-zero digits

def D : ℕ := 2^3 * 3^2 * 5 * 7  -- denominator of L in simplified form

def T (n : ℕ) : ℚ := (n * 5^(n-1)) * L + 1  -- expression for T_n

theorem smallest_n_for_T_n_is_integer : ∃ n : ℕ, 0 < n ∧ T n ∈ ℤ ∧ n = 504 :=
by
  use 504
  -- It remains to prove the conditions
  sorry

end smallest_n_for_T_n_is_integer_l553_553022


namespace introduce_trigonometric_functions_after_similarity_theorem_l553_553141

theorem introduce_trigonometric_functions_after_similarity_theorem
  (h : ∀ (ΔABC ΔDEF : Triangle),
    is_right_triangle ΔABC →
    is_right_triangle ΔDEF →
    equi_acute_angles ΔABC ΔDEF →
    similar_triangles ΔABC ΔDEF) :
  introduction_point_trigonometric_functions = after_similarity_theorem :=
begin
  sorry
end

end introduce_trigonometric_functions_after_similarity_theorem_l553_553141


namespace steven_needs_more_seeds_l553_553081

def apple_seeds : Nat := 6
def pear_seeds : Nat := 2
def grape_seeds : Nat := 3
def apples_set_aside : Nat := 4
def pears_set_aside : Nat := 3
def grapes_set_aside : Nat := 9
def seeds_required : Nat := 60

theorem steven_needs_more_seeds : 
  seeds_required - (apples_set_aside * apple_seeds + pears_set_aside * pear_seeds + grapes_set_aside * grape_seeds) = 3 := by
  sorry

end steven_needs_more_seeds_l553_553081


namespace shagreen_initial_area_l553_553070

theorem shagreen_initial_area (S : ℕ) (h1 : S > 0) (h2 : S / 3 > 0) (h3 : S / 21 > 0) 
  (h4 : S - (S / 3 + S / 21) = 0) : S = 42 :=
begin
  sorry
end

end shagreen_initial_area_l553_553070


namespace sequence_infinitely_many_powers_of_2_l553_553059

def is_power_of_two (x : ℕ) : Prop :=
∃ k : ℕ, x = 2^k

theorem sequence_infinitely_many_powers_of_2 (a : ℕ → ℕ) (h : ∀ n, a n = floor (n * Real.sqrt 2)) :
  ∃ (n : ℕ → ℕ), ∀ k : ℕ, is_power_of_two (a (n k)) :=
by
  sorry

end sequence_infinitely_many_powers_of_2_l553_553059


namespace quadrilateral_EFGH_EG_l553_553358

theorem quadrilateral_EFGH_EG :
  ∃ (EG : ℤ), EG = 21 ∧
    ∃ (EF FG GH HE : ℝ), EF = 7 ∧ FG = 21 ∧ GH = 7 ∧ HE = 13 ∧
      EF + FG > EG ∧ FG + GH > EG ∧ EG + GH > HE ∧ EG + EF > FG := 
begin
  sorry
end

end quadrilateral_EFGH_EG_l553_553358


namespace no_triang_exist_with_given_cond_l553_553322

noncomputable def is_impossible_triangle
  (a b : ℝ) (A : ℝ) : Prop :=
  ∀ (B : ℝ), (b * Real.sin (A * Real.pi / 180)) / a ≤ 1

theorem no_triang_exist_with_given_cond :
  ¬∃ (a b : ℝ) (A : ℝ), 
    a = 4 ∧
    b = 5 * Real.sqrt 2 ∧
    A = 45 ∧
    is_impossible_triangle a b A :=
begin
  sorry

end no_triang_exist_with_given_cond_l553_553322


namespace sample_size_community_A_l553_553620

variable (A B C H : ℕ) (total_families : ℕ) (sampling_ratio : ℚ)

def low_income_families_A := 360
def low_income_families_B := 270
def low_income_families_C := 180
def housing_units := 90

theorem sample_size_community_A (h1 : total_families = low_income_families_A + low_income_families_B + low_income_families_C)
  (h2 : sampling_ratio = housing_units / total_families) : 
  low_income_families_A * sampling_ratio = 40 :=
by
  rw [←h1, ←h2]
  sorry

end sample_size_community_A_l553_553620


namespace cube_number_sum_is_102_l553_553167

noncomputable def sum_of_cube_numbers (n1 n2 n3 n4 n5 n6 : ℕ) : ℕ := n1 + n2 + n3 + n4 + n5 + n6

theorem cube_number_sum_is_102 : 
  ∃ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 12 ∧ 
    n2 = n1 + 2 ∧ 
    n3 = n2 + 2 ∧ 
    n4 = n3 + 2 ∧ 
    n5 = n4 + 2 ∧ 
    n6 = n5 + 2 ∧ 
    ((n1 + n6 = n2 + n5) ∧ (n1 + n6 = n3 + n4)) ∧ 
    sum_of_cube_numbers n1 n2 n3 n4 n5 n6 = 102 :=
by
  sorry

end cube_number_sum_is_102_l553_553167


namespace steven_needs_more_seeds_l553_553083

theorem steven_needs_more_seeds :
  let total_seeds_needed := 60
  let seeds_per_apple := 6
  let seeds_per_pear := 2
  let seeds_per_grape := 3
  let apples_collected := 4
  let pears_collected := 3
  let grapes_collected := 9
  total_seeds_needed - (apples_collected * seeds_per_apple + pears_collected * seeds_per_pear + grapes_collected * seeds_per_grape) = 3 :=
by
  sorry

end steven_needs_more_seeds_l553_553083


namespace train_length_l553_553581

noncomputable def length_of_train (speed_kmph : ℝ) (time_sec : ℝ) (length_platform_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmph * 1000) / 3600
  let distance_covered := speed_ms * time_sec
  distance_covered - length_platform_m

theorem train_length :
  length_of_train 72 25 340.04 = 159.96 := by
  sorry

end train_length_l553_553581


namespace mike_earnings_l553_553418

theorem mike_earnings :
  let total_games := 16
  let non_working_games := 8
  let price_per_game := 7
  let working_games := total_games - non_working_games
  let earnings := working_games * price_per_game
  earnings = 56 := 
by
  sorry

end mike_earnings_l553_553418


namespace average_multiplied_by_5_l553_553153

theorem average_multiplied_by_5 (avg : ℝ) (n : ℕ) (h_avg : avg = 20) (h_n : n = 7) :
  let new_avg := (5 * n * avg) / n
  in new_avg = 100 := 
by
  sorry

end average_multiplied_by_5_l553_553153


namespace calc_fx_h_minus_fx_l553_553330

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

theorem calc_fx_h_minus_fx (x h : ℝ) : 
  f(x + h) - f(x) = h * (6 * x + 3 * h + 5) := 
by 
  sorry

end calc_fx_h_minus_fx_l553_553330


namespace root_of_quadratic_l553_553805

theorem root_of_quadratic (a b c : ℝ) :
  (4 * a + 2 * b + c = 0) ↔ (a * 2^2 + b * 2 + c = 0) :=
by
  sorry

end root_of_quadratic_l553_553805


namespace residents_count_l553_553557

theorem residents_count :
  (∃ (R N : ℤ), 
    R + N = 586 ∧ 
    12.95 * R + 17.95 * N = 9423.70 ∧
    R ≥ 0 ∧ N ≥ 0) →
  ∃ (R : ℤ), R = 220 :=
begin
  intro h,
  obtain ⟨R, N, h1, h2, h3, h4⟩ := h,
  sorry -- Proof is skipped
end

end residents_count_l553_553557


namespace smallest_n_for_T_integer_l553_553015

noncomputable def J := ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9}, (1 : ℚ) / i

def T (n : ℕ) : ℚ := ∑ x in finset.range (5^n + 1), 
  ∑ d in {
    digit | digit.to_nat ≠ 0 ∧ digit.to_nat < 10 
  }, (1 : ℚ) / (digit.to_nat : ℚ)

theorem smallest_n_for_T_integer : ∃ n : ℕ, T n ∈ ℤ ∧ ∀ m : ℕ, T m ∈ ℤ → 63 ≤ n :=
by {
  sorry
}

end smallest_n_for_T_integer_l553_553015


namespace principal_calculation_l553_553544

theorem principal_calculation (P : ℝ) (R : ℝ) (T : ℝ) (SI CI diff : ℝ): 
  R = 20 → 
  T = 2 → 
  SI = P * R * T / 100 → 
  CI = P * ((1 + R/100)^T) - P → 
  diff = CI - SI → 
  diff = 72 → 
  P = 1800 :=
by {
  intros,
  -- initialize variables
  have h1 : SI = P * 40 / 100 := by rwa [mul_assoc, mul_comm, mul_div_assoc', mul_comm],
  have h2 : CI = P * (1.2^2 - 1) := by rwa [pow_two, add_mul, mul_add, mul_sub, one_mul, sub_self],
  have h3 : CI - SI = P * 0.04 := by rwa [sub_mul, sub_div, div_eq_mul_one_div, mul_comm, sub_eq_add_neg],
  -- given diff = 72, prove principal
  calc
  P = 72 / 0.04 : by rwa [eq_div_of_mul_eq, mul_comm]
  ... = 7200 / 4 : by rwa [div_eq_div_iff mul_eq_mul', mul_comm, mul_div_comm]
  ... = 1800 : by rwa [div_eq_one]
}
-- adding sorry to skip proof
sorry

end principal_calculation_l553_553544


namespace denote_west_travel_l553_553339

theorem denote_west_travel (east_pos : 2 = +2) : (-3) = -3 :=
by
  sorry

end denote_west_travel_l553_553339


namespace problem_2022_circle_sum_zero_l553_553113

theorem problem_2022_circle_sum_zero (a : ℕ → ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 2022 → a i = sqrt 2 * a (i + 2) - sqrt 3 * a (i + 1))
  (h2 : a 2023 = a 1) (h3 : a 2024 = a 2) : 
  ∑ i in Finset.range 2022, a i * a (i + 2) = 0 :=
by
  sorry

end problem_2022_circle_sum_zero_l553_553113


namespace cos_fourth_power_sum_l553_553209

open Real

theorem cos_fourth_power_sum :
  (cos (0 : ℝ))^4 + (cos (π / 6))^4 + (cos (π / 3))^4 + (cos (π / 2))^4 +
  (cos (2 * π / 3))^4 + (cos (5 * π / 6))^4 + (cos π)^4 = 13 / 4 := 
by
  sorry

end cos_fourth_power_sum_l553_553209


namespace number_of_zeros_in_1000_pow_10_l553_553140

theorem number_of_zeros_in_1000_pow_10 :
  ∀ (a b : ℕ), a = 1000 → b = 10 → (10 ^ 3) ^ b = 10 ^ 30 :=
by
  intros a b ha hb
  rw ha at *
  rw hb at *
  have : (10 ^ 3) ^ 10 = 10 ^ (3 * 10) := pow_mul 10 3 10
  exact this

end number_of_zeros_in_1000_pow_10_l553_553140


namespace find_y_squared_l553_553384

variables {PQ RS y : ℝ}
variables {center_on_PQ tangent_to_PR tangent_to_QS : Prop}

-- Define the conditions
def is_isosceles_trapezoid (PQRS : Prop) := (PQRS ∧ (PQ = 120) ∧ (RS = 25) ∧ (PR = y) ∧ (QS = y))
def circle_center_on_PQ := center_on_PQ
def circle_tangent_to_PR_and_QS := tangent_to_PR ∧ tangent_to_QS

-- Define the theorem
theorem find_y_squared {PQRS : Prop} 
  (iso_trap : is_isosceles_trapezoid PQRS)
  (center_PQ : circle_center_on_PQ)
  (tangent_PR_QS : circle_tangent_to_PR_and_QS) : y^2 = 4350 :=
sorry

end find_y_squared_l553_553384


namespace prob_obtuse_angle_l553_553055

-- Define the vertices of the hexagon
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (3, 0)
def C : (ℝ × ℝ) := (4.5, Real.sqrt 3)
def D : (ℝ × ℝ) := (3, 2 * Real.sqrt 3)
def E : (ℝ × ℝ) := (0, 2 * Real.sqrt 3)
def F : (ℝ × ℝ) := (-1.5, Real.sqrt 3)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the condition that a point is inside the hexagon
def in_hexagon (P : ℝ × ℝ) : Prop :=
  let vertices := [A, B, C, D, E, F]
  -- Use a placeholder for actual in-polygon check
  sorry

-- Define the condition that the angle APB is obtuse
def angle_obtuse (P : ℝ × ℝ) : Prop :=
  ¬ (distance A P ^ 2 + distance P B ^ 2 ≤ distance A B ^ 2)

-- The probability that a randomly selected point in the hexagon makes the angle obtuse
noncomputable def probability_obtuse_angle : ℝ :=
  1 - (9 * Real.pi) / (54 * Real.sqrt 3)

-- The theorem to prove the probability of obtuse angle
theorem prob_obtuse_angle :
  probability_obtuse_angle = (6 * Real.sqrt 3 - Real.pi) / (6 * Real.sqrt 3) :=
sorry

end prob_obtuse_angle_l553_553055


namespace leila_distance_yards_l553_553775

/-- Leila ran fifteen marathons, each measuring 26 miles and 395 yards.
    Given one mile equals 1760 yards, if the total distance run by Leila is 
    expressed in miles and yards where 0 ≤ y < 1760, then y = 645 yards. -/
theorem leila_distance_yards :
  let marathons := 15
  let miles_per_marathon := 26
  let yards_per_marathon := 395
  let yards_per_mile := 1760
  let total_yards := marathons * yards_per_marathon
  let y := total_yards % yards_per_mile
  (0 ≤ y ∧ y < yards_per_mile) ∧ y = 645 :=
by
  let marathons := 15
  let miles_per_marathon := 26
  let yards_per_marathon := 395
  let yards_per_mile := 1760
  let total_yards := marathons * yards_per_marathon
  let y := total_yards % yards_per_mile
  have h : y = 645 := 
    sorry
  exact ⟨⟨Nat.zero_le y, Nat.lt_of_sub_eq_zero h⟩, h⟩

end leila_distance_yards_l553_553775


namespace g_is_odd_l553_553763

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  sorry

end g_is_odd_l553_553763


namespace sandy_hourly_wage_l553_553809

theorem sandy_hourly_wage (x : ℝ)
    (h1 : 10 * x + 6 * x + 14 * x = 450) : x = 15 :=
by
    sorry

end sandy_hourly_wage_l553_553809


namespace largest_good_number_le_2011_l553_553931

def is_good_number (n : ℕ) : Prop :=
  ∃ m k : ℕ, n = 2*m + 1 ∧ n = 3*k + 3

theorem largest_good_number_le_2011 : ∃ n : ℕ, is_good_number n ∧ n ≤ 2011 ∧ ∀ k : ℕ, is_good_number k ∧ k ≤ 2011 → k ≤ n :=
  ∃ (n = 2007), is_good_number 2007 ∧ 2007 ≤ 2011 sorry

end largest_good_number_le_2011_l553_553931


namespace simplify_fraction_l553_553446

theorem simplify_fraction :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := 
sorry

end simplify_fraction_l553_553446


namespace convert_to_polar_l553_553619

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (r, θ)

theorem convert_to_polar (x y : ℝ) (hx : x = 8) (hy : y = 3 * Real.sqrt 3) :
  polar_coordinates x y = (Real.sqrt 91, Real.arctan (3 * Real.sqrt 3 / 8)) :=
by
  rw [hx, hy]
  simp [polar_coordinates]
  -- place to handle conversions and simplifications if necessary
  sorry

end convert_to_polar_l553_553619


namespace race_head_start_l553_553162

theorem race_head_start (v_A v_B : ℕ) (h : v_A = 4 * v_B) (d : ℕ) : 
  100 / v_A = (100 - d) / v_B → d = 75 :=
by
  sorry

end race_head_start_l553_553162


namespace problem_statement_l553_553039

-- Define the variable w
def w : ℂ := complex.ofReal (real.cos (3 * real.pi / 8)) + complex.I * complex.ofReal (real.sin (3 * real.pi / 8))

-- State the main theorem to prove
theorem problem_statement : 
  2 * ((w / (1 + w^3)) + (w^2 / (1 + w^6)) + (w^3 / (1 + w^9))) = -2 := 
sorry

end problem_statement_l553_553039


namespace sqrt_inequality_l553_553898

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
by
  sorry

end sqrt_inequality_l553_553898


namespace barrels_oil_difference_l553_553860

/--
There are two barrels of oil, A and B.
1. $\frac{1}{3}$ of the oil is poured from barrel A into barrel B.
2. $\frac{1}{5}$ of the oil is poured from barrel B back into barrel A.
3. Each barrel contains 24kg of oil after the transfers.

Prove that originally, barrel A had 6 kg more oil than barrel B.
-/
theorem barrels_oil_difference :
  ∃ (x y : ℝ), (y = 48 - x) ∧
  (24 = (2 / 3) * x + (1 / 5) * (48 - x + (1 / 3) * x)) ∧
  (24 = (48 - x + (1 / 3) * x) * (4 / 5)) ∧
  (x - y = 6) :=
by
  sorry

end barrels_oil_difference_l553_553860


namespace value_of_f_2011_l553_553390

noncomputable def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 7

theorem value_of_f_2011 (a b c : ℝ) (h : f a b c (-2011) = -17) : f a b c 2011 = 31 :=
by {
  sorry
}

end value_of_f_2011_l553_553390


namespace sum_x_midpoints_of_triangle_l553_553504

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end sum_x_midpoints_of_triangle_l553_553504


namespace sum_of_twenty_fives_to_thousand_l553_553533

theorem sum_of_twenty_fives_to_thousand :
  ∃ (a b c d e f g h i j k : ℕ),
    a + b + c + d + e + f + g + h + i + j + k = 1000 ∧
    a = 555 ∧ b = 55 ∧ c = 55 ∧ d = 55 ∧ e = 55 ∧ f = 55 ∧ g = 55 ∧ h = 55 ∧ i = 55 ∧ j = 55 ∧ k = 5 :=
begin
  sorry
end

end sum_of_twenty_fives_to_thousand_l553_553533


namespace find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l553_553536

theorem find_counterfeit_80_coins_in_4_weighings :
  ∃ f : Fin 80 → Bool, (∃ i, f i = true) ∧ (∃ i j, f i ≠ f j) := sorry

theorem min_weighings_for_n_coins (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, 3^(k-1) < n ∧ n ≤ 3^k := sorry

end find_counterfeit_80_coins_in_4_weighings_min_weighings_for_n_coins_l553_553536


namespace problem1_problem2_l553_553965

/-- Problem 1 -/
theorem problem1 (a b : ℝ) : (a^2 - b)^2 = a^4 - 2 * a^2 * b + b^2 :=
by
  sorry

/-- Problem 2 -/
theorem problem2 (x : ℝ) : (2 * x + 1) * (4 * x^2 - 1) * (2 * x - 1) = 16 * x^4 - 8 * x^2 + 1 :=
by
  sorry

end problem1_problem2_l553_553965


namespace bacteria_growth_rate_l553_553922

theorem bacteria_growth_rate (B G : ℝ) (h : B * G^16 = 2 * B * G^15) : G = 2 :=
by
  sorry

end bacteria_growth_rate_l553_553922
