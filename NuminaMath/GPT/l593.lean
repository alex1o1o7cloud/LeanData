import Mathlib

namespace intersection_of_sets_l593_593854

theorem intersection_of_sets :
  let A := {1, 2, 3, 4, 5}
  let B := {3, 5, 6}
  A ∩ B = {3, 5} :=
by
  let A := {1, 2, 3, 4, 5}
  let B := {3, 5, 6}
  show A ∩ B = {3, 5}
  sorry

end intersection_of_sets_l593_593854


namespace billy_sleep_total_l593_593249

def billySleepHours : ℕ → ℝ
| 1 := 6
| 2 := real.sqrt (6 * 8)
| 3 := billySleepHours 2 + 0.25 * billySleepHours 2
| 4 := real.cbrt (6 * billySleepHours 3)
| _ := 0

theorem billy_sleep_total : billySleepHours 1 + billySleepHours 2 + billySleepHours 3 + billySleepHours 4 = 25.32 := 
by
  -- Define individual sleep hours based on the conditions
  let h1 := 6
  let h2 := real.sqrt (6 * 8)
  let h3 := h2 + 0.25 * h2
  let h4 := real.cbrt (6 * h3)
  
  -- Confirm that the total sum is approximately 25.32
  have : h1 + h2 + h3 + h4 ≈ 25.32 := by sorry
  sorry

end billy_sleep_total_l593_593249


namespace intersection_A_B_l593_593839

def A (x : ℝ) : Prop := (x ≥ 2 ∧ x ≠ 3)
def B (x : ℝ) : Prop := (3 ≤ x ∧ x ≤ 5)
def C := {x : ℝ | 3 < x ∧ x ≤ 5}

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = C :=
  by sorry

end intersection_A_B_l593_593839


namespace find_A_and_height_l593_593260

noncomputable def triangle_properties (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ) :=
  a = 7 ∧ b = 8 ∧ cos_B = -1 / 7 ∧ 
  h = (a : ℝ) * (Real.sqrt (1 - (cos_B)^2)) * (1 : ℝ) / b / 2

theorem find_A_and_height : 
  ∀ (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ), 
  triangle_properties a b B cos_B h → 
  ∃ A h1, A = Real.pi / 3 ∧ h1 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end find_A_and_height_l593_593260


namespace parallel_AH_GE_l593_593217

variable (A B C D E F G H : Type)
variables [Parallelogram : Parallelogram A B C D]
          [PointEonBC : OnSegment E B C]
          [PointFonCD : OnSegment F C D]
          [LineDParallelFB : Parallel (LineThrough D F) (LineThrough B H)]
          [IntersectionG : Intersection (LineThrough D F) (LineThrough A B) = G]
          [IntersectionH : Intersection (LineThrough D E) (LineThrough B F) = H]

theorem parallel_AH_GE : Parallel (LineThrough A H) (LineThrough G E) :=
by
  sorry

end parallel_AH_GE_l593_593217


namespace function_ordering_l593_593257

def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c > 0

theorem function_ordering (a b c : ℝ) (h1:  quadratic_inequality a b c (-2))
                                  (h2:  quadratic_inequality a b c 4)
                                  (h3:  ∀ x, quadratic_inequality a b c x ↔ x < -2 ∨ x > 4)
                                  (h4:  a > 0)
                                  (h5:  a * (-2)^2 + b * (-2) + c = 0)
                                  (h6:  a * 4^2 + b * 4 + c = 0) :
  f(2) < f(-1) ∧ f(-1) < f(5) := 
sorry
where f : ℝ → ℝ := λ x, a * x^2 + b * x + c

end function_ordering_l593_593257


namespace find_a_l593_593133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then log x - a * x else -log (-x) + a * x

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : ∀ (x : ℝ), -2 < x ∧ x < 0 → f a x ≥ 1) : a = 2 :=
by
  sorry

end find_a_l593_593133


namespace trip_to_Atlanta_equals_Boston_l593_593384

def distance_to_Boston : ℕ := 840
def daily_distance : ℕ := 40
def num_days (distance : ℕ) (daily : ℕ) : ℕ := distance / daily
def distance_to_Atlanta (days : ℕ) (daily : ℕ) : ℕ := days * daily

theorem trip_to_Atlanta_equals_Boston :
  distance_to_Atlanta (num_days distance_to_Boston daily_distance) daily_distance = distance_to_Boston :=
by
  -- Here we would insert the proof.
  sorry

end trip_to_Atlanta_equals_Boston_l593_593384


namespace problem_l593_593281

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

noncomputable def affine_transformation (p : ℝ × ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * p.1, 2 * p.2)

def curve_C2_eq (x' y' : ℝ) : Prop :=
  (x' ^ 2) / 3 + (y' ^ 2) / 4 = 1

def line_l_eq (x y : ℝ) : Prop :=
  2 * x - y = 8

noncomputable def distance (θ : ℝ) : ℝ :=
  (|4 * Real.sin (θ - Real.pi / 3) + 8|) / Real.sqrt 5

theorem problem {
  -- Conditions and given information
  θ : ℝ,
  P : ℝ × ℝ := affine_transformation (curve_C1 θ)
} :
  (curve_C2_eq P.1 P.2) ∧ (line_l_eq 2 (P.1 * 2) <-> P = (-3/2, 1) ∧ distance θ = 12 * Real.sqrt 5 / 5)
 := sorry

end problem_l593_593281


namespace coeff_x3_l593_593540

noncomputable def M (n : ℕ) : ℝ := (5 * (1:ℝ) - (1:ℝ)^(1/2)) ^ n
noncomputable def N (n : ℕ) : ℝ := 2 ^ n

theorem coeff_x3 (n : ℕ) (h : M n - N n = 240) : 
  (M 3) = 150 := sorry

end coeff_x3_l593_593540


namespace common_tangents_count_l593_593351

-- Define the equation of the first circle
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + 1 = 0

-- Define the equation of the second circle
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*y - 1 = 0

-- State the theorem that we want to prove
theorem common_tangents_count : 
  (∃ (cp : ℝ) (center1 center2 : ℝ × ℝ) (rad1 rad2 : ℝ), 
    circle1 cp.1 cp.2 → 
    circle2 cp.1 cp.2 →
    center1 = (2, -1) → 
    rad1 = 2 → 
    center2 = (-2, 2) → 
    rad2 = 3 → 
    3 = 3) := 
sorry

end common_tangents_count_l593_593351


namespace snack_cost_is_five_l593_593294

-- Define the cost of one ticket
def ticket_cost : ℕ := 18

-- Define the total number of people
def total_people : ℕ := 4

-- Define the total cost for tickets and snacks
def total_cost : ℕ := 92

-- Define the unknown cost of one set of snacks
def snack_cost := 92 - 4 * 18

-- Statement asserting that the cost of one set of snacks is $5
theorem snack_cost_is_five : snack_cost = 5 := by
  sorry

end snack_cost_is_five_l593_593294


namespace find_monotonic_interval_l593_593547

-- Definitions for the conditions
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ :=
  sin (ω * x + φ) + sqrt 3 * cos (ω * x + φ)

def passes_through (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.fst = p.snd

def adjacent_zeros (f : ℝ → ℝ) (x1 x2 : ℝ) : Prop :=
  f x1 = 0 ∧ f x2 = 0 ∧ |x1 - x2| = 6

-- The theorem statement
theorem find_monotonic_interval (ω : ℝ) (φ : ℝ) (k : ℤ)
  (hω : ω > 0)
  (h_pass_through : passes_through (f ω φ) (1, 2))
  (h_zeros : ∃ x1 x2, adjacent_zeros (f ω φ) x1 x2) :
  ∃ interval_k : set ℝ, 
    interval_k = {x | -5 + 12 * k ≤ x ∧ x ≤ 1 + 12 * k} :=
sorry

end find_monotonic_interval_l593_593547


namespace part_a_part_b_l593_593287

/-
### Part (a)
- Question: Is it possible to weight two coins such that the probabilities of landing on "heads" and "tails" are different, but the probabilities of any of the combinations "tails, tails", "heads, tails", "heads, heads" are the same?

Conditions: 
- Two coins with probabilities for heads and tails denoted as ph, 1 - ph for the first coin, and qh, 1 - qh for the second coin.
- Probabilities for outcomes "Heads, Heads", "Tails, Tails", "Heads, Tails", "Tails, Heads".

Conclusion: 
- No, it is not possible.
-/
theorem part_a (ph qh : ℝ) : 
  (ph ≠ qh) ∧ 
  (ph * qh = (1 - ph) * (1 - qh)) ∧ 
  (ph * (1 - qh) = (1 - ph) * qh) → 
  False :=
by
  sorry

/-
### Part (b)
- Question: Is it possible to weight two dice such that the probability of rolling any sum from 2 to 12 is the same?

Conditions: 
- Two dice with probabilities for faces 1 through 6 denoted as p_i for the first die, and q_j for the second die.
- Probability calculations for sums ranging from 2 to 12.

Conclusion: 
- No, it is not possible.
-/
theorem part_b (p : fin 6 → ℝ) (q : fin 6 → ℝ) : 
  (∀ s ∈ (finset.range 11).map (λ i, i + 2), ∃ k : ℝ, 
    (∑ (i : fin 6) in finset.univ, (∑ (j : fin 6) in finset.univ, if i + j + 2 = s then p i * q j else 0)) = k) → 
  False :=
by
  sorry

end part_a_part_b_l593_593287


namespace find_m_and_other_root_l593_593528

theorem find_m_and_other_root (m x_2 : ℝ) :
  (∃ (x_1 : ℝ), x_1 = -1 ∧ x_1^2 + m * x_1 - 5 = 0) →
  m = -4 ∧ ∃ (x_2 : ℝ), x_2 = 5 ∧ x_2^2 + m * x_2 - 5 = 0 :=
by
  sorry

end find_m_and_other_root_l593_593528


namespace car_distance_l593_593749

theorem car_distance (time_original : ℕ) (time_fraction : ℚ) (speed : ℕ)
  (h1 : time_original = 6)
  (h2 : time_fraction = 3 / 2)
  (h3 : speed = 70) : 
  let D := speed * (time_original * time_fraction) in D = 630 := 
by
  sorry

end car_distance_l593_593749


namespace painted_cubes_l593_593086

theorem painted_cubes (n : ℕ) (h1 : 3 < n)
  (h2 : 6 * (n - 2)^2 = 12 * (n - 2)) :
  n = 4 := by
  sorry

end painted_cubes_l593_593086


namespace part1_part2_part2_monotonically_decreasing_l593_593225

variables {x : ℝ}

def a (x : ℝ) : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), Real.sin (x / 2))

-- Part (1): Prove that x = 0 if a(x) is parallel to b(x) given x in [0, π/2]
theorem part1 (h_par : ∃ k : ℝ, k ≠ 0 ∧ a x = (k * b x)) (hx : x ∈ Set.Icc 0 (Real.pi / 2)) : x = 0 := sorry

-- Part (2): Prove that f(x) = cos(x) and find intervals where cos(x) is decreasing
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem part2 : f x = Real.cos x := sorry

theorem part2_monotonically_decreasing : 
  ∀ k : ℤ, Set.Icc (2 * k * Real.pi) ((2 * k + 1) * Real.pi) ⊆ {x | (f' x = -Real.sin x) ∧ (0 > f' x)} := sorry

end part1_part2_part2_monotonically_decreasing_l593_593225


namespace max_distinct_three_digit_numbers_l593_593266

theorem max_distinct_three_digit_numbers :
  let count := ∑ z in {3, 4, 5, 6, 7, 8, 9}, 10 in
  count = 70 :=
by
  sorry

end max_distinct_three_digit_numbers_l593_593266


namespace total_basketballs_l593_593630

def luccaBalls : Nat := 100
def lucienBalls : Nat := 200
def leticiaBalls : Nat := 150

def luccaBasketballPercent : ℝ := 0.10
def lucienBasketballPercent : ℝ := 0.20
def leticiaBasketballPercent : ℝ := 0.15

def luccaBasketballs : Nat := Nat.floor (luccaBasketballPercent * luccaBalls)
def lucienBasketballs : Nat := Nat.floor (lucienBasketballPercent * lucienBalls)
def leticiaBasketballs : Nat := Nat.floor (leticiaBasketballPercent * leticiaBalls)

theorem total_basketballs :
  luccaBasketballs + lucienBasketballs + leticiaBasketballs = 72 :=
by
  sorry

end total_basketballs_l593_593630


namespace problem_series_sum_l593_593082

noncomputable def series_sum : ℝ := ∑' n : ℕ, (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem problem_series_sum :
  series_sum = 1 / 200 :=
sorry

end problem_series_sum_l593_593082


namespace secret_known_on_sunday_l593_593324

-- Define the geometric series condition
def secret_geometric_series (total_students : ℕ) (days : ℕ) : Prop :=
  total_students = (∑ k in finset.range (days + 2), 3^k)

-- Define the specific day condition based on the number of students
def on_day_of_week (total_students : ℕ) (day : String) : Prop :=
  secret_geometric_series total_students 6 ∧ day = "Sunday"

theorem secret_known_on_sunday :
  on_day_of_week 1093 "Sunday" :=
sorry

end secret_known_on_sunday_l593_593324


namespace fib_100_mod_5_l593_593346

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end fib_100_mod_5_l593_593346


namespace train_length_proof_l593_593400

noncomputable def length_of_train : ℝ := 450.09

theorem train_length_proof
  (speed_kmh : ℝ := 60)
  (time_s : ℝ := 27) :
  (speed_kmh * (5 / 18) * time_s = length_of_train) :=
by
  sorry

end train_length_proof_l593_593400


namespace power_function_through_point_has_specific_form_l593_593884

variable (f : ℝ → ℝ)

theorem power_function_through_point_has_specific_form (h : f = λ x, x^(1/2)) (H : f 2 = Real.sqrt 2) : 
  ∀ x, f x = x^(1/2) := 
by
  sorry

end power_function_through_point_has_specific_form_l593_593884


namespace prob_statement_l593_593915

noncomputable def log (x : ℝ) := Real.log x

variables (a b : ℝ)

-- Given conditions
axiom h1 : a = log 64
axiom h2 : b = log 25

-- Proposition to prove
theorem prob_statement : 8^(a/b) + 5^(b/a) = 89 :=
sorry

end prob_statement_l593_593915


namespace geom_seq_preserving_functions_l593_593096

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def geom_seq_preserving (f : ℝ → ℝ) : Prop :=
  ∀ {a : ℕ → ℝ}, geometric_sequence a → geometric_sequence (λ n, f (a n))

theorem geom_seq_preserving_functions :
  geom_seq_preserving (λ x, x ^ 2) ∧ geom_seq_preserving (λ x, Real.sqrt (abs x)) ∧ 
  ¬ geom_seq_preserving (λ x, 2 ^ x) ∧ ¬ geom_seq_preserving (λ x, Real.log (abs x)) :=
by
  sorry

end geom_seq_preserving_functions_l593_593096


namespace perfect_squares_between_50_and_250_l593_593233

theorem perfect_squares_between_50_and_250 : 
  (card { n : ℕ | 50 ≤ n^2 ∧ n^2 ≤ 250 }) = 8 := 
by
  sorry

end perfect_squares_between_50_and_250_l593_593233


namespace count_valid_n_l593_593802

-- Define the criteria for n
def valid_n (n : ℕ) : Prop :=
  ∃ m k : ℕ, n = 2^m * 5^k ∧ n ≤ 200 ∧ (1 / n : ℚ) * 100 % 10 ≠ 0

-- State the theorem
theorem count_valid_n : {n : ℕ | valid_n n}.toFinset.card = 6 := by
  sorry

end count_valid_n_l593_593802


namespace min_employees_hired_l593_593451

theorem min_employees_hired 
  (|A| : Nat) (|B| : Nat) (|A ∩ B| : Nat)
  (hA : |A| = 120) 
  (hB : |B| = 95) 
  (hAB : |A ∩ B| = 40) 
  : |A ∪ B| = 175 := by
  -- Apply the principle of inclusion-exclusion
  have h : |A ∪ B| = |A| + |B| - |A ∩ B| := sorry
  -- Substitute the given values
  rw [hA, hB, hAB] at h
  -- Simplify the expression
  exact h

end min_employees_hired_l593_593451


namespace ab_ac_plus_bc_range_l593_593984

theorem ab_ac_plus_bc_range (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ (k : ℝ), k ≤ 0 ∧ k = ab + ac + bc :=
sorry

end ab_ac_plus_bc_range_l593_593984


namespace pentagon_area_ratio_l593_593614

-- Define the conditions in the problem
variables (A B C D E : Point)
variables (AB BC CE AD DE : ℝ)
variables (angle_ABC : ℝ)

-- Assume the given conditions
def pentagon_properties : Prop := 
  AB = 3 ∧ 
  BC = 5 ∧ 
  DE = 15 ∧ 
  angle_ABC = 120 ∧ 
  (AB ∥ CE) ∧ 
  (BC ∥ AD) ∧ 
  (AC ∥ DE)

-- Define the function modeling the problem statement
noncomputable def find_m_n_sum (m n : ℕ) : Prop := 
  m + n = 26 ∧ gcd m n = 1

-- Prove the final statement using the conditions
theorem pentagon_area_ratio (m n : ℕ) 
  (h1 : pentagon_properties)
  (h2 : ∃ m n, find_m_n_sum m n) :
  m + n = 26 := 
by
  sorry

end pentagon_area_ratio_l593_593614


namespace distance_ranges_l593_593200

noncomputable def hyperbola_eqn (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

def asymptote_l1 (x y : ℝ) : Prop :=
  y = (1 / 2) * x

def asymptote_l2 (x y : ℝ) : Prop :=
  y = -(1 / 2) * x

noncomputable def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_eqn P.1 P.2

noncomputable def distance_to_line (P : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  if line P.1 P.2 then abs (P.1 - 2 * P.2) / sqrt 5 else abs (P.1 + 2 * P.2) / sqrt 5

def range (d : ℝ) (I : Set ℝ) : Prop :=
  d ∈ I

theorem distance_ranges (P : ℝ × ℝ) (d1 : ℝ) :
  point_on_hyperbola P →
  range (distance_to_line P asymptote_l1) (Set.Icc (1/2) 1) →
  range (distance_to_line P asymptote_l2) (Set.Icc (4/5) (8/5)) :=
sorry

end distance_ranges_l593_593200


namespace concurrency_of_AA_l593_593583

open Classical

variable {A B C D E F O A' B' C' : Type}
variable [PlaneGeometry A B C D E F O A' B' C']

noncomputable theory

-- Given conditions
def incircle_tangent (ABC_Δ : Triangle A B C) (O_circle : Circle O) : Prop :=
  tangent_at O_circle ABC_Δ.sides.BC D ∧
  tangent_at O_circle ABC_Δ.sides.CA E ∧
  tangent_at O_circle ABC_Δ.sides.AB F

def rays_intersect (O_center : O) (DEF_Δ : Triangle D E F) : Prop :=
  intersects D O E F A' ∧
  intersects E O F D B' ∧
  intersects F O D E C'

-- Main theorem
theorem concurrency_of_AA'_BB'_CC' (ABC_Δ : Triangle A B C) (O_circle : Circle O)
  (inc : incircle_tangent ABC_Δ O_circle) (ray_inter : rays_intersect O_circle.center ABC_Δ) :
  concurrent (line_through A A') (line_through B B') (line_through C C') :=
sorry

end concurrency_of_AA_l593_593583


namespace range_of_m_l593_593538

-- Define the discriminant of the quadratic equation
def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b * b - 4 * a * c

-- Define the condition that the quadratic equation has two real roots
def has_two_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ quadratic_discriminant a b c ≥ 0

-- Statement of the proof problem
theorem range_of_m (m : ℝ) :
  has_two_real_roots (m - 1) (-real.sqrt (2 - m)) (-1/2) ↔ 0 ≤ m ∧ m ≤ 2 ∧ m ≠ 1 :=
sorry

end range_of_m_l593_593538


namespace rent_percentage_l593_593972

theorem rent_percentage (E : ℝ) (h1 : 0.2 * E) (h2 : 1.2 * E) (h3 : 0.3 * (1.2 * E)) : 
  (0.36 * E) / (0.2 * E) * 100 = 180 := by
  -- Proof goes here
  sorry

end rent_percentage_l593_593972


namespace find_m_l593_593187

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l593_593187


namespace intersection_point_l593_593389

theorem intersection_point : ∃ x y : ℝ, 2 * x + 3 * y - 7 = 0 ∧ 5 * x - y - 9 = 0 ∧ x = 2 ∧ y = 1 :=
by
  use 2
  use 1
  split
  { -- First equation: 2x + 3y - 7 = 0
    rw [mul_assoc, show 2 * 2 = 4 by norm_num, show 3 * 1 = 3 by norm_num]
    norm_num }
  split
  { -- Second equation: 5x - y - 9 = 0
    rw [mul_assoc, show 5 * 2 = 10 by norm_num, show -1 * 1 = -1 by norm_num]
    norm_num }
  split
  { refl } -- x = 2
  { refl } -- y = 1

end intersection_point_l593_593389


namespace min_blocks_for_wall_l593_593746

theorem min_blocks_for_wall (len height : ℕ) (blocks : ℕ → ℕ → ℕ)
  (block_1 : ℕ) (block_2 : ℕ) (block_3 : ℕ) :
  len = 120 → height = 9 →
  block_3 = 1 → block_2 = 2 → block_1 = 3 →
  blocks 5 41 + blocks 4 40 = 365 :=
by
  sorry

end min_blocks_for_wall_l593_593746


namespace good_numbers_up_to_2006_l593_593674

def is_good_number (k : ℕ) : Prop :=
  k = 1 ∨ k ≥ 4 ∧ ¬(k = 2 ∨ k = 3 ∨ k = 5)

def count_good_numbers (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter is_good_number |>.length

theorem good_numbers_up_to_2006 : count_good_numbers 2006 = 2003 :=
by
  sorry

end good_numbers_up_to_2006_l593_593674


namespace intersection_M_N_l593_593223

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l593_593223


namespace power_function_value_l593_593255

theorem power_function_value (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x ^ (1 / 2)) (H : f 9 = 3) : f 25 = 5 :=
by
  sorry

end power_function_value_l593_593255


namespace sqrt_product_simplification_l593_593466

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end sqrt_product_simplification_l593_593466


namespace min_value_of_f_l593_593483

noncomputable def f (x : ℝ) : ℝ := real.sin (2 * x - real.pi / 4)

theorem min_value_of_f :
  infi (f '' set.Icc 0 (real.pi / 2)) = -real.sqrt 2 / 2 :=
begin
  sorry
end

end min_value_of_f_l593_593483


namespace count_proper_subset_pairs_l593_593509

open Finset

-- Define the set S = {1, 2, 3, 4, 5, 6}
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- The main theorem statement
theorem count_proper_subset_pairs : 
  let number_of_pairs := ∑ B in S.powerset.filter (λ b, b ≠ ∅),
                           2^(B.card) - 1
  in
  number_of_pairs = 665 :=
by
  sorry

end count_proper_subset_pairs_l593_593509


namespace total_number_of_ways_is_144_l593_593644

def count_ways_to_place_letters_on_grid : Nat :=
  16 * 9

theorem total_number_of_ways_is_144 :
  count_ways_to_place_letters_on_grid = 144 :=
  by
    sorry

end total_number_of_ways_is_144_l593_593644


namespace probability_of_square_product_l593_593711

theorem probability_of_square_product :
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9 -- (1,1), (1,4), (2,2), (4,1), (3,3), (9,1), (4,4), (5,5), (6,6)
  favorable_outcomes / total_outcomes = 1 / 8 :=
by
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9
  have h1 : favorable_outcomes / total_outcomes = 1 / 8 := sorry
  exact h1

end probability_of_square_product_l593_593711


namespace problem_a_problem_b_problem_c_l593_593971

-- Problem (a)
def joaozinho_problem_a (n : ℕ) : Prop :=
  let digits := n.digits 10
  n < 10000 ∧ n ≥ 1000 ∧ (digits.head = (digits.tail.foldl (+) 0))

theorem problem_a : ∃ n, joaozinho_problem_a n ∧ n = 1001 :=
sorry

-- Problem (b)
def joaozinho_problem_b (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ d ∈ digits, d ≠ 0) ∧ (digits.head = (digits.tail.foldl (+) 0))

theorem problem_b : ∃ n, joaozinho_problem_b n ∧ n = 1111111119 :=
sorry

-- Problem (c)
def joaozinho_problem_c (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.nodup) ∧ (digits.head = (digits.tail.foldl (+) 0))

theorem problem_c : ∃ n, joaozinho_problem_c n ∧ n = 62109 :=
sorry

end problem_a_problem_b_problem_c_l593_593971


namespace triangle_right_angled_l593_593931

theorem triangle_right_angled 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (hABC : ∀ {A B C : ℝ}, A + B + C = π) 
  (h_sides : ∀ {a b c : ℝ}, ∃ A B C, a / sin A = b / sin B ∧ b / sin B = c / sin C)
  (h_parallel : bx + y * cos A + cos B = 0 ∧ ax + y * cos B + cos A = 0 → ∀ y, y ∈ ℝ) :
  
  a^2 + b^2 = c^2 :=
sorry

end triangle_right_angled_l593_593931


namespace part1_part2_l593_593341

-- Definitions for permutations and combinations
def A (n k : ℕ) := Nat.fact n / Nat.fact (n - k)
def C (n k : ℕ) := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

theorem part1 (n : ℕ) (h : A (2 * n) 3 = 10 * A n 3) : n = 8 := sorry

theorem part2 (n m : ℕ) : m * C n m = n * C (n - 1) (m - 1) := sorry

end part1_part2_l593_593341


namespace purchase_price_of_grinder_l593_593969

theorem purchase_price_of_grinder (G : ℝ) (H : 0.95 * G + 8800 - (G + 8000) = 50) : G = 15000 := 
sorry

end purchase_price_of_grinder_l593_593969


namespace remainder_of_power_modulo_l593_593718

theorem remainder_of_power_modulo : (3^2048) % 11 = 5 := by
  sorry

end remainder_of_power_modulo_l593_593718


namespace min_value_of_c_l593_593584

theorem min_value_of_c (a b c : ℝ) (h1 : a + b = 2) (h2 : ∠C = 120) :
  c ≥ √3 := by
  sorry

end min_value_of_c_l593_593584


namespace sin_315_eq_neg_sqrt2_div_2_l593_593009

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593009


namespace range_of_a_l593_593125

open Set

theorem range_of_a (a : ℝ) : (Iic 0 ∪ Ioi a = univ) → a ≤ 0 :=
begin
  -- skip the proof
  sorry
end

end range_of_a_l593_593125


namespace chord_ratio_l593_593555

noncomputable def line (k : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ x, p = (x, k * x) }

def circle1 : set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + p.2^2 = 1 }

def circle2 : set (ℝ × ℝ) :=
  { p | (p.1 - 3)^2 + p.2^2 = 1 }

theorem chord_ratio (k : ℝ) (h : 0 < k)
  (ratio_cond : 
    (λ l1 l2, l1 / l2 = 3)
      (2 * real.sqrt (1 - k^2 / (k^2 + 1)) / real.sqrt (k^2 + 1))
      (2 * real.sqrt (1 - 9 * k^2 / (k^2 + 1)) / real.sqrt (k^2 + 1))) :
  k = 1 / 3 ∨ k = -1 / 3 :=
sorry

end chord_ratio_l593_593555


namespace find_a_when_extremum_at_1_tangent_line_at_1_when_a_1_range_of_a_when_decreasing_l593_593550

-- Part (Ⅰ): If f(x) has an extremum at x=1, find the value of a
theorem find_a_when_extremum_at_1 (a : ℝ) : (∀ x : ℝ, f x = x + a / x + log x) →
  (∀ x : ℝ, f' x = 1 - a / x^2 + 1 / x) →
  f' 1 = 0 → a = 2 :=
by
  delta f'
  sorry

-- Part (Ⅱ): When a = 1, find the equation of the tangent line to the function f(x) at the point (1,f(1)).
theorem tangent_line_at_1_when_a_1 (f : ℝ → ℝ) (x : ℝ) (a : ℝ) : 
  a = 1 → 
  f x = x + 1 / x + log x →
  f 1 = 2 →
  (∀ x, f' x = 1 - 1 / x^2 + 1 / x) →
  (∀ x : ℝ, tangent_line f 1 (f 1)) = "y = x + 1" := 
by
  delta tangent_line
  sorry

-- Part (Ⅲ): If f(x) is monotonically decreasing in the interval (1,2), find the range of values for a.
theorem range_of_a_when_decreasing (a : ℝ) : 
  (∀ x : ℝ, f x = x + a / x + log x) →
  (∀ x : ℝ, f' x = 1 - a / x^2 + 1 / x) →
  (∀ x, 1 < x ∧ x < 2 → f' x ≤ 0) → a ≥ 6 :=
by
  delta f'
  sorry

end find_a_when_extremum_at_1_tangent_line_at_1_when_a_1_range_of_a_when_decreasing_l593_593550


namespace isosceles_trapezoid_AE_squared_l593_593273

variables (A B C D E F G : Point)
variables [IsoscelesTrapezoid ABCD]
variables (h_parallel : ∥ AB CD)
variables (h_AB : dist A B = 6)
variables (h_CD : dist C D = 14)
variables (h_angle : angle A E C = 90)
variables (h_CE_CB : dist C E = dist C B)

/-- Given the conditions of the isosceles trapezoid ABCD, 
we need to prove that AE^2 = 84. -/
theorem isosceles_trapezoid_AE_squared :
  (dist A E) ^ 2 = 84 :=
sorry

end isosceles_trapezoid_AE_squared_l593_593273


namespace incorrect_statement_l593_593597

def geom_seq (a r : ℝ) : ℕ → ℝ
| 0       => a
| (n + 1) => r * geom_seq a r n

theorem incorrect_statement
  (a : ℝ) (r : ℝ) (S6 : ℝ)
  (h1 : r = 1 / 2)
  (h2 : S6 = a * (1 - (1 / 2) ^ 6) / (1 - 1 / 2))
  (h3 : S6 = 378) :
  geom_seq a r 2 / S6 ≠ 1 / 8 :=
by 
  have h4 : a = 192 := by sorry
  have h5 : geom_seq 192 (1 / 2) 2 = 192 * (1 / 2) ^ 2 := by sorry
  exact sorry

end incorrect_statement_l593_593597


namespace graph_of_equation_pair_of_lines_l593_593804

theorem graph_of_equation_pair_of_lines (x y : ℝ) : x^2 - 9 * y^2 = 0 ↔ (x = 3 * y ∨ x = -3 * y) :=
by
  sorry

end graph_of_equation_pair_of_lines_l593_593804


namespace find_x2000_l593_593410

-- Given recurrence relation and initial condition
def recurrence_relation (x : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ n > 0, x (n + 1) = (1 + (2 : ℤ) / (n : ℤ)) * x n + (4 : ℤ) / (n : ℤ)

def initial_condition (x : ℕ → ℤ) : Prop :=
  x 1 = -1

-- We aim to prove that x 2000 = 2000998
theorem find_x2000 (x : ℕ → ℤ) :
  recurrence_relation x ∧ initial_condition x → x 2000 = 2000998 :=
by
  sorry

end find_x2000_l593_593410


namespace common_ratio_of_gp_l593_593403

variables {a r : ℝ}

-- Definition of the sum of the first n terms of a G.P.
def S (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

-- Given condition: The ratio of the sum of the first 6 terms to the sum of the first 3 terms is 126
def condition (a r : ℝ) := (S a r 6) / (S a r 3) = 126

-- Theorem stating the common ratio r must be 5 given the condition
theorem common_ratio_of_gp (a : ℝ) (h : condition a r) : r = 5 :=
sorry

end common_ratio_of_gp_l593_593403


namespace crow_speed_l593_593733

-- Definitions for the problem conditions
def nest_to_ditch_distance : ℝ := 400 -- in meters
def round_trip_count : ℕ := 15
def total_time_hours : ℝ := 1.5

-- The statement of the proof problem
theorem crow_speed :
  let distance_per_round_trip := 2 * nest_to_ditch_distance in
  let total_distance_km := (round_trip_count * distance_per_round_trip) / 1000 in
  total_distance_km / total_time_hours = 8 :=
by
  sorry

end crow_speed_l593_593733


namespace solve_x_l593_593655

theorem solve_x :
  (1 / 4 - 1 / 6) = 1 / (12 : ℝ) :=
by sorry

end solve_x_l593_593655


namespace annual_profit_growth_rate_l593_593196

variable (a : ℝ)

theorem annual_profit_growth_rate (ha : a > -1) : 
  (1 + a) ^ 12 - 1 = (1 + a) ^ 12 - 1 := 
by 
  sorry

end annual_profit_growth_rate_l593_593196


namespace planes_perpendicular_l593_593127

variables (α β : plane) (m n : line)

def perpendicular (x y : Type) : Prop := sorry -- Definition placeholder

axiom plane_diff : α ≠ β
axiom line_diff : m ≠ n
axiom m_perp_to_alpha : perpendicular m α
axiom n_perp_to_beta : perpendicular n β
axiom m_perp_to_n : perpendicular m n

theorem planes_perpendicular :
  perpendicular α β :=
sorry

end planes_perpendicular_l593_593127


namespace triangle_area_ratio_l593_593775

theorem triangle_area_ratio (a n m : ℕ) (h1 : 0 < a) (h2 : 0 < n) (h3 : 0 < m) :
  let area_A := (a^2 : ℝ) / (4 * n^2)
  let area_B := (a^2 : ℝ) / (4 * m^2)
  (area_A / area_B) = (m^2 : ℝ) / (n^2 : ℝ) :=
by
  sorry

end triangle_area_ratio_l593_593775


namespace sin_315_eq_neg_sqrt2_div_2_l593_593006

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593006


namespace option_D_is_div_by_9_l593_593728

-- Define the parameters and expressions
def A (k : ℕ) : ℤ := 6 + 6 * 7^k
def B (k : ℕ) : ℤ := 2 + 7^(k - 1)
def C (k : ℕ) : ℤ := 2 * (2 + 7^(k + 1))
def D (k : ℕ) : ℤ := 3 * (2 + 7^k)

-- Define the main theorem to prove that D is divisible by 9
theorem option_D_is_div_by_9 (k : ℕ) (hk : k > 0) : D k % 9 = 0 :=
sorry

end option_D_is_div_by_9_l593_593728


namespace tangent_point_l593_593492

noncomputable def exp (x : ℝ) : ℝ := Real.exp x

theorem tangent_point : ∃ x y : ℝ, y = exp x ∧ y' = exp x ∧ (y - exp x = exp x * (0 - x)) ∧ (0, 0) ∈ Set.Univ ∧ (x, y) = (1, exp 1) := 
by
  sorry

end tangent_point_l593_593492


namespace find_m_l593_593183

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l593_593183


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593020

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593020


namespace find_x_in_data_set_l593_593689

noncomputable def mean (data : List ℝ) : ℝ :=
data.sum / (data.length : ℝ)

def median (data : List ℝ) : ℝ :=
let sorted_data := data.qsort (≤)
if data.length % 2 = 1 then
  sorted_data.nthLe (data.length / 2) (by simp)
else
  (sorted_data.nthLe (data.length / 2 - 1) (by simp) + 
   sorted_data.nthLe (data.length / 2) (by simp)) / 2

def mode (data : List ℝ) : ℝ :=
(data.groupBy id).values.maxBy (λ xs => xs.length) |>.head

theorem find_x_in_data_set :
  ∃ x : ℝ, mean [60, 100, x, 40, 50, 200, 90] = x ∧
           median [60, 100, x, 40, 50, 200, 90] = x ∧
           mode [60, 100, x, 40, 50, 200, 90] = x ∧
           x = 90 :=
begin
  use 90,
  sorry
end

end find_x_in_data_set_l593_593689


namespace quadratic_function_properties_l593_593838

noncomputable def quadratic_function := 
  -((8:ℝ) / (3:ℝ)) * (Polynomial.X - 1) * (Polynomial.X - 5)

theorem quadratic_function_properties :
  (quadratic_function.eval 1 = 0) ∧ 
  (quadratic_function.eval 5 = 0) ∧ 
  (quadratic_function.eval 2 = 8) :=
by
  sorry

end quadratic_function_properties_l593_593838


namespace smallest_median_of_list_l593_593338

theorem smallest_median_of_list : ∀ (l : List ℕ), 
  (l = [3, 5, 7, 9, 11, 13]) → 
  ∀ (remaining : List ℕ), 
  (remaining.length = 6 ∧ ∀ x ∈ remaining, x < 20) → 
  let full_list := (l ++ remaining).qsort (λ x y => x < y)
  in (full_list.nth (full_list.length / 2 - 1) + full_list.nth (full_list.length / 2)) / 2 = 6 :=
by
  intros l hl remaining h_remaining
  let full_list := (l ++ remaining).qsort (λ x y => x < y)
  have h_length : full_list.length = 12 := by
    rw [← List.length_append, hl]
    exact Eq.symm (by simp [List.length_append, List.length])
  have h6 : full_list.nth 5 = 5 := sorry
  have h7 : full_list.nth 6 = 7 := sorry
  rw [h6, h7]
  exact Eq.refl 6
  sorry

end smallest_median_of_list_l593_593338


namespace original_square_perimeter_proof_l593_593445

noncomputable section

-- Definitions of conditions
def width (rect : ℝ) : ℝ := rect
def length (rect : ℝ) : ℝ := 4 * rect
def form_П (width : ℝ) (length : ℝ) : ℝ := 
  4 * length + 6 * width

def original_square_perimeter (width : ℝ) : ℝ := 4 * (4 * width)

-- Given conditions
variable (rect_width : ℝ)
variable (perimeter_П : ℝ)

-- Assumption from the problem's condition
axiom original_perimeter_Π_is_56 : form_П rect_width (length rect_width) = 56

-- Proof statement
theorem original_square_perimeter_proof : 
  original_square_perimeter rect_width = 32 :=
  sorry

end original_square_perimeter_proof_l593_593445


namespace proof_friendly_pairs_l593_593195

open Classical

noncomputable theory

variables {V : Type*} [Fintype V] [DecidableEq V]
variables (G : SimpleGraph V) (n : ℕ)
variables [h1 : Even n] (h_pos : 0 < n) (h_nodes : Fintype.card V = n)
variables (h_edges : G.edge_finset.card = n * n / 4)

def friendly (v₁ v₂ : V) : Prop :=
  ∃ z : V, G.Adj v₁ z ∧ G.Adj v₂ z

theorem proof_friendly_pairs : ∃ (m : ℕ), m ≥ 2 * Nat.choose (n / 2) 2 :=
sorry

end proof_friendly_pairs_l593_593195


namespace perimeter_shaded_region_l593_593598

theorem perimeter_shaded_region (O P Q : ℝ) (radius : ℝ) (hOQ : O = 0) (hOP : P = -radius) (hPQ : Q = radius) (h_radius : radius = 7) :
  let circumference := 2 * Real.pi * radius,
      arc_length := 3/4 * circumference in
  2 * radius + arc_length = 14 + (21/2) * Real.pi :=
by
  rw [h_radius, circumference, arc_length]
  sorry

end perimeter_shaded_region_l593_593598


namespace count_true_statements_l593_593454

def star (n : ℕ) : ℚ := 1 / (n ^ 2)

theorem count_true_statements : 
  let i := (star 4 + star 9 = star 13),
      ii := (star 8 - star 1 = star 7),
      iii := (star 3 * star 6 = star 18),
      iv := (star 16 / star 4 = star 4) in
  (ite i 1 0 + 
   ite ii 1 0 + 
   ite iii 1 0 + 
   ite iv 1 0) = 2 :=
by {
  -- insert the proof here
  sorry
}

end count_true_statements_l593_593454


namespace choose_5_person_committee_l593_593943

theorem choose_5_person_committee : nat.choose 12 5 = 792 := 
by
  sorry

end choose_5_person_committee_l593_593943


namespace only_function_B_has_inverse_l593_593091

-- Definitions based on the problem conditions
def function_A (x : ℝ) : ℝ := 3 - x^2 -- Parabola opening downwards with vertex at (0,3)
def function_B (x : ℝ) : ℝ := x -- Straight line with slope 1 passing through (0,0) and (1,1)
def function_C (x y : ℝ) : Prop := x^2 + y^2 = 4 -- Circle centered at (0,0) with radius 2

-- Theorem stating that only function B has an inverse
theorem only_function_B_has_inverse :
  (∀ y : ℝ, ∃! x : ℝ, function_B x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, function_A x = y) ∧
  (¬∀ y : ℝ, ∃! x : ℝ, ∃ y1 y2 : ℝ, function_C x y1 ∧ function_C x y2 ∧ y1 ≠ y2) :=
  by 
  sorry -- Proof not required

end only_function_B_has_inverse_l593_593091


namespace students_behind_minyoung_l593_593743

-- Definition of the initial conditions
def total_students : ℕ := 35
def students_in_front_of_minyoung : ℕ := 27

-- The question we want to prove
theorem students_behind_minyoung : (total_students - (students_in_front_of_minyoung + 1) = 7) := 
by 
  sorry

end students_behind_minyoung_l593_593743


namespace subset_sum_divisible_l593_593624

theorem subset_sum_divisible (k : ℕ) (n : ℕ) (a : ℕ → ℕ) 
  (h: cardinal.mk {i | i < n ∧ a i % (n + k) ∈ finset.univ} ≥ 2 * k) :
  ∃ s : finset ℕ, (s ⊆ finset.range n) ∧ ((finset.sum s a) % (n + k) = 0) :=
sorry

end subset_sum_divisible_l593_593624


namespace complex_number_solution_l593_593921

theorem complex_number_solution (z : ℂ) (h : z * complex.I = 2 / (1 + complex.I)) : z = -1 - complex.I :=
sorry

end complex_number_solution_l593_593921


namespace find_smaller_number_l593_593407

theorem find_smaller_number (x y : ℕ) (h₁ : y - x = 1365) (h₂ : y = 6 * x + 15) : x = 270 :=
sorry

end find_smaller_number_l593_593407


namespace tangent_line_from_point_to_circle_minimum_distance_from_line_to_tangent_l593_593136

noncomputable def tangent_lines_of_circle (P : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ → Prop) :=
  { l | ∃ (k : ℝ), l = (λ (x y : ℝ), x = 3) ∨ (l = (λ (x y : ℝ), 5 * x - 12 * y + 9 = 0)) }

theorem tangent_line_from_point_to_circle 
  (P : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ)
  (hP : P = (3, 2))
  (hc : c = (1, -1))
  (hr : r = 2) :
  tangent_lines_of_circle (3, 2) (1, -1) 2 ∈
  (λ (l : ℝ × ℝ → Prop), l (3, 2)) :=
begin
  sorry
end

-- Second theorem

noncomputable def minimum_length_QM (c : ℝ × ℝ) (r : ℝ) (line : ℝ × ℝ → Prop) :=
  ∀ Q M, line Q → tangent_lines_of_circle Q c r M → |Q - M| = sqrt 5

theorem minimum_distance_from_line_to_tangent 
  (c : ℝ × ℝ) (r : ℝ) (m : ℝ × ℝ → Prop)
  (hc : c = (1, -1))
  (hr : r = 2)
  (hm : m = (λ (x y : ℝ), 3 * x - 4 * y + 8 = 0)) :
  minimum_length_QM (1, -1) 2 (λ (x y : ℝ), 3 * x - 4 * y + 8 = 0) :=
begin
  sorry
end

end tangent_line_from_point_to_circle_minimum_distance_from_line_to_tangent_l593_593136


namespace find_integer_mod_l593_593822

theorem find_integer_mod (n : ℤ) (h1 : n ≡ 8173 [MOD 15]) (h2 : 0 ≤ n ∧ n ≤ 14) : n = 13 :=
sorry

end find_integer_mod_l593_593822


namespace repeating_decimal_eq_l593_593501

-- Defining the repeating decimal as a hypothesis
def repeating_decimal : ℚ := 0.7 + 3/10^2 * (1/(1 - 1/10))
-- We will prove this later by simplifying the fraction
def expected_fraction : ℚ := 11/15

theorem repeating_decimal_eq : repeating_decimal = expected_fraction := 
by
  sorry

end repeating_decimal_eq_l593_593501


namespace remainder_when_x_minus_y_div_18_l593_593226

variable (k m : ℤ)
variable (x y : ℤ)
variable (h1 : x = 72 * k + 65)
variable (h2 : y = 54 * m + 22)

theorem remainder_when_x_minus_y_div_18 :
  (x - y) % 18 = 7 := by
sorry

end remainder_when_x_minus_y_div_18_l593_593226


namespace jerry_reaches_four_l593_593295

/-- Jerry starts at 0 on the real number line. He tosses a fair coin 8 times. When he gets heads, he moves 1 unit in the positive direction; when he gets tails, he moves 1 unit in the negative direction. Prove that the probability that he reaches 4 at some time during this process is 7/32, and then show that the sum of the numerator and denominator of this probability in reduced form is 39. -/
theorem jerry_reaches_four :
  let a := 7
  let b := 32
  (∑ i in Finset.range 9, if (i - (8 - i)) = 4 then Nat.choose 8 i else 0) =
  7 / 32 ∧ (a + b = 39) := 
sorry

end jerry_reaches_four_l593_593295


namespace percentage_of_Luccas_balls_are_basketballs_l593_593993

-- Defining the variables and their conditions 
variables (P : ℝ) (Lucca_Balls : ℕ := 100) (Lucien_Balls : ℕ := 200)
variable (Total_Basketballs : ℕ := 50)

-- Condition that Lucien has 20% basketballs
def Lucien_Basketballs := (20 / 100) * Lucien_Balls

-- We need to prove that percentage of Lucca's balls that are basketballs is 10%
theorem percentage_of_Luccas_balls_are_basketballs :
  (P / 100) * Lucca_Balls + Lucien_Basketballs = Total_Basketballs → P = 10 :=
by
  sorry

end percentage_of_Luccas_balls_are_basketballs_l593_593993


namespace eight_letter_good_words_l593_593095

-- Definition of a good word sequence (only using A, B, and C)
inductive Letter
| A | B | C

-- Define the restriction condition for a good word
def is_valid_transition (a b : Letter) : Prop :=
  match a, b with
  | Letter.A, Letter.B => False
  | Letter.B, Letter.C => False
  | Letter.C, Letter.A => False
  | _, _ => True

-- Count the number of 8-letter good words
def count_good_words : ℕ :=
  let letters := [Letter.A, Letter.B, Letter.C]
  -- Initial 3 choices for the first letter
  let first_choices := letters.length
  -- Subsequent 7 letters each have 2 valid previous choices
  let subsequent_choices := 2 ^ 7
  first_choices * subsequent_choices

theorem eight_letter_good_words : count_good_words = 384 :=
by
  sorry

end eight_letter_good_words_l593_593095


namespace skater_speeds_and_times_l593_593958

theorem skater_speeds_and_times (v : ℝ)
  (h1 : (v + 1/3) * 600 - v * 600 = 200)
  (h2 : 2 = 400 / v - 400 / (v + 1/3)) :
  v = 8 ∧ v + 1/3 = 8 + 1/3 ∧ 10000 / (v + 1/3) = 1200 ∧ 10000 / v = 1250 :=
by
  -- Prove that the speed of the second skater is 8 m/s
  have v_eq : v = 8, sorry,
  -- Prove that the speed of the first skater is 8 + 1/3 m/s
  have v1_eq : v + 1/3 = 8 + 1/3, sorry,
  -- Prove that the time taken by the first skater is 1200 seconds (20 minutes)
  have time1_eq : 10000 / (v + 1/3) = 1200, sorry,
  -- Prove that the time taken by the second skater is 1250 seconds (20 minutes 50 seconds)
  have time2_eq : 10000 / v = 1250, sorry,
  exact ⟨v_eq, v1_eq, time1_eq, time2_eq⟩

end skater_speeds_and_times_l593_593958


namespace scale_division_l593_593442

theorem scale_division (total_feet : ℕ) (inches_extra : ℕ) (part_length : ℕ) (total_parts : ℕ) :
  total_feet = 6 → inches_extra = 8 → part_length = 20 → 
  total_parts = (6 * 12 + 8) / 20 → total_parts = 4 :=
by
  intros
  sorry

end scale_division_l593_593442


namespace find_c_l593_593487

theorem find_c (c : ℝ) : 
  (∀ x, (cx^4 + 15x^3 - 5cx^2 - 45x + 55 = 0) → (x + 5 = 0)) → c = 3.19 :=
by
  sorry

end find_c_l593_593487


namespace sin_315_eq_neg_sqrt2_over_2_l593_593033

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593033


namespace price_white_stamp_l593_593654

variable (price_per_white_stamp : ℝ)

theorem price_white_stamp (simon_red_stamps : ℕ)
                          (peter_white_stamps : ℕ)
                          (price_per_red_stamp : ℝ)
                          (money_difference : ℝ)
                          (h1 : simon_red_stamps = 30)
                          (h2 : peter_white_stamps = 80)
                          (h3 : price_per_red_stamp = 0.50)
                          (h4 : money_difference = 1) :
    money_difference = peter_white_stamps * price_per_white_stamp - simon_red_stamps * price_per_red_stamp →
    price_per_white_stamp = 1 / 5 :=
by
  intros
  sorry

end price_white_stamp_l593_593654


namespace ben_maintenance_expenditure_l593_593456

variable (initial_balance cheque_to_supplier payment_from_debtor final_balance expenditure_on_maintenance : ℝ)

theorem ben_maintenance_expenditure :
  initial_balance = 2000 →
  cheque_to_supplier = 600 →
  payment_from_debtor = 800 →
  final_balance = 1000 →
  expenditure_on_maintenance = (initial_balance - cheque_to_supplier + payment_from_debtor) - final_balance →
  expenditure_on_maintenance = 1200 := 
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4] at h5,
  linarith
end

end ben_maintenance_expenditure_l593_593456


namespace domain_of_f_l593_593354

noncomputable def f (x: ℝ): ℝ := 1 / Real.sqrt (x - 2)

theorem domain_of_f:
  {x: ℝ | 2 < x} = {x: ℝ | f x = 1 / Real.sqrt (x - 2)} :=
by
  sorry

end domain_of_f_l593_593354


namespace max_integer_value_l593_593916

-- We define the given expression and the conditions
def expr (y : ℝ) : ℝ := (4*y^2 + 8*y + 19) / (4*y^2 + 8*y + 5)

-- The theorem states that for all real y, the maximum integer value of the expression is 15.
theorem max_integer_value : ∀ y : ℝ, expr y ≤ 15 :=
by
  -- We state that the expression is bounded by 15. Using sorry to skip proof.
  sorry

end max_integer_value_l593_593916


namespace ratio_of_black_to_white_in_extended_pattern_l593_593088

theorem ratio_of_black_to_white_in_extended_pattern :
  ∀ (black_tiles_orig white_tiles_orig : ℕ)
  (border_pattern : ℕ → Prop),
  black_tiles_orig = 12 →
  white_tiles_orig = 23 →
  (∀ n, border_pattern n ↔ n % 2 = 0) →
  let added_tiles := 28 in
  let black_tiles_total := black_tiles_orig + (added_tiles / 2) in
  let white_tiles_total := white_tiles_orig + (added_tiles / 2) in
  (black_tiles_total : ℚ) / (white_tiles_total : ℚ) = 26 / 37 :=
begin
  intros,
  sorry, -- This is where the proof would go
end

end ratio_of_black_to_white_in_extended_pattern_l593_593088


namespace sin_315_eq_neg_sqrt_2_div_2_l593_593001

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l593_593001


namespace length_of_train_l593_593778

variable (L : ℝ) (S : ℝ)

-- Condition 1: The train crosses a 120 meters platform in 15 seconds
axiom condition1 : S = (L + 120) / 15

-- Condition 2: The train crosses a 250 meters platform in 20 seconds
axiom condition2 : S = (L + 250) / 20

-- The theorem to be proved
theorem length_of_train : L = 270 :=
by
  sorry

end length_of_train_l593_593778


namespace find_m_l593_593190

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l593_593190


namespace problem_l593_593242

-- Definitions for the given conditions
def cond1 (a : ℝ) : Prop := 100 ^ a = 4
def cond2 (b : ℝ) : Prop := 100 ^ b = 5

theorem problem (a b : ℝ) (h1 : cond1 a) (h2 : cond2 b) : 
  20 ^ ((1 - a - b) / (2 * (1 - b))) = Real.sqrt 20 := 
by 
  sorry

end problem_l593_593242


namespace no_two_consecutive_primes_in_sequence_l593_593120

theorem no_two_consecutive_primes_in_sequence {a b : ℕ} (h₁ : a > 1) (h₂ : b > 1) (h₃ : a > b) :
  ∃ (d : ℕ), x_n = λ n, (a ^ n - 1) / (b ^ n - 1) → d = 2 ∧ ¬(∀ n, Nat.Prime (x_n n) ∧ Nat.Prime (x_n (n + 1))) :=
begin
  sorry
end

end no_two_consecutive_primes_in_sequence_l593_593120


namespace prove_angle_A_l593_593962

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end prove_angle_A_l593_593962


namespace total_pet_food_weight_ounces_l593_593322

-- Define the conditions
def cat_food_bag_weight : ℕ := 3 -- each cat food bag weighs 3 pounds
def cat_food_bags : ℕ := 2 -- number of cat food bags
def dog_food_extra_weight : ℕ := 2 -- each dog food bag weighs 2 pounds more than each cat food bag
def dog_food_bags : ℕ := 2 -- number of dog food bags
def pounds_to_ounces : ℕ := 16 -- number of ounces in each pound

-- Calculate the total weight of pet food in ounces
theorem total_pet_food_weight_ounces :
  let total_cat_food_weight := cat_food_bags * cat_food_bag_weight,
      dog_food_bag_weight := cat_food_bag_weight + dog_food_extra_weight,
      total_dog_food_weight := dog_food_bags * dog_food_bag_weight,
      total_weight_pounds := total_cat_food_weight + total_dog_food_weight,
      total_weight_ounces := total_weight_pounds * pounds_to_ounces
  in total_weight_ounces = 256 :=
by 
  -- The proof is not required, so we leave it as sorry.
  sorry

end total_pet_food_weight_ounces_l593_593322


namespace num_odd_functions_l593_593089

def f1 (x : ℝ) : ℝ := x * Real.sin x
def f2 (x : ℝ) : ℝ := x * Real.cos x
def f3 (x : ℝ) : ℝ := x * |Real.cos x|
def f4 (x : ℝ) : ℝ := x * (2 : ℝ)^x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem num_odd_functions : 
  (if is_odd_function f1 then 1 else 0) +
  (if is_odd_function f2 then 1 else 0) +
  (if is_odd_function f3 then 1 else 0) +
  (if is_odd_function f4 then 1 else 0) = 2 := 
by sorry

end num_odd_functions_l593_593089


namespace product_of_digits_base8_of_12345_is_0_l593_593717

def base8_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else Nat.digits 8 n 

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_digits_base8_of_12345_is_0 :
  product_of_digits (base8_representation 12345) = 0 := 
sorry

end product_of_digits_base8_of_12345_is_0_l593_593717


namespace range_of_f_l593_593805

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 - 8 * (Real.sin x) - 3

theorem range_of_f : set.Icc (-11) 5 = set.range f := sorry

end range_of_f_l593_593805


namespace sum_possible_a1_l593_593898

-- Define the sequence and conditions
def sequence (k : ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → a (n + 1) = k * a n + 3 * k - 3

-- Given values for the sequence at specific indices
def given_values (a : ℕ → ℝ) : Prop :=
a 2 = -678 ∧ a 3 = -78 ∧ a 4 = -3.22 ∧ a 5 = 222 ∧ a 5 = 2222

-- Condition for k
def constant_k (k : ℝ) : Prop :=
k ≠ 0 ∧ k ≠ 1

-- Prove that the sum of all possible values of a₁ is 6023/3
theorem sum_possible_a1 (k : ℝ) (a : ℕ → ℝ) :
    sequence k a →
    given_values a →
    constant_k k →
    (∃ s1 s2 s3 : ℝ, (s1 = -3 ∧ s2 = -34/3 ∧ s3 = 2022) ∧ s1 + s2 + s3 = 6023 / 3) :=
by
  intro seq gv cons
  -- Insert steps to conclude the theorem
  sorry

end sum_possible_a1_l593_593898


namespace hyperbola_tangent_circle_asymptote_l593_593250

theorem hyperbola_tangent_circle_asymptote (m : ℝ) (hm : m > 0) :
  (∀ x y : ℝ, y^2 - (x^2 / m^2) = 1 → 
   ∃ (circle_center : ℝ × ℝ) (circle_radius : ℝ), 
   circle_center = (0, 2) ∧ circle_radius = 1 ∧ 
   ((∃ k : ℝ, k = m ∨ k = -m) → 
    ((k * 0 - 2 + 0) / real.sqrt ((k)^2 + (-1)^2) = 1))) →
  m = real.sqrt 3 / 3 :=
by sorry

end hyperbola_tangent_circle_asymptote_l593_593250


namespace solution_satisfies_inequality_l593_593110

noncomputable def largest_integer_pair_satisfying_inequality : ℕ × ℕ :=
  ⟨0, 0⟩

theorem solution_satisfies_inequality :
  ∀ x, (0 ≤ x ∧ x ≤ Real.pi / 2) →
        (Real.sin x ^ largest_integer_pair_satisfying_inequality.fst *
        Real.cos x ^ largest_integer_pair_satisfying_inequality.snd ≥
        (1 / 2) ^ ((largest_integer_pair_satisfying_inequality.fst +
        largest_integer_pair_satisfying_inequality.snd) / 2)) :=
by
  intros x hx
  have h : largest_integer_pair_satisfying_inequality = (0, 0) := rfl
  rw h
  simp
  sorry

end solution_satisfies_inequality_l593_593110


namespace find_a_from_binomial_expansion_l593_593855

theorem find_a_from_binomial_expansion (a : ℝ) (h_pos : a > 0)
  (h_coeff : ∀ x : ℝ, 135 = (Finset.range 7).sum (λ r, if (3 - r = -1) then (Nat.choose 6 r) * (-1) ^ r * a ^ (6 - r) else 0)) : a = 3 :=
by
  sorry

end find_a_from_binomial_expansion_l593_593855


namespace repeating_decimal_eq_l593_593502

-- Defining the repeating decimal as a hypothesis
def repeating_decimal : ℚ := 0.7 + 3/10^2 * (1/(1 - 1/10))
-- We will prove this later by simplifying the fraction
def expected_fraction : ℚ := 11/15

theorem repeating_decimal_eq : repeating_decimal = expected_fraction := 
by
  sorry

end repeating_decimal_eq_l593_593502


namespace cube_root_expression_l593_593118

theorem cube_root_expression (n : ℕ) : 
  Real.cbrt (∑ k in Finset.range (n + 1), 8 * (k : ℝ)^3 / ∑ k in Finset.range (n + 1), 27 * (k : ℝ)^3) = 2 / 3 := 
by
  sorry

end cube_root_expression_l593_593118


namespace jerusha_and_lottie_earnings_l593_593611

theorem jerusha_and_lottie_earnings :
  let J := 68
  let L := J / 4
  J + L = 85 := 
by
  sorry

end jerusha_and_lottie_earnings_l593_593611


namespace range_of_4x_2y_l593_593927

theorem range_of_4x_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 1) :
  2 ≤ 4 * x + 2 * y ∧ 4 * x + 2 * y ≤ 10 := 
sorry

end range_of_4x_2y_l593_593927


namespace cost_of_50_roses_l593_593786

-- Definitions for the problem's conditions
def cost_of_bouquet (n : ℕ) : ℝ :=
  if n ≤ 30 then
    (30 / 15) * n
  else
    (30 / 15) * 30 + (30 / 15 / 2) * (n - 30)

-- The theorem we need to prove
theorem cost_of_50_roses :
  cost_of_bouquet 50 = 80 :=
by
  sorry

end cost_of_50_roses_l593_593786


namespace fraction_simplify_l593_593815

theorem fraction_simplify:
  (1/5 + 1/7) / (3/8 - 1/9) = 864 / 665 :=
by
  sorry

end fraction_simplify_l593_593815


namespace solution_inequality_l593_593622

variable (p p' q q' : ℝ)
variable (hp : p ≠ 0) (hp' : p' ≠ 0)

theorem solution_inequality 
  (h1 : (∃ x, p * x + q' = 0)) 
  (h2 : (∃ x, p' * x + q = 0)) : 
  (∃ x₁, p * x₁ + q' = 0) → (∃ x₂, p' * x₂ + q = 0) → 
  (∃ x₁ x₂, x₁ > x₂ ↔ (q' / p < q / p')) := 
sorry

end solution_inequality_l593_593622


namespace regular_polygon_sides_l593_593770

theorem regular_polygon_sides (n : ℕ) (hn : RegularPolygon.angle n = 150) : n = 12 :=
by
  sorry

end regular_polygon_sides_l593_593770


namespace find_m_l593_593169

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l593_593169


namespace cost_of_trip_per_student_l593_593633

def raised_fund : ℕ := 50
def contribution_per_student : ℕ := 5
def num_students : ℕ := 20
def remaining_fund : ℕ := 10

theorem cost_of_trip_per_student :
  ((raised_fund - remaining_fund) / num_students) = 2 := by
  sorry

end cost_of_trip_per_student_l593_593633


namespace range_of_a_l593_593850

def f (x : ℝ) (a : ℝ) := x^3 - a*x - 1

def p (a : ℝ) : Prop := ∀ x ∈ Icc (-1:ℝ) (1:ℝ), deriv (f x a) ≤ 0

def q (a : ℝ) : Prop := a^2 - 4 ≥ 0

def p_true (a : ℝ) : Prop := a ≥ 3

def q_true (a : ℝ) : Prop := a ≤ -2 ∨ a ≥ 2

def either_p_or_q (a : ℝ) : Prop := p_true a ∧ ¬q_true a ∨ ¬p_true a ∧ q_true a

theorem range_of_a : ∀ a : ℝ, either_p_or_q a ↔ (a ≤ -2 ∨ (2 ≤ a ∧ a < 3)) :=
by
  sorry

end range_of_a_l593_593850


namespace find_satisfying_pairs_l593_593480

theorem find_satisfying_pairs (n p : ℕ) (prime_p : Nat.Prime p) :
  n ≤ 2 * p ∧ (p - 1)^n + 1 ≡ 0 [MOD n^2] →
  (n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
by sorry

end find_satisfying_pairs_l593_593480


namespace team_with_most_points_drew_at_least_one_l593_593586

theorem team_with_most_points_drew_at_least_one (n : ℕ)
  (match_points : ℕ → ℕ → ℕ)
  (points_for_win points_for_draw points_for_loss : ℕ)
  (least_points highest_points : ℕ) :
  n = 15 →
  points_for_win = 3 →
  points_for_draw = 2 →
  points_for_loss = 1 →
  least_points = 21 →
  highest_points = 35 →
  (∀ i j : ℕ, i ≠ j → match_points i j + match_points j i = 4) →
  (∀ i : ℕ, 21 ≤ match_points i 1 + match_points i 2 + ... + match_points i n ≤ 35) →
  ∃ i : ℕ, match_points i _ = points_for_draw :=
by sorry

end team_with_most_points_drew_at_least_one_l593_593586


namespace avg_first_30_multiples_29_l593_593715

theorem avg_first_30_multiples_29 :
  let n := 30
  let a1 := 29
  let aN := 29 * n
  let sum := (n / 2) * (a1 + aN)
  let average := sum / n
  average = 449.5 :=
by
  let n := 30
  let a1 := 29
  let aN := 29 * n
  let sum := (n / 2) * (a1 + aN)
  let average := sum / n
  show average = 449.5
  sorry

end avg_first_30_multiples_29_l593_593715


namespace avg_remaining_students_l593_593406

theorem avg_remaining_students (N : ℕ) (hN : N > 0) :
  let total_avg := 80 * N
  let num_95 := 0.10 * N
  let num_90 := 0.20 * N
  let total_95 := num_95 * 95
  let total_90 := num_90 * 90
  let num_remaining := N - num_95 - num_90
  let total_remaining := total_avg - total_95 - total_90
  let avg_remaining := total_remaining / num_remaining
  avg_remaining = 75 :=
by
  sorry

end avg_remaining_students_l593_593406


namespace length_of_FP_l593_593659

open Real

noncomputable def square_side_length := 5
noncomputable def area_of_square := square_side_length^2
noncomputable def area_each_part := area_of_square / 4

theorem length_of_FP (side_length : ℝ) (area_part : ℝ) : 
  side_length = 5 → area_part = 6.25 → 
  let FP := sqrt ((side_length^2 / 2) + (2.5 ^ 2)) 
  in FP = sqrt 31.25 := by
  intros
  sorry

end length_of_FP_l593_593659


namespace trigonometric_identity_1_trigonometric_identity_2_l593_593134

open Real

theorem trigonometric_identity_1 (α : ℝ) (h1 : α ∈ Ioc (π / 2) π) (h2 : sin α = 4 / 5) :
  cos (α - π / 4) = sqrt 2 / 10 :=
sorry

theorem trigonometric_identity_2 (α : ℝ) (h1 : α ∈ Ioc (π / 2) π) (h2 : sin α = 4 / 5) :
  sin (α / 2) ^ 2 + (sin (4 * α) * cos (2 * α)) / (1 + cos (4 * α)) = - 4 / 25 :=
sorry

end trigonometric_identity_1_trigonometric_identity_2_l593_593134


namespace number_of_girls_l593_593568

def total_students (T : ℕ) :=
  0.40 * T = 300

def girls_at_school (T : ℕ) :=
  0.60 * T = 450

theorem number_of_girls (T : ℕ) (h : total_students T) : girls_at_school T :=
  sorry

end number_of_girls_l593_593568


namespace find_a_plus_b_l593_593694

theorem find_a_plus_b (a b : ℝ) (h_sum : 2 * a = -6) (h_prod : a^2 - b = 1) : a + b = 5 :=
by {
  -- Proof would go here; we assume the theorem holds true.
  sorry
}

end find_a_plus_b_l593_593694


namespace total_distance_is_105_km_l593_593370

-- Define the boat's speed in still water
def boat_speed_still_water : ℝ := 50

-- Define the current speeds for each hour
def current_speed_first_hour : ℝ := 10
def current_speed_second_hour : ℝ := 20
def current_speed_third_hour : ℝ := 15

-- Calculate the effective speeds for each hour
def effective_speed_first_hour := boat_speed_still_water - current_speed_first_hour
def effective_speed_second_hour := boat_speed_still_water - current_speed_second_hour
def effective_speed_third_hour := boat_speed_still_water - current_speed_third_hour

-- Calculate the distance traveled in each hour
def distance_first_hour := effective_speed_first_hour * 1
def distance_second_hour := effective_speed_second_hour * 1
def distance_third_hour := effective_speed_third_hour * 1

-- Define the total distance
def total_distance_traveled := distance_first_hour + distance_second_hour + distance_third_hour

-- Prove that the total distance traveled is 105 km
theorem total_distance_is_105_km : total_distance_traveled = 105 := by
  sorry

end total_distance_is_105_km_l593_593370


namespace integral_bound_l593_593977

noncomputable def f (x : ℝ) : ℝ := (1 - x^2)^(3/2)

def M : ℝ := 3 / 2

theorem integral_bound : ∫ x in -1..(1 : ℝ), f x ≤ M := 
by sorry

end integral_bound_l593_593977


namespace sin_315_eq_neg_sqrt2_over_2_l593_593041

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593041


namespace length_of_train_is_179_96_l593_593780

-- Define the given conditions as constants
constant speed_kmph : ℝ := 70   -- Speed of the train in kmph
constant time_sec : ℝ := 20     -- Time taken to cross the platform in seconds
constant platform_length_m : ℝ := 208.92  -- Length of the platform in meters

-- Convert speed from kmph to m/s
def speed_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

-- Calculate the total distance covered by the train
def total_distance (speed_mps : ℝ) (time_sec : ℝ) : ℝ := speed_mps * time_sec

-- Calculate the length of the train
def train_length (total_distance : ℝ) (platform_length_m : ℝ) : ℝ := total_distance - platform_length_m

-- Main theorem to prove
theorem length_of_train_is_179_96 :
  train_length (total_distance (speed_mps speed_kmph) time_sec) platform_length_m = 179.96 :=
by
  sorry

end length_of_train_is_179_96_l593_593780


namespace range_of_x_l593_593521

theorem range_of_x (x : ℝ) : -2 * x + 3 ≤ 6 → x ≥ -3 / 2 :=
sorry

end range_of_x_l593_593521


namespace cubic_polynomial_solution_l593_593508

-- Define p(x)
def p (x : ℝ) : ℝ := - (5 / 6) * x ^ 3 + 5 * x ^ 2 - (85 / 6) * x - 5

-- Proposition that p(x) meets the specified conditions
theorem cubic_polynomial_solution :
  p 1 = -10 ∧ p 2 = -20 ∧ p 3 = -30 ∧ p 5 = -70 :=
by
  split
  · -- Prove p(1) = -10
    sorry
  split
  · -- Prove p(2) = -20
    sorry
  split
  · -- Prove p(3) = -30
    sorry
  · -- Prove p(5) = -70
    sorry

end cubic_polynomial_solution_l593_593508


namespace comparison_l593_593130

def a : ℝ := (1/2) * Real.cos (6 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * Real.pi / 180)
def b : ℝ := 2 * Real.sin (13 * Real.pi / 180) * Real.cos (13 * Real.pi / 180)
def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem comparison : a < c ∧ c < b :=
by
  sorry

end comparison_l593_593130


namespace find_m_l593_593184

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l593_593184


namespace sin_315_eq_neg_sqrt2_div_2_l593_593070

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593070


namespace coeff_x3_expansion_l593_593481

theorem coeff_x3_expansion : 
  let expansion := (1/x - x^2)^6
  (coeff_of_x3 expansion) = -20 :=
by
  sorry

end coeff_x3_expansion_l593_593481


namespace surface_area_of_modified_structure_l593_593085

-- Define the given conditions
def initial_cube_side_length : ℕ := 12
def smaller_cube_side_length : ℕ := 2
def smaller_cubes_count : ℕ := 72
def face_center_cubes_count : ℕ := 6

-- Define the calculation of the surface area
def single_smaller_cube_surface_area : ℕ := 6 * (smaller_cube_side_length ^ 2)
def added_surface_from_removed_center_cube : ℕ := 4 * (smaller_cube_side_length ^ 2)
def modified_smaller_cube_surface_area : ℕ := single_smaller_cube_surface_area + added_surface_from_removed_center_cube
def unaffected_smaller_cubes : ℕ := smaller_cubes_count - face_center_cubes_count

-- Define the given surface area according to the problem
def correct_surface_area : ℕ := 1824

-- The equivalent proof problem statement
theorem surface_area_of_modified_structure : 
    66 * single_smaller_cube_surface_area + 6 * modified_smaller_cube_surface_area = correct_surface_area := 
by
    -- placeholders for the actual proof
    sorry

end surface_area_of_modified_structure_l593_593085


namespace perfect_squares_l593_593680

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593680


namespace max_value_of_f_l593_593202

noncomputable def f (x a b : ℝ) : ℝ := (a - 1) * (sqrt (x - 3)) + (b - 1) * (sqrt (4 - x))

theorem max_value_of_f (a b : ℝ) (h₁ : 1 + 2 = a) (h₂ : 1 * 2 = b) : 
  ∃ x ∈ set.Icc (3:ℝ) (4:ℝ), f x a b = sqrt 5 :=
by
  have ha : a = 3 := by rwa [←h₁]
  have hb : b = 2 := by rwa [←h₂]
  use 19 / 5
  split
  sorry
  rw [f, ha, hb]
  sorry

end max_value_of_f_l593_593202


namespace part1_part2_part3_l593_593661

-- Definition of the given expression
def expr (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 2) - (5 * x^2 + 3 * x)

-- Condition 1: Given final result 2x^2 - 4x + 2
def target_expr1 (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 2

-- Condition 2: Given values for a and b by Student B
def student_b_expr (x : ℝ) : ℝ := (5 * x^2 - 3 * x + 2) - (5 * x^2 + 3 * x)

-- Condition 3: Result independent of x
def target_expr3 : ℝ := 2

-- Prove conditions and answers
theorem part1 (a b : ℝ) : (∀ x : ℝ, expr a b x = target_expr1 x) → a = 7 ∧ b = -1 :=
sorry

theorem part2 : (∀ x : ℝ, student_b_expr x = -6 * x + 2) :=
sorry

theorem part3 (a b : ℝ) : (∀ x : ℝ, expr a b x = 2) → a = 5 ∧ b = 3 :=
sorry

end part1_part2_part3_l593_593661


namespace find_m_l593_593154

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l593_593154


namespace landscape_breadth_l593_593440

theorem landscape_breadth (L B : ℕ)
  (h1 : B = 6 * L)
  (h2 : 4200 = (1 / 7 : ℚ) * 6 * L^2) :
  B = 420 := 
  sorry

end landscape_breadth_l593_593440


namespace XY_squared_is_112_l593_593596

noncomputable def compute_XY_squared (AB BC CD DA : ℝ) (angleC : ℝ) (X Y : ℝ × ℝ) : ℝ :=
  if h : AB = 15 ∧ BC = 15 ∧ CD = 20 ∧ DA = 20 ∧ angleC = 90 ∧ X = (7.5, 0) ∧ Y = (0, 10) then
    let AC := 15 * Math.sqrt 2 in
    (AC / 2) * (AC / 2)
  else
    0

theorem XY_squared_is_112.5 : 
  compute_XY_squared 15 15 20 20 90 (7.5, 0) (0, 10) = 112.5 := 
by 
  -- This is the proof placeholder
  sorry

end XY_squared_is_112_l593_593596


namespace polyhedron_no_inscribe_sphere_l593_593658

theorem polyhedron_no_inscribe_sphere (F : ℕ) (n m : ℕ) (polyhedron : Type) (h_total_faces : F = n + m)
  (h_no_common_edge_black : ∀ (f₁ f₂ : polyhedron), (f₁ ∈ black_faces) ∧ (f₂ ∈ black_faces) → ¬(share_edge f₁ f₂))
  (h_more_than_half_black : n > F / 2) :
  ¬inscribable_in_sphere polyhedron :=
sorry

end polyhedron_no_inscribe_sphere_l593_593658


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593019

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593019


namespace most_followers_after_three_weeks_l593_593660

def initial_followers_susy := 100
def initial_followers_sarah := 50
def first_week_gain_susy := 40
def second_week_gain_susy := first_week_gain_susy / 2
def third_week_gain_susy := second_week_gain_susy / 2
def first_week_gain_sarah := 90
def second_week_gain_sarah := first_week_gain_sarah / 3
def third_week_gain_sarah := second_week_gain_sarah / 3

def total_followers_susy := initial_followers_susy + first_week_gain_susy + second_week_gain_susy + third_week_gain_susy
def total_followers_sarah := initial_followers_sarah + first_week_gain_sarah + second_week_gain_sarah + third_week_gain_sarah

theorem most_followers_after_three_weeks : max total_followers_susy total_followers_sarah = 180 :=
by
  sorry

end most_followers_after_three_weeks_l593_593660


namespace marcy_total_time_l593_593998

theorem marcy_total_time 
    (petting_time : ℝ)
    (fraction_combing : ℝ)
    (H1 : petting_time = 12)
    (H2 : fraction_combing = 1/3) :
    (petting_time + (fraction_combing * petting_time) = 16) :=
  sorry

end marcy_total_time_l593_593998


namespace remainder_of_product_mod_17_l593_593392

theorem remainder_of_product_mod_17 :
  (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 :=
sorry

end remainder_of_product_mod_17_l593_593392


namespace banana_pie_angle_l593_593935

theorem banana_pie_angle
  (total_students : ℕ := 48)
  (chocolate_students : ℕ := 15)
  (apple_students : ℕ := 10)
  (blueberry_students : ℕ := 9)
  (remaining_students := total_students - (chocolate_students + apple_students + blueberry_students))
  (banana_students := remaining_students / 2) :
  (banana_students : ℝ) / total_students * 360 = 52.5 :=
by
  sorry

end banana_pie_angle_l593_593935


namespace flour_needed_l593_593632

-- Define the given conditions
def F_total : ℕ := 9
def F_added : ℕ := 3

-- State the main theorem to be proven
theorem flour_needed : (F_total - F_added) = 6 := by
  sorry -- Placeholder for the proof

end flour_needed_l593_593632


namespace number_of_girls_l593_593570

theorem number_of_girls (total_students boys girls : ℕ)
  (h1 : boys = 300)
  (h2 : (girls : ℝ) = 0.6 * total_students)
  (h3 : (boys : ℝ) = 0.4 * total_students) : 
  girls = 450 := by
  sorry

end number_of_girls_l593_593570


namespace rise_ratio_l593_593387

-- Definitions for given conditions in the problem
def r_n : ℝ := 4
def r_w : ℝ := 8
def k : ℝ := 1 -- assuming k is any positive real number, since it will cancel out
def h_n : ℝ := 2 * k
def h_w : ℝ := k

def V_marble : ℝ := 2 * (4 / 3) * π * (1)^3

-- Volumes of liquid initially in the cones
def V_n : ℝ := (1 / 3) * π * r_n^2 * h_n
def V_w : ℝ := (1 / 3) * π * r_w^2 * h_w

-- New volumes in each cone after adding the marbles
def V_n' : ℝ := V_n + V_marble
def V_w' : ℝ := V_w + V_marble

-- New heights after adding the marbles
def h_n' : ℝ := V_n' / ((1 / 3) * π * r_n^2)
def h_w' : ℝ := V_w' / ((1 / 3) * π * r_w^2)

-- Rise in the liquid levels
def Δh_n : ℝ := h_n' - h_n
def Δh_w : ℝ := h_w' - h_w

-- Ratio of the rise of the liquid level in the narrow cone to the wide cone
def ratio_rise : ℝ := Δh_n / Δh_w

theorem rise_ratio : ratio_rise = 2 :=
by
  rw [ratio_rise, Δh_n, Δh_w, h_n', h_w', V_n', V_w', V_n, V_w, V_marble, h_n, h_w, r_n, r_w, k]
  -- Provide the final mathematical manipulation and simplification here, skipping with 'sorry'
  sorry

end rise_ratio_l593_593387


namespace same_units_digit_pages_count_l593_593747

theorem same_units_digit_pages_count : 
  (∃ (p : Finset ℕ), p = (Finset.filter 
      (λ x, ((x % 10) = ((64 - x) % 10))) 
      (Finset.range 64)) ∧ p.card = 13 :=
by
  sorry

end same_units_digit_pages_count_l593_593747


namespace find_k_binom_l593_593505

/-- Define binomial coefficient -/
def binom : ℕ → ℕ → ℕ
| n, 0     := 1
| 0, k     := 0
| n+1, k+1 := binom n k + binom n (k+1)

/-- Main theorem statement -/
theorem find_k_binom (k : ℤ) : 
  (k = 1 → ∀ n : ℕ, n > 0 → (n + k : ℕ) ∣ binom (2 * n) n) ∧ 
  (k ≠ 1 → ∃ᶠ n in filter.at_top, ¬ (n + k : ℕ) ∣ binom (2 * n) n) :=
sorry

end find_k_binom_l593_593505


namespace maximize_l_l593_593315

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 8 * x + 3

theorem maximize_l (a : ℝ) (h : a < 0) : 
  ∃ l: ℝ, l = (√5 + 1) / 2 ∧ (∀ x ∈ Icc (0 : ℝ) l, abs (f a x) ≤ 5) ↔ a = -8 :=
sorry

end maximize_l_l593_593315


namespace additional_people_needed_to_mow_lawn_l593_593835

theorem additional_people_needed_to_mow_lawn :
  (∀ (k : ℕ), (∀ (n t : ℕ), n * t = k) → (4 * 6 = k) → (∃ (n : ℕ), n * 3 = k) → (8 - 4 = 4)) :=
by sorry

end additional_people_needed_to_mow_lawn_l593_593835


namespace find_m_l593_593173

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l593_593173


namespace sin_315_eq_neg_sqrt2_div_2_l593_593003

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593003


namespace missing_angle_in_convex_polygon_l593_593139

theorem missing_angle_in_convex_polygon (sum_other_angles : ℕ) (h1 : sum_other_angles = 2570) :
  ∃ (missing_angle : ℕ), missing_angle = 130 ∧ (sum_other_angles + missing_angle) % 180 = 0 :=
by
  use 130
  split
  case left =>
    sorry
  case right =>
    sorry

end missing_angle_in_convex_polygon_l593_593139


namespace cos_alpha_value_l593_593128

-- Define our conditions
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -5 / 13
axiom tan_alpha_pos : Real.tan α > 0

-- State our goal
theorem cos_alpha_value : Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_value_l593_593128


namespace extra_postage_count_is_2_l593_593790

structure Envelope where
  length : ℝ
  height : ℝ

def requiresExtraPostage (e : Envelope) : Prop :=
  let ratio := e.length / e.height
  ratio < 1.2 ∨ ratio > 2.8

def envelopeE := Envelope.mk 5 3
def envelopeF := Envelope.mk 10 2
def envelopeG := Envelope.mk 5 5
def envelopeH := Envelope.mk 12 5

def envelopes := [envelopeE, envelopeF, envelopeG, envelopeH]

def countEnvelopesRequiringExtraPostage (l : List Envelope) : ℕ :=
  l.count requiresExtraPostage

theorem extra_postage_count_is_2 :
  countEnvelopesRequiringExtraPostage envelopes = 2 :=
by
  sorry

end extra_postage_count_is_2_l593_593790


namespace minimum_transfers_required_l593_593376

def initial_quantities : List ℕ := [2, 12, 12, 12, 12]
def target_quantity := 10
def min_transfers := 4

theorem minimum_transfers_required :
  ∃ transfers : ℕ, transfers = min_transfers ∧
  ∀ quantities : List ℕ, List.sum initial_quantities = List.sum quantities →
  (∀ q ∈ quantities, q = target_quantity) :=
by
  sorry

end minimum_transfers_required_l593_593376


namespace maximize_total_profit_l593_593755

-- Define the profit functions
def P (m : ℝ) := (1 / 2) * m + 60
def Q (m : ℝ) := 70 + 6 * Real.sqrt m

-- Define the total capital invested
def total_capital : ℝ := 200

-- Define the total profit function and its domain
def profit (x : ℝ) := -(1 / 2) * x + 6 * Real.sqrt x + 230

-- Conditions for the invested capital
def domain := Set.Icc 25 175

theorem maximize_total_profit : 
  ∃ x ∈ domain, profit x = 248 :=
by
  sorry

end maximize_total_profit_l593_593755


namespace geometric_sequence_sum_point_on_line_l593_593698

theorem geometric_sequence_sum_point_on_line
  (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (r : ℝ)
  (h1 : a 1 = t)
  (h2 : ∀ n : ℕ, a (n + 1) = t * r ^ n)
  (h3 : ∀ n : ℕ, S n = t * (1 - r ^ n) / (1 - r))
  (h4 : ∀ n : ℕ, (S n, a (n + 1)) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 1})
  : t = 1 :=
by
  sorry

end geometric_sequence_sum_point_on_line_l593_593698


namespace sum_other_endpoint_coordinates_l593_593639

def one_endpoint := (6, 4)
def midpoint := (3, 10)

-- Define the coordinate of the other endpoint as (x, y)
def other_endpoint (x y : ℤ) : Prop :=
  (6 + x) / 2 = 3 ∧ (4 + y) / 2 = 10

-- Prove that the sum of the coordinates of the other endpoint is 16
theorem sum_other_endpoint_coordinates : ∃ (x y : ℤ), other_endpoint x y ∧ (x + y) = 16 :=
 by {
  -- Placeholder for the actual proof
  sorry
 }

end sum_other_endpoint_coordinates_l593_593639


namespace centers_of_upper_balls_collinear_l593_593104

-- Define the conditions for identical balls in a tightly packed arrangement
variable {r : ℝ} (A B C D F G : Point)

-- Assume the distance between the centers of adjacent balls is 2r
axiom dist_DA : dist D A = 2 * r
axiom dist_DF : dist D F = 2 * r
axiom dist_DG : dist D G = 2 * r

-- Prove that the centers A, B, and C lie on the same line
theorem centers_of_upper_balls_collinear : 
    collinear ℝ ({A, B, C} : set Point) := 
sorry

end centers_of_upper_balls_collinear_l593_593104


namespace find_m_l593_593186

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l593_593186


namespace percentage_decrease_y_correct_l593_593345

noncomputable def percentage_decrease_y (x y z k p : ℝ) (hx : z = x^2) (hxy : x^2 * y = k) : ℝ :=
(1 - 1 / (1 + p / 100)^2) * 100

theorem percentage_decrease_y_correct (x y k p : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (hxy : x^2 * y = k) :
  percentage_decrease_y x y (x^2) k p = (1 - 1 / (1 + 0.01 * p)^2) * 100 :=
by
  sorry

end percentage_decrease_y_correct_l593_593345


namespace probability_at_least_one_girl_l593_593836

theorem probability_at_least_one_girl (total_students boys girls k : ℕ) (h_total: total_students = 5) (h_boys: boys = 3) (h_girls: girls = 2) (h_k: k = 3) : 
  (1 - ((Nat.choose boys k) / (Nat.choose total_students k))) = 9 / 10 :=
by
  sorry

end probability_at_least_one_girl_l593_593836


namespace frustum_volume_l593_593697

theorem frustum_volume (r1 r2 l h : ℝ) (V : ℝ) 
  (hr1 : r1 = 1) 
  (hr2 : r2 = 4) 
  (hl : l = 3 * Real.sqrt 2) 
  (hh : h = 3) 
  (hV : V = (1 / 3) * Real.pi * (r1^2 + r2^2 + (r1 * r2)) * h) : 
  V = 21 * Real.pi :=
by
  rw [hr1, hr2, hl, hh] at hV
  exact hV

end frustum_volume_l593_593697


namespace volume_of_PQRS_l593_593503

noncomputable def volume_of_tetrahedron (P Q R S : ℝ^3) : ℝ :=
  let angle_PQR_QRS := 45
  let area_PQR := 150
  let area_QRS := 90
  let QR_length := 15
  have h_area_QRS : 1/2 * QR_length * h = area_QRS, from sorry,
  let h := 12
  have h_height_from_S := h * (Real.sin (45 * Real.pi / 180)), from sorry,
  let height_from_S := 6 * Real.sqrt 2
  have h_area_base := area_PQR, from sorry
  let area_base := 150
  have volume := 1/3 * area_base * height_from_S, from sorry
  volume

theorem volume_of_PQRS (P Q R S : ℝ^3) :
  volume_of_tetrahedron P Q R S = 300 * Real.sqrt 2 :=
sorry

end volume_of_PQRS_l593_593503


namespace sin_315_eq_neg_sqrt2_over_2_l593_593038

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593038


namespace minimum_value_z_achieve_minimum_value_z_l593_593885

def hyperbola (t : ℝ) (h : t ≠ 0) : ℝ → ℝ → Prop :=
  λ x y, 4 * x^2 - y^2 = t

def asymptote1 : ℝ → ℝ → Prop := 
  λ x y, y = 2 * x

def asymptote2 : ℝ → ℝ → Prop := 
  λ x y, y = -2 * x

def vertical_line : ℝ → ℝ → Prop := 
  λ x y, x = real.sqrt 2

def region_D (t : ℝ) (h : t ≠ 0) : ℝ → ℝ → Prop :=
  λ x y, (hyperbola t h x y ∨ asymptote1 x y ∨ asymptote2 x y ∨ vertical_line x y) ∧ 
          (0 ≤ x ∧ x ≤ real.sqrt 2) ∧
          (y = 2 * x ∨ y = -2 * x ∨ y = 0)

def z : ℝ → ℝ → ℝ := 
  λ x y, (1/2) * x - y

theorem minimum_value_z (t : ℝ) (h : t ≠ 0) : 
  ∀ x y, region_D t h x y → z x y ≥ -3 * real.sqrt 2 / 2 :=
begin
  sorry
end

theorem achieve_minimum_value_z (t : ℝ) (h : t ≠ 0) : 
  z (real.sqrt 2) (2 * real.sqrt 2) = -3 * real.sqrt 2 / 2 ∧
  z (real.sqrt 2) (-2 * real.sqrt 2) = -3 * real.sqrt 2 / 2 :=
begin
  sorry
end

end minimum_value_z_achieve_minimum_value_z_l593_593885


namespace find_slope_of_line_through_focus_l593_593895

-- Define the conditions
def is_point (x y : ℝ) : Prop := True

def is_parabola (x y : ℝ) : Prop := y^2 = 4 * x

def is_slope_of_line_through_focus_intersecting_parabola
  (k x1 y1 x2 y2 : ℝ) : Prop :=
  let F := (1,0) in -- Focus of the parabola
  y1 = k * (x1 - 1) ∧
  y2 = k * (x2 - 1) ∧
  y1^2 = 4 * x1 ∧
  y2^2 = 4 * x2

def angle_is_90_degrees (M A B : ℝ × ℝ) : Prop :=
  let MA := (A.1 + 1, A.2 - 1) in
  let MB := (B.1 + 1, B.2 - 1) in
  MA.1 * MB.1 + MA.2 * MB.2 = 0

-- Define the point M, parabola C, and conditions
noncomputable def M := (-1, 1)
noncomputable def parabola_C (x y : ℝ) : Prop := y^2 = 4 * x

-- Main statement to prove slope k = 2
theorem find_slope_of_line_through_focus 
  (k x1 y1 x2 y2 : ℝ)
  (H1 : is_point M.1 M.2)
  (H2 : is_parabola x1 y1)
  (H3 : is_parabola x2 y2)
  (H4 : angle_is_90_degrees M (x1, y1) (x2, y2))
  (H5 : is_slope_of_line_through_focus_intersecting_parabola k x1 y1 x2 y2) :
  k = 2 :=
sorry

end find_slope_of_line_through_focus_l593_593895


namespace sufficient_but_not_necessary_l593_593623

variable (x y : ℝ)

theorem sufficient_but_not_necessary (x_gt_y_gt_zero : x > y ∧ y > 0) : (x / y > 1) :=
by
  sorry

end sufficient_but_not_necessary_l593_593623


namespace max_regions_l593_593905

-- Define the conditions and parameters of the problem
variables (m n : ℕ)

-- Define the main theorem stating the maximum number of regions
theorem max_regions (m n : ℕ) : 
  (∃ segments : list (ℕ × ℕ), 
    (∀ (s1 s2 : ℕ × ℕ), s1 ∈ segments → s2 ∈ segments → s1 ≠ s2 → ¬ (segments_intersect s1 s2)) ∧ 
    (∀ s ∈ segments, consecutive_rows s) ∧
    (∀ s1 s2 : ℕ × ℕ, s1 ∈ segments → s2 ∈ segments → s1 ≠ s2 → one_segment_between_points s1 s2)) → 
  2 * m * n = maximum number of regions :=
sorry

-- Helper predicates (stubs):
-- Define what it means for two segments to intersect
def segments_intersect  : (ℕ × ℕ) → (ℕ × ℕ) → Prop := sorry
-- Define what it means for a segment to be drawn between consecutive rows
def consecutive_rows  : (ℕ × ℕ) → Prop := sorry
-- Define what it means for there to be at most one segment between any two points
def one_segment_between_points  : (ℕ × ℕ) → (ℕ × ℕ) → Prop := sorry

end max_regions_l593_593905


namespace perpendicular_MN_AB_l593_593352

variables (A B C E H_3 M N : Type) [PlaneGeometry A B C E]

-- Conditions
axiom angle_diff_A_B : ∠ B - ∠ A = 90°
axiom altitude_CH_3 : Altitude C H_3 A E
axiom perpendicular_H_3M : Perpendicular H_3 M (Segment A C)
axiom perpendicular_H_3N : Perpendicular H_3 N (Segment E C)
axiom base_H_3 : H_3 ∈ Segment (A, E)

-- Main proof statement
theorem perpendicular_MN_AB : Perpendicular (Line M N) (Line A B) :=
sorry

end perpendicular_MN_AB_l593_593352


namespace count_perfect_squares_between_50_and_250_l593_593231

theorem count_perfect_squares_between_50_and_250:
  ∃ (count : ℕ), count = 8 := by
  -- Define the set of perfect squares between 50 and 250
  let squares := { n | ∃ k : ℕ, n = k ^ 2 ∧ 50 < n ∧ n < 250 }
  -- Compute the number of elements in this set
  let count := (squares.filter λ n => 50 < n ∧ n < 250).card
  -- Assert that this number is 8
  have : count = 8 := sorry
  exact ⟨count, this⟩

end count_perfect_squares_between_50_and_250_l593_593231


namespace calc_expression_solve_equation_l593_593740

section calculation

-- Problem (1)
theorem calc_expression : 
  abs (real.sqrt 3 - 1) - 2 * real.cos (real.pi / 3) + (real.sqrt 3 - 2)^2 + real.sqrt 12 = 5 - real.sqrt 3 := 
sorry

-- Problem (2)
theorem solve_equation (x : ℝ) : 
  (2 * (x - 3)^2 = x^2 - 9) ↔ (x = 3 ∨ x = 9) := 
sorry

end calculation

end calc_expression_solve_equation_l593_593740


namespace max_different_dwarfs_l593_593268

theorem max_different_dwarfs 
  (x y z : ℕ) 
  (h1 : 1 ≤ x ∧ x ≤ 9) 
  (h2 : 1 ≤ z ∧ z ≤ 9) 
  (h3 : 0 ≤ y ∧ y ≤ 9) 
  (h4 : 100 * x + 10 * y + z + 198 = 100 * z + 10 * y + x) :
  x = z - 2 ∧ (3 ≤ z ∧ z ≤ 9) →
  ∃ (dwarfs : Finch (Fin 70)), True :=
begin
  -- Problem setup ensures all conditions are met and
  -- that there exist 70 unique (x, y, z) triples for dwarfs.
  sorry
end

end max_different_dwarfs_l593_593268


namespace probability_of_sum_six_l593_593708

noncomputable def probability_sum_six (A B : Finset ℕ) (hA : A = {2, 3, 4}) (hB : B = {2, 3, 4}) : ℚ :=
  let outcomes := { (a, b) | a ∈ A ∧ b ∈ B }
  let favorable := { (a, b) | a + b = 6 ∧ a ∈ A ∧ b ∈ B }
  (favorable.card : ℚ) / (outcomes.card : ℚ)

theorem probability_of_sum_six :
  probability_sum_six {2, 3, 4} {2, 3, 4} (rfl) (rfl) = 1 / 3 :=
sorry

end probability_of_sum_six_l593_593708


namespace option_C_is_nonnegative_rational_l593_593395

def isNonNegativeRational (x : ℚ) : Prop :=
  x ≥ 0

theorem option_C_is_nonnegative_rational :
  isNonNegativeRational (-( - (4^2 : ℚ))) :=
by
  sorry

end option_C_is_nonnegative_rational_l593_593395


namespace graph_shift_eq_l593_593871

theorem graph_shift_eq (f g : ℝ → ℝ) (h1 : ∀ x, f x = sin (π / 3 - x))
  (h2 : ∀ x, g x = -cos (π / 3 - x)) :
  ∀ x, g x = f (x + π / 2) := 
sorry

end graph_shift_eq_l593_593871


namespace second_derivative_of_f_l593_593541

noncomputable def f : ℝ → ℝ := λ x, (Real.exp x / x) + (x * Real.sin x)

theorem second_derivative_of_f (x : ℝ) :
  (Deriv.deriv^[2] f) x = (2 * x * Real.exp x * (1 - x)) / x^4 + Real.cos x - x * Real.sin x :=
sorry

end second_derivative_of_f_l593_593541


namespace hyperbola_equation_and_area_l593_593952

-- Defining the hyperbola and its properties
def hyperbola := { a : ℝ // a > 0 }
def eccentricity (a : ℝ) := (sqrt (a^2 + 1)) / a

-- Given conditions
def line_l (x y : ℝ) := y = x - 2

theorem hyperbola_equation_and_area (a : ℝ) (ha : a > 0)
  (h : eccentricity a = (2 * sqrt 3) / 3) :
  ((a = sqrt 3) ∧
  ((∀ x y : ℝ, line_l x y → (x=y+2) → (∃ y1 y2 : ℝ, (2 * y ^ 2 - 4 * y - 1 = 0 ∧
  |y1 - y2| = sqrt 6 ∧
  let F1F2 := 2 * sqrt (a^2 + 1) in
  (S_triangle_ABF1 := 1 / 2 * F1F2 * sqrt 6 = 2 * sqrt 18))))))
:= sorry

end hyperbola_equation_and_area_l593_593952


namespace average_value_of_u_l593_593818

-- Define the function u(x)
def u (x : ℝ) : ℝ := sin (2 * x) ^ 2

-- State the problem: proving the average value of u on [0, ∞) is 0.5
theorem average_value_of_u : 
  (∀ (a b : ℝ), 0 ≤ a → a < b → 
    ∀ (x : ℝ), 0 ≤ x →
      (x = ∫ t in a..b, u t / (b - a))) 
    →
  (∃ L : ℝ, tendsto (λ x, (∫ t in 0..x, u t) / x) at_top (𝓝 L) ∧ L = 0.5) :=
sorry

end average_value_of_u_l593_593818


namespace find_point_P_l593_593866

open Real

-- Define the curve y = x^2
def curve1 (x : ℝ) : ℝ := x^2

-- Define the curve y = 1/x for x > 0
def curve2 (x : ℝ) (hx : x > 0) : ℝ := 1 / x

-- The given point (2, 4) is on the curve y = x^2
def point_on_curve1 : Prop := curve1 2 = 4

-- The tangent lines at the given points are perpendicular
def tangent_perpendicular (x1 y1 : ℝ) (slope1 slope2 : ℝ) : Prop := (slope1 * slope2 = -1)

-- Specify the conditions
def conditions (P : ℝ × ℝ) : Prop :=
  let P_x := P.1 in
  let P_y := P.2 in
  P_x > 0 ∧
  curve2 P_x (by norm_num) = P_y ∧
  let slope1 := 2 * 2 in
  let slope2 := -1 / (P_x ^ 2) in
  point_on_curve1 ∧ tangent_perpendicular 2 4 slope1 slope2

-- Theorem to prove that the coordinates of point P are (2, 1/2)
theorem find_point_P : ∃ P : ℝ × ℝ, conditions P ∧ P = (2, 1/2) :=
by {
  use (2, 1/2),
  split,
  {
    unfold conditions,
    simp [curve2, point_on_curve1, tangent_perpendicular],
    sorry
  },
  {
    refl
  }
}

end find_point_P_l593_593866


namespace marcy_cat_time_l593_593995

theorem marcy_cat_time (petting_time combing_ratio : ℝ) :
  petting_time = 12 ∧ combing_ratio = 1/3 → (petting_time + petting_time * combing_ratio) = 16 :=
by
  intros h
  cases h with petting_eq combing_eq
  rw [petting_eq, combing_eq]
  norm_num


end marcy_cat_time_l593_593995


namespace count_perfect_squares_between_50_and_250_l593_593232

theorem count_perfect_squares_between_50_and_250:
  ∃ (count : ℕ), count = 8 := by
  -- Define the set of perfect squares between 50 and 250
  let squares := { n | ∃ k : ℕ, n = k ^ 2 ∧ 50 < n ∧ n < 250 }
  -- Compute the number of elements in this set
  let count := (squares.filter λ n => 50 < n ∧ n < 250).card
  -- Assert that this number is 8
  have : count = 8 := sorry
  exact ⟨count, this⟩

end count_perfect_squares_between_50_and_250_l593_593232


namespace closed_set_sqrt3m_plus_n_operation_preserving_f_x_eq_x_l593_593918

-- Definitions as given in the problem.
def is_closed_set (M : Set ℝ) : Prop :=
  ∀ x y ∈ M, (x + y) ∈ M ∧ (x * y) ∈ M

def is_additive (f : ℝ → ℝ) (M : Set ℝ) : Prop :=
  ∀ x y ∈ M, f (x + y) = f x + f y

def is_multiplicative (f : ℝ → ℝ) (M : Set ℝ) : Prop :=
  ∀ x y ∈ M, f (x * y) = f x * f y

def is_operation_preserving (f : ℝ → ℝ) (M : Set ℝ) : Prop :=
  is_additive f M ∧ is_multiplicative f M

-- Questions to be proven
theorem closed_set_sqrt3m_plus_n :
  let M := {x : ℝ | ∃ m n : ℚ, x = √3 * m + n}
  in is_closed_set M :=
sorry

theorem operation_preserving_f_x_eq_x :
  ∃ f : ℚ → ℚ, is_operation_preserving f (Set.univ : Set ℚ) ∧ (∀ x, f x = x) :=
sorry

end closed_set_sqrt3m_plus_n_operation_preserving_f_x_eq_x_l593_593918


namespace probability_multiple_of_3_l593_593472

theorem probability_multiple_of_3 : 
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
                   (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
                   (3,1), (3,2), (3,3), (3,4), (3,5), (3,6),
                   (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
                   (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
                   (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)] in
  let multiples_of_3 := [(1,3), (3,1), (1,6), (6,1), (2,3), (3,2),
                         (2,6), (6,2), (3,3), (3,4), (4,3), (3,5),
                         (5,3), (3,6), (6,3), (4,6), (6,4), (5,6),
                         (6,5), (6,6)] in
  (multiples_of_3.length / outcomes.length : ℚ) = 5 / 9 := 
by
  sorry

end probability_multiple_of_3_l593_593472


namespace ball_arrangements_part1_ball_arrangements_part2_l593_593414

theorem ball_arrangements_part1 : 
  ∃ (arrangements : ℕ), arrangements = 12 ∧ 
    (∀ (balls : fin 4 → fin 3), 
      (∃ boxA boxB boxC, 
        boxA ≠ boxB ∧ boxA ≠ boxC ∧ boxB ≠ boxC ∧ 
        balls 3 = 1 ∧ 
        (boxA ≠ 0 ∧ boxB ≠ 0 ∧ boxC ≠ 0))) :=
sorry

theorem ball_arrangements_part2 : 
  ∃ (arrangements : ℕ), arrangements = 36 ∧ 
    (∀ (balls : fin 4 → fin 3), 
      (balls 1 ≠ 0 ∧ balls 2 ≠ 1)) :=
sorry

end ball_arrangements_part1_ball_arrangements_part2_l593_593414


namespace simplest_form_l593_593512

theorem simplest_form (b : ℝ) (h : b ≠ 2) : 2 - (2 / (2 + b / (2 - b))) = 4 / (4 - b) :=
by sorry

end simplest_form_l593_593512


namespace Paul_took_seven_sweets_l593_593379

theorem Paul_took_seven_sweets
  (initial_sweets : ℕ := 22)
  (Jack_took : ℕ := initial_sweets / 2 + 4)
  (remaining_after_Jack : ℕ := initial_sweets - Jack_took)
  (remaining_after_Paul := 0) :
  remaining_after_Paul = remaining_after_Jack - Paul_took :=
begin
  sorry
end

end Paul_took_seven_sweets_l593_593379


namespace original_selling_price_l593_593401

theorem original_selling_price (P : ℝ) (S : ℝ) (h1 : S = 1.10 * P) (h2 : 1.17 * P = 1.10 * P + 35) : S = 550 := 
by
  sorry

end original_selling_price_l593_593401


namespace max_diff_prime_set_correct_l593_593616

noncomputable def max_diff_prime_set (a b c : ℕ) (h_distinct: list.nodup [a, b, c, a+b-c, a+c-b, b+c-a, a+b+c]) (h_prime: ∀ x ∈ [a, b, c, a+b-c, a+c-b, b+c-a, a+b+c], Nat.Prime x) (h_sum800: a + b = 800 ∨ a + c = 800 ∨ b + c = 800) : ℕ :=
  2 * a

theorem max_diff_prime_set_correct (a b c : ℕ) 
  (h_distinct: list.nodup [a, b, c, a+b-c, a+c-b, b+c-a, a+b+c])
  (h_prime: ∀ x ∈ [a, b, c, a+b-c, a+c-b, b+c-a, a+b+c], Nat.Prime x)
  (h_sum800: a + b = 800 ∨ a + c = 800 ∨ b + c = 800) :
  max_diff_prime_set a b c h_distinct h_prime h_sum800 = 1594 := by
  sorry

end max_diff_prime_set_correct_l593_593616


namespace part1_part2_l593_593876

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x)

theorem part1 : f (Real.pi / 8) = Real.sqrt 2 + 1 := sorry

theorem part2 : (∀ x1 x2 : ℝ, f (x1 + Real.pi) = f x1) ∧ (∀ x : ℝ, f x ≥ 1 - Real.sqrt 2) := 
  sorry

-- Explanation:
-- part1 is for proving f(π/8) = √2 + 1
-- part2 handles proving the smallest positive period and the minimum value of the function.

end part1_part2_l593_593876


namespace initial_minutes_planA_equivalence_l593_593752

-- Conditions translated into Lean:
variable (x : ℝ)

-- Definitions for costs
def planA_cost_12 : ℝ := 0.60 + 0.06 * (12 - x)
def planB_cost_12 : ℝ := 0.08 * 12

-- Theorem we want to prove
theorem initial_minutes_planA_equivalence :
  (planA_cost_12 x = planB_cost_12) → x = 6 :=
by
  intro h
  -- complete proof is skipped with sorry
  sorry

end initial_minutes_planA_equivalence_l593_593752


namespace find_m_l593_593165

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l593_593165


namespace sin_315_eq_neg_sqrt2_div_2_l593_593067

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593067


namespace sin_315_eq_neg_sqrt2_div_2_l593_593007

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593007


namespace ratio_of_larger_to_smaller_l593_593699

theorem ratio_of_larger_to_smaller (a b : ℝ) (h : a > 0) (h' : b > 0) (h_sum_diff : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l593_593699


namespace correct_propositions_l593_593121

-- Definitions based on the conditions
variable f : ℝ → ℝ

-- Propositions to verify
def prop1 : Prop := ∀ x, f(1-x) = f(x-1) → ∀ y, y = f(x) ∧ (∀ y, y = f(0)) ↔ (∀ x, f(x-1) = f(1-x))
def prop2 : Prop := ∀ x, f(1-x) = f(x-1) → graph (y=f(x))  symmetric_to x=1
def prop3 : Prop := ∀ x, f(1+x) = f(x-1) → periodic f 2
def prop4 : Prop := ∀ x, f(1-x) = -f(x-1) → odd f

-- Theorem to state the correct propositions
theorem correct_propositions : prop3 ∧ prop4 ∧ ¬prop1 ∧ ¬prop2 := 
by 
  -- skip the proof with sorry.
  sorry

end correct_propositions_l593_593121


namespace ellipse_equation_lambda_mu_const_l593_593453

-- Declare the variables and constants
variable (a b : ℝ)
variable (h1 : 0 < b)  -- b > 0
variable (h2 : b < a)  -- a > b

-- Given conditions
-- Condition 1: Equation of the ellipse
def ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- Condition 2: Eccentricity e = 1/2
def eccentricity := (b : ℝ) = a * (√3 / 2)

-- Condition 3: Radius of inscribed circle of quadrilateral is √3 / 2
def inradius := (√3 / 2) = (√3 / 2)

-- Theorem to prove the equation of the ellipse C
theorem ellipse_equation : 4 * x^2 + 3 * y^2 = 4 * a^2 :=
sorry

-- Further conditions and proofs for the second part
variable (F1 M N P : point) -- Points F1(left focus), M, N on the ellipse, and P on x = -4
variable (hF1 : F1.x = -a / 2) -- Coordinate of left focus
variable (hP : P.x = -4) -- Point P on the line x=-4
variable (lambda mu : ℝ)

-- Vectors representation
def PM := vector P M
def MF1 := vector M F1
def PN := vector P N
def NF1 := vector N F1

-- Relationship of vectors
def pm_eq := PM = λ * MF1
def pn_eq := PN = μ * NF1

-- Theorem to prove λ + μ is a constant
theorem lambda_mu_const : ∃ k : ℝ, λ + μ = k :=
sorry

end ellipse_equation_lambda_mu_const_l593_593453


namespace crossed_buses_l593_593789

-- Define the conditions
def leaves_from_dallas (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def leaves_from_austin (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def travel_time := 6

-- Statement to prove
theorem crossed_buses : 
  ∀ (start_time : ℕ), leaves_from_austin start_time → 
  start_time + travel_time ≤ 24 → 
  let end_time := start_time + travel_time in
  let intervals := list.range (end_time / 2) in
  intervals.filter (λ t, t ≥ (start_time / 2) ∧ leaves_from_dallas t) = [0, 1, 2] :=
by
  sorry

end crossed_buses_l593_593789


namespace sin_315_eq_neg_sqrt2_div_2_l593_593074

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593074


namespace measure_of_angle_BPC_l593_593599

-- Define the points and structures
variables (A B C D E P Q : Point)
variables (AB BC CD DA BE AC PQ : Line)
variables (y : ℝ)

-- Square and equilateral triangle condition
def square_ABCD : Prop := is_square A B C D ∧ length (segment A B) = 6
def equilateral_triangle_ABE : Prop := is_equilateral A B E ∧ length (segment A B) = 6

-- Intersection and perpendicular conditions
def intersect_BE_AC_at_P : Prop := is_intersecting BE AC P
def point_Q_on_BC_with_perpendicular_PQ : Prop := is_on_line Q BC ∧ is_perpendicular PQ BC ∧ length PQ = y

-- The final angle to prove
def angle_BPC : ℝ := 105

-- The final proof statement
theorem measure_of_angle_BPC 
  (h1 : square_ABCD A B C D)
  (h2 : equilateral_triangle_ABE A B E)
  (h3 : intersect_BE_AC_at_P P)
  (h4 : point_Q_on_BC_with_perpendicular_PQ Q) :
  measure_angle B P C = angle_BPC := 
sorry -- proof to be completed

end measure_of_angle_BPC_l593_593599


namespace smallest_x_for_palindrome_l593_593721

-- Define the condition for a number to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Mathematically equivalent proof problem statement
theorem smallest_x_for_palindrome : ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 2345) ∧ x = 97 :=
by sorry

end smallest_x_for_palindrome_l593_593721


namespace range_of_m_l593_593276

theorem range_of_m (A B : ℝ × ℝ) (hA : A = (1, 0)) (hB : B = (4, 0)) (m : ℝ) :
  (∃ P : ℝ × ℝ, (P.1 + m * P.2 - 1 = 0) ∧ (real.sqrt ((P.1 - 1)^2 + P.2^2) = 2 * real.sqrt ((P.1 - 4)^2 + P.2^2))) →
  m ≥ real.sqrt 3 ∨ m ≤ -real.sqrt 3 :=
by
  sorry

end range_of_m_l593_593276


namespace power_function_k_values_l593_593577

theorem power_function_k_values (k : ℝ) : (∃ (a : ℝ), (k^2 - k - 5) = a) → (k = 3 ∨ k = -2) :=
by
  intro h
  have h1 : k^2 - k - 5 = 1 := sorry -- Using the condition that it is a power function
  have h2 : k^2 - k - 6 = 0 := by linarith -- Simplify the equation
  exact sorry -- Solve the quadratic equation

end power_function_k_values_l593_593577


namespace min_result_of_expr_l593_593929

def min_result_expr : Int :=
  1 - 2 * 6 * 9

theorem min_result_of_expr : 
  (∀ f: List (Int → Int → Int), f ∈ [List.replicate 4 (+), List.replicate 4 (-), List.replicate 4 (*), List.replicate 4 (/)] →
  let expr := f.head (f.get 1 (f.get 2 (f.get 3 (1, 2), 6), 9)))
  expr ≥ min_result_expr) := 
  sorry

end min_result_of_expr_l593_593929


namespace find_m_l593_593156

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l593_593156


namespace sum_of_squares_l593_593662

theorem sum_of_squares (n : ℕ) : 
  (\sum i in Finset.range (n + 1), i^2) = n * (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

end sum_of_squares_l593_593662


namespace perfect_squares_l593_593685

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593685


namespace find_first_class_equipment_l593_593455

def higher_class_equipment_at_first_site (x : ℕ) : Prop :=
  x < 60

def first_class_equipment_at_second_site (y : ℕ) : Prop :=
  60 = y

def condition_equipment_transfers (x y : ℕ) : Prop :=
  7 * x + 5 * y = 650 ∧ 32 * x > 25 * y

theorem find_first_class_equipment (x y : ℕ) : 
  higher_class_equipment_at_first_site x →
  first_class_equipment_at_second_site y →
  condition_equipment_transfers x y →
  y = 60 :=
by
  intro h1 h2 h3
  cases h2
  exact h2

end find_first_class_equipment_l593_593455


namespace rotation_problem_l593_593970

theorem rotation_problem (P Q R : Type) (rotate_clockwise rotate_counterclockwise : Type → ℕ → Type → Type)
  (780_clockwise_landing : rotate_clockwise P 780 Q = R)
  (y_counterclockwise_landing : rotate_counterclockwise P y Q = R)
  (h_y_lt_360 : y < 360) : y = 300 := 
sorry

end rotation_problem_l593_593970


namespace sin_315_degree_l593_593051

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593051


namespace thm_b_thm_c_thm_d_l593_593144

-- Define non-zero vectors
variable (a b c : ℝ^3)

-- Define the inner product operation
def inner_product (v1 v2 : ℝ^3) : ℝ := v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Condition 1: Vectors a, b, c are non-zero
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- Theorem for condition B
theorem thm_b (h : c = a - (inner_product a a / inner_product a b) * b) : inner_product a c = 0 := sorry

-- Theorem for condition C
theorem thm_c (h : a = (norm b) * c) : ∃ k : ℝ, a = k • c := sorry

-- Theorem for condition D
theorem thm_d (h : norm (a - b) = norm a + norm b) : ∃ k : ℝ, a = k • b := sorry

end thm_b_thm_c_thm_d_l593_593144


namespace number_of_segments_divisible_by_6_l593_593421

/--
A closed broken line consists of segments of equal length with every three consecutive segments being pairwise perpendicular.
-/
def closed_broken_line (segments : List (ℝ × ℝ × ℝ)) : Prop :=
  (∀ (i : ℕ), i < segments.length → (∀ (j : ℕ), j < 3 →
    segments.nthLe i sorry ∉ set.range (vec_add (segments.nthLe ((i + j) % segments.length) sorry))) ∧
    ((list.cycle segments).nthLe i sorry = list.cycle segments).nthLe ((i + segments.length) % segments.length) sorry) ∧
  ∀ (i : ℕ), i < segments.length - 2 → 
    let s1 := (list.cycle segments).nthLe i sorry,
        s2 := (list.cycle segments).nthLe (i + 1) sorry,
        s3 := (list.cycle segments).nthLe (i + 2) sorry in
        s1.dot s2 = 0 ∧ s2.dot s3 = 0 ∧ s3.dot s1 = 0

theorem number_of_segments_divisible_by_6 (segments : List (ℝ × ℝ × ℝ)) 
  (h_closed : closed_broken_line segments) : 
  segments.length % 6 = 0 := 
sorry

end number_of_segments_divisible_by_6_l593_593421


namespace player_A_guarantee_win_l593_593301

-- Define a cyclic polygon and the circumcenter O
structure Polygon (V : Type) :=
(vertices : List V)
(cyclic : Prop)

-- Define when a polygon is acute-angled
def is_acute_angled {V : Type} [Euclidean_geometry : MetricSpace V]: Polygon V → Prop := sorry

-- Define the Matcha Sweep Game
structure MatchaSweepGame {V : Type} :=
(points : Set V)
(cyclic_polygon : Polygon V)
(circumcenter : V)
(not_on_diagonal : Prop) -- Ensure O is not on any diagonal
(has_winning_strategy_A : Prop)

-- Define the main theorem
theorem player_A_guarantee_win (V : Type) [MetricSpace V] 
  (P : Polygon V) (O : V) (S : Set V) (not_on_diagonal:Prop): (MatchaSweepGame := { points := S, cyclic_polygon := P, circumcenter := O, not_on_diagonal := not_on_diagonal, has_winning_strategy_A := sorry }) → ¬is_acute_angled P ↔ (MatchaSweepGame.has_winning_strategy_A := sorry) :=
sorry

end player_A_guarantee_win_l593_593301


namespace probability_same_color_l593_593418

/--
A bag contains 8 green balls, 6 white balls, 5 red balls, and 4 blue balls.
If three balls are drawn simultaneously, what is the probability that all 
three balls are from the same color?
--/
def same_color_probability : ℚ :=
  let total_ways := Nat.choose 23 3 in
  let green_ways := Nat.choose 8 3 in
  let white_ways := Nat.choose 6 3 in
  let red_ways := Nat.choose 5 3 in
  let blue_ways := Nat.choose 4 3 in
  ((green_ways + white_ways + red_ways + blue_ways) : ℚ) / (total_ways : ℚ)

/-- 
The probability that all three balls drawn are from the same color is 90/1771.
--/
theorem probability_same_color (h : same_color_probability = 90 / 1771) : true :=
  by
    sorry

end probability_same_color_l593_593418


namespace sin_315_eq_neg_sqrt2_div_2_l593_593013

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593013


namespace BE_length_l593_593604

def triangle (A B C : Type) := 
  angle A B C = 90 -- Assuming there's a predefined structure for angles in Mathlib

variables {A B C D E : Type}

theorem BE_length 
  (ABC_right : ∠ C = 90)
  (AC : ℝ) (AC_length : AC = 9) 
  (BC : ℝ) (BC_length : BC = 12) 
  (D_on_AB : D ∈ line(A, B))
  (E_on_BC : E ∈ line(B, C))
  (angle_BED : ∠ BED = 90)
  (DE : ℝ) (DE_length : DE = 6) :
  ∃ (BE : ℝ), BE = 8 :=
by {
  sorry
}

end BE_length_l593_593604


namespace even_and_odd_functions_satisfying_equation_l593_593856

theorem even_and_odd_functions_satisfying_equation :
  ∀ (f g : ℝ → ℝ),
    (∀ x : ℝ, f (-x) = f x) →                      -- condition 1: f is even
    (∀ x : ℝ, g (-x) = -g x) →                    -- condition 2: g is odd
    (∀ x : ℝ, f x - g x = x^3 + x^2 + 1) →        -- condition 3: f(x) - g(x) = x^3 + x^2 + 1
    f 1 + g 1 = 1 :=                              -- question: proof of f(1) + g(1) = 1
by
  intros f g h_even h_odd h_eqn
  sorry

end even_and_odd_functions_satisfying_equation_l593_593856


namespace find_m_l593_593172

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l593_593172


namespace find_missing_number_l593_593925

theorem find_missing_number 
  (x : ℝ) (y : ℝ)
  (h1 : (12 + x + 42 + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + y + 1023 + x) / 5 = 398.2) :
  y = 511 := 
sorry

end find_missing_number_l593_593925


namespace find_x_y_z_l593_593307

theorem find_x_y_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x * y = x + y) (h2 : y * z = 3 * (y + z)) (h3 : z * x = 2 * (z + x)) : 
  x + y + z = 12 :=
sorry

end find_x_y_z_l593_593307


namespace conic_section_focus_and_type_l593_593858

variables {θ : ℝ}

def m : ℝ × ℝ := (Real.sin θ, Real.cos θ)
def n : ℝ × ℝ := (1, 1)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem conic_section_focus_and_type
  (hθ_internal : θ ∈ Set.Icc 0 π)
  (h_dot : dot_product m n = 1 / 3) :
  ∃ k : ℝ, (k > 0) ∧ (k ≠ 1) ∧ (∀ x y : ℝ, x^2 * Real.sin θ - y^2 * Real.cos θ = 1 → y = 0) ∧ (∀ x y : ℝ, x^2 * Real.sin θ - y^2 * Real.cos θ = 1 → True) :=
sorry

end conic_section_focus_and_type_l593_593858


namespace hong_travel_limit_l593_593264

-- Definition of the problem conditions
def infinite_towns : Prop := true -- Infinity of towns is a conceptual condition

def road_connects (A B : Type) : Prop := ∃ (road : Type), road ∈ A → road ∈ B -- Each pair of towns is connected by a road

def initial_coins (n : ℕ) (town : Type) : Prop := ∀ (x : town), coins x = n -- Each town starts with n coins

def travel (A B : Type) (k : ℕ) (travel_day : ℕ) : Prop := 
A ≠ B ∧ travel_day = k ∧ sends_coins B A k ∧ unusable_after A B -- Traveling constraints

-- Mathematically equivalent proof problem
theorem hong_travel_limit (n : ℕ) : infinite_towns ∧
  (∀ (A B : Type), road_connects A B) ∧ 
  (initial_coins n) ∧
  (∀ (A B : Type) (k : ℕ) (d : ℕ), travel A B d → d = k → sends_coins B A k → unusable_after A B) → 
  (max_travel_days n <= n + 2 * n^(2/3)) :=
sorry

end hong_travel_limit_l593_593264


namespace min_colors_needed_l593_593738

-- Defining the problem setup
variable (n : ℕ) -- number of circles
variable (regions : Set (Set ℕ)) -- set of regions split by n circles
variable (adjacency : Set (ℕ × ℕ)) -- adjacency relationships between regions

-- Condition: Adjacent regions must be colored differently
def valid_coloring (coloring : ℕ → ℕ) : Prop :=
  ∀ {i j : ℕ}, (i, j) ∈ adjacency → coloring i ≠ coloring j

-- Theorem statement: Prove that 2 colors are sufficient
theorem min_colors_needed : ∃ (coloring : ℕ → ℕ), valid_coloring coloring ∧ ∀ (i j : ℕ), coloring i ∈ {0, 1} :=
  sorry

end min_colors_needed_l593_593738


namespace min_frac_sum_l593_593131

theorem min_frac_sum (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : 2 * a + b = 1) : 
  (3 / b + 2 / a) = 7 + 4 * Real.sqrt 3 := 
sorry

end min_frac_sum_l593_593131


namespace total_canvas_area_l593_593319

theorem total_canvas_area (rect_length rect_width tri1_base tri1_height tri2_base tri2_height : ℕ)
    (h1 : rect_length = 5) (h2 : rect_width = 8)
    (h3 : tri1_base = 3) (h4 : tri1_height = 4)
    (h5 : tri2_base = 4) (h6 : tri2_height = 6) :
    (rect_length * rect_width) + ((tri1_base * tri1_height) / 2) + ((tri2_base * tri2_height) / 2) = 58 := by
  sorry

end total_canvas_area_l593_593319


namespace proof_x_equals_90_l593_593690

open List

def data : List ℕ := [60, 100, x, 40, 50, 200, 90]

noncomputable def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / (l.length : ℚ)

theorem proof_x_equals_90 (x : ℕ) (h_mean : mean data = x) (h_median : data.nth 3 = some x) (h_mode : ∀ n ∈ data, n ≠ x → count n data ≤ 1) : x = 90 :=
by
  sorry

end proof_x_equals_90_l593_593690


namespace g_odd_l593_593286

def g (x : ℝ) : ℝ := x^3 - 2*x

theorem g_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end g_odd_l593_593286


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593021

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593021


namespace complex_number_in_first_quadrant_l593_593857

noncomputable def is_in_first_quadrant (x y : ℝ) : Prop :=
x > 0 ∧ y > 0

theorem complex_number_in_first_quadrant (x y : ℝ) (h : x / (1 + 1 : ℂ) = 1 - y * (complex.I : ℂ)) :
  is_in_first_quadrant x y := 
sorry

end complex_number_in_first_quadrant_l593_593857


namespace isosceles_triangle_square_vertices_l593_593982

theorem isosceles_triangle_square_vertices (A B C : Type) (distance : A → A → ℝ) (AB AC : ℝ) :
  AB = AC →
  ∃ n : ℕ, n = 2 ∧ ∀ S : Type, (∃ (p1 p2 : A), S = {p1, p2}) →
  ∃ (count : ℕ), count = 2 := 
begin
  sorry
end

end isosceles_triangle_square_vertices_l593_593982


namespace arrangement_plans_count_l593_593100

noncomputable def number_of_arrangement_plans (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
if num_teachers = 2 ∧ num_students = 4 then 12 else 0

theorem arrangement_plans_count :
  number_of_arrangement_plans 2 4 = 12 :=
by 
  sorry

end arrangement_plans_count_l593_593100


namespace largest_prime_factor_of_sum_of_divisors_of_450_l593_593980

open Nat

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, d

theorem largest_prime_factor_of_sum_of_divisors_of_450 : 
  let M := sum_of_divisors 450 in
  prime M :=
by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_450_l593_593980


namespace g_correct_l593_593246

noncomputable def f : ℚ[X] := 2 * X ^ 4 - X ^ 2 + 3 * X - 7

noncomputable def g : ℚ[X] := -2 * X ^ 4 + 4 * X ^ 2 - 2 * X + 2

theorem g_correct : (f + g = 3 * X ^ 2 + X - 5) ∧ (f = 2 * X ^ 4 - X ^ 2 + 3 * X - 7) → (g = -2 * X ^ 4 + 4 * X ^ 2 - 2 * X + 2) := 
by
  sorry

end g_correct_l593_593246


namespace num_orders_javier_constraint_l593_593293

noncomputable def num_valid_orders : ℕ :=
  Nat.factorial 5 / 2

theorem num_orders_javier_constraint : num_valid_orders = 60 := 
by
  sorry

end num_orders_javier_constraint_l593_593293


namespace parabola_focus_distance_l593_593957

theorem parabola_focus_distance (p : ℝ) : 
  (∀ (y : ℝ), y^2 = 2 * p * 4 → abs (4 + p / 2) = 5) → 
  p = 2 :=
by
  sorry

end parabola_focus_distance_l593_593957


namespace buns_problem_l593_593643

theorem buns_problem (N : ℕ) (x y u v : ℕ) 
  (h1 : 3 * x + 5 * y = 25)
  (h2 : 3 * u + 5 * v = 35)
  (h3 : x + y = N)
  (h4 : u + v = N) : 
  N = 7 := 
sorry

end buns_problem_l593_593643


namespace solve_inequality_l593_593280

theorem solve_inequality :
  {x : ℝ | |2 * x - 1| + |2 * x + 1| ≤ 6} = set.Icc (-3/2 : ℝ) (3/2 : ℝ) :=
by sorry

end solve_inequality_l593_593280


namespace find_f_l593_593309

noncomputable def f : ℝ → ℝ
| x := if hx : 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0

theorem find_f (f_defined : ∀ (x : ℝ), 0 ≤ x → (f x = if hx : 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0)) :
  (∀ x y, 0 ≤ x → 0 ≤ y → f (x * f y) * f y = f (x + y)) ∧ 
  (f 2 = 0) ∧ 
  (∀ x, 0 ≤ x → x < 2 → f x ≠ 0) :=
by
  sorry

end find_f_l593_593309


namespace true_propositions_count_l593_593206

theorem true_propositions_count :
  let proposition1 := ¬(∃ x : ℝ, x^2 + 1 > 3 * x) = ∀ x : ℝ, x^2 + 1 <= 3 * x
  let proposition2 := (∀ a > 2, a > 5) ∧ ¬(∀ a > 5, a > 2)
  let proposition3 := ¬(xy = 0 → x = 0 ∧ y = 0)
  let proposition4 := (¬ (p ∨ q) → ¬p ∧ ¬q)
  in (if proposition4 then 1 else 0) = 1 := sorry

end true_propositions_count_l593_593206


namespace train_crossing_time_l593_593402

theorem train_crossing_time (L : ℝ) (v_train_kmh v_man_kmh : ℝ)
    (L_eq : L = 500)
    (v_train_eq : v_train_kmh = 63)
    (v_man_eq : v_man_kmh = 3) :
    L / ((v_train_kmh * (1000 / 3600)) - (v_man_kmh * (1000 / 3600))) = 30 :=
by
    rw [L_eq, v_train_eq, v_man_eq]
    -- Convert speeds from km/hr to m/s
    suffices : L / ((63 * (1000 / 3600)) - (3 * (1000 / 3600))) = 30
    exact this
    -- Simplify the expression
    norm_num
    sorry

end train_crossing_time_l593_593402


namespace g_symmetric_solutions_l593_593627

def g : ℝ → ℝ :=
sorry -- The exact definition isn't necessary for the problem statement.

theorem g_symmetric_solutions :
  (∀ x : ℝ, x ≠ 0 → g(x) + 3 * g(1 / x) = 4 * x^2) →
  ∀ x : ℝ, x ≠ 0 → g(x) = g(-x) :=
sorry

end g_symmetric_solutions_l593_593627


namespace monotonic_intervals_of_f_inequality_for_zeros_of_g_l593_593873

-- Proof Problem 1: Monotonicity of f(x)
theorem monotonic_intervals_of_f (a : ℝ) :
  (a ≥ 0 → ∀ x y : ℝ, x < y → f x ≤ f y) ∧
  (a < 0 → ∀ x y : ℝ, (x < ln(-a) ∧ y < ln(-a) → f x ≥ f y) ∧ (x > ln(-a) ∧ y > ln(-a) → f x ≤ f y)) :=
by
  let f (x : ℝ) := Real.exp x + a * x
  sorry

-- Proof Problem 2: Inequality involving zeros of g(x)
theorem inequality_for_zeros_of_g (m : ℝ) (x₁ x₂ : ℝ) (hx : x₁ < x₂)
  (h₀ : g x₁ = 0) (h₁ : g x₂ = 0) :
  2 * Real.log x₁ + Real.log x₂ > Real.exp 1 :=
by
  let g (x : ℝ) := Real.exp (m * x) - Real.log x + (m - 1) * x
  sorry

end monotonic_intervals_of_f_inequality_for_zeros_of_g_l593_593873


namespace second_transfer_amount_l593_593641

theorem second_transfer_amount :
  ∃ (X : ℝ), 
  (let initial_balance := 400 in
   let final_balance := 307 in
   let first_transfer := 90 in
   let second_transfer := X in
   let service_charge := 0.02 in
   let total_deducted := (first_transfer + first_transfer * service_charge) + 
                         (second_transfer + second_transfer * service_charge) - 
                         second_transfer in
   initial_balance - final_balance = total_deducted) →
   X = 60 :=
by
  sorry

end second_transfer_amount_l593_593641


namespace range_of_m_l593_593097

def op_⊗ (x y : ℝ) : ℝ :=
if x ≤ y then x else y

theorem range_of_m (m : ℝ) :
  (op_⊗ (abs (m - 1)) m = abs (m - 1)) → m ≥ 1 / 2 :=
by 
  intros h
  sorry -- The proof steps would go here

end range_of_m_l593_593097


namespace find_m_l593_593160

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l593_593160


namespace range_of_a_l593_593552

def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then -x^2 + a * x + a + 1 else a * x + 4

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ a ∈ set.Iio 2 ∪ set.Ioi 4 :=
by
  sorry

end range_of_a_l593_593552


namespace factor_of_quadratic_polynomial_l593_593816

theorem factor_of_quadratic_polynomial (t : ℚ) :
  (8 * t^2 + 22 * t + 5 = 0) ↔ (t = -1/4) ∨ (t = -5/2) :=
by sorry

end factor_of_quadratic_polynomial_l593_593816


namespace choose_5_person_committee_l593_593944

theorem choose_5_person_committee : nat.choose 12 5 = 792 := 
by
  sorry

end choose_5_person_committee_l593_593944


namespace constant_k_value_l593_593579

theorem constant_k_value 
  (S : ℕ → ℕ)
  (h : ∀ n : ℕ, S n = 4 * 3^(n + 1) - k) :
  k = 12 :=
sorry

end constant_k_value_l593_593579


namespace sin_315_eq_neg_sqrt2_over_2_l593_593037

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593037


namespace gym_charges_per_payment_l593_593431

theorem gym_charges_per_payment :
  ∀ (members payments_per_month : ℕ) (total_income : ℕ),
    members = 300 ->
    payments_per_month = 2 ->
    total_income = 10800 ->
    (total_income / (members * payments_per_month)) = 18 := by
  intros members payments_per_month total_income Hm Hp Hi
  rw [Hm, Hp, Hi]
  calc
    10800 / (300 * 2) = 10800 / 600 : by rfl
    ... = 18 : by norm_num

end gym_charges_per_payment_l593_593431


namespace estimate_sqrt_19_between_6_and_7_l593_593496

theorem estimate_sqrt_19_between_6_and_7 (a b : ℝ) (h₁ : 4^2 = 16) (h₂ : 5^2 = 25) (h₃ : 16 < 19) (h₄ : 19 < 25) : 
  6 < 2 + sqrt 19 ∧ 2 + sqrt 19 < 7 :=
by
  have h₅ : 4 < sqrt 19,
  { sorry },
  have h₆ : sqrt 19 < 5,
  { sorry },
  split,
  { linarith },
  { linarith }

end estimate_sqrt_19_between_6_and_7_l593_593496


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593022

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593022


namespace expand_product_l593_593813

theorem expand_product (x : ℝ) :
  (2 * x^2 - 3 * x + 5) * (x^2 + 4 * x + 3) = 2 * x^4 + 5 * x^3 - x^2 + 11 * x + 15 :=
by
  -- Proof to be filled in
  sorry

end expand_product_l593_593813


namespace perfect_squares_count_in_range_l593_593237

theorem perfect_squares_count_in_range :
  let s := {x : ℕ | 50 ≤ x ∧ x ≤ 250 ∧ ∃ (n : ℕ), n^2 = x}
  ∈ (∃ (count : ℕ), count = 8) :=
by
  let s := {x : ℕ | 50 ≤ x ∧ x ≤ 250 ∧ ∃ (n : ℕ), n^2 = x}
  sorry

end perfect_squares_count_in_range_l593_593237


namespace num_four_digit_div_by_4_part_a_num_four_digit_div_by_4_part_b_l593_593228

def valid_last_two_digits (d1 d2 : ℕ) : Prop :=
  (d1 = 1 ∧ d2 = 2) ∨ (d1 = 2 ∧ d2 = 4) ∨ (d1 = 3 ∧ d2 = 2) ∨ (d1 = 4 ∧ d2 = 4)

theorem num_four_digit_div_by_4_part_a :
  (∃ n, n = 6 ∧ (∀ d1 d2 d3 d4 : ℕ, ∀ h1 h2 h3 h4 : ℕ, 
      h1 = d1 ∧ h2 = d2 ∧ h3 = d3 ∧ h4 = d4 ∧ 
      d1 ∈ {1, 2, 3, 4} ∧ d2 ∈ {1, 2, 3, 4} ∧ d3 ∈ {1, 2, 3, 4} ∧ d4 ∈ {1, 2, 3, 4} ∧ 
      (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4) ∧ 
      valid_last_two_digits d3 d4 → 
      ∃ n, n = d1*1000 + d2*100 + d3*10 + d4)) := sorry

noncomputable def valid_last_two_digits_repeat (d1 d2 : ℕ) : Prop :=
  (d1 = 1 ∧ d2 = 2) ∨ (d1 = 2 ∧ d2 = 4) ∨ (d1 = 3 ∧ d2 = 2) ∨ (d1 = 4 ∧ d2 = 4)

theorem num_four_digit_div_by_4_part_b :
  (∃ n, n = 64 ∧ (∀ d1 d2 d3 d4 : ℕ, ∀ h1 h2 h3 h4 : ℕ, 
    h1 = d1 ∧ h2 = d2 ∧ h3 = d3 ∧ h4 = d4 ∧ 
    d1 ∈ {1, 2, 3, 4} ∧ d2 ∈ {1, 2, 3, 4} ∧ d3 ∈ {1, 2, 3, 4} ∧ d4 ∈ {1, 2, 3, 4} ∧ 
    valid_last_two_digits_repeat d3 d4 → 
    ∃ n, n = d1*1000 + d2*100 + d3*10 + d4)) := sorry

end num_four_digit_div_by_4_part_a_num_four_digit_div_by_4_part_b_l593_593228


namespace perimeter_of_garden_l593_593663

def area (length width : ℕ) : ℕ := length * width

def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem perimeter_of_garden :
  ∀ (l w : ℕ), area l w = 28 ∧ l = 7 → perimeter l w = 22 := by
  sorry

end perimeter_of_garden_l593_593663


namespace max_distance_circle_to_line_l593_593362

-- Definitions for the circle and line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y = 0
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Proof statement
theorem max_distance_circle_to_line 
  (x y : ℝ)
  (h_circ : circle_eq x y)
  (h_line : ∀ (x y : ℝ), line_eq x y → true) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 :=
sorry

end max_distance_circle_to_line_l593_593362


namespace increasing_function_on_interval_l593_593565

theorem increasing_function_on_interval (m : ℝ) :
  (∀ x ∈ Ioo (-1 : ℝ) 1, -3 * x^2 + 2 * x + m ≥ 0) ↔ m ≥ 5 := 
sorry

end increasing_function_on_interval_l593_593565


namespace flag_covers_hole_l593_593709

noncomputable def cover_hole (A : ℝ × ℝ) (flag_points : set (ℝ × ℝ)) (B : ℝ × ℝ) : Prop :=
  ∀ (p ∈ flag_points), ∃ (q ∈ flag_points), (p + q) / 2 = A

theorem flag_covers_hole {A : ℝ × ℝ} {flag_points : set (ℝ × ℝ)} :
  ∃ (B : ℝ × ℝ), cover_hole A flag_points B :=
sorry

end flag_covers_hole_l593_593709


namespace part_one_and_two_l593_593542

noncomputable def f (x : ℝ) := log (2, -x^2 + 2*x + 3)
noncomputable def g (x : ℝ) := 1 / x

def A (x : ℝ) : Prop := -x ^ 2 + 2 * x + 3 > 0
def B (y : ℝ) : Prop := y ∈ (-∞, -(1 / 3)) ∨ y ∈ (1, ∞)
def C (x : ℝ) (m : ℝ) : Prop := 2 * x ^ 2 + m * x - 8 < 0

theorem part_one_and_two (m : ℝ) :
  let domain_A := {x : ℝ | A x},
      range_B := {y : ℝ | B y},
      set_complement_B := {z : ℝ | ¬ B z},
      union_equals : domain_A ∪ set_complement_B = (-1, 3),
      intersection_equals : domain_A ∩ range_B = (-1, - (1 / 3)) ∪ (1, 3) in
  (∀ x, domain_A ∩ range_B ⊆ {x : ℝ | C x m}) →
  -6 ≤ m ∧ m ≤ - (10 / 3) := 
by sorry

end part_one_and_two_l593_593542


namespace marcy_total_time_l593_593997

theorem marcy_total_time 
    (petting_time : ℝ)
    (fraction_combing : ℝ)
    (H1 : petting_time = 12)
    (H2 : fraction_combing = 1/3) :
    (petting_time + (fraction_combing * petting_time) = 16) :=
  sorry

end marcy_total_time_l593_593997


namespace sin_315_eq_neg_sqrt2_div_2_l593_593058

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593058


namespace sin_315_eq_neg_sqrt2_div_2_l593_593011

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593011


namespace fraction_simplification_l593_593337

variable (b x : ℝ)

theorem fraction_simplification :
  (sqrt (b^2 + x^4) - (x^4 - b^4) / sqrt (b^2 + x^4)) / (b^2 + x^4) = (b^4) / ((b^2 + x^4) ^ (3/2)) :=
by
  sorry

end fraction_simplification_l593_593337


namespace find_x_l593_593883

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x + 1 else |x|

theorem find_x (x : ℝ) : f x = 3 ↔ x = -3 ∨ x = 1 := by
  sorry

end find_x_l593_593883


namespace count_perfect_squares_between_50_and_250_l593_593230

theorem count_perfect_squares_between_50_and_250:
  ∃ (count : ℕ), count = 8 := by
  -- Define the set of perfect squares between 50 and 250
  let squares := { n | ∃ k : ℕ, n = k ^ 2 ∧ 50 < n ∧ n < 250 }
  -- Compute the number of elements in this set
  let count := (squares.filter λ n => 50 < n ∧ n < 250).card
  -- Assert that this number is 8
  have : count = 8 := sorry
  exact ⟨count, this⟩

end count_perfect_squares_between_50_and_250_l593_593230


namespace question_A_question_C_question_D_l593_593879

-- Definitions for the conditions
def f (x : ℝ) : ℝ := abs (exp x - 1)

variables (x1 x2 : ℝ)
hypothesis1 : x1 < 0
hypothesis2 : x2 > 0
hypothesis3 : tangent_perpendicular : ∀ x₁ x₂, x₁ < 0 → x₂ > 0 → 
  (fderiv ℝ f x₁) * (fderiv ℝ f x₂) = -1

-- Definitions for the statements to be proved
theorem question_A (x1 x2 : ℝ) 
  (h1 : x1 < 0) 
  (h2 : x2 > 0) 
  (h3 : tangent_perpendicular x1 x2 h1 h2) : 
  x1 + x2 = 0 := 
sorry

theorem question_C (x1 x2 : ℝ) 
  (h1 : x1 < 0) 
  (h2 : x2 > 0) 
  (h3 : tangent_perpendicular x1 x2 h1 h2) : 
  0 < ((f x2) - (f x1)) / (x2 - x1) := 
sorry

theorem question_D (x1 x2 : ℝ) 
  (h1 : x1 < 0) 
  (h2 : x2 > 0) 
  (h3 : tangent_perpendicular x1 x2 h1 h2) : 
  0 < exp x1 ∧ exp x1 < 1 := 
sorry

end question_A_question_C_question_D_l593_593879


namespace sin_315_eq_neg_sqrt2_div_2_l593_593063

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593063


namespace choose_5_from_12_l593_593942

theorem choose_5_from_12 : Nat.choose 12 5 = 792 := by
  sorry

end choose_5_from_12_l593_593942


namespace find_m_l593_593166

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l593_593166


namespace max_attendees_eq_two_l593_593809

-- Define days of the week
inductive Day : Type
| Mon | Tues | Wed | Thurs | Fri 

open Day

-- Define people
inductive Person : Type
| Alice | Bob | Cindy | David

open Person

-- Define non-attendance by each person
def cannot_attend (p : Person) : Day → Prop
| Alice := λ d, d = Mon ∨ d = Tues ∨ d = Fri
| Bob := λ d, d = Tues ∨ d = Wed
| Cindy := λ d, d = Mon ∨ d = Wed ∨ d = Thurs
| David := λ d, d = Thurs ∨ d = Fri

-- Define the set of all persons
def all_persons : List Person := [Alice, Bob, Cindy, David]

-- Calculate number of attendees on a given day
def attendees (d : Day) : ℕ :=
(all_persons.filter (λ p, ¬ cannot_attend p d)).length

-- Theorem: Prove that the maximum number of attendees is 2 on any day
theorem max_attendees_eq_two (d : Day) : attendees d = 2 :=
by
  -- symbolic computation for length of the filtered list of available persons
  sorry

end max_attendees_eq_two_l593_593809


namespace angle_AOD_is_36_degrees_l593_593954

theorem angle_AOD_is_36_degrees
  (y : ℝ)
  (h1 : ∠ COA = 90)
  (h2 : ∠ BOD = 90)
  (h3 : ∠ BOC = 2 * y) :
  y = 36 := by
  sorry

end angle_AOD_is_36_degrees_l593_593954


namespace ultramindmaster_secret_codes_count_l593_593937

/-- 
In the game UltraMindmaster, we need to find the total number of possible secret codes 
formed by placing pegs of any of eight different colors into five slots.
Colors may be repeated, and each slot must be filled.
-/
theorem ultramindmaster_secret_codes_count :
  let colors := 8
  let slots := 5
  colors ^ slots = 32768 := by
    sorry

end ultramindmaster_secret_codes_count_l593_593937


namespace sin_315_eq_neg_sqrt_2_div_2_l593_593000

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l593_593000


namespace quadratic_root_value_of_m_l593_593581

theorem quadratic_root_value_of_m :
  ∀ m : ℚ, (∀ x : ℂ, x = (-5 + complex.I * real.sqrt 231) / 12 ∨ x = (-5 - complex.I * real.sqrt 231) / 12 →
  6 * x^2 + 5 * x + m = 0) → m = 32 / 3 :=
by
  sorry

end quadratic_root_value_of_m_l593_593581


namespace fraction_sum_is_0_333_l593_593670

theorem fraction_sum_is_0_333 : (3 / 10 : ℝ) + (3 / 100) + (3 / 1000) = 0.333 := 
by
  sorry

end fraction_sum_is_0_333_l593_593670


namespace perfect_squares_count_in_range_l593_593236

theorem perfect_squares_count_in_range :
  let s := {x : ℕ | 50 ≤ x ∧ x ≤ 250 ∧ ∃ (n : ℕ), n^2 = x}
  ∈ (∃ (count : ℕ), count = 8) :=
by
  let s := {x : ℕ | 50 ≤ x ∧ x ≤ 250 ∧ ∃ (n : ℕ), n^2 = x}
  sorry

end perfect_squares_count_in_range_l593_593236


namespace min_value_of_quadratic_l593_593882

theorem min_value_of_quadratic (x : ℝ) : 
  (∃ x₀ : ℝ, (∀ x : ℝ, 2 * x ^ 2 - 8 * x + 9 ≥ 2 * x₀ ^ 2 - 8 * x₀ + 9) ∧ (2 * x₀ ^ 2 - 8 * x₀ + 9 = 1)) :=
by
  -- We need to state the conditions and provide as part of the theorem
  use 2
  split
  {
    intro x
    calc
      2 * x ^ 2 - 8 * x + 9 ≥ 1 : sorry
  }
  {
    calc
      2 * (2 : ℝ) ^ 2 - 8 * 2 + 9 = 1 : sorry
  }

end min_value_of_quadratic_l593_593882


namespace distribute_toys_l593_593490

theorem distribute_toys :
  let S := {A, B, C, D}
  let children := {1, 2, 3}
  let partitions := {p : finset (finset (fin 4)) // partition S p ∧ ∀ s ∈ p, 0 < s.card ∧ s.card ≤ 2}
  ∃ (valid_partitions : finset ℕ),
  valid_partitions.card = 30 ∧
  ∀ p ∈ valid_partitions, ∀ s ∈ p, (A ∈ s → B ∉ s) :=
sorry

end distribute_toys_l593_593490


namespace largest_binomial_coefficient_term_l593_593539

theorem largest_binomial_coefficient_term :
  ∃ (x : ℝ), 
    (binom 7 5) * (2 * sqrt x)^5 = 672 * x^(5/2) :=
by
  sorry

end largest_binomial_coefficient_term_l593_593539


namespace devin_makes_team_l593_593489

noncomputable def baseProbability := 0.05
noncomputable def heightFactor (h : ℝ) (a : ℝ) := 0.08 - 0.01 * (a - 12)
noncomputable def ageFactor (a : ℝ) := 0.03 * (a - 12)
noncomputable def ppgModifier (p : ℝ) := if p > 0.5 then 0.05 else 0.00

def devinsProbability (h a p : ℝ) : ℝ :=
  baseProbability +
  heightFactor h a * (h - 66) +
  ageFactor a * (a - 12) +
  ppgModifier p

theorem devin_makes_team :
  devinsProbability 68 13 0.4 = 0.22 :=
by
  sorry

end devin_makes_team_l593_593489


namespace problem_I_problem_II_l593_593145

def setA : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≤ 0}

theorem problem_I (a : ℝ) : (setB a ⊆ setA) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem problem_II (a : ℝ) : (setA ∩ setB a = {1}) ↔ a ≤ 1 := by
  sorry

end problem_I_problem_II_l593_593145


namespace triangle_perimeter_impossible_l593_593591

theorem triangle_perimeter_impossible (x : ℝ) (P : ℝ) 
  (h1 : 13 < x) (h2 : x < 37) (h3 : P = 37 + x) : 
  P ≠ 48 :=
by 
  have h4 : 50 < P := by linarith,
  have h5 : P < 74 := by linarith,
  intro h_eq,
  rw [h_eq] at h4 h5,
  linarith,
  sorry

end triangle_perimeter_impossible_l593_593591


namespace power_log_simplification_l593_593724

theorem power_log_simplification (x : ℝ) (h : x > 0) : (16^(Real.log x / Real.log 2))^(1/4) = x :=
by sorry

end power_log_simplification_l593_593724


namespace probability_at_least_one_six_l593_593423

theorem probability_at_least_one_six (h: ℚ) : h = 91 / 216 :=
by 
  sorry

end probability_at_least_one_six_l593_593423


namespace closer_vertex_exists_l593_593730

variables {Point : Type} [metric_space Point]
variable {Poly : set Point}
variable [convex_polytope Poly]
variables {P Q : Point}
variables (P_in_Poly : P ∈ Poly) (Q_in_Poly : Q ∈ Poly)

theorem closer_vertex_exists :
  ∃ V ∈ Poly, dist V Q < dist V P :=
begin
  sorry -- Proof goes here.
end

end closer_vertex_exists_l593_593730


namespace inscribed_sphere_radius_l593_593368

theorem inscribed_sphere_radius 
  (a : ℝ) 
  (h_angle : ∀ (lateral_face : ℝ), lateral_face = 60) : 
  ∃ (r : ℝ), r = a * (Real.sqrt 3) / 6 :=
by
  sorry

end inscribed_sphere_radius_l593_593368


namespace find_a_range_l593_593210

def f (a x : ℝ) : ℝ := x^2 + a * x

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, f a (f a x) ≤ f a x) → (a ≤ 0 ∨ a ≥ 2) :=
by
  sorry

end find_a_range_l593_593210


namespace odd_nat_existence_l593_593902

theorem odd_nat_existence (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (n : ℕ) :
  ∃ m : ℕ, (a^m * b^2 - 1) % 2^n = 0 ∨ (b^m * a^2 - 1) % 2^n = 0 := 
by
  sorry

end odd_nat_existence_l593_593902


namespace range_of_m_l593_593211

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ set.Icc m (m+1), f x < 0) 
    (hf : ∀ x, f x = x^2 + m * x - 1) : 
    -real.sqrt(2) / 2 < m ∧ m < 0 :=
  sorry

end range_of_m_l593_593211


namespace range_of_a_l593_593361

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), |x + 3| - |x - 1| ≤ a ^ 2 - 3 * a) ↔ a ≤ -1 ∨ a ≥ 4 :=
by
  sorry

end range_of_a_l593_593361


namespace magnitude_2a_minus_b_l593_593843

variable (a b : EuclideanSpace ℝ (Fin 3))
variable (aa : |a| = 2)
variable (bb : |b| = 3)
variable (angle_eq : Real.angle (angle a b) = Real.pi / 3)

theorem magnitude_2a_minus_b : 
  |2 • a - b| = Real.sqrt 13 :=
by sorry

end magnitude_2a_minus_b_l593_593843


namespace series_sum_eq_1_over_200_l593_593083

-- Definition of the nth term of the series
def nth_term (n : ℕ) : ℝ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

-- Definition of the series sum
def series_sum : ℝ :=
  ∑' n, nth_term n

-- Final proposition to prove the sum of the series is 1/200
theorem series_sum_eq_1_over_200 : series_sum = 1 / 200 :=
  sorry

end series_sum_eq_1_over_200_l593_593083


namespace sin_315_eq_neg_sqrt_2_div_2_l593_593002

-- Define the point Q on the unit circle at 315 degrees counterclockwise from (1, 0)
def Q : ℝ × ℝ := (Real.cos (315 * Real.pi / 180), Real.sin (315 * Real.pi / 180))

-- Prove that the sine of 315 degrees is -sqrt(2)/2
theorem sin_315_eq_neg_sqrt_2_div_2 : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := sorry

end sin_315_eq_neg_sqrt_2_div_2_l593_593002


namespace pencils_distribution_l593_593373

variable {a b c d : ℕ}
variable (y x : ℕ)

theorem pencils_distribution :
  a + b + c + d = 53 ∧
  y - x < 5 ∧
  a + b = 2c ∧
  c + b = 2d → b = 15 :=
by sorry

end pencils_distribution_l593_593373


namespace polynomial_division_correct_l593_593511

def dividend : ℤ[X] := 4 * X^5 - 3 * X^4 + 2 * X^3 - 5 * X^2 + 9 * X - 4
def divisor  : ℤ[X] := X + 3
def quotient : ℤ[X] := 4 * X^4 - 15 * X^3 + 47 * X^2 - 146 * X + 447

theorem polynomial_division_correct :
  (dividend / divisor) = quotient :=
sorry

end polynomial_division_correct_l593_593511


namespace probability_first_prize_l593_593452

-- Define the total number of tickets
def total_tickets : ℕ := 150

-- Define the number of first prizes
def first_prizes : ℕ := 5

-- Define the probability calculation as a theorem
theorem probability_first_prize : (first_prizes : ℚ) / total_tickets = 1 / 30 := 
by sorry  -- Placeholder for the proof

end probability_first_prize_l593_593452


namespace squares_equal_l593_593678

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l593_593678


namespace acute_angle_slope_neg_product_l593_593669

   theorem acute_angle_slope_neg_product (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) (acute_inclination : ∃ (k : ℝ), k > 0 ∧ y = -a/b): (a * b < 0) :=
   by
     sorry
   
end acute_angle_slope_neg_product_l593_593669


namespace largest_possible_integer_l593_593759

noncomputable def largest_integer_in_list : ℕ :=
let a := 9 in
let b := 9 in
let c := 10 in
let sum := 55 in
let (d, e) := (11, sum - (a + b + c + 11)) in
e

theorem largest_possible_integer : largest_integer_in_list = 16 :=
by
let a := 9
let b := 9
let c := 10
let sum := 55
let d := 11
let e := sum - (a + b + c + d)
have : a = 9 := by simp
have : b = 9 := by simp
have : c = 10 := by simp
have : sum = 55 := by simp
have : d = 11 := by simp
have : e = 16 := by simp [sum, a, b, c, d]
exact this

end largest_possible_integer_l593_593759


namespace sin_315_eq_neg_sqrt2_div_2_l593_593060

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593060


namespace find_BM_l593_593608

theorem find_BM (a c BM : ℝ) (h : a > 0) (h2 : c > 0)
  (AM_eq : AM = 60) (CM_eq : CM = 70)
  (area_ABM : ∀ (S: ℝ), 0 < S ∧ S = (1 / 2) * a * c → (1 / 3) * S = (1 / 2) * BM * AM)
  (area_BCM : ∀ (S: ℝ), 0 < S ∧ S = (1 / 2) * a * c → (1 / 4) * S = (1 / 2) * BM * CM) :
    BM ≈ 38 :=
by
  sorry

end find_BM_l593_593608


namespace max_distance_from_origin_l593_593391

def particle_position_x (t : Real) : Real := 3 * Real.cos t
def particle_position_y (t : Real) : Real := 4 * Real.sin t

def distance_from_origin (t : Real) : Real :=
  Real.sqrt ((particle_position_x t) ^ 2 + (particle_position_y t) ^ 2)

theorem max_distance_from_origin :
  ∃ t : Real, distance_from_origin t = 4 :=
sorry

end max_distance_from_origin_l593_593391


namespace total_cases_l593_593808

-- Define the number of boys' high schools and girls' high schools
def boys_high_schools : Nat := 4
def girls_high_schools : Nat := 3

-- Theorem to be proven
theorem total_cases (B G : Nat) (hB : B = boys_high_schools) (hG : G = girls_high_schools) : 
  B + G = 7 :=
by
  rw [hB, hG]
  exact rfl

end total_cases_l593_593808


namespace number_of_roof_tiles_l593_593753

def land_cost : ℝ := 50
def bricks_cost_per_1000 : ℝ := 100
def roof_tile_cost : ℝ := 10
def land_required : ℝ := 2000
def bricks_required : ℝ := 10000
def total_construction_cost : ℝ := 106000

theorem number_of_roof_tiles :
  let land_total := land_cost * land_required
  let bricks_total := (bricks_required / 1000) * bricks_cost_per_1000
  let remaining_cost := total_construction_cost - (land_total + bricks_total)
  let roof_tiles := remaining_cost / roof_tile_cost
  roof_tiles = 500 := by
  sorry

end number_of_roof_tiles_l593_593753


namespace find_original_integer_l593_593525

theorem find_original_integer (a b c d : ℕ) 
    (h1 : (b + c + d) / 3 + 10 = 37) 
    (h2 : (a + c + d) / 3 + 10 = 31) 
    (h3 : (a + b + d) / 3 + 10 = 25) 
    (h4 : (a + b + c) / 3 + 10 = 19) : 
    d = 45 := 
    sorry

end find_original_integer_l593_593525


namespace sin_315_degree_l593_593048

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593048


namespace train_crossing_time_l593_593906

def km_per_hr_to_m_per_s (v : ℝ) : ℝ := v * (1000 / 3600)

def total_distance (length_train length_bridge : ℝ) : ℝ := length_train + length_bridge

def time_to_cross (distance speed : ℝ) : ℝ := distance / speed

theorem train_crossing_time :
  let length_train := 450
  let speed_train_km_hr := 108
  let length_bridge := 250
  let speed_train_m_s := km_per_hr_to_m_per_s speed_train_km_hr
  let distance := total_distance length_train length_bridge
  time_to_cross distance speed_train_m_s = 23.33 :=
by
  sorry

end train_crossing_time_l593_593906


namespace distance_to_post_office_l593_593734

variable (D : ℚ)
variable (rate_to_post : ℚ := 25)
variable (rate_back : ℚ := 4)
variable (total_time : ℚ := 5 + 48 / 60)

theorem distance_to_post_office : (D / rate_to_post + D / rate_back = total_time) → D = 20 := by
  sorry

end distance_to_post_office_l593_593734


namespace sum_of_all_possible_abs_values_l593_593986

open Real

theorem sum_of_all_possible_abs_values (p q r s : ℝ) 
  (h₁ : |p - q| = 3) 
  (h₂ : |q - r| = 4) 
  (h₃ : |r - s| = 5) : 
  ∑ x in { abs (p - s) | (q = p + 3 ∨ q = p - 3) ∧ 
                         (r = q + 4 ∨ r = q - 4) ∧ 
                         (s = r + 5 ∨ s = r - 5) }, x = 24 :=
by sorry

end sum_of_all_possible_abs_values_l593_593986


namespace neg_p_necessary_not_sufficient_neg_q_l593_593626

def p (x : ℝ) := abs x < 1
def q (x : ℝ) := x^2 + x - 6 < 0

theorem neg_p_necessary_not_sufficient_neg_q :
  (¬ (∃ x, p x)) → (¬ (∃ x, q x)) ∧ ¬ ((¬ (∃ x, p x)) → (¬ (∃ x, q x))) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l593_593626


namespace sqrt_product_simplification_l593_593464

theorem sqrt_product_simplification (q : ℝ) : 
  sqrt (42 * q) * sqrt (7 * q) * sqrt (3 * q) = 126 * q * sqrt q := 
by
  sorry

end sqrt_product_simplification_l593_593464


namespace smallest_element_in_S_l593_593673

-- Define the function f satisfying the given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if 2 ≤ x ∧ x ≤ 4 then 1 - |x - 3| else sorry

-- State the properties of f: f(2x) = 2f(x)
axiom f_property (x : ℝ) (hx : x > 0) : f (2 * x) = 2 * f x

-- State that f satisfies the given condition in the interval [2, 4]
axiom f_interval (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 4) : f x = 1 - |x - 3|

-- Define the set S we are interested in
def S : set ℝ := { x | f x = f 36 }

-- State the problem: Find the smallest element in the set S
theorem smallest_element_in_S : ∃ x ∈ S, x = 12 ∧ ∀ y ∈ S, 12 ≤ y :=
by
  sorry

end smallest_element_in_S_l593_593673


namespace square_area_l593_593470

def side_length (perimeter : ℝ) := perimeter / 4
def area (side_length : ℝ) := side_length ^ 2

theorem square_area (h : side_length 40 = 10) : area 10 = 100 :=
by
  have h1 : side_length 40 = 10 := h
  rw [area, h1]
  norm_num
  sorry

end square_area_l593_593470


namespace inscribed_circle_radius_isosceles_triangle_l593_593595

noncomputable def isosceles_triangle_base : ℝ := 30 -- base AC
noncomputable def isosceles_triangle_equal_side : ℝ := 39 -- equal sides AB and BC

theorem inscribed_circle_radius_isosceles_triangle :
  ∀ (AC AB BC: ℝ), 
  AC = isosceles_triangle_base → 
  AB = isosceles_triangle_equal_side →
  BC = isosceles_triangle_equal_side →
  ∃ r : ℝ, r = 10 := 
by
  intros AC AB BC hAC hAB hBC
  sorry

end inscribed_circle_radius_isosceles_triangle_l593_593595


namespace sin_315_eq_neg_sqrt2_div_2_l593_593073

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593073


namespace ones_digit_of_34_exp_11_exp_34_l593_593510

theorem ones_digit_of_34_exp_11_exp_34 :
  (34 ^ (11 ^ 34)) % 10 = 4 := 
by
  -- Conditions
  have h1 : ∀ n, (34 ^ n) % 10 = (4 ^ n) % 10 := by sorry
  have h2 : ∀ n, n % 2 = 1 → (4 ^ n) % 10 = 4 := by sorry
  have h3 : (11 ^ 34) % 2 = 1 := by sorry
  
  -- Conclusion
  rw [h1 (11 ^ 34), h3],
  exact h2 (11 ^ 34) h3

end ones_digit_of_34_exp_11_exp_34_l593_593510


namespace triangle_ABC_l593_593933

theorem triangle_ABC (a b c : ℝ) (A B C : ℝ)
  (h1 : a + b = 5)
  (h2 : c = Real.sqrt 7)
  (h3 : 4 * (Real.sin ((A + B) / 2))^2 - Real.cos (2 * C) = 7 / 2) :
  (C = Real.pi / 3)
  ∧ (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by
  sorry

end triangle_ABC_l593_593933


namespace cyclists_meet_at_start_l593_593710

theorem cyclists_meet_at_start (T : ℚ) (h1 : T = 5 * 7 * 9 / gcd (5 * 7) (gcd (7 * 9) (9 * 5))) : T = 157.5 :=
by
  sorry

end cyclists_meet_at_start_l593_593710


namespace permutations_count_l593_593618

theorem permutations_count :
  ∃ P : Perm, 
    (P = λ i, if i ≤ 6 then 15 - i else if i = 7 then 1 else i) ∧
    (∀ i, (1 ≤ i ∧ i ≤ 7 → P i < P (i + 1))) ∧
    (∀ i, (8 ≤ i ∧ i ≤ 13 → P i < P (i + 1))) ∧
    ∃ permutations, perms P 1716 := 
sorry

end permutations_count_l593_593618


namespace monotonic_increasing_interval_l593_593671

noncomputable def f (x : Real) (φ : Real) : Real := cos (2 * x + φ)

theorem monotonic_increasing_interval 
  (φ : Real) (hφ : abs φ < π / 2) :
  ∃ I : Set Real, I = Icc (-π / 3) (π / 12) ∧ ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x φ ≤ f y φ :=
sorry

end monotonic_increasing_interval_l593_593671


namespace range_of_a_l593_593851

-- Definitions of propositions p and q

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem stating the range of values for a given p ∧ q is true

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≤ -2 ∨ a = 1) :=
by sorry

end range_of_a_l593_593851


namespace least_k_clique_l593_593953

/-- 
  A set of points with integer coordinates \(S\) on the coordinate plane.
  A definition of k-friends and k-cliques are given.
  Prove that the least positive integer k for which there exists a k-clique with more than 200 elements is k = 180180.
--/
theorem least_k_clique (S : Set (ℤ × ℤ)) (k_friends : (ℤ × ℤ) → (ℤ × ℤ) → ℝ → Prop) 
    (k_clique : Set (ℤ × ℤ) → ℝ → Prop) :
  (∃ T ⊆ S, k_clique T 180180 ∧ T.card > 200) →
  ∀ k, (∃ T ⊆ S, k_clique T k ∧ T.card > 200) → k ≥ 180180 :=
sorry

end least_k_clique_l593_593953


namespace find_m_l593_593179

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l593_593179


namespace ratio_area_sector_COE_to_circle_l593_593637

theorem ratio_area_sector_COE_to_circle
  (C D E : Point)
  (A B O : Point)
  (h_same_side : SameSide C D E A B)
  (angle_AOC : Angle A O C = 40)
  (angle_DOE : Angle D O E = 15)
  (angle_AOB : Angle A O B = 180) :
  (sector (C O E)).area / circle(O).area = 31 / 72 :=
begin 
  sorry
end

end ratio_area_sector_COE_to_circle_l593_593637


namespace g_2023_eq_0_l593_593247

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_defined (x : ℕ) : ∃ y : ℝ, g x = y

axiom g_initial : g 1 = 1

axiom g_functional (a b : ℕ) : g (a + b) = g a + g b - 2 * g (a * b + 1)

theorem g_2023_eq_0 : g 2023 = 0 :=
sorry

end g_2023_eq_0_l593_593247


namespace two_f_two_plus_three_f_neg_two_eq_neg_107_l593_593090

noncomputable def f (x : ℤ) : ℤ := 3 * x^3 - 2 * x^2 + 4 * x - 7

theorem two_f_two_plus_three_f_neg_two_eq_neg_107 :
  2 * (f 2) + 3 * (f (-2)) = -107 :=
by sorry

end two_f_two_plus_three_f_neg_two_eq_neg_107_l593_593090


namespace find_x_l593_593558

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x+1, -x)

def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x (x : ℝ) (h : perpendicular vector_a (vector_b x)) : x = 1 :=
by sorry

end find_x_l593_593558


namespace angle_C_eq_side_c_value_l593_593903

variables (A B C : ℝ) (a b c : ℝ)
variables (sin_A cos_A sin_B cos_B sin_C : ℝ)
variables (CA AB AC CB : ℝ)

-- Definitions based on the given problem
def vector_m := (sin_A, cos_A)
def vector_n := (cos_B, sin_B)

axiom inner_product_eq : sin_A * cos_B + sin_B * cos_A = sin (2 * C)

axiom geo_seq_sin : sin_A * sin_B = sin_C ^ 2

axiom law_of_sines : c^2 = a * b

axiom inner_product_CA_AB_AC : CA * (AB - AC) = 18

-- Proof Problem 1: Show that C = π/3
theorem angle_C_eq : cos C = 1 / 2 → C = π / 3 :=
-- proof goes here
sorry

-- Proof Problem 2: Show that c = 6
theorem side_c_value : 
  sin_A * sin_B = sin_C ^ 2 → 
  c^2 = a * b → 
  CA * (AB - AC) = 18 → 
  a * b = 36 → 
  c = 6 :=
-- proof goes here
sorry

end angle_C_eq_side_c_value_l593_593903


namespace calculate_24_points_l593_593457

theorem calculate_24_points :
  ∃ f : ℚ → ℚ → ℚ → ℚ → ℚ,
  (∀ x y z w, (x, y, z, w) ∈ 
    {(-6, -0.5, 2, 3), (-6, -0.5, 3, 2), (-6, 2, -0.5, 3), (-6, 2, 3, -0.5), 
     (-6, 3, -0.5, 2), (-6, 3, 2, -0.5), (-0.5, -6, 2, 3), (-0.5, -6, 3, 2),
     (-0.5, 2, -6, 3), (-0.5, 2, 3, -6), (-0.5, 3, -6, 2), (-0.5, 3, 2, -6),
     (2, -6, -0.5, 3), (2, -6, 3, -0.5), (2, -0.5, -6, 3), (2, -0.5, 3, -6),
     (2, 3, -6, -0.5), (2, 3, -0.5, -6),(3, -6, -0.5, 2), (3, -6, 2, -0.5),
     (3, -0.5, -6, 2), (3, -0.5, 2, -6), (3, 2, -6, -0.5), (3, 2, -0.5, -6)})
    → f x y z w = 24 :=
begin
  sorry
end

end calculate_24_points_l593_593457


namespace terminal_zeros_l593_593562

theorem terminal_zeros (a b c : ℕ) (h1 : a = 45) (h2 : b = 320) (h3 : c = 60) : 
  (count_terminal_zeros (a * b * c) = 3) := sorry

-- Auxiliary definition to count the number of terminal zeros in a number
noncomputable def count_terminal_zeros (n : ℕ) : ℕ := sorry

end terminal_zeros_l593_593562


namespace sin_315_degree_l593_593049

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593049


namespace simplify_radical_expression_l593_593458

variable (q : ℝ)
variable (hq : q > 0)

theorem simplify_radical_expression :
  (sqrt(42 * q) * sqrt(7 * q) * sqrt(3 * q) = 21 * q * sqrt(2 * q)) :=
by
  sorry

end simplify_radical_expression_l593_593458


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593027

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593027


namespace inscribed_square_properties_l593_593714

theorem inscribed_square_properties (r : ℝ) (s : ℝ) (d : ℝ) (A_circle : ℝ) (A_square : ℝ) (total_diagonals : ℝ) (hA_circle : A_circle = 324 * Real.pi) (hr : r = Real.sqrt 324) (hd : d = 2 * r) (hs : s = d / Real.sqrt 2) (hA_square : A_square = s ^ 2) (htotal_diagonals : total_diagonals = 2 * d) :
  A_square = 648 ∧ total_diagonals = 72 :=
by sorry

end inscribed_square_properties_l593_593714


namespace find_m_l593_593188

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l593_593188


namespace MN_parallel_BB1_l593_593603

-- Definitions
variables {Point : Type} [AffineSpace Point]

structure RectangularPrism :=
(base_is_square : ∀ (A B C D: Point), distance A B = distance A D ∧ distance B C = distance D C)
(AA1_perp_base : ∀ (A A1 : Point) (B C D : Point), A1 ≠ A ∧ (distance^2 A B + distance^2 B C = distance^2 A A1))
(MN_in_plane_ACCA1 : ∀ (M N : Point) (A C A1 : Point), M ∉ A ∧ M ∉ C ∧ N ∉ A ∧ N ∉ A1 ∧ M in_plane A C A1 ∧ N in_plane A C A1)
(MN_perp_AC : ∀ (M N A C : Point), MN ⊥ AC)

-- Theorem statement
theorem MN_parallel_BB1 (ABCDA1B1C1D1 : RectangularPrism) (M N B B1 : Point) :
  MN_in_plane_ACCA1 ∧ MN_perp_AC → MN ∥ BB1 :=
sorry

end MN_parallel_BB1_l593_593603


namespace abs_x_minus_y_l593_593313

theorem abs_x_minus_y (x y : ℂ) (h1 : x + y = Complex.sqrt 20) (h2 : x^2 + y^2 = 15) : 
  Complex.abs (x - y) = Complex.sqrt 10 :=
sorry

end abs_x_minus_y_l593_593313


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593018

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593018


namespace perfect_squares_between_50_and_250_l593_593234

theorem perfect_squares_between_50_and_250 : 
  (card { n : ℕ | 50 ≤ n^2 ∧ n^2 ≤ 250 }) = 8 := 
by
  sorry

end perfect_squares_between_50_and_250_l593_593234


namespace annual_income_l593_593507

-- Define given conditions
def investmentAmount : ℝ := 6800
def stockDividendRate : ℝ := 0.2
def stockPrice : ℝ := 136
def annualTaxRate : ℝ := 0.15

-- Define par value as typically assumed $100 unless otherwise stated
def parValue : ℝ := 100

-- Annual income calculation (main theorem statement)
theorem annual_income (invAmt : ℝ) (divRate : ℝ) (stkPrice : ℝ) (taxRate : ℝ) (parVal : ℝ) :
  invAmt = investmentAmount →
  divRate = stockDividendRate →
  stkPrice = stockPrice →
  taxRate = annualTaxRate →
  parVal = parValue →
  let annual_dividend := divRate * parVal in
  let number_of_shares := invAmt / stkPrice in
  let total_dividend_income := number_of_shares * annual_dividend in
  let tax_on_income := taxRate * total_dividend_income in
  let after_tax_income := total_dividend_income - tax_on_income in
  after_tax_income = 850 := 
sorry

end annual_income_l593_593507


namespace solve_the_problem_l593_593949

noncomputable def solve_problem : Prop :=
  ∀ (θ t α : ℝ),
    (∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = 4 * Real.sin θ) → 
    (∃ x y : ℝ, x = 1 + t * Real.cos α ∧ y = 2 + t * Real.sin α) →
    (∃ m n : ℝ, m = 1 ∧ n = 2) →
    (-2 = Real.tan α)

theorem solve_the_problem : solve_problem := by
  sorry

end solve_the_problem_l593_593949


namespace scalene_triangles_count_l593_593098

theorem scalene_triangles_count :
  (∃ n : ℕ, 
    n = { (a, b, c) : ℕ × ℕ × ℕ |
          a < b ∧ b < c ∧ 
          a + b + c < 16 ∧ 
          a + b > c ∧ a + c > b ∧ b + c > a }.to_finset.card) ∧ n = 6 :=
begin
  sorry
end

end scalene_triangles_count_l593_593098


namespace distance_M_to_NF_l593_593758

noncomputable def parabola : set (ℝ × ℝ) := { p | p.2 ^ 2 = 4 * p.1 }

def focus : ℝ × ℝ := (1, 0)

def slope : ℝ := real.sqrt 3

def directrix (x : ℝ) := x = -1

def point_on_directrix (N : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  directrix N.1 ∧ N.2 = M.2

def perpendicular (M N : ℝ × ℝ) : Prop :=
  (N.2 - M.2) * 1 - (N.1 - M.1) * 0 = 0

def line_through (F M : ℝ × ℝ) (slope : ℝ) : set (ℝ × ℝ) :=
  { q | q.2 = slope * (q.1 - F.1) }

def line_eq (N F : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ P, real.sqrt 3 * P.1 + P.2 - real.sqrt 3 = 0

def distance_from_point_to_line (M : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) : ℝ :=
  (real.abs (real.sqrt 3 * M.1 + M.2 - real.sqrt 3)) / real.sqrt (3 + 1)

theorem distance_M_to_NF (M N F : ℝ × ℝ)
  (hM : parabola M ∧ M.1 > 0 ∧ line_through F M slope M)
  (hN : point_on_directrix N M)
  (h_perp : perpendicular M N) :
  distance_from_point_to_line M (line_eq N F) = 2 * real.sqrt 3 := 
sorry

end distance_M_to_NF_l593_593758


namespace g_49_l593_593672

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eqn (x y : ℝ) : g (x^2 * y) = x * g y
axiom g_one_val : g 1 = 6

theorem g_49 : g 49 = 42 := by
  sorry

end g_49_l593_593672


namespace picture_area_l593_593563

theorem picture_area (x y : ℤ) (hx : 1 < x) (hy : 1 < y) (h : (x + 2) * (y + 4) = 45) : x * y = 15 := by
  sorry

end picture_area_l593_593563


namespace find_m_l593_593181

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l593_593181


namespace percentage_difference_percentage_by_which_x_is_less_than_y_l593_593259

theorem percentage_difference (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : y = 1.25 * x) : (x / y) = 0.8 :=
by sorry

# Check the resulting percentage
theorem percentage_by_which_x_is_less_than_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : y = 1.25 * x) : 100 * (1 - (x / y)) = 20 :=
by sorry

end percentage_difference_percentage_by_which_x_is_less_than_y_l593_593259


namespace general_term_formula_smallest_n_l593_593897

-- Part I: General term formula for the sequence
theorem general_term_formula
  (a : ℕ → ℕ)
  (h1 : ∀ n, a (n+1) = 2 * a n)
  (h2 : ∃ a1 a2 a3, a 1 = a1 ∧ a 2 = a2 ∧ a 3 = a3 ∧
    a1 = 2 * a1 ∧ a2 + 1 = a3) :
  ∀ n, a n = 2 ^ n := sorry

-- Part II: Smallest integer n such that b_1 + b_2 + ... + b_n > 45
theorem smallest_n
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (h1 : ∀ n, a (n+1) = 2 * a n)
  (h2 : ∃ a1 a2 a3, a 1 = a1 ∧ a 2 = a2 ∧ a 3 = a3 ∧
    a1 = 2 * a1 ∧ a2 + 1 = a3)
  (h3 : ∀ n, b n = Nat.log2 (a n))
  (h4 : ∀ n, b n = n) :
  ∃ (n : ℕ), b_1 + b_2 + ... + b_n > 45 ∧ ∀ k < n, b_1 + b_2 + ... + b_k ≤ 45 := sorry

end general_term_formula_smallest_n_l593_593897


namespace exact_days_two_friends_visit_l593_593476

-- Define the periodicities of Alice, Beatrix, and Claire
def periodicity_alice : ℕ := 1
def periodicity_beatrix : ℕ := 5
def periodicity_claire : ℕ := 7

-- Define the total days to be considered
def total_days : ℕ := 180

-- Define the number of days three friends visit together
def lcm_ab := Nat.lcm periodicity_alice periodicity_beatrix
def lcm_ac := Nat.lcm periodicity_alice periodicity_claire
def lcm_bc := Nat.lcm periodicity_beatrix periodicity_claire
def lcm_abc := Nat.lcm lcm_ab periodicity_claire

-- Define the counts of visitations
def count_ab := total_days / lcm_ab - total_days / lcm_abc
def count_ac := total_days / lcm_ac - total_days / lcm_abc
def count_bc := total_days / lcm_bc - total_days / lcm_abc

-- Finally calculate the number of days exactly two friends visit together
def days_two_friends_visit : ℕ := count_ab + count_ac + count_bc

-- The theorem to prove
theorem exact_days_two_friends_visit : days_two_friends_visit = 51 :=
by 
  -- This is where the actual proof would go
  sorry

end exact_days_two_friends_visit_l593_593476


namespace projection_of_v_on_w_l593_593515

def proj_w_v {α : Type*} [LinearOrderedField α] (v w : Matrix (Fin 2) (Fin 1) α) : Matrix (Fin 2) (Fin 1) α :=
  let vw_dot := v 0 0 * w 0 0 + v 1 0 * w 1 0
  let ww_dot := w 0 0 * w 0 0 + w 1 0 * w 1 0
  Matrix.of_fin (λ i j, (vw_dot / ww_dot) * w i j)

theorem projection_of_v_on_w :
  proj_w_v (Matrix.of_fin (λ i j, if i = 0 then 4 else -3 : ℚ)) (Matrix.of_fin (λ i j, if i = 0 then 7 else 6 : ℚ)) =
  Matrix.of_fin (λ i j, if i = 0 then (14 : ℚ) / 17 else (12 : ℚ) / 17) := by
sorry

end projection_of_v_on_w_l593_593515


namespace complex_solutions_z4_eq_neg4_l593_593114

theorem complex_solutions_z4_eq_neg4 (z : ℂ) : z^4 = -4 ↔ (z = 1 + 1 * complex.I ∨ z = 1 - 1 * complex.I ∨ z = -1 + 1 * complex.I ∨ z = -1 - 1 * complex.I) := by
  sorry

end complex_solutions_z4_eq_neg4_l593_593114


namespace female_democrats_count_l593_593707

-- Define the parameters and conditions
variables (F M D_f D_m D_total : ℕ)
variables (h1 : F + M = 840)
variables (h2 : D_total = 1/3 * (F + M))
variables (h3 : D_f = 1/2 * F)
variables (h4 : D_m = 1/4 * M)
variables (h5 : D_total = D_f + D_m)

-- State the theorem
theorem female_democrats_count : D_f = 140 :=
by
  sorry

end female_democrats_count_l593_593707


namespace cyclists_meet_time_l593_593408

/-- 
  Two cyclists start on a circular track from a given point but in opposite directions with speeds of 7 m/s and 8 m/s.
  The circumference of the circle is 180 meters.
  After what time will they meet at the starting point? 
-/
theorem cyclists_meet_time :
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  (circumference / (speed1 + speed2) = 12) :=
by
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  sorry

end cyclists_meet_time_l593_593408


namespace expand_polynomial_l593_593814

noncomputable def polynomial_expression (x : ℝ) : ℝ := -2 * (x - 3) * (x + 4) * (2 * x - 1)

theorem expand_polynomial (x : ℝ) :
  polynomial_expression x = -4 * x^3 - 2 * x^2 + 50 * x - 24 :=
sorry

end expand_polynomial_l593_593814


namespace polynomial_b_value_l593_593833

theorem polynomial_b_value :
  ∃ (z w : ℂ), (z * w = 7 + 2 * Complex.I) ∧ 
               ((Complex.conj z) + (Complex.conj w) = 2 + 3 * Complex.I) ∧ 
               (∀ (a c d : ℝ), ∀ (P : ℂ → ℂ), 
                 (P = λ x, x^4 + a * x^3 + 27 * x^2 + c * x + d) → 
                 P z = 0 ∧ P (Complex.conj z) = 0 ∧ P w = 0 ∧ P (Complex.conj w) = 0) :=
begin
  sorry
end

end polynomial_b_value_l593_593833


namespace overall_percentage_loss_l593_593774

noncomputable def original_price : ℝ := 100
noncomputable def increased_price : ℝ := original_price * 1.36
noncomputable def first_discount_price : ℝ := increased_price * 0.90
noncomputable def second_discount_price : ℝ := first_discount_price * 0.85
noncomputable def third_discount_price : ℝ := second_discount_price * 0.80
noncomputable def final_price_with_tax : ℝ := third_discount_price * 1.05
noncomputable def percentage_change : ℝ := ((final_price_with_tax - original_price) / original_price) * 100

theorem overall_percentage_loss : percentage_change = -12.6064 :=
by
  sorry

end overall_percentage_loss_l593_593774


namespace solution_set_of_inequality_g_geq_2_l593_593551

-- Definition of the function f
def f (x a : ℝ) := |x - a|

-- Definition of the function g
def g (x a : ℝ) := f x a + f (x + 2) a

-- Proof Problem I
theorem solution_set_of_inequality (a : ℝ) (x : ℝ) :
  a = -1 → (f x a ≥ 4 - |2 * x - 1|) ↔ (x ≤ -4/3 ∨ x ≥ 4/3) :=
by sorry

-- Proof Problem II
theorem g_geq_2 (a : ℝ) (x : ℝ) :
  (∀ x, f x a ≤ 1 → (0 ≤ x ∧ x ≤ 2)) → a = 1 → g x a ≥ 2 :=
by sorry

end solution_set_of_inequality_g_geq_2_l593_593551


namespace hyperbola_tangent_circle_asymptote_l593_593251

theorem hyperbola_tangent_circle_asymptote (m : ℝ) (hm : m > 0) :
  (∀ x y : ℝ, y^2 - (x^2 / m^2) = 1 → 
   ∃ (circle_center : ℝ × ℝ) (circle_radius : ℝ), 
   circle_center = (0, 2) ∧ circle_radius = 1 ∧ 
   ((∃ k : ℝ, k = m ∨ k = -m) → 
    ((k * 0 - 2 + 0) / real.sqrt ((k)^2 + (-1)^2) = 1))) →
  m = real.sqrt 3 / 3 :=
by sorry

end hyperbola_tangent_circle_asymptote_l593_593251


namespace count_lattice_right_triangles_with_incenter_l593_593325

def is_lattice_point (p : ℤ × ℤ) : Prop := ∃ x y : ℤ, p = (x, y)

def is_right_triangle (O A B : ℤ × ℤ) : Prop :=
  O = (0, 0) ∧ (O.1 = A.1 ∨ O.2 = A.2) ∧ (O.1 = B.1 ∨ O.2 = B.2) ∧
  (A.1 * B.2 - A.2 * B.1 ≠ 0) -- Ensure A and B are not collinear with O

def incenter (O A B : ℤ × ℤ) : ℤ × ℤ :=
  ((A.1 + B.1 - O.1) / 2, (A.2 + B.2 - O.2) / 2)

theorem count_lattice_right_triangles_with_incenter :
  let I := (2015, 7 * 2015)
  ∃ (O A B : ℤ × ℤ), is_right_triangle O A B ∧ incenter O A B = I :=
sorry

end count_lattice_right_triangles_with_incenter_l593_593325


namespace percentage_died_by_bombardment_l593_593940

def initial_population : ℕ := 4675
def remaining_population : ℕ := 3553
def left_percentage : ℕ := 20

theorem percentage_died_by_bombardment (x : ℕ) (h : initial_population * (100 - x) / 100 * 8 / 10 = remaining_population) : 
  x = 5 :=
by
  sorry

end percentage_died_by_bombardment_l593_593940


namespace hyperbola_focus_lambda_l593_593147

theorem hyperbola_focus_lambda :
  ∀ (P : ℝ × ℝ) (λ : ℝ),
    (∀ x y : ℝ, P = (√(1 + x^2 / 4), y) ∧ (P.1, P.2 ≠ 0) ∧
      ((sqrt(1 + x^2 / 4) + sqrt(5), y) • (sqrt(1 + x^2 / 4) - sqrt(5), y) = (0, 0)) ∧
      ((P.1 + sqrt(5), P.2) • ((sqrt(5) - P.1), P.2) = (sqrt(5) * sqrt(5) - (1 / 4 - 5 + P.2 * P.2) = 0)) 
      ∧ (abs (dist P (-sqrt(5), 0)) = abs (λ * dist P (sqrt(5), 0)))) →
    λ = 2 :=
sorry

end hyperbola_focus_lambda_l593_593147


namespace simplify_radical_expression_l593_593461

variable (q : ℝ)
variable (hq : q > 0)

theorem simplify_radical_expression :
  (sqrt(42 * q) * sqrt(7 * q) * sqrt(3 * q) = 21 * q * sqrt(2 * q)) :=
by
  sorry

end simplify_radical_expression_l593_593461


namespace solve_problem_l593_593575

noncomputable def problem_statement (x y : ℝ) : Prop :=
  y + 9 = (x - 3) ^ 3 ∧ x + 9 = (y - 3) ^ 3 ∧ x ≠ y

theorem solve_problem (x y : ℝ) (h : problem_statement x y) : x^2 + y^2 = 18 :=
begin
  -- The proof will go here
  sorry
end

end solve_problem_l593_593575


namespace part_a_part_b_l593_593846

variable {S : Point}
variable {a b c : Ray}
variable {α β γ : Ray}
variable {α' β' γ' : Ray}
variable {faceA faceB faceC : Plane}

-- Given conditions
variable (trihedral_angle : TrihedralAngle S a b c faceA faceB faceC)
variable (ray_conditions : RaysInFaces α β γ faceA faceB faceC)
variable (symmetry_conditions : RaysSymmetricBisectors α' β' γ' α β γ)

-- Questions
theorem part_a :
  (rays_in_plane α β γ ↔ rays_in_plane α' β' γ') := sorry

theorem part_b :
  (¬planes_intersect_line (plane_through a α) (plane_through b β) (plane_through c γ) ↔
    planes_intersect_line (plane_through a α') (plane_through b β') (plane_through c γ')) := sorry

end part_a_part_b_l593_593846


namespace no_prime_in_sum_100_consecutive_l593_593101

noncomputable def sum_100_consecutive (n : ℕ) : ℕ :=
  100 * n + 4950

theorem no_prime_in_sum_100_consecutive :
  ∀ n : ℕ, ¬ Prime (sum_100_consecutive n) :=
by
  intro n
  have h : sum_100_consecutive n = 50 * (2 * n + 99) := by
    rw [sum_100_consecutive, mul_add, mul_comm 100 n, mul_assoc, add_comm]
  rw h
  exact Prime.not_prime_mul (prime_two, 50) sorry

end no_prime_in_sum_100_consecutive_l593_593101


namespace sin_315_eq_neg_sqrt2_div_2_l593_593057

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593057


namespace sin_315_degree_l593_593046

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593046


namespace students_guess_zero_l593_593434

-- Define the problem conditions
variables (n : ℕ) (x : ℕ → ℝ)

-- All students write a response between 0 and 100, inclusive.
def valid_response (i : ℕ) : Prop := 0 ≤ x i ∧ x i ≤ 100

-- Each student's guess is two-thirds of the average of all responses.
def guess_rule (avg : ℝ) (i : ℕ) : Prop := x i = (2 / 3) * avg

-- The average of all responses
def average (n : ℕ) (x : ℕ → ℝ) : ℝ := (∑ i in finset.range n, x i) / n

-- Prove that the only stable solution for n students guessing two-thirds of the average is 0
theorem students_guess_zero (n : ℕ) (condition_response : ∀ (i : ℕ), i < n → valid_response n x)
                           (condition_guess : ∀ (i : ℕ), i < n → guess_rule (average n x) i) : 
  ∀ (i : ℕ), i < n → x i = 0 :=
begin
  sorry
end

end students_guess_zero_l593_593434


namespace distinct_roots_difference_l593_593987

theorem distinct_roots_difference (r s : ℝ) (h₀ : r ≠ s) (h₁ : r > s) (h₂ : ∀ x, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ x = r ∨ x = s) :
  r - s = Real.sqrt 29 :=
by
  sorry

end distinct_roots_difference_l593_593987


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593024

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593024


namespace product_of_slope_and_intercept_l593_593355

theorem product_of_slope_and_intercept {x1 y1 x2 y2 : ℝ} (h1 : x1 = -4) (h2 : y1 = -2) (h3 : x2 = 1) (h4 : y2 = 3) :
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m * b = 2 :=
by
  sorry

end product_of_slope_and_intercept_l593_593355


namespace exp_7pi_over_2_eq_i_l593_593093

theorem exp_7pi_over_2_eq_i : Complex.exp (7 * Real.pi * Complex.I / 2) = Complex.I :=
by
  sorry

end exp_7pi_over_2_eq_i_l593_593093


namespace j_eq_h_shifted_l593_593474

def h : ℝ → ℝ
| x => if -5 ≤ x ∧ x ≤ 0 then -3 - x
       else if 0 ≤ x ∧ x ≤ 3 then Real.sqrt (9 - (x - 3) ^ 2) - 3
       else if 3 ≤ x ∧ x ≤ 5 then 3 * (x - 3)
       else 0

def j (x : ℝ) : ℝ := h (5 - x)

theorem j_eq_h_shifted (x : ℝ) : j(x) = h(5 - x) := by
  -- Proof omitted
  sorry

end j_eq_h_shifted_l593_593474


namespace distinct_genre_pairs_l593_593911

theorem distinct_genre_pairs 
  (mystery_count : ℕ)
  (fantasy_count : ℕ)
  (biography_count : ℕ)
  (h_mystery : mystery_count = 5)
  (h_fantasy : fantasy_count = 3)
  (h_biography : biography_count = 2) : 
  (mystery_count * fantasy_count) + (mystery_count * biography_count) + (fantasy_count * biography_count) = 31 :=
by
  rw [h_mystery, h_fantasy, h_biography]
  norm_num
  sorry

end distinct_genre_pairs_l593_593911


namespace sin_315_eq_neg_sqrt2_div_2_l593_593056

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593056


namespace min_value_trig_expression_l593_593530

-- Define the problem statement and goal
theorem min_value_trig_expression (x : ℝ) (h1 : 0 < x ∧ x < π / 2) :
  ∃ y > 1, y = 1 / cos x ∧ (8 / (sin x) + 1 / (cos x)) = 10 := 
sorry

end min_value_trig_expression_l593_593530


namespace eggs_in_each_group_l593_593650

theorem eggs_in_each_group (eggs marbles groups : ℕ) 
  (h_eggs : eggs = 15)
  (h_groups : groups = 3) 
  (h_marbles : marbles = 4) :
  eggs / groups = 5 :=
by sorry

end eggs_in_each_group_l593_593650


namespace perfect_squares_l593_593683

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593683


namespace part_1_part_2_l593_593878

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 1 then - x^2 + a * x + 1 else a * x

theorem part_1 (x : ℝ) (a : ℝ) (ha : a = 1) :
  ∃ x1 x2, x1 ≠ x2 ∧ (f a) = (f a) := 
sorry

theorem part_2 (x : ℝ) (a : ℝ) :
  (f a) = (f a) ↔ (0 < a ∧ a < 2) :=
sorry

end part_1_part_2_l593_593878


namespace sequence_formula_l593_593522

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4 ^ n - 1 :=
by
  sorry

end sequence_formula_l593_593522


namespace survival_rate_definition_correct_l593_593372

-- Defining the survival rate in mathematical terms
def survival_rate (living : ℕ) (total : ℕ) : ℝ :=
  (living : ℝ) / (total : ℝ) * 100

-- The statement to be proved
theorem survival_rate_definition_correct (living total : ℕ) (h_total : total ≠ 0) :
  (survival_rate living total = (living : ℝ) / (total : ℝ) * 100) :=
by
  -- By definition of survival_rate
  sorry

end survival_rate_definition_correct_l593_593372


namespace polynomial_arithmetic_sequence_roots_l593_593108

theorem polynomial_arithmetic_sequence_roots (p q : ℝ) (h : ∃ a b c d : ℝ, 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a + 3*(b - a) = b ∧ b + 3*(c - b) = c ∧ c + 3*(d - c) = d ∧ 
  (a^4 + p * a^2 + q = 0) ∧ (b^4 + p * b^2 + q = 0) ∧ 
  (c^4 + p * c^2 + q = 0) ∧ (d^4 + p * d^2 + q = 0)) :
  p ≤ 0 ∧ q = 0.09 * p^2 := 
sorry

end polynomial_arithmetic_sequence_roots_l593_593108


namespace problem_proof_l593_593213

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 5
  else if -2 < x ∧ x < 2 then x^2
  else if 2 ≤ x ∧ x < 10 then 1 / x
  else 0 -- Undefined otherwise for the sake of completeness

theorem problem_proof :
  (∀ x, f(x) ∈ (-∞, 10)) ∧ 
  (f (-2) = 3) ∧ 
  (f (f (f (-2))) = 1 / 9) ∧ 
  (∀ m, f(m) < 2 → m ∈ (-∞, -3) ∪ (-sqrt 2, sqrt 2) ∪ [2, 10)) :=
by
  -- Implementing the proof here is skipped
  sorry

end problem_proof_l593_593213


namespace find_m_l593_593163

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l593_593163


namespace rectangle_R2_area_l593_593853

noncomputable def area_R2 (d1 d2 : ℝ) : ℝ := d1 * d2

theorem rectangle_R2_area (a1 a2 : ℝ) 
  (h1 : a1 = 4) 
  (h2 : a1 * a2 = 32) 
  (d2 : ℝ)
  (h3 : d2 = 20) 
  (h4 : ∃ k : ℝ, a2 = k * a1 ∧ d2 = √(1^2 + (k * 1)^2)) :
  area_R2 (k * 4) (2 * (k * 4)) = 160 :=
by
  sorry

end rectangle_R2_area_l593_593853


namespace quadratic_smaller_solution_l593_593825

theorem quadratic_smaller_solution : ∀ (x : ℝ), x^2 - 9 * x + 20 = 0 → x = 4 ∨ x = 5 :=
by
  sorry

end quadratic_smaller_solution_l593_593825


namespace central_angle_l593_593270

-- Definition: percentage corresponds to central angle
def percentage_equal_ratio (P : ℝ) (θ : ℝ) : Prop :=
  P = θ / 360

-- Theorem statement: Given that P = θ / 360, we want to prove θ = 360 * P
theorem central_angle (P θ : ℝ) (h : percentage_equal_ratio P θ) : θ = 360 * P :=
sorry

end central_angle_l593_593270


namespace sqrt_product_simplification_l593_593468

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end sqrt_product_simplification_l593_593468


namespace number_of_complementary_sets_l593_593497

-- Define the conditions for the card attributes and complementary set rules
structure Card :=
  (shape : Char)
  (color : Char)
  (shade : Char)
  (pattern : Char)

-- The deck consists of 81 unique cards with specific attributes
constant deck : List Card
axiom unique_cards : deck.length = 81
axiom unique_combinations : ∀ c1 c2 : Card, c1 ∈ deck → c2 ∈ deck → (c1 = c2 ↔ (c1.shape = c2.shape ∧ c1.color = c2.color ∧ c1.shade = c2.shade ∧ c1.pattern = c2.pattern))

-- Define what it means for three cards to be complementary
def complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape = c2.shape ∧ c2.shape = c3.shape) ∨ (c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape))
  ∧ ((c1.color = c2.color ∧ c2.color = c3.color) ∨ (c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color))
  ∧ ((c1.shade = c2.shade ∧ c2.shade = c3.shade) ∨ (c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade))
  ∧ ((c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern) ∨ (c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern))

-- Proving the number of complementary sets
theorem number_of_complementary_sets : 
  ∃ sets : List (Card × Card × Card), 
    (∀ s ∈ sets, complementary s.1 s.2 s.3) ∧ sets.length = 630 := 
sorry

end number_of_complementary_sets_l593_593497


namespace volleyball_tournament_ranking_count_l593_593494

theorem volleyball_tournament_ranking_count : 
  let saturday_matches := {EF, GH, IJ}
  let sunday_winners := 3!
  let sunday_losers := 3!
  let total_rankings := sunday_winners * sunday_losers
  total_rankings = 36 :=
by 
  sorry

end volleyball_tournament_ranking_count_l593_593494


namespace find_divisor_l593_593265

-- Define the conditions
def dividend := 689
def quotient := 19
def remainder := 5

-- Define the division formula
def division_formula (divisor : ℕ) : Prop := 
  dividend = (divisor * quotient) + remainder

-- State the theorem to be proved
theorem find_divisor :
  ∃ divisor : ℕ, division_formula divisor ∧ divisor = 36 :=
by
  sorry

end find_divisor_l593_593265


namespace research_development_success_l593_593751

theorem research_development_success 
  (P_A : ℝ)  -- probability of Team A successfully developing a product
  (P_B : ℝ)  -- probability of Team B successfully developing a product
  (independent : Bool)  -- independence condition (dummy for clarity)
  (h1 : P_A = 2/3)
  (h2 : P_B = 3/5) 
  (h3 : independent = true) :
  (1 - (1 - P_A) * (1 - P_B) = 13/15) :=
by
  sorry

end research_development_success_l593_593751


namespace integer_set_properties_l593_593413

theorem integer_set_properties (n : ℕ) (a : fin n → ℤ) (S : set ℤ)
  (h_gcd : int.gcd (list.of_fn a) = 1)
  (h_a : ∀ i, S (a i))
  (h_diff : ∀ i j, S (a i - a j))
  (h_closure : ∀ x y, S x → S y → S (x + y) → S (x - y)) :
  S = set.univ :=
begin
  sorry
end

end integer_set_properties_l593_593413


namespace find_a_b_find_k_range_l593_593894

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l593_593894


namespace cube_volume_from_sphere_volume_l593_593864

noncomputable def circumscribed_sphere_volume : ℝ := (32 / 3) * Real.pi

theorem cube_volume_from_sphere_volume
  (h : 4 / 3 * Real.pi * (2 : ℝ)^3 = circumscribed_sphere_volume) :
  let a := (4 * Real.sqrt 3) / 3 in a^3 = 64 * Real.sqrt 3 / 9 :=
by
  sorry

end cube_volume_from_sphere_volume_l593_593864


namespace axis_of_symmetry_smallest_positive_period_range_of_h_l593_593553

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x + π / 12) ^ 2
noncomputable def g (x : ℝ) : ℝ := 3 + 2 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, f (x + π) = f (x - π) ↔ x = (k * π) / 2 - π / 12 :=
sorry

theorem smallest_positive_period :
  ∃ T : ℝ, (T > 0) ∧ ∀ x : ℝ, h (x + T) = h x ∧ T = π :=
sorry

theorem range_of_h :
  ∀ x : ℝ, 3 ≤ h x ∧ h x ≤ 5 :=
sorry

end axis_of_symmetry_smallest_positive_period_range_of_h_l593_593553


namespace sin_315_eq_neg_sqrt2_over_2_l593_593029

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593029


namespace sqrt_trig_identity_l593_593336

variable (α : ℝ)

theorem sqrt_trig_identity (h : sin α * sin α + cos α * cos α = 1) : 
  sqrt (1 - sin α * sin α) = abs (cos α) :=
by 
  sorry

end sqrt_trig_identity_l593_593336


namespace overlapping_area_is_2_l593_593636

open Locale 
open Lean 

def Point := (ℝ × ℝ)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

def triangle1 : Point × Point × Point :=
  ((0, 0), (2, 2), (0, 2))

def triangle2 : Point × Point × Point :=
  ((2, 0), (0, 2), (2, 2))

def overlapping_area_of_triangles (t1 t2 : Point × Point × Point) : ℝ :=
  if (t1 = triangle1 ∨ t2 = triangle2) then
    2
  else
    sorry

theorem overlapping_area_is_2 :
  overlapping_area_of_triangles triangle1 triangle2 = 2 :=
by
  rw overlapping_area_of_triangles
  fin_cases sorry
  all_goals sorry

end overlapping_area_is_2_l593_593636


namespace rhombus_side_length_l593_593369

theorem rhombus_side_length (d1 : ℝ) (area : ℝ) (side : ℝ) 
  (h_d1 : d1 = 10) 
  (h_area : area = 244.9489742783178) 
  (h_side : side = 47.958315233127194) : 
  (side : ℝ) = 2 * real.sqrt ((2 * area) / d1 - 25) :=
by
  have h_d2 : (2 * area) / d1 = 48.98979485566356 :=
    by rw [h_d1, h_area]; norm_num
  have h_half_side : real.sqrt (575.0000000000001) = 23.979157616563597 :=
    by norm_num
  rw [←h_half_side]
  have h_side_calc : side = 2 * 23.979157616563597 :=
    by rw [h_side]; norm_num
  exact calc
    side = 2 * real.sqrt (575.0000000000001) := by rw h_side_calc
         ... = 2 * real.sqrt ((2 * area) / d1 - 25) := by rw [h_d2]; norm_num 

end rhombus_side_length_l593_593369


namespace power_function_through_point_l593_593536

def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_through_point :
  ∃ a : ℝ, power_function a 2 = 32 ∧ (∀ x : ℝ, power_function a x = x^5) :=
begin
  sorry
end

end power_function_through_point_l593_593536


namespace sin_315_eq_neg_sqrt2_over_2_l593_593039

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593039


namespace max_area_rectangle_in_circular_segment_correct_l593_593967

noncomputable def max_area_rectangle_in_circular_segment (r : ℝ) (α : ℝ) : ℝ :=
  if 0 < α ∧ α ≤ (π / 2) then
    let y := r / 4 * (cos α + sqrt (8 + cos α ^ 2))
    let x := sqrt (r ^ 2 - y ^ 2)
    2 * x * (y - r * cos α)
  else if (π / 2) < α ∧ α < (3 * π / 4) then
    2 * r ^ 2 * sin (2 * (α - (π / 2)))
  else if (3 * π / 4) ≤ α ∧ α ≤ π then
    r ^ 2 * sin α
  else
    0 -- or handle invalid α as needed

theorem max_area_rectangle_in_circular_segment_correct (r : ℝ) (α : ℝ) :
  0 < α → α ≤ π →
  ∃ T, T = max_area_rectangle_in_circular_segment r α := by
  sorry

end max_area_rectangle_in_circular_segment_correct_l593_593967


namespace prove_angle_A_l593_593961

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end prove_angle_A_l593_593961


namespace no_valid_years_between_10000_and_20000_l593_593765

def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString
  s = s.reverse

def two_digit_prime_palindromes : list ℕ := [11, 101, 131, 151, 181, 191]

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m > 1 ∧ m * m ≤ n → n % m ≠ 0)

def four_digit_prime (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ is_prime n

theorem no_valid_years_between_10000_and_20000 : 
  {n : ℕ | 10000 ≤ n ∧ n < 20000 ∧ is_palindrome n ∧ 
           ∃ a b, a ∈ two_digit_prime_palindromes ∧ four_digit_prime b ∧ n = a * b }.card = 0 := 
  sorry

end no_valid_years_between_10000_and_20000_l593_593765


namespace sequence_decreasing_l593_593218

noncomputable def sequence (n : ℕ) (a : ℝ) : ℝ :=
if h : n > 8 then (1 / 3 - a) * n + 8 else a ^ (n - 7)

theorem sequence_decreasing (a : ℝ) (h : ∀ n : ℕ, 0 < n → sequence n a > sequence (n + 1) a) : 
  1 / 2 ≤ a ∧ a < 1 :=
sorry

end sequence_decreasing_l593_593218


namespace minimal_value_of_M_l593_593974

theorem minimal_value_of_M {m n : ℕ} {a : ℕ → ℕ → ℕ} 
  (h1 : m > 1)
  (h2 : 3 ≤ n ∧ n < 2 * m ∧ odd n)
  (h3 : ∀ j, 1 ≤ j ∧ j ≤ n → ∃ (σ : Fin m → Fin m), 
        ∀ i, 1 ≤ i ∧ i ≤ m → a i j = σ ⟨i - 1, by linarith⟩.val + 1 )
  (h4 : ∀ i, 2 ≤ i ∧ i ≤ m → ∀ j, 1 ≤ j ∧ j < n → |a i j - a i (j + 1)| ≤ 1) :
  (∃ (M : ℕ), M = Nat.max (List.map (λ i, List.sum (List.map (λ j, a i j) (List.range n))) (List.range (m - 1)).map (λ x, x + 2)) 
  ∧ M = 7) :=
sorry

end minimal_value_of_M_l593_593974


namespace find_value_of_fraction_l593_593243

theorem find_value_of_fraction (x y z : ℝ)
  (h1 : 3 * x - 4 * y - z = 0)
  (h2 : x + 4 * y - 15 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3 * x * y - y * z) / (y^2 + z^2) = 2.4 :=
by
  sorry

end find_value_of_fraction_l593_593243


namespace max_small_planes_max_medium_planes_max_large_planes_l593_593449

noncomputable theory

-- Definition of the problem's conditions
def hangar_length : ℕ := 900
def small_plane_length : ℕ := 50
def medium_plane_length : ℕ := 75
def large_plane_length : ℕ := 110
def safety_gap : ℕ := 10

-- Calculating the required space for each type of plane including the safety gap
def space_per_small_plane : ℕ := small_plane_length + safety_gap
def space_per_medium_plane : ℕ := medium_plane_length + safety_gap
def space_per_large_plane : ℕ := large_plane_length + safety_gap

-- Theorems to prove the maximum number of planes of each type that fit in the hangar
theorem max_small_planes : hangar_length / space_per_small_plane = 15 := by sorry
theorem max_medium_planes : hangar_length / space_per_medium_plane = 10 := by sorry
theorem max_large_planes : hangar_length / space_per_large_plane = 7 := by sorry

end max_small_planes_max_medium_planes_max_large_planes_l593_593449


namespace tank_depth_l593_593446

theorem tank_depth :
  ∃ d : ℝ, 
    let A := (25 * 12) + 2 * (25 * d) + 2 * (12 * d) in 
    let cost_per_sq_m := 0.55 in
    let total_cost := 409.20 in
    total_cost = A * cost_per_sq_m ∧ d = 6 :=
sorry

end tank_depth_l593_593446


namespace percentage_two_sections_cleared_l593_593594

noncomputable def total_candidates : ℕ := 1200
def pct_cleared_all_sections : ℝ := 0.05
def pct_cleared_none_sections : ℝ := 0.05
def pct_cleared_one_section : ℝ := 0.25
def pct_cleared_four_sections : ℝ := 0.20
def cleared_three_sections : ℕ := 300

theorem percentage_two_sections_cleared :
  (total_candidates - total_candidates * (pct_cleared_all_sections + pct_cleared_none_sections + pct_cleared_one_section + pct_cleared_four_sections) - cleared_three_sections) / total_candidates * 100 = 20 := by
  sorry

end percentage_two_sections_cleared_l593_593594


namespace sin_315_eq_neg_sqrt2_div_2_l593_593076

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593076


namespace find_box_length_l593_593726

noncomputable def length_of_one_side_box
  (cost_per_box : ℝ) (total_cost : ℝ) (total_volume : ℝ) : ℝ :=
  let number_of_boxes := total_cost / cost_per_box in
  let volume_per_box := total_volume / number_of_boxes in
  volume_per_box^(1/3:ℝ)

theorem find_box_length :
  length_of_one_side_box 0.40 200 2400000 ≈ 16.8 :=
begin
  sorry
end

end find_box_length_l593_593726


namespace pairwise_independent_neither_sufficient_nor_necessary_l593_593146

-- Definitions of events A, B, and C
variables {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
variables (A B C : Set Ω)

-- Conditions stating that A, B, C are pairwise independent
def pairwise_independent (A B C : Set Ω) : Prop :=
  P(A ∩ B) = P A * P B ∧
  P(B ∩ C) = P B * P C ∧
  P(A ∩ C) = P A * P C

-- Definition of the desired equation to check for sufficiency and necessity
def check_equation (A B C : Set Ω) : Prop :=
  P(A ∩ B ∩ C) = P A * P B * P C

-- The main theorem statement
theorem pairwise_independent_neither_sufficient_nor_necessary (h: pairwise_independent P A B C) :
  ¬ (check_equation P A B C ∧
     (∀ A B C, check_equation P A B C → pairwise_independent P A B C)) :=
sorry

end pairwise_independent_neither_sufficient_nor_necessary_l593_593146


namespace marcy_cat_time_l593_593996

theorem marcy_cat_time (petting_time combing_ratio : ℝ) :
  petting_time = 12 ∧ combing_ratio = 1/3 → (petting_time + petting_time * combing_ratio) = 16 :=
by
  intros h
  cases h with petting_eq combing_eq
  rw [petting_eq, combing_eq]
  norm_num


end marcy_cat_time_l593_593996


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593016

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593016


namespace trigonometric_identity_l593_593258

theorem trigonometric_identity 
  (m n : ℝ) (α : ℝ) 
  (h1 : n / m = -2)
  (h2 : α ∈ set.Icc 0 Real.pi)
  (h3 : m ^ 2 + n ^ 2 = 1) :
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 :=
by
  sorry

end trigonometric_identity_l593_593258


namespace intersection_A_B_l593_593742

-- Define the sets A and B
def A : Set ℤ := {1, 3, 5, 7}
def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

-- The goal is to prove that A ∩ B = {3, 5}
theorem intersection_A_B : A ∩ B = {3, 5} :=
  sorry

end intersection_A_B_l593_593742


namespace proof_problem_l593_593343

theorem proof_problem
  (a b c : ℂ)
  (h1 : ac / (a + b) + ba / (b + c) + cb / (c + a) = -4)
  (h2 : bc / (a + b) + ca / (b + c) + ab / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 7 := 
sorry

end proof_problem_l593_593343


namespace terms_are_equal_l593_593311

theorem terms_are_equal (n : ℕ) (a b : ℕ → ℕ)
  (h_n : n ≥ 2018)
  (h_a : ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_b : ∀ i j : ℕ, i ≠ j → b i ≠ b j)
  (h_a_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i > 0)
  (h_b_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i > 0)
  (h_a_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 5 * n)
  (h_b_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i ≤ 5 * n)
  (h_arith : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → (a j * b i - a i * b j) * (j - i) = 0):
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i * b j = a j * b i :=
by
  sorry

end terms_are_equal_l593_593311


namespace evaluate_fraction_l593_593105

theorem evaluate_fraction (a b : ℤ) (h1 : a = 5) (h2 : b = -2) : (5 : ℝ) / (a + b) = 5 / 3 :=
by
  sorry

end evaluate_fraction_l593_593105


namespace sum_arithmetic_series_l593_593791

theorem sum_arithmetic_series : 
    let a₁ := 1
    let d := 2
    let n := 9
    let a_n := a₁ + (n - 1) * d
    let S_n := n * (a₁ + a_n) / 2
    a_n = 17 → S_n = 81 :=
by intros
   sorry

end sum_arithmetic_series_l593_593791


namespace sin_315_degree_l593_593052

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593052


namespace book_pages_l593_593241

theorem book_pages (x : ℕ) : 
  (x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15) - (1/3 * ((x - (1/5 * x + 12)) - (1/4 * (x - (1/5 * x + 12)) + 15)) + 18) = 62 →
  x = 240 :=
by
  -- This is where the proof would go, but it's omitted for this task.
  sorry

end book_pages_l593_593241


namespace no_positive_abc_exist_l593_593491

theorem no_positive_abc_exist (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * d^2 + b * d - c = 0) ∧ (sqrt a * d + sqrt b * sqrt d - sqrt c = 0) → false :=
by
  sorry

end no_positive_abc_exist_l593_593491


namespace first_term_of_geometric_sequence_l593_593356

theorem first_term_of_geometric_sequence (a r : ℝ)
  (h4 : a * r^3 = nat.factorial 6)
  (h7 : a * r^6 = nat.factorial 7) :
  a = 720 / 7 := sorry

end first_term_of_geometric_sequence_l593_593356


namespace num_red_hats_is_3_l593_593706

def num_children : ℕ := 8

def sees_red_hats (num_red_hats seen : ℕ) : Prop :=
  if seen ≥ 3 then true else false

theorem num_red_hats_is_3 (num_red_hats blue_balloons red_balloons : ℕ) :
  num_red_hats + blue_balloons = num_children →
  (∀ (c : ℕ), c < num_children → if sees_red_hats num_red_hats (num_red_hats - 1) then red_balloons else blue_balloons) →
  red_balloons > 0 ∧ blue_balloons > 0 →
  num_red_hats = 3 := 
  sorry

end num_red_hats_is_3_l593_593706


namespace find_b_and_area_l593_593222

variables (a c b : ℝ) (B : ℝ)

def cos_rule_b (a c B : ℝ) : ℝ :=
  real.sqrt (a^2 + c^2 - 2 * a * c * real.cos B)

theorem find_b_and_area (h_a : a = 3 * real.sqrt 3) 
  (h_c : c = 2) 
  (h_B : B = real.pi * 5 / 6) -- 150 degrees in radians
  (h_cos_B : real.cos B = - real.sqrt 3 / 2)
  (h_sin_B : real.sin B = 1 / 2) : 
  b = 7 ∧ 0.5 * a * c * real.sin B = (3 / 2) * real.sqrt 3 :=
by
  have h_b_sq : b^2 = a^2 + c^2 - 2 * a * c * real.cos B, by rw [real.cos, cos_rule_b]
  have h_b : b = real.sqrt h_b_sq, by rw [h_b_sq, real.sqrt]
  sorry

end find_b_and_area_l593_593222


namespace perfect_squares_between_50_and_250_l593_593235

theorem perfect_squares_between_50_and_250 : 
  (card { n : ℕ | 50 ≤ n^2 ∧ n^2 ≤ 250 }) = 8 := 
by
  sorry

end perfect_squares_between_50_and_250_l593_593235


namespace convex_symmetric_polygon_area_bound_l593_593615

-- Definitions and conditions
def isSymmetric (P : set (ℝ × ℝ)) (O : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ × ℝ), x ∈ P → (2 * O - x = y) → y ∈ P

def isConvex (P : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ × ℝ), x ∈ P → y ∈ P → ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → t • x + (1 - t) • y ∈ P

-- Problem statement
theorem convex_symmetric_polygon_area_bound (P : set (ℝ × ℝ))
  (O : ℝ × ℝ) (h_convex : isConvex P) (h_symmetric : isSymmetric P O) :
  ∃ (R : set (ℝ × ℝ)), (P ⊆ R) ∧ ∃ (f : ℝ), f ≤ sqrt 2 ∧ 
    (measure_theory.measure.countable (R \ P)).sum ∣↑(measure_theory.outer_measure.lebesgue_measure R)∣ / ∣↑(measure_theory.outer_measure.lebesgue_measure P)∣ := sorry

end convex_symmetric_polygon_area_bound_l593_593615


namespace max_value_inequality_l593_593990

theorem max_value_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 3) :
  (x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) ≤ 27 := 
sorry

end max_value_inequality_l593_593990


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593025

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593025


namespace find_m_l593_593164

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l593_593164


namespace xiaoming_password_count_l593_593727

theorem xiaoming_password_count : 
  let digits := [0, 5, 0, 9, 1, 9] in
  let pair_adjacent (x : Nat) (lst : List Nat) := 
    ∃ i, lst.take i = lst.take i.append [x, x] in
  (pair_adjacent 9 digits) ∧ (pair_adjacent 0 digits) → 
  ∃ (p : Fin 24), p.val = 24 :=
begin
  -- sorry is used to skip the proof in this statement.
  sorry

end xiaoming_password_count_l593_593727


namespace cab_income_third_day_l593_593748

noncomputable def cab_driver_income (day1 day2 day3 day4 day5 : ℕ) : ℕ := 
day1 + day2 + day3 + day4 + day5

theorem cab_income_third_day 
  (day1 day2 day4 day5 avg_income total_income day3 : ℕ)
  (h1 : day1 = 45)
  (h2 : day2 = 50)
  (h3 : day4 = 65)
  (h4 : day5 = 70)
  (h_avg : avg_income = 58)
  (h_total : total_income = 5 * avg_income)
  (h_day_sum : day1 + day2 + day4 + day5 = 230) :
  total_income - 230 = 60 :=
sorry

end cab_income_third_day_l593_593748


namespace sin_315_eq_neg_sqrt2_div_2_l593_593069

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593069


namespace sin_315_eq_neg_sqrt2_div_2_l593_593064

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593064


namespace equilibrium_constant_NH4I_l593_593339

theorem equilibrium_constant_NH4I
  (H2_concentration : ℝ)
  (HI_concentration : ℝ)
  (H2_at_equilibrium : H2_concentration = 0.5)
  (HI_at_equilibrium : HI_concentration = 4) :
  let NH3_concentration := HI_concentration + 2 * H2_concentration,
      K := NH3_concentration * HI_concentration in
  K = 20 :=
by
  sorry

end equilibrium_constant_NH4I_l593_593339


namespace find_m_l593_593162

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l593_593162


namespace distinct_pair_with_equal_values_l593_593312

theorem distinct_pair_with_equal_values 
  (p q n : ℕ) (hpq : p + q < n) 
  (x : fin (n + 1) → ℤ)
  (hx0 : x 0 = 0) (hxn : x ⟨n, nat.lt_succ_self n⟩ = 0)
  (h : ∀ i : fin n, (x ⟨i + 1, nat.succ_lt_succ (nat.lt_of_lt_succ i.2)⟩ - x ⟨i, i.2⟩ = p ∨ 
                     x ⟨i + 1, nat.succ_lt_succ (nat.lt_of_lt_succ i.2)⟩ - x ⟨i, i.2⟩ = -q)) :
  ∃ i j : fin (n + 1), i ≠ j ∧ x i = x j ∧ (i ≠ 0 ∨ j ≠ ⟨n, nat.lt_succ_self n⟩) :=
sorry

end distinct_pair_with_equal_values_l593_593312


namespace sqrt_product_simplification_l593_593469

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end sqrt_product_simplification_l593_593469


namespace min_expr_value_l593_593823

theorem min_expr_value : ∃ (x y : ℝ), ∀ a b : ℝ, 3 * a^2 + 4 * a * b + 2 * b^2 - 6 * a + 4 * b + 5 ≥ -1 :=
begin
  exist x = -2,
  exist y = -3,
  intro a,
  intro b,
  have h : 3 * a^2 + 4 * a * b + 2 * b^2 - 6 * a + 4 * b + 5 ≥ -1,
  sorry
end

end min_expr_value_l593_593823


namespace find_m_l593_593176

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l593_593176


namespace coefficient_a9_l593_593578

theorem coefficient_a9 (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℤ) :
  (x^2 + x^10 = a0 + a1 * (x + 1) + a2 * (x + 1)^2 + a3 * (x + 1)^3 +
   a4 * (x + 1)^4 + a5 * (x + 1)^5 + a6 * (x + 1)^6 + a7 * (x + 1)^7 +
   a8 * (x + 1)^8 + a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a10 = 1 →
  a9 = -10 :=
by
  sorry

end coefficient_a9_l593_593578


namespace f_is_periodic_l593_593140

-- Define the conditions for the function f
def f (x : ℝ) : ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x ≠ 0
axiom f_property : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f (x - a) = 1 / f x

-- Formal problem statement to be proven
theorem f_is_periodic : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = f (x + 2 * a) :=
by {
  sorry
}

end f_is_periodic_l593_593140


namespace no_city_more_than_5_planes_l593_593588

-- Define the city and distance properties
def unique_distances (cities : Type) (dist : cities → cities → ℝ) : Prop :=
  ∀ (a b c : cities), a ≠ b → a ≠ c → b ≠ c → dist a b ≠ dist a c

def flown_to_closest_city (cities : Type) (dist : cities → cities → ℝ) (flown_to : cities → cities) : Prop :=
  ∀ (c : cities), flown_to c = c' → (∀ (d : cities), d ≠ c → dist c c' ≤ dist c d)

-- Define the theorem stating the problem
theorem no_city_more_than_5_planes
  (cities : Type)
  (dist : cities → cities → ℝ)
  (flown_to : cities → cities) :
  unique_distances cities dist →
  flown_to_closest_city cities dist flown_to →
  ∀ (c : cities), (∑ x, if flown_to x = c then 1 else 0) ≤ 5 :=
by
  intros,
  sorry

end no_city_more_than_5_planes_l593_593588


namespace inequality1_solution_inequality2_solution_l593_593657

-- Definitions for the conditions
def cond1 (x : ℝ) : Prop := abs (1 - (2 * x - 1) / 3) ≤ 2
def cond2 (x : ℝ) : Prop := (2 - x) * (x + 3) < 2 - x

-- Lean 4 statement for the proof problem
theorem inequality1_solution (x : ℝ) : cond1 x → -1 ≤ x ∧ x ≤ 5 := by
  sorry

theorem inequality2_solution (x : ℝ) : cond2 x → x > 2 ∨ x < -2 := by
  sorry

end inequality1_solution_inequality2_solution_l593_593657


namespace ratio_of_faulty_boards_passing_verification_l593_593587

theorem ratio_of_faulty_boards_passing_verification :
  ∀ (total_boards failed_boards total_faulty_boards passed_boards faulty_passed_boards : ℕ),
    total_boards = 3200 ->
    failed_boards = 64 ->
    total_faulty_boards = 456 ->
    passed_boards = total_boards - failed_boards ->
    faulty_passed_boards = total_faulty_boards - failed_boards ->
    faulty_passed_boards * 8 = passed_boards := by
  intros total_boards failed_boards total_faulty_boards passed_boards faulty_passed_boards h_total h_failed h_faulty h_passed h_faulty_passed
  gg sorry

end ratio_of_faulty_boards_passing_verification_l593_593587


namespace sum_of_digits_of_stair_steps_is_18_l593_593612

theorem sum_of_digits_of_stair_steps_is_18 (n : ℕ) :
  (∃ c d : ℕ, c ∈ ({0,1,2} : Finset ℕ) ∧ d ∈ ({0,1,2,3} : Finset ℕ) ∧ 
              4 * (n + c) - 3 * (n + d) = 288 ∧
              (divMod n 10).fst + (divMod (n / 10) 10).fst + (divMod (n / 100) 10).fst = 18) :=
sorry

end sum_of_digits_of_stair_steps_is_18_l593_593612


namespace find_m_l593_593253

theorem find_m (m : ℝ) (h1 : ∀ x y : ℝ, (x ^ 2 + (y - 2) ^ 2 = 1) → (y = x / m ∨ y = -x / m)) (h2 : 0 < m) :
  m = (Real.sqrt 3) / 3 :=
by
  sorry

end find_m_l593_593253


namespace sin_315_eq_neg_sqrt2_div_2_l593_593072

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593072


namespace find_k_positive_integer_solution_l593_593199

theorem find_k_positive_integer_solution {k x : ℤ} (h : 9 * x - 3 = k * x + 14) (hx : x > 0) :
  k = 8 ∨ k = -8 :=
begin
  sorry
end

end find_k_positive_integer_solution_l593_593199


namespace point_direction_form_of_line_l593_593534

theorem point_direction_form_of_line (P : ℝ × ℝ) (d : ℝ × ℝ) (x y : ℝ) :
  P = (1, 2) → d = (3, -4) → ((x - 1) / 3 = (y - 2) / -4) :=
by
  intro hP hd
  rw [hP, hd]
  sorry

end point_direction_form_of_line_l593_593534


namespace functional_relationship_yx_selling_price_for_sales_profit_functional_relationship_wx_maximized_profit_l593_593397

noncomputable def y (x : ℝ) : ℝ := -10 * x + 280

theorem functional_relationship_yx (x : ℝ) (hx₁ : 6 ≤ x) (hx₂ : x ≤ 12) :
  y x = -10 * x + 280 :=
begin
  exact rfl,
end

theorem selling_price_for_sales_profit (x : ℝ) (hx : x = 10) :
  (x - 6) * y x = 720 :=
begin
  sorry,
end

noncomputable def w (x : ℝ) : ℝ := -10 * x ^ 2 + 220 * x - 1680

theorem functional_relationship_wx (x : ℝ) (hx₁ : 6 ≤ x) (hx₂ : x ≤ 12) :
  w x = -10 * x ^ 2 + 220 * x - 1680 :=
begin
  exact rfl,
end

theorem maximized_profit :
  ∃ x : ℝ, (6 ≤ x ∧ x ≤ 12) ∧ w x = 1210 :=
begin
  use 11,
  split,
  { split; norm_num, },
  { sorry, }
end

end functional_relationship_yx_selling_price_for_sales_profit_functional_relationship_wx_maximized_profit_l593_593397


namespace find_x_in_data_set_l593_593688

noncomputable def mean (data : List ℝ) : ℝ :=
data.sum / (data.length : ℝ)

def median (data : List ℝ) : ℝ :=
let sorted_data := data.qsort (≤)
if data.length % 2 = 1 then
  sorted_data.nthLe (data.length / 2) (by simp)
else
  (sorted_data.nthLe (data.length / 2 - 1) (by simp) + 
   sorted_data.nthLe (data.length / 2) (by simp)) / 2

def mode (data : List ℝ) : ℝ :=
(data.groupBy id).values.maxBy (λ xs => xs.length) |>.head

theorem find_x_in_data_set :
  ∃ x : ℝ, mean [60, 100, x, 40, 50, 200, 90] = x ∧
           median [60, 100, x, 40, 50, 200, 90] = x ∧
           mode [60, 100, x, 40, 50, 200, 90] = x ∧
           x = 90 :=
begin
  use 90,
  sorry
end

end find_x_in_data_set_l593_593688


namespace solution_set_l593_593863

variables {α : Type*} [LinearOrder α] [DenselyOrdered α]
variables (f : α → α) (x : α)

-- Condition definitions
def odd_function : Prop := ∀ x, f(-x) = -f(x)
def monotone_dec : Prop := ∀ x y, x < y → y < 0 → f(x) > f(y)
def f_3_eq_zero : Prop := f(3) = 0

-- Inequality definition
def inequality_holds : Prop := (x - 1) * f(x) > 0

-- Final statement
theorem solution_set (h1 : odd_function f) (h2 : monotone_dec f) (h3 : f_3_eq_zero f) :
  {x : α | inequality_holds f x} = set.Ioo (-1 : α) 0 ∪ set.Ioo (1 : α) 3 :=
sorry

end solution_set_l593_593863


namespace find_m_l593_593177

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l593_593177


namespace angle_B_l593_593605

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 3
noncomputable def A : ℝ := Real.pi / 6 -- Converting degrees to radians: 30 degrees = π/6 radians
def B1 := Real.pi / 3 -- 60 degrees
def B2 := 2 * Real.pi / 3 -- 120 degrees

theorem angle_B (B : ℝ) (h1 : real.sin B = b * real.sin A / a) : B = B1 ∨ B = B2 := 
sorry

end angle_B_l593_593605


namespace construct_triangle_l593_593798

-- Define the given conditions in Lean
variables (α m r : ℝ) (O B C A A_1 : Type)

-- Assuming given conditions about the triangle
axiom α_pos : 0 < α
axiom r_pos : 0 < r
axiom m_pos : 0 < m
axiom circumcircle_radius : ∀ (A' B' C' : Type), (distance O A') = r ∧ (distance O B') = r ∧ (distance O C') = r

-- Define the central angle BOC = 2α
noncomputable def central_angle (B C : Type) : ℝ := 2 * α
axiom central_angle_def : ∀ (B C : Type), central_angle B C = 2 * α

-- Define condition about the height from vertex to side
axiom height_condition : ∀ (A B C : Type), (height_from_vertex A B C) = m

-- The statement we want to prove:
theorem construct_triangle : 
  ∃ (A B C : Type), (angle BAC = α) ∧ (height_from_vertex A B C = m) ∧ (circumradius A B C = r) ∧ 
                    (∃ (A_1 : Type), (angle BA_1C = α) ∧ (height_from_vertex A_1 B C = m) ∧ (circumradius A_1 B C = r)) := sorry

end construct_triangle_l593_593798


namespace profit_without_discount_l593_593399

-- Definitions from conditions
def discount : ℝ := 0.05
def profit_with_discount : ℝ := 0.387

-- Statement to prove
theorem profit_without_discount 
  (d : ℝ)
  (p_d : ℝ)
  (CP : ℝ) 
  (SP : CP * (1 + p_d)):
  (d = 0.05) → 
  (p_d = 0.387) →
  (((SP - CP) / CP) * 100 = 38.7) :=
by {
  sorry
}

end profit_without_discount_l593_593399


namespace max_odd_row_sums_l593_593422

-- Definition of a cube with edge length 3
structure Cube :=
  (values : fin 3 × fin 3 × fin 3 → ℕ)
  (distinct_values : ∀ i j, i ≠ j → values i ≠ values j)
  (in_range : ∀ i, 1 ≤ values i ∧ values i ≤ 27)

-- Definition to get row sums in one direction
def row_sums (c : Cube) : fin 3 → fin 3 → ℕ :=
  λ i j, c.values (i, j, 0) + c.values (i, j, 1) + c.values (i, j, 2)

-- Extension of row sums to all directions
def all_row_sums (c : Cube) : list ℕ :=
  [ ... ] -- Placeholder for all 27 sums in x, y, z directions

-- Function to count odd numbers in a list
def count_odds (l : list ℕ) : ℕ :=
  l.countp (λ n, n % 2 = 1)

-- Problem statement
theorem max_odd_row_sums (c : Cube) : count_odds (all_row_sums c) ≤ 24 :=
  sorry

end max_odd_row_sums_l593_593422


namespace initial_pieces_l593_593792

-- Definitions based on given conditions
variable (left : ℕ) (used : ℕ)
axiom cond1 : left = 93
axiom cond2 : used = 4

-- The mathematical proof problem statement
theorem initial_pieces (left used : ℕ) (cond1 : left = 93) (cond2 : used = 4) : left + used = 97 :=
by
  sorry

end initial_pieces_l593_593792


namespace central_angle_of_regular_octagon_l593_593350

theorem central_angle_of_regular_octagon :
  (∑ k in finset.range 8, (45 : ℝ)) = 360 :=
by sorry

end central_angle_of_regular_octagon_l593_593350


namespace displacement_of_particle_l593_593438

theorem displacement_of_particle :
  ∫ t in (1 : ℝ)..(2 : ℝ), (t^2 - t + 2) = 17 / 6 :=
begin
  sorry
end

end displacement_of_particle_l593_593438


namespace tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l593_593872

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem tangent_line_at_x_equals_1 (a : ℝ) (x : ℝ) (h₀ : a = 2) (h₁ : x = 1) : 
  3 * x - (f a 1) - 1 = 0 := 
sorry

theorem monotonic_intervals (a x : ℝ) (h₀ : x > 0) :
  ((a >= 0 ∧ ∀ (x : ℝ), x > 0 → (f a x) > (f a (x - 1))) ∨ 
  (a < 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < -1/a → (f a x) > (f a (x - 1)) ∧ ∀ (x : ℝ), x > -1/a → (f a x) < (f a (x - 1)))) :=
sorry

theorem range_of_a (a x : ℝ) (h₀ : 0 < x) (h₁ : f a x < 2) : a < -1 / Real.exp (3) :=
sorry

end tangent_line_at_x_equals_1_monotonic_intervals_range_of_a_l593_593872


namespace general_formula_sequence_l593_593359

-- Define the sequence as an arithmetic sequence with the given first term and common difference
def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

-- Define given values
def a_1 : ℕ := 1
def d : ℕ := 2

-- State the theorem to be proved
theorem general_formula_sequence :
  ∀ n : ℕ, n > 0 → arithmetic_sequence a_1 d n = 2 * n - 1 :=
by
  intro n hn
  sorry

end general_formula_sequence_l593_593359


namespace infinite_primes_6k_plus_1_l593_593327

theorem infinite_primes_6k_plus_1 : ∀ N : ℕ, ∃ p : ℕ, prime p ∧ p > N ∧ ∃ k : ℕ, p = 6 * k + 1 :=
  sorry

end infinite_primes_6k_plus_1_l593_593327


namespace find_y1_l593_593602

theorem find_y1 : 
  (∃ (y1 x2 : ℝ), line_through_point_and_x_intercept (-12, y1) 4 ∧ line_through_point_and_y_intercept x2 3 ∧ line_intersects_another_point (4, 0)) → y1 = 0 :=
by
  sorry

-- Definitions of the conditions used in the theorem
def line_through_point_and_x_intercept (P1 : ℝ × ℝ) (x_intercepts : ℝ) : Prop :=
  let slope : ℝ := (0 - P1.snd) / (x_intercepts - P1.fst)
  (λ x, slope * (x - P1.fst) + P1.snd)

def line_through_point_and_y_intercept (x : ℝ) (y : ℝ) : Prop :=
  let slope : ℝ := (y - 0) / (x - 0)
  (λ x, slope * (x - x_intercepts) + 0)

def line_intersects_another_point (P : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), y = P.snd

end find_y1_l593_593602


namespace sin_315_degree_l593_593054

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593054


namespace seq_eq_n_l593_593106

theorem seq_eq_n (a : ℕ → ℤ) :
  (∀ n, a n ∈ ℤ) ∧
  a 0 = 0 ∧
  a 1 = 1 ∧
  (∃ infi_m, ∀ m ≥ infi_m, a m = m) ∧
  (∀ n ≥ 2, (finset.image (λ i, 2 * a i - a (i - 1)) (finset.range n)).val = (finset.range n).val) →
  (∀ n, a n = n) :=
by
  sorry

end seq_eq_n_l593_593106


namespace union_complement_eq_l593_593628

/-- The universal set U and sets A and B as given in the problem. -/
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

/-- The lean statement of our proof problem. -/
theorem union_complement_eq : A ∪ (U \ B) = {0, 1, 2, 3} := by
  sorry

end union_complement_eq_l593_593628


namespace range_of_a_l593_593254

def f (a x : ℝ) : ℝ :=
  if x ≤ a then cos x else 1 / x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -1 ≤ f a x ∧ f a x ≤ 1) → 1 ≤ a := 
sorry

end range_of_a_l593_593254


namespace can_disect_into_smaller_staircases_l593_593617

def is_staircase (n : ℕ) (S : Finset (ℕ × ℕ)) : Prop :=
  (∃ (f : ℕ → Finset (ℕ × ℕ)), (∀ i < n, (f i).card = i + 1) ∧ S = (Finset.range n).bUnion f)

theorem can_disect_into_smaller_staircases (n : ℕ) (hn : n > 0) (S : Finset (ℕ × ℕ)) (hS : is_staircase n S) :
  ∃ (T : Finset (ℕ × ℕ)), T ⊂ S ∧ is_staircase n T :=
sorry

end can_disect_into_smaller_staircases_l593_593617


namespace counterexample_prime_l593_593092

theorem counterexample_prime (n : ℕ) (hn_prime : Prime n) : 
  Prime (n + 2) → n = 3 := 
sorry

example : ∃ n : ℕ, Prime n ∧ Prime (n + 2) :=
⟨3, by norm_num, by norm_num⟩

end counterexample_prime_l593_593092


namespace part1_part2_l593_593889

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l593_593889


namespace min_value_abs_diff_l593_593245

theorem min_value_abs_diff (a b : ℕ) (h : 0 < a ∧ 0 < b ∧ ab - 7 * a + 2 * b = 15) : |a - b| = 19 :=
sorry

end min_value_abs_diff_l593_593245


namespace students_not_answering_yes_l593_593416

theorem students_not_answering_yes (total_students yes_M yes_R yes_only_M yes_both yes_one : ℕ) 
(h1 : total_students = 800)
(h2 : yes_M = 500)
(h3 : yes_R = 400)
(h4 : yes_only_M = 170)
(h5 : yes_both = yes_M - yes_only_M)
(h6 : yes_one = yes_only_M + yes_both + (yes_R - yes_both)) :
(total_students - yes_one = 230) :=
by
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end students_not_answering_yes_l593_593416


namespace angle_ACB_is_75_l593_593606

-- Define three points A, B, C in a triangle
variables {A B C P : Type}

-- Define angles and the point P according to given conditions
variables (h_angle_BAC : ∠ BAC = 45)
(h_P_trisect : ∃ P, trisects_side AC A P)
(h_angle_ABP : ∠ ABP = 15)

-- State what we need to prove: ∠ ACB = 75 degrees
theorem angle_ACB_is_75 
(h_angle_BAC :  ∠ BAC = 45)
(h_P_trisect : ∃ P, trisects_side AC A P)
(h_angle_ABP : ∠ ABP = 15) :
∠ ACB = 75 := 
sorry

end angle_ACB_is_75_l593_593606


namespace area_of_triangle_ABF_l593_593554

noncomputable def hyperbola (a b : ℝ) := {p : ℝ × ℝ | p.1^2 - p.2^2 / b^2 = 1}

namespace hyperbola_problem

def a : ℝ := 1
def b : ℝ := √3
def c: ℝ := √(a^2 + b^2)

def vertex_right : ℝ × ℝ := (a, 0)
def focus_right : ℝ × ℝ := (c, 0)

def asymptotes (a b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = b * p.1 ∨ p.2 = -b * p.1}

def line_BF : ℝ → ℝ := λ x, b * (x - 2)

def point_B : ℝ × ℝ := (1, -b)

def area_triangle (A B F : ℝ × ℝ) : ℝ := 0.5 * (abs (A.1 * (B.2 - F.2) + B.1 * (F.2 - A.2) + F.1 * (A.2 - B.2)))

theorem area_of_triangle_ABF :
  let A := vertex_right,
      F := focus_right,
      B := point_B
  in area_triangle A B F = √3 / 2 :=
by
  admit

end area_of_triangle_ABF_l593_593554


namespace coin_toss_sequences_l593_593473

noncomputable def count_sequences (n m : ℕ) : ℕ := Nat.choose (n + m - 1) (m - 1)

theorem coin_toss_sequences :
  ∃ (seq_count : ℕ), 
    seq_count = (count_sequences 3 3) * (count_sequences 6 4) ∧ seq_count = 840 :=
by
  use ((count_sequences 3 3) * (count_sequences 6 4))
  split
  { sorry, }
  { sorry, }

end coin_toss_sequences_l593_593473


namespace find_m_l593_593192

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l593_593192


namespace reinforcement_arrival_days_l593_593428

theorem reinforcement_arrival_days (x : ℕ) (h : x = 2000) (provisions_days : ℕ) (provisions_days_initial : provisions_days = 54) 
(reinforcement : ℕ) (reinforcement_val : reinforcement = 1300) (remaining_days : ℕ) (remaining_days_val : remaining_days = 20) 
(total_men : ℕ) (total_men_val : total_men = 3300) (equation : 2000 * (54 - x) = 3300 * 20) : x = 21 := 
by
  have eq1 : 2000 * 54 - 2000 * x = 3300 * 20 := by sorry
  have eq2 : 108000 - 2000 * x = 66000 := by sorry
  have eq3 : 2000 * x = 42000 := by sorry
  have eq4 : x = 21000 / 2000 := by sorry
  have eq5 : x = 21 := by sorry
  sorry

end reinforcement_arrival_days_l593_593428


namespace area_CMKN_l593_593638

open Real

-- Conditions
variables (A B C M N K : Point) (T : Triangle) (S : ℝ) 

-- Defining the points and conditions
def Triangle_ABC : Triangle := 
{
  A := A,
  B := B,
  C := C
}

def Points_Condition : Prop :=
  -- Points M and N on the sides
  (∃ k: ℝ, k = 1/4 ∧ col_linear ABC ∧ col_linear BCA) ∧
  -- Ratios CM:MB = 1:3 and AN:NC = 3:2
  (∃ {CM MB AN NC : ℝ}, CM = 1 ∧ MB = 3 ∧ AN = 3 ∧ NC = 2) ∧
  -- Intersection K of lines AM and BN
  (∃ K : Point, intersect Line(AM) Line(BN) K) ∧
  -- The area of triangle ABC is 1
  (Area Triangle_ABC T 1)

-- Goal
theorem area_CMKN {A B C M N K T : Point} (h: Points_Condition A B C M N K T) : 
  Area (Quadrilateral C M K N) 3/20 := 
sorry  -- Proof to be implemented

end area_CMKN_l593_593638


namespace base_conversion_l593_593666

theorem base_conversion (b : ℕ) (h_pos : b > 0) :
  (1 * 6 ^ 2 + 2 * 6 ^ 1 + 5 * 6 ^ 0 = 2 * b ^ 2 + 2 * b + 1) → b = 4 :=
by
  sorry

end base_conversion_l593_593666


namespace jan_bought_5_dozens_l593_593290

theorem jan_bought_5_dozens
  (cost_per_rose : ℕ)
  (discount_percent : ℚ)
  (amount_paid : ℚ)
  (one_dozen : ℕ) :
  (cost_per_rose = 6) →
  (discount_percent = 0.80) →
  (amount_paid = 288) →
  (one_dozen = 12) →
  (amount_paid / discount_percent / cost_per_rose / one_dozen = 5) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jan_bought_5_dozens_l593_593290


namespace quartic_polynomial_evaluation_l593_593767

noncomputable def p (n : ℕ) : ℚ 
def q (x : ℚ) := x^2 * p x - 1

theorem quartic_polynomial_evaluation : 
    (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5} → p n = 1 / (n:ℚ)^2) 
    → p 6 = -67 / 180 :=
begin
  sorry
end

end quartic_polynomial_evaluation_l593_593767


namespace extreme_value_when_a_zero_monotonicity_of_f_l593_593880

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + 1) * Real.exp x

theorem extreme_value_when_a_zero :
  (∃ x : ℝ, f 0 x = -(1 / Real.exp 2)) ∧ (∀ y : ℝ, f 0 y ≥ -(1 / Real.exp 2)) :=
sorry

theorem monotonicity_of_f (a : ℝ) :
  (a < 0 → (∀ x < -2, ∀ y > -2, f a x > f a y) ∧ (∀ x ∈ Ioo (-2 : ℝ) (-1/a : ℝ), f a x < f a y))
  ∧ (a = 0 → (∀ x < -2, f a x > f a x) ∧ (∀ y > -2, f a y > f a x))
  ∧ (0 < a ∧ a < 1/2 → (∀ x ∈ Ioo (-1/a : ℝ) (-2 : ℝ), f a x < f a y) ∧ (∀ x < -1/a, f a x > f a x) ∧ (∀ y > -2, f a y > f a x))
  ∧ (a = 1/2 → ∀ x < x', f a x ≤ f a x')
  ∧ (a > 1/2 → (∀ x ∈ Ioo (-2 : ℝ) (-1/a : ℝ), f a x < f a y) ∧ (∀ x < -2, f a x > f a x) ∧ (∀ y > -1/a, f a y > f a x)) :=
sorry

end extreme_value_when_a_zero_monotonicity_of_f_l593_593880


namespace sin_315_eq_neg_sqrt2_div_2_l593_593010

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593010


namespace choose_5_from_12_l593_593941

theorem choose_5_from_12 : Nat.choose 12 5 = 792 := by
  sorry

end choose_5_from_12_l593_593941


namespace perfect_squares_l593_593681

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593681


namespace distinct_pawn_arrangement_l593_593573

theorem distinct_pawn_arrangement :
  ∃ (f : Fin 5 → Fin 5), Bijective f ∧
  ∃ (g : Fin 5 → Fin 5), Bijective g ∧
  (∏ i : Fin 5, (5 - i)) * (∏ j : Fin 5, (5 - j)) = 14400 :=
by
  sorry

end distinct_pawn_arrangement_l593_593573


namespace sufficient_not_necessary_condition_l593_593486

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (a > b + 1) → (a > b) ∧ ¬((a > b) → (a > b + 1)) :=
by {
  intro h,
  split,
  { exact lt_of_lt_of_le h (by linarith) },
  {
    intro h',
    cases h' with contra,
    { contradiction },
    { exact λ x y, or.inr h }
  },
  sorry
}

end sufficient_not_necessary_condition_l593_593486


namespace volume_of_cylinder_l593_593652

-- Given conditions
def side_length_1 : ℝ := 4
def side_length_2 : ℝ := 2

-- Possible circumferences of the cylinder's base
def circumference_1 : ℝ := 4
def circumference_2 : ℝ := 2

theorem volume_of_cylinder :
  let R1 := circumference_1 / (2 * Real.pi), h1 := side_length_2,
      R2 := circumference_2 / (2 * Real.pi), h2 := side_length_1,
      V1 := Real.pi * R1 ^ 2 * h1, V2 := Real.pi * R2 ^ 2 * h2 in
  V1 = 8 / Real.pi ∨ V2 = 4 / Real.pi :=
by
  sorry

end volume_of_cylinder_l593_593652


namespace complex_eq_l593_593150

theorem complex_eq (z : ℂ) (h : 2 * complex.I / z = 1 - complex.I) : z = -1 + complex.I :=
  sorry

end complex_eq_l593_593150


namespace trajectory_M_l593_593436

variables (x y a b : ℝ)

def P_on_parabola (x y : ℝ) : Prop := y = x^2 + 1
def Q : (ℝ × ℝ) := (0, 1)
def M_is_midpoint (x y a b : ℝ) : Prop := x = 2 * a ∧ y = 2 * b - 1
def M_trajectory (a b : ℝ) : Prop := b = 2 * a^2 + 1

theorem trajectory_M (x y a b : ℝ) :
  P_on_parabola x y →
  M_is_midpoint x y a b →
  M_trajectory a b :=
by
  intros h1 h2
  obtain ⟨hx, hy⟩ := h2
  rw [hx, hy, h1] at *
  have : 2 * (2 * a^2 + 1 - 1) = (2 * a)^2 := sorry,
  exact this

end trajectory_M_l593_593436


namespace summable_divisible_100_l593_593647

theorem summable_divisible_100 (a : Fin 100 → ℕ) (h_diff : Function.Injective a) :
  (∃ i, i < 100 ∧ a i % 100 = 0) ∨ (∃ (s : Finset (Fin 100)), (∑ x in s, a x) % 100 = 0) :=
sorry

end summable_divisible_100_l593_593647


namespace solve_fraction_eq_l593_593656

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4 / 3 :=
by 
  sorry

end solve_fraction_eq_l593_593656


namespace coefficient_of_x21_l593_593955

def geometric_series_sum (n : ℕ) (x : ℝ) : ℝ :=
  (1 - x^(n+1)) / (1 - x)

theorem coefficient_of_x21 :
  ∑ k in Finset.range 21, (geometric_series_sum 20 x) * (geometric_series_sum 10 x)^3 = 932 :=
begin
  sorry
end

end coefficient_of_x21_l593_593955


namespace set_union_example_l593_593901

variable {α : Type*}

theorem set_union_example (a b : α) (A B : Set α)
  (hA : A = {1, 2 ^ a})
  (hB : B = {a, b})
  (h_inter : A ∩ B = {1 / 4}) :
  A ∪ B = {-2, 1, 1 / 4} :=
by
  sorry

end set_union_example_l593_593901


namespace cone_max_volume_height_l593_593378

theorem cone_max_volume_height (r h : ℝ) (s : ℝ) (h_eq : s = 18) (cone_eq : r^2 + h^2 = 18^2) :
  (h = 6 * sqrt 3) ∧ (∀ h', (r^2 + h'^2 = 18^2) → (V h' r ≤ V h r)) :=
by
  let V := λ h r, (1 / 3) * pi * r^2 * h
  sorry

end cone_max_volume_height_l593_593378


namespace sin_315_eq_neg_sqrt2_over_2_l593_593030

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593030


namespace problem_a_problem_b_l593_593396

def power2_chain (n : Nat) : Nat :=
  Nat.recOn n 1 (λ _ k, 2 ^ k)

def power3_chain (n : Nat) : Nat :=
  Nat.recOn n 1 (λ _ k, 3 ^ k)

def power4_chain (n : Nat) : Nat :=
  Nat.recOn n 1 (λ _ k, 4 ^ k)

theorem problem_a (n : Nat) (h : n ≥ 3) : power2_chain n < power3_chain (n-1) :=
  sorry

theorem problem_b (n : Nat) (h : n ≥ 2) : power3_chain n > 4 ^ power4_chain (n-1) :=
  sorry

end problem_a_problem_b_l593_593396


namespace factorial_division_l593_593812

theorem factorial_division :
  \dfrac{Nat.factorial 10}{Nat.factorial 7 * Nat.factorial 3} = 120 := 
sorry

end factorial_division_l593_593812


namespace volume_of_pyramid_l593_593771

variables (s h : ℝ)
-- Conditions
def pyramid_square_base (s : ℝ) : Prop := s > 0
def pyramid_total_surface_area (s : ℝ) : Prop := (s^2 + 4 * (s^2 / 3)) = 600
def triangular_face_area (s : ℝ) : Prop := (s^2 / 3) = (360 / 3)
def pyramid_height (s h : ℝ) : Prop := h = 4 * real.sqrt 10

-- Volume to prove
def pyramid_volume (s h : ℝ) : ℝ := 1/3 * (s^2) * real.sqrt (h^2 - (s/2)^2)

theorem volume_of_pyramid : 
  ∀ s h,
  pyramid_square_base s → pyramid_total_surface_area s → triangular_face_area s → pyramid_height s h → 
  pyramid_volume s h = 120 * real.sqrt 70 :=
begin
  intros s h h_base h_surface h_triangle h_height,
  sorry
end

end volume_of_pyramid_l593_593771


namespace find_a_b_sum_l593_593695

theorem find_a_b_sum
  (a b : ℝ)
  (h1 : 2 * a = -6)
  (h2 : a ^ 2 - b = 1) :
  a + b = 5 :=
by
  sorry

end find_a_b_sum_l593_593695


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593028

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593028


namespace find_m_l593_593189

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l593_593189


namespace sum_of_ages_l593_593347

theorem sum_of_ages 
  (a1 a2 a3 : ℕ) 
  (h1 : a1 ≠ a2) 
  (h2 : a1 ≠ a3) 
  (h3 : a2 ≠ a3) 
  (h4 : 1 ≤ a1 ∧ a1 ≤ 9) 
  (h5 : 1 ≤ a2 ∧ a2 ≤ 9) 
  (h6 : 1 ≤ a3 ∧ a3 ≤ 9) 
  (h7 : a1 * a2 = 18) 
  (h8 : a3 * min a1 a2 = 28) : 
  a1 + a2 + a3 = 18 := 
sorry

end sum_of_ages_l593_593347


namespace three_digit_diff_sum_of_digits_divisible_by_9_l593_593645

theorem three_digit_diff_sum_of_digits_divisible_by_9
  (a b c : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) (h_c : 0 ≤ c ∧ c ≤ 9) :
  ∃ k : ℤ, (100 * a + 10 * b + c - (a + b + c) = 9 * k) :=
begin
  sorry
end

end three_digit_diff_sum_of_digits_divisible_by_9_l593_593645


namespace find_planes_l593_593275

-- Define the conditions given in the problem
def plane_intersecting (n : ℕ) : Prop :=
  ∀ (k : ℕ), k = 1999 → ∀ i, 1 ≤ i < n-1 → ∃ (p : ℕ), p = 1999 ∧ (p ≠ k)

theorem find_planes (n : ℕ) :
  plane_intersecting n →
  n = 2000 ∨ n = 3998 :=
by
  sorry

end find_planes_l593_593275


namespace jane_stopped_babysitting_l593_593291

noncomputable def stopped_babysitting_years_ago := 12

-- Definitions for the problem conditions
def jane_age_started_babysitting := 20
def jane_current_age := 32
def oldest_child_current_age := 22

-- Final statement to prove the equivalence
theorem jane_stopped_babysitting : 
    ∃ (x : ℕ), 
    (jane_current_age - x = stopped_babysitting_years_ago) ∧
    (oldest_child_current_age - x ≤ 1/2 * (jane_current_age - x)) := 
sorry

end jane_stopped_babysitting_l593_593291


namespace baseball_team_selection_l593_593640

theorem baseball_team_selection : (nat.choose 14 5) = 2002 :=
by {
  -- This is where the proof would go
  sorry
}

end baseball_team_selection_l593_593640


namespace heartsuit_sum_l593_593117

def heartsuit (x : ℝ) : ℝ := (x + x^2) / 2

theorem heartsuit_sum : heartsuit 4 + heartsuit 5 + heartsuit 6 = 46 := by
  sorry

end heartsuit_sum_l593_593117


namespace find_f_neg3_l593_593207

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem find_f_neg3 : f (-3) = 21 := by
  sorry

end find_f_neg3_l593_593207


namespace part1_part2_l593_593326

open Real

-- Define propositions P and Q
def P (a x : ℝ) := a > 0 ∧ (x > a) ∧ (x < 3 * a)
def Q (x : ℝ) := (x - 3) / (x - 2) < 0

-- The first part of the proof problem
theorem part1 (x : ℝ) (a : ℝ) (ha : a = 1) (hpq : P a x ∧ Q x) : 2 < x ∧ x < 3 := by
  rw [ha] at *
  have hp := hpq.1
  have hq := hpq.2
  exact ⟨hq, hp.2⟩

-- The second part of the proof problem
theorem part2 : {a : ℝ | ∃ a, ¬P a → ¬Q (2 + a)} = set.Icc 1 2 := by
  sorry

end part1_part2_l593_593326


namespace not_rectangle_if_sides_equal_l593_593729

-- Definitions for conditions
def quadrilateral (α : Type) [EuclideanGeometry α] := Prop
def internal_angles_equal (q : quadrilateral α) := ∀ a b c d : angle, a = 90 ∧ b = 90 ∧ c = 90 ∧ d = 90
def sides_equal (q : quadrilateral α) := ∀ a b c d : side, a = b ∧ b = c ∧ c = d
def diagonals_equal_bisect (q : quadrilateral α) := ∀ d1 d2 : diagonal, (d1 = d2) ∧ (bisect d1 d2)
def parallelogram_equal_diagonals (q : quadrilateral α) (p : parallelogram α) := (∃ p, parallelogram q p) ∧ (diagonals_equal_bisect p)

-- Main theorem statement
theorem not_rectangle_if_sides_equal
  {α : Type} [EuclideanGeometry α] (q : quadrilateral α) :
  sides_equal q → ¬rectangle q := by sorry

end not_rectangle_if_sides_equal_l593_593729


namespace minimal_apples_l593_593783

theorem minimal_apples :
  ∃ (n : ℕ), 
    (∀ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
      p₁ + p₂ + p₃ + p₄ + p₅ = 100 ∧
      p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
      p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
      p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
      p₄ ≠ p₅ ∧
      (∃ (p₅ = max p₁ p₂ p₃ p₄ p₅),
        ∀ (k : ℕ), 
          (0 < p₁ / 100 * n ∧ 0 < p₂ / 100 * n ∧ 0 < p₃ / 100 * n ∧ 0 < p₄ / 100 * n) ∧
          (0 < p₁ / 100 * (n - p₅ * n / 100) ∧ 0 < p₂ / 100 * (n - p₅ * n / 100) ∧ 
           0 < p₃ / 100 * (n - p₅ * n / 100) ∧ 0 < p₄ / 100 * (n - p₅ * n / 100)) ∧
          k % (100 / lcm p₁ p₂ p₃ p₄) = 0 ∧
          p₂ = 15 ∧ p₃ = 20 ∧ p₄ = 25 ∧ p₅ = 30)
  → n = 20 :=
begin
  sorry
end

end minimal_apples_l593_593783


namespace sum_of_terms_l593_593848

variable (a : ℕ → ℤ)
variable (d : ℤ)

theorem sum_of_terms (h : ∑ i in finset.range 17, (a 1 + i * d) = 51) : 
  a 6 + a 10 = 6 :=
by sorry

end sum_of_terms_l593_593848


namespace magnitude_of_sum_of_perpendicular_vectors_l593_593221

theorem magnitude_of_sum_of_perpendicular_vectors :
  let a := (2 : ℝ, 1 : ℝ)
  let b := (x - 1, -x)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  ‖(2 + b.1, 1 + b.2)‖ = Real.sqrt 10 :=
by
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x - 1, -x)
  assume h : a.1 * b.1 + a.2 * b.2 = 0
  sorry

end magnitude_of_sum_of_perpendicular_vectors_l593_593221


namespace number_of_pens_l593_593367

theorem number_of_pens (x y : ℝ) (h1 : 60 * (x + 2 * y) = 50 * (x + 3 * y)) (h2 : x = 3 * y) : 
  (60 * (x + 2 * y)) / x = 100 :=
by
  sorry

end number_of_pens_l593_593367


namespace ordered_pairs_cardinality_l593_593832

theorem ordered_pairs_cardinality : ∃ (s : Set (ℝ × ℝ)), 
  (∀ p ∈ s, ∃ a b x y : ℤ, a * x + b * y = 1 ∧ x^2 + y^2 = 65) ∧ s.card = 128 := by
  sorry

end ordered_pairs_cardinality_l593_593832


namespace simplify_expression_l593_593485

theorem simplify_expression : 
  ∀ (a b c : ℤ), 
  a = (-5^2) → 
  b = (-5)^{11} → 
  c = (-5)^3 → 
  (a^4 * b) / c = 5^{16} :=
by
  intros a b c a_def b_def c_def
  sorry

end simplify_expression_l593_593485


namespace ball_arrangement_l593_593375

theorem ball_arrangement :
  (Nat.factorial 9) / ((Nat.factorial 2) * (Nat.factorial 3) * (Nat.factorial 4)) = 1260 := 
by
  sorry

end ball_arrangement_l593_593375


namespace sum_every_third_odd_integer_l593_593722

theorem sum_every_third_odd_integer (a₁ d n : ℕ) (S : ℕ) 
  (h₁ : a₁ = 201) 
  (h₂ : d = 6) 
  (h₃ : n = 50) 
  (h₄ : S = (n * (2 * a₁ + (n - 1) * d)) / 2) 
  (h₅ : a₁ + (n - 1) * d = 495) 
  : S = 17400 := 
  by sorry

end sum_every_third_odd_integer_l593_593722


namespace sin_315_eq_neg_sqrt2_over_2_l593_593036

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593036


namespace find_p_l593_593992

theorem find_p
  (p : ℝ)
  (h1 : ∃ (x y : ℝ), p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1)
  (h2 : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
         x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
         p * (x₁^2 - y₁^2) = (p^2 - 1) * x₁ * y₁ ∧ |x₁ - 1| + |y₁| = 1 ∧
         p * (x₂^2 - y₂^2) = (p^2 - 1) * x₂ * y₂ ∧ |x₂ - 1| + |y₂| = 1 ∧
         p * (x₃^2 - y₃^2) = (p^2 - 1) * x₃ * y₃ ∧ |x₃ - 1| + |y₃| = 1) :
  p = 1 ∨ p = -1 :=
by sorry

end find_p_l593_593992


namespace part1_part2_l593_593890

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l593_593890


namespace pow_rational_abs_condition_l593_593149

theorem pow_rational_abs_condition (a b : ℚ) (h : |a + 1| + |2013 - b| = 0) : a^b = -1 := 
by sorry

end pow_rational_abs_condition_l593_593149


namespace find_m_l593_593151

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l593_593151


namespace correct_rate_relationship_l593_593776

theorem correct_rate_relationship (f : ℕ → ℚ) (h1 : ∀ n, 1 ≤ n → n ≤ 10 → 0 ≤ f(n) ∧ f(n) ≤ 1) 
(h2 : ∀ m n, m < n → (m ≤ 10 ∧ n ≤ 10) → f(m)*(m:ℚ) ≤ f(n)*(n:ℚ)):
  ¬ (f 8 < f 9 ∧ f 9 = f 10) :=
sorry

end correct_rate_relationship_l593_593776


namespace sum_of_x_coordinates_l593_593826

theorem sum_of_x_coordinates (x : ℝ) (y : ℝ) :
    (y = abs (x^2 - 8*x + 12) ∧ y = 5 - x) → 
    Σ x, x = 16 := by
  sorry

end sum_of_x_coordinates_l593_593826


namespace other_diagonal_l593_593439

noncomputable def area (AB CD BD : ℝ) (h₁ h₂ : ℝ) := (1 / 2) * BD * (h₁ + h₂)

theorem other_diagonal (AB CD BD : ℝ) (h₁ h₂ : ℝ)
  (h1 : area AB CD BD h₁ h₂ = 32)
  (h2 : AB + CD + BD = 16) :
  ∃ AC : ℝ, AC = 8 * Real.sqrt 2 := 
begin
  sorry
end

end other_diagonal_l593_593439


namespace seconds_in_minutes_l593_593561

theorem seconds_in_minutes (minutes : ℝ) (conversion_factor : ℝ) : minutes = 7.8 → conversion_factor = 60 → (minutes * conversion_factor = 468) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end seconds_in_minutes_l593_593561


namespace problem_l593_593304

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := e / (2 * x) + Real.log x

def monotonic_intervals := 
  (∀ x, x > e / 2 → ∃ δ > 0, ∀ y, abs (y - x) < δ → f y > f x) ∧ 
  (∀ x, x > 0 ∧ x < e / 2 → ∃ δ > 0, ∀ y, abs (y - x) < δ → f y < f x)

def tangent_points (a b : ℝ) (x₁ x₂ x₃ : ℝ) := 
  (x₁ < x₂ ∧ x₂ < x₃ ∧ ∀ x ∈ {x₁, x₂, x₃}, x > 0 ∧ 
    (f x - b = ((2 * x - e) / (2 * x^2)) * (x - a)))

theorem problem (a b : ℝ) :
  monotonic_intervals ∧ 
  (a > e → 0 < b - f a ∧ b - f a < (a / (2 * e))) ∧
  (0 < a ∧ a < e → ∀ x₁ x₂ x₃, x₁ < x₂ ∧ x₂ < x₃ 
    → tangent_points a b x₁ x₂ x₃ 
    → (2 / e + (e - a) / (6 * e^2)) < (1 / x₁ + 1 / x₃) 
    ∧ (1 / x₁ + 1 / x₃) < (2 / a - (e - a) / (6 * e^2))) :=
by sorry

end problem_l593_593304


namespace midpoints_form_equilateral_triangle_l593_593518

-- Define the conditions of the problem
variables (A B C D O E F G : Type*)
variables [AffineSpace ℝ (Point A B C D)]
variables [Midpoint A B C D O E F G]

-- Conditions
variables (symmetric_trapezoid : Trapezoid A B C D) 
          (O : intersection (diagonal A C) (diagonal B D)) 
          (angle_OAB : ∠OAB = 60)

-- The definition of equilateral triangle
def equilateral_triangle (x y z : Point A B C D) : Prop :=
  (dist x y = dist y z) ∧ (dist y z = dist z x) ∧ (dist z x = dist x y)

-- Problem statement: Prove the midpoints form an equilateral triangle
theorem midpoints_form_equilateral_triangle :
  ∃ E F G : Point A B C D, 
    midpoint O A E ∧ midpoint O D F ∧ midpoint B C G ∧ equilateral_triangle E F G :=
by 
  sorry

end midpoints_form_equilateral_triangle_l593_593518


namespace number_of_girls_l593_593571

theorem number_of_girls (total_students boys girls : ℕ)
  (h1 : boys = 300)
  (h2 : (girls : ℝ) = 0.6 * total_students)
  (h3 : (boys : ℝ) = 0.4 * total_students) : 
  girls = 450 := by
  sorry

end number_of_girls_l593_593571


namespace total_students_in_all_halls_l593_593705

theorem total_students_in_all_halls :
  let G := 30 in
  let B := 2 * G in
  let C := G + 10 in
  let M := (3 / 5 : ℚ) * (G + B + C) in
  let A := (G * 5 : ℚ) in
  let Total := G + B + C + M + A in
  let P := (G + C) - 5 in
  let H := (3 / 4 : ℚ) * G in
  let L := H + 15 in
  let TotalAdditional := P + H + L in
  let GrandTotal := Total + TotalAdditional in
  GrandTotal = 484 := 
by {
  -- proof goes here
  sorry
}

end total_students_in_all_halls_l593_593705


namespace least_value_proof_l593_593271

noncomputable def least_value_4x_minus_x_plus_5 : ℚ :=
    have h1 : ∀ x : ℚ, (x > 10/3 ∧ x < 15/2) → 4*x - (x+5) ≥ 5 := by
        intros x hx
        rw sub_le_iff_le_add at hx
        exact le_of_lt (by linarith)
    5

-- Verify the condition ensures the minimum value
theorem least_value_proof (x : ℚ) (hx : x > 10/3 ∧ x < 15/2) : 4*x - (x+5) ≥ least_value_4x_minus_x_plus_5 := 
    by
    exact h1 x hx

end least_value_proof_l593_593271


namespace sin_315_eq_neg_sqrt2_div_2_l593_593055

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593055


namespace students_in_class_l593_593263

theorem students_in_class 
  (S : ℕ) 
  (hp : 0.80 * S = 0.80 * S) 
  (hp_and_parrot : 0.25 * (0.80 * S) = 8) : 
  S = 40 :=
sorry

end students_in_class_l593_593263


namespace standard_hyperbola_eq_asymptote_eq_constant_difference_l593_593215

variables {a b : ℝ} (b_pos : 0 < b)

-- Definition of the hyperbola C
def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

-- Standard equation of the hyperbola
theorem standard_hyperbola_eq : hyperbola_eq x y → (a = sqrt 2 ∧ b = sqrt 2) :=
by
  sorry

-- Asymptote equation of the hyperbola
theorem asymptote_eq : (∀ x y, hyperbola_eq x y → y = x ∨ y = -x) :=
by
  sorry

-- The constant difference term
theorem constant_difference {F1 F2 P A B : ℝ} (P_right : P > 2)
  (AF1 BF2 : P + F1 + A + F2 + B = C) :
  (abs (P - F1) / abs (A - F1)) - (abs (P - F2) / abs (B - F2)) = 6 :=
by 
  sorry

end standard_hyperbola_eq_asymptote_eq_constant_difference_l593_593215


namespace find_remainder_l593_593306

noncomputable def q (x : ℝ) : ℝ := (x^2010 + x^2009 + x^2008 + x + 1)
noncomputable def s (x : ℝ) := (q x) % (x^3 + 2*x^2 + 3*x + 1)

theorem find_remainder (x : ℝ) : (|s 2011| % 500) = 357 := by
    sorry

end find_remainder_l593_593306


namespace sin_315_degree_l593_593050

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593050


namespace sin_315_eq_neg_sqrt2_over_2_l593_593040

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593040


namespace total_distance_is_8_l593_593393

theorem total_distance_is_8 :
  let P := (2, 3)
  let Q := (5, 3)
  let R := (5, -2)
  dist P Q + dist Q R = 8 :=
by
  -- Definitions of points
  let P := (2, 3)
  let Q := (5, 3)
  let R := (5, -2)
  
  -- Use the distance formula
  have dPQ : dist P Q = 3 := sorry
  have dQR : dist Q R = 5 := sorry
  
  -- Add the distances
  show dPQ + dQR = 8 from sorry

end total_distance_is_8_l593_593393


namespace correct_probability_l593_593797

noncomputable def probability_black_region 
  (side_length : ℝ) 
  (tri_base : ℝ) 
  (tri_height : ℝ) 
  (diamond_side : ℝ) 
  (coin_diameter : ℝ) : ℝ :=
  let valid_region_area := (side_length - coin_diameter)^2
  let triangle_overlap_area := let quarter_circle_area := (π * (coin_diameter / 2)^2) / 4
                               let rectangle_area := tri_base * (coin_diameter / 2)
                               (tri_base * tri_height) / 2 + quarter_circle_area + rectangle_area
  let total_triangle_overlap_area := 4 * triangle_overlap_area
  let diamond_area := diamond_side^2
  let diamond_overlap_area := 4 * tri_base * (coin_diameter / 2) + π
  let total_black_area := total_triangle_overlap_area + diamond_area + diamond_overlap_area
  let probability := total_black_area / valid_region_area
  probability

theorem correct_probability : 
  probability_black_region 10 3 3 (3 * Real.sqrt 2) 2 = (48 + 4 * Real.sqrt 2 + 2 * π) / 64 := 
sorry

end correct_probability_l593_593797


namespace prime_factors_l593_593828

theorem prime_factors (x : ℕ) : 2 * x + 7 = 29 ↔ x = 11 := 
by
  split
  . intro h
    have : 2 * x = 22 := by linarith
    linarith
  . intro h
    rw h
    linarith

# Exit Lean 4 interactive mode with sorry
sorry

end prime_factors_l593_593828


namespace exists_two_color_line_l593_593429

theorem exists_two_color_line (grid : ℕ → ℕ → ℕ) (colors : fin 4 → ℕ) :
  (∀ (x y: ℕ), (x ≤ y) → ∃ (c1 c2 c3 c4 : colors),
    ∀ (i j : fin 2),
      grid (x + i.1) (y + j.1) ∈ {c1, c2, c3, c4} ∧
      c1 ≠ c2 ∧ c1 ≠ c3 ∧ c1 ≠ c4 ∧ c2 ≠ c3 ∧ c2 ≠ c4 ∧ c3 ≠ c4)
  → ∃ (line : ℕ → ℕ), (∃ (c1 c2: colors), ∀ (i : ℕ), 
      grid (line i).1 (line i).2 = c1 ∨ grid (line i).1 (line i).2 = c2) :=
by
  sorry

end exists_two_color_line_l593_593429


namespace find_circumradius_l593_593274

-- Given conditions
variables (b m : ℝ) (h a : ℝ)
def triangle_is_isosceles : Prop := True

-- Projection formula definition:
def projection_formula : Prop := a^2 = 2 * b * m

-- Pythagorean theorem in the right triangle formed by the altitude:
def pythagorean_theorem : Prop := h^2 = b^2 - (a / 2)^2

-- Express area Sₐᵇc:
def triangle_area (a h : ℝ) : ℝ := 0.5 * a * h

-- Formula for the circumradius R:
def circumradius (a b c S : ℝ) : ℝ := (a * b * c) / (4 * S)

-- Define the value to be proven equivalent:
def R := (b / 2) * sqrt (2 * b / (2 * b - m))

-- Combining all conditions together to state the final proof problem:
theorem find_circumradius :
  triangle_is_isosceles →
  projection_formula → 
  pythagorean_theorem →
  ∃ R, R = (b / 2) * sqrt (2 * b / (2 * b - m)) :=
by
  intro h1 h2 h3
  use R
  sorry

end find_circumradius_l593_593274


namespace compute_expression_l593_593795

theorem compute_expression :
  (3 + 3 / 8) ^ (2 / 3) - (5 + 4 / 9) ^ (1 / 2) + 0.008 ^ (2 / 3) / 0.02 ^ (1 / 2) * 0.32 ^ (1 / 2) / 0.0625 ^ (1 / 4) = 43 / 150 := 
sorry

end compute_expression_l593_593795


namespace sin_315_eq_neg_sqrt2_div_2_l593_593008

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593008


namespace otimes_self_twice_l593_593800

def otimes (x y : ℝ) := x^2 - y^2

theorem otimes_self_twice (a : ℝ) : (otimes (otimes a a) (otimes a a)) = 0 :=
  sorry

end otimes_self_twice_l593_593800


namespace find_m_find_range_l593_593208

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1/3) * x^3 - x^2 + m

theorem find_m :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, f x m ≤ 2/3) → f 0 m = 2/3 :=
by
  sorry

theorem find_range (hx : f 0 m = 2/3) :
  set.image (λ x => f x m) (set.Icc (-2 : ℝ) 2) = set.Icc (-6 : ℝ) (2/3) :=
by
  sorry

end find_m_find_range_l593_593208


namespace find_9b_l593_593917

variable (a b : ℚ)

theorem find_9b (h1 : 7 * a + 3 * b = 0) (h2 : a = b - 4) : 9 * b = 126 / 5 := 
by
  sorry

end find_9b_l593_593917


namespace find_a_plus_b_l593_593693

theorem find_a_plus_b (a b : ℝ) (h_sum : 2 * a = -6) (h_prod : a^2 - b = 1) : a + b = 5 :=
by {
  -- Proof would go here; we assume the theorem holds true.
  sorry
}

end find_a_plus_b_l593_593693


namespace max_chord_length_l593_593869

theorem max_chord_length :
  (∃ l, 
    ∀ θ : ℝ, 
      let f : ℝ → ℝ → ℝ := λ x y, 
        2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - 
        (8 * Real.sin θ + Real.cos θ + 1) * y in
    (∀ x : ℝ, f x (2 * x) = 0) → l = 8 * Real.sqrt 5) :=
begin
  sorry
end

end max_chord_length_l593_593869


namespace triangle_area_is_864_triangle_hypotenuse_is_60_l593_593939

-- Given conditions
variables {leg1 leg2 : ℝ} (h1 : leg1 = 36) (h2 : leg2 = 48)

-- Defining the area and hypotenuse for a right triangle
def area_of_triangle (leg1 leg2 : ℝ) : ℝ := (1 / 2) * leg1 * leg2

def hypotenuse_of_triangle (leg1 leg2 : ℝ) : ℝ := Real.sqrt (leg1^2 + leg2^2)

-- Theorem statements to prove
theorem triangle_area_is_864 (h1 : leg1 = 36) (h2 : leg2 = 48) :
  area_of_triangle leg1 leg2 = 864 :=
sorry

theorem triangle_hypotenuse_is_60 (h1 : leg1 = 36) (h2 : leg2 = 48) :
  hypotenuse_of_triangle leg1 leg2 = 60 :=
sorry

end triangle_area_is_864_triangle_hypotenuse_is_60_l593_593939


namespace range_of_k_for_lg_function_l593_593702

noncomputable def can_take_all_positive_real_values (k : ℝ) : Prop :=
  let t (x : ℝ) := x^2 + 3 * k * x + k^2 + 5
  ∃ x : ℝ, t x ≤ 0

theorem range_of_k_for_lg_function :
  (∀ k : ℝ, (can_take_all_positive_real_values k ↔ k ∈ Iio (-2) ∪ Ioi 2)) :=
by
  sorry

end range_of_k_for_lg_function_l593_593702


namespace valid_arrangements_count_l593_593945

theorem valid_arrangements_count (n : ℕ) (hn : 0 < n) : 
  ∑ k in finset.Icc 1 n, nat.choose (n-1) (k-1) = 2^(n-1) :=
by
  sorry

end valid_arrangements_count_l593_593945


namespace barbed_wire_height_l593_593348

theorem barbed_wire_height
  (area : ℝ)
  (cost_per_meter : ℝ)
  (gate_width : ℝ)
  (num_gates : ℝ)
  (total_cost : ℝ)
  (height : ℝ) :
  area = 3136 →
  cost_per_meter = 3.5 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 2331 →
  height = 3 :=
begin
  sorry
end

end barbed_wire_height_l593_593348


namespace find_a_b_find_k_range_l593_593893

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l593_593893


namespace arrangement_count_arrangement_count_with_conditions_l593_593934

section ChorusPerformance

-- Number of arrangements with 4 people per row, with no additional conditions
theorem arrangement_count (females : Fin 6) (lead_singer : Fin 1) (males : Fin 2)
  (rows : Fin 2) (people_per_row : Fin 4)
  : ∃ (n : ℕ), n = 8! := by sorry

-- Number of arrangements with the lead singer in the front row and male singers in the back row
theorem arrangement_count_with_conditions (females : Fin 6) (lead_singer : Fin 1) 
  (males : Fin 2) (rows : Fin 2) (people_per_row : Fin 4) 
  (front_row : Fin 4) (back_row : Fin 4) 
  : ∃ (n : ℕ), n = (Combinatorics.combination 5 3) * (4!) * (4!) := by sorry

end ChorusPerformance

end arrangement_count_arrangement_count_with_conditions_l593_593934


namespace angle_AM_equals_angle_BM_l593_593300

theorem angle_AM_equals_angle_BM
  (A B C D E M : Point)
  (isosceles_ABC : isosceles_triangle A B C)
  (midpoint_D : midpoint D B C)
  (angle_CDE_60 : angle C D E = 60)
  (midpoint_M : midpoint M D E) :
  angle A M E = angle B M D :=
sorry

end angle_AM_equals_angle_BM_l593_593300


namespace same_gender_probability_l593_593332

def SchoolA := {Teacher1, Teacher2, Teacher3}
def SchoolB := {Teacher4, Teacher5, Teacher6}
def maleA := {Teacher1, Teacher2}
def femaleA := {Teacher3}
def maleB := {Teacher4}
def femaleB := {Teacher5, Teacher6}

theorem same_gender_probability :
  let totalSelections := 9
  let favorableSelections := 4
  let probability := favorableSelections / totalSelections
  probability = 4 / 9 := by sorry

end same_gender_probability_l593_593332


namespace prove_f_neg_2_l593_593756

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f(-x) = -f(x)

theorem prove_f_neg_2 (f : ℝ → ℝ)
  (hf_odd : odd_function f)
  (hf_cond : ∀ x : ℝ, f(-x) = f(x + 3 / 2))
  (hf_value : f 2015 = 2) :
  f (-2) = -2 := 
sorry

end prove_f_neg_2_l593_593756


namespace scrap_cookie_radius_l593_593103

theorem scrap_cookie_radius (r: ℝ) (r_cookies: ℝ) (A_scrap: ℝ) (r_large: ℝ) (A_large: ℝ) (A_total_small: ℝ):
  r_cookies = 1.5 ∧
  r_large = r_cookies + 2 * r_cookies ∧
  A_large = π * r_large^2 ∧
  A_total_small = 8 * (π * r_cookies^2) ∧
  A_scrap = A_large - A_total_small ∧
  A_scrap = π * r^2
  → r = r_cookies
  :=
by
  intro h
  rcases h with ⟨hcookies, hrlarge, halarge, hatotalsmall, hascrap, hpi⟩
  sorry

end scrap_cookie_radius_l593_593103


namespace compute_floor_T_sq_l593_593625

noncomputable def T := ∑ i in Finset.range (2011) + 2, real.sqrt (1 + 1 / (i^2 : ℝ) + 1 / (i + 1)^2)

theorem compute_floor_T_sq : floor (T^2) = 4048142 :=
by
  sorry

end compute_floor_T_sq_l593_593625


namespace find_y_l593_593303

def diamond (a b : ℝ) : ℝ := (Real.sqrt (a + b + 3)) / (Real.sqrt (a - b + 1))

theorem find_y (y : ℝ) (h : diamond y 15 = 5) : 
  y = 46 / 3 :=
by
  sorry

end find_y_l593_593303


namespace log_equation_solution_l593_593830

theorem log_equation_solution :
  { x : ℝ | log 2 (x^3 - 20 * x^2 + 120 * x) = 7 } = {8, 4} :=
by
  -- skipping the proof
  sorry

end log_equation_solution_l593_593830


namespace perpendicular_planes_l593_593143

variables (l m : Line) (α β : Plane)
variables [line_subset_plane : l ⊆ α] [line_subset_plane' : m ⊆ β]

theorem perpendicular_planes (h : l ⊥ β) : α ⊥ β :=
sorry

end perpendicular_planes_l593_593143


namespace negation_proposition_l593_593363

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
sorry

end negation_proposition_l593_593363


namespace distance_from_focus_to_asymptote_l593_593527

-- Defining the hyperbola and its focus
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 3) - (y^2 / 3) = 1

def focus : ℝ × ℝ := (Real.sqrt 6, 0)

-- Defining the asymptote
def asymptote_eq (x y : ℝ) : Prop := y = x

-- Proof statement for the distance from the focus to one of the asymptotes
theorem distance_from_focus_to_asymptote :
  ∃ d : ℝ, d = Real.sqrt 3 ∧
  ∃ x y : ℝ, focus = (x, y) ∧ asymptote_eq x y :=
sorry

end distance_from_focus_to_asymptote_l593_593527


namespace max_value_of_a_l593_593358

noncomputable def f (x a : ℝ) : ℝ := exp x * (-x^2 + 2 * x + a)

def is_monotonically_increasing (f : ℝ → ℝ) (a : ℝ) : Prop := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ a + 1 → f x ≤ f y

theorem max_value_of_a :
  ∃ a : ℝ, is_monotonically_increasing (λ x => f x a) a ∧
  a = ( -1 + Real.sqrt 5 ) / 2 := 
sorry

end max_value_of_a_l593_593358


namespace sin_315_degree_l593_593043

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593043


namespace simplify_radical_expression_l593_593460

variable (q : ℝ)
variable (hq : q > 0)

theorem simplify_radical_expression :
  (sqrt(42 * q) * sqrt(7 * q) * sqrt(3 * q) = 21 * q * sqrt(2 * q)) :=
by
  sorry

end simplify_radical_expression_l593_593460


namespace baguettes_sold_after_first_batch_l593_593349

theorem baguettes_sold_after_first_batch:
  let batches_per_day := 3 in
  let baguettes_per_batch := 48 in
  let baguettes_sold_after_second_batch := 52 in
  let baguettes_sold_after_third_batch := 49 in
  let baguettes_left := 6 in
  (batches_per_day * baguettes_per_batch) - baguettes_left - (baguettes_sold_after_second_batch + baguettes_sold_after_third_batch) = 37 :=
by
  sorry

end baguettes_sold_after_first_batch_l593_593349


namespace triangle_area_ABC_l593_593471

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (7, 6)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Prove that the area of the triangle with the given vertices is 15
theorem triangle_area_ABC : triangle_area A B C = 15 :=
by
  -- Proof goes here
  sorry

end triangle_area_ABC_l593_593471


namespace find_xyz_l593_593526

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
  sorry

end find_xyz_l593_593526


namespace total_pet_food_weight_ounces_l593_593323

-- Define the conditions
def cat_food_bag_weight : ℕ := 3 -- each cat food bag weighs 3 pounds
def cat_food_bags : ℕ := 2 -- number of cat food bags
def dog_food_extra_weight : ℕ := 2 -- each dog food bag weighs 2 pounds more than each cat food bag
def dog_food_bags : ℕ := 2 -- number of dog food bags
def pounds_to_ounces : ℕ := 16 -- number of ounces in each pound

-- Calculate the total weight of pet food in ounces
theorem total_pet_food_weight_ounces :
  let total_cat_food_weight := cat_food_bags * cat_food_bag_weight,
      dog_food_bag_weight := cat_food_bag_weight + dog_food_extra_weight,
      total_dog_food_weight := dog_food_bags * dog_food_bag_weight,
      total_weight_pounds := total_cat_food_weight + total_dog_food_weight,
      total_weight_ounces := total_weight_pounds * pounds_to_ounces
  in total_weight_ounces = 256 :=
by 
  -- The proof is not required, so we leave it as sorry.
  sorry

end total_pet_food_weight_ounces_l593_593323


namespace sin_cos_product_l593_593564

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin x * Real.cos x = 2 / 5 := by
  sorry

end sin_cos_product_l593_593564


namespace quadrilateral_perimeter_l593_593763

-- Defining the projections condition
def orthogonal_projections_square (Q : Type) [Quadrilateral Q] (P1 P2 : Plane) : Prop :=
  square_projection Q P1 ∧ square_projection Q P2

-- Given conditions of the problem
def side_condition (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (A B : Point) (s₁ : ℝ), side_length Q A B s₁ ∧ s₁ = sqrt 5

-- The core proof statement
theorem quadrilateral_perimeter {Q : Type} [Quadrilateral Q] (P1 P2 : Plane) 
    (h_projections : orthogonal_projections_square Q P1 P2) 
    (h_side : side_condition Q) : 
  perimeter Q = 2 * (sqrt 5 + sqrt 7) :=
by
  sorry

end quadrilateral_perimeter_l593_593763


namespace sin_315_eq_neg_sqrt2_div_2_l593_593062

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593062


namespace sqrt_product_simplification_l593_593462

theorem sqrt_product_simplification (q : ℝ) : 
  sqrt (42 * q) * sqrt (7 * q) * sqrt (3 * q) = 126 * q * sqrt q := 
by
  sorry

end sqrt_product_simplification_l593_593462


namespace sin_315_eq_neg_sqrt2_div_2_l593_593080

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593080


namespace relationship_among_a_b_c_l593_593841

-- Definitions of a, b, c
def a : ℝ := (3 / 5) ^ (-1 / 3)
def b : ℝ := (3 / 5) ^ (-1 / 2)
def c : ℝ := (4 / 3) ^ (-1 / 2)

-- Proof statement
theorem relationship_among_a_b_c : c < a ∧ a < b :=
by {
  sorry
}

end relationship_among_a_b_c_l593_593841


namespace sum_sides_diagonals_12gon_l593_593441

theorem sum_sides_diagonals_12gon (a b c d : ℕ) (h : 
  (∃ (a b c d : ℕ), 
    ∑ (k : ℕ) in (range 2).map (λ x, x * 6), 
    if k = 1 then (12 * 24 * (sqrt (2 : ℝ) + sqrt 3 + sqrt 6)) else 12 * ∑ i in range 12, 
      if i = 1 then (12 + 12 * sqrt 2 + 12 * sqrt 3 + 12 * sqrt 6) else (24 * (sqrt 2 + sqrt 3 + sqrt 6))
  ) = (a, b, c, d)
  (sum_eq : 288 + 144 * sqrt 2 + 144 * sqrt 3 + 144 * sqrt 6 = a + b * sqrt 2 + c * sqrt 3 + d * sqrt 6)
) : a + b + c + d = 720 :=
sorry

end sum_sides_diagonals_12gon_l593_593441


namespace find_range_of_m_l593_593896

noncomputable def range_of_m (m : ℝ) : Prop :=
  (∃ p : Prop, p ↔ (∀ x : ℝ, x ≥ 2 → (x^2 - 2*m*x + 4) ≥ 0)) ∧
  (∃ q : Prop, q ↔ (∀ x : ℝ, m*x^2 + 2*(m-2)*x + 1 > 0)) ∧
  (p ∨ q) ∧ ¬(p ∧ q)

theorem find_range_of_m : range_of_m m ↔ (m ≤ 1) ∨ (2 < m ∧ m < 4) :=
sorry

end find_range_of_m_l593_593896


namespace part_l593_593983

noncomputable def f (a b x : ℝ) : ℝ := a^x - b * x + Real.exp 2

theorem part(Ⅲ)_a_equals_e (b : ℝ) (h1 : 1 < b) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₂ > (b * Real.log b / (2 * Real.exp 2)) * x₁ + (Real.exp 2 / b) := by
  sorry

end part_l593_593983


namespace find_base_l593_593951

theorem find_base (r : ℕ) : 
  (2 * r^2 + 1 * r + 0) + (2 * r^2 + 6 * r + 0) = 5 * r^2 + 0 * r + 0 → r = 7 :=
by
  sorry

end find_base_l593_593951


namespace plates_used_l593_593383

def plates_per_course : ℕ := 2
def courses_breakfast : ℕ := 2
def courses_lunch : ℕ := 2
def courses_dinner : ℕ := 3
def courses_late_snack : ℕ := 3
def courses_per_day : ℕ := courses_breakfast + courses_lunch + courses_dinner + courses_late_snack
def plates_per_day : ℕ := courses_per_day * plates_per_course

def parents_and_siblings_stay : ℕ := 6
def grandparents_stay : ℕ := 4
def cousins_stay : ℕ := 3

def parents_and_siblings_count : ℕ := 5
def grandparents_count : ℕ := 2
def cousins_count : ℕ := 4

def plates_parents_and_siblings : ℕ := parents_and_siblings_count * plates_per_day * parents_and_siblings_stay
def plates_grandparents : ℕ := grandparents_count * plates_per_day * grandparents_stay
def plates_cousins : ℕ := cousins_count * plates_per_day * cousins_stay

def total_plates_used : ℕ := plates_parents_and_siblings + plates_grandparents + plates_cousins

theorem plates_used (expected : ℕ) : total_plates_used = expected :=
by
  sorry

end plates_used_l593_593383


namespace cevian_ratio_identity_l593_593620

theorem cevian_ratio_identity
  {A B C P P_A P_B P_C : Type*}
  [Field A] [Field B] [Field C] [Field P] [Field P_A] [Field P_B] [Field P_C]
  (h1 : ∃ T, P = interior T ∧ T = △ A B C)
  (h2 : P_A = intersection (line_through A P) (segment B C))
  (h3 : P_B = intersection (line_through B P) (segment A C))
  (h4 : P_C = intersection (line_through C P) (segment A B)) :
  (segment_ratio P_B P P_B B + segment_ratio P_A P P_A A + segment_ratio P_C P P_C C) = 1 :=
begin
  -- proof omitted
  sorry
end

end cevian_ratio_identity_l593_593620


namespace proof_x_equals_90_l593_593691

open List

def data : List ℕ := [60, 100, x, 40, 50, 200, 90]

noncomputable def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / (l.length : ℚ)

theorem proof_x_equals_90 (x : ℕ) (h_mean : mean data = x) (h_median : data.nth 3 = some x) (h_mode : ∀ n ∈ data, n ≠ x → count n data ≤ 1) : x = 90 :=
by
  sorry

end proof_x_equals_90_l593_593691


namespace find_m_l593_593159

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l593_593159


namespace simplify_and_evaluate_expression_l593_593736

theorem simplify_and_evaluate_expression : 
  ∀ x : ℝ, x = 1 → ( (x^2 - 5) / (x - 3) - 4 / (x - 3) ) = 4 :=
by
  intros x hx
  simp [hx]
  have eq : (1 * 1 - 5) = -4 := by norm_num -- Verify that the expression simplifies correctly
  sorry -- Skip the actual complex proof steps

end simplify_and_evaluate_expression_l593_593736


namespace find_a_b_find_k_range_l593_593892

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l593_593892


namespace max_planes_15_points_l593_593482

noncomputable def max_planes (n : ℕ) : ℕ := n.choose 3

theorem max_planes_15_points (h : ∀ (points : finset (euclidean_space ℝ 3)), points.card = 15 → (∀ (s : finset (euclidean_space ℝ 3)), s ⊆ points → s.card = 4 → ¬ planar ℝ s)) : max_planes 15 = 455 :=
by
  sorry

end max_planes_15_points_l593_593482


namespace perfect_squares_l593_593682

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593682


namespace part1_part2_part3_l593_593282

noncomputable def triangle (A B C: Real) : Prop :=
  ∃ (a b c : Real), 
    A + B + C = π ∧
    cos (2 * A) - 3 * cos (B + C) = 1 ∧
    sin (2 * A) * sin (B + C) = 1 ∧
    a = b * sin B ∧
    b = c * sin C ∧
    a * sin A / (b * sin B) = 1 / 5

theorem part1 (A B C : Real) (a b c : Real) 
  (h₁ : cos (2 * A) - 3 * cos (B + C) = 1): A = π / 3 :=
sorry

theorem part2 (A B C : Real) (a b c : Real)
  (h₁ : a = 1) (h₂ : S = 5 * √3) (h₃ : b = 5)
  (h₄ : A = π / 3) : 
  sin B * sin C = 5 / 7 :=
sorry

theorem part3 (A B C : Real) (a b c : Real) 
  (h₁ : a = 1) (h₂ : A = π / 3) : 
  2 < a + b + c ∧ a + b + c ≤ 3 :=
sorry

end part1_part2_part3_l593_593282


namespace sum_of_squares_of_roots_l593_593115

theorem sum_of_squares_of_roots (r₁ r₂ : ℝ) 
  (h1 : r₁ + r₂ = 13) (h2 : r₁ * r₂ = 4) : 
  r₁^2 + r₂^2 = 161 := 
by
  calc r₁^2 + r₂^2
      = (r₁ + r₂)^2 - 2 * (r₁ * r₂) : by ring
  ... = 13^2 - 2 * 4 : by rw [h1, h2]
  ... = 169 - 8 : by norm_num
  ... = 161 : by norm_num

end sum_of_squares_of_roots_l593_593115


namespace borrowed_nickels_l593_593999

-- Define the initial and remaining number of nickels
def initial_nickels : ℕ := 87
def remaining_nickels : ℕ := 12

-- Prove that the number of nickels borrowed is 75
theorem borrowed_nickels : initial_nickels - remaining_nickels = 75 := by
  sorry

end borrowed_nickels_l593_593999


namespace min_value_5_sqrt_5_l593_593532

noncomputable def min_value (x : ℝ) (hx : 0 < x ∧ x < π / 2) : ℝ :=
  8 / Real.sin x + 1 / Real.cos x

theorem min_value_5_sqrt_5 :
  ∃ x ∈ Ioo 0 (π / 2), min_value x ⟨lt_of_lt_of_le zero_lt_one (le_of_lt $ half_lt_self (Real.pi_pos)), half_pos Real.pi_pos⟩ = 5 * Real.sqrt 5 :=
sorry

end min_value_5_sqrt_5_l593_593532


namespace inequality_2_inequality_1_9_l593_593308

variables {a : ℕ → ℝ}

-- Conditions
def non_negative (a : ℕ → ℝ) : Prop := ∀ n, a n ≥ 0
def boundary_zero (a : ℕ → ℝ) : Prop := a 1 = 0 ∧ a 9 = 0
def non_zero_interior (a : ℕ → ℝ) : Prop := ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i ≠ 0

-- Proof problems
theorem inequality_2 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 2 * a i := sorry

theorem inequality_1_9 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 1.9 * a i := sorry

end inequality_2_inequality_1_9_l593_593308


namespace central_angle_eq_one_l593_593860

noncomputable def radian_measure_of_sector (α r : ℝ) : Prop :=
  α * r = 2 ∧ (1 / 2) * α * r^2 = 2

-- Theorem stating the radian measure of the central angle is 1
theorem central_angle_eq_one (α r : ℝ) (h : radian_measure_of_sector α r) : α = 1 :=
by
  -- provide proof steps here
  sorry

end central_angle_eq_one_l593_593860


namespace sin_315_eq_neg_sqrt2_over_2_l593_593032

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593032


namespace four_letter_arrangements_l593_593907

theorem four_letter_arrangements : 
  ∃ n : ℕ, 
    (∀ (letters : list char), letters = ['A', 'B', 'C', 'D', 'E', 'F'] → 
    (∀ arr : list char, arr.length = 4 →
      list.nodup arr ∧ 
      arr.head = 'C' ∧ 
      'B' ∈ arr.tail → 
    n = 36)) := 
by 
  sorry

end four_letter_arrangements_l593_593907


namespace exists_spatial_quadrilateral_l593_593141

theorem exists_spatial_quadrilateral
    (n : ℕ) (l : ℕ) (q : ℕ)
    (h1 : n = q^2 + q + 1)
    (h2 : l ≥ 1 / 2 * q * (q +1)^2 + 1)
    (h3 : q ≥ 2)
    (h4 : ∀ (V : set (ℕ × ℕ)), ∀ A B C D ∈ V, ¬coplanar {A, B, C, D})
    (h5 : ∀ (V: set (ℕ × ℕ)), ∀ A ∈ V, (∃ B ∈ V, (A, B) ∈ V))
    (h6 : ∃ A ∈ (set.univ : set (ℕ × ℕ)), (∃k: ℕ, k ≥ q + 2 ∧ (A, k) ∈ (set.univ : set (ℕ × ℕ)))) :
    ∃ (A B C D : (ℕ × ℕ)), edge (A, B) ∧ edge (B, C) ∧ edge (C, D) ∧ edge (D, A) := 
sorry

end exists_spatial_quadrilateral_l593_593141


namespace find_theta_l593_593745

variables (m M : ℝ) (g θ : ℝ)
hypothesis h1 : M = 1.5 * m
hypothesis h2 : ∀ a, a = g / 3

theorem find_theta : θ = Real.arcsin (2 / 3) :=
sorry

end find_theta_l593_593745


namespace costOfBrantsRoyalBananaSplitSundae_l593_593398

-- Define constants for the prices of the known sundaes
def yvette_sundae_cost : ℝ := 9.00
def alicia_sundae_cost : ℝ := 7.50
def josh_sundae_cost : ℝ := 8.50

-- Define the tip percentage
def tip_percentage : ℝ := 0.20

-- Define the final bill amount
def final_bill : ℝ := 42.00

-- Calculate the total known sundaes cost
def total_known_sundaes_cost : ℝ := yvette_sundae_cost + alicia_sundae_cost + josh_sundae_cost

-- Define a proof to show that the cost of Brant's sundae is $10.00
theorem costOfBrantsRoyalBananaSplitSundae : 
  total_known_sundaes_cost + b = final_bill / (1 + tip_percentage) → b = 10 :=
sorry

end costOfBrantsRoyalBananaSplitSundae_l593_593398


namespace first_number_percentage_of_second_l593_593924

theorem first_number_percentage_of_second {X : ℝ} (H1 : ℝ) (H2 : ℝ) 
  (H1_def : H1 = 0.05 * X) (H2_def : H2 = 0.25 * X) : 
  (H1 / H2) * 100 = 20 :=
by
  sorry

end first_number_percentage_of_second_l593_593924


namespace paper_clips_in_2_cases_l593_593420

variable (c b : ℕ)

theorem paper_clips_in_2_cases : 2 * (c * b) * 600 = (2 * c * b * 600) := by
  sorry

end paper_clips_in_2_cases_l593_593420


namespace confidence_interval_for_mean_l593_593768

noncomputable def confidence_interval (samples : Fin 100 → ℝ) (sample_mean : ℝ) (variance : ℝ) (p : ℝ) : set ℝ :=
  {a : ℝ | (sample_mean - 0.233 < a) ∧ (a < sample_mean + 0.233)}

theorem confidence_interval_for_mean
  (a : ℝ)
  (ξ : ℝ → ℝ)
  (sample : Fin 100 → ℝ)
  (sample_mean : ℝ)
  (p : ℝ)
  (hξ : ∀ x, ξ x = Normal a 1)
  (h_sample_mean : sample_mean = 1.3)
  (hp : p = 0.98)
  (hsample_mean : (Finset.univ.sum (λ i, sample i)) / 100 = sample_mean) :
  confidence_interval sample sample_mean 1 p = { x | 1.067 < x ∧ x < 1.533 } :=
by
  sorry

end confidence_interval_for_mean_l593_593768


namespace f_periodicity_l593_593862

theorem f_periodicity 
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f(x + 5) = -f(x) + 2)
  (h2 : ∀ x ∈ Ioo 0 5, f(x) = x) : f 2018 = -1 :=
  sorry

end f_periodicity_l593_593862


namespace triangle_max_perimeter_is_3sqrt3_l593_593930

noncomputable def triangle_max_perimeter (a : ℝ) (A : ℝ) : ℝ :=
  if a = real.sqrt 3 ∧ A = real.pi / 3 then 3 * (real.sqrt 3) else 0

theorem triangle_max_perimeter_is_3sqrt3 :
  ∀ a A,
  a = real.sqrt 3 →
  A = real.pi / 3 →
  triangle_max_perimeter a A = 3 * (real.sqrt 3) :=
by
  intros a A ha hA
  unfold triangle_max_perimeter
  rw [if_pos (and.intro ha hA)]
  rfl

end triangle_max_perimeter_is_3sqrt3_l593_593930


namespace Ryan_learning_days_l593_593330

theorem Ryan_learning_days
  (hours_english_per_day : ℕ)
  (hours_chinese_per_day : ℕ)
  (total_hours : ℕ)
  (h1 : hours_english_per_day = 6)
  (h2 : hours_chinese_per_day = 7)
  (h3 : total_hours = 65) :
  total_hours / (hours_english_per_day + hours_chinese_per_day) = 5 := by
  sorry

end Ryan_learning_days_l593_593330


namespace average_salary_all_workers_l593_593665

-- Definitions based on the conditions
def technicians_avg_salary := 16000
def rest_avg_salary := 6000
def total_workers := 35
def technicians := 7
def rest_workers := total_workers - technicians

-- Prove that the average salary of all workers is 8000
theorem average_salary_all_workers :
  (technicians * technicians_avg_salary + rest_workers * rest_avg_salary) / total_workers = 8000 := by
  sorry

end average_salary_all_workers_l593_593665


namespace vector_calculation_l593_593316

variables (a b : ℝ × ℝ)

def a_def : Prop := a = (3, 5)
def b_def : Prop := b = (-2, 1)

theorem vector_calculation (h1 : a_def a) (h2 : b_def b) : a - 2 • b = (7, 3) :=
sorry

end vector_calculation_l593_593316


namespace geometric_seq_problem_l593_593201

open Finset

noncomputable def geometric_sequence {R : Type} [field R] (a r : R) : ℕ → R :=
λ n, a * r^n

theorem geometric_seq_problem {R : Type} [linear_ordered_field R] {a r : R}
  (h_pos : ∀ n, geometric_sequence a r n > 0)
  (h_eq : geometric_sequence a r 2 * geometric_sequence a r 4 +
          2 * geometric_sequence a r 3 * geometric_sequence a r 5 +
          geometric_sequence a r 4 * geometric_sequence a r 6 = 25) :
  geometric_sequence a r 3 + geometric_sequence a r 5 = 5 :=
sorry

end geometric_seq_problem_l593_593201


namespace parity_of_solutions_l593_593700

theorem parity_of_solutions
  (n m x y : ℤ)
  (hn : Odd n) 
  (hm : Odd m) 
  (h1 : x + 2 * y = n) 
  (h2 : 3 * x - y = m) :
  Odd x ∧ Even y :=
by
  sorry

end parity_of_solutions_l593_593700


namespace count_harmonic_vals_l593_593227

def floor (x : ℝ) : ℤ := sorry -- or use Mathlib function
def frac (x : ℝ) : ℝ := x - (floor x)

def is_harmonic_progression (a b c : ℝ) : Prop := 
  (1 / a) = (2 / b) - (1 / c)

theorem count_harmonic_vals :
  (∃ x, is_harmonic_progression x (floor x) (frac x)) ∧
  (∃! x1 x2, is_harmonic_progression x1 (floor x1) (frac x1) ∧
               is_harmonic_progression x2 (floor x2) (frac x2)) ∧
  x1 ≠ x2 :=
  sorry

end count_harmonic_vals_l593_593227


namespace quadrilateral_parallelogram_l593_593148

variables {ℝ ℝV : Type*} [NormedField ℝ] [NormedSpace ℝ ℝV]

theorem quadrilateral_parallelogram
  {O A B C D : ℝV}
  (h1 : \(\overrightarrow{AB}\) ∥ \(\overrightarrow{CD}\))
  (h2 : ∥\(\overrightarrow{OA}\) - \(\overrightarrow{OB}\)∥ = ∥\(\overrightarrow{OC}\) - \(\overrightarrow{OD}\)∥) :
  is_parallelogram O A B C D :=
begin
  sorry
end

end quadrilateral_parallelogram_l593_593148


namespace max_additional_spheres_l593_593427

noncomputable def frustumHeight := 8
noncomputable def sphereRadiusOne := 2
noncomputable def sphereRadiusTwo := 3

def canPlaceAdditionalSpheres : Nat := 2

theorem max_additional_spheres (frh : Nat) (r1 r2 : Nat) (max_spheres : Nat) : 
  frh = frustumHeight ∧ r1 = sphereRadiusOne ∧ r2 = sphereRadiusTwo → max_spheres = canPlaceAdditionalSpheres :=
by 
  intros h
  cases h with h_fr h_rest
  cases h_rest with h_r1 h_r2
  -- The proof is omitted as per instruction
  sorry

end max_additional_spheres_l593_593427


namespace volume_error_percentage_l593_593946

theorem volume_error_percentage (L B H : ℝ) :
  let V_actual := L * B * H in
  let L_erroneous := 1.08 * L in
  let B_erroneous := 0.95 * B in
  let H_erroneous := 0.90 * H in
  let V_erroneous := L_erroneous * B_erroneous * H_erroneous in
  let error_percentage := ((V_erroneous - V_actual) / V_actual) * 100 in
  error_percentage = -2.74 :=
by
  -- proof will go here
  sorry

end volume_error_percentage_l593_593946


namespace find_tangent_line_and_points_P_l593_593277

noncomputable def tangent_line_equation (k : ℝ) : Prop :=
  5 * 4 - 12 * 6 + 52 = 0 ∨ k = 4

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 16

def circle2 (x y : ℝ) : Prop := (x - 7)^2 + (y - 4)^2 = 4

def is_tangent_to_circle1 (x y : ℝ) : Prop :=
  tangent_line_equation x ∨ (x = 4 ∧ y = 6)

def perpendicular_chord_condition (P : ℝ × ℝ) : Prop :=
  let (a, b) := P in
  let c1_center := (0, 0) in
  let c2_center := (7, 4) in
  (∃ k : ℝ, (|b - a * k| / sqrt (1 + k^2) = 2 * |(7 + 4 * k - b * k - a)| / sqrt (1 + k^2))
   → (a, b) = (4, 6) ∨ (a, b) = (36 / 5, 2 / 5))

theorem find_tangent_line_and_points_P (a b k : ℝ) :
  circle1 4 6 ∧ circle2 a b ∧ (is_tangent_to_circle1 4 6) ∧ (perpendicular_chord_condition (a, b)) :=
  by
    sorry

end find_tangent_line_and_points_P_l593_593277


namespace inequality_holds_for_c_l593_593817
noncomputable def strictly_positive_real := {r : ℝ // r > 0}

theorem inequality_holds_for_c (x y z : strictly_positive_real) (c : strictly_positive_real)
  (hc : c.val ≤ 3^(-4/3)) : 
  (x.val^4 / ((y.val^2 + 1) * (z.val^2 + 1)) + 
   y.val^4 / ((x.val^2 + 1) * (z.val^2 + 1)) + 
   z.val^4 / ((x.val^2 + 1) * (y.val^2 + 1)) + 
   6 / (1 + c.val * (x.val * real.sqrt x.val + y.val * real.sqrt y.val + z.val * real.sqrt z.val)^(4 / 3))
  ) > 3 := 
sorry

end inequality_holds_for_c_l593_593817


namespace squares_equal_l593_593675

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l593_593675


namespace prob_of_multiples_l593_593928

-- Define the set of numbers
def S : Set ℕ := {4, 10, 20, 25, 40, 50, 100}

-- Define the predicate for a number being a multiple of 200
def is_multiple_of_200 (n : ℕ) : Prop := 200 ∣ n

-- Define the probability of two distinct members of the set having a product that is a multiple of 200
def prob_multiple_of_200 (s : Set ℕ) := let size := (s.to_finset.card.choose 2 : ℚ) in
  (s.to_finset.powerset.card.filter (fun t => t.card = 2 ∧ is_multiple_of_200 (t.to_finset.prod)) : ℚ) / size

-- Finally, the statement that needs to be proved
theorem prob_of_multiples :
  prob_multiple_of_200 S = 8 / 21 := 
sorry

end prob_of_multiples_l593_593928


namespace angle_A_120_l593_593965

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem angle_A_120 
  (h₁ : a^2 - b^2 = 3 * b * c)
  (h₂ : sin C = 2 * sin B) :
  A = 120 :=
sorry

end angle_A_120_l593_593965


namespace number_of_chickens_l593_593495

theorem number_of_chickens (c k : ℕ) (h1 : c + k = 120) (h2 : 2 * c + 4 * k = 350) : c = 65 :=
by sorry

end number_of_chickens_l593_593495


namespace sum_of_solutions_l593_593827

theorem sum_of_solutions :
  let solutions := {x : ℝ | 0 < x ∧ x - floor (x^2) = 1 / floor (x^2)} in
  let smallest_solutions := take 3 (data.finset.sort (≤) solutions) in
  sum smallest_solutions = 47 / 6 :=
by
  -- Definitions of intervals as identified in the solution steps:
  -- Interval 1: 1 ≤ x < √2
  have step1 : ∃ x, 1 ≤ x ∧ x < real.sqrt 2 ∧ x - 1 = 1 := by
    use 2
    simp [real.sqrt]
  -- Interval 2: √2 ≤ x < √3
  have step2 : ∃ x, real.sqrt 2 ≤ x ∧ x < real.sqrt 3 ∧ x - 2 = 0.5 := by
    use 2.5
    simp [real.sqrt]
  -- Interval 3: √3 ≤ x < √5
  have step3 : ∃ x, real.sqrt 3 ≤ x ∧ x < real.sqrt 5 ∧ x - 3 = 1/3 := by
    use 10/3
    simp [real.sqrt]
  -- Sum of solutions
  have correct_sum : (2 : ℝ) + 2.5 + (10 / 3) = 47 / 6 := by
    simp
  exact correct_sum

end sum_of_solutions_l593_593827


namespace max_value_a_l593_593725

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a = 3 :=
sorry

end max_value_a_l593_593725


namespace range_of_a_l593_593923

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 * Real.exp (-x1) = a) 
    ∧ (x2^2 * Real.exp (-x2) = a) ∧ (x3^2 * Real.exp (-x3) = a)) ↔ (0 < a ∧ a < 4 * Real.exp (-2)) :=
sorry

end range_of_a_l593_593923


namespace probability_of_event_l593_593635

noncomputable def drawing_probability : ℚ := 
  let total_outcomes := 81
  let successful_outcomes :=
    (9 + 9 + 9 + 9 + 9 + 7 + 5 + 3 + 1)
  successful_outcomes / total_outcomes

theorem probability_of_event :
  drawing_probability = 61 / 81 := 
by
  sorry

end probability_of_event_l593_593635


namespace equal_focal_distances_l593_593204

-- Define the first curve
def curve1 (m : ℝ) (h : m < 6) := ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (6 - m) = 1

-- Define the second curve
def curve2 (m : ℝ) (h : 5 < m ∧ m < 9) := ∀ (x y : ℝ), x^2 / (5 - m) + y^2 / (9 - m) = 1

-- Define the focal distance calculation for ellipse
def focal_distance_ellipse (m : ℝ) (h : m < 6) : ℝ :=
  let a_sq := 10 - m
  let b_sq := 6 - m
  Real.sqrt (a_sq - b_sq)

-- Define the focal distance calculation for hyperbola
def focal_distance_hyperbola (m : ℝ) (h : 5 < m ∧ m < 9) : ℝ :=
  let a_sq := 9 - m
  let b_sq := 5 - m
  Real.sqrt (a_sq + b_sq)

-- Theorem statement proving equal focal distances
theorem equal_focal_distances (m₁ m₂ : ℝ) (h₁ : m₁ < 6) (h₂ : 5 < m₂ ∧ m₂ < 9) :
  focal_distance_ellipse m₁ h₁ = focal_distance_hyperbola m₂ h₂ :=
by
  sorry

end equal_focal_distances_l593_593204


namespace range_of_m_l593_593870

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -a * Real.log x + x + (1 - a) / x

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := Real.exp x + m * x^2 - 2 * Real.exp 2 - 3

theorem range_of_m {a : ℝ} (a_eq : a = Real.exp 2 + 1) :
  (∀ x1 ∈ Set.Ici (1:ℝ), ∃ x2 ∈ Set.Ici (1:ℝ), g x2 m ≤ f x1 a) ↔ m ∈ Iic (Real.exp 2 - Real.exp 1) := 
sorry

end range_of_m_l593_593870


namespace necessary_but_not_sufficient_l593_593842

def p (a : ℝ) : Prop := (a - 1) * (a - 2) = 0
def q (a : ℝ) : Prop := a = 1

theorem necessary_but_not_sufficient (a : ℝ) : 
  (q a → p a) ∧ (p a → q a → False) :=
by
  sorry

end necessary_but_not_sufficient_l593_593842


namespace maria_dozen_flowers_l593_593834

theorem maria_dozen_flowers (x : ℕ) (h : 12 * x + 2 * x = 42) : x = 3 :=
by
  sorry

end maria_dozen_flowers_l593_593834


namespace CQ_length_l593_593975

-- Definitions of the geometrical setup
variables (A B C D E F P Q : Type)
variables [IsMidpoint E A B]
variables [IsMidpoint F C D]
variables [Collinear A P Q]
variables (EP PF QF : ℝ)
variables [h_EP : EP = 5]
variables [h_PF : PF = 3]
variables [h_QF : QF = 12]

-- Main theorem to prove
theorem CQ_length : ∃ (CQ : ℝ), CQ = 8 :=
by 
  sorry

end CQ_length_l593_593975


namespace sqrt_product_simplification_l593_593467

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end sqrt_product_simplification_l593_593467


namespace find_y_l593_593566

theorem find_y (x y : ℚ) (h1 : x = 151) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 342200) : 
  y = 342200 / 3354151 :=
by
  sorry

end find_y_l593_593566


namespace inequality_proof_l593_593989

variable (a b c : ℝ)

theorem inequality_proof (a b c : ℝ) :
    a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤
    1 + (1 / 3) * (a + b + c) ^ 2 :=
by
  sorry

end inequality_proof_l593_593989


namespace solve_inequality_l593_593506

theorem solve_inequality (x : ℝ) : 
  (x ≠ 1) → ( (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ) ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) := 
sorry

end solve_inequality_l593_593506


namespace polynomial_nonnegative_l593_593574

theorem polynomial_nonnegative (p q : ℝ) (h : q > p^2) :
  ∀ x : ℝ, x^2 + 2 * p * x + q ≥ 0 :=
by
  intro x
  have h2 : x^2 + 2 * p * x + q = (x + p)^2 + (q - p^2) := by sorry
  have h3 : (x + p)^2 ≥ 0 := by sorry
  have h4 : q - p^2 > 0 := h
  have h5 : (x + p)^2 + (q - p^2) ≥ 0 + 0 := by sorry
  linarith

end polynomial_nonnegative_l593_593574


namespace area_of_transformed_triangle_l593_593344

variables {x1 x2 x3 : ℝ} (f : ℝ → ℝ)

def original_area (x1 x2 x3 : ℝ) (f : ℝ → ℝ) : ℝ := 27 -- Given area of the original triangle

def transformed_points (x : ℝ) (f : ℝ → ℝ) : ℝ × ℝ :=
  (x/3, 3 * f x)

-- Prove that the area of the transformed triangle is also 27
theorem area_of_transformed_triangle (hx : function.support f = {x1, x2, x3})
  (area_original : original_area x1 x2 x3 f = 27) :
  let p1 := transformed_points x1 f,
      p2 := transformed_points x2 f,
      p3 := transformed_points x3 f in
  area_of_triangle p1 p2 p3 = 27 :=
sorry

end area_of_transformed_triangle_l593_593344


namespace product_of_primes_sum_101_l593_593371

theorem product_of_primes_sum_101 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 := by
  sorry

end product_of_primes_sum_101_l593_593371


namespace perfect_squares_l593_593679

theorem perfect_squares (a b c : ℤ)
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593679


namespace true_proposition_among_choices_l593_593920

theorem true_proposition_among_choices (p q : Prop) (hp : p) (hq : ¬ q) :
  p ∧ ¬ q :=
by
  sorry

end true_proposition_among_choices_l593_593920


namespace hexagon_perimeter_p_q_r_sum_l593_593432

def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def hexagon_points : List (ℝ × ℝ) := [(0, 0), (2, 1), (3, 3), (2, 4), (0, 3), (1, 1), (0, 0)]

def perimeter (points : List (ℝ × ℝ)) : ℝ :=
  (List.zipWith distance points (points.tail ++ [points.head])).sum

theorem hexagon_perimeter :
  let p := 0
  let q := 2
  let r := 4
  perimeter hexagon_points = p + q * real.sqrt 2 + r * real.sqrt 5 :=
by
  sorry

theorem p_q_r_sum :
  let p := 0
  let q := 2
  let r := 4
  p + q + r = 6 :=
by
  rfl

end hexagon_perimeter_p_q_r_sum_l593_593432


namespace value_of_f1_l593_593357

noncomputable def f (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 3

theorem value_of_f1 (m : ℝ) (h_increasing : ∀ x : ℝ, x ≥ -2 → 2 * x^2 - m * x + 3 ≤ 2 * (x + 1)^2 - m * (x + 1) + 3)
  (h_decreasing : ∀ x : ℝ, x ≤ -2 → 2 * (x - 1)^2 - m * (x - 1) + 3 ≤ 2 * x^2 - m * x + 3) : 
  f 1 (-8) = 13 := 
sorry

end value_of_f1_l593_593357


namespace sqrt_product_simplification_l593_593463

theorem sqrt_product_simplification (q : ℝ) : 
  sqrt (42 * q) * sqrt (7 * q) * sqrt (3 * q) = 126 * q * sqrt q := 
by
  sorry

end sqrt_product_simplification_l593_593463


namespace pavlosum_associative_l593_593642

def pavlosum (x y : ℝ) : ℝ := (x + y) / (1 - x * y)

theorem pavlosum_associative (a b c d : ℝ) :
  ((pavlosum (pavlosum a b) c) (pavlosum -d) = pavlosum a (pavlosum b (pavlosum c -d))) :=
sorry

end pavlosum_associative_l593_593642


namespace perfect_squares_count_in_range_l593_593238

theorem perfect_squares_count_in_range :
  let s := {x : ℕ | 50 ≤ x ∧ x ≤ 250 ∧ ∃ (n : ℕ), n^2 = x}
  ∈ (∃ (count : ℕ), count = 8) :=
by
  let s := {x : ℕ | 50 ≤ x ∧ x ≤ 250 ∧ ∃ (n : ℕ), n^2 = x}
  sorry

end perfect_squares_count_in_range_l593_593238


namespace coloring_theorem_l593_593713

noncomputable def chessboard := Fin 4 × Fin 19

def color := {r : chessboard → Fin 3 // function.surjective r}

theorem coloring_theorem (c : color) :
  ∃ (r1 r2 : Fin 4) (c1 c2 : Fin 19),
    r1 ≠ r2 ∧ c1 ≠ c2 ∧
    c.1 (r1, c1) = c.1 (r1, c2) ∧
    c.1 (r1, c2) = c.1 (r2, c1) ∧
    c.1 (r2, c1) = c.1 (r2, c2) :=
sorry

end coloring_theorem_l593_593713


namespace mathematicians_overlap_probability_l593_593712

noncomputable def probability_cont_keep_alive : ℝ := 0.9775

theorem mathematicians_overlap_probability :
  let a := ℝ;
  let b := ℝ;
  probability_cont_keep_alive = 0.9775 := 
begin
  sorry
end

end mathematicians_overlap_probability_l593_593712


namespace problem_seven_integers_l593_593334

theorem problem_seven_integers (a b c d e f g : ℕ) 
  (h1 : b = a + 1) 
  (h2 : c = b + 1) 
  (h3 : d = c + 1) 
  (h4 : e = d + 1) 
  (h5 : f = e + 1) 
  (h6 : g = f + 1) 
  (h_sum : a + b + c + d + e + f + g = 2017) : 
  a = 286 ∨ g = 286 :=
sorry

end problem_seven_integers_l593_593334


namespace f4_is_odd_l593_593488

noncomputable def f1 (x : ℝ) := 3 * x^3 + 2 * x^2 + 1
noncomputable def f2 (x : ℝ) := x^(-1 / 2)
noncomputable def f3 (x : ℝ) := 3^x
noncomputable def f4 (x : ℝ) := sqrt (4 - x^2) / (abs (x + 3) - 3)

theorem f4_is_odd : ∀ x : ℝ, f4 (-x) = -f4 (x) := by
  sorry

end f4_is_odd_l593_593488


namespace angle_A_120_l593_593964

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem angle_A_120 
  (h₁ : a^2 - b^2 = 3 * b * c)
  (h₂ : sin C = 2 * sin B) :
  A = 120 :=
sorry

end angle_A_120_l593_593964


namespace existence_of_points_l593_593279

theorem existence_of_points
  (k1 k2 : Circle)
  (X Y : Point) 
  (hX : X ∈ k1 ∧ X ∈ k2) 
  (hY : Y ∈ k1 ∧ Y ∈ k2) :
  ∃ (A B C D : Point), 
    (∀ k3 : Circle, 
      (A ∈ k3 ∧ B ∈ k3 ∧ isTangent k3 k1 A ∧ isTangent k3 k2 B ∧ meetsLine k3 (Line.mk X Y) C D) →
        passesThrough (Line.mk A C) P ∧
        passesThrough (Line.mk A D) P ∧
        passesThrough (Line.mk B C) P ∧
        passesThrough (Line.mk B D) P) :=
sorry

end existence_of_points_l593_593279


namespace math_problem_l593_593731

open Real

-- Define the logarithm properties
def log100 (x : ℝ) : ℝ := log x / log 100
def log_base_ab (x a b : ℝ) : ℝ := log x / log (a * b)

-- Assumptions
variables (a b : ℝ)
variable (H1 : log100 a = log a / 2)
variable (H2 : log100 b = log b / 2)
variable (H3 : log 100 = 2)
variable (H4 : ∀ x y, log (x * y) = log x + log y)

theorem math_problem :
  (b^(log100 a / log a) * a^(log100 b / log b))^(2 * log_base_ab (a + b) a b) = a + b :=
sorry

end math_problem_l593_593731


namespace remove_wallpaper_time_l593_593811

theorem remove_wallpaper_time :
  let dining_room_time := 3 * 1.5
  let living_room_time := 2 * 1 + 2 * 2.5
  let bedroom_time := 3 * 3
  let hallway_time := 1 * 4 + 4 * 2
  dining_room_time + living_room_time + bedroom_time + hallway_time = 32.5 :=
by {
  -- Definitions for each room's remaining wallpaper removal time
  let dining_room_time : ℝ := 3 * 1.5
  let living_room_time : ℝ := 2 * 1 + 2 * 2.5
  let bedroom_time : ℝ := 3 * 3
  let hallway_time : ℝ := 1 * 4 + 4 * 2

  -- Calculation of the total time
  let total_time : ℝ := dining_room_time + living_room_time + bedroom_time + hallway_time

  -- Assertion of the total time
  have : total_time = 32.5, 
  { sorry },
  exact this
}

end remove_wallpaper_time_l593_593811


namespace max_ab_of_tangent_circles_l593_593557

noncomputable def circle1 (a : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 - a)^2 + (p.2 + 2)^2 = 4}

noncomputable def circle2 (b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 + b)^2 + (p.2 + 2)^2 = 1}

theorem max_ab_of_tangent_circles : ∀ (a b : ℝ),
  (∃ p, circle1 a p ∧ circle2 b p) →
  (∃ d, d = (a + b)) →
  (∃ r1 r2, r1 = 2 ∧ r2 = 1 ∧ d = r1 - r2) →
  ab_max := 1 / 4 :=
begin
  sorry
end

end max_ab_of_tangent_circles_l593_593557


namespace mutual_correlation_l593_593737

noncomputable theory

variable {X : ℝ → ℝ} -- Random function X(t)
variable {τ : ℝ} -- Time difference τ
variable (r_x_dot_x : ℝ → ℝ) -- Cross-correlation function r_{x \dot{x}}(τ)
variable (r_dot_x_x : ℝ → ℝ) -- Cross-correlation function r_{\dot{x} x}(τ)
variable (k_x : ℝ → ℝ) -- Correlation function k_x(τ)
variable (k_x' : ℝ → ℝ) -- Derivative of the correlation function k_x(τ)

-- Definition of the derivative of X(t)
variable (X' : ℝ → ℝ)
axiom derivative_X : ∀ t, X' t = derivative X t

-- Conditions
axiom stationary : ∀ t1 t2, k_x (t2 - t1) = Expect[λ ω, X t1 ω * X t2 ω]
axiom cross_correlation_1 : r_x_dot_x τ = Expect[λ ω, X 0 ω * X' τ ω]
axiom cross_correlation_2 : r_dot_x_x τ = Expect[λ ω, X' 0 ω * X τ ω]

-- Theorem
theorem mutual_correlation :
  (r_x_dot_x τ = k_x' τ) ∧ (r_dot_x_x τ = - k_x' τ) :=
sorry

end mutual_correlation_l593_593737


namespace sum_k_eq_10000_l593_593310

def k (a : ℕ) : ℕ :=
  ((finset.range(a+1)).product (finset.range(a+1))).filter (λ p, nat.coprime p.1 p.2).card

theorem sum_k_eq_10000 :
  ∑ i in finset.range(100 + 1).filter (λ i, i > 0), k (100 / i) = 10000 :=
by
  sorry

end sum_k_eq_10000_l593_593310


namespace find_m_l593_593185

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l593_593185


namespace sin_315_degree_l593_593042

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593042


namespace circle_equation_and_line_conditions_l593_593845

-- Definitions from given conditions
def circle_center : Type := { coord : ℝ × ℝ // coord.2 = 0 }
def point_on_circle (C : circle_center) : Prop := (C.1.1 - 0)^2 + (C.1.2 - sqrt 3)^2 = (abs (C.1.1 + 1))^2
def tangent_condition (C : circle_center) : Prop := abs (C.1.1 + 1) = sqrt (C.1.1 - 0)^2 + (0 - sqrt 3)^2

def line_l (x : ℝ) : Type := {eq : ℝ // eq = ℝ × ℝ }
def chord_length_condition (x : ℝ) : Prop := x = (0,-2) and sqrt ((3 * x - -4 * 2)^2 + 8) = 2 * sqrt 3

-- Theorem to be proved
theorem circle_equation_and_line_conditions :
  ∃ C : circle_center, point_on_circle C ∧ tangent_condition C ∧
  (∃ l : ℝ, line_l l ∧ chord_length_condition l)

end circle_equation_and_line_conditions_l593_593845


namespace equation_one_solution_equation_two_solution_l593_593340

theorem equation_one_solution :
  (∀ x : ℝ, x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2) := by
  intro x
  split
  { intro h
    have factored := eq_comm.mp (mul_eq_zero.mp (eq_sub_eq_add_neg.mp h.symm)).symm
    cases factored
    { left, exact factored }
    { right, exact eq_add_of_sub_eq factored } }
  { intro h
    cases h
    { rw h, simp }
    { rw h, ring }}

theorem equation_two_solution :
  (∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := by
  intro x
  split
  { intro h
    have move_const := (sub_eq_zero.mp (eq_sub_eq_add_neg.mp h.symm)).sybase_of_eq go, or. ssolve
    {
      n_exact x - 2,
      ring },
    {
      lift (x - 2) to sqrt 3 with h,
      right, l_shift go },
    { stir add go llgo rsltcsym },
    { ex_meta, mda shift },
or rewrite sqrt,
apply sub_eq },
  eq_of_sqrt_eq_left,
ring sorry


end equation_one_solution_equation_two_solution_l593_593340


namespace increasing_function_id_l593_593107

theorem increasing_function_id
  (f : ℕ+ → ℕ+)
  (h1 : ∀ {x y : ℕ+}, x < y → f x < f y)
  (h2 : ∀ (x : ℕ+), f x * f (f x) ≤ x * x) :
  ∀ (x : ℕ+), f x = x := by
  sorry

end increasing_function_id_l593_593107


namespace tg_half_angle_inequality_l593_593409

variable (α β γ : ℝ)

theorem tg_half_angle_inequality 
  (h : α + β + γ = 180) : 
  (Real.tan (α / 2)) * (Real.tan (β / 2)) * (Real.tan (γ / 2)) ≤ (Real.sqrt 3) / 9 := 
sorry

end tg_half_angle_inequality_l593_593409


namespace perfect_squares_l593_593684

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593684


namespace part_I_part_II_l593_593142

-- Part (I)
theorem part_I (a₁ : ℝ) (d : ℝ) (S : ℕ → ℝ) (k : ℕ) :
  a₁ = 3 / 2 →
  d = 1 →
  (∀ n, S n = (n / 2 : ℝ) * (n + 2)) →
  S (k ^ 2) = S k ^ 2 →
  k = 4 :=
by
  intros ha₁ hd hSn hSeq
  sorry

-- Part (II)
theorem part_II (a : ℝ) (d : ℝ) (S : ℕ → ℝ) :
  (∀ k : ℕ, S (k ^ 2) = (S k) ^ 2) →
  ( (∀ n, a = 0 ∧ d = 0 ∧ a + d * (n - 1) = 0) ∨
    (∀ n, a = 1 ∧ d = 0 ∧ a + d * (n - 1) = 1) ∨
    (∀ n, a = 1 ∧ d = 2 ∧ a + d * (n - 1) = 2 * n - 1) ) :=
by
  intros hSeq
  sorry

end part_I_part_II_l593_593142


namespace sin_315_eq_neg_sqrt2_over_2_l593_593034

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593034


namespace sin_315_eq_neg_sqrt2_div_2_l593_593061

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593061


namespace part1_part2_l593_593904

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b : ℝ × ℝ := (3, -Real.sqrt 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) b

theorem part1 (hx : x ∈ Set.Icc 0 Real.pi) (h_perp : dot_product (a x) b = 0) : x = 5 * Real.pi / 6 :=
sorry

theorem part2 (hx : x ∈ Set.Icc 0 Real.pi) :
  (f x ≤ 2 * Real.sqrt 3) ∧ (f x = 2 * Real.sqrt 3 → x = 0) ∧
  (f x ≥ -2 * Real.sqrt 3) ∧ (f x = -2 * Real.sqrt 3 → x = 5 * Real.pi / 6) :=
sorry

end part1_part2_l593_593904


namespace log_value_bounds_l593_593374

theorem log_value_bounds : ∃ (c d : ℤ), c + d = 5 ∧ c < real.log 678 / real.log 10 ∧ real.log 678 / real.log 10 < d :=
by
  use [2, 3]
  sorry

end log_value_bounds_l593_593374


namespace cosine_lt_sine_neg_four_l593_593132

theorem cosine_lt_sine_neg_four : ∀ (m n : ℝ), m = Real.cos (-4) → n = Real.sin (-4) → m < n :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end cosine_lt_sine_neg_four_l593_593132


namespace sin_315_eq_neg_sqrt2_div_2_l593_593004

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593004


namespace sin_315_eq_neg_sqrt2_div_2_l593_593059

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593059


namespace quadratic_properties_l593_593764

theorem quadratic_properties (d e f : ℝ)
  (h1 : d * 1^2 + e * 1 + f = 3)
  (h2 : d * 2^2 + e * 2 + f = 0)
  (h3 : d * 9 + e * 3 + f = -3) :
  d + e + 2 * f = 19.5 :=
sorry

end quadratic_properties_l593_593764


namespace csc_7pi_over_6_l593_593504

theorem csc_7pi_over_6 : Real.csc (7 * Real.pi / 6) = -2 :=
by
  sorry

end csc_7pi_over_6_l593_593504


namespace nodes_on_parabola_in_unit_square_l593_593701

noncomputable def unit_square_subdivision_side_length := (2 : ℝ) * 10 ^ (-4)

theorem nodes_on_parabola_in_unit_square : ∃ n : ℕ, n = 49 ∧
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 4999 ∧ 1 ≤ j ∧ j ≤ 4999 →
  (j = (i^2 / 5000)) →
  (n = 49) :=
sorry

end nodes_on_parabola_in_unit_square_l593_593701


namespace composite_quadratic_l593_593868

theorem composite_quadratic (m n : ℤ) (x1 x2 : ℤ)
  (h1 : 2 * x1^2 + m * x1 + 2 - n = 0)
  (h2 : 2 * x2^2 + m * x2 + 2 - n = 0)
  (h3 : x1 ≠ 0) 
  (h4 : x2 ≠ 0) :
  ∃ (k : ℕ), ∃ (l : ℕ), 
    (k > 1) ∧ (l > 1) ∧ (k * l = (m^2 + n^2) / 4) := sorry

end composite_quadratic_l593_593868


namespace solve_quadratic_equation_l593_593829

theorem solve_quadratic_equation (x : ℝ) : 4 * (x - 1)^2 = 36 ↔ (x = 4 ∨ x = -2) :=
by sorry

end solve_quadratic_equation_l593_593829


namespace quad_area_EFGH_l593_593947

-- Definitions for given conditions
def angle_E : ℝ := 150
def angle_G : ℝ := 150
def length_EF : ℝ := 2
def length_FG : ℝ := 2 * Real.sqrt 2
def length_GH : ℝ := 4

-- Theorem to prove the area of the quadrilateral EFGH is 3 * sqrt 2
theorem quad_area_EFGH : 
  let A_EFG := (1 / 2) * length_EF * length_FG * Real.sin (angle_E * Real.pi / 180)
  let A_FGH := (1 / 2) * length_FG * length_GH * Real.sin (angle_G * Real.pi / 180)
  A_EFG + A_FGH = 3 * Real.sqrt 2 := by
  sorry

end quad_area_EFGH_l593_593947


namespace cars_pass_undelay_l593_593837

-- Define the angles at each starting point A_i
def angles : List ℝ := [
    60, 30, 15, 20, 155, 45, 10, 35, 140, 50,
    125, 65, 85, 86, 80, 75, 78, 115, 95, 25,
    28, 158, 30, 25, 5, 15, 160, 170, 20, 158
]

-- Define a condition that checks if a car at A_i will pass through all intersections
def passesAllIntersections (i : ℕ) : Prop :=
  ∀ j, j < i → angles.nthLe j (by linarith) < angles.nthLe i (by linarith)

-- Prove that cars starting from points A_14, A_23, and A_24 will pass through all intersections
theorem cars_pass_undelay : passesAllIntersections 13 ∧ passesAllIntersections 22 ∧ passesAllIntersections 23 :=
  by
    sorry

end cars_pass_undelay_l593_593837


namespace find_lambda_of_parallel_vectors_l593_593224

theorem find_lambda_of_parallel_vectors (λ : ℝ) 
  (a : ℝ × ℝ) (ha : a = (-3, 2)) (b : ℝ × ℝ) (hb : b = (-1, λ)) 
  (h_parallel : ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2) : λ = 2 / 3 :=
by
  sorry

end find_lambda_of_parallel_vectors_l593_593224


namespace sequence_closed_form_l593_593913

theorem sequence_closed_form (x : ℕ → ℝ) : 
  x 0 = 1 → 
  x 1 = 1 → 
  (∀ n ≥ 1, x (n + 1) = (x n)^2 / (x (n - 1) + 2 * x n)) → 
  ∀ n, x n = 1 / ((2 * n - 1)!!) :=
by
  intros h0 h1 h_rec
  sorry

end sequence_closed_form_l593_593913


namespace determine_complex_number_l593_593278

noncomputable def symmetric_complex_number : ℂ :=
  let w := 2 / (complex.I - 1)
  let wz := complex.conj w
  wz

theorem determine_complex_number (z : ℂ) (h : z = symmetric_complex_number) : z = -1 + complex.I :=
by { simp [symmetric_complex_number] at h, exact h }


end determine_complex_number_l593_593278


namespace kaleb_gave_boxes_l593_593297

theorem kaleb_gave_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (given_boxes : ℕ)
  (h1 : total_boxes = 14) 
  (h2 : pieces_per_box = 6) 
  (h3 : pieces_left = 54) :
  given_boxes = 5 :=
by
  -- Add your proof here
  sorry

end kaleb_gave_boxes_l593_593297


namespace zero_sum_neg_l593_593544

noncomputable def f (a x : ℝ) : ℝ := (x - 1) * exp x + a * x ^ 2

theorem zero_sum_neg (a : ℝ) (h : a > 0) (x1 x2 : ℝ) (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) : x1 + x2 < 0 :=
sorry

end zero_sum_neg_l593_593544


namespace rowing_speed_in_still_water_l593_593760

theorem rowing_speed_in_still_water (speed_of_current : ℝ) (time_seconds : ℝ) (distance_meters : ℝ) (S : ℝ)
  (h_current : speed_of_current = 3) 
  (h_time : time_seconds = 9.390553103577801) 
  (h_distance : distance_meters = 60) 
  (h_S : S = 20) : 
  (distance_meters / 1000) / (time_seconds / 3600) - speed_of_current = S :=
by 
  sorry

end rowing_speed_in_still_water_l593_593760


namespace distance_to_focus_l593_593353

theorem distance_to_focus (d : ℝ) (a : ℝ) (b : ℝ) :
  a = 2 ∧ b = 1 ∧ |12 - d| = 4 → d = 8 ∨ d = 16 :=
by
  intro h
  cases h with ha hb
  cases hb with hb1 hb2
  sorry

end distance_to_focus_l593_593353


namespace pages_read_today_l593_593912

theorem pages_read_today (pages_yesterday pages_total : ℕ) (h1 : pages_yesterday = 21) (h2 : pages_total = 38) :
  pages_total - pages_yesterday = 17 :=
by
  rw [h1, h2]
  sorry

end pages_read_today_l593_593912


namespace part_one_part_two_l593_593519

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 2

theorem part_one (a : ℝ) :
  (∀ x ≤ 0, f x a ≤ f x a) ∧ (∀ x ≥ 0, f x a ≥ f x a) ↔ a = 1 :=
sorry

theorem part_two :
  (∀ x : ℝ, g x ≤ f x 1) :=
sorry

end part_one_part_two_l593_593519


namespace problem_statement_l593_593847

variable {a : ℕ → ℝ} 
variable {a1 d : ℝ}
variable (h_arith : ∀ n, a (n + 1) = a n + d)  -- Arithmetic sequence condition
variable (h_d_nonzero : d ≠ 0)  -- d ≠ 0
variable (h_a1_nonzero : a1 ≠ 0)  -- a1 ≠ 0
variable (h_geom : (a 1) * (a 7) = (a 3) ^ 2)  -- Geometric sequence condition a2 = a 1, a4 = a 3, a8 = a 7

theorem problem_statement :
  (a 0 + a 4 + a 8) / (a 1 + a 2) = 3 :=
by
  sorry

end problem_statement_l593_593847


namespace volume_is_correct_l593_593807

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

noncomputable def volume_enclosed : ℝ :=
  18

theorem volume_is_correct :
  (λ (x y z : ℝ), f x y z ≤ 6) ⟹ volume_enclosed = 18 :=
by
  sorry

end volume_is_correct_l593_593807


namespace find_a_b_sum_l593_593696

theorem find_a_b_sum
  (a b : ℝ)
  (h1 : 2 * a = -6)
  (h2 : a ^ 2 - b = 1) :
  a + b = 5 :=
by
  sorry

end find_a_b_sum_l593_593696


namespace ratio_lateral_to_total_surface_area_l593_593366

def base_radius (r : ℝ) := r
def height (r : ℝ) := 2 * r

def lateral_area (r : ℝ) (l : ℝ) := 2 * Math.pi * r * l
def total_surface_area (r : ℝ) (l : ℝ) := 2 * Math.pi * r^2 + 2 * Math.pi * r * l

theorem ratio_lateral_to_total_surface_area (r : ℝ) : 
  let l := height r in
  (lateral_area r l) / (total_surface_area r l) = 2 / 3 :=
by
  sorry

end ratio_lateral_to_total_surface_area_l593_593366


namespace largest_prime_factor_sum_divisors_450_l593_593978

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.divisors n).sum

theorem largest_prime_factor_sum_divisors_450 :
  let M := sum_of_divisors 450 in
  M = 1209 ∧ Nat.greatestPrimeFactor M = 13 := by
sorry

end largest_prime_factor_sum_divisors_450_l593_593978


namespace asymptote_slopes_of_hyperbola_l593_593112

theorem asymptote_slopes_of_hyperbola :
  ∀ x y : ℝ,
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1 →
  (y = 1 + 4 / 3 * x + 8 / 3) ∨ (y = 1 - 4 / 3 * x + 8 / 3) →
  (abs (4 / 3)) = 4 / 3 :=
begin
  sorry,
end

end asymptote_slopes_of_hyperbola_l593_593112


namespace sqrt_product_simplification_l593_593465

theorem sqrt_product_simplification (q : ℝ) : 
  sqrt (42 * q) * sqrt (7 * q) * sqrt (3 * q) = 126 * q * sqrt q := 
by
  sorry

end sqrt_product_simplification_l593_593465


namespace trajectory_of_P_l593_593861
-- Import entire library for necessary definitions and theorems.

-- Define the properties of the conic sections.
def ellipse (x y : ℝ) (n : ℝ) : Prop :=
  x^2 / 4 + y^2 / n = 1

def hyperbola (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / 8 - y^2 / m = 1

-- Define the condition where the conic sections share the same foci.
def shared_foci (n m : ℝ) : Prop :=
  4 - n = 8 + m

-- The main theorem stating the relationship between m and n forming a straight line.
theorem trajectory_of_P : ∀ (n m : ℝ), shared_foci n m → (m + n + 4 = 0) :=
by
  intros n m h
  sorry

end trajectory_of_P_l593_593861


namespace lower_limit_b_l593_593576

theorem lower_limit_b (a b : ℤ) (h1 : 6 < a) (h2 : a < 17) (h3 : b < 29) 
  (h4 : ∃ min_b max_b, min_b = 4 ∧ max_b ≤ 29 ∧ 3.75 = (16 : ℚ) / (min_b : ℚ) - (7 : ℚ) / (max_b : ℚ)) : 
  b ≥ 4 :=
sorry

end lower_limit_b_l593_593576


namespace find_m_l593_593157

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l593_593157


namespace correct_answer_is_B_l593_593122

-- Definitions
def equal_internal_angles_inscribed_pentagon (P : Type) [metric_space P] [normed_group P] [inner_product_space ℝ P] (pentagon : finset P) : Prop :=
  ∃ A B C D E : P, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E ∧
    ∀ (x y : P), x ∈ pentagon → y ∈ pentagon → x ≠ y → ∠ x y x = ∠ y x y

def equal_internal_angles_inscribed_quadrilateral (Q : Type) [metric_space Q] [normed_group Q] [inner_product_space ℝ Q] (quadrilateral : finset Q) : Prop :=
  ∃ A B C D : Q, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ∀ (x y : Q), x ∈ quadrilateral → y ∈ quadrilateral → x ≠ y → ∠ x y x = ∠ y x y

-- Proof statement
theorem correct_answer_is_B :
  ∀ (P Q : Type) [metric_space P] [normed_group P] [inner_product_space ℝ P]
  [metric_space Q] [normed_group Q] [inner_product_space ℝ Q], 
  (equal_internal_angles_inscribed_pentagon P (finset.univ : finset P) → ∃ R : Type, regular_pentagon R) ∧
  (equal_internal_angles_inscribed_quadrilateral Q (finset.univ : finset Q) → ∀ R : Type, ¬ regular_quadrilateral R) :=
sorry

end correct_answer_is_B_l593_593122


namespace find_a_l593_593874

open Real

def f (x a : ℝ) : ℝ := x^2 - a * x - a

theorem find_a : (∃ a : ℝ, (∀ x ∈ Icc (0 : ℝ) (2 : ℝ), f x a ≤ 1) ∧ (f 0 a = -a) ∧ (f 2 a = 4 - 3 * a) ∧ (f 2 a = 1) ∧ (-a < 1)) ↔ a = 1 := by
  sorry

end find_a_l593_593874


namespace repeating_decimal_as_fraction_l593_593500

-- Define repeating decimal 0.7(3) as x
def x := 0.7 + 3 / 10 ^ (2 + n) where n is some natural number

theorem repeating_decimal_as_fraction :
    x = 11 / 15 := sorry

end repeating_decimal_as_fraction_l593_593500


namespace sin_315_eq_neg_sqrt2_div_2_l593_593014

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593014


namespace game_outcome_with_2033_coins_game_outcome_with_2034_coins_l593_593385

-- Definitions of the game rules, players, and winning conditions.
def winning_position (alice bob : ℕ → Prop) : ℕ → Prop :=
  λ n, alice n ∨ bob n

def alice_turn (n : ℕ) : Prop :=
  (n >= 2 ∧ winning_position (λ k, k = n - 2) (λ k, k = n - 2)) ∨
  (n >= 5 ∧ winning_position (λ k, k = n - 5) (λ k, k = n - 5))

def bob_turn (n : ℕ) : Prop :=
  (n >= 1 ∧ winning_position (λ k, k = n - 1) (λ k, k = n - 1)) ∨
  (n >= 3 ∧ winning_position (λ k, k = n - 3) (λ k, k = n - 3)) ∨
  (n >= 4 ∧ winning_position (λ k, k = n - 4) (λ k, k = n - 4))

-- Statement of the proof problem
theorem game_outcome_with_2033_coins :
  ¬ alice_turn 2033 → bob_turn 2033 := sorry

theorem game_outcome_with_2034_coins :
  alice_turn 2034 → ¬ bob_turn 2034 := sorry

end game_outcome_with_2033_coins_game_outcome_with_2034_coins_l593_593385


namespace non_congruent_isosceles_right_triangles_with_equal_inradius_and_area_l593_593909

theorem non_congruent_isosceles_right_triangles_with_equal_inradius_and_area :
  ∃! (a : ℝ), (a > 0) →
    let A := (1/2) * a^2,
        r := (a * (2 - Real.sqrt 2)) / 4
    in A = r :=
by
  sorry

end non_congruent_isosceles_right_triangles_with_equal_inradius_and_area_l593_593909


namespace triangle_is_right_l593_593272
-- necessary imports

-- definitions used from the condition
variables {ABC : Triangle}
variable (r : ℝ) -- radius of its excribed circle
variable (P : ℝ) -- perimeter of the triangle
variable (exradii_half_perimeter : r = P / 2)


-- the theorem statement
theorem triangle_is_right (exradii_half_perimeter : r = P / 2) : is_right_triangle ABC :=
sorry

end triangle_is_right_l593_593272


namespace find_area_shaded_l593_593651

noncomputable def area_shaded_region (AD CD : ℝ) : ℝ :=
  let AC := real.sqrt (AD^2 + CD^2) in
  let r := AC in
  let quarter_circle_area := (real.pi * r^2) / 4 in
  let rectangle_area := AD * CD in
  quarter_circle_area - rectangle_area

theorem find_area_shaded (AD CD : ℝ) (hAD : AD = 5) (hCD : CD = 12) :
  70 ≤ area_shaded_region AD CD ∧ area_shaded_region AD CD ≤ 74 := 
by
  sorry

end find_area_shaded_l593_593651


namespace sin_315_eq_neg_sqrt2_div_2_l593_593065

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593065


namespace sin_315_eq_neg_sqrt2_div_2_l593_593079

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593079


namespace sin_315_eq_neg_sqrt2_div_2_l593_593066

theorem sin_315_eq_neg_sqrt2_div_2 :
  sin (315 : ℝ) * π / 180 = - (Real.sqrt 2 / 2) := sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593066


namespace sin_315_eq_neg_sqrt2_div_2_l593_593075

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593075


namespace find_m_l593_593182

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l593_593182


namespace grape_juice_percentage_l593_593936

-- Define the conditions
def initial_volume : ℝ := 30
def initial_grape_juice_percent : ℝ := 0.10
def added_grape_juice : ℝ := 10

-- Define the final percentage of grape juice to prove
def final_grape_juice_percent : ℝ := 32.5

-- Statement to be proved in Lean
theorem grape_juice_percentage :
  let initial_grape_juice := initial_volume * initial_grape_juice_percent in
  let total_grape_juice := initial_grape_juice + added_grape_juice in
  let final_volume := initial_volume + added_grape_juice in
  (total_grape_juice / final_volume) * 100 = final_grape_juice_percent :=
by
  sorry

end grape_juice_percentage_l593_593936


namespace independence_of_events_l593_593919

noncomputable def is_independent (A B : Prop) (chi_squared : ℝ) := 
  chi_squared ≤ 3.841

theorem independence_of_events (A B : Prop) (chi_squared : ℝ) : 
  is_independent A B chi_squared → A ↔ B :=
by
  sorry

end independence_of_events_l593_593919


namespace current_value_l593_593592

-- Definitions of given values and relationships
def V : ℂ := 2 + 2 * Complex.i
def Z : ℂ := 3 - 4 * Complex.i
def I : ℂ := V / Z

theorem current_value :
  I = -2/25 + (14/25) * Complex.i := 
sorry

end current_value_l593_593592


namespace initial_milk_amounts_l593_593333

theorem initial_milk_amounts :
  let m1 := 6/42
  let m2 := 5/42
  let m3 := 4/42
  let m4 := 3/42
  let m5 := 2/42
  let m6 := 1/42
  let m7 := 0
  (m1 + m2 + m3 + m4 + m5 + m6 + m7 = 1/2) ∧ 
  (∀ i, distribute(i, [m1, m2, m3, m4, m5, m6, m7]) = [m1, m2, m3, m4, m5, m6, m7]) :=
by
  sorry

-- Helper function to simulate distribution, adjusted to fit mathematical principles
def distribute (i : ℕ) (mugs : List ℚ) : List ℚ :=
  sorry

end initial_milk_amounts_l593_593333


namespace wheel_travel_distance_l593_593448

theorem wheel_travel_distance (r : ℝ) (n : ℝ) : 
  r = 2 → n = 1.25 → let C := 2 * Real.pi * r in d = n * C → d = 5 * Real.pi := 
by 
  intros hr hn hC
  rw [←hC, hr, hn]
  sorry

end wheel_travel_distance_l593_593448


namespace bold_o_lit_cells_l593_593779

-- Define the conditions
def grid_size : ℕ := 5
def original_o_lit_cells : ℕ := 12 -- Number of cells lit in the original 'o'
def additional_lit_cells : ℕ := 12 -- Additional cells lit in the bold 'o'

-- Define the property to be proved
theorem bold_o_lit_cells : (original_o_lit_cells + additional_lit_cells) = 24 :=
by
  -- computation skipped
  sorry

end bold_o_lit_cells_l593_593779


namespace negation_of_proposition_l593_593692

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ ∃ x : ℝ, Real.exp x ≤ x^2 :=
by sorry

end negation_of_proposition_l593_593692


namespace parabola_coeff_sum_l593_593419

def parabola_vertex_form (a b c : ℚ) : Prop :=
  (∀ y : ℚ, y = 2 → (-3) = a * (y - 2)^2 + b * (y - 2) + c) ∧
  (∀ x y : ℚ, x = 1 ∧ y = -1 → x = a * y^2 + b * y + c) ∧
  (a < 0)  -- Since the parabola opens to the left, implying the coefficient 'a' is positive.

theorem parabola_coeff_sum (a b c : ℚ) :
  parabola_vertex_form a b c → a + b + c = -23 / 9 :=
by
  sorry

end parabola_coeff_sum_l593_593419


namespace problem_am_gm_inequality_l593_593988

theorem problem_am_gm_inequality
  (a b c : ℝ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_sq : a^2 + b^2 + c^2 = 3) : 
  (1 / (1 + a * b)) + (1 / (1 + b * c)) + (1 / (1 + c * a)) ≥ 3 / 2 :=
by
  sorry

end problem_am_gm_inequality_l593_593988


namespace simplify_radical_expression_l593_593459

variable (q : ℝ)
variable (hq : q > 0)

theorem simplify_radical_expression :
  (sqrt(42 * q) * sqrt(7 * q) * sqrt(3 * q) = 21 * q * sqrt(2 * q)) :=
by
  sorry

end simplify_radical_expression_l593_593459


namespace modulus_of_z2_plus_2i_l593_593198

noncomputable def z1 : ℂ := -1 + complex.i
noncomputable def z2 : ℂ := -2 / z1

theorem modulus_of_z2_plus_2i :
  complex.abs (z2 + 2 * complex.i) = real.sqrt 10 := 
sorry

end modulus_of_z2_plus_2i_l593_593198


namespace sin_315_degree_l593_593053

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593053


namespace length_QF_l593_593216

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def directrix (x y : ℝ) : Prop := x = 1 -- Directrix of the given parabola

def point_on_directrix (P : ℝ × ℝ) : Prop := directrix P.1 P.2

def point_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

def point_on_line_PF (P F Q : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ (Q.2 = m * (Q.1 - F.1) + F.2) ∧ point_on_parabola Q

def vector_equality (F P Q : ℝ × ℝ) : Prop :=
  (4 * (Q.1 - F.1), 4 * (Q.2 - F.2)) = (P.1 - F.1, P.2 - F.2)

theorem length_QF 
  (P Q : ℝ × ℝ)
  (hPd : point_on_directrix P)
  (hPQ : point_on_line_PF P focus Q)
  (hVec : vector_equality focus P Q) : 
  dist Q focus = 3 :=
by
  sorry

end length_QF_l593_593216


namespace cos_theta_value_l593_593203

def terminalSidePassesThrough (P : ℝ × ℝ) (θ : ℝ) : Prop :=
  P = (-12, 5)

theorem cos_theta_value (θ : ℝ) (h : terminalSidePassesThrough (-12, 5) θ) :
  cos θ = -12 / 13 :=
begin
  sorry
end

end cos_theta_value_l593_593203


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593023

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593023


namespace marcella_max_pairs_left_l593_593318

theorem marcella_max_pairs_left (pairs_total : ℕ)
                                (types : ℕ)
                                (colors : ℕ)
                                (initial_pairs : ℕ)
                                (shoes_to_lose : ℕ)
                                (pairs_lost : ℕ) :
  pairs_total = types * colors →
  types = 5 →
  colors = 5 →
  initial_pairs = 25 →
  shoes_to_lose = 9 →
  pairs_lost = (shoes_to_lose / 3) →
  initial_pairs - pairs_lost = 22 :=
by
  intros h_total h_types h_colors h_initial_pairs h_shoes_to_lose h_pairs_lost
  have h1 : pairs_total = 25 := by
    rw [h_total, h_types, h_colors]
  have h2 : pairs_lost = 3 := by
    rw [h_pairs_lost, Nat.div_eq_of_eq_mul_left _ _ _]
    norm_num
    rw h_shoes_to_lose
    linarith
  rw [h1, h2] at h_initial_pairs
  linarith

end marcella_max_pairs_left_l593_593318


namespace max_distinct_positive_integers_l593_593716

-- Define the main statement for our proof problem.

theorem max_distinct_positive_integers (m : ℕ → ℕ) (n : ℕ) 
  (h_distinct : ∀ i j, i ≠ j → m i ≠ m j) 
  (h_sum_eq : ∑ i in Finset.range n, (m i) ^ 2 = 3000) : 
  n ≤ 20 :=
sorry

end max_distinct_positive_integers_l593_593716


namespace cylinder_surface_area_l593_593197

theorem cylinder_surface_area (side_length : ℝ) (h : side_length = 2) : 
  let r := side_length / 2 in 
  let h := side_length in
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 6 * Real.pi :=
by 
  sorry

end cylinder_surface_area_l593_593197


namespace sin_315_eq_neg_sqrt2_over_2_l593_593035

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593035


namespace sin_315_eq_neg_sqrt2_div_2_l593_593071

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593071


namespace sin_315_eq_neg_sqrt2_div_2_l593_593012

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593012


namespace odd_digit_probability_l593_593668

theorem odd_digit_probability : 
  let digits := {1, 3, 4, 5}
  -- total number of four-digit arrangements
  let total_arrangements := 4!
  -- compute the favorable arrangements where the units digit is odd
  let favorable_arrangements := 3 * 3!
  -- probability is favorable_arrangements / total_arrangements
  in favorable_arrangements / total_arrangements = (3 / 4 : ℚ) :=
by
  sorry

end odd_digit_probability_l593_593668


namespace front_wheel_more_revolutions_l593_593667

theorem front_wheel_more_revolutions
  (c_f : ℕ) (c_b : ℕ) (d : ℕ)
  (H1 : c_f = 30) (H2 : c_b = 32) (H3 : d = 2400) :
  let F := d / c_f,
      B := d / c_b in
  F - B = 5 :=
by
  let F := 2400 / 30
  let B := 2400 / 32
  have H4 : F = 80 := by sorry
  have H5 : B = 75 := by sorry
  have H6 : F - B = 5 := by
    calc
      F - B = 80 - 75 : by rw [H4, H5]
          ... = 5 : by norm_num
  exact H6

end front_wheel_more_revolutions_l593_593667


namespace library_books_l593_593703

theorem library_books (shelves : ℕ) (books_per_shelf : ℕ) (h_shelves : shelves = 14240) (h_books_per_shelf : books_per_shelf = 8) :
  shelves * books_per_shelf = 113920 :=
by {
  rw [h_shelves, h_books_per_shelf],
  norm_num,
}

end library_books_l593_593703


namespace find_constant_t_l593_593819

theorem find_constant_t :
  ∀ t x, 
    (2*x^2 - 3*x + 4) * (5*x^2 + t*x + 9) = 10*x^4 - t^2*x^3 + 23*x^2 - 27*x + 36 ↔ t = -5 :=
begin
  sorry
end

end find_constant_t_l593_593819


namespace volume_of_bottle_l593_593380

theorem volume_of_bottle (r h : ℝ) (π : ℝ) (h₀ : π > 0)
  (h₁ : r^2 * h + (4 / 3) * r^3 = 625) :
  π * (r^2 * h + (4 / 3) * r^3) = 625 * π :=
by sorry

end volume_of_bottle_l593_593380


namespace problem1_problem2_l593_593739

-- Proof Problem 1: Given \( a^{\frac{1}{2}} + a^{-\frac{1}{2}} = 2 \), prove \( a + a^{-1} = 2 \) and \( a^2 + a^{-2} = 2 \).
theorem problem1 (a : ℝ) (h1 : a^(1/2) + a^(-1/2) = 2) : a + a⁻¹ = 2 ∧ a^2 + a^(-2) = 2 := 
sorry

-- Proof Problem 2: Given \( 0.2^x < 25 \), prove \( x > -2 \).
theorem problem2 (x : ℝ) (h2 : 0.2^x < 25) : x > -2 :=
sorry

end problem1_problem2_l593_593739


namespace subtraction_result_l593_593720

theorem subtraction_result :
  5.3567 - 2.1456 - 1.0211 = 2.1900 := 
sorry

end subtraction_result_l593_593720


namespace question1_question2_l593_593556

-- Question 1
theorem question1 (a : ℝ) (h : a = 1 / 2) :
  let A := {x | -1 / 2 < x ∧ x < 2}
  let B := {x | 0 < x ∧ x < 1}
  A ∩ B = {x | 0 < x ∧ x < 1} :=
by
  sorry

-- Question 2
theorem question2 (a : ℝ) :
  let A := {x | a - 1 < x ∧ x < 2 * a + 1}
  let B := {x | 0 < x ∧ x < 1}
  (A ∩ B = ∅) ↔ (a ≤ -1/2 ∨ a ≥ 2) :=
by
  sorry

end question1_question2_l593_593556


namespace problem_statement_l593_593415

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Ioo (-2 : ℝ) 2 then -x^2 + 1 else 
if x ∈ Ioo (-4 : ℝ) (-2 : ℝ) then -((x + 4)^2) + 1 else sorry

theorem problem_statement : 
  (∀ x : ℝ, (f (-x) = f x)) ∧ 
  (∀ x y : ℝ, f (4 - x) = f y) ∧ 
  (∀ x : ℝ, x ∈ Ioo (-2 : ℝ) 2 → f x = -x^2 + 1) → 
  (∀ x : ℝ, x ∈ Ioo (-4 : ℝ) (-2 : ℝ) → f x = -(x + 4)^2 + 1) :=
by
  sorry

end problem_statement_l593_593415


namespace motorcyclist_travel_distances_l593_593435

-- Define the total distance traveled in three days
def total_distance : ℕ := 980

-- Define the total distance traveled in the first two days
def first_two_days_distance : ℕ := 725

-- Define the extra distance traveled on the second day compared to the third day
def second_day_extra : ℕ := 123

-- Define the distances traveled on the first, second, and third days respectively
def day_1_distance : ℕ := 347
def day_2_distance : ℕ := 378
def day_3_distance : ℕ := 255

-- Formalize the theorem statement
theorem motorcyclist_travel_distances :
  total_distance = day_1_distance + day_2_distance + day_3_distance ∧
  first_two_days_distance = day_1_distance + day_2_distance ∧
  day_2_distance = day_3_distance + second_day_extra :=
by 
  sorry

end motorcyclist_travel_distances_l593_593435


namespace bottle_caps_per_box_l593_593292

theorem bottle_caps_per_box (total_bottle_caps boxes : ℕ) (hb : total_bottle_caps = 316) (bn : boxes = 79) :
  total_bottle_caps / boxes = 4 :=
by
  sorry

end bottle_caps_per_box_l593_593292


namespace fractional_eq_solve_simplify_and_evaluate_l593_593412

-- Question 1: Solve the fractional equation
theorem fractional_eq_solve (x : ℝ) (h1 : (x / (x + 1) = (2 * x) / (3 * x + 3) + 1)) : 
  x = -1.5 := 
sorry

-- Question 2: Simplify and evaluate the expression for x = -1
theorem simplify_and_evaluate (x : ℝ)
  (h2 : x ≠ 0) (h3 : x ≠ 2) (h4 : x ≠ -2) :
  (x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4) / ((x+2) / (x^3 - 4*x)) = 
  (x - 4) / (x - 2) ∧ 
  (x = -1) → ((x - 4) / (x - 2) = (5 / 3)) := 
sorry

end fractional_eq_solve_simplify_and_evaluate_l593_593412


namespace solve_for_x_l593_593484

noncomputable def value_of_x (x : ℝ) : Prop :=
  (log 7 (x - 3) + log (sqrt 7) (x^4 - 3) + log (1 / 7) (x - 3) = 5) ∧ (x > 0)

theorem solve_for_x : ∃ x : ℝ, value_of_x x ∧ x = (49 * real.sqrt 7 + 3)^(1/4) :=
by
  sorry

end solve_for_x_l593_593484


namespace cyclic_inequality_l593_593653

theorem cyclic_inequality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^3 * b^3 * (a * b - a * c - b * c + c^2) +
   b^3 * c^3 * (b * c - b * a - c * a + a^2) +
   c^3 * a^3 * (c * a - c * b - a * b + b^2)) ≥ 0 :=
sorry

end cyclic_inequality_l593_593653


namespace circle_equation_circle_radius_other_common_tangent_l593_593137

-- Condition 1: Circle center is on the line l: 3x - y = 0
def circle_center_on_line (a b : ℝ) : Prop := 3 * a = b

-- Condition 2: Circle is tangent to the line l1: x - y + 4 = 0
def circle_tangent_to_line (a b r : ℝ) : Prop := (a - b + 4) / Real.sqrt(2) = r

-- Condition 3: The chord cut by the line x - y = 0 has a length of 2 √6
def chord_length (a r : ℝ) : Prop := 2 * Real.sqrt(r^2 - (a / Real.sqrt(2))^2) = 2 * Real.sqrt(6)

-- Condition 4: Circle C is externally tangent to another circle
def externally_tangent_circles (a b r : ℝ) : Prop := 
  Real.sqrt((a - 2)^2 + (b - 6)^2) = r + Real.sqrt(32)

-- Question 1: Prove the equation of the circle
theorem circle_equation (a b r : ℝ) (h1: circle_center_on_line a b) (h2: circle_tangent_to_line a b r) (h3: chord_length a r) : 
  (a = 1/4) → (r = 7*Real.sqrt(2)/4) → ∀ x y, (x - a)^2 + (y - b)^2 = r^2 :=
sorry

-- Question 2: Prove the radius of the circle
theorem circle_radius (a b r : ℝ) (h1: circle_center_on_line a b) (h2: externally_tangent_circles a b r) :
  r = Real.sqrt(10) + Real.sqrt(2) :=
sorry

-- Question 3: Prove another common tangent line exists
theorem other_common_tangent (a b r : ℝ) (h1: circle_center_on_line a b) (h2: circle_tangent_to_line a b r) :
  ∃ k, (k = 1 ∨ k = -7) ∧ (k = 1 → x - y + 4 = 0) ∧ (k = -7 → 7 * x + y - 20 = 0) :=
sorry

end circle_equation_circle_radius_other_common_tangent_l593_593137


namespace cos_alpha_condition_l593_593529

theorem cos_alpha_condition (α : ℝ) : ( ∃ k : ℤ, α = 2*k*ℝπ + 5*ℝπ / 6 ) ↔ (cos α = -ℝ.sqrt 3 / 2) :=
sorry

end cos_alpha_condition_l593_593529


namespace odd_numbers_le_twice_switch_pairs_l593_593648

-- Number of odd elements in row n is denoted as numOdd n
def numOdd (n : ℕ) : ℕ := -- Definition of numOdd function
sorry

-- Number of switch pairs in row n is denoted as numSwitchPairs n
def numSwitchPairs (n : ℕ) : ℕ := -- Definition of numSwitchPairs function
sorry

-- Definition of Pascal's Triangle and conditions
def binom (n k : ℕ) : ℕ := if k > n then 0 else if k = 0 ∨ k = n then 1 else binom (n-1) (k-1) + binom (n-1) k

-- Check even or odd
def isOdd (n : ℕ) : Bool := n % 2 = 1

-- Definition of switch pair check
def isSwitchPair (a b : ℕ) : Prop := (isOdd a ∧ ¬isOdd b) ∨ (¬isOdd a ∧ isOdd b)

theorem odd_numbers_le_twice_switch_pairs (n : ℕ) :
  numOdd n ≤ 2 * numSwitchPairs (n-1) :=
sorry

end odd_numbers_le_twice_switch_pairs_l593_593648


namespace tangent_line_curve_l593_593580

theorem tangent_line_curve (a b : ℝ)
  (h1 : ∀ (x : ℝ), (x - (x^2 + a*x + b) + 1 = 0) ↔ (a = 1 ∧ b = 1))
  (h2 : ∀ (y : ℝ), (0, y) ∈ { p : ℝ × ℝ | p.2 = 0 ^ 2 + a * 0 + b }) :
  a = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_curve_l593_593580


namespace ellipse_eqn_and_line_eqn_l593_593524

theorem ellipse_eqn_and_line_eqn
  (a b c : ℝ)
  (ha : a > b)
  (hb : b > 0)
  (he : c = sqrt 6)
  (h_ell : a > b > 0)
  (ecc : a = 2 * sqrt 2)
  (ecc_eq: b = sqrt (a ^ 2 - c ^ 2)) :
  (∀ x y : ℝ, (x^2) / (2 * sqrt 2)^2 + (y^2) / (sqrt 2)^2 = 1 → ∀ k : ℝ, k ≠ 0 → (∀ M A B N: ℝ × ℝ,
  M = (0, -1) →
  (A = (x_1, y_1) → B = (x_2, y_2) → N = (n, 0) →
  (y_1 = - 7 / 5 * y_2) ∧ n = (x N) / k → y = k * x - 1))
    ∧ (y = x - 1 ∨ y = -x - 1)) :=
begin
  sorry
end

end ellipse_eqn_and_line_eqn_l593_593524


namespace find_b_of_roots_condition_l593_593244

theorem find_b_of_roots_condition
  (α β : ℝ)
  (h1 : α * β = -1)
  (h2 : α + β = -b)
  (h3 : α * β - 2 * α - 2 * β = -11) :
  b = -5 := 
  sorry

end find_b_of_roots_condition_l593_593244


namespace distance_between_vertices_of_hyperbola_l593_593109

theorem distance_between_vertices_of_hyperbola : 
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 36 = 1) → 
  distance (⟨11, 0⟩ : ℝ × ℝ) (⟨-11, 0⟩) = 22 :=
by
  sorry

end distance_between_vertices_of_hyperbola_l593_593109


namespace triangles_formed_by_lines_l593_593844

theorem triangles_formed_by_lines (n : ℕ) (h : n > 3) (no_parallel_lines : ∀ (l1 l2 : line), l1 ≠ l2 → ¬parallel l1 l2) 
(no_concurrent_lines : ∀ (l1 l2 l3 : line), l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → ¬concurrent l1 l2 l3) : 
∃ t : ℕ, t ≥ (2 * n - 2) / 3 :=
sorry

end triangles_formed_by_lines_l593_593844


namespace sum_sequence_l593_593087

def sequence (y : ℕ → ℕ) : Prop :=
  y 1 = 2 ∧ ∀ n, y (n + 1) = y n + n + 1

theorem sum_sequence (y : ℕ → ℕ) (n : ℕ) (h : sequence y) :
  (∑ k in Finset.range n, y (k + 1)) = 2 * n + n * (n - 1) * (n + 1) / 6 :=
by
  -- The proof is omitted
  sorry

end sum_sequence_l593_593087


namespace min_wins_six_l593_593773

noncomputable def minimum_wins (n k : ℕ) : ℕ :=
if (∃ k, (k : ℚ) / (n : ℚ) < 1 ∧ 
          ((k + 1 : ℚ) / (n + 1 : ℚ) - (k : ℚ) / (n : ℚ) = 1 / 6) ∧ 
          ((k + 3 : ℚ) / (n + 3 : ℚ) - (k + 1 : ℚ) / (n + 1 : ℚ) = 1 / 6)) 
then 6 else 0

theorem min_wins_six (n : ℕ) : minimum_wins n _ = 6 :=
sorry

end min_wins_six_l593_593773


namespace problem_statement_l593_593119

open BigOperators

def floor_div (a b : ℕ) : ℤ := Int.floor (a / b : ℚ)

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem problem_statement :
  (finset.range 1000).filter (λ n, ¬(is_divisible_by (floor_div 998 n + floor_div 999 n + floor_div 1000 n) 3)).card = 22 :=
sorry

end problem_statement_l593_593119


namespace find_second_cert_interest_rate_l593_593425

theorem find_second_cert_interest_rate
  (initial_investment : ℝ := 12000)
  (first_term_months : ℕ := 8)
  (first_interest_rate : ℝ := 8 / 100)
  (second_term_months : ℕ := 10)
  (final_amount : ℝ := 13058.40)
  : ∃ s : ℝ, (s = 3.984) := sorry

end find_second_cert_interest_rate_l593_593425


namespace integer_pairs_count_l593_593543

noncomputable def f (x : ℝ) : ℝ := 4 / (|x| + 2) - 1

theorem integer_pairs_count 
    (a b : ℤ) 
    (h1 : ∀ x, a ≤ x ∧ x ≤ b → f x ∈ set.Icc 0 1)
    (h2 : a ≤ b) : 
    ∃! pairs : ℕ, 
    pairs = 5 :=
by {
  sorry
}

end integer_pairs_count_l593_593543


namespace distance_PP_l593_593589

variables {P P' D D' A B A' B' : Type}

noncomputable def length (a b : Type) : ℝ := sorry

def midpoint (p1 p2 : Type) : Type := sorry

variables (a : ℝ)

-- Conditions
axiom length_AB : length A B = 3
axiom length_A'B' : length A' B' = 5

axiom midpoint_D : D = midpoint A B
axiom midpoint_D' : D' = midpoint A' B'

axiom distance_PD : ∀ P : Type, length P D = a

-- Proof of the main statement
theorem distance_PP'_relationship : ∀ P P' D D' A B A' B', length A B = 3 →
  length A' B' = 5 →
  D = midpoint A B →
  D' = midpoint A' B' →
  length P D = a →
  length P D + length P' D' = (8 / 3) * a :=
by
  intros,
  sorry

end distance_PP_l593_593589


namespace find_angle_and_perimeter_l593_593959

open Real

variables {A B C a b c : ℝ}

/-- If (2a - c)sinA + (2c - a)sinC = 2bsinB in triangle ABC -/
theorem find_angle_and_perimeter
  (h1 : (2 * a - c) * sin A + (2 * c - a) * sin C = 2 * b * sin B)
  (acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (b_eq : b = 1) :
  B = π / 3 ∧ (sqrt 3 + 1 < a + b + c ∧ a + b + c ≤ 3) :=
sorry

end find_angle_and_perimeter_l593_593959


namespace max_S_at_8_l593_593523

noncomputable def S (n : ℕ) : ℝ := sorry -- Define the sum of the first n terms of the sequence
variable { a : ℕ → ℝ } -- Define the arithmetic sequence

-- Given conditions
axiom S_con_16 : S 16 > 0
axiom S_con_17 : S 17 < 0

theorem max_S_at_8 : ∃ n, S n = S 8 ∧ ∀ m, S m ≤ S n :=
begin
  sorry
end

end max_S_at_8_l593_593523


namespace find_m_l593_593155

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l593_593155


namespace triangle_area_ABC_l593_593365

def point : Type := ℝ × ℝ

def reflect_over_y_axis (p : point) : point := (-p.1, p.2)
def reflect_over_y_eq_x (p : point) : point := (p.2, p.1)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def area_of_triangle (A B C : point) : ℝ :=
  let base := distance A B
  let height := | C.2 - A.2 |
  0.5 * base * height

theorem triangle_area_ABC :
  let A : point := (5, 3)
  let B : point := reflect_over_y_axis A
  let C : point := reflect_over_y_eq_x B
  area_of_triangle A B C = 40 :=
by
  let A : point := (5, 3)
  let B : point := reflect_over_y_axis A
  let C : point := reflect_over_y_eq_x B
  sorry

end triangle_area_ABC_l593_593365


namespace find_m_l593_593174

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l593_593174


namespace number_of_rows_of_red_notes_l593_593288

theorem number_of_rows_of_red_notes (R : ℕ) :
  let red_notes_in_each_row := 6
  let blue_notes_per_red_note := 2
  let additional_blue_notes := 10
  let total_notes := 100
  (6 * R + 12 * R + 10 = 100) → R = 5 :=
by
  intros
  sorry

end number_of_rows_of_red_notes_l593_593288


namespace skew_lines_projections_not_points_l593_593388

-- Defining the concepts of skew lines and orthogonal projections
def skew_lines (l1 l2 : Line ℝ) : Prop := ¬ ∃ p : ℝ → Vec3, ∃ t1 t2 : ℝ, l1 p t1 ≠ l2 p t2 ∧ l1 p t1 ≠ l2 p t2

def orthogonal_projection (l : Line ℝ) (plane : Plane ℝ) : Set ℝ := 
  {p : ℝ | ∃ t : ℝ, ∃ n1 n2 : Vec3, n1 ≠ n2 ∧ l p n1 = ⟨p, 0, 0⟩ ∨ l p n2 = ⟨p, 0, 0⟩}

-- The actual statement that we need to prove
theorem skew_lines_projections_not_points (l1 l2 : Line ℝ) (plane : Plane ℝ) 
    (h1 : skew_lines l1 l2) : ¬ (orthogonal_projection l1 plane = {pt1, pt2}) := 
sorry

end skew_lines_projections_not_points_l593_593388


namespace find_m_l593_593175

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l593_593175


namespace maximum_sum_at_20_l593_593317

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
variable (d : ℤ) 

-- Conditions
def arithmetic_seq (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d

def sum_of_terms (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1)) * d / 2

def problem_conditions (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (d : ℤ) : Prop :=
  a_n 1 + a_n 3 + a_n 8 = 99 ∧ a_n 5 = 31 ∧ arithmetic_seq a_n d ∧ sum_of_terms S_n a_n

theorem maximum_sum_at_20 
  (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (d : ℤ) (h : problem_conditions a_n S_n d) : 
  ∃ k : ℕ, (k = 20) ∧ (∀ n : ℕ, S_n n ≤ S_n k) :=
by
  sorry

end maximum_sum_at_20_l593_593317


namespace building_height_indeterminable_l593_593426

noncomputable def flagstaff_height := 17.5
noncomputable def flagstaff_shadow := 40.25
noncomputable def shadow_length_building := 28.75

/-- The height (h) and shadow length (s) relationship is nonlinear: h = a * sqrt(s) + b * s -/
def height_relation (h s a b : ℝ) : Prop :=
  h = a * Real.sqrt(s) + b * s

/-- We need another pair of (h, s) values to determine the height (h) of a shadow length (s). -/
theorem building_height_indeterminable
  (h1 s1 : ℝ) (a b : ℝ) :
  height_relation h1 s1 a b → s1 = flagstaff_shadow → h1 = flagstaff_height →
  ¬ ∃ h2, height_relation h2 shadow_length_building a b :=
by
  intros h1 s1 a b h_rel s1_eq_shadow h1_eq_height
  sorry

end building_height_indeterminable_l593_593426


namespace proof_problem_l593_593193

def f (x : ℝ) : ℝ := (1/3)^x - x + 4

variables (x₀ x₁ x₂ : ℝ)

-- x₀ is a zero of f
axiom h₀ : f x₀ = 0

-- x₁ is in the interval (2, x₀)
axiom h₁ : 2 < x₁ ∧ x₁ < x₀

-- x₂ is in the interval (x₀, +∞)
axiom h₂ : x₀ < x₂

theorem proof_problem : f x₁ > f x₂ :=
by sorry

end proof_problem_l593_593193


namespace max_different_dwarfs_l593_593269

theorem max_different_dwarfs 
  (x y z : ℕ) 
  (h1 : 1 ≤ x ∧ x ≤ 9) 
  (h2 : 1 ≤ z ∧ z ≤ 9) 
  (h3 : 0 ≤ y ∧ y ≤ 9) 
  (h4 : 100 * x + 10 * y + z + 198 = 100 * z + 10 * y + x) :
  x = z - 2 ∧ (3 ≤ z ∧ z ≤ 9) →
  ∃ (dwarfs : Finch (Fin 70)), True :=
begin
  -- Problem setup ensures all conditions are met and
  -- that there exist 70 unique (x, y, z) triples for dwarfs.
  sorry
end

end max_different_dwarfs_l593_593269


namespace hostel_manager_needs_26_packages_l593_593757

noncomputable def num_packages_needed (rooms1 rooms2 : Finset ℕ) : ℕ :=
  let digits := (rooms1 ∪ rooms2).to_list.join.to_string.data
  let digit_frequencies := digits.group_by id
  digit_frequencies.values.map List.length |> List.maximum'.get_or_else 0

theorem hostel_manager_needs_26_packages :
  num_packages_needed ({n | n ≥ 150 ∧ n ≤ 175}.to_finset) ({n | n ≥ 250 ∧ n ≤ 275}.to_finset) = 26 := 
sorry

end hostel_manager_needs_26_packages_l593_593757


namespace incorrect_propositions_l593_593314

noncomputable def vector {n : ℕ} := fin n → ℝ

variables (a b c : vector 3)

-- Proposition 1
def prop1 (a b : vector 3) := 
  ¬ (collinear a b → line_containing a b is_parallel)

-- Proposition 2
def prop2 (a b : vector 3) := 
  ¬ (skew_lines (line_containing a) (line_containing b) → ¬ coplanar a b)

-- Proposition 3
def prop3 (a b c : vector 3) := 
  ¬ (pairwise_coplanar a b c → coplanar a b c)

-- Proposition 4
def prop4 (a b c : vector 3) := 
  ¬ (∀ (p : vector 3), ∃ (x y z : ℝ), p = x • a + y • b + z • c)

theorem incorrect_propositions (a b c : vector 3) : 
  prop1 a b ∧ prop2 a b ∧ prop3 a b c ∧ prop4 a b c :=
by {
  sorry,
}

end incorrect_propositions_l593_593314


namespace chi_square_test_not_reject_l593_593772

theorem chi_square_test_not_reject 
  (n : ℕ) (s2 : ℝ) (sigma0_2 : ℝ) (alpha : ℝ) (k : ℕ) (chi2_crit : ℝ)
  (h_n : n = 21)
  (h_s2 : s2 = 16.2)
  (h_sigma0_2 : sigma0_2 = 15)
  (h_alpha : alpha = 0.01)
  (h_k : k = 20)
  (h_chi2crit : chi2_crit = 37.6) :
  ((n - 1 : ℝ) * s2) / sigma0_2 < chi2_crit :=
by
  -- The actual proof would go here.
  sorry

end chi_square_test_not_reject_l593_593772


namespace lottery_probability_l593_593443

-- Definition of the conditions
def winning_numbers : Finset ℕ := {8, 2, 5, 3, 7, 1}
def numbers_drawn : Finset ℕ := Finset.range 10

-- The proof goal
theorem lottery_probability : 
  (Finset.card {s : Finset ℕ | s ⊆ numbers_drawn ∧ Finset.card s = 6 ∧ 
    (s ∩ winning_numbers).card ≥ 5}.card : ℚ) / (Finset.card (numbers_drawn.powerset_len 6)) = 5 / 42 :=
sorry

end lottery_probability_l593_593443


namespace min_max_condition_zero_condition_l593_593548

noncomputable def f (ω x: ℝ) : ℝ := sin (ω * x) + cos (ω * x)

theorem min_max_condition (ω : ℝ) (hω : ω > 0) : 
  (∀ (x : ℝ), x ∈ (π/8, 5*π/8) → f ω x = f ω ((π/8) + ((5*π/8) - (π/8)) + x)) → 
  (∃! (x : ℝ), f ω x = f ω x) →
  ω = 4 := sorry

theorem zero_condition (ω : ℝ) (hω : ω > 0) : 
  (∀ (x : ℝ), x ∈ (π/8, 5*π/8) → f ω x = f ω ((π/8) + ((5*π/8) - (π/8)) + x)) → 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f ω x1 = 0 ∧ f ω x2 = 0 ∧ x1 ∈ (π/8, 5*π/8) ∧ x2 ∈ (π/8, 5*π/8)) →
  ω ∈ {10/3, 4, 6} := sorry

end min_max_condition_zero_condition_l593_593548


namespace incorrect_description_of_experiment_l593_593781

-- Definitions based on problem conditions
def preparation_of_culture_medium : Prop :=
  "Using distilled water to prepare beef extract peptone medium, which is then sterilized and poured into plates."

def soil_dilutions_spreading : Prop :=
  "Taking 0.1mL of soil dilutions of 10^4, 10^5, 10^6 times and sterile water, and spreading them on the plates of each group."

def incubation_procedure : Prop :=
  "Inverting the plates of the experimental group and the control group, and incubating at 37°C for 24-48 hours."

def bacterial_count_procedure : Prop :=
  "Choosing the plates from the experimental group with bacterial counts between 30 and 300 for counting after confirming sterility of the control group."

-- Lean theorem statement
theorem incorrect_description_of_experiment :
  ¬ bacterial_count_procedure :=
by
  sorry

end incorrect_description_of_experiment_l593_593781


namespace part1_part2_l593_593891

open Classical

theorem part1 (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 1) ∧ (b = 2) ∧ (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :=
by
  sorry

theorem part2 (x y k : ℝ) (a b : ℝ) :
  a = 1 ∧ b = 2 ∧ (x > 0) ∧ (y > 0) ∧ (1 / x + 2 / y = 1) ∧ (2 * x + y ≥ k^2 + k + 2) → -3 ≤ k ∧ k ≤ 2 :=
by
  sorry

end part1_part2_l593_593891


namespace combination_exists_l593_593559

theorem combination_exists 
  (S T Ti : ℝ) (x y z : ℝ)
  (h : 3 * S + 4 * T + 2 * Ti = 40) :
  ∃ x y z : ℝ, x * S + y * T + z * Ti = 60 :=
sorry

end combination_exists_l593_593559


namespace hexagonal_tile_difference_l593_593284

theorem hexagonal_tile_difference :
  let initial_blue_tiles := 15
  let initial_green_tiles := 9
  let new_green_border_tiles := 18
  let new_blue_border_tiles := 18
  let total_green_tiles := initial_green_tiles + new_green_border_tiles
  let total_blue_tiles := initial_blue_tiles + new_blue_border_tiles
  total_blue_tiles - total_green_tiles = 6 := by {
    sorry
  }

end hexagonal_tile_difference_l593_593284


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593017

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593017


namespace sin_C_in_right_triangle_l593_593948

theorem sin_C_in_right_triangle 
  (A B C : ℝ)
  (h1 : sin A = 3 / 5)
  (h2 : sin B = 1 / real.sqrt 2)
  (h3 : A + B + C = π / 2) :
  sin C = 1 / (5 * real.sqrt 2) :=
sorry

end sin_C_in_right_triangle_l593_593948


namespace average_of_remaining_numbers_l593_593664

theorem average_of_remaining_numbers 
  (S S' : ℝ)
  (h1 : S / 12 = 90)
  (h2 : S' = S - 80 - 82) :
  S' / 10 = 91.8 :=
sorry

end average_of_remaining_numbers_l593_593664


namespace EF_length_l593_593329

-- Defining the angles in degrees
def angleEFG : ℝ := 60
def angleEHG : ℝ := 50

-- Defining the lengths of the sides
def lengthEG : ℝ := 5
def lengthFH : ℝ := 7

theorem EF_length :
  ∀ (E F G H : Type) [inscribed_circle E F G H],
  (angleEFG = 60) → (angleEHG = 50) →
  (lengthEG = 5) → (lengthFH = 7) →
  EF = 7 :=
by {
  sorry
}

end EF_length_l593_593329


namespace possible_sampled_room_l593_593600

/-- 
  Given 60 examination rooms, numerically labeled from 001 to 060, 
  selected through systematic sampling into 12 samples, with one known sample room being 007. 
  Prove that 002 is a possible sampled room number.
-/
theorem possible_sampled_room :
  ∀ (rooms : Finset ℕ), 
    (rooms = Finset.range (60 + 1) \ {0} ∧ ∃ interval : ℕ, interval = 5 ∧ ∃ (sampled_rooms : Finset ℕ), 
    sampled_rooms = Finset.filter (λ x, x % interval = 2 % interval) rooms ∧ 7 ∈ sampled_rooms) → 
    2 ∈ sampled_rooms :=
by
  intro rooms
  intro h
  cases h with h1 h2
  cases h2 with interval h3
  cases h3 with interval_eq h4
  cases h4 with sampled_rooms h5
  cases h5 with sampled_rooms_eq h6
  exact sorry

end possible_sampled_room_l593_593600


namespace carter_baseball_cards_l593_593631

theorem carter_baseball_cards (m c : ℕ) (h1 : m = 210) (h2 : m = c + 58) : c = 152 := 
by
  sorry

end carter_baseball_cards_l593_593631


namespace find_m_l593_593170

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l593_593170


namespace find_a_from_constant_term_l593_593922

noncomputable def constant_term_in_binomial_expansion_eq (a : ℝ) : Prop :=
  (∀ (r : ℕ), r ≤ 4 → (choose 4 r * 2^(4 - r) * a^r = 96) → 4 - 2 * r = 0) ∧ a > 0

theorem find_a_from_constant_term 
  (h : constant_term_in_binomial_expansion_eq a) : a = 2 :=
sorry

end find_a_from_constant_term_l593_593922


namespace distribute_students_l593_593099

theorem distribute_students :
  ∃ n : ℕ, (let students := 9
                dorms := 3
            in    ∃ (f : Fin students → Fin dorms), 
                  (∀ d : Fin dorms, ∃! s : Fin students, f s = d ∨ (exists v : ℕ, v = 4) 
                      .= ∃ t : ℕ, t == 3570)) :=
begin
  sorry
end

end distribute_students_l593_593099


namespace hall_reunion_attendance_l593_593788

/-- At the Taj Hotel, two family reunions are happening: the Oates reunion and the Hall reunion.
All 150 guests at the hotel attend at least one of the reunions.
70 people attend the Oates reunion.
28 people attend both reunions.
Prove that 108 people attend the Hall reunion. -/
theorem hall_reunion_attendance (total oates both : ℕ) (h_total : total = 150) (h_oates : oates = 70) (h_both : both = 28) :
  ∃ hall : ℕ, total = oates + hall - both ∧ hall = 108 :=
by
  -- Proof will be skipped and not considered for this task
  sorry

end hall_reunion_attendance_l593_593788


namespace number_of_students_run_red_light_l593_593381

theorem number_of_students_run_red_light :
  let total_students := 300
  let yes_responses := 90
  let odd_id_students := 75
  let coin_probability := 1/2
  -- Calculate using the conditions:
  total_students / 2 - odd_id_students / 2 * coin_probability + total_students / 2 * coin_probability = 30 :=
by
  sorry

end number_of_students_run_red_light_l593_593381


namespace pet_food_weight_in_ounces_l593_593320

-- Define the given conditions
def cat_food_bags := 2
def cat_food_weight_per_bag := 3 -- in pounds
def dog_food_bags := 2
def additional_dog_food_weight := 2 -- additional weight per bag compared to cat food
def pounds_to_ounces := 16

-- Calculate the total weight of cat food in pounds
def total_cat_food_weight := cat_food_bags * cat_food_weight_per_bag

-- Calculate the weight of each bag of dog food in pounds
def dog_food_weight_per_bag := cat_food_weight_per_bag + additional_dog_food_weight

-- Calculate the total weight of dog food in pounds
def total_dog_food_weight := dog_food_bags * dog_food_weight_per_bag

-- Calculate the total weight of pet food in pounds
def total_pet_food_weight_pounds := total_cat_food_weight + total_dog_food_weight

-- Convert the total weight to ounces
def total_pet_food_weight_ounces := total_pet_food_weight_pounds * pounds_to_ounces

-- Statement of the problem in Lean 4
theorem pet_food_weight_in_ounces : total_pet_food_weight_ounces = 256 := by
  sorry

end pet_food_weight_in_ounces_l593_593320


namespace find_m_l593_593252

theorem find_m (m : ℝ) (h1 : ∀ x y : ℝ, (x ^ 2 + (y - 2) ^ 2 = 1) → (y = x / m ∨ y = -x / m)) (h2 : 0 < m) :
  m = (Real.sqrt 3) / 3 :=
by
  sorry

end find_m_l593_593252


namespace problem_series_sum_l593_593081

noncomputable def series_sum : ℝ := ∑' n : ℕ, (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem problem_series_sum :
  series_sum = 1 / 200 :=
sorry

end problem_series_sum_l593_593081


namespace number_of_girls_l593_593569

def total_students (T : ℕ) :=
  0.40 * T = 300

def girls_at_school (T : ℕ) :=
  0.60 * T = 450

theorem number_of_girls (T : ℕ) (h : total_students T) : girls_at_school T :=
  sorry

end number_of_girls_l593_593569


namespace tip_percentage_l593_593430

theorem tip_percentage (T : ℝ) (h1 : 10 * 24.265 = 242.65)
    (h2 : 211 + 211 * T = 242.65) :
    T ≈ 0.15 :=
by
  sorry

end tip_percentage_l593_593430


namespace new_ratio_is_2_to_1_l593_593405

-- Definitions based on conditions
def initial_mixture_volume : ℝ := 45
def initial_milk_to_water_ratio : ℝ × ℝ := (4, 1)
def additional_water_volume : ℝ := 9

-- The volume of milk in the initial mixture
def initial_milk_volume : ℝ := (initial_milk_to_water_ratio.1 / (initial_milk_to_water_ratio.1 + initial_milk_to_water_ratio.2)) * initial_mixture_volume

-- The volume of water in the initial mixture
def initial_water_volume : ℝ := initial_mixture_volume - initial_milk_volume

-- The volume of water after adding additional water
def new_water_volume : ℝ := initial_water_volume + additional_water_volume

-- The new ratio of milk to water
def new_milk_to_water_ratio : ℝ × ℝ := (initial_milk_volume, new_water_volume)

theorem new_ratio_is_2_to_1 : new_milk_to_water_ratio = (2, 1) :=
by
    -- Proof steps would go here, added sorry as the placeholder
    sorry

end new_ratio_is_2_to_1_l593_593405


namespace sin_315_eq_neg_sqrt2_div_2_l593_593077

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593077


namespace sin_315_eq_neg_sqrt2_div_2_l593_593015

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593015


namespace domain_transform_l593_593535

theorem domain_transform {f : ℝ → ℝ} :
  (∀ x, -2 ≤ x+1 ∧ x+1 ≤ 3 → ∃ y, y = f(x+1)) →
  (∀ x, 2 ≤ x ∧ x ≤ 9/2 → ∃ y, y = f(2*x-5)) :=
by
  sorry

end domain_transform_l593_593535


namespace largest_subset_no_quadruples_l593_593777

theorem largest_subset_no_quadruples (S : Set ℕ) (hS : ∀ a b ∈ S, (a = 4 * b ∨ b = 4 * a) → a = b) :
  S ⊆ {1..200} → S.card ≤ 196 := 
sorry

end largest_subset_no_quadruples_l593_593777


namespace sin_315_eq_neg_sqrt2_div_2_l593_593078

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593078


namespace count_distinct_digit_numbers_divisible_by_5_l593_593908

def has_four_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.nodup

def in_range (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem count_distinct_digit_numbers_divisible_by_5 :
  ({n : ℕ | has_four_distinct_digits n ∧ in_range n ∧ divisible_by_5 n}).card = 504 :=
sorry

end count_distinct_digit_numbers_divisible_by_5_l593_593908


namespace hoursWorkedPerDay_l593_593296

-- Define the conditions
def widgetsPerHour := 20
def daysPerWeek := 5
def totalWidgetsPerWeek := 800

-- Theorem statement
theorem hoursWorkedPerDay : (totalWidgetsPerWeek / widgetsPerHour) / daysPerWeek = 8 := 
  sorry

end hoursWorkedPerDay_l593_593296


namespace cherry_tree_height_difference_l593_593590

theorem cherry_tree_height_difference (a c : ℚ) (a_eq : a = 53 / 4) (c_eq : c = 147 / 8) :
  c - a = 41 / 8 := by
  rw [a_eq, c_eq]
  norm_num
  sorry

end cherry_tree_height_difference_l593_593590


namespace repeating_decimal_as_fraction_l593_593499

-- Define repeating decimal 0.7(3) as x
def x := 0.7 + 3 / 10 ^ (2 + n) where n is some natural number

theorem repeating_decimal_as_fraction :
    x = 11 / 15 := sorry

end repeating_decimal_as_fraction_l593_593499


namespace min_diff_gcd_lcm_l593_593360

/-- 
Given that the GCD of two numbers a and b is 3
and the LCM of these two numbers is 135,
prove that the minimum difference between a and b is 12.
--/
theorem min_diff_gcd_lcm (a b : ℕ) (h_gcd : Nat.gcd a b = 3) (h_lcm : Nat.lcm a b = 135) :
    ∃ a b, a ≠ b ∧ (Nat.abs (a - b) = 12) :=
begin
  sorry
end

end min_diff_gcd_lcm_l593_593360


namespace exists_k_consecutive_squareful_numbers_l593_593437

-- Define what it means for a number to be squareful
def is_squareful (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧ m * m ∣ n

-- State the theorem
theorem exists_k_consecutive_squareful_numbers (k : ℕ) : 
  ∃ (a : ℕ), ∀ i, i < k → is_squareful (a + i) :=
sorry

end exists_k_consecutive_squareful_numbers_l593_593437


namespace problem_simplify_l593_593335

variable (a : ℝ)

theorem problem_simplify (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3)) :=
sorry

end problem_simplify_l593_593335


namespace unique_line_equal_intercepts_l593_593687

-- Definitions of the point and line
structure Point where
  x : ℝ
  y : ℝ

def passesThrough (L : ℝ → ℝ) (P : Point) : Prop :=
  L P.x = P.y

noncomputable def hasEqualIntercepts (L : ℝ → ℝ) : Prop :=
  ∃ a, L 0 = a ∧ L a = 0

-- The main theorem statement
theorem unique_line_equal_intercepts (L : ℝ → ℝ) (P : Point) (hP : P.x = 2 ∧ P.y = 1) (h_equal_intercepts : hasEqualIntercepts L) :
  ∃! (L : ℝ → ℝ), passesThrough L P ∧ hasEqualIntercepts L :=
sorry

end unique_line_equal_intercepts_l593_593687


namespace min_value_5_sqrt_5_l593_593533

noncomputable def min_value (x : ℝ) (hx : 0 < x ∧ x < π / 2) : ℝ :=
  8 / Real.sin x + 1 / Real.cos x

theorem min_value_5_sqrt_5 :
  ∃ x ∈ Ioo 0 (π / 2), min_value x ⟨lt_of_lt_of_le zero_lt_one (le_of_lt $ half_lt_self (Real.pi_pos)), half_pos Real.pi_pos⟩ = 5 * Real.sqrt 5 :=
sorry

end min_value_5_sqrt_5_l593_593533


namespace range_of_x_l593_593514

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) 
  (ineq : |a + b| + |a - b| ≥ |a| * |x - 2|) : 
  0 ≤ x ∧ x ≤ 4 :=
  sorry

end range_of_x_l593_593514


namespace blue_ball_distance_l593_593704

theorem blue_ball_distance :
  ∀ (balls : Fin 2012 → Ball) (f : Fin 2012 → ℕ),
    (∀ i : Fin 2012, balls i ∈ {Ball.red, Ball.yellow, Ball.blue}) →
    (∀ i : Fin 2012, f i = i) →
    (∀ i : Fin 2012, i % 4 = 0 → balls i = Ball.yellow ∨ balls i = Ball.blue ∨ balls i = Ball.red) →
    (∀ i : Fin 2012, i % 4 = 1 → balls i = Ball.blue) →
    (∀ i : Fin 2012, i % 4 = 2 → balls i = Ball.blue ∧ balls i ≠ Ball.red) →
    (∀ i : Fin 2012, i % 4 = 3 → balls i = Ball.red) →
    let red_pos := 3 + 4 * 99,
        yellow_pos := 0 + 4 * 99,
        
        L100th_blue := (2 + (100 - 1) // 2 * 4) + (4 + 1) - 399,
        R100th_blue := 1613 - 1,
    ∃ (d : ℕ), d = red_pos - yellow_pos ∧ (d = 1213) ∧ (d = R100th_blue - L100th_blue)
    sorry
    
end blue_ball_distance_l593_593704


namespace least_possible_value_z_minus_x_l593_593404

theorem least_possible_value_z_minus_x
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  z - x = 9 :=
  sorry

end least_possible_value_z_minus_x_l593_593404


namespace triangle_position_after_rolling_square_l593_593444

theorem triangle_position_after_rolling_square
  (initial_position : String := "left")
  (rotation : ℕ := 4)
  (polygon_sides : ℕ := 8) :
  (initial_position = "left") →
  (rotation * (360 - ((polygon_sides - 2) * 180 / polygon_sides + 90)) ≡ 180 [MOD 360]) →
  (direction_after_rotation : String := "right") :=
by
  sorry

end triangle_position_after_rolling_square_l593_593444


namespace angle_A_120_l593_593963

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem angle_A_120 
  (h₁ : a^2 - b^2 = 3 * b * c)
  (h₂ : sin C = 2 * sin B) :
  A = 120 :=
sorry

end angle_A_120_l593_593963


namespace limit_an_to_a_l593_593646

theorem limit_an_to_a (ε : ℝ) (hε : ε > 0) : 
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  |(9 - (n^3 : ℝ)) / (1 + 2 * (n^3 : ℝ)) + 1/2| < ε :=
sorry

end limit_an_to_a_l593_593646


namespace distance_between_A_and_B_l593_593750

-- Define speeds, times, and distances as real numbers
def speed_A_to_B := 42.5
def time_travelled := 1.5
def remaining_to_midpoint := 26.0

-- Define the total distance between A and B as a variable
def distance_A_to_B : ℝ := 179.5

-- Prove that the distance between locations A and B is 179.5 kilometers given the conditions
theorem distance_between_A_and_B : (42.5 * 1.5 + 26) * 2 = 179.5 :=
by 
  sorry

end distance_between_A_and_B_l593_593750


namespace lecture_scheduling_l593_593761

theorem lecture_scheduling:
  (∃ arrangement : list ℕ, arrangement.length = 7 ∧ 
    (∀ (i j : ℕ), i < j → (arrangement[i] = 2 → arrangement[j] ≠ 1) ∧ (arrangement[i] = 3 → arrangement[j] ≠ 1) ∧ (arrangement[i] = 3 → arrangement[j] ≠ 2))) →
    7! / 2 / 6 = 240 :=
by
  sorry

end lecture_scheduling_l593_593761


namespace smallest_m_for_sum_of_funky_numbers_l593_593477

def is_funky (x : ℝ) : Prop :=
  ∃ n : ℕ, x = n * 10 ^ (-(n+1))

theorem smallest_m_for_sum_of_funky_numbers : ∃ m : ℕ, (∀ (f : ℕ → ℝ), (∀ i : ℕ, is_funky (f i)) → (∑ i in range m, f i) = 1 / 2) ∧ m = 5 :=
by
  sorry

end smallest_m_for_sum_of_funky_numbers_l593_593477


namespace complex_imaginary_part_l593_593138

open Complex

-- Define the given complex equation and extract its imaginary part
theorem complex_imaginary_part (z : ℂ) (hz : z * (1 - I) = Complex.abs (1 - I) + I) : 
  z.im = (Real.sqrt 2 + 1) / 2 :=
by
  let w := 1 - I
  have hw : Complex.abs w = Real.sqrt 2 := by
    rw [Complex.abs, Complex.norm_sq_apply, sub_self, of_real_one, norm_sq_eq_abs_sq, abs_I, abs_I]
    exact Real.sqrt_pos 2
  -- Use the provided condition to derive the imaginary part
  sorry

end complex_imaginary_part_l593_593138


namespace prob1_prob2_l593_593607

-- Proof problem 1: Prove \sin^2 \frac{B+C}{2} + \cos 2A = -\frac{1}{9} given \cos A = \frac{1}{3}
theorem prob1 (A B C : ℝ) (cos_A : ℝ) (h1 : cos_A = 1 / 3) :
  let sin_squared_half_B_plus_C := sin (B + C) / 2 * sin (B + C) / 2,
      cos_2A := 2 * cos_A * cos_A - 1
  in sin_squared_half_B_plus_C + cos_2A = -1 / 9 :=
by
  sorry

-- Proof problem 2: Prove \max (bc) = \frac{9}{4} given \cos A = \frac{1}{3} and a = \sqrt{3}
theorem prob2 (a b c : ℝ) (cos_A : ℝ) (h1 : cos_A = 1 / 3) (h2 : a = Real.sqrt 3) :
  let bc_max := max (b * c)
  in bc_max = 9 / 4 :=
by
  sorry

end prob1_prob2_l593_593607


namespace sara_saw_eight_dozens_l593_593331

variable (total_birds : ℕ) (dozen_size : ℕ)

def number_of_dozens (total_birds : ℕ) (dozen_size : ℕ) : ℕ := total_birds / dozen_size

theorem sara_saw_eight_dozens (h1 : total_birds = 96) (h2 : dozen_size = 12) :
  number_of_dozens total_birds dozen_size = 8 :=
by
  simp [number_of_dozens, h1, h2]
  sorry

end sara_saw_eight_dozens_l593_593331


namespace problem_part1_problem_part2_l593_593887

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l593_593887


namespace real_roots_implies_m_range_pq_false_and_pq_true_implies_m_range_l593_593852

variable (m : ℝ)

def p := ∃ x : ℝ, x² - 2*x + m = 0
def q := m ∈ Set.Icc (-1) 5

-- First proposition: If the equation has real roots then m ∈ (-∞, 1]
theorem real_roots_implies_m_range (hp : p m) : m ≤ 1 :=
by sorry

-- Second proposition: If p ∧ q is false and p ∨ q is true, then m ∈ (-∞, -1) ∪ (1, 5]
theorem pq_false_and_pq_true_implies_m_range (hpq_false : ¬ (p m ∧ q m)) (hpq_true : p m ∨ q m) : m ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioc 1 5 :=
by sorry

end real_roots_implies_m_range_pq_false_and_pq_true_implies_m_range_l593_593852


namespace misha_board_l593_593285

theorem misha_board (N : ℕ) (hN : 1 < N) : 
  ¬ (∃ (a : ℕ), (a > 1 ∧ (λ s t : set ℕ, t = s ∪ {d | d ∣ a ∧ d ≠ a} ∧ t.card = N^2))) :=
begin
  sorry,
end

end misha_board_l593_593285


namespace sin_315_eq_neg_sqrt2_div_2_l593_593068

noncomputable def sin_45 : ℝ := real.sin (real.pi / 4)

theorem sin_315_eq_neg_sqrt2_div_2 : real.sin (7 * real.pi / 4) = -sin_45 :=
by
  sorry

end sin_315_eq_neg_sqrt2_div_2_l593_593068


namespace pet_food_weight_in_ounces_l593_593321

-- Define the given conditions
def cat_food_bags := 2
def cat_food_weight_per_bag := 3 -- in pounds
def dog_food_bags := 2
def additional_dog_food_weight := 2 -- additional weight per bag compared to cat food
def pounds_to_ounces := 16

-- Calculate the total weight of cat food in pounds
def total_cat_food_weight := cat_food_bags * cat_food_weight_per_bag

-- Calculate the weight of each bag of dog food in pounds
def dog_food_weight_per_bag := cat_food_weight_per_bag + additional_dog_food_weight

-- Calculate the total weight of dog food in pounds
def total_dog_food_weight := dog_food_bags * dog_food_weight_per_bag

-- Calculate the total weight of pet food in pounds
def total_pet_food_weight_pounds := total_cat_food_weight + total_dog_food_weight

-- Convert the total weight to ounces
def total_pet_food_weight_ounces := total_pet_food_weight_pounds * pounds_to_ounces

-- Statement of the problem in Lean 4
theorem pet_food_weight_in_ounces : total_pet_food_weight_ounces = 256 := by
  sorry

end pet_food_weight_in_ounces_l593_593321


namespace find_m_l593_593161

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l593_593161


namespace constant_subtracted_l593_593914

theorem constant_subtracted (w : ℤ) (h1 : 2^(2*w) = 8^(w-4)) (h2 : w = 12) : 8^(w-4) = 2^(3*(w-4)) :=
by
  sorry

end constant_subtracted_l593_593914


namespace largest_prime_factor_sum_divisors_450_l593_593979

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.divisors n).sum

theorem largest_prime_factor_sum_divisors_450 :
  let M := sum_of_divisors 450 in
  M = 1209 ∧ Nat.greatestPrimeFactor M = 13 := by
sorry

end largest_prime_factor_sum_divisors_450_l593_593979


namespace max_min_value_sum_l593_593209

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x) * Real.sin (x - 2) + x + 1

theorem max_min_value_sum (M m : ℝ) 
  (hM : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ M)
  (hm : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ m)
  (hM_max : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = M)
  (hm_min : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = m)
  : M + m = 6 :=
sorry

end max_min_value_sum_l593_593209


namespace smallest_positive_period_f_interval_monotonically_decreasing_f_area_of_triangle_ABC_l593_593545

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x) ^ 2 - Real.sqrt 3

theorem smallest_positive_period_f :
  ∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x :=
sorry

theorem interval_monotonically_decreasing_f :
  ∀ k : ℤ, (k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12) → (∀ y z : ℝ, y < z → y ∈ (k * π + π / 12, k * π + 7 * π / 12) → z ∈ (k * π + π / 12, k * π + 7 * π / 12) → f y ≥ f z)
:=
sorry

theorem area_of_triangle_ABC (a b c A B C : ℝ) (h_triangle : Geometry.Triangle a b c A B C)
  (h_a : a = 7)
  (h_f : f (A / 2 - π / 6) = Real.sqrt 3)
  (h_sin_sum : Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14)
  (h_acute : 0 < A ∧ A < π / 2) :
  Geometry.TriangleArea a b c = 10 * Real.sqrt 3 :=
sorry

end smallest_positive_period_f_interval_monotonically_decreasing_f_area_of_triangle_ABC_l593_593545


namespace find_a_b_l593_593867

theorem find_a_b (a b : ℝ) (h₀ : b ≠ 0) (h₁ : (a + b * complex.I)^2 = -b * complex.I) : 
  a = -1/2 ∧ (b = 1/2 ∨ b = -1/2) :=
by
  sorry

end find_a_b_l593_593867


namespace partI_tangent_line_partII_range_of_a_l593_593549

variable (a : ℝ) (f : ℝ → ℝ)

noncomputable def tangent_line_at_one : ℝ → ℝ :=
  λ x, -2

theorem partI_tangent_line (h : a = 1) :
  (λ x, x^2 - 3 * x + Real.log x) = f → 
  (tangent_line_at_one a f 1  = -2) :=
  sorry

theorem partII_range_of_a (h : 0 < a)
    (h_min : ∀ x ∈ Set.Icc (1:ℝ) (Real.exp 1), 
    ax^2 - (a+2)x + Real.log x ≥ -2) :
  1 ≤ a :=
  sorry

end partI_tangent_line_partII_range_of_a_l593_593549


namespace num_sets_of_consecutive_integers_sum_to_30_l593_593239

theorem num_sets_of_consecutive_integers_sum_to_30 : 
  let S_n (n a : ℕ) := (n * (2 * a + n - 1)) / 2 
  ∃! (s : ℕ), s = 3 ∧ ∀ n, n ≥ 2 → ∃ a, S_n n a = 30 :=
by
  sorry

end num_sets_of_consecutive_integers_sum_to_30_l593_593239


namespace circle_at_most_two_rational_points_l593_593831

theorem circle_at_most_two_rational_points 
  (a b : ℝ) (r : ℝ) 
  (ha : ¬ (∃ q : ℚ, a = q))
  (hb : ¬ (∃ q : ℚ, b = q))
  (hr : r > 0)
  : ∃ (S : set (ℚ × ℚ)), ∀ (x y : ℝ),
    (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2 → 
    (x ∈ ℚ ∧ y ∈ ℚ) → 
    S x <= 2 := sorry

end circle_at_most_two_rational_points_l593_593831


namespace non_similar_regular_stars_l593_593478

def euler_totient_function (n : ℕ) : ℕ := 
  (List.range n).filter (λ k, Nat.gcd n k = 1).length

theorem non_similar_regular_stars (n : ℕ) (h : n = 1000) :
  (euler_totient_function n) / 2 - 1 = 199 := by
  have h_totient : euler_totient_function 1000 = 400 := by sorry
  rw [h] at h_totient
  rw [h_totient]
  norm_num

end non_similar_regular_stars_l593_593478


namespace find_a1_b1_l593_593796

def sequence (a1 b1 : ℕ) : ℕ → ℂ
| 1       := a1 + b1 * complex.I
| (n + 1) := let z_n := sequence a1 b1 n
             in (2 * z_n.re - z_n.im) + (2 * z_n.im + z_n.re) * complex.I

theorem find_a1_b1 (a1 b1 : ℕ) (H : sequence a1 b1 50 = 3 + 6 * complex.I) : a1 + b1 = C :=
sorry

end find_a1_b1_l593_593796


namespace find_smallest_positive_angle_l593_593513

noncomputable def theta_deg : ℝ := 50

theorem find_smallest_positive_angle (θ : ℝ) (hθ : θ = theta_deg) :
  cos θ = sin 70 + cos 50 - sin 20 - cos 10 :=
by
  sorry

end find_smallest_positive_angle_l593_593513


namespace proof_N_union_complement_M_eq_235_l593_593629

open Set

theorem proof_N_union_complement_M_eq_235 :
  let U := ({1,2,3,4,5} : Set ℕ)
  let M := ({1, 4} : Set ℕ)
  let N := ({2, 5} : Set ℕ)
  N ∪ (U \ M) = ({2, 3, 5} : Set ℕ) :=
by
  sorry

end proof_N_union_complement_M_eq_235_l593_593629


namespace set_equality_b_minus_a_l593_593621

theorem set_equality_b_minus_a (a b : ℝ) (h : {a, 1} = {0, a + b}) : b - a = 1 := 
by
  sorry

end set_equality_b_minus_a_l593_593621


namespace intersection_range_of_k_l593_593537

noncomputable def curve (x y : ℝ) : Prop :=
  sqrt (1 - (y - 1)^2) = abs x - 1

noncomputable def line (k x y : ℝ) : Prop :=
  k * x - y = 2

theorem intersection_range_of_k :
  (∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧ y1 ≠ y2 ∧ curve x1 y1 ∧ curve x2 y2 ∧ line k x1 y1 ∧ line k x2 y2) ↔
  (k ∈ set.Icc (-2 : ℝ) (-4/3) ∨ k ∈ set.Ico (4/3 : ℝ) 2) :=
sorry

end intersection_range_of_k_l593_593537


namespace proposition_2_true_l593_593985

-- Definitions of line and plane
variables (Line Plane : Type)
variables (m n : Line) 
variables (α β γ : Plane)

-- Predicates for parallelism and perpendicularity
variables (Parallel : Line → Plane → Prop)
variables (Perpendicular : Plane → Plane → Prop)
variables (IntersectAt : Plane → Plane → Line → Prop)

-- Proposition ② to be proved
theorem proposition_2_true :
  ∀ (m : Line) (α β : Plane),
  (Parallel m α) →
  (∃ n : Line, n ≠ m ∧ Parallel n α ∧ Parallel m n) →
  (Perpendicular m β) →
  ∃ (n : Line), (n ≠ m) ∧ Parallel n α ∧ Perpendicular n β → Perpendicular α β :=
by
  intro m α β h1 h2 h3 h4
  sorry  -- Proof goes here

end proposition_2_true_l593_593985


namespace groups_with_males_and_females_l593_593517

noncomputable def C : ℕ → ℕ → ℕ
| n, k := Nat.choose n k

theorem groups_with_males_and_females :
  let total_ways := C 7 3
  let all_males := C 4 3
  let all_females := C 3 3
  total_ways - all_males - all_females = 30 :=
by {
  let total_ways := C 7 3,
  let all_males := C 4 3,
  let all_females := C 3 3,
  sorry
}

end groups_with_males_and_females_l593_593517


namespace intersection_A_B_l593_593899

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {0, 1} := 
by sorry

end intersection_A_B_l593_593899


namespace find_roots_l593_593824

theorem find_roots (x : ℝ) (h : 21 / (x^2 - 9) - 3 / (x - 3) = 1) : x = -3 ∨ x = 7 :=
by {
  sorry
}

end find_roots_l593_593824


namespace problem_part1_problem_part2_l593_593886

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l593_593886


namespace find_a_l593_593956

theorem find_a (a : ℝ) : 
  (∃ r : ℕ, (10 - 3 * r = 1 ∧ (-a)^r * (Nat.choose 5 r) *  x^(10 - 2 * r - r) = x ∧ -10 = (-a)^3 * (Nat.choose 5 3)))
  → a = 1 :=
sorry

end find_a_l593_593956


namespace sky_changes_colors_l593_593498

theorem sky_changes_colors (hours : ℕ) (minutes_per_hour : ℕ) (interval : ℕ) (total_duration : ℕ) :
  hours = 2 →
  minutes_per_hour = 60 →
  interval = 10 →
  total_duration = hours * minutes_per_hour →
  total_duration / interval = 12 :=
by
  intros h_hours h_minutes_per_hour h_interval h_total_duration
  rw [h_hours, h_minutes_per_hour] at h_total_duration
  have h : 120 = total_duration := h_total_duration
  rw [←h]
  have h1 : 120 / 10 = 12 := by norm_num
  exact h1

end sky_changes_colors_l593_593498


namespace triangle_BC_length_l593_593283

variables (A B C D E G : Point)  -- Specify points

-- Conditions
variables (midpoint_D : midpoint_point A B C = D)
variables (midpoint_E : midpoint_point A C B = E)
variables (meet_G : meet_point A D B E = G)
variables (cyclic_GECD : cyclic_quad G E C D)
variables (AB_val : AB = 41)
variables (AC_val : AC = 31)

-- Proof statement: Given these conditions, BC = 49
theorem triangle_BC_length :
  BC = 49 :=
sorry

end triangle_BC_length_l593_593283


namespace Q1_Q2_l593_593741

-- Define the conditions for Problem 1
theorem Q1
  (f g : ℝ → ℝ)
  (h : ∀ x, -π/2 < f(x) + g(x) ∧ f(x) + g(x) < π/2 ∧
       -π/2 < f(x) - g(x) ∧ f(x) - g(x) < π/2) :
  ∀ x, cos (f x) > sin (g x) := sorry

-- Define the conditions for Problem 2
theorem Q2
  (h : ∀ x, -π/2 < cos x + sin x ∧ cos x + sin x < π/2 ∧
       -π/2 < cos x - sin x ∧ cos x - sin x < π/2) :
  ∀ x, cos (cos x) > sin (sin x) := sorry

end Q1_Q2_l593_593741


namespace angle_between_hands_at_12_15_l593_593785

theorem angle_between_hands_at_12_15 :
  let hour_hand_deg_per_minute := 0.5
  let minute_hand_deg_per_minute := 6
  let angle_per_clock_number := 30
  let time_minutes := 15
  let hour_angle := hour_hand_deg_per_minute * time_minutes
  let minute_position := (3 : Nat) -- 3 o'clock is the 15-minute mark
  let minute_angle := angle_per_clock_number * minute_position
  let angle := minute_angle - hour_angle
  angle = 82.5 := sorry

end angle_between_hands_at_12_15_l593_593785


namespace convertible_cars_correct_l593_593450

-- Definitions based on given conditions
def total_cars := 250

def percentage_speedster := 0.35
def percentage_turbo := 0.25
def percentage_cruiser := 0.30
def percentage_roadster := 0.10

def convertible_fraction_speedster := 4 / 5
def convertible_fraction_turbo := 3 / 5
def convertible_fraction_cruiser := 1 / 2

-- Calculation definitions
def num_speedsters := (percentage_speedster * total_cars).round
def num_turbos := (percentage_turbo * total_cars).round
def num_cruisers := (percentage_cruiser * total_cars).round
def num_roadsters := (percentage_roadster * total_cars).round

def num_convertible_speedsters := (convertible_fraction_speedster * num_speedsters).round
def num_convertible_turbos := (convertible_fraction_turbo * num_turbos).round
def num_convertible_cruisers := (convertible_fraction_cruiser * num_cruisers).round
def num_convertible_roadsters := num_roadsters -- All are convertibles

def total_convertibles := num_convertible_speedsters + num_convertible_turbos + num_convertible_cruisers + num_convertible_roadsters

-- Lean statement to prove the total number of convertibles is 171.
theorem convertible_cars_correct : total_convertibles = 171 := 
by 
  sorry

end convertible_cars_correct_l593_593450


namespace part1_part2_l593_593546

open Real

noncomputable def f (x : ℝ) (m : ℝ) := (1 / 2) * (cos x)^4 + sqrt 3 * (sin x) * (cos x) - (1 / 2) * (sin x)^4 + m

theorem part1 {m : ℝ} (h : ∀ x, f x m ≤ 3 / 2) :
  m = 1 / 2 ∧ ∀ x, f x m = 3 / 2 → (∃ k : ℤ, x = k * π + π / 6) :=
begin
  sorry
end

theorem part2 {m : ℝ} (hm : m = 1 / 2) :
  (∀ x, -π / 3 ≤ x ∧ x ≤ π / 6 → f x m ≤ f (x + π) m)∧
  (∀ x, f x m ≤ f (x + 2 * π) m ∧ f (x - 2 * π) m ≤ f x m) :=
begin
  sorry
end

end part1_part2_l593_593546


namespace perfect_squares_l593_593686

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l593_593686


namespace newOp_of_M_and_N_l593_593801

def newOp (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∨ x ∈ B ∧ x ∉ (A ∩ B)}

theorem newOp_of_M_and_N (M N : Set ℕ) :
  M = {0, 2, 4, 6, 8, 10} →
  N = {0, 3, 6, 9, 12, 15} →
  newOp (newOp M N) M = N :=
by
  intros hM hN
  sorry

end newOp_of_M_and_N_l593_593801


namespace decreasing_interval_of_function_l593_593212
noncomputable def f (a x : ℝ) : ℝ :=
  x^2 * Real.log x + a * x

theorem decreasing_interval_of_function (a : ℝ) :
  (∃ x > 0, (f a x)' < 0) ↔ a < 2 * Real.exp (-3 / 2) := 
by 
  sorry

end decreasing_interval_of_function_l593_593212


namespace irreducible_fractions_properties_l593_593116

theorem irreducible_fractions_properties : 
  let f1 := 11 / 2
  let f2 := 11 / 6
  let f3 := 11 / 3
  let reciprocal_sum := (2 / 11) + (6 / 11) + (3 / 11)
  (f1 + f2 + f3 = 11) ∧ (reciprocal_sum = 1) :=
by
  sorry

end irreducible_fractions_properties_l593_593116


namespace balloons_in_each_bag_of_round_balloons_l593_593610

variable (x : ℕ)

-- Definitions based on the problem's conditions
def totalRoundBalloonsBought := 5 * x
def totalLongBalloonsBought := 4 * 30
def remainingRoundBalloons := totalRoundBalloonsBought x - 5
def totalRemainingBalloons := remainingRoundBalloons x + totalLongBalloonsBought

-- Theorem statement based on the question and derived from the conditions and correct answer
theorem balloons_in_each_bag_of_round_balloons : totalRemainingBalloons x = 215 → x = 20 := by
  -- We acknowledge that the proof steps will follow here (omitted as per instructions)
  sorry

end balloons_in_each_bag_of_round_balloons_l593_593610


namespace angle_BAC_60_l593_593649

-- Defining the conditions
variables (A B C D : Point)
variables {AB BC CD DA : ℝ}
variables {angle_ABC angle_BCD : ℝ}

-- Given conditions
axiom AB_eq_BC : AB = BC
axiom BC_eq_CD : BC = CD
axiom CD_eq_DA : CD = DA
axiom ABC_eq_100 : angle_ABC = 100
axiom BCD_eq_140 : angle_BCD = 140

-- Goal to prove
theorem angle_BAC_60 : ∀ (A B C D : Point), 
  AB = BC → BC = CD → CD = DA → 
  angle_ABC = 100 → angle_BCD = 140 → 
  ∃ angle_BAC : ℝ, angle_BAC = 60 :=
begin
  intros,
  sorry
end

end angle_BAC_60_l593_593649


namespace sin_315_eq_neg_sqrt2_over_2_l593_593031

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l593_593031


namespace option_A_option_B_option_D_l593_593877

def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / (2 + 2 * Real.sin x * Real.cos x)

theorem option_A : ∀ x : ℝ, f (π / 2 - x) = f x := by
  sorry

theorem option_B : ∀ x : ℝ, f (-π / 2 - x) = -f x := by
  sorry

theorem option_D : ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = 1 / 2 := by
  sorry

end option_A_option_B_option_D_l593_593877


namespace direct_variation_y_value_l593_593766

theorem direct_variation_y_value (x y : ℝ) (hx1 : x ≤ 10 → y = 3 * x)
  (hx2 : x > 10 → y = 6 * x) : 
  x = 20 → y = 120 := by
  sorry

end direct_variation_y_value_l593_593766


namespace number_of_divisors_24_l593_593560

theorem number_of_divisors_24 : (nat.divisors 24).card = 8 :=
by sorry

end number_of_divisors_24_l593_593560


namespace perfect_square_quotient_l593_593859

theorem perfect_square_quotient (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h : (a * b + 1) ∣ (a * a + b * b)) : 
  ∃ k : ℕ, (a * a + b * b) = (a * b + 1) * (k * k) := 
sorry

end perfect_square_quotient_l593_593859


namespace find_smallest_nat_with_remainder_2_l593_593113

noncomputable def smallest_nat_with_remainder_2 : Nat :=
    let x := 26
    if x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] then x
    else 0

theorem find_smallest_nat_with_remainder_2 :
    ∃ x : Nat, x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] ∧ x = smallest_nat_with_remainder_2 :=
    sorry

end find_smallest_nat_with_remainder_2_l593_593113


namespace find_m_l593_593168

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l593_593168


namespace min_triangles_formed_l593_593417

theorem min_triangles_formed (n : ℕ) (h_n : n = 3000) (h1 : ∀ i j, i ≠ j → ¬parallel (lines i) (lines j)) 
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬concurrent (lines i) (lines j) (lines k)) : 
  ∃ t : ℕ, t ≥ 2000 :=
by
  sorry

end min_triangles_formed_l593_593417


namespace number_of_girls_l593_593567

def total_students (T : ℕ) :=
  0.40 * T = 300

def girls_at_school (T : ℕ) :=
  0.60 * T = 450

theorem number_of_girls (T : ℕ) (h : total_students T) : girls_at_school T :=
  sorry

end number_of_girls_l593_593567


namespace black_marbles_count_l593_593289

theorem black_marbles_count
  (yellow blue green : ℕ)
  (h_yellow : yellow = 12)
  (h_blue : blue = 10)
  (h_green : green = 5)
  (prob_black : 1 / ↑(yellow + blue + green + 1) = 1 / 28) :
  (∃ black : ℕ, black = 1) :=
by
  use 1
  sorry

end black_marbles_count_l593_593289


namespace triangle_inequality_proof_l593_593966

def triangle (α β γ : ℝ) := α + β > γ ∧ β + γ > α ∧ γ + α > β

noncomputable def area (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_inequality_proof (a b c S : ℝ) 
  (A : triangle a b c) 
  (H : S = area a b c) : 
  4 * real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ a^2 + b^2 + c^2 ∧ 
  a^2 + b^2 + c^2 ≤ 4 * real.sqrt 3 * S + 3 * (a - b)^2 + 3 * (b - c)^2 + 3 * (c - a)^2 :=
by
  sorry

end triangle_inequality_proof_l593_593966


namespace simplify_expression_l593_593248

theorem simplify_expression (m n : ℝ) (h : m^2 + 3 * m * n = 5) : 
  5 * m^2 - 3 * m * n - (-9 * m * n + 3 * m^2) = 10 :=
by 
  sorry

end simplify_expression_l593_593248


namespace spherical_coordinates_conversion_l593_593475

-- Define spherical coordinates conversion
def spherical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let rho := Real.sqrt (x*x + y*y + z*z)
  let phi := Real.acos (z / rho)
  let theta := Real.atan2 y x
  (rho, theta, phi)

-- Define the given point in rectangular coordinates
def given_point := (0 : ℝ, (3 * Real.sqrt 3 : ℝ), (-3 : ℝ))

-- Expected result in spherical coordinates
def expected_spherical_coordinates := (6 : ℝ, Real.pi / 2, 2 * Real.pi / 3)

-- Final problem statement
theorem spherical_coordinates_conversion :
  spherical_coordinates 0 (3 * Real.sqrt 3) (-3) = (6, Real.pi / 2, 2 * Real.pi / 3) := 
  sorry

end spherical_coordinates_conversion_l593_593475


namespace infinitely_many_primes_with_divisible_diff_l593_593609

open Nat

theorem infinitely_many_primes_with_divisible_diff (a : ℕ) (h_pos : 0 < a) : 
  ∃ (S : Set ℕ), (∀ p ∈ S, Prime p) ∧ Set.Infinite S ∧ ∀ p q ∈ S, p ≠ q → (p - q) % a = 0 := 
sorry

end infinitely_many_primes_with_divisible_diff_l593_593609


namespace product_of_sequence_l593_593256

-- We define the condition and the required proof.
theorem product_of_sequence (a b : ℕ) 
  (h : 4 / 3 * 5 / 4 * 6 / 5 * 7 / 6 * ... * a / b = 16) : a * b = 2256 :=
by
  sorry

end product_of_sequence_l593_593256


namespace shaded_area_of_room_l593_593744

theorem shaded_area_of_room :
  let tile_area := (2 * 2 : ℝ)
  let circle_area := real.pi * (1 * 1 : ℝ)
  let shaded_tile_area := tile_area - circle_area
  let num_tiles := (12 * 15) / (2 * 2 : ℝ)
  (num_tiles * shaded_tile_area = 180 - 45 * real.pi) :=
by
  sorry

end shaded_area_of_room_l593_593744


namespace wheel_radius_l593_593447

theorem wheel_radius (r : ℝ) (hπ : Real.pi ≈ 3.14159) 
  (hD : 1250 * (2 * π * r) = 1760) : r ≈ 0.224 :=
by
  sorry

end wheel_radius_l593_593447


namespace max_cross_section_sides_of_cube_l593_593799

theorem max_cross_section_sides_of_cube (cube : Type) [is_cube cube] : 
  ∃ (cross_section : polygon), 
  (is_cross_section_of_plane_and_cube cross_section cube) ∧ 
  (polygon.sides cross_section = 6) :=
sorry

end max_cross_section_sides_of_cube_l593_593799


namespace incorrect_major_premise_l593_593762

-- Define a structure for Line and Plane
structure Line : Type :=
  (name : String)

structure Plane : Type :=
  (name : String)

-- Define relationships: parallel and contains
def parallel (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Conditions
variables 
  (a b : Line) 
  (α : Plane)
  (H1 : line_in_plane a α) 
  (H2 : parallel_line_plane b α)

-- Major premise to disprove
def major_premise (l : Line) (p : Plane) : Prop :=
  ∀ (l_in : Line), line_in_plane l_in p → parallel l l_in

-- State the problem
theorem incorrect_major_premise : ¬major_premise b α :=
sorry

end incorrect_major_premise_l593_593762


namespace count_valid_telephone_numbers_l593_593094

def is_valid_telephone_number (numbers : list ℕ) : Prop :=
  numbers.length = 7 ∧ 
  (∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 9) ∧ 
  (list.sorted (≤) numbers) ∧
  (list.nodup numbers) ∧
  (∃ p ∈ numbers, nat.prime p)

theorem count_valid_telephone_numbers : 
  ∃ count : ℕ, count = 36 ∧ 
  count = (finset.univ.filter is_valid_telephone_number).card :=
sorry

end count_valid_telephone_numbers_l593_593094


namespace sixth_term_constant_and_coefficient_l593_593849

noncomputable def binomial_term (n r : ℕ) (x : ℚ) : ℚ :=
  (-3 : ℚ)^r * Nat.choose n r * x ^ ((n - 2 * r) / 3 : ℕ)

theorem sixth_term_constant_and_coefficient :
  (∃ n : ℕ, binomial_term n 5 1 = (-3)^5 * nat.choose n 5 ∧ (n - 10) % 3 = 0 ∧ n = 10) ∧
  (∃ r : ℕ, binomial_term 10 r (x : ℚ) = 405 * x^2 ∧ (10 - 2 * r) / 3 = 2) :=
by
  sorry

end sixth_term_constant_and_coefficient_l593_593849


namespace find_m_l593_593153

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l593_593153


namespace prove_angle_A_l593_593960

-- Definitions and conditions in triangle ABC
variables (A B C : ℝ) (a b c : ℝ) (h₁ : a^2 - b^2 = 3 * b * c) (h₂ : sin C = 2 * sin B)

-- Objective: Prove that angle A is 120 degrees
theorem prove_angle_A : A = 120 :=
sorry

end prove_angle_A_l593_593960


namespace number_of_subsets_PQ_l593_593220

def P := {3, 4, 5}
def Q := {6, 7}
def PQ := { (a, b) | a ∈ P, b ∈ Q }

theorem number_of_subsets_PQ :
  let n := 6 in
  (2 ^ n = 64) :=
by
  sorry

end number_of_subsets_PQ_l593_593220


namespace zero_in_interval_l593_593803

noncomputable def f (x : ℝ) : ℝ := 3 ^ x - 2

theorem zero_in_interval : ∃ c ∈ Ioo 0 1, f c = 0 :=
by {
  sorry
}

end zero_in_interval_l593_593803


namespace quadratic_inequality_iff_abs_a_le_two_l593_593411

-- Definitions from the condition
variable (a : ℝ)
def quadratic_expr (x : ℝ) : ℝ := x^2 + a * x + 1

-- Statement of the problem as a Lean 4 statement
theorem quadratic_inequality_iff_abs_a_le_two :
  (∀ x : ℝ, quadratic_expr a x ≥ 0) ↔ (|a| ≤ 2) := sorry

end quadratic_inequality_iff_abs_a_le_two_l593_593411


namespace unbroken_matches_l593_593968

theorem unbroken_matches (dozen_boxes : ℕ) (matches_per_box : ℕ) (broken_per_box : ℕ) :
  dozen_boxes = 5 -> matches_per_box = 20 -> broken_per_box = 3 ->
  60 * (matches_per_box - broken_per_box) = 1020 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- simplify to the final statement to match the proof problem
  show 60 * (20 - 3) = 1020
  have : 20 - 3 = 17 := rfl
  rw this
  show 60 * 17 = 1020
  sorry

end unbroken_matches_l593_593968


namespace Mindy_tax_rate_l593_593634

variables (M : Real) (r : Real)

-- Define the conditions
def Mork_tax_rate := 0.40
def Mindy_income := 4 * M
def combined_income := M + Mindy_income
def combined_tax_rate := 0.28

-- Theorem stating that Mindy's tax rate is 0.25
theorem Mindy_tax_rate : (Mork_tax_rate * M + r * Mindy_income) / combined_income = combined_tax_rate -> r = 0.25 :=
by
  intros h
  sorry

end Mindy_tax_rate_l593_593634


namespace smallest_positive_period_of_f_range_of_f_on_interval_l593_593875

noncomputable def f (x : ℝ) : ℝ := 2 * cos x ^ 2 + 2 * sqrt 3 * sin x * cos x

theorem smallest_positive_period_of_f :
  ∀ x, f (x + π) = f x :=
by {
  sorry
}

theorem range_of_f_on_interval :
  ∀ x, 0 < x ∧ x ≤ π / 3 → (2 ≤ f x ∧ f x ≤ 3) :=
by {
  sorry
}

end smallest_positive_period_of_f_range_of_f_on_interval_l593_593875


namespace binomial_10_10_binomial_10_9_l593_593793

-- Prove that \(\binom{10}{10} = 1\)
theorem binomial_10_10 : Nat.choose 10 10 = 1 :=
by sorry

-- Prove that \(\binom{10}{9} = 10\)
theorem binomial_10_9 : Nat.choose 10 9 = 10 :=
by sorry

end binomial_10_10_binomial_10_9_l593_593793


namespace exists_shape_l593_593102

-- The problem assumes existence of four shapes and five shapes
constant Shape : Type

constant shape_A B C D E : Shape

constant is_partition_four : Shape → Bool
constant is_partition_five : Shape → Bool

-- Assuming the conditions of the problem as constants
axiom cond1 : ∀ (F : Shape), is_partition_four F = true → (F = (A ⊔ B ⊔ C ⊔ D))
axiom cond2 : ∀ (G : Shape), is_partition_five G = true → (G = (A ⊔ B ⊔ C ⊔ D ⊔ E))
axiom allow_rotation : ∀ (F G : Shape), is_partition_four F = true ∧ is_partition_five G = true → is_rotation F G

-- Main theorem statement
theorem exists_shape partition_shape : ∃ (F : Shape), (is_partition_four F = true) ∧ (is_partition_five F = true) := 
by 
  sorry

end exists_shape_l593_593102


namespace min_value_trig_expression_l593_593531

-- Define the problem statement and goal
theorem min_value_trig_expression (x : ℝ) (h1 : 0 < x ∧ x < π / 2) :
  ∃ y > 1, y = 1 / cos x ∧ (8 / (sin x) + 1 / (cos x)) = 10 := 
sorry

end min_value_trig_expression_l593_593531


namespace total_cows_l593_593424

theorem total_cows (n : ℕ) 
  (h₁ : n / 3 + n / 6 + n / 9 + 8 = n) : n = 144 :=
by sorry

end total_cows_l593_593424


namespace find_m_l593_593191

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l593_593191


namespace T_10_equals_2_pow_45_l593_593601

-- Define the geometric sequence and arithmetic sequence conditions
def geom_sequence (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ n, a (n + 1) = q * a n

def arith_seq (b c d : ℝ) : Prop := 2 * c = b + d

-- Define the initial conditions
def a : ℕ → ℝ := λ n, 2^(n - 1)

theorem T_10_equals_2_pow_45 : 
  geom_sequence a ∧ 
  a 1 = 1 ∧ 
  arith_seq (4 * a 3) (2 * a 4) (a 5) ∧ 
  T n = ∏ i in fin.range n, a i →
  T 10 = 2^45 :=
by
  -- We state the theorem assumptions
  intro h1 h2 h3 h4,
  -- State proof here.
  sorry

end T_10_equals_2_pow_45_l593_593601


namespace eleven_pow_2010_mod_19_l593_593719

theorem eleven_pow_2010_mod_19 : (11 ^ 2010) % 19 = 3 := sorry

end eleven_pow_2010_mod_19_l593_593719


namespace num_lines_one_intersection_l593_593433

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define a line passing through a given point
def line_through_point (m c x : ℝ) : ℝ := m * (x - sqrt 2) + c

-- Prove there are exactly 3 lines passing through the point (√2, 0) and having exactly one common point with the hyperbola x^2 - y^2 = 2
theorem num_lines_one_intersection : 
  ∃ lines : Finset (ℝ × ℝ), 
    lines.card = 3 ∧ 
    ∀ (x y : ℝ), (x,y) ∈ lines → hyperbola x y ∧ 
    ∀ m c : ℝ, (∃ y : ℝ, line_through_point m c y = 0) :=
sorry

end num_lines_one_intersection_l593_593433


namespace sin_315_degree_l593_593044

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593044


namespace floor_factorial_expression_l593_593794

theorem floor_factorial_expression : 
  (Int.floor ((Nat.factorial 2010 - Nat.factorial 2007) / 
              (Nat.factorial 2009 - Nat.factorial 2008))) = 2010 :=
by sorry

end floor_factorial_expression_l593_593794


namespace find_a3_l593_593214

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a3
  (a : ℕ → α) (q : α)
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 2)
  (h_cond : a 3 * a 5 = 4 * (a 6) ^ 2) :
  a 3 = 1 :=
by
  sorry

end find_a3_l593_593214


namespace union_sets_l593_593126

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }

def setB : Set ℝ := { x | x^2 - 7 * x + 10 ≤ 0 }

theorem union_sets : setA ∪ setB = { x | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_sets_l593_593126


namespace tan_theta_neg_one_third_sin_sum_seven_fifths_l593_593194

open Real

-- Define θ as an angle in the second quadrant and the given condition
variable (θ : ℝ)
variable h1 : π / 2 < θ
variable h2 : θ < π
variable h3 : tan (θ + π / 4) = 1 / 2

-- Problem 1: Prove that tan θ = -1 / 3
theorem tan_theta_neg_one_third : tan θ = -1 / 3 :=
sorry

-- Define more conditions for problem 2
variable h4 : tan θ = -1 / 3

-- Problem 2: Prove that sin (π / 2 - 2 * θ) + sin (π + 2 * θ) = 7 / 5
theorem sin_sum_seven_fifths : sin (π / 2 - 2 * θ) + sin (π + 2 * θ) = 7 / 5 :=
sorry

end tan_theta_neg_one_third_sin_sum_seven_fifths_l593_593194


namespace speedster_convertibles_proof_l593_593732

-- Definitions based on conditions
def total_inventory (T : ℕ) : Prop := 2 / 3 * T = 2 / 3 * T
def not_speedsters (T : ℕ) : Prop := 1 / 3 * T = 60
def speedsters (T : ℕ) (S : ℕ) : Prop := S = 2 / 3 * T
def speedster_convertibles (S : ℕ) (C : ℕ) : Prop := C = 4 / 5 * S

theorem speedster_convertibles_proof (T S C : ℕ) (hT : total_inventory T) (hNS : not_speedsters T) (hS : speedsters T S) (hSC : speedster_convertibles S C) : C = 96 :=
by
  -- Proof goes here
  sorry

end speedster_convertibles_proof_l593_593732


namespace find_a_if_f_is_odd_l593_593840

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f(x)

def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x - abs a

theorem find_a_if_f_is_odd (a : ℝ) : is_odd_function (f a) → a = 0 := 
by
  intro h
  have h0 : f a 0 = 0 := by sorry
  have ha : |a| = 0 := by sorry
  have h1 : a = 0 := by sorry
  exact h1

end find_a_if_f_is_odd_l593_593840


namespace sum_of_digits_div_by_7_impossible_sum_of_digits_1_l593_593390

open Nat

theorem sum_of_digits_div_by_7 (n : ℕ) (h : n ≥ 2) : 
  ∃ (m : ℕ), (m % 7 = 0) ∧ (sum_digits m = n) :=
  sorry

theorem impossible_sum_of_digits_1 (m : ℕ) : 
  sum_digits m = 1 → m % 7 ≠ 0 :=
  sorry

end sum_of_digits_div_by_7_impossible_sum_of_digits_1_l593_593390


namespace line_intersects_circle_l593_593950

noncomputable def point_A_polar : ℝ × ℝ := (real.sqrt 2, real.pi / 4)

def line_polar_eq (ρ θ : ℝ) (a : ℝ) : Prop :=
  ρ * real.cos (θ - real.pi / 4) = a

def cartesian_eq_of_line (x y : ℝ) : Prop :=
  x + y - 2 = 0

def parametric_circle (α : ℝ) : ℝ × ℝ :=
  (1 + real.cos α, real.sin α)

def cartesian_eq_of_circle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + y ^ 2 = 1

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem line_intersects_circle :
  let A := point_A_polar in
  let a := real.sqrt 2 in
  line_polar_eq A.1 A.2 a ∧
  ∀ (x y : ℝ), cartesian_eq_of_line x y →
    ∀ (α : ℝ), (x, y) = parametric_circle α →
      distance (1, 0) (0, 1) < 1 →
        ∃ p, cartesian_eq_of_line p.1 p.2 ∧ cartesian_eq_of_circle p.1 p.2 :=
by
  sorry

end line_intersects_circle_l593_593950


namespace number_of_cars_in_train_l593_593787

theorem number_of_cars_in_train
  (constant_speed : Prop)
  (cars_in_12_seconds : ℕ)
  (time_to_clear : ℕ)
  (cars_per_second : ℕ → ℕ → ℚ)
  (total_time_seconds : ℕ) :
  cars_in_12_seconds = 8 →
  time_to_clear = 180 →
  cars_per_second cars_in_12_seconds 12 = 2 / 3 →
  total_time_seconds = 180 →
  cars_per_second cars_in_12_seconds 12 * total_time_seconds = 120 :=
by
  sorry

end number_of_cars_in_train_l593_593787


namespace problem_part1_problem_part2_l593_593888

-- Statement for Part (1)
theorem problem_part1 (a b : ℝ) (h_sol : {x : ℝ | ax^2 - 3 * x + 2 > 0} = {x : ℝ | x < 1 ∨ x > 2}) :
  a = 1 ∧ b = 2 := sorry

-- Statement for Part (2)
theorem problem_part2 (x y k : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) 
  (h_eq : (1 / x) + (2 / y) = 1) (h_ineq : 2 * x + y ≥ k^2 + k + 2) :
  -3 ≤ k ∧ k ≤ 2 := sorry

end problem_part1_problem_part2_l593_593888


namespace series_sum_eq_1_over_200_l593_593084

-- Definition of the nth term of the series
def nth_term (n : ℕ) : ℝ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

-- Definition of the series sum
def series_sum : ℝ :=
  ∑' n, nth_term n

-- Final proposition to prove the sum of the series is 1/200
theorem series_sum_eq_1_over_200 : series_sum = 1 / 200 :=
  sorry

end series_sum_eq_1_over_200_l593_593084


namespace multiples_between_1_and_300_l593_593229

theorem multiples_between_1_and_300 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 300 ∧ (n % 6 = 0) ∧ (n % 3 ≠ 0) ∧ (n % 8 ≠ 0) ∧ (n % 10 ≠ 0) → false :=
begin
  sorry,
end

end multiples_between_1_and_300_l593_593229


namespace angle_preservation_under_inversion_l593_593784

-- Define the intersecting circles and their properties
structure Circle (α : Type*) :=
(center : α) (radius : ℝ)

-- Given conditions
variables {α : Type*} [euclidean_space α]

def intersection_point (O1 O2 : Circle α) : α := sorry
def external_tangent_point (O : Circle α) : α × α := sorry

def midpoint (P Q : α) : α := sorry

-- Given intersecting points and external tangents
variables (O1 O2 : Circle α) (A : α)
(O1_ext1 O1_ext2 : external_tangent_point O1) 
(O2_ext1 O2_ext2 : external_tangent_point O2)

-- Midpoints of the external tangents
def M1 : α := midpoint O1_ext1.fst O2_ext1.fst
def M2 : α := midpoint O1_ext2.fst O2_ext2.fst

-- Given angles
def angle (P Q R : α) : ℝ := sorry

-- Proof statement
theorem angle_preservation_under_inversion (O1 O2 : Circle α) (A : α)
(O1_ext1 O1_ext2 : external_tangent_point O1)
(O2_ext1 O2_ext2 : external_tangent_point O2)
(M1 : α := midpoint O1_ext1.fst O2_ext1.fst)
(M2 : α := midpoint O1_ext2.fst O2_ext2.fst) :
  angle O1.center A O2.center = angle M1 A M2 := sorry

end angle_preservation_under_inversion_l593_593784


namespace sin_315_eq_neg_sqrt2_div_2_l593_593005

theorem sin_315_eq_neg_sqrt2_div_2 :
  Real.sin (315 * Real.pi / 180) = - (Real.sqrt 2 / 2) :=
by
  have h1 : Real.sin (360 * Real.pi / 180 - 45 * Real.pi / 180) = - Real.sin (45 * Real.pi / 180),
  by exact Real.sin_sub_pi_div_4_pi_div_4
  rw [← Real.sin_sub, h1]
  exact Real.sin_pi_div_4
sory

end sin_315_eq_neg_sqrt2_div_2_l593_593005


namespace number_of_beavers_in_first_group_l593_593342

theorem number_of_beavers_in_first_group
    (B : ℕ) : (B * 8 = 36 * 4) → B = 18 :=
by
  intro h
  rw h
  sorry

end number_of_beavers_in_first_group_l593_593342


namespace g_difference_l593_593305

def g (n : ℕ) : ℚ :=
  1/4 * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (r : ℕ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_difference_l593_593305


namespace remaining_lemon_heads_after_eating_l593_593973

-- Assume initial number of lemon heads is given
variables (initial_lemon_heads : ℕ)

-- Patricia eats 15 lemon heads
def remaining_lemon_heads (initial_lemon_heads : ℕ) : ℕ :=
  initial_lemon_heads - 15

theorem remaining_lemon_heads_after_eating :
  ∀ (initial_lemon_heads : ℕ), remaining_lemon_heads initial_lemon_heads = initial_lemon_heads - 15 :=
by
  intros
  rfl

end remaining_lemon_heads_after_eating_l593_593973


namespace sin_315_degree_is_neg_sqrt_2_div_2_l593_593026

-- Define the required angles and sine function
def θ := 315
def sin_315 : Real := Real.sin (θ * Real.pi / 180)

-- Define the expected result
def expected_value : Real := - (Real.sqrt 2 / 2)

-- The theorem to be proven
theorem sin_315_degree_is_neg_sqrt_2_div_2 : sin_315 = expected_value := by
  -- Proof is omitted
  sorry

end sin_315_degree_is_neg_sqrt_2_div_2_l593_593026


namespace squares_equal_l593_593677

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l593_593677


namespace elena_fraction_left_l593_593810

variable (M : ℝ) -- Total amount of money
variable (B : ℝ) -- Total cost of all the books

-- Condition: Elena spends one-third of her money to buy half of the books
def condition : Prop := (1 / 3) * M = (1 / 2) * B

-- Goal: Fraction of the money left after buying all the books is one-third
theorem elena_fraction_left (h : condition M B) : (M - B) / M = 1 / 3 :=
by
  sorry

end elena_fraction_left_l593_593810


namespace shaded_area_l593_593386

def rectangle_area (length width : ℝ) : ℝ := length * width

def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem shaded_area
  (length_r1 width_r1 : ℝ) (length_r2 width_r2 : ℝ)
  (leg_t1 leg_t2 : ℝ)
  (overlap_r : ℝ) (overlap_t1 : ℝ) (overlap_t2 : ℝ) :
  length_r1 = 4 → width_r1 = 12 →
  length_r2 = 5 → width_r2 = 7 →
  leg_t1 = 3 → leg_t2 = 3 →
  overlap_r = 12 → overlap_t1 = 2 → overlap_t2 = 1 →
  rectangle_area length_r1 width_r1 + rectangle_area length_r2 width_r2 + triangle_area leg_t1 leg_t2 - overlap_r - overlap_t1 - overlap_t2 = 72.5 :=
by {
  intros,
  simp [rectangle_area, triangle_area],
  rw [h, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8],
  norm_num,
  exact 72.5,
}

end shaded_area_l593_593386


namespace triangle_ratio_l593_593261

variables (X Y Z W : Type)
variables (XY YZ : X → ℝ) (ratio_XY_YZ rw : ℝ) (parallel_XY_Z : Prop)

/-- Given problem statement in Lean 4 -/
theorem triangle_ratio (h1: ratio_XY_YZ = 5 / 3) 
                       (h2: parallel_XY_Z)
                       (h3: ∀ (X : ℝ), ∃ W, W = ratio_XY_YZ * X)
                       (h4: XY Y = 5 * Z)
                       (h5: YZ W = 3 * Z) :
  WX / XZ = 2 := sorry

end triangle_ratio_l593_593261


namespace find_m_l593_593178

-- Define the conditions
variables {m : ℕ}

-- Define a theorem stating the equivalent proof problem
theorem find_m (h1 : m > 0) (h2 : Nat.lcm 40 m = 120) (h3 : Nat.lcm m 45 = 180) : m = 12 :=
sorry

end find_m_l593_593178


namespace universal_quantifiers_are_true_l593_593782

-- Declare the conditions as hypotheses
theorem universal_quantifiers_are_true :
  (∀ x : ℝ, x^2 - x + 0.25 ≥ 0) ∧ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry -- Proof skipped

end universal_quantifiers_are_true_l593_593782


namespace part1_part2_l593_593881

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - a / x - 5 * Real.log x
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := x * x - m * x + 4

theorem part1 (a : ℝ) (ha : a > 5/2) : ∀ x : ℝ, 0 < x → (g a)' x > 0 := 
sorry

theorem part2 (m : ℝ) (a_eq_2 : 2 = 2) : 
  (∃ x₁ ∈ Ioo 0 1, ∀ x₂ ∈ Icc 1 2, g 2 x₁ ≥ h m x₂) → m ≥ 8 - 5 * Real.log 2 :=
sorry

end part1_part2_l593_593881


namespace complex_cubed_minus_two_eq_neg_one_l593_593865

theorem complex_cubed_minus_two_eq_neg_one :
  let ω := - (1/2 : ℂ) + (√3/2 : ℂ) * Complex.I in
  ω^3 - 2 = -1 :=
by
  sorry

end complex_cubed_minus_two_eq_neg_one_l593_593865


namespace ellipse_area_limit_l593_593769

noncomputable def sum_ellipse_areas (a : ℝ) : ℝ :=
  (∀ n : ℕ, ∃ r s : ℝ, r = a / (2^((n : ℝ)/2)) ∧ s = a / (2^(1 + (n : ℝ)/2)) ∧ ∑ i in finset.range n, π * r * s) → π * a^2

theorem ellipse_area_limit (a : ℝ) : 
  tendsto (λ n, ∑ i in finset.range n, π * (a / (2^((i : ℝ)/2))) * (a / (2^(1 + (i : ℝ)/2)))) at_top (𝓝 (π * a^2)) :=
sorry

end ellipse_area_limit_l593_593769


namespace mara_marbles_l593_593994

theorem mara_marbles :
  ∃ (x : ℕ), (12 * x + 2 = 26) ∧ (x = 2) :=
by
  use 2
  -- Proof omitted
  sorry

end mara_marbles_l593_593994


namespace directrix_of_parabola_l593_593821

def parabola_directrix (x_y_eqn : ℝ → ℝ) : ℝ := by
  -- Assuming the parabola equation x = -(1/4) y^2
  sorry

theorem directrix_of_parabola : parabola_directrix (fun y => -(1/4) * y^2) = 1 := by
  sorry

end directrix_of_parabola_l593_593821


namespace prime_dates_2009_correct_l593_593932

open Nat

def prime_dates_2009 : Nat :=
  let february_prime_days : Set Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23}
  let march_prime_days : Set Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
  let may_prime_days : Set Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
  let july_prime_days : Set Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
  let november_prime_days : Set Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  february_prime_days.size + 
  march_prime_days.size + 
  may_prime_days.size + 
  july_prime_days.size + 
  november_prime_days.size

theorem prime_dates_2009_correct : prime_dates_2009 = 52 := by
  -- The proof is omitted as per the instruction, here we put sorry.
  sorry

end prime_dates_2009_correct_l593_593932


namespace max_diff_distance_l593_593619

def hyperbola_right_branch (x y : ℝ) : Prop := 
  (x^2 / 9) - (y^2 / 16) = 1 ∧ x > 0

def circle_1 (x y : ℝ) : Prop := 
  (x + 5)^2 + y^2 = 4

def circle_2 (x y : ℝ) : Prop := 
  (x - 5)^2 + y^2 = 1

theorem max_diff_distance 
  (P M N : ℝ × ℝ) 
  (hp : hyperbola_right_branch P.fst P.snd) 
  (hm : circle_1 M.fst M.snd) 
  (hn : circle_2 N.fst N.snd) :
  |dist P M - dist P N| ≤ 9 := 
sorry

end max_diff_distance_l593_593619


namespace FrankReadingTime_l593_593516

theorem FrankReadingTime :
  ∀ (num_chapters total_pages pages_per_day : ℕ),
    num_chapters = 41 →
    total_pages = 450 →
    pages_per_day = 15 →
    (total_pages / pages_per_day) = 30 := 
by
  intros num_chapters total_pages pages_per_day h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end FrankReadingTime_l593_593516


namespace tom_marbles_no_red_black_l593_593382

-- Definition of the marbles and the restriction on pairs
def marbles := ['red, 'green, 'blue, 'black] ++ (List.repeat 'yellow 4)
def distinct_colors := ['red, 'green, 'blue, 'black]

def choose_two_distinct_pairs (p1 p2 : Char) : Bool :=
  p1 ≠ p2 ∧ (p1, p2) ≠ ('red, 'black) ∧ (p2, p1) ≠ ('red, 'black)

def total_pairs : Nat :=
  -- Number of ways to choose two yellow marbles (identical)
  let yellow_pairs := 1
  -- Number of ways to choose two distinct colored marbles except (red, black)
  let distinct_colored_pairs := (combinations 2 distinct_colors).filter (λ pair => choose_two_distinct_pairs pair.1 pair.2).length
  -- Number of ways to choose one colored marble and one yellow marble
  let colored_and_yellow_pairs := distinct_colors.length
  yellow_pairs + distinct_colored_pairs + colored_and_yellow_pairs

theorem tom_marbles_no_red_black : total_pairs = 10 := by
  sorry

end tom_marbles_no_red_black_l593_593382


namespace range_of_g_l593_593111

noncomputable def g (x : ℝ) : ℝ := (cos x)^3 + 5 * (cos x)^2 + 3 * (cos x) + 2 * (1 - (cos x)^2) + 1 / (cos x + 2)

theorem range_of_g : 
  ∀ x : ℝ, cos x ≠ -2 → ∃ y ∈ set.Icc (0 : ℝ) (2 : ℝ), g x = y :=
sorry

end range_of_g_l593_593111


namespace units_digit_power_of_5_l593_593723

theorem units_digit_power_of_5 (n : ℕ) : (5^n % 10) = 5 := by
  sorry

# Testing the statement with the specific case n = 10
example : (5^10 % 10) = 5 := units_digit_power_of_5 10

end units_digit_power_of_5_l593_593723


namespace ratio_m_n_bounds_l593_593124

theorem ratio_m_n_bounds (n m : ℕ)
  (h1 : ∀ (a b : Fin 5), a ≠ b → ∃ k : Fin n, ∀ i : Fin n, (¬ ∀ j : Fin 5, a i j = b i j))
  (h2 : ∃ (a : Fin 5 → Fin n → Fin 2), ∀ i j, k : Fin n, (a i j).1 = (a b k).1 → i = j ∧ k = b) :
  (2/5 : ℚ) ≤ (m/n : ℚ) ∧ (m/n : ℚ) ≤ (3/5 : ℚ) :=
sorry

end ratio_m_n_bounds_l593_593124


namespace proof_of_problem_l593_593479

open set real

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → (1 / sqrt (1 - x) - 1 / sqrt (1 + x) ≥ 1 → sqrt (2 * sqrt 3 - 3) ≤ x ∧ x < 1)

theorem proof_of_problem : problem_statement := sorry

end proof_of_problem_l593_593479


namespace find_S11_l593_593302

variable {a : ℕ → ℕ}

def arithmetic_sum (n : ℕ) : ℕ := n * (a 1 + a n) / 2

axiom a_5_eq_5 (h : a 5 + a 6 + a 7 = 15) : a 6 = 5

theorem find_S11 (h : a 5 + a 6 + a 7 = 15) : arithmetic_sum 11 = 55 :=
sorry

end find_S11_l593_593302


namespace place_pawns_distinct_5x5_l593_593135

noncomputable def number_of_ways_place_pawns : ℕ :=
  5 * 4 * 3 * 2 * 1 * 120

theorem place_pawns_distinct_5x5 : number_of_ways_place_pawns = 14400 := by
  sorry

end place_pawns_distinct_5x5_l593_593135


namespace image_finite_or_countable_continuous_property_constant_l593_593976

noncomputable def interval_property_constant {a b : ℝ} (f : ℝ → ℝ) (a_x b_x : ℝ) (h : a < a_x ∧ a_x ≤ b_x ∧ b_x < b) : 
  ∀ x ∈ Ioc a b, f x = f (a_x + (b_x - a_x) / 2) := sorry

theorem image_finite_or_countable {a b : ℝ} (f : ℝ → ℝ) (h : ∀ x ∈ Ioc a b, ∃ a_x b_x : ℝ, a < a_x ∧ a_x ≤ x ∧ x ≤ b_x ∧ b_x < b 
  ∧ ∀ y ∈ Icc a_x b_x, f y = f x) : ∃ (s : set ℝ), (finite s ∨ countable s) ∧ (set_of (λ y, ∃ x ∈ Ioc a b, f x = y) = s) := sorry

theorem continuous_property_constant {a b : ℝ} (f : ℝ → ℝ) (h_cont : continuous f) (h : ∀ x ∈ Ioc a b, ∃ a_x b_x : ℝ, a < a_x ∧ a_x ≤ x ∧ x ≤ b_x ∧ b_x < b 
  ∧ ∀ y ∈ Icc a_x b_x, f y = f x) : ∃ c : ℝ, ∀ x ∈ Ioc a b, f x = c := sorry

end image_finite_or_countable_continuous_property_constant_l593_593976


namespace largest_prime_factor_of_sum_of_divisors_of_450_l593_593981

open Nat

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in divisors n, d

theorem largest_prime_factor_of_sum_of_divisors_of_450 : 
  let M := sum_of_divisors 450 in
  prime M :=
by
  sorry

end largest_prime_factor_of_sum_of_divisors_of_450_l593_593981


namespace tan_sum_formula_l593_593129

theorem tan_sum_formula (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) : 
  Real.tan (α + β) = -1 := by
sorry

end tan_sum_formula_l593_593129


namespace inequality_sin_x_div_x_squared_le_sin_x_div_x_le_sin_x2_div_x2_l593_593394

theorem inequality_sin_x_div_x_squared_le_sin_x_div_x_le_sin_x2_div_x2
  (x : ℝ) (hx : 0 < x ∧ x ≤ 1) : 
  let f := λ x, sin x / x in
  (f x)^2 < f x ∧ f x ≤ sin (x^2) / (x^2) :=
by
  -- the proof goes here
  sorry

end inequality_sin_x_div_x_squared_le_sin_x_div_x_le_sin_x2_div_x2_l593_593394


namespace number_of_girls_l593_593572

theorem number_of_girls (total_students boys girls : ℕ)
  (h1 : boys = 300)
  (h2 : (girls : ℝ) = 0.6 * total_students)
  (h3 : (boys : ℝ) = 0.4 * total_students) : 
  girls = 450 := by
  sorry

end number_of_girls_l593_593572


namespace exist_equal_disjoint_unions_l593_593613

theorem exist_equal_disjoint_unions (n : ℕ) (h : 0 < n) (A : Fin (n+1) → Set (Fin n))
  (hA : ∀ i, (A i).Nonempty) :
  ∃ (X Y : Fin (n+1) → Prop), (X ∩ Y = ∅) ∧ (∃ i, X i) ∧ (∃ j, Y j) ∧ 
  (⋃ (i : Fin (n+1)) (hi : X i), A i) = (⋃ (j : Fin (n+1)) (hj : Y j), A j) :=
by
  sorry

end exist_equal_disjoint_unions_l593_593613


namespace metal_contest_winner_l593_593493

theorem metal_contest_winner (x y : ℕ) (hx : 95 * x + 74 * y = 2831) : x = 15 ∧ y = 19 ∧ 95 * 15 > 74 * 19 := by
  sorry

end metal_contest_winner_l593_593493


namespace rhombus_area_correct_l593_593735

-- Define the lengths of the diagonals
def d1 : ℝ := 14
def d2 : ℝ := 20

-- Define the formula for the area of a rhombus
def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

-- Define the theorem stating the area of the rhombus
theorem rhombus_area_correct : rhombus_area d1 d2 = 140 := by
  sorry

end rhombus_area_correct_l593_593735


namespace intersection_of_A_and_B_l593_593900

open Set

variable {α : Type*} [LinearOrder α] [ArchimedeanOrderedAddCommGroup α]

def A : Set α := {x | 2^x > 1}
def B : Set α := {x | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 1} :=
by {
  sorry
}

end intersection_of_A_and_B_l593_593900


namespace find_m_l593_593158

open Nat  

-- Definition of lcm in Lean, if it's not already provided in Mathlib
def lcm (a b : Nat) : Nat := (a * b) / gcd a b

theorem find_m (m : ℕ) (h1 : 0 < m)
    (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180):
    m = 60 :=
sorry

end find_m_l593_593158


namespace percentage_of_laborers_present_l593_593582

theorem percentage_of_laborers_present (total_laborers present_laborers : ℕ)
  (h1 : total_laborers = 26) (h2 : present_laborers = 10) :
  (Real.round ((present_laborers / total_laborers : ℝ) * 100 * 10) / 10 = 38.5) :=
by
  rw [h1, h2]
  sorry

end percentage_of_laborers_present_l593_593582


namespace overall_average_score_l593_593262

def students_monday := 24
def students_tuesday := 4
def total_students := 28
def mean_score_monday := 82
def mean_score_tuesday := 90

theorem overall_average_score :
  (students_monday * mean_score_monday + students_tuesday * mean_score_tuesday) / total_students = 83 := by
sorry

end overall_average_score_l593_593262


namespace sin_315_degree_l593_593047

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593047


namespace domain_shift_l593_593520

noncomputable def domain := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }
noncomputable def shifted_domain := { x : ℝ | 2 ≤ x ∧ x ≤ 5 }

theorem domain_shift (f : ℝ → ℝ) (h : ∀ x, x ∈ domain ↔ (1 ≤ x ∧ x ≤ 4)) :
  ∀ x, x ∈ shifted_domain ↔ ∃ y, (y = x - 1) ∧ y ∈ domain :=
by
  sorry

end domain_shift_l593_593520


namespace weight_of_3_moles_l593_593910

noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_H : ℝ := 1.008
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_C6H8O7 : ℝ :=
  6 * atomic_mass_C + 8 * atomic_mass_H + 7 * atomic_mass_O

theorem weight_of_3_moles 
  (compound_molar_mass : ℝ := 6 * atomic_mass_C + 8 * atomic_mass_H + 7 * atomic_mass_O)
  (given_total_weight : ℝ := 576) :
  3 * compound_molar_mass ≈ given_total_weight :=
by sorry

end weight_of_3_moles_l593_593910


namespace find_m_l593_593171

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l593_593171


namespace perfect_square_expression_l593_593123

theorem perfect_square_expression (n : ℕ) : ∃ t : ℕ, n^2 - 4 * n + 11 = t^2 ↔ n = 5 :=
by
  sorry

end perfect_square_expression_l593_593123


namespace area_of_rhombus_l593_593754

open Real

theorem area_of_rhombus (s : ℝ) (A B C D : EuclideanSpace ℝ (Fin 3)) :
  A = (0, 0, 0) ∧ C = (s, s, s) ∧ 
  B = (0, s / 3, s) ∧ D = (s, 2 * s / 3, 0) →
  let d1 := dist A C in
  let d2 := dist B D in
  (Area_of_quadrilateral A B C D = s^2 * sqrt 33 / 6) := 
by {
  sorry
} 

end area_of_rhombus_l593_593754


namespace squares_equal_l593_593676

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l593_593676


namespace total_number_of_coins_l593_593377

theorem total_number_of_coins (x n : Nat) (h1 : 15 * 5 = 75) (h2 : 125 - 75 = 50)
  (h3 : x = 50 / 2) (h4 : n = x + 15) : n = 40 := by
  sorry

end total_number_of_coins_l593_593377


namespace pizzeria_large_pizzas_l593_593364

theorem pizzeria_large_pizzas (price_small : ℕ) (price_large : ℕ) (total_revenue : ℕ) (small_pizzas_sold : ℕ) (L : ℕ) 
    (h1 : price_small = 2) 
    (h2 : price_large = 8) 
    (h3 : total_revenue = 40) 
    (h4 : small_pizzas_sold = 8) 
    (h5 : price_small * small_pizzas_sold + price_large * L = total_revenue) :
    L = 3 := 
by 
  -- Lean will expect a proof here; add sorry for now
  sorry

end pizzeria_large_pizzas_l593_593364


namespace trigonometric_unique_solution_l593_593240

theorem trigonometric_unique_solution :
  (∃ x : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < (π / 2) ∧ 0 ≤ y ∧ y < (π / 2) ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 ∧
    Real.sin y = 0.6 ∧ Real.cos y = 0.8 → x = y) :=
by
  sorry

end trigonometric_unique_solution_l593_593240


namespace katie_more_games_l593_593298

noncomputable def katie_games : ℕ := 57 + 39
noncomputable def friends_games : ℕ := 34
noncomputable def games_difference : ℕ := katie_games - friends_games

theorem katie_more_games : games_difference = 62 :=
by
  -- Proof omitted
  sorry

end katie_more_games_l593_593298


namespace smallest_positive_period_of_f_l593_593806

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem smallest_positive_period_of_f : ∃ T > 0, (∀ x : ℝ, f(x + T) = f(x)) ∧ (∀ T' > 0, (∀ x : ℝ, f(x + T') = f(x)) → T ≤ T') := by
  use Real.pi
  sorry

end smallest_positive_period_of_f_l593_593806


namespace sin_315_degree_l593_593045

theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_degree_l593_593045


namespace find_b_if_continuous_l593_593991

def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 3 then 3*x^2 + 5 else b*x - 6

theorem find_b_if_continuous (b : ℝ) : (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 3) < δ → abs (f x b - f 3 b) < ε) → b = 38 / 3 := by
  sorry

end find_b_if_continuous_l593_593991


namespace find_m_l593_593167

open Nat

theorem find_m (m : ℕ) (h1 : lcm 40 m = 120) (h2 : lcm m 45 = 180) : m = 60 := 
by
  sorry  -- Proof goes here

end find_m_l593_593167


namespace fixed_point_pq_passes_through_T_l593_593299

-- Definitions for points and the problem setup
variables {A B C H P E F Q T : Type}
variables [Field ℝ] -- Assume working in a coordinate system over real numbers.

-- Convert conditions to variables in Lean
variables a acute_triangle : Prop
noncomputable def A : ℝ × ℝ := ⟨1, 0⟩ -- Coordinates of point A
noncomputable def B : ℝ × ℝ := ⟨b, 0⟩ -- Coordinates of point B
noncomputable def C : ℝ × ℝ := ⟨c, 0⟩ -- Coordinates of point C
noncomputable def H : ℝ × ℝ := ⟨0, 0⟩ -- Coordinates of point H
noncomputable def I : ℝ × ℝ := ⟨0, t⟩ -- Incenter I of triangle PBC

-- Given conditions
axiom cond1 : acute_triangle
axiom cond2 : H = ⟨0, 0⟩
-- Angle bisectors (k) and (l) meet on line AH
axiom cond3 : P ∈ ℝ × ℝ → True -- Variable point P with angle bisectors meeting on AH
axiom cond4 : E ∈ ℝ × ℝ → True -- Point E
axiom cond5 : F ∈ ℝ × ℝ → True -- Point F
axiom cond6 : Q ∈ ℝ × ℝ → True -- Point Q where EF meets AH

-- Fixed point T that line PQ passes through as P varies
noncomputable def fixed_point_T : ℝ × ℝ := ⟨(b + c) / (bc + 1), (2 * b * c) / (bc + 1)⟩

-- Theorem statement in Lean to prove PQ passes through fixed point T
theorem fixed_point_pq_passes_through_T (P Q : ℝ × ℝ) (cond1 : acute_triangle) (cond2 : H = ⟨0, 0⟩)
  (cond3 : P ∈ ℝ × ℝ → True) (cond4 : E ∈ ℝ × ℝ → True)
  (cond5 : F ∈ ℝ × ℝ → True) (cond6 : Q ∈ ℝ × ℝ → True) :
  ∀ P : ℝ × ℝ, ∃ T: ℝ × ℝ, line_passing_through PQ T :=
sorry

end fixed_point_pq_passes_through_T_l593_593299


namespace necessary_and_sufficient_l593_593328

noncomputable def div_cond (f x : ℕ → ℕ) (k : ℕ) : Prop :=
  (x ∣ (f^[k]) x) ↔ (x ∣ f x)

theorem necessary_and_sufficient (f : ℕ → ℕ) (x : ℕ) (k : ℕ) : 
  div_cond f x k := 
by 
  sorry

end necessary_and_sufficient_l593_593328


namespace y_intercept_is_correct_l593_593205

noncomputable def curve (x : ℝ) (n : ℕ) : ℝ := (1 - x) * x^n

noncomputable def tangent_y_intercept (n : ℕ) : ℝ :=
  let y := curve (1/2) n in
  let y' := (n - 1) * (1/2)^n in
  y - y' * (1/2)

theorem y_intercept_is_correct (n : ℕ) (h_pos : 0 < n) :
  tangent_y_intercept n = (2 - n) * (1/2)^(n+1) := 
sorry

end y_intercept_is_correct_l593_593205


namespace find_m_l593_593152

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l593_593152


namespace sin_A_eq_4_div_5_l593_593938

variable (A B C : Type) [RightTriangle ABC]
variables (AB BC : ℝ)
variable (h1 : ∠BAC = 90)
variable (h2 : AB = 15)
variable (h3 : BC = 20)

theorem sin_A_eq_4_div_5 :
  ∃ AC : ℝ, AC = Real.sqrt (AB^2 + BC^2) ∧ 
  ∃ sinA : ℝ, sinA = BC / AC ∧ sinA = 4 / 5 :=
by
  sorry

end sin_A_eq_4_div_5_l593_593938


namespace find_m_l593_593180

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l593_593180


namespace students_failed_is_correct_l593_593593

-- Define the total number of students
def total_students : ℕ := 700

-- Define the passing percentage
def passing_percentage : ℝ := 0.35

-- Define the failing percentage
def failing_percentage : ℝ := 1 - passing_percentage

-- Define the number of students who failed
def students_failed : ℕ := failing_percentage * total_students

-- Theorem representation of the problem
theorem students_failed_is_correct : students_failed = 455 :=
by
  -- This is where the proof will go
  sorry

end students_failed_is_correct_l593_593593


namespace term_34_l593_593219

def sequence (n : Nat) : Real :=
  if n = 0 then 1   -- Since Lean uses zero-based indexing, the third term corresponds to a_34.
  else if n = 1 then 1
  else sequence (n - 1) / (3 * sequence (n - 1) + 1)

theorem term_34 : sequence 34 = 1 / 100 :=
by
  sorry

end term_34_l593_593219


namespace possible_case_l593_593926

-- Define the logical propositions P and Q
variables (P Q : Prop)

-- State the conditions given in the problem
axiom h1 : P ∨ Q     -- P ∨ Q is true
axiom h2 : ¬ (P ∧ Q) -- P ∧ Q is false

-- Formulate the proof problem in Lean
theorem possible_case : P ∧ ¬Q :=
by
  sorry -- Proof to be filled in later

end possible_case_l593_593926


namespace lowest_point_on_graph_l593_593820

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2 * x + 2) / (x + 1)

theorem lowest_point_on_graph : ∃ (x y : ℝ), x = 0 ∧ y = 2 ∧ ∀ z > -1, f z ≥ y ∧ f x = y := by
  sorry

end lowest_point_on_graph_l593_593820


namespace sum_of_numbers_in_grid_l593_593585

-- The universe in which our sets of numbers and grid cells exist.
universe u

-- Define the type for the grid cells.
structure Grid (α : Type u) :=
  (cells : Fin 5 → Fin 5 → α)

-- Define a type for our numbers.
inductive Num
  | one : Num
  | two : Num
  | three : Num
  | four : Num

-- A function that assigns a sum value to each number type.
def value : Num → ℕ
  | Num.one   => 1
  | Num.two   => 2
  | Num.three => 3
  | Num.four  => 4

-- Assumption: The grid conditions and connections are abstracted out.

-- Final Lean statement
theorem sum_of_numbers_in_grid (G : Grid Num) (cond : ∀ i j, ∃ k : Num, G.cells i j = k ∧ Num → Prop) : 
  (∑ i j, value (G.cells i j) ) = 66 :=
by
  sorry

end sum_of_numbers_in_grid_l593_593585


namespace max_distinct_three_digit_numbers_l593_593267

theorem max_distinct_three_digit_numbers :
  let count := ∑ z in {3, 4, 5, 6, 7, 8, 9}, 10 in
  count = 70 :=
by
  sorry

end max_distinct_three_digit_numbers_l593_593267
