import Mathlib

namespace f_60_value_l379_379569

noncomputable def f : ℕ → ℕ := sorry

axiom increasing : ∀ n : ℕ, (n > 0) → f(n + 1) > f(n)
axiom multiplicative : ∀ m n : ℕ, (m > 0 ∧ n > 0) → f(m * n) = f(m) * f(n)
axiom unique_solution : ∀ m n : ℕ, (m ≠ n ∧ m^n = n^m) → (f(m) = n ∨ f(n) = m)

theorem f_60_value : f 60 = 3600 := sorry

end f_60_value_l379_379569


namespace identify_pyramid_scheme_l379_379648

-- Definitions for the individual conditions
def high_returns (investment_opportunity : Prop) : Prop := 
  ∃ significantly_higher_than_average_returns : Prop, investment_opportunity = significantly_higher_than_average_returns

def lack_of_information (company : Prop) : Prop := 
  ∃ incomplete_information : Prop, company = incomplete_information

def aggressive_advertising (advertising : Prop) : Prop := 
  ∃ aggressive_ad : Prop, advertising = aggressive_ad

-- Main definition combining all conditions
def is_financial_pyramid_scheme (investment_opportunity company advertising : Prop) : Prop :=
  high_returns investment_opportunity ∧ lack_of_information company ∧ aggressive_advertising advertising

-- Theorem statement
theorem identify_pyramid_scheme 
  (investment_opportunity company advertising : Prop) 
  (h1 : high_returns investment_opportunity)
  (h2 : lack_of_information company)
  (h3 : aggressive_advertising advertising) : 
  is_financial_pyramid_scheme investment_opportunity company advertising :=
by 
  apply And.intro;
  {
    exact h1,
    apply And.intro;
    {
      exact h2,
      exact h3,
    }
  }

end identify_pyramid_scheme_l379_379648


namespace same_tangent_conditions_at_zero_monotonic_intervals_F_max_ab_value_l379_379493

-- Condition definitions
def f (x: ℝ) := Real.exp x
def g (a b x: ℝ) := a * x + b
def h (a b x: ℝ) := x * (g a b x) + 1
def F (a b x: ℝ) := (h a b x) / (f x)

-- Prove the conditions for same tangent at x = 0
theorem same_tangent_conditions_at_zero (a b : ℝ) (h : a ≠ 0) : 
  (h (a := a) (b := b) 0 = f 0) ∧ (h' (a := a) (b := b) 0 = f' 0) → b = 1 :=
sorry

-- Determine the intervals of monotonicity for F(x)
theorem monotonic_intervals_F (b: ℝ) : 
  let F' := λ x =>  (- x^2 + (2 - b) * x + (b - 1)) / (Real.exp x) in 
    if b > 0 then 
      decreasing_on (λ x, F (1 b x)) a b x )
      increasing_here (λ x, F (1 b x)) 
    else if b == 0 then 
      decreasing_on (λ x, F (1 b x))
    else 
      decreasing_on (λ x, F (1 b x))
      increasing_here (λ x, F (1 b x))
:= sorry

-- Prove the maximum value of ab is e/2
theorem max_ab_value (a : ℝ) (b : ℝ) : 
  (∀ x ∈ ℝ, f x ≥ g a b x) → ab ≤ Real.exp 1 / 2 :=
sorry

end same_tangent_conditions_at_zero_monotonic_intervals_F_max_ab_value_l379_379493


namespace range_of_m_l379_379136

theorem range_of_m (x y m : ℝ) 
    (h_circle : (x + 2)^2 + (y - m)^2 = 3)
    (h_chord : 3 - ((x + 2)^2 + (y - m)^2) = x^2 + y^2) :
    -real.sqrt 2 ≤ m ∧ m ≤ real.sqrt 2 :=
by
  -- this is where the proof would go, but we just state the theorem
  sorry

end range_of_m_l379_379136


namespace divides_f_then_divides_n_l379_379572

-- Given conditions
variables {ℕ : Type} [Nat.Units] 
variable (f : ℕ → ℕ)
variable (a b : ℕ)
hypothesis rel_prime_f_a_f_b : Nat.Coprime a b → Nat.Coprime (f a) (f b)
hypothesis bounded_f : ∀ n, n ≤ f n ∧ f n ≤ n + 2012

-- Prove statement
theorem divides_f_then_divides_n (n p : ℕ) (hp : Nat.Prime p) (hpdivf : p ∣ f n) : p ∣ n :=
by
  sorry

end divides_f_then_divides_n_l379_379572


namespace rational_not_integral_solution_l379_379623

theorem rational_not_integral_solution : 
  ∀ (x : ℝ), 
  (∀ R H : ℝ, R = 8 ∧ H = 3) → 
  ∀ (V : ℝ → ℝ → ℝ), 
  (V = fun R H =>  π * R^2 * H) →
  ((π * (8 + x)^2 * 3 = π * 8^2 * (3 + x)) ↔ (x = 16 / 3)) → 
  (∃ (y : ℚ), x = y ∧ x ≠ ⌊x⌋) := 
begin
  intro x,
  intro h_rh,
  intro V,
  intro h_V,
  intro h_equiv,
  use (16 / 3 : ℝ),
  split,
  {
    -- Proof part: x = 16 / 3
    sorry,
  },
  {
    -- Proof part: Show x is rational but not integral
    sorry,
  },
end

end rational_not_integral_solution_l379_379623


namespace max_value_f_not_symmetry_axis_center_of_symmetry_monotonic_decreasing_l379_379821

-- Define the vectors m and n, and the function f(x)
def m : ℝ × ℝ := (real.sqrt 3, 1)
def n (x : ℝ) : ℝ × ℝ := (real.cos (2 * x), real.sin (2 * x))
def f (x : ℝ) : ℝ := (real.sqrt 3) * real.cos (2 * x) + real.sin (2 * x)

-- The maximum value of f(x) is 2
theorem max_value_f : ∀ x : ℝ, f x ≤ 2 := by sorry

-- The line x = -π/12 is not a symmetry axis of the graph of f(x)
theorem not_symmetry_axis : ¬ (∀ x : ℝ, f x = f (-x - real.pi / 12)) := by sorry

-- The point (π/3, 0) is a center of symmetry of the graph of f(x)
theorem center_of_symmetry : ∀ x : ℝ, f (π / 3 - x) = - f (π / 3 + x) := by sorry

-- f(x) is monotonically decreasing on (π/12, 7π/12)
theorem monotonic_decreasing : ∀ x : ℝ, (π / 12 < x ∧ x < 7 * π / 12) → f' x < 0 := by sorry

end max_value_f_not_symmetry_axis_center_of_symmetry_monotonic_decreasing_l379_379821


namespace larger_section_volume_l379_379418

-- Definition of points and midpoints in the prism
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (3, 0, 0)
def E : (ℝ × ℝ × ℝ) := (0, 3, 0)
def G : (ℝ × ℝ × ℝ) := (3, 3, 1)
def F : (ℝ × ℝ × ℝ) := (0, 3, 1)

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def P := midpoint B G
def Q := midpoint E F

-- Volume of the larger section
theorem larger_section_volume : 
  let vol := 7 / 2 in 
  vol = 7/2 :=
sorry

end larger_section_volume_l379_379418


namespace range_of_omega_l379_379790

theorem range_of_omega 
  (ω : ℝ) 
  (hω : 0 < ω) 
  : (∀ x : ℝ, (π / 2 < x) ∧ (x < π) → (fderiv ℝ (λ x, sin (ω * x + π / 4)) x) < 0) ↔ (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) := by
  sorry

end range_of_omega_l379_379790


namespace sum_even_integers_102_to_200_l379_379972

theorem sum_even_integers_102_to_200 : 
  (finset.sum (finset.filter (λ n, n % 2 = 0) (finset.range' 102 200.succ))) = 7550 :=
sorry

end sum_even_integers_102_to_200_l379_379972


namespace problem_conditions_l379_379564

noncomputable def f (a b c x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem problem_conditions (a b c : ℝ) (h0 : a + b + c = 0)
  (h1 : f a b c 0 > 0) (h2 : f a b c 1 > 0) :
    (a > 0 ∧ -2 < b / a ∧ b / a < -1) ∧
    (∃ z1 z2 : ℝ, 0 < z1 ∧ z1 < 1 ∧ 0 < z2 ∧ z2 < 1 ∧ z1 ≠ z2 ∧ f a b c z1 = 0 ∧ f a b c z2 = 0) :=
by
  sorry

end problem_conditions_l379_379564


namespace algorithm_combination_l379_379631

def algorithm_structure (seq cond loop : Prop) : Prop :=
  seq ∧ (cond ∨ ¬ cond) ∧ (loop ∨ ¬ loop)

theorem algorithm_combination :
  ∀ (seq cond loop : Prop),
    seq → (cond ∨ ¬ cond) → (loop ∨ ¬ loop) →
    (algorithm_structure seq cond loop ↔ (True)) :=
begin
  intros seq cond loop hseq hcond hloop,
  unfold algorithm_structure,
  split,
  { intro h,
    exact trivial },
  { intro h,
    exact ⟨hseq, hcond, hloop⟩ },
end

end algorithm_combination_l379_379631


namespace arithmetic_sequence_length_l379_379826

theorem arithmetic_sequence_length :
  ∀ (a d l : ℕ), a = 4 → d = 3 → l = 205 → 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 68 := 
by
  intros a d l ha hd hl
  use 68
  rw [ha, hd, hl]
  simp
  sorry

end arithmetic_sequence_length_l379_379826


namespace boat_travel_distance_upstream_l379_379522

variable (D : ℝ)  -- Distance traveled upstream in km
variable (u : ℝ) := 6  -- Speed of boat in still water in km/hr
variable (v : ℝ) := 2  -- Speed of river in km/hr
variable (T : ℝ) := 21  -- Total journey time in hours

theorem boat_travel_distance_upstream :
  (D / (u - v) + D / (u + v) = T) → D = 56 :=
by
  intros h
  sorry

end boat_travel_distance_upstream_l379_379522


namespace evaluate_expression_l379_379746

theorem evaluate_expression (b : ℤ) (x : ℤ) (h : x = b + 9) : (x - b + 5 = 14) :=
by
  sorry

end evaluate_expression_l379_379746


namespace symmetric_point_x_axis_l379_379537

variable (P : (ℝ × ℝ)) (x : ℝ) (y : ℝ)

-- Given P is a point (x, y)
def symmetric_about_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

-- Special case for the point (-2, 3)
theorem symmetric_point_x_axis : 
  symmetric_about_x_axis (-2, 3) = (-2, -3) :=
by 
  sorry

end symmetric_point_x_axis_l379_379537


namespace maximum_take_home_pay_l379_379851

def tax_rate (x : ℝ) : ℝ := (x + 5) / 100
def income (x : ℝ) : ℝ := 1000 * x
def tax (x : ℝ) : ℝ := income x * tax_rate x
def take_home_pay (x : ℝ) : ℝ := income x - tax x

theorem maximum_take_home_pay : 
  ∃ x : ℝ, x = 47.5 ∧ ∀ y : ℝ, take_home_pay y ≤ take_home_pay x :=
by
  sorry

end maximum_take_home_pay_l379_379851


namespace schoolchildren_lineup_l379_379308

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379308


namespace fall_semester_length_l379_379945

theorem fall_semester_length (h_weekdays : ∀ d : ℕ, d ∈ {1, 2, 3, 4, 5} → Paris_studies 3 hours_on d)
                            (h_weekend_saturday : Paris_studies 4 hours_on 6)
                            (h_weekend_sunday : Paris_studies 5 hours_on 7)
                            (h_total_hours : Paris_studies 360 total_hours) :
                            fall_semester_last_for_weeks = 15 :=
by
  sorry

end fall_semester_length_l379_379945


namespace simplify_expression_l379_379202

theorem simplify_expression :
  (sqrt 300 / sqrt 75 - sqrt 128 / sqrt 32) = 0 := 
begin
  sorry
end

end simplify_expression_l379_379202


namespace min_value_expr_l379_379561

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 := 
sorry

end min_value_expr_l379_379561


namespace compute_km_l379_379095

namespace VectorProof

variables (m k : ℝ)

def a : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (k, 1)

def are_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1
def are_perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem compute_km : 
  are_parallel a b ∧ are_perpendicular a c → k * m = -1 :=
by
  sorry

end VectorProof

end compute_km_l379_379095


namespace cabbages_produced_l379_379693

theorem cabbages_produced (x y : ℕ) (h1 : y = x + 1) (h2 : x^2 + 199 = y^2) : y^2 = 10000 :=
by
  sorry

end cabbages_produced_l379_379693


namespace who_is_next_to_Denis_l379_379258

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379258


namespace solve_quadratic1_solve_quadratic2_l379_379927

theorem solve_quadratic1 :
  (∀ x, x^2 + x - 4 = 0 → x = ( -1 + Real.sqrt 17 ) / 2 ∨ x = ( -1 - Real.sqrt 17 ) / 2) := sorry

theorem solve_quadratic2 :
  (∀ x, (2*x + 1)^2 + 15 = 8*(2*x + 1) → x = 1 ∨ x = 2) := sorry

end solve_quadratic1_solve_quadratic2_l379_379927


namespace circle_equation_with_segment_PQ_as_diameter_l379_379818

theorem circle_equation_with_segment_PQ_as_diameter :
    ∃ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 5 :=
by
  let P := (4 : ℝ, 0 : ℝ)
  let Q := (0 : ℝ, 2 : ℝ)
  let mid := ( (P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let r := ( (P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt / 2
  have h_mid : mid = (2, 1) := by sorry
  have h_r : r = (5).sqrt := by sorry
  use [2, 1]
  simp [h_mid, h_r]
  sorry

end circle_equation_with_segment_PQ_as_diameter_l379_379818


namespace relationship_between_a_b_c_l379_379562

theorem relationship_between_a_b_c (a b c : ℝ) (ha : a = 2^(-1 / 2)) (hb : b = (1 / 2)^(-2)) (hc : c = log 10 (1 / 2)) : b > a ∧ a > c :=
by
  -- Definitions from the problem
  have ha' : a = 1 / real.sqrt 2 := by sorry
  have hb' : b = 4 := by sorry
  have hc' : c < 0 := by sorry
  -- Show that the relationship holds
  have h1 : 4 > 1 / real.sqrt 2 := by sorry
  have h2 : 1 / real.sqrt 2 > c := by sorry
  -- Combining both parts
  exact ⟨h1, h2⟩

end relationship_between_a_b_c_l379_379562


namespace layoffs_rounds_l379_379689

theorem layoffs_rounds (n : ℕ) : 
  ∃ n : ℕ, 1000 - 1000 * (0.9)^n = 271 :=
begin
  use 4,
  sorry
end

end layoffs_rounds_l379_379689


namespace find_ABC_l379_379440

theorem find_ABC :
  ∃ A B C : ℝ, 
    (∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 → 
      -x^2 + 3*x - 4 = A*x^2 + A + B*x^2 + C*x) ∧
    A = -4 ∧ B = 3 ∧ C = 3 :=
by
  sorry

end find_ABC_l379_379440


namespace initial_typists_count_l379_379111

theorem initial_typists_count
  (letters_per_20_min : Nat)
  (letters_total_1_hour : Nat)
  (letters_typists_count : Nat)
  (n_typists_init : Nat)
  (h1 : letters_per_20_min = 46)
  (h2 : letters_typists_count = 30)
  (h3 : letters_total_1_hour = 207) :
  n_typists_init = 20 :=
by {
  sorry
}

end initial_typists_count_l379_379111


namespace even_integers_count_form_3k_plus_4_l379_379499

theorem even_integers_count_form_3k_plus_4 
  (n : ℕ) (h1 : 20 ≤ n ∧ n ≤ 250)
  (h2 : ∃ k : ℕ, n = 3 * k + 4 ∧ Even n) : 
  ∃ N : ℕ, N = 39 :=
by {
  sorry
}

end even_integers_count_form_3k_plus_4_l379_379499


namespace pyramid_scheme_indicator_l379_379645

def financial_pyramid_scheme_indicator (high_return lack_full_information aggressive_advertising : Prop) : Prop :=
  high_return ∧ lack_full_information ∧ aggressive_advertising

theorem pyramid_scheme_indicator
  (high_return : Prop)
  (lack_full_information : Prop)
  (aggressive_advertising : Prop)
  (indicator : financial_pyramid_scheme_indicator high_return lack_full_information aggressive_advertising) :
  indicator = (high_return ∧ lack_full_information ∧ aggressive_advertising) :=
sorry

end pyramid_scheme_indicator_l379_379645


namespace inscribed_circle_radius_l379_379780

theorem inscribed_circle_radius :
  ∀ (r : ℝ), 
    (∀ (R : ℝ), R = 12 →
      (∀ (d : ℝ), d = 12 → r = 3)) :=
by sorry

end inscribed_circle_radius_l379_379780


namespace james_average_speed_l379_379876

-- Define the conditions
def hour2_distance : ℝ := 18
def hour2_increase_percent : ℝ := 0.20
def hour3_increase_percent : ℝ := 0.25
def total_time_hours : ℝ := 3

-- Derived data from conditions
noncomputable def hour1_distance : ℝ := hour2_distance / (1 + hour2_increase_percent)
noncomputable def hour3_distance : ℝ := hour2_distance * (1 + hour3_increase_percent)
noncomputable def total_distance : ℝ := hour1_distance + hour2_distance + hour3_distance
noncomputable def average_speed : ℝ := total_distance / total_time_hours

-- The proof statement
theorem james_average_speed : average_speed = 18.5 := by
  -- Define the assumptions and compute the values manually
  have h1 : hour1_distance = 15 := by sorry
  have h2 : hour3_distance = 22.5 := by sorry
  have h3 : total_distance = 55.5 := by sorry
  have h4 : average_speed = 18.5 := by sorry
  show average_speed = 18.5 from h4

end james_average_speed_l379_379876


namespace find_functions_and_monotonicity_l379_379481

-- Define the given conditions and the functions
def linear_function (f : ℝ → ℝ) := ∃ a b : ℝ, a ≠ 0 ∧  ∀ x : ℝ, f(x) = a * x + b
def inverse_proportion_function (g : ℝ → ℝ) := ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, g(x) = k / x

variables (f g h : ℝ → ℝ)

-- State the problem conditions
axiom f_linear : linear_function f
axiom g_inverse : inverse_proportion_function g
axiom f_f_eq : ∀ x : ℝ, f(f(x)) = x + 2
axiom g_at_one : g(1) = -1

-- Define h(x) = f(x) + g(x)
def h (x : ℝ) := f(x) + g(x)

-- Prove the correct answers
theorem find_functions_and_monotonicity :
  (f = (λ x, x + 1) ∧ g = (λ x, -1 / x)) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x1 < x2 → h(x1) < h(x2)) :=
by {
  sorry
}

end find_functions_and_monotonicity_l379_379481


namespace randy_wipes_days_l379_379595

theorem randy_wipes_days (wipes_per_pack : ℕ) (packs_needed : ℕ) (wipes_per_walk : ℕ) (walks_per_day : ℕ) (total_wipes : ℕ) (wipes_per_day : ℕ) (days_needed : ℕ) 
(h1 : wipes_per_pack = 120)
(h2 : packs_needed = 6)
(h3 : wipes_per_walk = 4)
(h4 : walks_per_day = 2)
(h5 : total_wipes = packs_needed * wipes_per_pack)
(h6 : wipes_per_day = wipes_per_walk * walks_per_day)
(h7 : days_needed = total_wipes / wipes_per_day) : 
days_needed = 90 :=
by sorry

end randy_wipes_days_l379_379595


namespace Diana_friends_count_l379_379007

theorem Diana_friends_count (totalErasers : ℕ) (erasersPerFriend : ℕ) 
  (h1: totalErasers = 3840) (h2: erasersPerFriend = 80) : 
  totalErasers / erasersPerFriend = 48 := 
by 
  sorry

end Diana_friends_count_l379_379007


namespace geometric_description_l379_379425

-- Define the condition
def quadratic_condition (x y : ℝ) :=
  ∀ t : ℝ, -1 ≤ t ∧ t ≤ 1 → t^2 + y * t + x ≥ 0

-- State the geometric description
theorem geometric_description (x y : ℝ) :
  (quadratic_condition x y) ↔ 
  (y ∈ (-2 : ℝ)..2 ∧ x ≥ y^2 / 4) ∨ (|y| ≥ 2 ∧ x ≥ y - 1 ∧ x ≥ -y - 1) :=
sorry

end geometric_description_l379_379425


namespace sum_even_integers_correct_l379_379965

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end sum_even_integers_correct_l379_379965


namespace positive_integers_expressible_l379_379004

theorem positive_integers_expressible :
  ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ (x^2 + y) / (x * y + 1) = 1 ∧
  ∃ (x' y' : ℕ), (x' > 0) ∧ (y' > 0) ∧ (x' ≠ x ∨ y' ≠ y) ∧ (x'^2 + y') / (x' * y' + 1) = 1 :=
by
  sorry

end positive_integers_expressible_l379_379004


namespace zoo_animal_difference_l379_379677

variable (giraffes non_giraffes : ℕ)

theorem zoo_animal_difference (h1 : giraffes = 300) (h2 : giraffes = 3 * non_giraffes) : giraffes - non_giraffes = 200 :=
by 
  sorry

end zoo_animal_difference_l379_379677


namespace vector_addition_subtraction_identity_l379_379204

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (BC AB AC : V)

theorem vector_addition_subtraction_identity : BC + AB - AC = 0 := 
by sorry

end vector_addition_subtraction_identity_l379_379204


namespace equilateral_triangle_inscribed_inequality_l379_379628

theorem equilateral_triangle_inscribed_inequality
  (a b c : ℝ) (S : ℝ) (DE : ℝ)
  (h_DE : DE = 
           (2 * real.sqrt 2 * S) / 
           real.sqrt (a^2 + b^2 + c^2 + 4 * real.sqrt 3 * S)) :
  DE >= (2 * real.sqrt 2 * S) / 
        real.sqrt (a^2 + b^2 + c^2 + 4 * real.sqrt 3 * S) :=
by sorry

end equilateral_triangle_inscribed_inequality_l379_379628


namespace range_of_a_l379_379840

theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 2)^x₁ > (a - 2)^x₂) → (2 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l379_379840


namespace sum_of_symmetrical_roots_l379_379904

theorem sum_of_symmetrical_roots {f : ℝ → ℝ} 
  (h_symm : ∀ x, f (3 + x) = f (3 - x))
  (h_roots : ∃ S : Finset ℝ, S.card = 6 ∧ ∀ x ∈ S, f x = 0) :
  (∃ S : Finset ℝ, S.card = 6 ∧ (∀ x ∈ S, f x = 0) ∧ S.sum id = 18) :=
begin
  sorry
end

end sum_of_symmetrical_roots_l379_379904


namespace num_divisors_not_divisible_by_2_of_360_l379_379098

def is_divisor (n d : ℕ) : Prop := d ∣ n

def is_prime (p : ℕ) : Prop := Nat.Prime p

noncomputable def prime_factors (n : ℕ) : List ℕ := sorry -- To be implemented if needed

def count_divisors_not_divisible_by_2 (n : ℕ) : ℕ :=
  let factors : List ℕ := prime_factors 360
  let a := 0
  let b_choices := [0, 1, 2]
  let c_choices := [0, 1]
  (b_choices.length) * (c_choices.length)

theorem num_divisors_not_divisible_by_2_of_360 :
  count_divisors_not_divisible_by_2 360 = 6 :=
by sorry

end num_divisors_not_divisible_by_2_of_360_l379_379098


namespace max_p_division_l379_379352

theorem max_p_division (p : ℕ) (a : ℕ → ℕ)
  (h1 : ∀ i, a i % 2 = 0)
  (h2 : ∀ i j, i < j → a i < a j)
  (h3 : ∀ i, a i ≤ 18)
  (h4 : ∀ sum_i, sum_i = ∑ i in finset.range p, a i → sum_i = 32) : p ≤ 6 := sorry

end max_p_division_l379_379352


namespace find_cube_edge_length_l379_379367

-- Define parameters based on the problem conditions
def is_solution (n : ℕ) : Prop :=
  n > 4 ∧
  (6 * (n - 4)^2 = (n - 4)^3)

-- The main theorem statement
theorem find_cube_edge_length : ∃ n : ℕ, is_solution n ∧ n = 10 :=
by
  use 10
  sorry

end find_cube_edge_length_l379_379367


namespace trailing_zeros_50_l379_379635

theorem trailing_zeros_50! : (nat.factorial 50).trailingZeroes = 12 := 
by 
  -- the proof goes here
  sorry

end trailing_zeros_50_l379_379635


namespace matrix_multiplication_correct_l379_379416

-- Definitions of the matrices
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1], ![-3, 4]]

def B : Matrix (Fin 2) Unit ℤ :=
  ![![3], ![-1]]

-- Expected result
def C : Matrix (Fin 2) Unit ℤ :=
  ![![7], ![-13]]

-- Statement of the proof problem
theorem matrix_multiplication_correct :
  A.mul B = C := by
  sorry

end matrix_multiplication_correct_l379_379416


namespace angle_between_a_b_parallel_c_ab_l379_379040

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

-- Define the vectors
def vec_a : ℝ × ℝ := (-3, 1)
def vec_b : ℝ × ℝ := (1, -2)
def vec_c : ℝ × ℝ := (1, 1)

-- Define the problems to prove
theorem angle_between_a_b : 
  let dot_product := (-3) * 1 + 1 * (-2)
  let norm_a := Real.sqrt ((-3) ^ 2 + 1 ^ 2)
  let norm_b := Real.sqrt (1 ^ 2 + (-2) ^ 2)
  let cos_theta := dot_product / (norm_a * norm_b)
  cos_theta = -Real.sqrt 2 / 2 → 
  Real.arccos cos_theta = 3 * Real.pi / 4 :=
sorry

theorem parallel_c_ab (h : vec_c = vec_a + k * vec_b) : k = 4 / 3 :=
sorry

end angle_between_a_b_parallel_c_ab_l379_379040


namespace rhombus_angle_bcd_l379_379540

-- Define the rhombus and the properties it must satisfy
structure Rhombus (A B C D : Type) :=
(equal_sides : ∀ (x y : Type), x = y)

-- Define a theorem stating the angle to be proven
theorem rhombus_angle_bcd (A B C D : Type) [Rhombus A B C D] : is_rhombus A B C D → angle A C D = 120 :=
by
  sorry

end rhombus_angle_bcd_l379_379540


namespace differential_eq_solution_l379_379343

theorem differential_eq_solution (y : ℝ → ℝ) (y0 : y 0 = 1) :
  (∀ x, (2 * y x - 3) + (derivative y x) * (2 * x + 3 * (y x)^2) = 0) →
  ∃ C : ℝ, (∀ x, 2 * x * y x - 3 * x + (y x)^3 + C = 0) ∧ C = -1 := 
by
  intro exact_diff
  use -1
  sorry

end differential_eq_solution_l379_379343


namespace ordering_y1_y2_y3_l379_379508

-- Conditions
def A (y₁ : ℝ) : Prop := ∃ b : ℝ, y₁ = -4^2 + 2*4 + b
def B (y₂ : ℝ) : Prop := ∃ b : ℝ, y₂ = -(-1)^2 + 2*(-1) + b
def C (y₃ : ℝ) : Prop := ∃ b : ℝ, y₃ = -(1)^2 + 2*1 + b

-- Question translated to a proof problem
theorem ordering_y1_y2_y3 (y₁ y₂ y₃ : ℝ) :
  A y₁ → B y₂ → C y₃ → y₁ < y₂ ∧ y₂ < y₃ :=
sorry

end ordering_y1_y2_y3_l379_379508


namespace floor_of_negative_sqrt_l379_379016

noncomputable def eval_expr : ℚ := -real.sqrt (64 / 9)

theorem floor_of_negative_sqrt : ⌊eval_expr⌋ = -3 :=
by
  -- skip proof
  sorry

end floor_of_negative_sqrt_l379_379016


namespace part_I_part_II_l379_379054

section
variable {a : ℕ → ℕ} (S : ℕ → ℝ)

-- Conditions
def condition1 := a 1 = 2
def condition2 := ∀ n, a (n + 1) = 3 * a n + 2

-- Part (I) to prove the formula and geometric sequence
theorem part_I (h1 : condition1) (h2 : condition2) :
  (∀ n, a n + 1 = 3 ^ n) :=
sorry

-- Expression for S_n
def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (3 ^ (i + 1) : ℝ) / (a (i + 1) * a (i + 2))

-- Part (II) to prove the sum formula
theorem part_II (h1 : condition1) (h2 : condition2) (n : ℕ) :
  S_n n = 1 / 4 - 1 / (2 * (3 ^ (n + 1) - 1)) :=
sorry

end

end part_I_part_II_l379_379054


namespace remaining_pages_l379_379957

theorem remaining_pages (total_pages : ℕ) (science_project_percentage : ℕ) (math_homework_pages : ℕ)
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 25) 
  (h3 : math_homework_pages = 10) : 
  total_pages - (total_pages * science_project_percentage / 100) - math_homework_pages = 80 := by
  sorry

end remaining_pages_l379_379957


namespace problem1_problem2_l379_379069

variables (α : ℝ) (a b c : ℝ × ℝ)

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.sin α, 2 * Real.cos α)
def vector_c (α : ℝ) : ℝ × ℝ := (Real.cos α, -2 * Real.sin α)

theorem problem1 (h1 : α > π) (h2 : α < 2 * π) (h3 : vector_a.1 * (vector_b α).1 + vector_a.2 * (vector_b α).2 = 0) : α = 5 * π / 4 :=
by sorry

theorem problem2 (h1 : α > π) (h2 : α < 2 * π) (h4 : (vector_b α + vector_c α).prod_norm_sq = 3) : Real.sin α + Real.cos α = - Real.sqrt 15 / 3 :=
by sorry

end problem1_problem2_l379_379069


namespace square_side_length_l379_379755

theorem square_side_length(area_sq_cm : ℕ) (h : area_sq_cm = 361) : ∃ side_length : ℕ, side_length ^ 2 = area_sq_cm ∧ side_length = 19 := 
by 
  use 19
  sorry

end square_side_length_l379_379755


namespace bullet_trains_crossing_time_l379_379347

noncomputable def first_train_length : ℝ := 140
noncomputable def second_train_length : ℝ := 200
noncomputable def first_train_speed_kmph : ℝ := 60
noncomputable def second_train_speed_kmph : ℝ := 40
noncomputable def relative_speed_mps : ℝ := (first_train_speed_kmph + second_train_speed_kmph) * (1000 / 3600)
noncomputable def combined_length : ℝ := first_train_length + second_train_length
noncomputable def crossing_time : ℝ := combined_length / relative_speed_mps

theorem bullet_trains_crossing_time : crossing_time ≈ 12.24 := sorry

end bullet_trains_crossing_time_l379_379347


namespace min_translation_l379_379423

noncomputable def det (a1 a2 a3 a4 : ℝ) : ℝ :=
  a1 * a4 - a2 * a3

noncomputable def f (x : ℝ) : ℝ :=
  det (sqrt 3) (cos x) 1 (sin x)

theorem min_translation (m : ℝ) (h : m > 0) :
  (∀ x, f(x + m) = f(-x - m)) → m = 2 * π / 3 :=
by
  sorry

end min_translation_l379_379423


namespace hyperbola_parabola_solution_l379_379477

noncomputable def hyperbola_parabola_focus_equation : Prop :=
  let a := 1 / (Real.sqrt 5)
  let b_squared := (1 : ℝ)^2 - a^2
  let hyperbola_eq := (λ x y : ℝ, 5 * x^2 - (5 / 4) * y^2 = 1)
  let affect : ℝ * ℝ → Prop := λ f, f = (1, 0)
  let ecc := (Real.sqrt 5) = 1 / a
  ∃ f : ℝ × ℝ, affect f ∧ ecc ∧ (hyperbola_eq = (λ x y, 5*x^2 - 5*y^2 / 4 = 1))

theorem hyperbola_parabola_solution : hyperbola_parabola_focus_equation := sorry

end hyperbola_parabola_solution_l379_379477


namespace monotonic_intervals_1_plus_sin_x_monotonic_intervals_neg_cos_x_l379_379447

noncomputable def increasing_intervals_sin (k : ℤ) : Set ℝ :=
  {x | 2 * k * Real.pi - Real.pi / 2 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 2}

noncomputable def decreasing_intervals_sin (k : ℤ) : Set ℝ :=
  {x | 2 * k * Real.pi + Real.pi / 2 ≤ x ∧ x ≤ 2 * k * Real.pi + 3 * Real.pi / 2}

noncomputable def increasing_intervals_cos (k : ℤ) : Set ℝ :=
  {x | 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi}

noncomputable def decreasing_intervals_cos (k : ℤ) : Set ℝ :=
  {x | 2 * k * Real.pi - Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi}

theorem monotonic_intervals_1_plus_sin_x (k : ℤ) :
  ∀ x : ℝ, x ∈ Real :=
  ∀ x : ℝ, (x ∈ increasing_intervals_sin k ∨ x ∈ decreasing_intervals_sin k) := sorry

theorem monotonic_intervals_neg_cos_x (k : ℤ) :
  ∀ x : ℝ, x ∈ Real :=
  ∀ x : ℝ, (x ∈ increasing_intervals_cos k ∨ x ∈ decreasing_intervals_cos k) := sorry

end monotonic_intervals_1_plus_sin_x_monotonic_intervals_neg_cos_x_l379_379447


namespace max_abs_diff_f_l379_379081

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f (k : ℝ) (h₁ : -3 ≤ k) (h₂ : k ≤ -1) (x₁ x₂ : ℝ) (h₃ : k ≤ x₁) (h₄ : x₁ ≤ k + 2) (h₅ : k ≤ x₂) (h₆ : x₂ ≤ k + 2) :
  |f x₁ - f x₂| ≤ 4 * Real.exp 1 := sorry

end max_abs_diff_f_l379_379081


namespace sum_binom_series_l379_379005

variables (n k : ℕ)

theorem sum_binom_series : (∑ i in finset.range (k + 1), nat.choose (n + i) i) = nat.choose (n + k + 1) k :=
sorry

end sum_binom_series_l379_379005


namespace sufficient_condition_l379_379072

variables (a b : ℝ^2)
variables (θ : ℝ)

-- Defining unit vectors and the angle between them
def is_unit_vector (v : ℝ^2) : Prop := 
  ∥v∥ = 1

def angle_between_vectors (v w : ℝ^2) (θ : ℝ) : Prop := 
  ∃ h:θ ∈ [0, π], v ⬝ w = ∥v∥ * ∥w∥ * real.cos θ

-- Propositions p and q
def prop_p (a b : ℝ^2) : Prop :=
  ∥a - b∥ > 1

def prop_q (θ : ℝ) : Prop :=
  θ ∈ Icc (π / 2) (5 * π / 6)

-- Main theorem statement
theorem sufficient_condition (h1: is_unit_vector a) (h2: is_unit_vector b) (h3: angle_between_vectors a b θ) :
  prop_p a b → prop_q θ :=
sorry

end sufficient_condition_l379_379072


namespace who_is_next_to_Denis_l379_379254

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379254


namespace lamp_turnoff_ways_l379_379697

theorem lamp_turnoff_ways :
  ∃ (n : ℕ), n = 35 ∧
  let total_lamps := 11 in
  let turnoff_lamps := 3 in
  let end_lamps_on := 2 in
  let available_positions := total_lamps - end_lamps_on in
  n = Nat.choose (available_positions - 1) turnoff_lamps := 
by
  sorry

end lamp_turnoff_ways_l379_379697


namespace Denis_next_to_Anya_Gena_l379_379263

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379263


namespace locus_of_P_l379_379470

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem locus_of_P (P : ℝ × ℝ)
  (M : (ℝ × ℝ)) (N : (ℝ × ℝ))
  (hM : M = (-5,0)) (hN : N = (5,0))
  (hPMN : distance P M + distance P N = 26) :
  (P : ℝ × ℝ) →
  (P.1, P.2) ∈ { P | P.1^2 / 169 + P.2^2 / 144 = 1 } :=
by {
  sorry -- Proof implementation goes here
}

end locus_of_P_l379_379470


namespace num_mappings_with_condition_l379_379090

open Function

theorem num_mappings_with_condition :
  let P := {x, y, z}
  let Q := {1, 2}
  let f : P → Q := fun _ => 1  -- a dummy default
  (f(y) = 2 → (Finset.card (Finset.univ.filter (λ f : P → Q, f y = 2)) = 4)) :=
by
  sorry

end num_mappings_with_condition_l379_379090


namespace length_of_CD_eq_2_l379_379394

def points_on_sides (A B C D M N: Point) : Prop :=
  on_side M A D ∧ on_side N B C

def ratio_condition (A D B C M N : Point) (AD BC : ℝ) : Prop :=
  AM / AD = 1 / 3 ∧ BN / BC = 1 / 3

def angle_between_conditions (MN AB CD : Line) : Prop :=
  angle_between MN AB = angle_between MN CD

def find_length_CD (A B C D M N : Point) (AB_length : ℝ) (AD_length BC_length : ℝ)
  (h1 : points_on_sides A B C D M N)
  (h2 : ratio_condition A D B C M N AD_length BC_length)
  (h3 : angle_between_conditions (line_through M N) (line_through A B) (line_through C D))
  (h4 : AB_length = 1) : ℝ :=
2

theorem length_of_CD_eq_2
  (A B C D M N : Point)
  (AB_length AD_length BC_length : ℝ)
  (h1 : points_on_sides A B C D M N)
  (h2 : ratio_condition A D B C M N AD_length BC_length)
  (h3 : angle_between_conditions (line_through M N) (line_through A B) (line_through C D))
  (h4 : AB_length = 1) : find_length_CD A B C D M N AB_length AD_length BC_length h1 h2 h3 h4 = 2 :=
sorry

end length_of_CD_eq_2_l379_379394


namespace student_score_70_l379_379383

theorem student_score_70 (total_questions correct_responses : ℕ) :
  total_questions = 100 →
  correct_responses = 90 →
  correct_responses - 2 * (total_questions - correct_responses) = 70 := 
by
  intro h_total h_correct
  rw [h_total, h_correct]
  simp
  -- Proof omitted
  sorry

end student_score_70_l379_379383


namespace polynomials_satisfying_condition_1_l379_379685

noncomputable def P : Polynomial ℝ := sorry

theorem polynomials_satisfying_condition_1 (P : Polynomial ℝ) : 
  (∀ x : ℝ, P (x^2) = (P x)^2) ↔ (P = 0 ∨ ∃ n : ℕ, P = X^n) :=
sorry

end polynomials_satisfying_condition_1_l379_379685


namespace alex_total_marbles_l379_379404

theorem alex_total_marbles
  (total_marbles : ℕ)
  (ratio_b : ℕ) (ratio_a : ℕ) (ratio_j : ℕ)
  (marbles_given : ℕ) 
  (total_ratio : ratio_b + ratio_a + ratio_j = 15)
  (total_marbles_eq : total_marbles = 600)
  (part_value : total_marbles / 15 = 40)
  (initial_b_marbles : ratio_b * (total_marbles / 15) = 120)
  (initial_a_marbles : ratio_a * (total_marbles / 15) = 200)
  (marbles_given_eq : initial_b_marbles / 2 = 60)
  (alex_final_marbles : initial_a_marbles + marbles_given_eq = 260) : 
  true :=
by
  sorry

end alex_total_marbles_l379_379404


namespace maximum_sum_l379_379899

theorem maximum_sum {n : ℕ} (h : n ≥ 2) (a : Fin n → ℝ) (h₀ : ∀ i, 0 < a i ∧ a i < 1) (h₁ : a 0 = a (n - 1)) :
  ∑ i in Finset.range n, Real.root 6 (a i * (1 - a ((i + 1) % n))) ≤ n / Real.root 3 2 :=
sorry

end maximum_sum_l379_379899


namespace part1_solution_part2_solution_l379_379076

-- Definitions based on the given conditions
def eq1 (x y m : ℝ) := 2 * x - y = m
def eq2 (x y m : ℝ) := 3 * x + 2 * y = m + 7

-- Problem Part 1: When m = 0, the solution to the system of equations
theorem part1_solution :
  ∃ (x y : ℝ), eq1 x y 0 ∧ eq2 x y 0 ∧ x = 1 ∧ y = 2 :=
by
  existsi 1
  existsi 2
  apply And.intro
  show eq1 1 2 0, by sorry
  apply And.intro
  show eq2 1 2 0, by sorry
  apply And.intro
  show 1 = 1, by rfl
  show 2 = 2, by rfl

-- Problem Part 2: When A(-2, 3), the value of m that satisfies the equations
theorem part2_solution :
  let x := -2
  let y := 3
  ∃ (m : ℝ), eq1 x y m ∧ m = -7 :=
by
  existsi (-7 : ℝ)
  apply And.intro
  show eq1 (-2) 3 (-7), by sorry
  show -7 = -7, by rfl

end part1_solution_part2_solution_l379_379076


namespace g_16_48_eq_96_l379_379424

-- Definition of the function and its properties
def g (x y : ℕ) : ℕ := sorry

axiom g_cond1 : ∀ x : ℕ, g(x, x) = 2 * x
axiom g_cond2 : ∀ x y : ℕ, g(x, y) = g(y, x)
axiom g_cond3 : ∀ x y : ℕ, (x + y) * g(x, y) = x * g(x, x + y)

theorem g_16_48_eq_96 : g 16 48 = 96 :=
by
  sorry

end g_16_48_eq_96_l379_379424


namespace china_math_competition_1990_l379_379895

theorem china_math_competition_1990 :
  ∀ (E G : Set ℕ),
  E = { x | 1 ≤ x ∧ x ≤ 200 } →
  (∀ (a_i a_j : ℕ), a_i ∈ G → a_j ∈ G → a_i ≠ a_j → a_i + a_j ≠ 201) →
  (∑ i in G, i) = 10080 →
  (∃ (k : ℕ), k % 4 = 0 ∧
  ∃ c : ℕ, ∑ i in G, i^2 = c) :=
by
  sorry

end china_math_competition_1990_l379_379895


namespace identify_pyramid_scheme_l379_379651

-- Definitions based on the given conditions
def high_return : Prop := offers_significantly_higher_than_average_returns
def lack_of_information : Prop := lack_of_complete_information_about_company
def aggressive_advertising : Prop := aggressive_advertising_occurs

-- Defining the predicate 
def is_pyramid_scheme (all_conditions : Prop) : Prop :=
  offers_significantly_higher_than_average_returns ∧
  lack_of_complete_information_about_company ∧
  aggressive_advertising_occurs

-- The main theorem to prove
theorem identify_pyramid_scheme :
  (high_return ∧ lack_of_information ∧ aggressive_advertising) → is_pyramid_scheme (high_return ∧ lack_of_information ∧ aggressive_advertising) :=
by
  intro h
  exact h

end identify_pyramid_scheme_l379_379651


namespace inequality_proof_l379_379459

-- Define the conditions a, b, c ≥ 0 and a + b + c = 1
variables {a b c : ℝ}
hypothesis ha : a ≥ 0
hypothesis hb : b ≥ 0
hypothesis hc : c ≥ 0
hypothesis hsum : a + b + c = 1

-- State the theorem to be proved
theorem inequality_proof :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
by {
  have ha : a ≥ 0 := ha,
  have hb : b ≥ 0 := hb,
  have hc : c ≥ 0 := hc,
  have hsum : a + b + c = 1 := hsum,
  sorry -- Proof goes here
}

end inequality_proof_l379_379459


namespace right_triangle_and_circumradius_l379_379794

theorem right_triangle_and_circumradius (a b c : ℕ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  (a^2 + b^2 = c^2) ∧ (c / 2 = 7.5) :=
by
  sorry

end right_triangle_and_circumradius_l379_379794


namespace next_to_Denis_l379_379274

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379274


namespace bisector_inequality_l379_379723

theorem bisector_inequality
    {B C K L D A: Type}
    (hk : is_orthogonal_projection B D A K)
    (hl : is_orthogonal_projection C D A L)
    (ha : angle_bisector_intersects_circumcircle D A)
    (hbd : length_bisector_AD A D = 2 * cos ((angle B - angle C)/2))
    (hbk : length_BK B K A D = 2 * cos (angle B) * sin ((angle B + angle C)/2))
    (hcl : length_CL C L A D = 2 * cos (angle C) * sin ((angle B + angle C)/2)) :
    length_bisector_AD A D ≥ length_BK B K A D + length_CL C L A D :=
sorry

end bisector_inequality_l379_379723


namespace problem1_problem2_l379_379789

variables (x y : ℝ)

-- Given Conditions
def given_conditions :=
  (x = 2 + Real.sqrt 3) ∧ (y = 2 - Real.sqrt 3)

-- Problem 1
theorem problem1 (h : given_conditions x y) : x^2 + y^2 = 14 :=
sorry

-- Problem 2
theorem problem2 (h : given_conditions x y) : (x / y) - (y / x) = 8 * Real.sqrt 3 :=
sorry

end problem1_problem2_l379_379789


namespace correct_calculation_l379_379665

theorem correct_calculation :
  (∃ (A B C D : Prop),
    A = (sqrt 3 + sqrt 3 = sqrt 6) ∧
    B = (2 - sqrt 2 = sqrt 2) ∧
    C = (sqrt 3 * sqrt 3 = sqrt 6) ∧
    D = (2 / sqrt 2 = sqrt 2) ∧
    D) :=
begin
  use [sqrt 3 + sqrt 3 = sqrt 6,
       2 - sqrt 2 = sqrt 2,
       sqrt 3 * sqrt 3 = sqrt 6,
       2 / sqrt 2 = sqrt 2],
  split,
  { sorry }, -- A
  split,
  { sorry }, -- B
  split,
  { sorry }, -- C
  { sorry }, -- D
end

end correct_calculation_l379_379665


namespace next_to_Denis_l379_379276

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379276


namespace cyclic_quadrilateral_XA_l379_379916

-- Definition of the problem
variables (A B C A' B' C' X : Type*)
variables [triangle A B C] [point_on_side A' B C] [point_on_side B' C A] [point_on_side C' A B]
variables (angle_AXB_eq : ∠(A, X, B) = ∠(A', C', B') + ∠(A, C, B))
variables (angle_BXC_eq : ∠(B, X, C) = ∠(B', A', C') + ∠(B, A, C))

-- Statement to prove
theorem cyclic_quadrilateral_XA'BC : cyclic X A' B C :=
begin
  sorry
end

end cyclic_quadrilateral_XA_l379_379916


namespace inverse_square_variation_l379_379800

theorem inverse_square_variation (x y : ℝ) (k : ℝ) (h1 : 3^2 * 20 = k) (h2 : y = 5000) : x = 3 * sqrt 10 / 50 :=
by
  have h3 : k = 180 := by
    calc
      k = 3^2 * 20 : by rw h1
      ... = 180 : by norm_num
  have h4 : x^2 * y = 180 := by
    rw [h2, h3]
  have h5 : x^2 = 180 / 5000 := by
    rw [h4]
    ring
  have h6 : x = sqrt (180 / 5000) := by
    exact sqrt_eq (180 / 5000)
  calc
    x = sqrt (9 / 250) : by rw h5
    ... = 3 * sqrt 10 / 50 : sorry -- Steps simplified here

-- Final result:
sorry

end inverse_square_variation_l379_379800


namespace find_third_side_l379_379524

theorem find_third_side (a b : ℝ) (c : ℕ) 
  (h1 : a = 3.14)
  (h2 : b = 0.67)
  (h_triangle_ineq : a + b > ↑c ∧ a + ↑c > b ∧ b + ↑c > a) : 
  c = 3 := 
by
  -- Proof goes here
  sorry

end find_third_side_l379_379524


namespace equation_of_circle_l379_379050

theorem equation_of_circle 
  (x y : ℝ) 
  (a b r : ℝ) 
  (h1 : a = b + 1) 
  (h2 : r = abs (4 * a + 3 * b + 14) / 5)
  (h3 : abs (3 * a + 4 * b + 10) / 5 = sqrt (r ^ 2 - 9)) : 
  (x - 2) ^ 2 + (y - 1) ^ 2 = 25 := 
sorry

end equation_of_circle_l379_379050


namespace evaluate_fg_l379_379505

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x - 5

theorem evaluate_fg : f (g 4) = 9 := by
  sorry

end evaluate_fg_l379_379505


namespace merchant_fraud_l379_379530

theorem merchant_fraud (initial_gold initial_pigs half_pigs sold1_pig_for_gold1 sold2_pigs_for_gold timeframe pigs_left : ℕ)
  (gold_from_pigs1 gold_from_pigs2 total_gold pigs_sold1 pigs_sold2 : ℕ)
  (initial_pigs = 500) 
  (initial_gold = 200)
  (half_pigs = initial_pigs / 2) 
  (sold1_pig_for_gold1 = 1) 
  (sold2_pigs_for_gold = 1) 
  (gold_from_pigs1 = 80) 
  (gold_from_pigs2 = 120) 
  (total_gold = initial_gold) 
  (pigs_left = 20) 
  (true_by_fraction : 
    let cost_per_pig := initial_gold / initial_pigs in
    let price_per_pig1 := sold1_pig_for_gold1 / 3 in
    let price_per_pig2 := sold2_pigs_for_gold / 2 in
    let total_pigs_sold := pigs_sold1 + pigs_sold2 in
    let price_for_all_pigs := gold_from_pigs1 + gold_from_pigs2 in
    let avg_price_per_pig := price_for_all_pigs / total_pigs_sold in 
    let initial_correct_price := (2 / 5) in
    avg_price_per_pig < initial_correct_price ):
  (∃ fraud, fraud = true_by_fraction) :=
sorry

end merchant_fraud_l379_379530


namespace arithmetic_sequence_a10_l379_379355

theorem arithmetic_sequence_a10 (a : ℕ → ℝ) 
    (h1 : a 2 = 2) 
    (h2 : a 3 = 4) : 
    a 10 = 18 := 
sorry

end arithmetic_sequence_a10_l379_379355


namespace areas_equal_l379_379161

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c))

theorem areas_equal : 
  area_of_triangle 13 13 10 = area_of_triangle 13 13 24 := by
  sorry

end areas_equal_l379_379161


namespace merchant_profit_condition_l379_379369

theorem merchant_profit_condition (L : ℝ) (P : ℝ) (S : ℝ) (M : ℝ) :
  (P = 0.70 * L) →
  (S = 0.80 * M) →
  (S - P = 0.30 * S) →
  (M = 1.25 * L) := 
by
  intros h1 h2 h3
  sorry

end merchant_profit_condition_l379_379369


namespace vectors_linearly_independent_l379_379922

open Real

def a1 : ℝ × ℝ × ℝ := (2, 1, 3)
def a2 : ℝ × ℝ × ℝ := (1, 0, -1)
def a3 : ℝ × ℝ × ℝ := (0, 0, 1)

theorem vectors_linearly_independent 
  (λ1 λ2 λ3 : ℝ)
  (h : λ1 * a1 + λ2 * a2 + λ3 * a3 = (0, 0, 0)) :
  λ1 = 0 ∧ λ2 = 0 ∧ λ3 = 0 := 
sorry

end vectors_linearly_independent_l379_379922


namespace projection_equivalence_l379_379702

theorem projection_equivalence :
  let v1 := ⟨2, -4⟩ : ℝ²
  let p1 := ⟨1, -1⟩ : ℝ²
  let v2 := ⟨-3, 2⟩ : ℝ²
  let p2 := Proj p1 v2
  Proj p1 v1 = p1 → p2 = ⟨-5/2, 5/2⟩ :=
by
  sorry

end projection_equivalence_l379_379702


namespace second_largest_four_digit_in_pascal_l379_379336

-- Define the conditions about Pascal's Triangle
lemma pascal_triangle_symmetric (n k : ℕ) (h : k ≤ n) :
  binomial n k = binomial n (n - k) :=
by sorry

lemma pascal_row_begin_with_one (n : ℕ) :
  binomial n 0 = 1 :=
by sorry

lemma pascal_binomial_expansion (n k : ℕ) (h : k ≤ n) :
  binomial n k = (n choose k) :=
by sorry

-- Main theorem proving that the second largest four-digit number in Pascal's Triangle is 9998
theorem second_largest_four_digit_in_pascal :
  ∃ k n, k < n ∧ 9998 = binomial n k ∧ binomial n k < 10000 :=
by sorry

end second_largest_four_digit_in_pascal_l379_379336


namespace sum_even_102_to_200_l379_379978

noncomputable def sum_even_integers (a b : ℕ) :=
  let n := (b - a) / 2 + 1
  in (n * (a + b)) / 2

theorem sum_even_102_to_200 :
  sum_even_integers 102 200 = 7550 := 
by
  have n : ℕ := (200 - 102) / 2 + 1
  have sum : ℕ := (n * (102 + 200)) / 2
  have n_50 : n = 50 := by sorry
  have sum_7550 : sum = 7550 := by sorry
  exact sum_7550 

end sum_even_102_to_200_l379_379978


namespace vasya_triangle_rotation_l379_379351

theorem vasya_triangle_rotation :
  (∀ (θ1 θ2 θ3 : ℝ), (12 * θ1 = 360) ∧ (6 * θ2 = 360) ∧ (θ1 + θ2 + θ3 = 180) → ∃ n : ℕ, (n * θ3 = 360) ∧ n ≥ 4) :=
by
  -- The formal proof is omitted, inserting "sorry" to indicate incomplete proof
  sorry

end vasya_triangle_rotation_l379_379351


namespace cube_root_neg_64_l379_379221

theorem cube_root_neg_64 : (∛(-64) = -4) :=
by sorry

end cube_root_neg_64_l379_379221


namespace plane_parallel_line_condition_l379_379473

variables (Plane Line : Type)
variables (α β : Plane) (m : Line)

-- Assuming necessary conditions
axiom distinct_planes (α β : Plane) : α ≠ β
axiom line_in_plane (m : Line) (α : Plane) : m ∈ α
axiom planes_parallel (α β : Plane) : α ∥ β

-- Defining the concept of parallelism for line and plane
axiom line_parallel_plane (m : Line) (β : Plane) : m ∥ β

-- Proof problem statement
theorem plane_parallel_line_condition (h1 : α ≠ β) (h2 : m ∈ α) (h3 : α ∥ β) :
  ∃ (cond : Prop), cond = (α ∥ β) ∧ (α ∥ β → m ∥ β) ∧ ¬(m ∥ β → α ∥ β) := sorry

end plane_parallel_line_condition_l379_379473


namespace train_crossing_time_correct_l379_379142

variable (L : ℕ) (v_km_hr : ℕ) (conversion_factor : ℕ)

-- Given Conditions
def length_of_train := 50
def speed_of_train_kmph := 144
def kmhr_to_ms := 5 / 18

-- Claim to Prove
theorem train_crossing_time_correct :
  (length_of_train / (speed_of_train_kmph * kmhr_to_ms)) = 1.25 := 
sorry

end train_crossing_time_correct_l379_379142


namespace domain_and_range_of_g_l379_379172

-- Define the function f with given domain and range
axiom f : ℝ → ℝ
axiom f_domain : ∀ x, 1 ≤ x ∧ x ≤ 3
axiom f_range : ∀ y, (∃ x, f x = y) ↔ (2 ≤ y ∧ y ≤ 4)

-- Define the function g
def g (x : ℝ) : ℝ := 3 - f (2 * x - 1)

-- Prove that the ordered quadruple (a, b, c, d) for the domain and range of g is (1, 2, -1, 1)
theorem domain_and_range_of_g : (1, 2, -1, 1) = (1, 2, -1, 1) :=
by
  -- Proof not required
  sorry

end domain_and_range_of_g_l379_379172


namespace equation_of_circle_correct_l379_379232

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end equation_of_circle_correct_l379_379232


namespace food_lasts_for_days_l379_379400

-- Define the conditions
def dog_feed_amount : ℕ → ℕ
| 1 := 250
| 2 := 350
| 3 := 450
| 4 := 550
| 5 := 300
| 6 := 400
| _ := 0

def meals_per_day : ℕ := 3
def sack_weight_grams : ℕ := 50000
def number_of_sacks : ℕ := 2

-- Convert total food supply into grams
def total_food_grams : ℕ := number_of_sacks * sack_weight_grams

-- Calculate daily consumption
def daily_consumption : ℕ := meals_per_day * (dog_feed_amount 1 + dog_feed_amount 2 + dog_feed_amount 3 + dog_feed_amount 4 + dog_feed_amount 5 + dog_feed_amount 6)

-- Define the main statement
theorem food_lasts_for_days : total_food_grams / daily_consumption = 14 :=
by
  -- Details of the proof are omitted
  sorry

end food_lasts_for_days_l379_379400


namespace alpha_beta_arrangements_l379_379097

noncomputable def arrangements (total_letters : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial total_letters / List.prod (frequencies.map Nat.factorial)

theorem alpha_beta_arrangements :
  arrangements 9 [4, 1, 1, 1, 1, 1, 1] = 15120 :=
by
  unfold arrangements
  simp [Nat.factorial, List.prod, List.map]
  sorry

end alpha_beta_arrangements_l379_379097


namespace distance_focus_to_line_l379_379943

noncomputable def distance_point_to_line (x1 y1 A B C : ℝ) : ℝ :=
  (abs (A * x1 + B * y1 + C)) / (sqrt (A^2 + B^2))

theorem distance_focus_to_line :
  let x1 := 1
  let y1 := 0
  let A := - (sqrt 3)
  let B := 1
  let C := 0
  distance_point_to_line x1 y1 A B C = sqrt 3 / 2 := 
by
  sorry

end distance_focus_to_line_l379_379943


namespace number_of_valid_words_l379_379498

theorem number_of_valid_words : 
  let letters := ['A', 'B', 'C', 'D', 'E'] in
  ∃ n : ℕ, n = 24 ∧ 
    (∀ word : list char, word.length = 3 ∧ 
      (∀ c ∈ word, c ∈ letters) ∧ 
      ('A' ∈ word ∧ 'B' ∈ word) ↔ true) := 
sorry

end number_of_valid_words_l379_379498


namespace last_two_digits_sum_is_20_l379_379406

theorem last_two_digits_sum_is_20 :
  (7! * 3 + 14! * 3 + 21! * 3 + 28! * 3 + 35! * 3 + 42! * 3 + 49! * 3 + 56! * 3 + 63! * 3 + 70! * 3 + 77! * 3 + 84! * 3 + 91! * 3 + 98! * 3 + 105! * 3) % 100 = 20 :=
by
  /- Proof steps would go here -/
  sorry

end last_two_digits_sum_is_20_l379_379406


namespace soap_use_period_l379_379743

def cost_per_soap : ℝ := 4
def total_spent : ℝ := 96
def days_in_year : ℝ := 365
def total_years : ℝ := 2

theorem soap_use_period :
  let number_of_bars := total_spent / cost_per_soap in
  let total_days := total_years * days_in_year in
  let period_per_bar := total_days / number_of_bars in
  period_per_bar ≈ 30.42 :=
by
  let number_of_bars := total_spent / cost_per_soap
  let total_days := total_years * days_in_year
  let period_per_bar := total_days / number_of_bars
  have : period_per_bar ≈ 30.42 := sorry
  exact this

end soap_use_period_l379_379743


namespace fraction_mistake_l379_379521

theorem fraction_mistake (n : ℕ) (h : n = 288) (student_answer : ℕ) 
(h_student : student_answer = 240) : student_answer / n = 5 / 6 := 
by 
  -- Given that n = 288 and the student's answer = 240;
  -- we need to prove that 240/288 = 5/6
  sorry

end fraction_mistake_l379_379521


namespace paying_students_pay_7_dollars_l379_379398

theorem paying_students_pay_7_dollars
  (total_students : ℕ)
  (free_lunch_percentage : ℝ)
  (total_cost : ℝ)
  (total_students_eq : total_students = 50)
  (free_lunch_percentage_eq : free_lunch_percentage = 0.4)
  (total_cost_eq : total_cost = 210) :
  ( (total_cost / (total_students * (1 - free_lunch_percentage))) = 7 ) :=
by
  have h1 : total_students * (1 - free_lunch_percentage) = 30, 
  { rw [total_students_eq, free_lunch_percentage_eq],
    norm_num },
  have h2 : total_cost / 30 = 7, 
  { rw [total_cost_eq, h1],
    norm_num },
  exact h2

end paying_students_pay_7_dollars_l379_379398


namespace angle_DPO_l379_379141

open Real

theorem angle_DPO (D G O P : Point) 
  (h₁ : ∠ DOG = 50)
  (h₂ : ∠ DGO = ∠ DOG)
  (h₃ : bisector (OP) (∠ DOG)) : ∠ DPO = 75 :=
sorry

end angle_DPO_l379_379141


namespace area_bounded_by_f_g_and_segment_l379_379897

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x < y → f x ≤ f y

def is_continuous (f : ℝ → ℝ) : Prop :=
continuous_on f (set.Icc 0 3)

def is_inverse (f g : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (g x) = x ∧ g (f x) = x

axiom h1 : is_increasing f
axiom h2 : is_continuous f
axiom h3 : is_inverse f g
axiom h4 : ∀ x > 0, g x > f x
axiom h5 : f 0 = 0
axiom h6 : f 3 = 2
axiom h7 : ∫ x in 0..3, f x = 2

theorem area_bounded_by_f_g_and_segment:
  (∫ x in 0..3, f x) + 
  0.5 * abs (3 * 3 + 2 * 0 - 3 * 2) = 4.5 :=
by
  sorry

end area_bounded_by_f_g_and_segment_l379_379897


namespace base_number_is_five_l379_379835

theorem base_number_is_five (x k : ℝ) (h1 : x^k = 5) (h2 : x^(2 * k + 2) = 400) : x = 5 :=
by
  sorry

end base_number_is_five_l379_379835


namespace scheduling_competitions_l379_379000

-- Define the problem conditions
def scheduling_conditions (gyms : ℕ) (sports : ℕ) (max_sports_per_gym : ℕ) : Prop :=
  gyms = 4 ∧ sports = 3 ∧ max_sports_per_gym = 2

-- Define the main statement
theorem scheduling_competitions :
  scheduling_conditions 4 3 2 →
  (number_of_arrangements = 60) :=
by
  sorry

end scheduling_competitions_l379_379000


namespace find_explicit_formula_and_range_l379_379474

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f(x) = a * x^2 + b * x + c

theorem find_explicit_formula_and_range (f : ℝ → ℝ)
  (h_quad : quadratic_function f)
  (h0 : f 0 = 0)
  (h_rec : ∀ x, f (x + 1) = f x + x + 1) :
  (f = (λ x, (1 / 2) * x^2 + (1 / 2) * x)) ∧ (set.range (λ x, f (x^2 - 2)) = set.Ici (-1 / 8)) :=
sorry

end find_explicit_formula_and_range_l379_379474


namespace determine_m_for_value_range_l379_379490

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

theorem determine_m_for_value_range :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end determine_m_for_value_range_l379_379490


namespace nonneg_for_all_x_iff_a_in_range_l379_379750

def f (x a : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

theorem nonneg_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end nonneg_for_all_x_iff_a_in_range_l379_379750


namespace cosine_fourth_minus_sine_fourth_l379_379020

theorem cosine_fourth_minus_sine_fourth (θ : ℝ) : cos (4 * θ) - sin (4 * θ) = cos (2 * 2 * θ) - sin (2 * 2 * θ) := 
begin
    let θ := 15 * (π / 180),
    calc
        cos θ^4 - sin θ^4 = (cos θ^2 + sin θ^2) * (cos θ^2 - sin θ^2) : 
          by rw [cos_pow_two_add_sin_pow_two_eq_one]
        --> := sorry,
end

end cosine_fourth_minus_sine_fourth_l379_379020


namespace proof_problem_l379_379813

-- Given conditions for propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Combined proposition p and q
def p_and_q (a : ℝ) := p a ∧ q a

-- Statement of the proof problem: Prove that p_and_q a → a ≤ -1
theorem proof_problem (a : ℝ) : p_and_q a → (a ≤ -1) :=
by
  sorry

end proof_problem_l379_379813


namespace sum_of_solutions_l379_379109

theorem sum_of_solutions (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := 
by {
  -- missing proof part
  sorry
}

end sum_of_solutions_l379_379109


namespace f_nested_seven_l379_379071

-- Definitions for the given conditions
variables (f : ℝ → ℝ) (odd_f : ∀ x, f (-x) = -f x)
variables (period_f : ∀ x, f (x + 4) = f x)
variables (f_one : f 1 = 4)

theorem f_nested_seven (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = -f x)
  (period_f : ∀ x, f (x + 4) = f x)
  (f_one : f 1 = 4) :
  f (f 7) = 0 :=
sorry

end f_nested_seven_l379_379071


namespace fourth_roots_of_neg_16_l379_379769

theorem fourth_roots_of_neg_16 : 
  { z : ℂ | z^4 = -16 } = { sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I, 
                            sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I } :=
by
  sorry

end fourth_roots_of_neg_16_l379_379769


namespace find_expression_value_l379_379462

theorem find_expression_value (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3 * y^3) / 9 = 73 / 3 :=
by
  sorry

end find_expression_value_l379_379462


namespace cosine_distribution_equality_l379_379885


def uniform_distribution_on_2pi (x : ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x < 2 * Real.pi 

theorem cosine_distribution_equality :
  let varphi psi : ℝ := sorry in
  (uniform_distribution_on_2pi varphi) ∧ (uniform_distribution_on_2pi psi) ∧ (Indep varphi psi) →
  (cos (varphi) + cos (psi))/2 = cos (varphi) * cos (psi) :=
by
  sorry

end cosine_distribution_equality_l379_379885


namespace unit_digit_4137_pow_754_l379_379348

theorem unit_digit_4137_pow_754 : (4137 ^ 754) % 10 = 9 := by
  sorry

end unit_digit_4137_pow_754_l379_379348


namespace denis_neighbors_l379_379298

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

end denis_neighbors_l379_379298


namespace find_f8_minus_f14_l379_379510

-- Given conditions
variables {α : Type*} [ring_hom_class α ℤ ℝ]
variables (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f(x + p) = f(x)

axiom f_is_odd : odd_function f
axiom f_is_periodic : periodic_function f 5
axiom f_one : f 1 = 1
axiom f_two : f 2 = 2

-- Theorem to prove
theorem find_f8_minus_f14 : f(8) - f(14) = -1 :=
by
  sorry

end find_f8_minus_f14_l379_379510


namespace sample_data_within_interval_probability_l379_379360

-- Define the intervals and their frequencies
def freq1 := 2
def freq2 := 3
def freq3 := 4
def freq4 := 5
def freq5 := 4
def freq6 := 2

-- Define the sample size
def sample_size : ℕ := 20

-- The condition we need to prove
theorem sample_data_within_interval_probability : 
  (freq1 + freq2 + freq3 + freq4) / sample_size = 0.7 := 
by 
  sorry

end sample_data_within_interval_probability_l379_379360


namespace sqrt_2_minus_sqrt_8_eq_neg_sqrt_2_l379_379411

theorem sqrt_2_minus_sqrt_8_eq_neg_sqrt_2 : sqrt 2 - sqrt 8 = -sqrt 2 := by
  have h : sqrt 8 = 2 * sqrt 2 := by
    rw [sqrt, sqrt, sqrt]
    sorry
  rw [h]
  sorry

end sqrt_2_minus_sqrt_8_eq_neg_sqrt_2_l379_379411


namespace matchstick_rearrangement_l379_379870

theorem matchstick_rearrangement (initial_configuration: list ℕ) (n: ℕ) :
  (n = 0 ∨ n = "NIL") ∧ initial_configuration = [5, 7] ∧ 
  (number_of_matches initial_configuration = 6) -> 
  (∃ new_configuration: list ℕ, number_of_moves initial_configuration new_configuration = 2 ∧ 
  (new_configuration = [0] ∨ new_configuration = ["NIL"])) :=
by
  sorry

end matchstick_rearrangement_l379_379870


namespace standing_next_to_Denis_l379_379310

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379310


namespace part1_part2_l379_379089

-- Definition of Set A
def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- Definition of Set B
def B : Set ℝ := { x | x ≥ 3 }

-- The Complement of the Intersection of A and B
def C_R (S : Set ℝ) : Set ℝ := { x | ¬ (x ∈ S) }

-- Set C
def C (a : ℝ) : Set ℝ := { x | x ≤ a }

-- Lean statement for part 1
theorem part1 : C_R (A ∩ B) = { x | x < 3 ∨ x > 6 } :=
by sorry

-- Lean statement for part 2
theorem part2 (a : ℝ) (hA_C : A ⊆ C a) : a ≥ 6 :=
by sorry

end part1_part2_l379_379089


namespace gcd_fraction_coprime_l379_379634

theorem gcd_fraction_coprime {a b c : ℕ} (h : Nat.gcd a (b * c) = Nat.gcd (a * b) c) : 
  Nat.coprime (a / Nat.gcd a c) (c / Nat.gcd a c) ∧ Nat.coprime (a / Nat.gcd a c) b :=
sorry

end gcd_fraction_coprime_l379_379634


namespace arithmetic_sequence_thirtieth_term_l379_379236

theorem arithmetic_sequence_thirtieth_term
  (a_1 : ℤ)
  (a_13 : ℤ)
  (h1 : a_1 = 10)
  (h2 : a_13 = 50) :
  let d := (a_13 - a_1) / 12 in
  a_1 + 29 * d = 100 :=
by
  sorry

end arithmetic_sequence_thirtieth_term_l379_379236


namespace concurrency_of_lines_on_circumcircle_l379_379798

theorem concurrency_of_lines_on_circumcircle
  (A B C T A1 B1 C1 A2 B2 C2 : Point)
  (hT_inside : Inside T (Triangle A B C))
  (h_reflect_A1 : Reflection T (Line B C) = A1)
  (h_reflect_B1 : Reflection T (Line C A) = B1)
  (h_reflect_C1 : Reflection T (Line A B) = C1)
  (h_circle : Circumcircle (Triangle A1 B1 C1) = Γ)
  (h_intersect_A2 : SecondIntersection (Line A1 T) Γ = A2)
  (h_intersect_B2 : SecondIntersection (Line B1 T) Γ = B2)
  (h_intersect_C2 : SecondIntersection (Line C1 T) Γ = C2)
  : ∃ K, Concurrent (Line A A2) (Line B B2) (Line C C2) ∧ OnCircle K Γ :=
sorry

end concurrency_of_lines_on_circumcircle_l379_379798


namespace total_days_at_protests_l379_379145

theorem total_days_at_protests :
  let first_protest := 4
  let second_protest := 4 + 0.25 * 4
  let third_protest := second_protest + 0.50 * second_protest
  let fourth_protest := third_protest + 0.75 * third_protest
  first_protest + second_protest + third_protest + fourth_protest = 29.625 :=
by
  let first_protest := 4
  let second_protest := 4 + 0.25 * 4
  let third_protest := second_protest + 0.50 * second_protest
  let fourth_protest := third_protest + 0.75 * third_protest
  have h1 : first_protest = 4 := by rfl
  have h2 : second_protest = 4 + 0.25 * 4 := by rfl
  have h3 : third_protest = second_protest + 0.50 * second_protest := by rfl
  have h4 : fourth_protest = third_protest + 0.75 * third_protest := by rfl
  have h_total : first_protest + second_protest + third_protest + fourth_protest = 4 + (4 + 0.25 * 4) + ((4 + 0.25 * 4) + 0.50 * (4 + 0.25 * 4)) + (((4 + 0.25 * 4) + 0.50 * (4 + 0.25 * 4)) + 0.75 * ((4 + 0.25 * 4) + 0.50 * (4 + 0.25 * 4))) := by rfl
  have h_calc : 4 + (4 + 1) + (5 + 2.5) + (7.5 + 5.625) = 29.625 := by
    calc
      4 + (4 + 1) + (5 + 2.5) + (7.5 + 5.625)
      = 4 + 5 + 7.5 + 13.125 : by norm_num
      _ = 29.625 : by norm_num
  exact h_calc

end total_days_at_protests_l379_379145


namespace one_minute_interval_exists_l379_379632

-- Define the constants
def path_length : ℕ := 100  -- length in meters
def speed1 := 1  -- speed in km/h
def speed2 := 2  -- speed in km/h
def speed3 := 3  -- speed in km/h

-- Convert speeds from km/h to m/min
def speed1_m_per_min := speed1 * 1000 / 60
def speed2_m_per_min := speed2 * 1000 / 60
def speed3_m_per_min := speed3 * 1000 / 60

-- Define the problem statement
theorem one_minute_interval_exists :
  ∃ t : ℕ, ∃ d : ℕ, t >= 0 ∧ t < 60 ∧
  ∀ i ∈ [1, 2, 3], 
    let speed := match i with
    | 1 => speed1_m_per_min
    | 2 => speed2_m_per_min
    | 3 => speed3_m_per_min
    end
  in 
    (t mod (2 * path_length / speed / 60)) < (path_length / speed / 60) :=
sorry

end one_minute_interval_exists_l379_379632


namespace angle_C_value_max_sin_sum_l379_379516

variable {a b c : ℝ} -- variables for the sides
variable {A B C : ℝ} -- variables for the angles
variable (h1 : 0 < C) (h2 : C < Real.pi) -- condition on angle C being in the range (0, π)
variable (h3 : a * Real.tan C = 2 * c * Real.sin A) -- main equation given in the problem
variable (h4 : a / Real.sin A = c / Real.sin C) -- sine law applied

-- Problem I
theorem angle_C_value : C = Real.pi / 3 := by
  sorry

-- Problem II
theorem max_sin_sum : ∃ A, B, (0 < A) ∧ (A < 2 * Real.pi / 3) ∧ B = 2 * Real.pi / 3 - A ∧ Real.sin A + Real.sin B = Real.sqrt 3 := by
  sorry

end angle_C_value_max_sin_sum_l379_379516


namespace denis_neighbors_l379_379295

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

end denis_neighbors_l379_379295


namespace votes_cast_l379_379344

theorem votes_cast (V : ℝ) (candidate_votes : ℝ) (rival_margin : ℝ)
  (h1 : candidate_votes = 0.30 * V)
  (h2 : rival_margin = 4000)
  (h3 : 0.30 * V + (0.30 * V + rival_margin) = V) :
  V = 10000 := 
by 
  sorry

end votes_cast_l379_379344


namespace probability_gcd_is_one_l379_379325

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def total_pairs := Finset.card ((Finset.range 8).image (λ a, (Finset.range 8).image (λ b, (a, b))))

def relatively_prime_pairs := Finset.card ((Finset.range 8).image (λ a, (Finset.range 8).image (λ b, if a ≠ b ∧ is_relatively_prime a b then (a, b) else ⊤)))

theorem probability_gcd_is_one :
  (relatively_prime_pairs : ℚ) / (total_pairs : ℚ) = 3 / 4 :=
sorry

end probability_gcd_is_one_l379_379325


namespace equation_of_circle_correct_l379_379233

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end equation_of_circle_correct_l379_379233


namespace find_x_l379_379506

theorem find_x (x : ℕ) (hx : x > 0) : 1^(x + 3) + 2^(x + 2) + 3^x + 4^(x + 1) = 1958 → x = 4 :=
sorry

end find_x_l379_379506


namespace necessary_and_sufficient_condition_total_sum_n_2002_l379_379912

theorem necessary_and_sufficient_condition (n : ℕ) :
  (∃ p : ℕ, n = 3 * p + 1) ↔ n % 3 = 1 := sorry

theorem total_sum_n_2002 : 
  ((2002 % 3 = 1) ∧ 
  (∑ i in finset.range 2002.succ, i) + 6 * (∑ i in finset.range 2002.succ, i) = 12880278) := sorry

end necessary_and_sufficient_condition_total_sum_n_2002_l379_379912


namespace value_of_a_l379_379452

theorem value_of_a (a : ℝ) (x : ℝ) (h : (a - 1) * x^2 + x + a^2 - 1 = 0) : a = -1 :=
sorry

end value_of_a_l379_379452


namespace schoolchildren_lineup_l379_379302

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379302


namespace triangle_area_eq_product_projections_l379_379118

theorem triangle_area_eq_product_projections
    {A B C P Q D E : Type*}
    [instA : ∃ (A' : A)]
    [instB : ∃ (B' : B)]
    [instC : ∃ (C' : C)]
    [instP : ∃ (P' : P)]
    [instQ : ∃ (Q' : Q)]
    [instD : ∃ (D' : D)]
    [instE : ∃ (E' : E)]
    (triangle_ABC : ∀ (A B C : Type*), Prop)
    (angle_bisector_AD : ∀ (A D : Type*), Prop)
    (external_angle_bisector_AE : ∀ (A E : Type*), Prop)
    (BP_perpendicular_AD : ∀ (B P D : Type*), Prop)
    (CQ_perpendicular_AE : ∀ (C Q E : Type*), Prop)
    (area_triangle : ∀ (A B C : Type*), Real)
    (projection_AP : Real)
    (projection_AQ : Real)
    (h1 : triangle_ABC A B C)
    (h2 : angle_bisector_AD A D)
    (h3 : external_angle_bisector_AE A E)
    (h4 : BP_perpendicular_AD B P D)
    (h5 : CQ_perpendicular_AE C Q E)
    (h6 : area_triangle A B C = projection_AP * projection_AQ) 
    : (area_triangle A B C) = (projection_AP * projection_AQ) :=
sorry

end triangle_area_eq_product_projections_l379_379118


namespace average_speed_is_correct_l379_379357

-- Define the times in hours
def t1 : ℝ := 15 / 60
def t2 : ℝ := 30 / 60
def t3 : ℝ := 45 / 60
def t4 : ℝ := 90 / 60

-- Define the speeds in km/h
def v1 : ℝ := 50
def v2 : ℝ := 80
def v3 : ℝ := 60
def v4 : ℝ := 100

-- Calculate individual distances
def d1 := v1 * t1
def d2 := v2 * t2
def d3 := v3 * t3
def d4 := v4 * t4

-- Total distance
def total_distance := d1 + d2 + d3 + d4

-- Total time
def total_time := t1 + t2 + t3 + t4

-- Proof statement
theorem average_speed_is_correct : (total_distance / total_time) = 82.5 := by
  sorry

end average_speed_is_correct_l379_379357


namespace students_in_class_l379_379936

theorem students_in_class (N : ℕ) 
  (avg_age_class : ℕ) (avg_age_4 : ℕ) (avg_age_10 : ℕ) (age_15th : ℕ) 
  (total_age_class : ℕ) (total_age_4 : ℕ) (total_age_10 : ℕ)
  (h1 : avg_age_class = 15)
  (h2 : avg_age_4 = 14)
  (h3 : avg_age_10 = 16)
  (h4 : age_15th = 9)
  (h5 : total_age_class = avg_age_class * N)
  (h6 : total_age_4 = 4 * avg_age_4)
  (h7 : total_age_10 = 10 * avg_age_10)
  (h8 : total_age_class = total_age_4 + total_age_10 + age_15th) :
  N = 15 :=
by
  sorry

end students_in_class_l379_379936


namespace standing_next_to_Denis_l379_379313

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379313


namespace denis_neighbors_l379_379286

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379286


namespace find_g1_l379_379799

variables {f g : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_g1 (hf : odd_function f)
                (hg : even_function g)
                (h1 : f (-1) + g 1 = 2)
                (h2 : f 1 + g (-1) = 4) :
                g 1 = 3 :=
sorry

end find_g1_l379_379799


namespace symmetric_point_of_M_origin_l379_379479

-- Define the point M with given coordinates
def M : (ℤ × ℤ) := (-3, -5)

-- The theorem stating that the symmetric point of M about the origin is (3, 5)
theorem symmetric_point_of_M_origin :
  let symmetric_point : (ℤ × ℤ) := (-M.1, -M.2)
  symmetric_point = (3, 5) :=
by
  -- (Proof should be filled)
  sorry

end symmetric_point_of_M_origin_l379_379479


namespace inequality_holds_l379_379037

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, a*x^2 + 2*a*x - 2 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_holds_l379_379037


namespace polygon_sides_14_l379_379373

theorem polygon_sides_14 (n : ℕ) (θ : ℝ) 
  (h₀ : (n - 2) * 180 - θ = 2000) :
  n = 14 :=
sorry

end polygon_sides_14_l379_379373


namespace new_number_formed_l379_379837

theorem new_number_formed (t u : ℕ) (ht : t < 10) (hu : u < 10) : 3 * 100 + (10 * t + u) = 300 + 10 * t + u := 
by {
  sorry
}

end new_number_formed_l379_379837


namespace max_integers_greater_than_17_l379_379626

def nine_integers := { x : List ℤ // x.length = 9 ∧ x.sum = 21 }

theorem max_integers_greater_than_17 : 
  ∀ (x : nine_integers), 
  ∃ k ≤ 9, k = 8 ∧ ∀ i < k, (x.val.nth i).get_or_else 0 > 17 := 
sorry

end max_integers_greater_than_17_l379_379626


namespace amount_lent_to_C_l379_379368

-- Define the conditions
def P_B : ℝ := 5000
def T_B : ℝ := 2
def P_C (X : ℝ) : ℝ := X
def T_C : ℝ := 4
def R : ℝ := 10
def total_interest : ℝ := 2200

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

-- Define the specific interest amounts for B and C
def interest_B : ℝ := simple_interest P_B R T_B
def interest_C (X : ℝ) : ℝ := simple_interest (P_C X) R T_C

-- The proposition to prove
theorem amount_lent_to_C (X : ℝ) (h : interest_B + interest_C X = total_interest) : X = 3000 :=
by
  sorry

end amount_lent_to_C_l379_379368


namespace winner_percentage_l379_379528

theorem winner_percentage (votes_winner : ℕ) (votes_difference : ℕ) (total_votes : ℕ) 
  (h1 : votes_winner = 1044) 
  (h2 : votes_difference = 288) 
  (h3 : total_votes = votes_winner + (votes_winner - votes_difference)) :
  (votes_winner * 100) / total_votes = 58 :=
by
  sorry

end winner_percentage_l379_379528


namespace sum_reciprocals_l379_379568

variable (S : Set ℕ) [DecidableEq ℕ]

theorem sum_reciprocals (hS : ∀ x ∈ S, 0 < x) : 
  (∃ (F G : Finset ℕ) (hf : ↑F ⊆ S) (hg : ↑G ⊆ S), F ≠ G ∧ (F.sum (λ x, (1 : ℚ) / x) = G.sum (λ x, (1 : ℚ) / x))) ∨ 
  (∃ r ∈ Set.Ioo (0 : ℚ) 1, ∀ (F : Finset ℕ) (hf : ↑F ⊆ S), F.sum (λ x, (1 : ℚ) / x) ≠ r) := by
  sorry

end sum_reciprocals_l379_379568


namespace shaded_region_volume_l379_379245

/-- Definition of the radii and heights of the rectangles -/
def radius1 : ℝ := 5
def height1 : ℝ := 1
def radius2 : ℝ := 2
def height2 : ℝ := 3

/-- Definitions of the volumes of the individual cylinders -/
def volume_large_cylinder := Real.pi * radius1^2 * height1
def volume_small_cylinder := Real.pi * radius2^2 * height2

/-- Total volume of the solid formed by rotating the shaded region -/
theorem shaded_region_volume :
  volume_large_cylinder + volume_small_cylinder = 37 * Real.pi := by
  sorry

end shaded_region_volume_l379_379245


namespace part1_part2_l379_379888

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x : ℝ) := x^2 - 1

theorem part1 {x : ℝ} (h : 1 ≤ x) : f x ≤ (1 / 2) * g x := by
  sorry

theorem part2 {m : ℝ} : (∀ x, 1 ≤ x → f x - m * g x ≤ 0) → m ≥ (1 / 2) := by
  sorry

end part1_part2_l379_379888


namespace max_value_of_d_l379_379889

-- Define the conditions of the problem
def f : ℝ → ℝ := sorry
def h : ℝ → ℝ := sorry

axiom range_f : ∀ x, -3 ≤ f(x) ∧ f(x) ≤ 4
axiom range_h : ∀ x, -1 ≤ h(x) ∧ h(x) ≤ 3

-- State the theorem
theorem max_value_of_d : 
  ∀ c d,
  (∀ x, c ≤ f(x) * h(x) ∧ f(x) * h(x) ≤ d) →
  d = 12 :=
by
  intros c d h_cond
  have : ∀ x, f(x) * h(x) ≤ 12, from sorry,
  have : d ≤ 12 := sorry,
  -- Construct example to show d can achieve 12
  use (4 * 3),
  have : ∀ x, f x = 4 ∧ h x = 3 → f x * h x = 12 := sorry,
  exact sorry

end max_value_of_d_l379_379889


namespace tan_shifted_strictly_increasing_k_l379_379446

noncomputable def strictlyIncreasingInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x < f y

def tangentShifted (x : ℝ) : ℝ := Real.tan (x + Real.pi / 4)

theorem tan_shifted_strictly_increasing_k (k : ℤ) :
  strictlyIncreasingInterval tangentShifted (k * Real.pi - (3 / 4) * Real.pi) (k * Real.pi + Real.pi / 4) :=
sorry

end tan_shifted_strictly_increasing_k_l379_379446


namespace rope_length_l379_379932

theorem rope_length (h1 : ∃ x : ℝ, 4 * x = 20) : 
  ∃ l : ℝ, l = 35 := by
sorry

end rope_length_l379_379932


namespace train_length_l379_379710

theorem train_length (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 600.0000000000001 = V * 54) : 
  L = 300.00000000000005 :=
by 
  sorry

end train_length_l379_379710


namespace lena_nicole_candy_difference_l379_379152

variables (L K N : ℕ)

theorem lena_nicole_candy_difference
  (hL : L = 16)
  (hLK : L + 5 = 3 * K)
  (hKN : K = N - 4) :
  L - N = 5 :=
sorry

end lena_nicole_candy_difference_l379_379152


namespace floor_abs_sum_l379_379023

theorem floor_abs_sum : 
  let a := -7.9
  let b := 7.9
  let fl_b := 7
  let fl_a := -8
  in (Int.floor (abs a) + abs (Int.floor a) = 15) := by
  sorry

end floor_abs_sum_l379_379023


namespace proof_problem_l379_379250

-- Definitions for balls and events
inductive Ball : Type
| red : Ball
| white : Ball
| black : Ball

def bag : List Ball := [Ball.red, Ball.red, Ball.red, Ball.white, Ball.white, Ball.black]

def pairs : List (Ball × Ball) :=
[(Ball.red, Ball.red), (Ball.white, Ball.white), (Ball.red, Ball.black), 
 (Ball.red, Ball.white), (Ball.black, Ball.white)]

def event_A (p : Ball × Ball) : Prop := 
  p.1 = Ball.white ∨ p.2 = Ball.white

def event_B (p : Ball × Ball) : Prop := 
  p = (Ball.white, Ball.white)

def event_C (p : Ball × Ball) : Prop := 
  (p.1 = Ball.white ∧ p.2 ≠ Ball.white) ∨ (p.1 ≠ Ball.white ∧ p.2 = Ball.white)

def event_D (p : Ball × Ball) : Prop := 
  (p.1 = Ball.white ∨ p.2 = Ball.white) ∨ (p = (Ball.red, Ball.black))

def mutually_exclusive_but_not_exhaustive (e1 e2 : (Ball × Ball) → Prop) : Prop := 
  (∀ p, ¬ (e1 p ∧ e2 p)) ∧ ¬ (∀ p, e1 p ∨ e2 p)

theorem proof_problem : mutually_exclusive_but_not_exhaustive 
  (λ p, event_A p) (λ p, (event_A p ∧ event_D p)) :=
by
  intros p
  sorry

end proof_problem_l379_379250


namespace arseniy_sitting_time_l379_379720

open Nat

noncomputable def time_hands_opposite (t : ℕ) : Bool :=
  let minute_position := 6 * t
  let hour_position := 30 * (t / 60) + 0.5 * (t % 60)
  abs (minute_position - hour_position) = 180

noncomputable def time_hands_overlap (t : ℕ) : Bool :=
  let minute_position := 6 * t
  let hour_position := 30 * (t / 60) + 0.5 * (t % 60)
  minute_position = hour_position

theorem arseniy_sitting_time :
  (∃ (t₁ t₂ : ℕ), 480 ≤ t₁ ∧ t₁ < 540 ∧ time_hands_opposite t₁ ∧ 1320 ≤ t₂ ∧ t₂ < 1380 ∧ time_hands_overlap t₂ ∧ t₂ - t₁ = 6 * 60) := sorry

end arseniy_sitting_time_l379_379720


namespace maxwell_walking_speed_l379_379576

-- Define Maxwell's walking speed
def Maxwell_speed (v : ℕ) : Prop :=
  ∀ t1 t2 : ℕ, t1 = 10 → t2 = 9 →
  ∀ d1 d2 : ℕ, d1 = 10 * v → d2 = 6 * t2 →
  ∀ d_total : ℕ, d_total = 94 →
  d1 + d2 = d_total

theorem maxwell_walking_speed : Maxwell_speed 4 :=
by
  sorry

end maxwell_walking_speed_l379_379576


namespace part_one_part_two_l379_379466

variable {a : ℕ → ℕ}

-- Conditions
axiom a1 : a 1 = 3
axiom recurrence_relation : ∀ n, a (n + 1) = 2 * (a n) + 1

-- Proof of the first part
theorem part_one: ∀ n, (a (n + 1) + 1) = 2 * (a n + 1) :=
by
  sorry

-- General formula for the sequence
theorem part_two: ∀ n, a n = 2^(n + 1) - 1 :=
by
  sorry

end part_one_part_two_l379_379466


namespace actual_distance_l379_379225

/-- The distance between two towns on a map is 9 inches. -/
def distance_on_map : ℝ := 9

/-- The scale is 0.5 inches = 5 miles. -/
def scale_inches : ℝ := 0.5
def scale_miles : ℝ := 5

/-- Calculate the actual distance between the towns given the scale and distance on the map. -/
theorem actual_distance (d_map : ℝ) (scale_inch : ℝ) (scale_mile : ℝ) :
  d_map = 9 ∧ scale_inch = 0.5 ∧ scale_mile = 5 →
  d_map * (scale_mile / scale_inch) = 90 :=
by
  intro h
  cases h with h_distance h_rest
  cases h_rest with h_scale_inch h_scale_mile
  rw [h_distance, h_scale_inch, h_scale_mile]
  norm_num
  sorry

end actual_distance_l379_379225


namespace hyperbola_asymptote_distance_l379_379797

section
open Function Real

variables (O P : ℝ × ℝ) (C : ℝ × ℝ → Prop) (M : ℝ × ℝ)
          (dist_asymptote : ℝ)

-- Conditions
def is_origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def on_hyperbola (P : ℝ × ℝ) : Prop := P.1 ^ 2 / 9 - P.2 ^ 2 / 16 = 1
def unit_circle (M : ℝ × ℝ) : Prop := sqrt (M.1 ^ 2 + M.2 ^ 2) = 1
def orthogonal (O M P : ℝ × ℝ) : Prop := O.1 * P.1 + O.2 * P.2 = 0
def min_PM (dist : ℝ) : Prop := dist = 1 -- The minimum distance when |PM| is minimized

-- Proof problem
theorem hyperbola_asymptote_distance :
  is_origin O → 
  on_hyperbola P → 
  unit_circle M → 
  orthogonal O M P → 
  min_PM (sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2)) → 
  dist_asymptote = 12 / 5 :=
sorry
end

end hyperbola_asymptote_distance_l379_379797


namespace find_y_l379_379868

theorem find_y (y : ℝ) (h1 : ∠ABC = 90) (h2 : ∠ABD = 3 * y) (h3 : ∠DBC = 2 * y) : y = 18 := by
  sorry

end find_y_l379_379868


namespace sum_of_valid_c_is_minus_20_l379_379034

noncomputable def is_rational_root (a b c x : ℚ) : Prop :=
  x^2 + a*x + b = c

def satisfies_conditions (c : ℤ) : Prop :=
  c <= 15 ∧
  ∃ (a b : ℚ), is_rational_root a b c (c^2 - 7)

def sum_valid_c (cs: List ℤ) : ℤ :=
  cs.filter satisfies_conditions |>.sum

theorem sum_of_valid_c_is_minus_20 :
  sum_valid_c [-12, -10, -6, 0, 8] = -20 := by
  sorry

end sum_of_valid_c_is_minus_20_l379_379034


namespace volume_of_pyramid_is_correct_l379_379679

noncomputable def volume_of_pyramid 
  (a b c : ℝ) 
  (SABC_pyramid : ∀ x : ℝ, x = 1) -- All lateral edges are 1
  (dihedral_angle : angle b 90 * 1 = 1) -- One dihedral angle at the base is 90 degrees
  : ℝ :=
  let V := (2 : ℝ) / (9 * Real.sqrt 3) in
  V

theorem volume_of_pyramid_is_correct :
  volume_of_pyramid 1 1 1 := sorry

end volume_of_pyramid_is_correct_l379_379679


namespace sum_S_1_to_99_l379_379402

def S : ℕ → ℕ 
| 1     := 1 
| 2     := 5
| 3     := 15
| 4     := 34
| 5     := 65
| 6     := 111
| 7     := 175
| n     := sorry  -- general formula for S_n is omitted

theorem sum_S_1_to_99 : (∑ i in finset.range 100, S (i + 1)) = 18145 := 
sorry

end sum_S_1_to_99_l379_379402


namespace present_age_of_dan_l379_379674

theorem present_age_of_dan (x : ℕ) : (x + 16 = 4 * (x - 8)) → x = 16 :=
by
  intro h
  sorry

end present_age_of_dan_l379_379674


namespace triangle_type_l379_379061

variables (O A B C : Type) [add_comm_group O] [vector_space ℝ O]

open_locale big_operators

-- Defining vectors as points on the plane.
variables (OA OB OC : O) -- vectors from O to A, O to B, and O to C respectively

-- Non-collinear condition
variable (h_non_collinear : ∃ (A B C : O), A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- Given condition
def given_condition (OB OC OA : O) : Prop :=
(OB - OC) •ᵥ (OB + OC - 2 • OA) = 0

-- Resulting condition for the type of triangle ABC
def is_isosceles_triangle_with_base_BC (A B C : O) : Prop :=
  let CB := B - C in
  ∃ (D : O), D = (B + C) / 2 ∧ CB •ᵥ (2 • D - 2 • A) = 0

theorem triangle_type (h : given_condition OB OC OA) : is_isosceles_triangle_with_base_BC A B C :=
sorry

end triangle_type_l379_379061


namespace schoolchildren_lineup_l379_379304

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379304


namespace sum_even_102_to_200_l379_379980

noncomputable def sum_even_integers (a b : ℕ) :=
  let n := (b - a) / 2 + 1
  in (n * (a + b)) / 2

theorem sum_even_102_to_200 :
  sum_even_integers 102 200 = 7550 := 
by
  have n : ℕ := (200 - 102) / 2 + 1
  have sum : ℕ := (n * (102 + 200)) / 2
  have n_50 : n = 50 := by sorry
  have sum_7550 : sum = 7550 := by sorry
  exact sum_7550 

end sum_even_102_to_200_l379_379980


namespace unique_N_l379_379754

-- Given conditions and question in the problem
variable (N : Matrix (Fin 2) (Fin 2) ℝ)

-- Problem statement: prove that the matrix defined below is the only matrix satisfying the given condition
theorem unique_N 
  (h : ∀ (w : Fin 2 → ℝ), N.mulVec w = -7 • w) 
  : N = ![![-7, 0], ![0, -7]] := 
sorry

end unique_N_l379_379754


namespace next_to_Denis_l379_379270

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379270


namespace correct_calculation_l379_379660

variable (a b : ℝ)

theorem correct_calculation : (-a^3)^2 = a^6 := 
by 
  sorry

end correct_calculation_l379_379660


namespace inscribed_square_in_isosceles_triangle_l379_379875

-- Definitions of the triangle and the inscribed square
structure Triangle where
  a b c : ℝ

structure Square where
  side : ℝ

-- Proof problem statement
theorem inscribed_square_in_isosceles_triangle :
  ∀ (T : Triangle) (S : Square),
    T.a = T.b ∧
    T.a = 10 ∧
    T.c = 12 →
    S.side = 4.8 :=
by
  intro T S
  assume h : T.a = T.b ∧ T.a = 10 ∧ T.c = 12
  sorry

end inscribed_square_in_isosceles_triangle_l379_379875


namespace five_digit_numbers_count_l379_379331

theorem five_digit_numbers_count : 
  let digits := [1, 2, 3]
  let numbers := { n : list ℕ | n.length = 5 ∧ (∀ (i : ℕ), i < 4 → n[i] ≠ n[i + 1]) ∧ (∀ d ∈ n, d ∈ digits) }
  in numbers.card = 42 :=
by
  sorry

end five_digit_numbers_count_l379_379331


namespace probability_king_ace_correct_l379_379324

noncomputable def probability_king_ace : ℚ :=
  4 / 663

theorem probability_king_ace_correct :
  ∀ (deck : Finset ℕ), deck.card = 52 →
  (∃ (f : ℕ → Finset ℕ) , (f 1).card = 51 ∧ (f 2).card = 50 ∧ (f 51).card = 1) →
  (∃ (g : ℕ → ℕ), g 1 = 4 ∧ g 2 = 3) →
  probability_king_ace = 4 / 663 := by
  intros deck hdeck hfex hgex
  sorry

end probability_king_ace_correct_l379_379324


namespace population_2002_l379_379519

-- Predicate P for the population of rabbits in a given year
def P : ℕ → ℝ := sorry

-- Given conditions
axiom cond1 : ∃ k : ℝ, P 2003 - P 2001 = k * P 2002
axiom cond2 : ∃ k : ℝ, P 2002 - P 2000 = k * P 2001
axiom condP2000 : P 2000 = 50
axiom condP2001 : P 2001 = 80
axiom condP2003 : P 2003 = 186

-- The statement we need to prove
theorem population_2002 : P 2002 = 120 :=
by
  sorry

end population_2002_l379_379519


namespace hyperbola_equation_l379_379088

theorem hyperbola_equation
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (e : ℝ) (he : e = 2 * Real.sqrt 3 / 3)
  (dist_from_origin : ∀ A B : ℝ × ℝ, A = (0, -b) ∧ B = (a, 0) →
    abs (a * b) / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2) :
  (a^2 = 3 ∧ b^2 = 1) → (∀ x y : ℝ, (x^2 / 3 - y^2 = 1)) := 
sorry

end hyperbola_equation_l379_379088


namespace prove_ffsinalpha_l379_379087

def f (x : ℚ) : ℚ :=
if x < 0 then 5 * x + 4 else 2 ^ x

def sin_alpha (α : ℝ) : ℝ :=
-3 / 5

theorem prove_ffsinalpha (α : ℝ) (P : ℝ × ℝ) (hP : P = (4, -3)) :
  f (f (sin_alpha α)) = 2 :=
sorry

end prove_ffsinalpha_l379_379087


namespace complex_addition_problem_l379_379894

-- Conditions from the problem
def A : Complex := 3 - 2 * Complex.i
def M : Complex := -5 + 3 * Complex.i
def S : Complex := -2 * Complex.i
def P : Complex := 3

-- Question converted into a Lean theorem
theorem complex_addition_problem : A + M + S - P = -5 - Complex.i := by
  sorry

end complex_addition_problem_l379_379894


namespace proof1_proof2_l379_379048

noncomputable def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

def recurrence_relation : Prop :=
  ∀ n, a (n + 3) - 9 * a (n + 2) + 27 * a (n + 1) - 27 * a n = 0

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n, a n * x^n

def series_evaluation (x : ℝ) : Prop :=
  series_sum x = (1 - 3*x + 18*x^2) / (1 - 3*x)^3

theorem proof1 : recurrence_relation := 
  by sorry

theorem proof2 : ∀ x : ℝ, series_evaluation x := 
  by sorry

end proof1_proof2_l379_379048


namespace monotonic_decreasing_interval_l379_379619

-- Define the inner function t
def t (x : ℝ) : ℝ := x^2 - x - 2

-- Define the outer function f
def f (x: ℝ) : ℝ := Real.log (t x)

-- Define the domain condition
def domain_condition (x : ℝ) : Prop := t x > 0

-- Define the interval
def is_monotonic_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ ⦃x y⦄, a < x → x < y → y < b → f x ≥ f y

-- The theorem we want to prove
theorem monotonic_decreasing_interval :
  is_monotonic_decreasing f (-∞) (-1) :=
sorry

end monotonic_decreasing_interval_l379_379619


namespace least_possible_value_l379_379428

theorem least_possible_value (x y : ℝ) : (3 * x * y - 1)^2 + (x - y)^2 ≥ 1 := sorry

end least_possible_value_l379_379428


namespace addition_and_rounding_l379_379388

theorem addition_and_rounding :
  let num1 : ℝ := 34.2871
  let num2 : ℝ := 9.602
  let sum : ℝ := num1 + num2
  let rounded_sum := Real.round (sum * 100) / 100
  rounded_sum = 43.89 :=
by
  simp [num1, num2, sum, rounded_sum]
  sorry

end addition_and_rounding_l379_379388


namespace parallel_vectors_perpendicular_vectors_l379_379820

noncomputable def vec_a : ℝ × ℝ := (1, -2)
noncomputable def vec_b : ℝ × ℝ := (3, 4)

theorem parallel_vectors (k : ℝ) :
  (3 * vec_a.1 - vec_b.1, 3 * vec_a.2 - vec_b.2) = (0, -10) →
  (1 + 3 * k, -2 + 4 * k) = (1, -2) + (k * vec_b.1, k * vec_b.2) →
  (3 * vec_a - vec_b) ∥ (vec_a + k * vec_b) →
  k = -1 / 3 :=
sorry

theorem perpendicular_vectors (m : ℝ) :
  (m * vec_a.1 - vec_b.1, m * vec_a.2 - vec_b.2) = (m - 3, -2 * m - 4) →
  ((1, -2) ⋅ ((m - 3), -2 * m - 4)) = 0 →
  (vec_a ⊙ (m * vec_a - vec_b)) = 0 →
  m = -1 :=
sorry

end parallel_vectors_perpendicular_vectors_l379_379820


namespace binomial_coefficient_fourth_term_l379_379426

theorem binomial_coefficient_fourth_term (n k : ℕ) (hn : n = 5) (hk : k = 3) : Nat.choose n k = 10 := by
  sorry

end binomial_coefficient_fourth_term_l379_379426


namespace compare_constants_l379_379886

noncomputable def a := 1 / Real.exp 1
noncomputable def b := Real.log 2 / 2
noncomputable def c := Real.log 3 / 3

theorem compare_constants : b < c ∧ c < a := by
  sorry

end compare_constants_l379_379886


namespace jason_stacked_bales_l379_379988

theorem jason_stacked_bales (initial_bales : ℕ) (final_bales : ℕ) (stored_bales : ℕ) 
  (h1 : initial_bales = 73) (h2 : final_bales = 96) : stored_bales = final_bales - initial_bales := 
by
  rw [h1, h2]
  sorry

end jason_stacked_bales_l379_379988


namespace probability_satisfies_condition_in_interval_l379_379491

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 2

def interval : set ℝ := {x | -5 ≤ x ∧ x ≤ 5}

def satisfies_condition (x : ℝ) : Prop := f(x) ≤ 0

theorem probability_satisfies_condition_in_interval :
  let ℓ := interval.count (λ x, satisfies_condition x) in
  (ℓ / (interval.count (λ x, true))) = 0.3 :=
sorry

end probability_satisfies_condition_in_interval_l379_379491


namespace P_sum_eq_one_and_condition_l379_379075

noncomputable def P (x : ℕ) : ℝ 
| 0       := 0.1
| 1       := 0.1
| 2       := 0.2 -- from the proof steps, A is resolved to be 0.2
| 3       := 0.3
| 4       := 0.2
| 5       := 0.1
| _       := 0 -- assuming it's zero for other values not given in the table

theorem P_sum_eq_one_and_condition
  (h0 : P 0 = 0.1)
  (h1 : P 1 = 0.1)
  (h2 : P 2 = 0.2)
  (h3 : P 3 = 0.3)
  (h4 : P 4 = 0.2)
  (h5 : P 5 = 0.1)
  (h_sum : P 0 + P 1 + P 2 + P 3 + P 4 + P 5 = 1) :
  P 1 + P 2 + P 3 = 0.6 := by
  sorry

end P_sum_eq_one_and_condition_l379_379075


namespace triangle_angle_type_l379_379591

theorem triangle_angle_type (a b c R : ℝ) (hc_max : c ≥ a ∧ c ≥ b) :
  (a^2 + b^2 + c^2 - 8 * R^2 > 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 = 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α = π / 2 ∨ β = π / 2 ∨ γ = π / 2)) ∧
  (a^2 + b^2 + c^2 - 8 * R^2 < 0 → 
     ∃ α β γ : ℝ, α + β + γ = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2)) :=
sorry

end triangle_angle_type_l379_379591


namespace quadratic_root_zero_l379_379455

theorem quadratic_root_zero (a : ℝ) : 
  ((a-1) * 0^2 + 0 + a^2 - 1 = 0) 
  → a ≠ 1 
  → a = -1 := 
by
  intro h1 h2
  sorry

end quadratic_root_zero_l379_379455


namespace sum_even_integers_correct_l379_379967

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end sum_even_integers_correct_l379_379967


namespace circumcenter_is_incenter_l379_379859

open EuclideanGeometry -- Assuming Euclidean geometry functionalities are available.

variable {A B C X Y O : Point}

-- Definitions of the conditions:
def acute_angled_triangle (A B C : Point) : Prop :=
  (angle A B C < 90) ∧ (angle B C A < 90) ∧ (angle C A B < 90)

def largest_angle_ABC (A B C : Point) : Prop :=
  (angle A B C > angle B C A) ∧ (angle A B C > angle C A B)

def perpendicular_bisector_intersect (A B C X Y : Point) : Prop :=
  (is_perpendicular_bisector_of_line BC AX C) ∧ 
  (is_perpendicular_bisector_of_line BA AY C)

-- The theorem statement:
theorem circumcenter_is_incenter {A B C X Y O : Point} :
  acute_angled_triangle A B C ∧ largest_angle_ABC A B C 
  ∧ perpendicular_bisector_intersect A B C X Y 
  ∧ (O = circumcenter A B C) → O = incenter B X Y := 
sorry

end circumcenter_is_incenter_l379_379859


namespace union_sets_l379_379472

open Set

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def A : Set α := { x | x^2 ≤ 4 }
def B : Set α := { x | x < 1 }
def C : Set α := { x | x ≤ 2 }

theorem union_sets (A_def : A = { x : α | x^2 ≤ 4 }) (B_def : B = { x : α | x < 1 }) :
  A ∪ B = C := 
sorry

end union_sets_l379_379472


namespace pyramid_scheme_indicator_l379_379644

def financial_pyramid_scheme_indicator (high_return lack_full_information aggressive_advertising : Prop) : Prop :=
  high_return ∧ lack_full_information ∧ aggressive_advertising

theorem pyramid_scheme_indicator
  (high_return : Prop)
  (lack_full_information : Prop)
  (aggressive_advertising : Prop)
  (indicator : financial_pyramid_scheme_indicator high_return lack_full_information aggressive_advertising) :
  indicator = (high_return ∧ lack_full_information ∧ aggressive_advertising) :=
sorry

end pyramid_scheme_indicator_l379_379644


namespace next_to_Denis_l379_379273

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379273


namespace prob_both_make_shots_l379_379330

open ProbabilityTheory

-- Defining the probabilities given in the conditions
def probA : ℚ := 2 / 5
def probB : ℚ := 1 / 2

-- The events A and B are independent
def independent_AB : Prop := Independent (event A) (event B)

-- The statement to prove
theorem prob_both_make_shots :
  independent_AB →
  probA * probB = 1 / 5 :=
by
  sorry

end prob_both_make_shots_l379_379330


namespace find_points_C_l379_379942

-- Define the points A and B within a square grid
variables {A B : ℝ × ℝ}

-- Define the area function for a triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the condition for the required area of the triangle
def has_required_area (C : ℝ × ℝ) : Prop :=
  triangle_area A B C = 4.5

theorem find_points_C (A B : ℝ × ℝ) :
  ∃ (C₁ C₂ C₃ C₄ C₅ C₆ C₇ C₈ : ℝ × ℝ),
  has_required_area C₁ ∧ has_required_area C₂ ∧
  has_required_area C₃ ∧ has_required_area C₄ ∧
  has_required_area C₅ ∧ has_required_area C₆ ∧
  has_required_area C₇ ∧ has_required_area C₈ :=
sorry

end find_points_C_l379_379942


namespace exists_polynomial_h_l379_379554

variables {R : Type*} [CommRing R] {f g : R[X]} (a : R[X] × R[X] → R[X])

theorem exists_polynomial_h (h : R[X]) :
  (∀ x y : R, f.eval x - f.eval y = a (x, y) * (g.eval x - g.eval y)) →
  ∃ h : R[X], ∀ x : R, f.eval x = h.eval (g.eval x) :=
sorry

end exists_polynomial_h_l379_379554


namespace simplify_expression_l379_379782

theorem simplify_expression (a : ℝ) (h : 1 < a ∧ a < 2) : sqrt ((a - 3) ^ 2) + abs (1 - a) = 2 := by
  sorry

end simplify_expression_l379_379782


namespace denis_neighbors_l379_379299

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

end denis_neighbors_l379_379299


namespace k_even_l379_379106

theorem k_even (n a b k : ℕ) (h1 : 2^n - 1 = a * b) (h2 : 2^k ∣ 2^(n-2) + a - b):
  k % 2 = 0 :=
sorry

end k_even_l379_379106


namespace Denis_next_to_Anya_Gena_l379_379266

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379266


namespace topaz_sapphire_value_equal_l379_379191

/-
  Problem statement: Given the following conditions:
  1. One sapphire and two topazes are three times more valuable than an emerald: S + 2T = 3E
  2. Seven sapphires and one topaz are eight times more valuable than an emerald: 7S + T = 8E
  
  Prove that the value of one topaz is equal to the value of one sapphire (T = S).
-/

theorem topaz_sapphire_value_equal
  (S T E : ℝ) 
  (h1 : S + 2 * T = 3 * E) 
  (h2 : 7 * S + T = 8 * E) :
  T = S := 
  sorry

end topaz_sapphire_value_equal_l379_379191


namespace unique_pairing_l379_379607

-- Define the problem conditions
def knows_adjacent (n i : ℕ) : ℕ := (i + 1) % n + 1 
def knows_across  (n i : ℕ) : ℕ := (i + n / 2) % n + 1

-- Define the main theorem
theorem unique_pairing : ∀ (n : ℕ) (h : n = 12) (k : ℕ) (hk : k = 6),
  ∃! pairing : list (ℕ × ℕ), 
    (∀ (i : ℕ), i ∈ list.range n → 
      (pairing.contains (i, knows_adjacent n i)) ∨ 
      (pairing.contains (i, knows_across n i))) := sorry

end unique_pairing_l379_379607


namespace floor_neg_sqrt_eval_l379_379018

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end floor_neg_sqrt_eval_l379_379018


namespace bacteria_growth_time_l379_379215

theorem bacteria_growth_time (n0 : ℕ) (n : ℕ) (rate : ℕ) (time_step : ℕ) (final : ℕ)
  (h0 : n0 = 200)
  (h1 : rate = 3)
  (h2 : time_step = 5)
  (h3 : n = n0 * rate ^ final)
  (h4 : n = 145800) :
  final = 30 := 
sorry

end bacteria_growth_time_l379_379215


namespace find_y_l379_379866

-- Definitions of angles and the given problem.
def angle_ABC : ℝ := 90
def angle_ABD (y : ℝ) : ℝ := 3 * y
def angle_DBC (y : ℝ) : ℝ := 2 * y

-- The theorem stating the problem
theorem find_y (y : ℝ) (h1 : angle_ABC = 90) (h2 : angle_ABD y + angle_DBC y = angle_ABC) : y = 18 :=
  by 
  sorry

end find_y_l379_379866


namespace polyhedron_volume_l379_379495

theorem polyhedron_volume :
  ∀ (a b : ℝ), 
  (0 < a ∧ 0 < b ∧ a < b) → 
  let rectangle1 := λ x, |x - a/2| + |x + a/2| = a
  let rectangle2 := λ y, |y - a/2| + |y + a/2| = a
  let rectangle3 := λ z, |z - b/2| + |z + b/2| = b
  in volume (cuboid a a b) = a^2 * b :=
begin
  sorry
end

end polyhedron_volume_l379_379495


namespace same_heads_prob_sum_l379_379143

-- Define the probability of heads when flipping the biased coin
def biased_coin_heads_prob : ℚ := 2 / 5

-- Define the generating function for the unbiased (fair) coins and the biased coin
def fair_coin_gf : ℕ → ℕ := λ k, if k = 0 then 1 else if k = 1 then 1 else 0
def biased_coin_gf : ℕ → ℕ := λ k, if k = 0 then 3 else if k = 1 then 2 else 0

-- Define the combined generating function for three coins: two fair, one biased
def combined_gf : ℕ → ℕ := λ k, match k with
  | 0 => 3
  | 1 => 8
  | 2 => 7
  | 3 => 2
  | _ => 0

-- Define the sum of squares of coefficients of the generating function
def sum_of_squares (f : ℕ → ℕ) : ℕ := 
  f 0 ^ 2 + f 1 ^ 2 + f 2 ^ 2 + f 3 ^ 2

-- Define the sum of all coefficients of the generating function
def sum_of_coeffs (f : ℕ → ℕ) : ℕ := 
  f 0 + f 1 + f 2 + f 3

-- The probability that Jackie and Phil get the same number of heads
def same_heads_prob : ℚ := 
  sum_of_squares combined_gf / (sum_of_coeffs combined_gf ^ 2 : ℕ)

-- Proof that the value of m + n is 263
theorem same_heads_prob_sum : (63 + 200 = 263) :=
by sorry

end same_heads_prob_sum_l379_379143


namespace alternating_coefs_l379_379902

theorem alternating_coefs {n : ℕ} (a : Fin n → ℝ) (A : Fin n → ℝ) (x : ℝ)
  (h_ordered : ∀ i j, i < j → a i < a j)
  (h_eq : ∑ i in Finset.univ, (A i / (x + a i)) = (∏ i in Finset.univ, (x + a i))⁻¹) :
  ∀ i, (i % 2 = 0 → A i > 0) ∧ (i % 2 = 1 → A i < 0) :=
by
  sorry

end alternating_coefs_l379_379902


namespace complex_number_coordinates_l379_379610

-- Define the complex number
def given_complex_number : ℂ := complex.I * (2 - complex.I)

-- State the theorem
theorem complex_number_coordinates : (⟨given_complex_number.re, given_complex_number.im⟩ : ℝ × ℝ) = (1, 2) := by
  sorry

end complex_number_coordinates_l379_379610


namespace scientific_notation_of_investment_l379_379911

theorem scientific_notation_of_investment : 41800000000 = 4.18 * 10^10 := 
by
  sorry

end scientific_notation_of_investment_l379_379911


namespace sum_poisson_prob_eq_one_l379_379198

noncomputable def poisson_prob (k : ℕ) (λ : ℝ) : ℝ :=
  (λ ^ k) * (Real.exp (-λ)) / Nat.factorial k

theorem sum_poisson_prob_eq_one (λ : ℝ) (h : 0 < λ) :
  ∑' k, poisson_prob k λ = 1 :=
by
  sorry

end sum_poisson_prob_eq_one_l379_379198


namespace area_of_M_l379_379571

noncomputable def M (n : ℕ) : Set ℂ :=
{z | ∑ k in Finset.range n.succ, 1 / Complex.abs (z - k) ≥ 1}

theorem area_of_M (n : ℕ) (hn : 0 < n) :
  ∃ R : ℝ, (R = Set.measure (M n)) ∧ (R ≥ (11 * n^2 + 1) * Real.pi / 12) :=
sorry

end area_of_M_l379_379571


namespace problem1_problem2_l379_379819

-- Define vector and orthogonality condition
def m : ℝ × ℝ := (1, real.sqrt 3)
def n (t : ℝ) : ℝ × ℝ := (2, t)

def orthogonal (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def angle (u v : ℝ × ℝ) (θ : ℝ) : Prop :=
  real.cos θ = (u.1 * v.1 + u.2 * v.2) / (real.sqrt (u.1 ^ 2 + u.2 ^ 2) * real.sqrt (v.1 ^ 2 + v.2 ^ 2))

theorem problem1 (t : ℝ) (h : orthogonal m (n t)) : t = -2 * real.sqrt 3 / 3 := sorry

theorem problem2 (t : ℝ) (h : angle m (n t) (real.pi / 6)) : t = 2 * real.sqrt 3 / 3 := sorry

end problem1_problem2_l379_379819


namespace apples_to_cucumbers_l379_379112

theorem apples_to_cucumbers (a b c : ℕ) 
    (h₁ : 10 * a = 5 * b) 
    (h₂ : 3 * b = 4 * c) : 
    (24 * a) = 16 * c := 
by
  sorry

end apples_to_cucumbers_l379_379112


namespace problem_statement_l379_379036

-- Definition of operation nabla
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- Main theorem statement
theorem problem_statement : nabla 2 (nabla 0 (nabla 1 7)) = 71859 :=
by
  -- Computational proof
  sorry

end problem_statement_l379_379036


namespace original_grain_amount_l379_379379

theorem original_grain_amount 
  (grain_spilled : ℕ)
  (grain_remaining : ℕ)
  : grain_spilled = 49952 → grain_remaining = 918 → (grain_spilled + grain_remaining = 50870) := 
by
  intros h1 h2
  rw [h1, h2]
  exact Nat.add_comm 49952 918
  simp

end original_grain_amount_l379_379379


namespace equivalent_conditions_l379_379882

open Real

theorem equivalent_conditions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / x + 1 / y + 1 / z ≤ 1) ↔
  (∀ a b c d : ℝ, a + b + c > d → a^2 * x + b^2 * y + c^2 * z > d^2) :=
by
  sorry

end equivalent_conditions_l379_379882


namespace overall_percentage_gain_is_0_98_l379_379675

noncomputable def original_price : ℝ := 100
noncomputable def increased_price := original_price * 1.32
noncomputable def after_first_discount := increased_price * 0.90
noncomputable def final_price := after_first_discount * 0.85
noncomputable def overall_gain := final_price - original_price
noncomputable def overall_percentage_gain := (overall_gain / original_price) * 100

theorem overall_percentage_gain_is_0_98 :
  overall_percentage_gain = 0.98 := by
  sorry

end overall_percentage_gain_is_0_98_l379_379675


namespace count_positive_multiples_of_7_ending_in_5_below_1500_l379_379099

theorem count_positive_multiples_of_7_ending_in_5_below_1500 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (k < 1500) → ((k % 7 = 0) ∧ (k % 10 = 5) → (∃ m : ℕ, k = 35 + 70 * m) ∧ (0 ≤ m) ∧ (m < 21))) :=
sorry

end count_positive_multiples_of_7_ending_in_5_below_1500_l379_379099


namespace cowboy_cost_problem_l379_379320

/-- The cost of a sandwich, a cup of coffee, and a donut adds up to 0.40 dollars given the expenditure details of two cowboys. -/
theorem cowboy_cost_problem (S C D : ℝ) (h1 : 4 * S + C + 10 * D = 1.69) (h2 : 3 * S + C + 7 * D = 1.26) :
  S + C + D = 0.40 :=
by
  sorry

end cowboy_cost_problem_l379_379320


namespace number_of_valid_4_digit_integers_l379_379823

def valid_first_two_digits : Finset ℕ := {1, 4, 5, 6}
def valid_last_two_digits : Finset ℕ := {3, 5, 7}
def pairs_with_even_sum (A B : ℕ) := A + B % 2 = 0
def unique_pairs_with_even_sum : Finset (ℕ × ℕ) := 
  valid_last_two_digits.product valid_last_two_digits
  .filter (λ p, p.1 ≠ p.2 ∧ (pairs_with_even_sum p.1 p.2))

theorem number_of_valid_4_digit_integers : 
  valid_first_two_digits.card * valid_first_two_digits.card * unique_pairs_with_even_sum.card = 96 :=
by sorry

end number_of_valid_4_digit_integers_l379_379823


namespace symmetric_circle_eq_l379_379463

theorem symmetric_circle_eq :
  (∃ f : ℝ → ℝ → Prop, (∀ x y, f x y ↔ (x - 2)^2 + (y + 1)^2 = 1)) →
  (∃ line : ℝ → ℝ → Prop, (∀ x y, line x y ↔ x - y + 3 = 0)) →
  (∃ eq : ℝ → ℝ → Prop, (∀ x y, eq x y ↔ (x - 4)^2 + (y - 5)^2 = 1)) :=
by
  sorry

end symmetric_circle_eq_l379_379463


namespace Denis_next_to_Anya_Gena_l379_379265

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379265


namespace jia_final_profit_is_one_l379_379550

-- Definition of the initial transaction and profits/losses
def initial_amount : ℝ := 1000
def first_transaction_income : ℝ := initial_amount * 1.1
def second_transaction_income : ℝ := - (initial_amount * 0.9)
def third_transaction_income (yi_sale_price : ℝ) : ℝ := yi_sale_price * 0.9

-- So the yu_sale_price = initial_amount * 0.9
def yi_sale_price : ℝ := initial_amount * 0.9

-- Total income calculation
def total_income : ℝ := -initial_amount + first_transaction_income + second_transaction_income + third_transaction_income(yi_sale_price)

-- The proof problem statement
theorem jia_final_profit_is_one : total_income = 1 := by
  sorry

end jia_final_profit_is_one_l379_379550


namespace coloring_circle_l379_379431

def a_n (n m : Nat) : Nat :=
  if n < 2 ∨ m < 2 then 0 else (m - 1)^n + (-1)^n * (m - 1)

theorem coloring_circle (n m : Nat) (h1 : n ≥ 2) (h2 : m ≥ 2) :
  a_n n m = (m - 1)^n + (-1)^n * (m - 1) :=
by sorry

end coloring_circle_l379_379431


namespace irrational_triangle_exists_l379_379134

theorem irrational_triangle_exists :
  ∃ (A B C : Type) [has_coords A B C] (AD : ℝ) (AB BC CA : ℝ),
    let ⟨circ, EF, angle_ADC⟩ := (circle 1 passing_through_vertex A tangent BC at D intersecting sides AB AC at E F respectively),
    EF bisects AFD ∧ angle_ADC = 80 →
    (AB / AD + BC / AD + CA / AD)^2 = 1 + sqrt 3 :=
begin
  sorry
end

end irrational_triangle_exists_l379_379134


namespace part1_solution_part2_solution_l379_379078

variables (x y m : ℤ)

-- Given the system of equations
def system_of_equations (x y m : ℤ) : Prop :=
  (2 * x - y = m) ∧ (3 * x + 2 * y = m + 7)

-- Part (1) m = 0, find x = 1, y = 2
theorem part1_solution : system_of_equations x y 0 → x = 1 ∧ y = 2 :=
sorry

-- Part (2) point A(-2,3) in the second quadrant with distances 3 and 2, find m = -7
def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

def distance_to_axes (x y dx dy : ℤ) : Prop :=
  y = dy ∧ x = -dx

theorem part2_solution : is_in_second_quadrant x y →
  distance_to_axes x y 2 3 →
  system_of_equations x y m →
  m = -7 :=
sorry

end part1_solution_part2_solution_l379_379078


namespace number_of_professors_after_reduction_l379_379380

theorem number_of_professors_after_reduction 
  (original_number : ℝ) 
  (reduction_rate : ℝ) 
  (h_original : original_number = 243.75) 
  (h_rate : reduction_rate = 0.20) :
  original_number - (reduction_rate * original_number) = 195 := 
by 
  rw [h_original, h_rate]
  norm_num
  sorry

end number_of_professors_after_reduction_l379_379380


namespace angle_obtuseness_l379_379391

theorem angle_obtuseness (a1 a2 a3 : Type) 
  (angle_skew_lines : a1 → Prop) 
  (angle_line_plane : a2 → Prop) 
  (angle_dihedral : a3 → Prop) :
  (∀ x, angle_skew_lines x → (0 < x ∧ x ≤ π/2)) → 
  (∀ y, angle_line_plane y → (0 ≤ y ∧ y ≤ π/2)) → 
  (∃ z, angle_dihedral z ∧ (0 ≤ z ∧ z ≤ π)) → 
  exists z, angle_dihedral z ∧ z > π/2 :=
begin
  intros h1 h2 h3,
  rcases h3 with ⟨z, hz⟩,
  use z,
  sorry
end

end angle_obtuseness_l379_379391


namespace toothpicks_in_arithmetic_sequence_l379_379731

theorem toothpicks_in_arithmetic_sequence :
  let a1 := 5
  let d := 3
  let n := 15
  let a_n n := a1 + (n - 1) * d
  let sum_to_n n := n * (2 * a1 + (n - 1) * d) / 2
  sum_to_n n = 390 := by
  sorry

end toothpicks_in_arithmetic_sequence_l379_379731


namespace inequality_proof_l379_379158

theorem inequality_proof (k n : ℕ) (h1 : k ≥ 1) (h2 : n ≥ 1) 
  (x : ℕ → ℝ) 
  (h3 : (∑ j in Finset.range n, 1 / (x j ^ (2^k) + k)) = 1 / k) :
  (∑ j in Finset.range n, 1 / (x j ^ (2^(k+1)) + k + 2)) ≤ 1 / (k + 1) :=
  sorry

end inequality_proof_l379_379158


namespace paying_students_pay_7_dollars_l379_379399

theorem paying_students_pay_7_dollars
  (total_students : ℕ)
  (free_lunch_percentage : ℝ)
  (total_cost : ℝ)
  (total_students_eq : total_students = 50)
  (free_lunch_percentage_eq : free_lunch_percentage = 0.4)
  (total_cost_eq : total_cost = 210) :
  ( (total_cost / (total_students * (1 - free_lunch_percentage))) = 7 ) :=
by
  have h1 : total_students * (1 - free_lunch_percentage) = 30, 
  { rw [total_students_eq, free_lunch_percentage_eq],
    norm_num },
  have h2 : total_cost / 30 = 7, 
  { rw [total_cost_eq, h1],
    norm_num },
  exact h2

end paying_students_pay_7_dollars_l379_379399


namespace circle_eq_l379_379227

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end circle_eq_l379_379227


namespace solve_z_pow_eq_neg_sixteen_l379_379759

theorem solve_z_pow_eq_neg_sixteen (z : ℂ) :
  z^4 = -16 ↔ 
  z = complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) - complex.I * complex.sqrt(2) ∨ 
  z = complex.sqrt(2) - complex.I * complex.sqrt(2) :=
by
  sorry

end solve_z_pow_eq_neg_sixteen_l379_379759


namespace minimum_colors_for_phone_wires_l379_379637

-- Definitions
def phone_graph := SimpleGraph (Fin 20) -- the graph of 20 phones
def no_more_than_two_wires (G : phone_graph) := ∀ (v : Fin 20), G.degree v ≤ 2
def minimal_coloring_num (G : phone_graph) (c : ℕ) := 
  ∀ (v : Fin 20), ∀ (e1 e2 ∈ G.incidence_set v), G.edge_color e1 ≠ G.edge_color e2

-- Statement
theorem minimum_colors_for_phone_wires (G : phone_graph) 
  (h1 : no_more_than_two_wires G) : 
  ∃ (c : ℕ), minimal_coloring_num G c ∧ c = 3 := 
sorry

end minimum_colors_for_phone_wires_l379_379637


namespace intervals_of_monotonicity_max_integer_k_product_inequality_l379_379809

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / (x - 1)

theorem intervals_of_monotonicity : 
  (∀ x ∈ Set.Ioo 0 1, f' x < 0) ∧ (∀ x ∈ Set.Ioi 1, f' x < 0) := sorry

theorem max_integer_k (k : ℤ) (x : ℝ) (hx : 1 < x) : 
  f x > k / x ↔ k ≤ 3 := sorry

theorem product_inequality (n : ℕ) (hn : 0 < n) : 
  ∏ i in Finset.range n, (1 + i.succ * (i + 2)) > Real.exp (2 * n - 3) := sorry

end intervals_of_monotonicity_max_integer_k_product_inequality_l379_379809


namespace expected_swaps_l379_379772

-- Define the problem conditions
def five_pairs_twins := 5

-- Define the circle configuration and adjacent swaps
def circle {α : Type*} := list α
def swap_adjacent {α : Type*} (l : circle α) (i : ℕ) : circle α :=
  let len := l.length in
  if len = 0 then l
  else if i < len - 1 then
    (l.take i) ++ [l.nth_le (i + 1) sorry] ++ [l.nth_le i sorry] ++ (l.drop (i + 2))
  else
    l

-- Define the expected value of swaps
noncomputable def expected_swaps_needed : ℚ := 926 / 945

-- Main statement: proving the expected number of swaps
theorem expected_swaps :
  expected_swaps_needed = 926 / 945 :=
sorry

end expected_swaps_l379_379772


namespace number_of_integers_x_l379_379773

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

def valid_range_x (x : ℝ) : Prop :=
  13 < x ∧ x < 43

def conditions_for_acute_triangle (x : ℝ) : Prop :=
  (x > 28 ∧ x^2 < 1009) ∨ (x ≤ 28 ∧ x > 23.64)

theorem number_of_integers_x (count : ℤ) :
  (∃ (x : ℤ), valid_range_x x ∧ is_triangle 15 28 x ∧ is_acute_triangle 15 28 x ∧ conditions_for_acute_triangle x) →
  count = 8 :=
sorry

end number_of_integers_x_l379_379773


namespace a3_possible_values_l379_379464

noncomputable def geometric_sequence (a r : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := geometric_sequence n * r

theorem a3_possible_values (r a_2 a_3 : ℕ) (S3 : ℕ) 
  (h1 : r ≠ 0)
  (h2 : S3 = 6)
  (h3 : a_1 = 2)
  (h4 : a_2 = a_1 * r)
  (h5 : a_3 = a_2 * r) 
  (h6 : S3 = a_1 + a_2 + a_3) : 
  a_3 = 2 ∨ a_3 = 8 := 
sorry

end a3_possible_values_l379_379464


namespace isosceles_triangle_n_value_l379_379110

noncomputable def n_values : set ℕ := {15, 16}

def isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∧ c ≠ a) ∨ (a = c ∧ b ≠ a) ∨ (b = c ∧ a ≠ b)

def roots_of_quadratic (a b c x : ℕ) : Prop :=
  x * x - a * x + b = 0

theorem isosceles_triangle_n_value {a b c n : ℕ} (h_triangle : isosceles_triangle a b c)
  (h_side : a = 3)
  (h_roots : ∀ x : ℕ, roots_of_quadratic 8 n x → x ∈ {a, b, c}) :
  n ∈ n_values :=
by
  sorry

end isosceles_triangle_n_value_l379_379110


namespace sum_of_averages_l379_379771

def even_integers (n : ℕ) : list ℕ :=
  list.filter (λ x, x % 2 = 0) (list.range (n + 1))

def even_perfect_squares (n : ℕ) : list ℕ :=
  list.filter (λ x, x % 2 = 0) (list.map (λ x, x * x) (list.range (n + 1)))

noncomputable def average (l : list ℕ) : ℚ :=
  l.sum / l.length

theorem sum_of_averages : 
  average (even_integers 100) + average (even_integers 50) + average (even_perfect_squares 15) = 155 := 
by 
  sorry

end sum_of_averages_l379_379771


namespace find_y_l379_379869

theorem find_y (y : ℝ) (h1 : ∠ABC = 90) (h2 : ∠ABD = 3 * y) (h3 : ∠DBC = 2 * y) : y = 18 := by
  sorry

end find_y_l379_379869


namespace heart_then_club_probability_l379_379998

theorem heart_then_club_probability :
  (13 / 52) * (13 / 51) = 13 / 204 := by
  sorry

end heart_then_club_probability_l379_379998


namespace man_speed_l379_379696

theorem man_speed (rest_time_per_km : ℕ := 5) (total_km_covered : ℕ := 5) (total_time_min : ℕ := 50) : 
  (total_time_min - rest_time_per_km * (total_km_covered - 1)) / 60 * total_km_covered = 10 := by
  sorry

end man_speed_l379_379696


namespace choose_starters_l379_379914

-- Conditions as definitions
def num_players : ℕ := 14
def starters_to_choose : ℕ := 6

-- Proof statement
theorem choose_starters (n k : ℕ) (h_n : n = num_players) (h_k : k = starters_to_choose) :
    nat.choose n k = 3003 :=
by {
  rw [h_n, h_k],
  exact sorry,
}

end choose_starters_l379_379914


namespace smallest_q_exists_l379_379900

noncomputable def p_q_r_are_consecutive_terms (p q r : ℝ) : Prop :=
∃ d : ℝ, p = q - d ∧ r = q + d

theorem smallest_q_exists
  (p q r : ℝ)
  (h1 : p_q_r_are_consecutive_terms p q r)
  (h2 : p > 0) 
  (h3 : q > 0) 
  (h4 : r > 0)
  (h5 : p * q * r = 216) :
  q = 6 :=
sorry

end smallest_q_exists_l379_379900


namespace solve_z_pow_eq_neg_sixteen_l379_379758

theorem solve_z_pow_eq_neg_sixteen (z : ℂ) :
  z^4 = -16 ↔ 
  z = complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) - complex.I * complex.sqrt(2) ∨ 
  z = complex.sqrt(2) - complex.I * complex.sqrt(2) :=
by
  sorry

end solve_z_pow_eq_neg_sixteen_l379_379758


namespace exists_2018_distinct_natural_numbers_no_arithmetic_sequence_l379_379729

/-- 
Prove that there exist 2018 distinct natural numbers, all smaller than 150,000,
such that no three numbers form an arithmetic sequence.
-/
theorem exists_2018_distinct_natural_numbers_no_arithmetic_sequence :
  ∃ (S : Finset ℕ), S.card = 2018 ∧ (∀ n ∈ S, n < 150000) ∧
    (∀ a b c ∈ S, a ≠ b → b ≠ c → a ≠ c → a + c ≠ 2 * b) :=
sorry

end exists_2018_distinct_natural_numbers_no_arithmetic_sequence_l379_379729


namespace fourth_roots_of_neg_16_l379_379766

theorem fourth_roots_of_neg_16 : 
  { z : ℂ | z^4 = -16 } = { sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I, 
                            sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I } :=
by
  sorry

end fourth_roots_of_neg_16_l379_379766


namespace sum_even_integers_102_to_200_l379_379974

theorem sum_even_integers_102_to_200 : 
  let sequence := list.range' 102 100 
  ∧ (∀ x ∈ sequence, x % 2 = 0) →
  list.sum sequence = 7550 := 
by 
  let sequence := list.range' 102 100 
  have even_sequence : ∀ x ∈ sequence, x % 2 = 0 := 
    sorry 
  have sum_sequence : list.sum sequence = 7550 := 
    sorry 
  exact sum_sequence 

end sum_even_integers_102_to_200_l379_379974


namespace B_contains_only_one_element_l379_379575

def setA := { x | (x - 1/2) * (x - 3) = 0 }

def setB (a : ℝ) := { x | Real.log (x^2 + a * x + a + 9 / 4) = 0 }

theorem B_contains_only_one_element (a : ℝ) :
  (∃ x, setB a x ∧ ∀ y, setB a y → y = x) →
  (a = 5 ∨ a = -1) :=
by
  intro h
  -- Proof would go here
  sorry

end B_contains_only_one_element_l379_379575


namespace hyperbola_scaling_transform_l379_379615

theorem hyperbola_scaling_transform:
  ∀ (x y x' y' : ℝ),
  (x' = 3 * x) →
  (2 * y' = y) →
  (x^2 - y^2 / 64 = 1) →
  (x'^2 / 9 - y'^2 / 16 = 1) :=
by
  intros x y x' y' hx' hy' h_Original.
  sorry

end hyperbola_scaling_transform_l379_379615


namespace floor_of_neg_sqrt_frac_l379_379013

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end floor_of_neg_sqrt_frac_l379_379013


namespace parabola_standard_equation_l379_379802

theorem parabola_standard_equation (d : ℝ) (h : d = -2) : ∃ p : ℝ, p = 2 ∧ x^2 = 4 * p * y :=
by
  -- Given conditions
  let directrix := d
  have h1 : directrix = -2 := h

  -- Derive the focus 
  let p := 2

  -- Conclusion
  have focus := (0, p)
  use p
  split
  · exact rfl
  · exact sorry

end parabola_standard_equation_l379_379802


namespace denis_neighbors_l379_379301

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

end denis_neighbors_l379_379301


namespace at_most_prime_factors_b_l379_379603

open Nat

theorem at_most_prime_factors_b 
  (a b : ℕ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (gcd_primes : (gcd a b).prime_factors.toFinset.card = 5)
  (lcm_primes : (lcm a b).prime_factors.toFinset.card = 23)
  (more_prime_factors_a : (prime_factors a).toFinset.card > (prime_factors b).toFinset.card) : 
  (prime_factors b).toFinset.card ≤ 13 := 
sorry

end at_most_prime_factors_b_l379_379603


namespace distance_midpoint_A_B_to_C_l379_379795

-- Definitions of points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := {x := 3, y := 4, z := 1}
def B : Point3D := {x := 1, y := 0, z := 5}
def C : Point3D := {x := 0, y := 1, z := 0}

-- Definition of the midpoint of two points in 3D space
def midpoint (P Q : Point3D) : Point3D :=
  { x := (P.x + Q.x) / 2,
    y := (P.y + Q.y) / 2,
    z := (P.z + Q.z) / 2 }

-- Definition of the distance between two points in 3D space
def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2 + (Q.z - P.z)^2)

-- Main theorem
theorem distance_midpoint_A_B_to_C :
  distance (midpoint A B) C = Real.sqrt 14 := by
  sorry

end distance_midpoint_A_B_to_C_l379_379795


namespace sum_of_first_5_terms_of_b_is_10_l379_379133

noncomputable def a_3 : ℤ := 2

noncomputable def a_2 : ℤ := a_3 * a_3 / 2

noncomputable def a_4 : ℤ := 2 * a_3 / a_2

-- Define initial conditions: 
def cnd1 : 2 * a_3 - a_2 * a_4 = 0 := by sorry
def b_3 := a_3 

-- Define the sum of the first 5 terms for the arithmetic sequence {b_n}
noncomputable def sum_first_5_b (a_3 : ℤ) := 5 * a_3

theorem sum_of_first_5_terms_of_b_is_10 
  (a_3 : ℤ)
  (h1 : 2 * a_3 - a_2 * a_4 = 0)
  (h2 : a_3 = 2)
  : sum_first_5_b a_3 = 10 := by
sorry

end sum_of_first_5_terms_of_b_is_10_l379_379133


namespace solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l379_379602

variable (a x : ℝ)

theorem solve_inequality_case_a_lt_neg1 (h : a < -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

theorem solve_inequality_case_a_eq_neg1 (h : a = -1) :
  ((x - 1) * (x + a) > 0) ↔ (x ≠ 1) := sorry

theorem solve_inequality_case_a_gt_neg1 (h : a > -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

end solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l379_379602


namespace difference_between_largest_and_third_smallest_l379_379753

-- Define the problem statement.
theorem difference_between_largest_and_third_smallest (d1 d2 d3 : ℕ) (h1 : d1 = 1) (h2 : d2 = 6) (h3 : d3 = 8):
  let digits := [d1, d2, d3],
      largest := 861,
      third_smallest := 618 in
  largest - third_smallest = 243 :=
by
  sorry

end difference_between_largest_and_third_smallest_l379_379753


namespace sequence_general_formula_l379_379792

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h₁ : a 1 = 2)
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) :
  ∀ n, a n = 1 + 2^(n - 1) := 
sorry

end sequence_general_formula_l379_379792


namespace percentage_error_square_area_l379_379719

theorem percentage_error_square_area 
  (a : ℝ) 
  (h_measured_side : 1.08 * a)
  (h_actual_area : a^2)
  (h_erroneous_area : (1.08 * a)^2)
  (h : (h_erroneous_area - h_actual_area) / h_actual_area * 100): 
  ((1.08 * a)^2 - a^2) / a^2 * 100 = 16.64 :=
by
  sorry

end percentage_error_square_area_l379_379719


namespace prove_r_s_l379_379555

noncomputable def distinct_polynomials (P Q : ℝ[X]) : Prop :=
P ≠ Q ∧ degree P > 0 ∧ degree Q > 0

theorem prove_r_s (r s : ℕ) (P Q : ℝ[X]) (h1 : r > s) (h2 : distinct_polynomials P Q) 
  (h3 : ∀ x : ℝ, P.eval x ^ r - P.eval x ^ s = Q.eval x ^ r - Q.eval x ^ s) : (r, s) = (2, 1) :=
sorry

end prove_r_s_l379_379555


namespace fraction_of_outer_circle_not_covered_by_inner_circles_l379_379520

noncomputable def pi := Real.pi

def radius_outer : ℝ := 15
def radius_inner1 : ℝ := 6
def radius_inner2 : ℝ := 8
def radius_inner3 : ℝ := 10

def area_circle (r : ℝ) : ℝ := pi * r^2

def area_outer : ℝ := area_circle radius_outer
def area_inner1 : ℝ := area_circle radius_inner1
def area_inner2 : ℝ := area_circle radius_inner2
def area_inner3 : ℝ := area_circle radius_inner3

def area_combined : ℝ := area_inner1 + area_inner2 + area_inner3
def area_not_covered : ℝ := area_outer - area_combined

def fraction_not_covered : ℝ := area_not_covered / area_outer

theorem fraction_of_outer_circle_not_covered_by_inner_circles :
  fraction_not_covered = 1 / 9 :=
by
  sorry

end fraction_of_outer_circle_not_covered_by_inner_circles_l379_379520


namespace inverse_condition_l379_379841

theorem inverse_condition (a : ℝ) : 
  (∀ x, x ∈ set.Iic 4 → function.injective (λ x, x^2 + 2 * (a - 1) * x + 2)) ↔ a ≤ -3 :=
by
  sorry

end inverse_condition_l379_379841


namespace sequence_solution_l379_379460

theorem sequence_solution (a : ℕ → ℤ) :
  a 0 = -1 →
  a 1 = 1 →
  (∀ n ≥ 2, a n = 2 * a (n - 1) + 3 * a (n - 2) + 3^n) →
  ∀ n, a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by
  -- Detailed proof steps will go here.
  sorry

end sequence_solution_l379_379460


namespace ellipse_vector_sum_eq_zero_l379_379801

def point := ℝ × ℝ

def is_on_ellipse (P : point) : Prop :=
  (P.1^2) / 25 + (P.2^2) / 16 = 1

def F : point := (3, 0)

def vector_sub (P1 P2 : point) : point :=
  (P1.1 - P2.1, P1.2 - P2.2)

def vector_zero : point := (0, 0)

def vector_sum (v1 v2 v3 : point) : point :=
  (v1.1 + v2.1 + v3.1, v1.2 + v2.2 + v3.2)

def vector_magnitude (v : point) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem ellipse_vector_sum_eq_zero
  (A B C : point)
  (hA : is_on_ellipse A)
  (hB : is_on_ellipse B)
  (hC : is_on_ellipse C)
  (h_sum : vector_sum (vector_sub A F) (vector_sub B F) (vector_sub C F) = vector_zero) :
  vector_magnitude (vector_sub A F) + 
  vector_magnitude (vector_sub B F) + 
  vector_magnitude (vector_sub C F) = 48 / 5 :=
sorry

end ellipse_vector_sum_eq_zero_l379_379801


namespace minimum_number_of_guests_l379_379238

def total_food : ℤ := 327
def max_food_per_guest : ℤ := 2

theorem minimum_number_of_guests :
  ∀ (n : ℤ), total_food ≤ n * max_food_per_guest → n = 164 :=
by
  sorry

end minimum_number_of_guests_l379_379238


namespace combined_average_age_of_fifth_graders_teachers_and_parents_l379_379983

theorem combined_average_age_of_fifth_graders_teachers_and_parents
  (num_fifth_graders : ℕ) (avg_age_fifth_graders : ℕ)
  (num_teachers : ℕ) (avg_age_teachers : ℕ)
  (num_parents : ℕ) (avg_age_parents : ℕ)
  (h1 : num_fifth_graders = 40) (h2 : avg_age_fifth_graders = 10)
  (h3 : num_teachers = 4) (h4 : avg_age_teachers = 40)
  (h5 : num_parents = 60) (h6 : avg_age_parents = 34)
  : (num_fifth_graders * avg_age_fifth_graders + num_teachers * avg_age_teachers + num_parents * avg_age_parents) /
    (num_fifth_graders + num_teachers + num_parents) = 25 :=
by sorry

end combined_average_age_of_fifth_graders_teachers_and_parents_l379_379983


namespace sum_series_l379_379408

theorem sum_series : 
  ∑ i in Finset.range (10001 + 1), if i % 2 = 0 then i else -i = 15001 :=
by
  sorry

end sum_series_l379_379408


namespace S_formula_l379_379193

namespace TriangleProof

-- Define variables a, b, c, d as sides of the triangle and A as an angle with the condition A ≠ 90°
variables {a b c d : ℝ} {A : ℝ}

-- Assuming conditions provided
axiom angle_ne_90 (hA : A ≠  90) : Prop

theorem S_formula (hA : A ≠ 90) : 
    let S := (a^2 + d^2 - b^2 - c^2) / 4 * Real.tan A in
    S = (a^2 + d^2 - b^2 - c^2) / 4 * Real.tan A :=
by
    sorry

end TriangleProof

end S_formula_l379_379193


namespace triangle_area_l379_379641

theorem triangle_area {m1 m2 b1 b2 : ℝ} 
  (h_perpendicular : m1 * m2 = -1)
  (h_sum_intercepts : b1 + b2 = 4)
  (h_intersection1 : 12 = 4 * m1 + b1)
  (h_intersection2 : 12 = 4 * m2 + b2) :
  let R := (0, b1)
      S := (0, b2)
  in let B := (4, 12)
       area_BRS := 0.5 * (4 * abs m1) * 4
   in area_BRS = 8 := 
sorry

end triangle_area_l379_379641


namespace compound_interest_calculation_l379_379339

-- Given conditions
def P : ℝ := 20000
def r : ℝ := 0.03
def t : ℕ := 5

-- The amount after t years with compound interest
def A := P * (1 + r) ^ t

-- Prove the total amount is as given in choice B
theorem compound_interest_calculation : 
  A = 20000 * (1 + 0.03) ^ 5 :=
by
  sorry

end compound_interest_calculation_l379_379339


namespace next_to_Denis_l379_379275

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379275


namespace incorrect_statements_l379_379671

-- Definitions for the vectors and necessary vector operations
structure Vector (R : Type*) := 
  (x y z : R)

-- Definitions for vector operations
def parallel {R : Type*} [Field R] (a b : Vector R) : Prop :=
  ∃ (λ : R), b = ⟨λ * a.x, λ * a.y, λ * a.z⟩

def angle_obtuse {R : Type*} [OrderedField R] (a b : Vector R) : Prop :=
  a.x * b.x + a.y * b.y + a.z * b.z < 0

noncomputable def coplanar {R : Type*} [Field R] (A B C P : Vector R) : Prop :=
  ∃ μ ν ξ : R, μ + ν + ξ = 1 ∧ 
  P = ⟨μ * A.x + ν * B.x + ξ * C.x, μ * A.y + ν * B.y + ξ * C.y, μ * A.z + ν * B.z + ξ * C.z⟩

noncomputable def basis_planes {R : Type*} [Field R] (O A B C : Vector R) : Prop :=
  ∃ (λ μ ν : R), A = ⟨λ * O.x, λ * O.y, λ * O.z⟩ ∧ 
                  B = ⟨μ * O.x, μ * O.y, μ * O.z⟩ ∧ 
                  C = ⟨ν * O.x, ν * O.y, ν * O.z⟩

-- Conditions from the problem
def vector_a : Vector ℝ := ⟨2, 2, 1⟩
def vector_b (x : ℝ) : Vector ℝ := ⟨4, -2 + x, x⟩

-- Problem statement in Lean
theorem incorrect_statements :
  ¬ (∀ (a b : Vector ℝ), parallel a b → 
       ∃ (λ : ℝ), b = ⟨λ * a.x, λ * a.y, λ * a.z⟩) ∨ 
  (∃ (A B C O : Vector ℝ) (P : Vector ℝ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    coplanar A B C P ∧ 
    P = ⟨(3 / 4) * O.x + (1 / 8) * B.x + (1 / 8) * C.x, (3 / 4) * O.y + (1 / 8) * B.y + (1 / 8) * C.y, (3 / 4) * O.z + (1 / 8) * B.z + (1 / 8) * C.z⟩ ∧ 
    (O ≠ A ∧ O ≠ B ∧ O ≠ C)) ∨ 
  (∃ (x : ℝ), angle_obtuse vector_a (vector_b x) → 
       x < (4 / 7)) ∨ 
  (∃ (O A B C : Vector ℝ),
    basis_planes O A B C → 
    ∃ μ ν λ : ℝ, μ + ν + λ  ≠ 0 ∧ 
    O ≠ A ∧ O ≠ B ∧ O ≠ C ∧ 
    ¬ coplanar O A B C) :=
sorry

end incorrect_statements_l379_379671


namespace total_oranges_is_correct_l379_379179

/-- Define the number of boxes and the number of oranges per box -/
def boxes : ℕ := 7
def oranges_per_box : ℕ := 6

/-- Prove that the total number of oranges is 42 -/
theorem total_oranges_is_correct : boxes * oranges_per_box = 42 := 
by 
  sorry

end total_oranges_is_correct_l379_379179


namespace roots_of_z4_plus_16_eq_0_l379_379762

noncomputable def roots_of_quartic_eq : Set ℂ :=
  { z | z^4 + 16 = 0 }

theorem roots_of_z4_plus_16_eq_0 :
  roots_of_quartic_eq = { z | z = complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 - complex.I * complex.sqrt 2 ∨
                             z = complex.sqrt 2 - complex.I * complex.sqrt 2 } :=
by
  sorry

end roots_of_z4_plus_16_eq_0_l379_379762


namespace age_of_new_person_l379_379937

theorem age_of_new_person (n : ℕ) (initial_avg_age : ℕ) (new_avg_age : ℕ) (initial_n : ℕ) 
    (h1 : initial_avg_age = 14) (h2 : new_avg_age = 16) (h3 : initial_n = 8) : 
    let total_age_initial := initial_avg_age * initial_n,
        total_age_new := new_avg_age * (initial_n + 1),
        age_of_new_person := total_age_new - total_age_initial
    in age_of_new_person = 32 :=
by 
  let total_age_initial := 14 * 8
  let total_age_new := 16 * 9
  let age_of_new_person := total_age_new - total_age_initial
  exact congr_arg2 Int.sub (by rfl) (by rfl) -- This ensures it computes correctly

end age_of_new_person_l379_379937


namespace find_a1996_l379_379605

theorem find_a1996 :
  ∃ (a : ℕ), (a = 16 ∧ ∃ (a_list : List ℕ) (k_list : List ℕ), 
  (∀ (i : ℕ), i < 1996 → 0 ≠ a_list.getOrElse i 0) ∧
  List.sorted (· < ·) k_list ∧ 
  ∏ n in Finset.range 1996, (1 + n * (x ^ (3 ^ n))) = 
  1 + ∑ m in Finset.range a_list.length, (a_list.getOrElse m 0) * (x ^ (k_list.getOrElse m 0)))) :=
by
  sorry

end find_a1996_l379_379605


namespace commercial_break_total_time_l379_379433

theorem commercial_break_total_time (c1 c2 c3 : ℕ) (c4 : ℕ → ℕ) (interrupt restart : ℕ) 
  (h1 : c1 = 5) (h2 : c2 = 6) (h3 : c3 = 7) 
  (h4 : ∀ i, i < 11 → c4 i = 2) 
  (h_interrupt : interrupt = 3)
  (h_restart : restart = 2) :
  c1 + c2 + c3 + (11 * 2) + interrupt + 2 * restart = 47 := 
  by
  sorry

end commercial_break_total_time_l379_379433


namespace standing_next_to_Denis_l379_379317

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379317


namespace no_such_parallelepipeds_l379_379432

def has_common_point (P_i P_j : Set Point) : Prop :=
  ∃ p : Point, p ∈ P_i ∧ p ∈ P_j

def intersects_iff_mod (P : Fin 12 → Set Point) : Prop :=
  ∀ i j, has_common_point (P i) (P j) ↔ i ≠ j + 1 ∧ i ≠ j - 1

theorem no_such_parallelepipeds
  (P : Fin 12 → Set Point) 
  (Hparallels : ∀ i, edges_parallel_to_coordinate_axes (P i)) 
  (Hmod12 : intersects_iff_mod P) : 
  False :=
by sorry

end no_such_parallelepipeds_l379_379432


namespace triangle_problem_l379_379849

-- Defining the conditions as Lean constructs
variable (a c : ℝ)
variable (b : ℝ := 3)
variable (cosB : ℝ := 1 / 3)
variable (dotProductBACBC : ℝ := 2)
variable (cosB_minus_C : ℝ := 23 / 27)

-- Define the problem as a theorem in Lean 4
theorem triangle_problem
  (h1 : a > c)
  (h2 : a * c * cosB = dotProductBACBC)
  (h3 : a^2 + c^2 = 13) :
  a = 3 ∧ c = 2 ∧ cosB_minus_C = 23 / 27 := by
  sorry

end triangle_problem_l379_379849


namespace determine_k_l379_379905

-- Definitions for vector space and collinearity
def non_zero_vector (v : ℝ^n) : Prop := v ≠ 0
def not_collinear (a b : ℝ^n) : Prop := ¬ collinear a b
def oppositely_collinear (u v : ℝ^n) : Prop := ∃ λ : ℝ, λ < 0 ∧ u = λ • v

theorem determine_k (a b : ℝ^n) (h1 : non_zero_vector a) (h2 : non_zero_vector b) (h3 : not_collinear a b) :
  ∃ k : ℝ, oppositely_collinear (k • a + b) (a + k • b) ∧ k = -1 :=
sorry

end determine_k_l379_379905


namespace ratio_triangle_to_square_l379_379862

theorem ratio_triangle_to_square
  (A B C D E F : Type)
  [square : square A B C D]
  (on_side_AD : is_on_side E A D)
  (AE_ED_ratio : ∃ k : ℕ, AE = k • ED)
  (k_val : k = 3)
  (on_side_BC : is_on_side F B C)
  (BF_FC_ratio : ∃ l : ℕ, BF = l • FC)
  (l_val : l = 3) :
  (area (triangle A E F)) / (area (square A B C D)) = 9 / 32 :=
sorry

end ratio_triangle_to_square_l379_379862


namespace probability_three_same_color_l379_379149

theorem probability_three_same_color (pairs : Fin 10 → Fin 2) (shoes : Fin 20) :
  (∃ (color : Fin 10), 
     ∀ (i j : Fin 3), pairs (shoes i) = pairs (shoes j)) = false :=
by
  sorry

end probability_three_same_color_l379_379149


namespace cost_per_book_eq_three_l379_379741

-- Let T be the total amount spent, B be the number of books, and C be the cost per book
variables (T B C : ℕ)
-- Conditions: Edward spent $6 (T = 6) to buy 2 books (B = 2)
-- Each book costs the same amount (C = T / B)
axiom total_amount : T = 6
axiom number_of_books : B = 2

-- We need to prove that each book cost $3
theorem cost_per_book_eq_three (h1 : T = 6) (h2 : B = 2) : (T / B) = 3 := by
  sorry

end cost_per_book_eq_three_l379_379741


namespace inequality_solution_set_l379_379175

variable (f : ℝ → ℝ)

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, (0 < x ∧ x < y) → f x < f y
def f_one_zero (f : ℝ → ℝ) : Prop := f 1 = 0

-- Proof goal
theorem inequality_solution_set :
  is_odd_function f →
  is_increasing_on_pos f →
  f_one_zero f →
  ∀ x, (x ∈ ((-1 : Set.ℝ) ∩ (0 : Set.ℝ)) ∪ ((0 : Set.ℝ) ∩ (1 : Set.ℝ))) →
  (f x - f (-x)) / x < 0 :=
by
  intros h1 h2 h3 x hx
  sorry

end inequality_solution_set_l379_379175


namespace size_of_intersection_leq_fraction_l379_379057

theorem size_of_intersection_leq_fraction {A B : Finset ℝ} (x : ℝ) (hx : x ∈ {a + b | a ∈ A, b ∈ B}) :
  |A.filter (λ a, (x - a) ∈ B)| ≤ (|{a - b | a ∈ A, b ∈ B}| : ℕ)^2 / |{a + b | a ∈ A, b ∈ B}| :=
by sorry

end size_of_intersection_leq_fraction_l379_379057


namespace equation_of_circle_correct_l379_379234

open Real

def equation_of_circle_through_points (x y : ℝ) :=
  x^2 + y^2 - 4 * x - 6 * y

theorem equation_of_circle_correct :
  ∀ (x y : ℝ),
    (equation_of_circle_through_points (0 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (4 : ℝ) (0 : ℝ) = 0) →
    (equation_of_circle_through_points (-1 : ℝ) (1 : ℝ) = 0) →
    (equation_of_circle_through_points x y = 0) :=
by 
  sorry

end equation_of_circle_correct_l379_379234


namespace paul_sara_see_again_time_l379_379192

theorem paul_sara_see_again_time
  (sara_speed : ℝ) (paul_speed : ℝ) (paths_distance : ℝ) (pond_diameter : ℝ)
  (initial_distance : ℝ) (T : ℝ) :
  sara_speed = 4 ∧ paul_speed = 2 ∧ paths_distance = 250 ∧ pond_diameter = 150 ∧ initial_distance = 250 ∧ T = 50 →
  (T.num + T.denom = 51) :=
sorry

end paul_sara_see_again_time_l379_379192


namespace alberto_more_than_bjorn_and_charlie_l379_379518

theorem alberto_more_than_bjorn_and_charlie (time : ℕ) 
  (alberto_speed bjorn_speed charlie_speed: ℕ) 
  (alberto_distance bjorn_distance charlie_distance : ℕ) :
  time = 6 ∧ alberto_speed = 10 ∧ bjorn_speed = 8 ∧ charlie_speed = 9
  ∧ alberto_distance = alberto_speed * time
  ∧ bjorn_distance = bjorn_speed * time
  ∧ charlie_distance = charlie_speed * time
  → (alberto_distance - bjorn_distance = 12) ∧ (alberto_distance - charlie_distance = 6) :=
by
  sorry

end alberto_more_than_bjorn_and_charlie_l379_379518


namespace putnam_inequality_l379_379556

theorem putnam_inequality (n : ℕ) (x : ℕ → ℝ) : (∑ 1 ≤ i < j ≤ n, |x i + x j|) ≥ ((n - 2) / 2) * (∑ i in finset.range(n), |x i|) := 
sorry

end putnam_inequality_l379_379556


namespace peter_collected_total_money_l379_379588

def price_per_jumbo := 9.00
def price_per_regular := 4.00
def total_pumpkins := 80
def regular_pumpkins := 65

def jumbo_pumpkins := total_pumpkins - regular_pumpkins
def money_from_regular := regular_pumpkins * price_per_regular
def money_from_jumbo := jumbo_pumpkins * price_per_jumbo
def total_money := money_from_regular + money_from_jumbo

theorem peter_collected_total_money : total_money = 395.00 :=
by
  -- conditions used to define all variables
  -- target to prove: total_money = 395.00
  sorry

end peter_collected_total_money_l379_379588


namespace roses_cut_calculation_l379_379318

variable {initialRoses : ℕ}
variable {rosesAfterCutting : ℕ}
variable {rosesCut : ℕ}

theorem roses_cut_calculation (h1 : initialRoses = 7) (h2 : rosesAfterCutting = 20) : 
  rosesCut = rosesAfterCutting - initialRoses → rosesCut = 13 := 
by
  intros h
  rw [h1, h2] at h
  rw [h]
  exact rfl

end roses_cut_calculation_l379_379318


namespace find_a_l379_379083

theorem find_a (a x : ℝ) (h_domain : x ∈ set.Icc (1/2 : ℝ) 1) 
    (h_slope : 1/2 ≤ a - 4 * x ^ 3 ∧ a - 4 * x ^ 3 ≤ 4) : 
  a = 9 / 2 := 
by
  sorry

end find_a_l379_379083


namespace smallest_value_of_x_l379_379653

theorem smallest_value_of_x :
  ∃ x : ℝ, (x / 4 + 2 / (3 * x) = 5 / 6) ∧ (∀ y : ℝ,
    (y / 4 + 2 / (3 * y) = 5 / 6) → x ≤ y) :=
sorry

end smallest_value_of_x_l379_379653


namespace sphere_surface_area_l379_379705

noncomputable def surface_area_of_sphere (AB AC BC AA_1 : ℝ) : ℝ :=
  let cos_B := (BC^2 + AB^2 - AC^2) / (2 * BC * AB) in
  let sin_B := real.sqrt (1 - cos_B^2) in
  let r := AC / (2 * sin_B) in
  let R := real.sqrt (r^2 + (AA_1 / 2)^2) in
  4 * real.pi * R^2

theorem sphere_surface_area : surface_area_of_sphere 3 5 7 2 = (208 * real.pi) / 3 :=
by
  sorry

end sphere_surface_area_l379_379705


namespace oriented_area_ratio_sum_zero_l379_379377

variable {Point : Type*} -- Define the type for points
variable [HasArea Point] -- Assume we have a way to measure the oriented area of triangles
variable [HasDistance Point] -- Assume we have a way to measure distance between points

-- Define the properties of secant and oriented area
def not_on_sides_or_extensions (M A B C : Point) : Prop := sorry
def secant_intersects_sides (M A B C A1 B1 C1 : Point) : Prop := sorry

theorem oriented_area_ratio_sum_zero 
  (M A B C A1 B1 C1 : Point)
  (h₀ : not_on_sides_or_extensions M A B C)
  (h₁ : secant_intersects_sides M A B C A1 B1 C1) :
  area_ratio (triangle_area A B M) (distance M C1) +
  area_ratio (triangle_area B C M) (distance M A1) +
  area_ratio (triangle_area C A M) (distance M B1) = 0 :=
sorry

end oriented_area_ratio_sum_zero_l379_379377


namespace limit_of_one_minus_inverse_l379_379407

theorem limit_of_one_minus_inverse (x : ℕ → ℝ) 
  (h₁ : ∀ n, x n = 1 - (1 / ↑n)) : 
  Filter.Tendsto x Filter.atTop (𝓝 1) :=
by
  sorry

end limit_of_one_minus_inverse_l379_379407


namespace T_n_polynomial_l379_379592

noncomputable def T_n (n : ℕ) (x : ℝ) : ℝ := Real.cos (n * Real.arccos x)

theorem T_n_polynomial (n : ℕ) :
  ∃ p : Polynomial ℝ, 
    Polynomial.degree p = n ∧ 
    Polynomial.leadingCoeff p = 2^(n - 1) ∧ 
    (∀ x, x ∈ Icc (-1:ℝ) 1 → T_n n x = p.eval x) ∧ 
    ∀ k, 1 ≤ k ∧ k ≤ n → T_n n (Real.cos ((2 * k - 1) * π / (2 * n))) = 0 ∧ 
    T_n n (1) = 1 ∧ 
    T_n n (-1) = -1 := 
sorry

end T_n_polynomial_l379_379592


namespace floor_neg_sqrt_eval_l379_379019

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end floor_neg_sqrt_eval_l379_379019


namespace calculation_correct_l379_379662

theorem calculation_correct :
  ∀ (x : ℤ), -2 * (x + 1) = -2 * x - 2 :=
by
  intro x
  calc
    -2 * (x + 1) = -2 * x + -2 * 1 : by sorry
              ... = -2 * x - 2 : by sorry

end calculation_correct_l379_379662


namespace C_is_werewolf_l379_379680

variables (A B C : Type)
variables (is_knight : A → Prop) (is_liar : A → Prop) (is_werewolf : A → Prop)
variables (at_least_one_knight : Prop) (at_least_one_liar : Prop)

-- Conditions
axiom knight_or_liar : ∀ x, is_knight x ∨ is_liar x
axiom not_both_knight_and_werewolf : ∀ x, ¬ (is_knight x ∧ is_werewolf x)
axiom at_least_one_werewolf : ∃ x, is_werewolf x

-- Statements by A and B
axiom A_statement : at_least_one_knight
axiom B_statement : at_least_one_liar

-- Problem: Prove C is a werewolf
theorem C_is_werewolf : is_werewolf C :=
sorry

end C_is_werewolf_l379_379680


namespace identify_pyramid_scheme_l379_379647

-- Definitions for the individual conditions
def high_returns (investment_opportunity : Prop) : Prop := 
  ∃ significantly_higher_than_average_returns : Prop, investment_opportunity = significantly_higher_than_average_returns

def lack_of_information (company : Prop) : Prop := 
  ∃ incomplete_information : Prop, company = incomplete_information

def aggressive_advertising (advertising : Prop) : Prop := 
  ∃ aggressive_ad : Prop, advertising = aggressive_ad

-- Main definition combining all conditions
def is_financial_pyramid_scheme (investment_opportunity company advertising : Prop) : Prop :=
  high_returns investment_opportunity ∧ lack_of_information company ∧ aggressive_advertising advertising

-- Theorem statement
theorem identify_pyramid_scheme 
  (investment_opportunity company advertising : Prop) 
  (h1 : high_returns investment_opportunity)
  (h2 : lack_of_information company)
  (h3 : aggressive_advertising advertising) : 
  is_financial_pyramid_scheme investment_opportunity company advertising :=
by 
  apply And.intro;
  {
    exact h1,
    apply And.intro;
    {
      exact h2,
      exact h3,
    }
  }

end identify_pyramid_scheme_l379_379647


namespace noIntegerRoot_l379_379892

noncomputable def polyWithIntCoeffsSixRoots (p : ℤ[X]) (a1 a2 a3 a4 a5 a6 : ℤ) : Prop :=
  (p.eval a1) = -12 ∧ (p.eval a2) = -12 ∧ (p.eval a3) = -12 ∧
  (p.eval a4) = -12 ∧ (p.eval a5) = -12 ∧ (p.eval a6) = -12 ∧
  a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 ∧
  a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 ∧
  a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 ∧
  a4 ≠ a5 ∧ a4 ≠ a6 ∧
  a5 ≠ a6

theorem noIntegerRoot {p : ℤ[X]} {a1 a2 a3 a4 a5 a6 : ℤ} :
  polyWithIntCoeffsSixRoots p a1 a2 a3 a4 a5 a6 →
  ¬∃ k : ℤ, (p.eval k) = 0 := by
  intros h 
  sorry

end noIntegerRoot_l379_379892


namespace product_equals_sum_only_in_two_cases_l379_379196

theorem product_equals_sum_only_in_two_cases (x y : ℤ) : 
  x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) :=
by 
  sorry

end product_equals_sum_only_in_two_cases_l379_379196


namespace quadratic_sum_zero_l379_379596

noncomputable def f (a b : ℝ) (ci : List ℝ) (x : ℕ → ℝ) (i : ℕ) : ℝ :=
  a * (x i) ^ 2 + b * (x i) + (ci.get! i)

theorem quadratic_sum_zero 
  (a b : ℝ) 
  (ci : List ℝ) 
  (x : ℕ → ℝ) 
  (h : ∀ i, (i < 2020) → f a b ci x i = 0 ) :
  (Finset.sum (Finset.range 2020) (λ i, f a b ci x ((i + 1) % 2020))) = 0 :=
by 
  sorry

end quadratic_sum_zero_l379_379596


namespace tan_theta_expr_l379_379560

theorem tan_theta_expr (θ : ℝ) (x : ℝ) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : sin (θ / 2) = sqrt ((x + 1) / (2 * x)))
  (h3 : 1 < x) :
  tan θ = sqrt (2 * x - 1) / (x - 1) :=
by
  sorry

end tan_theta_expr_l379_379560


namespace emily_initial_marbles_l379_379744

open Nat

theorem emily_initial_marbles (E : ℕ) (h : 3 * E - (3 * E / 2 + 1) = 8) : E = 6 :=
sorry

end emily_initial_marbles_l379_379744


namespace _l379_379224

noncomputable theorem distance_between_parallel_lines :
  ∀ (x y : ℝ),
  (2*x + 4*y + 4 = 0) → (2*x + 4*y - 1 = 0) → real.sqrt 5 / 2 :=
by {
  intros,
  sorry,
}

end _l379_379224


namespace value_of_a_l379_379453

theorem value_of_a (a : ℝ) (x : ℝ) (h : (a - 1) * x^2 + x + a^2 - 1 = 0) : a = -1 :=
sorry

end value_of_a_l379_379453


namespace completing_the_square_l379_379337

theorem completing_the_square :
  ∃ d, (∀ x: ℝ, (x^2 - 6 * x + 5 = 0) → ((x - 3)^2 = d)) ∧ d = 4 :=
by
  -- proof goes here
  sorry

end completing_the_square_l379_379337


namespace average_of_modified_set_l379_379478

theorem average_of_modified_set (a1 a2 a3 a4 a5 : ℝ) (h : (a1 + a2 + a3 + a4 + a5) / 5 = 8) :
  ((a1 + 10) + (a2 - 10) + (a3 + 10) + (a4 - 10) + (a5 + 10)) / 5 = 10 :=
by 
  sorry

end average_of_modified_set_l379_379478


namespace g_value_at_neg3_l379_379604

noncomputable def g : ℚ → ℚ := sorry

theorem g_value_at_neg3 (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 2 * x^2) : 
  g (-3) = 98 / 13 := 
sorry

end g_value_at_neg3_l379_379604


namespace problem_inconsistency_l379_379009

def games_inconsistency (won_first_100 win_percent_total total_games : ℕ) (percent_won_first_100 percent_won_total : ℝ) : Prop :=
  let won_first_100_games := percent_won_first_100 * 100 in
  let total_games_won := percent_won_total * total_games in
  (total_games < 100) ∧ (won_first_100_games = won_first_100) ∧ (total_games_won = win_percent_total)

theorem problem_inconsistency : games_inconsistency 65 70 75 0.65 0.70 :=
  by
  sorry

end problem_inconsistency_l379_379009


namespace range_of_m_l379_379509

theorem range_of_m (m : ℝ) : (∃ x ∈ set.Icc (0 : ℝ) 2, x^3 - 3 * x + m = 0) → m ∈ set.Icc (-2 : ℝ) 2 :=
by {
  sorry
}

end range_of_m_l379_379509


namespace jump_f_at_2_jump_phi_at_2_5_jump_F_at_neg_1_infinite_discontinuity_F_at_0_l379_379451

noncomputable def f : ℝ → ℝ :=
  λ x, if x <= 2 then -((1 / 2) * x^2) else x

theorem jump_f_at_2 : 
  discontinuous_at f 2 ∧ (real.lim (λ x, f x) (Filter.lt 2) - real.lim (λ x, f x) (Filter.gt 2) = 4) :=
sorry

noncomputable def phi : ℝ → ℝ :=
  λ x, if 0 ≤ x ∧ x ≤ 1 then 2 * real.sqrt x 
       else if 1 < x ∧ x < 2.5 then 4 - 2 * x 
       else if 2.5 ≤ x then 2 * x - 7 
       else 0 -- an arbitrary value for undefined cases

theorem jump_phi_at_2_5 : 
  discontinuous_at phi 2.5 ∧ (real.lim (λ x, phi x) (Filter.lt 2.5) - real.lim (λ x, phi x) (Filter.gt 2.5) = -1) :=
sorry

noncomputable def F : ℝ → ℝ :=
  λ x, if x < -1 then 2 * x + 5 
       else if x >= -1 then 1 / x 
       else 0 -- an arbitrary value for undefined cases

theorem jump_F_at_neg_1 : 
  discontinuous_at F (-1) ∧ (real.lim (λ x, F x) (Filter.lt (-1)) - real.lim (λ x, F x) (Filter.gt (-1)) = -4) :=
sorry

theorem infinite_discontinuity_F_at_0 : 
  discontinuous_at F 0 ∧ (real.lim (λ x, F x) (Filter.lt 0) = -real.infinity ∧ real.lim (λ x, F x) (Filter.gt 0) = real.infinity) :=
sorry

end jump_f_at_2_jump_phi_at_2_5_jump_F_at_neg_1_infinite_discontinuity_F_at_0_l379_379451


namespace find_R_l379_379162

noncomputable def R (z : ℂ) : ℂ := z^2 + z - 1

theorem find_R :
  ∃ Q : ℂ → ℂ, ∀ z : ℂ, z^(2023) - 1 = (z^3 - 1) * Q(z) + R(z) ∧ degree R < 3 :=
begin
  -- Define the polynomial to simplify the expressions
  use (λ z, (z^(2023) - 1 - R(z)) / (z^3 - 1)),
  intro z,
  split,
  { calc
      z^(2023) - 1
        = (z^3 - 1) * ((z^(2023) - 1 - R(z)) / (z^3 - 1)) + R(z) : by field_simp [mul_comm]
    , -- Further steps to be proven
  , 
  },

-- Degree proof
  sorry
end

end find_R_l379_379162


namespace new_rope_length_l379_379706

section GrazingArea

def initial_rope_length : ℝ := 9
def additional_area : ℝ := 1408

def radius_squared (r : ℝ) : ℝ := r * r
def grazing_area (r : ℝ) : ℝ := Real.pi * radius_squared r

theorem new_rope_length :
  let new_area := grazing_area initial_rope_length + additional_area in
  let new_length := Real.sqrt (new_area / Real.pi) in
  new_length = 23 :=
sorry

end GrazingArea

end new_rope_length_l379_379706


namespace find_x_l379_379025

theorem find_x (x : ℕ) : (x > 20) ∧ (x < 120) ∧ (∃ y : ℕ, x = y^2) ∧ (x % 3 = 0) ↔ (x = 36) ∨ (x = 81) :=
by
  sorry

end find_x_l379_379025


namespace prop_2_prop_3_l379_379045

variables (m n : Line) (α β : Plane)

-- Conditions for proposition ②
axiom perp_1 : m ⟂ n
axiom perp_2 : m ⟂ α
axiom not_sub : ¬ (n ⊆ α)

-- Proposition ②: If \(m \perp n\), \(m \perp α\), and \(n \not\subset α\), then \(n \parallel α\).
theorem prop_2 : (n ∥ α) := sorry

-- Conditions for proposition ③
axiom perp_3 : α ⟂ β
axiom perp_4 : m ⟂ α
axiom perp_5 : n ⟂ β

-- Proposition ③: If \(\alpha \perp \beta\), \(m \perp α\), and \(n \perp \beta\), then \(m \perp n\).
theorem prop_3 : (m ⟂ n) := sorry

end prop_2_prop_3_l379_379045


namespace q_minus_one_div_floor_fact_div_q_l379_379159

theorem q_minus_one_div_floor_fact_div_q (n q : ℤ) (hn : n ≥ 5) (hq1 : 2 ≤ q) (hq2 : q ≤ n) : (q - 1) ∣ (⌊((n - 1)! : ℚ) / q⌋) := 
sorry

end q_minus_one_div_floor_fact_div_q_l379_379159


namespace concurrency_on_euler_line_l379_379872

theorem concurrency_on_euler_line 
  (A B C D E F X Y Z : Type)
  [Triangle A B C]
  [IncircleTouchPoints A B C D E F]
  (hX : AltitudeFoot D E F X)
  (hY : AltitudeFoot E D F Y)
  (hZ : AltitudeFoot F D E Z) : 
  Concurrence AX BY CZ (EulerLine D E F) :=
sorry

end concurrency_on_euler_line_l379_379872


namespace root_of_equation_l379_379105

theorem root_of_equation : 
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = (x - 1) / x) →
  f (4 * (1 / 2)) = (1 / 2) :=
by
  sorry

end root_of_equation_l379_379105


namespace pairs_of_edges_determine_plane_l379_379100

-- Definition of a rectangular cuboid and its edge properties.
def rectangular_cuboid := { edges : fin 12 // 
  ∃ l w h : ℕ, l != w ∧ w != h ∧ h != l ∧
  ∀ e, 
    ((e < 4 ∨ (4 ≤ e ∧ e < 8) ∨ (8 ≤ e ∧ e < 12)) ∧
    (∃ i j : ℕ, (i != j) ∧ 
      (i < 3 ∧ j < 3 ∧ i != j))) }

-- Theorem to prove the number of unordered pairs of edges that determine a plane.
theorem pairs_of_edges_determine_plane (cuboid : rectangular_cuboid) : 
  finset.card (finset.unordered_pairs cuboid.edges) = 66 :=
  sorry

end pairs_of_edges_determine_plane_l379_379100


namespace Lopez_family_seating_arrangements_l379_379910

theorem Lopez_family_seating_arrangements :
  let adults : Finset String := {"Mr. Lopez", "Mrs. Lopez"}
  let children : Finset String := {"boy", "girl"}
  let family : Finset String := adults ∪ children
  ∃ (arrangements : Finset (Finset String)),
    (∃ driver front_passenger back_passenger1 back_passenger2,
       driver ∈ adults ∧
       front_passenger ∈ family ∧
       front_passenger ≠ driver ∧
       back_passenger1 ∈ family ∧
       back_passenger1 ≠ driver ∧
       back_passenger1 ≠ front_passenger ∧
       back_passenger2 ∈ family ∧
       back_passenger2 ≠ driver ∧
       back_passenger2 ≠ front_passenger ∧
       back_passenger2 ≠ back_passenger1 ∧
       arrangements = {driver, front_passenger, back_passenger1, back_passenger2}) ∧
    arrangements.card = 12 :=
sorry
\
end Lopez_family_seating_arrangements_l379_379910


namespace arithmetic_seq_inequality_l379_379165

variable {α : Type*} [OrderedField α]

/-- Mathematical Problem:
    Let {a_n} be a positive arithmetic sequence such that n ∈ ℕ* and n ≥ 2.
    Prove: (1 + 1/a_1)(1 + 1/a_2) ... (1 + 1/a_n) ≤ (1 + (a_1 + a_n) / (2 a_1 a_n))^n
-/
theorem arithmetic_seq_inequality (a : ℕ → α) (d : α) (n : ℕ) (h_pos : ∀ i, a i > 0)
  (h_arith_seq : ∀ i, a i = a 1 + (i - 1) * d) (h_nat : 2 ≤ n) :
  (∏ i in Finset.range n, (1 + 1 / a (1 + i))) ≤ (1 + (a 1 + a n) / (2 * a 1 * a n))^n :=
by
  sorry

end arithmetic_seq_inequality_l379_379165


namespace probability_of_x_such_that_there_exists_y_l379_379102

theorem probability_of_x_such_that_there_exists_y 
  (x : ℤ)
  (hx : 1 ≤ x ∧ x ≤ 15) :
  (∃ y : ℤ, x * y - 6 * x - 3 * y = 3) →
  (∃ n : ℕ, n = (Finset.filter (λ x, ∃ y : ℤ, x * y - 6 * x - 3 * y = 3) (Finset.range 16)).card ∧ n = 3) ∧ 3 / 15 = 1 / 5 :=
by
  sorry

end probability_of_x_such_that_there_exists_y_l379_379102


namespace g_composed_g_has_exactly_two_distinct_real_roots_l379_379565

theorem g_composed_g_has_exactly_two_distinct_real_roots (d : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 + 4 * x + d) = 0 ∧ (y^2 + 4 * y + d) = 0) ↔ d = 8 :=
sorry

end g_composed_g_has_exactly_two_distinct_real_roots_l379_379565


namespace schoolchildren_lineup_l379_379309

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379309


namespace probability_at_least_one_die_less_3_l379_379326

-- Definitions
def total_outcomes_dice : ℕ := 64
def outcomes_no_die_less_3 : ℕ := 36
def favorable_outcomes : ℕ := total_outcomes_dice - outcomes_no_die_less_3
def probability : ℚ := favorable_outcomes / total_outcomes_dice

-- Theorem statement
theorem probability_at_least_one_die_less_3 :
  probability = 7 / 16 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_die_less_3_l379_379326


namespace no_permutations_satisfy_condition_l379_379030

theorem no_permutations_satisfy_condition :
  ∀ (b : Fin 7 → Fin 8), Function.Bijective b →
  (∏ i : Fin 7, (b i + i + 1) / 2) ≤ fact 7 → (∏ i : Fin 7, (b i + i + 1) / 2) = fact 7 :=
by
  sorry

end no_permutations_satisfy_condition_l379_379030


namespace mrs_hilt_rows_of_pies_l379_379580

def number_of_pies (pecan_pies: Nat) (apple_pies: Nat) : Nat := pecan_pies + apple_pies

def rows_of_pies (total_pies: Nat) (pies_per_row: Nat) : Nat := total_pies / pies_per_row

theorem mrs_hilt_rows_of_pies :
  let pecan_pies := 16 in
  let apple_pies := 14 in
  let pies_per_row := 5 in
  rows_of_pies (number_of_pies pecan_pies apple_pies) pies_per_row = 6 :=
by 
  sorry

end mrs_hilt_rows_of_pies_l379_379580


namespace f_even_and_periodic_l379_379614

def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin (2 * x))^2

theorem f_even_and_periodic :
  (∀ x, f x = f (-x)) ∧ (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π / 2) :=
by
  sorry

end f_even_and_periodic_l379_379614


namespace derivative_at_zero_l379_379786

def f (x : ℝ) : ℝ := (x^2 - 2 * x) * Real.exp x

theorem derivative_at_zero : (deriv f 0) = -2 :=
by
  sorry

end derivative_at_zero_l379_379786


namespace integral_fx_l379_379457

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_fx :
  ∫ x in -Real.pi..0, f x = -2 - (1/2) * Real.pi ^ 2 :=
by
  sorry

end integral_fx_l379_379457


namespace min_value_x_minus_y_l379_379458

theorem min_value_x_minus_y (x y : ℝ) (h1 : 0 ≤ x ∧ x ≤ 2 * real.pi) (h2 : 0 ≤ y ∧ y ≤ 2 * real.pi)
  (h3 : 2 * real.sin x * real.cos y - real.sin x + real.cos y = 1 / 2) : x - y = -real.pi / 2 :=
sorry

end min_value_x_minus_y_l379_379458


namespace reflection_image_sum_slope_intercept_l379_379950

theorem reflection_image_sum_slope_intercept (m b : ℝ) 
  (h1 : ∃ x y, (x, y) = (1, 2) ∧ (x' y', (x', y') = (7, 6) ∧ reflects_across_line m b (x, y) = (x', y'))) :
  m + b = 8.5 := 
by 
  sorry

end reflection_image_sum_slope_intercept_l379_379950


namespace fewest_digits_to_erase_is_2_l379_379620

def sum_of_digits (n : Nat) : Nat := String.foldl (λ acc d => acc + (d.to_nat - '0'.to_nat)) 0 (toString n)

def is_divisible_by_9 (n : Nat) : Bool := sum_of_digits n % 9 = 0

def fewest_digits_to_erase (original_number : Nat) : Nat :=
  if original_number = 123454321 then 2 else 0  -- This matches the condition

theorem fewest_digits_to_erase_is_2 :
  ∀ n : Nat, (n = 123454321) → fewest_digits_to_erase n = 2 :=
by
  intro n hn
  rw [hn]
  unfold fewest_digits_to_erase
  simp
  sorry

end fewest_digits_to_erase_is_2_l379_379620


namespace integer_n_satisfies_conditions_l379_379332

theorem integer_n_satisfies_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n < 127 ∧ 126 * n ≡ 103 [MOD 127] ∧ n = 24 :=
begin
  sorry
end

end integer_n_satisfies_conditions_l379_379332


namespace correct_option_l379_379667

variable (a : ℝ)

theorem correct_option (h1 : 5 * a^2 - 4 * a^2 = a^2)
                       (h2 : a^7 / a^4 = a^3)
                       (h3 : (a^3)^2 = a^6)
                       (h4 : a^2 * a^3 = a^5) : 
                       a^7 / a^4 = a^3 := 
by
  exact h2

end correct_option_l379_379667


namespace compute_d_e_sum_l379_379563

variables (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ)

theorem compute_d_e_sum 
  (h : ∀ x : ℝ, 
        x^8 - 2 * x^7 + 2 * x^6 - 2 * x^5 + 2 * x^4 - 2 * x^3 + 2 * x^2 - 2 * x + 1 = 
        (x^2 + d₁ * x + e₁) * (x^2 + d₂ * x + e₂) * (x^2 + d₃ * x + e₃) * (x^2 + d₄ * x + e₄)) :
  d₁ * e₁ + d₂ * e₂ + d₃ * e₃ + d₄ * e₄ = -2 := 
begin 
  sorry
end

end compute_d_e_sum_l379_379563


namespace point_location_l379_379021

variable {Point : Type} (M : Point) [Plane : Type] (α : Plane) [Line : Type] (a : Line)

-- Line a passes through point M which is outside of plane α
axiom line_contains_point : M ∈ a
axiom point_outside_plane : M ∉ α

theorem point_location :
  M ∉ α ∧ M ∈ a :=
by
  apply And.intro
  · exact point_outside_plane
  · exact line_contains_point

end point_location_l379_379021


namespace jenni_age_l379_379964

theorem jenni_age 
    (B J : ℤ)
    (h1 : B + J = 70)
    (h2 : B - J = 32) : 
    J = 19 :=
by
  sorry

end jenni_age_l379_379964


namespace circle_passing_through_points_eq_l379_379230

theorem circle_passing_through_points_eq :
  ∃ D E F, (∀ x y, x^2 + y^2 + D*x + E*y + F = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) ∧
  (D = -4 ∧ E = -6 ∧ F = 0) :=
begin
  sorry
end

end circle_passing_through_points_eq_l379_379230


namespace find_courtyard_length_l379_379692

-- Definitions based on conditions
def width_courtyard : ℝ := 18
def length_courtyard (total_area : ℝ) (width : ℝ) : ℝ := total_area / width
def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.10
def number_of_bricks : ℕ := 22500
def area_one_brick : ℝ := brick_length * brick_width
def total_area_covered : ℝ := number_of_bricks * area_one_brick

-- Theorem statement
theorem find_courtyard_length (total_area_covered : ℝ) (width_courtyard : ℝ) : length_courtyard total_area_covered width_courtyard = 25 :=
by
  sorry

end find_courtyard_length_l379_379692


namespace largest_consecutive_odd_integer_sum_l379_379984

theorem largest_consecutive_odd_integer_sum
  (x : Real)
  (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = -378.5) :
  x + 8 = -79.7 + 8 :=
by
  sorry

end largest_consecutive_odd_integer_sum_l379_379984


namespace rotational_transform_preserves_expression_l379_379594

theorem rotational_transform_preserves_expression
  (a b c : ℝ)
  (ϕ : ℝ)
  (a1 b1 c1 : ℝ)
  (x' y' x'' y'' : ℝ)
  (h1 : x'' = x' * Real.cos ϕ + y' * Real.sin ϕ)
  (h2 : y'' = -x' * Real.sin ϕ + y' * Real.cos ϕ)
  (def_a1 : a1 = a * (Real.cos ϕ)^2 - 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.sin ϕ)^2)
  (def_b1 : b1 = a * (Real.cos ϕ) * (Real.sin ϕ) + b * ((Real.cos ϕ)^2 - (Real.sin ϕ)^2) - c * (Real.cos ϕ) * (Real.sin ϕ))
  (def_c1 : c1 = a * (Real.sin ϕ)^2 + 2 * b * (Real.cos ϕ) * (Real.sin ϕ) + c * (Real.cos ϕ)^2) :
  a1 * c1 - b1^2 = a * c - b^2 := sorry

end rotational_transform_preserves_expression_l379_379594


namespace denis_neighbors_l379_379290

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379290


namespace mark_siblings_l379_379181

theorem mark_siblings (total_eggs : ℕ) (eggs_per_person : ℕ) (persons_including_mark : ℕ) (h1 : total_eggs = 24) (h2 : eggs_per_person = 6) (h3 : persons_including_mark = total_eggs / eggs_per_person) : persons_including_mark - 1 = 3 :=
by 
  sorry

end mark_siblings_l379_379181


namespace total_wheels_l379_379552

def cars := 2
def car_wheels := 4
def bikes_with_one_wheel := 1
def bikes_with_two_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def roller_skate_wheels := 3 -- since one is missing a wheel
def wheelchair_wheels := 6 -- 4 large + 2 small wheels
def wagon_wheels := 4

theorem total_wheels : cars * car_wheels + 
                        bikes_with_one_wheel * 1 + 
                        bikes_with_two_wheels * 2 + 
                        trash_can_wheels + 
                        tricycle_wheels + 
                        roller_skate_wheels + 
                        wheelchair_wheels + 
                        wagon_wheels = 31 :=
by
  sorry

end total_wheels_l379_379552


namespace correct_option_l379_379669

-- Define the variable 'a' as a real number
variable (a : ℝ)

-- Define propositions for each option
def option_A : Prop := 5 * a ^ 2 - 4 * a ^ 2 = 1
def option_B : Prop := (a ^ 7) / (a ^ 4) = a ^ 3
def option_C : Prop := (a ^ 3) ^ 2 = a ^ 5
def option_D : Prop := a ^ 2 * a ^ 3 = a ^ 6

-- State the main proposition asserting that option B is correct and others are incorrect
theorem correct_option :
  option_B a ∧ ¬option_A a ∧ ¬option_C a ∧ ¬option_D a :=
  by sorry

end correct_option_l379_379669


namespace football_starting_lineup_count_l379_379587

theorem football_starting_lineup_count :
  (∃ team : Finset ℕ, team.card = 12 ∧
    (∃ offensive_lineman_candidates : Finset ℕ, offensive_lineman_candidates.card = 4 ∧
    offensive_lineman_candidates ⊆ team) ∧
    (∃ reflex_candidates : Finset ℕ, reflex_candidates.card = 2 ∧
    reflex_candidates ⊆ team) ∧
    (∃ wide_receiver_candidates : Finset ℕ, wide_receiver_candidates.card = 9 ∧
    wide_receiver_candidates ⊆ team ∧
    disjoint offensive_lineman_candidates reflex_candidates ∧
    disjoint offensive_lineman_candidates wide_receiver_candidates ∧
    disjoint reflex_candidates wide_receiver_candidates)) →
    4 * 2 * 1 * 9 = 72 :=
begin
  sorry
end

end football_starting_lineup_count_l379_379587


namespace mass_percentage_H_is_1734_l379_379029

-- Define the compound C4H10
def C4H10 := (4, 10)

-- Define the molar masses
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008

-- Calculating the molar mass of C4H10
def molar_mass_C4H10 (c : ℝ) (h : ℝ) : ℝ :=
  (4 * c) + (10 * h)

-- Calculating the mass of hydrogen in C4H10
def mass_H_in_C4H10 (h : ℝ) : ℝ := 10 * h

-- Calculating the mass percentage of hydrogen in C4H10
def mass_percentage_H_in_C4H10 (c_mass : ℝ) (h_mass : ℝ) : ℝ :=
  (mass_H_in_C4H10 h_mass) / (molar_mass_C4H10 c_mass h_mass) * 100

-- Theorem to prove the mass percentage of H in C4H10
theorem mass_percentage_H_is_1734 :
  mass_percentage_H_in_C4H10 molar_mass_C molar_mass_H = 17.34 := 
sorry

end mass_percentage_H_is_1734_l379_379029


namespace cos_double_angle_value_l379_379788

theorem cos_double_angle_value (α : ℝ) (h : tan (α + π / 4) = 2) : cos (2 * α) = 4 / 5 :=
sorry

end cos_double_angle_value_l379_379788


namespace range_of_e_l379_379060

theorem range_of_e (a b c d e : ℝ)
  (h1 : a + b + c + d + e = 8)
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l379_379060


namespace food_left_after_bbqs_l379_379778

noncomputable def mushrooms_bought : ℕ := 15
noncomputable def chicken_bought : ℕ := 20
noncomputable def beef_bought : ℕ := 10

noncomputable def mushrooms_consumed : ℕ := 5 * 3
noncomputable def chicken_consumed : ℕ := 4 * 2
noncomputable def beef_consumed : ℕ := 2 * 1

noncomputable def mushrooms_left : ℕ := mushrooms_bought - mushrooms_consumed
noncomputable def chicken_left : ℕ := chicken_bought - chicken_consumed
noncomputable def beef_left : ℕ := beef_bought - beef_consumed

noncomputable def total_food_left : ℕ := mushrooms_left + chicken_left + beef_left

theorem food_left_after_bbqs : total_food_left = 20 :=
  by
    unfold total_food_left mushrooms_left chicken_left beef_left
    unfold mushrooms_consumed chicken_consumed beef_consumed
    unfold mushrooms_bought chicken_bought beef_bought
    sorry

end food_left_after_bbqs_l379_379778


namespace inequality_gx_range_of_a_l379_379201

-- Define the functions f and g
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 3)
def g (x : ℝ) : ℝ := abs (x - 1) + 2

-- Prove that g(x) < 5 implies -2 < x < 4
theorem inequality_gx (x : ℝ) : g(x) < 5 ↔ -2 < x ∧ x < 4 := by
  sorry

-- Prove the range of a given the condition
theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) → (a ≥ -1 ∨ a ≤ -5) := by
  sorry

end inequality_gx_range_of_a_l379_379201


namespace area_of_triangle_DEF_l379_379038

theorem area_of_triangle_DEF {s : ℕ} (h : s = 2) : 
  let area := (sqrt 3 / 4) * s^2
  in area = sqrt 3 := 
by 
  simp [h]
  sorry

end area_of_triangle_DEF_l379_379038


namespace probability_at_least_one_die_less_than_3_l379_379328

theorem probability_at_least_one_die_less_than_3 :
  let total_outcomes := 8 * 8,
      favorable_outcomes := total_outcomes - (6 * 6)
  in (favorable_outcomes / total_outcomes : ℚ) = 7 / 16 := by
  sorry

end probability_at_least_one_die_less_than_3_l379_379328


namespace area_of_inscribed_square_l379_379708

-- Define the right triangle with segments m and n on the hypotenuse
variables {m n : ℝ}

-- Noncomputable setting for non-constructive aspects
noncomputable def inscribed_square_area (m n : ℝ) : ℝ :=
  (m * n)

-- Theorem stating that the area of the inscribed square is m * n
theorem area_of_inscribed_square (m n : ℝ) : inscribed_square_area m n = m * n :=
by sorry

end area_of_inscribed_square_l379_379708


namespace most_likely_end_number_l379_379239

def starting_number := 15
def threshold := 51
def possible_moves := {1, 2, 3, 4, 5}

theorem most_likely_end_number
  (f : ℕ → ℕ) -- f denotes the sequence of numbers on the blackboard
  (h0 : f 0 = starting_number)
  (h_step : ∀ n, f (n + 1) = f n + Classical.choice (set.exists_mem_of_finite_ne_empty possible_moves))
  (h_end : ∃ n, f n > threshold) :
  ∃ n, f n = 54 :=
sorry

end most_likely_end_number_l379_379239


namespace num_trains_encountered_l379_379712

noncomputable def train_travel_encounters : ℕ := 5

theorem num_trains_encountered (start_time : ℕ) (duration : ℕ) (daily_departure : ℕ) 
  (train_journey_duration : ℕ) (daily_start_interval : ℕ) 
  (end_time : ℕ) (number_encountered : ℕ) :
  (train_journey_duration = 3 * 24 * 60 + 30) → -- 3 days and 30 minutes in minutes
  (daily_start_interval = 24 * 60) →             -- interval between daily train starts (in minutes)
  (number_encountered = 5) :=
by
  sorry

end num_trains_encountered_l379_379712


namespace inequality_holds_for_all_x_l379_379751

theorem inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ (-2 < k ∧ k < 6) :=
by
  sorry

end inequality_holds_for_all_x_l379_379751


namespace circle_tangent_to_parabola_directrix_and_yaxis_l379_379449

noncomputable def circle_eq (x y: ℝ) (t : ℝ) : Prop := 
  t = 1 ∨ t = -1 ∧
  ((x - t)^2 + (y - 1 / 2)^2 = 1)

theorem circle_tangent_to_parabola_directrix_and_yaxis (t : ℝ)
  (ht: t = 1 ∨ t = -1) :
  ∃ x y : ℝ, 
  circle_eq x y t := by
sory

end circle_tangent_to_parabola_directrix_and_yaxis_l379_379449


namespace total_worth_correct_l379_379598

def row1_gold_bars : ℕ := 5
def row1_weight_per_bar : ℕ := 2
def row1_cost_per_kg : ℕ := 20000

def row2_gold_bars : ℕ := 8
def row2_weight_per_bar : ℕ := 3
def row2_cost_per_kg : ℕ := 18000

def row3_gold_bars : ℕ := 3
def row3_weight_per_bar : ℕ := 5
def row3_cost_per_kg : ℕ := 22000

def row4_gold_bars : ℕ := 4
def row4_weight_per_bar : ℕ := 4
def row4_cost_per_kg : ℕ := 25000

def total_worth : ℕ :=
  (row1_gold_bars * row1_weight_per_bar * row1_cost_per_kg)
  + (row2_gold_bars * row2_weight_per_bar * row2_cost_per_kg)
  + (row3_gold_bars * row3_weight_per_bar * row3_cost_per_kg)
  + (row4_gold_bars * row4_weight_per_bar * row4_cost_per_kg)

theorem total_worth_correct : total_worth = 1362000 := by
  sorry

end total_worth_correct_l379_379598


namespace median_of_right_triangle_l379_379116

theorem median_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  c / 2 = 5 :=
by
  rw [h3]
  norm_num

end median_of_right_triangle_l379_379116


namespace fourth_roots_of_neg_16_l379_379768

theorem fourth_roots_of_neg_16 : 
  { z : ℂ | z^4 = -16 } = { sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I, 
                            sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I } :=
by
  sorry

end fourth_roots_of_neg_16_l379_379768


namespace find_c_l379_379748

theorem find_c (c : ℝ) (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ 3 * x^2 + 12 * x - 27 = 0)
                      (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ 4 * x^2 - 12 * x + 5 = 0) :
                      c = -8.5 :=
by
  sorry

end find_c_l379_379748


namespace tangent_line_at_1_eqn_l379_379027

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2) / x

theorem tangent_line_at_1_eqn : ∀ x y : ℝ, (f 1) = 2 → (Deriv f 1) = -1 → (x = 1) → (y = 2) → 
  x + y - 3 = 0 :=
by 
  intros x y h1 h2 hx hy
  rw [hx, hy, h1, h2]
  exact sorry

end tangent_line_at_1_eqn_l379_379027


namespace sum_of_squares_l379_379108

theorem sum_of_squares (x y : ℝ) (h1 : y + 6 = (x - 3)^2) (h2 : x + 6 = (y - 3)^2) (hxy : x ≠ y) : x^2 + y^2 = 43 :=
sorry

end sum_of_squares_l379_379108


namespace solution_set_of_quadratic_inequality_l379_379624

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end solution_set_of_quadratic_inequality_l379_379624


namespace distribute_handouts_l379_379101

open Nat

theorem distribute_handouts (k n : ℕ) : 
  n > 0 → (binomial (k + n - 1) (n - 1) = distribute_count k n) := 
sorry

noncomputable def distribute_count (k n : ℕ) : ℕ :=
  binomial (k + n - 1) (n - 1)

end distribute_handouts_l379_379101


namespace total_potatoes_now_l379_379908

def initial_potatoes : ℕ := 8
def uneaten_new_potatoes : ℕ := 3

theorem total_potatoes_now : initial_potatoes + uneaten_new_potatoes = 11 := by
  sorry

end total_potatoes_now_l379_379908


namespace is_increasing_interval_l379_379952

noncomputable def f (x : ℝ) : ℝ := Real.logBase (1/2) (x^2 - 2*x - 3)

theorem is_increasing_interval : ∀ x y : ℝ, x < y → x < -1 → y < -1 → f x < f y := 
by 
  intros x y hx hyx hy
  sorry

end is_increasing_interval_l379_379952


namespace calories_in_300g_l379_379180

/-
Define the conditions of the problem.
-/

def lemon_juice_grams := 150
def sugar_grams := 200
def lime_juice_grams := 50
def water_grams := 500

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 390
def lime_juice_calories_per_100g := 20
def water_calories := 0

/-
Define the total weight of the beverage.
-/
def total_weight := lemon_juice_grams + sugar_grams + lime_juice_grams + water_grams

/-
Define the total calories of the beverage.
-/
def total_calories := 
  (lemon_juice_calories_per_100g * lemon_juice_grams / 100) + 
  (sugar_calories_per_100g * sugar_grams / 100) + 
  (lime_juice_calories_per_100g * lime_juice_grams / 100) + 
  water_calories

/-
Prove the number of calories in 300 grams of the beverage.
-/
theorem calories_in_300g : (total_calories / total_weight) * 300 = 278 := by
  sorry

end calories_in_300g_l379_379180


namespace least_perimeter_l379_379953

theorem least_perimeter (a b : ℕ) (ha : a = 36) (hb : b = 45) (c : ℕ) (hc1 : c > 9) (hc2 : c < 81) : 
  a + b + c = 91 :=
by
  -- Placeholder for proof
  sorry

end least_perimeter_l379_379953


namespace trains_clear_time_l379_379990

theorem trains_clear_time :
  let length1 := 250 -- length of the first train in meters
  let length2 := 300 -- length of the second train in meters
  let length3 := 350 -- length of the third train in meters
  let speed1_kmph := 110 -- speed of the first train in kmph
  let speed2_kmph := 90 -- speed of the second train in kmph
  let speed3_kmph := 120 -- speed of the third train in kmph
  let kmph_to_mps := 1000 / 3600 -- conversion factor from kmph to mps

  let speed1_mps := speed1_kmph * kmph_to_mps -- speed of the first train in mps
  let speed2_mps := speed2_kmph * kmph_to_mps -- speed of the second train in mps
  let speed3_mps := speed3_kmph * kmph_to_mps -- speed of the third train in mps

  let relative_speed_1_and_2_mps := speed1_mps - speed2_mps -- relative speed of the first and second train in mps
  let relative_speed_1_and_3_mps := speed1_mps + speed3_mps -- relative speed of the first and third train in mps

  let total_length := length1 + length2 + length3 -- total length of the trains in meters

  let time_to_clear := total_length / relative_speed1_and_3_mps -- time to clear in seconds
  time_to_clear ≈ 14.08 := sorry

end trains_clear_time_l379_379990


namespace triangle_inequality_isosceles_perimeter_l379_379126

def isosceles (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ c = a)

theorem triangle_inequality (a b c : ℝ) : a + b > c ∧ b + c > a ∧ c + a > b :=
  sorry

theorem isosceles_perimeter (a b side1 side2 : ℝ) (h₀ : isosceles a b side1) 
  (h₁ : side1 = 4) (h₂ : side2 = 9) 
  (h₃ : triangle_inequality side1 side1 side2) :
  a + b + side2 = 22 := 
  sorry

end triangle_inequality_isosceles_perimeter_l379_379126


namespace contractor_absent_days_l379_379691

noncomputable def solve_contractor_problem : Prop :=
  ∃ (x y : ℕ), 
    x + y = 30 ∧ 
    25 * x - 750 / 100 * y = 555 ∧
    y = 6

theorem contractor_absent_days : solve_contractor_problem :=
  sorry

end contractor_absent_days_l379_379691


namespace repeating_decimal_sum_is_fraction_l379_379747

noncomputable def repeating_decimal_to_fraction_sum : Prop :=
  let x := 0.123123123.to_rat
  let y := 0.456745674567.to_rat
  let z := 0.8989898989.to_rat
  (x + y + z) = (14786 / 9999)

theorem repeating_decimal_sum_is_fraction : repeating_decimal_to_fraction_sum :=
by
  -- The actual conversion logic would involve showing the transformation
  -- steps for the repeating decimals to the stated fractions, and then the
  -- algebraic summation with a common denominator.
  sorry

end repeating_decimal_sum_is_fraction_l379_379747


namespace combined_6th_grade_percent_is_15_l379_379627

-- Definitions
def annville_students := 100
def cleona_students := 200

def percent_6th_annville := 11
def percent_6th_cleona := 17

def total_students := annville_students + cleona_students
def total_6th_students := (percent_6th_annville * annville_students / 100) + (percent_6th_cleona * cleona_students / 100)

def percent_6th_combined := (total_6th_students * 100) / total_students

-- Theorem statement
theorem combined_6th_grade_percent_is_15 : percent_6th_combined = 15 :=
by
  sorry

end combined_6th_grade_percent_is_15_l379_379627


namespace probability_textbook2_left_of_textbook4_probability_textbook2_left_of_textbook3_left_of_textbook4_l379_379342

-- Definitions for the textbooks and the condition of randomness
def textbooks := {1, 2, 3, 4}

-- Question 1: Probability that Textbook 2 is to the left of Textbook 4
theorem probability_textbook2_left_of_textbook4 : 
  (number_of_arrangements (λ l, l.index 2 < l.index 4) textbooks) / (number_of_arrangements id textbooks) = 1 / 2 := 
sorry

-- Question 2: Probability that Textbook 2 is to the left of Textbook 3, and Textbook 3 is to the left of Textbook 4
theorem probability_textbook2_left_of_textbook3_left_of_textbook4 : 
  (number_of_arrangements (λ l, l.index 2 < l.index 3 ∧ l.index 3 < l.index 4) textbooks) / (number_of_arrangements id textbooks) = 1 / 4 := 
sorry

end probability_textbook2_left_of_textbook4_probability_textbook2_left_of_textbook3_left_of_textbook4_l379_379342


namespace ratio_of_areas_of_triangles_l379_379590

theorem ratio_of_areas_of_triangles
  (A B C D : Type*)
  [triangle_eq A B C]
  (H1 : equilateral_triangle A B C)
  (H2 : lies_on D A C)
  (H3 : angle B D C = 60) :
  area (triangle A D B) / area (triangle C D B) = (2 + sqrt 3) / (3 * sqrt 3 - 1) :=
sorry

end ratio_of_areas_of_triangles_l379_379590


namespace correct_option_l379_379666

variable (a : ℝ)

theorem correct_option (h1 : 5 * a^2 - 4 * a^2 = a^2)
                       (h2 : a^7 / a^4 = a^3)
                       (h3 : (a^3)^2 = a^6)
                       (h4 : a^2 * a^3 = a^5) : 
                       a^7 / a^4 = a^3 := 
by
  exact h2

end correct_option_l379_379666


namespace number_of_labelings_l379_379543

-- Define the concept of a truncated chessboard with 8 squares
structure TruncatedChessboard :=
(square_labels : Fin 8 → ℕ)
(condition : ∀ i j, i ≠ j → square_labels i ≠ square_labels j)

-- Assuming a wider adjacency matrix for "connected" (has at least one common vertex)
def connected (i j : Fin 8) : Prop := sorry

-- Define the non-consecutiveness condition
def non_consecutive (board : TruncatedChessboard) :=
  ∀ i j, connected i j → (board.square_labels i ≠ board.square_labels j + 1 ∧
                          board.square_labels i ≠ board.square_labels j - 1)

-- Theorem statement
theorem number_of_labelings : ∃ c : Fin 8 → ℕ, ∀ b : TruncatedChessboard, non_consecutive b → 
  (b.square_labels = c) := sorry

end number_of_labelings_l379_379543


namespace area_of_ABCD_l379_379633

-- Definitions of the conditions
def shorter_side : ℕ := 7
def num_rectangles : ℕ := 3
def length_ABCD : ℕ := num_rectangles * shorter_side
def width_ABCD : ℕ := 3 * shorter_side

-- The final proof statement
theorem area_of_ABCD :
  length_ABCD * width_ABCD = 441 :=
by
  have h_length : length_ABCD = 21 := by sorry
  have h_width : width_ABCD = 21 := by sorry
  calc
    length_ABCD * width_ABCD = 21 * 21    : by rw [h_length, h_width]
                       ... = 441           : by norm_num

end area_of_ABCD_l379_379633


namespace problem_1_problem_2_l379_379468

noncomputable def S (a : ℕ → ℝ) : ℕ → ℝ
| 0       => 0
| (n + 1) => S a n + a (n + 1)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d, ∀ n, a n = a 1 + (n - 1) * d

theorem problem_1 (a : ℕ → ℝ) (h : ∀ m n ∈ { k : ℕ | k > 0 }, m ≠ n ∨ n = 0 → 
  (2 * S a (m + n) / (m + n) = a m + a n + (a m - a n) / (m - n))): is_arithmetic_sequence a :=
sorry

def c (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a (n + 1) * a (n + 2) - a n ^ 2

theorem problem_2 (a : ℕ → ℝ) (d : ℝ) (hd : d = 1 / 3) (hc : ∀ n, c a n = a (n + 1) * a (n + 2) - a n ^ 2) :
  ∃ (p q : ℕ) (a1 : ℝ), a1 = (1 / 18) ∧ p > 0 ∧ q > 0 ∧ (a p + c a q) ∈ ℤ :=
sorry

end problem_1_problem_2_l379_379468


namespace sum_first_15_terms_is_15_l379_379527

-- Definitions based on conditions.
variable {a : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}
variable {a1 a2 a5 a8 a11 a14 : ℝ}

-- Arithmetic sequence definitions
def a_n (n : ℕ) : ℝ := a + (n - 1) * d

-- Given Condition
def given_condition : Prop := a_n 2 - a_n 5 + a_n 8 - a_n 11 + a_n 14 = 1

-- Sum of the first 15 terms of the sequence
def sum_of_first_15_terms : ℝ := (15 / 2) * (a + a_n 15)

-- Theorem statement
theorem sum_first_15_terms_is_15 (h : given_condition) : sum_of_first_15_terms = 15 := 
  sorry

end sum_first_15_terms_is_15_l379_379527


namespace zircon_halfway_distance_l379_379622

-- Defining the given conditions
def perigee : ℝ := 3  -- Closest distance to star in AU
def apogee : ℝ := 15  -- Furthest distance from star in AU

-- Definition of the problem
theorem zircon_halfway_distance :
  ellipse semimajor semiminor star focus → -- Orbit is elliptical
  focus = star →                          -- Star is at one focus
  semimajor = (perigee + apogee) / 2 →    -- Semi-major axis calculation
  semiminor = sqrt ((semimajor^2) - (focus - star)^2) → -- Semi-minor using focus distance
  halfway_distance = semimajor :=         -- Proving distance at halfway is semi-major axis
sorry

end zircon_halfway_distance_l379_379622


namespace sum_of_integers_ways_l379_379921

theorem sum_of_integers_ways (n : ℕ) (h : n > 0) : 
  ∃ ways : ℕ, ways = 2^(n-1) := sorry

end sum_of_integers_ways_l379_379921


namespace equation_of_line_passing_through_ellipse_midpoint_l379_379485

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem equation_of_line_passing_through_ellipse_midpoint
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (1, 1))
  (hA : ellipse x1 y1)
  (hB : ellipse x2 y2)
  (midAB : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1) :
  ∃ (a b c : ℝ), a = 4 ∧ b = 3 ∧ c = -7 ∧ a * P.2 + b * P.1 + c = 0 :=
sorry

end equation_of_line_passing_through_ellipse_midpoint_l379_379485


namespace p_sufficient_but_not_necessary_for_q_l379_379787

def condition_p (x : ℝ) : Prop := x^2 - 9 > 0
def condition_q (x : ℝ) : Prop := x^2 - (5 / 6) * x + (1 / 6) > 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x, condition_p x → condition_q x) ∧ ¬(∀ x, condition_q x → condition_p x) :=
sorry

end p_sufficient_but_not_necessary_for_q_l379_379787


namespace find_line_m_l379_379739

-- Define the conditions of the problem
def line_equation_ell (x y : ℝ) : Prop := 3 * x + 4 * y = 0
def point_Q : ℝ × ℝ := (-3, 2)
def point_Qdd : ℝ × ℝ := (-4, -3)
def lines_intersect_at_origin (L1 L2 : ℝ × ℝ → Prop) : Prop :=
  ∃ x y : ℝ, x = 0 ∧ y = 0 ∧ L1 (x, y) ∧ L2 (x, y)

-- Declare the proof problem
theorem find_line_m :
  ∃ (m: ℝ × ℝ → Prop), 
    lines_intersect_at_origin line_equation_ell m ∧
    (reflect (reflect point_Q line_equation_ell) m = point_Qdd) ∧
    (∀ x y : ℝ, m (x, y) ↔ 7 * x - y = 0) :=
sorry

end find_line_m_l379_379739


namespace larger_number_is_21_l379_379640

theorem larger_number_is_21 (x y : ℤ) (h1 : x + y = 35) (h2 : x - y = 7) : x = 21 := 
by 
  sorry

end larger_number_is_21_l379_379640


namespace yellow_square_area_l379_379381

theorem yellow_square_area (s k : ℝ) (h_cross_eq: k = s / 4)
  (h_cross_area : (0.5 * s^2) = k^2 + 4 * (s - k)^2 / 2) :
  (k^2) / (s^2) * 100 = 6.25 :=
by
  -- Conditions of the problem
  have h_total_flag_area : s^2 = s * s, from by sorry
  have h_yellow_square_area : k^2 = (s / 4)^2, from by sorry
  
  -- Proof of correct answer
  have h_yellow_area_percent : (k^2 / s^2) * 100 = 6.25, from by sorry
  
  show (k^2 / s^2) * 100 = 6.25, from h_yellow_area_percent

end yellow_square_area_l379_379381


namespace pulled_distance_l379_379147

noncomputable def distance_pulled (L : ℝ) (d1 : ℝ) (h_slide : ℝ) : ℝ :=
  let h1 := real.sqrt (L^2 - d1^2) 
  let h2 := h1 - h_slide
  let d2 := real.sqrt (L^2 - h2^2)
  d2 - d1

theorem pulled_distance :
  distance_pulled 25 7 4 = 8 := 
by
  sorry

end pulled_distance_l379_379147


namespace jordan_rectangle_width_l379_379413

theorem jordan_rectangle_width
  (carol_length : ℝ)
  (carol_width : ℝ)
  (jordan_length : ℝ)
  (area_equal : carol_length * carol_width = jordan_length * jordan_width)
  (carol_length_val : carol_length = 15)
  (carol_width_val : carol_width = 20)
  (jordan_length_val : jordan_length = 6) :
  jordan_width = 50 :=
by {
  -- Definitions from conditions
  let carol_area := carol_length * carol_width,
  let jordan_area := jordan_length * jordan_width,

  -- Use the condition areas are equal
  have h1 : carol_area = jordan_area,
  {
    rw [carol_length_val, carol_width_val, jordan_length_val],
    exact area_equal,
  },

  -- Calculate the area of Carol's rectangle
  have h2 : carol_area = 300,
  {
    rw [carol_length_val, carol_width_val],
    norm_num,
  },

  -- Use the fact that the areas are equal to find the width of Jordan's rectangle
  have h3 : 300 = 6 * jordan_width,
  {
    rw [← h2, ← h1],
  },

  -- Solve for the width of Jordan's rectangle
  have h4 : jordan_width = 50,
  {
    linarith,
  },

  -- Conclusion
  exact h4,
}

end jordan_rectangle_width_l379_379413


namespace area_of_hexagon_l379_379156

open Complex

-- Define the polynomial whose roots are the points
def poly := Polynomial.X ^ 6 + 6 * Polynomial.X ^ 3 - 216

-- Define the points in the complex plane
def points : Fin 6 → ℂ 
| 0 => Complex.cpow 12 (Complex.of_real 2/3 * Complex.i)
| 1 => Complex.cpow 12 (Complex.of_real 4/3 * Complex.i)
| 2 => Complex.cpow 12 (Complex.of_real 0)
| 3 => -Complex.cpow 18 (Complex.of_real 2/3 * Complex.i)
| 4 => -Complex.cpow 18 (Complex.of_real 4/3 * Complex.i)
| 5 => -Complex.cpow 18 (Complex.of_real 0)

-- Prove the area of the hexagon formed by these points is 9√3
theorem area_of_hexagon :
  IsConvexHexagon (points 0) (points 1) (points 2) (points 3) (points 4) (points 5) →
  (calculate_area (points 0) (points 1) (points 2) (points 3) (points 4) (points 5) = 9 * Real.sqrt 3) :=
by
  sorry

-- Helper function to check if given points form a convex hexagon
def IsConvexHexagon 
  (P1 P2 P3 P4 P5 P6 : ℂ) : Prop :=
  -- define what it means for these points to form a convex hexagon
  sorry

-- Helper function to calculate the area of the hexagon from six points
def calculate_area 
  (P1 P2 P3 P4 P5 P6 : ℂ) : ℝ :=
  -- define the area calculation based on given points
  sorry

end area_of_hexagon_l379_379156


namespace fraction_of_full_fare_half_ticket_l379_379374

theorem fraction_of_full_fare_half_ticket (F R : ℝ) 
  (h1 : F + R = 216) 
  (h2 : F + (1/2)*F + 2*R = 327) : 
  (1/2) = 1/2 :=
by
  sorry

end fraction_of_full_fare_half_ticket_l379_379374


namespace Denis_next_to_Anya_Gena_l379_379269

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379269


namespace min_balls_draw_l379_379393

theorem min_balls_draw (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 2016) : ∃ n, n = 22122 ∧
  (∀ draws : ℕ → ℕ, (∑ i in (finset.range 2016), (draws i)) = n → 
    ∃ b, draws b ≥ 12) :=
sorry

end min_balls_draw_l379_379393


namespace standing_next_to_Denis_l379_379315

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379315


namespace share_of_Bs_profit_l379_379387

-- We define investments and profit.
variables {A B C : ℝ} (total_profit : ℝ)
-- Conditions from the problem.
def investments_condition_1 : Prop := A = 3 * B
def investments_condition_2 : Prop := B = (2/3) * C
def total_profit_condition : Prop := total_profit = 8800

-- Define the share of B in the profit.
def share_of_B (B C total_profit : ℝ) : ℝ := ((2/3 * C) / (7/3 * C)) * total_profit

-- Statement of the theorem to be proved.
theorem share_of_Bs_profit
  (C : ℝ)
  (hA : investments_condition_1 A B)
  (hB : investments_condition_2 B C)
  (hprofit : total_profit_condition total_profit) :
  share_of_B B C total_profit = 2514.29 :=
by
  sorry

end share_of_Bs_profit_l379_379387


namespace calculate_div_expression_l379_379410

variable (x y : ℝ)

theorem calculate_div_expression : (6 * x^3 * y^2) / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end calculate_div_expression_l379_379410


namespace Denis_next_to_Anya_Gena_l379_379264

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379264


namespace number_of_banana_groups_l379_379216

theorem number_of_banana_groups (total_bananas groups_size : ℕ) 
                                (h_total : total_bananas = 180) 
                                (h_size : groups_size = 18) : 
                                total_bananas / groups_size = 10 :=
by
  rw [h_total, h_size]
  norm_num

end number_of_banana_groups_l379_379216


namespace who_is_next_to_denis_l379_379283

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379283


namespace segment_intersections_l379_379094

variable (A B : Type) [MetricSpace A] [MetricSpace B]
variable (circle1 circle2 : Set A) (point1 point2 : A)
variable [OrthogonalCircles circle1 circle2]
variable [Diameter circle1 (M M') : LineSegment A] 
variable [Diameter circle2 (N N') : LineSegment A]
variable [IntersectsOrthogonally circle1 circle2]

theorem segment_intersections (
    h1 : circle1 = {x | dist x point1 = dist x point2},
    h2 : circle2 = {x | dist x point1 = dist x point2},
    h3 : Orthogonal (diameter (M M')) (diameter (N N')),
    h4 : ∀ x ∈ circle1 ∩ circle2, (x = point1 ∨ x = point2)
  ) : 
  (dist M N = dist point1 M ∨ dist M N = dist point2 M) ∧
  (dist M N' = dist point1 M ∨ dist M N' = dist point2 M) ∧ 
  (dist M' N = dist point1 M' ∨ dist M' N = dist point2 M') ∧ 
  (dist M' N' = dist point1 M' ∨ dist M' N' = dist point2 M') :=
sorry

end segment_intersections_l379_379094


namespace greatest_drop_june_increase_april_l379_379989

-- January price change
def jan : ℝ := -1.00

-- February price change
def feb : ℝ := 3.50

-- March price change
def mar : ℝ := -3.00

-- April price change
def apr : ℝ := 4.50

-- May price change
def may : ℝ := -1.50

-- June price change
def jun : ℝ := -3.50

def greatest_drop : List (ℝ × String) := [(jan, "January"), (mar, "March"), (may, "May"), (jun, "June")]

def greatest_increase : List (ℝ × String) := [(feb, "February"), (apr, "April")]

theorem greatest_drop_june_increase_april :
  (∀ d ∈ greatest_drop, d.1 ≤ jun) ∧ (∀ i ∈ greatest_increase, i.1 ≤ apr) :=
by
  sorry

end greatest_drop_june_increase_april_l379_379989


namespace percentage_of_water_in_juice_l379_379497

-- Define the initial condition for tomato puree water percentage
def puree_water_percentage : ℝ := 0.20

-- Define the volume of tomato puree produced from tomato juice
def volume_puree : ℝ := 3.75

-- Define the volume of tomato juice used to produce the puree
def volume_juice : ℝ := 30

-- Given conditions and definitions, prove the percentage of water in tomato juice
theorem percentage_of_water_in_juice :
  ((volume_juice - (volume_puree - puree_water_percentage * volume_puree)) / volume_juice) * 100 = 90 :=
by sorry

end percentage_of_water_in_juice_l379_379497


namespace find_angle_A_find_c_l379_379515

theorem find_angle_A (a b c : ℝ) (h : b^2 + c^2 - a^2 = bc) : 
  ∃ A : ℝ, 0 < A ∧ A < π ∧ cos A = 1 / 2 ∧ A = π / 3 :=
by 
  use π / 3
  split
  { sorry } -- 0 < π / 3
  split
  { sorry } -- π / 3 < π
  split
  { sorry } -- cos (π / 3) = 1 / 2
  { refl } -- π / 3 = π / 3

theorem find_c (a : ℝ) (h1 : sin (π / 3) = sqrt 3 / 2) (b : ℝ) (h2 : b = 1) 
    (area : ℝ) (h3 : area = (3 * sqrt 3) / 4) : 
  ∃ c : ℝ, c = 3 :=
by 
  use 3
  split
  { sorry } -- Proof that the conditions yield c = 3

end find_angle_A_find_c_l379_379515


namespace solve_z_pow_eq_neg_sixteen_l379_379757

theorem solve_z_pow_eq_neg_sixteen (z : ℂ) :
  z^4 = -16 ↔ 
  z = complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) - complex.I * complex.sqrt(2) ∨ 
  z = complex.sqrt(2) - complex.I * complex.sqrt(2) :=
by
  sorry

end solve_z_pow_eq_neg_sixteen_l379_379757


namespace sum_even_102_to_200_l379_379979

noncomputable def sum_even_integers (a b : ℕ) :=
  let n := (b - a) / 2 + 1
  in (n * (a + b)) / 2

theorem sum_even_102_to_200 :
  sum_even_integers 102 200 = 7550 := 
by
  have n : ℕ := (200 - 102) / 2 + 1
  have sum : ℕ := (n * (102 + 200)) / 2
  have n_50 : n = 50 := by sorry
  have sum_7550 : sum = 7550 := by sorry
  exact sum_7550 

end sum_even_102_to_200_l379_379979


namespace nine_pow_l379_379503

theorem nine_pow (y : ℝ) (h : 9^(3 * y) = 729) : 9^(3 * y - 2) = 9 := 
sorry

end nine_pow_l379_379503


namespace MrsHiltRows_l379_379579

theorem MrsHiltRows :
  let (a : ℕ) := 16
  let (b : ℕ) := 14
  let (r : ℕ) := 5
  (a + b) / r = 6 := by
  sorry

end MrsHiltRows_l379_379579


namespace find_g_l379_379618

-- Given conditions
def line_equation (x y : ℝ) : Prop := y = 2 * x - 10
def parameterization (g : ℝ → ℝ) (t : ℝ) : Prop := 20 * t - 8 = 2 * g t - 10

-- Statement to prove
theorem find_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ x y, line_equation x y → parameterization g t) →
  g t = 10 * t + 1 :=
sorry

end find_g_l379_379618


namespace cos_double_alpha_pi_over2_l379_379507

theorem cos_double_alpha_pi_over2 (α : ℝ) (h : sin α = -2 * cos α) : cos (2 * α + π / 2) = 4 / 5 :=
sorry

end cos_double_alpha_pi_over2_l379_379507


namespace find_a5_l379_379051

-- Define the geometric sequence and terms' sum property
variables (a : ℕ → ℚ) -- sequence terms (ℚ for generality)
variables (S : ℕ → ℚ) -- sum of terms

-- Conditions
axiom sum_of_sequence_property (n : ℕ) : S (2 * n) = 4 * (finset.sum (finset.range n) (λ i, a (2 * i + 1)))
axiom product_condition : a 1 * a 2 * a 3 = 27

-- Goal
theorem find_a5 : a 5 = 3^4 := sorry

end find_a5_l379_379051


namespace probability_area_of_rectangle_greater_than_20_is_2_over_3_l379_379698

-- Define the length of the segment AB
def length_AB : ℝ := 12

-- Probability calculation
def probability_area_greater_than_20 : ℝ := 2 / 3

-- Define the problem statement in Lean
theorem probability_area_of_rectangle_greater_than_20_is_2_over_3 :
  ∀ (C : ℝ), C ∈ set.Icc 0 length_AB → 
  let AC := C in
  let BC := length_AB - C in
  let area := AC * BC in
  ∃ P : ℝ, 
  P = probability_area_greater_than_20 ∧
  ∀ x, 2 < x ∧ x < 10 → area > 20 → P = probability_area_greater_than_20 :=
by
  sorry

end probability_area_of_rectangle_greater_than_20_is_2_over_3_l379_379698


namespace sin_of_angle_sum_l379_379049

theorem sin_of_angle_sum
  (α : ℝ)
  (h : cos (α - π / 6) + sin α = (4 / 5) * sqrt 3) :
  sin (α + 7 * π / 6) = - (4 / 5) :=
by {
  sorry
}

end sin_of_angle_sum_l379_379049


namespace problem1_minimum_value_problem2_range_of_a_l379_379903

-- Define the function f
def f (x a : ℝ) : ℝ := |2 * x - a| + |x + a|

-- Problem 1: Prove that when a = 1, the minimum value of f(x) is 3/2
theorem problem1_minimum_value : ∀ x : ℝ, f x 1 ≥ 3/2 := by
  sorry

-- Problem 2: Prove the range for a such that the inequality holds for x in [1, 2]
theorem problem2_range_of_a : 
  (∃ x : ℝ, x ∈ set.Icc 1 2 ∧ f x a < 5 / x + a) →
  0 < a ∧ a < 6 := by
  sorry

end problem1_minimum_value_problem2_range_of_a_l379_379903


namespace pipe_B_emptying_time_l379_379589

-- Define the fill rate of Pipe A
def rate_of_pipe_A : ℚ := 1 / 60

-- Define the fill rate of both pipes working together
def combined_rate_A_and_B : ℚ := 1 / 180

-- Let T_B be the time in which Pipe B alone can empty the cistern
def time_to_empty_by_pipe_B (rate_of_pipe_A : ℚ) (combined_rate_A_and_B : ℚ) : ℚ :=
  1 / (rate_of_pipe_A - combined_rate_A_and_B)

-- State the theorem, i.e., the proof goal
theorem pipe_B_emptying_time :
  let rate_of_pipe_A := 1 / 60 in
  let combined_rate_A_and_B := 1 / 180 in
  time_to_empty_by_pipe_B rate_of_pipe_A combined_rate_A_and_B = 90 := 
by
  sorry

end pipe_B_emptying_time_l379_379589


namespace distance_AO_is_sqrt_22_l379_379140

-- Define the coordinates of points A and the origin O in three-dimensional space.
def A : ℝ × ℝ × ℝ := (2, 3, 3)
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the Euclidean distance function in three-dimensional space.
def distance (P1 P2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P2.1 - P1.1)^2 + (P2.2 - P1.2)^2 + (P2.3 - P1.3)^2)

-- Theorem stating the distance between point A and the origin O is √22.
theorem distance_AO_is_sqrt_22 : distance A O = real.sqrt 22 :=
by
  sorry

end distance_AO_is_sqrt_22_l379_379140


namespace roots_of_z4_plus_16_eq_0_l379_379763

noncomputable def roots_of_quartic_eq : Set ℂ :=
  { z | z^4 + 16 = 0 }

theorem roots_of_z4_plus_16_eq_0 :
  roots_of_quartic_eq = { z | z = complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 - complex.I * complex.sqrt 2 ∨
                             z = complex.sqrt 2 - complex.I * complex.sqrt 2 } :=
by
  sorry

end roots_of_z4_plus_16_eq_0_l379_379763


namespace sum_even_integers_102_to_200_l379_379975

theorem sum_even_integers_102_to_200 : 
  let sequence := list.range' 102 100 
  ∧ (∀ x ∈ sequence, x % 2 = 0) →
  list.sum sequence = 7550 := 
by 
  let sequence := list.range' 102 100 
  have even_sequence : ∀ x ∈ sequence, x % 2 = 0 := 
    sorry 
  have sum_sequence : list.sum sequence = 7550 := 
    sorry 
  exact sum_sequence 

end sum_even_integers_102_to_200_l379_379975


namespace who_is_next_to_Denis_l379_379257

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379257


namespace quadratic_function_asymptotes_and_value_l379_379949

theorem quadratic_function_asymptotes_and_value
  (q : ℝ → ℝ)
  (hq : quadratic q)
  (hasym1 : ∀ x, (x = -3) → q(x) = 0)
  (hasym2 : ∀ x, (x = 2) → q(x) = 0)
  (hval : q(0) = -12) :
  q = λ x, 2 * x^2 + 2 * x - 12 :=
sorry

end quadratic_function_asymptotes_and_value_l379_379949


namespace find_other_leg_l379_379512

theorem find_other_leg (c a b : ℕ) (h_c : c = 15) (h_a : a = 9) : b = 12 :=
by 
  have h : c^2 = a^2 + b^2 := sorry
  have h_sub : 15^2 = 9^2 + b^2 := by rw [h_c, h_a, h]; sorry
  have h_eq : 225 = 81 + b^2 := by norm_num at h_sub; exact h_sub
  have h_sq : b^2 = 144 := by linarith at h_eq; exact h_eq
  have b_val : b = 12 := by rw [← eq_iff_sq_eq_sq, h_sq]; norm_num
  exact b_val

noncomputable def eq_iff_sq_eq_sq (x y : ℝ) : x^2 = y^2 ↔ x = y ∨ x = -y :=
begin
  split;
  intro h,
  { rw [← sub_eq_zero, ← mul_self_eq_zero_iff'],
    use h, },
  { simp [h] }
end

end find_other_leg_l379_379512


namespace soap_bubble_thickness_scientific_notation_l379_379248

theorem soap_bubble_thickness_scientific_notation :
  (0.0007 * 0.001) = 7 * 10^(-7) := by
sorry

end soap_bubble_thickness_scientific_notation_l379_379248


namespace weight_of_6_moles_of_HClO2_correct_l379_379334

def molecular_weight_HClO2 (H_weight : ℝ) (Cl_weight : ℝ) (O_weight : ℝ) : ℝ :=
  H_weight + Cl_weight + 2 * O_weight

def weight_of_6_moles_of_HClO2 (H_weight : ℝ) (Cl_weight : ℝ) (O_weight : ℝ) : ℝ :=
  6 * molecular_weight_HClO2 H_weight Cl_weight O_weight

-- Conditions
def H_weight := 1.01 
def Cl_weight := 35.45
def O_weight := 16.00

-- Theorem to prove
theorem weight_of_6_moles_of_HClO2_correct :
  weight_of_6_moles_of_HClO2 H_weight Cl_weight O_weight = 410.76 :=
by
  -- proof goes here
  sorry

end weight_of_6_moles_of_HClO2_correct_l379_379334


namespace probability_sum_divisible_by_3_l379_379363

theorem probability_sum_divisible_by_3 (dice_count : ℕ) (total_events : ℕ) (valid_events : ℕ) : 
  dice_count = 3 ∧ total_events = 6 * 6 * 6 ∧ valid_events = 72 → 
  (valid_events : ℚ) / (total_events : ℚ) = 1 / 3 :=
by
  intros h
  have h_dice_count : dice_count = 3 := h.1
  have h_total_events : total_events = 6 * 6 * 6 := h.2.1
  have h_valid_events : valid_events = 72 := h.2.2
  sorry

end probability_sum_divisible_by_3_l379_379363


namespace value_of_3a_minus_b_l379_379925
noncomputable def solveEquation : Type := sorry

theorem value_of_3a_minus_b (a b : ℝ) (h1 : a = 3 + Real.sqrt 15) (h2 : b = 3 - Real.sqrt 15) (h3 : a ≥ b) :
  3 * a - b = 6 + 4 * Real.sqrt 15 :=
sorry

end value_of_3a_minus_b_l379_379925


namespace domino_probability_double_l379_379361

theorem domino_probability_double : 
  (∀ (s : set (ℕ × ℕ)), (∀ x, x ∈ s → (fst x < 10) ∧ (snd x < 10)) →
   (fractional (count {x ∈ s | x.1 = x.2})) / (fractional (count s)) = 1 / 10) :=
by
  sorry

end domino_probability_double_l379_379361


namespace denis_neighbors_l379_379294

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

end denis_neighbors_l379_379294


namespace solve_z_pow_eq_neg_sixteen_l379_379756

theorem solve_z_pow_eq_neg_sixteen (z : ℂ) :
  z^4 = -16 ↔ 
  z = complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) - complex.I * complex.sqrt(2) ∨ 
  z = complex.sqrt(2) - complex.I * complex.sqrt(2) :=
by
  sorry

end solve_z_pow_eq_neg_sixteen_l379_379756


namespace floor_of_negative_sqrt_l379_379014

noncomputable def eval_expr : ℚ := -real.sqrt (64 / 9)

theorem floor_of_negative_sqrt : ⌊eval_expr⌋ = -3 :=
by
  -- skip proof
  sorry

end floor_of_negative_sqrt_l379_379014


namespace quadrilateral_diagonals_midpoint_l379_379611

noncomputable def is_midpoint (P A B : Point) : Prop := dist P A = dist P B

theorem quadrilateral_diagonals_midpoint
    (A B C D P : Point)
    (h1 : collinear A C P)
    (h2 : collinear B D P)
    (h3 : Area (Triangle.mk A B P) ^ 2 + Area (Triangle.mk C D P) ^ 2 =
          Area (Triangle.mk B C P) ^ 2 + Area (Triangle.mk A D P) ^ 2)
    : ∃ Q R S, (Q, R ∈ set.insert A (set.insert B (set.insert C (set.singleton D)))) ∧ 
      (Q = R → False) ∧ 
      ((S = Q ∨ S = R) ∧ is_midpoint P Q R) :=
by
  sorry

end quadrilateral_diagonals_midpoint_l379_379611


namespace cos_of_angle_C_in_right_triangle_l379_379861

theorem cos_of_angle_C_in_right_triangle (A B C : Type) 
  [right_angle_triangle A B C]
  (angle_B : B = 90)
  (side_AB : real)
  (side_AC : real)
  (side_AB_eq : AB = 8)
  (side_AC_eq : AC = 6) : 
  cos C = 4 / 5 := 
sorry

end cos_of_angle_C_in_right_triangle_l379_379861


namespace probability_of_at_least_one_2_or_4_l379_379340

theorem probability_of_at_least_one_2_or_4 :
  let p := (1 / 3 : ℚ)
  let q := (2 / 3 : ℚ)
  1 - (q * q) = (5 / 9 : ℚ) :=
by
  let p := (1 / 3 : ℚ)
  let q := (2 / 3 : ℚ)
  have h : 1 - (q * q) = 1 - (4 / 9 : ℚ),
  by rw mul_self_div (2 : ℚ) (3 : ℚ)
  have h2 : 1 - (4 / 9 : ℚ) = (5 / 9 : ℚ),
  by ring
  exact eq.trans h h2

end probability_of_at_least_one_2_or_4_l379_379340


namespace cole_drive_time_l379_379414

theorem cole_drive_time :
  ∃ (D : ℝ), (D / 30 + D / 90 = 2) → ((D / 30) * 60 = 90) :=
begin
  sorry
end

end cole_drive_time_l379_379414


namespace prove_DI_squared_l379_379567

variables (A B C I D : Type)
variables [is_triangle A B C] [acute_triangle A B C]
variables {r : ℝ} {R : ℝ} {h_a : ℝ}
variables (incenter : Incenter A B C I) (altitude_AD : Altitude A D B C h_a)

theorem prove_DI_squared (DI : ℝ) :
  DI^2 = (2 * R - h_a) * (h_a - 2 * r) :=
sorry

end prove_DI_squared_l379_379567


namespace find_inv_f_of_1_l379_379829

def f (x : ℝ) : ℝ := -1 + real.logb 3 (x^2)
def inv_f (y : ℝ) : ℝ := - (real.sqrt (3 ^ (y + 1)))

theorem find_inv_f_of_1 (h : ∀ x : ℝ, x < 0 → f x = 1 → x = -3) :
  inv_f 1 = -3 :=
by sorry

end find_inv_f_of_1_l379_379829


namespace area_bounded_by_sine_correct_l379_379442

noncomputable def area_bounded_by_sine : ℝ :=
  ∫ x in -Real.pi / 2 .. -Real.pi / 4, Real.sin x

theorem area_bounded_by_sine_correct :
  area_bounded_by_sine = (Real.pi / 4 - Real.sqrt 2 / 2) := 
by
  sorry

end area_bounded_by_sine_correct_l379_379442


namespace NES_sale_price_l379_379995

-- Define all the conditions
def SNES_value : ℝ := 150
def trade_in_percentage : ℝ := 0.8
def additional_money : ℝ := 80
def change_received : ℝ := 10
def game_value : ℝ := 30

-- Proving the sale price of the NES
theorem NES_sale_price :
  let trade_in_value := SNES_value * trade_in_percentage in
  let total_spent := trade_in_value + additional_money in
  let total_received := change_received + game_value in
  total_spent - total_received = 160 :=
by
  sorry

end NES_sale_price_l379_379995


namespace convert_to_polar_coordinates_l379_379733

def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.atan (y / x) + if y ≥ 0 then 0 else Real.pi
  (r, θ)

theorem convert_to_polar_coordinates :
  polar_coordinates (-2) (2 * Real.sqrt 3) = (4, 2 * Real.pi / 3) := by
  sorry

end convert_to_polar_coordinates_l379_379733


namespace NES_sale_price_l379_379996

-- Define all the conditions
def SNES_value : ℝ := 150
def trade_in_percentage : ℝ := 0.8
def additional_money : ℝ := 80
def change_received : ℝ := 10
def game_value : ℝ := 30

-- Proving the sale price of the NES
theorem NES_sale_price :
  let trade_in_value := SNES_value * trade_in_percentage in
  let total_spent := trade_in_value + additional_money in
  let total_received := change_received + game_value in
  total_spent - total_received = 160 :=
by
  sorry

end NES_sale_price_l379_379996


namespace triangle_area_sum_l379_379322

theorem triangle_area_sum (P Q R S : Type) [metric_space P] [add_group P]
  (QR : ℝ) (k l : ℕ) (hQR : QR = 30)
  (incircle_trisects_median : ∀ (PQR_points_inc : ∃ X Y Z, (X = inscribed_circle.tangent_point QR P)
    ∧ (Y = inscribed_circle.tangent_point QR R) ∧ (Z = inscribed_circle.tangent_point QR S)), 
      ∃ n, n = median_length P Q R S / 3):
  k + l = 57 :=
by
  sorry

end triangle_area_sum_l379_379322


namespace tangent_line_range_of_a_l379_379117

def f (x : ℝ) : ℝ := x^3 - x
def g (x a : ℝ) : ℝ := x^2 - a^2 + a

theorem tangent_line_range_of_a :
  (∃ (l : ℝ → ℝ), ∃ (s t : ℝ), l = λ x, (3 * s^2 - 1) * (x - s) + s^3 - s ∧ 
      l = λ x, 2 * t * (x - t) + t^2 - a^2 + a)
  → 𝓘 (ℝ) (λ a : ℝ, a ≥ (1 - Real.sqrt 5) / 2 ∧ a ≤ (1 + Real.sqrt 5) / 2) :=
by
  sorry

end tangent_line_range_of_a_l379_379117


namespace sum_slope_and_intercept_PZ_l379_379123

-- Define the coordinates
def X : ℝ × ℝ := (0, 8)
def Y : ℝ × ℝ := (0, 0)
def Z : ℝ × ℝ := (10, 0)

-- Midpoint of XY
def P : ℝ × ℝ := ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- Coordinates of midpoints:
#eval P -- Should evaluate to (0, 4)

-- Define the slope and y-intercept function
def slope (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

def y_intercept (A B : ℝ × ℝ) : ℝ :=
  A.2 - (slope A B * A.1)
  
-- Sum of slope and y-intercept
def sum_slope_and_intercept (A B : ℝ × ℝ) : ℝ :=
  slope A B + y_intercept A B

-- The Lean 4 statement to prove the sum of the slope and y-intercept is 18/5
theorem sum_slope_and_intercept_PZ :
  sum_slope_and_intercept P Z = 18 / 5 := by
  sorry

end sum_slope_and_intercept_PZ_l379_379123


namespace homework_done_l379_379909

theorem homework_done :
  ∃ (D E C Z M : Prop),
    -- Statements of students
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    -- Truth-telling condition
    ((D → D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (E → ¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (C → ¬ D ∧ ¬ E ∧ C ∧ ¬ Z ∧ ¬ M) ∧
    (Z → ¬ D ∧ ¬ E ∧ ¬ C ∧ Z ∧ ¬ M) ∧
    (M → ¬ D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ M)) ∧
    -- Number of students who did their homework condition
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) := 
sorry

end homework_done_l379_379909


namespace Denis_next_to_Anya_Gena_l379_379262

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379262


namespace trajectory_of_M_l379_379174

variable (x y : ℝ)

def circle_center := (-1 : ℝ, 0 : ℝ)
def fixed_point := (1 : ℝ, 0 : ℝ)
def radius := (10 : ℝ)
def ellipse_eq := ( x^2 / 25 + y^2 / 24 = 1 )

theorem trajectory_of_M :
  let C := circle_center in
  let A := fixed_point in
  let r := radius in
  ∃ M : ℝ × ℝ,
    (M.1^2 / 25 + M.2^2 / 24 = 1) :=
  sorry

end trajectory_of_M_l379_379174


namespace movie_guests_end_l379_379855

def initial_total : Nat := 80
def initial_women : Nat := 0.35 * initial_total
def initial_men : Nat := 30
def initial_children : Nat := initial_total - initial_women - initial_men

def first_half_left_men : Nat := 0.25 * initial_men
def first_half_left_children : Nat := 0.10 * initial_children
def first_half_new_guests_men : Nat := 2
def first_half_new_guests_children : Nat := 3

def second_half_left_men : Nat := 0.20 * (initial_men - first_half_left_men + first_half_new_guests_men)
def second_half_left_children : Nat := 0.15 * (initial_children - first_half_left_children + first_half_new_guests_children)
def second_half_left_women : Nat := 2
def second_half_new_guests_children : Nat := 3

def final_men : Nat := (initial_men - first_half_left_men + first_half_new_guests_men) - second_half_left_men
def final_children : Nat := (initial_children - first_half_left_children + first_half_new_guests_children) - second_half_left_children + second_half_new_guests_children
def final_women : Nat := initial_women - second_half_left_women
def final_total : Nat := final_men + final_children + final_women

theorem movie_guests_end (h: final_total = 69) : final_total = 69 := by
  exact h

end movie_guests_end_l379_379855


namespace number_of_correct_statements_l379_379836

variables {a b : ℝ} {f : ℝ → ℝ}

-- Proposition P states:
-- If the derivative implies no extreme points, then it has no zeros
def proposition_P (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Ioo a b, f' x = 0 → ∀ y ∈ Ioo a b, f′ y ≠ 0)

-- Inverse: If no zeros, then no extreme points
def inverse_P (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Ioo a b, f′ x ≠ 0 → ∀ y ∈ Ioo a b, f' y = 0)

-- Converse is the same as Inverse but directly stated
def converse_P (f : ℝ → ℝ) (a b : ℝ) : Prop := inverse_P f a b

-- Contrapositive
def contrapositive_P (f : ℝ → ℝ) (a b : ℝ) : Prop := proposition_P f a b

theorem number_of_correct_statements (f : ℝ → ℝ) (a b : ℝ) :
  (proposition_P f a b → inverse_P f a b ∧ converse_P f a b ∧ contrapositive_P f a b) ∧
  (inverse_P f a b → proposition_P f a b ∧ converse_P f a b ∧ contrapositive_P f a b) ∧
  (converse_P f a b → proposition_P f a b ∧ converse_P f a b ∧ contrapositive_P f a b) ∧
  (contrapositive_P f a b → proposition_P f a b ∧ converse_P f a b ∧ inverse_P f a b) :=
by sorry

end number_of_correct_statements_l379_379836


namespace third_side_length_integer_l379_379525

noncomputable
def side_a : ℝ := 3.14

noncomputable
def side_b : ℝ := 0.67

def is_valid_triangle_side (side: ℝ) : Prop :=
  side_a - side_b < side ∧ side < side_a + side_b

theorem third_side_length_integer (side: ℕ) : is_valid_triangle_side side.to_real → side = 3 :=
  by
  sorry

end third_side_length_integer_l379_379525


namespace probability_not_all_in_same_cafeteria_l379_379252

-- Define the setup
def cafeterias : Type := {A, B}
def students : Type := {a, b, c}

-- Define the scenario: students randomly choose one of the cafeterias
def random_choice (s : students) : cafeterias := sorry

-- Define the condition that checks if all students are in the same cafeteria
def all_in_same_cafeteria : (students → cafeterias) → Prop :=
  λ f, (f a = f b) ∧ (f b = f c)

-- The main theorem proving the desired probability
theorem probability_not_all_in_same_cafeteria :
  probability (λ f : (students → cafeterias), ¬ all_in_same_cafeteria f) = 3 / 4 := 
sorry

end probability_not_all_in_same_cafeteria_l379_379252


namespace cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l379_379656

/- Definitions -/
def is_isosceles_right_triangle (triangle : Type) (a b c : ℝ) (angleA angleB angleC : ℝ) : Prop :=
  -- A triangle is isosceles right triangle if it has two equal angles of 45 degrees and a right angle of 90 degrees
  a = b ∧ angleA = 45 ∧ angleB = 45 ∧ angleC = 90

/- Main Problem Statement -/
theorem cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles
  (T1 T2 : Type) (a1 b1 c1 a2 b2 c2 : ℝ) 
  (angleA1 angleB1 angleC1 angleA2 angleB2 angleC2 : ℝ) :
  is_isosceles_right_triangle T1 a1 b1 c1 angleA1 angleB1 angleC1 →
  is_isosceles_right_triangle T2 a2 b2 c2 angleA2 angleB2 angleC2 →
  ¬ (∃ (a b c : ℝ), a = b ∧ b = c ∧ a = c ∧ (a + b + c = 180)) :=
by
  intros hT1 hT2
  intro h
  sorry

end cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l379_379656


namespace desargues_theorem_l379_379178

open_locale classical

noncomputable theory

variables {A A1 B B1 C C1 O A2 B2 C2 : Type*}
variables [affine_space ℝ (affine_space ℝ (affine_space ℝ euclidean_space))]

/- Assume that lines AA1, BB1, and CC1 intersect at a single point O -/
variables (A A1 B B1 C C1 O: euclidean_space)
variables (hA : A ≠ O) (hA1 : A1 ≠ O) (hB : B ≠ O) (hB1 : B1 ≠ O) (hC : C ≠ O) (hC1 : C1 ≠ O)
variables (hAA1 : line_through A O = line_through A1 O)
variables (hBB1 : line_through B O = line_through B1 O)
variables (hCC1 : line_through C O = line_through C1 O)

/- Define intersection points A2, B2, and C2 -/
def A2 := inter_line (line_through B C) (line_through B1 C1)
def B2 := inter_line (line_through A C) (line_through A1 C1)
def C2 := inter_line (line_through A B) (line_through A1 B1)

/- Prove that points A2, B2, and C2 are collinear -/
theorem desargues_theorem 
  (hA2 : A2 = inter_line (line_through B C) (line_through B1 C1))
  (hB2 : B2 = inter_line (line_through A C) (line_through A1 C1))
  (hC2 : C2 = inter_line (line_through A B) (line_through A1 B1)) :
  collinear {A2, B2, C2} :=
begin
  sorry
end

end desargues_theorem_l379_379178


namespace no_solution_1221_l379_379926

def equation_correctness (n : ℤ) : Prop :=
  -n^3 + 555^3 = n^2 - n * 555 + 555^2

-- Prove that the prescribed value 1221 does not satisfy the modified equation by contradiction
theorem no_solution_1221 : ¬ ∃ n : ℤ, equation_correctness n ∧ n = 1221 := by
  sorry

end no_solution_1221_l379_379926


namespace trapezoid_TQSR_area_l379_379684

-- Definitions of points and trapezoid's area calculation
structure Point where
  x : ℝ
  y : ℝ

-- Defining the rectangle PQRS
structure Rectangle where
  P Q R S : Point
  area : ℝ

-- The instance of the problem
def PQRS : Rectangle :=
{ P := {x := 0, y := 0},
  Q := {x := 5, y := 0},
  R := {x := 5, y := 4},
  S := {x := 0, y := 4},
  area := 20 }

-- Function to compute the area of trapezoid TQSR
def trapezoid_area (T Q S R : Point) : ℝ :=
  let base1 := T.x - Q.x
  let base2 := R.x - S.x
  let height := S.y 
  (base1 + base2) * height / 2

-- Specific points T, Q, S, R for the given diagram
def T : Point := {x := 2, y := 4}
def Q : Point := {x := 5, y := 0}
def S : Point := {x := 0, y := 4}
def R : Point := {x := 5, y := 4}

-- The theorem we want to prove
theorem trapezoid_TQSR_area : trapezoid_area T Q S R = 20 := by
  sorry

end trapezoid_TQSR_area_l379_379684


namespace length_of_bridge_l379_379709

def train_length : ℝ := 200
def train_speed_kmh : ℝ := 72
def crossing_time : ℝ := 16.5986721062315

-- Given definitions and computations
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
noncomputable def distance_covered : ℝ := train_speed_ms * crossing_time
noncomputable def bridge_length : ℝ := distance_covered - train_length

-- Stating the theorem
theorem length_of_bridge :
  bridge_length ≈ 132.17344212463 :=
sorry

end length_of_bridge_l379_379709


namespace who_is_next_to_denis_l379_379281

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379281


namespace students_sign_up_ways_l379_379251

theorem students_sign_up_ways (n m : ℕ) (h_n : n = 6) (h_m : m = 3) : m^n = 729 := 
by
  -- Substitute the values from the conditions
  rw [h_n, h_m]
  -- Verify that 3^6 = 729
  norm_num
  -- Conclude the theorem with the known result
  sorry

end students_sign_up_ways_l379_379251


namespace no_linear_term_in_product_l379_379514

theorem no_linear_term_in_product (m : ℝ) :
  (∀ (x : ℝ), (x - 3) * (3 * x + m) - (3 * x^2 - 3 * m) = 0) → m = 9 :=
by
  intro h
  sorry

end no_linear_term_in_product_l379_379514


namespace binomial_identity_l379_379918

theorem binomial_identity (n m : ℕ) (h : 0 ≤ m ∧ m ≤ n) : 
  (∑ k in Finset.range (n+1), Nat.choose n k * Nat.choose k m) = Nat.choose n m * 2^(n-m) := 
by
  sorry

end binomial_identity_l379_379918


namespace length_of_first_train_l379_379384

theorem length_of_first_train 
  (speed_first_train_kmph : ℝ) 
  (speed_second_train_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (length_second_train_m : ℝ) 
  (hspeed_first : speed_first_train_kmph = 120) 
  (hspeed_second : speed_second_train_kmph = 80) 
  (htime : crossing_time_s = 9) 
  (hlength_second : length_second_train_m = 320.04) :
  ∃ (length_first_train_m : ℝ), abs (length_first_train_m - 180) < 0.1 :=
by
  sorry

end length_of_first_train_l379_379384


namespace pair_count_is_837_l379_379735

noncomputable def number_of_pairs : ℕ :=
  let log_base_2 := Real.log 2
  let log_base_3 := Real.log 3
  let k := log_base_3 / log_base_2 in
  (1:ℕ).upto 4012 |>.sum (λ m, (1:ℕ).filter (λ n, 
    Real.ofInt n * log_base_3 < Real.ofInt m * log_base_2 ∧
    Real.ofInt (m + 3) * log_base_2 < Real.ofInt (n + 1) * log_base_3
  ).card)

theorem pair_count_is_837 : number_of_pairs = 837 := sorry

end pair_count_is_837_l379_379735


namespace equal_areas_of_ngons_l379_379323

noncomputable def area_of_ngon (n : ℕ) (sides : Fin n → ℝ) (radius : ℝ) (circumference : ℝ) : ℝ := sorry

theorem equal_areas_of_ngons 
  (n : ℕ) 
  (sides1 sides2 : Fin n → ℝ) 
  (radius : ℝ) 
  (circumference : ℝ)
  (h_sides : ∀ i : Fin n, ∃ j : Fin n, sides1 i = sides2 j)
  (h_inscribed1 : area_of_ngon n sides1 radius circumference = area_of_ngon n sides1 radius circumference)
  (h_inscribed2 : area_of_ngon n sides2 radius circumference = area_of_ngon n sides2 radius circumference) :
  area_of_ngon n sides1 radius circumference = area_of_ngon n sides2 radius circumference :=
sorry

end equal_areas_of_ngons_l379_379323


namespace circle_passing_through_points_eq_l379_379229

theorem circle_passing_through_points_eq :
  ∃ D E F, (∀ x y, x^2 + y^2 + D*x + E*y + F = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) ∧
  (D = -4 ∧ E = -6 ∧ F = 0) :=
begin
  sorry
end

end circle_passing_through_points_eq_l379_379229


namespace sum_even_integers_102_to_200_l379_379976

theorem sum_even_integers_102_to_200 : 
  let sequence := list.range' 102 100 
  ∧ (∀ x ∈ sequence, x % 2 = 0) →
  list.sum sequence = 7550 := 
by 
  let sequence := list.range' 102 100 
  have even_sequence : ∀ x ∈ sequence, x % 2 = 0 := 
    sorry 
  have sum_sequence : list.sum sequence = 7550 := 
    sorry 
  exact sum_sequence 

end sum_even_integers_102_to_200_l379_379976


namespace circle_passing_through_points_eq_l379_379231

theorem circle_passing_through_points_eq :
  ∃ D E F, (∀ x y, x^2 + y^2 + D*x + E*y + F = 0 ↔ 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) ∧
  (D = -4 ∧ E = -6 ∧ F = 0) :=
begin
  sorry
end

end circle_passing_through_points_eq_l379_379231


namespace sandwich_soda_total_cost_l379_379654

theorem sandwich_soda_total_cost :
  let sandwich_cost := 3.49
      soda_cost := 1.35
      num_sandwiches := 2
      num_sodas := 4
      sandwich_tax_rate := 0.06
      soda_tax_rate := 0.03
      discount_rate := 0.10
      sandwich_cost_total := num_sandwiches * sandwich_cost
      soda_cost_total := num_sodas * soda_cost
      sandwich_tax := sandwich_tax_rate * sandwich_cost_total
      soda_tax := soda_tax_rate * soda_cost_total
      total_before_discount := (sandwich_cost_total + sandwich_tax) + (soda_cost_total + soda_tax)
      discount := discount_rate * total_before_discount
      final_total_cost := total_before_discount - discount
      rounded_total_cost := Float.round (final_total_cost * 100.0) / 100.0
  in rounded_total_cost = 11.66 := by sorry

end sandwich_soda_total_cost_l379_379654


namespace stock_price_end_of_second_year_l379_379434

theorem stock_price_end_of_second_year 
  (initial_price : ℕ) 
  (perc_increase_first_year : ℚ) 
  (perc_decrease_second_year : ℚ) 
  (final_price : ℚ) 
  (h1 : initial_price = 100)
  (h2 : perc_increase_first_year = 0.50)
  (h3 : perc_decrease_second_year = 0.30) :
  final_price = (initial_price : ℚ) * (1 + perc_increase_first_year) * (1 - perc_decrease_second_year) :=
by
  have first_year_price : ℚ := (initial_price : ℚ) * (1 + perc_increase_first_year)
  have second_year_price : ℚ := first_year_price * (1 - perc_decrease_second_year)
  have h4 : first_year_price = 150 := by sorry
  have h5 : second_year_price = 105 := by sorry
  show final_price = 105 from h5

end stock_price_end_of_second_year_l379_379434


namespace find_s_l379_379024

theorem find_s (s : ℝ) :
  (∃ (s : ℝ), (s, 7) lies_on the_line_through (0, 2) and (-10, 0)) → s = 25 :=
by
  sorry

end find_s_l379_379024


namespace area_of_triangle_l379_379441

noncomputable def triangle_area_by_medians (a b : ℝ) (cos_theta : ℝ) : ℝ := 
  ((2/3 * a) * (1/3 * b) * real.sqrt (1 - (cos_theta)^2))

theorem area_of_triangle 
  (a b : ℝ)
  (cos_theta : ℝ)
  (h_a : a = 3)
  (h_b : b = 2 * real.sqrt 7)
  (h_cos_theta : cos_theta = -3/4)
  :
  6 * triangle_area_by_medians a b cos_theta = 7 := 
sorry

end area_of_triangle_l379_379441


namespace part1_solution_part2_solution_l379_379077

-- Definitions based on the given conditions
def eq1 (x y m : ℝ) := 2 * x - y = m
def eq2 (x y m : ℝ) := 3 * x + 2 * y = m + 7

-- Problem Part 1: When m = 0, the solution to the system of equations
theorem part1_solution :
  ∃ (x y : ℝ), eq1 x y 0 ∧ eq2 x y 0 ∧ x = 1 ∧ y = 2 :=
by
  existsi 1
  existsi 2
  apply And.intro
  show eq1 1 2 0, by sorry
  apply And.intro
  show eq2 1 2 0, by sorry
  apply And.intro
  show 1 = 1, by rfl
  show 2 = 2, by rfl

-- Problem Part 2: When A(-2, 3), the value of m that satisfies the equations
theorem part2_solution :
  let x := -2
  let y := 3
  ∃ (m : ℝ), eq1 x y m ∧ m = -7 :=
by
  existsi (-7 : ℝ)
  apply And.intro
  show eq1 (-2) 3 (-7), by sorry
  show -7 = -7, by rfl

end part1_solution_part2_solution_l379_379077


namespace range_of_a_l379_379475

noncomputable def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f(x) = f(-x)
noncomputable def is_monotonically_increasing_on (f : ℝ → ℝ) (S : set ℝ) := ∀ x y ∈ S, x < y → f(x) ≤ f(y)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h1 : is_even f) (h2 : is_monotonically_increasing_on f (set.Iic 0))
  (h3 : f (|a - 1|) > f (-1/2)) : a ∈ set.Ioo (1/2) (3/2) :=
sorry

end range_of_a_l379_379475


namespace factor_x4_plus_16_l379_379586

theorem factor_x4_plus_16 : ∃ p q : ℝ[X], (p = (X^2 - 4*X + 4)) ∧ (q = (X^2 + 4*X + 4)) ∧ (p * q = X^4 + 16) := 
by {
  use [(X^2 - 4*X + 4), (X^2 + 4*X + 4)],
  split,
  { refl },
  split,
  { refl },
  sorry
}

end factor_x4_plus_16_l379_379586


namespace absolute_value_of_difference_of_roots_is_correct_l379_379924

noncomputable def absolute_value_of_difference_of_roots (k : ℝ) : ℝ :=
  let r1 := (k+3 + real.sqrt ((k+3)^2 - 4*k)) / 2
  let r2 := (k+3 - real.sqrt ((k+3)^2 - 4*k)) / 2
  abs (r1 - r2)

theorem absolute_value_of_difference_of_roots_is_correct (k : ℝ) :
  absolute_value_of_difference_of_roots k = real.sqrt (k^2 + 2*k + 9) :=
by
  sorry

end absolute_value_of_difference_of_roots_is_correct_l379_379924


namespace correct_option_D_l379_379658

theorem correct_option_D (a : ℝ) : (-a^3)^2 = a^6 :=
sorry

end correct_option_D_l379_379658


namespace dot_product_expression_l379_379559

variable {V : Type*} [inner_product_space ℝ V]

variables (a b c d : V)
variable (k : ℝ)

/- Given conditions -/
axiom ha : inner a b = 2
axiom hb : inner a c = -1
axiom hc : inner b c = 4
axiom hd : inner a d = 3

theorem dot_product_expression :
  inner b (5 • c - 3 • d + 4 • a) = 28 - 3 * inner b d :=
by
  /- Proof to be filled -/
  sorry

end dot_product_expression_l379_379559


namespace particle_velocity_and_inflection_l379_379371

theorem particle_velocity_and_inflection 
  (x : ℝ → ℝ) (y : ℝ → ℝ)
  (h1 : x = λ t, t^3 - t) 
  (h2 : y = λ t, t^4 + t) :
  (∃ (t0 : ℝ), t0 = 0 ∧ (∀ t ≠ 0, 
    (let v := sqrt ((3 * t^2 - 1)^2 + (4 * t^3 + 1)^2) in v < sqrt 2)))
  ∧ (let g t := (4 * t^3 + 1) / (3 * t^2 - 1) in
    ∃ (t0 : ℝ), t0 = 0 ∧ (∀ t ≠ 0, 
    (let dg := diff (λ t, (12 * t^4 - 12 * t^2 - 6 * t) / ((3 * t^2 - 1)^2)) in
     dg < 0) ∧ (let f := g'' 0 in f = 0))) := sorry

end particle_velocity_and_inflection_l379_379371


namespace relationship_abc_l379_379042

noncomputable def a : ℝ := 0.9 ^ 1.1
noncomputable def b : ℝ := 0.9 ^ 1.09
noncomputable def c : ℝ := Real.logBase (1/3) 2

theorem relationship_abc : c < a ∧ a < b := by
  sorry

end relationship_abc_l379_379042


namespace solve_for_x_l379_379208

theorem solve_for_x : ∃ x : ℝ, 3^x * 9^x = 27^(x - 4) := 
by
  use -6
  sorry

end solve_for_x_l379_379208


namespace correct_option_c_l379_379664

theorem correct_option_c (x : ℝ) : -2 * (x + 1) = -2 * x - 2 :=
  by
  -- Proof can be omitted
  sorry

end correct_option_c_l379_379664


namespace polygon_sides_l379_379700

theorem polygon_sides (side_length perimeter : ℕ) (h1 : side_length = 4) (h2 : perimeter = 24) : 
  perimeter / side_length = 6 :=
by 
  sorry

end polygon_sides_l379_379700


namespace rachel_milk_correct_l379_379008

-- Define the initial amount of milk Don has
def don_milk : ℚ := 1 / 5

-- Define the fraction of milk Rachel drinks
def rachel_drinks_fraction : ℚ := 2 / 3

-- Define the total amount of milk Rachel drinks
def rachel_milk : ℚ := rachel_drinks_fraction * don_milk

-- The goal is to prove that Rachel drinks a specific amount of milk
theorem rachel_milk_correct : rachel_milk = 2 / 15 :=
by
  -- The proof would be here
  sorry

end rachel_milk_correct_l379_379008


namespace no_happy_family_possible_disjoint_happy_families_possible_l379_379187

-- Definitions based on conditions
variables (n k : ℕ)
variables (pop : Type)
variables (lovers : set (pop × pop))

-- Happy family definition
def happy_family (x y z : pop) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (lovers (x, y) ∧ lovers (y, x)) ∧ 
  (lovers (x, z) ∧ lovers (z, x)) ∧ (lovers (y, z) ∧ lovers (z, y))

-- Conditions in Lean
axioms (M W A : fin n → pop)
axiom loves_reciprocal : ∀ a b : pop, lovers (a, b) → lovers (b, a)
axiom loves_count : ∀ x : pop, ∃ S ⊆ finset.univ (fintype.of_fin n), S.card ≥ k ∧ ∀ y ∈ S, lovers (x, y)

-- Proof problem 1
theorem no_happy_family_possible (h1 : n = 2 * k) : 
  ∃ (M W A : fin n → pop), (∀ sf, sf = happy_family (M sf) (W sf) (A sf) → false) := 
sorry

-- Proof problem 2
theorem disjoint_happy_families_possible (h2 : 4 * k ≥ 3 * n) : 
  ∃ (fam : fin n → pop × pop × pop), (∀ i, ∃ (x ∈ W) (y ∈ M) (z ∈ A), fam i = (x, y, z) ∧ happy_family x y z) := 
sorry

end no_happy_family_possible_disjoint_happy_families_possible_l379_379187


namespace geometric_sequence_sum_l379_379121

theorem geometric_sequence_sum (a_1 q n S : ℕ) (h1 : a_1 = 2) (h2 : q = 2) (h3 : S = 126) 
    (h4 : S = (a_1 * (1 - q^n)) / (1 - q)) : 
    n = 6 :=
by
  sorry

end geometric_sequence_sum_l379_379121


namespace gcd_n_cube_plus_m_square_l379_379450

theorem gcd_n_cube_plus_m_square (n m : ℤ) (h : n > 2^3) : Int.gcd (n^3 + m^2) (n + 2) = 1 :=
by
  sorry

end gcd_n_cube_plus_m_square_l379_379450


namespace next_to_Denis_l379_379277

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379277


namespace correct_option_c_l379_379663

theorem correct_option_c (x : ℝ) : -2 * (x + 1) = -2 * x - 2 :=
  by
  -- Proof can be omitted
  sorry

end correct_option_c_l379_379663


namespace triangle_inequality_l379_379715

theorem triangle_inequality (a b c : ℕ) : 
    a + b > c ∧ a + c > b ∧ b + c > a ↔ 
    (a, b, c) = (2, 3, 4) ∨ (a, b, c) = (3, 4, 7) ∨ (a, b, c) = (4, 6, 2) ∨ (a, b, c) = (7, 10, 2)
    → (a + b > c ∧ a + c > b ∧ b + c > a ↔ (a, b, c) = (2, 3, 4)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a         ↔ (a, b, c) = (3, 4, 7)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a        ↔ (a, b, c) = (4, 6, 2)) ∧
      (a + b < c ∨ a + c < b ∨ b + c < a        ↔ (a, b, c) = (7, 10, 2)) :=
sorry

end triangle_inequality_l379_379715


namespace integral_evaluation_l379_379429

noncomputable def integral_value : ℝ :=
  ∫ x in 0..(Real.pi / 2), 3 * x - Real.sin x

theorem integral_evaluation :
  integral_value = (3 * Real.pi^2) / 8 - 1 :=
by
  sorry

end integral_evaluation_l379_379429


namespace total_installments_is_52_l379_379688

def total_payments (installs_first25 installs_remain n : ℕ) (avg_payment : ℚ) :=
  (installs_first25 * 500 + installs_remain * 600) / n = avg_payment

theorem total_installments_is_52
  (avg_payment : ℚ)
  (h_avg_payment : avg_payment = 57500 / 104)
  (h_first25 : ∀ n, installs_first25 = 25) :
  ∃ n, total_payments 25 (n - 25) n avg_payment ∧ n = 52 :=
by sorry

end total_installments_is_52_l379_379688


namespace percentage_increase_l379_379846

theorem percentage_increase (L : ℕ) (h1 : L + 450 = 1350) :
  (450 / L : ℚ) * 100 = 50 := by
  sorry

end percentage_increase_l379_379846


namespace sum_of_c_l379_379793

noncomputable def a (n : ℕ) : ℝ := (3 / 2)^(n - 1)
noncomputable def b : ℕ → ℝ
| 0     := -1
| (n+1) := b n + n

noncomputable def c (n : ℕ) : ℝ := (2 * a n * b n) / (n + 1)
noncomputable def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, c (k + 1))

theorem sum_of_c (n : ℕ) : 
  T n = 8 + 3 * (n - 4) * (3 / 2)^(n - 1) := sorry

end sum_of_c_l379_379793


namespace toms_sixth_time_l379_379321

theorem toms_sixth_time (t : ℕ) (h : Multiset.median ({92, 86, 101, 95, 90, t} : Multiset ℕ) = 93) : t = 94 :=
by
-- Proof goes here
sorry

end toms_sixth_time_l379_379321


namespace parallel_ST_BD_l379_379155

noncomputable theory

variables {P Q R S T U V W X Y Z : Type*}
variables (AB AD CD CB BC DA : ℝ)
variables (X Y E F G H S T : P)

-- Conditions
hypothesis cyclicQuadrilateral : ∀ {A B C D : P}, concyclic A B C D12
hypothesis ratio_eq1 : AB / AD = CD / CB
hypothesis midpoint_E : is_midpoint E AB
hypothesis midpoint_F : is_midpoint F BC
hypothesis midpoint_G : is_midpoint G CD
hypothesis midpoint_H : is_midpoint H DA
hypothesis angle_bisector_S : S ∈ angle_bisector X B A
hypothesis angle_bisector_T : T ∈ angle_bisector Y A D

theorem parallel_ST_BD : parallel ST BD :=
sorry

end parallel_ST_BD_l379_379155


namespace denis_neighbors_l379_379300

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

end denis_neighbors_l379_379300


namespace roots_of_z4_plus_16_eq_0_l379_379765

noncomputable def roots_of_quartic_eq : Set ℂ :=
  { z | z^4 + 16 = 0 }

theorem roots_of_z4_plus_16_eq_0 :
  roots_of_quartic_eq = { z | z = complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 - complex.I * complex.sqrt 2 ∨
                             z = complex.sqrt 2 - complex.I * complex.sqrt 2 } :=
by
  sorry

end roots_of_z4_plus_16_eq_0_l379_379765


namespace circle_region_between_lines_l379_379638

noncomputable def circleRegionArea (a : ℝ) (h : a > 0) : ℝ :=
  let r1 := π * a^2 in
  r1

theorem circle_region_between_lines (a : ℝ) (h : a > 0) :
  let c1 := x^2 + y^2 = a^2
  let c2 := x^2 + y^2 = 9*a^2
  let l1 := y = a * sqrt 3
  let l2 := y = -a * sqrt 3
  let l3 := y = 3*a * sqrt 3
  let l4 := y = -3*a * sqrt 3
  circleRegionArea a h = π * a^2 :=
by
  sorry

end circle_region_between_lines_l379_379638


namespace sum_even_integers_102_to_200_l379_379969

theorem sum_even_integers_102_to_200 : 
  (finset.sum (finset.filter (λ n, n % 2 = 0) (finset.range' 102 200.succ))) = 7550 :=
sorry

end sum_even_integers_102_to_200_l379_379969


namespace problem1_problem2_l379_379210

-- Define the concept of a regular hexagon, segments, and coloring

-- Main assumption structures
structure RegularHexagon where
  vertices : Fin 6 → Fin 6
  edges : Fin 15 → (Fin 6 × Fin 6)

inductive Color
| Red
| Blue

-- Problem 1: If 15 segments are colored, then at least two monochromatic triangles exist
theorem problem1 (hex : RegularHexagon) (coloring : Fin 15 → Color) :
  (∃ t1 t2 : Fin 6 × Fin 6 × Fin 6, t1 ≠ t2 ∧ 
    (coloring (index t1.fst, t1.snd) = coloring (index t1.snd, t1.thd) 
    ∧ coloring (index t1.snd, t1.thd) = coloring (index t1.fst, t1.thd)) ∧
    (coloring (index t2.fst, t2.snd) = coloring (index t2.snd, t2.thd) 
    ∧ coloring (index t2.snd, t2.thd) = coloring (index t2.fst, t2.thd))) :=
sorry

-- Problem 2: If 14 segments are colored, then no monochromatic triangle necessarily exists
theorem problem2 (hex : RegularHexagon) (coloring : Fin 14 → Color) :
  ¬ (∃ t : Fin 6 × Fin 6 × Fin 6,
    (coloring (index t.fst, t.snd) = coloring (index t.snd, t.thd) 
    ∧ coloring (index t.snd, t.thd) = coloring (index t.fst, t.thd))) :=
sorry

end problem1_problem2_l379_379210


namespace moles_of_NaCl_formed_l379_379001

theorem moles_of_NaCl_formed
  (moles_NaOH : ℕ)
  (moles_HCl : ℕ)
  (balanced_reaction : ∀ (n : ℕ), (n : ℕ) * 1 = (n : ℕ) * 1 + (n : ℕ) * 1 - (n : ℕ) * 1)
  : moles_NaOH = 4 → moles_HCl = 3 → 3 = 3 :=
by
  intro h1 h2
  rw [h1, h2]
  exact rfl

end moles_of_NaCl_formed_l379_379001


namespace smallest_positive_period_max_min_value_l379_379810

def f (x : ℝ) : ℝ := sin x ^ 2 + sqrt 3 * sin x * cos x

theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem max_min_value (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) :
  0 ≤ f x ∧ f x ≤ 3 / 2 :=
by sorry

end smallest_positive_period_max_min_value_l379_379810


namespace orthocenter_of_triangle_BDF_l379_379686

noncomputable def P_interior (ABC : Triangle) (P : Point) : Prop := 
  is_interior_of_triangle ABC P

noncomputable def Q_interior (ABC : Triangle) (Q : Point) : Prop := 
  is_interior_of_triangle ABC Q

noncomputable def feet_of_perpendiculars 
  (P : Point) 
  (lines : { l1 l2 l3 : Line // Line.orthogonal P l1 ∧ Line.orthogonal P l2 ∧ Line.orthogonal P l3 }) 
  : Point × Point × Point := sorry

theorem orthocenter_of_triangle_BDF 
  {ABC : Triangle} (P Q : Point) 
  (hP_interior : P_interior ABC P) 
  (hQ_interior : Q_interior ABC Q)
  (h1 : ∠ABC.C = ∠Other_C ABC B Q)
  (h2 : ∠ABC.A = ∠Other_A ABC B Q)
  (hDEF_perpendiculars : feet_of_perpendiculars P ABC.lines = (ABC.lines.B D, ABC.lines.A D, ABC.lines.C D))
  (hDEF_90 : ∠DEF = 90) : 
  is_orthocenter Q (triangle BDF) := sorry

end orthocenter_of_triangle_BDF_l379_379686


namespace solution_set_of_inequality_l379_379448

theorem solution_set_of_inequality:
  {x : ℝ | |x - 5| + |x + 1| < 8} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end solution_set_of_inequality_l379_379448


namespace find_n_l379_379070

-- Defining the expansion condition for the given polynomial
def expansion_condition (n : ℕ) : Prop :=
  (n.naiveBinom 0 + (1 / 4) * n.naiveBinom 2 = 2 * (1 / 2) * n.naiveBinom 1)

def max_coefficient_term (r : ℕ) : Prop :=
  (1 / 2^r * 8.naiveBinom r ≥ 1 / 2^(r + 1) * 8.naiveBinom (r + 1)) ∧
  (1 / 2^r * 8.naiveBinom r ≥ 1 / 2^(r - 1) * 8.naiveBinom (r - 1))

theorem find_n (n : ℕ) (r : ℕ) (T3 T4 : ℕ → ℝ) : (expansion_condition n) → n = 8 ∧ T3 3 = 7 * x^5 ∧ T4 4 = 7 * x^(9 / 2) :=
by
  sorry

end find_n_l379_379070


namespace tiered_water_pricing_usage_l379_379992

theorem tiered_water_pricing_usage (total_cost : ℤ) (water_used : ℤ) :
  (total_cost = 60) →
  (water_used > 12 ∧ water_used ≤ 18) →
  (3 * 12 + (water_used - 12) * 6 = total_cost) →
  water_used = 16 :=
by
  intros h_cost h_range h_eq
  sorry

end tiered_water_pricing_usage_l379_379992


namespace average_salary_of_factory_personnel_l379_379854

theorem average_salary_of_factory_personnel :
  let supervisors := 6 in
  let laborers := 42 in
  let avg_supervisor_salary := 2450 in
  let avg_laborer_salary := 950 in
  let total_supervisors_salary := supervisors * avg_supervisor_salary in
  let total_laborers_salary := laborers * avg_laborer_salary in
  let total_salary := total_supervisors_salary + total_laborers_salary in
  let total_personnel := supervisors + laborers in
  total_salary / total_personnel = 1137.50 :=
by sorry

end average_salary_of_factory_personnel_l379_379854


namespace min_cells_to_prevent_shape_placement_l379_379584

theorem min_cells_to_prevent_shape_placement (n : ℕ) (h₁ : n = 12) :
  ∃ (m : ℕ), m = 72 ∧ (∀ shape_placement, ¬ can_fit_unshaded (12 * 12) (12 * 12 - 72)) := 
sorry

end min_cells_to_prevent_shape_placement_l379_379584


namespace arithmetic_sequence_terms_before_neg3_l379_379502

theorem arithmetic_sequence_terms_before_neg3 :
  ∀ (n : ℕ), (a₁ : ℤ) → dare : ℤ → 
  n > 0 → 
  (a₁ = 105) → 
  (dare = -6) → 
  (aₙ = a₁ + (n - 1) * dare) → 
  aₙ = -3 → n = 19 ∧ sorry

end arithmetic_sequence_terms_before_neg3_l379_379502


namespace count_remarkable_numbers_l379_379177

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_four_divisors (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n = a * b * c * d ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 
  (1 ≤ a) ∧ (a < b) ∧ (b < c) ∧ (c < d) ∧
  (∀ x : ℕ, x ∣ n ↔ x = 1 ∨ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
  ∃ p1 p2, is_prime p1 ∧ is_prime p2 ∧ n = p1 * p2

def is_remarkable (n : ℕ) : Prop :=
  has_four_divisors n ∧ n > 9 ∧ n < 100

-- Main goal
theorem count_remarkable_numbers : 
  (finset.filter is_remarkable (finset.range 100)).card = 36 :=
by
  sorry

end count_remarkable_numbers_l379_379177


namespace lena_nicole_candy_difference_l379_379153

variables (L K N : ℕ)

theorem lena_nicole_candy_difference
  (hL : L = 16)
  (hLK : L + 5 = 3 * K)
  (hKN : K = N - 4) :
  L - N = 5 :=
sorry

end lena_nicole_candy_difference_l379_379153


namespace find_range_of_x0_l379_379489

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 1 else Real.logb 2 (x + 1)

theorem find_range_of_x0 (x0 : ℝ) : f x0 < 1 ↔ x0 < 1 := by
  sorry

end find_range_of_x0_l379_379489


namespace max_value_of_f_range_of_m_l379_379811

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem max_value_of_f (a b : ℝ) (x : ℝ) (h1 : 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_tangent : ∀ (x : ℝ), f a b x - ((-1/2) * x + (Real.log 1 - 1/2)) = 0) : 
  ∃ x_max, f a b x_max = -1/2 := sorry

theorem range_of_m (m : ℝ) 
  (h_ineq : ∀ (a : ℝ) (x : ℝ), 1 ≤ a ∧ a ≤ 3 / 2 ∧ 1 ≤ x ∧ x ≤ Real.exp 2 → a * Real.log x ≥ m + x) : 
  m ≤ 2 - Real.exp 2 := sorry

end max_value_of_f_range_of_m_l379_379811


namespace sum_even_integers_102_to_200_l379_379973

theorem sum_even_integers_102_to_200 : 
  let sequence := list.range' 102 100 
  ∧ (∀ x ∈ sequence, x % 2 = 0) →
  list.sum sequence = 7550 := 
by 
  let sequence := list.range' 102 100 
  have even_sequence : ∀ x ∈ sequence, x % 2 = 0 := 
    sorry 
  have sum_sequence : list.sum sequence = 7550 := 
    sorry 
  exact sum_sequence 

end sum_even_integers_102_to_200_l379_379973


namespace set_union_is_correct_l379_379092

noncomputable def M (a : ℝ) : Set ℝ := {3, 2^a}
noncomputable def N (a b : ℝ) : Set ℝ := {a, b}

variable (a b : ℝ)
variable (h₁ : M a ∩ N a b = {2})
variable (h₂ : ∃ a b, N a b = {1, 2} ∧ M a = {3, 2} ∧ M a ∪ N a b = {1, 2, 3})

theorem set_union_is_correct :
  M 1 ∪ N 1 2 = {1, 2, 3} :=
by
  sorry

end set_union_is_correct_l379_379092


namespace find_year_l379_379242

def price_P (n : ℕ) : ℝ := 4.2 + 0.4 * n
def price_Q (n : ℕ) : ℝ := 6.3 + 0.15 * n

theorem find_year :
  ∃ n : ℕ, price_P n = price_Q n + 0.4 :=
by {
  use 10,
  simp [price_P, price_Q],
  norm_num,
}

end find_year_l379_379242


namespace sequence_contains_infinite_powers_of_two_l379_379197

def a_n (n : ℕ) : ℕ := ⌊n * Real.sqrt 2⌋

theorem sequence_contains_infinite_powers_of_two :
  ∃ (k : ℕ) (infinitely_many_n : ℕ → Prop), (∀ k > 0, infinitely_many_n (λ n, a_n n = 2^k)) :=
begin
  sorry
end

end sequence_contains_infinite_powers_of_two_l379_379197


namespace problem_incorrect_statement_D_l379_379670

theorem problem_incorrect_statement_D :
  (∀ x y, x = -y → x + y = 0) ∧
  (∃ x : ℕ, x^2 + 2 * x = 0) ∧
  (∀ x y : ℝ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (¬ (∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ (x + y > 2))) :=
by sorry

end problem_incorrect_statement_D_l379_379670


namespace simplify_expression_l379_379923

open Real

theorem simplify_expression (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3 * a) (h3 : b ≠ a) (h4 : b ≠ -a) : 
  ((2 * b + a - (4 * a ^ 2 - b ^ 2) / a) / (b ^ 3 + 2 * a * b ^ 2 - 3 * a ^ 2 * b)) *
  ((a ^ 3 * b - 2 * a ^ 2 * b ^ 2 + a * b ^ 3) / (a ^ 2 - b ^ 2)) = 
  (a - b) / (a + b) :=
by
  sorry

end simplify_expression_l379_379923


namespace max_triangles_no_tetrahedron_l379_379047

theorem max_triangles_no_tetrahedron (points : Finset Point) 
  (h_points : points.card = 9)
  (h_no_four_coplanar : ∀ (a b c d : Point), a ∈ points → b ∈ points → c ∈ points → d ∈ points → ¬ Plane a b c d)
  (h_no_tetrahedron : ∀ (a b c d : Point), a ∈ points → b ∈ points → c ∈ points → d ∈ points → ¬ Tetrahedron a b c d) :
  ∃(triangles: Finset (Finset Point)), triangles.card = 27 := 
sorry

end max_triangles_no_tetrahedron_l379_379047


namespace prove_inequality_l379_379573

noncomputable def satisfying_condition (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c = real.sqrt 7 a + real.sqrt 7 b + real.sqrt 7 c)

theorem prove_inequality (a b c : ℝ) (h : satisfying_condition a b c) : 
 a^a * b^b * c^c ≥ 1 := sorry

end prove_inequality_l379_379573


namespace find_c_plus_inv_b_l379_379929

variable (a b c : ℝ)

def conditions := 
  (a * b * c = 1) ∧ 
  (a + 1/c = 7) ∧ 
  (b + 1/a = 16)

theorem find_c_plus_inv_b (h : conditions a b c) : 
  c + 1/b = 25 / 111 :=
sorry

end find_c_plus_inv_b_l379_379929


namespace solve_z_pow_eq_neg_sixteen_l379_379760

theorem solve_z_pow_eq_neg_sixteen (z : ℂ) :
  z^4 = -16 ↔ 
  z = complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) + complex.I * complex.sqrt(2) ∨ 
  z = -complex.sqrt(2) - complex.I * complex.sqrt(2) ∨ 
  z = complex.sqrt(2) - complex.I * complex.sqrt(2) :=
by
  sorry

end solve_z_pow_eq_neg_sixteen_l379_379760


namespace min_val_exp_l379_379807

theorem min_val_exp (x y : ℝ) (h : x + 2 * y = 6) :
  2 ^ x + 4 ^ y ≥ 16 :=
by
  sorry

end min_val_exp_l379_379807


namespace denis_neighbors_l379_379297

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

end denis_neighbors_l379_379297


namespace sum_even_integers_102_to_200_l379_379971

theorem sum_even_integers_102_to_200 : 
  (finset.sum (finset.filter (λ n, n % 2 = 0) (finset.range' 102 200.succ))) = 7550 :=
sorry

end sum_even_integers_102_to_200_l379_379971


namespace distinct_positive_roots_log_sum_eq_5_l379_379774

theorem distinct_positive_roots_log_sum_eq_5 (a b : ℝ)
  (h : ∀ (x : ℝ), (8 * x ^ 3 + 6 * a * x ^ 2 + 3 * b * x + a = 0) → x > 0) 
  (h_sum : ∀ u v w : ℝ, (8 * u ^ 3 + 6 * a * u ^ 2 + 3 * b * u + a = 0) ∧
                       (8 * v ^ 3 + 6 * a * v ^ 2 + 3 * b * v + a = 0) ∧
                       (8 * w ^ 3 + 6 * a * w ^ 2 + 3 * b * w + a = 0) → 
                       u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ 
                       (Real.log (u) / Real.log (3) + Real.log (v) / Real.log (3) + Real.log (w) / Real.log (3) = 5)) :
  a = -1944 :=
sorry

end distinct_positive_roots_log_sum_eq_5_l379_379774


namespace circle_equation_range_of_k_l379_379839

theorem circle_equation_range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 4*k*x - 2*y + 5*k = 0) ↔ (k > 1 ∨ k < 1/4) :=
by
  sorry

end circle_equation_range_of_k_l379_379839


namespace imaginary_part_of_product_l379_379951

noncomputable def z : ℂ := 1 - 2 * Complex.i
noncomputable def w : ℂ := Complex.i

theorem imaginary_part_of_product : Complex.im (z * w) = 1 := by
  have h1 : z = 1 - 2 * Complex.i := by rfl
  have h2 : w = Complex.i := by rfl
  have h3 : z * w = (1 - 2 * Complex.i) * Complex.i := by rw [h1, h2]
  sorry

end imaginary_part_of_product_l379_379951


namespace semicircle_to_cone_distance_l379_379707

def semicircle_to_cone_height (R : ℝ) (r : ℝ) (h : ℝ) (H : ℝ) : Prop :=
  2 * Real.pi * r = Real.pi * R ∧
  h = sqrt (R^2 - r^2) ∧
  H = 2 * (h * r / R)

theorem semicircle_to_cone_distance 
  (R : ℝ) (r : ℝ) (h : ℝ) (H : ℝ) 
  (h1 : R = 4) 
  (h2 : semicircle_to_cone_height R r h H) :
  H = 2 * sqrt 3 := 
sorry

end semicircle_to_cone_distance_l379_379707


namespace set_of_x_when_f_maximum_maximum_area_of_triangle_l379_379044

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 2 * cos x ^ 2 + 1

theorem set_of_x_when_f_maximum :
  {x | ∃ k : ℤ, x = k * π + π / 3} :=
by sorry

theorem maximum_area_of_triangle 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : f C = 2) 
  (h2 : c = sqrt 3)
  (h3 : 0 < C ∧ C < π) :
  ∃ (ab_area : ℝ), ab_area = 3 * sqrt 3 / 4 :=
by sorry

end set_of_x_when_f_maximum_maximum_area_of_triangle_l379_379044


namespace sin_minus_cos_value_l379_379065

theorem sin_minus_cos_value
  (α : ℝ)
  (h1 : Real.tan α = (Real.sqrt 3) / 3)
  (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α - Real.cos α = -1/2 + Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_value_l379_379065


namespace calculate_R_l379_379695

variable (P : ℝ) (rC : ℝ) (T : ℝ) (gainB : ℝ) (I : ℝ)

-- Let P = 3200 (Principal loaned by A to B and by B to C)
def P := 3200
-- Let rC = 14.5% (Rate at which B lends money to C)
def rC := 14.5
-- Let T = 5 (Time period in years)
def T := 5
-- Let gainB = 400 (Gain of B after 5 years)
def gainB := 400
-- Interest B earns from C
def I := P * rC / 100 * T

theorem calculate_R : 
    let P := 3200
    let rC := 14.5
    let T := 5
    let gainB := 400
    let I := 3200 * 14.5 / 100 * 5
    ∃ R : ℝ, (P * R / 100 * T = I - gainB) → (R = 12) :=
by
    sorry

end calculate_R_l379_379695


namespace Bruce_riding_time_l379_379548

-- Define the conditions and what we need to prove
theorem Bruce_riding_time (b s : ℝ) :
  let d := 90 * b in
  45 * (b + s) = d →
  s = b →
  d / s = 90 := by
    intros _ h₁ h₂
    sorry  -- Proof not required, only statement

end Bruce_riding_time_l379_379548


namespace tangent_parabola_locus_l379_379736

theorem tangent_parabola_locus (u v p : ℝ) :
  let P := (u, v)
  in is_tangent P (λ p, y^2 - 2 * p * x = 0) 45 =
  ((u + 3 / 2 * p)^2 - v^2 = 2 * p^2) := 
sorry

end tangent_parabola_locus_l379_379736


namespace first_platform_length_l379_379711

noncomputable def length_of_first_platform (t1 t2 l_train l_plat2 time1 time2 : ℕ) : ℕ :=
  let s1 := (l_train + t1) / time1
  let s2 := (l_train + l_plat2) / time2
  if s1 = s2 then t1 else 0

theorem first_platform_length:
  ∀ (time1 time2 : ℕ) (l_train l_plat2 : ℕ), time1 = 15 → time2 = 20 → l_train = 350 → l_plat2 = 250 → length_of_first_platform 100 l_plat2 l_train l_plat2 time1 time2 = 100 :=
by
  intros time1 time2 l_train l_plat2 ht1 ht2 ht3 ht4
  rw [ht1, ht2, ht3, ht4]
  dsimp [length_of_first_platform]
  rfl

end first_platform_length_l379_379711


namespace lena_more_candy_bars_than_nicole_l379_379150

theorem lena_more_candy_bars_than_nicole
  (Lena Kevin Nicole : ℕ)
  (h1 : Lena = 16)
  (h2 : Lena + 5 = 3 * Kevin)
  (h3 : Kevin + 4 = Nicole) :
  Lena - Nicole = 5 :=
by
  sorry

end lena_more_candy_bars_than_nicole_l379_379150


namespace number_of_valid_cube_positions_l379_379699

theorem number_of_valid_cube_positions : 
  let initial_shape : Set (Set ℝ²) := { [(0,0), (1,0), (2,0), (1,-1), (1,1)] } in 
  let possible_positions : Finset (ℝ²) := { (0,1), (1,2), (2,1), (3,0), (2,-1), (1,-2), (0,-1), (-1,0), (1,-1), (1,1) } in
  let valid_positions := possible_positions.filter (λ p, can_fold_with_cube_missing_face (insert p initial_shape)) in
  valid_positions.card = 5 :=
by
  sorry

-- Stub functions just to illustrate structure. Detailed implementations would be needed.
noncomputable def can_fold_with_cube_missing_face (added_square_shape : Set ℝ²) : Prop :=
  -- Logic to determine if adding a square allows folding into a cube with one face missing 
  sorry

end number_of_valid_cube_positions_l379_379699


namespace trig_simplify_l379_379600

theorem trig_simplify (α : ℝ) :
  (sin (2 * real.pi - α) * cos (3 * real.pi + α) * cos ((3 / 2) * real.pi - α)) /
  (sin (-real.pi + α) * sin (3 * real.pi - α) * cos (-α - real.pi)) = -1 :=
by
  sorry

end trig_simplify_l379_379600


namespace sum_first_four_terms_of_geometric_sequence_l379_379052

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), a i

theorem sum_first_four_terms_of_geometric_sequence :
  ∀ (a : ℕ → ℝ),
    geometric_sequence a →
    a 0 = 1 →
    (1 / a 0) - (1 / a 1) = 2 / a 2 →
    sum_of_first_n_terms a 3 = 15 :=
by
  sorry

end sum_first_four_terms_of_geometric_sequence_l379_379052


namespace two_triangles_with_equal_angles_and_sides_not_congruent_l379_379412

theorem two_triangles_with_equal_angles_and_sides_not_congruent :
  ∃ (ΔABC ΔDEF : Triangle), 
    (∀ i, angle ΔABC i = angle ΔDEF i) ∧ 
    (side ΔABC 1 = side ΔDEF 1) ∧ (side ΔABC 2 = side ΔDEF 2) ∧
    ¬ congruent ΔABC ΔDEF :=
by
  sorry

end two_triangles_with_equal_angles_and_sides_not_congruent_l379_379412


namespace regular_tetrahedron_l379_379542

noncomputable def is_regular_tetrahedron (A1 A2 A3 A4 : ℝ × ℝ × ℝ) : Prop :=
  let d := dist A1 A2 in
  dist A1 A2 = d ∧ dist A1 A3 = d ∧ dist A1 A4 = d ∧ dist A2 A3 = d ∧ dist A2 A4 = d ∧ dist A3 A4 = d

theorem regular_tetrahedron
  (A1 A2 A3 A4 O : ℝ × ℝ × ℝ)
  (r R : ℝ)
  (radii : ℝ × ℝ × ℝ × ℝ)
  (h1 : dist A1 A2 = radii.1 + radii.2)
  (h2 : dist A1 A3 = radii.1 + radii.3)
  (h3 : dist A1 A4 = radii.1 + radii.4)
  (h4 : dist A2 A3 = radii.2 + radii.3)
  (h5 : dist A2 A4 = radii.2 + radii.4)
  (h6 : dist A3 A4 = radii.3 + radii.4)
  (ho1 : dist O A1 = r + radii.1)
  (ho2 : dist O A2 = r + radii.2)
  (ho3 : dist O A3 = r + radii.3)
  (ho4 : dist O A4 = r + radii.4)
  (he1 : dist O ((A1 + A2) / 2) = R)
  (he2 : dist O ((A1 + A3) / 2) = R)
  (he3 : dist O ((A1 + A4) / 2) = R)
  (he4 : dist O ((A2 + A3) / 2) = R)
  (he5 : dist O ((A2 + A4) / 2) = R)
  (he6 : dist O ((A3 + A4) / 2) = R)
  : is_regular_tetrahedron A1 A2 A3 A4 :=
sorry

end regular_tetrahedron_l379_379542


namespace f_value_at_pi_over_8_f_smallest_positive_period_f_monotonic_decrease_intervals_l379_379488

def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 - Real.cos (2 * x + Real.pi / 2)

theorem f_value_at_pi_over_8 :
  f (Real.pi / 8) = Real.sqrt 2 + 1 := 
sorry

theorem f_smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := 
sorry

theorem f_monotonic_decrease_intervals (k : ℤ) :
  set.Ioc (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8) ⊆ {x | ∃ y, f y = x} :=
sorry

end f_value_at_pi_over_8_f_smallest_positive_period_f_monotonic_decrease_intervals_l379_379488


namespace schoolchildren_lineup_l379_379306

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379306


namespace range_of_third_side_of_acute_triangle_l379_379482

theorem range_of_third_side_of_acute_triangle:
  ∀ (a : ℝ), (∀ (A B C : ℝ), 
    A = 1 ∧ B = 3 ∧ C = a ∧
    (A + B > C ∧ A + C > B ∧ B + C > A) ∧ -- Triangle inequality
    (cos (angle_sum (atan (A / √(B^2 + C^2 - A^2))) (atan (B / √(A^2 + C^2 - B^2)))) > 0)) → -- Acute angle condition  
    2 * sqrt 2 < a ∧ a < sqrt 10 :=
sorry

end range_of_third_side_of_acute_triangle_l379_379482


namespace mrs_hilt_rows_of_pies_l379_379581

def number_of_pies (pecan_pies: Nat) (apple_pies: Nat) : Nat := pecan_pies + apple_pies

def rows_of_pies (total_pies: Nat) (pies_per_row: Nat) : Nat := total_pies / pies_per_row

theorem mrs_hilt_rows_of_pies :
  let pecan_pies := 16 in
  let apple_pies := 14 in
  let pies_per_row := 5 in
  rows_of_pies (number_of_pies pecan_pies apple_pies) pies_per_row = 6 :=
by 
  sorry

end mrs_hilt_rows_of_pies_l379_379581


namespace identify_pyramid_scheme_l379_379650

-- Definitions based on the given conditions
def high_return : Prop := offers_significantly_higher_than_average_returns
def lack_of_information : Prop := lack_of_complete_information_about_company
def aggressive_advertising : Prop := aggressive_advertising_occurs

-- Defining the predicate 
def is_pyramid_scheme (all_conditions : Prop) : Prop :=
  offers_significantly_higher_than_average_returns ∧
  lack_of_complete_information_about_company ∧
  aggressive_advertising_occurs

-- The main theorem to prove
theorem identify_pyramid_scheme :
  (high_return ∧ lack_of_information ∧ aggressive_advertising) → is_pyramid_scheme (high_return ∧ lack_of_information ∧ aggressive_advertising) :=
by
  intro h
  exact h

end identify_pyramid_scheme_l379_379650


namespace problem_solution_l379_379724

theorem problem_solution (x y : ℝ) (h₁ : (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1))
  (h₂ : y ≠ 0) :
  (x = 0 ∧ (y = 1/2 ∨ y = -1/2)) :=
by {
  sorry -- Proof required
}

end problem_solution_l379_379724


namespace pyramid_scheme_indicator_l379_379643

def financial_pyramid_scheme_indicator (high_return lack_full_information aggressive_advertising : Prop) : Prop :=
  high_return ∧ lack_full_information ∧ aggressive_advertising

theorem pyramid_scheme_indicator
  (high_return : Prop)
  (lack_full_information : Prop)
  (aggressive_advertising : Prop)
  (indicator : financial_pyramid_scheme_indicator high_return lack_full_information aggressive_advertising) :
  indicator = (high_return ∧ lack_full_information ∧ aggressive_advertising) :=
sorry

end pyramid_scheme_indicator_l379_379643


namespace radius_of_inscribed_sphere_l379_379532

variables (A B C D : Type) [euclidean_space A B C D]

-- Definitions of points
variable (AB : ℝ)
variable (BC : ℝ)
variable (AC : ℝ)

-- Given conditions:
def rect_ABCD_folded_diagonal_AC (AB BC AC : ℝ) :=
  AB = 4 ∧ BC = 3 ∧ AC = real.sqrt(AB^2 + BC^2)

-- Proving the radius of the inscribed sphere
theorem radius_of_inscribed_sphere {AB BC AC : ℝ} :
  AB = 4 → BC = 3 → AC = real.sqrt(AB^2 + BC^2) → 
  let height := BC / 2 in
  radius_of_sphere_inscribed_in_tetrahedron D A B C = height :=
begin
  intros h1 h2 h3, 
  rw [h1, h2] at *,
  sorry
end

end radius_of_inscribed_sphere_l379_379532


namespace floor_of_negative_sqrt_l379_379015

noncomputable def eval_expr : ℚ := -real.sqrt (64 / 9)

theorem floor_of_negative_sqrt : ⌊eval_expr⌋ = -3 :=
by
  -- skip proof
  sorry

end floor_of_negative_sqrt_l379_379015


namespace who_is_next_to_Denis_l379_379256

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379256


namespace fourth_root_equiv_l379_379395

theorem fourth_root_equiv (x : ℝ) (hx : 0 < x) : (x * (x ^ (3 / 4))) ^ (1 / 4) = x ^ (7 / 16) :=
sorry

end fourth_root_equiv_l379_379395


namespace kite_area_proof_l379_379781

-- Define the vertices of the kite and the central square within the kite
def kite_vertices : List (ℤ × ℤ) := [(1, 6), (4, 7), (7, 6), (4, 0)]
def square_center : (ℤ × ℤ) := (4, 3)
def square_side_length : ℕ := 2

-- Define the total kite area
def total_kite_area : ℕ := 10

-- Formalize the proof statement
theorem kite_area_proof : 
  (calculate_kite_area kite_vertices square_center square_side_length) = total_kite_area :=
sorry

-- Here we assume the existence of calculate_kite_area function

end kite_area_proof_l379_379781


namespace only_neg3_smaller_than_neg2_l379_379716

theorem only_neg3_smaller_than_neg2 :
  ∀ (a : ℤ), a ∈ SetOfNumbers a → a < -2 ↔ a = -3 :=
by
  sorry

def SetOfNumbers (a : ℤ) : Prop :=
  a = -3 ∨ a = | -4 | ∨ a = 0 ∨ a = - (-2)

end only_neg3_smaller_than_neg2_l379_379716


namespace problem_1_problem_2_l379_379176

open Set

def A := {x : ℝ | 1 ≤ 3^x ∧ 3^x < 9}
def B := {x : ℝ | Real.log x / Real.log 2 ≥ 0}
def C (a : ℝ) := {x : ℝ | x > -a}

theorem problem_1 : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

theorem problem_2 (a : ℝ) : B ∩ C a = B → a > -1 :=
by sorry

end problem_1_problem_2_l379_379176


namespace probability_palindromic_phone_number_l379_379865

theorem probability_palindromic_phone_number (m : ℕ) (h : m ≠ 0) :
  let total_numbers := m * 10^6 in
  let palindromic_numbers := m * 10^3 in
  total_numbers ≠ 0 → palindromic_numbers / total_numbers = 0.001 :=
by
  intro total_numbers palindromic_numbers h_total
  -- Here would go the proof steps
  sorry

end probability_palindromic_phone_number_l379_379865


namespace sum_even_102_to_200_l379_379977

noncomputable def sum_even_integers (a b : ℕ) :=
  let n := (b - a) / 2 + 1
  in (n * (a + b)) / 2

theorem sum_even_102_to_200 :
  sum_even_integers 102 200 = 7550 := 
by
  have n : ℕ := (200 - 102) / 2 + 1
  have sum : ℕ := (n * (102 + 200)) / 2
  have n_50 : n = 50 := by sorry
  have sum_7550 : sum = 7550 := by sorry
  exact sum_7550 

end sum_even_102_to_200_l379_379977


namespace locus_of_centers_l379_379419

set_option pp.notation false -- To ensure nicer looking lean code.

-- Define conditions for circles C_3 and C_4
def C3 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C4 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Statement to prove the locus of centers satisfies the equation
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)) →
  (a^2 + 18 * b^2 - 6 * a - 440 = 0) :=
by
  sorry -- Proof not required as per the instructions

end locus_of_centers_l379_379419


namespace folded_paper_length_is_3_l379_379132

-- Defining the problem conditions
def square_side_length : ℝ := 8
def midpoint_half_distance : ℝ := square_side_length / 2
def folded_point_to_f_magnitude (x : ℝ) : Prop :=
  let l1 := square_side_length - x in
  let l2 := x^2 + midpoint_half_distance^2 in
  l1^2 = l2

-- Defining the theorem to be proved
theorem folded_paper_length_is_3 : ∃ x : ℝ, folded_point_to_f_magnitude x ∧ x = 3 :=
by {
  sorry
}

end folded_paper_length_is_3_l379_379132


namespace proof_solution_l379_379115

noncomputable def proof_problem : Prop :=
  ∀ (a : ℝ) (b : ℝ),
    a ≠ 0 →
    (2 - (Complex.i : ℂ)) * (a * Complex.i) = 4 - b * Complex.i →
    b = -8

-- Skipping the proof with sorry
theorem proof_solution : proof_problem := by
  intros a b hneq heq
  sorry

end proof_solution_l379_379115


namespace complement_of_M_l379_379093

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M :
  ∀ x, x ∈ U \ M ↔ x < -2 ∨ x > 2 :=
by
  sorry

end complement_of_M_l379_379093


namespace find_minimal_sum_of_distances_find_minimal_sum_of_squares_l379_379053

variable {lineQ1 : Type} [LinearMap.lineQ1] 
variables (l : lineQ1) (M : lineQ1 → lineQ1) (A : lineQ1 → lineQ1) (B : lineQ1 → lineQ1) (O : lineQ1)

def Line_passes_through (l : lineQ1) (M : lineQ1 → lineQ1) : Prop := 
  ∃ t, l t = M

def Intersects_positive_semi_axes (l : lineQ1) (A : lineQ1 → lineQ1) (B : lineQ1 → lineQ1) : Prop := 
  ∃ a b, A a = (a, 0) ∧ B b = (0, b) ∧ a > 0 ∧ b > 0 ∧ l = lineQ1 ℕ

def Origin (O : lineQ1) : Prop :=
  O = (0, 0)

def distance (P Q : lineQ1) : ℝ := sorry -- Define Euclidean distance

def squared_distance (P Q : lineQ1) : ℝ := (distance P Q)^2

theorem find_minimal_sum_of_distances (l : lineQ1) (M : lineQ1 → lineQ1) 
  (A : lineQ1 → lineQ1) (B : lineQ1 → lineQ1) (O : lineQ1) 
  (h1 : Line_passes_through l M) 
  (h2 : Intersects_positive_semi_axes l A B)
  (h3 : Origin O) :
  (distance O A + distance O B) = 4 ∧ 
  l = (λ (t : lineQ1), (t + 1, t - 1)) := sorry

theorem find_minimal_sum_of_squares (l : lineQ1) (M : lineQ1 → lineQ1) 
  (A : lineQ1 → lineQ1) (B : lineQ1 → lineQ1) (O : lineQ1)
  (h1 : Line_passes_through l M) 
  (h2 : Intersects_positive_semi_axes l A B)
  (h3 : Origin O) :
  (squared_distance M A + squared_distance M B) = 4 ∧ 
  l = (λ (t : lineQ1), (t + 1, t - 1)) := sorry

end find_minimal_sum_of_distances_find_minimal_sum_of_squares_l379_379053


namespace water_speed_calc_l379_379372

entity swimming_problem where
  parameter swimming_still_water_speed : ℝ
  parameter time_against_current : ℝ
  parameter distance_against_current : ℝ
  parameter water_speed : ℝ
  
  def effective_speed_against_current := swimming_still_water_speed - water_speed
  
  axiom distance_formula :
    distance_against_current = effective_speed_against_current * time_against_current
  
  theorem water_speed_calc :
    swimming_still_water_speed = 6 ∧ time_against_current = 3.5 ∧ distance_against_current = 14 →
    water_speed = 2 := by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  have h4 : effective_speed_against_current = 14 / 3.5 := by
    rw [distance_formula, h3]
    exact rfl
  have h5 : water_speed = 6 - effective_speed_against_current := by
    rw [← h1, ← h4]
    exact rfl
  exact h5

end water_speed_calc_l379_379372


namespace vectors_relationship_l379_379138

open Real

def vector_a : ℝ × ℝ × ℝ := (-2, 1, 3)
def vector_b : ℝ × ℝ × ℝ := (1, -1, 1)
def vector_c : ℝ × ℝ × ℝ := (1, -1/2, -3/2)

theorem vectors_relationship :
  let dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
  dot_product vector_a vector_b = 0 ∧ ∃ k : ℝ, vector_a = (k * vector_c.1, k * vector_c.2, k * vector_c.3) :=
by
  sorry

end vectors_relationship_l379_379138


namespace MrsHiltRows_l379_379578

theorem MrsHiltRows :
  let (a : ℕ) := 16
  let (b : ℕ) := 14
  let (r : ℕ) := 5
  (a + b) / r = 6 := by
  sorry

end MrsHiltRows_l379_379578


namespace distance_from_point_to_plane_l379_379444

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M1 : Point3D := ⟨1, 2, 0⟩
def M2 : Point3D := ⟨3, 0, -3⟩
def M3 : Point3D := ⟨5, 2, 6⟩
def M0 : Point3D := ⟨-13, -8, 16⟩

def plane (p1 p2 p3 : Point3D) : (ℝ × ℝ × ℝ × ℝ) :=
  let x1 := p1.x
  let y1 := p1.y
  let z1 := p1.z
  let x2 := p2.x
  let y2 := p2.y
  let z2 := p2.z
  let x3 := p3.x
  let y3 := p3.y
  let z3 := p3.z
  let A := (y1 - y2) * (z2 - z3) - (y2 - y3) * (z1 - z2)
  let B := (z1 - z2) * (x2 - x3) - (z2 - z3) * (x1 - x2)
  let C := (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)
  let D := - (A * x1 + B * y1 + C * z1)
  (A, B, C, D)

def distance (plane : ℝ × ℝ × ℝ × ℝ) (point : Point3D) : ℝ :=
  let (A, B, C, D) := plane
  let numerator := abs (A * point.x + B * point.y + C * point.z + D)
  let denominator := (A^2 + B^2 + C^2).sqrt
  numerator / denominator

theorem distance_from_point_to_plane : distance (plane M1 M2 M3) M0 = 134 / 7 := by
  sorry

end distance_from_point_to_plane_l379_379444


namespace minimum_distance_to_line_in_range_l379_379566

def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

def axis_of_symmetry (Q : ℝ × ℝ) : Prop := Q.1 = -1 ∧ Q.2 ≠ 0

def line_equation (Q : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ t : ℝ, l = λ (x y : ℝ), x - t * y + t^2 = -1

def minimum_distance_range_condition (d : ℝ) : Prop :=
  d > 0 ∧ d < 1

theorem minimum_distance_to_line_in_range (P : ℝ × ℝ) (Q : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  parabola P →
  axis_of_symmetry Q →
  line_equation Q l →
  ∃ d : ℝ, minimum_distance_range_condition d :=
sorry

end minimum_distance_to_line_in_range_l379_379566


namespace tan_identity_proof_l379_379041

noncomputable def tan_add_pi_over_3 (α β : ℝ) : ℝ :=
  Real.tan (α + Real.pi / 3)

theorem tan_identity_proof 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan (β - Real.pi / 3) = 1 / 4) :
  tan_add_pi_over_3 α β = 7 / 23 := 
sorry

end tan_identity_proof_l379_379041


namespace num_valid_x_values_is_50_l379_379164

def sum_of_digits (n : Nat) : Nat :=
  (n.digits 10).sum

def num_valid_x_values : Nat :=
  List.length (List.filter (λ x, sum_of_digits (sum_of_digits x) = 4) (List.range' 100 (999 - 100 + 1)))

theorem num_valid_x_values_is_50 : num_valid_x_values = 50 := by
  sorry

end num_valid_x_values_is_50_l379_379164


namespace martin_boxes_l379_379182

theorem martin_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (number_of_boxes : ℕ) 
  (h1 : total_crayons = 56) (h2 : crayons_per_box = 7) 
  (h3 : total_crayons = crayons_per_box * number_of_boxes) : 
  number_of_boxes = 8 :=
by 
  sorry

end martin_boxes_l379_379182


namespace seats_scientific_notation_l379_379213

theorem seats_scientific_notation : 
  (13000 = 1.3 * 10^4) := 
by 
  sorry 

end seats_scientific_notation_l379_379213


namespace domain_of_power_function_l379_379613

theorem domain_of_power_function : 
  (∃ dom : set ℝ, dom = {x | x ≥ 0} ∧ ∀ x, x ∈ dom → (x^{(1 / 4)} : ℝ)) :=
by
  sorry

end domain_of_power_function_l379_379613


namespace lim_pn_div_qn_lim_pn_div_rn_lim_pn_div_sn_l379_379681

noncomputable def pn (n : ℕ) : ℝ := (1 + real.sqrt 2 + real.sqrt 3) ^ n
noncomputable def qn (n : ℕ) : ℝ := real.sqrt 2 * pn n
noncomputable def rn (n : ℕ) : ℝ := real.sqrt 3 * pn n
noncomputable def sn (n : ℕ) : ℝ := real.sqrt 6 * pn n

theorem lim_pn_div_qn (n : ℕ) : tendsto (λ n, (pn n) / (qn n)) at_top (𝓝 (real.sqrt 2)) :=
sorry

theorem lim_pn_div_rn (n : ℕ) : tendsto (λ n, (pn n) / (rn n)) at_top (𝓝 (real.sqrt 3)) :=
sorry

theorem lim_pn_div_sn (n : ℕ) : tendsto (λ n, (pn n) / (sn n)) at_top (𝓝 (real.sqrt 6)) :=
sorry

end lim_pn_div_qn_lim_pn_div_rn_lim_pn_div_sn_l379_379681


namespace necessary_but_not_sufficient_l379_379046

variables (x : ℝ)

def p : Prop := x < 3
def q : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient : (q → p) ∧ ¬(p → q) := 
by {
  sorry
}

end necessary_but_not_sufficient_l379_379046


namespace cost_price_of_table_l379_379676

theorem cost_price_of_table (CP SP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3000) : CP = 2500 := by
    sorry

end cost_price_of_table_l379_379676


namespace team_A_wins_after_4_games_l379_379999

-- Defining the probabilities based on the given conditions
def prob_A_win : ℝ := 3 / 5
def prob_B_win : ℝ := 2 / 5

-- Lean statement to prove the probability that Team A wins after 4 games is 162/625
theorem team_A_wins_after_4_games : 
  let p := 3 * (prob_A_win ^ 3) * (prob_B_win) * prob_A_win in
  p = 162 / 625 := 
by
  -- Expected proof will go here
  sorry

end team_A_wins_after_4_games_l379_379999


namespace sin_2a_tan_pi_over_3_plus_a_l379_379785

open Real TrigonometricFunction

noncomputable def sin_value : ℝ := sqrt 5 / 5
noncomputable def angle_range (a : ℝ) : Prop := π/2 < a ∧ a < π

theorem sin_2a (a : ℝ) (h1 : sin a = sin_value) (h2 : angle_range a) :
  sin (2 * a) = -4 / 5 :=
sorry

theorem tan_pi_over_3_plus_a (a : ℝ) (h1 : sin a = sin_value) (h2 : angle_range a) :
  tan (π/3 + a) = 5 * sqrt 3 - 8 :=
sorry

end sin_2a_tan_pi_over_3_plus_a_l379_379785


namespace root_of_f_l379_379948

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

theorem root_of_f (h_inv : f_inv 0 = 2) (h_interval : 1 ≤ (f_inv 0) ∧ (f_inv 0) ≤ 4) : f 2 = 0 := 
sorry

end root_of_f_l379_379948


namespace three_Z_five_l379_379842

def Z (a b : ℤ) : ℤ := b + 7 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = -1 := by
  sorry

end three_Z_five_l379_379842


namespace soda_last_days_l379_379734

theorem soda_last_days (daily_consumption_ml : ℕ) (total_volume_l : ℕ) (ml_per_l : ℕ) :
  (total_volume_l * ml_per_l) / daily_consumption_ml = 4 :=
by
  assume daily_consumption_ml = 500
  assume total_volume_l = 2
  assume ml_per_l = 1000
  calc (total_volume_l * ml_per_l) / daily_consumption_ml
      = (2 * 1000) / 500 : by sorry
      ... = 2000 / 500   : by sorry
      ... = 4            : by sorry

end soda_last_days_l379_379734


namespace slices_with_both_toppings_correct_l379_379356

noncomputable def slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices : ℕ) : ℕ :=
  let N := total_slices
  let P := pepperoni_slices
  let M := mushroom_slices
  (P + M - N)

theorem slices_with_both_toppings_correct (H1 : total_slices = 12) (H2 : pepperoni_slices = 6) (H3 : mushroom_slices = 10) :
  slices_with_both_toppings total_slices pepperoni_slices mushroom_slices = 4 :=
by
  rw [slices_with_both_toppings, H1, H2, H3]
  sorry

end slices_with_both_toppings_correct_l379_379356


namespace who_is_next_to_Denis_l379_379255

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379255


namespace triangle_area_l379_379725

-- Define the vertices of the triangle
def u : ℝ × ℝ × ℝ := (4, -2, 3)
def v : ℝ × ℝ × ℝ := (1, -3, 2)
def w : ℝ × ℝ × ℝ := (7, 4, 6)

-- Define the function to compute the area of the triangle
def area_of_triangle (u v w : ℝ × ℝ × ℝ) : ℝ :=
  let a := (v.1 - u.1, v.2 - u.2, v.3 - u.3) in
  let b := (w.1 - u.1, w.2 - u.2, w.3 - u.3) in
  let cross_product := (
    a.2 * b.3 - a.3 * b.2,
    a.3 * b.1 - a.1 * b.3,
    a.1 * b.2 - a.2 * b.1
  ) in
  0.5 * real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)

-- Assert the theorem to be proved
theorem triangle_area : area_of_triangle u v w = 9 * real.sqrt 10 / 2 :=
by 
  -- Place proof here
  sorry

end triangle_area_l379_379725


namespace ten_gentlemen_probability_l379_379437

noncomputable def harmonic_number (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), 1 / (i + 1 : ℝ)

noncomputable def probability (n : ℕ) : ℝ :=
  (harmonic_number n) / n

theorem ten_gentlemen_probability :
  let p_10 := ∏ i in finset.range 10, probability (i + 1)
  in p_10 ≈ 0.000516 :=
by
  let p_10 := ∏ i in finset.range 10, probability (i + 1)
  have : p_10 ≈ 0.000516, from 
    sorry
  exact this

end ten_gentlemen_probability_l379_379437


namespace find_a_l379_379456

-- Define the complex numbers and multiplication
def complex (r i : ℝ) := ℂ.mk r i

-- The given conditions
variables (a : ℝ) (z : ℂ)
hypothesis h1 : (2+complex.i) * z = a + complex.i * 2
hypothesis h2 : z.re = 2 * z.im

-- The goal statement
theorem find_a : a = 3/2 :=
by sorry

end find_a_l379_379456


namespace sum_fibonacci_series_l379_379887

noncomputable def fibonacci (a₁ a₂ : ℕ) : ℕ → ℕ
| 1 := a₁
| 2 := a₂
| (n + 1) := fibonacci (n - 1) + fibonacci n

lemma fibonacci_recurrence (n ≥ 1) : 
  fibonacci 2 3 (n + 2) = fibonacci 2 3 (n + 1) + fibonacci 2 3 n := 
by sorry

theorem sum_fibonacci_series : 
  (∑' n, (fibonacci 2 3 n) / 3^(n+1)) = 2/5 :=
by sorry

end sum_fibonacci_series_l379_379887


namespace circle_eq_l379_379228

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end circle_eq_l379_379228


namespace tens_digit_8_2015_l379_379652

theorem tens_digit_8_2015 : ∀ n ∈ [1, 2, ..., 20], ∃ ten_digit: ℕ, ten_digit = 3 :=
  sorry

end tens_digit_8_2015_l379_379652


namespace problem1_problem2_l379_379405

def f (x y : ℝ) : ℝ := x^2 * y

def P0 : ℝ × ℝ := (5, 4)

def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

def Δf (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  f (P.1 + Δx) (P.2 + Δy) - f P.1 P.2

def df (f : ℝ → ℝ → ℝ) (P : ℝ × ℝ) (Δx Δy : ℝ) : ℝ :=
  (2 * P.1 * P.2) * Δx + (P.1^2) * Δy

theorem problem1 : Δf f P0 Δx Δy = -1.162 := 
  sorry

theorem problem2 : df f P0 Δx Δy = -1 :=
  sorry

end problem1_problem2_l379_379405


namespace candidate_lost_by_2100_votes_l379_379687

theorem candidate_lost_by_2100_votes :
  let total_votes := 6999.999999999999
  let rounded_votes := 7000
  let candidate_percentage := 0.35
  let rival_percentage := 0.65
  let candidate_votes := candidate_percentage * rounded_votes
  let rival_votes := rival_percentage * rounded_votes
  let vote_difference := rival_votes - candidate_votes
in vote_difference = 2100 := by
sorry

end candidate_lost_by_2100_votes_l379_379687


namespace arithmetic_sequence_problem_l379_379064

noncomputable def a_n (a1 d : ℚ) (n : ℕ) : ℚ := a1 + n * d

theorem arithmetic_sequence_problem
  (a1 d : ℚ)
  (h1 : ∀ (n : ℕ) (hn : n > 0), a_n a1 d n + a_n a1 d (n+1) = 3*n + 5)
  : a1 = 7/4 :=
by {
  have h1_n_1: a1 + d = 3 * 0 + 5 / 2, from sorry,
  have h1_n_2: a1 + d + d = 3 * 1 + 5, from sorry,
  sorry
}

end arithmetic_sequence_problem_l379_379064


namespace probability_A_in_middle_l379_379655

theorem probability_A_in_middle : 
  let arrangements := ['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA'] in
  let favorable := ['BAC', 'CBA'] in
  (favorable.length: ℚ) / (arrangements.length: ℚ) = 1 / 3 :=
by
  sorry

end probability_A_in_middle_l379_379655


namespace correct_option_l379_379668

-- Define the variable 'a' as a real number
variable (a : ℝ)

-- Define propositions for each option
def option_A : Prop := 5 * a ^ 2 - 4 * a ^ 2 = 1
def option_B : Prop := (a ^ 7) / (a ^ 4) = a ^ 3
def option_C : Prop := (a ^ 3) ^ 2 = a ^ 5
def option_D : Prop := a ^ 2 * a ^ 3 = a ^ 6

-- State the main proposition asserting that option B is correct and others are incorrect
theorem correct_option :
  option_B a ∧ ¬option_A a ∧ ¬option_C a ∧ ¬option_D a :=
  by sorry

end correct_option_l379_379668


namespace polynomial_coefficients_sum_l379_379843

theorem polynomial_coefficients_sum :
  let p := (5 * x^3 - 3 * x^2 + x - 8) * (8 - 3 * x)
  let a := -15
  let b := 49
  let c := -27
  let d := 32
  let e := -64
  16 * a + 8 * b + 4 * c + 2 * d + e = 44 := 
by
  sorry

end polynomial_coefficients_sum_l379_379843


namespace greatest_possible_large_chips_l379_379985

theorem greatest_possible_large_chips :
  ∃ l s : ℕ, ∃ p : ℕ, s + l = 61 ∧ s = l + p ∧ Nat.Prime p ∧ l = 29 :=
sorry

end greatest_possible_large_chips_l379_379985


namespace pyramid_parallel_midline_l379_379871

-- Definitions representing the conditions
def isRegularPyramid (P A B C D O E : Type) : Prop :=
  -- ∙ P, A, B, C, D form a regular pyramid
  true ∧ 
  -- ∙ O is the projection of P onto the base
  true ∧
  -- ∙ E is the midpoint of PC
  true ∧
  -- ∙ The base ABCD is a square
  true ∧
  -- ∙ O is the midpoint of AC
  true

-- To prove: AP ∥ OE given the above conditions
theorem pyramid_parallel_midline (P A B C D O E : Type) 
  (h : isRegularPyramid P A B C D O E) : 
  -- AP ∥ OE
  (AP ∥ OE) :=
sorry

end pyramid_parallel_midline_l379_379871


namespace solve_for_b_l379_379906

theorem solve_for_b (b : ℝ) (m : ℝ) (h : b > 0)
  (h1 : ∀ x : ℝ, x^2 + b * x + 54 = (x + m) ^ 2 + 18) : b = 12 :=
by
  sorry

end solve_for_b_l379_379906


namespace roses_in_garden_l379_379549

theorem roses_in_garden (x : ℕ) (cut : ℕ) (in_vase : ℕ) (total : ℕ) :
  cut = 13 → in_vase = 7 → total = 20 → 
  in_vase + cut = total → 
  x = cut :=
by
  intros h_cut h_in_vase h_total h_eq
  rw [h_cut] at h_eq
  rw [h_in_vase] at h_eq
  rw [h_total] at h_eq
  exact eq.symm h_cut

end roses_in_garden_l379_379549


namespace probability_sine_interval_l379_379135

theorem probability_sine_interval : 
  let interval := set.Icc (0/1 : ℝ) (Real.pi / 2)
  let event := {x | x ∈ interval ∧ ((1 / 2 : ℝ) ≤ Real.sin x ∧ Real.sin x ≤ Real.sqrt 3 / 2)}
  (∀ x ∈ interval, real.is_measurable_set event) →
  (∀ x ∈ interval, real.is_finite_measure event) →
  (real.measure_of event / real.measure_of interval = 1 / 3) :=
sorry

end probability_sine_interval_l379_379135


namespace number_of_boys_in_second_group_l379_379831

noncomputable def daily_work_done_by_man (M : ℝ) (B : ℝ) : Prop :=
  M = 2 * B

theorem number_of_boys_in_second_group
  (M B : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = (13 * M + 24 * B) * 4)
  (h2 : daily_work_done_by_man M B) :
  24 = 24 :=
by
  -- The proof is omitted.
  sorry

end number_of_boys_in_second_group_l379_379831


namespace solve_for_x_l379_379206

-- Definitions from conditions
def exponential_equation (x : ℝ) : Prop :=
  3^x * 9^x = 27^(x - 12)

-- Stating the proof problem
theorem solve_for_x : ∃ (x : ℝ), exponential_equation x ∧ x = 12 :=
sorry

end solve_for_x_l379_379206


namespace length_TU_l379_379544

noncomputable section

variables {P Q R S T U : Type} [inner_product_space ℝ P]

-- Definitions for points and distances
variables (QR PS V : P)
variables (P Q R S T U : P)
variables (a b c d : ℝ)

variables (PQ_parallel_RS : (QR + PS) / 2 = V)
variables (midpoints : T = (P + Q) / 2 ∧ U = (S + R) / 2)
variables (angles : ∠P = 45 * π / 180 ∧ ∠S = 45 * π / 180)

variables (base_lengths : dist P Q = 800 ∧ dist S R = 1600)

theorem length_TU (QR_parallel_PS : dist QR PS = 800 + 800) : dist T U = 400 :=
by
  -- The conditions are explicitly stated 
  have h1 : dist QR PS = 1600 := QR_parallel_PS
  have h2 : ∠ PVS = 90 * π / 180 := by sorry
  have h3 : dist VU = 800 := by sorry
  have h4 : dist VR = 400 := by sorry
  have collinear : collinear ℝ ![V, T, U] := by sorry
  calc
    dist T U = dist VU - dist VT : by sorry
    ...       = 800 - 400 : by sorry
    ...       = 400 : by sorry

end length_TU_l379_379544


namespace remainder_of_sum_l379_379166

theorem remainder_of_sum (a b c : ℕ) (h₁ : a * b * c % 7 = 1) (h₂ : 2 * c % 7 = 5) (h₃ : 3 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_of_sum_l379_379166


namespace proper_subset_count_l379_379621

theorem proper_subset_count (A : Finset ℕ) (hA : A = {2, 3}) : A.powerset.filter (λ S, S ≠ A).card = 3 :=
by
  have h : A = {2, 3} := hA
  sorry

end proper_subset_count_l379_379621


namespace problem1_problem2_l379_379084

def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem problem1 (x : ℝ) : f x ≥ 4 ↔ x ≤ -4/3 ∨ x ≥ 4/3 := 
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, f x > a) ↔ a < 3/2 := 
  sorry

end problem1_problem2_l379_379084


namespace find_y_l379_379867

-- Definitions of angles and the given problem.
def angle_ABC : ℝ := 90
def angle_ABD (y : ℝ) : ℝ := 3 * y
def angle_DBC (y : ℝ) : ℝ := 2 * y

-- The theorem stating the problem
theorem find_y (y : ℝ) (h1 : angle_ABC = 90) (h2 : angle_ABD y + angle_DBC y = angle_ABC) : y = 18 :=
  by 
  sorry

end find_y_l379_379867


namespace jeff_total_jars_l379_379144

theorem jeff_total_jars (x : ℕ) : 
  16 * x + 28 * x + 40 * x + 52 * x = 2032 → 4 * x = 56 :=
by
  intro h
  -- additional steps to solve the problem would go here.
  sorry

end jeff_total_jars_l379_379144


namespace candle_height_problem_l379_379997

-- Define the conditions given in the problem
def same_initial_height (height : ℝ := 1) := height = 1

def burn_rate_first_candle := 1 / 5

def burn_rate_second_candle := 1 / 4

def height_first_candle (t : ℝ) := 1 - (burn_rate_first_candle * t)

def height_second_candle (t : ℝ) := 1 - (burn_rate_second_candle * t)

-- Define the proof problem
theorem candle_height_problem : ∃ t : ℝ, height_first_candle t = 3 * height_second_candle t ∧ t = 40 / 11 :=
by
  sorry

end candle_height_problem_l379_379997


namespace james_training_hours_in_a_year_l379_379877

-- Definitions based on conditions
def trains_twice_a_day : ℕ := 2
def hours_per_training : ℕ := 4
def days_trains_per_week : ℕ := 7 - 2
def weeks_per_year : ℕ := 52

-- Resultant computation
def daily_training_hours : ℕ := trains_twice_a_day * hours_per_training
def weekly_training_hours : ℕ := daily_training_hours * days_trains_per_week
def yearly_training_hours : ℕ := weekly_training_hours * weeks_per_year

-- Statement to prove
theorem james_training_hours_in_a_year : yearly_training_hours = 2080 := by
  -- proof goes here
  sorry

end james_training_hours_in_a_year_l379_379877


namespace seconds_in_3_hours_45_minutes_l379_379827

theorem seconds_in_3_hours_45_minutes :
  let hours := 3
  let minutes := 45
  let minutes_in_hour := 60
  let seconds_in_minute := 60
  (hours * minutes_in_hour + minutes) * seconds_in_minute = 13500 := by
  sorry

end seconds_in_3_hours_45_minutes_l379_379827


namespace tangent_line_existence_l379_379382

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 1

def is_tangent_line (l : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), (l x y → circle_eq x y) ∧ (circle_eq x y → l x y)

theorem tangent_line_existence :
  ∃ l : ℝ → ℝ → Prop, is_tangent_line l ∧ (l = λ x y, y = 4) ∨ (l = λ x y, 3 * x + 4 * y = 13) :=
sorry

end tangent_line_existence_l379_379382


namespace find_y_l379_379107

-- Define the given conditions
variables (x y : ℚ)
hypothesis1 : x - y = 20
hypothesis2 : 3 * (x + y) = 15

-- The statement to be proven
theorem find_y : y = - (15 / 2) :=
by
  -- Proof steps would go here, but for now, we add sorry to skip proof
  sorry

end find_y_l379_379107


namespace inequality_relation_l379_379557

noncomputable def P : ℝ := Real.log 3 / Real.log 2
noncomputable def Q : ℝ := Real.log 2 / Real.log 3
noncomputable def R : ℝ := Real.log Q / Real.log 2

theorem inequality_relation : R < Q ∧ Q < P := by
  sorry

end inequality_relation_l379_379557


namespace hens_in_caravan_l379_379850

theorem hens_in_caravan :
  ∃ (H : ℕ), H = 50 ∧ 
    let number_of_goats := 45,
        number_of_camels := 8,
        number_of_keepers := 15,
        total_heads := H + number_of_goats + number_of_camels + number_of_keepers,
        total_feet := 2 * H + 4 * number_of_goats + 4 * number_of_camels + 2 * number_of_keepers in
    total_feet = total_heads + 224 := 
by
  sorry

end hens_in_caravan_l379_379850


namespace schoolchildren_lineup_l379_379307

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379307


namespace sqrt_47_minus_2_range_l379_379435

theorem sqrt_47_minus_2_range (h : 6 < Real.sqrt 47 ∧ Real.sqrt 47 < 7) : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end sqrt_47_minus_2_range_l379_379435


namespace remaining_pages_l379_379958

theorem remaining_pages (total_pages : ℕ) (science_project_percentage : ℕ) (math_homework_pages : ℕ)
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 25) 
  (h3 : math_homework_pages = 10) : 
  total_pages - (total_pages * science_project_percentage / 100) - math_homework_pages = 80 := by
  sorry

end remaining_pages_l379_379958


namespace constant_term_expansion_l379_379131

theorem constant_term_expansion : 
  let f := (fun (x : ℝ) => (2 * x - 1/ (sqrt x)) ^ 6)
  in constant_term f = 60 :=
by
  unfold constant_term
  -- Proof goes here
  sorry

end constant_term_expansion_l379_379131


namespace solution_set_of_inequality_l379_379803

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf_diff : differentiable ℝ f)
  (hf_ineq : ∀ x, deriv f x > f x) :
  {x : ℝ | e^(f (real.log x)) - x * f 1 < 0} = set.Ioo 0 (real.exp 1) :=
by sorry

end solution_set_of_inequality_l379_379803


namespace symmetric_point_xOy_l379_379139

def Point := (ℝ × ℝ × ℝ)

def symmetric_point (P : Point) : Point :=
  (P.1, P.2, -P.3)

theorem symmetric_point_xOy (P : Point) (hP : P = (1, 2, 3)) : symmetric_point P = (1, 2, -3) := by
  sorry

end symmetric_point_xOy_l379_379139


namespace extraMaterialNeeded_l379_379907

-- Box dimensions
def smallBoxLength (a : ℝ) : ℝ := a
def smallBoxWidth (b : ℝ) : ℝ := 1.5 * b
def smallBoxHeight (c : ℝ) : ℝ := c

def largeBoxLength (a : ℝ) : ℝ := 1.5 * a
def largeBoxWidth (b : ℝ) : ℝ := 2 * b
def largeBoxHeight (c : ℝ) : ℝ := 2 * c

-- Volume calculations
def volumeSmallBox (a b c : ℝ) : ℝ := a * (1.5 * b) * c
def volumeLargeBox (a b c : ℝ) : ℝ := (1.5 * a) * (2 * b) * (2 * c)

-- Surface area calculations
def surfaceAreaSmallBox (a b c : ℝ) : ℝ := 2 * (a * (1.5 * b)) + 2 * (a * c) + 2 * ((1.5 * b) * c)
def surfaceAreaLargeBox (a b c : ℝ) : ℝ := 2 * ((1.5 * a) * (2 * b)) + 2 * ((1.5 * a) * (2 * c)) + 2 * ((2 * b) * (2 * c))

-- Proof statement
theorem extraMaterialNeeded (a b c : ℝ) :
  (volumeSmallBox a b c = 1.5 * a * b * c) ∧ (volumeLargeBox a b c = 6 * a * b * c) ∧ 
  (surfaceAreaLargeBox a b c - surfaceAreaSmallBox a b c = 3 * a * b + 4 * a * c + 5 * b * c) :=
by
  sorry

end extraMaterialNeeded_l379_379907


namespace sum_trigonometric_identity_l379_379170

theorem sum_trigonometric_identity :
  let k := Real.pi / 180 in
  ∑ n in Finset.range 89, 1 / (Real.cos (n * k) * Real.cos ((n + 1) * k)) = Real.cos k / Real.sin k ^ 2 := by
  sorry

end sum_trigonometric_identity_l379_379170


namespace f_of_9_eq_11_l379_379931

def f : ℤ → ℤ
| x := if x ≥ 10 then x - 2 else f (f (x + 6))

theorem f_of_9_eq_11 : f 9 = 11 :=
sorry

end f_of_9_eq_11_l379_379931


namespace probability_of_at_least_one_contract_l379_379690

noncomputable def P_A : ℚ := 3 / 4
noncomputable def P_not_B : ℚ := 5 / 9
noncomputable def P_A_and_B : ℚ := 0.3944444444444444

def P_B : ℚ := 1 - P_not_B
def P_A_or_B : ℚ := P_A + P_B - P_A_and_B

theorem probability_of_at_least_one_contract : P_A_or_B = 29 / 36 := 
sorry

end probability_of_at_least_one_contract_l379_379690


namespace who_is_next_to_denis_l379_379279

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379279


namespace f_value_at_2_9_l379_379422

-- Define the function f with its properties as conditions
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the domain of f
axiom f_domain : ∀ x, 0 ≤ x ∧ x ≤ 1

-- Condition (i)
axiom f_0_eq : f 0 = 0

-- Condition (ii)
axiom f_monotone : ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- Condition (iii)
axiom f_symmetry : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (1 - x) = 3/4 - f x / 2

-- Condition (iv)
axiom f_scale : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f (x / 3) = f x / 3

-- Proof goal
theorem f_value_at_2_9 : f (2/9) = 5/24 := by
  sorry

end f_value_at_2_9_l379_379422


namespace exists_c_with_same_nonzero_decimal_digits_l379_379570

theorem exists_c_with_same_nonzero_decimal_digits (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  ∃ (c : ℕ), 0 < c ∧ (∃ (k : ℕ), (c * m) % 10^k = (c * n) % 10^k) := 
sorry

end exists_c_with_same_nonzero_decimal_digits_l379_379570


namespace simeon_fewer_servings_l379_379599

-- Given conditions
def prevDailyIntake : ℝ := 64
def prevServingSize : ℝ := 8
def newDailyIntake : ℝ := 80
def newServingSize : ℝ := 12

-- Define the number of servings per day previously and now
def prevServingsPerDay : ℝ := prevDailyIntake / prevServingSize
def newServingsPerDay : ℝ := Real.ceil (newDailyIntake / newServingSize) -- ceil for rounding up

-- Theorem to prove
theorem simeon_fewer_servings :
  prevServingsPerDay - newServingsPerDay = 1 :=
by
  sorry

end simeon_fewer_servings_l379_379599


namespace who_is_next_to_denis_l379_379280

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379280


namespace third_side_length_integer_l379_379526

noncomputable
def side_a : ℝ := 3.14

noncomputable
def side_b : ℝ := 0.67

def is_valid_triangle_side (side: ℝ) : Prop :=
  side_a - side_b < side ∧ side < side_a + side_b

theorem third_side_length_integer (side: ℕ) : is_valid_triangle_side side.to_real → side = 3 :=
  by
  sorry

end third_side_length_integer_l379_379526


namespace coefficient_of_x_in_binomial_expansion_l379_379938

theorem coefficient_of_x_in_binomial_expansion :
  (∃ r : ℕ, 5.choose r * (-1)^r * 2^(5-r) * x^(3*r - 5) = 80) :=
by
  sorry

end coefficient_of_x_in_binomial_expansion_l379_379938


namespace smallest_positive_x_for_max_value_l379_379417

theorem smallest_positive_x_for_max_value :
  ∃ x > 0, (∀ y > 0, x ≤ y) ∧ (∀ a b : ℝ, a = x / 5 → b = x / 13 → sin a + sin b = 2) :=
sorry

end smallest_positive_x_for_max_value_l379_379417


namespace alpha_gt_beta_neither_necessary_nor_sufficient_l379_379068

variable {α β : ℝ}

-- Defining the first quadrant condition
def first_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π/2

-- The main statement
theorem alpha_gt_beta_neither_necessary_nor_sufficient (hα : first_quadrant α) (hβ : first_quadrant β) :
  (α > β) → (sin α > sin β) ∨ (sin α < sin β) → ¬(α > β) :=
sorry

end alpha_gt_beta_neither_necessary_nor_sufficient_l379_379068


namespace number_of_isthmian_arrangements_l379_379583

noncomputable def isthmian_arrangements : ℕ :=
  (nat.factorial 5 * nat.factorial 4) / 4

theorem number_of_isthmian_arrangements : isthmian_arrangements = 720 :=
by
  sorry

end number_of_isthmian_arrangements_l379_379583


namespace incorrect_statement_l379_379834

variables (p q : Prop)
variable h_p : ¬p
variable h_q : q

theorem incorrect_statement : ¬ (¬q) :=
by
  sorry

end incorrect_statement_l379_379834


namespace distinct_values_of_f_l379_379169

-- Define the floor function and the range of k values
def floor (r : ℝ) : ℤ := ⌊r⌋

-- Define function f(x)
def f (x : ℝ) : ℤ :=
  ∑ k in finset.range 11, (floor (k + 2) * x - (k + 2) * floor x)

-- The proposition we want to prove
theorem distinct_values_of_f :
  ∃ n : ℕ, n = 46 ∧ (∀ x : ℝ, x ≥ 0 → ∃ l : finset ℤ, (∀ y : ℝ, y ≥ 0 → f y ∈ l) ∧ l.card = n) :=
by sorry

end distinct_values_of_f_l379_379169


namespace number_of_valid_integers_l379_379824

noncomputable def count_valid_integers : ℤ :=
  fintype.card {n : ℤ // n ≠ -1 ∧ n ≠ 0 ∧ n ≠ 1 ∧ (1 / |n|) ≥ (1 / 12)}

theorem number_of_valid_integers : count_valid_integers = 22 := by
  sorry

end number_of_valid_integers_l379_379824


namespace cost_of_green_lettuce_l379_379880

-- Definitions based on the conditions given in the problem
def cost_per_pound := 2
def weight_red_lettuce := 6 / cost_per_pound
def total_weight := 7
def weight_green_lettuce := total_weight - weight_red_lettuce

-- Problem statement: Prove that the cost of green lettuce is $8
theorem cost_of_green_lettuce : (weight_green_lettuce * cost_per_pound) = 8 :=
by
  sorry

end cost_of_green_lettuce_l379_379880


namespace range_of_m_intersect_extension_PQ_l379_379484

noncomputable def directed_segment_intersect_range (m : ℝ) : Prop :=
  let P := (-1 : ℝ, 1 : ℝ)
  let Q := (2 : ℝ, 2 : ℝ)
  -- Line given by its equation
  let l := λ (x y: ℝ), x + m * y + m = 0
  -- The intersecting line with extended PQ
  ∃ x : ℝ, ∃ y : ℝ, l x y ∧ y = x + 2 ∧ x > 2

theorem range_of_m_intersect_extension_PQ : 
  directed_segment_intersect_range m → -3 < m ∧ m < 0 :=
sorry

end range_of_m_intersect_extension_PQ_l379_379484


namespace sin_inequality_condition_l379_379749

theorem sin_inequality_condition (y : ℝ) (h : 0 ≤ y ∧ y ≤ (π / 2)) : 
  (∀ x, 0 ≤ x ∧ x ≤ π → sin (x + y) < sin x + sin y) → y = 0 := 
begin
  sorry
end

end sin_inequality_condition_l379_379749


namespace ratio_of_inscribed_quadrilaterals_l379_379574

theorem ratio_of_inscribed_quadrilaterals (K K1 : Circle) (R R1 : ℝ) (hR1 : R1 > R) 
  (ABCD : inscribed_quadrilateral K) (A1 B1 C1 D1 : C1) :
  inscribed_quadrilateral K1 A1 B1 C1 D1 →
  (lies_on_ray A1 C D) (lies_on_ray B1 D A) (lies_on_ray C1 A B) (lies_on_ray D1 B C) →
  area_ratios R R1 (ABCD) (A₁B₁C₁D₁) ≥ (R1^2) / (R^2)
| ((_: K.K1) → sorry

end ratio_of_inscribed_quadrilaterals_l379_379574


namespace find_third_side_l379_379523

theorem find_third_side (a b : ℝ) (c : ℕ) 
  (h1 : a = 3.14)
  (h2 : b = 0.67)
  (h_triangle_ineq : a + b > ↑c ∧ a + ↑c > b ∧ b + ↑c > a) : 
  c = 3 := 
by
  -- Proof goes here
  sorry

end find_third_side_l379_379523


namespace arcsin_cos_arcsin_arccos_sin_arccos_l379_379032

-- Define the statement
theorem arcsin_cos_arcsin_arccos_sin_arccos (x : ℝ) 
  (h1 : -1 ≤ x) 
  (h2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) := 
sorry

end arcsin_cos_arcsin_arccos_sin_arccos_l379_379032


namespace denis_neighbors_l379_379288

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379288


namespace P_intersection_Q_l379_379173

noncomputable def P : Set ℝ := {x | ∃ y, y = log x / log 2}
noncomputable def Q : Set ℝ := {y | ∃ x, y = x^3}

theorem P_intersection_Q : P ∩ Q = (set.Ioi 0) := by 
  sorry

end P_intersection_Q_l379_379173


namespace angle_PQR_is_90_l379_379130

theorem angle_PQR_is_90 
  (RTS_is_straight : ∃ R T S : ℝ → ℝ, T = 0 ∧ S = 2 * T ∧ angle R T S = 180)
  (angle_QTS : ∃ Q T S : ℝ → ℝ, T = 0 ∧ S = Q + 1 ∧ angle Q T S = 70)
  (isosceles_RTQ : ∃ R T Q, R = T ∧ Q = T ∨ R = Q )
  (isosceles_PTS : ∃ P T S, P = T ∧ S = T ∨ P = S )
  : angle_PQR = 90 :=
by
  sorry

end angle_PQR_is_90_l379_379130


namespace next_perfect_square_l379_379056

theorem next_perfect_square (n : ℤ) (hn : Even n) (x : ℤ) (hx : x = n^2) : 
  ∃ y : ℤ, y = x + 2 * n + 1 ∧ (∃ m : ℤ, y = m^2) ∧ m > n :=
by
  sorry

end next_perfect_square_l379_379056


namespace who_is_next_to_Denis_l379_379261

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379261


namespace schoolchildren_lineup_l379_379305

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379305


namespace inscribed_radius_correct_l379_379737

def inscribed_radius (a b c : ℝ) : ℝ :=
  1 / ((1 / a) + (1 / b) + (1 / c) + 2 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c))))

theorem inscribed_radius_correct :
  inscribed_radius 5 10 20 ≈ 1.381 := by
  sorry

end inscribed_radius_correct_l379_379737


namespace tax_rate_correct_l379_379878

-- Definitions
def total_spent : ℝ := 45
def tax_free_items_cost : ℝ := 39.7
def sales_tax_rate : ℝ := 0.30
def sales_tax_amount : ℝ := sales_tax_rate * total_spent
def taxable_items_cost_before_tax : ℝ := total_spent - tax_free_items_cost

-- The problem is to find the tax rate on taxable purchases
def tax_rate_on_taxable_purchases (sales_tax_amount taxable_items_cost_before_tax : ℝ) : ℝ := 
  (sales_tax_amount / taxable_items_cost_before_tax) * 100

-- Given the conditions, prove that the tax rate on taxable purchases is approximately 254.717%
theorem tax_rate_correct :
  tax_rate_on_taxable_purchases sales_tax_amount taxable_items_cost_before_tax ≈ 254.717 := 
  by
    sorry

end tax_rate_correct_l379_379878


namespace standing_next_to_Denis_l379_379312

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379312


namespace sum_even_integers_102_to_200_l379_379970

theorem sum_even_integers_102_to_200 : 
  (finset.sum (finset.filter (λ n, n % 2 = 0) (finset.range' 102 200.succ))) = 7550 :=
sorry

end sum_even_integers_102_to_200_l379_379970


namespace schoolchildren_lineup_l379_379303

theorem schoolchildren_lineup :
  ∀ (line : list string),
  (line.length = 5) →
  (line.head = some "Borya") →
  ((∃ i j, i ≠ j ∧ (line.nth i = some "Vera") ∧ (line.nth j = some "Anya") ∧ (i + 1 = j ∨ i = j + 1)) ∧ 
   (∀ i j, (line.nth i = some "Vera") → (line.nth j = some "Gena") → ((i + 1 ≠ j) ∧ (i ≠ j + 1)))) →
  (∀ i j k, (line.nth i = some "Anya") → (line.nth j = some "Borya") → (line.nth k = some "Gena") → 
            ((i + 1 ≠ j) ∧ (i ≠ j + 1) ∧ (i + 1 ≠ k) ∧ (i ≠ k + 1))) →
  ∃ i, (line.nth i = some "Denis") ∧ 
       (((line.nth (i + 1) = some "Anya") ∨ (line.nth (i - 1) = some "Anya")) ∧ 
        ((line.nth (i + 1) = some "Gena") ∨ (line.nth (i - 1) = some "Gena")))
:= by
 sorry

end schoolchildren_lineup_l379_379303


namespace F_50_is_58_over_3_l379_379732

noncomputable def F : ℕ → ℚ
| 1     := 3
| (n+1) := (3 * F n + 1) / 3

theorem F_50_is_58_over_3 : F 50 = 58 / 3 :=
by sorry

end F_50_is_58_over_3_l379_379732


namespace remainder_of_polynomial_division_l379_379338

theorem remainder_of_polynomial_division :
  let f : ℝ → ℝ := λ x, x ^ 11 + 2 in
  let a : ℝ := 1 in
  f a = 3 :=
by
  sorry

end remainder_of_polynomial_division_l379_379338


namespace find_x_5pi_over_4_l379_379026

open Real

theorem find_x_5pi_over_4 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = -sqrt 2) : x = 5 * π / 4 := 
sorry

end find_x_5pi_over_4_l379_379026


namespace number_of_solutions_eq_l379_379002

noncomputable def number_of_solutions (a : ℝ) : ℕ :=
  if a < 1/6 then 0 else
  if a = 1/6 then 1 else
  if a < 1/2 then 2 else
  if a = 1/2 then 3 else
  if a < 2/3 then 4 else
  if a = 2/3 then 3 else 0

theorem number_of_solutions_eq (a : ℝ) :
  ∀ x : ℝ, 
  (sqrt (x + 3) = a * x + 2) ↔
  number_of_solutions a = 
   if a < 1/6 then 0 else
   if a = 1/6 then 1 else
   if a < 1/2 then 2 else
   if a = 1/2 then 3 else
   if a < 2/3 then 4 else
   if a = 2/3 then 3 else 0 :=
by
  sorry -- Proof to be provided

end number_of_solutions_eq_l379_379002


namespace cost_for_2_hours_is_20_l379_379219

noncomputable def cost_up_to_2_hours := 
  let C := (3.5833333333333335 * 9) - (7 * 1.75)
  C

theorem cost_for_2_hours_is_20 :
  let C := (3.5833333333333335 * 9) - (7 * 1.75) in
  C = 20 :=
by
  sorry

end cost_for_2_hours_is_20_l379_379219


namespace pinwheel_area_l379_379333

theorem pinwheel_area
  (grid_size : ℕ)
  (grid_size = 6)
  (center_x = grid_size / 2)
  (center_y = grid_size / 2)
  (triangle_area = (0 + boundary_points / 2 - 1) for each of the four arms)
  : total_area = 4 :=
sorry

end pinwheel_area_l379_379333


namespace eval_expr_l379_379436

-- Declare the imaginary unit i
def i : ℂ := complex.I

-- Declare exponents.
def n1 : ℕ := 14764
def n2 : ℕ := 14765
def n3 : ℕ := 14766
def n4 : ℕ := 14767

-- Enforce the cycles of i's powers.
axiom i_cycle : ∀ (n : ℕ), i ^ (4 * n) = 1 ∧ i ^ (4 * n + 1) = i ∧ i ^ (4 * n + 2) = -1 ∧ i ^ (4 * n + 3) = -i

-- The theorem we need to prove.
theorem eval_expr : (i ^ n1) + (i ^ n2) + (i ^ n3) + (i ^ n4) = 0 := by
  sorry

end eval_expr_l379_379436


namespace total_population_l379_379857

variables (b g t a : ℕ)

-- Given conditions
def condition1 : Prop := b = 4 * g
def condition2 : Prop := g = 8 * t
def condition3 : Prop := t = 2 * a

-- Prove the total number of boys, girls, teachers, and administrators is 83a
theorem total_population : condition1 ∧ condition2 ∧ condition3 → b + g + t + a = 83 * a := 
by
  intro h
  obtain ⟨hb, hgt, hta⟩ := h

  -- Use the conditions to transform the expressions
  rw [hb, hgt, hta]
  sorry   -- Proof is skipped as per instruction

end total_population_l379_379857


namespace probability_at_least_one_die_less_3_l379_379327

-- Definitions
def total_outcomes_dice : ℕ := 64
def outcomes_no_die_less_3 : ℕ := 36
def favorable_outcomes : ℕ := total_outcomes_dice - outcomes_no_die_less_3
def probability : ℚ := favorable_outcomes / total_outcomes_dice

-- Theorem statement
theorem probability_at_least_one_die_less_3 :
  probability = 7 / 16 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_die_less_3_l379_379327


namespace tan_alpha_of_sin2_alpha_add_cos2_alpha_l379_379830

theorem tan_alpha_of_sin2_alpha_add_cos2_alpha (α : ℝ) (hα1 : 0 < α) (hα2 : α < (π / 2))
  (h : sin α ^ 2 + cos (2 * α) = 1 / 4) :
  tan α = sqrt 3 :=
by
  sorry

end tan_alpha_of_sin2_alpha_add_cos2_alpha_l379_379830


namespace circumcenter_coords_l379_379058

-- Define the given points A, B, and C
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-5, 1)
def C : ℝ × ℝ := (3, -5)

-- The target statement to prove
theorem circumcenter_coords :
  ∃ x y : ℝ, (x - 2)^2 + (y - 2)^2 = (x + 5)^2 + (y - 1)^2 ∧
             (x - 2)^2 + (y - 2)^2 = (x - 3)^2 + (y + 5)^2 ∧
             x = -1 ∧ y = -2 :=
by
  sorry

end circumcenter_coords_l379_379058


namespace quadrilateral_not_necessarily_parallelogram_l379_379853

noncomputable def is_convex_quadrilateral (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] 
  [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] : Prop :=
sorry

noncomputable def is_parallelogram (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] 
  [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] : Prop :=
sorry

theorem quadrilateral_not_necessarily_parallelogram 
  (A B C D : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] 
  [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (h1 : dist A B = dist C D) 
  (h2 : ∠ A B = ∠ C D) : 
  ¬ is_parallelogram A B C D :=
sorry

end quadrilateral_not_necessarily_parallelogram_l379_379853


namespace ratio_of_girls_to_boys_l379_379982

-- defining the conditions
variables (students boys girls : ℕ)
variables (total_cups cups_per_boy cups : ℕ)
variables (ratio_girls_boys : ℕ × ℕ)

-- Assuming the initial conditions
def class_conditions : Prop :=
  students = 30 ∧
  boys = 10 ∧
  cups_per_boy = 5 ∧
  total_cups = 90 ∧
  (boys * cups_per_boy) + (girls * cups_per_boy) = total_cups

-- defining the proof problem
theorem ratio_of_girls_to_boys (h : class_conditions students boys girls total_cups cups_per_boy total_cups cups ratio_girls_boys) :
  ratio_girls_boys = (4, 5) :=
sorry

end ratio_of_girls_to_boys_l379_379982


namespace markup_percent_based_on_discounted_price_l379_379701

-- Defining the conditions
def original_price : ℝ := 1
def discount_percent : ℝ := 0.2
def discounted_price : ℝ := original_price * (1 - discount_percent)

-- The proof problem statement
theorem markup_percent_based_on_discounted_price :
  (original_price - discounted_price) / discounted_price = 0.25 :=
sorry

end markup_percent_based_on_discounted_price_l379_379701


namespace exists_multicolored_triangle_l379_379122

theorem exists_multicolored_triangle (n : ℕ) :
  ∀ (G : Type) [Fintype G] [Finite G] (e_chess e_go e_checkers : G → G → Prop),
  (∃ (vertices : Finₓ (3 * n + 1)), true) →
  (∀ v : G, (∃ v_chess : Finₓ n, e_chess v v_chess) ∧
            (∃ v_go : Finₓ n, e_go v v_go) ∧
            (∃ v_checkers : Finₓ n, e_checkers v v_checkers)) →
  (∃ (v1 v2 v3 : G), e_chess v1 v2 ∧ e_go v2 v3 ∧ e_checkers v3 v1) :=
by 
  intros
  simp only []
  -- Proof goes here.
  sorry

end exists_multicolored_triangle_l379_379122


namespace cos_squared_is_continuous_everywhere_l379_379593

noncomputable def cos_squared_continuous (x : ℝ) : Prop :=
  continuous (λ x, Real.cos (x^2))

theorem cos_squared_is_continuous_everywhere : ∀ x : ℝ, cos_squared_continuous x :=
begin
  intro x,
  unfold cos_squared_continuous,
  exact continuous.comp Real.continuous_cos (continuous_pow 2),
end

end cos_squared_is_continuous_everywhere_l379_379593


namespace number_of_elements_in_S_l379_379896

def f (x : ℝ) : ℝ := (x + 8) / x

def f_seq : Nat → (ℝ → ℝ)
| 0     := f
| (n+1) := f ∘ f_seq n

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

def S : Set ℝ := { x | ∃ n : Nat, is_fixed_point (f_seq n) x }

theorem number_of_elements_in_S : S.card = 2 := sorry

end number_of_elements_in_S_l379_379896


namespace possible_numbers_tom_l379_379636

theorem possible_numbers_tom (n : ℕ) (h1 : 180 ∣ n) (h2 : 75 ∣ n) (h3 : 500 < n ∧ n < 2500) : n = 900 ∨ n = 1800 :=
sorry

end possible_numbers_tom_l379_379636


namespace room_number_for_key_36_l379_379212

theorem room_number_for_key_36 :
  ∃ n : ℕ, n ∈ (setOf (λ x, (x % 5 = 3 ∧ x % 7 = 6))) ∧ (1 ≤ n ∧ n ≤ 30) ∧ n = 13 :=
by {
  use 13,
  split,
  { split,
    { intro a,
      simpl in a,
      ring [mod_eq_of_lt (27 % 7) 6] },
    { simp } },
  { exact 13 }
}

end room_number_for_key_36_l379_379212


namespace John_used_16_bulbs_l379_379146

variable (X : ℕ)

theorem John_used_16_bulbs
  (h1 : 40 - X = 2 * 12) :
  X = 16 := 
sorry

end John_used_16_bulbs_l379_379146


namespace odometer_reading_before_trip_l379_379582

-- Define the given conditions
def odometer_reading_lunch : ℝ := 372.0
def miles_traveled : ℝ := 159.7

-- Theorem to prove that the odometer reading before the trip was 212.3 miles
theorem odometer_reading_before_trip : odometer_reading_lunch - miles_traveled = 212.3 := by
  sorry

end odometer_reading_before_trip_l379_379582


namespace sum_even_integers_correct_l379_379968

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end sum_even_integers_correct_l379_379968


namespace part1_solution_set_l379_379775

theorem part1_solution_set (a : ℝ) (x : ℝ) : a = -2 → (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0 ↔ x ≠ -1 :=
by sorry

end part1_solution_set_l379_379775


namespace problem1_problem2_l379_379091

-- Definition of A, B, and C
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≥ 2}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Problem 1: Prove that A ∩ (¬ₙB) = {x | 1 < x < 2}
theorem problem1 : A ∩ {x | x < 2} = {x | 1 < x < 2} :=
sorry

-- Problem 2: Prove that ∀a, (A ∩ C(a) = C(a)) → a ≥ 1
theorem problem2 (a : ℝ) : (A ∩ C(a) = C(a)) → a ≥ 1 :=
sorry

end problem1_problem2_l379_379091


namespace sum_of_angles_l379_379713

theorem sum_of_angles 
    (ABC_isosceles : ∃ (A B C : Type) (angleBAC : ℝ), (AB = AC) ∧ (angleBAC = 25))
    (DEF_isosceles : ∃ (D E F : Type) (angleEDF : ℝ), (DE = DF) ∧ (angleEDF = 40)) 
    (AD_parallel_CE : Prop) : 
    ∃ (angleDAC angleADE : ℝ), angleDAC = 77.5 ∧ angleADE = 70 ∧ (angleDAC + angleADE = 147.5) :=
by {
  sorry
}

end sum_of_angles_l379_379713


namespace probability_of_nickel_l379_379366

noncomputable def value_of_quarter : ℚ := 15.00
noncomputable def value_of_dime : ℚ := 5.00
noncomputable def value_of_nickel : ℚ := 3.75

noncomputable def worth_of_quarter : ℚ := 0.25
noncomputable def worth_of_dime : ℚ := 0.10
noncomputable def worth_of_nickel : ℚ := 0.05

noncomputable def num_quarters : ℕ := (value_of_quarter / worth_of_quarter).toNat
noncomputable def num_dimes : ℕ := (value_of_dime / worth_of_dime).toNat
noncomputable def num_nickels : ℕ := (value_of_nickel / worth_of_nickel).toNat
noncomputable def total_coins : ℕ := num_quarters + num_dimes + num_nickels

theorem probability_of_nickel : (num_nickels : ℚ) / (total_coins : ℚ) = 15 / 37 := 
by
  sorry

end probability_of_nickel_l379_379366


namespace equal_roots_of_quadratic_eq_l379_379513

theorem equal_roots_of_quadratic_eq (n : ℝ) : (∃ x : ℝ, (x^2 - x + n = 0) ∧ (Δ = 0)) ↔ n = 1 / 4 :=
by
  have h₁ : Δ = 0 := by sorry  -- The discriminant condition
  sorry  -- Placeholder for completing the theorem proof

end equal_roots_of_quadratic_eq_l379_379513


namespace find_intervals_of_f_l379_379086

theorem find_intervals_of_f (a b : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = - 1 / 2 * x ^ 2 + 13 / 2) 
  (h_min : ∀ x ∈ set.Icc a b, f a = max_value)
  (h_max : ∀ x ∈ set.Icc a b, f b = min_value)
  : (a = 1 ∧ b = 3) ∨ (a = (-2) - real.sqrt 17 ∧ b = 13 / 4) :=
begin
  sorry,
end

end find_intervals_of_f_l379_379086


namespace nes_sale_price_l379_379994

noncomputable def price_of_nes
    (snes_value : ℝ)
    (tradein_rate : ℝ)
    (cash_given : ℝ)
    (change_received : ℝ)
    (game_value : ℝ) : ℝ :=
  let tradein_credit := snes_value * tradein_rate
  let additional_cost := cash_given - change_received
  let total_cost := tradein_credit + additional_cost
  let nes_price := total_cost - game_value
  nes_price

theorem nes_sale_price 
  (snes_value : ℝ)
  (tradein_rate : ℝ)
  (cash_given : ℝ)
  (change_received : ℝ)
  (game_value : ℝ) :
  snes_value = 150 → tradein_rate = 0.80 → cash_given = 80 → change_received = 10 → game_value = 30 →
  price_of_nes snes_value tradein_rate cash_given change_received game_value = 160 := by
  intros
  sorry

end nes_sale_price_l379_379994


namespace number_of_correct_propositions_is_2_l379_379816

variables (m n l : Line) (α β : Plane)

-- Proposition definitions as per conditions in a)
def proposition_1 : Prop := (m ∥ n ∧ n ⊆ α) → (m ∥ α)
def proposition_2 : Prop := (l ⊥ α ∧ m ⊥ β ∧ l ∥ m) → (α ∥ β)
def proposition_3 : Prop := (m ⊆ α ∧ n ⊆ α ∧ m ∥ β ∧ n ∥ β) → (α ∥ β)
def proposition_4 : Prop := (α ⊥ β ∧ (α ∩ β = m) ∧ n ⊆ β ∧ n ⊥ m) → (n ⊥ α)
def proposition_5 : Prop := (α ∥ β ∧ m ∥ n ∧ m ⊥ α) → (n ⊥ β)

-- Main theorem statement
theorem number_of_correct_propositions_is_2 :
  (proposition_2 m n l α β) ∧ (proposition_5 m n l α β) ∧ 
  (¬ proposition_1 m n l α β) ∧ (¬ proposition_3 m n l α β) ∧ (¬ proposition_4 m n l α β) → 
  (number_of_correct_propositions = 2) :=
sorry

end number_of_correct_propositions_is_2_l379_379816


namespace max_integer_a_l379_379776

theorem max_integer_a :
  ∀ (a: ℤ), (∀ x: ℝ, (a + 1) * x^2 - 2 * x + 3 = 0 → (a = -2 → (-12 * a - 8) ≥ 0)) → (∀ a ≤ -2, a ≠ -1) :=
by
  sorry

end max_integer_a_l379_379776


namespace total_female_officers_l379_379190

theorem total_female_officers (F : ℕ) (percent_on_duty : ℝ) (total_on_duty male_on_duty : ℕ) (h1 : percent_on_duty = 0.65) (h2 : total_on_duty = 475) (h3 : male_on_duty = 315) (h4 : 0.65 * F ≈ 160) :
  F ≈ 246 :=
by 
  sorry

end total_female_officers_l379_379190


namespace modulus_z_111_l379_379898

noncomputable def z : ℕ → ℂ
| 1     := 0
| (n+1) := z n ^ 2 + complex.I

theorem modulus_z_111 : complex.abs (z 111) = real.sqrt 2 := 
by
  sorry

end modulus_z_111_l379_379898


namespace find_solutions_l379_379439

noncomputable def sqrt_exp (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a ^ x)

theorem find_solutions (x : ℝ) :
  sqrt_exp (3 + Real.sqrt 5) x ^ 2 + sqrt_exp (3 - Real.sqrt 5) x ^ 2 = 18 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end find_solutions_l379_379439


namespace min_f_in_interval_l379_379954

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (π / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem min_f_in_interval : 
  ∀ x, π / 4 ≤ x ∧ x ≤ π / 2 → f(x) ≥ 2 :=
by
  sorry

end min_f_in_interval_l379_379954


namespace custom_bookcase_length_l379_379184

-- Define the conditions given in the problem
def bookcase_length_inches : ℝ := 48
def shelving_unit_length_meters : ℝ := 1.2
def inches_per_meter : ℝ := 39.37
def extra_shelf_length_inches : ℝ := 36
def inches_per_foot : ℝ := 12

-- Convert lengths and combine them to prove the final length
def final_custom_bookcase_length_feet : ℝ :=
  (bookcase_length_inches + shelving_unit_length_meters * inches_per_meter) / inches_per_foot
    + (extra_shelf_length_inches / inches_per_foot)

-- Statement of the problem
theorem custom_bookcase_length : 
  final_custom_bookcase_length_feet = 10.937 :=
sorry

end custom_bookcase_length_l379_379184


namespace probability_at_least_one_die_less_than_3_l379_379329

theorem probability_at_least_one_die_less_than_3 :
  let total_outcomes := 8 * 8,
      favorable_outcomes := total_outcomes - (6 * 6)
  in (favorable_outcomes / total_outcomes : ℚ) = 7 / 16 := by
  sorry

end probability_at_least_one_die_less_than_3_l379_379329


namespace solve_y_l379_379438

theorem solve_y : ∃ y : ℝ, (y^2 - 3*y - 10) / (y + 2) + (4*y^2 + 17*y - 15) / (4*y - 1) = 5 ∧ y = -2.5 :=
begin
  sorry
end

end solve_y_l379_379438


namespace sum_even_integers_correct_l379_379966

variable (S1 S2 : ℕ)

-- Definition: The sum of the first 50 positive even integers
def sum_first_50_even_integers : ℕ := 2550

-- Definition: The sum of even integers from 102 to 200 inclusive
def sum_even_integers_from_102_to_200 : ℕ := 7550

-- Condition: The sum of the first 50 positive even integers is 2550
axiom sum_first_50_even_integers_given : S1 = sum_first_50_even_integers

-- Problem statement: Prove that the sum of even integers from 102 to 200 inclusive is 7550
theorem sum_even_integers_correct :
  S1 = sum_first_50_even_integers →
  S2 = sum_even_integers_from_102_to_200 →
  S2 = 7550 :=
by
  intros h1 h2
  rw [h2]
  sorry

end sum_even_integers_correct_l379_379966


namespace rectangle_minimum_area_l379_379703

theorem rectangle_minimum_area (l w : ℤ) (h1 : l + w = 60) (h2 : l > 0) (h3 : w > 0) : 
  ∃ A, (A = l * w) ∧ (∀ l' w', l' + w' = 60 ∧ l' > 0 ∧ w' > 0 → l' * w' ≥ A) ∧ A = 59 := 
by { -- sorry: missing proof }

end rectangle_minimum_area_l379_379703


namespace bayes_theorem_l379_379606

noncomputable def probability_D : ℚ := 1 / 250
noncomputable def probability_not_D : ℚ := 1 - probability_D
noncomputable def probability_T_given_D : ℚ := 1
noncomputable def probability_T_given_not_D : ℚ := 0.03 

theorem bayes_theorem :
  let q := (probability_T_given_D * probability_D) /
           ((probability_T_given_D * probability_D) + (probability_T_given_not_D * probability_not_D))
  in q = 1 / 10 :=
by
  sorry

end bayes_theorem_l379_379606


namespace adjacent_book_left_of_middle_l379_379350

theorem adjacent_book_left_of_middle (
  n : Nat,
  p_diff : Int,
  rightmost_book_price : Int,
  book_prices : List Int) 
  (h1 : n = 2 * 431)
  (h2 : p_diff = 2)
  (h3 : ∀ i, i < book_prices.length → book_prices.get i.succ = book_prices.get i + p_diff)
  (h4 : rightmost_book_price = List.last book_prices 0)
  (h5 : ∃ m, m = (List.length book_prices / 2) ∧ (book_prices.get m + book_prices.get (m - 1) = rightmost_book_price)) :
  ∃ m, book_prices.get (m - 1) = book_prices.get m - 2 :=
sorry

end adjacent_book_left_of_middle_l379_379350


namespace sin_alpha_value_l379_379806

theorem sin_alpha_value (x y : ℝ) (h₁ : x = -√3) (h₂ : y = 1) (h₃ : sqrt (x^2 + y^2) = 2) : 
  sin (α : ℝ) = y / (sqrt (x^2 + y^2)) :=
by
  sorry

end sin_alpha_value_l379_379806


namespace elements_positive_l379_379168

theorem elements_positive {n : ℤ} (E : Set ℝ) (h1 : E.card = 2 * n + 1)
  (h2 : ∀ S T : Finset ℝ, S.card = n + 1 → T.card = n → S ∪ T = E.toFinset → S.sum id > T.sum id) :
  ∀ x ∈ E, 0 < x :=
begin
  sorry
end

end elements_positive_l379_379168


namespace triangle_DEF_side_length_d_l379_379874

variable {D E : ℝ} (b c : ℝ := 7) (cos_D_E : ℝ := (55/64))

theorem triangle_DEF_side_length_d (d : ℝ) 
  (h_b : b = 7)
  (h_c : c = 8)
  (h_cos_D_E : cos (D - E) = 55 / 64) :
  d = Real.sqrt 105 := 
by
  sorry

end triangle_DEF_side_length_d_l379_379874


namespace solve_for_x_l379_379207

theorem solve_for_x : ∃ x : ℝ, 3^x * 9^x = 27^(x - 4) := 
by
  use -6
  sorry

end solve_for_x_l379_379207


namespace quadratic_solution_1_equation_solution_2_l379_379601

theorem quadratic_solution_1 
  (a b c: ℝ) (a_eq: a = 1) (b_eq: b = -4) (c_eq: c = -8)
  (discriminant: ℝ) (discriminant_eq: discriminant = b^2 - 4 * a * c)
  (x1 x2: ℝ) 
  (x1_eq: x1 = (4 + Real.sqrt(48)) / 2) 
  (x2_eq: x2 = (4 - Real.sqrt(48)) / 2) :
  (a = 1) → (b = -4) → (c = -8) → (discriminant = 48) → 
  (x1 = 2 + 2 * Real.sqrt(3)) → (x2 = 2 - 2 * Real.sqrt(3)) :=
by
  sorry

theorem equation_solution_2 
  (x: ℝ) 
  (h: (x - 2)^2 = 2 * x - 4) :
  (x = 2 ∨ x = 4) :=
by
  sorry

end quadratic_solution_1_equation_solution_2_l379_379601


namespace external_angle_bisector_inequality_l379_379585

noncomputable def triangle (A B C : Type) := 
A ≠ B ∧ B ≠ C ∧ A ≠ C

theorem external_angle_bisector_inequality
  (A B C M : Type) 
  [triangle A B C] 
  (hM : M ≠ C) 
  (h_ext : ∃ (L : Type), on_external_angle_bisector(C, A, B, M, L)) :
  (MA + MB > CA + CB) :=
sorry

end external_angle_bisector_inequality_l379_379585


namespace square_side_length_rectangle_perimeter_l379_379096

noncomputable theory  -- necessary for handling real numbers

-- Condition: The lengths of the two sides AB and BC of rectangle ABCD are two real roots of the given quadratic equation.
-- Question 1
def side_length_square (m : ℝ) : Prop :=
  ∃ (x : ℝ), x^2 - m * x + (m / 2) - (1 / 4) = 0 ∧ m = 1 ∧ x = 1 / 2 

-- Question 2
def perimeter_rectangle (AB : ℝ) (perimeter : ℝ) : Prop :=
  ∃ (BC m : ℝ), AB = 2 ∧ (2^2 - 2 * m + (m / 2) - (1 / 4) = 0) ∧
  (BC^2 - m * BC + (m / 2) - (1 / 4) = 0) ∧
  perimeter = 2 * (AB + BC) ∧ BC = 1 / 2

-- Proof statements combining the conditions and the conclusions:
theorem square_side_length : side_length_square 1 :=
sorry

theorem rectangle_perimeter : perimeter_rectangle 2 5 :=
sorry

end square_side_length_rectangle_perimeter_l379_379096


namespace length_down_correct_l379_379365

variable (rate_up rate_down time_up time_down length_down : ℕ)
variable (h1 : rate_up = 8)
variable (h2 : time_up = 2)
variable (h3 : time_down = time_up)
variable (h4 : rate_down = (3 / 2) * rate_up)
variable (h5 : length_down = rate_down * time_down)

theorem length_down_correct : length_down = 24 := by
  sorry

end length_down_correct_l379_379365


namespace parabola_vertex_coordinates_l379_379939

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), y = -3 * (x + 1)^2 - 2 → (x, y) = (-1, -2) := by
  sorry

end parabola_vertex_coordinates_l379_379939


namespace who_is_next_to_denis_l379_379285

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379285


namespace find_a_from_area_l379_379533

theorem find_a_from_area :
  ∃ a : ℝ, 0 < a ∧ (∫ x in 0..a, real.sqrt x) = 2 / 3 ∧ a = 1 :=
by
  sorry

end find_a_from_area_l379_379533


namespace degrees_multiplication_proof_l379_379683

/-- Convert a measurement given in degrees and minutes to purely degrees. -/
def degrees (d : Int) (m : Int) : ℚ := d + m / 60

/-- Given conditions: -/
def lhs : ℚ := degrees 21 17
def rhs : ℚ := degrees 106 25

/-- The theorem to prove the mathematical problem. -/
theorem degrees_multiplication_proof : lhs * 5 = rhs := sorry

end degrees_multiplication_proof_l379_379683


namespace floor_neg_sqrt_eval_l379_379017

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end floor_neg_sqrt_eval_l379_379017


namespace parabola_focus_l379_379445

theorem parabola_focus :
  ∀ y, x = - (1 / 16) * y^2 → focus x y = (-4, 0) :=
by
  sorry

end parabola_focus_l379_379445


namespace product_of_numbers_l379_379981

theorem product_of_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 150)
  (h2 : 7 * x = n)
  (h3 : y - 10 = n)
  (h4 : z + 10 = n) : x * y * z = 48000 := 
by 
  sorry

end product_of_numbers_l379_379981


namespace distance_between_X_Y_l379_379189

-- Definitions from conditions
def yolanda_rate : ℝ := 3 -- miles per hour
def bob_rate : ℝ := 4 -- miles per hour
def bob_distance : ℝ := 4 -- miles
def time_difference : ℝ := 1 -- hour

-- Lean statement to prove the problem
theorem distance_between_X_Y : 
  let yolanda_time := (bob_distance / bob_rate) + time_difference in
  let yolanda_distance := yolanda_rate * yolanda_time in
  let x_y_distance := yolanda_distance + bob_distance in
  x_y_distance = 10 :=
by
  sorry

end distance_between_X_Y_l379_379189


namespace prob_not_snowing_l379_379960

theorem prob_not_snowing (P_snowing : ℚ) (h : P_snowing = 1/4) : 1 - P_snowing = 3/4 := by
  sorry

end prob_not_snowing_l379_379960


namespace max_monthly_profit_l379_379480

/-- Given conditions about the cost price, initial selling price, sales volume, and the effect of price changes,
    this theorem proves that the maximum monthly profit of 2400 yuan is achieved at the selling prices of 55 or 56 yuan per unit.
 -/
theorem max_monthly_profit :
  ∀ (x : ℕ),
  0 < x → x ≤ 15 →
  let y := -10 * x^2 + 170 * x + 2100 in
  (x = 5 ∨ x = 6) → y = 2400 := by
  sorry

end max_monthly_profit_l379_379480


namespace dacid_weighted_average_l379_379421

theorem dacid_weighted_average :
  let english := 96
  let mathematics := 95
  let physics := 82
  let chemistry := 87
  let biology := 92
  let weight_english := 0.20
  let weight_mathematics := 0.25
  let weight_physics := 0.15
  let weight_chemistry := 0.25
  let weight_biology := 0.15
  (english * weight_english) + (mathematics * weight_mathematics) +
  (physics * weight_physics) + (chemistry * weight_chemistry) +
  (biology * weight_biology) = 90.8 :=
by
  sorry

end dacid_weighted_average_l379_379421


namespace denis_neighbors_l379_379296

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

end denis_neighbors_l379_379296


namespace triangle_with_given_sides_l379_379386

theorem triangle_with_given_sides (y : ℤ) (hy_pos : 0 < y) 
  (h_tri : 6 + (y + 1) > y^2 + 2y + 3 ∧ (y + 1) + (y^2 + 2y + 3) > 6) : y = 2 := 
by sorry

end triangle_with_given_sides_l379_379386


namespace identify_pyramid_scheme_l379_379649

-- Definitions based on the given conditions
def high_return : Prop := offers_significantly_higher_than_average_returns
def lack_of_information : Prop := lack_of_complete_information_about_company
def aggressive_advertising : Prop := aggressive_advertising_occurs

-- Defining the predicate 
def is_pyramid_scheme (all_conditions : Prop) : Prop :=
  offers_significantly_higher_than_average_returns ∧
  lack_of_complete_information_about_company ∧
  aggressive_advertising_occurs

-- The main theorem to prove
theorem identify_pyramid_scheme :
  (high_return ∧ lack_of_information ∧ aggressive_advertising) → is_pyramid_scheme (high_return ∧ lack_of_information ∧ aggressive_advertising) :=
by
  intro h
  exact h

end identify_pyramid_scheme_l379_379649


namespace probability_same_number_l379_379389

noncomputable def multiple_of_20 : Set ℕ := {n | n < 300 ∧ (20 ∣ n)}
noncomputable def multiple_of_30 : Set ℕ := {n | n < 300 ∧ (30 ∣ n)}

theorem probability_same_number :
  (final_prob : ℚ) → (final_prob = (5 : ℚ) / 150) :=
begin
  let count_multiples_60 := 5,
  let count_combinations := 150,
  let final_prob := (count_multiples_60 : ℚ) / count_combinations,
  sorry
end

end probability_same_number_l379_379389


namespace denis_neighbors_l379_379292

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379292


namespace corresponding_side_of_larger_triangle_l379_379612

theorem corresponding_side_of_larger_triangle
  (diff_area : ℝ) (ratio_area : ℝ) (side_smaller : ℝ) (perimeter_smaller : ℝ) (int_area_sum : ℝ)
  (h1 : diff_area = 50) (h2 : ratio_area = 9) (h3 : side_smaller = 5) (h4 : perimeter_smaller = 15)
  (h5 : ∃ (A_S : ℕ), (A_L A_S : ℝ) × (A_S = int_area_sum))
  :
  ∃ (s_L : ℝ), (s_L = 15) := 
by
  sorry

end corresponding_side_of_larger_triangle_l379_379612


namespace tenth_month_payment_full_loan_total_expenditure_l379_379199

noncomputable def initial_payment : ℝ := 150
noncomputable def monthly_payment : ℝ := 50
noncomputable def interest_rate : ℝ := 0.01
noncomputable def initial_debt : ℝ := 1150 - initial_payment

noncomputable def payment_n (n : ℕ) : ℝ :=
  60 - 0.5 * (n - 1)

noncomputable def total_expenditure (monthly_payments_count : ℕ) : ℝ :=
  initial_payment + ∑ n in finset.range monthly_payments_count, payment_n (n+1)

theorem tenth_month_payment : payment_n 10 = 55.5 := sorry
theorem full_loan_total_expenditure : total_expenditure 20 = 1255 := sorry

end tenth_month_payment_full_loan_total_expenditure_l379_379199


namespace factorize_expression1_factorize_expression2_l379_379022

variable {R : Type*} [CommRing R]

theorem factorize_expression1 (x y : R) : x^2 + 2 * x + 1 - y^2 = (x + y + 1) * (x - y + 1) :=
  sorry

theorem factorize_expression2 (m n p : R) : m^2 - n^2 - 2 * n * p - p^2 = (m + n + p) * (m - n - p) :=
  sorry

end factorize_expression1_factorize_expression2_l379_379022


namespace quadratic_graph_passes_through_point_l379_379844

theorem quadratic_graph_passes_through_point
  (a b c : ℝ)
  (h : a - b + c = 0) :
  ∃ x y : ℝ, x = -1 ∧ y = 0 ∧ y = a * x^2 + b * x + c :=
by
  use [-1, 0]
  simp
  exact h

end quadratic_graph_passes_through_point_l379_379844


namespace fourth_roots_of_neg_16_l379_379767

theorem fourth_roots_of_neg_16 : 
  { z : ℂ | z^4 = -16 } = { sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I, 
                            sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I } :=
by
  sorry

end fourth_roots_of_neg_16_l379_379767


namespace angle_RPS_is_27_l379_379539

theorem angle_RPS_is_27 (PQ BP PR QS QS PSQ QPRS : ℝ) :
  PQ + PSQ + QS = 180 ∧ 
  QS = 48 ∧ 
  PSQ = 38 ∧ 
  QPRS = 67
  → (QS - QPRS = 27) := 
by {
  sorry
}

end angle_RPS_is_27_l379_379539


namespace esther_average_speed_l379_379745

variable (S : ℝ) -- Average speed in the morning in miles per hour
variable (D : ℝ) := 18 -- Distance to work in miles
variable (E : ℝ) := 30 -- Average speed in the evening in miles per hour
variable (T : ℝ) := 1 -- Total commuting time in hours

theorem esther_average_speed :
  (D / S + D / E = T) → S = 45 :=
by
  intro h
  sorry

end esther_average_speed_l379_379745


namespace rhombus_diagonal_length_l379_379940

-- Definitions of given conditions
def d1 : ℝ := 10
def Area : ℝ := 60

-- Proof of desired condition
theorem rhombus_diagonal_length (d2 : ℝ) : 
  (Area = d1 * d2 / 2) → d2 = 12 :=
by
  sorry

end rhombus_diagonal_length_l379_379940


namespace smallest_gcd_six_l379_379237

theorem smallest_gcd_six (x : ℕ) (hx1 : 70 ≤ x) (hx2 : x ≤ 90) (hx3 : Nat.gcd 24 x = 6) : x = 78 :=
by
  sorry

end smallest_gcd_six_l379_379237


namespace correct_understanding_of_meiosis_and_fertilization_l379_379341

-- Definitions for the conditions
def ConditionA : Prop := ∀ (offspring : Type) (parent1 parent2 : Type), offspring obtains half DNA from each parent
def ConditionB : Prop := ∀ (sperm egg : Type), cell membrane fluidity is the material basis for recognition
def ConditionC : Prop := ∀ (sperm egg : Type), fusion of sperm and egg is merging of cytoplasm
def ConditionD : Prop := ∀ (offspring parent1 parent2 : Type), offspring have different genetic combinations from their parents

-- The statement to prove
theorem correct_understanding_of_meiosis_and_fertilization :
  ¬ConditionA ∧ ¬ConditionB ∧ ¬ConditionC ∧ ConditionD :=
by
  sorry

end correct_understanding_of_meiosis_and_fertilization_l379_379341


namespace count_multiples_of_7_ending_in_3_l379_379500

theorem count_multiples_of_7_ending_in_3 :
  let numbers := { n : ℕ | n > 0 ∧ (n * 7) < 1000 ∧ ((n * 7) % 10) = 3 } in
  set.card numbers = 14 :=
sorry

end count_multiples_of_7_ending_in_3_l379_379500


namespace arctan_tan_computation_l379_379415

theorem arctan_tan_computation :
  ∃ θ : ℝ, θ = 65 ∧ ∃ ϕ : ℝ, ϕ = 40 ∧ 0 ≤ θ ∧ θ ≤ 180 ∧ 0 ≤ ϕ ∧ ϕ ≤ 180 ∧
  (∀ degrees : ℝ, arctan (tan θ - 2 * tan ϕ) = 25) :=
by
  sorry

end arctan_tan_computation_l379_379415


namespace derivative_of_f_l379_379752

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.sqrt (1 - Real.exp x))

theorem derivative_of_f (x : ℝ) (h : x ≤ 0) : 
  Real.deriv f x = -Real.exp (x / 2) / (2 * Real.sqrt (1 - Real.exp x)) :=
sorry

end derivative_of_f_l379_379752


namespace new_bag_specific_color_jellybeans_l379_379986

def average_jellybeans_per_bag (before new_bag: ℕ) (initial_avg new_avg: ℝ) (total_bags_before: ℕ) (added_bags: ℕ) : Prop :=
  initial_avg = 117 ∧ 
  new_avg = initial_avg + 7 ∧ 
  total_bags_before = 34 ∧ 
  added_bags = 1 ∧ 
  before = initial_avg * total_bags_before ∧
  new_bag = new_avg * (total_bags_before + added_bags) - before

def average_jellybeans_per_color (before: ℕ) (colors: ℕ) (total_bags: ℕ) : Prop :=
  colors = 5 ∧ 
  total_bags = 34 ∧ 
  before = 3978 ∧ 
  before / (colors * total_bags) = 23.4 

def specific_color_jellybeans (average_per_color: ℝ): ℝ :=
  2 * average_per_color

theorem new_bag_specific_color_jellybeans :
  ∀ (initial_avg new_avg: ℝ) (total_bags_before added_bags: ℕ) (before: ℕ) (colors: ℕ) (average_color: ℝ),
  initial_avg = 117 →
  new_avg = initial_avg + 7 →
  total_bags_before = 34 →
  added_bags = 1 →
  before = 3978 →
  before / (colors * total_bags_before) = 23.4 →
  specific_color_jellybeans (before / (colors * total_bags_before)) = 2 * 23.4 →
  specific_color_jellybeans (before / (colors * total_bags_before)) = 47 :=
by {
  intros,
  sorry
}

end new_bag_specific_color_jellybeans_l379_379986


namespace prob_neither_perfect_square_nor_cube_nor_prime_l379_379959

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def probability_neither_perfect_square_nor_cube_nor_prime : ℚ :=
  17 / 25

theorem prob_neither_perfect_square_nor_cube_nor_prime :
  (∑ n in (Finset.range 200).filter (λ n, ¬ (is_perfect_square (n+1) ∨ is_perfect_cube (n+1) ∨ is_prime (n+1))), 1) / 200 = probability_neither_perfect_square_nor_cube_nor_prime :=
begin
  sorry
end

end prob_neither_perfect_square_nor_cube_nor_prime_l379_379959


namespace normal_distribution_symmetry_l379_379376

noncomputable def normal_probability_symmetry (X : ℝ → ℝ) (mean : ℝ) : Prop :=
  ∀ (a b : ℝ), (mean - a < X) ∧ (X < mean - b) → 
  P(a < X < b) = P(mean + a < X < mean + b)

theorem normal_distribution_symmetry :
  ∀ (X : ℝ → ℝ) (mean : ℝ) (a b : ℝ),
  (X follows a normal distribution ∧ mean = 500 ∧ P(400 < X < 450) = 0.3) →
  P(550 < X < 600) = 0.3 :=
sorry

end normal_distribution_symmetry_l379_379376


namespace roots_of_z4_plus_16_eq_0_l379_379761

noncomputable def roots_of_quartic_eq : Set ℂ :=
  { z | z^4 + 16 = 0 }

theorem roots_of_z4_plus_16_eq_0 :
  roots_of_quartic_eq = { z | z = complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 - complex.I * complex.sqrt 2 ∨
                             z = complex.sqrt 2 - complex.I * complex.sqrt 2 } :=
by
  sorry

end roots_of_z4_plus_16_eq_0_l379_379761


namespace next_to_Denis_l379_379271

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379271


namespace who_is_next_to_Denis_l379_379260

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379260


namespace solve_garden_width_l379_379704

noncomputable def garden_width_problem (w l : ℕ) :=
  (w + l = 30) ∧ (w * l = 200) ∧ (l = w + 8) → w = 11

theorem solve_garden_width (w l : ℕ) : garden_width_problem w l :=
by
  intro h
  -- Omitting the actual proof
  sorry

end solve_garden_width_l379_379704


namespace product_expression_simplification_l379_379157

noncomputable def theta := 2 * Real.pi / 2015

def product_expr := ∏ k in finset.range 1440, (Real.cos (2^k * theta) - 1 / 2)

theorem product_expression_simplification : ∃ (a : ℕ) (b : ℤ), b % 2 = 1 ∧ product_expr = b / 2^a ∧ a + b = 1441 := 
sorry

end product_expression_simplification_l379_379157


namespace point_inside_circle_l379_379003

-- Define the parametric equations of the circle
def parametric_x (θ : ℝ) : ℝ := -1 + 8 * Real.cos θ
def parametric_y (θ : ℝ) : ℝ := 8 * Real.sin θ

-- Define the circle centered at (-1, 0) with radius 8
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 64

-- Define the point in question
def point := (1, 2)

-- Statement to prove
theorem point_inside_circle : circle_equation point.1 point.2 → (1 + 1)^2 + 2^2 < 64 :=
by simp [circle_equation, point]; exact (4 + 4 < 64)

end point_inside_circle_l379_379003


namespace football_preference_related_to_gender_prob_dist_and_expectation_X_l379_379608

-- Definition of conditions
def total_students : ℕ := 100
def male_students : ℕ := 60
def female_students : ℕ := 40
def male_not_enjoy : ℕ := 10
def female_enjoy_fraction : ℚ := 1 / 4

def alpha : ℚ := 0.001
def chi_squared (a b c d n : ℕ) : ℚ := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))
def chi_squared_critical_value : ℚ := 10.828

-- Part 1: Independence test problem statement
theorem football_preference_related_to_gender :
  let a := 50 in
  let b := 10 in
  let c := 10 in
  let d := 30 in
  let n := 100 in
  chi_squared a b c d n > chi_squared_critical_value :=
by
  sorry -- Proof of chi-squared calculation

-- Conditions for the probability problem
def total_selected_students : ℕ := 8
def male_selected_students : ℕ := 2
def female_selected_students : ℕ := 6

-- Part 2: Probability distribution and expectation problem statement
theorem prob_dist_and_expectation_X :
  let X_vals := [0, 1, 2].to_finset in
  let P : ℕ → ℚ := λ x, 
    match x with
    | 0 => 15 / 28
    | 1 => 3 / 7
    | 2 => 1 / 28
    | _ => 0
    end in
  let E_X : ℚ := 1 / 2 in

  ∀ x ∈ X_vals, P x ∈ {15 / 28, 3 / 7, 1 / 28} ∧
  ∑ x in X_vals, P x = 1 ∧
  ∑ x in X_vals, x * P x = E_X :=
by
  sorry -- Proof of probability distribution and expectation

end football_preference_related_to_gender_prob_dist_and_expectation_X_l379_379608


namespace prime_divisors_of_630_l379_379501

open Nat

def is_prime_divisor (n p : ℕ) : Prop :=
  (p ∣ n) ∧ Prime p

theorem prime_divisors_of_630 :
  let n := 630
  let prime_divisors := {p : ℕ | is_prime_divisor n p}
  (card prime_divisors = 4) ∧ (∑ p in prime_divisors, p = 17) :=
by
  sorry

end prime_divisors_of_630_l379_379501


namespace pages_remaining_l379_379955

def total_pages : ℕ := 120
def science_project_pages : ℕ := (25 * total_pages) / 100
def math_homework_pages : ℕ := 10
def total_used_pages : ℕ := science_project_pages + math_homework_pages
def remaining_pages : ℕ := total_pages - total_used_pages

theorem pages_remaining : remaining_pages = 80 := by
  sorry

end pages_remaining_l379_379955


namespace problem_1_problem_2_l379_379858

variable {A B C : ℝ}
variable {O : Type*}

noncomputable def acute_triangle := ∀ (A B C : ℝ), (0 < A ∧ A < π/2) ∧ (0 < B ∧ B < π/2) ∧ (0 < C ∧ C < π/2) 

theorem problem_1 (h : acute_triangle A B C) 
  (equation : (sin B ^ 2 + sin C ^ 2 - sin A ^ 2) * tan A = sin B * sin C) : 
  A = π / 6 ∧ (π / 3 < B ∧ B < π / 2) := 
sorry

theorem problem_2 (h : acute_triangle A B C) (O : Type*) 
  (dot_product_eqn : (∃ (OB OC : O → ℝ), OB • OC = 1 / 2)) : 
  ∃ (OA AB AC : O → ℝ), range (λ x, OA • (AB + AC)) = [-sqrt(3) - 2, - 7/2) := 
sorry

end problem_1_problem_2_l379_379858


namespace probability_of_odd_divisor_of_24_factorial_l379_379240

-- Define the prime factorization result of 24!
def prime_factorization_of_24_factorial := (2, 22) :: (3, 10) :: (5, 4) :: (7, 3) :: (11, 2) :: (13, 1) :: (17, 1) :: (19, 1) :: (23, 1) :: []

-- State the problem
theorem probability_of_odd_divisor_of_24_factorial : 
  let total_factors := (22 + 1) * (10 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1) ^ 5,
      odd_factors := (10 + 1) * (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1) ^ 5
  in total_factors ≠ 0 → (odd_factors / total_factors = 1 / 23) :=
by
  sorry

end probability_of_odd_divisor_of_24_factorial_l379_379240


namespace orthocenter_distance_equal_l379_379066

theorem orthocenter_distance_equal {A B C A1 B1 C1 O : Point}
  (hA_on_B1C1 : A ∈ line_segment B1 C1)
  (hB_on_C1A1 : B ∈ line_segment C1 A1)
  (hC_on_A1B1 : C ∈ line_segment A1 B1)
  (h_angle_ABC_eq_A1B1C1 : ∠ B A C = ∠ B1 A1 C1)
  (h_angle_BCA_eq_B1C1A1 : ∠ C B A = ∠ C1 B1 A1)
  (h_angle_CAB_eq_C1A1B1 : ∠ A C B = ∠ A1 C1 B1)
  (H H1 : Point)
  (H_orthocenter_ABC : is_orthocenter H A B C)
  (H1_orthocenter_A1B1C1 : is_orthocenter H1 A1 B1 C1)
  (O_circumcenter_ABC : is_circumcenter O A B C) :
  distance O H = distance O H1 :=
sorry

end orthocenter_distance_equal_l379_379066


namespace correct_calculation_l379_379659

variable (a b : ℝ)

theorem correct_calculation : (-a^3)^2 = a^6 := 
by 
  sorry

end correct_calculation_l379_379659


namespace fraction_area_rectangle_l379_379185

def point (x y : ℕ) : ℕ × ℕ := (x, y)

def is_rectangle_formed (p1 p2 p3 p4 : ℕ × ℕ) : Prop :=
  p1 = point 1 1 ∧ p2 = point 1 3 ∧ p3 = point 3 3 ∧ p4 = point 3 1

theorem fraction_area_rectangle (p1 p2 p3 p4 : ℕ × ℕ) (h : is_rectangle_formed p1 p2 p3 p4) : 
  (let larger_square_area := 4 * 4 in
   let rectangle_area := 2 * 2 in
   let fraction := rectangle_area / larger_square_area in
   fraction = 1 / 4) := 
by
  sorry

end fraction_area_rectangle_l379_379185


namespace least_time_for_horses_to_meet_l379_379629

-- Define the first 7 prime numbers
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

-- Define a function to get the k-th prime number
def kth_prime (k : ℕ) : ℕ := primes.getD (k - 1) 2 -- default to 2 if out of bounds

-- Define the condition function to be true if any 4's LCM is 210
def condition_to_check : ℕ → List ℕ → Bool
| 0, _ => false
| (n+1), lst => lst.length == 4 ∧ lst.foldl Nat.lcm 1 = 210 || condition_to_check n (lst.eraseIdx n)

-- Lean statement equivalent to the problem
theorem least_time_for_horses_to_meet : ∃ T > 0, ∃ s ⊆ {1, 2, 3, 4, 5, 6, 7}, s.size = 4 ∧ Nat.lcm_list (s.map kth_prime) = 210 :=
by
  sorry

end least_time_for_horses_to_meet_l379_379629


namespace nine_pow_l379_379504

theorem nine_pow (y : ℝ) (h : 9^(3 * y) = 729) : 9^(3 * y - 2) = 9 := 
sorry

end nine_pow_l379_379504


namespace value_of_x_l379_379035

theorem value_of_x :
  let percent (p : Float) (n : Float) := (p / 100.0) * n in
  let part1 := percent 47 1442 in
  let part2 := percent 36 1412 in
  let result := (part1 - part2) + 66.0 in
  result = 235.42 :=
by
  -- Definitions from conditions
  let percent (p : Float) (n : Float) := (p / 100.0) * n 
  let part1 := percent 47 1442
  let part2 := percent 36 1412
  let result := (part1 - part2) + 66.0

  -- Proof skipped
  sorry

end value_of_x_l379_379035


namespace restore_axes_and_unit_length_l379_379740

theorem restore_axes_and_unit_length (a b: ℝ) :
  (∀ x: ℝ, y = x^2) →
  (using_compass_and_ruler: true) →
  (exists x_eq_one_sq : ∃ x, y = x^2 ∧ (x, y) = (1, 1)) →
  xOy_axes_and_unit_length_restored: true :=
by
  sorry

end restore_axes_and_unit_length_l379_379740


namespace constant_term_binomial_l379_379538

noncomputable def binomial_expr (x : ℝ) := (x^(1/3) - 2 / x)^8

-- Define a function to find the sum of binomial coefficients
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

-- Main theorem
theorem constant_term_binomial (n : ℕ) (x : ℝ) (h : 2^n = 256) : 
  ∃ k : ℕ, binomial_expr x = 112 := 
begin
  sorry
end

end constant_term_binomial_l379_379538


namespace yellow_peaches_l379_379630

theorem yellow_peaches (red green total : ℕ) (h_red : red = 7) (h_green : green = 8) (h_total : total = 30) : total - (red + green) = 15 :=
by
  rw [h_red, h_green, h_total]
  simp
  -- sorry

end yellow_peaches_l379_379630


namespace fourth_roots_of_neg_16_l379_379770

theorem fourth_roots_of_neg_16 : 
  { z : ℂ | z^4 = -16 } = { sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) + sqrt (2 : ℂ) * complex.I, 
                            - sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I, 
                            sqrt (2 : ℂ) - sqrt (2 : ℂ) * complex.I } :=
by
  sorry

end fourth_roots_of_neg_16_l379_379770


namespace floor_of_neg_sqrt_frac_l379_379011

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end floor_of_neg_sqrt_frac_l379_379011


namespace find_r_prove_inequality_l379_379247

-- Define the initial geometric sequence properties and conditions
def geometric_seq_sum (n : ℕ) (b r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), S n = b ^ n + r

-- Define the proof that r = -1
theorem find_r (b : ℝ) (h₁ : 0 < b) (h₂ : b ≠ 1) (Sn : ℕ → ℝ) 
  (h : geometric_seq_sum ℕ b r Sn) : r = -1 :=
  sorry

-- Define the sequence b_n and the inequality to be proven
def b_n (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * (Real.logb 2 (a n) + 1)

def prod_inequality (n : ℕ) (b : ℕ → ℝ) : Prop :=
  ∏ i in finset.range n, (b i + 1) / b i > Real.sqrt (n + 1)

-- Prove the inequality holds for geometric sequence conditions with b = 2
theorem prove_inequality (a : ℕ → ℝ) (n : ℕ) 
  (h₁ : ∀ n, a n = 2 ^ (n - 1)) 
  (h₂ : ∀ n, b_n n a = 2 * n) : 
  ∀ n, prod_inequality n (fun n => b_n n a) :=
  sorry

end find_r_prove_inequality_l379_379247


namespace find_curves_p_eq_pm_pn_l379_379817

-- Definitions of points M and N
def M : ℝ × ℝ := (1, 5/4)
def N : ℝ × ℝ := (-4, -5/4)

-- Define the four curves as subsets of ℝ × ℝ
def curve1 (p : ℝ × ℝ) : Prop := 4 * p.1 + 2 * p.2 - 1 = 0
def curve2 (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 3
def curve3 (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 / 4 = 1
def curve4 (p : ℝ × ℝ) : Prop := p.1^2 - p.2^2 / 4 = 1

-- Statement that the curves curve2 and curve4 meet the condition described
theorem find_curves_p_eq_pm_pn :
  { curve2, curve4 } = { P : (ℝ × ℝ) → Prop | (∃ p : ℝ × ℝ, P p ∧ (∃ p, |(p.1 - M.1) + (p.2 - M.2)| = |(p.1 - N.1) + (p.2 - N.2)|)) } :=
sorry

end find_curves_p_eq_pm_pn_l379_379817


namespace min_black_squares_l379_379717

/-- Given a wooden cube measuring 5x5x5, where each face is divided into unit squares, 
with each unit square colored using one of three colors (black, white, or red) such that 
no two squares sharing an edge have the same color, prove the minimum number of black 
squares required. -/
theorem min_black_squares (cube_size : ℕ) (face_size : ℕ)
  (colors : ℕ) (adjacent_color_diff : ∀ (square1 square2 : ℕ), square1 ≠ square2) :
  (cube_size = 5) ∧ (face_size = 25) ∧ (colors = 3) ∧ (∀ (square1 square2 : ℕ), square1 ≠ square2 → square1 * square2 = black square) →
  ∃ (min_black : ℕ), min_black = 18 :=
by
  sorry

end min_black_squares_l379_379717


namespace circumcenter_of_projection_point_l379_379476

open EuclideanGeometry

noncomputable def is_projection (P O : Point) (T : Triangle) : Prop :=
  ∃ (vectors: Vector), ∀ A ∈ T.vertices, 
    O = A + Vector.proj (P - A) (Plane.spanning_vectors T)

theorem circumcenter_of_projection_point 
  (P O A B C : Point) (h₀ : ¬Collinear {A, B, C})
  (h₁ : is_projection P O (Triangle.mk A B C))
  (h₂ : dist P A = dist P B)
  (h₃ : dist P B = dist P C) : 
  Circumcenter (Triangle.mk A B C) = O :=
by
  sorry

end circumcenter_of_projection_point_l379_379476


namespace symmetry_axis_g_range_l379_379804

-- Define the function f(x)
def f (x : ℝ) : ℝ := √3 * sin (ω * x) * cos (ω * x) + cos (ω * x) ^ 2

-- Condition: Distance between symmetry centers is π/4, implying ω = 1
def ω := 1

-- Symmetry axis equation to be proven
theorem symmetry_axis (k : ℤ) : f x = sin (2 * ω * x + π / 6) + 1 / 2 → 
  ∃ n : ℤ, x = n * (π / 2) + π / 6 :=
by sorry

-- Define the transformed function g(x)
def g (x : ℝ) : ℝ := sin (4 * x - π / 6) + 1 / 2

-- Interval for g(x) and its range to be proven
theorem g_range :
  ∀ x, x ∈ Ioo (-π / 12) (π / 3) →
    g x ∈ Ioo (-1 / 2 : ℝ) (3 / 2 : ℝ) :=
by sorry

end symmetry_axis_g_range_l379_379804


namespace person_age_l379_379672

theorem person_age (A : ℕ) (h : 6 * (A + 6) - 6 * (A - 6) = A) : A = 72 := 
by
  sorry

end person_age_l379_379672


namespace a_squared_plus_b_squared_less_than_c_squared_l379_379195

theorem a_squared_plus_b_squared_less_than_c_squared 
  (a b c : Real) 
  (h : a^2 + b^2 + a * b + b * c + c * a < 0) : 
  a^2 + b^2 < c^2 := 
  by 
  sorry

end a_squared_plus_b_squared_less_than_c_squared_l379_379195


namespace abhay_speed_l379_379529

theorem abhay_speed
    (A S : ℝ)
    (h1 : 30 / A = 30 / S + 2)
    (h2 : 30 / (2 * A) = 30 / S - 1) :
    A = 5 * Real.sqrt 6 :=
by
  sorry

end abhay_speed_l379_379529


namespace find_edge_lengths_sum_l379_379791

noncomputable def sum_edge_lengths (a d : ℝ) (volume surface_area : ℝ) : ℝ :=
  if (a - d) * a * (a + d) = volume ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = surface_area then
    4 * ((a - d) + a + (a + d))
  else
    0

theorem find_edge_lengths_sum:
  (∃ a d : ℝ, (a - d) * a * (a + d) = 512 ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = 352) →
  sum_edge_lengths (Real.sqrt 59) 1 512 352 = 12 * Real.sqrt 59 :=
by
  sorry

end find_edge_lengths_sum_l379_379791


namespace train_trip_distance_l379_379385

theorem train_trip_distance 
  (x : ℝ) -- speed of the train in miles per hour
  (D : ℝ) -- total distance of the trip in miles
  (hx1 : D = x * (1 + 0.75 + 3 * (D - x) / (2 * x) - 4)) 
  (hx2 : D = (x + 120) / x + 3 * (D - x - 120) / (2 * x) + 0.75 - 2.75)
  : D = 550 := 
by
sabot

end train_trip_distance_l379_379385


namespace length_DR_eq_zero_l379_379127

-- Define the rectangle dimensions.
def length : ℝ := 2
def height : ℝ := 1

-- Define the center and radius of the inscribed circle.
def center : ℝ × ℝ := (0, 0)
def radius : ℝ := height / 2

-- Define points D and Q.
def D : ℝ × ℝ := (0, -radius)
def Q : ℝ × ℝ := (0, radius)

-- Define point R as the second intersection of line DQ and circle.
def R : ℝ × ℝ := (0, -radius)

theorem length_DR_eq_zero :
  real.dist D R = 0 :=
by
  -- Proof goes here
  sorry

end length_DR_eq_zero_l379_379127


namespace increase_is_50_percent_l379_379913

def original_lines (L : ℕ) : Prop := L + 80 = 240

def percentage_increase (L : ℕ) (increase_percentage : ℕ) : Prop :=
  increase_percentage = (80 * 100) / L

theorem increase_is_50_percent (L : ℕ) (increase_percentage : ℕ) :
  original_lines L → percentage_increase L increase_percentage → increase_percentage = 50 := by
  intros h1 h2
  rw [original_lines, percentage_increase] at h1 h2
  sorry

end increase_is_50_percent_l379_379913


namespace proof_inequality_l379_379059

noncomputable def proof_problem (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : Prop :=
  (1 - p^m)^n + (1 - q^n)^m ≥ 1

theorem proof_inequality (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
by
  sorry

end proof_inequality_l379_379059


namespace lena_more_candy_bars_than_nicole_l379_379151

theorem lena_more_candy_bars_than_nicole
  (Lena Kevin Nicole : ℕ)
  (h1 : Lena = 16)
  (h2 : Lena + 5 = 3 * Kevin)
  (h3 : Kevin + 4 = Nicole) :
  Lena - Nicole = 5 :=
by
  sorry

end lena_more_candy_bars_than_nicole_l379_379151


namespace union_of_A_and_B_l379_379471

-- Define the sets A and B.
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x < 1}

-- State the theorem that the union of sets A and B is equal to the set of all real numbers.
theorem union_of_A_and_B : A ∪ B = set.univ :=
by
  sorry -- Proof placeholder.

end union_of_A_and_B_l379_379471


namespace ratio_of_height_increases_l379_379721

-- Definitions for the conditions
def r1 : ℝ := 3
def r2 : ℝ := 6
def r_ball : ℝ := 1
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Displaced liquid volumes for each container
def V_displaced : ℝ := volume_sphere r_ball
def V_narrow (h1 : ℝ) : ℝ := Real.pi * r1^2 * h1
def V_wide (h2 : ℝ) : ℝ := Real.pi * r2^2 * h2

-- Proof statement
theorem ratio_of_height_increases (h1 h2 : ℝ) 
  (h_narrow : V_narrow h1 = V_displaced) 
  (h_wide : V_wide h2 = V_displaced) :
  h1 / h2 = 4 := by
  sorry

end ratio_of_height_increases_l379_379721


namespace can_form_equilateral_triangle_100_can_form_equilateral_triangle_99_l379_379461

def can_form_equilateral_triangle_given_matches (n : ℕ) : Bool :=
  let S := n * (n + 1) / 2
  S % 3 = 0

theorem can_form_equilateral_triangle_100 : can_form_equilateral_triangle_given_matches 100 = false := 
by 
  -- Calculating S for n = 100
  have S := 100 * 101 / 2
  -- Checking if S is divisible by 3
  show S % 3 = 0 from sorry

theorem can_form_equilateral_triangle_99 : can_form_equilateral_triangle_given_matches 99 = true := 
by 
  -- Calculating S for n = 99
  have S := 99 * 100 / 2
  -- Checking if S is divisible by 3
  show S % 3 = 0 from sorry

end can_form_equilateral_triangle_100_can_form_equilateral_triangle_99_l379_379461


namespace solve_x_l379_379103

theorem solve_x (x : ℝ) (h : 7^log 7 12 = 6 * x + 3) : x = 3 / 2 :=
by
  sorry

end solve_x_l379_379103


namespace floor_of_neg_sqrt_frac_l379_379012

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end floor_of_neg_sqrt_frac_l379_379012


namespace total_pieces_10_rows_l379_379738

-- Define the conditions for the rods
def rod_seq (n : ℕ) : ℕ := 3 * n

-- Define the sum of the arithmetic sequence for rods
def sum_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

-- Define the conditions for the connectors
def connector_seq (n : ℕ) : ℕ := n + 1

-- Define the sum of the arithmetic sequence for connectors
def sum_connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Define the total pieces calculation
def total_pieces (n : ℕ) : ℕ := sum_rods n + sum_connectors (n + 1)

-- The target statement
theorem total_pieces_10_rows : total_pieces 10 = 231 :=
by
  sorry

end total_pieces_10_rows_l379_379738


namespace range_of_a_l379_379082

noncomputable def f (a x : ℝ) : ℝ := 
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  0 < a ∧ a ≤ 3/4 :=
by {
  sorry
}

end range_of_a_l379_379082


namespace speed_conversion_l379_379358

/-- 
Given a speed of 1.2 kilometers per hour, prove that the equivalent speed in meters per second is approximately 0.333.
-/
theorem speed_conversion (kph : ℝ) (h : kph = 1.2) : (kph * 1000 / 3600) ≈ 0.333 := by
  sorry

end speed_conversion_l379_379358


namespace min_value_of_expression_l379_379893

noncomputable def minimum_possible_value : ℤ :=
  let s : List ℤ := [-8, -6, -4, -1, 3, 5, 7, 10]
  let subsets := s.toFinset.powerset.filter (λ ss => ss.card = 4)
  let x_values := subsets.map (λ subset => subset.sum)
  let expression_values := x_values.map (λ x => x * x + (6 - x) * (6 - x))
  expression_values.minD

theorem min_value_of_expression
  (p q r s t u v w : ℤ)
  (h_distinct : List.nodup [p, q, r, s, t, u, v, w])
  (h_set : [p, q, r, s, t, u, v, w].perm [-8, -6, -4, -1, 3, 5, 7, 10]) :
  (minimum_possible_value = 18) := by
  sorry

end min_value_of_expression_l379_379893


namespace euler_school_voting_problem_l379_379722

theorem euler_school_voting_problem :
  let U := 198
  let A := 149
  let B := 119
  let AcBc := 29
  U - AcBc = 169 → 
  A + B - (U - AcBc) = 99 :=
by
  intros h₁
  sorry

end euler_school_voting_problem_l379_379722


namespace decreasing_interval_g_l379_379511

noncomputable def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := (e^x) * (f x)
def g' (x : ℝ) : ℝ := (2 * x + x^2) * (e ^ x)

theorem decreasing_interval_g :
  (∀ x ∈ Ioo (-2 : ℝ) (0 : ℝ), g' x < 0) :=
by 
  sorry

end decreasing_interval_g_l379_379511


namespace biff_break_even_time_l379_379403

noncomputable def total_cost_excluding_wifi : ℝ :=
  11 + 3 + 16 + 8 + 10 + 35 + 0.1 * 35

noncomputable def total_cost_including_wifi_connection : ℝ :=
  total_cost_excluding_wifi + 5

noncomputable def effective_hourly_earning : ℝ := 12 - 1

noncomputable def hours_to_break_even : ℝ :=
  total_cost_including_wifi_connection / effective_hourly_earning

theorem biff_break_even_time : hours_to_break_even ≤ 9 := by
  sorry

end biff_break_even_time_l379_379403


namespace triangle_side_relationship_l379_379194

theorem triangle_side_relationship (A B C K : Point)
  (h1 : ∠ BAC = 2 * ∠ ABC)
  (h2 : is_angle_bisector AK)
  (h3 : K ∈ segment[BC]) :
  BC^2 = (AC + AB) * AC := 
sorry

end triangle_side_relationship_l379_379194


namespace trader_discount_l379_379673

noncomputable def cost_price := 100
noncomputable def marked_price := cost_price + (0.5 * cost_price)
noncomputable def loss := 0.01 * cost_price
noncomputable def selling_price := cost_price - loss
noncomputable def discount := marked_price - selling_price

theorem trader_discount (cp mp sp : ℝ) (h_cp : cp = cost_price) 
                                        (h_mp : mp = marked_price) 
                                        (h_sp : sp = selling_price) :
  discount = 51 :=
by
  sorry

end trader_discount_l379_379673


namespace coordinates_provided_l379_379536

-- Define the coordinates of point P in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P with its given coordinates
def P : Point := {x := 3, y := -5}

-- Lean 4 statement for the proof problem
theorem coordinates_provided : (P.x, P.y) = (3, -5) := by
  -- Proof not provided
  sorry

end coordinates_provided_l379_379536


namespace hall_volume_l379_379345

theorem hall_volume : 
  ∀ (l : ℝ) (b : ℝ) (h : ℝ), 
  l = 15 → 
  b = 12 → 
  (2 * l * b = 2 * (l * h) + 2 * (b * h)) → 
  (l * b * h = 1201.8) :=
by
  intros l b h l_eq b_eq area_eq
  subst l_eq
  subst b_eq
  simp at area_eq
  field_simp at area_eq
  linarith
  sorry  -- Terminates the statement pending the actual proof

end hall_volume_l379_379345


namespace part_one_part_two_l379_379137

open Real

theorem part_one (x y : ℝ) : let O := (0, 0) in 
  (∀ t : ℝ, (x - sqrt 3 * y = 4)) → x ^ 2 + y ^ 2 = 4 :=
by
  -- Proof for part one
  sorry

theorem part_two (A B P : ℝ × ℝ) :
  let O := (0, 0) in
  let A := (-2, 0) in
  let B := (2, 0) in
  (x^2 + y^2 < 4) →
  sqrt ((P.1 + 2)^2 + P.2^2) * sqrt ((P.1 - 2)^2 + P.2^2) = (P.1)^2 + (P.2)^2 →
  -2 ≤ (P.1)^2 + (P.2)^2 - 4 ∧ (P.1)^2 + (P.2)^2 - 4 < 0 :=
by
  -- Proof for part two
  sorry

end part_one_part_two_l379_379137


namespace area_comparison_l379_379128

variables (A B C D K L : Point)
variables (h c : ℝ)

def right_triangle (A B C : Point) : Prop :=
  ∃ angle_ABC angle_ACB : ℝ, angle_ABC = 90 ∧ angle_ACB = 90

def altitude_on_hypotenuse (A D B C : Point) : Prop :=
  ∃ h : ℝ, AD = h

def area_triangle (A B C : Point) : ℝ :=
  1/2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def incenters_connected (A B C D K L : Point) : Prop :=
  ∃ M N : Point, M.incenter (triangle ABD) ∧ N.incenter (triangle ACD) ∧ line (M, N).intersect AB = K ∧ line (M, N).intersect AC = L

theorem area_comparison (A B C D K L : Point) (h c : ℝ) (Hright : right_triangle A B C)
  (Haltitude : altitude_on_hypotenuse A D B C) (HareaABC : c ≥ 2 * h) :
  area_triangle A B C ≥ 2 * area_triangle A K L :=
sorry

end area_comparison_l379_379128


namespace range_of_x_minus_cos_y_l379_379881

theorem range_of_x_minus_cos_y
  (x y : ℝ)
  (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (A : Set ℝ), A = {z | -1 ≤ z ∧ z ≤ 1 + Real.sqrt 3} ∧ x - Real.cos y ∈ A :=
by
  sorry

end range_of_x_minus_cos_y_l379_379881


namespace jane_age_l379_379642

theorem jane_age (j : ℕ) 
  (h₁ : ∃ (k : ℕ), j - 2 = k^2)
  (h₂ : ∃ (m : ℕ), j + 2 = m^3) :
  j = 6 :=
sorry

end jane_age_l379_379642


namespace number_of_children_l379_379183

theorem number_of_children 
    (total_granola_bars : ℕ)
    (granola_bars_eaten : ℕ)
    (granola_bars_per_child : ℕ)
    (remaining_granola_bars : total_granola_bars - granola_bars_eaten = 120)
    (number_of_children_per_granola_bars : 120 / granola_bars_per_child = 6) : 
    total_granola_bars = 200 ∧ 
    granola_bars_eaten = 80 ∧
    granola_bars_per_child = 20 → 
    ∃ c : ℕ, c = 6 :=
by
  intro h
  cases h with ht h
  cases h with he hg
  use 6
  sorry

end number_of_children_l379_379183


namespace polynomial_ac_sum_l379_379043

theorem polynomial_ac_sum (a b c d e : ℝ)
  (h : (λ x, a * x^4 + b * x^3 + c * x^2 + d * x + e) =
       (λ x, (2 * x - 1)^4)) :
  a + c = 40 := by
  sorry

end polynomial_ac_sum_l379_379043


namespace sequence_equal_l379_379353

variable {n : ℕ} (h1 : 2 ≤ n)
variable (a : ℕ → ℝ)
variable (h2 : ∀ i, a i ≠ -1)
variable (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
variable (h4 : a n = a 0)
variable (h5 : a (n + 1) = a 1)

theorem sequence_equal 
  (h1 : 2 ≤ n)
  (h2 : ∀ i, a i ≠ -1) 
  (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
  (h4 : a n = a 0)
  (h5 : a (n + 1) = a 1) :
  ∀ i, a i = a 0 := 
sorry

end sequence_equal_l379_379353


namespace right_angled_triangles_count_acute_angled_triangles_count_l379_379241

-- Define the distance between points in a regular 2n-sided polygon
-- with vertices labeled A₁, A₂,..., A₂ₙ.
def is_right_angled_triangle (i j k : ℕ) (n : ℕ) : Prop :=
  sorry  -- Define whether the triangle AᵢAⱼAₖ is right-angled

def is_acute_angled_triangle (i j k : ℕ) (n : ℕ) : Prop :=
  sorry  -- Define whether the triangle AᵢAⱼAₖ is acute-angled

theorem right_angled_triangles_count (n : ℕ) (h : 2 ≤ n) : 
  ∃ R_n : ℕ, R_n = 2 * n * (n - 1) ∧
  R_n = ∑ (i j k : ℕ) in (finset.range (2 * n)).powerset_len 3,
        if is_right_angled_triangle i j k n then 1 else 0 :=
sorry

theorem acute_angled_triangles_count (n : ℕ) (h : 3 ≤ n) :
  ∃ A_n : ℕ, A_n = (n * (n - 1) * (n - 2)) / 3 ∧
  A_n = ∑ (i j k : ℕ) in (finset.range (2 * n)).powerset_len 3,
        if is_acute_angled_triangle i j k n then 1 else 0 :=
sorry

end right_angled_triangles_count_acute_angled_triangles_count_l379_379241


namespace discarded_number_is_55_l379_379214

noncomputable def sum_of_50_numbers := 2500
def discarded_average {X : ℕ} (S : ℕ) := (S - X - 45) / 48 = 50

theorem discarded_number_is_55 : ∀ X : ℕ,
    (sum_of_50_numbers - X - 45) / 48 = 50 → X = 55 := 
by
  assume X,
  assume h : (sum_of_50_numbers - X - 45) / 48 = 50,
  calc
    (sum_of_50_numbers - X - 45) / 48 = 50 : h
    ... → sorry -- Include steps to solve and confirm X = 55.

end discarded_number_is_55_l379_379214


namespace number_of_boys_in_second_group_l379_379832

noncomputable def daily_work_done_by_man (M : ℝ) (B : ℝ) : Prop :=
  M = 2 * B

theorem number_of_boys_in_second_group
  (M B : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = (13 * M + 24 * B) * 4)
  (h2 : daily_work_done_by_man M B) :
  24 = 24 :=
by
  -- The proof is omitted.
  sorry

end number_of_boys_in_second_group_l379_379832


namespace minimum_value_of_g_gm_equal_10_implies_m_is_5_l379_379085

/-- Condition: Definition of the function y in terms of x and m -/
def y (x m : ℝ) : ℝ := x^2 + m * x - 4

/-- Theorem about finding the minimum value of g(m) -/
theorem minimum_value_of_g (m : ℝ) :
  ∃ g : ℝ, g = (if m ≥ -4 then 2 * m
      else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
      else 4 * m + 12) := by
  sorry

/-- Theorem that if the minimum value of g(m) is 10, then m must be 5 -/
theorem gm_equal_10_implies_m_is_5 :
  ∃ m, (if m ≥ -4 then 2 * m
       else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
       else 4 * m + 12) = 10 := by
  use 5
  sorry

end minimum_value_of_g_gm_equal_10_implies_m_is_5_l379_379085


namespace sum_q_t_8_eq_128_l379_379558

noncomputable def T : Type :=
  { b : Fin 8 → Fin 2 // True }

noncomputable def q_t (t : T) : Polynomial ℚ :=
  Polynomial.interpolate (Fin 8) (λ i => (t.val i : ℚ))

noncomputable def q (x : ℚ) : ℚ :=
  ∑ t in Finset.univ.image (λ b : Fin 8 → Fin 2 => ⟨b, trivial⟩ : Finset T), q_t t x

theorem sum_q_t_8_eq_128 : q 8 = 128 := 
sorry

end sum_q_t_8_eq_128_l379_379558


namespace binom_divisibility_l379_379883

theorem binom_divisibility (k n : ℕ) (p : ℕ) (h1 : k > 1) (h2 : n > 1) 
  (h3 : p = 2 * k - 1) (h4 : Nat.Prime p) (h5 : p ∣ (Nat.choose n 2 - Nat.choose k 2)) : 
  p^2 ∣ (Nat.choose n 2 - Nat.choose k 2) := 
sorry

end binom_divisibility_l379_379883


namespace tangent_angle_range_l379_379915

theorem tangent_angle_range :
  ∀ x : ℝ, ∃ α : ℝ, 0 ≤ α ∧ α < π → 
  α ∈ [0, (π/2)) ∪ [(3 * π / 4), π) ↔ 
  let y := (1 / 3) * x^3 - 2 * x^2 + 3 * x;
  let dy := diff y;
  let tanα := dy in
  (-1 ≤ tanα ∧ tanα < 0 → α ∈ [(3 * π / 4), π)) ∧
  (0 ≤ tanα → α ∈ [0, (π/2))) :=
begin
  sorry
end

end tangent_angle_range_l379_379915


namespace complement_intersection_l379_379815

noncomputable def U := Set.univ : Set ℝ

def A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 > 0 }

def B : Set ℝ := { x : ℝ | 2 < x ∧ x < 4 }

theorem complement_intersection :
  ((U \ B) ∩ A) = ( { x : ℝ | x < -1 } ∪ { x : ℝ | x ≥ 4 }) := by
  sorry

end complement_intersection_l379_379815


namespace angle_A_value_possible_values_for_b_c_l379_379847

-- Given triangle setup conditions
variables {A B C : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C]
variables (a b c : ℝ)
variables (S : ℝ)
variables (angle_A angle_B angle_C : ℝ)
variable (acute : Prop)

-- Given conditions for first question
axiom sqrt3_c_eq_2a_sin_C : sqrt 3 * c = 2 * a * sin angle_C
axiom sin_C_nonzero : sin angle_C ≠ 0

-- First part: prove angle_A size
theorem angle_A_value :
  angle_A = π / 3 ∨ angle_A = 2 * π / 3 :=
sorry

-- Additional conditions for the second question:
axiom angle_A_acute : angle_A = π / 3
axiom a_value : a = 2 * sqrt 3
axiom area_value : S = 2 * sqrt 3

-- Second part: prove possible values for b and c
theorem possible_values_for_b_c :
  (b = 4 ∧ c = 2) ∨ (b = 2 ∧ c = 4) :=
sorry

end angle_A_value_possible_values_for_b_c_l379_379847


namespace average_speed_l379_379609

-- Define the given conditions as Lean variables and constants
variables (v : ℕ)

-- The average speed problem in Lean
theorem average_speed (h : 8 * v = 528) : v = 66 :=
sorry

end average_speed_l379_379609


namespace concyclic_points_l379_379933

variables {A B C D E F G L M : Type*} [euclidean_space A] [euclidean_space B]
  [euclidean_space C] [euclidean_space D] [euclidean_space E]
  [euclidean_space F] [euclidean_space G] [euclidean_space L] [euclidean_space M]

-- Definitions of the points and rotations
def is_square (A B C D : Type*) : Prop :=
  -- A definition that A, B, C, D form a square goes here

def is_rotation (p₁ p₂ : Type*) (α : ℝ) (center : Type*) : Prop :=
  -- A definition that p1 is rotated by an angle α around center to get p2 

def is_intersection (l₁ l₂ : Type*) : Type* :=
  -- A definition of the intersection of two lines l1 and l2

def lies_on_circumcircle (p q r s : Type*) (l : Type*) : Prop :=
  -- A definition that point l lies on the circle passing through points p, q, r, s

-- Main Theorem statement
theorem concyclic_points
  (sq : is_square A B C D)
  (α : ℝ)
  (rot_D_to_E : is_rotation D E α A)
  (rot_B_to_G : is_rotation B G α C)
  (F_inter : F = is_intersection (line_through A E) (line_through B G))
  (L_circum : lies_on_circumcircle G C B L)
  (M_circum : lies_on_circumcircle B D E M) :
  lies_on_circumcircle A B F L M := sorry

end concyclic_points_l379_379933


namespace greatest_integer_thinking_of_l379_379879

theorem greatest_integer_thinking_of :
  ∃ n : ℕ, n < 150 ∧ (n % 9 = 7) ∧ (n % 11 = 7) ∧ (n % 5 = 1) ∧ (n = 142) :=
begin
 sorry
end

end greatest_integer_thinking_of_l379_379879


namespace cube_root_neg_64_l379_379222

theorem cube_root_neg_64 : real.cbrt (-64) = -4 :=
by
  sorry

end cube_root_neg_64_l379_379222


namespace range_of_fP_l379_379062

noncomputable def f (d : ℝ) := real.sqrt (d^2 - 1)

theorem range_of_fP :
  let C := {p : ℝ × ℝ | p.1^2 + (p.2 - 4)^2 = 1}
  let D := {p : ℝ × ℝ | (p.1 + 4)^2 + (p.2 - 1)^2 = 4}
  let P := (x y : ℝ)
  let d := real.sqrt ((x + 4)^2 + (y - 1)^2)
  d ∈ {d : ℝ | ∃ p ∈ D, (p.1 - 4)^2 + (p.2 - 1)^2 = 4} →
  2 * real.sqrt 2 ≤ f d ∧ f d ≤ 4 * real.sqrt 3 :=
by
  sorry

end range_of_fP_l379_379062


namespace sequence_an_correct_and_range_a_l379_379465

-- Define sequences and their summations based on given conditions
def sequence_sn (n : ℕ) : ℕ := (1 / 2) * n^2 + (1 / 2) * n
def sequence_an (n : ℕ) : ℕ := n

def partial_sum_Tn (n : ℕ) : ℕ :=
  ∑ k in finset.range n, (1 / sequence_an(k) * sequence_an(k + 2))

-- State the theorem with conditions and what to prove
theorem sequence_an_correct_and_range_a (a : ℝ) (n : ℕ) (hn : n > 0) :
  (∃ an_formula : ℕ → ℕ, ∀ n : ℕ, an_formula n = n) ∧
  (∀ Tn : ℕ, Tn = ∑ k in finset.range n, 1 / (sequence_an k * sequence_an (k + 2)) → 
   (Tn > 1 / 3 * real.log a (1 - a) ∀ (n : ℕ), n > 0) → 0 < a ∧ a < 1/2) :=
begin
  sorry
end

end sequence_an_correct_and_range_a_l379_379465


namespace division_exponentiation_addition_l379_379409

theorem division_exponentiation_addition :
  6 / -3 + 2^2 * (1 - 4) = -14 := by
sorry

end division_exponentiation_addition_l379_379409


namespace find_r_for_f2_eq_0_l379_379890

def f (x r : ℝ) : ℝ := 2 * x^4 + x^3 + x^2 - 3 * x + r

theorem find_r_for_f2_eq_0 : ∃ r : ℝ, f 2 r = 0 := 
  by {
    use -38,
    simp [f],
    norm_num
  }

end find_r_for_f2_eq_0_l379_379890


namespace cardboard_circles_fit_on_table_l379_379359

theorem cardboard_circles_fit_on_table
  (original_circles : set (real × real))
  (H1 : ∀ c ∈ original_circles, c.1^2 + c.2^2 ≤ 1)
  (H2 : ∀ c1 c2 ∈ original_circles, c1 ≠ c2 → (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 ≥ 1)
  (H3 : ∀ c ∈ original_circles, 0 ≤ c.1 ∧ c.1 ≤ 5 ∧ 0 ≤ c.2 ∧ c.2 ≤ 8)
  (new_circle : real × real)
  (H4 : new_circle.1^2 + new_circle.2^2 ≤ 2^2)
  (H5 : ∀ c ∈ original_circles, (new_circle.1 - c.1)^2 + (new_circle.2 - c.2)^2 ≥ (2/2 + 1/2)^2) :
  ∃ (new_placement : set (real × real)),
    (∀ c ∈ new_placement, c.1^2 + c.2^2 ≤ 1 ∨ c = new_circle) ∧
    (∀ c1 c2 ∈ new_placement, c1 ≠ c2 → (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 ≥ 1) ∧
    (∀ c ∈ new_placement, 0 ≤ c.1 ∧ c.1 ≤ 7 ∧ 0 ≤ c.2 ∧ c.2 ≤ 7) :=
by
  sorry

end cardboard_circles_fit_on_table_l379_379359


namespace paying_students_pay_7_dollars_l379_379397

-- Define the number of students and the percentage of students receiving free lunch
def total_students : ℕ := 50
def percent_free_lunch : ℝ := 0.4
def total_cost : ℝ := 210

-- Define the number of paying students and the price per paying student
def paying_students : ℕ := (1 - percent_free_lunch) * total_students
def price_per_paying_student : ℝ := total_cost / paying_students

-- The theorem to prove
theorem paying_students_pay_7_dollars :
  price_per_paying_student = 7 :=
by
  -- Proof goes here
  sorry

end paying_students_pay_7_dollars_l379_379397


namespace find_angle_BDA_l379_379545

-- Define the Triangle and relevant angles
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Define the angles given
def angle_ABC : ℝ := 74
def angle_BCA : ℝ := 48
def angle_sum_triangle : ℝ := 180

-- Define the problem to prove
def angle_in_triangle (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] :=
  let angle_BAC := angle_sum_triangle - angle_ABC - angle_BCA in
  let angle_BAD := angle_BAC / 2 in
  angle_BAD = 29

-- Begin the theorem with the setup of the angles and conditions
theorem find_angle_BDA
  (h1 : angle_ABC = 74)
  (h2 : angle_BCA = 48)
  (h3 : angle_sum_triangle = 180)
  : angle_in_triangle A B C D :=
begin
  sorry
end

end find_angle_BDA_l379_379545


namespace hyperbola_eccentricity_is_2_l379_379616

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
(h3 : b = sqrt 3 * a) (h4 : c = sqrt (a^2 + b^2)) : ℝ := c / a

theorem hyperbola_eccentricity_is_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
(h3 : b = sqrt 3 * a) (h4 : c = sqrt (a^2 + b^2)) : hyperbola_eccentricity a b c h1 h2 h3 h4 = 2 :=
begin
  sorry
end

end hyperbola_eccentricity_is_2_l379_379616


namespace standing_next_to_Denis_l379_379311

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379311


namespace dodecahedron_has_150_interior_diagonals_l379_379825

def dodecahedron_diagonals (vertices : ℕ) (adjacent : ℕ) : ℕ :=
  let total := vertices * (vertices - adjacent - 1) / 2
  total

theorem dodecahedron_has_150_interior_diagonals :
  dodecahedron_diagonals 20 4 = 150 :=
by
  sorry

end dodecahedron_has_150_interior_diagonals_l379_379825


namespace possible_values_of_a_l379_379796

def A (a : ℝ) : Set ℝ := { x | 0 < x ∧ x < a }
def B : Set ℝ := { x | 1 < x ∧ x < 2 }
def complement_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem possible_values_of_a (a : ℝ) :
  (∃ x, x ∈ A a) →
  B ⊆ complement_R (A a) →
  0 < a ∧ a ≤ 1 :=
by 
  sorry

end possible_values_of_a_l379_379796


namespace problem_statement_l379_379891

def imaginary_unit : ℂ := complex.i
def z : ℂ := 1 - 2 * imaginary_unit
def z_conjugate : ℂ := conj z
def result : ℂ := z + imaginary_unit * z_conjugate
def point_in_third_quadrant (p : ℂ) := p.re < 0 ∧ p.im < 0

theorem problem_statement : point_in_third_quadrant result := by
  sorry

end problem_statement_l379_379891


namespace cube_root_neg_64_l379_379220

theorem cube_root_neg_64 : (∛(-64) = -4) :=
by sorry

end cube_root_neg_64_l379_379220


namespace line_integral_part_a_l379_379726

theorem line_integral_part_a (L : Path (π/2, 2) (π/6, 1) ℝ²) :
  ∫ (p : ℝ × ℝ) in L, 2 * p.2 * sin (2 * p.1) ∂p.1 - cos (2 * p.1) ∂p.2 = -5/2 :=
sorry

end line_integral_part_a_l379_379726


namespace investigate_f_l379_379547

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

#check
theorem investigate_f :
  (∀ x, x ∈ Set.Ioc (-∞) (-2) ∪ Set.Ioc 1 ∞ → f x < f (x + 1)) ∧
  (∀ x, x ∈ Set.Ioc (-2) 1 → f x > f (x + 1)) ∧
  (f (-2) = 21) ∧ 
  (f 1 = -6) ∧ 
  ((∀ x, x ∈ Set.Icc (-1) 5 → f x ≤ 266) ∧
  (f 5 = 266) ∧ 
  (∀ x, x ∈ Set.Icc (-1) 5 → f (-6) ≤ f x)) :=
by sorry

end investigate_f_l379_379547


namespace solve_for_x_l379_379205

-- Definitions from conditions
def exponential_equation (x : ℝ) : Prop :=
  3^x * 9^x = 27^(x - 12)

-- Stating the proof problem
theorem solve_for_x : ∃ (x : ℝ), exponential_equation x ∧ x = 12 :=
sorry

end solve_for_x_l379_379205


namespace untested_probability_range_l379_379243

noncomputable def probability_untested_items (n : ℕ) (p : ℝ) (a b : ℕ) : ℝ :=
  let μ := n * p
  let q := 1 - p
  let σ := Real.sqrt (n * p * q)
  let z1 := (a - μ) / σ
  let z2 := (b - μ) / σ
  let Φ := λ x : ℝ => (Real.erf (x / Real.sqrt 2) + 1) / 2
  in Φ z2 - Φ z1

theorem untested_probability_range (h_n : 400 = 400) (h_p : 0.2 = 0.2) :
  abs (probability_untested_items 400 0.2 70 100 - 0.8882) < 0.01 :=
by
  sorry

end untested_probability_range_l379_379243


namespace hannah_final_pay_l379_379822

theorem hannah_final_pay : (30 * 18) - (5 * 3) + (15 * 4) - (((30 * 18) - (5 * 3) + (15 * 4)) * 0.10 + ((30 * 18) - (5 * 3) + (15 * 4)) * 0.05) = 497.25 :=
by
  sorry

end hannah_final_pay_l379_379822


namespace paying_students_pay_7_dollars_l379_379396

-- Define the number of students and the percentage of students receiving free lunch
def total_students : ℕ := 50
def percent_free_lunch : ℝ := 0.4
def total_cost : ℝ := 210

-- Define the number of paying students and the price per paying student
def paying_students : ℕ := (1 - percent_free_lunch) * total_students
def price_per_paying_student : ℝ := total_cost / paying_students

-- The theorem to prove
theorem paying_students_pay_7_dollars :
  price_per_paying_student = 7 :=
by
  -- Proof goes here
  sorry

end paying_students_pay_7_dollars_l379_379396


namespace certain_number_l379_379833

theorem certain_number (x certain_number : ℕ) (h1 : x = 3327) (h2 : 9873 + x = certain_number) : 
  certain_number = 13200 := 
by
  sorry

end certain_number_l379_379833


namespace triangle_angle_incenter_parallel_l379_379848

-- Define the problem context and prove the required angle property

theorem triangle_angle_incenter_parallel (A B C I F P : Point) 
  (h1 : IsTriangle A B C)
  (h2 : Angle A = 60°)
  (h3 : IsIncenter I A B C)
  (h4 : LineThrough I ParallelTo (LineThrough A C))
  (h5 : LineIntersect (LineThrough I F) (LineThrough A B) F)
  (h6 : OnLine P (LineThrough B C))
  (h7 : 3 * LineSegment B P = LineSegment B C) :
  Angle B F P = (1 / 2) * Angle B := 
sorry

end triangle_angle_incenter_parallel_l379_379848


namespace trapezoid_area_correct_l379_379218

noncomputable def trapezoid_area (a b leg1 leg2 : ℝ) : ℝ :=
  let h := 3 in -- As derived from the Pythagorean theorem in the given solution
  let base_sum := a + b in
  (1 / 2) * base_sum * h

theorem trapezoid_area_correct :
  trapezoid_area 10 21 (Real.sqrt 34) (3 * Real.sqrt 5) = 93 / 2 := by
  sorry

end trapezoid_area_correct_l379_379218


namespace total_sugar_l379_379682

theorem total_sugar (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by {
  -- The proof goes here
  sorry
}

end total_sugar_l379_379682


namespace a_2015_value_l379_379055

noncomputable def a : ℕ → ℝ 
| 0     := 0 -- zero term for convenience of indexing 
| 1     := real.sqrt 3
| (n+1) := floor (a n) + 1 / fract (a n)

theorem a_2015_value :
  a 2015 = 3021 + real.sqrt 3 :=
sorry

end a_2015_value_l379_379055


namespace fermats_little_theorem_l379_379919

theorem fermats_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : (a^p - a) % p = 0 := 
by sorry

end fermats_little_theorem_l379_379919


namespace triangles_arithmetic_progression_l379_379920

noncomputable def triangle_properties (a b c : ℝ) : Prop :=
  a = 2 * b - c ∧ c = 2 * b - a

theorem triangles_arithmetic_progression 
  (a b c : ℝ)
  (h : triangle_properties a b c)
  (I O : Point)
  (I_O_line_perpendicular_bisector : ∃ (B : Point), line_through I O ⊥ bisector B) :
  ∃ (B : Point), I_O_line_perpendicular_bisector :=
sorry

end triangles_arithmetic_progression_l379_379920


namespace correct_proposition_l379_379080

-- Definitions based on the given conditions
def proposition1 : Prop :=
  ¬ ∃ x : ℝ, sin x + cos x = 3/2

def proposition2 : Prop :=
  ¬ ∀ x ≠ 2 * (π : ℝ) * n + π / 2, n : ℤ, 
    y = (sin^2 x - sin x) / (sin x - 1) → y = -.y

def proposition3 : Prop :=
  ¬ ∀ x : ℝ, y = abs (sin x - 1/2) -> periodic y π

def proposition4 : Prop :=
  let y1 := λ x : ℝ, log (abs (x - 1))
  let y2 := λ x : ℝ, -2 * cos (π * x)
  let intersections : list ℝ := filter (λ x, -2 * (cos (π * x)) = log (abs (x - 1))) (range -2 4) in
  sum intersections = 6

-- Main theorem
theorem correct_proposition : proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 := sorry

end correct_proposition_l379_379080


namespace product_of_roots_t_cube_eq_125_l379_379031

theorem product_of_roots_t_cube_eq_125
  (t : ℂ) 
  (h : t^3 = 125) : 
  ((5 : ℂ) * (complex.exp (2 * real.pi * complex.I / 3)) * (complex.exp (-2 * real.pi * complex.I / 3)) = 125) := 
sorry

end product_of_roots_t_cube_eq_125_l379_379031


namespace maria_minimum_workers_l379_379420

theorem maria_minimum_workers
  (total_job : ℝ)
  (days_worked : ℕ)
  (initial_workers : ℕ)
  (job_done_in_days : ℝ)
  (total_days : ℕ)
  (remaining_days : ℕ)
  (productivity_constant : bool)
  (remaining_job : ℝ)
  (required_rate : ℝ)
  (worker_rate : ℝ)
  (w : ℕ) :
  total_job = 1 →
  days_worked = 10 →
  initial_workers = 10 →
  job_done_in_days = 2 / 5 →
  total_days = 40 →
  remaining_days = 30 →
  productivity_constant = tt →
  remaining_job = 3 / 5 →
  required_rate = remaining_job / remaining_days →
  worker_rate = job_done_in_days / (initial_workers * days_worked) →
  (required_rate = w * worker_rate) →
  w = 5 :=
by
  intros
  sorry

end maria_minimum_workers_l379_379420


namespace length_of_AX_l379_379546

theorem length_of_AX
  (A B C X : Point)
  (h_triangle : triangle A B C)
  (h_CX_bisects_angle : angle_bisector_of ∠ACB C X)
  (h_AB : dist A B = 50)
  (h_BC : dist B C = 45)
  (h_AC : dist A C = 40)
  (h_BX : dist B X = 30)
  : dist A X = 80 / 3 :=
by
  sorry

end length_of_AX_l379_379546


namespace unique_b_infinitely_many_solutions_l379_379006

theorem unique_b_infinitely_many_solutions :
  ∃! b : ℝ, ∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 10) :=
by 
  use -6
  intro x
  calc
    5 * (3 * x - (-6)) = 3 * (5 * x + 10) : sorry 

end unique_b_infinitely_many_solutions_l379_379006


namespace distance_between_intersections_max_x_plus_y_on_C2_l379_379541

noncomputable def C1 : ℝ × ℝ → Prop := λ p, p.1 + p.2 - 1 = 0
noncomputable def C2 : ℝ × ℝ → Prop := λ p, (p.1 - 1) ^ 2 + (p.2 + 2) ^ 2 = 5

def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_between_intersections :
  ∃ A B : ℝ × ℝ, C1 A ∧ C2 A ∧ C1 B ∧ C2 B ∧ distance A B = 2 * sqrt 3 :=
sorry

theorem max_x_plus_y_on_C2 :
  ∃ (M : ℝ × ℝ), C2 M ∧ (M.1 + M.2) = sqrt 10 - 1 :=
sorry

end distance_between_intersections_max_x_plus_y_on_C2_l379_379541


namespace gervais_avg_mileage_l379_379039
variable (x : ℤ)

def gervais_daily_mileage : Prop := ∃ (x : ℤ), (3 * x = 1250 - 305) ∧ x = 315

theorem gervais_avg_mileage : gervais_daily_mileage :=
by
  sorry

end gervais_avg_mileage_l379_379039


namespace correct_option_D_l379_379657

theorem correct_option_D (a : ℝ) : (-a^3)^2 = a^6 :=
sorry

end correct_option_D_l379_379657


namespace polar_equation_is_circle_l379_379443

-- Define the polar coordinates equation condition
def polar_equation (r θ : ℝ) : Prop := r = 5

-- Define what it means for a set of points to form a circle centered at the origin with a radius of 5
def is_circle_radius_5 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- State the theorem we want to prove
theorem polar_equation_is_circle (r θ : ℝ) (x y : ℝ) (h1 : polar_equation r θ)
  (h2 : x = r * Real.cos θ) (h3 : y = r * Real.sin θ) : is_circle_radius_5 x y := 
sorry

end polar_equation_is_circle_l379_379443


namespace complementary_angle_decrease_l379_379639

theorem complementary_angle_decrease (A B : ℝ) (h1 : A + B = 90) (h2 : A / B = 3 / 7) :
  (A * 1.2 + (90 - A * 1.2)) = 90 ∧ ((B - (90 - A * 1.2)) / B) * 100 = 8.571 := 
by 
  -- Define the new angle after increase
  let A_new := A * 1.2
  -- Complementary angle property
  have h3 : A_new + (90 - A_new) = 90, from by linarith
  
  -- Calculate the decrease percentage
  have h4 : ((B - (90 - A_new)) / B) * 100 = 8.571, from sorry

  -- Proving
  exact ⟨h3, h4⟩

end complementary_angle_decrease_l379_379639


namespace part1_geometric_sequence_part2_sum_property_l379_379467

-- Define the sequence {an}
def a : ℕ → ℝ
| 0       := 1   -- Offset by 1 for indexing in Lean
| (n + 1) := 1/3^n - a n

-- Define the sequence {bn}
def b (n : ℕ) : ℝ := 3^(n - 1) * a n - 1/4

-- Define Sn as per the given sequence sum
def S (n : ℕ) : ℝ := (finset.range n).sum (λ k, 3^k * a (k + 1))

-- The first part of the proof problem: {bn} is a geometric sequence
theorem part1_geometric_sequence : ∀ n, b (n + 1) / b n = -3 := by
  intro n
  sorry

-- The second part of the proof problem: Prove the given Sn formula
theorem part2_sum_property (n : ℕ) : 4 * S n - 3^n * a n = n := by
  sorry

end part1_geometric_sequence_part2_sum_property_l379_379467


namespace Louisa_total_travel_time_l379_379188

theorem Louisa_total_travel_time :
  ∀ (v : ℝ), v > 0 → (200 / v) + 4 = (360 / v) → (200 / v) + (360 / v) = 14 :=
by
  intros v hv eqn
  sorry

end Louisa_total_travel_time_l379_379188


namespace arith_seq_a4_a10_l379_379129

variable {a : ℕ → ℕ}
axiom hp1 : a 1 + a 2 + a 3 = 32
axiom hp2 : a 11 + a 12 + a 13 = 118

theorem arith_seq_a4_a10 :
  a 4 + a 10 = 50 :=
by
  have h1 : a 2 = 32 / 3 := sorry
  have h2 : a 12 = 118 / 3 := sorry
  have h3 : a 2 + a 12 = 50 := sorry
  exact sorry

end arith_seq_a4_a10_l379_379129


namespace num_elements_in_B_l379_379814

-- Condition: Define set A and set B
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | ∃ a b ∈ A, x = a - b}

-- The theorem proving the number of elements in set B is 5
theorem num_elements_in_B : B.toFinset.card = 5 :=
by
  sorry

end num_elements_in_B_l379_379814


namespace determine_r_l379_379845

theorem determine_r (S : ℕ → ℤ) (r : ℤ) (n : ℕ) (h1 : 2 ≤ n) (h2 : ∀ k, S k = 2^k + r) : 
  r = -1 :=
sorry

end determine_r_l379_379845


namespace denis_neighbors_l379_379293

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379293


namespace exam_total_questions_l379_379125

/-- 
In an examination, a student scores 4 marks for every correct answer 
and loses 1 mark for every wrong answer. The student secures 140 marks 
in total. Given that the student got 40 questions correct, 
prove that the student attempted a total of 60 questions. 
-/
theorem exam_total_questions (C W T : ℕ) 
  (score_correct : C = 40)
  (total_score : 4 * C - W = 140)
  (total_questions : T = C + W) : 
  T = 60 := 
by 
  -- Proof omitted
  sorry

end exam_total_questions_l379_379125


namespace pages_read_on_fourth_day_l379_379496

theorem pages_read_on_fourth_day :
  let day1 := 63
  let day2 := 2 * day1
  let day3 := day2 + 10
  let total_pages_in_book := 354
  let total_read_first_three_days := day1 + day2 + day3
  total_pages_in_book - total_read_first_three_days = 29 := 
by
  let day1 := 63
  let day2 := 2 * day1
  let day3 := day2 + 10
  let total_pages_in_book := 354
  have total_read_first_three_days : day1 + day2 + day3 = 325 := by sorry
  exact Eq.trans (Eq.symm (Nat.sub_eq (Nat.le.intro sorry))) rfl

end pages_read_on_fourth_day_l379_379496


namespace tan_cot_equivalence_l379_379828

theorem tan_cot_equivalence (θ : ℝ) (c d : ℝ) (h : (tan θ) ^ 4 / c + (cot θ) ^ 4 / d = 1 / (c + d)) :
  (tan θ) ^ 8 / c ^ 3 + (cot θ) ^ 8 / d ^ 3 = (c ^ 5 + d ^ 5) / (c * d) ^ 4 := 
by 
  sorry

end tan_cot_equivalence_l379_379828


namespace ratio_of_inscribed_square_areas_l379_379335

theorem ratio_of_inscribed_square_areas (r : ℝ) (hr : r > 0) :
  let s1 := r / Real.sqrt 2 in
  let s2 := 2 * Real.sqrt 2 * r in
  (s1^2 / s2^2) = 1 / 16 := 
by
  let s1 := r / Real.sqrt 2
  let s2 := 2 * Real.sqrt 2 * r
  calc
    s1^2 / s2^2 = (r^2 / 2) / (8 * r^2) : by sorry
              ... = (1 / 16) : by sorry

end ratio_of_inscribed_square_areas_l379_379335


namespace shaded_area_l379_379235

-- Define the given constants
def CD_length : ℝ := 96
def inner_radius : ℝ := 36
def outer_radius : ℝ := 60

-- Define the length of CQ based on problem conditions
def CQ_length : ℝ := CD_length / 2

-- Theorem statement: Prove area of the shaded region is 2304 * π square units
theorem shaded_area :
  let inner_circle_area := π * inner_radius^2
  let outer_circle_area := π * outer_radius^2
  let shaded_area := outer_circle_area - inner_circle_area
  shaded_area = 2304 * π :=
by
  sorry

end shaded_area_l379_379235


namespace C1_C2_intersection_AB_length_and_product_l379_379535

noncomputable def parametric_C1 (t : ℝ) : ℝ × ℝ :=
(1 - (Real.sqrt 2) / 2 * t, 1 + (Real.sqrt 2) / 2 * t)

def polar_equation_C2 (rho theta : ℝ) : ℝ :=
rho ^ 2 - 2 * rho * Real.cos theta - 3

def cartesian_equation_C2 (x y : ℝ) : Prop :=
(x - 1) ^ 2 + y ^ 2 = 4

def point_P := (Real.sqrt 2, Real.pi / 4)

theorem C1_C2_intersection_AB_length_and_product :
  (∀ t : ℝ, let (x, y) := parametric_C1 t in cartesian_equation_C2 x y → ∃ t1 t2 : ℝ,
    t1 + t2 = - Real.sqrt 2 ∧ t1 * t2 = -3) ∧ 
  (let AB_length := Real.sqrt (2 + 12) in AB_length = Real.sqrt 14) ∧
  (let PA_PB_product := |(-3 : ℝ)| in PA_PB_product = 3) :=
by sorry

end C1_C2_intersection_AB_length_and_product_l379_379535


namespace standing_next_to_Denis_l379_379316

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379316


namespace constant_term_expansion_l379_379113

theorem constant_term_expansion :
  (binomial 9 2 = 36) →
  let general_term (r : ℕ) := binomial 9 r * (9 ^ (9 - r)) * ((-1/3) ^ r) * (x ^ (9 - (3 * r / 2))) 
  let r := 6
  = 84 := by
  sorry

end constant_term_expansion_l379_379113


namespace sum_factorials_mod_31_l379_379727

theorem sum_factorials_mod_31 : 
  (∑ n in finset.range 51, n.factorial) % 31 = 29 :=
begin
  -- Proof goes here
  sorry
end

end sum_factorials_mod_31_l379_379727


namespace next_to_Denis_l379_379272

def student := {name : String}

def Borya : student := {name := "Borya"}
def Anya : student := {name := "Anya"}
def Vera : student := {name := "Vera"}
def Gena : student := {name := "Gena"}
def Denis : student := {name := "Denis"}

variables l : list student

axiom h₁ : l.head = Borya
axiom h₂ : ∃ i, l.get? i = some Vera ∧ (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ l.get? (i - 1) ≠ some Gena ∧ l.get? (i + 1) ≠ some Gena
axiom h₃ : ∀ i j, l.get? i ∈ [some Anya, some Gena, some Borya] → l.get? j ∈ [some Anya, some Gena, some Borya] → abs (i - j) ≠ 1

theorem next_to_Denis : ∃ i, l.get? i = some Denis → (l.get? (i - 1) = some Anya ∨ l.get? (i + 1) = some Anya) ∧ (l.get? (i - 1) = some Gena ∨ l.get? (i + 1) = some Gena) :=
sorry

end next_to_Denis_l379_379272


namespace find_m_l379_379838

theorem find_m (x y m : ℝ) :
  (∃ a b c d, (a ≠ 0 ∧ c ≠ 0) ∧ (x^2 - m * y^2 + 2 * x + 2 * y = 0)) → m = 1 :=
sorry

end find_m_l379_379838


namespace find_angle_C_max_sin_AB_l379_379873
noncomputable def problem_conditions (a b c S : ℝ) : Prop :=
S = a^2 + b^2 - c^2

theorem find_angle_C (a b c S : ℝ) (h : problem_conditions a b c S) : 
∃ C : ℝ, C = 60 := 
by sorry

theorem max_sin_AB (a b c S : ℝ) (h : problem_conditions a b c S) : 
∃ M : ℝ, M = 1 + (sqrt 3) / 2 := 
by sorry

end find_angle_C_max_sin_AB_l379_379873


namespace max_bus_capacity_l379_379124

-- Definitions and conditions
def left_side_regular_seats := 12
def left_side_priority_seats := 3
def right_side_regular_seats := 9
def right_side_priority_seats := 2
def right_side_wheelchair_space := 1
def regular_seat_capacity := 3
def priority_seat_capacity := 2
def back_row_seat_capacity := 7
def standing_capacity := 14

-- Definition of total bus capacity
def total_bus_capacity : ℕ :=
  (left_side_regular_seats * regular_seat_capacity) + 
  (left_side_priority_seats * priority_seat_capacity) + 
  (right_side_regular_seats * regular_seat_capacity) + 
  (right_side_priority_seats * priority_seat_capacity) + 
  back_row_seat_capacity + 
  standing_capacity

-- Theorem to prove
theorem max_bus_capacity : total_bus_capacity = 94 := by
  -- skipping the proof
  sorry

end max_bus_capacity_l379_379124


namespace problem_solution_l379_379104

theorem problem_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end problem_solution_l379_379104


namespace differentiable_limit_implies_derivative_eq_one_third_l379_379211

variable {α : Type*} [normed_field α] [complete_space α] (f : α → α) (x₀ : α)

theorem differentiable_limit_implies_derivative_eq_one_third
  (h_diff : differentiable_at α f x₀)
  (h_limit : filter.tendsto (λ Δx, (f (x₀ + 3 * Δx) - f x₀) / Δx) (nhds 0) (nhds 1)) :
  deriv f x₀ = 1 / 3 :=
by {
  sorry,
}

end differentiable_limit_implies_derivative_eq_one_third_l379_379211


namespace arithmetic_sequence_a4_l379_379805

theorem arithmetic_sequence_a4
    (a : ℕ → ℤ)
    (a_sum : ℤ := (a 1 + a 5) * 5 / 2)
    (S5 : a_sum = 35)
    (a5 : a 5 = 11)
    (a_formula : ∀ n : ℕ, a n = a 1 + (n - 1) * ((a 5 - a 1) / 4)) :
  a 4 = 9 :=
by
  simp [a_sum, S5, a5, a_formula]
  sorry

end arithmetic_sequence_a4_l379_379805


namespace who_is_next_to_denis_l379_379284

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379284


namespace standing_next_to_Denis_l379_379314

def students := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def positions (line : list string) := 
  ∀ (student : string), student ∈ students → ∃ (i : ℕ), line.nth i = some student

theorem standing_next_to_Denis (line : list string) 
  (h1 : ∃ (i : ℕ), line.nth i = some "Borya" ∧ i = 0)
  (h2 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth j = some "Vera" ∧ abs (i - j) = 1 ∧
       line.nth k = some "Gena" ∧ abs (j - k) ≠ 1)
  (h3 : ∃ (i j k : ℕ), line.nth i = some "Anya" ∧ line.nth k = some "Gena" ∧
       (abs (i - 0) ≥ 2 ∧ abs (k - 0) ≥ 2) ∧ abs (i - k) ≥ 2) :
  ∃ (a b : ℕ), line.nth a = some "Anya" ∧ line.nth b = some "Gena" ∧ 
  ∀ (d : ℕ), line.nth d = some "Denis" ∧ (abs (d - a) = 1 ∨ abs (d - b) = 1) :=
sorry

end standing_next_to_Denis_l379_379314


namespace number_of_solutions_depends_on_positions_l379_379186

-- Define the basic conditions
variables {l p q : Line}
variables {A B : Point}
variables {intersect_pairwise : ∀ (a b : Line), a ≠ b → ∃ (P : Point), Incidence P a ∧ Incidence P b }
variables {on_lines : Incidence A l ∧ Incidence B p}
variables {perp_bisect : Perpendicular q (LineSegment A B) ∧ Bisection q (LineSegment A B)}

-- Define the proof problem
theorem number_of_solutions_depends_on_positions :
  (IsBisector q (AngleBetween l p) → infinitely_many_solutions)
  ∧ (Parallel q (Bisector l p) ∧ ¬Coincident q (Bisector l p) → no_solution)
  ∧ ((¬IsBisector q (AngleBetween l p)) ∧ (¬(Parallel q (Bisector l p) ∧ ¬Coincident q (Bisector l p))) → unique_solution) :=
sorry

end number_of_solutions_depends_on_positions_l379_379186


namespace slope_angle_of_vertical_line_l379_379246

theorem slope_angle_of_vertical_line :
  let l := { p : ℝ × ℝ | p.1 = 2 } in
  let slope_angle := π / 2 in
  ∀ p₁ p₂ ∈ l, p₁.1 = p₂.1 → slope_angle = π / 2 :=
by
  intro l slope_angle p₁ p₂ h1 h2 h3
  sorry

end slope_angle_of_vertical_line_l379_379246


namespace nes_sale_price_l379_379993

noncomputable def price_of_nes
    (snes_value : ℝ)
    (tradein_rate : ℝ)
    (cash_given : ℝ)
    (change_received : ℝ)
    (game_value : ℝ) : ℝ :=
  let tradein_credit := snes_value * tradein_rate
  let additional_cost := cash_given - change_received
  let total_cost := tradein_credit + additional_cost
  let nes_price := total_cost - game_value
  nes_price

theorem nes_sale_price 
  (snes_value : ℝ)
  (tradein_rate : ℝ)
  (cash_given : ℝ)
  (change_received : ℝ)
  (game_value : ℝ) :
  snes_value = 150 → tradein_rate = 0.80 → cash_given = 80 → change_received = 10 → game_value = 30 →
  price_of_nes snes_value tradein_rate cash_given change_received game_value = 160 := by
  intros
  sorry

end nes_sale_price_l379_379993


namespace part1_solution_part2_solution_l379_379079

variables (x y m : ℤ)

-- Given the system of equations
def system_of_equations (x y m : ℤ) : Prop :=
  (2 * x - y = m) ∧ (3 * x + 2 * y = m + 7)

-- Part (1) m = 0, find x = 1, y = 2
theorem part1_solution : system_of_equations x y 0 → x = 1 ∧ y = 2 :=
sorry

-- Part (2) point A(-2,3) in the second quadrant with distances 3 and 2, find m = -7
def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

def distance_to_axes (x y dx dy : ℤ) : Prop :=
  y = dy ∧ x = -dx

theorem part2_solution : is_in_second_quadrant x y →
  distance_to_axes x y 2 3 →
  system_of_equations x y m →
  m = -7 :=
sorry

end part1_solution_part2_solution_l379_379079


namespace painted_cube_probability_l379_379010

-- Define the problem
variable (C : Type) [Fintype C] [DecidableEq C] (colors : Finset C)
variable (cube_faces : Fin 6 → C) -- The function that assigns a color to each face of the cube

-- Given each face of the cube can be independently colored as red, blue, or green each with a probability of 1/3
axiom cube_coloring : ∀ n, n ∈ cube_faces → colors.card = 3

-- Define what it means for the four vertical faces to be the same color
def four_vertical_faces_same (cube_faces : Fin 6 → C) : Prop := 
  ∃ c : C, ∀ i j : Fin 4, cube_faces ⟨i.1, by linarith⟩ = c ∧ cube_faces ⟨j.1, by linarith⟩ = c

-- Define the main theorem stating the proof problem
theorem painted_cube_probability (h : ∀ (c: C), c ∈ colors) : ( ∑ cube in (Finset.univ : Finset (Fin 6 → C)), 
  if four_vertical_faces_same cube then 1 else 0).to_real / (3^6) = 31 / 243 := sorry

end painted_cube_probability_l379_379010


namespace min_distance_from_C1_to_C2_l379_379534

-- Definitions based on the conditions
def line1 (t k : ℝ) : ℝ × ℝ := (t - real.sqrt 3, k * t)
def line2 (m k : ℝ) : ℝ × ℝ := (real.sqrt 3 - m, m / (3 * k))

-- C2 in Cartesian form as provided
def C2 (x y : ℝ) : Prop := x + y - 8 = 0

-- General equation of C1
def C1_general (x y : ℝ) : Prop := (x^2 / 3 + y^2 = 1) ∧ (y ≠ 0)

-- Parametric equations of C1
def C1_parametric (α : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.cos α, real.sin α)

-- Minimum distance from point Q on C1 to C2
def min_distance (p q : ℝ) : ℝ := 
  let d := (real.abs (real.sqrt 3 * real.cos p + real.sin p - 8)) / real.sqrt 2 in
  d

-- Main statement
theorem min_distance_from_C1_to_C2 (α : ℝ) (hα : α ≠ π * ↑(int.of_nat ℤ)) :
  C1_general (real.sqrt 3 * real.cos α) (real.sin α) →
  min_distance α α ≥ 3 * real.sqrt 2 :=
sorry

end min_distance_from_C1_to_C2_l379_379534


namespace sequence_first_20_sum_l379_379494

def sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) :=
  S 1 = 5 ∧ (∀ n ≥ 2, a n = S n - S (n - 1))

def term_seq (a : ℕ → ℤ) (b : ℕ → ℚ) :=
  ∀ n, b n = 1 / (a n * a (n + 1))

theorem sequence_first_20_sum :
  (∀ n, S n = 6 * n - n ^ 2) →
  sequence_sum S a →
  term_seq a b →
  (finset.range 20).sum (λ n, b (n + 1)) = -4 / 35 :=
by
  sorry

end sequence_first_20_sum_l379_379494


namespace roots_of_z4_plus_16_eq_0_l379_379764

noncomputable def roots_of_quartic_eq : Set ℂ :=
  { z | z^4 + 16 = 0 }

theorem roots_of_z4_plus_16_eq_0 :
  roots_of_quartic_eq = { z | z = complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 + complex.I * complex.sqrt 2 ∨
                             z = -complex.sqrt 2 - complex.I * complex.sqrt 2 ∨
                             z = complex.sqrt 2 - complex.I * complex.sqrt 2 } :=
by
  sorry

end roots_of_z4_plus_16_eq_0_l379_379764


namespace calculation_correct_l379_379661

theorem calculation_correct :
  ∀ (x : ℤ), -2 * (x + 1) = -2 * x - 2 :=
by
  intro x
  calc
    -2 * (x + 1) = -2 * x + -2 * 1 : by sorry
              ... = -2 * x - 2 : by sorry

end calculation_correct_l379_379661


namespace standard_lamp_probability_l379_379742

-- Define the given probabilities
def P_A1 : ℝ := 0.45
def P_A2 : ℝ := 0.40
def P_A3 : ℝ := 0.15

def P_B_given_A1 : ℝ := 0.70
def P_B_given_A2 : ℝ := 0.80
def P_B_given_A3 : ℝ := 0.81

-- Define the calculation for the total probability of B
def P_B : ℝ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

-- The statement to prove
theorem standard_lamp_probability : P_B = 0.7565 := by sorry

end standard_lamp_probability_l379_379742


namespace painting_methods_l379_379987

theorem painting_methods:
  ∃ (n : ℕ), n = 30 ∧ 
  (∀ (balls : list ℕ), balls.length = 8 →
  (∀ (b : ℕ), b ∈ balls → b = 0 ∨ b = 1) →
  (list.count balls 1 = 5) →
  (list.count balls 0 = 3) →
  (∃ (i : ℕ), 0 ≤ i ∧ i ≤ 5 ∧ balls.counts consecutively 1 3)) :=
sorry

end painting_methods_l379_379987


namespace metal_waste_l379_379375

theorem metal_waste (l b : ℝ) (h : l > b) : l * b - (b^2 / 2) = 
  (l * b - (π * (b / 2)^2)) + (π * (b / 2)^2 - (b^2 / 2)) := by
  sorry

end metal_waste_l379_379375


namespace altered_solution_ratio_l379_379961

theorem altered_solution_ratio (initial_bleach : ℕ) (initial_detergent : ℕ) (initial_water : ℕ) :
  initial_bleach / initial_detergent = 2 / 25 ∧
  initial_detergent / initial_water = 25 / 100 →
  (initial_detergent / initial_water) / 2 = 1 / 8 →
  initial_water = 300 →
  (300 / 8) = 37.5 := 
by 
  sorry

end altered_solution_ratio_l379_379961


namespace sale_in_fourth_month_l379_379694

def sale_in_five_months (first second third fourth fifth : ℕ) := first + second + third + fourth + fifth

theorem sale_in_fourth_month (first_month second_month third_month fifth_month : ℕ) (average_sale : ℕ) (total_months : ℕ) :
  total_months = 5 → -- There are five months
  average_sale = 7800 → -- The average sale over these months is 7800
  first_month = 5700 →
  second_month = 8550 →
  third_month = 6855 →
  fifth_month = 14045 →
  let total_sales := average_sale * total_months in
  let fourth_month := total_sales - (first_month + second_month + third_month + fifth_month) in
  fourth_month = 3850 :=
by intros; sorry

end sale_in_fourth_month_l379_379694


namespace part_a_part_b_l379_379160

noncomputable theory
open_locale classical

variables (p k : ℕ) (f g : polynomial ℤ)

def t (p k : ℕ) : ℕ := ∑ i in finset.range (nat.floor (log p k) + 1), k / p^i

-- Define polynomial properties
def polynomial_properties (p k : ℕ) (f : polynomial ℤ) : Prop :=
  f.degree = k ∧ f.leading_coeff = 1 ∧ p ∣ f.coeff 0

-- Statement (a)
theorem part_a (hp : nat.prime p) (hk : 0 < k) (hf : polynomial_properties p k f) :
  ∃ n : ℕ, p ∣ f.eval n ∧ ¬ p^(t p k + 1) ∣ f.eval n :=
sorry

-- Define properties for polynomial g
def polynomial_properties_g (p k : ℕ) (g : polynomial ℤ) : Prop :=
  g.degree = k ∧ g.leading_coeff = 1 ∧ p ∣ g.coeff 0

-- Statement (b)
theorem part_b (hp : nat.prime p) (hk : 0 < k) :
  ∃ g : polynomial ℤ, polynomial_properties_g p k g ∧ (∀ n : ℕ, p ∣ g.eval n → p^(t p k) ∣ g.eval n) :=
sorry

end part_a_part_b_l379_379160


namespace product_of_roots_is_neg4_l379_379167

-- Define the polynomial
def poly : Polynomial ℤ := 3 * Polynomial.X^4 - 8 * Polynomial.X^3 + Polynomial.X^2 + 4 * Polynomial.X - 12

-- Define the proof problem
theorem product_of_roots_is_neg4 : 
  let a := 3 in
  let e := -12 in
  (a, b, c, d : ℤ) -> 0 = poly.eval a -> Polynomial.eval (poly) b = 0 -> Polynomial.eval (poly) c = 0 -> Polynomial.eval (poly) d = 0 -> (a * b * c * d) = -4 :=
by
  sorry

end product_of_roots_is_neg4_l379_379167


namespace correct_answer_is_C_l379_379864

structure Point where
  x : ℤ
  y : ℤ

def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def A : Point := ⟨1, -1⟩
def B : Point := ⟨0, 2⟩
def C : Point := ⟨-3, 2⟩
def D : Point := ⟨4, 0⟩

theorem correct_answer_is_C : inSecondQuadrant C := sorry

end correct_answer_is_C_l379_379864


namespace sequence_100_l379_379962

noncomputable def sequence (a : ℕ → ℕ) (R : ℕ) : Prop :=
a 1 = R ∧ ∀ n ≥ 1, a (n + 1) = a n + 2 * n

theorem sequence_100 (a : ℕ → ℕ) (R S : ℕ) 
    (h1 : a 1 = R)
    (h2 : ∀ n ≥ 1, a (n + 1) = a n + 2 * n) :
    a 100 = R + 9900 := 
sorry

end sequence_100_l379_379962


namespace sophie_one_dollar_bills_l379_379928

theorem sophie_one_dollar_bills (x y z: ℕ) :
  x + y + z = 60 ∧ x + 2 * y + 5 * z = 175 → x = 5 :=
by 
  assume h : x + y + z = 60 ∧ x + 2 * y + 5 * z = 175
  -- Continues the formal proof if needed.
  sorry

end sophie_one_dollar_bills_l379_379928


namespace Denis_next_to_Anya_Gena_l379_379268

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379268


namespace max_teams_for_student_impossible_8_teams_l379_379364

-- Definitions based on problem conditions
axiom students : Type
axiom number_of_students : Fintype.card students = 10

def is_team (team : Finset students) : Prop :=
  team.card = 4

def team_intersection (team1 team2 : Finset students) : Prop :=
  (team1 ∩ team2).card = 1

-- Question (a): Prove that the maximum number of teams a student can participate in is 3
theorem max_teams_for_student (s : students) (teams : Finset (Finset students)) :
  (∀ t ∈ teams, is_team t) →
  (∀ t1 t2 ∈ teams, t1 ≠ t2 → team_intersection t1 t2) →
  (∃ (n : ℕ), n = 3 ∧ ∀ (team_set : Finset (Finset students)), team_set.card ≤ n) :=
sorry

-- Question (b): Prove that it is impossible to have 8 teams given the same conditions
theorem impossible_8_teams (teams : Finset (Finset students)) :
  (∀ t ∈ teams, is_team t) →
  (∀ t1 t2 ∈ teams, t1 ≠ t2 → team_intersection t1 t2) →
  (¬ teams.card = 8) :=
sorry

end max_teams_for_student_impossible_8_teams_l379_379364


namespace scientific_notation_correct_l379_379354

/-- diameter in meters -/
def willow_catkin_diameter : ℝ := 0.0000105

/-- diameter in scientific notation -/
def willow_catkin_diameter_scientific : ℝ := 1.05 * 10^(-5)

theorem scientific_notation_correct :
  willow_catkin_diameter = willow_catkin_diameter_scientific :=
sorry

end scientific_notation_correct_l379_379354


namespace circle_eq_l379_379226

theorem circle_eq (D E : ℝ) :
  (∀ {x y : ℝ}, (x = 0 ∧ y = 0) ∨
               (x = 4 ∧ y = 0) ∨
               (x = -1 ∧ y = 1) → 
               x^2 + y^2 + D * x + E * y = 0) →
  (D = -4 ∧ E = -6) :=
by
  intros h
  have h1 : 0^2 + 0^2 + D * 0 + E * 0 = 0 := by exact h (Or.inl ⟨rfl, rfl⟩)
  have h2 : 4^2 + 0^2 + D * 4 + E * 0 = 0 := by exact h (Or.inr (Or.inl ⟨rfl, rfl⟩))
  have h3 : (-1)^2 + 1^2 + D * (-1) + E * 1 = 0 := by exact h (Or.inr (Or.inr ⟨rfl, rfl⟩))
  sorry -- proof steps would go here to eventually show D = -4 and E = -6

end circle_eq_l379_379226


namespace largest_solution_correct_l379_379028

def floor_largest_solution : Real :=
  let fractional_part (x : Real) : Real := x - Real.floor x
  let equation (x : Real) : Prop := Real.floor x = 10 + 150 * fractional_part x
  let largest_solution : Real := 159.9933
  largest_solution

theorem largest_solution_correct :
  ∃ x : Real, equation x ∧ 
    (∀ y : Real, equation y → y ≤ x) ∧ 
    x = floor_largest_solution :=
by
  sorry

end largest_solution_correct_l379_379028


namespace cakes_remaining_l379_379401

theorem cakes_remaining (cakes_made : ℕ) (cakes_sold : ℕ) (h_made : cakes_made = 149) (h_sold : cakes_sold = 10) :
  (cakes_made - cakes_sold) = 139 :=
by
  cases h_made
  cases h_sold
  sorry

end cakes_remaining_l379_379401


namespace complement_angle_l379_379783

theorem complement_angle (A : ℝ) (h1 : A = 40) : ∃ B : ℝ, A + B = 90 ∧ B = 50 :=
by
  use 50
  split
  · rw h1
    norm_num
  · norm_num

end complement_angle_l379_379783


namespace intersection_ray_C1_find_m_l379_379863

-- Parametric definition of curve C1
def curve_C1 (α : ℝ) : ℝ × ℝ := (3 + sqrt 2 * cos α, 1 + sqrt 2 * sin α)

-- Equation of ray l in polar coordinates
def ray_l (θ : ℝ) (ρ : ℝ) : Prop := θ = π / 4 ∧ ρ ≥ 0

-- Equation of curve C2 in polar coordinates
def curve_C2 (θ ρ m : ℝ) : Prop := ρ * (sin θ + 2 * cos θ) = ρ^2 * cos θ^2 + m

-- Proving the number of common points between ray l and curve C1
theorem intersection_ray_C1 : ∃! P : ℝ × ℝ, 
  ∃ α : ℝ, curve_C1 α = P ∧ ∃ ρ : ℝ, ∃ θ : ℝ, ray_l θ ρ ∧ P = (ρ * cos θ, ρ * sin θ) :=
sorry

-- Proving the value of m given the roots of the equation from curve C2 and the condition |OA| = |AB|
theorem find_m (ρ1 ρ2 m : ℝ) (h_eqns : ρ1 + ρ2 = 3 * sqrt 2 ∧ ρ1 * ρ2 = 2 * m) 
  (h_condition : ρ2 = 2 * ρ1) : m = 2 :=
sorry

end intersection_ray_C1_find_m_l379_379863


namespace sarah_garden_area_l379_379531

theorem sarah_garden_area :
  (length width : ℝ) 
  (h1 : 1500 = 30 * length) 
  (h2 : 1500 = 12 * (2 * (length + width))) : 
  length * width = 625 :=
by
  sorry

end sarah_garden_area_l379_379531


namespace smallest_n_l379_379063

theorem smallest_n (n : ℕ) (h : 0 < n) : 
  (1 / (n : ℝ)) - (1 / (n + 1 : ℝ)) < 1 / 15 → n = 4 := sorry

end smallest_n_l379_379063


namespace combination_sum_l379_379730

theorem combination_sum :
  (Nat.choose 7 4) + (Nat.choose 7 3) = 70 := by
  sorry

end combination_sum_l379_379730


namespace who_is_next_to_denis_l379_379278

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379278


namespace sequence_starts_to_become_less_than_zero_from_10th_term_sum_of_specific_terms_of_sequence_is_minus_20_l379_379483

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

theorem sequence_starts_to_become_less_than_zero_from_10th_term :
  ∀ n, n ≥ 10 → arithmetic_sequence 25 (-3) n < 0 := 
by 
  intros n hn
  simp [arithmetic_sequence]
  sorry

theorem sum_of_specific_terms_of_sequence_is_minus_20 :
  ∑ i in (range 10).map (λ n => 2 * n + 1), arithmetic_sequence 25 (-3) i = -20 := 
by 
  sorry

end sequence_starts_to_become_less_than_zero_from_10th_term_sum_of_specific_terms_of_sequence_is_minus_20_l379_379483


namespace inequality_abc_l379_379901

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
sorry

end inequality_abc_l379_379901


namespace coins_donated_l379_379597

theorem coins_donated (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (coins_left : ℕ) : 
  pennies = 42 ∧ nickels = 36 ∧ dimes = 15 ∧ coins_left = 27 → (pennies + nickels + dimes - coins_left) = 66 :=
by
  intros h
  sorry

end coins_donated_l379_379597


namespace observation_count_l379_379319

theorem observation_count (mean_before mean_after : ℝ) 
  (wrong_value : ℝ) (correct_value : ℝ) (n : ℝ) :
  mean_before = 36 →
  correct_value = 60 →
  wrong_value = 23 →
  mean_after = 36.5 →
  n = 74 :=
by
  intros h_mean_before h_correct_value h_wrong_value h_mean_after
  sorry

end observation_count_l379_379319


namespace average_age_of_two_women_l379_379935

theorem average_age_of_two_women (A W1 W2 : ℝ) :
  (∀ (age_diff : ℝ), 7 * (A + 4) = 7 * A - 26 - 30 + W1 + W2) →
  (W1 + W2) / 2 = 42 :=
begin
  assume h,
  have h1 : 7 * (A + 4) = 7 * A + 28, 
  { linarith },
  rw h at h1,
  linarith,
  sorry
end

end average_age_of_two_women_l379_379935


namespace rakesh_gross_salary_before_tax_l379_379200

variable (S : ℝ)
variable (net_salary_after_tax fd_amount remaining_amount groceries_expenses utilities_expenses vacation_expense total_expenses : ℝ)

def deductions_and_expenses : Prop :=
  net_salary_after_tax = S * 0.93 ∧
  fd_amount = S * 0.15 + 200 ∧
  remaining_amount = net_salary_after_tax - fd_amount ∧
  groceries_expenses = 0.30 * remaining_amount ∧        -- Groceries expenses might fluctuate by 2%, not covered here directly for simplicity.
  utilities_expenses = 0.20 * remaining_amount ∧
  vacation_expense = 0.05 * remaining_amount ∧
  total_expenses = groceries_expenses + utilities_expenses + vacation_expense + 1500

theorem rakesh_gross_salary_before_tax (h : deductions_and_expenses S net_salary_after_tax fd_amount remaining_amount groceries_expenses utilities_expenses vacation_expense total_expenses) :
  remaining_amount - total_expenses = 2380 → S ≈ 11305.41 :=
by
  intros h391
  sorry

end rakesh_gross_salary_before_tax_l379_379200


namespace tower_remainder_l379_379378

def edge_lengths : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def valid_tower (tower : List ℕ) : Prop :=
  ∀ (i : ℕ), (i < tower.length - 1) → (tower[i + 1] = tower[i] + 1 ∨ tower[i + 1] = tower[i] + 2)

def all_towers : List (List ℕ) :=
  tower_list edge_lengths where
    tower_list [] := [[]]
    tower_list (h :: t) :=
      let previous_towers := tower_list t
      previous_towers ++ (previous_towers.bind (λ x, 
        [x ++ [h]]))

def valid_towers : List (List ℕ) :=
  all_towers.filter valid_tower

def T : ℕ := valid_towers.length

theorem tower_remainder : T % 100 = 32 :=
  by
    sorry

end tower_remainder_l379_379378


namespace polar_coordinates_of_point_l379_379244

theorem polar_coordinates_of_point (x y : ℝ) (h : x = 1 ∧ y = -sqrt 3) :
  ∃ ρ θ : ℝ, ρ = 2 ∧ θ = -π / 3 ∧ (x = ρ * cos θ ∧ y = ρ * sin θ) :=
by
  sorry

end polar_coordinates_of_point_l379_379244


namespace fraction_students_received_Bs_l379_379517

theorem fraction_students_received_Bs (fraction_As : ℝ) (fraction_As_or_Bs : ℝ) (h1 : fraction_As = 0.7) (h2 : fraction_As_or_Bs = 0.9) :
  fraction_As_or_Bs - fraction_As = 0.2 :=
by
  sorry

end fraction_students_received_Bs_l379_379517


namespace part1_part2_l379_379678

-- Define the sequences and initial point
def P (a b : ℕ → ℝ) (n : ℕ) : Prop :=
  (n > 0) → 
  (a (n+1) = a n * b (n+1)) ∧ 
  (b (n+1) = b n / (1 - 4 * (a n)^2))

-- Initial Point
def P1 : Prop := (a 1 = 1) ∧ (b 1 = -1)

-- Line equation
def line_eq (x y : ℝ) : Prop := (3 * y + 2 * x = -1)

-- Proof that Pn lies on line
theorem part1 : 
  (∀ n : ℕ, n > 0 → P a b n) → P1 → ∀ n, n > 0 → line_eq (a n) (b n) :=
by
  intros
  sorry

-- Definition for inequality condition
def ineq (a b : ℕ → ℝ) (n k : ℝ) : Prop :=
  (1 + a 1) * (1 + a 2) * ... * (1 + a n) ≥ k / √(b 2 * b 3 * ... * b (n+1))

-- Proof for the inequality condition
theorem part2 : 
  (∀ n : ℕ, n > 0 → P a b n) → P1 → ineq a b n (4 * √3 / 9) :=
by
  intros
  sorry

end part1_part2_l379_379678


namespace simplify_and_evaluate_expression_l379_379203

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1/2) (h2 : y = -2) :
  ((x + 2 * y) ^ 2 - (x + y) * (x - y)) / (2 * y) = -4 := by
  sorry

end simplify_and_evaluate_expression_l379_379203


namespace contest_score_difference_l379_379856

theorem contest_score_difference :
  let P60 := 0.15
  let P75 := 0.20
  let P80 := 0.25
  let P85 := 0.10
  let P90 := 1 - (P60 + P75 + P80 + P85)
  let num_contestants := 20
  let num_60 := P60 * num_contestants
  let num_75 := P75 * num_contestants
  let num_80 := P80 * num_contestants
  let num_85 := P85 * num_contestants
  let num_90 := P90 * num_contestants
  let scores := List.replicate num_60 60 ++ List.replicate num_75 75 ++ List.replicate num_80 80 ++ List.replicate num_85 85 ++ List.replicate num_90 90
  let median := 80
  let mean := (60 * num_60 + 75 * num_75 + 80 * num_80 + 85 * num_85 + 90 * num_90) / num_contestants
  (median - mean) = 0.5 := sorry

end contest_score_difference_l379_379856


namespace find_C_l379_379171

noncomputable def A : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a}
def isSolutionC (C : Set ℝ) : Prop := C = {2, 3}

theorem find_C : ∃ C : Set ℝ, isSolutionC C ∧ ∀ a, (A ∪ B a = A) ↔ a ∈ C :=
by
  sorry

end find_C_l379_379171


namespace peanut_butter_sandwich_days_l379_379148

theorem peanut_butter_sandwich_days 
  (H : ℕ)
  (total_days : ℕ)
  (probability_ham_and_cake : ℚ)
  (ham_probability : ℚ)
  (cake_probability : ℚ)
  (Ham_days : H = 3)
  (Total_days : total_days = 5)
  (Ham_probability_val : ham_probability = H / 5)
  (Cake_probability_val : cake_probability = 1 / 5)
  (Probability_condition : ham_probability * cake_probability = 0.12) :
  5 - H = 2 :=
by 
  sorry

end peanut_butter_sandwich_days_l379_379148


namespace red_tint_percentage_modified_mixture_l379_379718

-- Definitions based on conditions:
def original_volume : ℝ := 40
def original_red_tint_percentage : ℝ := 35 / 100
def added_red_tint : ℝ := 10

-- Required proof statement:
theorem red_tint_percentage_modified_mixture :
  let original_red_tint := original_volume * original_red_tint_percentage
  let total_red_tint := original_red_tint + added_red_tint
  let modified_volume := original_volume + added_red_tint
  (total_red_tint / modified_volume) * 100 = 48 := by
  sorry

end red_tint_percentage_modified_mixture_l379_379718


namespace denis_neighbors_l379_379289

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379289


namespace native_answer_l379_379349

-- Define properties to represent native types
inductive NativeType
| normal
| zombie
| half_zombie

-- Define the function that determines the response of a native
def response (native : NativeType) : String :=
  match native with
  | NativeType.normal => "да"
  | NativeType.zombie => "да"
  | NativeType.half_zombie => "да"

-- Define the main theorem
theorem native_answer (native : NativeType) : response native = "да" :=
by sorry

end native_answer_l379_379349


namespace juice_cost_l379_379120

theorem juice_cost (J : ℝ) (h1 : 15 * 3 + 25 * 1 + 12 * J = 88) : J = 1.5 :=
by
  sorry

end juice_cost_l379_379120


namespace range_of_ϕ_l379_379812

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (2 * x + ϕ) + 1

theorem range_of_ϕ (ϕ : ℝ) (h1 : abs ϕ ≤ Real.pi / 2) 
    (h2 : ∀ (x : ℝ), -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ϕ > 1) :
  Real.pi / 6 ≤ ϕ ∧ ϕ ≤ Real.pi / 3 :=
sorry

end range_of_ϕ_l379_379812


namespace complex_div_conjugate_l379_379073

theorem complex_div_conjugate (z : ℂ) (h : (1 + complex.i) * z = 2) : z = 1 - complex.i := 
by 
  sorry

end complex_div_conjugate_l379_379073


namespace joy_fourth_rod_selection_l379_379553

def lengths : List ℕ := List.range (50 / 2 + 1) |>.map (· * 2)

def used_rods : List ℕ := [5, 10, 21]

def valid_rods_for_quadrilateral (lengths used_rods: List ℕ) : List ℕ :=
  lengths.filter (λ l => 6 < l ∧ l < 36 ∧ ¬ used_rods.contains l)

theorem joy_fourth_rod_selection :
  valid_rods_for_quadrilateral lengths used_rods = 13 := 
sorry

end joy_fourth_rod_selection_l379_379553


namespace cube_root_neg_64_l379_379223

theorem cube_root_neg_64 : real.cbrt (-64) = -4 :=
by
  sorry

end cube_root_neg_64_l379_379223


namespace pages_remaining_l379_379956

def total_pages : ℕ := 120
def science_project_pages : ℕ := (25 * total_pages) / 100
def math_homework_pages : ℕ := 10
def total_used_pages : ℕ := science_project_pages + math_homework_pages
def remaining_pages : ℕ := total_pages - total_used_pages

theorem pages_remaining : remaining_pages = 80 := by
  sorry

end pages_remaining_l379_379956


namespace tangerine_in_third_row_l379_379777

-- Define datatype for Row and Tree types
inductive Row : Type
| first | second | third | fourth | fifth

inductive Tree : Type
| apple
| pear
| orange
| lemon
| tangerine

-- Define adjacency relation
def adjacent (r1 r2 : Row) : Prop :=
  (r1 = Row.first ∧ r2 = Row.second) ∨
  (r1 = Row.second ∧ r2 = Row.first) ∨
  (r1 = Row.second ∧ r2 = Row.third) ∨
  (r1 = Row.third ∧ r2 = Row.second) ∨
  (r1 = Row.third ∧ r2 = Row.fourth) ∨
  (r1 = Row.fourth ∧ r2 = Row.third) ∨
  (r1 = Row.fourth ∧ r2 = Row.fifth) ∨
  (r1 = Row.fifth ∧ r2 = Row.fourth)

-- Define tree rows
variables (row_of : Tree → Row)

-- Conditions
axiom orange_lemon_adjacent : adjacent (row_of Tree.orange) (row_of Tree.lemon)
axiom pear_not_adj_orange_lemon : ¬adjacent (row_of Tree.pear) (row_of Tree.orange) ∧ ¬adjacent (row_of Tree.pear) (row_of Tree.lemon)
axiom apple_next_pear_not_orange_lemon : adjacent (row_of Tree.apple) (row_of Tree.pear) ∧ ¬adjacent (row_of Tree.apple) (row_of Tree.orange) ∧ ¬adjacent (row_of Tree.apple) (row_of Tree.lemon)

-- Proof statement: Tangerine tree must be in the third row
theorem tangerine_in_third_row : row_of Tree.tangerine = Row.third :=
  sorry

end tangerine_in_third_row_l379_379777


namespace ellipse_semi_minor_axis_is_2_sqrt_3_l379_379860

/-- 
  Given an ellipse with the center at (2, -1), 
  one focus at (2, -3), and one endpoint of a semi-major axis at (2, 3), 
  we prove that the semi-minor axis is 2√3.
-/
theorem ellipse_semi_minor_axis_is_2_sqrt_3 :
  let center := (2, -1)
  let focus := (2, -3)
  let endpoint := (2, 3)
  let c := Real.sqrt ((2 - 2)^2 + (-3 + 1)^2)
  let a := Real.sqrt ((2 - 2)^2 + (3 + 1)^2)
  let b2 := a^2 - c^2
  let b := Real.sqrt b2
  c = 2 ∧ a = 4 ∧ b = 2 * Real.sqrt 3 := 
by
  sorry

end ellipse_semi_minor_axis_is_2_sqrt_3_l379_379860


namespace students_with_all_three_pets_l379_379852

variables (TotalStudents HaveDogs HaveCats HaveOtherPets NoPets x y z w : ℕ)

theorem students_with_all_three_pets :
  TotalStudents = 40 →
  HaveDogs = 20 →
  HaveCats = 16 →
  HaveOtherPets = 8 →
  NoPets = 7 →
  x = 12 →
  y = 3 →
  z = 11 →
  TotalStudents - NoPets = 33 →
  x + y + w = HaveDogs →
  z + w = HaveCats →
  y + w = HaveOtherPets →
  x + y + z + w = 33 →
  w = 5 :=
by
  intros h1 h2 h3 h4 h5 hx hy hz h6 h7 h8 h9
  sorry

end students_with_all_three_pets_l379_379852


namespace trapezoid_loci_l379_379469

-- Define the oriented line and fixed point A
variables {A : Point} {Δ : Line}

-- Define the conditions for the trapezoid ABCD
variables {a l : ℝ} -- fixed constants
variables {B C D E F H : Point}
-- E, F are midpoints of AB and CD respectively
def is_midpoint (X Y Z : Point) : Prop := 
  2 * vector.sum (to_vec X - to_vec Z) = to_vec Y - to_vec Z

-- Conditions for the trapezoid
def trapezoid_conditions (A B C D E F : Point) (Δ : Line) (a l : ℝ) : Prop :=
  |AB| ≤ a ∧
  |EF| = l ∧
  sum_squares_nonparallel_sides ABCD is_constant ∧
  on_line AB Δ

-- Define the loci
def is_on_segment (X Y : Point) (Δ : Line) : Prop := 
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ X = t * Y + (1 - t) * A

def is_on_circle (X H : Point) (l : ℝ) : Prop := 
  vector.norm (to_vec X - to_vec H) = l

def is_in_rect_region (C H : Point) (l a : ℝ) : Prop :=
  ∃ x, is_on_circle x H l ∧ 
  ∃ y, is_on_circle y H ("mirrored point logic here") ∧
  within_bounds (rect_bounds logic here)

-- The proof problem itself
theorem trapezoid_loci (A B C D E F H : Point) (Δ : Line) (a l : ℝ) :
  (trapezoid_conditions A B C D E F Δ a l) →
  (is_on_segment B A Δ) ∧
  (is_on_circle D H l) ∧
  (is_in_rect_region C H l a) :=
  sorry

end trapezoid_loci_l379_379469


namespace log_simplification_l379_379487

theorem log_simplification (p q r s u z : ℝ) 
  (hpq : p > 0) (hqr : q > 0) (hrr : r > 0) (hsr : s > 0) (hpr : p ≠ 0) (hqr : q ≠ 0) (hrr : r ≠ 0) (hsr : s ≠ 0)
  (hpu : p ≥ 1) (hqu : q ≥ 1) (hru : r ≥ 1) (hsu : s ≥ 1) (huu : u ≠ 0) (hzu : z ≠ 0) :
  log (p/q) + log (q/r) + log (r/s) - log (pz/(su)) = log (u/z) :=
by
  -- as instructed, we skip the proof details
  sorry

end log_simplification_l379_379487


namespace heat_required_l379_379963

theorem heat_required (m : ℝ) (c₀ : ℝ) (alpha : ℝ) (t₁ t₂ : ℝ) :
  m = 2 ∧ c₀ = 150 ∧ alpha = 0.05 ∧ t₁ = 20 ∧ t₂ = 100 →
  let Δt := t₂ - t₁
  let c_avg := (c₀ * (1 + alpha * t₁) + c₀ * (1 + alpha * t₂)) / 2
  let Q := c_avg * m * Δt
  Q = 96000 := by
  sorry

end heat_required_l379_379963


namespace combined_area_of_three_walls_l379_379991

theorem combined_area_of_three_walls (A : ℝ) :
  (A - 2 * 30 - 3 * 45 = 180) → (A = 375) :=
by
  intro h
  sorry

end combined_area_of_three_walls_l379_379991


namespace correct_propositions_l379_379392

theorem correct_propositions 
  (L : Type) [linear_order L] -- To denote lines
  (P : Type) [linear_order P] -- To denote planes
  (line_perpendicular_to_plane : P → L → Prop) -- Represents a line perpendicular to a plane
  (plane_perpendicular_to_line : L → P → Prop) -- Represents a plane perpendicular to a line
  (plane_parallel_to_plane : P → P → Prop) -- Represents whether two planes are parallel
  (line_parallel_to_line : L → L → Prop) -- Represents whether two lines are parallel
  (prop1 : ∀ (l1 l2 : L) (p : P), plane_perpendicular_to_line l1 p → plane_perpendicular_to_line l2 p → l1 ≠ l2 → ¬ line_parallel_to_line l1 l2) -- Proposition 1 condition
  (prop2 : ∀ (p1 p2 : P) (l : L), plane_perpendicular_to_line l p1 → plane_perpendicular_to_line l p2 →  p1 ≠ p2 → plane_parallel_to_plane p1 p2) -- Proposition 2 condition
  (prop3 : ∀ (l1 l2 : L) (p : P), line_perpendicular_to_plane p l1 → line_perpendicular_to_plane p l2 → l1 ≠ l2 → line_parallel_to_line l1 l2) -- Proposition 3 condition
  (prop4 : ∀ (p1 p2 p3 : P), plane_perpendicular_to_plane p1 p2 → plane_perpendicular_to_plane p1 p3 → p2 ≠ p3 → (plane_parallel_to_plane p2 p3 ∨ ¬plane_parallel_to_plane p2 p3)) -- Proposition 4 condition
  : (prop2 ∧ prop3) ∧ ¬prop1 ∧ ¬prop4 :=
by
  sorry

end correct_propositions_l379_379392


namespace tricycles_count_l379_379370

-- Define the variables for number of bicycles, tricycles, and scooters.
variables (b t s : ℕ)

-- Define the total number of children and total number of wheels conditions.
def children_condition := b + t + s = 10
def wheels_condition := 2 * b + 3 * t + 2 * s = 27

-- Prove that number of tricycles t is 4 under these conditions.
theorem tricycles_count : children_condition b t s → wheels_condition b t s → t = 4 := by
  sorry

end tricycles_count_l379_379370


namespace monotonic_increasing_a_l379_379492

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (2 - a * x^2)

theorem monotonic_increasing_a (a : ℝ) (h : ∀ x y ∈ Icc (0 : ℝ) 2, x < y → f a x < f a y) : a ∈ Ioo (1/2) 1 :=
sorry

end monotonic_increasing_a_l379_379492


namespace triangle_area_l379_379934

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def has_perimeter (a b c p : ℝ) : Prop :=
  a + b + c = p

def has_altitude (base side altitude : ℝ) : Prop :=
  (base / 2) ^ 2 + altitude ^ 2 = side ^ 2

def area_of_triangle (a base altitude : ℝ) : ℝ :=
  0.5 * base * altitude

theorem triangle_area (a b c : ℝ)
  (h_iso : is_isosceles a b c)
  (h_p : has_perimeter a b c 40)
  (h_alt : has_altitude (2 * a) b 12) :
  area_of_triangle a (2 * a) 12 = 76.8 :=
by
  sorry

end triangle_area_l379_379934


namespace largest_common_divisor_l379_379714

theorem largest_common_divisor (d h m s : ℕ) : 
  40 ∣ (1000000 * d + 10000 * h + 100 * m + s - (86400 * d + 3600 * h + 60 * m + s)) :=
by
  sorry

end largest_common_divisor_l379_379714


namespace quadratic_root_zero_l379_379454

theorem quadratic_root_zero (a : ℝ) : 
  ((a-1) * 0^2 + 0 + a^2 - 1 = 0) 
  → a ≠ 1 
  → a = -1 := 
by
  intro h1 h2
  sorry

end quadratic_root_zero_l379_379454


namespace max_stamps_with_budget_l379_379114

theorem max_stamps_with_budget
    (price_per_stamp : ℕ)
    (discount_threshold : ℕ)
    (discount_rate : ℚ)
    (budget : ℕ)
    (price_per_stamp_eq : price_per_stamp = 35)
    (discount_threshold_eq : discount_threshold = 100)
    (discount_rate_eq : discount_rate = 0.95)
    (budget_eq : budget = 3200) :
  let cost_without_discount (n : ℕ) := price_per_stamp * n
  let cost_with_discount (n : ℕ) := (price_per_stamp * discount_rate).natAbs * n
  max_stamps := 
    if  cost_without_discount (budget / price_per_stamp).natAbs  ≤ budget then 
        (budget / price_per_stamp).natAbs 
    else 
        nat.min  (discount_threshold-1)  (budget / cost_with_discount ).natAbs in
   nat.min budget / price_per_stamp 
    = 91 :=
by
  sorry

end max_stamps_with_budget_l379_379114


namespace translation_of_graph_l379_379947

theorem translation_of_graph (f : ℝ → ℝ) (x : ℝ) :
  f x = 2 ^ x →
  f (x - 1) + 2 = 2 ^ (x - 1) + 2 :=
by
  intro
  sorry

end translation_of_graph_l379_379947


namespace DianaHourlyRateIsCorrect_l379_379430

noncomputable def Diana_regular_hourly_rate : ℝ :=
  let total_hours_regular := 10 + 15 + 15 + 10 : ℝ
  let total_hours_double := 10 : ℝ
  let total_earnings := 1800 : ℝ
  let additional_earnings_saturday := 200 : ℝ
  let performance_bonus := 150 : ℝ
  let total_earnings_from_work := total_earnings - additional_earnings_saturday - performance_bonus
  let earnings_equation := 50 * R + 20 * R = total_earnings_from_work
  sorry

theorem DianaHourlyRateIsCorrect : Diana_regular_hourly_rate = 20.71 := sorry

end DianaHourlyRateIsCorrect_l379_379430


namespace find_x_find_min_norm_c_l379_379784

section Problem
open Real

def vector_a : ℝ × ℝ := (3, -1)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

variable (b : ℝ × ℝ)
variable (x : ℝ)

def vector_c : ℝ × ℝ := (x * vector_a.1 + (1 - x) * b.1, x * vector_a.2 + (1 - x) * b.2)

-- Condition: vector_a ⊥ vector_c
def a_perp_c : Prop := dot_product vector_a (vector_c b x) = 0

-- Given that a ⊥ c, prove that x = 1/3
theorem find_x (h : a_perp_c b x) : x = 1/3 := sorry

-- Additional condition
variable (vector_b : ℝ × ℝ)
def norm (u : ℝ × ℝ) : ℝ := sqrt (u.1^2 + u.2^2)

-- Given norm of b
def norm_b : Prop := norm vector_b = sqrt 5

-- Given norm condition, prove minimum norm of c
theorem find_min_norm_c (h_b : norm_b vector_b) : ∃ x, norm (vector_c vector_b x) = 1 := sorry

end find_x_find_min_norm_c_l379_379784


namespace Denis_next_to_Anya_Gena_l379_379267

inductive Person
| Anya
| Borya
| Vera
| Gena
| Denis

open Person

def isNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ∃ n, line.nth n = some p1 ∧ line.nth (n + 1) = some p2 ∨ line.nth (n - 1) = some p2

def notNextTo (p1 p2 : Person) (line : List Person) : Prop :=
  ¬ isNextTo p1 p2 line

theorem Denis_next_to_Anya_Gena :
  ∀ (line : List Person),
    line.length = 5 →
    line.head = some Borya →
    isNextTo Vera Anya line →
    notNextTo Vera Gena line →
    notNextTo Anya Borya line →
    notNextTo Gena Borya line →
    notNextTo Anya Gena line →
    isNextTo Denis Anya line ∧ isNextTo Denis Gena line :=
by
  intros line length_five borya_head vera_next_anya vera_not_next_gena anya_not_next_borya gena_not_next_borya anya_not_next_gena
  sorry

end Denis_next_to_Anya_Gena_l379_379267


namespace hyperbola_eccentricity_l379_379944

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ), a = 3 → b = 4 → c = Real.sqrt (a^2 + b^2) → c / a = 5 / 3 :=
by
  intros a b c ha hb h_eq
  sorry

end hyperbola_eccentricity_l379_379944


namespace who_is_next_to_Denis_l379_379259

-- Definitions according to conditions
def SchoolChildren := ["Anya", "Borya", "Vera", "Gena", "Denis"]

def Position : Type := ℕ

def is_next_to (a b : Position) := abs (a - b) = 1

def valid_positions (positions : SchoolChildren → Position) :=
  positions "Borya" = 1 ∧
  is_next_to (positions "Vera") (positions "Anya") ∧
  ¬is_next_to (positions "Vera") (positions "Gena") ∧
  ¬is_next_to (positions "Anya") (positions "Borya") ∧
  ¬is_next_to (positions "Gena") (positions "Borya") ∧
  ¬is_next_to (positions "Anya") (positions "Gena")

theorem who_is_next_to_Denis:
  ∃ positions : SchoolChildren → Position, valid_positions positions ∧ 
  (is_next_to (positions "Denis") (positions "Anya") ∧
   is_next_to (positions "Denis") (positions "Gena")) :=
by
  sorry

end who_is_next_to_Denis_l379_379259


namespace line_through_center_parallel_to_given_line_l379_379427

def point_in_line (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  -a / b

theorem line_through_center_parallel_to_given_line :
  ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -4 ∧
    point_in_line (2, 0) a b c ∧
    slope_of_line a b c = slope_of_line 2 (-1) 1 :=
by
  sorry

end line_through_center_parallel_to_given_line_l379_379427


namespace hyperbola_equation_l379_379625

theorem hyperbola_equation (x y : ℝ) :
  (∃ c : ℝ, c^2 = 25 - 16 ∧ 
  ∃ a : ℝ, a = real.sqrt 3 ∧ 
  ∃ b : ℝ, b^2 = 9 - 3 ∧ 
  ∃ P : ℝ × ℝ, P = (2, real.sqrt 2) ∧ 
  x^2 / a^2 - y^2 / b^2 = 1 ∧ 
  (x, y) = P) -> 
  (x^2 / 3 - y^2 / 6 = 1) := 
sorry

end hyperbola_equation_l379_379625


namespace minimize_AP_BP_l379_379884

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨2, 0⟩
def B : Point := ⟨6, 5⟩
def parabola (P : Point) : Prop := P.y ^ 2 = 8 * P.x

def distance (P Q : Point) : ℝ :=
  (Real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2))

theorem minimize_AP_BP (P : Point) (hP : parabola P) :
  ∃ P, parabola P ∧ distance A P + distance B P = 8 := sorry

end minimize_AP_BP_l379_379884


namespace hyperbola_equiv_l379_379033

-- The existing hyperbola
def hyperbola1 (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- The new hyperbola with same asymptotes passing through (2, 2) should have this form
def hyperbola2 (x y : ℝ) : Prop := (x^2 / 3 - y^2 / 12 = 1)

theorem hyperbola_equiv (x y : ℝ) :
  (hyperbola1 2 2) →
  (y^2 / 4 - x^2 / 4 = -3) →
  (hyperbola2 x y) :=
by
  intros h1 h2
  sorry

end hyperbola_equiv_l379_379033


namespace area_of_triangle_KLM_l379_379346

theorem area_of_triangle_KLM
  (O : Point)
  (S A B C K L M : Point)
  (sphere_centered_at_O : Sphere O)
  (touch_points : sphere_centered_at_O.touches_edges [S A, S B, S C] [K, L, M] ∧
                  sphere_centered_at_O.touches_base (Plane A B C))
  (tangent_plane_section_area : tangent_plane_through_closest_point_to S sphere_centered_at_O = 5)
  (angle_KSO : ∠ K S O = Real.arccos (sqrt 21 / 5))
  : triangle_area K L M = 9.8 :=
sorry

end area_of_triangle_KLM_l379_379346


namespace base9_first_digit_is_4_l379_379217

-- Define the base three representation of y
def y_base3 : Nat := 112211

-- Function to convert a given number from base 3 to base 10
def base3_to_base10 (n : Nat) : Nat :=
  let rec convert (n : Nat) (acc : Nat) (place : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * (3 ^ place)) (place + 1)
  convert n 0 0

-- Compute the base 10 representation of y
def y_base10 : Nat := base3_to_base10 y_base3

-- Function to convert a given number from base 10 to base 9
def base10_to_base9 (n : Nat) : List Nat :=
  let rec convert (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc
    else convert (n / 9) ((n % 9) :: acc)
  convert n []

-- Compute the base 9 representation of y as a list of digits
def y_base9 : List Nat := base10_to_base9 y_base10

-- Get the first digit (most significant digit) of the base 9 representation of y
def first_digit_base9 (digits : List Nat) : Nat :=
  digits.headD 0

-- The statement to prove
theorem base9_first_digit_is_4 : first_digit_base9 y_base9 = 4 := by sorry

end base9_first_digit_is_4_l379_379217


namespace increasing_interval_height_max_l379_379486

noncomputable def f (k x : ℝ) : ℝ := -2 * Real.sin (2 * x - Real.pi / 3)

theorem increasing_interval (k : ℤ) : 
  ∀ x, (f k (5 * Real.pi / 12 + k * Real.pi) ≤ f k x) ∧ (f k x ≤ f k (11 * Real.pi / 12 + k * Real.pi)) := 
by sorry

structure Triangle :=
(A B C : ℝ)
(a b c : ℝ)

noncomputable def height_on_a (t : Triangle) (k : ℝ) : ℝ :=
  (1 / 2) * Real.sqrt (1 + k^2) * (4 / Real.sqrt (1 + 4 * k^2)) * (|k - 1| / Real.sqrt (k^2 + 1))

theorem height_max (A : ℝ) (a : ℝ) (f_A : ℝ) (b c : ℝ) (k : ℝ) :
  acute_triangle A a b c →
  A = Real.pi / 3 →
  a = 3 →
  f_A = -Real.sqrt 3 →
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) →
  height_on_a {A := A, B := 0, C := 0, a := a, b := b, c := c} k ≤ 3 * Real.sqrt 3 / 2 := 
by sorry

end increasing_interval_height_max_l379_379486


namespace identify_pyramid_scheme_l379_379646

-- Definitions for the individual conditions
def high_returns (investment_opportunity : Prop) : Prop := 
  ∃ significantly_higher_than_average_returns : Prop, investment_opportunity = significantly_higher_than_average_returns

def lack_of_information (company : Prop) : Prop := 
  ∃ incomplete_information : Prop, company = incomplete_information

def aggressive_advertising (advertising : Prop) : Prop := 
  ∃ aggressive_ad : Prop, advertising = aggressive_ad

-- Main definition combining all conditions
def is_financial_pyramid_scheme (investment_opportunity company advertising : Prop) : Prop :=
  high_returns investment_opportunity ∧ lack_of_information company ∧ aggressive_advertising advertising

-- Theorem statement
theorem identify_pyramid_scheme 
  (investment_opportunity company advertising : Prop) 
  (h1 : high_returns investment_opportunity)
  (h2 : lack_of_information company)
  (h3 : aggressive_advertising advertising) : 
  is_financial_pyramid_scheme investment_opportunity company advertising :=
by 
  apply And.intro;
  {
    exact h1,
    apply And.intro;
    {
      exact h2,
      exact h3,
    }
  }

end identify_pyramid_scheme_l379_379646


namespace eccentricity_of_ellipse_l379_379249

-- Definitions
variables (a b c e : ℝ) (h1 : a > b) (h2 : b > 0)

-- Conditions
def ellipse_eq := ∀ x y: ℝ, (x^2 / a^2 + y^2 / b^2 = 1)
def vertex_A := (a, 0)
def vertex_B := (0, b)
def focus_F := (-c, 0)
def angle_B_eq_90 := ∠ ((-c, 0), (0, b), (a, 0)) = real.pi / 2

-- Proof statement
theorem eccentricity_of_ellipse : e = (real.sqrt 5 - 1) / 2 := 
sorry

end eccentricity_of_ellipse_l379_379249


namespace watermelon_ratio_l379_379577

theorem watermelon_ratio (michael_weight : ℕ) (john_weight : ℕ) (clay_weight : ℕ)
  (h₁ : michael_weight = 8) 
  (h₂ : john_weight = 12) 
  (h₃ : john_weight * 2 = clay_weight) :
  clay_weight / michael_weight = 3 :=
by {
  sorry
}

end watermelon_ratio_l379_379577


namespace max_legs_lengths_l379_379617

theorem max_legs_lengths (a x y : ℝ) (h₁ : x^2 + y^2 = a^2) (h₂ : 3 * x + 4 * y ≤ 5 * a) :
  3 * x + 4 * y = 5 * a → x = (3 * a / 5) ∧ y = (4 * a / 5) :=
by
  sorry

end max_legs_lengths_l379_379617


namespace positive_integer_root_k_l379_379808

theorem positive_integer_root_k (k : ℕ) :
  (∃ x : ℕ, x > 0 ∧ x * x - 34 * x + 34 * k - 1 = 0) ↔ k = 1 :=
by
  sorry

end positive_integer_root_k_l379_379808


namespace total_number_of_parts_l379_379779

-- Identify all conditions in the problem: sample size and probability
def sample_size : ℕ := 30
def probability : ℝ := 0.25

-- Statement of the proof problem: The total number of parts N is 120 given the conditions
theorem total_number_of_parts (N : ℕ) (h : (sample_size : ℝ) / N = probability) : N = 120 :=
sorry

end total_number_of_parts_l379_379779


namespace range_of_a_l379_379209

open Set

theorem range_of_a (a : ℝ) : (-3 < a ∧ a < -1) ↔ (∀ x, x < -1 ∨ 5 < x ∨ (a < x ∧ x < a+8)) :=
sorry

end range_of_a_l379_379209


namespace who_is_next_to_denis_l379_379282

-- Define the five positions
inductive Pos : Type
| p1 | p2 | p3 | p4 | p5
open Pos

-- Define the people
inductive Person : Type
| Anya | Borya | Vera | Gena | Denis
open Person

-- Define the conditions as predicates
def isAtStart (p : Person) : Prop := p = Borya

def areNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop := 
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) = 1

def notNextTo (p1 p2 : Person) (posMap : Person → Pos) : Prop :=
  match posMap p1, posMap p2 with
  | p1, p2 => (abs (p1.to_nat - p2.to_nat)) ≠ 1

-- Define the adjacency conditions
def conditions (posMap : Person → Pos) :=
  isAtStart (posMap Borya) ∧
  areNextTo Vera Anya posMap ∧
  notNextTo Vera Gena posMap ∧
  notNextTo Anya Borya posMap ∧
  notNextTo Anya Gena posMap ∧
  notNextTo Borya Gena posMap

-- The final goal: proving Denis is next to Anya and Gena
theorem who_is_next_to_denis (posMap : Person → Pos)
  (h : conditions posMap) :
  areNextTo Denis Anya posMap ∧ areNextTo Denis Gena posMap :=
sorry

end who_is_next_to_denis_l379_379282


namespace percentage_forgot_homework_l379_379253

def total_students_group_A : ℕ := 30
def total_students_group_B : ℕ := 50
def forget_percentage_A : ℝ := 0.20
def forget_percentage_B : ℝ := 0.12

theorem percentage_forgot_homework :
  let num_students_forgot_A := forget_percentage_A * total_students_group_A
  let num_students_forgot_B := forget_percentage_B * total_students_group_B
  let total_students_forgot := num_students_forgot_A + num_students_forgot_B
  let total_students := total_students_group_A + total_students_group_B
  let percentage_forgot := (total_students_forgot / total_students) * 100
  percentage_forgot = 15 := sorry

end percentage_forgot_homework_l379_379253


namespace B_and_C_are_complementary_l379_379362

noncomputable def is_complementary (E1 E2 : set ℕ) : Prop :=
  E1 ∩ E2 = ∅ ∧ E1 ∪ E2 = {1, 2, 3, 4, 5, 6}

def A := {n | n = 1 ∨ n = 3 ∨ n = 5}
def B := {n | n ≤ 3}
def C := {n | n ≥ 4}

theorem B_and_C_are_complementary : is_complementary B C :=
by
  sorry

end B_and_C_are_complementary_l379_379362


namespace max_norm_inequality_l379_379154

noncomputable def max_norm (coeffs : List ℝ) : ℝ :=
coeffs.map (λ x => abs x).maximumD 0

theorem max_norm_inequality (n : ℕ) (f_coeffs g_coeffs : Fin n.succ → ℝ) (r : ℝ) :
  let f : Polynomial ℝ := Polynomial.sum (Fin n.succ) (λ i => C (f_coeffs i) * X^i)
  let g : Polynomial ℝ := Polynomial.sum (Fin (n + 2)) (λ i => C (g_coeffs i) * X^i)
  f ≠ 0 →
  g ≠ 0 →
  g = Polynomial.sum (Fin n.succ) (λ i => C (r * f_coeffs i) * X^(i + 1)) + 
      Polynomial.sum (Fin n.succ) (λ i => C (f_coeffs i) * X^i) →
  let a := max_norm (List.ofFn f_coeffs)
  let c := max_norm (List.ofFn g_coeffs)
  a / c ≤ n + 1 :=
by
  intros
  sorry

end max_norm_inequality_l379_379154


namespace solution_set_of_f_less_than_exp_l379_379067

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_f_less_than_exp (hx_diff : Differentiable ℝ f)
    (hx_ineq : ∀ x : ℝ, f x > deriv[2] f x)
    (hx_odd : ∀ x : ℝ, f x - 1 = - (f (-x) - 1)) :
  { x : ℝ | f x < exp x } = { x : ℝ | 0 < x } :=
sorry

end solution_set_of_f_less_than_exp_l379_379067


namespace poly_at_neg4_l379_379728

def poly (x : ℤ) : ℤ :=  ((((3 * x + 5) * x + 6) * x + 79) * x - 8) * x + 35) * x + 12

theorem poly_at_neg4 (v_4 : ℤ) : poly (-4) = v_4 :=
  by 
  have v_0 := 3
  have v_1 := v_0 * (-4) + 5
  have v_2 := v_1 * (-4) + 6
  have v_3 := v_2 * (-4) + 79
  have v_4 := v_3 * (-4) - 8
  exact v_4 = 220

#eval poly (-4)  -- This can be used to verify the result is indeed 220

end poly_at_neg4_l379_379728


namespace denis_neighbors_l379_379287

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379287


namespace total_letters_in_all_names_l379_379551

theorem total_letters_in_all_names :
  let jonathan_first := 8
  let jonathan_surname := 10
  let younger_sister_first := 5
  let younger_sister_surname := 10
  let older_brother_first := 6
  let older_brother_surname := 10
  let youngest_sibling_first := 4
  let youngest_sibling_hyphenated_surname := 15
  jonathan_first + jonathan_surname + younger_sister_first + younger_sister_surname +
  older_brother_first + older_brother_surname + youngest_sibling_first + youngest_sibling_hyphenated_surname = 68 := by
  sorry

end total_letters_in_all_names_l379_379551


namespace proposition_p_proposition_q_correct_statements_l379_379917

def line_perpendicular_to_plane (l : Type) (π : Type) : Prop := 
∀ (l' : Type), l' ∈ π → perpendicular l l'

def line_parallel_to_plane (l : Type) (π : Type) : Prop := 
∀ (l' : Type), l' ∈ π → parallel l l'

theorem proposition_p (l : Type) (π : Type) (h : ∀ l' : Type, l' ∈ π → perpendicular l l') : 
  line_perpendicular_to_plane l π :=
by 
  sorry

theorem proposition_q (l : Type) (π : Type) (h : ∀ l' : Type, l' ∈ π → ¬ parallel l l') : 
  ¬ line_parallel_to_plane l π :=
by 
  sorry

theorem correct_statements (l : Type) (π : Type)
  (h1 : ∀ l' : Type, l' ∈ π → perpendicular l l')
  (h2 : ∀ l' : Type, l' ∈ π → ¬ parallel l l') :
  proposition_p l π h1 = True ∧ proposition_q l π h2 = True :=
by 
  exact ⟨proposition_p l π h1, proposition_q l π h2⟩

end proposition_p_proposition_q_correct_statements_l379_379917


namespace incorrect_median_l379_379941

/-- 
Given:
- A stem-and-leaf plot representation.
- Player B's scores are mainly between 30 and 40 points.
- Player B has 13 scores.
Prove:
The judgment "The median score of player B is 28" is incorrect.
-/
theorem incorrect_median (scores : List ℕ) (H_len : scores.length = 13) (H_range : ∀ x ∈ scores, 30 ≤ x ∧ x ≤ 40) 
  (H_median : ∃ median, median = scores.nthLe 6 sorry ∧ median = 28) : False := 
sorry

end incorrect_median_l379_379941


namespace cube_parallel_faces_l379_379390

theorem cube_parallel_faces : 
  ∀ (faces : Fin 6 → Set ℝ^3),
  (∀ i, ∃ j, i ≠ j ∧ faces i = faces j ∪ faces j ) → 
  (∀ i j, faces i ∩ faces j = ∅ ∨ (∀ (x : ℝ^3), x ∈ faces i ↔ x ∈ faces j)) → 
  ∃ (pairs: Fin 3 → (Fin 6) × (Fin 6)), 
  (∀ k, let (i, j) := pairs k in faces i = faces j) :=
by
  sorry

end cube_parallel_faces_l379_379390


namespace problem_statement_l379_379119

noncomputable def triangleABC (A B C : ℝ × ℝ) : Prop :=
  (dist A B = 5) ∧ (dist A C = 5) ∧ (B = (-1,3)) ∧ (C = (4,-2))

noncomputable def eulerLineTangentToCircle (euler : ℝ → ℝ → Prop) (x y r : ℝ) : Prop :=
  euler x y ∧ (x - 3)^2 + y^2 = r^2

theorem problem_statement (A B C : ℝ × ℝ) (r : ℝ) (M : ℝ → ℝ → Prop):
  triangleABC A B C →
  let EulerLine := λ x y, (x - y - 1 = 0) in
  eulerLineTangentToCircle EulerLine 3 0 r →
  r = sqrt 2 →
  M = (λ x y, (x - 3)^2 + y^2 = 2) →
  (∀ P ∈ {p : ℝ × ℝ | M p.fst p.snd}, dist P B = sqrt 23) :=
begin
  intros hABC hEuler hr hM,
  sorry
end

end problem_statement_l379_379119


namespace regression_and_points_imply_m_and_yhat_l379_379074

noncomputable def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

variable (x_vals : List ℝ) (y_vals : List ℝ)
variable (m : ℝ)
variable (linear_regression : ℝ → ℝ := λ x, 6 * x + -5)

def y_vals_complete : List ℝ :=
  [7.5, 11.5, m, 31.5, 36.5, 43.5]

def average_x : ℝ := average [2, 3, 4, 6, 7, 8]
def average_y_with_m : ℝ := average y_vals_complete

theorem regression_and_points_imply_m_and_yhat :
  average_x = 5 ∧
  average_y_with_m = 25 ∧
  (linear_regression 5 = 25) →
  m = 19.5 ∧ linear_regression 12 = 67 :=
begin
  sorry
end

end regression_and_points_imply_m_and_yhat_l379_379074


namespace zero_sequences_420_l379_379163

def ordered_triples (n : ℕ) : Finset (ℕ × ℕ × ℕ) :=
  {x | 1 ≤ x.1 ∧ x.1 ≤ n ∧ 1 ≤ x.2 ∧ x.2 ≤ n ∧ 1 ≤ x.3 ∧ x.3 ≤ n}.to_finset

def zero_sequence_count (n : ℕ) : ℕ :=
  let triples := ordered_triples n in
  let cond1 := {t ∈ triples | t.1 = t.2} in
  let cond2 := {t ∈ triples | t.2 = t.3 ∧ t.1 ≠ t.2} in
  let overlap := {t ∈ triples | t.1 = t.2 ∧ t.2 = t.3} in
  cond1.card + cond2.card - overlap.card

theorem zero_sequences_420 : zero_sequence_count 15 = 420 := sorry

end zero_sequences_420_l379_379163


namespace rational_sides_of_squares_l379_379930

-- Define a rectangle and its subdivision into squares
variables {a b : ℝ} {n : ℕ} {x : Fin n → ℝ}

-- The proof statement
theorem rational_sides_of_squares
  (hab : a ≠ 0 ∧ b ≠ 0) 
  (hsub : ∀ i, i < n → ∃ q : ℚ, x i = q * a ∨ x i = q * b):
  (∀ i, i < n → ∃ q : ℚ, x i = q * a ∧ ∃ q' : ℚ, x i = q' * b) :=
sorry

end rational_sides_of_squares_l379_379930


namespace denis_neighbors_l379_379291

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l379_379291


namespace inequality_solution_set_l379_379946

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the differentiable function

theorem inequality_solution_set (h_diff : differentiable_on ℝ f (Ioi 0))
  (h_condition : ∀ x > 0, (deriv f (deriv f x) + (2 / x) * f x > 0)) :
  { x : ℝ | x + 2018 > 0 ∧ ( (x + 2018) * f (x + 2018) / 3 < 3 * f(3) / (x+2018) ) } =
  { x : ℝ | -2018 < x ∧ x < -2015 } :=
sorry

end inequality_solution_set_l379_379946
