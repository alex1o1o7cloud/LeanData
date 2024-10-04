import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Arithmetic.Order
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Modeq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Approx
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Seq.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import algebra.module.basic
import data.rat.basic

namespace positive_difference_abs_eq_l646_646944

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l646_646944


namespace lottery_probability_approximation_l646_646962

noncomputable def binom (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem lottery_probability_approximation :
  (1 - (binom 85 5 / binom 90 5)) ≈ 0.25365 := 
sorry

end lottery_probability_approximation_l646_646962


namespace sin_area_interval_l646_646872

theorem sin_area_interval : 
  (∀ (n : ℕ) (hn : n > 0), ∫ x in (0 : ℝ)..(π / n), sin (n * x) = 2 / n) → 
  ∫ x in (0 : ℝ)..(2 * π / 3), sin (3 * x) = 4 / 3 :=
by
  intros h
  have h3 := h 3 (by decide)
  sorry

end sin_area_interval_l646_646872


namespace first_term_geometric_sequence_l646_646889

noncomputable def a (r : ℝ) : ℝ := 362880 / (r ^ 3)

theorem first_term_geometric_sequence :
  ∃ r : ℝ, r ^ 3 = 110 ∧ a (∛110) = 3308 := by
  sorry

end first_term_geometric_sequence_l646_646889


namespace exceeds_threshold_at_8_l646_646653

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def exceeds_threshold (n : ℕ) : Prop :=
  geometric_sum 2 2 n ≥ 500

theorem exceeds_threshold_at_8 :
  ∀ n < 8, ¬exceeds_threshold n ∧ exceeds_threshold 8 :=
by
  sorry

end exceeds_threshold_at_8_l646_646653


namespace part_a_part_b_part_c_part_d_l646_646992

-- Part (a)
noncomputable def numberOfWaysAtoC (n : ℕ) : ℚ :=
  if n % 2 = 0 then (4^(n/2) - 1) / 3 else 0

theorem part_a :
  ∀ (n : ℕ), numberOfWaysAtoC n = if n % 2 = 0 then (4^(n/2) - 1) / 3 else 0 := 
by
  intro n
  sorry -- Proof to be added

-- Part (b)
noncomputable def numberOfWaysAtoCWithoutD (n : ℕ) : ℚ :=
  if n % 2 = 0 then 3^(n/2 - 1) else 0

theorem part_b :
  ∀ (n : ℕ), numberOfWaysAtoCWithoutD n = if n % 2 = 0 then 3^(n/2 - 1) else 0 := 
by
  intro n
  sorry -- Proof to be added

-- Part (c)
noncomputable def probabilityAlive (n : ℕ) : ℚ :=
  let k := (n + 1) / 2
  (3/4)^(k - 1)

theorem part_c :
  ∀ (n : ℕ), probabilityAlive n = let k := (n + 1) / 2 in (3/4)^(k - 1) := 
by
  intro n
  sorry -- Proof to be added

-- Part (d)
noncomputable def expectedLifespan : ℚ := 9

theorem part_d :
  expectedLifespan = 9 := 
by
  sorry -- Proof to be added

end part_a_part_b_part_c_part_d_l646_646992


namespace digit_150th_of_5_div_13_l646_646924

theorem digit_150th_of_5_div_13 : 
    ∀ k : ℕ, (k = 150) → (fractionalPartDigit k (5 / 13) = 5) :=
by 
  sorry

end digit_150th_of_5_div_13_l646_646924


namespace town_population_l646_646783

theorem town_population (P : ℕ) : 
  (∀ (car_pollution_per_year bus_pollution_per_year bus_capacity : ℕ)
  (percent_people_take_bus : ℝ)
  (emission_reduction : ℤ),
  car_pollution_per_year = 10 →
  bus_pollution_per_year = 100 →
  bus_capacity = 40 →
  percent_people_take_bus = 0.25 →
  emission_reduction = 100 →
  (10 * P - (bus_pollution_per_year + (percent_people_take_bus * ↑P * car_pollution_per_year.toℝ).toℤ)) = emission_reduction) →
  P = 80 :=
by
  intro h
  sorry

end town_population_l646_646783


namespace sum_even_coefficients_of_polynomial_l646_646758

theorem sum_even_coefficients_of_polynomial :
  let polynomial := (1 + x + x^2)^6,
      coeffs := polynomial.coeffs,
      a0 := coeffs 0, a1 := coeffs 1, a2 := coeffs 2, a4 := coeffs 4,
      a6 := coeffs 6, a8 := coeffs 8, a10 := coeffs 10, a12 := coeffs 12 in
  a2 + a4 + a6 + a8 + a10 + a12 = 364 :=
by sorry

end sum_even_coefficients_of_polynomial_l646_646758


namespace symmetric_points_power_l646_646789

variables (m n : ℝ)

def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_power 
  (h : symmetric_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2023 = -1 :=
by 
  sorry

end symmetric_points_power_l646_646789


namespace log_sum_val_l646_646732

noncomputable def f (x : ℝ) (n : ℕ) : ℝ := x^(n + 1)

theorem log_sum_val :
  (∑ n in Finset.range 2013, Real.logb 2014 ((n + 1) / (n + 2))) = -1 := by
  sorry

end log_sum_val_l646_646732


namespace focus_intersect_l646_646065

variable (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)

noncomputable def ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem focus_intersect (AF1 F1B AF2 F2C : ℝ) (F1 F2 : ℝ × ℝ) (B C : ℝ × ℝ) :
  let e := real.sqrt (a^2 - b^2) / a
  e = real.sqrt (a^2 - b^2) / a →
  (AF1 / F1B) + (AF2 / F2C) = (4 * a^2 / b^2) - 1 := by
  sorry

end focus_intersect_l646_646065


namespace polynomial_strictly_monotonic_l646_646831

-- Conditions: P(x) is a polynomial, P(P(x)) is strictly monotonic and P(P(P(x))) is strictly monotonic.
variable {R : Type*} [Ring R] [LinearOrderedField R]

/-
  P is the polynomial function we will be dealing with.
  P(P(x)) and P(P(P(x))) being strictly monotonic implies P(x) should be.
-/
theorem polynomial_strictly_monotonic (P : R → R)
  (h_poly : ∀ x, IsPolynomial (P x))
  (h_pp_strict_mono : StrictMono (λ x, P (P x)))
  (h_ppp_strict_mono : StrictMono (λ x, P (P (P x)))) :
  StrictMono P :=
by
  sorry

end polynomial_strictly_monotonic_l646_646831


namespace number_of_valid_sets_l646_646045

open Set

-- Definitions for elements
constant a1 a2 a3 a4 a5: Type
constant U : Set Type := {a1, a2, a3, a4, a5}

-- Define the properties of M
def is_valid_M (M : Set Type) : Prop :=
  M ⊆ U ∧ M ∩ {a1, a2, a3} = {a1, a2}

theorem number_of_valid_sets : 
  {M : Set Type // is_valid_M M}.card = 4 :=
sorry

end number_of_valid_sets_l646_646045


namespace product_consecutive_two_digits_l646_646834

theorem product_consecutive_two_digits (a b c : ℕ) : 
  ¬(∃ n : ℕ, (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2)) :=
by
  sorry

end product_consecutive_two_digits_l646_646834


namespace reading_schedule_l646_646468

theorem reading_schedule 
  (peter_reading_speed : ℕ) (kristin_reading_factor : ℕ) (total_books : ℕ)
  (daily_reading_hours : ℕ) (peter_per_book_hours : ℕ) 
  (kristin_per_book_hours : ℕ) (days_for_kristin : ℕ) 
  (days_for_peter : ℕ) (peter_daily_hours : ℕ) :
  peter_reading_speed = 3 →
  kristin_reading_factor = 3 →
  total_books = 20 →
  daily_reading_hours = 16 →
  peter_per_book_hours = 12 →
  kristin_per_book_hours = peter_per_book_hours * kristin_reading_factor →
  days_for_kristin = (kristin_per_book_hours * total_books) / daily_reading_hours →
  days_for_kristin = 45 →
  days_for_peter = (peter_per_book_hours * total_books) / daily_reading_hours →
  days_for_peter = 45 →
  peter_daily_hours = (peter_per_book_hours * total_books) / days_for_kristin →
  peter_daily_hours = 16 * 5.33 :=
sorry

end reading_schedule_l646_646468


namespace tangent_of_angle_l646_646735

theorem tangent_of_angle (x y : ℝ) (h : (x, y) = (1, real.sqrt 3)) : real.tan α = real.sqrt 3 :=
by
  have h1 : x = 1, from h.1
  have h2 : y = real.sqrt 3, from h.2
  sorry

end tangent_of_angle_l646_646735


namespace alex_chairs_l646_646014

theorem alex_chairs (x y z : ℕ) (h : x + y + z = 74) : z = 74 - x - y :=
by
  sorry

end alex_chairs_l646_646014


namespace total_cost_correct_l646_646575

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end total_cost_correct_l646_646575


namespace h_at_3_eq_3_l646_646211

-- Define the function h(x) based on the given condition
noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * 
    (x^32 + 1) * (x^64 + 1) * (x^128 + 1) * (x^256 + 1) * (x^512 + 1) - 1) / 
  (x^(2^10 - 1) - 1)

-- State the required theorem
theorem h_at_3_eq_3 : h 3 = 3 := by
  sorry

end h_at_3_eq_3_l646_646211


namespace abundant_numbers_less_than_25_l646_646649

def proper_factors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d > 0 ∧ n % d = 0)

def is_abundant (n : ℕ) : Prop :=
  (proper_factors n).sum > n

def count_abundant_numbers_up_to (m : ℕ) : ℕ :=
  (List.range m).countp is_abundant

theorem abundant_numbers_less_than_25 : count_abundant_numbers_up_to 25 = 4 :=
by
  sorry

end abundant_numbers_less_than_25_l646_646649


namespace remaining_sausage_meat_l646_646033

-- Define the conditions
def total_meat_pounds : ℕ := 10
def sausage_links : ℕ := 40
def links_eaten_by_Brandy : ℕ := 12
def pounds_to_ounces : ℕ := 16

-- Calculate the remaining sausage meat and prove the correctness
theorem remaining_sausage_meat :
  (total_meat_pounds * pounds_to_ounces - links_eaten_by_Brandy * (total_meat_pounds * pounds_to_ounces / sausage_links)) = 112 :=
by
  sorry

end remaining_sausage_meat_l646_646033


namespace problem_statement_l646_646005

theorem problem_statement (x y : ℝ) 
  (h1 : 2 * x, 1, y - 1 form_arithmetic_seq)
  (h2 : y + 3, |x + 1| + |x - 1|, real.cos (real.arccos x) form_geometric_seq) 
  : (x + 1) * (y + 1) = 4 := sorry

end problem_statement_l646_646005


namespace hyperbola_eccentricity_l646_646066

noncomputable def parabola_focus (C1 : ℝ → ℝ → Prop) := (1, 0)
noncomputable def hyperbola_focus (C2 : ℝ → ℝ → Prop) := (1, 0)

def parabola (x y : ℝ) := y^2 = 4 * x
def hyperbola (x y a b : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

def common_point (x y : ℝ) (a b : ℝ) := parabola x y ∧ hyperbola x y a b

def directrix_point (M : ℝ × ℝ) := M.snd = -1

def isosceles_right_triangle (F P M : ℝ × ℝ) := 
  let d1 := (P.1 - F.1)^2 + (P.2 - F.2)^2 
  let d2 := (P.1 - M.1)^2 + (P.2 - M.2)^2
  d1 = 4 ∧ d2 = 4 ∧ P.1 = 1 ∧ P.2 = 2

theorem hyperbola_eccentricity :
  ∀ (a b : ℝ),
  (a > 0) → 
  (b > 0) → 
  common_point 1 2 a b → 
  isosceles_right_triangle (1, 0) (1, 2) (1, -1) →
  let e := Real.sqrt (1 / (3 - 2 * Real.sqrt 2)) in
  e = Real.sqrt 2 + 1 :=
by
  intros a b ha hb h_common h_triangle
  let e := Real.sqrt (1 / (3 - 2 * Real.sqrt 2))
  have he : e = Real.sqrt 2 + 1 := sorry
  exact he

end hyperbola_eccentricity_l646_646066


namespace rick_books_total_l646_646863

theorem rick_books_total 
  (N : ℕ)
  (h : N / 16 = 25) : 
  N = 400 := 
  sorry

end rick_books_total_l646_646863


namespace correct_total_gems_l646_646518

def total_gems (d r : ℕ) (b : ℝ) : ℕ :=
  let initial_gems := d * r
  let bonus_gems := (b * initial_gems).toNat
  initial_gems + bonus_gems

theorem correct_total_gems :
  total_gems 250 100 0.20 = 30000 :=
by
  -- Proof would typically go here, but we are focusing only on the statement.
  sorry

end correct_total_gems_l646_646518


namespace total_lawns_mowed_l646_646031

theorem total_lawns_mowed (earned_per_lawn forgotten_lawns total_earned : ℕ) 
    (h1 : earned_per_lawn = 9) 
    (h2 : forgotten_lawns = 8) 
    (h3 : total_earned = 54) : 
    ∃ (total_lawns : ℕ), total_lawns = 14 :=
by
    sorry

end total_lawns_mowed_l646_646031


namespace equal_pairwise_angle_magnitude_l646_646769

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem equal_pairwise_angle_magnitude 
  (a b c : ℝ × ℝ × ℝ) 
  (hab : ∃ θ, θ ≠ 0 ∧ θ ≠ π ∧ θ = 2 * real.pi / 3)
  (hac : ∃ θ, θ ≠ 0 ∧ θ ≠ π ∧ θ = 2 * real.pi / 3)
  (hbc : ∃ θ, θ ≠ 0 ∧ θ ≠ π ∧ θ = 2 * real.pi / 3)
  (ha : magnitude a = 1)
  (hb : magnitude b = 1)
  (hc : magnitude c = 2) :
  magnitude (a.1 + b.1 - c.1, a.2 + b.2 - c.2, a.3 + b.3 - c.3) = 3 := 
sorry

end equal_pairwise_angle_magnitude_l646_646769


namespace PTScoll_l646_646849

-- Definitions
variables (A B P Q R L T K S : Type) [linear_ordered_field A B P]
variables (a b : LinearOrderedField)
variables (l : Line) -- Assume a type representing lines
variables [perpendicular a l] [perpendicular b l]
variables [on_line A l] [on_line B l] [on_line P l]
variables [intersects P a Q] [intersects P b R]
variables [perpendicular_through A BQ L] [intersects BQ L] [intersects BR T]
variables [perpendicular_through B AR K] [intersects AR K] [intersects AQ S]

-- Statement
theorem PTScoll (h1 : ∀ A B P : l, P = T ∧ P = S) : collinear P T S := by sorry

end PTScoll_l646_646849


namespace smallest_multiple_5_711_l646_646152

theorem smallest_multiple_5_711 : ∃ n : ℕ, n = Nat.lcm 5 711 ∧ n = 3555 := 
by
  sorry

end smallest_multiple_5_711_l646_646152


namespace relationship_P_Q_l646_646829

theorem relationship_P_Q (x : ℝ) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.exp x + Real.exp (-x)) 
  (hQ : Q = (Real.sin x + Real.cos x) ^ 2) : 
  P ≥ Q := 
sorry

end relationship_P_Q_l646_646829


namespace sum_abs_le_three_l646_646817

open Complex

theorem sum_abs_le_three (n : ℕ) (h6n : 6 * n > 0) (a : Fin n → ℂ) 
  (h : ∀ I : Finset (Fin n), I.Nonempty → 
    abs ((∏ j in I, (1 + a j)) - 1) ≤ 1 / 2) : 
  (∑ i : Fin n, abs (a i)) ≤ 3 := 
by 
  sorry

end sum_abs_le_three_l646_646817


namespace minimum_discount_l646_646061

def cost_price := 1000
def marked_price := 1500
def profit_percentage := 0.05

theorem minimum_discount :
  ∀ (x : ℝ), (1500 * x - cost_price ≥ cost_price * profit_percentage) → (x ≥ 0.7) :=
by
  sorry

end minimum_discount_l646_646061


namespace cosine_function_properties_l646_646219

theorem cosine_function_properties :
  ∀ (x : ℝ), 
  let y := -5 * cos (x + π / 4) + 2 in
  amplitude y = 5 ∧ phase_shift y = -π / 4 ∧ vertical_shift y = 2 :=
by
  sorry

end cosine_function_properties_l646_646219


namespace friends_number_options_l646_646596

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l646_646596


namespace toads_per_acre_l646_646054

theorem toads_per_acre (b g : ℕ) (h₁ : b = 25 * g)
  (h₂ : b / 4 = 50) : g = 8 :=
by
  -- Condition h₁: For every green toad, there are 25 brown toads.
  -- Condition h₂: One-quarter of the brown toads are spotted, and there are 50 spotted brown toads per acre.
  sorry

end toads_per_acre_l646_646054


namespace smallest_num_rectangles_l646_646124

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l646_646124


namespace ellipse_equation_l646_646344

theorem ellipse_equation (a b c c1 : ℝ)
  (h_hyperbola_eq : ∀ x y, (y^2 / 4 - x^2 / 12 = 1))
  (h_sum_eccentricities : (c / a) + (c1 / 2) = 13 / 5)
  (h_foci_x_axis : c1 = 4) :
  (a = 5 ∧ b = 4 ∧ c = 3) → 
  ∀ x y, (x^2 / 25 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l646_646344


namespace intersection_M_N_l646_646011

-- Definitions of sets M and N
def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Proof statement showing the intersection of M and N
theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l646_646011


namespace tangent_line_at_origin_common_points_curve_l646_646738

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_at_origin :
  ∃ (k : ℝ), tangent (f := f) (0 : ℝ) (0 : ℝ) = k * x :=
sorry

theorem common_points_curve (m : ℝ) (hm : m > 0) :
  if h : m ≤ Real.exp 2 / 4 then
    if m = Real.exp 2 / 4 then
      ∃ (x : ℝ), (0 < x) ∧ (Real.exp x = m * x ^ 2)
    else
      ∀ (x : ℝ), ¬ (0 < x) → (Real.exp x = m * x ^ 2)
  else
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (0 < x₁) ∧ (0 < x₂) ∧ (Real.exp x₁ = m * x₁ ^ 2) ∧ (Real.exp x₂ = m * x₂ ^ 2) :=
sorry

end tangent_line_at_origin_common_points_curve_l646_646738


namespace coefficient_of_determination_is_one_l646_646378

-- Definitions based on the conditions
variables {α : Type*} [linear_ordered_field α]

def all_points_on_line (points : list (α × α)) (m : α) (nonzero_slope : m ≠ 0) : Prop :=
  ∀ x y, (x, y) ∈ points → ∃ b, y = m * x + b

def coefficient_of_determination (points : list (α × α)) : α := sorry

def sum_of_squares_of_residuals (points : list (α × α)) : α := 
  ∑ p in points, let (x, y) := p in (y - (m * x + b))^2

-- The proof problem: prove R^2 = 1
theorem coefficient_of_determination_is_one
  {points : list (α × α)} {m : α} (nonzero_slope : m ≠ 0) 
  (h : all_points_on_line points m nonzero_slope) :
  coefficient_of_determination points = 1 := sorry

end coefficient_of_determination_is_one_l646_646378


namespace bisects_AP_l646_646414

theorem bisects_AP
  (A D P E B C O1 O2 : Type)
  [hd1 : is_convex_quadrilateral ADPE] 
  (h1 : ∡(ADP) = ∡(EPC))
  (h2 : extend AD = B)
  (h3 : extend AE = C)
  (h4 : ∡(DPB) = ∡(EPC)) 
  (h5 : is_circumcenter O1 (△ADE))
  (h6 : is_circumcenter O2 (△ABC))
  (h7 : ∀ x, x ∈ circumcircle(△ADE) ↔ ¬(x ∈ circumcircle(△ABC))) :
  bisects (O1O2) AP :=
sorry

end bisects_AP_l646_646414


namespace digit_150_l646_646932

def decimal_rep : ℚ := 5 / 13

def cycle_length : ℕ := 6

theorem digit_150 (n : ℕ) (h : n = 150) : Nat.digit (n % cycle_length) (decimal_rep) = 5 := by
  sorry

end digit_150_l646_646932


namespace minimal_wall_area_l646_646450

theorem minimal_wall_area (m n : ℕ) (h : m ≥ n) :
  let areas := [1, 2..n].map (λ x, m * x)
  in (areas.sum = m * (n * (n + 1) / 2)) :=
by sorry

end minimal_wall_area_l646_646450


namespace truck_loading_time_l646_646191

noncomputable def combined_rate (r1 r2 : ℝ) : ℝ := r1 + r2

theorem truck_loading_time :
  let rate1 := (1 : ℝ) / 5
  let rate2 := (1 : ℝ) / 4
  let combinedRate := combined_rate rate1 rate2
  combinedRate ≠ 0 →
  (1 / combinedRate) ≈ 2.22 :=
by
  intros rate1 rate2 combinedRate combinedRate_nonzero
  have hr : rate1 + rate2 = 9 / 20 := by norm_num
  have combinedRate_def : combinedRate = 9 / 20 := by
    rw [combined_rate, hr]
  rw [combinedRate_def]
  norm_num -- 'norm_num' simplifies numerical expressions
  sorry

end truck_loading_time_l646_646191


namespace decreasing_function_interval_l646_646531

theorem decreasing_function_interval (x : ℝ) (h : x ∈ Ioi 0) : 
  ¬ (∀ x ∈ Ioi 0, ∀ y ∈ Ioi 0, x < y → e^(-x) ≥ e^(-y)) ∨
  ¬ (∀ x ∈ Ioi 0, ∀ y ∈ Ioi 0, x < y → (-(x^2) + 2 * x) ≥ (-(y^2) + 2 * y)) ∨
  ¬ (∀ x ∈ Ioi 1, ∀ y ∈ Ioi 1, x < y → log (1/2 : ℝ) (x - 1) ≥ log (1/2 : ℝ) (y - 1)) ∨
  (∀ x ∈ Ioi 0, ∀ y ∈ Ioi 0, x < y → e^(-x) > e^(-y)) := 
sorry

end decreasing_function_interval_l646_646531


namespace transformed_graph_matches_b_l646_646672

def f (x : ℝ) : ℝ :=
  if x >= -3 ∧ x <= 0 then -2 - x
  else if x >= 0 ∧ x <= 2 then real.sqrt (4 - (x - 2) ^ 2) - 2
  else if x >= 2 ∧ x <= 3 then 2 * (x - 2)
  else 0 -- assuming 0 outside the given range

def g (x : ℝ) : ℝ := f (x + 2) - 1

-- Assuming graph_b describes the transformation and is defined somewhere
def graph_b (x : ℝ) : ℝ := sorry -- This would be the actual transformation of f.

theorem transformed_graph_matches_b : g = graph_b := sorry

end transformed_graph_matches_b_l646_646672


namespace compute_avg_interest_rate_l646_646998

variable (x : ℝ)

/-- The total amount of investment is $5000 - x at 3% and x at 7%. The incomes are equal 
thus we are asked to compute the average rate of interest -/
def avg_interest_rate : Prop :=
  let i_3 := 0.03 * (5000 - x)
  let i_7 := 0.07 * x
  i_3 = i_7 ∧
  (2 * i_3) / 5000 = 0.042

theorem compute_avg_interest_rate 
  (condition : ∃ x : ℝ, 0.03 * (5000 - x) = 0.07 * x) :
  avg_interest_rate x :=
by
  sorry

end compute_avg_interest_rate_l646_646998


namespace breaststroke_speed_correct_l646_646517

-- Defining the given conditions
def total_distance : ℕ := 500
def front_crawl_speed : ℕ := 45
def front_crawl_time : ℕ := 8
def total_time : ℕ := 12

-- Definition of the breaststroke speed given the conditions
def breaststroke_speed : ℕ :=
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  breaststroke_distance / breaststroke_time

-- Theorem to prove the breaststroke speed is 35 yards per minute
theorem breaststroke_speed_correct : breaststroke_speed = 35 :=
  sorry

end breaststroke_speed_correct_l646_646517


namespace compare_fractions_l646_646140

theorem compare_fractions : (31 : ℚ) / 11 > (17 : ℚ) / 14 := 
by
  sorry

end compare_fractions_l646_646140


namespace first_year_after_2010_with_digit_sum_5_l646_646776

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10)

theorem first_year_after_2010_with_digit_sum_5 : 
  ∃ (y : ℕ), y > 2010 ∧ sum_of_digits(y) = 5 ∧ ∀ (z : ℕ), (z > 2010 ∧ z < y) → sum_of_digits(z) ≠ 5 :=
begin
  use 2012,
  split,
  { exact dec_trivial }, -- proof that 2012 > 2010
  split,
  { exact dec_trivial }, -- proof that sum_of_digits(2012) = 5
  { intros z h,
    cases h with h₁ h₂,
    -- proof that for all z between 2010 and 2012, sum_of_digits(z) ≠ 5
    sorry }
end

end first_year_after_2010_with_digit_sum_5_l646_646776


namespace angle_MN_AD_45_l646_646782

-- Define the setup for a regular tetrahedron and its properties
variables (A B C D M N : ℝ × ℝ × ℝ)
variables (mid_AB : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2))
variables (mid_CD : N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2, (C.3 + D.3) / 2))

noncomputable def angle_between_vectors (u v : ℝ × ℝ × ℝ) : ℝ :=
real.acos ((u.1 * v.1 + u.2 * v.2 + u.3 * v.3) /
  (real.sqrt (u.1^2 + u.2^2 + u.3^2) * real.sqrt (v.1^2 + v.2^2 + v.3^2)))

-- Define the vectors MN and AD
def vec_MN : ℝ × ℝ × ℝ := (N.1 - M.1, N.2 - M.2, N.3 - M.3)
def vec_AD : ℝ × ℝ × ℝ := (D.1 - A.1, D.2 - A.2, D.3 - A.3)

-- Assert that the angle between MN and AD is 45 degrees
theorem angle_MN_AD_45 :
  angle_between_vectors (vec_MN N M) (vec_AD D A) = real.pi / 4 :=
sorry

end angle_MN_AD_45_l646_646782


namespace angles_of_intersecting_lines_l646_646467

noncomputable def angle1 : ℝ := 41
noncomputable def angle2 := 180 - angle1
noncomputable def angle3 := angle1
noncomputable def angle4 := angle2

theorem angles_of_intersecting_lines (a1 a2 a3 a4 : ℝ) (h1 : a1 = 41) :
a2 = 180 - a1 ∧ a3 = a1 ∧ a4 = a2 :=
begin
  sorry
end

end angles_of_intersecting_lines_l646_646467


namespace volume_of_box_l646_646429

-- Problem: Given a box with certain dimensions, prove its volume.
theorem volume_of_box : 
  let H := 12 
  let L := 3 * H 
  let W := L / 4 
  V = L * W * H
  in V = 3888 := 
by
  let H := 12 
  let L := 3 * H 
  let W := L / 4 
  let V := L * W * H
  show V = 3888, 
  sorry

end volume_of_box_l646_646429


namespace f_prime_at_zero_l646_646290

noncomputable def f (x : ℝ) : ℝ := exp x + 2 * x * f' 1

-- Theorem: Given the function f and its conditions, prove that f'(0) = 1 - 2 * real.exp 1
theorem f_prime_at_zero : (∃ (f' : ℝ → ℝ), (∀ x, deriv f x = f' x) ∧ f' = fun x => exp x + 2 * f' 1) → f' 0 = 1 - 2 * real.exp 1 :=
by
  sorry

end f_prime_at_zero_l646_646290


namespace general_term_no_geometric_sequence_l646_646274

-- Defining the arithmetic sequence
def arithmetic_seq (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

-- Sum of the first n terms in arithmetic sequence
def sum_arithmetic_seq (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ := (n : ℝ) / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
def a1 := sqrt 2 + 1
def S4 := 4 * sqrt 2 + 16

-- Extracting common difference d from given conditions
def common_difference : ℝ := 2

-- General term formula of the arithmetic sequence
theorem general_term :
    ∀ n : ℕ, n > 0 → arithmetic_seq a1 common_difference n = 2 * n - 1 + sqrt 2 := by
  sorry

-- Definition of b_n
def b_n (n : ℕ) := sum_arithmetic_seq a1 common_difference n / n

-- Prove that the sequence {b_n} does not contain three distinct terms forming a geometric sequence
theorem no_geometric_sequence :
    ¬ ∃ (r s t : ℕ), r ≠ s ∧ s ≠ t ∧ t ≠ r ∧ (b_n s) ^ 2 = b_n r * b_n t := by
  sorry

end general_term_no_geometric_sequence_l646_646274


namespace positive_difference_abs_eq_15_l646_646949

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l646_646949


namespace circle_through_and_tangent_l646_646971

noncomputable def circle_eq (a b r : ℝ) (x y : ℝ) : ℝ :=
  (x - a) ^ 2 + (y - b) ^ 2 - r ^ 2

theorem circle_through_and_tangent
(h1 : circle_eq 1 2 2 1 0 = 0)
(h2 : ∀ x y, circle_eq 1 2 2 x y = 0 → (x = 1 → y = 2 ∨ y = -2))
: ∀ x y, circle_eq 1 2 2 x y = 0 → (x - 1) ^ 2 + (y - 2) ^ 2 = 4 :=
by
  sorry

end circle_through_and_tangent_l646_646971


namespace fraction_sum_is_five_l646_646679

noncomputable def solve_fraction_sum (x y z : ℝ) : Prop :=
  (x + 1/y = 5) ∧ (y + 1/z = 2) ∧ (z + 1/x = 3) ∧ 0 < x ∧ 0 < y ∧ 0 < z → 
  (x / y + y / z + z / x = 5)
    
theorem fraction_sum_is_five (x y z : ℝ) : solve_fraction_sum x y z :=
  sorry

end fraction_sum_is_five_l646_646679


namespace safe_combination_l646_646170

def decode_combination (N I M A b : ℕ) : Prop :=
  ∀ (a b : ℕ), a ≠ b → a ≠ N + 10 * I + 100 * M ∧ b ≠ N + 10 * I + 100 * M → 
  let NIM := N + 10 * I + 100 * M,
      AM := A + 10 * M,
      MIA := M + 10 * I + 100 * A,
      MINA := M + 10 * I + 100 * N + 1000 * A in
    NIM + AM + MIA = MINA ∧
    b = 10 ∧ MINA = 845

theorem safe_combination :
  ∃ (N I M A b : ℕ), decode_combination N I M A b :=
sorry

end safe_combination_l646_646170


namespace exists_three_numbers_sum_at_least_five_l646_646452

theorem exists_three_numbers_sum_at_least_five 
  {x : Fin 9 → ℝ}
  (h : 0 ≤ x 0 ∧ 0 ≤ x 1 ∧ 0 ≤ x 2 ∧ 0 ≤ x 3 ∧ 0 ≤ x 4 
       ∧ 0 ≤ x 5 ∧ 0 ≤ x 6 ∧ 0 ≤ x 7 ∧ 0 ≤ x 8)
  (sum_square_ge_25 : ∑ i in Finset.univ, x i ^ 2 ≥ 25) :
  ∃ (i j k : Fin 9), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ x i + x j + x k ≥ 5 :=
begin
  -- proof omitted
  sorry
end

end exists_three_numbers_sum_at_least_five_l646_646452


namespace positive_difference_abs_eq_15_l646_646947

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l646_646947


namespace candidate_percentage_l646_646168

open Real

variables (P : Real) (votes_cast : Real) (votes_difference : Real)

theorem candidate_percentage (votes_cast = 5300) (votes_difference = 1908) : 
  (P / 100) * votes_cast + 1908 = votes_cast -> 
  P = 32 := 
by
  sorry

end candidate_percentage_l646_646168


namespace xy_value_l646_646830

structure Point (R : Type) := (x : R) (y : R)

def A : Point ℝ := ⟨2, 7⟩ 
def C : Point ℝ := ⟨4, 3⟩ 

def is_midpoint (A B C : Point ℝ) : Prop :=
  (C.x = (A.x + B.x) / 2) ∧ (C.y = (A.y + B.y) / 2)

theorem xy_value (x y : ℝ) (B : Point ℝ := ⟨x, y⟩) (H : is_midpoint A B C) :
  x * y = -6 := 
sorry

end xy_value_l646_646830


namespace equidistant_points_are_on_intersection_line_l646_646675

noncomputable def trihedral_angle_equidistant_points (a b c : Line) (P_ab P_bc P_ca : Plane) (S : Line) : Prop :=
  (∃ (a b c : Line) (P_ab P_bc P_ca : Plane),
    -- Conditions:
    (P_ab ≠ P_bc) ∧ (P_bc ≠ P_ca) ∧ (P_ca ≠ P_ab) ∧
    (S = intersection(P_ab, P_bc)) ∧
    (S = intersection(P_bc, P_ca)) ∧
    (S = intersection(P_ca, P_ab))
   )

theorem equidistant_points_are_on_intersection_line
    (a b c : Line) (P_ab P_bc P_ca : Plane) (S : Line) :
    trihedral_angle_equidistant_points a b c P_ab P_bc P_ca S :=
sorry

end equidistant_points_are_on_intersection_line_l646_646675


namespace smallest_num_rectangles_to_cover_square_l646_646114

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l646_646114


namespace ways_to_choose_4_non_coplanar_points_l646_646637

theorem ways_to_choose_4_non_coplanar_points (tetrahedron_points : Finset ℝ) (h : tetrahedron_points.card = 10) :
  ∃ (S : Finset (Finset ℝ)), (∀ s ∈ S, s.card = 4 ∧ (¬ coplanar s) ∧ (s ⊆ tetrahedron_points)) → S.card = 141 :=
sorry

end ways_to_choose_4_non_coplanar_points_l646_646637


namespace trisectors_equilateral_l646_646424

theorem trisectors_equilateral {A B C A1 B1 C1 : Type}
  [triangle A B C] (trisectors_intersect_A1 : ∀ {triABC : triangle}, intersect (trisector B) (trisector C) = A1)
  (trisectors_intersect_B1 : ∀ {triABC : triangle}, intersect (trisector A) (trisector C) = B1)
  (trisectors_intersect_C1 : ∀ {triABC : triangle}, intersect (trisector A) (trisector B) = C1) :
  equilateral_triangle A1 B1 C1 :=
sorry

end trisectors_equilateral_l646_646424


namespace cost_price_of_computer_table_l646_646081

theorem cost_price_of_computer_table (S : ℝ) (C : ℝ) (h1 : 1.80 * C = S) (h2 : S = 3500) : C = 1944.44 :=
by
  sorry

end cost_price_of_computer_table_l646_646081


namespace problem1_problem2_l646_646041

-- Problem (1)
theorem problem1 (a : ℚ) (h : a = -1/2) : 
  a * (a - 4) - (a + 6) * (a - 2) = 16 := by
  sorry

-- Problem (2)
theorem problem2 (x y : ℚ) (hx : x = 8) (hy : y = -8) :
  (x + 2 * y) * (x - 2 * y) - (2 * x - y) * (-2 * x - y) = 0 := by
  sorry

end problem1_problem2_l646_646041


namespace total_money_spent_l646_646426

-- Definitions based on conditions
def num_bars_of_soap : Nat := 20
def weight_per_bar_of_soap : Float := 1.5
def cost_per_pound_of_soap : Float := 0.5

def num_bottles_of_shampoo : Nat := 15
def weight_per_bottle_of_shampoo : Float := 2.2
def cost_per_pound_of_shampoo : Float := 0.8

-- The theorem to prove
theorem total_money_spent :
  let cost_per_bar_of_soap := weight_per_bar_of_soap * cost_per_pound_of_soap
  let total_cost_of_soap := Float.ofNat num_bars_of_soap * cost_per_bar_of_soap
  let cost_per_bottle_of_shampoo := weight_per_bottle_of_shampoo * cost_per_pound_of_shampoo
  let total_cost_of_shampoo := Float.ofNat num_bottles_of_shampoo * cost_per_bottle_of_shampoo
  total_cost_of_soap + total_cost_of_shampoo = 41.40 := 
by
  -- proof goes here
  sorry

end total_money_spent_l646_646426


namespace integer_division_probability_l646_646074

theorem integer_division_probability :
  let S := {s : Int | -5 < s ∧ s < 7}
  let M := {m : Int | 2 < m ∧ m < 9}
  (s ∈ S) → (m ∈ M) → (s % m = 0) →
  (finset.card (finset.filter (λ (p : Int × Int), p.1 % p.2 = 0) ((finset.product (S.to_finset) (M.to_finset)))) / finset.card (finset.product (S.to_finset) (M.to_finset))) = 2 / 11 :=
by
  sorry

end integer_division_probability_l646_646074


namespace largest_tile_size_l646_646636

theorem largest_tile_size (len_cm : ℕ) (width_cm : ℕ)
  (H_len : len_cm = 378) (H_width : width_cm = 525) :
  ∃ (gcd : ℕ), gcd = (Nat.gcd len_cm width_cm) ∧ gcd = 21 :=
by {
  use 21,
  split,
  { rw [Nat.gcd_comm, Nat.gcd_eq_left_iff_dvd],
    exact dvd_gcd (dvd.intro 18 rfl) (dvd.intro 25 rfl)},
  { refl }
}

end largest_tile_size_l646_646636


namespace group_friends_opponents_l646_646623

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l646_646623


namespace area_projection_eq_area_original_eq_l646_646716

theorem area_projection_eq (a : ℝ) : 
  let original_area := (sqrt 3 / 4) * a^2 in
  let projection_area := original_area * (sqrt 2 / 4) in
  (projection_area = (sqrt 6 / 16) * a^2) :=
by sorry

theorem area_original_eq (a : ℝ) : 
  let projection_area := (sqrt 3 / 4) * a^2 in
  let original_area := projection_area / (sqrt 2 / 4) in
  (original_area = (sqrt 6 / 2) * a^2) :=
by sorry

end area_projection_eq_area_original_eq_l646_646716


namespace group_friends_opponents_l646_646621

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l646_646621


namespace stock_price_at_end_of_second_year_l646_646682

def stock_price_first_year (initial_price : ℝ) : ℝ :=
  initial_price * 2

def stock_price_second_year (price_after_first_year : ℝ) : ℝ :=
  price_after_first_year * 0.75

theorem stock_price_at_end_of_second_year : 
  (stock_price_second_year (stock_price_first_year 100) = 150) :=
by
  sorry

end stock_price_at_end_of_second_year_l646_646682


namespace parabola_zero_difference_l646_646082

theorem parabola_zero_difference :
  ∀ a b c : ℝ,
  (∀ x : ℝ, (⟨1, -2⟩:ℝ×ℝ) = ⟨x - 1, a * (x - 1)^2 - 2⟩) →
  (∃ y : ℝ, y = a * (3 - 1)^2 - 2 ∧ (3, 10) = (3, y)) →
  (let m := 1 + sqrt (2/3:ℝ);
       n := 1 - sqrt (2/3:ℝ)
   in m > n → m - n = (2 * sqrt 6) / 3) :=
sorry

end parabola_zero_difference_l646_646082


namespace cost_per_ounce_l646_646160

theorem cost_per_ounce (total_cost : ℕ) (num_ounces : ℕ) (h1 : total_cost = 84) (h2 : num_ounces = 12) : (total_cost / num_ounces) = 7 :=
by
  sorry

end cost_per_ounce_l646_646160


namespace right_triangle_count_l646_646372

theorem right_triangle_count :
  ∃! (a b : ℕ), (a^2 + b^2 = (b + 3)^2) ∧ (b < 50) :=
by
  sorry

end right_triangle_count_l646_646372


namespace smallest_num_rectangles_to_cover_square_l646_646113

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l646_646113


namespace sum_of_odd_increasing_function_on_arithmetic_sequence_l646_646267

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n, a (n + 1) = a n + d

theorem sum_of_odd_increasing_function_on_arithmetic_sequence
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_function f)
  (h_arith_seq : arithmetic_sequence a)
  (h_a1007_pos : a 1007 > 0) :
  ∑ k in Finset.range 2013, f (a (k + 1)) > 0 :=
sorry

end sum_of_odd_increasing_function_on_arithmetic_sequence_l646_646267


namespace find_initial_interest_rate_l646_646197

-- Definitions of the initial conditions
def P1 : ℝ := 3000
def P2 : ℝ := 1499.9999999999998
def P_total : ℝ := 4500
def r2 : ℝ := 0.08
def total_annual_income : ℝ := P_total * 0.06

-- Defining the problem as a statement to prove
theorem find_initial_interest_rate (r1 : ℝ) :
  (P1 * r1) + (P2 * r2) = total_annual_income → r1 = 0.05 := by
  sorry

end find_initial_interest_rate_l646_646197


namespace problem_l646_646764

theorem problem (a b c : ℤ) (h1 : 0 < c) (h2 : c < 90) (h3 : Real.sqrt (9 - 8 * Real.sin (50 * Real.pi / 180)) = a + b * Real.sin (c * Real.pi / 180)) : 
  (a + b) / c = 1 / 2 :=
by
  sorry

end problem_l646_646764


namespace range_of_t_l646_646352

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * Real.exp x

def t_range (t : ℝ) : Prop := (t > -2) ∧ (t ≤ 0)

theorem range_of_t (t : ℝ) : (∀ x ∈ set.Icc (-2 : ℝ) t, (∀ a b : ℝ, a ≤ b → f a ≤ f b)) ↔ t_range t :=
by
  sorry

end range_of_t_l646_646352


namespace dealer_profit_percentage_l646_646539

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℝ) (sp_total : ℝ) (sp_count : ℝ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  let profit_percentage := (profit_per_article / cp_per_article) * 100
  profit_percentage

theorem dealer_profit_percentage :
  profit_percentage 25 15 38 12 = 89.99 := by
  sorry

end dealer_profit_percentage_l646_646539


namespace dot_product_a_b_l646_646363

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_a_b : a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end dot_product_a_b_l646_646363


namespace book_price_percentage_change_l646_646641

theorem book_price_percentage_change (P : ℝ) (x : ℝ) (h : P * (1 - (x / 100) ^ 2) = 0.90 * P) : x = 32 := by
sorry

end book_price_percentage_change_l646_646641


namespace quadratic_roots_distinct_and_m_value_l646_646750

theorem quadratic_roots_distinct_and_m_value (m : ℝ) (α β : ℝ) (h_equation : ∀ x, x^2 - 2 * x - 3 * m^2 = 0 → (Root_of(x) = α ∨ Root_of(x) = β)) 
(h_alpha_beta : α + 2 * β = 5) :
  (2^2 - 4 * 1 * -3 * m^2 > 0) ∧ (m^2 = 1) :=
by
  have h_discriminant : 4 + 12 * m^2 > 0 := by sorry
  have h_root_sum : α + β = 2 := by sorry
  have h_root_product : α * β = -3 * m^2 := by sorry
  have h_quad_solved : (β = 3) ∧ (α = -1) := by sorry
  have h_m2 : m^2 = 1 := by sorry
  exact ⟨h_discriminant,h_m2⟩

end quadratic_roots_distinct_and_m_value_l646_646750


namespace vector_q_properties_l646_646008

noncomputable section
open Matrix

-- Define the vectors p, r, and q
def p : ℝ × ℝ × ℝ := (3, -2, -2)
def r : ℝ × ℝ × ℝ := (-1, -1, 1)
def q : ℝ × ℝ × ℝ := (-3 * Real.sqrt 51 / 17, -2 * Real.sqrt 51 / 17, 9 * Real.sqrt 51 / 17)

-- Define the dot product of two 3D vectors
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the norm of a 3D vector
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- State the theorem
theorem vector_q_properties :
  (∃ s : ℝ, q = (λ p r s, (p.1 + s * (r.1 - p.1), p.2 + s * (r.2 - p.2), p.3 + s * (r.3 - p.3))) p r s) ∧
  (dot_product p q / (norm p * norm q) = dot_product q r / (norm q * norm r)) :=
by
  sorry

end vector_q_properties_l646_646008


namespace avg_k_for_polynomial_roots_l646_646297

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l646_646297


namespace lemonade_calories_is_correct_l646_646436

def lemon_juice_content := 150
def sugar_content := 150
def water_content := 450

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def water_calories_per_100g := 0

def total_weight := lemon_juice_content + sugar_content + water_content
def caloric_density :=
  (lemon_juice_content * lemon_juice_calories_per_100g / 100) +
  (sugar_content * sugar_calories_per_100g / 100) +
  (water_content * water_calories_per_100g / 100)
def calories_per_gram := caloric_density / total_weight

def calories_in_300_grams := 300 * calories_per_gram

theorem lemonade_calories_is_correct : calories_in_300_grams = 258 := by
  sorry

end lemonade_calories_is_correct_l646_646436


namespace possible_number_of_friends_l646_646601

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l646_646601


namespace f_neg_1001_eq_2005_l646_646046

def f : ℝ → ℝ := sorry

theorem f_neg_1001_eq_2005 (h1 : ∀ x y : ℝ, f(x * y) + x = x * f(y) + f(x)) (h2 : f(-1) = 5) : f(-1001) = 2005 :=
sorry

end f_neg_1001_eq_2005_l646_646046


namespace distinct_positive_integers_eq_odd_integers_sum_limited_repetitions_eq_non_divisible_by_k_l646_646544

-- Part (a)
theorem distinct_positive_integers_eq_odd_integers_sum (n : ℕ) :
  (number_of_representations_as_sum_of_distinct_positive_integers n) =
  (number_of_representations_as_sum_of_positive_odd_integers n) :=
sorry

-- Part (b)
theorem limited_repetitions_eq_non_divisible_by_k (n k : ℕ) :
  (number_of_representations_as_sum_with_repetition_limit n k) =
  (number_of_representations_as_sum_excluding_multiples_of n k) :=
sorry

end distinct_positive_integers_eq_odd_integers_sum_limited_repetitions_eq_non_divisible_by_k_l646_646544


namespace volume_of_inscribed_cone_l646_646072

theorem volume_of_inscribed_cone 
  (H : ℝ) 
  (α : ℝ) 
  (H_pos : 0 < H) 
  (α_nonzero : 0 < α)
  (α_lt_pi_div_two : α < π / 2) 
  (mutually_perpendicular : true) :
  let OD := H * real.sin α,
      OO₁ := OD * real.sin α,
      O₁D := OD * real.cos α,
      V := (1/3) * π * (O₁D)^2 * OO₁ in
  V = (1/3) * π * H^3 * (real.sin α)^4 * (real.cos α)^2 :=
sorry

end volume_of_inscribed_cone_l646_646072


namespace average_k_l646_646305

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l646_646305


namespace find_x_l646_646760

theorem find_x (x : ℝ) (h : log 3 (x * x * x) + log 9 x + log 27 x = 10) :
  x = 3 ^ (60 / 23) :=
sorry

end find_x_l646_646760


namespace triangle_cross_section_possible_l646_646766

inductive GeometricSolid
| Cube
| Cylinder
| Cone
| RegularTriangularPrism

open GeometricSolid

def canFormTriangle (solid : GeometricSolid) : Prop :=
  solid = Cube ∨ solid = Cone ∨ solid = RegularTriangularPrism

theorem triangle_cross_section_possible (solid : GeometricSolid) :
  canFormTriangle solid ↔ solid = Cube ∨ solid = Cone ∨ solid = RegularTriangularPrism :=
by
  intro h
  sorry

end triangle_cross_section_possible_l646_646766


namespace jerry_age_l646_646459

variable (M J : ℕ) -- Declare Mickey's and Jerry's ages as natural numbers

-- Define the conditions as hypotheses
def condition1 := M = 2 * J - 6
def condition2 := M = 18

-- Theorem statement where we need to prove J = 12 given the conditions
theorem jerry_age
  (h1 : condition1 M J)
  (h2 : condition2 M) :
  J = 12 :=
sorry

end jerry_age_l646_646459


namespace group_friends_opponents_l646_646620

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l646_646620


namespace product_of_roots_l646_646678

-- Define the fourth-degree polynomial
def poly1 (x : ℝ) : ℝ := 3 * x ^ 4 - 9 * x ^ 3 + 27 * x - 81

-- Define the second-degree polynomial
def poly2 (x : ℝ) : ℝ := 4 * x ^ 2 - 16 * x + 64

-- Define the full polynomial equation as their product
def full_poly (x : ℝ) : ℝ := poly1 x * poly2 x

-- State the theorem to prove the product of the roots is -432
theorem product_of_roots : ∏ x in (finset.filter (λ x, full_poly x = 0) (finset.range 7)), x = -432 :=
by
  sorry

end product_of_roots_l646_646678


namespace expression_one_eq_expression_two_eq_l646_646205

variable (a : ℝ)

theorem expression_one_eq : 
  (-2 * a) ^ 6 * (-3 * a ^ 3) + (2 * a) ^ 2 * 3 = -192 * a ^ 9 + 12 * a ^ 2 :=
sorry

theorem expression_two_eq : 
  abs (-1 / 8) + real.pi ^ 3 + (-1 / 2) ^ 3 - (1 / 3) ^ 2 = real.pi ^ 3 - 1 / 9 :=
sorry

end expression_one_eq_expression_two_eq_l646_646205


namespace min_megabytes_for_plan_Y_more_economical_l646_646646

theorem min_megabytes_for_plan_Y_more_economical :
  ∃ (m : ℕ), 2500 + 10 * m < 15 * m ∧ m = 501 :=
by
  sorry

end min_megabytes_for_plan_Y_more_economical_l646_646646


namespace possible_number_of_friends_l646_646603

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l646_646603


namespace average_k_positive_int_roots_l646_646342

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l646_646342


namespace johns_salary_before_bonus_l646_646806

theorem johns_salary_before_bonus 
  (last_year_salary last_year_bonus this_year_total: ℝ)
  (bonus_percentage: ℝ)
  (h1: last_year_salary = 100000)
  (h2: last_year_bonus = 10000)
  (h3: this_year_total = 220000)
  (h4: bonus_percentage = 0.10) :
  (∃ (this_year_salary: ℝ), this_year_salary + this_year_salary * bonus_percentage = this_year_total ∧ this_year_salary = 200000) :=
by {
  have : last_year_bonus / last_year_salary = bonus_percentage, by sorry,
  simp at this,
  use 200000,
  split,
  simp, 
  sorry
}

end johns_salary_before_bonus_l646_646806


namespace sum_of_squares_of_products_eq_factorial_l646_646670

open Nat

-- Definitions for sets and conditions
def validSets (n : ℕ) : List (List ℕ) :=
  List.filter (λ s, ∀ i ∈ s, ∀ j ∈ s, i ≠ j + 1 ∧ i ≠ j - 1) (List.powerset (List.range (n + 1)))

def productOfSet (s : List ℕ) : ℕ := s.foldr (*) 1

def sumOfSquaresOfProducts (n : ℕ) : ℕ :=
  (validSets n).foldr (λ s acc, acc + (productOfSet s)^2) 0

-- The theorem statement
theorem sum_of_squares_of_products_eq_factorial (n : ℕ) : sumOfSquaresOfProducts n = (nat.factorial (n + 1)) - 1 := by
  sorry

end sum_of_squares_of_products_eq_factorial_l646_646670


namespace little_johns_money_left_l646_646457

def J_initial : ℝ := 7.10
def S : ℝ := 1.05
def F : ℝ := 1.00

theorem little_johns_money_left :
  J_initial - (S + 2 * F) = 4.05 :=
by sorry

end little_johns_money_left_l646_646457


namespace series_logarithmic_sum_l646_646204

theorem series_logarithmic_sum :
  ∑' n : ℕ, if n ≥ 2 then (Real.log (n^3 + 1) - Real.log (n^3 - 1)) else 0 = Real.log (3 / 2) :=
by
  sorry

end series_logarithmic_sum_l646_646204


namespace possible_number_of_friends_l646_646631

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l646_646631


namespace closest_integer_to_series_sum_l646_646700

theorem closest_integer_to_series_sum :
  let s := 1000 * ∑ n in finset.range (20000 + 1), if n ≥ 10 then (1 : ℝ) / (n^2 - 4) else 0
  abs (s - 99) <= 0.5 := 
by 
  -- The proof will be filled here.
  sorry

end closest_integer_to_series_sum_l646_646700


namespace vector_decomposition_l646_646972

noncomputable def x : ℝ × ℝ × ℝ := (5, 15, 0)
noncomputable def p : ℝ × ℝ × ℝ := (1, 0, 5)
noncomputable def q : ℝ × ℝ × ℝ := (-1, 3, 2)
noncomputable def r : ℝ × ℝ × ℝ := (0, -1, 1)

theorem vector_decomposition : x = (4 : ℝ) • p + (-1 : ℝ) • q + (-18 : ℝ) • r :=
by
  sorry

end vector_decomposition_l646_646972


namespace intersection_S_T_l646_646752

open Set

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} :=
by
  sorry

end intersection_S_T_l646_646752


namespace average_of_distinct_k_l646_646314

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l646_646314


namespace transformation_C1_to_C2_l646_646720

def C1 (x : ℝ) : ℝ := Real.sin (π / 2 + 2 * x)
def C2 (x : ℝ) : ℝ := -Real.cos (5 * π / 6 - 3 * x)

theorem transformation_C1_to_C2 :
  ∀ x, C2 x = C1 ((x + π / 18) * 3 / 2) :=
sorry

end transformation_C1_to_C2_l646_646720


namespace find_three_digit_number_l646_646689

def is_valid_three_digit_number (M G U : ℕ) : Prop :=
  M ≠ G ∧ G ≠ U ∧ M ≠ U ∧ 
  0 ≤ M ∧ M ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧
  100 * M + 10 * G + U = (M + G + U) * (M + G + U - 2)

theorem find_three_digit_number : ∃ (M G U : ℕ), 
  is_valid_three_digit_number M G U ∧
  100 * M + 10 * G + U = 195 :=
by
  sorry

end find_three_digit_number_l646_646689


namespace group_of_friends_l646_646584

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l646_646584


namespace increasing_interval_of_log_function_l646_646891

noncomputable def log_base_0_2 (x : ℝ) : ℝ := Real.log x / Real.log 0.2

theorem increasing_interval_of_log_function : 
  (∃ I : Set ℝ, I = {x : ℝ | x < 1} ∧ ∀ x ∈ I, 0 < x^2 - 6 * x + 5 ∧ 
  ∀ x ∈ I, ∀ y ∈ I, x < y → log_base_0_2 (x^2 - 6 * x + 5) < log_base_0_2 (y^2 - 6 * y + 5)) :=
begin
  sorry
end

end increasing_interval_of_log_function_l646_646891


namespace digit_150_l646_646933

def decimal_rep : ℚ := 5 / 13

def cycle_length : ℕ := 6

theorem digit_150 (n : ℕ) (h : n = 150) : Nat.digit (n % cycle_length) (decimal_rep) = 5 := by
  sorry

end digit_150_l646_646933


namespace find_x4_l646_646103

theorem find_x4 (x_1 x_2 : ℝ) (h1 : 0 < x_1) (h2 : x_1 < x_2) 
  (P : (ℝ × ℝ)) (Q : (ℝ × ℝ)) (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (R : (ℝ × ℝ)) (x_4 : ℝ) :
  R = ((x_1 + x_2) / 2, (Real.log x_1 + Real.log x_2) / 2) →
  Real.log x_4 = (Real.log x_1 + Real.log x_2) / 2 →
  x_4 = Real.sqrt 1000 :=
by 
  intro hR hT
  sorry

end find_x4_l646_646103


namespace find_distance_to_school_l646_646805

variable (v d : ℝ)
variable (h_rush_hour : d = v * (1 / 2))
variable (h_no_traffic : d = (v + 20) * (1 / 4))

theorem find_distance_to_school (h_rush_hour : d = v * (1 / 2)) (h_no_traffic : d = (v + 20) * (1 / 4)) : d = 10 := by
  sorry

end find_distance_to_school_l646_646805


namespace equation_conditions_l646_646885

theorem equation_conditions (m n : ℤ) (h1 : m ≠ 1) (h2 : n = 1) :
  ∃ x : ℤ, (m - 1) * x = 3 ↔ m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end equation_conditions_l646_646885


namespace tangent_line_a_1_max_value_a_2_max_value_interval_l646_646354

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * x

-- Given a=1, find the tangent line to f(x) at x=e
theorem tangent_line_a_1 : 
  let a := 1
  let x := Real.exp 1
  tangent_equation := (y - (f x a)) = (1 / x - a) * (x - e)
  tangent_equation = y = (1 / exp 1 - 1) * x := sorry

-- Given a=2, find the maximum value of f(x) 
theorem max_value_a_2 : 
  let a := 2
  max_value := f (1 / 2) a
  max_value = -1 - log 2 := sorry

-- Finding the max value of f(x) on [1, e] for a in different ranges
theorem max_value_interval : 
  ∀ (a : ℝ), 
  let interval_max_value :=
    if a ≤ 1 / Real.exp 1 then 1 - a * Real.exp 1
    else if 1 / Real.exp 1 < a ∧ a < 1 then -log a - 1
    else -a
  maximum := max (f 1 a) (f Real.exp 1 a)
  maximum = interval_max_value := sorry

end tangent_line_a_1_max_value_a_2_max_value_interval_l646_646354


namespace intersection_of_A_and_B_solution_set_x2_ax_minus_b_l646_646258

-- Define the sets A and B
def setA : Set ℝ := {x : ℝ | x^2 - 2 * x - 3 < 0}
def setB : Set ℝ := {x : ℝ | x^2 - 5 * x + 6 < 0}

-- Statement (1): Prove that A ∩ B = {x | -1 < x < 2}
theorem intersection_of_A_and_B : 
  setA ∩ setB = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

-- Assume the solution set of x^2 + ax + b < 0 is {x | -1 < x < 2}
variables {a b : ℝ}
def solution_set : Set ℝ := {x : ℝ | x^2 + a * x + b < 0}
def expected_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Conditions for a and b
axiom root_conditions : ∀ {a b : ℝ}, solution_set = expected_set → a = -1 ∧ b = -2

-- Statement (2): Prove the solution set of x^2 + ax - b < 0 is {x | x < -1 ∨ x > 2}
theorem solution_set_x2_ax_minus_b : 
  ∀ {a b : ℝ}, expected_set = {x : ℝ | x^2 + a * x + b < 0} →
  {x : ℝ | x^2 + a * x - b < 0} = {x : ℝ | x < -1 ∨ x > 2} :=
sorry

end intersection_of_A_and_B_solution_set_x2_ax_minus_b_l646_646258


namespace avg_of_k_with_positive_integer_roots_l646_646324

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l646_646324


namespace donuts_left_l646_646199

theorem donuts_left (t : ℕ) (c1 : ℕ) (c2 : ℕ) (c3 : ℝ) : t = 50 ∧ c1 = 2 ∧ c2 = 4 ∧ c3 = 0.5 
  → (t - c1 - c2) / 2 = 22 :=
by
  intros
  cases H with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5, h6]
  norm_num
  sorry

end donuts_left_l646_646199


namespace garden_area_l646_646428

theorem garden_area
  (total_posts : ℕ)
  (post_spacing : ℕ)
  (corner_posts : ℕ)
  (long_side_posts_condition : ∀ s : ℕ, 2 * s + 3 = (2 * s + 3)) :
  ∃ s l : ℕ, 
  let short_side_posts := s + 2 in
  let long_side_posts := 2 * short_side_posts + 3 in
  let short_side_len := (short_side_posts - 1) * post_spacing in
  let long_side_len := (long_side_posts - 1) * post_spacing in
  total_posts = 28 → 
  post_spacing = 3 → 
  corner_posts = 4 →
  (short_side_len * long_side_len = 630) :=
by
  sorry

end garden_area_l646_646428


namespace point_Q_rotates_clockwise_twice_omega_l646_646765

theorem point_Q_rotates_clockwise_twice_omega
  (ω : ℝ)
  (P : ℝ × ℝ → Prop)
  (hP : ∀ t, P (Real.cos (ω * t), Real.sin (ω * t)))
  (Q : ℝ × ℝ)
  (hQ : Q = (-2 * (Real.cos (ω * t)) * (Real.sin (ω * t)), (Real.sin (ω * t))^2 - (Real.cos (ω * t))^2)) :
  ∃ t, holds (λ t, (Q (Real.sin (2 * ω * t), -Real.cos (2 * ω * t))) = true) :=
begin
  sorry
end

end point_Q_rotates_clockwise_twice_omega_l646_646765


namespace group_of_friends_l646_646585

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l646_646585


namespace part_a_first_1000_decimal_places_zero_part_b_first_1000_decimal_places_zero_part_c_first_1000_decimal_places_zero_l646_646240

noncomputable def first_1000_decimal_places_zero_part_a : Prop :=
  ∀ (k : ℕ), k < 1000 → ∀ (d : ℕ), (d < 10) →
  let x := Real.exp (1999 * Real.log (6 + Real.sqrt 35)) in
  ∃ n : ℕ, (x - n) * 10^k < d / 10

noncomputable def first_1000_decimal_places_zero_part_b : Prop :=
  ∀ (k : ℕ), k < 1000 → ∀ (d : ℕ), (d < 10) →
  let x := Real.exp (1999 * Real.log (6 + Real.sqrt 37)) in
  ∃ n : ℕ, (x - n) * 10^k < d / 10

noncomputable def first_1000_decimal_places_zero_part_c : Prop :=
  ∀ (k : ℕ), k < 1000 → ∀ (d : ℕ), (d < 10) →
  let x := Real.exp (2000 * Real.log (6 + Real.sqrt 37)) in
  ∃ n : ℕ, (x - n) * 10^k < d / 10

theorem part_a_first_1000_decimal_places_zero : first_1000_decimal_places_zero_part_a :=
sorry

theorem part_b_first_1000_decimal_places_zero : first_1000_decimal_places_zero_part_b :=
sorry

theorem part_c_first_1000_decimal_places_zero : first_1000_decimal_places_zero_part_c :=
sorry

end part_a_first_1000_decimal_places_zero_part_b_first_1000_decimal_places_zero_part_c_first_1000_decimal_places_zero_l646_646240


namespace ratio_of_terms_l646_646792

noncomputable theory

open BigOperators

variables {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequences as functions.
def a (n : ℕ) : α := sorry  -- Define the arithmetic sequence a_n
def b (n : ℕ) : α := sorry  -- Define the arithmetic sequence b_n

-- Define the sums of the first n terms for each sequence.
def S (n : ℕ) : α := ∑ i in Finset.range n, a i
def T (n : ℕ) : α := ∑ i in Finset.range n, b i

-- The given condition about the ratios of the sums.
axiom ratio_condition (n : ℕ) : S n / T n = 2 * n / (3 * n + 1)

-- The proof task: show that the ratio of the 10th terms matches the given ratio under the conditions.
theorem ratio_of_terms :
  (a 10) / (b 10) = 19 / 29 :=
sorry

end ratio_of_terms_l646_646792


namespace sprinkles_subtraction_l646_646664

def initial_cans : ℕ := 12
def remaining_cans : ℕ := 3
def half_initial := initial_cans / 2

theorem sprinkles_subtraction :
  ∃ x : ℕ, half_initial - x = remaining_cans → x = 3 :=
by
  use 3
  simp
  sorry

end sprinkles_subtraction_l646_646664


namespace Lorelai_jellybeans_eaten_l646_646032

theorem Lorelai_jellybeans_eaten :
  ∀ (Gigi Rory Luke Lorelai : ℕ),
  Gigi = 15 → 
  Rory = Gigi + 30 → 
  Luke = 2 * Rory → 
  Lorelai = 3 * (Gigi + Luke) → 
  Lorelai = 315 :=
by 
  intros Gigi Rory Luke Lorelai hGigi hRory hLuke hLorelai
  rw [hGigi, hRory, hLuke, hLorelai]
  sorry

end Lorelai_jellybeans_eaten_l646_646032


namespace triangle_is_great_iff_right_isosceles_A_l646_646917

structure triangle :=
(A B C : Point)
(angle_A : Angle A B C)
(angle_B : Angle B C A)
(angle_C : Angle C A B)

def is_right_angle (A B C : Point) : Prop :=
∃ O : Point, Angle A O B = 90 ∧ O ∈ circumcircle A B C

def isosceles_at_A (A B C : Point) : Prop :=
dist A B = dist A C

def great_triangle (A B C : Point) : Prop :=
∀ D : Point, D ∈ segment B C →
  let P := foot_of_perpendicular D A B,
      Q := foot_of_perpendicular D A C,
      D' := reflection D P Q
  in D' ∈ circumcircle A B C

theorem triangle_is_great_iff_right_isosceles_A (A B C : Point) : great_triangle A B C ↔ (is_right_angle A B C ∧ isosceles_at_A A B C) :=
begin
  split,
  {
    intro h,
    -- To be proved: h implies right angle at A and isosceles triangle at A.
    sorry,
  },
  {
    intro h,
    -- To be proved: right angle at A and isosceles triangle at A implies h.
    sorry,
  }
end

end triangle_is_great_iff_right_isosceles_A_l646_646917


namespace exists_a_b_l646_646890

noncomputable def f : ℕ → ℕ
| 1 := 1
| (n + 1) := if h : ∃ m > 0, ∃ (AP : ℕ → ℕ), (AP 0 = n) ∧ (AP m = n) ∧ (∀ k < m, AP (k + 1) = AP k + 1) ∧ (∀ k ≤ m, f (AP k) = f n) 
             then max {m | ∃ (AP : ℕ → ℕ), (AP 0 = n) ∧ (AP m = n) ∧ (∀ k < m, AP (k + 1) = AP k + 1) ∧ (∀ k ≤ m, f (AP k) = f n)} 
             else 1

theorem exists_a_b : ∃ a b : ℕ, ∀ n : ℕ, 0 < n → f (a * n + b) = n + 2 := by
  sorry

end exists_a_b_l646_646890


namespace problem1_problem2_l646_646155

noncomputable def f (x : ℝ) := x^2
noncomputable def g (x : ℝ) := x - 1
noncomputable def F (x m : ℝ) := f x - m * g x + 1 - m - m^2

theorem problem1 (b : ℝ) : (∃ x : ℝ, f x < b * g x) → (b < 0 ∨ b > 4) :=
sorry

theorem problem2 (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |F x m| ≤ |F (x + 1) m|)
  → (-√(4/5) ≤ m ∧ m ≤ √(4/5) ∨ m ≥ 2) :=
sorry

end problem1_problem2_l646_646155


namespace percentage_of_whole_l646_646987

-- Define the conditions
def Part : ℕ := 70
def Whole : ℕ := 125

-- State the problem: Prove that 70 is 56% of 125
theorem percentage_of_whole (Part Whole : ℕ) (h1 : Part = 70) (h2 : Whole = 125) :
  (Part / Whole : ℚ) * 100 = 56 := 
by
  subst h1
  subst h2
  sorry

end percentage_of_whole_l646_646987


namespace max_value_expression_l646_646446

theorem max_value_expression : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 →
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 256 / 243 :=
by
  intros x y z hx hy hz hsum
  sorry

end max_value_expression_l646_646446


namespace odd_function_increasing_l646_646719

variables {f : ℝ → ℝ}

/-- Let f be an odd function defined on (-∞, 0) ∪ (0, ∞). 
If ∀ y z ∈ (0, ∞), y ≠ z → (f y - f z) / (y - z) > 0, then f(-3) > f(-5). -/
theorem odd_function_increasing {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ y z : ℝ, y > 0 → z > 0 → y ≠ z → (f y - f z) / (y - z) > 0) :
  f (-3) > f (-5) :=
sorry

end odd_function_increasing_l646_646719


namespace ducks_counted_l646_646222

theorem ducks_counted (x y : ℕ) (h1 : x + y = 300) (h2 : 2 * x + 4 * y = 688) : x = 256 :=
by
  sorry

end ducks_counted_l646_646222


namespace expression_evaluation_l646_646226

theorem expression_evaluation : (16^3 + 3 * 16^2 + 3 * 16 + 1 = 4913) :=
by
  sorry

end expression_evaluation_l646_646226


namespace reconstruct_quadrilateral_l646_646280

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A A'' B'' C'' D'' : V)

def trisect_segment (P Q R : V) : Prop :=
  Q = (1 / 3 : ℝ) • P + (2 / 3 : ℝ) • R

theorem reconstruct_quadrilateral
  (hB : trisect_segment A B A'')
  (hC : trisect_segment B C B'')
  (hD : trisect_segment C D C'')
  (hA : trisect_segment D A D'') :
  A = (2 / 26) • A'' + (6 / 26) • B'' + (6 / 26) • C'' + (12 / 26) • D'' :=
sorry

end reconstruct_quadrilateral_l646_646280


namespace find_y_l646_646778

-- Definitions
variables (y a b c d e : ℤ)

-- Conditions
def magic_sum := y + 23 + 104

def condition1 : a = y - 99 := by
  sorry

def condition2 : b = 2y - 203 := by
  sorry

def condition3 := 5 + y - 99 + 2y - 203 = magic_sum

-- Theorem
theorem find_y : y = 220 := by
  sorry

end find_y_l646_646778


namespace non_congruent_parallelograms_l646_646210

def side_lengths_sum (a b : ℕ) : Prop :=
  a + b = 25

def is_congruent (a b : ℕ) (a' b' : ℕ) : Prop :=
  (a = a' ∧ b = b') ∨ (a = b' ∧ b = a')

def non_congruent_count (n : ℕ) : Prop :=
  ∀ (a b : ℕ), side_lengths_sum a b → 
  ∃! (m : ℕ), is_congruent a b m b

theorem non_congruent_parallelograms :
  ∃ (n : ℕ), non_congruent_count n ∧ n = 13 :=
sorry

end non_congruent_parallelograms_l646_646210


namespace necessary_but_not_sufficient_condition_l646_646356

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem necessary_but_not_sufficient_condition (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f := 
sorry

end necessary_but_not_sufficient_condition_l646_646356


namespace roots_of_unity_l646_646858

theorem roots_of_unity (n : ℕ) (k : ℕ) (hk : k < n) : 
  ∃ (x : ℂ), x^n = 1 ∧ x = complex.exp (complex.I * (2 * k * real.pi / n)) :=
by
  sorry

end roots_of_unity_l646_646858


namespace possible_number_of_friends_l646_646625

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l646_646625


namespace number_of_friends_l646_646613

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l646_646613


namespace max_colors_for_valid_coloring_l646_646529

-- Define the 4x4 grid as a type synonym for a set of cells
def Grid4x4 := Fin 4 × Fin 4

-- Condition: Define a valid coloring function for a 4x4 grid
def valid_coloring (colors : ℕ) (f : Grid4x4 → Fin colors) : Prop :=
  ∀ i j : Fin 3, ∃ c : Fin colors, (f (i, j) = c ∨ f (i+1, j) = c) ∧ (f (i+1, j) = c ∨ f (i, j+1) = c)

-- The main theorem to prove
theorem max_colors_for_valid_coloring : 
  ∃ (colors : ℕ), colors = 11 ∧ ∀ f : Grid4x4 → Fin colors, valid_coloring colors f :=
sorry

end max_colors_for_valid_coloring_l646_646529


namespace investor_share_price_l646_646572

theorem investor_share_price (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) (price_per_share : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 40 →
  roi = 0.25 →
  ((dividend_rate * face_value) / price_per_share) = roi →
  price_per_share = 20 :=
by 
  intros h1 h2 h3 h4
  sorry

end investor_share_price_l646_646572


namespace positive_difference_abs_eq_15_l646_646956

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l646_646956


namespace pyramid_solution_l646_646982

noncomputable def pyramid_problem 
    (height : ℝ) (AB : ℝ) (SA : ℝ) (SB : ℝ) (SC : ℝ) 
    (sphere_touches : Π (face : ℝ), Bool) :=
  height = 2 * Real.sqrt 5 ∧
  AB = 6 ∧
  SA = 5 ∧
  SB = 7 ∧
  SC = 2 * Real.sqrt 10 ∧
  (∀ (face : ℝ), sphere_touches face) →
  (BC = 9 ∧
   CD = 24 / 5 ∧
   radius = 2 * Real.sqrt (6 / 5) ∧
   dihedral_angle = 2 * Real.arcsin (Real.sqrt (77 / 87)))

axiom sphere_touches_edges : ∀ (face : ℝ), Bool

theorem pyramid_solution :
  pyramid_problem 2 * Real.sqrt 5 6 5 7 2 * Real.sqrt 10 sphere_touches_edges :=
sorry

end pyramid_solution_l646_646982


namespace volume_of_box_l646_646430

-- Problem: Given a box with certain dimensions, prove its volume.
theorem volume_of_box : 
  let H := 12 
  let L := 3 * H 
  let W := L / 4 
  V = L * W * H
  in V = 3888 := 
by
  let H := 12 
  let L := 3 * H 
  let W := L / 4 
  let V := L * W * H
  show V = 3888, 
  sorry

end volume_of_box_l646_646430


namespace mutually_exclusive_not_complementary_l646_646253

-- Define the basic events and conditions
structure Pocket :=
(red : ℕ)
(black : ℕ)

-- Define the event type
inductive Event
| atleast_one_black : Event
| both_black : Event
| atleast_one_red : Event
| both_red : Event
| exactly_one_black : Event
| exactly_two_black : Event
| none_black : Event

def is_mutually_exclusive (e1 e2 : Event) : Prop :=
  match e1, e2 with
  | Event.exactly_one_black, Event.exactly_two_black => true
  | Event.exactly_two_black, Event.exactly_one_black => true
  | _, _ => false

def is_complementary (e1 e2 : Event) : Prop :=
  e1 = Event.none_black ∧ e2 = Event.both_red ∨
  e1 = Event.both_red ∧ e2 = Event.none_black

-- Given conditions
def pocket : Pocket := { red := 2, black := 2 }

-- Proof problem setup
theorem mutually_exclusive_not_complementary : 
  is_mutually_exclusive Event.exactly_one_black Event.exactly_two_black ∧
  ¬ is_complementary Event.exactly_one_black Event.exactly_two_black :=
by
  sorry

end mutually_exclusive_not_complementary_l646_646253


namespace divisor_is_36_l646_646398

theorem divisor_is_36
  (Dividend Quotient Remainder : ℕ)
  (h1 : Dividend = 690)
  (h2 : Quotient = 19)
  (h3 : Remainder = 6)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Divisor = 36 :=
sorry

end divisor_is_36_l646_646398


namespace part1_part2_l646_646747

-- Define the quadratic equation and its discriminant
def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

-- Define the conditions
def quadratic_equation (m : ℝ) : ℝ :=
  quadratic_discriminant 1 (-2) (-3 * m^2)

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem part1 (m : ℝ) : 
  quadratic_equation m > 0 :=
by
  sorry

-- Part 2: Find the value of m given the roots satisfy the equation α + 2β = 5
theorem part2 (α β m : ℝ) (h1 : α + β = 2) (h2 : α + 2 * β = 5) : 
  m = 1 ∨ m = -1 :=
by
  sorry


end part1_part2_l646_646747


namespace visual_range_with_telescope_l646_646569

theorem visual_range_with_telescope (original_range : ℝ) (percentage_increase : ℝ) (final_range : ℝ) 
  (h1 : original_range = 90) 
  (h2 : percentage_increase = 0.6667)
  (h3 : final_range = original_range + (original_range * percentage_increase)) : 
  final_range = 150 :=
by 
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end visual_range_with_telescope_l646_646569


namespace range_f_l646_646245

noncomputable def f (x : ℝ) : ℝ := 
  if x ≠ -5 then (3 * (x + 5) * (x - 4)) / (x + 5) else 0

theorem range_f : range f = set.univ \ (-27) := 
by
  sorry

end range_f_l646_646245


namespace positive_difference_abs_eq_15_l646_646955

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l646_646955


namespace ratio_of_dad_to_jayson_l646_646966

-- Define the conditions
def JaysonAge : ℕ := 10
def MomAgeWhenBorn : ℕ := 28
def MomCurrentAge (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ) : ℕ := MomAgeWhenBorn + JaysonAge
def DadCurrentAge (MomCurrentAge : ℕ) : ℕ := MomCurrentAge + 2

-- Define the proof problem
theorem ratio_of_dad_to_jayson (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ)
  (h1 : JaysonAge = 10) (h2 : MomAgeWhenBorn = 28) :
  DadCurrentAge (MomCurrentAge JaysonAge MomAgeWhenBorn) / JaysonAge = 4 :=
by 
  sorry

end ratio_of_dad_to_jayson_l646_646966


namespace cosine_sine_range_l646_646785

theorem cosine_sine_range (a b c : ℝ) (A B C : ℝ) (h : 0 < A ∧ A < π / 2) 
  (h1 : 0 < B ∧ B < π / 2) (h2 : 0 < C ∧ C < π / 2) 
  (acute_ABC : (a + b + c) * (a + c - b) = (2 + real.sqrt 3) * a * c) :
  ∃ (low high : ℝ), low = real.sqrt 3 / 2 ∧ high = 3 / 2 ∧ (low < (real.cos A + real.sin C) ∧ (real.cos A + real.sin C) < high) :=
by sorry

end cosine_sine_range_l646_646785


namespace unique_decomposition_l646_646859

-- Define functions representing the types of distributions
variables (F F_d F_abc F_sc : ℝ → ℝ)
variables (f : ℝ → ℝ)
variables (alpha1 alpha2 alpha3 : ℝ)

-- Assumptions on alpha coefficients
axiom nonnegative_coeffs : alpha1 ≥ 0 ∧ alpha2 ≥ 0 ∧ alpha3 ≥ 0
axiom sum_of_coeffs : alpha1 + alpha2 + alpha3 = 1

-- Definitions of the distribution functions
axiom F_d_def : ∀ x, F_d x = ∑ (k : ℕ) in {k | x ≥ k}, (F_d(k) - F_d(k - 1))
axiom F_abc_def : ∀ x, F_abc x = ∫ y in -∞..x, f y

-- Conditions on the function f
axiom f_nonnegative : ∀ x, 0 ≤ f x
axiom f_integrable : ∫ x in -∞..∞, f x = 1

-- Definition of singular distribution function
axiom F_sc_def : ∀ x, continuous F_sc x ∧ set.les_measure_zero {x | F_sc' x ≠ 0} -- assuming F_sc' is the derivative of F_sc representing growth points

-- Representation and uniqueness
theorem unique_decomposition :
  F = alpha1 * F_d + alpha2 * F_abc + alpha3 * F_sc → F ∈ { F | ∃ alpha1' alpha2' alpha3', alpha1' ≥ 0 ∧ alpha2' ≥ 0 ∧ alpha3' ≥ 0 ∧ alpha1' + alpha2' + alpha3' = 1 ∧ F = alpha1' * F_d + alpha2' * F_abc + alpha3' * F_sc } :=
sorry

end unique_decomposition_l646_646859


namespace conjugate_quadrant_l646_646879

def complex_quadrant (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "on border"

theorem conjugate_quadrant {z : ℂ} (h : (z - 3) * (2 - complex.I) = 5 * complex.I) :
  complex_quadrant (conj z) = "fourth quadrant" :=
sorry

end conjugate_quadrant_l646_646879


namespace range_of_a_l646_646751

theorem range_of_a (a : ℝ) (h : a > 0) :
  let A := {x : ℝ | x^2 + 2 * x - 8 > 0}
  let B := {x : ℝ | x^2 - 2 * a * x + 4 ≤ 0}
  (∃! x : ℤ, (x : ℝ) ∈ A ∩ B) → (13 / 6 ≤ a ∧ a < 5 / 2) :=
by
  sorry

end range_of_a_l646_646751


namespace true_propositions_is_one_l646_646907

-- Define propositions
def prop1 : Prop := ∀ (L1 L2 : Line) (P : Plane), 
  (is_perpendicular L1 P) ∧ (is_perpendicular L2 P) → is_parallel L1 L2

def prop2 : Prop := ∀ (a b : Line) (P : Point),
  is_skew_lines a b → ∃ (L : Line), 
  passes_through L P ∧ intersects L a ∧ intersects L b

def prop3 : Prop := ∀ (L : Line) (P : Plane), 
  is_parallel L P → ∀ (L_plane : Line), 
  (L_plane ∈ P) → is_parallel L L_plane

def prop4 : Prop := ∀ (L : Line) (P : Plane), 
  is_perpendicular L P → ∀ (L_plane : Line), 
  (L_plane ∈ P) → is_perpendicular L L_plane

-- Number of true propositions
def number_of_true_propositions : Nat :=
  if prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 then 1
  else if prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 then 2
  else if prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4 then 3
  else if prop1 ∧ prop2 ∧ prop3 ∧ prop4 then 4
  else if ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 then 1
  else if ¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4 then 2
  else if ¬prop1 ∧ prop2 ∧ prop3 ∧ prop4 then 3
  else if ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 then 1
  else if ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 then 2
  else if ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 then 1
  else 0

-- Theorem to prove the solution
theorem true_propositions_is_one : number_of_true_propositions = 1 :=
by
  sorry

end true_propositions_is_one_l646_646907


namespace incorrect_statement_C_l646_646478

-- Definitions of conditions as hypotheses
def spatial_temporal_restrictions := ∀ (P : Type), exists (s t : Set P), true
def organic_combination_individuals (P : Type) := ∀ (x : P), true
def population_unchanging (P : Type) := ∀ (x : P), population_characteristics x = population_characteristics x
def characteristics_not_individual (P : Type) := ∀ (x : P), ¬individual_characteristics x 

-- Definitions of what population characteristics might include
def population_characteristics (P : Type) := ∀ (x : P), exists (density birth_rate death_rate immigration_rate age_composition sex_ratio : ℕ), true
def individual_characteristics (P : Type) := ∀ (x : P), exists (unique_individual_traits : ℕ), true

-- Theorem statement proving that statement C is incorrect
theorem incorrect_statement_C (P : Type) (h1 : spatial_temporal_restrictions) 
  (h2 : organic_combination_individuals P) 
  (h3 : population_unchanging P) 
  (h4 : characteristics_not_individual P) : 
  ¬(∀ (x : P), population_characteristics x = population_characteristics x) :=
by
  sorry

end incorrect_statement_C_l646_646478


namespace average_k_l646_646301

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l646_646301


namespace part1_part2_l646_646748

-- Define the quadratic equation and its discriminant
def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

-- Define the conditions
def quadratic_equation (m : ℝ) : ℝ :=
  quadratic_discriminant 1 (-2) (-3 * m^2)

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem part1 (m : ℝ) : 
  quadratic_equation m > 0 :=
by
  sorry

-- Part 2: Find the value of m given the roots satisfy the equation α + 2β = 5
theorem part2 (α β m : ℝ) (h1 : α + β = 2) (h2 : α + 2 * β = 5) : 
  m = 1 ∨ m = -1 :=
by
  sorry


end part1_part2_l646_646748


namespace average_k_l646_646302

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l646_646302


namespace p_q_relation_n_le_2_p_q_relation_n_ge_3_l646_646821

open Real -- for ℝ
open Nat -- for ℕ

definition P (n : ℕ) (x : ℝ) : ℝ := (1-x)^(2*n-1)
definition Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n-1) * x + (n-1) * (2*n-1) * x^2

theorem p_q_relation_n_le_2 (n : ℕ+) (x : ℝ)
  (h_n_le_2 : n.val <= 2) : 
  if n.val = 1 then P n x = Q n x 
  else if x = 0 then P n x = Q n x
  else if x > 0 then P n x < Q n x
  else P n x > Q n x :=
sorry

theorem p_q_relation_n_ge_3 (n : ℕ+) (x : ℝ)
  (h_n_ge_3 : n.val >= 3) :
  if x = 0 then P n x = Q n x
  else if x > 0 then P n x < Q n x
  else P n x > Q n x :=
sorry

end p_q_relation_n_le_2_p_q_relation_n_ge_3_l646_646821


namespace distance_from_point_to_x_axis_l646_646064

theorem distance_from_point_to_x_axis (x y : ℝ) (hP : x = 3 ∧ y = -4) : abs(y) = 4 := by
  cases hP with
  | intro _ hy =>
    have : y = -4 := hy
    rw [this]
    simp
    sorry

end distance_from_point_to_x_axis_l646_646064


namespace count_valid_pairs_l646_646757

theorem count_valid_pairs (i j: ℤ) (h : 0 ≤ i ∧ i < j ∧ j ≤ 49) : 
  set.count (λ (i j : ℤ),  6^j - 6^i % 210 = 0 ∧ 0 ≤ i ∧ i < j ∧ j ≤ 49) = 600 := 
sorry

end count_valid_pairs_l646_646757


namespace person_picking_number_who_announced_6_is_1_l646_646048

theorem person_picking_number_who_announced_6_is_1
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ)
  (h₁ : a₁₀ + a₂ = 2)
  (h₂ : a₁ + a₃ = 4)
  (h₃ : a₂ + a₄ = 6)
  (h₄ : a₃ + a₅ = 8)
  (h₅ : a₄ + a₆ = 10)
  (h₆ : a₅ + a₇ = 12)
  (h₇ : a₆ + a₈ = 14)
  (h₈ : a₇ + a₉ = 16)
  (h₉ : a₈ + a₁₀ = 18)
  (h₁₀ : a₉ + a₁ = 20) :
  a₆ = 1 :=
by
  sorry

end person_picking_number_who_announced_6_is_1_l646_646048


namespace cross_product_l646_646237

open Real

def vector1 : ℝ × ℝ × ℝ := (4, -2, 5)
def vector2 : ℝ × ℝ × ℝ := (-1, 3, 6)

theorem cross_product (u v w : ℝ × ℝ × ℝ) (hx : u = vector1) (hy : v = vector2) :
  w = (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1) → w = (-27, -29, 10) :=
by
  intros _ _
  subst hx hy
  simp
  sorry

end cross_product_l646_646237


namespace positive_difference_eq_30_l646_646957

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l646_646957


namespace smallest_number_h_divisible_8_11_24_l646_646964

theorem smallest_number_h_divisible_8_11_24 : 
  ∃ h : ℕ, (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 ∧ h = 259 :=
by
  sorry

end smallest_number_h_divisible_8_11_24_l646_646964


namespace sequence_converges_to_limit_l646_646814

noncomputable theory

def sequence (a k : ℕ) (k_ge_2 : 2 ≤ k) (a_pos : 0 < a): ℕ → ℕ
| 1 := a
| (n + 2) := sequence (n + 1) + (k / sequence (n + 1))

theorem sequence_converges_to_limit (a k : ℕ) (k_ge_2 : 2 ≤ k) (a_pos : 0 < a) :
  ∃ L, ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (sequence a k k_ge_2 a_pos n - L) < ε ∧ 
  L = a + 1 + (k - a) :=
begin
  -- proof goes here
  sorry
end

end sequence_converges_to_limit_l646_646814


namespace positive_difference_abs_eq_l646_646945

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l646_646945


namespace normal_prob_calc_l646_646343

noncomputable def normal_prob :=
  let X := Normal 1 σ^2 in
  P(X > 2) = 0.15 ∧ P(0 ≤ X ∧ X ≤ 1) = 0.35

theorem normal_prob_calc (σ : ℝ) : normal_prob :=
by
  sorry

end normal_prob_calc_l646_646343


namespace symmetric_origin_sum_l646_646259

-- Definitions based on conditions
def point_symmetric_origin (M N : ℤ × ℤ) : Prop := 
  M = (-N.1, -N.2)

def M := (a : ℤ, -3)
def N := (4, b : ℤ)

-- Theorem statement
theorem symmetric_origin_sum (a b : ℤ) (h1 : M = (a, -3)) (h2 : N = (4, b)) (h3 : point_symmetric_origin M N) : 
  a + b = -1 := 
by
  sorry

end symmetric_origin_sum_l646_646259


namespace y_intercept_condition_l646_646362

variables {x1 x2 y1 y2 m : ℝ}
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Points A and B on the parabola
def A := (x1, parabola x1)
def B := (x2, parabola x2)

-- Midpoint of AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1)/2, (A.2 + B.2)/2)

-- Perpendicular bisector of AB with slope 2
def perpendicular_bisector (midpoint : ℝ × ℝ) (slope : ℝ) : ℝ × ℝ :=
  (midpoint.1, slope * midpoint.1 + midpoint.2)

-- Setting the y-intercept condition
def intercept_y (b' m : ℝ) : Prop :=
  b' = 5/16 + m

-- Final range condition
def y_intercept_range (b' : ℝ) : Prop :=
  b' > 9/32

theorem y_intercept_condition :
  forall m > -1/32,
  intercept_y (5/16 + m) m →
  y_intercept_range (5/16 + m) :=
by
  intros m hm h_intercept
  sorry

end y_intercept_condition_l646_646362


namespace solve_trig_eq_l646_646706

-- Given conditions
variables {n : ℕ}
variables {x : ℝ}

-- Helper definition for solutions
def solutions (n : ℕ) (x : ℝ) : Prop :=
  if n = 1 then
    ∃ m k : ℤ, x = 2 * π * m ∨ x = π / 2 + 2 * π * k
  else if ∃ l : ℕ, n = 4 * l - 2 ∨ n = 4 * l + 1 then
    ∃ m : ℤ, x = 2 * π * m
  else if ∃ l : ℕ, n = 4 * l ∨ n = 4 * l - 1 then
    ∃ m : ℤ, x = π * m
  else
    false

-- Main theorem statement
theorem solve_trig_eq (n : ℕ) (x : ℝ) (h₁ : n ∈ ℕ) (h₂ : sin x * sin (2 * x) * ... * sin (n * x) + cos x * cos (2 * x) * ... * cos (n * x) = 1) :
  solutions n x := sorry

end solve_trig_eq_l646_646706


namespace albrecht_correct_substitution_l646_646408

theorem albrecht_correct_substitution (a b : ℕ) (h : (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9) :
  (a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2) :=
by
  -- The proof will be filled in here
  sorry

end albrecht_correct_substitution_l646_646408


namespace angle_EFG_magnitude_l646_646547

/- Definitions -/
variables {E F G H : Type} [RealSpace E] [RealSpace H] [Point F E] [Point G E]
variables [RightAngledTriangle E F G] (midpoint H F G)

-- Given conditions
-- 1. Triangle EFG is right-angled at E
-- 2. H is the midpoint of FG => FH = HG
-- 3. EF = 2 * EH

/-- Proof statement -/
theorem angle_EFG_magnitude :
  (right_angle_at E F G) ∧ (midpoint H F G) ∧ (EF = 2 * EH) → (angle E F G = 60) :=
begin
  -- proof omitted
  sorry
end

end angle_EFG_magnitude_l646_646547


namespace positive_difference_eq_30_l646_646961

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l646_646961


namespace lindas_average_speed_l646_646015

theorem lindas_average_speed
  (dist1 : ℕ) (time1 : ℝ)
  (dist2 : ℕ) (time2 : ℝ)
  (h1 : dist1 = 450)
  (h2 : time1 = 7.5)
  (h3 : dist2 = 480)
  (h4 : time2 = 8) :
  (dist1 + dist2) / (time1 + time2) = 60 :=
by
  sorry

end lindas_average_speed_l646_646015


namespace circle_equation_l646_646238

-- Define the conditions
def chord_length_condition (a b r : ℝ) : Prop := r^2 = a^2 + 1
def arc_length_condition (b r : ℝ) : Prop := r^2 = 2 * b^2
def min_distance_condition (a b : ℝ) : Prop := a = b

-- The main theorem stating the final answer
theorem circle_equation (a b r : ℝ) (h1 : chord_length_condition a b r)
    (h2 : arc_length_condition b r) (h3 : min_distance_condition a b) :
    ((x - a)^2 + (y - a)^2 = 2) ∨ ((x + a)^2 + (y + a)^2 = 2) :=
sorry

end circle_equation_l646_646238


namespace tangent_line_at_0_eq_y_2x_1_number_of_zeros_of_h_l646_646355

def g (x : ℝ) : ℝ := Real.exp x * (x + 1)

def tangent_eq (m b : ℝ) :=
  ∀ x : ℝ, (m * x + b)

-- Statement for the tangent line proof
theorem tangent_line_at_0_eq_y_2x_1 :
  let p := (0, 1)
  let gx := g x
  deriv g 0 = 2 ∧ g 0 = 1 →
  tangent_eq 2 1 :=
by {
  sorry
}

-- Defining h(x) with the parameters
def h (a x : ℝ) : ℝ := g x - a * (x^3 + x^2)

-- Statement for the zeros of the function h(x)
theorem number_of_zeros_of_h (a : ℝ) (h₀ : a > 0) (hx : ∀ x, x > 0) :
  (a = Real.exp 2 / 4 ∧ ∃ z : ℝ, h a z = 0 ∧ h 0 = 1) ∨
  (a > Real.exp 2 / 4 ∧ ∃ z w : ℝ, z ≠ w ∧ h a z = 0 ∧ h a w = 0) ∨
  (0 < a ∧ a < Real.exp 2 / 4 ∧ ¬∃ z : ℝ, h a z = 0) :=
by {
  sorry
}

end tangent_line_at_0_eq_y_2x_1_number_of_zeros_of_h_l646_646355


namespace supplementary_angles_difference_l646_646079

theorem supplementary_angles_difference 
  (x : ℝ) 
  (h1 : 5 * x + 3 * x = 180) 
  (h2 : 0 < x) : 
  abs (5 * x - 3 * x) = 45 :=
by sorry

end supplementary_angles_difference_l646_646079


namespace volume_transformed_parallelepiped_l646_646089

variables (a b c : ℝ^3)

-- Given condition
def volume_determined_by_abc : ℝ := abs (a • (b × c)) = 8

-- Proving the volume of the new parallelepiped
theorem volume_transformed_parallelepiped (h : volume_determined_by_abc a b c) : 
  abs ((2 • a - b) • ((b + 2 • c) × (c + 4 • a))) = 16 :=
sorry

end volume_transformed_parallelepiped_l646_646089


namespace product_of_sequence_lt_two_l646_646270

theorem product_of_sequence_lt_two {a : ℕ → ℝ} (h₁ : a 1 = 1/2)
  (h₂ : ∀ n, a (n + 1) = (a n)^2 / (1 + sin (a n))) :
  ∀ n, ∏ i in Finset.range n, (1 + a i) < 2 := by
sorry

end product_of_sequence_lt_two_l646_646270


namespace arithmetic_sequence_sum_l646_646790

variable {α : Type*} [LinearOrderedField α]
variable (a₁ d : α)

noncomputable def a_n (n : ℕ) : α := a₁ + n * d

theorem arithmetic_sequence_sum
  (h : 2 * (a_n a₁ d 6) = (a_n a₁ d 8) + 7) :
  let S₉ := (9 / 2) * ((a_n a₁ d 0) + (a_n a₁ d 8))
  in S₉ = 63 := by
  sorry

end arithmetic_sequence_sum_l646_646790


namespace hannah_mugs_problem_l646_646368

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end hannah_mugs_problem_l646_646368


namespace cover_square_with_rectangles_l646_646134

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l646_646134


namespace find_unknown_blankets_rate_l646_646996

noncomputable def unknown_blankets_rate : ℝ :=
  let total_cost_3_blankets := 3 * 100
  let discount := 0.10 * total_cost_3_blankets
  let cost_3_blankets_after_discount := total_cost_3_blankets - discount
  let cost_1_blanket := 150
  let tax := 0.15 * cost_1_blanket
  let cost_1_blanket_after_tax := cost_1_blanket + tax
  let total_avg_price_per_blanket := 150
  let total_blankets := 6
  let total_cost := total_avg_price_per_blanket * total_blankets
  (total_cost - cost_3_blankets_after_discount - cost_1_blanket_after_tax) / 2

theorem find_unknown_blankets_rate : unknown_blankets_rate = 228.75 :=
  by
    sorry

end find_unknown_blankets_rate_l646_646996


namespace sum_of_ratios_of_squares_l646_646084

theorem sum_of_ratios_of_squares (r : ℚ) (a b c : ℤ) (h1 : r = 45 / 64) 
  (h2 : r = (a * (Real.sqrt b)) / c) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hc : c = 8) : a + b + c = 16 := 
by
  sorry

end sum_of_ratios_of_squares_l646_646084


namespace trapezoid_diagonal_relation_l646_646497

variables {A_1 A_2 A_3 A_4 : Type*}
variables (e f r_1 r_2 r_3 r_4 : ℝ)

def is_trapezoid (A_1 A_2 A_3 A_4 : Type*) := ∀ (e f : ℝ), (A_1A_3 = e) ∧ (A_2A_4 = f)

theorem trapezoid_diagonal_relation
  (r₁ r₂ r₃ r₄ e f : ℝ)
  (h_trapezoid : is_trapezoid A_1 A_2 A_3 A_4)
  (h_radius_1 : r_1 = circumradius (A_2, A_3, A_4))
  (h_radius_2 : r_2 = circumradius (A_1, A_3, A_4))
  (h_radius_3 : r_3 = circumradius (A_1, A_2, A_4))
  (h_radius_4 : r_4 = circumradius (A_1, A_2, A_3)) :
  (r₂ + r₄) / e = (r₁ + r₃) / f :=
begin
  sorry
end

end trapezoid_diagonal_relation_l646_646497


namespace probability_gpa_at_least_3_is_2_over_9_l646_646843

def gpa_points (grade : ℕ) : ℕ :=
  match grade with
  | 4 => 4 -- A
  | 3 => 3 -- B
  | 2 => 2 -- C
  | 1 => 1 -- D
  | _ => 0 -- otherwise

def probability_of_GPA_at_least_3 : ℚ :=
  let points_physics := gpa_points 4
  let points_chemistry := gpa_points 4
  let points_biology := gpa_points 3
  let total_known_points := points_physics + points_chemistry + points_biology
  let required_points := 18 - total_known_points -- 18 points needed in total for a GPA of at least 3.0
  -- Probabilities in Mathematics:
  let prob_math_A := 1 / 9
  let prob_math_B := 4 / 9
  let prob_math_C :=  4 / 9
  -- Probabilities in Sociology:
  let prob_soc_A := 1 / 3
  let prob_soc_B := 1 / 3
  let prob_soc_C := 1 / 3
  -- Calculate the total probability of achieving at least 7 points from Mathematics and Sociology
  let prob_case_1 := prob_math_A * prob_soc_A -- Both A in Mathematics and Sociology
  let prob_case_2 := prob_math_A * prob_soc_B -- A in Mathematics and B in Sociology
  let prob_case_3 := prob_math_B * prob_soc_A -- B in Mathematics and A in Sociology
  prob_case_1 + prob_case_2 + prob_case_3 -- Total Probability

theorem probability_gpa_at_least_3_is_2_over_9 : probability_of_GPA_at_least_3 = 2 / 9 :=
by sorry

end probability_gpa_at_least_3_is_2_over_9_l646_646843


namespace one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646920

theorem one_fiftieth_digit_of_five_over_thirteen_is_five :
  (decimal_fraction 5 13).digit 150 = 5 :=
by sorry

end one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646920


namespace Kataleya_bought_peaches_l646_646642

-- Define the conditions as assumptions
def store_discount (total_amount: ℝ) : ℝ :=
  let n := total_amount / 10
  2 * n.floor

def actual_spent (paid_amount discount: ℝ) : ℝ :=
  paid_amount + discount

def peach_cost : ℝ := 0.40

def peaches_bought (amount: ℝ) (cost: ℝ) : ℕ :=
  (amount / cost).nat_abs

-- Formulate the theorem to prove
theorem Kataleya_bought_peaches (paid_amount : ℝ) (peach_cost : ℝ) (total_peaches : ℕ) :
  paid_amount = 128 → peach_cost = 0.40 → total_peaches = 380 :=
by
  intros h1 h2
  sorry

end Kataleya_bought_peaches_l646_646642


namespace sum_S_takes_both_values_l646_646475

def S (x : Real) (a : Fin 32 → Real) : Real :=
  (32 : Fin 33).toList.reverse.foldl (λ acc i, acc + a ⟨i, Nat.lt_succ_of_lt i.2⟩ * Real.cos (i * x)) (Real.cos (32 * x))

theorem sum_S_takes_both_values (a : Fin 32 → Real) : ∃ x y : Real, S x a > 0 ∧ S y a < 0 :=
  sorry

end sum_S_takes_both_values_l646_646475


namespace positive_difference_abs_eq_15_l646_646948

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l646_646948


namespace positive_difference_eq_30_l646_646960

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l646_646960


namespace problem_statement_l646_646733

open Real

noncomputable def log_base (a x : ℝ) := (log x) / (log a)

def f (a x : ℝ) :=
if x < 0 then log_base a (-x)
else log_base a x

theorem problem_statement (f : ℝ → ℝ) (h1 : ∀ x, f x = f (-x)) (h2 : ∀ x > 0, f x = log_base a x) (h3 : f 5 = 2)
  : (a = sqrt 5 ∧ 
     (∀ x, f x = (if x < 0 then log_base (sqrt 5) (-x) else log_base (sqrt 5) x)) ∧ 
     (solution_set : set ℝ, {x | f x > 4} = set.Ioo 25 ∞ ∪ set.Ioo (-∞) (-25))) :=
begin
  sorry
end

end problem_statement_l646_646733


namespace smallest_num_rectangles_l646_646126

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l646_646126


namespace pairs_of_green_shirted_students_l646_646779

theorem pairs_of_green_shirted_students
  (red_students : ℕ) (green_students : ℕ) (total_students : ℕ) (total_pairs : ℕ)
  (red_red_pairs : ℕ)
  (H1 : red_students = 63)
  (H2 : green_students = 81)
  (H3 : total_students = 144)
  (H4 : total_pairs = 72)
  (H5 : red_red_pairs = 26)
  (H6 : total_students = red_students + green_students)
  (H7 : 2 * total_pairs = total_students) :
  let green_green_pairs := (green_students - (red_students - 2 * red_red_pairs)) / 2 in
  green_green_pairs = 35 :=
by
  intros
  sorry

end pairs_of_green_shirted_students_l646_646779


namespace exists_decreasing_then_increasing_sequence_not_exists_increasing_then_decreasing_sequence_l646_646525

-- Definitions
def sequence (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, x i ^ n

-- Problem (a): There exists a sequence that decreases up to a_5 and then starts increasing
theorem exists_decreasing_then_increasing_sequence :
  ∃ (x : ℕ → ℝ), (∀ i, x i > 0) ∧ sequence x 1 > sequence x 2 ∧
  sequence x 2 > sequence x 3 ∧ sequence x 3 > sequence x 4 ∧ 
  sequence x 4 > sequence x 5 ∧ sequence x 5 < sequence x 6 :=
sorry

-- Problem (b): There does not exist a sequence that increases up to a_5 and then starts decreasing 
theorem not_exists_increasing_then_decreasing_sequence :
  ¬ ∃ (x : ℕ → ℝ), (∀ i, x i > 0) ∧ sequence x 1 < sequence x 2 ∧
  sequence x 2 < sequence x 3 ∧ sequence x 3 < sequence x 4 ∧ 
  sequence x 4 < sequence x 5 ∧ sequence x 5 > sequence x 6 :=
sorry

end exists_decreasing_then_increasing_sequence_not_exists_increasing_then_decreasing_sequence_l646_646525


namespace possible_number_of_friends_l646_646630

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l646_646630


namespace arithmetic_sequence_sum_l646_646791

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_sum (h1 : a 2 + a 3 = 2) (h2 : a 4 + a 5 = 6) : a 5 + a 6 = 8 :=
sorry

end arithmetic_sequence_sum_l646_646791


namespace find_value_of_c_l646_646771

theorem find_value_of_c (a c : ℝ) (h_parallel: 3/6 = -2/a)
  (h_distance :  real.sqrt 13 * abs((c/2 + 1)) = 2 * real.sqrt 13) :
  c = 2 ∨ c = -6 :=
by
  sorry

end find_value_of_c_l646_646771


namespace annabelle_savings_l646_646654

noncomputable def weeklyAllowance : ℕ := 30
noncomputable def junkFoodFraction : ℚ := 1 / 3
noncomputable def sweetsCost : ℕ := 8

theorem annabelle_savings :
  let junkFoodCost := weeklyAllowance * junkFoodFraction
  let totalSpent := junkFoodCost + sweetsCost
  let savings := weeklyAllowance - totalSpent
  savings = 12 := 
by
  sorry

end annabelle_savings_l646_646654


namespace original_amount_of_potatoes_l646_646427

theorem original_amount_of_potatoes (P : ℕ) 
    (Gina : ℕ := 69) 
    (Tom : ℕ := 2 * Gina) 
    (Anne : ℕ := Tom / 3) 
    (remaining : ℕ := 47) :
    P - Gina - Tom - Anne = remaining → 
    P = 300 := 
by 
  intros h
  have h1 : Gina = 69 := rfl
  have h2 : Tom = 2 * Gina := rfl
  have h3 : Anne = Tom / 3 := rfl
  have h4 : remaining = 47 := rfl
  rw [h1, h2, h3, h4] at h
  sorry

end original_amount_of_potatoes_l646_646427


namespace sum_of_solutions_eq_neg4_l646_646816

theorem sum_of_solutions_eq_neg4 :
  ∃ (n : ℕ) (solutions : Fin n → ℝ × ℝ),
    (∀ i, ∃ (x y : ℝ), solutions i = (x, y) ∧ abs (x - 3) = abs (y - 9) ∧ abs (x - 9) = 2 * abs (y - 3)) ∧
    (Finset.univ.sum (fun i => (solutions i).1 + (solutions i).2) = -4) :=
sorry

end sum_of_solutions_eq_neg4_l646_646816


namespace friends_game_l646_646609

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l646_646609


namespace domain_is_correct_l646_646883

def domain_of_function (f : ℝ → ℝ) : Set ℝ := 
  {x : ℝ | -3 ≤ x ∧ x ≤ 3 ∧ x ≠ 2}

theorem domain_is_correct :
  (∀ x, (2 - x ≠ 0) ∧ (9 - x^2 ≥ 0) ↔ x ∈ domain_of_function (λ x, (1 / (2 - x) + real.sqrt (9 - x^2)))) :=
  sorry

end domain_is_correct_l646_646883


namespace hannah_mugs_problem_l646_646367

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end hannah_mugs_problem_l646_646367


namespace notebook_width_l646_646538

theorem notebook_width
  (circumference : ℕ)
  (length : ℕ)
  (width : ℕ)
  (H1 : circumference = 46)
  (H2 : length = 9)
  (H3 : circumference = 2 * (length + width)) :
  width = 14 :=
by
  sorry -- proof is omitted

end notebook_width_l646_646538


namespace limit_proof_l646_646490

variable {α : Type*} [TopologicalSpace α] [NormedField α] {f : α → α} {x₀ : α}

-- Given conditions
variable (h_diff : DifferentiableAt α f x₀)
variable (h_deriv : deriv f x₀ = 1)

-- Statement to prove
theorem limit_proof : (tendsto (λ (Δx : α), (f (x₀ + 2 * Δx) - f x₀) / Δx) (𝓝 0) (𝓝 2)) :=
sorry

end limit_proof_l646_646490


namespace simplest_square_root_l646_646533

-- Define the square roots given in the options
def sqrt_option_A : ℝ := real.sqrt 0.1
def sqrt_option_B : ℝ := real.sqrt 8
def sqrt_option_C (a : ℝ) : ℝ := real.sqrt (a^2)
def sqrt_option_D : ℝ := real.sqrt 3

-- Define what it means for a square root to be in simplest form
def simplest_form (x : ℝ) : Prop := 
  ∀ y : ℝ, real.sqrt y = x → y = 3

-- The main theorem (problem statement)
theorem simplest_square_root : simplest_form (real.sqrt 3) := sorry

end simplest_square_root_l646_646533


namespace two_colonies_reach_limit_same_time_l646_646974

theorem two_colonies_reach_limit_same_time
  (doubles_in_size : ∀ (n : ℕ), n = n * 2)
  (reaches_limit_in_25_days : ∃ N : ℕ, ∀ t : ℕ, t = 25 → N = N * 2^t) :
  ∀ t : ℕ, t = 25 := sorry

end two_colonies_reach_limit_same_time_l646_646974


namespace tax_percentage_excess_l646_646777

/--
In Country X, each citizen is taxed an amount equal to 15 percent of the first $40,000 of income,
plus a certain percentage of all income in excess of $40,000. A citizen of Country X is taxed a total of $8,000
and her income is $50,000.

Prove that the percentage of the tax on the income in excess of $40,000 is 20%.
-/
theorem tax_percentage_excess (total_tax : ℝ) (first_income : ℝ) (additional_income : ℝ) (income : ℝ) (tax_first_part : ℝ) (tax_rate_first_part : ℝ) (tax_rate_excess : ℝ) (tax_excess : ℝ) :
  total_tax = 8000 →
  first_income = 40000 →
  additional_income = 10000 →
  income = first_income + additional_income →
  tax_rate_first_part = 0.15 →
  tax_first_part = tax_rate_first_part * first_income →
  tax_excess = total_tax - tax_first_part →
  tax_rate_excess * additional_income = tax_excess →
  tax_rate_excess = 0.20 :=
by
  intro h_total_tax h_first_income h_additional_income h_income h_tax_rate_first_part h_tax_first_part h_tax_excess h_tax_equation
  sorry

end tax_percentage_excess_l646_646777


namespace rationalize_denominator_l646_646861

theorem rationalize_denominator :
  ∃ A B C D : ℤ,
  (1 ≤ A) ∧ (B = 2) ∧ (C = 5) ∧ (D = 4) ∧ (D > 0) ∧ (∃ k : ℤ, (k^2).natAbs ∣ B → False) ∧ 
  (A + B + C + D = 12) ∧ 
  (real.sqrt 50 / (real.sqrt 25 - real.sqrt 5) = (A * real.sqrt (B : ℝ) + C) / D) :=
by
  sorry

end rationalize_denominator_l646_646861


namespace odd_periodic_function_l646_646067

noncomputable def f : ℤ → ℤ := sorry

theorem odd_periodic_function (f_odd : ∀ x : ℤ, f (-x) = -f x)
  (period_f_3x1 : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 3) + 1))
  (f_one : f 1 = -1) : f 2006 = 1 :=
sorry

end odd_periodic_function_l646_646067


namespace remainder_of_power_mod_2000_l646_646703

theorem remainder_of_power_mod_2000 :
  let N := 2^4 in
  2^(2^N) % 2000 = 536 :=
by 
  let N := 2^4
  have h1 : N = 16 := by norm_num
  have h2 : 2^(2^16) % 2000 = 536 := sorry
  exact h2

end remainder_of_power_mod_2000_l646_646703


namespace average_k_positive_int_roots_l646_646336

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l646_646336


namespace average_k_positive_int_roots_l646_646338

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l646_646338


namespace probability_diff_topics_l646_646178

theorem probability_diff_topics
  (num_topics : ℕ)
  (num_combinations : ℕ)
  (num_different_combinations : ℕ)
  (h1 : num_topics = 6)
  (h2 : num_combinations = num_topics * num_topics)
  (h3 : num_combinations = 36)
  (h4 : num_different_combinations = num_topics * (num_topics - 1))
  (h5 : num_different_combinations = 30) :
  (num_different_combinations / num_combinations) = 5 / 6 := 
by 
  sorry

end probability_diff_topics_l646_646178


namespace inequality_x4_y4_z2_l646_646546

theorem inequality_x4_y4_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^4 + y^4 + z^2 ≥  xyz * 8^(1/2) :=
  sorry

end inequality_x4_y4_z2_l646_646546


namespace smallest_number_divisible_conditions_l646_646561

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l646_646561


namespace mowing_lawn_time_l646_646018

def maryRate := 1 / 3
def tomRate := 1 / 4
def combinedRate := 7 / 12
def timeMaryAlone := 1
def lawnLeft := 1 - (timeMaryAlone * maryRate)

theorem mowing_lawn_time:
  (7 / 12) * (8 / 7) = (2 / 3) :=
by
  sorry

end mowing_lawn_time_l646_646018


namespace coins_remainder_l646_646562

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l646_646562


namespace average_k_positive_int_roots_l646_646341

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l646_646341


namespace time_for_train_to_cross_platform_stationary_l646_646581

theorem time_for_train_to_cross_platform_stationary (speed_kmph : ℝ) (length_train length_platform : ℝ) : 
  speed_kmph = 72 → 
  length_train = 50 → 
  length_platform = 250 → 
  let distance := length_train + length_platform in
  let speed_mps := speed_kmph * (5 / 18) in
  let time := distance / speed_mps in
  time = 15 := 
by 
  -- Place the proof here
  sorry

end time_for_train_to_cross_platform_stationary_l646_646581


namespace total_chairs_carried_l646_646435

theorem total_chairs_carried :
  ∃ total_chairs : ℕ,
  let trips_Kingsley := 10,
      trips_Friend1 := 11,
      trips_Friend2 := 12,
      trips_Friend3 := 13,
      trips_Friend4 := 14,
      trips_Friend5 := 15,
      chairs_per_trip_Kingsley := 7,
      chairs_per_trip_Friend1 := 6,
      chairs_per_trip_Friend2 := 8,
      chairs_per_trip_Friend3 := 5,
      chairs_per_trip_Friend4 := 9,
      chairs_per_trip_Friend5 := 7 in
  total_chairs = (chairs_per_trip_Kingsley * trips_Kingsley) +
                 (chairs_per_trip_Friend1 * trips_Friend1) +
                 (chairs_per_trip_Friend2 * trips_Friend2) +
                 (chairs_per_trip_Friend3 * trips_Friend3) +
                 (chairs_per_trip_Friend4 * trips_Friend4) +
                 (chairs_per_trip_Friend5 * trips_Friend5) ∧
  total_chairs = 528 :=
by
  sorry

end total_chairs_carried_l646_646435


namespace smallest_positive_angle_l646_646088

theorem smallest_positive_angle (deg : ℤ) (k : ℤ) (h : deg = -2012) : ∃ m : ℤ, m = 148 ∧ 0 ≤ m ∧ m < 360 ∧ (∃ n : ℤ, deg + 360 * n = m) :=
by
  sorry

end smallest_positive_angle_l646_646088


namespace at_least_thirty_distinct_distances_l646_646463

theorem at_least_thirty_distinct_distances (pts : Fin 2004 → ℝ × ℝ) :
  ∃ s : Set ℝ, s ⊆ { dist (pts i) (pts j) | i j : Fin 2004 } ∧ s.card ≥ 30 :=
sorry

end at_least_thirty_distinct_distances_l646_646463


namespace three_digit_problem_l646_646691

theorem three_digit_problem :
  ∃ (M Γ U : ℕ), 
    M ≠ Γ ∧ M ≠ U ∧ Γ ≠ U ∧
    M ≤ 9 ∧ Γ ≤ 9 ∧ U ≤ 9 ∧
    100 * M + 10 * Γ + U = (M + Γ + U) * (M + Γ + U - 2) ∧
    100 * M + 10 * Γ + U = 195 :=
by
  sorry

end three_digit_problem_l646_646691


namespace avg_of_k_with_positive_integer_roots_l646_646326

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l646_646326


namespace f_monotonicity_g_inequality_mean_value_inequality_l646_646277

-- Definitions
def f (x : ℝ) : ℝ := x * Real.log x - x
def g (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + 1

-- Problem 1: Monotonicity of f(x)
theorem f_monotonicity :
  (∀ x ∈ set.Ioo 0 1, deriv f x < 0) ∧ (∀ x ∈ set.Ioi 1, deriv f x > 0) :=
sorry

-- Problem 2: Range of a for g(x) ≤ 0
theorem g_inequality (a : ℝ) :
  (∀ x > 0, g x a ≤ 0) → a ≥ 1 :=
sorry

-- Problem 3: Given 0 < m < x < n, inequality involving derivatives of f(x)
theorem mean_value_inequality (m x n : ℝ) (h₁ : 0 < m) (h₂ : m < x) (h₃ : x < n) :
  (f x - f m) / (x - m) < (f x - f n) / (x - n) :=
sorry

end f_monotonicity_g_inequality_mean_value_inequality_l646_646277


namespace problem_xy_minimized_problem_x_y_minimized_l646_646713

open Real

theorem problem_xy_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 16 ∧ y = 2 ∧ x * y = 32 := 
sorry

theorem problem_x_y_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 8 + 2 * sqrt 2 ∧ y = 1 + sqrt 2 ∧ x + y = 9 + 4 * sqrt 2 := 
sorry

end problem_xy_minimized_problem_x_y_minimized_l646_646713


namespace minimum_questions_l646_646855

-- Define conditions
def isCheckerboardPattern (grid : Array (Array Bool)) : Prop :=
  ∀ i j, grid[i][j] = (i % 2 = j % 2)

-- Define the problem
theorem minimum_questions {grid : Array (Array Bool)} :
  grid.size = 8 →
  (∀ row, grid[row].size = 8) →
  isCheckerboardPattern grid →
  ∃ (minimum_number_of_questions : ℕ), minimum_number_of_questions = 2 :=
by
  intros
  let minimum_number_of_questions : ℕ := 2
  use minimum_number_of_questions
  -- Proof here
  sorry

end minimum_questions_l646_646855


namespace closest_vertex_to_origin_is_9_9_l646_646485

-- Given conditions
def center_EFGH := (5, 5 : ℝ × ℝ)
def area_EFGH := 16
def scale_factor := 3
def dilation_center := (0, 0 : ℝ × ℝ)

-- The vertices of the square taking into account the given conditions
def vertices_EFGH := 
  let s := real.sqrt area_EFGH in
  let (cx, cy) := center_EFGH in
  [(cx - s/2, cy - s/2), (cx + s/2, cy - s/2),
   (cx + s/2, cy + s/2), (cx - s/2, cy + s/2)]

def dilate (p: ℝ × ℝ) (c: ℝ × ℝ) (f: ℝ) :=
  ((p.1 - c.1) * f + c.1, (p.2 - c.2) * f + c.2)

def dilated_vertices :=
  vertices_EFGH.map (λ v, dilate v dilation_center scale_factor)

-- Theorem to prove
theorem closest_vertex_to_origin_is_9_9 : 
  (0, 0 : ℝ × ℝ).dist (9, 9) ≤ 
  (0, 0).dist (21, 9) ∧ (0, 0).dist (9, 9) ≤ 
  (0, 0).dist (21, 21) ∧ (0, 0).dist (9, 9) ≤ 
  (0, 0).dist (9, 21) := by sorry

end closest_vertex_to_origin_is_9_9_l646_646485


namespace sector_area_l646_646640

-- Definitions of the given conditions
def theta : ℝ := 135
def arc_length : ℝ := 3 * Real.pi

-- The proof statement
theorem sector_area (theta angle : ℝ) (arc_length length : ℝ) : 
  angle = 135 → length = 3 * Real.pi →
  (∃ R : ℝ, arc_length = (theta / 360) * 2 * Real.pi * R ∧ ∃ area : ℝ, area = (theta / 360) * Real.pi * R^2 ∧ area = 6 * Real.pi) :=
by
  intros h_angle h_length
  use (4 : ℝ)   -- Radius, proven via the provided solution steps.
  split
  · rw [h_length, h_angle]
    norm_num,
  · use (6 * Real.pi)
    split
    · norm_num,
    · refl
  sorry

end sector_area_l646_646640


namespace find_number_l646_646997

theorem find_number :
  let sum := 2123 + 1787,
      difference := 2123 - 1787,
      quotient := 6 * difference,
      remainder := 384 in
  (number : ℕ) =
  (number = (sum * quotient) + remainder) →
  number = 7884144 :=
by intros; sorry

end find_number_l646_646997


namespace find_hyperbola_equation_l646_646359

noncomputable def hyperbolaEquation (a b : ℝ) (P : ℝ × ℝ) (r : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ P = (3, 5 / 2) ∧ (r = 1) ∧ (9 / a^2 - 25 / (4 * b^2) = 1)

theorem find_hyperbola_equation :
  ∃ (a b : ℝ), hyperbolaEquation a b (3, 5 / 2) 1 ∧ (a = 2) ∧ (b = Real.sqrt 5) :=
begin
  sorry
end

end find_hyperbola_equation_l646_646359


namespace sqrt_sum_squares_l646_646476

-- Define the conditions
variables (a b : ℝ)
hypothesis h1 : (a + b) / 2 = 20
hypothesis h2 : Real.sqrt (a * b) = 16

-- Prove that a^2 + b^2 = 1088
theorem sqrt_sum_squares :
  a^2 + b^2 = 1088 :=
by
  -- The actual proof steps are not needed, so we use sorry here
  sorry

end sqrt_sum_squares_l646_646476


namespace positive_difference_abs_eq_15_l646_646950

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l646_646950


namespace one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646922

theorem one_fiftieth_digit_of_five_over_thirteen_is_five :
  (decimal_fraction 5 13).digit 150 = 5 :=
by sorry

end one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646922


namespace digital_earth_technologies_l646_646511

-- Define conditions as hypotheses
constant SustainableDevelopment : Prop
constant GlobalPositioningTechnology : Prop
constant GeographicInformationSystem : Prop
constant GlobalPositioningSystem : Prop
constant VirtualTechnology : Prop
constant NetworkTechnology : Prop

-- Define the question about supporting technologies for digital Earth
def supporting_technologies_digital_earth (SD GPT GIS GPS VT NT : Prop) :=
  SD ∧ GPT ∧ GIS ∧ GPS ∧ VT ∧ NT

-- Proof statement (with correct answer)
theorem digital_earth_technologies : 
  supporting_technologies_digital_earth SustainableDevelopment GlobalPositioningTechnology GeographicInformationSystem GlobalPositioningSystem VirtualTechnology NetworkTechnology :=
begin
  -- Hypotheses about the components of the digital Earth
  have h1 : SustainableDevelopment := sorry,
  have h2 : GlobalPositioningTechnology := sorry,
  have h3 : GeographicInformationSystem := sorry,
  have h4 : GlobalPositioningSystem := sorry,
  have h5 : VirtualTechnology := sorry,
  have h6 : NetworkTechnology := sorry,
  
  -- Conclude the proof using the above hypotheses
  exact ⟨h1, h2, h3, h4, h5, h6⟩,
end

end digital_earth_technologies_l646_646511


namespace quadruple_equation_solution_count_l646_646731

theorem quadruple_equation_solution_count (
    a b c d : ℕ
) (h_pos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_order: a < b ∧ b < c ∧ c < d) 
  (h_equation: 2 * a + 2 * b + 2 * c + 2 * d = d^2 - c^2 + b^2 - a^2) : 
  num_correct_statements = 2 :=
sorry

end quadruple_equation_solution_count_l646_646731


namespace lines_parallel_l646_646897

theorem lines_parallel :
  ∀ (x y : ℝ), (x - y + 2 = 0) ∧ (x - y + 1 = 0) → False :=
by
  intros x y h
  sorry

end lines_parallel_l646_646897


namespace random_events_l646_646192

-- Define what it means for an event to be random
def is_random_event (e : Prop) : Prop := ∃ (h : Prop), e ∨ ¬e

-- Define the events based on the problem statements
def event1 := ∃ (good_cups : ℕ), good_cups = 3
def event2 := ∃ (half_hit_targets : ℕ), half_hit_targets = 50
def event3 := ∃ (correct_digit : ℕ), correct_digit = 1
def event4 := true -- Opposite charges attract each other, which is always true
def event5 := ∃ (first_prize : ℕ), first_prize = 1

-- State the problem as a theorem
theorem random_events :
  is_random_event event1 ∧ is_random_event event2 ∧ is_random_event event3 ∧ is_random_event event5 :=
by
  sorry

end random_events_l646_646192


namespace gcf_90_135_225_l646_646109

-- Define the three integers involved
def a : ℕ := 90
def b : ℕ := 135
def c : ℕ := 225

-- Define the GCD function
def gcd (m n : ℕ) : ℕ := Nat.gcd m n

-- Compute the greatest common factor of 90, 135, and 225
def greatest_common_factor (x y z : ℕ) : ℕ := gcd z (gcd x y)

-- The theorem stating that the GCF of 90, 135, and 225 is 45
theorem gcf_90_135_225 : greatest_common_factor a b c = 45 := by
  sorry

end gcf_90_135_225_l646_646109


namespace coins_remainder_divide_by_nine_remainder_l646_646557

def smallest_n (n : ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_remainder (n : ℕ) (h : smallest_n n) : (∃ m : ℕ, n = 54) :=
  sorry

theorem divide_by_nine_remainder (n : ℕ) (h : smallest_n n) (h_smallest: coins_remainder n h) : n % 9 = 0 :=
  sorry

end coins_remainder_divide_by_nine_remainder_l646_646557


namespace friends_number_options_l646_646590

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l646_646590


namespace D_l646_646914

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.fst, -p.snd)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.fst, p.snd)

def D_initial := (4, -3) : ℝ × ℝ

def D_after_transformations :=
  reflect_over_y_axis (reflect_over_x_axis (rotate_90_clockwise D_initial))

-- The statement to prove
theorem D'_after_transform : D_after_transformations = (3, 4) :=
  sorry

end D_l646_646914


namespace count_integers_in_sqrt_sequence_l646_646254

theorem count_integers_in_sqrt_sequence : 
  let f : ℕ → ℝ := λ n, (2401: ℝ) ^ (1 / n)
  in (count (λ n, int.cast (f n) = f n) (range (100))) = 2 :=
by
  sorry

end count_integers_in_sqrt_sequence_l646_646254


namespace square_pattern_58_l646_646394

theorem square_pattern_58 :
  58^2 = 56 * 60 + 4 :=
by
  calc 58^2 = 3364 : by norm_num
         ... = 56 * 60 + 4 : by norm_num

end square_pattern_58_l646_646394


namespace max_area_triangle_eq_max_area_quadrilateral_eq_l646_646145

/-- 
  Prove that for any triangle with a given perimeter P, 
  an equilateral triangle with perimeter P has the largest area.
--/
theorem max_area_triangle_eq (P : ℝ) (hP : P > 0) :
  ∀ {a b c : ℝ}, a + b + c = P → a > 0 → b > 0 → c > 0 → (area_of_triangle a b c) ≤ (area_of_equilateral_triangle P) :=
sorry

/-- 
  Prove that for any quadrilateral with a given perimeter P, 
  a square with perimeter P has the largest area.
--/
theorem max_area_quadrilateral_eq (P : ℝ) (hP : P > 0) :
  ∀ {a b c d : ℝ}, a + b + c + d = P → a > 0 → b > 0 → c > 0 → d > 0 → (area_of_quadrilateral a b c d) ≤ (area_of_square P) :=
sorry

end max_area_triangle_eq_max_area_quadrilateral_eq_l646_646145


namespace group_friends_opponents_l646_646618

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l646_646618


namespace solution_set_of_f_double_exp_inequality_l646_646388

theorem solution_set_of_f_double_exp_inequality (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ 0 < f x) :
  {x : ℝ | f (2^x) < 0} = {x : ℝ | x > 0} :=
sorry

end solution_set_of_f_double_exp_inequality_l646_646388


namespace positive_difference_eq_30_l646_646958

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l646_646958


namespace green_toads_per_acre_l646_646056

-- Conditions definitions
def brown_toads_per_green_toad : ℕ := 25
def spotted_fraction : ℝ := 1 / 4
def spotted_brown_per_acre : ℕ := 50

-- Theorem statement to prove the main question
theorem green_toads_per_acre :
  (brown_toads_per_green_toad * spotted_brown_per_acre * spotted_fraction).to_nat / brown_toads_per_green_toad = 8 := 
sorry

end green_toads_per_acre_l646_646056


namespace lines_concurrent_l646_646416

open EuclideanGeometry

-- Define the cyclic quadrilateral and the necessary points and lines
variables {A B C D E F G H I O P : Point}
variable [Incircle O A B C D]
variable [Intersection (Line A B) (Line D C) E]
variable [Intersection (Line A D) (Line B C) F]
variable [Circumcircle (Triangle E F C) P]
variable [Intersection (Circumcircle P) (Incircle O) G]
variable [Intersection (Line A G) (Line E F) H]
variable [Intersection (Line H C) (Incircle O) I]

-- Main theorem to prove: AI, GC, and FE are concurrent
theorem lines_concurrent 
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : PointIntersection (line A B) (line D C) E)
  (h3 : PointIntersection (line A D) (line B C) F)
  (h4 : Circumcircle E F C P)
  (h5 : CircleIntersection P O G)
  (h6 : PointIntersection (line A G) (line E F) H)
  (h7 : PointIntersection (line H C) O I) :
  Concurrent (line A I) (line G C) (line F E) :=
by sorry

end lines_concurrent_l646_646416


namespace positive_difference_abs_eq_15_l646_646951

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l646_646951


namespace smallest_number_divisible_conditions_l646_646560

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l646_646560


namespace distance_from_origin_to_line_l646_646153

theorem distance_from_origin_to_line 
  (M : ℝ × ℝ) (N : ℝ × ℝ) 
  (on_hyperbola : 2 * M.1 ^ 2 - M.2 ^ 2 = 1) 
  (on_ellipse : 4 * N.1 ^ 2 + N.2 ^ 2 = 1) 
  (perpendicular : M.1 * N.1 + M.2 * N.2 = 0) :
  let d := abs (M.1 * N.1 + M.2 * N.2) / (sqrt (M.1 ^ 2 + M.2 ^ 2) * sqrt (N.1 ^ 2 + N.2 ^ 2))
  in d = sqrt(3) / 3 :=
by sorry

end distance_from_origin_to_line_l646_646153


namespace sum_first_40_terms_of_a_eq_1240_l646_646361

noncomputable def a : ℕ → ℝ
| 0       := 0 -- This will not be used as we start from natural numbers
| (n + 1) := if (n + 1) % 2 = 0 then 
               a n + (3 * (n) - 1) 
             else 
               3 * n - 1 - a n

theorem sum_first_40_terms_of_a_eq_1240 : 
  (finset.range 40).sum a = 1240 := 
sorry

end sum_first_40_terms_of_a_eq_1240_l646_646361


namespace age_difference_l646_646393

theorem age_difference (A B : ℕ) (h1 : B = 37) (h2 : A + 10 = 2 * (B - 10)) : A - B = 7 :=
by
  sorry

end age_difference_l646_646393


namespace three_digit_problem_l646_646690

theorem three_digit_problem :
  ∃ (M Γ U : ℕ), 
    M ≠ Γ ∧ M ≠ U ∧ Γ ≠ U ∧
    M ≤ 9 ∧ Γ ≤ 9 ∧ U ≤ 9 ∧
    100 * M + 10 * Γ + U = (M + Γ + U) * (M + Γ + U - 2) ∧
    100 * M + 10 * Γ + U = 195 :=
by
  sorry

end three_digit_problem_l646_646690


namespace number_of_friends_l646_646611

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l646_646611


namespace find_work_days_q_l646_646150

-- Required definitions based on conditions in the problem:
def work_rate_p : ℝ := 1 / 15
def work_days_q : ℝ  -- the number of days for the second person to do the work alone, which we want to find.

def work_rate_q (d : ℝ) : ℝ := 1 / d
def days_worked_together : ℝ := 4
def fraction_work_left : ℝ := 0.5333333333333333
def fraction_work_completed : ℝ := 1 - fraction_work_left

noncomputable def equation (d : ℝ) : Prop := 
  4 * (work_rate_p + work_rate_q d) = fraction_work_completed

theorem find_work_days_q : equation 20 := 
by 
  intro d,
  sorry  -- The proof is to be constructed.

end find_work_days_q_l646_646150


namespace smallest_and_largest_group_sizes_l646_646159

theorem smallest_and_largest_group_sizes (S T : Finset ℕ) (hS : S.card + T.card = 20)
  (h_union: (S ∪ T) = (Finset.range 21) \ {0}) (h_inter: S ∩ T = ∅)
  (sum_S : S.sum id = 210 - T.sum id) (prod_T : T.prod id = 210 - S.sum id) :
  T.card = 3 ∨ T.card = 5 := 
sorry

end smallest_and_largest_group_sizes_l646_646159


namespace max_constant_term_l646_646530

theorem max_constant_term (c : ℝ) : 
  (∀ x : ℝ, (x^2 - 6 * x + c = 0 → (x^2 - 6 * x + c ≥ 0))) → c ≤ 9 :=
by sorry

end max_constant_term_l646_646530


namespace cover_square_with_rectangles_l646_646137

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l646_646137


namespace vector_inequality_abc_l646_646007

variables (a b c : ℝ^3)

theorem vector_inequality_abc (a b c : ℝ^3) :
  ∥a∥ + ∥b∥ + ∥c∥ + ∥a + b + c∥ ≥ ∥a + b∥ + ∥b + c∥ + ∥c + a∥ :=
by sorry

end vector_inequality_abc_l646_646007


namespace minimum_cost_for_28_apples_l646_646401

/--
Conditions:
  - apples can be bought at a rate of 4 for 15 cents,
  - apples can be bought at a rate of 7 for 30 cents,
  - you need to buy exactly 28 apples.
Prove that the minimum total cost to buy exactly 28 apples is 120 cents.
-/
theorem minimum_cost_for_28_apples : 
  let cost_4_for_15 := 15
  let cost_7_for_30 := 30
  let apples_needed := 28
  ∃ (n m : ℕ), n * 4 + m * 7 = apples_needed ∧ n * cost_4_for_15 + m * cost_7_for_30 = 120 := sorry

end minimum_cost_for_28_apples_l646_646401


namespace find_k_l646_646164

def total_balls (k : ℕ) : ℕ := 7 + k

def probability_green (k : ℕ) : ℚ := 7 / (total_balls k)
def probability_purple (k : ℕ) : ℚ := k / (total_balls k)

def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * 3 + (probability_purple k) * (-1)

theorem find_k (k : ℕ) (h_pos : k > 0) (h_exp_value : expected_value k = 1) : k = 7 :=
sorry

end find_k_l646_646164


namespace possible_number_of_friends_l646_646602

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l646_646602


namespace books_problem_l646_646908

theorem books_problem 
  (chinese_books : ℕ)
  (math_books : ℕ)
  (english_books : ℕ) :
  chinese_books = 10 →
  math_books = 9 →
  english_books = 8 →
  (chinese_books * math_books + chinese_books * english_books + english_books * math_books) = 242 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact sorry

end books_problem_l646_646908


namespace avg_k_for_polynomial_roots_l646_646294

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l646_646294


namespace projection_reflection_matrix_l646_646241

theorem projection_reflection_matrix :
  let v := (⟨3, 4⟩ : ℝ × ℝ)
  let mat : matrix (fin 2) (fin 2) ℝ := ![![(-7 : ℝ) / 25, (24 : ℝ) / 25], ![(24 : ℝ) / 25, 7 / 25]]
  ∀ (x y : ℝ),
  let proj := ((x, y) : ℝ × ℝ)
  let proj_v := ((proj.1 * v.1 + proj.2 * v.2) / (v.1^2 + v.2^2)) • v
  let reflect := 2 • proj_v - proj
  vector.of_tuple reflect = ![
    ![(proj.1 * (v.1 * v.1 / (v.1^2 + v.2^2)) - proj.1), 
      (proj.2 * (v.1 * v.2 / (v.1^2 + v.2^2)) - proj.2)],
    ![(proj.1 * (v.2 * v.1 / (v.1^2 + v.2^2)) - proj.1), 
      (proj.2 * (v.2 * v.2 / (v.1^2 + v.2^2)) - proj.2)]
  ] :=
  sorry

end projection_reflection_matrix_l646_646241


namespace euclidean_remainder_2022_l646_646111

theorem euclidean_remainder_2022 : 
  (2022 ^ (2022 ^ 2022)) % 11 = 5 := 
by sorry

end euclidean_remainder_2022_l646_646111


namespace functional_eq_linear_l646_646232

theorem functional_eq_linear {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x + y) * (f x - f y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end functional_eq_linear_l646_646232


namespace sum_is_seventeen_l646_646019

variable (x y : ℕ)

def conditions (x y : ℕ) : Prop :=
  x > y ∧ x - y = 3 ∧ x * y = 56

theorem sum_is_seventeen (x y : ℕ) (h: conditions x y) : x + y = 17 :=
by
  sorry

end sum_is_seventeen_l646_646019


namespace find_digits_l646_646793

theorem find_digits (x y z : ℕ) (hx : x ≤ 9) (hy : y ≤ 9) (hz : z ≤ 9)
    (h_eq : (10*x+5) * (300 + 10*y + z) = 7850) : x = 2 ∧ y = 1 ∧ z = 4 :=
by {
  sorry
}

end find_digits_l646_646793


namespace janet_needs_9_dog_collars_l646_646803

variable (D : ℕ)

theorem janet_needs_9_dog_collars (h1 : ∀ d : ℕ, d = 18)
  (h2 : ∀ c : ℕ, c = 10)
  (h3 : (18 * D) + (3 * 10) = 192) :
  D = 9 :=
by
  sorry

end janet_needs_9_dog_collars_l646_646803


namespace number_of_friends_l646_646612

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l646_646612


namespace sum_of_squares_binomial_l646_646728

open_locale big_operators

theorem sum_of_squares_binomial (n : ℕ) (x : ℝ) (hn : 0 < n) :
  ∑ k in finset.range (n + 1), k^2 * (nat.choose n k) * x^k * (1 - x)^(n - k) = n * (n - 1) * x^2 + n * x :=
sorry

end sum_of_squares_binomial_l646_646728


namespace avg_of_k_with_positive_integer_roots_l646_646322

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l646_646322


namespace tangent_line_slope_range_l646_646504

noncomputable def f (x : ℝ) : ℝ := x^3 - real.sqrt 3 * x + 2

theorem tangent_line_slope_range :
  ∀ (x : ℝ), (3 * x^2 - real.sqrt 3) ∈ set.Ici (-real.sqrt 3) :=
by
  sorry

end tangent_line_slope_range_l646_646504


namespace complex_number_identity_l646_646878

open Complex

theorem complex_number_identity : (2 - I : ℂ) / (1 + 2 * I) = -I := 
by 
  sorry

end complex_number_identity_l646_646878


namespace avg_of_k_with_positive_integer_roots_l646_646327

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l646_646327


namespace find_matrix_l646_646701

open Matrix

def mat (α : Type*) [Semiring α] : Type* := Matrix (Fin 2) (Fin 2) α
def vec (α : Type*) [Semiring α] : Type* := Fin 2 → α

theorem find_matrix (M : mat ℝ) : 
  (∀ v : vec ℝ, M.mul_vec v = 3 • v) ↔ M = λ i j, if i = j then 3 else 0 :=
by {
  sorry
}

end find_matrix_l646_646701


namespace player_a_all_opportunities_player_a_probability_distribution_math_expectation_l646_646049

-- Define success probability and failure probability
def success_probability : ℝ := 3 / 5
def failure_probability := 1 - success_probability

-- Define the possible scores and their probabilities
def pmf_x : Pmf ℝ :=
  Pmf.ofFinset {0, 50, 100, 150} (λ x,
    if x = 0 then failure_probability^2 
    else if x = 50 then success_probability * (failure_probability^2) + failure_probability * success_probability * failure_probability
    else if x = 100 then (3.choose 2) * (success_probability^2) * failure_probability
    else success_probability^3)

-- Lean statement to prove both conditions
theorem player_a_all_opportunities :
  (3:ℝ) * (1 - failure_probability^2) =  3 * (21/25) := by
  sorry

theorem player_a_probability_distribution_math_expectation:
  pmf_x.prob 0 = 4/25 ∧ pmf_x.prob 50 = 24/125 ∧ pmf_x.prob 100 = 54/125 ∧ pmf_x.prob 150 = 27/125
  ∧ pmf_x.exp = 85.2 := by
  sorry

end player_a_all_opportunities_player_a_probability_distribution_math_expectation_l646_646049


namespace parallel_lines_k_value_l646_646894

theorem parallel_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (4-k) * y + 1 = 0 → 2 * (k-3) * x - 2 * y + 3 = 0 →
   ((k = 3) ∨ (k = 5))) := 
begin
  sorry
end

end parallel_lines_k_value_l646_646894


namespace part1_part2_l646_646840

-- Define the set P for a given a
def P (a : ℝ) : set ℝ := { x : ℝ | (x - a) * (x + 1) ≤ 0 }

-- Define the set Q
def Q : set ℝ := { x : ℝ | |x - 1| ≤ 1 }

-- Problem (1)
theorem part1 : P 3 = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

-- Problem (2)
theorem part2 (a : ℝ) (Q_subset_P : Q ⊆ P a) : 2 ≤ a :=
sorry

end part1_part2_l646_646840


namespace rectangles_count_l646_646413

theorem rectangles_count (n : ℕ) (h : n = 6) : (nat.choose n 2) * (nat.choose n 2) = 225 :=
by
  have hn : nat.choose n 2 = 15 := by sorry
  rw [hn]
  norm_num

end rectangles_count_l646_646413


namespace perimeter_of_table_l646_646162

/-- Define Initial Setup and Conditions -/
variables {P : Type} 
variables (table : Type) [Group table] (edge : table → table → ℝ) (distance_traveled : ℝ)

/-- Define the distance and angles -/
def distance_7_meters (P : table) (velocity_angle : ℝ) := 
  velocity_angle = π / 4 ∧ distance_traveled = 7

/-- Define what we need to prove: Perimeter is closest to 7.5 meters -/
theorem perimeter_of_table 
  (h : ∀ Q : table, distance_7_meters P (π / 4)) 
  (total_distance : ℝ)
  (perimeter : ℝ) : 
  total_distance = 7 → perimeter ≈ 7.5 :=
sorry

end perimeter_of_table_l646_646162


namespace number_of_friends_l646_646617

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l646_646617


namespace maximize_revenue_l646_646172

variables (TotalRooms : ℕ) (InitialRate : ℕ) (InitialOccupancyRate : ℚ)
variables (ReductionStep : ℕ) (AdditionalRooms : ℕ)

-- Define initial conditions
def initialConditions := (TotalRooms = 100) ∧ (InitialRate = 400) ∧ 
                         (InitialOccupancyRate = 0.5) ∧ (ReductionStep = 20) ∧ 
                         (AdditionalRooms = 5)

-- Define the function for revenue 
def revenue (rate : ℕ) (occupiedRooms : ℕ) : ℚ := rate * occupiedRooms

-- Define the function for number of occupied rooms given a rate decrease
def occupiedRooms (rateDecreaseStep : ℕ) : ℕ :=
  50 + (rateDecreaseStep / 20) * 5

-- Define the function to get the optimal rate
def optimalRate : ℕ := 300

-- The main theorem we need to prove
theorem maximize_revenue : initialConditions → (optimalRate = 300) :=
begin
  sorry
end

end maximize_revenue_l646_646172


namespace solution_l646_646984

noncomputable def sqrt_system 
  (x y : ℝ) 
  (hx : x = 1232 + 1/2) 
  (hy : y = 1232 + 1/2) 
  : Prop :=
  sqrt (1008 + 1/4 + x) + sqrt (1008 + 1/4 + y) = 114 ∧ 
  sqrt (1008 + 1/4 - x) + sqrt (1008 + 1/4 - y) = 56

theorem solution 
  (h : sqrt_system 1232.5 1232.5 (by norm_num) (by norm_num)) :
  1232.5 = 1232 + 1/2 ∧ 1232.5 = 1232 + 1/2 :=
by 
  sorry

end solution_l646_646984


namespace Oliver_Battle_Gremlins_Card_Count_l646_646021

theorem Oliver_Battle_Gremlins_Card_Count 
  (MonsterClubCards AlienBaseballCards BattleGremlinsCards : ℕ)
  (h1 : MonsterClubCards = 2 * AlienBaseballCards)
  (h2 : BattleGremlinsCards = 3 * AlienBaseballCards)
  (h3 : MonsterClubCards = 32) : 
  BattleGremlinsCards = 48 := by
  sorry

end Oliver_Battle_Gremlins_Card_Count_l646_646021


namespace original_height_in_feet_l646_646809

-- Define the current height in inches
def current_height_in_inches : ℚ := 180

-- Define the percentage increase in height
def percentage_increase : ℚ := 0.5

-- Define the conversion factor from inches to feet
def inches_to_feet : ℚ := 12

-- Define the initial height in inches
def initial_height_in_inches : ℚ := current_height_in_inches / (1 + percentage_increase)

-- Prove that the original height in feet was 10 feet
theorem original_height_in_feet : initial_height_in_inches / inches_to_feet = 10 :=
by
  -- Placeholder for the full proof
  sorry

end original_height_in_feet_l646_646809


namespace trajectory_eq_l646_646787

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation for point C given the parameters s and t
def C (s t : ℝ) : ℝ × ℝ := (s * 2 + t * -1, s * 1 + t * -2)

-- Prove the equation of the trajectory of C given s + t = 1
theorem trajectory_eq (s t : ℝ) (h : s + t = 1) : ∃ x y : ℝ, C s t = (x, y) ∧ x - y - 1 = 0 := by
  -- The proof will be added here
  sorry

end trajectory_eq_l646_646787


namespace travel_time_X_to_Y_l646_646190

noncomputable def distanceXY : ℝ := 1000
noncomputable def returnTime : ℝ := 4
noncomputable def averageSpeed : ℝ := 142.85714285714286

theorem travel_time_X_to_Y :
  let totalRoundTripDistance := 2 * distanceXY,
      totalRoundTripTime := totalRoundTripDistance / averageSpeed in
  totalRoundTripTime - returnTime = 10 := 
by
  sorry

end travel_time_X_to_Y_l646_646190


namespace measure_of_angle_B_and_side_b_l646_646390

noncomputable def triangleABC (a b c A B C : ℝ) :=
  ∃ R : ℝ, a = 2 * R * real.sin A ∧ b = 2 * R * real.sin B ∧ c = 2 * R * real.sin C ∧
           a + c = 2 ∧ 
           (real.sin B * real.cos C = 2 * real.sin A * real.cos B + real.cos B * real.sin C)

theorem measure_of_angle_B_and_side_b (a b c A B C : ℝ) 
  (h₁ : triangleABC a b c A B C) 
  (h₂ : S = (sqrt 3) / 4) :
  B = 2 / 3 * real.pi ∧ b = sqrt 3 :=
begin
  sorry
end

end measure_of_angle_B_and_side_b_l646_646390


namespace find_sister_candy_l646_646433

/-- Define Katie's initial amount of candy -/
def Katie_candy : ℕ := 10

/-- Define the amount of candy eaten the first night -/
def eaten_candy : ℕ := 9

/-- Define the amount of candy left after the first night -/
def remaining_candy : ℕ := 7

/-- Define the number of candies Katie's sister had -/
def sister_candy (S : ℕ) : Prop :=
  Katie_candy + S - eaten_candy = remaining_candy

/-- Theorem stating that Katie's sister had 6 pieces of candy -/
theorem find_sister_candy : ∃ S, sister_candy S ∧ S = 6 :=
by
  sorry

end find_sister_candy_l646_646433


namespace third_term_expansion_l646_646220

theorem third_term_expansion :
  let f := (1 - x) * (1 + 2 * x) ^ 5 in
  (f.expand(2, x)) = 30 * x^2 :=
by
  sorry

end third_term_expansion_l646_646220


namespace p_q_relation_n_le_2_p_q_relation_n_ge_3_l646_646820

open Real -- for ℝ
open Nat -- for ℕ

definition P (n : ℕ) (x : ℝ) : ℝ := (1-x)^(2*n-1)
definition Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n-1) * x + (n-1) * (2*n-1) * x^2

theorem p_q_relation_n_le_2 (n : ℕ+) (x : ℝ)
  (h_n_le_2 : n.val <= 2) : 
  if n.val = 1 then P n x = Q n x 
  else if x = 0 then P n x = Q n x
  else if x > 0 then P n x < Q n x
  else P n x > Q n x :=
sorry

theorem p_q_relation_n_ge_3 (n : ℕ+) (x : ℝ)
  (h_n_ge_3 : n.val >= 3) :
  if x = 0 then P n x = Q n x
  else if x > 0 then P n x < Q n x
  else P n x > Q n x :=
sorry

end p_q_relation_n_le_2_p_q_relation_n_ge_3_l646_646820


namespace angle_between_vectors_l646_646755

variable {a b : EuclideanSpace ℝ (Fin 3)}

theorem angle_between_vectors (ha : ‖a‖ = Real.sqrt 2) (hb : ‖b‖ = 2) (h_perp : (a + b) ∙ a = 0) :
  real.angle a b = (3 * Real.pi) / 4 :=
sorry

end angle_between_vectors_l646_646755


namespace sufficient_but_not_necessary_condition_l646_646768

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  ({-1, a^2} ∩ {2, 4} = {4} → (a = -2 → {x | (x = -1) ∨ (x = 4)} ∩ {2, 4} = {4})) ∧
  ({-1, a^2} ∩ {2, 4} = {4} → ∃ b : ℝ, b ≠ -2 ∧ {-1, b^2} ∩ {2, 4} = {4}) := 
sorry

end sufficient_but_not_necessary_condition_l646_646768


namespace difference_between_smallest_integers_divisible_by_2_to_13_with_remainder_1_l646_646098

theorem difference_between_smallest_integers_divisible_by_2_to_13_with_remainder_1 : 
  let lcm_2_to_13 := Nat.lcm_list [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] in
  lcm_2_to_13 = 360360 := 
by 
  let lcm_2_to_13 := Nat.lcm_list [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  show lcm_2_to_13 = 360360 from sorry

end difference_between_smallest_integers_divisible_by_2_to_13_with_remainder_1_l646_646098


namespace v_function_expression_f_max_value_l646_646491

noncomputable def v (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2
else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2)
else 0

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2 * x
else if 4 < x ∧ x ≤ 20 then - (1/8) * x^2 + (5/2) * x
else 0

theorem v_function_expression :
  ∀ x, 0 < x ∧ x ≤ 20 → 
  v x = (if 0 < x ∧ x ≤ 4 then 2 else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2) else 0) :=
by sorry

theorem f_max_value :
  ∃ x, 0 < x ∧ x ≤ 20 ∧ f x = 12.5 :=
by sorry

end v_function_expression_f_max_value_l646_646491


namespace chord_bisect_angle_l646_646397

theorem chord_bisect_angle (AB AC : ℝ) (angle_CAB : ℝ) (h1 : AB = 2) (h2 : AC = 1) (h3 : angle_CAB = 120) : 
  ∃ x : ℝ, x = 3 := 
by
  -- Proof goes here
  sorry

end chord_bisect_angle_l646_646397


namespace average_of_distinct_k_l646_646311

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l646_646311


namespace geometric_sequence_common_ratio_l646_646580

theorem geometric_sequence_common_ratio :
  ∃ r : ℝ, (r = -2) ∧ (r * 25 = -50) ∧ (r * -50 = 100) ∧ (r * 100 = -200) :=
by {
  use -2,
  split,
  exact rfl,
  split,
  norm_num,
  split,
  norm_num,
  norm_num,
}

end geometric_sequence_common_ratio_l646_646580


namespace friends_number_options_l646_646593

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l646_646593


namespace remaining_apples_l646_646214

def initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem remaining_apples : initial_apples - shared_apples = 13 :=
by
  sorry

end remaining_apples_l646_646214


namespace number_of_different_tower_heights_l646_646023

structure Brick :=
  (dim1 : ℕ)
  (dim2 : ℕ)
  (dim3 : ℕ)

def bricks : List Brick := List.replicate 100 { dim1 := 3, dim2 := 11, dim3 := 20 }

def orientations (b : Brick) : List ℕ := [b.dim1, b.dim2, b.dim3]

theorem number_of_different_tower_heights :
  (∃ num_heights : ℕ, achievable_heights bricks orientations = num_heights ∧ num_heights = 1606) :=
sorry

end number_of_different_tower_heights_l646_646023


namespace point_in_first_quadrant_l646_646058

def complex_in_first_quadrant (z : ℂ) := z.re > 0 ∧ z.im > 0

theorem point_in_first_quadrant (z : ℂ) (h : z / complex.I = 2 - 3 * complex.I) : 
  complex_in_first_quadrant z :=
by
  sorry

end point_in_first_quadrant_l646_646058


namespace employed_male_percent_problem_l646_646417

noncomputable def employed_percent_population (total_population_employed_percent : ℝ) (employed_females_percent : ℝ) : ℝ :=
  let employed_males_percent := (1 - employed_females_percent) * total_population_employed_percent
  employed_males_percent

theorem employed_male_percent_problem :
  employed_percent_population 0.72 0.50 = 0.36 := by
  sorry

end employed_male_percent_problem_l646_646417


namespace part_a_part_b_l646_646975

-- Definitions for part a)
def heptagon : Type := {A B C D E F G : Type}
def triangles_from_heptagon (H : heptagon) : list (set Type) :=
[ {H.1, H.2, H.3}, {H.2, H.3, H.4}, {H.3, H.4, H.5}, {H.4, H.5, H.6}, {H.5, H.6, H.7}, {H.6, H.7, H.1}, {H.7, H.1, H.2} ]

theorem part_a (H : heptagon) : (∀ (S : finset (set Type)), S.card = 6 → ∃ (P Q : Type), ∀ t ∈ S, P ∈ t ∧ Q ∈ t) ∧ 
  ¬ (∃ (P Q : Type), ∀ t ∈ triangles_from_heptagon H, P ∈ t ∧ Q ∈ t ) := sorry

-- Definitions for part b)
def quadrilaterals_and_central_heptagon (H : heptagon) : list (set Type) :=
[ {H.1, H.2, H.3, H.4}, {H.2, H.3, H.4, H.5}, {H.3, H.4, H.5, H.6}, {H.4, H.5, H.6, H.7}, {H.5, H.6, H.7, H.1}, {H.6, H.7, H.1, H.2}, {H.7, H.1, H.2, H.3}, {H.1, H.2, H.3, H.4, H.5, H.6, H.7} ]

theorem part_b (H : heptagon) : (∀ (S : finset (set Type)), S.card = 7 → ∃ (P Q : Type), ∀ t ∈ S, P ∈ t ∧ Q ∈ t) ∧
  ¬ (∃ (P Q : Type), ∀ t ∈ quadrilaterals_and_central_heptagon H, P ∈ t ∧ Q ∈ t) := sorry

end part_a_part_b_l646_646975


namespace committee_probability_l646_646086

def num_boys : ℕ := 10
def num_girls : ℕ := 15
def num_total : ℕ := 25
def committee_size : ℕ := 5

def num_ways_total : ℕ := Nat.choose num_total committee_size
def num_ways_boys_only : ℕ := Nat.choose num_boys committee_size
def num_ways_girls_only : ℕ := Nat.choose num_girls committee_size

def probability_boys_or_girls_only : ℚ :=
  (num_ways_boys_only + num_ways_girls_only) / num_ways_total

def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - probability_boys_or_girls_only

theorem committee_probability :
  probability_at_least_one_boy_and_one_girl = 475 / 506 :=
sorry

end committee_probability_l646_646086


namespace Frost_Town_snow_probability_l646_646503

noncomputable def binomial_probability : ℕ → ℕ → ℝ → ℝ :=
  λ n k p, (Nat.choose n k) * (p^k) * ((1-p)^(n-k))

theorem Frost_Town_snow_probability :
  let p := (1 : ℝ) / 5
  let prob := (binomial_probability 31 0 p) + 
              (binomial_probability 31 1 p) + 
              (binomial_probability 31 2 p) + 
              (binomial_probability 31 3 p)
  in prob = 0.336 :=
by
  sorry

end Frost_Town_snow_probability_l646_646503


namespace square_area_of_circle_radius_and_conditions_l646_646876

noncomputable def square_area (r : ℝ) : ℝ :=
  let x := sqrt (1 / 5) in
  4 * x^2

theorem square_area_of_circle_radius_and_conditions (r : ℝ) (h1 : r = 1) (h2 : True) (h3 : True) :
  square_area r = 4 / 5 :=
by
  rw [h1, square_area]
  sorry

end square_area_of_circle_radius_and_conditions_l646_646876


namespace friends_number_options_l646_646595

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l646_646595


namespace valid_sequences_l646_646231

/-- A finite sequence (x₀, x₁, ..., xₙ) such that for every j (0 ≤ j ≤ n), x_j equals the number of times j appears in the sequence. -/
def valid_sequence (x : list ℕ) : Prop :=
  ∀ j, x.count j = x.nth j.getOrElse 0

theorem valid_sequences (x : list ℕ) : valid_sequence x ↔
  x = [2, 0, 2, 0] ∨ x = [1, 2, 1, 0] ∨ x = [2, 1, 2, 0, 0] ∨
  ∃ p, x = [p] ++ [2, 1] ++ list.replicate (p - 3) 0 ++ [1, 0, 0, 0] :=
sorry

end valid_sequences_l646_646231


namespace range_of_a_l646_646772

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x - 12| < 6 → False) → (a ≤ 6 ∨ a ≥ 18) :=
by 
  intro h
  sorry

end range_of_a_l646_646772


namespace curve_transformation_equiv_l646_646239

-- Define the initial curve equation
def initial_curve (x y : ℝ) : Prop := 2 * x^2 - 2 * x * y + 1 = 0

-- Define the transformation matrices
def matrix_N : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![-1, 1]]
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]

-- Define the product of matrices M and N
def matrix_MN : Matrix (Fin 2) (Fin 2) ℝ := matrix_M.mul matrix_N

-- Given x'' = x and y'' = 2y, transform the initial equation
def transformed_curve (x'' y'' : ℝ) : Prop := 6 * x''^2 - 2 * x'' * y'' + 1 = 0

-- Theorem stating the equivalence of the initial curve and the transformed curve under given transformations
theorem curve_transformation_equiv :
  ∀ (x y : ℝ),
  initial_curve x y →
  transformed_curve x (2 * y) :=
by
  sorry

end curve_transformation_equiv_l646_646239


namespace positive_difference_abs_eq_l646_646942

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l646_646942


namespace probability_x_y_even_minus_odd_l646_646514

structure Person :=
(age : ℕ)
(height : ℕ)

def boys := [ { Person.mk 4 44 }, { Person.mk 6 48 }, { Person.mk 7 52 } ]
def girls := [ { Person.mk 5 42 }, { Person.mk 8 51 }, { Person.mk 9 55 } ]

noncomputable def probability_difference : ℚ :=
let combinations := (boys.combination 2).product (girls.combination 2) in
let sums := combinations.map (λ c, let ages_sum := c.1.1.age + c.1.2.age + c.2.1.age + c.2.2.age in 
                                   let heights_sum := c.1.1.height + c.1.2.height + c.2.1.height + c.2.2.height in 
                                   ages_sum + heights_sum) in
let even_count := sums.count (λ s, s % 2 = 0) in
let odd_count := sums.count (λ s, s % 2 = 1) in
(odd_count - even_count) / sums.length

theorem probability_x_y_even_minus_odd : 
  probability_difference = 1/9 :=
sorry

end probability_x_y_even_minus_odd_l646_646514


namespace find_multiple_l646_646230

theorem find_multiple (n : ℕ) (h₁ : n = 5) (m : ℕ) (h₂ : 7 * n - 15 > m * n) : m = 3 :=
by
  sorry

end find_multiple_l646_646230


namespace convex_polyhedron_same_number_of_sides_l646_646473

theorem convex_polyhedron_same_number_of_sides {N : ℕ} (hN : N ≥ 4): 
  ∃ (f1 f2 : ℕ), (f1 >= 3 ∧ f1 < N ∧ f2 >= 3 ∧ f2 < N) ∧ f1 = f2 :=
by
  sorry

end convex_polyhedron_same_number_of_sides_l646_646473


namespace hypotenuse_length_l646_646773

-- Translation of conditions to Lean definitions:
def a : ℕ := 8
def b : ℕ := 15
def c : ℕ := 17

-- Lean statement to prove that given the lengths of two right triangle sides, the hypotenuse length is 17.
theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 15) (h₃ : c = 17)
  (h : a^2 + b^2 = c^2) : c = 17 :=
by {
  rw [h₁, h₂, h₃],
  sorry
}

end hypotenuse_length_l646_646773


namespace positive_difference_abs_eq_l646_646943

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l646_646943


namespace avg_k_value_l646_646334

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l646_646334


namespace total_copies_produced_l646_646911

theorem total_copies_produced
  (rate_A : ℕ)
  (rate_B : ℕ)
  (rate_C : ℕ)
  (time_A : ℕ)
  (time_B : ℕ)
  (time_C : ℕ)
  (total_time : ℕ)
  (ha : rate_A = 10)
  (hb : rate_B = 10)
  (hc : rate_C = 10)
  (hA_time : time_A = 15)
  (hB_time : time_B = 20)
  (hC_time : time_C = 25)
  (h_total_time : total_time = 30) :
  rate_A * time_A + rate_B * time_B + rate_C * time_C = 600 :=
by 
  -- Machine A: 10 copies per minute * 15 minutes = 150 copies
  -- Machine B: 10 copies per minute * 20 minutes = 200 copies
  -- Machine C: 10 copies per minute * 25 minutes = 250 copies
  -- Hence, the total number of copies = 150 + 200 + 250 = 600
  sorry

end total_copies_produced_l646_646911


namespace differentiable_function_inequality_l646_646287

variable {f : ℝ → ℝ}

/-- Given a differentiable function f, for all real x, if f(x) < f'(x), 
    then it follows that f(1) > e * f(0) and f(2019) > e^2019 * f(0). -/
theorem differentiable_function_inequality
  (h_differentiable : ∀ x, differentiable_at ℝ f x)
  (h_inequality : ∀ x, f(x) < deriv f x) :
  f(1) > Real.exp 1 * f(0) ∧ f(2019) > Real.exp 2019 * f(0) := 
sorry

end differentiable_function_inequality_l646_646287


namespace intersection_complement_eq_l646_646986

open Set

namespace MathProof

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {3, 4, 5} →
  B = {1, 3, 6} →
  A ∩ (U \ B) = {4, 5} :=
by
  intros hU hA hB
  sorry

end MathProof

end intersection_complement_eq_l646_646986


namespace min_k_l_sum_l646_646489

theorem min_k_l_sum (k l : ℕ) (hk : 120 * k = l^3) (hpos_k : k > 0) (hpos_l : l > 0) :
  k + l = 255 :=
sorry

end min_k_l_sum_l646_646489


namespace skew_lines_from_6_points_l646_646410

theorem skew_lines_from_6_points (points : List Point) (h_points : points.length = 6) : 
    (∀ (p1 p2 p3 p4 : Point), p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → 
                                       p3 ≠ p4 → 
    ¬ AffineSpan ℝ {p1, p2, p3, p4}) → 
    (number_of_pairs_of_skew_lines points = 45) := 
sorry

end skew_lines_from_6_points_l646_646410


namespace alcohol_concentration_l646_646188

theorem alcohol_concentration 
  (x : ℝ) -- concentration of alcohol in the first vessel (as a percentage)
  (h1 : 0 ≤ x ∧ x ≤ 100) -- percentage is between 0 and 100
  (h2 : (x / 100) * 2 + (55 / 100) * 6 = (37 / 100) * 10) -- given condition for concentration balance
  : x = 20 :=
sorry

end alcohol_concentration_l646_646188


namespace avg_k_value_l646_646333

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l646_646333


namespace decrease_in_temperature_l646_646379

theorem decrease_in_temperature (increase_temp : ℤ → Prop) :
  (increase_temp 2 → -3 = -3) :=
by
  intro h
  exact eq.refl (-3)

end decrease_in_temperature_l646_646379


namespace smallest_num_rectangles_to_cover_square_l646_646116

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l646_646116


namespace total_songs_bought_l646_646142

def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7

theorem total_songs_bought :
  (country_albums + pop_albums) * songs_per_album = 70 := by
  sorry

end total_songs_bought_l646_646142


namespace friends_number_options_l646_646594

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l646_646594


namespace possible_values_of_expression_l646_646285

noncomputable def sign (x : ℝ) : ℝ := x / |x|

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (sign a + sign b + sign c + sign d + sign (a * b * c * d)) ∈ ({5, 1, -3, -5} : set ℝ) :=
by {
  -- placeholder for proof
  sorry
}

end possible_values_of_expression_l646_646285


namespace find_c_l646_646676

noncomputable theory

def satisfies_quadratic (c : ℚ) :=
  ∃ (x_floor : ℤ) (x_frac : ℚ), 
    (3 * x_floor ^ 2 + 8 * x_floor - 35 = 0 ∧
     4 * x_frac ^ 2 - 12 * x_frac + 5 = 0 ∧ 
     x_frac = c - x_floor)

theorem find_c : satisfies_quadratic (-9/2) :=
  sorry

end find_c_l646_646676


namespace count_connected_ways_l646_646545

def graph := { V : finset ℕ // V.card = 8 } × { E : finset (ℕ × ℕ) // E.card = 12 }

def choose_5_roads (E : finset (ℕ × ℕ)) := E.subsets 5

def connected_after_removal (G : graph) (edges_to_remove : finset (ℕ × ℕ)) : Prop :=
  -- This is a placeholder for the condition that the graph remains connected
  sorry

theorem count_connected_ways (G : graph) :
  (choose_5_roads G.2).filter (connected_after_removal G) .card = 384 := 
sorry

end count_connected_ways_l646_646545


namespace possible_number_of_friends_l646_646626

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l646_646626


namespace smallest_number_divisible_conditions_l646_646559

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l646_646559


namespace total_cost_correct_l646_646574

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end total_cost_correct_l646_646574


namespace friends_game_l646_646605

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l646_646605


namespace possible_number_of_friends_l646_646600

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l646_646600


namespace sqrt_fraction_1_sqrt_fraction_2_sqrt_general_sqrt_product_l646_646154

theorem sqrt_fraction_1 : sqrt (1 - 9 / 25) = 4 / 5 :=
sorry

theorem sqrt_fraction_2 : sqrt (1 - 15 / 64) = 7 / 8 :=
sorry

theorem sqrt_general (n : Nat) : sqrt (1 - (2 * n + 1) / ((n + 1)^2)) = n / (n + 1) :=
sorry

theorem sqrt_product (n : Nat) : sqrt (∏ i in Finset.range (n + 1), 1 - (2 * i + 1) / ((i + 1)^2)) = ∏ i in Finset.range n, i / (i + 1) :=
sorry

end sqrt_fraction_1_sqrt_fraction_2_sqrt_general_sqrt_product_l646_646154


namespace coins_remainder_divide_by_nine_remainder_l646_646554

def smallest_n (n : ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_remainder (n : ℕ) (h : smallest_n n) : (∃ m : ℕ, n = 54) :=
  sorry

theorem divide_by_nine_remainder (n : ℕ) (h : smallest_n n) (h_smallest: coins_remainder n h) : n % 9 = 0 :=
  sorry

end coins_remainder_divide_by_nine_remainder_l646_646554


namespace problem_statement_l646_646674

noncomputable def sequence (n : ℕ) : ℂ :=
  if n = 1 then (-1 + complex.I * (real.sqrt 3)) / 2
  else if n = 2 then (-1 - complex.I * (real.sqrt 3)) / 2
  else sorry -- Placeholder for further terms of the sequence based on the recurrence relation

lemma recurrence_relation (n : ℕ) (h : 2 ≤ n) :
  (sequence (n + 1) * sequence (n - 1) - sequence n ^ 2) 
  + complex.I * (sequence (n - 1) + sequence (n + 1) - 2 * sequence n) = 0 :=
sorry  -- This lemma corresponds to the given recurrence relation

theorem problem_statement : ∀ n : ℕ, 0 < n → 
  sequence n ^ 2 + sequence (n + 1) ^ 2 + sequence (n + 2) ^ 2 = 
  sequence n * sequence (n + 1) + sequence (n + 1) * sequence (n + 2) + sequence (n + 2) * sequence n :=
begin
  sorry, -- Proof goes here
end

end problem_statement_l646_646674


namespace right_triangles_count_l646_646794

-- Definitions according to the conditions.
variables (A B C D P Q : Type)
variables (rectangle : set (set (Type))) -- Representing the rectangle property.
variables (is_midpoint : ∀ {A B P : Type}, P = midpoint A B)
variables (ratio_DQ_QC : ∀ {D Q C : Type}, DQ = 2 * QC)

-- The main statement to prove.
theorem right_triangles_count (h1 : rectangle {A, B, C, D}) 
                              (h2 : is_midpoint A B P) 
                              (h3 : ratio_DQ_QC D Q C) : 
                                count_right_triangles {A, B, C, D, P, Q} = 8 :=
by
  sorry

end right_triangles_count_l646_646794


namespace find_x3_l646_646522

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1)

theorem find_x3
  (x1 x2 : ℝ)
  (h1 : 0 < x1)
  (h2 : x1 < x2)
  (h1_eq : x1 = 1)
  (h2_eq : x2 = Real.exp 3)
  : ∃ x3 : ℝ, x3 = Real.log (2 / 3 + 1 / 3 * Real.exp (Real.exp 3 - 1)) + 1 :=
by
  sorry

end find_x3_l646_646522


namespace smallest_number_value_l646_646510

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a + b + c = 100 ∧
  c = 2 * a ∧
  c - b = 10

theorem smallest_number_value (h : conditions a b c) : a = 22 :=
by
  sorry

end smallest_number_value_l646_646510


namespace positive_difference_abs_eq_15_l646_646952

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l646_646952


namespace range_of_a_l646_646711

-- Define the function f(x) = ln(x) + a * (1 - x)
def f (a x : ℝ) : ℝ := Real.log x + a * (1 - x)

-- Given conditions: f(x) has a maximum value greater than 2a - 2
-- Prove that the range of a is 0 < a < 1
theorem range_of_a (a : ℝ) (h_max : ∃ x, (f a x > 2 * a - 2)) : 0 < a ∧ a < 1 :=
by sorry

end range_of_a_l646_646711


namespace solve_conditions_l646_646443

noncomputable def solve_for_p_and_q : ℝ × ℝ :=
  let p := -31 / 12 in
  let q := 41 / 12 in
  (p, q)

theorem solve_conditions :
  ∃ p q : ℝ, (3 * 2 + p * 1 + (-1) * q = 0) ∧ (3^2 + p^2 + (-1)^2 = 2^2 + 1^2 + q^2) ∧
  solve_for_p_and_q = (p, q) :=
by
  exists (-31 / 12) (41 / 12)
  constructor
  { norm_num }
  constructor
  { norm_num }
  rfl

end solve_conditions_l646_646443


namespace men_absent_is_5_l646_646171

-- Define the given conditions
def original_number_of_men : ℕ := 30
def planned_days : ℕ := 10
def actual_days : ℕ := 12

-- Prove the number of men absent (x) is 5, under given conditions
theorem men_absent_is_5 : ∃ x : ℕ, 30 * planned_days = (original_number_of_men - x) * actual_days ∧ x = 5 :=
by
  sorry

end men_absent_is_5_l646_646171


namespace avg_k_value_l646_646329

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l646_646329


namespace symbols_in_P_l646_646409
-- Importing the necessary library

-- Define the context P and the operations
def context_P : Type := sorry

def mul_op (P : context_P) : String := "*"
def div_op (P : context_P) : String := "/"
def exp_op (P : context_P) : String := "∧"
def sqrt_op (P : context_P) : String := "SQR"
def abs_op (P : context_P) : String := "ABS"

-- Define what each symbol represents in the context of P
theorem symbols_in_P (P : context_P) :
  (mul_op P = "*") ∧
  (div_op P = "/") ∧
  (exp_op P = "∧") ∧
  (sqrt_op P = "SQR") ∧
  (abs_op P = "ABS") := 
sorry

end symbols_in_P_l646_646409


namespace problem_l646_646158

noncomputable def num_students : ℕ := 1650
noncomputable def num_rows : ℕ := 22
noncomputable def num_columns : ℕ := 75
noncomputable def max_pairs : ℕ := 11
noncomputable def max_boys : ℕ := 928

def conditions (a : Fin num_rows.succ → Fin num_columns.succ) : Prop :=
  ∀ u v : Fin num_columns.succ, u ≠ v →
  (∑ i in (Finset.range num_rows), if a i = u then 1 else 0 = a i) +
  (∑ i in (Finset.range num_rows), if a i = v then 1 else 0 = a i) ≤ max_pairs

theorem problem (a : Fin num_rows.succ → Fin num_columns.succ) (h : conditions a) : 
  (∑ i in (Finset.range num_rows), a i) ≤ max_boys :=
sorry

end problem_l646_646158


namespace average_of_distinct_k_l646_646313

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l646_646313


namespace average_of_k_l646_646321

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l646_646321


namespace maximum_value_of_F_l646_646070

def f (ω x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6)

def g (ω x : ℝ) : ℝ := Real.sin (ω * (x - Real.pi / 6) - Real.pi / 6)

def F (ω x : ℝ) : ℝ := f ω x + g ω x

theorem maximum_value_of_F (ω : ℝ) (hω : 0 < ω ∧ ω < 6) :
  ∃ x, F ω x = √3 := sorry

end maximum_value_of_F_l646_646070


namespace expression_of_function_f_f_increasing_on_interval_neg1_1_solution_set_of_inequality_l646_646256

variable (a b x : ℝ)
variable (f : ℝ → ℝ) {h₁ : f = λ x, (a * x + b) / (1 + x^2)} {h₂ : ∀ x, f (-x) = -f x}
variable (t : ℝ) {h₃ : -1 < t ∧ t < 1 ∧ 2 * t - 1 < 1 ∧ (2 * t - 1) > -1}

-- (Ⅰ) Expression for function f(x)
theorem expression_of_function_f :
  (∀ x, f x = (2 * x) / (1 + x^2)) :=
sorry

-- (Ⅱ) f(x) is an increasing function on (-1, 1)
theorem f_increasing_on_interval_neg1_1 :
  (∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2) :=
sorry

-- (Ⅲ) Solution set of inequality f(2t-1) + f(t) < 0
theorem solution_set_of_inequality :
  (f (2 * t - 1) + f t < 0) ↔ (0 < t ∧ t < 1 / 3) :=
sorry

end expression_of_function_f_f_increasing_on_interval_neg1_1_solution_set_of_inequality_l646_646256


namespace avg_k_for_polynomial_roots_l646_646295

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l646_646295


namespace line_intersects_y_axis_at_l646_646196

theorem line_intersects_y_axis_at : 
  ∃ y : ℝ, (∀ x : ℝ, (x, y) ∈ {(2, 3), (6, -9)} → ∃ m b : ℝ, y = m * x + b) ∧ (∀ x : ℝ, x = 0 → y = 9) :=
by
  sorry

end line_intersects_y_axis_at_l646_646196


namespace ratio_of_areas_of_S1_and_S3_l646_646438

noncomputable def S1 : Set (ℝ × ℝ) := 
  {p | log 10 (1 + p.1^2 + p.2^2) ≤ 1 + log 10 (p.1 + p.2)}

noncomputable def S3 : Set (ℝ × ℝ) := 
  {p | log 10 (3 + p.1^2 + p.2^2) ≤ 2 + log 10 (p.1 + p.2)}

theorem ratio_of_areas_of_S1_and_S3 : 
  let area_S1 := (Math.pi * 7^2)
  let area_S3 := (Math.pi * sqrt 4997 ^ 2)
  area_S3 / area_S1 = 102 :=
by
  sorry

end ratio_of_areas_of_S1_and_S3_l646_646438


namespace sum_of_values_b_l646_646000

noncomputable def T : ℤ := ∑ b in (Finset.filter (λ b : ℤ, ∃ r s : ℤ, (r + s = -b) ∧ (r * s = 2016 * b)) (Finset.range 1000000)), b

theorem sum_of_values_b :
  |T| = 665280 :=
by
  sorry

end sum_of_values_b_l646_646000


namespace average_speed_round_trip_l646_646176

-- Definitions and conditions
variables (D : ℝ) (speed_PtoQ speed_QtoP : ℝ)
def time_PtoQ := D / speed_PtoQ
def time_QtoP := D / (speed_PtoQ * 1.5) -- since speed is increased by 50%

-- Proof problem statement
theorem average_speed_round_trip :
  speed_PtoQ = 60 → speed_QtoP = 90 → 
  (let total_distance := 2 * D in
   let total_time := time_PtoQ D speed_PtoQ + time_QtoP D speed_PtoQ in
   total_distance / total_time = 72) :=
by
  intros,
  sorry -- Placeholder for the proof

end average_speed_round_trip_l646_646176


namespace locus_of_P_l646_646717

theorem locus_of_P (A B C D O1 I1 O2 I2 P : Type) [equilateral_triangle A B C] 
  (hD : point_on_side D B C) 
  (hO1 : is_circumcenter O1 (triangle A B D)) 
  (hI1 : is_incenter I1 (triangle A B D)) 
  (hO2 : is_circumcenter O2 (triangle A D C)) 
  (hI2 : is_incenter I2 (triangle A D C)) 
  (hP : line_intersection P (line O1 I1) (line O2 I2)) :
  ∃ (locus : set (Type)), locus = { P | ∃ (x y : ℝ), P = (x, y) ∧ y^2 - (x^2 / 3) = 1 ∧ -1 < x ∧ y < 0 } :=
sorry

end locus_of_P_l646_646717


namespace average_of_distinct_k_l646_646312

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l646_646312


namespace possible_value_of_sum_l646_646047

theorem possible_value_of_sum (x y : ℝ) 
  (h1 : x ^ 2 + 3 * x * y + y ^ 2 = 909)
  (h2 : 3 * x ^ 2 + x * y + 3 * y ^ 2 = 1287) : 
  x + y = 27 ∨ x + y = -27 :=
begin
  sorry,
end

end possible_value_of_sum_l646_646047


namespace real_solutions_quadratic_l646_646248

theorem real_solutions_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 - 4 * x + a = 0) ↔ a ≤ 4 :=
by sorry

end real_solutions_quadratic_l646_646248


namespace trigonometric_identity_l646_646761

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := 
by sorry

end trigonometric_identity_l646_646761


namespace smallest_number_of_rectangles_l646_646122

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l646_646122


namespace possible_number_of_friends_l646_646629

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l646_646629


namespace seq_formula_l646_646506

noncomputable def seq : ℕ → ℚ 
| 0       := 1
| (n + 1) := 2 * seq n / (2 + seq n)

theorem seq_formula (n : ℕ) : seq n = 2 / (n + 1) :=
by 
  induction n with k ih
  · simp [seq]
  · simp [seq, ih]
  · sorry

end seq_formula_l646_646506


namespace Robert_photo_count_l646_646842

theorem Robert_photo_count (k : ℕ) (hLisa : ∃ n : ℕ, k = 8 * n) : k = 24 - 16 → k = 24 :=
by
  intro h
  sorry

end Robert_photo_count_l646_646842


namespace coins_remainder_l646_646552

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l646_646552


namespace mathematial_problem_l646_646741

theorem mathematial_problem
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = |real.log (x + 1)|)
  (h2 : f a = f (-(b + 1) / (b + 2)))
  (h3 : f (10 * a + 6 * b + 21) = 4 * real.log 2)
  (h4 : a < b) : 
  (a = -2/5 ∧ (a + 1) * (b + 2) = 1) :=
by
  sorry

end mathematial_problem_l646_646741


namespace dollars_tina_l646_646854

open Real

theorem dollars_tina (P Q R S T : ℤ)
  (h1 : abs (P - Q) = 21)
  (h2 : abs (Q - R) = 9)
  (h3 : abs (R - S) = 7)
  (h4 : abs (S - T) = 6)
  (h5 : abs (T - P) = 13)
  (h6 : P + Q + R + S + T = 86) :
  T = 16 :=
sorry

end dollars_tina_l646_646854


namespace geometric_sequence_property_l646_646268

-- Define the geometric sequence terms
variable {a : ℕ → ℝ}

-- Define the given condition
def condition1 : Prop := a 6 + a 8 = 4

-- Define the proposition to be proven
theorem geometric_sequence_property (h1 : condition1) : a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end geometric_sequence_property_l646_646268


namespace number_of_odd_integers_l646_646813

-- Define the function f based on the given conditions
def f : ℕ+ → ℕ
| ⟨1, _⟩     := 1
| ⟨k, hk⟩ := 
    if even: (2 * (k / 2)) = k then 
      f ⟨k / 2, nat.div_pos hk (by norm_num)⟩
    else 
      let n := k / 2 in
      f ⟨n, nat.pos_of_ne_zero (by exact_mod_cast nat.div_pos hk (by norm_num))⟩ + f ⟨n.succ, nat.succ_pos'⟩

-- Define Euler's totient function φ
def euler_totient (n : ℕ) : ℕ := finset.card { m : finset.range (n + 1) // nat.coprime m n }

theorem number_of_odd_integers (n : ℕ+) :
  { m : ℕ+ | odd m ∧ f m = n } .to_finset.card = euler_totient n :=
sorry

end number_of_odd_integers_l646_646813


namespace total_value_of_item_l646_646138

variable (V : ℝ) -- Total value of the item

def import_tax (V : ℝ) := 0.07 * (V - 1000) -- Definition of import tax

theorem total_value_of_item
  (htax_paid : import_tax V = 112.70) :
  V = 2610 := 
by
  sorry

end total_value_of_item_l646_646138


namespace f_eq_zero_range_x_l646_646292

-- Definition of the function f on domain ℝ*
def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_domain : ∀ x : ℝ, x ≠ 0 → f x = f x
axiom f_4 : f 4 = 1
axiom f_multiplicative : ∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → f (x1 * x2) = f x1 + f x2
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y

-- Problem (1): Prove f(1) = 0
theorem f_eq_zero : f 1 = 0 :=
sorry

-- Problem (2): Prove range 3 < x ≤ 5 given the inequality condition
theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 :=
sorry

end f_eq_zero_range_x_l646_646292


namespace direction_vectors_of_line_l646_646882

theorem direction_vectors_of_line : 
  ∃ v : ℝ × ℝ, (3 * v.1 - 4 * v.2 = 0) ∧ (v = (1, 3/4) ∨ v = (4, 3)) :=
by
  sorry

end direction_vectors_of_line_l646_646882


namespace at_least_one_with_3_distinct_prime_factors_l646_646027

theorem at_least_one_with_3_distinct_prime_factors
    (n : ℕ) (h : n ≥ 93) :
    ∃ k ∈ (Finset.range 10).map (Function.add n), 3 ≤ Prime.factorization k.vars.count :=
sorry

end at_least_one_with_3_distinct_prime_factors_l646_646027


namespace sum_of_products_of_operations_l646_646094

theorem sum_of_products_of_operations :
  let initial_board : List ℕ := [1, 3, 5, 7, 9]
  let operations (board : List ℕ) (paper : List ℕ) :=
    ∃ (a b : ℕ) (board' : List ℕ), 
      a ∈ board ∧ b ∈ board ∧ a ≠ b ∧
      board' = (board.erase a).erase b ++ [a + b] ∧
      paper = paper ++ [a * b]
  in ∀ final_board final_paper,
    (final_board, final_paper) = operations (operations (operations (operations (initial_board, [])).1).1).1).1) 
    → final_board.length = 1 ∧ final_paper.sum = 230 :=
begin
  sorry
end

end sum_of_products_of_operations_l646_646094


namespace digit_150th_of_5_div_13_l646_646925

theorem digit_150th_of_5_div_13 : 
    ∀ k : ℕ, (k = 150) → (fractionalPartDigit k (5 / 13) = 5) :=
by 
  sorry

end digit_150th_of_5_div_13_l646_646925


namespace friends_game_l646_646604

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l646_646604


namespace area_of_triangle_DBC_is_12_5_l646_646780

noncomputable def A : ℝ × ℝ := (0, 5)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (10, 0)
noncomputable def D : ℝ × ℝ := (0, (5 / 2))
noncomputable def E : ℝ × ℝ := ((10 / 2), 0)

def length_BC : ℝ := real.sqrt ((C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2)

def height_D : ℝ := D.2 - B.2

def area_triangle_DBC : ℝ := (1 / 2) * length_BC * height_D

theorem area_of_triangle_DBC_is_12_5 : area_triangle_DBC = 12.5 := by
  -- The proof is not required, so we use sorry.
  sorry

end area_of_triangle_DBC_is_12_5_l646_646780


namespace stephanie_swimming_days_two_years_l646_646869

def is_same_parity (a b : ℕ) : Bool :=
  (a % 2 = b % 2)

def swims (day month year : ℕ) : Bool :=
  is_same_parity day month && is_same_parity month year

def count_swimming_days (year : ℕ) : ℕ :=
  let monthsWith31Days := [1, 3, 5, 7, 9, 11]
  let monthsWith30Days := [4, 6, 8, 10, 12]
  let februaryDays := if year % 4 = 0 && (year % 100 != 0 ∨ year % 400 = 0) then 29 else 28
  
  let countDaysInMonths(months : List ℕ, daysInMonth : ℕ → ℕ) : ℕ :=
    months.foldr (fun month acc => acc + (List.range (daysInMonth month)).count (swims year month)) 0
  let oddYearDays := countDaysInMonths(monthsWith31Days, fun m => 31) + countDaysInMonths([9, 11], fun m => 30)
  let evenYearDays := countDaysInMonths(monthsWith30Days, fun m => 30) + countDaysInMonths([2], fun _ => februaryDays)
  
  if year % 2 = 0 then -- Even year
    oddYearDays
  else -- Odd year
    evenYearDays

theorem stephanie_swimming_days_two_years (start_year : ℕ) : count_swimming_days start_year + count_swimming_days (start_year + 1) = 183 :=
  sorry

end stephanie_swimming_days_two_years_l646_646869


namespace one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646921

theorem one_fiftieth_digit_of_five_over_thirteen_is_five :
  (decimal_fraction 5 13).digit 150 = 5 :=
by sorry

end one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646921


namespace range_of_log2_sqrt_cos_l646_646963

noncomputable def cos_func : ℝ → ℝ := λ x, real.cos x
noncomputable def sqrt_func : ℝ → ℝ := λ x, real.sqrt x
noncomputable def log2_func : ℝ → ℝ := λ x, real.log x / real.log 2

theorem range_of_log2_sqrt_cos :
  ∀ x : ℝ, (-90 : ℝ) < x ∧ x < (90 : ℝ) → -∞ < log2_func (sqrt_func (cos_func x)) ∧ log2_func (sqrt_func (cos_func x)) ≤ 0 :=
begin
  sorry
end

end range_of_log2_sqrt_cos_l646_646963


namespace balance_after_6_months_l646_646651

noncomputable def final_balance : ℝ :=
  let balance_m1 := 5000 * (1 + 0.04 / 12)
  let balance_m2 := (balance_m1 + 1000) * (1 + 0.042 / 12)
  let balance_m3 := balance_m2 * (1 + 0.038 / 12)
  let balance_m4 := (balance_m3 - 1500) * (1 + 0.05 / 12)
  let balance_m5 := (balance_m4 + 750) * (1 + 0.052 / 12)
  let balance_m6 := (balance_m5 - 1000) * (1 + 0.045 / 12)
  balance_m6

theorem balance_after_6_months : final_balance = 4371.51 := sorry

end balance_after_6_months_l646_646651


namespace avg_k_for_polynomial_roots_l646_646300

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l646_646300


namespace distance_M0_to_plane_is_correct_l646_646983

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance_point_to_plane (p : Point3D) (A B C D : ℝ) : ℝ :=
  abs (A * p.x + B * p.y + C * p.z + D) / Real.sqrt (A * A + B * B + C * C)

noncomputable def plane_through_points 
  (p1 p2 p3 : Point3D) : ℝ × ℝ × ℝ × ℝ :=
  let A := (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y)
  let B := (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z)
  let C := (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
  let D := - (A * p1.x + B * p1.y + C * p1.z)
  (A, B, C, D)

def M0 : Point3D := { x := 2, y := -1, z := 4 }
def M1 : Point3D := { x := 1, y := 2, z := 0 }
def M2 : Point3D := { x := 1, y := -1, z := 2 }
def M3 : Point3D := { x := 0, y := 1, z := -1 }

noncomputable def plane_coeffs : ℝ × ℝ × ℝ × ℝ :=
  plane_through_points M1 M2 M3

theorem distance_M0_to_plane_is_correct :
  distance_point_to_plane M0 plane_coeffs.1 plane_coeffs.2 plane_coeffs.3 plane_coeffs.4 = 
  1 / Real.sqrt 38 := by
  sorry

end distance_M0_to_plane_is_correct_l646_646983


namespace coins_remainder_l646_646563

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l646_646563


namespace locus_of_Q_l646_646515

noncomputable def A_x : ℝ := -4
noncomputable def parabola (p : ℝ) (h : p > 0) : Set (ℝ × ℝ) := { x | ∃ y, y^2 = 2 * p * x.fst }
noncomputable def B : ℝ × ℝ := (1, 2)

theorem locus_of_Q (p : ℝ) (h : p > 0) (λ₁ λ₂ : ℝ) (h₁ : λ₁ > 0) (h₂ : λ₂ > 0) 
  (condition : (2 / λ₁) + (3 / λ₂) = 15) :
  ∃ eq : polynomial ℝ, ∀ Q, on_locus Q eq ↔ 
    ∃ P : ℝ × ℝ, P ∈ parabola p h ∧ P ≠ B ∧ 
    ∃ (E F D : ℝ × ℝ), 
      vector_related P E A λ₁ ∧
      vector_related P F B λ₂ ∧
      line_intersection (line E F) (line P.source D.source) Q := 
sorry

end locus_of_Q_l646_646515


namespace Suma_work_time_l646_646479

theorem Suma_work_time (W : ℝ) (h1 : W > 0) :
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  suma_time = 8 :=
by 
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  exact sorry

end Suma_work_time_l646_646479


namespace sum_of_squares_of_geometric_progression_l646_646102

theorem sum_of_squares_of_geometric_progression 
  {b_1 q S_1 S_2 : ℝ} 
  (h1 : |q| < 1) 
  (h2 : S_1 = b_1 / (1 - q))
  (h3 : S_2 = b_1 / (1 + q)) : 
  (b_1^2 / (1 - q^2)) = S_1 * S_2 := 
by
  sorry

end sum_of_squares_of_geometric_progression_l646_646102


namespace percentage_games_won_l646_646179

theorem percentage_games_won (total_games : ℕ) (wins : ℕ)
  (h_total_games : total_games = 130)
  (h_wins : wins = 78) :
  (wins.to_rat / total_games.to_rat) * 100 = 60 := by
  sorry

end percentage_games_won_l646_646179


namespace coins_remainder_l646_646553

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l646_646553


namespace geometric_sequence_problem_l646_646726

variable (a : ℕ → ℝ)
variable (r : ℝ) (hpos : ∀ n, 0 < a n)

theorem geometric_sequence_problem
  (hgeom : ∀ n, a (n+1) = a n * r)
  (h_eq : a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) :
  a 2 + a 6 = 2 :=
sorry

end geometric_sequence_problem_l646_646726


namespace half_radius_of_circle_y_l646_646148

theorem half_radius_of_circle_y
  (r_x r_y : ℝ)
  (hx : π * r_x ^ 2 = π * r_y ^ 2)
  (hc : 2 * π * r_x = 10 * π) :
  r_y / 2 = 2.5 :=
by
  sorry

end half_radius_of_circle_y_l646_646148


namespace min_rectangles_to_cover_square_exactly_l646_646128

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l646_646128


namespace range_of_a_l646_646260

open Real

-- Definition of the conditions given in the problem.
def conditions (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ log a 3 < 1

-- Statement of the problem, which is a theorem in Lean.
theorem range_of_a (a : ℝ) (h : conditions a) : a ∈ (Iio 1) ∪ (Ioi 3) :=
by 
  sorry

end range_of_a_l646_646260


namespace expected_value_of_unfair_die_l646_646645

-- Define the probabilities for each face of the die.
def prob_face (n : ℕ) : ℚ :=
  if n = 8 then 5/14 else 1/14

-- Define the expected value of a roll of this die.
def expected_value : ℚ :=
  (1 / 14) * 1 + (1 / 14) * 2 + (1 / 14) * 3 + (1 / 14) * 4 + (1 / 14) * 5 + (1 / 14) * 6 + (1 / 14) * 7 + (5 / 14) * 8

-- The statement to prove: the expected value of a roll of this die is 4.857.
theorem expected_value_of_unfair_die : expected_value = 4.857 := by
  sorry

end expected_value_of_unfair_die_l646_646645


namespace flights_distribution_l646_646573

theorem flights_distribution (G : SimpleGraph (Fin 20)) 
  (h1 : ∀ v, G.degree v = 4) : 
  ∃ f : G.edge → Fin 2, ∀ v, Finset.card (f '' G.incident_edges v) = 2 := 
sorry

end flights_distribution_l646_646573


namespace percentage_increase_l646_646498

theorem percentage_increase (N : ℝ) (P : ℝ) (h1 : N + (P / 100) * N - (N - 25 / 100 * N) = 30) (h2 : N = 80) : P = 12.5 :=
by
  sorry

end percentage_increase_l646_646498


namespace fill_cup_times_l646_646202

theorem fill_cup_times (total_water_needed : ℚ) (cup_capacity : ℚ) (fills_required : ℕ) :
  total_water_needed = 13 / 3 ∧ cup_capacity = 1 / 6 → fills_required = 26 := by
  intro h
  cases h with h1 h2
  have h3 : fills_required = (total_water_needed / cup_capacity).natAbs := by
    calc
      total_water_needed / cup_capacity = (13 / 3) / (1 / 6) := by rw [h1, h2]
      ... = (13 / 3) * (6 / 1) := div_mul_div
      ... = 13 * 6 / 3 := mul_comm
      ... = 78 / 3 := by norm_num
      ... = 26 := by norm_num
  exact h3

end fill_cup_times_l646_646202


namespace no_four_digit_number_differs_from_reverse_by_1008_l646_646206

theorem no_four_digit_number_differs_from_reverse_by_1008 :
  ∀ a b c d : ℕ, 
  a < 10 → b < 10 → c < 10 → d < 10 → (999 * (a - d) + 90 * (b - c) ≠ 1008) :=
by
  intro a b c d ha hb hc hd h
  sorry

end no_four_digit_number_differs_from_reverse_by_1008_l646_646206


namespace a_plus_b_l646_646824
open Set

-- Definition of the set S
def S : Set (List ℕ) :=
  {l | l.perm [1, 2, 3, 4, 5] ∧ l.head ≠ 1}

-- Count the number of permutations where the second term is 2
def count_favorable_permutations : ℕ :=
  3 * (factorial 3)  -- 3 choices for the first term, 3! arrangements for the rest

-- Total number of permutations in S
def total_permutations_in_S : ℕ :=
  5 * (factorial 4)  -- 5 choices minus 1 for the first term being 1 

-- Probability calculation
def probability_of_second_term_2 : ℚ :=
  (count_favorable_permutations : ℚ) / (total_permutations_in_S : ℚ)

-- Statement to prove
theorem a_plus_b : 3 + 16 = 19 :=
by sorry

end a_plus_b_l646_646824


namespace number_of_zeros_l646_646276

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def conditions (f : ℝ → ℝ) (f'' : ℝ → ℝ) :=
  odd_function f ∧ ∀ x : ℝ, x < 0 → (2 * f x + x * f'' x < x * f x)

theorem number_of_zeros (f : ℝ → ℝ) (f'' : ℝ → ℝ) (h : conditions f f'') :
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_l646_646276


namespace cosine_of_angle_between_l646_646745

variables (a b : Vector3) [NormedAddCommGroup (Vector3)] [InnerProductSpace ℝ (Vector3)]

def ab_magnitude_eq_2 (a b : Vector3) : Prop :=
  ∥a∥ = 2 ∧ ∥b∥ = 2

def a_dot_a_plus_b (a b : Vector3) : Prop :=
  inner a (a + b) = 7

theorem cosine_of_angle_between (a b : Vector3) 
  (h1 : ab_magnitude_eq_2 a b) 
  (h2 : a_dot_a_plus_b a b) :
  inner a b / (∥a∥ * ∥b∥) = 3 / 4 :=
sorry

end cosine_of_angle_between_l646_646745


namespace a_minus_b_l646_646076

noncomputable def find_a_b (a b : ℝ) :=
  ∃ k : ℝ, ∀ (x : ℝ) (y : ℝ), 
    (k = 2 + a) ∧ 
    (y = k * x + 1) ∧ 
    (y = x^2 + a * x + b) ∧ 
    (x = 1) ∧ (y = 3)

theorem a_minus_b (a b : ℝ) (h : find_a_b a b) : a - b = -2 := by 
  sorry

end a_minus_b_l646_646076


namespace radius_of_circle_through_AD_tangent_to_BC_l646_646486

theorem radius_of_circle_through_AD_tangent_to_BC (A B C D : Point) (r : ℝ) 
  (h_square : square A B C D) (h_side : distance A B = 14) (h_circle : circle_through A D) (h_tangent : tangent_to_circle B C) : 
  r = 8.75 := sorry

end radius_of_circle_through_AD_tangent_to_BC_l646_646486


namespace coins_remainder_l646_646565

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l646_646565


namespace tina_sequence_erasure_l646_646912

open Nat

def initial_sequence : List ℕ :=
  List.repeat [1, 2, 3, 4, 5, 6] 2500 >>= id

def erase_every_nth {α : Type} (lst : List α) (n : ℕ) : List α :=
  lst.enum.filter (λ ⟨idx, _⟩, (idx + 1) % n ≠ 0).map Prod.snd

def final_sequence : List ℕ :=
  erase_every_nth (erase_every_nth (erase_every_nth initial_sequence 4) 5) 6

def positions := [3018, 3019, 3020]  -- zero-indexed

def sum_positions (lst : List ℕ) (pos : List ℕ) : ℕ :=
  pos.map (λ i, lst.nth_le i sorry).sum

theorem tina_sequence_erasure :
  sum_positions final_sequence positions = 5 := 
sorry

end tina_sequence_erasure_l646_646912


namespace find_angle_eq_120_degrees_l646_646826

variables {V : Type*} [inner_product_space ℝ V]

def norm (v : V) : ℝ := real.sqrt (inner_product_space.norm_sq ℝ v)

noncomputable def vector_angle {a b : V} (h₀ : a ≠ 0) (h₁ : b ≠ 0) 
  (h₂ : norm a = norm b) (h₃ : norm (a + b) = norm a) : ℝ :=
let d := norm a in
real.acos ((a ⬝ b) / (d * d))

theorem find_angle_eq_120_degrees {a b : V} (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : norm a = norm b) (h₃ : norm (a + b) = norm a) :
  vector_angle h₀ h₁ h₂ h₃ = 120 :=
by { sorry }

end find_angle_eq_120_degrees_l646_646826


namespace group_of_friends_l646_646588

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l646_646588


namespace not_valid_pair_16_4_l646_646812

noncomputable def valid_pair (a : ℕ → ℤ) (i j : ℕ) : Prop :=
  (∀ n, a (n + 48) ≡ a n [ZMOD 35]) ∧
  (∀ n, a (n + i) ≡ a n [ZMOD 5]) ∧
  (∀ n, a (n + j) ≡ a n [ZMOD 7])

theorem not_valid_pair_16_4 (a : ℕ → ℤ) :
  (∀ n, a (n + 48) ≡ a n [ZMOD 35]) →
  ¬ valid_pair a 16 4 :=
begin
  intros h,
  unfold valid_pair,
  intro h1,
  cases h1 with h2 h3,
  cases h3 with h4 h5,
  -- Additional proof steps would go here, skipping for now
  sorry
end

end not_valid_pair_16_4_l646_646812


namespace volume_is_85_l646_646207

/-!
# Proof Problem
Prove that the total volume of Carl's and Kate's cubes is 85, given the conditions,
Carl has 3 cubes each with a side length of 3, and Kate has 4 cubes each with a side length of 1.
-/

-- Definitions for the problem conditions:
def volume_of_cube (s : ℕ) : ℕ := s^3

def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_of_cube s

-- Given conditions
def carls_cubes_volume : ℕ := total_volume 3 3
def kates_cubes_volume : ℕ := total_volume 4 1

-- The total volume of Carl's and Kate's cubes:
def total_combined_volume : ℕ := carls_cubes_volume + kates_cubes_volume

-- Prove the total volume is 85
theorem volume_is_85 : total_combined_volume = 85 :=
by sorry

end volume_is_85_l646_646207


namespace derivative_of_sin_squared_l646_646495

variable (x : ℝ)

def f (x : ℝ) : ℝ := sin x ^ 2

theorem derivative_of_sin_squared :
  deriv f x = sin (2 * x) :=
sorry

end derivative_of_sin_squared_l646_646495


namespace distance_between_parallel_lines_l646_646063

theorem distance_between_parallel_lines :
  let line1 := λ x y : Real, 3 * x + 4 * y - 5 = 0,
      line2 := λ x y : Real, 6 * x + 8 * y - 15 = 0 in
  let a := 3,
      b := 4,
      c1 := -5,
      c2 := -15 / 2 in
  Real.abs (c2 - c1) / Real.sqrt (a^2 + b^2) = 1 / 2 :=
  sorry

end distance_between_parallel_lines_l646_646063


namespace stratified_sampling_correct_l646_646187

noncomputable def stratified_sampling (total senior intermediate junior other sample_size : ℕ) : Prop :=
  let sampling_fraction := sample_size / total in
  let senior_sample := senior / (total / sample_size) in
  let intermediate_sample := intermediate / (total / sample_size) in
  let junior_sample := junior / (total / sample_size) in
  let other_sample := other / (total / sample_size) in
  senior_sample = 8 ∧ intermediate_sample = 16 ∧ junior_sample = 10 ∧ other_sample = 6

theorem stratified_sampling_correct :
  stratified_sampling 800 160 320 200 120 40 :=
by 
  sorry

end stratified_sampling_correct_l646_646187


namespace patricia_lemon_heads_packages_l646_646763

theorem patricia_lemon_heads_packages (ate gave : ℕ) (package_size : ℕ) 
  (h_ate : ate = 15) (h_gave : gave = 5) (h_package_size : package_size = 3) : 
  (ate + gave + package_size - 1) / package_size = 7 :=
by
  rw [h_ate, h_gave, h_package_size]
  sorry

end patricia_lemon_heads_packages_l646_646763


namespace avg_k_value_l646_646335

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l646_646335


namespace cosine_F_in_triangle_DEF_l646_646420

theorem cosine_F_in_triangle_DEF
  (D E F : ℝ)
  (h_triangle : D + E + F = π)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = - (16 / 65) := by
  sorry

end cosine_F_in_triangle_DEF_l646_646420


namespace intergalactic_math_olympiad_score_l646_646656

theorem intergalactic_math_olympiad_score :
  ∃ score : ℕ, score = 1 ∧ 
  (let n := 6,
       problem_scores := (λ (i : Fin n), 1),
       total_scores := problem_scores 0 * problem_scores 1 * problem_scores 2 * problem_scores 3 * problem_scores 4 * problem_scores 5,
       total_participants := 8^n,
       positive_scores := 7^n,
       rank := positive_scores in 
    total_participants = 262144 ∧ rank = 117649 ∧ ∀ i, 0 ≤ problem_scores i ∧ problem_scores i ≤ 7 ∧
    total_scores ≤ total_participants ∧ 
    total_scores = rank
  ) :=
by {
  sorry
}

end intergalactic_math_olympiad_score_l646_646656


namespace problem_correct_statements_l646_646535

noncomputable def statement_A (α : ℝ) (h : 0 < α ∧ α < π / 2) : Prop := 
  (0 < α/2) ∧ (α/2 < π / 2)

noncomputable def statement_B (f : ℝ → ℝ) (ϕ : ℝ) (h : ∀ x : ℝ, f x = sin (x + ϕ + π / 4)) : Prop := 
  ¬ (∀ x : ℝ, f x = f (-x)) ∧ ϕ = 3 * π / 4

noncomputable def statement_C (x : ℝ) (f : ℝ → ℝ) : Prop := 
  x = π / 3 ∧ f x = 2 * cos (2 * x + π / 3)

noncomputable def statement_D (θ : ℝ) (r : ℝ) : Prop := 
  θ = π / 3 ∧ r = 1 ∧ (r * θ ≠ 60)

theorem problem_correct_statements :
  (0 < α ∧ α < π/2 → statement_A α (by sorry)) ∧
  (∀ (f : ℝ → ℝ), ∀ (ϕ : ℝ), ∀ (h : ∀ x : ℝ, f x = sin (x + ϕ + π/4)), statement_B f ϕ h) ∧
  (∀ (x : ℝ), ∀ (f : ℝ → ℝ), statement_C x f) ∧
  (statement_D (π / 3) 1) :=
by sorry

end problem_correct_statements_l646_646535


namespace Eliza_first_more_than_300_paperclips_on_Thursday_l646_646224

theorem Eliza_first_more_than_300_paperclips_on_Thursday :
  ∃ k : ℕ, 5 * 3^k > 300 ∧ k = 4 := 
by
  sorry

end Eliza_first_more_than_300_paperclips_on_Thursday_l646_646224


namespace tens_digit_of_seven_times_cubed_is_one_l646_646147

-- Variables and definitions
variables (p : ℕ) (h1 : p < 10)

-- Main theorem statement
theorem tens_digit_of_seven_times_cubed_is_one (hp : p < 10) :
  let N := 11 * p
  let m := 7
  let result := m * N^3
  (result / 10) % 10 = 1 := 
sorry

end tens_digit_of_seven_times_cubed_is_one_l646_646147


namespace ellipse_line_intersect_find_line_given_chord_length_l646_646348

noncomputable theory

def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

theorem ellipse_line_intersect (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line m x y) ↔ (-Real.sqrt(5) / 2 ≤ m ∧ m ≤ Real.sqrt(5) / 2) :=
by sorry

theorem find_line_given_chord_length (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, ellipse x1 y1 ∧ ellipse x2 y2 ∧ line m x1 y1 ∧ line m x2 y2 ∧
   Real.dist (x1, y1) (x2, y2) = 4 * Real.sqrt(2) / 5) → (m = 1 / 2 ∨ m = -1 / 2) :=
by sorry

end ellipse_line_intersect_find_line_given_chord_length_l646_646348


namespace collinear_condition_l646_646080

-- Definitions
variables {α : Type*} [Field α] [VectorSpace α (Vector α)] {a b : Vector α}

-- Assumptions
axiom a_non_zero : a ≠ 0

-- The necessary and sufficient condition for collinearity
theorem collinear_condition : (∃ λ : α, b = λ • a) ↔ b ≠ 0 ∧ ∃! λ : α, b = λ • a :=
sorry

end collinear_condition_l646_646080


namespace total_plums_picked_l646_646844

-- Conditions
def Melanie_plums : ℕ := 4
def Dan_plums : ℕ := 9
def Sally_plums : ℕ := 3

-- Proof statement
theorem total_plums_picked : Melanie_plums + Dan_plums + Sally_plums = 16 := by
  sorry

end total_plums_picked_l646_646844


namespace bucket_full_weight_l646_646990

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = p) 
  (h2 : x + (3 / 4) * y = q) : 
  x + y = (8 * q - 3 * p) / 5 := 
  by
    sorry

end bucket_full_weight_l646_646990


namespace min_transport_cost_l646_646681

-- Definitions based on conditions
def total_washing_machines : ℕ := 100
def typeA_max_count : ℕ := 4
def typeB_max_count : ℕ := 8
def typeA_cost : ℕ := 400
def typeA_capacity : ℕ := 20
def typeB_cost : ℕ := 300
def typeB_capacity : ℕ := 10

-- Minimum transportation cost calculation
def min_transportation_cost : ℕ :=
  let typeA_trucks_used := min typeA_max_count (total_washing_machines / typeA_capacity)
  let remaining_washing_machines := total_washing_machines - typeA_trucks_used * typeA_capacity
  let typeB_trucks_used := min typeB_max_count (remaining_washing_machines / typeB_capacity)
  typeA_trucks_used * typeA_cost + typeB_trucks_used * typeB_cost

-- Lean 4 statement to prove the minimum transportation cost
theorem min_transport_cost : min_transportation_cost = 2200 := by
  sorry

end min_transport_cost_l646_646681


namespace conjugate_of_z_l646_646384

theorem conjugate_of_z (z : ℂ) (h : z * complex.I - 3 * complex.I = complex.abs (3 + 4 * complex.I)) :
  complex.conj z = 3 + 5 * complex.I :=
sorry

end conjugate_of_z_l646_646384


namespace EquilateralTriangleIfAcuteAndArea_l646_646406

variable (A B C : Type) [EuclideanGeometry A B C] -- Assume A, B, and C are types with Euclidean geometry structure
variable (angleA_eq_pi_div_3 : angle A = π / 3)
variable (RegionG : set A) -- Define Region G
variable (areaG_eq_one_third_areaABC : area RegionG = (1 / 3) * area (triangle A B C))

theorem EquilateralTriangleIfAcuteAndArea (
  h1 : acute_triangle A B C,
  h2 : angleA_eq_pi_div_3,
  h3 : ∀ P, P ∈ RegionG ↔ (dist P A ≤ dist P B ∧ dist P A ≤ dist P C),
  h4 : areaG_eq_one_third_areaABC
) : equilateral_trianlge A B C :=
sorry

end EquilateralTriangleIfAcuteAndArea_l646_646406


namespace intersection_points_l646_646437

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)

noncomputable def g (x : ℝ) : ℝ := (-3*x^2 - 6*x + 115) / (x - 2)

theorem intersection_points:
  ∃ (x1 x2 : ℝ), x1 ≠ -3 ∧ x2 ≠ -3 ∧ (f x1 = g x1) ∧ (f x2 = g x2) ∧ 
  (x1 = -11 ∧ f x1 = -2) ∧ (x2 = 3 ∧ f x2 = -2) := 
sorry

end intersection_points_l646_646437


namespace avg_k_value_l646_646332

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l646_646332


namespace min_rectangles_to_cover_square_exactly_l646_646132

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l646_646132


namespace number_of_correct_propositions_l646_646442

noncomputable def alpha : Type := sorry
noncomputable def beta : Type := sorry
noncomputable def l : Type := sorry
noncomputable def A : Type := sorry
noncomputable def B : Type := sorry
noncomputable def C : Type := sorry

axiom prop1 : (A ∈ l) → (A ∈ alpha) → (B ∈ l) → (B ∈ alpha) → l ⊆ alpha
axiom prop2 : (A ∈ alpha) → (A ∈ beta) → (B ∈ alpha) → (B ∈ beta) → (alpha ∩ beta = AB)
axiom prop3 : (l ∉ alpha) → (A ∈ l) → (A ∉ alpha)

theorem number_of_correct_propositions : 2 := sorry

end number_of_correct_propositions_l646_646442


namespace problem1_problem2_problem3_l646_646743

noncomputable def y (x a : ℝ) : ℝ := real.log (x^2 + 2 * x + a)

-- Theorem for problem 1
theorem problem1 (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ 1 < a :=
sorry

-- Theorem for problem 2
theorem problem2 (a : ℝ) : (∀ y : ℝ, 0 ≤ y ↔ ∃ x : ℝ, y = real.log (x^2 + 2 * x + a)) ↔ a = 2 :=
sorry

-- Theorem for problem 3
theorem problem3 (a : ℝ) : (∀ y : ℝ, ∃ x : ℝ, y = real.log (x^2 + 2 * x + a)) ↔ a ≤ 1 :=
sorry

end problem1_problem2_problem3_l646_646743


namespace range_for_definitions_range_for_inequality_l646_646500

-- Proof problem 1: Range of values for a for f(x) and g(x) to be defined for x in [a+2, a+3]
theorem range_for_definitions (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (∀ x, x ∈ Icc (a + 2) (a + 3) → x - 3 * a > 0 ∧ x - a > 0) ↔ (0 < a ∧ a < 1) :=
sorry

-- Proof problem 2: Range of values for a such that |f(x) - g(x)| ≤ 1 in [a+2, a+3]
theorem range_for_inequality (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (∀ x, x ∈ Icc (a + 2) (a + 3) → |(log a (x - 3 * a)) - (log a (1 / (x - a)))| ≤ 1) ↔ (0 < a ∧ a ≤ (9 - real.sqrt 57) / 12) :=
sorry

end range_for_definitions_range_for_inequality_l646_646500


namespace find_k_l646_646165

def total_balls (k : ℕ) : ℕ := 7 + k

def probability_green (k : ℕ) : ℚ := 7 / (total_balls k)
def probability_purple (k : ℕ) : ℚ := k / (total_balls k)

def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * 3 + (probability_purple k) * (-1)

theorem find_k (k : ℕ) (h_pos : k > 0) (h_exp_value : expected_value k = 1) : k = 7 :=
sorry

end find_k_l646_646165


namespace bicycle_wheel_diameter_l646_646163

noncomputable def pi : ℝ := Real.pi

theorem bicycle_wheel_diameter :
  ∀ (revolutions : ℝ) (distance : ℝ), 
    revolutions = 522.0841599665866 →
    distance = 1000 →
    let d := distance / (pi * revolutions) in
    d ≈ 0.6096 :=
by
  intros
  let d := distance / (pi * revolutions)
  have h : d ≈ 0.6096 := sorry
  exact h

end bicycle_wheel_diameter_l646_646163


namespace sec_sum_l646_646375

theorem sec_sum (x y : ℝ) (h1 : sin x + sin y = 45 / 53) (h2 : cos x + cos y = 28 / 53) : 
  sec x + sec y = 53 / 14 :=
by
  sorry

end sec_sum_l646_646375


namespace quarters_number_l646_646523

theorem quarters_number (total_value : ℝ)
    (bills1 : ℝ := 2)
    (bill5 : ℝ := 5)
    (dimes : ℝ := 20 * 0.1)
    (nickels : ℝ := 8 * 0.05)
    (pennies : ℝ := 35 * 0.01) :
    total_value = 13 → (total_value - (bills1 + bill5 + dimes + nickels + pennies)) / 0.25 = 13 :=
by
  intro h
  have h_total := h
  sorry

end quarters_number_l646_646523


namespace necessary_but_not_sufficient_l646_646712

variable (a b : Real)

-- Definitions of p and q
def p : Prop := (b + 2) / (a + 2) > b / a
def q : Prop := a > b ∧ b > 0

-- Formal statement of the equivalence problem
theorem necessary_but_not_sufficient : (q → p) ∧ ¬(p → q) :=
by
  sorry

end necessary_but_not_sufficient_l646_646712


namespace chef_earning_less_than_manager_l646_646195

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.25

theorem chef_earning_less_than_manager : manager_wage - chef_wage = 3.1875 :=
by
  -- All values are precalculated based on the conditions
  let m_wage : ℝ := 8.50
  let d_wage : ℝ := m_wage / 2
  let c_wage : ℝ := d_wage * 1.25
  have h₁ : d_wage = 4.25 := by norm_num
  have h₂ : c_wage = 5.3125 := by norm_num
  have h₃ : m_wage - c_wage = 8.50 - 5.3125 := by norm_num
  show 3.1875 = 3.1875, from rfl

end chef_earning_less_than_manager_l646_646195


namespace distinct_brick_heights_l646_646868

theorem distinct_brick_heights : ∃ (n : ℕ), n = 781 ∧ (∀ height ∈ (finset.range 781).image (λ h, 120 + (p * 4) + (q * 13)), ∃ (p q : ℕ), height = 120 + (p * 4) + (q * 13)) :=
by
  sorry

end distinct_brick_heights_l646_646868


namespace simplify_power_of_power_l646_646866

theorem simplify_power_of_power (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end simplify_power_of_power_l646_646866


namespace number_of_friends_l646_646615

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l646_646615


namespace trapezoid_isosceles_diagonal_l646_646100

structure Trapezoid :=
(a b c d : ℝ) -- sides of the trapezoid
(A B C D : ℝ) -- specific lengths

-- Defining the given trapezoid ABCD
def ABCD : Trapezoid :=
{ a := 25,
  b := 13,
  c := 15,
  d := 17,
  A := 25,
  B := 13,
  C := 15,
  D := 17 }

-- Helper definition for checking isosceles trapezoid
def is_isosceles (t : Trapezoid) : Prop := 
  t.a = t.b ∨ t.c = t.d

-- Computing the length of the diagonal AC given the trapezoid conditions
noncomputable def diagonal_AC (t : Trapezoid) : ℝ :=
  Real.sqrt ((5 + 13)^2 + (10*Real.sqrt 2)^2)

-- Main theorem statement
theorem trapezoid_isosceles_diagonal (t : Trapezoid) 
(h1 : t.A = 25) (h2 : t.B = 13) (h3 : t.C = 15) (h4 : t.D = 17)
: is_isosceles t ∧ diagonal_AC t = Real.sqrt 524 :=
begin
  sorry
end

end trapezoid_isosceles_diagonal_l646_646100


namespace compare_P_Q_l646_646822

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (1 - x)^(2*n - 1)
noncomputable def Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n - 1)*x + (n - 1)*(2*n - 1)*x^2

theorem compare_P_Q :
  ∀ (n : ℕ) (x : ℝ), n > 0 →
  ((n = 1 → P n x = Q n x) ∧
   (n = 2 → ((x = 0 → P n x = Q n x) ∧ (x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x))) ∧
   (n ≥ 3 → ((x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x)))) :=
by
  intros
  sorry

end compare_P_Q_l646_646822


namespace proof_solution_l646_646999

def proof_problem : Prop :=
  ∀ (s c p d : ℝ), 
  4 * s + 8 * c + p + 2 * d = 5.00 → 
  5 * s + 11 * c + p + 3 * d = 6.50 → 
  s + c + p + d = 1.50

theorem proof_solution : proof_problem :=
  sorry

end proof_solution_l646_646999


namespace group_friends_opponents_l646_646619

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l646_646619


namespace mouse_area_reachable_l646_646634

noncomputable def area_of_set_S (ω: ℝ) (r: ℝ → ℝ) : ℝ :=
  ∫ θ in (0 : ℝ) .. π, (1/2) * (r θ)^2

theorem mouse_area_reachable :
  let ω := π in
  let r := λ θ : ℝ, 1 - θ / π in
  area_of_set_S ω r = π / 6 :=
by
  sorry

end mouse_area_reachable_l646_646634


namespace gcd_90_450_l646_646699

theorem gcd_90_450 : Int.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l646_646699


namespace equilateral_triangle_area_l646_646841

theorem equilateral_triangle_area : 
  ∀ (A B C : ℝ × ℝ) (D E : ℝ × ℝ) (ℓ : set (ℝ × ℝ)),
  equilateral_triangle A B C →
  line_through_point A ℓ →
  orthogonal_projection B ℓ D →
  orthogonal_projection C ℓ E →
  distance D E = 1 →
  2 * distance B D = distance C E →
  ∃ m n : ℕ, area_equilateral_triangle A B C = m * real.sqrt n ∧ ¬ (∃ p: ℕ, nat.prime p ∧ p^2 ∣ n) ∧ m + n = 10 :=
sorry

end equilateral_triangle_area_l646_646841


namespace hyperbola_eccentricity_l646_646767

theorem hyperbola_eccentricity (a b c : ℝ) (h : c^2 = a^2 + b^2) 
  (asymptote_passes : ∃ (m : ℝ), (3 * b / a = m ∧ m * 3 = -4)) :
  c / a = 5 / 3 :=
by
  -- Using the condition 3b = 4a derived from the asymptote passing through (3, -4)
  have h1 : 3 * b = 4 * a := by
    cases asymptote_passes with m hm
    rw [←hm.1] at hm
    linarith

  -- Substitute b = 4a / 3 into the relation c^2 = a^2 + b^2
  have h2 : b = (4 / 3) * a := by
    linarith [h1]

  -- Substitute into the hyperbola property
  calc
    c^2 = a^2 + b^2           : h
    ... = a^2 + ((4 / 3) * a)^2 : by rw [h2]
    ... = a^2 + (16 / 9) * a^2 : by ring
    ... = (25 / 9) * a^2       : by ring
    ... = (5 / 3 * a)^2        : by ring

  -- Since c^2 = (5 / 3 * a)^2, we have c = 5 / 3 * a, hence c/a = 5/3
  have h3 : c = (5 / 3) * a := by
    exact sqrt_eq (by positivity) (25 / 9 * a^2).symm

  -- Therefore, the eccentricity c/a is 5/3
  linarith

end hyperbola_eccentricity_l646_646767


namespace solve_triangle_sides_l646_646799

noncomputable def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop := 
  a = 2 ∧ 2 * sin A = sin C ∧ cos (2 * C) = -1/4 ∧ 0 < C ∧ C < π/2 ∧ cos C = √6 / 4 ∧ c = 4 ∧ b = 2 * √6

theorem solve_triangle_sides :
  ∀ (a b c A B C : ℝ), triangle_sides a b c A B C →
  cos (2 * C) = -1/4 ∧ 0 < C ∧ C < π/2 →
  cos C = √6 / 4 ∧ a = 2 ∧ 2 * sin A = sin C →
  c = 4 ∧ b = 2 * √6 :=
by 
  intros a b c A B C h1 h2 h3
  sorry

end solve_triangle_sides_l646_646799


namespace imaginary_part_of_fraction_l646_646073

theorem imaginary_part_of_fraction : 
  let z := (2 * complex.I - 5) / (2 - complex.I) in 
  complex.imag_part z = -1 / 5 := 
by 
  sorry

end imaginary_part_of_fraction_l646_646073


namespace intersect_sets_l646_646282

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersect_sets : M ∩ N = {1, 2} :=
by
  sorry

end intersect_sets_l646_646282


namespace gcd_90_450_l646_646696

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l646_646696


namespace find_f_6_l646_646209

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f(x) = f(x - 2) + 3
axiom h2 : f(2) = 4

theorem find_f_6 : f(6) = 10 := sorry

end find_f_6_l646_646209


namespace validate_operations_l646_646532

variable (a : ℝ)

theorem validate_operations :
  (a + a^2 ≠ a^3) ∧
  (a * a^2 = a^3) ∧
  (a^6 / a^2 ≠ a^3) ∧
  ((a^(-1)) ^ 3 ≠ a^3) :=
by {
  -- Proofs would go here, but adding sorry to skip the actual steps
  sorry,
}

end validate_operations_l646_646532


namespace ratio_of_perimeters_l646_646180

noncomputable def perimeter (length : ℕ) (width : ℕ) : ℕ := 2 * (length + width)

theorem ratio_of_perimeters : 
  let side : ℕ := 8 in
  let large_perimeter : ℕ := perimeter side (side / 2) in
  let small_perimeter : ℕ := perimeter (side / 2) (side / 2) in
  (small_perimeter : ℚ) / (large_perimeter : ℚ) = 2 / 3 :=
by
  sorry

end ratio_of_perimeters_l646_646180


namespace group_of_friends_l646_646587

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l646_646587


namespace avg_k_for_polynomial_roots_l646_646299

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l646_646299


namespace cover_square_with_rectangles_l646_646135

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l646_646135


namespace ratio_A_to_B_investment_l646_646161

variable {I T : ℕ}  -- B's investment amount and investment period
variable {profit_B profit_total : ℕ}
variable (x : ℕ)  -- Multiple of B's investment that A invests

-- Conditions
axiom A_investment_eq : A_investment = x * I
axiom A_period_eq : A_period = 2 * T
axiom B_profit_eq : profit_B = 4000
axiom total_profit_eq : profit_total = 28000

-- Define B's investment and period
def B_investment := I
def B_period := T

-- Define A's profit calculation
def A_profit := profit_total - profit_B

-- Proof statement
theorem ratio_A_to_B_investment (h : A_profit = 24000) : x = 3 :=
by {
   -- Add proof steps here
   sorry
}

#check ratio_A_to_B_investment

end ratio_A_to_B_investment_l646_646161


namespace donuts_left_l646_646198

theorem donuts_left (t : ℕ) (c1 : ℕ) (c2 : ℕ) (c3 : ℝ) : t = 50 ∧ c1 = 2 ∧ c2 = 4 ∧ c3 = 0.5 
  → (t - c1 - c2) / 2 = 22 :=
by
  intros
  cases H with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5, h6]
  norm_num
  sorry

end donuts_left_l646_646198


namespace min_area_sum_le_half_l646_646009

variables {A B C D E F : Type}

-- Define points and conditions required for the problem
variable [add_comm_monoid A] [add_comm_monoid B] [add_comm_monoid C] [add_comm_monoid D] [add_comm_monoid E] [add_comm_monoid F]
variable (M : Set α) (β : Triangle α) (S : Triangle α → ℝ)

-- Define the given conditions
def convex_pentagon_area_one (ABCDE : Set α) (F : α) : Prop :=
  (is_convex_pentagon ABCDE ∧ area ABCDE = 1 ∧ is_point_inside_pentagon ABCDE F ∧ ¬is_on_any_diagonal ABCDE F)

-- Define the main theorem to be proven
theorem min_area_sum_le_half (ABCDE : Set α) (F : α) (h : convex_pentagon_area_one ABCDE F) :
  let M := {A, B, C, D, E, F}
  let a (M : Set α) := min {S(β1) + S(β2) | (β1, β2 : Triangle α) (hβ1 : (vertices β1) ⊆ M) (hβ2 : (vertices β2) ⊆ M) (disjoint (vertices β1) (vertices β2))}
  a (M) ≤ 1/2 :=
sorry

end min_area_sum_le_half_l646_646009


namespace ryan_office_population_l646_646864

noncomputable def ryan_office_size (M W : ℕ) (equal_men_women : M = W) (women_reduction : 6 = 0.20 * W) : ℕ :=
  M + W

theorem ryan_office_population (M W : ℕ) (equal_men_women : M = W) (women_reduction : 6 = 1 / 5 * W) :
  ryan_office_size M W equal_men_women women_reduction = 60 :=
sorry

end ryan_office_population_l646_646864


namespace infinite_solutions_of_linear_eq_l646_646902

theorem infinite_solutions_of_linear_eq (x y : ℝ) : ∃ (f : ℝ → ℝ), (∀ x, 2 * f(x) + x = 5) :=
by {
  -- Omitting the actual proof details
  sorry
}

end infinite_solutions_of_linear_eq_l646_646902


namespace smallest_possible_difference_after_101_years_l646_646673

theorem smallest_possible_difference_after_101_years {D E : ℤ} 
  (init_dollar : D = 6) 
  (init_euro : E = 7)
  (transformations : ∀ D E : ℤ, 
    (D', E') = (D + E, 2 * D + 1) ∨ (D', E') = (D + E, 2 * D - 1) ∨ 
    (D', E') = (D + E, 2 * E + 1) ∨ (D', E') = (D + E, 2 * E - 1)) :
  ∃ n_diff : ℤ, 101 = 2 * n_diff ∧ n_diff = 2 :=
sorry

end smallest_possible_difference_after_101_years_l646_646673


namespace domain_of_tan_function_l646_646884

theorem domain_of_tan_function :
  (∀ x : ℝ, ∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2 ↔ x ≠ (k * π) / 2 + 3 * π / 8) :=
sorry

end domain_of_tan_function_l646_646884


namespace total_football_games_l646_646804

theorem total_football_games (a b c : ℕ) (h1 : a = 11) (h2 : b = 17) (h3 : c = 16) : a + b + c = 44 := by
  rw [h1, h2, h3]
  exact Nat.add_assoc 11 17 16
  sorry

end total_football_games_l646_646804


namespace cost_of_first_book_l646_646038

-- Define the initial amount of money Shelby had.
def initial_amount : ℕ := 20

-- Define the cost of the second book.
def cost_of_second_book : ℕ := 4

-- Define the cost of one poster.
def cost_of_poster : ℕ := 4

-- Define the number of posters bought.
def num_posters : ℕ := 2

-- Define the total cost that Shelby had to spend on posters.
def total_cost_of_posters : ℕ := num_posters * cost_of_poster

-- Define the total amount spent on books and posters.
def total_spent (X : ℕ) : ℕ := X + cost_of_second_book + total_cost_of_posters

-- Prove that the cost of the first book is 8 dollars.
theorem cost_of_first_book (X : ℕ) (h : total_spent X = initial_amount) : X = 8 :=
by
  sorry

end cost_of_first_book_l646_646038


namespace geometric_sequence_product_l646_646715

variable {α : Type} [CommRing α]

theorem geometric_sequence_product (a : ℕ → α) (r : α)
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * r)
  (h_prod : a 10 * a 11 = 2) :
  (∏ i in Finset.range 20, a (i + 1)) = 1024 := 
sorry

end geometric_sequence_product_l646_646715


namespace minimum_value_of_expression_l646_646002

noncomputable theory

open Classical

-- Define the necessary variables and conditions
variables (n : ℕ) (a b : ℝ)

-- Define the main problem statement
theorem minimum_value_of_expression (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end minimum_value_of_expression_l646_646002


namespace pocket_knife_value_l646_646520

noncomputable def value_of_pocket_knife (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    let total_rubles := n * n
    let tens (x : ℕ) := x / 10
    let units (x : ℕ) := x % 10
    let e := units n
    let d := tens n
    let remaining := total_rubles - ((total_rubles / 10) * 10)
    if remaining = 6 then 4 else sorry

theorem pocket_knife_value (n : ℕ) : value_of_pocket_knife n = 2 := by
  sorry

end pocket_knife_value_l646_646520


namespace zero_if_sum_of_squares_eq_zero_l646_646860

theorem zero_if_sum_of_squares_eq_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_if_sum_of_squares_eq_zero_l646_646860


namespace quadratic_translation_l646_646756

theorem quadratic_translation :
  ∀ (x : ℝ),
    (∃ (a b : ℝ), (∀ y : ℝ, y = (2 * x - 1) * (x + 2) + 1 → y = 2 * (x + a) ^ 2 + b) ∧ a = -3/4 ∧ b = 17/8) →
    ∀ y₁ y₂ : ℝ, y₁ = (2 * x - 1) * (x + 2) + 1 ∧ y₂ = 2 * x ^ 2 → (y₁ = y₂ + 17/8) ∧ (y₁ = y₂ - 17/8)
   := 
begin
  sorry
end

end quadratic_translation_l646_646756


namespace min_x_eq_floor_sqrt_n_l646_646451

def floor (x : ℝ) : ℤ := Int.floor x

theorem min_x_eq_floor_sqrt_n (n : ℤ) (h_n : n ≥ 2) : 
  let x : ℕ → ℤ := λ i, match i with
                          | 0 => n
                          | i+1 => floor ((x i + y i) / 2)
  let y : ℕ → ℤ := λ i, match i with
                          | 0 => 1
                          | i+1 => floor (n / (x (i+1)))
  in ∃ i, x i = Int.floor (Real.sqrt n) :=
sorry

end min_x_eq_floor_sqrt_n_l646_646451


namespace least_number_added_to_250000_palindrome_l646_646528

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem least_number_added_to_250000_palindrome :
  ∃ n : ℕ, is_palindrome (250000 + n) ∧ ∀ m : ℕ, m < n → ¬ is_palindrome (250000 + m) ∧ n = 52 :=
by sorry

end least_number_added_to_250000_palindrome_l646_646528


namespace vertex_numbering_l646_646524

variable {V : Type} [Fintype V] [DecidableEq V]
variable {E : Type} [Fintype E]
variable (G : SimpleGraph V) (w : E → ℝ) (v : V)

-- Assumption: The graph is finite, simple, and non-directed
-- Assumption: Positive weights on edges such that the sum of weights at any vertex is less than 1
def valid_weighting (w : E → ℝ) (G : SimpleGraph V) : Prop :=
  ∀ v : V, (∑ (e : E) in G.incident_edges v, w e) < 1

variable (h_valid : valid_weighting w G)

-- The theorem to prove
theorem vertex_numbering :
  ∃ (x : V → ℝ), ∀ v, x v = 2022 + ∑ e in G.incident_edges v, (w e) * (x (G.other_vertex v e)) :=
sorry

end vertex_numbering_l646_646524


namespace avg_ticket_cost_per_person_l646_646537

-- Define the conditions
def full_price : ℤ := 150
def half_price : ℤ := full_price / 2
def num_full_price_tickets : ℤ := 2
def num_half_price_tickets : ℤ := 2
def free_tickets : ℤ := 1
def total_people : ℤ := 5

-- Prove that the average cost of tickets per person is 90 yuan
theorem avg_ticket_cost_per_person : ((num_full_price_tickets * full_price + num_half_price_tickets * half_price) / total_people) = 90 := 
by 
  sorry

end avg_ticket_cost_per_person_l646_646537


namespace constant_seq_decreasing_implication_range_of_values_l646_646507

noncomputable def sequences (a b : ℕ → ℝ) := 
  (∀ n, a (n+1) = (1/2) * a n + (1/2) * b n) ∧
  (∀ n, (1/b (n+1)) = (1/2) * (1/a n) + (1/2) * (1/b n))

theorem constant_seq (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) :
  ∃ c, ∀ n, a n * b n = c :=
sorry

theorem decreasing_implication (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) (h_dec : ∀ n, a (n+1) < a n) :
  a 1 > b 1 :=
sorry

theorem range_of_values (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 = 4) (h_b1 : b 1 = 1) :
  ∀ n ≥ 2, 2 < a n ∧ a n ≤ 5/2 :=
sorry

end constant_seq_decreasing_implication_range_of_values_l646_646507


namespace area_of_quadrilateral_l646_646693

-- Definitions of the given conditions
def diagonal_length : ℝ := 40
def offset1 : ℝ := 11
def offset2 : ℝ := 9

-- The area of the quadrilateral
def quadrilateral_area : ℝ := 400

-- Proof statement
theorem area_of_quadrilateral :
  (1/2 * diagonal_length * offset1 + 1/2 * diagonal_length * offset2) = quadrilateral_area :=
by sorry

end area_of_quadrilateral_l646_646693


namespace no_such_number_exists_l646_646028

theorem no_such_number_exists : ¬ ∃ n : ℕ, 10^(n+1) + 35 ≡ 0 [MOD 63] :=
by {
  sorry 
}

end no_such_number_exists_l646_646028


namespace circumcircle_circumference_thm_triangle_perimeter_thm_l646_646827

-- Definition and theorem for the circumference of the circumcircle
def circumcircle_circumference (a b c R : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * R = c / (Real.sqrt (1 - cosC^2)) 
  ∧ 2 * R * Real.pi = 3 * Real.pi

theorem circumcircle_circumference_thm (a b c R : ℝ) (cosC : ℝ) :
  circumcircle_circumference a b c R cosC → 2 * R * Real.pi = 3 * Real.pi :=
by
  intro h;
  sorry

-- Definition and theorem for the perimeter of the triangle
def triangle_perimeter (a b c : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * a = 3 * b ∧ (a + b + c) = 5 + Real.sqrt 5

theorem triangle_perimeter_thm (a b c : ℝ) (cosC : ℝ) :
  triangle_perimeter a b c cosC → (a + b + c) = 5 + Real.sqrt 5 :=
by
  intro h;
  sorry

end circumcircle_circumference_thm_triangle_perimeter_thm_l646_646827


namespace vertices_of_triangle_on_regular_18_gon_l646_646784

theorem vertices_of_triangle_on_regular_18_gon
  {K O M A : Point} (circumcenter : is_circumcenter K)
  (incenter : is_incenter O) (orthocenter : is_orthocenter M)
  (h : Circle K O M A) (arc_eq : ∡KO = ∡OM ∧ ∡OM = ∡MA) :
  ∃ P Q R : Point, is_vertex P ∧ is_vertex Q ∧ is_vertex R ∧ 
  (Triangle P Q R ∧ 
   (is_vertex_of_regular_18_gon P ∧ 
    is_vertex_of_regular_18_gon Q ∧ 
    is_vertex_of_regular_18_gon R)) := 
sorry

end vertices_of_triangle_on_regular_18_gon_l646_646784


namespace dice_probability_l646_646461

/-- 
Nathan rolls two six-sided dice. 
The first die results in an even number, which is one of {2, 4, 6}.
The second die results in a number less than or equal to three, which is one of {1, 2, 3}. 
Prove that the probability of these combined events is 1/4.
-/
def probability_of_combined_events : ℚ := 
  let p_even_first_die := 1 / 2
  let p_less_than_equal_three_second_die := 1 / 2
  p_even_first_die * p_less_than_equal_three_second_die

theorem dice_probability :
  probability_of_combined_events = 1 / 4 :=
by
  sorry

end dice_probability_l646_646461


namespace sum_of_series_S_sum_of_series_t_ratio_const_l646_646387

variables {a b c : ℝ} (n : ℕ) (t : ℝ)
variables (S S_1 S_2 S_3 ... S_{n-1} : ℝ) (t_1 t_2 t_3 ... t_{n-1} : ℝ)
variables (a_1 b_1 c_1 a_2 b_2 c_2 a_3 b_3 c_3 ... a_{n-1} b_{n-1} c_{n-1} : ℝ)

-- Condition: Sums of squares of the sides
def sum_of_squares : Prop := 
  S = a^2 + b^2 + c^2 ∧ 
  S_1 = a_1^2 + b_1^2 + c_1^2 ∧ 
  S_2 = a_2^2 + b_2^2 + c_2^2 ∧ 
  S_3 = a_3^2 + b_3^2 + c_3^2 ∧
  -- continue for all S_{n-1}

-- Given/Conditions
axiom sides_triangle : ∀ {a b c : ℝ}, a > 0 ∧ b > 0 ∧ c > 0
axiom area_triangle : ∀ {t : ℝ}, t > 0

-- Prove I) 
theorem sum_of_series_S : 
  sides_triangle a b c → 
  area_triangle t → 
  sum_of_squares S S_1 S_2 S_3 ... S_{n-1} →
  S + S_1 + S_2 + ... + S_{n-1} = n^3 * S := 
sorry 

-- Prove II)
theorem sum_of_series_t : 
  sides_triangle a b c → 
  area_triangle t → 
  sum_of_squares S S_1 S_2 S_3 ... S_{n-1} →
  t + t_1 + t_2 + ... + t_{n-1} = n^3 * t :=
sorry 

-- Prove III)
theorem ratio_const : 
  sides_triangle a b c → 
  area_triangle t → 
  sum_of_squares S S_1 S_2 S_3 ... S_{n-1} →
  (∀ {t_1 t_2 ... t_{n-1}}, (∃ k : ℝ, k = S / t ∧ k = S_1 / t_1 ∧ k = S_2 / t_2 ∧ ... ∧ k = S_{n-1} / t_{n-1})) :=
sorry 

end sum_of_series_S_sum_of_series_t_ratio_const_l646_646387


namespace transform_curve_eq_l646_646695

theorem transform_curve_eq : 
  ∀ (x y : ℝ), x * y = 1 → (let x' := (√2 / 2) * (x + y) in 
                           let y' := (√2 / 2) * (y - x) in 
                           x'^2 - y'^2 = 2) := 
by 
  intros x y Hxy
  let x' := (√2 / 2) * (x + y)
  let y' := (√2 / 2) * (y - x)
  sorry

end transform_curve_eq_l646_646695


namespace simplify_expression_l646_646040

theorem simplify_expression : (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by
  sorry

end simplify_expression_l646_646040


namespace triangle_KLM_equilateral_l646_646057

-- Definitions of triangle and its points
structure Triangle (α : Type) :=
(A B C : α)

-- Incenter and touch points definition
structure Incircle (T : Triangle ℝ) :=
(O : ℝ × ℝ)
(C1 A1 : ℝ × ℝ)
(touch_AB : ∃ r, dist (O) (C1) = r ∧ dist (O) (A1) = r)

-- Intersection points K, L and midpoint M
structure PointsAndConditions (T : Triangle ℝ) (I : Incircle T) :=
(K L M : ℝ × ℝ)
(CoAo_intersection : line_through T.C I.O T.A ∧ line_through T.C I.O T.B
  ∧ intersect (line_through I.C1 I.A1) (line_through T.C I.O) = some K
  ∧ intersect (line_through I.C1 I.A1) (line_through T.A I.O) = some L)
(midpoint_M : M = midpoint T.A T.C)

-- Angles and equilateral triangle condition
structure AnglesAndEquilateral (T : Triangle ℝ) (P : PointsAndConditions T) :=
(angle_ABC : ∠ T.A T.B T.C = 60)
(equilateral_KLM : equilateral (Triangle.mk P.K P.L P.M))

-- Main theorem statement
theorem triangle_KLM_equilateral (T : Triangle ℝ) (I : Incircle T) (P : PointsAndConditions T I) (A : AnglesAndEquilateral T P) :
  A.equilateral_KLM :=
by 
  sorry

end triangle_KLM_equilateral_l646_646057


namespace program_output_l646_646350

-- Assumptions based on the given problem
def initial_A : Int := -6
def initial_B : Int := 2

theorem program_output :
  let A1 := if initial_A < 0 then -initial_A else initial_A in
  let B1 := initial_B ^ 2 in
  let A2 := A1 + B1 in
  let C := A2 - 2 * B1 in
  let A_final := A2 / C in
  let B_final := B1 * C + 1 in
  A_final = 5 ∧ B_final = 9 ∧ C = 2 :=
by
  -- Proof omitted
  sorry

end program_output_l646_646350


namespace find_tan_F_l646_646801

theorem find_tan_F (D E F : ℝ) (h1 : D + E + F = π) (h2 : cot D * cot F = 1 / 3) (h3 : cot E * cot F = 1 / 27) :
  tan F = real.sqrt 51 :=
sorry

end find_tan_F_l646_646801


namespace exists_special_integer_l646_646815

theorem exists_special_integer (n : ℕ) (hn : n > 1) (k : ℕ) (hk : k = n.prime_divisors.length) :
  ∃ a : ℕ, 1 < a ∧ a < n / k + 1 ∧ n ∣ a^2 - a :=
by
  sorry

end exists_special_integer_l646_646815


namespace total_cost_for_seeds_l646_646578

theorem total_cost_for_seeds :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  in total = 18.00 :=
by
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  have h1 : total = 18.00,
  {
    sorry
  }
  exact h1

end total_cost_for_seeds_l646_646578


namespace coins_remainder_l646_646550

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l646_646550


namespace average_k_positive_int_roots_l646_646337

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l646_646337


namespace smallest_n_with_314_in_decimal_l646_646881

theorem smallest_n_with_314_in_decimal {m n : ℕ} (h_rel_prime : Nat.gcd m n = 1) (h_m_lt_n : m < n) 
  (h_contains_314 : ∃ k : ℕ, (10^k * m) % n == 314) : n = 315 :=
sorry

end smallest_n_with_314_in_decimal_l646_646881


namespace group_of_friends_l646_646586

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l646_646586


namespace max_f_value_f_monotonically_decreasing_f_range_l646_646456

-- Definitions and assumptions
def f (x : ℝ) : ℝ := 2 * cos x ^ 2 + sqrt 3 * sin (2 * x) - 1

-- ∀ x, part 1: show the maximum value of f(x)
theorem max_f_value :
  ∃ x, f x = 2 := sorry

-- ∀ k ∈ ℤ, part 2: show the interval of monotonic decrease
theorem f_monotonically_decreasing (k : ℤ) :
  ∀ x, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 → is_decreasing_on ℝ f 
:= sorry

-- part 3: show the range of f(x) on the given interval
theorem f_range :
  ∀ x, -π / 6 ≤ x ∧ x ≤ π / 3 → -1 ≤ f x ∧ f x ≤ 2 := sorry

end max_f_value_f_monotonically_decreasing_f_range_l646_646456


namespace identify_150th_digit_l646_646930

def repeating_sequence : List ℕ := [3, 8, 4, 6, 1, 5]

theorem identify_150th_digit :
  (150 % 6 = 0) →
  nth repeating_sequence 5 = 5 :=
by
  intros h
  rewrite_modulo h
  rfl

end identify_150th_digit_l646_646930


namespace tangent_line_eqn_l646_646174

theorem tangent_line_eqn :
  ∃ k : ℝ, 
  x^2 + y^2 - 4*x + 3 = 0 → 
  (∃ x y : ℝ, (x-2)^2 + y^2 = 1 ∧ x > 2 ∧ y < 0 ∧ y = k*x) → 
  k = - (Real.sqrt 3) / 3 := 
by
  sorry

end tangent_line_eqn_l646_646174


namespace average_k_l646_646303

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l646_646303


namespace folding_cranes_together_l646_646247

theorem folding_cranes_together (rateA rateB combined_time : ℝ)
  (hA : rateA = 1 / 30)
  (hB : rateB = 1 / 45)
  (combined_rate : ℝ := rateA + rateB)
  (h_combined_rate : combined_rate = 1 / combined_time):
  combined_time = 18 :=
by
  sorry

end folding_cranes_together_l646_646247


namespace reeya_fourth_subject_score_l646_646477

theorem reeya_fourth_subject_score
  (s1 s2 s3 : ℕ) (avg : ℕ) (n : ℕ)
  (h1 : s1 = 55) (h2 : s2 = 67) (h3 : s3 = 76)
  (h_avg : avg = 67) (h_n : n = 4) :
  let total_required_score := avg * n in
  let sum_first_three := s1 + s2 + s3 in
  let fourth_subject_score := total_required_score - sum_first_three in
  fourth_subject_score = 70 :=
by
  sorry

end reeya_fourth_subject_score_l646_646477


namespace find_line_equation_l646_646173

-- Define the circle
def circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 9

-- Define the line passing through point (1, 1)
def passes_through (l : ℝ → ℝ) : Prop := l 1 = 1

-- Define the condition that |AB| = 4
def distance_AB (l : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), 
    circle x1 y1 ∧ 
    circle x2 y2 ∧ 
    y1 = l x1 ∧ 
    y2 = l x2 ∧ 
    abs ( (x2 - x1)^2 + (y2 - y1)^2 ) = 16

-- Define the equation of the line l as x + 2y - 3 = 0
def line_equation (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the statement of the proof problem
theorem find_line_equation :
  ∃ (l : ℝ → ℝ), passes_through l ∧ distance_AB l → ∀ x y, line_equation x y :=
sorry

end find_line_equation_l646_646173


namespace find_f3_l646_646870

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, f(x) = 5 * (f ⁻¹' {x}).some + 8
axiom h2 : f 1 = 5
axiom linear_f : ∃ a b: ℝ, ∀ x : ℝ, f(x) = a * x + b

theorem find_f3 : f 3 = 2 * real.sqrt 5 + 5 := sorry

end find_f3_l646_646870


namespace determine_constants_l646_646218

theorem determine_constants : 
  ∃ (a b : ℚ), (a = 9 / 10 ∧ b = 9 / 10 ∧
  a • ℚ • (3, 4) + b • ℚ • (-3, 6) =  (0, 9)) := 
by 
  use 9/10 
  use 9/10
  split
  simp -- a = 9/10
  split
  simp -- b = 9/10
  sorry -- a • (3, 4) + b • (-3, 6) =  (0, 9)

end determine_constants_l646_646218


namespace geometric_progression_l646_646236

theorem geometric_progression (p : ℝ) 
  (a b c : ℝ)
  (h1 : a = p - 2)
  (h2 : b = 2 * Real.sqrt p)
  (h3 : c = -3 - p)
  (h4 : b ^ 2 = a * c) : 
  p = 1 := 
by 
  sorry

end geometric_progression_l646_646236


namespace proposition_two_correct_proposition_five_correct_l646_646710

-- Definitions for conditions and propositions
variables {a b : ℝ → ℝ → Prop} {α β γ : ℝ → ℝ → ℝ → Prop}

def line_perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := ∀ p q : ℝ × ℝ, l1 p ∧ l2 q → p ≠ q
def plane_perpendicular (p1 p2 : ℝ → ℝ → ℝ → Prop) : Prop := ∀ l1 l2 : ℝ → ℝ → Prop, (∀ x, p1 x → l1 x) ∧ (∀ y, p2 y → l2 y) → line_perpendicular l1 l2

def plane_contains_line (p : ℝ → ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop := ∀ x, l x → p x
def line_within_plane_perpendicular (l : ℝ → ℝ → Prop) (p : ℝ → ℝ → ℝ → Prop) : Prop := ∀ l2 : ℝ → ℝ → Prop, (∀ x, p x → l2 x) → line_perpendicular l l2

-- Proof Statements 
theorem proposition_two_correct (h1 : plane_contains_line α a)
  (h2 : line_within_plane_perpendicular a β) : plane_perpendicular α β := 
sorry

theorem proposition_five_correct (h1 : line_within_plane_perpendicular a α)
  (h2 : line_within_plane_perpendicular b β) : parallel α β := 
sorry

#lint -- checks for potential issues in the code, such as unused variables or undefined symbols

end proposition_two_correct_proposition_five_correct_l646_646710


namespace coleFenceCostCorrect_l646_646663

noncomputable def coleFenceCost : ℕ := 455

def woodenFenceCost : ℕ := 15 * 6
def woodenFenceNeighborContribution : ℕ := woodenFenceCost / 3
def coleWoodenFenceCost : ℕ := woodenFenceCost - woodenFenceNeighborContribution

def metalFenceCost : ℕ := 15 * 8
def coleMetalFenceCost : ℕ := metalFenceCost

def hedgeCost : ℕ := 30 * 10
def hedgeNeighborContribution : ℕ := hedgeCost / 2
def coleHedgeCost : ℕ := hedgeCost - hedgeNeighborContribution

def installationFee : ℕ := 75
def soilPreparationFee : ℕ := 50

def totalCost : ℕ := coleWoodenFenceCost + coleMetalFenceCost + coleHedgeCost + installationFee + soilPreparationFee

theorem coleFenceCostCorrect : totalCost = coleFenceCost := by
  -- Skipping the proof steps with sorry
  sorry

end coleFenceCostCorrect_l646_646663


namespace vertex_of_parabola_l646_646060

theorem vertex_of_parabola :
  ∀ (x : ℝ), (∃ y : ℝ, y = -(x - 2)^2 + 5) → (2, 5) ∈ set.range (λ x, -(x - 2)^2 + 5) :=
by
  sorry

end vertex_of_parabola_l646_646060


namespace parts_cost_correct_l646_646633

def cost_per_hour := 60
def hours_per_day := 8
def days_worked := 14
def total_cost := 9220

def labor_cost_per_day := cost_per_hour * hours_per_day
def total_labor_cost := labor_cost_per_day * days_worked

def parts_cost := total_cost - total_labor_cost

theorem parts_cost_correct : parts_cost = 2500 := by
  unfold parts_cost labor_cost_per_day total_labor_cost
  sorry

end parts_cost_correct_l646_646633


namespace smallest_num_rectangles_to_cover_square_l646_646117

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l646_646117


namespace cos_2x_value_l646_646257

theorem cos_2x_value (x : ℝ)
  (h₁ : x ∈ Ioo (π / 4) (π / 2))
  (h₂ : sin (π / 4 - x) = -3 / 5) :
  cos (2 * x) = -24 / 25 :=
by
  sorry

end cos_2x_value_l646_646257


namespace min_rectangles_to_cover_square_exactly_l646_646129

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l646_646129


namespace find_pairs_xy_l646_646687

theorem find_pairs_xy :
  ∃ L : list (ℕ × ℕ), 
    (∀ (x y : ℕ), (x > 0) → (y > 0) → 
    (let d := Nat.gcd x y in x * y * d = x + y + d^2) ↔ ((x, y) ∈ L))
    ∧ L = [(2, 2), (2, 3), (3, 2)] :=
by
  sorry

end find_pairs_xy_l646_646687


namespace decimal_150th_digit_frac_one_thirteen_l646_646918

theorem decimal_150th_digit_frac_one_thirteen :
  (∃ (d : ℕ), ∀ n, n ≥ 0 → d * 10^(n + 6) + d * 10^(n + 5) + d * 10^(n + 4) + d * 10^(n + 3) + d * 10^(n + 2) + d * 10^n = 10^6 - 1
  → (fractional_part (1 / 13) * (10 ^ (150 - 1)) % 10 = 3)) :=
by
  sorry

end decimal_150th_digit_frac_one_thirteen_l646_646918


namespace find_number_l646_646549

theorem find_number (x : ℝ) (h : 5 * 1.6 - (2 * 1.4) / x = 4) : x = 0.7 :=
by
  sorry

end find_number_l646_646549


namespace cos_F_in_triangle_l646_646422

theorem cos_F_in_triangle (D E F : ℝ) (sin_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) 
  (h1 : sin_D = 4 / 5) 
  (h2 : cos_E = 12 / 13) 
  (D_plus_E_plus_F : D + E + F = π) :
  cos_F = -16 / 65 :=
by
  sorry

end cos_F_in_triangle_l646_646422


namespace difference_between_20th_and_first_15_l646_646189

def grains_on_square (k : ℕ) : ℕ := 2^k

def total_grains_on_first_15_squares : ℕ :=
  (Finset.range 15).sum (λ k => grains_on_square (k + 1))

def grains_on_20th_square : ℕ := grains_on_square 20

theorem difference_between_20th_and_first_15 :
  grains_on_20th_square - total_grains_on_first_15_squares = 983042 :=
by
  sorry

end difference_between_20th_and_first_15_l646_646189


namespace part_I_solution_part_II_solution_l646_646742

-- Definition of the function f(x)
def f (x a : ℝ) := |x - a| + |2 * x - 1|

-- Part (I) when a = 1, find the solution set for f(x) ≤ 2
theorem part_I_solution (x : ℝ) : f x 1 ≤ 2 ↔ 0 ≤ x ∧ x ≤ 4 / 3 :=
by sorry

-- Part (II) if the solution set for f(x) ≤ |2x + 1| contains [1/2, 1], find the range of a
theorem part_II_solution (a : ℝ) :
  (∀ x : ℝ, 1 / 2 ≤ x ∧ x ≤ 1 → f x a ≤ |2 * x + 1|) → -1 ≤ a ∧ a ≤ 5 / 2 :=
by sorry

end part_I_solution_part_II_solution_l646_646742


namespace club_co_presidents_l646_646570

noncomputable def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem club_co_presidents : comb 18 3 = 816 := by
  unfold comb
  rw [Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_zero]
  norm_num
  sorry

end club_co_presidents_l646_646570


namespace number_of_boys_is_50_l646_646978

-- Definitions for conditions:
def total_students : Nat := 100
def boys (x : Nat) : Nat := x
def girls (x : Nat) : Nat := x

-- Theorem statement:
theorem number_of_boys_is_50 (x : Nat) (g : Nat) (h1 : x + g = total_students) (h2 : g = boys x) : boys x = 50 :=
by
  sorry

end number_of_boys_is_50_l646_646978


namespace log_seq_value_l646_646505

variable {a : ℕ → ℝ}

-- Define the conditions
axiom seq_prop : ∀ n : ℕ, n > 0 → (a (n + 1) = 3 + a n)
axiom seq_sum_prop : a 2 + a 4 + a 6 = 9

-- State the theorem to prove
theorem log_seq_value : log (1 / 6) (a 5 + a 7 + a 9) = -2 := by
  sorry

end log_seq_value_l646_646505


namespace part1_part2_part3_l646_646347

def a (n : ℕ) : ℤ := 13 - 2 * n

theorem part1 : |a 1| + |a 2| + |a 3| = 27 := 
by {
  sorry
}

theorem part2 : |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 52 :=
by {
  sorry
}

theorem part3 (n : ℕ) : 
  if h1 : 1 ≤ n ∧ n ≤ 6 then |a 1| + |a 2| + ... + |a n| = 12 * n - n^2
  else if h2 : n ≥ 7 then |a 1| + |a 2| + ... + |a n| = n^2 - 12 * n + 72 :=
by {
  sorry
}

end part1_part2_part3_l646_646347


namespace david_investment_years_l646_646469

theorem david_investment_years (P : ℝ) (A_peter : ℝ) (A_david : ℝ) (r : ℝ) (t : ℝ) :
  P = 710 →
  A_peter = 815 →
  A_david = 850 →
  A_peter = P + P * r * 3 →
  r = 0.05 →
  A_david = P + P * r * t →
  t = 4 :=
by
  intros hP hA_peter hA_david hPeter_formula hr hDavid_formula
  have r_value : r = 0.05 := hr
  have Peter_eval : 815 = 710 + 710 * r * 3 := by exact hPeter_formula
  have David_eval : 850 = 710 + 710 * r * t := by exact hDavid_formula
  rw [←Peter_eval, ←David_eval] at *
  sorry

end david_investment_years_l646_646469


namespace female_managers_count_l646_646396

def total_employees : ℕ := sorry
def female_employees : ℕ := 700
def managers : ℕ := (2 * total_employees) / 5
def male_employees : ℕ := total_employees - female_employees
def male_managers : ℕ := (2 * male_employees) / 5

theorem female_managers_count :
  ∃ (fm : ℕ), managers = fm + male_managers ∧ fm = 280 := by
  sorry

end female_managers_count_l646_646396


namespace gcd_90_450_l646_646698

theorem gcd_90_450 : Int.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l646_646698


namespace inequality_solution_is_correct_l646_646043

noncomputable def problem_ineq_solution (x : ℝ) : Prop :=
  (x / (x - 5) ≥ 0) ↔ (x ∈ set.Iic 0 ∪ set.Ioi 5)

theorem inequality_solution_is_correct : ∀ x : ℝ, problem_ineq_solution x :=
by
  intro x
  unfold problem_ineq_solution
  split; intros h; apply sorry

end inequality_solution_is_correct_l646_646043


namespace minimum_sum_of_ten_numbers_l646_646091

theorem minimum_sum_of_ten_numbers (a : Fin 10 → ℕ) (distinct : Function.Injective a)
    (H1 : ∃ (S ⊆ Finset.univ : Finset (Fin 10)), S.card = 5 ∧ (S.card = 5 → ∃ i ∈ S, a i % 2 = 0))
    (H2 : ∑ i, a i % 2 = 1) :
    ∑ i, a i ≥ 51 :=
by
  sorry

end minimum_sum_of_ten_numbers_l646_646091


namespace hannah_mugs_problem_l646_646369

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end hannah_mugs_problem_l646_646369


namespace plane_equation_l646_646291

theorem plane_equation
  (A B C D : ℤ)
  (hA : A > 0)
  (h_gcd : Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1)
  (h_point : (A * 4 + B * (-4) + C * 5 + D = 0)) :
  A = 4 ∧ B = -4 ∧ C = 5 ∧ D = -57 :=
  sorry

end plane_equation_l646_646291


namespace quadratic_has_min_value_when_x_is_positive_l646_646213

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 5

theorem quadratic_has_min_value_when_x_is_positive :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), f(x) ≤ f(y) :=
sorry

end quadratic_has_min_value_when_x_is_positive_l646_646213


namespace avg_of_k_with_positive_integer_roots_l646_646328

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l646_646328


namespace find_side_length_l646_646418

noncomputable def cos (x : ℝ) := Real.cos x

theorem find_side_length
  (A : ℝ) (c : ℝ) (b : ℝ) (a : ℝ)
  (hA : A = Real.pi / 3)
  (hc : c = Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 3) :
  a = 3 := 
sorry

end find_side_length_l646_646418


namespace positive_difference_solutions_abs_eq_30_l646_646940

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l646_646940


namespace initial_books_correct_l646_646852

def sold_books : ℕ := 78
def left_books : ℕ := 37
def initial_books : ℕ := sold_books + left_books

theorem initial_books_correct : initial_books = 115 := by
  sorry

end initial_books_correct_l646_646852


namespace number_of_herrings_l646_646683

theorem number_of_herrings (total_fishes pikes sturgeons herrings : ℕ)
  (h1 : total_fishes = 145)
  (h2 : pikes = 30)
  (h3 : sturgeons = 40)
  (h4 : total_fishes = pikes + sturgeons + herrings) :
  herrings = 75 :=
by
  sorry

end number_of_herrings_l646_646683


namespace power_equality_l646_646759

theorem power_equality (p : ℕ) : 16^10 = 4^p → p = 20 :=
by
  intro h
  -- proof goes here
  sorry

end power_equality_l646_646759


namespace positive_difference_solutions_abs_eq_30_l646_646941

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l646_646941


namespace find_eighth_number_l646_646873

-- Define the given problem with the conditions
noncomputable def sum_of_sixteen_numbers := 16 * 55
noncomputable def sum_of_first_eight_numbers := 8 * 60
noncomputable def sum_of_last_eight_numbers := 8 * 45
noncomputable def sum_of_last_nine_numbers := 9 * 50
noncomputable def sum_of_first_ten_numbers := 10 * 62

-- Define what we want to prove
theorem find_eighth_number :
  (exists (x : ℕ), x = 90) →
  sum_of_first_eight_numbers = 480 →
  sum_of_last_eight_numbers = 360 →
  sum_of_last_nine_numbers = 450 →
  sum_of_first_ten_numbers = 620 →
  sum_of_sixteen_numbers = 880 →
  x = 90 :=
by sorry

end find_eighth_number_l646_646873


namespace seating_arrangement_l646_646407

theorem seating_arrangement (n : ℕ) (h : n = 7) : 
  let people : Fin n := Fin.ofNat 7 
  let alice : Fin n := 0
  let bob : Fin people.succ := 1
  2 * (Fin.factorial (n - 1)) = 240 :=
begin
  sorry
end

end seating_arrangement_l646_646407


namespace local_min_f_range_b_l646_646740

-- Part 1: Proving the local minimum value of f(x) for b = -4
theorem local_min_f (x : ℝ) (h_pos : 0 < x) : 
  let f := λ x, 3 * x - 1 / x - 4 * log x in 
  (f 1 = 2) ∧ 
  ∃ c, c ∈ set.Ioo 0.33 1 ∧ is_local_min_on f (set.Icc 0.33 1) 1 := 
sorry

-- Part 2: Finding the range of values for b
theorem range_b (x : ℝ) (h_in : x ∈ set.Icc 1 real.exp 1) :
  ∃ b, (4 * x - 1 / x - (3 * x - 1 / x + b * log x) < - (1 + b) / x) ↔ 
  (b < -2) ∨ (b > (real.exp 2 + 1) / (real.exp 1 - 1)) := 
sorry

end local_min_f_range_b_l646_646740


namespace range_of_m_l646_646385

noncomputable def f (x m : ℝ) : ℝ := x^2 - x + m * (2 * x + 1)

theorem range_of_m (m : ℝ) : (∀ x > 1, 0 < 2 * x + (2 * m - 1)) ↔ (m ≥ -1/2) := by
  sorry

end range_of_m_l646_646385


namespace at_least_30_percent_have_all_three_colors_l646_646989

theorem at_least_30_percent_have_all_three_colors
  (C : ℕ) -- number of children
  (even_flags : ∃ n, n % 2 = 0) -- an even number of flags in the box
  (picks_three : ∀ c, c ∈ (finset.range C), 3 * c) -- each child picks exactly three flags
  (blue_flags : ∀ c, c ∈ (finset.range C), c = 0.55 * C)
  (red_flags : ∀ c, c ∈ (finset.range C), c = 0.45 * C)
  (green_flags : ∀ c, c ∈ (finset.range C), c = 0.30 * C) :
  ∃ x, (x / C ≥ 0.30) ∧ (x <= C) :=
sorry

end at_least_30_percent_have_all_three_colors_l646_646989


namespace games_bought_l646_646022

def initial_money : ℕ := 35
def spent_money : ℕ := 7
def cost_per_game : ℕ := 4

theorem games_bought : (initial_money - spent_money) / cost_per_game = 7 := by
  sorry

end games_bought_l646_646022


namespace jenna_siblings_product_l646_646399

theorem jenna_siblings_product :
  ∀ (jamie_sisters jamie_brothers : ℕ),
    jamie_sisters = 4 →
    jamie_brothers = 6 →
    let S := jamie_sisters - 1 in
    let B := jamie_brothers in
    S * B = 24 :=
by
  intros jamie_sisters jamie_brothers h1 h2
  dsimp
  rw [h1, h2]
  norm_num

end jenna_siblings_product_l646_646399


namespace enclosed_area_l646_646107

theorem enclosed_area (x y : ℝ) (h : x^2 + y^2 = 2 * (|x| + |y|)) : 
  let region := {z : ℝ × ℝ | (z.1^2 + z.2^2 = 2 * (|z.1| + |z.2|))} in
  (measure_theory.measure.restrict measure_theory.measure_space.volume region).measure_univ = 2 * real.pi :=
sorry

end enclosed_area_l646_646107


namespace complex_magnitude_difference_eq_one_l646_646754

noncomputable def magnitude (z : Complex) : ℝ := Complex.abs z

/-- Lean 4 statement of the problem -/
theorem complex_magnitude_difference_eq_one (z₁ z₂ : Complex) (h₁ : magnitude z₁ = 1) (h₂ : magnitude z₂ = 1) (h₃ : magnitude (z₁ + z₂) = Real.sqrt 3) : magnitude (z₁ - z₂) = 1 := 
sorry

end complex_magnitude_difference_eq_one_l646_646754


namespace basil_has_winning_strategy_l646_646464

-- Definitions based on conditions
def piles : Nat := 11
def stones_per_pile : Nat := 10
def peter_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3
def basil_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3

-- The main theorem to prove Basil has a winning strategy
theorem basil_has_winning_strategy 
  (total_stones : Nat := piles * stones_per_pile) 
  (peter_first : Bool := true) :
  exists winning_strategy_for_basil, 
    ∀ (piles_remaining : Nat) (sum_stones_remaining : Nat),
    sum_stones_remaining = piles_remaining * stones_per_pile ∨
    (1 ≤ piles_remaining ∧ piles_remaining ≤ piles) ∧
    (0 ≤ sum_stones_remaining ∧ sum_stones_remaining ≤ total_stones)
    → winning_strategy_for_basil = true := 
sorry -- The proof is omitted

end basil_has_winning_strategy_l646_646464


namespace pants_price_l646_646434

theorem pants_price (P : ℝ) 
  (h_shirts : 5 = 5 * 1) 
  (h_pants_earning : 5 * P) 
  (h_half : (1 / 2) * (5 + 5 * P) = 10) 
  : P = 3 :=
sorry

end pants_price_l646_646434


namespace orbitals_with_electrons_l646_646371

theorem orbitals_with_electrons (Z : ℕ) (config : Nat → ℕ) 
  (h : Z = 26) (h_config : config 1 = 2 ∧ config 2 = 2 ∧ config 3 = 6 ∧ config 4 = 2 ∧ 
                        config 5 = 6 ∧ config 6 = 6 ∧ config 7 = 2) : 
  ∑ i in {1, 2, 3, 4, 5, 6, 7}, if config i > 0 then 1 else 0 = 15 := 
by
  -- Configuration for orbitals: 1s, 2s, 2p, 3s, 3p, 3d, 4s
  have h1 : config 1 > 0 := sorry,
  have h2 : config 2 > 0 := sorry,
  have h3 : config 3 > 0 := sorry,
  have h4 : config 4 > 0 := sorry,
  have h5 : config 5 > 0 := sorry,
  have h6 : config 6 > 0 := sorry,
  have h7 : config 7 > 0 := sorry,
  sorry -- actual proof steps are omitted

end orbitals_with_electrons_l646_646371


namespace total_cost_correct_l646_646576

-- Define the individual costs and quantities
def pumpkin_cost : ℝ := 2.50
def tomato_cost : ℝ := 1.50
def chili_pepper_cost : ℝ := 0.90

def pumpkin_quantity : ℕ := 3
def tomato_quantity : ℕ := 4
def chili_pepper_quantity : ℕ := 5

-- Define the total cost calculation
def total_cost : ℝ :=
  pumpkin_quantity * pumpkin_cost +
  tomato_quantity * tomato_cost +
  chili_pepper_quantity * chili_pepper_cost

-- Prove the total cost is $18.00
theorem total_cost_correct : total_cost = 18.00 := by
  sorry

end total_cost_correct_l646_646576


namespace friends_game_l646_646610

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l646_646610


namespace sequence_nth_term_l646_646360

theorem sequence_nth_term (n : ℕ) : 
  ∀ a : ℕ → ℤ, (a 1 = 1) ∧ (∀ n, a (n + 1) = a n + 2) → a n = (2 * n) - 1 :=
by
  intro a
  intro h
  cases h with a1 h_rec
  sorry

end sequence_nth_term_l646_646360


namespace possible_number_of_friends_l646_646599

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l646_646599


namespace sheets_per_package_l646_646807

variable (x : ℕ)

-- Definition of the conditions
def two_boxes_contain_five_packages_each : Prop := (2 * 5 = 10)
def each_package_contains_x_sheets : Prop := (x > 0)
def each_issue_uses_25_sheets : Prop := (25 > 0)
def julie_can_print_100_newspapers : Prop := (100 * 25 = 2500)

theorem sheets_per_package 
  (H1 : two_boxes_contain_five_packages_each)
  (H2 : each_package_contains_x_sheets)
  (H3 : each_issue_uses_25_sheets)
  (H4 : julie_can_print_100_newspapers) :
  10 * x = 2500 → x = 250 := 
by
  intro h1
  constr
  sorry

end sheets_per_package_l646_646807


namespace joanna_needs_more_hours_to_finish_book_l646_646093
-- Import the necessary library

-- Define the problem conditions and prove the final answer

theorem joanna_needs_more_hours_to_finish_book :
  let total_pages := 248
  let pages_per_hour := 16
  let hours_monday := 3
  let hours_tuesday := 6.5
  let pages_read_monday := hours_monday * pages_per_hour
  let pages_read_tuesday := hours_tuesday * pages_per_hour
  let total_pages_read := pages_read_monday + pages_read_tuesday
  let pages_left := total_pages - total_pages_read
  let hours_needed := pages_left / pages_per_hour
  in hours_needed = 6 :=
by
  sorry

end joanna_needs_more_hours_to_finish_book_l646_646093


namespace ratio_of_distances_l646_646659

namespace CarDistanceRatio

-- Define the conditions
def speed_A := 80  -- Speed of Car A in km/hr
def time_A := 5    -- Time taken by Car A in hours
def speed_B := 100 -- Speed of Car B in km/hr
def time_B := 2    -- Time taken by Car B in hours

-- Define the distances based on the conditions
def distance_A := speed_A * time_A
def distance_B := speed_B * time_B

-- Define the statement of the problem
theorem ratio_of_distances : (distance_A : distance_B) = (2 : 1) := by
  sorry

end CarDistanceRatio

end ratio_of_distances_l646_646659


namespace donuts_left_l646_646201

def initial_donuts : ℕ := 50
def after_bill_eats (initial : ℕ) : ℕ := initial - 2
def after_secretary_takes (remaining_after_bill : ℕ) : ℕ := remaining_after_bill - 4
def coworkers_take (remaining_after_secretary : ℕ) : ℕ := remaining_after_secretary / 2
def final_donuts (initial : ℕ) : ℕ :=
  let remaining_after_bill := after_bill_eats initial
  let remaining_after_secretary := after_secretary_takes remaining_after_bill
  remaining_after_secretary - coworkers_take remaining_after_secretary

theorem donuts_left : final_donuts 50 = 22 := by
  sorry

end donuts_left_l646_646201


namespace ramola_rank_last_is_14_l646_646149

-- Define the total number of students
def total_students : ℕ := 26

-- Define Ramola's rank from the start
def ramola_rank_start : ℕ := 14

-- Define a function to calculate the rank from the last given the above conditions
def ramola_rank_from_last (total_students ramola_rank_start : ℕ) : ℕ :=
  total_students - ramola_rank_start + 1

-- Theorem stating that Ramola's rank from the last is 14th
theorem ramola_rank_last_is_14 :
  ramola_rank_from_last total_students ramola_rank_start = 14 :=
by
  -- Proof goes here
  sorry

end ramola_rank_last_is_14_l646_646149


namespace find_initial_music_files_l646_646105

-- Define the initial state before any deletion
def initial_files (music_files : ℕ) (video_files : ℕ) : ℕ := music_files + video_files

-- Define the state after deleting files
def files_after_deletion (initial_files : ℕ) (deleted_files : ℕ) : ℕ := initial_files - deleted_files

-- Theorem to prove that the initial number of music files was 13
theorem find_initial_music_files 
  (video_files : ℕ) (deleted_files : ℕ) (remaining_files : ℕ) 
  (h_videos : video_files = 30) (h_deleted : deleted_files = 10) (h_remaining : remaining_files = 33) : 
  ∃ (music_files : ℕ), initial_files music_files video_files - deleted_files = remaining_files ∧ music_files = 13 :=
by {
  sorry
}

end find_initial_music_files_l646_646105


namespace abs_sqrt17_pure_imaginary_l646_646261

theorem abs_sqrt17_pure_imaginary (b : ℝ) (h : (2 + b * complex.I) * (2 - complex.I) = (2 * b - 2) * complex.I) : complex.abs (1 + b * complex.I) = real.sqrt 17 := by
  -- Proof goes here
  sorry

end abs_sqrt17_pure_imaginary_l646_646261


namespace max_unique_three_digit_numbers_l646_646709

-- In Lean code, we translate the conditions and the requirement to prove the maximum number
-- given the constraints.
theorem max_unique_three_digit_numbers : 
  ∃ (num_sets : Finset (Finset (Fin 5))), 
    (∀ (x y : Finset (Fin 5)), x ≠ y → swap_adjacent_free x y) ∧
    num_sets.card = 75 :=
  sorry

end max_unique_three_digit_numbers_l646_646709


namespace right_trapezoid_of_similar_and_opposite_orientation_l646_646062

noncomputable def deltoid (A B C D : Type*) [affine_space A B C D] : Prop :=
(B - A = C - D) ∧ (A - B = D - C)

noncomputable def similar {A₁ B₁ C₁ D₁ A₂ B₂ C₂ D₂ : Type*}
  [affine_space A₁ B₁ C₁ D₁] [affine_space A₂ B₂ C₂ D₂] : Prop :=
(exists (f : → A₂), f A₁ = A₂ ∧ f B₁ = B₂ ∧ f C₁ = C₂ ∧ f D₁ = D₂ ∧ is_similar_transform f)

noncomputable def opposite_orientation (A B C D E F : Type*) [affine_space A B C]
  [affine_space D E F] : Prop :=
    (orientation B - A = orientation E - D) ∧ (orientation C - B = orientation F - E)

theorem right_trapezoid_of_similar_and_opposite_orientation
  (A B C D E F : Type*) [affine_space A B C D] [affine_space C D E F]
  (h1 : deltoid A B C D) (h2 : deltoid C D E F)
  (h3 : similar A B C D C D E F)
  (h4 : opposite_orientation A B C D E F) : 
  is_right_trapezoid A B E F :=
sorry

end right_trapezoid_of_similar_and_opposite_orientation_l646_646062


namespace improper_fraction_2012a_div_b_l646_646144

theorem improper_fraction_2012a_div_b
  (a b : ℕ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) :
  2012 * a > b :=
by 
  sorry

end improper_fraction_2012a_div_b_l646_646144


namespace range_of_k_condition_l646_646466

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := (4 - k) / x

theorem range_of_k_condition (k x1 x2 y1 y2 : ℝ) 
    (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 < y2) 
    (hA : inverse_proportion_function k x1 = y1) 
    (hB : inverse_proportion_function k x2 = y2) : 
    k < 4 :=
sorry

end range_of_k_condition_l646_646466


namespace angle_AMD_is_45_deg_l646_646862

theorem angle_AMD_is_45_deg :
  ∀ (A B C D M : Point) (AB BC : ℝ),
    Rectangle A B C D →
    A.1 < B.1 → A.2 = B.2 →
    B.1 = C.1 → B.2 < C.2 →
    A.1 = D.1 → A.2 < D.2 →
    B.1 - A.1 = 8 →
    C.2 - B.2 = 4 →
    M.1 > A.1 → M.1 < B.1 → M.2 = A.2 →
    ∠ AM D = ∠ CM D →
    ∠ AM D = 45 :=
by
  intros A B C D M AB BC hRect hAB hBC hBD hCD hDA hABlen hBClen hA1 hunB1 hunA2 hangleEquiv
  sorry

end angle_AMD_is_45_deg_l646_646862


namespace inequality_holds_l646_646281

-- Given real numbers a and b that satisfy |a + b| ≤ 2
variables (a b : ℝ)
hypothesis h : |a + b| ≤ 2

-- Prove that |a^2 + 2a - b^2 + 2b| ≤ 4 (|a| + 2)
theorem inequality_holds (a b : ℝ) (h : |a + b| ≤ 2) : |a^2 + 2a - b^2 + 2b| ≤ 4 (|a| + 2) :=
sorry

end inequality_holds_l646_646281


namespace angle_ACB_is_80_l646_646266

/-- Given a convex quadrilateral ABCD, such that ∠ BAC = 20°, ∠ CAD = 60°, ∠ ADB = 50°, and 
∠ BDC = 10°, prove that ∠ ACB = 80°. -/
theorem angle_ACB_is_80
  (A B C D : Type)
  [IsConvexQuadrilateral A B C D]
  (angle_BAC : ℝ)
  (angle_CAD : ℝ)
  (angle_ADB : ℝ)
  (angle_BDC : ℝ)
  (h1 : angle_BAC = 20)
  (h2 : angle_CAD = 60)
  (h3 : angle_ADB = 50)
  (h4 : angle_BDC = 10) :
  ∠ ACB = 80 :=
sorry

end angle_ACB_is_80_l646_646266


namespace value_of_k_l646_646264

noncomputable def f (x : ℝ) : ℝ :=
  let abs := Real.abs in
  Σ i in (Finset.range 2018 ∪ Finset.range (-2017)), abs (x + i - 2017)

theorem value_of_k (a : ℕ) (g : ℝ → ℝ) (k : ℝ) (m : ℝ) (n : ℕ) :
  (f (a^2 - 3 * a + 2) = f (a - 1)) → 
  ((g x = (x^2 * (x^2 + k^2 + 2 * k - 4) + 4) / ((x^2 + 2)^2 - 2 * x^2)) ∧ 
  ∀ x, g x ≥ m) → 
  (m + n = 3) → 
  n = 2 → 
  m = 1 → 
  (k = -1 + Real.sqrt 7 ∨ k = -1 - Real.sqrt 7) :=
by
  sorry

end value_of_k_l646_646264


namespace compare_P_Q_l646_646823

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (1 - x)^(2*n - 1)
noncomputable def Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n - 1)*x + (n - 1)*(2*n - 1)*x^2

theorem compare_P_Q :
  ∀ (n : ℕ) (x : ℝ), n > 0 →
  ((n = 1 → P n x = Q n x) ∧
   (n = 2 → ((x = 0 → P n x = Q n x) ∧ (x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x))) ∧
   (n ≥ 3 → ((x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x)))) :=
by
  intros
  sorry

end compare_P_Q_l646_646823


namespace domain_of_v_l646_646936

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt (2 * x - 4))

theorem domain_of_v :
  {x : ℝ | 2 * x - 4 > 0} = {x : ℝ | x > 2} :=
by
  ext x
  unfold v
  simp
  sorry

end domain_of_v_l646_646936


namespace angle_C_proof_l646_646800

variables {A B C : Real} (a b c : ℝ)
axiom condition1 : ∀ (A B C : Real), a = A ∧ b = B ∧ c = C

noncomputable def angle_C (a b c : ℝ) : ℝ :=
  if 2 * c * cos B = 2 * a + b then
    2 * ℝ.pi / 3
  else
    0

theorem angle_C_proof : angle_C a b c = 2 * ℝ.pi / 3 :=
begin
  sorry
end

end angle_C_proof_l646_646800


namespace anna_stops_700th_draw_l646_646095

theorem anna_stops_700th_draw (marbles : Finset (Fin 800)) 
  (colors : Fin 100 → Finset marbles) 
  (h1 : ∀ c, (colors c).card = 8)
  (h2 : ∑ c, (colors c).erase 699.card = 101)
  (h3 : ∑ c, if c.erase 699.card = 1 then 1 else 0 = 99)
  (h4 : ∑ c, if c.erase 699.card = 2 then 1 else 0 = 1) :
  probability (colors (erase 699)).erase 700.card = \frac{99}{101}) :=
sorry

end anna_stops_700th_draw_l646_646095


namespace calc_g_f_3_l646_646832

def f (x : ℕ) : ℕ := x^3 + 3

def g (x : ℕ) : ℕ := 2 * x^2 + 3 * x + 2

theorem calc_g_f_3 : g (f 3) = 1892 := by
  sorry

end calc_g_f_3_l646_646832


namespace determinant_of_exp_matrix_is_zero_l646_646447

open Matrix

noncomputable def determinant_of_exp_matrix (A B C : ℝ) : ℝ :=
  det ![
    #[Real.exp A, Real.exp (-A), 1],
    #[Real.exp B, Real.exp (-B), 1],
    #[Real.exp C, Real.exp (-C), 1]
  ]

theorem determinant_of_exp_matrix_is_zero (A B C : ℝ) (h : A + B + C = Real.pi) : 
  determinant_of_exp_matrix A B C = 0 :=
  sorry

end determinant_of_exp_matrix_is_zero_l646_646447


namespace frequency_of_group_l646_646639

-- Defining the conditions
def sample_size : ℕ := 100
def num_groups : ℕ := 10
def class_interval : ℕ := 10
def height_of_rectangle : ℝ := 0.03

-- Defining the theorem
theorem frequency_of_group : 
  let relative_frequency := height_of_rectangle * (class_interval : ℝ) in
  let frequency := relative_frequency * (sample_size : ℝ) in
  frequency = 30 := 
by
  -- Proof will be provided here
  sorry

end frequency_of_group_l646_646639


namespace average_of_k_l646_646320

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l646_646320


namespace roots_of_pure_imaginary_k_l646_646671

namespace ProofProblem

def discriminant (a b c : ℂ) : ℂ := b^2 - 4 * a * c

def root1 (a b c : ℂ) : ℂ := (-b + complex.sqrt (discriminant a b c)) / (2 * a)
def root2 (a b c : ℂ) : ℂ := (-b - complex.sqrt (discriminant a b c)) / (2 * a)

theorem roots_of_pure_imaginary_k (k : ℂ) (hk : k.im ≠ 0 ∧ k.re = 0) :
  (root1 10 (-5) (-k)).im = 0 ∧ (root2 10 (-5) (-k)).re = 0 ∨
  (root1 10 (-5) (-k)).re = 0 ∧ (root2 10 (-5) (-k)).im = 0 :=
sorry

end ProofProblem

end roots_of_pure_imaginary_k_l646_646671


namespace digit_150_l646_646931

def decimal_rep : ℚ := 5 / 13

def cycle_length : ℕ := 6

theorem digit_150 (n : ℕ) (h : n = 150) : Nat.digit (n % cycle_length) (decimal_rep) = 5 := by
  sorry

end digit_150_l646_646931


namespace no_sequences_periodic_l646_646985

-- Definition of sequence A
def seqA : ℕ → ℤ
| 0 := 1
| 1 := 1
| 2 := 0
| 3 := 1
| n := if h : ∃ k, n = k + (k + 1) then 0 else 0

-- Definition of sequence B
def seqB : ℕ → ℤ
| 0 := 1
| 1 := 2
| 2 := 1
| 3 := 2
| n := if h : ∃ k, n = k*2 + k then 3 else 1

-- Definition of sequence C (adding corresponding elements of seqA and seqB)
def seqC (n : ℕ) : ℤ := seqA n + seqB n

-- The main theorem to prove that none of the sequences A, B, or C are periodic
theorem no_sequences_periodic :
  ¬(∃ P, ∀ n, seqA (n + P) = seqA n) ∧
  ¬(∃ P, ∀ n, seqB (n + P) = seqB n) ∧
  ¬(∃ P, ∀ n, seqC (n + P) = seqC n) :=
sorry

end no_sequences_periodic_l646_646985


namespace arithmetic_sequence_sum_l646_646668

theorem arithmetic_sequence_sum :
  (∑ k in Finset.range 10, (2 * (k + 1)) / 7) = 110 / 7 :=
by
  sorry

end arithmetic_sequence_sum_l646_646668


namespace friends_game_l646_646606

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l646_646606


namespace cover_square_with_rectangles_l646_646133

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l646_646133


namespace find_f_2008_l646_646833

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero : f 0 = 2008

axiom f_inequality1 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality2 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 : f 2008 = 2^2008 + 2007 :=
sorry

end find_f_2008_l646_646833


namespace knight_problem_chessboard_l646_646850

theorem knight_problem_chessboard (C : Type) [finite C] [inhabited C] [linear_order C]:
  (∀ (x y: ℕ), x = 8 -> y = 8) -> (∀ (n: ℕ), n = 63) -> (∀ (p: ℕ × ℕ), p = (8, 8)) ->
  knight_moves (0 0) (8, 8) -> False :=
begin
  sorry
end

end knight_problem_chessboard_l646_646850


namespace mean_score_of_all_students_l646_646020

theorem mean_score_of_all_students
  (M N A : ℕ) (m n a : ℕ)
  (hM : M = 85)
  (hN : N = 75)
  (hA : A = 65)
  (hm : m = 16)
  (hn : n = 36)
  (ha : a = 24)
  (h_ratio : 3*a = 4*n ∧ 3*n = 5*m ∧ 2*m = 3*a) :
  let total_score := 85*m + 75*n + 65*a,
      total_students := m + n + a,
      mean_score := total_score / total_students
  in mean_score = 74 :=
by {
  -- these are the given conditions
  have h_m_rat : 16 = 16 := rfl,
  have h_n_rat : 36 = 36 := rfl,
  have h_a_rat : 24 = 24 := rfl,
  
  -- calculating total score
  let total_score := 85 * m + 75 * n + 65 * a,
  have total_score_eq : total_score = 85 * 16 + 75 * 36 + 65 * 24 := rfl,
  
  -- calculating total number of students
  let total_students := m + n + a,
  have total_students_eq : total_students = 16 + 36 + 24 := rfl,
  
  have H1 : total_score = 5620 := by simp [total_score_eq],

  have H2 : total_students = 76 := by simp [total_students_eq],

  let mean_score := total_score / total_students,
  have : mean_score = 74 := by
    rw [total_score_eq, total_students_eq],
    simp,

  exact this
}

end mean_score_of_all_students_l646_646020


namespace percent_decrease_is_approx_27_l646_646811

noncomputable def price_per_pack_last_month : ℝ := 7 / 3
noncomputable def price_per_pack_this_month : ℝ := 12 / 7

def percent_decrease (old_price new_price : ℝ) : ℝ :=
  ((old_price - new_price) / old_price) * 100

theorem percent_decrease_is_approx_27 :
  abs (percent_decrease price_per_pack_last_month price_per_pack_this_month - 27) < 1 :=
sorry

end percent_decrease_is_approx_27_l646_646811


namespace max_value_of_y_l646_646702

noncomputable def y (x : ℝ) : ℝ :=
  sin (x + π / 4) - cos (x + π / 3) + sin (x + π / 6)

theorem max_value_of_y : 
  ∃ x ∈ Icc (-π/4 : ℝ) (0 : ℝ), y x = 1 :=
by
  sorry

end max_value_of_y_l646_646702


namespace sum_of_cubes_form_l646_646026

theorem sum_of_cubes_form (a b : ℤ) (x1 y1 x2 y2 : ℤ)
  (h1 : a = x1^2 + 3 * y1^2) (h2 : b = x2^2 + 3 * y2^2) :
  ∃ x y : ℤ, a^3 + b^3 = x^2 + 3 * y^2 := sorry

end sum_of_cubes_form_l646_646026


namespace max_distance_circle_ellipse_l646_646024

theorem max_distance_circle_ellipse:
  (∀ P Q : ℝ × ℝ, 
     (P.1^2 + (P.2 - 3)^2 = 1 / 4) → 
     (Q.1^2 + 4 * Q.2^2 = 4) → 
     ∃ Q_max : ℝ × ℝ, 
         Q_max = (0, -1) ∧ 
         (∀ P : ℝ × ℝ, P.1^2 + (P.2 - 3)^2 = 1 / 4 →
         |dist P Q_max| = 9 / 2)) := 
sorry

end max_distance_circle_ellipse_l646_646024


namespace second_man_speed_l646_646521

/-- A formal statement of the problem -/
theorem second_man_speed (v : ℝ) 
  (start_same_place : ∀ t : ℝ, t ≥ 0 → 2 * t = (10 - v) * 1) : 
  v = 8 :=
by
  sorry

end second_man_speed_l646_646521


namespace binomial_expansion_evaluation_l646_646227

theorem binomial_expansion_evaluation : 
  (8 ^ 4 + 4 * (8 ^ 3) * 2 + 6 * (8 ^ 2) * (2 ^ 2) + 4 * 8 * (2 ^ 3) + 2 ^ 4) = 10000 := 
by 
  sorry

end binomial_expansion_evaluation_l646_646227


namespace sum_of_coordinates_G_l646_646415

theorem sum_of_coordinates_G : 
  let A := (0, 8)
  let B := (0, 0)
  let C := (10, 0)
  let D := ((0 + 0) / 2, (8 + 0) / 2) -- Midpoint A, B
  let E := ((0 + 10) / 2, (0 + 0) / 2) -- Midpoint B, C
  let F := ((0 + 10) / 2, (8 + 0) / 2) -- Midpoint A, C
  let G := (10, 0)
  in (G.1 + G.2) = 10 := by 
  sorry

end sum_of_coordinates_G_l646_646415


namespace money_per_percentage_point_l646_646853

theorem money_per_percentage_point
  (plates : ℕ) (total_states : ℕ) (total_amount : ℤ)
  (h_plates : plates = 40) (h_total_states : total_states = 50) (h_total_amount : total_amount = 160) :
  total_amount / (plates * 100 / total_states) = 2 :=
by
  -- Omitted steps of the proof
  sorry

end money_per_percentage_point_l646_646853


namespace p_plus_q_l646_646071

noncomputable def p (x : ℝ) := x - 1
noncomputable def q (x : ℝ) := 3 * x^2 - 9 * x + 6

theorem p_plus_q (x : ℝ) : p(x) + q(x) = 3 * x^2 - 8 * x + 5 :=
by {
  sorry
}

end p_plus_q_l646_646071


namespace value_of_a_l646_646721

theorem value_of_a (a : ℝ) : (∃ (a : ℝ), ∃ (B := (0, 5) : ℝ × ℝ), 
  let area := 10 in 
  let height := 5 in 
  let base := abs a in 
  (1/2) * base * height = area) → (a = 4 ∨ a = -4) :=
sorry

end value_of_a_l646_646721


namespace possible_erased_numbers_l646_646847

-- Definition for the sum of the first n natural numbers
def sum_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to prove the possible numbers that could have been erased
theorem possible_erased_numbers (x : ℕ) (h : x ∈ finset.range 21) :
  (∃ y ∈ finset.erase (finset.range 21) x, y * 19 = sum_nat 20 - x) ↔ x = 1 ∨ x = 20 :=
by
  sorry

end possible_erased_numbers_l646_646847


namespace desired_ratio_l646_646516

variables {A B C D P Q M N : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P] [MetricSpace Q] [MetricSpace M] [MetricSpace N]

-- Define vertices and points
variables (ABCD_is_parallelogram : parallelogram A B C D)
variables (line_through_D_1 : ∃ P M, line_intersects D P M A B C)
variables (line_through_D_2 : ∃ Q N, line_intersects D Q N A B C)

theorem desired_ratio (h1 : ∃ P M, line_intersects D P M A B C)
                      (h2 : ∃ Q N, line_intersects D Q N A B C) :
    (dist M N) / (dist P Q) = (dist M C) / (dist A Q) ∧ (dist N C) / (dist A P) = (dist M C) / (dist A Q) := 
begin
    sorry
end

end desired_ratio_l646_646516


namespace arithmetic_series_sum_l646_646667

theorem arithmetic_series_sum : 
  (∑ k in Finset.range 10, (2 * (k + 1) : ℚ) / 7) = 110 / 7 := 
by
  sorry

end arithmetic_series_sum_l646_646667


namespace increasing_function_condition_l646_646269

theorem increasing_function_condition (k : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k - 6) * x1 + (2 * k + 1) < (2 * k - 6) * x2 + (2 * k + 1)) ↔ (k > 3) :=
by
  -- To prove the statement, we would need to prove it in both directions.
  sorry

end increasing_function_condition_l646_646269


namespace tan_alpha_equals_4_over_3_l646_646725

noncomputable def given_condition : Prop := 
  ∃ α : ℝ, (sin (π / 2 + α) = -3 / 5) ∧ (α > -π) ∧ (α < 0)

theorem tan_alpha_equals_4_over_3 (h : given_condition) : ∃ α : ℝ, α ∈ Ioo (-π) 0 ∧ tan α = 4 / 3 :=
  sorry

end tan_alpha_equals_4_over_3_l646_646725


namespace trigonometric_identity_l646_646867

theorem trigonometric_identity :
  (\sin (15 * Real.pi / 180) + \sin (30 * Real.pi / 180) + \sin (45 * Real.pi / 180) + \sin (60 * Real.pi / 180) + \sin (75 * Real.pi / 180)) /
  (\sin (15 * Real.pi / 180) * \cos (15 * Real.pi / 180) * \cos (30 * Real.pi / 180)) = 
  4 * Real.sqrt 3 :=
sorry

end trigonometric_identity_l646_646867


namespace algebraic_expression_evaluation_l646_646904

theorem algebraic_expression_evaluation (x y : ℤ) (h1 : x = -2) (h2 : y = -4) : 2 * x^2 - y + 3 = 15 :=
by
  rw [h1, h2]
  sorry

end algebraic_expression_evaluation_l646_646904


namespace total_combined_area_l646_646175

-- Definition of the problem conditions
def base_parallelogram : ℝ := 20
def height_parallelogram : ℝ := 4
def base_triangle : ℝ := 20
def height_triangle : ℝ := 2

-- Given the conditions, we want to prove:
theorem total_combined_area :
  (base_parallelogram * height_parallelogram) + (0.5 * base_triangle * height_triangle) = 100 :=
by
  sorry  -- proof goes here

end total_combined_area_l646_646175


namespace alcohol_percentage_proof_l646_646157

-- Define initial conditions
def volume_water : ℝ := 1
def volume_solution : ℝ := 3
def percentage_alcohol_solution : ℝ := 33

-- Define the variables used in the proof
def percentage_alcohol_new_mixture : ℝ :=
  let amount_alcohol : ℝ := volume_solution * (percentage_alcohol_solution / 100)
  let total_volume : ℝ := volume_solution + volume_water
  (amount_alcohol / total_volume) * 100

-- The proof statement
theorem alcohol_percentage_proof :
  percentage_alcohol_new_mixture = 24.75 :=
by
  sorry

end alcohol_percentage_proof_l646_646157


namespace problem_correct_statements_l646_646534

noncomputable def statement_A (α : ℝ) (h : 0 < α ∧ α < π / 2) : Prop := 
  (0 < α/2) ∧ (α/2 < π / 2)

noncomputable def statement_B (f : ℝ → ℝ) (ϕ : ℝ) (h : ∀ x : ℝ, f x = sin (x + ϕ + π / 4)) : Prop := 
  ¬ (∀ x : ℝ, f x = f (-x)) ∧ ϕ = 3 * π / 4

noncomputable def statement_C (x : ℝ) (f : ℝ → ℝ) : Prop := 
  x = π / 3 ∧ f x = 2 * cos (2 * x + π / 3)

noncomputable def statement_D (θ : ℝ) (r : ℝ) : Prop := 
  θ = π / 3 ∧ r = 1 ∧ (r * θ ≠ 60)

theorem problem_correct_statements :
  (0 < α ∧ α < π/2 → statement_A α (by sorry)) ∧
  (∀ (f : ℝ → ℝ), ∀ (ϕ : ℝ), ∀ (h : ∀ x : ℝ, f x = sin (x + ϕ + π/4)), statement_B f ϕ h) ∧
  (∀ (x : ℝ), ∀ (f : ℝ → ℝ), statement_C x f) ∧
  (statement_D (π / 3) 1) :=
by sorry

end problem_correct_statements_l646_646534


namespace line_parallel_passing_through_point_line_perpendicular_area_l646_646349

theorem line_parallel_passing_through_point (l : line ℝ) (p : point ℝ) :
  (l = {a := 3, b := 4, c := -12}) → (p = (-1, 3)) → ∃ l' : line ℝ, (l'.a = 3 ∧ l'.b = 4 ∧ l'.c = -9) := by
  sorry

theorem line_perpendicular_area (l : line ℝ) (S : ℝ × ℝ → ℝ)
  (area : ℝ) :
  (l = {a := 3, b := 4, c := -12}) → (S (x, y) = 1/2 * |x| * |y|) → (area = 4) →
  ∃ l' : line ℝ, (l'.a = 4 ∧ l'.b = 3 ∧ (l'.c = - √6 ∨ l'.c = √6)) := by
  sorry

end line_parallel_passing_through_point_line_perpendicular_area_l646_646349


namespace exists_divisible_by_sum_of_digits_l646_646025

open Nat

/-!
Prove that among any 18 consecutive positive integers not exceeding 2005,
there is at least one divisible by the sum of its digits.
-/
theorem exists_divisible_by_sum_of_digits (a : ℕ) (h₁ : 1 ≤ a)
  (h₂ : a + 17 ≤ 2005) : ∃ k ∈ (finset.range 18).image (λ i, a + i), 
    exists (m : ℕ), (m = (a + k).digits 10).sum ∧ (a + k) % m = 0 := 
sorry

end exists_divisible_by_sum_of_digits_l646_646025


namespace angle_BKC_greater_than_90_l646_646051

theorem angle_BKC_greater_than_90
  (A B C P Q H K : Point)
  (hABC : Triangle ABC)
  (h_angle_B : angle (A, B, C) > 60)
  (h_angle_C : angle (A, C, B) > 60)
  (h_concyclic : Concyclic {A, P, Q, H})
  (h_midpoint_K : midpoint K P Q) :
  angle B K C > 90 :=
sorry

end angle_BKC_greater_than_90_l646_646051


namespace find_matrix_N_l646_646692

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![47/9, -16/9], 
    ![-10/3, 14/3]]

def v₁ : Fin 2 → ℚ := ![4, 1]
def v₂ : Fin 2 → ℚ := ![1, -2]

def w₁ : Fin 2 → ℚ := ![12, 10]
def w₂ : Fin 2 → ℚ := ![7, -8]

theorem find_matrix_N :
  (N ⬝ v₁ = w₁) ∧ (N ⬝ v₂ = w₂) :=
by
  sorry

end find_matrix_N_l646_646692


namespace complement_of_N_in_U_l646_646013

def U : Set ℕ := { x | x ∈ Nat ∧ x ≤ 5 }
def N : Set ℕ := {2, 4}

theorem complement_of_N_in_U :
  {x | x ∈ U ∧ x ∉ N} = {1, 3, 5} :=
by
  sorry

end complement_of_N_in_U_l646_646013


namespace find_value_of_m_l646_646744

/-- Given the parabola y = 4x^2 + 4x + 5 and the line y = 8mx + 8m intersect at exactly one point,
    prove the value of m^{36} + 1155 / m^{12} is 39236. -/
theorem find_value_of_m (m : ℝ) (h: ∃ x, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m ∧
  ∀ x₁ x₂, 4 * x₁^2 + 4 * x₁ + 5 = 8 * m * x₁ + 8 * m →
  4 * x₂^2 + 4 * x₂ + 5 = 8 * m * x₂ + 8 * m → x₁ = x₂) :
  m^36 + 1155 / m^12 = 39236 := 
sorry

end find_value_of_m_l646_646744


namespace find_b_c_range_f_range_k_l646_646739

section

variable {f : ℝ → ℝ}
variable (b c : ℝ)

-- Condition 1: Function definition 
def f (x : ℝ) : ℝ := x ^ 2 - b * x + c

-- Condition 2: Axis of symmetry is x = 1
def axis_of_symmetry (x : ℝ) := x = 1

-- Condition 3: f(0) = -1
def f_zero_eq_neg_one := f 0 = -1

-- Proof Problem 1
theorem find_b_c (h_symm : axis_of_symmetry (f ^.deriv 1 / (2 * f ^.deriv 2)) ) (h_f0 : f 0 = -1) : 
  b = -2 ∧ c = -1 := sorry 

-- Range of f(x) on [0, 3] when b = -2 and c = -1
theorem range_f (hb : b = -2) (hc : c = -1) (h_dom : ∀ x, 0 ≤ x ∧ x ≤ 3) : 
  set.range (λ x : ℝ, f x) = set.Icc (-2) 2 := sorry

-- Range of k such that f(log2 k) > f(2) holds
theorem range_k (hb : b = -2) (hc : c = -1) : 
  {k : ℝ | f (Real.log2 k) > f 2} = {k : ℝ | k > 4 ∨ (0 < k ∧ k < 1)} := sorry 

end

end find_b_c_range_f_range_k_l646_646739


namespace group_friends_opponents_l646_646622

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l646_646622


namespace hernandez_residency_months_l646_646460

theorem hernandez_residency_months
    (taxable_income : ℕ)
    (tax_rate : ℝ)
    (total_tax : ℝ)
    (m : ℕ)
    (year_months : ℕ := 12)
    (taxable_income_value : taxable_income = 42500)
    (tax_rate_value : tax_rate = 0.04)
    (total_tax_value : total_tax = 1275) :
    m = 9 :=
by
    have h_tax_equation : total_tax = taxable_income * tax_rate * (m.to_real / year_months.to_real) :=
        by rw [taxable_income_value, tax_rate_value, total_tax_value]
    rw [h_tax_equation]
    sorry

end hernandez_residency_months_l646_646460


namespace carla_counted_tuesday_l646_646099

theorem carla_counted_tuesday :
  (let ceiling_tiles := 38;
       books := 75;
       ceiling_tiles_count := ceiling_tiles * 2;
       books_count := books * 3;
       total_count := ceiling_tiles_count + books_count
   in total_count = 301) :=
by
  unfold ceiling_tiles books ceiling_tiles_count books_count total_count;
  sorry

end carla_counted_tuesday_l646_646099


namespace min_rectangles_to_cover_square_exactly_l646_646130

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l646_646130


namespace constant_term_in_expansion_l646_646935

theorem constant_term_in_expansion : 
  (∃ c : ℚ, constant_term (expand_binomial (12 : ℕ) (λ x, sqrt x + 7 / x^2) (1 : ℕ)) = c ∧ c = 3234) :=
sorry

end constant_term_in_expansion_l646_646935


namespace number_of_common_divisors_l646_646184

/--
A ticket to a school concert costs $x$ dollars, where $x$ is a whole number.
A group of 11th graders buys tickets costing a total of $60, and a group of 12th graders buys tickets costing a total of $90.
Prove that there are 8 possible values for $x$ that are common divisors of both $60$ and $90$.
-/
theorem number_of_common_divisors : 
  ∃ num_values : ℕ, num_values = finset.card (finset.filter (λ d, 60 % d = 0 ∧ 90 % d = 0) (finset.range 31)) ∧ num_values = 8 :=
by {
  -- Define the set of whole number divisors of 60 and 90 within the range.
  let divisors_60_90 := finset.filter (λ d, 60 % d = 0 ∧ 90 % d = 0) (finset.range 31),
  use finset.card divisors_60_90,
  -- Proof that there are 8 common divisors.
  split,
  { refl },
  { sorry } -- Proof of the calculation
}

end number_of_common_divisors_l646_646184


namespace avg_k_for_polynomial_roots_l646_646298

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l646_646298


namespace phil_quarters_collected_each_month_l646_646856

theorem phil_quarters_collected_each_month : 
  ∃ (x : ℕ), 
  let quarters_before_loss := (4 * 105) / 3,
      quarters_collected_every_third_month := 12 / 3,
      quarters_before_monthly_collection := quarters_before_loss - quarters_collected_every_third_month,
      initial_quarters := 50 in
  (quarters_before_monthly_collection - 12 * x) / 2 = initial_quarters ∧ x = 3 :=
by sorry

end phil_quarters_collected_each_month_l646_646856


namespace distinct_real_roots_k_root_condition_k_l646_646746

-- Part (1) condition: The quadratic equation has two distinct real roots
theorem distinct_real_roots_k (k : ℝ) : (∃ x : ℝ, x^2 + 2*x + k = 0) ∧ (∀ x y : ℝ, x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0 → x ≠ y) → k < 1 := 
sorry

-- Part (2) condition: m is a root and satisfies m^2 + 2m = 2
theorem root_condition_k (m k : ℝ) : m^2 + 2*m = 2 → m^2 + 2*m + k = 0 → k = -2 := 
sorry

end distinct_real_roots_k_root_condition_k_l646_646746


namespace graph_of_equation_represents_three_lines_l646_646526

theorem graph_of_equation_represents_three_lines (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
    ((a * x + b * y + c = 0) ∧ (a * x + b * y + c ≠ 0)) ∨
    ((a * x + b * y + c = 0) ∨ (a * x + b * y + c ≠ 0)) ∨
    (a * x + b * y + c = 0)) :=
by
  sorry

end graph_of_equation_represents_three_lines_l646_646526


namespace cross_section_of_prism_with_plane_ABC_exists_l646_646718

variable {P : Type*} [AffineSpace ℝ P] -- Assuming real number field and affine space

variable (a b c : Line P) -- The edges of the prism
variable (A B C : P) -- Points on the edges
variable (planeABC : Plane P) -- Plane formed by A, B, and C

-- Conditions
axiom point_A_on_edge_a : A ∈ a
axiom point_B_on_edge_b : B ∈ b
axiom point_C_on_edge_c : C ∈ c

-- The proof statement
theorem cross_section_of_prism_with_plane_ABC_exists :
  ∃ (X : P), (X ∈ planeABC) ∧ (∃ A1 B1 : P, A1 ∈ a ∧ B1 ∈ b ∧ line A B = line A1 B1) :=
sorry

end cross_section_of_prism_with_plane_ABC_exists_l646_646718


namespace find_three_digit_number_l646_646688

def is_valid_three_digit_number (M G U : ℕ) : Prop :=
  M ≠ G ∧ G ≠ U ∧ M ≠ U ∧ 
  0 ≤ M ∧ M ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧
  100 * M + 10 * G + U = (M + G + U) * (M + G + U - 2)

theorem find_three_digit_number : ∃ (M G U : ℕ), 
  is_valid_three_digit_number M G U ∧
  100 * M + 10 * G + U = 195 :=
by
  sorry

end find_three_digit_number_l646_646688


namespace avg_of_k_with_positive_integer_roots_l646_646323

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l646_646323


namespace train_crossing_time_l646_646582

-- Definitions based on the conditions given in the problem
def speed_km_per_hr : ℝ := 72
def length_train : ℝ := 220
def length_platform : ℝ := 300

-- Adding a conversion factor from km/hr to m/s
def conversion_factor : ℝ := 5 / 18

-- Calculating the speed in m/s
def speed_m_per_s : ℝ := speed_km_per_hr * conversion_factor

-- The total distance covered by the train to cross the platform
def total_distance : ℝ := length_train + length_platform

-- Using the formula time = distance / speed to calculate the time
def time_to_cross_platform : ℝ := total_distance / speed_m_per_s

theorem train_crossing_time :
  time_to_cross_platform = 26 := sorry

end train_crossing_time_l646_646582


namespace other_endpoint_of_diameter_l646_646662

def center : ℝ × ℝ := (3, 4)
def endpoint1 : ℝ × ℝ := (1, -2)
def endpoint2 : ℝ × ℝ := (5, 10)

theorem other_endpoint_of_diameter :
  let (cx, cy) := center in
  let (ex1, ey1) := endpoint1 in
  let (ex2, ey2) := endpoint2 in
  (ex2, ey2) = (2 * cx - ex1, 2 * cy - ey1) :=
by {
  sorry
}

end other_endpoint_of_diameter_l646_646662


namespace average_k_positive_int_roots_l646_646340

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l646_646340


namespace avg_k_for_polynomial_roots_l646_646296

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l646_646296


namespace cost_equation_l646_646568

variables (x y z : ℝ)

theorem cost_equation (h1 : 2 * x + y + 3 * z = 24) (h2 : 3 * x + 4 * y + 2 * z = 36) : x + y + z = 12 := by
  -- proof steps would go here, but are omitted as per instruction
  sorry

end cost_equation_l646_646568


namespace product_of_t_values_l646_646244

theorem product_of_t_values : 
  (∏ t in {t | (t / 2)^2 = 100}, t) = -400 :=
by
  sorry

end product_of_t_values_l646_646244


namespace positive_difference_abs_eq_l646_646946

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l646_646946


namespace distance_P1_P2_l646_646279

-- Define the points and their symmetric conditions
def P : ℝ × ℝ × ℝ := (1, 2, 3)
def P1 : ℝ × ℝ × ℝ := (-1, 2, -3) -- Symmetric point of P about the y-axis
def P2 : ℝ × ℝ × ℝ := (1, -2, 3) -- Symmetric point of P about the coordinate plane xOz

-- Define the Euclidean distance formula in 3D
noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

-- Mathematical proof statement
theorem distance_P1_P2 : distance P1 P2 = 2 * Real.sqrt 14 :=
by
  sorry

end distance_P1_P2_l646_646279


namespace abc_product_l646_646677

-- Define the partial fraction decomposition
def partial_fraction (A B C : Rat) (x : Rat) : Prop :=
  x^2 - 19 = A * (x + 3) * (x - 4) + B * (x - 1) * (x - 4) + C * (x - 1) * (x + 3)

-- Define the condition for the values of A, B, C in the partial fraction decomposition
def values_of_ABC : Prop :=
  ∃ (A B C : Rat),
  A = 3/2 ∧ B = -5/14 ∧ C = -1/7 ∧
  partial_fraction A B C

-- Prove that the product ABC is 15/196 given the conditions
theorem abc_product : values_of_ABC → 3/2 * (-5/14) * (-1/7) = 15/196 :=
by
  intro h
  cases h with A h
  cases h with B h
  cases h with C h
  cases h with hA h
  cases h with hB h
  cases h with hC hpf
  simp [hA, hB, hC]
  norm_num
  sorry

end abc_product_l646_646677


namespace wheels_on_each_other_axle_l646_646903

def truck_toll_wheels (t : ℝ) (x : ℝ) (w : ℕ) : Prop :=
  t = 1.50 + 1.50 * (x - 2) ∧ (w = 18) ∧ (∀ y : ℕ, y = 18 - 2 - 4 *(x - 5) / 4)

theorem wheels_on_each_other_axle :
  ∀ t x w, truck_toll_wheels t x w → w = 18 ∧ x = 5 → (18 - 2) / 4 = 4 :=
by
  intros t x w h₁ h₂
  have h₃ : t = 6 := sorry
  have h₄ : x = 4 := sorry
  have h₅ : w = 18 := sorry
  have h₆ : (18 - 2) / 4 = 4 := sorry
  exact h₆

end wheels_on_each_other_axle_l646_646903


namespace sin_trig_identity_l646_646454

noncomputable def alpha_in_degrees (α : ℝ) := α ∈ set.Ioo 0 90
noncomputable def sin_75_plus_2alpha (α : ℝ) := Real.sin (75 + 2 * α * Real.pi / 180) = -3 / 5

theorem sin_trig_identity (α : ℝ) (hα : alpha_in_degrees α) (h : sin_75_plus_2alpha α) :
  Real.sin ((15 + α) * Real.pi / 180) * Real.sin ((75 - α) * Real.pi / 180) = Real.sqrt 2 / 20 := 
sorry

end sin_trig_identity_l646_646454


namespace point_coincides_with_center_l646_646037

theorem point_coincides_with_center
    (circle_center : Point)
    (unit_circle : Circle circle_center 1)
    (points : Fin 7 → Point)
    (h_dist : ∀ (i j : Fin 7), i ≠ j → dist (points i) (points j) ≥ 1)
    (h_in_circle : ∀ i : Fin 7, points i ∈ unit_circle) :
    ∃ i : Fin 7, points i = circle_center :=
    sorry

end point_coincides_with_center_l646_646037


namespace log_cosine_range_l646_646512

noncomputable def log_base_three (a : ℝ) : ℝ := Real.log a / Real.log 3

theorem log_cosine_range (x : ℝ) (hx : x ∈ Set.Ioo (Real.pi / 2) (7 * Real.pi / 6)) :
    ∃ y, y = log_base_three (1 - 2 * Real.cos x) ∧ y ∈ Set.Ioc 0 1 :=
by
  sorry

end log_cosine_range_l646_646512


namespace hyperbola_focal_length_and_eccentricity_l646_646888

noncomputable def a := 1
noncomputable def b := real.sqrt 3
noncomputable def c := real.sqrt (a ^ 2 + b ^ 2)
noncomputable def focal_length := 2 * c
noncomputable def eccentricity := c / a

theorem hyperbola_focal_length_and_eccentricity :
  (focal_length = 4) ∧ (eccentricity = 2) :=
by
  sorry

end hyperbola_focal_length_and_eccentricity_l646_646888


namespace circle_tangent_to_asymptotes_perpendicular_bisector_y_intercept_range_l646_646358

-- Definitions of the hyperbola, asymptotes, and specific points
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1
def right_focus : ℝ × ℝ := (Float.sqrt 2, 0)
def asymptote_1 (x y : ℝ) : Prop := x + y = 0
def asymptote_2 (x y : ℝ) : Prop := x - y = 0

-- Definition of specific point P
def P : ℝ × ℝ := (0, -1)

-- Problems as proof statements in Lean
theorem circle_tangent_to_asymptotes : ∃ x y : ℝ, (x - Float.sqrt 2)^2 + y^2 = 1 := 
sorry

theorem perpendicular_bisector_y_intercept_range : ∀ (k : ℝ),
  (1 < k ∧ k < Float.sqrt 2) → (∃ t : ℝ, t > 2) :=
sorry

end circle_tangent_to_asymptotes_perpendicular_bisector_y_intercept_range_l646_646358


namespace triangle_area_l646_646541

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) :
  perimeter = 36 → inradius = 2.5 → area = inradius * (perimeter / 2) → area = 45 :=
by
  intros h_perimeter h_inradius h_area
  rw [h_perimeter, h_inradius] at h_area
  have h : 2.5 * (36 / 2) = 45 := by norm_num
  rw h at h_area
  exact h_area

end triangle_area_l646_646541


namespace total_tiles_on_floor_l646_646632

def is_square_floored (s : ℕ) : Prop :=
  let total_tiles := s * s in
  let black_border := 4 * s - 4 in
  black_border = 204

theorem total_tiles_on_floor (s : ℕ) (h : is_square_floored s) : s * s = 2704 :=
  sorry

end total_tiles_on_floor_l646_646632


namespace area_ratio_l646_646030

-- Define a regular decagon and pentagon
structure RegularDecagon (α : Type) :=
  (vertices : Fin 10 → α)
  -- additional properties ensuring the shape is a regular decagon
  (is_regular : ∃ (s : ℝ), ∀ i, dist (vertices i) (vertices (i + 1) % 10) = s)

structure RegularPentagon (α : Type) :=
  (vertices : Fin 5 → α)
  -- additional properties ensuring the shape is a regular pentagon
  (is_regular : ∃ (s : ℝ), ∀ i, dist (vertices i) (vertices (i + 1) % 5) = s)

-- Area of a regular decagon
noncomputable def area_regular_decagon {α : Type} [MetricSpace α] (d : RegularDecagon α) : ℝ := sorry

-- Area of a regular pentagon
noncomputable def area_regular_pentagon {α : Type} [MetricSpace α] (p : RegularPentagon α) : ℝ := sorry

-- Prove the ratio of the areas
theorem area_ratio {α : Type} [MetricSpace α] 
  (d : RegularDecagon α) (p : RegularPentagon α)
  (h : ∀ i, p.vertices i = d.vertices (2 * i % 10)) :
  area_regular_pentagon p / area_regular_decagon d = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end area_ratio_l646_646030


namespace solve_trig_eq_l646_646482

theorem solve_trig_eq (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : ∃ k : ℤ, a * sin (arctan (b / a) + k * π) + b * cos (arctan (b / a) + k * π) = c := 
sorry

end solve_trig_eq_l646_646482


namespace quadratic_roots_distinct_and_m_value_l646_646749

theorem quadratic_roots_distinct_and_m_value (m : ℝ) (α β : ℝ) (h_equation : ∀ x, x^2 - 2 * x - 3 * m^2 = 0 → (Root_of(x) = α ∨ Root_of(x) = β)) 
(h_alpha_beta : α + 2 * β = 5) :
  (2^2 - 4 * 1 * -3 * m^2 > 0) ∧ (m^2 = 1) :=
by
  have h_discriminant : 4 + 12 * m^2 > 0 := by sorry
  have h_root_sum : α + β = 2 := by sorry
  have h_root_product : α * β = -3 * m^2 := by sorry
  have h_quad_solved : (β = 3) ∧ (α = -1) := by sorry
  have h_m2 : m^2 = 1 := by sorry
  exact ⟨h_discriminant,h_m2⟩

end quadratic_roots_distinct_and_m_value_l646_646749


namespace coins_remainder_divide_by_nine_remainder_l646_646556

def smallest_n (n : ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_remainder (n : ℕ) (h : smallest_n n) : (∃ m : ℕ, n = 54) :=
  sorry

theorem divide_by_nine_remainder (n : ℕ) (h : smallest_n n) (h_smallest: coins_remainder n h) : n % 9 = 0 :=
  sorry

end coins_remainder_divide_by_nine_remainder_l646_646556


namespace smallest_num_rectangles_l646_646127

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l646_646127


namespace closest_point_on_line_l646_646243

theorem closest_point_on_line (x y : ℝ) (h : y = (x - 3) / 3) : 
  (∃ p : ℝ × ℝ, p = (4, -2) ∧ ∀ q : ℝ × ℝ, (q.1, q.2) = (x, y) ∧ q ≠ p → dist p q ≥ dist p (33/10, 1/10)) :=
sorry

end closest_point_on_line_l646_646243


namespace problem_statement_l646_646818

open Finset BigOperators

-- Define the problem with given conditions
noncomputable def num_functions_satisfying_conditions : ℕ :=
  let A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  Finset.card ((A.product A).filter (λ p, ∃ c ∈ A, (∀ x ∈ A, f(f(x)) = c) ∧ ∃ y ∈ A, f(y) ≠ y))

-- Define the target value based on the solution
def expected_remainder : ℕ := 992

-- Statement to prove
theorem problem_statement :
  (num_functions_satisfying_conditions % 1000) = expected_remainder :=
  by sorry

end problem_statement_l646_646818


namespace pyramid_height_correct_l646_646499

noncomputable def pyramid_height : ℝ :=
  let ab := 15 * Real.sqrt 3
  let bc := 14 * Real.sqrt 3
  let base_area := ab * bc
  let volume := 750
  let height := 3 * volume / base_area
  height

theorem pyramid_height_correct : pyramid_height = 25 / 7 :=
by
  sorry

end pyramid_height_correct_l646_646499


namespace possible_number_of_friends_l646_646597

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l646_646597


namespace sam_friend_points_l646_646034

theorem sam_friend_points (sam_points total_points : ℕ) (h1 : sam_points = 75) (h2 : total_points = 87) :
  total_points - sam_points = 12 :=
by sorry

end sam_friend_points_l646_646034


namespace number_of_negatives_l646_646796

-- Definitions of the given expressions
def expr1 := - (1 ^ 2)
def expr2 := abs (-1)
def expr3 := 1 / -1
def expr4 := (-1) ^ 2023
def expr5 := - (-1)

-- The proof statement
theorem number_of_negatives : 
  (expr1 = -1 → true) ∧ 
  (expr2 = -1 → false) ∧ 
  (expr3 = -1 → true) ∧ 
  (expr4 = -1 → true) ∧ 
  (expr5 = -1 → false) →
  (count_negative : ℕ) = 3 :=
begin
  sorry
end

end number_of_negatives_l646_646796


namespace group_of_friends_l646_646589

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l646_646589


namespace triangle_area_relation_l646_646638

theorem triangle_area_relation :
  let A := (1 / 2) * 5 * 5
  let B := (1 / 2) * 12 * 12
  let C := (1 / 2) * 13 * 13
  A + B = C :=
by
  sorry

end triangle_area_relation_l646_646638


namespace maxCables_needed_l646_646571

-- Definitions for the problem
structure Company (X Y : Type) :=
(employees_X : X → Prop)
(employees_Y : Y → Prop)
(total_X : ℕ)
(total_Y : ℕ)
(condition : total_X = 25 ∧ total_Y = 15)

def maxCables (X Y : Type) [Fintype X] [Fintype Y] (C : Company X Y) : ℕ := 
  (Fintype.card X - 1) * Fintype.card Y + 1

theorem maxCables_needed (X Y : Type) [Fintype X] [Fintype Y] (C : Company X Y) (h : C.condition) :
  maxCables X Y C = 351 :=
by
  -- Place proof here
  sorry

end maxCables_needed_l646_646571


namespace amount_remaining_l646_646977

theorem amount_remaining (deposit : ℝ) (percent : ℝ) (total_price : ℝ) : 
  deposit = 120 ∧ percent = 0.10 ∧ total_price = deposit / percent → 
  total_price - deposit = 1080 :=
by
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2]
  sorry

end amount_remaining_l646_646977


namespace abs_diff_eq_two_l646_646470

noncomputable def sqrt_e := Real.sqrt Real.exp(1)

-- Define the equation y^2 + x^4 = 2x^2 y + 1
def eq (x y : ℝ) := y^2 + x^4 = 2 * x^2 * y + 1

-- Define a and b such that (sqrt(e), a) and (sqrt(e), b) are distinct points on the given equation
variables (a b : ℝ)
hypothesis h₀ : eq sqrt_e a
hypothesis h₁ : eq sqrt_e b
hypothesis h₂ : a ≠ b

-- Proof statement to show |a - b| = 2
theorem abs_diff_eq_two : |a - b| = 2 :=
sorry

end abs_diff_eq_two_l646_646470


namespace total_circle_area_within_triangle_l646_646273

-- Define the sides of the triangle
def triangle_sides : Prop := ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5

-- Define the radii and center of the circles at each vertex of the triangle
def circle_centers_and_radii : Prop := ∃ (r : ℝ) (A B C : ℝ × ℝ), r = 1

-- The formal statement that we need to prove:
theorem total_circle_area_within_triangle :
  triangle_sides ∧ circle_centers_and_radii → 
  (total_area_of_circles_within_triangle = π / 2) := sorry

end total_circle_area_within_triangle_l646_646273


namespace max_students_distributed_equally_l646_646077

theorem max_students_distributed_equally (pens pencils : ℕ) (h1 : pens = 3528) (h2 : pencils = 3920) : 
  Nat.gcd pens pencils = 392 := 
by 
  sorry

end max_students_distributed_equally_l646_646077


namespace smallest_num_rectangles_l646_646123

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l646_646123


namespace infinite_points_of_one_color_l646_646684

theorem infinite_points_of_one_color (colors : ℤ → Prop) (red blue : ℤ → Prop)
  (h_colors : ∀ n : ℤ, colors n → (red n ∨ blue n))
  (h_red_blue : ∀ n : ℤ, red n → ¬ blue n)
  (h_blue_red : ∀ n : ℤ, blue n → ¬ red n) :
  ∃ c : ℤ → Prop, (∀ k : ℕ, ∃ infinitely_many p : ℤ, c p ∧ p % k = 0) :=
by
  sorry

end infinite_points_of_one_color_l646_646684


namespace probability_of_negative_product_l646_646101

def set_integers : Set ℤ := { -5, -8, -1, 7, 4, 2, -3 }

noncomputable def probability_negative_product : ℚ :=
  let negs := {a | a ∈ set_integers ∧ a < 0}
  let pos := {a | a ∈ set_integers ∧ a > 0}
  let successful_outcomes := negs.card * pos.card
  let total_outcomes := (set_integers.card.choose 2)
  (successful_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_of_negative_product :
  probability_negative_product = 4 / 7 := 
by
  sorry

end probability_of_negative_product_l646_646101


namespace toads_per_acre_l646_646053

theorem toads_per_acre (b g : ℕ) (h₁ : b = 25 * g)
  (h₂ : b / 4 = 50) : g = 8 :=
by
  -- Condition h₁: For every green toad, there are 25 brown toads.
  -- Condition h₂: One-quarter of the brown toads are spotted, and there are 50 spotted brown toads per acre.
  sorry

end toads_per_acre_l646_646053


namespace regularQuadrilateralPrismCharacterization_l646_646880

def isRegularQuadrilateralPrism (base : Set Point) (sideEdges : Set (Point × Point)) :=
  ∃ square : Set Point, isSquare square ∧ square = base ∧
  ∀ sideEdge ∈ sideEdges, isPerpendicularToBase base sideEdge

def isConditionC (base : Set Point) (vertices : Set Point) :=
  isRhombus base ∧
  ∃ v ∈ vertices, ∀ edge ∈ incidentEdges v,
  areMutuallyPerpendicular edge

theorem regularQuadrilateralPrismCharacterization 
(base : Set Point) (sideEdges : Set (Point × Point)) (vertices : Set Point) :
  isRegularQuadrilateralPrism base sideEdges ↔ isConditionC base vertices :=
sorry

end regularQuadrilateralPrismCharacterization_l646_646880


namespace mean_equality_l646_646078

theorem mean_equality (y : ℝ) : 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 :=
by
  intros h
  sorry

end mean_equality_l646_646078


namespace hannah_mugs_problem_l646_646365

theorem hannah_mugs_problem :
  ∀ (total_mugs blue_mugs red_mugs yellow_mugs other_mugs : ℕ),
  total_mugs = 40 →
  yellow_mugs = 12 →
  red_mugs = yellow_mugs / 2 →
  blue_mugs = 3 * red_mugs →
  other_mugs = total_mugs - (blue_mugs + red_mugs + yellow_mugs) →
  other_mugs = 4 :=
by
  intros total_mugs blue_mugs red_mugs yellow_mugs other_mugs
  intros h_total h_yellow h_red h_blue h_other
  have h1: red_mugs = 6, by linarith [h_yellow, h_red]
  have h2: blue_mugs = 18, by linarith [h1, h_blue]
  have h3: other_mugs = 4, by linarith [h_total, h2, h1, h_yellow, h_other]
  exact h3

end hannah_mugs_problem_l646_646365


namespace length_chord_MN_l646_646994

-- Definitions and conditions
def P : ℝ × ℝ := (-2, 2)
def hyperbola (x y : ℝ) : Prop := x^2 - 2 * y^2 = 8
def midpoint (M N P : ℝ × ℝ) : Prop := (M.1 + N.1) / 2 = P.1 ∧ (M.2 + N.2) / 2 = P.2

-- Theorem
theorem length_chord_MN {M N : ℝ × ℝ}
  (P : ℝ × ℝ)
  (hP : P = (-2, 2))
  (h_mid : midpoint M N P)
  (hM : hyperbola M.1 M.2)
  (hN : hyperbola N.1 N.2) :
  let |MN| := real.sqrt (1 + (abs ((M.2 - N.2) / (M.1 - N.1)))^2) * real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
  in |MN| = 2 * real.sqrt 30 :=
by
  sorry

end length_chord_MN_l646_646994


namespace tickets_system_l646_646050

variable (x y : ℕ)

theorem tickets_system (h1 : x + y = 20) (h2 : 2800 * x + 6400 * y = 74000) :
  (x + y = 20) ∧ (2800 * x + 6400 * y = 74000) :=
by {
  exact (And.intro h1 h2)
}

end tickets_system_l646_646050


namespace red_bean_percentage_l646_646250

theorem red_bean_percentage :
  let bagA_beans := 24
  let bagB_beans := 32
  let bagC_beans := 36
  let bagD_beans := 40

  let bagA_red_ratio := 0.40
  let bagB_red_ratio := 0.30
  let bagC_red_ratio := 0.25
  let bagD_red_ratio := 0.15

  let bagA_red_beans := bagA_beans * bagA_red_ratio
  let bagB_red_beans := bagB_beans * bagB_red_ratio
  let bagC_red_beans := bagC_beans * bagC_red_ratio
  let bagD_red_beans := bagD_beans * bagD_red_ratio

  let total_red_beans := bagA_red_beans + bagB_red_beans + bagC_red_beans + bagD_red_beans
  let total_beans := bagA_beans + bagB_beans + bagC_beans + bagD_beans

  let red_bean_percentage := (total_red_beans / total_beans) * 100

  red_bean_percentage ≈ 27 := sorry

end red_bean_percentage_l646_646250


namespace hannah_mugs_problem_l646_646364

theorem hannah_mugs_problem :
  ∀ (total_mugs blue_mugs red_mugs yellow_mugs other_mugs : ℕ),
  total_mugs = 40 →
  yellow_mugs = 12 →
  red_mugs = yellow_mugs / 2 →
  blue_mugs = 3 * red_mugs →
  other_mugs = total_mugs - (blue_mugs + red_mugs + yellow_mugs) →
  other_mugs = 4 :=
by
  intros total_mugs blue_mugs red_mugs yellow_mugs other_mugs
  intros h_total h_yellow h_red h_blue h_other
  have h1: red_mugs = 6, by linarith [h_yellow, h_red]
  have h2: blue_mugs = 18, by linarith [h1, h_blue]
  have h3: other_mugs = 4, by linarith [h_total, h2, h1, h_yellow, h_other]
  exact h3

end hannah_mugs_problem_l646_646364


namespace subset_sum_condition_l646_646714

theorem subset_sum_condition (x : Fin 100 → ℝ) 
  (h_sum : ∑ i, x i = 1) 
  (h_diff : ∀ k : Fin 99, | x k.succ - x k | < 1 / 50) :
  ∃ (s : Finset (Fin 100)), s.card = 50 ∧ 
  | (∑ i in s, x i) - 1/2 | ≤ 1/100 :=
by
  sorry

end subset_sum_condition_l646_646714


namespace range_of_sum_l646_646723

theorem range_of_sum (a b : ℝ) (h : 2^a + 2^b = 1) : a + b ≤ -2 :=
sorry

end range_of_sum_l646_646723


namespace color_of_last_bead_l646_646566

theorem color_of_last_bead (cycle_len num_cycles : ℕ) (seq : List String) (first_bead : String) :
  cycle_len = 10 →
  num_cycles = 9 →
  seq = ["white", "white", "black", "black", "black", "yellow", "green", "green", "green", "green"] →
  first_bead = "white" →
  (seq.length * num_cycles + 1 = 91) →
  seq.head = some first_bead →
  first_bead = "white" := 
by
  sorry

end color_of_last_bead_l646_646566


namespace total_cost_for_seeds_l646_646577

theorem total_cost_for_seeds :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  in total = 18.00 :=
by
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  have h1 : total = 18.00,
  {
    sorry
  }
  exact h1

end total_cost_for_seeds_l646_646577


namespace coins_remainder_divide_by_nine_remainder_l646_646555

def smallest_n (n : ℕ) : Prop :=
  n % 8 = 6 ∧ n % 7 = 5

theorem coins_remainder (n : ℕ) (h : smallest_n n) : (∃ m : ℕ, n = 54) :=
  sorry

theorem divide_by_nine_remainder (n : ℕ) (h : smallest_n n) (h_smallest: coins_remainder n h) : n % 9 = 0 :=
  sorry

end coins_remainder_divide_by_nine_remainder_l646_646555


namespace find_two_digit_number_l646_646898

theorem find_two_digit_number (N : ℕ) (a b c : ℕ) 
  (h_end_digits : N % 1000 = c + 10 * b + 100 * a)
  (hN2_end_digits : N^2 % 1000 = c + 10 * b + 100 * a)
  (h_nonzero : a ≠ 0) :
  10 * a + b = 24 := 
by
  sorry

end find_two_digit_number_l646_646898


namespace minimum_a_condition_l646_646283

theorem minimum_a_condition :
  ∃ a : ℝ, (a > 0) ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9) ∧ a = 4 :=
by
  use 4
  split
  . norm_num
  . split
    . intros x y hx hy
      calc
        (x + y) * (1 / x + 4 / y) ≥ 9 := sorry
    . norm_num

end minimum_a_condition_l646_646283


namespace chip_exit_boundary_l646_646395

-- Define the 4x4 grid cells and initial state.
inductive Cell : Type
| A1 | A2 | A3 | A4
| B1 | B2 | B3 | B4
| C1 | C2 | C3 | C4
| D1 | D2 | D3 | D4
deriving DecidableEq

-- Define the direction of arrows in the cells.
inductive Direction : Type
| Left | Right | Up | Down
deriving DecidableEq

-- Initial arrow directions in each cell
def initialArrowDirection : Cell → Direction
| Cell.C1 := Direction.Up
| Cell.C2 := Direction.Right
| Cell.C3 := Direction.Up
| Cell.C4 := Direction.Left
| Cell.B1 := Direction.Up
| Cell.B2 := Direction.Left
| Cell.B3 := Direction.Up
| Cell.B4 := Direction.Left
| Cell.A1 := Direction.Right
| Cell.A2 := Direction.Left
| Cell.A3 := Direction.Right
| Cell.A4 := Direction.Down
| Cell.D1 := Direction.Right
| Cell.D2 := Direction.Up
| Cell.D3 := Direction.Left
| Cell.D4 := Direction.Right

-- Define the movement transitions based on the current cell and direction.
def move : Cell × Direction → Cell
| (Cell.C2, Direction.Right) := Cell.C3
| (Cell.C3, Direction.Up)    := Cell.B3
| (Cell.B3, Direction.Up)    := Cell.A3
| (Cell.A3, Direction.Right) := Cell.A4
| (Cell.A4, Direction.Down)  := Cell.B4
| (Cell.B4, Direction.Left)  := Cell.B3
| (Cell.B3, Direction.Down)  := Cell.C3
| (Cell.C3, Direction.Down)  := Cell.D3
| (Cell.D3, Direction.Left)  := Cell.D2
| (Cell.D2, Direction.Up)    := Cell.C2
| (Cell.C2, Direction.Left)  := Cell.C1
| (Cell.C1, Direction.Up)    := Cell.B1
| (Cell.B1, Direction.Up)    := Cell.A1
| (Cell.A1, Direction.Right) := Cell.A2
| (_, _)                     := Cell.A2 -- Default case to ensure totality

-- Define the theorem to prove the exit cell based on initial conditions and movement rules
theorem chip_exit_boundary : move (move (move (move (move (move (move (move (move (move (move (move (move (move (Cell.C2, Direction.Right))))))))))))) = Cell.A2 :=
by sorry

end chip_exit_boundary_l646_646395


namespace cos_B_value_triangle_perimeter_l646_646423

section TriangleProblem

variables (A B C : Real) (a b c : Real)
variables (h₁ : a * real.cos C + c * real.cos A = 4 * b * real.cos B)
variables (h₂ : b = 2 * real.sqrt 19)
variables (area : Real)
variables (h₃ : area = 6 * real.sqrt 15)

theorem cos_B_value : real.cos B = 1 / 4 :=
by sorry

theorem triangle_perimeter : let P := a + b + c in P = 14 + 2 * real.sqrt 19 :=
by sorry

end TriangleProblem

end cos_B_value_triangle_perimeter_l646_646423


namespace identify_150th_digit_l646_646928

def repeating_sequence : List ℕ := [3, 8, 4, 6, 1, 5]

theorem identify_150th_digit :
  (150 % 6 = 0) →
  nth repeating_sequence 5 = 5 :=
by
  intros h
  rewrite_modulo h
  rfl

end identify_150th_digit_l646_646928


namespace minimum_value_expression_l646_646444

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  6 * a^3 + 9 * b^3 + 32 * c^3 + (1 / (4 * a * b * c)) ≥ 6 :=
begin
  sorry
end

end minimum_value_expression_l646_646444


namespace find_angle_between_vectors_l646_646278

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

def non_zero_vectors (a b : V) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def magnitude_relation (a b : V) : Prop :=
  ∥a∥ = real.sqrt 3 * ∥b∥

def orthogonal_vectors (a b : V) : Prop :=
  inner (a - b) (a - 3 • b) = 0

theorem find_angle_between_vectors
  (h1 : non_zero_vectors a b)
  (h2 : magnitude_relation a b)
  (h3 : orthogonal_vectors a b) :
  ∃ θ : ℝ, θ = (real.pi / 6) :=
sorry

end find_angle_between_vectors_l646_646278


namespace inverse_proportion_value_k_l646_646288

theorem inverse_proportion_value_k : 
  (∃ k : ℝ, ∃ x y : ℝ, x = 2 ∧ y = 5 ∧ y = k / x) → (k = 10) :=
by
  intros h,
  obtain ⟨k, x, y, hx, hy, hxy⟩ := h,
  rw [hx, hy] at hxy,
  linarith,
  sorry 

end inverse_proportion_value_k_l646_646288


namespace coin_toss_prob_l646_646967

def mutually_exclusive (A B : Set) : Prop :=
  ∀ ω, ω ∈ A → ω ∉ B

def complementary (A B : Set) : Prop :=
  Aᶜ = B

theorem coin_toss_prob : ∀ (E1 E2 : Set),
  E1 = {ω | ω.countPos = 1} ∧ E2 = {ω | ω.countPos = 2} →
  mutually_exclusive E1 E2 ∧ ¬complementary E1 E2 :=
by
  sorry

end coin_toss_prob_l646_646967


namespace max_f_value_max_ab_bc_value_l646_646839

theorem max_f_value : 
  let m := 2 in
  ∀ x : ℝ, max (|x - 1| - 2 * |x + 1|) m := sorry

theorem max_ab_bc_value :
  let m := 2 in
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + 2 * b^2 + c^2 = m → 
  max (a * b + b * c) (1 : ℝ) := sorry

end max_f_value_max_ab_bc_value_l646_646839


namespace total_profit_is_correct_l646_646644

-- Definitions of the investments
def A_initial_investment : ℝ := 12000
def B_investment : ℝ := 16000
def C_investment : ℝ := 20000
def D_investment : ℝ := 24000
def E_investment : ℝ := 18000
def C_profit_share : ℝ := 36000

-- Definitions of the time periods (in months)
def time_6_months : ℝ := 6
def time_12_months : ℝ := 12

-- Calculations of investment-months for each person
def A_investment_months : ℝ := A_initial_investment * time_6_months
def B_investment_months : ℝ := B_investment * time_12_months
def C_investment_months : ℝ := C_investment * time_12_months
def D_investment_months : ℝ := D_investment * time_12_months
def E_investment_months : ℝ := E_investment * time_6_months

-- Calculation of total investment-months
def total_investment_months : ℝ :=
  A_investment_months + B_investment_months + C_investment_months +
  D_investment_months + E_investment_months

-- The main theorem stating the total profit calculation
theorem total_profit_is_correct :
  ∃ TP : ℝ, (C_profit_share / C_investment_months) = (TP / total_investment_months) ∧ TP = 135000 :=
by
  sorry

end total_profit_is_correct_l646_646644


namespace sequence_no_rational_squares_l646_646825

open Nat

noncomputable def a : ℕ → ℚ
| 0       := 2016
| (n + 1) := a n + 2 / a n

theorem sequence_no_rational_squares : ∀ n, ∀ q : ℚ, a n ≠ q^2 := by
  sorry

end sequence_no_rational_squares_l646_646825


namespace sin_alpha_solution_l646_646263

theorem sin_alpha_solution (α β : ℝ) 
  (h1 : 1 - real.cos α - real.cos β + real.sin α * real.cos β = 0)
  (h2 : 1 + real.cos α - real.sin β + real.sin α * real.cos β = 0) :
  real.sin α = (1 - real.sqrt 10) / 3 :=
by
  sorry

end sin_alpha_solution_l646_646263


namespace attendees_count_l646_646906

theorem attendees_count (k : ℕ) (n : ℕ) 
  (n_eq : n = 12 * k) 
  (shakes_hands : ∀ p : ℕ, p < n → (handshakes_with : ℕ) := 3 * k + 6) 
  (common_handshakes : ∀ a b : ℕ, a < n → b < n → (common_shakes : ℕ)) :
  n = 24 := 
sorry

end attendees_count_l646_646906


namespace cosine_F_in_triangle_DEF_l646_646419

theorem cosine_F_in_triangle_DEF
  (D E F : ℝ)
  (h_triangle : D + E + F = π)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = - (16 / 65) := by
  sorry

end cosine_F_in_triangle_DEF_l646_646419


namespace average_k_l646_646304

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l646_646304


namespace find_integer_solutions_l646_646234

theorem find_integer_solutions :
  ∀ (x y : ℕ), 0 < x → 0 < y → (2 * x^2 + 5 * x * y + 2 * y^2 = 2006 ↔ (x = 28 ∧ y = 3) ∨ (x = 3 ∧ y = 28)) :=
by
  sorry

end find_integer_solutions_l646_646234


namespace inequality_proof_l646_646835

variable {α : Type}
variables {n k : ℕ} {a b : α}

theorem inequality_proof 
  (n : ℕ) 
  (a : Finₓ n → ℝ)
  (h1 : ∀ i, 0 < a i)
  (h2 : ∑ i in Finset.range n, a i < 1) :
  (∏ i in Finset.range n, a i) * (1 - ∑ i in Finset.range n, a i) / 
  ((∑ i in Finset.range n, a i) * ∏ i in Finset.range n, (1 - a i)) ≤ 
  (1 : ℝ) / n^(n + 1) :=
begin
  sorry
end

end inequality_proof_l646_646835


namespace simplify_fraction_150_div_225_l646_646039

theorem simplify_fraction_150_div_225 :
  let a := 150
  let b := 225
  let gcd_ab := Nat.gcd a b
  let num_fact := 2 * 3 * 5^2
  let den_fact := 3^2 * 5^2
  gcd_ab = 75 →
  num_fact = a →
  den_fact = b →
  (a / gcd_ab) / (b / gcd_ab) = (2 / 3) :=
  by
    intros 
    sorry

end simplify_fraction_150_div_225_l646_646039


namespace more_than_half_palindromes_sum_non_palindrome_l646_646896

theorem more_than_half_palindromes_sum_non_palindrome : 
∀ (palindromes : Finset ℕ), 
(∀ p ∈ palindromes, is_palindrome p) ∧ 
(∀ p q ∈ palindromes, p ≠ q → ¬ is_palindrome (p + q)) → 
palindromes.card ≤ 45 := 
begin
  sorry
end

def is_palindrome (n : ℕ) : Prop := 
  let digits := Int.digits 10 (to_digit n) in
  list.reverse digits = digits

end more_than_half_palindromes_sum_non_palindrome_l646_646896


namespace matrix_identity_l646_646838

-- Define the matrix N
def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 3* y, z], ![x, 2 * y, -z], ![2 * x, -y, z]]

-- Hypothesis that N^T * N is the identity matrix
theorem matrix_identity (x y z : ℝ) (h : (N x y z)ᵀ * N x y z = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :
  x^2 + y^2 + z^2 = 127 / 210 :=
by
  sorry

end matrix_identity_l646_646838


namespace count_integers_of_form_3bcd_l646_646217

theorem count_integers_of_form_3bcd (b c d : ℕ) :
  (3000 ≤ 3000 + b * 100 + c * 10 + d) ∧ (3000 + b * 100 + c * 10 + d < 4000) ∧
  (b + c = d) ∧ (0 ≤ b ∧ b < 10) ∧ (0 ≤ c ∧ c < 10) ∧ (0 ≤ d ∧ d < 10) → 
  ∑ i in {0..9}, (9 - i + 1) = 55 :=
by {
  sorry
}

end count_integers_of_form_3bcd_l646_646217


namespace Marty_combinations_l646_646017

theorem Marty_combinations : 
  let colors := 4
  let decorations := 3
  colors * decorations = 12 :=
by
  sorry

end Marty_combinations_l646_646017


namespace average_score_l646_646183

theorem average_score (N : ℕ) (p3 p2 p1 p0 : ℕ) (n : ℕ) 
  (H1 : N = 3)
  (H2 : p3 = 30)
  (H3 : p2 = 50)
  (H4 : p1 = 10)
  (H5 : p0 = 10)
  (H6 : n = 20)
  (H7 : p3 + p2 + p1 + p0 = 100) :
  (3 * (p3 * n / 100) + 2 * (p2 * n / 100) + 1 * (p1 * n / 100) + 0 * (p0 * n / 100)) / n = 2 :=
by 
  sorry

end average_score_l646_646183


namespace group_of_friends_l646_646583

theorem group_of_friends (n m : ℕ) (h : (n - 1) * m = 15) : 
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by 
  have h_cases : (
    ∃ k, k = (n - 1) ∧ k * m = 15 ∧ (k = 1 ∨ k = 3 ∨ k = 5 ∨ k = 15)
  ) := 
  sorry
  cases h_cases with k hk,
  cases hk with hk1 hk2,
  cases hk2 with hk2_cases hk2_valid_cases,
  cases hk2_valid_cases,
  { -- case 1: k = 1/ (n-1 = 1), and m = 15
    subst k,
    have h_m_valid : m = 15 := hk2_valid_cases,
    subst h_m_valid,
    left,
    calc 
    n * 15 = (1 + 1) * 15 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  },
  { -- case 2: k = 3 / (n-1 = 3), and m = 5
    subst k,
    have h_m_valid : m = 5 := hk2_valid_cases,
    subst h_m_valid,
    right,
    left,
    calc 
    n * 5 = (3 + 1) * 5 : by {simp, exact rfl}
    ... = 20 : by {norm_num}
  },
  { -- case 3: k = 5 / (n-1 = 5), and m = 3,
    subst k,
    have h_m_valid : m = 3 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    left,
    calc 
    n * 3 = (5 + 1) * 3 : by {simp, exact rfl}
    ... = 18 : by {norm_num}
  },
  { -- case 4: k = 15 / (n-1 = 15), and m = 1
    subst k,
    have h_m_valid : m = 1 := hk2_valid_cases,
    subst h_m_valid,
    right,
    right,
    right,
    calc 
    n * 1 = (15 + 1) * 1 : by {simp, exact rfl}
    ... = 16 : by {norm_num}
  }

end group_of_friends_l646_646583


namespace range_of_m_l646_646357

noncomputable def f (x m : ℝ) : ℝ := |x^2 - 4| + x^2 + m * x

theorem range_of_m 
  (f_has_two_distinct_zeros : ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 3 ∧ f a m = 0 ∧ f b m = 0) :
  -14 / 3 < m ∧ m < -2 :=
sorry

end range_of_m_l646_646357


namespace combined_purchase_savings_is_zero_l646_646181

def price_per_window : ℕ := 100

def windows_discount (n : ℕ) : ℕ := (n / 3) + n

def cost (n : ℕ) (price : ℕ) : ℕ := 
  let discounted_window := windows_discount n in
  ((n + discounted_window - n) * price)

theorem combined_purchase_savings_is_zero :
  let dave_windows := 10
  let doug_windows := 12
  let total_windows := dave_windows + doug_windows
  let dave_cost := cost dave_windows price_per_window
  let doug_cost := cost doug_windows price_per_window
  let combined_cost := cost total_windows price_per_window
  (dave_cost + doug_cost - combined_cost) = 0 := 
begin
  -- Definitions and provided conditions
  sorry
end

end combined_purchase_savings_is_zero_l646_646181


namespace smallest_multiple_l646_646246

open Nat

theorem smallest_multiple :
  ∃ n : ℕ, 0 < n ∧ (15 ∣ n) ∧ (20 ∣ n) ∧ (18 ∣ n) ∧ ∀ m : ℕ, 0 < m ∧ (15 ∣ m) ∧ (20 ∣ m) ∧ (18 ∣ m) → n ≤ m :=
begin
  use 180,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm h15m h20m h18m,
    rcases hm with ⟨h_pos, h_m⟩,
    sorry
  }
end

end smallest_multiple_l646_646246


namespace number_of_friends_l646_646616

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l646_646616


namespace octahedron_side_length_l646_646991

theorem octahedron_side_length 
  (side_length_cube : ℝ) 
  (Q1 Q2 Q3 Q4 Q1' Q2' Q3' Q4' : ℝ × ℝ × ℝ)
  (c1 : side_length_cube = 2)
  (c2 : Q1 = (0, 0, 0))
  (c3 : Q1' = (2, 2, 2))
  (c4 : ∀ i ∈ [Q2, Q3, Q4], (i.1 - Q1.1)^2 + (i.2 - Q1.2)^2 + (i.3 - Q1.3)^2 = 2^2)
  (c5 : ∀ i, Q1 + i = Q1' → (Q1.1 + i)x(yz).1 = Q1'.1 ∧ (Q1.2 + i)x(yz).2 = Q1'.2 ∧ (Q1.3 + i)x(yz).3 = Q1'.3)
  (c6 : ∀ (seg: ℝ × ℝ × ℝ), seg ∈ [ (Q1, Q2), (Q1, Q3), (Q1, Q4), (Q1', Q2'), (Q1', Q3'), (Q1', Q4') ]) :
    ¬((s : ∃ side_length_octahedron ℝ = 2))
sorry

end octahedron_side_length_l646_646991


namespace count_primes_in_sequence_l646_646487

def sequence_term (n : ℕ) : ℕ :=
  let k := Nat.log10 (5^n) + 1
  (5^n) * (10^k) + 37

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def count_primes_in_sequence_up_to (N : ℕ) : ℕ :=
  (List.range N).countp (λ n, is_prime (sequence_term n))

theorem count_primes_in_sequence : count_primes_in_sequence_up_to 1 = 1 :=
by
  -- Skip the actual proof as per instructions
  sorry

end count_primes_in_sequence_l646_646487


namespace problem_statement_l646_646828

noncomputable def a : ℝ := log 5 4
noncomputable def b : ℝ := (log 5 3) ^ 2
noncomputable def c : ℝ := log 4 5

theorem problem_statement : b < a ∧ a < c := 
by sorry

end problem_statement_l646_646828


namespace quadratic_inequality_solution_l646_646156

theorem quadratic_inequality_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m ≤ 0) ↔ m ≤ 1 :=
sorry

end quadratic_inequality_solution_l646_646156


namespace positive_difference_abs_eq_15_l646_646954

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l646_646954


namespace distribute_neg3_l646_646968

theorem distribute_neg3 (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y :=
by sorry

end distribute_neg3_l646_646968


namespace square_area_l646_646786

open Real

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  (√((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2 + (q.3 - p.3) ^ 2))

noncomputable def area_of_square (A O : ℝ × ℝ × ℝ) : ℝ :=
  let d := distance A O in
  let s := d / (√2) in
  s ^ 2

theorem square_area (A O : ℝ × ℝ × ℝ)
  (hA : A = (-6, -4, 2))
  (hO : O = (3, 2, -1)) :
  area_of_square A O = 63 := by
  sorry

end square_area_l646_646786


namespace sum_first_99_natural_numbers_l646_646658

theorem sum_first_99_natural_numbers :
  let a1 := 1
  let a99 := 99
  let d := 1
  let n := 99
  (∑ k in finset.range n, k + 1) = 4950 :=
by
  sorry

end sum_first_99_natural_numbers_l646_646658


namespace average_marks_passed_boys_l646_646874

theorem average_marks_passed_boys (N : ℕ) (total_avg : ℕ) (num_passed : ℕ) (num_failed : ℕ) (failed_avg : ℕ) (marks_passed_avg: ℕ):
    N = 120 →
    total_avg = 38 →
    num_passed = 115 →
    num_failed = 5 →
    failed_avg = 15 →
    (total_avg * N = marks_passed_avg * num_passed + failed_avg * num_failed) →
    marks_passed_avg = 39 :=
by
  intros hN hTotalAvg hNumPassed hNumFailed hFailedAvg hEquation
  calc
    marks_passed_avg = 4485 / 115 : sorry
               ... = 39         : sorry
  sorry

end average_marks_passed_boys_l646_646874


namespace one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646919

theorem one_fiftieth_digit_of_five_over_thirteen_is_five :
  (decimal_fraction 5 13).digit 150 = 5 :=
by sorry

end one_fiftieth_digit_of_five_over_thirteen_is_five_l646_646919


namespace new_person_weight_l646_646875

theorem new_person_weight 
  (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) 
  (h1: n = 8) (h2 : avg_increase = 4.2) (h3 : old_weight = 65) :
  old_weight + n * avg_increase = 98.6 :=
by {
  rw [h1, h2, h3],
  norm_num,
  -- 65 + 8 * 4.2 = 98.6
}

end new_person_weight_l646_646875


namespace ways_to_turn_off_lights_l646_646402

-- Define the problem conditions
def streetlights := 12
def can_turn_off := 3
def not_turn_off_at_ends := true
def not_adjacent := true

-- The theorem to be proved
theorem ways_to_turn_off_lights : 
  ∃ n, 
  streetlights = 12 ∧ 
  can_turn_off = 3 ∧ 
  not_turn_off_at_ends ∧ 
  not_adjacent ∧ 
  n = 56 :=
by 
  sorry

end ways_to_turn_off_lights_l646_646402


namespace positive_difference_solutions_abs_eq_30_l646_646937

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l646_646937


namespace kobe_initial_order_l646_646810

theorem kobe_initial_order : 
  ∀ (K : ℕ), 
  (let P := 2 * K in 
   P + P = 20) → 
  K = 5 :=
by
  sorry

end kobe_initial_order_l646_646810


namespace count_n_square_of_integer_l646_646708

theorem count_n_square_of_integer : 
  (cardinal.mk {n : ℤ | ∃ k : ℤ, (n / (30 - n) = k^2) ∧ 0 ≤ n ∧ n < 30}) = 3 := 
sorry

end count_n_square_of_integer_l646_646708


namespace pentagon_area_l646_646635

theorem pentagon_area (a b c d e : ℤ) (O : 31 * 25 = 775) (H : 12^2 + 5^2 = 13^2) 
  (rect_side_lengths : (a, b, c, d, e) = (13, 19, 20, 25, 31)) :
  775 - 1/2 * 12 * 5 = 745 := 
by
  sorry

end pentagon_area_l646_646635


namespace ratio_of_areas_of_trapezoid_to_triangle_l646_646519

/-- Given an equilateral triangle AHI, such that lines BC, DE, and FG are parallel to HI,
    and AB = BD = DF = FH = (1/3) * AH, the ratio of the area of trapezoid FGIH to the area of triangle AHI is 3/4. -/
theorem ratio_of_areas_of_trapezoid_to_triangle
  (A H I B C D E F G : Point)
  (h_triangle : equilateral_triangle A H I)
  (h_parallel_BC_DE_FG_HI : parallel BC HI ∧ parallel DE HI ∧ parallel FG HI)
  (h_segment : AB = BD ∧ BD = DF ∧ DF = FH ∧ AB = (1/3) * AH) :
  (area FGIH) / (area AHI) = 3 / 4 :=
sorry

end ratio_of_areas_of_trapezoid_to_triangle_l646_646519


namespace area_of_shaded_region_l646_646177

-- Define the vertices based on the problem's conditions
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (0, 12)
def C : (ℝ × ℝ) := (12, 12)
def D : (ℝ × ℝ) := (12, 0)
def E : (ℝ × ℝ) := (12, 0)
def F : (ℝ × ℝ) := (24, 0)
def G : (ℝ × ℝ) := (12, 12)

-- Define the intersection point H of the line BF with the line EG
def H : (ℝ × ℝ) := (12, 6)

-- Derive the area of triangle EDH
def area_EDH : ℝ := (1 / 2) * (E.1 - D.1) * (H.2 - D.2)

-- The theorem to prove
theorem area_of_shaded_region : area_EDH = 36 := by
  sorry

end area_of_shaded_region_l646_646177


namespace average_k_positive_int_roots_l646_646339

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l646_646339


namespace sphere_surface_area_l646_646345

theorem sphere_surface_area
  (V : ℝ)
  (r : ℝ)
  (h : ℝ)
  (R : ℝ)
  (V_cone : V = (2 * π) / 3)
  (r_cone_base : r = 1)
  (cone_height : h = 2 * V / (π * r^2))
  (sphere_radius : R^2 - (R - h)^2 = r^2):
  4 * π * R^2 = 25 * π / 4 :=
by
  sorry

end sphere_surface_area_l646_646345


namespace xq_xh_meet_probability_l646_646973

noncomputable def prob_meet (x y : ℝ) : Prop :=
  80 < x ∧ x < 120 ∧ 60 < y ∧ y < 120 ∧ abs (x - y) ≤ 10

noncomputable def prob_space (x y : ℝ) : Prop :=
  80 < x ∧ x < 120 ∧ 60 < y ∧ y < 120

theorem xq_xh_meet_probability :
  (measure_theory.measure_of (set_of (λ (xy : ℝ × ℝ), prob_meet xy.fst xy.snd)) /
   measure_theory.measure_of (set_of (λ (xy : ℝ × ℝ), prob_space xy.fst xy.snd))
  ) = 5 / 16 :=
by sorry

end xq_xh_meet_probability_l646_646973


namespace absolute_value_expression_l646_646665

theorem absolute_value_expression : real :=
  ∀ (e : real), e = real.exp 1 → | e - | e - 5 | | = 2 * e - 5 :=
by
  intros e heq
  sorry

end absolute_value_expression_l646_646665


namespace exist_set_with_divisible_products_l646_646680

theorem exist_set_with_divisible_products :
  ∃ (S : Finset ℕ), S.card = 2020 ∧
  (∀ (J : Finset ℕ), J.card = 101 → J ⊆ S → (∏ x in J, x) % (∑ x in J, x) = 0) :=
sorry

end exist_set_with_divisible_products_l646_646680


namespace rank_from_left_l646_646182

variable (total_students rank_right : ℕ)

-- Given conditions
axiom total_students_value : total_students = 20
axiom rank_right_value : rank_right = 13

-- The theorem to be proved
theorem rank_from_left : total_students = rank_right + 8 - 1 := by
  rw [total_students_value, rank_right_value]
  sorry

end rank_from_left_l646_646182


namespace marie_saves_money_in_17_days_l646_646016

noncomputable def number_of_days_needed (cash_register_cost revenue tax_rate costs : ℝ) : ℕ := 
  let net_revenue := revenue / (1 + tax_rate) 
  let daily_profit := net_revenue - costs
  Nat.ceil (cash_register_cost / daily_profit)

def marie_problem_conditions : Prop := 
  let bread_daily_revenue := 40 * 2
  let bagels_daily_revenue := 20 * 1.5
  let cakes_daily_revenue := 6 * 12
  let muffins_daily_revenue := 10 * 3
  let daily_revenue := bread_daily_revenue + bagels_daily_revenue + cakes_daily_revenue + muffins_daily_revenue
  let fixed_daily_costs := 20 + 2 + 80 + 30
  fixed_daily_costs = 132 ∧ daily_revenue = 212 ∧ 8 / 100 = 0.08

theorem marie_saves_money_in_17_days : marie_problem_conditions → number_of_days_needed 1040 212 0.08 132 = 17 := 
by 
  intro h
  -- Proof goes here.
  sorry

end marie_saves_money_in_17_days_l646_646016


namespace expression_value_l646_646762

theorem expression_value (x y : ℝ) (h : x + y = -1) : 
  x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l646_646762


namespace _l646_646271

def data : List ℝ := [4.7, 4.8, 5.1, 5.4, 5.5]

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

example : variance data = 0.1 :=
by
  -- note that we are only stating the theorem here, hence adding 'sorry'
  sorry

end _l646_646271


namespace gcd_90_450_l646_646697

theorem gcd_90_450 : Nat.gcd 90 450 = 90 := by
  sorry

end gcd_90_450_l646_646697


namespace maximum_area_triangle_ABF_l646_646036

noncomputable def circle_parametric_eq := 
  (α : ℝ) → ℝ × ℝ := λ α, (3 + 2 * Math.sin α, -4 + 2 * Math.cos α)

def A := (-2 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 2 : ℝ)
def distance (p1 p2 : ℝ × ℝ) := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def line_eq (x y : ℝ) := x - y + 2 = 0

theorem maximum_area_triangle_ABF :
  ∃ (F : ℝ × ℝ), 
    F ∈ setOf (circle_parametric_eq) ∧ 
    line_eq F.1 F.2 ∧ 
    ∀ G, 
      G ∈ setOf (circle_parametric_eq) →
      Real.Area (A, B, G) ≤ 9 + 2 * Real.sqrt 2 :=
sorry

end maximum_area_triangle_ABF_l646_646036


namespace number_of_hens_l646_646685

-- Definitions based on conditions
def total_eggs : ℝ := 303.0
def eggs_per_hen : ℝ := 10.82142857

-- Theorem statement
theorem number_of_hens : total_eggs / eggs_per_hen ≈ 28 := 
by sorry

end number_of_hens_l646_646685


namespace coins_in_fifth_piggy_bank_l646_646143

theorem coins_in_fifth_piggy_bank :
  ∀ (first second third fourth sixth : ℕ),
    first = 72 → 
    second = 81 → 
    third = 90 → 
    fourth = 99 → 
    sixth = 117 →
    (second - first = 9) →
    (third - second = 9) →
    (fourth - third = 9) →
    (sixth - fourth = 18) →
    (∃ fifth : ℕ, fourth + 9 = fifth ∧ fifth = 108) :=
by
  intros first second third fourth sixth h1 h2 h3 h4 h5 h6 h7 h8 h9
  use (fourth + 9)
  split
  · exact h8
  · rw [←h4]
    exact h8
  sorry

end coins_in_fifth_piggy_bank_l646_646143


namespace other_root_of_quadratic_l646_646382

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) (h_root : 4 * a * 0^2 - 2 * a * 0 + c = 0) :
  ∃ t : ℝ, (4 * a * t^2 - 2 * a * t + c = 0) ∧ t = 1 / 2 :=
by
  sorry

end other_root_of_quadratic_l646_646382


namespace range_of_x_l646_646265

theorem range_of_x (f : ℝ → ℝ) (hf_cont : continuous f)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_increasing : ∀ x y, 0 < x → x < y → f x < f y)
  (hf_ineq : ∀ x, f x > f (2 - x)) : ∀ x, x > 1 :=
by
  sorry

end range_of_x_l646_646265


namespace smol_sum_eq_14_l646_646215

def smol_function (x : ℝ) := Real.sqrt (x * x^3)

theorem smol_sum_eq_14 : smol_function 1 + smol_function 2 + smol_function 3 = 14 := by
  -- Proof steps would go here, but adding sorry to skip proof as instructed
  sorry

end smol_sum_eq_14_l646_646215


namespace total_washing_time_l646_646970

/-- Define the time taken for different parts of washing a normal car -/
def time_wash_windows := 4
def time_wash_body := 7
def time_clean_tires := 4
def time_wax_car := 9

/-- Define the number of each type of vehicle washed -/
def num_normal_cars := 3
def num_big_suvs := 2
def num_minivans := 1

/-- Define the time multipliers for big SUVs and minivans compared to normal cars -/
def big_suv_multiplier := 2
def minivan_multiplier := 1.5

/-- Define the break time between each vehicle washed -/
def break_time := 5
def num_vehicles := num_normal_cars + num_big_suvs + num_minivans
def num_breaks := num_vehicles - 1

/-- Prove that the total time spent washing all vehicles and taking breaks is 229 minutes -/
theorem total_washing_time : 
  let normal_car_time := time_wash_windows + time_wash_body + time_clean_tires + time_wax_car in
  let big_suv_time := big_suv_multiplier * normal_car_time in
  let minivan_time := minivan_multiplier * normal_car_time in
  let total_washing_time := 
    (num_normal_cars * normal_car_time) + 
    (num_big_suvs * big_suv_time) + 
    (num_minivans * minivan_time) in
  let total_break_time := num_breaks * break_time in
  let total_time := total_washing_time + total_break_time in
  total_time = 229 :=
by
  sorry

end total_washing_time_l646_646970


namespace find_vec_a_l646_646729

def vec_a (x y : ℝ) : ℝ × ℝ × ℝ := (x, 2 * y - 1, -1 / 4)
def vec_b : ℝ × ℝ × ℝ := (-1, 2, 1)
def vec_c : ℝ × ℝ × ℝ := (3, 1 / 2, -2)

theorem find_vec_a (x y : ℝ)
  (h1 : vec_a x y = (-9 / 52, 1 / 26, -1 / 4))
  (h2 : inner (x, 2 * y - 1, -1 / 4) (-1, 2, 1) = 0)
  (h3 : inner (x, 2 * y - 1, -1 / 4) (3, 1 / 2, -2) = 0) :
  vec_a x y = (-9 / 52, 1 / 26, -1 / 4) := by
  sorry

end find_vec_a_l646_646729


namespace neighbors_receive_mangoes_l646_646845

-- Definitions of the conditions
def harvested_mangoes : ℕ := 560
def sold_mangoes : ℕ := harvested_mangoes / 2
def given_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Calculation of mangoes left
def mangoes_left : ℕ := harvested_mangoes - sold_mangoes - given_to_family

-- The statement we want to prove
theorem neighbors_receive_mangoes : mangoes_left / num_neighbors = 19 := by
  sorry

end neighbors_receive_mangoes_l646_646845


namespace memory_image_regression_l646_646567

noncomputable def mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length)

noncomputable def regression_intercept (mean_x mean_y : ℝ) (slope : ℝ) : ℝ :=
  mean_y - slope * mean_x

theorem memory_image_regression :
  let memory_capacity : List ℝ := [4, 6, 8, 10];
  let image_recognition : List ℝ := [3, 5, 6, 8];
  let mean_x := mean memory_capacity;
  let mean_y := mean image_recognition;
  let slope := 0.8;
  let intercept := regression_intercept mean_x mean_y slope;
  let regression_eq (x : ℝ) : ℝ := slope * x + intercept;
  regression_eq 12 = 9.5 :=
by
  let memory_capacity : List ℝ := [4, 6, 8, 10];
  let image_recognition : List ℝ := [3, 5, 6, 8];
  let mean_x := mean memory_capacity;
  let mean_y := mean image_recognition;
  let slope := 0.8;
  let intercept := regression_intercept mean_x mean_y slope;
  let regression_eq (x : ℝ) : ℝ := slope * x + intercept;
  have : mean_x = 7 := by sorry;
  have : mean_y = 5.5 := by sorry;
  have : intercept = -0.1 := by sorry;
  show regression_eq 12 = 9.5 from by sorry

end memory_image_regression_l646_646567


namespace investment_amount_is_correct_l646_646650

noncomputable def investment_principal
  (PMT : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  PMT * (1 - (1 + r / n) ^ (-n * t)) / (r / n)

theorem investment_amount_is_correct :
  investment_principal 234 0.09 12 1 ≈ 2607.47 := 
by
  sorry

end investment_amount_is_correct_l646_646650


namespace exists_periodic_function_l646_646193

def periodic_function_example (f : ℝ → ℝ) : Prop :=
  f = (λ x, Real.cos (π * x / 6) + Real.sin (π * x / 6))

theorem exists_periodic_function (f : ℝ → ℝ) (h : ∀ x, f (x + 1) + f (x - 1) = Real.sqrt 3 * f x) :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ periodic_function_example f :=
sorry

end exists_periodic_function_l646_646193


namespace decrease_in_temperature_l646_646380

theorem decrease_in_temperature (increase_temp : ℤ → Prop) :
  (increase_temp 2 → -3 = -3) :=
by
  intro h
  exact eq.refl (-3)

end decrease_in_temperature_l646_646380


namespace problem1_xy_value_problem2_min_value_l646_646029

-- Define the first problem conditions
def problem1 (x y : ℝ) : Prop :=
  x^2 - 2 * x * y + 2 * y^2 + 6 * y + 9 = 0

-- Prove that xy = 9 given the above condition
theorem problem1_xy_value (x y : ℝ) (h : problem1 x y) : x * y = 9 :=
  sorry

-- Define the second problem conditions
def expression (m : ℝ) : ℝ :=
  m^2 + 6 * m + 13

-- Prove that the minimum value of the expression is 4
theorem problem2_min_value : ∃ m, expression m = 4 :=
  sorry

end problem1_xy_value_problem2_min_value_l646_646029


namespace six_people_with_two_between_l646_646481

theorem six_people_with_two_between (people : Fin 6) (A B : Fin 6) :
  ∃ (ways : ℕ), ways = 144 ∧ 
  ∃ (f : Fin 6 → Fin 6), 
    (f ≠ A) ∧ (f ≠ B) ∧ 
    (∃ (g: Fin 4 → Fin 4), g ≠ f) := sorry

end six_people_with_two_between_l646_646481


namespace evaluate_sum_of_powers_of_i_l646_646686

theorem evaluate_sum_of_powers_of_i (i : ℂ) (h : i^4 = 1) : i^5 + i^{13} + i^{-7} = i :=
by {
  sorry
}

end evaluate_sum_of_powers_of_i_l646_646686


namespace hyeyoung_walked_correct_l646_646893

/-- The length of the promenade near Hyeyoung's house is 6 kilometers (km). -/
def promenade_length : ℕ := 6

/-- Hyeyoung walked from the starting point to the halfway point of the trail. -/
def hyeyoung_walked : ℕ := promenade_length / 2

/-- The distance Hyeyoung walked is 3 kilometers (km). -/
theorem hyeyoung_walked_correct : hyeyoung_walked = 3 := by
  sorry

end hyeyoung_walked_correct_l646_646893


namespace zero_point_interval_l646_646346

def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

theorem zero_point_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 :=
sorry

end zero_point_interval_l646_646346


namespace triangular_region_area_l646_646186

noncomputable def line1 (x : ℝ) : ℝ := (2 / 3) * x + 4
noncomputable def line2 (x : ℝ) : ℝ := -3 * x + 9
noncomputable def line3 (y : ℝ) : Prop := y = 2

theorem triangular_region_area : 
  let intersection1 := ((-3 : ℝ), (2 : ℝ)),
      intersection2 := ((7 / 3 : ℝ), (2 : ℝ)),
      intersection3 := ((15 / 11 : ℝ), (54 / 11 : ℝ))
  in
  let base := (16 / 3 : ℝ),
      height := (32 / 11 : ℝ)
  in 
  (1 / 2) * base * height = 7.76 :=
by
  sorry

end triangular_region_area_l646_646186


namespace triple_sum_equals_seven_l646_646909

theorem triple_sum_equals_seven {k m n : ℕ} (hk : 0 < k) (hm : 0 < m) (hn : 0 < n)
  (hcoprime : Nat.gcd k m = 1 ∧ Nat.gcd k n = 1 ∧ Nat.gcd m n = 1)
  (hlog : k * Real.log 5 / Real.log 400 + m * Real.log 2 / Real.log 400 = n) :
  k + m + n = 7 := by
  sorry

end triple_sum_equals_seven_l646_646909


namespace part_I_extreme_value_part_II_monotonically_increasing_l646_646837

def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

-- Part (I)
theorem part_I_extreme_value (a : ℝ) (h : a = -1/2) : 
  ∃ x : ℝ, f a x = 0 :=
by
  sorry

-- Part (II)
theorem part_II_monotonically_increasing (a : ℝ) : 
  (∀ x : ℝ, 0 < x → 0 ≤ (2 * a * x + 1 - Real.log x) / x^2) → (a ≥ 1/2 * Real.exp (-2)) :=
by
  sorry

end part_I_extreme_value_part_II_monotonically_increasing_l646_646837


namespace pct_three_petals_is_75_l646_646808

-- Given Values
def total_clovers : Nat := 200
def pct_two_petals : Nat := 24
def pct_four_petals : Nat := 1

-- Statement: Prove that the percentage of clovers with three petals is 75%
theorem pct_three_petals_is_75 :
  (100 - pct_two_petals - pct_four_petals) = 75 := by
  sorry

end pct_three_petals_is_75_l646_646808


namespace possible_number_of_friends_l646_646627

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l646_646627


namespace domain_of_function_l646_646694

theorem domain_of_function :
  {x : ℝ | (x + 1 ≥ 0) ∧ (2 - x ≠ 0)} = {x : ℝ | -1 ≤ x ∧ x ≠ 2} :=
by {
  sorry
}

end domain_of_function_l646_646694


namespace average_of_k_l646_646318

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l646_646318


namespace total_weight_of_oranges_l646_646910

theorem total_weight_of_oranges :
  let capacity1 := 80
  let capacity2 := 50
  let capacity3 := 60
  let filled1 := 3 / 4
  let filled2 := 3 / 5
  let filled3 := 2 / 3
  let weight_per_orange1 := 0.25
  let weight_per_orange2 := 0.30
  let weight_per_orange3 := 0.40
  let num_oranges1 := capacity1 * filled1
  let num_oranges2 := capacity2 * filled2
  let num_oranges3 := capacity3 * filled3
  let total_weight1 := num_oranges1 * weight_per_orange1
  let total_weight2 := num_oranges2 * weight_per_orange2
  let total_weight3 := num_oranges3 * weight_per_orange3
  total_weight1 + total_weight2 + total_weight3 = 40 := by
  sorry

end total_weight_of_oranges_l646_646910


namespace sin_addition_formula_example_l646_646221

  theorem sin_addition_formula_example :
    sin 14 * cos 16 + cos 14 * sin 16 = 1 / 2 := by
  sorry
  
end sin_addition_formula_example_l646_646221


namespace find_k_l646_646166

-- Defining the conditions
variable {k : ℕ} -- k is a non-negative integer

-- Given conditions as definitions
def green_balls := 7
def purple_balls := k
def total_balls := green_balls + purple_balls
def win_amount := 3
def lose_amount := -1

-- Defining the expected value equation
def expected_value [fact (total_balls > 0)] : ℝ :=
  (green_balls.toℝ / total_balls.toℝ * win_amount) +
  (purple_balls.toℝ / total_balls.toℝ * lose_amount)

-- The required theorem/assertion to prove
theorem find_k (hk : k > 0) (h : expected_value = 1) : k = 7 :=
sorry

end find_k_l646_646166


namespace locus_of_intersection_point_l646_646272

theorem locus_of_intersection_point (A B C D F M : Point) 
  (h_ext_bisector : external_angle_bisector A B C D F)
  (h_dist_relation : dist A D * dist A F = dist A B * dist A C) :
  ∃ O : Point, is_circle (circle O B C) ∧ M ∈ circle O B C ∧
  central_angle (circle O B C) B C = (∠BAC + ∠BCA) / 2 :=
sorry

end locus_of_intersection_point_l646_646272


namespace max_sundays_in_50_days_l646_646110

theorem max_sundays_in_50_days : 
  ∀ (start_day : ℕ), 
  0 ≤ start_day ∧ start_day < 7 →
  ∃ (n : ℕ), n = 7 ∧ (\( \text{sundays_in_first_50_days} \)) = n :=
by 
  sorry

end max_sundays_in_50_days_l646_646110


namespace friend_spent_seven_l646_646542

/-- You and your friend spent a total of $11 for lunch.
    Your friend spent $3 more than you.
    Prove that your friend spent $7 on their lunch. -/
theorem friend_spent_seven (you friend : ℝ) 
  (h1: you + friend = 11) 
  (h2: friend = you + 3) : 
  friend = 7 := 
by 
  sorry

end friend_spent_seven_l646_646542


namespace distinguishable_colorings_tetrahedron_l646_646223

/-- Considering faces of a regular tetrahedron each can be painted one of four colors
    (red, white, blue, or green). Two colorings are indistinguishable if the tetrahedra are
    congruent by rotations. The number of distinguishable colorings of the tetrahedron is 35. -/
theorem distinguishable_colorings_tetrahedron : 
  let colors := 4 in let faces := 4 in 
  ∃ number_of_colorings : ℕ,
    number_of_colorings = 35 := sorry

end distinguishable_colorings_tetrahedron_l646_646223


namespace smallest_constant_c_inequality_l646_646275

theorem smallest_constant_c_inequality (n : ℕ) (h : n ≥ 2) :
  ∃ (c : ℝ), (∀ (x : Fin n → ℝ), (∀ i, 0 ≤ x i) →
    (∑ i j in Finset.range (n + 1), x i * x j * (x i ^ 2 + x j ^ 2)) ≤ c * (∑ i in Finset.range (n + 1), x i) ^ 4)
  ∧ c = 1 / 8 := sorry

end smallest_constant_c_inequality_l646_646275


namespace angle_neg_a_c_l646_646383

variables {V : Type*} [inner_product_space ℝ V] (a b c : V) 

-- Condition 1: The angle between vectors a and b is 70 degrees
def angle_a_b : real.angle := 70 * real.angle.pi / 180 

-- Condition 2: The angle between vectors b and c is 50 degrees
def angle_b_c : real.angle := 50 * real.angle.pi / 180 

-- Prove the angle between -a and c is 60 degrees
theorem angle_neg_a_c : inner_product_space.angle (-a) c = 60 * real.angle.pi / 180 :=
sorry

end angle_neg_a_c_l646_646383


namespace identify_150th_digit_l646_646929

def repeating_sequence : List ℕ := [3, 8, 4, 6, 1, 5]

theorem identify_150th_digit :
  (150 % 6 = 0) →
  nth repeating_sequence 5 = 5 :=
by
  intros h
  rewrite_modulo h
  rfl

end identify_150th_digit_l646_646929


namespace T_10_equals_2_l646_646705

def T (n : ℕ) : ℕ :=
  if n = 1 then 2
  else 2 * (n / n)  -- given the strict constraints, the pattern observed is T(n) = 2 for n ≥ 1

theorem T_10_equals_2 : T 10 = 2 := 
by
  rw [T]
  norm_num
  sorry

end T_10_equals_2_l646_646705


namespace continuous_surjective_example_verification_l646_646439

-- Part (a): Prove that if f is continuous on [0,1], then f is surjective
theorem continuous_surjective (f : ℝ → ℝ) (h1 : ∀ (y : ℝ) (ε : ℝ), 0 < ε → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ |f x - y| < ε) :
  (continuous_on f (set.Icc 0 1)) → (∀ y ∈ set.Icc 0 1, ∃ x ∈ set.Icc 0 1, f x = y) :=
by { sorry }

-- Part (b): Provide an example function
noncomputable def example_function (x : ℝ) : ℝ :=
if rational (x) then x else 0

theorem example_verification (h1 : ∀ (y : ℝ) (ε : ℝ), 0 < ε → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ |example_function x - y| < ε) :
  ¬(∀ y ∈ set.Icc 0 1, ∃ x ∈ set.Icc 0 1, example_function x = y) :=
by { sorry }

end continuous_surjective_example_verification_l646_646439


namespace kinetic_energy_at_3_seconds_is_900_l646_646988

noncomputable def s (t : ℝ) : ℝ := 2 * t^2 + 3 * t - 1

def mass : ℝ := 8

def velocity (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  deriv s t

def time : ℝ := 3

def kinetic_energy (m v : ℝ) : ℝ :=
  (1 / 2) * m * v^2

theorem kinetic_energy_at_3_seconds_is_900 :
  kinetic_energy mass (velocity s time) = 900 :=
by sorry

end kinetic_energy_at_3_seconds_is_900_l646_646988


namespace lowest_temperature_l646_646492

theorem lowest_temperature 
  (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 60)
  (max_range : ∀ i j, temps i - temps j ≤ 75) : 
  ∃ L : ℝ, L = 0 ∧ ∃ i, temps i = L :=
by 
  sorry

end lowest_temperature_l646_646492


namespace part1_part2_l646_646775

variables (A B C : ℝ)
variables (a b c : ℝ) -- sides of the triangle opposite to angles A, B, and C respectively

-- Part (I): Prove that c / a = 2 given b(cos A - 2 * cos C) = (2 * c - a) * cos B
theorem part1 (h1 : b * (Real.cos A - 2 * Real.cos C) = (2 * c - a) * Real.cos B) : c / a = 2 :=
sorry

-- Part (II): Prove that b = 2 given the results from part (I) and additional conditions
theorem part2 (h1 : c / a = 2) (h2 : Real.cos B = 1 / 4) (h3 : a + b + c = 5) : b = 2 :=
sorry

end part1_part2_l646_646775


namespace firecracker_fragment_speed_l646_646981

theorem firecracker_fragment_speed 
  (v₀ : ℝ) (t : ℝ) (v₁ : ℝ) (g : ℝ) (m₁ : ℝ) (m₂ : ℝ) :
  v₀ = 20 → t = 3 → v₁ = 16 → g = 10 → m₁ / m₂ = 1 / 2 →
  let v := v₀ - g * t in
  let v₂x := - (m₁ * v₁ / m₂) in
  let v₂y := (3 * v + m₁ * v) / m₂ in
  sqrt(v₂x^2 + v₂y^2) = 17 :=
by
  intros
  sorry

end firecracker_fragment_speed_l646_646981


namespace find_x_l646_646979

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 :=
by sorry

end find_x_l646_646979


namespace number_of_zeros_in_interval_l646_646502

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 7

theorem number_of_zeros_in_interval : 
  ∃! c : ℝ, 0 < c ∧ c < 2 ∧ f c = 0 :=
begin
  sorry,
end

end number_of_zeros_in_interval_l646_646502


namespace largest_multiple_of_10_l646_646527

theorem largest_multiple_of_10 (n : ℕ) (h1 : ∀ k, 1 ≤ k → k ≤ n → 10 * k ∈ {10, 20, 30, ..., 10 * n})
  (h2 : (∑ k in finset.range (n + 1), 10 * k) / n = 205) : 10 * n = 400 :=
by
  sorry

end largest_multiple_of_10_l646_646527


namespace original_selling_price_l646_646151

-- Definitions based on the conditions
def original_price : ℝ := 933.33

-- Given conditions
def discount_rate : ℝ := 0.40
def price_after_discount : ℝ := 560.0

-- Lean theorem statement to prove that original selling price (x) is equal to 933.33
theorem original_selling_price (x : ℝ) 
  (h1 : x * (1 - discount_rate) = price_after_discount) : 
  x = original_price :=
  sorry

end original_selling_price_l646_646151


namespace sufficient_budget_for_kvass_l646_646543

variables (x y : ℝ)

theorem sufficient_budget_for_kvass (h1 : x + y = 1) (h2 : 0.6 * x + 1.2 * y = 1) : 
  3 * y ≥ 1.44 * y :=
by
  sorry

end sufficient_budget_for_kvass_l646_646543


namespace triangle_theorem_l646_646707

variable {α : Type*}

structure Triangle (α : Type*) :=
(A B C : α)

structure PointOnSegment (α : Type*) (A B : α) :=
(P : α)

structure Intersection (α : Type*) (A B C : α) :=
(P : α)

theorem triangle_theorem 
  (T : Triangle α)
  (D : PointOnSegment α T.B T.C)
  (E : PointOnSegment α T.C T.A)
  (F : PointOnSegment α T.A T.B)
  (P : Intersection α T.A D.P F.P)
  : let AB_AF := (dist T.A T.B) / (dist T.A F.P)
    let AC_AE := (dist T.A T.C) / (dist T.A E.P)
    let DC := (dist D.P T.C)
    let DB := (dist D.P T.B)
    let AD_AP := (dist T.A D.P) / (dist T.A P.P)
    let BC := (dist T.B T.C)
    in AB_AF * DC + AC_AE * DB = AD_AP * BC := 
sorry

end triangle_theorem_l646_646707


namespace cos_alpha_condition_l646_646059

theorem cos_alpha_condition (k : ℤ) (α : ℝ) :
  (α = 2 * k * Real.pi - Real.pi / 4 -> Real.cos α = Real.sqrt 2 / 2) ∧
  (Real.cos α = Real.sqrt 2 / 2 -> ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 4 ∨ α = 2 * k * Real.pi - Real.pi / 4) :=
by
  sorry

end cos_alpha_condition_l646_646059


namespace find_a_value_l646_646284

theorem find_a_value (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
by
  -- proof steps
  sorry

end find_a_value_l646_646284


namespace find_m_if_one_common_length_of_chord_l646_646797

noncomputable def polarEquation_of_line {ρ θ m : ℝ} : ℝ := ρ * (4 * real.cos θ + 3 * real.sin θ) - m

def parametric_equations_C (t : ℝ) : ℝ × ℝ :=
  (4 * t^2, 4 * t)

def rectangular_equation_of_line (x y m : ℝ) : Prop :=
  4 * x + 3 * y - m = 0

def standard_form_of_curve_C (x y : ℝ) : Prop := y^2 = 4 * x

theorem find_m_if_one_common (m : ℝ) :
  (∃ t : ℝ, rectangular_equation_of_line (4 * t^2) (4 * t) m) ↔ m = - 9 / 4 :=
sorry

theorem length_of_chord (m : ℝ) (l : ℝ) :
  m = 4 → (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ rectangular_equation_of_line (4 * t1^2) (4 * t1) m ∧ rectangular_equation_of_line (4 * t2^2) (4 * t2) m) →
  l = 25 / 4 :=
sorry

end find_m_if_one_common_length_of_chord_l646_646797


namespace cookies_indeterminate_l646_646458

theorem cookies_indeterminate (bananas : ℕ) (boxes : ℕ) (bananas_per_box : ℕ) (cookies : ℕ)
  (h1 : bananas = 40)
  (h2 : boxes = 8)
  (h3 : bananas_per_box = 5)
  : ∃ c : ℕ, c = cookies :=
by sorry

end cookies_indeterminate_l646_646458


namespace brocard_angle_max_30_l646_646146

-- Definitions and conditions
variable {A B C M : Type} -- Define the types of the points

-- Assume ABC is a triangle
axiom is_triangle (A B C : Type) : Prop

-- Define Brocard angle \varphi
variable {φ : Real}

-- Assume brocard_angle of triangle ABC is \(φ\)
axiom brocard_angle (A B C : Type) (φ : Real) : Prop

-- Assume M is a point inside triangle ABC
axiom point_in_triangle (M A B C : Type) : Prop

-- Prove that at least one of ∠ ABM, ∠ BCM, or ∠ CAM does not exceed 30°
theorem brocard_angle_max_30 
    (A B C M : Type)
    [is_triangle A B C]
    [brocard_angle A B C φ]
    [point_in_triangle M A B C] :
    ∃ (α β γ : Real), α = 30 ∨ β = 30 ∨ γ = 30 :=
by
  sorry

end brocard_angle_max_30_l646_646146


namespace solution_set_of_c_l646_646216

theorem solution_set_of_c
  (x y c : ℝ) (h1 : sqrt (x^2 * y^2) = c^(2*c))
  (h2 : log c (x^(log c y)) + log c (y^(log c x)) = 8*c^4)
  (hc_pos : c > 0) :
  c ∈ Icc 0 (1/2) :=
by
  sorry

end solution_set_of_c_l646_646216


namespace avg_k_value_l646_646331

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l646_646331


namespace joanna_needs_more_hours_to_finish_book_l646_646092
-- Import the necessary library

-- Define the problem conditions and prove the final answer

theorem joanna_needs_more_hours_to_finish_book :
  let total_pages := 248
  let pages_per_hour := 16
  let hours_monday := 3
  let hours_tuesday := 6.5
  let pages_read_monday := hours_monday * pages_per_hour
  let pages_read_tuesday := hours_tuesday * pages_per_hour
  let total_pages_read := pages_read_monday + pages_read_tuesday
  let pages_left := total_pages - total_pages_read
  let hours_needed := pages_left / pages_per_hour
  in hours_needed = 6 :=
by
  sorry

end joanna_needs_more_hours_to_finish_book_l646_646092


namespace min_rectangles_to_cover_square_exactly_l646_646131

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l646_646131


namespace sprint_race_outcomes_l646_646704

-- Definition of friends
inductive Friend : Type
| Abe : Friend
| Bobby : Friend
| Charles : Friend
| Devin : Friend
| Erin : Friend

-- Counting the number of valid outcomes for 1st-2nd-3rd place given Bobby must finish first.
def no_of_outcomes (F : Type) [F = Friend] : ℕ :=
  4 * 3   -- 4 possibilities for 2nd place, 3 for 3rd place (Note: handling the specific condition directly)

theorem sprint_race_outcomes : no_of_outcomes Friend = 12 := by
  sorry

end sprint_race_outcomes_l646_646704


namespace max_radius_inscribed_circle_l646_646655

theorem max_radius_inscribed_circle 
  (E : Set (ℝ × ℝ)) (hE : ∀ x y, (x, y) ∈ E ↔ x^2 / 4 + y^2 = 1)
  (O : Set (ℝ × ℝ)) (hO : ∀ x y, (x, y) ∈ O ↔ x^2 + y^2 = 1)
  (F : ℝ × ℝ) (hF : F = (2, 0))
  (A B : ℝ × ℝ) (hA : (A.1, A.2) ∈ E)
  (hB : (B.1, B.2) ∈ E)
  (hAB : ∀ x y, (x, y) ∈ AB ↔ (x, y) ∈ E ∧ (x, y) ∈ O)
  : ∃ r : ℝ, r = (3 - Real.sqrt 3) / 4 :=
begin
  sorry,
end

end max_radius_inscribed_circle_l646_646655


namespace smallest_number_of_rectangles_l646_646121

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l646_646121


namespace smallest_number_of_rectangles_l646_646118

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l646_646118


namespace smallest_rectangle_area_l646_646108

-- Definitions based on conditions
def diameter : ℝ := 10
def length : ℝ := diameter
def width : ℝ := diameter + 2

-- Theorem statement
theorem smallest_rectangle_area : (length * width) = 120 :=
by
  -- The proof would go here, but we provide sorry for now
  sorry

end smallest_rectangle_area_l646_646108


namespace first_fib_exceeds_1000_l646_646887

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem first_fib_exceeds_1000 :
  ∃ n, fibonacci n > 1000 ∧ ∀ m < n, fibonacci m ≤ 1000 :=
begin
  use 17,
  split,
  { -- Show that F_17 > 1000
    show fibonacci 17 > 1000,
    -- The actual proof steps would go here
    sorry },
  { -- Show that for all m < 17, F_m <= 1000
    assume m hm,
    show fibonacci m ≤ 1000,
    -- The actual proof steps would go here
    sorry }
end

end first_fib_exceeds_1000_l646_646887


namespace system_infinite_solutions_l646_646044

theorem system_infinite_solutions :
  ∃ (x y : ℚ), (3 * x - 4 * y = 5) ∧ (9 * x - 12 * y = 15) ↔ (3 * x - 4 * y = 5) :=
by
  sorry

end system_infinite_solutions_l646_646044


namespace min_sum_four_consecutive_nat_nums_l646_646251

theorem min_sum_four_consecutive_nat_nums (a : ℕ) (h1 : a % 11 = 0) (h2 : (a + 1) % 7 = 0)
    (h3 : (a + 2) % 5 = 0) (h4 : (a + 3) % 3 = 0) : a + (a + 1) + (a + 2) + (a + 3) = 1458 :=
  sorry

end min_sum_four_consecutive_nat_nums_l646_646251


namespace angles_parallel_equal_or_complementary_l646_646774

noncomputable def angles_in_space (α₁ α₂ : ℝ) (parallel : Bool) (anti_parallel : Bool) : Bool :=
  if parallel && !anti_parallel then α₁ = α₂
  else if parallel && anti_parallel then α₁ + α₂ = 90
  else false

theorem angles_parallel_equal_or_complementary (α₁ α₂ : ℝ) (parallel : Bool) (anti_parallel : Bool) :
  parallel → (parallel → ¬anti_parallel → α₁ = α₂)
  → (parallel → anti_parallel → α₁ + α₂ = 90) :
  angles_in_space α₁ α₂ parallel anti_parallel = true :=
by
  sorry

end angles_parallel_equal_or_complementary_l646_646774


namespace range_of_a_l646_646353

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 1 + a
def g (x : ℝ) : ℝ := 3 * Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Icc (1/Real.exp 1) Real.exp 1, f a x = -g x) →
  (0 ≤ a ∧ a ≤ Real.exp 3 - 4) :=
begin
  sorry
end

end range_of_a_l646_646353


namespace losing_positions_no_more_than_half_l646_646915

theorem losing_positions_no_more_than_half (n : ℕ) (pos : Finset ℕ) (out_degree in_degree : ℕ → ℕ) :
  (∀ x ∈ pos, out_degree x + in_degree x = n) →
  (∀ x ∈ pos, out_degree x = 0 → x"loses") →
  (pos.card = 2*num_total_positions) →
  ∃ l : Finset ℕ, ∀ x ∈ l, x "loses" →
  l.card ≤ pos.card / 2 :=
begin
  sorry
end

end losing_positions_no_more_than_half_l646_646915


namespace number_of_possible_values_of_m_l646_646488

theorem number_of_possible_values_of_m:
  ∃ (s : Finset ℕ), s.card = 29 ∧ ∀ m ∈ s, ∃ (n : ℕ), 
  1 ≤ m ∧ m ≤ 49 ∧ n ≥ 0 ∧ m ∣ (n^(n+1) + 1) :=
begin
  sorry
end

end number_of_possible_values_of_m_l646_646488


namespace average_k_l646_646306

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l646_646306


namespace john_needs_one_plank_l646_646249

theorem john_needs_one_plank (total_nails : ℕ) (nails_per_plank : ℕ) (extra_nails : ℕ) (P : ℕ)
    (h1 : total_nails = 11)
    (h2 : nails_per_plank = 3)
    (h3 : extra_nails = 8)
    (h4 : total_nails = nails_per_plank * P + extra_nails) :
    P = 1 :=
by
    sorry

end john_needs_one_plank_l646_646249


namespace max_CF_value_l646_646405

variable (A B C D E F : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variable (triangle : A ≃ B)  (altitude_AD : ℝ) (altitude_BE : ℝ) (altitude_CF : ℝ)

namespace TriangleAltitudes

theorem max_CF_value 
  (AD : altitude_AD = 4) 
  (BE : altitude_BE = 12) 
  (h_triangle_inequality1 : 2 * (AD/(altitude_CF)) < altitude_BE /(altitude_CF))
  (h_triangle_inequality2 : altitude_BE / (altitude_CF) > AD /  (altitude_CF)):
  ∃ CF ∈ ℤ, CF = 5:= 
sorry

end TriangleAltitudes

end max_CF_value_l646_646405


namespace sin_prod_tan_eq_complex_trig_expr_eq_l646_646412

noncomputable def sin_prod_tan (α : ℝ) : ℝ :=
let x := -1 in
let y := 2 in
let r := real.sqrt (x^2 + y^2) in
let sin_α := y / r in
let tan_α := y / x in
sin_α * tan_α

theorem sin_prod_tan_eq :
  sin_prod_tan α = - (4 * real.sqrt(5)) / 5 :=
begin
  -- Proof Skipped
  sorry
end

noncomputable def complex_trig_expr (α : ℝ) : ℝ :=
let sin_α := (2 * real.sqrt 5) / 5 in
let tan_α := -2 in
let cos_α := sin_α / tan_α in
(sin (α + real.pi / 2) * cos (7 * real.pi / 2 - α) * tan (2 * real.pi - α)) / 
(sin (2 * real.pi - α) * tan (-α))

theorem complex_trig_expr_eq :
  complex_trig_expr α = - (real.sqrt 5) / 5 :=
begin
  -- Proof Skipped
  sorry
end

end sin_prod_tan_eq_complex_trig_expr_eq_l646_646412


namespace number_of_distributions_l646_646090

-- Definitions used in conditions:
def num_books : ℕ := 5
def num_students : ℕ := 3

-- The condition that each student receives at least one book
def each_student_receives_at_least_one_book (distribution : list (list ℕ)) : Prop :=
  ∀ student_books, student_books ∈ distribution → student_books.length ≥ 1

-- The main theorem to be proved
theorem number_of_distributions (distribution : list (list ℕ)) :
  each_student_receives_at_least_one_book distribution ∧ distribution.length = num_students → 
  (finset.card (finset.filter (λ d, each_student_receives_at_least_one_book d) (finset.powerset (finset.range num_books)))) = 150 :=
begin
  sorry
end

end number_of_distributions_l646_646090


namespace hannah_mugs_problem_l646_646366

theorem hannah_mugs_problem :
  ∀ (total_mugs blue_mugs red_mugs yellow_mugs other_mugs : ℕ),
  total_mugs = 40 →
  yellow_mugs = 12 →
  red_mugs = yellow_mugs / 2 →
  blue_mugs = 3 * red_mugs →
  other_mugs = total_mugs - (blue_mugs + red_mugs + yellow_mugs) →
  other_mugs = 4 :=
by
  intros total_mugs blue_mugs red_mugs yellow_mugs other_mugs
  intros h_total h_yellow h_red h_blue h_other
  have h1: red_mugs = 6, by linarith [h_yellow, h_red]
  have h2: blue_mugs = 18, by linarith [h1, h_blue]
  have h3: other_mugs = 4, by linarith [h_total, h2, h1, h_yellow, h_other]
  exact h3

end hannah_mugs_problem_l646_646366


namespace only_n_is_zero_l646_646233

theorem only_n_is_zero (n : ℕ) (h : (n^2 + 1) ∣ n) : n = 0 := 
by sorry

end only_n_is_zero_l646_646233


namespace correct_div_value_l646_646373

theorem correct_div_value (x : ℝ) (h : 25 * x = 812) : x / 4 = 8.12 :=
by sorry

end correct_div_value_l646_646373


namespace golden_rectangle_perimeter_l646_646916

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 - 1) / 2

variables (ABCD : Type) [metric_space ABCD] [rectangle ABCD]
variables (AB : ℝ)
variables (ratio : ℝ := (Real.sqrt 5 - 1) / 2)
variables (side_length : ℝ := Real.sqrt 5 - 1)

-- Conditions
def golden_rectangle (ABCD : Type) : Prop :=
  ∃ (width length : ℝ),
    width / length = ratio ∧
    (∃ (A B : ℝ), A = side_length ∧ B = side_length / ratio ∧ ((2 * A + 2 * B) = 4 ∨ (2 * A + 2 * B) = 2 * Real.sqrt 5 + 2))

-- Theorem statement
theorem golden_rectangle_perimeter (ABCD : Type) [metric_space ABCD] [rectangle ABCD]
  (h : golden_rectangle ABCD) : ∃ (perimeter : ℝ), perimeter = 4 ∨ perimeter = 2 * Real.sqrt 5 + 2 :=
sorry

end golden_rectangle_perimeter_l646_646916


namespace average_of_distinct_k_l646_646308

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l646_646308


namespace smallest_num_rectangles_l646_646125

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l646_646125


namespace pq_sum_l646_646003

open Real

theorem pq_sum (p q : ℝ) (hp : p^3 - 18 * p^2 + 81 * p - 162 = 0) (hq : 4 * q^3 - 24 * q^2 + 45 * q - 27 = 0) :
    p + q = 8 ∨ p + q = 8 + 6 * sqrt 3 ∨ p + q = 8 - 6 * sqrt 3 :=
sorry

end pq_sum_l646_646003


namespace smallest_number_divisible_conditions_l646_646558

theorem smallest_number_divisible_conditions :
  ∃ n : ℕ, n % 8 = 6 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m % 8 = 6 ∧ m % 7 = 5 → n ≤ m →
  n % 9 = 0 := by
  sorry

end smallest_number_divisible_conditions_l646_646558


namespace digits_prime_factor_ge_11_l646_646480

theorem digits_prime_factor_ge_11 (N : ℕ) (h1 : N > 10) (h2 : ∀ d, d ∈ N.digits 10 → d ∈ {1, 3, 7, 9}) : 
  ∃ p : ℕ, p.prime ∧ p ≥ 11 ∧ p ∣ N :=
by sorry

end digits_prime_factor_ge_11_l646_646480


namespace jellybean_third_less_than_second_l646_646494

noncomputable def jellybean_difference : ℕ :=
let first_guess := 100 in
let second_guess := 8 * first_guess in
let fourth_guess := 525 in
let average_three_guesses := (λ x, (first_guess + second_guess + x) / 3 + 25) in
let third_guess := (λ x, if average_three_guesses x = fourth_guess then x else 0) (fourth_guess * 3 - first_guess - second_guess) in
second_guess - third_guess

theorem jellybean_third_less_than_second :
  jellybean_difference = 200 := by
  sorry

end jellybean_third_less_than_second_l646_646494


namespace friends_game_l646_646608

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l646_646608


namespace find_angle_CFD_l646_646404

open Real

-- Definitions for the conditions in the problem
def right_angle : ℝ := 90

variables {A B C D F : ℝ × ℝ}
variables (triangle_ABC : (B.1 = C.1 ∧ B.2 ≠ C.2) ∧ (A.2 = B.2) ∧ (C = (B.1, A.2)))
variables (D_on_circum_BC : (B.1 = C.1 ∧ D.2 = -C.2))
variables (F_on_tangent : F.2 = -D.2)
variables (angle_CAB : ∠ A C B = 46)

-- The theorem to prove the desired angle measure
theorem find_angle_CFD :
  ∠ C F D = 92 :=
sorry

end find_angle_CFD_l646_646404


namespace area_triangle_foci_l646_646724

noncomputable def hyperbola_foci_coordinates (a : ℝ) := 
  (real.sqrt (a^2 + 1), 0), (-real.sqrt (a^2 + 1), 0)

theorem area_triangle_foci (a : ℝ) (P : ℝ × ℝ) :
  let F1 := (-real.sqrt (a^2 + 1), 0),
      F2 := (real.sqrt (a^2 + 1), 0) in
  P.1^2 - P.2^2 = 1 →
  ∃ θ : ℝ, θ = real.pi / 3 ∧ 
  (F1.1 - P.1)^2 + F1.2^2 + (F2.1 - P.1)^2 + F2.2^2 - 
  2 * real.norm (F1.1 - P.1, F1.2) * real.norm (F2.1 - P.1, F2.2) * real.cos θ = 8 →
  (1 / 2) * real.norm (F1.1 - P.1, F1.2) * real.norm (F2.1 - P.1, F2.2) * real.sin (real.pi / 3) = real.sqrt (3) :=
-- Proof omitted
sorry

end area_triangle_foci_l646_646724


namespace donuts_left_l646_646200

def initial_donuts : ℕ := 50
def after_bill_eats (initial : ℕ) : ℕ := initial - 2
def after_secretary_takes (remaining_after_bill : ℕ) : ℕ := remaining_after_bill - 4
def coworkers_take (remaining_after_secretary : ℕ) : ℕ := remaining_after_secretary / 2
def final_donuts (initial : ℕ) : ℕ :=
  let remaining_after_bill := after_bill_eats initial
  let remaining_after_secretary := after_secretary_takes remaining_after_bill
  remaining_after_secretary - coworkers_take remaining_after_secretary

theorem donuts_left : final_donuts 50 = 22 := by
  sorry

end donuts_left_l646_646200


namespace positive_difference_solutions_abs_eq_30_l646_646939

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l646_646939


namespace next_instance_of_same_digits_l646_646075

theorem next_instance_of_same_digits :
  ∃ (d1 m1 y1 h1 min1 d2 m2 y2 h2 min2 : ℕ),
    (d1 = 25) ∧ (m1 = 5) ∧ (y1 = 1994) ∧ (h1 = 2) ∧ (min1 = 45) ∧
    (d2 = 1) ∧ (m2 = 8) ∧ (y2 = 1994) ∧ (h2 = 2) ∧ (min2 = 45) ∧
    (digits_used_by_date_time d1 m1 y1 h1 min1 = {0, 1, 2, 4, 5, 9}) ∧
    (digits_used_by_date_time d2 m2 y2 h2 min2 = {0, 1, 2, 4, 5, 9}) ∧
    (next_date_time d1 m1 y1 h1 min1 d2 m2 y2 h2 min2) := 
sorry

noncomputable def digits_used_by_date_time 
  (d m y h min : ℕ) : set ℕ :=
{0, 1, 2, 4, 5, 9}  -- a function that returns a set of all digits used in a date and time. Here it's prefilled for simplification.

noncomputable def next_date_time 
  (d1 m1 y1 h1 min1 d2 m2 y2 h2 min2 : ℕ) : Prop :=
  _ -- a function that determines if (d2, m2, y2, h2, min2) is the next date-time after (d1, m1, y1, h1, min1). This is also simplified.

end next_instance_of_same_digits_l646_646075


namespace shaded_area_of_inscribed_circle_l646_646097

theorem shaded_area_of_inscribed_circle (side : ℝ) (h_side : side = 12) :
  let A_square := side^2,
      r := side / 2,
      A_circle := Real.pi * r^2,
      A_shaded := A_square - A_circle
  in A_shaded = 144 - Real.pi * 36 :=
by
  -- Using the provided conditions in the problem
  rw [h_side],
  let A_square := 12^2,
  let r := 12 / 2,
  let A_circle := Real.pi * r^2,
  let A_shaded := A_square - A_circle,
  calc
    A_shaded = 144 - Real.pi * 36 : sorry

end shaded_area_of_inscribed_circle_l646_646097


namespace friends_game_l646_646607

theorem friends_game
  (n m : ℕ)
  (h : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
begin
  sorry
end

end friends_game_l646_646607


namespace log_inequality_solution_l646_646899

theorem log_inequality_solution (x : ℝ) :
  x + 1 > 0 → 3 - x > 0 → log x + 1 < log (3 - x) → -1 < x ∧ x < 1 :=
by
  sorry

end log_inequality_solution_l646_646899


namespace at_least_one_of_each_probability_l646_646400

theorem at_least_one_of_each_probability :
  let forks := 6 in
  let spoons := 8 in
  let knives := 6 in
  let total_items := forks + spoons + knives in
  let total_ways := Nat.choose total_items 4 in
  let ways_no_forks := Nat.choose (spoons + knives) 4 in
  let ways_no_spoons := Nat.choose (forks + knives) 4 in
  let ways_no_knives := Nat.choose (forks + spoons) 4 in
  let ways_no_forks_or_spoons := 0 in -- {no_forks, no_spoons cases are impossible here as discussed}
  let ways_no_forks_or_knives := Nat.choose spoons 4 in
  let ways_no_spoons_or_knives := Nat.choose forks 4 in
  let invalid_ways := ways_no_forks + ways_no_spoons + ways_no_knives - ways_no_forks_or_spoons - ways_no_forks_or_knives - ways_no_spoons_or_knives in
  let valid_ways := total_ways - invalid_ways in
  (valid_ways : ℚ) / (total_ways : ℚ) = 1 / 2 
:= by
  sorry

end at_least_one_of_each_probability_l646_646400


namespace quadratic_integer_solution_l646_646212

def num_distinct_rationals_bound_by_200_with_integer_solution : ℕ := 118

theorem quadratic_integer_solution :
  ∃ (k_set : Set ℤ), (∀ k ∈ k_set, abs k < 200 ∧ ∃ x : ℤ, 3 * x^2 + k * x + 18 = 0) ∧
  k_set.card = num_distinct_rationals_bound_by_200_with_integer_solution :=
by
  sorry

end quadratic_integer_solution_l646_646212


namespace max_min_diff_eq_l646_646068

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 2) - Real.sqrt (x^2 - 3*x + 3)

theorem max_min_diff_eq : 
  (∀ x : ℝ, ∃ max min : ℝ, max = Real.sqrt (8 - Real.sqrt 3) ∧ min = -Real.sqrt (8 - Real.sqrt 3) ∧ 
  (max - min = 2 * Real.sqrt (8 - Real.sqrt 3))) :=
sorry

end max_min_diff_eq_l646_646068


namespace friends_number_options_l646_646592

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l646_646592


namespace average_k_l646_646307

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l646_646307


namespace incorrect_statement_a_l646_646969

theorem incorrect_statement_a :
  ¬(∀ (Q : Type) [quadrilateral Q], 
    (∀ (d1 d2 : diagonal Q), d1.perpendicular d2 ∧ d1.length = d2.length → is_square Q)) :=
by
  -- Proof omitted
  sorry

structure quadrilateral (Q : Type) :=
(diagonal : Type) 
(is_square : bool)

axiom diagonal.properties : 
  ∀ (Q : Type) [quadrilateral Q], 
  ∀ (d1 d2 : diagonal Q), 
    d1.perpendicular d2 ∧ d1.length = d2.length → 
    (quadrilateral.is_square Q ↔ true)

end incorrect_statement_a_l646_646969


namespace ellipse_line_intersection_l646_646293

theorem ellipse_line_intersection (m : ℝ) : m > 0 ∧ m ≥ 1 ∧ m ≠ 5 :=
by {
  have h1 := 1 / m ≤ 1,
  split,
  -- Proof that m > 0 will be provided here
  sorry,
  split,
  -- Proof that m ≥ 1 will be provided here
  sorry,
  -- Proof that m ≠ 5 will be provided here
  sorry
}

end ellipse_line_intersection_l646_646293


namespace positive_difference_eq_30_l646_646959

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end positive_difference_eq_30_l646_646959


namespace max_disjoint_subsets_sum_l646_646509

noncomputable def S : Set ℕ := {15, 14, 13, 11, 8}

theorem max_disjoint_subsets_sum :
  (∀ A B : Set ℕ, A ⊆ S → B ⊆ S → A ≠ B → A ∩ B = ∅ → (A.sum id ≠ B.sum id)) ∧
  S.sum id = 61 :=
by 
  sorry

end max_disjoint_subsets_sum_l646_646509


namespace possible_number_of_friends_l646_646598

-- Condition statements as Lean definitions
variables (player : Type) (plays : player → player → Prop)
variables (n m : ℕ)

-- Condition 1: Every pair of players are either allies or opponents
axiom allies_or_opponents : ∀ A B : player, plays A B ∨ ¬ plays A B

-- Condition 2: If A allies with B, and B opposes C, then A opposes C
axiom transitive_playing : ∀ (A B C : player), plays A B → ¬ plays B C → ¬ plays A C

-- Condition 3: Each player has exactly 15 opponents
axiom exactly_15_opponents : ∀ A : player, (count (λ B, ¬ plays A B) = 15)

-- Theorem to prove the number of players in the group
theorem possible_number_of_friends (num_friends : ℕ) : 
  (∃ (n m : ℕ), (n-1) * m = 15 ∧ n * m = num_friends) → 
  num_friends = 16 ∨ num_friends = 18 ∨ num_friends = 20 ∨ num_friends = 30 :=
by
  sorry

end possible_number_of_friends_l646_646598


namespace cat_vs_dog_and_bird_food_l646_646035

noncomputable def cat_packages : ℕ := 8
noncomputable def dog_packages : ℕ := 5
noncomputable def bird_packages : ℕ := 3
noncomputable def cans_per_cat_package : ℕ := 12
noncomputable def cans_per_dog_package : ℕ := 7
noncomputable def cans_per_bird_package : ℕ := 4

def initial_cat_cans : ℕ := cat_packages * cans_per_cat_package
def initial_dog_cans : ℕ := dog_packages * cans_per_dog_package
def initial_bird_cans : ℕ := bird_packages * cans_per_bird_package

def remaining_cat_cans : ℕ := initial_cat_cans / 2
def remaining_dog_cans : ℕ := initial_dog_cans - (initial_dog_cans / 4)
def remaining_bird_cans : ℕ := initial_bird_cans

def total_remaining_dog_and_bird_cans : ℕ := remaining_dog_cans + remaining_bird_cans

theorem cat_vs_dog_and_bird_food :
  remaining_cat_cans = total_remaining_dog_and_bird_cans + 9 :=
by
  sorry

end cat_vs_dog_and_bird_food_l646_646035


namespace length_PR_in_triangle_l646_646802

/-- In any triangle PQR, given:
  PQ = 7, QR = 10, median PS = 5,
  the length of PR must be sqrt(149). -/
theorem length_PR_in_triangle (PQ QR PS : ℝ) (PQ_eq : PQ = 7) (QR_eq : QR = 10) (PS_eq : PS = 5) : 
  ∃ (PR : ℝ), PR = Real.sqrt 149 := 
sorry

end length_PR_in_triangle_l646_646802


namespace mode_of_scores_is_85_l646_646087

-- Define the scores based on the given stem-and-leaf plot
def scores : List ℕ := [50, 55, 55, 62, 62, 68, 70, 71, 75, 79, 81, 81, 83, 85, 85, 85, 92, 96, 96, 98, 100, 100]

-- Define a function to compute the mode
def mode (s : List ℕ) : ℕ :=
  s.foldl (λ acc x => if s.count x > s.count acc then x else acc) 0

-- The theorem to prove that the mode of the scores is 85
theorem mode_of_scores_is_85 : mode scores = 85 :=
by
  -- The proof is omitted
  sorry

end mode_of_scores_is_85_l646_646087


namespace sqrt_of_2a_plus_b_minus_c_l646_646289

noncomputable def sqrt_cubed_root (a : ℝ) (b : ℝ) (c : ℝ) : Prop :=
  (∃ (a : ℝ), ∃ (b : ℝ), ∃ (c : ℝ),
    (sqrt (3 * a + b - 1) = 4 ∨ sqrt (3 * a + b - 1) = -4) ∧
    (c = 3) ∧
    sqrt (2 * a + b - c) = 3 ∧
    (5 * a + 2) ^ (1/3 : ℝ) = 3)
---- now, I am trying to write the theorem that proves that the sq root of 2a + b - c = +/-3
theorem sqrt_of_2a_plus_b_minus_c (a b c : ℝ) (h1 : (5 * a + 2) ^ (1/3 : ℝ) = 3)
                                    (h2 : sqrt (3 * a + b - 1) = 4 ∨ sqrt (3 * a + b - 1) = -4 )
                                    (h3 : c = 3) :
  sqrt(2 * a + b - c) = 3 ∨ sqrt(2 * a + b - c) = -3 := 
sorry

end sqrt_of_2a_plus_b_minus_c_l646_646289


namespace fraction_irreducible_l646_646471

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l646_646471


namespace derivative_at_pi_l646_646737

theorem derivative_at_pi : 
  (fun x => (Real.sin x + Real.cos x))' π = -1 := 
by
  -- omitted proof
  sorry

end derivative_at_pi_l646_646737


namespace part_a_division_part_b_division_l646_646513

theorem part_a_division (m n : ℕ) (m_eq : m = 3) (n_eq : n = 5) :
  ∃ (parts : list ℚ), 
  (∀ (p : ℚ), p ∈ parts → p > 1 / 3) ∧ 
  (∑ p in parts, p = m * n) :=
sorry

theorem part_b_division (m n : ℕ) (m_eq : m = 5) (n_eq : n = 3) :
  ∃ (parts : list ℚ), 
  (∀ (p : ℚ), p ∈ parts → p > 1 / 5) ∧
  (∑ p in parts, p = m * n) :=
sorry

end part_a_division_part_b_division_l646_646513


namespace value_of_f_at_5_3_pi_l646_646069

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ π / 2 then 
    sin x 
  else if h : -π / 2 ≤ x ∧ x < 0 then 
    sin (-x)
  else 
    f (x - π * ⌊(2*x/π) + 1/2⌋)

lemma f_periodic_pi (x : ℝ) : f (x + π) = f x :=
begin
  sorry
end

lemma f_even (x : ℝ) : f x = f (-x) :=
begin
  sorry
end

theorem value_of_f_at_5_3_pi : f (5 / 3 * π) = -sqrt 3 / 2 :=
begin
  sorry
end

end value_of_f_at_5_3_pi_l646_646069


namespace find_f_prime_two_l646_646262

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * (f' 1)
noncomputable def f' (x : ℝ) : ℝ := 2 * x + 3 * (f' 1)

theorem find_f_prime_two (f'1 : ℝ) (h : f'1 = -1) : f' 2 = 1 := by
  sorry

end find_f_prime_two_l646_646262


namespace roots_condition_implies_m_range_l646_646389

theorem roots_condition_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ (x₁^2 + (m-1)*x₁ + m^2 - 2 = 0) ∧ (x₂^2 + (m-1)*x₂ + m^2 - 2 = 0))
  → -2 < m ∧ m < 1 :=
by
  sorry

end roots_condition_implies_m_range_l646_646389


namespace shen_winning_probability_sum_l646_646865

/-!
# Shen Winning Probability

Prove that the sum of the numerator and the denominator, m + n, 
of the simplified fraction representing Shen's winning probability is 184.
-/

theorem shen_winning_probability_sum :
  let m := 67
  let n := 117
  m + n = 184 :=
by sorry

end shen_winning_probability_sum_l646_646865


namespace odd_n_possible_l646_646905

variable (n : ℕ)
variable (p : Fin n → Type)
variable (χ : (Fin n → ℕ))
variable (S : Fin 2 → Finset (Fin n))

def is_fair_carpool_system (m : ℕ) (c : ℕ) : Prop :=
  ∀ i : Fin n, (Finset.card (Finset.filter (λ j, i ∈ S j) (Finset.range m)) = c) ∧
  ∀ j : Fin m, (Finset.sum (S j) (λ x, χ x) = n)

theorem odd_n_possible (n : ℕ) (h1 : is_fair_carpool_system n 2) (h2 : ¬ is_fair_carpool_system n 1) : ¬ even n :=
  sorry

#check odd_n_possible

end odd_n_possible_l646_646905


namespace determinant_problem_l646_646374

-- Define the variables and conditions
variables (p q r s : ℝ)
hypothesis (h : p * s - q * r = 7)

-- State the theorem
theorem determinant_problem : (p + r) * s - (q + s) * r = 7 :=
by sorry

end determinant_problem_l646_646374


namespace average_of_k_l646_646319

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l646_646319


namespace complement_of_A_with_respect_to_U_l646_646753

open Set

-- Definitions
def U : Set ℤ := {-1, 1, 3}
def A : Set ℤ := {-1}

-- Theorem statement
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {1, 3} :=
by
  sorry

end complement_of_A_with_respect_to_U_l646_646753


namespace iterative_average_difference_l646_646652

theorem iterative_average_difference {α : Type*} [LinearOrder α] [DivisionRing α] [AddGroup α] :
  let S := [1, 2, 3, 5, 6 : α],
  let iterative_average (lst : List α) : α :=
      lst.foldl (fun x y => (x + y) / 2) (lst.head)
  in abs ((list.all_permutations S).map iterative_average).max' -
           ((list.all_permutations S).map iterative_average).min' = 2.75 :=
by sorry

end iterative_average_difference_l646_646652


namespace arithmetic_series_sum_l646_646666

theorem arithmetic_series_sum : 
  (∑ k in Finset.range 10, (2 * (k + 1) : ℚ) / 7) = 110 / 7 := 
by
  sorry

end arithmetic_series_sum_l646_646666


namespace journey_time_ratio_l646_646169

theorem journey_time_ratio (D : ℝ)
  (h1 : D > 0)
  (to_SF_speed : ℝ := 63)
  (back_speed : ℝ := 42) :
  let time_to_SF := D / to_SF_speed
  let time_back := D / back_speed
  (time_back / time_to_SF) = (3 / 2) :=
by
  have h2 : to_SF_speed ≠ 0 := by norm_num
  have h3 : back_speed ≠ 0 := by norm_num
  have time_to_SF_def : time_to_SF = D / to_SF_speed := rfl
  have time_back_def : time_back = D / back_speed := rfl
  have ratio_def : (time_back / time_to_SF) = (D / back_speed) / (D / to_SF_speed) := rfl
  rw [time_to_SF_def, time_back_def, ratio_def]
  field_simp [h2, h3, h1]
  norm_num

end journey_time_ratio_l646_646169


namespace digit_150_l646_646934

def decimal_rep : ℚ := 5 / 13

def cycle_length : ℕ := 6

theorem digit_150 (n : ℕ) (h : n = 150) : Nat.digit (n % cycle_length) (decimal_rep) = 5 := by
  sorry

end digit_150_l646_646934


namespace digit_150th_of_5_div_13_l646_646923

theorem digit_150th_of_5_div_13 : 
    ∀ k : ℕ, (k = 150) → (fractionalPartDigit k (5 / 13) = 5) :=
by 
  sorry

end digit_150th_of_5_div_13_l646_646923


namespace find_positive_integers_unique_solution_l646_646235

theorem find_positive_integers_unique_solution :
  ∃ x r p n : ℕ,  
  0 < x ∧ 0 < r ∧ 0 < n ∧  Nat.Prime p ∧ 
  r > 1 ∧ n > 1 ∧ x^r - 1 = p^n ∧ 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := 
    sorry

end find_positive_integers_unique_solution_l646_646235


namespace irrational_number_among_list_l646_646647

theorem irrational_number_among_list :
  ¬(∀ x ∈ [-3/2, -Real.sqrt 4, 0.23, Real.pi / 3], irrational x) →
  irrational (Real.pi / 3) :=
begin
  -- Definitions from the problem
  let a := -3/2,
  let b := -Real.sqrt 4,
  let c := 0.23,
  let d := Real.pi / 3,

  -- Proof assertion
  have h : irrational d,
  from sorry, -- proof omitted,
end

end irrational_number_among_list_l646_646647


namespace geometric_series_sum_150_terms_l646_646886

theorem geometric_series_sum_150_terms (a : ℕ) (r : ℝ)
  (h₁ : a = 250)
  (h₂ : (a - a * r ^ 50) / (1 - r) = 625)
  (h₃ : (a - a * r ^ 100) / (1 - r) = 1225) :
  (a - a * r ^ 150) / (1 - r) = 1801 := by
  sorry

end geometric_series_sum_150_terms_l646_646886


namespace tetrahedron_volume_l646_646411

noncomputable def volume_tetrahedron (A₁ A₂ : ℝ) (θ : ℝ) (d : ℝ) : ℝ :=
  (A₁ * A₂ * Real.sin θ) / (3 * d)

theorem tetrahedron_volume:
  ∀ (PQ PQR PQS : ℝ) (θ : ℝ),
  PQ = 5 → PQR = 20 → PQS = 18 → θ = Real.pi / 4 → volume_tetrahedron PQR PQS θ PQ = 24 * Real.sqrt 2 :=
by
  intros
  unfold volume_tetrahedron
  sorry

end tetrahedron_volume_l646_646411


namespace friends_number_options_l646_646591

theorem friends_number_options (T : ℕ)
  (h_opp : ∀ (A B C : ℕ), (plays_together A B ∧ plays_against B C) → plays_against A C)
  (h_15_opp : ∀ A, count_opponents A = 15) :
  T ∈ {16, 18, 20, 30} := 
  sorry

end friends_number_options_l646_646591


namespace max_average_time_per_maze_l646_646252

theorem max_average_time_per_maze 
  (time_spent_current_maze : ℕ) 
  (num_previous_mazes : ℕ) 
  (avg_time_previous_mazes : ℕ) 
  (additional_time_allowed : ℕ) :
  (total_time_all_mazes : ℕ) = (num_previous_mazes + 1) * (avg_time_previous_mazes + additional_time_allowed - time_spent_current_maze) / (num_previous_mazes + 1) :=
begin
  have previous_mazes_time : ℕ := num_previous_mazes * avg_time_previous_mazes,
  have current_maze_time : ℕ := time_spent_current_maze + additional_time_allowed,
  have total_time_all_mazes := previous_mazes_time + current_maze_time,
  sorry
end

end max_average_time_per_maze_l646_646252


namespace triangle_equilateral_proof_expression_range_proof_l646_646392

noncomputable def triangle_equilateral (A B C : ℝ) (h1 : sin^2 B = sin A * sin C) (h2: 2 * B = A + C) : Bool :=
  if B = π / 3 ∧ (∃ a c : ℝ, a = c) then true else false

noncomputable def expression_range (A : ℝ) (hA1 : 0 < A) (hA2 : A < 2 * pi / 3) : Set ℝ :=
  { x | 1 / 4 < x ∧ x ≤ 1 / 2 }

theorem triangle_equilateral_proof (A B C: ℝ) (h1 : sin^2 B = sin A * sin C) (h2: 2 * B = A + C) :
  triangle_equilateral A B C h1 h2 = true :=
sorry

theorem expression_range_proof (A : ℝ) (hA1 : 0 < A) (hA2 : A < 2 * pi / 3) :
  ∃ x ∈ expression_range A hA1 hA2, x = sin^2 (C / 2) + sqrt 3 * sin (A / 2) * cos (A / 2) - 1 / 2 :=
sorry

end triangle_equilateral_proof_expression_range_proof_l646_646392


namespace triangle_area_le_2_l646_646403

theorem triangle_area_le_2 
  (P : Fin 6 → ℤ × ℤ)
  (h1 : ∀ i, |(P i).fst| ≤ 2 ∧ |(P i).snd| ≤ 2)
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear (P i) (P j) (P k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
  triangle_area (P i) (P j) (P k) ≤ 2 :=
sorry

def collinear (A B C : ℤ × ℤ) : Prop :=
  (B.snd - A.snd) * (C.fst - A.fst) = (C.snd - A.snd) * (B.fst - A.fst)

def triangle_area (A B C : ℤ × ℤ) : ℝ :=
  1 / 2 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

end triangle_area_le_2_l646_646403


namespace box_volume_l646_646432

def height := 12
def length := 3 * height
def width := length / 4

theorem box_volume : (height * length * width) = 3888 := by
  sorry

end box_volume_l646_646432


namespace identify_150th_digit_l646_646927

def repeating_sequence : List ℕ := [3, 8, 4, 6, 1, 5]

theorem identify_150th_digit :
  (150 % 6 = 0) →
  nth repeating_sequence 5 = 5 :=
by
  intros h
  rewrite_modulo h
  rfl

end identify_150th_digit_l646_646927


namespace problem_statement_l646_646351

noncomputable def f (x : ℝ) : ℝ := (2 / 4^x) - x

def a : ℝ := 0
def b : ℝ := Real.log 2 / Real.log 0.4 -- using properties of logarithm base change
def c : ℝ := Real.log 3 / Real.log 4

theorem problem_statement : f a < f c ∧ f c < f b :=
by {
  -- Note: Proof not required, providing sorry for placeholder
  sorry
}

end problem_statement_l646_646351


namespace work_done_by_force_l646_646286

noncomputable def displacement (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem work_done_by_force :
  let F := (5, 2)
  let A := (-1, 3)
  let B := (2, 6)
  let AB := displacement A B
  dot_product F AB = 21 := by
  sorry

end work_done_by_force_l646_646286


namespace simplify_fraction_l646_646736

theorem simplify_fraction :
  let cos_15 := real.cos (real.pi / 12)
  let expression := 32 * cos_15^4 - 10 - 8 * real.sqrt 3
  (5 / (1 + real.cbrt expression)) = (1 - real.cbrt 4 + real.cbrt 16) :=
by sorry

end simplify_fraction_l646_646736


namespace probability_log_floor_difference_zero_l646_646445

theorem probability_log_floor_difference_zero :
  let prob_event : ℝ :=
    ∫ x in (0:ℝ) .. 1, if (⌊ log10 (3 * x) ⌋ - ⌊ log10 x ⌋ = 0) then (1:ℝ) / (1 - 0) else 0
  in prob_event = 2 / 9 := by
  sorry

end probability_log_floor_difference_zero_l646_646445


namespace geom_sequence_second_term_l646_646993

noncomputable def geom_sequence_term (a r : ℕ) (n : ℕ) : ℕ := a * r^(n-1)

theorem geom_sequence_second_term 
  (a1 a5: ℕ) (r: ℕ) 
  (h1: a1 = 5)
  (h2: a5 = geom_sequence_term a1 r 5)
  (h3: a5 = 320)
  (h_r: r^4 = 64): 
  geom_sequence_term a1 r 2 = 10 :=
by
  sorry

end geom_sequence_second_term_l646_646993


namespace sum_medians_is_64_l646_646194

noncomputable def median (l: List ℝ) : ℝ := sorry  -- Placeholder for median calculation

open List

/-- Define the scores for players A and B as lists of real numbers -/
def player_a_scores : List ℝ := sorry
def player_b_scores : List ℝ := sorry

/-- Prove that the sum of the medians of the scores lists is 64 -/
theorem sum_medians_is_64 : median player_a_scores + median player_b_scores = 64 := sorry

end sum_medians_is_64_l646_646194


namespace mans_rate_l646_646995

-- Define the conditions
def speedWithStream : ℝ := 18
def speedAgainstStream : ℝ := 8

-- Define the man's rate in still water
def mansRateInStillWater : ℝ := (speedWithStream + speedAgainstStream) / 2

-- Prove that the man's rate in still water is 13 km/h
theorem mans_rate : mansRateInStillWater = 13 := by
  unfold mansRateInStillWater
  unfold speedWithStream speedAgainstStream
  -- Calculate the average speed
  have h : (18 + 8) / 2 = 13 := by norm_num
  exact h

end mans_rate_l646_646995


namespace positive_difference_solutions_abs_eq_30_l646_646938

theorem positive_difference_solutions_abs_eq_30 :
  (let x1 := 18 in let x2 := -12 in x1 - x2 = 30) :=
by
  let x1 := 18
  let x2 := -12
  show x1 - x2 = 30
  sorry

end positive_difference_solutions_abs_eq_30_l646_646938


namespace remainder_98_pow_50_mod_100_l646_646112

theorem remainder_98_pow_50_mod_100 :
  (98 : ℤ) ^ 50 % 100 = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l646_646112


namespace line_properties_l646_646536

def is_parallel_to_y_axis (line : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ y : ℝ, line k y

def cuts_segment_from_x_axis (line : ℝ → ℝ → Prop) (length : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 - x2).abs = length ∧ ∀ y : ℝ, line x1 y ∨ line x2 y

noncomputable def equation_of_parallel_y_axis_line : ℝ → (ℝ → ℝ → Prop)
| k := λ x y, x = k

theorem line_properties :
  (∃ line : ℝ → ℝ → Prop, is_parallel_to_y_axis line ∧ cuts_segment_from_x_axis line 3) ∧ 
  (∃ (line_A : ℝ → ℝ → Prop), (∀ y : ℝ, line_A 3 y) ∧ line_A 3 4) ∧
  (∃ (line_B : ℝ → ℝ → Prop), (∀ y : ℝ, line_B 3 y) ∧ ¬ line_B (-3) 2) :=
by
  let line := equation_of_parallel_y_axis_line 3
  have h1 : is_parallel_to_y_axis line := sorry
  have h2 : cuts_segment_from_x_axis line 3 := sorry
  have h3 : ∀ y, line 3 y := sorry
  have h4 : line 3 4 := sorry
  have h5 : ¬ line (-3) 2 := sorry
  exact ⟨⟨line, h1, h2⟩, ⟨line, h3, h4⟩, ⟨line, h3, h5⟩⟩

end line_properties_l646_646536


namespace range_a_l646_646730

theorem range_a (x a : ℝ) (h1 : x^2 - 8 * x - 33 > 0) (h2 : |x - 1| > a) (h3 : a > 0) :
  0 < a ∧ a ≤ 4 :=
by
  sorry

end range_a_l646_646730


namespace smallest_whole_number_divisibility_l646_646965

theorem smallest_whole_number_divisibility : ∃ x : ℕ, 
  (x % 6 = 2) ∧ 
  (x % 5 = 3) ∧ 
  (x % 7 = 1) ∧ 
  (∀ y : ℕ, (y % 6 = 2) ∧ (y % 5 = 3) ∧ (y % 7 = 1) → x ≤ y) := 
begin
  have x := 92,
  use x,
  split,
  { exact (92 % 6 = 2), sorry },
  split,
  { exact (92 % 5 = 3), sorry },
  split,
  { exact (92 % 7 = 1), sorry },
  { intros y,
    assume h,
    exact (x ≤ y), sorry }
end

end smallest_whole_number_divisibility_l646_646965


namespace pencil_red_length_l646_646376

theorem pencil_red_length (total_length : ℝ) (green_fraction : ℝ) (gold_fraction : ℝ) (red_fraction : ℝ) 
                           (total_length_eq : total_length = 15) 
                           (green_fraction_eq : green_fraction = 7 / 10) 
                           (gold_fraction_eq : gold_fraction = 3 / 7) 
                           (red_fraction_eq : red_fraction = 2 / 3) : 
    let remaining_after_green := total_length * (1 - green_fraction) in
    let remaining_after_gold := remaining_after_green * (1 - gold_fraction) in
    let red_length := remaining_after_gold * red_fraction in
    red_length ≈ 1.71429 :=
by 
  sorry

end pencil_red_length_l646_646376


namespace sequence_relation_l646_646734

theorem sequence_relation
  (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h1 : ∀ n, b (n + 1) * a n + b n * a (n + 1) = (-2)^n + 1)
  (h2 : ∀ n, b n = (3 + (-1 : ℚ)^(n-1)) / 2)
  (h3 : a 1 = 2) :
  ∀ n, a (2 * n) = (1 - 4^n) / 2 :=
by
  intro n
  sorry

end sequence_relation_l646_646734


namespace central_angle_proof_l646_646848

noncomputable def central_angle (l r : ℝ) : ℝ :=
  l / r

theorem central_angle_proof :
  central_angle 300 100 = 3 :=
by
  -- The statement of the theorem aligns with the given problem conditions and the expected answer.
  sorry

end central_angle_proof_l646_646848


namespace f_divisibility_equiv_l646_646440

theorem f_divisibility_equiv (f : ℕ → ℕ) :
  (∀ m n : ℕ, m ≤ n → f(m) + n ∣ f(n) + m) ↔ (∀ m n : ℕ, m ≥ n → f(m) + n ∣ f(n) + m) :=
sorry

end f_divisibility_equiv_l646_646440


namespace wire_length_l646_646052

theorem wire_length (d h1 h2 : ℕ) (H1 : d = 20) (H2 : h1 = 10) (H3 : h2 = 25) :
    ∃ c, c = 25 ∧ c * c = d * d + (h2 - h1) * (h2 - h1) := by
  use 25
  split
  . rfl
  . have H4 : (h2 - h1) = 15 := sorry
    -- Using H1, H2, H3, calculate and apply the Pythagorean theorem.
    sorry

end wire_length_l646_646052


namespace area_difference_l646_646496

noncomputable def diagonal : ℝ := 5
noncomputable def diameter : ℝ := 5

theorem area_difference :
  let s := diagonal / real.sqrt 2,
      square_area := s^2,
      r := diameter / 2,
      circle_area := real.pi * r^2
  in real.abs (circle_area - square_area - 7.1) < 0.1 :=
by
  let s := 5 / real.sqrt 2
  let square_area := s^2
  let r := 5 / 2
  let circle_area := real.pi * r^2
  have h : square_area = 12.5 := by
    sorry
  have h' : circle_area = real.pi * (5 / 2)^2 := by
    sorry
  have h'' : circle_area ≈ 19.63 := by
    sorry
  show real.abs (circle_area - square_area - 7.1) < 0.1
  sorry

end area_difference_l646_646496


namespace average_headcount_nearest_whole_number_estimation_l646_646225

def student_headcount_03_04 : ℕ := 10500
def student_headcount_04_05 : ℕ := 10700
def student_headcount_05_06_estimated : ℕ := 11300
def error_margin : ℕ := 50

theorem average_headcount_nearest_whole_number_estimation :
  |student_headcount_03_04 - error_margin| + |student_headcount_04_05 - error_margin| + |student_headcount_05_06_estimated - error_margin| + |student_headcount_03_04 + error_margin| + |student_headcount_04_05 + error_margin| + |student_headcount_05_06_estimated + error_margin| / (6 * 3) = 10833 :=
sorry

end average_headcount_nearest_whole_number_estimation_l646_646225


namespace pitchers_prepared_correct_l646_646661

-- Define the given conditions
def glasses_per_pitcher : ℝ := 4.5
def total_glasses_served : ℝ := 30

-- State the problem to prove the number of pitchers prepared.
theorem pitchers_prepared_correct :
  ceil (total_glasses_served / glasses_per_pitcher) = 7 := by
  -- Proof to be provided here
  sorry

end pitchers_prepared_correct_l646_646661


namespace min_value_of_sin_function_l646_646727

open Real

theorem min_value_of_sin_function (x : ℝ) (hx : 0 < x ∧ x < π / 2) : 
  ∃ y, (y = (2 * sin x ^ 2 + 1) / sin (2 * x)) ∧ y = √3 :=
by
  sorry

end min_value_of_sin_function_l646_646727


namespace f_strictly_increasing_intervals_l646_646391

structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ) -- angles in radians
  (cosA : ℝ)
  (a_eq : a = 4 * Real.sqrt 3)
  (b_eq : b = 4)
  (cosA_eq : cosA = -1 / 2)

axiom measure_of_B (T : Triangle) : T.B = π / 6

def f (x : ℝ) (C B : ℝ) : ℝ :=
  cos (2 * x) + (C / 2) * sin (x^2 + B)

theorem f_strictly_increasing_intervals (T : Triangle) (hB_eq : T.B = π / 6) :
  ∀ k : ℤ, ∀ x : ℝ,
    -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π → strict_mono_on (λ x, f x (π - T.A - T.B) T.B) (Icc (-π / 3 + k * π) (π / 6 + k * π)) :=
sorry

end f_strictly_increasing_intervals_l646_646391


namespace second_number_is_15_l646_646096

theorem second_number_is_15 (n : ℕ) (h₁ : n > 0)
  (h₂ : ∃ (factors : Finset ℕ), factors = {k | k ≤ 15 ∧ 15 % k = 0 ∧ n % k = 0} ∧ factors.card = 3) :
  n = 15 :=
by
  sorry

end second_number_is_15_l646_646096


namespace unit_price_ratio_l646_646203

theorem unit_price_ratio (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (1.1 * p / (1.4 * v)) / (0.85 * p / (1.3 * v)) = 13 / 11 :=
by
  sorry

end unit_price_ratio_l646_646203


namespace smallest_number_of_rectangles_l646_646120

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l646_646120


namespace inequality_and_equality_condition_l646_646449

variable {n : ℕ} (a : Fin n → ℝ)
  
theorem inequality_and_equality_condition 
  (h_pos : ∀ i, a i > 0) :
  (∑ i, 1 / a i) ≥ (n^2 : ℝ) / (∑ i, a i) 
  ∧ ((∀ i j, a i = a j) ↔ (∑ i, 1 / a i) = (n^2 : ℝ) / (∑ i, a i)) := 
sorry

end inequality_and_equality_condition_l646_646449


namespace infinitely_many_good_primes_infinitely_many_non_good_primes_l646_646381

def is_good_prime (p : ℕ) : Prop :=
∀ a b : ℕ, a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p]

theorem infinitely_many_good_primes :
  ∃ᶠ p in at_top, is_good_prime p := sorry

theorem infinitely_many_non_good_primes :
  ∃ᶠ p in at_top, ¬ is_good_prime p := sorry

end infinitely_many_good_primes_infinitely_many_non_good_primes_l646_646381


namespace geom_sequence_arith_ratio_l646_646781

variable (a : ℕ → ℝ) (q : ℝ)
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_arith : 3 * a 0 + 2 * a 1 = 2 * (1/2) * a 2)

theorem geom_sequence_arith_ratio (ha : 3 * a 0 + 2 * a 1 = a 2) :
    (a 8 + a 9) / (a 6 + a 7) = 9 := sorry

end geom_sequence_arith_ratio_l646_646781


namespace find_a_for_no_x2_term_l646_646386

theorem find_a_for_no_x2_term :
  ∀ a : ℝ, (∀ x : ℝ, (3 * x^2 + 2 * a * x + 1) * (-3 * x) - 4 * x^2 = -9 * x^3 + (-6 * a - 4) * x^2 - 3 * x) →
  (¬ ∃ x : ℝ, (-6 * a - 4) * x^2 ≠ 0) →
  a = -2 / 3 :=
by
  intros a h1 h2
  sorry

end find_a_for_no_x2_term_l646_646386


namespace train_speed_l646_646185

/-- 
If a train 1500 m long can cross an electric pole in 120 sec, 
then the speed of the train is 12.5 m/s.
-/
theorem train_speed (distance time : ℕ) (h_distance : distance = 1500) (h_time : time = 120) : 
  distance / time = 12.5 := by
  sorry

end train_speed_l646_646185


namespace find_b_l646_646377

theorem find_b 
  (a b : ℤ)
  (h1 : ∃ c : ℤ, ∀ x : ℝ, ax^3 + bx^2 + 2 = (x^2 - 2x - 1) * (a*x - c))
  (h2 : 2 - a = 0) : b = -6 :=
sorry

end find_b_l646_646377


namespace correct_judgment_l646_646455

open Real

def period_sin2x (T : ℝ) : Prop := ∀ x, sin (2 * x) = sin (2 * (x + T))
def smallest_positive_period_sin2x : Prop := ∃ T > 0, period_sin2x T ∧ ∀ T' > 0, period_sin2x T' → T ≤ T'
def smallest_positive_period_sin2x_is_pi : Prop := ∃ T, smallest_positive_period_sin2x ∧ T = π

def symmetry_cosx (L : ℝ) : Prop := ∀ x, cos (L - x) = cos (L + x)
def symmetry_about_line_cosx (L : ℝ) : Prop := L = π / 2

def p : Prop := smallest_positive_period_sin2x_is_pi
def q : Prop := symmetry_about_line_cosx (π / 2)

theorem correct_judgment : ¬ (p ∧ q) :=
by 
  sorry

end correct_judgment_l646_646455


namespace average_mileage_round_trip_l646_646643

def average_gas_mileage 
  (dist_to_friend : ℕ) 
  (mpg_sedan : ℕ) 
  (dist_back : ℕ) 
  (mpg_truck : ℕ) : ℕ :=
  let total_distance := dist_to_friend + dist_back in
  let gas_used_sedan := dist_to_friend / mpg_sedan in
  let gas_used_truck := dist_back / mpg_truck in
  let total_gas_used := gas_used_sedan + gas_used_truck in
  total_distance * 3 / total_gas_used

theorem average_mileage_round_trip : average_gas_mileage 150 25 200 15 = 18 := by {
  sorry
}

end average_mileage_round_trip_l646_646643


namespace total_cost_for_seeds_l646_646579

theorem total_cost_for_seeds :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  in total = 18.00 :=
by
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_qty := 3
  let tomato_qty := 4
  let chili_pepper_qty := 5
  let total := (pumpkin_qty * pumpkin_price) + (tomato_qty * tomato_price) + (chili_pepper_qty * chili_pepper_price)
  have h1 : total = 18.00,
  {
    sorry
  }
  exact h1

end total_cost_for_seeds_l646_646579


namespace number_of_sets_of_integers_l646_646453

theorem number_of_sets_of_integers (a b c d : ℕ) (positive_a : a > 0) (positive_b : b > 0) (positive_c : c > 0) (positive_d : d > 0)
  (h_lcm : ∀ x y z: ℕ, (x, y, z) ∈ ({a, b, c, d}.subsetsOfCard 3 : Set (Set ℕ)) → Int.lcm (Int.lcm x y) z = (3^3 * 7^5)) :
  {x : ℕ // (∃ t : Finset ℕ, t.card = 4 ∧ t = {a, b, c, d})}.card = 11457 :=
  by sorry

end number_of_sets_of_integers_l646_646453


namespace complex_sum_angle_l646_646657

open Complex Real

theorem complex_sum_angle (r : ℝ) :
  e ^ (11 * π * Complex.I / 60) + 
  e ^ (23 * π * Complex.I / 60) + 
  e ^ (35 * π * Complex.I / 60) + 
  e ^ (47 * π * Complex.I / 60) + 
  e ^ (59 * π * Complex.I / 60) = r * e ^ (7 * π * Complex.I / 12) :=
sorry

end complex_sum_angle_l646_646657


namespace range_of_k_condition_l646_646465

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := (4 - k) / x

theorem range_of_k_condition (k x1 x2 y1 y2 : ℝ) 
    (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 < y2) 
    (hA : inverse_proportion_function k x1 = y1) 
    (hB : inverse_proportion_function k x2 = y2) : 
    k < 4 :=
sorry

end range_of_k_condition_l646_646465


namespace locus_is_circle_l646_646722

open EuclideanGeometry

noncomputable def locus_of_points_p
  (A B C D P : Point)
  (e : Line)
  (O : Point)
  (r : ℝ)
  (hA : Point_on_Line e A)
  (hB : Point_on_Line e B)
  (hC : Point_on_Line e C)
  (hD : Point_on_Line e D) : Prop :=
  ∠ (P - A) (P - B) = ∠ (P - C) (P - D)

-- The main theorem statement in Lean 4
theorem locus_is_circle
  (A B C D : Point)
  (e : Line)
  (O : Point)
  (r : ℝ)
  (hA : Point_on_Line e A)
  (hB : Point_on_Line e B)
  (hC : Point_on_Line e C)
  (hD : Point_on_Line e D)
  (hO_eq : r = real.sqrt ((dist O A) * (dist O D)))
  (hO_eq' : r = real.sqrt ((dist O C) * (dist O B))) :
  ∀ P : Point, locus_of_points_p A B C D P e O r → dist P O = r :=
by
  sorry

end locus_is_circle_l646_646722


namespace average_of_distinct_k_l646_646310

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l646_646310


namespace probability_union_A_B_l646_646913

open Probability

-- Definitions of events A and B
def event_A (ω : ω) : Prop := coin_flip ω = Heads
def event_B (ω : ω) : Prop := die_roll ω = 3

-- Given conditions
axiom fair_coin : Probability(coin_flip = Heads) = 1 / 2
axiom fair_die : Probability(die_roll = 3) = 1 / 6
axiom independent_A_B : independent event_A event_B

-- Target proof statement
theorem probability_union_A_B : 
  Probability(event_A ∪ event_B) = 7 / 12 :=
sorry -- Proof is not required

end probability_union_A_B_l646_646913


namespace hyperbola_standard_equation_l646_646508

def ellipse_equation (x y : ℝ) : Prop :=
  (y^2) / 16 + (x^2) / 12 = 1

def hyperbola_equation (x y : ℝ) : Prop :=
  (y^2) / 2 - (x^2) / 2 = 1

def passes_through_point (x y : ℝ) : Prop :=
  x = 1 ∧ y = Real.sqrt 3

theorem hyperbola_standard_equation (x y : ℝ) (hx : passes_through_point x y)
  (ellipse_foci_shared : ∀ x y : ℝ, ellipse_equation x y → ellipse_equation x y)
  : hyperbola_equation x y := 
sorry

end hyperbola_standard_equation_l646_646508


namespace highest_power_of_three_l646_646846

theorem highest_power_of_three (M : ℕ) (h : M = 3132338687) : M % 9 ≠ 0 ∧ M % 3 = 0 :=
  by
  unfold M
  sorry

end highest_power_of_three_l646_646846


namespace find_h_l646_646229

def f (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

theorem find_h : ∃ a h k, (h = -3 / 2) ∧ (f x = a * (x - h)^2 + k) :=
by
  -- Proof steps would go here
  sorry

end find_h_l646_646229


namespace correct_derivative_log_base_2_l646_646141

-- Definitions of the functions involved in the problem
noncomputable def ln : ℝ → ℝ := sorry
noncomputable def log_base (b x : ℝ) : ℝ := sorry

-- The problem translated into a mathematic proof problem in Lean 4
theorem correct_derivative_log_base_2 (x : ℝ) (hx : x > 0) : 
  (derivative (λ x, log_base 2 x)) x = 1 / (x * ln 2) :=
sorry

end correct_derivative_log_base_2_l646_646141


namespace proof_math_problem_l646_646001

noncomputable def math_problem (n : ℕ) (a : Fin n → ℝ) : Prop :=
  (∀ i, 0 ≤ a i) ∧ (∑ i, a i = n) →
  (∑ i, (a i)^2 / (1 + (a i)^4)) ≤ (∑ i, 1 / (1 + (a i)))

theorem proof_math_problem (n : ℕ) (a : Fin n → ℝ) :
  math_problem n a :=
by
  sorry

end proof_math_problem_l646_646001


namespace prime_factorial_power_l646_646004

theorem prime_factorial_power (p n a : ℕ) (hp : Prime p) 
  (ha : ∀ k : ℕ, p^k ∣ n! ↔ k ≤ a) : 
  a = (n / p).floor + (n / p^2).floor + (n / p^3).floor + ... :=
by 
  -- Prove here
  sorry

end prime_factorial_power_l646_646004


namespace possible_values_2n_plus_m_l646_646892

theorem possible_values_2n_plus_m :
  ∀ (n m : ℤ), 3 * n - m < 5 → n + m > 26 → 3 * m - 2 * n < 46 → 2 * n + m = 36 :=
by sorry

end possible_values_2n_plus_m_l646_646892


namespace solution_l646_646770

noncomputable def hyperbola_asymptote_intersects_circle
    (m : ℝ) : Prop :=
    let circle_center : ℝ × ℝ := (-1, 0)
    let radius : ℝ := 2
    let d := (λ (a b c x1 y1 : ℝ), abs (a * x1 + b * y1 + c) / real.sqrt (a ^ 2 + b ^ 2))
    let AB_dist := (8 * real.sqrt 5) / 5
    let asymptote1_dist := d 4 (real.sqrt m) 0 (-1) 0
    let asymptote2_dist := d 4 (-real.sqrt m) 0 (-1) 0
    let correct_dist := real.sqrt (radius ^ 2 - (AB_dist / 2) ^ 2)
    asymptote1_dist = correct_dist ∧ asymptote2_dist = correct_dist

theorem solution : hyperbola_asymptote_intersects_circle 4 :=
by sorry

end solution_l646_646770


namespace number_of_friends_l646_646614

theorem number_of_friends (P : ℕ) (n m : ℕ) (h1 : ∀ (A B C : ℕ), (A = B ∨ A ≠ B) ∧ (B = C ∨ B ≠ C) → (n-1) * m = 15):
  P = 16 ∨ P = 18 ∨ P = 20 ∨ P = 30 :=
sorry

end number_of_friends_l646_646614


namespace min_value_of_quadratic_l646_646242

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem min_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f(x) ≤ f(y) ∧ f(x) = -1 :=
begin
  sorry
end

end min_value_of_quadratic_l646_646242


namespace Carol_saves_9_per_week_l646_646660

variable (C : ℤ)

def Carol_savings (weeks : ℤ) : ℤ :=
  60 + weeks * C

def Mike_savings (weeks : ℤ) : ℤ :=
  90 + weeks * 3

theorem Carol_saves_9_per_week (h : Carol_savings C 5 = Mike_savings 5) : C = 9 :=
by
  dsimp [Carol_savings, Mike_savings] at h
  sorry

end Carol_saves_9_per_week_l646_646660


namespace box_volume_l646_646431

def height := 12
def length := 3 * height
def width := length / 4

theorem box_volume : (height * length * width) = 3888 := by
  sorry

end box_volume_l646_646431


namespace no_integer_roots_l646_646472
open Polynomial

theorem no_integer_roots {p : ℤ[X]} (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_pa : p.eval a = 1) (h_pb : p.eval b = 1) (h_pc : p.eval c = 1) : 
  ∀ m : ℤ, p.eval m ≠ 0 :=
by
  sorry

end no_integer_roots_l646_646472


namespace correct_statement_D_valid_l646_646798

def smoking_related_to_lung_cancer (conf_level : ℝ) : Prop :=
  conf_level >= 0.99

def among_100_smokers_no_lung_cancer (conf_level : ℝ) : Prop :=
  conf_level >= 0.99 → ∃ (possible : Prop), possible = true

theorem correct_statement_D_valid (conf_level : ℝ) : 
  smoking_related_to_lung_cancer conf_level → among_100_smokers_no_lung_cancer conf_level :=
by
  intro h
  use Exists.intro true 
  sorry

end correct_statement_D_valid_l646_646798


namespace polynomial_root_condition_exists_t_l646_646871

theorem polynomial_root_condition_exists_t :
  let a b c : ℝ
  let p := polynomial.C 7 + polynomial.X * (polynomial.C 5 + polynomial.X * (polynomial.C (-3) + polynomial.X))
  let q := polynomial.X ^ 3 + polynomial.C r * polynomial.X ^ 2 + polynomial.C s * polynomial.X + polynomial.C t
  (p.is_root a ∧ p.is_root b ∧ p.is_root c) → 
  let ab := a + b
  let bc := b + c
  let ca := c + a
  (q.is_root ab ∧ q.is_root bc ∧ q.is_root ca) → 
  t = 8 :=
by {
  intros a b c p q hp hq,
  sorry
}

end polynomial_root_condition_exists_t_l646_646871


namespace intersection_M_N_l646_646836

-- Definitions
def M : set ℝ := {x | abs (x - 1) < 1}
def N : set ℝ := {x | x * (x - 3) < 0}

-- Proposition
theorem intersection_M_N : (M ∩ N) = {x | 0 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l646_646836


namespace solve_system_of_equations_l646_646483

theorem solve_system_of_equations :
  (∃ x y : ℝ, (x / y + y / x = 173 / 26) ∧ (1 / x + 1 / y = 15 / 26) ∧ ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13))) :=
by
  sorry

end solve_system_of_equations_l646_646483


namespace pumps_time_to_empty_pool_l646_646980

theorem pumps_time_to_empty_pool :
  (1 / (1 / 6 + 1 / 9) * 60) = 216 :=
by
  norm_num
  sorry

end pumps_time_to_empty_pool_l646_646980


namespace growth_ratio_bounds_l646_646106

-- Define a set A to be a good set if it consists of non-intersecting circles
structure GoodSet (A : Set Circle) : Prop :=
(non_intersecting : ∀ (c1 c2 : Circle), c1 ∈ A → c2 ∈ A → c1 ≠ c2 → ¬(c1 ∩ c2).Nonempty)

-- Define equivalence of good sets by a transformation that avoids intersections
def equivalent (A B : Set Circle) [GoodSet A] [GoodSet B] : Prop :=
∃ (f : Circle → Circle), (∀ c ∈ A, f c ∈ B) ∧ (∀ c1 c2 ∈ A, c1 ≠ c2 → ¬(f c1 ∩ f c2).Nonempty)

-- Define the sequence a_n as the number of inequivalent good sets with n elements
def a_n : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 2
| 3       := 4
| 4       := 9
| (n + 1) := sorry -- Define recurrence relation in general

-- Now state the growth ratio bounds
theorem growth_ratio_bounds : 
  2 < liminf (λ n, (a_n n)^(1/n)) ∧ limsup (λ n, (a_n n)^(1/n)) < 4 :=
sorry

end growth_ratio_bounds_l646_646106


namespace age_ratio_l646_646083

theorem age_ratio (R D : ℕ) (hR : R + 4 = 32) (hD : D = 21) : R / D = 4 / 3 := 
by sorry

end age_ratio_l646_646083


namespace find_k_l646_646167

-- Defining the conditions
variable {k : ℕ} -- k is a non-negative integer

-- Given conditions as definitions
def green_balls := 7
def purple_balls := k
def total_balls := green_balls + purple_balls
def win_amount := 3
def lose_amount := -1

-- Defining the expected value equation
def expected_value [fact (total_balls > 0)] : ℝ :=
  (green_balls.toℝ / total_balls.toℝ * win_amount) +
  (purple_balls.toℝ / total_balls.toℝ * lose_amount)

-- The required theorem/assertion to prove
theorem find_k (hk : k > 0) (h : expected_value = 1) : k = 7 :=
sorry

end find_k_l646_646167


namespace sufficient_but_not_necessary_l646_646255

theorem sufficient_but_not_necessary (a b c d : ℝ) (h₁ : a > b) (h₂ : c > d) : 
  a + c > b + d ∧ ¬(∀ (a b c d : ℝ), a + c > b + d → a > b ∧ c > d) :=
by {
  -- Sufficiency proof for a + c > b + d with the given conditions
  exact ⟨add_lt_add h₁ h₂, 
    -- Proof of non-necessity: providing a counterexample
    begin 
      intro h,
        -- Example where a + c > b + d but not (a > b and c > d)
      have h_example : (∀ (a b c d : ℝ), (∃ (a b c d : ℝ), (a = b) ∧ (c > d) ∧ (a + c > b + d)) -> ¬(a > b ∧ c > d)),
      { 
        intros a b c d h_eq,
        have h_ex := h_eq a b c d,
        have h1 := h_ex ⟨a, b, c, d, rfl, h₂, add_lt_add_right h₂ b⟩,
        exact h1,
      },
      
      -- Conclude the overall proof
      exact h_example 
    end⟩,
}

end sufficient_but_not_necessary_l646_646255


namespace smallest_number_of_rectangles_l646_646119

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end smallest_number_of_rectangles_l646_646119


namespace coins_remainder_l646_646551

theorem coins_remainder (n : ℕ) (h₁ : n % 8 = 6) (h₂ : n % 7 = 5) : n % 9 = 1 := by
  sorry

end coins_remainder_l646_646551


namespace triangle_ABC_solution_l646_646012

-- Given conditions
variables (A B C : ℝ) (a b c : ℝ)
variable (a_pos : a > 0) -- assume positive side lengths
variable (b_pos : b > 0)
variable (c_pos : c > 0)
variable (C_eq : C = real.pi / 3)
variable (sin_B_eq : real.sin B = 2 * real.sin A)
variable (c_eq : c = 2 * real.sqrt 3)

-- To prove:
-- 1. a = 2 and b = 4
-- 2. The area of triangle ABC is 2 * real.sqrt 3
theorem triangle_ABC_solution :
  c = 2 * real.sqrt 3 ∧
  real.sin B = 2 * real.sin A ∧
  C = real.pi / 3 →
  a = 2 ∧ b = 4 ∧
  (1 / 2) * a * b * real.sin C = 2 * real.sqrt 3 :=
begin
  intros h,
  obtain ⟨h1, h2, h3⟩ := h,
  sorry
end

end triangle_ABC_solution_l646_646012


namespace number_of_subsets_l646_646501

open_locale classical -- To use classical logic.

theorem number_of_subsets (M : set ℕ) (h : M = {x ∈ {n : ℕ | 0 < n} | x * (x - 3) < 0}) :
  (set.to_finset M).card = 4 :=
by sorry

end number_of_subsets_l646_646501


namespace avg_of_k_with_positive_integer_roots_l646_646325

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l646_646325


namespace digit_150th_of_5_div_13_l646_646926

theorem digit_150th_of_5_div_13 : 
    ∀ k : ℕ, (k = 150) → (fractionalPartDigit k (5 / 13) = 5) :=
by 
  sorry

end digit_150th_of_5_div_13_l646_646926


namespace green_toads_per_acre_l646_646055

-- Conditions definitions
def brown_toads_per_green_toad : ℕ := 25
def spotted_fraction : ℝ := 1 / 4
def spotted_brown_per_acre : ℕ := 50

-- Theorem statement to prove the main question
theorem green_toads_per_acre :
  (brown_toads_per_green_toad * spotted_brown_per_acre * spotted_fraction).to_nat / brown_toads_per_green_toad = 8 := 
sorry

end green_toads_per_acre_l646_646055


namespace incircle_distance_relation_l646_646448

variable (A B C Q P : Point)
variable (a b c : ℝ)
variable (a_def : a = dist B C)
variable (b_def : b = dist C A)
variable (c_def : c = dist A B)
variable (Q_is_incenter : is_incenter Q A B C)

theorem incircle_distance_relation (A B C Q P : Point) (a b c : ℝ)
  (a_def : a = dist B C) (b_def : b = dist C A) (c_def : c = dist A B)
  (Q_is_incenter : is_incenter Q A B C) :
  a * (dist P A)^2 + b * (dist P B)^2 + c * (dist P C)^2 =
  a * (dist Q A)^2 + b * (dist Q B)^2 + c * (dist Q C)^2 + (a + b + c) * (dist Q P)^2 :=
sorry

end incircle_distance_relation_l646_646448


namespace find_divisor_l646_646139

theorem find_divisor (n x : ℕ) (h1 : n = 3) (h2 : (n / x : ℝ) * 12 = 9): x = 4 := by
  sorry

end find_divisor_l646_646139


namespace parabola_focus_distance_l646_646010

theorem parabola_focus_distance :
  ∀ (P : ℝ × ℝ), 
  let F := (2, 0), l := {p : ℝ × ℝ | p.1 = -2}
  in P ∈ {p : ℝ × ℝ | p.2^2 = 8 * p.1} ∧
     (∃ A : ℝ × ℝ, A ∈ l ∧ P.2 = A.2) ∧
     (angle F (P.1 - F.1, P.2 - F.2) = real.pi * 2 / 3)
  → dist P F = 8 / 3 :=
by
  -- proof to be provided
  sorry

end parabola_focus_distance_l646_646010


namespace smallest_num_rectangles_to_cover_square_l646_646115

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l646_646115


namespace positive_difference_abs_eq_15_l646_646953

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l646_646953


namespace solve_system_of_odes_l646_646484

theorem solve_system_of_odes (C₁ C₂ : ℝ) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = (C₁ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, y t = (C₁ + C₂ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, deriv x t = 2 * x t + y t) ∧
    (∀ t, deriv y t = 4 * y t - x t) :=
by
  sorry

end solve_system_of_odes_l646_646484


namespace quadratic_polynomial_value_bound_l646_646857

theorem quadratic_polynomial_value_bound (a b : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |(x^2 + a * x + b)| ≥ 1/2 :=
by
  sorry

end quadratic_polynomial_value_bound_l646_646857


namespace coins_remainder_l646_646564

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l646_646564


namespace factorization_from_left_to_right_l646_646648

theorem factorization_from_left_to_right (a x y b : ℝ) :
  (a * (a + 1) = a^2 + a ∨
   a^2 + 3 * a - 1 = a * (a + 3) + 1 ∨
   x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) ∨
   (a - b)^3 = -(b - a)^3) →
  (x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)) := sorry

end factorization_from_left_to_right_l646_646648


namespace arithmetic_sequence_sum_l646_646669

theorem arithmetic_sequence_sum :
  (∑ k in Finset.range 10, (2 * (k + 1)) / 7) = 110 / 7 :=
by
  sorry

end arithmetic_sequence_sum_l646_646669


namespace cos_F_in_triangle_l646_646421

theorem cos_F_in_triangle (D E F : ℝ) (sin_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) 
  (h1 : sin_D = 4 / 5) 
  (h2 : cos_E = 12 / 13) 
  (D_plus_E_plus_F : D + E + F = π) :
  cos_F = -16 / 65 :=
by
  sorry

end cos_F_in_triangle_l646_646421


namespace arc_lengths_l646_646877

-- Definitions for the given conditions
def circumference : ℝ := 80  -- Circumference of the circle

-- Angles in degrees
def angle_AOM : ℝ := 45
def angle_MOB : ℝ := 90

-- Radius of the circle using the formula C = 2 * π * r
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

-- Calculate the arc lengths using the angles
noncomputable def arc_length_AM : ℝ := (angle_AOM / 360) * circumference
noncomputable def arc_length_MB : ℝ := (angle_MOB / 360) * circumference

-- The theorem stating the required lengths
theorem arc_lengths (h : circumference = 80 ∧ angle_AOM = 45 ∧ angle_MOB = 90) :
  arc_length_AM = 10 ∧ arc_length_MB = 20 :=
by
  sorry

end arc_lengths_l646_646877


namespace negate_one_even_l646_646462

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_one_even (a b c : ℕ) :
  (∃! x, x = a ∨ x = b ∨ x = c ∧ is_even x) ↔
  (∃ x y, x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧
    x ≠ y ∧ is_even x ∧ is_even y) ∨
  (is_odd a ∧ is_odd b ∧ is_odd c) :=
by {
  sorry
}

end negate_one_even_l646_646462


namespace probability_perfect_square_multiple_three_l646_646042

noncomputable def perfect_squares (n : ℕ) : List ℕ :=
  List.filter (λ k, ∃ m : ℕ, m * m = k) (List.range' 1 n)

noncomputable def multiples_of_three (l : List ℕ) : List ℕ :=
  List.filter (λ k, k % 3 = 0) l

theorem probability_perfect_square_multiple_three :
  let cards := List.range' 1 61 in
  let ps := perfect_squares 61 in
  let ps_multiples_of_three := multiples_of_three ps in
  (List.length ps_multiples_of_three : ℚ) / (List.length cards : ℚ) = 1 / 30 :=
sorry

end probability_perfect_square_multiple_three_l646_646042


namespace number_of_elements_with_first_digit_8_l646_646006

noncomputable def first_digit (n : ℕ) : ℕ :=
  let d : ℕ := n.digits 10 in d.headI

def set_S (max_k : ℕ) : Finset ℕ := 
  (Finset.range (max_k + 1)).image (λ k, 8^k)

theorem number_of_elements_with_first_digit_8 :
  ((set_S 5000).filter (λ x, first_digit x = 8)).card = 184 :=
sorry

end number_of_elements_with_first_digit_8_l646_646006


namespace group_friends_opponents_l646_646624

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l646_646624


namespace average_of_k_l646_646315

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l646_646315


namespace countFourDigitNumbersSum_l646_646819

def countFourDigitDivBy3 : Nat :=
  let total = 9999 - 1000 + 1
  total / 3

def countFourDigitMultiplesOf7 : Nat :=
  let first = 1001
  let last = 9996
  (last - first) / 7 + 1

theorem countFourDigitNumbersSum : countFourDigitDivBy3 + countFourDigitMultiplesOf7 = 4286 := by
  sorry

end countFourDigitNumbersSum_l646_646819


namespace jenna_mean_l646_646425

theorem jenna_mean :
  let scores := [84, 90, 87, 93, 88, 92] in
  (List.sum scores) / (length scores) = 89 := by
  sorry

end jenna_mean_l646_646425


namespace avg_weight_b_c_43_l646_646493

noncomputable def weights_are_correct (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧ (A + B) / 2 = 40 ∧ B = 31

theorem avg_weight_b_c_43 (A B C : ℝ) (h : weights_are_correct A B C) : (B + C) / 2 = 43 :=
by sorry

end avg_weight_b_c_43_l646_646493


namespace second_quadrant_m_l646_646788

-- Definitions for given points and vectors
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 1 }
def B : Point := { x := 2, y := 3 }
def C : Point := { x := 4, y := 2 }

-- Definition for vector subtraction
def vector_sub (p q : Point) : Point :=
  { x := p.x - q.x, y := p.y - q.y }

-- Definitions for AB, AC and the point P
def AB : Point := vector_sub B A
def AC : Point := vector_sub C A

def P (m : ℝ) : Point :=
  { x := AB.x - m * AC.x, y := AB.y - m * AC.y }

-- The main theorem to prove
theorem second_quadrant_m (m : ℝ) (P : Point) :
  (P = { x := AB.x - m * AC.x, y := AB.y - m * AC.y }) →
  (P.x < 0) →
  (P.y > 0) →
  m ∈ Ioo (2/3 : ℝ) 3 :=
by
  sorry

end second_quadrant_m_l646_646788


namespace find_13th_result_l646_646540

theorem find_13th_result
  (avg_all : ℕ → ℕ → ℕ)
  (avg_first12 : ℕ → ℕ → ℕ)
  (avg_last12 : ℕ → ℕ → ℕ)
  (sum_all : ℕ)
  (sum_first12 : ℕ)
  (sum_last12 : ℕ) :
  avg_all 25 20 = sum_all →
  avg_first12 12 14 = sum_first12 →
  avg_last12 12 17 = sum_last12 →
  sum_all - (sum_first12 + sum_last12) = 128 :=
by {
  intro h1,
  intro h2,
  intro h3,
  sorry
}

end find_13th_result_l646_646540


namespace percentage_of_reporters_not_covering_local_politics_country_x_l646_646228

theorem percentage_of_reporters_not_covering_local_politics_country_x
    (total_reporters : ℕ)
    (h_total_reporters: total_reporters = 100)
    (percentage_local_politics : ℕ)
    (h_percentage_local_politics : percentage_local_politics = 20)
    (percentage_not_covering_politics : ℕ)
    (h_percentage_not_covering_politics : percentage_not_covering_politics = 75) :
    let reporters_local_politics := total_reporters * percentage_local_politics / 100 in
    let reporters_not_covering_politics := total_reporters * percentage_not_covering_politics / 100 in
    let reporters_covering_politics := total_reporters - reporters_not_covering_politics in
    let reporters_covering_politics_but_not_local := reporters_covering_politics - reporters_local_politics in
    reporters_covering_politics_but_not_local * 100 / reporters_covering_politics = 20 := 
by 
  sorry

end percentage_of_reporters_not_covering_local_politics_country_x_l646_646228


namespace triangles_with_equal_bases_and_heights_have_equal_areas_l646_646104

theorem triangles_with_equal_bases_and_heights_have_equal_areas
  (b1 b2 h1 h2 : ℝ) (triangle1 triangle2 : Type)
  (base1 : triangle1 → ℝ) (height1 : triangle1 → ℝ) (area1 : triangle1 → ℝ)
  (base2 : triangle2 → ℝ) (height2 : triangle2 → ℝ) (area2 : triangle2 → ℝ)
  (hb1 : ∀ t, base1 t = b1)
  (hb2 : ∀ t, base2 t = b2)
  (hh1 : ∀ t, height1 t = h1)
  (hh2 : ∀ t, height2 t = h2)
  (same_base : b1 = b2)
  (same_height : h1 = h2)
  (area_formula1 : ∀ t, area1 t = (base1 t * height1 t) / 2)
  (area_formula2 : ∀ t, area2 t = (base2 t * height2 t) / 2) :
  ∀ t1 t2, area1 t1 = area2 t2 := 
by
  intros t1 t2
  rw [area_formula1, area_formula2]
  rw [hb1, hb2, hh1, hh2]
  rw [same_base, same_height]
  exact sorry

end triangles_with_equal_bases_and_heights_have_equal_areas_l646_646104


namespace possible_number_of_friends_l646_646628

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end possible_number_of_friends_l646_646628


namespace base_of_power_expr_l646_646795

-- Defining the power expression as a condition
def power_expr : ℤ := (-4 : ℤ) ^ 3

-- The Lean statement for the proof problem
theorem base_of_power_expr : ∃ b : ℤ, (power_expr = b ^ 3) ∧ (b = -4) := 
sorry

end base_of_power_expr_l646_646795


namespace average_value_of_set_l646_646208

variable {T : Finset ℕ}
variable {b_1 b_m : ℕ}

theorem average_value_of_set (h1 : T.nonempty)
  (h2 : ∀ x, x ∈ T → 0 < x)
  (h3 : ∑ x in (T.erase b_m), x = 34 * (T.card - 1))
  (h4 : ∑ x in (T.erase b_1).erase b_m, x = 37 * (T.card - 2))
  (h5 : ∑ x in (T.erase b_1), x + b_m = 42 * (T.card - 1))
  (h6 : b_m = b_1 + 80) :
  ((∑ x in T, x) : ℚ) / T.card = 38.82 := 
sorry

end average_value_of_set_l646_646208


namespace find_height_l646_646085

theorem find_height (x : ℝ) :
  (∃ x, 35 = 0.1 * x + 20) → x = 150 :=
by
  intro h
  cases h with a ha
  have h1 : 35 - 20 = 0.1 * a := by linarith
  have h2 : 15 / 0.1 = a := by linarith
  have h3 : a = 150 := by linarith
  exact h3

end find_height_l646_646085


namespace ring_toss_total_amount_l646_646900

-- Defining the amounts made in the two periods
def amount_first_period : Nat := 382
def amount_second_period : Nat := 374

-- The total amount made
def total_amount : Nat := amount_first_period + amount_second_period

-- Statement that the total amount calculated is equal to the given answer
theorem ring_toss_total_amount :
  total_amount = 756 := by
  sorry

end ring_toss_total_amount_l646_646900


namespace tangent_line_equation_sequence_general_formula_tangent_parallel_range_split_numbers_cube_l646_646548

-- Problem 1
theorem tangent_line_equation : 
  ∀ (x : ℝ), (1, 1) ∈ ({p | ∃ (y : ℝ), p = (x, x * (3 * log x + 1))} : set (ℝ × ℝ)) → y - 1 = 4 * (x - 1) :=
sorry

-- Problem 2
theorem sequence_general_formula (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ n ∈ ℕ, a (n + 1) = a n / (1 + a n)) → ∀ n ∈ ℕ, a n = 1 / n :=
sorry

-- Problem 3
theorem tangent_parallel_range (a x : ℝ) (f : ℝ → ℝ) :
  f = fun x => log x + a * x ∧ (∀ x, deriv f x = 2x - y) → a < 2 :=
sorry

-- Problem 4
theorem split_numbers_cube (n m : ℕ) :
  (n > 1 ∧ (∃ k ∈ ℕ, m = 2015 ∧ ((k + 2) * (k - 1)) / 2 = m )) → m = 45 :=
sorry

end tangent_line_equation_sequence_general_formula_tangent_parallel_range_split_numbers_cube_l646_646548


namespace average_of_distinct_k_l646_646309

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l646_646309


namespace average_of_k_l646_646316

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l646_646316


namespace angle_BAC_eq_60_l646_646895

variables {A B C H I O : Type}
variables [ScaleneTriangle ABC] [AltitudeIntersection H ABC]
variables [Incenter I ABC] [Circumcenter O BHC] [LiesOnSegment I OA]

theorem angle_BAC_eq_60 (h1 : ScaleneTriangle ABC)
                       (h2 : AltitudeIntersection H ABC)
                       (h3 : Incenter I ABC)
                       (h4 : Circumcenter O BHC)
                       (h5 : LiesOnSegment I OA) : 
                       angle A B C = 60 :=
by sorry

end angle_BAC_eq_60_l646_646895


namespace omino_tilings_2_by_10_l646_646370

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n+2) => fib n + fib (n+1)

def omino_tilings (n : ℕ) : ℕ :=
  fib (n + 1)

theorem omino_tilings_2_by_10 : omino_tilings 10 = 3025 := by
  sorry

end omino_tilings_2_by_10_l646_646370


namespace cover_square_with_rectangles_l646_646136

theorem cover_square_with_rectangles :
  ∃ (n : ℕ), 
    ∀ (a b : ℕ), 
      (a = 3) ∧ 
      (b = 4) ∧ 
      (n = (12 * 12) / (a * b)) ∧ 
      (144 = n * (a * b)) ∧ 
      (3 * 4 = a * b) 
  → 
    n = 12 :=
by
  sorry

end cover_square_with_rectangles_l646_646136


namespace avg_k_value_l646_646330

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l646_646330


namespace average_of_k_l646_646317

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l646_646317


namespace Nagel_point_exists_l646_646474

variables {a b c : ℝ} (ABC : Triangle ℝ)
variables {A' B' C' : Point ℝ}
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
#check semi_perimeter

def tangency_point_segment_lengths (a b c : ℝ) (p : ℝ) :=
  ∃ (A' B' C' : Point ℝ),
  A' = p - c ∧ B' = p - a ∧ C' = p - b

def Nagel_point_concurrency_condition (ABC : Triangle ℝ) 
  (A' B' C' : Point ℝ) : Prop :=
  let p := semi_perimeter ABC.a ABC.b ABC.c in
  tangency_point_segment_lengths ABC.a ABC.b ABC.c p ∧
  concurrent_at_nagel_point ABC A' B' C'

theorem Nagel_point_exists (ABC : Triangle ℝ)
  (A' B' C' : Point ℝ) :
  Nagel_point_concurrency_condition ABC A' B' C' :=
by
-- Proof omitted
sorry

end Nagel_point_exists_l646_646474


namespace quotient_is_seven_l646_646851

def dividend : ℕ := 22
def divisor : ℕ := 3
def remainder : ℕ := 1

theorem quotient_is_seven : ∃ quotient : ℕ, dividend = (divisor * quotient) + remainder ∧ quotient = 7 := by
  sorry

end quotient_is_seven_l646_646851


namespace max_average_of_set_l646_646901

theorem max_average_of_set (S : Finset ℕ) (hS : S.card = 6) (distinct_elems : S.val.nodup)
  (s1 s2 s3 s4 s5 s6 : ℕ)
  (h_order : s1 < s2 ∧ s2 < s3 ∧ s3 < s4 ∧ s4 < s5 ∧ s5 < s6)
  (h_min_avg : (s1 + s2) = 10)
  (h_max_avg : (s5 + s6) = 44) :
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 ≤ 15.5 :=
sorry

end max_average_of_set_l646_646901


namespace total_payment_is_correct_l646_646976

noncomputable def total_payment (a_days b_days c_days : ℕ) (b_share : ℚ) : ℚ := 
  let aWork := (1 : ℚ) / a_days
  let bWork := (1 : ℚ) / b_days
  let cWork := (1 : ℚ) / c_days
  let totalWork := aWork + bWork + cWork
  let b_work_share := bWork / totalWork
  let total_payment := b_share / b_work_share
  total_payment

theorem total_payment_is_correct : 
  total_payment 6 8 12 600 = 1800 := 
by
  dsimp [total_payment]
  norm_num
  sorry

end total_payment_is_correct_l646_646976


namespace area_sum_unit_square_l646_646441

noncomputable def sum_of_areas_of_triangles : ℝ :=
  ∑' i : ℕ, if h : i > 0 then 1 / (4 * 2^(2*(i-1) - 1)) else 0

theorem area_sum_unit_square (ABCD : ℝ × ℝ × ℝ × ℝ)
  (h_unit : (ABCD.fst).fst = 1 ∧ (ABCD.snd).fst = 1)
  (mid_Q1 : (Q1 : ℝ × ℝ) → Q1 = (0.5, 0))
  (P_i_func : (P_i : ℕ → ℝ × ℝ) → ∀ i, P_i i = intersection (A, Q i) (B, D))
  (foot_Q_ip1 : (Q_ip1 : ℕ → ℝ × ℝ) → ∀ i, Q_ip1 (i + 1) = foot_perpendicular (P_i i) (C, D)) :
  sum_of_areas_of_triangles = 1 / 4 :=
by
  sorry

end area_sum_unit_square_l646_646441
