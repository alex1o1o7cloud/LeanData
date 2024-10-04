import Complex
import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Algebra.Ring.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Comb
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.List.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction.Finite
import Mathlib.SetTheory.Set.Basic
import Mathlib.Tactic
import Mathlib.Tactic.CaseSimp
import Mathlib.Topology.Basic
import Real
import data.set.basic

namespace gcd_5280_12155_l624_624111

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end gcd_5280_12155_l624_624111


namespace fraction_addition_l624_624627

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624627


namespace find_n_l624_624258

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_lcm1 : Nat.lcm 40 n = 120) (h_lcm2 : Nat.lcm n 45 = 180) : n = 12 :=
sorry

end find_n_l624_624258


namespace f1_monotonically_increasing_f2_monotonically_decreasing_in_interval_l624_624849

open Classical

noncomputable def f1 (x : ℝ) : ℝ := (x^2 - 2*x) * Real.exp x
noncomputable def f2 (x : ℝ) (m : ℝ) : ℝ := (x^2 + m * x) * Real.exp x

theorem f1_monotonically_increasing :
  (∀ x : ℝ, f1(x) ≤ f1(x+1)) :=
by sorry

theorem f2_monotonically_decreasing_in_interval (m : ℝ) :
  (∀ x ∈ Set.Icc (1:ℝ) (3:ℝ), f2(x, m) ≤ f2(x, m+1)) → m ≤ -15/4 :=
by sorry

end f1_monotonically_increasing_f2_monotonically_decreasing_in_interval_l624_624849


namespace find_number_l624_624872

theorem find_number (x q : ℕ) (h1 : x = 7 * q) (h2 : q + x + 7 = 175) : x = 147 := 
by
  sorry

end find_number_l624_624872


namespace total_red_marbles_l624_624353

theorem total_red_marbles (jessica_marbles sandy_marbles alex_marbles : ℕ) (dozen : ℕ)
  (h_jessica : jessica_marbles = 3 * dozen)
  (h_sandy : sandy_marbles = 4 * jessica_marbles)
  (h_alex : alex_marbles = jessica_marbles + 2 * dozen)
  (h_dozen : dozen = 12) :
  jessica_marbles + sandy_marbles + alex_marbles = 240 :=
by
  sorry

end total_red_marbles_l624_624353


namespace problem_l624_624073

def g (x : ℝ) : ℝ := Real.sqrt (x / 2)

theorem problem 
  (h₀ : 0 < (1 / 2))
  (h₁ : 0 < (g (1 / 2) + 1))
  (h₂ : 0 < (g (g (1 / 2) + 1) + 1))
  (h₃ : 0 < (g (g (g (1 / 2) + 1) + 1) + 1))
  (h₄ : 0 < (g (g (g (g (1 / 2) + 1) + 1) + 1) + 1))
  (h₅ : 0 < (cos (15 / 4))):
  g (g (g (g (g (1 / 2) + 1) + 1) + 1) + 1) = cos (15 / 4 * Real.pi / 180) ∧ 19 = 15 + 4 :=
sorry

end problem_l624_624073


namespace find_all_functions_l624_624935

theorem find_all_functions (n : ℕ) (h_pos : 0 < n) (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x)^n * f (x + y) = (f x)^(n + 1) + x^n * f y) ↔
  (if n % 2 = 1 then ∀ x, f x = 0 ∨ f x = x else ∀ x, f x = 0 ∨ f x = x ∨ f x = -x) :=
sorry

end find_all_functions_l624_624935


namespace cubic_solution_l624_624424

theorem cubic_solution (a b c d : ℝ) (h_cond1 : a * d = b * c) (h_cond2 : b * d < 0) (h_neq0 : a * d ≠ 0) :
  (∃ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0) ↔
  (∃ x : ℝ, x = -b / a ∨ x = sqrt (-d / b) ∨ x = -sqrt (-d / b)) :=
by
  sorry

end cubic_solution_l624_624424


namespace num_ways_to_place_digits_l624_624311

theorem num_ways_to_place_digits : 
  ∃ n : ℕ, n = (5!) ∧ n = 120 :=
begin
  use 120,
  split,
  { rw nat.factorial,
    norm_num },
  { refl }
end

end num_ways_to_place_digits_l624_624311


namespace left_ant_speed_l624_624106

noncomputable def ant_speed : ℝ :=
  let sqrt3 : ℝ := Real.sqrt 3
  let left_velocity := 3 - sqrt3
  Real.sqrt (left_velocity^2 + (-(Real.sqrt 2) * left_velocity)^2)

theorem left_ant_speed :
  ∀ (x1 y1 x2 y2 : ℝ),
    x1 = -1 ∧ y1 = 1 ∧ x2 = 1 ∧ y2 = 1 ∧  
    ∀ t : ℝ, let mx := (x1 + x2) / 2 + t, my := 1
             in (mx, my) = ((- Real.sqrt 2 / 2), 1 / 2) →
    ant_speed = 3 * Real.sqrt 3 - 3 := 
by
  sorry

end left_ant_speed_l624_624106


namespace max_volume_of_ideal_gas_l624_624404

variables {P T P_0 T_0 a b c R : ℝ}
variables (h_eq : (P / P_0 - a)^2 + (T / T_0 - b)^2 = c^2)
variables (h_ineq : c^2 < a^2 + b^2)

theorem max_volume_of_ideal_gas :
  (V : ℝ) = (R * T_0 / P_0) * ((a * sqrt(a^2 + b^2 - c^2) + b * c) / (b * sqrt(a^2 + b^2 - c^2) - a * c)) :=
sorry

end max_volume_of_ideal_gas_l624_624404


namespace max_heaps_l624_624021

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624021


namespace probability_of_last_two_marbles_one_green_one_red_l624_624510

theorem probability_of_last_two_marbles_one_green_one_red : 
    let total_marbles := 10
    let blue := 4
    let white := 3
    let red := 2
    let green := 1
    let total_ways := Nat.choose total_marbles 8
    let favorable_ways := Nat.choose (total_marbles - red - green) 6
    total_ways = 45 ∧ favorable_ways = 28 →
    (favorable_ways : ℚ) / total_ways = 28 / 45 :=
by
    intros total_marbles blue white red green total_ways favorable_ways h
    sorry

end probability_of_last_two_marbles_one_green_one_red_l624_624510


namespace arithmetic_series_sum_l624_624119

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 56
  let n := 19
  let pairs_sum := (n-1) / 2 * (-3)
  let single_term := 56
  2 - 5 + 8 - 11 + 14 - 17 + 20 - 23 + 26 - 29 + 32 - 35 + 38 - 41 + 44 - 47 + 50 - 53 + 56 = 29 :=
by
  sorry

end arithmetic_series_sum_l624_624119


namespace largest_integer_n_l624_624210

theorem largest_integer_n (n : ℤ) (h : n^2 - 13 * n + 40 < 0) : n = 7 :=
by
  sorry

end largest_integer_n_l624_624210


namespace area_of_figure_l624_624901

theorem area_of_figure :
  let base_rect_area := 10 * 4
  let middle_rect_area := 7 * 4
  let topmost_rect_area := 3 * 3
  base_rect_area + middle_rect_area + topmost_rect_area = 77 :=
by
  -- Definition of the areas as given in conditions
  let base_rect_area := 10 * 4
  let middle_rect_area := 7 * 4
  let topmost_rect_area := 3 * 3
  -- Definition of total area
  let total_area := base_rect_area + middle_rect_area + topmost_rect_area
  -- Proof that total_area = 77
  show total_area = 77, by
    -- Substitute values
    let base_rect_area := 40
    let middle_rect_area := 28
    let topmost_rect_area := 9
    let total_area := 40 + 28 + 9
    -- Check equality
    have h : total_area = 77 := by sorry
    exact h

end area_of_figure_l624_624901


namespace karan_borrowed_years_l624_624037

noncomputable def simple_interest_time (P I r : ℝ) : ℝ :=
  I / (P * r)

theorem karan_borrowed_years : 
  simple_interest_time 5266.23 2843.77 0.06 ≈ 9 :=
by
  sorry

end karan_borrowed_years_l624_624037


namespace remaining_units_correct_l624_624547

-- Definitions based on conditions
def total_units : ℕ := 2000
def fraction_built_in_first_half : ℚ := 3/5
def additional_units_by_october : ℕ := 300

-- Calculate units built in the first half of the year
def units_built_in_first_half : ℚ := fraction_built_in_first_half * total_units

-- Remaining units after the first half of the year
def remaining_units_after_first_half : ℚ := total_units - units_built_in_first_half

-- Remaining units after building additional units by October
def remaining_units_to_be_built : ℚ := remaining_units_after_first_half - additional_units_by_october

-- Theorem statement: Prove remaining units to be built is 500
theorem remaining_units_correct : remaining_units_to_be_built = 500 := by
  sorry

end remaining_units_correct_l624_624547


namespace find_x_l624_624214

noncomputable def required_x : Real :=
  sqrt 80.859375

theorem find_x (x: ℝ) (h1: 4 * sqrt (9 + x) + 4 * sqrt (9 - x) = 10 * sqrt 3) (h2: 0 < x) : 
  x = required_x :=
by
  sorry

end find_x_l624_624214


namespace fraction_filled_in_5_minutes_l624_624529

-- Conditions
def fill_time : ℕ := 55 -- Total minutes to fill the cistern
def duration : ℕ := 5  -- Minutes we are examining

-- The theorem to prove that the fraction filled in 'duration' minutes is 1/11
theorem fraction_filled_in_5_minutes : (duration : ℚ) / (fill_time : ℚ) = 1 / 11 :=
by
  have fraction_per_minute : ℚ := 1 / fill_time
  have fraction_in_5_minutes : ℚ := duration * fraction_per_minute
  sorry -- Proof steps would go here, if needed.

end fraction_filled_in_5_minutes_l624_624529


namespace valid_parameterizations_l624_624079

-- Condition: Line equation
def line_eq (x y : ℝ) : Prop := (y = 3 * x + 4)

-- Parameterizations
def paramA (t : ℝ) : ℝ × ℝ := (0, 4) + t • (1, 3)
def paramB (t : ℝ) : ℝ × ℝ := (-4/3, 0) + t • (-3, -1)
def paramC (t : ℝ) : ℝ × ℝ := (2, 10) + t • (9, 3)
def paramD (t : ℝ) : ℝ × ℝ := (1, 1) + t • (1/3, 1)
def paramE (t : ℝ) : ℝ × ℝ := (-4, 0) + t • (1/3, 1)

-- Valid parameterization
def valid_param (param : ℝ → ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ) (t : ℝ), param t = (x, y) ∧ line_eq x y

-- Statement
theorem valid_parameterizations :
  valid_param paramA ∧
  valid_param paramB ∧
  valid_param paramE :=
sorry

end valid_parameterizations_l624_624079


namespace probability_of_first_yellow_is_0_5_probability_of_second_yellow_is_0_5_l624_624800

noncomputable def bag_contents : List String := ["a", "b", "c", "d"]

def is_yellow (ball : String) : Bool :=
  ball = "a" ∨ ball = "b"

def probability_of_first_yellow : ℚ :=
  let yellow_count := bag_contents.countp is_yellow
  (yellow_count : ℚ) / (bag_contents.length : ℚ)

def probability_of_second_yellow : ℚ :=
  let outcomes := bag_contents.product bag_contents.filter (≠)
  let yellow_second_count := outcomes.count (λ ⟨x, y⟩, is_yellow y)
  (yellow_second_count : ℚ) / (outcomes.length : ℚ)

theorem probability_of_first_yellow_is_0_5 :
  probability_of_first_yellow = 0.5 :=
by sorry

theorem probability_of_second_yellow_is_0_5 :
  probability_of_second_yellow = 0.5 :=
by sorry

end probability_of_first_yellow_is_0_5_probability_of_second_yellow_is_0_5_l624_624800


namespace max_heaps_l624_624022

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624022


namespace intersection_count_is_one_l624_624758

theorem intersection_count_is_one :
  (∀ x y : ℝ, y = 2 * x^3 + 6 * x + 1 → y = -3 / x^2) → ∃! p : ℝ × ℝ, p.2 = 2 * p.1^3 + 6 * p.1 + 1 ∧ p.2 = -3 / p.1 :=
sorry

end intersection_count_is_one_l624_624758


namespace symmetric_point_in_first_quadrant_l624_624333

def symmetric_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

def quadrant (p : ℝ × ℝ) : string :=
  if p.1 > 0 ∧ p.2 > 0 then "I"
  else if p.1 < 0 ∧ p.2 > 0 then "II"
  else if p.1 < 0 ∧ p.2 < 0 then "III"
  else if p.1 > 0 ∧ p.2 < 0 then "IV"
  else "On Axis"

theorem symmetric_point_in_first_quadrant (x y : ℝ) (hx : x < 0) (hy : y > 0) :
  quadrant (symmetric_y_axis (x, y)) = "I" :=
by
  sorry

end symmetric_point_in_first_quadrant_l624_624333


namespace domain_translation_l624_624265

theorem domain_translation (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < 3 * x + 2 ∧ 3 * x + 2 < 1 → (∃ y : ℝ, f (3 * x + 2) = y)) →
  (∀ x : ℝ, ∃ y : ℝ, f (2 * x - 1) = y ↔ (3 / 2) < x ∧ x < 3) :=
sorry

end domain_translation_l624_624265


namespace sum_max_min_S_l624_624384

theorem sum_max_min_S (x y z : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) 
    (h1 : x + y = 10) (h2 : y + z = 8) : 
    let S := x + z in
    (S.min + S.max = 20) :=
by
  sorry

end sum_max_min_S_l624_624384


namespace fraction_addition_l624_624625

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624625


namespace closest_point_on_line_l624_624213

open Real

-- Definition of the line
def line (x : ℝ) : ℝ := (1 / 2) * x + 3

-- Given point
def point : ℝ × ℝ := (2, -1)

-- The closest point on the line to the given point
def closest_point : ℝ × ℝ := (0, 3)

theorem closest_point_on_line :
  ∀ p : ℝ × ℝ, p = (0, 3) ↔ ∀ x : ℝ, line x = p.2 ∧ (forall q : ℝ × ℝ, (line q.1, q.2) ∈ Metric.ball p q.2 → q = (2, -1)) :=
by
  sorry

end closest_point_on_line_l624_624213


namespace additionalPeopleNeededToMowLawn_l624_624799

def numberOfPeopleNeeded (people : ℕ) (hours : ℕ) : ℕ :=
  (people * 8) / hours

theorem additionalPeopleNeededToMowLawn : numberOfPeopleNeeded 4 3 - 4 = 7 :=
by
  sorry

end additionalPeopleNeededToMowLawn_l624_624799


namespace angle_C_is_75_l624_624343

noncomputable def A : ℝ := 60
noncomputable def B : ℝ := 45

theorem angle_C_is_75 :
  (|Real.cos A.to_real - 1/2| + 2 * (1 - Real.tan B.to_real)^2 = 0) ->
  (A + B < 180) ->
  ∃ C : ℝ, C = 75 :=
by
  intro h1 h2
  use 75
  sorry

end angle_C_is_75_l624_624343


namespace leak_empty_time_l624_624046

variables {A L : ℝ}

def TankFillRateWithoutLeak : Prop := A = 1 / 2
def CombinedFillRateWithLeak : Prop := A - L = 1 / 3

theorem leak_empty_time
  (h1 : TankFillRateWithoutLeak)
  (h2 : CombinedFillRateWithLeak) :
  (1 / L) = 6 :=
by
  sorry

end leak_empty_time_l624_624046


namespace ratio_of_areas_l624_624446

theorem ratio_of_areas (s : ℝ) 
  (h1 : ∀ (s : ℝ), s > 0) : 
  let R_long := 1.2 * s,
      R_short := 0.8 * s,
      area_R := R_long * R_short,
      area_S := s^2
  in area_R / area_S = 24 / 25 :=
by
  let R_long := 1.2 * s
  let R_short := 0.8 * s
  let area_R := R_long * R_short
  let area_S := s^2
  have h2 : s > 0 := h1 s
  have h3 : area_R = 0.96 * s^2 := by sorry
  have h4 : area_R / area_S = 0.96 := by sorry
  have h5 : 0.96 = 24 / 25 := by norm_num
  exact eq.trans h4 h5

end ratio_of_areas_l624_624446


namespace max_heaps_l624_624018

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624018


namespace highest_y_coordinate_of_ellipse_eq_l624_624138

theorem highest_y_coordinate_of_ellipse_eq : 
  (∃ x y : ℝ, (x * x / 49 + (y - 3) * (y - 3) / 25 = 1) ∧ 
  (∀ y' : ℝ, (∃ x' : ℝ, x' * x' / 49 + (y' - 3) * (y' - 3) / 25 = 1) → y' ≤ 8)) := 
begin
  sorry
end

end highest_y_coordinate_of_ellipse_eq_l624_624138


namespace line_equation_conditions_l624_624780

theorem line_equation_conditions 
  (l : ℝ → ℝ) 
  (slope_cond : ∃ m b, l = (λ x, m * x + b) ∧ m = 3 / 4)
  (area_cond : ∃ x y, (0, y) ∈ (λ x, m * x + b).graph ∧ (x, 0) ∈ (λ x, m * x + b).graph ∧ 1 / 2 * |x| * |y| = 6) 
  (point_cond : l 4 = -3 ∧ ∃ a b, l = (λ x, x / a + x / b) ∧ a ≠ 0 ∧ b ≠ 0 ∧ |a| = |b|) :
  ∃ (eqn : ℝ → ℝ) (c₁ c₂ : ℝ), 
    eqn = (λ x, 3 / 4 * x + c₁) ∨ 
    eqn = (λ x, x / 1 + x / 1 - c₂) ∨ 
    eqn = (λ x, 3 * x + 4 * eqn x) := 
sorry

end line_equation_conditions_l624_624780


namespace total_red_marbles_l624_624354

theorem total_red_marbles (jessica_marbles sandy_marbles alex_marbles : ℕ) (dozen : ℕ)
  (h_jessica : jessica_marbles = 3 * dozen)
  (h_sandy : sandy_marbles = 4 * jessica_marbles)
  (h_alex : alex_marbles = jessica_marbles + 2 * dozen)
  (h_dozen : dozen = 12) :
  jessica_marbles + sandy_marbles + alex_marbles = 240 :=
by
  sorry

end total_red_marbles_l624_624354


namespace add_fractions_l624_624660

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624660


namespace add_fractions_l624_624662

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624662


namespace siding_cost_l624_624973

-- Define the dimensions and cost of the siding
def wall_width := 8
def wall_height := 10
def roof_panel_width := 8
def roof_panel_height := 5
def siding_section_width := 10
def siding_section_height := 12
def siding_section_cost := 30.50

-- Calculate areas
def wall_area := wall_width * wall_height
def roof_panel_area := roof_panel_width * roof_panel_height
def total_roof_area := 2 * roof_panel_area
def total_siding_area := wall_area + total_roof_area

-- Calculate number of sections needed
def siding_section_area := siding_section_width * siding_section_height
def sections_needed := (total_siding_area / siding_section_area).ceil.toNat

-- Calculate total cost
def total_cost := sections_needed * siding_section_cost

-- Theorem to prove the total cost
theorem siding_cost : total_cost = 61 := by
  sorry

end siding_cost_l624_624973


namespace add_fractions_l624_624571

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624571


namespace inequality_solution_set_l624_624787

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ -2 ≤ x ∧ x < 1 := 
sorry

end inequality_solution_set_l624_624787


namespace bounces_less_than_50_l624_624140

noncomputable def minBouncesNeeded (initialHeight : ℝ) (bounceFactor : ℝ) (thresholdHeight : ℝ) : ℕ :=
  ⌈(Real.log (thresholdHeight / initialHeight) / Real.log (bounceFactor))⌉₊

theorem bounces_less_than_50 :
  minBouncesNeeded 360 (3/4 : ℝ) 50 = 8 :=
by
  sorry

end bounces_less_than_50_l624_624140


namespace remaining_units_correct_l624_624548

-- Definitions based on conditions
def total_units : ℕ := 2000
def fraction_built_in_first_half : ℚ := 3/5
def additional_units_by_october : ℕ := 300

-- Calculate units built in the first half of the year
def units_built_in_first_half : ℚ := fraction_built_in_first_half * total_units

-- Remaining units after the first half of the year
def remaining_units_after_first_half : ℚ := total_units - units_built_in_first_half

-- Remaining units after building additional units by October
def remaining_units_to_be_built : ℚ := remaining_units_after_first_half - additional_units_by_october

-- Theorem statement: Prove remaining units to be built is 500
theorem remaining_units_correct : remaining_units_to_be_built = 500 := by
  sorry

end remaining_units_correct_l624_624548


namespace train_pass_time_correct_l624_624746

-- Define the conditions
def length_of_train : ℝ := 200  -- in meters
def length_of_bridge : ℝ := 180  -- in meters
def speed_of_train_kmh : ℝ := 65  -- in kilometers per hour

-- Convert speed from km/h to m/s (1 km = 1000 meters, 1 hour = 3600 seconds)
def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

-- Calculate the total distance to be covered
def total_distance : ℝ := length_of_train + length_of_bridge

-- Calculate the time to pass the bridge
def time_to_pass_bridge : ℝ := total_distance / speed_of_train_ms

-- The proposed proof statement
theorem train_pass_time_correct : time_to_pass_bridge ≈ 21.04 := by
  sorry  -- Proof goes here

end train_pass_time_correct_l624_624746


namespace add_fractions_l624_624663

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624663


namespace add_fractions_l624_624738

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624738


namespace ratio_of_large_rooms_l624_624153

-- Definitions for the problem conditions
def total_classrooms : ℕ := 15
def total_students : ℕ := 400
def desks_in_large_room : ℕ := 30
def desks_in_small_room : ℕ := 25

-- Define x as the number of large (30-desk) rooms and y as the number of small (25-desk) rooms
variables (x y : ℕ)

-- Two conditions provided by the problem
def classrooms_condition := x + y = total_classrooms
def students_condition := desks_in_large_room * x + desks_in_small_room * y = total_students

-- Our main theorem to prove
theorem ratio_of_large_rooms :
  classrooms_condition x y →
  students_condition x y →
  (x : ℚ) / (total_classrooms : ℚ) = 1 / 3 :=
by
-- Here we would have our proof, but we leave it as "sorry" since the task only requires the statement.
sorry

end ratio_of_large_rooms_l624_624153


namespace f_of_f_of_inv_four_l624_624441

noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then real.log x / real.log 2 else 3 ^ x

theorem f_of_f_of_inv_four : 
  f (f (1 / 4)) = 1 / 9 := 
sorry

end f_of_f_of_inv_four_l624_624441


namespace avg_speed_is_20_l624_624074

-- Define the total distance and total time
def total_distance : ℕ := 100
def total_time : ℕ := 5

-- Define the average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The theorem to prove the average speed given the distance and time
theorem avg_speed_is_20 : average_speed total_distance total_time = 20 :=
by
  sorry

end avg_speed_is_20_l624_624074


namespace logical_contradiction_l624_624477

-- Definitions based on the conditions
def all_destroying (x : Type) : Prop := ∀ y : Type, y ≠ x → y → false
def indestructible (x : Type) : Prop := ∀ y : Type, y = x → y → false

theorem logical_contradiction (x : Type) :
  (all_destroying x ∧ indestructible x) → false :=
by
  sorry

end logical_contradiction_l624_624477


namespace sets_not_equal_l624_624088

theorem sets_not_equal : 
  (set_of (λ x : ℝ, ∃ y : ℝ, y = x^2 + 1)) ≠ 
  (set_of (λ y : ℝ, ∃ x : ℝ, y = x^2 + 1)) ∧
  (set_of (λ x : ℝ, ∃ y : ℝ, y = x^2 + 1)) ≠ 
  (set_of (λ p : ℝ × ℝ, p.2 = p.1^2 + 1)) :=
by
  sorry

end sets_not_equal_l624_624088


namespace median_of_set_is_a_l624_624305

theorem median_of_set_is_a (a : ℤ) (c : ℝ) (h1 : a ≠ 0) (h2 : c > 0) (h3 : a * c^2 = Real.log c / Real.log c):
  set.median ({0, 1, (a : ℝ), c, 1 / c} : set ℝ) = (a : ℝ) :=
by
  sorry

end median_of_set_is_a_l624_624305


namespace min_poly_degree_rational_roots_l624_624426

theorem min_poly_degree_rational_roots :
  ∃ (P : Polynomial ℚ) (hP : P ≠ 0), (P.eval (3 - 2*Real.sqrt 2) = 0) ∧ 
                                    (P.eval (5 + Real.sqrt 3) = 0) ∧
                                    (P.eval (10 - 3*Real.sqrt 2) = 0) ∧
                                    (P.eval (-Real.sqrt 3) = 0) ∧
                                    P.degree = 8 :=
begin
  sorry
end

end min_poly_degree_rational_roots_l624_624426


namespace add_fractions_result_l624_624589

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624589


namespace fraction_addition_l624_624652

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624652


namespace fraction_addition_l624_624651

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624651


namespace HP_l624_624805

variables (A B C P D E F H H' A' P' : Point)
variables (circumcircle : Circumcircle A B C)
variables (H_orthocenter : H = orthocenter A B C)
variables (H'_on_circumcircle : H' ∈ circumcircle)
variables (H'_defined : H' = second_intersection (extension A H) circumcircle)
variables (A'_defined : A' = intersection (extension A H) (line_through B C))
variables (P_on_arc : P ∈ arc_not_containing A (circumcircle))
variables (PD_perpendicular : is_perpendicular P D (line_through B C))
variables (PE_perpendicular : is_perpendicular P E (line_through A C))
variables (PF_perpendicular : is_perpendicular P F (line_through A B))
variables (D_E_F_collinear : collinear D E F)
variables (PD_to_PD' : P' on_extension P D PD = distance PD = distance P' D)

theorem HP'_parallel_EF : is_parallel (line_through H P') (line_through E F) :=
sorry

end HP_l624_624805


namespace fraction_addition_l624_624617

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624617


namespace fraction_people_with_dog_l624_624965

theorem fraction_people_with_dog 
  (max_capacity : ℕ)
  (weight_per_person : ℕ)
  (dog_weight_fraction : ℚ)
  (total_weight_with_dog : ℕ)
  (dog_weight := (dog_weight_fraction * weight_per_person : ℚ)) :
  max_capacity = 6 →
  weight_per_person = 140 →
  dog_weight_fraction = (1/4:ℚ) →
  total_weight_with_dog = 595 →
  (total_weight_with_dog - dog_weight).natAbs / weight_per_person = 4 →
  (4 / max_capacity : ℚ) = 2 / 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end fraction_people_with_dog_l624_624965


namespace complex_exponential_form_theta_eq_pi_div_3_l624_624487

theorem complex_exponential_form_theta_eq_pi_div_3:
  ∃ θ : ℝ, 1 + complex.I * √3 = 2 * complex.exp (complex.I * θ) ∧ θ = (π / 3) :=
sorry

end complex_exponential_form_theta_eq_pi_div_3_l624_624487


namespace sqrt_meaningful_condition_l624_624471

theorem sqrt_meaningful_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (1 - x)) → x ≤ 1 :=
by
  assume h,
  sorry

end sqrt_meaningful_condition_l624_624471


namespace fans_per_set_l624_624395

theorem fans_per_set (total_fans : ℕ) (sets_of_bleachers : ℕ) (fans_per_set : ℕ)
  (h1 : total_fans = 2436) (h2 : sets_of_bleachers = 3) : fans_per_set = 812 :=
by
  sorry

end fans_per_set_l624_624395


namespace percentage_of_valid_votes_l624_624893

theorem percentage_of_valid_votes 
  (total_votes : ℕ) 
  (invalid_percentage : ℕ) 
  (candidate_valid_votes : ℕ)
  (percentage_invalid : invalid_percentage = 15)
  (total_votes_eq : total_votes = 560000)
  (candidate_votes_eq : candidate_valid_votes = 380800) 
  : (candidate_valid_votes : ℝ) / (total_votes * (0.85 : ℝ)) * 100 = 80 := 
by 
  sorry

end percentage_of_valid_votes_l624_624893


namespace curve_has_axis_of_symmetry_l624_624069

theorem curve_has_axis_of_symmetry (x y : ℝ) :
  (x^2 - x * y + y^2 + x - y - 1 = 0) ↔ (x+y = 0) :=
sorry

end curve_has_axis_of_symmetry_l624_624069


namespace final_sugar_percentage_is_17_l624_624964

def sugar_percentage_final_solution (original_solution_weight : ℕ) (original_sugar_percent : ℕ) (replace_fraction : ℕ) (second_solution_weight : ℕ) (second_sugar_percent : ℕ) : ℕ :=
  let original_sugar := original_solution_weight * original_sugar_percent / 100
  let replaced_sugar := original_sugar * replace_fraction / original_solution_weight
  let remaining_sugar := original_sugar - replaced_sugar 
  let added_sugar := second_solution_weight * second_sugar_percent / 100
  let total_sugar := remaining_sugar + added_sugar
  let final_solution_weight := original_solution_weight
  total_sugar * 100 / final_solution_weight

theorem final_sugar_percentage_is_17 :
  sugar_percentage_final_solution 100 10 25 25 38 = 17 :=
by 
  simp [sugar_percentage_final_solution]
  -- Proof goes here
  sorry

end final_sugar_percentage_is_17_l624_624964


namespace number_of_possible_flags_l624_624759

-- Define the number of colors available
def num_colors : ℕ := 3

-- Define the number of stripes on the flag
def num_stripes : ℕ := 3

-- Define the total number of possible flags
def total_flags : ℕ := num_colors ^ num_stripes

-- The statement we need to prove
theorem number_of_possible_flags : total_flags = 27 := by
  sorry

end number_of_possible_flags_l624_624759


namespace greatest_number_of_rented_trucks_l624_624109

theorem greatest_number_of_rented_trucks
  (total_trucks : ℕ)
  (rented_percentage_returned : ℝ)
  (trucks_on_lot_second_saturday : ℕ)
  (H1 : total_trucks = 45)
  (H2 : rented_percentage_returned = 0.40)
  (H3 : trucks_on_lot_second_saturday ≥ 25)
  : ∃ R : ℕ, R ≤ 33 ∧ ∀ r : ℕ, r < R → 0.60 * r ≤ total_trucks - trucks_on_lot_second_saturday := 
sorry

end greatest_number_of_rented_trucks_l624_624109


namespace find_a_l624_624133

theorem find_a (n k : ℕ) (h1 : 1 < k) (h2 : k < n)
  (h3 : ((n * (n + 1)) / 2 - k) / (n - 1) = 10) : n + k = 29 :=
by
  -- Proof omitted
  sorry

end find_a_l624_624133


namespace add_fractions_l624_624656

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624656


namespace add_fractions_l624_624569

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624569


namespace monotonic_increasing_intervals_l624_624081

def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_increasing_intervals :
  (∀ x ∈ Ioo (-∞) (-1), deriv f x > 0) ∧ (∀ x ∈ Ioo 11 (∞), deriv f x > 0) :=
by
  sorry

end monotonic_increasing_intervals_l624_624081


namespace add_fractions_l624_624742

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624742


namespace smallest_value_1000a_100b_10c_d_l624_624937

open Complex

noncomputable def ζ : ℂ := Complex.exp(2 * π * I / 13)

def exist_pos_ints (a b c d : ℕ) :=
  a > b ∧ b > c ∧ c > d ∧ 0 < d

theorem smallest_value_1000a_100b_10c_d 
  (a b c d : ℕ) 
  (h_condition: exist_pos_ints a b c d)
  (h_abs: abs (ζ^a + ζ^b + ζ^c + ζ^d) = Real.sqrt 3):
  1000 * a + 100 * b + 10 * c + d = 7521 := 
sorry

end smallest_value_1000a_100b_10c_d_l624_624937


namespace problem1_problem2_l624_624748

/-- 
  Problem 1: Prove that 32^(3/5) + 0.5^-2 = 12.
-/
theorem problem1 : 32^(3/5 : ℝ) + 0.5^(-2 : ℝ) = 12 := 
  sorry

/-- 
  Problem 2: Prove that 2^(log2 3) * log2 (1 / 8) + lg 4 + 2 * lg 5 = -7.
  Note: log2 denotes logarithm base 2, and lg denotes logarithm base 10.
-/
theorem problem2 : 2^(Real.log 3 / Real.log 2) * (Real.log (1 / 8) / Real.log 2) 
                   + Real.log 4 / Real.log 10 
                   + 2 * (Real.log 5 / Real.log 10) = -7 := 
  sorry

end problem1_problem2_l624_624748


namespace data_has_single_median_l624_624494

-- Define what it means to have a single median
def has_single_median (data : List ℝ) : Prop :=
  ∃ (m : ℝ), (sorted data ∧ (length data odd → m = data[length data / 2])
                        ∧ (length data even → m = (data[length data / 2 - 1] + data[length data / 2]) / 2))

-- Define the problem: prove that a set of data has only one median
theorem data_has_single_median (data : List ℝ) : has_single_median data :=
sorry

end data_has_single_median_l624_624494


namespace fraction_addition_l624_624648

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624648


namespace total_age_l624_624427

noncomputable def age_susan := 15
noncomputable def age_arthur := age_susan + 2
noncomputable def age_bob := 11
noncomputable def age_tom := age_bob - 3
noncomputable def age_emily := age_susan / 2
noncomputable def age_david := (age_arthur + age_tom + age_emily) / 3
noncomputable def age_youngest := age_emily - 2.5

theorem total_age (hs: age_susan = 15) (hb: age_bob = 11)
    (harthur: age_arthur = age_susan + 2)
    (htom: age_tom = age_bob - 3)
    (hemily: age_emily = age_susan / 2)
    (hdavid: age_david = (age_arthur + age_tom + age_emily) / 3)
    (d1: age_susan - age_tom = 2 * (age_emily - age_david))
    (d2: age_emily = age_youngest + 2.5) :
    age_susan + age_arthur + age_tom + age_bob + age_emily + age_david + age_youngest = 74.5 :=
by
  sorry

end total_age_l624_624427


namespace largest_possible_k_satisfies_triangle_condition_l624_624211

theorem largest_possible_k_satisfies_triangle_condition :
  ∃ k : ℕ, 
    k = 2009 ∧ 
    ∀ (b r w : Fin 2009 → ℝ), 
    (∀ i : Fin 2009, i ≤ i.succ → b i ≤ b i.succ ∧ r i ≤ r i.succ ∧ w i ≤ w i.succ) → 
    (∃ (j : Fin 2009), 
      b j + r j > w j ∧ b j + w j > r j ∧ r j + w j > b j) :=
sorry

end largest_possible_k_satisfies_triangle_condition_l624_624211


namespace evaluate_expression_l624_624983

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)
variable (h4 : ∀ x, g (g_inv x) = x)
variable (h5 : ∀ x, g_inv (g x) = x)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 :=
by
  -- The proof is omitted
  sorry

end evaluate_expression_l624_624983


namespace sum_of_fractions_l624_624705

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624705


namespace parabola_properties_l624_624854

-- Step 1: Define the conditions given in the problem.
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0
def point_on_parabola (p xP yP : ℝ) : Prop := (parabola p xP yP) ∧ yP = 4
def focus_distance (xP yP p : ℝ) : Prop := abs (xP + p / 2) = 4

-- Step 2: Define the proof goal.
theorem parabola_properties (p xP yP xA yA xB yB : ℝ)
  (hp_parabola : parabola p xP yP)
  (hp_point : point_on_parabola p xP yP)
  (hp_focus : focus_distance xP yP p)
  (hA : parabola p xA yA ∧ yA ≤ 0)
  (hB : parabola p xB yB ∧ yB ≤ 0)
  (angle_bisector_perpendicular : ∀ (k : ℝ), k ≠ 0 → yA - 4 = k * (xA - 2) ∧ yB - 4 = - (1/k) * (xB - 2)) :
  -- Step 3: Express the answers as follows:
  (∀ x y, parabola 4 x y → y^2 = 8 * x) ∧ (∀ x y, x + y = 0 → (par ::#@@'boxed{}24)) :=
by sorry

end parabola_properties_l624_624854


namespace sum_of_number_and_its_conjugate_l624_624753

-- Define the number and its radical conjugate 
def number : ℝ := 12 - real.sqrt 50
def radical_conjugate : ℝ := 12 + real.sqrt 50

-- State the proof problem
theorem sum_of_number_and_its_conjugate : number + radical_conjugate = 24 :=
by
  sorry

end sum_of_number_and_its_conjugate_l624_624753


namespace spaceship_not_moving_time_l624_624533

-- Definitions based on the conditions given
def total_journey_time : ℕ := 3 * 24  -- 3 days in hours

def first_travel_time : ℕ := 10
def first_break_time : ℕ := 3
def second_travel_time : ℕ := 10
def second_break_time : ℕ := 1

def subsequent_travel_period : ℕ := 11  -- 11 hours traveling, then 1 hour break

-- Function to compute total break time
def total_break_time (total_travel_time : ℕ) : ℕ :=
  let remaining_time := total_journey_time - (first_travel_time + first_break_time + second_travel_time + second_break_time)
  let subsequent_breaks := remaining_time / subsequent_travel_period
  first_break_time + second_break_time + subsequent_breaks

theorem spaceship_not_moving_time : total_break_time total_journey_time = 8 := by
  sorry

end spaceship_not_moving_time_l624_624533


namespace fraction_addition_l624_624684

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624684


namespace distinct_remainders_mod_p_of_three_consecutive_integers_l624_624830

theorem distinct_remainders_mod_p_of_three_consecutive_integers (p : ℕ) (hp : Nat.Prime p) (hp_ge_5 : p ≥ 5) :
  let num_distinct_remainders := Nat.descend_floor_div (2 * p + 1) 3 in
  num_distinct_remainders = (∑ x : Fin p, let prod := x * (x + 1) * (x + 2) in Finset.card (Finset.image (λ n, n % p) (Finset.range prod))) := 
sorry

end distinct_remainders_mod_p_of_three_consecutive_integers_l624_624830


namespace sum_of_squares_increased_l624_624818

theorem sum_of_squares_increased (x : Fin 100 → ℝ) 
  (h : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  ∑ i, (x i + 4) ^ 2 = ∑ i, x i ^ 2 + 800 := 
by
  sorry

end sum_of_squares_increased_l624_624818


namespace least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l624_624482

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_primes_greater_than_five : List ℕ :=
  [7, 11, 13]

theorem least_positive_integer_divisible_by_three_smallest_primes_greater_than_five : 
  ∃ n : ℕ, n > 0 ∧ (∀ p ∈ smallest_primes_greater_than_five, p ∣ n) ∧ n = 1001 := by
  sorry

end least_positive_integer_divisible_by_three_smallest_primes_greater_than_five_l624_624482


namespace ratio_of_areas_l624_624444

def side_length_S : ℝ := sorry
def longer_side_R : ℝ := 1.2 * side_length_S
def shorter_side_R : ℝ := 0.8 * side_length_S
def area_S : ℝ := side_length_S ^ 2
def area_R : ℝ := longer_side_R * shorter_side_R

theorem ratio_of_areas (side_length_S : ℝ) :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l624_624444


namespace evaluate_expression_l624_624196

theorem evaluate_expression :
  (Real.exp (Real.log 2) + Real.log10 (1 / 100) + (Real.sqrt 2014 - 2015) ^ Real.log10 1) +
  (- ((8 / 27) ^ (-2 / 3)) * ((-8) ^ (2 / 3)) + abs (-100) ^ Real.sqrt 0.25 + (3 - Real.pi) ^ 4.root 4) =
  1 + (Real.pi - 2) :=
by
  sorry

end evaluate_expression_l624_624196


namespace fraction_addition_l624_624602

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624602


namespace incorrect_statement_l624_624495

axiom parallel_lines_distance (l1 l2 : Line) (h : Parallel l1 l2) : ∃ d, ∀ p1 p2, 
  (p1 ∈ l1 ∧ p2 ∈ l2 ∧ ¬Collinear p1 p2 (l1.Inter l2)) → Distance p1 p2 = d

axiom rhombus_diagonals (r : Rhombus) : ∃ d1 d2, DiagonalLength r d1 d2 ∧ d1 ≠ d2

axiom quad_diagonals_bisect (q : Quadrilateral) (h : Bisect q.Diagonal1 q.Diagonal2) : 
  Parallelogram q

axiom quad_three_right_angles (q : Quadrilateral) (h: RightAngle q.Angle1 ∧ RightAngle q.Angle2 
  ∧ RightAngle q.Angle3) : Rectangle q

theorem incorrect_statement (r : Rhombus) : ¬ ∃ d, DiagonalEqual r d := 
sorry

end incorrect_statement_l624_624495


namespace parabola_incorrect_statement_B_l624_624493

theorem parabola_incorrect_statement_B 
  (y₁ y₂ : ℝ → ℝ) 
  (h₁ : ∀ x, y₁ x = 2 * x^2) 
  (h₂ : ∀ x, y₂ x = -2 * x^2) : 
  ¬ (∀ x < 0, y₁ x < y₁ (x + 1)) ∧ (∀ x < 0, y₂ x < y₂ (x + 1)) := 
by 
  sorry

end parabola_incorrect_statement_B_l624_624493


namespace sum_of_fractions_l624_624706

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624706


namespace nancy_pics_uploaded_l624_624960

theorem nancy_pics_uploaded (a b n : ℕ) (h₁ : a = 11) (h₂ : b = 8) (h₃ : n = 5) : a + b * n = 51 := 
by 
  sorry

end nancy_pics_uploaded_l624_624960


namespace common_area_of_triangles_and_circle_l624_624535

-- Given conditions
def square_side_length : ℝ := 4
def triangle_side_length := square_side_length
def circle_radius := square_side_length / 2

-- The mathematically equivalent proof problem in Lean 4
theorem common_area_of_triangles_and_circle :
  area_common_to_two_triangles_and_circle square_side_length triangle_side_length circle_radius = 4 * Real.pi :=
by sorry

end common_area_of_triangles_and_circle_l624_624535


namespace total_people_at_evening_l624_624952

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l624_624952


namespace eccentricity_of_ellipse_l624_624999

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 - b^2)
  in c / a

theorem eccentricity_of_ellipse : 
  eccentricity 4 (2 * Real.sqrt 2) = (Real.sqrt 2) / 2 :=
by
  sorry

end eccentricity_of_ellipse_l624_624999


namespace remove_unit_cubes_preserve_surface_area_l624_624114

theorem remove_unit_cubes_preserve_surface_area :
  ∃ (remaining_cubes : Finset (Fin 27)), remaining_cubes.card = 17 ∧
  (let original_surface_area := 6 * (3 ^ 2) in
   let resulting_surface_area := 6 * (3 ^ 2) in
   resulting_surface_area = original_surface_area) :=
sorry

end remove_unit_cubes_preserve_surface_area_l624_624114


namespace focus_of_parabola_l624_624435

theorem focus_of_parabola : 
  (∃ p : ℝ, y^2 = 4 * p * x ∧ p = 1 ∧ ∃ c : ℝ × ℝ, c = (1, 0)) :=
sorry

end focus_of_parabola_l624_624435


namespace remainder_of_2n_l624_624129

theorem remainder_of_2n (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := 
sorry

end remainder_of_2n_l624_624129


namespace students_not_visiting_any_l624_624092

-- Define the given conditions as Lean definitions
def total_students := 52
def visited_botanical := 12
def visited_animal := 26
def visited_technology := 23
def visited_botanical_animal := 5
def visited_botanical_technology := 2
def visited_animal_technology := 4
def visited_all_three := 1

-- Translate the problem statement and proof goal
theorem students_not_visiting_any :
  total_students - (visited_botanical + visited_animal + visited_technology 
  - visited_botanical_animal - visited_botanical_technology 
  - visited_animal_technology + visited_all_three) = 1 :=
by
  -- The proof is omitted
  sorry

end students_not_visiting_any_l624_624092


namespace exists_infinitely_many_fractions_l624_624932

theorem exists_infinitely_many_fractions 
  (x : ℝ) (h_irrational : irrational x) (h_pos : 0 < x) : 
  ∃ᶠ p q : ℚ, |x - (p / q)| ≤ 1 / q^2 := 
sorry

end exists_infinitely_many_fractions_l624_624932


namespace direction_vector_of_line_l624_624449

def matrix_P : matrix (fin 3) (fin 3) ℚ :=
  ![
    ![\frac{1}{4}, -\frac{1}{8}, \frac{1}{8}],
    ![-\frac{1}{8}, \frac{3}{4}, \frac{1}{8}],
    ![\frac{1}{8}, \frac{1}{8}, \frac{1}{2}]
  ]

def std_basis_i : vector ℚ 3 :=
  ![
    1,
    0,
    0
  ]

def direction_vector : vector ℚ 3 :=
  ![
    2,
    -1,
    1
  ]

theorem direction_vector_of_line (proj_matrix : matrix (fin 3) (fin 3) ℚ)
  (basis_vector : vector ℚ 3) :
  proj_matrix * basis_vector = 1/8 • direction_vector ↔ 
  proj_matrix = matrix_P ∧ basis_vector = std_basis_i := 
by
  sorry

end direction_vector_of_line_l624_624449


namespace fraction_addition_l624_624603

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624603


namespace meena_cookies_left_l624_624948

def dozen : ℕ := 12

def baked_cookies : ℕ := 5 * dozen
def mr_stone_buys : ℕ := 2 * dozen
def brock_buys : ℕ := 7
def katy_buys : ℕ := 2 * brock_buys
def total_sold : ℕ := mr_stone_buys + brock_buys + katy_buys
def cookies_left : ℕ := baked_cookies - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end meena_cookies_left_l624_624948


namespace prob_2_pow_x_in_1_2_eql_1_div_4_l624_624050

noncomputable def prob_2_pow_x_in_1_2 : ℝ :=
let A := { x : ℝ | 1 ≤ 2^x ∧ 2^x ≤ 2 }
let B := Icc (-2 : ℝ) 2 
(PMF.classicalOfFinSet B (by simp)).prob A

theorem prob_2_pow_x_in_1_2_eql_1_div_4 :
  prob_2_pow_x_in_1_2 = 1 / 4 := by
  sorry

end prob_2_pow_x_in_1_2_eql_1_div_4_l624_624050


namespace triangle_DOE_area_l624_624551

theorem triangle_DOE_area
  (area_ABC : ℝ)
  (DO : ℝ) (OB : ℝ)
  (EO : ℝ) (OA : ℝ)
  (h_area_ABC : area_ABC = 1)
  (h_DO_OB : DO / OB = 1 / 3)
  (h_EO_OA : EO / OA = 4 / 5)
  : (1 / 4) * (4 / 9) * area_ABC = 11 / 135 := 
by 
  sorry

end triangle_DOE_area_l624_624551


namespace fraction_addition_l624_624622

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624622


namespace parallelogram_diagonal_angle_l624_624077

theorem parallelogram_diagonal_angle
  (a b m n : ℝ)
  (h_m : m^2 = a^2 + b^2 + 2 * a * b * real.cos (real.pi / 4))
  (h_n : n^2 = a^2 + b^2 - 2 * a * b * real.cos (real.pi / 4)) :
  a^4 + b^4 = m^2 * n^2 ↔ real.cos (real.pi / 4) = 1 / real.sqrt 2 :=
sorry -- Proof required

end parallelogram_diagonal_angle_l624_624077


namespace fraction_addition_l624_624619

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624619


namespace fry_cutlets_within_time_l624_624766

theorem fry_cutlets_within_time:
  ∃ t < 20, 
    (∀ cutlet ∈ {1, 2, 3}, ∃ t₁ t₂,
       0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ t ∧ 
       (cutlet ∈ {1, 2} → t₂ - t₁ = 10) ∧
       (cutlet ∉ {1, 2} → t₂ - t₁ = 10)) :=
sorry

end fry_cutlets_within_time_l624_624766


namespace person_b_plough_time_l624_624498

theorem person_b_plough_time :
  ∀ (A B : ℝ),
    (A + B = 1 / 10) → (A = 1 / 15) → (B = 1 / 30) :=
by
  intro A B h_combined_work h_a_work
  have h_b_work : B = 1 / 30
  sorry

end person_b_plough_time_l624_624498


namespace find_abscissas_l624_624433

theorem find_abscissas (x_A x_B : ℝ) (y_A y_B : ℝ) : 
  ((y_A = x_A^2) ∧ (y_B = x_B^2) ∧ (0, 15) = (0,  (5 * y_B + 3 * y_A) / 8) ∧ (5 * x_B + 3 * x_A = 0)) → 
  ((x_A = -5 ∧ x_B = 3) ∨ (x_A = 5 ∧ x_B = -3)) :=
by
  sorry

end find_abscissas_l624_624433


namespace add_fractions_l624_624741

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624741


namespace factorization_correct_l624_624072

theorem factorization_correct (C D : ℤ) (h : 15 = C * D ∧ 48 = 8 * 6 ∧ -56 = -8 * D - 6 * C):
  C * D + C = 18 :=
  sorry

end factorization_correct_l624_624072


namespace monotonicity_of_f_range_of_a_l624_624284

noncomputable def f (a x : ℝ) := a * Real.log x + (1 / 2) * x^2
noncomputable def g (a x : ℝ) := (a + 1) * x

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → f a x = a * Real.log x + (1 / 2) * x^2) →
  ((0 ≤ a → ∀ x : ℝ, 0 < x → 0 < a + x^2 / x) ∧
    (a < 0 → ∀ x : ℝ, 0 < x → ((x < sqrt (-a) → 0 < a + x^2 / x) ∧ (sqrt (-a) < x → 0 < a + x^2 / x)))) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f a x ≤ g a x) →
  a ∈ Ici (2 / (2 - Real.log 2)) := sorry

end monotonicity_of_f_range_of_a_l624_624284


namespace polynomial_expansion_l624_624286

theorem polynomial_expansion :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a_1 * x^4 + a_2 * x^3 + a_3 * x^2 + 16 * x + 4) :=
by
  sorry

end polynomial_expansion_l624_624286


namespace handshake_arrangements_9_people_l624_624889

theorem handshake_arrangements_9_people : 
  let n : ℕ := 9 in
  ∑ i in {9, 6*3, 5*4, 3*3*3}, (cycle_arrangements n i) = 30016 :=
by
  sorry

/-- Cycle arrangements computation, provided as a placeholder. -/
noncomputable def cycle_arrangements (total_people: ℕ) (cycle_type: ℕ) : ℕ :=
  match cycle_type with
  | 9 => fact 8 / 2
  | 18 => (nat.choose 9 3) * (fact 5 / 2) / 2
  | 20 => (nat.choose 9 4) * (fact 3 / 2) * (fact 4 / 2)
  | 27 => fact 9 / (fact 3 * fact 3 * fact 3 * 3)
  | _ => 0

/-- Factorial function. -/
noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

end handshake_arrangements_9_people_l624_624889


namespace find_alpha_l624_624041

variables {V : Type} [InnerProductSpace ℝ V]
variables {A B C D E T : V}
variables (AB AC BE DC : AffineSubspace ℝ V)

/-- Given conditions for the point D and E on the sides AB and AC, respectively, with the given vector equation -/
def condition_vector_equation (DA DB EA EC : V) : Prop :=
  DA + DB + EA + EC = (0 : V)

/-- Define the centroid of triangle ABC -/
def is_centroid (T A B C : V) : Prop :=
  T = (A + B + C) / 3

/-- Define \( \overrightarrow{TB} + \overrightarrow{TC} = - \overrightarrow{TA} \) -/
theorem find_alpha (hD : ∃ DA, D = D ∧ DA = A - D ∧ B - D = DB)
  (hE : ∃ EA, E = E ∧ EA = A - E ∧ C - E = EC)
  (h1 : condition_vector_equation (A - D) (B - D) (A - E) (C - E))
  (h2 : is_centroid T A B C) :
  (B - T) + (C - T) = -1 • (A - T) :=
sorry

end find_alpha_l624_624041


namespace seymour_fertilizer_requirement_l624_624418

theorem seymour_fertilizer_requirement :
  let flats_petunias := 4
  let petunias_per_flat := 8
  let flats_roses := 3
  let roses_per_flat := 6
  let venus_flytraps := 2
  let fert_per_petunia := 8
  let fert_per_rose := 3
  let fert_per_venus_flytrap := 2

  let total_petunias := flats_petunias * petunias_per_flat
  let total_roses := flats_roses * roses_per_flat
  let fert_petunias := total_petunias * fert_per_petunia
  let fert_roses := total_roses * fert_per_rose
  let fert_venus_flytraps := venus_flytraps * fert_per_venus_flytrap

  let total_fertilizer := fert_petunias + fert_roses + fert_venus_flytraps
  total_fertilizer = 314 := sorry

end seymour_fertilizer_requirement_l624_624418


namespace jaya_rank_from_bottom_l624_624912

theorem jaya_rank_from_bottom
  (total_students : ℕ)
  (rank_from_top : ℕ)
  (h1 : total_students = 53)
  (h2 : rank_from_top = 5) :
  let rank_from_bottom := total_students - rank_from_top + 1 in
  rank_from_bottom = 49 := by
  simp [h1, h2]
  done

end jaya_rank_from_bottom_l624_624912


namespace non_sibling_probability_l624_624891

-- Conditions outlined in the problem
def num_people : ℕ := 6
def num_sibling_sets : ℕ := 3
def siblings_per_set : ℕ := 2

-- Function to calculate combinations
def combinations (n k : ℕ) : ℕ := n.choose k

-- Main theorem statement
theorem non_sibling_probability : 
  let total_ways := combinations num_people 2,
      sibling_ways := num_sibling_sets,
      non_sibling_ways := total_ways - sibling_ways in
  (rat.mk non_sibling_ways total_ways) = rat.mk 4 5 := sorry

end non_sibling_probability_l624_624891


namespace sum_of_squares_increase_by_800_l624_624811

theorem sum_of_squares_increase_by_800
  (x : Fin 100 → ℝ)
  (h : ∑ j, x j ^ 2 = ∑ j, (x j + 2) ^ 2) :
  (∑ j, (x j + 4) ^ 2) - (∑ j, x j ^ 2) = 800 := 
by
  sorry

end sum_of_squares_increase_by_800_l624_624811


namespace inequality_problem_l624_624378

-- Define the problem conditions and goal
theorem inequality_problem (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) : 
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
sorry

end inequality_problem_l624_624378


namespace smallest_number_of_slices_l624_624045

-- Definition of the number of slices in each type of cheese package
def slices_of_cheddar : ℕ := 12
def slices_of_swiss : ℕ := 28

-- Predicate stating that the smallest number of slices of each type Randy could have bought is 84
theorem smallest_number_of_slices : Nat.lcm slices_of_cheddar slices_of_swiss = 84 := by
  sorry

end smallest_number_of_slices_l624_624045


namespace infinite_n_exists_l624_624412

noncomputable def infinite_n_proof : Prop :=
  ∀ k : ℕ, ∃ n : ℕ, n = 4 * k + 1 ∧
    (∃ a b c : Fin n → ℕ, 
      (∀ i : Fin n, a i + b i + c i = a 0 + b 0 + c 0) ∧ 
      (a 0 + b 0 + c 0) % 6 = 0 ∧
      (Finset.univ.sum a) = (Finset.univ.sum b) ∧ 
      (Finset.univ.sum b) = (Finset.univ.sum c) ∧
      (Finset.univ.sum a) % 6 = 0)

theorem infinite_n_exists : infinite_n_proof := 
  sorry

end infinite_n_exists_l624_624412


namespace twelfth_term_l624_624090

-- Definitions based on the given conditions
def a_3_condition (a d : ℚ) : Prop := a + 2 * d = 10
def a_6_condition (a d : ℚ) : Prop := a + 5 * d = 20

-- The main theorem stating that the twelfth term is 40
theorem twelfth_term (a d : ℚ) (h1 : a_3_condition a d) (h2 : a_6_condition a d) :
  a + 11 * d = 40 :=
sorry

end twelfth_term_l624_624090


namespace sum_of_solutions_l624_624798

-- Definitions used in conditions
def is_floor (x : ℝ) (n : ℤ) : Prop := n ≤ x ∧ x < n + 1

def equation (x : ℝ) : Prop :=
  ∃ (n : ℤ), is_floor x n ∧ (x - n = 1 / (n + 0.5))

-- The main theorem statement
theorem sum_of_solutions :
  ∑ x in {1 + 2 / 3, 2 + 2 / 5, 3 + 2 / 7}, x = 7 + 37 / 105 :=
by sorry

end sum_of_solutions_l624_624798


namespace find_h_3_l624_624374

-- Define that h is a linear function and other specific conditions
variables {h : ℝ → ℝ}
variables {a b : ℝ}
noncomputable def h_inverse (y : ℝ) : ℝ := (y - b) / a

-- Conditions given in the problem
axiom linear_h : (∀ x, h(x) = a * x + b)
axiom condition_eq : (∀ x, h(x) = 3 * h_inverse(x) + 9)
axiom at_one_condition : h(1) = 5

-- Question to prove
theorem find_h_3 : h(3) = -6 * real.sqrt 3 + 3 :=
by
  sorry -- Proof not required

end find_h_3_l624_624374


namespace people_at_the_beach_l624_624955

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l624_624955


namespace range_arcsin_add_arctan_l624_624191

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arctan x

theorem range_arcsin_add_arctan :
  (λ x, Real.arcsin x + Real.arctan x) '' set.Icc (-1:ℝ) 1 = set.Icc (-3*Real.pi/4) (3*Real.pi/4) :=
by
  sorry

end range_arcsin_add_arctan_l624_624191


namespace probability_even_dots_after_addition_l624_624398

theorem probability_even_dots_after_addition :
  let initial_faces := [1, 2, 3, 4, 5, 6],
      faces_after_addition := initial_faces.map (λ n => n + 1),
      even_faces_after_addition := (faces_after_addition.filter (λ n => n % 2 = 0)).length,
      total_faces := faces_after_addition.length
  in (even_faces_after_addition : ℚ) / total_faces = 1 / 2 :=
by
  sorry

end probability_even_dots_after_addition_l624_624398


namespace add_fractions_l624_624576

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624576


namespace problem1_l624_624505

theorem problem1 : sqrt 9 - (1/2)^(-2) - 2^0 = -2 := by
  sorry

end problem1_l624_624505


namespace fraction_addition_l624_624681

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624681


namespace fascinating_phone_numbers_count_l624_624183

def is_fascinating (d : Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10) : Prop :=
  let (d1, d2, d3, d4, d5, d6, d7, d8) := d
  (d1, d2, d3, d4) = (d2, d3, d4, d5) ∨ (d1, d2, d3, d4) = (d5, d6, d7, d8)

def number_of_fascinating_phone_numbers : Nat :=
  10000

theorem fascinating_phone_numbers_count :
  (∃ count : Nat, count = ∑ (d1 d2 d3 d4 d5 d6 d7 d8 : Fin 10), if is_fascinating (d1, d2, d3, d4, d5, d6, d7, d8) then 1 else 0) →
  count = number_of_fascinating_phone_numbers :=
sorry

end fascinating_phone_numbers_count_l624_624183


namespace michael_total_score_l624_624393

theorem michael_total_score (junior_year_score : ℕ) (percentage_increase : ℕ) :
  junior_year_score = 260 → percentage_increase = 20 →
  (junior_year_score + ((percentage_increase * junior_year_score) / 100) + junior_year_score) = 572 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end michael_total_score_l624_624393


namespace wage_constraint_l624_624100

/-- Wage constraints for hiring carpenters and tilers given a budget -/
theorem wage_constraint (x y : ℕ) (h_carpenter_wage : 50 * x + 40 * y = 2000) : 5 * x + 4 * y = 200 := by
  sorry

end wage_constraint_l624_624100


namespace measureAngleBAC_correct_l624_624910

noncomputable def measureAngleBAC (A B C : Type) [EuclideanGeometry A] 
    (AX XY YB BC : Real) 
    (h1 : AX = XY) 
    (h2 : XY = YB) 
    (h3 : YB = BC / 2) 
    (AB : Real) 
    (h4 : AB = 2 * BC) 
    (AngleABC : Real) 
    (h5 : AngleABC = 90) : Real :=
22.5

theorem measureAngleBAC_correct (A B C : Type) [EuclideanGeometry A] 
    (AX XY YB BC : Real) 
    (h1 : AX = XY) 
    (h2 : XY = YB) 
    (h3 : YB = BC / 2) 
    (AB : Real) 
    (h4 : AB = 2 * BC) 
    (AngleABC : Real) 
    (h5 : AngleABC = 90) : 
  measureAngleBAC A B C AX XY YB BC h1 h2 h3 AB h4 AngleABC h5 = 22.5 := 
sorry

end measureAngleBAC_correct_l624_624910


namespace eccentricity_of_hyperbola_l624_624242

noncomputable def hyperbola := {a b : ℝ // a > 0 ∧ b > 0 ∧ (∀ x y, (x^2 / a^2 - y^2 / b^2 = 1))}
noncomputable def angle_between_asymptotes (a b : ℝ) := ∀ θ : ℝ, θ = 60

theorem eccentricity_of_hyperbola :
  ∀ (a b : ℝ) (h : a > 0) (h' : b > 0),
  angle_between_asymptotes a b 60 →
  (∃ e : ℝ, e = 2 * Real.sqrt 3 / 3 ∨ e = 2) :=
by
  sorry

end eccentricity_of_hyperbola_l624_624242


namespace calc_probability_l624_624922

-- Definitions of the given conditions
def P (S : Set α) (prob : α → Bool) : ℝ := ∑ x in S, if prob x then 1 else 0 
                                      -- This is an illustrative example. Usually, probabilities are defined over measurable spaces, but for simplicity, we assume a sum over a finite set.

variables {α : Type} [Fintype α] {A B : Set α} {prob : α → Bool}

def conditional_probability (A B : Set α) (prob : α → Bool) : ℝ :=
  P (A ∩ B) prob / P B prob

-- Theorem stating the proof problem
theorem calc_probability (hB : P B prob = 1/2)
                         (hA_given_B : conditional_probability A B prob = 1/3) :
  P (A ∩ B) prob = 1/6 :=
sorry

end calc_probability_l624_624922


namespace sum_outside_slices_l624_624509

-- Define the block and its conditions.
def block (n : ℕ) := fin n × fin n × fin n

-- Conditions: each column of 20 units sums to 1
def column_sums_to_one (b : block 20 → ℝ) : Prop :=
∀ i j : fin 20, (∑ k : fin 20, b (i, j, k)) = 1 ∧
                (∑ k : fin 20, b (i, k, j)) = 1 ∧
                (∑ k : fin 20, b (k, i, j)) = 1

-- There exist a special unit cube
def special_cube (b : block 20 → ℝ) : Prop :=
∃ x y z : fin 20, b (x, y, z) = 10

-- The slices containing the special unit cube
def slices (b : block 20 → ℝ) (x y z: fin 20) : Prop :=
(∑ i : fin 20, b (i, y, z)) = 20 ∧
(∑ j : fin 20, b (x, j, z)) = 20 ∧
(∑ k : fin 20, b (x, y, k)) = 20

-- Final statement to prove that the sum outside the slices is 333
theorem sum_outside_slices (b : block 20 → ℝ) 
  (h1 : column_sums_to_one b) 
  (h2 : special_cube b) 
  (h3: ∃ (x y z: fin 20), slices b x y z) : 
  ∑ (i j k : fin 20), b ⟨i,j,k⟩ - 67 = 333 := sorry

end sum_outside_slices_l624_624509


namespace count_ordered_pairs_no_distinct_real_solutions_l624_624797

theorem count_ordered_pairs_no_distinct_real_solutions :
  {n : Nat // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (4 * b^2 - 4 * c ≤ 0) ∧ (4 * c^2 - 4 * b ≤ 0) ∧ n = 1} :=
sorry

end count_ordered_pairs_no_distinct_real_solutions_l624_624797


namespace sum_of_fractions_l624_624709

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624709


namespace symmetric_center_of_octagon_l624_624160

theorem symmetric_center_of_octagon 
  (H_convex : ConvexOctagon)
  (H_equal_angles : ∀ (A B C D E F G H: Point), 
    Angle A B + Angle B C + Angle C D + Angle D E + Angle E F + Angle F G + Angle G H + Angle H A = 1080)
  (H_rational_lengths : ∀ (A B C D E F G H: Segment), 
    RationalLength A B ∧ RationalLength B C ∧ RationalLength C D ∧ RationalLength D E ∧ 
    RationalLength E F ∧ RationalLength F G ∧ RationalLength G H ∧ RationalLength H A) :
  ∃ (center: Point), CenterSymmetry center :=
by
  sorry

end symmetric_center_of_octagon_l624_624160


namespace max_heaps_660_stones_l624_624031

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624031


namespace cevian_concurrency_l624_624344

-- Define the type for a triangle
structure Triangle :=
(A B C : Type)

-- Define the type for a point on a side of a triangle
structure IncircleTouchingPoints (T : Triangle) :=
(M N P : Type)

-- A definition for proving lines intersection in a triangle
def incircle_concurrent (T : Triangle) (P : IncircleTouchingPoints T) : Prop :=
  -- The proposition that AM, BN, and CP are concurrent
  ∃ O : Type,  -- There exists a point O
    O = sorry -- Placeholder to indicate that in a formal proof, we would assert this point

-- Given
variable {T : Triangle}
variable {P : IncircleTouchingPoints T}

-- The statement to prove
theorem cevian_concurrency : incircle_concurrent T P :=
sorry -- Proof placeholder

end cevian_concurrency_l624_624344


namespace find_a_if_f_is_odd_l624_624307

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_if_f_is_odd :
  (∀ x : ℝ, f 1 x = -f 1 (-x)) ↔ (1 = 1) :=
by
  sorry

end find_a_if_f_is_odd_l624_624307


namespace add_fractions_result_l624_624594

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624594


namespace total_dolls_l624_624167

theorem total_dolls (big_boxes : ℕ) (dolls_per_big_box : ℕ) (small_boxes : ℕ) (dolls_per_small_box : ℕ)
  (h1 : dolls_per_big_box = 7) (h2 : big_boxes = 5) (h3 : dolls_per_small_box = 4) (h4 : small_boxes = 9) :
  big_boxes * dolls_per_big_box + small_boxes * dolls_per_small_box = 71 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end total_dolls_l624_624167


namespace add_fractions_l624_624729

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624729


namespace factor_expression_l624_624180

theorem factor_expression (x : ℝ) : 16 * x ^ 2 + 8 * x = 8 * x * (2 * x + 1) :=
by
  -- Problem: Completely factor the expression
  -- Given Condition
  -- Conclusion
  sorry

end factor_expression_l624_624180


namespace f_bound_l624_624278

noncomputable def f (x a : ℝ) := x^2 + a * log (1 + x)

theorem f_bound (a x : ℝ) (h1 : 0 < a) (h2 : a < 1/2) (hx1 : x = -1/2) (hx2 : x = 0) :
         f x a > (1 - 2 * log 2) / 4 :=
by
  sorry

end f_bound_l624_624278


namespace piece_visits_at_least_two_cells_twice_l624_624963

-- Define the board dimensions.
def board_size : ℕ := 100

-- Define the start and end positions.
def start_pos : (ℕ × ℕ) := (1, 1)
def mid_pos : (ℕ × ℕ) := (1, 100)
def end_pos : (ℕ × ℕ) := (1, 100)

-- Movement rules: alternating horizontal and vertical, starting with horizontal.
def is_horizontal (n : ℕ) : bool := n % 2 = 1

-- The main theorem statement
theorem piece_visits_at_least_two_cells_twice :
  ∃ (pos1 pos2 : ℕ × ℕ), pos1 ≠ pos2 ∧ 
  (∃ n1 n2, n1 ≠ n2 ∧ ((move_sequence n1) = pos1) ∧ ((move_sequence n2) = pos2)) :=
sorry

end piece_visits_at_least_two_cells_twice_l624_624963


namespace circle_properties_l624_624290

theorem circle_properties (x y : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 4 → x^2 + y^2 + 4*x - 2*y + 1 = 0) →
  (∃ A B : ℝ, (∃ x y : ℝ, x^2 + y^2 = 4 ∧ x^2 + y^2 + 4*x - 2*y + 1 = 0) ∧
  circles_have_two_common_tangents ∧
  circles_are_symmetric_about_line_AB ∧
  max_value_EF = 4 + real.sqrt 5) :=
begin
  sorry
end

end circle_properties_l624_624290


namespace total_texts_sent_l624_624351

theorem total_texts_sent (grocery_texts : ℕ) (response_texts_ratio : ℕ) (police_texts_percentage : ℚ) :
  grocery_texts = 5 →
  response_texts_ratio = 5 →
  police_texts_percentage = 0.10 →
  let response_texts := grocery_texts * response_texts_ratio
  let previous_texts := response_texts + grocery_texts
  let police_texts := previous_texts * police_texts_percentage
  response_texts + grocery_texts + police_texts = 33 :=
by
  sorry

end total_texts_sent_l624_624351


namespace fraction_addition_l624_624674

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624674


namespace complex_multiplication_l624_624931

noncomputable def i : ℂ := complex.I  -- Imaginary unit
noncomputable def z1 : ℂ := 1 + 2 * i  -- First complex number
noncomputable def z2 : ℂ := -3 * i     -- Second complex number

theorem complex_multiplication :
  z1 * z2 = 6 - 3 * i := 
by  -- The proof goes here
  sorry

end complex_multiplication_l624_624931


namespace total_number_of_people_l624_624513

-- Definitions corresponding to conditions
variables (A C : ℕ)
variables (cost_adult cost_child total_revenue : ℝ)
variables (ratio_child_adult : ℝ)

-- Assumptions given in the problem
axiom cost_adult_def : cost_adult = 7
axiom cost_child_def : cost_child = 3
axiom total_revenue_def : total_revenue = 6000
axiom ratio_def : C = 3 * A
axiom revenue_eq : total_revenue = cost_adult * A + cost_child * C

-- The main statement to prove
theorem total_number_of_people : A + C = 1500 :=
by
  sorry  -- Proof of the theorem

end total_number_of_people_l624_624513


namespace solve_sin_eqn_l624_624420

theorem solve_sin_eqn (x : ℝ) : 
  (sin(2 * x) - real.pi * sin(x)) * sqrt(11 * x^2 - x^4 - 10) = 0 ↔ 
  x = -real.sqrt 10 ∨ x = -real.pi ∨ x = -1 ∨ x = 1 ∨ x = real.pi ∨ x = real.sqrt 10 :=
by
  have domain_condition :
    11 * x^2 - x^4 - 10 ≥ 0
  from sorry,
  sorry

end solve_sin_eqn_l624_624420


namespace fraction_addition_l624_624678

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624678


namespace solve_for_x_l624_624128

theorem solve_for_x (x : ℝ) (h : 5 / (4 + 1 / x) = 1) : x = 1 :=
by
  sorry

end solve_for_x_l624_624128


namespace polynomial_degree_l624_624479

noncomputable def p (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - 4
noncomputable def q (x : ℝ) : ℝ := 4 * x^11 - 8 * x^9 + 3 * x^2 + 12
noncomputable def r (x : ℝ) : ℝ := 2 * x^2 + 4

theorem polynomial_degree :
  polynomial.degree ((λ x, p x * q x) - (λ x, r x ^ 6)) = 14 :=
by sorry

end polynomial_degree_l624_624479


namespace sum_f_eq_l624_624276

def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

theorem sum_f_eq : (∑ k in finset.range 2017 \ {0}, f ((k:ℝ) / 2017)) = 1008 :=
by sorry

end sum_f_eq_l624_624276


namespace slope_angle_range_l624_624968

theorem slope_angle_range (x α : ℝ) (h_curve : α = 3 * x^2 - 1) :
  ∃ α_set : set ℝ, α_set = {α | 0 ≤ α ∧ α < π/2 ∨ 3π/4 ≤ α ∧ α < π} :=
by
  sorry

end slope_angle_range_l624_624968


namespace sum_fractions_eq_l624_624690

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624690


namespace fraction_addition_l624_624635

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624635


namespace fraction_addition_l624_624613

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624613


namespace determined_sequences_l624_624880

-- Define arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) (S₁ S₂ : ℤ) :=
  S₁ = a 1 ∧ S₂ = a 1 + a 2

-- Define geometric sequence condition with sum S₁ and S₂
def geometric_sequence_S1_S2 (a : ℕ → ℤ) (S₁ S₂ : ℤ) :=
  S₁ = a 1 ∧ S₂ = a 1 + a 1 * a 2

-- Define geometric sequence condition with sum S₁ and S₃
def geometric_sequence_S1_S3 (a : ℕ → ℤ) (S₁ S₃ : ℤ) :=
  S₁ = a 1 ∧ S₃ = a 1 + a 1 * a 2 + a 1 * a 2 * a 3

-- Define the recurrence relation sequence condition
def recurrence_sequence (a : ℕ → ℤ) (c a b : ℤ) :=
  a 1 = c ∧ (∀ n, a (2*n+2) = a (2*n) + a) ∧ (∀ n, a (2*n+1) = a (2*n-1) + b)

-- The main theorem stating when a sequence is determined
theorem determined_sequences (a : ℕ → ℤ) (S₁ S₂ S₃ a c b : ℤ) :
  arithmetic_sequence a S₁ S₂ ∨
  geometric_sequence_S1_S2 a S₁ S₂ ↔
  arithmetic_sequence a S₁ S₂ ∨
  geometric_sequence_S1_S2 a S₁ S₂ :=
  sorry

end determined_sequences_l624_624880


namespace no_such_n_exists_l624_624361

theorem no_such_n_exists : 
  ∀ (n : ℕ), ∀ (D_n : set ℕ), (D_n = {d | d ∣ n}) →
  ¬(∃ (A G : set ℕ), 
    A ⊆ D_n ∧ G ⊆ D_n ∧ 
    disjoint A G ∧ 
    3 ≤ A.card ∧ 3 ≤ G.card ∧ 
    (∃ k m : ℕ, A = {1, 1+k, 1+2k, ..., 1+mk}) ∧
    (∃ q : ℕ, ∃ z : ℕ, G = {s, sq, sq^2, ..., sq^z})) := 
begin
  sorry
end

end no_such_n_exists_l624_624361


namespace value_of_y_l624_624318

theorem value_of_y :
  let x := (21 / 2) * (40 + 60)
  let y := (60 - 40) / 2 + 1
  x + y = 1061 → y = 11 :=
by
  let x := (21 : ℝ) / 2 * (40 + 60)
  let y := (60 - 40) / 2 + 1
  have h1 : x = 1050 := by sorry
  have h2 : y = 11 := by sorry
  intro h
  exact h2

end value_of_y_l624_624318


namespace midpoint_on_line_l624_624047

-- Defining Point structure
structure Point where
  x : ℝ
  y : ℝ

def line_eq_p (a : ℝ) : Point :=
  ⟨a, 5 * a + 3⟩

def Q : Point := ⟨3, -2⟩

def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

theorem midpoint_on_line (a : ℝ) : 
  let P := line_eq_p a
  let M := midpoint P Q
  M.y = 5 * M.x - 7 := 
by
  sorry

end midpoint_on_line_l624_624047


namespace PetyaColorsAll64Cells_l624_624399

-- Assuming a type for representing cell coordinates
structure Cell where
  row : ℕ
  col : ℕ

def isColored (c : Cell) : Prop := true  -- All cells are colored
def LShapedFigures : Set (Set Cell) := sorry  -- Define what constitutes an L-shaped figure

theorem PetyaColorsAll64Cells :
  (∀ tilesVector ∈ LShapedFigures, ¬∀ cell ∈ tilesVector, isColored cell) → (∀ c : Cell, c.row < 8 ∧ c.col < 8 ∧ isColored c) := sorry

end PetyaColorsAll64Cells_l624_624399


namespace circle_tangent_perpendicular_l624_624475

theorem circle_tangent_perpendicular 
  (O1 O2 O : Type) 
  [MetricSpace O1] [MetricSpace O2] [MetricSpace O]
  (C A B P : O)
  (tangent : O1 → O2 → Prop)
  (touches : O1 → O → Prop)
  (PAeqPB : dist P A = dist P B) 
  (tangent_at_C : tangent O1 O2 → touches O1 O → Prop)
  (common_tangent_meets : ∃ P, tangent_at_C (O1, O2) (C) (O1, O) )
  (O_eq_C : O = (some ∃ O, (touches O1 O)) )
  : dist P O = dist A B → (PO ⊥ A B) :=
sorry

end circle_tangent_perpendicular_l624_624475


namespace parabola_vertex_position_l624_624756

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem parabola_vertex_position (x y : ℝ) :
  (∃ a b : ℝ, f a = y ∧ g b = y ∧ a = 1 ∧ b = -1)
  → (1 > -1) ∧ (f 1 > g (-1)) :=
by
  sorry

end parabola_vertex_position_l624_624756


namespace infinite_geometric_sum_l624_624852

noncomputable def geometric_sequence (n : ℕ) : ℝ := 3 * (-1 / 2)^(n - 1)

theorem infinite_geometric_sum :
  ∑' n, geometric_sequence n = 2 :=
sorry

end infinite_geometric_sum_l624_624852


namespace add_fractions_result_l624_624592

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624592


namespace sum_fractions_eq_l624_624698

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624698


namespace add_fractions_l624_624731

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624731


namespace fraction_addition_l624_624616

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624616


namespace smallest_positive_value_l624_624117

theorem smallest_positive_value (x : ℝ) (hx : x > 0) (h : x / 7 + 2 / (7 * x) = 1) : 
  x = (7 - Real.sqrt 41) / 2 :=
sorry

end smallest_positive_value_l624_624117


namespace find_all_functions_satisfying_functional_equation_l624_624778

theorem find_all_functions_satisfying_functional_equation :
  ∀ (f : ℚ → ℚ), (∀ x y : ℚ, f (x + y) = f x + f y) →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by {
  intro f,
  assume h,
  -- Here, "sorry" is a placeholder.
  sorry,
}

end find_all_functions_satisfying_functional_equation_l624_624778


namespace calculation_l624_624561

theorem calculation : Real.floor (Real.abs (-5.7)) + Real.abs (Real.floor (-5.7)) = 11 := by
  sorry

end calculation_l624_624561


namespace right_triangle_inscribed_circle_bisector_l624_624330

theorem right_triangle_inscribed_circle_bisector
  (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = 3)
  (F1 : ℝ) (h3 : F1 = -2 + real.sqrt 10)
  (XY XZ : ℝ) (h4 : XY = 2 * real.sqrt 10 - 4)
  (h5 : XZ = real.sqrt 10 - 2)
  (YZ : ℝ) (h6 : YZ = 2 * real.sqrt 6)
  (X1 : ℝ) :
  XX1 = 2 * real.sqrt 6 / 3 := sorry

end right_triangle_inscribed_circle_bisector_l624_624330


namespace domain_of_function_l624_624263

theorem domain_of_function (f : ℝ → ℝ) (h₀ : Set.Ioo 0 1 ⊆ {x | f (3 * x + 2)}) :
  Set.Ioo (3 / 2) 3 ⊆ {x | f (2 * x - 1)} :=
by
  sorry

end domain_of_function_l624_624263


namespace add_fractions_l624_624666

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624666


namespace add_fractions_l624_624740

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624740


namespace total_points_other_five_l624_624336

theorem total_points_other_five
  (x : ℕ) -- total number of points scored by the team
  (d : ℕ) (e : ℕ) (f : ℕ) (y : ℕ) -- points scored by Daniel, Emma, Fiona, and others respectively
  (hd : d = x / 3) -- Daniel scored 1/3 of the team's points
  (he : e = 3 * x / 8) -- Emma scored 3/8 of the team's points
  (hf : f = 18) -- Fiona scored 18 points
  (h_other : ∀ i, 1 ≤ i ∧ i ≤ 5 → y ≤ 15 / 5) -- Other 5 members scored no more than 3 points each
  (h_total : d + e + f + y = x) -- Total points equation
  : y = 14 := sorry -- Final number of points scored by the other 5 members

end total_points_other_five_l624_624336


namespace sum_even_integers_neg15_to_5_l624_624118

theorem sum_even_integers_neg15_to_5 :
  (∑ k in (-15 : ℤ) to (5 : ℤ)), if k % 2 = 0 then k else 0 = -50 :=
by sorry

end sum_even_integers_neg15_to_5_l624_624118


namespace sum_fractions_eq_l624_624688

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624688


namespace sum_of_squares_increased_l624_624820

theorem sum_of_squares_increased (x : Fin 100 → ℝ) 
  (h : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  ∑ i, (x i + 4) ^ 2 = ∑ i, x i ^ 2 + 800 := 
by
  sorry

end sum_of_squares_increased_l624_624820


namespace trig_product_identity_l624_624181

theorem trig_product_identity :
  let sin_30 := 0.5
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let cos_60 := 0.5 in
  (1 - 1 / sin_30) * (1 + 1 / cos_60) * (1 - 1 / cos_30) * (1 + 1 / sin_60) = 1 :=
by
  sorry

end trig_product_identity_l624_624181


namespace length_of_PM_equals_six_l624_624907

-- Define the problem in Lean
def problem_statement : Prop :=
  ∀ (A B C D M P : Point),
  (AB = 14) ∧ (BC = 16) ∧ (AC = 26) ∧
  (is_midpoint M B C) ∧ (is_angle_bisector AD BAC) ∧ 
  (is_perpendicular_from P B AD)
  → PM = 6

theorem length_of_PM_equals_six : problem_statement := sorry

end length_of_PM_equals_six_l624_624907


namespace fraction_addition_l624_624628

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624628


namespace part1_part2_l624_624319

axiom triangle (A B C : ℝ)
axiom a b c : ℝ
axiom cos : ℝ → ℝ

noncomputable def condition1 := b * cos C = (2 * a - c) * cos B
noncomputable def condition2 := b = sqrt 7 ∧ a + c = 4

theorem part1 : condition1 → B = π / 3 := sorry

theorem part2 : condition2 → condition1 → (a = 1 ∧ c = 3) ∨ (a = 3 ∧ c = 1) := sorry

end part1_part2_l624_624319


namespace permutation_sum_average_value_p_plus_q_l624_624224

-- Define the relevant sum and average value computation
theorem permutation_sum_average_value :
  let perms := (Equiv.perm (Fin 12)) in
  let sum_value (a : Fin 12 → ℕ) : ℕ :=
    |a 0 - a 1| + |a 2 - a 3| + |a 4 - a 5| + |a 6 - a 7| + |a 8 - a 9| + |a 10 - a 11| in
  (∑ p in perms, sum_value p) / perms.card = 572 / 11 :=
by sorry

theorem p_plus_q : 572 + 11 = 583 :=
by sorry

end permutation_sum_average_value_p_plus_q_l624_624224


namespace projection_of_CE_in_CF_l624_624834

noncomputable def side_length := 4

structure Point :=
(x : ℝ) 
(y : ℝ)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := side_length, y := 0 }
def D : Point := { x := 0, y := side_length }
def C : Point := { x := side_length, y := side_length }
def E : Point := { x := side_length / 2, y := 0 }
def F : Point := { x := 0, y := side_length / 4 }

def vector (P Q : Point) : Point :=
{ x := Q.x - P.x, y := Q.y - P.y }

def dot_product (v1 v2 : Point) : ℝ :=
v1.x * v2.x + v1.y * v2.y

def magnitude (v : Point) : ℝ :=
real.sqrt (v.x * v.x + v.y * v.y)

def projection (v w : Point) : ℝ :=
(dot_product v w) / (magnitude w)

theorem projection_of_CE_in_CF :
  projection (vector C E) (vector C F) = 4 :=
sorry

end projection_of_CE_in_CF_l624_624834


namespace missy_serving_time_l624_624961

def num_patients : ℕ := 30
def fraction_special_dietary : ℚ := 2 / 5
def increased_serving_time_fraction : ℚ := 1.5
def serving_time_standard : ℕ := 5

def num_special_dietary : ℕ := (fraction_special_dietary * num_patients).to_nat
def num_standard : ℕ := num_patients - num_special_dietary
def serving_time_special : ℚ := serving_time_standard * increased_serving_time_fraction
def total_serving_time_standard : ℕ := num_standard * serving_time_standard
def total_serving_time_special : ℚ := num_special_dietary * serving_time_special
def total_serving_time : ℚ := total_serving_time_standard + total_serving_time_special

theorem missy_serving_time : total_serving_time = 180 := by
  sorry

end missy_serving_time_l624_624961


namespace sum_of_fractions_l624_624703

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624703


namespace coconut_grove_l624_624886

theorem coconut_grove (x Y : ℕ) (h1 : 3 * x ≠ 0) (h2 : (x+3) * 60 + x * Y + (x-3) * 180 = 3 * x * 100) (hx : x = 6) : Y = 120 :=
by 
  sorry

end coconut_grove_l624_624886


namespace sophie_oranges_per_day_l624_624113

/-- Sophie and Hannah together eat a certain number of fruits in 30 days.
    Given Hannah eats 40 grapes every day, prove that Sophie eats 20 oranges every day. -/
theorem sophie_oranges_per_day (total_fruits : ℕ) (grapes_per_day : ℕ) (days : ℕ)
  (total_days_fruits : total_fruits = 1800) (hannah_grapes : grapes_per_day = 40) (days_count : days = 30) :
  (total_fruits - grapes_per_day * days) / days = 20 :=
by
  sorry

end sophie_oranges_per_day_l624_624113


namespace factor_correct_l624_624773

-- Define the polynomial p(x)
def p (x : ℤ) : ℤ := 6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2

-- Define the potential factors of p(x)
def f1 (x : ℤ) : ℤ := 3 * x^2 + 93 * x
def f2 (x : ℤ) : ℤ := 2 * x^2 + 178 * x + 5432

theorem factor_correct : ∀ x : ℤ, p x = f1 x * f2 x := by
  sorry

end factor_correct_l624_624773


namespace ratio_of_tetrahedron_to_cube_volume_l624_624152

theorem ratio_of_tetrahedron_to_cube_volume (x : ℝ) (hx : 0 < x) :
  let V_cube := x^3
  let a_tetrahedron := (x * Real.sqrt 3) / 2
  let V_tetrahedron := (a_tetrahedron^3 * Real.sqrt 2) / 12
  (V_tetrahedron / V_cube) = (Real.sqrt 6 / 32) :=
by
  sorry

end ratio_of_tetrahedron_to_cube_volume_l624_624152


namespace files_to_folders_l624_624034

theorem files_to_folders : 
  ∀ (initial_files deleted_files files_per_folder : ℕ),
    initial_files = 93 → 
    deleted_files = 21 → 
    files_per_folder = 8 → 
    (initial_files - deleted_files) / files_per_folder = 9 :=
by
  intros initial_files deleted_files files_per_folder h_initial_files h_deleted_files h_files_per_folder
  rw [h_initial_files, h_deleted_files, h_files_per_folder]
  sorry

end files_to_folders_l624_624034


namespace euler_line_through_circumcenter_l624_624434

theorem euler_line_through_circumcenter {A B C : Type*} [incircle : circle] (O : Point) (ABC : Triangle A B C) 
  (A' B' C' : Point) 
  (incircle_tangent_A' : tangent (incircle) (side BC) A') 
  (incircle_tangent_B' : tangent (incircle) (side CA) B') 
  (incircle_tangent_C' : tangent (incircle) (side AB) C') 
  (euler_line_parallel_BC : parallel (euler_line (Triangle A' B' C')) (side BC)) : 
  passes_through (euler_line (Triangle A' B' C')) (circumcenter (Triangle A B C)) :=
sorry

end euler_line_through_circumcenter_l624_624434


namespace no_exactly_three_eulerian_circuits_l624_624969

def is_circuit (G : SimpleGraph V) (C : List V) : Prop :=
  (∀ i, G.Adj (C.nth i) (C.nth (i + 1 % C.length))) ∧ C.Nodup

def is_eulerian_circuit (G : SimpleGraph V) (C : List V) : Prop :=
  is_circuit G C ∧ ∀ e, e ∈ G.edge_set → e ∈ (edges_of_list C).to_finset

theorem no_exactly_three_eulerian_circuits (G : SimpleGraph V) : ¬ (∃ C1 C2 C3 : List V, is_eulerian_circuit G C1 ∧
  is_eulerian_circuit G C2 ∧ is_eulerian_circuit G C3 ∧ C1 ≠ C2 ∧ C2 ≠ C3 ∧ C1 ≠ C3 ∧ 
  ∀ C, is_eulerian_circuit G C → (C = C1 ∨ C = C2 ∨ C = C3)) :=
sorry

end no_exactly_three_eulerian_circuits_l624_624969


namespace slope_angle_range_l624_624078

noncomputable def line_through_circle (α : ℝ) : Prop :=
  let k := Real.tan α in
  (k ≤ -Real.sqrt 3 ∨ k ≥ Real.sqrt 3)

theorem slope_angle_range : 
  ∃ α : ℝ, (\( 0 < α ∧ α < π \)) ∧ 
           (line_through_circle α) ∧ 
           (\( α ≠ π / 2 \)) ∧ 
           ( ∀ x, line_through_circle x → 
           (Real.atan x = α → \( \frac {\pi}{3} \leq α ∧ α \leq \frac {2\pi}{3} ))) :=
sorry

end slope_angle_range_l624_624078


namespace add_fractions_l624_624720

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624720


namespace find_n_term_x3_l624_624270

-- Provided conditions
variables (n : ℕ) (hn : 1 ≤ n)
variables (h_ratio : binomialCoeff n 1 * 5 = binomialCoeff n 2 * 2)

-- Derived results
theorem find_n : n = 6 := 
by 
  sorry

theorem term_x3 (n : ℕ) (h : n = 6) : ∃ k : ℕ, (2*x + 1/(√x))^n = (240*x^3) * k :=
by
  sorry

end find_n_term_x3_l624_624270


namespace exists_independent_nice_set_l624_624749

def isIndependent (S : Set ℕ) : Prop :=
  ∀ {x y : ℕ}, x ∈ S → y ∈ S → x ≠ y → Nat.coprime x y

def isNice (S : Set ℕ) : Prop :=
  ∀ T ⊆ S, T ≠ ∅ → ∃ m : ℕ, (∑ t in T, t) = m * T.card

theorem exists_independent_nice_set (n : ℕ) (hn : 0 < n) :
  ∃ S : Finset ℕ, S.card = n ∧ isIndependent S ∧ isNice S := sorry

end exists_independent_nice_set_l624_624749


namespace percentage_showed_up_l624_624879

variable (total_laborers : ℕ) (present_laborers : ℕ)

theorem percentage_showed_up (h1 : total_laborers = 156) (h2 : present_laborers = 70) :
  ((present_laborers : ℝ) / (total_laborers : ℝ) * 100).round = 44.9 :=
by
  sorry

end percentage_showed_up_l624_624879


namespace conic_section_locus_l624_624401

noncomputable theory

variables (a b c x y : ℝ)

def locus_equation (a b c x y : ℝ) : Prop :=
  b^2 * x^2 - 2 * a * b * x * y + a * (a - c) * y^2 - b^2 * c * x + 2 * a * b * c * y = 0

theorem conic_section_locus (a b c : ℝ) (hbc : b > 0 ∧ c > 0) (ha_ne_c : a ≠ c) :
  ∃ x y : ℝ, locus_equation a b c x y ∧
    ((a > 0 ∧ "hyperbola") ∨ (a < 0 ∧ "ellipse") ∨ (a = 0 → false)) :=
by
  sorry

end conic_section_locus_l624_624401


namespace sum_of_first_n_terms_b_l624_624245

def S (n : ℕ) : ℕ := 3 ^ n + 1

def a : ℕ → ℕ
| 0     := 4
| (n+1) := 2 * 3 ^ n

def b (n : ℕ) : ℚ :=
if n = 0 then 1/4 else n / (2 * 3 ^ (n - 1))

def T : ℕ → ℚ
| 0     := 0  -- No terms to sum at n = 0
| n + 1 := 1 / 4 + (1 / 2) * ((5 / 4) - (2 * (n + 1) + 3) / (4 * 3 ^ n))

theorem sum_of_first_n_terms_b (n : ℕ) : T n = (7 / 8) - (2 * n + 3) / (8 * 3 ^ (n - 1)) := 
sorry

end sum_of_first_n_terms_b_l624_624245


namespace sum_series_0_to_50_l624_624230

theorem sum_series_0_to_50 : (∑ i in Finset.range 51, (-1 : ℤ)^i * i) = 25 :=
by
  sorry

end sum_series_0_to_50_l624_624230


namespace standard_equation_of_C_slope_angle_of_l_sum_of_distances_PA_PB_l624_624897

-- Conditions
def parametric_curve (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, Real.sin α)

def polar_line (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2

def point_P : ℝ × ℝ := (0, 2)

-- Questions
theorem standard_equation_of_C :
  (∀ α : ℝ, (parametric_curve α).1 ^ 2 / 9 + (parametric_curve α).2 ^ 2 = 1) :=
sorry

theorem slope_angle_of_l (ρ θ : ℝ) (h : polar_line ρ θ) :
  θ = Real.pi / 4 :=
sorry

theorem sum_of_distances_PA_PB (A B : ℝ × ℝ) (hA : ∃ ρ θ, ρ * Real.cos θ = A.1 ∧ ρ * Real.sin θ = A.2 ∧ polar_line ρ θ ∧ (A.1 ^ 2 / 9 + A.2 ^ 2 = 1)) 
  (hB : ∃ ρ θ, ρ * Real.cos θ = B.1 ∧ ρ * Real.sin θ = B.2 ∧ polar_line ρ θ ∧ (B.1 ^ 2 / 9 + B.2 ^ 2 = 1)) :
  Real.sqrt ((A.1 - 0) ^ 2 + (A.2 - 2) ^ 2) + Real.sqrt ((B.1 - 0) ^ 2 + (B.2 - 2) ^ 2) = 
  18 * Real.sqrt 2 / 5 :=
sorry

end standard_equation_of_C_slope_angle_of_l_sum_of_distances_PA_PB_l624_624897


namespace proof_divisibility_l624_624383

noncomputable def a : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := r * a (n + 1) + s * a n

noncomputable def f (n : ℕ) : ℕ :=
(nat.prod (λ i => a (i + 1))) n

theorem proof_divisibility (r s : ℕ) (hr : 0 < r) (hs : 0 < s) (n k : ℕ) (hkn : 0 < k ∧ k < n) :
  ∃ m : ℕ, m * (f k * f (n - k)) = f n :=
sorry

end proof_divisibility_l624_624383


namespace sunflower_seeds_more_than_half_on_day_three_l624_624038

-- Define the initial state and parameters
def initial_sunflower_seeds : ℚ := 0.4
def initial_other_seeds : ℚ := 0.6
def daily_added_sunflower_seeds : ℚ := 0.2
def daily_added_other_seeds : ℚ := 0.3
def daily_sunflower_eaten_factor : ℚ := 0.7
def daily_other_eaten_factor : ℚ := 0.4

-- Define the recurrence relations for sunflower seeds and total seeds
def sunflower_seeds (n : ℕ) : ℚ :=
  match n with
  | 0     => initial_sunflower_seeds
  | (n+1) => daily_sunflower_eaten_factor * sunflower_seeds n + daily_added_sunflower_seeds

def total_seeds (n : ℕ) : ℚ := 1 + (n : ℚ) * 0.5

-- Define the main theorem stating that on Tuesday (Day 3), sunflower seeds are more than half
theorem sunflower_seeds_more_than_half_on_day_three : sunflower_seeds 2 / total_seeds 2 > 0.5 :=
by
  -- Formal proof will go here
  sorry

end sunflower_seeds_more_than_half_on_day_three_l624_624038


namespace distance_between_points_is_4_l624_624327

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_between_points_is_4 :
  distance (1, 0, 2) (2, real.sqrt 6, -1) = 4 :=
by
  sorry

end distance_between_points_is_4_l624_624327


namespace fraction_addition_l624_624595

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624595


namespace sqrt_meaningful_condition_l624_624470

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end sqrt_meaningful_condition_l624_624470


namespace transformed_point_final_coordinates_l624_624085

def rotate_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.3, p.2)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, p.3)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, p.3)

theorem transformed_point_final_coordinates
  : let initial_point := (1, 2, 3) in
    let after_first_rotation := rotate_x initial_point in
    let after_reflection_xz := reflect_xz after_first_rotation in
    let after_second_rotation := rotate_x after_reflection_xz in
    let final_point := reflect_yz after_second_rotation in
    final_point = (-1, -2, 3) :=
by
  sorry

end transformed_point_final_coordinates_l624_624085


namespace complement_of_intersection_l624_624939

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {y | ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ y = -x^2}

theorem complement_of_intersection :
  (Set.compl (A ∩ B) = {x | x < -2 ∨ x > 0 }) :=
by
  sorry

end complement_of_intersection_l624_624939


namespace factor_expression_l624_624178

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := 
by
  sorry

end factor_expression_l624_624178


namespace simplify_expr1_simplify_expr2_simplify_expr3_l624_624057

-- Definition for the first expression proof problem
theorem simplify_expr1 (a b : ℝ) (h : (a - b)^2 + ab ≠ 0) :
  (a^3 + b^3) / ((a - b)^2 + ab) = a + b := 
sorry

-- Definition for the second expression proof problem
theorem simplify_expr2 (x a : ℝ) (h : x^2 - 4a^2 ≠ 0) :
  (x^2 - 4ax + 4a^2) / (x^2 - 4a^2) = (x - 2a) / (x + 2a) := 
sorry

-- Definition for the third expression proof problem
theorem simplify_expr3 (x y : ℝ) (h : x ≠ 0) (h' : y ≠ 2) :
  (xy - 2x - 3y + 6) / (xy - 2x) = (x - 3) / x :=
sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l624_624057


namespace factorize_difference_of_squares_l624_624777

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
by 
  sorry

end factorize_difference_of_squares_l624_624777


namespace add_fractions_l624_624655

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624655


namespace total_dolls_l624_624165

theorem total_dolls (big_boxes : ℕ) (dolls_per_big_box : ℕ) (small_boxes : ℕ) (dolls_per_small_box : ℕ)
  (h1 : dolls_per_big_box = 7) (h2 : big_boxes = 5) (h3 : dolls_per_small_box = 4) (h4 : small_boxes = 9) :
  big_boxes * dolls_per_big_box + small_boxes * dolls_per_small_box = 71 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end total_dolls_l624_624165


namespace tan_arccot_l624_624751

theorem tan_arccot (x y : ℝ) (h1 : x = 5) (h2 : y = 12) :
  Real.tan (Real.arccot (x / y)) = y / x :=
by
  rw [h1, h2]
  -- proof goes here
  sorry

end tan_arccot_l624_624751


namespace num_valid_subsets_l624_624867

def original_set : Finset ℕ := (Finset.range 15).map (Nat.add 1)
def removed_subset (s : Finset ℕ) : Prop := s.card = 3 ∧ (∑ x in s, x) = 36
def target_sum : ℕ := 84
def remaining_set (s : Finset ℕ) : Finset ℕ := original_set \ s
def remaining_sum (s : Finset ℕ) : ℕ := ∑ x in (remaining_set s), x

theorem num_valid_subsets :
  (Finset.filter (λ s : Finset ℕ, removed_subset s ∧ remaining_sum s = target_sum) (Finset.powersetLen 3 original_set)).card = 3 :=
sorry

end num_valid_subsets_l624_624867


namespace count_cubes_2_9_to_2_17_l624_624863

noncomputable def lower_bound := 2^9 + 1
noncomputable def upper_bound := 2^17 + 1

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def count_perfect_cubes_between (a b : ℕ) : ℕ :=
  (finset.range (b + 1)).filter (λ n, a ≤ n ∧ is_perfect_cube n).card

theorem count_cubes_2_9_to_2_17 : count_perfect_cubes_between lower_bound upper_bound = 42 := by
  sorry

end count_cubes_2_9_to_2_17_l624_624863


namespace arrangement_count_general_arrangement_count_at_ends_arrangement_count_females_together_l624_624462

variable (men women : ℕ)
variable (first_last_must_be_male female_athletes_not_adjacent : Prop)
variable (female_athletes_together_ends female_athletes_together : Prop)

-- Prove the number of valid arrangements given the constraints for the first question
theorem arrangement_count_general
  (h : men = 7) (h' : women = 3)
  (condition1 : first_last_must_be_male)
  (condition2 : female_athletes_not_adjacent) :
  (factorial 7) * (6 * 5 * 4) = 604800 := sorry

-- Prove the number of valid arrangements when female athletes are grouped at ends
theorem arrangement_count_at_ends
  (h : men = 7) (h' : women = 3)
  (condition3 : female_athletes_together_ends) :
  2 * (factorial 3) * (factorial 7) = 60480 := sorry

-- Prove the number of valid arrangements when female athletes are together
theorem arrangement_count_females_together
  (h : men = 7) (h' : women = 3)
  (condition4 : female_athletes_together) :
  (factorial 8) * (factorial 3) = 241920 := sorry

end arrangement_count_general_arrangement_count_at_ends_arrangement_count_females_together_l624_624462


namespace fraction_addition_l624_624615

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624615


namespace billy_watches_videos_l624_624556

-- Conditions definitions
def num_suggestions_per_list : Nat := 15
def num_iterations : Nat := 5
def pick_index_on_final_list : Nat := 5

-- Main theorem statement
theorem billy_watches_videos : 
  num_suggestions_per_list * num_iterations + (pick_index_on_final_list - 1) = 79 :=
by
  sorry

end billy_watches_videos_l624_624556


namespace min_period_of_symmetric_sine_l624_624938

noncomputable def minimum_positive_period (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem min_period_of_symmetric_sine (ω : ℝ) (P_is_symm_center : ∃ P, P = ⟨0, 0⟩)
  (min_dist_to_symm_axis : ∃ P, (0 : ℝ) ≤ P ∧ P ≤ Real.pi / 4) :
  minimum_positive_period ω = Real.pi :=
sorry

end min_period_of_symmetric_sine_l624_624938


namespace concert_ticket_sales_l624_624096

theorem concert_ticket_sales (A C : ℕ) (total : ℕ) :
  (C = 3 * A) →
  (7 * A + 3 * C = 6000) →
  (total = A + C) →
  total = 1500 :=
by
  intros
  -- The proof is not required
  sorry

end concert_ticket_sales_l624_624096


namespace segment_lengths_l624_624137

theorem segment_lengths (AB BC CD DE EF : ℕ) 
  (h1 : AB > BC)
  (h2 : BC > CD)
  (h3 : CD > DE)
  (h4 : DE > EF)
  (h5 : AB = 2 * EF)
  (h6 : AB + BC + CD + DE + EF = 53) :
  (AB, BC, CD, DE, EF) = (14, 12, 11, 9, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 11, 8, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 10, 9, 7) :=
sorry

end segment_lengths_l624_624137


namespace abigail_total_savings_l624_624158

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_total_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end abigail_total_savings_l624_624158


namespace find_c_d_c_plus_d_l624_624927

noncomputable def real := ℝ

def U (x y z : real) : Prop :=
  log10 (2 * x + 3 * y) = z ∧
  log10 (x^3 + y^3) = 2 * z ∧
  x = y * tan z

theorem find_c_d :
  ∀ (x y z : real), U x y z → x^3 + y^3 = (1/8) * 10^(4 * z) + 12 * 10^(2 * z) :=
by sorry

theorem c_plus_d :
  (1/8 : real) + 12 = 97/8 :=
by norm_num

end find_c_d_c_plus_d_l624_624927


namespace tanya_cannot_determine_exact_masses_in_4_weighings_tanya_can_determine_exact_masses_in_4_weighings_special_balance_l624_624500

-- Definitions for Problem (a)
def weights : List ℕ := [1000, 1002, 1004, 1005] -- The weights Tanya has
def weighings : ℕ := 4 -- The number of weighings allowed

-- The main theorem stating Tanya cannot determine the exact mass of each weight in 4 weighings
theorem tanya_cannot_determine_exact_masses_in_4_weighings 
(weights : List ℕ) (weighings : ℕ) (distinct_weights : weights.Nodup) (length_weights : weights.length = 4) 
(hypothesis : ∀ w : ℕ, w ∈ weights → w = 1000 ∨ w = 1002 ∨ w = 1004 ∨ w = 1005) : 
  (weighings = 4) → 
  (∃ conf1 conf2 : List ℕ, conf1 ≠ conf2 ∧ conf1.perm weights ∧ conf2.perm weights) :=
by sorry

-- Definitions for Problem (b)
def special_balance_scale (leftPan rightPan : ℕ) : Ordering := 
  if leftPan + 1 < rightPan then Ordering.lt
  else if rightPan < leftPan + 1 then Ordering.gt
  else Ordering.eq

-- The main theorem stating Tanya can determine the exact mass of each weight in 4 weighings with special balance scale
theorem tanya_can_determine_exact_masses_in_4_weighings_special_balance 
(weights : List ℕ) (weighings : ℕ) (distinct_weights : weights.Nodup) (length_weights : weights.length = 4) 
(hypothesis : ∀ w : ℕ, w ∈ weights → w = 1000 ∨ w = 1002 ∨ w = 1004 ∨ w = 1005) 
(balance : ∀ left right, special_balance_scale left right ≠ Ordering.eq ∨ left + 1 = right) : 
  (weighings = 4) → 
  (∀ conf1 conf2 : List ℕ, conf1.perm weights → conf2.perm weights → conf1 = conf2) :=
by sorry

end tanya_cannot_determine_exact_masses_in_4_weighings_tanya_can_determine_exact_masses_in_4_weighings_special_balance_l624_624500


namespace sum_f_1_to_240_l624_624225

def f (n : ℕ) : ℕ :=
  if ∃ k : ℕ, n = k^2 then 0 else ⌊1 / (n.to_real.sqrt - ⌊n.to_real.sqrt⌋)⌋

theorem sum_f_1_to_240 : ∑ k in Finset.range 241, f k = 768 := by
  sorry

end sum_f_1_to_240_l624_624225


namespace total_people_at_beach_l624_624959

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end total_people_at_beach_l624_624959


namespace highest_certificate_probability_probability_exactly_two_l624_624066

/-- Probabilities of passing the theoretical exam for A, B, and C --/
def P_theoretical : ℕ → ℝ
| 1 := 4/5
| 2 := 3/4
| 3 := 2/3
| _ := 0

/-- Probabilities of passing the practical operation exam for A, B, and C --/
def P_practical : ℕ → ℝ
| 1 := 1/2
| 2 := 2/3
| 3 := 5/6
| _ := 0

/-- Probabilities of obtaining the "certificate of passing" for A, B, and C --/
def P_certificate (n : ℕ) : ℝ :=
  P_theoretical n * P_practical n

/-- Probabilities of passing both exams for A, B, and C --/
theorem highest_certificate_probability : 
  P_certificate 3 > P_certificate 2 ∧ P_certificate 2 > P_certificate 1 :=
by
  sorry

/-- Probability that exactly two out of A, B, and C obtain the "certificate of passing" --/
def P_exactly_two_pass : ℝ :=
  P_certificate 1 * P_certificate 2 * (1 - P_certificate 3) +
  P_certificate 1 * (1 - P_certificate 2) * P_certificate 3 +
  (1 - P_certificate 1) * P_certificate 2 * P_certificate 3

theorem probability_exactly_two : 
  P_exactly_two_pass = 11/30 :=
by
  sorry

end highest_certificate_probability_probability_exactly_two_l624_624066


namespace count_of_throwers_l624_624397

noncomputable def numberOfThrowers : ℕ :=
  let T := 31
  let totalPlayers := 70
  let totalRightHandedPlayers := 57
  let nonThrowers := totalPlayers - T
  let rightHandedNonThrowers := (2 * nonThrowers) / 3
  have eq1 : T + nonThrowers = totalPlayers := by simp [totalPlayers, nonThrowers]
  have eq2 : rightHandedNonThrowers + T = totalRightHandedPlayers := by simp [rightHandedNonThrowers, totalRightHandedPlayers, T]
  T

theorem count_of_throwers (T : ℕ) (totalPlayers : ℕ) (totalRightHandedPlayers : ℕ) (nonThrowers : ℕ) (rightHandedNonThrowers : ℕ) :
  totalPlayers = 70 →
  totalRightHandedPlayers = 57 →
  T + nonThrowers = totalPlayers →
  (2 * nonThrowers) / 3 + T = totalRightHandedPlayers →
  T = 31 :=
by 
  intros h_totalPlayers h_totalRightHandedPlayers h_eq1 h_eq2
  have h_nonThrowers : nonThrowers = 70 - T := by linarith [h_eq1, h_totalPlayers]
  rw h_nonThrowers at h_eq2
  have : (2 * (70 - T)) / 3 + T = 57 := h_eq2
  linarith

end count_of_throwers_l624_624397


namespace solve_exp_inequality_l624_624060

theorem solve_exp_inequality (x : ℝ) : 1 ≤ 2^x ∧ 2^x ≤ 8 ↔ 0 ≤ x ∧ x ≤ 3 :=
by {
  sorry,
}

end solve_exp_inequality_l624_624060


namespace fraction_addition_l624_624623

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624623


namespace max_heaps_660_l624_624002

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624002


namespace james_drive_time_to_canada_l624_624348

theorem james_drive_time_to_canada : 
  ∀ (distance speed stop_time : ℕ), 
    speed = 60 → 
    distance = 360 → 
    stop_time = 1 → 
    (distance / speed) + stop_time = 7 :=
by
  intros distance speed stop_time h1 h2 h3
  sorry

end james_drive_time_to_canada_l624_624348


namespace units_digit_of_product_l624_624789

theorem units_digit_of_product : 
  (27 % 10 = 7) ∧ (68 % 10 = 8) → ((27 * 68) % 10 = 6) :=
by sorry

end units_digit_of_product_l624_624789


namespace liam_shots_l624_624033

theorem liam_shots (initial_made initial_attempts new_attempts total_avg new_made : ℕ) :
  initial_made = 24 →
  initial_attempts = 60 →
  new_attempts = 15 →
  total_avg = 45 →
  ((initial_made + new_made) * 100 / (initial_attempts + new_attempts) = total_avg) →
  new_made = 9 :=
by
  intros,
  sorry

end liam_shots_l624_624033


namespace mutually_exclusive_not_complementary_l624_624098

def event_odd (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_greater_than_5 (n : ℕ) : Prop := n = 6

theorem mutually_exclusive_not_complementary :
  (∀ n : ℕ, event_odd n → ¬ event_greater_than_5 n) ∧
  (∃ n : ℕ, ¬ event_odd n ∧ ¬ event_greater_than_5 n) :=
by
  sorry

end mutually_exclusive_not_complementary_l624_624098


namespace correct_derivatives_l624_624190

theorem correct_derivatives :
  ((deriv (λ x : ℝ, 2 * x / (x^2 + 1)) = (λ x : ℝ, (2 - 2 * x^2) / (x^2 + 1)^2)) ∧
   (deriv (λ x : ℝ, real.exp (3 * x + 1)) = (λ x : ℝ, 3 * real.exp (3 * x + 1)))) :=
by { split; sorry }

end correct_derivatives_l624_624190


namespace sqrt_meaningful_condition_l624_624472

theorem sqrt_meaningful_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (1 - x)) → x ≤ 1 :=
by
  assume h,
  sorry

end sqrt_meaningful_condition_l624_624472


namespace sum_of_monomials_l624_624317

theorem sum_of_monomials (m n : ℕ) (h1 : m = 3) (h2 : n = 2) : m + n = 5 := 
by
  rw [h1, h2]
  exact rfl

end sum_of_monomials_l624_624317


namespace books_selection_count_l624_624228

-- Define the total number of books on the shelf
def total_books : ℕ := 8

-- Define the number of books to be selected
def books_to_select : ℕ := 5

-- Define the inclusion of one specific book
def specific_book_included : ℕ := 1

-- Define the problem statement
theorem books_selection_count : ∃ n : ℕ, n = (finset.card (finset.filter (λ s, specific_book_included ∈ s) (finset.powerset_len books_to_select (finset.range total_books)))) := 35 := 
by {
  -- Placeholder for the proof
  sorry
}

end books_selection_count_l624_624228


namespace sum_of_squares_increase_by_800_l624_624810

theorem sum_of_squares_increase_by_800
  (x : Fin 100 → ℝ)
  (h : ∑ j, x j ^ 2 = ∑ j, (x j + 2) ^ 2) :
  (∑ j, (x j + 4) ^ 2) - (∑ j, x j ^ 2) = 800 := 
by
  sorry

end sum_of_squares_increase_by_800_l624_624810


namespace smallest_period_and_monotonicity_l624_624277

theorem smallest_period_and_monotonicity 
  (ω : ℝ) (φ : ℝ) 
  (hω : ω > 0) (hφ : |φ| < π / 2)
  (symmetry_axes : ∀ x : ℝ, f x = f (x + π / 2))
  (function_def : ∀ x : ℝ, f x = sin (ω * x + φ) - sqrt 3 * cos (ω * x + φ)) :
  (∀ x : ℝ, f (x + π) = f x) ∧ 
  (∀ x : ℝ, (0 < x ∧ x < π / 2) → f x < f (x + ε)) :=
by
  -- Proof steps will go here
  sorry

end smallest_period_and_monotonicity_l624_624277


namespace tangent_line_to_circle_polar_l624_624905

-- Definitions
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def point_polar_coordinates (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 ∧ θ = Real.pi / 4
def tangent_line_polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- Theorem Statement
theorem tangent_line_to_circle_polar {ρ θ : ℝ} :
  (∃ ρ θ, polar_circle_equation ρ θ) →
  (∃ ρ θ, point_polar_coordinates ρ θ) →
  tangent_line_polar_equation ρ θ :=
sorry

end tangent_line_to_circle_polar_l624_624905


namespace max_heaps_of_stones_l624_624015

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624015


namespace sum_of_squares_increase_by_800_l624_624808

theorem sum_of_squares_increase_by_800
  (x : Fin 100 → ℝ)
  (h : ∑ j, x j ^ 2 = ∑ j, (x j + 2) ^ 2) :
  (∑ j, (x j + 4) ^ 2) - (∑ j, x j ^ 2) = 800 := 
by
  sorry

end sum_of_squares_increase_by_800_l624_624808


namespace find_factorial_number_l624_624544

theorem find_factorial_number : 
  ∃ (n : ℤ), n = 1307674368000 ∧ (n = (Nat.fact 15)) :=
begin
  use 1307674368000,
  split,
  {
    refl,
  },
  {
    sorry,
  }
end

end find_factorial_number_l624_624544


namespace min_bounces_for_height_less_than_two_l624_624512

theorem min_bounces_for_height_less_than_two : 
  ∃ (k : ℕ), (20 * (3 / 4 : ℝ)^k < 2 ∧ ∀ n < k, ¬(20 * (3 / 4 : ℝ)^n < 2)) :=
sorry

end min_bounces_for_height_less_than_two_l624_624512


namespace sum_of_squares_change_l624_624814

def x : ℕ → ℝ := sorry
def y (i : ℕ) : ℝ := x i + 2
def z (i : ℕ) : ℝ := x i + 4

theorem sum_of_squares_change :
  (∑ j in Finset.range 100, (z j)^2) - (∑ j in Finset.range 100, (x j)^2) = 800 :=
by
  sorry

end sum_of_squares_change_l624_624814


namespace probability_circles_intersection_l624_624243

theorem probability_circles_intersection (a : ℝ) (h : a ∈ Icc (-3) 3):
  ((2:ℝ) - 0) / ((3:ℝ) - (-3:ℝ)) = 1/3 :=
by
  sorry

end probability_circles_intersection_l624_624243


namespace sqrt_meaningful_condition_l624_624469

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end sqrt_meaningful_condition_l624_624469


namespace problem1_problem2_l624_624882

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosB : ℝ)

-- Condition: a, b, c are in geometric progression
axiom geom_prog (h1 : b ^ 2 = a * c) : true

-- Condition: cosB = 3/5
axiom cosB_val (h2 : cosB = 3 / 5) : true

-- Condition: dot product of vectors BA and BC = 3
axiom dot_product (h3 : 3 = 3) : true

-- Proof problem 1: Given conditions, prove the value of (cos A / sin A) + (cos C / sin C) = 5 / 4
theorem problem1 (h1 : b ^ 2 = a * c) (h2 : cosB = 3 / 5) : (cos A / sin A + cos C / sin C) = 5 / 4 := by
  sorry

-- Proof problem 2: Given conditions, prove the value of a + c = sqrt 21
theorem problem2 (h3 : overrightarrow_BA_dot_overrightarrow_BC = 3) (h2 : cosB = 3 / 5) : a + c = Real.sqrt 21 := by
  sorry

end problem1_problem2_l624_624882


namespace finite_decimal_conversion_l624_624122

theorem finite_decimal_conversion :
  (∃ (a b : ℕ), a = 9 ∧ b = 12 ∧ (∃ (c : ℕ), (a * c) % b = 0) ∧
    (∀ (d : ℕ), (b = 2 ^ d ∨ b = 5 ^ d ∨ (b = (2 ^ d * 5 ^ d))))) :=
by
  use [9, 12, 4]
  constructor
  . exact rfl
  constructor
  . exact rfl
  constructor
  . use 3
  . exact rfl
  . sorry

end finite_decimal_conversion_l624_624122


namespace greatest_growth_rate_city_is_G_l624_624326

noncomputable def population_1990 : ℕ → ℕ
| 0 := 50
| 1 := 60
| 2 := 70
| 3 := 100
| 4 := 150

noncomputable def population_2000 : ℕ → ℕ
| 0 := 60
| 1 := 90
| 2 := 80
| 3 := 110
| 4 := 180

noncomputable def growth_rate (city : ℕ) : ℝ :=
if city = 2 then (80 / 70) * 1.10 else (population_2000 city) / (population_1990 city)

theorem greatest_growth_rate_city_is_G : growth_rate 1 > growth_rate 0 ∧
                                          growth_rate 1 > growth_rate 2 ∧
                                          growth_rate 1 > growth_rate 3 ∧
                                          growth_rate 1 > growth_rate 4 := sorry

end greatest_growth_rate_city_is_G_l624_624326


namespace find_number_l624_624094

theorem find_number (x : ℝ) (h : x / 5 + 23 = 42) : x = 95 :=
by
  -- Proof placeholder
  sorry

end find_number_l624_624094


namespace largest_even_n_satisfying_ineq_l624_624209

theorem largest_even_n_satisfying_ineq :
  ∃ n : ℕ, ∀ x : ℝ, even n ∧ (sin x)^n - (cos x)^n ≤ 1 ∧ ∀ m : ℕ, even m ∧ (∀ x : ℝ, (sin x)^m - (cos x)^m ≤ 1) → m ≤ n :=
begin
  sorry
end

end largest_even_n_satisfying_ineq_l624_624209


namespace line_passes_through_point_l624_624148

theorem line_passes_through_point (m : ℝ) :
  let line_eq := λ x y : ℝ, mx - y + m + 2 = 0 in
  line_eq (-1) 2 :=
by
  -- sorry is used here to indicate the proof is omitted
  sorry

end line_passes_through_point_l624_624148


namespace max_heaps_660_l624_624007

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624007


namespace tiles_per_row_l624_624432

theorem tiles_per_row (area_sq_ft : ℕ) (tile_size_inch : ℕ) (side_length_ft side_length_inch : ℕ)
  (h1 : area_sq_ft = 144)
  (h2 : tile_size_inch = 8)
  (h3 : side_length_ft = Int.sqrt area_sq_ft)
  (h4 : side_length_inch = side_length_ft * 12) :
  side_length_inch / tile_size_inch = 18 :=
by
  -- Declaring necessary variables
  have h5 : area_sq_ft = 144 := h1
  have h6 : tile_size_inch = 8 := h2
  have h7 : side_length_ft = Int.sqrt area_sq_ft := h3
  have h8 : side_length_inch = side_length_ft * 12 := h4

  -- Step 1: Calculate the side-length in feet
  have step1 : side_length_ft = 12 :=
    calc side_length_ft = Int.sqrt 144 : h7
                   ... = 12          : by norm_num

  -- Step 2: Convert side length from feet to inches
  have step2 : side_length_inch = 12 * 12 :=
    calc side_length_inch = side_length_ft * 12 : h8
                        ... = 12 * 12           : by rw [step1]

  -- Step 3: Divide side length in inches by tile size to find number of tiles per row
  have step3 : side_length_inch / tile_size_inch = 144 / 8 :=
    calc side_length_inch / tile_size_inch = (12 * 12) / 8 : by rw [step2]
                                       ... = 144 / 8       : by norm_num

  show side_length_inch / tile_size_inch = 18 from
    calc side_length_inch / tile_size_inch = 144 / 8 : step3
                                       ... = 18       : by norm_num

  sorry

end tiles_per_row_l624_624432


namespace max_heaps_of_stones_l624_624009

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624009


namespace functions_equal_abs_sqrt_l624_624123

theorem functions_equal_abs_sqrt (x : ℝ) : 
  abs x = sqrt (x^2) :=
sorry

end functions_equal_abs_sqrt_l624_624123


namespace concurrency_of_AP_BD_CE_l624_624933

open EuclideanGeometry

variable (A B C P D E : Point)
variable (h : IsIncenter P A B C)
variable (h₁ : angle A P B = angle A C B + angle A P C - angle A B C)
variable (h₂ : IsIncenter D A P B)
variable (h₃ : IsIncenter E A P C)

theorem concurrency_of_AP_BD_CE :
  concurrency [line_through A P, line_through B D, line_through C E] :=
  sorry

end concurrency_of_AP_BD_CE_l624_624933


namespace pudding_distribution_l624_624091

theorem pudding_distribution {puddings students : ℕ} (h1 : puddings = 315) (h2 : students = 218) : 
  ∃ (additional_puddings : ℕ), additional_puddings >= 121 ∧ ∃ (cups_per_student : ℕ), 
  (puddings + additional_puddings) ≥ students * cups_per_student :=
by
  sorry

end pudding_distribution_l624_624091


namespace remaining_units_l624_624546

theorem remaining_units : 
  ∀ (total_units : ℕ) (first_half_fraction : ℚ) (additional_units : ℕ), 
  total_units = 2000 →
  first_half_fraction = 3 / 5 →
  additional_units = 300 →
  (total_units - (first_half_fraction * total_units).toNat - additional_units) = 500 := by
  intros total_units first_half_fraction additional_units htotal hunits_fraction hadditional
  sorry

end remaining_units_l624_624546


namespace part_a_solution_l624_624134

theorem part_a_solution (x y z : ℝ) :
  (x + 3 * y = 4 * y^3) ∧ (y + 3 * z = 4 * z^3) ∧ (z + 3 * x = 4 * x^3) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) ∨
  (x = cos(π / 14) ∧ y = -cos(5 * π / 14) ∧ z = cos(3 * π / 14)) ∨
  (x = -cos(π / 14) ∧ y = cos(5 * π / 14) ∧ z = -cos(3 * π / 14)) ∨
  (x = cos(π / 7) ∧ y = -cos(2 * π / 7) ∧ z = cos(3 * π / 7)) ∨
  (x = -cos(π / 7) ∧ y = cos(2 * π / 7) ∧ z = -cos(3 * π / 7)) ∨
  (x = cos(π / 13) ∧ y = -cos(π / 13) ∧ z = cos(3 * π / 13)) ∨
  (x = -cos(π / 13) ∧ y = cos(π / 13) ∧ z = -cos(3 * π / 13)) ∨
  (x = cos(π / 13) ∧ y = -cos(3 * π / 13) ∧ z = cos(π / 13)) :=
sorry

end part_a_solution_l624_624134


namespace sum_of_squares_increase_by_800_l624_624812

theorem sum_of_squares_increase_by_800
  (x : Fin 100 → ℝ)
  (h : ∑ j, x j ^ 2 = ∑ j, (x j + 2) ^ 2) :
  (∑ j, (x j + 4) ^ 2) - (∑ j, x j ^ 2) = 800 := 
by
  sorry

end sum_of_squares_increase_by_800_l624_624812


namespace problem1_increasing_decreasing_intervals_problem1_extremum_problem2_range_of_a_l624_624275

noncomputable def f : ℝ → ℝ := λ x, (1 / 2) * x^2 + x - 2 * real.log x
noncomputable def g : ℝ → ℝ := λ x, (1 / 2) * x^2 + a * x - 2 * real.log x

-- Problem 1
theorem problem1_increasing_decreasing_intervals :
  (∀ x : ℝ, (0 < x ∧ x < 1) → f' x < 0) ∧
  (∀ x : ℝ, (1 < x) → f' x > 0) :=
sorry

theorem problem1_extremum :
  f 1 = 3 / 2 :=
sorry

-- Problem 2
theorem problem2_range_of_a {a : ℝ} :
  (∀ x : ℝ, (0 < x ∧ x ≤ 2) → g' x ≤ 0) → a ≤ -1 :=
sorry

end problem1_increasing_decreasing_intervals_problem1_extremum_problem2_range_of_a_l624_624275


namespace standard_equation_of_ellipse_range_of_triangle_area_l624_624261

noncomputable def focus_coordinates : ℝ × ℝ := (0, -1)
noncomputable def eccentricity : ℝ := (Real.sqrt 3) / 3
noncomputable def ellipse_center : ℝ × ℝ := (0, 0)
noncomputable def semi_major_axis (c : ℝ) (e : ℝ) : ℝ := c / e
noncomputable def semi_minor_axis (a : ℝ) (c : ℝ) : ℝ := Real.sqrt (a^2 - c^2)

theorem standard_equation_of_ellipse :
  let c := 1
  let a := semi_major_axis c eccentricity
  let b := semi_minor_axis a c
  ∃ a b : ℝ, a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ 
  (∀ x y : ℝ, (y^2 / a^2 + x^2 / b^2 = 1 ↔ y^2 / 3 + x^2 / 2 = 1)) :=
sorry

theorem range_of_triangle_area :
  let c := 1
  let a := semi_major_axis c eccentricity
  let b := semi_minor_axis a c
  ∀ k : ℝ,
  let f2 := (0, 1)
  let t := Real.sqrt (1 + k^2)
  0 < t →
  (∀ x1 x2 : ℝ, x1 + x2 = 4 * k / (3 + 2 * k^2) ∧ x1 * x2 = -4 / (3 + 2 * k^2) →
    0 < |x1 - x2| ∧ |x1 - x2| ≤ 4 * Real.sqrt 3 / 3) → 
  (∃ s : ℝ, s = |x1 - x2| ∧ s ∈ set.Ioc 0 (4 * Real.sqrt 3 / 3)) :=
sorry

end standard_equation_of_ellipse_range_of_triangle_area_l624_624261


namespace Tony_fever_day_5_Tony_fever_above_threshold_l624_624104

def Tony_normal_temperature : ℝ := 95
def fever_threshold : ℝ := 100

def illness_A_temperature_increase : ℝ := 10
def illness_B_temperature_increase : ℝ := 4
def illness_C_temperature_decrease : ℝ := -2

def overlap_AB_temperature_increase : ℝ := illness_B_temperature_increase * 2
def all_illnesses_overlap_decrease : ℝ := -3

def Tony_temperature_day_5 : ℝ := 
  Tony_normal_temperature + 
  illness_A_temperature_increase + 
  overlap_AB_temperature_increase + 
  illness_C_temperature_decrease + 
  all_illnesses_overlap_decrease

theorem Tony_fever_day_5: Tony_temperature_day_5 = 108 :=
by {
  simp [Tony_normal_temperature, illness_A_temperature_increase, overlap_AB_temperature_increase,
    illness_C_temperature_decrease, all_illnesses_overlap_decrease],
  norm_num,
}

theorem Tony_fever_above_threshold: Tony_temperature_day_5 - fever_threshold = 8 :=
by {
  simp [Tony_temperature_day_5, fever_threshold, Tony_fever_day_5],
  norm_num,
}

end Tony_fever_day_5_Tony_fever_above_threshold_l624_624104


namespace committee_selection_count_l624_624885

-- Definition of the problem condition: Club of 12 people, one specific person must always be on the committee.
def club_size : ℕ := 12
def committee_size : ℕ := 4
def specific_person_included : ℕ := 1

-- Number of ways to choose 3 members from the other 11 people
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem committee_selection_count : choose 11 3 = 165 := 
  sorry

end committee_selection_count_l624_624885


namespace complex_exponential_form_theta_eq_pi_div_3_l624_624488

theorem complex_exponential_form_theta_eq_pi_div_3:
  ∃ θ : ℝ, 1 + complex.I * √3 = 2 * complex.exp (complex.I * θ) ∧ θ = (π / 3) :=
sorry

end complex_exponential_form_theta_eq_pi_div_3_l624_624488


namespace quadratic_intersects_x_axis_l624_624857

theorem quadratic_intersects_x_axis (a b : ℝ) (h : a ≠ 0) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 - (b^2 / (4 * a)) = 0 ∧ a * x2^2 + b * x2 - (b^2 / (4 * a)) = 0 := by
  sorry

end quadratic_intersects_x_axis_l624_624857


namespace sum_of_squares_increased_l624_624819

theorem sum_of_squares_increased (x : Fin 100 → ℝ) 
  (h : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  ∑ i, (x i + 4) ^ 2 = ∑ i, x i ^ 2 + 800 := 
by
  sorry

end sum_of_squares_increased_l624_624819


namespace Iris_shorts_l624_624346

theorem Iris_shorts :
  ∃ s, (3 * 10) + s * 6 + (4 * 12) = 90 ∧ s = 2 := 
by
  existsi 2
  sorry

end Iris_shorts_l624_624346


namespace total_people_at_evening_l624_624953

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l624_624953


namespace sum_of_x_and_y_l624_624314

theorem sum_of_x_and_y (x y : ℕ) (h_pos_x: 0 < x) (h_pos_y: 0 < y) (h_gt: x > y) (h_eq: x + x * y = 391) : x + y = 39 :=
by
  sorry

end sum_of_x_and_y_l624_624314


namespace number_of_pairs_l624_624459

open Nat

theorem number_of_pairs (a b : ℕ) (ha1 : a > 0) (hb1 : b > 0)
  (h_sum : a + b = 667)
  (h_lcm_gcd : lcm a b = 120 * gcd a b) :
  (setOf (λ ab : ℕ × ℕ, ab.1 + ab.2 = 667 ∧ lcm ab.1 ab.2 = 120 * gcd ab.1 ab.2)).card = 2 := 
by
  sorry

end number_of_pairs_l624_624459


namespace doubled_perimeter_of_square_l624_624536

theorem doubled_perimeter_of_square (s : ℝ) (h : s^2 = 900) : 2 * (4 * s) = 240 :=
by
  have s_pos : s = real.sqrt 900 := by sorry
  have perimeter : 4 * s = 120 := by sorry
  have new_perimeter : 2 * (4 * s) = 2 * 120 := by sorry
  show 2 * (4 * s) = 240, by sorry

end doubled_perimeter_of_square_l624_624536


namespace power_exponent_multiplication_l624_624175

variable (a : ℝ)

theorem power_exponent_multiplication : (a^3)^2 = a^6 := sorry

end power_exponent_multiplication_l624_624175


namespace Yanni_found_money_l624_624125

-- Definitions derived from conditions:
def initial_money : ℝ := 0.85
def mother_gift : ℝ := 0.40
def toy_cost : ℝ := 1.60
def money_left : ℝ := 0.15

-- Mathematical proof problem, prove the amount of money found == 0.50 given the conditions
theorem Yanni_found_money :
  initial_money + mother_gift + (Yanni_found_money) - toy_cost = money_left → 
  Yanni_found_money = 0.50 :=
sorry

end Yanni_found_money_l624_624125


namespace scientific_notation_of_188_million_l624_624203

theorem scientific_notation_of_188_million : 
  (188000000 : ℝ) = 1.88 * 10^8 := 
by
  sorry

end scientific_notation_of_188_million_l624_624203


namespace people_at_the_beach_l624_624956

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l624_624956


namespace curveC1_rect_eq_and_type_intersection_distance_l624_624339

def curveC1_polar_equation (ρ θ : ℝ) : Prop := ρ^2 - 6 * ρ * cos θ + 5 = 0

def curveC2_parametric (t : ℝ) : ℝ × ℝ :=
  (t * cos (π / 6), t * sin (π / 6))

theorem curveC1_rect_eq_and_type :
  ∀ (x y : ℝ), (x^2 + y^2 - 6 * x + 5 = 0 ↔ (x - 3)^2 + y^2 = 4) ∧
  (∀ curveC1_cart_eq : x^2 + y^2 - 6 * x + 5 = 0 → (x - 3)^2 + y^2 = 4, ∃ c : ℝ × ℝ × ℝ, c = ((3, 0), 2)) :=
sorry

theorem intersection_distance :
  ∀ (t : ℝ), (let (x, y) := curveC2_parametric t in (x - 3)^2 + y^2 = 4 ↔ t^2 - 3 * (sqrt 3) * t + 5 = 0) ∧
  (∀ t1 t2 : ℝ, (t1 + t2 = 3 * sqrt 3 ∧ t1 * t2 = 5) → abs (t2 - t1) = sqrt 7) :=
sorry

end curveC1_rect_eq_and_type_intersection_distance_l624_624339


namespace fraction_addition_l624_624640

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624640


namespace integral_equality_l624_624768

-- Define the integrand
def integrand (x : ℝ) : ℝ :=
  (Real.sqrt (4 - (x - 2)^2) - x)

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 2

-- The statement to prove
theorem integral_equality :
  ∫ x in lower_bound .. upper_bound, integrand x = Real.pi - 2 :=
sorry

end integral_equality_l624_624768


namespace add_fractions_result_l624_624584

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624584


namespace find_f_neg_two_l624_624829

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (h1 : ∀ a b : ℝ, f (a + b) = f a * f b)
variable (h2 : ∀ x : ℝ, f x > 0)
variable (h3 : f 1 = 1 / 2)

-- State the theorem to prove that f(-2) = 4
theorem find_f_neg_two : f (-2) = 4 :=
by
  sorry

end find_f_neg_two_l624_624829


namespace positive_integer_base_conversion_l624_624992

theorem positive_integer_base_conversion (A B : ℕ) (h1 : A < 9) (h2 : B < 7) 
(h3 : 9 * A + B = 7 * B + A) : 9 * 3 + 4 = 31 :=
by sorry

end positive_integer_base_conversion_l624_624992


namespace fraction_of_color_films_l624_624518

theorem fraction_of_color_films (x y : ℕ) (hx : x ≠ 0) : 
  let bw_films := 40 * x in
  let color_films := 4 * y in
  let selected_bw := (y * bw_films) / (x * 100) in
  let selected_color := color_films in
  let total_selected := selected_bw + selected_color in
  (selected_color : ℚ) / total_selected = 10 / 11 := 
by
  sorry

end fraction_of_color_films_l624_624518


namespace simplify_fraction_l624_624308

noncomputable def simplified_expression (x y : ℝ) : ℝ :=
  (x^2 - (4 / y)) / (y^2 - (4 / x))

theorem simplify_fraction {x y : ℝ} (h : x * y ≠ 4) :
  simplified_expression x y = x / y := 
by 
  sorry

end simplify_fraction_l624_624308


namespace smallest_integer_in_range_l624_624457

-- Given conditions
def is_congruent_6 (n : ℕ) : Prop := n % 6 = 1
def is_congruent_7 (n : ℕ) : Prop := n % 7 = 1
def is_congruent_8 (n : ℕ) : Prop := n % 8 = 1

-- Lean statement for the proof problem
theorem smallest_integer_in_range :
  ∃ n : ℕ, (n > 1) ∧ is_congruent_6 n ∧ is_congruent_7 n ∧ is_congruent_8 n ∧ (n = 169) ∧ (120 ≤ n ∧ n < 210) :=
by
  sorry

end smallest_integer_in_range_l624_624457


namespace fraction_addition_l624_624638

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624638


namespace find_c_in_line_equation_l624_624904

theorem find_c_in_line_equation (c : ℝ) (h : let x := -c / 3, y := -c / 5 in x + y = 16) : c = -30 :=
by
  sorry

end find_c_in_line_equation_l624_624904


namespace factor_expression_l624_624177

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := 
by
  sorry

end factor_expression_l624_624177


namespace oil_press_false_statement_l624_624508

theorem oil_press_false_statement :
  (∀ (presses : ℕ) (oil : ℕ), presses = 5 ∧ oil = 260 → oil / presses * 20 = 1040) →
  ¬ (20 * 260 / 5 = 7200) :=
by
  intros h
  have : 20 * 260 / 5 = 1040 := by sorry
  exact this ⟻ sorry

end oil_press_false_statement_l624_624508


namespace problem_proof_l624_624251

-- Definitions and conditions
variable {α β : Type} [TopologicalSpace α] [TopologicalSpace β]
variable {a b : Set α} [Sub α a] [Sub β b]
variable {line c : Set α} [c ⊆ α]
variable {planeα planeβ : Set α} [Plane planeα] [Plane planeβ] 

-- a ⊥ planeα and b ∥ planeα
variable (h1 : is_perp a planeα) (h2 : is_parallel b planeα)

-- Questions involve lines and planes relationships
def Proposition1 := ∀ (c : Set α), (c ⊆ planeα) → is_perp c a
def Proposition2 := ∃ (c : Set α), (c ∉ planeα) ∧ is_perp c b ∧ is_perp c a
def Proposition3 := ∀ (planeβ : Set α), (a ⊆ planeβ) → is_perp planeβ planeα
def Proposition4 := ∃ (planeβ : Set α), is_perp planeβ planeα ∧ is_perp b planeβ

-- Proof problem statement
theorem problem_proof 
  (h1 : is_perp a planeα) 
  (h2 : is_parallel b planeα) : 
  Proposition1 ∧ Proposition2 ∧ Proposition3 ∧ Proposition4 := 
by 
  -- Proof goes here
  sorry

end problem_proof_l624_624251


namespace angle_B_equals_pi_over_6_l624_624883

theorem angle_B_equals_pi_over_6 (A B C : ℝ) (a b c : ℝ) 
  (ha : a = A)
  (hb : b = B)
  (hc : c = C) 
  (h : a^2 + c^2 - b^2 = real.sqrt 3 * a * c) : B = real.pi / 6 :=
sorry

end angle_B_equals_pi_over_6_l624_624883


namespace find_a_l624_624364

def M : Set ℝ := {x | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem find_a (a : ℝ) : N a ⊆ M ↔ a = -1 ∨ a = 0 ∨ a = 2/3 := 
by
  sorry

end find_a_l624_624364


namespace find_x_for_integers_l624_624503

theorem find_x_for_integers (x : ℝ)
  (a := x - Real.sqrt 2)
  (b := x - (1 / x))
  (c := x + (1 / x))
  (d := x^2 + 2 * Real.sqrt 2)
  (h : (∃! x, (x ≠ Int)) = 1) :
  x = Real.sqrt 2 - 1 :=
sorry

end find_x_for_integers_l624_624503


namespace number_of_possible_values_of_f1_product_of_n_and_s_l624_624189

universe u

noncomputable def S := {x : ℝ // x ≠ 0}

variable {k : ℝ}
variables (f : S → S)

axiom f_condition_1 (x : S) : f ⟨1 / x.1, by {have := x.property, linarith}⟩ = ⟨1 / k * x.1 * f x.1, by {sorry}⟩
axiom f_condition_2 (x y : S) (h : x.1 + y.1 ≠ 0) :
  f ⟨1 / x.1, by {have := x.property, linarith}⟩ + f ⟨1 / y.1, by {have := y.property, linarith}⟩ = k + f ⟨1 / (k * x.1 + k * y.1), by {sorry}⟩

def f1 := f ⟨1, by {norm_num}⟩

theorem number_of_possible_values_of_f1 : ∃! v : S, f1 = v := 
sorry

theorem product_of_n_and_s : (1 : ℝ) * (k * k + k) = k * k + k :=
by sorry

end number_of_possible_values_of_f1_product_of_n_and_s_l624_624189


namespace add_fractions_l624_624726

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624726


namespace max_volume_correct_l624_624403

noncomputable def max_volume (R T0 P0 a b c : ℝ) (h : c^2 < a^2 + b^2) : ℝ :=
  let sqrt_term := real.sqrt (a^2 + b^2 - c^2)
  (R * T0 / P0) * ((a * sqrt_term + b * c) / (b * sqrt_term - a * c))

theorem max_volume_correct (R T0 P0 a b c : ℝ) (h : c^2 < a^2 + b^2) :
  max_volume R T0 P0 a b c h = (R * T0 / P0) * ((a * real.sqrt (a^2 + b^2 - c^2) + b * c) / (b * real.sqrt (a^2 + b^2 - c^2) - a * c)) :=
by {
  refl
}

end max_volume_correct_l624_624403


namespace smallest_odd_five_digit_number_tens_place_l624_624112

theorem smallest_odd_five_digit_number_tens_place :
  ∀ (n : ℕ),
  (∃ (a b c d e : ℕ),
    {a, b, c, d, e} = {1, 2, 3, 4, 9} ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    n % 2 = 1) →
  (∃ (d : ℕ), d = 9 ∧ n = 23491) :=
by
  sorry

end smallest_odd_five_digit_number_tens_place_l624_624112


namespace add_fractions_l624_624667

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624667


namespace ball_returns_to_P_l624_624115

theorem ball_returns_to_P
  (R d : ℝ)
  (PO_eq_d : PO = d)
  (arbitrary_point_P : P ∈ diameter O)
  (path_PA_AB : path P → A → B)
  (reflect_angle_eq : ∀ (x : pt), ∠QPA x = 2 * ∠OAB)
  (isosceles_triangle_OAB : is_isosceles (triangle O A B))
  (reflection_rule : ∠PAB = ∠PBA) :
  let α := ∠APO,
      r := R - d,
      α_eq_sin_inv := α = arcsin (∠PO r),
      R_eq_sqrt := R = sqrt ((R - d)*2*8) in
  ∃ (r : ℝ), r = (1/2) * (sqrt(R^2 + 2 * (r - d))) := 
sorry

end ball_returns_to_P_l624_624115


namespace solve_sin_eqn_l624_624421

theorem solve_sin_eqn (x : ℝ) : 
  (sin(2 * x) - real.pi * sin(x)) * sqrt(11 * x^2 - x^4 - 10) = 0 ↔ 
  x = -real.sqrt 10 ∨ x = -real.pi ∨ x = -1 ∨ x = 1 ∨ x = real.pi ∨ x = real.sqrt 10 :=
by
  have domain_condition :
    11 * x^2 - x^4 - 10 ≥ 0
  from sorry,
  sorry

end solve_sin_eqn_l624_624421


namespace jump_difference_l624_624972

variable (runningRicciana jumpRicciana runningMargarita : ℕ)

theorem jump_difference :
  (runningMargarita + (2 * jumpRicciana - 1)) - (runningRicciana + jumpRicciana) = 1 :=
by
  -- Given conditions
  let runningRicciana := 20
  let jumpRicciana := 4
  let runningMargarita := 18
  -- The proof is omitted (using 'sorry')
  sorry

end jump_difference_l624_624972


namespace power_function_exponent_l624_624856

theorem power_function_exponent 
  (a : ℝ) 
  (h : (λ x : ℝ, x^a) 4 = 2) : 
  a = 1 / 2 :=
sorry

end power_function_exponent_l624_624856


namespace rajas_salary_percentage_less_than_rams_l624_624049

-- Definitions from the problem conditions
def raja_salary : ℚ := sorry -- Placeholder, since Raja's salary doesn't need a fixed value
def ram_salary : ℚ := 1.25 * raja_salary

-- Theorem to be proved
theorem rajas_salary_percentage_less_than_rams :
  ∃ r : ℚ, (ram_salary - raja_salary) / ram_salary * 100 = 20 :=
by
  sorry

end rajas_salary_percentage_less_than_rams_l624_624049


namespace sum_of_squares_increase_by_l624_624825

theorem sum_of_squares_increase_by {n : ℕ} (h1 : n = 100) (x : Fin n.succ → ℝ)
  (h2 : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  (∑ i, (x i + 4) ^ 2) = (∑ i, x i ^ 2) + 800 :=
by sorry

end sum_of_squares_increase_by_l624_624825


namespace third_term_of_sequence_l624_624246

theorem third_term_of_sequence (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = (1 / 2) * a n + (1 / (2 * n))) : a 3 = 3 / 4 := by
  sorry

end third_term_of_sequence_l624_624246


namespace impossible_odd_sum_l624_624313

theorem impossible_odd_sum (n m : ℤ) (h1 : (n^3 + m^3) % 2 = 0) (h2 : (n^3 + m^3) % 4 = 0) : (n + m) % 2 = 0 :=
sorry

end impossible_odd_sum_l624_624313


namespace remainder_when_divided_by_x_minus_4_l624_624116

noncomputable def f (x : ℝ) : ℝ := x^4 - 9 * x^3 + 21 * x^2 + x - 18

theorem remainder_when_divided_by_x_minus_4 : f 4 = 2 :=
by
  sorry

end remainder_when_divided_by_x_minus_4_l624_624116


namespace points_needed_proof_l624_624942

variable (last_home_game_score first_away_game_score second_away_game_score third_away_game_score next_game_score : ℕ)

def cumulative_score : ℕ :=
  last_home_game_score + first_away_game_score + second_away_game_score + third_away_game_score

def cumulative_score_goal (four_times_last_home: ℕ) : Prop := 
  four_times_last_home = 4 * last_home_game_score

def points_needed_in_next_game (goal current: ℕ) : ℕ :=
  goal - current

theorem points_needed_proof :
  last_home_game_score = 62 ->
  first_away_game_score = last_home_game_score / 2 ->
  second_away_game_score = first_away_game_score + 18 ->
  third_away_game_score = second_away_game_score + 2 ->
  next_game_score = 55 ->
  cumulative_score_goal 248 ->
  points_needed_in_next_game 248 cumulative_score = next_game_score :=
by {
  intros h1 h2 h3 h4 h5 h6,
  have current_score := h1 + h2 + h3 + h4,
  rw h5,
  have goal := h6,
  apply congr,
  exact h5,
  sorry
}

end points_needed_proof_l624_624942


namespace acute_triangle_angle_range_l624_624892

theorem acute_triangle_angle_range {A B C D M H : Type*} [triangle A B C] 
  (angle_bisector : is_angle_bisector A D) 
  (median : is_median B M) 
  (altitude : is_altitude C H) 
  (intersect_at_single_point: AD ∩ BM ∩ CH ≠ ∅) :
  51.833 < measure_angle A && measure_angle A < 90 :=
sorry

end acute_triangle_angle_range_l624_624892


namespace complementary_angles_in_triangle_l624_624911

-- Definitions
variables {α β γ : Type} [ordered_field α]

-- Declare the sides and angles
variables (a b c : α) (A B C : α)

-- Translate the given conditions
def sides_opposite_angles_in_triangle (A B C a b c : α) : Prop :=
  -- Placeholder for any additional conditions for triangle if needed
  true

def sides_satisfy_equation (a b c : α) : Prop :=
  b^2 - a^2 = c^2

-- Lean Theorem Statement
theorem complementary_angles_in_triangle
  (A B C a b c : α)
  (h1 : sides_opposite_angles_in_triangle A B C a b c)
  (h2 : sides_satisfy_equation a b c) :
  A + C = 90 :=
begin
  sorry
end

end complementary_angles_in_triangle_l624_624911


namespace max_heaps_660_l624_624008

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624008


namespace polar_coordinates_of_M_parametric_equation_of_line_AM_l624_624839

def point_on_semicircle (theta : ℝ) : Prop :=
  0 ≤ theta ∧ theta ≤ π

def coordinates_of_P (theta : ℝ) : (ℝ × ℝ) :=
  (cos theta, sin theta)

def coordinates_of_A : (ℝ × ℝ) :=
  (1, 0)

def coordinates_of_O : (ℝ × ℝ) :=
  (0, 0)

def point_on_ray_OP (P M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≥ 0 ∧ M.1 = t * P.1 ∧ M.2 = t * P.2

def length_OM (M : ℝ × ℝ) : ℝ :=
  real.sqrt (M.1^2 + M.2^2)

def arc_length_AP (P : ℝ × ℝ) : ℝ :=
  (1 - θ) * (π / 2)

theorem polar_coordinates_of_M (M : ℝ × ℝ) :
  ∀ θ : ℝ, point_on_semicircle θ → 
  length_OM M = π / 3 → M = (π / 3, π / 3) :=
sorry

theorem parametric_equation_of_line_AM (A M : ℝ × ℝ) :
  M = (π / 6, (√3 * π) / 6) → A = (1,0) →
  ∀ t : ℝ, (AM.1 = 1 + (π / 6 - 1) * t) ∧ (AM.2 = (√3 * π / 6) * t) :=
sorry

end polar_coordinates_of_M_parametric_equation_of_line_AM_l624_624839


namespace centroid_coordinates_of_tetrahedron_l624_624838

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (O A B C G G1 : V) (OG1_subdivides : G -ᵥ O = 3 • (G1 -ᵥ G))
variable (A_centroid : G1 -ᵥ O = (1/3 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O))

-- The main proof problem
theorem centroid_coordinates_of_tetrahedron :
  G -ᵥ O = (1/4 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O) :=
sorry

end centroid_coordinates_of_tetrahedron_l624_624838


namespace fraction_addition_l624_624631

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624631


namespace maximum_triangle_area_l624_624063

noncomputable def max_triangle_area (a b c : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  in sqrt (p * (p - a) * (p - b) * (p - c))

theorem maximum_triangle_area (a b : ℝ) (h1 : a + b = 10) (h2 : c = 6) :
  max_triangle_area a b c ≤ 12 :=
by
  sorry

end maximum_triangle_area_l624_624063


namespace no_equalities_l624_624061

def f1 (x : ℤ) : ℤ := x * (x - 2007)
def f2 (x : ℤ) : ℤ := (x - 1) * (x - 2006)
def f1004 (x : ℤ) : ℤ := (x - 1003) * (x - 1004)

theorem no_equalities (x : ℤ) (h : 0 ≤ x ∧ x ≤ 2007) :
  ¬(f1 x = f2 x ∨ f1 x = f1004 x ∨ f2 x = f1004 x) :=
by
  sorry

end no_equalities_l624_624061


namespace total_daily_cost_correct_l624_624350

/-- Definition of the daily wages of each type of worker -/
def daily_wage_worker : ℕ := 100
def daily_wage_electrician : ℕ := 2 * daily_wage_worker
def daily_wage_plumber : ℕ := (5 * daily_wage_worker) / 2 -- 2.5 times daily_wage_worker
def daily_wage_architect : ℕ := 7 * daily_wage_worker / 2 -- 3.5 times daily_wage_worker

/-- Definition of the total daily cost for one project -/
def daily_cost_one_project : ℕ :=
  2 * daily_wage_worker +
  daily_wage_electrician +
  daily_wage_plumber +
  daily_wage_architect

/-- Definition of the total daily cost for three projects -/
def total_daily_cost_three_projects : ℕ :=
  3 * daily_cost_one_project

/-- Theorem stating the overall labor costs for one day for all three projects -/
theorem total_daily_cost_correct :
  total_daily_cost_three_projects = 3000 :=
by
  -- Proof omitted
  sorry

end total_daily_cost_correct_l624_624350


namespace area_of_region_l624_624478

theorem area_of_region : 
  ∀ (x y : ℝ), x^2 + y^2 + 4 * x - 6 * y = 9 → (22 * Real.pi) :=
begin
  sorry
end

end area_of_region_l624_624478


namespace students_taking_history_or_geography_but_not_both_l624_624204

theorem students_taking_history_or_geography_but_not_both 
  (h : ℕ) (b : ℕ) (g_only : ℕ) : ∃ n : ℕ, n = 33 :=
by {
  let h_only := h - b,   -- Students taking only history
  let h_or_g_not_both := h_only + g_only,
  use h_or_g_not_both,
  sorry    -- Proof omitted
}

end students_taking_history_or_geography_but_not_both_l624_624204


namespace length_MN_eq_a_l624_624176

open EuclideanGeometry

variables {K1 K2 : Circle} {A B C M N : Point} {a : ℝ}

-- Conditions
axiom circles_intersect (h1 : K1 ∩ K2 = {A}) 
axiom line_through_centers (h2 : Line_through_centers K1 K2 A B C)
axiom third_line_parallel (h3 : Third_line_parallel A B C M N)
axiom length_BC (h4 : dist B C = a)

-- Theorem to prove
theorem length_MN_eq_a : dist M N = a := 
sorry

end length_MN_eq_a_l624_624176


namespace right_triangle_cos_pq_l624_624324

theorem right_triangle_cos_pq (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : c = 13) (h2 : b / c = 5/13) : a = 12 :=
by
  sorry

end right_triangle_cos_pq_l624_624324


namespace spaceship_not_moving_time_l624_624534

-- Definitions based on the conditions given
def total_journey_time : ℕ := 3 * 24  -- 3 days in hours

def first_travel_time : ℕ := 10
def first_break_time : ℕ := 3
def second_travel_time : ℕ := 10
def second_break_time : ℕ := 1

def subsequent_travel_period : ℕ := 11  -- 11 hours traveling, then 1 hour break

-- Function to compute total break time
def total_break_time (total_travel_time : ℕ) : ℕ :=
  let remaining_time := total_journey_time - (first_travel_time + first_break_time + second_travel_time + second_break_time)
  let subsequent_breaks := remaining_time / subsequent_travel_period
  first_break_time + second_break_time + subsequent_breaks

theorem spaceship_not_moving_time : total_break_time total_journey_time = 8 := by
  sorry

end spaceship_not_moving_time_l624_624534


namespace red_fraction_exists_l624_624936

-- Define r(n) which is the fraction of integers painted red
def r (n : ℕ) : ℚ := sorry  -- The actual function r(n) needs to be defined based on the painting function, omitted for brevity

-- Define the problem parameters
def N (p : ℕ) [Fact (Nat.Prime p)] : ℕ := (p^3 - p) / 4 - 1

theorem red_fraction_exists (p : ℕ) [hp : Fact (Nat.Prime p)] (hpo : Odd p) :
  ∃ a (ha : a ∈ Finset.range p \ {0}), ∀ n ∈ Finset.range (N p), r n ≠ (a : ℚ) / p :=
sorry

end red_fraction_exists_l624_624936


namespace subset_count_of_condition_l624_624801

theorem subset_count_of_condition :
  ∃ (S : finset (finset ℕ)), 
  (∀ subset ∈ S, 
    subset.card = 5 ∧ 
    ∀ {n m ∈ subset}, n + m ≠ 11) ∧ 
  S.card = 32 :=
begin
  sorry
end

end subset_count_of_condition_l624_624801


namespace problem_1_problem_2_l624_624837

noncomputable def A := {x : ℝ | x^2 - x - 6 < 0}
noncomputable def B := {x : ℝ | x^2 + 2x - 8 ≥ 0}
noncomputable def C_R_B := {x : ℝ | -4 < x ∧ x < 2}

theorem problem_1 : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} :=
sorry

theorem problem_2 : A ∪ C_R_B = {x : ℝ | -4 < x ∧ x < 3} :=
sorry

end problem_1_problem_2_l624_624837


namespace power_of_five_in_factorial_sum_l624_624197

theorem power_of_five_in_factorial_sum : 
  (∃ k : ℕ, 5^k ∣ (150! + 151! + 152!) ∧ ¬ 5^(k+1) ∣ (150! + 151! + 152!)) ∧
  (∀ k : ℕ, 5^k ∣ (150! + 151! + 152!) → k ≤ 37) :=
sorry

end power_of_five_in_factorial_sum_l624_624197


namespace expected_value_winnings_10_sided_die_l624_624521

theorem expected_value_winnings_10_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let primes := [2, 3, 5, 7]
  let probability (x : ℕ) : ℚ := if x ∈ primes then 1 / 10 else 0
  let winnings (x : ℕ) : ℕ := if x ∈ primes then x else 0
  let expected_value : ℚ := ∑ x in primes, probability x * winnings x
  expected_value = 1.7 :=
by 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let primes := [2, 3, 5, 7]
  let probability (x : ℕ) : ℚ := if x ∈ primes then 1 / 10 else 0
  let winnings (x : ℕ) : ℕ := if x ∈ primes then x else 0
  let expected_value : ℚ := ∑ x in primes, probability x * winnings x
  have h : expected_value = (1/10 * 2) + (1/10 * 3) + (1/10 * 5) + (1/10 * 7)
    := by sorry
  have h2 : expected_value = 1.7 := by sorry
  exact h2

end expected_value_winnings_10_sided_die_l624_624521


namespace add_fractions_result_l624_624586

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624586


namespace hyperbola_and_trig_l624_624458

theorem hyperbola_and_trig (m : ℝ) :
  (∃ (x : ℝ), sin x - cos x = 2) = false ∧ (-3 < m ∧ m < 5 → ¬ ((m - 5) * (m + 3) > 0)) := 
by sorry

end hyperbola_and_trig_l624_624458


namespace sound_speed_temperature_l624_624087

theorem sound_speed_temperature (v : ℝ) (T : ℝ) (h1 : v = 0.4) (h2 : T = 15 * v^2) :
  T = 2.4 :=
by {
  sorry
}

end sound_speed_temperature_l624_624087


namespace curves_tangent_at_m_eq_two_l624_624755

-- Definitions of the ellipsoid and hyperbola equations.
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 1)^2 = 1

-- The proposition to be proved.
theorem curves_tangent_at_m_eq_two :
  ∃ m : ℝ, (∀ x y : ℝ, ellipse x y ∧ hyperbola x y m → m = 2) :=
sorry

end curves_tangent_at_m_eq_two_l624_624755


namespace fraction_addition_l624_624679

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624679


namespace convert_speed_kmh_to_ms_l624_624143

-- Define the given speed in km/h
def speed_kmh : ℝ := 1.1076923076923078

-- Define the conversion factor from km/h to m/s
def conversion_factor : ℝ := 3.6

-- State the theorem
theorem convert_speed_kmh_to_ms (s : ℝ) (h : s = speed_kmh) : (s / conversion_factor) = 0.3076923076923077 := by
  -- Skip the proof as instructed
  sorry

end convert_speed_kmh_to_ms_l624_624143


namespace six_points_plane_l624_624408

theorem six_points_plane (A B C D E F : Point) :
    (A = ⟨0, 0⟩) ∧ 
    (B = ⟨1, 0⟩) ∧ 
    (C = ⟨0, 1⟩) ∧ 
    (D = ⟨2, 0⟩) ∧ 
    (E = ⟨-1, 2⟩) ∧ 
    (F = ⟨0, -1⟩) →
    (∀ P Q : Point, (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F) →
                   (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F) →
                   P ≠ Q →
                   let line := Line.through P Q in
                   let half_planes := line.half_planes in
                   ∃ (points_in_half_plane₁ points_in_half_plane₂ : Finset Point),
                     points_in_half_plane₁.card ≠ points_in_half_plane₂.card ∧
                     points_in_half_plane₁.union points_in_half_plane₂ = {A, B, C, D, E, F} ∧
                     points_in_half_plane₁ ∈ half_planes ∧
                     points_in_half_plane₂ ∈ half_planes :=
sorry

end six_points_plane_l624_624408


namespace hawks_points_l624_624141

theorem hawks_points (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 6) : H = 38 :=
sorry

end hawks_points_l624_624141


namespace fraction_addition_l624_624677

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624677


namespace watched_videos_correct_l624_624557

-- Conditions
def num_suggestions_per_time : ℕ := 15
def times : ℕ := 5
def chosen_position : ℕ := 5

-- Question
def total_videos_watched : ℕ := num_suggestions_per_time * times - (num_suggestions_per_time - chosen_position)

-- Proof
theorem watched_videos_correct : total_videos_watched = 65 := by
  sorry

end watched_videos_correct_l624_624557


namespace sum_log_floor_l624_624745

theorem sum_log_floor :
  ∑ N in finset.range (2048 + 1), 3 * (nat.log 2 N).toNat = 55323 :=
by sorry

end sum_log_floor_l624_624745


namespace sum_of_coefficients_of_y_l624_624201

theorem sum_of_coefficients_of_y :
  let expr := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 6)
  in (31 * 1 + 15 * 1 + 28 * 1) = 74 := 
by
  sorry

end sum_of_coefficients_of_y_l624_624201


namespace fraction_addition_l624_624621

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624621


namespace sum_of_all_possible_values_of_N_with_equation_l624_624452

def satisfiesEquation (N : ℝ) : Prop :=
  N * (N - 4) = -7

theorem sum_of_all_possible_values_of_N_with_equation :
  (∀ N, satisfiesEquation N → N + (4 - N) = 4) :=
sorry

end sum_of_all_possible_values_of_N_with_equation_l624_624452


namespace axis_of_symmetry_eq_l624_624054

theorem axis_of_symmetry_eq : 
  ∃ k : ℤ, (λ x => 2 * Real.cos (2 * x)) = (λ x => 2 * Real.sin (2 * (x + π / 3) - π / 6)) ∧
            x = (1/2) * k * π ∧ x = -π / 2 := 
by
  sorry

end axis_of_symmetry_eq_l624_624054


namespace xyz_value_l624_624984

-- Define the basic conditions
variables (x y z : ℝ)
variables (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
variables (h1 : x * y = 40 * (4:ℝ)^(1/3))
variables (h2 : x * z = 56 * (4:ℝ)^(1/3))
variables (h3 : y * z = 32 * (4:ℝ)^(1/3))
variables (h4 : x + y = 18)

-- The target theorem
theorem xyz_value : x * y * z = 16 * (895:ℝ)^(1/2) :=
by
  -- Here goes the proof, but we add 'sorry' to end the theorem placeholder
  sorry

end xyz_value_l624_624984


namespace hui_book_pages_l624_624300

theorem hui_book_pages :
  ∃ x : ℕ, 
    (let y1 := x - (x / 6 + 10) in
     let y2 := y1 - (y1 / 5 + 14) in
     let y3 := y2 - (y2 / 4 + 16) in
     y3 = 72) → x = 200 :=
begin
  sorry
end

end hui_book_pages_l624_624300


namespace sum_fractions_eq_l624_624691

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624691


namespace fraction_addition_l624_624683

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624683


namespace division_results_in_integer_probability_l624_624443

def A : set ℤ := {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6}
def B : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

noncomputable def valid_pairs : set (ℤ × ℕ) :=
  {p | p.1 ∈ A ∧ p.2 ∈ B ∧ p.2 ∣ p.1}

noncomputable def total_pairs : ℕ := fintype.card (A ×ˢ B)

noncomputable def integer_division_probability : ℚ :=
  (fintype.card valid_pairs : ℚ) / (total_pairs : ℚ)

theorem division_results_in_integer_probability :
  integer_division_probability = 27 / 80 := 
begin
  sorry
end

end division_results_in_integer_probability_l624_624443


namespace exists_x_iff_sum_sin_squared_leq_two_l624_624368

open Real

theorem exists_x_iff_sum_sin_squared_leq_two 
  (θ1 θ2 θ3 θ4 : ℝ) (h1 : θ1 ∈ Ioo (-π / 2) (π / 2)) 
  (h2 : θ2 ∈ Ioo (-π / 2) (π / 2)) 
  (h3 : θ3 ∈ Ioo (-π / 2) (π / 2)) 
  (h4 : θ4 ∈ Ioo (-π / 2) (π / 2)) :
  (∃ x : ℝ, 
    cos θ1 ^ 2 * cos θ2 ^ 2 - (sin θ1 * sin θ2 - x) ^ 2 ≥ 0 ∧ 
    cos θ3 ^ 2 * cos θ4 ^ 2 - (sin θ3 * sin θ4 - x) ^ 2 ≥ 0) ↔ 
    (∑ i in Finset.range 4, sin (θ1 + i) ^ 2) ≤ 2 * 
    (1 + (∏ i in Finset.range 4, sin (θ1 + i)) + (∏ i in Finset.range 4, cos (θ1 + i))) := 
sorry

end exists_x_iff_sum_sin_squared_leq_two_l624_624368


namespace sum_difference_l624_624981

open List

def replaced_digit (n : ℕ) : ℕ :=
  let digits := toDigits 10 n
  let new_digits := digits.map (λ d, if d = 3 then 2 else d)
  ofDigits 10 new_digits

theorem sum_difference :
  let Star_sum := (list.range' 1 50).sum
  let Emilio_sum := ((list.range' 1 50).map replaced_digit).sum
  Star_sum - Emilio_sum = 105 :=
by
  let Star_sum := (list.range' 1 50).sum
  let Emilio_sum := ((list.range' 1 50).map replaced_digit).sum
  have : Star_sum - Emilio_sum = 105 := sorry
  exact this

end sum_difference_l624_624981


namespace sum_fractions_eq_l624_624694

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624694


namespace sufficient_but_not_necessary_condition_for_parallelism_l624_624387

-- Define the two lines
def line1 (x y : ℝ) (m : ℝ) : Prop := 2 * x - m * y = 1
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 1) * x - y = 1

-- Define the parallel condition for the two lines
def parallel (m : ℝ) : Prop :=
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 m ∧ line2 x2 y2 m ∧ (2 * m + 1 = 0 ∧ m^2 - m - 2 = 0)) ∨ 
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 2 ∧ line2 x2 y2 2)

theorem sufficient_but_not_necessary_condition_for_parallelism :
  ∀ m, (parallel m) ↔ (m = 2) :=
by sorry

end sufficient_but_not_necessary_condition_for_parallelism_l624_624387


namespace evaluate_expression_l624_624195

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 :=
by
  -- sorry is used to skip the proof
  sorry

end evaluate_expression_l624_624195


namespace inequality_proof_equality_condition_l624_624842

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x * (y + z - x) ^ 2 + y * (z + x - y) ^ 2 + z * (x + y - z) ^ 2 ≥
  2 * x * y * z * (x / (y + z) + y / (z + x) + z / (x + y)) :=
begin
  sorry
end

theorem equality_condition (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x = y ∧ y = z) ↔ 
  x * (y + z - x) ^ 2 + y * (z + x - y) ^ 2 + z * (x + y - z) ^ 2 = 
  2 * x * y * z * (x / (y + z) + y / (z + x) + z / (x + y)) :=
begin
  sorry
end

end inequality_proof_equality_condition_l624_624842


namespace solve_for_a_l624_624870

theorem solve_for_a (a x : ℝ) (h : x = 3) (eqn : a * x - 5 = x + 1) : a = 3 :=
by
  -- proof omitted
  sorry

end solve_for_a_l624_624870


namespace angle_relation_convex_quadrilateral_l624_624239

-- Defining the angle relation problem in Lean 4
theorem angle_relation_convex_quadrilateral
  {A B C D I_A I_B I_C I_D : Type*}
  [InCircle.inscribed_quadrilateral A B C D I_A I_B I_C I_D]
  (H1 : ∠ B I_A A + ∠ I_C I_A I_D = 180) :
  ∠ B I_B A + ∠ I_C I_B I_D = 180 := 
sorry

end angle_relation_convex_quadrilateral_l624_624239


namespace correct_proposition_l624_624252

-- Definition of proposition p
def p (x y : ℝ) : Prop := (sqrt x + sqrt y = 0) → (x = 0 ∧ y = 0)

-- Definition of proposition q
def q (x : ℝ) : Prop := (x^2 + 4 * x - 5 = 0) → (x = -5)

-- Statement to be proven
theorem correct_proposition (x y : ℝ) : (p x y ∨ q x) :=
begin
  -- We will have to show that either p (x, y) is true or q (x) is true under their definitions.
  sorry,
end

end correct_proposition_l624_624252


namespace solve_sin_equation_l624_624422

theorem solve_sin_equation
  (a1 : Real) :
  ( (Real.sin (2 * a1) - Real.pi * Real.sin a1) * Real.sqrt (11 * a1^2 - a1^4 - 10) = 0 ) →
  ( a1 ∈ {-Real.sqrt 10, -Real.pi, -1, 1, Real.pi, Real.sqrt 10} ) :=
by sorry

end solve_sin_equation_l624_624422


namespace add_fractions_l624_624665

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624665


namespace weight_of_new_student_l624_624501

theorem weight_of_new_student (W : ℝ) (x : ℝ) (h1 : 5 * W - 92 + x = 5 * (W - 4)) : x = 72 :=
sorry

end weight_of_new_student_l624_624501


namespace chromatic_number_no_n_clique_l624_624415

theorem chromatic_number_no_n_clique (n : ℕ) (h : n > 3) :
  ∃ (G : SimpleGraph), G.chromatic_number = n ∧ ¬G.contains_clique n :=
sorry

end chromatic_number_no_n_clique_l624_624415


namespace train_speed_equals_36_0036_l624_624531

noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_equals_36_0036 :
  train_speed 70 6.999440044796416 = 36.0036 :=
by
  unfold train_speed
  sorry

end train_speed_equals_36_0036_l624_624531


namespace fraction_addition_l624_624647

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624647


namespace commodity_Y_increase_l624_624451

theorem commodity_Y_increase (y : ℕ) :
  let price_X_2001 := 420 in
  let increase_X_per_year := 30 in
  let price_Y_2001 := 440 in
  let year_start := 2001 in
  let year_end := 2010 in
  let gap := year_end - year_start in
  let increase_X_2010 := increase_X_per_year * gap in
  let price_X_2010 := price_X_2001 + increase_X_2010 in
  let price_diff_2010 := 70 in
  let price_Y_2010 := price_X_2010 - price_diff_2010 in
  let total_increase_Y := price_Y_2010 - price_Y_2001 in
  total_increase_Y = y * gap →
  y = 20 :=
by sorry

end commodity_Y_increase_l624_624451


namespace add_fractions_l624_624566

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624566


namespace sum_fractions_eq_l624_624695

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624695


namespace max_heaps_660_l624_624003

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624003


namespace baker_gain_percentage_l624_624511

variable (cost_milk cost_flour cost_sugar selling_price total_cost cost_per_cake gain_per_cake : ℝ)

def percentage_gain (gain_per_cake cost_per_cake : ℝ) : ℝ :=
  (gain_per_cake / cost_per_cake) * 100

theorem baker_gain_percentage :
  cost_milk = 12 →
  cost_flour = 8 →
  cost_sugar = 10 →
  selling_price = 60 →
  total_cost = cost_milk + cost_flour + cost_sugar →
  cost_per_cake = total_cost / 5 →
  gain_per_cake = selling_price - cost_per_cake →
  percentage_gain gain_per_cake cost_per_cake = 900 :=
by
  intros h_milk h_flour h_sugar h_selling h_total h_cost_per_cake h_gain_per_cake
  rw [h_milk, h_flour, h_sugar, h_selling, h_total, h_cost_per_cake, h_gain_per_cake]
  norm_num
  sorry

end baker_gain_percentage_l624_624511


namespace sin_of_tan_l624_624881

theorem sin_of_tan (A : ℝ) (hA_acute : 0 < A ∧ A < π / 2) (h_tan_A : Real.tan A = (Real.sqrt 2) / 3) :
  Real.sin A = (Real.sqrt 22) / 11 :=
sorry

end sin_of_tan_l624_624881


namespace complex_w_sixth_power_l624_624916

theorem complex_w_sixth_power :
  let w := Complex.ofReal (-1) / 2 + Complex.I * Real.sqrt 3 / 2 in
  w ^ 6 = 1 / 4 :=
by
  sorry

end complex_w_sixth_power_l624_624916


namespace sqrt_meaningful_condition_l624_624468

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end sqrt_meaningful_condition_l624_624468


namespace rate_of_barbed_wire_is_correct_l624_624429

noncomputable def rate_of_drawing_barbed_wire 
(area : ℕ) 
(perimeter_sub_gates : ℕ) 
(total_cost : ℕ) : ℕ :=
  total_cost / perimeter_sub_gates

theorem rate_of_barbed_wire_is_correct 
  (area : ℕ) 
  (perimeter_sub_gates : ℕ) 
  (total_cost : ℕ) 
  (side_length : ℕ := Nat.sqrt area) 
  (perimeter := 4 * side_length) 
  (gates_width : ℕ := 2) 
  (expected_rate : ℚ := 10.5) :
  area = 3136 ∧ perimeter_sub_gates = (perimeter - gates_width) ∧ total_cost = 2331 →
  rate_of_drawing_barbed_wire area perimeter_sub_gates total_cost = 10.5 :=
by
  sorry

end rate_of_barbed_wire_is_correct_l624_624429


namespace sqrt_pos_condition_l624_624467

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end sqrt_pos_condition_l624_624467


namespace trigonometric_identity_l624_624804

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

end trigonometric_identity_l624_624804


namespace simplify_radical_expression_l624_624174

noncomputable def simpl_radical_form (q : ℝ) : ℝ :=
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3)

theorem simplify_radical_expression (q : ℝ) :
  simpl_radical_form q = 3 * q^3 * Real.sqrt 10 :=
by
  sorry

end simplify_radical_expression_l624_624174


namespace maximum_ab_ac_bc_l624_624369

theorem maximum_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 5) : 
  ab + ac + bc ≤ 25 / 6 :=
sorry

end maximum_ab_ac_bc_l624_624369


namespace remainder_division_of_power_divisor_l624_624920

theorem remainder_division_of_power_divisor :
  let n := 3 in
  ∃ k : ℕ, (12 ^ 2015 + 13 ^ 2015) = 5 ^ n * k ∧ k % 1000 = 625 :=
by
  sorry

end remainder_division_of_power_divisor_l624_624920


namespace gcd_Sn_S3n_l624_624236

def Sn (n : ℕ) : ℚ := (n^4 * (n+1)^4) / 8
def S3n (n : ℕ) : ℚ := (n^4 * 3^4 * (3*n+1)^4) / 8

theorem gcd_Sn_S3n (n : ℕ) : 
  Nat.gcd (Sn n).numerator (S3n n).numerator = 
  if n % 2 = 0 then 
    (n^4) 
  else 
    (n^4 * 81) := sorry

end gcd_Sn_S3n_l624_624236


namespace diagonals_intersect_at_same_angles_l624_624966

theorem diagonals_intersect_at_same_angles
  {A B C D E F : Type}
  (no_parallel_sides_Petya : ∀ {P Q : Type}, ¬(P = Q))
  (no_parallel_sides_Vasya : ∀ {P Q : Type}, ¬(P = Q))
  (angles_Petya : set (ℝ))
  (angles_Vasya : set (ℝ))
  (Petya_has_angles : angles_Petya = {α , α, β, γ})
  (Vasya_has_angles : angles_Vasya = {α , α, β, γ}) :
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ :=
by
  sorry

end diagonals_intersect_at_same_angles_l624_624966


namespace solution_1_solution_2_l624_624747

noncomputable def problem_1 : (ℝ) :=
  sqrt 3 * 612 * (3 * 3 / 2)

theorem solution_1 : problem_1 = 3 :=
by
  sorry

noncomputable def problem_2 : (ℝ) :=
  (log 5)^2 - (log 2)^2 + log 4

theorem solution_2 : problem_2 = 1 :=
by
  sorry

end solution_1_solution_2_l624_624747


namespace count_partition_pairs_l624_624365

-- Define the conditions
def isPartition (A B : Set ℕ) : Prop :=
  A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧
  A ∩ B = ∅ ∧
  (∀ n ∈ A, n ≠ A.card) ∧
  (∀ n ∈ B, n ≠ B.card)

-- Define the problem statement
theorem count_partition_pairs : 
  (∃ (M : ℕ), M = 6476 ∧ 
  ∀ (A B : Set ℕ), isPartition A B → A.nonempty ∧ B.nonempty) := 
  sorry

end count_partition_pairs_l624_624365


namespace add_fractions_result_l624_624591

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624591


namespace solve_problem_l624_624460

noncomputable def problem_statement : Prop :=
  ∀ (T0 Ta T t1 T1 h t2 T2 : ℝ),
    T0 = 88 ∧ Ta = 24 ∧ T1 = 40 ∧ t1 = 20 ∧
    T1 - Ta = (T0 - Ta) * ((1/2)^(t1/h)) ∧
    T2 = 32 ∧ T2 - Ta = (T1 - Ta) * ((1/2)^(t2/h)) →
    t2 = 10

theorem solve_problem : problem_statement := sorry

end solve_problem_l624_624460


namespace add_fractions_l624_624661

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624661


namespace fraction_addition_l624_624673

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624673


namespace liars_guessing_game_l624_624986

/-- The liar's guessing game is a game played between two players A and B. 
The rules of the game depend on two positive integers k and n. At the start 
of the game A chooses integers x and N with 1 ≤ x ≤ N. Player A keeps x 
secret, and truthfully tells N to player B. Player B now tries to obtain 
information about x by asking player A questions as follows: each question 
consists of B specifying an arbitrary set S of positive integers (possibly one
specified in some previous question), and asking A whether x belongs to S. 
Player B may ask as many questions as he wishes. After each question, 
player A must immediately answer it with 'yes' or 'no', but is allowed to lie 
as many times as she wants; the only restriction is that, among any k + 1 
consecutive answers, at least one answer must be truthful.

After B has asked as many questions as he wants, he must specify a set 
X of at most n positive integers. If x belongs to X, then B wins; otherwise, he loses.

We need to prove that:
1. If n ≥ 2^k, then B can guarantee a win.
2. For all sufficiently large k, there exists an integer n ≥ (1.99)^k such that B 
cannot guarantee a win.
-/
theorem liars_guessing_game (k n: ℕ) (N x : ℕ) (S : set ℕ) : 
  (n ≥ 2^k) → (B_wins k n N x S) 
  ∧ (∃ k', ∀ k ≥ k', ∃ n', n' ≥ (1.99)^k ∧ ¬ B_wins k n' N x S) := 
sorry

end liars_guessing_game_l624_624986


namespace comic_books_total_l624_624887

theorem comic_books_total (C : ℕ)
  (h1 : 0.30 * C = 0.30 * C)
  (h2 : 0.70 * C = 120) :
  C = 172 :=
by
  sorry

end comic_books_total_l624_624887


namespace maximum_value_of_expression_l624_624372

-- Define the given condition
def condition (a b c : ℝ) : Prop := a + 3 * b + c = 5

-- Define the objective function
def objective (a b c : ℝ) : ℝ := a * b + a * c + b * c

-- Main theorem statement
theorem maximum_value_of_expression (a b c : ℝ) (h : condition a b c) : 
  ∃ (a b c : ℝ), condition a b c ∧ objective a b c = 25 / 3 :=
sorry

end maximum_value_of_expression_l624_624372


namespace collinear_M_a_M_b_M_c_and_parallel_lines_l624_624909

noncomputable section

variables {A B C A₀ B₀ A₁ B₁ Mₐ M_b M_c : Point}
variables (triangle ABC : Triangle)
variables (AA₀ BB₀ : Median ABC)
variables (AA₁ BB₁ : Altitude ABC)
variables (circumcircle_CA₀B₀ circumcircle_CA₁B₁ : Circle)
variables (intersect_Mc : circumcircle_CA₀B₀ ∩ circumcircle_CA₁B₁ = {M_c})
variables (M_a := define_similarly A)
variables (M_b := define_similarly B)

-- Define the parallel lines
variables (line_AM_a : Line)
variables (line_BM_b : Line)
variables (line_CM_c : Line)
variables (line_OH : Line) -- Line connecting orthocenter and circumcenter

-- Assumptions
variables (line_AM_a_parallel_line_OH : Parallel line_AM_a line_OH)
variables (line_BM_b_parallel_line_OH : Parallel line_BM_b line_OH)
variables (line_CM_c_parallel_line_OH : Parallel line_CM_c line_OH)

-- Theorem statement

theorem collinear_M_a_M_b_M_c_and_parallel_lines :
  collinear {Mₐ, M_b, M_c} ∧
  Parallel line_AM_a line_OH ∧
  Parallel line_BM_b line_OH ∧
  Parallel line_CM_c line_OH :=
sorrey

end collinear_M_a_M_b_M_c_and_parallel_lines_l624_624909


namespace min_circle_distance_to_line_l624_624238

theorem min_circle_distance_to_line :
  ∃ (a b r : ℝ), 
    (2 * sqrt(r^2 - a^2) = 2) ∧ 
    (r^2 = 2 * b^2) ∧ 
    ((r^2 = 2 * b^2) → (2 * b^2 - a^2 = 1)) ∧ 
    ((a - 2 * b) / sqrt(5) = d → d ≥ 0) ∧
    ((a = 1 ∧ b = 1 ∧ r = sqrt(2)) ∨ (a = -1 ∧ b = -1 ∧ r = sqrt(2))) → 
    ((x - a)^2 + (y - b)^2 = r^2) :=
by
  sorry

end min_circle_distance_to_line_l624_624238


namespace trig_expression_equals_neg_sqrt_three_l624_624199

noncomputable def evaluateTrigExpression : ℝ :=
  4 * real.cos (real.pi / 18) - (real.cos (real.pi / 18) / real.sin (real.pi / 18))

theorem trig_expression_equals_neg_sqrt_three : evaluateTrigExpression = -real.sqrt 3 :=
by
  sorry

end trig_expression_equals_neg_sqrt_three_l624_624199


namespace total_wheels_in_garage_l624_624099

theorem total_wheels_in_garage : 
  let cars_wheels := 2 * 4 in
  let lawnmower_wheels := 4 in
  let bicycles_wheels := 3 * 2 in
  let tricycle_wheels := 3 in
  let unicycle_wheels := 1 in
  let skateboard_wheels := 4 in
  let wheelbarrow_wheels := 1 in
  let wagon_wheels := 4 in
  cars_wheels + lawnmower_wheels + bicycles_wheels + tricycle_wheels + unicycle_wheels + skateboard_wheels + wheelbarrow_wheels + wagon_wheels = 31 :=
by 
  let cars_wheels := 2 * 4
  let lawnmower_wheels := 4
  let bicycles_wheels := 3 * 2
  let tricycle_wheels := 3
  let unicycle_wheels := 1
  let skateboard_wheels := 4
  let wheelbarrow_wheels := 1
  let wagon_wheels := 4
  have h : cars_wheels + lawnmower_wheels + bicycles_wheels + tricycle_wheels + unicycle_wheels + skateboard_wheels + wheelbarrow_wheels + wagon_wheels = 31 := by sorry
  exact h

end total_wheels_in_garage_l624_624099


namespace sum_fractions_eq_l624_624699

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624699


namespace volleyball_boys_pine_l624_624358

def total_students := 120
def boys := 70
def girls := 50
def students_maple := 50
def students_pine := 40
def students_oak := 30
def boys_maple := 30
def volleyball_girls := 15

theorem volleyball_boys_pine : 
  -- Given conditions
  total_students = 120 ∧ 
  boys = 70 ∧ 
  girls = 50 ∧ 
  students_maple = 50 ∧ 
  students_pine = 40 ∧ 
  students_oak = 30 ∧ 
  boys_maple = 30 ∧ 
  volleyball_girls = 15 → 
  -- Show the number of boys attending the volleyball workshop from Pine is 12.
  ∃ (volleyball_boys_pine : ℕ), volleyball_boys_pine = 12 := 
by
  intros h
  use 12
  sorry

end volleyball_boys_pine_l624_624358


namespace circle_C1_eq_circle_C2_eq_l624_624896

-- Define points as structures
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given points
def M : Point := {x := -1, y := 1}
def N : Point := {x := 0, y := 2}
def Q : Point := {x := 2, y := 0}

-- Definitions of equations of circles
def circle_eq (h k r : ℝ) : (Point → Prop) := λ P, (P.x - h)^2 + (P.y - k)^2 = r

-- Circle C1 passing through points M, N, and Q
def C1 : Point → Prop := circle_eq (1/2) (1/2) (5/2)

-- Reflect a point across the line MN
def reflect (P : Point) (M N : Point) : Point := 
  let mn_line_eq := x - y + 2 = 0 -- Line MN equation 
  let d := / ((sq M.x - N.x) + (sq M.y - N.y)) 
  {x := (P.x - d * (sq N.x - P.x + sq N.y - P.y)) / dec + 1. 
  y := (P.y - d * (sq N.x - P.x + sq N.y - P.y))/dec + 1  }

-- Center of C2 is reflection of center of C1
def C2_center : Point := reflect {x := 1/2, y := 1/2} M N

-- Circle C2
def C2 : Point → Prop := circle_eq (-3/2) (5/2) (5/2)

-- Proof statements
theorem circle_C1_eq : ∀ (P : Point), (P = M ∨ P = N ∨ P = Q) → C1 P :=
sorry

theorem circle_C2_eq : ∀ (P : Point), C2 (reflect P M N) = C1 P :=
sorry

end circle_C1_eq_circle_C2_eq_l624_624896


namespace fully_simplify_expression_l624_624229

theorem fully_simplify_expression :
  (3 + 4 + 5 + 6) / 2 + (3 * 6 + 9) / 3 = 18 :=
by
  sorry

end fully_simplify_expression_l624_624229


namespace non_parallel_edges_l624_624146

-- Define the dimensions of the cuboid
def a : ℕ := 8
def b : ℕ := 6
def c : ℕ := 4

-- Define the total number of edges in a cuboid
def total_edges : ℕ := 12

-- Define the edges groups
def edges_group (dim : ℕ) : ℕ := 4

-- Prove that the number of edges not parallel to a given edge is 8
theorem non_parallel_edges (d : ℕ) (h_d : d = a ∨ d = b ∨ d = c) : 
  edges_group b + edges_group c = 8 :=
by
  unfold edges_group
  simp [b, c]
  exact rfl 

end non_parallel_edges_l624_624146


namespace presidency_meeting_ways_l624_624517

theorem presidency_meeting_ways :
  let schools := 4
  let members_per_school := 5
  let host_reps := 3
  let other_reps := 2
  let ways_choose_host := Nat.choose schools 1
  let ways_choose_host_reps := Nat.choose members_per_school host_reps
  let ways_choose_other_reps := Nat.choose members_per_school other_reps
  ways_choose_host * ways_choose_host_reps * ways_choose_other_reps^ (schools - 1) = 40000 :=
by
  sorry

end presidency_meeting_ways_l624_624517


namespace inequality_proof_l624_624794

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / Real.sqrt (a^2 + 8 * b * c)) + 
  (b / Real.sqrt (b^2 + 8 * c * a)) + 
  (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proof_l624_624794


namespace range_of_a_l624_624267

-- Define the conditions
def is_arithmetic_sequence (seq : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = d

def a_n_is_increasing (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_n n < a_n (n + 1)

-- The arithmetic sequence log_a b_n
def log_a_b_n (a : ℝ) (b_n : ℕ → ℝ) : ℕ → ℝ :=
  λ n, Real.log (b_n n) / Real.log a

-- Define the sequences a_n and b_n
def a_n (a : ℝ) (b_n : ℕ → ℝ) : ℕ → ℝ :=
  λ n, b_n n * Real.log a / Real.log (b_n n)

def b_n (a : ℝ) : ℕ → ℝ := λ n, a ^ (n + 1)

-- Main theorem
theorem range_of_a :
  ∀ (a : ℝ), (a > 0) → (a ≠ 1) →
  is_arithmetic_sequence (log_a_b_n a (b_n a)) 1 →
  a_n_is_increasing (a_n a (b_n a)) →
  a ∈ Set.Ioo 0 (2/3) ∪ Set.Ioi 1 :=
by
  intros a ha hne hseq hincr
  sorry

end range_of_a_l624_624267


namespace fraction_addition_l624_624632

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624632


namespace watched_videos_correct_l624_624558

-- Conditions
def num_suggestions_per_time : ℕ := 15
def times : ℕ := 5
def chosen_position : ℕ := 5

-- Question
def total_videos_watched : ℕ := num_suggestions_per_time * times - (num_suggestions_per_time - chosen_position)

-- Proof
theorem watched_videos_correct : total_videos_watched = 65 := by
  sorry

end watched_videos_correct_l624_624558


namespace exercise_tenth_day_l624_624294

-- Declare conditions
variables (h1 h2 : ℕ) (d1 d2 total_days : ℕ) (target_avg : ℕ)

-- Define the given conditions
def h1 := 120 -- 2 hours in minutes
def d1 := 4
def h2 := 105 -- 1 hour 45 minutes in minutes
def d2 := 5
def total_days := 10
def target_avg := 110 -- Target average in minutes

-- Calculate the current total minutes exercised in the first 9 days
def total_minutes_first_9_days := h1 * d1 + h2 * d2

-- Calculate the target total minutes over 10 days to achieve the target average
def target_total_minutes := target_avg * total_days

-- Calculate the required exercise time on the tenth day
def ex_tenth := target_total_minutes - total_minutes_first_9_days

theorem exercise_tenth_day : ex_tenth = 95 := 
    by
    simp [ex_tenth, target_total_minutes, total_minutes_first_9_days]
    sorry

end exercise_tenth_day_l624_624294


namespace theta_of_complex_1_i_sqrt3_eq_pi_div3_l624_624485

noncomputable def theta_of_complex (re im : ℝ) : ℝ :=
  complex.arg (complex.mk re im)

theorem theta_of_complex_1_i_sqrt3_eq_pi_div3 :
  theta_of_complex 1 (real.sqrt 3) = real.pi / 3 := 
sorry

end theta_of_complex_1_i_sqrt3_eq_pi_div3_l624_624485


namespace basketball_game_points_l624_624321

theorem basketball_game_points
  (a b : ℕ) 
  (r : ℕ := 2)
  (S_E : ℕ := a / 2 * (1 + r + r^2 + r^3))
  (S_T : ℕ := 4 * b)
  (h1 : S_E = S_T + 2)
  (h2 : S_E < 100)
  (h3 : S_T < 100)
  : (a / 2 + a / 2 * r + b + b = 19) :=
by sorry

end basketball_game_points_l624_624321


namespace add_fractions_result_l624_624580

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624580


namespace primes_satisfy_eq_l624_624779

theorem primes_satisfy_eq (p q : ℕ) (hp : p.prime) (hq : q.prime) : 
  p + q = (p - q)^3 → (p = 5 ∧ q = 3) ∨ (p = 3 ∧ q = 5) :=
by
  sorry

end primes_satisfy_eq_l624_624779


namespace parabola_equation_and_focus_line_passing_through_focus_l624_624853

-- Definitions based on conditions
def parabola (p : ℝ) := ∀ x y, y^2 = 2 * p * x
def point_A_on_C (p a : ℝ) := ∃ y, (2, a) = (2, y) ∧ y^2 = 2 * p * 2
def distance_AF (p : ℝ) := dist (2, a) (p / 2, 0) = 3

-- Part 1: Proving the equation and focus of the parabola
theorem parabola_equation_and_focus (p a : ℝ) (h₀ : parabola p) (h₁ : point_A_on_C p a) (h₂ : distance_AF p) : 
  (∀ x y, y^2 = 4 * x ∧ parabola 2) ∧ (focus = (1,0)) := sorry

-- Part 2: Proving the equation of the line l
theorem line_passing_through_focus (p : ℝ) (F_point : (ℝ × ℝ)) (m : ℝ) (B : ℝ × ℝ := (-1, 1)) 
(l : ∀ y, x = m * y + 1) (M N : ℝ × ℝ) (h₀ : parabola 2) (h₁ : ∃ x1 x2 m1 m2: ℝ, 
  y₁ + y₂ = 4 * m ∧ y₁ * y₂ = -4 ∧
  (dist B M) * (dist B N) = 0 ∧
  F = (1, 0)) : 
  l = (2 * x - y - 2) := sorry

end parabola_equation_and_focus_line_passing_through_focus_l624_624853


namespace add_fractions_l624_624739

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624739


namespace rotated_P_l624_624342

noncomputable def O : (ℝ × ℝ) := (0, 0)
noncomputable def Q : (ℝ × ℝ) := (6, 0)
def P_in_first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0
def angle_PQO_90 (P : ℝ × ℝ) : Prop := ∃ θ : ℝ, θ = 90 ∧ (P, Q, O).2 = θ
def angle_POQ_45 (P : ℝ × ℝ) : Prop := ∃ θ : ℝ, θ = 45 ∧ (P, O, Q).2 = θ

def cond_P (P : ℝ × ℝ) : Prop :=
  P_in_first_quadrant P ∧ angle_PQO_90 P ∧ angle_POQ_45 P

theorem rotated_P (P : ℝ × ℝ) (h : cond_P P) : 
  rotate_90_ccw O P = (-6, 6) :=
sorry

end rotated_P_l624_624342


namespace exists_half_l624_624382

theorem exists_half {n : ℕ} (a : Fin n → ℝ)
  (h_pos : ∀ i, 0 < a i) (h_lt_one : ∀ i, a i < 1)
  (h_sum : ∑ S in Finset.powersetLen ((Finset.range n).card-1) (Finset.range n),
           ∏ i in S, a i * ∏ j in (Finset.range n).erase i, (1 - a j) = 1/2) :
  ∃ k, a k = 1/2 :=
sorry

end exists_half_l624_624382


namespace value_b8_l624_624928

def sequence (b : ℕ → ℕ) : Prop :=
  b 1 = 2 ∧ (∀ m n, b (m + n) = 2 * b m + 2 * b n + 2 * m * n)

theorem value_b8 (b : ℕ → ℕ) (h_seq : sequence b) : b 8 = 224 :=
by
  sorry

end value_b8_l624_624928


namespace add_fractions_l624_624718

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624718


namespace angle_bisector_divides_CD_l624_624890

variable (A B C D Q : Point)
variable (AB AD BC CD : ℝ)

-- Definitions of the conditions provided
def isRightTrapezoid (A B C D : Point) : Prop :=
  isRightAngle A B C ∧ isBase AD ∧ isBase BC ∧ isLeg AB ∧ isLeg CD

def bisectsAngle (A B C Q : Point) : Prop :=
  isAngleBisector B Q A C

def bisectsSideInEqualRatio (C D Q : Point) : Prop :=
  segmentLength C Q = segmentLength Q D

theorem angle_bisector_divides_CD
  (h1 : isRightTrapezoid A B C D)
  (h2 : segmentLength AB = segmentLength AD + segmentLength BC)
  (h3 : bisectsAngle A B C Q) :
  bisectsSideInEqualRatio C D Q :=
sorry -- Proof omitted

end angle_bisector_divides_CD_l624_624890


namespace max_heaps_660_l624_624006

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624006


namespace longest_side_of_triangle_l624_624157

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

noncomputable def longest_side_length :=
  max (distance (3, 3) (8, 3)) (max (distance (3, 3) (7, 7)) (distance (7, 7) (8, 3)))

theorem longest_side_of_triangle :
  let p1 := (3 : ℝ, 3 : ℝ)
  let p2 := (7 : ℝ, 7 : ℝ)
  let p3 := (8 : ℝ, 3 : ℝ)
  longest_side_length = 4 * Real.sqrt 2 :=
by sorry

end longest_side_of_triangle_l624_624157


namespace find_least_nonneg_integer_l624_624782

theorem find_least_nonneg_integer (n : ℕ) (k : ℕ) :
  (∃ k : ℕ, (n^k) % (10^2012) = 10^2012 - 1) ↔ n = 71 :=
begin
  sorry
end

end find_least_nonneg_integer_l624_624782


namespace sum_of_fractions_l624_624711

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624711


namespace fraction_addition_l624_624646

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624646


namespace no_real_roots_abs_eq_l624_624071

theorem no_real_roots_abs_eq (x : ℝ) : 
  |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 → false :=
by sorry

end no_real_roots_abs_eq_l624_624071


namespace degree_of_monomial_l624_624995

theorem degree_of_monomial: 
  let a : ℚ := -2/5
  let f : ℚ[x, y] := a * (mv_polynomial.X x)^2 * (mv_polynomial.X y) 
  (mv_polynomial.total_degree f) = 3 :=
sorry

end degree_of_monomial_l624_624995


namespace solve_for_s_l624_624188

def F (x y z : ℝ) : ℝ := x * y^z

theorem solve_for_s :
  ∃ s > 0, F s s 2 = 1024 ∧ s = 8 * (2 ^ (1/3)) :=
sorry

end solve_for_s_l624_624188


namespace find_angle_B_and_side_c_l624_624320

theorem find_angle_B_and_side_c 
  (a b : ℝ) (A B C : ℝ) 
  (h₁ : a = 3 * real.sqrt 2) 
  (h₂ : b = 3 * real.sqrt 3) 
  (h₃ : A = 45) 
  (h_triangle : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * real.cos (A * real.pi / 180)) : 
  B = 60 ∧ c = (3 * real.sqrt 6 + 3 * real.sqrt 2) / 2 := 
sorry

end find_angle_B_and_side_c_l624_624320


namespace garden_area_proof_l624_624349

noncomputable def garden_area (posts : ℕ) (spacing : ℕ) (long_side_factor : ℕ) : ℕ := 
  let shorter_posts := (posts + 4) / 8
  let longer_posts := long_side_factor * shorter_posts
  let short_side_length := (shorter_posts - 1) * spacing
  let long_side_length := (longer_posts - 1) * spacing
  short_side_length * long_side_length

theorem garden_area_proof : garden_area 24 3 3 = 144 := by
  simp [garden_area]
  sorry

end garden_area_proof_l624_624349


namespace find_a_minus_b_l624_624303

theorem find_a_minus_b (x : ℚ) (a b : ℚ) (hpos : x > 0) :
    (a / (2^x - 1) + b / (2^x + 3) = (2 * 2^x + 5) / ((2^x - 1) * (2^x + 3))) →
    a - b = 3 / 2 :=
    sorry

end find_a_minus_b_l624_624303


namespace max_heaps_of_stones_l624_624010

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624010


namespace customer_purchases_90_percent_l624_624089

variable (P Q : ℝ) 

theorem customer_purchases_90_percent (price_increase_expenditure_diff : 
  (1.25 * P * R / 100 * Q = 1.125 * P * Q)) : 
  R = 90 := 
by 
  sorry

end customer_purchases_90_percent_l624_624089


namespace add_fractions_l624_624724

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624724


namespace max_free_squares_l624_624400

theorem max_free_squares (n : ℕ) :
  ∀ (initial_positions : ℕ), 
    (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → initial_positions = 2) →
    (∀ (i j : ℕ) (move1 move2 : ℕ × ℕ),
       1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n →
       move1 = (i + 1, j) ∨ move1 = (i - 1, j) ∨ move1 = (i, j + 1) ∨ move1 = (i, j - 1) →
       move2 = (i + 1, j) ∨ move2 = (i - 1, j) ∨ move2 = (i, j + 1) ∨ move2 = (i, j - 1) →
       move1 ≠ move2) →
    ∃ free_squares : ℕ, free_squares = n^2 :=
by
  sorry

end max_free_squares_l624_624400


namespace total_people_at_beach_l624_624958

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end total_people_at_beach_l624_624958


namespace prob1_l624_624506

theorem prob1 (a : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0) → (∀ x y : ℝ, x - y + a = 0) → 
  (∀ (x y : ℝ), abs (1 - 2 + a) / sqrt (1^2 + (-1)^2) = sqrt 2 / 2) → (a = 0 ∨ a = 2) := 
by sorry

end prob1_l624_624506


namespace fraction_addition_l624_624676

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624676


namespace balloons_total_l624_624065

theorem balloons_total (a b : ℕ) (h1 : a = 47) (h2 : b = 13) : a + b = 60 := 
by
  -- Since h1 and h2 provide values for a and b respectively,
  -- the result can be proved using these values.
  sorry

end balloons_total_l624_624065


namespace factorize_x2_minus_9_l624_624774

theorem factorize_x2_minus_9 (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
sorry

end factorize_x2_minus_9_l624_624774


namespace add_fractions_l624_624735

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624735


namespace sum_of_squares_increased_l624_624821

theorem sum_of_squares_increased (x : Fin 100 → ℝ) 
  (h : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  ∑ i, (x i + 4) ^ 2 = ∑ i, x i ^ 2 + 800 := 
by
  sorry

end sum_of_squares_increased_l624_624821


namespace add_fractions_result_l624_624583

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624583


namespace sum_of_distances_independent_of_X_sum_unit_normals_eq_zero_iff_distances_independent_l624_624971

noncomputable def unit_outward_normals (i : ℕ) : ℝ³ := sorry
def internal_point_of_polyhedron : ℝ³ := sorry
def arbitrary_points_on_faces (i : ℕ) : ℝ³ := sorry
def distance_from_point_to_plane (X : ℝ³) (i : ℕ) : ℝ := (X - arbitrary_points_on_faces i) • unit_outward_normals i

theorem sum_of_distances_independent_of_X
  (sum_unit_normals_eq_zero : ∑ i in finset.range k, unit_outward_normals i = 0) :
  ∀ (X : ℝ³) (Y : ℝ³),
    (∑ i in finset.range k, distance_from_point_to_plane X i) = 
    (∑ i in finset.range k, distance_from_point_to_plane Y i) :=
sorry

theorem sum_unit_normals_eq_zero_iff_distances_independent :
  (∀ (X Y : ℝ³), (∑ i in finset.range k, distance_from_point_to_plane X i) = 
    (∑ i in finset.range k, distance_from_point_to_plane Y i)) ↔
  (∑ i in finset.range k, unit_outward_normals i = 0) :=
sorry

end sum_of_distances_independent_of_X_sum_unit_normals_eq_zero_iff_distances_independent_l624_624971


namespace total_cans_mike_collected_l624_624035

-- Given Conditions
def cans_collected_monday : ℕ := 450
def multiplier : ℝ := 1.5
def cans_collected_tuesday : ℕ := (multiplier * cans_collected_monday).to_nat

-- Total Cans Collected
def total_cans_collected : ℕ := cans_collected_monday + cans_collected_tuesday

-- Theorem Statement
theorem total_cans_mike_collected : total_cans_collected = 1125 := by
  -- Proof would go here
  sorry

end total_cans_mike_collected_l624_624035


namespace eccentricity_value_exists_fixed_point_C_l624_624271

-- Definition of the ellipse:
def ellipse (a b : ℝ) (h1 : a > b) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Definition of semi-focal distance and eccentricity:
def eccentricity (a c : ℝ) : ℝ :=
  c / a

noncomputable def eccentricity_of_ellipse (a b c : ℝ) (h1 : a = 2 * b) (h2 : a^2 = b^2 + c^2) : ℝ :=
  c / a

theorem eccentricity_value {a b c : ℝ} (h1 : a > b) (h2 : a = 2 * b) (h3 : a^2 = b^2 + c^2) (h4 : 0 < c) :
  eccentricity_of_ellipse a b c h2 h3 = (Real.sqrt 2) / 2 :=
sorry

-- For the existence of point C on the x-axis:
def fixed_point_C (a b c : ℝ) (h1 : a > b) (h2 : c = 3) : ℝ × ℝ :=
  (29 / 8, 0)

theorem exists_fixed_point_C {a b : ℝ} (h1 : a > b) (h2 : c = 3) :
  ∃ C : ℝ × ℝ, C = fixed_point_C a b c :=
⟨(29 / 8, 0), rfl⟩

end eccentricity_value_exists_fixed_point_C_l624_624271


namespace financed_amount_correct_l624_624861

-- Define the conditions
def monthly_payment : ℝ := 150.0
def years : ℝ := 5.0
def months_in_a_year : ℝ := 12.0

-- Define the total number of months
def total_months : ℝ := years * months_in_a_year

-- Define the amount financed
def total_financed : ℝ := monthly_payment * total_months

-- State the theorem
theorem financed_amount_correct :
  total_financed = 9000 :=
by
  -- Provide the proof here
  sorry

end financed_amount_correct_l624_624861


namespace max_heaps_660_l624_624001

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624001


namespace cartoon_length_l624_624950

theorem cartoon_length (reality_show_count : ℕ) (reality_show_time : ℕ) (total_tv_time : ℕ) 
                       (h1 : reality_show_count = 5) (h2 : reality_show_time = 28) 
                       (h3 : total_tv_time = 150) :
  let total_reality_shows_time := reality_show_count * reality_show_time in
  total_tv_time - total_reality_shows_time = 10 := 
by
  sorry

end cartoon_length_l624_624950


namespace extremum_values_l624_624442

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x

theorem extremum_values :
  (∀ x, f x ≤ 5) ∧ f (-1) = 5 ∧ (∀ x, f x ≥ -27) ∧ f 3 = -27 :=
by
  sorry

end extremum_values_l624_624442


namespace shaded_triangle_probability_l624_624322

variable (A B C D E F : Type)
variable (triangles : Finset (Finset Type))

/- We define the six triangles explicitly -/
def T_ABE : Finset Type := {A, B, E}
def T_ABF : Finset Type := {A, B, F}
def T_AEF : Finset Type := {A, E, F}
def T_BCF : Finset Type := {B, C, F}
def T_BDF : Finset Type := {B, D, F}
def T_CDF : Finset Type := {C, D, F}

/- We define the set of all triangles -/
def all_triangles : Finset (Finset Type) := {T_ABE, T_ABF, T_AEF, T_BCF, T_BDF, T_CDF}

/- Define what it means for a triangle to be shaded -/
def is_shaded (triangle : Finset Type) : Bool :=
  triangle = T_CDF

/- The main theorem: probability that a selected triangle has all or part of its interior shaded -/
theorem shaded_triangle_probability : 
  @Finset.card (Finset Type) all_triangles = 6 ∧
  @Finset.card (Finset, {t ∈ all_triangles | is_shaded t}) = 1 →
  (1 : ℚ) / (6 : ℚ) = 1 / 6 :=
by sorry

end shaded_triangle_probability_l624_624322


namespace total_pieces_seven_rows_l624_624192

def rods_in_row (n : ℕ) : ℕ :=
  3 * n

def total_rods (rows : ℕ) : ℕ :=
  3 * (rows * (rows + 1)) / 2

def total_connectors (rows : ℕ) : ℕ :=
  (rows + 1) * (rows + 2) / 2

def total_pieces (rows : ℕ) : ℕ :=
  total_rods rows + total_connectors rows

theorem total_pieces_seven_rows : total_pieces 7 = 120 :=
by {
  show total_pieces 7 = 120,
  sorry 
}

end total_pieces_seven_rows_l624_624192


namespace max_heaps_l624_624023

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624023


namespace find_a5_l624_624221

variable {a : ℕ → ℝ}  -- Define the sequence a(n)

-- Define the conditions of the problem
variable (a1_positive : ∀ n, a n > 0)
variable (geo_seq : ∀ n, a (n + 1) = a n * 2)
variable (condition : (a 3) * (a 11) = 16)

theorem find_a5 (a1_positive : ∀ n, a n > 0) (geo_seq : ∀ n, a (n + 1) = a n * 2)
(condition : (a 3) * (a 11) = 16) : a 5 = 1 := by
  sorry

end find_a5_l624_624221


namespace cyclic_AICG_collinear_BIG_l624_624075

variables {A B C D E F G I : Point}
variables {ABC_triangle : Triangle A B C}
variables {incircle_ABC : Circle}
variables (touches_AB : incircle_ABC.TouchesSide ABC_triangle.side_AB at D)
variables (touches_BC : incircle_ABC.TouchesSide ABC_triangle.side_BC at E)
variables (touches_CA : incircle_ABC.TouchesSide ABC_triangle.side_CA at F)
variables (parallel_ADF : Line_through A ∥ Line_through D F)
variables (parallel_CEF : Line_through C ∥ Line_through E F)
variables (G_intersection : G = Point_of_intersection (Line_through A parallel_to DF) (Line_through C parallel_to EF))

-- Part (a): Prove that the quadrilateral AICG is cyclic
theorem cyclic_AICG : CyclicQuadrilateral A I C G :=
sorry

-- Part (b): Prove that the points B, I, G are collinear
theorem collinear_BIG : Collinear B I G :=
sorry

end cyclic_AICG_collinear_BIG_l624_624075


namespace ratio_boxes_produced_by_machines_l624_624496

theorem ratio_boxes_produced_by_machines (x : ℕ) (k : ℕ) 
(h1 : 10 * (x / 10 + k * x / 5) = 5 * x)
(h2 : k = 2) :
(rate_B10 : 10 * 2 * x / 5 = 4 * x) → 
(rate_10A_corect : 10 * (x / 10) = x) →
(rate_ratio : 4 * x / x = 4) :=
begin
  sorry,
end

end ratio_boxes_produced_by_machines_l624_624496


namespace billy_watches_videos_l624_624555

-- Conditions definitions
def num_suggestions_per_list : Nat := 15
def num_iterations : Nat := 5
def pick_index_on_final_list : Nat := 5

-- Main theorem statement
theorem billy_watches_videos : 
  num_suggestions_per_list * num_iterations + (pick_index_on_final_list - 1) = 79 :=
by
  sorry

end billy_watches_videos_l624_624555


namespace geometric_sequence_property_l624_624222

-- Define the sum of the first n terms of a geometric sequence given a common ratio r and first term a
def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

variables {a r x y z : ℝ} {m : ℕ}

-- Given conditions
def S_m := geometric_sum a r m = x
def S_2m := geometric_sum a r (2 * m) = y
def S_3m := geometric_sum a r (3 * m) = z

-- Proof problem
theorem geometric_sequence_property (S_m : geometric_sum a r m = x) 
                                    (S_2m : geometric_sum a r (2 * m) = y)
                                    (S_3m : geometric_sum a r (3 * m) = z) : 
  x^2 + y^2 = xy + xz :=
sorry

end geometric_sequence_property_l624_624222


namespace range_of_d_l624_624223

theorem range_of_d (a_1 d : ℝ) (h : (a_1 + 2 * d) * (a_1 + 3 * d) + 1 = 0) :
  d ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end range_of_d_l624_624223


namespace max_heaps_l624_624017

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624017


namespace total_dolls_l624_624172

-- Defining the given conditions as constants.
def big_boxes : Nat := 5
def small_boxes : Nat := 9
def dolls_per_big_box : Nat := 7
def dolls_per_small_box : Nat := 4

-- The main theorem we want to prove
theorem total_dolls : (big_boxes * dolls_per_big_box) + (small_boxes * dolls_per_small_box) = 71 :=
by
  rw [Nat.mul_add, Nat.mul_eq_mul, Nat.mul_eq_mul]
  exact sorry

end total_dolls_l624_624172


namespace max_heaps_of_stones_l624_624011

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624011


namespace max_heaps_of_stones_l624_624014

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624014


namespace sin_alpha_value_l624_624234

noncomputable def f (x : ℝ) : ℝ := (2 - sin x) / (2 + cos x)

theorem sin_alpha_value (a α : ℝ) (h_deriv : deriv f a = 0) (h_acute : 0 < α ∧ α < π / 2) :
  sin α = (1 + Real.sqrt 7) / 4 :=
sorry

end sin_alpha_value_l624_624234


namespace parallel_lines_l624_624291

theorem parallel_lines (a : ℝ) : 
  let l1 := λ x y, a * x + (a + 2) * y + 2
  let l2 := λ x y, x + a * y + 1
  (∀ x y: ℝ, l1 x y = 0 → l2 x y ≠ 0) → a = -1 := 
by
  sorry

end parallel_lines_l624_624291


namespace fraction_addition_l624_624601

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624601


namespace inequality_inequality_l624_624975

open Real

theorem inequality_inequality (n : ℕ) (k : ℝ) (hn : 0 < n) (hk : 0 < k) : 
  1 - 1/k ≤ n * (k^(1 / n) - 1) ∧ n * (k^(1 / n) - 1) ≤ k - 1 := 
  sorry

end inequality_inequality_l624_624975


namespace max_k_2023_condition_l624_624781

theorem max_k_2023_condition :
  ∃ k : ℕ, k ≤ 2023 ∧ (∀ (redSet : Finset ℕ),
    redSet.card = k ∧ redSet ⊆ Finset.range 2024 →
    (∃ blueSet : Finset ℕ, blueSet ⊆ (Finset.range 2024).eraseFinset redSet ∧
      redSet.sum id = blueSet.sum id)) ∧ k = 673 :=
by
  sorry

end max_k_2023_condition_l624_624781


namespace saree_sale_price_l624_624454

theorem saree_sale_price :
  ∀ (original_price discount1 discount2 : ℝ), 
    original_price = 298 →
    discount1 = 0.12 →
    discount2 = 0.15 →
    let initial_discount := original_price * discount1 in
    let price_after_first_discount := original_price - initial_discount in
    let second_discount := price_after_first_discount * discount2 in
    let final_price := price_after_first_discount - second_discount in
    final_price ≈ Rs. 223 :=
begin
  assume original_price discount1 discount2,
  assume h1: original_price = 298,
  assume h2:  discount1 = 0.12,
  assume h3:  discount2 = 0.15,
  let i_d := original_price * discount1,
  let p_a_f_d := original_price - i_d,
  let s_d := p_a_f_d * discount2,
  let f_p := p_a_f_d - s_d,
  sorry
end

end saree_sale_price_l624_624454


namespace intersection_distance_product_of_distances_to_M_l624_624332

noncomputable def point : Type := ℝ × ℝ

def curve1_param (φ : ℝ) : point :=
  (1 / (Real.tan φ), 1 / (Real.tan φ)^2)

def curve2_polar (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) = 1

def intersection_points (A B : point) : Prop :=
  ∃ φ1 φ2 θ1 θ2 ρ, 
    A = curve1_param φ1 ∧ B = curve1_param φ2 ∧ 
    curve2_polar ρ θ1 ∧ curve2_polar ρ θ2

theorem intersection_distance : 
  ∀ (A B : point), intersection_points A B → 
    ∥A - B∥ = Real.sqrt 10 :=
by sorry 

theorem product_of_distances_to_M :
  ∀ (A B M : point), intersection_points A B → 
    M = (-1, 2) → 
    ∥M - A∥ * ∥M - B∥ = 2 :=
by sorry

end intersection_distance_product_of_distances_to_M_l624_624332


namespace fraction_addition_l624_624599

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624599


namespace negate_proposition_l624_624841

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- The original proposition p
def p : Prop := ∀ x, is_odd x → is_even (2 * x)

-- The negation of the proposition p
def neg_p : Prop := ∃ x, is_odd x ∧ ¬ is_even (2 * x)

-- Proof problem statement: Prove that the negation of proposition p is as defined in neg_p
theorem negate_proposition :
  (∀ x, is_odd x → is_even (2 * x)) ↔ (∃ x, is_odd x ∧ ¬ is_even (2 * x)) :=
sorry

end negate_proposition_l624_624841


namespace coefficient_of_x4_in_expansion_l624_624231

theorem coefficient_of_x4_in_expansion :
  let f := (1 - (1 / x)) * (1 + x) ^ 7 in
  ((f : ℝ[x]).coeff 4) = 14 :=
by
  sorry

end coefficient_of_x4_in_expansion_l624_624231


namespace add_fractions_l624_624730

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624730


namespace incorrect_match_l624_624491

def Scientist := String
def Field := String

def matches (scientist : Scientist) (field : Field) : Prop :=
  match (scientist, field) with
  | ("Descartes", "Analytic Geometry") => True
  | ("Pascal", "Probability Theory") => True
  | ("Cantor", "Set Theory") => True
  | ("Zu Chongzhi", "Calculation of Pi") => True
  | _ => False

theorem incorrect_match : ∀(s : Scientist) (f : Field), ¬ matches s f → (s, f) = ("Zu Chongzhi", "Complex Number Theory") :=
by
  intros s f h
  cases s
  cases f
  sorry 

end incorrect_match_l624_624491


namespace find_k_position_l624_624226

-- Definitions for the binary code and XOR operation.
def xor (a b : Bool) : Bool := a != b

-- Parity check equations
def parity_checks (x1 x2 x3 x4 x5 x6 x7 : Bool) : Prop :=
  xor (xor x4 x5) (xor x6 x7) = false ∧
  xor (xor x2 x3) (xor x6 x7) = false ∧
  xor (xor x1 x3) (xor x5 x7) = false

-- Function to introduce a bit error at the k-th position
def introduce_error (code : List Bool) (k : Nat) : List Bool :=
  code.set k (not code[k])

-- The binary code
def original_code : List Bool := [true, true, false, true, true, false, true]

-- The problem statement with the specific error position
theorem find_k_position :
  ∃ k, k = 5 ∧
  let erroneous_code := introduce_error original_code (k - 1) in
  parity_checks erroneous_code[0] erroneous_code[1] erroneous_code[2]
                erroneous_code[3] erroneous_code[4] erroneous_code[5]
                erroneous_code[6] :=
by 
  -- Construct the proof here
  sorry

end find_k_position_l624_624226


namespace add_fractions_result_l624_624588

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624588


namespace sum_of_angles_l624_624257

theorem sum_of_angles (α β : ℝ) (h1 : tan α = -3 * sqrt 3) (h2 : tan β = 4)
  (h3 : α ∈ Ioo (-π/2) (π/2)) (h4 : β ∈ Ioo (-π/2) (π/2)) :
  α + β = -2 * π / 3 := 
sorry

end sum_of_angles_l624_624257


namespace systematic_sampling_count_l624_624155

theorem systematic_sampling_count :
  ∀ (total_people selected_people initial_draw : ℕ),
  total_people = 960 →
  selected_people = 32 →
  initial_draw = 9 →
  let common_difference := total_people / selected_people in
  let general_term (n : ℕ) := initial_draw + common_difference * (n - 1) in
  let count_in_interval := (25 - 16 + 1) in
  count_in_interval = 10 :=
begin
  intros total_people selected_people initial_draw h_total h_selected h_initial,
  let common_difference := total_people / selected_people,
  let general_term (n : ℕ) := initial_draw + common_difference * (n - 1),
  let count_in_interval := (25 - 16 + 1),
  have : common_difference = 30, by { rw [h_total, h_selected], exact nat.div_self (nat.pos_of_ne_zero (by norm_num)) },
  have : general_term 1 = initial_draw, by refl,
  have : ∀ n, general_term n = 30 * n - 29, by {
    intro n,
    simp only [general_term, common_difference],
    rw [nat.mul_sub_left_distrib, nat.sub_self, mul_one, nat.add_sub_left (le_of_lt (by norm_num : 0 < 9))],
    rw [mul_comm] },
  exact sorry
end

end systematic_sampling_count_l624_624155


namespace money_difference_l624_624552

-- Given conditions
def packs_per_hour_peak : Nat := 6
def packs_per_hour_low : Nat := 4
def price_per_pack : Nat := 60
def hours_per_day : Nat := 15

-- Calculate total sales in peak and low seasons
def total_sales_peak : Nat :=
  packs_per_hour_peak * price_per_pack * hours_per_day

def total_sales_low : Nat :=
  packs_per_hour_low * price_per_pack * hours_per_day

-- The Lean statement proving the correct answer
theorem money_difference :
  total_sales_peak - total_sales_low = 1800 :=
by
  sorry

end money_difference_l624_624552


namespace f_eccentricity_l624_624256

noncomputable def hyperbola_eccentricity := 
  ∃ (a b c e : ℝ), 
  a > 0 ∧ b > 0 ∧ 
  (∃ (F1 F2 : ℝ × ℝ),
    let h := ∀ x y, x * x / (a * a) - y * y / (b * b) = 1 in
    let c := Math.sqrt (a * a + b * b) in
    let e := c / a in
    let circle := ∀ x y, x * x + y * y = c * c in
    ∃ (M N : ℝ × ℝ),
      M.1 > 0 ∧ M.2 > 0 ∧ N.1 > 0 ∧ N.2 > 0 ∧
      let asymptote := ∀ x y, y = b / a * x in
      let is_parallel := ∀ x y, y * e = M.2 * (N.1 + F1.1 - F2.1) in
      MF1 ∥ ON ∧
      Math.pow (e, 3) + 2 * Math.pow (e, 2) - 2 * e - 2 = 0 ∧
      f (e) = 2
  )

theorem f_eccentricity : ∀ (a b e : ℝ), 
  a > 0 ∧ b > 0 ∧ 
  (
    let c := Math.sqrt (a * a + b * b) in
    e = c / a ∧
    Math.pow (e, 3) + 2 * Math.pow (e, 2) - 2 * e - 2 = 0
  ) → 
  (x^2 + 2 * x - 2 / x = 2) :=
by
  sorry

end f_eccentricity_l624_624256


namespace part1_arith_seq_part2_max_m_part3_min_t_l624_624373

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := x / (x + 1)

-- Part 1.
def sequence_x (x : ℝ) (n : ℕ) : ℝ := 
nat.rec_on n (g 1)
  (λ n xn, g xn)

theorem part1_arith_seq :
  ∃ a d : ℝ, ∀ n : ℕ, n ≥ 1 → (1 / sequence_x 1 n) = a + n * d := 
sorry

-- Part 2.
theorem part2_max_m : 
  ∃ m : ℤ, (∃∀ x > 0, f(x) > m * g(x) - 1) ∧ ∀ m', (∃∀ x > 0, f(x) > m' * g(x) - 1) → m' ≤ m := 
sorry

-- Part 3.
def sum_g (n : ℕ) : ℝ := 
finset.sum (finset.range n) (λ k, g (k + 1))

theorem part3_min_t :
  ∃ t : ℕ, (∀ n : ℕ, n ≥ t → f(n - t.to_real) < n - sum_g n) ∧ (∀ t', (∀ n : ℕ, n ≥ t' → f(n - t'.to_real) < n - sum_g n) → t' ≥ t) :=
sorry

end part1_arith_seq_part2_max_m_part3_min_t_l624_624373


namespace count_lines_4x4_grid_excluding_vertical_l624_624296

def number_of_lines_through_two_points_4x4_grid_excluding_vertical : ℕ :=
  84

theorem count_lines_4x4_grid_excluding_vertical :
  (∀ (grid : fin 4 → fin 4), ∃ n : ℕ, n = number_of_lines_through_two_points_4x4_grid_excluding_vertical) :=
  begin 
    sorry 
  end

end count_lines_4x4_grid_excluding_vertical_l624_624296


namespace find_m__l624_624289

noncomputable def C1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2*x + 2*y - 2 = 0
noncomputable def C2 (m : ℝ) : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2*m*x = 0
def chord_length : ℝ := 2

theorem find_m_ satisfying_conditions (m : ℝ)
  (hC1 : ∀ x y : ℝ, C1 x y)
  (hC2 : ∀ x y : ℝ, C2 m x y)
  (hm : 0 < m)
  (hchord : chord_length = 2) :
  m = (Real.sqrt 6 / 2) :=
sorry

end find_m__l624_624289


namespace initial_observations_l624_624991

theorem initial_observations {n : ℕ} (S : ℕ) (new_observation : ℕ) 
  (h1 : S = 15 * n) (h2 : new_observation = 14 - n)
  (h3 : (S + new_observation) / (n + 1) = 14) : n = 6 :=
sorry

end initial_observations_l624_624991


namespace exist_pairwise_distinct_gcd_l624_624379

theorem exist_pairwise_distinct_gcd (S : Set ℕ) (h_inf : S.Infinite) 
  (h_gcd : ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ gcd a b ≠ gcd c d) :
  ∃ x y z : ℕ, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x := 
by sorry

end exist_pairwise_distinct_gcd_l624_624379


namespace power_sum_l624_624437

theorem power_sum : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end power_sum_l624_624437


namespace inverse_solution_correct_l624_624930

noncomputable def f (a b c x : ℝ) : ℝ :=
  1 / (a * x^2 + b * x + c)

theorem inverse_solution_correct (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c x = 1 ↔ x = (-b + Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) ∨
               x = (-b - Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) :=
by
  sorry

end inverse_solution_correct_l624_624930


namespace fraction_addition_l624_624604

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624604


namespace complex_number_quadrant_l624_624248

noncomputable def sum_infinite_geometric_series (a1 r : ℝ) (r_lt_1 : r < 1) : ℝ :=
  a1 / (1 - r)

theorem complex_number_quadrant 
  (S_n : ℕ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ n : ℕ, S_n n = 1 - (1 / 2) ^ n) 
  (h2 : a = sum_infinite_geometric_series 1 (1 / 2) (by norm_num))
  (h3 : ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (S_n n - a) < ε) :
  let z := (1 : ℂ) / (a + (Complex.I : ℂ)) in
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_quadrant_l624_624248


namespace max_heaps_l624_624019

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624019


namespace proof_2720000_scientific_l624_624062

def scientific_notation (n : ℕ) : ℝ := 
  2.72 * 10^6 

theorem proof_2720000_scientific :
  scientific_notation 2720000 = 2.72 * 10^6 := by
  sorry

end proof_2720000_scientific_l624_624062


namespace complete_graph_17_three_color_l624_624765

theorem complete_graph_17_three_color (V : Type) [Fintype V] [DecidableEq V] (G : SimpleGraph V) 
  (h : Fintype.card V = 17) (color : (G.edgeSet : Set (Sym2 V)) → Fin 3) :
  ∃ (u v w : V), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ color ⟦(u,v)⟧ = color ⟦(v,w)⟧ ∧ color ⟦(v,w)⟧ = color ⟦(u,w)⟧ :=
sorry

end complete_graph_17_three_color_l624_624765


namespace coeff_x2_in_binom_expansion_is_8_l624_624770

/-- The binomial expansion of (x + 2/x)^4 -/
def binom_expansion (x : ℂ) : ℂ := (x + 2/x) ^ 4

/-- The coefficient of x^2 in the binomial expansion of (x + 2/x)^4 is 8 -/
theorem coeff_x2_in_binom_expansion_is_8 :
  ∃ (c : ℂ), (binom_expansion x).coeff 2 = c ∧ c = 8 :=
sorry

end coeff_x2_in_binom_expansion_is_8_l624_624770


namespace probability_three_of_a_kind_after_reroll_l624_624220

theorem probability_three_of_a_kind_after_reroll 
  (no_four_of_a_kind : ∀ (d : Fin 5 → Fin 6), ¬ (∃ i, (∑ j, if d j = d i then 1 else 0) ≥ 4))
  (three_of_a_kind : ∃ (d : Fin 5 → Fin 6), ∃ i, (∑ j, if d j = d i then 1 else 0) = 3) :
  let dice_rolls := λ (d : Fin 2 → Fin 6), true in
  (∃ (d' : Fin 2 → Fin 6), 
    (∑ j, if d' j = (choose (three_of_a_kind.some)).some then 1 else 0) ≥ 1 ∨
    (d' 0 = d' 1) ∧ (d' 0 ≠ (choose (three_of_a_kind.some)).some)) →
  (∃ (d' : Fin 2 → Fin 6), 
    (∑ j, if d' j = (choose (three_of_a_kind.some)).some then 1 else 0) + 3 ≥ 3 ∨
    (d' 0 = d' 1) ∧ (d' 0 ≠ (choose (three_of_a_kind.some)).some) → 
  ((Prob dice_rolls).toNonnegRealdice_rolls = 4/9) := sorry

end probability_three_of_a_kind_after_reroll_l624_624220


namespace count_three_digit_numbers_with_sum_of_digits_odd_l624_624161

def is_three_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000

def digits_composed_of_1_2_3_4_5_without_repetition (n : Nat) : Prop :=
  ∃ a b c : Nat, n = 100 * a + 10 * b + c ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  a ∈ {1, 2, 3, 4, 5} ∧
  b ∈ {1, 2, 3, 4, 5} ∧
  c ∈ {1, 2, 3, 4, 5}

def sum_of_digits_is_odd (n : Nat) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a + b + c) % 2 = 1

theorem count_three_digit_numbers_with_sum_of_digits_odd :
  Nat.card {n : Nat // is_three_digit_number n ∧ digits_composed_of_1_2_3_4_5_without_repetition n ∧ sum_of_digits_is_odd n} = 24 :=
by
  sorry

end count_three_digit_numbers_with_sum_of_digits_odd_l624_624161


namespace tomTotalWeightMoved_is_525_l624_624102

-- Tom's weight
def tomWeight : ℝ := 150

-- Weight in each hand
def weightInEachHand : ℝ := 1.5 * tomWeight

-- Weight vest
def weightVest : ℝ := 0.5 * tomWeight

-- Total weight moved
def totalWeightMoved : ℝ := (weightInEachHand * 2) + weightVest

theorem tomTotalWeightMoved_is_525 : totalWeightMoved = 525 := by
  sorry

end tomTotalWeightMoved_is_525_l624_624102


namespace find_p_q_coprime_sum_l624_624107

theorem find_p_q_coprime_sum (x y n m: ℕ) (h_sum: x + y = 30)
  (h_prob: ((n/x) * (n-1)/(x-1) * (n-2)/(x-2)) * ((m/y) * (m-1)/(y-1) * (m-2)/(y-2)) = 18/25)
  : ∃ p q : ℕ, p.gcd q = 1 ∧ p + q = 1006 :=
by
  sorry

end find_p_q_coprime_sum_l624_624107


namespace exists_infinitely_many_n_l624_624413

theorem exists_infinitely_many_n (k : ℕ) :
  ∃ (n : ℕ), ∃ (f : ℕ → ℕ), ∀ (i j : ℕ), 
    (1 ≤ i ∧ i ≤ n) → 
    (1 ≤ j ∧ j ≤ 3) → 
    (f (i, 1) + f (i, 2) + f (i, 3)) % 6 = 0 ∧
    (∑ i in range n, f (i, 1)) % 6 = 0 ∧ 
    (∑ i in range n, f (i, 2)) % 6 = 0 ∧ 
    (∑ i in range n, f (i, 3)) % 6 = 0 :=
begin
  sorry
end

end exists_infinitely_many_n_l624_624413


namespace equation_of_perpendicular_line_l624_624208

-- Definitions for the conditions
def Point (x y : ℝ) := (x, y)
def Line (a b c : ℝ) := λ (x y : ℝ), a * x + b * y + c = 0

-- Given conditions
def A : (ℝ × ℝ) := Point 2 6
def L : (ℝ → ℝ → Prop) := Line 1 (-1) (-2)

-- The statement to prove: The equation of the line that passes through A and is perpendicular to L
theorem equation_of_perpendicular_line :
  ∃ a b c : ℝ, (Line a b c) 2 6 ∧ (∀ x y, L x y → (Line a b c) x y = 0) ∧ (a * b = 0) ∧ a = 1 ∧ b = 1 ∧ c = -8 :=
by
  sorry

end equation_of_perpendicular_line_l624_624208


namespace range_of_a_l624_624253

def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, ¬ (x^2 + (a-1)*x + 1 ≤ 0)
def proposition_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a-1)^x < (a-1)^y

theorem range_of_a (a : ℝ) :
  ¬ (proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a) →
  (-1 < a ∧ a ≤ 2) ∨ (3 ≤ a) :=
by sorry

end range_of_a_l624_624253


namespace solve_sin_equation_l624_624423

theorem solve_sin_equation
  (a1 : Real) :
  ( (Real.sin (2 * a1) - Real.pi * Real.sin a1) * Real.sqrt (11 * a1^2 - a1^4 - 10) = 0 ) →
  ( a1 ∈ {-Real.sqrt 10, -Real.pi, -1, 1, Real.pi, Real.sqrt 10} ) :=
by sorry

end solve_sin_equation_l624_624423


namespace range_of_a_l624_624139

noncomputable def condition_p (a : ℝ) : Prop := ∀ x > 0, is_monotone_decreasing (log a (x+3))
noncomputable def condition_q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + (2*a-3)*x₁ + 1 = 0) ∧ (x₂^2 + (2*a-3)*x₂ + 1 = 0)

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : condition_p a ∨ condition_q a) (h4 : ¬ (condition_p a ∧ condition_q a)) : a ∈ Set.Icc (1/2) 1 ∪ Set.Ioc (5/2) ∞ := by
  sorry

end range_of_a_l624_624139


namespace range_g_l624_624367

noncomputable def f (x : ℝ) : ℝ := (1 - Real.exp x) / (1 + Real.exp x)

def g (x : ℝ) : ℝ := Int.floor (f x) + Int.floor (f (-x))

theorem range_g :
  Set.range g = {0, -1} :=
sorry

end range_g_l624_624367


namespace intersection_M_N_l624_624858

-- Define the sets based on the given conditions
def M : Set ℝ := {x | x + 2 < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {x | x < -2} := by
sorry

end intersection_M_N_l624_624858


namespace correct_option_is_optionB_l624_624988

-- Definitions based on conditions
def optionA : ℝ := 0.37 * 1.5
def optionB : ℝ := 3.7 * 1.5
def optionC : ℝ := 0.37 * 1500
def original : ℝ := 0.37 * 15

-- Statement to prove that the correct answer (optionB) yields the same result as the original expression
theorem correct_option_is_optionB : optionB = original :=
sorry

end correct_option_is_optionB_l624_624988


namespace lucille_house_shorter_than_average_l624_624390

variable (lh fn hn : ℕ)

theorem lucille_house_shorter_than_average
    (h_lh : lh = 80)
    (h_fn : fn = 70)
    (h_hn : hn = 99) :
    let avg_height := (lh + fn + hn) / 3 in
    avg_height - lh = 3 :=
by
  rw [h_lh, h_fn, h_hn]
  have h_total : 80 + 70 + 99 = 249 := by norm_num
  have h_avg : 249 / 3 = 83 := by norm_num
  rw [h_total, h_avg]
  norm_num
  sorry

end lucille_house_shorter_than_average_l624_624390


namespace area_of_triangle_COD_eq_8p_l624_624410

-- Define points C, O, and D
def C (p : ℝ) : ℝ × ℝ := (0, p)
def O : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (16, 0)

-- Definition of the function to compute the area of ΔCOD
def area_of_triangle_COD (p : ℝ) : ℝ :=
  1/2 * 16 * p

theorem area_of_triangle_COD_eq_8p (p : ℝ) : area_of_triangle_COD p = 8 * p :=
by
  unfold area_of_triangle_COD
  have h : (1 / 2 * 16 * p = 8 * p) := by linarith
  exact h

end area_of_triangle_COD_eq_8p_l624_624410


namespace fraction_addition_l624_624606

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624606


namespace union_complement_l624_624389

-- Define the universal set U as the set of real numbers
def U := set ℝ

-- Define the set A as {x | x < 0}
def A := {x : ℝ | x < 0}

-- Define the set B as {x | x > 1}
def B := {x : ℝ | x > 1}

-- Define the complement of B in U
def complement_B := {x : ℝ | x ≤ 1}

-- The goal is to prove that A ∪ complement_B is {x | x ≤ 1}
theorem union_complement (x : ℝ) : x ∈ A ∪ complement_B ↔ x ≤ 1 := by
  sorry

end union_complement_l624_624389


namespace project_hours_l624_624406

variable (K : ℕ)

theorem project_hours 
    (h_total : K + 2 * K + 3 * K + K / 2 = 180)
    (h_k_nearest : K = 28) :
    3 * K - K = 56 := 
by
  -- Proof goes here
  sorry

end project_hours_l624_624406


namespace sum_fractions_eq_l624_624692

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624692


namespace no_opposite_configuration_l624_624136

def points : Fin 2022 → ℤ
| i => if i % 2 = 0 then 1 else -1

def initialColors : Fin 2022 → ℤ
| i => if i = 0 then -1 else 1

def flipAdjacent (colors : Fin 2022 → ℤ) (i : Fin 2022) : Fin 2022 → ℤ :=
if colors i = colors (⟨i + 1 % 2022, sorry⟩ : Fin 2022) then
  fun j => if j = i ∨ j = (⟨i + 1 % 2022, sorry⟩ : Fin 2022) then -colors j else colors j
else
  colors

def flipWithGap (colors : Fin 2022 → ℤ) (i : Fin 2022) : Fin 2022 → ℤ :=
if colors i ≠ colors (⟨i + 2 % 2022, sorry⟩ : Fin 2022) then
  fun j => if j = i ∨ j = (⟨i + 2 % 2022, sorry⟩ : Fin 2022) then -colors j else colors j
else
  colors

theorem no_opposite_configuration :
  ¬ ∃ (f : Fin 2022 → ℤ),
    (∀ i, f i = -initialColors i) ∧
    (∃ fn : ℕ → Fin 2022 → ℤ, fn 0 = initialColors ∧
      (∀ n, fn (n + 1) = flipAdjacent (fn n) ∨ fn (n + 1) = flipWithGap (fn n))) :=
by skip_proof_reasoning sorry

end no_opposite_configuration_l624_624136


namespace transformed_roots_l624_624388

def roots_of_quadratic (a b c : ℚ) : set ℚ := 
  {x : ℚ | a * x^2 + b * x + c = 0}

theorem transformed_roots (α β : ℚ) 
  (h1 : roots_of_quadratic 2 (-5) 3 = {α, β}) :
  roots_of_quadratic 1 9 20 = {2*α - 7, 2*β - 7} :=
by
  -- the proof would go here
  sorry

end transformed_roots_l624_624388


namespace inverse_of_3_mod_243_l624_624205

theorem inverse_of_3_mod_243 : ∃ x : ℤ, (0 ≤ x ∧ x ≤ 242 ∧ 3 * x ≡ 1 [MOD 243]) :=
by
  use 324
  split
  · norm_num
  split
  · norm_num
  · norm_num
  · ring_nf, exact_mod_cast rfl, sorry

end inverse_of_3_mod_243_l624_624205


namespace sid_fraction_left_l624_624055

noncomputable def fraction_left (original total_spent remaining additional : ℝ) : ℝ :=
  (remaining - additional) / original

theorem sid_fraction_left 
  (original : ℝ := 48) 
  (spent_computer : ℝ := 12) 
  (spent_snacks : ℝ := 8) 
  (remaining : ℝ := 28) 
  (additional : ℝ := 4) :
  fraction_left original (spent_computer + spent_snacks) remaining additional = 1 / 2 :=
by
  sorry

end sid_fraction_left_l624_624055


namespace count_perfect_divisors_l624_624875

def is_cyclic_perfect_divisor (N : ℕ) : Prop :=
∀ a b c : ℕ, (0 ≤ a) ∧ (a < 10) ∧ (0 ≤ b) ∧ (b < 10) ∧ (0 ≤ c) ∧ (c < 10) →
(N ∣ (100 * a + 10 * b + c) →
(N ∣ (100 * b + 10 * c + a) ∧ N ∣ (100 * c + 10 * a + b)))

theorem count_perfect_divisors : {N : ℕ // is_cyclic_perfect_divisor N}.count = 14 :=
sorry

end count_perfect_divisors_l624_624875


namespace problem_solution_l624_624832

noncomputable def a_sequence : ℕ → ℕ := sorry
noncomputable def S_n : ℕ → ℕ := sorry
noncomputable def b_sequence : ℕ → ℕ := sorry
noncomputable def c_sequence : ℕ → ℕ := sorry
noncomputable def T_n : ℕ → ℕ := sorry

theorem problem_solution (n : ℕ) (a_condition : ∀ n : ℕ, 2 * S_n = (n + 1) ^ 2 * a_sequence n - n ^ 2 * a_sequence (n + 1))
                        (b_condition : ∀ n : ℕ, b_sequence 1 = a_sequence 1 ∧ (n ≠ 0 → n * b_sequence (n + 1) = a_sequence n * b_sequence n)) :
  (∀ n, a_sequence n = 2 * n) ∧
  (∀ n, b_sequence n = 2 ^ n) ∧
  (∀ n, T_n n = 2 ^ (n + 1) + n ^ 2 + n - 2) :=
sorry


end problem_solution_l624_624832


namespace pipe_A_fill_time_l624_624542

theorem pipe_A_fill_time 
  (t : ℝ)
  (ht : (1 / t - 1 / 6) = 4 / 15.000000000000005) : 
  t = 30 / 13 :=  
sorry

end pipe_A_fill_time_l624_624542


namespace reduced_price_is_25_l624_624499

def original_price (P : ℝ) (X : ℝ) (R : ℝ) : Prop :=
  R = 0.85 * P ∧ 
  500 = X * P ∧ 
  500 = (X + 3) * R

theorem reduced_price_is_25 (P X R : ℝ) (h : original_price P X R) :
  R = 25 :=
by
  sorry

end reduced_price_is_25_l624_624499


namespace area_of_stripe_l624_624522

-- Definitions based on the conditions
def diameter : ℝ := 40
def height : ℝ := 60
def stripe_width : ℝ := 4
def revolutions : ℕ := 3

-- Prove that the area of the stripe is 480π square feet
theorem area_of_stripe : 
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let stripe_length := revolutions * circumference
  let stripe_height := height / 2
  let area := stripe_length * stripe_width
  area = 480 * Real.pi := 
  by
    sorry

end area_of_stripe_l624_624522


namespace water_in_glasses_equal_l624_624464

theorem water_in_glasses_equal (V : ℝ) :
  (V * (list.prod (list.map (λ n, 1 + n / 100) (list.range' 1 27)))) =
  (V * (list.prod (list.map (λ n, 1 + n / 100) (list.reverse (list.range' 1 27))))) :=
by
  sorry

end water_in_glasses_equal_l624_624464


namespace required_volume_proof_l624_624298

-- Defining the conditions
def initial_volume : ℝ := 60
def initial_concentration : ℝ := 0.10
def final_concentration : ℝ := 0.15

-- Defining the equation
def required_volume (V : ℝ) : Prop :=
  (initial_concentration * initial_volume + V = final_concentration * (initial_volume + V))

-- Stating the proof problem
theorem required_volume_proof :
  ∃ V : ℝ, required_volume V ∧ V = 3 / 0.85 :=
by {
  -- Proof skipped
  sorry
}

end required_volume_proof_l624_624298


namespace sequence_contains_1992_smallest_n_for_1992_l624_624282

-- Define the largest odd divisor function g(x)
def g (x : ℕ) : ℕ :=
  if x = 0 then 0 else
  let odd_divisors := {d | d ∣ x ∧ d % 2 = 1}
  in Finset.max' (Finset.filter (λ d, d ∈ odd_divisors) (Finset.range (x + 1))) sorry

-- Define the function f(x) based on whether x is even or odd
def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 + x / g x else 2 ^ ((x + 1) / 2)

-- Define the sequence x_n such that x_1 = 1 and x_(n+1) = f(x_n)
def x : ℕ → ℕ
| 0     := 1
| (n+1) := f (x n)

-- Prove that x_8253 = 1992
theorem sequence_contains_1992 : x 8252 = 1992 :=
  sorry

-- Prove that n = 8252 is the smallest such n for which x_n = 1992
theorem smallest_n_for_1992 : ∀ m < 8252, x m ≠ 1992 :=
  sorry

end sequence_contains_1992_smallest_n_for_1992_l624_624282


namespace part1_part2_l624_624293

-- Definitions
def m (x : ℝ) : Prod ℝ ℝ := (Real.sin x, -1)
def n (x : ℝ) : Prod ℝ ℝ := (Real.sqrt 3 * Real.cos x, -0.5)
def f (x : ℝ) : ℝ := ((m x).fst + (n x).fst) * (m x).fst + ((m x).snd + (n x).snd) * (m x).snd

-- Constants for the triangle
def a : ℝ := 2 * Real.sqrt 3
def c : ℝ := 4
def max_f_in_interval : ℝ := 3
def A : ℝ := Real.pi / 3
def b : ℝ -- it's implied but not explicitly given
def S : ℝ := 2 * Real.sqrt 3

-- Main theorem statements
theorem part1 (x : ℝ) : f(x) = Real.sin (2 * x - Real.pi / 6) + 2 ∧
  ∀ k : ℤ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi → 
  (f x).1 > 0 :=
sorry

theorem part2 : f(A) = max_f_in_interval ∧ 
  (a = 2 * Real.sqrt 3 ∧ c = 4 → 
    ∃ B : ℝ, B = Real.pi / 6 ∧ 
    (∃ S : ℝ, S = 2 * Real.sqrt 3)) :=
sorry

end part1_part2_l624_624293


namespace condition_for_fg_eq_gf_l624_624254

variables (a b c d : ℝ)

def f (x : ℝ) := a*x + b
def g (x : ℝ) := c*x + d

theorem condition_for_fg_eq_gf :
  (∀ x, f(g x) = g(f x)) ↔ (b = d ∨ a = c + 1) :=
by
  intro h,
  sorry

end condition_for_fg_eq_gf_l624_624254


namespace total_red_marbles_l624_624356

-- Definitions derived from the problem conditions
def Jessica_red_marbles : ℕ := 3 * 12
def Sandy_red_marbles : ℕ := 4 * Jessica_red_marbles
def Alex_red_marbles : ℕ := Jessica_red_marbles + 2 * 12

-- Statement we need to prove that total number of marbles is 240
theorem total_red_marbles : 
  Jessica_red_marbles + Sandy_red_marbles + Alex_red_marbles = 240 := by
  -- We provide the proof later
  sorry

end total_red_marbles_l624_624356


namespace stacked_height_is_correct_l624_624095

-- Define the radius based on the given diameter
def radius (d : ℝ) : ℝ := d / 2

-- Define the height of an equilateral triangle given the side length
def height_equilateral_triangle (s : ℝ) : ℝ := (s * Real.sqrt 3) / 2

-- The stacked height of three spherical balls in a triangular pyramid formation
def stacked_height : ℝ :=
  let r : ℝ := radius 12
  let h_triangle : ℝ := height_equilateral_triangle (2 * r)
  r + h_triangle + r

theorem stacked_height_is_correct :
  stacked_height = 12 + 6 * Real.sqrt 3 :=
by
  sorry

end stacked_height_is_correct_l624_624095


namespace fraction_addition_l624_624645

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624645


namespace max_heaps_of_stones_l624_624013

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624013


namespace sum_of_squares_increase_by_l624_624824

theorem sum_of_squares_increase_by {n : ℕ} (h1 : n = 100) (x : Fin n.succ → ℝ)
  (h2 : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  (∑ i, (x i + 4) ^ 2) = (∑ i, x i ^ 2) + 800 :=
by sorry

end sum_of_squares_increase_by_l624_624824


namespace fraction_addition_l624_624653

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624653


namespace find_abs_product_l624_624386

noncomputable def distinct_nonzero_real (a b c : ℝ) : Prop :=
a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

theorem find_abs_product (a b c : ℝ) (h1 : distinct_nonzero_real a b c) 
(h2 : a + 1/(b^2) = b + 1/(c^2))
(h3 : b + 1/(c^2) = c + 1/(a^2)) :
  |a * b * c| = 1 :=
sorry

end find_abs_product_l624_624386


namespace sum_fractions_eq_l624_624686

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624686


namespace sum_of_roots_is_correct_l624_624381

noncomputable def f (x : ℝ) := x^2 + 14*x + 7

theorem sum_of_roots_is_correct :
  let Φ (g : ℝ → ℝ) := g⁻¹ = g ∘ (λ x, (4 * x)⁻¹)
  in ∑ x in {x : ℝ | Φ f }, x = 
  sorry := sorry

end sum_of_roots_is_correct_l624_624381


namespace nonneg_int_solutions_to_ineq_system_l624_624212

open Set

theorem nonneg_int_solutions_to_ineq_system :
  {x : ℤ | (5 * x - 6 ≤ 2 * (x + 3)) ∧ ((x / 4 : ℚ) - 1 < (x - 2) / 3)} = {0, 1, 2, 3, 4} :=
by
  sorry

end nonneg_int_solutions_to_ineq_system_l624_624212


namespace gcd_288_123_l624_624476

-- Define the conditions
def cond1 : 288 = 2 * 123 + 42 := by sorry
def cond2 : 123 = 2 * 42 + 39 := by sorry
def cond3 : 42 = 39 + 3 := by sorry
def cond4 : 39 = 13 * 3 := by sorry

-- Prove that GCD of 288 and 123 is 3
theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end gcd_288_123_l624_624476


namespace meena_cookies_left_l624_624945

-- Define the given conditions in terms of Lean definitions
def total_cookies_baked := 5 * 12
def cookies_sold_to_stone := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

-- Define the total cookies sold
def total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy

-- Define the number of cookies left
def cookies_left := total_cookies_baked - total_cookies_sold

-- Prove that the number of cookies left is 15
theorem meena_cookies_left : cookies_left = 15 := by
  -- The proof is omitted (sorry is used to skip proof)
  sorry

end meena_cookies_left_l624_624945


namespace simplify_frac_l624_624056

variable (b c : ℕ)
variable (b_val : b = 2)
variable (c_val : c = 3)

theorem simplify_frac : (15 * b ^ 4 * c ^ 2) / (45 * b ^ 3 * c) = 2 :=
by
  rw [b_val, c_val]
  sorry

end simplify_frac_l624_624056


namespace factorize_x2_minus_9_l624_624775

theorem factorize_x2_minus_9 (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
sorry

end factorize_x2_minus_9_l624_624775


namespace add_fractions_l624_624577

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624577


namespace min_cells_for_vasya_to_lose_l624_624039

theorem min_cells_for_vasya_to_lose : 
  ∀ (grid : fin 4 × fin 4) (painted_cells : set (fin 4 × fin 4)), 
  (∀ (L_shape : (fin 4 × fin 4) → set (fin 4 × fin 4)), 
    (∀ ⦃cell : fin 4 × fin 4⦄, cell ∈ painted_cells → ∃ (l : fin 4 × fin 4 → bool), 
    (∀ c ∈ painted_cells, l c = tt → l ∈ L_shape cell)) → 
      card painted_cells < 16 → 
        (∃ (Painted : fin 4 × fin 4 → bool), 
         ∀ (L : set (fin 4 × fin 4)), 
         ∃ (L' : set (fin 4 × fin 4)), 
         L = L'.map (λ x, (x.1, x.2)) ∧ disjoint L painted_cells)) :=
begin
  sorry
end

end min_cells_for_vasya_to_lose_l624_624039


namespace sum_of_digits_l624_624871

theorem sum_of_digits (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h3 : 34 * a + 42 * b = 142) : a + b = 4 := 
by
  sorry

end sum_of_digits_l624_624871


namespace problem_intersection_l624_624924

open set

-- Defining the sets A and B
def A := {x : ℕ | x ≠ 0 ∧ sqrt x ≤ 2}
def B := {y : ℕ | ∃ x : ℕ, y = x^2 + 2}

-- The proof statement to be proven:
theorem problem_intersection : A ∩ B = {2, 3, 4} := by 
  sorry

end problem_intersection_l624_624924


namespace add_fractions_result_l624_624581

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624581


namespace add_fractions_l624_624715

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624715


namespace interest_difference_l624_624150

theorem interest_difference
  (principal : ℕ) (rate : ℚ) (time : ℕ) (interest : ℚ) (difference : ℚ)
  (h1 : principal = 600)
  (h2 : rate = 0.05)
  (h3 : time = 8)
  (h4 : interest = principal * (rate * time))
  (h5 : difference = principal - interest) :
  difference = 360 :=
by sorry

end interest_difference_l624_624150


namespace women_in_first_class_correct_l624_624974

-- We start by defining the conditions provided in the problem
def total_passengers : ℕ := 300
def percent_women : ℝ := 0.70
def percent_first_class : ℝ := 0.15

-- Define the number of women on the ship
def number_of_women : ℕ := (total_passengers * percent_women).toNat

-- Define the number of women in first class
def number_of_women_in_first_class : ℕ := (number_of_women * percent_first_class).toNat

-- The theorem that encapsulates the question and conditions, proving that the number of women in first class is 31
theorem women_in_first_class_correct :
    number_of_women_in_first_class = 31 := by
  sorry

end women_in_first_class_correct_l624_624974


namespace max_a2_b2_c2_d2_l624_624385

-- Define the conditions for a, b, c, d
variables (a b c d : ℝ) 

-- Define the hypotheses from the problem
variables (h₁ : a + b = 17)
variables (h₂ : ab + c + d = 94)
variables (h₃ : ad + bc = 195)
variables (h₄ : cd = 120)

-- Define the final statement to be proved
theorem max_a2_b2_c2_d2 : ∃ (a b c d : ℝ), a + b = 17 ∧ ab + c + d = 94 ∧ ad + bc = 195 ∧ cd = 120 ∧ (a^2 + b^2 + c^2 + d^2) = 918 :=
by sorry

end max_a2_b2_c2_d2_l624_624385


namespace OH_squared_l624_624048

variables {α β γ R : ℝ}

/-- Given conditions: magnitudes and trigonometric identity -/
def magnitudes_eq_R (OA OB OC : R) : Prop := norm OA = R ∧ norm OB = R ∧ norm OC = R
def trig_identity (α β γ : ℝ) : Prop := cosα^2 + cosβ^2 + cosγ^2 = 1 - 2 * (cosα * cosβ * cosγ)

/-- The proof goal -/
theorem OH_squared (OA OB OC : ℝ) (H_mag : magnitudes_eq_R OA OB OC) (H_trig : trig_identity α β γ) :
  OH^2 = R^2 * (1 - 8 * cosα * cosβ * cosγ) :=
sorry

end OH_squared_l624_624048


namespace interest_rate_difference_l624_624537

theorem interest_rate_difference (Principal : ℝ) (Time : ℝ) (InterestDiff : ℝ) (dR: ℝ) : 
  Principal = 2500 → 
  Time = 3 → 
  InterestDiff = 75 → 
  7500 / 100 * dR = InterestDiff → 
  dR = 1 :=
by 
  intros h1 h2 h3 h4 
  rw [h1, h2] at h4 
  exact (eq_div_iff (by norm_num)).1 h4

end interest_rate_difference_l624_624537


namespace three_digit_whole_numbers_count_l624_624295

def digits := {4, 7, 9}

def three_digit_whole_numbers := { x | ∃ a b c : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = 100 * a + 10 * b + c }

-- Number of 3-digit whole numbers formed by the digits 4, 7, and 9 without repetition
theorem three_digit_whole_numbers_count : 
  Fintype.card three_digit_whole_numbers = 6 := 
  sorry

end three_digit_whole_numbers_count_l624_624295


namespace tylenol_pill_mg_l624_624352

noncomputable def tylenol_dose_per_pill : ℕ :=
  let mg_per_dose := 1000
  let hours_per_dose := 6
  let days := 14
  let pills := 112
  let doses_per_day := 24 / hours_per_dose
  let total_doses := doses_per_day * days
  let total_mg := total_doses * mg_per_dose
  total_mg / pills

theorem tylenol_pill_mg :
  tylenol_dose_per_pill = 500 := by
  sorry

end tylenol_pill_mg_l624_624352


namespace add_fractions_l624_624716

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624716


namespace kris_fraction_l624_624474

-- Definitions based on problem conditions
def Trey (kris : ℕ) := 7 * kris
def Kristen := 12
def Trey_kristen_diff := 9
def Kris_fraction_to_Kristen (kris : ℕ) : ℚ := kris / Kristen

-- Theorem statement: Proving the required fraction
theorem kris_fraction (kris : ℕ) (h1 : Trey kris = Kristen + Trey_kristen_diff) : 
  Kris_fraction_to_Kristen kris = 1 / 4 :=
by
  sorry

end kris_fraction_l624_624474


namespace number_of_possible_values_for_x0_l624_624803

noncomputable def sequence (x_0 : ℝ) (n : ℕ) : ℝ :=
  nat.rec_on n x_0 (λ n x_n_minus_1,
    if 2 * x_n_minus_1 < 1 then
      2 * x_n_minus_1
    else
      2 * x_n_minus_1 - 1)

theorem number_of_possible_values_for_x0 :
  {x_0 : ℝ // 0 ≤ x_0 ∧ x_0 < 1 ∧ sequence x_0 6 = x_0}.card = 63 :=
by sorry

end number_of_possible_values_for_x0_l624_624803


namespace range_of_a_l624_624833

noncomputable def f (x : ℝ) : ℝ := 
if x ∈ Icc (-1 : ℝ) 3 then 
  if x < -1 then (x+2)^2 
  else if x ≤ 0 then x^2 
  else if x ≤ 1 then x^2 
  else (x-2)^2
else -1/f (x-2)

def g (x a : ℝ) : ℝ := f x - Real.log x⁺

theorem range_of_a (a : ℝ) :
(∀ x ∈ Icc (-1 : ℝ) 3, g x a = 0) ↔ a ∈ Icc (5 : ℝ) (Real.top) := sorry

end range_of_a_l624_624833


namespace domain_translation_l624_624264

theorem domain_translation (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < 3 * x + 2 ∧ 3 * x + 2 < 1 → (∃ y : ℝ, f (3 * x + 2) = y)) →
  (∀ x : ℝ, ∃ y : ℝ, f (2 * x - 1) = y ↔ (3 / 2) < x ∧ x < 3) :=
sorry

end domain_translation_l624_624264


namespace number_of_terms_is_five_l624_624902

variable (a_n : ℕ → ℕ) -- Define the geometric sequence {a_n}

/-- Define the conditions given in the problem -/
axiom condition1 : ∃ n, a_n 1 + a_n n = 82
axiom condition2 : ∃ n, a_n 3 * a_n (n - 2) = 81
axiom condition3 : ∃ n, ∑ i in range (n+1), a_n i = 121

/-- The goal is to prove the number of terms n is 5 -/
theorem number_of_terms_is_five : ∃ n, a_n 1 + a_n n = 82 ∧ a_n 3 * a_n (n - 2) = 81 ∧ (∑ i in range (n+1), a_n i = 121) ∧ n = 5 := 
sorry -- Proof is not required

end number_of_terms_is_five_l624_624902


namespace piles_have_each_suit_l624_624520

def initial_deck : List Nat := ([0, 1, 2, 3] : List Nat).repeat (36 / 4)

def possible_transformed_deck (A' A'' : List Nat) : Prop := 
  ∃ B : List Nat, 
    (B.perm (A' ++ A'')) ∧
    (∀ i < (36 / 4),
     let pile := B.drop (4 * i) |>.take 4
     in (pile.nodup ∧ pile.toFinset = {0, 1, 2, 3}.toFinset))

theorem piles_have_each_suit (A : List Nat) (h : A = initial_deck) :
  ∀ A' A'' : List Nat,
  (A' ++ A'') ⊆ A → (A'.length + A''.length = 36) →
  (possible_transformed_deck A' A'') :=
sorry

end piles_have_each_suit_l624_624520


namespace tangent_line_at_one_l624_624807

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp 1

theorem tangent_line_at_one : ∀ (x : ℝ), (1 : ℝ) -> (f(1) = 0) -> (f'(x) = Real.exp x) ->
  (∀ y : ℝ, y = (Real.exp 1)*x - Real.exp 1) :=
by
  sorry

end tangent_line_at_one_l624_624807


namespace derivative_sequence_problem_solution_l624_624847

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

def f_seq : ℕ → (ℝ → ℝ)
| 1     := fun x => Real.cos x - Real.sin x
| 2     := fun x => -Real.sin x - Real.cos x
| 3     := fun x => -Real.cos x + Real.sin x
| 4     := f
| (n+4) := f_seq (n+1)

theorem derivative_sequence (n : ℕ) :
  f_seq (4 * n + 1) = fun x => Real.cos x - Real.sin x ∧
  f_seq (4 * n + 2) = fun x => -Real.sin x - Real.cos x ∧
  f_seq (4 * n + 3) = fun x => -Real.cos x + Real.sin x ∧
  f_seq (4 * n + 4) = f := sorry

theorem problem_solution : f_seq 2016 (Real.pi / 3) = (Real.sqrt 3 + 1) / 2 := by
  have h1 : 2016 = 4 * 504 := by norm_num
  rw [h1, add_zero]
  have h2 : f_seq 2016 = f := by exact (derivative_sequence 504).right.right.right
  rw [h2]
  unfold f
  rw [Real.sin_pi_div_three, Real.cos_pi_div_three]
  norm_num

end derivative_sequence_problem_solution_l624_624847


namespace general_term_a_sum_b_n_l624_624926

noncomputable def S (n : ℕ) : ℝ := sorry
noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))

axiom a_pos (n : ℕ) : a n > 0
axiom a_relation (n : ℕ) : a n ^ 2 + 2 * a n = 4 * S n + 3

theorem general_term_a : ∀ n : ℕ, a n = 2 * n + 1 := sorry

theorem sum_b_n (n : ℕ) : ∑ k in finset.range n, b k = n / (3 * (2 * n + 3)) := sorry

end general_term_a_sum_b_n_l624_624926


namespace fixed_point_existence_l624_624247

noncomputable def ellipse_standard_eq (a b : ℝ) (h1 : a > b > 0) (e : ℝ) (h_e : e = 1 / 2) (d : ℝ) (h_d : d = 2) : Prop :=
  let c := e * a in
  let c_squared := c^2 in
  let b_squared := a^2 - c^2 in
  ∃ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def line_fixed_point (k m a b : ℝ) (h : k ≠ 0) (h1 : a = 2) (h2 : b^2 = 3) : Prop :=
  ∃ x y : ℝ, 
  let ellipse := ∃ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 in
  let intersect_points := ∃ A B : ℝ×ℝ, 
    (A.1, A.2) ≠ (B.1, B.2) ∧ 
    (A.1, A.2) ∈ ellipse ∧ (B.1, B.2) ∈ ellipse ∧ 
    ∃ line_eq : ℝ, line_eq = (y = k * x + m) in
  ∃ D : ℝ × ℝ, (2,0) ∈ ellipse ∧ 
  let circle_with_AB_diameter := ∃ circle : ℝ → ℝ → Prop, circle D A B in
  ∀ l : ℝ → ℝ → Prop, l passes through fixed point (2 / 7, 0)

theorem fixed_point_existence (a b : ℝ) (h1 : a > b > 0) (e : ℝ) (h_e : e = 1 / 2) (d : ℝ) (h_d : d = 2)
  (k m : ℝ) (h : k ≠ 0) : 
  (ellipse_standard_eq a b h1 e h_e d h_d) 
  → (line_fixed_point k m a b h h1 rfl) :=
sorry

end fixed_point_existence_l624_624247


namespace sum_fractions_eq_l624_624689

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624689


namespace fraction_addition_l624_624610

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624610


namespace largest_power_prime_2017_divides_factorial_l624_624752

theorem largest_power_prime_2017_divides_factorial : ∑ i in Finset.range (Nat.log 2017 2017 + 1), ⌊2017 / 2017 ^ i⌋ = 1 :=
by
  have h : ∀ i > 0, 2017 / 2017 ^ i = 0 := sorry
  have h0 : 2017 / 2017 ^ 0 = 1 := sorry
  rw sum_eq_single 0 (λ b hb hn, h b hn) (by simp),
  exact h0

end largest_power_prime_2017_divides_factorial_l624_624752


namespace ratio_of_areas_l624_624445

def side_length_S : ℝ := sorry
def longer_side_R : ℝ := 1.2 * side_length_S
def shorter_side_R : ℝ := 0.8 * side_length_S
def area_S : ℝ := side_length_S ^ 2
def area_R : ℝ := longer_side_R * shorter_side_R

theorem ratio_of_areas (side_length_S : ℝ) :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l624_624445


namespace fixed_point_a_zero_two_fixed_points_range_a_l624_624790

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log 2 (a * 4^(x - 1 / 2) - (a - 1) * 2^(x - 1) + a / 2 + 1 / 4)

theorem fixed_point_a_zero :
  ∃ x : ℝ, f 0 x = x ∧ x = -1 :=
sorry

theorem two_fixed_points_range_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f a x1 = x1 ∧ f a x2 = x2) →
  1/2 < a ∧ a < real.sqrt 3 / 3 :=
sorry

end fixed_point_a_zero_two_fixed_points_range_a_l624_624790


namespace sum_of_fractions_l624_624714

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624714


namespace sin_cos_product_l624_624899

noncomputable theory

-- Define the angles and their intersection points with the unit circle
variables (α β : ℝ)
variables (Pα : ℝ × ℝ) (Pβ : ℝ × ℝ)
variables (hPα : Pα = (12 / 13, 5 / 13)) (hPβ : Pβ = (-3 / 5, 4 / 5))

-- Main theorem stating the desired equivalence
theorem sin_cos_product (hα : sin α = 5 / 13) (hβ : cos β = -3 / 5) :
  sin α * cos β = -15 / 65 :=
by
  sorry

end sin_cos_product_l624_624899


namespace longest_side_of_enclosure_l624_624949

theorem longest_side_of_enclosure (l w : ℝ)
  (h_perimeter : 2 * l + 2 * w = 240)
  (h_area : l * w = 8 * 240) :
  max l w = 80 :=
by
  sorry

end longest_side_of_enclosure_l624_624949


namespace convert_polar_to_rect_l624_624185

theorem convert_polar_to_rect (ρ θ : ℝ) (x y : ℝ) 
  (h1 : x = ρ * cos θ) 
  (h2 : y = ρ * sin θ) 
  (h3 : 3 * ρ * cos θ + 4 * ρ * sin θ = 2) : 
  3 * x + 4 * y - 2 = 0 := 
by 
  sorry

end convert_polar_to_rect_l624_624185


namespace fraction_addition_l624_624624

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624624


namespace number_of_students_l624_624130

theorem number_of_students (n T : ℕ) (h1 : T = n * 90) 
(h2 : T - 120 = (n - 3) * 95) : n = 33 := 
by
sorry

end number_of_students_l624_624130


namespace Mike_picked_7_apples_l624_624036

def num_apples_Mike_picked (M : Real) : Prop :=
  ∃ (N : Real) (K : Real) (L : Real), 
    N = 3.0 ∧ 
    K = 6.0 ∧ 
    L = 10 ∧
    M - N + K = L

theorem Mike_picked_7_apples : num_apples_Mike_picked 7.0 :=
by
  unfold num_apples_Mike_picked
  use [3.0, 6.0, 10]
  simp
  sorry

end Mike_picked_7_apples_l624_624036


namespace intersection_complement_M_N_l624_624859

-- Definition of M
def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }

-- Definition of N
def N : Set ℝ := { x | 2^x < 2 }

-- Statement of the theorem
theorem intersection_complement_M_N :
  M ∩ (Set.univ \ N) = { x | 1 ≤ x ∧ x < 3 } := 
by 
  sorry  

end intersection_complement_M_N_l624_624859


namespace translate_line_downwards_l624_624080

theorem translate_line_downwards :
  ∀ (x : ℝ), (∀ (y : ℝ), (y = 2 * x + 1) → (y - 2 = 2 * x - 1)) :=
by
  intros x y h
  rw [h]
  sorry

end translate_line_downwards_l624_624080


namespace add_fractions_l624_624573

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624573


namespace total_books_is_correct_l624_624554

-- Definitions based on the conditions
def initial_books_benny : Nat := 24
def books_given_to_sandy : Nat := 10
def books_tim : Nat := 33

-- Definition based on the computation in the solution
def books_benny_now := initial_books_benny - books_given_to_sandy
def total_books : Nat := books_benny_now + books_tim

-- The statement to be proven
theorem total_books_is_correct : total_books = 47 := by
  sorry

end total_books_is_correct_l624_624554


namespace add_fractions_l624_624728

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624728


namespace probability_product_divisible_by_3_l624_624110

-- Define the problem setup
def die := {1, 2, 3, 4, 5, 6}

-- The event that a product is divisible by 3
def event_product_divisible_by_3 (rolls : list ℕ) : Prop :=
  (rolls.product % 3) = 0

-- The main theorem to prove
theorem probability_product_divisible_by_3 :
  probability (event_product_divisible_by_3 (rolls : list ℕ)) = 211 / 243 := 
sorry

end probability_product_divisible_by_3_l624_624110


namespace value_of_a6_in_arithmetic_sequence_l624_624335

/-- In the arithmetic sequence {a_n}, if a_2 and a_{10} are the two roots of the equation
    x^2 + 12x - 8 = 0, prove that the value of a_6 is -6. -/
theorem value_of_a6_in_arithmetic_sequence :
  ∃ a_2 a_10 : ℤ, (a_2 + a_10 = -12 ∧
  (2: ℤ) * ((a_2 + a_10) / (2 * 1)) = a_2 + a_10 ) → 
  ∃ a_6: ℤ, a_6 = -6 :=
by
  sorry

end value_of_a6_in_arithmetic_sequence_l624_624335


namespace max_sum_of_min_elements_l624_624463

noncomputable def sequence (k: ℕ) : ℕ → ℝ
| 2 => 1 / 2
| (n+3) => (1 / 2^(nat.log2 (n + 2)))

noncomputable def sum_sequence (n: ℕ) : ℝ :=
(∑ k in finset.range n \ finset.range 2, sequence (k + 2))

theorem max_sum_of_min_elements :
  sum_sequence 2024 = 1405 / 256 :=
by
  -- This is the mathematical equivalent of the given problem.
  -- Proof omitted.
  sorry

end max_sum_of_min_elements_l624_624463


namespace add_fractions_result_l624_624582

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624582


namespace number_of_students_is_20_l624_624990

-- Define the constants and conditions
def average_age_all_students (N : ℕ) : ℕ := 20
def average_age_9_students : ℕ := 11
def average_age_10_students : ℕ := 24
def age_20th_student : ℕ := 61

theorem number_of_students_is_20 (N : ℕ) 
  (h1 : N * average_age_all_students N = 99 + 240 + 61) 
  (h2 : 99 = 9 * average_age_9_students) 
  (h3 : 240 = 10 * average_age_10_students) 
  (h4 : N = 9 + 10 + 1) : N = 20 :=
sorry

end number_of_students_is_20_l624_624990


namespace sum_of_fractions_l624_624702

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624702


namespace vector_orthogonal_implies_m_value_l624_624802

theorem vector_orthogonal_implies_m_value :
  ∀ (m : ℝ), let a := (1, 2) 
             let b := (m, 1) in 
             (a.1 * b.1 + a.2 * b.2 = 0) → m = -2 := by
  sorry

end vector_orthogonal_implies_m_value_l624_624802


namespace calculation_l624_624562

theorem calculation : Real.floor (Real.abs (-5.7)) + Real.abs (Real.floor (-5.7)) = 11 := by
  sorry

end calculation_l624_624562


namespace Aarti_work_completion_time_l624_624543

-- Define the conditions
def three_times_days := 27
def Aarti_work_days := 3 * x

-- State the theorem
theorem Aarti_work_completion_time (x : ℕ) : 3 * x = three_times_days → x = 9 :=
by
  sorry

end Aarti_work_completion_time_l624_624543


namespace dan_marbles_l624_624186

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) : 
  original_marbles = 64 ∧ given_marbles = 14 → remaining_marbles = 50 := 
by 
  sorry

end dan_marbles_l624_624186


namespace sum_of_squares_increased_l624_624822

theorem sum_of_squares_increased (x : Fin 100 → ℝ) 
  (h : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  ∑ i, (x i + 4) ^ 2 = ∑ i, x i ^ 2 + 800 := 
by
  sorry

end sum_of_squares_increased_l624_624822


namespace orthocenters_collinear_l624_624151

-- Definitions of the geometric conditions
variables {A B C D M X Y Z Hx Hy Hz : Type} 

-- Assume square and M on BC
variables {Square ABCD : Prop} (hSquare : Square A B C D)
variables {M_on_BC : Prop} (hM : M_on_BC M B C)

-- Definitions of the incenters
variables {Incenter_ABM : X -> Prop} (hX : Incenter_ABM X A B M)
variables {Incenter_CMD : Y -> Prop} (hY : Incenter_CMD Y C M D)
variables {Incenter_AMD : Z -> Prop} (hZ : Incenter_AMD Z A M D)

-- Definitions of the orthocenters
variables {Orthocenter_AXB : Hx -> Prop} (hHx : Orthocenter_AXB Hx A X B)
variables {Orthocenter_CYD : Hy -> Prop} (hHy : Orthocenter_CYD Hy C Y D)
variables {Orthocenter_AZD : Hz -> Prop} (hHz : Orthocenter_AZD Hz A Z D)

-- Theorem statement
theorem orthocenters_collinear (hSquare : Square A B C D) 
    (hM : M_on_BC M B C) 
    (hX : Incenter_ABM X A B M) 
    (hY : Incenter_CMD Y C M D) 
    (hZ : Incenter_AMD Z A M D) 
    (hHx : Orthocenter_AXB Hx A X B) 
    (hHy : Orthocenter_CYD Hy C Y D) 
    (hHz : Orthocenter_AZD Hz A Z D) : 
    Collinear Hx Hy Hz := 
sorry

end orthocenters_collinear_l624_624151


namespace fibonacci_mod_150_eq_8_l624_624428

def fibonacci : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fibonacci n + fibonacci (n + 1)

theorem fibonacci_mod_150_eq_8 :
  (fibonacci 150) % 9 = 8 := by
  sorry

end fibonacci_mod_150_eq_8_l624_624428


namespace oliver_needs_shelves_l624_624396

theorem oliver_needs_shelves (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) (books_left : ℕ) (shelves_needed : ℕ) : 
  total_books = 46 → 
  books_taken = 10 → 
  books_per_shelf = 4 → 
  books_left = total_books - books_taken → 
  shelves_needed = books_left / books_per_shelf → 
  shelves_needed = 9 :=
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [← h5]
  linarith
  sorry

end oliver_needs_shelves_l624_624396


namespace volume_pyramid_SO1BO2_l624_624831

variables {a b : ℝ} {α : ℝ}
variables (R : ℝ := a / (2 * Real.cos α))

theorem volume_pyramid_SO1BO2 (h : SABC_is_regular_pyramid) :
  volume (pyramid SO1 BO2) = (a^2 * b * Real.sin α) / (12 * (Real.cos α)^2) := sorry

end volume_pyramid_SO1BO2_l624_624831


namespace total_assembly_time_l624_624944

def chairs := 2
def tables := 2
def bookshelf := 1
def tv_stand := 1

def time_per_chair := 8
def time_per_table := 12
def time_per_bookshelf := 25
def time_per_tv_stand := 35

theorem total_assembly_time : (chairs * time_per_chair) + (tables * time_per_table) + (bookshelf * time_per_bookshelf) + (tv_stand * time_per_tv_stand) = 100 := by
  sorry

end total_assembly_time_l624_624944


namespace increasing_interval_of_f_on_0_pi_l624_624283

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 4)

theorem increasing_interval_of_f_on_0_pi {ω : ℝ} (hω : ω > 0)
  (h_symmetry : ∀ x, f ω x = g x) :
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi ∧ ∀ x1 x2, (0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi) → f ω x1 < f ω x2} = 
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi / 8} :=
sorry

end increasing_interval_of_f_on_0_pi_l624_624283


namespace surface_area_of_sphere_l624_624431

-- Noncomputable theory because of real numbers
noncomputable theory

-- Definitions of the conditions
def area_of_section_is_pi : Prop :=
  ∃ O : Type*,
  ∃ r : ℝ, 
  ∃ α : ℝ → ℝ → ℝ, 
  (π = π * r^2) ∧ (r = 1)

def distance_center_to_plane_is_sqrt15 (d : ℝ) : Prop :=
  d = Real.sqrt 15
  
-- The theorem to prove the surface area of the sphere
theorem surface_area_of_sphere (d : ℝ) 
  (h1: area_of_section_is_pi)
  (h2: distance_center_to_plane_is_sqrt15 d) : 
  ∃ s : ℝ, s = 64 * π :=
sorry

end surface_area_of_sphere_l624_624431


namespace people_at_the_beach_l624_624954

-- Conditions
def initial : ℕ := 3  -- Molly and her parents
def joined : ℕ := 100 -- 100 people joined at the beach
def left : ℕ := 40    -- 40 people left at 5:00

-- Proof statement
theorem people_at_the_beach : initial + joined - left = 63 :=
by
  sorry

end people_at_the_beach_l624_624954


namespace max_prob_min_prob_even_min_prob_odd_l624_624868

-- Definitions corresponding to the conditions
def total_books (n : ℕ) := n
def favorite_books (k n : ℕ) := k ≤ n

-- Function to calculate the probability of k favorite books ending up next to each other.
noncomputable def P (n k : ℕ) : ℚ :=
(k.factorial * (n - k + 1).factorial) / n.factorial

-- Maximum probability: k=1 or k=n
theorem max_prob (n : ℕ) (h₁ : 1 ≤ n) : P n 1 = 1 ∧ P n n = 1 :=
sorry

-- Minimum probability
theorem min_prob_even (n : ℕ) (h₁ : n % 2 = 0) : P n (n / 2) =
min (λ k, P n k) :=
sorry

theorem min_prob_odd (n : ℕ) (h₁ : n % 2 = 1) : P n ((n + 1) / 2) =
min (λ k, P n k) :=
sorry

end max_prob_min_prob_even_min_prob_odd_l624_624868


namespace total_dolls_count_l624_624168

-- Define the conditions
def big_box_dolls : Nat := 7
def small_box_dolls : Nat := 4
def num_big_boxes : Nat := 5
def num_small_boxes : Nat := 9

-- State the theorem that needs to be proved
theorem total_dolls_count : 
  big_box_dolls * num_big_boxes + small_box_dolls * num_small_boxes = 71 := 
by
  sorry

end total_dolls_count_l624_624168


namespace fraction_addition_l624_624620

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624620


namespace trig_identity_proof_l624_624898

theorem trig_identity_proof (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (h1 : sin α = 4 / 5) (h2 : cos α = 3 / 5) :
  let lhs := (sin (2 * π + α) * cos (π + α) * tan (3 * π - α)) / (cos (π / 2 - α) * tan (-π - α)) in
  lhs = -3 / 5 := 
by {
  sorry,
}

end trig_identity_proof_l624_624898


namespace tom_tim_typing_ratio_l624_624497

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) :
  M / T = 5 :=
by
  -- Proof to be completed
  sorry

end tom_tim_typing_ratio_l624_624497


namespace ratio_of_areas_l624_624447

theorem ratio_of_areas (s : ℝ) 
  (h1 : ∀ (s : ℝ), s > 0) : 
  let R_long := 1.2 * s,
      R_short := 0.8 * s,
      area_R := R_long * R_short,
      area_S := s^2
  in area_R / area_S = 24 / 25 :=
by
  let R_long := 1.2 * s
  let R_short := 0.8 * s
  let area_R := R_long * R_short
  let area_S := s^2
  have h2 : s > 0 := h1 s
  have h3 : area_R = 0.96 * s^2 := by sorry
  have h4 : area_R / area_S = 0.96 := by sorry
  have h5 : 0.96 = 24 / 25 := by norm_num
  exact eq.trans h4 h5

end ratio_of_areas_l624_624447


namespace max_heaps_of_stones_l624_624016

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624016


namespace parabola_and_midpoint_trajectory_and_distance_l624_624835

theorem parabola_and_midpoint_trajectory_and_distance :
  (∃ p : ℝ, p > 0 ∧ ∀ (x y : ℝ), (2, 1) ∈ C_1(x^2 = 2 * p * y) ↔ x^2 = 4 * y) ∧
  (∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ), x₁^2 = 4 * y₁ ∧ x₂^2 = 4 * y₂ →
    (x₁ - x₂) * (x₁ + x₂) = 4 * (y₁ - y₂) →
    ∀ (y₀ : ℝ), (y₀ - 2) / (x₀ / 2) = x₀ →
      x₀^2 = 2 * y₀ - 4 →
      ∀ (x y : ℝ), x^2 = 2 * y - 4) ∧
  (∀ (x₃ y₃ x₄ y₄ : ℝ), ((x * x₃ - 2 * y - 2 * y₃ = 0) ∧ (x * x₄ - y - y₄ + 4 = 0)) →
    ∃ d : ℝ, d = (x₃^2 + 4) / (2 * sqrt (x₃^2 + 1)) ∧ d >= sqrt 3) :=
begin
  split,
  { use 2,
    split,
    { norm_num, },
    { intros x y,
      simp only [eq_self_iff_true, mem_set_of_eq, iff_true],
      intro h,
      calc x^2 = 2 * 2 * y : by rw h
         ... = 4 * y : by ring, } },
  split,
  { intros x₁ y₁ x₂ y₂ x₀ y₀ h₁ h₂ y₀ h₃,
    rw [← h₃, ← sq, ← sub_eq_zero] at *,
    calc x₀^2 = 2 * y₀ - 4 : by assumption },
  { intros x₃ y₃ x₄ y₄ h₄,
    use sqrt(3),
    split,
    { sorry, },
    { sorry, }, }
end

end parabola_and_midpoint_trajectory_and_distance_l624_624835


namespace simplify_expr1_simplify_expr2_l624_624754

-- Definition for the expression (2x - 3y)²
def expr1 (x y : ℝ) : ℝ := (2 * x - 3 * y) ^ 2

-- Theorem to prove that (2x - 3y)² = 4x² - 12xy + 9y²
theorem simplify_expr1 (x y : ℝ) : expr1 x y = 4 * (x ^ 2) - 12 * x * y + 9 * (y ^ 2) := 
sorry

-- Definition for the expression (x + y) * (x + y) * (x² + y²)
def expr2 (x y : ℝ) : ℝ := (x + y) * (x + y) * (x ^ 2 + y ^ 2)

-- Theorem to prove that (x + y) * (x + y) * (x² + y²) = x⁴ + 2x²y² + y⁴ + 2x³y + 2xy³
theorem simplify_expr2 (x y : ℝ) : expr2 x y = x ^ 4 + 2 * (x ^ 2) * (y ^ 2) + y ^ 4 + 2 * (x ^ 3) * y + 2 * x * (y ^ 3) := 
sorry

end simplify_expr1_simplify_expr2_l624_624754


namespace real_part_of_complex_l624_624846

theorem real_part_of_complex (a : ℝ) (h : a^2 + 2 * a - 15 = 0 ∧ a + 5 ≠ 0) : a = 3 :=
by sorry

end real_part_of_complex_l624_624846


namespace sum_of_fractions_l624_624707

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624707


namespace parabola_max_area_solution_l624_624985

noncomputable def parabola_max_area (a : ℝ) : Prop :=
  ∃ c : ℝ, a > 0 ∧ (∀ x : ℝ, x ≠ 0 → (a * x^2 + c = 1 - |x|)) ∧
    (∃ (x0 : ℝ), x0 = -1 / (2 * a) ∧ (c = 1 + (1 / (4 * a)))) ∧
    (∫ x in (-(1 / (2 * a))), (1 / (2 * a)), a * x^2 + 1 + (1 / (4 * a)) dx = 
    2 * (-(1 / (6 * a^2)) - (1 / (2 * a)))) ∧ ...

theorem parabola_max_area_solution : ∃ a : ℝ, ∃ c : ℝ, c > 0 ∧  a = 1 ∧ c = 3 / 4 ∧ 
  (parabola_max_area a) ( y = -x^2 + \frac{3}{4} ) :=
begin
  sorry,
end

end parabola_max_area_solution_l624_624985


namespace fraction_addition_l624_624626

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624626


namespace add_fractions_result_l624_624590

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624590


namespace max_heaps_l624_624024

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624024


namespace simplify_polynomial_l624_624977

variable {x : ℝ} -- Assume x is a real number

theorem simplify_polynomial :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 6 * x - 15) = x^2 + 2 * x + 10 :=
sorry

end simplify_polynomial_l624_624977


namespace f_is_monotonically_decreasing_on_interval_f_maximum_value_on_interval_f_minimum_value_on_interval_l624_624280

-- Define the function y = (3 + x) / (x - 2) on the interval [3, 6]
def f (x : ℝ) : ℝ := (3 + x) / (x - 2)

-- Define the interval [3, 6]
def is_in_interval (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 6

-- 1. Prove monotonicity: The function f is monotonically decreasing on [3, 6].

theorem f_is_monotonically_decreasing_on_interval (x1 x2 : ℝ) (h1 : is_in_interval x1) (h2 : is_in_interval x2) (h : x1 < x2) : 
  f x1 ≥ f x2 :=
by
  sorry

-- 2. Maximum and minimum values of the function f on the interval [3, 6].

theorem f_maximum_value_on_interval : 
  f 3 = 6 :=
by
  sorry

theorem f_minimum_value_on_interval : 
  f 6 = 9 / 4 :=
by
  sorry

end f_is_monotonically_decreasing_on_interval_f_maximum_value_on_interval_f_minimum_value_on_interval_l624_624280


namespace definite_integral_cos_eq_zero_l624_624194

noncomputable def integral_cos_pi : Real :=
  ∫ x in (0 : Real)..π, Real.cos x

theorem definite_integral_cos_eq_zero : integral_cos_pi = 0 :=
by
  sorry

end definite_integral_cos_eq_zero_l624_624194


namespace additional_distance_traveled_l624_624040

theorem additional_distance_traveled (a b : ℝ) : (∃ extra_distance : ℝ, extra_distance = a + b) :=
by
  exists a + b
  sorry

end additional_distance_traveled_l624_624040


namespace max_heaps_660_l624_624004

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624004


namespace point_to_real_l624_624769

-- Condition: Real numbers correspond one-to-one with points on the number line.
def real_numbers_correspond (x : ℝ) : Prop :=
  ∃ (p : ℝ), p = x

-- Condition: Any real number can be represented by a point on the number line.
def represent_real_by_point (x : ℝ) : Prop :=
  real_numbers_correspond x

-- Condition: Conversely, any point on the number line represents a real number.
def point_represents_real (p : ℝ) : Prop :=
  ∃ (x : ℝ), x = p

-- Condition: The number represented by any point on the number line is either a rational number or an irrational number.
def rational_or_irrational (p : ℝ) : Prop :=
  (∃ q : ℚ, (q : ℝ) = p) ∨ (¬∃ q : ℚ, (q : ℝ) = p)

theorem point_to_real (p : ℝ) : represent_real_by_point p ∧ point_represents_real p ∧ rational_or_irrational p → real_numbers_correspond p :=
by sorry

end point_to_real_l624_624769


namespace range_x_f_leq_2_l624_624941

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2^(1 - x) else 1 - Real.log2 x

theorem range_x_f_leq_2 : {x : ℝ | f x ≤ 2} = {x : ℝ | 0 ≤ x} :=
by
  sorry

end range_x_f_leq_2_l624_624941


namespace fraction_addition_l624_624672

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624672


namespace monotonicity_f_max_g_on_pos_l624_624851

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  real.log (x + 1) - (a * x^2 + x) / (x + 1)^2

def g (x : ℝ) : ℝ :=
  x * real.log (1 + 1/x) + (1/x) * real.log (1 + x)

theorem monotonicity_f (a : ℝ) (h : 1 < a ∧ a ≤ 2) :
  ∀ (x : ℝ), if 1 < a ∧ a < 3/2 then
    (if x > -1 ∧ x < 2 * a - 3 then differentiable_increasing_on f (set.Ico (-1) (2 * a - 3))) ∧
    (differentiable_decreasing_on f (set.Ico (2 * a - 3) 0)) ∧
    (differentiable_increasing_on f (set.Ioi 0))
  else if a = 3/2 then
    differentiable_increasing_on f (set.Ioi (-1))
  else
    (differentiable_increasing_on f (set.Ico (-1) 0)) ∧
    (differentiable_decreasing_on f (set.Ico 0 (2 * a - 3))) ∧
    (differentiable_increasing_on f (set.Ioi (2 * a - 3))) :=
sorry

theorem max_g_on_pos (x : ℝ) (h : x > 0) : g x ≤ 2 * real.log 2 :=
sorry

end monotonicity_f_max_g_on_pos_l624_624851


namespace sum_of_fractions_l624_624708

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624708


namespace total_people_at_evening_l624_624951

def initial_people : ℕ := 3
def people_joined : ℕ := 100
def people_left : ℕ := 40

theorem total_people_at_evening : initial_people + people_joined - people_left = 63 := by
  sorry

end total_people_at_evening_l624_624951


namespace solution_is_correct_l624_624301

-- Define the options
inductive Options
| A_some_other
| B_someone_else
| C_other_person
| D_one_other

-- Define the condition as a function that returns the correct option
noncomputable def correct_option : Options :=
Options.B_someone_else

-- The theorem stating that the correct option must be the given choice
theorem solution_is_correct : correct_option = Options.B_someone_else :=
by
  sorry

end solution_is_correct_l624_624301


namespace fraction_addition_l624_624618

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624618


namespace min_staff_members_l624_624340

theorem min_staff_members
  (num_male_students : ℕ)
  (num_benches_3_students : ℕ)
  (num_benches_4_students : ℕ)
  (num_female_students : ℕ)
  (total_students : ℕ)
  (total_seating_capacity : ℕ)
  (additional_seats_required : ℕ)
  (num_staff_members : ℕ)
  (h1 : num_female_students = 4 * num_male_students)
  (h2 : num_male_students = 29)
  (h3 : num_benches_3_students = 15)
  (h4 : num_benches_4_students = 14)
  (h5 : total_seating_capacity = 3 * num_benches_3_students + 4 * num_benches_4_students)
  (h6 : total_students = num_male_students + num_female_students)
  (h7 : additional_seats_required = total_students - total_seating_capacity)
  (h8 : num_staff_members = additional_seats_required)
  : num_staff_members = 44 := 
sorry

end min_staff_members_l624_624340


namespace complement_of_union_l624_624366

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union (hU : U = {1, 2, 3, 4}) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  (U \ (M ∪ N)) = {4} :=
by
  sorry

end complement_of_union_l624_624366


namespace range_of_sum_is_correct_l624_624850

def f (x : ℝ) : ℝ :=
if x ≤ 0 then -2 * x - x^2 else abs (Real.log x)

theorem range_of_sum_is_correct (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : f a = f b) (h5 : f b = f c) (h6 : f c = f d) :
  1 < a + b + c + 2 * d ∧ a + b + c + 2 * d < 181 / 10 :=
sorry

end range_of_sum_is_correct_l624_624850


namespace sum_of_digits_of_y_coordinate_of_C_l624_624923

theorem sum_of_digits_of_y_coordinate_of_C :
  ∃ A B C : ℝ × ℝ, 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  A.2 = A.1^2 ∧ B.2 = B.1^2 ∧ C.2 = C.1^2 ∧
  (A.2 = B.2) ∧ -- Line AB parallel to x-axis
  ((∃ m n : ℝ, m ≠ n ∧ A = (m, m^2) ∧ B = (n, n^2) ∧ C = ((m+n)/2, ((m+n)/2)^2)) ∨
   (∃ m n : ℝ, m ≠ n ∧ A = ((m+n)/2, ((m+n)/2)^2) ∧ B = (m, m^2) ∧ C = (n, n^2)) ∨
   (∃ m n : ℝ, m ≠ n ∧ A = (n, n^2) ∧ B = ((m+n)/2, ((m+n)/2)^2) ∧ C = (m, m^2))) ∧
   (1/2 * abs (A.1 - B.1) * abs (A.2 - C.2) = 2016) ∧ -- Area condition
  (∃ y_C : ℕ, y_C = floor (C.2) ∧ (y_C.digits.sum = 26)) :=
sorry

end sum_of_digits_of_y_coordinate_of_C_l624_624923


namespace add_fractions_l624_624717

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624717


namespace trapezoid_median_l624_624996

theorem trapezoid_median {BC AD : ℝ} (h AC CD : ℝ) (h_nonneg : h = 2) (AC_eq_CD : AC = 4) (BC_eq_0 : BC = 0) 
: (AD = 4 * Real.sqrt 3) → (median = 3 * Real.sqrt 3) := by
  sorry

end trapezoid_median_l624_624996


namespace add_fractions_result_l624_624593

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624593


namespace max_heaps_660_stones_l624_624026

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624026


namespace trajectory_of_centroid_l624_624409

def foci (F1 F2 : ℝ × ℝ) : Prop := 
  F1 = (0, 1) ∧ F2 = (0, -1)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 3) + (P.2^2 / 4) = 1

def centroid_eq (G : ℝ × ℝ) : Prop :=
  ∃ P : ℝ × ℝ, on_ellipse P ∧ 
  foci (0, 1) (0, -1) ∧ 
  G = (P.1 / 3, (1 + -1 + P.2) / 3)

theorem trajectory_of_centroid :
  ∀ G : ℝ × ℝ, (centroid_eq G → 3 * G.1^2 + (9 * G.2^2) / 4 = 1 ∧ G.1 ≠ 0) :=
by 
  intros G h
  sorry

end trajectory_of_centroid_l624_624409


namespace vector_addition_example_l624_624216

theorem vector_addition_example :
  (⟨-3, 2, -1⟩ : ℝ × ℝ × ℝ) + (⟨1, 5, -3⟩ : ℝ × ℝ × ℝ) = ⟨-2, 7, -4⟩ :=
by
  sorry

end vector_addition_example_l624_624216


namespace problem_l624_624269

variable x : ℝ
variable a : ℝ := Real.log x / Real.log 3
variable b : ℝ := Real.sin x
variable c : ℝ := 2 ^ x

theorem problem (h₀ : 0 < x ∧ x < 1) : a < b ∧ b < c := by
  -- here we would provide the proof steps, but for now we use sorry to skip it
  sorry

end problem_l624_624269


namespace geom_seq_common_ratio_l624_624268

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {q : ℝ}

-- Given: The sum of the first n terms of a geometric sequence {a_n} is S_n, and S_3 + S_6 = 0
-- Prove: q = -∛2 (the common ratio of the sequence)
theorem geom_seq_common_ratio (hS₃₆ : S_3 + S_6 = 0) (hseq : S_6 = S_3 + q^3 * S_3) (hS₃_ne_zero : S_3 ≠ 0) :
  q = -real.cbrt 2 :=
sorry

end geom_seq_common_ratio_l624_624268


namespace max_heaps_of_stones_l624_624012

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l624_624012


namespace sum_of_distances_independent_of_square_position_l624_624530

open EuclideanGeometry

variable {M O : Point}
variable {r : ℝ}
variable (A B C D : Point)

noncomputable def is_square_inscribed_in_circle (ABCD : (Point × Point × Point × Point)) (O : Point) (r : ℝ) : Prop :=
  let ⟨A, B, C, D⟩ := ABCD in
  Circle O r ∧
  (dist O A = r) ∧ (dist O B = r) ∧ (dist O C = r) ∧ (dist O D = r) ∧
  (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D A)

noncomputable def sum_of_fourth_powers_of_distances (M A B C D : Point) : ℝ :=
  (dist M A)^4 + (dist M B)^4 + (dist M C)^4 + (dist M D)^4

theorem sum_of_distances_independent_of_square_position
  {M O : Point} {r : ℝ}
  (ABCD : Point × Point × Point × Point)
  (h_inscribed : is_square_inscribed_in_circle ABCD O r) :
  ∃ s, sum_of_fourth_powers_of_distances M A B C D = s :=
by
  cases ABCD with A B C D
  apply exists.intro (4 * ((dist O M)^2 + r^2)),
  sorry

end sum_of_distances_independent_of_square_position_l624_624530


namespace sum_of_inscribed_angles_of_pentagon_l624_624528

theorem sum_of_inscribed_angles_of_pentagon (arc_sum : ℝ) 
  (inscribed_angle_theorem : ∀ (arc : ℝ), arc / 2): 
  arc_sum = 360 → 
  ∑ (i : fin 5), inscribed_angle_theorem 72 = 180 :=
by
  intro h1
  have h2 : ∑ (i : fin 5), 72 / 2 = 5 * 36 := by sorry
  rw [h2]
  norm_num

end sum_of_inscribed_angles_of_pentagon_l624_624528


namespace add_fractions_l624_624572

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624572


namespace monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l624_624848

-- Proof Problem I
noncomputable def f1 (x : ℝ) := x^2 + x - Real.log x

theorem monotonic_intervals_a1 : 
  (∀ x, 0 < x ∧ x < 1 / 2 → f1 x < 0) ∧ (∀ x, 1 / 2 < x → f1 x > 0) := 
sorry

-- Proof Problem II
noncomputable def f2 (x : ℝ) (a : ℝ) := x^2 + a * x - Real.log x

theorem decreasing_on_1_to_2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f2 x a ≤ 0) → a ≤ -7 / 2 :=
sorry

-- Proof Problem III
noncomputable def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_minimum_value :
  ∃ a : ℝ, (∀ x, 0 < x ∧ x ≤ Real.exp 1 → g x a = 3) ∧ a = Real.exp 2 :=
sorry

end monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l624_624848


namespace fourth_student_seat_number_l624_624884

theorem fourth_student_seat_number :
  ∃ (n : ℕ), n = 19 ∧ 
  ∀ (u v w x y z : ℕ) (students_sampled : set ℕ), 
  52 = u ∧ 4 = v ∧ 13 = w ∧ {6, 32, 45} ⊂ students_sampled →
  students_sampled = {6, 19, 32, 45} :=
by
  sorry

end fourth_student_seat_number_l624_624884


namespace points_on_unit_circle_and_angles_imply_distance_condition_l624_624918

theorem points_on_unit_circle_and_angles_imply_distance_condition
  {P A B C : ℝ × ℝ × ℝ} (hP : P.2 ≠ 0) 
  (hA : A.2 = 0) (hB : B.2 = 0) (hC : C.2 = 0)
  (hA_on_circle : A.1.1^2 + A.1.2^2 = 1)
  (hB_on_circle : B.1.1^2 + B.1.2^2 = 1)
  (hC_on_circle : C.1.1^2 + C.1.2^2 = 1)
  (h_angles : ∠APB = 90) (h_angles_2 : ∠APC = 90) (h_angles_3 : ∠BPC = 90) :
  P.1.1^2 + P.1.2^2 + 2 * P.2^2 = 1 := 
by sorry

end points_on_unit_circle_and_angles_imply_distance_condition_l624_624918


namespace sum_of_squares_change_l624_624813

def x : ℕ → ℝ := sorry
def y (i : ℕ) : ℝ := x i + 2
def z (i : ℕ) : ℝ := x i + 4

theorem sum_of_squares_change :
  (∑ j in Finset.range 100, (z j)^2) - (∑ j in Finset.range 100, (x j)^2) = 800 :=
by
  sorry

end sum_of_squares_change_l624_624813


namespace cows_eat_total_husk_l624_624888

theorem cows_eat_total_husk:
  ∀ (num_cows : ℕ) (num_days : ℕ) (husk_per_cow_per_days : ℕ),
  num_cows = 50 →
  num_days = 50 →
  husk_per_cow_per_days = 1 →
  num_cows * husk_per_cow_per_days = 50 := 
by 
  intros num_cows num_days husk_per_cow_per_days H1 H2 H3
  rw [H1, H3]
  exact rfl

#check cows_eat_total_husk

end cows_eat_total_husk_l624_624888


namespace complex_exponential_form_theta_eq_pi_div_3_l624_624489

theorem complex_exponential_form_theta_eq_pi_div_3:
  ∃ θ : ℝ, 1 + complex.I * √3 = 2 * complex.exp (complex.I * θ) ∧ θ = (π / 3) :=
sorry

end complex_exponential_form_theta_eq_pi_div_3_l624_624489


namespace time_to_cross_second_platform_l624_624538

-- Defining the conditions
def length_first_platform : ℝ := 170
def time_first_platform : ℝ := 15
def length_train : ℝ := 70
def length_second_platform : ℝ := 250

-- Proving the time to cross the second platform
theorem time_to_cross_second_platform : 
  let total_distance_first := length_train + length_first_platform in
  let speed := total_distance_first / time_first_platform in
  let total_distance_second := length_train + length_second_platform in
  total_distance_second / speed = 20 := by
  let total_distance_first := length_train + length_first_platform
  let speed := total_distance_first / time_first_platform
  let total_distance_second := length_train + length_second_platform
  have h1 : speed = 16 := by
    calc
      speed = total_distance_first / time_first_platform := rfl
      ... = (70 + 170) / 15 := rfl
      ... = 240 / 15 := rfl
      ... = 16 := by norm_num
  show total_distance_second / speed = 20 from by
    calc
      total_distance_second / speed = (70 + 250) / speed := rfl
      ... = 320 / 16 := by rw h1
      ... = 20 := by norm_num

end time_to_cross_second_platform_l624_624538


namespace sum_of_x_coordinates_of_solutions_l624_624788

theorem sum_of_x_coordinates_of_solutions :
  (∑ x in {x : ℝ | | x ^ 2 - 4 * x + 3 | = 7 - 2 * x}, x) = 2 := 
by
  sorry

end sum_of_x_coordinates_of_solutions_l624_624788


namespace even_number_of_irreducible_fractions_l624_624411

def is_irreducible_proper_fraction (k n : ℕ) : Prop := 
  k > 0 ∧ k < n ∧ Nat.gcd k n = 1

def count_irreducible_proper_fractions (n : ℕ) : ℕ := 
  (Finset.range n).filter (λ k => is_irreducible_proper_fraction k n).card

theorem even_number_of_irreducible_fractions (n : ℕ) (h : n > 2) : 
  even (count_irreducible_proper_fractions n) := 
sorry

end even_number_of_irreducible_fractions_l624_624411


namespace iterated_f_five_times_l624_624312

def f (x : ℝ) : ℝ := -1 / x

theorem iterated_f_five_times {x : ℝ} (hx : x ≠ 0) : f (f (f (f (f x)))) = x :=
by
  sorry

end iterated_f_five_times_l624_624312


namespace days_for_Q_wages_l624_624154

variables (P Q S : ℝ) (D : ℝ)

theorem days_for_Q_wages (h1 : S = 24 * P) (h2 : S = 15 * (P + Q)) : S = D * Q → D = 40 :=
by
  sorry

end days_for_Q_wages_l624_624154


namespace add_fractions_l624_624664

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624664


namespace find_brown_mms_second_bag_l624_624051

variable (x : ℕ)

-- Definitions based on the conditions
def BrownMmsFirstBag := 9
def BrownMmsThirdBag := 8
def BrownMmsFourthBag := 8
def BrownMmsFifthBag := 3
def AveBrownMmsPerBag := 8
def NumBags := 5

-- Condition specifying the average brown M&Ms per bag
axiom average_condition : AveBrownMmsPerBag = (BrownMmsFirstBag + x + BrownMmsThirdBag + BrownMmsFourthBag + BrownMmsFifthBag) / NumBags

-- Prove the number of brown M&Ms in the second bag
theorem find_brown_mms_second_bag : x = 12 := by
  sorry

end find_brown_mms_second_bag_l624_624051


namespace face_value_of_shares_l624_624149

theorem face_value_of_shares (investment : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (dividend_received : ℝ) (F : ℝ)
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_received = 720) :
  (1.20 * F = investment) ∧ (0.06 * F = dividend_received) ∧ (F = 12000) :=
by
  sorry

end face_value_of_shares_l624_624149


namespace fraction_addition_l624_624650

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624650


namespace total_people_at_beach_l624_624957

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end total_people_at_beach_l624_624957


namespace sum_of_squares_change_l624_624817

def x : ℕ → ℝ := sorry
def y (i : ℕ) : ℝ := x i + 2
def z (i : ℕ) : ℝ := x i + 4

theorem sum_of_squares_change :
  (∑ j in Finset.range 100, (z j)^2) - (∑ j in Finset.range 100, (x j)^2) = 800 :=
by
  sorry

end sum_of_squares_change_l624_624817


namespace domain_and_range_of_f_l624_624998

noncomputable def f (x : ℝ) : ℝ := 
  Math.cos (2 * Real.arccos x) + 4 * Real.arcsin (Real.sin (x / 2))

theorem domain_and_range_of_f :
  ∀ x : ℝ, abs x ≤ 1 → -3 / 2 ≤ f x ∧ f x ≤ 3 :=
  sorry

end domain_and_range_of_f_l624_624998


namespace add_fractions_l624_624743

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624743


namespace add_fractions_l624_624734

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624734


namespace cylinder_not_uniquely_determined_l624_624976

theorem cylinder_not_uniquely_determined (S V : ℝ) (hS : S^3 > 54 * Real.pi * V^2) :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧
  let h1 := (S - 2 * Real.pi * r1^2) / (2 * Real.pi * r1)
      h2 := (S - 2 * Real.pi * r2^2) / (2 * Real.pi * r2)
  in (2 * Real.pi * r1 * h1 + 2 * Real.pi * r1^2 = S) ∧
     (2 * Real.pi * r2 * h2 + 2 * Real.pi * r2^2 = S) ∧
     (Real.pi * r1^2 * h1 = V) ∧
     (Real.pi * r2^2 * h2 = V) :=
begin
  sorry
end

end cylinder_not_uniquely_determined_l624_624976


namespace factorial_last_nonzero_digit_non_periodic_l624_624380

def last_nonzero_digit (n : ℕ) : ℕ :=
  -- function to compute last nonzero digit of n!
  sorry

def sequence_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a n = a (n + T)

theorem factorial_last_nonzero_digit_non_periodic : ¬ ∃ T, sequence_periodic last_nonzero_digit T :=
  sorry

end factorial_last_nonzero_digit_non_periodic_l624_624380


namespace quadratic_root_continued_fraction_l624_624970

theorem quadratic_root_continued_fraction (a b : ℤ) :
    (∃ (α β : ℚ), let cf := [a; b]^ in 
      (α = cf) ∧ (α + β = a) ∧ (α * β = -1) → β = -1 / (continued_fraction b a)) :=
sorry

end quadratic_root_continued_fraction_l624_624970


namespace total_dolls_l624_624173

-- Defining the given conditions as constants.
def big_boxes : Nat := 5
def small_boxes : Nat := 9
def dolls_per_big_box : Nat := 7
def dolls_per_small_box : Nat := 4

-- The main theorem we want to prove
theorem total_dolls : (big_boxes * dolls_per_big_box) + (small_boxes * dolls_per_small_box) = 71 :=
by
  rw [Nat.mul_add, Nat.mul_eq_mul, Nat.mul_eq_mul]
  exact sorry

end total_dolls_l624_624173


namespace range_of_a_l624_624840

def increasing {α : Type*} [Preorder α] (f : α → α) := ∀ x y, x ≤ y → f x ≤ f y

theorem range_of_a
  (f : ℝ → ℝ)
  (increasing_f : increasing f)
  (h_domain : ∀ x, 1 ≤ x ∧ x ≤ 5 → (f x = f x))
  (h_ineq : ∀ a, 1 ≤ a + 1 ∧ a + 1 ≤ 5 ∧ 1 ≤ 2 * a - 1 ∧ 2 * a - 1 ≤ 5 ∧ f (a + 1) < f (2 * a - 1)) :
  (2 : ℝ) < a ∧ a ≤ (3 : ℝ) := 
by
  sorry

end range_of_a_l624_624840


namespace find_k_l624_624860

-- Definitions of given vectors and the condition that the vectors are parallel.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Condition for vectors to be parallel in 2D is that their cross product is zero.
def parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_k : ∀ k : ℝ, parallel vector_a (vector_b k) → k = -2 :=
by
  intro k
  intro h
  sorry

end find_k_l624_624860


namespace sum_of_fractions_l624_624701

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624701


namespace inner_product_is_eight_l624_624304

noncomputable theory

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

def norm_eq_two (v : V) := ∥v∥ = 2
def norm_diff_eq_2sqrt2 := ∥a - b∥ = 2 * real.sqrt 2
def c_eq2a_b_plus_cross := c = 2 • a + b + 2 • (a × b)

theorem inner_product_is_eight
  (ha : norm_eq_two a)
  (hb : norm_eq_two b)
  (hdiff : norm_diff_eq_2sqrt2 a b)
  (hc : c_eq2a_b_plus_cross a b c) :
  inner a c = 8 :=
begin
  sorry
end

end inner_product_is_eight_l624_624304


namespace equilateral_triangle_y_coordinate_l624_624162

theorem equilateral_triangle_y_coordinate (x₁ y₁ x₂ y₂ : ℝ) (h₁ : x₁ = 3) (h₂ : y₁ = 7) (h₃ : x₂ = 13) (h₄ : y₂ = 7) :
  ∃ y₃ : ℝ, y₃ = 7 + 5 * Real.sqrt 3 :=
by
  use 7 + 5 * Real.sqrt 3
  sorry

end equilateral_triangle_y_coordinate_l624_624162


namespace real_when_m_eq_5_pure_imaginary_when_m_eq_3_or_m_eq_neg2_l624_624217

noncomputable def isReal (z : ℂ) : Prop :=
  z.im = 0

noncomputable def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0

def z (m : ℚ) : ℂ :=
  (m^2 - m - 6 : ℚ) / (m + 3 : ℚ) + (m^2 - 2*m - 15 : ℚ) * Complex.I

-- Prove that z is a real number when m = 5
theorem real_when_m_eq_5 : isReal (z (5 : ℚ)) :=
by sorry

-- Prove that z is a pure imaginary number when m = 3 or m = -2
theorem pure_imaginary_when_m_eq_3_or_m_eq_neg2 :
  isPureImaginary (z (3 : ℚ)) ∨ isPureImaginary (z ((-2) : ℚ)) :=
by sorry

end real_when_m_eq_5_pure_imaginary_when_m_eq_3_or_m_eq_neg2_l624_624217


namespace part1_part2_l624_624273

open Real Set

variables (p k x y x₁ x₂ y₁ y₂ : ℝ)
variable (A B : (ℝ × ℝ))
variable [fact (p > 0)] [fact (k ≠ 0)]

def parabola (x y : ℝ) : Prop := x ^ 2 = 2 * p * y
def line_through_focus (x y : ℝ) : Prop := y = k * x + p / 2

def points_A_B_on_parabola (A B : (ℝ × ℝ)) : Prop := 
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧ line_through_focus p k A.1 A.2 ∧ line_through_focus p k B.1 B.2

def tangent_slope (x : ℝ) : ℝ := x / p
def point_M (k p : ℝ) : (ℝ × ℝ) := (p * k, -p / 2)

theorem part1 (hA : parabola p A.1 A.2) (hB : parabola p B.1 B.2) (hLineA : line_through_focus p k A.1 A.2) (hLineB : line_through_focus p k B.1 B.2) :
  let A1 := A.1, B1 := B.1, A2 := A.2, B2 := B.2
  A1 * B1 + A2 * B2 = -3 / 4 * p^2 := by 
  sorry

theorem part2 (area_ACBD : ℝ) :
  (area_ACBD = 32 / 3 * p^2) → 
  (k = sqrt 3 ∨ k = -sqrt 3 ∨ k = sqrt 3 / 3 ∨ k = -sqrt 3 / 3) := by
  sorry

end part1_part2_l624_624273


namespace fraction_addition_l624_624637

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624637


namespace meena_cookies_left_l624_624947

def dozen : ℕ := 12

def baked_cookies : ℕ := 5 * dozen
def mr_stone_buys : ℕ := 2 * dozen
def brock_buys : ℕ := 7
def katy_buys : ℕ := 2 * brock_buys
def total_sold : ℕ := mr_stone_buys + brock_buys + katy_buys
def cookies_left : ℕ := baked_cookies - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end meena_cookies_left_l624_624947


namespace contractor_initial_plan_l624_624144

theorem contractor_initial_plan 
  (people1 : ℕ) (people2 : ℕ) (t1 t2 total_work : ℕ) 
  (fraction1 fraction2 : ℚ)
  (h_people1 : people1 = 10)
  (h_t1 : t1 = 20)
  (h_fraction1 : fraction1 = 1 / 4)
  (h_people2 : people2 = 8)
  (h_t2 : t2 = 75)
  (h_fraction2 : fraction2 = 3 / 4)
  (h_total_work : total_work = 800) :
  ∃ (D : ℕ), people1 * D = total_work ∧ D = 80 :=
by
  use 80
  split
  · sorry -- proof that 800 = 10 * 80
  · refl -- proof that D = 80

end contractor_initial_plan_l624_624144


namespace distance_between_foci_of_hyperbola_l624_624207

def hyperbola : Prop := 3 * x^2 - 18 * x - 2 * y^2 - 4 * y = 54

noncomputable def distance_between_foci (x y : ℝ) : ℝ :=
  2 * Real.sqrt (395 / 6)

theorem distance_between_foci_of_hyperbola {x y : ℝ} (h : hyperbola x y) : 
  distance_between_foci x y = Real.sqrt (1580 / 6) :=
sorry

end distance_between_foci_of_hyperbola_l624_624207


namespace seats_on_each_bus_l624_624455

-- Define the given conditions
def totalStudents : ℕ := 45
def totalBuses : ℕ := 5

-- Define what we need to prove - 
-- that the number of seats on each bus is 9
def seatsPerBus (students : ℕ) (buses : ℕ) : ℕ := students / buses

theorem seats_on_each_bus : seatsPerBus totalStudents totalBuses = 9 := by
  -- Proof to be filled in later
  sorry

end seats_on_each_bus_l624_624455


namespace add_fractions_l624_624721

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624721


namespace fraction_addition_l624_624598

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624598


namespace add_fractions_l624_624719

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624719


namespace problem1_problem2_problem3_property1_problem3_property2_l624_624771

-- Definition of generalized binomial coefficient
def generalized_binomial (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1
  else (x * (x - 1) * ... * (x - (m.to_nat - 1))) / (Nat.factorial m)

-- Problem 1: Calculate the value of C_{-15}^3
theorem problem1 : generalized_binomial (-15) 3 = -680 := sorry

-- Problem 2: For x > 0, find the value of x where C_x^3 / (C_x^1)^2 attains its minimum value
theorem problem2 (x : ℝ) (hx : 0 < x) : 
  (∃ y : ℝ, generalized_binomial y 3 / (generalized_binomial y 1)^(2 : ℕ) = 
  (1 / 6 : ℝ) * (y + 2 / y - 3) ∧ y = Real.sqrt 2) := sorry

-- Problem 3: Show which properties can be extended to generalized binomial coefficients for x ∈ ℝ and m as a positive integer
theorem problem3_property1 (x : ℝ) (m : ℕ) (hx_nonint : ¬(∃ n : ℝ, x = n)): 
  ¬(generalized_binomial x m = generalized_binomial x (x - m)) := sorry

theorem problem3_property2 (x : ℝ) (m : ℕ) 
  (hm_pos : 0 < m) : generalized_binomial x m + generalized_binomial x (m - 1) = generalized_binomial (x + 1) m := sorry

end problem1_problem2_problem3_property1_problem3_property2_l624_624771


namespace angle_sum_of_triangle_bisector_issue_l624_624419

-- Given trigonometric equations involving angles in a triangle
variable {α β : ℝ}

-- Prove that α + β = 90°
theorem angle_sum_of_triangle (h1 : sin α + cos β = 1) (h2 : cos α + sin β = 1) : α + β = 90 := by
  sorry

-- Given conditions of a triangle involving a bisector
variable {A B C D : Type} [LinearOrder A] {triangle : A → A → A → Prop} {bisector : A → A → Prop}

-- Prove that one side is greater than the other
theorem bisector_issue (h1 : triangle A B C) (h2 : bisector B D) (h3: ∠ADB > ∠CBD ∧ ∠ADB > ∠ABD) : A > D := by
  sorry 

end angle_sum_of_triangle_bisector_issue_l624_624419


namespace third_square_area_difference_l624_624084

def side_length (p : ℕ) : ℕ :=
  p / 4

def area (s : ℕ) : ℕ :=
  s * s

theorem third_square_area_difference
  (p1 p2 p3 : ℕ)
  (h1 : p1 = 60)
  (h2 : p2 = 48)
  (h3 : p3 = 36)
  : area (side_length p3) = area (side_length p1) - area (side_length p2) :=
by
  sorry

end third_square_area_difference_l624_624084


namespace sequence_sum_formula_l624_624285

noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

def a (n : ℕ) : ℕ := 2^(n-1) + 1

def S (n : ℕ) : ℕ := ∑ k in Finset.range (n+1), a (k+1) * C n k 

theorem sequence_sum_formula (n : ℕ) : 
  S n = 3^n + 2^n := by
  sorry

end sequence_sum_formula_l624_624285


namespace initial_ratio_alcohol_water_l624_624525

theorem initial_ratio_alcohol_water 
  (Alcohol : ℚ) (Water_added : ℚ) (new_ratio_num new_ratio_den initial_water : ℚ)
  (h_alcohol : Alcohol = 10)
  (h_water_added : Water_added = 10)
  (h_new_ratio : new_ratio_num = 2 ∧ new_ratio_den = 7)
  (h_new_ratio_condition : Alcohol / (initial_water + Water_added) = new_ratio_num / new_ratio_den) :
  Alcohol / initial_water = 2 / 5 := 
begin
  sorry,
end

end initial_ratio_alcohol_water_l624_624525


namespace holloway_soccer_team_l624_624163

theorem holloway_soccer_team (P M : Finset ℕ) (hP_union_M : (P ∪ M).card = 20) 
(hP : P.card = 12) (h_int : (P ∩ M).card = 6) : M.card = 14 := 
by
  sorry

end holloway_soccer_team_l624_624163


namespace maximum_ab_ac_bc_l624_624370

theorem maximum_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 5) : 
  ab + ac + bc ≤ 25 / 6 :=
sorry

end maximum_ab_ac_bc_l624_624370


namespace theta_of_complex_1_i_sqrt3_eq_pi_div3_l624_624484

noncomputable def theta_of_complex (re im : ℝ) : ℝ :=
  complex.arg (complex.mk re im)

theorem theta_of_complex_1_i_sqrt3_eq_pi_div3 :
  theta_of_complex 1 (real.sqrt 3) = real.pi / 3 := 
sorry

end theta_of_complex_1_i_sqrt3_eq_pi_div3_l624_624484


namespace fraction_addition_l624_624596

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624596


namespace roses_count_l624_624394

def total_roses : Nat := 80
def red_roses : Nat := 3 * total_roses / 4
def remaining_roses : Nat := total_roses - red_roses
def yellow_roses : Nat := remaining_roses / 4
def white_roses : Nat := remaining_roses - yellow_roses

theorem roses_count :
  red_roses + white_roses = 75 :=
by
  sorry

end roses_count_l624_624394


namespace add_fractions_l624_624737

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624737


namespace initial_overs_l624_624338

theorem initial_overs {x : ℝ} (h1 : 4.2 * x + (83 / 15) * 30 = 250) : x = 20 :=
by
  sorry

end initial_overs_l624_624338


namespace max_sections_with_4_lines_l624_624906

theorem max_sections_with_4_lines {R : Type} [is_rectangle R] (lines : list (line R)) (h : lines.length = 4) :
  max_sections R lines = 11 := 
sorry

end max_sections_with_4_lines_l624_624906


namespace find_positive_pairs_l624_624377

noncomputable def polynomial_factorization_exists (n : ℕ) (a b : ℝ) : Prop :=
  ∀ x : ℂ, x^2 + (a : ℂ) * x + (b : ℂ) = 0 → (a : ℂ) * x^(2 * n) + ((a : ℂ) * x + b)^(2 * n) = 0

theorem find_positive_pairs (n : ℕ) : 
  (∀ (a b : ℝ), (a > 0) → (b > 0) → ¬ polynomial_factorization_exists n a b) ↔ n = 1 ∨ (n ≥ 2 ∧ 
  ∃ k : ℕ, (n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n) ∧ 
  a = (2 * Real.cos ((2 * k + 1) * Real.pi / (2 * n)))^(Real.log 2 / (2 * Real.pi - 1)) ∧ 
  b = (2 * Real.cos ((2 * k + 1) * Real.pi / (2 * n)))^(2 / (k - 1))) :=
begin
  sorry
end

end find_positive_pairs_l624_624377


namespace fraction_addition_l624_624642

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624642


namespace sum_of_squares_change_l624_624815

def x : ℕ → ℝ := sorry
def y (i : ℕ) : ℝ := x i + 2
def z (i : ℕ) : ℝ := x i + 4

theorem sum_of_squares_change :
  (∑ j in Finset.range 100, (z j)^2) - (∑ j in Finset.range 100, (x j)^2) = 800 :=
by
  sorry

end sum_of_squares_change_l624_624815


namespace selection_methods_l624_624524

open Nat

def combination (n k : ℕ) := n.fact / (k.fact * (n - k).fact)

theorem selection_methods : 
    let boys := 6
    let girls := 2
    let select_boys := 3
    let select_girls := 1
    combination boys select_boys * combination girls select_girls = 40 :=
by
  let boys := 6
  let girls := 2
  let select_boys := 3
  let select_girls := 1
  have c₁ : combination boys select_boys = 20 := by
    rw [combination, Nat.fact]
    norm_num
  have c₂ : combination girls select_girls = 2 := by
    rw [combination, Nat.fact]
    norm_num
  calc 
    combination boys select_boys * combination girls select_girls 
        = 20 * 2 : by rw [c₁, c₂]
    _ = 40 : by norm_num

end selection_methods_l624_624524


namespace lacrosse_more_than_football_l624_624187

-- Define the constants and conditions
def total_bottles := 254
def football_players := 11
def bottles_per_football_player := 6
def soccer_bottles := 53
def rugby_bottles := 49

-- Calculate the number of bottles needed by each team
def football_bottles := football_players * bottles_per_football_player
def other_teams_bottles := football_bottles + soccer_bottles + rugby_bottles
def lacrosse_bottles := total_bottles - other_teams_bottles

-- The theorem to be proven
theorem lacrosse_more_than_football : lacrosse_bottles - football_bottles = 20 :=
by
  sorry

end lacrosse_more_than_football_l624_624187


namespace center_of_symmetry_l624_624760

-- Define the determinant operation for a 2x2 matrix
def determinant (a1 a2 a3 a4 : ℝ) : ℝ :=
  a1 * a4 - a2 * a3

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  determinant (sin (2 * x)) (real.sqrt 3) (cos (2 * x)) 1

-- Define the translated function g(x)
def g (x : ℝ) : ℝ :=
  f (x + π/6)

theorem center_of_symmetry :
  ∃ k : ℤ, (π / 2 * ↑k, 0) = (π / 2, 0) :=
by
  -- Proof is skipped
  sorry

end center_of_symmetry_l624_624760


namespace max_heaps_660_stones_l624_624028

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624028


namespace f_exponent_inequality_l624_624241

-- Conditions
variable {R : Type*} [LinearOrderedField R]
variable (f : R → R)
variable (h_derivative : ∀ x : R, deriv f x < f x)

-- Definition
def F (x : R) : R := f x / Real.exp x

-- Goal
theorem f_exponent_inequality (h_deriv : ∀ x : R, deriv f x < f x) :
  f 2 < Real.exp 2 * f 0 ∧
  f 2012 < Real.exp 2012 * f 0 := by
  sorry

end f_exponent_inequality_l624_624241


namespace sum_fractions_eq_l624_624697

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624697


namespace a0_sn_and_comparison_l624_624828

theorem a0_sn_and_comparison (n : ℕ) (h : 0 < n) :
    let x := (Nat.succ n)
    let S_n := (∑ i in Finset.range n, (coeff x i))
    (a₀ = 2^n) ∧ (S_n = 3^n - 2^n) ∧ 
    ((n = 1 ∨ n ≥ 4 → 3^n > (n - 1) * 2^n + 2 * n^2) ∧ (n = 2 → 3^n < (n - 1) *  2^n + 2 * n^2)) :=
by
  sorry

end a0_sn_and_comparison_l624_624828


namespace tree_ratio_l624_624331

theorem tree_ratio (A P C : ℕ) 
  (hA : A = 58)
  (hP : P = 3 * A)
  (hC : C = 5 * P) : (A, P, C) = (1, 3 * 58, 15 * 58) :=
by
  sorry

end tree_ratio_l624_624331


namespace add_fractions_l624_624722

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624722


namespace inscribed_circle_radius_l624_624328

theorem inscribed_circle_radius (A p r s : ℝ) (h₁ : A = 2 * p) (h₂ : p = 2 * s) (h₃ : A = r * s) : r = 4 :=
by sorry

end inscribed_circle_radius_l624_624328


namespace colored_pictures_count_l624_624126

def initial_pictures_count : ℕ := 44 + 44
def pictures_left : ℕ := 68

theorem colored_pictures_count : initial_pictures_count - pictures_left = 20 := by
  -- Definitions and proof will go here
  sorry

end colored_pictures_count_l624_624126


namespace quadratic_no_real_roots_l624_624287

theorem quadratic_no_real_roots (m : ℝ) : 
  (m + 2) * (m + 2) * x^2 - x + m = 0
  (1 - 4 * (m + 2) * m < 0) → 
  (m < -1 - (Real.sqrt 5) / 2 ∨ m > -1 + (Real.sqrt 5) / 2 ∧ m ≠ -2) :=
sorry

end quadratic_no_real_roots_l624_624287


namespace player_lane_permutations_l624_624083

theorem player_lane_permutations (n : ℕ) (h1 : ∃ m, m = n / 2) (h2 : ∀ player, player plays (2 * (m - 1) + 1)) : 
  (player_lane_permutations n 8 120960 : ℕ) = 120960 :=
sorry

end player_lane_permutations_l624_624083


namespace find_number_of_integers_l624_624784
-- Import the entire Mathlib library

theorem find_number_of_integers :
  { n : ℤ | 1 + (⌊ (105 * n : ℚ) / 106 ⌋) = (⌈ (104 * n : ℚ) / 105 ⌉) }.finite ∧
  (set.to_finset { n : ℤ | 1 + (⌊ (105 * n : ℚ) / 106 ⌋) = (⌈ (104 * n : ℚ) / 105 ⌉) }).card = 11130 :=
by 
  -- Proof Goes Here
  sorry

end find_number_of_integers_l624_624784


namespace sum_odd_even_3786_l624_624876

theorem sum_odd_even_3786 :
  let m := (Array.range 112).filter (· % 2 = 1) |>.sum
  let t := (Array.range 51).filter (· % 2 = 0) |>.sum
  m + t = 3786 :=
by
  let m := (Array.range 112).filter (· % 2 = 1) |>.sum
  let t := (Array.range 51).filter (· % 2 = 0) |>.sum
  sorry

end sum_odd_even_3786_l624_624876


namespace problem_I_problem_II_problem_III_l624_624341

-- Problem (I)
theorem problem_I (a_n : ℕ → ℤ) (h : ∀ n, a_n n = (-1)^(n+1)) :
  V 5 = 8 :=
sorry

-- Problem (II)
theorem problem_II (a b : ℤ) (a_n : ℕ → ℤ) (m : ℕ) (h1 : 1 < m) (h2 : a > b)
  (h3 : a_n 1 = a) (h4 : a_n m = b) :
  V m = a - b ↔ ∀ i, 1 ≤ i ∧ i < m → a_n (i + 1) ≤ a_n i :=
sorry

-- Problem (III)
theorem problem_III (a_n : ℕ → ℕ) (m : ℕ) (h : ∀ n, 1 ≤ n ∧ n ≤ m → a_n n ≥ 0)
  (h_sum : ∑ i in finset.range m, a_n (i + 1) = m^2) :
  (V m = 0 ∨ V m = if m = 2 then 4 else 2*m^2) :=
sorry

end problem_I_problem_II_problem_III_l624_624341


namespace necessary_but_not_sufficient_l624_624067

theorem necessary_but_not_sufficient (x y : ℕ) : x + y = 3 → (x = 1 ∧ y = 2) ↔ (¬ (x = 0 ∧ y = 3)) := by
  sorry

end necessary_but_not_sufficient_l624_624067


namespace equal_lengths_AC_CE_l624_624363

/-!
# Triangle Geometry Proof

Proving that in a given right triangle with specific point placements and perpendicular lines,
certain lengths in the geometry are equal.
-/

theorem equal_lengths_AC_CE
  {A B C D E : Type}
  (hABC : triangle A B C)
  (h_right_angle : right_angle C A B)
  (hCD_eq_BC : distance C D = distance B C)
  (hD_between_A_C : between D A C)
  (h_perpendicular_DE_AB : perpendicular DE AB)
  (h_intersection_DE_BC : intersects DE BC E) :
  distance A C = distance C E :=
  sorry

end equal_lengths_AC_CE_l624_624363


namespace remaining_units_l624_624545

theorem remaining_units : 
  ∀ (total_units : ℕ) (first_half_fraction : ℚ) (additional_units : ℕ), 
  total_units = 2000 →
  first_half_fraction = 3 / 5 →
  additional_units = 300 →
  (total_units - (first_half_fraction * total_units).toNat - additional_units) = 500 := by
  intros total_units first_half_fraction additional_units htotal hunits_fraction hadditional
  sorry

end remaining_units_l624_624545


namespace sara_change_l624_624417

def price_book1 : ℝ := 5.5
def price_book2 : ℝ := 6.5
def price_notebook : ℝ := 3.0
def price_bookmarks : ℝ := 2.0
def discount_books : ℝ := 0.10
def discount_other : ℝ := 0.05
def sales_tax : ℝ := 0.07
def conversion_rate : ℝ := 1.1
def payment_usd : ℝ := 50.0

def total_cost_eur := price_book1 + price_book2 + price_notebook + price_bookmarks
def total_discount_eur := (price_book1 + price_book2) * discount_books + 
                          (price_notebook + price_bookmarks) * discount_other
def cost_after_discount_eur := total_cost_eur - total_discount_eur
def cost_after_tax_eur := cost_after_discount_eur * (1 + sales_tax)
def total_cost_usd := cost_after_tax_eur * conversion_rate
def change_usd := payment_usd - total_cost_usd

theorem sara_change : Float.round (change_usd * 100.0) / 100.0 = 31.70 := by
  sorry

end sara_change_l624_624417


namespace factor_diff_of_squares_l624_624772

-- Define the expression t^2 - 49 and show it is factored as (t - 7)(t + 7)
theorem factor_diff_of_squares (t : ℝ) : t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end factor_diff_of_squares_l624_624772


namespace smallest_circle_radius_l624_624516

-- Define the problem setup
theorem smallest_circle_radius (r : ℝ) (r > 0) :
    ∃ (r1 : ℝ), r1 = r / 6 :=
sorry

end smallest_circle_radius_l624_624516


namespace ticket_cost_is_nine_l624_624553

theorem ticket_cost_is_nine (bought_tickets : ℕ) (left_tickets : ℕ) (spent_dollars : ℕ) 
  (h1 : bought_tickets = 6) 
  (h2 : left_tickets = 3) 
  (h3 : spent_dollars = 27) : 
  spent_dollars / (bought_tickets - left_tickets) = 9 :=
by
  -- Using the imported library and the given conditions
  sorry

end ticket_cost_is_nine_l624_624553


namespace triangle_from_intersection_l624_624237

theorem triangle_from_intersection (a : ℝ) (h₀ : a ≠ 1) (h₁ : a ≠ -1) :
  let A := (1 : ℝ, a)
  let B := (-1 : ℝ, -a)
  let C := (-a : ℝ, -1)
  let slope_AC := (C.2 - A.2) / (C.1 - A.1)
  let slope_BC := (C.2 - B.2) / (C.1 - B.1)
  slope_AC * slope_BC = -1 := by
  sorry

end triangle_from_intersection_l624_624237


namespace problem_statement_l624_624757

theorem problem_statement (p q : Prop):
  let statements := [
    p ∨ ¬q,  -- Statement (1)
    p ∧ q,   -- Statement (2)
    ¬p ∧ q,  -- Statement (3)
    ¬p ∧ ¬q  -- Statement (4)
  ]
  let number_of_implications := (statements.map (λ stmt, stmt → ¬(p ∧ q))).count (λ b, b)
  number_of_implications = 3 := sorry

end problem_statement_l624_624757


namespace add_fractions_l624_624567

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624567


namespace exist_three_integers_l624_624764

theorem exist_three_integers :
  ∃ (a b c : ℤ), a * b - c = 2018 ∧ b * c - a = 2018 ∧ c * a - b = 2018 := 
sorry

end exist_three_integers_l624_624764


namespace polynomials_with_same_Hp_l624_624796

noncomputable def Hp (p : polynomial ℂ) : set ℂ :=
  {z : ℂ | complex.abs (p.eval z) = 1}

theorem polynomials_with_same_Hp (p q : polynomial ℂ) (hp : ¬ is_constant p) (hq : ¬ is_constant q)
  (h : Hp p = Hp q) :
  ∃ (r : polynomial ℂ) (m n : ℕ) (ξ : ℂ), p = r ^ m ∧ q = ξ * r ^ n ∧ complex.abs ξ = 1 :=
sorry

end polynomials_with_same_Hp_l624_624796


namespace expression_to_irreducible_fraction_l624_624202

theorem expression_to_irreducible_fraction :
  (6 + (16 / 2015)) * (9 + (17 / 2016)) - (2 + (1999 / 2015)) * (17 + (1999 / 2016)) - 27 * (16 / 2015) = 17 / 224 := 
by
  -- Definitions based on the conditions
  let a := (16 : ℚ) / 2015
  let b := (17 : ℚ) / 2016
  
  -- Theorem to prove
  have h : (6 + a) * (9 + b) - (3 - a) * (18 - b) - 27 * a = 17 / 224
  sorry

end expression_to_irreducible_fraction_l624_624202


namespace card_count_correct_l624_624359

theorem card_count_correct :
  ∀ (digs : Finset ℕ), digs = {1, 2, 3, 4, 5, 6, 7} →
  ∃ count : ℕ,
    (∀ (a b : ℕ), a ∈ digits_combinations(digs, 3) →
    b ∈ digits_combinations(digs \ {find_unused_digit(digs)}, 3) →
    a ≠ b →
    divisible_by_81(a * b) ∧ divisible_by_9(a + b)) →
  count = 36 :=
by
  intros
  sorry

end card_count_correct_l624_624359


namespace sqrt_pos_condition_l624_624465

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end sqrt_pos_condition_l624_624465


namespace part_a_part_b_l624_624240

-- Convex quadrilateral ABCD
variables {A B C D : Type*} [ConvexQuadrilateral A B C D]

-- Assumptions for part (a)
variable (h_obtuse_A : IsObtuse ∠A)
variable (h_obtuse_B : IsObtuse ∠B)

-- Assumptions for part (b)
variable (h_angle_A_gt_D : ∠A > ∠D)
variable (h_angle_B_gt_C : ∠B > ∠C)

-- Conclusion to prove for part (a)
theorem part_a : AB < CD :=
sorry

-- Conclusion to prove for part (b)
theorem part_b : AB < CD :=
sorry

end part_a_part_b_l624_624240


namespace minimum_value_PM_l624_624259

def point_on_line (P : ℝ × ℝ) : Prop := ∃ x y, P = (x, y) ∧ x + y + 3 = 0

def line_intersects_circle_once (P M : ℝ × ℝ) : Prop :=
  (∃ x y, M = (x, y) ∧ (x - 5)^2 + y^2 = 16) ∧ 
  (∃ l, ∀ t, l t = (P.1 + t * (M.1 - P.1), P.2 + t * (M.2 - P.2)) → 
    ∀ t₁ t₂, l t₁ = l t₂ → t₁ = t₂)

def center : ℝ × ℝ := (5, 0)
def radius : ℝ := 4

theorem minimum_value_PM 
  (P M : ℝ × ℝ)
  (hP : point_on_line P)
  (hM : line_intersects_circle_once P M) 
  : dist P M = 4 := 
sorry

end minimum_value_PM_l624_624259


namespace fraction_addition_l624_624633

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624633


namespace mono_sine_cosine_same_l624_624316

theorem mono_sine_cosine_same {φ : ℝ} (hφ : φ = Real.pi / 2) : 
  (∀ x1 x2, 0 ≤ x1 ∧ x2 ≤ Real.pi / 2 ∧ x1 < x2 → cos (2 * x1) > cos (2 * x2)) →
  (∀ x1 x2, 0 ≤ x1 ∧ x2 ≤ Real.pi / 2 ∧ x1 < x2 → sin (x1 + φ) > sin (x2 + φ)) := 
sorry

end mono_sine_cosine_same_l624_624316


namespace necessary_but_not_sufficient_l624_624255

variable {f : ℝ → ℝ}

def is_constant_derivative (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, has_deriv_at f c x

def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

theorem necessary_but_not_sufficient :
  (∀ f, is_linear_function f → is_constant_derivative f) ∧ ¬ (∀ f, is_constant_derivative f → is_linear_function f) :=
by
  sorry

end necessary_but_not_sufficient_l624_624255


namespace angle_movement_condition_l624_624147

noncomputable def angle_can_reach_bottom_right (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) : Prop :=
  (m % 2 = 1) ∧ (n % 2 = 1)

theorem angle_movement_condition (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n) :
  angle_can_reach_bottom_right m n h1 h2 ↔ (m % 2 = 1 ∧ n % 2 = 1) :=
sorry

end angle_movement_condition_l624_624147


namespace count_cubes_2_9_to_2_17_l624_624864

noncomputable def lower_bound := 2^9 + 1
noncomputable def upper_bound := 2^17 + 1

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

def count_perfect_cubes_between (a b : ℕ) : ℕ :=
  (finset.range (b + 1)).filter (λ n, a ≤ n ∧ is_perfect_cube n).card

theorem count_cubes_2_9_to_2_17 : count_perfect_cubes_between lower_bound upper_bound = 42 := by
  sorry

end count_cubes_2_9_to_2_17_l624_624864


namespace limit_as_x_approaches_2_l624_624198

noncomputable def limit_expr (x k : ℝ) : ℝ :=
  (x^2 - 2*x + k) / (x - 2)

theorem limit_as_x_approaches_2 (k : ℝ) (h : k = 4) :
  limit (fun x => limit_expr x k) (nhds 2) = 0 := by
  sorry

end limit_as_x_approaches_2_l624_624198


namespace add_fractions_l624_624732

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624732


namespace floor_abs_add_abs_floor_neg_5_7_l624_624563

theorem floor_abs_add_abs_floor_neg_5_7 : 
  (Int.floor (abs (-5.7)) + abs (Int.floor (-5.7))) = 11 :=
by 
  sorry

end floor_abs_add_abs_floor_neg_5_7_l624_624563


namespace total_dolls_l624_624171

-- Defining the given conditions as constants.
def big_boxes : Nat := 5
def small_boxes : Nat := 9
def dolls_per_big_box : Nat := 7
def dolls_per_small_box : Nat := 4

-- The main theorem we want to prove
theorem total_dolls : (big_boxes * dolls_per_big_box) + (small_boxes * dolls_per_small_box) = 71 :=
by
  rw [Nat.mul_add, Nat.mul_eq_mul, Nat.mul_eq_mul]
  exact sorry

end total_dolls_l624_624171


namespace sum_fractions_eq_l624_624685

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624685


namespace treasure_location_l624_624979

def Dwarf := ℕ
def Location := ℕ
def Statement := Location

-- Positions of dwarves
def dwarf1 : Dwarf := 1
def dwarf2 : Dwarf := 2
def dwarf3 : Dwarf := 3
def dwarf4 : Dwarf := 4
def dwarf5 : Dwarf := 5
def dwarf6 : Dwarf := 6

-- Statements made by each dwarf
def statement1 : Statement := 1 -- cave
def statement2 : Statement := 2 -- bottom of the lake
def statement3 : Statement := 3 -- castle
def statement4 : Statement := 4 -- fairy forest
def statement5 : Statement := 2 -- bottom of the lake

-- Conditions based on the problem
def truthAdjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = dwarf1 ∧ d2 = dwarf2) ∨ (d1 = dwarf2 ∧ d2 = dwarf3) ∨ (d1 = dwarf3 ∧ d2 = dwarf4) ∨
  (d1 = dwarf4 ∧ d2 = dwarf5) ∨ (d1 = dwarf5 ∧ d2 = dwarf6) ∨ (d1 = dwarf6 ∧ d2 = dwarf1)
  
def lieAdjacent (d1 d2 : Dwarf) : Prop :=
  (d1 = dwarf1 ∧ d2 = dwarf6) ∨ (d1 = dwarf2 ∧ d2 = dwarf1) ∨ (d1 = dwarf3 ∧ d2 = dwarf2) ∨
  (d1 = dwarf4 ∧ d2 = dwarf3) ∨ (d1 = dwarf5 ∧ d2 = dwarf4) ∨ (d1 = dwarf6 ∧ d2 = dwarf5)

def nonAdjacent (d1 d2 : Dwarf) : Prop := ¬truthAdjacent d1 d2 ∧ ¬lieAdjacent d1 d2

theorem treasure_location
  (truth_tellers : {d1 d2 : Dwarf // truthAdjacent d1 d2})
  (liars : {d3 d4 : Dwarf // lieAdjacent d3 d4})
  (rest : {d5 d6 : Dwarf // nonAdjacent d5 d6})
  (statements : ∀ (d : Dwarf), Statement)
  (h1 : statements dwarf1 = statement1)
  (h2 : statements dwarf2 = statement2)
  (h3 : statements dwarf3 = statement3)
  (h4 : statements dwarf4 = statement4)
  (h5 : statements dwarf5 = statement5) :
  ∃ loc : Location, loc = 1 := 
sorry

end treasure_location_l624_624979


namespace soup_weight_after_four_days_l624_624560

theorem soup_weight_after_four_days :
  ∀ (w0 : ℝ) (d1 d2 d3 d4 : ℝ),
  w0 = 80 → d1 = 0.40 → d2 = 0.35 → d3 = 0.55 → d4 = 0.50 →
  let w1 := w0 * (1 - d1) in
  let w2 := w1 * (1 - d2) in
  let w3 := w2 * (1 - d3) in
  let w4 := w3 * (1 - d4) in
  w4 = 7.02 :=
begin
  intros w0 d1 d2 d3 d4 h0 h1 h2 h3 h4,
  simp [h0, h1, h2, h3, h4],
  norm_num,
  sorry
end

end soup_weight_after_four_days_l624_624560


namespace emma_coins_missing_fraction_l624_624193

theorem emma_coins_missing_fraction (x : ℕ) :
  let lost := (1 / 3 : ℚ) * x,
      found := (3 / 4 : ℚ) * lost,
      total_fetched := (2 / 3 : ℚ) * x + found
  in x - total_fetched = (1 / 12 : ℚ) * x := 
by
  sorry

end emma_coins_missing_fraction_l624_624193


namespace lentil_dishes_count_l624_624541

theorem lentil_dishes_count :
  ∃ (B S L T : ℕ), B = 2 * T ∧ S = 3 * T ∧
                   3 + 4 + 2 + B + S + L + T = 20 ∧
                   6 * T + L = 11 ∧
                   L + 3 + 2 = 10 :=
by {
  set T := 1 with hT,
  set B := 2 * T with hB,
  set S := 3 * T with hS,
  set L := 5 with hL,
  use [B, S, L, T],
  simp [hB, hS, hL],
  norm_num,
  simp [hT],
  norm_num,
  sorry
}

end lentil_dishes_count_l624_624541


namespace domain_of_v_l624_624480

def v (x : ℝ) : ℝ := 1 / (x ^ (3 / 2))

theorem domain_of_v : ∀ x: ℝ, (v x) = 1 / (x ^ (3 / 2)) -> (0 < x ↔ ∃ y: ℝ, 0 < y ∧ y < x ∧ v x = 1 / (x ^ (3 / 2))) :=
by
  sorry

end domain_of_v_l624_624480


namespace sophomores_bought_15_more_markers_l624_624900

theorem sophomores_bought_15_more_markers (f_cost s_cost marker_cost : ℕ) (hf: f_cost = 267) (hs: s_cost = 312) (hm: marker_cost = 3) : 
  (s_cost / marker_cost) - (f_cost / marker_cost) = 15 :=
by
  sorry

end sophomores_bought_15_more_markers_l624_624900


namespace find_f5_l624_624806

variable {F : ℚ → ℚ}

-- Conditions
def functional_eq (x : ℚ) : F (2 * x + 1) = x^2 - 2 * x := sorry

-- Theorem to prove
theorem find_f5 (h : ∀ x : ℚ, functional_eq x) : F 5 = 0 :=
by
  sorry

end find_f5_l624_624806


namespace solve_cubic_equation_l624_624059

noncomputable def equation (x : ℝ) : ℝ :=
  real.cbrt (10 * x - 2) + real.cbrt (20 * x + 3) - 5 * real.cbrt x

theorem solve_cubic_equation :
  equation 0 = 0 ∧ equation (-1 / 25) = 0 ∧ equation (1 / 375) = 0 :=
by {
  split,
  { rw [equation, real.cbrt_zero, real.cbrt_zero, real.cbrt_zero],
    norm_num },
  split,
  { rw [equation, real.cbrt_of_neg (show 10 * (-1 / 25) - 2 < 0, by norm_num),
               real.cbrt_of_neg (show 20 * (-1 / 25) + 3 < 0, by norm_num),
               real.cbrt_of_neg (show (-1 / 25) < 0, by norm_num)],
    norm_num },
  { rw [equation, real.cbrt_of_sub_one (show 10 * (1 / 375) - 2 < 0, by norm_num),
               real.cbrt_of_sub_one (show 20 * (1 / 375) + 3 < 0, by norm_num),
               real.cbrt_of_pos (show (1 / 375) > 0, by norm_num)],
    norm_num }
}

end solve_cubic_equation_l624_624059


namespace factorize_difference_of_squares_l624_624776

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
by 
  sorry

end factorize_difference_of_squares_l624_624776


namespace sum_fractions_eq_l624_624693

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624693


namespace add_fractions_l624_624657

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624657


namespace fraction_addition_l624_624609

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624609


namespace fraction_addition_l624_624654

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624654


namespace total_air_removed_after_5_strokes_l624_624874

theorem total_air_removed_after_5_strokes:
  let initial_air := 1
  let remaining_air_after_first_stroke := initial_air * (2 / 3)
  let remaining_air_after_second_stroke := remaining_air_after_first_stroke * (3 / 4)
  let remaining_air_after_third_stroke := remaining_air_after_second_stroke * (4 / 5)
  let remaining_air_after_fourth_stroke := remaining_air_after_third_stroke * (5 / 6)
  let remaining_air_after_fifth_stroke := remaining_air_after_fourth_stroke * (6 / 7)
  initial_air - remaining_air_after_fifth_stroke = 5 / 7 := by
  sorry

end total_air_removed_after_5_strokes_l624_624874


namespace add_fractions_l624_624725

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624725


namespace smallest_prime_sum_l624_624843

open Nat

theorem smallest_prime_sum 
  (a b c d : ℕ) 
  (ha : Prime a) 
  (hb : Prime b) 
  (hc : Prime c) 
  (hd : Prime d)
  (hproduct : ∃ x : ℕ, a * b * c * d = 55 * (x + 27)) : 
  a + b + c + d = 20 := 
sorry

end smallest_prime_sum_l624_624843


namespace sum_of_squares_increase_by_800_l624_624809

theorem sum_of_squares_increase_by_800
  (x : Fin 100 → ℝ)
  (h : ∑ j, x j ^ 2 = ∑ j, (x j + 2) ^ 2) :
  (∑ j, (x j + 4) ^ 2) - (∑ j, x j ^ 2) = 800 := 
by
  sorry

end sum_of_squares_increase_by_800_l624_624809


namespace interest_approx_l624_624962

noncomputable def original_purchase_price : ℝ := 2345
noncomputable def down_payment : ℝ := 385
noncomputable def monthly_payment : ℝ := 125
noncomputable def number_of_payments : ℝ := 18

def total_amount_paid : ℝ :=
  down_payment + (monthly_payment * number_of_payments)

def extra_amount_paid : ℝ :=
  total_amount_paid - original_purchase_price

def interest_percent : ℝ :=
  (extra_amount_paid / original_purchase_price) * 100

theorem interest_approx : abs (interest_percent - 12.4) < 0.1 :=
by
  sorry

end interest_approx_l624_624962


namespace collinear_A_M_N_l624_624502

variables {A B C P Q M E F N : Type*}

-- Conditions
variables [is_parallel BC PQ]
variables [P_on AB : P ∈ line AB]
variables [Q_on AC : Q ∈ line AC]
variables [M_inside_APQ : M ∈ triangle APQ]
variables [E_on_PQ : E ∈ segment PQ]
variables [F_on_PQ : F ∈ segment PQ]
variables (ω1 : circle PMF) (ω2 : circle QME)
variables [N_on_ω1 : N ∈ ω1]
variables [N_on_ω2 : N ∈ ω2]

-- Statement to be proved
theorem collinear_A_M_N : collinear {A, M, N} :=
sorry

end collinear_A_M_N_l624_624502


namespace perfect_cubes_between_bounds_l624_624866

-- Definitions of the given bounds
def lower_bound : ℕ := 2^9 + 1
def upper_bound : ℕ := 2^17 + 1

-- Theorem statement indicating the number of perfect cubes between these bounds is 40
theorem perfect_cubes_between_bounds : 
  (number_of_perfect_cubes lower_bound upper_bound) = 40 := sorry

-- Placeholder function for the number of perfect cubes within a given inclusive interval
noncomputable def number_of_perfect_cubes (a b : ℕ) : ℕ := sorry

end perfect_cubes_between_bounds_l624_624866


namespace sam_gave_joan_seashells_l624_624913

variable (original_seashells : ℕ) (total_seashells : ℕ)

theorem sam_gave_joan_seashells (h1 : original_seashells = 70) (h2 : total_seashells = 97) :
  total_seashells - original_seashells = 27 :=
by
  sorry

end sam_gave_joan_seashells_l624_624913


namespace log_equation_solution_l624_624980

theorem log_equation_solution (x : ℝ) (h : log 3 x + log 9 (x ^ 3) = 6) : x = real.exp ((12 : ℝ) / 5 * real.log (3 : ℝ)) := 
by
  sorry

end log_equation_solution_l624_624980


namespace sufficient_but_not_necessary_condition_for_monotonic_increase_l624_624761

theorem sufficient_but_not_necessary_condition_for_monotonic_increase (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 - 2 * a * x + 2) ≥ 0) ↔ a ∈ [-√6, √6] ∧ a ∈ [1,2] :=
by sorry

end sufficient_but_not_necessary_condition_for_monotonic_increase_l624_624761


namespace area_of_quadrilateral_l624_624360

theorem area_of_quadrilateral (A B C D : Point)
  (h_convex : convex_quadrilateral A B C D)
  (h_AC : distance A C = 20)
  (h_BC : distance B C = 12)
  (h_BD : distance B D = 17)
  (h1 : angle A C B = 80)
  (h2 : angle D B A = 70) : 
  area_quadrilateral A B C D = 85 := 
sorry

end area_of_quadrilateral_l624_624360


namespace add_fractions_l624_624668

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624668


namespace range_f_real_l624_624274

noncomputable def f (a : ℝ) (x : ℝ) :=
  if x > 1 then (a ^ x) else (4 - a / 2) * x + 2

theorem range_f_real (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ (1 < a ∧ a ≤ 4) :=
by
  sorry

end range_f_real_l624_624274


namespace add_fractions_l624_624575

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624575


namespace fraction_addition_l624_624639

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624639


namespace max_volume_of_ideal_gas_l624_624405

variables {P T P_0 T_0 a b c R : ℝ}
variables (h_eq : (P / P_0 - a)^2 + (T / T_0 - b)^2 = c^2)
variables (h_ineq : c^2 < a^2 + b^2)

theorem max_volume_of_ideal_gas :
  (V : ℝ) = (R * T_0 / P_0) * ((a * sqrt(a^2 + b^2 - c^2) + b * c) / (b * sqrt(a^2 + b^2 - c^2) - a * c)) :=
sorry

end max_volume_of_ideal_gas_l624_624405


namespace remaining_integers_count_is_60_l624_624456

def T : Finset ℕ := Finset.range 101

noncomputable def number_of_remaining_integers : ℕ :=
  let multiples_of_4 := T.filter (λ n => n % 4 = 0)
  let remaining_after_4 := T.sdiff multiples_of_4
  let multiples_of_5 := T.filter (λ n => n % 5 = 0)
  let multiples_of_20 := T.filter (λ n => n % 20 = 0)
  let unique_multiples_of_5 := multiples_of_5.sdiff multiples_of_20
  remaining_after_4.sdiff unique_multiples_of_5

theorem remaining_integers_count_is_60 : number_of_remaining_integers = 60 := by
  sorry

end remaining_integers_count_is_60_l624_624456


namespace fraction_addition_l624_624600

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624600


namespace maximum_area_of_triangle_ABQ_l624_624272

open Real

structure Point3D where
  x : ℝ
  y : ℝ

def circle_C (Q : Point3D) : Prop := (Q.x - 3)^2 + (Q.y - 4)^2 = 4

def A := Point3D.mk 1 0
def B := Point3D.mk (-1) 0

noncomputable def area_triangle (P Q R : Point3D) : ℝ :=
  (1 / 2) * abs ((P.x * (Q.y - R.y)) + (Q.x * (R.y - P.y)) + (R.x * (P.y - Q.y)))

theorem maximum_area_of_triangle_ABQ : ∀ (Q : Point3D), circle_C Q → area_triangle A B Q ≤ 6 := by
  sorry

end maximum_area_of_triangle_ABQ_l624_624272


namespace overall_average_correct_l624_624532

noncomputable def overall_average : ℝ :=
  let students1 := 60
  let students2 := 35
  let students3 := 45
  let students4 := 42
  let avgMarks1 := 50
  let avgMarks2 := 60
  let avgMarks3 := 55
  let avgMarks4 := 45
  let total_students := students1 + students2 + students3 + students4
  let total_marks := (students1 * avgMarks1) + (students2 * avgMarks2) + (students3 * avgMarks3) + (students4 * avgMarks4)
  total_marks / total_students

theorem overall_average_correct : overall_average = 52.00 := by
  sorry

end overall_average_correct_l624_624532


namespace solid_volume_cylinder_l624_624461

theorem solid_volume_cylinder {r h : ℝ} (r_pos : 0 < r) (h_pos : 0 < h) 
    (diameters : r = 3 ∨ r = 2) (heights : h = 8 ∨ h = 12)
    (volume : ℝ) :
    volume = π * r^2 * h → volume ≠ 24 * π ∧ volume ≠ 48 * π ∧ volume ≠ 72 * π :=
by
  intro V
  split
  { intro H
    have : 24 * π = 72 * π / 3 := sorry
    rw [this] at H
    exact sorry },
  split
  { intro H
    cases diameters
    { cases heights
      { simp [diameters, heights, V] at H },
      { simp [diameters, heights, V] at H } },
    { cases heights
      { simp [diameters, heights, V] at H },
      { simp [diameters, heights, V] at H } } },
  { intro H
    cases diameters
    { cases heights
      { simp [diameters, heights, V] at H },
      { simp [diameters, heights, V] at H } },
    { cases heights
      { simp [diameters, heights, V] at H },
      { simp [diameters, heights, V] at H } } }

end solid_volume_cylinder_l624_624461


namespace max_profit_at_9_l624_624519

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 10.8 - (1 / 30) * x^2
else if h : x > 10 then 108 / x - 1000 / (3 * x^2)
else 0

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then 8.1 * x - x^3 / 30 - 10
else if h : x > 10 then 98 - 1000 / (3 * x) - 2.7 * x
else 0

theorem max_profit_at_9 : W 9 = 38.6 :=
sorry

end max_profit_at_9_l624_624519


namespace necessary_condition_for_inequality_l624_624082

theorem necessary_condition_for_inequality 
  (m : ℝ) : (∀ x : ℝ, x^2 - 2 * x + m > 0) → m > 0 :=
by 
  sorry

end necessary_condition_for_inequality_l624_624082


namespace tiling_impossible_l624_624438

theorem tiling_impossible
  (m n : ℕ)
  (grid : matrix (fin m) (fin n) (fin 4)) -- 4 distinct colors for the grid
  (initial_tiling : (matrix (fin m) (fin n) (option bool)) → bool)
  (broken_tile : fin m × fin n)
  (replacement : (matrix (fin m) (fin n) (option bool)) → bool) :
  ¬ (∃ new_tiling : (matrix (fin m) (fin n) (option bool)),
        replacement new_tiling ∧ 
        new_tiling broken_tile = some true ∧ 
        forall pos : (fin m × fin n), 
          if pos ≠ broken_tile then initial_tiling new_tiling
          else true) :=
begin
  -- Skipping proof with sorry
  sorry
end

end tiling_impossible_l624_624438


namespace volleyball_team_starters_l624_624044

-- Define the total number of ways to choose 7 starters
def num_ways_to_choose_team : ℕ := 5434

-- State the conditions about the volleyball players
variable (total_players : ℕ) (triplets : Finset ℕ) (twins : Finset ℕ)

def volleyball_team_conditions : Prop :=
  total_players = 16 ∧
  triplets = {0, 1, 2} ∧ -- Alicia, Amanda, Anna represented by indices
  twins = {3, 4} -- Beth, Brenda represented by indices

-- The main theorem statement
theorem volleyball_team_starters (total_players = 16)
  (triplets = {0, 1, 2}) (twins = {3, 4}) :
  ∃ ways : ℕ, ways = num_ways_to_choose_team :=
sorry

end volleyball_team_starters_l624_624044


namespace pizza_area_increase_percent_l624_624407

def area_of_circle (r : ℝ) : ℝ := real.pi * r^2

theorem pizza_area_increase_percent (initial_diameter final_diameter : ℝ) (h₁ : initial_diameter = 10) (h₂ : final_diameter = 12) :
  let r₁ := initial_diameter / 2
  let r₂ := final_diameter / 2
  let initial_area := area_of_circle r₁
  let final_area := area_of_circle r₂
  let increase_in_area := final_area - initial_area
  (increase_in_area / initial_area) * 100 = 44 := 
sorry

end pizza_area_increase_percent_l624_624407


namespace selection_ways_l624_624785

def ways_to_select_president_and_secretary (n : Nat) : Nat :=
  n * (n - 1)

theorem selection_ways :
  ways_to_select_president_and_secretary 5 = 20 :=
by
  sorry

end selection_ways_l624_624785


namespace perfect_cubes_between_bounds_l624_624865

-- Definitions of the given bounds
def lower_bound : ℕ := 2^9 + 1
def upper_bound : ℕ := 2^17 + 1

-- Theorem statement indicating the number of perfect cubes between these bounds is 40
theorem perfect_cubes_between_bounds : 
  (number_of_perfect_cubes lower_bound upper_bound) = 40 := sorry

-- Placeholder function for the number of perfect cubes within a given inclusive interval
noncomputable def number_of_perfect_cubes (a b : ℕ) : ℕ := sorry

end perfect_cubes_between_bounds_l624_624865


namespace add_fractions_l624_624568

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624568


namespace fraction_transform_l624_624120

theorem fraction_transform (x : ℕ) (h : 9 * (537 - x) = 463 + x) : x = 437 :=
by
  sorry

end fraction_transform_l624_624120


namespace fraction_addition_l624_624597

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624597


namespace coloring_exists_l624_624414

open Finset Function

theorem coloring_exists (M : Finset ℕ) (hM : ∀ n, n ∈ M ↔ n ∈ (range 2010).map (λ x, x + 1)) :
  ∃ (c : ℕ → ℕ), (∀ x ∈ M, c x ∈ {1, 2, 3, 4, 5}) ∧ 
    ∀ (a d : ℕ), d ≠ 0 → 
      (∀ n, n < 9 → M ( a + n * d )) → -- all elements a + n * d should be in M
        ∃ i j, i ≠ j ∧ c(a + i * d) ≠ c(a + j * d) :=
by
  sorry

end coloring_exists_l624_624414


namespace add_fractions_l624_624579

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624579


namespace triangle_ZEN_area_correct_l624_624105

open Classical

noncomputable def triangle_ZEN_area : ℝ :=
  let XZ := 10
  let YZ := 26
  let XY := sqrt (XZ^2 + YZ^2)
  let ZO := (XZ * YZ) / XY
  let NO := XY / 2
  let ZE := ZO
  let EN := NO
  (ZE * EN) / 2

theorem triangle_ZEN_area_correct : triangle_ZEN_area = 65 := by
  sorry

end triangle_ZEN_area_correct_l624_624105


namespace motorist_first_half_speed_l624_624526

-- Define the conditions
variable (t : ℝ) (v2 : ℝ) (d : ℝ)
variable (half_time : t / 2 = 3)
variable (second_half_speed : v2 = 48)
variable (total_distance : d = 324)

-- Define the first half distance
def first_half_distance (t : ℝ) (v2 : ℝ) (d : ℝ) : ℝ :=
d - v2 * (t / 2)

-- Define the first half speed
def first_half_speed (t : ℝ) (v2 : ℝ) (d : ℝ) : ℝ :=
(first_half_distance t v2 d) / (t / 2)

-- Prove that the speed of the motorist during the first half of the journey was 60 km/h
theorem motorist_first_half_speed (t : ℝ) (v2 : ℝ) (d : ℝ)
    (half_time : t / 2 = 3)
    (second_half_speed : v2 = 48)
    (total_distance : d = 324) :
    first_half_speed t v2 d = 60 :=
by
  sorry

end motorist_first_half_speed_l624_624526


namespace add_fractions_l624_624723

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624723


namespace add_fractions_l624_624669

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624669


namespace find_a_if_f_is_even_l624_624440

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 2 := by
  sorry

end find_a_if_f_is_even_l624_624440


namespace relationship_between_a_b_c_l624_624233

def a := Real.log 0.99 / Real.log 365
def b := 1.01 ^ 365
def c := 0.99 ^ 365

theorem relationship_between_a_b_c : a < c ∧ c < b := 
by
  sorry

end relationship_between_a_b_c_l624_624233


namespace license_plate_combinations_l624_624297

noncomputable def num_license_plates (num_consonants num_vowels num_even_digits : ℕ) : ℕ :=
  num_consonants * num_vowels * num_consonants * num_even_digits * num_even_digits

theorem license_plate_combinations :
    num_license_plates 18 8 5 = 25920 :=
by
  unfold num_license_plates
  simp
  norm_num
  sorry

end license_plate_combinations_l624_624297


namespace perp_lines_value_of_m_parallel_lines_value_of_m_l624_624292

theorem perp_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) * ((m - 2) / 3) = -1)) → 
  m = 1 / 2 := 
sorry

theorem parallel_lines_value_of_m (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 6 = 0) ∧ (∀ x y : ℝ, (m - 2) * x + 3 * y + 2 * m = 0) →
  (m ≠ 0) →
  (∀ x y : ℝ, (x + m * y + 6 = 0) → (∃ x' y' : ℝ, (m - 2) * x' + 3 * y' + 2 * m = 0) → 
  (∀ x y x' y' : ℝ, -((1 : ℝ) / m) = ((m - 2) / 3))) → 
  m = -1 := 
sorry

end perp_lines_value_of_m_parallel_lines_value_of_m_l624_624292


namespace Ellen_current_age_l624_624392

-- Definitions from the given problem conditions.
variable (E : ℕ)  -- Denote Ellen's current age as E.
axiom Martha_age : ℕ := 32  -- Martha's current age.
axiom twice_older_condition : Martha_age = 2 * (E + 6)  -- Martha is twice as old as Ellen will be in six years.

-- Proposition to prove
theorem Ellen_current_age : E = 10 :=
by
  -- It is given that Martha is 32 years old.
  have h₁: Martha_age = 32 := rfl
  -- Using the condition provided in the problem,
  -- 32 = 2 * (E + 6).
  have h₂ : 32 = 2 * (E + 6) := twice_older_condition
  -- Therefore, we can solve the equation to get E = 10.
  sorry

end Ellen_current_age_l624_624392


namespace bob_password_probability_l624_624559

noncomputable def count_even_numbers := 4
noncomputable def total_single_digit_numbers := 9
noncomputable def count_vowels := 5
noncomputable def total_letters := 26
noncomputable def count_numbers_greater_than_5 := 4

def probability_even := (count_even_numbers : ℝ) / total_single_digit_numbers
def probability_vowel := (count_vowels : ℝ) / total_letters
def probability_greater_than_5 := (count_numbers_greater_than_5 : ℝ) / total_single_digit_numbers

def probability := probability_even * probability_vowel * probability_greater_than_5

theorem bob_password_probability :
  probability = 40 / 1053 := 
by
  unfold probability probability_even probability_vowel probability_greater_than_5
  simp
  norm_num
  sorry

end bob_password_probability_l624_624559


namespace part1_part2_l624_624279

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- Problem (1)
theorem part1 (a : ℝ) (h : a = 1) : 
  ∀ x : ℝ, f x a ≥ 2 ↔ x ≤ 0 ∨ x ≥ 2 := 
  sorry

-- Problem (2)
theorem part2 (a : ℝ) (h : a > 1) : 
  (∀ x : ℝ, f x a + abs (x - 1) ≥ 2) ↔ a ≥ 3 := 
  sorry

end part1_part2_l624_624279


namespace no_solution_for_seven_ninths_l624_624281

def f (x : ℝ) : ℝ := cos x * sin (2 * x)

theorem no_solution_for_seven_ninths : ¬ ∃ x ∈ set.Ico 0 real.pi, f x = 7 / 9 := 
by sorry

end no_solution_for_seven_ninths_l624_624281


namespace last_digit_independent_of_order_l624_624507

theorem last_digit_independent_of_order 
    (p q r : ℕ) 
    (hp_even : p % 2 = 0) 
    (hq_even : q % 2 = 0)
    (hr_even : r % 2 = 0)
    (steps : List (ℕ × ℕ × ℕ)): 
  ∃ k, (k = 0 ∨ k = 1 ∨ k = 2) ∧ final_digit steps = k ∧ 
  ∀ steps', final_digit steps' = k :=
sorry

end last_digit_independent_of_order_l624_624507


namespace circle_area_outside_square_l624_624182

-- Definitions for the problem
def side_length_of_square : ℝ := 2
def radius_of_circle : ℝ := (sqrt 3) / 3

-- Theorem statement translating the problem
theorem circle_area_outside_square :
  (π * radius_of_circle^2) / side_length_of_square^2 = π / 3 :=
by
  -- The proof is omitted for now
  sorry

end circle_area_outside_square_l624_624182


namespace fraction_addition_l624_624634

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624634


namespace total_calories_consumed_l624_624142

def caramel_cookies := 10
def caramel_calories := 18

def chocolate_chip_cookies := 8
def chocolate_chip_calories := 22

def peanut_butter_cookies := 7
def peanut_butter_calories := 24

def selected_caramel_cookies := 5
def selected_chocolate_chip_cookies := 3
def selected_peanut_butter_cookies := 2

theorem total_calories_consumed : 
  (selected_caramel_cookies * caramel_calories) + 
  (selected_chocolate_chip_cookies * chocolate_chip_calories) + 
  (selected_peanut_butter_cookies * peanut_butter_calories) = 204 := 
by
  sorry

end total_calories_consumed_l624_624142


namespace fraction_addition_l624_624612

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624612


namespace add_fractions_l624_624733

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624733


namespace total_weight_four_pets_l624_624200

-- Define the weights
def Evan_dog := 63
def Ivan_dog := Evan_dog / 7
def combined_weight_dogs := Evan_dog + Ivan_dog
def Kara_cat := combined_weight_dogs * 5
def combined_weight_dogs_and_cat := Evan_dog + Ivan_dog + Kara_cat
def Lisa_parrot := combined_weight_dogs_and_cat * 3
def total_weight := Evan_dog + Ivan_dog + Kara_cat + Lisa_parrot

-- Total weight of the four pets
theorem total_weight_four_pets : total_weight = 1728 := by
  sorry

end total_weight_four_pets_l624_624200


namespace prove_travel_cost_l624_624967

noncomputable def least_expensive_travel_cost
  (a_cost_per_km : ℝ) (a_booking_fee : ℝ) (b_cost_per_km : ℝ)
  (DE DF EF : ℝ) :
  ℝ := by
  let a_cost_DE := DE * a_cost_per_km + a_booking_fee
  let b_cost_DE := DE * b_cost_per_km
  let cheaper_cost_DE := min a_cost_DE b_cost_DE

  let a_cost_EF := EF * a_cost_per_km + a_booking_fee
  let b_cost_EF := EF * b_cost_per_km
  let cheaper_cost_EF := min a_cost_EF b_cost_EF

  let a_cost_DF := DF * a_cost_per_km + a_booking_fee
  let b_cost_DF := DF * b_cost_per_km
  let cheaper_cost_DF := min a_cost_DF b_cost_DF

  exact cheaper_cost_DE + cheaper_cost_EF + cheaper_cost_DF

def travel_problem : Prop :=
  let DE := 5000
  let DF := 4000
  let EF := 2500 -- derived from the Pythagorean theorem
  least_expensive_travel_cost 0.12 120 0.20 DE DF EF = 1740

theorem prove_travel_cost : travel_problem := sorry

end prove_travel_cost_l624_624967


namespace fraction_addition_l624_624607

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624607


namespace complement_of_A_in_U_l624_624288

open Set

-- Define the universal set U and the set A.
def U := { x : ℝ | x < 4 }
def A := { x : ℝ | x < 1 }

-- Theorem statement of the complement of A with respect to U equaling [1, 4).
theorem complement_of_A_in_U : (U \ A) = { x : ℝ | 1 ≤ x ∧ x < 4 } :=
sorry

end complement_of_A_in_U_l624_624288


namespace fraction_addition_l624_624682

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624682


namespace sqrt_pos_condition_l624_624466

theorem sqrt_pos_condition (x : ℝ) : (1 - x) ≥ 0 ↔ x ≤ 1 := 
by 
  sorry

end sqrt_pos_condition_l624_624466


namespace max_heaps_660_stones_l624_624029

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624029


namespace sum_of_squares_increase_by_l624_624827

theorem sum_of_squares_increase_by {n : ℕ} (h1 : n = 100) (x : Fin n.succ → ℝ)
  (h2 : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  (∑ i, (x i + 4) ^ 2) = (∑ i, x i ^ 2) + 800 :=
by sorry

end sum_of_squares_increase_by_l624_624827


namespace sum_of_fractions_l624_624713

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624713


namespace fraction_addition_l624_624614

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624614


namespace find_constant_t_l624_624903

theorem find_constant_t (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t + 5^n) ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ (a 1 = S 1) ∧ 
  (∃ q, ∀ n ≥ 1, a (n + 1) = q * a n) → 
  t = -1 := by
  sorry

end find_constant_t_l624_624903


namespace hyperbola_equation_l624_624844

-- Definitions of the conditions
def is_asymptote_1 (y x : ℝ) : Prop :=
  y = 2 * x

def is_asymptote_2 (y x : ℝ) : Prop :=
  y = -2 * x

def passes_through_focus (x y : ℝ) : Prop :=
  x = 1 ∧ y = 0

-- The statement to be proved
theorem hyperbola_equation :
  (∀ x y : ℝ, passes_through_focus x y → x^2 - (y^2 / 4) = 1) :=
sorry

end hyperbola_equation_l624_624844


namespace fraction_addition_l624_624630

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624630


namespace add_fractions_result_l624_624587

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624587


namespace add_fractions_l624_624727

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end add_fractions_l624_624727


namespace unique_m_l624_624762

theorem unique_m :
  ∃! m, m > 0 ∧ log 4 (log 32 m) = log 8 (log 8 m) ∧ m = 32768 :=
by
  sorry

end unique_m_l624_624762


namespace domain_of_function_l624_624262

theorem domain_of_function (f : ℝ → ℝ) (h₀ : Set.Ioo 0 1 ⊆ {x | f (3 * x + 2)}) :
  Set.Ioo (3 / 2) 3 ⊆ {x | f (2 * x - 1)} :=
by
  sorry

end domain_of_function_l624_624262


namespace fraction_addition_l624_624643

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624643


namespace quilt_cost_calculation_l624_624914

theorem quilt_cost_calculation :
  let length := 12
  let width := 15
  let cost_per_sq_foot := 70
  let sales_tax_rate := 0.05
  let discount_rate := 0.10
  let area := length * width
  let cost_before_discount := area * cost_per_sq_foot
  let discount_amount := cost_before_discount * discount_rate
  let cost_after_discount := cost_before_discount - discount_amount
  let sales_tax_amount := cost_after_discount * sales_tax_rate
  let total_cost := cost_after_discount + sales_tax_amount
  total_cost = 11907 := by
  {
    sorry
  }

end quilt_cost_calculation_l624_624914


namespace sum_f_from_n2023_to_2023_l624_624940

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x - Real.exp (-x) + Real.sin x + 1

theorem sum_f_from_n2023_to_2023 : 
  ∑ x in Finset.range (2 * 2023 + 1) \ {0}, (f (-2023 + x) + f (2023 - x)) = 2 * 2023 * 2 ∧
  f 0 = 1 ∧
  ∑ x in (Finset.range (2 * 2023 + 1)), f (-2023 + x) = 4047 :=
by
  let start_n := -2023
  let end_n := 2023
  have h1: ∀ x: ℝ, f (x) + f (-x) = 2 := sorry
  show ∑ x in Finset.range (2 * end_n + 1) \ {0}, (f (start_n + x) + f (end_n - x)) = 2 * end_n * 2 ∧
  f 0 = 1 ∧
  ∑ x in Finset.range (2 * end_n + 1), f (start_n + x) = 4047 from sorry

end sum_f_from_n2023_to_2023_l624_624940


namespace sum_fractions_eq_l624_624687

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624687


namespace fraction_addition_l624_624641

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624641


namespace remainder_N_mod_2017_l624_624103

def total_voters : ℕ := 2015
def percent_for_drum : ℚ := 60 / 100

-- Calculate the number of votes for Drum
def votes_for_drum : ℕ := (percent_for_drum * total_voters).toNat -- This will give 1209

-- Theorem stating the result modulo 2017
theorem remainder_N_mod_2017 :
  let N := ∑ k in finset.Icc 807 total_voters, nat.choose total_voters k
  N % 2017 = 605 := 
sorry

end remainder_N_mod_2017_l624_624103


namespace total_dolls_count_l624_624169

-- Define the conditions
def big_box_dolls : Nat := 7
def small_box_dolls : Nat := 4
def num_big_boxes : Nat := 5
def num_small_boxes : Nat := 9

-- State the theorem that needs to be proved
theorem total_dolls_count : 
  big_box_dolls * num_big_boxes + small_box_dolls * num_small_boxes = 71 := 
by
  sorry

end total_dolls_count_l624_624169


namespace triangle_area_inscribed_in_circle_l624_624540

def inscribed_triangle_area (R : ℝ) : ℝ := 
  (Real.sqrt 3 / 4) * R^2

theorem triangle_area_inscribed_in_circle (R : ℝ) (hR : 0 < R) :
  ∃ (area : ℝ), area = inscribed_triangle_area R ∧ 
  ∀ (A B C : ℝ) (hA : A = 15) (hB : B = 60) (hC : C = 105) (hSum : A + B + C = 180),
  ∃ (area : ℝ), area = inscribed_triangle_area R := 
sorry

end triangle_area_inscribed_in_circle_l624_624540


namespace function_equality_l624_624206

theorem function_equality
  (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x = max (2 * x * y - f y)) →
  (∀ x : ℝ, f x = x^2) :=
by
  assume h : ∀ x y : ℝ, f x = max (2 * x * y - f y)
  intro x
  -- proof goes here
  sorry

end function_equality_l624_624206


namespace coplanar_lines_and_plane_l624_624792

-- Definitions of plane and line
variables {plane : Type} {line : Type} 
variables {m n : line} {α : plane}

-- Definition of conditions
def is_coplanar (m n : line) : Prop := ∃ α : plane, m ⊆ α ∧ n ⊆ α
def is_parallel (m n : line) : Prop := ∀ x ∈ m, ∀ y ∈ n, x ≠ y
def is_subset (m : line) (α : plane) : Prop := m ⊆ α 
def is_parallel_to_plane (n : line) (α : plane) : Prop := ∀ x ∈ n, ∀ y ∈ α, x ≠ y

-- Theorem statement
theorem coplanar_lines_and_plane (m n : line) (α : plane) 
  (h1 : is_coplanar m n) 
  (h2 : is_subset m α)
  (h3 : is_parallel_to_plane n α) : is_parallel m n :=
sorry

end coplanar_lines_and_plane_l624_624792


namespace expression_evaluation_l624_624978

noncomputable def x := Real.sqrt 5 + 1
noncomputable def y := Real.sqrt 5 - 1

theorem expression_evaluation : 
  ( ( (5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2) ) / (1 / (x^2 * y - x * y^2)) ) = 12 := 
by 
  -- Provide a proof here
  sorry

end expression_evaluation_l624_624978


namespace inequality_holds_l624_624795

theorem inequality_holds (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  (a^3 / (a^3 + 15 * b * c * d))^(1/2) ≥ a^(15/8) / (a^(15/8) + b^(15/8) + c^(15/8) + d^(15/8)) :=
sorry

end inequality_holds_l624_624795


namespace max_value_of_x2_plus_y2_l624_624855

theorem max_value_of_x2_plus_y2 {x y : ℝ} 
  (h1 : x ≥ 1)
  (h2 : y ≥ x)
  (h3 : x - 2 * y + 3 ≥ 0) : 
  x^2 + y^2 ≤ 18 :=
sorry

end max_value_of_x2_plus_y2_l624_624855


namespace sin_double_angle_ordering_l624_624549

variable (A B C : ℝ) -- Define the angles of the triangle as real numbers
variable (h_triangle : 0 < A ∧ A < B ∧ B < C ∧ C < π / 2) -- Conditions for an acute-angled triangle

theorem sin_double_angle_ordering (h_sum : A + B + C = π) : sin (2 * A) > sin (2 * B) ∧ sin (2 * B) > sin (2 * C) :=
  sorry

-- Additional lemma to state all angles are positive
lemma all_angles_positive : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π := by
  sorry

end sin_double_angle_ordering_l624_624549


namespace total_red_marbles_l624_624355

-- Definitions derived from the problem conditions
def Jessica_red_marbles : ℕ := 3 * 12
def Sandy_red_marbles : ℕ := 4 * Jessica_red_marbles
def Alex_red_marbles : ℕ := Jessica_red_marbles + 2 * 12

-- Statement we need to prove that total number of marbles is 240
theorem total_red_marbles : 
  Jessica_red_marbles + Sandy_red_marbles + Alex_red_marbles = 240 := by
  -- We provide the proof later
  sorry

end total_red_marbles_l624_624355


namespace find_unknown_numbers_l624_624310

def satisfies_condition1 (A B : ℚ) : Prop := 
  0.05 * A = 0.20 * 650 + 0.10 * B

def satisfies_condition2 (A B : ℚ) : Prop := 
  A + B = 4000

def satisfies_condition3 (B C : ℚ) : Prop := 
  C = 2 * B

def satisfies_condition4 (A B C D : ℚ) : Prop := 
  A + B + C = 0.40 * D

theorem find_unknown_numbers (A B C D : ℚ) :
  satisfies_condition1 A B → satisfies_condition2 A B →
  satisfies_condition3 B C → satisfies_condition4 A B C D →
  A = 3533 + 1/3 ∧ B = 466 + 2/3 ∧ C = 933 + 1/3 ∧ D = 12333 + 1/3 :=
by
  sorry

end find_unknown_numbers_l624_624310


namespace max_heaps_660_stones_l624_624025

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624025


namespace find_a_b_c_l624_624070

theorem find_a_b_c :
  ∃ a b c : ℕ, a = 1 ∧ b = 17 ∧ c = 2 ∧ (Nat.gcd a c = 1) ∧ a + b + c = 20 :=
by {
  -- the proof would go here
  sorry
}

end find_a_b_c_l624_624070


namespace max_heaps_660_l624_624005

-- Define the conditions and goal
theorem max_heaps_660 (h : ∀ (a b : ℕ), a ∈ heaps → b ∈ heaps → a ≤ b → b < 2 * a) :
  ∃ heaps : finset ℕ, heaps.sum id = 660 ∧ heaps.card = 30 :=
by
  -- Initial definitions
  have : ∀ (heaps : finset ℕ), heaps.sum id = 660 → heaps.card ≤ 30,
  sorry
  -- Construct existence of heaps with the required conditions
  refine ⟨{15, 15, 16, 16, 17, 17, 18, 18, ..., 29, 29}.to_finset, _, _⟩,
  sorry

end max_heaps_660_l624_624005


namespace sequence_general_formula_l624_624302

def S (n : ℕ) : ℤ := -n^2 + 6*n + 7

def a (n : ℕ) : ℤ :=
 if n = 1 then 12 else -2*n + 7

theorem sequence_general_formula (n : ℕ) : 
  (a n) = (if n = 1 then 12 else -((-n)^2 + 6*n + 7) + ((-(n-1)^2 + 6*(n-1) + 7))) :=
sorry

end sequence_general_formula_l624_624302


namespace factor_expression_l624_624179

theorem factor_expression (x : ℝ) : 16 * x ^ 2 + 8 * x = 8 * x * (2 * x + 1) :=
by
  -- Problem: Completely factor the expression
  -- Given Condition
  -- Conclusion
  sorry

end factor_expression_l624_624179


namespace meena_cookies_left_l624_624946

-- Define the given conditions in terms of Lean definitions
def total_cookies_baked := 5 * 12
def cookies_sold_to_stone := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

-- Define the total cookies sold
def total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy

-- Define the number of cookies left
def cookies_left := total_cookies_baked - total_cookies_sold

-- Prove that the number of cookies left is 15
theorem meena_cookies_left : cookies_left = 15 := by
  -- The proof is omitted (sorry is used to skip proof)
  sorry

end meena_cookies_left_l624_624946


namespace floor_abs_add_abs_floor_neg_5_7_l624_624564

theorem floor_abs_add_abs_floor_neg_5_7 : 
  (Int.floor (abs (-5.7)) + abs (Int.floor (-5.7))) = 11 :=
by 
  sorry

end floor_abs_add_abs_floor_neg_5_7_l624_624564


namespace range_of_eccentricities_l624_624097

theorem range_of_eccentricities :
  ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ) (λ : ℝ), 
  (λ ≥ 1) →
  P = (P.1, P.2) → 
  (P.1^2 / 3 + P.2^2 / 2 = 1) →
  -- Definition of H as (3, P.2)
  let H := (3 : ℝ, P.2) in
  -- Definition of Q in terms of P, H and λ
  Q = (λ * P.1 + 3 * (1 + λ) / λ, P.2) →
  let eccentricity := sqrt(1 - 2 / (3 * λ^2)) in
  eccentricity ∈ set.Ico (sqrt 3 / 3) 1 :=
sorry

end range_of_eccentricities_l624_624097


namespace maximum_value_of_expression_l624_624371

-- Define the given condition
def condition (a b c : ℝ) : Prop := a + 3 * b + c = 5

-- Define the objective function
def objective (a b c : ℝ) : ℝ := a * b + a * c + b * c

-- Main theorem statement
theorem maximum_value_of_expression (a b c : ℝ) (h : condition a b c) : 
  ∃ (a b c : ℝ), condition a b c ∧ objective a b c = 25 / 3 :=
sorry

end maximum_value_of_expression_l624_624371


namespace six_is_not_good_four_is_good_ten_twenty_four_is_good_l624_624894

theorem six_is_not_good (n : ℕ) (S : ℕ) (board : fin n → fin n → ℕ) :
  n = 6 → ¬ (∀ board, ∃ k, ∀ x y, board x y = k) :=
by sorry

theorem four_is_good (n : ℕ) (S : ℕ) (board : fin n → fin n → ℕ) :
  n = 4 → (∀ board, ∃ k, ∀ x y, board x y = k) :=
by sorry

theorem ten_twenty_four_is_good (n : ℕ) (S : ℕ) (board : fin n → fin n → ℕ) :
  n = 1024 → (∀ board, ∃ k, ∀ x y, board x y = k) :=
by sorry

end six_is_not_good_four_is_good_ten_twenty_four_is_good_l624_624894


namespace sum_fractions_eq_l624_624696

theorem sum_fractions_eq (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end sum_fractions_eq_l624_624696


namespace theta_of_complex_1_i_sqrt3_eq_pi_div3_l624_624486

noncomputable def theta_of_complex (re im : ℝ) : ℝ :=
  complex.arg (complex.mk re im)

theorem theta_of_complex_1_i_sqrt3_eq_pi_div3 :
  theta_of_complex 1 (real.sqrt 3) = real.pi / 3 := 
sorry

end theta_of_complex_1_i_sqrt3_eq_pi_div3_l624_624486


namespace add_fractions_l624_624658

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624658


namespace orthocenter_incircle_midpoint_l624_624345

/-- In triangle ABC, if M is the orthocenter,
and the incircle touches AC at P and BC at Q with center O,
prove that if M lies on the line PQ, then the line MO passes through the midpoint of side AB. -/
theorem orthocenter_incircle_midpoint
  (A B C M P Q O : Type)
  [T : Triangle A B C]
  (h_orthocenter : is_orthocenter M T)
  (h_incircle_touches_AC : touches_incircle_at T AC P O)
  (h_incircle_touches_BC : touches_incircle_at T BC Q O)
  (h_M_on_PQ : lies_on M PQ) :
  passes_through_midpoint MO AB :=
sorry

end orthocenter_incircle_midpoint_l624_624345


namespace fraction_addition_l624_624680

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624680


namespace sum_of_squares_increase_by_l624_624823

theorem sum_of_squares_increase_by {n : ℕ} (h1 : n = 100) (x : Fin n.succ → ℝ)
  (h2 : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  (∑ i, (x i + 4) ^ 2) = (∑ i, x i ^ 2) + 800 :=
by sorry

end sum_of_squares_increase_by_l624_624823


namespace find_f_l624_624159

-- defining the function transformation
def transform_func (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x, - (f (x - π) * sin (x - π))

-- defining the target function
def target_func : ℝ → ℝ :=
  λ x, 1 - 2 * sin x ^ 2

-- stating the Lean theorem
theorem find_f (f : ℝ → ℝ) :
  (∀ x, transform_func f x = target_func x) → f = λ x, 2 * cos x :=
by sorry

end find_f_l624_624159


namespace two_digit_number_representation_l624_624793

theorem two_digit_number_representation (m n : ℕ) (hm : m < 10) (hn : n < 10) : 10 * n + m = m + 10 * n :=
by sorry

end two_digit_number_representation_l624_624793


namespace conjugate_of_z_l624_624994

/-- Given a complex number z satisfying z(-1+i) = |1+3i|^2, prove that the conjugate of z is -5+5i. -/
theorem conjugate_of_z (z : ℂ) (h : z * (-1 + complex.I) = complex.abs (1 + 3 * complex.I) ^ 2) : 
  complex.conj z = -5 + 5 * complex.I :=
sorry

end conjugate_of_z_l624_624994


namespace side_length_of_square_ground_l624_624539

theorem side_length_of_square_ground
    (radius : ℝ)
    (Q_area : ℝ)
    (pi : ℝ)
    (quarter_circle_area : Q_area = (pi * (radius^2) / 4))
    (pi_approx : pi = 3.141592653589793)
    (Q_area_val : Q_area = 15393.804002589986)
    (radius_val : radius = 140) :
    ∃ (s : ℝ), s^2 = radius^2 :=
by
  sorry -- Proof not required per the instructions

end side_length_of_square_ground_l624_624539


namespace find_m_l624_624917

noncomputable def sequence (x : ℕ → ℕ) : Prop :=
  (x 1 = 3) ∧ (∀ n : ℕ, x (n+1) = floor (sqrt 2 * x n))

def arithmetic_progression (x : ℕ → ℕ) (m : ℕ) : Prop :=
  (x (m + 2) = 2 * x (m + 1) - x m)

theorem find_m (x : ℕ → ℕ) (h_seq : sequence x) : ∃ (m : ℤ), m = 1 ∨ m = 3 := 
sorry

end find_m_l624_624917


namespace total_pages_in_book_l624_624750

def pages_already_read : ℕ := 147
def pages_left_to_read : ℕ := 416

theorem total_pages_in_book : pages_already_read + pages_left_to_read = 563 := by
  sorry

end total_pages_in_book_l624_624750


namespace rationalize_fraction_l624_624058

theorem rationalize_fraction :
  (5 : ℚ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = 
  (5 * Real.sqrt 2) / 36 :=
by
  sorry

end rationalize_fraction_l624_624058


namespace fraction_addition_l624_624611

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end fraction_addition_l624_624611


namespace find_rectangle_pairs_l624_624943

theorem find_rectangle_pairs (w l : ℕ) (hw : w > 0) (hl : l > 0) (h : w * l = 18) : 
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) ∨
  (w, l) = (6, 3) ∨ (w, l) = (9, 2) ∨ (w, l) = (18, 1) :=
by
  sorry

end find_rectangle_pairs_l624_624943


namespace trapezoid_angles_and_k_l624_624086

noncomputable def angles_and_k (k : Real) : Prop :=
  let α := Real.arcsin (2 * (1 + k) / (π * k^2))
  let β := Real.arcsin (2 * (1 + k) / (π * k))
  α ≥ 0 ∧ α ≤ π ∧ β ≥ 0 ∧ β ≤ π ∧ k ≥ 2 / (π - 2)

theorem trapezoid_angles_and_k :
  ∀ (k : Real), k ≥ 2 / (π - 2) → angles_and_k k := by
  intros k h
  dsimp [angles_and_k]
  sorry

end trapezoid_angles_and_k_l624_624086


namespace place_circle_without_overlap_l624_624323

open Function

theorem place_circle_without_overlap (R : set (ℝ × ℝ)) (hR_dim : ∃ w h, R = set.Icc (0, 0) (w, h) ∧ w = 20 ∧ h = 25)
  (S : ℕ → set (ℝ × ℝ)) (hS_count : ∃ n, n = 120)
  (hS_dims : ∀ i, i < 120 → ∃ x y, S i = set.Icc (x, y) (x+1, y+1)) :
  ∃ c : ℝ × ℝ, set.Icc (c.1 - 0.5, c.2 - 0.5) (c.1 + 0.5, c.2 + 0.5) ⊆ R ∧
  (∀ i, i < 120 → set.Icc (c.1 - 0.5, c.2 - 0.5) (c.1 + 0.5, c.2 + 0.5) ∩ S i = ∅) := 
sorry

end place_circle_without_overlap_l624_624323


namespace fraction_addition_l624_624608

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624608


namespace number_of_integers_with_1_and_2_left_of_3_l624_624997

theorem number_of_integers_with_1_and_2_left_of_3 : 
  let digits := {1, 2, 3, 4, 5, 6}
  in let permutations := list.permutations digits.to_list
  in let valid_permutations := 
      permutations.filter (λ p, p.index_of 1 < p.index_of 3 ∧ p.index_of 2 < p.index_of 3)
  in valid_permutations.length = 240 :=
sorry

end number_of_integers_with_1_and_2_left_of_3_l624_624997


namespace functions_with_corridor_of_width_2_l624_624791

def has_corridor_of_width (f : ℝ → ℝ) (D : set ℝ) (d : ℝ) : Prop :=
  ∃ (k m1 m2 : ℝ), (∀ x ∈ D, k * x + m1 ≤ f x ∧ f x ≤ k * x + m2) ∧ (abs (m1 - m2) = d)

def func1 (x : ℝ) : ℝ := x^2
def domain1 : set ℝ := {x | 0 ≤ x}

def func2 (x : ℝ) : ℝ := real.sqrt (4 - x^2)
def domain2 : set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

def func3 (x : ℝ) : ℝ :=
  if x ≤ 0 then real.exp x - 1 else 1 - real.exp (-x)
def domain3 : set ℝ := set.univ

def func4 (x : ℝ) : ℝ := 2 / x
def domain4 : set ℝ := {x | abs x ≥ 4}

theorem functions_with_corridor_of_width_2 :
  has_corridor_of_width func2 domain2 2 ∧
  has_corridor_of_width func3 domain3 2 ∧
  has_corridor_of_width func4 domain4 2 :=
by
  admit -- This is a placeholder for the actual proof.

end functions_with_corridor_of_width_2_l624_624791


namespace vector_magnitude_range_l624_624845

variables {x y : ℝ}
variables (e1 e2 : ℝ)

def angle_between_unit_vectors (e1 e2 : ℝ) (θ : ℝ) : Prop := e1 * e2 = cos θ
def vector_magnitude (x y : ℝ) (e1 e2 : ℝ) : ℝ := real.sqrt (x^2 + y^2 - x * y)

theorem vector_magnitude_range
  (angle_between_unit_vectors (1:ℝ) (1:ℝ) (2 * real.pi / 3))
  (h : vector_magnitude x y 1 1 = real.sqrt 3) :
  1 ≤ vector_magnitude x y 1 1 ∧ vector_magnitude x y 1 1 ≤ 3 :=
by
  sorry

end vector_magnitude_range_l624_624845


namespace train_length_l624_624873

-- Definitions for conditions
def speed_kmph := 160
def time_seconds := 18
def speed_mps := (speed_kmph * 1000) / 3600

-- Statement of the problem
theorem train_length : 
    let distance := speed_mps * time_seconds in
    abs (distance - 800) < 1 :=
by
  sorry

end train_length_l624_624873


namespace max_piles_l624_624000

theorem max_piles (n : ℕ) (m : ℕ) : 
  (∀ x y : ℕ, x ∈ n ∧ y ∈ n → x < 2 * y → x > 0) → 
  ( ∑ i in n.to_finset, i) = m → 
  m = 660 →
  n.card = 30 :=
sorry

end max_piles_l624_624000


namespace find_second_number_l624_624783

theorem find_second_number (n : ℕ) 
  (h1 : Nat.lcm 24 (Nat.lcm n 42) = 504)
  (h2 : 504 = 2^3 * 3^2 * 7) 
  (h3 : Nat.lcm 24 42 = 168) : n = 3 := 
by 
  sorry

end find_second_number_l624_624783


namespace remainder_x2y_eq_x_mod_m_l624_624375

variables {m x y : ℕ}
-- m is a positive integer and x, y are invertible under modulo m
variables (hm : m > 0) (hx : IsInvertible x m) (hy : IsInvertible y m)
-- x ≡ y⁻¹ (mod m)
variables (hxy : x ≡ y⁻¹ [MOD m])

theorem remainder_x2y_eq_x_mod_m : (x^2 * y) % m = x % m :=
by
  sorry

end remainder_x2y_eq_x_mod_m_l624_624375


namespace rectangular_plot_breadth_l624_624076

theorem rectangular_plot_breadth :
  ∀ (l b : ℝ), (l = 3 * b) → (l * b = 588) → (b = 14) :=
by
  intros l b h1 h2
  sorry

end rectangular_plot_breadth_l624_624076


namespace sum_of_five_primes_multiple_of_4_l624_624219

open Finset

theorem sum_of_five_primes_multiple_of_4 :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
  let total_ways := (primes.toFinset.choose 5).card
  let favorable_ways := ((primes.toFinset.erase 2).choose 4).card
  (favorable_ways / total_ways : ℚ) = 55 / 528 := by
  sorry

end sum_of_five_primes_multiple_of_4_l624_624219


namespace average_age_of_3_students_l624_624989

theorem average_age_of_3_students :
  ∀ (total_students : ℕ) (avg_total_age : ℕ) (num_students11 : ℕ) (avg_age11 : ℕ) (age15th : ℕ),
    total_students = 15 →
    avg_total_age = 15 →
    num_students11 = 11 →
    avg_age11 = 16 →
    age15th = 7 →
    (42 / 3 = 14) :=
by
  intros total_students avg_total_age num_students11 avg_age11 age15th
  intros ht hs tr ar af

  have h1 : total_students = 15 := ht
  have h2 : avg_total_age = 15 := hs
  have h3 : num_students11 = 11 := tr
  have h4 : avg_age11 = 16 := ar
  have h5 : age15th = 7 := af

  calc
    (225 - 176 - 7) / 3 = 42 / 3 := ...
    42 / 3 = 14 := rfl


end average_age_of_3_students_l624_624989


namespace sum_of_fractions_l624_624712

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624712


namespace exists_possible_event_l624_624132

structure Girl :=
  (name : String)
  (beauty : ℕ)
  (intelligence : ℕ)

structure Dance :=
  (man : String)
  (girl : Girl)

def possible_event (girls : List Girl) (dances : List Dance) :=
  (∀ i : ℕ, i < dances.length - 1 → 
    (dances[i].girl.beauty < dances[i+1].girl.beauty ∨ dances[i].girl.intelligence < dances[i+1].girl.intelligence)) ∧
  ∃ i : ℕ, i < dances.length - 1 ∧
    (dances[i].girl.beauty < dances[i+1].girl.beauty ∧ dances[i].girl.intelligence < dances[i+1].girl.intelligence)

def girls := [
  Girl.mk "Anna" 1 3, 
  Girl.mk "Vera" 2 2, 
  Girl.mk "Svetlana" 3 1
]

def dances := [
  Dance.mk "Man1" (Girls[0]),
  Dance.mk "Man1" (Girls[1]),
  Dance.mk "Man2" (Girls[2]),
  Dance.mk "Man2" (Girls[0]),
  Dance.mk "Man3" (Girls[1]),
  Dance.mk "Man3" (Girls[2])
]

theorem exists_possible_event : possible_event girls dances := by
  sorry

end exists_possible_event_l624_624132


namespace find_third_number_l624_624064

-- Definitions
def A : ℕ := 600
def B : ℕ := 840
def LCM : ℕ := 50400
def HCF : ℕ := 60

-- Theorem to be proven
theorem find_third_number (C : ℕ) (h_lcm : Nat.lcm (Nat.lcm A B) C = LCM) (h_hcf : Nat.gcd (Nat.gcd A B) C = HCF) : C = 6 :=
by -- proof
  sorry

end find_third_number_l624_624064


namespace add_fractions_l624_624565

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624565


namespace coefficient_x3_in_expansion_l624_624993

theorem coefficient_x3_in_expansion :
  (let a := (x^2 : ℚ[x]); b := (1/x : ℚ[x]) in
   polynomial.coeff ((a + b)^6) 3) = 20 :=
by 
  -- The proof details would go here normally, but we're skipping it as per instructions.
  sorry

end coefficient_x3_in_expansion_l624_624993


namespace add_fractions_result_l624_624585

noncomputable def add_fractions (a : ℝ) (h : a ≠ 0): ℝ := (3 / a) + (2 / a)

theorem add_fractions_result (a : ℝ) (h : a ≠ 0) : add_fractions a h = 5 / a := by
  sorry

end add_fractions_result_l624_624585


namespace determine_OP_l624_624218

theorem determine_OP 
  (a b c d k : ℝ)
  (h1 : k * b ≤ c) 
  (h2 : (A : ℝ) = a)
  (h3 : (B : ℝ) = k * b)
  (h4 : (C : ℝ) = c)
  (h5 : (D : ℝ) = k * d)
  (AP_PD : ∀ (P : ℝ), (a - P) / (P - k * d) = k * (k * b - P) / (P - c))
  :
  ∃ P : ℝ, P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
sorry

end determine_OP_l624_624218


namespace volume_ratio_of_rotated_solids_l624_624448

theorem volume_ratio_of_rotated_solids (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := π * b^2 * a
  let V2 := π * a^2 * b
  V1 / V2 = b / a :=
by
  intros
  -- Proof omitted
  sorry

end volume_ratio_of_rotated_solids_l624_624448


namespace sqrt_meaningful_condition_l624_624473

theorem sqrt_meaningful_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (1 - x)) → x ≤ 1 :=
by
  assume h,
  sorry

end sqrt_meaningful_condition_l624_624473


namespace fraction_addition_l624_624649

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624649


namespace sum_of_fractions_l624_624704

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624704


namespace polynomial_sequence_properties_l624_624919

noncomputable def sequence_f (a : ℝ) : ℕ → ℝ[X]
| 0       := 1
| (n + 1) := λ x, x * (sequence_f a n) x + (sequence_f a n) (a * x)

/-- Prove the given sequence of polynomials satisfies the properties --/
theorem polynomial_sequence_properties (a : ℝ) : 
    (∀ n : ℕ, ∀ x : ℝ, (sequence_f a n) x = x^n * (sequence_f a n) x⁻¹) 
  ∧ (∀ n : ℕ, ∀ x : ℝ, (sequence_f a n) x = ∑ k in Finset.range(n + 1), 
                         (Nat.choose n k) * a^(k * (k - 1) / 2) * x^k) := by 
  sorry

end polynomial_sequence_properties_l624_624919


namespace minimal_volume_around_sphere_l624_624124

noncomputable def minimal_cone_volume (r : ℝ) : ℝ :=
  (8 * r^3 * Real.pi) / 3

theorem minimal_volume_around_sphere (r : ℝ) : 
  ∃ V, minimal_cone_volume r = V ∧ V = (8 * r^3 * Real.pi) / 3 := 
by {
  use minimal_cone_volume r,
  split,
  refl,
  refl,
}

end minimal_volume_around_sphere_l624_624124


namespace incorrect_judgment_D_l624_624492

/-- Prove that the judgment D is incorrect given conditions A, B, and C -/
theorem incorrect_judgment_D (a b m : ℝ) (p q : Prop) (B : B 4 0.25 → var) :
  (∀ a b m, (a * m^2 < b * m^2) → (m^2 > 0) → (a < b)) ∧
  (¬ (∀ x ∈ ℝ, x^3 - x^2 - 1 ≤ 0) ↔ (∃ x₀ ∈ ℝ, x₀^3 - x₀^2 - 1 > 0)) ∧
  (¬ p ∧ ¬ q → ¬ (p ∧ q)) →
  ¬ (B)
:= by
  sorry

end incorrect_judgment_D_l624_624492


namespace geometric_sequence_seventh_term_l624_624439

theorem geometric_sequence_seventh_term (a r : ℝ) 
  (h4 : a * r^3 = 16) 
  (h9 : a * r^8 = 2) : 
  a * r^6 = 8 := 
sorry

end geometric_sequence_seventh_term_l624_624439


namespace sin_cos_sum_of_cos2theta_l624_624869

theorem sin_cos_sum_of_cos2theta (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < π / 2) (h2 : cos (2 * θ) = b) :
  sin θ + cos θ = sqrt (2 - b) :=
by
  sorry

end sin_cos_sum_of_cos2theta_l624_624869


namespace fraction_addition_l624_624629

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624629


namespace div_by_3_iff_sum_digits_div_by_3_div_by_9_iff_sum_digits_div_by_9_l624_624053

open Nat

theorem div_by_3_iff_sum_digits_div_by_3 (n : ℕ) (a : ℕ → ℕ) (h : n = ∑ i in range (nat.digits 10 n).length, a i * 10 ^ i) :
  (n % 3 = 0 ↔ ∑ i in range (nat.digits 10 n).length, a i % 3 = 0) :=
sorry

theorem div_by_9_iff_sum_digits_div_by_9 (n : ℕ) (a : ℕ → ℕ) (h : n = ∑ i in range (nat.digits 10 n).length, a i * 10 ^ i) :
  (n % 9 = 0 ↔ ∑ i in range (nat.digits 10 n).length, a i % 9 = 0) :=
sorry

end div_by_3_iff_sum_digits_div_by_3_div_by_9_iff_sum_digits_div_by_9_l624_624053


namespace transform_sin_to_cos_l624_624101

theorem transform_sin_to_cos :
  ∀ x : ℝ, cos (2 * x) = sin (2 * (x + π / 4)) :=
by sorry

end transform_sin_to_cos_l624_624101


namespace sum_of_squares_increase_by_l624_624826

theorem sum_of_squares_increase_by {n : ℕ} (h1 : n = 100) (x : Fin n.succ → ℝ)
  (h2 : ∑ i, x i ^ 2 = ∑ i, (x i + 2) ^ 2) :
  (∑ i, (x i + 4) ^ 2) = (∑ i, x i ^ 2) + 800 :=
by sorry

end sum_of_squares_increase_by_l624_624826


namespace constant_term_in_expansion_term_with_largest_coefficient_l624_624260

-- Given condition
def binomial_coefficient_condition (n : ℕ) (term_idx : ℕ) : Prop :=
  n = 11 ∧ term_idx = 5

-- Prove the constant term in the expansion
theorem constant_term_in_expansion (n : ℕ) (term_idx : ℕ) 
  (h : binomial_coefficient_condition n term_idx) : 
  (∑ r in finset.range(n + 1), binomial (n) r * (2:ℤ)^(n - r) * x^(11 - (4 * r / 3)) ) = 1232 := by
    sorry

-- Prove the term with the largest coefficient in the expansion
theorem term_with_largest_coefficient (n : ℕ) (term_idx : ℕ) 
  (h : binomial_coefficient_condition n term_idx) : 
  ∃ T₆ T₇, 
    (T₆ = nat.choose 11 5 * (2:ℤ)^6 * x^(10 / 3)) ∧ 
    (T₇ = nat.choose 11 6 * (2:ℤ)^5 * x^4) ∧ 
    T₆ = 1232 * x^(10 / 3) ∧ 
    T₇ = 1232 * x^4 := by
    sorry

end constant_term_in_expansion_term_with_largest_coefficient_l624_624260


namespace max_heaps_660_stones_l624_624032

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624032


namespace a_minus_b_eq_10_l624_624514

variables (a b c : ℝ)

-- Conditions
def condition1 := a = 1/3 * (b + c)
def condition2 := b = 2/7 * (a + c)
def condition3 := a + b + c = 360

-- The theorem we want to prove
theorem a_minus_b_eq_10 (h1 : condition1) (h2 : condition2) (h3 : condition3) : a - b = 10 :=
sorry

end a_minus_b_eq_10_l624_624514


namespace symmetric_point_l624_624068

theorem symmetric_point (x0 y0 : ℝ) (P : ℝ × ℝ) (line : ℝ → ℝ) 
  (hP : P = (-1, 3)) (hline : ∀ x, line x = x) :
  ((x0, y0) = (3, -1)) ↔
    ( ∃ M : ℝ × ℝ, M = ((x0 - -1) / 2, (y0 + 3) / 2) ∧ M.1 = M.2 ) ∧ 
    ( ∃ l : ℝ, l = (y0 - 3) / (x0 + 1) ∧ l = -1 ) :=
by
  sorry

end symmetric_point_l624_624068


namespace relationship_among_a_b_c_l624_624306

noncomputable def a : ℝ := 0.3^2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := 2^0.3

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l624_624306


namespace convex_polyhedron_P_T_V_sum_eq_34_l624_624145

theorem convex_polyhedron_P_T_V_sum_eq_34
  (F : ℕ) (V : ℕ) (E : ℕ) (T : ℕ) (P : ℕ) 
  (hF : F = 32)
  (hT1 : 3 * T + 5 * P = 960)
  (hT2 : 2 * E = V * (T + P))
  (hT3 : T + P - 2 = 60)
  (hT4 : F + V - E = 2) :
  P + T + V = 34 := by
  sorry

end convex_polyhedron_P_T_V_sum_eq_34_l624_624145


namespace coupon_percentage_l624_624416

theorem coupon_percentage (P i d final_price total_price discount_amount percentage: ℝ)
  (h1 : P = 54) (h2 : i = 20) (h3 : d = 0.20 * i) 
  (h4 : total_price = P - d) (h5 : final_price = 45) 
  (h6 : discount_amount = total_price - final_price) 
  (h7 : percentage = (discount_amount / total_price) * 100) : 
  percentage = 10 := 
by
  sorry

end coupon_percentage_l624_624416


namespace total_dolls_l624_624166

theorem total_dolls (big_boxes : ℕ) (dolls_per_big_box : ℕ) (small_boxes : ℕ) (dolls_per_small_box : ℕ)
  (h1 : dolls_per_big_box = 7) (h2 : big_boxes = 5) (h3 : dolls_per_small_box = 4) (h4 : small_boxes = 9) :
  big_boxes * dolls_per_big_box + small_boxes * dolls_per_small_box = 71 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  exact rfl

end total_dolls_l624_624166


namespace john_average_speed_l624_624357

/-- Definitions based on the problem conditions -/

def t1 : ℝ := 20 / 60
def s1 : ℝ := 20
def t2 : ℝ := 120 / 60
def s2 : ℝ := 5

def d1 : ℝ := s1 * t1
def d2 : ℝ := s2 * t2
def d_total : ℝ := d1 + d2
def t_total : ℝ := t1 + t2

/-- Statement to prove -/
theorem john_average_speed : d_total / t_total = 7 := 
by sorry

end john_average_speed_l624_624357


namespace add_fractions_l624_624578

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624578


namespace add_fractions_l624_624574

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624574


namespace parallel_lines_in_non_parallel_planes_l624_624347

variable {ℝ : Type*} [LinearOrderedField ℝ]

noncomputable
def is_possible_to_draw_parallel_lines
  (π1 π2 : set (ℝ × ℝ × ℝ))
  (h1 : ¬ parallel π1 π2)
  (L : ℝ × ℝ × ℝ → Prop)
  (hL : L ∈ π1 ∩ π2) : Prop :=
∃ (A1 A2 : ℝ × ℝ × ℝ → Prop),
  (A1 ∈ π1) ∧ (A2 ∈ π2) ∧ (parallel A1 L) ∧ (parallel A2 L) ∧ (parallel A1 A2)

theorem parallel_lines_in_non_parallel_planes
  {π1 π2 : set (ℝ × ℝ × ℝ)}
  (h1 : ¬ parallel π1 π2)
  (L : ℝ × ℝ × ℝ → Prop)
  (hL : L ∈ π1 ∩ π2) : 
  is_possible_to_draw_parallel_lines π1 π2 h1 L hL :=
sorry

end parallel_lines_in_non_parallel_planes_l624_624347


namespace total_dolls_count_l624_624170

-- Define the conditions
def big_box_dolls : Nat := 7
def small_box_dolls : Nat := 4
def num_big_boxes : Nat := 5
def num_small_boxes : Nat := 9

-- State the theorem that needs to be proved
theorem total_dolls_count : 
  big_box_dolls * num_big_boxes + small_box_dolls * num_small_boxes = 71 := 
by
  sorry

end total_dolls_count_l624_624170


namespace circumsphere_radius_l624_624908

-- Definitions and conditions
variable (S A B C : Type) [Point3D S A B C]
variable (ASB CSB : Triangle S A B)
variable (base : Triangle A B C)

-- Given conditions
variable (r β α : Real)
variable [IsEqualLateralFaces ASB CSB]
variable [PerpendicularToBasePlane ASB CSB base]
variable [InclinedFaceToBasePlane S A C base β]

-- Desired result
theorem circumsphere_radius (r β α : Real) :
  let R := r * Real.sqrt (1 + Real.cos (α / 2) ^ 2 * Real.tan β ^ 2)
  in R = r * Real.sqrt (1 + Real.cos (α / 2) ^ 2 * Real.tan β ^ 2) :=
sorry

end circumsphere_radius_l624_624908


namespace symmetric_points_y_axis_l624_624877

theorem symmetric_points_y_axis (a b : ℝ) (h1 : a - b = -3) (h2 : 2 * a + b = 2) :
  a = -1 / 3 ∧ b = 8 / 3 :=
by
  sorry

end symmetric_points_y_axis_l624_624877


namespace chromatic_decomposition_l624_624982

noncomputable 
def chromatic_number (G : Type) [graph G] : ℕ := sorry -- assume we have a definition of chromatic number

def decomposition_possible (G : Type) [graph G] (n : ℕ) (k : ℕ) (Gs : list (Type)) [∀ G', G' ∈ Gs → graph G'] : Prop :=
  (chromatic_number G = n) ∧
  (∀ i, i < k → chromatic_number (Gs.nth i) ≤ 2) ∧
  (G = ⋃ i < k, Gs.nth i) ∧
  (k ≥ ⌈log n / log 2⌉) ∧ 
  ∃ Gs', (Gs'.length = ⌈log n / log 2⌉) ∧ (∀ i, i < Gs'.length → chromatic_number (Gs'.nth i) ≤ 2) ∧ (G = ⋃ i < Gs'.length, Gs'.nth i)

theorem chromatic_decomposition {G : Type} [graph G] (n k : ℕ) (Gs : list (Type)) [∀ G', G' ∈ Gs → graph G'] :
  chromatic_number G = n →
  (∀ i, i < k → chromatic_number (Gs.nth i) ≤ 2) →
  (G = ⋃ i < k, Gs.nth i) →
  k ≥ ⌈log n / log 2⌉ ∧
  ∃ Gs', (Gs'.length = ⌈log n / log 2⌉) ∧ (∀ i, i < Gs'.length → chromatic_number (Gs'.nth i) ≤ 2) ∧ (G = ⋃ i < Gs'.length, Gs'.nth i) := 
sorry

end chromatic_decomposition_l624_624982


namespace circle_equation_correct_l624_624436

-- Define the conditions
def center := (-1, 3)
def radius := 2

-- Define the equation of the circle given center and radius
def circle_eq (x y : ℝ) := (x + 1)^2 + (y - 3)^2 = 4

-- Theorem to prove that the circle equation is as expected given the conditions
theorem circle_equation_correct : ∀ (x y : ℝ),
  let a := -1 in
  let b := 3 in
  let r := 2 in
  (x - a)^2 + (y - b)^2 = r^2 → circle_eq x y :=
by
  intros x y a b r h
  dsimp [a, b, r, circle_eq]
  rw h
  sorry

end circle_equation_correct_l624_624436


namespace tenth_term_of_sequence_l624_624184

variable (a : ℕ → ℚ) (n : ℕ)

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem tenth_term_of_sequence :
  let a₁ := (5 : ℚ)
  let r := (5 / 3 : ℚ)
  geometric_sequence a₁ r 10 = (9765625 / 19683 : ℚ) :=
by
  sorry

end tenth_term_of_sequence_l624_624184


namespace normal_dist_prob_2_l624_624266

noncomputable def xi_prob_geq_2 (σ : ℝ) (h1 : P (-2 ≤ ξ ∧ ξ ≤ 0) = 0.2) : Prop :=
  let ξ := measure_theory.probability_theory.Normal 0 σ^2 in
  P (ξ ≥ 2) = 0.3

theorem normal_dist_prob_2 (σ : ℝ) (h1 : P (-2 ≤ ξ ∧ ξ ≤ 0) = 0.2) : xi_prob_geq_2 σ h1 := 
sorry

end normal_dist_prob_2_l624_624266


namespace find_n_l624_624362

noncomputable def EulerTotient (n : ℕ) : ℕ := ∏ p in (nat.factors n).to_finset, (1 - 1 / p)
noncomputable def DedekindTotient (n : ℕ) : ℕ := ∏ p in (nat.factors n).to_finset, (1 + 1 / p)

def valid_n (n : ℕ) : Prop :=
  EulerTotient n ∣ (n + DedekindTotient n)

def valid_n_set : Set ℕ :=
  {1} ∪ {n | ∃ n1 : ℕ, n = 2 ^ n1} ∪ {n | ∃ n1 n2 : ℕ, n = 2 ^ n1 * 3 ^ n2} ∪ {n | ∃ n1 n2 : ℕ, n = 2 ^ n1 * 5 ^ n2}

theorem find_n (n : ℕ) : valid_n n ↔ n ∈ valid_n_set := sorry

end find_n_l624_624362


namespace fill_diagram_ways_l624_624895

theorem fill_diagram_ways :
  let lcm := Nat.lcm (Nat.lcm 3 9) (Nat.lcm 27 6)
  in ∃ n, (n = 20) ∧ (∀ a, (2700 ≤ a ∧ a ≤ 3749) ∧ (a % lcm = 0) → ∃! m, m = n) :=
sorry

end fill_diagram_ways_l624_624895


namespace max_heaps_660_stones_l624_624030

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624030


namespace douglas_won_in_Y_l624_624329

theorem douglas_won_in_Y (percent_total_vote : ℕ) (percent_vote_X : ℕ) (ratio_XY : ℕ) (P : ℕ) :
  percent_total_vote = 54 →
  percent_vote_X = 62 →
  ratio_XY = 2 →
  P = 38 :=
by
  sorry

end douglas_won_in_Y_l624_624329


namespace Jorge_age_in_2005_l624_624915

theorem Jorge_age_in_2005
  (age_Simon_2010 : ℕ)
  (age_difference : ℕ)
  (age_of_Simon_2010 : age_Simon_2010 = 45)
  (age_difference_Simon_Jorge : age_difference = 24)
  (age_Simon_2005 : ℕ := age_Simon_2010 - 5)
  (age_Jorge_2005 : ℕ := age_Simon_2005 - age_difference) :
  age_Jorge_2005 = 16 := by
  sorry

end Jorge_age_in_2005_l624_624915


namespace find_n_minus_m_l624_624490

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6 * x - 8 * y + 25 - r^2 = 0

-- Given conditions
def circles_intersect (r : ℝ) : Prop :=
(r > 0) ∧ (∃ x y, circle1 x y ∧ circle2 x y r)

-- Prove the range of r for intersection
theorem find_n_minus_m : 
(∀ (r : ℝ), 2 ≤ r ∧ r ≤ 12 ↔ circles_intersect r) → 
12 - 2 = 10 :=
by
  sorry

end find_n_minus_m_l624_624490


namespace tank_capacity_l624_624127

noncomputable def leak_rate (C : ℝ) := C / 6
noncomputable def inlet_rate := 240
noncomputable def net_emptying_rate (C : ℝ) := C / 8

theorem tank_capacity : ∀ (C : ℝ), 
  (inlet_rate - leak_rate C = net_emptying_rate C) → 
  C = 5760 / 7 :=
by 
  sorry

end tank_capacity_l624_624127


namespace vectors_are_coplanar_l624_624550

def are_coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  let (ax, ay, az) := a
  let (bx, by, bz) := b
  let (cx, cy, cz) := c
  let det := ax * (by * cz - bz * cy) -
             ay * (bx * cz - bz * cx) +
             az * (bx * cy - by * cx)
  det = 0

def vector_a : ℝ × ℝ × ℝ := (3, 2, 1)
def vector_b : ℝ × ℝ × ℝ := (2, 3, 4)
def vector_c : ℝ × ℝ × ℝ := (3, 1, -1)

theorem vectors_are_coplanar : are_coplanar vector_a vector_b vector_c :=
  by
  unfold are_coplanar
  rfl -- This is where actual proof would go
  sorry

end vectors_are_coplanar_l624_624550


namespace fraction_addition_l624_624675

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624675


namespace gcd_360_1260_l624_624481

theorem gcd_360_1260 : gcd 360 1260 = 180 := by
  /- 
  Prime factorization of 360 and 1260 is given:
  360 = 2^3 * 3^2 * 5
  1260 = 2^2 * 3^2 * 5 * 7
  These conditions are implicitly used to deduce the answer.
  -/
  sorry

end gcd_360_1260_l624_624481


namespace product_distances_lower_bound_l624_624934

noncomputable section
open Real

theorem product_distances_lower_bound 
  (n : ℕ)
  (d : ℝ) 
  (h_d : d > 0) 
  (P : Fin (n + 1) → ℝ × ℝ)
  (h_dist : ∀ (i j : Fin (n + 1)), i ≠ j → dist (P i) (P j) ≥ d) : 
  let distances := λ i, dist (P 0) (P i)
  (∏ i in Finset.range (n + 1), distances i) > (d / 3)^n * sqrt (factorial (n + 1)) :=
sorry

end product_distances_lower_bound_l624_624934


namespace difference_in_soda_bottles_l624_624523

-- Define the given conditions
def regular_soda_bottles : ℕ := 81
def diet_soda_bottles : ℕ := 60

-- Define the difference in the number of bottles
def difference_bottles : ℕ := regular_soda_bottles - diet_soda_bottles

-- The theorem we want to prove
theorem difference_in_soda_bottles : difference_bottles = 21 := by
  sorry

end difference_in_soda_bottles_l624_624523


namespace sum_of_fractions_l624_624700

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624700


namespace fraction_addition_l624_624644

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l624_624644


namespace loga_greater_than_one_l624_624315

theorem loga_greater_than_one (a : ℝ) (h : ∀ x ≥ 2, log a x > 1) : 1 < a ∧ a < 2 :=
sorry

end loga_greater_than_one_l624_624315


namespace slower_pipe_time_l624_624043

/-
One pipe can fill a tank four times as fast as another pipe. 
If together the two pipes can fill the tank in 40 minutes, 
how long will it take for the slower pipe alone to fill the tank?
-/

theorem slower_pipe_time (t : ℕ) (h1 : ∀ t, 1/t + 4/t = 1/40) : t = 200 :=
sorry

end slower_pipe_time_l624_624043


namespace sum_of_first_five_is_40_l624_624334

variable (a : ℕ → ℝ) -- define the arithmetic sequence
variable (a1 a5 : ℝ)
variable (S5 : ℝ)
variable (d : ℝ) -- common difference of the sequence

-- Assumptions
axiom arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a n + d
axiom a1_a5_sum : a 1 + a 5 = 16

-- Definition of sum of first 5 terms of an arithmetic sequence
def sum_first_five (a : ℕ → ℝ) := 1/2 * 5 * (a 1 + a 5)

-- The proof goal:
theorem sum_of_first_five_is_40 : sum_first_five a = 40 :=
by
  rw [sum_first_five, a1_a5_sum]
  norm_num
  sorry

end sum_of_first_five_is_40_l624_624334


namespace max_volume_correct_l624_624402

noncomputable def max_volume (R T0 P0 a b c : ℝ) (h : c^2 < a^2 + b^2) : ℝ :=
  let sqrt_term := real.sqrt (a^2 + b^2 - c^2)
  (R * T0 / P0) * ((a * sqrt_term + b * c) / (b * sqrt_term - a * c))

theorem max_volume_correct (R T0 P0 a b c : ℝ) (h : c^2 < a^2 + b^2) :
  max_volume R T0 P0 a b c h = (R * T0 / P0) * ((a * real.sqrt (a^2 + b^2 - c^2) + b * c) / (b * real.sqrt (a^2 + b^2 - c^2) - a * c)) :=
by {
  refl
}

end max_volume_correct_l624_624402


namespace algebraic_expression_evaluation_l624_624483

theorem algebraic_expression_evaluation (x y : ℝ) : 
  3 * (x^2 - 2 * x * y + y^2) - 3 * (x^2 - 2 * x * y + y^2 - 1) = 3 :=
by
  sorry

end algebraic_expression_evaluation_l624_624483


namespace square_circle_radius_l624_624925

theorem square_circle_radius
  (A B C D : Point)
  (side : ℝ)
  (h_square : sq_ABCD A B C D side)
  (O : Point)
  (r : ℝ)
  (circle_through_BC : ∀ P ∈ [B, C], dist P O = r)
  (tangent_to_DA_ext : tangent_to_extension_of_DA O r A D) :
  r = 6 :=
by
  sorry

-- Assuming sq_ABCD, point and distance are appropriately defined in the context.

end square_circle_radius_l624_624925


namespace lasagna_pieces_l624_624391

-- Given conditions
def Manny_piece : ℕ := 1
def Aaron_piece : ℕ := 0
def Kai_piece : ℕ := 2 * Manny_piece
def Raphael_piece : ℕ := (Manny_piece : ℝ) / 2
def Lisa_piece : ℝ := 2 + Raphael_piece
def Priya_piece : ℝ := (Manny_piece : ℝ) / 3

-- Theorem stating the total number of pieces
theorem lasagna_pieces : Nat.ceil (Manny_piece + Aaron_piece + Kai_piece + Lisa_piece + Priya_piece) = 6 := by
  sorry

end lasagna_pieces_l624_624391


namespace polygon_perimeter_l624_624878

theorem polygon_perimeter (a b : ℕ) (h : adjacent_sides_perpendicular) :
  perimeter = 2 * (a + b) :=
sorry

end polygon_perimeter_l624_624878


namespace angle_bisector_AQ_l624_624042

variable {A B C D D1 B1 Q : Type*}
variables [affine_space.point A]
variables [parallelogram ABCD]
variables [on_line D1 BC]
variables [on_line B1 DC]
variables (BD1_eq_DB1 : segment_length BD1 = segment_length DB1)
variables (is_intersection : ∃ Q, intersection (line BB1) (line DD1))

theorem angle_bisector_AQ (h : BD1_eq_DB1) (h_inter : is_intersection Q BB1 DD1) :
  is_angle_bisector AQ ∠BAD :=
begin
  sorry
end

end angle_bisector_AQ_l624_624042


namespace fraction_addition_l624_624671

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624671


namespace sum_of_fractions_l624_624710

variable {a : ℝ}

theorem sum_of_fractions (h : a ≠ 0) : (3 / a + 2 / a) = 5 / a := 
by sorry

end sum_of_fractions_l624_624710


namespace area_of_sector_equals_13_75_cm2_l624_624987

noncomputable def radius : ℝ := 5 -- radius in cm
noncomputable def arc_length : ℝ := 5.5 -- arc length in cm
noncomputable def circumference : ℝ := 2 * Real.pi * radius -- circumference of the circle
noncomputable def area_of_circle : ℝ := Real.pi * radius^2 -- area of the entire circle

theorem area_of_sector_equals_13_75_cm2 :
  (arc_length / circumference) * area_of_circle = 13.75 :=
by sorry

end area_of_sector_equals_13_75_cm2_l624_624987


namespace relationship_f_g_h_l624_624235

def f (a x : ℝ) : ℝ := a^x
def g (a x : ℝ) : ℝ := log a x
def h (a x : ℝ) : ℝ := x^a

variable (a : ℝ)
variable (hyp_a : 0 < a ∧ a < 1)

theorem relationship_f_g_h (ha : 0 < a ∧ a < 1) : h a 2 > f a 2 ∧ f a 2 > g a 2 :=
  sorry

end relationship_f_g_h_l624_624235


namespace add_fractions_l624_624570

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l624_624570


namespace add_fractions_l624_624736

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624736


namespace chord_length_of_intersecting_circle_and_line_l624_624250

-- Define the conditions in Lean
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ
def line_equation (ρ θ : ℝ) : Prop := 3 * ρ * Real.cos θ - 4 * ρ * Real.sin θ - 1 = 0

-- Define the problem to prove the length of the chord
theorem chord_length_of_intersecting_circle_and_line 
  (ρ θ : ℝ) (hC : circle_equation ρ θ) (hL : line_equation ρ θ) : 
  ∃ l : ℝ, l = 2 * Real.sqrt 3 :=
by 
  sorry

end chord_length_of_intersecting_circle_and_line_l624_624250


namespace add_fractions_l624_624659

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end add_fractions_l624_624659


namespace find_principal_solve_principal_l624_624215

-- Define the given conditions as constants
constant final_amount : ℝ := 1120
constant rate1 : ℝ := 0.03
constant rate2 : ℝ := 0.04
constant rate3 : ℝ := 0.05

-- Define the theorem to find the principal
theorem find_principal (P : ℝ) :
  final_amount = P * (1 + rate1) * (1 + rate2) * (1 + rate3) :=
  sorry

-- A separate theorem can be written to solve for P
theorem solve_principal : 
  let P := final_amount / ((1 + rate1) * (1 + rate2) * (1 + rate3)) in
  P ≈ 996.45 := 
  sorry

end find_principal_solve_principal_l624_624215


namespace second_train_speed_l624_624108

theorem second_train_speed (v : ℝ) :
  (∃ t : ℝ, 20 * t = v * t + 75 ∧ 20 * t + v * t = 675) → v = 16 :=
by
  sorry

end second_train_speed_l624_624108


namespace remainder_when_dividing_sum_l624_624767

theorem remainder_when_dividing_sum (k m : ℤ) (c d : ℤ) (h1 : c = 60 * k + 47) (h2 : d = 42 * m + 17) :
  (c + d) % 21 = 1 :=
by
  sorry

end remainder_when_dividing_sum_l624_624767


namespace number_of_pens_sold_l624_624156

variables (C N : ℝ) (gain_percentage : ℝ) (gain : ℝ)

-- Defining conditions given in the problem
def trader_gain_cost_pens (C N : ℝ) : ℝ := 30 * C
def gain_percentage_condition (gain_percentage : ℝ) : Prop := gain_percentage = 0.30
def gain_condition (C N : ℝ) : Prop := (0.30 * N * C) = 30 * C

-- Defining the theorem to prove
theorem number_of_pens_sold
  (h_gain_percentage : gain_percentage_condition gain_percentage)
  (h_gain : gain_condition C N) :
  N = 100 :=
sorry

end number_of_pens_sold_l624_624156


namespace fraction_addition_l624_624636

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a :=
by
  sorry

end fraction_addition_l624_624636


namespace sumProdInequality_l624_624425

open Real

theorem sumProdInequality (n : ℕ) (x : Fin n → ℝ) (s : ℝ)
  (h1 : ∀ i, 0 ≤ x i ∧ x i ≤ 1)
  (h2 : ∑ i, x i = s) :
  (∑ i, x i / (s + 1 - x i)) + ∏ i, (1 - x i) ≤ 1 :=
by
  sorry

end sumProdInequality_l624_624425


namespace fraction_addition_l624_624670

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
by
  sorry

end fraction_addition_l624_624670


namespace symm_central_origin_l624_624763

noncomputable def f₁ (x : ℝ) : ℝ := 3^x

noncomputable def f₂ (x : ℝ) : ℝ := -3^(-x)

theorem symm_central_origin :
  ∀ x : ℝ, ∃ x' y y' : ℝ, (f₁ x = y) ∧ (f₂ x' = y') ∧ (x' = -x) ∧ (y' = -y) :=
by
  sorry

end symm_central_origin_l624_624763


namespace smallest_n_inequality_l624_624786

theorem smallest_n_inequality :
  ∃ (n : ℤ), (∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^2 ≤ n * (w^4 + x^4 + y^4 + z^4)) ∧
    (n = 4) ∧ ∀ (m : ℤ), (∀ (w x y z : ℝ), (w^2 + x^2 + y^2 + z^2)^2 ≤ m * (w^4 + x^4 + y^4 + z^4)) → 4 ≤ m :=
begin
  sorry
end

end smallest_n_inequality_l624_624786


namespace find_base_l624_624430

-- Define the conditions
def height : ℕ := 5
def area : ℕ := 10

-- Define the base as a variable to be found
def base : ℕ := 4

-- The formula for the area of a triangle
def triangle_area (base height : ℕ) : ℕ := (base * height) / 2

-- Hypothesis stating the given area
axiom area_hyp : triangle_area base height = area

-- The theorem we want to prove
theorem find_base : base = 4 :=
by 
    -- Insert the proof steps here
    sorry

end find_base_l624_624430


namespace projection_of_a_onto_b_l624_624232

-- Definitions of vectors a and b
def a := (-2 : ℝ, -1 : ℝ)
def b := (1 : ℝ, 2 : ℝ)

-- Theorem to prove the projection of a onto b is as specified
theorem projection_of_a_onto_b : 
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_magnitude_sq := b.1 * b.1 + b.2 * b.2
  let c := (dot_product / b_magnitude_sq * b.1, dot_product / b_magnitude_sq * b.2)
  c = (-4 / 5 : ℝ, -8 / 5 : ℝ) := 
by
  sorry

end projection_of_a_onto_b_l624_624232


namespace trigonometric_inequality_l624_624135

theorem trigonometric_inequality
  (ν : ℕ)
  (x : Fin ν → ℝ)
  (h1 : ∀ i j : Fin ν, i < j → x i < x j)
  (h2 : ∀ i : Fin ν, 0 ≤ x i ∧ x i < π/2) :
  (∑ i in Finset.range (ν - 1), Real.sin (2 * x i))
  - (∑ i in Finset.range (ν - 1), Real.sin (x i - x (i + 1))) <
  (π / 2) + (∑ i in Finset.range (ν - 1), Real.sin (x i + x (i + 1))) := by
  sorry

end trigonometric_inequality_l624_624135


namespace smallest_x1_divides_x2006_l624_624249

-- Definitions and conditions
def seq (x : ℕ) : ℕ → ℕ
| 0       := x
| (n + 1) := seq n ^ 2 + ∑ i in finset.range (n + 1), (seq i) ^ 2

-- Theorem statement
theorem smallest_x1_divides_x2006 :
  ∃ x1 : ℕ, (∀ n, seq x1 n = x n) → (2006 ∣ seq x1 2006) ∧ x1 = 118 :=
sorry

end smallest_x1_divides_x2006_l624_624249


namespace sum_of_angles_of_roots_l624_624453

theorem sum_of_angles_of_roots (θ : ℕ → ℝ) (h1 : ∀ k, 0 ≤ θ k ∧ θ k < 360)
  (h2 : ∀ k, complex.exp (complex.I * θ k * real.pi / 180) ^ 8 = complex.exp (complex.I * 135 * real.pi / 180)) :
  (∑ k in finset.range 8, θ k) = 1575 := 
sorry

end sum_of_angles_of_roots_l624_624453


namespace quadratic_properties_l624_624244

noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x => a * x^2 + b * x + c

def min_value_passing_point (f : ℝ → ℝ) : Prop :=
  (f (-1) = -4) ∧ (f 0 = -3)

def intersects_x_axis (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  (f p1.1 = p1.2) ∧ (f p2.1 = p2.2)

def max_value_in_interval (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ max_val

theorem quadratic_properties :
  ∃ f : ℝ → ℝ,
    min_value_passing_point f ∧
    intersects_x_axis f (1, 0) (-3, 0) ∧
    max_value_in_interval f (-2) 2 5 :=
by
  sorry

end quadratic_properties_l624_624244


namespace min_cubes_fully_enclosed_structure_l624_624527

theorem min_cubes_fully_enclosed_structure (cubes : ℕ) : 
  (∀ cube, cube.snap = 1 ∧ cube.receptacles = 5 ∧ 
            cube.connections ≤ 1) ∧ 
  (∀ structure, structure.enclosed) →
  cubes ≥ 4 :=
sorry

end min_cubes_fully_enclosed_structure_l624_624527


namespace determine_n_l624_624121

theorem determine_n (n k : ℕ) (a b : ℕ) (h_k_pos : k > 0) (h_n_ge_3: n ≥ 3) (h_ab_ne_0 : a * b ≠ 0) (h_a_eq_k2b : a = k^2 * b)
  (sum_zero : ∑ i in [2, 3], (n.choose i) * a^(n-i) * b^i = 0) :
  n = 3 * k + 2 :=
sorry

end determine_n_l624_624121


namespace difference_of_smallest_integers_l624_624093

theorem difference_of_smallest_integers : 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 3) 4) 5) 6) 7) 8) 9) 10) 11) 12) 13)
  in 2 * lcm_val = 360360 :=
by 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 3) 4) 5) 6) 7) 8) 9) 10) 11) 12) 13);
  calc
  2 * lcm_val = 2 * (2^3 * 3^2 * 5 * 7 * 11 * 13) : by sorry
  ... = 360360 : by sorry

end difference_of_smallest_integers_l624_624093


namespace decrease_percent_in_revenue_l624_624131

theorem decrease_percent_in_revenue
  (T C : ℝ) -- T = original tax, C = original consumption
  (h1 : 0 < T) -- ensuring that T is positive
  (h2 : 0 < C) -- ensuring that C is positive
  (new_tax : ℝ := 0.75 * T) -- new tax is 75% of original tax
  (new_consumption : ℝ := 1.10 * C) -- new consumption is 110% of original consumption
  (original_revenue : ℝ := T * C) -- original revenue
  (new_revenue : ℝ := (0.75 * T) * (1.10 * C)) -- new revenue
  (decrease_percent : ℝ := ((T * C - (0.75 * T) * (1.10 * C)) / (T * C)) * 100) -- decrease percent
  : decrease_percent = 17.5 :=
by
  sorry

end decrease_percent_in_revenue_l624_624131


namespace collinear_SN_T_and_perpendicular_OM_MN_l624_624515

noncomputable def circle (α : Type) [field α] := { center : α × α, radius : α }

variable {α : Type} [field α] (O S T M N : α × α) -- declare the points O, S, T, M, N

variable (C1 C2 : circle α) -- two internally tangent circles such that 
/- C1 is tangent to the larger circle at S and C2 is tangent at T and intersects at M and N.
   N is closer to ST than M -/
variable (R : α) -- radius of the larger circle
variable (R1 R2 : α) -- radii of the two smaller circles

hypothesis tangent_C1_S : (C1.center.1 - S.1)^2 + (C1.center.2 - S.2)^2 = R1 ^ 2
hypothesis tangent_C2_T : (C2.center.1 - T.1)^2 + (C2.center.2 - T.2)^2 = R2 ^ 2
hypothesis internally_tangent_C1_C2 : (C1.center.1 - C2.center.1)^2 + (C1.center.2 - C2.center.2)^2 = (R1 + R2) ^ 2
hypothesis intersects_MN : ((C1.center.1 - M.1) * (C2.center.1 - M.1) + (C1.center.2 - M.2) * (C2.center.2 - M.2) = 0) ∧ 
                           ((C1.center.1 - N.1) * (C2.center.1 - N.1) + (C1.center.2 - N.2) * (C2.center.2 - N.2) = 0)

theorem collinear_SN_T_and_perpendicular_OM_MN : 
  (S.1 - N.1) * (T.2 - N.2) = (S.2 - N.2) * (T.1 - N.1) ↔      -- S, N, T are collinear
  (M.1 - O.1) * (N.1 - M.1) + (M.2 - O.2) * (N.2 - M.2) = 0 := -- OM perpendicular to MN
sorry

end collinear_SN_T_and_perpendicular_OM_MN_l624_624515


namespace cos_alpha_plus_pi_over_6_l624_624309

theorem cos_alpha_plus_pi_over_6 {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 2)
  (h_equation : (cos (2 * α) / (1 + tan (α) ^ 2)) = 3 / 8) :
  cos (α + π / 6) = 1 / 2 :=
sorry

end cos_alpha_plus_pi_over_6_l624_624309


namespace sufficient_not_necessary_l624_624504

theorem sufficient_not_necessary (x : ℝ) : (x > 4) → (x ≥ 4) ∧ (¬ (x ≥ 4) → x > 4) :=
begin
  sorry
end

end sufficient_not_necessary_l624_624504


namespace max_heaps_660_stones_l624_624027

theorem max_heaps_660_stones :
  ∀ (heaps : List ℕ), (sum heaps = 660) → (∀ i j, i ≠ j → heaps[i] < 2 * heaps[j]) → heaps.length ≤ 30 :=
sorry

end max_heaps_660_stones_l624_624027


namespace max_distance_product_l624_624376

noncomputable def max_value_PA_PB : ℝ :=
  let m : ℝ := sorry
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 3)
  let P : ℝ × ℝ := sorry
  let PA := dist A P
  let PB := dist B P
  PA * PB

theorem max_distance_product
  (m : ℝ)
  (A : ℝ × ℝ := (0, 0))
  (B : ℝ × ℝ := (1, 3))
  (line1 : x ℝ → y ℝ → x + m * y = 0)
  (line2 : x ℝ → y ℝ → m * x - y - m + 3 = 0)
  (P : ℝ × ℝ := ((y + 3 / m) / (1 + (1 / m) ^ 2), (-1 / m) * ((y + 3 / m) / (1 + (1 / m) ^ 2)))):
  dist A P * dist B P = 5 :=
sorry

end max_distance_product_l624_624376


namespace intersection_on_line_AB_l624_624836

variables {α : Type*} [linear_ordered_field α]

structure Point (α : Type*) :=
(x : α)
(y : α)

def segment (A B : Point α) := {P : Point α // ∃ t : α, 0 ≤ t ∧ t ≤ 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)}

variables (A B D E F G H I : Point α)

-- Given points D and E on segment AB
axiom D_on_AB : D ∈ segment A B
axiom E_on_AB : E ∈ segment A B

-- Equilateral triangles are constructed on segments AD, DB, AE, and EB in the same half-plane with third vertices F, G, H, and I respectively
-- Prove that if the lines FI and GH are not parallel, then their intersection point lies on line AB
theorem intersection_on_line_AB (h1 : ¬parallel F I G H) : ∃ P : Point α, P ∈ segment A B ∧ P ∈ line_through F I ∧ P ∈ line_through G H := 
sorry

end intersection_on_line_AB_l624_624836


namespace probability_diff_multiple_of_10_l624_624052

-- Define the range of integers
def range_set : Finset ℕ := Finset.range 2011

-- Define the predicate for checking differences being multiples of 10
def has_diff_multiple_of_10 (s : Finset ℕ) : Prop :=
  ∃ x y, x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (∃ k, x - y = k * 10)

-- Define the main theorem
theorem probability_diff_multiple_of_10 : 
  ∃ P : ℝ, 7 ≤ range_set.card → (P ≈ 0.99) :=
by
  sorry

end probability_diff_multiple_of_10_l624_624052


namespace count_factors_90_multiple_of_6_is_4_l624_624299

open Nat

/-- Define the prime factorization of 90. -/
def prime_factorization_90 (n : ℕ) : bool :=
  n = 2 * 3^2 * 5

/-- Check if a number is a factor of 90. -/
def is_factor_of_90 (d : ℕ) : Prop :=
  90 % d = 0

/-- Check if a factor is also a multiple of 6. -/
def is_multiple_of_6 (d : ℕ) : Prop :=
  d % 6 = 0

/-- Define the count of positive factors of 90 that are multiples of 6. -/
def count_factors_of_90_multiple_of_6 : ℕ :=
  (Finset.range (90 + 1)).filter (λ d, is_factor_of_90 d ∧ is_multiple_of_6 d).card

/-- The desired result is 4. -/
theorem count_factors_90_multiple_of_6_is_4 : 
  count_factors_of_90_multiple_of_6 = 4 :=
sorry

end count_factors_90_multiple_of_6_is_4_l624_624299


namespace red_sequence_2018th_num_l624_624325

/-- Define the sequence of red-colored numbers based on the given conditions. -/
def red_sequenced_num (n : Nat) : Nat :=
  let k := Nat.sqrt (2 * n - 1) -- estimate block number
  let block_start := if k % 2 == 0 then (k - 1)*(k - 1) else k * (k - 1) + 1
  let position_in_block := n - (k * (k - 1) / 2) - 1
  if k % 2 == 0 then block_start + 2 * position_in_block else block_start + 2 * position_in_block

/-- Statement to assert the 2018th number is 3972 -/
theorem red_sequence_2018th_num : red_sequenced_num 2018 = 3972 := by
  sorry

end red_sequence_2018th_num_l624_624325


namespace correct_conclusions_l624_624450

open Real

noncomputable def parabola (a b c : ℝ) : ℝ → ℝ :=
  λ x => a*x^2 + b*x + c

theorem correct_conclusions (a b c m n : ℝ)
  (h1 : c < 0)
  (h2 : parabola a b c 1 = 1)
  (h3 : parabola a b c m = 0)
  (h4 : parabola a b c n = 0)
  (h5 : n ≥ 3) :
  (4*a*c - b^2 < 4*a) ∧
  (n = 3 → ∃ t : ℝ, parabola a b c 2 = t ∧ t > 1) ∧
  (∀ x : ℝ, parabola a b (c - 1) x = 0 → (0 < m ∧ m ≤ 1/3)) :=
sorry

end correct_conclusions_l624_624450


namespace points_concyclic_l624_624921

open EuclideanGeometry

-- Given ABCD is a non-intersecting quadrilateral inscribed in a circle Γ 
def non_intersecting_quadrilateral (A B C D : Point) (Γ : Circle) : Prop :=
  cyclic_quadrilateral A B C D ∧ ¬lines_intersect A B C D

-- The tangents to Γ at A and B intersect at P
def tangents_intersect (A B : Point) (Γ : Circle) (P : Point) : Prop :=
  let tangentA := tangent_at_point Γ A
  let tangentB := tangent_at_point Γ B
  tangentA ≠ tangentB ∧ intersects tangentA tangentB P

-- Line parallel to AC passing through P intersects AD at X
def line_parallel_intersection_AC (A C P : Point) (AD_line : Line) (X : Point) : Prop :=
  ∃ l : Line, parallel l AC ∧ passes_through l P ∧ intersects l AD_line X

-- Line parallel to BD passing through P intersects BC at Y
def line_parallel_intersection_BD (B D P : Point) (BC_line : Line) (Y : Point) : Prop :=
  ∃ l : Line, parallel l BD ∧ passes_through l P ∧ intersects l BC_line Y

-- Problem statement: Show that the points X, Y, C, D are concyclic
theorem points_concyclic (A B C D P X Y : Point) (Γ : Circle) 
  (h1 : non_intersecting_quadrilateral A B C D Γ)
  (h2 : tangents_intersect A B Γ P)
  (h3 : ∃ AD_line : Line, passes_through AD_line A ∧ passes_through AD_line D ∧ line_parallel_intersection_AC A C P AD_line X)
  (h4 : ∃ BC_line : Line, passes_through BC_line B ∧ passes_through BC_line C ∧ line_parallel_intersection_BD B D P BC_line Y) :
  concyclic X Y C D := sorry

end points_concyclic_l624_624921


namespace total_number_of_arrangements_l624_624227
noncomputable theory

-- Definitions for the problem conditions
def volunteers : ℕ := 3
def tasks : ℕ := 4
def each_person_completes_at_least_one_task : Prop := ∀ v : ℕ, v < volunteers → ∃ t : ℕ, t < tasks
def each_task_completed_by_one_person : Prop := ∃ f : ℕ → ℕ, function.injective f ∧ ∀ t : ℕ, t < tasks → (f t) < volunteers

-- Statement of the proof problem
theorem total_number_of_arrangements :
  each_person_completes_at_least_one_task ∧ each_task_completed_by_one_person → 
  (∃ n : ℕ, n = 36) :=
by
  sorry

end total_number_of_arrangements_l624_624227


namespace sum_of_squares_change_l624_624816

def x : ℕ → ℝ := sorry
def y (i : ℕ) : ℝ := x i + 2
def z (i : ℕ) : ℝ := x i + 4

theorem sum_of_squares_change :
  (∑ j in Finset.range 100, (z j)^2) - (∑ j in Finset.range 100, (x j)^2) = 800 :=
by
  sorry

end sum_of_squares_change_l624_624816


namespace value_of_expression_l624_624929

theorem value_of_expression (a b : ℝ) (h1 : a ≠ b)
  (h2 : a^2 + 2 * a - 2022 = 0)
  (h3 : b^2 + 2 * b - 2022 = 0) :
  a^2 + 4 * a + 2 * b = 2018 :=
by
  sorry

end value_of_expression_l624_624929


namespace dartboard_distribution_l624_624164

theorem dartboard_distribution :
  ∀ (darts : ℕ) (boards : ℕ), darts = 6 → boards = 5 →
  (∃ (lists : ℕ), lists = 2) :=
by
  intro darts boards h1 h2
  use 2
  sorry

end dartboard_distribution_l624_624164


namespace total_people_in_groups_l624_624337

theorem total_people_in_groups
    (art_group : ℕ)
    (dance_group : ℕ)
    (h1 : art_group = 25)
    (h2 : dance_group = nat.floor (1.4 * ↑art_group)) :
    art_group + dance_group = 55 :=
by
  sorry

end total_people_in_groups_l624_624337


namespace fraction_addition_l624_624605

variable (a : ℝ) -- Introduce a real number 'a'

theorem fraction_addition :
  (3 / a) + (2 / a) = 5 / a :=
sorry -- Skipping the proof steps as instructed

end fraction_addition_l624_624605


namespace add_fractions_l624_624744

theorem add_fractions (a : ℝ) (h : a ≠ 0): (3 / a) + (2 / a) = 5 / a := by
  sorry

end add_fractions_l624_624744


namespace max_heaps_l624_624020

theorem max_heaps (stone_count : ℕ) (h1 : stone_count = 660) (heaps : list ℕ) 
  (h2 : ∀ a b ∈ heaps, a <= b → b < 2 * a): heaps.length <= 30 :=
sorry

end max_heaps_l624_624020


namespace rice_price_l624_624862

theorem rice_price (P : ℝ) : 
  (49 * 6.6) + (56 * P) = (105 * 8.2) → P = 9.6 := 
by 
  intros h,
  sorry

end rice_price_l624_624862
