import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearRegression
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Group.Basic
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Analysis.SpecialFunctions.Inverse
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Graph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Constructions

namespace no_multiple_of_another_from_245_254_425_452_524_l561_561428

-- Define the set of numbers
def nums : List ℕ := [245, 254, 425, 452, 524]

-- Define a predicate that checks if a number is formed by digits 2, 4, and 5 exactly once
def formed_by_digits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.erase 2 (digits.erase 4 (digits.erase 5 [])) = []

-- Define a predicate that checks if any number in the set is a multiple of another
def no_multiples_in_set (l : List ℕ) : Prop :=
  ∀ x y ∈ l, x ≠ y → ¬(x % y = 0)

-- The main theorem to prove
theorem no_multiple_of_another_from_245_254_425_452_524 :
  (∀ n ∈ nums, formed_by_digits n) → no_multiples_in_set nums :=
by
  sorry

end no_multiple_of_another_from_245_254_425_452_524_l561_561428


namespace units_digit_of_8_pow_47_l561_561442

theorem units_digit_of_8_pow_47 : (8 ^ 47) % 10 = 2 := by
  sorry

end units_digit_of_8_pow_47_l561_561442


namespace angle_B_eq_18_l561_561682

theorem angle_B_eq_18 
  (A B : ℝ) 
  (h1 : A = 4 * B) 
  (h2 : 90 - B = 4 * (90 - A)) : 
  B = 18 :=
by
  sorry

end angle_B_eq_18_l561_561682


namespace grill_ran_for_16_hours_l561_561470

def coals_burn_time_A (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 15 * 20)) 0

def coals_burn_time_B (bags : List ℕ) : ℕ :=
  bags.foldl (λ acc n => acc + (n / 10 * 30)) 0

def total_grill_time (bags_A bags_B : List ℕ) : ℕ :=
  coals_burn_time_A bags_A + coals_burn_time_B bags_B

def bags_A : List ℕ := [60, 75, 45]
def bags_B : List ℕ := [50, 70, 40, 80]

theorem grill_ran_for_16_hours :
  total_grill_time bags_A bags_B = 960 / 60 :=
by
  unfold total_grill_time coals_burn_time_A coals_burn_time_B
  unfold bags_A bags_B
  norm_num
  sorry

end grill_ran_for_16_hours_l561_561470


namespace max_area_rectangle_l561_561854

theorem max_area_rectangle (x : ℝ) (h : 2 * x + 2 * (20 - x) = 40) : 
  ∃ (A : ℝ), A = 100 ∧ (∀ y, (A_area y h) ≤ 100) :=
by
  sorry

end max_area_rectangle_l561_561854


namespace find_cost_price_l561_561058

-- Define the variables and necessary calculations
variables (x : ℝ)
def increased_price := x * 1.4
def discounted_price := increased_price * 0.9
def total_price_with_taxi := discounted_price - 50
def profit := 340

-- State the theorem including the conditions and the question
theorem find_cost_price (h : total_price_with_taxi x = x + profit) : x = 1500 :=
  by
  sorry

end find_cost_price_l561_561058


namespace sixth_grader_won_tournament_l561_561291

theorem sixth_grader_won_tournament (n : ℕ) : 
  ∃ (won: ℕ), 
      let seventh_graders := 3 * n 
      let total_players := n + seventh_graders 
      let total_matches := total_players * (total_players - 1) / 2 
      let W6 := total_matches / 2 -- wins by sixth graders
      let W7 := total_matches / 2 -- wins by seventh graders
      W6 = W7 ∧ W6 > 0 →
      ∃ i, i ≤ n ∧ ∃ j, j <= total_players ∧ i ≠ j ∧ i won = j won := 
  sorry

end sixth_grader_won_tournament_l561_561291


namespace missing_digit_in_decimal_representation_of_power_of_two_l561_561848

theorem missing_digit_in_decimal_representation_of_power_of_two :
  (∃ m : ℕ, m < 10 ∧
   ∀ (n : ℕ), (0 ≤ n ∧ n < 10 → n ≠ m) →
     (45 - m) % 9 = (2^29) % 9) :=
sorry

end missing_digit_in_decimal_representation_of_power_of_two_l561_561848


namespace max_min_ab_l561_561670

def f (x : ℝ) : ℝ := x - Real.sqrt x

theorem max_min_ab 
  (a b : ℝ)
  (h : f (a+1) + f (b+2) = 3) :
  (a + b ≤ 1 + Real.sqrt 7) ∧ (a + b ≥ (1 + Real.sqrt 13) / 2) := by
  sorry

end max_min_ab_l561_561670


namespace figure_area_l561_561476

-- Define the conditions of the problem
def fourteenSidedFigureOnGraphPaper (figure : Set (ℝ × ℝ)) : Prop :=
  -- Assume it is a fourteen-sided polygon, the exact definition can be further expanded if needed
  fourteen_sided_figure figure ∧
  -- Assume it is composed of unit squares and triangles on 1cm x 1cm graph paper
  composed_of_unit_squares_and_triangles figure

-- Define what we mean by a "fourteen-sided figure"
def fourteen_sided_figure (figure : Set (ℝ × ℝ)) : Prop := sorry -- actual geometric definition to be filled

-- Define what we mean by "composed of unit squares and triangles"
def composed_of_unit_squares_and_triangles (figure : Set (ℝ × ℝ)) : Prop := sorry -- actual definition to be filled

-- Our main theorem statement
theorem figure_area {figure : Set (ℝ × ℝ)} (h : fourteenSidedFigureOnGraphPaper figure) : measure_theory.measure_space.measure figure = 14 :=
sorry

end figure_area_l561_561476


namespace number_of_integers_between_sqrt8_and_sqrt72_l561_561211

theorem number_of_integers_between_sqrt8_and_sqrt72 : 
  let a := Int.ceil (Real.sqrt 8)
  let b := Int.floor (Real.sqrt 72)
  b - a + 1 = 6 :=
begin
  sorry
end

end number_of_integers_between_sqrt8_and_sqrt72_l561_561211


namespace votes_for_sue_l561_561415

-- Conditions from the problem
def total_votes := 1000
def category1_percent := 20 / 100   -- 20%
def category2_percent := 45 / 100   -- 45%
def sue_percent := 1 - (category1_percent + category2_percent)  -- Remaining percentage

-- Mathematically equivalent proof problem
theorem votes_for_sue : sue_percent * total_votes = 350 :=
by
  -- reminder: we do not need to provide the proof here
  sorry

end votes_for_sue_l561_561415


namespace positive_value_t_l561_561137

-- Defining variables and hypothesis
def complex_magnitude (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

theorem positive_value_t (t : ℝ) (h : complex_magnitude (-3) t = 5 * real.sqrt 5) : 
    t = 2 * real.sqrt 29 :=
by {
    -- Proof goes here
    sorry
}

end positive_value_t_l561_561137


namespace count_integers_between_sqrts_l561_561238

theorem count_integers_between_sqrts : 
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  (upper_bound - lower_bound + 1) = 6 :=
by
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  calc (upper_bound - lower_bound + 1) = 6 : sorry

end count_integers_between_sqrts_l561_561238


namespace positive_difference_is_two_l561_561873

-- Define the dimensions of the rectangle
def rectangle_length : ℕ := 3
def rectangle_width : ℕ := 2

-- Define the dimensions of the cross-shaped figure
def central_square_side : ℕ := 3
def extension_side : ℕ := 1
def number_of_extensions : ℕ := 4

-- Calculate the perimeter of the rectangle
def perimeter_rectangle : ℕ := 2 * (rectangle_length + rectangle_width)

-- Calculate the perimeter of the central square of the cross-shaped figure
def perimeter_central_square : ℕ := 4 * central_square_side

-- Calculate the contributions to the perimeter from the extensions
def perimeter_extensions : ℕ := number_of_extensions * 2
def perimeter_overlap : ℕ := number_of_extensions * 2

-- Calculate the perimeter of the cross-shaped figure
def perimeter_cross_shaped : ℕ := perimeter_central_square + perimeter_extensions - perimeter_overlap

-- The positive difference in perimeters
def positive_difference : ℕ := abs (perimeter_rectangle - perimeter_cross_shaped)

-- The theorem to prove
theorem positive_difference_is_two : positive_difference = 2 := by
    -- Proof to be filled
    sorry

end positive_difference_is_two_l561_561873


namespace shooter_probabilities_l561_561057

theorem shooter_probabilities (p : ℝ) (n : ℕ) (hp : p = 0.9) (hn : n = 4) : 
  (∃ third_shot : ℝ, third_shot = p) ∧ 
  (∃ not_exactly_three : ℝ, not_exactly_three ≠ (finset.range (4 + 1)).choose 3 * p^3 * (1 - p)) ∧ 
  (∃ at_least_once : ℝ, at_least_once = 1 - (1 - p)^n) := 
sorry

end shooter_probabilities_l561_561057


namespace relationship_between_y1_y2_y3_l561_561717

noncomputable def y_function (x : ℝ) : ℝ :=
  (x + 1) ^ 2 - 3

def y1 : ℝ := y_function (-2)
def y2 : ℝ := y_function (-1)
def y3 : ℝ := y_function 2

theorem relationship_between_y1_y2_y3 :
  y3 > y1 ∧ y1 > y2 :=
by
  have hy1 : y1 = y_function (-2) := rfl
  have hy2 : y2 = y_function (-1) := rfl
  have hy3 : y3 = y_function 2 := rfl
  rw [hy1, hy2, hy3]
  -- The rest of the proof can be filled in by calculating and comparing the values.
  sorry

end relationship_between_y1_y2_y3_l561_561717


namespace max_a_condition_slope_condition_exponential_inequality_l561_561337

noncomputable def f (x a : ℝ) := Real.exp x - a * (x + 1)
noncomputable def g (x a : ℝ) := f x a + a / Real.exp x

theorem max_a_condition (a : ℝ) (h_pos : a > 0) 
  (h_nonneg : ∀ x : ℝ, f x a ≥ 0) : a ≤ 1 := sorry

theorem slope_condition (a m : ℝ) 
  (ha : a ≤ -1) 
  (h_slope : ∀ x1 x2 : ℝ, x1 ≠ x2 → 
    (g x2 a - g x1 a) / (x2 - x1) > m) : m ≤ 3 := sorry

theorem exponential_inequality (n : ℕ) (hn : n > 0) : 
  (2 * (Real.exp n - 1)) / (Real.exp 1 - 1) ≥ n * (n + 1) := sorry

end max_a_condition_slope_condition_exponential_inequality_l561_561337


namespace volume_of_region_l561_561613

theorem volume_of_region :
  ∃ (V : ℝ), V = 9 ∧
  ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 :=
sorry

end volume_of_region_l561_561613


namespace equation_of_circle_equation_of_line_l_l561_561658

-- Define the circle passing through points and the problem conditions
def point := (ℝ × ℝ)
def circle (C : set point) := ∃ (a b : ℝ) (r : ℝ), ∀ (p : point), p ∈ C ↔ (p.1 - a)^2 + (p.2 - b)^2 = r

def M : point := (3, -3)
def N : point := (-2, 2)
def length_y_intercept : ℝ := 4 * Real.sqrt 3

-- Assertion A: Equation of the circle C
theorem equation_of_circle :
  ∃ (C : set point), circle C ∧ M ∈ C ∧ N ∈ C ∧ ∃ (a : ℝ), (a - 3)^2 + (a + 2)^2 = 12 + a^2 ∧ a = 1 ∧
  (∀ (p : point), p ∈ C ↔ (p.1 - 1)^2 + p.2^2 = 13) := sorry

-- Assertion B: Equation of the line l
theorem equation_of_line_l :
  ∃ (l : set point), 
  (∀ (p₁ p₂ : point), p₁ ∈ l → p₂ ∈ l → p₁ ≠ p₂ → 
  ∃ m : ℝ, p₁.2 = -p₁.1 + m ∧ p₂.2 = -p₂.1 + m) ∧
  ((∀ (p : point), p ∈ l → ∀ (x₁ x₂ : ℝ), 
  (x₁ + x₂ = 1 + m) ∧ (x₁ * x₂ = (m^2 - 12) / 2)) ∧ 
  (m^2 - m * (1 + m) + m^2 - 12 = 0) ∧ (m = 4 ∨ m = -3) ∧
  ((∀ (p : point), p ∈ l ↔ p.2 = -p.1 + 4) ∨ (∀ (p : point), p ∈ l ↔ p.2 = -p.1 - 3))) := sorry

end equation_of_circle_equation_of_line_l_l561_561658


namespace probability_auntie_em_can_park_l561_561051

/-- A parking lot has 20 spaces in a row. -/
def total_spaces : ℕ := 20

/-- Fifteen cars arrive, each requiring one parking space, and their drivers choose spaces at random from among the available spaces. -/
def cars : ℕ := 15

/-- Auntie Em's SUV requires 3 adjacent empty spaces. -/
def required_adjacent_spaces : ℕ := 3

/-- Calculate the probability that there are 3 consecutive empty spaces among the 5 remaining spaces after 15 cars are parked in 20 spaces.
Expected answer is (12501 / 15504) -/
theorem probability_auntie_em_can_park : 
    (1 - (↑(Nat.choose 15 5) / ↑(Nat.choose 20 5))) = (12501 / 15504) := 
sorry

end probability_auntie_em_can_park_l561_561051


namespace new_person_weight_l561_561386

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (weight_replaced : ℝ) : 
  num_persons = 8 ∧ avg_increase = 3.5 ∧ weight_replaced = 65 → 
  let total_weight_increase := num_persons * avg_increase in
  let W_new := weight_replaced + total_weight_increase in
  W_new = 93 :=
begin
  intros h,
  cases h with h1 h23,
  cases h23 with h2 h3,
  rw [h1, h2, h3],
  let total_weight_increase := 8 * 3.5,
  have : total_weight_increase = 28 := by norm_num,
  let W_new := 65 + total_weight_increase,
  rw this at W_new,
  norm_num
end

end new_person_weight_l561_561386


namespace inequality_solution_set_l561_561991

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} =
  {x : ℝ | 0 < x ∧ x ≤ 1 / 8} ∪ {x : ℝ | 2 < x ∧ x ≤ 6} :=
by
  -- Proof will go here
  sorry

end inequality_solution_set_l561_561991


namespace allison_greater_prob_l561_561956

noncomputable def prob_allison_greater (p_brian : ℝ) (p_noah : ℝ) : ℝ :=
  p_brian * p_noah

theorem allison_greater_prob : prob_allison_greater (2/3) (1/2) = 1/3 :=
by {
  -- Calculate the combined probability
  sorry
}

end allison_greater_prob_l561_561956


namespace find_p_of_probability_l561_561073

-- Define the conditions and the problem statement
theorem find_p_of_probability
  (A_red_prob : ℚ := 1/3) -- probability of drawing a red ball from bag A
  (A_to_B_ratio : ℚ := 1/2) -- ratio of number of balls in bag A to bag B
  (combined_red_prob : ℚ := 2/5) -- total probability of drawing a red ball after combining balls
  : p = 13 / 30 := by
  sorry

end find_p_of_probability_l561_561073


namespace monotonic_sine_cosine_l561_561277

theorem monotonic_sine_cosine (a : ℝ) :
  (∀ x, x ∈ Ioo (2 * π / 3) (7 * π / 6) → monotone_on (λ x, sin x + a * cos x) (Ioo (2 * π / 3) (7 * π / 6))) ↔ -real.sqrt 3 / 3 ≤ a ∧ a ≤ real.sqrt 3 :=
sorry

end monotonic_sine_cosine_l561_561277


namespace extra_bananas_each_child_gets_l561_561355

theorem extra_bananas_each_child_gets
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (absent_children : ℕ)
  (present_children : ℕ)
  (total_bananas : ℕ)
  (bananas_each_present_child_gets : ℕ)
  (extra_bananas : ℕ) :
  total_children = 840 ∧
  bananas_per_child = 2 ∧
  absent_children = 420 ∧
  present_children = total_children - absent_children ∧
  total_bananas = total_children * bananas_per_child ∧
  bananas_each_present_child_gets = total_bananas / present_children ∧
  extra_bananas = bananas_each_present_child_gets - bananas_per_child →
  extra_bananas = 2 :=
by
  sorry

end extra_bananas_each_child_gets_l561_561355


namespace number_of_divisors_2310_l561_561584

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l561_561584


namespace probability_passing_through_C_l561_561498

theorem probability_passing_through_C :
  (∀ p : rat, p = 1 →
  (∀ x y : rat, x = y / 2 →
  (∀ a b : rat, a = b / 2 →
  (p(C) = 21 / 32)))) :=
begin
  sorry
end

end probability_passing_through_C_l561_561498


namespace instantaneous_velocity_at_2_l561_561842

def displacement (t : ℝ) : ℝ := 14 * t - t^2 

def velocity (t : ℝ) : ℝ :=
  sorry -- The velocity function which is the derivative of displacement

theorem instantaneous_velocity_at_2 :
  velocity 2 = 10 := 
  sorry

end instantaneous_velocity_at_2_l561_561842


namespace area_ratio_l561_561857

-- Define the conditions: perimeters relation
def condition (a b : ℝ) := 4 * a = 16 * b

-- Define the theorem to be proved
theorem area_ratio (a b : ℝ) (h : condition a b) : (a * a) = 16 * (b * b) :=
sorry

end area_ratio_l561_561857


namespace exists_two_inscribed_tetrahedra_l561_561437

def is_coplanar (P Q R S : Point) : Prop :=
  ∃ (v1 v2 v3 v4 : Vect3), 
    P = v1 ∧ Q = v2 ∧ R = v3 ∧ S = v4 ∧ (
      v1 - v2).cross(v1 - v3).dot((v1 - v4)) = 0

def inscribed (A B C D E F G H : Point) : Prop :=
  is_coplanar A F H G ∧
  is_coplanar B E G H ∧
  is_coplanar C E F H ∧
  is_coplanar D E F G ∧
  is_coplanar E B D C ∧
  is_coplanar F A C D ∧
  is_coplanar G A B D ∧
  is_coplanar H A B C

theorem exists_two_inscribed_tetrahedra :
  ∃ (A B C D E F G H : Point),
    inscribed A B C D E F G H ∧
    inscribed E F G H A B C D ∧
    ∀ (P Q : Point),
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F ∨ P = G ∨ P = H) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F ∨ Q = G ∨ Q = H) →
      P ≠ Q := sorry

end exists_two_inscribed_tetrahedra_l561_561437


namespace intersecting_lines_l561_561405

theorem intersecting_lines (c d : ℝ) 
  (h1 : 3 = (1/3 : ℝ) * 0 + c)
  (h2 : 0 = (1/3 : ℝ) * 3 + d) :
  c + d = 2 := 
by {
  sorry
}

end intersecting_lines_l561_561405


namespace number_of_divisors_of_2310_l561_561589

theorem number_of_divisors_of_2310 : 
  let n := 2310 in
  let prime_factors := (2, 1) :: (3, 1) :: (5, 1) :: (7, 1) :: (11, 1) :: [] in
  n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 →
  (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 32 :=
begin
  intro h,
  sorry
end

end number_of_divisors_of_2310_l561_561589


namespace find_x_l561_561445

def four_digit_number (a : ℕ) : ℕ := 2803 + 100 * a

noncomputable def solve_x (a : ℕ) : ℕ :=
  let N := four_digit_number a in
  let x := (7 * N - 14552) / 9 in 
  x

theorem find_x : ∃ a x : ℕ, a = 1 ∧ solve_x a = 641 := by
  use 1
  simp [solve_x, four_digit_number]
  norm_num
  sorry

end find_x_l561_561445


namespace pyramid_volume_l561_561086

noncomputable def volume_of_tetrahedron (A B C D E F: (ℝ × ℝ)) :=
  let base_area : ℝ := 1 / 2 * (B.1 * C.2 - C.1 * B.2)   -- Area of triangle ABC
  let height : ℝ := 12     -- Height obtained from the correct intersection of altitudes
  base_area * height / 3

theorem pyramid_volume :
  let A := (0:ℝ, 0:ℝ)
  let B := (34:ℝ, 0:ℝ)
  let C := (16:ℝ, 24:ℝ)
  let D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let E := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)
  let F := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  volume_of_tetrahedron A B C D E F = 408 :=
begin
  let A := (0 : ℝ, 0 : ℝ),
  let B := (34 : ℝ, 0 : ℝ),
  let C := (16 : ℝ, 24 : ℝ),
  let D := ((34 + 16) / 2, (0 + 24) / 2),
  let E := ((16 + 0) / 2, (24 + 0) / 2),
  let F := ((0 + 34) / 2, (0 + 0) / 2),
  sorry
end

end pyramid_volume_l561_561086


namespace area_of_triangle_CBE_is_one_fourth_l561_561170

namespace Geometry

def Point : Type := ℝ × ℝ

structure Square where
  A B C D : Point
  side_length : ℝ
  (AB : A.1 < B.1) (AD : A.2 < D.2)
  (side_length_eq : side_length = B.1 - A.1)
  (AB_eq_AD : B.1 - A.1 = D.2 - A.2)

structure Circle where
  center : Point
  radius : ℝ

def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def is_tangent (line : Point → Point → Point) (circ : Circle) : Prop :=
  ∃ (E : Point), distance circ.center E = circ.radius ∧ line E circ.center = circ.radius

noncomputable def find_triangle_area (E : Point) (B C : Point) : ℝ :=
  abs (E.1 * (B.2 - C.2) + B.1 * (C.2 - E.2) + C.1 * (E.2 - B.2)) / 2

theorem area_of_triangle_CBE_is_one_fourth (E B C : Point) (ABCD : Square)
    (M := midpoint ABCD.A ABCD.D)
    (Γ := Circle.mk M (ABCD.side_length / 2))
    (hE_on_AB : ∃ x:ℝ, E = (ABCD.A.1 + x * (ABCD.B.1 - ABCD.A.1), ABCD.A.2))
    (hCE_tangent : is_tangent (λ p q, distance p q) Γ) :
    find_triangle_area E B C = 1 / 4 :=
  sorry

end Geometry

end area_of_triangle_CBE_is_one_fourth_l561_561170


namespace robert_total_balls_l561_561811

-- Define the conditions
def robert_initial_balls : ℕ := 25
def tim_balls : ℕ := 40

-- Mathematically equivalent proof problem
theorem robert_total_balls : 
  robert_initial_balls + (tim_balls / 2) = 45 := by
  sorry

end robert_total_balls_l561_561811


namespace largest_among_four_l561_561683

theorem largest_among_four (a b : ℝ) (h : 0 < a ∧ a < b ∧ a + b = 1) :
  a^2 + b^2 = max (max (max a (1/2)) (2*a*b)) (a^2 + b^2) :=
by
  sorry

end largest_among_four_l561_561683


namespace honesty_l561_561065

inductive Person
| Alice
| Bob
| Charlie
| Eve
deriving DecidableEq

open Person

def statement (p : Person) : Prop :=
  match p with
  | Alice => ¬ (statement Eve ∧ statement Bob)
  | Bob => statement Charlie → False
  | Charlie => statement Alice → False
  | Eve => statement Bob → False

theorem honesty :
  (statement Alice = False) ∧
  (statement Bob = False) ∧
  (statement Charlie = True) ∧
  (statement Eve = True) :=
by
  -- Please provide the steps to the proof here
  sorry

end honesty_l561_561065


namespace part1_proof_part2_proof_l561_561849

def FamilyA := { boys := 0, girls := 0 }
def FamilyB := { boys := 1, girls := 0 }
def FamilyC := { boys := 0, girls := 1 }
def FamilyD := { boys := 1, girls := 1 }
def FamilyE := { boys := 1, girls := 2 }

def totalChildren := FamilyA.boys + FamilyA.girls + FamilyB.boys + FamilyB.girls + FamilyC.boys + FamilyC.girls + FamilyD.boys + FamilyD.girls + FamilyE.boys + FamilyE.girls

def totalGirls := FamilyA.girls + FamilyB.girls + FamilyC.girls + FamilyD.girls + FamilyE.girls

theorem part1_proof : (2/3 * 3/7) / (4/7) = 1/2 := by sorry

def combinations := [
  { (FamilyA, FamilyB, FamilyC), 1 },
  { (FamilyA, FamilyB, FamilyD), 0 },
  { (FamilyA, FamilyB, FamilyE), 1 },
  { (FamilyA, FamilyC, FamilyD), 1 },
  { (FamilyA, FamilyC, FamilyE), 2 },
  { (FamilyA, FamilyD, FamilyE), 1 },
  { (FamilyB, FamilyC, FamilyD), 1 },
  { (FamilyB, FamilyC, FamilyE), 2 },
  { (FamilyB, FamilyD, FamilyE), 1 },
  { (FamilyC, FamilyD, FamilyE), 2 }
]

def probX_eq_0 := 1 / 10
def probX_eq_1 := 6 / 10
def probX_eq_2 := 3 / 10

def expected_value := 0 * probX_eq_0 + 1 * probX_eq_1 + 2 * probX_eq_2

theorem part2_proof : expected_value = 6 / 5 := by sorry

end part1_proof_part2_proof_l561_561849


namespace smallest_possible_d_l561_561816

noncomputable def vector_set_has_three_equal_sum (v : Fin 4 → ℕ) : Prop :=
  let σ_v := {v' | ∃ (perm : Equiv.Perm (Fin 4)), v' = perm v}
  ∃ s, (finset.filter (λ x, x.sum = s) σ_v).card = 3

theorem smallest_possible_d (a b c d : ℕ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) 
  (h_set : vector_set_has_three_equal_sum ![a, b, c, d]) :
  d = 6 :=
sorry

end smallest_possible_d_l561_561816


namespace napoleons_theorem_l561_561026

open EuclideanGeometry

/-- Given a triangle ABC and external equilateral triangles A'BC, AB'C, and ABC' with centers A_1, B_1, and C_1, show that A_1B_1C_1 is equilateral. -/
theorem napoleons_theorem (A B C A' B' C': Point) 
    (hA' : equilateral_triangle A' B C)
    (hB' : equilateral_triangle AB' C)
    (hC' : equilateral_triangle ABC'):
    let A_1 := centroid A' B C
    let B_1 := centroid A B' C
    let C_1 := centroid A B C'
    equilateral_triangle A_1 B_1 C_1 :=
by
  sorry

end napoleons_theorem_l561_561026


namespace lawrence_average_work_hours_l561_561318

def total_hours_worked (monday tuesday wednesday thursday friday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday + friday

theorem lawrence_average_work_hours :
  let monday := 8
  let tuesday := 8
  let wednesday := 5.5
  let thursday := 5.5
  let friday := 8
  let total_hours := total_hours_worked monday tuesday wednesday thursday friday
  let days_worked := 5
  total_hours / days_worked = 7 :=
by
  -- Notice that we'll need to use rationals or real numbers to handle 5.5
  -- This part of Lean requires corresponding numeric modules:
  have h1 : (8 : ℚ) + 8 + 5.5 + 5.5 + 8 = 35, by norm_num
  have h2 : (35 / 5 : ℚ) = 7, by norm_num
  exact_mod_cast h2 

end lawrence_average_work_hours_l561_561318


namespace lunks_for_apples_l561_561696

noncomputable def lunks_per_kunks := 7 / 4
noncomputable def kunks_per_apples := 3 / 5
def apples_needed := 24
noncomputable def kunks_needed_for_apples := (kunks_per_apples * apples_needed).ceil
noncomputable def lunks_needed := (lunks_per_kunks * kunks_needed_for_apples).ceil

theorem lunks_for_apples :
  lunks_needed = 27 :=
sorry

end lunks_for_apples_l561_561696


namespace votes_for_sue_l561_561414

-- Conditions from the problem
def total_votes := 1000
def category1_percent := 20 / 100   -- 20%
def category2_percent := 45 / 100   -- 45%
def sue_percent := 1 - (category1_percent + category2_percent)  -- Remaining percentage

-- Mathematically equivalent proof problem
theorem votes_for_sue : sue_percent * total_votes = 350 :=
by
  -- reminder: we do not need to provide the proof here
  sorry

end votes_for_sue_l561_561414


namespace max_min_values_l561_561846

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x + 1

theorem max_min_values :
  (∀ x ∈ set.Icc (-3 : ℝ) 0, f x ≤ 3) ∧ (∃ x ∈ set.Icc (-3 : ℝ) 0, f x = 3) ∧
  (∀ x ∈ set.Icc (-3 : ℝ) 0, f x ≥ -17) ∧ (∃ x ∈ set.Icc (-3 : ℝ) 0, f x = -17) :=
  by
  sorry

end max_min_values_l561_561846


namespace probability_two_different_colors_l561_561917

noncomputable def probability_different_colors (total_balls red_balls black_balls : ℕ) : ℚ :=
  let total_ways := (Finset.range total_balls).card.choose 2
  let diff_color_ways := (Finset.range black_balls).card.choose 1 * (Finset.range red_balls).card.choose 1
  diff_color_ways / total_ways

theorem probability_two_different_colors (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ)
  (h_total : total_balls = 5) (h_red : red_balls = 2) (h_black : black_balls = 3) :
  probability_different_colors total_balls red_balls black_balls = 3 / 5 :=
by
  subst h_total
  subst h_red
  subst h_black
  -- Here the proof would follow using the above definitions and reasoning
  sorry

end probability_two_different_colors_l561_561917


namespace range_of_a_l561_561720

theorem range_of_a (a : ℝ) (h : ∀ x ∈ set.Icc 2 3, x^2 - a ≥ 0) : a ≤ 4 :=
sorry

end range_of_a_l561_561720


namespace sqrt_nested_expression_l561_561443

theorem sqrt_nested_expression : sqrt (36 * sqrt 16) = 12 := by
  -- The statement equivalently translates the problem into a Lean theorem,
  -- proving that sqrt(36 * sqrt(16)) = 12.
  sorry

end sqrt_nested_expression_l561_561443


namespace floor_sqrt_23_squared_l561_561556

theorem floor_sqrt_23_squared : (⌊Real.sqrt 23⌋) ^ 2 = 16 := by
  have h1 : (4:ℝ) < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < (5:ℝ) := sorry
  have h3 : (⌊Real.sqrt 23⌋ : ℝ) = 4 :=
    by sorry
  show 4^2 = 16 from by sorry

end floor_sqrt_23_squared_l561_561556


namespace sin_2alpha_minus_2cos2_alpha_l561_561632

theorem sin_2alpha_minus_2cos2_alpha (α : ℝ) (h : tan (α - π / 4) = 2) : sin (2 * α) - 2 * cos (α) ^ 2 = -4 / 5 :=
sorry

end sin_2alpha_minus_2cos2_alpha_l561_561632


namespace hours_needed_to_finish_book_l561_561427

theorem hours_needed_to_finish_book
  (total_pages : ℕ)
  (reading_rate : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℝ)
  (pages_read_by_monday : ℕ := monday_hours * reading_rate)
  (pages_read_by_tuesday : ℝ := tuesday_hours * reading_rate)
  (total_pages_read : ℝ := pages_read_by_monday + pages_read_by_tuesday)
  (remaining_pages : ℝ := total_pages - total_pages_read)
  (hours_needed : ℝ := remaining_pages / reading_rate) :
  total_pages = 387 → reading_rate = 12 → monday_hours = 3 → tuesday_hours = 6.5 → hours_needed = 22.75 :=
by
  intros h_total_pages h_reading_rate h_monday_hours h_tuesday_hours
  rw [h_total_pages, h_reading_rate, h_monday_hours, h_tuesday_hours]
  calc
    hours_needed
      = (387 - (3 * 12 + 6.5 * 12)) / 12 : by sorry
    ... = 273 / 12 : by sorry
    ... = 22.75 : by sorry

end hours_needed_to_finish_book_l561_561427


namespace no_positive_integer_solutions_l561_561346

theorem no_positive_integer_solutions (p n : ℕ) (hp : nat.prime p) (hp_mod : p % 4 = 3) (hn_pos : 0 < n) :
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ p ^ n = x ^ 2 + y ^ 2 :=
by
  sorry

end no_positive_integer_solutions_l561_561346


namespace true_propositions_l561_561068

theorem true_propositions :
  (¬ (∀ (A B : Type) (f : A → B) (x y : A), f x = f y → x = y)) ∧   -- (1)
  ((∃ (x : ℝ), x = Real.pi ^ Real.sqrt 2)) ∧                        -- (2)
  (¬ (∃ (x : ℝ), (x^2 + 2 * x + 3 = 0))) ∧                          -- (3)
  (¬ (∀ (x y : ℝ), x^2 ≠ y^2 ↔ x ≠ y ∨ x ≠ -y)) ∧                    -- (4)
  (¬ (∀ (a b : ℕ), (a % 2 = 0 ∧ b % 2 = 0) → ((a + b) % 2 = 0) → (a + b) % 2 ≠ 1)) ∧  -- (5)
  (¬ (∀ (p q : Prop), ¬ (p ∨ q) ↔ ¬ p ∧ ¬ q)) ∧                    -- (6)
  (¬ (∀ (a b c : ℝ), (∀ (x : ℝ), ¬ (a * x^2 + b * x + c ≤ 0)) → a > 0 ∧ b^2 - 4 * a * c < 0))  -- (7)
  := by
    sorry

end true_propositions_l561_561068


namespace lunks_for_apples_l561_561698

noncomputable def lunks_per_kunks := 7 / 4
noncomputable def kunks_per_apples := 3 / 5
def apples_needed := 24
noncomputable def kunks_needed_for_apples := (kunks_per_apples * apples_needed).ceil
noncomputable def lunks_needed := (lunks_per_kunks * kunks_needed_for_apples).ceil

theorem lunks_for_apples :
  lunks_needed = 27 :=
sorry

end lunks_for_apples_l561_561698


namespace universal_negation_example_l561_561406

theorem universal_negation_example :
  (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) →
  (¬ (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) = (∃ x : ℝ, x^2 - 3 * x + 1 > 0)) :=
by
  intro h
  sorry

end universal_negation_example_l561_561406


namespace largest_angle_in_triangle_l561_561734

noncomputable def angle_sum : ℝ := 120 -- $\frac{4}{3}$ of 90 degrees
noncomputable def angle_difference : ℝ := 20

theorem largest_angle_in_triangle :
  ∃ (a b c : ℝ), a + b + c = 180 ∧ a + b = angle_sum ∧ b = a + angle_difference ∧
  max a (max b c) = 70 :=
by
  sorry

end largest_angle_in_triangle_l561_561734


namespace cube_volume_l561_561841

theorem cube_volume (d_AF : Real) (h : d_AF = 6 * Real.sqrt 2) : ∃ (V : Real), V = 216 :=
by {
  sorry
}

end cube_volume_l561_561841


namespace integer_count_between_sqrt8_and_sqrt72_l561_561220

theorem integer_count_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ( ∀ x : ℤ, (⌊Real.sqrt 8⌋.to_nat + 1) ≤ x ∧ x ≤ ⌊Real.sqrt 72⌋ - 1 → x = 6 ) :=
by 
  -- Define the floor and ceiling functions
  have sqrt_8_ceil : ⌈Real.sqrt 8⌉ = 3 := sorry
  have sqrt_72_floor : ⌊Real.sqrt 72⌋ = 8 := sorry

  use 6,
  split,
  { refl },
  { intros x hx,
    sorry
  }

-- Additional supporting facts
lemma sqrt_8_approx : Real.sqrt 8 ≈ 2.83 := sorry
lemma sqrt_72_approx : Real.sqrt 72 ≈ 8.49 := sorry

end integer_count_between_sqrt8_and_sqrt72_l561_561220


namespace lunks_needed_for_24_apples_l561_561701

-- Define the conditions as Lean definitions
def lunks_per_kunks := 7 / 4
def kunks_per_apples := 3 / 5
def apples_needed := 24

-- State the theorem
theorem lunks_needed_for_24_apples : 
  let k := (3 * apples_needed) / 5 in 
  let rounded_k := k.ceil in 
  let l := (7 * rounded_k) / 4 in 
  l.ceil = 27 :=
by 
  let k := (3 * apples_needed) / 5
  let rounded_k := k.ceil
  let l := (7 * rounded_k) / 4
  have h1 : k = (3 * apples_needed) / 5 := rfl
  have h2 : rounded_k = k.ceil := rfl
  have h3 : l = (7 * rounded_k) / 4 := rfl
  show l.ceil = 27, from sorry

end lunks_needed_for_24_apples_l561_561701


namespace incorrect_option_D_l561_561614

-- Conditions
variable (x : Fin 5 → ℕ) (y : Fin 5 → ℕ)
variable h1 : (∑ i, x i) = (∑ i, y i)
variable h2 : (∑ i, x i) = 10
variable h3 : ∀ i, x i + y i = 4

-- Statement of the problem to prove
theorem incorrect_option_D : (∑ i, (x i)^2) ≠ (∑ i, (y i)^2) :=
sorry

end incorrect_option_D_l561_561614


namespace blue_marbles_l561_561466

theorem blue_marbles {B : ℕ} (h1 : B + 7 + (13 - B) = 20) (h2 : (7 + (13 - B)) / 20 = 0.75) : B = 5 :=
by {
  -- combine the given equations h1 and h2
  sorry
}

end blue_marbles_l561_561466


namespace find_x_in_second_quadrant_l561_561331

variable (α : ℝ) (x : ℝ)

theorem find_x_in_second_quadrant 
  (h1 : ∃ x, x < 0 ∧ P : ℝ × ℝ)
  (h2 : ∀ θ, θ = α ∧ P (x, 4) = (cos α * (1 / 5) * x)) :
   x = -3 :=
sorry

end find_x_in_second_quadrant_l561_561331


namespace function_increasing_on_interval_l561_561067

noncomputable def f : ℝ → ℝ := λ x, 2^(x - 2)

theorem function_increasing_on_interval (x : ℝ) (hx : x > 1) :
  monotone_on f (set.Ioi 1) :=
sorry

end function_increasing_on_interval_l561_561067


namespace arithmetic_sequence_inequality_l561_561334

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

-- All terms are positive
def all_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

theorem arithmetic_sequence_inequality
  (h_arith_seq : is_arithmetic_sequence a a1 d)
  (h_non_zero_diff : d ≠ 0)
  (h_positive : all_positive a) :
  (a 1) * (a 8) < (a 4) * (a 5) :=
by
  sorry

end arithmetic_sequence_inequality_l561_561334


namespace variance_of_sample_l561_561847

theorem variance_of_sample (a b : ℝ)
  (h₁ : a + b = 5)
  (h₂ : a * b = 4)
  (h₃ : (a + 3 + 5 + 7) / 4 = b) :
  let variance := ((a - b) ^ 2 + (3 - b) ^ 2 + (5 - b) ^ 2 + (7 - b) ^ 2) / 4 in
  variance = 5 :=
by sorry

end variance_of_sample_l561_561847


namespace range_of_a_line_not_passing_second_quadrant_l561_561836

theorem range_of_a_line_not_passing_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), ((a - 2) * y = (3a - 1) * x - 1) →
    ¬(x < 0 ∧ y > 0)) → 
  a ≥ 2 :=
by
  sorry

end range_of_a_line_not_passing_second_quadrant_l561_561836


namespace coloring_ways_3x3_grid_l561_561763

-- Define the grid and coloring problem
def Grid := Array (Array Nat) -- a 3x3 grid is represented by arrays of color indices (0, 1, 2 for red, green, blue)

-- A coloring function
def valid_coloring (grid : Grid) : Bool :=
  (0 ≤ List.all_pairs (List.range 3) (List.range 3)).all (λ (i, j),
    let color := grid[i][j];
    (if i > 0 then grid[i-1][j] ≠ color else True) &&
    (if i < 2 then grid[i+1][j] ≠ color else True) &&
    (if j > 0 then grid[i][j-1] ≠ color else True) &&
    (if j < 2 then grid[i][j+1] ≠ color else True))

-- Define the problem statement
theorem coloring_ways_3x3_grid : (Finset.univ.filter (λ grid, valid_coloring grid)).card = 6 := by sorry

end coloring_ways_3x3_grid_l561_561763


namespace xyz_divisor_l561_561783

-- Definitions of sets A and B and set S as per the problem conditions
def A : Set ℕ := sorry -- Placeholder for set A
def B : Set ℕ := sorry -- Placeholder for set B
def S : Set ℕ := {ab | a ∈ A ∧ b ∈ B} -- Set S consisting of products of elements from A and B

-- Proving the main statement
theorem xyz_divisor (hA : 2 ≤ A.card) (hB : 2 ≤ B.card) (hS : S.card = A.card + B.card - 1) :
  ∃ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∣ (y * z) :=
sorry

end xyz_divisor_l561_561783


namespace lines_are_skew_l561_561109

def line1 (a t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 1 + 4 * t, a + 5 * t)
  
def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 3 + 3 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) : (∀ t u : ℝ, line1 a t ≠ line2 u) ↔ a ≠ -4/5 :=
sorry

end lines_are_skew_l561_561109


namespace count_integers_between_sqrt8_and_sqrt72_l561_561205

theorem count_integers_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℕ, 3 ≤ x ∧ x ≤ 8 → x ∈ finset.range (8 - 3 + 1) :=
by
  let a := real.sqrt 8
  let b := real.sqrt 72
  have ha : 3 = nat_ceil a := sorry
  have hb : 8 = nat_floor b := sorry
  use 6
  split
  {
    exact sorry
  }
  {
    intros x hx
    have h_valid : 3 ≤ x ∧ x ≤ 8 := hx
    rw finset.mem_range
    sorry
  }

end count_integers_between_sqrt8_and_sqrt72_l561_561205


namespace count_integers_between_sqrt8_and_sqrt72_l561_561210

theorem count_integers_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℕ, 3 ≤ x ∧ x ≤ 8 → x ∈ finset.range (8 - 3 + 1) :=
by
  let a := real.sqrt 8
  let b := real.sqrt 72
  have ha : 3 = nat_ceil a := sorry
  have hb : 8 = nat_floor b := sorry
  use 6
  split
  {
    exact sorry
  }
  {
    intros x hx
    have h_valid : 3 ≤ x ∧ x ≤ 8 := hx
    rw finset.mem_range
    sorry
  }

end count_integers_between_sqrt8_and_sqrt72_l561_561210


namespace number_of_positive_divisors_2310_l561_561597

theorem number_of_positive_divisors_2310 : 
  let n := 2310 in
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)] in
  let t := (factorization.map (λ p : ℕ × ℕ, p.snd + 1)).prod in
  t = 32 :=
by
  let n := 2310
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let t := (factorization.map (λ p, p.snd + 1)).prod
  sorry

end number_of_positive_divisors_2310_l561_561597


namespace minimum_red_balls_l561_561457

-- Definitions:
def is_red : ball → Prop := sorry
def is_blue (b : ball) : Prop := ¬ (is_red b)
def majority_red_triplet (triple : list ball) : Prop := (triple.filter is_red).length ≥ 2
def majority_blue_triplet (triple : list ball) : Prop := (triple.filter is_blue).length ≥ 2

-- Problem conditions:
variables (circle : list ball)
(h1 : circle.length = 58)         -- There are 58 balls.
(h2 : ∀ triple : list ball, triple.length = 3 → 
    majority_red_triplet triple ↔ majority_blue_triplet triple) -- Number of majority triple conditions.

-- Theorem statement:
theorem minimum_red_balls (circle : list ball) (h1 : circle.length = 58)
    (h2 : ∀ triple : list ball, triple.length = 3 → 
    majority_red_triplet triple ↔ majority_blue_triplet triple) : 
    (circle.filter is_red).length ≥ 20 := 
sorry

end minimum_red_balls_l561_561457


namespace angle_at_apex_correct_l561_561797

open Real

-- Define the key points and radii of the spheres
structure Sphere :=
(center : Point)
(radius : ℝ)

-- Define the specific spheres
def sphere1 : Sphere := ⟨(0, 0), 2⟩
def sphere2 : Sphere := ⟨(sqrt 40, 0), 2⟩
def sphere3 : Sphere := ⟨(sqrt 40 / 2, 6), 5⟩

-- Define the vertex of the cone
def vertex : Point := (sqrt 40 / 2, 3)

-- Define the angle calculation
noncomputable def angle_at_apex (s1 s2 s3 : Sphere) (v : Point) : ℝ :=
2 * arccot (72 : ℝ)

-- Checking the computed value
theorem angle_at_apex_correct :
  angle_at_apex sphere1 sphere2 sphere3 vertex = 2 * arccot (72 : ℝ) :=
sorry

end angle_at_apex_correct_l561_561797


namespace countable_set_with_uncountable_almost_disjoint_family_l561_561308

theorem countable_set_with_uncountable_almost_disjoint_family :
  ∃ (Y : Set ℕ) (ℱ : Set (Set ℕ)),
    (Set.Countable Y) ∧ ¬(Set.Countable ℱ) ∧ 
    (∀ (A B ∈ ℱ), A ≠ B → (Set.Finite (A ∩ B))) :=
begin
  sorry
end

end countable_set_with_uncountable_almost_disjoint_family_l561_561308


namespace find_min_value_l561_561122

noncomputable def problem (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧
  (27^x + y^4 - 3^x - 1 = 0)

theorem find_min_value :
  ∃ x y : ℝ, problem x y ∧ 
  (∀ (x' y' : ℝ), problem x' y' → (x^3 + y^3) ≤ (x'^3 + y'^3)) ∧ (x^3 + y^3 = -1) := 
sorry

end find_min_value_l561_561122


namespace sequence_is_constant_l561_561161

theorem sequence_is_constant
  (a : ℕ+ → ℝ)
  (S : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, S n + S (n + 1) = a (n + 1))
  : ∀ n : ℕ+, a n = 0 :=
by
  sorry

end sequence_is_constant_l561_561161


namespace deviation_angle_of_light_ray_l561_561484

theorem deviation_angle_of_light_ray (α : ℝ) (n : ℝ) (θ_d : ℝ) 
    (hα : α = 30)               -- condition: angle of incidence 30°
    (hn : n = 1.5)              -- condition: refractive index of glass 1.5
    (hθd : θ_d = 180 - 2 * α)   -- formula for deviation angle
    : θ_d = 120 :=              -- the correct answer

by {
    rw [hα, hθd],
    norm_num
}

end deviation_angle_of_light_ray_l561_561484


namespace solve_a_b_l561_561108

theorem solve_a_b (a b : ℝ) (k : ℤ) :
  (∀ x : ℝ, 2 * cos (x + b / 2) ^ 2 - 2 * sin (a * x - π / 2) * cos (a * x - π / 2) = 1) ↔
  (a = 1 ∧ ∃ k : ℤ, b = -3 * π / 2 + 2 * k * π) ∨ (a = -1 ∧ ∃ k : ℤ, b = 3 * π / 2 + 2 * k * π) :=
by sorry

end solve_a_b_l561_561108


namespace TJs_average_time_l561_561821

theorem TJs_average_time 
  (total_distance : ℝ) 
  (distance_half : ℝ)
  (time_first_half : ℝ) 
  (time_second_half : ℝ) 
  (H1 : total_distance = 10) 
  (H2 : distance_half = total_distance / 2) 
  (H3 : time_first_half = 20) 
  (H4 : time_second_half = 30) :
  (time_first_half + time_second_half) / total_distance = 5 :=
by
  sorry

end TJs_average_time_l561_561821


namespace sum_abc_l561_561429

theorem sum_abc (A B C : ℕ) (hposA : 0 < A) (hposB : 0 < B) (hposC : 0 < C) (hgcd : Nat.gcd A (Nat.gcd B C) = 1)
  (hlog : A * Real.log 5 / Real.log 100 + B * Real.log 2 / Real.log 100 = C) : A + B + C = 5 :=
sorry

end sum_abc_l561_561429


namespace find_x_squared_plus_inv_squared_l561_561456

theorem find_x_squared_plus_inv_squared (x : ℝ) (hx : x + (1 / x) = 4) : x^2 + (1 / x^2) = 14 := 
by
sorry

end find_x_squared_plus_inv_squared_l561_561456


namespace correct_calculation_l561_561893

theorem correct_calculation :
  (∀ (x y : ℝ), (x^2 * x^3 = x^5)) ∧
  (¬ ∀ (x : ℝ), 4 * x^2 + 2 * x^2 = 6 * x^4) ∧
  (¬ ∀ (x y : ℝ), (x - y)^2 = x^2 - y^2) ∧
  (¬ ∀ (x : ℝ), (x^3)^2 = x^5) :=
by
  split
  repeat { sorry }

end correct_calculation_l561_561893


namespace train_pass_time_l561_561483

def speed_jogger := 9   -- in km/hr
def distance_ahead := 240   -- in meters
def length_train := 150   -- in meters
def speed_train := 45   -- in km/hr

noncomputable def time_to_pass_jogger : ℝ :=
  let speed_jogger_mps := speed_jogger * (1000 / 3600)
  let speed_train_mps := speed_train * (1000 / 3600)
  let relative_speed := speed_train_mps - speed_jogger_mps
  let total_distance := distance_ahead + length_train
  total_distance / relative_speed

theorem train_pass_time : time_to_pass_jogger = 39 :=
  by
    sorry

end train_pass_time_l561_561483


namespace correct_statements_l561_561401

-- Definitions for the problem
def towns_are_80km_apart : Prop := true
def cyclist_left_3_hours_earlier : Prop := true
def cyclist_arrived_1_hour_earlier : Prop := true
def cyclist_accelerated_then_constant_speed : Prop := true
def motorcyclist_constant_speed : Prop := true
def motorcyclist_caught_up_after_1_5_hours : Prop := true

-- Theorem statement
theorem correct_statements :
  towns_are_80km_apart →
  cyclist_left_3_hours_earlier →
  cyclist_arrived_1_hour_earlier →
  cyclist_accelerated_then_constant_speed →
  motorcyclist_constant_speed →
  motorcyclist_caught_up_after_1_5_hours →
  ( 
    "①: Cyclist left 3 hours earlier and arrived 1 hour earlier, 
    ②: Cyclist accelerated then moved at constant speed, 
    ③: Motorcyclist caught up after 1.5 hours"
  ) = "B: ①②③" :=
by
  intros
  have h1 : cyclist_left_3_hours_earlier ∧ cyclist_arrived_1_hour_earlier := ⟨‹cyclist_left_3_hours_earlier›, ‹cyclist_arrived_1_hour_earlier›⟩,
  have h2 : cyclist_accelerated_then_constant_speed ∧ motorcyclist_constant_speed := ⟨‹cyclist_accelerated_then_constant_speed›, ‹motorcyclist_constant_speed›⟩,
  have h3 : motorcyclist_caught_up_after_1_5_hours := ‹motorcyclist_caught_up_after_1_5_hours›,
  sorry -- proof

end correct_statements_l561_561401


namespace tangent_line_eq_l561_561112

theorem tangent_line_eq (x y : ℝ) (h : y = 2 * x^2 + 1) : 
  (x = -1 ∧ y = 3) → (4 * x + y + 1 = 0) :=
by
  intros
  sorry

end tangent_line_eq_l561_561112


namespace product_of_second_and_fourth_term_l561_561872

theorem product_of_second_and_fourth_term (a : ℕ → ℤ) (d : ℤ) (h₁ : a 10 = 25) (h₂ : d = 3)
  (h₃ : ∀ n, a n = a 1 + (n - 1) * d) : a 2 * a 4 = 7 :=
by
  -- Assuming necessary conditions are defined
  sorry

end product_of_second_and_fourth_term_l561_561872


namespace trigonometric_identity_l561_561066

theorem trigonometric_identity :
  (1 / 2 - (Real.cos (15 * Real.pi / 180)) ^ 2) = - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_identity_l561_561066


namespace volume_of_prism_l561_561825

theorem volume_of_prism (a b c : ℝ) 
  (h₁ : a * b = 18) 
  (h₂ : a * c = 50) 
  (h₃ : b * c = 75) : 
  a * b * c = 150 * real.sqrt 3 := 
by 
  sorry

end volume_of_prism_l561_561825


namespace sum_of_solutions_l561_561009

-- Define the equation and the conditions as Lean predicates
def equation (x : ℝ) : Prop := x = |3 * x - |80 - 3 * x||

-- Conditions for the analysis of the equation
def case1_condition (x : ℝ) : Prop := 3 * x ≤ 80
def case2_condition (x : ℝ) : Prop := 3 * x > 80

theorem sum_of_solutions : 
  let solutions := { x | equation x ∧ (case1_condition x ∨ case2_condition x) } in 
  ∑ x in solutions.to_list, x = 752 / 7 :=
sorry

end sum_of_solutions_l561_561009


namespace correlation_coefficient_sign_l561_561198

-- Definitions of conditions
variables {x y : Type} [linear_ordered_field x]
variables (r : ℝ) (a b : ℝ) -- Correlation coefficient, intercept, and slope of the regression line

-- The given linear relationship and regression equation
def linear_relationship : Prop :=
  ∀ x y, y = a + b * x

-- The proof statement we want to achieve
theorem correlation_coefficient_sign :
  linear_relationship → ((0 < r) ↔ (0 < b)) ∧ ((r < 0) ↔ (b < 0)) :=
by intros; sorry

end correlation_coefficient_sign_l561_561198


namespace inf_non_square_product_exists_l561_561657

theorem inf_non_square_product_exists (a b : ℕ) (h : (a * b) ∉ {x | ∃ y : ℕ, y^2 = x}) :
  ∃ᶠ n in at_top, ∀ n > 0, ¬ ∃ (k : ℕ), k^2 = (a^n - 1)*(b^n - 1) :=
begin
  sorry
end

end inf_non_square_product_exists_l561_561657


namespace lunks_for_apples_l561_561687

theorem lunks_for_apples : ∀ (lun_to_kun : ℕ) (num_lun : ℕ) (num_kun : ℕ) (kun_to_app : ℕ) (num_kun2 : ℕ) (num_app : ℕ),
  lun_to_kun = 7 ∧ num_kun = 4 ∧ kun_to_app = 3 ∧ num_kun2 = 5 ∧ num_app = 24 → 
  ((num_app * kun_to_app * num_lun / (num_kun2 * lun_to_kun)) ≤ 27) :=
by
  intros lun_to_kun num_lun num_kun kun_to_app num_kun2 num_app
  assume h_conditions
  sorry

end lunks_for_apples_l561_561687


namespace find_other_number_l561_561124

def smallest_multiple_of_711 (n : ℕ) : ℕ := Nat.lcm n 711

theorem find_other_number (n : ℕ) : smallest_multiple_of_711 n = 3555 → n = 5 := by
  sorry

end find_other_number_l561_561124


namespace find_m_of_cos_alpha_l561_561172

theorem find_m_of_cos_alpha
  (m : ℝ)
  (α : ℝ)
  (h1 : ∃ P, P = (-8 * m, -3))
  (h2 : cos α = -4 / 5)
  (h3 : P = (-8 * m, -3))
  : m = 1 / 2 := sorry

end find_m_of_cos_alpha_l561_561172


namespace number_of_divisors_of_square_l561_561265

theorem number_of_divisors_of_square {n : ℕ} (h : ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ n = p * q) : Nat.totient (n^2) = 9 :=
sorry

end number_of_divisors_of_square_l561_561265


namespace intersection_and_distance_l561_561752

noncomputable def C1_parametric_eq (p t : ℝ) : ℝ × ℝ :=
  (2 * p * t, 2 * p * Real.sqrt t)

noncomputable def C1_Cartesian_eq (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y ≥ 0

noncomputable def C1_polar_eq (p ρ θ : ℝ) : Prop :=
  ρ * Real.sin θ ^ 2 = 2 * p * Real.cos θ ∧ 0 < θ ∧ θ ≤ Real.pi / 2

noncomputable def C2_polar_eq (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

theorem intersection_and_distance (p : ℝ) (A B : ℝ × ℝ × ℝ) 
  (hA : A = (0, 0, 0)) (hAB_distance : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 2 * Real.sqrt 3) :
  (C1_Cartesian_eq p (B.1) (B.2)) ∧ (C2_polar_eq (Real.sqrt (B.1 ^ 2 + B.2 ^ 2)) (B.3)) → 
  p = 3 * Real.sqrt 3 / 2 :=
sorry

end intersection_and_distance_l561_561752


namespace molecular_weight_of_carbon_part_l561_561116

-- Defining the relevant parameters
def formula_C4H8O2 : String := "C4H8O2"
def molecular_weight_compound : Float := 88.0
def atomic_weight_carbon : Float := 12.01
def number_of_carbons : Int := 4

-- Proving the molecular weight of the carbon part of the compound is 48.04 g/mol
theorem molecular_weight_of_carbon_part : 
  (number_of_carbons * atomic_weight_carbon) = 48.04 := by
  sorry

end molecular_weight_of_carbon_part_l561_561116


namespace proof_problem_l561_561776

def greatest_int (x : ℝ) : ℤ := int.floor x

def M (x : ℝ) : ℝ := Real.sqrt (greatest_int (Real.sqrt x))
def N (x : ℝ) : ℤ := greatest_int (Real.sqrt (Real.sqrt x))

theorem proof_problem (x : ℝ) (hx : 1 ≤ x) : ∃ M N, 
  M = Real.sqrt (greatest_int (Real.sqrt x)) ∧ 
  N = greatest_int (Real.sqrt (Real.sqrt x)) ∧ 
  ¬ (M = N ∨ M < N ∨ M > N) := 
sorry

end proof_problem_l561_561776


namespace robert_balls_l561_561814

theorem robert_balls (R T : ℕ) (hR : R = 25) (hT : T = 40 / 2) : R + T = 45 :=
by
  sorry

end robert_balls_l561_561814


namespace stratified_sampling_l561_561625

open Finset

-- Definitions for combinatorial counting
def choose (n k : ℕ) : ℕ := (range n).choose k

-- Problem statement in Lean
theorem stratified_sampling (n_females n_males k_females k_males : ℕ) :
  n_females = 6 → n_males = 4 → k_females = 3 → k_males = 2 →
  choose n_females k_females * choose n_males k_males = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end stratified_sampling_l561_561625


namespace total_number_of_subsets_of_P_l561_561192

noncomputable theory
open Set

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℤ := {x | x^2 - x - 2 ≤ 0}
def P : Set ℤ := M ∩ N

theorem total_number_of_subsets_of_P : ∃ n, n = 4 ∧ Fintype.card (Set.toFinset P) = n := 
by {
  have M := {0, 1, 2, 3},
  have N := {x | x^2 - x - 2 ≤ 0},
  let P := Set.Inter M N,
  sorry
}

end total_number_of_subsets_of_P_l561_561192


namespace length_of_platform_is_correct_l561_561481

-- Definitions for conditions
def speed_kmph : ℝ := 72
def time_seconds : ℝ := 30
def length_train : ℝ := 350.048

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

-- Speed in m/s
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Total distance covered in 30 seconds which is length_train + length_platform
def total_distance : ℝ := speed_mps * time_seconds

-- Define the length of the platform
def length_platform : ℝ := total_distance - length_train

-- Theorem stating the length of the platform
theorem length_of_platform_is_correct : length_platform = 249.952 := by
  -- Assuming all necessary definitions and conditions
  sorry

end length_of_platform_is_correct_l561_561481


namespace auction_site_TVs_proof_l561_561967

variable (first_store_TVs online_store_TVs auction_site_TVs : ℕ)
variable (total_TVs : ℕ)

-- Conditions
def condition1 : Prop := first_store_TVs = 8
def condition2 : Prop := online_store_TVs = 3 * first_store_TVs
def condition3 : Prop := total_TVs = first_store_TVs + online_store_TVs + auction_site_TVs
def condition4 : Prop := total_TVs = 42

-- Theorem asserting the number of TVs looked at the auction site
theorem auction_site_TVs_proof 
  (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  auction_site_TVs = 10 := by
  sorry

end auction_site_TVs_proof_l561_561967


namespace largest_six_consecutive_nonprime_under_50_l561_561681

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → (m = 1 ∨ m = n)

def consecutiveNonPrimes (m : ℕ) : Prop :=
  ∀ i : ℕ, i < 6 → ¬ isPrime (m + i)

theorem largest_six_consecutive_nonprime_under_50 (n : ℕ) :
  (n < 50 ∧ consecutiveNonPrimes n) →
  n + 5 = 35 :=
by
  intro h
  sorry

end largest_six_consecutive_nonprime_under_50_l561_561681


namespace range_of_a_l561_561832

noncomputable def domain_of_f : set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

noncomputable def solution_set_B (a : ℝ) : set ℝ := { x : ℝ | a ≤ x ∧ x ≤ a + 3 }

theorem range_of_a (a : ℝ) : (domain_of_f ⊆ solution_set_B a) ↔ (-1 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l561_561832


namespace problem_part1_problem_part2_l561_561154

noncomputable def C : ℝ := sorry

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 3) * sqrt 3

theorem problem_part1 (a b c : ℝ) (A B C : ℝ) (h1 : a^2 + b^2 = 6 * a * b * cos C)
  (h2 : sin C ^ 2 = 2 * sqrt 3 * sin A * sin B) : 
  C = π / 6 := sorry

theorem problem_part2 (A : ℝ) (hf : π / 3 < A ∧ A < π / 2) : 
  - (3 / 2) < f A ∧ f A < 0 := sorry

end problem_part1_problem_part2_l561_561154


namespace daily_coffee_machine_cost_l561_561761

def coffee_machine_cost := 200 -- $200
def discount := 20 -- $20
def daily_coffee_cost := 2 * 4 -- $8/day
def days_to_pay_off := 36 -- 36 days

theorem daily_coffee_machine_cost :
  (days_to_pay_off * daily_coffee_cost - (coffee_machine_cost - discount)) / days_to_pay_off = 3 := 
by
  -- Using the given conditions: 
  -- coffee_machine_cost = 200
  -- discount = 20
  -- daily_coffee_cost = 8
  -- days_to_pay_off = 36
  sorry

end daily_coffee_machine_cost_l561_561761


namespace quotient_base6_2134_div_14_eq_81_l561_561104

noncomputable def quotient_base6_div (a b : ℕ) : ℕ :=
  let a_base10 := 2 * 6^3 + 1 * 6^2 + 3 * 6^1 + 4 * 6^0
      b_base10 := 1 * 6^1 + 4 * 6^0
      quotient_base10 := a_base10 / b_base10
      q_quot := quotient_base10 / 6
      r_quot := quotient_base10 % 6
  in q_quot * 10 + r_quot

-- Prove that the quotient of 2134 in base 6 divided by 14 in base 6 is equal to 81 in base 6
theorem quotient_base6_2134_div_14_eq_81 : quotient_base6_div 2134 14 = 81 :=
by
  sorry

end quotient_base6_2134_div_14_eq_81_l561_561104


namespace length_BD_l561_561289

-- Definitions of points and lengths
variables {A B C D : Type} [point A] [point B] [point C] [point D]

-- Given conditions:
-- Point D is on line AC
-- Angles: ∠ADB = 90°, ∠ABD = 30°, ∠BAC = 45°
-- Length of AD = 6 units

axiom angle_ADB {A B C D : Type} [point A] [point B] [point C] [point D] : 
  angle_on_AC D -> angle A D B = 90

axiom angle_ABD {A B C D : Type} [point A] [point B] [point C] [point D] : 
  angle_on_AC D -> angle A B D = 30

axiom angle_BAC {A B C D : Type} [point A] [point B] [point C] [point D] : 
  angle_on_AC D -> angle B A C = 45

axiom length_AD {A B C D : Type} [point A] [point B] [point C] [point D] : 
  length_on_AC D -> length_x_y A D 6

-- Theorem to prove that the length BD = 3√3
theorem length_BD {A B C D : Type} [point A] [point B] [point C] [point D] :
  length_on_AC D :=
  begin
    sorry
  end

end length_BD_l561_561289


namespace cost_of_croissants_l561_561622

theorem cost_of_croissants (n : ℕ) (s : ℕ) (c : ℕ) (price_per_dozen : ℕ → ℝ) (people : ℕ) (sandwiches_pp : ℕ) (croissants_per_dozen : ℕ) :
  people = 24 → sandwiches_pp = 2 → croissants_per_dozen = 12 → price_per_dozen croissants_per_dozen = 8.0 →
  n = people * sandwiches_pp → s = n / croissants_per_dozen → c = s * price_per_dozen croissants_per_dozen → c = 32.0 := by
  intro h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2] at h5
  rw [Nat.mul_comm] at h5
  rw [←h5] at h6
  rw [h3] at h6
  have h8 : n / croissants_per_dozen = s := by
    rw [←h6]
    rfl
  rw [←h6, h4] at h7
  sorry

end cost_of_croissants_l561_561622


namespace imaginary_part_of_z_l561_561714

theorem imaginary_part_of_z :
  ∀ (z : ℂ), (3 - 4*complex.I) * z = complex.abs (4 + 3*complex.I) → complex.im z = 4/5 :=
by
  intro z
  sorry

end imaginary_part_of_z_l561_561714


namespace hexagon_side_length_l561_561536

/- Definitions and Conditions -/
variables (A B C D E F G H J : Type*)
variables (a b c : ℝ) -- lengths of sides of triangle ABC
variables (α β γ : ℝ) -- angles at vertices A, B, and C respectively

/- Main Theorem -/
theorem hexagon_side_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
(h_triangle_inequality_1 : a + b > c) (h_triangle_inequality_2 : a + c > b) (h_triangle_inequality_3 : b + c > a) :
∃ d : ℝ, d = (a * b * c) / (a * b + b * c + c * a) :=
begin
  sorry
end

end hexagon_side_length_l561_561536


namespace total_cost_is_32_l561_561620

-- Step d: Rewrite the math proof problem in Lean 4 statement.

-- Define the number of people on the committee
def num_people : ℕ := 24

-- Define the number of sandwiches per person
def sandwiches_per_person : ℕ := 2

-- Define the number of croissants per set
def croissants_per_set : ℕ := 12

-- Define the cost per set of croissants
def cost_per_set : ℕ := 8

-- Define the number of croissants needed
def croissants_needed : ℕ := sandwiches_per_person * num_people

-- Calculate the number of sets needed
def sets_needed : ℕ := croissants_needed / croissants_per_set

-- Calculate the total cost
def total_cost : ℕ := sets_needed * cost_per_set

-- Prove the total cost is $32
theorem total_cost_is_32 : total_cost = 32 := 
by
  -- importing necessary library
  -- proving the total cost calculation, we already know
  -- the number of croissants_needed, sets_needed, calculating total_cost
  rw [croissants_needed, sets_needed, total_cost],
  simp,
  sorry

end total_cost_is_32_l561_561620


namespace projection_of_vector_l561_561118

def vector := ℝ × ℝ × ℝ

def proj (a b : vector) : vector :=
  let dot_prod (u v : vector) : ℝ :=
    u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let scalar := (dot_prod a b) / (dot_prod b b)
  (scalar * b.1, scalar * b.2, scalar * b.3)

theorem projection_of_vector :
  proj (3, 0, -2) (2, 1, -1) = (8/3, 4/3, -4/3) :=
by 
  sorry

end projection_of_vector_l561_561118


namespace sum_of_roots_eq_6_l561_561711

theorem sum_of_roots_eq_6 : ∀ (x1 x2 : ℝ), (x1 * x1 = x1 ∧ x1 * x2 = x2) → (x1 + x2 = 6) :=
by
   intro x1 x2 hx
   have H : x1 + x2 = 6 := sorry
   exact H

end sum_of_roots_eq_6_l561_561711


namespace solution_to_problem_l561_561563

theorem solution_to_problem (a x y n m : ℕ) (h1 : a * (x^n - x^m) = (a * x^m - 4) * y^2)
  (h2 : m % 2 = n % 2) (h3 : (a * x) % 2 = 1) : 
  x = 1 :=
sorry

end solution_to_problem_l561_561563


namespace generating_function_equivalence_probability_correct_l561_561907

variable (X : ℕ → ℕ → ℝ)
variable (n : ℕ)
variable (b : ℕ → ℝ)

def binomial_moments (b : ℕ → ℝ) (k : ℕ) : Prop :=
  b k = (k.factorial⁻¹ * (finset.sum (finset.range k.succ)
    (λ i, (X i k.to_nat))) : ℝ)

def generating_function (G : ℝ → ℝ) (s : ℝ) : Prop :=
  G s = (finset.sum (finset.range (n + 1))
    (λ k, b k * (s-1)^k))

def equivalent_generating_function (G : ℝ → ℝ) (s : ℝ) : Prop :=
  G s = (finset.sum (finset.range (n + 1))
    (λ i, s^i * (finset.sum (finset.Icc i n)
    (λ k, (-1)^(k - i) * (nat.choose k i : ℝ) * b k))))

def probability (p : ℕ → ℝ) : Prop :=
  ∀ i, p i = (finset.sum (finset.Icc i n)
    (λ k, (-1)^(k - i) * (nat.choose k i : ℝ) * b k))

theorem generating_function_equivalence (G : ℝ → ℝ) (s : ℝ) :
  (∀ k, binomial_moments b k) →
  generating_function G s →
  equivalent_generating_function G s := sorry

theorem probability_correct (p : ℕ → ℝ) :
  (∀ k, binomial_moments b k) →
  ∀ i, (probability p i) := sorry

end generating_function_equivalence_probability_correct_l561_561907


namespace speed_of_current_l561_561931

theorem speed_of_current (d : ℝ) (c : ℝ) : 
  ∀ (h1 : ∀ (t : ℝ), d = (30 - c) * (40 / 60)) (h2 : ∀ (t : ℝ), d = (30 + c) * (25 / 60)), 
  c = 90 / 13 := by
  sorry

end speed_of_current_l561_561931


namespace triangle_area_difference_l561_561301

noncomputable def areaOfRightTriangle (base height : ℝ) : ℝ := 0.5 * base * height

theorem triangle_area_difference (A B C E D : Type) 
  [AB : A ≠ B] [BC : B ≠ C] [AE : A ≠ E]
  (AB_length : ℝ := 5) (BC_length : ℝ := 7) (AE_length : ℝ := 9)
  (intersect: D ∈ line (AC) ∧ D ∈ line (BE))
  (right_angle_EAB : ∀ {r}, ∠ EAB r = 90)
  (right_angle_ABC : ∀ {r}, ∠ ABC r = 90) :
    abs (areaOfRightTriangle AB_length AE_length - areaOfRightTriangle AB_length BC_length) = 5 :=
by
  sorry


end triangle_area_difference_l561_561301


namespace prove_inequality1_prove_inequality2_l561_561634

noncomputable def inequality1 (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : Prop :=
  sqrt (a^2 - a * b + b^2) + sqrt (b^2 - b * c + c^2) > sqrt (c^2 - c * a + a^2)

theorem prove_inequality1 (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : inequality1 a b c h :=
sorry

noncomputable def inequality2 (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : Prop :=
  sqrt (a^2 + a * b + b^2) + sqrt (b^2 + b * c + c^2) > sqrt (c^2 + c * a + a^2)

theorem prove_inequality2 (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) : inequality2 a b c h :=
sorry

end prove_inequality1_prove_inequality2_l561_561634


namespace initial_weight_of_cheese_l561_561023

theorem initial_weight_of_cheese :
  let initial_weight : Nat := 850
  -- final state after 3 bites
  let final_weight1 : Nat := 25
  let final_weight2 : Nat := 25
  -- third state
  let third_weight1 : Nat := final_weight1 + final_weight2
  let third_weight2 : Nat := final_weight1
  -- second state
  let second_weight1 : Nat := third_weight1 + third_weight2
  let second_weight2 : Nat := third_weight1
  -- first state
  let first_weight1 : Nat := second_weight1 + second_weight2
  let first_weight2 : Nat := second_weight1
  -- initial state
  let initial_weight1 : Nat := first_weight1 + first_weight2
  let initial_weight2 : Nat := first_weight1
  initial_weight = initial_weight1 + initial_weight2 :=
by
  sorry

end initial_weight_of_cheese_l561_561023


namespace number_of_integers_between_sqrt8_and_sqrt72_l561_561212

theorem number_of_integers_between_sqrt8_and_sqrt72 : 
  let a := Int.ceil (Real.sqrt 8)
  let b := Int.floor (Real.sqrt 72)
  b - a + 1 = 6 :=
begin
  sorry
end

end number_of_integers_between_sqrt8_and_sqrt72_l561_561212


namespace zero_distribution_l561_561408

def f (x : ℝ) : ℝ := -x^3 + x^2 + x - 2

theorem zero_distribution :
  (∃! x : ℝ, f x = 0 ∧ x ∈ set.Ioo (-∞) (-1/3)) :=
by
  have h₁ : f (-1/3) < 0 := by sorry
  have h₂ : f (1) < 0 := by sorry
  sorry

end zero_distribution_l561_561408


namespace ratio_of_areas_of_circles_l561_561270

theorem ratio_of_areas_of_circles (C_A C_B L R_A R_B : ℝ)
  (h_C_A : C_A = 2 * Real.pi * R_A)
  (h_C_B : C_B = 2 * Real.pi * R_B)
  (h_arc_A : 45 / 360 * C_A = L)
  (h_arc_B : 30 / 360 * C_B = L) :
  (Real.pi * R_A ^ 2) / (Real.pi * R_B ^ 2) = 4 / 9 := 
sorry

end ratio_of_areas_of_circles_l561_561270


namespace new_person_weight_l561_561385

theorem new_person_weight (W : ℝ) (old_weight : ℝ) (increase_per_person : ℝ) (num_persons : ℕ)
  (h1 : old_weight = 68)
  (h2 : increase_per_person = 5.5)
  (h3 : num_persons = 5)
  (h4 : W = old_weight + increase_per_person * num_persons) :
  W = 95.5 :=
by
  sorry

end new_person_weight_l561_561385


namespace minimum_toothpicks_for_5_squares_l561_561356

theorem minimum_toothpicks_for_5_squares :
  let single_square_toothpicks := 4
  let additional_shared_side_toothpicks := 3
  ∃ n, n = single_square_toothpicks + 4 * additional_shared_side_toothpicks ∧ n = 15 :=
by
  sorry

end minimum_toothpicks_for_5_squares_l561_561356


namespace sum_alternating_powers_l561_561438

theorem sum_alternating_powers (n : ℕ) (hn : n = 2006) : ∑ k in Finset.range (n + 1), (-1)^(k + 1) = 0 := 
by
  sorry

end sum_alternating_powers_l561_561438


namespace equation_of_perpendicular_line_l561_561395

theorem equation_of_perpendicular_line (a b c : ℝ) (p : ℝ × ℝ)
  (h1 : a = -1) (h2 : b = 3) (h3 : c = 1) (h4 : p = (-1, 3)) :
  ∃ m n k : ℝ, x - 2 * y + 1 = 0 ∧ y + 2 * x - 1 = 0 :=
by
  have slope_of_original : ℝ := 1 / 2
  have slope_of_perpendicular : ℝ := -2
  have point : ℝ × ℝ := (-1, 3)
  have line_eq : ℝ := y - 3 + slope_of_perpendicular * x
  have simplified_line_eq : ℝ := y + 2 * x - 1
  exact (m = 2 ∧ n = -1 ∧ k = 0)

end equation_of_perpendicular_line_l561_561395


namespace four_red_four_black_points_parallelogram_l561_561307

theorem four_red_four_black_points_parallelogram :
  ∃ (R B : Fin 4 → ℝ × ℝ), (∀ i j k : Fin 4, i ≠ j → i ≠ k → j ≠ k →
  ∃ l : Fin 4, (l ∉ {i, j, k}) ∧ parallelogram (R i, R j, R k, B l) ∨ parallelogram (B i, B j, B k, R l)) :=
sorry

end four_red_four_black_points_parallelogram_l561_561307


namespace integer_count_between_sqrt8_and_sqrt72_l561_561223

theorem integer_count_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ( ∀ x : ℤ, (⌊Real.sqrt 8⌋.to_nat + 1) ≤ x ∧ x ≤ ⌊Real.sqrt 72⌋ - 1 → x = 6 ) :=
by 
  -- Define the floor and ceiling functions
  have sqrt_8_ceil : ⌈Real.sqrt 8⌉ = 3 := sorry
  have sqrt_72_floor : ⌊Real.sqrt 72⌋ = 8 := sorry

  use 6,
  split,
  { refl },
  { intros x hx,
    sorry
  }

-- Additional supporting facts
lemma sqrt_8_approx : Real.sqrt 8 ≈ 2.83 := sorry
lemma sqrt_72_approx : Real.sqrt 72 ≈ 8.49 := sorry

end integer_count_between_sqrt8_and_sqrt72_l561_561223


namespace max_disconnected_empty_regions_l561_561357

def strip := list (list bool)

structure chessboard := 
  (size : ℕ)
  (strips : list strip)
  (no_overlap : ∀ s1 s2 ∈ strips, s1 ≠ s2 → disjoint s1 s2)

def is_region_disconnected (board : chessboard) (region1 region2 : list (nat × nat)) : Prop :=
  ∀ pos1 ∈ region1, ∀ pos2 ∈ region2, ¬ adjacent pos1 pos2

def number_of_empty_regions (board : chessboard) : ℕ :=
  sorry -- Placeholder for the region counting logic

theorem max_disconnected_empty_regions (m n : ℕ) (board : chessboard)
    (h_size : board.size = m)
    (h_strip_count : board.strips.length = n): 
    number_of_empty_regions board ≤ n + 1 :=
sorry

end max_disconnected_empty_regions_l561_561357


namespace max_value_of_g_on_interval_l561_561115

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g_on_interval : ∃ x : ℝ, (0 ≤ x ∧ x ≤ Real.sqrt 2) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ Real.sqrt 2) → g y ≤ g x) ∧ g x = 25 / 8 := by
  sorry

end max_value_of_g_on_interval_l561_561115


namespace polar_line_angle_xaxis_l561_561095

-- Define the polar equation condition
def polarEq (r θ : ℝ) : Prop := θ = π / 6

-- Define the statement to prove
theorem polar_line_angle_xaxis :
  (∃ r : ℝ, polarEq r (π / 6)) →
  (∀ r : ℝ, polarEq r (π / 6)) ↔
  ∃ m : ℝ, m = Real.tan (π / 6) := 
sorry
 

end polar_line_angle_xaxis_l561_561095


namespace probability_of_rain_at_least_once_l561_561132

theorem probability_of_rain_at_least_once 
  (P_sat : ℝ) (P_sun : ℝ) (P_mon : ℝ)
  (h_sat : P_sat = 0.30)
  (h_sun : P_sun = 0.60)
  (h_mon : P_mon = 0.50) :
  (1 - (1 - P_sat) * (1 - P_sun) * (1 - P_mon)) * 100 = 86 :=
by
  rw [h_sat, h_sun, h_mon]
  sorry

end probability_of_rain_at_least_once_l561_561132


namespace max_area_rectangle_l561_561856

theorem max_area_rectangle (P : ℕ) (hP : P = 40) : ∃ A : ℕ, A = 100 ∧ ∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A := by
  sorry

end max_area_rectangle_l561_561856


namespace probability_interval_l561_561627

open ProbabilityTheory

-- Define the normal random variable X with mean 5 and variance 4
def X : ProbabilityMassFunction ℝ := ⟨λ x, pdf (Normal 5 (sqrt 4)) x⟩

-- State the property to prove: P(1 < X ≤ 7) = 0.9759
theorem probability_interval :
  (ProbabilityMassFunction.prob X (λ x, 1 < x ∧ x ≤ 7)) = 0.9759 :=
sorry

end probability_interval_l561_561627


namespace count_integers_between_sqrts_l561_561232

theorem count_integers_between_sqrts : 
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  (upper_bound - lower_bound + 1) = 6 :=
by
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  calc (upper_bound - lower_bound + 1) = 6 : sorry

end count_integers_between_sqrts_l561_561232


namespace cubic_root_expression_l561_561339

theorem cubic_root_expression (p q r : ℝ) (h1 : p + q + r = 0) (h2 : p * q + p * r + q * r = -2) (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -24 :=
sorry

end cubic_root_expression_l561_561339


namespace intersection_of_A_and_B_l561_561718

def set_A : Set ℝ := {x | -x^2 - x + 6 > 0}
def set_B : Set ℝ := {x | 5 / (x - 3) ≤ -1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x | -2 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l561_561718


namespace polynomial_identity_l561_561106

theorem polynomial_identity {P : ℝ[X]} (hP : P ≠ 0) 
  (h : ∀ (x : ℝ), P.eval (x^2) = (P.eval x)^2) :
  ∃ n : ℕ, P = polynomial.C 1 * polynomial.X ^ n :=
sorry

end polynomial_identity_l561_561106


namespace number_of_positive_divisors_2310_l561_561596

theorem number_of_positive_divisors_2310 : 
  let n := 2310 in
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)] in
  let t := (factorization.map (λ p : ℕ × ℕ, p.snd + 1)).prod in
  t = 32 :=
by
  let n := 2310
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let t := (factorization.map (λ p, p.snd + 1)).prod
  sorry

end number_of_positive_divisors_2310_l561_561596


namespace largest_result_among_expressions_l561_561509

def E1 : ℕ := 992 * 999 + 999
def E2 : ℕ := 993 * 998 + 998
def E3 : ℕ := 994 * 997 + 997
def E4 : ℕ := 995 * 996 + 996

theorem largest_result_among_expressions : E4 > E1 ∧ E4 > E2 ∧ E4 > E3 :=
by sorry

end largest_result_among_expressions_l561_561509


namespace number_of_divisors_of_2310_l561_561574

theorem number_of_divisors_of_2310 : 
  let n := 2310 in 
  let prime_factors := [2, 3, 5, 7, 11] in
  ∃ k : ℕ, k = prime_factors.length ∧
  (∀ i, i < k → prime_factors.nth i = some 2 ∨ prime_factors.nth i = some 3 ∨ prime_factors.nth i = some 5 ∨ prime_factors.nth i = some 7 ∨ prime_factors.nth i = some 11) →
  (n.factorization.to_nat * 1).0 = 32 :=
begin
  sorry
end

end number_of_divisors_of_2310_l561_561574


namespace max_value_inequality_l561_561188

theorem max_value_inequality (a x₁ x₂ : ℝ) (h_a : a < 0)
  (h_sol : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x₁ < x ∧ x < x₂) :
    x₁ + x₂ + a / (x₁ * x₂) ≤ - 4 * Real.sqrt 3 / 3 := by
  sorry

end max_value_inequality_l561_561188


namespace time_ratio_l561_561725

theorem time_ratio : 
  (total_exam_time_hours = 3) ∧ 
  (time_for_A_minutes = 120) → 
  (time_for_A_minutes : time_for_B_minutes) = 2 : 1 :=
by
  sorry

end time_ratio_l561_561725


namespace distance_from_dormitory_to_city_l561_561454

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h1 : D = (1/2) * D + (1/4) * D + 6) : D = 24 := 
  sorry

end distance_from_dormitory_to_city_l561_561454


namespace max_sum_when_product_is_399_l561_561730

theorem max_sum_when_product_is_399 :
  ∃ (X Y Z : ℕ), X * Y * Z = 399 ∧ X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X ∧ X + Y + Z = 29 :=
by
  sorry

end max_sum_when_product_is_399_l561_561730


namespace scale_length_l561_561494

theorem scale_length (num_parts : ℕ) (part_length : ℕ) (total_length : ℕ) 
  (h1 : num_parts = 5) (h2 : part_length = 16) : total_length = 80 :=
by
  sorry

end scale_length_l561_561494


namespace final_cost_correct_l561_561077

def dozen_cost : ℝ := 18
def num_dozen : ℝ := 2.5
def discount_rate : ℝ := 0.15

def cost_before_discount : ℝ := num_dozen * dozen_cost
def discount_amount : ℝ := discount_rate * cost_before_discount

def final_cost : ℝ := cost_before_discount - discount_amount

theorem final_cost_correct : final_cost = 38.25 := by
  -- The proof would go here, but we just provide the statement.
  sorry

end final_cost_correct_l561_561077


namespace line_tangent_to_ellipse_l561_561999

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = mx + 2 → x^2 + 9 * y^2 = 9 → ∃ u, y = u) → m^2 = 1 / 3 := 
by
  intro h
  sorry

end line_tangent_to_ellipse_l561_561999


namespace first_player_winning_l561_561985

def configuration (k : ℕ) (n : ℕ) : Type := fin k → ℕ

def winning_positions (k : ℕ) : set (configuration k n) := sorry -- Assume the set B is defined

theorem first_player_winning (k : ℕ) (n : ℕ) (initial_config : configuration k n) :
  initial_config ∈ winning_positions k ↔ ∃ str, winning_strategy str := sorry

end first_player_winning_l561_561985


namespace petya_ice_cream_money_dale_additional_nuts_l561_561680

-- Part 1: Determine if Petya has enough money for the ice cream
theorem petya_ice_cream_money : 
  ∀ (n : ℕ), n = 400 → (n^5 - (n-1)^2 * (n^3 + 2*n^2 + 3*n + 4)) ≤ 2000 := 
by 
  intro n h_n
  rw h_n
  sorry

-- Part 2: Determine the additional nuts Dale needs to gather
theorem dale_additional_nuts : 
  ∀ (chip_nuts dale_nuts add_nuts : ℕ), chip_nuts = 120 → dale_nuts = 147 → 4 * add_nuts + chip_nuts = add_nuts + dale_nuts → add_nuts = 9 := 
by
  intros chip_nuts dale_nuts add_nuts h_chip h_dale h_eq
  rw [h_chip, h_dale] at h_eq
  sorry

end petya_ice_cream_money_dale_additional_nuts_l561_561680


namespace find_x_l561_561488

theorem find_x (x : ℝ) (h1 : x ≠ 0) (h2 : x = (1 / x) * (-x) + 3) : x = 2 :=
by
  sorry

end find_x_l561_561488


namespace exists_integers_n_and_ai_l561_561878

open Nat

theorem exists_integers_n_and_ai :
  ∃ n : ℤ, ∃ (a : Fin 2012 → ℤ),
  (∀ i, 1 < a i) ∧
  (n^2 = ∑ i in Fin.range 2012, a i ^ prime (i + 1)) :=
by
  sorry

end exists_integers_n_and_ai_l561_561878


namespace dan_money_left_l561_561539

def money_left (initial : ℝ) (candy_bar : ℝ) (chocolate : ℝ) (soda : ℝ) (gum : ℝ) : ℝ :=
  initial - candy_bar - chocolate - soda - gum

theorem dan_money_left :
  money_left 10 2 3 1.5 1.25 = 2.25 :=
by
  sorry

end dan_money_left_l561_561539


namespace number_of_integers_satisfying_inequality_l561_561851

theorem number_of_integers_satisfying_inequality : 
  (∃ l : List ℕ, l = List.filter (λp, -1 < Real.sqrt p - Real.sqrt 100 ∧ Real.sqrt p - Real.sqrt 100 < 1) (List.range 121) ∧ l.length = 39) := 
sorry

end number_of_integers_satisfying_inequality_l561_561851


namespace number_of_ways_to_select_team_l561_561353

theorem number_of_ways_to_select_team : 
  let boys := 6
  let girls := 8
  let n := boys + girls
  let r := 6
  combinatoria.binom n r = 3003 :=
by 
  let boys := 6
  let girls := 8
  let n := boys + girls
  let r := 6
  calc
    combinatoria.binom n r 
    = combinatoria.binom 14 6 : by rw [show n = 14 by {dsimp [n, boys, girls], ring}]
    = 3003               : by sorry -- Proof calculation placeholder

end number_of_ways_to_select_team_l561_561353


namespace range_of_x_l561_561168

variable {α : Type*} [LinearOrder α]

def f (x : α) : α := sorry -- Assuming the definition of f is given

axiom f_mono : Monotone f
axiom f_domain : ∀ x, 0 ≤ x → f x

theorem range_of_x (x : α) :
  0 ≤ x → f (2 * x - 1) < f (1 / 3) ↔ (1 / 2 ≤ x ∧ x < 2 / 3) :=
by
  sorry

end range_of_x_l561_561168


namespace unique_prime_solution_l561_561093

-- Define the problem in terms of prime numbers and checking the conditions
open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_solution (p : ℕ) (hp : is_prime p) (h1 : is_prime (p^2 - 6)) (h2 : is_prime (p^2 + 6)) : p = 5 := 
sorry

end unique_prime_solution_l561_561093


namespace last_number_of_ratio_l561_561904

theorem last_number_of_ratio (A B C : ℕ) (h1 : 5 * B = A) (h2 : 4 * B = C) (h3 : A + B + C = 1000) : C = 400 :=
by
  sorry

end last_number_of_ratio_l561_561904


namespace median_of_set_l561_561153

theorem median_of_set (x y : ℝ) (h_avg : (7 + 8 + 9 + x + y) / 5 = 8) :
  median ([7, 8, 9, x, y] : multiset ℝ) = 8 :=
by sorry

end median_of_set_l561_561153


namespace partition_possible_iff_l561_561096

theorem partition_possible_iff (a b : ℕ) (n : ℕ) (a' b' : ℕ) (Ha : a = 2 ^ n * a') (Hb : b = 2 ^ n * b') (Ha_odd : Odd a') (Hb_odd : Odd b') :
  (∃ H1 H2 : Set ℕ, H1 ∪ H2 = { x | x > 0 } ∧ H1 ∩ H2 = ∅ ∧
    (∀ x y ∈ H1, x ≠ y → x - y ≠ a ∧ x - y ≠ b) ∧
    (∀ x y ∈ H2, x ≠ y → x - y ≠ a ∧ x - y ≠ b)) ↔ True := sorry

end partition_possible_iff_l561_561096


namespace weight_of_replaced_man_l561_561826

variable (avg_weight_before : ℝ)  -- the average weight before replacement
variable (avg_weight_after : ℝ := avg_weight_before + 2.5)  -- the average weight after replacement
variable (weight_new_man : ℝ := 93)  -- weight of the new man
variable (total_men : ℕ := 10)  -- number of men
variable (increment_per_man : ℝ := 2.5)  -- increase in average weight per man

noncomputable def total_weight_before := total_men * avg_weight_before
noncomputable def total_weight_after := total_men * avg_weight_after
noncomputable def weight_replaced_man := weight_new_man - (total_weight_after - total_weight_before)

theorem weight_of_replaced_man : weight_replaced_man = 68 :=
by
  sorry

end weight_of_replaced_man_l561_561826


namespace four_x_sq_plus_nine_y_sq_l561_561684

theorem four_x_sq_plus_nine_y_sq (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 9)
  (h2 : x * y = -12) : 
  4 * x^2 + 9 * y^2 = 225 := 
by
  sorry

end four_x_sq_plus_nine_y_sq_l561_561684


namespace train_crossing_time_approx_l561_561952

def speed_km_per_hr := 60 -- Speed in km/h
def train_length_m := 350 -- Length of the train in meters

-- Conversion of speed from km/h to m/s
def speed_m_per_s : ℚ := (speed_km_per_hr * 1000) / 3600

-- Theorem stating that the time for the train to cross the pole is approximately 21 seconds.
theorem train_crossing_time_approx : 
  (train_length_m / speed_m_per_s) ≈ 21 := 
by
  sorry

end train_crossing_time_approx_l561_561952


namespace line_not_in_second_quadrant_l561_561837

variable (a : ℝ)

def line_eq (a : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y, (a - 2) * y = (3 * a - 1) * x - 1

theorem line_not_in_second_quadrant (a : ℝ) :
  (∀ x y, x ≤ 0 ∧ y ≥ 0 → ¬line_eq a x y) ↔ a ≥ 2 := by
  sorry

end line_not_in_second_quadrant_l561_561837


namespace predict_grandson_height_l561_561485

noncomputable def heights : List ℝ := [173, 170, 176, 182]

theorem predict_grandson_height (heights : List ℝ) :
  let b := (heights.nth_le 0  sorry * heights.nth_le 1 sorry + heights.nth_le 1 sorry * heights.nth_le 2 sorry + heights.nth_le 2 sorry * heights.nth_le 3 sorry - 3 * heights.nth_le 0 sorry * heights.nth_le 2 sorry) / (heights.nth_le 0 sorry ^ 2 + heights.nth_le 1 sorry ^ 2 + heights.nth_le 2 sorry ^ 2 - 3 * heights.nth_le 0 sorry ^ 2)
  let a := 3
  let y := b * heights.nth_le 3 sorry + a
  y = 185 :=
by
  sorry

end predict_grandson_height_l561_561485


namespace concurrency_of_lines_l561_561748

noncomputable theory
open_locale classical

variables {Point : Type*} [MetricSpace Point] [InnerProductSpace ℝ Point]
variables (A B C D E F M H P N G Q T : Point)

-- Conditions
variables (ABCDEF_complete : CompleteQuadrilateral A B C D E F)
variables (AC_gt_AE : dist A C > dist A E)
variables (BE_perp_AC : ∠ B E = ⟂)
variables (CF_perp_AB : ∠ C F = ⟂)
variables (B_perp_M : is_perpendicular (line_through B M) (line_through C E))
variables (M_on_CE : M ∈ (line_through C E))
variables (H_on_CD : H ∈ (line_through C D))
variables (P_on_extension_EA : P ∈ (extension A E))
variables (F_perp_N : is_perpendicular (line_through F N) (line_through C E))
variables (N_on_CE : N ∈ (line_through C E))
variables (G_on_DE : G ∈ (line_through D E))
variables (Q_on_extension_CA : Q ∈ (extension C A))

-- Proof Goal
theorem concurrency_of_lines : AreConcurrent (line_through P Q) (line_through B F) (line_through H G) (line_through M N) :=
sorry

end concurrency_of_lines_l561_561748


namespace sequence_length_arithmetic_sequence_l561_561972

theorem sequence_length_arithmetic_sequence :
  ∃ n : ℕ, ∀ (a d : ℕ), a = 2 → d = 3 → a + (n - 1) * d = 2014 ∧ n = 671 :=
by {
  sorry
}

end sequence_length_arithmetic_sequence_l561_561972


namespace unique_nonzero_solution_l561_561890

theorem unique_nonzero_solution (x : ℝ) (h : x ≠ 0) : (3 * x)^3 = (9 * x)^2 → x = 3 :=
by
  sorry

end unique_nonzero_solution_l561_561890


namespace remainder_7_pow_7_pow_7_pow_7_mod_2000_l561_561995

theorem remainder_7_pow_7_pow_7_pow_7_mod_2000 :
  7 ^ (7 ^ (7 ^ 7)) % 2000 = 343 :=
by
  have lambda_2000 : ∀ (n : ℕ), n = 2000 → nat.carmichael n = 200 := sorry
  have pow_7_7_mod_200 : ∀ (n : ℕ),  (7 ^ 7) % 200 = 43 := sorry
  have pow_7_43_mod_2000 : ∀ (n : ℕ), (7 ^ 43) % 2000 = 343 := sorry
  sorry

end remainder_7_pow_7_pow_7_pow_7_mod_2000_l561_561995


namespace sum_of_squares_of_roots_l561_561532

noncomputable def polynomial := λ x : ℝ, x^4 + 12 * x^3 + x^2 + 20 * x + 15

theorem sum_of_squares_of_roots :
  let r1, r2, r3, r4 in 
  (r1 + r2 + r3 + r4)^2 - 2 * (r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4) = 142 :=
by
  let r1 := -12 -- Using Vieta's formulas for sum of roots
  let r2 := 1  -- Using Vieta's formulas for sum of product of roots taken two at a time
  let r3 := sorry
  let r4 := sorry
  sorry -- The proof step will be filled here following Lean syntax.

end sum_of_squares_of_roots_l561_561532


namespace cos_identity_l561_561015

theorem cos_identity (α β γ : ℝ)
  (h : cos (2 * α) + cos (2 * β) + cos (2 * γ) + 4 * cos α * cos β * cos γ + 1 = 0) :
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 + 2 * cos α * cos β * cos γ = 1 :=
by
  sorry

end cos_identity_l561_561015


namespace gender_judgment_independence_test_l561_561101

def num_male : ℕ := 2548
def male_opposition : ℕ := 1560
def num_female : ℕ := 2452
def female_opposition : ℕ := 1200

theorem gender_judgment_independence_test :
  (let total_participants := num_male + num_female,
       total_opposition := male_opposition + female_opposition in
  -- Given the data of male and female participants and their opposition counts,
  -- the most convincing method to demonstrate the relationship is the
  -- Independence test (Chi-Square test of independence).
  true) :=
  sorry

end gender_judgment_independence_test_l561_561101


namespace problem_HMMT_before_HMT_l561_561052
noncomputable def probability_of_sequence (seq: List Char) : ℚ := sorry
def probability_H : ℚ := 1 / 3
def probability_M : ℚ := 1 / 3
def probability_T : ℚ := 1 / 3

theorem problem_HMMT_before_HMT : probability_of_sequence ['H', 'M', 'M', 'T'] = 1 / 4 :=
sorry

end problem_HMMT_before_HMT_l561_561052


namespace cindy_used_15_stickers_l561_561528

variable {S X : ℕ} -- S and X are natural numbers (non-negative integers)

theorem cindy_used_15_stickers (h1 : ∀ S, ∃ X, S + 18 = (S - X) + 33) : X = 15 := by
  have h2 : ∀ S, S + 18 = S - X + 33 := by
    intro S
    exact h1 S
  have h3 : ∀ S, S + 18 = S - X + 33 → 18 = -X + 33 := by
    intro S h
    rw [add_comm S 18, add_comm S (- X + 33)] at h
    linarith
  have h4 : 18 = -X + 33 := h3 S (h2 S)
  have h5 : 18 + X = 33 := by linarith
  exact (nat.add_sub_cancel_left 18 33).symm ▸ h5 ▸ rfl


end cindy_used_15_stickers_l561_561528


namespace probability_palindrome_divisible_by_11_l561_561933

theorem probability_palindrome_divisible_by_11 :
  let palindromes := { n | ∃ (a b : ℕ), a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 101 * a + 10 * b + a},
      palindromes_div_11 := {n ∈ palindromes | n % 11 = 0} in
  (palindromes_div_11.card : ℚ) / (palindromes.card : ℚ) = 1 / 9 :=
sorry

end probability_palindrome_divisible_by_11_l561_561933


namespace lunks_to_apples_l561_561694

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l561_561694


namespace trigonometric_expression_l561_561145

theorem trigonometric_expression (x : ℝ) (h : Real.tan x = -1/2) : 
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 := 
sorry

end trigonometric_expression_l561_561145


namespace remainder_product_191_193_197_mod_23_l561_561889

theorem remainder_product_191_193_197_mod_23 :
  (191 * 193 * 197) % 23 = 14 := by
  sorry

end remainder_product_191_193_197_mod_23_l561_561889


namespace math_problem_l561_561976

theorem math_problem :
  |(-3 : ℝ)| - Real.sqrt 8 - (1/2 : ℝ)⁻¹ + 2 * Real.cos (Real.pi / 4) = 1 - Real.sqrt 2 :=
by
  sorry

end math_problem_l561_561976


namespace exponent_sum_l561_561706

theorem exponent_sum (α β : ℝ) (h : complex.exp (complex.I * α) + complex.exp (complex.I * β) = (1/3 : ℂ) + (1/2 : ℂ) * complex.I) :
  complex.exp (-complex.I * α) + complex.exp (-complex.I * β) = (1/3 : ℂ) - (1/2 : ℂ) * complex.I :=
sorry

end exponent_sum_l561_561706


namespace female_listeners_number_l561_561426

theorem female_listeners_number:
  let males_listen := 75 in
  let males_dont_listen := 85 in
  let females_listen := ?females_listen in
  let females_dont_listen := 135 in
  let undeclared_listen := 20 in
  let undeclared_dont_listen := 15 in
  let total_listen := 160 in
  let total_dont_listen := 235 in
  (males_listen + females_listen + undeclared_listen = total_listen) ∧
  (males_dont_listen + females_dont_listen + undeclared_dont_listen = total_dont_listen) →
  females_listen = 65 := 
by
  sorry

end female_listeners_number_l561_561426


namespace B3_inverse_l561_561264

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

def B_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![(3 : ℝ), -1; 1, 1]

noncomputable def B3_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![(20 : ℝ), -12; 12, -4]

theorem B3_inverse (h : B⁻¹ = B_inv) : (B^3)⁻¹ = B3_inv :=
by
  sorry

end B3_inverse_l561_561264


namespace arrange_books_l561_561500

theorem arrange_books :
  let geometry := 5
  let number_theory := 3
  let basic_algebra := 2
  ∃ arrangements : ℕ, 
    (basic_algebra_adjacent geometry number_theory basic_algebra arrangements
    ∧ arrangements = 1008) := sorry

end arrange_books_l561_561500


namespace number_of_cups_needed_to_fill_container_l561_561927

theorem number_of_cups_needed_to_fill_container (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 640) (h2 : cup_capacity = 120) : 
  (container_capacity + cup_capacity - 1) / cup_capacity = 6 :=
by
  sorry

end number_of_cups_needed_to_fill_container_l561_561927


namespace limit_S1_minus_S2_div_ln_t_l561_561982

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (Real.tan θ, 1 / Real.cos θ)

noncomputable def Q_point (t : ℝ) : ℝ × ℝ :=
  (t, Real.sqrt (t^2 + 1))

noncomputable def S1 (t : ℝ) : ℝ :=
  ∫ x in 0..t, Real.sqrt (x^2 + 1)

noncomputable def S2 (t : ℝ) : ℝ :=
  (1 / 2) * t * Real.sqrt (t^2 + 1)

theorem limit_S1_minus_S2_div_ln_t (t : ℝ) (ht : t > 0) :
  tendsto (λ t, (S1 t - S2 t) / Real.log t) at_top (𝓝 (1 / 2)) :=
  sorry

end limit_S1_minus_S2_div_ln_t_l561_561982


namespace max_colored_cells_on_cube_l561_561102

theorem max_colored_cells_on_cube (n : ℕ) (h : n = 1000) :
  let total_cells := 6 * n^2 in
  let max_colored := total_cells - 2 * n in
  max_colored = 2998000 :=
by
  sorry

end max_colored_cells_on_cube_l561_561102


namespace max_product_of_roots_l561_561602

noncomputable def max_prod_roots_m : ℝ :=
  let m := 4.5
  m

theorem max_product_of_roots (m : ℕ) (h : 36 - 8 * m ≥ 0) : m = max_prod_roots_m :=
  sorry

end max_product_of_roots_l561_561602


namespace isosceles_triangle_distance_AB_l561_561736

noncomputable def distance_from_C_to_AB (A B C : ℝ) : ℝ :=
  let AC := 37
  let exterior_angle_B := 60
  -- distance from vertex C to line AB
  18.5

theorem isosceles_triangle_distance_AB {A B C : ℝ} :
  isosceles_triangle A B C ∧ AC = 37 ∧ exterior_angle_B = 60 → 
  distance_from_C_to_AB A B C = 18.5 := 
begin
  sorry
end

end isosceles_triangle_distance_AB_l561_561736


namespace log_arithmetic_sequence_l561_561169

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (a : ℕ → ℝ)
variable (h_arithmetic : is_arithmetic_sequence a)
variable (h_condition : a 2 + a 8 + a 14 = 3)

-- Statement to prove
theorem log_arithmetic_sequence : log 2 (a 3 + a 13) = 1 :=
by 
  -- Proof is omitted
  sorry

end log_arithmetic_sequence_l561_561169


namespace stratified_sampling_third_year_students_l561_561469

/-- 
A university's mathematics department has a total of 5000 undergraduate students, 
with the first, second, third, and fourth years having a ratio of their numbers as 4:3:2:1. 
If stratified sampling is employed to select a sample of 200 students from all undergraduates,
prove that the number of third-year students to be sampled is 40.
-/
theorem stratified_sampling_third_year_students :
  let total_students := 5000
  let ratio_first_second_third_fourth := (4, 3, 2, 1)
  let sample_size := 200
  let third_year_ratio := 2
  let total_ratio_units := 4 + 3 + 2 + 1
  let proportion_third_year := third_year_ratio / total_ratio_units
  let expected_third_year_students := sample_size * proportion_third_year
  expected_third_year_students = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l561_561469


namespace intersection_unique_element_l561_561788

noncomputable def A := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def B (r : ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem intersection_unique_element (r : ℝ) (hr : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ A ∧ p ∈ B r) → (r = 3 ∨ r = 7) :=
sorry

end intersection_unique_element_l561_561788


namespace value_of_a9_maximum_value_of_Sn_l561_561298

-- Define the arithmetic sequence with initial conditions and common difference
variables {a : ℕ → ℤ} {S : ℕ → ℤ}
variable d : ℤ
variable a₁ : ℤ
variable n : ℕ

-- Define the given conditions
def arithmetic_sequence : Prop := 
  (a 2 = a₁ + d) ∧
  (a 2 = 8) ∧
  (a 4 = a₁ + 3 * d) ∧ 
  (a 4 = 4)

-- Prove the value of a_9
theorem value_of_a9 (h : arithmetic_sequence) : a 9 = -6 :=
sorry

-- Define the sum of the first n terms
def sum_of_first_n_terms : Prop :=
  ∀ n : ℕ, S n = (n * (2 * a₁ + (n - 1) * d)) / 2

-- Prove the maximum value of S_n
theorem maximum_value_of_Sn (h : arithmetic_sequence) (hSum : sum_of_first_n_terms) : 
  ∃ (n : ℕ), (n = 5 ∨ n = 6) ∧ S n = 30 :=
sorry

end value_of_a9_maximum_value_of_Sn_l561_561298


namespace diamonds_in_F10_l561_561496

def diamonds_in_figure (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (Nat.add (Nat.mul (n - 1) n) 0) / 2

theorem diamonds_in_F10 : diamonds_in_figure 10 = 136 :=
by
  sorry

end diamonds_in_F10_l561_561496


namespace max_GREECE_val_l561_561540

variables (V E R I A G C : ℕ)
noncomputable def verify : Prop :=
  (V * 100 + E * 10 + R - (I * 10 + A)) = G^(R^E) * (G * 100 + R * 10 + E + E * 100 + C * 10 + E) ∧
  G ≠ 0 ∧ E ≠ 0 ∧ V ≠ 0 ∧ I ≠ 0 ∧
  V ≠ E ∧ V ≠ R ∧ V ≠ I ∧ V ≠ A ∧ V ≠ G ∧ V ≠ C ∧
  E ≠ R ∧ E ≠ I ∧ E ≠ A ∧ E ≠ G ∧ E ≠ C ∧
  R ≠ I ∧ R ≠ A ∧ R ≠ G ∧ R ≠ C ∧
  I ≠ A ∧ I ≠ G ∧ I ≠ C ∧
  A ≠ G ∧ A ≠ C ∧
  G ≠ C

theorem max_GREECE_val : ∃ V E R I A G C : ℕ, verify V E R I A G C ∧ (G * 100000 + R * 10000 + E * 1000 + E * 100 + C * 10 + E = 196646) :=
sorry

end max_GREECE_val_l561_561540


namespace side_length_of_square_base_l561_561311

theorem side_length_of_square_base :
  ∃ s : ℝ, 
  let h := 8 in 
  let D := 2700 in 
  let W := 86400 in
  V = W / D ∧ V = s^2 * h ∧ s = 2 :=
sorry

end side_length_of_square_base_l561_561311


namespace work_fraction_completed_after_first_phase_l561_561042

-- Definitions based on conditions
def total_work := 1 -- Assume total work as 1 unit
def initial_days := 100
def initial_people := 10
def first_phase_days := 20
def fired_people := 2
def remaining_days := 75
def remaining_people := initial_people - fired_people

-- Hypothesis about the rate of work initially and after firing people
def initial_rate := total_work / initial_days
def first_phase_work := first_phase_days * initial_rate
def remaining_work := total_work - first_phase_work
def remaining_rate := remaining_work / remaining_days

-- Proof problem statement: 
theorem work_fraction_completed_after_first_phase :
  (first_phase_work / total_work) = (15 / 64) :=
by
  -- This is the place where the actual formal proof should be written.
  sorry

end work_fraction_completed_after_first_phase_l561_561042


namespace perimeter_of_arrangement_l561_561830

-- Define the conditions given in the problem
def area_of_figure : ℝ := 225 -- Total area in cm^2
def number_of_squares : ℕ := 6 -- Number of identical squares

-- Define the question as the statement to prove
theorem perimeter_of_arrangement : 
  let area_of_square := area_of_figure / number_of_squares,
      side_length := real.sqrt area_of_square,
      perimeter := 8 * side_length
  in (225 / 6 ≠ 0) → -- Ensure division is valid and meaningful
     (50 - 1 < perimeter) ∧ (perimeter < 50 + 1) := 
by
  sorry

end perimeter_of_arrangement_l561_561830


namespace TJs_average_time_l561_561822

theorem TJs_average_time 
  (total_distance : ℝ) 
  (distance_half : ℝ)
  (time_first_half : ℝ) 
  (time_second_half : ℝ) 
  (H1 : total_distance = 10) 
  (H2 : distance_half = total_distance / 2) 
  (H3 : time_first_half = 20) 
  (H4 : time_second_half = 30) :
  (time_first_half + time_second_half) / total_distance = 5 :=
by
  sorry

end TJs_average_time_l561_561822


namespace equidistant_point_on_x_axis_l561_561886

theorem equidistant_point_on_x_axis (x : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 0)) (hB : B = (3, 5)) :
  (Real.sqrt ((x - (-3))^2)) = (Real.sqrt ((x - 3)^2 + 25)) →
  x = 25 / 12 := 
by 
  sorry

end equidistant_point_on_x_axis_l561_561886


namespace area_of_larger_square_is_16_times_l561_561860

-- Define the problem conditions
def perimeter_condition (a b : ℝ) : Prop :=
  4 * a = 4 * 4 * b

-- Define the relationship between the areas of the squares given the side lengths
def area_ratio (a b : ℝ) : ℝ :=
  (a * a) / (b * b)

theorem area_of_larger_square_is_16_times (a b : ℝ) (h : perimeter_condition a b) : area_ratio a b = 16 :=
by 
  unfold perimeter_condition at h
  rw [mul_assoc, mul_comm 4 b] at h
  have ha : a = 4 * b := (mul_right_inj' (ne_of_gt (show 0 < (4:ℝ), by norm_num))).mp h
  unfold area_ratio
  rw [ha, mul_pow, pow_two, mul_pow]
  exact (by norm_num : (4:ℝ)^2 = 16)

end area_of_larger_square_is_16_times_l561_561860


namespace average_speed_correct_l561_561489

-- Definitions for the distances
def distance_PQ : ℝ := 120
def distance_QR : ℝ := 150
def distance_RP : ℝ := 180

-- Definitions for the speeds
def speed_bicycle : ℝ := 30
def speed_motorbike_base : ℝ := 30
def speed_motorbike_uphill : ℝ := speed_motorbike_base * 1.3 * 0.85
def speed_motorbike_downhill : ℝ := speed_motorbike_base * 1.25

-- Definitions for the layover times
def layover_Q : ℝ := 1
def layover_R : ℝ := 2

-- Calculations for the times on each leg of the journey
def time_PQ : ℝ := distance_PQ / speed_bicycle
def time_QR : ℝ := distance_QR / speed_motorbike_uphill
def time_RP : ℝ := distance_RP / speed_motorbike_downhill

-- Total distance and total time
def total_distance : ℝ := distance_PQ + distance_QR + distance_RP
def total_time : ℝ := time_PQ + layover_Q + time_QR + layover_R + time_RP

-- Average speed calculation
def average_speed : ℝ := total_distance / total_time

-- The proof statement
theorem average_speed_correct : abs (average_speed - 27.57) < 0.01 := by
  sorry

end average_speed_correct_l561_561489


namespace line_not_in_second_quadrant_l561_561838

variable (a : ℝ)

def line_eq (a : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y, (a - 2) * y = (3 * a - 1) * x - 1

theorem line_not_in_second_quadrant (a : ℝ) :
  (∀ x y, x ≤ 0 ∧ y ≥ 0 → ¬line_eq a x y) ↔ a ≥ 2 := by
  sorry

end line_not_in_second_quadrant_l561_561838


namespace original_angle_measure_l561_561388

-- Definition of the problem conditions
def original_angle (x : ℝ) : Prop :=
  let complement := 5 * x + 7 in
  x + complement = 90

-- Statement of the theorem: The measure of the original angle is 13.833 degrees
theorem original_angle_measure : ∃ x : ℝ, original_angle x ∧ x = 13.833 :=
by
  sorry

end original_angle_measure_l561_561388


namespace height_percentage_difference_l561_561060

theorem height_percentage_difference (A B : ℝ) (h : B = A * (4/3)) : 
  (A * (1/3) / B) * 100 = 25 := by
  sorry

end height_percentage_difference_l561_561060


namespace cone_cosine_l561_561404

variable (l r : ℝ)

def cone_angle_unfolded : ℝ := (4 / 3) * Real.pi
def cone_slant_height_base_cos (h : l > 0) : ℝ := r / l

theorem cone_cosine (h : l > 0) :
  (θ : ℝ) = (2 * Real.pi * r / l) = cone_angle_unfolded →
  r = (2 / 3) * l →
  cone_slant_height_base_cos l r h = (2 / 3) :=
by
  intros h₁ h₂
  simp [cone_slant_height_base_cos, cone_angle_unfolded]
  sorry

end cone_cosine_l561_561404


namespace distance_in_scientific_notation_l561_561392

def scientific_notation_distance (d : ℕ) : Prop :=
  d = 150000000

theorem distance_in_scientific_notation (n : ℕ) :
  scientific_notation_distance (1.5 * 10 ^ n) ↔ n = 8 :=
sorry

end distance_in_scientific_notation_l561_561392


namespace g_extreme_value_f_ge_g_l561_561647

noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1) - 2 / x + 1
noncomputable def g (x : ℝ) : ℝ := Real.log x / x + 2

theorem g_extreme_value :
  ∃ (x : ℝ), x = Real.exp 1 ∧ g x = 1 / Real.exp 1 + 2 :=
by sorry

theorem f_ge_g (x : ℝ) (hx : 0 < x) : f x >= g x :=
by sorry

end g_extreme_value_f_ge_g_l561_561647


namespace cost_price_per_meter_l561_561901

theorem cost_price_per_meter (total_meters selling_price loss_per_meter : ℕ) : 
  (total_meters = 600) → (selling_price = 36000) → (loss_per_meter = 10) → 
  let total_cost_price := selling_price + (loss_per_meter * total_meters)
  in (total_cost_price / total_meters) = 70 :=
by
  intros h1 h2 h3
  sorry

end cost_price_per_meter_l561_561901


namespace initial_sugar_weight_l561_561411

def packs := 35
def weight_per_pack := 400
def remaining_sugar := 100
def sold_packs := packs / 2
def remaining_packs := packs - sold_packs
def weight_remaining_packs := remaining_packs * weight_per_pack
def used_sugar := 0.4 * weight_remaining_packs

theorem initial_sugar_weight :
  packs * weight_per_pack + remaining_sugar = 14100 := 
sorry

end initial_sugar_weight_l561_561411


namespace face_perpendicular_sin_alpha_range_l561_561963

variable (AB BC CD : ℝ) (x : ℝ)

-- Condition Definitions
def perpendicular (v1 v2 : ℝ) : Prop := v1 * v2 = 0
def distance (v : ℝ) := v

-- Given conditions
axiom AB_perp_BC : perpendicular AB BC
axiom BC_perp_CD : perpendicular BC CD
axiom CD_perp_AB : perpendicular CD AB
axiom AB_eq_one : AB = 1
axiom BC_eq_one : BC = 1
axiom CD_eq_x : CD = x

-- Equivalent problem requiring proof of the statements
theorem face_perpendicular : perpendicular AB CD := by sorry

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.sqrt 2)) * Real.sqrt (1 + 1 / (1 + x^2))

theorem sin_alpha_range (x : ℝ) : 
  1 / (Real.sqrt 2) * Real.sqrt (1 + 1 / (1 + x^2)) ∈ (Real.sqrt(2) / 2, 1) := by sorry


end face_perpendicular_sin_alpha_range_l561_561963


namespace monkey_giraffe_difference_l561_561979

def zebras : ℕ := 12
def camels : ℕ := zebras / 2
def monkeys : ℕ := 4 * camels
def parrots : ℕ := (monkeys - 5) + Int.toNat (0.5 * (monkeys - 5))
def adult_giraffes : ℕ := 3 * parrots + 1
def baby_giraffes : ℕ := Int.toNat (0.25 * adult_giraffes)
def total_giraffes : ℕ := adult_giraffes + baby_giraffes
def difference : ℤ := monkeys - total_giraffes

theorem monkey_giraffe_difference :
  difference = -86 := by
  sorry

end monkey_giraffe_difference_l561_561979


namespace find_y_of_arithmetic_mean_l561_561656

theorem find_y_of_arithmetic_mean (y : ℝ) (h: (7 + 12 + 19 + 8 + 10 + y) / 6 = 15) : y = 34 :=
by {
  -- Skipping the proof
  sorry
}

end find_y_of_arithmetic_mean_l561_561656


namespace correct_sum_l561_561897

theorem correct_sum (initial_sum : ℕ) (excess_units : ℕ) (deficit_tens : ℕ) (erroneous_sum : ℕ) :
  erroneous_sum = initial_sum + excess_units - deficit_tens * 10 →  
  initial_sum = 1990 := 
by 
  intros h 
  have excess_units : excess_units = 6 := rfl 
  have deficit_tens : deficit_tens = 5 := rfl 
  have erroneous_sum : erroneous_sum = 1946 := rfl 
  rw [excess_units, deficit_tens, erroneous_sum] at h 
  rw add_comm at h 
  rw [add_comm (50 - 6)] at h 
  norm_num at h 
  exact h

end correct_sum_l561_561897


namespace sum_of_ages_l561_561922

theorem sum_of_ages (b f : ℕ) (h_b_range : 13 ≤ b ∧ b ≤ 19)
 (h_four_digit_num : (100 * f + b) - |f - b| = 4289) :
  f + b = 59 :=
sorry

end sum_of_ages_l561_561922


namespace lawrence_work_hours_l561_561317

theorem lawrence_work_hours :
  let hours_mon := 8
  let hours_tue := 8
  let hours_fri := 8
  let hours_wed := 5.5
  let hours_thu := 5.5
  let total_hours := hours_mon * 1 + hours_tue * 1 + hours_fri * 1 + hours_wed * 1 + hours_thu * 1
  total_hours / 7 = 5 :=
by {
  let hours_mon := 8
  let hours_tue := 8
  let hours_fri := 8
  let hours_wed := 5.5
  let hours_thu := 5.5
  let total_hours := hours_mon * 1 + hours_tue * 1 + hours_fri * 1 + hours_wed * 1 + hours_thu * 1
  have total_hours_eq : total_hours = 35, by norm_num,
  rw total_hours_eq,
  norm_num,
}

end lawrence_work_hours_l561_561317


namespace number_of_integers_between_sqrt8_and_sqrt72_l561_561217

theorem number_of_integers_between_sqrt8_and_sqrt72 : 
  let a := Int.ceil (Real.sqrt 8)
  let b := Int.floor (Real.sqrt 72)
  b - a + 1 = 6 :=
begin
  sorry
end

end number_of_integers_between_sqrt8_and_sqrt72_l561_561217


namespace points_lie_on_line_l561_561619

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  x + y = 4 :=
by
  let x := (2 * t + 2) / t
  let y := (2 * t - 2) / t
  sorry

end points_lie_on_line_l561_561619


namespace quadratic_roots_p_value_l561_561129

-- Given conditions as definitions
variables (A B C r s p q: ℝ)
variable (h_roots_of_original : Polynomial.root (polynomial.mk [C, B, A]) r)
variable (k_roots_of_original : Polynomial.root (polynomial.mk [C, B, A]) s)
variable (h_r_s_sum : r + s = -B / A)
variable (h_quad_new : r * s = C / A)
variable (h_roots_of_new : Polynomial.root (polynomial.mk [q, p, 1]) (r + 3))
variable (k_roots_of_new : Polynomial.root (polynomial.mk [q, p, 1]) (s + 3))

-- Prove the statement
theorem quadratic_roots_p_value :
  p = B / A - 6 :=
sorry

end quadratic_roots_p_value_l561_561129


namespace cylinder_tin_diameter_l561_561390

theorem cylinder_tin_diameter (V h : ℝ) (pi_approx : ℝ) (h_pos : 0 < h) (V_pos : 0 < V) :
  (pi_approx = Real.pi) → (h = 2) → (V = 98) → let r := real.sqrt (V / (pi_approx * h)) in
  2 * r = 7.9 :=
by
  intros
  sorry

end cylinder_tin_diameter_l561_561390


namespace min_value_l561_561121

noncomputable def conditions (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧ 
  (27^x + y^4 - 3^x - 1 = 0)

theorem min_value (x y : ℝ) (h : conditions x y) : ∃ x y, (x^3 + y^3 = -1) :=
sorry

end min_value_l561_561121


namespace pqrsquared_l561_561531

-- Define p, q, and r as the roots of the given polynomial
noncomputable def pqr_are_roots (f : ℝ → ℝ) : Prop :=
  ∃ p q r : ℝ, f = 3 * polynomial.X ^ 3 - 2 * polynomial.X ^ 2 + 5 * polynomial.X + 15 ∧ 
  polynomial.aeval p f = 0 ∧ polynomial.aeval q f = 0 ∧ polynomial.aeval r f = 0

-- Define the main statement we want to prove
theorem pqrsquared (p q r : ℝ) (h : pqr_are_roots (λ x => 3 * x^3 - 2 * x^2 + 5 * x + 15)) :
  p^2 + q^2 + r^2 = - 26 / 9 := 
sorry

end pqrsquared_l561_561531


namespace problem_part1_problem_part2_problem_part3_l561_561174

-- Define the context for the problem
variables {x : ℝ} (n : ℕ)

-- Define the known property of the problem
noncomputable def binomial_ratio_condition (n : ℕ) : Prop :=
  (∑ i in finset.range(n + 1), ((-2) ^ i) * (nat.choose n i) : ℝ)

-- The first proof statement
theorem problem_part1 (h: (nat.choose (n - 3) 3) / (nat.choose (n - 2) 2) = 8 / 3)
  : n = 10 :=
sorry

-- The second proof statement
theorem problem_part2 (h : n = 10) 
  : (-2)^2 * (nat.choose 10 2) = 180 :=
sorry

-- The third proof statement
theorem problem_part3 
  : (∑ i in finset.range(11), ((-2) ^ i) * (nat.choose 10 i)) = 1 :=
sorry

end problem_part1_problem_part2_problem_part3_l561_561174


namespace problem_statement_l561_561652

variables (α : Type*) [Plane α] (l m n : Line α)

def perp (l n : Line α) : Prop := Perpendicular l n
def parallel (l : Line α) (α : Plane α) : Prop := Parallel l α

theorem problem_statement (h1 : perp l α) (h2 : parallel m α) : perp l m :=
sorry

end problem_statement_l561_561652


namespace solve_mod_equation_l561_561997

theorem solve_mod_equation (y b n : ℤ) (h1 : 15 * y + 4 ≡ 7 [ZMOD 18]) (h2 : y ≡ b [ZMOD n]) (h3 : 2 ≤ n) (h4 : b < n) : b + n = 11 :=
sorry

end solve_mod_equation_l561_561997


namespace base8_to_base10_l561_561537

theorem base8_to_base10 :
  ∀ (n : ℕ), (n = 2 * 8^2 + 4 * 8^1 + 3 * 8^0) → (n = 163) :=
by
  intros n hn
  sorry

end base8_to_base10_l561_561537


namespace english_only_students_l561_561453

theorem english_only_students (T B G_total : ℕ) (hT : T = 40) (hB : B = 12) (hG_total : G_total = 22) :
  (T - (G_total - B) - B) = 18 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end english_only_students_l561_561453


namespace integer_count_between_sqrt8_and_sqrt72_l561_561222

theorem integer_count_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ( ∀ x : ℤ, (⌊Real.sqrt 8⌋.to_nat + 1) ≤ x ∧ x ≤ ⌊Real.sqrt 72⌋ - 1 → x = 6 ) :=
by 
  -- Define the floor and ceiling functions
  have sqrt_8_ceil : ⌈Real.sqrt 8⌉ = 3 := sorry
  have sqrt_72_floor : ⌊Real.sqrt 72⌋ = 8 := sorry

  use 6,
  split,
  { refl },
  { intros x hx,
    sorry
  }

-- Additional supporting facts
lemma sqrt_8_approx : Real.sqrt 8 ≈ 2.83 := sorry
lemma sqrt_72_approx : Real.sqrt 72 ≈ 8.49 := sorry

end integer_count_between_sqrt8_and_sqrt72_l561_561222


namespace lunks_needed_for_24_apples_l561_561700

-- Define the conditions as Lean definitions
def lunks_per_kunks := 7 / 4
def kunks_per_apples := 3 / 5
def apples_needed := 24

-- State the theorem
theorem lunks_needed_for_24_apples : 
  let k := (3 * apples_needed) / 5 in 
  let rounded_k := k.ceil in 
  let l := (7 * rounded_k) / 4 in 
  l.ceil = 27 :=
by 
  let k := (3 * apples_needed) / 5
  let rounded_k := k.ceil
  let l := (7 * rounded_k) / 4
  have h1 : k = (3 * apples_needed) / 5 := rfl
  have h2 : rounded_k = k.ceil := rfl
  have h3 : l = (7 * rounded_k) / 4 := rfl
  show l.ceil = 27, from sorry

end lunks_needed_for_24_apples_l561_561700


namespace rectangular_container_volume_l561_561444

theorem rectangular_container_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 :=
by
  sorry

end rectangular_container_volume_l561_561444


namespace log_sum_of_geometric_sequence_l561_561279

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r^n

theorem log_sum_of_geometric_sequence :
  ∀ (a r : ℝ),
  (∀ n, geometric_sequence a r n > 0) →
  geometric_sequence a r 8 * geometric_sequence a r 13 + geometric_sequence a r 9 * geometric_sequence a r 12 = 2^6 →
  (∑ n in finset.range 20, real.logb 2 (geometric_sequence a r n)) = 50 :=
by
  sorry

end log_sum_of_geometric_sequence_l561_561279


namespace complex_proof_problem_l561_561604

theorem complex_proof_problem (i : ℂ) (h1 : i^2 = -1) :
  (i^2 + i^3 + i^4) / (1 - i) = (1 / 2) - (1 / 2) * i :=
by
  -- Proof will be provided here
  sorry

end complex_proof_problem_l561_561604


namespace log_equation_solution_l561_561603

theorem log_equation_solution (x : ℝ) : log 8 (x + 8) = 3 ↔ x = 504 :=
by
  sorry

end log_equation_solution_l561_561603


namespace words_per_page_l561_561032

theorem words_per_page (p : ℕ) (hp : p ≤ 120) (h : 150 * p ≡ 210 [MOD 221]) : p = 98 := by
  sorry

end words_per_page_l561_561032


namespace find_f_and_g_minimum_l561_561643

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

def g (f : ℝ → ℝ) (m x : ℝ) : ℝ := f(x) - 2 * m * x + 2

theorem find_f_and_g_minimum (a b c m : ℝ) :
  (∀ x : ℝ, f x = a * x^2 + b * x + c) →
  f 0 = 0 →
  (∀ x : ℝ, f (x + 2) - f x = 4 * x) →
  f = (λ x, x^2 - 2 * x) ∧
  (∀ x : ℝ, x ≥ 1 → g f m x = 
    if m ≤ 0 then 1 - 2 * m 
    else -m^2 - 2 * m + 1) :=
by
  sorry

end find_f_and_g_minimum_l561_561643


namespace area_OMVK_l561_561980

def AreaOfQuadrilateral (S_OKSL S_ONAM S_OMVK : ℝ) : ℝ :=
  let S_ABCD := 4 * (S_OKSL + S_ONAM)
  S_ABCD - S_OKSL - 24 - S_ONAM

theorem area_OMVK {S_OKSL S_ONAM : ℝ} (h_OKSL : S_OKSL = 6) (h_ONAM : S_ONAM = 12) : 
  AreaOfQuadrilateral S_OKSL S_ONAM 30 = 30 :=
by
  sorry

end area_OMVK_l561_561980


namespace largest_prime_factor_4752_l561_561000

theorem largest_prime_factor_4752 : ∃ p : ℕ, p = 11 ∧ prime p ∧ (∀ q : ℕ, prime q ∧ q ∣ 4752 → q ≤ 11) :=
by
  sorry

end largest_prime_factor_4752_l561_561000


namespace solve_expression_l561_561930

def ratio_value (x y z : ℝ) (h1 : x < y) (h2 : y < z) (h3 : x / z = z / (x + y + z)) : ℝ :=
let Q := x / z in Q^(Q^(Q^2 + Q⁻¹) + Q⁻¹) + Q⁻¹

theorem solve_expression (x y z : ℝ) (h1 : x < y) (h2 : y < z) (h3 : x / z = z / (x + y + z)) :
  ratio_value x y z h1 h2 h3 = Real.sqrt 2 :=
sorry

end solve_expression_l561_561930


namespace find_z_l561_561130

variables (z : ℝ)
variables (v : ℝ × ℝ) (w : ℝ × ℝ)
variables (proj_w_v : ℝ × ℝ)

-- Defining context where v, w, and their projection are given.
def given_conditions : Prop :=
  v = (2, z) ∧
  w = (8, 4) ∧
  proj_w_v = ( -12, -6)

-- The theorem to prove
theorem find_z (h : given_conditions z v w proj_w_v) : z = -34 :=
begin
  sorry -- Proof omitted
end

end find_z_l561_561130


namespace president_vice_committee_count_l561_561739

open Nat

noncomputable def choose_president_vice_committee (total_people : ℕ) : ℕ :=
  let choose : ℕ := 56 -- binomial 8 3 is 56
  total_people * (total_people - 1) * choose

theorem president_vice_committee_count :
  choose_president_vice_committee 10 = 5040 :=
by
  sorry

end president_vice_committee_count_l561_561739


namespace part_I_part_II_l561_561781

-- Definitions for part (I)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2
def condition (m : ℝ) : Prop := ∀ x : ℝ, f(m)(x) + 2 ≥ 0

-- Part (I): Prove the range of m
theorem part_I (m : ℝ) : condition m ↔ m ≥ 1/3 := sorry

-- Definitions for part (II)
def condition_two (m : ℝ) : Prop := m < 0
def solution_set (m : ℝ) (x : ℝ) : Set ℝ :=
  if m ≤ -1 then {y : ℝ | y < -1/m ∨ 1 < y}
  else if -1 < m ∧ m < 0 then {y : ℝ | y < 1 ∨ -1/m < y}
  else ∅

-- Part (II): Prove the inequality solution sets
theorem part_II (m : ℝ) (hc : condition_two m) : ∀ x, f(m)(x) < m - 1 ↔ (x ∈ solution_set m x) := sorry

end part_I_part_II_l561_561781


namespace max_value_q_l561_561343

open Nat

theorem max_value_q (X Y Z : ℕ) (h : 2 * X + 3 * Y + Z = 18) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 24 :=
sorry

end max_value_q_l561_561343


namespace distance_A_to_line_l561_561831

def distance_point_to_line (x1 y1 a b c : ℝ) : ℝ :=
  abs (a * x1 + b * y1 + c) / sqrt (a^2 + b^2)

theorem distance_A_to_line : distance_point_to_line 3 2 1 1 3 = 4 * sqrt 2 :=
by
  sorry

end distance_A_to_line_l561_561831


namespace choose_president_vp_and_committee_l561_561740

theorem choose_president_vp_and_committee :
  ∃ (n : ℕ) (k : ℕ), n = 10 ∧ k = 3 ∧ 
  let ways_to_choose_president := 10 in
  let ways_to_choose_vp := 9 in
  let ways_to_choose_committee := (Nat.choose 8 3) in
  ways_to_choose_president * ways_to_choose_vp * ways_to_choose_committee = 5040 :=
begin
  use [10, 3],
  simp [Nat.choose],
  sorry
end

end choose_president_vp_and_committee_l561_561740


namespace polynomial_divisible_by_three_factors_l561_561366

def P (x : ℝ) : ℝ := (x + 1) ^ 6 - x ^ 6 - 2 * x - 1

theorem polynomial_divisible_by_three_factors :
  (P 0 = 0) ∧ (P (-1) = 0) ∧ (P (-1/2) = 0) → 
  polynomial_divisible_by (P, x * (x + 1) * (2 * x + 1)) :=
by
  sorry

end polynomial_divisible_by_three_factors_l561_561366


namespace largest_number_l561_561425

theorem largest_number (a b c : ℝ) (h1 : a + b + c = 67) (h2 : c - b = 7) (h3 : b - a = 5) : c = 86 / 3 := 
by sorry

end largest_number_l561_561425


namespace square_area_rational_l561_561721

-- Define the condition: the side length of the square is a rational number.
def is_rational (x : ℚ) : Prop := true

-- Define the theorem to be proved: If the side length of a square is rational, then its area is rational.
theorem square_area_rational (s : ℚ) (h : is_rational s) : is_rational (s * s) := 
sorry

end square_area_rational_l561_561721


namespace sum_problem3_equals_50_l561_561978

-- Assume problem3_condition is a placeholder for the actual conditions described in problem 3
-- and sum_problem3 is a placeholder for the sum of elements described in problem 3.

axiom problem3_condition : Prop
axiom sum_problem3 : ℕ

theorem sum_problem3_equals_50 (h : problem3_condition) : sum_problem3 = 50 :=
sorry

end sum_problem3_equals_50_l561_561978


namespace number_of_divisors_2310_l561_561588

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l561_561588


namespace number_of_5digit_even_divisible_by_5_l561_561203

-- Definitions for the problem conditions
def is_even_digit (d : ℕ) : Prop := d ∈ {0, 2, 4, 6, 8}
def is_5digit_even_divisible_by_5 (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    2 ≤ a ∧ a ≤ 8 ∧ a % 2 = 0 ∧  -- ten-thousands place
    is_even_digit b ∧             -- thousands place
    is_even_digit c ∧             -- hundreds place
    is_even_digit d ∧             -- tens place
    e = 0                         -- units place

theorem number_of_5digit_even_divisible_by_5 :
  (nat.card {n // is_5digit_even_divisible_by_5 n} = 500) :=
sorry

end number_of_5digit_even_divisible_by_5_l561_561203


namespace sin_alpha_l561_561350

def f (x : ℝ) : ℝ := sqrt 3 + (sin x / (1 + cos x))
def zeros : ℕ → ℝ := λ n, 2 * n * Real.pi + (4 * Real.pi / 3) -- Ascending positive zeros

noncomputable def α : ℝ := 12 * zeros 3 + 201

theorem sin_alpha (x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 : ℝ) :
    f x1 = 0 →
    f x2 = 0 →
    f x3 = 0 →
    f x4 = 0 →
    f x5 = 0 →
    f x6 = 0 →
    f x7 = 0 →
    f x8 = 0 →
    f x9 = 0 →
    f x10 = 0 →
    f x11 = 0 →
    f x12 = 0 →
    f x13 = 0 →
    f x14 = 0 →
    f x15 = 0 →
    f x16 = 0 →
    f x17 = 0 →
    f x18 = 0 →
    f x19 = 0 →
    f x20 = 0 →
    sin α = -sqrt (3) / 2 :=
sorry

end sin_alpha_l561_561350


namespace sum_of_fourth_powers_eq_6nR4_l561_561939

noncomputable theory

-- Define a regular n-gon inscribed in a circle of radius R
structure RegularNGon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (radius : ℝ)
  (is_regular : ∀ i j, (dist (vertices i) (0,0) = radius) ∧ 
                       (dist (vertices i) (vertices j) = 
                        dist (vertices ((i + 1) % n)) (vertices ((j + 1) % n))))

-- Define the problem conditions
variables {n : ℕ} (R : ℝ) (polygon : RegularNGon n) (X : ℝ × ℝ)

-- Define the problem to prove
theorem sum_of_fourth_powers_eq_6nR4 :
  let XAi := λ (Ai : ℝ × ℝ), dist X Ai in
  (∑ i, (XAi (polygon.vertices i)) ^ 4) = 6 * n * R ^ 4 :=
sorry

end sum_of_fourth_powers_eq_6nR4_l561_561939


namespace acute_triangle_orthocenter_l561_561760

variables (A B C H : Point) (a b c h_a h_b h_c : Real)

def acute_triangle (α β γ : Point) : Prop := 
-- Definition that ensures triangle αβγ is acute
sorry

def orthocenter (α β γ ω : Point) : Prop := 
-- Definition that ω is the orthocenter of triangle αβγ 
sorry

def sides_of_triangle (α β γ : Point) : (Real × Real × Real) := 
-- Function that returns the side lengths of triangle αβγ as (a, b, c)
sorry

def altitudes_of_triangle (α β γ θ : Point) : (Real × Real × Real) := 
-- Function that returns the altitudes of triangle αβγ with orthocenter θ as (h_a, h_b, h_c)
sorry

theorem acute_triangle_orthocenter 
  (A B C H : Point)
  (a b c h_a h_b h_c : Real)
  (ht : acute_triangle A B C)
  (orth : orthocenter A B C H)
  (sides : sides_of_triangle A B C = (a, b, c))
  (alts : altitudes_of_triangle A B C H = (h_a, h_b, h_c)) :
  AH * h_a + BH * h_b + CH * h_c = (a^2 + b^2 + c^2) / 2 :=
by sorry


end acute_triangle_orthocenter_l561_561760


namespace range_of_m_l561_561278

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x < 3) ↔ (x / 3 < 1 - (x - 3) / 6 ∧ x < m)) → m ≥ 3 :=
by
  sorry

end range_of_m_l561_561278


namespace problem1_problem2_l561_561461

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 3 * a^3 + 2 * b^3 ≥ 3 * a^2 * b + 2 * a * b^2 := 
by
  sorry

theorem problem2 (a b : ℝ) (h1 : abs a < 1) (h2 : abs b < 1) : abs (1 - a * b) > abs (a - b) := 
by
  sorry

end problem1_problem2_l561_561461


namespace problem_M_value_l561_561784

-- Define the set T
def T : Finset ℝ := {3^0, 3^1, 3^2, 3^3, 3^4, 3^5, 3^6, 3^7}

-- Define the function M that calculates the sum of products of all pairs
def M : ℝ :=
  ∑ x in T, ∑ y in T, if x ≠ y then x * y else 0

-- Claim to prove
theorem problem_M_value : M = 25079720 := by
  sorry

end problem_M_value_l561_561784


namespace minimum_value_m_n_l561_561629

variable {a b : ℝ}

def arithmetic_mean_condition : Prop := a > 0 ∧ b > 0 ∧ (a + b = 1)
def m : ℝ := a + a⁻¹
def n : ℝ := b + b⁻¹

theorem minimum_value_m_n (h : arithmetic_mean_condition) : m + n = 5 :=
sorry

end minimum_value_m_n_l561_561629


namespace divisors_360_l561_561063

def counts_divisors_360 : Prop :=
  ∃ d : ℕ, (d = 24 ∧ (∀ n : ℕ, n ∣ 360 ↔ n ∈ finset.range 361 ∧ finset.card (finset.filter (λ x, x ∣ 360) (finset.range 361)) = 24))

theorem divisors_360 : counts_divisors_360 :=
  sorry

end divisors_360_l561_561063


namespace sphere_radius_is_16_25_l561_561945

def sphere_in_cylinder_radius (r : ℝ) : Prop := 
  ∃ (x : ℝ), (x ^ 2 + 15 ^ 2 = r ^ 2) ∧ ((x + 10) ^ 2 = r ^ 2) ∧ (r = 16.25)

theorem sphere_radius_is_16_25 : 
  sphere_in_cylinder_radius 16.25 :=
sorry

end sphere_radius_is_16_25_l561_561945


namespace final_person_is_Nicky_l561_561315

def last_person_standing : String :=
  let players := ["Laura", "Mike", "Nicky", "Olivia"]
  -- Define a function to check if a number should cause elimination
  let is_eliminated (n : Nat) : Bool :=
    n % 6 == 0 || n % 7 == 0 || n.toString.contains '6' || n.toString.contains '7'
  -- Function to find the last person standing
  let find_last_person (players : List String) : String :=
    let rec aux (players : List String) (count : Nat) : String :=
      if players.length == 1 then
        players.headD ""
      else
        let to_remove := count % players.length
        if is_eliminated count then
          aux (players.removeNth to_remove) (count + 1)
        else
          aux players (count + 1)
    aux players 1
  find_last_person players

theorem final_person_is_Nicky : last_person_standing = "Nicky" :=
  by
    -- Here should be the proof steps, which is skipped
    sorry

end final_person_is_Nicky_l561_561315


namespace segments_interior_proof_l561_561914

noncomputable def count_internal_segments (squares hexagons octagons : Nat) : Nat := 
  let vertices := (squares * 4 + hexagons * 6 + octagons * 8) / 3
  let total_segments := (vertices * (vertices - 1)) / 2
  let edges_along_faces := 3 * vertices
  (total_segments - edges_along_faces) / 2

theorem segments_interior_proof : count_internal_segments 12 8 6 = 840 := 
  by sorry

end segments_interior_proof_l561_561914


namespace count_integers_between_sqrt8_sqrt72_l561_561246

-- Define the square roots of 8 and 72 for reference
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt(8)
def minIntAboveSqrt8 : ℤ := Int.ceil sqrt8

-- Define the largest integer less than sqrt(72)
def maxIntBelowSqrt72 : ℤ := Int.floor sqrt72

-- State the main theorem to prove
theorem count_integers_between_sqrt8_sqrt72 :
  minIntAboveSqrt8 = 3 ∧ maxIntBelowSqrt72 = 8 → 
  (maxIntBelowSqrt72 - minIntAboveSqrt8 + 1) = 6 :=
by
  sorry

end count_integers_between_sqrt8_sqrt72_l561_561246


namespace intersection_points_line_and_transformed_curve_l561_561189

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 - (1 / 2) * t, 2 + (Real.sqrt 3 / 2) * t)

def curve_polar (ρ : ℝ) : Prop :=
  ρ = 2

def transformation (x y : ℝ) : ℝ × ℝ :=
  (x, 2 * y)

theorem intersection_points_line_and_transformed_curve :
  (∀ t : ℝ, ∃ x y : ℝ, (x, y) = line_parametric t) →
  (∃ ρ : ℝ, curve_polar ρ) →
  (∃ x' y' : ℝ, (x', y') = transformation x y) →
  let L_cartesian := λ x : ℝ, 2 + Real.sqrt 3 - Real.sqrt 3 * x
  let C_cartesian := λ x y : ℝ, x^2 + y^2 = 4
  let C'_cartesian := λ x y' : ℝ, 4 * x^2 + y'^2 = 16
  ∃ n : ℕ, n = 2 :=
by
  -- Proof goes here
  sorry

end intersection_points_line_and_transformed_curve_l561_561189


namespace intersection_point_property_l561_561645

-- Definitions for ellipse, hyperbola, and relevant conditions
variable (m n a b : ℝ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_mn : m > n)
variable (ell : x y : ℝ → Prop) (h_ell : ∀ x y, (ell x y) ↔ (x^2 / m + y^2 / n = 1))
variable (hyp : x y : ℝ → Prop) (h_hyp : ∀ x y, (hyp x y) ↔ (x^2 / a - y^2 / b = 1))
variable (P : ℝ × ℝ) (H_Pell : ell P.fst P.snd) (H_Phyp : hyp P.fst P.snd)
variable (F1 F2 : ℝ × ℝ) (H_same_foci : ∀ x y, ell x y → hyp x y → (true)) -- Placeholder for the same foci condition

-- Main statement to be proved
theorem intersection_point_property :
  let PF1 := dist P F1
  let PF2 := dist P F2
  PF1 * PF2 = m - a :=
sorry

end intersection_point_property_l561_561645


namespace simplify_fraction_l561_561523

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b)) :=
by
  sorry

end simplify_fraction_l561_561523


namespace reciprocal_of_neg_five_l561_561865

theorem reciprocal_of_neg_five : ∃ b : ℚ, (-5) * b = 1 ∧ b = -1/5 :=
by
  sorry

end reciprocal_of_neg_five_l561_561865


namespace digit_matching_equalities_l561_561512

theorem digit_matching_equalities :
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ 99 → 0 ≤ b ∧ b ≤ 99 →
    ((a = 98 ∧ b = 1 ∧ (98 + 1)^2 = 100*98 + 1) ∨
     (a = 20 ∧ b = 25 ∧ (20 + 25)^2 = 100*20 + 25)) :=
by
  intros a b ha hb
  sorry

end digit_matching_equalities_l561_561512


namespace tan_ratio_l561_561330

theorem tan_ratio (α β : ℝ) 
  (h1 : Real.sin (α + β) = (Real.sqrt 3) / 2) 
  (h2 : Real.sin (α - β) = (Real.sqrt 2) / 2) : 
  (Real.tan α) / (Real.tan β) = (5 + 2 * Real.sqrt 6) / (5 - 2 * Real.sqrt 6) :=
by
  sorry

end tan_ratio_l561_561330


namespace number_of_divisors_2310_l561_561580

-- Define the number whose divisors are being counted
def n : ℕ := 2310

-- Define the prime factorization of the number
def factorization : n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num

-- Define the formula for the number of divisors
def num_divisors (n : ℕ) : ℕ :=
  let e_1 := 1
  let e_2 := 1
  let e_3 := 1
  let e_4 := 1
  let e_5 := 1
  in (e_1 + 1) * (e_2 + 1) * (e_3 + 1) * (e_4 + 1) * (e_5 + 1)

-- State the problem in a theorem
theorem number_of_divisors_2310 : num_divisors n = 32 :=
by
  rw [num_divisors, factorization]
  sorry

end number_of_divisors_2310_l561_561580


namespace proposition_holds_n_2019_l561_561616

theorem proposition_holds_n_2019 (P: ℕ → Prop) 
  (H1: ∀ k : ℕ, k > 0 → ¬ P (k + 1) → ¬ P k) 
  (H2: P 2018) : 
  P 2019 :=
by 
  sorry

end proposition_holds_n_2019_l561_561616


namespace slope_angle_bisector_l561_561996

open Real

theorem slope_angle_bisector (m1 m2 : ℝ) (h1 : m1 = 2) (h2 : m2 = -3) : 
    let k := (m1 + m2 + sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2) in
    m1 = 2 ∧ m2 = -3 → k = (-1 + sqrt 14) / 7 := 
by
  intros
  simp [h1, h2]
  sorry

end slope_angle_bisector_l561_561996


namespace fraction_division_l561_561080

def frac1 (x : ℝ) : ℝ := (x^2 - 5 * x + 6) / (x^2 - 1)
def frac2 (x : ℝ) : ℝ := (x - 3) / (x^2 + x)
def simplified_frac1 (x : ℝ) : ℝ := ((x - 2) * (x - 3)) / ((x - 1) * (x + 1))
def simplified_frac2 (x : ℝ) : ℝ := (x - 3) / (x * (x + 1))

theorem fraction_division (x : ℝ) (h1 : frac1 x = simplified_frac1 x) (h2 : frac2 x = simplified_frac2 x) :
  (frac1 x) / (frac2 x) = x * (x - 2) / (x - 1) :=
by 
  sorry

end fraction_division_l561_561080


namespace part1_max_min_part2_monotonic_l561_561666

noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := -x^2 + 2*x*(Real.tan θ) + 1

theorem part1_max_min (x : ℝ) (θ : ℝ) (h1 : x ∈ set.Icc (- Real.sqrt 3) 1) (h2 : θ = - Real.pi / 4) :
  (∃ x_max x_min, x_max = -1 ∧ f x_max θ = 2 ∧ x_min = 1 ∧ f x_min θ = -2) := sorry

theorem part2_monotonic (θ : ℝ) (h : θ ∈ set.Ioo (- Real.pi / 2) (Real.pi / 2)) : 
  (∀ x, x ∈ set.Icc (- Real.sqrt 3) 1 → f x θ ≤ f (- Real.sqrt 3) θ ∨ f x θ ≥ f 1 θ) ↔ 
  (θ ∈ set.Icc (- Real.pi) (- Real.pi / 3) ∪ set.Icc (Real.pi / 4) (Real.pi / 2)) := sorry

end part1_max_min_part2_monotonic_l561_561666


namespace computer_operations_l561_561925

theorem computer_operations : 
  ∀ (rate_multiplications rate_additions : ℕ) (time_multiplications_secs time_additions_secs : ℕ),
  rate_multiplications = 5000 →
  time_multiplications_secs = 1800 →
  rate_additions = 2 * rate_multiplications →
  time_additions_secs = 5400 →
  (rate_multiplications * time_multiplications_secs + rate_additions * time_additions_secs = 63000000) :=
begin
  sorry
end

end computer_operations_l561_561925


namespace train_platform_time_l561_561911

theorem train_platform_time :
  ∀ (L_train L_platform T_tree S D T_platform : ℝ),
    L_train = 1200 ∧ 
    T_tree = 120 ∧ 
    L_platform = 1100 ∧ 
    S = L_train / T_tree ∧ 
    D = L_train + L_platform ∧ 
    T_platform = D / S →
    T_platform = 230 :=
by
  intros
  sorry

end train_platform_time_l561_561911


namespace caleb_spent_in_total_l561_561526

variable (num_burgers : Nat) (cost_single : ℝ) (cost_double : ℝ) (num_double : Nat) (total_cost : ℝ)

def hamburgers : Nat := 50    -- Condition 1
def cost_single_burger : ℝ := 1    -- Condition 2
def cost_double_burger : ℝ := 1.5  -- Condition 3
def double_burgers : Nat := 49     -- Condition 4

theorem caleb_spent_in_total :
  total_cost = double_burgers * cost_double_burger + (hamburgers - double_burgers) * cost_single_burger := 
by
  have h1 : hamburgers - double_burgers = 1 := sorry
  have h2 : double_burgers * cost_double_burger = 73.5 := sorry
  have h3 : (hamburgers - double_burgers) * cost_single_burger = 1.0 := sorry
  show total_cost = 74.5 from sorry

end caleb_spent_in_total_l561_561526


namespace simplify_fraction_l561_561522

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b)) :=
by
  sorry

end simplify_fraction_l561_561522


namespace lim_power_infinity_greater_lim_power_infinity_lesser_l561_561363

theorem lim_power_infinity_greater (a : ℝ) (h : a > 1) :
  filter.tendsto (λ x : ℝ, a ^ x) filter.at_top filter.at_top :=
sorry

theorem lim_power_infinity_lesser (a : ℝ) (h : a < 1) :
  filter.tendsto (λ x : ℝ, a ^ x) filter.at_top (nhds 0) :=
sorry

end lim_power_infinity_greater_lim_power_infinity_lesser_l561_561363


namespace interval_integer_count_l561_561226

-- Define the problem conditions
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt8
def lower_bound : Int := ceil sqrt8

-- Define the largest integer less than sqrt72
def upper_bound : Int := floor sqrt72

-- Prove the number of integers between sqrt8 and sqrt72 is 6
theorem interval_integer_count : 
    (upper_bound - lower_bound + 1) = 6 := 
by
  -- Steps and proofs would go here, but we use sorry to skip the proof for now
  sorry

end interval_integer_count_l561_561226


namespace sum_squares_of_sines_l561_561533

-- Condition: Sum of squares of sines of angles from 0° to 360° in steps of 10°
theorem sum_squares_of_sines : 
  let angles (n : ℕ) := 10 * n
  let T := ∑ n in finset.range 37, real.sin (angles n * real.pi / 180)^2
  T = 18 :=
by
  sorry

end sum_squares_of_sines_l561_561533


namespace hens_on_farm_l561_561474

theorem hens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H + R = 75) : H = 67 :=
by
  sorry

end hens_on_farm_l561_561474


namespace neg_p_necessary_not_sufficient_for_neg_p_or_q_l561_561158

variables (p q : Prop)

theorem neg_p_necessary_not_sufficient_for_neg_p_or_q :
  (¬ p → ¬ (p ∨ q)) ∧ (¬ (p ∨ q) → ¬ p) :=
by {
  sorry
}

end neg_p_necessary_not_sufficient_for_neg_p_or_q_l561_561158


namespace similar_triangle_shortest_side_l561_561940

theorem similar_triangle_shortest_side (a b c h1 h2 : ℝ) (h : a = 15 ∧ c = 25 ∧ h2 = 50 ∧ c = 25) :
  let scale_factor := h2 / c in
  let shortest_side_second_triangle := a * scale_factor in
  shortest_side_second_triangle = 30 :=
by
  sorry

end similar_triangle_shortest_side_l561_561940


namespace john_investment_l561_561767

theorem john_investment (x : ℝ) (h1 : 0.11 * x + 0.125 * 8200 = 1282) : x = 2336.36 :=
by
s

end john_investment_l561_561767


namespace lifting_to_bodyweight_ratio_l561_561075

variable (t : ℕ) (w : ℕ) (p : ℕ) (delta_w : ℕ)

def lifting_total_after_increase (t : ℕ) (p : ℕ) : ℕ :=
  t + (t * p / 100)

def bodyweight_after_increase (w : ℕ) (delta_w : ℕ) : ℕ :=
  w + delta_w

theorem lifting_to_bodyweight_ratio (h_t : t = 2200) (h_w : w = 245) (h_p : p = 15) (h_delta_w : delta_w = 8) :
  lifting_total_after_increase t p / bodyweight_after_increase w delta_w = 10 :=
  by
    -- Use the given conditions
    rw [h_t, h_w, h_p, h_delta_w]
    -- Calculation steps are omitted, directly providing the final assertion
    sorry

end lifting_to_bodyweight_ratio_l561_561075


namespace cost_of_5_dozens_l561_561955

-- State the conditions
def cost_of_3_dozens : ℝ := 21.90
def dozens_purchased : ℕ := 3
def dozens_needed : ℕ := 5

-- Define the cost per dozen
def cost_per_dozen : ℝ := cost_of_3_dozens / dozens_purchased

-- Prove the cost for five dozens
theorem cost_of_5_dozens : dozens_needed * cost_per_dozen = 36.50 :=
by
  -- Sorry to skip the proof
  sorry

end cost_of_5_dozens_l561_561955


namespace number_of_divisors_2310_l561_561578

-- Define the number whose divisors are being counted
def n : ℕ := 2310

-- Define the prime factorization of the number
def factorization : n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num

-- Define the formula for the number of divisors
def num_divisors (n : ℕ) : ℕ :=
  let e_1 := 1
  let e_2 := 1
  let e_3 := 1
  let e_4 := 1
  let e_5 := 1
  in (e_1 + 1) * (e_2 + 1) * (e_3 + 1) * (e_4 + 1) * (e_5 + 1)

-- State the problem in a theorem
theorem number_of_divisors_2310 : num_divisors n = 32 :=
by
  rw [num_divisors, factorization]
  sorry

end number_of_divisors_2310_l561_561578


namespace solve_for_2a_plus_b_l561_561257

variable (a b : ℝ)

theorem solve_for_2a_plus_b (h1 : 4 * a ^ 2 - b ^ 2 = 12) (h2 : 2 * a - b = 4) : 2 * a + b = 3 := 
by
  sorry

end solve_for_2a_plus_b_l561_561257


namespace georgia_problem_l561_561626

/-- 
    Georgia is working on a test with 75 problems. 
    After 20 minutes, she has completed 10 problems. 
    She has 40 minutes left and 45 problems left to solve.
    Prove that the ratio of the number of problems she completed in the second 20 minutes 
    to the number of problems she completed in the first 20 minutes is 2:1. 
-/
theorem georgia_problem (total_problems first_20_problems second_40_remaining_problems : ℕ) 
  (total_eq : total_problems = 75) 
  (first_20_eq : first_20_problems = 10)
  (second_40_remaining_eq : second_40_remaining_problems = 45) :
  let second_20_problems := (total_problems - second_40_remaining_problems) - first_20_problems in
  (second_20_problems / first_20_problems) = 2 := 
by
  -- This proof is to be filled in.
  sorry

end georgia_problem_l561_561626


namespace floor_sqrt_23_squared_l561_561551

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end floor_sqrt_23_squared_l561_561551


namespace permutations_count_l561_561409

-- Define the conditions
variable (n : ℕ)
variable (a : Fin n → ℕ)

-- Define the main proposition
theorem permutations_count (hn : 2 ≤ n) (h_perm : ∀ k : Fin n, a k ≥ k.val - 2) :
  ∃! L, L = 2 * 3 ^ (n - 2) :=
by
  sorry

end permutations_count_l561_561409


namespace find_f_2009_l561_561640

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom cond2 : f 1 = 2

theorem find_f_2009 : f 2009 = 2 := by
  sorry

end find_f_2009_l561_561640


namespace number_of_positive_divisors_2310_l561_561600

theorem number_of_positive_divisors_2310 : 
  let n := 2310 in
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)] in
  let t := (factorization.map (λ p : ℕ × ℕ, p.snd + 1)).prod in
  t = 32 :=
by
  let n := 2310
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let t := (factorization.map (λ p, p.snd + 1)).prod
  sorry

end number_of_positive_divisors_2310_l561_561600


namespace eccentricity_of_hyperbola_is_sqrt3_plus1_l561_561025

variables {a b c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b)

def hyperbola := ∃ (x y : ℝ), (x^2) / (a^2) - (y^2) / (b^2) = 1

variables {F1 F2 P : ℝ × ℝ} (on_hyperbola : hyperbola P.1 P.2)
    (dot_product_zero : (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0)
    (radius_ratio_cond : (√(3) - 1) / 2 = r_inscribed / r_circumscribed)

theorem eccentricity_of_hyperbola_is_sqrt3_plus1
    (e : ℝ) (h : e = c / a) : e = sqrt 3 + 1 :=
sorry

end eccentricity_of_hyperbola_is_sqrt3_plus1_l561_561025


namespace polar_eq_of_circle_l561_561923

def polar_center : ℝ × ℝ := (2, π / 4)

def pole_pass_through_origin (ρ θ : ℝ) : Prop := ρ = 0 → θ = 0

theorem polar_eq_of_circle (ρ θ : ℝ) (h_center : polar_center = (2, π / 4))
  (h_pole: pole_pass_through_origin ρ θ) :
  ρ = 2 * sqrt 2 * (sin θ + cos θ) := sorry

end polar_eq_of_circle_l561_561923


namespace find_a1_l561_561661

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with a common ratio q > 0
axiom geom_seq : (∀ n, a (n + 1) = a n * q)

-- Given conditions of the problem
def condition1 : q > 0 := sorry
def condition2 : a 5 * a 7 = 4 * (a 4) ^ 2 := sorry
def condition3 : a 2 = 1 := sorry

-- Prove that a_1 = sqrt 2 / 2
theorem find_a1 : a 1 = (Real.sqrt 2) / 2 := sorry

end find_a1_l561_561661


namespace circle_square_area_l561_561542

theorem circle_square_area :
  let square := {p : ℝ × ℝ | -15 ≤ p.1 ∧ p.1 ≤ 15 ∧ -15 ≤ p.2 ∧ p.2 ≤ 15}
  let circle := {p : ℝ × ℝ | (p.1 - 8)^2 + (p.2 - 15)^2 = 64}
  let intersection := square ∩ circle
  (∃! area : ℝ, area = 64 * Real.pi) :=
by
  -- Definitions and constraints from the problem
  let square := {p : ℝ × ℝ | -15 ≤ p.1 ∧ p.1 ≤ 15 ∧ -15 ≤ p.2 ∧ p.2 ≤ 15}
  let circle := {p : ℝ × ℝ | (p.1 - 8)^2 + (p.2 - 15)^2 = 64}
  let intersection := square ∩ circle

  -- Statement of existence and uniqueness of the area
  exists_unique! (64 * Real.pi) sorry

end circle_square_area_l561_561542


namespace function_properties_sum_of_roots_l561_561669

-- Definitions based on conditions
def f (x : ℝ) := Real.sin (3 * x - π / 4)

theorem function_properties (x : ℝ) (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < π/2) :
  (f (π / 4) = 1) ∧ (f (7 * π / 12) = -1) ∧ f = (λ x, Real.sin (3 * x - π / 4)) :=
by {
  sorry -- proof of analytical expression and transformations
}

theorem sum_of_roots (a : ℝ) (ha : 0 < a ∧ a < 1) : 
  ∑ x in { x | 0 ≤ x ∧ x ≤ 2 * π ∧ f x = a }, x = 11 * π / 2 :=
by {
  sorry -- proof of sum of real roots
}

end function_properties_sum_of_roots_l561_561669


namespace probability_of_pentagon_with_positive_area_l561_561615

theorem probability_of_pentagon_with_positive_area (n : ℕ) (hn : n = 15) : 
  let total_segments := nat.choose n 2,
      required_probability := 70 / 100
  in 
  ∃ selected_segments : finset (fin (total_segments)),
    selected_segments.card = 5 ∧
    (∀ (s : finset (fin (selected_segments.card))), 
     (∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c → 
      s.sum = a + b + c → a + b > c ∧ a + c > b ∧ b + c > a)) ∧ 
    (selected_segments.card.to_real / total_segments.to_real = required_probability) :=
sorry

end probability_of_pentagon_with_positive_area_l561_561615


namespace probability_tuesday_l561_561043

def days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def consecutive_days (d1 d2 : String) : Bool :=
  (d1, d2) = ("Monday", "Tuesday") ∨
  (d1, d2) = ("Tuesday", "Wednesday") ∨
  (d1, d2) = ("Wednesday", "Thursday") ∨
  (d1, d2) = ("Thursday", "Friday")

def includes_tuesday (d1 d2 : String) : Bool :=
  d1 = "Tuesday" ∨ d2 = "Tuesday"

theorem probability_tuesday :
  (count (λ p, consecutive_days p.1 p.2 && includes_tuesday p.1 p.2)
    [(days[0], days[1]), (days[1], days[2]), (days[2], days[3]), (days[3], days[4])] / 
  count (λ p, consecutive_days p.1 p.2)
    [(days[0], days[1]), (days[1], days[2]), (days[2], days[3]), (days[3], days[4])]) = 1 / 2 := by
  sorry

end probability_tuesday_l561_561043


namespace limit_seq_iff_limit_func_l561_561805

variables {α : Type*} [TopologicalSpace α]

theorem limit_seq_iff_limit_func (f : α → ℝ) (x₀ a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, (abs (x - x₀) < δ) → abs (f x - a) < ε) ↔
  (∀ (a_n : ℕ → ℝ), (∀ n, a_n n ∈ set.univ) → filter.tendsto a_n filter.at_top (nhds x₀) ∧ (∀ n, a_n n ≠ x₀) → filter.tendsto (λ n, f (a_n n)) filter.at_top (nhds a)) :=
sorry

end limit_seq_iff_limit_func_l561_561805


namespace kenneth_speed_l561_561971

theorem kenneth_speed (biff_speed : ℕ) (race_distance : ℕ) (kenneth_extra_distance : ℕ) :
  biff_speed = 50 → race_distance = 500 → kenneth_extra_distance = 10 →
  let biff_time := race_distance / biff_speed in
  let kenneth_distance := race_distance + kenneth_extra_distance in
  let kenneth_speed := kenneth_distance / biff_time in
  kenneth_speed = 51 :=
by
  intros h1 h2 h3
  let biff_time := race_distance / biff_speed
  let kenneth_distance := race_distance + kenneth_extra_distance
  let kenneth_speed := kenneth_distance / biff_time
  rw [h1, h2, h3]
  have biff_time_n : biff_time = 10 := by norm_num
  have kenneth_distance_n : kenneth_distance = 510 := by norm_num
  rw [biff_time_n, kenneth_distance_n]
  norm_num
  sorry

end kenneth_speed_l561_561971


namespace interval_integer_count_l561_561228

-- Define the problem conditions
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt8
def lower_bound : Int := ceil sqrt8

-- Define the largest integer less than sqrt72
def upper_bound : Int := floor sqrt72

-- Prove the number of integers between sqrt8 and sqrt72 is 6
theorem interval_integer_count : 
    (upper_bound - lower_bound + 1) = 6 := 
by
  -- Steps and proofs would go here, but we use sorry to skip the proof for now
  sorry

end interval_integer_count_l561_561228


namespace number_of_divisors_of_2310_l561_561594

theorem number_of_divisors_of_2310 : 
  let n := 2310 in
  let prime_factors := (2, 1) :: (3, 1) :: (5, 1) :: (7, 1) :: (11, 1) :: [] in
  n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 →
  (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 32 :=
begin
  intro h,
  sorry
end

end number_of_divisors_of_2310_l561_561594


namespace infinitely_many_composites_l561_561808

theorem infinitely_many_composites (t : ℕ) :
  ∃ n : ℕ, n = 3^(2^t) - 2^(2^t) ∧ Composite n ∧ n ∣ (3^(n-1) - 2^(n-1)) :=
sorry

end infinitely_many_composites_l561_561808


namespace product_of_solutions_l561_561888

theorem product_of_solutions :
  (∃ a b c : ℝ, a = -2 ∧ b = 6 ∧ c = 45 ∧ (∀ x : ℝ, -2 * x^2 + 6 * x + 45 = 0) →
  ((c / a) = -22.5)) :=
begin
  sorry
end

end product_of_solutions_l561_561888


namespace find_k_l561_561149

variable {α : Type*} [LinearOrderedField α]

-- Definition of arithmetic sequence
def arithmetic_seq (a d : α) (n : ℕ) : α :=
  a + n * d

-- Given conditions
variables {a d : α}
variables (h1 : arithmetic_seq a d 3 + arithmetic_seq a d 6 + arithmetic_seq a d 9 = 17)
variables (h2 : (∑ i in Finset.range 11, arithmetic_seq a d (3 + i)) = 77)

-- Conjecture
theorem find_k (h : arithmetic_seq a d k = 13) : k = 18 :=
sorry

end find_k_l561_561149


namespace star_operation_l561_561487

def star (a b : ℚ) : ℚ := 2 * a - b + 1

theorem star_operation :
  star 1 (star 2 (-3)) = -5 :=
by
  -- Calcualtion follows the steps given in the solution, 
  -- but this line is here just to satisfy the 'rewrite the problem' instruction.
  sorry

end star_operation_l561_561487


namespace count_integers_between_sqrt8_sqrt72_l561_561247

-- Define the square roots of 8 and 72 for reference
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt(8)
def minIntAboveSqrt8 : ℤ := Int.ceil sqrt8

-- Define the largest integer less than sqrt(72)
def maxIntBelowSqrt72 : ℤ := Int.floor sqrt72

-- State the main theorem to prove
theorem count_integers_between_sqrt8_sqrt72 :
  minIntAboveSqrt8 = 3 ∧ maxIntBelowSqrt72 = 8 → 
  (maxIntBelowSqrt72 - minIntAboveSqrt8 + 1) = 6 :=
by
  sorry

end count_integers_between_sqrt8_sqrt72_l561_561247


namespace fourth_intersection_point_l561_561749

noncomputable def fourth_point_of_intersection : Prop :=
  let hyperbola (x y : ℝ) := x * y = 1
  let circle (x y : ℝ) := (x - 1)^2 + (y + 1)^2 = 10
  let known_points : List (ℝ × ℝ) := [(3, 1/3), (-4, -1/4), (1/2, 2)]
  let fourth_point := (-1/6, -6)
  (hyperbola 3 (1/3)) ∧ (hyperbola (-4) (-1/4)) ∧ (hyperbola (1/2) 2) ∧
  (circle 3 (1/3)) ∧ (circle (-4) (-1/4)) ∧ (circle (1/2) 2) ∧ 
  (hyperbola (-1/6) (-6)) ∧ (circle (-1/6) (-6)) ∧ 
  ∀ (x y : ℝ), (hyperbola x y) → (circle x y) → ((x, y) = fourth_point ∨ (x, y) ∈ known_points)
  
theorem fourth_intersection_point :
  fourth_point_of_intersection :=
sorry

end fourth_intersection_point_l561_561749


namespace midpoint_of_A_and_B_is_correct_l561_561649

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def midpoint (A B : Point3D) : Point3D :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2,
  z := (A.z + B.z) / 2 }

def A : Point3D := { x := 3, y := 2, z := 3 }
def B : Point3D := { x := 1, y := 1, z := 4 }

theorem midpoint_of_A_and_B_is_correct : midpoint A B = { x := 2, y := 3 / 2, z := 7 / 2 } :=
by 
  -- proof goes here
  sorry

end midpoint_of_A_and_B_is_correct_l561_561649


namespace proof_general_term_and_T_n_l561_561155

-- Define the arithmetic sequence and summation properties
def arithmetic_sequence (a_n : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n : ℕ, a_n n = a₁ + a_ℕ n * d
  
def sum_first_n_terms (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S_n n = (n * (a₁ + a_n n)) / 2

-- Define the given conditions
constant {a₁ d : ℤ}
constant (a_n : ℕ → ℤ)

-- Given a_n is an arithmetic sequence
axiom a_sequence : arithmetic_sequence a_n a₁ d

-- Given conditions for specific terms and sums
axiom cond_1 : 2 * a_n 5 - sum_first_n_terms a_n 4 = 2
axiom cond_2 : 3 * a_n 2 + a_n 6 = 32

-- Define the general term formula
def general_term_formula (a_n : ℕ → ℤ) := ∀ n : ℕ, a_n n = 3 * n - 1

-- Define the alternate sequence formula
noncomputable def T_n (a_n : ℕ → ℤ) (n : ℕ) :=
  (finset.range n).sum (λ (k : ℕ), a n k / 2^k)

-- The theorem to prove the calculated T_n
theorem proof_general_term_and_T_n :
  general_term_formula a_n ∧ (∀ n : ℕ, T_n a_n n = 5 / 2 - (3 * n + 5) / 2^(n+1)) :=
sorry

end proof_general_term_and_T_n_l561_561155


namespace unique_representation_cardinality_difference_l561_561463

/-- Definition of weight of a number n -/
def weight (n : ℕ) : ℕ :=
  (nat.find (exists_eq_sum n)).1

/-- Theorem to prove uniqueness of the representation. -/
theorem unique_representation (n : ℕ) (h : 0 < n) :
  ∃ k (m : fin (2*k+1) → ℕ),
  n = ∑ j in finset.range (2*k+1), (-1)^(j:ℕ) * 2^(m j) ∧
  strict_mono m :=
sorry

/-- Set A: numbers with even weight in the range from 1 to 2^2017 -/
def set_A := {n : ℕ | 1 ≤ n ∧ n ≤ 2^2017 ∧ even (weight n)}

/-- Set B: numbers with odd weight in the range from 1 to 2^2017 -/
def set_B := {n : ℕ | 1 ≤ n ∧ n ≤ 2^2017 ∧ odd (weight n)}

/-- The cardinality difference of sets A and B is 0 -/
theorem cardinality_difference :
  (finset.card set_A) - (finset.card set_B) = 0 :=
sorry

end unique_representation_cardinality_difference_l561_561463


namespace problem_statement_l561_561327

noncomputable def a := 1 / (2 * Real.exp(1))
noncomputable def b := 3 / 2
noncomputable def c := Real.sqrt (Real.pi^3 / 4)

def P (x : ℝ) (n : ℕ) : Prop :=
  ∃ (p : ℚ[X]), ∀ i, p.monic ∧ p.degree = n ∧ P.coeff i = x

def Q (x : ℝ) (n : ℕ) : ℝ :=
  ∏ i in Finset.range (n + 1), (x + i + 1) ^ 2

def m_n (n : ℕ) : ℝ :=
  Nat.Minimize (λ P : ℝ, ∑ i in Finset.range (n + 1), i ^ 2 * (P.coeff i^2) ^ 2 / (Q (i) n)) sorry

theorem problem_statement : ⌊2019 * a * b * c ^ 2⌋ = 4318 := sorry

end problem_statement_l561_561327


namespace camp_organizer_needs_more_bottles_l561_561033

variable (cases : ℕ) (bottles_per_case : ℕ) (cases_bought : ℕ)
variable (children_group1 : ℕ) (children_group2 : ℕ) (children_group3 : ℕ)
variable (bottles_per_day : ℕ) (days : ℕ)

noncomputable def bottles_needed := 
  let children_group4 := (children_group1 + children_group2 + children_group3) / 2
  let total_children := children_group1 + children_group2 + children_group3 + children_group4
  let total_bottles_needed := total_children * bottles_per_day * days
  let total_bottles_purchased := cases_bought * bottles_per_case
  total_bottles_needed - total_bottles_purchased

theorem camp_organizer_needs_more_bottles :
  cases = 13 →
  bottles_per_case = 24 →
  cases_bought = 13 →
  children_group1 = 14 →
  children_group2 = 16 →
  children_group3 = 12 →
  bottles_per_day = 3 →
  days = 3 →
  bottles_needed cases bottles_per_case cases_bought children_group1 children_group2 children_group3 bottles_per_day days = 255 := by
  sorry

end camp_organizer_needs_more_bottles_l561_561033


namespace interval_integer_count_l561_561227

-- Define the problem conditions
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt8
def lower_bound : Int := ceil sqrt8

-- Define the largest integer less than sqrt72
def upper_bound : Int := floor sqrt72

-- Prove the number of integers between sqrt8 and sqrt72 is 6
theorem interval_integer_count : 
    (upper_bound - lower_bound + 1) = 6 := 
by
  -- Steps and proofs would go here, but we use sorry to skip the proof for now
  sorry

end interval_integer_count_l561_561227


namespace radius_ratio_l561_561044

noncomputable def volume_large_sphere : ℝ := 432 * Real.pi

noncomputable def volume_small_sphere : ℝ := 0.08 * volume_large_sphere

noncomputable def radius_large_sphere : ℝ :=
  (3 * volume_large_sphere / (4 * Real.pi)) ^ (1 / 3)

noncomputable def radius_small_sphere : ℝ :=
  (3 * volume_small_sphere / (4 * Real.pi)) ^ (1 / 3)

theorem radius_ratio (V_L V_s : ℝ) (hL : V_L = 432 * Real.pi) (hS : V_s = 0.08 * V_L) :
  (radius_small_sphere / radius_large_sphere) = (2/5)^(1/3) :=
by
  sorry

end radius_ratio_l561_561044


namespace sum_p_q_r_l561_561085

def b (n : ℕ) : ℕ :=
if n < 1 then 0 else
if n < 2 then 2 else
if n < 4 then 4 else
if n < 7 then 6
else 6 -- Continue this pattern for illustration; an infinite structure would need proper handling for all n.

noncomputable def p := 2
noncomputable def q := 0
noncomputable def r := 0

theorem sum_p_q_r : p + q + r = 2 :=
by sorry

end sum_p_q_r_l561_561085


namespace count_values_with_g30_eq_16_l561_561133

def g1 (n : ℕ) : ℕ := 4 * (n.divisors.count id)

def g (j n : ℕ) : ℕ := 
  if j = 1 then g1 n
  else g1 (g (j - 1) n)

theorem count_values_with_g30_eq_16 : { n | n ≤ 100 ∧ g 30 n = 16 }.card = 4 := 
by
  sorry

end count_values_with_g30_eq_16_l561_561133


namespace circles_internally_tangent_l561_561861

-- Definition of first circle equation (condition 1)
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y + 12 = 0

-- Definition of second circle equation (condition 2)
def circle2_eq (x y : ℝ) : Prop := (x - 7)^2 + (y - 1)^2 = 36

-- Proof problem statement: The circles are internally tangent
theorem circles_internally_tangent :
  ∃ c1 r1 c2 r2, c1 = (3, -2) ∧ r1 = 1 ∧
                  c2 = (7, 1) ∧ r2 = 6 ∧
                  ∀ d, d = real.sqrt ((3 - 7)^2 + (-2 - 1)^2) → d = 5 ∧
                  d = r2 - r1 :=
sorry

end circles_internally_tangent_l561_561861


namespace number_of_positive_divisors_2310_l561_561595

theorem number_of_positive_divisors_2310 : 
  let n := 2310 in
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)] in
  let t := (factorization.map (λ p : ℕ × ℕ, p.snd + 1)).prod in
  t = 32 :=
by
  let n := 2310
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let t := (factorization.map (λ p, p.snd + 1)).prod
  sorry

end number_of_positive_divisors_2310_l561_561595


namespace find_a_l561_561293

theorem find_a (a b : ℝ) (h₀ : b = 4) (h₁ : (4, b) ∈ {p | p.snd = 0.75 * p.fst + 1}) 
  (h₂ : (a, 5) ∈ {p | p.snd = 0.75 * p.fst + 1}) (h₃ : (a, b+1) ∈ {p | p.snd = 0.75 * p.fst + 1}) : 
  a = 5.33 :=
by 
  sorry

end find_a_l561_561293


namespace circle_shaded_area_correct_l561_561040

noncomputable def circleAreaDifference : ℝ :=
  let radius_small := 2
  let radius_large := 3
  let center_1 := (0, -2)
  let center_2 := (0, 2)
  let distance_centers := 4
  4 * ((9 * Real.pi / 8) - (sqrt 5) - (Real.pi / 2)) = 5 * Real.pi / 2 - 4 * sqrt 5

theorem circle_shaded_area_correct :
  circleAreaDifference =
   (\frac{5\pi}{2} - 4\sqrt{5}) := 
by
  sorry

end circle_shaded_area_correct_l561_561040


namespace range_of_a_l561_561668

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a * x^2 - x + 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ -1) ↔ (a ≥ 1/12) :=
by
  sorry

end range_of_a_l561_561668


namespace alice_must_fill_cup_l561_561064

-- Define the problem conditions
def total_cups_sugar : ℚ := 3 + 1 / 2
def sugar_at_home : ℚ := 3 / 4
def cup_capacity : ℚ := 1 + 1 / 2
def ounces_per_cup : ℚ := 8

-- Define conversion from cups to ounces
def cups_to_ounces (cups : ℚ) : ℚ := cups * ounces_per_cup

-- Main statement to prove
theorem alice_must_fill_cup :
  let total_ounces_sugar : ℚ := cups_to_ounces total_cups_sugar
  let ounces_available : ℚ := cups_to_ounces sugar_at_home
  let remaining_ounces : ℚ := total_ounces_sugar - ounces_available
  let fills_needed : ℚ := remaining_ounces / cup_capacity
  ceil fills_needed = 15 :=
sorry

end alice_must_fill_cup_l561_561064


namespace limit_of_fraction_l561_561974

theorem limit_of_fraction (n : ℕ) :
  (real.lim (λ n : ℕ, (3 ^ n - 2 ^ n) / (3 ^ (n + 1) + 2 ^ (n + 1)))) = 1 / 3 :=
by sorry

end limit_of_fraction_l561_561974


namespace minimize_t_l561_561650

variable (Q : ℝ) (Q_1 Q_2 Q_3 Q_4 Q_5 Q_6 Q_7 Q_8 Q_9 : ℝ)

-- Definition of the sum of undirected lengths
def t (Q : ℝ) := 
  abs (Q - Q_1) + abs (Q - Q_2) + abs (Q - Q_3) + 
  abs (Q - Q_4) + abs (Q - Q_5) + abs (Q - Q_6) + 
  abs (Q - Q_7) + abs (Q - Q_8) + abs (Q - Q_9)

-- Statement that t is minimized when Q = Q_5
theorem minimize_t : ∀ Q : ℝ, t Q ≥ t Q_5 := 
sorry

end minimize_t_l561_561650


namespace tangent_diff_l561_561173

noncomputable def α (p : Point) : ℝ :=
arctan p.2 p.1

theorem tangent_diff (p : Point) (h : p = (-√3, 2)) :
  tan (α p - π / 6) = -3*√3 := by
  sorry

end tangent_diff_l561_561173


namespace product_abc_d_l561_561177

noncomputable theory
open_locale classical

variables (a b c d : ℚ)

theorem product_abc_d 
  (h1 : 4 * a + 2 * b + 6 * c + 8 * d = 48)
  (h2 : 4 * d + 2 * c = 2 * b)
  (h3 : 4 * b + 2 * c = 2 * a)
  (h4 : c + 2 = d) :
  a * b * c * d = -11033 / 1296 :=
sorry

end product_abc_d_l561_561177


namespace dot_BC_l561_561723

variables (A B C : Type) [AddCommGroup A] [VectorSpace ℝ A]

def vector_AB : A := (⟨3, -1⟩ : A)
def n : A := (⟨2, 1⟩ : A)
def dot_AC_eq_7 : (n ⬝ C - n ⬝ A) = 7

theorem dot_BC :
  (n ⬝ (C - B)) = 2 :=
sorry

end dot_BC_l561_561723


namespace polynomial_solution_l561_561010

theorem polynomial_solution (p : ℤ → ℤ → ℤ) (h : ∀ x y, p (x + y) (x * y) = ∑ k in Finset.range 21, (x ^ (20 - k)) * (y ^ k)) :
  p z t = z^{20} - 19 * z^{18} * t + 153 * z^{16} * (t^2) - 680 * z^{14} * (t^3) + 1820 * z^{12} * (t^4) 
           - 3003 * z^{10} * (t^5) + 3003 * z^{8} * (t^6) - 1716 * z^{6} * (t^7) 
           + 495 * z^{4} * (t^8) - 55 * z^{2} * (t^9) + (t^{10}) :=
by
  sorry

end polynomial_solution_l561_561010


namespace fg_of_one_eq_onehundredandfive_l561_561707

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2)^3

theorem fg_of_one_eq_onehundredandfive : f (g 1) = 105 :=
by
  -- proof would go here
  sorry

end fg_of_one_eq_onehundredandfive_l561_561707


namespace TVs_auction_site_l561_561969

variable (TV_in_person : Nat)
variable (TV_online_multiple : Nat)
variable (total_TVs : Nat)

theorem TVs_auction_site :
  ∀ (TV_in_person : Nat) (TV_online_multiple : Nat) (total_TVs : Nat), 
  TV_in_person = 8 → TV_online_multiple = 3 → total_TVs = 42 →
  (total_TVs - (TV_in_person + TV_online_multiple * TV_in_person) = 10) :=
by
  intros TV_in_person TV_online_multiple total_TVs h1 h2 h3
  rw [h1, h2, h3]
  sorry

end TVs_auction_site_l561_561969


namespace time_to_cross_pole_l561_561950

noncomputable theory

def speed_kmh := 60      -- Speed in km/hr
def train_length := 350  -- Length in meters

def speed_ms := (speed_kmh * 1000) / 3600 -- Speed converted to m/s
def crossing_time := train_length / speed_ms -- Time to cross the pole

theorem time_to_cross_pole : crossing_time = 21 := 
by {
  -- Convert speed from km/hr to m/s
  have speed_ms_eq : speed_ms = 50 / 3 := by {
    calc
      speed_ms = (60 * 1000) / 3600 := rfl
            ... = 60000 / 3600 := rfl
            ... = 600 / 36 := rfl
            ... = 50 / 3 := rfl
  },

  -- Use the formula time = distance / speed
  have time_eq : crossing_time = 350 / (50 / 3) := rfl,

  -- Calculate the time to verify it is 21 seconds
  calc
    crossing_time = 350 / (50 / 3) := time_eq
               ... = 350 * (3 / 50) := by rw div_eq_mul_inv
               ... = (350 * 3) / 50 := rfl
               ... = 1050 / 50 := rfl
               ... = 21 := by norm_num
}

#print time_to_cross_pole

end time_to_cross_pole_l561_561950


namespace domain_h_l561_561994

def h (x : ℝ) : ℝ := (x^3 - 3 * x^2 + 5 * x + 2) / (x^2 - 5 * x + 6)

theorem domain_h : ∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ↔ x ∈ (-∞, 2) ∪ (2, 3) ∪ (3, ∞) :=
by
  sorry

end domain_h_l561_561994


namespace willy_days_to_finish_series_l561_561446

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def days_to_finish (total_episodes : ℕ) (episodes_per_day : ℕ) : ℕ :=
  total_episodes / episodes_per_day

theorem willy_days_to_finish_series : 
  total_episodes 3 20 = 60 → 
  days_to_finish 60 2 = 30 :=
by
  intros h1
  rw [h1]
  rfl

end willy_days_to_finish_series_l561_561446


namespace speed_skater_passings_l561_561431

theorem speed_skater_passings (L : ℝ) (h1 : 1 < L) (h2 : L < 3.14) (h3 : ((3.14 - 1) * (60 / 2.14)) / 2 = 117) :
  16 := sorry

end speed_skater_passings_l561_561431


namespace count_integers_between_sqrts_l561_561234

theorem count_integers_between_sqrts : 
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  (upper_bound - lower_bound + 1) = 6 :=
by
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  calc (upper_bound - lower_bound + 1) = 6 : sorry

end count_integers_between_sqrts_l561_561234


namespace smallest_m_l561_561790

noncomputable def f (x : ℝ) : ℝ := sorry

theorem smallest_m (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x) (hy : y ≤ 1) (h_eq : f 0 = f 1) 
(h_lt : forall x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → |f x - f y| < |x - y|): 
|f x - f y| < 1 / 2 := 
sorry

end smallest_m_l561_561790


namespace term_position_l561_561465

theorem term_position (a : ℕ → ℕ) (d : ℕ) (first_term : ℕ) (n : ℕ) :
  (∀ n, a n = first_term + n * d) →
  a n = 2005 → n = 334 := 
by {
  -- We are given the sequence information
  intros h_seq h_2005, 
  -- Using the given conditions
  have h_seq_form : ∀ n, a n = 7 + n * 6 := by {
    intro n,
    exact h_seq n, 
  },
  -- Substitute the information we have
  have h_equation : 7 + 334 * 6 = 2005 := by norm_num,
  -- Show that this leads to n being 334
  have h_solution : 334 = n := by {
    rw [h_seq_form, h_2005] at h_equation,
    exact h_equation }
}


end term_position_l561_561465


namespace volume_of_region_l561_561606

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  ∫∫∫ {xyz : ℝ × ℝ × ℝ | f xyz.1 xyz.2 xyz.3 ≤ 6} 1 = 27/2 := 
sorry

end volume_of_region_l561_561606


namespace son_shoveling_time_l561_561012

-- Given conditions
def son_shoveling_rate (S : ℝ) : Prop :=
  S > 0

def wayne_shoveling_rate (S : ℝ) : ℝ :=
  6 * S

def neighbor_shoveling_rate (S : ℝ) : ℝ :=
  2 * wayne_shoveling_rate S

def combined_shoveling_rate (S : ℝ) : ℝ :=
  S + wayne_shoveling_rate S + neighbor_shoveling_rate S

def total_shoveling_time : ℝ :=
  2  -- 2 hours

-- Proving the son's shoveling time
theorem son_shoveling_time (S : ℝ) (hS : son_shoveling_rate S) : 1 / S = 38 :=
by
  have work_done : combined_shoveling_rate S * total_shoveling_time = 1 := sorry
  have solve_S : S = 1 / 38 := sorry
  show 1 / S = 38 from sorry

end son_shoveling_time_l561_561012


namespace reciprocal_of_neg_five_l561_561864

theorem reciprocal_of_neg_five : ∃ b : ℚ, (-5) * b = 1 ∧ b = -1/5 :=
by
  sorry

end reciprocal_of_neg_five_l561_561864


namespace auction_site_TVs_proof_l561_561968

variable (first_store_TVs online_store_TVs auction_site_TVs : ℕ)
variable (total_TVs : ℕ)

-- Conditions
def condition1 : Prop := first_store_TVs = 8
def condition2 : Prop := online_store_TVs = 3 * first_store_TVs
def condition3 : Prop := total_TVs = first_store_TVs + online_store_TVs + auction_site_TVs
def condition4 : Prop := total_TVs = 42

-- Theorem asserting the number of TVs looked at the auction site
theorem auction_site_TVs_proof 
  (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  auction_site_TVs = 10 := by
  sorry

end auction_site_TVs_proof_l561_561968


namespace tangent_line_eq_l561_561039

noncomputable def circle_tangent_line : Prop :=
  let circle : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 4}
  let M : ℝ × ℝ := (4, -1)
  ∃ (P Q : ℝ × ℝ), (P.1 * 4 - P.2 = 4) ∧ (Q.1 * 4 - Q.2 = 4) ∧ 
  (∀ p : ℝ × ℝ, p ∈ circle → (p = P ∨ p = Q)) ∧
  (∀ p1 p2, p1 = P → p2 = Q → 4 * p1.1 - p1.2 - 4 = 0)

-- assertion that needs to be proven
theorem tangent_line_eq :
  circle_tangent_line :=
sorry

end tangent_line_eq_l561_561039


namespace ratio_of_part_to_whole_l561_561799

theorem ratio_of_part_to_whole : 
  (1 / 4) * (2 / 5) * P = 15 → 
  (40 / 100) * N = 180 → 
  P / N = 1 / 6 := 
by
  intros h1 h2
  sorry

end ratio_of_part_to_whole_l561_561799


namespace math_equivalent_proof_l561_561035

-- Define the probabilities given the conditions
def P_A1 := 3 / 4
def P_A2 := 2 / 3
def P_A3 := 1 / 2
def P_B1 := 3 / 5
def P_B2 := 2 / 5

-- Define events
def P_C : ℝ := (P_A1 * P_B1 * (1 - P_A2)) + (P_A1 * P_B1 * P_A2 * P_B2 * (1 - P_A3))

-- Probability distribution of X
def P_X_0 : ℝ := (1 - P_A1) + P_C
def P_X_600 : ℝ := P_A1 * (1 - P_B1)
def P_X_1500 : ℝ := P_A1 * P_B1 * P_A2 * (1 - P_B2)
def P_X_3000 : ℝ := P_A1 * P_B1 * P_A2 * P_B2 * P_A3

-- Expected value of X
def E_X : ℝ := 600 * P_X_600 + 1500 * P_X_1500 + 3000 * P_X_3000

-- Statement to prove P(C) and expected value E(X)
theorem math_equivalent_proof :
  P_C = 21 / 100 ∧ 
  P_X_0 = 23 / 50 ∧
  P_X_600 = 3 / 10 ∧
  P_X_1500 = 9 / 50 ∧
  P_X_3000 = 3 / 50 ∧ 
  E_X = 630 := 
by 
  sorry

end math_equivalent_proof_l561_561035


namespace num_of_n_values_l561_561416

noncomputable def polynomial_n_values (x : ℤ → ℤ) : ℕ :=
  let P := x^3 - 2004*x^2 + x.1*x.2*x.3
  let a + b = 1002 in
  let c = 1002 in
  let distinct_positive_integers := λ (a b : ℕ), a + b = 1002 ∧ a ≠ b
  let n := 1002 * x.1 * x.2
  finset.card { n | ∃ (a b : ℕ), distinct_positive_integers a b }

theorem num_of_n_values : polynomial_n_values = 1001 := 
  sorry

end num_of_n_values_l561_561416


namespace integral_evaluation_l561_561020

noncomputable def integrand (x : ℝ) : ℝ :=
  (Real.sin x) / (1 + Real.cos x + Real.sin x)^2

noncomputable def definite_integral : ℝ :=
  ∫ x in 0..(Real.pi / 2), integrand x

theorem integral_evaluation : definite_integral = Real.log 2 - 1 / 2 :=
sorry

end integral_evaluation_l561_561020


namespace students_at_end_of_year_l561_561294

-- Define the initial number of students
def initial_students : Nat := 10

-- Define the number of students who left during the year
def students_left : Nat := 4

-- Define the number of new students who arrived during the year
def new_students : Nat := 42

-- Proof problem: the number of students at the end of the year
theorem students_at_end_of_year : initial_students - students_left + new_students = 48 := by
  sorry

end students_at_end_of_year_l561_561294


namespace median_is_30_l561_561729

def donation_amounts : list ℕ := [10, 30, 40, 50, 15, 20, 50]

theorem median_is_30 : (List.median donation_amounts) = 30 :=
by
  sorry

end median_is_30_l561_561729


namespace percentage_of_part_over_whole_l561_561028

theorem percentage_of_part_over_whole (Part Whole : ℕ) (h1 : Part = 120) (h2 : Whole = 50) :
  (Part / Whole : ℚ) * 100 = 240 := by
  sorry

end percentage_of_part_over_whole_l561_561028


namespace fraction_simplification_solve_fractional_equation_l561_561464

def simplify_fraction (x : ℝ) : Prop :=
  ∀ x, (x ≠ 2 ∧ x ≠ -2) → 
      (frac {4 / (x^2 - 4) - 1 / (x - 2)} = -1 / (x + 2))

def solve_equation (x : ℝ) : Prop :=
  ∀ x, (x ≠ 2 ∧ x ≠ -2) →
      (4 / (x^2 - 4) - 1 / (x - 2) = 1 / 2) ↔ (x = -4)

theorem fraction_simplification (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  simplify_fraction x := 
  sorry

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  solve_equation x := 
  sorry

end fraction_simplification_solve_fractional_equation_l561_561464


namespace focus_of_parabola_eq_l561_561565

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := -5 * x^2 + 10 * x - 2

-- Statement of the theorem to find the focus of the given parabola
theorem focus_of_parabola_eq (x : ℝ) : 
  let vertex_x := 1
  let vertex_y := 3
  let a := -5
  ∃ focus_x focus_y, 
    focus_x = vertex_x ∧ 
    focus_y = vertex_y - (1 / (4 * a)) ∧
    focus_x = 1 ∧
    focus_y = 59 / 20 := 
  sorry

end focus_of_parabola_eq_l561_561565


namespace number_of_divisors_2310_l561_561583

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l561_561583


namespace number_of_positive_divisors_2310_l561_561598

theorem number_of_positive_divisors_2310 : 
  let n := 2310 in
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)] in
  let t := (factorization.map (λ p : ℕ × ℕ, p.snd + 1)).prod in
  t = 32 :=
by
  let n := 2310
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let t := (factorization.map (λ p, p.snd + 1)).prod
  sorry

end number_of_positive_divisors_2310_l561_561598


namespace correct_operations_result_l561_561370

-- Define conditions and the problem statement
theorem correct_operations_result (x : ℝ) (h1: x / 8 - 12 = 18) : (x * 8) * 12 = 23040 :=
by
  sorry

end correct_operations_result_l561_561370


namespace simplify_fraction_l561_561817

theorem simplify_fraction (x : ℝ) (hx : x ≠ 0) : (6 / (5 * x^(-4)) * (5 * x^3) / 3) = 2 * x^7 := by
  sorry

end simplify_fraction_l561_561817


namespace calculate_expression_l561_561520

theorem calculate_expression :
  16 * (1/2) * 4 * (1/16) / 2 = 1 := 
by
  sorry

end calculate_expression_l561_561520


namespace lawrence_work_hours_l561_561316

theorem lawrence_work_hours :
  let hours_mon := 8
  let hours_tue := 8
  let hours_fri := 8
  let hours_wed := 5.5
  let hours_thu := 5.5
  let total_hours := hours_mon * 1 + hours_tue * 1 + hours_fri * 1 + hours_wed * 1 + hours_thu * 1
  total_hours / 7 = 5 :=
by {
  let hours_mon := 8
  let hours_tue := 8
  let hours_fri := 8
  let hours_wed := 5.5
  let hours_thu := 5.5
  let total_hours := hours_mon * 1 + hours_tue * 1 + hours_fri * 1 + hours_wed * 1 + hours_thu * 1
  have total_hours_eq : total_hours = 35, by norm_num,
  rw total_hours_eq,
  norm_num,
}

end lawrence_work_hours_l561_561316


namespace arithmetic_sequence_exists_l561_561665

def f (x : ℝ) : ℝ := x^3 + sin x

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def meets_conditions (a : ℕ → ℝ) : Prop :=
  (∀ i, 1 ≤ i ∧ i ≤ 5 → a i + a (11 - i) = 0) ∧
  (∑ n in Finset.range 10, f (a (n + 1)) = 0)

theorem arithmetic_sequence_exists :
  ∃ (a : ℕ → ℝ) (d : ℝ), d ≠ 0 ∧ is_arithmetic_sequence a d ∧ meets_conditions a :=
sorry

end arithmetic_sequence_exists_l561_561665


namespace minimum_value_l561_561349

theorem minimum_value : 
  ∀ a b : ℝ, 0 < a → 0 < b → a + 2 * b = 3 → (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
by
  sorry

end minimum_value_l561_561349


namespace polygon_sides_eq_eight_l561_561006

theorem polygon_sides_eq_eight (x : ℕ) (h : x ≥ 3) 
  (h1 : 2 * (x - 2) = 180 * (x - 2) / 90) 
  (h2 : ∀ x, x + 2 * (x - 2) = x * (x - 3) / 2) : 
  x = 8 :=
by
  sorry

end polygon_sides_eq_eight_l561_561006


namespace socks_selection_l561_561047

theorem socks_selection :
  ∀ (R Y G B O : ℕ), 
    R = 80 → Y = 70 → G = 50 → B = 60 → O = 40 →
    (∃ k, k = 38 ∧ ∀ (N : ℕ → ℕ), (N R + N Y + N G + N B + N O ≥ k)
          → (exists (pairs : ℕ), pairs ≥ 15 ∧ pairs = (N R / 2) + (N Y / 2) + (N G / 2) + (N B / 2) + (N O / 2) )) :=
by
  sorry

end socks_selection_l561_561047


namespace reciprocal_neg_5_l561_561862

theorem reciprocal_neg_5 : ∃ x : ℚ, -5 * x = 1 ∧ x = -1/5 :=
by
  sorry

end reciprocal_neg_5_l561_561862


namespace volume_of_region_l561_561607

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  ∫∫∫ {xyz : ℝ × ℝ × ℝ | f xyz.1 xyz.2 xyz.3 ≤ 6} 1 = 27/2 := 
sorry

end volume_of_region_l561_561607


namespace time_to_cross_pole_l561_561951

noncomputable theory

def speed_kmh := 60      -- Speed in km/hr
def train_length := 350  -- Length in meters

def speed_ms := (speed_kmh * 1000) / 3600 -- Speed converted to m/s
def crossing_time := train_length / speed_ms -- Time to cross the pole

theorem time_to_cross_pole : crossing_time = 21 := 
by {
  -- Convert speed from km/hr to m/s
  have speed_ms_eq : speed_ms = 50 / 3 := by {
    calc
      speed_ms = (60 * 1000) / 3600 := rfl
            ... = 60000 / 3600 := rfl
            ... = 600 / 36 := rfl
            ... = 50 / 3 := rfl
  },

  -- Use the formula time = distance / speed
  have time_eq : crossing_time = 350 / (50 / 3) := rfl,

  -- Calculate the time to verify it is 21 seconds
  calc
    crossing_time = 350 / (50 / 3) := time_eq
               ... = 350 * (3 / 50) := by rw div_eq_mul_inv
               ... = (350 * 3) / 50 := rfl
               ... = 1050 / 50 := rfl
               ... = 21 := by norm_num
}

#print time_to_cross_pole

end time_to_cross_pole_l561_561951


namespace range_of_a_l561_561641

noncomputable def piecewise_function (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2 - a * x else (1 / 3) * x^3 - (3 / 2) * a * x^2 + (2 * a^2 + 2) * x - (11 / 6)

theorem range_of_a {a : ℝ} :
  (∀ x1 x2 : ℝ, x1 < x2 → piecewise_function a x1 - piecewise_function a x2 < 2 * x1 - 2 * x2) →
  a < -2 :=
sorry

end range_of_a_l561_561641


namespace fold_triangle_length_DE_l561_561827

-- Define the problem and conditions: triangle ABC with base 15 cm and the fold properties
def triangle_base_length (ABC : Triangle) : ℝ := 15
def is_folded_over_base (ABC XYZ : Triangle) : Prop := XYZ.base = (0.5 : ℝ) * ABC.base

-- Given the condition of areas
def correct_area_ratio (ABC XYZ : Triangle) : Prop := XYZ.area = 0.25 * ABC.area

-- Define what needs to be proven: the length of DE given the conditions
def length_of_DE (ABC XYZ : Triangle) (DE : ℝ) : Prop :=
  is_folded_over_base ABC XYZ ∧ correct_area_ratio ABC XYZ ∧ DE = 7.5

-- Main theorem statement
theorem fold_triangle_length_DE (ABC XYZ : Triangle) (DE : ℝ) 
  (h₁ : triangle_base_length ABC = 15)
  (h₂ : is_folded_over_base ABC XYZ)
  (h₃ : correct_area_ratio ABC XYZ) :
  length_of_DE ABC XYZ DE :=
sorry

end fold_triangle_length_DE_l561_561827


namespace conditional_probability_l561_561432

def sample_space := {1, 2, 3, 4, 5, 6}
def event_A := {1, 3, 5}
def event_B := {3}
def intersection_AB := event_A ∩ event_B

def P_event (s : Finset ℕ) : ℚ := (s.card : ℚ) / (sample_space.card : ℚ)

theorem conditional_probability :
  P_event intersection_AB / P_event event_A = 1/3 := by sorry

end conditional_probability_l561_561432


namespace find_interval_for_a_l561_561107

-- Define the system of equations as a predicate
def system_of_equations (a x y z : ℝ) : Prop := 
  x + y + z = 0 ∧ x * y + y * z + a * z * x = 0

-- Define the condition that (0, 0, 0) is the only solution
def unique_solution (a : ℝ) : Prop :=
  ∀ x y z : ℝ, system_of_equations a x y z → x = 0 ∧ y = 0 ∧ z = 0

-- Rewrite the proof problem as a Lean statement
theorem find_interval_for_a :
  ∀ a : ℝ, unique_solution a ↔ 0 < a ∧ a < 4 :=
by
  sorry

end find_interval_for_a_l561_561107


namespace parabola_vertex_focus_l561_561381

/-- Suppose that on a parabola with vertex \( V \) and focus \( F \) there exists a point \( A \) such that \( AF = 25 \) and \( AV = 26 \). Prove that the sum of all possible values of the length \( FV \) is equal to 50/3.  -/
theorem parabola_vertex_focus (A V F : ℝ × ℝ) (hAF : dist A F = 25) (hAV : dist A V = 26) : 
    let FV := dist F V in FV = 50 / 3 :=
sorry

end parabola_vertex_focus_l561_561381


namespace count_integers_between_sqrt8_and_sqrt72_l561_561209

theorem count_integers_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℕ, 3 ≤ x ∧ x ≤ 8 → x ∈ finset.range (8 - 3 + 1) :=
by
  let a := real.sqrt 8
  let b := real.sqrt 72
  have ha : 3 = nat_ceil a := sorry
  have hb : 8 = nat_floor b := sorry
  use 6
  split
  {
    exact sorry
  }
  {
    intros x hx
    have h_valid : 3 ≤ x ∧ x ≤ 8 := hx
    rw finset.mem_range
    sorry
  }

end count_integers_between_sqrt8_and_sqrt72_l561_561209


namespace bases_with_final_digit_five_l561_561135

def count_bases := ∀ (b : ℕ), 3 ≤ b ∧ b ≤ 10 → 620 % b = 0

theorem bases_with_final_digit_five :
  (Finset.filter (λ b => count_bases b) (Finset.Icc 3 10)).card = 2 :=
by
  sorry

end bases_with_final_digit_five_l561_561135


namespace find_x_l561_561379

theorem find_x (x : ℝ) (a b : ℝ) (h₀ : a * b = 4 * a - 2 * b)
  (h₁ : 3 * (6 * x) = -2) :
  x = 17 / 2 :=
by
  sorry

end find_x_l561_561379


namespace part_a_part_b_part_c_l561_561909

-- Part (a) Lean Statement
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ k : ℝ, k = 2 * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (b) Lean Statement
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ q : ℝ, q = 1 - p ∧ ∃ r : ℝ, r = 2 * p / (2 * p + (1 - p) ^ 2)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (c) Lean Statement
theorem part_c (N : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ S : ℝ, S = N * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

end part_a_part_b_part_c_l561_561909


namespace union_of_A_and_B_l561_561462

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem union_of_A_and_B :
  A ∪ B = {-1, 0, 1, 2, 4} :=
by
  sorry

end union_of_A_and_B_l561_561462


namespace lunks_for_apples_l561_561686

theorem lunks_for_apples : ∀ (lun_to_kun : ℕ) (num_lun : ℕ) (num_kun : ℕ) (kun_to_app : ℕ) (num_kun2 : ℕ) (num_app : ℕ),
  lun_to_kun = 7 ∧ num_kun = 4 ∧ kun_to_app = 3 ∧ num_kun2 = 5 ∧ num_app = 24 → 
  ((num_app * kun_to_app * num_lun / (num_kun2 * lun_to_kun)) ≤ 27) :=
by
  intros lun_to_kun num_lun num_kun kun_to_app num_kun2 num_app
  assume h_conditions
  sorry

end lunks_for_apples_l561_561686


namespace period_translation_symmetry_l561_561181

theorem period_translation_symmetry (f : ℝ → ℝ) (ω : ℝ) (h1 : ω > 0)
  (h2 : ∃ p > 0, ∀ x, f(x + p) = f x) 
  (h3 : ∀ x, f x = real.sin (ω * x + (π / 6))) :
  (∀ x, f(x - π / 3) = real.sin(2 * x)) → 
  (∀ x, real.sin(2 * x) = -real.sin(-2 * x)) :=
by 
  sorry

end period_translation_symmetry_l561_561181


namespace problem_statement_l561_561338

-- Definitions of the propositions
def p : Prop := k = 0
def q : Prop := ∃! x y : ℝ, y = k * x + 1 ∧ y ^ 2 = 4 * x

-- Theorem stating the proof problem
theorem problem_statement : (p → q) ∧ ¬(p → ¬q) :=
by
  sorry

end problem_statement_l561_561338


namespace perpendicular_bisector_value_of_b_l561_561843

theorem perpendicular_bisector_value_of_b :
  (∃ b : ℝ, (∀ p : ℝ × ℝ, p = (4, 6) → (∃ line : ℝ → ℝ → ℝ, line = (λ x y, x + y) ∧ line (fst p) (snd p) = b)) → b = 10) :=
begin
  sorry
end

end perpendicular_bisector_value_of_b_l561_561843


namespace chord_length_example_l561_561566

noncomputable def chord_length (r : ℝ) (d : ℝ) : ℝ := 2 * real.sqrt(r^2 - d^2)

theorem chord_length_example :
  let circle_eq := λ x y : ℝ, x^2 + (y - 2)^2 = 4,
      line_eq := λ x y : ℝ, y = -real.sqrt(3) * x,
      center := (0, 2),
      r := 2,
      d := 1
  in chord_length r d = 2 * real.sqrt(3) :=
by
  intros
  rw [chord_length]
  rw [pow_two, pow_two]
  rw [sub_self, zero_add]
  sorry

end chord_length_example_l561_561566


namespace number_of_children_is_six_l561_561387

variables (A V S : ℕ) (n : ℕ)

-- Conditions from the problem
def condition_1 := if Anya gives half of her mushrooms to Vitya, all children will have the same number of mushrooms.
def condition_2 := if Anya gives all her mushrooms to Sasha, Sasha will have as many mushrooms as all the others combined.

-- Proof Statement
theorem number_of_children_is_six (h1 : (A/2 + V + (n-3) * (A/2)) = n * (A/2)) 
           (h2 : S + A = (n - 1) * (A/2)) : 
           n = 6 :=
sorry

end number_of_children_is_six_l561_561387


namespace percentage_score_70_79_l561_561929

theorem percentage_score_70_79 (f : ℕ → ℕ) :
  f 90 = 3 →
  f 80 = 5 →
  f 70 = 8 →
  f 60 = 4 →
  f 50 = 1 →
  f 0 = 3 →
  (8 / (3 + 5 + 8 + 4 + 1 + 3) * 100) = 33.33 :=
by
  intros h90 h80 h70 h60 h50 h0
  have total : 3 + 5 + 8 + 4 + 1 + 3 = 24 := by simp
  have fact : 8 / 24 * 100 = 33.33 := sorry
  exact fact

end percentage_score_70_79_l561_561929


namespace largest_prime_factor_of_4752_l561_561003

theorem largest_prime_factor_of_4752 : ∃ p : ℕ, nat.prime p ∧ p ∣ 4752 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 4752 → q ≤ p :=
begin
  -- Proof goes here
  sorry
end

end largest_prime_factor_of_4752_l561_561003


namespace base8_to_base10_l561_561538

theorem base8_to_base10 :
  ∀ (n : ℕ), (n = 2 * 8^2 + 4 * 8^1 + 3 * 8^0) → (n = 163) :=
by
  intros n hn
  sorry

end base8_to_base10_l561_561538


namespace range_of_a_l561_561335

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then
  x - Real.exp x + 2
else
  (1 / 3) * x^3 - 4 * x + a

theorem range_of_a (a : ℝ) : (∃! x, f x a = 0) ↔ a > 16 / 3 :=
by
  sorry

end range_of_a_l561_561335


namespace percentage_error_l561_561455

theorem percentage_error (x : ℝ) : ((x * 3 - x / 5) / (x * 3) * 100) = 93.33 := 
  sorry

end percentage_error_l561_561455


namespace largest_lambda_l561_561136

noncomputable theory
open Real

def conditions (n : ℕ) (α : Fin n → ℝ) : Prop :=
  n ≥ 2 ∧ (∀ i, 0 < α i ∧ α i < π / 2) ∧
  3 * ((∑ i, (tan (α i))^2) * (∑ i, (cot (α i))^2)) +
  11 * ((∑ i, (sin (α i))^2) * (∑ i, (csc (α i))^2)) +
  11 * ((∑ i, (cos (α i))^2) * (∑ i, (sec (α i))^2)) ≥
  25 * (∑ i, sin (α i)) ^ 2 + 25 * (∑ i, cos (α i)) ^ 2 + 
  λ (α 0 - α (n - 1)) ^ 2

theorem largest_lambda (n : ℕ) (α : Fin n → ℝ) : conditions n α → λ n = 25 :=
by
  sorry

end largest_lambda_l561_561136


namespace routes_in_3x3_grid_l561_561534

theorem routes_in_3x3_grid : 
  let grid_size := 3 in
  (choose (2 * grid_size) grid_size) = 20 :=
by
  let grid_size := 3
  sorry

end routes_in_3x3_grid_l561_561534


namespace interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l561_561667

noncomputable def f (x k : ℝ) : ℝ := Real.log x - k * x + 1

theorem interval_increase_for_k_eq_2 :
  ∃ k : ℝ, k = 2 → 
  ∃ a b : ℝ, 0 < b ∧ b = 1 / 2 ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → (Real.log x - 2 * x + 1 < Real.log x - 2 * x + 1)) := 
sorry

theorem range_of_k_if_f_leq_0 :
  ∀ (k : ℝ), (∀ x : ℝ, 0 < x → Real.log x - k * x + 1 ≤ 0) →
  ∃ k_min : ℝ, k_min = 1 ∧ k ≥ k_min :=
sorry

end interval_increase_for_k_eq_2_range_of_k_if_f_leq_0_l561_561667


namespace inequality_l561_561495

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1/2 ∧ ∀ k, a (k + 1) = -a k + 1/(2 - a k)

theorem inequality
  (a : ℕ → ℝ)
  (h : sequence a) :
  ∀ n : ℕ,
  (n ≥ 1) →
  ( (n / (2 * (∑ i in finset.range n, a i)) - 1)^n ≤ 
    (∑ i in finset.range n, a i / n)^n *
    ∏ i in finset.range n, (1 / a i - 1) ) :=
by
  sorry

end inequality_l561_561495


namespace sum_of_eight_numbers_l561_561274

theorem sum_of_eight_numbers (avg : ℚ) (n : ℕ) (sum : ℚ) 
  (h_avg : avg = 5.3) (h_n : n = 8) : sum = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l561_561274


namespace sector_triangle_perimeter_l561_561942

noncomputable def perimeter_combined_sector_triangle
  (θ : ℝ) (r : ℝ) (arc_length : ℝ) (chord_length : ℝ) : ℝ :=
  2 * r + arc_length + chord_length

theorem sector_triangle_perimeter (hθ : θ = 120)
  (hr : r = 4.8)
  (harc : arc_length = (2 * Real.pi * r * θ / 360))
  (hchord : chord_length = 2 * r * Real.sin (θ / 2 * Real.pi / 180)) :
  perimeter_combined_sector_triangle θ r arc_length chord_length ≈ 27.97 :=
by
  sorry

end sector_triangle_perimeter_l561_561942


namespace remainder_1493824_div_4_l561_561008

theorem remainder_1493824_div_4 : 1493824 % 4 = 0 :=
by
  sorry

end remainder_1493824_div_4_l561_561008


namespace max_value_m_l561_561148

variable {a b m : ℝ}

theorem max_value_m (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
by 
  sorry

end max_value_m_l561_561148


namespace time_between_four_and_five_straight_line_l561_561514

theorem time_between_four_and_five_straight_line :
  ∃ t : ℚ, t = 21 + 9/11 ∨ t = 54 + 6/11 :=
by
  sorry

end time_between_four_and_five_straight_line_l561_561514


namespace external_angle_theorem_proof_l561_561750

theorem external_angle_theorem_proof
    (x : ℝ)
    (FAB : ℝ)
    (BCA : ℝ)
    (ABC : ℝ)
    (h1 : FAB = 70)
    (h2 : BCA = 20 + x)
    (h3 : ABC = x + 20)
    (h4 : FAB = ABC + BCA) : 
    x = 15 :=
  by
  sorry

end external_angle_theorem_proof_l561_561750


namespace trig_identity_solution_l561_561450

theorem trig_identity_solution (z : ℂ) (h : (Real.cos z)^3 * Real.cos (3 * z) + (Real.sin z)^3 * Real.sin (3 * z) = Real.sqrt 2 / 4) :
  ∃ k : ℤ, z = (Real.pi / 8) * (8 * k + 1) ∨ z = (Real.pi / 8) * (8 * k - 1) :=
sorry

end trig_identity_solution_l561_561450


namespace magnitude_of_angle_A_and_area_of_triangle_l561_561735

/-- Define vectors m and n -/
def vector_m (A : ℝ) : ℝ × ℝ := (1 / 2, Real.cos A)

def vector_n (A : ℝ) : ℝ × ℝ := (Real.sin A, -Real.sqrt 3 / 2)

/-- m is perpendicular to n -/
def m_perp_n (A : ℝ) : Prop :=
  let m := vector_m A
  let n := vector_n A
  m.1 * n.1 + m.2 * n.2 = 0

/-- Main theorem statement -/
theorem magnitude_of_angle_A_and_area_of_triangle (a b : ℝ) (ha : a = 7) (hb : b = 8) (A : ℝ) 
    (hA : A = 60 * Real.pi / 180) :
  m_perp_n A →
  A = 60 * Real.pi / 180 ∧
  let area := (1 / 2) * a * b * Real.sin A in
  area = 10 * Real.sqrt 3 :=
begin
  sorry
end

end magnitude_of_angle_A_and_area_of_triangle_l561_561735


namespace lunks_needed_for_24_apples_l561_561702

-- Define the conditions as Lean definitions
def lunks_per_kunks := 7 / 4
def kunks_per_apples := 3 / 5
def apples_needed := 24

-- State the theorem
theorem lunks_needed_for_24_apples : 
  let k := (3 * apples_needed) / 5 in 
  let rounded_k := k.ceil in 
  let l := (7 * rounded_k) / 4 in 
  l.ceil = 27 :=
by 
  let k := (3 * apples_needed) / 5
  let rounded_k := k.ceil
  let l := (7 * rounded_k) / 4
  have h1 : k = (3 * apples_needed) / 5 := rfl
  have h2 : rounded_k = k.ceil := rfl
  have h3 : l = (7 * rounded_k) / 4 := rfl
  show l.ceil = 27, from sorry

end lunks_needed_for_24_apples_l561_561702


namespace fruit_basket_cost_is_28_l561_561477

def basket_total_cost : ℕ := 4 * 1 + 3 * 2 + (24 / 12) * 4 + 2 * 3 + 2 * 2

theorem fruit_basket_cost_is_28 : basket_total_cost = 28 := by
  sorry

end fruit_basket_cost_is_28_l561_561477


namespace complement_union_l561_561195

def is_pos_int_less_than_9 (x : ℕ) : Prop := x > 0 ∧ x < 9

def U : Set ℕ := {x | is_pos_int_less_than_9 x}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union :
  (U \ (M ∪ N)) = {2, 4, 8} :=
by
  sorry

end complement_union_l561_561195


namespace pyramid_volume_l561_561403

-- Define the conditions
def height_vertex_to_center_base := 12 -- cm
def side_of_square_base := 10 -- cm
def base_area := side_of_square_base * side_of_square_base -- cm²
def volume := (1 / 3) * base_area * height_vertex_to_center_base -- cm³

-- State the theorem
theorem pyramid_volume : volume = 400 := 
by
  -- Placeholder for the proof
  sorry

end pyramid_volume_l561_561403


namespace series_irrational_l561_561364

theorem series_irrational (g : ℤ) (hg : g ≥ 2) : 
  (∑' n : ℕ, 1 / (g : ℚ)^(n^2)) ∉ ℚ ∧ (∑' n : ℕ, 1 / (g : ℚ)^(factorial n)) ∉ ℚ :=
sorry

end series_irrational_l561_561364


namespace probability_of_multiple_l561_561506

open Nat

-- Define the conditions
def whole_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 12}
def valid_square_numbers := {n : ℕ | n = 1 ∨ n = 4 ∨ n = 9}
def total_assignments := 12 * 11 * 10

-- Function to check if Al's number is a multiple of both Bill's and Cal's numbers
def is_multiple (al b c : ℕ) : Prop := al % b = 0 ∧ al % c = 0

noncomputable def valid_assignments : ℕ :=
  (if is_multiple 4 1 2 then 2 else 0) +
  (if is_multiple 4 2 1 then 2 else 0) +
  (if is_multiple 9 1 3 then 2 else 0) +
  (if is_multiple 9 3 1 then 2 else 0)

-- Probability calculation
def probability := valid_assignments.toRat / total_assignments.toRat

-- The actual proof statement
theorem probability_of_multiple : 
  probability = (1 : ℚ) / 330 :=
by
  sorry

end probability_of_multiple_l561_561506


namespace original_cost_prices_in_USD_l561_561510

-- Define the exchange rates
def exchange_rate_GBP_to_USD := 1.38
def exchange_rate_CHF_to_USD := 1.08
def exchange_rate_JPY_to_USD := 0.0091

-- Define the selling prices
def selling_price_book_GBP := 290
def selling_price_watch_CHF := 520
def selling_price_headphones_JPY := 15000

-- Define the profit margins
def profit_margin_book := 0.20
def profit_margin_watch := 0.30
def profit_margin_headphones := 0.15

-- Calculate the original cost prices in the local currencies
def original_cost_price_book_GBP := selling_price_book_GBP / (1 + profit_margin_book)
def original_cost_price_watch_CHF := selling_price_watch_CHF / (1 + profit_margin_watch)
def original_cost_price_headphones_JPY := selling_price_headphones_JPY / (1 + profit_margin_headphones)

-- Convert the original cost prices to USD
def original_cost_price_book_USD := original_cost_price_book_GBP * exchange_rate_GBP_to_USD
def original_cost_price_watch_USD := original_cost_price_watch_CHF * exchange_rate_CHF_to_USD
def original_cost_price_headphones_USD := original_cost_price_headphones_JPY * exchange_rate_JPY_to_USD

-- The expected original cost prices in USD
def expected_cost_price_book_USD := 333.70
def expected_cost_price_watch_USD := 432
def expected_cost_price_headphones_USD := 118.60

-- Prove that the calculated original cost prices match the expected values
theorem original_cost_prices_in_USD :
  original_cost_price_book_USD = expected_cost_price_book_USD ∧
  original_cost_price_watch_USD = expected_cost_price_watch_USD ∧
  original_cost_price_headphones_USD = expected_cost_price_headphones_USD :=
by {
  sorry
}

end original_cost_prices_in_USD_l561_561510


namespace arithmetic_sequence_problem_l561_561156

variables {a : ℕ → ℕ} (d a1 : ℕ)

def arithmetic_sequence (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : arithmetic_sequence 1 + arithmetic_sequence 3 + arithmetic_sequence 9 = 20) :
  4 * arithmetic_sequence 5 - arithmetic_sequence 7 = 20 :=
by
  sorry

end arithmetic_sequence_problem_l561_561156


namespace angle_ARC_obtuse_l561_561755

theorem angle_ARC_obtuse
  (A B C P Q R D I: Point)
  (triangle_ABC : Triangle A B C)
  (incircle_touches_AB_at_P : IsTangencyPoint (incircle triangle_ABC) AB P)
  (incircle_touches_BC_at_Q : IsTangencyPoint (incircle triangle_ABC) BC Q)
  (median_B_AC_intersects_PQ_at_R : IsMedian B AC R ∧ OnLine R PQ)
  (collinear_D_I_R: Collinear D I R)
  (circle_with_diameter_AC : CirclePassingThroughDiameter A C)
  (X Y: Point)
  (AI_intersects_PQ_at_X : Intersect (LineThroughPoints A I) PQ X)
  (CI_intersects_PQ_at_Y : Intersect (LineThroughPoints C I) PQ Y)
  (R_between_X_Y : Between R X Y)
  : angle A R C > 90° := sorry

end angle_ARC_obtuse_l561_561755


namespace compare_y1_y2_l561_561803

def parabola (x : ℝ) (c : ℝ) : ℝ := -x^2 + 4 * x + c

theorem compare_y1_y2 (c y1 y2 : ℝ) :
  parabola (-1) c = y1 →
  parabola 1 c = y2 →
  y1 < y2 :=
by
  intro h1 h2
  sorry

end compare_y1_y2_l561_561803


namespace fraction_zero_solution_l561_561280

theorem fraction_zero_solution (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := 
by sorry

end fraction_zero_solution_l561_561280


namespace solve_matrix_eq_l561_561320

noncomputable def matrix_B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 3], ![2, 1, 2], ![3, 2, 1]]

noncomputable def matrix_I : Matrix (Fin 3) (Fin 3) ℚ :=
  Matrix.eye 3

noncomputable def matrix_Z : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 0, 0], ![0, 0, 0], ![0, 0, 0]]

theorem solve_matrix_eq :
  ∃ (a b c : ℚ), (matrix_B^3 + a • matrix_B^2 + b • matrix_B + c • matrix_I = matrix_Z)
  ∧ a = 0 ∧ b = -283/13 ∧ c = 902/13 := sorry

end solve_matrix_eq_l561_561320


namespace solve_fractional_equation_l561_561420

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) :
  (2 / x = 3 / (x + 1)) → (x = 2) :=
by
  -- Proof will be filled in here
  sorry

end solve_fractional_equation_l561_561420


namespace find_min_value_l561_561123

noncomputable def problem (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧
  (27^x + y^4 - 3^x - 1 = 0)

theorem find_min_value :
  ∃ x y : ℝ, problem x y ∧ 
  (∀ (x' y' : ℝ), problem x' y' → (x^3 + y^3) ≤ (x'^3 + y'^3)) ∧ (x^3 + y^3 = -1) := 
sorry

end find_min_value_l561_561123


namespace A_receives_more_than_B_l561_561921

variable (A B C : ℝ)

axiom h₁ : A = 1/3 * (B + C)
axiom h₂ : B = 2/7 * (A + C)
axiom h₃ : A + B + C = 720

theorem A_receives_more_than_B : A - B = 20 :=
by
  sorry

end A_receives_more_than_B_l561_561921


namespace shoes_sold_last_week_l561_561793

theorem shoes_sold_last_week (total_target : ℕ) (sold_this_week : ℕ) (needed_more : ℕ) (sold_so_far : ℕ):
  total_target = 80 → sold_this_week = 12 → needed_more = 41 → sold_so_far = total_target - needed_more →
  sold_so_far - sold_this_week = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end shoes_sold_last_week_l561_561793


namespace number_of_divisors_2310_l561_561586

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l561_561586


namespace find_length_of_ej_l561_561802

noncomputable def side_length_of_square (area : ℕ) := 
  let s := real.sqrt (area)
  s

noncomputable def length_ej (s : ℝ) (area_triangle_HJG : ℝ) : ℝ := 
  let HG_HJ := real.sqrt (2 * area_triangle_HJG)
  let EJ := real.sqrt (s^2 + HG_HJ^2)
  EJ

theorem find_length_of_ej :
  ∀ (area_square : ℕ) (area_triangle_HJG : ℝ),
  area_square = 144 →
  area_triangle_HJG = 90 →
  let s := side_length_of_square area_square in
  let EJ := length_ej s area_triangle_HJG in
  EJ = 18 := 
by
  intros
  -- Proof skipped
  sorry

end find_length_of_ej_l561_561802


namespace find_x_condition_l561_561984

-- Definitions based on conditions
def first_set : List ℝ := [12, 32, 56, 78, 91]
def second_set (x : ℝ) : List ℝ := [7, 47, 67, 105, x]

-- Mean calculation for a list of real numbers
def mean (l : List ℝ) : ℝ := l.sum / l.length

-- Lean 4 statement for the proof
theorem find_x_condition :
  ∃ x : ℝ, (mean first_set) = (mean (second_set x)) + 10 :=
sorry

end find_x_condition_l561_561984


namespace exterior_angle_of_regular_octagon_l561_561753

theorem exterior_angle_of_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ (sum_angles : ℕ), (sum_angles = (n - 2) * 180) → 
  ∀ (interior_angle : ℕ), (interior_angle = sum_angles / n) → 
  ∀ (exterior_angle : ℕ), (exterior_angle = 180 - interior_angle) → 
  exterior_angle = 45 :=
by
  intros n hn sumA hsumA intA hintA extA hextA
  rw [hn, hsumA, hintA, hextA]
  sorry

end exterior_angle_of_regular_octagon_l561_561753


namespace resulting_polygon_properties_l561_561197

-- Definitions for vertices and configurations of the triangles
variables {A B C D E F O X Y Z : Type*}
variables 
  [Inhabited A] [Inhabited B] [Inhabited C] 
  [Inhabited D] [Inhabited E] [Inhabited F] 
  [Inhabited O] [Inhabited X] [Inhabited Y] [Inhabited Z]

-- Points X and Y are in triangles ABC and DEF, respectively
variables (X_in_ABC : A → B → C → X)
variables (Y_in_DEF : D → E → F → Y)

-- Proof Problem
theorem resulting_polygon_properties
  (triangle_ABC : A → B → C → Type*)
  (triangle_DEF : D → E → F → Type*)
  (OXYZ : O → X → Y → Z → Prop)
  (exist_parallelogram : ∀ (X Y : Type*), exists Z, OXYZ O X Y Z)
  (perimeter_triangle_ABC : ℝ)
  (perimeter_triangle_DEF : ℝ) :
  -- Resulting shape is a polygon
  (∃ P : Type*, is_polygon P ∧ (∃ n, n = 3 ∨ n ≤ 6)) ∧
  -- The number of sides
  (side_count P = 3 ∨ side_count P = 6) ∧
  -- The perimeter
  (perimeter P = perimeter_triangle_ABC + perimeter_triangle_DEF) :=
sorry

end resulting_polygon_properties_l561_561197


namespace simplify_expression_l561_561377

variable (a b : ℝ)

theorem simplify_expression (a b : ℝ) :
  (6 * a^5 * b^2) / (3 * a^3 * b^2) + ((2 * a * b^3)^2) / ((-b^2)^3) = -2 * a^2 :=
by 
  sorry

end simplify_expression_l561_561377


namespace percentage_x_is_10_percent_l561_561943

-- Definitions of the given conditions
def volume_x := 200 -- milliliters
def volume_y := 50 -- milliliters
def percentage_y := 0.30 -- 30% alcohol by volume in solution y
def desired_percentage := 0.14 -- 14% desired alcohol by volume
def total_volume := volume_x + volume_y -- Total volume after mixing

-- The goal is to prove P
theorem percentage_x_is_10_percent (P : ℝ) 
  (h1 : total_volume = 250)
  (h2 : desired_percentage * total_volume = (volume_x * P + volume_y * percentage_y)) :
  P = 0.10 := by
  -- Given conditions
  have h3 : total_volume = volume_x + volume_y := by sorry
  have h4 : volume_y * percentage_y = 15 := by sorry
  have h5 : desired_percentage * total_volume = 35 := by sorry

  -- Main proof (this part will contain all step proofs, hence sorry is added here)
  sorry

end percentage_x_is_10_percent_l561_561943


namespace tangents_sum_eq_l561_561082

noncomputable def f (x : ℝ) : ℝ := max (-8 * x - 29) (max (3 * x + 2) (7 * x - 4))

theorem tangents_sum_eq (q : ℝ → ℝ) (a1 a2 a3 : ℝ) (h1 : ∀ x, q x = f x → x ∈ {a1, a2, a3}) : 
  q a1 = f a1 ∧ q a2 = f a2 ∧ q a3 = f a3 ∧ a1 + a2 + a3 = -163 / 22 :=
sorry

end tangents_sum_eq_l561_561082


namespace number_of_divisors_of_2310_l561_561571

theorem number_of_divisors_of_2310 : 
  let n := 2310 in 
  let prime_factors := [2, 3, 5, 7, 11] in
  ∃ k : ℕ, k = prime_factors.length ∧
  (∀ i, i < k → prime_factors.nth i = some 2 ∨ prime_factors.nth i = some 3 ∨ prime_factors.nth i = some 5 ∨ prime_factors.nth i = some 7 ∨ prime_factors.nth i = some 11) →
  (n.factorization.to_nat * 1).0 = 32 :=
begin
  sorry
end

end number_of_divisors_of_2310_l561_561571


namespace count_integers_between_sqrt8_sqrt72_l561_561248

-- Define the square roots of 8 and 72 for reference
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt(8)
def minIntAboveSqrt8 : ℤ := Int.ceil sqrt8

-- Define the largest integer less than sqrt(72)
def maxIntBelowSqrt72 : ℤ := Int.floor sqrt72

-- State the main theorem to prove
theorem count_integers_between_sqrt8_sqrt72 :
  minIntAboveSqrt8 = 3 ∧ maxIntBelowSqrt72 = 8 → 
  (maxIntBelowSqrt72 - minIntAboveSqrt8 + 1) = 6 :=
by
  sorry

end count_integers_between_sqrt8_sqrt72_l561_561248


namespace choose_president_vp_and_committee_l561_561741

theorem choose_president_vp_and_committee :
  ∃ (n : ℕ) (k : ℕ), n = 10 ∧ k = 3 ∧ 
  let ways_to_choose_president := 10 in
  let ways_to_choose_vp := 9 in
  let ways_to_choose_committee := (Nat.choose 8 3) in
  ways_to_choose_president * ways_to_choose_vp * ways_to_choose_committee = 5040 :=
begin
  use [10, 3],
  simp [Nat.choose],
  sorry
end

end choose_president_vp_and_committee_l561_561741


namespace minimize_travel_time_l561_561072

-- Definitions and conditions
def grid_size : ℕ := 7
def mid_point : ℕ := (grid_size + 1) / 2
def is_meeting_point (p : ℕ × ℕ) : Prop := 
  p = (mid_point, mid_point)

-- Main theorem statement to be proven
theorem minimize_travel_time : 
  ∃ (p : ℕ × ℕ), is_meeting_point p ∧
  (∀ (q : ℕ × ℕ), is_meeting_point q → p = q) :=
sorry

end minimize_travel_time_l561_561072


namespace sufficient_not_necessary_l561_561147

variables {a β : Type} [Plane a] [Plane β] {l : Line}

-- Hypothesis: a and β are different planes, and l is a line within plane a.
axiom planes_different : a ≠ β
axiom line_within_plane : l ∈ a

-- Goal: prove that a ∥ β implies l ∥ β but not the converse.
theorem sufficient_not_necessary (h_parallel_planes : a ∥ β) : l ∥ β ∧ ¬ (l ∥ β → a ∥ β) :=
by sorry

end sufficient_not_necessary_l561_561147


namespace range_of_a_line_not_passing_second_quadrant_l561_561835

theorem range_of_a_line_not_passing_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), ((a - 2) * y = (3a - 1) * x - 1) →
    ¬(x < 0 ∧ y > 0)) → 
  a ≥ 2 :=
by
  sorry

end range_of_a_line_not_passing_second_quadrant_l561_561835


namespace find_smaller_circle_radius_l561_561636

noncomputable def smaller_circle_radius (R : ℝ) : ℝ :=
  R / (Real.sqrt 2 - 1)

theorem find_smaller_circle_radius (R : ℝ) (x : ℝ) :
  (∀ (c1 c2 c3 c4 : ℝ),  c1 = c2 ∧ c2 = c3 ∧ c3 = c4 ∧ c4 = x
  ∧ c1 + c2 = 2 * c3 * Real.sqrt 2)
  → x = smaller_circle_radius R :=
by 
  intros h
  sorry

end find_smaller_circle_radius_l561_561636


namespace market_value_of_stock_l561_561913

-- Define the given conditions.
def face_value : ℝ := 100
def dividend_per_share : ℝ := 0.09 * face_value
def yield : ℝ := 0.08

-- State the problem: proving the market value of the stock.
theorem market_value_of_stock : (dividend_per_share / yield) * 100 = 112.50 := by
  -- Placeholder for the proof
  sorry

end market_value_of_stock_l561_561913


namespace rs_division_l561_561017

theorem rs_division (a b c : ℝ) 
  (h1 : a = 1 / 2 * b)
  (h2 : b = 1 / 2 * c)
  (h3 : a + b + c = 700) : 
  c = 400 :=
sorry

end rs_division_l561_561017


namespace slope_of_line_l561_561618

-- Defining the conditions
def intersects_on_line (s x y : ℝ) : Prop :=
  (2 * x + 3 * y = 8 * s + 6) ∧ (x + 2 * y = 5 * s - 1)

-- Theorem stating that the slope of the line on which all intersections lie is 2
theorem slope_of_line {s x y : ℝ} :
  (∃ s x y, intersects_on_line s x y) → (∃ (m : ℝ), m = 2) :=
by sorry

end slope_of_line_l561_561618


namespace eq_solution_l561_561423

theorem eq_solution (x : ℝ) (h : 2 / x = 3 / (x + 1)) : x = 2 :=
by
  sorry

end eq_solution_l561_561423


namespace find_radius_closest_tenth_l561_561491

noncomputable def lattice_point_probability (d : ℝ) : ℝ := π * d^2

theorem find_radius_closest_tenth
  (square_area : ℝ)
  (probability : ℝ)
  (lattice_point_probability_eq : lattice_point_probability d = 1 / 4)
  (square_area_eq : square_area = 1000000)
  (probability_eq : probability = 1 / 4) :
  (d ≈ 0.3) :=
by
  have : π * d^2 = 1 / 4, from lattice_point_probability_eq
  sorry

end find_radius_closest_tenth_l561_561491


namespace infinitely_many_n_l561_561373

theorem infinitely_many_n (a b c : ℤ):
  ∃ᶠ n in at_top, ∃ k m n : ℤ, n = 2 ^ (30 * k + 15) * 3 ^ (30 * m + 20) * 5 ^ (30 * n + 24) ∧
    (2 * n) = (2 * 2 ^ (30 * k + 15) * 3 ^ (30 * m + 20) * 5 ^ (30 * n + 24)) ∧
    (3 * n) = (3 * 2 ^ (30 * k + 15) * 3 ^ (30 * m + 20) * 5 ^ (30 * n + 24)) ∧
    (5 * n) = (5 * 2 ^ (30 * k + 15) * 3 ^ (30 * m + 20) * 5 ^ (30 * n + 24)) ∧
    Nat.sqrt (2 * n) ^ 2 = 2 * n ∧
    Nat.cbrt (3 * n) ^ 3 = 3 * n ∧
    (5 * n) ^ 5 = 5 * n :=
by sorry

end infinitely_many_n_l561_561373


namespace length_of_train_l561_561503

-- Given conditions:
-- 1. The train's speed is 45 km/hr (convertible to m/s).
-- 2. The train crosses the bridge in 30 seconds.
-- 3. The length of the bridge is 255 meters.

theorem length_of_train
    (train_speed_kmph : ℕ) (crossing_time_sec : ℕ) (bridge_length_meters : ℕ)
    (h1 : train_speed_kmph = 45) (h2 : crossing_time_sec = 30) (h3 : bridge_length_meters = 255) :
    let train_speed_mps := (train_speed_kmph * 1000) / 3600,
        total_distance := train_speed_mps * crossing_time_sec,
        train_length := total_distance - bridge_length_meters in
    train_length = 120 := by
  sorry

end length_of_train_l561_561503


namespace subtract_square_l561_561260

theorem subtract_square (n : ℝ) (h : n = 68.70953354520753) : (n^2 - 20^2) = 4321.000000000001 := by
  sorry

end subtract_square_l561_561260


namespace probability_sum_3_correct_l561_561369

noncomputable def probability_of_sum_3 : ℚ := 2 / 36

theorem probability_sum_3_correct :
  probability_of_sum_3 = 1 / 18 :=
by
  sorry

end probability_sum_3_correct_l561_561369


namespace weeks_per_month_l561_561924

-- Define the given conditions
def num_employees_initial : Nat := 500
def additional_employees : Nat := 200
def hourly_wage : Nat := 12
def daily_work_hours : Nat := 10
def weekly_work_days : Nat := 5
def total_monthly_pay : Nat := 1680000

-- Calculate the total number of employees after hiring
def total_employees : Nat := num_employees_initial + additional_employees

-- Calculate the pay rates
def daily_pay_per_employee : Nat := hourly_wage * daily_work_hours
def weekly_pay_per_employee : Nat := daily_pay_per_employee * weekly_work_days

-- Calculate the total weekly pay for all employees
def total_weekly_pay : Nat := weekly_pay_per_employee * total_employees

-- Define the statement to be proved
theorem weeks_per_month
  (h1 : total_employees = num_employees_initial + additional_employees)
  (h2 : daily_pay_per_employee = hourly_wage * daily_work_hours)
  (h3 : weekly_pay_per_employee = daily_pay_per_employee * weekly_work_days)
  (h4 : total_weekly_pay = weekly_pay_per_employee * total_employees)
  (h5 : total_monthly_pay = 1680000) :
  total_monthly_pay / total_weekly_pay = 4 :=
by sorry

end weeks_per_month_l561_561924


namespace Louisa_traveled_240_miles_first_day_l561_561358

noncomputable def distance_first_day (h : ℕ) := 60 * (h - 3)

theorem Louisa_traveled_240_miles_first_day :
  ∃ h : ℕ, 420 = 60 * h ∧ distance_first_day h = 240 :=
by
  sorry

end Louisa_traveled_240_miles_first_day_l561_561358


namespace inequality_holds_for_all_z_l561_561547

theorem inequality_holds_for_all_z (x y z : ℝ) (hx : x > 0) : y - z < real.sqrt (z^2 + x^2) :=
sorry

end inequality_holds_for_all_z_l561_561547


namespace area_ratio_l561_561858

-- Define the conditions: perimeters relation
def condition (a b : ℝ) := 4 * a = 16 * b

-- Define the theorem to be proved
theorem area_ratio (a b : ℝ) (h : condition a b) : (a * a) = 16 * (b * b) :=
sorry

end area_ratio_l561_561858


namespace saving_time_for_downpayment_l561_561352

def annual_salary : ℚ := 150000
def saving_rate : ℚ := 0.10
def house_cost : ℚ := 450000
def downpayment_rate : ℚ := 0.20

theorem saving_time_for_downpayment : 
  (downpayment_rate * house_cost) / (saving_rate * annual_salary) = 6 :=
by
  sorry

end saving_time_for_downpayment_l561_561352


namespace replace_cos_with_sin_l561_561806

-- Define the function f
def f (k : ℕ) (x : ℝ) : ℝ := (List.range (k + 1)).map (λ j => cos (2^j * x)).prod

-- Define the new function f₁ based on replacing one cos with sin
def f₁ (k n : ℕ) (x : ℝ) : ℝ :=
  (List.range (k + 1)).map (λ j => if j = n then sin (2^j * x) else cos (2^j * x)).prod

theorem replace_cos_with_sin {k : ℕ} (hk : k > 10) :
  ∃ n, ∀ x : ℝ, |f₁ k n x| ≤ 3 * 2^(-1-k) := sorry

end replace_cos_with_sin_l561_561806


namespace min_value_expression_l561_561570

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (inf {x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ x = (|6 * a - 4 * b| + |3 * (a + b * Real.sqrt 3) + 2 * (a * Real.sqrt 3 - b)|) / Real.sqrt (a^2 + b^2)}) = Real.sqrt 39 :=
by
  sorry

end min_value_expression_l561_561570


namespace find_n_square_divides_exponential_plus_one_l561_561105

theorem find_n_square_divides_exponential_plus_one :
  ∀ n : ℕ, (n^2 ∣ 2^n + 1) → (n = 1) :=
by
  sorry

end find_n_square_divides_exponential_plus_one_l561_561105


namespace dot_product_of_vectors_magnitude_of_vector_projection_of_vector_l561_561167

-- Defining the vectors and their magnitudes
variables {V : Type} [inner_product_space ℝ V]
variables (a b : V)
variable (angle_ab : ℝ)
variable (norm_a : ℝ)
variable (norm_b : ℝ)
variable (cos_theta : ℝ)

-- Conditions given in the problem
axiom angle_is_120 : angle_ab = real.pi * (2 / 3)  -- 120 degrees in radians
axiom norm_a_is_4 : ‖a‖ = 4
axiom norm_b_is_2 : ‖b‖ = 2
axiom cos_of_angle : cos_theta = real.cos angle_ab

-- Proof problem statements
theorem dot_product_of_vectors : 
  inner ((2 • a) - b) (a + (3 • b)) = 4 := sorry

theorem magnitude_of_vector : 
  ∥(2 • a) - (3 • b)∥ = 2 * real.sqrt 37 := sorry

theorem projection_of_vector :
  (2 * ‖a‖ * cos_theta) = -4 := sorry

end dot_product_of_vectors_magnitude_of_vector_projection_of_vector_l561_561167


namespace count_integers_between_sqrts_l561_561236

theorem count_integers_between_sqrts : 
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  (upper_bound - lower_bound + 1) = 6 :=
by
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  calc (upper_bound - lower_bound + 1) = 6 : sorry

end count_integers_between_sqrts_l561_561236


namespace parabola_directrix_l561_561111

theorem parabola_directrix (x : ℝ) : ∃ d : ℝ, (∀ x : ℝ, 4 * x ^ 2 - 3 = d) → d = -49 / 16 :=
by
  sorry

end parabola_directrix_l561_561111


namespace Jenny_has_6_cards_l561_561310

variable (J : ℕ)

noncomputable def Jenny_number := J
noncomputable def Orlando_number := J + 2
noncomputable def Richard_number := 3 * (J + 2)
noncomputable def Total_number := J + (J + 2) + 3 * (J + 2)

theorem Jenny_has_6_cards
  (h1 : Orlando_number J = J + 2)
  (h2 : Richard_number J = 3 * (J + 2))
  (h3 : Total_number J = 38) : J = 6 :=
by
  sorry

end Jenny_has_6_cards_l561_561310


namespace exists_three_digit_numbers_with_property_l561_561949

open Nat

def is_three_digit_number (n : ℕ) : Prop := (100 ≤ n ∧ n < 1000)

def distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def inserts_zeros_and_is_square (n : ℕ) (k : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  let transformed_number := a * 10^(2*k + 2) + b * 10^(k + 1) + c
  ∃ x : ℕ, transformed_number = x * x

theorem exists_three_digit_numbers_with_property:
  ∃ n1 n2 : ℕ, 
    is_three_digit_number n1 ∧ 
    is_three_digit_number n2 ∧ 
    distinct_digits n1 ∧ 
    distinct_digits n2 ∧ 
    ( ∀ k, inserts_zeros_and_is_square n1 k ) ∧ 
    ( ∀ k, inserts_zeros_and_is_square n2 k ) ∧ 
    n1 ≠ n2 := 
sorry

end exists_three_digit_numbers_with_property_l561_561949


namespace arthur_walks_distance_l561_561961

theorem arthur_walks_distance :
  ∀ (blocks_east blocks_north blocks_first blocks_other distance_first distance_other : ℕ)
  (fraction_first fraction_other : ℚ),
    blocks_east = 8 →
    blocks_north = 16 →
    blocks_first = 10 →
    blocks_other = (blocks_east + blocks_north) - blocks_first →
    fraction_first = 1 / 3 →
    fraction_other = 1 / 4 →
    distance_first = blocks_first * fraction_first →
    distance_other = blocks_other * fraction_other →
    (distance_first + distance_other) = 41 / 6 :=
by
  intros blocks_east blocks_north blocks_first blocks_other distance_first distance_other fraction_first fraction_other
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end arthur_walks_distance_l561_561961


namespace moving_circle_passes_focus_l561_561049

noncomputable def parabola (x : ℝ) : Set (ℝ × ℝ) := {p | p.2 ^ 2 = 8 * p.1}
def is_tangent (c : ℝ × ℝ) (r : ℝ) : Prop := c.1 = -2 ∨ c.1 = -2 + 2 * r

theorem moving_circle_passes_focus
  (center : ℝ × ℝ) (H1 : center ∈ parabola center.1)
  (H2 : is_tangent center 2) :
  ∃ focus : ℝ × ℝ, focus = (2, 0) ∧ ∃ r : ℝ, ∀ p ∈ parabola center.1, dist center p = r := sorry

end moving_circle_passes_focus_l561_561049


namespace calculate_expression_l561_561515

def thirteen_power_thirteen_div_thirteen_power_twelve := 13 ^ 13 / 13 ^ 12
def expression := (thirteen_power_thirteen_div_thirteen_power_twelve ^ 3) * (3 ^ 3)
/- We define the main statement to be proven -/
theorem calculate_expression : (expression / 2 ^ 6) = 926 := sorry

end calculate_expression_l561_561515


namespace volume_of_region_l561_561611

theorem volume_of_region :
  ∃ (V : ℝ), V = 9 ∧
  ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 :=
sorry

end volume_of_region_l561_561611


namespace distance_between_lateral_edge_and_base_diagonal_l561_561418

noncomputable def distance_between_edge_and_diagonal 
(a : ℝ) : ℝ :=
  let α := Real.arctan (1 / Real.sqrt 2) in
  let cosα := Real.cos α in
  let sinα := Real.sin α in
  a * Real.sqrt(6) / 6

theorem distance_between_lateral_edge_and_base_diagonal 
  (a : ℝ) (h : ∠PKM = π / 4) : 
  distance_between_edge_and_diagonal a = a * Real.sqrt(6) / 6 :=
sorry

end distance_between_lateral_edge_and_base_diagonal_l561_561418


namespace graph_passes_through_point_l561_561400

variable (a : ℝ)
variable (h1 : a > 0)
variable (h2 : a ≠ 1)

theorem graph_passes_through_point : ∃ x y : ℝ, x = 3 ∧ y = 1 ∧ y = a ^ (x - 3) :=
by
  use 3
  use 1
  simp
  exact sorry

end graph_passes_through_point_l561_561400


namespace confidence_test_correctness_l561_561299

theorem confidence_test_correctness
  (χ2_value : ℝ)
  (confidence_level_99 : χ2_value > 6.635 → Prop)
  (confidence_level_95 : χ2_value > 0.0 → Prop)
  (error_rate_95 : confidence_level_95 → Prop) :
  (confidence_level_99 χ2_value → ¬ ∀ infants, (infants = 1000) → (999 have_kidney_stones)) ∧ 
  (confidence_level_99 χ2_value → ¬ ∀ infant, (infant consumes_formula) → (99% infant have_kidney_stones)) ∧
  (confidence_level_95 χ2_value ↔ error_rate_95) :=
by
  sorry

end confidence_test_correctness_l561_561299


namespace problem1_problem2_l561_561375

theorem problem1 (x y z: ℝ)
    (h1: x = 4*(Real.sqrt 3-2)^4)
    (h2: y = (0.25)^(1 / 2))
    (h3: z = (1 / Real.sqrt 2)^(-4)):
    x - y * z = -80*Real.sqrt 3 + 194 := by 
    -- the proof will go here
    sorry

theorem problem2: 1 / 2 * (Real.log 25) + (Real.log 2) - (Real.log 0.1) = 2 := by 
    -- the proof will go here
    sorry

end problem1_problem2_l561_561375


namespace count_integers_between_sqrt8_sqrt72_l561_561252

-- Define the square roots of 8 and 72 for reference
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt(8)
def minIntAboveSqrt8 : ℤ := Int.ceil sqrt8

-- Define the largest integer less than sqrt(72)
def maxIntBelowSqrt72 : ℤ := Int.floor sqrt72

-- State the main theorem to prove
theorem count_integers_between_sqrt8_sqrt72 :
  minIntAboveSqrt8 = 3 ∧ maxIntBelowSqrt72 = 8 → 
  (maxIntBelowSqrt72 - minIntAboveSqrt8 + 1) = 6 :=
by
  sorry

end count_integers_between_sqrt8_sqrt72_l561_561252


namespace f_one_eq_two_l561_561775

def S := {x : ℝ // 0 < x}

def f (x : S) : S

axiom f_property_i (x : S) : f ⟨1 / x.val, sorry⟩ = ⟨x.val * (f x).val, sorry⟩
axiom f_property_ii (x y : S) : f x + f y = ⟨x.val + y.val + f ⟨x.val * y.val, sorry⟩.val, sorry⟩

theorem f_one_eq_two : f ⟨1, sorry⟩ = ⟨2, sorry⟩ :=
sorry

end f_one_eq_two_l561_561775


namespace maria_total_cost_l561_561351

-- Define the conditions as variables in the Lean environment
def daily_rental_rate : ℝ := 35
def mileage_rate : ℝ := 0.25
def rental_days : ℕ := 3
def miles_driven : ℕ := 500

-- Now, state the theorem that Maria’s total payment should be $230
theorem maria_total_cost : (daily_rental_rate * rental_days) + (mileage_rate * miles_driven) = 230 := 
by
  -- no proof required, just state as sorry
  sorry

end maria_total_cost_l561_561351


namespace determine_b_l561_561276

noncomputable def f (x b : ℝ) : ℝ := x^3 - b * x^2 + 1/2

theorem determine_b (b : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 b = 0 ∧ f x2 b = 0) → b = 3/2 :=
by
  sorry

end determine_b_l561_561276


namespace floor_sqrt_23_squared_eq_16_l561_561555

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end floor_sqrt_23_squared_eq_16_l561_561555


namespace floor_sqrt_23_squared_l561_561552

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end floor_sqrt_23_squared_l561_561552


namespace lines_divide_plane_l561_561815

theorem lines_divide_plane : ∀ (n : ℕ), n = 7 → (∀ l1 l2 l3 : ℕ, l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3)
→ (1 + n + n * (n - 1) / 2 = 29) :=
by
  intros n n_eq h
  rw n_eq
  have h1 : 1 + 7 + 7 * (7 - 1) / 2 = 29 := by norm_num
  exact h1

end lines_divide_plane_l561_561815


namespace interval_integer_count_l561_561225

-- Define the problem conditions
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt8
def lower_bound : Int := ceil sqrt8

-- Define the largest integer less than sqrt72
def upper_bound : Int := floor sqrt72

-- Prove the number of integers between sqrt8 and sqrt72 is 6
theorem interval_integer_count : 
    (upper_bound - lower_bound + 1) = 6 := 
by
  -- Steps and proofs would go here, but we use sorry to skip the proof for now
  sorry

end interval_integer_count_l561_561225


namespace people_not_like_both_l561_561360

-- Conditions
def total_people : ℕ := 1500
def percentage_not_like_radio : ℚ := 0.35
def percentage_not_like_both : ℚ := 0.15

-- Calculation
noncomputable def number_not_like_radio : ℚ := percentage_not_like_radio * total_people
noncomputable def number_not_like_both (n_radio : ℚ) : ℚ := percentage_not_like_both * n_radio

-- Rounding function to handle decimals
noncomputable def round (x : ℚ) : ℕ := Int.toNat (Real.toInt x)

-- The theorem to prove
theorem people_not_like_both :
  round (number_not_like_both number_not_like_radio) = 79 :=
by sorry

end people_not_like_both_l561_561360


namespace tangent_line_at_point_l561_561545

-- Define the function y
def f : ℝ → ℝ := λ x, x * exp x + 2 * x + 1

-- Define the point of tangency
def point_of_tangency := (0 : ℝ, 1 : ℝ)

-- Define the target equation of the tangent line
def tangent_line (x : ℝ) : ℝ := 3 * x + 1

-- The theorem to prove
theorem tangent_line_at_point :
  tangent_line (point_of_tangency.1) = point_of_tangency.2 :=
by
  -- equations and conditions
  have deriv : deriv f point_of_tangency.1 = 3 := sorry,
  real.rfl

end tangent_line_at_point_l561_561545


namespace total_number_of_rulers_l561_561874

-- Given conditions
def initial_rulers : ℕ := 11
def rulers_added_by_tim : ℕ := 14

-- Given question and desired outcome
def total_rulers (initial_rulers rulers_added_by_tim : ℕ) : ℕ :=
  initial_rulers + rulers_added_by_tim

-- The proof problem statement
theorem total_number_of_rulers : total_rulers 11 14 = 25 := by
  sorry

end total_number_of_rulers_l561_561874


namespace lunks_needed_for_24_apples_l561_561704

-- Define the conditions as Lean definitions
def lunks_per_kunks := 7 / 4
def kunks_per_apples := 3 / 5
def apples_needed := 24

-- State the theorem
theorem lunks_needed_for_24_apples : 
  let k := (3 * apples_needed) / 5 in 
  let rounded_k := k.ceil in 
  let l := (7 * rounded_k) / 4 in 
  l.ceil = 27 :=
by 
  let k := (3 * apples_needed) / 5
  let rounded_k := k.ceil
  let l := (7 * rounded_k) / 4
  have h1 : k = (3 * apples_needed) / 5 := rfl
  have h2 : rounded_k = k.ceil := rfl
  have h3 : l = (7 * rounded_k) / 4 := rfl
  show l.ceil = 27, from sorry

end lunks_needed_for_24_apples_l561_561704


namespace overdue_book_fine_day5_l561_561451

noncomputable def fine : ℕ → ℝ
| 1 => 0.07
| (n+1) => min (fine n + 0.30) (2 * fine n)

theorem overdue_book_fine_day5 : fine 5 = 0.86 :=
sorry

end overdue_book_fine_day5_l561_561451


namespace taxi_ride_cost_l561_561499

-- Lean statement
theorem taxi_ride_cost (base_fare : ℝ) (rate1 : ℝ) (rate1_miles : ℝ) (rate2 : ℝ) (total_miles : ℝ) 
  (h_base_fare : base_fare = 2.00)
  (h_rate1 : rate1 = 0.30)
  (h_rate1_miles : rate1_miles = 3)
  (h_rate2 : rate2 = 0.40)
  (h_total_miles : total_miles = 8) :
  let rate1_cost := rate1 * rate1_miles
  let rate2_cost := rate2 * (total_miles - rate1_miles)
  base_fare + rate1_cost + rate2_cost = 4.90 := by
  sorry

end taxi_ride_cost_l561_561499


namespace pq_sufficient_but_not_necessary_condition_l561_561655

theorem pq_sufficient_but_not_necessary_condition (p q : Prop) (hpq : p ∧ q) :
  ¬¬p = p :=
by
  sorry

end pq_sufficient_but_not_necessary_condition_l561_561655


namespace min_value_ratio_l561_561653

-- Conditions
variables (a b : ℝ)
variables (h₀ : a > 0) (h₁ : b > 0) (h₂ : 2 * a + b = 1)

-- The main goal to prove
theorem min_value_ratio : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ (2 / a + 1 / b) = 9 :=
by
  use [1 / 3, 1 / 3]
  split
  { apply one_div_pos.2, norm_num }
  split
  { apply one_div_pos.2, norm_num }
  split
  { norm_num }
  { field_simp, norm_num }

end min_value_ratio_l561_561653


namespace lawrence_average_work_hours_l561_561319

def total_hours_worked (monday tuesday wednesday thursday friday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday + friday

theorem lawrence_average_work_hours :
  let monday := 8
  let tuesday := 8
  let wednesday := 5.5
  let thursday := 5.5
  let friday := 8
  let total_hours := total_hours_worked monday tuesday wednesday thursday friday
  let days_worked := 5
  total_hours / days_worked = 7 :=
by
  -- Notice that we'll need to use rationals or real numbers to handle 5.5
  -- This part of Lean requires corresponding numeric modules:
  have h1 : (8 : ℚ) + 8 + 5.5 + 5.5 + 8 = 35, by norm_num
  have h2 : (35 / 5 : ℚ) = 7, by norm_num
  exact_mod_cast h2 

end lawrence_average_work_hours_l561_561319


namespace trigonometric_sum_proof_l561_561521

theorem trigonometric_sum_proof : 
  ∑ x in finset.range 45 \ finset.range 2, 2 * sin(x) * sin(2) * (1 + sec(x - 2) * sec(x + 2)) 
  = ∑ n in finset.range 5, (-1)^n * sin^2([1, 2, 47, 48, 49][n]) / cos([1, 2, 47, 48, 49][n])
 :=
  sorry

end trigonometric_sum_proof_l561_561521


namespace cost_of_croissants_l561_561623

theorem cost_of_croissants (n : ℕ) (s : ℕ) (c : ℕ) (price_per_dozen : ℕ → ℝ) (people : ℕ) (sandwiches_pp : ℕ) (croissants_per_dozen : ℕ) :
  people = 24 → sandwiches_pp = 2 → croissants_per_dozen = 12 → price_per_dozen croissants_per_dozen = 8.0 →
  n = people * sandwiches_pp → s = n / croissants_per_dozen → c = s * price_per_dozen croissants_per_dozen → c = 32.0 := by
  intro h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2] at h5
  rw [Nat.mul_comm] at h5
  rw [←h5] at h6
  rw [h3] at h6
  have h8 : n / croissants_per_dozen = s := by
    rw [←h6]
    rfl
  rw [←h6, h4] at h7
  sorry

end cost_of_croissants_l561_561623


namespace problem_triangle_area_l561_561993

noncomputable def complexEquationsArea : ℝ :=
  let z1 := Complex.sqrt (3 + 3 * Complex.i * Real.sqrt 7)
  let z2 := Complex.sqrt (5 + 5 * Complex.i)
  let triangleArea (a b c : Complex) : ℝ := 
    1 / 2 * Complex.abs ((b - a) * Complex.conj (c - a)).im
  in triangleArea z1 (-z1) z2

theorem problem_triangle_area : complexEquationsArea = 5 * Real.sqrt 14 := 
  sorry

end problem_triangle_area_l561_561993


namespace lunks_to_apples_l561_561690

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l561_561690


namespace find_number_l561_561932

theorem find_number (n : ℝ) (h : n / 0.06 = 16.666666666666668) : n = 1 :=
by
  sorry

end find_number_l561_561932


namespace count_integers_between_sqrt8_and_sqrt72_l561_561208

theorem count_integers_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℕ, 3 ≤ x ∧ x ≤ 8 → x ∈ finset.range (8 - 3 + 1) :=
by
  let a := real.sqrt 8
  let b := real.sqrt 72
  have ha : 3 = nat_ceil a := sorry
  have hb : 8 = nat_floor b := sorry
  use 6
  split
  {
    exact sorry
  }
  {
    intros x hx
    have h_valid : 3 ≤ x ∧ x ≤ 8 := hx
    rw finset.mem_range
    sorry
  }

end count_integers_between_sqrt8_and_sqrt72_l561_561208


namespace kaleb_sold_books_l561_561313

theorem kaleb_sold_books (initial_books sold_books purchased_books final_books : ℕ)
  (H_initial : initial_books = 34)
  (H_purchased : purchased_books = 7)
  (H_final : final_books = 24) :
  sold_books = 17 :=
by
  have H_equation : (initial_books - sold_books) + purchased_books = final_books,
    by sorry
  rw [H_initial, H_purchased, H_final] at H_equation,
  sorry

end kaleb_sold_books_l561_561313


namespace willy_days_to_finish_series_l561_561447

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def days_to_finish (total_episodes : ℕ) (episodes_per_day : ℕ) : ℕ :=
  total_episodes / episodes_per_day

theorem willy_days_to_finish_series : 
  total_episodes 3 20 = 60 → 
  days_to_finish 60 2 = 30 :=
by
  intros h1
  rw [h1]
  rfl

end willy_days_to_finish_series_l561_561447


namespace sherry_catches_train_probability_l561_561906

theorem sherry_catches_train_probability :
  let p_train_arrival := 0.75
  let p_notice_given_train := 0.25
  let p_miss_one_min := p_train_arrival * (1 - p_notice_given_train) + (1 - p_train_arrival)
  let p_miss_five_min := p_miss_one_min ^ 5
  let p_catch_five_min := 1 - p_miss_five_min
  p_catch_five_min ≈ 0.718 :=
by
  let p_train_arrival := 0.75
  let p_notice_given_train := 0.25
  let p_miss_one_min := p_train_arrival * (1 - p_notice_given_train) + (1 - p_train_arrival)
  let p_miss_five_min := p_miss_one_min ^ 5
  let p_catch_five_min := 1 - p_miss_five_min
  have h : p_catch_five_min ≈ 0.718 := sorry
  exact h

end sherry_catches_train_probability_l561_561906


namespace balls_in_small_box_l561_561286

variable (n m : ℕ)

theorem balls_in_small_box (n m : ℕ) (h : 100 ≥ n + m) : ∃ x : ℕ, x = 100 - n - m := 
by 
  exists 100 - n - m 
  ring

end balls_in_small_box_l561_561286


namespace incorrect_calculation_l561_561894

theorem incorrect_calculation : ¬ (∀ a : ℝ, a^3 + a^3 = 2 * a^6) :=
by
  assume h : ∀ a : ℝ, a^3 + a^3 = 2 * a^6
  sorry

end incorrect_calculation_l561_561894


namespace find_slope_l561_561045

theorem find_slope
  (k : ℝ)
  (A B M : ℝ × ℝ)
  (hA : A.2 ^ 2 = 4 * A.1)
  (hB : B.2 ^ 2 = 4 * B.1)
  (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hTangent : (M.1 - 5) ^ 2 + M.2 ^ 2 = 9)
  (hMidpointEq : k * M.2 = 2)
  (hTangentLineEq : M.2 / (M.1 - 5) = -1 / k) :
  k = ±(2 * Real.sqrt 5 / 5) :=
sorry

end find_slope_l561_561045


namespace number_of_companies_l561_561965

theorem number_of_companies (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by
  sorry

end number_of_companies_l561_561965


namespace truePropositions_l561_561176

noncomputable def prop1 {L1 L2 P : Type} [Plane P] : Prop :=
  ∀ (l1 l2 : L1) (p : P), (parallel l1 p ∧ parallel l2 p) → parallel l1 l2

noncomputable def prop2 {L1 P P' : Type} [Plane P] [Plane P'] : Prop :=
  ∀ (p : P) (p' : P') (l : L1), (perpendicular l p' ∧ onPlane l p) → perpendicular p p'

noncomputable def prop3 {L1 : Type} : Prop :=
  ∀ (l1 l2 l3 : L1), (perpendicular l1 l3 ∧ perpendicular l2 l3) → parallel l1 l2

noncomputable def prop4 {L1 P P' : Type} [Plane P] [Plane P'] : Prop :=
  ∀ (p : P) (p' : P') (l : L1), (perpendicular p p' ∧ ¬ perpendicular l (intersection p p')) → ¬ perpendicular l p'

theorem truePropositions : 
  (prop2 ∧ prop4) ∧ ¬prop1 ∧ ¬prop3 :=
by 
  sorry

end truePropositions_l561_561176


namespace positive_y_equals_32_l561_561097

theorem positive_y_equals_32 (y : ℝ) (h : y^2 = 1024) (hy : 0 < y) : y = 32 :=
sorry

end positive_y_equals_32_l561_561097


namespace eq_solution_l561_561422

theorem eq_solution (x : ℝ) (h : 2 / x = 3 / (x + 1)) : x = 2 :=
by
  sorry

end eq_solution_l561_561422


namespace tangent_product_theorem_for_inside_point_tangent_product_theorem_for_outside_point_l561_561306

noncomputable def tangent_product_in_circle (R d : ℝ) (hR : 0 < R) (hd : 0 ≤ d) (hbound : d < R) : Prop :=
  ∀ (A B M O : ℝ → ℝ)
  (hAOM : Real.angle A O M = 2 * θ_1)
  (hBOM : Real.angle B O M = 2 * θ_2),
  (θ_1 + θ_2 = π ∨ θ_1 + θ_2 = 2 * π) →
  tan (θ_1) * tan (θ_2) = (R - d) / (R + d)

theorem tangent_product_theorem_for_inside_point {R d : ℝ} (hR : 0 < R) (hd : 0 ≤ d) (hbound : d < R) :
  tangent_product_in_circle R d hR hd hbound :=
sorry

noncomputable def tangent_product_in_circle_outside_point (R d : ℝ) (hR : 0 < R) (hd : 0 ≤ d) (hbound : d ≥ R) : Prop :=
  ∀ (A B M O : ℝ → ℝ)
  (hAOM : Real.angle A O M = 2 * θ_1)
  (hBOM : Real.angle B O M = 2 * θ_2),
  (θ_1 + θ_2 = π ∨ θ_1 + θ_2 = 2 * π) →
  tan (θ_1) * tan (θ_2) = (d - R) / (d + R)

theorem tangent_product_theorem_for_outside_point {R d : ℝ} (hR : 0 < R) (hd : 0 ≤ d) (hbound : d ≥ R) :
  tangent_product_in_circle_outside_point R d hR hd hbound :=
sorry

end tangent_product_theorem_for_inside_point_tangent_product_theorem_for_outside_point_l561_561306


namespace part_a_part_b_l561_561770

theorem part_a (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 
  ∃ q, q ≠ p ∧ Nat.Prime q ∧ q ∣ (p - 1)^p + 1 :=
sorry

theorem part_b (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p)
  (pi : ℕ → ℕ) (ai : ℕ → ℕ) (h_prod : ((p - 1)^p + 1) = ∏ i in Finset.range n, pi i ^ ai i) :
  (∑ i in Finset.range n, pi i * ai i) ≥ p^2 / 2 :=
sorry

end part_a_part_b_l561_561770


namespace expression_bounds_l561_561772

noncomputable def expression (p q r s : ℝ) : ℝ :=
  Real.sqrt (p^2 + (2 - q)^2) + Real.sqrt (q^2 + (2 - r)^2) +
  Real.sqrt (r^2 + (2 - s)^2) + Real.sqrt (s^2 + (2 - p)^2)

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2)
  (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) : 
  4 * Real.sqrt 2 ≤ expression p q r s ∧ expression p q r s ≤ 8 :=
by
  sorry

end expression_bounds_l561_561772


namespace play_role_assignments_l561_561490

def specific_role_assignments (men women remaining either_gender_roles : ℕ) : ℕ :=
  men * women * Nat.choose remaining either_gender_roles

theorem play_role_assignments :
  specific_role_assignments 6 7 11 4 = 13860 := by
  -- The given problem statement implies evaluating the specific role assignments
  sorry

end play_role_assignments_l561_561490


namespace tom_total_out_of_pocket_cost_l561_561881

def initial_doctor_visit_cost : ℕ := 300
def cast_cost : ℕ := 200
def insurance_coverage_initial : ℕ := 60
def num_physical_therapy_sessions : ℕ := 8
def physical_therapy_cost_per_session : ℕ := 100
def insurance_coverage_physical_therapy : ℕ := 40
def copay_per_session : ℕ := 20

theorem tom_total_out_of_pocket_cost :
  let initial_total := initial_doctor_visit_cost + cast_cost in
  let initial_covered := initial_total * insurance_coverage_initial / 100 in
  let initial_out_of_pocket := initial_total - initial_covered in

  let total_physical_therapy_cost := physical_therapy_cost_per_session * num_physical_therapy_sessions in
  let physical_therapy_covered := total_physical_therapy_cost * insurance_coverage_physical_therapy / 100 in
  let physical_therapy_out_of_pocket := total_physical_therapy_cost - physical_therapy_covered in
  let total_copay := copay_per_session * num_physical_therapy_sessions in
  let total_physical_therapy_out_of_pocket := physical_therapy_out_of_pocket + total_copay in

  let total_out_of_pocket := initial_out_of_pocket + total_physical_therapy_out_of_pocket in
  total_out_of_pocket = 840 :=
by
  sorry

end tom_total_out_of_pocket_cost_l561_561881


namespace min_xyz_product_l561_561340

open Real

noncomputable def minimum_product (x y z : ℝ) : ℝ :=
  if (0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ max (max x y) z ≤ 3 * min (min x y) z)
  then x * y * z
  else 1  -- this should never happen as it's just for satisfying completeness.

theorem min_xyz_product:
  ∃ x y z > 0, x + y + z = 1 ∧ (max (max x y) z) ≤ 3 * (min (min x y) z) ∧ 
  (∀ a b c > 0, a + b + c = 1 ∧ (max (max a b) c ≤ 3 * min (min a b) c) → a * b * c ≥ x * y * z) ∧ 
  x * y * z = 1 / 36 :=
begin
  -- Define variables according to the given conditions
  let x := 1 / 6,
  let y := 1 / 2,
  let z := 1 / 3,
  
  -- Ensure the positivity condition
  have hx : 0 < x := by norm_num,
  have hy : 0 < y := by norm_num,
  have hz : 0 < z := by norm_num,

  -- Check the sum condition
  have hs : x + y + z = 1 := by norm_num,

  -- Check the ratio condition
  have hr : max (max x y) z ≤ 3 * min (min x y) z := by norm_num,

  use [x, y, z, hx, hy, hz],
  refine ⟨hs, hr, _, _⟩,
  {
    -- Prove that the product is indeed minimized
    intros a b c hpa hpb hpc hsum hmax,
    sorry, -- This proof will check that the product for any a, b, c is ≥ x * y * z 
  },
  {
    -- Confirm the calculated product is the correct minimum
    show x * y * z = 1 / 36, by norm_num,
  }
end

end min_xyz_product_l561_561340


namespace professor_D_error_l561_561508

noncomputable def polynomial_calculation_error (n : ℕ) : Prop :=
  ∃ (f : ℝ → ℝ), (∀ i : ℕ, i ≤ n+1 → f i = 2^i) ∧ f (n+2) ≠ 2^(n+2) - n - 3

theorem professor_D_error (n : ℕ) : polynomial_calculation_error n :=
  sorry

end professor_D_error_l561_561508


namespace number_of_integers_between_sqrt8_sqrt72_l561_561240

theorem number_of_integers_between_sqrt8_sqrt72 :
  let sqrt_8 := Real.sqrt 8
  let sqrt_72 := Real.sqrt 72
  let lower_bound := sqrt_8.ceil
  let upper_bound := sqrt_72.floor
  ∃ n : ℕ, lower_bound ≤ n ∧ n ≤ upper_bound → n = 6 := by
sorry

end number_of_integers_between_sqrt8_sqrt72_l561_561240


namespace least_possible_b_l561_561870

theorem least_possible_b (a b : ℕ) (h1 : a + b = 120) (h2 : (Prime a ∨ ∃ p : ℕ, Prime p ∧ a = 2 * p)) (h3 : Prime b) (h4 : a > b) : b = 7 :=
sorry

end least_possible_b_l561_561870


namespace largest_prime_factor_of_4752_l561_561002

theorem largest_prime_factor_of_4752 : ∃ p : ℕ, nat.prime p ∧ p ∣ 4752 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 4752 → q ≤ p :=
begin
  -- Proof goes here
  sorry
end

end largest_prime_factor_of_4752_l561_561002


namespace extreme_value_at_neg3_l561_561399

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

theorem extreme_value_at_neg3 (a : ℝ) : (∃ c : ℝ, c = -3 ∧ (f a)' c = 0) → a = 5 :=
by
  intro h
  have : (f a)' = λ x, 3 * x^2 + 2 * a * x + 3 := by sorry
  have eq : 3 * (-3)^2 + 2 * a * (-3) + 3 = 0 := by sorry
  have : 27 - 6 * a + 3 = 0 := by sorry
  simp at this
  exact this

end extreme_value_at_neg3_l561_561399


namespace number_of_integers_between_sqrt8_sqrt72_l561_561241

theorem number_of_integers_between_sqrt8_sqrt72 :
  let sqrt_8 := Real.sqrt 8
  let sqrt_72 := Real.sqrt 72
  let lower_bound := sqrt_8.ceil
  let upper_bound := sqrt_72.floor
  ∃ n : ℕ, lower_bound ≤ n ∧ n ≤ upper_bound → n = 6 := by
sorry

end number_of_integers_between_sqrt8_sqrt72_l561_561241


namespace line_AM_bisects_BC_l561_561908

-- Definitions of points, lines, tangency, and intersection
variable {α : Type} [metric_space α]

/-- Points involved in the problem -/
variables (A B C M : α)

/-- Circles and their tangency properties -/
variables (S1 S2 : set α) 
variables [is_circle S1 A C B] [is_circle S2 C B]
variable [∀ P, tangent_of S1 A P ↔ (P = A ∨ tangent_on S1 A P)]
variable [∀ P, tangent_of S2 C P ↔ (P = C ∨ tangent_on S2 C P)]
variable [∀ P, ∃! Q, lies_on P S1 ↔ lies_on Q S2]

/-- Intersection point of the circles -/
variables [intersect_at S1 S2 M]

/-- Proof statement: AM bisects BC -/
theorem line_AM_bisects_BC (h1 : tangent_on (line_through A C) S1)
  (h2 : tangent_on (line_through C B) S2)
  (h3 : intersects_at (circle S1) (circle S2) M) :
  exists P, midpoint_of_segment (line_through B C) P ∧ lies_on P (line_through A M) := sorry

end line_AM_bisects_BC_l561_561908


namespace triangles_congruent_by_angle_bisector_and_adjacent_side_l561_561367

variables {A A₁ B B₁ C C₁ D D₁ : Type*}
variables [metric_space A] [metric_space A₁] [metric_space B] [metric_space B₁] [metric_space C] [metric_space C₁]
variables [metric_space D] [metric_space D₁]

variables (A B C D A₁ B₁ C₁ D₁ : Type*)
variables (AD : metric_line A D) (A₁D₁ : metric_line A₁ D₁)
variables (AC : metric_segment A C) (A₁C₁ : metric_segment A₁ C₁)
variables (BAC : angle B A C) (B₁A₁C₁ : angle B₁ A₁ C₁)

-- Conditions: 
variable (h1 : AD = A₁D₁)
variable (h2 : AC = A₁C₁)
variable (h3 : BAC = B₁A₁C₁)

theorem triangles_congruent_by_angle_bisector_and_adjacent_side :
  ∀ (A B C : Type*) (A₁ B₁ C₁ : Type*) (D D₁ : Type*),
  (AD = A₁D₁) → (AC = A₁C₁) → (BAC = B₁A₁C₁) → 
  triangle_congruent A B C A₁ B₁ C₁ :=
by sorry

end triangles_congruent_by_angle_bisector_and_adjacent_side_l561_561367


namespace sum_of_eight_numbers_l561_561273

theorem sum_of_eight_numbers (avg : ℚ) (n : ℕ) (sum : ℚ) 
  (h_avg : avg = 5.3) (h_n : n = 8) : sum = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l561_561273


namespace range_of_m_for_real_roots_value_of_m_for_specific_roots_l561_561672

open Real

variable {m x : ℝ}

def quadratic (m : ℝ) (x : ℝ) := x^2 + 2*(m-1)*x + m^2 + 2 = 0
  
theorem range_of_m_for_real_roots (h : ∃ x : ℝ, quadratic m x) : m ≤ -1/2 :=
sorry

theorem value_of_m_for_specific_roots
  (h : quadratic m x)
  (Hroots : ∃ x1 x2 : ℝ, quadratic m x1 ∧ quadratic m x2 ∧ (x1 - x2)^2 = 18 - x1 * x2) :
  m = -2 :=
sorry

end range_of_m_for_real_roots_value_of_m_for_specific_roots_l561_561672


namespace intersection_points_eq_4_l561_561084

variable {B : ℝ} (hB : B > 0)

def y_equals_Bx2 (x y : ℝ) := y = B * x^2

def curve (x y : ℝ) := y^2 + 4 = x^2 + 6 * y

theorem intersection_points_eq_4 : ∀ (x y : ℝ), y_equals_Bx2 B x y → curve x y → (x, y) ∈ {p : ℝ × ℝ | y_equals_Bx2 B p.1 p.2 ∧ curve p.1 p.2} :=
begin
  sorry
end

end intersection_points_eq_4_l561_561084


namespace president_vice_committee_count_l561_561738

open Nat

noncomputable def choose_president_vice_committee (total_people : ℕ) : ℕ :=
  let choose : ℕ := 56 -- binomial 8 3 is 56
  total_people * (total_people - 1) * choose

theorem president_vice_committee_count :
  choose_president_vice_committee 10 = 5040 :=
by
  sorry

end president_vice_committee_count_l561_561738


namespace perimeter_ratio_l561_561382

/-- Suppose we have a square piece of paper, 6 inches on each side, folded in half horizontally. 
The paper is then cut along the fold, and one of the halves is subsequently cut again horizontally 
through all layers. This results in one large rectangle and two smaller identical rectangles. 
Find the ratio of the perimeter of one smaller rectangle to the perimeter of the larger rectangle. -/
theorem perimeter_ratio (side_length : ℝ) (half_side_length : ℝ) (double_half_side_length : ℝ) :
    side_length = 6 →
    half_side_length = side_length / 2 →
    double_half_side_length = 1.5 * 2 →
    (2 * (half_side_length / 2 + side_length)) / (2 * (half_side_length + side_length)) = (5 / 6) :=
by
    -- Declare the side lengths
    intros h₁ h₂ h₃
    -- Insert the necessary algebra (proven manually earlier)
    sorry

end perimeter_ratio_l561_561382


namespace minimum_distance_l561_561183

theorem minimum_distance (m n : ℝ) (a : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ 4) 
  (h3 : m * Real.sqrt (Real.log a - 1 / 4) + 2 * a + 1 / 2 * n = 0) : 
  Real.sqrt (m^2 + n^2) = 4 * Real.sqrt (Real.log 2) / Real.log 2 :=
sorry

end minimum_distance_l561_561183


namespace complement_of_alpha_l561_561628

def angle := ℝ

-- Define the given condition
def alpha : angle := 37 + 45 / 60

-- Define what it means to be a complement
def is_complement (x y : angle) : Prop := x + y = 90

-- State the main theorem
theorem complement_of_alpha : ∃ β : angle, is_complement alpha β ∧ β = 52 + 15 / 60 :=
by
  let beta := 52 + 15 / 60
  use beta
  dsimp [is_complement, alpha, beta]
  norm_num
  sorry

end complement_of_alpha_l561_561628


namespace num_integer_pairs_l561_561651

def set_A (a : ℕ) : Set ℕ := { x | 5 * x ≤ a }
def set_B (b : ℕ) : Set ℕ := { x | 6 * x > b }
def N := { n : ℕ | true }

theorem num_integer_pairs (A_intersect_B_eq : (set_A a ∩ set_B b ∩ N) = {2, 3, 4}) (a_natural : a ∈ ℕ) (b_natural : b ∈ ℕ) :
  ∃! (pairs : Finset (ℕ × ℕ)), pairs.card = 30 :=
sorry

end num_integer_pairs_l561_561651


namespace eccentricity_is_3_over_5_max_product_MF_NF_l561_561642

-- Definitions for the parametric equations of the line
def line_x (t α : ℝ) : ℝ := t * Real.cos α + 3
def line_y (t α : ℝ) : ℝ := t * Real.sin α

-- Definitions for the parametric equations of the ellipse
def ellipse_x (θ : ℝ) : ℝ := 5 * Real.cos θ
def ellipse_y (θ m : ℝ) : ℝ := m * Real.sin θ

-- Condition: The line passes through the right focus (3, 0) of the ellipse
def passes_through_focus (t : ℝ) (α : ℝ) : Prop :=
  line_x t α = 3 ∧ line_y t α = 0

-- Proof of the first statement: Eccentricity of the ellipse
theorem eccentricity_is_3_over_5 (m : ℝ) :
  passes_through_focus t α →
  (3 : ℝ) / 5 = 3 / 5 :=
by
  sorry

-- Proof of the second statement: Maximum product of distances MF and NF
theorem max_product_MF_NF (m : ℝ) α :
  (line_x t α = 3) →
  (|t * t |) ≤ 16 :=
by
  sorry

end eccentricity_is_3_over_5_max_product_MF_NF_l561_561642


namespace KN_eq_NA_l561_561151

noncomputable theory

variables {A B C K L M N : Type}

-- Assume A, B, and C are points forming a non-isosceles triangle
variable [triangle A B C]

-- Assume K is the foot of the angle bisector from A
variable (K : foot_of_angle_bisector A)

-- Assume L is the foot of the angle bisector from B
variable (L : foot_of_angle_bisector B)

-- Assume M is the intersection of the perpendicular bisector of BL and AK
variable (M : intersection (perpendicular_bisector B L) (line A K))

-- Assume N is a point on BL such that KN is parallel to ML
variable (N : point_on_line B L)
variable (parallel KN ML : parallel (line K N) (line M L))

-- Prove KN = NA
theorem KN_eq_NA : distance K N = distance N A :=
  sorry

end KN_eq_NA_l561_561151


namespace function_passes_through_fixed_point_l561_561397

theorem function_passes_through_fixed_point (a : ℝ) (ha1 : 0 < a) (ha2 : a ≠ 1) :
  f 2015 = 2016 :=
  by
    let f (x : ℝ) := a^(x - 2015) + 2015
    show f 2015 = 2016
    calc
    f 2015 = a^(2015 - 2015) + 2015 : by rfl
           ... = 1 + 2015 : by rw [pow_zero]
           ... = 2016 : by rfl

end function_passes_through_fixed_point_l561_561397


namespace volume_of_region_l561_561605

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  ∫∫∫ {xyz : ℝ × ℝ × ℝ | f xyz.1 xyz.2 xyz.3 ≤ 6} 1 = 27/2 := 
sorry

end volume_of_region_l561_561605


namespace arnold_protein_intake_l561_561071

def protein_in_collagen_powder (scoops : ℕ) : ℕ := if scoops = 1 then 9 else 18

def protein_in_protein_powder (scoops : ℕ) : ℕ := 21 * scoops

def protein_in_steak : ℕ := 56

def protein_in_greek_yogurt : ℕ := 15

def protein_in_almonds (cups : ℕ) : ℕ := 6 * cups

theorem arnold_protein_intake :
  protein_in_collagen_powder 1 + 
  protein_in_protein_powder 2 + 
  protein_in_steak + 
  protein_in_greek_yogurt + 
  protein_in_almonds 2 = 134 :=
by
  -- Sorry, the proof is omitted intentionally
  sorry

end arnold_protein_intake_l561_561071


namespace triangle_side_proportions_l561_561845

theorem triangle_side_proportions (a b c 2r : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < 2r)
    (h_ap : ∃ d, a = 2r + d ∧ b = 2r + 2d ∧ c = 2r + 3d) :
    ∃ k, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k :=
sorry

end triangle_side_proportions_l561_561845


namespace find_fourth_student_in_sample_l561_561737

theorem find_fourth_student_in_sample :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 48 ∧ 
           (∀ (k : ℕ), k = 29 → 1 ≤ k ∧ k ≤ 48 ∧ ((k = 5 + 2 * 12) ∨ (k = 41 - 12)) ∧ n = 17) :=
sorry

end find_fourth_student_in_sample_l561_561737


namespace solve_trigonometric_problem_l561_561530

noncomputable def trigonometric_problem : Prop :=
  let θ := real.pi / 10  -- 18 degrees
  let η := 3 * real.pi / 10  -- 54 degrees
  let γ := 2 * real.pi / 5  -- 72 degrees
  let δ := real.pi / 5  -- 36 degrees
  sin θ * sin η * sin γ * sin δ = (real.sqrt 5 + 1) / 16

theorem solve_trigonometric_problem : trigonometric_problem :=
by
  sorry

end solve_trigonometric_problem_l561_561530


namespace max_pairs_with_distinct_sums_l561_561139

theorem max_pairs_with_distinct_sums :
  ∃ k : ℕ, (∀ (a b : ℕ), ((a + b) ≤ 3005) → (∀ i j, (i < k → j < k → i ≠ j → (a_i + b_i ≠ a_j + b_j))) → k = 1201 :=
by
  sorry

end max_pairs_with_distinct_sums_l561_561139


namespace people_came_later_l561_561617

theorem people_came_later (lollipop_ratio initial_people lollipops : ℕ) 
  (h1 : lollipop_ratio = 5) 
  (h2 : initial_people = 45) 
  (h3 : lollipops = 12) : 
  (lollipops * lollipop_ratio - initial_people) = 15 := by 
  sorry

end people_came_later_l561_561617


namespace father_walk_time_l561_561449

-- Xiaoming's cycling speed is 4 times his father's walking speed.
-- Xiaoming continues for another 18 minutes to reach B after meeting his father.
-- Prove that Xiaoming's father needs 288 minutes to walk from the meeting point to A.
theorem father_walk_time {V : ℝ} (h₁ : V > 0) (h₂ : ∀ t : ℝ, t > 0 → 18 * V = (V / 4) * t) :
  288 = 4 * 72 :=
by
  sorry

end father_walk_time_l561_561449


namespace cos_pi_over_6_add_alpha_l561_561460

variable (α : ℝ)

theorem cos_pi_over_6_add_alpha (h : sin (π / 3 - α) = 1 / 6) :
    cos (π / 6 + α) = 1 / 6 := by
  sorry

end cos_pi_over_6_add_alpha_l561_561460


namespace sin_15_deg_eq_l561_561098

theorem sin_15_deg_eq : 
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
by
  -- conditions
  have h1 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by sorry
  have h4 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  
  -- proof
  sorry

end sin_15_deg_eq_l561_561098


namespace number_of_divisors_of_square_l561_561266

theorem number_of_divisors_of_square {n : ℕ} (h : ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ n = p * q) : Nat.totient (n^2) = 9 :=
sorry

end number_of_divisors_of_square_l561_561266


namespace find_n_l561_561272

theorem find_n (n : ℤ) 
  (h : (3 + 16 + 33 + (n + 1)) / 4 = 20) : n = 27 := 
by
  sorry

end find_n_l561_561272


namespace remainder_of_arithmetic_sequence_l561_561441

theorem remainder_of_arithmetic_sequence (a d l : ℕ) (h_a : a = 1) (h_d : d = 2) (h_l : l = 19) :
  (let n := (l - a) / d + 1 in (n * (a + l)) / 2 % 12 = 4) :=
by
  sorry

end remainder_of_arithmetic_sequence_l561_561441


namespace range_of_a_values_l561_561635

open Real

noncomputable def distance_from_point_to_line (a : ℝ) : ℝ :=
  abs(a) / sqrt(2)

def circle_radius := 2

def range_of_a : Set ℝ :=
  {x | -3*sqrt 2 < x ∧ x < 3*sqrt 2 }

theorem range_of_a_values (a : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (abs (x + y - a) = 1)) → a ∈ range_of_a :=
by
  sorry

end range_of_a_values_l561_561635


namespace find_m_from_power_function_l561_561662

theorem find_m_from_power_function :
  (∃ a : ℝ, (2 : ℝ) ^ a = (Real.sqrt 2) / 2) →
  (∃ m : ℝ, (m : ℝ) ^ (-1 / 2 : ℝ) = 2) →
  ∃ m : ℝ, m = 1 / 4 :=
by
  intro h1 h2
  sorry

end find_m_from_power_function_l561_561662


namespace find_def_l561_561782

-- Let t_k denote the sum of the k-th powers of the roots of the polynomial x^3 - 6x^2 + 11x - 18
def t (k : ℕ) : ℝ := sorry

-- Given initial conditions
axiom t_0 : t 0 = 3
axiom t_1 : t 1 = 6
axiom t_2 : t 2 = 14
axiom t_3 : t 3 = 72

-- Recursive relationship
axiom rec_relation (k : ℕ) (h : k ≥ 2) : t (k+1) = d * t k + e * t (k-1) + f * t (k-2)

-- We need to prove that d + e + f = 13
theorem find_def (d e f : ℝ) (h1 : 72 = d * 14 + e * 6 + f * 3)
  (h2 : 72 = 6 * 14 - 11 * 6 + 18 * 3) : d + e + f = 13 := by
  sorry

end find_def_l561_561782


namespace exists_point_on_exactly_two_lines_l561_561786

theorem exists_point_on_exactly_two_lines (n : ℕ) (lines : Fin n → set (ℝ × ℝ))
  (h_n : n ≥ 2)
  (h_not_all_concurrent : ¬ ∃ P, ∀ i, ∃ Q ∈ lines i, P = Q)
  (h_no_two_parallel : ∀ i j, i ≠ j → ∃ P ∈ lines i, ∃ Q ∈ lines j, P ≠ Q) :
  ∃ P, ∃ i j, i ≠ j ∧ P ∈ lines i ∧ P ∈ lines j ∧ ∀ k, k ≠ i → k ≠ j → P ∉ lines k :=
sorry

end exists_point_on_exactly_two_lines_l561_561786


namespace fraction_zero_solution_l561_561281

theorem fraction_zero_solution (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := 
by sorry

end fraction_zero_solution_l561_561281


namespace proof_k_k_prime_const_l561_561157

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ≠ (2, 0) →
      x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∃ c : ℝ, a^2 = b^2 + c^2 ∧ c / a = sqrt 3 / 2 ∧ a = 2)

noncomputable def line_l_equation (k : ℝ) : Prop :=
  k ≠ 0 ∧ ∀ (p : ℝ × ℝ), p = (1, 0) → p.2 = k * (p.1 - 1)

noncomputable def intersection (k : ℝ) : Prop :=
  ∀ (E F : ℝ × ℝ), 
    (E.1^2 + 4 * (k * (E.1 - 1))^2 = 4 ∧
    F.1^2 + 4 * (k * (F.1 - 1))^2 = 4) →
    (E.1 + F.1 = 8 * k^2 / (1 + 4 * k^2) ∧ 
    E.1 * F.1 = (4 * k^2 - 4) / (4 * k^2 + 1))

theorem proof_k_k_prime_const :
  ellipse_equation ∧
  line_l_equation k ∧
  intersection k →
  ∃ (k' : ℝ), k * k' = -1/4 := sorry

end proof_k_k_prime_const_l561_561157


namespace largest_prime_divisor_of_factorials_l561_561114

theorem largest_prime_divisor_of_factorials :
  ∀ n m : ℕ, n = 11 → m = 12 → 
  (∃ p : ℕ, p ∈ nat.prime_factors (n ! + m !) ∧ ∀ q : ℕ, q ∈ nat.prime_factors (n ! + m !) → q ≤ p ∧ p = 13) :=
by sorry

end largest_prime_divisor_of_factorials_l561_561114


namespace cos_relationship_l561_561263

theorem cos_relationship (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (hy : 0 ≤ y ∧ y ≤ π)
    (h : ∀ t : ℝ, (- 2 * cos t - (1 / 2) * cos x * cos y) * cos x * cos y - 1 - cos x + cos y - cos (2 * t) < 0) :
    0 ≤ x ∧ x < y ∧ y ≤ π :=
sorry

end cos_relationship_l561_561263


namespace number_of_solutions_10000_l561_561958

theorem number_of_solutions_10000 : 
  ∃ (count : ℕ), count = 2857 ∧ 
  count = (∑ x in finset.filter (λ x : ℕ, (2 ^ x - x ^ 2) % 7 = 0) (finset.range 10000), 1) := 
by
  sorry

end number_of_solutions_10000_l561_561958


namespace problem1_f_ge_sin_problem2_A_eq_B_l561_561184

-- Definition of the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

-- Part (1): Statement for the first proof problem
theorem problem1_f_ge_sin (x : ℝ) : 
  f x 1 2 ≥ sin x := sorry

-- Definitions of the sets A and B
def A (x a b : ℝ) : Prop := f x a b < a + b + 2
def B (x a b : ℝ) : Prop := abs (2 * x - (a + b)) < a + b + 2

-- Part (2): Statement for the second proof problem
theorem problem2_A_eq_B (a b : ℝ) (h: -1 < a ∧ a < b) :
  A = B := sorry

end problem1_f_ge_sin_problem2_A_eq_B_l561_561184


namespace total_cost_formula_sufficient_funds_minimize_cost_l561_561497

/-- 
A store plans to purchase 36 desks, each valued at 20 yuan, in batches within a month. 
Each batch purchases x desks (x is a positive integer), and each batch requires 
a shipping fee of 4 yuan. The storage fee for storing the purchased desks for a month 
is proportional to the total value of the desks purchased per batch (excluding shipping fees).
If 4 desks are purchased per batch, the total shipping and storage fees for the month 
will be 52 yuan. There are only 48 yuan available for the month to cover shipping and storage fees.
-/
def total_cost (x : ℕ) : ℕ := (144 / x) + 4 * x

theorem total_cost_formula  (x : ℕ) (hx : 0 < x) (hx36 : x ≤ 36) : 
  total_cost x = 144 / x + 4 * x := 
sorry

theorem sufficient_funds (x : ℕ) (hx : 4 ≤ x) (hx9 : x ≤ 9) : 
  total_cost x ≤ 48 := 
sorry

theorem minimize_cost (x : ℕ) (hx : x = 6) : 
  total_cost x ≤ total_cost y 
  for (y : ℕ) (hy : 0 < y) (hy36 : y ≤ 36) := 
sorry

end total_cost_formula_sufficient_funds_minimize_cost_l561_561497


namespace div_by_13_l561_561785

theorem div_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) % 13 = 0 :=
by
  sorry

end div_by_13_l561_561785


namespace volume_ratio_is_correct_l561_561493

-- Define the height ratio
def height_ratio : ℝ := 0.8

-- Define the initial heights
def h (i : ℕ) : ℝ :=
  match i with
  | 1     => 1
  | 2     => 0.8
  | 3     => 0.64
  | 4     => 0.512
  | 5     => 0.4096
  | _     => 0  -- For any other index

-- Define the initial radius
def r (i : ℕ) : ℝ :=
  match i with
  | 1     => 1
  | 2     => 0.8
  | 3     => 0.64
  | 4     => 0.512
  | 5     => 0.4096
  | _     => 0  -- For any other index

-- Define the volume formula
def volume (i : ℕ) : ℝ :=
  (1 / 3) * Real.pi * (r i) ^ 2 * (h i) 

-- Define the ratio of interest
def volume_ratio : ℝ := volume 2 / volume 1

-- The main proof goal
theorem volume_ratio_is_correct : volume_ratio = 64 / 125 := by
  sorry

end volume_ratio_is_correct_l561_561493


namespace count_prime_numbers_with_conditions_l561_561679

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def valid_number (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100 ∧ (n % 10) = 3 ∧ sum_of_digits n % 4 ≠ 0 ∧ is_prime n

theorem count_prime_numbers_with_conditions : 
  { n : ℕ | valid_number n }.card = 4 := sorry

end count_prime_numbers_with_conditions_l561_561679


namespace binom_1409_1_equals_1409_l561_561529

theorem binom_1409_1_equals_1409 : nat.choose 1409 1 = 1409 :=
by
  sorry

end binom_1409_1_equals_1409_l561_561529


namespace reflection_of_point_l561_561938

def vector2d := ℝ × ℝ

def midpoint (p1 p2 : vector2d) : vector2d :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def projection (v u : vector2d) : vector2d :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (v proj_v : vector2d) : vector2d :=
  (2 * proj_v.1 - v.1, 2 * proj_v.2 - v.2)

theorem reflection_of_point :
  let v1 := (2 : ℝ, 4 : ℝ)
  let v2 := (10 : ℝ, -2 : ℝ)
  let reflect_vector := (6, 1)
  let point := (1 : ℝ, 6 : ℝ)
  let proj := projection point reflect_vector
  reflection point proj = (107 / 37, -198 / 37) :=
by
  sorry

end reflection_of_point_l561_561938


namespace six_letter_good_words_with_C_l561_561090

def is_good_word (w : list char) : Prop :=
  (∀ (i : ℕ), i < w.length - 1 → 
    (w[i] = 'A' → w[i + 1] ≠ 'B') ∧
    (w[i] = 'B' → w[i + 1] ≠ 'C') ∧
    (w[i] = 'C' → w[i + 1] ≠ 'A')) ∧
  ('C' ∈ w)

theorem six_letter_good_words_with_C :
  {w : list char // w.length = 6 ∧ is_good_word w}.card = 94 :=
sorry

end six_letter_good_words_with_C_l561_561090


namespace part1_shape_of_triangle_part2_roots_of_equation_l561_561152

noncomputable def triangle_shape (a b c : ℝ) (h : -1 = -(a + c) / (2b)) (habc : a + c - b - b = 0) : Prop :=
a = b

theorem part1_shape_of_triangle (a b c : ℝ) (h : (a + c) * (-1)^2 + 2 * b * (-1) + (b - c) = 0) :
  triangle_shape a b c h (by {
  rw [mul_one, mul_neg, mul_one, add_neg_eq_sub] at h,
  rw [← eq_neg_iff_add_eq_zero] at h,
  rw [add_assoc, add_assoc] at h,
  exact h, 
}) :=
eq.symm sorry -- isosceles triangle

theorem part2_roots_of_equation (a b c : ℝ) (h : a = b ∧ b = c) :
  let coeffs := ((2 * a), (2 * a), 0:ℝ) in
  let discriminant := (2*a)^2 - 4*(2*a)*0 in
  discriminant = 0 ∧ 0 * (---=(---coeffs)))
  sorry -- x = 0 and x = -1

end part1_shape_of_triangle_part2_roots_of_equation_l561_561152


namespace david_average_marks_l561_561088

def david_marks := [86, 85, 82, 87, 85]  -- each subject's marks

def total_marks : ℕ := david_marks.foldl (+) 0  -- sum of all marks
def number_of_subjects : ℕ := david_marks.length -- total number of subjects

def average_marks : ℚ := total_marks / number_of_subjects  -- calculate the average

theorem david_average_marks : average_marks = 85 := 
  by
    -- Proof omitted; assertion based on provided conditions and correct answer
    sorry

end david_average_marks_l561_561088


namespace distinct_functions_count_l561_561398

theorem distinct_functions_count :
  ∃ f : {1, 2, 3, ..., 12} → ℤ, 
    f 1 = 1 ∧ 
    (∀ x : {i // i < 12}, |f (x + 1) - f x| = 1) ∧
    (∃ r : ℤ, f 6 = f 1 * r ∧ f 12 = f 1 * (r ^ 2)) ∧
    count_func f = 155 :=
sorry

end distinct_functions_count_l561_561398


namespace volume_of_region_l561_561609

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  (∫ x in -∞..∞, ∫ y in -∞..∞, ∫ z in -∞..∞, if f x y z ≤ 6 then 1 else 0) = 24 := 
sorry

end volume_of_region_l561_561609


namespace number_of_subsets_of_M_l561_561852

-- Define the set M
def M : Set ℕ := {x | x * (x + 2) ≤ 0}

-- Define the proof statement
theorem number_of_subsets_of_M :
  let num_subsets := 2^ (Set.toFinset M).card in num_subsets = 2 :=
by
  sorry

end number_of_subsets_of_M_l561_561852


namespace complex_magnitude_squared_l561_561638

variable {a b : ℝ}
variable (hb : b ≠ 0)

noncomputable def z := a + b * complex.I

theorem complex_magnitude_squared :
  |(z * z)| = |z| ^ 2 ∧ z * z ≠ |z| ^ 2 :=
by
  sorry

end complex_magnitude_squared_l561_561638


namespace distance_from_D_to_midpoint_EF_l561_561295

/-- A triangle defined by three points in a Euclidean space. -/
structure Triangle :=
  (D E F : Point)
  (is_right_angle : is_right_triangle D E F)

/-- Given a right triangle DEF with DE = 15, DF = 9, and EF = 12, 
prove that the distance from D to the midpoint of segment EF is 7.5 units. -/
theorem distance_from_D_to_midpoint_EF (D E F : Point) (t : Triangle D E F)
  (h_DE : dist D E = 15) (h_DF : dist D F = 9) (h_EF : dist E F = 12) :
  dist D (midpoint E F) = 7.5 :=
sorry

end distance_from_D_to_midpoint_EF_l561_561295


namespace arithmetic_sequence_a7_l561_561747

noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence_a7 
  (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 4 - a 1) / 3)
  (h_a1 : a 1 = 3)
  (h_a4 : a 4 = 5) : 
  a 7 = 7 :=
by
  sorry

end arithmetic_sequence_a7_l561_561747


namespace fruit_basket_cost_l561_561479

theorem fruit_basket_cost :
  let bananas_cost   := 4 * 1
  let apples_cost    := 3 * 2
  let strawberries_cost := (24 / 12) * 4
  let avocados_cost  := 2 * 3
  let grapes_cost    := 2 * 2
  bananas_cost + apples_cost + strawberries_cost + avocados_cost + grapes_cost = 28 := 
by
  let groceries_cost := bananas_cost + apples_cost + strawberries_cost + avocados_cost + grapes_cost
  exact sorry

end fruit_basket_cost_l561_561479


namespace rhinos_horn_segment_area_l561_561726

theorem rhinos_horn_segment_area :
  let full_circle_area (r : ℝ) := π * r^2
  let quarter_circle_area (r : ℝ) := (1 / 4) * full_circle_area r
  let half_circle_area (r : ℝ) := (1 / 2) * full_circle_area r
  let larger_quarter_circle_area := quarter_circle_area 4
  let smaller_half_circle_area := half_circle_area 2
  let rhinos_horn_segment_area := larger_quarter_circle_area - smaller_half_circle_area
  rhinos_horn_segment_area = 2 * π := 
by sorry 

end rhinos_horn_segment_area_l561_561726


namespace power_mul_eq_l561_561078

variable (a : ℝ)

theorem power_mul_eq :
  (-a)^2 * a^4 = a^6 :=
by sorry

end power_mul_eq_l561_561078


namespace number_of_divisors_2310_l561_561585

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l561_561585


namespace determine_m_l561_561259

theorem determine_m (a k m : ℝ) (h1 : log a m = c - 3 * log a k) (h2 : c = log a (a^2)) :
  m = a^2 / k^3 := by
sorry

end determine_m_l561_561259


namespace beta_value_l561_561256

variables (α β : ℝ)

noncomputable def sin_alpha := (4 / 7) * real.sqrt 3
noncomputable def cos_alpha_beta := - 11 / 14 

axiom h1 : 0 < α ∧ α < real.pi / 2  -- α is an acute angle
axiom h2 : 0 < β ∧ β < real.pi / 2  -- β is an acute angle

theorem beta_value : sin α = sin_alpha ∧ cos (α + β) = cos_alpha_beta → β = real.pi / 3 :=
sorry

end beta_value_l561_561256


namespace no_such_function_T_exists_l561_561342

-- Define a polynomial P with integer coefficients
noncomputable def P (x : ℤ) : ℤ := sorry

-- Define the main theorem statement
theorem no_such_function_T_exists (P : ℤ → ℤ) (hP_nonconst: ∃ c, P c ≠ 0) :
  ∀ (T : ℤ → ℤ), ¬(∀ n : ℕ, n ≥ 1 → (∃ S : finset ℤ, S.card = P n ∧ ∀ x ∈ S, T^[n] x = x)) :=
by
  sorry

end no_such_function_T_exists_l561_561342


namespace rational_add_positive_square_l561_561705

theorem rational_add_positive_square (a : ℚ) : a^2 + 1 > 0 := by
  sorry

end rational_add_positive_square_l561_561705


namespace integer_count_between_sqrt8_and_sqrt72_l561_561221

theorem integer_count_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ( ∀ x : ℤ, (⌊Real.sqrt 8⌋.to_nat + 1) ≤ x ∧ x ≤ ⌊Real.sqrt 72⌋ - 1 → x = 6 ) :=
by 
  -- Define the floor and ceiling functions
  have sqrt_8_ceil : ⌈Real.sqrt 8⌉ = 3 := sorry
  have sqrt_72_floor : ⌊Real.sqrt 72⌋ = 8 := sorry

  use 6,
  split,
  { refl },
  { intros x hx,
    sorry
  }

-- Additional supporting facts
lemma sqrt_8_approx : Real.sqrt 8 ≈ 2.83 := sorry
lemma sqrt_72_approx : Real.sqrt 72 ≈ 8.49 := sorry

end integer_count_between_sqrt8_and_sqrt72_l561_561221


namespace books_fill_shelf_l561_561309

theorem books_fill_shelf
  (A H S M E : ℕ)
  (h1 : A ≠ H) (h2 : S ≠ M) (h3 : M ≠ H) (h4 : E > 0)
  (Eq1 : A > 0) (Eq2 : H > 0) (Eq3 : S > 0) (Eq4 : M > 0)
  (h5 : A ≠ S) (h6 : E ≠ A) (h7 : E ≠ H) (h8 : E ≠ S) (h9 : E ≠ M) :
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end books_fill_shelf_l561_561309


namespace polynomial_simplification_simplify_expression_evaluate_expression_l561_561024

-- Prove that the correct simplification of 6mn - 2m - 3(m + 2mn) results in -5m.
theorem polynomial_simplification (m n : ℤ) :
  6 * m * n - 2 * m - 3 * (m + 2 * m * n) = -5 * m :=
by {
  sorry
}

-- Prove that simplifying a^2b^3 - 1/2(4ab + 6a^2b^3 - 1) + 2(ab - a^2b^3) results in -4a^2b^3 + 1/2.
theorem simplify_expression (a b : ℝ) :
  a^2 * b^3 - 1/2 * (4 * a * b + 6 * a^2 * b^3 - 1) + 2 * (a * b - a^2 * b^3) = -4 * a^2 * b^3 + 1/2 :=
by {
  sorry
}

-- Prove that evaluating the expression -4a^2b^3 + 1/2 at a = 1/2 and b = 3 results in -26.5
theorem evaluate_expression :
  -4 * (1/2) ^ 2 * 3 ^ 3 + 1/2 = -26.5 :=
by {
  sorry
}

end polynomial_simplification_simplify_expression_evaluate_expression_l561_561024


namespace OA_dot_OB_FM_dot_AB_constant_l561_561296

noncomputable def parabola : Set (ℝ × ℝ) := {(x, y) | x^2 = 4 * y}

structure Point (α : Type*) :=
  (x : α)
  (y : α)

variables (F : Point ℝ) (A B M : Point ℝ)

def focus : Point ℝ := ⟨0, 1⟩

-- Coordinates for points A and B on the parabola
def on_parabola (p : Point ℝ) : Prop := p.x^2 = 4 * p.y

-- Given conditions
def condition_AF_FB (λ : ℝ) : Prop :=
  A.x - F.x = λ * (B.x - F.x) ∧ F.y - A.y = λ * (B.y - F.y)

-- Intersection point M of tangents to parabola at A and B
def intersection_of_tangents (A B M : Point ℝ) : Prop :=
  (M.y = (1 / 2) * A.x * (M.x - A.x) + A.y) ∧ (M.y = (1 / 2) * B.x * (M.x - B.x) + B.y)

-- Conditions and goals
axiom points_on_parabola : on_parabola A ∧ on_parabola B
axiom lambda_condition : ∃ λ : ℝ, condition_AF_FB λ
axiom intersection_condition : intersection_of_tangents A B M

-- Goal 1: Prove the dot product of OA and OB is -3
theorem OA_dot_OB : (A.x * B.x + A.y * B.y) = -3 :=
  sorry

-- Goal 2:  Prove the dot product of FM and AB is constant
theorem FM_dot_AB_constant : ∃ c : ℝ, ∀ A B M, intersection_of_tangents A B M → (F.x * (B.x - A.x) + (F.y - 1) * (B.y - A.y)) = c :=
  sorry

end OA_dot_OB_FM_dot_AB_constant_l561_561296


namespace exists_point_D_l561_561196

noncomputable def point_D (A B C : Point) (h : ¬ collinear A B C) : Point :=
  let D := ... -- define D such that it completes the parallelogram
  have h_D : ∀ d, ∀ (A' B' C' : Point) , (projection d A = A') → (projection d B = B') → (projection d C = C') → (D C' = D A' + D B'),
  from sorry,
  D

theorem exists_point_D (A B C : Point) (h : ¬ collinear A B C) :
  ∃ D, ∀ (d : Line) (A' B' C' : Point),
    projection d A = A' →
    projection d B = B' →
    projection d C = C' →
    vector_eq (vector_from D C') (vector_add (vector_from D A') (vector_from D B')) :=
begin
  use point_D A B C h,
  intros d A' B' C' h_A' h_B' h_C',
  exact h_D d A' B' C' h_A' h_B' h_C',
end

end exists_point_D_l561_561196


namespace number_of_divisors_of_2310_l561_561591

theorem number_of_divisors_of_2310 : 
  let n := 2310 in
  let prime_factors := (2, 1) :: (3, 1) :: (5, 1) :: (7, 1) :: (11, 1) :: [] in
  n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 →
  (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 32 :=
begin
  intro h,
  sorry
end

end number_of_divisors_of_2310_l561_561591


namespace tunnel_length_proof_l561_561912

variable (train_length : ℝ) (train_speed : ℝ) (time_in_tunnel : ℝ)

noncomputable def tunnel_length (train_length train_speed time_in_tunnel : ℝ) : ℝ :=
  (train_speed / 60) * time_in_tunnel - train_length

theorem tunnel_length_proof 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 30) 
  (h_time_in_tunnel : time_in_tunnel = 4) : 
  tunnel_length 2 30 4 = 2 := by
    simp [tunnel_length, h_train_length, h_train_speed, h_time_in_tunnel]
    norm_num
    sorry

end tunnel_length_proof_l561_561912


namespace remaining_black_area_after_five_changes_l561_561069

-- Define a function that represents the change process
noncomputable def remaining_black_area (iterations : ℕ) : ℚ :=
  (3 / 4) ^ iterations

-- Define the original problem statement as a theorem in Lean
theorem remaining_black_area_after_five_changes :
  remaining_black_area 5 = 243 / 1024 :=
by
  sorry

end remaining_black_area_after_five_changes_l561_561069


namespace max_area_rectangle_l561_561853

theorem max_area_rectangle (x : ℝ) (h : 2 * x + 2 * (20 - x) = 40) : 
  ∃ (A : ℝ), A = 100 ∧ (∀ y, (A_area y h) ≤ 100) :=
by
  sorry

end max_area_rectangle_l561_561853


namespace lunks_for_apples_l561_561685

theorem lunks_for_apples : ∀ (lun_to_kun : ℕ) (num_lun : ℕ) (num_kun : ℕ) (kun_to_app : ℕ) (num_kun2 : ℕ) (num_app : ℕ),
  lun_to_kun = 7 ∧ num_kun = 4 ∧ kun_to_app = 3 ∧ num_kun2 = 5 ∧ num_app = 24 → 
  ((num_app * kun_to_app * num_lun / (num_kun2 * lun_to_kun)) ≤ 27) :=
by
  intros lun_to_kun num_lun num_kun kun_to_app num_kun2 num_app
  assume h_conditions
  sorry

end lunks_for_apples_l561_561685


namespace crackers_per_friend_l561_561794

theorem crackers_per_friend (total_crackers : ℕ) (friends : ℕ) (h1 : total_crackers = 81) (h2 : friends = 27) : total_crackers / friends = 3 :=
by
  rw [h1, h2]
  norm_num
  sorry

end crackers_per_friend_l561_561794


namespace power_of_10_digits_l561_561807

theorem power_of_10_digits (n : ℕ) (hn : n > 1) :
  (∃ k : ℕ, (2^(n-1) < 10^k ∧ 10^k < 2^n) ∨ (5^(n-1) < 10^k ∧ 10^k < 5^n)) ∧ ¬((∃ k : ℕ, 2^(n-1) < 10^k ∧ 10^k < 2^n) ∧ (∃ k : ℕ, 5^(n-1) < 10^k ∧ 10^k < 5^n)) :=
sorry

end power_of_10_digits_l561_561807


namespace number_of_divisors_of_2310_l561_561576

theorem number_of_divisors_of_2310 : 
  let n := 2310 in 
  let prime_factors := [2, 3, 5, 7, 11] in
  ∃ k : ℕ, k = prime_factors.length ∧
  (∀ i, i < k → prime_factors.nth i = some 2 ∨ prime_factors.nth i = some 3 ∨ prime_factors.nth i = some 5 ∨ prime_factors.nth i = some 7 ∨ prime_factors.nth i = some 11) →
  (n.factorization.to_nat * 1).0 = 32 :=
begin
  sorry
end

end number_of_divisors_of_2310_l561_561576


namespace train_crossing_time_approx_l561_561953

def speed_km_per_hr := 60 -- Speed in km/h
def train_length_m := 350 -- Length of the train in meters

-- Conversion of speed from km/h to m/s
def speed_m_per_s : ℚ := (speed_km_per_hr * 1000) / 3600

-- Theorem stating that the time for the train to cross the pole is approximately 21 seconds.
theorem train_crossing_time_approx : 
  (train_length_m / speed_m_per_s) ≈ 21 := 
by
  sorry

end train_crossing_time_approx_l561_561953


namespace fish_pond_area_increase_l561_561898

theorem fish_pond_area_increase (x : ℝ) : 
  let original_length := 5 * x,
      original_width := 5 * x - 4,
      new_length := original_length + 2,
      new_width := original_width + 2,
      original_area := original_length * original_width,
      new_area := new_length * new_width,
      area_increase := new_area - original_area
  in area_increase = 20 * x - 4 := 
by 
  unfold let original_length original_width new_length new_width original_area new_area area_increase 
  sorry

end fish_pond_area_increase_l561_561898


namespace triangle_OPQ_equilateral_l561_561879

-- Definitions for the geometrical entities and conditions
variables 
  (O A B C K L P Q : Type) 
  [Triangle ABC : EquilateralTriangle A B C]
  [IsOnSide O ABC]
  [ParallelThroughPoint OKP O A B C]
  [ParallelThroughPoint OLP O A B C]

-- The main theorem
theorem triangle_OPQ_equilateral 
  (h1 : IsPointOnCircle O K L P Q)
  (h2 : ∀ X Y, X ≠ K ∧ Y ≠ L → Parallel X Y)
  (h3 : Angle O K L = Angle P Q O ∧ Angle Q P O = 60):
  IsEquilateralTriangle O P Q :=
by
  sorry

end triangle_OPQ_equilateral_l561_561879


namespace root_difference_l561_561094

theorem root_difference (p : ℝ) (r s : ℝ) 
  (h₁ : r + s = 2 * p) 
  (h₂ : r * s = (p^2 - 4) / 3) : 
  r - s = 2 * (Real.sqrt 3) / 3 :=
by
  sorry

end root_difference_l561_561094


namespace length_of_train_l561_561502

theorem length_of_train (speed_kmh : ℝ) (time_min : ℝ) (tunnel_length_m : ℝ) (train_length_m : ℝ) :
  speed_kmh = 78 → time_min = 1 → tunnel_length_m = 500 → train_length_m = 800.2 :=
by
  sorry

end length_of_train_l561_561502


namespace length_HI_l561_561757

open Real
open EuclideanGeometry

variables (A B C : Point)
variables (O I H D : Point)
variables (r R : ℝ)

-- Define the conditions
def conditions (A B C O I H D : Point) (r R : ℝ) : Prop :=
  ∠BAC = 45 ∧
  dist B C = 1 ∧
  inscribed A B C O ∧
  incircle A B C I D r ∧
  ∠DBC = 15 ∧
  orthocenter A B C H

-- Define the proof problem
theorem length_HI (A B C O I H D : Point) (r R : ℝ) 
  (h : conditions A B C O I H D r R) : 
  dist H I = sqrt 2 - 1/2 :=
sorry

end length_HI_l561_561757


namespace max_isosceles_with_odd_sides_l561_561473

open Function

noncomputable def is_odd (n : ℕ) : Prop := n % 2 = 1

def isosceles_with_odd_sides (tri : Fin 2006) (P : Finset (Fin 2006 × Fin 2006)) : Prop :=
  ∃ a b c : Fin 2006, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    ((a, b) ∈ P ∨ (b, a) ∈ P) ∧
    ((b, c) ∈ P ∨ (c, b) ∈ P) ∧
    ((c, a) ∈ P ∨ (a, c) ∈ P) ∧
    is_odd (abs (a - b)) ∧
    is_odd (abs (b - c)) ∧
    (abs (a - b) = abs (b - c) ∨ abs (b - c) = abs (c - a) ∨ abs (c - a) = abs (a - b))

theorem max_isosceles_with_odd_sides (P : Fin 2006) (diagonals : Finset (Fin 2006 × Fin 2006)) 
  (h1 : P.card = 2006)
  (h2 : diagonals.card = 2003)
  (h3 : ∀ d ∈ diagonals, is_odd (d.1 - d.2)) :
  ∃ T : Finset (Fin 2006).triples, T.card = 1003 ∧ ∀ tri ∈ T, isosceles_with_odd_sides tri diagonals := sorry

end max_isosceles_with_odd_sides_l561_561473


namespace opposite_of_lime_is_black_l561_561378

-- Given colors of the six faces
inductive Color
| Purple | Cyan | Magenta | Silver | Lime | Black

-- Hinged squares forming a cube
structure Cube :=
(top : Color) (bottom : Color) (front : Color) (back : Color) (left : Color) (right : Color)

-- Condition: Magenta is on the top
def magenta_top (c : Cube) : Prop := c.top = Color.Magenta

-- Problem statement: Prove the color opposite to Lime is Black
theorem opposite_of_lime_is_black (c : Cube) (HM : magenta_top c) (HL : c.front = Color.Lime)
    (HBackFace : c.back = Color.Black) : c.back = Color.Black := 
sorry

end opposite_of_lime_is_black_l561_561378


namespace probability_closer_to_5_than_1_round_to_tenth_l561_561492

/-- 
  A point is selected at random from the segment of the number line between 0 and 8. 
  Prove that the probability that the point is closer to 5 than to 1, expressed 
  as a decimal to the nearest tenth, is 0.6.
-/
theorem probability_closer_to_5_than_1_round_to_tenth :
  (let prob : ℚ := 5 / 8 in 
   Real.toRat (Float.toRat (Float.round prob 1))) = 0.6 :=
by
  let prob : ℚ := 5 / 8
  have rounded_prob : Float := Float.round prob 1
  have rat_rounded_prob : ℚ := Real.toRat rounded_prob
  have : rat_rounded_prob = 0.6
  sorry

end probability_closer_to_5_than_1_round_to_tenth_l561_561492


namespace number_of_divisors_2310_l561_561577

-- Define the number whose divisors are being counted
def n : ℕ := 2310

-- Define the prime factorization of the number
def factorization : n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num

-- Define the formula for the number of divisors
def num_divisors (n : ℕ) : ℕ :=
  let e_1 := 1
  let e_2 := 1
  let e_3 := 1
  let e_4 := 1
  let e_5 := 1
  in (e_1 + 1) * (e_2 + 1) * (e_3 + 1) * (e_4 + 1) * (e_5 + 1)

-- State the problem in a theorem
theorem number_of_divisors_2310 : num_divisors n = 32 :=
by
  rw [num_divisors, factorization]
  sorry

end number_of_divisors_2310_l561_561577


namespace number_of_integers_between_sqrt8_sqrt72_l561_561243

theorem number_of_integers_between_sqrt8_sqrt72 :
  let sqrt_8 := Real.sqrt 8
  let sqrt_72 := Real.sqrt 72
  let lower_bound := sqrt_8.ceil
  let upper_bound := sqrt_72.floor
  ∃ n : ℕ, lower_bound ≤ n ∧ n ≤ upper_bound → n = 6 := by
sorry

end number_of_integers_between_sqrt8_sqrt72_l561_561243


namespace sqrt_double_sqrt_four_l561_561869

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_double_sqrt_four :
  sqrt (sqrt 4) = sqrt 2 ∨ sqrt (sqrt 4) = -sqrt 2 :=
by
  sorry

end sqrt_double_sqrt_four_l561_561869


namespace custom_mul_4_3_l561_561261

-- Define the binary operation a*b = a^2 - ab + b^2
def custom_mul (a b : ℕ) : ℕ := a^2 - a*b + b^2

-- State the theorem to prove that 4 * 3 = 13
theorem custom_mul_4_3 : custom_mul 4 3 = 13 := by
  sorry -- Proof will be filled in here

end custom_mul_4_3_l561_561261


namespace max_switches_l561_561875

theorem max_switches (n : ℕ) (heights: Fin n → ℕ) (h_sorted: ∀ (i j : Fin n), i < j → heights i < heights j) :
  ∑ i in Finset.range n, ∑ j in Finset.range (i + 2, n), (j - i - 1) ≤ (n * (n - 1) * (n - 2)) / 6 :=
by sorry

end max_switches_l561_561875


namespace must_be_negative_when_x_is_negative_l561_561708

open Real

theorem must_be_negative_when_x_is_negative (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := 
by
  sorry

end must_be_negative_when_x_is_negative_l561_561708


namespace smallest_n_for_identity_l561_561125

noncomputable def rot_matrix := 
  ![![Real.cos (160 * Real.pi / 180), -Real.sin (160 * Real.pi / 180)], 
    ![Real.sin (160 * Real.pi / 180), Real.cos (160 * Real.pi / 180)]]  -- Define the rotation matrix for 160 degrees

def is_identity {n : ℕ} (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = Matrix.identity 2  -- Check if the matrix is the identity matrix

theorem smallest_n_for_identity :
  ∃ n : ℕ, 0 < n ∧ is_identity (rot_matrix ^ n) ∧ n = 9 := 
sorry

end smallest_n_for_identity_l561_561125


namespace initial_percentage_of_water_l561_561486

variable (P : ℝ) -- The initial percentage of water
variable (initial_volume : ℝ := 150) -- The initial total volume
variable (added_water : ℝ := 30) -- The amount of water added
variable (target_percentage : ℝ := 0.25) -- Target percentage of water in the new mixture

theorem initial_percentage_of_water :
  let initial_water := (P / 100) * initial_volume in
  let new_volume := initial_volume + added_water in
  let required_water := target_percentage * new_volume in
  let initial_water_calc := required_water - added_water in
  initial_water = initial_water_calc → P = 10 :=
by
  intros initial_water new_volume required_water initial_water_calc h
  sorry

end initial_percentage_of_water_l561_561486


namespace minimum_area_triangle_BCE_l561_561646

variables {Point : Type} [MetricSpace Point]
variables (A B C D E : Point) (AC DE : LineSegment Point)

-- Assuming conditions in the problem:
def DE_parallel_to_AC := IsParallel DE AC
def angle_ADC_eq_90_deg := ∠ A D C = π / 2
def AC_eq_12 := AC.length = 12
def CD_eq_6 := SegmentLength C D = 6
def AC_bisects_angle_DAB := BisectsAngle AC (angle A D B)
def angle_BCE_eq_60_deg := ∠ B C E = π / 3

-- Prove that the minimum area of triangle BCE is 27
theorem minimum_area_triangle_BCE :
  DE_parallel_to_AC →
  angle_ADC_eq_90_deg →
  AC_eq_12 →
  CD_eq_6 →
  AC_bisects_angle_DAB →
  angle_BCE_eq_60_deg →
  (triangleArea B C E).min = 27 := 
by
  sorry

end minimum_area_triangle_BCE_l561_561646


namespace fraction_identity_l561_561525

noncomputable def simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ℝ :=
  (1 / (2 * a * b)) + (b / (4 * a))

theorem fraction_identity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  simplify_fraction a b h₁ h₂ = (2 + b^2) / (4 * a * b) :=
by sorry

end fraction_identity_l561_561525


namespace gerald_baseball_supplies_spending_l561_561140

-- Definitions using the given conditions
def season_length_months : ℕ := 4
def charge_per_chore : ℕ := 10
def average_chores_per_month : ℕ := 5
def non_baseball_months : ℕ := 8

-- The proof problem statement
theorem gerald_baseball_supplies_spending :
  let total_savings := average_chores_per_month * charge_per_chore * non_baseball_months in
  let monthly_spending := total_savings / season_length_months in
  monthly_spending = 100 := by
  sorry

end gerald_baseball_supplies_spending_l561_561140


namespace mean_median_difference_l561_561728

def scores_distribution (scores : List ℕ) : Prop :=
  scores.filter (λ x => x = 65).length = 8 ∧
  scores.filter (λ x => x = 75).length = 12 ∧
  scores.filter (λ x => x = 80).length = 6 ∧
  scores.filter (λ x => x = 90).length = 10 ∧
  scores.filter (λ x => x = 100).length = 4

def class_size (scores : List ℕ) : Prop :=
  scores.length = 40

noncomputable def mean (scores : List ℕ) : ℝ :=
  (scores.map (λ x => (x : ℝ))).sum / (scores.length : ℝ)

def median (scores : List ℕ) : ℝ :=
  let sorted := scores.sort
  if scores.length % 2 = 0 then
    ((sorted.get! (scores.length / 2 - 1) : ℝ) + (sorted.get! (scores.length / 2) : ℝ)) / 2
  else
    sorted.get! (scores.length / 2)  

theorem mean_median_difference :
  ∀ (scores : List ℕ),
    class_size scores →
    scores_distribution scores →
    |mean scores - median scores| = 5 :=
by
  intros scores h_class_size h_scores_distribution
  sorry

end mean_median_difference_l561_561728


namespace bisecting_vector_unit_l561_561332

noncomputable def a : ℝ^3 := ![4, 3, 0]
noncomputable def b : ℝ^3 := ![1, -1, 2]
noncomputable def v_unit : ℝ^3 := ![-5 / Real.sqrt 33, -2 / Real.sqrt 33, -2 / Real.sqrt 33]

theorem bisecting_vector_unit :
  ∃ (v : ℝ^3), (∥v∥ = 1) ∧ (2 * b = (k * (a + 5 * v))) :=
  sorry

end bisecting_vector_unit_l561_561332


namespace number_of_integers_between_sqrt8_sqrt72_l561_561245

theorem number_of_integers_between_sqrt8_sqrt72 :
  let sqrt_8 := Real.sqrt 8
  let sqrt_72 := Real.sqrt 72
  let lower_bound := sqrt_8.ceil
  let upper_bound := sqrt_72.floor
  ∃ n : ℕ, lower_bound ≤ n ∧ n ≤ upper_bound → n = 6 := by
sorry

end number_of_integers_between_sqrt8_sqrt72_l561_561245


namespace lunks_for_apples_l561_561689

theorem lunks_for_apples : ∀ (lun_to_kun : ℕ) (num_lun : ℕ) (num_kun : ℕ) (kun_to_app : ℕ) (num_kun2 : ℕ) (num_app : ℕ),
  lun_to_kun = 7 ∧ num_kun = 4 ∧ kun_to_app = 3 ∧ num_kun2 = 5 ∧ num_app = 24 → 
  ((num_app * kun_to_app * num_lun / (num_kun2 * lun_to_kun)) ≤ 27) :=
by
  intros lun_to_kun num_lun num_kun kun_to_app num_kun2 num_app
  assume h_conditions
  sorry

end lunks_for_apples_l561_561689


namespace hyperbola_equation_l561_561150

open Real

theorem hyperbola_equation (e e' : ℝ) (h₁ : 2 * x^2 + y^2 = 2) (h₂ : e * e' = 1) :
  y^2 - x^2 = 2 :=
sorry

end hyperbola_equation_l561_561150


namespace ao_ab_eq_co_cb_l561_561362

variables (A B C M N O : Point)
variable [T : Triangle A B C]
variables (AM AN CM CN AO CO AB CB : Real)
variables (a b c m n o : Real)

-- Define conditions
variable (hM : LineSegment A B M)
variable (hN : LineSegment B C N)
variable (hO : IntersectionPoint CM AN O)
variable (hCond : AM + AN = CM + CN)

-- The goal is to prove AO + AB = CO + CB.
theorem ao_ab_eq_co_cb
  (hM : LineSegment A B M)
  (hN : LineSegment B C N)
  (hO : IntersectionPoint (Segment C M) (Segment A N) O)
  (hCond : AM + AN = CM + CN) :
  AO + AB = CO + CB :=
sorry

end ao_ab_eq_co_cb_l561_561362


namespace parallel_lines_a_l561_561791

theorem parallel_lines_a (a : ℝ) 
  (l1 : ∀ x y : ℝ, ax + 2 * y + a = 0)
  (l2 : ∀ x y : ℝ, 2 * x + ay - a = 0) 
  (parallel : ∀ x y : ℝ, l1 x y → l2 x y → ∀ k : ℝ, a * k = 2) :
  a = 2 ∨ a = -2 := by
  sorry

end parallel_lines_a_l561_561791


namespace correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l561_561011

theorem correct_division (x : ℝ) : x^6 / x^3 = x^3 := by 
  sorry

theorem incorrect_addition (x : ℝ) : ¬(x^2 + x^3 = 2 * x^5) := by 
  sorry

theorem incorrect_multiplication (x : ℝ) : ¬(x^2 * x^3 = x^6) := by 
  sorry

theorem incorrect_squaring (x : ℝ) : ¬((-x^3) ^ 2 = -x^6) := by 
  sorry

theorem only_correct_operation (x : ℝ) : 
  (x^6 / x^3 = x^3) ∧ ¬(x^2 + x^3 = 2 * x^5) ∧ ¬(x^2 * x^3 = x^6) ∧ ¬((-x^3) ^ 2 = -x^6) := 
  by
    exact ⟨correct_division x, incorrect_addition x, incorrect_multiplication x,
           incorrect_squaring x⟩

end correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l561_561011


namespace sum_of_squares_of_solutions_l561_561998

theorem sum_of_squares_of_solutions :
  (∑ x in {x : ℝ | |5 * x| - 7 = 38}, x^2) = 162 := by
  sorry

end sum_of_squares_of_solutions_l561_561998


namespace absolute_value_problem_l561_561709

def x : ℝ := -0.239

theorem absolute_value_problem : 
  |x - 1| + |x - 3| + |x - 5| + ... + |x - 1997| - |x| - |x - 2| - |x - 4| - ... - |x - 1996| = 999 := by
  sorry

end absolute_value_problem_l561_561709


namespace volume_of_region_l561_561608

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  (∫ x in -∞..∞, ∫ y in -∞..∞, ∫ z in -∞..∞, if f x y z ≤ 6 then 1 else 0) = 24 := 
sorry

end volume_of_region_l561_561608


namespace problem_l561_561654

noncomputable def f (x : ℚ) : ℚ := (1 / x^2) + (1 / (x^2 + 1))

theorem problem (x : ℚ) (h : x = -3) : f (f x) = 23600 / 1001 := 
by {
  sorry,
}

end problem_l561_561654


namespace function_solution_l561_561344

open Real

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem function_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x + y + 3) : 
  ∀ x : ℝ, f x = x + 4 :=
begin
  intro x,
  sorry
end

end function_solution_l561_561344


namespace periodic_seq_identity_l561_561092

noncomputable def a_seq (a_1 a_2 : ℂ) : ℕ+ → ℂ
| 1 := a_1
| 2 := a_2
| (n + 1) := 
  have h : (a_seq (n - 1) a_1 a_2) * (a_seq (n + 1) a_1 a_2) - (a_seq n a_1 a_2) ^ 2 = 
              -i * ((a_seq (n + 1) a_1 a_2) + (a_seq (n - 1) a_1 a_2) - 2 * (a_seq n a_1 a_2)) from sorry,
  sorry

theorem periodic_seq_identity (a_1 a_2 : ℂ) 
  (h₁ : a_1^2 + i * a_1 - 1 = 0)
  (h₂ : a_2^2 + i * a_2 - 1 = 0)
  (h₃ : ∀ (n ≥ 2), (a_seq (n + 1) a_1 a_2) * (a_seq (n - 1) a_1 a_2) - (a_seq n a_1 a_2) ^ 2 = 
        -i * ((a_seq (n + 1) a_1 a_2) + (a_seq (n - 1) a_1 a_2) - 2 * (a_seq n a_1 a_2))) :
  ∀ n : ℕ+, 
    (a_seq n a_1 a_2) ^ 2 + (a_seq (n + 1) a_1 a_2) ^ 2 + (a_seq (n + 2) a_1 a_2) ^ 2 = 
    (a_seq n a_1 a_2) * (a_seq (n + 1) a_1 a_2) + (a_seq (n + 1) a_1 a_2) * (a_seq (n + 2) a_1 a_2) + 
    (a_seq (n + 2) a_1 a_2) * (a_seq n a_1 a_2) :=
by
  sorry

end periodic_seq_identity_l561_561092


namespace floor_sqrt_23_squared_l561_561557

theorem floor_sqrt_23_squared : (⌊Real.sqrt 23⌋) ^ 2 = 16 := by
  have h1 : (4:ℝ) < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < (5:ℝ) := sorry
  have h3 : (⌊Real.sqrt 23⌋ : ℝ) = 4 :=
    by sorry
  show 4^2 = 16 from by sorry

end floor_sqrt_23_squared_l561_561557


namespace tangent_line_at_point_l561_561839

theorem tangent_line_at_point :
  ∀ (x y : ℝ) (h : y = x^3 - 2 * x + 1),
    ∃ (m b : ℝ), (1, 0) = (x, y) → (m = 1) ∧ (b = -1) ∧ (∀ (z : ℝ), z = m * x + b) := sorry

end tangent_line_at_point_l561_561839


namespace yarn_for_second_ball_l561_561768

variable (first_ball second_ball third_ball : ℝ) (yarn_used : ℝ)

-- Conditions
variable (h1 : first_ball = second_ball / 2)
variable (h2 : third_ball = 3 * first_ball)
variable (h3 : third_ball = 27)

-- Question: Prove that the second ball used 18 feet of yarn.
theorem yarn_for_second_ball (h1 : first_ball = second_ball / 2) (h2 : third_ball = 3 * first_ball) (h3 : third_ball = 27) :
  second_ball = 18 := by
  sorry

end yarn_for_second_ball_l561_561768


namespace min_number_of_elements_in_B_l561_561773

universe u
variable {α : Type u}

def min_elements (A : ℕ → set α) (B : set α) : ℕ :=
  (finset.univ.image (λ i, A i)).bUnion_finset.card

theorem min_number_of_elements_in_B (A : fin 7 → set α) (h₁ : ∀ i, (finset.univ : finset (fin 7)).card = 2)
  (h₂ : A 0 ∩ A 6 = ∅)
  (h₃ : ∀ i : fin 6, A i ∩ A (i + 1) = ∅) :
  min_elements A (⋃ i, A i) = 5 :=
begin
  sorry
end

end min_number_of_elements_in_B_l561_561773


namespace initial_peanuts_l561_561876

theorem initial_peanuts (x : ℕ) (h : x + 4 = 8) : x = 4 :=
sorry

end initial_peanuts_l561_561876


namespace factorial_trailing_zeros_500_l561_561992

theorem factorial_trailing_zeros_500 :
  let count_factors_of_five (n : ℕ) : ℕ := n / 5 + n / 25 + n / 125
  count_factors_of_five 500 = 124 :=
by
  sorry  -- The proof is not required as per the instructions.

end factorial_trailing_zeros_500_l561_561992


namespace share_of_A_l561_561061

variables (investment_A : ℕ) (investment_B : ℕ) (investment_C : ℕ) (total_profit : ℕ)

def gcd (a b : ℕ) : ℕ := nat.gcd a b

theorem share_of_A
  (hA : investment_A = 6300)
  (hB : investment_B = 4200)
  (hC : investment_C = 10500)
  (profit : total_profit = 14200) :
  let total_ratio_parts := (investment_A / (gcd investment_A (gcd investment_B investment_C))) +
                          (investment_B / (gcd investment_A (gcd investment_B investment_C))) +
                          (investment_C / (gcd investment_A (gcd investment_B investment_C))),
      ratio_A := investment_A / (gcd investment_A (gcd investment_B investment_C)),
      A_share := (ratio_A * total_profit) / total_ratio_parts
  in A_share = 4260 := by {
  sorry
}

end share_of_A_l561_561061


namespace morleys_theorem_l561_561304

theorem morleys_theorem (A B C A1 B1 C1 : Type) 
  [triangle A B C] 
  (trisect_B_C_to_A1 : trisector B C A1)
  (trisect_A_C_to_B1 : trisector A C B1)
  (trisect_A_B_to_C1 : trisector A B C1) : 
  equilateral A1 B1 C1 :=
sorry

end morleys_theorem_l561_561304


namespace actual_cost_of_gas_card_expression_for_y_price_difference_l561_561458

noncomputable theory

open_locale classical

-- Given conditions
def faceValue : ℝ := 1000
def discountRate : ℝ := 0.10
def discountPerLiter : ℝ := 0.30
def originalPriceOil : ℝ := 7.30

-- Calculated results
def discountedFaceValue : ℝ := faceValue * (1 - discountRate)
def actualSpent : ℝ := faceValue - (faceValue * discountRate)

-- The expression for the price per liter after applying the discount
def priceAfterDiscount (x: ℝ) : ℝ := 0.9 * x - discountPerLiter

-- Lean 4 statement for proofs
theorem actual_cost_of_gas_card : actualSpent = 900 :=
by sorry

theorem expression_for_y (x : ℝ) : priceAfterDiscount x = 0.9 * x - 0.27 :=
by sorry

theorem price_difference : originalPriceOil - priceAfterDiscount originalPriceOil = 1 :=
by sorry

end actual_cost_of_gas_card_expression_for_y_price_difference_l561_561458


namespace sphere_radius_eq_l561_561944

noncomputable def volume_sphere (R : ℝ) := (4 / 3) * Real.pi * R ^ 3
noncomputable def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r ^ 2 * h
noncomputable def lateral_surface_area_cone (r s : ℝ) := Real.pi * r * s
noncomputable def total_surface_area_cone (r s : ℝ) := Real.pi * r * s + Real.pi * r ^ 2

theorem sphere_radius_eq (R r s h : ℝ)
  (h_volume_equal : volume_sphere R = volume_cone r h)
  (h_lateral_area : lateral_surface_area_cone r s = 80 * Real.pi)
  (h_total_area : total_surface_area_cone r s = 144 * Real.pi)
  (h_r_eq : r = 8)
  (h_s_eq : s = 10)
  (h_h_eq : h = 6) :
  R = Real.cbrt 96 :=
by
  sorry

end sphere_radius_eq_l561_561944


namespace find_differential_dy_l561_561633

variable (x : ℝ)

def y : ℝ := exp (-x) * sin (2 * x)

theorem find_differential_dy : 
  ∃ dy : ℝ, dy = (exp (-x) * (2 * cos (2 * x) - sin (2 * x))) * dx :=
by
  sorry

end find_differential_dy_l561_561633


namespace area_of_triangle_PF1F2_l561_561983

open Real

section 

def equilateral_hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

def perpendicular_lines (m n : ℝ) : Prop := m^2 + n^2 = 8

def hyperbola_focal_distance : ℝ := 2 * sqrt 2

def hyperbola_def (a : ℝ) := a = 1

noncomputable def area_of_triangle (m n : ℝ) : ℝ := 1/2 * m * n

theorem area_of_triangle_PF1F2 :
  ∃ (P : ℝ × ℝ) (F₁ F₂ : ℝ), 
    equilateral_hyperbola P.1 P.2 ∧
    perpendicular_lines F₁ F₂ ∧
    dist (F₁, 0) (F₂, 0) = hyperbola_focal_distance ∧
    area_of_triangle F₁ F₂ = 1 :=
by
  sorry

end

end area_of_triangle_PF1F2_l561_561983


namespace number_of_solutions_l561_561254

theorem number_of_solutions :
  {p : ℝ × ℝ // p.1 + 2 * p.2 = 4 ∧ | |p.1| - |p.2| | = sin (Real.pi / 4)}.set.card = 2 :=
begin
  sorry

end number_of_solutions_l561_561254


namespace dan_has_more_balloons_l561_561989

-- Constants representing the number of balloons Dan and Tim have
def dans_balloons : ℝ := 29.0
def tims_balloons : ℝ := 4.142857143

-- Theorem: The ratio of Dan's balloons to Tim's balloons is 7
theorem dan_has_more_balloons : dans_balloons / tims_balloons = 7 := 
by
  sorry

end dan_has_more_balloons_l561_561989


namespace lunks_to_apples_l561_561691

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l561_561691


namespace reciprocal_neg_5_l561_561863

theorem reciprocal_neg_5 : ∃ x : ℚ, -5 * x = 1 ∧ x = -1/5 :=
by
  sorry

end reciprocal_neg_5_l561_561863


namespace find_radius_of_sphere_touching_midpoints_l561_561434

-- Define the geometrical setup and conditions
structure Triangle :=
(K L M N : Point)
(KL LM KN : ℝ)
(angle_KLM angle_LKN : ℝ)
(equal_triangles : K ≠ L ∧ K ≠ M ∧ K ≠ N ∧ L ≠ M ∧ L ≠ N ∧ M ≠ N ∧
                   Euc_2.Triangle K L M ∧ Euc_2.Triangle K L N ∧ 
                   Euc_2.Triangle (A B C) = Euc_2.Triangle (P Q R))

def angle60 := π / 3

noncomputable def radius_sphere {T : Triangle}
  (h1 : T.angle_KLM = angle60)
  (h2 : T.angle_LKN = angle60)
  (h3 : T.KL = 1)
  (h4 : T.LM = 6)
  (h5 : T.KN = 6)
  (h6 : ArePerpendicular (Plane T.K T.L T.M) (Plane T.K T.L T.N)) : ℝ :=
  sorry

-- Define the theorem to state the final radius result
theorem find_radius_of_sphere_touching_midpoints (T : Triangle)
  (h1 : T.angle_KLM = angle60)
  (h2 : T.angle_LKN = angle60)
  (h3 : T.KL = 1)
  (h4 : T.LM = 6)
  (h5 : T.KN = 6)
  (h6 : ArePerpendicular (Plane T.K T.L T.M) (Plane T.K T.L T.N)) :
  radius_sphere h1 h2 h3 h4 h5 h6 = (real.sqrt 127) / 2 :=
  sorry

end find_radius_of_sphere_touching_midpoints_l561_561434


namespace area_of_quadrilateral_PQRS_l561_561368

structure Quadrilateral :=
  (P Q R S : Type)
  (PQ QR RS PS : ℝ)
  (anglePQR : ℝ)
  (convex : Bool)

def quadrilateral_data : Quadrilateral := {
  P := ℝ,
  Q := ℝ,
  R := ℝ,
  S := ℝ,
  PQ := 6,
  QR := 8,
  RS := 15,
  PS := 17,
  anglePQR := 90,
  convex := true
}

theorem area_of_quadrilateral_PQRS : quadrilateral_data.PQ = 6 ∧
  quadrilateral_data.QR = 8 ∧
  quadrilateral_data.RS = 15 ∧
  quadrilateral_data.PS = 17 ∧
  quadrilateral_data.anglePQR = 90 ∧
  quadrilateral_data.convex = true →
  area quadrilateral_data ≈ 98.5 :=
by
  sorry

end area_of_quadrilateral_PQRS_l561_561368


namespace L_M_N_collinear_l561_561347

-- Definitions and conditions
variable (A B C L M N : Type)
variable [has_mem A C] [has_mem A B]
variable (h_right_triangle : right_triangle_at A B C)
variable (hL_on_BC : L ∈ [B, C])
variable (h_circumcircle_ABL : ∃ (circ : Type) (has_mem_in_circ : has_mem circ (Δ A B L)), 
  (M ∈ circ) ∧ (M ∈ [A, C]))
variable (h_circumcircle_CAL : ∃ (circ : Type) (has_mem_in_circ : has_mem circ (Δ C A L)), 
  (N ∈ circ) ∧ (N ∈ [A, B]))

-- Proof statement
theorem L_M_N_collinear : collinear L M N :=
sorry

end L_M_N_collinear_l561_561347


namespace lunks_needed_for_24_apples_l561_561703

-- Define the conditions as Lean definitions
def lunks_per_kunks := 7 / 4
def kunks_per_apples := 3 / 5
def apples_needed := 24

-- State the theorem
theorem lunks_needed_for_24_apples : 
  let k := (3 * apples_needed) / 5 in 
  let rounded_k := k.ceil in 
  let l := (7 * rounded_k) / 4 in 
  l.ceil = 27 :=
by 
  let k := (3 * apples_needed) / 5
  let rounded_k := k.ceil
  let l := (7 * rounded_k) / 4
  have h1 : k = (3 * apples_needed) / 5 := rfl
  have h2 : rounded_k = k.ceil := rfl
  have h3 : l = (7 * rounded_k) / 4 := rfl
  show l.ceil = 27, from sorry

end lunks_needed_for_24_apples_l561_561703


namespace smallest_value_x_abs_eq_32_l561_561126

theorem smallest_value_x_abs_eq_32 : ∃ x : ℚ, (x = -29 / 5) ∧ (|5 * x - 3| = 32) ∧ 
  (∀ y : ℚ, (|5 * y - 3| = 32) → (x ≤ y)) :=
by
  sorry

end smallest_value_x_abs_eq_32_l561_561126


namespace num_of_odd_digit_5_digit_nums_div_by_5_and_3_l561_561202

theorem num_of_odd_digit_5_digit_nums_div_by_5_and_3 : 
  {n : ℕ // 10000 ≤ n ∧ n ≤ 99999 ∧ (∀ d ∈ (int.to_digits 10 n), d % 2 = 1) ∧ n % 5 = 0 ∧ n % 3 = 0}.card = 250 :=
sorry

end num_of_odd_digit_5_digit_nums_div_by_5_and_3_l561_561202


namespace sum_linear_function_l561_561336

theorem sum_linear_function (f : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ x, f x = 2 * x - 1)
  (h2 : f 8 = 15)
  (h3 : ∃ a b r, f 2 = a * r ∧ f 5 = a * r^2 ∧ f 14 = a * r^3)
  : ∑ i in Finset.range n, f (i + 1) = n^2 := by
  sorry

end sum_linear_function_l561_561336


namespace find_x_minus_y_l561_561787

noncomputable theory

variables {x y : ℝ}

def is_prime (n : ℝ) : Prop := ∃ a b : ℝ, a * b = n ∧ a > 1 ∧ b > 1

theorem find_x_minus_y (hx : is_prime (x - y)) (hxy2 : is_prime (x^2 - y^2)) (hxy3 : is_prime (x^3 - y^3)) : x - y = 3 := 
by {
  sorry
}

end find_x_minus_y_l561_561787


namespace find_1482_1484_th_digits_l561_561507

-- Define the predicate stating the numbers start with digit 1
def numbers_start_with_1 (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ((10^k) ≤ n ∧ n < (10^(k+1)) ∧ (n / (10^k)) = 1)

-- Define the predicate for the digit at specific position in the sequence
def digit_at_position (pos v : ℕ) : Prop :=
  ∃ n m : ℕ, numbers_start_with_1 n ∧ (sequence_len n m) ∧ (pos ≤ m) ∧ (v = digit_at n (pos - sum_prev n))

-- Lean statement for the particular mathematical problem
theorem find_1482_1484_th_digits :
  (digit_at_position 1482 1) ∧ (digit_at_position 1483 2) ∧ (digit_at_position 1484 9) :=
sorry

end find_1482_1484_th_digits_l561_561507


namespace number_of_integers_between_sqrt8_and_sqrt72_l561_561214

theorem number_of_integers_between_sqrt8_and_sqrt72 : 
  let a := Int.ceil (Real.sqrt 8)
  let b := Int.floor (Real.sqrt 72)
  b - a + 1 = 6 :=
begin
  sorry
end

end number_of_integers_between_sqrt8_and_sqrt72_l561_561214


namespace solution_set_of_inequality_l561_561166

-- Definition of the function f
def f (x : ℝ) : ℝ := if x >= 0 then x^3 + 2*x else (-x)^3 + 2*(-x)

lemma even_function (x : ℝ) : f (-x) = f x := by
  suffices h : (-x)^3 + 2*(-x) = x^3 + 2*x by
    simp [f, if_neg (not_le_of_gt (neg_pos.mpr (lt_of_le_not_le (le_abs_self x) (not_le.mpr (neg_pos.mpr (abs_pos.mpr (ne_of_lt (abs_pos.mpr (neg_ne_zero.mpr (neg_pos.mpr (ne_of_lt (abs_pos.mpr (ne_of_zero.mpr (x ≠ 0))))))))))))))), h]
  ring

-- Main theorem stating the solution set
theorem solution_set_of_inequality : {x : ℝ | f (x - 2) < 3} = set.Ioo 1 3 := by
  sorry

end solution_set_of_inequality_l561_561166


namespace range_of_a_l561_561187

noncomputable def inequality_always_true (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - a * x + 2 > 0

theorem range_of_a :
  { a : ℝ | ∀ x : ℝ, a * x^2 - a * x + 2 > 0 } = { a : ℝ | 0 ≤ a ∧ a < 8 } :=
begin
  sorry
end

end range_of_a_l561_561187


namespace part_1_part_2_l561_561191

-- Define the sequence
def a_sequence : ℕ → ℝ
| 0 := 1/2
| (n + 1) := a_sequence n ^ 2 + a_sequence n + 1

-- First part: Prove that a_{n+1}/a_n ≥ 3
theorem part_1 (n : ℕ) : 
  a_sequence (n+1) / a_sequence n ≥ 3 :=
sorry

-- Define the sum of the first n terms of the reciprocal sequence
def S_n (n : ℕ) : ℝ :=
(∑ i in Finset.range n, 1 / a_sequence (i + 1))

-- Second part: Prove that S_n < 3
theorem part_2 (n : ℕ) : 
  S_n n < 3 :=
sorry

end part_1_part_2_l561_561191


namespace floor_sqrt_23_squared_eq_16_l561_561554

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end floor_sqrt_23_squared_eq_16_l561_561554


namespace measure_obtuse_angle_PSQ_l561_561743

-- Define the right triangle with angles 45 degrees each at P and Q
variables {P Q R S : Type}
variables [RightTriangle P Q R]

-- Define the given conditions: ∠P = 45°, ∠Q = 45°
def angle_P := 45
def angle_Q := 45

-- Define the fact that the angle bisectors of ∠P and ∠Q intersect at S
def bisectors_intersect_at_S (P Q R S : Type) : Prop :=
  -- This can be used to elaborate how S is defined in terms of bisectors
  sorry 

-- Prove the measure of obtuse angle PSQ
theorem measure_obtuse_angle_PSQ :
  angle_P = 45 ∧ angle_Q = 45 ∧ bisectors_intersect_at_S P Q R S →
  measure_angle PSQ = 135 :=
begin
  assume h,
  sorry
end

end measure_obtuse_angle_PSQ_l561_561743


namespace smallest_possible_number_l561_561796

-- Define the initial number as a list of digits
def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the transformation rule
def transform (digits : List ℕ) : List ℕ := sorry

-- Define the target number after transformations
def target_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

-- Statement to prove that target_number is the smallest possible number obtained from initial_number
theorem smallest_possible_number : ∃ (seq : List ℕ), 
  (seq = initial_number ∨ ∃ k, ((List.range k).all (λ i, transform seq = transform (transform seq))) ∧ seq = target_number) :=
sorry

end smallest_possible_number_l561_561796


namespace find_polynomial_l561_561937

theorem find_polynomial (A : ℤ[x]) :
  A = -3 * X^2 + X + 3 →
  A + 2 * X^2 - 4 * X - 3 = - X^2 - 3 * X := by
  intro h
  rw [h]
  sorry

end find_polynomial_l561_561937


namespace stickers_per_bottle_l561_561549

-- Definitions based on conditions
def initial_bottles : ℕ := 10
def lost_at_school : ℕ := 2
def stolen_at_dance_practice : ℕ := 1
def total_stickers : ℕ := 21

-- Proof problem statement
theorem stickers_per_bottle : 
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance_practice in
  remaining_bottles > 0 →
  total_stickers % remaining_bottles = 0 →
  total_stickers / remaining_bottles = 3 :=
by
  intros h_rem h_div
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance_practice
  have pos_rem : remaining_bottles = 7 := rfl
  have div_exact : total_stickers = 3 * remaining_bottles := sorry
  show total_stickers / remaining_bottles = 3 from sorry

end stickers_per_bottle_l561_561549


namespace volume_of_region_l561_561610

def f (x y z : ℝ) : ℝ :=
  |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z|

theorem volume_of_region : 
  (∫ x in -∞..∞, ∫ y in -∞..∞, ∫ z in -∞..∞, if f x y z ≤ 6 then 1 else 0) = 24 := 
sorry

end volume_of_region_l561_561610


namespace custom_op_subtraction_l561_561710

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_subtraction :
  (custom_op 4 2) - (custom_op 2 4) = -8 := by
  sorry

end custom_op_subtraction_l561_561710


namespace pet_store_satisfaction_l561_561936

theorem pet_store_satisfaction :
  let puppies := 15
  let kittens := 6
  let hamsters := 8
  let friends := 3
  puppies * kittens * hamsters * friends.factorial = 4320 := by
  sorry

end pet_store_satisfaction_l561_561936


namespace expectation_example_a_expectation_example_b_l561_561113

-- Part (a)
theorem expectation_example_a (X Y : ℝ → ℝ) [ProbabilitySpace ℝ] :
  (Expectation X = 5) → (Expectation Y = 3) → (Expectation (X + 2 * Y) = 11) :=
by
  intros h1 h2
  sorry

-- Part (b)
theorem expectation_example_b (X Y : ℝ → ℝ) [ProbabilitySpace ℝ] :
  (Expectation X = 2) → (Expectation Y = 6) → (Expectation (3 * X + 4 * Y) = 30) :=
by
  intros h1 h2
  sorry

end expectation_example_a_expectation_example_b_l561_561113


namespace largest_prime_factor_of_4752_l561_561004

theorem largest_prime_factor_of_4752 : ∃ p : ℕ, nat.prime p ∧ p ∣ 4752 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 4752 → q ≤ p :=
begin
  -- Proof goes here
  sorry
end

end largest_prime_factor_of_4752_l561_561004


namespace min_value_l561_561120

noncomputable def conditions (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧ 
  (27^x + y^4 - 3^x - 1 = 0)

theorem min_value (x y : ℝ) (h : conditions x y) : ∃ x y, (x^3 + y^3 = -1) :=
sorry

end min_value_l561_561120


namespace circle_condition_l561_561834

def represents_circle (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x + 1/2)^2 + (y + m)^2 = 5/4 - m

theorem circle_condition (m : ℝ) : represents_circle m ↔ m < 5/4 :=
by sorry

end circle_condition_l561_561834


namespace polynomial_sum_l561_561780

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
noncomputable def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
noncomputable def h (x : ℝ) : ℝ := 4 * x^2 + 6 * x + 3
noncomputable def j (x : ℝ) : ℝ := 3 * x^2 - x + 2

theorem polynomial_sum :
  (λ x => f x + g x + h x + j x) = (λ x => -x^2 + 11 * x - 9) :=
by
  intro x
  sorry

end polynomial_sum_l561_561780


namespace distance_big_rock_correct_l561_561900

noncomputable def rower_in_still_water := 7 -- km/h
noncomputable def river_flow := 2 -- km/h
noncomputable def total_trip_time := 1 -- hour

def distance_to_big_rock (D : ℝ) :=
  (D / (rower_in_still_water - river_flow)) + (D / (rower_in_still_water + river_flow)) = total_trip_time

theorem distance_big_rock_correct {D : ℝ} (h : distance_to_big_rock D) : D = 45 / 14 :=
sorry

end distance_big_rock_correct_l561_561900


namespace count_divisibles_between_100_and_500_l561_561253

-- Statement of the math proof problem
theorem count_divisibles_between_100_and_500 :
  (set.count {n : ℕ | 100 ≤ n ∧ n ≤ 500 ∧ n % 7 = 0}) = 57 :=
sorry

end count_divisibles_between_100_and_500_l561_561253


namespace number_of_zeros_of_f_l561_561194

noncomputable def sgn : ℝ → ℝ := 
  λ x, if x > 0 then 1 else if x = 0 then 0 else -1

noncomputable def f (x : ℝ) : ℝ := sgn (Real.log x) - Real.log x

theorem number_of_zeros_of_f : 
  (∃ x > 0, f x = 0) ∧ (∃ x = 1, f x = 0) ∧ (∃ x > 0, x < 1, f x = 0) :=
sorry

end number_of_zeros_of_f_l561_561194


namespace number_of_integers_between_sqrt8_sqrt72_l561_561239

theorem number_of_integers_between_sqrt8_sqrt72 :
  let sqrt_8 := Real.sqrt 8
  let sqrt_72 := Real.sqrt 72
  let lower_bound := sqrt_8.ceil
  let upper_bound := sqrt_72.floor
  ∃ n : ℕ, lower_bound ≤ n ∧ n ≤ upper_bound → n = 6 := by
sorry

end number_of_integers_between_sqrt8_sqrt72_l561_561239


namespace calc_7_op_4_minus_4_op_7_l561_561091

def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

theorem calc_7_op_4_minus_4_op_7 : (op 7 4) - (op 4 7) = -12 := by
  sorry

end calc_7_op_4_minus_4_op_7_l561_561091


namespace speed_of_current_l561_561059

def rowing_speed_still_water : ℝ := 120 -- km/h
def distance_covered_downstream : ℝ := 0.5 -- km
def time_taken_seconds : ℝ := 9.99920006399488 -- seconds
def seconds_to_hours_factor : ℝ := 1 / 3600

noncomputable def time_taken_hours : ℝ := time_taken_seconds * seconds_to_hours_factor

theorem speed_of_current :
  let downstream_speed := distance_covered_downstream / time_taken_hours in
  let speed_of_current := downstream_speed - rowing_speed_still_water in
  speed_of_current = 60 :=
by
  sorry

end speed_of_current_l561_561059


namespace transformed_parabolas_combined_l561_561934

theorem transformed_parabolas_combined (a b c : ℝ) :
  let f (x : ℝ) := a * (x - 3) ^ 2 + b * (x - 3) + c
  let g (x : ℝ) := -a * (x + 4) ^ 2 - b * (x + 4) - c
  ∀ x, (f x + g x) = -14 * a * x - 19 * a - 7 * b :=
by
  -- This is a placeholder for the actual proof using the conditions
  sorry

end transformed_parabolas_combined_l561_561934


namespace R_depends_on_d_and_n_l561_561396

def arith_seq_sum (a d n : ℕ) (S1 S2 S3 : ℕ) : Prop := 
  (S1 = n * (a + (n - 1) * d / 2)) ∧ 
  (S2 = n * (2 * a + (2 * n - 1) * d)) ∧ 
  (S3 = 3 * n * (a + (3 * n - 1) * d / 2))

theorem R_depends_on_d_and_n (a d n S1 S2 S3 : ℕ) 
  (hS1 : S1 = n * (a + (n - 1) * d / 2))
  (hS2 : S2 = n * (2 * a + (2 * n - 1) * d))
  (hS3 : S3 = 3 * n * (a + (3 * n - 1) * d / 2)) 
  : S3 - S2 - S1 = 2 * n^2 * d  :=
by
  sorry

end R_depends_on_d_and_n_l561_561396


namespace y_axis_symmetry_l561_561275

theorem y_axis_symmetry (x y : ℝ) (P : ℝ × ℝ) (hx : P = (-5, 3)) : 
  (P.1 = -5 ∧ P.2 = 3) → (P.1 * -1, P.2) = (5, 3) :=
by
  intro h
  rw [hx]
  simp [Neg.neg, h]
  sorry

end y_axis_symmetry_l561_561275


namespace garden_length_l561_561844

theorem garden_length (w l : ℝ) (h1 : l = 2 + 3 * w) (h2 : 2 * l + 2 * w = 100) : l = 38 :=
sorry

end garden_length_l561_561844


namespace ratio_of_toms_age_to_anns_age_l561_561511

def ann_age := 6

def sum_ages_in_10_years := 38

def ratio_of_ages (x : ℕ) : Prop := 
  let tom_age := ann_age * x in
  ann_age + tom_age = 3 * ann_age

theorem ratio_of_toms_age_to_anns_age : 
  ∀ (x : ℕ), 
    (6 + (6 * x) + 10 + 10 = sum_ages_in_10_years) → 
    ratio_of_ages x := 
by
  intros x h
  sorry

end ratio_of_toms_age_to_anns_age_l561_561511


namespace market_value_of_house_l561_561762

theorem market_value_of_house 
  (M : ℝ) -- Market value of the house
  (S : ℝ) -- Selling price of the house
  (P : ℝ) -- Pre-tax amount each person gets
  (after_tax : ℝ := 135000) -- Each person's amount after taxes
  (tax_rate : ℝ := 0.10) -- Tax rate
  (num_people : ℕ := 4) -- Number of people splitting the revenue
  (over_market_value_rate : ℝ := 0.20): 
  S = M + over_market_value_rate * M → 
  (num_people * P) = S → 
  after_tax = (1 - tax_rate) * P → 
  M = 500000 := 
by
  sorry

end market_value_of_house_l561_561762


namespace rational_terms_in_expansion_l561_561300

theorem rational_terms_in_expansion (x : ℝ) : 
  ∀ a b : ℝ, 
  (C 8 0, C 8 1, C 8 2).form_arithmetic_sequence → 
  (a = x) → 
  (b = 1/2) → 
  let T (r : ℕ) := (binomial 8 r) * a ^ (8 - r) * b ^ r in 
  (T 0 = x ^ 8) ∧ (T 4 = x ^ 4) ∧ (T 8 = 1) := 
by
  unfold binomial
  unfold form_arithmetic_sequence
  unfold T
  sorry

end rational_terms_in_expansion_l561_561300


namespace problem1_problem2_C_problem2_area_l561_561285
noncomputable def triangleABC (a b c A B C : ℝ) := 
  ∃ (a b c : ℝ) (A B C : ℝ), 
    a = 2 ∧ c = 3/2 ∧ cos A = 1/3 ∧ (a, b, c) = (A, B, C)

theorem problem1 (a b c A B C : ℝ) (h1: cos A = 1/3): 
  cos^2 ((B+C)/2) + cos(2*A) = -4/9 := sorry

theorem problem2_C (a b c A B C : ℝ) (h1 : a = 2) (h2 : c = 3/2) (h3 : cos A = 1/3) :
  C = π/4 := sorry

theorem problem2_area (a b c A B C : ℝ) (h1 : a = 2) (h2 : c = 3/2) (h3 : cos A = 1/3) :
  let area := 1/2 * a * c * sin B in
  area = 1 + sqrt 2 / 4 := sorry

end problem1_problem2_C_problem2_area_l561_561285


namespace odd_function_a_monotonic_increasing_a_lt_0_range_k_over_a_l561_561777

noncomputable def f (a x : ℝ) : ℝ := (2^x + a) / (2^x - a)

-- Statement 1
theorem odd_function_a (a : ℝ) (h : ∀ x, f a (-x) = -f a x) : a = 1 ∨ a = -1 := sorry

-- Statement 2
theorem monotonic_increasing_a_lt_0 (a : ℝ) (h : a < 0) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ := sorry

-- Statement 3
theorem range_k_over_a (a m n k : ℝ) (h1 : a ≠ 0) (h2 : m < n)
    (h3 : set.range (λ x, f a x) = set.Icc (k / 2^m) (k / 2^n))
    : set.range (λ x, k / a) = set.Ioc 0 (3 - 2 * (2:ℝ).sqrt) ∪ {-1} := sorry

end odd_function_a_monotonic_increasing_a_lt_0_range_k_over_a_l561_561777


namespace cassie_brian_meeting_time_l561_561527

noncomputable def time_in_hours (hour : ℕ) (minute : ℕ) : ℝ := hour + minute / 60.0

def departure_cassie := time_in_hours 7 45
def departure_brian := time_in_hours 8 15
def distance := 75
def rate_cassie := 15
def rate_brian := 18
def rest_cassie := 15.0 / 60.0 -- 15 minutes converted to hours
def meeting_time_expected := time_in_hours 10 25

theorem cassie_brian_meeting_time :
  let x := 87.75 / 33
  let meeting_time := departure_cassie + x
  meeting_time = meeting_time_expected :=
by
  sorry

end cassie_brian_meeting_time_l561_561527


namespace part_I_part_II_l561_561673

-- Definition of the sequence a_n with given conditions
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else (n^2 + n) / 2

-- Define the sum of the first n terms S_n
def S_n (n : ℕ) : ℕ :=
  (n + 2) / 3 * a_n n

-- Define the sequence b_n in terms of a_n
def b_n (n : ℕ) : ℚ := 1 / a_n n

-- Define the sum of the first n terms of b_n
def T_n (n : ℕ) : ℚ :=
  2 * (1 - 1 / (n + 1))

-- Theorem statement for part (I)
theorem part_I (n : ℕ) : 
  a_n 2 = 3 ∧ a_n 3 = 6 ∧ (∀ (n : ℕ), n ≥ 2 → a_n n = (n^2 + n) / 2) := sorry

-- Theorem statement for part (II)
theorem part_II (n : ℕ) : 
  T_n n = 2 * (1 - 1 / (n + 1)) := sorry

end part_I_part_II_l561_561673


namespace distance_from_O_to_plane_ABC_l561_561660

theorem distance_from_O_to_plane_ABC (O S A B C : Point)
  (h1 : S ∈ sphere O 10)
  (h2 : A ∈ sphere O 10)
  (h3 : B ∈ sphere O 10)
  (h4 : C ∈ sphere O 10)
  (hSA : dist S A = dist S B)
  (hSB : dist S B = dist S C)
  (hSC : dist S C = dist S A)
  (hAB : dist A B = dist B C)
  (hAngleACB : angle A C B = 90)
  : dist_from_point_to_plane O A B C = 5 := sorry

end distance_from_O_to_plane_ABC_l561_561660


namespace domain_of_f_l561_561833

noncomputable def f (x : ℝ) : ℝ := 
  (1 / (sqrt (2 - 3 * x))) + ((2 * x - 1) ^ 0)

def domain_condition1 (x : ℝ) := 2 - 3 * x > 0

def domain_condition2 (x : ℝ) := (2 * x) ≠ 1

def in_domain (x : ℝ) := x < 2 / 3 ∧ x ≠ 1 / 2

theorem domain_of_f (x : ℝ) : (in_domain x) ↔ (domain_condition1 x ∧ domain_condition2 x) := by
  sorry

end domain_of_f_l561_561833


namespace find_c30_l561_561190

noncomputable def polynomial_product (c : ℕ → ℕ) (z : ℤ) : ℤ :=
  ∏ k in finset.range 31.erase 0, (1 - z^k)^(c k)

def simplified_product : ℤ := 1 - 3 * z

theorem find_c30 (c : ℕ → ℕ) (z : ℤ) (h : polynomial_product c z = simplified_product) :
  c 30 = (43046721 * 43046720) / (2 * 16) :=
sorry

end find_c30_l561_561190


namespace total_area_is_71_l561_561535

noncomputable def area_of_combined_regions 
  (PQ QR RS TU : ℕ) 
  (PQRSTU_is_rectangle : true) 
  (right_angles : true): ℕ :=
  let Area_PQRSTU := PQ * QR
  let VU := TU - PQ
  let WT := TU - RS
  let Area_triangle_PVU := (1 / 2) * VU * PQ
  let Area_triangle_RWT := (1 / 2) * WT * RS
  Area_PQRSTU + Area_triangle_PVU + Area_triangle_RWT

theorem total_area_is_71
  (PQ QR RS TU : ℕ) 
  (h1 : PQ = 8)
  (h2 : QR = 6)
  (h3 : RS = 5)
  (h4 : TU = 10)
  (PQRSTU_is_rectangle : true)
  (right_angles : true) :
  area_of_combined_regions PQ QR RS TU PQRSTU_is_rectangle right_angles = 71 :=
by
  -- The proof is omitted as per the instructions
  sorry

end total_area_is_71_l561_561535


namespace correct_variance_l561_561941

variables {α : Type*} [field α]

noncomputable theory

def sample_M (n : ℕ) (x : ℕ → α) : set α := {xi | ∃ i (hi : i < n), xi = x i}
def mean (s : set α) : α := s.sum / s.card

def sample_N (x : α → α) (M : set α) : set α := {x xi | xi ∈ M}

def variance (s : set α) (μ : α): α := 
  (s.sum (λ xi, (xi - μ)^2)) / (s.card : α)

theorem correct_variance (n : ℕ) (x : ℕ → α) 
  (hM_mean : mean (sample_M n x) = 5)
  (hN_mean : mean (sample_N (λ xi, xi^2) (sample_M n x)) = 34) :
  variance (sample_M n x) 5 = 9 := 
sorry

end correct_variance_l561_561941


namespace candy_sampling_percentage_l561_561727

theorem candy_sampling_percentage (total_percentage caught_percentage not_caught_percentage : ℝ) 
  (h1 : caught_percentage = 22 / 100) 
  (h2 : total_percentage = 24.444444444444443 / 100) 
  (h3 : not_caught_percentage = 2.444444444444443 / 100) :
  total_percentage = caught_percentage + not_caught_percentage :=
by
  sorry

end candy_sampling_percentage_l561_561727


namespace number_of_streams_l561_561746

theorem number_of_streams (S A B C D : Type) (f : S → A) (f1 : A → B) :
  (∀ (x : ℕ), x = 1000 → 
  (x * 375 / 1000 = 375 ∧ x * 625 / 1000 = 625) ∧ 
  (S ≠ C ∧ S ≠ D ∧ C ≠ D)) →
  -- Introduce some conditions to represent the described transition process
  -- Specifically the conditions mentioning the lakes and transitions 
  ∀ (transition_count : ℕ), 
    (transition_count = 4) →
    ∃ (number_of_streams : ℕ), number_of_streams = 3 := 
sorry

end number_of_streams_l561_561746


namespace angle_between_line_and_plane_l561_561754

-- Definitions of the pyramid S-ABCD, points O, P, and the condition SO = OD
variables (S A B C D O P : Type)
variables [HasProjection O S A B C D] [Midpoint P S D]
variables (SO OD : Real)
variables (angle_BC_PAC : Real)

-- Defining the condition that SO = OD
def equal_edges : Prop := SO = OD

-- The main statement to prove
theorem angle_between_line_and_plane (h1 : equal_edges SO OD) :
  angle_BC_PAC = 30 :=
sorry

end angle_between_line_and_plane_l561_561754


namespace max_min_difference_eq_g_l561_561162

variable {t : ℝ} (α β : ℝ) (f : ℝ → ℝ)

noncomputable def f := λ x, (2 * x - t) / (x^2 + 1)
noncomputable def g := λ t, 8 * Real.sqrt (t^2 + 1) * (2 * t^2 + 5) / (16 * t^2 + 25)

theorem max_min_difference_eq_g {α β : ℝ} (h : 4 * α^2 - 4 * t * α - 1 = 0) (h1 : 4 * β^2 - 4 * t * β - 1 = 0) (h2 : α ≠ β) : 
  (f β - f α) = g t := 
sorry

end max_min_difference_eq_g_l561_561162


namespace max_single_player_salary_l561_561054

theorem max_single_player_salary (num_players : ℕ) (min_salary max_total_salary max_single_salary : ℕ) 
  (h1 : num_players = 23)
  (h2 : min_salary = 20000)
  (h3 : max_total_salary = 800000)
  (h4 : max_single_salary = 450000) :
  max_possible_salary = 360000 := 
begin
  sorry
end

end max_single_player_salary_l561_561054


namespace grace_age_is_60_l561_561201

def Grace : ℕ := 60
def motherAge : ℕ := 80
def grandmotherAge : ℕ := 2 * motherAge
def graceAge : ℕ := (3 / 8) * grandmotherAge

theorem grace_age_is_60 : graceAge = Grace := by
  sorry

end grace_age_is_60_l561_561201


namespace count_integers_between_sqrt8_and_sqrt72_l561_561207

theorem count_integers_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℕ, 3 ≤ x ∧ x ≤ 8 → x ∈ finset.range (8 - 3 + 1) :=
by
  let a := real.sqrt 8
  let b := real.sqrt 72
  have ha : 3 = nat_ceil a := sorry
  have hb : 8 = nat_floor b := sorry
  use 6
  split
  {
    exact sorry
  }
  {
    intros x hx
    have h_valid : 3 ≤ x ∧ x ≤ 8 := hx
    rw finset.mem_range
    sorry
  }

end count_integers_between_sqrt8_and_sqrt72_l561_561207


namespace pyramid_height_l561_561050

def height_of_pyramid (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem pyramid_height (n : ℕ) : height_of_pyramid n = 2 * (n - 1) :=
by
  -- The proof would typically go here
  sorry

end pyramid_height_l561_561050


namespace angle_between_lines_AB1_and_BC1_l561_561866

-- Define the structure of a regular triangular prism
structure TriangularPrism (α : Type*) [InnProdSpace α] :=
  (A B C A1 B1 C1 : α)
  (h_side_perp_base : ∀ (u : α), ⟪u, A1 - A⟫ = ⟨u, B - A⟩)
  (h_A1_twice_h : ‖A1 - A‖ = 2 * (√3 / 2 * ‖B - A‖))

-- The goal is to prove the angle between lines AB1 and BC1 is 51°19'
theorem angle_between_lines_AB1_and_BC1
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
  (P : TriangularPrism α)
  : angle P.A P.B1 P.B P.C1 = 51 * (π / 180) + 19 * (π / 180) / 60 :=
begin
  sorry,
end

end angle_between_lines_AB1_and_BC1_l561_561866


namespace gingerbread_price_today_is_5_l561_561013

-- Given conditions
variables {x y a b k m : ℤ}

-- Price constraints
axiom price_constraint_yesterday : 9 * x + 7 * y < 100
axiom price_constraint_today1 : 9 * a + 7 * b > 100
axiom price_constraint_today2 : 2 * a + 11 * b < 100

-- Price change constraints
axiom price_change_gingerbread : a = x + k
axiom price_change_pastries : b = y + m
axiom gingerbread_change_range : |k| ≤ 1
axiom pastries_change_range : |m| ≤ 1

theorem gingerbread_price_today_is_5 : a = 5 :=
by
  sorry

end gingerbread_price_today_is_5_l561_561013


namespace find_principal_amount_l561_561391

noncomputable def principal_amount (difference : ℝ) (rate : ℝ) : ℝ :=
  let ci := rate / 2
  let si := rate
  difference / (ci ^ 2 - 1 - si)

theorem find_principal_amount :
  principal_amount 4.25 0.10 = 1700 :=
by 
  sorry

end find_principal_amount_l561_561391


namespace maximize_OP_OB_l561_561745

noncomputable def A : ℝ × ℝ := (2, 0)
noncomputable def B : ℝ × ℝ := (0, 2)
noncomputable def O : ℝ × ℝ := (0, 0)

def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  real.sqrt ((p₁.1 - p₂.1) ^ 2 + (p₁.2 - p₂.2) ^ 2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def max_OP_OB (P : ℝ × ℝ) (h : distance P A = 1) : ℝ :=
  magnitude ((P.1, P.2 + B.2))

theorem maximize_OP_OB (P : ℝ × ℝ) (h : distance P A = 1) :
  ∃ P, max_OP_OB P h = 2 * real.sqrt 2 + 1 :=
sorry

end maximize_OP_OB_l561_561745


namespace base_2_representation_of_96_l561_561439

theorem base_2_representation_of_96 : nat_to_digit 2 96 = [1, 1, 0, 0, 0, 0, 0] := sorry

end base_2_representation_of_96_l561_561439


namespace gcd_five_pentagonal_and_n_plus_one_l561_561131

-- Definition of the nth pentagonal number
def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n - 1)) / 2

-- Proof statement
theorem gcd_five_pentagonal_and_n_plus_one (n : ℕ) (h : 0 < n) : 
  Nat.gcd (5 * pentagonal_number n) (n + 1) = 1 :=
sorry

end gcd_five_pentagonal_and_n_plus_one_l561_561131


namespace binomial_sum_simplifies_l561_561818

theorem binomial_sum_simplifies (n : ℕ) (p : ℝ) :
  (∑ k in Finset.range n.succ, (k * (Nat.choose n k) * p^k * (1 - p)^(n - k))) = n * p :=
by
  sorry

end binomial_sum_simplifies_l561_561818


namespace line_tangent_to_circle_eq_l561_561715

theorem line_tangent_to_circle_eq (k : ℝ) :
    (∀ x y: ℝ, y = k * x → (x^2 + (y-4)^2 = 4 ∧ 4 / sqrt(1 + k^2) = 2) → k = -sqrt 3) :=
by
  sorry

end line_tangent_to_circle_eq_l561_561715


namespace number_of_integers_between_sqrt8_sqrt72_l561_561244

theorem number_of_integers_between_sqrt8_sqrt72 :
  let sqrt_8 := Real.sqrt 8
  let sqrt_72 := Real.sqrt 72
  let lower_bound := sqrt_8.ceil
  let upper_bound := sqrt_72.floor
  ∃ n : ℕ, lower_bound ≤ n ∧ n ≤ upper_bound → n = 6 := by
sorry

end number_of_integers_between_sqrt8_sqrt72_l561_561244


namespace train_speed_proof_l561_561029

open Real

noncomputable def manSpeed_kmh := 3 -- Speed of the man in km/hr
noncomputable def trainLength_m := 900 -- Length of the train in meters
noncomputable def crossingTime_s := 53.99568034557235 -- Time taken to cross the man in seconds

noncomputable def manSpeed_ms := (manSpeed_kmh * 1000) / 3600 -- Converting man's speed from km/hr to m/s

noncomputable def relativeSpeed_ms := trainLength_m / crossingTime_s -- Calculating relative speed of train with respect to the man

noncomputable def trainSpeed_ms := relativeSpeed_ms + manSpeed_ms -- Calculating train's speed in m/s

noncomputable def trainSpeed_kmh := trainSpeed_ms * 3.6 -- Converting train's speed from m/s to km/hr

theorem train_speed_proof : 
  trainSpeed_kmh ≈ 63.0036 := 
by 
  sorry

end train_speed_proof_l561_561029


namespace noncongruent_triangles_count_l561_561678

/-- Prove that the number of noncongruent integer-sided triangles 
with positive area and perimeter less than 20, 
which are neither equilateral, isosceles, nor right triangles, is 15. -/
theorem noncongruent_triangles_count : 
  ∃ n : ℕ, 
  (∀ (a b c : ℕ) (h : a ≤ b ∧ b ≤ c),
    a + b + c < 20 ∧ a + b > c ∧ a^2 + b^2 ≠ c^2 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → n ≥ 15) :=
sorry

end noncongruent_triangles_count_l561_561678


namespace highest_power_of_3_divides_A_l561_561322

-- Given:
-- A: the number of 2019-digit numbers made of 2 different digits
-- Prove:
-- The highest power of 3 that divides A is 5
theorem highest_power_of_3_divides_A :
  ∃ (A : ℕ), (A = 81 * (2 ^ 2018 - 1)) ∧ (nat.find_greatest (λ (k : ℕ), 3^k ∣ A) = 5) :=
by
  sorry

end highest_power_of_3_divides_A_l561_561322


namespace unknown_rate_of_two_towels_l561_561505

theorem unknown_rate_of_two_towels :
  (∃ x : ℝ,
  let total_cost_of_towels := 300 + 750 + 2 * x in
  let total_number_of_towels := 10 in
  let average_price_of_towels := total_cost_of_towels / total_number_of_towels in
  average_price_of_towels = 160) → x = 275 :=
by
  sorry

end unknown_rate_of_two_towels_l561_561505


namespace total_money_l561_561712

variable (Sally Jolly Molly : ℕ)

-- Conditions
def condition1 (Sally : ℕ) : Prop := Sally - 20 = 80
def condition2 (Jolly : ℕ) : Prop := Jolly + 20 = 70
def condition3 (Molly : ℕ) : Prop := Molly + 30 = 100

-- The theorem to prove
theorem total_money (h1: condition1 Sally)
                    (h2: condition2 Jolly)
                    (h3: condition3 Molly) :
  Sally + Jolly + Molly = 220 :=
by
  sorry

end total_money_l561_561712


namespace hyperbola_focal_length_correct_l561_561186

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Define the focal length
def focal_length : ℝ := 2 * real.sqrt (9 + 4)

-- Theorem statement
theorem hyperbola_focal_length_correct : ∀ x y : ℝ, hyperbola x y → focal_length = 2 * real.sqrt 13 :=
by
  sorry

end hyperbola_focal_length_correct_l561_561186


namespace probability_of_spade_or_king_l561_561903

open Classical

-- Pack of cards containing 52 cards
def total_cards := 52

-- Number of spades in the deck
def num_spades := 13

-- Number of kings in the deck
def num_kings := 4

-- Number of overlap (king of spades)
def num_king_of_spades := 1

-- Total favorable outcomes
def total_favorable_outcomes := num_spades + num_kings - num_king_of_spades

-- Probability of drawing a spade or a king
def probability_spade_or_king := (total_favorable_outcomes : ℚ) / total_cards

theorem probability_of_spade_or_king : probability_spade_or_king = 4 / 13 := by
  sorry

end probability_of_spade_or_king_l561_561903


namespace sum_of_coeffs_l561_561884

-- Define the points A, B, C, and P.
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (4, 7)
def P : ℝ × ℝ := (3, 3)

-- Define the distance formula.
def dist (X Y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((X.1 - Y.1) ^ 2 + (X.2 - Y.2) ^ 2)

-- Calculate the distances from P to A, B, and C.
def AP := dist A P
def BP := dist B P
def CP := dist C P

-- Prove the result that a + c = 4 where a and c are the coefficients of sqrt terms.
theorem sum_of_coeffs : AP + BP + CP = 3 * Real.sqrt 2 + Real.sqrt 34 + Real.sqrt 17 →
  4 :=
sorry

end sum_of_coeffs_l561_561884


namespace required_locus_l561_561828

-- Definitions for the points and conditions
variables {Point : Type*} [metric_space Point]
variables (A O K L : Point) 

def on_bisector_ray_without_endpoints (P : Point) : Prop :=
  (P ≠ A) ∧ (P ≠ O) ∧ is_collinear A O P

def on_segment_without_endpoints (K L P : Point) : Prop := 
  is_between K L P ∧ (P ≠ K) ∧ (P ≠ L)

theorem required_locus (P : Point) :
  (on_bisector_ray_without_endpoints A O P ∨ on_segment_without_endpoints K L P) →
  (P ≠ A) ∧ (P ≠ O) ∧ (P ≠ K) ∧ (P ≠ L) :=
sorry

end required_locus_l561_561828


namespace min_sum_of_primes_l561_561962

open Classical

theorem min_sum_of_primes (k m n p : ℕ) (h1 : 47 + m = k) (h2 : 53 + n = k) (h3 : 71 + p = k)
  (pm : Prime m) (pn : Prime n) (pp : Prime p) :
  m + n + p = 57 ↔ (k = 76 ∧ m = 29 ∧ n = 23 ∧ p = 5) :=
by {
  sorry
}

end min_sum_of_primes_l561_561962


namespace number_of_divisors_2310_l561_561587

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l561_561587


namespace find_angle_B_l561_561305

/-!
# Proof of Angle B in Triangle ABC

Given the sides a, b, and angle A in △ABC, prove that the measure of angle B is 60° or 120°.
-/

theorem find_angle_B 
  (a b : ℝ) (A B : ℝ)
  (ha : a = 2)
  (hb : b = 2 * Real.sqrt 3)
  (hA : A = Real.pi / 6)
  (h1 : sin B = (b * sin A) / a) :
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
by
  sorry

end find_angle_B_l561_561305


namespace number_of_divisors_of_2310_l561_561593

theorem number_of_divisors_of_2310 : 
  let n := 2310 in
  let prime_factors := (2, 1) :: (3, 1) :: (5, 1) :: (7, 1) :: (11, 1) :: [] in
  n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 →
  (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 32 :=
begin
  intro h,
  sorry
end

end number_of_divisors_of_2310_l561_561593


namespace minimum_value_l561_561905

def f (x : ℝ) : ℝ := |x - 4| + |x + 7| + |x - 5|

theorem minimum_value : ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x = 4 :=
by
  -- Sorry is used here to skip the proof
  sorry

end minimum_value_l561_561905


namespace largest_prime_factor_of_abc_l561_561769

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

def satisfies_conditions (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a^2 + b^2 = c^2

def concatenate_digits (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem largest_prime_factor_of_abc (a b c : ℕ) (h1 : satisfies_conditions a b c) :
  let abc := concatenate_digits a b c in
  is_three_digit abc →
  Nat.prime 29 ∧ ∀ p, Nat.prime p ∧ p ∣ abc → p ≤ 29 := sorry

end largest_prime_factor_of_abc_l561_561769


namespace concurrency_of_lines_l561_561321

-- Definitions based on problem conditions
variables {A B C P B_a B_c C_a A_c : Type} 

-- Assume A, B, C, P, B_a, B_c, C_a, and A_c are points in some geometric space
variables [Point A] [Point B] [Point C] [Point P] [Point B_a] [Point B_c] [Point C_a] [Point A_c]

-- Placeholder assumptions for squares constructions based on the given problem
axiom AB_cB_aC_square : square A B_c B_a C
axiom CA_bA_cB_square : square C A_b A_c B
axiom BC_aC_bA_square : square B C_a C_b A
axiom B_cB_c'B_a'B_a_square : square B_c B_c' B_a' B_a

-- Assume P is the center of square B_c B_c' B_a' B_a
axiom P_is_center : center P B_c B_c' B_a' B_a

-- Concurrency goal
theorem concurrency_of_lines : concurrent BP C_aB_a A_cB_c :=
sorry

end concurrency_of_lines_l561_561321


namespace cassidy_grounded_l561_561079

noncomputable def groundDays : ℤ :=
  let initial_grounding : ℤ := 14
  let extra_subjects : ℤ := 4 * 3
  let extra_extracurricular : ℤ := 2 * (3 / 2)
  let volunteering : ℤ := 2
  initial_grounding + extra_subjects + extra_extracurricular - volunteering

theorem cassidy_grounded : groundDays = 27 := by
  have h1 : 4 * 3 = 12 := rfl
  have h2 : 2 * (3 / 2) = 3 := rfl
  have h3 : 14 + 12 + 3 = 29 := rfl
  have h4 : 29 - 2 = 27 := rfl
  rw [←h1, ←h2, ←h3, ←h4]
  sorry

end cassidy_grounded_l561_561079


namespace train_combined_distance_l561_561440

/-- Prove that the combined distance covered by three trains is 3480 km,
    given their respective speeds and travel times. -/
theorem train_combined_distance : 
  let speed_A := 150 -- Speed of Train A in km/h
  let time_A := 8     -- Time Train A travels in hours
  let speed_B := 180 -- Speed of Train B in km/h
  let time_B := 6     -- Time Train B travels in hours
  let speed_C := 120 -- Speed of Train C in km/h
  let time_C := 10    -- Time Train C travels in hours
  let distance_A := speed_A * time_A -- Distance covered by Train A
  let distance_B := speed_B * time_B -- Distance covered by Train B
  let distance_C := speed_C * time_C -- Distance covered by Train C
  let combined_distance := distance_A + distance_B + distance_C -- Combined distance covered by all trains
  combined_distance = 3480 :=
by
  sorry

end train_combined_distance_l561_561440


namespace interest_years_calculation_l561_561948

theorem interest_years_calculation 
  (total_sum : ℝ)
  (second_sum : ℝ)
  (interest_rate_first : ℝ)
  (interest_rate_second : ℝ)
  (time_second : ℝ)
  (interest_second : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : total_sum = 2795)
  (h2 : second_sum = 1720)
  (h3 : interest_rate_first = 3)
  (h4 : interest_rate_second = 5)
  (h5 : time_second = 3)
  (h6 : interest_second = (second_sum * interest_rate_second * time_second) / 100)
  (h7 : interest_second = 258)
  (h8 : x = (total_sum - second_sum))
  (h9 : (interest_rate_first * x * y) / 100 = interest_second)
  : y = 8 := sorry

end interest_years_calculation_l561_561948


namespace flooring_area_already_installed_l561_561433

variable (living_room_length : ℕ) (living_room_width : ℕ) 
variable (flooring_sqft_per_box : ℕ)
variable (remaining_boxes_needed : ℕ)
variable (already_installed : ℕ)

theorem flooring_area_already_installed 
  (h1 : living_room_length = 16)
  (h2 : living_room_width = 20)
  (h3 : flooring_sqft_per_box = 10)
  (h4 : remaining_boxes_needed = 7)
  (h5 : living_room_length * living_room_width = 320)
  (h6 : already_installed = 320 - remaining_boxes_needed * flooring_sqft_per_box) : 
  already_installed = 250 :=
by
  sorry

end flooring_area_already_installed_l561_561433


namespace number_of_divisors_of_2310_l561_561572

theorem number_of_divisors_of_2310 : 
  let n := 2310 in 
  let prime_factors := [2, 3, 5, 7, 11] in
  ∃ k : ℕ, k = prime_factors.length ∧
  (∀ i, i < k → prime_factors.nth i = some 2 ∨ prime_factors.nth i = some 3 ∨ prime_factors.nth i = some 5 ∨ prime_factors.nth i = some 7 ∨ prime_factors.nth i = some 11) →
  (n.factorization.to_nat * 1).0 = 32 :=
begin
  sorry
end

end number_of_divisors_of_2310_l561_561572


namespace fraction_multiplication_division_l561_561516

theorem fraction_multiplication_division :
  ((3 / 4) * (5 / 6)) / (7 / 8) = 5 / 7 :=
by
  sorry

end fraction_multiplication_division_l561_561516


namespace non_foreign_male_part_time_students_l561_561964

theorem non_foreign_male_part_time_students (
    (total_students: ℕ) 
    (female_fraction: ℝ) 
    (full_time_fraction: ℝ)
    (foreign_fraction_male: ℝ)
    (total_students = 3600):
    female_fraction = 2/3
    full_time_fraction = 3/5
    foreign_fraction_male = 1/10 :
    (non_foreign_male_part_time_students: ℕ) :
    non_foreign_male_part_time_students = 432 :=
  by
    let male_students := total_students / 3
    let foreign_male_students := (1 / 10) * male_students
    let non_foreign_male_students := male_students - foreign_male_students
    let part_time_fraction := 2 / 5
    let non_foreign_male_part_time_students := (non_foreign_male_students * part_time_fraction).to_nat
    sorry

end non_foreign_male_part_time_students_l561_561964


namespace bucket_water_total_l561_561918

theorem bucket_water_total (initial_gallons : ℝ) (added_gallons : ℝ) (total_gallons : ℝ) : 
  initial_gallons = 3 ∧ added_gallons = 6.8 → total_gallons = 9.8 :=
by
  { sorry }

end bucket_water_total_l561_561918


namespace polar_to_rectangular_line_equation_point_not_on_line_curve_distance_min_max_l561_561671

theorem polar_to_rectangular (r θ : ℝ) (h : r = 4 ∧ θ = π / 6) :
  let P := (2 * real.sqrt 3, 2) in
  P = (r * real.cos θ, r * real.sin θ) := by
  sorry

theorem line_equation (x t : ℝ) (h : x = (1 / 2) * t) :
  let y := (real.sqrt 3 / 2) * t - 1 in
  y = real.sqrt 3 * x - 1 := by
  sorry

theorem point_not_on_line (x y t : ℝ) 
  (e1 : x = (1 / 2) * t) 
  (e2 : y = (real.sqrt 3 / 2) * t - 1) 
  (pointP : (x, y) = (2 * real.sqrt 3, 2)) :
  ¬ (pointP.2 = real.sqrt 3 * pointP.1 - 1) := by
  sorry

theorem curve_distance_min_max (θ : ℝ) :
  let Qx := real.cos θ in
  let Qy := 2 + real.sin θ in
  let dist := (λ (θ : ℝ), abs (2 * real.cos (θ + π / 6) - 3) / 2) in
  (Inf (set.range dist) = 1 / 2) ∧ (Sup (set.range dist) = 5 / 2) := by
  sorry

end polar_to_rectangular_line_equation_point_not_on_line_curve_distance_min_max_l561_561671


namespace like_terms_sum_l561_561258

theorem like_terms_sum (m n : ℕ) (h₁ : 2 = m) (h₂ : n = 3) : m + n = 5 :=
by
  rw [←h₁, ←h₂]
  exact add_comm 2 3

end like_terms_sum_l561_561258


namespace distinct_books_distribution_identical_books_distribution_l561_561027

-- 1. Prove the number of ways to distribute 5 distinct books to 3 students.
theorem distinct_books_distribution (n m : ℕ) (distinct_books : n = 5) (students : m = 3)
  (each_student_one_book : ∀ b (h₁ : b < n), ∃ s (h₂ : s < m), true) :
    ∃ k, k = nat.factorial n / (nat.factorial (n - m) * nat.factorial m) :=
sorry

-- 2. Prove the number of ways to distribute 5 identical books to 3 students is 1.
theorem identical_books_distribution (n m : ℕ) (identical_books : n = 5) (students : m = 3)
  (each_student_one_book : ∀ b (h₁ : b < n), ∃ s (h₂ : s < m), true) :
    1 = 1 :=
sorry

end distinct_books_distribution_identical_books_distribution_l561_561027


namespace expression_equals_answer_l561_561128

noncomputable def evaluate_expression : ℚ :=
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 +
  (2013^2 * 2014 - 2015) / Nat.factorial 2014

theorem expression_equals_answer :
  evaluate_expression = 
  1 / Nat.factorial 2009 + 
  1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 
  1 / Nat.factorial 2014 :=
by
  sorry

end expression_equals_answer_l561_561128


namespace infinite_integral_solutions_l561_561324

theorem infinite_integral_solutions (a b c d e f : ℤ) (h1 : b^2 - 4*a*c > 0) (h2 : ¬ is_square (b^2 - 4*a*c))
  (h3 : 4*a*c*f + b*d*e - a*e^2 - c*d^2 - f*b^2 ≠ 0)
  (h4 : ∃ x y : ℤ, a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0) :
  ∃∞ x y : ℤ, a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0 := 
sorry

end infinite_integral_solutions_l561_561324


namespace total_milk_is_31_5_l561_561361

def Mitch_consumption := 3 + 2 + 1
def Sister_consumption := 1.5 + 3 + 1.5 + 1
def Mother_consumption := 0.5 + 2.5 + 1
def Father_consumption := 2 + 1 + 3 + 1
def Extra_soy_milk_used := 15 / 2

def total_milk_consumed :=
  Mitch_consumption + (Sister_consumption - 0.5) + Mother_consumption + Father_consumption + Extra_soy_milk_used

theorem total_milk_is_31_5 :
  total_milk_consumed = 31.5 :=
  sorry

end total_milk_is_31_5_l561_561361


namespace largest_is_b_l561_561986

def largest_decimal := 
let a := 8.12356
let b := 8 + 1235 / 9999 -- 8.1235555... repeating 5
let c := 8 + 123356 / 999999 -- 8.12356356356... repeating 356
let d := 8 + 123562356 / 99999999 -- 8.123562356... repeating 2356
let e := 8 + 1235612356 / 99999999999 -- 8.1235612356... repeating 12356
in 
b

theorem largest_is_b :
  ∀ a b c d e, 
    (a = 8.12356 ∧
    b = 8 + 1235 / 9999 ∧
    c = 8 + 123356 / 999999 ∧
    d = 8 + 123562356 / 99999999 ∧ 
    e = 8 + 1235612356 / 99999999999) → 
    max (max (max (max a b) c) d) e = b :=
by 
  intros a b c d e H 
  rcases H with ⟨ha, hb, hc, hd, he⟩ 
  rw [ha, hb, hc, hd, he] 
  sorry

end largest_is_b_l561_561986


namespace find_area_of_polygon_l561_561732

def square_area (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d ∧ d = 5

def midpoint (a : ℝ) (b : ℝ) : ℝ :=
  (a + b) / 2

def polygon_area : ℝ := 37.5

theorem find_area_of_polygon
  (squares_area : ∀ (A B C D E F G H : ℝ),
  square_area A B C D →
  square_area E F G H →
  midpoint B F = midpoint C E ∧ midpoint B F = 2.5)
  (polygon : ℝ) :
  polygon = polygon_area :=
sorry

end find_area_of_polygon_l561_561732


namespace sue_received_votes_l561_561413

theorem sue_received_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  (sue_percentage * total_votes) = 350 := by
  sorry

end sue_received_votes_l561_561413


namespace min_a2_b2_l561_561664

theorem min_a2_b2 (a b : ℝ) 
  (h : |show 10 / a - 10 / b, by ring| / |show 10 * a - 10 * b, by ring| = 1 / 6) : 
  a^2 + b^2 = 12 :=
by sorry

end min_a2_b2_l561_561664


namespace lion_room_is_3_l561_561046

/-!
  A lion is hidden in one of three rooms. A note on the door of room 1 reads "The lion is here".
  A note on the door of room 2 reads "The lion is not here". A note on the door of room 3 reads "2+3=2×3".
  Only one of these notes is true. Prove that the lion is in room 3.
-/

def note1 (lion_room : ℕ) : Prop := lion_room = 1
def note2 (lion_room : ℕ) : Prop := lion_room ≠ 2
def note3 (lion_room : ℕ) : Prop := 2 + 3 = 2 * 3
def lion_is_in_room3 : Prop := ∀ lion_room, (note1 lion_room ∨ note2 lion_room ∨ note3 lion_room) ∧
  (note1 lion_room → note2 lion_room = false) ∧ (note1 lion_room → note3 lion_room = false) ∧
  (note2 lion_room → note1 lion_room = false) ∧ (note2 lion_room → note3 lion_room = false) ∧
  (note3 lion_room → note1 lion_room = false) ∧ (note3 lion_room → note2 lion_room = false) → lion_room = 3

theorem lion_room_is_3 : lion_is_in_room3 := 
  by
  sorry

end lion_room_is_3_l561_561046


namespace largest_circle_area_l561_561419

theorem largest_circle_area (PQ QR PR : ℝ)
  (h_right_triangle: PR^2 = PQ^2 + QR^2)
  (h_circle_areas_sum: π * (PQ/2)^2 + π * (QR/2)^2 + π * (PR/2)^2 = 338 * π) :
  π * (PR/2)^2 = 169 * π :=
by
  sorry

end largest_circle_area_l561_561419


namespace count_remainder_l561_561329

def has_more_ones_than_zeros (n : ℕ) : Prop :=
  let binary_digits := n.binary_digits
  (binary_digits.count (1) > binary_digits.count (0))

def num_integers_with_more_1s_than_0s_up_to_5000 : ℕ :=
  (List.range (5000 + 1)).count (λ n, has_more_ones_than_zeros n)

theorem count_remainder : 
  num_integers_with_more_1s_than_0s_up_to_5000 % 1000 = 733 :=
by
  sorry

end count_remainder_l561_561329


namespace sum_DE_EF_FG_l561_561824

noncomputable def area_of_polygon := 96
noncomputable def AB := 10
noncomputable def BC := 11
noncomputable def HA := 6

theorem sum_DE_EF_FG : ∀ (DE EF FG : ℕ), 
    (∃ DE EF FG, (area_of_polygon + (DE * HA) + (DE * FG // 2) = 110) 
    ∧ (DE * 11 + 3 * DE = area_of_polygon + 14) 
    → DE + EF + FG = 11) 
:= by
  sorry

end sum_DE_EF_FG_l561_561824


namespace solve_fraction_zero_l561_561722

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 16) / (4 - x) = 0) (h2 : 4 - x ≠ 0) : x = -4 :=
sorry

end solve_fraction_zero_l561_561722


namespace ellipse_equation_l561_561393

theorem ellipse_equation (c : ℝ) (h1 : c = sqrt 2)
                        (midpoint_x : ℝ) (h2 : midpoint_x = -2 / 3)
                        (line_eq : ∀ x y, y = x + 1) :
    ∃ a b : ℝ, a ^ 2 = 4 ∧ b ^ 2 = 2 ∧ ∀ x y : ℝ, (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1 := 
begin
    sorry
end

end ellipse_equation_l561_561393


namespace min_value_expression_l561_561567

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ m : ℝ, m = sqrt 39 ∧
  (∀ a b > 0, 
    (|6 * a - 4 * b| + |3 * (a + b * sqrt 3) + 2 * (a * sqrt 3 - b)|) / sqrt (a^2 + b^2) ≥ m) :=
sorry

end min_value_expression_l561_561567


namespace trajectory_of_P_distance_EF_l561_561303

section Exercise

-- Define the curve C in polar coordinates
def curve_C (ρ' θ: ℝ) : Prop :=
  ρ' * Real.cos (θ + Real.pi / 4) = 1

-- Define the relationship between OP and OQ
def product_OP_OQ (ρ ρ' : ℝ) : Prop :=
  ρ * ρ' = Real.sqrt 2

-- Define the trajectory of point P (C1) as the goal
theorem trajectory_of_P (ρ θ: ℝ) (hC: curve_C ρ' θ) (hPQ: product_OP_OQ ρ ρ') :
  ρ = Real.cos θ - Real.sin θ :=
sorry

-- Define the coordinates and the curve C2
def curve_C2 (x y t: ℝ) : Prop :=
  x = 0.5 - Real.sqrt 2 / 2 * t ∧ y = Real.sqrt 2 / 2 * t

-- Define the line l in Cartesian coordinates that needs to be converted to polar
def line_l (x y: ℝ) : Prop :=
  y = -Real.sqrt 3 * x

-- Define the distance |EF| to be proved
theorem distance_EF (θ ρ_1 ρ_2: ℝ) (hx: curve_C2 (0.5 - Real.sqrt 2 / 2 * t) (Real.sqrt 2 / 2 * t) t)
  (hE: θ = 2 * Real.pi / 3 ∨ θ = -Real.pi / 3)
  (hρ1: ρ_1 = Real.cos (-Real.pi / 3) - Real.sin (-Real.pi / 3))
  (hρ2: ρ_2 = 0.5 * (Real.sqrt 3 + 1)) :
  |ρ_1 + ρ_2| = Real.sqrt 3 + 1 :=
sorry

end Exercise

end trajectory_of_P_distance_EF_l561_561303


namespace cunegonde_blocks_l561_561988

def can_jump (a b : ℕ) : Prop := abs (a - b) = 1

def count_arrangements (n : ℕ) : ℕ :=
  2^(n-1)

theorem cunegonde_blocks (n : ℕ) : 
  ∃ s : list ℕ, (∀ k, k ∈ s → 1 ≤ k ∧ k ≤ n) ∧ (∀ k, k < list.length s - 1 → can_jump (s.nth_le k (sorry)) (s.nth_le (k + 1) (sorry))) ∧ list.nodup s ∧ list.length s = n → 
  list.perm {1, 2, ..., n} s ∧ count_arrangements n = 2^(n-1) :=
sorry

end cunegonde_blocks_l561_561988


namespace number_of_integers_between_sqrt10_sqrt200_l561_561677

theorem number_of_integers_between_sqrt10_sqrt200 :
  let a := Real.sqrt 10
      b := Real.sqrt 200 in
  (Nat.ceil a ≤ Nat.floor b) →
  ∃ n : ℕ, n = Nat.floor b - Nat.ceil a + 1 ∧ n = 11 :=
by
  intro a := Real.sqrt 10
  intro b := Real.sqrt 200
  intro h
  existsi Nat.floor b - Nat.ceil a + 1
  split
  {
    sorry
  }
  {
    sorry
  }

end number_of_integers_between_sqrt10_sqrt200_l561_561677


namespace area_of_triangle_CDM_l561_561758

theorem area_of_triangle_CDM :
  ∀ (A B C M D: ℝ) (AC BC: ℝ) (AD BD: ℝ) (right_angle_at_C: ∀ α: ℝ, α = 90),
    AC = 9 →
    BC = 40 →
    AD = 26 →
    BD = 26 →
    ∃ (x y z: ℤ), x = 14415 ∧ y = 113 ∧ z = 328 ∧ (area_of_triangle_CD D M C = (x * real.sqrt y / z)) := 
begin
  assume A B C M D AC BC AD BD right_angle_at_C,
  assume h1: AC = 9,
  assume h2: BC = 40,
  assume h3: AD = 26,
  assume h4: BD = 26,
  sorry
end

end area_of_triangle_CDM_l561_561758


namespace b_50_eq_121_pow_12_25_l561_561990

noncomputable def b : ℕ → ℝ
| 1       := 1
| (n + 1) := (121 * (b n)^4) ^ (1 / 4)

theorem b_50_eq_121_pow_12_25 : b 50 = 121 ^ 12.25 :=
by
  sorry

end b_50_eq_121_pow_12_25_l561_561990


namespace problem_I_problem_II_problem_III_l561_561178

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.log x / Real.log a

theorem problem_I (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (h_f1 : f a 1 = 2) : a = 2 :=
sorry

theorem problem_II (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (h_min_f : ∀ x ∈ set.Icc 1 2, f a x ≥ 5) : a = 5 :=
sorry

theorem problem_III (ha_pos : ∀ a > 0 ∧ a ≠ 1, ∃ x ∈ set.Icc 1 2, f a x ≥ a^2) : true :=
sorry

end problem_I_problem_II_problem_III_l561_561178


namespace domain_of_g_l561_561543

def g (x : ℝ) : ℝ := (x^2 + 2*x + 1) / real.sqrt (x^2 - 5*x + 6)

theorem domain_of_g : {x : ℝ | ∃ y, g x = y} = {x : ℝ | x < 2 ∨ x > 3} :=
by 
  sorry

end domain_of_g_l561_561543


namespace find_range_of_m_l561_561159

noncomputable def f (x : ℝ) : ℝ := (x - 4) / (x - 3)

def P (m : ℝ) : Prop :=
  ∀ x ∈ Set.Ici m, monotone_on f (Set.Ici x)

def Q (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), 4 * Real.sin (2 * x + Real.pi / 4) ≤ m

def problem (m : ℝ) : Prop :=
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → -2 * Real.sqrt 2 ≤ m ∧ m ≤ 3

theorem find_range_of_m (m : ℝ) : problem m := sorry

end find_range_of_m_l561_561159


namespace find_b_value_l561_561504

def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

def triangle_DEF := (0, 0, 0, 3, 10, 0)

def horizontal_line_divides_triangle_equally (b : ℝ) : Prop :=
  let full_area := triangle_area 10 3 in
  let half_area := full_area / 2 in
  let upper_triangle_area := triangle_area 10 (3 - b) in
  upper_triangle_area = half_area

theorem find_b_value : horizontal_line_divides_triangle_equally 1.5 :=
by {
  sorry
}

end find_b_value_l561_561504


namespace number_of_integers_between_sqrt8_sqrt72_l561_561242

theorem number_of_integers_between_sqrt8_sqrt72 :
  let sqrt_8 := Real.sqrt 8
  let sqrt_72 := Real.sqrt 72
  let lower_bound := sqrt_8.ceil
  let upper_bound := sqrt_72.floor
  ∃ n : ℕ, lower_bound ≤ n ∧ n ≤ upper_bound → n = 6 := by
sorry

end number_of_integers_between_sqrt8_sqrt72_l561_561242


namespace perfect_play_winner_l561_561074

theorem perfect_play_winner (A B : ℕ) :
    (A = B → (∃ f : ℕ → ℕ, ∀ n, 0 < f n ∧ f n ≤ B ∧ f n = B - A → false)) ∧
    (A ≠ B → (∃ g : ℕ → ℕ, ∀ n, 0 < g n ∧ g n ≤ B ∧ g n = A - B → false)) :=
sorry

end perfect_play_winner_l561_561074


namespace identity_function_applied_l561_561829

noncomputable def number : ℝ := 0.004

theorem identity_function_applied (x : ℝ) (f : ℝ → ℝ) (h : ∀ y, f y = y) :
  f (69.28 * x) / 0.03 = 9.237333333333334 → x = number :=
by
  intros h1
  skip -- proof goes here

#check identity_function_applied

end identity_function_applied_l561_561829


namespace interval_integer_count_l561_561229

-- Define the problem conditions
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt8
def lower_bound : Int := ceil sqrt8

-- Define the largest integer less than sqrt72
def upper_bound : Int := floor sqrt72

-- Prove the number of integers between sqrt8 and sqrt72 is 6
theorem interval_integer_count : 
    (upper_bound - lower_bound + 1) = 6 := 
by
  -- Steps and proofs would go here, but we use sorry to skip the proof for now
  sorry

end interval_integer_count_l561_561229


namespace bar_and_line_charts_use_unit_length_l561_561076

-- Definitions for bar charts and line charts
def bar_chart_meaning (bc : Type) : Prop :=
  ∀ g : Grid, bc g → (g.unit_length_used_for_quantity = true)

def line_chart_meaning (lc : Type) : Prop :=
  ∀ g : Grid, lc g → (g.unit_length_used_for_quantity = true)

-- Theorem statement
theorem bar_and_line_charts_use_unit_length 
  {bc : Type} {lc : Type} (bc_meaning : bar_chart_meaning bc) 
  (lc_meaning : line_chart_meaning lc) : 
  ∀ (chart : bc ⊕ lc), 
  (chart.unit_length_used_for_quantity = true) :=
begin
  sorry
end

end bar_and_line_charts_use_unit_length_l561_561076


namespace angle_between_p_and_v_is_90_degrees_l561_561348

open Real

noncomputable theory

def p : ℝ × ℝ × ℝ := (2, -3, -4)
def q : ℝ × ℝ × ℝ := (sqrt 3, 5, -2)
def r : ℝ × ℝ × ℝ := (10, -3, 14)

def dot (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def v : ℝ × ℝ × ℝ :=
  let pr := dot p r in
  let pq := dot p q in
  (pr * q.1 - pq * r.1, pr * q.2 - pq * r.2, pr * q.3 - pq * r.3)

def angle_is_90_degrees (a b : ℝ × ℝ × ℝ) : Prop :=
  dot a b = 0

theorem angle_between_p_and_v_is_90_degrees : angle_is_90_degrees p v := 
    by sorry

end angle_between_p_and_v_is_90_degrees_l561_561348


namespace integer_count_between_sqrt8_and_sqrt72_l561_561218

theorem integer_count_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ( ∀ x : ℤ, (⌊Real.sqrt 8⌋.to_nat + 1) ≤ x ∧ x ≤ ⌊Real.sqrt 72⌋ - 1 → x = 6 ) :=
by 
  -- Define the floor and ceiling functions
  have sqrt_8_ceil : ⌈Real.sqrt 8⌉ = 3 := sorry
  have sqrt_72_floor : ⌊Real.sqrt 72⌋ = 8 := sorry

  use 6,
  split,
  { refl },
  { intros x hx,
    sorry
  }

-- Additional supporting facts
lemma sqrt_8_approx : Real.sqrt 8 ≈ 2.83 := sorry
lemma sqrt_72_approx : Real.sqrt 72 ≈ 8.49 := sorry

end integer_count_between_sqrt8_and_sqrt72_l561_561218


namespace kaleb_sold_books_l561_561314

theorem kaleb_sold_books (initial_books sold_books purchased_books final_books : ℕ)
  (H_initial : initial_books = 34)
  (H_purchased : purchased_books = 7)
  (H_final : final_books = 24) :
  sold_books = 17 :=
by
  have H_equation : (initial_books - sold_books) + purchased_books = final_books,
    by sorry
  rw [H_initial, H_purchased, H_final] at H_equation,
  sorry

end kaleb_sold_books_l561_561314


namespace total_value_is_155_l561_561312

def coin_count := 20
def silver_coin_count := 10
def silver_coin_value_total := 30
def gold_coin_count := 5
def regular_coin_value := 1

def silver_coin_value := silver_coin_value_total / 4
def gold_coin_value := 2 * silver_coin_value

def total_silver_value := silver_coin_count * silver_coin_value
def total_gold_value := gold_coin_count * gold_coin_value
def regular_coin_count := coin_count - (silver_coin_count + gold_coin_count)
def total_regular_value := regular_coin_count * regular_coin_value

def total_collection_value := total_silver_value + total_gold_value + total_regular_value

theorem total_value_is_155 : total_collection_value = 155 := 
by
  sorry

end total_value_is_155_l561_561312


namespace lunks_for_apples_l561_561695

noncomputable def lunks_per_kunks := 7 / 4
noncomputable def kunks_per_apples := 3 / 5
def apples_needed := 24
noncomputable def kunks_needed_for_apples := (kunks_per_apples * apples_needed).ceil
noncomputable def lunks_needed := (lunks_per_kunks * kunks_needed_for_apples).ceil

theorem lunks_for_apples :
  lunks_needed = 27 :=
sorry

end lunks_for_apples_l561_561695


namespace solve_for_a_l561_561778

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by 
  sorry

end solve_for_a_l561_561778


namespace odd_and_monotonically_increasing_l561_561895

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f x < f y

theorem odd_and_monotonically_increasing (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3) →
  isOddFunction f ∧ isMonotonicallyIncreasing f ∈ set.Ioi 0 :=
by
  intro h
  sorry

end odd_and_monotonically_increasing_l561_561895


namespace divide_by_fraction_l561_561517

theorem divide_by_fraction (a b : ℝ) (hb : b ≠ 0) : a / (1 / b) = a * b :=
by
  -- This step shows the intention to state the conditions and the problem in Lean format.
  sorry

example : 12 / (1 / 4) = 48 :=
by
  exact divide_by_fraction 12 4 (by norm_num)

end divide_by_fraction_l561_561517


namespace no_real_pair_ab_l561_561716

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 3

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f(x) = x

def has_distinct_critical_points (f : ℝ → ℝ) : Prop := 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (deriv f x₁ = 0 ∧ deriv f x₂ = 0)

theorem no_real_pair_ab (a b : ℝ) (h : has_distinct_critical_points (λ x => f x a b)) :
  ¬ ( ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_fixed_point (λ x => f x a b) x₁ ∧ is_fixed_point (λ x => f x a b) x₂ ) :=
sorry

end no_real_pair_ab_l561_561716


namespace classify_ten_digit_numbers_l561_561081

def is_even (n : ℕ) := n % 2 = 0
def count_twos (n : ℕ) : ℕ := (nat.digits 10 n).count 2
def is_ten_digit_2_1 (n : ℕ) : Prop := nat.digits 10 n ∘ nat.length = 10 ∧ ∀ d ∈ nat.digits 10 n, d = 1 ∨ d = 2

theorem classify_ten_digit_numbers :
  ∀ (n₁ n₂ : ℕ), is_ten_digit_2_1 n₁ → is_ten_digit_2_1 n₂ →
  let class := if is_even (count_twos n₁) then "Class 1" else "Class 2" in
  let class' := if is_even (count_twos n₂) then "Class 1" else "Class 2" in
  class = class' → 
  let sum_digits := nat.digits 10 (n₁ + n₂) in 
  sum_digits.count 3 ≥ 2 :=
begin
  sorry
end

end classify_ten_digit_numbers_l561_561081


namespace range_of_omega_l561_561676

-- Define the problem's conditions
def a (ω x : ℝ) : ℝ × ℝ := (Real.sin (ω / 2 * x), Real.sin (ω * x))
def b (ω x : ℝ) : ℝ × ℝ := (Real.sin (ω / 2 * x), 1 / 2)
def f (ω x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2 - 1 / 2

-- Target theorem statement
theorem range_of_omega (ω : ℝ) (hω : 0 < ω) :
    (∀ x ∈ Ioo π (2 * π), f ω x ≠ 0) ↔ (ω ∈ Ioc 0 (1 / 8) ∪ Icc (1 / 4) (5 / 8)) := 
  sorry

end range_of_omega_l561_561676


namespace fraction_identity_one_fraction_identity_two_l561_561141

variable (α : ℝ)

-- First part of the problem
theorem fraction_identity_one (h1 : 2 * sin α + cos α = 0) : 
  (2 * cos α - sin α) / (sin α + cos α) = 5 :=
sorry

-- Second part of the problem
theorem fraction_identity_two (h1 : 2 * sin α + cos α = 0) :
  sin α / (sin α ^ 3 - cos α ^ 3) = (5 / 3) :=
sorry

end fraction_identity_one_fraction_identity_two_l561_561141


namespace leading_coefficient_is_correct_l561_561519

/-- Define the polynomial -/
def polynomial := 4 * (λ x, x^4 + x^3) - 2 * (λ x, x^4 - 2 * x^3 + 1) + 5 * (λ x, 3 * x^4 - x^2 + 2)

noncomputable def leading_coefficient_of_polynomial : ℕ :=
17

/-- The theorem statement -/
theorem leading_coefficient_is_correct :
  leading_coefficient_of_polynomial = 17 := 
sorry

end leading_coefficient_is_correct_l561_561519


namespace sum_second_and_third_smallest_is_468_l561_561127

-- Define the set of digits to be used
def digits := {1, 3, 5}

-- Function to generate all three-digit numbers using above digits exactly once
def generate_three_digit_numbers (d : Finset ℕ) : Finset ℕ :=
  d.bind (λ h, d.erase h).bind (λ t, d.erase h).erase t).image 
    (λ u, 100 * h + 10 * t + u)

-- Second smallest three-digit number using the digits 1, 3, 5
def second_smallest_three_digit_number := 153

-- Third smallest three-digit number using the digits 1, 3, 5
def third_smallest_three_digit_number := 315

-- The sum of the second smallest and third smallest three-digit number
theorem sum_second_and_third_smallest_is_468 : 
  second_smallest_three_digit_number + third_smallest_three_digit_number = 468 :=
by
sorrry

end sum_second_and_third_smallest_is_468_l561_561127


namespace all_roads_one_way_l561_561417

-- Definitions for the conditions in the problem
variable (N : Type) [Fintype N] (road : N → N → Prop)
variable (connects : ∀ (A B : N), ∃ p : List N, path road A B p)

-- Hypothesis reflecting the given conditions
def initial_conditions (N : Type) [Fintype N] (road : N → N → Prop) : Prop :=
  (∀ (A B : N), A ≠ B → road A B ∨ road B A) ∧
  (∀ (A B : N), road A B → connects A B) ∧
  (∀ (A B : N), connects A B → connects B A)

-- The final goal to prove
theorem all_roads_one_way (N : Type) [Fintype N] (road : N → N → Prop) :
  initial_conditions N road →
  ∃ (one_way : N → N → Prop), 
    (∀ (A B : N), connects A B) →
    (∀ (A B : N), connects A B ↔ (one_way A B ∨ one_way B A)) :=
by
  sorry

end all_roads_one_way_l561_561417


namespace Hawks_wins_20_l561_561850

def wins : List ℕ := [18, 20, 23, 28, 32]

def Hawks_games (Hawks Falcons Raiders Wolves : ℕ) : Prop :=
  Hawks < Falcons ∧
  Wolves > 15 ∧
  (Raiders > Wolves ∧ Raiders < Falcons) ∧
  Falcons ∈ wins ∧ 32 ∈ wins

theorem Hawks_wins_20 : 
  ∃ Hawks Falcons Raiders Wolves, 
    wins.Hawks_games Hawks Falcons Raiders Wolves ∧ Hawks = 20 :=
by
  sorry

end Hawks_wins_20_l561_561850


namespace ladder_distance_from_wall_l561_561014

theorem ladder_distance_from_wall (θ : ℝ) (L : ℝ) (d : ℝ) 
  (h_angle : θ = 60) (h_length : L = 19) (h_cos : Real.cos (θ * Real.pi / 180) = 0.5) : 
  d = 9.5 :=
by
  sorry

end ladder_distance_from_wall_l561_561014


namespace sum_of_inverses_of_squares_l561_561795

theorem sum_of_inverses_of_squares (n : ℕ) (h : 2 ≤ n) : 
  1 + ∑ i in Finset.range (n - 1), 1 / (i + 2) ^ 2 < (2 * n - 1) / n :=
sorry

end sum_of_inverses_of_squares_l561_561795


namespace problem1_problem2_l561_561199

variables {R : Type*} [linear_ordered_field R]

def vector_a : R × R := (4, 3)
def vector_b : R × R := (-1, 2)

-- Problem (1)
theorem problem1 :
  let vector_c := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) in
  real.sqrt (vector_c.1 * vector_c.1 + vector_c.2 * vector_c.2) = real.sqrt 26 :=
sorry

-- Problem (2)
theorem problem2 (λ : R) :
  let vector_d := (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2) 
  let vector_e := (2 * vector_a.1 + vector_b.1, 2 * vector_a.2 + vector_b.2) in
  (vector_d.1 / vector_e.1 = vector_d.2 / vector_e.2) → λ = - (1 / 2) :=
sorry

end problem1_problem2_l561_561199


namespace square_area_l561_561946

noncomputable def side_length_square (s : ℝ) : Prop :=
  ∃ t : ℝ, s^2 = 3 * t ∧ 4 * s = (Real.sqrt 3) / 4 * t^2

theorem square_area (s : ℝ) : side_length_square s → s^2 = 12 * Real.cbrt 4 :=
by sorry

end square_area_l561_561946


namespace number_of_divisors_of_2310_l561_561575

theorem number_of_divisors_of_2310 : 
  let n := 2310 in 
  let prime_factors := [2, 3, 5, 7, 11] in
  ∃ k : ℕ, k = prime_factors.length ∧
  (∀ i, i < k → prime_factors.nth i = some 2 ∨ prime_factors.nth i = some 3 ∨ prime_factors.nth i = some 5 ∨ prime_factors.nth i = some 7 ∨ prime_factors.nth i = some 11) →
  (n.factorization.to_nat * 1).0 = 32 :=
begin
  sorry
end

end number_of_divisors_of_2310_l561_561575


namespace equal_area_triangles_if_and_only_if_parallel_l561_561389

theorem equal_area_triangles_if_and_only_if_parallel (A B C D O : Type)
  (h_diagonals_intersect: AC ∩ BD = {O})
  (h_parallel: BC ∥ AD) :
  (area (triangle A O D) = area (triangle C O D)) ↔
  (area (triangle A O B) = area (triangle C O B)) ∧ (AB ∥ CD) := by
  sorry

end equal_area_triangles_if_and_only_if_parallel_l561_561389


namespace fruit_basket_cost_l561_561480

theorem fruit_basket_cost :
  let bananas_cost   := 4 * 1
  let apples_cost    := 3 * 2
  let strawberries_cost := (24 / 12) * 4
  let avocados_cost  := 2 * 3
  let grapes_cost    := 2 * 2
  bananas_cost + apples_cost + strawberries_cost + avocados_cost + grapes_cost = 28 := 
by
  let groceries_cost := bananas_cost + apples_cost + strawberries_cost + avocados_cost + grapes_cost
  exact sorry

end fruit_basket_cost_l561_561480


namespace number_of_men_in_first_group_l561_561036

-- Conditions as definitions
def wall_length_1 := 112 -- length of the wall built by the first group in metres
def days_1 := 6 -- days taken by the first group
def wall_length_2 := 70 -- length of the wall built by the second group in metres
def days_2 := 3 -- days taken by the second group
def men_2 := 25 -- number of men in the second group

-- Main statement
theorem number_of_men_in_first_group : 
  let men_per_day_2 := (wall_length_2 / men_2 / days_2) in
  let wall_per_man_day_1 := (men_per_day_2 * days_1) in
  (wall_length_1 / wall_per_man_day_1) = 20 :=
sorry

end number_of_men_in_first_group_l561_561036


namespace travel_time_difference_l561_561030

def totalTime (d : ℕ) (speed : ℕ) (stop : ℕ) : ℝ :=
  (d.to_real / speed.to_real) + stop.to_real / 60

theorem travel_time_difference :
  let speed := 20
  let stop := 15
  let trip_1_distance := 100
  let trip_2_distance := 85
  totalTime trip_1_distance speed stop -
  totalTime trip_2_distance speed 0 = 1 :=
by 
  sorry

end travel_time_difference_l561_561030


namespace intersection_M_N_l561_561675

def M : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def N : Set ℝ := {x | -4 < x ∧ x ≤ 2}
def intersection : Set ℝ := {-1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l561_561675


namespace percent_proof_l561_561713

theorem percent_proof (P : ℝ) : (P / 100) * 500 = (50 / 100) * 600 → P = 60 :=
by
  intro h
  have h1 : P * 5 = 50 * 6 :=
    by rw [← mul_div_assoc, ← mul_div_assoc, div_eq_mul_one_div, div_eq_mul_one_div] at h
       exact mul_left_inj' (show 500 ≠ (0:ℝ) by norm_num) h
  have h2 : P * 5 = 300 :=
    by rw [mul_comm 50 6]
  have h3 : P = 60 :=
    by exact eq_of_mul_eq_mul_left (show (5:ℝ) ≠ 0 by norm_num) h2
  exact h3

end percent_proof_l561_561713


namespace consecutive_product_not_mth_power_l561_561977

theorem consecutive_product_not_mth_power (n m k : ℕ) :
  ¬ ∃ k, (n - 1) * n * (n + 1) = k^m := 
sorry

end consecutive_product_not_mth_power_l561_561977


namespace tripod_height_l561_561954

-- Defining the given conditions as constants
constant leg_length : ℝ := 6
constant top_height_initial : ℝ := 5
constant broken_leg_length : ℝ := 4

theorem tripod_height :
  ∃ (m n : ℕ) (n_non_div_sq: ¬∃ p : ℕ, prime p ∧ p^2 ∣ n), 
    let h := m / real.sqrt (n : ℝ) in
      let target := (m + real.sqrt (n : ℝ)) in 
        (⌊ target ⌋ = 11) := 
sorry

end tripod_height_l561_561954


namespace extreme_point_f_f_geq_g_l561_561631

noncomputable def f (x a n : ℝ) : ℝ := a * x - a / x - 51 * n * x
def g (x m : ℝ) : ℝ := x^2 - m * x + 4

theorem extreme_point_f (a n : ℝ) (hx : f 2 a n = 0) : a = 2 :=
sorry

theorem f_geq_g (x1 : ℝ) (hx1 : x1 ∈ Ioo 0 1) (a : ℝ) (ha : a = 2)
  (m : ℝ) (h : ∀ x2, x2 ∈ Icc 1 2 → f x1 a 1 ≥ g x2 m) : m ≥ 8 - 5 * Real.log 2 :=
sorry

end extreme_point_f_f_geq_g_l561_561631


namespace power_sums_l561_561142

-- Definitions as per the given conditions
variables (m n a b : ℕ)
variables (hm : 0 < m) (hn : 0 < n)
variables (ha : 2^m = a) (hb : 2^n = b)

-- The theorem statement
theorem power_sums (hmn : 0 < m + n) : 2^(m + n) = a * b :=
by
  sorry

end power_sums_l561_561142


namespace count_valid_b1_l561_561779

def sequence_next (b : ℕ) : ℕ :=
  if b % 3 = 0 then b / 3 else 2 * b + 1

def condition_met (b1 b2 b3 b4 : ℕ) : Prop :=
  b1 < b2 ∧ b1 < b3 ∧ b1 < b4

def valid_b1 (b1 : ℕ) : Prop :=
  let b2 := sequence_next b1 in
  let b3 := sequence_next b2 in
  let b4 := sequence_next b3 in
  condition_met b1 b2 b3 b4

theorem count_valid_b1 : (Finset.filter (λ b1, valid_b1 b1) (Finset.range 2501)).card = 833 :=
  sorry

end count_valid_b1_l561_561779


namespace solve_fractional_equation_l561_561421

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) :
  (2 / x = 3 / (x + 1)) → (x = 2) :=
by
  -- Proof will be filled in here
  sorry

end solve_fractional_equation_l561_561421


namespace range_of_floor_f_l561_561180

def f (x : ℝ) : ℝ := (3^x / (3^x + 1)) - (1/3)

theorem range_of_floor_f : (set.range (λ x : ℝ, int.floor (f x))) = ({-1, 0} : set ℤ) :=
  sorry

end range_of_floor_f_l561_561180


namespace length_of_AB_l561_561284

theorem length_of_AB (AM BN : ℝ) (h1 : AM = 20) (h2 : BN = 15)
  (h3 : ∀ G, AG : GM = BG : GN = 2 : 1) (h4 : ∀ A B G, AG ⊥ BG):
  let AB := sqrt ((2/3 * AM)^2 + (2/3 * BN)^2)
  in AB = 50/3 :=
by
  sorry

end length_of_AB_l561_561284


namespace TVs_auction_site_l561_561970

variable (TV_in_person : Nat)
variable (TV_online_multiple : Nat)
variable (total_TVs : Nat)

theorem TVs_auction_site :
  ∀ (TV_in_person : Nat) (TV_online_multiple : Nat) (total_TVs : Nat), 
  TV_in_person = 8 → TV_online_multiple = 3 → total_TVs = 42 →
  (total_TVs - (TV_in_person + TV_online_multiple * TV_in_person) = 10) :=
by
  intros TV_in_person TV_online_multiple total_TVs h1 h2 h3
  rw [h1, h2, h3]
  sorry

end TVs_auction_site_l561_561970


namespace findB_coords_l561_561297

namespace ProofProblem

-- Define point A with its coordinates.
def A : ℝ × ℝ := (-3, 2)

-- Define a property that checks if a line segment AB is parallel to the x-axis.
def isParallelToXAxis (A B : (ℝ × ℝ)) : Prop :=
  A.2 = B.2

-- Define a property that checks if the length of line segment AB is 4.
def hasLengthFour (A B : (ℝ × ℝ)) : Prop :=
  abs (A.1 - B.1) = 4

-- The proof problem statement.
theorem findB_coords :
  ∃ B : ℝ × ℝ, isParallelToXAxis A B ∧ hasLengthFour A B ∧ (B = (-7, 2) ∨ B = (1, 2)) :=
  sorry

end ProofProblem

end findB_coords_l561_561297


namespace smallest_positive_period_of_f_minimum_value_of_f_in_interval_l561_561200

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sin (2 * x))
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def f (x m : ℝ) : ℝ := (vec_a x).1 * vec_b.1 + (vec_a x).2 * vec_b.2 + m

theorem smallest_positive_period_of_f :
  ∀ (x : ℝ) (m : ℝ), ∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) m = f x m) → p = Real.pi := 
sorry

theorem minimum_value_of_f_in_interval :
  ∀ (x m : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → ∃ m : ℝ, (∀ x : ℝ, f x m ≥ 5) ∧ m = 5 + Real.sqrt 3 :=
sorry

end smallest_positive_period_of_f_minimum_value_of_f_in_interval_l561_561200


namespace find_value_of_c_l561_561564

theorem find_value_of_c (c : ℝ) (h1 : c > 0) (h2 : c + ⌊c⌋ = 23.2) : c = 11.7 :=
sorry

end find_value_of_c_l561_561564


namespace order_abc_l561_561165

noncomputable def a : ℝ := Real.logBase 0.6 0.7
noncomputable def b : ℝ := Real.log 0.7
noncomputable def c : ℝ := 3 ^ 0.7

theorem order_abc : b < a ∧ a < c := 
by
  sorry

end order_abc_l561_561165


namespace round_robin_chess_l561_561290

/-- 
In a round-robin chess tournament, two boys and several girls participated. 
The boys together scored 8 points, while all the girls scored an equal number of points.
We are to prove that the number of girls could have participated in the tournament is 7 or 14,
given that a win is 1 point, a draw is 0.5 points, and a loss is 0 points.
-/
theorem round_robin_chess (n : ℕ) (x : ℚ) (h : 2 * n * x + 16 = n ^ 2 + 3 * n + 2) : n = 7 ∨ n = 14 :=
sorry

end round_robin_chess_l561_561290


namespace number_of_divisors_2310_l561_561581

-- Define the number whose divisors are being counted
def n : ℕ := 2310

-- Define the prime factorization of the number
def factorization : n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num

-- Define the formula for the number of divisors
def num_divisors (n : ℕ) : ℕ :=
  let e_1 := 1
  let e_2 := 1
  let e_3 := 1
  let e_4 := 1
  let e_5 := 1
  in (e_1 + 1) * (e_2 + 1) * (e_3 + 1) * (e_4 + 1) * (e_5 + 1)

-- State the problem in a theorem
theorem number_of_divisors_2310 : num_divisors n = 32 :=
by
  rw [num_divisors, factorization]
  sorry

end number_of_divisors_2310_l561_561581


namespace two_O1O2_eq_AB_l561_561292

-- Given elements and their properties
variables {A B C D E F O1 O2 : Point}

-- Definitions of the points and their properties
def in_triangle (A B C : Point) : Prop := ∃ (D : Point), is_right_triangle A B C
def foot_of_altitude (C : Point) (AB : Line) : Point := D
def reflection (D : Point) (AC BC : Line) : (Point × Point) := (E, F)
def circumcenter (tri : Triangle) : Point := O1 -- For triangle ECB, for instance

-- Conditions given in the problem
axiom angle_ACB_is_90 (A B C : Point) : angle A C B = 90
axiom foot_D (D AC BC : Point) : is_foot_of_altitude D C AB
axiom reflections_D_EF (D E F AC BC : Point) : is_reflection D AC = E ∧ is_reflection D BC = F
axiom circumcenters_O1_O2 (O1 O2 ECB FCA : Triangle) : circumcenter ECB = O1 ∧ circumcenter FCA = O2

-- The theorem to be proved
theorem two_O1O2_eq_AB : 2 * (distance O1 O2) = distance A B := sorry

end two_O1O2_eq_AB_l561_561292


namespace arithmetic_sequence_term_2018_l561_561171

theorem arithmetic_sequence_term_2018 
  (a : ℕ → ℤ)  -- sequence a_n with ℕ index and ℤ values
  (S9 : Σ (n : ℕ), 9 + 9 * (1 / 2) * (fin n) = 27)  -- sum of the first 9 terms is 27
  (a10 : Σ (n : ℕ), 10 = fin n)  -- rich sum value 

: a 2018 = 2016 := by
  sorry

end arithmetic_sequence_term_2018_l561_561171


namespace number_of_divisors_of_2310_l561_561592

theorem number_of_divisors_of_2310 : 
  let n := 2310 in
  let prime_factors := (2, 1) :: (3, 1) :: (5, 1) :: (7, 1) :: (11, 1) :: [] in
  n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 →
  (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 32 :=
begin
  intro h,
  sorry
end

end number_of_divisors_of_2310_l561_561592


namespace attendance_correct_l561_561424

def total_seats : ℕ := 60000
def percentage_sold : ℚ := 0.75
def fans_stayed_home : ℕ := 5000

theorem attendance_correct :
  let seats_sold := (percentage_sold * total_seats : ℚ).toNat in
  let fans_attended := seats_sold - fans_stayed_home in
  fans_attended = 40000 :=
by
  sorry

end attendance_correct_l561_561424


namespace D_72_l561_561774

-- Define the function D(n) as described in the problem conditions
def D (n : ℕ) : ℕ :=
  (list.prod (multiset.powerset (multiset.filter (λ x, (1 < x)) (multiset.pmap (λ a (k : a > 1), nat.find (nat.dvd_iff.is_div_iff _).mpr (nat.find_spec _).1) (range n).to_multiset (λ x, _)))).map (λ x, list.prod x))

-- The main theorem definition for the given problem
theorem D_72 : D 72 = 26 := 
by {
  -- your proof here
  sorry
}

end D_72_l561_561774


namespace inequality_problem_l561_561345

theorem inequality_problem
  (n : ℕ) (h1 : n ≥ 1)
  (x : Fin (n + 1) → ℝ) (h2 : ∀ i j : Fin (n + 1), i < j → x i > x j) :
  x 0 + ∑ i in Finset.range n, (1 / (x i - x (i + 1))) ≥ x n + 2 * n :=
by
  sorry

end inequality_problem_l561_561345


namespace ellipse_circle_intersection_range_y_diff_l561_561175

theorem ellipse_circle_intersection (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
(maj_axis_length : 2 * a = 4) (on_ellipse : a * a = 2 ∧ b * b = 3) (x y : ℝ)
(h : x^2 / 4 + y^2 / 3 = 1) : x^2 + (y - 2)^2 = 1 :=
    sorry

theorem range_y_diff (x1 y1 x2 y2 m : ℝ) (h : x1 - m * y1 + 1 = 0 ∧ x2 - m * y2 + 1 = 0)
(h1 : x1^2 / 4 + y1^2 / 3 = 1 ∧ x2^2 / 4 + y2^2 / 3 = 1)
(focus_distance_cond : 2 / sqrt (1 + m^2) > sqrt 2) :
∃ l u, (|y1 - y2| = 24 / 13 ∧ |y1 - y2| ≤ 3) :=
    sorry

end ellipse_circle_intersection_range_y_diff_l561_561175


namespace find_tangent_line_l561_561659

theorem find_tangent_line {x y : ℝ} :
  (∀ l₁ : ℝ → ℝ → Prop, 
     (l₁ = λ x y, 3 * x + 4 * y + b = 0) ∧
     (∃ c : ℝ × ℝ, c = (0, -1) ∧ circle c 1 (x, y)) ∧
     (tangent l₁ (circle c 1) c) ∧
     (parallel l₁ (3 * x + 4 * y - 6 = 0)))
  → (3 * x + 4 * y - 1 = 0) ∨ (3 * x + 4 * y + 9 = 0) :=
begin
  sorry -- Proof goes here
end

end find_tangent_line_l561_561659


namespace triangle_area_triangle_side_length_l561_561759

variable {A B C D: Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [LinearOrder B] [LinearOrder C] [LinearOrder D]

theorem triangle_area
  (AB : ℝ) (AC : ℝ) (sinB : ℝ) (cosB : ℝ) : 
  AB = 6 → AC = 4 * Real.sqrt 2 → sinB = 2 * Real.sqrt 2 / 3 → cosB = 1 / 3 → 
  let S := 1 / 2 * AB * cosB * sinB 
  in S = 4 * Real.sqrt 2 := 
sorry

theorem triangle_side_length
  (AB : ℝ) (AC : ℝ) (sinB : ℝ) (cosB : ℝ) (BD : ℝ) (DC : ℝ) (AD : ℝ): 
  AB = 6 → AC = 4 * Real.sqrt 2 → sinB = 2 * Real.sqrt 2 / 3 → cosB = 1 / 3 → 
  BD = 2 * DC → AD = 3 * Real.sqrt 2 → 
  let BC := Real.sqrt 69
  in BC = Real.sqrt 69 := 
sorry

end triangle_area_triangle_side_length_l561_561759


namespace transaction_result_l561_561482

theorem transaction_result
  (house_selling_price store_selling_price : ℝ)
  (house_loss_perc : ℝ)
  (store_gain_perc : ℝ)
  (house_selling_price_eq : house_selling_price = 15000)
  (store_selling_price_eq : store_selling_price = 15000)
  (house_loss_perc_eq : house_loss_perc = 0.1)
  (store_gain_perc_eq : store_gain_perc = 0.3) :
  (store_selling_price + house_selling_price - ((house_selling_price / (1 - house_loss_perc)) + (store_selling_price / (1 + store_gain_perc)))) = 1795 :=
by
  sorry

end transaction_result_l561_561482


namespace binom_fraction_l561_561981

noncomputable def binom (a : ℝ) (b : ℕ) : ℝ :=
  (finset.range b).prod (λ k, a - k) / nat.factorial b

theorem binom_fraction : 
  (binom (1/3) 2016) * (3^2016) / (binom 4032 2016) = -(1 / 4031) := by
  sorry

end binom_fraction_l561_561981


namespace gold_coins_percentage_l561_561070

theorem gold_coins_percentage (A B : ℝ) (h1 : A = 0.35) (h2 : B = 0.3) : 
  let C := 1 - A in
  let D := 1 - B in
  let P := C * D in
  P = 0.455 :=
by
  have C_def : C = 0.65 := by simp [C, h1]
  have D_def : D = 0.7 := by simp [D, h2]
  have P_def : P = C * D := rfl
  have expected_P : 0.65 * 0.7 = 0.455 := by norm_num
  rw [C_def, D_def, P_def]
  exact expected_P

end gold_coins_percentage_l561_561070


namespace robert_balls_l561_561813

theorem robert_balls (R T : ℕ) (hR : R = 25) (hT : T = 40 / 2) : R + T = 45 :=
by
  sorry

end robert_balls_l561_561813


namespace aunt_gemma_dog_food_l561_561966

theorem aunt_gemma_dog_food :
  ∀ (dogs : ℕ) (grams_per_meal : ℕ) (meals_per_day : ℕ) (sack_kg : ℕ) (days : ℕ), 
    dogs = 4 →
    grams_per_meal = 250 →
    meals_per_day = 2 →
    sack_kg = 50 →
    days = 50 →
    (dogs * meals_per_day * grams_per_meal * days) / (1000 * sack_kg) = 2 :=
by
  intros dogs grams_per_meal meals_per_day sack_kg days
  intros h_dogs h_grams_per_meal h_meals_per_day h_sack_kg h_days
  sorry

end aunt_gemma_dog_food_l561_561966


namespace number_of_divisors_of_2310_l561_561573

theorem number_of_divisors_of_2310 : 
  let n := 2310 in 
  let prime_factors := [2, 3, 5, 7, 11] in
  ∃ k : ℕ, k = prime_factors.length ∧
  (∀ i, i < k → prime_factors.nth i = some 2 ∨ prime_factors.nth i = some 3 ∨ prime_factors.nth i = some 5 ∨ prime_factors.nth i = some 7 ∨ prime_factors.nth i = some 11) →
  (n.factorization.to_nat * 1).0 = 32 :=
begin
  sorry
end

end number_of_divisors_of_2310_l561_561573


namespace fantasticbobob_init_seat_count_l561_561452

noncomputable def num_possible_initial_seats (n : ℕ) : ℕ :=
  if n = 1 then 1 else 
    let rec aux (k : ℕ) : ℕ :=
      if k = 1 then 1
      else if k % 2 = 1 then 
        aux (k - 2) + 4 * ((k - 1) / 2 + 1)
      else
        aux (k - 1)
    in aux n

theorem fantasticbobob_init_seat_count : num_possible_initial_seats 29 = 421 := 
by
  sorry

end fantasticbobob_init_seat_count_l561_561452


namespace train_speed_and_length_l561_561601

theorem train_speed_and_length (V l : ℝ) 
  (h1 : 7 * V = l) 
  (h2 : 25 * V = 378 + l) : 
  V = 21 ∧ l = 147 :=
by
  sorry

end train_speed_and_length_l561_561601


namespace binary_string_probability_consecutive_zeros_l561_561031

def is_binary_string (s : String) : Prop :=
  ∀ (c : Char), c ∈ s → (c = '0' ∨ c = '1')

def has_consecutive_zeros (s : String) : Prop :=
  ∃ (i : ℕ), i < s.length - 1 ∧ s.get i = '0' ∧ s.get (i + 1) = '0'

theorem binary_string_probability_consecutive_zeros :
  ∃ p : ℚ, p = 55 / 64 ∧ 
  ∀ s: String, is_binary_string s ∧ s.length = 10 → 
  (has_consecutive_zeros s ↔ p = 55 / 64) :=
by
  sorry

end binary_string_probability_consecutive_zeros_l561_561031


namespace object_casts_elliptical_shadow_l561_561959

-- Definitions of shapes
inductive Shape
| ellipse : Shape
| circle : Shape

-- Object's orientation relative to horizontal plane
inductive Orientation
| parallel : Orientation
| not_parallel : Orientation

-- Function that determines object shape based on orientation
def object_shape (o : Orientation) : Shape :=
match o with
| Orientation.parallel      => Shape.ellipse
| Orientation.not_parallel  => Shape.circle

-- Theorem: The shape of the object is either an ellipse or a circle
theorem object_casts_elliptical_shadow (o : Orientation) : 
  object_shape(o) = Shape.ellipse ∨ object_shape(o) = Shape.circle :=
by
  cases o
  case parallel =>
    left
    rfl
  case not_parallel =>
    right
    rfl

end object_casts_elliptical_shadow_l561_561959


namespace min_colors_for_G_l561_561083

open Nat

-- Define the graph as a structure with vertices and edge condition
structure Graph :=
  (V : Finset ℕ)
  (E : ℕ → ℕ → Prop)
  (edge_cond : ∀ i j ∈ V, E i j ↔ (i ∣ j .))

-- Define a specific graph G with vertices 1 to 1000 and edge condition
def G : Graph :=
  { V := (Finset.range 1000).map Nat.succ,
    E := λ i j, (i ∣ j),
    edge_cond := λ i j, Iff.rfl }

-- Define the proposition for the minimum number of colors required
def min_colors_needed (G : Graph) (n : ℕ) : Prop :=
  ∀ f : ∀ v ∈ G.V, ℕ, (∀ v1 v2 ∈ G.V, G.E v1 v2 → f v1 ≠ f v2) → ∃ (m ≤ n),
    (∀ v ∈ G.V, f v < m)

-- State the proof as a theorem statement
theorem min_colors_for_G : min_colors_needed G 10 :=
sorry

end min_colors_for_G_l561_561083


namespace households_with_bike_only_l561_561731

theorem households_with_bike_only (total neither both car : ℕ) (h1 : total = 90) 
    (h2 : neither = 11) (h3 : both = 16) (h4 : car = 44) : 
    (total - neither - car + both - both) = 35 :=
by
    rw [h1, h2, h3, h4]
    calc
      90 - 11 - 44 + 16 - 16 = 79 - 44 + 16 - 16 : by norm_num
                      ... = 35 : by norm_num
    assumption

end households_with_bike_only_l561_561731


namespace roi_is_25_percent_l561_561041

def dividend_rate : ℝ := 0.125
def face_value : ℝ := 50
def purchase_price : ℝ := 25

def dividend_per_share : ℝ := dividend_rate * face_value
def roi_percentage : ℝ := (dividend_per_share / purchase_price) * 100

theorem roi_is_25_percent :
  roi_percentage = 25 :=
by
  unfold dividend_per_share
  unfold roi_percentage
  have h1 : dividend_per_share = 6.25 := by
    simp [dividend_per_share, dividend_rate, face_value]
  rw [h1]
  have h2 : roi_percentage = (6.25 / purchase_price) * 100 := by
    simp [roi_percentage, dividend_per_share]
  rw [h2]
  have h3 : (6.25 / 25) = 0.25 := by
    norm_num
  rw [h3]
  norm_num
  sorry

end roi_is_25_percent_l561_561041


namespace simple_interest_borrowed_rate_l561_561053

theorem simple_interest_borrowed_rate
  (P_borrowed P_lent : ℝ)
  (n_years : ℕ)
  (gain_per_year : ℝ)
  (simple_interest_lent_rate : ℝ)
  (SI_lending : ℝ := P_lent * simple_interest_lent_rate * n_years / 100)
  (total_gain : ℝ := gain_per_year * n_years) :
  SI_lending = 1000 →
  total_gain = 100 →
  ∀ (SI_borrowing : ℝ), SI_borrowing = SI_lending - total_gain →
  ∀ (R_borrowed : ℝ), SI_borrowing = P_borrowed * R_borrowed * n_years / 100 →
  R_borrowed = 9 := 
by
  sorry

end simple_interest_borrowed_rate_l561_561053


namespace expression_value_l561_561134

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem expression_value :
  ∀ y, y = 8.4 → floor 6.5 * floor (2 / 3) + floor 2 * 7.2 + floor y - 6.0 = 16.4 :=
by
  intro y hy
  rw [hy]
  norm_num
  sorry

end expression_value_l561_561134


namespace largest_prime_factor_4752_l561_561001

theorem largest_prime_factor_4752 : ∃ p : ℕ, p = 11 ∧ prime p ∧ (∀ q : ℕ, prime q ∧ q ∣ 4752 → q ≤ 11) :=
by
  sorry

end largest_prime_factor_4752_l561_561001


namespace ratio_of_erasers_l561_561960

theorem ratio_of_erasers (a n : ℕ) (ha : a = 4) (hn : n = a + 12) :
  n / a = 4 :=
by
  sorry

end ratio_of_erasers_l561_561960


namespace couch_price_is_300_l561_561765

variables (C : ℝ)
noncomputable def table_price := 3 * C
noncomputable def couch_price := 5 * table_price
noncomputable def total_cost := C + table_price + couch_price

theorem couch_price_is_300 (h : total_cost = 380) : couch_price = 300 :=
by sorry

end couch_price_is_300_l561_561765


namespace distance_from_origin_l561_561430

theorem distance_from_origin :
  ∀ {O P : ℝ × ℝ × ℝ} (d1 d2 d3 : ℝ),
  O = (0, 0, 0) → -- Origin point O
  P = (d1, d2, d3) → -- Point P with given distances
  d1 = 3 → d2 = 4 → d3 = 5 → -- Distances to the planes
  ∥⟨d1, d2, d3⟩ - ⟨0, 0, 0⟩∥ = 5 * real.sqrt 2 := -- Prove the required distance
by
  intros O P d1 d2 d3 hO hP hd1 hd2 hd3
  rw [hO, hP, hd1, hd2, hd3]
  norm_num
  sorry

end distance_from_origin_l561_561430


namespace max_area_rectangle_l561_561855

theorem max_area_rectangle (P : ℕ) (hP : P = 40) : ∃ A : ℕ, A = 100 ∧ ∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A := by
  sorry

end max_area_rectangle_l561_561855


namespace mode_of_scores_is_102_l561_561472

-- Define the scores as given in the stem-and-leaf plot
def scores : List ℕ :=
  [60, 60, 60, 72, 75, 81, 81, 90, 93, 97, 98, 97, 98, 102, 102, 102, 102, 106, 106, 106, 111, 113, 115]

-- Define what it means to be the mode of a list (appearing most frequently)
def mode (l : List ℕ) : ℕ :=
  l.maxBy (λ i, l.count i) |>.getOrElse 0

-- Prove that the mode of the scores is 102
theorem mode_of_scores_is_102 : mode scores = 102 :=
by
  sorry

end mode_of_scores_is_102_l561_561472


namespace equilateral_triangle_l561_561756

variables {V : Type} [inner_product_space ℝ V] 
variables {A B C P : V} 
variables {a b c : ℝ} 

-- Condition 1: P is the midpoint of side BC
def midpoint (B C P : V) : Prop := 2 • P = B + C

-- Condition 2: Given lengths of sides opposite angles A, B, C
variable (a b c : ℝ)

-- Condition 3: Given vector equation
def vector_equation (A B C P : V) (a b c : ℝ) : Prop :=
  c • (C - A) + a • (A - P) + b • (P - B) = 0

-- Prove the shape of triangle ABC

theorem equilateral_triangle (h_midpoint : midpoint B C P)
                             (h_vector_eq : vector_equation A B C P a b c) : 
  a = b ∧ b = c :=
sorry

end equilateral_triangle_l561_561756


namespace dexter_filled_fewer_boxes_with_football_cards_l561_561099

-- Conditions
def boxes_with_basketball_cards : ℕ := 9
def cards_per_basketball_box : ℕ := 15
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255

-- Definition of the main problem statement
def fewer_boxes_with_football_cards : Prop :=
  let basketball_cards := boxes_with_basketball_cards * cards_per_basketball_box
  let football_cards := total_cards - basketball_cards
  let boxes_with_football_cards := football_cards / cards_per_football_box
  boxes_with_basketball_cards - boxes_with_football_cards = 3

theorem dexter_filled_fewer_boxes_with_football_cards : fewer_boxes_with_football_cards :=
by
  sorry

end dexter_filled_fewer_boxes_with_football_cards_l561_561099


namespace bird_families_flew_to_Asia_l561_561896

-- Variables/Parameters
variable (A : ℕ) (X : ℕ)
axiom hA : A = 47
axiom hX : X = A + 47

-- Theorem Statement
theorem bird_families_flew_to_Asia : X = 94 :=
by
  sorry

end bird_families_flew_to_Asia_l561_561896


namespace fraction_zero_x_eq_2_l561_561282

theorem fraction_zero_x_eq_2 (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 :=
by sorry

end fraction_zero_x_eq_2_l561_561282


namespace number_of_integers_between_sqrt8_and_sqrt72_l561_561213

theorem number_of_integers_between_sqrt8_and_sqrt72 : 
  let a := Int.ceil (Real.sqrt 8)
  let b := Int.floor (Real.sqrt 72)
  b - a + 1 = 6 :=
begin
  sorry
end

end number_of_integers_between_sqrt8_and_sqrt72_l561_561213


namespace intersection_M_N_l561_561193

open Set

def M : Set ℝ := { x | (1/2)^x ≤ 1 }
def N : Set ℝ := { x | Real.log10 (2 - x) < 0 }

theorem intersection_M_N : M ∩ N = { x | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_M_N_l561_561193


namespace joan_gave_melanie_apples_l561_561764

theorem joan_gave_melanie_apples (original_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : original_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  sorry

end joan_gave_melanie_apples_l561_561764


namespace complex_addition_result_l561_561560

-- Define the complex numbers and the multiplication factor
def complex_num1 : ℂ := -2 + 5 * complex.I
def complex_num2 : ℂ := 3 * complex.I
def factor : ℝ := 3

-- Define the result of the operations
def result1 : ℂ := factor * complex_num1
def result2 : ℂ := result1 + complex_num2

theorem complex_addition_result : 
  result2 = -6 + 18 * complex.I := 
by 
  -- Proof omitted
  sorry

end complex_addition_result_l561_561560


namespace floor_sqrt_23_squared_l561_561558

theorem floor_sqrt_23_squared : (⌊Real.sqrt 23⌋) ^ 2 = 16 := by
  have h1 : (4:ℝ) < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < (5:ℝ) := sorry
  have h3 : (⌊Real.sqrt 23⌋ : ℝ) = 4 :=
    by sorry
  show 4^2 = 16 from by sorry

end floor_sqrt_23_squared_l561_561558


namespace largest_interior_angle_of_triangle_l561_561840

theorem largest_interior_angle_of_triangle (a b c ext : ℝ)
    (h1 : a + b + c = 180)
    (h2 : a / 4 = b / 5)
    (h3 : a / 4 = c / 6)
    (h4 : c + 120 = a + 180) : c = 72 :=
by
  sorry

end largest_interior_angle_of_triangle_l561_561840


namespace limit_series_product_eq_l561_561087

variable (a r s : ℝ)

noncomputable def series_product_sum_limit : ℝ :=
∑' n : ℕ, (a * r^n) * (a * s^n)

theorem limit_series_product_eq :
  |r| < 1 → |s| < 1 → series_product_sum_limit a r s = a^2 / (1 - r * s) :=
by
  intro hr hs
  sorry

end limit_series_product_eq_l561_561087


namespace point_on_line_l561_561548

theorem point_on_line (k : ℝ) (x y : ℝ) (h : x = -1/3 ∧ y = 4) (line_eq : 1 + 3 * k * x = -4 * y) : k = 17 :=
by
  rcases h with ⟨hx, hy⟩
  sorry

end point_on_line_l561_561548


namespace count_integers_between_sqrts_l561_561237

theorem count_integers_between_sqrts : 
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  (upper_bound - lower_bound + 1) = 6 :=
by
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  calc (upper_bound - lower_bound + 1) = 6 : sorry

end count_integers_between_sqrts_l561_561237


namespace table_tennis_scenarios_l561_561882

-- Defining the conditions and the main proof statement
theorem table_tennis_scenarios (minimum_games maximum_games : ℕ) (no_ties : Prop) (first_to_3_games : Prop) :
  minimum_games = 3 → maximum_games = 5 → no_ties →
  (first_to_3_games →
  (∃ scenarios : ℕ, scenarios = 20)) :=
by
  intro h_min h_max h_no_ties h_first_to_3
  use 20
  sorry

end table_tennis_scenarios_l561_561882


namespace book_cost_l561_561624

theorem book_cost (p : ℝ) (h1 : 14 * p < 25) (h2 : 16 * p > 28) : 1.75 < p ∧ p < 1.7857 :=
by
  -- This is where the proof would go
  sorry

end book_cost_l561_561624


namespace number_of_integers_between_sqrt8_and_sqrt72_l561_561216

theorem number_of_integers_between_sqrt8_and_sqrt72 : 
  let a := Int.ceil (Real.sqrt 8)
  let b := Int.floor (Real.sqrt 72)
  b - a + 1 = 6 :=
begin
  sorry
end

end number_of_integers_between_sqrt8_and_sqrt72_l561_561216


namespace fruit_basket_cost_is_28_l561_561478

def basket_total_cost : ℕ := 4 * 1 + 3 * 2 + (24 / 12) * 4 + 2 * 3 + 2 * 2

theorem fruit_basket_cost_is_28 : basket_total_cost = 28 := by
  sorry

end fruit_basket_cost_is_28_l561_561478


namespace total_tape_length_is_230_l561_561255

def tape_length (n : ℕ) (len_piece : ℕ) (overlap : ℕ) : ℕ :=
  len_piece + (n - 1) * (len_piece - overlap)

theorem total_tape_length_is_230 :
  tape_length 15 20 5 = 230 := 
    sorry

end total_tape_length_is_230_l561_561255


namespace total_cost_is_32_l561_561621

-- Step d: Rewrite the math proof problem in Lean 4 statement.

-- Define the number of people on the committee
def num_people : ℕ := 24

-- Define the number of sandwiches per person
def sandwiches_per_person : ℕ := 2

-- Define the number of croissants per set
def croissants_per_set : ℕ := 12

-- Define the cost per set of croissants
def cost_per_set : ℕ := 8

-- Define the number of croissants needed
def croissants_needed : ℕ := sandwiches_per_person * num_people

-- Calculate the number of sets needed
def sets_needed : ℕ := croissants_needed / croissants_per_set

-- Calculate the total cost
def total_cost : ℕ := sets_needed * cost_per_set

-- Prove the total cost is $32
theorem total_cost_is_32 : total_cost = 32 := 
by
  -- importing necessary library
  -- proving the total cost calculation, we already know
  -- the number of croissants_needed, sets_needed, calculating total_cost
  rw [croissants_needed, sets_needed, total_cost],
  simp,
  sorry

end total_cost_is_32_l561_561621


namespace chromium_alloy_problem_l561_561742

theorem chromium_alloy_problem (x : ℝ) : 
  (1.8 + 0.08 * x = (1 / 11) * (15 + x)) → (x ≈ 40) :=
by  
  sorry

end chromium_alloy_problem_l561_561742


namespace distance_between_given_planes_l561_561110

noncomputable def distance_between_planes : ℝ :=
  let plane1 : ℝ × ℝ × ℝ → ℝ := λ p, 3 * p.1 - p.2 + p.3 - 2
  let plane2 : ℝ × ℝ × ℝ → ℝ := λ p, 6 * p.1 - 2 * p.2 + 2 * p.3 + 4
  let normal_vector : ℝ × ℝ × ℝ := (3, -1, 1)
  let point_on_plane1 : ℝ × ℝ × ℝ := (0, -2, 0)
  let norm := real.sqrt (3^2 + (-1)^2 + 1^2)
  let numer := abs (6*point_on_plane1.1 - 2*point_on_plane1.2 + 2*point_on_plane1.3 + 4)
  numer / norm

theorem distance_between_given_planes : distance_between_planes = 8 * real.sqrt 11 / 11 :=
by
  sorry

end distance_between_given_planes_l561_561110


namespace g_of_5_eq_15_l561_561541

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem g_of_5_eq_15 : g 5 = 15 := 
by
  rw [g]
  simp
  sorry

end g_of_5_eq_15_l561_561541


namespace lowest_price_increase_revenue_l561_561288

variables (a k : ℝ)

-- Define the last year's parameters
def last_year_price : ℝ := 0.8
def last_year_consumption : ℝ := a
def cost_price : ℝ := 0.3
def min_price : ℝ := 0.55
def max_price : ℝ := 0.75
def desired_price : ℝ := 0.4

-- Define the relationship between price and consumption increase
def consumption_increase (x : ℝ) : ℝ := k / (x - desired_price) + last_year_consumption

-- Define the power department's revenue function this year
def revenue (x : ℝ) : ℝ := (consumption_increase a k x) * (x - cost_price)

-- Main theorem to prove
theorem lowest_price_increase_revenue (h_k : k = 0.2 * a) :
  0.6 ≤ x ∧ x ≤ 0.75 → 
  (∀ x, 0.55 ≤ x ∧ x ≤ 0.75 →
    revenue a k x ≥ (last_year_consumption * (last_year_price - cost_price) * 1.2)) := sorry

end lowest_price_increase_revenue_l561_561288


namespace robert_total_balls_l561_561812

-- Define the conditions
def robert_initial_balls : ℕ := 25
def tim_balls : ℕ := 40

-- Mathematically equivalent proof problem
theorem robert_total_balls : 
  robert_initial_balls + (tim_balls / 2) = 45 := by
  sorry

end robert_total_balls_l561_561812


namespace lunks_to_apples_l561_561692

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l561_561692


namespace avg_ge_neg_half_l561_561325

noncomputable def sequence : ℕ → ℤ
| 0       := 0
| (i + 1) := if i % 2 = 0 then sequence i + 1 else -sequence i - 1

theorem avg_ge_neg_half (n : ℕ) (hn : n ≥ 1) : 
  (↑( (Finset.range n).sum sequence )) / n ≥ -(1 / 2 : ℤ) := by
  sorry

end avg_ge_neg_half_l561_561325


namespace divisors_of_n_squared_l561_561268

-- Definition for a number having exactly 4 divisors
def has_four_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^3

-- Theorem statement
theorem divisors_of_n_squared (n : ℕ) (h : has_four_divisors n) : 
  Nat.divisors_count (n^2) = 7 :=
by
  sorry

end divisors_of_n_squared_l561_561268


namespace sachin_age_l561_561018

theorem sachin_age (S R : ℕ) (h1 : R = S + 18) (h2 : S * 9 = R * 7) : S = 63 := 
by
  sorry

end sachin_age_l561_561018


namespace opposite_of_neg_2023_l561_561410

theorem opposite_of_neg_2023 : ∃ x : ℤ, (x + (-2023) = 0) ∧ x = 2023 := 
by
  use 2023
  constructor
  · apply Int.add_eq_zero_iff_eq_neg.2
    exact eq.refl 2023
  · exact eq.refl 2023

end opposite_of_neg_2023_l561_561410


namespace number_of_divisors_of_2310_l561_561590

theorem number_of_divisors_of_2310 : 
  let n := 2310 in
  let prime_factors := (2, 1) :: (3, 1) :: (5, 1) :: (7, 1) :: (11, 1) :: [] in
  n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 →
  (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) = 32 :=
begin
  intro h,
  sorry
end

end number_of_divisors_of_2310_l561_561590


namespace reciprocal_product_l561_561007

theorem reciprocal_product :
  (1 / 3 * 3 / 4) ⁻¹ = 4 := 
  by
    sorry

end reciprocal_product_l561_561007


namespace line_length_difference_l561_561562

theorem line_length_difference
  (length_white : ℝ)
  (length_blue : ℝ)
  (h_white : length_white = 7.666666666666667)
  (h_blue : length_blue = 3.3333333333333335) :
  length_white - length_blue = 4.333333333333333 :=
by
  rw [h_white, h_blue]
  norm_num

end line_length_difference_l561_561562


namespace TJs_average_time_per_km_l561_561820

theorem TJs_average_time_per_km :
  let total_distance := 10
  let first_half_time := 20
  let second_half_time := 30
  let total_time := first_half_time + second_half_time
  let average_time_per_km := total_time / total_distance
  average_time_per_km = 5 :=
by
  let total_distance := 10
  let first_half_time := 20
  let second_half_time := 30
  let total_time := first_half_time + second_half_time
  let average_time_per_km := total_time / total_distance
  show average_time_per_km = 5 from
    sorry  -- proof goes here

end TJs_average_time_per_km_l561_561820


namespace find_EF_l561_561328

noncomputable def isosceles_trapezoid (A B C D E F : ℝ) (AD BC : ℝ) (diagonals_eq : ℝ) (angle_AD_eq : ℝ) : Prop :=
  AD = 16 * Real.sqrt 10 ∧ 8 * Real.sqrt 50 = diagonals_eq ∧ angle_AD_eq = π / 4

noncomputable def distances (E A D : ℝ) : Prop :=
  E = 8 * Real.sqrt 10 ∧ D = 24 * Real.sqrt 10

theorem find_EF {A B C D E F : ℝ} (h1 : isosceles_trapezoid A B C D E F 16 (8 * Real.sqrt 50) (π / 4))
  (h2 : distances (8 * Real.sqrt 10) 16) : EF = 32 * Real.sqrt 5 :=
by sorry

end find_EF_l561_561328


namespace age_sum_future_total_l561_561871

theorem age_sum_future_total :
  ∀ (F S : ℕ), F + S = 55 ∧ F = 37 ∧ S = 18 → (F + 19) + (S + 19) = 93 :=
by
  intros F S h
  cases h with h1 h2
  cases h2 with hF hS
  rw [hF, hS] at h1
  rw [hF, hS]
  exact rfl

end age_sum_future_total_l561_561871


namespace centers_form_regular_ngon_iff_affine_regular_ngon_l561_561359

noncomputable def center_of_ngon (A : Fin n -> Complex) : Fin n -> Complex := sorry

def affine_regular_ngon (A : Fin n -> Complex) : Prop :=
  ∀ j : Fin n, cos (2 * π / n : ℝ) * A j = A (j - 1) + A (j + 1)

def forms_regular_ngon (B : Fin n -> Complex) : Prop :=
  ∀ j : Fin n, B j = (Complex.exp (2 * π * Complex.I / n)) * B (j - 1)

theorem centers_form_regular_ngon_iff_affine_regular_ngon {n : ℕ} (A : Fin n -> Complex) (hA : Affine.convex_ngon A) :
  forms_regular_ngon (center_of_ngon A) ↔ affine_regular_ngon A := sorry

end centers_form_regular_ngon_iff_affine_regular_ngon_l561_561359


namespace domain_of_f_l561_561544

open Set

def f (x : ℝ) : ℝ := real.cbrt (x - 3) + real.cbrt (5 - x) + real.sqrt (x + 1)

theorem domain_of_f : {x : ℝ | ∃ y, y = f x} = Ici (-1) := by
  -- Proof starts here
  sorry

end domain_of_f_l561_561544


namespace ship_length_correct_l561_561056

noncomputable def ship_length : ℝ :=
  let speed_kmh := 24
  let speed_mps := speed_kmh * 1000 / 3600
  let time := 202.48
  let bridge_length := 900
  let total_distance := speed_mps * time
  total_distance - bridge_length

theorem ship_length_correct : ship_length = 450.55 :=
by
  -- This is where the proof would be written, but we're skipping the proof as per instructions
  sorry

end ship_length_correct_l561_561056


namespace algebraic_expression_value_l561_561719

variables {m n : ℝ}

theorem algebraic_expression_value (h : n = 3 - 5 * m) : 10 * m + 2 * n - 3 = 3 :=
by sorry

end algebraic_expression_value_l561_561719


namespace number_of_divisors_2310_l561_561582

-- Define the number whose divisors are being counted
def n : ℕ := 2310

-- Define the prime factorization of the number
def factorization : n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num

-- Define the formula for the number of divisors
def num_divisors (n : ℕ) : ℕ :=
  let e_1 := 1
  let e_2 := 1
  let e_3 := 1
  let e_4 := 1
  let e_5 := 1
  in (e_1 + 1) * (e_2 + 1) * (e_3 + 1) * (e_4 + 1) * (e_5 + 1)

-- State the problem in a theorem
theorem number_of_divisors_2310 : num_divisors n = 32 :=
by
  rw [num_divisors, factorization]
  sorry

end number_of_divisors_2310_l561_561582


namespace min_value_expression_l561_561569

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (inf {x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ x = (|6 * a - 4 * b| + |3 * (a + b * Real.sqrt 3) + 2 * (a * Real.sqrt 3 - b)|) / Real.sqrt (a^2 + b^2)}) = Real.sqrt 39 :=
by
  sorry

end min_value_expression_l561_561569


namespace lunks_to_apples_l561_561693

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end lunks_to_apples_l561_561693


namespace no_partition_of_integers_l561_561374

theorem no_partition_of_integers (A B C : Set ℕ) :
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (∀ a b, a ∈ A ∧ b ∈ B → (a^2 - a * b + b^2) ∈ C) ∧
  (∀ a b, a ∈ B ∧ b ∈ C → (a^2 - a * b + b^2) ∈ A) ∧
  (∀ a b, a ∈ C ∧ b ∈ A → (a^2 - a * b + b^2) ∈ B) →
  False := 
sorry

end no_partition_of_integers_l561_561374


namespace negation_of_exists_sin_cos_le_sqrt2_l561_561407

theorem negation_of_exists_sin_cos_le_sqrt2 :
  ¬ (∃ x_0 : ℝ, sin x_0 + cos x_0 ≤ real.sqrt 2) ↔ ∀ x : ℝ, sin x + cos x > real.sqrt 2 :=
by {
  sorry,
}

end negation_of_exists_sin_cos_le_sqrt2_l561_561407


namespace volume_of_region_l561_561612

theorem volume_of_region :
  ∃ (V : ℝ), V = 9 ∧
  ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 6 :=
sorry

end volume_of_region_l561_561612


namespace sue_received_votes_l561_561412

theorem sue_received_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  (sue_percentage * total_votes) = 350 := by
  sorry

end sue_received_votes_l561_561412


namespace remainder_division_l561_561119

theorem remainder_division (α : ℂ) (hα : α^4 + α^3 + α^2 + α + 1 = 0) : 
  ∃ r, r = 1 ∧ ∀ β : ℂ, (β = α) → (β^(30) + β^(24) + β^(18) + β^(12) + β^(6) + 1) % (β^4 + β^3 + β^2 + β + 1) = r :=
by
  have hα5 : α^5 = 1,
  { sorry },
  use 1,
  split,
  { refl },
  { intros β hβ,
    have hβ5 : β^5 = 1,
    { rw [←hβ, hα5] },
    calc
      (β^(30) + β^(24) + β^(18) + β^(12) + β^(6) + 1) % (β^4 + β^3 + β^2 + β + 1)
          = (1 + β^4 + β^3 + β^2 + β + 1) % (β^4 + β^3 + β^2 + β + 1) : sorry
      ... = 1 : sorry
  }

end remainder_division_l561_561119


namespace number_of_divisors_2310_l561_561579

-- Define the number whose divisors are being counted
def n : ℕ := 2310

-- Define the prime factorization of the number
def factorization : n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 := by norm_num

-- Define the formula for the number of divisors
def num_divisors (n : ℕ) : ℕ :=
  let e_1 := 1
  let e_2 := 1
  let e_3 := 1
  let e_4 := 1
  let e_5 := 1
  in (e_1 + 1) * (e_2 + 1) * (e_3 + 1) * (e_4 + 1) * (e_5 + 1)

-- State the problem in a theorem
theorem number_of_divisors_2310 : num_divisors n = 32 :=
by
  rw [num_divisors, factorization]
  sorry

end number_of_divisors_2310_l561_561579


namespace suzanna_history_book_pages_l561_561383

theorem suzanna_history_book_pages (H G M S : ℕ) 
  (h_geography : G = H + 70)
  (h_math : M = (1 / 2) * (H + H + 70))
  (h_science : S = 2 * H)
  (h_total : H + G + M + S = 905) : 
  H = 160 := 
by
  sorry

end suzanna_history_book_pages_l561_561383


namespace pencils_removed_l561_561513

theorem pencils_removed (initial_pencils removed_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : remaining_pencils = 83) 
  (h3 : removed_pencils = initial_pencils - remaining_pencils) : 
  removed_pencils = 4 :=
sorry

end pencils_removed_l561_561513


namespace third_number_hcf_lcm_l561_561402

theorem third_number_hcf_lcm (N : ℕ) 
  (HCF : Nat.gcd (Nat.gcd 136 144) N = 8)
  (LCM : Nat.lcm (Nat.lcm 136 144) N = 2^4 * 3^2 * 17 * 7) : 
  N = 7 := 
  sorry

end third_number_hcf_lcm_l561_561402


namespace cauchy_schwarz_example_l561_561371

theorem cauchy_schwarz_example (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end cauchy_schwarz_example_l561_561371


namespace zoo_charge_for_child_l561_561823

theorem zoo_charge_for_child (charge_adult : ℕ) (total_people total_bill children : ℕ) (charge_child : ℕ) : 
  charge_adult = 8 → total_people = 201 → total_bill = 964 → children = 161 → 
  total_bill - (total_people - children) * charge_adult = children * charge_child → 
  charge_child = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end zoo_charge_for_child_l561_561823


namespace problem_l561_561333

/-- 
Let {a_n} be a sequence of positive terms, and its partial sum S_n satisfies 
the equation 4S_n = (a_n - 1)(a_n + 3). Determine the value of a_2018.
-/

def a_seq (n : ℕ) : ℕ := sorry -- define the sequence a_n

def S_n (n : ℕ) : ℕ := (∑ k in finset.range n, a_seq k) -- define the partial sum S_n

theorem problem (n : ℕ) (h1 : ∀ n, a_seq n > 0) (h2 : ∀ n, 4 * S_n n = (a_seq n - 1) * (a_seq n + 3)) :
  a_seq 2018 = 4037 :=
by {
  sorry
}

end problem_l561_561333


namespace parking_garage_savings_l561_561935

theorem parking_garage_savings :
  let weekly_cost := 10
  let monthly_cost := 35
  let weeks_per_year := 52
  let months_per_year := 12
  let annual_weekly_cost := weekly_cost * weeks_per_year
  let annual_monthly_cost := monthly_cost * months_per_year
  let annual_savings := annual_weekly_cost - annual_monthly_cost
  annual_savings = 100 := 
by
  sorry

end parking_garage_savings_l561_561935


namespace fraction_identity_l561_561524

noncomputable def simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : ℝ :=
  (1 / (2 * a * b)) + (b / (4 * a))

theorem fraction_identity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  simplify_fraction a b h₁ h₂ = (2 + b^2) / (4 * a * b) :=
by sorry

end fraction_identity_l561_561524


namespace max_sum_achievable_l561_561915

-- Define the game rules and conditions
def player := ℕ → ℕ × ℕ → ℕ

def A_fills (k : ℕ) (x y : ℕ) : ℕ := 
  if k % 2 = 0 then 1 else 0

def B_fills (k : ℕ) (x y : ℕ) : ℕ := 
  if k % 2 = 1 then 0 else 1

-- Define the grid size
def grid_size : ℕ := 5

-- Define the calculation of the sum in each 3x3 sub-grid
def subgrid_sum (grid : ℕ × ℕ → ℕ) (i j : ℕ) : ℕ :=
  (list.fin_range 3).sum (λ di, (list.fin_range 3).sum (λ dj, grid (i + di, j + dj)))

-- Define the maximum sum that A can achieve
def max_subgrid_sum : ℕ :=
  finset.univ.fold max 0 (λ i, finset.univ.fold max 0 (λ j, 
    if i + 2 < grid_size ∧ j + 2 < grid_size then subgrid_sum (A_fills 0) i j else 0))

-- Prove that the maximum achievable sum in any 3x3 sub-grid is 6
theorem max_sum_achievable : max_subgrid_sum = 6 :=
sorry

end max_sum_achievable_l561_561915


namespace min_ratio_OA_OC_OB_OD_l561_561751

variables {A B C D O : ℝ}
-- Assume the points form a square and OA, OB, OC, and OD represent the distances from O to each vertex of the square

noncomputable def OA : ℝ := real.sqrt ((O.x - A.x)^2 + (O.y - A.y)^2)
noncomputable def OB : ℝ := real.sqrt ((O.x - B.x)^2 + (O.y - B.y)^2)
noncomputable def OC : ℝ := real.sqrt ((O.x - C.x)^2 + (O.y - C.y)^2)
noncomputable def OD : ℝ := real.sqrt ((O.x - D.x)^2 + (O.y - D.y)^2)

theorem min_ratio_OA_OC_OB_OD (O A B C D : Point) (h: is_square A B C D) :
  ∃ O, (OA + OC) / (OB + OD) = 1 / real.sqrt 2 :=
sorry

end min_ratio_OA_OC_OB_OD_l561_561751


namespace find_four_digit_numbers_l561_561877

theorem find_four_digit_numbers
  (A B : ℕ)
  (hA_digits : 1000 ≤ A ∧ A < 10000)
  (hB_digits : 1000 ≤ B ∧ B < 10000)
  (hlog : ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ log10 A = a + log10 b)
  (hsum_digits : let d_thousands := B / 1000 % 10 in
                 let d_units := B % 10 in
                 d_thousands + d_units = (5/2 : ℚ) * (hlog.some_spec.some : ℕ)) -- b
  (hrel : let b := hlog.some_spec.some in
          B = A / 2 - (5 * b + 1))
  : A = 4000 ∧ B = 1979 :=
by sorry

end find_four_digit_numbers_l561_561877


namespace TJs_average_time_per_km_l561_561819

theorem TJs_average_time_per_km :
  let total_distance := 10
  let first_half_time := 20
  let second_half_time := 30
  let total_time := first_half_time + second_half_time
  let average_time_per_km := total_time / total_distance
  average_time_per_km = 5 :=
by
  let total_distance := 10
  let first_half_time := 20
  let second_half_time := 30
  let total_time := first_half_time + second_half_time
  let average_time_per_km := total_time / total_distance
  show average_time_per_km = 5 from
    sorry  -- proof goes here

end TJs_average_time_per_km_l561_561819


namespace draw_9_cards_ensure_even_product_l561_561016

theorem draw_9_cards_ensure_even_product :
  ∀ (cards : Finset ℕ), (∀ x ∈ cards, 1 ≤ x ∧ x ≤ 16) →
  (cards.card = 9) →
  (∃ (subset : Finset ℕ), subset ⊆ cards ∧ ∃ k ∈ subset, k % 2 = 0) :=
by
  sorry

end draw_9_cards_ensure_even_product_l561_561016


namespace movie_ticket_distribution_l561_561100

theorem movie_ticket_distribution (A B P1 P2 P3 : Type) [Finite A] [Finite B] [Finite P1] [Finite P2] [Finite P3] :
  let tickets := [1, 2, 3, 4, 5]
  let distribute := λ (x y : Type), (x, y)
  (count (λ t, 
    let dist := distribute (t A t B t P1 t P2 t P3) in
    -- Ensuring A and B receive consecutive tickets
    ∃ i j, dist A = i ∧ dist B = j ∧ (j = i + 1 ∨ i = j + 1))
    tickets) = 48 :=
sorry

end movie_ticket_distribution_l561_561100


namespace minimum_questions_needed_a_l561_561885

theorem minimum_questions_needed_a (n : ℕ) (m : ℕ) (h1 : m = n) (h2 : m < 2 ^ n) :
  ∃Q : ℕ, Q = n := sorry

end minimum_questions_needed_a_l561_561885


namespace black_wins_l561_561800

-- Definitions of initial setup and conditions
structure Chessboard :=
  (pieces : ℕ → ℕ → option (Σ color : bool, pos : ℕ))

def initial_board : Chessboard :=
  { pieces := λ row col,
      if row = 1 then some ⟨tt, col⟩ -- White pieces on the 1st row
      else if row = 8 then some ⟨ff, col⟩ -- Black pieces on the 8th row
      else none }

structure Move :=
  (piece : Σ color : bool, pos : ℕ)
  (new_row : ℕ)

-- Rules of the game
def valid_move (b : Chessboard) (m : Move) : Prop :=
  ∃ old_row col,
    b.pieces old_row col = some m.piece ∧
    b.pieces m.new_row col = none ∧
    (m.new_row > old_row ∨ m.new_row < old_row)

structure Game := 
  (board : Chessboard)
  (turn : bool) -- true for White's turn, false for Black's turn

-- Initial game setup
def initial_game : Game := { board := initial_board, turn := true }

-- Prove that Black has a winning strategy
theorem black_wins : ∃ strategy : (Game → Move), ∀ g : Game, ¬g.turn → valid_move g.board (strategy g) :=
  sorry

end black_wins_l561_561800


namespace butanoic_acid_molecular_weight_l561_561518

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight_butanoic_acid : ℝ :=
  4 * atomic_weight_C + 8 * atomic_weight_H + 2 * atomic_weight_O

theorem butanoic_acid_molecular_weight :
  molecular_weight_butanoic_acid = 88.104 :=
by
  -- proof not required
  sorry

end butanoic_acid_molecular_weight_l561_561518


namespace parallelogram_cosine_l561_561868

-- Defining points in 2D
structure Point where
  x : ℝ
  y : ℝ

-- Function to calculate midpoint of two points
def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

-- Function to calculate vector between two points
def vector (A B : Point) : Point :=
  { x := B.x - A.x, y := B.y - A.y }

-- Dot product of two vectors
def dot (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Magnitude of a vector
def magnitude (v : Point) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

-- Cosine of the angle between two vectors
def cos_angle (v1 v2 : Point) : ℝ :=
  dot v1 v2 / (magnitude v1 * magnitude v2)

-- Theorem
theorem parallelogram_cosine :
  let A : Point := { x := 0, y := 0 }
  let B : Point := { x := 4, y := 0 }
  let D : Point := { x := 4 * -1/2, y := 4 * Real.sqrt(3) / 2 }
  let C : Point := { x := 2, y := 2 * Real.sqrt(3) }
  let M : Point := midpoint B C
  let N : Point := midpoint C D
  cos_angle (vector A M) (vector A N) = 3 * Real.sqrt 21 / 14 := by
  sorry

end parallelogram_cosine_l561_561868


namespace number_of_positive_divisors_2310_l561_561599

theorem number_of_positive_divisors_2310 : 
  let n := 2310 in
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)] in
  let t := (factorization.map (λ p : ℕ × ℕ, p.snd + 1)).prod in
  t = 32 :=
by
  let n := 2310
  let factorization := [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1)]
  let t := (factorization.map (λ p, p.snd + 1)).prod
  sorry

end number_of_positive_divisors_2310_l561_561599


namespace floor_sqrt_23_squared_l561_561550

theorem floor_sqrt_23_squared : (Nat.floor (Real.sqrt 23)) ^ 2 = 16 :=
by
  -- Proof is omitted
  sorry

end floor_sqrt_23_squared_l561_561550


namespace maximize_expected_score_l561_561037

namespace KnowledgeCompetition

-- Define probabilities and point values
def prob_correct_A : ℝ := 0.8
def prob_correct_B : ℝ := 0.6
def points_A := 20
def points_B := 80

-- Define probabilities of incorrect answers
def prob_incorrect_A : ℝ := 1 - prob_correct_A
def prob_incorrect_B : ℝ := 1 - prob_correct_B

-- Distribution for X when starting with type A
def P_X_0 : ℝ := prob_incorrect_A
def P_X_20 : ℝ := prob_correct_A * prob_incorrect_B
def P_X_100 : ℝ := prob_correct_A * prob_correct_B

-- Expected score when starting with type A
def E_X : ℝ := P_X_0 * 0 + P_X_20 * points_A + P_X_100 * (points_A + points_B)

-- Define probabilities for the distribution of Y when starting with type B
def P_Y_0 : ℝ := prob_incorrect_B
def P_Y_80 : ℝ := prob_correct_B * prob_incorrect_A
def P_Y_100 : ℝ := prob_correct_B * prob_correct_A

-- Expected score when starting with type B
def E_Y : ℝ := P_Y_0 * 0 + P_Y_80 * points_B + P_Y_100 * (points_B + points_A)

-- Proof statement that to maximize the expected cumulative score, Xiao Ming should start with type B questions
theorem maximize_expected_score : E_Y > E_X :=
by
  let expr_E_X := P_X_0 * 0 + P_X_20 * points_A + P_X_100 * (points_A + points_B)
  let expr_E_Y := P_Y_0 * 0 + P_Y_80 * points_B + P_Y_100 * (points_B + points_A)
  have E_X_eq : E_X = expr_E_X := rfl
  have E_Y_eq : E_Y = expr_E_Y := rfl
  have calc_E_X : expr_E_X = 0.2 * 0 + 0.32 * 20 + 0.48 * 100 := by norm_num
  have calc_E_Y : expr_E_Y = 0.4 * 0 + 0.12 * 80 + 0.48 * 100 := by norm_num
  rw [E_X_eq, calc_E_X, E_Y_eq, calc_E_Y]
  norm_num
  sorry

end KnowledgeCompetition

end maximize_expected_score_l561_561037


namespace max_abs_g_eq_eight_l561_561185

noncomputable theory

open Real

def f (a b c x : ℝ) := a * x^2 + b * x + c
def g (a b c x : ℝ) := c * x^2 + b * x + a

theorem max_abs_g_eq_eight {a b c : ℝ}
  (h : ∀ x ∈ Icc (0 : ℝ) 1, abs (f a b c x) ≤ 1) :
  (∀ x ∈ Icc (0 : ℝ) 1, abs (g a b c x) ≤ 8) ∧ 
  (∃ x ∈ Icc (0 : ℝ) 1, abs (g a b c x) = 8) :=
sorry

end max_abs_g_eq_eight_l561_561185


namespace square_rem_1_mod_9_l561_561269

theorem square_rem_1_mod_9 (N : ℤ) (h : N % 9 = 1 ∨ N % 9 = 8) : (N * N) % 9 = 1 :=
by sorry

end square_rem_1_mod_9_l561_561269


namespace trip_time_proof_l561_561919

-- Defining the conditions
def original_time : ℝ := 5 + 1/3
def original_speed : ℝ := 80
def percentage_increase : ℝ := 0.20
def new_speed : ℝ := 70

-- Defining the statement
theorem trip_time_proof : 
  let original_distance := original_speed * original_time in
  let new_distance := original_distance * (1 + percentage_increase) in
  let new_time := new_distance / new_speed in
  Float.round (new_time * 100) / 100 = 7.31 :=
by
  sorry

end trip_time_proof_l561_561919


namespace problem1_problem2_l561_561459

-- Problem 1
theorem problem1 (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 0) :
  (x^2 + x) / (x^2 - 2 * x + 1) / (2 / (x - 1) - 1 / x) = x^2 / (x - 1) := by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (hx1 : x > 0) :
  (2 * x + 1) / 3 - (5 * x - 1) / 2 < 1 ∧ 
  (5 * x - 1 < 3 * (x + 2)) →
  x = 1 ∨ x = 2 ∨ x = 3 := by
  sorry

end problem1_problem2_l561_561459


namespace calculation_correct_l561_561975

theorem calculation_correct :
  8^(2 / 3) + (-1)^0 - (1 / 2)^(-2) - 25^(-1 / 2) = 4 / 5 :=
by
  -- The actual code here would involve Lean's norm_num to prove the statement
  sorry

end calculation_correct_l561_561975


namespace time_after_midnight_1453_minutes_l561_561891

def minutes_to_time (minutes : Nat) : Nat × Nat :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_of_day (hours : Nat) : Nat × Nat :=
  let days := hours / 24
  let remaining_hours := hours % 24
  (days, remaining_hours)

theorem time_after_midnight_1453_minutes : 
  let midnight := (0, 0) -- Midnight as a tuple of hours and minutes
  let total_minutes := 1453
  let (total_hours, minutes) := minutes_to_time total_minutes
  let (days, hours) := time_of_day total_hours
  days = 1 ∧ hours = 0 ∧ minutes = 13
  := by
    let midnight := (0, 0)
    let total_minutes := 1453
    let (total_hours, minutes) := minutes_to_time total_minutes
    let (days, hours) := time_of_day total_hours
    sorry

end time_after_midnight_1453_minutes_l561_561891


namespace count_integers_between_sqrts_l561_561233

theorem count_integers_between_sqrts : 
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  (upper_bound - lower_bound + 1) = 6 :=
by
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  calc (upper_bound - lower_bound + 1) = 6 : sorry

end count_integers_between_sqrts_l561_561233


namespace lunks_for_apples_l561_561699

noncomputable def lunks_per_kunks := 7 / 4
noncomputable def kunks_per_apples := 3 / 5
def apples_needed := 24
noncomputable def kunks_needed_for_apples := (kunks_per_apples * apples_needed).ceil
noncomputable def lunks_needed := (lunks_per_kunks * kunks_needed_for_apples).ceil

theorem lunks_for_apples :
  lunks_needed = 27 :=
sorry

end lunks_for_apples_l561_561699


namespace number_of_integers_between_sqrt8_and_sqrt72_l561_561215

theorem number_of_integers_between_sqrt8_and_sqrt72 : 
  let a := Int.ceil (Real.sqrt 8)
  let b := Int.floor (Real.sqrt 72)
  b - a + 1 = 6 :=
begin
  sorry
end

end number_of_integers_between_sqrt8_and_sqrt72_l561_561215


namespace no_positive_integer_solutions_l561_561365

theorem no_positive_integer_solutions (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  x^3 + 2 * y^3 ≠ 4 * z^3 :=
by
  sorry

end no_positive_integer_solutions_l561_561365


namespace irrational_root_quadratic_l561_561271

theorem irrational_root_quadratic (p q : ℚ) (x1 x2 : ℝ) 
  (h_eq : ∃ (p q : ℚ), (x1^2 + p * x1 + q = 0)) 
  (h_irr : ¬(x1 ∈ ℚ ∧ x2 ∈ ℚ)) 
  (h_relation : x1 = x2^2) : p = 1 ∧ q = 1 := 
by {
  sorry 
}

end irrational_root_quadratic_l561_561271


namespace area_of_larger_square_is_16_times_l561_561859

-- Define the problem conditions
def perimeter_condition (a b : ℝ) : Prop :=
  4 * a = 4 * 4 * b

-- Define the relationship between the areas of the squares given the side lengths
def area_ratio (a b : ℝ) : ℝ :=
  (a * a) / (b * b)

theorem area_of_larger_square_is_16_times (a b : ℝ) (h : perimeter_condition a b) : area_ratio a b = 16 :=
by 
  unfold perimeter_condition at h
  rw [mul_assoc, mul_comm 4 b] at h
  have ha : a = 4 * b := (mul_right_inj' (ne_of_gt (show 0 < (4:ℝ), by norm_num))).mp h
  unfold area_ratio
  rw [ha, mul_pow, pow_two, mul_pow]
  exact (by norm_num : (4:ℝ)^2 = 16)

end area_of_larger_square_is_16_times_l561_561859


namespace max_interior_angles_less_than_120_l561_561724

theorem max_interior_angles_less_than_120 (n : ℕ) (θ : ℕ → ℕ) 
  (polygon_convex : True)
  (sum_exterior_angles : ∑ i in Finset.range n, (180 - θ i) = 360) 
  (interior_angle_condition : ∀ i, θ i < 120 → 180 - θ i > 60)
  (count_angles_less_than_120 : ∀ s, s = {i | θ i < 120} → s.card = k) :
  k ≤ 5 := 
sorry

end max_interior_angles_less_than_120_l561_561724


namespace count_integers_between_sqrt8_sqrt72_l561_561249

-- Define the square roots of 8 and 72 for reference
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt(8)
def minIntAboveSqrt8 : ℤ := Int.ceil sqrt8

-- Define the largest integer less than sqrt(72)
def maxIntBelowSqrt72 : ℤ := Int.floor sqrt72

-- State the main theorem to prove
theorem count_integers_between_sqrt8_sqrt72 :
  minIntAboveSqrt8 = 3 ∧ maxIntBelowSqrt72 = 8 → 
  (maxIntBelowSqrt72 - minIntAboveSqrt8 + 1) = 6 :=
by
  sorry

end count_integers_between_sqrt8_sqrt72_l561_561249


namespace pumps_empty_pool_together_in_144_minutes_l561_561448

theorem pumps_empty_pool_together_in_144_minutes (timeA timeB : ℕ) 
    (h1 : timeA = 4) (h2 : timeB = 6) : 
    ((1 / timeA.toReal) + (1 / timeB.toReal) > 0) → 144 :=
by
  sorry

end pumps_empty_pool_together_in_144_minutes_l561_561448


namespace star_number_of_intersections_2018_25_l561_561987

-- Definitions for the conditions
def rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def star_intersections (n k : ℕ) : ℕ := 
  n * (k - 1)

-- The main theorem
theorem star_number_of_intersections_2018_25 :
  2018 ≥ 5 ∧ 25 < 2018 / 2 ∧ rel_prime 2018 25 → 
  star_intersections 2018 25 = 48432 :=
by
  intros h
  sorry

end star_number_of_intersections_2018_25_l561_561987


namespace geometric_series_formula_geometric_series_n_eq_1_left_side_l561_561804

-- Define the sequence sum for arbitrary n
noncomputable def geometric_sum (a : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a ^ i

-- State the theorem for the general case
theorem geometric_series_formula {a : ℝ} {n : ℕ} (h₀ : a ≠ 0) (h₁ : a ≠ 1) (hn : 0 < n) :
  geometric_sum a n = (1 - a^(n+1)) / (1 - a) :=
sorry

-- State the specific case for n = 1
theorem geometric_series_n_eq_1_left_side {a : ℝ} (h₀ : a ≠ 0) (h₁ : a ≠ 1) :
  geometric_sum a 1 = 1 + a + a^2 :=
sorry

end geometric_series_formula_geometric_series_n_eq_1_left_side_l561_561804


namespace cone_height_approx_20_l561_561926

noncomputable def cone_height (V : ℝ) (vertex_angle_deg : ℝ) : ℝ :=
  let r_approx : ℝ := (V * 3 / (π * 0.414)) ^ (1/3) in
  r_approx * 0.414

theorem cone_height_approx_20 
  (volume : ℝ) 
  (vertex_angle : ℝ) 
  (V_eq : volume = 15552 * π) 
  (angle_eq : vertex_angle = 45) : 
  abs (cone_height volume vertex_angle - 20.0) < 0.1 := 
by
  sorry

end cone_height_approx_20_l561_561926


namespace cubes_with_one_face_painted_l561_561436

theorem cubes_with_one_face_painted :
  ∀ (a b c : ℕ),
  a = 9 → b = 10 → c = 11 →
  2 * ((a - 2) * (b - 2)) + 2 * ((a - 2) * (c - 2)) + 2 * ((b - 2) * (c - 2)) = 382 :=
by
  intros a b c
  assume h₁ : a = 9
  assume h₂ : b = 10
  assume h₃ : c = 11
  rw [h₁, h₂, h₃]
  sorry

end cubes_with_one_face_painted_l561_561436


namespace no_intersection_slope_range_l561_561546

variable {R : Type*} [LinearOrderedField R]

def point (x y : R) := (x, y)
def line_through (P : R × R) (k : R) : R → R := λ x, k * x + P.2

theorem no_intersection_slope_range
  (P : R × R) (A : R × R) (B : R × R)
  (l : ∀ k : R, R → R)
  (slope_PA := (A.2 - P.2) / (A.1 - P.1))
  (slope_PB := (B.2 - P.2) / (B.1 - P.1)) :
  (∀ k, l k = line_through P k → ( ∀ x : R, (l k x) ≠ (line_through A ((B.2 - A.2) / (B.1 - A.1)) x) →
  (k < slope_PA ∨ k > slope_PB))) :=
sorry

end no_intersection_slope_range_l561_561546


namespace minimize_ratio_surface_area_to_volume_l561_561138

-- Definitions
def edge_length := 1.0

def tetrahedron_surface_area (a : ℝ) : ℝ := 4 * (real.sqrt 3 / 4 * a^2)

def tetrahedron_volume (a : ℝ) : ℝ := (a^3 / (6 * real.sqrt 2))

-- The function representing the ratio after truncation
def ratio_surface_area_to_volume (n : ℕ) (h : n > 1) : ℝ :=
  let A := tetrahedron_surface_area edge_length 
  let V := tetrahedron_volume edge_length 
  let A_n := A * (1 - 2 / (n^2)) 
  let V_n := V * (1 - 4 / (n^3)) 
  (A_n / V_n)

theorem minimize_ratio_surface_area_to_volume : ∀ n : ℕ, n > 1 → ratio_surface_area_to_volume n = ratio_surface_area_to_volume 3 := by
  sorry

end minimize_ratio_surface_area_to_volume_l561_561138


namespace rhombus_dif_squares_eq_prod_distances_l561_561798

variables (A B C D M : Point)
variables (d1 d2 : Line)

-- Conditions
def is_rhombus (A B C D : Point) : Prop :=
  distance A B = distance B C ∧
  distance B C = distance C D ∧
  distance C D = distance D A ∧
  is_parallel (Line.mk A C) (Line.mk B D)

def on_diagonal (M : Point) (A C : Point) : Prop :=
  collinear M A C

-- Question/Goal
theorem rhombus_dif_squares_eq_prod_distances 
  (h1 : is_rhombus A B C D) 
  (h2 : on_diagonal M A C) 
  (h3 : intersects_diagonals A C B D O) 
  : distance B A ^ 2 - distance B M ^ 2 = distance M A * distance M C :=
sorry

end rhombus_dif_squares_eq_prod_distances_l561_561798


namespace type_I_patterns_count_l561_561341

open Finset

def euler_totient (d : ℕ) : ℕ := (range d).filter (nat.coprime d).card

noncomputable def number_of_type_I_patterns (n m : ℕ) : ℕ :=
  (1 / n) * (range n).filter (λ d, n % d = 0).sum (λ d, euler_totient d * m ^ (n / d))

theorem type_I_patterns_count (n m : ℕ) :
  number_of_type_I_patterns n m = (1 / n) * (range n).filter (λ d, n % d = 0).sum (λ d, euler_totient d * m ^ (n / d)) :=
  sorry

end type_I_patterns_count_l561_561341


namespace integer_count_between_sqrt8_and_sqrt72_l561_561219

theorem integer_count_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ( ∀ x : ℤ, (⌊Real.sqrt 8⌋.to_nat + 1) ≤ x ∧ x ≤ ⌊Real.sqrt 72⌋ - 1 → x = 6 ) :=
by 
  -- Define the floor and ceiling functions
  have sqrt_8_ceil : ⌈Real.sqrt 8⌉ = 3 := sorry
  have sqrt_72_floor : ⌊Real.sqrt 72⌋ = 8 := sorry

  use 6,
  split,
  { refl },
  { intros x hx,
    sorry
  }

-- Additional supporting facts
lemma sqrt_8_approx : Real.sqrt 8 ≈ 2.83 := sorry
lemma sqrt_72_approx : Real.sqrt 72 ≈ 8.49 := sorry

end integer_count_between_sqrt8_and_sqrt72_l561_561219


namespace professors_chair_ways_l561_561103

theorem professors_chair_ways : 
  let chairs := 11
  let students := 7
  let professors := 4
  (forall (p : Fin professors), 
    (exists (c : Fin chairs), 
      1 < c.toNat ∧ c.toNat < chairs - 1)) →
  ∃ (ways : Nat), ways = 24 :=
by
  let chairs := 11
  let students := 7
  let professors := 4
  assume seating_constraints : 
    forall (p : Fin professors),
    (exists (c : Fin chairs), 
      1 < c.toNat ∧ c.toNat < chairs - 1)
  show ∃ (ways : Nat), ways = 24
  exact ⟨24, sorry⟩

end professors_chair_ways_l561_561103


namespace count_integers_between_sqrt8_sqrt72_l561_561250

-- Define the square roots of 8 and 72 for reference
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt(8)
def minIntAboveSqrt8 : ℤ := Int.ceil sqrt8

-- Define the largest integer less than sqrt(72)
def maxIntBelowSqrt72 : ℤ := Int.floor sqrt72

-- State the main theorem to prove
theorem count_integers_between_sqrt8_sqrt72 :
  minIntAboveSqrt8 = 3 ∧ maxIntBelowSqrt72 = 8 → 
  (maxIntBelowSqrt72 - minIntAboveSqrt8 + 1) = 6 :=
by
  sorry

end count_integers_between_sqrt8_sqrt72_l561_561250


namespace divisibility_ac_bd_l561_561372

-- Conditions definitions
variable (a b c d : ℕ)
variable (hab : a ∣ b)
variable (hcd : c ∣ d)

-- Goal
theorem divisibility_ac_bd : (a * c) ∣ (b * d) :=
  sorry

end divisibility_ac_bd_l561_561372


namespace max_f_value_l561_561323

-- Definitions
def P := { x : Vector Nat 2012 // ∀ i, x.get i ≤ 20 ∧ x.get i ≥ 1 }

def decreasing (A : Set P) : Prop :=
∀ x ∈ A, ∀ y : P, (∀ i, y.val.get i ≤ x.val.get i) → y ∈ A

def increasing (B : Set P) : Prop :=
∀ x ∈ B, ∀ y : P, (∀ i, y.val.get i ≥ x.val.get i) → y ∈ B

def f (A B : Set P) : ℚ :=
|A ∩ B| / (|A| * |B|)

-- Main statement
theorem max_f_value (A B : Set P) (hA : decreasing A) (hB : increasing B) (hA_nonempty : A.nonempty) (hB_nonempty : B.nonempty) :
  f A B ≤ (1 / 20)^2012 := sorry

end max_f_value_l561_561323


namespace area_EFG_le_max_perimeter_EFG_le_max_l561_561644

variables {A B C D E F G: Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables {triangle : Type} [MeasurableSpace triangle]

def area (t: triangle) : ℝ := sorry -- Assume a definition for area of a triangle

def perimeter (t: triangle) : ℝ := sorry -- Assume a definition for perimeter of a triangle

def on_segment (P Q R : Type) : Prop := sorry -- Assume a definition for points on a segment

-- Conditions
variable (tetrahedron : A × B × C × D)
variable (EonAB : on_segment E A B)
variable (FonAC : on_segment F A C)
variable (GonAD : on_segment G A D)

-- Proofs to be provided
theorem area_EFG_le_max {ABC ABD ACD BCD : triangle} :
  area (⟨E, F, G⟩ : triangle) ≤ max (area ABC) (max (area ABD) (max (area ACD) (area BCD))) := 
sorry

theorem perimeter_EFG_le_max {ABC ABD ACD BCD : triangle} :
  perimeter (⟨E, F, G⟩ : triangle) ≤ max (perimeter ABC) (max (perimeter ABD) (max (perimeter ACD) (perimeter BCD))) := 
sorry

end area_EFG_le_max_perimeter_EFG_le_max_l561_561644


namespace aiyanna_cookies_l561_561957

theorem aiyanna_cookies (a b : ℕ) (h₁ : a = 129) (h₂ : b = a + 11) : b = 140 := by
  sorry

end aiyanna_cookies_l561_561957


namespace scientific_notation_correct_l561_561561

-- Define the original number we want to express in scientific notation
def original_number : ℝ := 0.00000573

-- Define the scientific notation form we want to check against
def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * 10 ^ n

-- The specific values for a and n
def a : ℝ := 5.73
def n : ℤ := -6

-- The statement we need to prove
theorem scientific_notation_correct : original_number = scientific_notation a n :=
by sorry

end scientific_notation_correct_l561_561561


namespace floor_sqrt_23_squared_eq_16_l561_561553

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end floor_sqrt_23_squared_eq_16_l561_561553


namespace distance_between_points_is_correct_l561_561973

theorem distance_between_points_is_correct :
  let x1 := 1
  let y1 := 2
  let x2 := 8
  let y2 := 14
  (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = real.sqrt 193) :=
by
  let x1 := 1
  let y1 := 2
  let x2 := 8
  let y2 := 14
  show (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = real.sqrt 193)
  sorry

end distance_between_points_is_correct_l561_561973


namespace mouse_average_shortest_distance_in_square_l561_561048

noncomputable def average_shortest_distance
  (side_length : ℝ := 15)
  (scurry_distance : ℝ := 9)
  (first_turn_distance : ℝ := 3)
  (second_turn_distance : ℝ := 1.5) : ℝ :=
let diagonal_length := (real.sqrt (side_length^2 + side_length^2)),
    initial_x := (scurry_distance * side_length / diagonal_length),
    initial_y := (scurry_distance * side_length / diagonal_length),
    after_first_turn_x := initial_x + first_turn_distance,
    after_first_turn_y := initial_y,
    final_x := after_first_turn_x,
    final_y := after_first_turn_y - second_turn_distance in
((final_x + final_y + (side_length - final_x) + (side_length - final_y)) / 4)

theorem mouse_average_shortest_distance_in_square :
  average_shortest_distance = 7.5 :=
sorry

end mouse_average_shortest_distance_in_square_l561_561048


namespace projection_is_p_l561_561892

section ProjectionProof

open Real

noncomputable def p : ℝ × ℝ := (48 / 53, 168 / 53)
def point_a : ℝ × ℝ := (5, 2)
def point_b : ℝ × ℝ := (-2, 4)
def direction_vector : ℝ × ℝ := (-7, 2)

def line (t : ℝ) : ℝ × ℝ :=
  (point_a.fst + t * direction_vector.fst, point_a.snd + t * direction_vector.snd)

def is_orthogonal (v w : ℝ × ℝ) : Prop :=
  v.fst * w.fst + v.snd * w.snd = 0

theorem projection_is_p :
  ∃ v : ℝ, line v = p ∧ is_orthogonal (line v) direction_vector :=
begin
  use 31 / 53,
  split,
  {
    -- Proof of line (31 / 53) = p
    rw line,
    simp,
    field_simp,
  },
  {
    -- Proof of is_orthogonal (line (31 / 53)) direction_vector
    rw is_orthogonal,
    simp,
    field_simp,
    ring,
  }
end

end ProjectionProof

end projection_is_p_l561_561892


namespace find_z_find_m_l561_561789

noncomputable def complex_number (a b : ℝ) (h : a > 0) : Prop :=
  ∃ z : ℂ, z = ⟨a, b⟩ ∧ abs z = sqrt 10 ∧ (1 - 2 * I) * z ∈ {(x : ℂ) | x.re = -x.im ∨ x.re = x.im}

theorem find_z (a b : ℝ) (h1 : a > 0) (h2 : abs (⟨a, b⟩ : ℂ) = sqrt 10)
  (h3 : (1 - 2 * I) * ⟨a, b⟩ ∈ {(z : ℂ) | z.re = -z.im ∨ z.re = z.im}) :
  ∃ z : ℂ, z = ⟨3, 1⟩ :=
sorry

theorem find_m (m : ℝ) (h : ∃ z : ℂ, z = ⟨3, 1⟩ ∧ (⟨3, -1⟩ : ℂ) + (m + I) / (1 - I) = pure_imaginary_part) :
  m = -5 :=
sorry

end find_z_find_m_l561_561789


namespace last_number_in_sequence_l561_561733

theorem last_number_in_sequence :
  ∃ (a : Fin 100 → ℝ), (∀ i : Fin 98, a (⟨i + 2, by simp⟩) = a (⟨i + 1, by simp⟩) * a (⟨i + 3, by simp⟩)) ∧ a 0 = 2018 ∧ a 99 = (1 / 2018) :=
by  
  sorry

end last_number_in_sequence_l561_561733


namespace boundary_length_is_30_8_l561_561883

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def pi : ℝ := Real.pi

def length_of_boundary (area : ℝ) (points_per_side : ℕ) : ℝ :=
  let side := sqrt area
  let segment_length := side / points_per_side
  let straight_segments := segment_length * 4
  let quarter_circle_arcs := 2 * pi * (segment_length / 4)
  straight_segments + quarter_circle_arcs

theorem boundary_length_is_30_8
  (area : ℝ)
  (points_per_side : ℕ)
  (h_area : area = 81)
  (h_pts_per_side : points_per_side = 3) :
  length_of_boundary area points_per_side = 30.8 := by
  -- Given area = 81 and points_per_side = 3
  -- Prove that length_of_boundary 81 3 = 30.8
  sorry

end boundary_length_is_30_8_l561_561883


namespace stratified_sampling_group_C_l561_561055

theorem stratified_sampling_group_C
  (total_cities : ℕ)
  (cities_group_A : ℕ)
  (cities_group_B : ℕ)
  (cities_group_C : ℕ)
  (total_selected : ℕ)
  (C_subset_correct: total_cities = cities_group_A + cities_group_B + cities_group_C)
  (total_cities_correct: total_cities = 48)
  (cities_group_A_correct: cities_group_A = 8)
  (cities_group_B_correct: cities_group_B = 24)
  (total_selected_correct: total_selected = 12)
  : (total_selected * cities_group_C) / total_cities = 4 :=
by 
  sorry

end stratified_sampling_group_C_l561_561055


namespace solve_for_x_l561_561910

theorem solve_for_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 := by
  sorry

end solve_for_x_l561_561910


namespace number_of_special_permutations_l561_561771

noncomputable def count_special_permutations : ℕ :=
  (Nat.choose 12 6)

theorem number_of_special_permutations : count_special_permutations = 924 :=
  by
    sorry

end number_of_special_permutations_l561_561771


namespace fraction_zero_x_eq_2_l561_561283

theorem fraction_zero_x_eq_2 (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 :=
by sorry

end fraction_zero_x_eq_2_l561_561283


namespace integer_count_between_sqrt8_and_sqrt72_l561_561224

theorem integer_count_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ( ∀ x : ℤ, (⌊Real.sqrt 8⌋.to_nat + 1) ≤ x ∧ x ≤ ⌊Real.sqrt 72⌋ - 1 → x = 6 ) :=
by 
  -- Define the floor and ceiling functions
  have sqrt_8_ceil : ⌈Real.sqrt 8⌉ = 3 := sorry
  have sqrt_72_floor : ⌊Real.sqrt 72⌋ = 8 := sorry

  use 6,
  split,
  { refl },
  { intros x hx,
    sorry
  }

-- Additional supporting facts
lemma sqrt_8_approx : Real.sqrt 8 ≈ 2.83 := sorry
lemma sqrt_72_approx : Real.sqrt 72 ≈ 8.49 := sorry

end integer_count_between_sqrt8_and_sqrt72_l561_561224


namespace integer_root_of_integers_l561_561380

theorem integer_root_of_integers (x : ℝ) (h1 : (x^3 - x) ∈ ℤ) (h2 : (x^4 - x) ∈ ℤ) : x ∈ ℤ :=
by
  sorry

end integer_root_of_integers_l561_561380


namespace centrally_symmetric_polygon_l561_561639

theorem centrally_symmetric_polygon (P : Polygon) (O : Point) 
  (h_convex : Convex P) 
  (h_O_inside : O ∈ P)
  (h_line_equal_area : ∀ l : Line, O ∈ l → divides_into_equal_areas P l) :
  (CentrallySymmetric P O) := 
sorry

end centrally_symmetric_polygon_l561_561639


namespace isosceles_right_triangle_hypotenuse_l561_561019

theorem isosceles_right_triangle_hypotenuse :
  ∀ (a c : ℝ),
    (c * sqrt 2 + c = 8 + 8 * sqrt 2) →
    c = 8 :=
by
  intros a c h
  sorry

end isosceles_right_triangle_hypotenuse_l561_561019


namespace integer_part_m_eq_one_l561_561674

noncomputable def seq : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := (seq n) ^ 2 + (seq n)

noncomputable def m := ∑ i in (Finset.range 2016).map (λ n, n + 1), (1 / (seq i + 1))

theorem integer_part_m_eq_one : ⌊m⌋ = 1 := by
  sorry

end integer_part_m_eq_one_l561_561674


namespace adrian_water_amount_l561_561062

theorem adrian_water_amount
  (O S W : ℕ) 
  (h1 : S = 3 * O)
  (h2 : W = 5 * S)
  (h3 : O = 4) : W = 60 :=
by
  sorry

end adrian_water_amount_l561_561062


namespace ones_digit_sum_l561_561117

theorem ones_digit_sum (k: ℕ):
  (∑ i in finset.range 2014, (i ^ 2013) % 10) % 10 = 1 :=
begin
  -- We'll leave the proof as an exercise
  sorry
end

end ones_digit_sum_l561_561117


namespace john_spent_l561_561766

-- Given definitions from the conditions.
def total_time_in_hours := 4
def additional_minutes := 35
def break_time_per_break := 10
def number_of_breaks := 5
def cost_per_5_minutes := 0.75
def playing_cost (total_time_in_hours additional_minutes break_time_per_break number_of_breaks : ℕ) 
  (cost_per_5_minutes : ℝ) : ℝ :=
  let total_minutes := total_time_in_hours * 60 + additional_minutes
  let break_time := number_of_breaks * break_time_per_break
  let actual_playing_time := total_minutes - break_time
  let number_of_intervals := actual_playing_time / 5
  number_of_intervals * cost_per_5_minutes

-- Statement to be proved.
theorem john_spent (total_time_in_hours := 4) (additional_minutes := 35) (break_time_per_break := 10) 
  (number_of_breaks := 5) (cost_per_5_minutes := 0.75) :
  playing_cost total_time_in_hours additional_minutes break_time_per_break number_of_breaks cost_per_5_minutes = 33.75 := 
by
  sorry

end john_spent_l561_561766


namespace triangle_is_equilateral_l561_561144

-- Definitions of points P, P1, P2, O, A, and B.
variable {P P1 P2 O A B : Type*}

-- Angles and symmetry properties with respect to lines
variable [has_angle P O A] [has_angle P O B] [has_angle A O B]

-- Conditions
variable (angle_AOB : angle (A, O, B) = 30)
variable (sym_P1 : symmetric_with_respect_to P OB P1)
variable (sym_P2 : symmetric_with_respect_to P OA P2)

-- Lean 4 Statement to prove triangle P1 O P2 is equilateral
theorem triangle_is_equilateral (angle_condition : ∠ A O B = 30) 
  (P_inside_AOB : P ∈ interior_of ∠ A O B) 
  (P1_symmetric_OB : P1 = symmetric_point P OB)
  (P2_symmetric_OA : P2 = symmetric_point P OA) :
  ∠ P1 O P2 = 60 ∧
  distance O P1 = distance O P2 ∧
  distance O P1 = distance O P :=
sorry

end triangle_is_equilateral_l561_561144


namespace train_length_l561_561501

theorem train_length (bridge_length time_seconds speed_kmh : ℝ) (S : speed_kmh = 64) (T : time_seconds = 45) (B : bridge_length = 300) : 
  ∃ (train_length : ℝ), train_length = 500 := 
by
  -- Add your proof here 
  sorry

end train_length_l561_561501


namespace range_of_k_l561_561471

theorem range_of_k 
    (k : ℝ)
    (passes_through : ∃ (x y : ℝ), x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0 ∧ (x, y) = (-1, 0)) :
    k ∈ set.Iio (-1) ∪ set.Ioi 4 := 
by
  sorry

end range_of_k_l561_561471


namespace min_value_expression_l561_561568

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ m : ℝ, m = sqrt 39 ∧
  (∀ a b > 0, 
    (|6 * a - 4 * b| + |3 * (a + b * sqrt 3) + 2 * (a * sqrt 3 - b)|) / sqrt (a^2 + b^2) ≥ m) :=
sorry

end min_value_expression_l561_561568


namespace no_n_1989_digit_l561_561809

theorem no_n_1989_digit (n : ℕ) (h_digits : (nat.digits 10 n).length = 1989)
  (h_3_fives : (nat.digits 10 n).count 5 ≥ 3)
  (h_product_eq_sum : (nat.digits 10 n).prod = (nat.digits 10 n).sum) : false :=
sorry

end no_n_1989_digit_l561_561809


namespace line_through_two_points_eq_l561_561394

theorem line_through_two_points_eq (x1 y1 x2 y2 x y : ℝ) 
  : ((y - y1) * (x2 - x1) = (x - x1) * (y2 - y1))

end line_through_two_points_eq_l561_561394


namespace filtration_processes_required_l561_561038

theorem filtration_processes_required 
  (initial_impurity : ℝ)
  (reduction_factor : ℝ)
  (max_impurity : ℝ) :
  initial_impurity = 2 / 100 →
  reduction_factor = 1 / 2 →
  max_impurity = 1 / 1000 →
  ∃ n : ℕ, (initial_impurity * reduction_factor^n ≤ max_impurity) ∧ n = 5 :=
by
  intros h_initial_impurity h_reduction_factor h_max_impurity
  use 5
  split
  · sorry
  · rfl

end filtration_processes_required_l561_561038


namespace probability_joe_finds_bus_l561_561468
open Real

-- Definitions for the times and conditions
def arrives_between (t : ℝ) := 0 ≤ t ∧ t ≤ 90

-- The probability problem statement
theorem probability_joe_finds_bus :
  let y := ℝ in
  let x := ℝ in
  (arrives_between y) ∧ (arrives_between x) ∧ (0 ≤ y ∧ y ≤ 60) ∧ (y ≤ x ∧ x ≤ y + 30) →
  probability (x, y) to (x, y) satisfying (arrives_between y) ∧ (arrives_between x) ∧ (0 ≤ y ∧ y ≤ 60) ∧ (y ≤ x ∧ x ≤ y + 30) = 5 / 18 :=
sorry

end probability_joe_finds_bus_l561_561468


namespace count_integers_between_sqrt8_sqrt72_l561_561251

-- Define the square roots of 8 and 72 for reference
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt(8)
def minIntAboveSqrt8 : ℤ := Int.ceil sqrt8

-- Define the largest integer less than sqrt(72)
def maxIntBelowSqrt72 : ℤ := Int.floor sqrt72

-- State the main theorem to prove
theorem count_integers_between_sqrt8_sqrt72 :
  minIntAboveSqrt8 = 3 ∧ maxIntBelowSqrt72 = 8 → 
  (maxIntBelowSqrt72 - minIntAboveSqrt8 + 1) = 6 :=
by
  sorry

end count_integers_between_sqrt8_sqrt72_l561_561251


namespace a_plus_b_is_68_l561_561947

open Real

-- Define the problem's conditions as given

noncomputable def side_length_square : ℝ := 8
noncomputable def side_length_triangle : ℝ := 2
noncomputable def side_length_diamond : ℝ := 2 * sqrt 2
noncomputable def diameter_coin : ℝ := 1

-- Probability formula given in the problem
def probability (a b : ℝ) : ℝ := (1 / 196) * (a + b * sqrt 2 + π)

-- The goal is to find the integer values of a and b such that the sum a + b is an integer 68

theorem a_plus_b_is_68 :
  ∃ (a b : ℕ), probability (a : ℝ) (b : ℝ) = 1 / 196 * (32 + 36 * sqrt 2 + π) ∧ a + b = 68 :=
by
  sorry

end a_plus_b_is_68_l561_561947


namespace camp_organizer_needs_more_bottles_l561_561034

variable (cases : ℕ) (bottles_per_case : ℕ) (cases_bought : ℕ)
variable (children_group1 : ℕ) (children_group2 : ℕ) (children_group3 : ℕ)
variable (bottles_per_day : ℕ) (days : ℕ)

noncomputable def bottles_needed := 
  let children_group4 := (children_group1 + children_group2 + children_group3) / 2
  let total_children := children_group1 + children_group2 + children_group3 + children_group4
  let total_bottles_needed := total_children * bottles_per_day * days
  let total_bottles_purchased := cases_bought * bottles_per_case
  total_bottles_needed - total_bottles_purchased

theorem camp_organizer_needs_more_bottles :
  cases = 13 →
  bottles_per_case = 24 →
  cases_bought = 13 →
  children_group1 = 14 →
  children_group2 = 16 →
  children_group3 = 12 →
  bottles_per_day = 3 →
  days = 3 →
  bottles_needed cases bottles_per_case cases_bought children_group1 children_group2 children_group3 bottles_per_day days = 255 := by
  sorry

end camp_organizer_needs_more_bottles_l561_561034


namespace angle_DKC_eq_18_l561_561022

-- Define the points and angles
structure Trapezoid where
  A B C D K : Type
  (AD_parallel_BC : A ∥ C)
  (angle_ABC : ℝ := 108)
  (angle_ADC : ℝ := 54)
  (AK_eq_BC : AK = BC)

-- Define the main theorem to prove
theorem angle_DKC_eq_18 (trapezoid : Trapezoid) (angle_BKC_eq_27 : trapezoid.BKC = 27) : trapezoid.DKC = 18 := 
  sorry

end angle_DKC_eq_18_l561_561022


namespace pounds_of_oranges_l561_561928

noncomputable def price_of_pounds_oranges (E O : ℝ) (P : ℕ) : Prop :=
  let current_total_price := E
  let increased_total_price := 1.09 * E + 1.06 * (O * P)
  (increased_total_price - current_total_price) = 15

theorem pounds_of_oranges (E O : ℝ) (P : ℕ): 
  E = O * P ∧ 
  (price_of_pounds_oranges E O P) → 
  P = 100 := 
by
  sorry

end pounds_of_oranges_l561_561928


namespace probability_personA_not_personB_l561_561887

theorem probability_personA_not_personB :
  let n := Nat.choose 5 3
  let m := Nat.choose 1 1 * Nat.choose 3 2
  (m / n : ℚ) = 3 / 10 :=
by
  -- Proof omitted
  sorry

end probability_personA_not_personB_l561_561887


namespace time_difference_l561_561902

def Danny_time := 29
def Steve_time := 2 * Danny_time

def midway_time_Danny := Danny_time / 2
def midway_time_Steve := Steve_time / 2

theorem time_difference :
  (midway_time_Steve - midway_time_Danny) = 14.5 := by
  sorry

end time_difference_l561_561902


namespace taxi_fare_relationship_taxi_fare_relationship_simplified_l561_561287

variable (x : ℝ) (y : ℝ)

-- Conditions
def starting_fare : ℝ := 14
def additional_fare_per_km : ℝ := 2.4
def initial_distance : ℝ := 3
def total_distance (x : ℝ) := x
def total_fare (x : ℝ) (y : ℝ) := y
def distance_condition (x : ℝ) := x > 3

-- Theorem Statement
theorem taxi_fare_relationship (h : distance_condition x) :
  total_fare x y = additional_fare_per_km * (total_distance x - initial_distance) + starting_fare :=
by
  sorry

-- Simplified Theorem Statement
theorem taxi_fare_relationship_simplified (h : distance_condition x) :
  y = 2.4 * x + 6.8 :=
by
  sorry

end taxi_fare_relationship_taxi_fare_relationship_simplified_l561_561287


namespace count_integers_between_sqrt8_and_sqrt72_l561_561204

theorem count_integers_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℕ, 3 ≤ x ∧ x ≤ 8 → x ∈ finset.range (8 - 3 + 1) :=
by
  let a := real.sqrt 8
  let b := real.sqrt 72
  have ha : 3 = nat_ceil a := sorry
  have hb : 8 = nat_floor b := sorry
  use 6
  split
  {
    exact sorry
  }
  {
    intros x hx
    have h_valid : 3 ≤ x ∧ x ≤ 8 := hx
    rw finset.mem_range
    sorry
  }

end count_integers_between_sqrt8_and_sqrt72_l561_561204


namespace nearest_integer_pow_l561_561005

noncomputable def nearest_integer_to_power : ℤ := 
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_pow : nearest_integer_to_power = 7414 := 
  by
    unfold nearest_integer_to_power
    sorry -- Proof skipped

end nearest_integer_pow_l561_561005


namespace car_speed_l561_561920

variable (D : ℝ) (V : ℝ)

theorem car_speed
  (h1 : 1 / ((D / 3) / 80) + (D / 3) / 15 + (D / 3) / V = D / 30) :
  V = 35.625 :=
by 
  sorry

end car_speed_l561_561920


namespace triangle_angle_sum_l561_561867

theorem triangle_angle_sum (A : ℕ) (h1 : A = 55) (h2 : ∀ (B : ℕ), B = 2 * A) : (A + 2 * A = 165) :=
by
  sorry

end triangle_angle_sum_l561_561867


namespace derivative_of_even_function_is_odd_l561_561354

theorem derivative_of_even_function_is_odd 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f (-x) = f x)
  (g : ℝ → ℝ)
  (hg : ∀ x, g x = (f' x)) :
  ∀ x, g (-x) = - g x := 
  by
  sorry

end derivative_of_even_function_is_odd_l561_561354


namespace fiction_books_count_l561_561916

noncomputable def num_books := 
  ∃ n : ℕ, (∏ i in finset.range (3), (n - i))^2 = 36

theorem fiction_books_count : num_books := by
  use 3
  simp
  sorry

end fiction_books_count_l561_561916


namespace angle_5_measure_l561_561792

variable (m n k : Line)
variable (intersect : ∃ p, p ∈ m ∧ p ∈ n)
variable (transversal : is_transversal k m n)
variable (angle1 angle2 angle5 : ℝ)
variable (h1 : angle1 = (1/4) * angle2)
variable (supplementary : angle2 + angle5 = 180)

theorem angle_5_measure : angle5 = 36 := 
by 
  sorry

end angle_5_measure_l561_561792


namespace box_box_15_eq_60_l561_561089

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x => n % x = 0).sum id

def box (n : ℕ) : ℕ :=
  sum_of_factors n

theorem box_box_15_eq_60 : box (box 15) = 60 := by
  sorry

end box_box_15_eq_60_l561_561089


namespace cos_B_in_triangle_l561_561146

theorem cos_B_in_triangle
  (A B C a b c : ℝ)
  (h1 : Real.sin A = 2 * Real.sin C)
  (h2 : b^2 = a * c)
  (h3 : 0 < b)
  (h4 : 0 < c)
  (h5 : a = 2 * c)
  : Real.cos B = 3 / 4 := 
sorry

end cos_B_in_triangle_l561_561146


namespace evaluate_expression_l561_561559

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by sorry

end evaluate_expression_l561_561559


namespace ellipse_equation_max_area_triangle_OPQ_l561_561648

-- Definition of point A and other constants as needed
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := -2 }
def O : Point := { x := 0, y := 0 }

-- Eccentricity and slope conditions
def eccentricity := (Math.sqrt 3) / 2
def slope_AF := (2 * (Math.sqrt 3)) / 3

-- Proof to show the equation of the ellipse
theorem ellipse_equation :
  ∃ a b c : ℝ, 0 < b ∧ b < a ∧ (c / a = eccentricity) ∧ (c = Math.sqrt 3) ∧
    (A :: (a, c)) ∧ (slope_AF = (2 * Math.sqrt 3) / 3) ∧
    (∀ x y : ℝ, (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 ↔ (x ^ 2) / 4 + (y ^ 2) = 1) :=
sorry

-- Proof to find the equation of the line l when the area of triangle OPQ is maximized
theorem max_area_triangle_OPQ :
  ∃ k : ℝ, (k ^ 2 > 3 / 4) ∧ ∀ x y : ℝ, 
    d = 2 / Math.sqrt (k ^ 2 + 1) ∧
    |pq| = (4 * Math.sqrt (k ^ 2 + 1) * Math.sqrt (4 * k ^ 2 - 3)) / (4 * k ^ 2 + 1) ∧
    let t = Math.sqrt (4 * k ^ 2 - 3) in
    (t > 0) ∧
    (∀ t : ℝ, (t ^ 2 + 4) ≤ (4 / t) ^ 2 ↔ (t = 2)) ∧
    ∃ l_eq : String, 
    (l_eq = format!"y = {(Math.sqrt 7) / 2}x - 2" ∨ l_eq = format!"y = {-((Math.sqrt 7) / 2)}x - 2") :=
sorry

end ellipse_equation_max_area_triangle_OPQ_l561_561648


namespace cos_squared_diff_tan_l561_561164

theorem cos_squared_diff_tan (α : ℝ) (h : Real.tan α = 3) :
  Real.cos (α + π/4) ^ 2 - Real.cos (α - π/4) ^ 2 = -3 / 5 :=
by
  sorry

end cos_squared_diff_tan_l561_561164


namespace magnitude_z_plus_2_l561_561637

noncomputable def z : ℂ := (1 + Complex.I) / Complex.I
noncomputable def w : ℂ := z + 2
def magnitude_w : ℝ := Complex.abs w

theorem magnitude_z_plus_2 :
  magnitude_w = Real.sqrt 10 :=
sorry

end magnitude_z_plus_2_l561_561637


namespace triangle_classification_triangle_classification_right_triangle_classification_obtuse_l561_561810

def triangle_nature (a b c R : ℝ) : String :=
  if a^2 + b^2 + c^2 - 8 * R^2 > 0 then "acute"
  else if a^2 + b^2 + c^2 - 8 * R^2 = 0 then "right"
  else "obtuse"

theorem triangle_classification (a b c R : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : c = max a (max b c)):
  a^2 + b^2 + c^2 - 8 * R^2 > 0 →
  triangle_nature a b c R = "acute" :=
sorry

theorem triangle_classification_right (a b c R : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : c = max a (max b c)):
  a^2 + b^2 + c^2 - 8 * R^2 = 0 →
  triangle_nature a b c R = "right" :=
sorry

theorem triangle_classification_obtuse (a b c R : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) (h4 : c = max a (max b c)):
  a^2 + b^2 + c^2 - 8 * R^2 < 0 →
  triangle_nature a b c R = "obtuse" :=
sorry

end triangle_classification_triangle_classification_right_triangle_classification_obtuse_l561_561810


namespace florist_sold_16_roses_l561_561475

-- Definitions for initial and final states
def initial_roses : ℕ := 37
def picked_roses : ℕ := 19
def final_roses : ℕ := 40

-- Defining the variable for number of roses sold
variable (x : ℕ)

-- The statement to prove
theorem florist_sold_16_roses
  (h : initial_roses - x + picked_roses = final_roses) : x = 16 := 
by
  -- Placeholder for proof
  sorry

end florist_sold_16_roses_l561_561475


namespace solution_set_inequality_min_value_expr_l561_561179

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Prove the solution set for the inequality f(x) ≤ 6
theorem solution_set_inequality : { x : ℝ | x ≥ -1 ∧ f x ≤ 6 } = { x : ℝ | -1 ≤ x ∧ x ≤ 4 } := by
  sorry

-- Find the minimum value of f(x)
noncomputable def min_f_value : ℝ := Inf (f '' { x : ℝ | x ≥ -1 })

-- Hypothesis: If the minimum value of f(x) is n and 2n a b = a + 2b, find the minimum value of 2a + b
theorem min_value_expr (n a b : ℝ) (h1 : min_f_value = n) (h2 : 2 * n * a * b = a + 2 * b) (h3 : 0 < a) (h4 : 0 < b) : 
  2 * a + b ≥ 9 / 8 := by
  sorry

end solution_set_inequality_min_value_expr_l561_561179


namespace minimum_sum_is_162_l561_561326

open Finset

-- Define permutations a_i, b_i, c_i of the set {1, ..., 6}
variable {a b c : Fin 6 → ℕ}
variable (h1 : ∀ i, a i ∈ {1, 2, 3, 4, 5, 6})
variable (h2 : ∀ i, b i ∈ {1, 2, 3, 4, 5, 6})
variable (h3 : ∀ i, c i ∈ {1, 2, 3, 4, 5, 6})

-- Define that they are permutations
variable (ha : ∀ j ∈ {1, 2, 3, 4, 5, 6}, ∃ i, a i = j)
variable (hb : ∀ j ∈ {1, 2, 3, 4, 5, 6}, ∃ i, b i = j)
variable (hc : ∀ j ∈ {1, 2, 3, 4, 5, 6}, ∃ i, c i = j)

theorem minimum_sum_is_162 :
  ∃ (i : Fin 6 → ℕ), (∀ i, a i ∈ {1, 2, 3, 4, 5, 6}) →
  (∀ i, b i ∈ {1, 2, 3, 4, 5, 6}) →
  (∀ i, c i ∈ {1, 2, 3, 4, 5, 6}) →
  (∀ j ∈ {1, 2, 3, 4, 5, 6}, ∃ i, a i = j) →
  (∀ j ∈ {1, 2, 3, 4, 5, 6}, ∃ i, b i = j) →
  (∀ j ∈ {1, 2, 3, 4, 5, 6}, ∃ i, c i = j) →
  ∑ i, a i * b i * c i = 162 :=
by
  sorry

end minimum_sum_is_162_l561_561326


namespace interval_integer_count_l561_561231

-- Define the problem conditions
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt8
def lower_bound : Int := ceil sqrt8

-- Define the largest integer less than sqrt72
def upper_bound : Int := floor sqrt72

-- Prove the number of integers between sqrt8 and sqrt72 is 6
theorem interval_integer_count : 
    (upper_bound - lower_bound + 1) = 6 := 
by
  -- Steps and proofs would go here, but we use sorry to skip the proof for now
  sorry

end interval_integer_count_l561_561231


namespace problem_statement_l561_561262

-- Define the statement for positive integers m and n
def div_equiv (m n : ℕ) : Prop :=
  19 ∣ (11 * m + 2 * n) ↔ 19 ∣ (18 * m + 5 * n)

-- The final theorem statement
theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : div_equiv m n :=
by
  sorry

end problem_statement_l561_561262


namespace first_player_win_boards_l561_561435

-- Define what it means for a player to guarantee a win
def first_player_guarantees_win (n m : ℕ) : Prop :=
  ¬(n % 2 = 1 ∧ m % 2 = 1)

-- The main theorem that matches the math proof problem
theorem first_player_win_boards : (first_player_guarantees_win 6 7) ∧
                                  (first_player_guarantees_win 6 8) ∧
                                  (first_player_guarantees_win 7 8) ∧
                                  (first_player_guarantees_win 8 8) ∧
                                  ¬(first_player_guarantees_win 7 7) := 
by 
sorry

end first_player_win_boards_l561_561435


namespace C_share_correct_l561_561899

noncomputable def C_share (B_invest: ℝ) (total_profit: ℝ) : ℝ :=
  let A_invest := 3 * B_invest
  let C_invest := (3 * B_invest) * (3/2)
  let total_invest := (3 * B_invest + B_invest + C_invest)
  (C_invest / total_invest) * total_profit

theorem C_share_correct (B_invest total_profit: ℝ) 
  (hA : ∀ x: ℝ, A_invest = 3 * x)
  (hC : ∀ x: ℝ, C_invest = (3 * x) * (3/2)) :
  C_share B_invest 12375 = 6551.47 :=
by
  sorry

end C_share_correct_l561_561899


namespace lunks_for_apples_l561_561697

noncomputable def lunks_per_kunks := 7 / 4
noncomputable def kunks_per_apples := 3 / 5
def apples_needed := 24
noncomputable def kunks_needed_for_apples := (kunks_per_apples * apples_needed).ceil
noncomputable def lunks_needed := (lunks_per_kunks * kunks_needed_for_apples).ceil

theorem lunks_for_apples :
  lunks_needed = 27 :=
sorry

end lunks_for_apples_l561_561697


namespace divisors_of_n_squared_l561_561267

-- Definition for a number having exactly 4 divisors
def has_four_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^3

-- Theorem statement
theorem divisors_of_n_squared (n : ℕ) (h : has_four_divisors n) : 
  Nat.divisors_count (n^2) = 7 :=
by
  sorry

end divisors_of_n_squared_l561_561267


namespace average_of_other_four_points_l561_561384

theorem average_of_other_four_points (d : Fin 5 → ℝ) 
  (h_avg : (∑ i, d i) / 5 = 81) (h_one : ∃ i, d i = 85) : 
  (∑ i in {x // x ≠ (classical.some h_one)}, d i) / 4 = 80 := 
by
  sorry

end average_of_other_four_points_l561_561384


namespace lunks_for_apples_l561_561688

theorem lunks_for_apples : ∀ (lun_to_kun : ℕ) (num_lun : ℕ) (num_kun : ℕ) (kun_to_app : ℕ) (num_kun2 : ℕ) (num_app : ℕ),
  lun_to_kun = 7 ∧ num_kun = 4 ∧ kun_to_app = 3 ∧ num_kun2 = 5 ∧ num_app = 24 → 
  ((num_app * kun_to_app * num_lun / (num_kun2 * lun_to_kun)) ≤ 27) :=
by
  intros lun_to_kun num_lun num_kun kun_to_app num_kun2 num_app
  assume h_conditions
  sorry

end lunks_for_apples_l561_561688


namespace max_negative_numbers_min_distinct_bottom_row_l561_561021

-- Define the condition under which numbers in a column are related as one being the square of the other.
def column_relation (a b : ℝ) : Prop := a = b^2 ∨ b = a^2

-- Define the maximum number of negative numbers in the table.
theorem max_negative_numbers {t : ℕ} (h_t : t = 35) :
  ∃ m, m = 35 ∧ (∀ over_negative_count < 35, ∃ negative_numbers : ℕ, table : list (list ℝ), by
    ∃ table, sorry) :=
begin
  use 35,
  split,
  {
    exact rfl,
  },
  sorry,
end

-- Define the minimum number of distinct numbers in the bottom row.
theorem min_distinct_bottom_row {r : ℕ} (h_r : r = 35) :
  ∃ n, n = 12 ∧ (∀ over_count > 12, ∃ lower_row : list ℝ, table : list (list ℝ), by
    ∃ table, sorry) :=
begin
  use 12,
  split,
  {
    exact rfl,
  },
  sorry,
end

end max_negative_numbers_min_distinct_bottom_row_l561_561021


namespace tan_alpha_value_l561_561163

theorem tan_alpha_value (α : Real) 
  (h : (sin α + 2 * cos α) / (5 * cos α - sin α) = 5 / 16) : 
  tan α = -1 / 3 :=
by
  sorry

end tan_alpha_value_l561_561163


namespace simplify_fraction_l561_561376

theorem simplify_fraction :
  (3 : ℝ) / (2 * real.sqrt 50 + 3 * real.sqrt 8 + real.sqrt 18)
  = 3 * real.sqrt 2 / 38 :=
by
  sorry

end simplify_fraction_l561_561376


namespace pen_arrangements_l561_561801

theorem pen_arrangements : 
  let n := 7 in
  let pens := {1, 2, 3, 4, 5, 6, 7} in
  let holder_A : set ℕ := {} in
  let holder_B : set ℕ := {} in
  let arrangements := (∃ A B : set ℕ, A ∪ B = pens ∧ A ∩ B = ∅ ∧ 2 ≤ A.card ∧ 2 ≤ B.card ∧ arrangement n A ∧ arrangement n B) in
  arrangements = 112 :=
  sorry

end pen_arrangements_l561_561801


namespace trajectory_of_C_l561_561744

noncomputable def Point := (ℝ × ℝ)

def O : Point := (0, 0)
def A : Point := (2, 1)
def B : Point := (-1, -2)

def C (s t : ℝ) (hst : s + t = 1) : Point := 
  let (ax, ay) := A
  let (bx, by) := B
  (s * ax + t * bx, s * ay + t * by)

theorem trajectory_of_C (s t : ℝ) (hst : s + t = 1) :
  ∃ x y : ℝ, C s t hst = (x, y) ∧ (x - y - 1 = 0) := sorry

end trajectory_of_C_l561_561744


namespace complement_union_eq_l561_561143

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {1,3,5,7}
def B : Set ℕ := {2,4,5}

theorem complement_union_eq : (U \ (A ∪ B)) = {6,8} := by
  sorry

end complement_union_eq_l561_561143


namespace count_integers_between_sqrt8_and_sqrt72_l561_561206

theorem count_integers_between_sqrt8_and_sqrt72 : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℕ, 3 ≤ x ∧ x ≤ 8 → x ∈ finset.range (8 - 3 + 1) :=
by
  let a := real.sqrt 8
  let b := real.sqrt 72
  have ha : 3 = nat_ceil a := sorry
  have hb : 8 = nat_floor b := sorry
  use 6
  split
  {
    exact sorry
  }
  {
    intros x hx
    have h_valid : 3 ≤ x ∧ x ≤ 8 := hx
    rw finset.mem_range
    sorry
  }

end count_integers_between_sqrt8_and_sqrt72_l561_561206


namespace interval_integer_count_l561_561230

-- Define the problem conditions
def sqrt8 : Real := Real.sqrt 8
def sqrt72 : Real := Real.sqrt 72

-- Define the smallest integer greater than sqrt8
def lower_bound : Int := ceil sqrt8

-- Define the largest integer less than sqrt72
def upper_bound : Int := floor sqrt72

-- Prove the number of integers between sqrt8 and sqrt72 is 6
theorem interval_integer_count : 
    (upper_bound - lower_bound + 1) = 6 := 
by
  -- Steps and proofs would go here, but we use sorry to skip the proof for now
  sorry

end interval_integer_count_l561_561230


namespace number_of_elements_in_P_l561_561160

open Set

noncomputable def M : Set ℕ := {1, 2}
noncomputable def N : Set ℕ := {3, 4, 5}
noncomputable def P : Set ℕ := {x | ∃ a b, a ∈ M ∧ b ∈ N ∧ x = a + b}

theorem number_of_elements_in_P : ∃ n : ℕ, n = 4 ∧ ∀ x : ℕ, x ∈ P → x ∈ {4, 5, 6, 7} :=
by
sorry

end number_of_elements_in_P_l561_561160


namespace perpendicular_lines_through_vertex_l561_561880

variables {Point Line : Type}

-- Definitions and conditions
def is_perpendicular (l1 l2 : Line) : Prop := sorry
def drop_perpendicular (a : Point) (l : Line) : Point := sorry
def passes_through (l : Line) (p : Point) : Prop := sorry

variables (C A B : Point)
variables (l1 l2 : Line)
variables (A1 A2 B1 B2 : Point)
variables (angle_90 : ∀ p : Point, ∃ l1 l2 : Line, is_perpendicular l1 l2 ∧ passes_through l1 p ∧ passes_through l2 p)

-- New statement
theorem perpendicular_lines_through_vertex
  (right_angle_at_C : ∃ l1 l2, is_perpendicular l1 l2 ∧ passes_through l1 C ∧ passes_through l2 C)
  (A1_perp : A1 = drop_perpendicular A l1)
  (A2_perp : A2 = drop_perpendicular A l2)
  (B1_perp : B1 = drop_perpendicular B l1)
  (B2_perp : B2 = drop_perpendicular B l2) :
  is_perpendicular (line_through A1 A2) (line_through B1 B2) :=
sorry

end perpendicular_lines_through_vertex_l561_561880


namespace problem1_part1_problem1_part2_l561_561663

theorem problem1_part1 (m : ℝ) (h1: cos α = m / 5) (h2: P = (m, -m - 1)) : m^2 + m - 12 = 0 := 
sorry

theorem problem1_part2 (m : ℝ) (h : m > 0) (h_solved : m = 3) : 
  (sin (3 * π + α) * cos (3 * π / 2 - α)) / 
  (cos (α - π) * sin (π / 2 + α)) = - 16 / 9 :=
by
  have h_cos : cos α = 3 / 5 := sorry
  have h_sin : sin α = -4 / 5 := sorry
  sorry

end problem1_part1_problem1_part2_l561_561663


namespace basketball_lineup_selection_l561_561467

theorem basketball_lineup_selection:
  ∀ (n k: ℕ), n = 12 -> k = 5 -> 
  (∑ (captain ∈ finset.range n), (nat.choose (n - 1) (k - 1))) = 3960 :=
by
  sorry

end basketball_lineup_selection_l561_561467


namespace tangent_perpendicular_to_line_l561_561182

theorem tangent_perpendicular_to_line (e : ℝ) (m : ℝ) :
  (∃ x : ℝ, has_deriv_at (λ x, exp x - m*x + 1) (exp x - m) x ∧ (exp x - m) * e = -1) → m > 1 / e :=
by
  sorry

end tangent_perpendicular_to_line_l561_561182


namespace points_left_of_origin_l561_561302

theorem points_left_of_origin : 
  let p1 := -( -8 )
  let p2 := (-1) ^ 2023
  let p3 := - (3 ^ 2)
  let p4 := -1 - 11
  let p5 := -(2 / 5)
  4 = ([p1, p2, p3, p4, p5].filter (λ x => x < 0)).length := 
by
  let p1 := -( -8 )
  let p2 := (-1) ^ 2023
  let p3 := - (3 ^ 2)
  let p4 := -1 - 11
  let p5 := -(2 / 5)
  have h1: p1 = 8 := by sorry
  have h2: p2 = -1 := by sorry
  have h3: p3 = -9 := by sorry
  have h4: p4 = -12 := by sorry
  have h5: p5 = -(2 / 5) := by sorry
  have h_points: [8, -1, -9, -12, -(2 / 5)] = [p1, p2, p3, p4, p5] := 
    by simp [h1, h2, h3, h4, h5]
  have h_left_points: [8, -1, -9, -12, -(2 / 5)].filter 
    (λ x => x < 0) = [-1, -9, -12, -(2 / 5)] := 
    by simp
  have h_len: [-1, -9, -12, -(2 / 5)].length = 4 := by simp
  simp [h_points, h_left_points, h_len]

end points_left_of_origin_l561_561302


namespace count_integers_between_sqrts_l561_561235

theorem count_integers_between_sqrts : 
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  (upper_bound - lower_bound + 1) = 6 :=
by
  let lower_bound := Nat.ceil (Real.sqrt 8)
  let upper_bound := Nat.floor (Real.sqrt 72)
  calc (upper_bound - lower_bound + 1) = 6 : sorry

end count_integers_between_sqrts_l561_561235


namespace trig_relationship_l561_561630

theorem trig_relationship : 
  let a := Real.sin (145 * Real.pi / 180)
  let b := Real.cos (52 * Real.pi / 180)
  let c := Real.tan (47 * Real.pi / 180)
  a < b ∧ b < c :=
by 
  sorry

end trig_relationship_l561_561630
