import Mathlib

namespace simplest_fraction_coprime_l154_154436

-- Let a simplest fraction be defined as follows:
def simplest_fraction (a b : ℕ) : Prop := Nat.coprime a b

-- The theorem statement we want to prove:
theorem simplest_fraction_coprime (a b : ℕ) (h : simplest_fraction a b) : Nat.coprime a b :=
by
  sorry

end simplest_fraction_coprime_l154_154436


namespace sufficient_m_value_l154_154768

theorem sufficient_m_value (m : ℕ) : 
  ((8 = m ∨ 9 = m) → 
  (m^2 + m^4 + m^6 + m^8 ≥ 6^3 + 6^5 + 6^7 + 6^9)) := 
by 
  sorry

end sufficient_m_value_l154_154768


namespace prob_all_heads_or_tails_five_coins_l154_154204

theorem prob_all_heads_or_tails_five_coins :
  (number_of_favorable_outcomes : ℕ) (total_number_of_outcomes : ℕ) (probability : ℚ) 
  (h_favorable : number_of_favorable_outcomes = 2)
  (h_total : total_number_of_outcomes = 32)
  (h_probability : probability = number_of_favorable_outcomes / total_number_of_outcomes) :
  probability = 1 / 16 :=
by
  sorry

end prob_all_heads_or_tails_five_coins_l154_154204


namespace bags_have_same_number_of_balls_l154_154455

-- Given definition that there are 7*54 balls and 2n + 1 bags
variables (n : ℕ)
def total_balls := 7 * 54
def num_bags := 2 * n + 1

-- Given condition that removing any one bag allows partitioning the remainder
def partition_condition (a : fin num_bags → ℕ) :=
  ∀ i, ∃ b : fin (2 * n) → ℕ, 
    multiset.card (multiset.attach b) = n ∧
    multiset.sum (multiset.attach b) = multiset.sum (multiset.attach b.codediff)

-- The target statement to prove that all bags have the same number of balls
theorem bags_have_same_number_of_balls :
  ∀ (a : fin num_bags → ℕ), 
  multiset.sum (multiset.attach a) = total_balls ∧ partition_condition a → 
  ∀ i j, a i = a j :=
sorry

end bags_have_same_number_of_balls_l154_154455


namespace product_geq_power_l154_154109

theorem product_geq_power (n : ℕ) (x : Fin (n+2) → ℝ)
  (h_pos : ∀ i, 0 < x i)
  (h_sum : (∑ i, 1 / (1 + x i)) = 1) : 
  (∏ i, x i) ≥ n^(n+1) := 
by
  sorry

end product_geq_power_l154_154109


namespace evaluate_Q_l154_154607

theorem evaluate_Q (n : ℕ) (h : n = 2007) :
  (∏ k in Finset.range (n + 1), (1 - (1 / (k + 2))) : ℚ) = 1 / (n + 1) :=
by
  sorry

end evaluate_Q_l154_154607


namespace solution_range_l154_154580

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l154_154580


namespace project_inflation_cost_increase_l154_154513

theorem project_inflation_cost_increase :
  let original_lumber_cost := 450
  let original_nails_cost := 30
  let original_fabric_cost := 80
  let lumber_inflation := 0.2
  let nails_inflation := 0.1
  let fabric_inflation := 0.05
  
  let new_lumber_cost := original_lumber_cost * (1 + lumber_inflation)
  let new_nails_cost := original_nails_cost * (1 + nails_inflation)
  let new_fabric_cost := original_fabric_cost * (1 + fabric_inflation)
  
  let total_increased_cost := (new_lumber_cost - original_lumber_cost) 
                            + (new_nails_cost - original_nails_cost) 
                            + (new_fabric_cost - original_fabric_cost)
  total_increased_cost = 97 := sorry

end project_inflation_cost_increase_l154_154513


namespace even_and_monotonically_decreasing_D_l154_154547

-- Definitions of the functions as Lean expressions
def fA (x : ℝ) : ℝ := 1 / x
def fB (x : ℝ) : ℝ := Real.exp (-x)
def fC (x : ℝ) : ℝ := 1 - x^2
def fD (x : ℝ) : ℝ := Real.log (Real.abs x)

-- The problem statement to prove
theorem even_and_monotonically_decreasing_D :
  (∀ x, fD (-x) = fD x) ∧ (∀ x y, x < y ∧ y < 0 → fD x > fD y) :=
by
  -- Proof is omitted
  sorry

end even_and_monotonically_decreasing_D_l154_154547


namespace expected_value_two_flips_l154_154876

open Probability

-- Define the biased coin probabilities
def biased_coin : Pmf bool :=
Pmf.ofProbFn (λ b, if b then 2/5 else 3/5)

-- Define the winnings for heads and tails
def winnings (b : bool) : ℚ :=
if b then 4 else -1

-- Expected value of winnings for a single flip
def E₁ : ℚ :=
Pmf.esum (biased_coin.bind (λ b, Pmf.return (winnings b)))

-- Expected value of winnings after two independent flips
def E₂ := 2 * E₁

-- Statement of the proof problem
theorem expected_value_two_flips : E₂ = 2 :=
by 
  rw [←mul_assoc]
  have hE₁ : E₁ = 1 := sorry
  rw [hE₁]
  norm_num

end expected_value_two_flips_l154_154876


namespace sum_of_reciprocal_squares_l154_154789

theorem sum_of_reciprocal_squares (n : ℕ) (h : n = 2015) :
  1 + ∑ k in finset.range(1, n+1), (1 / (k^2 : ℝ)) < ↑(2 * n - 1) / ↑n := by
  sorry

end sum_of_reciprocal_squares_l154_154789


namespace handrail_length_correct_l154_154906

-- Define the conditions
def turn_angle_degrees : ℝ := 315
def height : ℝ := 12
def radius : ℝ := 3

-- Define the arc length calculation based on given angle, height, and radius
def arc_length : ℝ := (turn_angle_degrees / 360) * 2 * Real.pi * radius

-- Calculate the handrail length
def handrail_length : ℝ := Real.sqrt (height^2 + arc_length^2)

-- Statement to be proven
theorem handrail_length_correct : handrail_length ≈ 20.4 :=
begin
  sorry,
end

end handrail_length_correct_l154_154906


namespace different_nat_numbers_impossible_equal_sums_l154_154738

theorem different_nat_numbers_impossible_equal_sums (a b : ℕ) (h_diff : a ≠ b) (h_eq_sums : ∀ (x y z : ℕ), x + y + z = a + y + z → x = a) : False :=
by
  have contra : a = b := by
    apply @eq.trans (x + y + z) (a + y + z) y
    apply h_eq_sums
    simp
  contradiction
  sorry

end different_nat_numbers_impossible_equal_sums_l154_154738


namespace min_value_ineq_least_3_l154_154650

noncomputable def min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : ℝ :=
  1 / (x + y) + (x + y) / z

theorem min_value_ineq_least_3 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  min_value_ineq x y z h1 h2 h3 h4 ≥ 3 :=
sorry

end min_value_ineq_least_3_l154_154650


namespace count_arithmetic_sequence_sets_l154_154925

open Nat Set

theorem count_arithmetic_sequence_sets : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let arithmetic_sequence (s : Set ℕ) := ∃ (a d : ℕ), 
    ∀ (k : ℕ), k < 4 → a + k * d ∈ s
  (∃ s : Set ℕ, s ⊆ digits ∧ s.card = 4 ∧ arithmetic_sequence s) :=
  sorry

end count_arithmetic_sequence_sets_l154_154925


namespace range_of_k_l154_154983

noncomputable def f (x : Real) : Real :=
  2 * Real.log x - Real.log (x - 1)

noncomputable def g (k : Real) (x : Real) : Real :=
  Real.log (k^(Real.exp x) - Real.exp(2 * x) + k)

theorem range_of_k (k : Real) :
  (∀ x₁, 1 < x₁ → ∃ x₂, x₂ ≤ 0 ∧ f x₁ + g k x₂ > 0) ↔ (sqrt 5 - 2 < k) := 
sorry

end range_of_k_l154_154983


namespace smallest_positive_period_monotonic_decreasing_interval_l154_154984

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sin x * Real.cos x

theorem smallest_positive_period (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ T > 0 → T = Real.pi :=
by
  sorry

theorem monotonic_decreasing_interval :
  (∀ x, x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8) → ∃ k : ℤ, 
     f (x + k * π) = f x ∧ f (x + k * π) ≤ f (x + (k + 1) * π)) :=
by
  sorry

end smallest_positive_period_monotonic_decreasing_interval_l154_154984


namespace select_representatives_l154_154405

theorem select_representatives :
  let boys := 5 
  let girls := 3 
  let choose := Nat.choose in
  (choose boys 2 * choose girls 1) + (choose boys 1 * choose girls 2) = 45 :=
by
  intros boys girls choose
  have h1 : choose boys 2 = 10 := Nat.choose_eq 5 2
  have h2 : choose girls 1 = 3 := Nat.choose_eq 3 1
  have h3 : choose boys 1 = 5 := Nat.choose_eq 5 1
  have h4 : choose girls 2 = 3 := Nat.choose_eq 3 2
  calc
    choose boys 2 * choose girls 1 + choose boys 1 * choose girls 2
      = 10 * 3 + 5 * 3 : by rw [h1, h2, h3, h4]
      = 30 + 15 : by norm_num
      = 45 : by norm_num

end select_representatives_l154_154405


namespace orthocenter_on_line_AD_l154_154920

theorem orthocenter_on_line_AD
  (A B C D E H O1 O2 O3 : euclidean_space ℝ (fin 2))
  (hO1 : is_circumcenter A B E O1)
  (hO2 : is_circumcenter A D E O2)
  (hO3 : is_circumcenter C D E O3)
  (hH : is_orthocenter O1 O2 O3 H) : 
  collinear_ ℝ ({A, D, H}) :=
sorry

end orthocenter_on_line_AD_l154_154920


namespace coefficient_of_x_squared_l154_154331

theorem coefficient_of_x_squared :
  let f := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4
  in (coeff x^2 (f)) = 9 :=
by
  sorry

end coefficient_of_x_squared_l154_154331


namespace probability_heads_9_or_more_12_flips_l154_154026

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154026


namespace sum_of_first_44_terms_l154_154443

def sequence : ℕ → ℕ
| 0       := 1
| (n + 1) := if (sum_up_to_pos (λ k, if k > n then 1 else 3) = 3) then 1 else 3

theorem sum_of_first_44_terms :
  (finset.sum (finset.range 44) sequence) = 116 :=
by
  sorry

end sum_of_first_44_terms_l154_154443


namespace six_rational_right_triangles_same_perimeter_l154_154169

theorem six_rational_right_triangles_same_perimeter :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ a₄ b₄ c₄ a₅ b₅ c₅ a₆ b₆ c₆ : ℕ),
    a₁^2 + b₁^2 = c₁^2 ∧ a₂^2 + b₂^2 = c₂^2 ∧ a₃^2 + b₃^2 = c₃^2 ∧
    a₄^2 + b₄^2 = c₄^2 ∧ a₅^2 + b₅^2 = c₅^2 ∧ a₆^2 + b₆^2 = c₆^2 ∧
    a₁ + b₁ + c₁ = 720 ∧ a₂ + b₂ + c₂ = 720 ∧ a₃ + b₃ + c₃ = 720 ∧
    a₄ + b₄ + c₄ = 720 ∧ a₅ + b₅ + c₅ = 720 ∧ a₆ + b₆ + c₆ = 720 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧ (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₄, b₄, c₄) ∧ (a₁, b₁, c₁) ≠ (a₅, b₅, c₅) ∧
    (a₁, b₁, c₁) ≠ (a₆, b₆, c₆) ∧ (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₂, b₂, c₂) ≠ (a₄, b₄, c₄) ∧ (a₂, b₂, c₂) ≠ (a₅, b₅, c₅) ∧
    (a₂, b₂, c₂) ≠ (a₆, b₆, c₆) ∧ (a₃, b₃, c₃) ≠ (a₄, b₄, c₄) ∧
    (a₃, b₃, c₃) ≠ (a₅, b₅, c₅) ∧ (a₃, b₃, c₃) ≠ (a₆, b₆, c₆) ∧
    (a₄, b₄, c₄) ≠ (a₅, b₅, c₅) ∧ (a₄, b₄, c₄) ≠ (a₆, b₆, c₆) ∧
    (a₅, b₅, c₅) ≠ (a₆, b₆, c₆) :=
sorry

end six_rational_right_triangles_same_perimeter_l154_154169


namespace sum_of_segments_l154_154408

theorem sum_of_segments (a b c : ℝ) (h : a = 10) :

  let x_n (n: ℕ) := (5 * n / 4) in
  (∑ n in (Finset.range 7).succ, x_n(n)) = 35 :=
by
  sorry

end sum_of_segments_l154_154408


namespace constant_term_expansion_l154_154727

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem constant_term_expansion : (x : ℂ) (h : x ≠ 0) : 
  let general_term := λ r, (2^r) * (binomial_coeff 6 r) * (x^(6 - (3 * r / 2))) in 
  general_term 4 = 240 := 
by 
  sorry

end constant_term_expansion_l154_154727


namespace graph_shift_sine_l154_154842

theorem graph_shift_sine :
  ∀ x : ℝ, y = sin (2 * x + π / 4) ↔ y = sin (2 * (x + π / 8)) :=
by
  sorry

end graph_shift_sine_l154_154842


namespace cylinder_height_l154_154887

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3

theorem cylinder_height (d_cylinder d_sphere h_cylinder : ℝ) :
  (d_cylinder = 6) →
  (d_sphere = 3) →
  cylinder_volume (d_cylinder / 2) h_cylinder = sphere_volume (d_sphere / 2) →
  h_cylinder = 0.5 :=
by
  intros h1 h2 h3
  sorry

end cylinder_height_l154_154887


namespace unique_coloring_l154_154432

-- Define the colors
inductive Color
| red
| yellow
| green

open Color

-- Define the conditions
structure HexagonColoring :=
(color : Hexagon → Color)
(no_adjacent_same_color : ∀ (h1 h2 : Hexagon), h1.adjacent h2 → color h1 ≠ color h2)
(left_of_R_is_red : ∀ (h : Hexagon), h.left_of_R → color h = red)
(R_is_green : ∀ (h : Hexagon), h.R_hexagon → color h = green)

-- Define the theorem
theorem unique_coloring : ∃! (hc : HexagonColoring), hc :=
by
  sorry

end unique_coloring_l154_154432


namespace manufacturer_not_fraudulent_l154_154119

theorem manufacturer_not_fraudulent (label_mass lower_bound upper_bound actual_mass : ℝ)
  (h_bound : lower_bound = label_mass - 3)
  (h_upper : upper_bound = label_mass + 3)
  (h_label : label_mass = 200)
  (h_actual : actual_mass = 198)
  (h_range : lower_bound ≤ actual_mass ∧ actual_mass ≤ upper_bound) :
  "not engaged in fraudulent behavior" :=
by {
  have h1 : lower_bound = 197, by rw [h_bound, h_label]; norm_num,
  have h2 : upper_bound = 203, by rw [h_upper, h_label]; norm_num,
  have h_check := h_range,
  rw [h1, h2] at h_check,
  simp only [le_refl, le_add_iff_nonneg_right, nonneg_add_iff_nonneg_and_le],
  exact "not engaged in fraudulent behavior",
}

end manufacturer_not_fraudulent_l154_154119


namespace number_of_people_in_first_group_l154_154702

variables (W : ℕ) -- amount of work
variables (P : ℕ) -- number of people in the first group
variables (work_rate_first_group : ℕ) (work_rate_second_group : ℕ)

-- First condition: P people can do 3W in 3 days -> work rate is W per day
def first_group_work_rate_condition (work_rate_first_group : ℕ) : Prop :=
  work_rate_first_group = W

-- Second condition: 4 people can do 4W in 3 days -> work rate is (4/3)W per day
def second_group_work_rate_condition (work_rate_second_group : ℕ) : Prop :=
  work_rate_second_group = (4 * W) / 3

-- Overall proof statement: P = 3
theorem number_of_people_in_first_group
  (h1 : first_group_work_rate_condition W work_rate_first_group)
  (h2 : second_group_work_rate_condition work_rate_second_group) : 
  P = 3 :=
by
  -- Proof goes here
  sorry

end number_of_people_in_first_group_l154_154702


namespace isosceles_triangle_BCE_l154_154334

variable (A B C D M E : Type*) [Trapezoid A B C D] 
          (mid_AD : M = midpoint A D)
          (E_on_BM : lies_on E (segment B M))
          (angle_eq : ∠ A D B = ∠ M A E = ∠ B M C)

theorem isosceles_triangle_BCE : isosceles_triangle B C E := by
  sorry

end isosceles_triangle_BCE_l154_154334


namespace base_7_is_good_number_l154_154764

def is_good_number (m: ℕ) : Prop :=
  ∃ (p: ℕ) (n: ℕ), Prime p ∧ n ≥ 2 ∧ m = p^n

theorem base_7_is_good_number : 
  ∀ b: ℕ, (is_good_number (b^2 - (2 * b + 3))) → b = 7 :=
by
  intro b h
  sorry

end base_7_is_good_number_l154_154764


namespace ab_eq_zero_implies_a_or_b_eq_zero_l154_154863

theorem ab_eq_zero_implies_a_or_b_eq_zero (a b : ℝ) (h : a * b = 0) : a = 0 ∨ b = 0 :=
by {
  have h1 : ¬ (a ≠ 0 ∧ b ≠ 0) := sorry,
  sorry
}

end ab_eq_zero_implies_a_or_b_eq_zero_l154_154863


namespace find_x_plus_y_l154_154693

theorem find_x_plus_y :
  ∀ (x y : ℝ), (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 :=
by
  intros x y h
  sorry

end find_x_plus_y_l154_154693


namespace intersection_of_M_and_N_l154_154353

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l154_154353


namespace sum_of_slopes_l154_154849

theorem sum_of_slopes (m1 m2 : ℝ) (b1 b2 : ℝ) (x y : ℝ) :
  (m1 = -5) →
  (b1 = 14) →
  (m2 = 3) →
  (b2 = -1) →
  (x = 15 / 8) →
  (y = 37 / 8) →
  (-5 * x + 14 = 3 * x - 1) →
  m1 + m2 = -2 :=
begin
  sorry,
end

end sum_of_slopes_l154_154849


namespace max_matrix_det_l154_154197

noncomputable def matrix_det (θ : ℝ) : ℝ :=
  by
    let M := ![
      ![1, 1, 1],
      ![1, 1 + Real.sin θ ^ 2, 1],
      ![1 + Real.cos θ ^ 2, 1, 1]
    ]
    exact Matrix.det M

theorem max_matrix_det : ∃ θ : ℝ, matrix_det θ = 3/4 :=
  sorry

end max_matrix_det_l154_154197


namespace correct_expression_l154_154148

theorem correct_expression : 
  (\sqrt[3](-27) = -3) ∧ ¬(\sqrt{16} = ± 4) ∧ ¬(±\sqrt{16} = 4) ∧ ¬(\sqrt{(-4)^2} = -4) :=
by
  -- rest of the proof here
  sorry

end correct_expression_l154_154148


namespace coffee_weekly_spending_l154_154385

-- Definitions of parameters based on given conditions.
def daily_cups : ℕ := 2
def ounces_per_cup : ℝ := 1.5
def days_per_week : ℕ := 7
def ounces_per_bag : ℝ := 10.5
def cost_per_bag : ℝ := 8
def weekly_milk_gallons : ℝ := 1 / 2
def cost_per_gallon_milk : ℝ := 4

-- Theorem statement of the proof problem.
theorem coffee_weekly_spending : 
  daily_cups * ounces_per_cup * days_per_week / ounces_per_bag * cost_per_bag +
  weekly_milk_gallons * cost_per_gallon_milk = 18 :=
by sorry

end coffee_weekly_spending_l154_154385


namespace shaded_area_l154_154919

theorem shaded_area (R : ℝ) (π : ℝ) (h1 : π * (R / 2)^2 * 2 = 1) : 
  (π * R^2 - (π * (R / 2)^2 * 2)) = 1 := 
by
  sorry

end shaded_area_l154_154919


namespace coin_flip_probability_l154_154011

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154011


namespace tetrahedron_inner_product_sum_l154_154498

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions for vectors a, b, c, d, and u
variables (a b c d u : V)

-- Conditions: a, b, c, and d are unit vectors
axiom h_a : ‖a‖ = 1
axiom h_b : ‖b‖ = 1
axiom h_c : ‖c‖ = 1
axiom h_d : ‖d‖ = 1

-- Condition: sum of vectors a, b, c, d is zero
axiom h_sum : a + b + c + d = 0

-- Conditions: specific inner products between different vectors
axiom h_ab : ⟪a, b⟫ = -1/3
axiom h_ac : ⟪a, c⟫ = -1/3
axiom h_ad : ⟪a, d⟫ = -1/3
axiom h_bc : ⟪b, c⟫ = -1/3
axiom h_bd : ⟪b, d⟫ = -1/3
axiom h_cd : ⟪c, d⟫ = -1/3

-- Arbitrary vector u definition
variable u : V 

-- The theorem to be proved
theorem tetrahedron_inner_product_sum :
  (⟪a, u⟫ • a) + (⟪b, u⟫ • b) + (⟪c, u⟫ • c) + (⟪d, u⟫ • d) = (4 / 3 : ℝ) • u :=
sorry

end tetrahedron_inner_product_sum_l154_154498


namespace union_complement_eq_l154_154779

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l154_154779


namespace last_remaining_number_is_1888_l154_154561

noncomputable def problem_last_remaining_number : ℕ :=
  let numbers := (List.range (1987 + 1)).tail!  -- All numbers from 1 to 1987
  let rec eliminate (numbers : List ℕ) (round : ℕ) : List ℕ :=
    if numbers.length <= 1 then numbers else
    let new_numbers := numbers.filter (λ x, (x - 1 - round) % (2 + round) != 0)
    eliminate new_numbers (round + 1)
  eliminate numbers 0

theorem last_remaining_number_is_1888 : problem_last_remaining_number = 1888 :=
by
  sorry

end last_remaining_number_is_1888_l154_154561


namespace right_angled_triangle_set_C_l154_154154

theorem right_angled_triangle_set_C : 
  ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by 
  use [3, 4, 5]
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end right_angled_triangle_set_C_l154_154154


namespace exists_complex_unit_det_bound_l154_154343
open Real

noncomputable theory

variables {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ)
  [Fact (2 ≤ n)]

theorem exists_complex_unit_det_bound (A B : Matrix (Fin n) (Fin n) ℝ)
  (Fact (2 ≤ n)) :
  ∃ (z : ℂ), abs z = 1 ∧ re (det (A.map coe + z • B.map coe)) ≥ det A + det B :=
sorry

end exists_complex_unit_det_bound_l154_154343


namespace smallest_integer_with_16_divisors_l154_154065

-- Define the condition for the number of divisors of an integer
def num_divisors (n : ℕ) : ℕ :=
  (n.factorization.map (λ p a, a + 1)).prod

-- Define the problem statement as a theorem
theorem smallest_integer_with_16_divisors : ∃ n : ℕ, num_divisors n = 16 ∧ (∀ m : ℕ, num_divisors m = 16 → n ≤ m) :=
by
  -- Placeholder to skip proof
  sorry

end smallest_integer_with_16_divisors_l154_154065


namespace smallest_integer_with_16_divisors_l154_154061

-- Define the condition for the number of divisors of an integer
def num_divisors (n : ℕ) : ℕ :=
  (n.factorization.map (λ p a, a + 1)).prod

-- Define the problem statement as a theorem
theorem smallest_integer_with_16_divisors : ∃ n : ℕ, num_divisors n = 16 ∧ (∀ m : ℕ, num_divisors m = 16 → n ≤ m) :=
by
  -- Placeholder to skip proof
  sorry

end smallest_integer_with_16_divisors_l154_154061


namespace candy_cost_l154_154291

theorem candy_cost (cost_per_gumdrop : ℕ) (num_gumdrops : ℕ) (h1 : cost_per_gumdrop = 8) (h2 : num_gumdrops = 28) : cost_per_gumdrop * num_gumdrops = 224 :=
by
  rw [h1, h2]
  norm_num

end candy_cost_l154_154291


namespace range_h_and_sum_l154_154179

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 3 * x^2)

theorem range_h_and_sum (a b : ℝ) (ha : a = 0) (hb : b = 3) :
  (∀ y, h y ∈ set.Ioc a b) ∧ (a + b = 3) :=
by
  sorry

end range_h_and_sum_l154_154179


namespace total_annual_interest_l154_154799

def annualInterest (P1 P2 : ℝ) (rate1 rate2 : ℝ) : ℝ := 
  (P1 * (rate1 / 100)) + (P2 * (rate2 / 100))

theorem total_annual_interest 
  (P1 P2 : ℝ)
  (h₀ : P1 = 2799.9999999999995)
  (h₁ : P1 + P2 = 4000)
  (rate1 rate2 : ℝ)
  (h₂ : rate1 = 3)
  (h₃ : rate2 = 5) :
  annualInterest P1 P2 rate1 rate2 = 144 :=
by
  sorry

end total_annual_interest_l154_154799


namespace intersection_M_N_l154_154350

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l154_154350


namespace five_coins_all_heads_or_tails_l154_154212

theorem five_coins_all_heads_or_tails : 
  (1 / 2) ^ 5 + (1 / 2) ^ 5 = 1 / 16 := 
by 
  sorry

end five_coins_all_heads_or_tails_l154_154212


namespace find_weight_per_square_inch_l154_154584

-- Define the TV dimensions and other given data
def bill_tv_width : ℕ := 48
def bill_tv_height : ℕ := 100
def bob_tv_width : ℕ := 70
def bob_tv_height : ℕ := 60
def weight_difference_pounds : ℕ := 150
def ounces_per_pound : ℕ := 16

-- Compute areas
def bill_tv_area := bill_tv_width * bill_tv_height
def bob_tv_area := bob_tv_width * bob_tv_height

-- Assume weight per square inch
def weight_per_square_inch : ℕ := 4

-- Total weight computation given in ounces
def bill_tv_weight := bill_tv_area * weight_per_square_inch
def bob_tv_weight := bob_tv_area * weight_per_square_inch
def weight_difference_ounces := weight_difference_pounds * ounces_per_pound

-- The theorem to prove
theorem find_weight_per_square_inch : 
  bill_tv_weight - bob_tv_weight = weight_difference_ounces → weight_per_square_inch = 4 :=
by
  intros
  /- Proof by computation -/
  sorry

end find_weight_per_square_inch_l154_154584


namespace angle_CED_constant_circumcircle_through_fixed_point_l154_154342

-- Declare the initial setup and point definitions
variables {A B C D E O : Point}
-- Assume \(\omega\) is a semicircle with \(AB\) as the diameter
variables {ω : Semicircle A B O}
-- Assume {CD} is a chord of \(\omega\) with constant length
variables {CD : Chord C D ω}
-- Assume \(C, D\) are in the interior of arc \(AB\)
variables (hC : C ∈ ω.interior_arc A B) (hD : D ∈ ω.interior_arc A B)
-- Assume \(E\) is on the diameter \(AB\) with equal inclination from \(C\) and \(D\)
variables (hE : E ∈ line_through A B)
variables (hInclination : inclination C E (line_through A B) = inclination D E (line_through A B))

-- Problem (a): Prove \(\angle CED\) is constant
theorem angle_CED_constant : ∀ (C D E : Point),
  C ∈ ω ∧ D ∈ ω ∧ E ∈ line_through A B ∧ inclination C E (line_through A B) = inclination D E (line_through A B) → 
  ∃ θ : angle, ∀ P Q R : Point, \[θ = ∠PQR] :=
sorry

-- Problem (b): Prove the circumcircle of \(\triangle CED\) passes through a fixed point
theorem circumcircle_through_fixed_point : ∀ (C D E : Point), 
  C ∈ ω ∧ D ∈ ω ∧ E ∈ line_through A B ∧ inclination C E (line_through A B) = inclination D E (line_through A B) → 
  ∃ Q : Point, ∀ α β γ : Point, \[circumcircle(α β γ) passes through Q] :=
sorry

end angle_CED_constant_circumcircle_through_fixed_point_l154_154342


namespace angle_ABC_is_40_l154_154711

-- Define the conditions
open Real

variables (O A B C : Point) -- Points O, A, B, C
variable (circle : Circle O) -- Circle centered at O

-- Angles between the points on the circle
variable (angle_AOB : ℝ)
variable (angle_BOC : ℝ)

-- Conditions
axiom AOB_is_130 : angle_AOB = 130
axiom BOC_is_150 : angle_BOC = 150
axiom ABC_on_circle : IsOnCircle A circle ∧ IsOnCircle B circle ∧ IsOnCircle C circle

-- Theorem: Prove that the angle ABC is 40 degrees
theorem angle_ABC_is_40 : ∃ (angle_ABC : ℝ), angle_ABC = 40 := by
  -- Definitions and necessary calculations would be here, which we skip
  sorry

end angle_ABC_is_40_l154_154711


namespace value_of_a_plus_b_l154_154835

-- Define the given nested fraction expression
def nested_expr := 1 + 1 / (1 + 1 / (1 + 1))

-- Define the simplified form of the expression
def simplified_form : ℚ := 13 / 8

-- The greatest common divisor condition
def gcd_condition : ℕ := Nat.gcd 13 8

-- The ultimate theorem to prove
theorem value_of_a_plus_b : 
  nested_expr = simplified_form ∧ gcd_condition = 1 → 13 + 8 = 21 := 
by 
  sorry

end value_of_a_plus_b_l154_154835


namespace one_thirds_of_nine_halfs_l154_154686

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l154_154686


namespace lucas_fence_painting_l154_154700

-- Define the conditions
def total_time := 60
def time_painting := 12
def rate_per_minute := 1 / total_time

-- State the theorem
theorem lucas_fence_painting :
  let work_done := rate_per_minute * time_painting
  work_done = 1 / 5 :=
by
  -- Proof omitted
  sorry

end lucas_fence_painting_l154_154700


namespace percent_volume_removed_is_3_29_l154_154526

def original_box_volume : ℝ := 18 * 12 * 9

def small_cube_volume : ℝ := 2 * 2 * 2

def total_removed_volume : ℝ := 8 * small_cube_volume

def percentage_removed (original_volume removed_volume : ℝ) : ℝ :=
  (removed_volume / original_volume) * 100

theorem percent_volume_removed_is_3_29 :
  percentage_removed original_box_volume total_removed_volume = 3.29 :=
begin
  sorry
end

end percent_volume_removed_is_3_29_l154_154526


namespace rectangle_diagonals_equal_not_parallelogram_l154_154483

structure Rectangle :=
(opposite_sides_parallel : ∀ {a b : ℝ}, a ≠ b → a = b)
(opposite_sides_equal : ∀ {a b : ℝ}, a ≠ b → a = b)
(diagonals_bisect_each_other : ∀ {a b c d : ℝ}, a = c → b = d → a = c ∧ b = d)
(diagonals_equal : ∀ {a b : ℝ}, a = b)

structure Parallelogram :=
(opposite_sides_parallel : ∀ {a b : ℝ}, a ≠ b → a = b)
(opposite_sides_equal : ∀ {a b : ℝ}, a ≠ b → a = b)
(diagonals_bisect_each_other : ∀ {a b c d : ℝ}, a = c → b = d → a = c ∧ b = d)

theorem rectangle_diagonals_equal_not_parallelogram :
  (∀ r : Rectangle, r.diagonals_equal) ∧ ¬ (∀ p : Parallelogram, p.diagonals_equal) :=
by
  sorry

end rectangle_diagonals_equal_not_parallelogram_l154_154483


namespace find_m_n_and_max_value_l154_154662

-- Define the function f
def f (m n : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + 3 * m + n

-- Define a predicate for the function being even
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the conditions and what we want to prove
theorem find_m_n_and_max_value :
  ∀ m n : ℝ,
    is_even_function (f m n) →
    (m - 1 ≤ 2 * m) →
      (m = 1 / 3 ∧ n = 0) ∧ 
      (∀ x : ℝ, -2 / 3 ≤ x ∧ x ≤ 2 / 3 → f (1/3) 0 x ≤ 31 / 27) :=
by
  sorry

end find_m_n_and_max_value_l154_154662


namespace price_of_second_container_l154_154512

def volume (r h : ℝ) : ℝ := π * r^2 * h

def price_proportional_to_volume (price_per_unit_volume volume : ℝ) : ℝ := price_per_unit_volume * volume

theorem price_of_second_container :
  let d1 := 5 / 2 in
  let h1 := 6 in
  let price1 := 0.75 in
  let volume1 := volume d1 h1 in

  let d2 := 10 / 2 in
  let h2 := 9 in
  let volume2 := volume d2 h2 in

  let price_ratio := volume2 / volume1 in
  let price2 := price_ratio * price1 in

  price2 = 4.50 :=
by
  sorry

end price_of_second_container_l154_154512


namespace relationship_a_e_l154_154252

theorem relationship_a_e (a : ℝ) (h : 0 < a ∧ a < 1) : a < Real.exp a - 1 ∧ Real.exp a - 1 < a ^ Real.exp 1 := by
  sorry

end relationship_a_e_l154_154252


namespace cube_root_calculation_l154_154926

theorem cube_root_calculation: 
  ∃ x : ℝ, x = (∛((5 ^ 3 * 3) + (5 ^ 3 * 3) + (5 ^ 3 * 3))) = 5 * 3^(2/3) :=
by
  use 5 * 3^(2/3)
  sorry

end cube_root_calculation_l154_154926


namespace fourth_derivative_l154_154611

variable (x : ℝ)

noncomputable def y : ℝ := (x^3 + 3) * Real.exp(4 * x + 3)

theorem fourth_derivative :
  (deriv^[4] (fun x => (x^3 + 3) * Real.exp(4 * x + 3))) x = (256 * x^3 + 768 * x^2 + 576 * x + 864) * Real.exp(4 * x + 3) :=
sorry

end fourth_derivative_l154_154611


namespace b_work_rate_l154_154490

theorem b_work_rate :
  let a_rate := 1 / 6 in
  let c_rate := 1 / 12 in
  let total_earnings := 2340 in
  let b_share := 780 in
  let combined_rate := a_rate + c_rate in
  b_share = total_earnings / 3 →
  (1 / combined_rate) = 4 :=
by
  sorry

end b_work_rate_l154_154490


namespace max_checkers_on_board_l154_154793

-- Define the size of the board.
def board_size : ℕ := 8

-- Define the max number of checkers per row/column.
def max_checkers_per_line : ℕ := 3

-- Define the conditions of the board.
structure BoardConfiguration :=
  (rows : Fin board_size → Fin (max_checkers_per_line + 1))
  (columns : Fin board_size → Fin (max_checkers_per_line + 1))
  (valid : ∀ (i : Fin board_size), rows i ≤ max_checkers_per_line ∧ columns i ≤ max_checkers_per_line)

-- Define the function to calculate the total number of checkers.
def total_checkers (config : BoardConfiguration) : ℕ :=
  Finset.univ.sum (λ i => config.rows i + config.columns i)

-- The theorem which states that the maximum number of checkers is 30.
theorem max_checkers_on_board : ∃ (config : BoardConfiguration), total_checkers config = 30 :=
  sorry

end max_checkers_on_board_l154_154793


namespace solution_interval_l154_154573

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l154_154573


namespace cost_of_pack_of_socks_is_5_l154_154917

-- Conditions definitions
def shirt_price : ℝ := 12.00
def short_price : ℝ := 15.00
def trunks_price : ℝ := 14.00
def shirts_count : ℕ := 3
def shorts_count : ℕ := 2
def total_bill : ℝ := 102.00
def total_known_cost : ℝ := 3 * shirt_price + 2 * short_price + trunks_price

-- Definition of the problem statement
theorem cost_of_pack_of_socks_is_5 (S : ℝ) : total_bill = total_known_cost + S + 0.2 * (total_known_cost + S) → S = 5 := 
by
  sorry

end cost_of_pack_of_socks_is_5_l154_154917


namespace molecular_weight_1_mole_l154_154044

-- Define the molecular weight of 3 moles
def molecular_weight_3_moles : ℕ := 222

-- Prove that the molecular weight of 1 mole is 74 given the molecular weight of 3 moles
theorem molecular_weight_1_mole (mw3 : ℕ) (h : mw3 = 222) : mw3 / 3 = 74 :=
by
  sorry

end molecular_weight_1_mole_l154_154044


namespace tan_ratio_l154_154752

variable (a b : Real)

theorem tan_ratio (h1 : Real.sin (a + b) = 5 / 8) (h2 : Real.sin (a - b) = 1 / 4) : 
  (Real.tan a) / (Real.tan b) = 7 / 3 := 
by 
  sorry

end tan_ratio_l154_154752


namespace initial_percentage_increase_l154_154521

theorem initial_percentage_increase 
  (W R : ℝ) 
  (P : ℝ)
  (h1 : R = W * (1 + P/100)) 
  (h2 : R * 0.70 = W * 1.18999999999999993) :
  P = 70 :=
by sorry

end initial_percentage_increase_l154_154521


namespace katya_classmates_l154_154794

-- Let N be the number of Katya's classmates
variable (N : ℕ)

-- Let K be the number of candies Artyom initially received
variable (K : ℕ)

-- Condition 1: After distributing some candies, Katya had 10 more candies left than Artyom
def condition_1 := K + 10

-- Condition 2: Katya gave each child, including herself, one more candy, so she gave out N + 1 candies in total
def condition_2 := N + 1

-- Condition 3: After giving out these N + 1 candies, everyone in the class has the same number of candies.
def condition_3 : Prop := (K + 1) = (condition_1 K - condition_2 N) / (N + 1)


-- Goal: Prove the number of Katya's classmates N is 9.
theorem katya_classmates : N = 9 :=
by
  -- Restate the conditions in Lean
  
  -- Apply the conditions to find that the only viable solution is N = 9
  sorry

end katya_classmates_l154_154794


namespace average_score_difference_l154_154942

theorem average_score_difference {A B : ℝ} (hA : (19 * A + 125) / 20 = A + 5) (hB : (17 * B + 145) / 18 = B + 6) :
  (B + 6) - (A + 5) = 13 :=
  sorry

end average_score_difference_l154_154942


namespace jenny_commute_time_l154_154340

-- Define the constants and variables based on the conditions
def bus_time : ℝ := 15
def walk_distance_indirect : ℝ := 0.75
def walk_distance_direct : ℝ := 1.5

-- Define the indifference condition for the total time, which translates to the equation given in the solution
theorem jenny_commute_time (T : ℝ) (h1 : bus_time + walk_distance_indirect * T = walk_distance_direct * T) : 
  walk_distance_direct * T = 30 :=
begin
  -- Use the provided indifference condition to infer the result
  suffices h2 : T = 20,
  { rw h2, exact mul_comm 1.5 20, norm_num },
  linarith,
end

end jenny_commute_time_l154_154340


namespace Angela_height_is_157_l154_154159

variable (height_Amy height_Helen height_Angela : ℕ)

-- The conditions
axiom h_Amy : height_Amy = 150
axiom h_Helen : height_Helen = height_Amy + 3
axiom h_Angela : height_Angela = height_Helen + 4

-- The proof to show Angela's height is 157 cm
theorem Angela_height_is_157 : height_Angela = 157 :=
by
  rw [h_Amy] at h_Helen
  rw [h_Helen] at h_Angela
  exact h_Angela

end Angela_height_is_157_l154_154159


namespace prob_all_heads_or_tails_five_coins_l154_154205

theorem prob_all_heads_or_tails_five_coins :
  (number_of_favorable_outcomes : ℕ) (total_number_of_outcomes : ℕ) (probability : ℚ) 
  (h_favorable : number_of_favorable_outcomes = 2)
  (h_total : total_number_of_outcomes = 32)
  (h_probability : probability = number_of_favorable_outcomes / total_number_of_outcomes) :
  probability = 1 / 16 :=
by
  sorry

end prob_all_heads_or_tails_five_coins_l154_154205


namespace min_dot_product_rectangle_min_dot_product_rectangle_proof_l154_154325

theorem min_dot_product_rectangle (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 2) 
                                  (h2 : 0 ≤ x ∧ x ≤ ∞) :
  let P := (2 - x, 0)
  let Q := (2, 1 + x)
  let A := (0, 1)
  (PA := (A.1 - P.1, A.2 - P.2))
  (PQ := (Q.1 - P.1, Q.2 - P.2))
  let PA_dot_PQ := PA.1 * PQ.1 + PA.2 * PQ.2
  PA_dot_PQ = (x^2 - x + 1) :=
begin
  sorry
end

theorem min_dot_product_rectangle_proof :
  (min_value : ℝ) (∀ (x : ℝ), x^2 - x + 1 ≥ min_value) :=
begin
  have h := @min_dot_product_rectangle,
  -- Complete the square
  let g := λ x : ℝ, x^2 - x + 1,
  have g_ge_3over4 : ∀ x, g x ≥ 3 / 4, {
    intros x,
    sorry -- Completing the square and showing the minimum
  },
  use 3 / 4,
  exact g_ge_3over4,
end

end min_dot_product_rectangle_min_dot_product_rectangle_proof_l154_154325


namespace quadratic_transform_l154_154823

theorem quadratic_transform : ∀ (x : ℝ), x^2 = 3 * x + 1 ↔ x^2 - 3 * x - 1 = 0 :=
by
  sorry

end quadratic_transform_l154_154823


namespace integer_solutions_exist_l154_154760

theorem integer_solutions_exist (a : ℕ) (ha : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 := 
sorry

end integer_solutions_exist_l154_154760


namespace equivalent_single_discount_l154_154883

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.30
noncomputable def second_discount : ℝ := 0.15
noncomputable def third_discount : ℝ := 0.10

theorem equivalent_single_discount :
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  (1 - (final_price / original_price)) = 0.4645 :=
by
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  sorry

end equivalent_single_discount_l154_154883


namespace min_value_of_seq_l154_154672

theorem min_value_of_seq 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (m a₁ : ℝ)
  (h1 : ∀ n, a n + a (n + 1) = n * (-1) ^ ((n * (n + 1)) / 2))
  (h2 : m + S 2015 = -1007)
  (h3 : a₁ * m > 0) :
  ∃ x, x = (1 / a₁) + (4 / m) ∧ x = 9 :=
by
  sorry

end min_value_of_seq_l154_154672


namespace total_time_to_fill_tank_l154_154488

-- Definitions as per conditions
def tank_fill_time_for_one_tap (total_time : ℕ) : Prop :=
  total_time = 16

def number_of_taps_for_second_half (num_taps : ℕ) : Prop :=
  num_taps = 4

-- Theorem statement to prove the total time taken to fill the tank
theorem total_time_to_fill_tank : ∀ (time_one_tap time_total : ℕ),
  tank_fill_time_for_one_tap time_one_tap →
  number_of_taps_for_second_half 4 →
  time_total = 10 :=
by
  intros time_one_tap time_total h1 h2
  -- Proof needed here
  sorry

end total_time_to_fill_tank_l154_154488


namespace area_relation_l154_154501

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  0.5 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_relation (A B C A' B' C' : ℝ × ℝ) (hAA'BB'CC'parallel: 
  ∃ k : ℝ, (A'.1 - A.1 = k * (B'.1 - B.1)) ∧ (A'.2 - A.2 = k * (B'.2 - B.2)) ∧ 
           (B'.1 - B.1 = k * (C'.1 - C.1)) ∧ (B'.2 - B.2 = k * (C'.2 - C.2))) :
  3 * (area_triangle A B C + area_triangle A' B' C') = 
    area_triangle A B' C' + area_triangle B C' A' + area_triangle C A' B' +
    area_triangle A' B C + area_triangle B' C A + area_triangle C' A B := 
sorry

end area_relation_l154_154501


namespace profit_percentage_l154_154868

theorem profit_percentage (C : ℝ) (hC : C > 0) :
  15 * C < 20 * C → (20 * C - 15 * C) / (15 * C) * 100 ≈ 33.33 := by
  intros h
  have h₁ : 0 < 15 * C := by
    sorry
  have h₂ : 5 * C / (15 * C) * 100 = 33.33 := by
    sorry
  exact h₂

end profit_percentage_l154_154868


namespace connie_start_marbles_l154_154934

variable (marbles_total marbles_given marbles_left : ℕ)

theorem connie_start_marbles :
  marbles_given = 73 → marbles_left = 70 → marbles_total = marbles_given + marbles_left → marbles_total = 143 :=
by intros; sorry

end connie_start_marbles_l154_154934


namespace one_thirds_in_nine_halves_l154_154681

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l154_154681


namespace intersection_and_union_l154_154503

def A (x : ℝ) : Set ℝ := {2, -1, x^2 - x + 1}
def B (x y : ℝ) : Set ℝ := {2y, -4, x + 4}
def C : Set ℝ := {-1}

theorem intersection_and_union (x y : ℝ) :
  (A x ∩ B x y = C) →
  (A x ∪ B x y = {2, -1, x^2 - x + 1, 2y, -4, x + 4}) :=
by
  sorry

end intersection_and_union_l154_154503


namespace reflection_line_equation_l154_154847

theorem reflection_line_equation {D E F D' E' F' : ℝ × ℝ} 
  (hD : D = (3, 2)) (hE : E = (8, 7)) (hF : F = (6, -4))
  (hD' : D' = (9, 2)) (hE' : E' = (14, 7)) (hF' : F' = (12, -4)) :
  ∃ M : ℝ, (∀ p ∈ {D, E, F}, ∃ p' ∈ {D', E', F'}, 
    ∃ mid : ℝ × ℝ, mid = ((p.1 + p'.1) / 2, p.2) ∧ M = mid.1) ∧
    (∀ (x : ℝ), M = x) :=
begin
  have M := 6,
  use M,
  split,
  {
    intros p hp,
    cases hp,
    { use D', split, assumption, 
      use (6, 2), split, refl, exact rfl },
    cases hp,
    { use E', split, assumption, 
      use (11, 7), split, refl, exact rfl },
    cases hp,
    { use F', split, assumption,
      use (9, -4), split, refl, exact rfl },
    contradiction
  },
  {
    intro x,
    exact rfl
  }
end

end reflection_line_equation_l154_154847


namespace A_ge_B_l154_154344

open Real

theorem A_ge_B (x : ℝ) (n : ℕ) (hx : 0 < x) (hn : 0 < n) :
  x^n + x^(-n) ≥ x^(n - 1) + x^(1 - n) :=
sorry

end A_ge_B_l154_154344


namespace decreasing_sufficient_condition_l154_154227

theorem decreasing_sufficient_condition {a : ℝ} (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (a-2)*x^3 > (a-2)*y^3) :=
by
  sorry

end decreasing_sufficient_condition_l154_154227


namespace sum_of_series_equivalent_l154_154592

noncomputable def criterion (a b c d : ℕ) : Prop := 1 ≤ a ∧ a < b ∧ b < c ∧ c < d

theorem sum_of_series_equivalent :
  (∑ a b c d in {x | 1 ≤ x ∧ x < b ∧ b < c ∧ c < d}, 1 / (2^a * 4^b * 8^c * 16^d).to_real) = (2 : ℝ) / (240975 : ℝ) :=
by
  sorry

end sum_of_series_equivalent_l154_154592


namespace interval_of_min_max_value_l154_154301

def f (x : ℝ) : ℝ := -0.5 * x^2 + 13 / 2

theorem interval_of_min_max_value (a b : ℝ) (ha : f a = 2 * b) (hb : f b = 2 * a) :
  [a, b] = [1, 3] ∨ [a, b] = [-2 - real.sqrt 17, 13 / 4] := by
  sorry

end interval_of_min_max_value_l154_154301


namespace find_vector_at_t_zero_l154_154131

def vector_at_t (a d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (a.1 + t*d.1, a.2 + t*d.2)

theorem find_vector_at_t_zero :
  ∃ (a d : ℝ × ℝ),
    vector_at_t a d 1 = (2, 3) ∧
    vector_at_t a d 4 = (8, -5) ∧
    vector_at_t a d 5 = (10, -9) ∧
    vector_at_t a d 0 = (0, 17/3) :=
by
  sorry

end find_vector_at_t_zero_l154_154131


namespace measure_of_angle_H_in_parallelogram_l154_154322

theorem measure_of_angle_H_in_parallelogram (H : Type) [AddGroup H] {E F G : H}
  (h_parallelogram : parallelogram E F G H)
  (angle_F : measure_of_angle F = 125) :
  measure_of_angle H = 55 := 
sorry

end measure_of_angle_H_in_parallelogram_l154_154322


namespace platform_length_proof_l154_154118

noncomputable def train_length : ℝ := 1200
noncomputable def time_to_cross_tree : ℝ := 120
noncomputable def time_to_cross_platform : ℝ := 240
noncomputable def speed_of_train : ℝ := train_length / time_to_cross_tree
noncomputable def platform_length : ℝ := 2400 - train_length

theorem platform_length_proof (h1 : train_length = 1200) (h2 : time_to_cross_tree = 120) (h3 : time_to_cross_platform = 240) :
  platform_length = 1200 := by
  sorry

end platform_length_proof_l154_154118


namespace collinear_E_F_N_l154_154751

-- Definitions of points and properties given in the problem
variables {A B C D E F M N : Point}

-- Assume ABCD is a cyclic quadrilateral
axiom cyclic_quadrilateral : CyclicQuadrilateral A B C D

-- Definitions of points E, F, M, and properties of N
axiom E_def : E = line_through A D ∩ line_through B C
axiom F_def : F = line_through A C ∩ line_through B D
axiom M_midpoint : M = midpoint C D
axiom N_circumcircle : N ∈ circumcircle A M B ∧ N ≠ M
axiom N_ratio : dist A M / dist B M = dist A N / dist B N

-- The proposition to prove
theorem collinear_E_F_N : Collinear E F N :=
by
  sorry

end collinear_E_F_N_l154_154751


namespace fencing_problem_l154_154867

def total_fencing_required (L W F : ℝ) (A : ℝ) : Prop :=
  A = L * W → F = 2 * W + L

theorem fencing_problem : 
  ∀ (L W F : ℝ), (L = 80) → (680 = L * W) → F = 2 * W + L → F = 97 :=
by 
  intros L W F hL hA hF
  rw hL at hA
  have hW : W = 680 / 80 := by
    linarith
  rw [hW, hL]
  linarith

end fencing_problem_l154_154867


namespace carousel_revolutions_l154_154891

/-- Prove that the number of revolutions a horse 4 feet from the center needs to travel the same distance
as a horse 16 feet from the center making 40 revolutions is 160 revolutions. -/
theorem carousel_revolutions (r₁ : ℕ := 16) (revolutions₁ : ℕ := 40) (r₂ : ℕ := 4) :
  (revolutions₁ * (r₁ / r₂) = 160) :=
sorry

end carousel_revolutions_l154_154891


namespace sum_of_squares_divisible_by_sum_l154_154305

theorem sum_of_squares_divisible_by_sum (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h_bound : a < 2017 ∧ b < 2017 ∧ c < 2017)
    (h_mod : (a^3 - b^3) % 2017 = 0 ∧ (b^3 - c^3) % 2017 = 0 ∧ (c^3 - a^3) % 2017 = 0) :
    (a^2 + b^2 + c^2) % (a + b + c) = 0 :=
by
  sorry

end sum_of_squares_divisible_by_sum_l154_154305


namespace george_change_sum_l154_154979

theorem george_change_sum :
  let amounts := {n | 0 ≤ n ∧ n < 100 
                   ∧ (∃ k : ℕ, n = 5 * k + 4) 
                   ∧ (∃ m : ℕ, n = 10 * m + 9)} in
  ∑ x in amounts, x = 344 := 
by
  sorry

end george_change_sum_l154_154979


namespace age_ratio_l154_154189

theorem age_ratio (R F : ℕ) 
  (h1 : F + 8 = 2.5 * (R + 8))
  (h2 : F + 16 = 2 * (R + 16)) : (F / R) = 4 := 
sorry

end age_ratio_l154_154189


namespace complete_square_min_value_relationship_M_N_triangle_shape_l154_154402

section completion

-- Problem 1
theorem complete_square (x : ℝ) : (x^2 - 4*x + 5) = (x - 2)^2 + 1 :=
by { sorry }

-- Problem 2
theorem min_value (x : ℝ) : ∃ m, m = 2 ∧ ∀ y, y = x^2 - 2*x + 3 → y ≥ m :=
by { sorry }

-- Problem 3
theorem relationship_M_N (a : ℝ) (M := a^2 - a) (N := a - 2) : M > N :=
by { sorry }

-- Problem 4
theorem triangle_shape (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6*a - 10*b - 8*c + 50 = 0) : a = 3 ∧ b = 4 ∧ c = 5 :=
begin
  sorry,
end

end completion

end complete_square_min_value_relationship_M_N_triangle_shape_l154_154402


namespace return_speed_l154_154509

theorem return_speed (distance : ℝ) (speed_to : ℝ) (total_time : ℝ) (time_to : ℝ) : 
  total_time = time_to + distance / speed_to → 
  distance / (total_time - time_to) = 2 :=
by
  intro h,
  have h1 : distance = 6, by sorry,
  have h2 : speed_to = 3, by sorry,
  have h3 : total_time = 5, by sorry,
  have h4 : time_to = 2, by sorry,
  rw [h1, h2, h3, h4],
  ring

end return_speed_l154_154509


namespace polynomial_degree_at_most_2017_l154_154957

noncomputable def polynomial_condition (P : ℝ → ℝ) :=
  ∀ x : ℝ, P(x) + (∑ i in finset.range 1008, (↑(nat.choose 2018 (2*i+2)) : ℝ) * P(x + (2*i+2))) + P(x + 2018) =
           (∑ i in finset.range 1008,  (↑(nat.choose 2018 (2*i+1)) : ℝ) * P(x + (2*i+1))) + (↑(nat.choose 2018 2107) : ℝ) * P(x + 2017)

theorem polynomial_degree_at_most_2017 (P : ℝ → ℝ) (hP : polynomial_condition P) : degree P ≤ 2017 :=
sorry

end polynomial_degree_at_most_2017_l154_154957


namespace OC_equals_fraction_l154_154141

noncomputable def OC {θ : ℝ} (s : ℝ) : ℝ :=
  1 / (1 + s)

theorem OC_equals_fraction (θ : ℝ) (s : ℝ) (c : ℝ) :
  s = Real.sin (2 * θ) ∧ c = Real.cos (2 * θ) →
  let OC_val := OC s in
  OC_val = (1 / (1 + s)) := 
by
  intro h
  obtain ⟨hs, hc⟩ := h
  rw [OC]
  sorry

end OC_equals_fraction_l154_154141


namespace domain_of_f_l154_154600

def f (x : ℝ) : ℝ := 1 / (3 * x - 1 + x - 6)

theorem domain_of_f :
  {x : ℝ | ∃ (y : ℝ), y = 1 / (3 * x - 1 + x - 6)} = { x : ℝ | x ≠ 7 / 4 } := 
by
  sorry

end domain_of_f_l154_154600


namespace directrix_of_parabola_l154_154964

theorem directrix_of_parabola : ∀ (x : ℝ), y = (x^2 - 8*x + 12) / 16 → ∃ (d : ℝ), d = -1/2 := 
sorry

end directrix_of_parabola_l154_154964


namespace tan_alpha_value_l154_154642

open Real

theorem tan_alpha_value (α : ℝ) (h1 : sin α + cos α = -1 / 2) (h2 : 0 < α ∧ α < π) : tan α = -1 / 3 :=
sorry

end tan_alpha_value_l154_154642


namespace minimum_lines_intersecting_rays_l154_154375

theorem minimum_lines_intersecting_rays (k : ℕ) (hk : 0 < k) 
  (P : EuclideanSpace ℝ 2) (α : AffineSubspace ℝ (EuclideanSpace ℝ 2)) (hα : P ∈ α) :
  ∃ n, (∀ (lines : finset (AffineSubspace ℝ (EuclideanSpace ℝ 2))), 
    (∀ l ∈ lines, P ∉ l) ∧ (∀ (ray : EuclideanSpace ℝ 2 → Prop), 
      (∀ x ∈ α, ray x → ∃ l ∈ lines, ray x ∩ l ≠ ∅) → size lines ≥ k) → size lines = 2 * k + 1) :=
sorry

end minimum_lines_intersecting_rays_l154_154375


namespace smallest_number_ends_six_increases_fourfold_l154_154203

theorem smallest_number_ends_six_increases_fourfold :
  ∃ X : ℕ, (X % 10 = 6) ∧ (X < 200000) ∧ (4 * X = (6 * 10^(nat.log10 (X / 10 + 1)) + (X / 10))) ∧ X = 153846 :=
begin
  sorry
end

end smallest_number_ends_six_increases_fourfold_l154_154203


namespace probability_heads_at_least_9_of_12_flips_l154_154002

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154002


namespace probability_arithmetic_sequence_l154_154709

-- Define the conditions for the problem
def balls : List ℕ := [2, 3, 4, 6]

def is_arithmetic_sequence (a b c : ℕ) : Prop := b - a = c - b

def count_arithmetic_sequences (lst : List ℕ) : ℕ :=
  List.length $ (lst.combinations 3).filter (λ l, match l with
    | [a, b, c] := is_arithmetic_sequence a b c
    | _ := false
  end)

-- Define the main theorem
theorem probability_arithmetic_sequence : (count_arithmetic_sequences balls) / (balls.combinations 3).length = 1 / 2 :=
  sorry

end probability_arithmetic_sequence_l154_154709


namespace find_f_2016_l154_154230

noncomputable def f : ℕ → ℝ → ℝ
| 0       => λ x, Real.cos x
| (n + 1) => λ x, (f n (x)).derivative

theorem find_f_2016 (x : ℝ) : f 2016 x = Real.cos x := 
by
  sorry

end find_f_2016_l154_154230


namespace smallest_integer_with_16_divisors_l154_154073

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (number_of_divisors m = 16) → (m ≥ n)) ∧ (n = 120) :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154073


namespace find_wheel_circumference_l154_154538

noncomputable def wheel_distance := 1056  -- distance in cm
noncomputable def wheel_revolutions := 4.003639672429482  -- number of revolutions
noncomputable def wheel_circumference := wheel_distance / wheel_revolutions  -- circumference calculation

theorem find_wheel_circumference :
  wheel_circumference ≈ 263.792 :=
by
  sorry

end find_wheel_circumference_l154_154538


namespace smallest_integer_with_16_divisors_l154_154063

-- Define the condition for the number of divisors of an integer
def num_divisors (n : ℕ) : ℕ :=
  (n.factorization.map (λ p a, a + 1)).prod

-- Define the problem statement as a theorem
theorem smallest_integer_with_16_divisors : ∃ n : ℕ, num_divisors n = 16 ∧ (∀ m : ℕ, num_divisors m = 16 → n ≤ m) :=
by
  -- Placeholder to skip proof
  sorry

end smallest_integer_with_16_divisors_l154_154063


namespace euclidean_algorithm_steps_fibonacci_inequalities_l154_154379

theorem euclidean_algorithm_steps_fibonacci_inequalities
  (a b d : ℕ) 
  (n : ℕ) 
  (h_gcd : Nat.gcd a b = d) 
  (h_ab : a > b)
  (h_steps : -- condition that Euclidean algorithm for (a, b) stops after 'n' steps)
  : (a ≥ d * fibonacci (n + 2)) ∧ (b ≥ d * fibonacci (n + 1)) :=
sorry

end euclidean_algorithm_steps_fibonacci_inequalities_l154_154379


namespace square_area_percentage_error_l154_154550

theorem square_area_percentage_error (s : ℝ) (h : s > 0) : 
  let measured_s := 1.06 * s in
  let actual_area := s^2 in
  let calculated_area := measured_s^2 in
  let error_area := calculated_area - actual_area in
  let percentage_error := (error_area / actual_area) * 100 in
  percentage_error = 12.36 := 
by
  sorry

end square_area_percentage_error_l154_154550


namespace sequence_problem_l154_154647

theorem sequence_problem
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : 1 + a1 + a1 = a1 + a1)
  (h2 : b1 * b1 = b2)
  (h3 : 4 = b2 * b2):
  (a1 + a2) / b2 = 2 :=
by
  -- The proof would go here
  sorry

end sequence_problem_l154_154647


namespace flip_coin_probability_l154_154031

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154031


namespace smallest_integer_with_16_divisors_l154_154057

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ :=
  n.factors.to_finset.prod fun p => (n.factorization p + 1)

-- Define the positive integer n which we need to prove has 16 divisors
def smallest_positive_integer_with_16_divisors (n : ℕ) : Prop :=
  num_divisors n = 16

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, smallest_positive_integer_with_16_divisors n ∧ ∀ m : ℕ, m < n → ¬smallest_positive_integer_with_16_divisors m :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154057


namespace max_value_objective_function_l154_154676

theorem max_value_objective_function (x y : ℝ) (h : some_constraints_on x y) :
  max (2 * x + y) = 2 :=
sorry

end max_value_objective_function_l154_154676


namespace angle_BOC_58_l154_154442

-- Define the conditions given in the problem

variables {A B C D O : Type}
variables [Quad : CyclicQuadrilateral A B C D] -- quadrilateral ABCD is cyclic
variable (BC_eq_CD : dist B C = dist C D) -- BC = CD
variable (angle_BCA_64 : angle B C A = 64) -- ∠BCA = 64 degrees
variable (angle_ACD_70 : angle A C D = 70) -- ∠ACD = 70 degrees
variable (angle_ADO_32 : on_segment A C O ∧ angle A D O = 32) -- O is on AC and ∠ADO = 32 degrees

-- Goal: Prove that ∠BOC = 58 degrees
theorem angle_BOC_58 :
  angle B O C = 58 :=
sorry

end angle_BOC_58_l154_154442


namespace problem_statement_l154_154771

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l154_154771


namespace max_value_of_sine_expression_l154_154997

theorem max_value_of_sine_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hπx : x ≤ π) (hπy : y ≤ π) (hπz : z ≤ π) :
  (A = sin (x - y) + sin (y - z) + sin (z - x)) → 
  A ≤ 2 :=
sorry

end max_value_of_sine_expression_l154_154997


namespace sprint_jog_difference_l154_154178

variable (distance_sprinted distance_jogged : ℝ)

-- Given conditions
def distance_sprinted : ℝ := 0.875
def distance_jogged : ℝ := 0.75

-- Lean statement for the proof problem
theorem sprint_jog_difference : distance_sprinted - distance_jogged = 0.125 := 
by {
  sorry
}

end sprint_jog_difference_l154_154178


namespace positive_integer_solutions_condition_l154_154991

theorem positive_integer_solutions_condition (a : ℕ) (A B : ℝ) :
  (∃ (x y z : ℕ), x^2 + y^2 + z^2 = (13 * a)^2 ∧
  x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = (1/4) * (2 * A + B) * (13 * a)^4)
  ↔ A = (1 / 2) * B := 
sorry

end positive_integer_solutions_condition_l154_154991


namespace al_sandwich_combinations_l154_154424

def types_of_bread : ℕ := 5
def types_of_meat : ℕ := 6
def types_of_cheese : ℕ := 5

def restricted_turkey_swiss_combinations : ℕ := 5
def restricted_white_chicken_combinations : ℕ := 5
def restricted_rye_turkey_combinations : ℕ := 5

def total_sandwich_combinations : ℕ := types_of_bread * types_of_meat * types_of_cheese

def valid_sandwich_combinations : ℕ :=
  total_sandwich_combinations - restricted_turkey_swiss_combinations
  - restricted_white_chicken_combinations - restricted_rye_turkey_combinations

theorem al_sandwich_combinations : valid_sandwich_combinations = 135 := 
  by
  sorry

end al_sandwich_combinations_l154_154424


namespace check_polynomials_l154_154087

-- Define the expressions provided
def expr1 := (1 : ℝ) / x
def expr2 := 2 * x + y
def expr3 := (1 / 3) * a^2 * b
def expr4 := (x - y) / π
def expr5 := (5 * y) / (4 * x)
def expr6 := (0 : ℝ)

-- Define a function to check if expressions are polynomials
noncomputable def isPolynomial (expr : ℝ) : Prop := sorry

-- List of conditions to check which are polynomials
def polynomials : List (ℝ) := [expr2, expr3, expr4, expr6]

-- Total count of polynomials in the given expressions
def totalPolynomials : ℕ := List.length polynomials

-- Theorem stating that there are exactly 4 polynomials
theorem check_polynomials : totalPolynomials = 4 := by sorry

end check_polynomials_l154_154087


namespace smallest_integer_with_16_divisors_l154_154054

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ :=
  n.factors.to_finset.prod fun p => (n.factorization p + 1)

-- Define the positive integer n which we need to prove has 16 divisors
def smallest_positive_integer_with_16_divisors (n : ℕ) : Prop :=
  num_divisors n = 16

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, smallest_positive_integer_with_16_divisors n ∧ ∀ m : ℕ, m < n → ¬smallest_positive_integer_with_16_divisors m :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154054


namespace new_perimeter_l154_154529

variable (s x : ℝ) -- define variables for side length and original perimeter

-- Defining the conditions based on the problem
def original_perimeter : ℝ := 4 * s

def new_side_length : ℝ := 12 * s

theorem new_perimeter (h : x = 4 * s) : 12 * x = 48 * s := by
  calc
    12 * x = 12 * (4 * s) : by rw [h]
    ...   = 48 * s        : by ring

#check new_perimeter

end new_perimeter_l154_154529


namespace flip_coin_probability_l154_154030

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154030


namespace total_handshakes_l154_154566

def num_inter_team_handshakes : ℕ := 6 * 6
def num_intra_team_handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2
def num_handshakes_with_referees (num_team_members : ℕ) (num_referees : ℕ) : ℕ := num_team_members * num_referees

theorem total_handshakes (num_team_members : ℕ) (num_referees : ℕ) :
  num_team_members = 7 ∧ num_referees = 2 →
  num_inter_team_handshakes + 2 * num_intra_team_handshakes num_team_members + num_handshakes_with_referees (2 * num_team_members) num_referees + num_referees * (num_referees - 1) / 2 = 107 := 
by
  intros h
  cases h with h_team_members h_referees
  rw [h_team_members, h_referees]
  sorry

end total_handshakes_l154_154566


namespace number_of_red_balls_eq_47_l154_154121

theorem number_of_red_balls_eq_47
  (T : ℕ) (white green yellow purple : ℕ)
  (neither_red_nor_purple_prob : ℚ)
  (hT : T = 100)
  (hWhite : white = 10)
  (hGreen : green = 30)
  (hYellow : yellow = 10)
  (hPurple : purple = 3)
  (hProb : neither_red_nor_purple_prob = 0.5)
  : T - (white + green + yellow + purple) = 47 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end number_of_red_balls_eq_47_l154_154121


namespace sequence_100_l154_154732

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else sequence (n - 1) + 4

theorem sequence_100 : sequence 100 = 397 :=
by {
  -- The proof would go here
  sorry
}

end sequence_100_l154_154732


namespace smallest_integer_with_16_divisors_l154_154049

-- Define prime factorization and the function to count divisors
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def number_of_divisors (n : ℕ) : ℕ :=
  (prime_factorization n).foldr (λ ⟨_, a⟩ acc, acc * (a + 1)) 1

-- Main theorem stating the smallest positive integer with exactly 16 divisors is 210
theorem smallest_integer_with_16_divisors : 
  ∃ n, n > 0 ∧ number_of_divisors n = 16 ∧ ∀ m, m > 0 ∧ number_of_divisors m = 16 → n ≤ m :=
begin
  use 210,
  split,
  { -- Prove 210 > 0
    exact nat.zero_lt_succ _,
  },
  split,
  { -- Prove number_of_divisors 210 = 16
    sorry,
  },
  { -- Prove minimality
    intros m hm1 hm2,
    sorry,
  }
end

end smallest_integer_with_16_divisors_l154_154049


namespace infinite_grid_coloring_l154_154952

theorem infinite_grid_coloring (color : ℕ × ℕ → Fin 4)
  (h_coloring_condition : ∀ (i j : ℕ), color (i, j) ≠ color (i + 1, j) ∧
                                      color (i, j) ≠ color (i, j + 1) ∧
                                      color (i, j) ≠ color (i + 1, j + 1) ∧
                                      color (i + 1, j) ≠ color (i, j + 1)) :
  ∃ m : ℕ, ∃ a b : Fin 4, ∀ n : ℕ, color (m, n) = a ∨ color (m, n) = b :=
sorry

end infinite_grid_coloring_l154_154952


namespace num_girls_l154_154715

theorem num_girls (boys girls : ℕ) (h1 : girls = boys + 228) (h2 : boys = 469) : girls = 697 :=
sorry

end num_girls_l154_154715


namespace proof_ordered_pairs_l154_154183

noncomputable def count_ordered_pairs := 
  {p : ℕ × ℕ // (p.1 > 0) ∧ (p.2 > 0) ∧ 
    (p.1 * real.sqrt p.2 + p.2 * real.sqrt p.1 - real.sqrt (2007 * p.1) - real.sqrt (2007 * p.2) + real.sqrt (2007 * (p.1 * p.2)) = 2007)}

#eval finset.card (set.to_finset (set_of (λ p : ℕ × ℕ, p ∈ count_ordered_pairs))) -- should be 6

theorem proof_ordered_pairs :
  finset.card (set.to_finset (set_of (λ p : ℕ × ℕ, p ∈ count_ordered_pairs))) = 6 := 
sorry

end proof_ordered_pairs_l154_154183


namespace minimal_number_of_cubes_to_cover_snaps_l154_154134

def Cube : Type := { snaps : Fin 2, receptacles : Fin 4 }

theorem minimal_number_of_cubes_to_cover_snaps : 
  ∃ (n : ℕ), (n = 4) ∧ (∀ cubes : vector Cube n, only_receptacle_holes_visible cubes) :=
sorry

end minimal_number_of_cubes_to_cover_snaps_l154_154134


namespace f_fe_eq_neg1_f_x_gt_neg1_solution_l154_154228

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x)
  else if x < 0 then 1 / x
  else 0  -- handle the case for x = 0 explicitly if needed

theorem f_fe_eq_neg1 : 
  f (f (Real.exp 1)) = -1 := 
by
  -- proof to be filled in
  sorry

theorem f_x_gt_neg1_solution :
  {x : ℝ | f x > -1} = {x : ℝ | (x < -1) ∨ (0 < x ∧ x < Real.exp 1)} :=
by
  -- proof to be filled in
  sorry

end f_fe_eq_neg1_f_x_gt_neg1_solution_l154_154228


namespace books_loaned_out_correct_l154_154895

noncomputable def books_loaned_out_A : ℝ := ((300 - 244) / 0.60).round
noncomputable def books_loaned_out_B : ℝ := ((450 - 386) / 0.70).round
noncomputable def books_loaned_out_C : ℝ := ((350 - 290) / 0.80).round
noncomputable def books_loaned_out_D : ℝ := ((100 - 76) / 0.50).round

theorem books_loaned_out_correct :
  books_loaned_out_A = 93 ∧
  books_loaned_out_B = 91 ∧
  books_loaned_out_C = 75 ∧
  books_loaned_out_D = 48 :=
  by
  sorry

end books_loaned_out_correct_l154_154895


namespace sets_tossed_per_show_l154_154589

-- Definitions
def sets_used_per_show : ℕ := 5
def number_of_shows : ℕ := 30
def total_sets_used : ℕ := 330

-- Statement to prove
theorem sets_tossed_per_show : 
  (total_sets_used - (sets_used_per_show * number_of_shows)) / number_of_shows = 6 := 
by
  sorry

end sets_tossed_per_show_l154_154589


namespace no_sqrt3_combination_l154_154913

-- Conditions
def sqrt_32 : ℝ := Real.sqrt 32
def sqrt_neg_27 : ℝ := -Real.sqrt 27
def sqrt_12 : ℝ := Real.sqrt 12
def sqrt_third : ℝ := Real.sqrt (1 / 3)

-- The proof problem statement
theorem no_sqrt3_combination :
  sqrt_32 = 4 * Real.sqrt 2 ∧ sqrt_neg_27 = -3 * Real.sqrt 3 ∧
  sqrt_12 = 2 * Real.sqrt 3 ∧ sqrt_third = Real.sqrt 3 / 3 →
  (∀ (r : ℝ), r = sqrt_32 → ¬ (r = Real.sqrt 3)) ∧ 
  ¬(sqrt_neg_27 = Real.sqrt 3) ∧ ¬(sqrt_12 = Real.sqrt 3) ∧ ¬(sqrt_third = Real.sqrt 3) :=
by
  sorry

end no_sqrt3_combination_l154_154913


namespace bo_learning_days_l154_154922

theorem bo_learning_days (total_words : ℕ) (known_percentage : ℝ) (words_per_day : ℕ) 
  (h_total_words : total_words = 800)
  (h_known_percentage : known_percentage = 0.20)
  (h_words_per_day : words_per_day = 16) :
  (total_words - (known_percentage * total_words).to_nat) / words_per_day = 40 :=
by {
  sorry
}

end bo_learning_days_l154_154922


namespace flip_coin_probability_l154_154036

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154036


namespace parallel_vectors_x_eq_l154_154980

theorem parallel_vectors_x_eq : ∀ (x : ℝ), (let a := (3, 1); let b := (x, x-1); (a.1 * (b.2 - 1)) - b.1 * a.2 = 0) ↔ x = 3 / 2 :=
by
  intros x
  simp [Prod]
  split
  intro h
  sorry
  intro hx
  rw hx
  sorry

end parallel_vectors_x_eq_l154_154980


namespace angle_B_is_40_degrees_l154_154737

theorem angle_B_is_40_degrees (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A = 3 * angle_B)
  (h2 : angle_B = 2 * angle_C)
  (triangle_sum : angle_A + angle_B + angle_C = 180) :
  angle_B = 40 :=
by
  sorry

end angle_B_is_40_degrees_l154_154737


namespace snail_returns_to_starting_point_l154_154525

-- Define the variables and conditions
variables (a1 a2 b1 b2 : ℕ)

-- Prove that snail can return to starting point after whole number of hours
theorem snail_returns_to_starting_point (h1 : a1 = a2) (h2 : b1 = b2) : (a1 + b1 : ℕ) = (a1 + b1 : ℕ) :=
by sorry

end snail_returns_to_starting_point_l154_154525


namespace probability_heads_9_or_more_12_flips_l154_154025

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154025


namespace sequence_equation_l154_154831

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 1 then 12 else 3^(n+1)

theorem sequence_equation (n : ℕ) (h : n > 0) :
    (∑ i in Finset.range n, (1 / 3^i.succ : ℝ) * sequence i.succ) = 3 * n + 1 :=
sorry

end sequence_equation_l154_154831


namespace rectangle_width_is_3_l154_154419

-- Define the given conditions
def length_square : ℝ := 9
def length_rectangle : ℝ := 27

-- Calculate the area based on the given conditions
def area_square : ℝ := length_square * length_square

-- Define the area equality condition
def area_equality (width_rectangle : ℝ) : Prop :=
  area_square = length_rectangle * width_rectangle

-- The theorem stating the width of the rectangle
theorem rectangle_width_is_3 (width_rectangle: ℝ) :
  area_equality width_rectangle → width_rectangle = 3 :=
by
  -- Skipping the proof itself as instructed
  intro h
  sorry

end rectangle_width_is_3_l154_154419


namespace weekly_coffee_cost_l154_154382

-- Definitions and conditions
def cups_per_day := 2
def ounces_per_cup := 1.5
def bag_cost := 8
def ounces_per_bag := 10.5
def milk_per_week := 1 / 2
def milk_cost_per_gallon := 4

-- Compute the total weekly cost
theorem weekly_coffee_cost :
  let coffee_per_week := cups_per_day * ounces_per_cup * 7
  let bags_per_week := coffee_per_week / ounces_per_bag
  let coffee_cost_per_week := bags_per_week * bag_cost
  let milk_cost_per_week := milk_per_week * milk_cost_per_gallon
  coffee_cost_per_week + milk_cost_per_week = 18 := by
  sorry

end weekly_coffee_cost_l154_154382


namespace smallest_integer_with_16_divisors_l154_154071

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (∀ m : ℕ, (has_16_divisors m → m ≥ n)) ∧ n = 384 :=
by
  def has_16_divisors (n : ℕ) : Prop :=
    ∃ a₁ a₂ a₃ k₁ k₂ k₃, n = k₁ ^ a₁ * k₂ ^ a₂ * k₃ ^ a₃ ∧ (a₁ + 1) * (a₂ + 1) * (a₃ + 1) = 16
  -- Proof will go here
  sorry

end smallest_integer_with_16_divisors_l154_154071


namespace age_conversion_in_base_l154_154156

theorem age_conversion_in_base (n : ℕ) (h : n = 7231) :
  Nat.ofDigits 16 [9, 9, 14] = Nat.ofDigits 8 [7, 2, 3, 1] :=
by
  have h_eq : 7231 = Nat.ofDigits 8 [7, 2, 3, 1] := rfl
  rw [h, h_eq]
  sorry

end age_conversion_in_base_l154_154156


namespace inverse_function_correct_l154_154818

def f (x : ℝ) : ℝ := 4^x - 1

theorem inverse_function_correct : ∀ (y : ℝ), f (log 4 (y + 1)) = y :=
by
  intro y
  sorry

end inverse_function_correct_l154_154818


namespace distance_between_points_l154_154192

def distance_on_line (a b : ℝ) : ℝ := |b - a|

theorem distance_between_points (a b : ℝ) : distance_on_line a b = |b - a| :=
by sorry

end distance_between_points_l154_154192


namespace sum_arithmetic_sequence_l154_154367

theorem sum_arithmetic_sequence (S : ℕ → ℝ) (h1 : S 4 ≠ 0) (h2 : S 8 = 3 * S 4) (h3 : ∃ λ : ℝ, S 12 = λ * S 8) : 
∃ λ : ℝ, λ = 2 :=
by
  obtain ⟨λ, h⟩ := h3
  use λ
  have : 2 * (S 8 - S 4) = S 4 + (S 12 - S 8), by sorry
  have : 2 * (3 * S 4 - S 4) = S 4 + (λ * 3 *S 4 - 3 * S 4), by sorry
  have : 4 * S 4 = (1 + 3 * λ - 3) * S 4, by sorry
  have : 4 = 1 + 3 * λ - 3, by sorry
  have : 4 = 3 * λ - 2, by sorry
  have : 3 * λ = 6, by sorry
  exact_mod_cast (show λ = 2, by field_simp [*])

end sum_arithmetic_sequence_l154_154367


namespace find_y_l154_154904

variables (y : ℝ)

def rectangle_vertices (A B C D : (ℝ × ℝ)) : Prop :=
  (A = (-2, y)) ∧ (B = (10, y)) ∧ (C = (-2, 1)) ∧ (D = (10, 1))

def rectangle_area (length height : ℝ) : Prop :=
  length * height = 108

def positive_value (x : ℝ) : Prop :=
  0 < x

theorem find_y (A B C D : (ℝ × ℝ)) (hV : rectangle_vertices y A B C D) (hA : rectangle_area 12 (y - 1)) (hP : positive_value y) :
  y = 10 :=
sorry

end find_y_l154_154904


namespace bakery_cake_price_l154_154133

theorem bakery_cake_price (R : ℝ) (discount_pct : ℝ) (one_third_pound_price : ℝ)
  (h1 : discount_pct = 0.40) (h2 : one_third_pound_price = 5) : 
  R = 37.5 :=
by 
  have h := (one_third_pound_price = 5) 
    have h₁ := (discount_pct = 0.40)
  sorry

end bakery_cake_price_l154_154133


namespace probability_of_even_distinct_digits_l154_154916

noncomputable def probability_even_distinct_digits : ℚ :=
  let total_numbers := 9000
  let favorable_numbers := 2744
  favorable_numbers / total_numbers

theorem probability_of_even_distinct_digits : 
  probability_even_distinct_digits = 343 / 1125 :=
by
  sorry

end probability_of_even_distinct_digits_l154_154916


namespace area_of_triangle_PQR_l154_154132

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 5 }

-- Define the lines using their slopes and the point P
def line1 (x : ℝ) : ℝ := -x + 7
def line2 (x : ℝ) : ℝ := -2 * x + 9

-- Definitions of points Q and R, which are the x-intercepts
def Q : Point := { x := 7, y := 0 }
def R : Point := { x := 4.5, y := 0 }

-- Theorem statement
theorem area_of_triangle_PQR : 
  let base := 7 - 4.5
  let height := 5
  (1 / 2) * base * height = 6.25 := by
  sorry

end area_of_triangle_PQR_l154_154132


namespace number_smaller_than_neg3_exists_l154_154151

def numbers := [0, -1, -5, -1/2]

theorem number_smaller_than_neg3_exists : ∃ x ∈ numbers, x < -3 :=
by
  let x := -5
  have h : x ∈ numbers := by simp [numbers]
  have h_lt : x < -3 := by norm_num
  exact ⟨x, h, h_lt⟩ -- show that -5 meets the criteria

end number_smaller_than_neg3_exists_l154_154151


namespace volume_sum_eq_148_l154_154937

noncomputable def volume_within_one_unit (a b c : ℕ) : ℝ :=
  let parallelepiped_volume := a * b * c
  let smaller_parallelepipeds_volume := 2 * (1 * a * b) + 2 * (1 * a * c) + 2 * (1 * b * c)
  let quarter_cylinders_volume := (4 * π * a / 4) + (4 * π * b / 4) + (4 * π * c / 4)
  let octants_volume := 8 * (4 * π * 1 / 3 / 8)
  parallelepiped_volume + smaller_parallelepipeds_volume + quarter_cylinders_volume + octants_volume

theorem volume_sum_eq_148 : 
  let V := volume_within_one_unit 2 3 6
  ∃ (m n p : ℕ), (V = (m + n * π) / p) ∧ (Nat.coprime n p) ∧ (m + n + p = 148) := by 
  sorry

end volume_sum_eq_148_l154_154937


namespace greatest_possible_L_l154_154900

theorem greatest_possible_L :
  ∃ L M N O P Q R S T U : ℕ,
    L < M ∧ M < N ∧ O < P ∧ P < Q ∧ R < S ∧ S < T ∧ T < U ∧
    L + M + N = 10 ∧
    (O, P, Q) ∈ ({(1, 3, 5), (3, 5, 7), (5, 7, 9)} : set (ℕ × ℕ × ℕ)) ∧
    (R, S, T, U) ∈ ({(0, 2, 4, 6), (2, 4, 6, 8)} : set (ℕ × ℕ × ℕ × ℕ)) ∧
    distinct [L, M, N, O, P, Q, R, S, T, U] ∧
    L = 1 :=
by
  apply Exists.intro 1 sorry
  apply Exists.intro 0 sorry
  apply Exists.intro 9 sorry
  apply Exists.intro 3 sorry
  apply Exists.intro 5 sorry
  apply Exists.intro 7 sorry
  apply Exists.intro 2 sorry
  apply Exists.intro 4 sorry
  apply Exists.intro 6 sorry
  apply Exists.intro 8 sorry
  split
  sorry

end greatest_possible_L_l154_154900


namespace min_distance_AB_l154_154275

noncomputable def line_eqn (λ : ℝ) : ℝ × ℝ → Prop :=
  λ p => λ * p.1 - p.2 - 4 * λ + 3 = 0

def circle_eqn (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 - 6 * p.1 - 8 * p.2 + 21 = 0

theorem min_distance_AB (λ : ℝ) :
  let A B : ℝ × ℝ := (λ, 0), (-λ, 0) in -- hypothetical points A and B for type checking
  let distance := λ p q : ℝ × ℝ => real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) in
  (line_eqn λ A ∧ circle_eqn A) →
  (line_eqn λ B ∧ circle_eqn B) →
  distance A B = 2 * real.sqrt 2 := 
sorry

end min_distance_AB_l154_154275


namespace correct_mark_value_l154_154903

variable (num_pupils : ℕ) (wrong_mark correct_mark : ℕ) (increase_in_average : ℕ)

-- Given conditions
def num_pupils_value : Prop := num_pupils = 40
def wrong_mark_value : Prop := wrong_mark = 83
def increase_in_average_value : Prop := increase_in_average = (num_pupils / 2)

-- Mathematical statement
theorem correct_mark_value
  (h1 : num_pupils_value)
  (h2 : wrong_mark_value)
  (h3 : increase_in_average_value) :
  wrong_mark - correct_mark = increase_in_average ↔ correct_mark = 63 :=
begin
  sorry
end

end correct_mark_value_l154_154903


namespace problem_part1_problem_part2_l154_154655

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (m n x : ℝ) : ℝ :=
  (m - 3^x) / (n + 3^x)

theorem problem_part1 (m n : ℝ) :
  is_odd_function (f m n) → m = 1 ∧ n = 1 :=
by
  sorry

theorem problem_part2 (k : ℝ) :
  (∃ t ∈ set.Icc 0 4, f 1 1 (k - 2*t^2) + f 1 1 (4*t - 2*t^2) < 0) →
  k > -1 :=
by
  sorry

end problem_part1_problem_part2_l154_154655


namespace solution_interval_l154_154572

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l154_154572


namespace shelves_needed_l154_154499

theorem shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (shelves : ℕ) :
  total_books = 34 →
  books_taken = 7 →
  books_per_shelf = 3 →
  remaining_books = total_books - books_taken →
  shelves = remaining_books / books_per_shelf →
  shelves = 9 :=
by
  intros h_total h_taken h_per_shelf h_remaining h_shelves
  rw [h_total, h_taken, h_per_shelf] at *
  sorry

end shelves_needed_l154_154499


namespace sum_series_eq_l154_154932

noncomputable def sum_series : ℕ → ℚ :=
λ n, ∑' n, (5 * n - 2) / 3^n

theorem sum_series_eq : sum_series 1 = 11 / 4 := 
sorry

end sum_series_eq_l154_154932


namespace union_complement_eq_l154_154775

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l154_154775


namespace delta_property_l154_154213

noncomputable def Δ (x y : ℝ) : ℝ := sorry

theorem delta_property : 7 Δ 3 = 36 :=
begin
  sorry
end

end delta_property_l154_154213


namespace alyosha_possible_l154_154144

theorem alyosha_possible (current_date : ℕ) (day_before_yesterday_age current_year_age next_year_age : ℕ) : 
  (next_year_age = 12 ∧ day_before_yesterday_age = 9 ∧ current_year_age = 12 - 1)
  → (current_date = 1 ∧ current_year_age = 11 → (∃ bday : ℕ, bday = 31)) := 
by
  sorry

end alyosha_possible_l154_154144


namespace monkey_count_l154_154127

theorem monkey_count (piles_1 piles_2 hands_1 hands_2 bananas_1_per_hand bananas_2_per_hand total_bananas_per_monkey : ℕ) 
  (h1 : piles_1 = 6) 
  (h2 : piles_2 = 4) 
  (h3 : hands_1 = 9) 
  (h4 : hands_2 = 12) 
  (h5 : bananas_1_per_hand = 14) 
  (h6 : bananas_2_per_hand = 9) 
  (h7 : total_bananas_per_monkey = 99) : 
  (piles_1 * hands_1 * bananas_1_per_hand + piles_2 * hands_2 * bananas_2_per_hand) / total_bananas_per_monkey = 12 := 
by 
  sorry

end monkey_count_l154_154127


namespace jasons_tip_l154_154744

theorem jasons_tip :
  ∀ (meal_cost : ℝ) (tax_rate : ℝ) (total_paid : ℝ),
    meal_cost = 15 ∧ tax_rate = 0.2 ∧ total_paid = 20 →
    total_paid - (meal_cost + meal_cost * tax_rate) = 2 :=
by
  intros meal_cost tax_rate total_paid h,
  have h1 : meal_cost = 15 := h.1,
  have h2 : tax_rate = 0.2 := h.2.1,
  have h3 : total_paid = 20 := h.2.2,
  sorry

end jasons_tip_l154_154744


namespace train_speed_l154_154869

theorem train_speed (L : ℝ) (T : ℝ) (hL : L = 200) (hT : T = 20) :
  L / T = 10 := by
  rw [hL, hT]
  norm_num
  done

end train_speed_l154_154869


namespace smallest_n_exists_l154_154976

theorem smallest_n_exists (n : ℕ) (hpos : n > 0) : 
  (∃ (x : Fin n → ℝ), (∑ i, x i) = 2000 ∧ (∑ i, (x i)^4) = 3200000) ↔ n = 50 := by
sorry

end smallest_n_exists_l154_154976


namespace sum_of_sines_l154_154489

theorem sum_of_sines (a : ℝ) (n : ℕ) (hn : n ≥ 2) :
  (∑ k in Finset.range (n-1), Real.sin (a * (k+1))) = 
  (Real.cos (a / 2) - Real.cos ((n - 0.5) * a)) / (2 * Real.sin (a / 2)) :=
sorry

end sum_of_sines_l154_154489


namespace affirmative_answers_count_l154_154840

def num_girls := 30
def num_red_dresses := 13
def num_blue_dresses := 17

def circle_condition (position: ℕ) (colors: Fin num_girls → Bool) : Bool :=
  let left := (position + num_girls - 1) % num_girls
  let right := (position + 1) % num_girls
  (colors left = colors position) ∧ (colors position = colors right)

theorem affirmative_answers_count 
  (colors: Fin num_girls → Bool)  -- False represents red, True represents blue
  (h_red_dresses: (Finset.filter (λ x, ¬ colors x) (Finset.range num_girls)).card = num_red_dresses)
  (h_blue_dresses: (Finset.filter (λ x, colors x) (Finset.range num_girls)).card = num_blue_dresses)
  : 
  Finset.card (Finset.filter (λ i, circle_condition i colors) (Finset.range num_girls)) = num_blue_dresses :=
  sorry

end affirmative_answers_count_l154_154840


namespace square_distance_B_to_center_l154_154517

theorem square_distance_B_to_center
  (r : ℝ) (A B C : ℝ × ℝ) (h_radius : r = sqrt 50)
  (h_distance_AB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36)
  (h_distance_BC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4)
  (h_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0)
  (h_A_on_circle : A.1^2 + A.2^2 = 50)
  (h_C_on_circle : C.1^2 + C.2^2 = 50) :
  B.1^2 + B.2^2 = 26 :=
by {
  sorry
}

end square_distance_B_to_center_l154_154517


namespace probability_no_defective_pens_l154_154710

theorem probability_no_defective_pens
  (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ)
  (h_total : total_pens = 10)
  (h_defective : defective_pens = 2)
  (h_selected : selected_pens = 2) :
  let non_defective_pens := total_pens - defective_pens in
  let prob_first_non_defective := (non_defective_pens : ℚ) / total_pens in
  let prob_second_non_defective := ((non_defective_pens - 1) : ℚ) / (total_pens - 1) in
  prob_first_non_defective * prob_second_non_defective = 28 / 45 :=
by
  sorry

end probability_no_defective_pens_l154_154710


namespace prove_problem_statement_l154_154310

variable {A B C : Type}
variables (a b c : ℝ)
variables (cosB : ℝ)
variables (BA BC : ℝ)
variables (triangle_ABC : ℝ)

noncomputable def problem_statement : Prop :=
  a > c ∧
  BA = 2 ∧
  cosB = 1 / 3 ∧
  b = 3 ∧
  (a, c) = (3, 2) ∧
  cosB * cosB.sqrt (1 - cosB^2) + sinB * sinC sqrt (1 - $sinC^2) = 23 / 27 

theorem prove_problem_statement :
  problem_statement A B C a b c cosB BA BC :=
  sorry

end prove_problem_statement_l154_154310


namespace regression_analysis_statements_l154_154094

theorem regression_analysis_statements :
  (A_correct ∧ B_correct ∧ C_correct ∧ ¬D_correct) :=
begin
  -- Definitions of the statements based on conditions from part (a)
  let A_correct := ∀ r, (|r| ≤ 1) → 
                    (|r| = 1 → ∃ x y, (∀ a b, y = a * x → a = r) ∧ (0 < |r| < 1 → ∃ u v, y = u * v ∧ 0 < u < 1)),
  let B_correct := ∀ x y (n : ℕ),
                    let mean := λ xs, (list.sum xs) / (list.length xs) in
                    let _ := (∀ b a u, list.length x = n ∧ list.length y = n →
                                (∀ xi yi, list.member xi x ∧ list.member yi y →
                                  yi = b * xi + a) →
                                (mean x, mean y) ∈ set_of y = b * (mean x) + a),
  let C_correct := ∀ (residuals : list ℝ), residuals.nth 0 = (list.sum (residuals.map (λ r, r * r)) / residuals.length),
  let D_correct := ∀ k (K2 : random_variable),
                    let relation := λ X Y, ∑ x y, K2 x y ->
                                    K2.values < 1 / (metric_space.dist X Y) in
                    ¬ relation categorical_var1 categorical_var2,

  -- Proving that A_correct, B_correct, C_correct are true and D_correct is false
  split,
  { intros r h1 h2,
    sorry }, -- Proof for A_correct
  { intros x y n mean b a u h1 h2,
    sorry }, -- Proof for B_correct
  { intros residuals,
    sorry }, -- Proof for C_correct
  { intros k K2 relation h1 h2 h3,
    sorry } -- Proof for D_correct
end

end regression_analysis_statements_l154_154094


namespace intersection_M_N_l154_154361

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l154_154361


namespace bear_laps_in_one_hour_l154_154105

def lake_perimeter : ℝ := 1000
def salmon1_speed : ℝ := 500  -- in meters per minute
def salmon2_speed : ℝ := 750  -- in meters per minute
def bear_speed : ℝ := 200     -- in meters per minute

theorem bear_laps_in_one_hour : 
  ∀ (t : ℝ), t = 60 → 
  (bear_speed * t / lake_perimeter) = 7 :=
by
  intro t ht
  rw ht
  have h1 : bear_speed * 60 = 12000 := by norm_num
  rw h1
  have h2 : 12000 / lake_perimeter = 7 := by norm_num
  rw h2
  exact h2

end bear_laps_in_one_hour_l154_154105


namespace smallest_integer_with_16_divisors_l154_154083

theorem smallest_integer_with_16_divisors : 
  ∃ (n : ℕ), (∃ (p_1 p_2 p_3 : ℕ) (a_1 a_2 a_3 : ℕ), 
  (p_1 = 2 ∧ p_2 = 3 ∧ a_1 = 3 ∧ a_2 = 3 ∧ n = p_1 ^ a_1 * p_2 ^ a_2) ∧
  (∀ m, m > 0 → (∃ b1 b2 ..., m has exactly 16 positive divisors) → 216 ≤ m)) := 
sorry

end smallest_integer_with_16_divisors_l154_154083


namespace angle_C_is_pi_over_2_l154_154309

theorem angle_C_is_pi_over_2
  (a b : ℝ) (angle_A : ℝ)
  (h_a : a = 3)
  (h_b : b = sqrt 3)
  (h_angle_A : angle_A = π / 3) :
  let angle_B := asin (1/2)
  let angle_C := π - angle_A - angle_B
  in angle_C = π / 2 :=
by
  sorry

end angle_C_is_pi_over_2_l154_154309


namespace Angela_height_is_157_l154_154160

variable (height_Amy height_Helen height_Angela : ℕ)

-- The conditions
axiom h_Amy : height_Amy = 150
axiom h_Helen : height_Helen = height_Amy + 3
axiom h_Angela : height_Angela = height_Helen + 4

-- The proof to show Angela's height is 157 cm
theorem Angela_height_is_157 : height_Angela = 157 :=
by
  rw [h_Amy] at h_Helen
  rw [h_Helen] at h_Angela
  exact h_Angela

end Angela_height_is_157_l154_154160


namespace find_radius_second_circle_l154_154123

noncomputable def radius_circle (r R : ℝ) (AB : ℝ) : Prop :=
AB = 4 ∧ r = 2 ∧ ∃ (CD : ℝ), 2 * real.sqrt (r * R) = CD ∧ (CD = AB)

theorem find_radius_second_circle : 
  radius_circle 2 8 4 := 
by 
  sorry

end find_radius_second_circle_l154_154123


namespace range_of_solutions_l154_154576

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l154_154576


namespace parallelogram_angle_H_l154_154320

theorem parallelogram_angle_H (F H : ℝ) (h1 : F = 125) (h2 : F + H = 180) : H = 55 :=
by
  have h3 : H = 180 - F := by linarith
  rw [h1] at h3
  rw [h3]
  norm_num

end parallelogram_angle_H_l154_154320


namespace find_coefficients_l154_154826

open Polynomial

def polynomial (a b : ℕ) := a * X^4 + b * X^3 + 40 * X^2 - 20 * X + 8

def factor := 2 * X^2 - 3 * X + 2

theorem find_coefficients :
  ∃ a b : ℕ, polynomial a b = polynomial 112 (-152) ∧ polynomial a b % factor = 0 :=
sorry

end find_coefficients_l154_154826


namespace tricia_age_is_5_l154_154848

theorem tricia_age_is_5 :
  (∀ Amilia Yorick Eugene Khloe Rupert Vincent : ℕ,
    Tricia = 5 ∧
    (3 * Tricia = Amilia) ∧
    (4 * Amilia = Yorick) ∧
    (2 * Eugene = Yorick) ∧
    (Eugene / 3 = Khloe) ∧
    (Khloe + 10 = Rupert) ∧
    (Vincent = 22)) → 
  Tricia = 5 :=
by
  sorry

end tricia_age_is_5_l154_154848


namespace problem_A_problem_B_problem_C_problem_D_l154_154623

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := sqrt 2 * sin (m * x + n)

theorem problem_A (m : ℝ) (n : ℝ) : 
  ¬ (∀ x : ℝ, f (x + 2 * π) m n = f x m n) :=
sorry

theorem problem_B (n : ℝ) : 
  ∃ m : ℝ, ∀ x : ℝ, f (x + 1) m n = f x m n :=
sorry

theorem problem_C (m : ℝ) (n : ℝ) : 
  ¬ (∀ x : ℝ, f (-x) m n ≠ f x m n) :=
sorry

theorem problem_D (n : ℝ) : 
  ∃ m : ℝ, ∀ x : ℝ, f (-x) m n = -f x m n :=
sorry

end problem_A_problem_B_problem_C_problem_D_l154_154623


namespace largest_integer_divisible_example_1748_largest_n_1748_l154_154855

theorem largest_integer_divisible (n : ℕ) (h : (n + 12) ∣ (n^3 + 160)) : n ≤ 1748 :=
by
  sorry

theorem example_1748 : 1748^3 + 160 = 1760 * 3045738 :=
by
  sorry

theorem largest_n_1748 (n : ℕ) (h : 1748 ≤ n) : (n + 12) ∣ (n^3 + 160) :=
by
  sorry

end largest_integer_divisible_example_1748_largest_n_1748_l154_154855


namespace constant_S13_l154_154621

noncomputable def S (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem constant_S13 (a d p : ℝ) 
  (h : a + a + 3 * d + a + 7 * d = p) : 
  S a d 13 = 13 * p / 18 :=
by
  unfold S
  sorry

end constant_S13_l154_154621


namespace chord_length_range_l154_154299

open Real

def chord_length_ge (t : ℝ) : Prop :=
  let r := sqrt 8
  let l := (4 * sqrt 2) / 3
  let d := abs t / sqrt 2
  let s := l / 2
  s ≤ sqrt (r^2 - d^2)

theorem chord_length_range (t : ℝ) : chord_length_ge t ↔ -((8 * sqrt 2) / 3) ≤ t ∧ t ≤ (8 * sqrt 2) / 3 :=
by
  sorry

end chord_length_range_l154_154299


namespace smallest_integer_with_16_divisors_l154_154076

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (number_of_divisors m = 16) → (m ≥ n)) ∧ (n = 120) :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154076


namespace partition_complete_graph_into_triangles_l154_154395

theorem partition_complete_graph_into_triangles (n : ℕ) :
  ∃ (partition : Multiset (Cycle 3)), ∀ (G : Graph (Fin (3^n))),
    G = completeGraph (Fin (3^n)) → edges G = Multiset.bind partition Cycle.edges :=
sorry

end partition_complete_graph_into_triangles_l154_154395


namespace unique_positive_integer_n_l154_154173

theorem unique_positive_integer_n (n : ℕ) :
  (∑ k in range(n-2), (k + 3) * 3^(k + 2)) = 3^(n+8)
  → n = 2188 :=
by
  sorry

end unique_positive_integer_n_l154_154173


namespace unique_sums_of_subsets_l154_154955

theorem unique_sums_of_subsets :
  ∃ (a : Fin 8 → ℕ), 
  (∀ i : Fin 8, 0 < a i ∧ a i < 100) ∧
  (∀ s₁ s₂ : Finset (Fin 8), s₁ ≠ s₂ → ∑ i in s₁, a i ≠ ∑ i in s₂, a i) ∧
  a = ![3, 6, 12, 24, 48, 95, 96, 97] := 
begin
  sorry,
end

end unique_sums_of_subsets_l154_154955


namespace parallelogram_angle_H_l154_154321

theorem parallelogram_angle_H (F H : ℝ) (h1 : F = 125) (h2 : F + H = 180) : H = 55 :=
by
  have h3 : H = 180 - F := by linarith
  rw [h1] at h3
  rw [h3]
  norm_num

end parallelogram_angle_H_l154_154321


namespace polyhedron_not_necessarily_pyramid_l154_154901

-- Define the conditions for the polyhedron

structure Polyhedron where
  faces : Type
  is_polygon : Prop             -- one face is a polygon
  other_faces_are_triangles : Prop   -- the rest of the faces are triangles

-- Define the pyramid
def is_pyramid (P : Polyhedron) : Prop :=
  ∃ v, ∀ f ∈ P.faces, f = polygon ∨ (∃ t, t = triangle ∧ t.share_vertex v)

-- The theorem we want to frame and prove
theorem polyhedron_not_necessarily_pyramid
  (P : Polyhedron) (h1 : P.is_polygon) (h2 : P.other_faces_are_triangles) :
  ¬is_pyramid P := 
sorry

end polyhedron_not_necessarily_pyramid_l154_154901


namespace solve_equation_l154_154181

theorem solve_equation (a b : ℕ) : 
  (a^2 = b * (b + 7) ∧ a ≥ 0 ∧ b ≥ 0) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end solve_equation_l154_154181


namespace binary_to_decimal_l154_154421

theorem binary_to_decimal (h : 110011₂ = 51): true := sorry

end binary_to_decimal_l154_154421


namespace range_of_solutions_l154_154569

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l154_154569


namespace gold_copper_ratio_l154_154287

theorem gold_copper_ratio (G C : ℕ) (h : 19 * G + 9 * C = 17 * (G + C)) : G = 4 * C :=
by
  sorry

end gold_copper_ratio_l154_154287


namespace integral_f_l154_154264

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then sqrt (1 - x^2) else exp x

theorem integral_f :
  ∫ x in -1..2, f x = (Real.pi / 2) + exp 2 - exp 1 :=
sorry

end integral_f_l154_154264


namespace percentage_error_in_area_l154_154553

theorem percentage_error_in_area (S : ℝ) (h : S > 0) :
  let S' := S * 1.06
  let A := S^2
  let A' := (S')^2
  (A' - A) / A * 100 = 12.36 := by
  sorry

end percentage_error_in_area_l154_154553


namespace Shara_savings_l154_154530

theorem Shara_savings (P : ℝ) (d : ℝ) (paid : ℝ):
  d = 0.08 → paid = 184 → P = 200 → (P * (1 - d) = paid) → (P - paid = 16) :=
by
  intros hd hpaid hP heq
  -- It follows from the conditions given
  sorry

end Shara_savings_l154_154530


namespace stripe_area_equals_expected_l154_154128

-- Definition of the conditions
def diameter : ℝ := 40
def width : ℝ := 4
def revolutions : ℝ := 1.5

-- Definition of the expected answer for the area of the stripe
def expected_area : ℝ := 240 * Real.pi

-- The Lean statement asserting the expected area under given conditions
theorem stripe_area_equals_expected : 
  let circumference := Real.pi * diameter
  let length := revolutions * circumference
  let area := width * length
  area = expected_area :=
by
  sorry

end stripe_area_equals_expected_l154_154128


namespace number_of_australians_l154_154157

-- Conditions are given here as definitions
def total_people : ℕ := 49
def number_americans : ℕ := 16
def number_chinese : ℕ := 22

-- Goal is to prove the number of Australians is 11
theorem number_of_australians : total_people - (number_americans + number_chinese) = 11 := by
  sorry

end number_of_australians_l154_154157


namespace count_Z_functions_l154_154889

def is_Z_function (f : ℝ → ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

def func1 (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + x - 2
def func2 (x : ℝ) : ℝ := 2*x - (Real.sin x + Real.cos x)
def func3 (x : ℝ) : ℝ := Real.exp x + 1
noncomputable def func4 (x : ℝ) : ℝ := if x = 0 then 0 else Real.log (Real.abs x)

theorem count_Z_functions : (is_Z_function func1) ∧ (is_Z_function func2) ∧ (is_Z_function func3) ∧ ¬(is_Z_function func4) :=
begin
    sorry,
end

end count_Z_functions_l154_154889


namespace f_is_odd_and_periodic_l154_154429

def f (x : ℝ) : ℝ := 1 - 2 * sin(x - (π / 4)) ^ 2

theorem f_is_odd_and_periodic :
  (∀ x, f (-x) = -f x) ∧ (∃ p > 0, (∀ x, f (x + p) = f x) ∧ (p ≤ π)) :=
sorry

end f_is_odd_and_periodic_l154_154429


namespace decimal_6_to_binary_is_110_l154_154468

def decimal_to_binary (n : ℕ) : ℕ :=
  -- This is just a placeholder definition. Adjust as needed for formalization.
  sorry

theorem decimal_6_to_binary_is_110 :
  decimal_to_binary 6 = 110 := 
sorry

end decimal_6_to_binary_is_110_l154_154468


namespace part1_solution_part2_solution_l154_154274

-- Part (1)
theorem part1_solution (x : ℝ) : (|x - 2| + |x - 1| ≥ 2) ↔ (x ≥ 2.5 ∨ x ≤ 0.5) := sorry

-- Part (2)
theorem part2_solution (a : ℝ) (h : a > 0) : (∀ x, |a * x - 2| + |a * x - a| ≥ 2) → a ≥ 4 := sorry

end part1_solution_part2_solution_l154_154274


namespace sum_of_solutions_eq_8_l154_154478

theorem sum_of_solutions_eq_8 : 
  (∑ x in {x : ℝ | |x^2 - 16x + 68| = 4}.to_finset, x) = 8 :=
sorry

end sum_of_solutions_eq_8_l154_154478


namespace intervals_of_monotonicity_maximum_value_on_interval_l154_154630

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 * exp (-a * x)

theorem intervals_of_monotonicity (a : ℝ) (ha : 0 < a) :
  (∀ x : ℝ, 0 < x → x < 2 / a → f'.derivative x > 0) ∧
  (∀ x : ℝ, x < 0 ∨ 2 / a < x → f'.derivative x < 0) :=
sorry

theorem maximum_value_on_interval (a : ℝ) (ha : 0 < a) :
  (a < 1 → ∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ 4 * exp (-2 * a)) ∧
  (1 ≤ a ∧ a ≤ 2 → ∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ (4 / a^2) * exp (-2)) ∧
  (a > 2 → ∀ x, 1 ≤ x ∧ x ≤ 2 → f x a ≤ exp (-a)) :=
sorry

end intervals_of_monotonicity_maximum_value_on_interval_l154_154630


namespace probability_heads_at_least_9_of_12_flips_l154_154007

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154007


namespace measure_of_angle_H_in_parallelogram_l154_154323

theorem measure_of_angle_H_in_parallelogram (H : Type) [AddGroup H] {E F G : H}
  (h_parallelogram : parallelogram E F G H)
  (angle_F : measure_of_angle F = 125) :
  measure_of_angle H = 55 := 
sorry

end measure_of_angle_H_in_parallelogram_l154_154323


namespace question1_question2_l154_154837

-- Define the conditions and outputs as types and functions where necessary
-- Question 1
theorem question1 (boys girls : ℕ) (h_b : boys = 5) (h_g : girls = 3) : 
  (∃ (arrangements : ℕ), arrangements = factorial (5 + 1) * factorial 3) :=
by
  use 4320
  sorry

-- Question 2
theorem question2 (boys girls : ℕ) (h_b : boys = 5) (h_g : girls = 3) : 
  (∃ (arrangements : ℕ), arrangements = (choose 3 2) * (choose 5 3) * factorial 5) :=
by
  use 3600
  sorry

end question1_question2_l154_154837


namespace polar_equations_proof_l154_154332

noncomputable def polar_equations_max_value :=
  let C1_cartesian : (ℝ × ℝ) → Prop := λ p, p.1 + p.2 = 4
  let C2_parametric : ℝ → ℝ × ℝ := λ θ, (1 + Real.cos θ, Real.sin θ)
  let α_interval : Set.Ioo (-Real.pi / 4) (Real.pi / 2)
  let polar_coord_eqn_C1 := (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 4
  let polar_coord_eqn_C2 := (ρ θ : ℝ), ρ = 2 * Real.cos θ
  let ρ1 := λ α : ℝ, 4 / (Real.cos α + Real.sin α)
  let ρ2 := λ α : ℝ, 2 * Real.cos α
  let fraction := λ α : ℝ, 1 / 4 * (Real.sqrt 2 * Real.cos (2 * α - Real.pi / 4) + 1)
  let maximum_value := fraction (Real.pi / 8)
  maximum_value = 1 / 4 * (Real.sqrt 2 + 1)

theorem polar_equations_proof :
  polar_equations_max_value := by
  sorry

end polar_equations_proof_l154_154332


namespace ab_value_l154_154297

theorem ab_value (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 29) : a * b = 2 :=
by
  -- proof will be provided here
  sorry

end ab_value_l154_154297


namespace domain_of_g_l154_154601

theorem domain_of_g (x : ℝ) : 
  (g(x) = Mathlib.tan(Mathlib.arccos(2 * x))) → (x ∈ Set.Icc (-0.5 : ℝ) 0 ∪ Set.Icc 0 0.5) :=
by
  have h₁ : -1 ≤ 2 * x ∧ 2 * x ≤ 1 := sorry
  have h₂ : Mathlib.arccos(2 * x) ≠ (real.pi / 2) := sorry
  sorry

end domain_of_g_l154_154601


namespace sequence_t_inequality_l154_154633

theorem sequence_t_inequality (t : ℝ) : 
  (∀ n : ℕ, n > 0 → let a : ℕ → ℝ := λ n, if n = 0 then 0 else 1 / (n * (n + 1)) in
  (4 / (n^2) + 1 / n + t * a n) ≥ 0) →
  t ≥ -9 := 
by
  sorry

end sequence_t_inequality_l154_154633


namespace weekly_coffee_cost_l154_154383

-- Definitions and conditions
def cups_per_day := 2
def ounces_per_cup := 1.5
def bag_cost := 8
def ounces_per_bag := 10.5
def milk_per_week := 1 / 2
def milk_cost_per_gallon := 4

-- Compute the total weekly cost
theorem weekly_coffee_cost :
  let coffee_per_week := cups_per_day * ounces_per_cup * 7
  let bags_per_week := coffee_per_week / ounces_per_bag
  let coffee_cost_per_week := bags_per_week * bag_cost
  let milk_cost_per_week := milk_per_week * milk_cost_per_gallon
  coffee_cost_per_week + milk_cost_per_week = 18 := by
  sorry

end weekly_coffee_cost_l154_154383


namespace smallest_integer_with_16_divisors_l154_154050

-- Define prime factorization and the function to count divisors
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def number_of_divisors (n : ℕ) : ℕ :=
  (prime_factorization n).foldr (λ ⟨_, a⟩ acc, acc * (a + 1)) 1

-- Main theorem stating the smallest positive integer with exactly 16 divisors is 210
theorem smallest_integer_with_16_divisors : 
  ∃ n, n > 0 ∧ number_of_divisors n = 16 ∧ ∀ m, m > 0 ∧ number_of_divisors m = 16 → n ≤ m :=
begin
  use 210,
  split,
  { -- Prove 210 > 0
    exact nat.zero_lt_succ _,
  },
  split,
  { -- Prove number_of_divisors 210 = 16
    sorry,
  },
  { -- Prove minimality
    intros m hm1 hm2,
    sorry,
  }
end

end smallest_integer_with_16_divisors_l154_154050


namespace union_complement_eq_l154_154783

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l154_154783


namespace curves_with_SCT_l154_154705

def f1 (x y : ℝ) : Prop := x^2 - y^2 = 1
def f2 (x y : ℝ) : Prop := y = x^2 - |x|
def f3 (x y : ℝ) : Prop := y = 3 * sin x + 4 * cos x
def f4 (x y : ℝ) : Prop := |x| + 1 = √(4 - y^2)

def self_common_tangent (f : ℝ → ℝ → Prop) : Prop := 
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
  (∃ (m b : ℝ), ∀ (p : ℝ × ℝ), f p.1 p.2 → p.2 = m * p.1 + b) ∧
  ((p1.2 = m * p1.1 + b) ∧ (p2.2 = m * p2.1 + b))

theorem curves_with_SCT :
  ¬ self_common_tangent f1 ∧ 
  self_common_tangent f2 ∧ 
  self_common_tangent f3 ∧ 
  ¬ self_common_tangent f4 := 
sorry

end curves_with_SCT_l154_154705


namespace no_valid_labeling_of_red_centers_l154_154875

noncomputable def red_centers_available : ℕ := 96

theorem no_valid_labeling_of_red_centers :
  ∀ (C : fin red_centers_available → ℝ × ℝ),
    (∀ i, dist (C i) (C ((i + 1) % red_centers_available)) = real.sqrt 13) →
    ¬ (∃ O : ℝ × ℝ, (∀ i, C i = 2 * O - C ((red_centers_available - 1 - i) % red_centers_available))) :=
begin
  sorry
end

end no_valid_labeling_of_red_centers_l154_154875


namespace min_eq_neg_one_implies_x_eq_two_l154_154370

theorem min_eq_neg_one_implies_x_eq_two (x : ℝ) (h : min (2*x - 5) (x + 1) = -1) : x = 2 :=
sorry

end min_eq_neg_one_implies_x_eq_two_l154_154370


namespace sum_of_coeffs_in_expansion_l154_154724

theorem sum_of_coeffs_in_expansion (n : ℕ) : 
  (1 - 2 : ℤ)^n = (-1 : ℤ)^n :=
by
  sorry

end sum_of_coeffs_in_expansion_l154_154724


namespace coefficient_of_x3_l154_154300

theorem coefficient_of_x3 (n : ℕ) (h1 : ∀ n, (∃ r, 3 * r - n = 0 ∧ nat.binomial n r * (-1)^r = 15) → n = 6) :
  (∀ r, 3 * r - 6 = 3 → r = 3) → - nat.binomial 6 3 = -20 :=
by
  intro h2
  specialize h2 3
  sorry -- Proof is omitted as per the instructions

end coefficient_of_x3_l154_154300


namespace sum_of_squares_inequality_l154_154469

theorem sum_of_squares_inequality {n : ℕ} {x y z : Fin n → ℝ}
  (h1 : ∀ i : Fin n, x i ≥ x (i.succ) ∨ i = Fin.last n)
  (h2 : ∀ i : Fin n, y i ≥ y (i.succ) ∨ i = Fin.last n)
  (h3 : ∃ σ : Equiv.Perm (Fin n), ∀ i : Fin n, z (σ i) = y i) :
  (∑ i, (x i - y i) ^ 2) ≤ (∑ i, (x i - z i) ^ 2) := by
  sorry

end sum_of_squares_inequality_l154_154469


namespace isosceles_triangle_angle_l154_154993

theorem isosceles_triangle_angle (
  (A B C : Type)
  [EuclideanGeometry A B C]
  (H_iso : isosceles_triangle A B C)
  (H_base_angles : ∠ABC = 50 ∧ ∠ACB = 50)
  (D E P : Type)
  [lies_on D (line BC)]
  [lies_on E (line AC)]
  [intersection AD BE = P]
  (H_angles_given : ∠ABE = 30 ∧ ∠BAD = 50)
) : ∠BED = 40 :=
sorry

end isosceles_triangle_angle_l154_154993


namespace probability_of_forming_triangle_by_selecting_three_segments_from_five_l154_154857

-- Define the lengths of the five line segments
def lengths : List ℕ := [3, 4, 5, 6, 7]

-- Define a function to check if three sides can form a triangle
def forms_triangle (a b c : ℕ) : Bool :=
  (a + b > c) && (a + c > b) && (b + c > a)

-- Get all combinations of three lengths from the list
def all_combinations : List (ℕ × ℕ × ℕ) :=
  [(3, 4, 5), (3, 4, 6), (3, 4, 7), (3, 5, 6), (3, 5, 7), (3, 6, 7), (4, 5, 6), (4, 5, 7), (4, 6, 7), (5, 6, 7)]

-- Calculate the count of valid combinations (that form a triangle)
def valid_combinations : List (ℕ × ℕ × ℕ) :=
  all_combinations.filter (λ (abc : ℕ × ℕ × ℕ), forms_triangle abc.1 abc.2 abc.3)

-- Define the probability as a ratio of valid combinations to total combinations
def probability : ℚ :=
  valid_combinations.length / all_combinations.length

-- Theorem stating the required probability
theorem probability_of_forming_triangle_by_selecting_three_segments_from_five :
  probability = 9 / 10 := by
  sorry

end probability_of_forming_triangle_by_selecting_three_segments_from_five_l154_154857


namespace total_weight_of_shells_l154_154341

noncomputable def initial_weight : ℝ := 5.25
noncomputable def weight_large_shell_g : ℝ := 700
noncomputable def grams_per_pound : ℝ := 453.592
noncomputable def additional_weight : ℝ := 4.5

/-
We need to prove:
5.25 pounds (initial weight) + (700 grams * (1 pound / 453.592 grams)) (weight of large shell in pounds) + 4.5 pounds (additional weight) = 11.293235835 pounds
-/
theorem total_weight_of_shells :
  initial_weight + (weight_large_shell_g / grams_per_pound) + additional_weight = 11.293235835 := by
    -- Proof will be inserted here
    sorry

end total_weight_of_shells_l154_154341


namespace number_of_dogs_l154_154315

theorem number_of_dogs 
  (d c b : Nat) 
  (ratio : d / c / b = 3 / 7 / 12) 
  (total_dogs_and_bunnies : d + b = 375) :
  d = 75 :=
by
  -- Using the hypothesis and given conditions to prove d = 75.
  sorry

end number_of_dogs_l154_154315


namespace cylinder_volume_l154_154620

def side_length : ℝ := 10
def radius : ℝ := side_length / 2
def height : ℝ := side_length
def volume (r h : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume : volume radius height = 250 * π := by
  sorry

end cylinder_volume_l154_154620


namespace cannot_form_set_l154_154089

-- Definitions based on the given conditions
def A : Set := {x : Type | x = "Table Tennis Player" ∧ x participates in "Hangzhou Asian Games"}
def B : Set := {x ∈ ℕ | x > 0 ∧ x < 5}
def C : Prop := False -- C cannot form a set
def D : Set := {x : Real | ¬ (x ∈ ℚ)}

-- Theorem stating which group cannot form a set
theorem cannot_form_set : (C = False) :=
by
  sorry

end cannot_form_set_l154_154089


namespace part_1_part_2_l154_154663

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem part_1 :
  ∀ x : ℝ,
    f(x) = 2 * Real.sin (2 * x + Real.pi / 6) := sorry

theorem part_2 :
  ∀ x : ℝ,
    x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2) →
    f(x) ∈ Set.Icc (-1) 2 := sorry

end part_1_part_2_l154_154663


namespace tournament_games_l154_154733

theorem tournament_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 5) : 
  (n * (n - 1) / 2) * k = 2175 := by
  sorry

end tournament_games_l154_154733


namespace AKM_ALM_sum_30_degrees_l154_154634

/-- Define the equilateral triangle and other conditions from the problem -/
open EuclideanGeometry

variables (A B C K L M : Point)
variables (hABC_equilateral : ∀ a b c : Point, equilateral a b c → ∠ B A C = 60)
variables (hDivideBC : dist B K = dist K L ∧ dist K L = dist L C)
variables (hM_ratio : ∃ a c, dist A M / dist M C = 1 / 2)

/-- Statement of the theorem. -/
theorem AKM_ALM_sum_30_degrees (hABC_equilateral : equilateral A B C)
  (hDivideBC : dist B K = dist K L ∧ dist K L = dist L C)
  (hM_ratio : dist A M / dist M C = 1 / 2) :
  ∠ A K M + ∠ A L M = 30 :=
sorry

end AKM_ALM_sum_30_degrees_l154_154634


namespace function_evaluation_l154_154664

def f (x : ℝ) : ℝ := 1 / (1 + 2^x)

theorem function_evaluation :
  f (-1/3) + f (-1) + f (0) + f (1) + f (1/3) = 5/2 := by
  sorry

end function_evaluation_l154_154664


namespace sum_of_areas_of_infinite_squares_l154_154528

/-- The side length of each square is halved and multiplied by sqrt(2) iteratively. 
    If the first square's side length is 4 cm, prove that the sum of the areas of
    all squares is 32 cm^2. -/
theorem sum_of_areas_of_infinite_squares :
  let side_length := 4
  let initial_area := side_length * side_length
  let ratio := 0.5 * Real.sqrt 2
  let area_sequence := (λ n : ℕ, initial_area * ratio ^ (2 * n))
  (∑' n, area_sequence n) = 32 :=
by
  sorry

end sum_of_areas_of_infinite_squares_l154_154528


namespace car_population_is_l154_154878

def fuel_consumption (car: Type) (model: Type) : Prop := 
  ∀ x: car, model x → Prop

def survey_contains (sample_size: ℕ) : Prop :=
  sample_size = 20

def population (car: Type) (model: Type) : Prop := 
  fuel_consumption car model

def sample_population (sample_size: ℕ) (car: Type) (model: Type) : Prop :=
  survey_contains sample_size ∧ fuel_consumption car model

theorem car_population_is 
  (car: Type) (model: Type)
  (h: ∀x: car, model x → Prop) 
  (hs: survey_contains 20) :
  population car model ↔ (fuel_consumption car model) :=
by {
  sorry
}

end car_population_is_l154_154878


namespace sum_of_smallest_and_second_smallest_l154_154839

-- Define the set of numbers
def numbers : Set ℕ := {10, 11, 12, 13}

-- Define the smallest and second smallest numbers
def smallest_number : ℕ := 10
def second_smallest_number : ℕ := 11

-- Prove the sum of the smallest and the second smallest numbers
theorem sum_of_smallest_and_second_smallest : smallest_number + second_smallest_number = 21 := by
  sorry

end sum_of_smallest_and_second_smallest_l154_154839


namespace probability_heads_at_least_9_of_12_flips_l154_154010

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154010


namespace sin_value_l154_154224

theorem sin_value (alpha : ℝ) (h1 : -π / 6 < alpha ∧ alpha < π / 6)
  (h2 : Real.cos (alpha + π / 6) = 4 / 5) :
  Real.sin (2 * alpha + π / 12) = 17 * Real.sqrt 2 / 50 :=
by
    sorry

end sin_value_l154_154224


namespace factorizable_polynomials_count_l154_154201

theorem factorizable_polynomials_count :
  (∑ a in finset.range 200, a + 1) = 10010 :=
begin
  -- proof omitted
  sorry
end

end factorizable_polynomials_count_l154_154201


namespace length_QI_l154_154736

theorem length_QI (P Q R I : Type) 
  (PQ PR QR : ℝ) 
  (PQ_eq : PQ = 39) 
  (PR_eq : PR = 36) 
  (QR_eq : QR = 15) 
  (incenter_I : is_incenter P Q R I) 
  : QI = 39 / Real.sqrt 90 := 
sorry

end length_QI_l154_154736


namespace least_positive_not_factorial_and_not_prime_factorial_l154_154042

noncomputable def least_positive_not_factorial_and_not_prime : ℕ :=
  62

theorem least_positive_not_factorial_and_not_prime_factorial : 
  ∀ n : ℕ, ¬ least_positive_not_factorial_and_not_prime ∣ (31.factorial) := 
by
  sorry

end least_positive_not_factorial_and_not_prime_factorial_l154_154042


namespace triangle_ABC1_is_right_angled_l154_154318

-- Definitions for right-angled triangle and projections in a plane
structure RightAngledTriangle (A B C : Type) :=
  (AB_right_angle : is_right_angle A B)
  (AB_in_plane : is_in_plane AB alpha)
  (C_outside_plane : is_outside_plane C alpha)
  (C_projection : projection(C, alpha) = C1)
  (C1_not_on_AB : C1 ∉ AB)

-- Theorem: Triangle ABC1 is right-angled
theorem triangle_ABC1_is_right_angled
  {A B C C1 : Type} {alpha : Plane}
  (h : RightAngledTriangle A B C) : is_right_angled_triangle A B C1 :=
by sorry

end triangle_ABC1_is_right_angled_l154_154318


namespace solution_interval_l154_154571

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l154_154571


namespace shuttle_speed_in_kph_l154_154102

def sec_per_min := 60
def min_per_hour := 60
def sec_per_hour := sec_per_min * min_per_hour
def speed_in_kps := 12
def speed_in_kph := speed_in_kps * sec_per_hour

theorem shuttle_speed_in_kph :
  speed_in_kph = 43200 :=
by
  -- No proof needed
  sorry

end shuttle_speed_in_kph_l154_154102


namespace polygon_with_same_center_circles_is_regular_l154_154126

def is_inscribed (P : polygon) (C : circle) := 
  P.has_inscribed_circle C

def is_circumscribed (P : polygon) (C : circle) := 
  P.has_circumscribed_circle C

def same_center (C₁ C₂ : circle) := 
  C₁.center = C₂.center

def is_convex (P : polygon) := 
  P.is_convex

def is_regular (P : polygon) := 
  P.is_regular

theorem polygon_with_same_center_circles_is_regular
  (P : polygon) (C₁ C₂ : circle)
  (h1 : is_convex P)
  (h2 : is_inscribed P C₁)
  (h3 : is_circumscribed P C₂)
  (h4 : same_center C₁ C₂) :
  is_regular P := 
sorry

end polygon_with_same_center_circles_is_regular_l154_154126


namespace father_current_age_l154_154804

theorem father_current_age (F S : ℕ) 
  (h₁ : F - 6 = 5 * (S - 6)) 
  (h₂ : (F + 6) + (S + 6) = 78) : 
  F = 51 := 
sorry

end father_current_age_l154_154804


namespace rectangle_x_is_18_l154_154241

-- Definitions for the conditions
def rectangle (a b x : ℕ) : Prop := 
  (a = 2 * b) ∧
  (x = 2 * (a + b)) ∧
  (x = a * b)

-- Theorem to prove the equivalence of the conditions and the answer \( x = 18 \)
theorem rectangle_x_is_18 : ∀ a b x : ℕ, rectangle a b x → x = 18 :=
by
  sorry

end rectangle_x_is_18_l154_154241


namespace incorrect_statement_C_l154_154767

def p : Prop := ∀ x, ¬ symm_y_axis (translate_left (sin (2 * x + π / 3)) (π / 6))
def q : Prop := ∀ x, x ∈ Ioo -1 (real∞) → ¬ increasing (abs (3^x - 1))

theorem incorrect_statement_C : ¬ (p ∨ q) := by
sorry

end incorrect_statement_C_l154_154767


namespace maximize_profit_l154_154304

variables (purchase_price selling_price_initial sales_volume_initial price_increase sales_volume_decrease : ℝ)
variables (profit_increase : ℝ → ℝ)

noncomputable def profit_function (x : ℝ) : ℝ :=
  let selling_price := selling_price_initial + price_increase * x in
  let sales_volume := sales_volume_initial - sales_volume_decrease * x in
  (selling_price - purchase_price) * sales_volume

theorem maximize_profit (h_purchase_price : purchase_price = 8)
                        (h_selling_price_initial : selling_price_initial = 10)
                        (h_sales_volume_initial : sales_volume_initial = 200)
                        (h_price_increase : price_increase = 0.5)
                        (h_sales_volume_decrease : sales_volume_decrease = 20)
                        (h_profit_increase : profit_function = λ x, (2 + x) * (200 - 20 * x)) :
  ∃ x : ℝ, (selling_price_initial + price_increase * x = 14) ∧ 
           (profit_function x = 720) := 
begin
  sorry
end

end maximize_profit_l154_154304


namespace smallest_M_exists_l154_154616

theorem smallest_M_exists : ∃ (M : ℕ), 
  (M > 0) ∧ 
  (M % 49 = 0) ∧
  (M + 1) % 9 = 0 ∧ 
  (M + 2) % 25 = 0 ∧
  (∀ N, (N > 0) → 
         ((N % 49 = 0 ∧ 
           (N + 1) % 9 = 0 ∧ 
           (N + 2) % 25 = 0) 
           → N ≥ M)) :=
begin
  use 98,
  split,
  { exact nat.succ_pos' 97, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  intros N N_pos div_cond,
  by_cases (N = 98),
  { rw h, exact le_refl 98, },
  { have : N ≥ 98 := sorry, -- proof elided
    exact this }
end

end smallest_M_exists_l154_154616


namespace find_q_interval_l154_154608

open Real

theorem find_q_interval : ∀ q : ℝ, (0 < q ∧ ∃ x : ℝ, x^2 - 8 * x + q < 0) → q ∈ set.Ioo 0 16 :=
by
  intros q h
  cases h with hqpos hx
  unfold set.Ioo
  split
  exact hqpos
  -- skipped proof
  sorry

end find_q_interval_l154_154608


namespace problem1_problem2_l154_154113

theorem problem1
  (x y : ℝ)
  (hx : x > 0) (hy : y > 0)
  (h : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

theorem problem2
  (n : ℕ)
  (hn : 0 < n) :
  sqrt (n + 1) - sqrt n > sqrt (n + 2) - sqrt (n + 1) :=
sorry

end problem1_problem2_l154_154113


namespace intersection_M_N_l154_154362

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l154_154362


namespace soda_difference_l154_154514

theorem soda_difference (diet_soda: ℕ) (regular_soda: ℕ) (h1: diet_soda = 19) (h2: regular_soda = 60) : regular_soda - diet_soda = 41 :=
by
  rw [h1, h2]
  norm_num
  sorry

end soda_difference_l154_154514


namespace tamika_always_greater_carlos_l154_154808

theorem tamika_always_greater_carlos :
  ∀ t c, t ∈ ({7 * 11, 7 * 14, 11 * 14} : set ℕ) → c ∈ ({2 + 4, 2 + 7, 4 + 7} : set ℕ) → t > c :=
by
  intros t c ht hc
  have tamika_mul_1 : 7 * 11 = 77 := rfl
  have tamika_mul_2 : 7 * 14 = 98 := rfl
  have tamika_mul_3 : 11 * 14 = 154 := rfl
  have carlos_add_1 : 2 + 4 = 6 := rfl
  have carlos_add_2 : 2 + 7 = 9 := rfl
  have carlos_add_3 : 4 + 7 = 11 := rfl
  cases ht with
  | or.inl h   => subst h; linarith
  | or.inr ht' => cases ht' with
    | or.inl h   => subst h; linarith
    | or.inr h   => subst h; linarith
  sorry

end tamika_always_greater_carlos_l154_154808


namespace time_ratio_A_to_B_l154_154908

theorem time_ratio_A_to_B (T_A T_B : ℝ) (hB : T_B = 36) (hTogether : 1 / T_A + 1 / T_B = 1 / 6) : T_A / T_B = 1 / 5 :=
by
  sorry

end time_ratio_A_to_B_l154_154908


namespace union_complement_eq_l154_154782

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l154_154782


namespace range_of_a_l154_154697

noncomputable def is_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f y < f x

def f (a x : ℝ) : ℝ := -x^2 + 2 * a * x
def g (a x : ℝ) : ℝ := a / (x + 1)

theorem range_of_a (a : ℝ) :
  (is_decreasing (f a) {x : ℝ | 1 ≤ x ∧ x ≤ 2} ∧ is_decreasing (g a) {x : ℝ | 1 ≤ x ∧ x ≤ 2}) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end range_of_a_l154_154697


namespace spherical_triangle_area_l154_154610

theorem spherical_triangle_area (R α β γ : ℝ) :
  α > 0 → β > 0 → γ > 0 → α + β + γ > π →
  (area_of_spherical_triangle R α β γ = R^2 * (α + β + γ - π)) :=
by
  sorry

end spherical_triangle_area_l154_154610


namespace probability_heads_at_least_9_of_12_flips_l154_154003

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154003


namespace math_problems_not_a_set_l154_154092

-- Define the conditions in Lean
def is_well_defined (α : Type) : Prop := sorry

-- Type definitions for the groups of objects
def table_tennis_players : Type := sorry
def positive_integers_less_than_5 : Type := sorry
def irrational_numbers : Type := sorry
def math_problems_2023_college_exam : Type := sorry

-- Defining specific properties of each group
def well_defined_table_tennis_players : is_well_defined table_tennis_players := sorry
def well_defined_positive_integers_less_than_5 : is_well_defined positive_integers_less_than_5 := sorry
def well_defined_irrational_numbers : is_well_defined irrational_numbers := sorry

-- The key property that math problems from 2023 college entrance examination cannot form a set.
theorem math_problems_not_a_set : ¬ is_well_defined math_problems_2023_college_exam := sorry

end math_problems_not_a_set_l154_154092


namespace prob_allergic_prescribed_l154_154911

def P (a : Prop) : ℝ := sorry

axiom P_conditional (A B : Prop) : P B > 0 → P (A ∧ B) = P A * P (B ∧ A) / P B

def A : Prop := sorry -- represent the event that a patient is prescribed Undetenin
def B : Prop := sorry -- represent the event that a patient is allergic to Undetenin

axiom P_A : P A = 0.10
axiom P_B_given_A : P (B ∧ A) / P A = 0.02
axiom P_B : P B = 0.04

theorem prob_allergic_prescribed : P (A ∧ B) / P B = 0.05 :=
by
  have h1 : P (A ∧ B) / P A = 0.10 * 0.02 := sorry -- using definition of P_A and P_B_given_A
  have h2 : P (A ∧ B) = 0.002 := sorry -- calculating the numerator P(B and A)
  exact sorry -- use the axiom P_B to complete the theorem

end prob_allergic_prescribed_l154_154911


namespace smallest_integer_with_16_divisors_l154_154074

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (number_of_divisors m = 16) → (m ≥ n)) ∧ (n = 120) :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154074


namespace number_of_n_l154_154971

theorem number_of_n (h1: n > 0) (h2: n ≤ 2000) (h3: ∃ m, 10 * n = m^2) : n = 14 :=
by sorry

end number_of_n_l154_154971


namespace no_integer_solutions_for_system_l154_154182

theorem no_integer_solutions_for_system :
  ∀ (y z : ℤ),
    (2 * y^2 - 2 * y * z - z^2 = 15) ∧ 
    (6 * y * z + 2 * z^2 = 60) ∧ 
    (y^2 + 8 * z^2 = 90) 
    → False :=
by 
  intro y z
  simp
  sorry

end no_integer_solutions_for_system_l154_154182


namespace total_volume_relation_l154_154522

variable (r : ℝ) -- Given the radius 'r' which is a real number

-- Definitions for the volumes of cone, cylinder, and sphere
def volume_cone (r : ℝ) := (1 / 3) * Math.pi * r^3
def volume_cylinder (r : ℝ) := Math.pi * r^3
def volume_sphere (r : ℝ) := (4 / 3) * Math.pi * r^3

-- The statement to prove the relationship among their volumes
theorem total_volume_relation :
  volume_cone r + volume_cylinder r + volume_sphere r = (8 / 3) * Math.pi * r^3 :=
by {
  sorry -- Skip proof
}

end total_volume_relation_l154_154522


namespace theater_queue_arrangement_l154_154716

theorem theater_queue_arrangement : 
  let n := 7 -- total number of people
  let pair := 2 -- Alice and Bob considered as one unit
  let k := n - pair + 1 -- reducing to 6 units
  (Nat.factorial k) * (Nat.factorial pair) = 1440 :=
by
  let n := 7
  let pair := 2
  let k := n - pair + 1
  have h_k : k = 6 := rfl
  have h1 : Nat.factorial k = Nat.factorial 6 := by rw [h_k]
  have h_fac6 : Nat.factorial 6 = 720 := by norm_num
  have h_fac2 : Nat.factorial 2 = 2 := by norm_num
  rw [h1, h_fac6, h_fac2]
  norm_num
  exact rfl

end theater_queue_arrangement_l154_154716


namespace probability_heads_in_12_flips_l154_154001

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l154_154001


namespace tan_double_angle_difference_l154_154643

variable {α β : Real}

theorem tan_double_angle_difference (h1 : Real.tan α = 1 / 2) (h2 : Real.tan (α - β) = 1 / 5) :
  Real.tan (2 * α - β) = 7 / 9 := 
sorry

end tan_double_angle_difference_l154_154643


namespace problem_statement_l154_154774

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l154_154774


namespace angela_height_l154_154161

def height_of_Amy : ℕ := 150
def height_of_Helen : ℕ := height_of_Amy + 3
def height_of_Angela : ℕ := height_of_Helen + 4

theorem angela_height : height_of_Angela = 157 := by
  sorry

end angela_height_l154_154161


namespace opposite_of_neg_five_halves_l154_154824

theorem opposite_of_neg_five_halves : -(- (5 / 2: ℝ)) = 5 / 2 :=
by
    sorry

end opposite_of_neg_five_halves_l154_154824


namespace union_complement_eq_l154_154784

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l154_154784


namespace ordered_triples_count_l154_154688

theorem ordered_triples_count :
  let count_valid_triples : ℕ :=
    (finset.range 2011).sum (λ a, finset.range (2011 - a)).sum
  count_valid_triples = 9045 := 
begin
  sorry
end

end ordered_triples_count_l154_154688


namespace celeste_can_empty_table_l154_154590

def can_empty_table (n : ℕ) : Prop :=
  ∀ m > 0, ∀ (initial_candies : list ℕ), 
  (∀ candy, candy ∈ initial_candies → 1 ≤ candy ∧ candy ≤ n) →
  ∃ final_candies : list ℕ, final_candies = [] ∧ true -- assuming operations lead to empty

theorem celeste_can_empty_table (n : ℕ) : can_empty_table n ↔ n % 3 ≠ 0 :=
begin
  sorry
end

end celeste_can_empty_table_l154_154590


namespace smallest_integer_with_16_divisors_l154_154068

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (∀ m : ℕ, (has_16_divisors m → m ≥ n)) ∧ n = 384 :=
by
  def has_16_divisors (n : ℕ) : Prop :=
    ∃ a₁ a₂ a₃ k₁ k₂ k₃, n = k₁ ^ a₁ * k₂ ^ a₂ * k₃ ^ a₃ ∧ (a₁ + 1) * (a₂ + 1) * (a₃ + 1) = 16
  -- Proof will go here
  sorry

end smallest_integer_with_16_divisors_l154_154068


namespace sally_total_fries_l154_154404

theorem sally_total_fries :
  let initial_fries := 14
  let mark_fries := (1 / 3) * 36
  let jessica_fries := 0.5 * 24
  initial_fries + mark_fries + jessica_fries = 38 :=
by
  let initial_fries := 14
  let mark_fries := (1 / 3) * 36
  let jessica_fries := 0.5 * 24
  calc initial_fries + mark_fries + jessica_fries = 14 + 12 + 12 : by simp [initial_fries, mark_fries, jessica_fries]
                                             ... = 38           : by norm_num

end sally_total_fries_l154_154404


namespace problem_l154_154695

theorem problem (x y : ℝ) (h : (3 * x - y + 5)^2 + |2 * x - y + 3| = 0) : x + y = -3 := 
by
  sorry

end problem_l154_154695


namespace isabel_initial_candy_l154_154338

theorem isabel_initial_candy (total_candy : ℕ) (candy_given : ℕ) (initial_candy : ℕ) :
  candy_given = 25 → total_candy = 93 → total_candy = initial_candy + candy_given → initial_candy = 68 :=
by
  intros h_candy_given h_total_candy h_eq
  rw [h_candy_given, h_total_candy] at h_eq
  sorry

end isabel_initial_candy_l154_154338


namespace inequality_solution_l154_154038

theorem inequality_solution (p : ℝ) :
  (∀ q > 0, (3 * (p * q ^ 2 + p ^ 2 * q + 3 * q ^ 2 + 3 * p * q) / (p + q)) > 2 * p ^ 2 * q) →
  p ∈ set.Ico 0 3 :=
  by
  intro h
  sorry

end inequality_solution_l154_154038


namespace count_polynomials_in_H_l154_154345

noncomputable def polynomial_count (N : ℕ) : Prop :=
  ∃ (H : set (polynomial ℂ)),
    (∀ P ∈ H, ∃ (n : ℕ) (c : fin n → ℤ), 
      P = polynomial.monomial n 1 + ∑ i in finset.range n, polynomial.monomial i (c i)) ∧ 
    (∀ P ∈ H, ∃ (a b : ℤ) (n : ℕ), 
      P.has_root (a + b * complex.I) ∧ P.has_root (a - b * complex.I)) ∧
    (∀ P ∈ H, b ≠ 0) ∧
    ((H.card : ℕ) = N)

theorem count_polynomials_in_H (N : ℕ) : polynomial_count N := sorry

end count_polynomials_in_H_l154_154345


namespace basketball_team_starting_lineup_l154_154391

theorem basketball_team_starting_lineup (n k : ℕ) (hn : n = 12) (hk : k = 5) :
  (n * nat.choose (n - 1) (k - 1)) = 3960 :=
by
  sorry

end basketball_team_starting_lineup_l154_154391


namespace minimum_value_of_expression_l154_154613

noncomputable def f (x : ℝ) : ℝ := 3 * real.sqrt x + 1 / x

theorem minimum_value_of_expression : ∃ x > 0, ∀ y > 0, f x ≤ f y ∧ f x = 4 := by
  sorry

end minimum_value_of_expression_l154_154613


namespace angle_Z_of_pentagon_l154_154546

theorem angle_Z_of_pentagon (V W X Y Z : Type) 
  (side_length : V → W → X → Y → Z → ℝ)
  (h1 : ∀ A B, side_length A B = side_length V W)
  (h2 : angle V W V = 120)
  (h3 : angle W X W = 120) : 
  angle W Z Y = 120 := 
sorry

end angle_Z_of_pentagon_l154_154546


namespace number_of_such_lines_l154_154516

noncomputable def hyperbola : Set (ℝ × ℝ) :=
  { p | ∃ (x y : ℝ), 2 * x^2 - y^2 - 8 * x + 6 = 0 }

def passes_through_foci (l : ℝ → ℝ) : Prop :=
  ∀ (F : ℝ × ℝ), F ∈ foci → F ∈ set.line_span { (l 0, l 1) }

def |AB|_eq_4 (A B : ℝ × ℝ) : Prop :=
  abs (dist A B) = 4

theorem number_of_such_lines :
  ∃ (l : ℝ → ℝ), (∀ (A B : ℝ × ℝ), A ∈ hyperbola ∧ B ∈ hyperbola ∧ passes_through_foci l ∧ |AB|_eq_4 A B)
    → l = 3 := 
sorry

end number_of_such_lines_l154_154516


namespace regression_analysis_correct_statements_l154_154095

theorem regression_analysis_correct_statements :
  (∀ (r : ℝ), abs r ≤ 1 → abs r = 1 → ∀ (x y : ℤ), 
    has_stronger_linear_correlation x y) ∧
  (∀ (x y : ℤ) (b a : ℝ), linear_regression_line (λ x, b * x + a) (average x) (average y)) ∧
  (∀ (model : ℤ → ℝ), (sum_of_squared_residuals model < ε → 
    model_is_better_fit model)) ∧
  ¬ (∀ (X Y : Type) (obs_val_k : ℝ), categorical_variables_relationship X Y obs_val_k → 
    obs_val_k_indicates_stronger_relationship obs_val_k) :=
by sorry

end regression_analysis_correct_statements_l154_154095


namespace maximum_guards_for_concave_hexagon_l154_154815

def isConcaveHexagon (hex : Type) : Prop := 
  -- Definition of concave hexagon goes here
  sorry

def guardVisibility (hex : Type) (guards : Finset hex) : Prop :=
  ∀ point ∈ walls hex, ∃ guard ∈ guards, canSee point guard

theorem maximum_guards_for_concave_hexagon (hex : Type) 
  (h : isConcaveHexagon hex) :
  ∃ (guards : Finset hex), 
    guardVisibility hex guards ∧ guards.card = 2 :=
sorry

end maximum_guards_for_concave_hexagon_l154_154815


namespace hyperbola_eccentricity_l154_154631

open Real

def hyperbola : Prop :=
  ∃ (a b c : ℝ), 
    a > 0 ∧ 
    b > 0 ∧ 
    (∀ x y: ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ x y: ℝ, (x - c)^2 + y^2 = 4*a^2) ∧
    (∃ l: ℝ, l = 2*b) ∧ 
    (c^2 = 3*a^2 ∧ b^2 = 2*a^2)

theorem hyperbola_eccentricity : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (∀ x y: ℝ, (x - c)^2 + y^2 = 4*a^2) ∧ 
  (∃ l : ℝ, l = 2*b) ∧ c^2 = 3*a^2 ∧ b^2 = 2*a^2 → 
  a > 0 ∧ b > 0 ∧ c = a * (√3) →
  √(a^2 + b^2) = √(a^2 + a^2 * 2) := 
sorry

end hyperbola_eccentricity_l154_154631


namespace correct_operation_l154_154480

theorem correct_operation :
  (3 * a^3 - 2 * a^3 = a^3) ∧ ¬(m - 4 * m = -3) ∧ ¬(a^2 * b - a * b^2 = 0) ∧ ¬(2 * x + 3 * x = 5 * x^2) :=
by
  sorry

end correct_operation_l154_154480


namespace smallest_integer_with_16_divisors_l154_154082

theorem smallest_integer_with_16_divisors : 
  ∃ (n : ℕ), (∃ (p_1 p_2 p_3 : ℕ) (a_1 a_2 a_3 : ℕ), 
  (p_1 = 2 ∧ p_2 = 3 ∧ a_1 = 3 ∧ a_2 = 3 ∧ n = p_1 ^ a_1 * p_2 ^ a_2) ∧
  (∀ m, m > 0 → (∃ b1 b2 ..., m has exactly 16 positive divisors) → 216 ≤ m)) := 
sorry

end smallest_integer_with_16_divisors_l154_154082


namespace coin_flip_probability_l154_154017

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154017


namespace smallest_integer_with_16_divisors_l154_154070

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (∀ m : ℕ, (has_16_divisors m → m ≥ n)) ∧ n = 384 :=
by
  def has_16_divisors (n : ℕ) : Prop :=
    ∃ a₁ a₂ a₃ k₁ k₂ k₃, n = k₁ ^ a₁ * k₂ ^ a₂ * k₃ ^ a₃ ∧ (a₁ + 1) * (a₂ + 1) * (a₃ + 1) = 16
  -- Proof will go here
  sorry

end smallest_integer_with_16_divisors_l154_154070


namespace band_arrangement_l154_154505

theorem band_arrangement : 
  {x : ℕ | 4 ≤ x ∧ x ≤ 25 ∧ 100 % x = 0}.finite ∧ 
  {x : ℕ | 4 ≤ x ∧ x ≤ 25 ∧ 100 % x = 0}.card = 5 := 
by 
  sorry

end band_arrangement_l154_154505


namespace Reeta_pencils_l154_154163

-- Let R be the number of pencils Reeta has
variable (R : ℕ)

-- Condition 1: Anika has 4 more than twice the number of pencils as Reeta
def Anika_pencils := 2 * R + 4

-- Condition 2: Together, Anika and Reeta have 64 pencils
def combined_pencils := R + Anika_pencils R

theorem Reeta_pencils (h : combined_pencils R = 64) : R = 20 :=
by
  sorry

end Reeta_pencils_l154_154163


namespace sequence_contains_infinitely_many_palindromes_l154_154393

/-- 
Palindrome sequence:
Prove that the sequence defined by \( x_n = 2013 + 317n \) contains infinitely many numbers whose decimal expansions are palindromes.
-/
theorem sequence_contains_infinitely_many_palindromes :
  ∃ᶠ n in Filter.atTop, (let x := 2013 + 317 * n in palindrome (nat.digits 10 x)) :=
sorry

-- Additional definition necessary for the proof setup:
-- Define a function to check if a number is a palindrome
def palindrome (l : list ℕ) : Prop :=
  l = l.reverse

-- Aggregate the theorem back with the function definition
#align sequence_contains_infinitely_many_palindromes

end sequence_contains_infinitely_many_palindromes_l154_154393


namespace proof_problem_l154_154231

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem proof_problem (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : f x * f (-x) = 1 := 
by 
  sorry

end proof_problem_l154_154231


namespace g_value_2023_l154_154373

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_def_pos : ∀ x > 0, g x > 0
axiom g_eq_sqrt : ∀ x y : ℝ, x > y → y > 0 → g (x - y) = real.sqrt (g (x * y) + 3 * x - y)

theorem g_value_2023 : g 2023 = 1 :=
by
  sorry

end g_value_2023_l154_154373


namespace can_combine_with_sqrt2_l154_154088

theorem can_combine_with_sqrt2 :
  (∃ (x : ℝ), x = 2 * Real.sqrt 6 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 * Real.sqrt 3 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧ ∃ (y : ℝ), y = Real.sqrt 2) :=
sorry

end can_combine_with_sqrt2_l154_154088


namespace Shara_savings_l154_154531

theorem Shara_savings (P : ℝ) (d : ℝ) (paid : ℝ):
  d = 0.08 → paid = 184 → P = 200 → (P * (1 - d) = paid) → (P - paid = 16) :=
by
  intros hd hpaid hP heq
  -- It follows from the conditions given
  sorry

end Shara_savings_l154_154531


namespace tangent_PQ_UT_l154_154845

variables {A B C P D U Q T R S : Point}
variables {O : Circle}

-- Conditions as definitions
def triangle_inscribed (ABC : Triangle) (O : Circle) : Prop :=
  O.circumscribes ABC

def tangent_PD_from_A (A P D : Point) (BC : Line) : Prop :=
  tangent_from A P ∧ D ∈ BC ∧ P ∈ ray DA

def line_PU_intersects_circle_O 
  (PU AB AC : Line) (O : Circle) (Q T R S : Point) : Prop :=
  PU ∩ O = {Q, T} ∧ 
  PU ∩ AB = R ∧ 
  PU ∩ AC = S

def given_equality (Q R S T : Point) : Prop :=
  dist Q R = dist S T

-- Problem statement
theorem tangent_PQ_UT
  (triangle_ABC : Triangle)
  (O : Circle)
  (HPQ : triangle_inscribed triangle_ABC O)
  (H1 : tangent_PD_from_A A P D BC)
  (H2 : line_PU_intersects_circle_O PU AB AC O Q T R S)
  (H3 : given_equality Q R S T) :
  dist P Q = dist U T := 
sorry

end tangent_PQ_UT_l154_154845


namespace cube_root_fraction_equivalence_l154_154954

theorem cube_root_fraction_equivalence (h : 12.75 = 51 / 4) :
  ∛(8 / 12.75) = ∛(32 / 51) :=
  sorry

end cube_root_fraction_equivalence_l154_154954


namespace sum_of_solutions_eq_l154_154084

theorem sum_of_solutions_eq (x : ℝ) (H : x = abs (3 * x - abs (80 - 3 * x))) :
    ∑ x in { y : ℝ | y = abs (3 * y - abs (80 - 3 * y)) }, y = 752 / 7 :=
sorry

end sum_of_solutions_eq_l154_154084


namespace range_of_a_l154_154273

-- Definitions of the functions
noncomputable def f (x a : ℝ) := x^2 - a * x - a * Real.log x
noncomputable def g (x : ℝ) := -x^3 + 5/2 * x^2 + 2 * x - 6

-- The maximum value of g(x) on [1, 4] is denoted as b
noncomputable def b := Real.max (g 1) (g 4)

-- Theorem statement
theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, x ≥ 1 → f x a ≥ b) : a ≤ 1 :=
sorry

end range_of_a_l154_154273


namespace ratio_area_triangle_PXZ_area_square_PQRS_l154_154326

theorem ratio_area_triangle_PXZ_area_square_PQRS :
  ∀ (s : ℝ) (P Q R S X Y Z : ℝ × ℝ),
  (Q = (s, 0)) →
  (R = (s, s)) →
  (X = (s / 2, 0)) →
  (Y = (s, s / 2)) →
  (Z = ((s / 2) / 2, (s / 2) / 2)) →
  let area_square := s * s in
  let area_triangle_PXZ := 0.5 * (s/2) * (s/4) in
  (area_triangle_PXZ / area_square) = 1 / 16 := 
by
  sorry

end ratio_area_triangle_PXZ_area_square_PQRS_l154_154326


namespace probability_all_heads_or_tails_l154_154209

def coin_outcomes := {heads, tails}

def total_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes : ℕ := 2

def probability_five_heads_or_tails (n : ℕ) (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_all_heads_or_tails :
  probability_five_heads_or_tails 5 (total_outcomes 5) favorable_outcomes = 1 / 16 :=
by
  sorry

end probability_all_heads_or_tails_l154_154209


namespace smallest_integer_with_16_divisors_l154_154059

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ :=
  n.factors.to_finset.prod fun p => (n.factorization p + 1)

-- Define the positive integer n which we need to prove has 16 divisors
def smallest_positive_integer_with_16_divisors (n : ℕ) : Prop :=
  num_divisors n = 16

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, smallest_positive_integer_with_16_divisors n ∧ ∀ m : ℕ, m < n → ¬smallest_positive_integer_with_16_divisors m :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154059


namespace sum_of_digits_of_possible_steps_l154_154597

theorem sum_of_digits_of_possible_steps :
  let steps := {n : Nat | ∃ (a b : ℕ), a ∈ {0, 1} ∧ b ∈ {0, 1, 2, 3} ∧ n = 60 - 2 * a + b};
  let s := steps.foldr (·+·) 0;
  let digit_sum := s.toDigits.foldr (·+·) 0;
  digit_sum = 12 :=
by
  sorry

end sum_of_digits_of_possible_steps_l154_154597


namespace solution_set_of_inequality_l154_154446

theorem solution_set_of_inequality :
  { x : ℝ | - (1 : ℝ) / 2 < x ∧ x <= 1 } =
  { x : ℝ | (x - 1) / (2 * x + 1) <= 0 ∧ x ≠ - (1 : ℝ) / 2 } :=
by
  sorry

end solution_set_of_inequality_l154_154446


namespace garden_length_increase_l154_154890

variable (L W : ℝ)  -- Original length and width
variable (X : ℝ)    -- Percentage increase in length

theorem garden_length_increase :
  (1 + X / 100) * 0.8 = 1.1199999999999999 → X = 40 :=
by
  sorry

end garden_length_increase_l154_154890


namespace copper_needed_for_mixture_l154_154101

variable (T : ℝ) (M : ℝ)

-- Conditions
def has_lead_percent (p : ℝ) := p = 0.25
def has_copper_percent (p : ℝ) := p = 0.60
def lead_mass (m : ℝ) := m = 5
def total_mass_given (T : ℝ) := T = 20
def copper_required (M : ℝ) := M = 0.60 * T

-- Proof statement
theorem copper_needed_for_mixture :
  ∀ (cobalt_percent lead_percent copper_percent : ℝ)
  (used_lead_mass : ℝ),
  has_lead_percent lead_percent → has_copper_percent copper_percent →
  lead_mass used_lead_mass →
  total_mass_given T →
  copper_required M →
  M = 12 :=
by
  intros
  rw [has_lead_percent, has_copper_percent, lead_mass, total_mass_given, copper_required] at *
  sorry

end copper_needed_for_mixture_l154_154101


namespace log_equation_solution_l154_154625

theorem log_equation_solution (a x : ℝ) (h1 : 4^a = 2) (h2 : real.log10 x = a) : x = real.sqrt 10 := 
by 
  sorry -- Proof placeholder

end log_equation_solution_l154_154625


namespace smallest_integer_with_16_divisors_l154_154066

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (∀ m : ℕ, (has_16_divisors m → m ≥ n)) ∧ n = 384 :=
by
  def has_16_divisors (n : ℕ) : Prop :=
    ∃ a₁ a₂ a₃ k₁ k₂ k₃, n = k₁ ^ a₁ * k₂ ^ a₂ * k₃ ^ a₃ ∧ (a₁ + 1) * (a₂ + 1) * (a₃ + 1) = 16
  -- Proof will go here
  sorry

end smallest_integer_with_16_divisors_l154_154066


namespace polyhedron_intersections_lead_to_new_polyhedron_with_450_edges_l154_154510

def convex_polyhedron (Q : Type) [Polyhedron Q] : Prop :=
  ∃ (V : Set V) (n : ℕ), ∃ (edges_150 : cardinal.mk (Set {e : Edge Q // meets_at_vertex e V}) = 150)

def planes_intersect_edges (Q : Type) [Polyhedron Q] (P : Set (Plane Q)) : Prop :=
  ∀ (k : ℕ) (V_k : Vertex Q), k < n → intersects_edges_through_vertex (P k) (V_k)

def no_planes_overlap (Q : Type) [Polyhedron Q] (P : Set (Plane Q)) : Prop :=
  ∀ (i j : ℕ) (i ≠ j), ¬overlap_within_volume_or_surface (P i) (P j)

def each_cut_divides_edges (Q : Type) [Polyhedron Q] : Prop :=
  ∀ (e : Edge Q) (P : Plane Q), divides_edge (e, P) → segments_count e (P) = 3

noncomputable def new_polyhedron_has_450_edges (Q : Type) [Polyhedron Q] (P : Set (Plane Q)) : Prop :=
  ∃ (R : Polyhedron Q), edge_count R = 450

theorem polyhedron_intersections_lead_to_new_polyhedron_with_450_edges 
  (Q : Type) [Polyhedron Q] (P : Set (Plane Q)) 
  (h_polyhedron : convex_polyhedron Q)
  (h_planes_intersect : planes_intersect_edges Q P)
  (h_no_overlap : no_planes_overlap Q P)
  (h_cut_divides : each_cut_divides_edges Q) :
  new_polyhedron_has_450_edges Q P :=
sorry

end polyhedron_intersections_lead_to_new_polyhedron_with_450_edges_l154_154510


namespace zero_in_interval_l154_154696

theorem zero_in_interval (a : ℝ) (h : 3 < a) : 
  ∃! x ∈ Ioo 0 2, (x^2 - a * x + 1 = 0) :=
sorry

end zero_in_interval_l154_154696


namespace compute_series_sum_l154_154930

noncomputable def term (n : ℕ) : ℝ := (5 * n - 2) / (3 ^ n)

theorem compute_series_sum : 
  ∑' n, term n = 11 / 4 := 
sorry

end compute_series_sum_l154_154930


namespace probability_heads_9_or_more_12_flips_l154_154022

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154022


namespace total_blue_paint_cans_l154_154392

-- Define the conditions
def ratio_blue_to_green : ℕ × ℕ := (4, 3)
def total_cans : ℕ := 35
def special_blue_cans : ℕ := 2

-- Define the proof problem
theorem total_blue_paint_cans : 
  (let total_parts := ratio_blue_to_green.1 + ratio_blue_to_green.2 in
  let blue_fraction := (ratio_blue_to_green.1 : ℚ) / total_parts in
  let blue_cans := blue_fraction * total_cans in
  blue_cans.to_nat + special_blue_cans = 22) :=
sorry

end total_blue_paint_cans_l154_154392


namespace problem_statement_l154_154770

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l154_154770


namespace centroid_positions_l154_154807

/-- Suppose we have a rectangle of length 15 units and width 10 units, with 60 points equally spaced
    around its perimeter, including the four vertices. The remaining points divide each side into 
    parts with 14 points on the longer sides and 8 points on the shorter sides.

Given three non-collinear points (P, Q, and R) among these 60 points, the possible positions for the 
centroid of triangle PQR is 1276. -/
theorem centroid_positions {P Q R : Point} 
  (hP : P ∈ perimeter_points (15, 10) 60) 
  (hQ : Q ∈ perimeter_points (15, 10) 60)
  (hR : R ∈ perimeter_points (15, 10) 60)
  (h_non_collinear : ¬ collinear P Q R) :
  number_of_centroid_positions (rectangle_points (15, 10) 60) = 1276 := 
sorry

/-- Assume a Point structure to represent the coordinates. -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Function to generate 60 equally spaced points around the perimeter of given rectangle.
    Takes length and width of the rectangle, along with total number of points as arguments. -/
def perimeter_points (length width : ℝ) (num_points : ℕ) : set Point := 
  sorry  -- Implementation of the function that generates the points

/-- Function to calculate the number of possible centroid positions given a set of points around
    the perimeter of the rectangle. The conditions of spacing ensures certain limits on the 
    possible coordinate values of the centroid. -/
def number_of_centroid_positions (points : set Point) : ℕ := 
  sorry  -- Implementation of the function

/-- Function to check if three points are collinear. -/
def collinear (P Q R : Point) : Prop :=
  (Q.y - P.y) * (R.x - Q.x) = (R.y - Q.y) * (Q.x - P.x)

end centroid_positions_l154_154807


namespace number_of_perfect_square_multiples_le_2000_l154_154969

theorem number_of_perfect_square_multiples_le_2000 :
  {n : ℕ | n ≤ 2000 ∧ ∃ k : ℕ, 10 * n = k^2}.finite.card = 14 := by
sorry

end number_of_perfect_square_multiples_le_2000_l154_154969


namespace sum_of_first_n_terms_l154_154644

variable (a : ℕ → ℤ) (b : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a_n_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom a_3 : a 3 = -6
axiom a_6 : a 6 = 0
axiom b_1 : b 1 = -8
axiom b_2 : b 2 = a 1 + a 2 + a 3

-- Correct answer to prove
theorem sum_of_first_n_terms : S n = 4 * (1 - 3^n) := sorry

end sum_of_first_n_terms_l154_154644


namespace road_length_l154_154915

theorem road_length (L : ℝ) (h1 : 300 = 200 + 100)
  (h2 : 50 * 100 = 2.5 / (L / 300))
  (h3 : 75 + 50 = 125)
  (h4 : (125 / 50) * (2.5 / 100) * 200 = L - 2.5) : L = 15 := 
by
  sorry

end road_length_l154_154915


namespace compound_interest_paid_back_l154_154387

theorem compound_interest_paid_back :
  ∀ (P r : ℝ) (t : ℕ), P = 150 ∧ r = 0.06 ∧ t = 2 → (P * (1 + r)^t = 168.54) :=
by
  intro P r t h
  cases h with hP hr
  cases hr with hr ht
  rw [hP, hr, ht]
  sorry

end compound_interest_paid_back_l154_154387


namespace rhombus_area_l154_154812

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 18) (h2 : d2 = 14) : (d1 * d2) / 2 = 126 :=
by
  rw [h1, h2]
  norm_num
  -- Here "norm_num" will simplify the numeric expression (18 * 14) / 2 to 126.
  sorry

end rhombus_area_l154_154812


namespace sum_geom_S40_l154_154832

variable {a : ℝ} {q : ℝ}

-- Definition of the sum of the first n terms of a geometric sequence
def S (n : ℕ) : ℝ := a * (q^n - 1) / (q - 1)

-- Given conditions
variable (h1 : S 10 = 10)
variable (h2 : S 30 = 70)

-- Statement to prove
theorem sum_geom_S40 : S 40 = 150 :=
by
  sorry

end sum_geom_S40_l154_154832


namespace p_sufficient_for_q_l154_154245

def p (x : ℝ) : Prop := x^2 < 2 * x + 3
def q (x : ℝ) : Prop := |x - 1| ≤ 2

theorem p_sufficient_for_q : (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  split
  · intro x hp
    unfold p at hp
    unfold q
    calc
      x^2 < 2 * x + 3 := hp
      _   ≤ 4*(x-1)^2 := sorry
  · sorry

end p_sufficient_for_q_l154_154245


namespace animal_count_l154_154899

theorem animal_count (dogs : ℕ) (cats : ℕ) (birds : ℕ) (fish : ℕ)
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) : 
  dogs + cats + birds + fish = 39 :=
by
  sorry

end animal_count_l154_154899


namespace flip_coin_probability_l154_154035

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154035


namespace evaluate_expression_l154_154187

theorem evaluate_expression : 
  1 + 2 / (3 + 4 / (5 + 6 / 7)) = 233 / 151 := 
by 
  sorry

end evaluate_expression_l154_154187


namespace maximize_angle_exists_l154_154990

structure Plane (P : Type*) :=
(point : P → Prop)

variables {P : Type*} [MetricSpace P]

-- Given a plane α and points A, B not on α
variables (α : Plane P) (A B : P)
variables (hA : ¬α.point A) (hB : ¬α.point B)

-- Define the angle function
noncomputable def angle (p q r : P) : ℝ := sorry

-- Define the maximization condition
def maximizes_angle (P : P) : Prop :=
∀ P' : P, (α.point P → α.point P') → angle A P B ≥ angle A P' B

-- The statement to prove
theorem maximize_angle_exists :
  ∃ P : P, α.point P ∧ maximizes_angle α A B P :=
sorry

end maximize_angle_exists_l154_154990


namespace ellipse_midpoint_line_eq_l154_154253

noncomputable def equation_of_line_AB
  (A B : ℝ × ℝ)
  (P : ℝ × ℝ := (-2, 1))
  (hA : (A.1 ^ 2) / 16 + (A.2 ^ 2) / 4 = 1)
  (hB : (B.1 ^ 2) / 16 + (B.2 ^ 2) / 4 = 1)
  (hP : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (-2, 1))
  : Prop :=
  ∀ (x y : ℝ), x - 2 * y + 4 = 0

theorem ellipse_midpoint_line_eq 
  (A B : ℝ × ℝ)
  (hA : (A.1 ^ 2) / 16 + (A.2 ^ 2) / 4 = 1)
  (hB : (B.1 ^ 2) / 16 + (B.2 ^ 2) / 4 = 1)
  (hP : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (-2, 1))
  : equation_of_line_AB A B :=
  sorry

end ellipse_midpoint_line_eq_l154_154253


namespace molecular_weight_of_one_mole_l154_154046

theorem molecular_weight_of_one_mole (molecular_weight_3_moles : ℕ) (h : molecular_weight_3_moles = 222) : (molecular_weight_3_moles / 3) = 74 := 
by
  sorry

end molecular_weight_of_one_mole_l154_154046


namespace square_in_circular_segment_square_with_relaxed_constraints_l154_154176

noncomputable def central_angle (chord_length radius : ℝ) : ℝ := 
  2 * real.arcsin (chord_length / (2 * radius))

theorem square_in_circular_segment (radius chord_length : ℝ) (h : 0 < chord_length ∧ chord_length < 2 * radius) :
  ∃ s : ℝ, ∃ square : set (ℝ × ℝ),
    (∀ p ∈ square, dist (0,0) p ≤ radius) ∧ -- square vertices are within the circle
    (∀ p ∈ square, p.2 ≥ 0) ∧ -- square vertices are above x-axis
    (central_angle chord_length radius ≤ 270) := 
sorry

theorem square_with_relaxed_constraints (radius chord_length : ℝ) (h : 0 < chord_length ∧ chord_length < 2 * radius) :
  ∃ s : ℝ, ∃ square : set (ℝ × ℝ),
    (∃ p1 p2 ∈ square, dist (0,0) p1 = radius ∧ dist (0,0) p2 = radius) ∧ -- two vertices on the circle
    (∀ p ∈ square, p.2 ≥ 0) -- square vertices are above x-axis
 :=
sorry

end square_in_circular_segment_square_with_relaxed_constraints_l154_154176


namespace minimize_segment_sum_l154_154461

theorem minimize_segment_sum (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  ∃ x y : ℝ, x = Real.sqrt (a * b) ∧ y = Real.sqrt (a * b) ∧ x * y = a * b ∧ x + y = 2 * Real.sqrt (a * b) := 
by
  sorry

end minimize_segment_sum_l154_154461


namespace smallest_integer_with_16_divisors_l154_154055

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ :=
  n.factors.to_finset.prod fun p => (n.factorization p + 1)

-- Define the positive integer n which we need to prove has 16 divisors
def smallest_positive_integer_with_16_divisors (n : ℕ) : Prop :=
  num_divisors n = 16

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, smallest_positive_integer_with_16_divisors n ∧ ∀ m : ℕ, m < n → ¬smallest_positive_integer_with_16_divisors m :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154055


namespace surrounding_circle_radius_l154_154881

theorem surrounding_circle_radius (r : ℝ) :
  (∃ r, let diagonal := 2 + 2 * r in
         (2 * r) ^ 2 + (2 * r) ^ 2 = diagonal ^ 2 ∧
         r > 0) → r = 1 + Real.sqrt 2 :=
begin
  sorry
end

end surrounding_circle_radius_l154_154881


namespace directrix_equation_l154_154966

def parabola_directrix (x : ℝ) : ℝ :=
  (x^2 - 8*x + 12) / 16

theorem directrix_equation :
  ∀ x, parabola_directrix x = y → y = -5/4 :=
sorry

end directrix_equation_l154_154966


namespace line_intersects_y_axis_l154_154892

-- Define the points
def P1 : ℝ × ℝ := (3, 18)
def P2 : ℝ × ℝ := (-9, -6)

-- State that the line passing through P1 and P2 intersects the y-axis at (0, 12)
theorem line_intersects_y_axis :
  ∃ y : ℝ, (∃ m b : ℝ, ∀ x : ℝ, y = m * x + b ∧ (m = (P2.2 - P1.2) / (P2.1 - P1.1)) ∧ (P1.2 = m * P1.1 + b) ∧ (x = 0) ∧ y = 12) :=
sorry

end line_intersects_y_axis_l154_154892


namespace gerald_apples_l154_154394

theorem gerald_apples (G : ℕ) (pam_bags : ℕ) (total_apples : ℕ) 
  (h1 : pam_bags = 10) 
  (h2 : total_apples = 1200) 
  (h3 : ∀ k, pam_bags * (3 * k) = total_apples) : 
  G = 40 :=
by
  have h4 : 10 * (3 * G) = 1200 := h3 G
  have h5 : 30 * G = 1200 := by rw [mul_assoc] at h4; exact h4
  have h6 : G = 1200 / 30 := by rw [mul_div_cancel_left _ (by norm_num : 30 ≠ 0)] at h5; exact h5
  norm_num at h6
  exact h6

end gerald_apples_l154_154394


namespace pythagoras_voted_both_issues_l154_154564

open Finset

noncomputable def pythagoras_referendum : Prop :=
  let students := 250
  let favor_first := 171
  let favor_second := 141
  let against_both := 39
  let favor_any := students - against_both
  favor_any = 211 ∧ (favor_first + favor_second - favor_any) = 101

theorem pythagoras_voted_both_issues (students favor_first favor_second against_both favor_any : ℕ)
  (h1 : students = 250)
  (h2 : favor_first = 171)
  (h3 : favor_second = 141)
  (h4 : against_both = 39)
  (h5 : favor_any = students - against_both) :
  (favor_any = 211 ∧ (favor_first + favor_second - favor_any) = 101) :=
by
  rw [h1, h2, h3, h4] at h5
  split
  { exact h5 }
  { rw h5
    norm_num }

#print axioms pythagoras_voted_both_issues

end pythagoras_voted_both_issues_l154_154564


namespace m_range_l154_154251

theorem m_range (A B : Set ℝ) (m : ℝ) :
  (A = {x | x^2 - 2 * x - 3 <= 0}) →
  (B = {x | abs (x - m) > 3}) →
  (A ⊆ B) →
  m ∈ (-∞, -4) ∪ (6, ∞) :=
by
  intros hA hB h_subset
  sorry

end m_range_l154_154251


namespace distinct_prime_sum_product_l154_154452

open Nat

-- Definitions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- The problem statement
theorem distinct_prime_sum_product (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) 
    (h3 : is_prime c) (h4 : a ≠ 1) (h5 : b ≠ 1) (h6 : c ≠ 1) 
    (h7 : a ≠ b) (h8 : b ≠ c) (h9 : a ≠ c) : 

    1994 + a + b + c = a * b * c :=
sorry

end distinct_prime_sum_product_l154_154452


namespace highest_rank_athlete_l154_154103

noncomputable def highest_average_rank_top_athlete (n m : ℕ) (ranks : ℕ → ℕ → ℕ) : ℚ :=
  ((finset.range n).sum (λ i, (finset.range m).sum λ j, ranks i j)) / m

axiom athlete_rank_constraints (n m : ℕ) (ranks : ℕ → ℕ → ℕ) : 
  (∀ i j k, i < n ∧ j < m ∧ k < m → abs (ranks i j - ranks i k) ≤ 3)

theorem highest_rank_athlete (n m : ℕ) (ranks : ℕ → ℕ → ℕ) :
  n = 20 → m = 9 → athlete_rank_constraints n m ranks → 
  highest_average_rank_top_athlete n m ranks ≤ (8 / 3) :=
by
  rintros _ _ _ hnm
  sorry  -- Write the required proof here

end highest_rank_athlete_l154_154103


namespace solve_for_x_requires_numerical_methods_l154_154717

-- Define trapezoid proportions and side lengths
variables {A B C D : Type*}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables {AC BD BC DA : forall (AB CD h x : ℚ), ℚ}
variables {h d x : ℚ}

-- Define the equality relationship
theorem solve_for_x_requires_numerical_methods 
  (h d x : ℚ) 
  (trapezoid : ∀ (A B C D : Type*), metric_space A → metric_space B → metric_space C → metric_space D → Prop)
  (AC BD BC DA : ℕ) 
  (H1 : h > 0) 
  (H2 : d > 0)
  (H3 : AC + BD = BC + DA)
  (H4 : DA = x) 
  (H5 : BC = h + 2 * x) 
  : 
  "Numerical methods required" := by
  sorry

end solve_for_x_requires_numerical_methods_l154_154717


namespace angle_man_fixed_l154_154233

theorem angle_man_fixed {P Q l : Set Point} (H1 : ∀ y : Real, P = point.mk 0 2)
  (H2 : ∀ x : Real, Q = point.mk x 0) (H3 : circle.mk P 1 ∈ Circ)
  (H4 : ∀ (Q : Point) (r : Real), circle.mk Q r ∈ Circ ∧ circle.mk Q r ∈ Tangent (circle.mk P 1))
  (H5 : ∀ (M N : Point), diameter M N Q) :
  ∃ A : Point, ∀ (M N : Point), diameter M N Q → ∠MAN = 60 :=
sorry

end angle_man_fixed_l154_154233


namespace animal_count_l154_154898

theorem animal_count (dogs : ℕ) (cats : ℕ) (birds : ℕ) (fish : ℕ)
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) : 
  dogs + cats + birds + fish = 39 :=
by
  sorry

end animal_count_l154_154898


namespace conic_is_ellipse_l154_154949

noncomputable def is_conic_ellipse (x y : ℝ) : Prop :=
  sqrt ((x - 2)^2 + (y + 2)^2) + sqrt ((x - 6)^2 + y^2) = 12

theorem conic_is_ellipse : ∀ (x y : ℝ), is_conic_ellipse x y ↔ E :=
by sorry

end conic_is_ellipse_l154_154949


namespace significant_figures_and_precision_l154_154821

-- Definition of the function to count significant figures
def significant_figures (n : Float) : Nat :=
  -- Implementation of a function that counts significant figures
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- Definition of the function to determine precision
def precision (n : Float) : String :=
  -- Implementation of a function that returns the precision
  -- Skipping actual implementation, assuming it is correct.
  sorry

-- The target number
def num := 0.03020

-- The properties of the number 0.03020
theorem significant_figures_and_precision :
  significant_figures num = 4 ∧ precision num = "ten-thousandth" :=
by
  sorry

end significant_figures_and_precision_l154_154821


namespace range_of_solutions_l154_154570

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l154_154570


namespace max_profit_allocation_l154_154884

theorem max_profit_allocation :
  ∃ (f g : ℝ → ℝ) (k1 k2 : ℝ),
  (∀ x, f(x) = k1 * x) ∧
  (f 40 = 20) ∧
  (∀ x, g(x) = k2 * sqrt x) ∧
  (g 40 = 10) ∧
  (f(100 * 9.75/100) + g(100 * 0.25/100) = 41 / 8) :=
sorry

end max_profit_allocation_l154_154884


namespace rationalize_denominator_theorem_l154_154798

noncomputable def rationalize_denominator : Prop :=
  let num := 5
  let den := 2 + Real.sqrt 5
  let conj := 2 - Real.sqrt 5
  let expr := (num * conj) / (den * conj)
  expr = -10 + 5 * Real.sqrt 5

theorem rationalize_denominator_theorem : rationalize_denominator :=
  sorry

end rationalize_denominator_theorem_l154_154798


namespace probability_of_two_even_numbers_l154_154221

-- Define the set of numbers
def numbers := {1, 2, 3, 4, 5, 6}

-- Define the set of even numbers in the given set
def even_numbers := {2, 4, 6}

-- Define the combination function (Mathlib likely has this as a built-in function)
noncomputable def combination (n k : ℕ) : ℕ :=
  nat.choose n k

-- Total number of ways to choose 2 numbers from the set
def total_outcomes := combination 6 2

-- Number of ways to choose 2 even numbers from the set of even numbers
def even_outcomes := combination 3 2

-- The probability of selecting two even numbers
def probability_even : ℚ := even_outcomes / total_outcomes

-- The statement we want to prove
theorem probability_of_two_even_numbers :
  probability_even = 1 / 3 :=
by
  sorry

end probability_of_two_even_numbers_l154_154221


namespace stickers_total_correct_l154_154836

-- Define the conditions
def stickers_per_page : ℕ := 10
def pages_total : ℕ := 22

-- Define the total number of stickers
def total_stickers : ℕ := pages_total * stickers_per_page

-- The statement we want to prove
theorem stickers_total_correct : total_stickers = 220 :=
by {
  sorry
}

end stickers_total_correct_l154_154836


namespace DM_is_altitude_of_tetrahedron_l154_154714

-- Given definitions and assumptions

variable {V : Type*} [EuclideanGeometry V]

structure RegularTetrahedron (A B C D : V) : Prop :=
  regular : IsRegularTetrahedron A B C D

variable {A B C D M : V}
variable [RegularTetrahedron A B C D]

-- Conditions
variable (M_on_ABC : PointOnPlane M (PlaneSpan A B C))
variable (radii_eq : (CircumsphereRadius (Tetrahedron A B M D) = CircumsphereRadius (Tetrahedron B C M D)) ∧ (CircumsphereRadius (Tetrahedron B C M D) = CircumsphereRadius (Tetrahedron C A M D)))

-- The theorem to prove
theorem DM_is_altitude_of_tetrahedron : IsAltitude D M (Tetrahedron A B C D) :=
sorry

end DM_is_altitude_of_tetrahedron_l154_154714


namespace one_thirds_in_nine_halves_l154_154682

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l154_154682


namespace area_triangle_AED_l154_154729

open_locale classical

theorem area_triangle_AED 
  (A B C D O E : Type) 
  (h1 : ∀ (s : Type), s ∈ {A, B} ∧ s ∈ {C, D} → s = O)
  (h2 : ∀ (m : Type), is_midpoint m B C = E)
  (area_ABO : ℝ := 45)
  (area_ADO : ℝ := 18)
  (area_CDO : ℝ := 69) :
  area_triangle A E D = 75 :=
sorry

end area_triangle_AED_l154_154729


namespace prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l154_154563

-- Define the probability of a genotype given two mixed genotype (rd) parents producing a child.
def prob_genotype_dd : ℚ := (1/2) * (1/2)
def prob_genotype_rr : ℚ := (1/2) * (1/2)
def prob_genotype_rd : ℚ := 2 * (1/2) * (1/2)

-- Assertion that the probability of a child displaying the dominant characteristic (dd or rd) is 3/4.
theorem prob_dominant_trait_one_child : 
  prob_genotype_dd + prob_genotype_rd = 3/4 := sorry

-- Define the probability of two children both being rr.
def prob_both_rr_two_children : ℚ := prob_genotype_rr * prob_genotype_rr

-- Assertion that the probability of at least one of two children displaying the dominant characteristic is 15/16.
theorem prob_at_least_one_dominant_trait_two_children : 
  1 - prob_both_rr_two_children = 15/16 := sorry

end prob_dominant_trait_one_child_prob_at_least_one_dominant_trait_two_children_l154_154563


namespace jasons_tip_l154_154743

theorem jasons_tip :
  ∀ (meal_cost : ℝ) (tax_rate : ℝ) (total_paid : ℝ),
    meal_cost = 15 ∧ tax_rate = 0.2 ∧ total_paid = 20 →
    total_paid - (meal_cost + meal_cost * tax_rate) = 2 :=
by
  intros meal_cost tax_rate total_paid h,
  have h1 : meal_cost = 15 := h.1,
  have h2 : tax_rate = 0.2 := h.2.1,
  have h3 : total_paid = 20 := h.2.2,
  sorry

end jasons_tip_l154_154743


namespace cubic_poly_roots_ratio_l154_154484

noncomputable theory

def cubic_polynomial_with_roots (α β γ : ℝ) := 
  ∃(p : ℝ → ℝ), ∀ x, p(x) = (x - α) * (x - β) * (x - γ)

def derivative_divides_scaled_polynomial (p : ℝ → ℝ) :=
  ∃ (q : ℝ → ℝ), q = p' ∧ ∀ x, p(2*x) = q(x)* h(x)

def find_ratios (α β γ : ℝ) : Prop :=
  α : β : γ = 1 : 1 : -1

theorem cubic_poly_roots_ratio (α β γ : ℝ) 
  (h1 : cubic_polynomial_with_roots α β γ) 
  (h2 : derivative_divides_scaled_polynomial (λ x, (x - α) * (x - β) * (x - γ))) :
  find_ratios α β γ :=
sorry

end cubic_poly_roots_ratio_l154_154484


namespace urn_probability_three_red_three_blue_l154_154557
open Nat

theorem urn_probability_three_red_three_blue :
  let initial_urn := (1, 1) -- 1 red ball, 1 blue ball
  let operations := 4 -- Four drawing and returning operations
  let final_urn := 6 -- Final count of balls in the urn
  let desired_count := (3, 3) -- Desired count of red and blue balls
  -- Prove the probability of having 3 red balls and 3 blue balls
  probability_three_red_three_blue initial_urn operations final_urn = 1 / 5 :=
sorry

end urn_probability_three_red_three_blue_l154_154557


namespace problem_statement_l154_154769

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l154_154769


namespace cylinder_volume_l154_154656

theorem cylinder_volume (r h : ℝ) (π : ℝ) 
  (h_pos : 0 < π) 
  (cond1 : 2 * π * r * h = 100 * π) 
  (cond2 : 4 * r^2 + h^2 = 200) : 
  (π * r^2 * h = 250 * π) := 
by 
  sorry

end cylinder_volume_l154_154656


namespace probability_heads_9_or_more_12_flips_l154_154027

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154027


namespace probability_heads_9_or_more_12_flips_l154_154024

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154024


namespace triangle_square_side_ratio_l154_154527

theorem triangle_square_side_ratio (s_t s_s : ℝ) 
  (h_triangle_area : ∀ s, s_t = s → ∀ t, t = s → ∃ a, A_triangle s_t a = 1)
  (h_square_area : ∀ s, s_s = s → ∀ t, t = s → ∃ b, A_square s_s b = 1)
  (h_area_equal : ∀ s, (∃ a, h_triangle_area s a) → ∃ b, h_square_area s b)
  (h_area_formulas : (A_triangle : ℝ → ℝ) = (λ s, s^2 * (real.sqrt 3) / 4) ∧
                      (A_square : ℝ → ℝ) = (λ s, s^2)) :
  (∀ r, r = s_t / s_s → r = 2 * (1 / (real.sqrt (real.sqrt 3)))) :=
by
  sorry

end triangle_square_side_ratio_l154_154527


namespace slope_at_origin_l154_154605

-- Let f represent the function e^x
def f (x : ℝ) := Real.exp x

-- Define the derivative f' at any point x
def f' (x : ℝ) := Real.exp x

-- The main theorem to state that the derivative of f at x = 0 is 1
theorem slope_at_origin : f' 0 = 1 :=
  by
  -- Proof goes here
  sorry

end slope_at_origin_l154_154605


namespace one_thirds_of_nine_halfs_l154_154687

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l154_154687


namespace problem_1_problem_2_problem_3_l154_154543

-- Definition and proof state for problem 1
theorem problem_1 (a b m n : ℕ) (h₀ : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

-- Definition and proof state for problem 2
theorem problem_2 (a m n : ℕ) (h₀ : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = 13 ∨ a = 7 := by
  sorry

-- Definition and proof state for problem 3
theorem problem_3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
  sorry

end problem_1_problem_2_problem_3_l154_154543


namespace max_value_of_sine_expression_l154_154996

theorem max_value_of_sine_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hπx : x ≤ π) (hπy : y ≤ π) (hπz : z ≤ π) :
  (A = sin (x - y) + sin (y - z) + sin (z - x)) → 
  A ≤ 2 :=
sorry

end max_value_of_sine_expression_l154_154996


namespace problem_1_problem_2_problem_3_l154_154542

-- Definition and proof state for problem 1
theorem problem_1 (a b m n : ℕ) (h₀ : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n := by
  sorry

-- Definition and proof state for problem 2
theorem problem_2 (a m n : ℕ) (h₀ : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = 13 ∨ a = 7 := by
  sorry

-- Definition and proof state for problem 3
theorem problem_3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
  sorry

end problem_1_problem_2_problem_3_l154_154542


namespace grazing_area_maximized_at_B_l154_154142
noncomputable theory

-- Defining the positions of the stakes
def A := (0, 0)
def B := (3, 0)
def C := (6, 0)
def D := (9, 0)

-- Define a function for grazing area calculation
def grazing_area (stake : ℤ × ℤ) : ℝ :=
  if stake = A ∨ stake = C ∨ stake = D then
    (1 / 4) * Real.pi * (4 ^ 2) + (1 / 4) * Real.pi * (1 ^ 2)
  else if stake = B then
    (1 / 2) * Real.pi * (4 ^ 2)
  else
    0 -- In case of invalid stake

-- The main statement to prove
theorem grazing_area_maximized_at_B :
  ∀ (a b c d : ℤ × ℤ), 
    a = A ∧ b = B ∧ c = C ∧ d = D →
    grazing_area A ≤ grazing_area B ∧
    grazing_area C ≤ grazing_area B ∧
    grazing_area D ≤ grazing_area B :=
by
  sorry

end grazing_area_maximized_at_B_l154_154142


namespace area_midpoint_quadrilateral_half_l154_154678

-- Given a concave quadrilateral ABCD
variables {A B C D E F G H : Point}

-- Assume E, F, G, H are the midpoints of the sides of quadrilateral ABCD
axiom midpoint_E : midpoint E A B
axiom midpoint_F : midpoint F B C
axiom midpoint_G : midpoint G C D
axiom midpoint_H : midpoint H D A

-- Define the area of quadrilaterals
noncomputable def area_quadrilateral (p1 p2 p3 p4 : Point) : ℝ := -- area computation function

-- The theorem statement
theorem area_midpoint_quadrilateral_half (A B C D : Point) (E F G H : Point)
  (hE : midpoint E A B) (hF : midpoint F B C) (hG : midpoint G C D) (hH : midpoint H D A) :
  area_quadrilateral E F G H = (1 / 2) * area_quadrilateral A B C D :=
by sorry

end area_midpoint_quadrilateral_half_l154_154678


namespace constant_term_expansion_l154_154728

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem constant_term_expansion : (x : ℂ) (h : x ≠ 0) : 
  let general_term := λ r, (2^r) * (binomial_coeff 6 r) * (x^(6 - (3 * r / 2))) in 
  general_term 4 = 240 := 
by 
  sorry

end constant_term_expansion_l154_154728


namespace brian_distance_more_miles_l154_154100

variables (s t d m n : ℝ)
-- Mike's distance
variable (hd : d = s * t)
-- Steve's distance condition
variable (hsteve : d + 90 = (s + 6) * (t + 1.5))
-- Brian's distance
variable (hbrian : m = (s + 12) * (t + 3))

theorem brian_distance_more_miles :
  n = m - d → n = 200 :=
sorry

end brian_distance_more_miles_l154_154100


namespace total_meal_cost_with_tip_l154_154166

-- Definitions based on conditions
def cost_appetizer : ℝ := 9.00
def cost_entree : ℝ := 20.00
def num_entrees : ℕ := 2
def cost_dessert : ℝ := 11.00
def tip_percentage : ℝ := 0.30

-- Formal declaration to prove the entire price of the meal including the tip
theorem total_meal_cost_with_tip :
  let total_appetizer := cost_appetizer,
      total_entrees := cost_entree * num_entrees,
      total_dessert := cost_dessert,
      meal_cost := total_appetizer + total_entrees + total_dessert,
      tip_amount := meal_cost * tip_percentage,
      total_cost := meal_cost + tip_amount
  in total_cost = 78.00 := by sorry

end total_meal_cost_with_tip_l154_154166


namespace coffee_weekly_spending_l154_154384

-- Definitions of parameters based on given conditions.
def daily_cups : ℕ := 2
def ounces_per_cup : ℝ := 1.5
def days_per_week : ℕ := 7
def ounces_per_bag : ℝ := 10.5
def cost_per_bag : ℝ := 8
def weekly_milk_gallons : ℝ := 1 / 2
def cost_per_gallon_milk : ℝ := 4

-- Theorem statement of the proof problem.
theorem coffee_weekly_spending : 
  daily_cups * ounces_per_cup * days_per_week / ounces_per_bag * cost_per_bag +
  weekly_milk_gallons * cost_per_gallon_milk = 18 :=
by sorry

end coffee_weekly_spending_l154_154384


namespace lines_concurrent_or_parallel_l154_154765

variables {A B C D M N : Point}
variables {AB : Line Segment} {CD : Line Segment} {AD : Line Segment} {BC : Line Segment}
variables {Δ : Line}

-- Defining the points and the lines
def midpoint (P Q : Point) : Point := sorry -- Assuming the midpoint is defined
def trapezoid (A B C D : Point) : Prop := sorry -- Definition of a trapezoid
def parallel (l₁ l₂ : Line) : Prop := sorry -- Definition of parallel lines
def concurrent (l₁ l₂ l₃ : Line) : Prop := sorry -- Definition of concurrent lines

theorem lines_concurrent_or_parallel 
  (trapezoid_ABCD : trapezoid A B C D)
  (parallel_AB_CD : parallel (Line_through A B) (Line_through C D))
  (M_midpoint_AB : M = midpoint A B)
  (N_midpoint_CD : N = midpoint C D)
  (Δ_through_MN : Δ = Line_through M N) :
  concurrent Δ (Line_through A D) (Line_through B C) ∨
  parallel (Line_through A D) (Line_through B C) :=
  sorry

end lines_concurrent_or_parallel_l154_154765


namespace expression_evaluation_l154_154112

theorem expression_evaluation : 
  let sqrt_two := Real.sqrt 2
  let sqrt_three := Real.sqrt 3
  let tan_60 := Real.tan (Real.pi / 3)
  sqrt_two < sqrt_three →
  tan_60 = sqrt_three →
  (| sqrt_two - sqrt_three | - tan_60 + 1 / sqrt_two) = - sqrt_two / 2 :=
by
  sorry

end expression_evaluation_l154_154112


namespace range_of_g_l154_154615

variable {A : ℝ}

theorem range_of_g (hA : ∀ n : ℤ, A ≠ n * π / 2) : 
  (∃ x, x = cos A ^ 3 + 5 ∧ x ∈ Ioo 5 6) :=
by sorry

end range_of_g_l154_154615


namespace sam_daisy_weight_difference_l154_154449

theorem sam_daisy_weight_difference :
  ∃ (Sam Daisy : ℝ),
  let Jack := 52 in
  let total_weight := 240 in
  let average_weight := 60 in
  let Lisa := 1.4 * Jack in
  Jack = 0.8 * Sam ∧
  Daisy = (1/3) * (Jack + Lisa) ∧
  Jack + Sam + Lisa + Daisy = total_weight ∧
  average_weight = total_weight / 4 →
  Sam - Daisy = 23.4 :=
by
  sorry

end sam_daisy_weight_difference_l154_154449


namespace probability_all_heads_or_tails_l154_154207

def coin_outcomes := {heads, tails}

def total_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes : ℕ := 2

def probability_five_heads_or_tails (n : ℕ) (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_all_heads_or_tails :
  probability_five_heads_or_tails 5 (total_outcomes 5) favorable_outcomes = 1 / 16 :=
by
  sorry

end probability_all_heads_or_tails_l154_154207


namespace union_complement_eq_l154_154786

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l154_154786


namespace hyperlandia_harmonious_l154_154721

-- Defining the binary sequence and Hamming distance
def binary_seq (n : ℕ) := Vector ℕ n

def hamming_distance {n : ℕ} (x y : binary_seq n) : ℕ :=
(x.to_list.zip y.to_list).count (λ p, p.1 ≠ p.2)

noncomputable def median_vertex {n : ℕ} (x y z : binary_seq n) : binary_seq n :=
Vector.of_fn (λ i, if x.nth i = y.nth i then x.nth i else if x.nth i = z.nth i then x.nth i else y.nth i)

-- The statement to be proved
theorem hyperlandia_harmonious (n : ℕ) (x y z : binary_seq n) :
  ∃ m : binary_seq n, hamming_distance x m + hamming_distance y m + hamming_distance z m ≤
                      hamming_distance x y + hamming_distance x z + hamming_distance y z :=
by
  use median_vertex x y z
  sorry

end hyperlandia_harmonious_l154_154721


namespace lambda_mu_product_correct_l154_154735

noncomputable def lambda_mu_product (A B C P : Type*) [has_vector_space ℝ A B C P]
  (angle_A : A × B × C → Prop)
  (side_AC : ℝ)
  (side_AB : ℝ)
  (perp : A × P × (B × C) → Prop)
  (lambda : ℝ)
  (mu : ℝ)
  (AP_decomp : ∀ (AB AC AP : VectorSpace ℝ), AP = λ AB + μ AC)
  (triangle : Prop)
  : Prop :=
  triangle ∧ (angle_A (A, B, C) = 90) ∧ (side_AC = 1) ∧ (side_AB = 2) ∧
  (perp (A, P, B, C)) ∧ (AP_decomp = λ AB + μ AC) → (λ * μ = 4 / 25)

axiom triangle_ABC : types :: Triangle ABC
axiom angle_A_spec : ∠A = 90
axiom side_AC_spec : AC = 1
axiom side_AB_spec : AB = 2
axiom perp_spec : line_through A P is_perpendicular B C
axiom AP_decomp_spec : AP = λ AB + μ AC

theorem lambda_mu_product_correct : lambda_mu_product 
  triangle 
  angle_A_spec 
  side_AC_spec 
  side_AB_spec 
  perp_spec
  AP_decomp_spec :=
  sorry

end lambda_mu_product_correct_l154_154735


namespace tangent_line_eq_l154_154267

def f (x : ℝ) : ℝ := x^3 - x^2 - 3 * x + 3

theorem tangent_line_eq (x : ℝ) (f : ℝ → ℝ): (f 1 = 0) → 
                         ((deriv f 1 = -2) → 
                         (∀ x, (λ y, y - 0 = -2 * (x - 1)) = (λ y, y = -2 * x + 2))) :=
by
  intro h1
  intro h2
  exact sorry

end tangent_line_eq_l154_154267


namespace annie_serious_accident_probability_l154_154918

theorem annie_serious_accident_probability :
  (∀ temperature : ℝ, temperature < 32 → ∃ skid_chance_increase : ℝ, skid_chance_increase = 5 * ⌊ (32 - temperature) / 3 ⌋ / 100) →
  (∀ control_regain_chance : ℝ, control_regain_chance = 0.4) →
  (∀ control_loss_chance : ℝ, control_loss_chance = 1 - control_regain_chance) →
  (temperature = 8) →
  (serious_accident_probability = skid_chance_increase * control_loss_chance) →
  serious_accident_probability = 0.24 := by
  sorry

end annie_serious_accident_probability_l154_154918


namespace quadratic_unique_root_l154_154371

theorem quadratic_unique_root (b c : ℝ)
  (h₁ : b = c^2 + 1)
  (h₂ : (x^2 + b * x + c = 0) → ∃! x : ℝ, x^2 + b * x + c = 0) :
  c = 1 ∨ c = -1 := 
sorry

end quadratic_unique_root_l154_154371


namespace range_of_a_l154_154988

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then |x - 2 * a| else x + 1 / (x - 2) + a

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f a 2 ≤ f a x) : 1 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_a_l154_154988


namespace sequence_formula_correct_l154_154218

noncomputable def S (n : ℕ) : ℕ := 2^n - 3

def a (n : ℕ) : ℤ :=
  if n = 1 then -1
  else 2^(n-1)

theorem sequence_formula_correct (n : ℕ) :
  a n = (if n = 1 then -1 else 2^(n-1)) :=
by
  sorry

end sequence_formula_correct_l154_154218


namespace intersection_of_sets_example_l154_154359

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l154_154359


namespace smaller_than_neg3_l154_154153

theorem smaller_than_neg3 :
  (∃ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3) ∧ ∀ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3 → x = -5 :=
by
  sorry

end smaller_than_neg3_l154_154153


namespace ten_digit_number_l154_154956

theorem ten_digit_number (N : ℕ) (digits : Fin 10 → ℕ) : 
  (∀ i, digits i = (N.digits.filter (λ d, d = i)).length) ∧
  (N.digits.length = 10) →
  N = 6210001000 :=
by
  intros h
  cases h with digits_count length_count
  sorry

end ten_digit_number_l154_154956


namespace jerry_age_l154_154386

theorem jerry_age (M J : ℝ) (h₁ : M = 17) (h₂ : M = 2.5 * J - 3) : J = 8 :=
by
  -- The proof is omitted.
  sorry

end jerry_age_l154_154386


namespace interest_payment_frequency_is_three_months_l154_154739

-- Definitions specific to the problem conditions
def principal : ℝ := 10000
def annual_rate : ℝ := 0.095
def total_period_years : ℝ := 1.5
def interest_per_payment : ℝ := 237.5

-- Statements to be proved
def total_interest : ℝ := principal * annual_rate * total_period_years
def num_payments : ℝ := total_interest / interest_per_payment
def frequency_payments : ℝ := 18 / num_payments

-- The final proof statement
theorem interest_payment_frequency_is_three_months : frequency_payments = 3 :=
sorry

end interest_payment_frequency_is_three_months_l154_154739


namespace probability_heads_at_least_9_of_12_flips_l154_154005

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154005


namespace smallest_integer_with_16_divisors_l154_154048

-- Define prime factorization and the function to count divisors
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def number_of_divisors (n : ℕ) : ℕ :=
  (prime_factorization n).foldr (λ ⟨_, a⟩ acc, acc * (a + 1)) 1

-- Main theorem stating the smallest positive integer with exactly 16 divisors is 210
theorem smallest_integer_with_16_divisors : 
  ∃ n, n > 0 ∧ number_of_divisors n = 16 ∧ ∀ m, m > 0 ∧ number_of_divisors m = 16 → n ≤ m :=
begin
  use 210,
  split,
  { -- Prove 210 > 0
    exact nat.zero_lt_succ _,
  },
  split,
  { -- Prove number_of_divisors 210 = 16
    sorry,
  },
  { -- Prove minimality
    intros m hm1 hm2,
    sorry,
  }
end

end smallest_integer_with_16_divisors_l154_154048


namespace parabola_translation_correct_l154_154844

variable (x : ℝ)

def original_parabola : ℝ := 5 * x^2

def translated_parabola : ℝ := 5 * (x - 2)^2 + 3

theorem parabola_translation_correct :
  translated_parabola x = 5 * (x - 2)^2 + 3 :=
by
  sorry

end parabola_translation_correct_l154_154844


namespace find_a_find_m_l154_154665

-- Conditions: f(x) is an odd function with domain \(\mathbb{R}\)
def f (x : ℝ) (a : ℝ) : ℝ := (a - real.exp x) / (real.exp x + a)

-- Proof problem 1: Prove that a = 1 given f(x) is odd
theorem find_a (a : ℝ) (h : ∀ x : ℝ, f x a = - f (-x) a) : a = 1 :=
sorry

-- Conditions: f(2^(x+1) - 4^x) + f(1 - m) > 0 always holds for all x ∈ [1, 2]
def g (x : ℝ) : ℝ := f (2^(x + 1) - 4^x) 1 -- f uses the value a = 1

-- Proof problem 2: Prove that the range of real number m is (1, ∞)
theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → g x + f (1 - m) 1 > 0) : 1 < m :=
sorry

end find_a_find_m_l154_154665


namespace composite_exists_for_x_64_l154_154232

-- Define the conditions
def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

-- Main statement
theorem composite_exists_for_x_64 :
  ∃ n : ℕ, is_composite (n^4 + 64) :=
sorry

end composite_exists_for_x_64_l154_154232


namespace range_of_f_l154_154945

-- Definition of the operation
def op (a b : ℝ) : ℝ :=
if a ≤ b then a else b

-- Definition of the function f(x)
def f (x : ℝ) : ℝ :=
op (Real.cos x) (Real.sin x)

-- Theorem statement to prove the range of f(x)
theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) (Real.sqrt 2 / 2) :=
sorry

end range_of_f_l154_154945


namespace num_not_divisible_by_5_l154_154549

-- Defining the set of digits and the conditions
def digits := {0, 1, 2, 5}
def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
def no_repetition (n : List ℕ) : Prop := n.nodup

-- Define the property of not being divisible by 5
def not_divisible_by_5 (n : ℕ) : Prop := ¬ (n % 5 = 0)

-- Main statement to be proven
theorem num_not_divisible_by_5 : 
  ∃ n, is_four_digit n ∧ no_repetition (n.digits 10) ∧ not_divisible_by_5 n ∧ (count n = 8) :=
sorry

end num_not_divisible_by_5_l154_154549


namespace triangle_area_l154_154471

open Real

theorem triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : c = 3) :
    let s := (a + b + c) / 2 in
    let area := sqrt (s * (s - a) * (s - b) * (s - c)) in
    area = 6 := by
sory

end triangle_area_l154_154471


namespace smallest_integer_with_16_divisors_l154_154078

theorem smallest_integer_with_16_divisors : 
  ∃ (n : ℕ), (∃ (p_1 p_2 p_3 : ℕ) (a_1 a_2 a_3 : ℕ), 
  (p_1 = 2 ∧ p_2 = 3 ∧ a_1 = 3 ∧ a_2 = 3 ∧ n = p_1 ^ a_1 * p_2 ^ a_2) ∧
  (∀ m, m > 0 → (∃ b1 b2 ..., m has exactly 16 positive divisors) → 216 ≤ m)) := 
sorry

end smallest_integer_with_16_divisors_l154_154078


namespace solution_set_inequality_l154_154426

theorem solution_set_inequality (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_f_neg4 : f (-4) = 0) (h_f_1 : f 1 = 0)
  (h_decreasing_0_3 : ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → y ≤ 3 → f x ≤ f y)
  (h_increasing_3_inf : ∀ ⦃x y⦄, 3 ≤ x → x ≤ y → f x ≤ f y) :
  {x : ℝ | x^3 * f x < 0} = (Iio (-4)) ∪ Ioo (-1) 0 ∪ Ioo 1 4 :=
by sorry

end solution_set_inequality_l154_154426


namespace relationship_of_arithmetic_progression_l154_154703

theorem relationship_of_arithmetic_progression (x y z d : ℝ) (h1 : x + (y - z) + d = y + (z - x))
    (h2 : y + (z - x) + d = z + (x - y))
    (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
    x = y + d / 2 ∧ z = y + d := by
  sorry

end relationship_of_arithmetic_progression_l154_154703


namespace middle_segment_of_triangle_l154_154909

theorem middle_segment_of_triangle (b : ℝ) (h₁ : b = 18)
  (h₂ : ∀ A₁ A₂ A₃ : ℝ, A₁ : ℝ = 1 / 4) (A₂ = 1 / 2) (A₃ = 1 / 4): 
  ∃ PQ : ℝ, PQ = 9 * Real.sqrt 3 :=
by {
  use 9 * Real.sqrt 3,
  sorry
}

end middle_segment_of_triangle_l154_154909


namespace division_of_fractions_l154_154853

theorem division_of_fractions :
  (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 :=
by
  sorry

end division_of_fractions_l154_154853


namespace number_of_games_in_division_l154_154506

theorem number_of_games_in_division (P Q : ℕ) (h1 : P > 2 * Q) (h2 : Q > 6) (schedule_eq : 4 * P + 5 * Q = 82) : 4 * P = 52 :=
by sorry

end number_of_games_in_division_l154_154506


namespace stratified_sampling_l154_154143

section StratifiedSampling

variable (total_employees : ℕ) -- Total number of employees
variable (under_35 : ℕ) -- Number of employees under 35 years old
variable (between_35_and_49 : ℕ) -- Number of employees between 35 and 49 years old
variable (fifty_and_above : ℕ) -- Number of employees 50 years old or above
variable (sample_size : ℕ) -- Total sample size needed

-- Given conditions
def conditions : Prop := (total_employees = 500) ∧
                         (under_35 = 125) ∧
                         (between_35_and_49 = 280) ∧
                         (fifty_and_above = 95) ∧
                         (sample_size = 100)

-- Expected number of samples from each group using stratified sampling
def expected_samples_under_35 : ℕ := (sample_size * under_35) / total_employees
def expected_samples_between_35_and_49 : ℕ := (sample_size * between_35_and_49) / total_employees
def expected_samples_fifty_and_above : ℕ := (sample_size * fifty_and_above) / total_employees

-- Proof problem statement
theorem stratified_sampling (h : conditions) :
  expected_samples_under_35 total_employees under_35 sample_size = 25 ∧
  expected_samples_between_35_and_49 total_employees between_35_and_49 sample_size = 56 ∧
  expected_samples_fifty_and_above total_employees fifty_and_above sample_size = 19 :=
by
  sorry

end StratifiedSampling

end stratified_sampling_l154_154143


namespace find_smallest_n_general_term_a_l154_154278

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0 else n^2 + 2 * n

noncomputable def b (n : ℕ) : ℕ :=
  if n = 0 then 0 else log2 (n^2 + n) - log2 (a n)

noncomputable def S (n : ℕ) : ℕ :=
  if n = 0 then 0 else finset.sum (finset.range n) (λk, b (k + 1))

theorem find_smallest_n (n : ℕ) (h₀ : 1 - log2 (n+2) < -4) : n > 30 :=
sorry

theorem general_term_a (n : ℕ) (h₀ : n ≠ 0) : a n = n^2 + 2 * n :=
sorry

end find_smallest_n_general_term_a_l154_154278


namespace triangle_area_ratio_l154_154706

noncomputable def area_ratio (AD DC : ℝ) (h : ℝ) : ℝ :=
  (1 / 2) * AD * h / ((1 / 2) * DC * h)

theorem triangle_area_ratio (AD DC : ℝ) (h : ℝ) (condition1 : AD = 5) (condition2 : DC = 7) :
  area_ratio AD DC h = 5 / 7 :=
by
  sorry

end triangle_area_ratio_l154_154706


namespace combined_volume_of_two_positions_l154_154242

/-- Define a regular pentagonal prism with unit edge length placed twice successively on a square plate of unit edge length
    and prove the combined volume of the space occupied by the two positions of the prism is 2.144 volume units -/
theorem combined_volume_of_two_positions : 
  let edge_length := 1
  let prism_base_area := (5 / 4) * (Real.tan (54 * Real.pi / 180)) -- Area of pentagonal base
  let single_prism_volume := prism_base_area -- Since edge length (height) is 1
  let overlapping_volume := 1.2967 /- Calculated using the given problem's overlapping volume formula -/
  in 2 * single_prism_volume - overlapping_volume = 2.144 :=
by
  let edge_length := 1
  let prism_base_area := (5 / 4) * (Real.tan (54 * Real.pi / 180))
  let single_prism_volume := prism_base_area
  let overlapping_volume := 1.2967
  have h: 2 * single_prism_volume - overlapping_volume = 2.144 := sorry
  exact h

end combined_volume_of_two_positions_l154_154242


namespace speed_ratio_l154_154114

theorem speed_ratio (a b v1 v2 S : ℝ) (h1 : S = a * (v1 + v2)) (h2 : S = b * (v1 - v2)) (h3 : a ≠ b) : 
  v1 / v2 = (a + b) / (b - a) :=
by
  -- proof skipped
  sorry

end speed_ratio_l154_154114


namespace distribution_percentage_l154_154486

-- Define the conditions in Lean
variables (m : ℝ) (d : ℝ)

-- Define percentages as probabilities
variables (distribution_within_one_sd : ℝ) (distribution_less_than_m_plus_d : ℝ)

-- Express the condition that 60% of the distribution lies within one standard deviation
axiom (H1 : distribution_within_one_sd = 0.60)

-- Goal: We want to prove that 80% of the distribution is less than m + d
theorem distribution_percentage (H1 : distribution_within_one_sd = 0.60) :
    distribution_less_than_m_plus_d = 0.80 :=
sorry

end distribution_percentage_l154_154486


namespace molecular_weight_of_one_mole_l154_154047

theorem molecular_weight_of_one_mole (molecular_weight_3_moles : ℕ) (h : molecular_weight_3_moles = 222) : (molecular_weight_3_moles / 3) = 74 := 
by
  sorry

end molecular_weight_of_one_mole_l154_154047


namespace distance_between_points_l154_154193

theorem distance_between_points :
  let x1 := 1
  let y1 := 16
  let x2 := 9
  let y2 := 3
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = Real.sqrt 233 :=
by
  sorry

end distance_between_points_l154_154193


namespace area_of_rectangle_l154_154191

variable (a b c d : ℝ)
variables (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0)

theorem area_of_rectangle :
  let length := a - (a - 2 * b),
      width := d - (-2 * c),
      area := length * width 
  in area = 2 * b * d + 4 * b * c := 
by
  sorry

end area_of_rectangle_l154_154191


namespace one_thirds_in_nine_halves_l154_154680

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l154_154680


namespace sequence_property_l154_154731

noncomputable def sequence (n : ℕ) : ℚ
| 0 => 1
| 1 => 2 / 3
| (n + 2) => 2 / (sequence n + sequence (n + 1))

theorem sequence_property :
  sequence 5 = 2 / 7 :=
by
  sorry

end sequence_property_l154_154731


namespace frac_part_sum_l154_154978

def floor (x : ℝ) : ℤ := x.to_int

noncomputable def frac_part (x : ℝ) : ℝ := x - ↑(floor x)

theorem frac_part_sum :
  frac_part 3.8 + frac_part (-1.7) - frac_part 1 = 1.1 :=
by sorry

end frac_part_sum_l154_154978


namespace total_factors_a3_b4_c5_l154_154459

-- Definitions of a, b, c having exactly 3 divisors, hence they are squares of distinct primes
variables {a b c : ℕ}
variables {q1 q2 q3 : ℕ}

-- Conditions: a = q1^2, b = q2^2, c = q3^2, where q1, q2, q3 are distinct primes
noncomputable def is_prime (n : ℕ) : Prop := n = 2 ∨ ∃ m, nat.prime m ∧ n = m
axiom q1_prime : is_prime q1
axiom q2_prime : is_prime q2
axiom q3_prime : is_prime q3
axiom distinct_primes : q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3

-- a, b, c having exactly 3 natural number factors means they are squares of primes
axiom a_def : a = q1^2
axiom b_def : b = q2^2
axiom c_def : c = q3^2

-- Mathematical fact about the number of divisors
def number_of_factors (n : ℕ) : ℕ :=
(n.dvd.card - 1).succ

-- Main theorem statement
theorem total_factors_a3_b4_c5 : number_of_factors (a^3 * b^4 * c^5) = 693 :=
by
  sorry

end total_factors_a3_b4_c5_l154_154459


namespace union_complement_eq_l154_154777

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l154_154777


namespace find_p_l154_154390

noncomputable def ellipse : set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}
def F : ℝ × ℝ := (1, 0)
def P (p : ℝ) : ℝ × ℝ := (p, 0)

theorem find_p :
  ∃ p : ℝ, p > 0 ∧ (∀ (A B : ℝ × ℝ), A ∈ ellipse → B ∈ ellipse → 
    (A.2 / (A.1 - p) = - B.2 / (B.1 - p)) → p = 2) :=
begin
  use 2,
  split,
  { linarith, }, -- p > 0
  { intros A B helA helB hangle,
    sorry, -- Proof omitted
  }
end

end find_p_l154_154390


namespace camel_movement_impossible_l154_154797

/-- Prove that it is impossible for a camel to traverse each field of a 25-field board exactly
once starting from an octagonal field given the movement rules.
-/
theorem camel_movement_impossible
  (fields : Fin 25) (colors : Fin 25 → color)
  (start : Fin 25) (move_rule : Fin 25 → Fin 25)
  (octagonal_start : is_octagonal start) :
  ¬ (∀ field : Fin 25, ∃ path : List (Fin 25), path.head = start ∧ path.Nodup ∧ path.length = 25 ∧ path.all (move_rule := move_rule)) :=
  sorry

end camel_movement_impossible_l154_154797


namespace area_covered_by_three_layers_of_carpets_l154_154829

theorem area_covered_by_three_layers_of_carpets :
  let hall_width := 10
  let hall_height := 10
  let carpet1_width := 6
  let carpet1_height := 8
  let carpet2_width := 6
  let carpet2_height := 6
  let carpet3_width := 5
  let carpet3_height := 7

  ∃ (total_area : ℕ), total_area = 6 :=
begin
  sorry
end

end area_covered_by_three_layers_of_carpets_l154_154829


namespace percent_of_value_is_correct_l154_154470

theorem percent_of_value_is_correct :
  (360 * 0.42 = 151.2) :=
by { norm_num }

end percent_of_value_is_correct_l154_154470


namespace max_value_sin_cos_sum_l154_154249

theorem max_value_sin_cos_sum (n : ℕ) (x : Fin n → ℝ) :
  let A := (Finset.univ.sum (fun i => Real.sin (x i))) *
           (Finset.univ.sum (fun i => Real.cos (x i)))
  in A ≤ n ^ 2 / 2 :=
by
  sorry

end max_value_sin_cos_sum_l154_154249


namespace locus_of_C_is_intersection_line_l154_154632

-- Definitions and Assumptions
variable (α : Plane) (A B : Point) (AB : Line)
variable (h1 : A ∉ α) (h2 : B ∈ α)
variable (hAB : AB.contains A ∧ AB.contains B)
variable (l : ∀ (p : Point), p ≠ A → Line) -- moving line l

-- Constraints
axiom h_l_perp_AB : ∀ p ≠ A, (l p)⊥AB
axiom h_l_through_A : ∀ p ≠ A, A ∈ l p 

-- Point C where line l intersects plane α
axiom C : ∀ (p : Point), p ≠ A → ∃ C, C ∈ α ∧ C ∈ l p

-- Plane γ is defined as the plane passing through point A and perpendicular to line AB
axiom γ : Plane
axiom h_γ_through_A : A ∈ γ 
axiom h_γ_perp_AB : γ⊥AB

-- Intersection line of plane γ and plane α
axiom intersection_line : Line
axiom h_intersection_is_line : ¬parallel α γ 
axiom h_intersection_contains_C : ∀ (p : Point), p ≠ A → ∃ intersection_line, C p ∈ intersection_line

-- Lean 4 Theorem Statement
theorem locus_of_C_is_intersection_line :
  ∀ (p : Point), p ≠ A → ∃ intersection_line, C p ∈ intersection_line :=
by
  sorry

end locus_of_C_is_intersection_line_l154_154632


namespace range_of_solutions_l154_154578

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l154_154578


namespace possible_values_f_one_l154_154762

noncomputable def f (x : ℝ) : ℝ := sorry

variables (a b : ℝ)
axiom f_equation : ∀ x y : ℝ, 
  f ((x - y) ^ 2) = a * (f x)^2 - 2 * x * f y + b * y^2

theorem possible_values_f_one : f 1 = 1 ∨ f 1 = 2 :=
sorry

end possible_values_f_one_l154_154762


namespace intervals_of_increase_of_f_l154_154602

theorem intervals_of_increase_of_f :
  ∀ k : ℤ,
  ∀ x y : ℝ,
  k * π - (5 / 8) * π ≤ x ∧ x ≤ y ∧ y ≤ k * π - (1 / 8) * π →
  3 * Real.sin ((π / 4) - 2 * x) - 2 ≤ 3 * Real.sin ((π / 4) - 2 * y) - 2 :=
by
  sorry

end intervals_of_increase_of_f_l154_154602


namespace union_A_B_complement_intersection_A_B_l154_154282

open Set

noncomputable def U := univ : Set ℝ

noncomputable def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}
noncomputable def B : Set ℝ := {x | 2x - 9 ≥ 6 - 3x}

theorem union_A_B : A ∪ B = {x | x ≥ 2} :=
by
  sorry

theorem complement_intersection_A_B : U \ (A ∩ B) = {x | x < 3 ∨ x ≥ 4} :=
by
  sorry

end union_A_B_complement_intersection_A_B_l154_154282


namespace intersection_of_sets_example_l154_154358

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l154_154358


namespace sum_of_valid_four_digit_numbers_l154_154617

-- Conditions
def is_valid_digit (d : ℕ) : Prop := d = 3 ∨ d = 6 ∨ d = 9
def is_valid_four_digit_number (n : ℕ) : Prop := 
  1000 ≤ n ∧ n < 10000 ∧ 
  is_valid_digit (n / 1000) ∧ 
  is_valid_digit ((n % 1000) / 100) ∧ 
  is_valid_digit ((n % 100) / 10) ∧ 
  is_valid_digit (n % 10)

-- Proof problem
theorem sum_of_valid_four_digit_numbers : 
  (∑ n in Finset.filter is_valid_four_digit_number (Finset.range 10000), n) = 539946 :=
by
  sorry

end sum_of_valid_four_digit_numbers_l154_154617


namespace remaining_number_not_zero_l154_154792

theorem remaining_number_not_zero:
  ∀ (nums : List ℕ), (∀ n ∈ nums, 1 ≤ n ∧ n ≤ 2019) →
  (∀ n ∈ nums, 2 ≤ nums.length) →
  ((λ op (nums : List ℕ), ∃ a b, a ∈ nums ∧ b ∈ nums ∧
                               nums = nums.erase a ∧ nums = nums.erase b ++ [|a - b|])) →
  (∃ n, n ∈ nums ∧ nums.length = 1 → n ≠ 0) :=
by
  sorry

end remaining_number_not_zero_l154_154792


namespace probability_heads_at_least_9_of_12_flips_l154_154006

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154006


namespace particular_proposition_l154_154481

-- Definitions representing the propositions
def proposition_A : Prop := ∀ f : ℝ → ℝ, even_function f → symmetric_about_y_axis f
def proposition_B : Prop := ∀ p : Prism, right_prism p → parallelepiped p
def proposition_C : Prop := ∀ l1 l2 : Line, non_intersecting l1 l2 → parallel l1 l2
def proposition_D : Prop := ∃ x : ℝ, x ≥ 3

-- The state that we intend to prove
theorem particular_proposition :
  (∃ x : ℝ, x ≥ 3) := sorry

end particular_proposition_l154_154481


namespace regression_analysis_statements_l154_154093

theorem regression_analysis_statements :
  (A_correct ∧ B_correct ∧ C_correct ∧ ¬D_correct) :=
begin
  -- Definitions of the statements based on conditions from part (a)
  let A_correct := ∀ r, (|r| ≤ 1) → 
                    (|r| = 1 → ∃ x y, (∀ a b, y = a * x → a = r) ∧ (0 < |r| < 1 → ∃ u v, y = u * v ∧ 0 < u < 1)),
  let B_correct := ∀ x y (n : ℕ),
                    let mean := λ xs, (list.sum xs) / (list.length xs) in
                    let _ := (∀ b a u, list.length x = n ∧ list.length y = n →
                                (∀ xi yi, list.member xi x ∧ list.member yi y →
                                  yi = b * xi + a) →
                                (mean x, mean y) ∈ set_of y = b * (mean x) + a),
  let C_correct := ∀ (residuals : list ℝ), residuals.nth 0 = (list.sum (residuals.map (λ r, r * r)) / residuals.length),
  let D_correct := ∀ k (K2 : random_variable),
                    let relation := λ X Y, ∑ x y, K2 x y ->
                                    K2.values < 1 / (metric_space.dist X Y) in
                    ¬ relation categorical_var1 categorical_var2,

  -- Proving that A_correct, B_correct, C_correct are true and D_correct is false
  split,
  { intros r h1 h2,
    sorry }, -- Proof for A_correct
  { intros x y n mean b a u h1 h2,
    sorry }, -- Proof for B_correct
  { intros residuals,
    sorry }, -- Proof for C_correct
  { intros k K2 relation h1 h2 h3,
    sorry } -- Proof for D_correct
end

end regression_analysis_statements_l154_154093


namespace root_in_interval_l154_154155

theorem root_in_interval (f : ℝ → ℝ) (a b : ℝ) (h1 : a = 0) (h2 : b = 1) 
  (hfa : f a = -3) (hfb : f b = 2) : ∃ c ∈ Icc a b, f c = 0 :=
by
  let f := λ x : ℝ, x^3 + 4 * x - 3
  sorry

end root_in_interval_l154_154155


namespace kids_milk_consumption_percentage_l154_154177

-- Definitions of the conditions
def total_milk : ℝ := 16
def leftover_milk : ℝ := 2
def kids_percentage (x : ℝ) : ℝ := x / 100
def remaining_milk_after_kids (x : ℝ) : ℝ := total_milk - kids_percentage x * total_milk
def cooked_milk (x : ℝ) : ℝ := remaining_milk_after_kids x / 2

-- Statement of the problem
theorem kids_milk_consumption_percentage (x : ℝ) (h: cooked_milk x = leftover_milk) : x = 75 :=
by
  sorry

end kids_milk_consumption_percentage_l154_154177


namespace probability_heads_at_least_9_of_12_flips_l154_154004

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154004


namespace max_value_A_is_n_squared_over_2_l154_154247

noncomputable def max_value_A (n : ℕ) : ℝ :=
  ∀ (x : Fin n → ℝ), let A := (∑ i in Finset.univ, Real.sin (x i)) * (∑ i in Finset.univ, Real.cos (x i))
  in A ≤ n^2 / 2

theorem max_value_A_is_n_squared_over_2 (n : ℕ) : 
  (∃ (x : Fin n → ℝ), (let A := (∑ i in Finset.univ, Real.sin (x i)) * (∑ i in Finset.univ, Real.cos (x i))
  in A = n^2 / 2)) :=
sorry

end max_value_A_is_n_squared_over_2_l154_154247


namespace right_triangle_third_side_l154_154645

noncomputable def sqrt6 := Real.sqrt 6

def integer_part (x : ℝ) : ℕ := Nat.floor x

def decompose (x : ℝ) : ℕ × ℝ :=
  let b := Nat.floor x
  (b, x - b)

theorem right_triangle_third_side :
  let a := integer_part sqrt6 in
  let ⟨b, c⟩ := decompose (2 + sqrt6) in
  c > 0 ∧ c < 1 →
  b ∈ ℕ →
  (a = 2 ∧ b = 4) →
  (∃ s : ℝ, (s = 2 * Real.sqrt 5 ∨ s = 2 * Real.sqrt 3)) :=
by
  intros a bc decompose_r a_eq_b_eq
  sorry

end right_triangle_third_side_l154_154645


namespace walked_8_miles_if_pace_4_miles_per_hour_l154_154690

-- Define the conditions
def walked_some_miles_in_2_hours (d : ℝ) : Prop :=
  d = 2

def pace_same_4_miles_per_hour (p : ℝ) : Prop :=
  p = 4

-- Define the proof problem
theorem walked_8_miles_if_pace_4_miles_per_hour :
  ∀ (d p : ℝ), walked_some_miles_in_2_hours d → pace_same_4_miles_per_hour p → (p * d = 8) :=
by
  intros d p h1 h2
  rw [h1, h2]
  exact sorry

end walked_8_miles_if_pace_4_miles_per_hour_l154_154690


namespace range_of_a_l154_154272

noncomputable def f (x : ℝ) : ℝ := 8 * x / (1 + x^2)

noncomputable def g (x a : ℝ) : ℝ := x^2 - a * x + 1

theorem range_of_a :
  (∀ x1 : ℝ, 0 ≤ x1 → ∃! x0 : ℝ, -1 ≤ x0 ∧ x0 ≤ 2 ∧ f x1 = g x0 a) ↔ a ∈ (-∞ : ℝ) ∪ (-2) ∪ (5/2 : ℝ) ∪ (+∞ : ℝ) :=
sorry

end range_of_a_l154_154272


namespace intersection_of_M_and_N_l154_154352

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l154_154352


namespace solve_for_range_of_a_l154_154244

-- Definitions for points, line, and circles
structure Point := (x : ℝ) (y : ℝ)

-- Conditions in the given problem
def O : Point := ⟨0, 0⟩
def A : Point := ⟨0, 3⟩
def line_l (p : Point) : Prop := p.y = p.x + 1
def radius_C : ℝ := 1
def center_C_on_line (c : Point) : Prop := line_l c
def on_circle (c : Point) (r : ℝ) (p : Point) : Prop := (p.x - c.x)^2 + (p.y - c.y)^2 = r^2

-- Given condition |MA| = 2|MO|
def distance (p1 p2 : Point) : ℝ := real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)
def condition (m c o a : Point) : Prop := distance m a = 2 * distance m o

-- Lean 4 theorem equivalent to solving the problem
theorem solve_for_range_of_a (a : ℝ) (center_C : Point)
  (hC_on_line : center_C_on_line center_C)
  (center_C_x : center_C.x = a)
  (exists_point_M : ∃ m : Point, on_circle center_C radius_C m ∧ condition m center_C O A) :
  -1 - real.sqrt 7 / 2 ≤ a ∧ a ≤ -1 + real.sqrt 7 / 2 :=
begin
  sorry
end

end solve_for_range_of_a_l154_154244


namespace solution_range_l154_154581

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l154_154581


namespace train_speed_in_kmph_l154_154535

theorem train_speed_in_kmph (time_to_cross: ℕ) (train_length: ℕ) (h_time : time_to_cross = 9) (h_length : train_length = 75) : 
  (train_length / time_to_cross) * 3.6 = 30 :=
by 
  rw [h_time, h_length]
  sorry

end train_speed_in_kmph_l154_154535


namespace constant_term_expansion_l154_154726

theorem constant_term_expansion :
  let general_term (r : ℕ) := (binom 6 r) * (2 ^ r) * (x ^ (6 - (3 * r) / 2))
  let constant_term_in_expansion := general_term 4
  constant_term_in_expansion = 240 := by
  sorry

end constant_term_expansion_l154_154726


namespace coin_flip_probability_l154_154015

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154015


namespace probability_three_mn_sub_m_sub_n_multiple_of_five_l154_154803

open Finset

noncomputable def is_multiple_of_five (m n : ℤ) : Prop :=
  (3 * m * n - m - n) % 5 = 0

noncomputable def probability : ℚ :=
  let s := {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let total_pairs := (s.product s).filter (λ p, p.1 ≠ p.2)
  let valid_pairs := total_pairs.filter (λ p, is_multiple_of_five p.1 p.2)
  (valid_pairs.card : ℚ) / (total_pairs.card : ℚ)

theorem probability_three_mn_sub_m_sub_n_multiple_of_five :
  probability = (2 : ℚ) / (9 : ℚ) :=
sorry

end probability_three_mn_sub_m_sub_n_multiple_of_five_l154_154803


namespace drain_pool_time_correct_l154_154494

def pool_dimensions (width length depth : ℕ) (capacity_percent : ℕ) : Prop :=
  capacity_percent = 80 ∧ depth = 10 ∧ width = 60 ∧ length = 150

def drain_rate (rate : ℕ) : Prop := 
  rate = 60

def time_to_drain_pool (total_volume rate : ℕ) : ℕ :=
  total_volume / rate

theorem drain_pool_time_correct :
  ∀ (width length depth : ℕ) (capacity_percent : ℕ) (rate : ℕ),
  pool_dimensions width length depth capacity_percent →
  drain_rate rate →
  time_to_drain_pool ((8 * width * length * depth) / 10) rate / 60 = 20 :=
by
  intros width length depth capacity_percent rate h_dim h_rate,
  have h_volume := (8 * width * length * depth) / 10,
  have h_time := h_volume / rate,
  have h_time_minutes := h_time / 60,
  sorry

end drain_pool_time_correct_l154_154494


namespace construct_cyclic_quadrilateral_l154_154175

theorem construct_cyclic_quadrilateral 
  (a b c d : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d) 
  (h_cyclic : (a + b + c + d) > 0) : 
  ∃ (A B C D : Point), 
  dist A B = a ∧ 
  dist B C = b ∧ 
  dist C D = c ∧ 
  dist D A = d ∧ 
  cyclic A B C D :=
sorry

end construct_cyclic_quadrilateral_l154_154175


namespace exist_max_gcd_triplet_and_uniqueness_l154_154398

theorem exist_max_gcd_triplet_and_uniqueness:
  ∃ (a b c : ℕ), 
    a + b + c = 1998 ∧
    (∀ (a' b' c' : ℕ), a' + b' + c' = 1998 → gcd (gcd a' b') c' ≤ gcd (gcd a b) c)
    ∧ 0 < a ∧ a < b ∧ b ≤ c ∧ c < 2 * a
    ∧ ∃ (d : ℕ), ∀ (x y z : ℕ), (x, y, z) ∈ {(518, 592, 888), (518, 666, 814), (518, 740, 740), (592, 666, 740)} → a = x ∧ b = y ∧ c = z :=
sorry

end exist_max_gcd_triplet_and_uniqueness_l154_154398


namespace b_minus_a_eq_neg_six_l154_154825

-- Define point and transformations
structure Point where
  x : ℝ
  y : ℝ

def rotate_counterclockwise (P : Point) (C : Point) (θ : ℝ) : Point :=
  ⟨ C.x + (P.x - C.x) * Real.cos θ - (P.y - C.y) * Real.sin θ,
    C.y + (P.x - C.x) * Real.sin θ + (P.y - C.y) * Real.cos θ ⟩

def reflect_y_eq_x (P : Point) : Point :=
  ⟨ P.y, P.x ⟩

noncomputable def transform (P : Point) : Point :=
  let after_rotation := rotate_counterclockwise P ⟨2, 3⟩ (Real.pi / 2)
  reflect_y_eq_x after_rotation

-- Lean 4 statement to prove given conditions and question
theorem b_minus_a_eq_neg_six (a b : ℝ) (P : Point) (H : P = ⟨a, b⟩)
  (H_transform : transform P = ⟨-3, 1⟩) : b - a = -6 := by
  sorry

end b_minus_a_eq_neg_six_l154_154825


namespace odd_function_fixed_point_l154_154295

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_fixed_point 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f) :
  f (0) = 0 → f (-1 + 1) - 2 = -2 :=
by
  sorry

end odd_function_fixed_point_l154_154295


namespace math_proof_problem_l154_154329

noncomputable def C1 (θ : ℝ) := (real.cos θ, real.sin θ)

noncomputable def line_l (ρ θ : ℝ) : Prop := ρ * (2 * real.cos θ - real.sin θ) = 6

def cartesian_equation_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

def cartesian_equation_line (x y : ℝ) : Prop := 2 * x + y - 6 = 0

def point_P_min_distance_to_line (x y : ℝ) : Prop := 
  x = 2 * real.sqrt 5 / 5 ∧ y = real.sqrt 5 / 5

def minimum_distance (d : ℝ) : Prop :=
  d = 6 * real.sqrt 5 / 5 - 1

theorem math_proof_problem :
  (∀ (θ : ℝ), ∃ (x y : ℝ), cartesian_equation_C1 x y) ∧
  (∃ (ρ θ : ℝ), line_l ρ θ → ∃ (x y : ℝ), cartesian_equation_line x y) ∧
  (∃ (x y : ℝ), point_P_min_distance_to_line x y) ∧
  (∃ (d : ℝ), minimum_distance d) :=
  sorry

end math_proof_problem_l154_154329


namespace sin_double_angle_l154_154624

theorem sin_double_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : sin (α / 2) = √3 / 3) :
  sin α = 2 * √2 / 3 :=
by
  sorry

end sin_double_angle_l154_154624


namespace sum_of_integers_l154_154406

theorem sum_of_integers (n : ℕ) (h : n ≥ 1) : ∑ i in finset.range(n + 1), i = n * (n + 1) / 2 :=
sorry

end sum_of_integers_l154_154406


namespace positive_product_count_l154_154730

-- Definitions based on the conditions from the problem statement
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

def product_of_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  ∏ i in Finset.range n, geometric_sequence a₁ q (i + 1)

-- The problem asks us to prove there are 2 positive values among Π₈, Π₉, Π₁₀, Π₁₁
theorem positive_product_count :
  let a₁ := 512
  let q := -1 / 2
  let Π₈ := product_of_sequence a₁ q 8
  let Π₉ := product_of_sequence a₁ q 9
  let Π₁₀ := product_of_sequence a₁ q 10
  let Π₁₁ := product_of_sequence a₁ q 11
  (Π₈ > 0) + (Π₉ > 0) + (Π₁₀ > 0) + (Π₁₁ > 0) = 2 :=
sorry

end positive_product_count_l154_154730


namespace two_boys_same_price_l154_154453

open Real Finset

noncomputable def cats_20 := {r : ℝ | 12 ≤ r ∧ r ≤ 15 ∧ r ∈ {n/100 | n ∈ Icc 1200 1500}} .toFinset

noncomputable def sacks_20 := {r : ℝ | 0.10 ≤ r ∧ r ≤ 1 ∧ r ∈ {n/100 | n ∈ Icc 10 100}} .toFinset

theorem two_boys_same_price:
  ∃ (cat1 cat2 sack1 sack2 : ℝ), 
    cat1 ∈ cats_20 ∧ cat2 ∈ cats_20 ∧ sack1 ∈ sacks_20 ∧ sack2 ∈ sacks_20 ∧ 
    (cat1 + sack1 = cat2 + sack2) ∧ ¬ (cat1 = cat2 ∧ sack1 = sack2) :=
by
  sorry

end two_boys_same_price_l154_154453


namespace original_polynomial_l154_154139

theorem original_polynomial {x y : ℝ} (P : ℝ) :
  P - (-x^2 * y) = 3 * x^2 * y - 2 * x * y - 1 → P = 2 * x^2 * y - 2 * x * y - 1 :=
sorry

end original_polynomial_l154_154139


namespace calc_even_odd_prob_broken_mul_calc_higher_prob_even_with_mul_l154_154108

/-- Given a calculator with digits from 0 to 9 and two operations (addition and multiplication).
    Initially, the display shows 0. If the multiplication button is broken, prove that the
    probabilities of ending with an even number and ending with an odd number are equal. -/
theorem calc_even_odd_prob_broken_mul :
  -- Define the conditions and assertions as required.
  sorry

/-- Given a calculator with the same initial conditions but with a functional multiplication button,
    prove that the probability of ending with an even number is higher than the probability of ending
    with an odd number. -/
theorem calc_higher_prob_even_with_mul :
  -- Define the conditions and assertions as required.
  sorry

end calc_even_odd_prob_broken_mul_calc_higher_prob_even_with_mul_l154_154108


namespace tea_preparation_time_l154_154086

theorem tea_preparation_time
    (wash_kettle_time : ℕ)
    (boil_water_time : ℕ)
    (wash_cups_time : ℕ)
    (get_tea_leaves_time : ℕ)
    (brew_tea_time : ℕ) :
    wash_kettle_time = 1 →
    boil_water_time = 10 →
    wash_cups_time = 2 →
    get_tea_leaves_time = 1 →
    brew_tea_time = 1 →
    minimum_time_required wash_kettle_time boil_water_time wash_cups_time get_tea_leaves_time brew_tea_time = 11 :=
by
  sorry

noncomputable def minimum_time_required
    (wash_kettle : ℕ)
    (boil_water : ℕ)
    (wash_cups : ℕ)
    (get_tea_leaves : ℕ)
    (brew_tea : ℕ) : ℕ :=
boil_water -- inclusive of both wash_kettle, wash_cups & get_tea_leaves
  + brew_tea

end tea_preparation_time_l154_154086


namespace queenie_daily_earnings_l154_154399

/-- Define the overtime earnings per hour. -/
def overtime_pay_per_hour : ℤ := 5

/-- Define the total amount received. -/
def total_received : ℤ := 770

/-- Define the number of days worked. -/
def days_worked : ℤ := 5

/-- Define the number of overtime hours. -/
def overtime_hours : ℤ := 4

/-- State the theorem to find out Queenie's daily earnings. -/
theorem queenie_daily_earnings :
  ∃ D : ℤ, days_worked * D + overtime_hours * overtime_pay_per_hour = total_received ∧ D = 150 :=
by
  use 150
  sorry

end queenie_daily_earnings_l154_154399


namespace minimum_value_difference_l154_154670

theorem minimum_value_difference (m : ℝ) (h : m > 0) :
  ∃ a b : ℝ, (exp a = m ∧ (log (b / 2) + 1 / 2 = m) ∧ (b - a = 2 + log 2)) :=
sorry

end minimum_value_difference_l154_154670


namespace proof_problem_l154_154312

open Real

noncomputable def problem_statement (PQ QR PR QS : ℝ) (S : Set ℝ) : Prop :=
  PQ = 6 ∧ QR = 8 ∧ PR = 10 ∧ QS ∈ S ∧ QS = 6 → 
  (PS PR : ℝ) (PS_rat SR_rat : ℝ) (angle_QSR : ℝ) =>
  PS = 18 / 5 ∧ 
  SR = PR - PS ∧ 
  PS_rat = PS / SR ∧ 
  SR_rat = SR / PR ∧ 
  angle_QSR = asin (64 / 75) ∧ 
  PS_rat = 9 / 16 ∧ 
  angle_QSR = asin (64 / 75)

theorem proof_problem : problem_statement 6 8 10 6 {PS | PS < 10} :=
  by sorry

end proof_problem_l154_154312


namespace weighted_average_l154_154595

def group1_results := 40
def group1_avg := 30
def group1_weight := 2

def group2_results := 30
def group2_avg := 40
def group2_weight := 3

def group3_results := 50
def group3_avg := 20
def group3_weight := 1

theorem weighted_average :
  (group1_avg * group1_weight + group2_avg * group2_weight + group3_avg * group3_weight) / 
  (group1_weight + group2_weight + group3_weight) = 33.33 :=
by sorry

end weighted_average_l154_154595


namespace correct_proposition_l154_154660

theorem correct_proposition : 
  (∀ P : ℝ × ℝ, ∀ l : ℝ → ℝ → Prop, l = (λ x y, 2 * x - y = 1) 
    → (P = (2, 3) → l 2 3) 
    → (∀ P', (P ≠ P') 
    → (dist P' P = dist (P', P),
    ∀ A B : ℝ × ℝ, ∀ k : ℝ, k > 0 
    → ∀ P : ℝ × ℝ, (dist P A - dist P B = k) 
    → (¬((P = A) ∧ (P = B)))
    → (∃ H : P = P, dist P A + dist P B = dist P B + k),
∀ α : ℝ, ∀ P : ℝ × ℝ, dist (P 0 0) 
    = (sin α - cos α) 
    → ∀ Q : ℝ × ℝ, dist (Q 1 1) - dist (Q 0 1) = dist (Q 1 1),
∀ P : ℝ × ℝ, ∀ A B : ℝ × ℝ, A ≠ B 
    → dist P A + dist P B = 2 * dist A B 
    → dist P A + dist P B = dist (P, B) 
)

-- sorry

end correct_proposition_l154_154660


namespace basketball_lineup_ways_l154_154508

theorem basketball_lineup_ways :
  let positions := 6
  let members := 15 in
  (members * (members - 1) * (members - 2) * (members - 3) * (members - 4) * (members - 5)) = 3603600 :=
by
  let positions := 6
  let members := 15
  calc
    members * (members - 1) * (members - 2) * (members - 3) * (members - 4) * (members - 5) = 3603600 := by sorry

end basketball_lineup_ways_l154_154508


namespace savings_fraction_l154_154539

variable (P : ℝ) 
variable (S : ℝ)
variable (E : ℝ)
variable (T : ℝ)

theorem savings_fraction :
  (12 * P * S) = 2 * P * (1 - S) → S = 1 / 7 :=
by
  intro h
  sorry

end savings_fraction_l154_154539


namespace distance_P_to_AB_eq_4sqrt6_l154_154328

-- Define the polar constants
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * (sin θ)^2 = 4 * cos θ

-- Define the parametric constants
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (2 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t)

-- Define the Cartesian equation of the curve
def cartesian_curve (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the Cartesian equation of the line
def cartesian_line (x y : ℝ) : Prop :=
  x - y - 2 = 0

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

-- Define the point coordinates
def point_P : ℝ × ℝ := (2, 0)

-- Define the intersection points A and B through parameter t1 and t2
def points_A_B (t1 t2 : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (parametric_line t1, parametric_line t2)

-- The final theorem proving |PA| + |PB| = 4√6
theorem distance_P_to_AB_eq_4sqrt6 (t1 t2 : ℝ) (h1 : t1 + t2 = 4 * sqrt 2) (h2 : t1 * t2 = -16) : 
  dist point_P (parametric_line t1) + dist point_P (parametric_line t2) = 4 * sqrt 6 :=
by 
  sorry

end distance_P_to_AB_eq_4sqrt6_l154_154328


namespace susan_age_proof_l154_154117

variable (current_year : ℕ)
variable (james_age : ℕ) (janet_age : ℕ) (susan_age : ℕ)

-- Conditions
def james_future_age_condition : Prop := james_age + 15 = 37
def james_age_difference_condition : Prop := james_age - 8 = 2 * (janet_age - 8)
def susan_birth_condition : Prop := susan_age + 3 = janet_age

-- Question and Correct Answer
def susan_age_in_five_years : Prop := susan_age + 5 = 17

theorem susan_age_proof 
  (james_future_age : james_future_age_condition)
  (james_age_difference : james_age_difference_condition)
  (susan_birth : susan_birth_condition) : 
  susan_age_in_five_years := 
by
  sorry

end susan_age_proof_l154_154117


namespace trapezoid_smallest_angle_l154_154938

theorem trapezoid_smallest_angle
  (a d : ℝ)
  (h1 : a + (a + d) + (a + 2d) + (a + 3d) = 360)
  (h2 : a + 3d = 150) :
  a = 30 :=
by
  -- proof goes here
  sorry

end trapezoid_smallest_angle_l154_154938


namespace probability_heads_9_or_more_12_flips_l154_154020

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154020


namespace initial_volume_shampoo_l154_154540

theorem initial_volume_shampoo (V : ℝ) 
  (replace_rate : ℝ)
  (use_rate : ℝ)
  (t : ℝ) 
  (hot_sauce_fraction : ℝ) 
  (hot_sauce_amount : ℝ) : 
  replace_rate = 1/2 → 
  use_rate = 1 → 
  t = 4 → 
  hot_sauce_fraction = 0.25 → 
  hot_sauce_amount = t * replace_rate → 
  hot_sauce_amount = hot_sauce_fraction * V → 
  V = 8 :=
by 
  intro h_replace_rate h_use_rate h_t h_hot_sauce_fraction h_hot_sauce_amount h_hot_sauce_amount_eq
  sorry

end initial_volume_shampoo_l154_154540


namespace probability_heads_in_12_flips_l154_154000

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l154_154000


namespace prob1_l154_154115

theorem prob1 (a b : ℝ) : 
  a^2 + b^2 + 3 ≥ a * b + sqrt 3 * (a + b) :=
by sorry

end prob1_l154_154115


namespace hulk_jump_distance_l154_154809

theorem hulk_jump_distance :
  ∃ n : ℕ, 3^n > 1500 ∧ ∀ m < n, 3^m ≤ 1500 := 
sorry

end hulk_jump_distance_l154_154809


namespace arc_length_of_curve_l154_154924

noncomputable def arc_length_polar (ρ : ℝ → ℝ) (φ₀ φ₁ : ℝ) : ℝ :=
  ∫ θ in φ₀..φ₁, Real.sqrt (ρ θ ^ 2 + (deriv ρ θ) ^ 2)

def curve_eqn (θ : ℝ) : ℝ := 8 * Real.cos θ

theorem arc_length_of_curve : 
  arc_length_polar curve_eqn 0 (Real.pi / 4) = 2 * Real.pi :=
by
  sorry

end arc_length_of_curve_l154_154924


namespace max_value_sin_cos_sum_l154_154250

theorem max_value_sin_cos_sum (n : ℕ) (x : Fin n → ℝ) :
  let A := (Finset.univ.sum (fun i => Real.sin (x i))) *
           (Finset.univ.sum (fun i => Real.cos (x i)))
  in A ≤ n ^ 2 / 2 :=
by
  sorry

end max_value_sin_cos_sum_l154_154250


namespace probability_adjacent_white_or_blue_l154_154858

def totalArrangements : ℕ := 7! / (3! * 2! * 2!)

def adjWhite : ℕ := 6! / (3! * 1! * 2!)

def adjBlue : ℕ := 6! / (3! * 2! * 1!)

def adjBoth : ℕ := 5! / (3! * 1! * 1!)

def favorableArrangements : ℕ := adjWhite + adjBlue - adjBoth

def probability : ℚ := favorableArrangements / totalArrangements

theorem probability_adjacent_white_or_blue :
  probability = 10 / 21 := by
  sorry

end probability_adjacent_white_or_blue_l154_154858


namespace sum_of_x₁_x₃_x₅_minimal_degree_of_G_l154_154138

variables {x₁ x₂ x₃ x₄ x₅ : ℝ}

-- Given conditions
def polynomial_G_with_real_coefficients (G : ℝ → ℝ) : Prop :=
  (G x₁ = 2022) ∧ (G x₂ = 2022) ∧ (G x₃ = 2022) ∧ (G x₄ = 2022) ∧ (G x₅ = 2022)
  ∧ (x₁ < x₂) ∧ (x₂ < x₃) ∧ (x₃ < x₄) ∧ (x₄ < x₅)
  ∧ (∀ x, G (-6 + x) = G (-6 - x)) -- symmetry with respect to x = -6

-- Part (a): Find x₁ + x₃ + x₅
theorem sum_of_x₁_x₃_x₅ (G : ℝ → ℝ) (hG : polynomial_G_with_real_coefficients G) :
  x₁ + x₃ + x₅ = -18 :=
sorry

-- Part (b): Determine the minimal degree of G(x)
theorem minimal_degree_of_G (G : ℝ → ℝ) (hG : polynomial_G_with_real_coefficients G) :
  ∃ n : ℕ, polynomial.degree G = (n : ℕ) ∧ n = 6 :=
sorry

end sum_of_x₁_x₃_x₅_minimal_degree_of_G_l154_154138


namespace problem_l154_154694

theorem problem (x y : ℝ) (h : (3 * x - y + 5)^2 + |2 * x - y + 3| = 0) : x + y = -3 := 
by
  sorry

end problem_l154_154694


namespace domain_length_l154_154612

noncomputable theory

def f (x : ℝ) := log (1/3) (log 8 (log (1/8) (log 8 (log (1/8) x))))

theorem domain_length : ∃ (m n : ℤ), Nat.coprime m n ∧ m + n = 20988567 := 
sorry

end domain_length_l154_154612


namespace average_difference_l154_154104

theorem average_difference :
  let avg1 := (200 + 400) / 2
  let avg2 := (100 + 200) / 2
  avg1 - avg2 = 150 :=
by
  sorry

end average_difference_l154_154104


namespace productSumDivisibleBy2011_l154_154396

open Nat

-- Define the odd product sequence
def oddProduct : ℕ :=
  ∏ i in (range 1005).map (λ n => 2 * n + 1), i 

-- Define the even product sequence
def evenProduct : ℕ :=
  ∏ i in (range 1005).map (λ n => 2 * (n + 1)), i

-- The theorem stating the desired divisibility result
theorem productSumDivisibleBy2011 :
  (oddProduct + evenProduct) % 2011 = 0 :=
by
  sorry

end productSumDivisibleBy2011_l154_154396


namespace locus_of_equally_tall_flagpoles_l154_154196
  
  /-- Define two points and their heights --/
  structure Flagpole :=
  (height : ℝ)
  (position : ℝ × ℝ)
  
  /-- Define the locus problem --/
  theorem locus_of_equally_tall_flagpoles
    (h k a : ℝ)
    (H K : ℝ × ℝ)
    (H_flagpole : Flagpole := {height := h, position := H})
    (K_flagpole : Flagpole := {height := k, position := K})
    (distance_HK : ∥H - K∥ = 2 * a) :
    ∃ (A B : ℝ × ℝ), A ≠ B ∧ ∀ P : ℝ × ℝ, 
    (∥P - H∥ / ∥P - K∥ = h / k) ↔ 
    ∃ C : ℝ × ℝ, C = {val := (A, B)} ∧ ∥P - C.1∥ + ∥P - C.2∥ = ∥A - B∥ :=
  sorry
  
end locus_of_equally_tall_flagpoles_l154_154196


namespace liam_remaining_money_l154_154787

noncomputable def octal_to_decimal (n : Nat) : Nat :=
  let digits := n.digits 8
  digits.foldl (λ acc d, d + 8 * acc) 0

noncomputable def start_amount := octal_to_decimal 5376
noncomputable def ticket_cost := 1200
noncomputable def remaining_amount := 1614

theorem liam_remaining_money :
  start_amount - ticket_cost = remaining_amount :=
by
  sorry

end liam_remaining_money_l154_154787


namespace smallest_integer_with_16_divisors_l154_154064

-- Define the condition for the number of divisors of an integer
def num_divisors (n : ℕ) : ℕ :=
  (n.factorization.map (λ p a, a + 1)).prod

-- Define the problem statement as a theorem
theorem smallest_integer_with_16_divisors : ∃ n : ℕ, num_divisors n = 16 ∧ (∀ m : ℕ, num_divisors m = 16 → n ≤ m) :=
by
  -- Placeholder to skip proof
  sorry

end smallest_integer_with_16_divisors_l154_154064


namespace correct_result_l154_154467

theorem correct_result (incorrect_result: ℕ) (ones_place_wrong: ℕ) (tens_place_wrong: ℕ) (ones_place_correct: ℕ) (tens_place_correct: ℕ): incorrect_result - (ones_place_wrong - ones_place_correct) + (tens_place_correct * 10 - tens_place_wrong * 10) = 422 := 
by {let incorrect_result := 387, 
    let ones_place_wrong := 8, 
    let tens_place_wrong := 5, 
    let ones_place_correct := 3, 
    let tens_place_correct := 9, 
    sorry}

end correct_result_l154_154467


namespace sum_infinite_series_eq_l154_154619

theorem sum_infinite_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 999 : ℝ) ^ n = 1000 / 998 := by
sorry

end sum_infinite_series_eq_l154_154619


namespace math_problem_l154_154256

-- Define functions f and g
def f (x : ℝ) : ℝ := x^2 - 3 * x + 6
def g (x : ℝ) : ℝ := 2 * x + 4

-- State the theorem
theorem math_problem : f(g(3)) - g(f(3)) = 60 := by
  sorry

end math_problem_l154_154256


namespace interior_angle_sum_of_regular_polygon_l154_154820

theorem interior_angle_sum_of_regular_polygon (h: ∀ θ, θ = 45) :
  ∃ s, s = 1080 := by
  sorry

end interior_angle_sum_of_regular_polygon_l154_154820


namespace russel_carousel_rides_l154_154403

variable (tickets_used : Nat) (tickets_shooting : Nat) (tickets_carousel : Nat)
variable (total_tickets : Nat)
variable (times_shooting : Nat)

theorem russel_carousel_rides :
    times_shooting = 2 →
    tickets_shooting = 5 →
    tickets_carousel = 3 →
    total_tickets = 19 →
    tickets_used = total_tickets - (times_shooting * tickets_shooting) →
    tickets_used / tickets_carousel = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end russel_carousel_rides_l154_154403


namespace coin_flip_probability_l154_154012

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154012


namespace union_complement_eq_l154_154780

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l154_154780


namespace three_lines_intersect_single_point_l154_154423

theorem three_lines_intersect_single_point (a : ℝ) :
  (∀ x y : ℝ, (x + 2*y + a) * (x^2 - y^2) = 0) ↔ a = 0 := by
  sorry

end three_lines_intersect_single_point_l154_154423


namespace probability_heads_9_or_more_12_flips_l154_154023

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154023


namespace max_value_of_f_on_interval_l154_154819

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Icc (-1 : ℝ) 1 ∧ (∀ (x : ℝ), x ∈ Icc (-1 : ℝ) 1 → f x ≤ f c) ∧ f c = 2 :=
sorry

end max_value_of_f_on_interval_l154_154819


namespace copper_atoms_in_compound_l154_154125

noncomputable def numCopperAtoms (C_atoms : ℕ) (O_atoms : ℕ) (compound_weight : ℕ) (Cu_weight : ℝ) (C_weight : ℝ) (O_weight : ℝ) : ℕ :=
  let weight_C : ℝ := C_atoms * C_weight
  let weight_O : ℝ := O_atoms * O_weight
  let total_weight_non_Cu : ℝ := weight_C + weight_O
  let weight_Cu : ℝ := compound_weight - total_weight_non_Cu
  let num_Cu_atoms : ℝ := weight_Cu / Cu_weight
  num_Cu_atoms.round

theorem copper_atoms_in_compound :
  numCopperAtoms 1 3 124 63.55 12.01 16.00 = 1 :=
by
  sorry

end copper_atoms_in_compound_l154_154125


namespace symmetric_point_y_axis_l154_154243

open Point
open Geometry

-- Definition: Point A is given in the problem
def A : ℝ × ℝ := (-2, 4)

-- Definition: Symmetric coordinates of point A over the y-axis need to be determined
def symmetric_to_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem symmetric_point_y_axis :
  symmetric_to_y_axis A = (2, 4) :=
by
  sorry

end symmetric_point_y_axis_l154_154243


namespace jogger_distance_ahead_l154_154130

theorem jogger_distance_ahead
  (speed_jogger_kmh : ℕ)
  (speed_train_kmh : ℕ)
  (train_length_m : ℕ)
  (time_to_pass_jogger_s : ℕ)
  (conversion_factor : ℚ)
  (relative_speed_mps : ℚ)
  (distance_covered_m : ℚ)
  : speed_jogger_kmh = 9 →
    speed_train_kmh = 45 →
    train_length_m = 120 →
    time_to_pass_jogger_s = 36 →
    conversion_factor = (5 / 18 : ℚ) →
    relative_speed_mps = (speed_train_kmh - speed_jogger_kmh) * conversion_factor →
    distance_covered_m = relative_speed_mps * time_to_pass_jogger_s →
    ∃ D : ℚ, distance_covered_m = train_length_m + D ∧ D = 240 :=
by
  intros
  have h_speed_jogger_kmh : speed_jogger_kmh = 9 := by assumption
  have h_speed_train_kmh : speed_train_kmh = 45 := by assumption
  have h_train_length_m : train_length_m = 120 := by assumption
  have h_time_to_pass_jogger_s : time_to_pass_jogger_s = 36 := by assumption
  have h_conversion_factor : conversion_factor = (5 / 18 : ℚ) := by assumption
  have h_relative_speed_mps : relative_speed_mps = (speed_train_kmh - speed_jogger_kmh) * conversion_factor := by assumption
  have h_distance_covered_m : distance_covered_m = relative_speed_mps * time_to_pass_jogger_s := by assumption
  use (distance_covered_m - train_length_m)
  split
  · simp [h_distance_covered_m, h_train_length_m]
  · sorry

end jogger_distance_ahead_l154_154130


namespace elmo_clone_wins_l154_154186

theorem elmo_clone_wins (n : ℕ) (h : n ≥ 3) : "Elmo's clone wins" :=
by
  sorry

end elmo_clone_wins_l154_154186


namespace union_complement_eq_l154_154776

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l154_154776


namespace bananas_left_in_jar_l154_154599

noncomputable def initial_bananas : ℕ := 50
noncomputable def removed_percentage : ℝ := 0.20
noncomputable def sam_adds : ℕ := 15

theorem bananas_left_in_jar (initial_bananas : ℕ) (removed_percentage : ℝ) (sam_adds : ℕ) : ℕ :=
  let removed_bananas := removed_percentage * initial_bananas
  let remaining_bananas := initial_bananas - removed_bananas.toNat
  remaining_bananas + sam_adds = 55 :=
begin
  sorry
end

end bananas_left_in_jar_l154_154599


namespace locus_of_lines_is_cone_l154_154043

theorem locus_of_lines_is_cone
  (A B C : Type) [EuclideanSpace A] [EuclideanSpace B]
  (K : A) (hK : midpoint A B K)
  (h_triangle : is_triangle A B C) :
  locus_of_lines_through_C_equidistant_from_A_B A B C = cone_with_apex C :=
sorry

end locus_of_lines_is_cone_l154_154043


namespace smallest_integer_with_16_divisors_l154_154075

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (number_of_divisors m = 16) → (m ≥ n)) ∧ (n = 120) :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154075


namespace total_animals_for_sale_l154_154896

theorem total_animals_for_sale (dogs cats birds fish : ℕ) 
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) :
  dogs + cats + birds + fish = 39 := 
by
  sorry

end total_animals_for_sale_l154_154896


namespace probability_heads_at_least_9_of_12_flips_l154_154008

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154008


namespace num_triangles_l154_154456

-- Definitions based on provided conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def lengths : List ℕ := [2, 3, 4, 6]

-- Main theorem to prove
theorem num_triangles : (lengths.combinations 3).count (λ t, match t with | [a, b, c] => is_triangle a b c | _ => false end) = 2 :=
by 
  sorry

end num_triangles_l154_154456


namespace total_animals_for_sale_l154_154897

theorem total_animals_for_sale (dogs cats birds fish : ℕ) 
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) :
  dogs + cats + birds + fish = 39 := 
by
  sorry

end total_animals_for_sale_l154_154897


namespace brad_trips_to_fill_barrel_l154_154165

-- Lean statement for the problem
theorem brad_trips_to_fill_barrel :
  let r := 8
  let h := 20
  let r_b := 8
  let loss := 0.1
  let bucket_volume := (2 / 3) * Real.pi * r_b^3
  let effective_trip_volume := (1 - loss) * bucket_volume
  let barrel_volume := Real.pi * r^2 * h
  let trips_needed := (barrel_volume / effective_trip_volume).ceil
  trips_needed = 5 :=
by
  sorry

end brad_trips_to_fill_barrel_l154_154165


namespace sufficient_but_not_necessary_condition_l154_154255

theorem sufficient_but_not_necessary_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (ab > 1) → (a + b > 2) ?
sorry

end sufficient_but_not_necessary_condition_l154_154255


namespace product_of_N_values_l154_154565

theorem product_of_N_values 
  (N : ℝ) 
  (M L M_5 L_5 : ℝ)
  (h1 : M = L + N)
  (h2 : M_5 = M - 7)
  (h3 : L_5 = L + 5)
  (h4 : |M_5 - L_5| = 4) :
  N ∈ {8, 16} ∧ {8, 16}.prod = 128 :=
by sorry

end product_of_N_values_l154_154565


namespace mike_pumpkins_l154_154801

def pumpkins : ℕ :=
  let sandy_pumpkins := 51
  let total_pumpkins := 74
  total_pumpkins - sandy_pumpkins

theorem mike_pumpkins : pumpkins = 23 :=
by
  sorry

end mike_pumpkins_l154_154801


namespace hypercube_diagonals_not_perpendicular_l154_154339

theorem hypercube_diagonals_not_perpendicular {n : ℕ} (h : n ≥ 3) :
  ¬ (∀ d1 d2 : ℝ^n, is_diagonal d1 ∧ is_diagonal d2 → perpendicular d1 d2 ∧ bisects_each_other d1 d2) :=
sorry

end hypercube_diagonals_not_perpendicular_l154_154339


namespace area_of_bounded_region_l154_154960

-- Define the equations for the ellipse and the piecewise line
def ellipse (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1
def abs_line (x y : ℝ) : Prop := y = |x| + 1

-- Define a proof that the area of the region bounded by the above two curves is 2 * arccos 0.6
theorem area_of_bounded_region :
  let region_area := 2 * Real.arccos 0.6 in
  ∃ x y, ellipse x y ∧ abs_line x y → (region_area = 2 * Real.arccos 0.6) :=
by
  sorry

end area_of_bounded_region_l154_154960


namespace smallest_BC_l154_154593

theorem smallest_BC (AB AC AD BD BC CD : ℕ) 
  (h1 : AB = AC) 
  (h2 : AD^2 = 72)
  (h3 : BC ≥ CD)
  (h4 : ∃ (D: Point), D ∈ segment BC ∧ (AD ⊥ BC)) : 
  ∃ (BCmin : ℕ), BCmin = 11 := 
sorry

end smallest_BC_l154_154593


namespace min_value_dot_product_l154_154336

variables {A B C M N : Type} [EuclideanSpace ℝ A]
variable [AffineSpace A ℝ]
variable [AddGroup A] 
variables {AM AN : ℝ} [InnerProductSpace ℝ A]

-- Definitions:
variable (M_midpoint : midpoint (B + C))
variable (N_midpoint : midpoint (B + M))
variable (angle_A_eq_pi_over_3 : inner (normalize (outdir AB A)) (normalize (outdir AC A)) = 1/2)
variable (area_ABC_sqrt3 : area ABC = √3)

-- Proof statement:
theorem min_value_dot_product :
  ∃ AM AN : ℝ, innerProduct AM AN = √3 + 1 :=
sorry

end min_value_dot_product_l154_154336


namespace distance_between_skew_lines_l154_154722
noncomputable theory
open Real

-- Define the points in the cuboid
def A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def A₁ := (0 : ℝ, 0 : ℝ, 1 : ℝ)
def B := (1 : ℝ, 0 : ℝ, 0 : ℝ)
def B₁ := (1 : ℝ, 0 : ℝ, 1 : ℝ)
def D := (0 : ℝ, 2 : ℝ, 0 : ℝ)
def D₁ := (0 : ℝ, 2 : ℝ, 1 : ℝ)

-- Function to calculate Euclidean distance between two points
def dist (p q : ℝ × ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

-- Proving the distance between skew lines A₁D and B₁D₁ is 2/3
theorem distance_between_skew_lines : dist_geo_lines (line_through_points A₁ D) (line_through_points B₁ D₁) = 2 / 3 :=
sorry

end distance_between_skew_lines_l154_154722


namespace find_constants_u_v_l154_154963

theorem find_constants_u_v : 
  ∃ u v : ℝ, (∀ x : ℝ, 9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 :=
sorry

end find_constants_u_v_l154_154963


namespace cubic_three_real_roots_l154_154816

theorem cubic_three_real_roots (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧
   x₁ ^ 3 - 3 * x₁ - a = 0 ∧
   x₂ ^ 3 - 3 * x₂ - a = 0 ∧
   x₃ ^ 3 - 3 * x₃ - a = 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end cubic_three_real_roots_l154_154816


namespace inequality_problem_l154_154759

variable {a b : ℕ}

theorem inequality_problem (a : ℕ) (b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq_1_a : a ≠ 1) (h_neq_1_b : b ≠ 1) :
  ((a^5 - 1:ℚ) / (a^4 - 1)) * ((b^5 - 1) / (b^4 - 1)) > (25 / 64 : ℚ) * (a + 1) * (b + 1) :=
by
  sorry

end inequality_problem_l154_154759


namespace range_of_solutions_l154_154567

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l154_154567


namespace smallest_integer_with_16_divisors_l154_154081

theorem smallest_integer_with_16_divisors : 
  ∃ (n : ℕ), (∃ (p_1 p_2 p_3 : ℕ) (a_1 a_2 a_3 : ℕ), 
  (p_1 = 2 ∧ p_2 = 3 ∧ a_1 = 3 ∧ a_2 = 3 ∧ n = p_1 ^ a_1 * p_2 ^ a_2) ∧
  (∀ m, m > 0 → (∃ b1 b2 ..., m has exactly 16 positive divisors) → 216 ≤ m)) := 
sorry

end smallest_integer_with_16_divisors_l154_154081


namespace perfect_square_factors_count_l154_154289

theorem perfect_square_factors_count :
  (∃ (count : ℕ), count = 560 ∧ ∀ (pf : ℕ), 
    (pf = 2^12 * 3^15 * 7^18) →
    ( ∃ (f : ℕ → ℕ → ℕ → ℕ), 
      f(7)(8)(10) = count)) :=
begin
  -- Initialize the given product
  let product := 2^12 * 3^15 * 7^18,
  -- Define the function that counts perfect square factors
  let count_perfect_squares := λ (a b c : ℕ), (a * b * c),
  -- Define the conditions for a, b, and c
  have h2 : ∀ (n : ℕ), ∃ (a : ℕ), a = 7 → n = count_perfect_squares 7 8 10,
      from λ n, ⟨7, rfl⟩,
  have h3 : ∀ (n : ℕ), ∃ (b : ℕ), b = 8 → n = count_perfect_squares 7 8 10,
      from λ n, ⟨8, rfl⟩,
  have h7 : ∀ (n : ℕ), ∃ (c : ℕ), c = 10 → n = count_perfect_squares 7 8 10,
      from λ n, ⟨10, rfl⟩,
  -- Sum up and conclude
  use 560,
  split,
  { refl },
  { intros pf hpf,
    use count_perfect_squares,
    exact rfl },
end

end perfect_square_factors_count_l154_154289


namespace tangent_lines_center_range_l154_154327

/-- Problem (1) -/
/-- If the center C (3, 2) lies on line l: y = 2x - 4 and y = x - 1, 
    then prove the equation of the tangent lines passing through A (0, 3) should be y = 3 or 3x + 4y - 12 = 0 -/
theorem tangent_lines (A : ℝ × ℝ) (C : ℝ × ℝ) (l1 l2 : ℝ → ℝ) :
  A = (0, 3) →
  C = (3, 2) →
  l1 = λ x, 2 * x - 4 →
  l2 = λ x, x - 1 →
  (C.1 - 3) * (C.1 - 3) + (C.2 - 2) * (C.2 - 2) = 1 →
  (A.1 = 0 → A.2 = 3) →
  (A.2 = 3 → ∃ k : ℝ, k = 0 ∨ k = -3/4) →
  ∀ k : ℝ, k = 0 → (A.2 = k * A.1 + 3 ∨ 3 * A.1 + 4 * A.2 - 12 = 0) :=
sorry

/-- Problem (2) -/
/-- If there exists a point M on circle C such that |MA| = 2|MO|,
    and center of circle C is on line l: y = 2x - 4,
    find the range of values for the x-coordinate a of the center C
    to be within [0, 12/5] -/
theorem center_range (A M O : ℝ × ℝ) (C : ℝ) (l : ℝ → ℝ) :
  A = (0, 3) →
  O = (0, 0) →
  M ∈ (λ x : ℝ, (x + 0)^2 + (x - (-1))^2 = 4) →
  l = λ x, 2 * x - 4 →
  ∃ a : ℝ, 0 ≤ a ∧ a ≤ 12 / 5 →
  ∀ a : ℝ, (a, 2 * a - 4) = (a, l a) → (|MA| = 2 * |MO|) :=
sorry

end tangent_lines_center_range_l154_154327


namespace pecan_weight_is_correct_l154_154290

noncomputable def pecan_weight_of_mixture :=
  ∃ (C : ℝ) (p : ℝ),
    let total_cost_pecans := p * 5.60 in
    let total_cost_cashews := 2 * C in
    let total_weight := p + 2 in
    let total_cost_mixture := total_weight * 4.34 in
    total_cost_pecans + total_cost_cashews = total_cost_mixture ∧
    total_weight = 3.33 ∧
    p = 1.33

theorem pecan_weight_is_correct : pecan_weight_of_mixture :=
begin
  sorry
end

end pecan_weight_is_correct_l154_154290


namespace statement_A_statement_B_statement_C_statement_D_l154_154482

-- Definition of vectors a, b, c and their properties
variables {ℝ : Type*} [normed_add_comm_group ℝ] [inner_product_space ℝ ℝ]
variables (a b c : ℝ)
variables (a_nonzero b_nonzero : a ≠ 0 ∧ b ≠ 0)

-- Statements to be proven
theorem statement_A : inner (a + b) c = inner a c + inner b c := sorry

theorem statement_B : (norm a < norm b ∧ ∀ k, a = k • b) → false := sorry

theorem statement_C : (norm (a + b) = norm (a - b)) → inner a b = 0 := sorry

theorem statement_D : let a := (2 : ℝ, real.sqrt 3) in
    let b := (1 : ℝ, real.sqrt 3) in 
    ∥a + proj a b∥ = norm (a - proj a b) := sorry

end statement_A_statement_B_statement_C_statement_D_l154_154482


namespace scientific_notation_of_trillion_l154_154418

theorem scientific_notation_of_trillion (h_trillion : 1.5 * 10^12 = 1.5e12) : 
  1.5 * 10^12 = 1.5 * 10^12 := 
begin
  sorry
end

end scientific_notation_of_trillion_l154_154418


namespace rope_cut_probability_l154_154532

/--
A thin rope has a length of 1 meter. If the rope is randomly cut from the middle, 
the probability that both resulting pieces will have a length greater than 1/8 meter is 3/4.
-/
theorem rope_cut_probability :
  (∃ (cut_point : ℝ) (h : 0 ≤ cut_point ∧ cut_point ≤ 1), 
    (1/8 < cut_point ∧ cut_point < 7/8) → 
    (probability_of_cut : ℝ) 
  ) ∧
    (probability_of_cut = 3/4) :=
sorry

end rope_cut_probability_l154_154532


namespace rational_segments_l154_154562

-- Definitions for AD, BD, and CD being rational
variables (x y z : ℚ)

-- Semicircle properties and segment definitions
def AB : ℚ := (x^2 + y^2).sqrt
def OC : ℚ := AB / 2
def OD : ℚ := (OC - z / 2)

-- Propositions to prove
theorem rational_segments :
  ∀ (x y z : ℚ), 
  (AD : ℚ) = x →
  (BD : ℚ) = y → 
  (CD : ℚ) = z →
  rational (AD) → 
  rational (BD) → 
  rational (CD) →
  rational (OD * (real.cos (real.pi / 2))) ∧ -- rational (OE)
  rational ((OD * (real.cos (real.pi / 2)))^2 + z^2).sqrt ∧ -- rational (DE)
  ∀ (segment : ℚ), segment ∈ {AD, BD, CD, AB, OC, OD, OD * (real.cos (real.pi / 2)), ((OD * (real.cos (real.pi / 2)))^2 + z^2).sqrt } → 
  rational segment :=
by
  sorry

end rational_segments_l154_154562


namespace positive_integer_cases_l154_154219

theorem positive_integer_cases (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℤ, (abs (x^2 - abs x)) / x = n ∧ n > 0) ↔ (∃ m : ℤ, (x = m) ∧ (m > 1 ∨ m < -1)) :=
by
  sorry

end positive_integer_cases_l154_154219


namespace arithmetic_sequence_solution_l154_154368

theorem arithmetic_sequence_solution :
  ∀ (n : ℕ), 
  let a : ℕ → ℤ := λ n, 2 * (n : ℤ) - 9,
  S_n : ℕ → ℤ := λ n, n * n - 8 * n
  in 
  a 1 = -7 ∧ 
  S_n 3 = -15 ∧ 
  S_n n = (n - 4) * (n - 4) - 16 ∧ 
  (∃ n_min : ℕ, S_n n_min = -16) := 
by
  sorry

end arithmetic_sequence_solution_l154_154368


namespace triangle_is_right_l154_154335

-- Definitions based on the conditions given in the problem
variables {a b c A B C : ℝ}

-- Introduction of the conditions in Lean
def is_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)

def given_condition (A b c : ℝ) : Prop :=
  (Real.cos (A / 2))^2 = (b + c) / (2 * c)

-- Theorem statement to prove the conclusion based on given conditions
theorem triangle_is_right (a b c A B C : ℝ) 
  (h_triangle : is_triangle a b c A B C)
  (h_given : given_condition A b c) :
  A = 90 := sorry

end triangle_is_right_l154_154335


namespace one_thirds_in_nine_halves_l154_154684

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l154_154684


namespace quadratic_transform_l154_154822

theorem quadratic_transform : ∀ (x : ℝ), x^2 = 3 * x + 1 ↔ x^2 - 3 * x - 1 = 0 :=
by
  sorry

end quadratic_transform_l154_154822


namespace circumscribed_circle_radii_equal_circumscribed_circle_radii_different_for_non_acute_specific_case_non_acute_and_isosceles_analyze_triangle_radii_l154_154437

variables {A B C H D E : Type} [geometry A B C H]
variables (acute_ABC : is_acute_triangle A B C)
variables (C_ge_90 : angle A C B ≥ 90)
variables (AC_gt_BC : AC > BC)
variables (D_on_AC : point_on_line D A C ∧ AD = BD)
variables (E_on_AB : point_on_line E A B ∧ angle A E D = angle A C B)

theorem circumscribed_circle_radii_equal :
  circumscribed_circle_radius A B H = circumscribed_circle_radius B C H ∧
  circumscribed_circle_radius B C H = circumscribed_circle_radius C A H :=
sorry

theorem circumscribed_circle_radii_different_for_non_acute :
  C_ge_90 → AC_gt_BC →
  ¬is_isosceles_triangle A B C →
  circumscribed_circle_radius A B H ≠ circumscribed_circle_radius B C H :=
sorry

theorem specific_case_non_acute_and_isosceles :
  C_ge_90 → AC = BC →
  circumscribed_circle_radius B C D > circumscribed_circle_radius C A D :=
sorry

theorem analyze_triangle_radii :
  acute_ABC → circumscribed_circle_radii_equal ∨
  C_ge_90 → AC_gt_BC → circumscribed_circle_radii_different_for_non_acute ∨
  C_ge_90 → AC = BC → specific_case_non_acute_and_isosceles →
  (∀ T, is_triangle T → circumscribed_circle_radii_equal ∨ specific_case_non_acute_and_isosceles) :=
sorry

end circumscribed_circle_radii_equal_circumscribed_circle_radii_different_for_non_acute_specific_case_non_acute_and_isosceles_analyze_triangle_radii_l154_154437


namespace total_blocks_used_l154_154400

theorem total_blocks_used (blocks_tower : ℕ) (blocks_house : ℕ) (total_blocks : ℕ) :
  blocks_tower = 27 → blocks_house = 53 → total_blocks = 27 + 53 → total_blocks = 80 :=
by
  intros h1 h2 h3
  rw [←h3, h1, h2]
  rfl

end total_blocks_used_l154_154400


namespace tenth_term_in_arithmetic_sequence_l154_154860

theorem tenth_term_in_arithmetic_sequence :
  let a := (1:ℚ) / 2
  let d := (1:ℚ) / 3
  a + 9 * d = 7 / 2 :=
by
  sorry

end tenth_term_in_arithmetic_sequence_l154_154860


namespace shop_width_correct_l154_154435

-- Definition of the shop's monthly rent
def monthly_rent : ℝ := 2400

-- Definition of the shop's length in feet
def shop_length : ℝ := 10

-- Definition of the annual rent per square foot
def annual_rent_per_sq_ft : ℝ := 360

-- The mathematical assertion that the width of the shop is 8 feet
theorem shop_width_correct (width : ℝ) :
  (monthly_rent * 12) / annual_rent_per_sq_ft / shop_length = width :=
by
  sorry

end shop_width_correct_l154_154435


namespace number_of_girls_in_school_l154_154425

theorem number_of_girls_in_school :
  ∃ G B : ℕ, 
    G + B = 1600 ∧
    (G * 200 / 1600) - 20 = (B * 200 / 1600) ∧
    G = 860 :=
by
  sorry

end number_of_girls_in_school_l154_154425


namespace ratio_of_red_to_total_l154_154677

def hanna_erasers : Nat := 4
def tanya_total_erasers : Nat := 20

def rachel_erasers (hanna_erasers : Nat) : Nat :=
  hanna_erasers / 2

def tanya_red_erasers (rachel_erasers : Nat) : Nat :=
  2 * (rachel_erasers + 3)

theorem ratio_of_red_to_total (hanna_erasers tanya_total_erasers : Nat)
  (hanna_has_4 : hanna_erasers = 4) 
  (tanya_total_is_20 : tanya_total_erasers = 20) 
  (twice_as_many : hanna_erasers = 2 * (rachel_erasers hanna_erasers)) 
  (three_less_than_half : rachel_erasers hanna_erasers = (1 / 2:Rat) * (tanya_red_erasers (rachel_erasers hanna_erasers)) - 3) :
  (tanya_red_erasers (rachel_erasers hanna_erasers)) / tanya_total_erasers = 1 / 2 := by
  sorry

end ratio_of_red_to_total_l154_154677


namespace second_chapter_pages_is_80_l154_154120

def first_chapter_pages : ℕ := 37
def second_chapter_pages : ℕ := first_chapter_pages + 43

theorem second_chapter_pages_is_80 : second_chapter_pages = 80 :=
by
  sorry

end second_chapter_pages_is_80_l154_154120


namespace five_coins_all_heads_or_tails_l154_154211

theorem five_coins_all_heads_or_tails : 
  (1 / 2) ^ 5 + (1 / 2) ^ 5 = 1 / 16 := 
by 
  sorry

end five_coins_all_heads_or_tails_l154_154211


namespace sequence_general_term_and_product_l154_154238

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1
def b_n (n : ℕ) : ℝ := 4 / (a_n n ^ 2 - 1)

def T_n (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), b_n k

theorem sequence_general_term_and_product :
  (∀ n, a_n n = 2 * n + 1) ∧ (T_1 T_2 … T_10 = 1 / 11) :=
by
  sorry

end sequence_general_term_and_product_l154_154238


namespace bullet_trains_crossing_time_l154_154463

theorem bullet_trains_crossing_time
  (length_train1 : ℝ) (length_train2 : ℝ)
  (speed_train1_km_hr : ℝ) (speed_train2_km_hr : ℝ)
  (opposite_directions : Prop)
  (h_length1 : length_train1 = 140)
  (h_length2 : length_train2 = 170)
  (h_speed1 : speed_train1_km_hr = 60)
  (h_speed2 : speed_train2_km_hr = 40)
  (h_opposite : opposite_directions = true) :
  ∃ t : ℝ, t = 11.16 :=
by
  sorry

end bullet_trains_crossing_time_l154_154463


namespace coin_flip_probability_l154_154014

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154014


namespace even_function_extension_l154_154259

variable {ℝ : Type*} [LinearOrderedField ℝ]

-- Given that f(x) is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

-- Given function f(x) = x^3 + x + 1 for x > 0 and extending to other values by evenness
def f (x : ℝ) : ℝ :=
if x > 0 then x^3 + x + 1 else if x < 0 then (-x)^3 + (-x) + 1 else 1

noncomputable def f_extension (x : ℝ) : ℝ :=
-x^3 - x + 1

theorem even_function_extension (x: ℝ) (h_even : is_even_function f) (h_positive : f x = x^3 + x + 1) :
  f x = f_extension x :=
by
  sorry

end even_function_extension_l154_154259


namespace probability_transformed_z_in_S_l154_154519

noncomputable def S : set ℂ := 
  {z : ℂ | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

noncomputable def T : set ℂ := 
  {z : ℂ | |(z.re - z.im)| ≤ 2 ∧ |(z.re + z.im)| ≤ 2}

theorem probability_transformed_z_in_S :
  (∀ z ∈ S, (1/2 + 1/2 * complex.I) * z ∈ S) ↔ 1 := 
sorry

end probability_transformed_z_in_S_l154_154519


namespace evaluate_expression_at_2_l154_154409

theorem evaluate_expression_at_2 :
  (let x := 2 in (x + 2) * (x - 3) - x * (2 * x - 1)) = -10 := by
  sorry

end evaluate_expression_at_2_l154_154409


namespace coin_flip_probability_l154_154013

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154013


namespace intersection_M_N_l154_154363

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l154_154363


namespace real_solutions_l154_154609

-- Given the condition (equation)
def quadratic_equation (x y : ℝ) : Prop :=
  x^2 + 2 * x * Real.sin (x * y) + 1 = 0

-- The main theorem statement proving the solutions for x and y
theorem real_solutions (x y : ℝ) (k : ℤ) :
  quadratic_equation x y ↔
  (x = 1 ∧ (y = (Real.pi / 2 + 2 * k * Real.pi) ∨ y = (-Real.pi / 2 + 2 * k * Real.pi))) ∨
  (x = -1 ∧ (y = (-Real.pi / 2 + 2 * k * Real.pi) ∨ y = (Real.pi / 2 + 2 * k * Real.pi))) :=
by
  sorry

end real_solutions_l154_154609


namespace largest_vertex_sum_of_parabola_l154_154215

theorem largest_vertex_sum_of_parabola 
  (a T : ℤ)
  (hT : T ≠ 0)
  (h1 : 0 = a * 0^2 + b * 0 + c)
  (h2 : 0 = a * (2 * T) ^ 2 + b * (2 * T) + c)
  (h3 : 36 = a * (2 * T + 2) ^ 2 + b * (2 * T + 2) + c) :
  ∃ N : ℚ, N = -5 / 4 :=
sorry

end largest_vertex_sum_of_parabola_l154_154215


namespace smallest_number_of_students_l154_154790

theorem smallest_number_of_students :
  ∃ n : ℕ,
  (∃ (k : ℕ), n = 18 * k) ∧  -- condition for June 1
  (∀ (d : ℕ), d ∣ n → d ≠ 18 → d ≠ 1 → d ∉ {2, 3, 4, 6, 8, 9, 12, 16, 24, 27, 36, 54, 64, 81, 108}) ∧
  -- condition for unique number setup from June 4 to June 12
  (∀ d : ℕ, d ∣ n → d ∈ {3}) ∧ -- possible configurations for June 3
  n = 72 := 
sorry

end smallest_number_of_students_l154_154790


namespace lcm_midstep_l154_154701

def lcm (m n : ℕ) : ℕ := sorry -- Insert definition for LCM

theorem lcm_midstep (a b c d : ℕ): 
  lcm (lcm a b) (lcm c d) = 144 :=
by 
  -- Define given values 
  let a := 12
  let b := 16
  let c := 18
  let d := 24
  sorry

end lcm_midstep_l154_154701


namespace circle_centers_lie_on_smaller_sphere_l154_154479

-- Given definitions and conditions
def SphereCenter (C : Point) (R : ℝ) := { P : Point | dist P C = R } -- Original sphere definition with center C and radius R
def CircleOnSphere (C : Point) (R r : ℝ) (P : Point) := P ∈ SphereCenter C R -- Circle lying on the surface of the sphere
def SmallerSphereCenter (C : Point) (R r : ℝ) := { Q : Point | dist Q C = real.sqrt(R^2 - r^2) } -- Smaller sphere definition

-- Proof statement
theorem circle_centers_lie_on_smaller_sphere (C : Point) (R r : ℝ) :
  ∀ P : Point, CircleOnSphere C R r P → P ∈ SmallerSphereCenter C R r :=
sorry

end circle_centers_lie_on_smaller_sphere_l154_154479


namespace total_revenue_full_price_tickets_l154_154880

theorem total_revenue_full_price_tickets (f q : ℕ) (p : ℝ) :
  f + q = 170 ∧ f * p + q * (p / 4) = 2917 → f * p = 1748 := by
  sorry

end total_revenue_full_price_tickets_l154_154880


namespace smallest_integer_with_16_divisors_l154_154077

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (number_of_divisors m = 16) → (m ≥ n)) ∧ (n = 120) :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154077


namespace number_of_deleted_apps_l154_154943

def initial_apps := 16
def remaining_apps := 8

def deleted_apps : ℕ := initial_apps - remaining_apps

theorem number_of_deleted_apps : deleted_apps = 8 := 
by
  unfold deleted_apps initial_apps remaining_apps
  rfl

end number_of_deleted_apps_l154_154943


namespace Poly_irreducible_l154_154761

open Polynomial

noncomputable def irreducible_poly (n : ℕ) (a : Fin n → ℤ) : Polynomial ℤ :=
  (∏ i, (X - C (a i))) - 1

theorem Poly_irreducible (n : ℕ) (a : Fin n → ℤ) (h_distinct : Function.Injective a) :
    Irreducible (irreducible_poly n a) :=
  sorry

end Poly_irreducible_l154_154761


namespace minimize_perimeter_l154_154720

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def A : Point := { x := 1, y := -2 }
def B : Point := { x := 4, y := 0 }
def P (a : ℝ) : Point := { x := a, y := 1 }
def N (a : ℝ) : Point := { x := a + 1, y := 1 }

def perimeter (a : ℝ) : ℝ :=
  distance (P a) A + distance A B + distance B (N a) + distance (N a) (P a)

theorem minimize_perimeter : ∃ (a : ℝ), a = 5 / 2 ∧
  ∀ (x : ℝ), perimeter a ≤ perimeter x := 
by
  use 5 / 2
  sorry

end minimize_perimeter_l154_154720


namespace min_value_of_reciprocals_l154_154698

theorem min_value_of_reciprocals (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = 3) :
  ∃ x y, (1 / x + 1 / y) = 1 + 2 * real.sqrt 2 / 3 :=
sorry

end min_value_of_reciprocals_l154_154698


namespace urn_probability_three_red_three_blue_l154_154556
open Nat

theorem urn_probability_three_red_three_blue :
  let initial_urn := (1, 1) -- 1 red ball, 1 blue ball
  let operations := 4 -- Four drawing and returning operations
  let final_urn := 6 -- Final count of balls in the urn
  let desired_count := (3, 3) -- Desired count of red and blue balls
  -- Prove the probability of having 3 red balls and 3 blue balls
  probability_three_red_three_blue initial_urn operations final_urn = 1 / 5 :=
sorry

end urn_probability_three_red_three_blue_l154_154556


namespace problem_1_problem_2_problem_3_l154_154671

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 4 * x + a - 5

def g (x : ℝ) (m : ℝ) : ℝ := m * 4^(x - 1) - 2 * m + 7

theorem problem_1 (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x a = 0) ↔ 0 ≤ a ∧ a ≤ 8 :=
sorry

theorem problem_2 (m : ℝ) :
  (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 2 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 0 = g x2 m) ↔ 
  (m ∈ set.Iic (-7/2) ∨ m ∈ set.Ici 7) :=
sorry

theorem problem_3 (t : ℝ) :
  let D := set.image (λ x, f x 0) (set.Icc t 2) in
  set.Nonempty D ∧ (∃ t : ℝ, (∀ p q ∈ D, ∃ d ∈ D, d = q - p) ∧ (∀ p q ∈ D, q - p = 6 - 4 * t)) →
  t = (-4 - 3 * real.sqrt 2) ∨ t = (-5 / 2) :=
sorry

end problem_1_problem_2_problem_3_l154_154671


namespace cannot_form_set_l154_154090

-- Definitions based on the given conditions
def A : Set := {x : Type | x = "Table Tennis Player" ∧ x participates in "Hangzhou Asian Games"}
def B : Set := {x ∈ ℕ | x > 0 ∧ x < 5}
def C : Prop := False -- C cannot form a set
def D : Set := {x : Real | ¬ (x ∈ ℚ)}

-- Theorem stating which group cannot form a set
theorem cannot_form_set : (C = False) :=
by
  sorry

end cannot_form_set_l154_154090


namespace relationship_a_b_c_l154_154222

noncomputable def a := Real.log 3 / Real.log (1/2)
noncomputable def b := Real.log (1/2) / Real.log 3
noncomputable def c := Real.exp (0.3 * Real.log 2)

theorem relationship_a_b_c : 
  a < b ∧ b < c := 
by {
  sorry
}

end relationship_a_b_c_l154_154222


namespace probability_heads_9_or_more_12_flips_l154_154028

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154028


namespace common_ranking_possible_l154_154106

-- Represent the number of participants and judges
variables {n m : ℕ}

-- A relation saying that participant A is considered better than participant B by a majority of the judges
def majority_better (judge_opinions : ℕ → (ℕ → ℕ) → Prop) : Prop :=
∀ {A B : ℕ}, (∀ j : ℕ, judge_opinions j A > judge_opinions j B) → (∀ j : ℕ, judge_opinions j B > judge_opinions j C) → ¬ (∀ j : ℕ, judge_opinions j C > judge_opinions j A)

-- Problem conditions
axiom no_three_judges (judge_opinions : ℕ → (ℕ → ℕ) → Prop) 
  (A B C : ℕ) :
  ¬ (∀ (j1 j2 j3 : ℕ), (judge_opinions j1 A B ∧ judge_opinions j1 B C) ∧ 
  (judge_opinions j2 B C ∧ judge_opinions j2 C A) ∧ 
  (judge_opinions j3 C A ∧ judge_opinions j3 A B))

-- The statement to be proved
theorem common_ranking_possible (judge_opinions : ℕ → (ℕ → ℕ) → Prop)
  (h : no_three_judges judge_opinions) :
  ∃ (ranking : ℕ → ℕ), (∀ A B : ℕ, ranking A < ranking B → majority_better judge_opinions A B) :=
sorry

end common_ranking_possible_l154_154106


namespace square_window_side_length_eq_42_l154_154951

noncomputable def side_length_of_square_window : ℕ :=
  let width := 10 in  -- Since width = 10 as arbitrary example in solution
  let height := 3 * width in
  3 * width + 4 * 3

theorem square_window_side_length_eq_42 :
  side_length_of_square_window = 42 :=
by
  sorry

end square_window_side_length_eq_42_l154_154951


namespace increase_in_average_l154_154462

theorem increase_in_average {a1 a2 a3 a4 : ℕ} 
                            (h1 : a1 = 92) 
                            (h2 : a2 = 89) 
                            (h3 : a3 = 91) 
                            (h4 : a4 = 93) : 
    ((a1 + a2 + a3 + a4 : ℚ) / 4) - ((a1 + a2 + a3 : ℚ) / 3) = 0.58 := 
by
  sorry

end increase_in_average_l154_154462


namespace length_of_KN_l154_154308

theorem length_of_KN 
(AB BC CA : ℝ)
(N : Point)
(K : Point)
(h1 : AB = 15)
(h2 : BC = 13)
(h3 : CA = 14)
(h4 : midpoint N A C)
(h5 : foot K B AC)
: length K N = 5 :=
sorry

end length_of_KN_l154_154308


namespace basket_can_hold_40_fruits_l154_154507

-- Let us define the number of oranges as 10
def oranges : ℕ := 10

-- There are 3 times as many apples as oranges
def apples : ℕ := 3 * oranges

-- The total number of fruits in the basket
def total_fruits : ℕ := oranges + apples

theorem basket_can_hold_40_fruits (h₁ : oranges = 10) (h₂ : apples = 3 * oranges) : total_fruits = 40 :=
by
  -- We assume the conditions and derive the conclusion
  sorry

end basket_can_hold_40_fruits_l154_154507


namespace number_of_people_l154_154389

theorem number_of_people
  (x y : ℕ)
  (h1 : x + y = 28)
  (h2 : 2 * x + 4 * y = 92) :
  x = 10 :=
by
  sorry

end number_of_people_l154_154389


namespace min_sum_of_factors_l154_154440

theorem min_sum_of_factors (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 1806) :
  x + y + z ≥ 72 := 
sorry

end min_sum_of_factors_l154_154440


namespace radius_of_sphere_l154_154885

noncomputable def cone_to_sphere_radius (h d_percent_wasted : ℝ) (volume_cone percentage_wasted : ℝ) : ℝ :=
  let volume_sphere := volume_cone / (1 - percentage_wasted) in
  let r_cube := (3 * volume_sphere) / (4 * π) in
  r_cube^(1/3)

theorem radius_of_sphere : cone_to_sphere_radius 9 18 0.75 (1/3 * π * 9^2 * 9) 0.75 = 9 := 
  sorry

end radius_of_sphere_l154_154885


namespace quadratic_inequality_solution_set_l154_154447

theorem quadratic_inequality_solution_set :
  {x : ℝ | - x ^ 2 + 4 * x + 12 > 0} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end quadratic_inequality_solution_set_l154_154447


namespace composite_and_sum_divisible_by_3_probability_l154_154699

theorem composite_and_sum_divisible_by_3_probability :
  (∃ (dices : Fin 7 → Fin 6), let product := (List.ofFn dices).prod; 
    Nat.isComposite product ∧ Nat.Mod (List.ofFn dices).sum 3 = 0) →
  (∃ probability, probability = 1 / 3) :=
sorry

end composite_and_sum_divisible_by_3_probability_l154_154699


namespace a_plus_b_four_l154_154217

noncomputable def f (a b x : ℝ) :=
if x < 3 then a * x + b else 9 - 2 * x

theorem a_plus_b_four (a b : ℝ) :
    (∀ x : ℝ, f a b (f a b x) = x) ∧ (∀ f, ∀ x : ℝ, (continuous_at (λ y, f a b y) 3)) → a + b = 4 :=
by sorry

end a_plus_b_four_l154_154217


namespace domain_of_composed_function_l154_154669

variable {α : Type*} [LinearOrderedField α]

def function_domain (f : α → α) (dom : Set α) : Prop :=
  ∀ x, f x ∈ dom → x ∈ dom

theorem domain_of_composed_function :
  ∀ (f : α → α), 
    function_domain f (Set.Icc (-(1 : α)) (4 : α)) →
    function_domain (λ x, f (2 * x - 1)) (Set.Icc (0 : α) (5 / 2 : α)) :=
  sorry

end domain_of_composed_function_l154_154669


namespace sqrt_combination_l154_154145

theorem sqrt_combination : 
    ∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 8) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 3))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 12))) ∧ 
    (¬(∃ (k : ℝ), (k * Real.sqrt 2 = Real.sqrt 0.2))) :=
by
  sorry

end sqrt_combination_l154_154145


namespace probability_all_heads_or_tails_l154_154208

def coin_outcomes := {heads, tails}

def total_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes : ℕ := 2

def probability_five_heads_or_tails (n : ℕ) (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_all_heads_or_tails :
  probability_five_heads_or_tails 5 (total_outcomes 5) favorable_outcomes = 1 / 16 :=
by
  sorry

end probability_all_heads_or_tails_l154_154208


namespace smallest_integer_with_16_divisors_l154_154069

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (∀ m : ℕ, (has_16_divisors m → m ≥ n)) ∧ n = 384 :=
by
  def has_16_divisors (n : ℕ) : Prop :=
    ∃ a₁ a₂ a₃ k₁ k₂ k₃, n = k₁ ^ a₁ * k₂ ^ a₂ * k₃ ^ a₃ ∧ (a₁ + 1) * (a₂ + 1) * (a₃ + 1) = 16
  -- Proof will go here
  sorry

end smallest_integer_with_16_divisors_l154_154069


namespace A_alone_completion_time_l154_154485

theorem A_alone_completion_time:
  (A B: ℕ) -- A and B are natural numbers representing the days taken by A and B alone to finish the work.
  (h1 : 1 / A + 1 / B = 1 / 40) -- Together, they can finish the work in 40 days
  (h2 : 10 * (1 / 40) = 1 / 4) -- Work done together in 10 days
  (h3 : 15 * (1 / A) = 3 / 4) -- Work done by A alone in 15 days
  : A = 20 := sorry

end A_alone_completion_time_l154_154485


namespace operation_result_l154_154627

variable (a b : ℝ)

def custom_oplus (a b : ℝ) : ℝ :=
  if a < b then a else b

def custom_otimes (a b : ℝ) : ℝ :=
  if a < b then b else a

theorem operation_result :
  (custom_oplus 2003 2004) |> custom_otimes (custom_oplus 2005 2006) = 2005 :=
by
  sorry

end operation_result_l154_154627


namespace prove_francie_weeks_l154_154220

def francie_weeks_increased_allowance (initial_allowance_weeks : ℕ) (initial_allowance : ℕ) 
(increased_allowance : ℕ) (video_game_cost : ℕ) (remaining_money : ℕ) (total_savings : ℕ) : 
Prop :=
let initial_savings := initial_allowance_weeks * initial_allowance in
let increased_savings := total_savings - initial_savings in
let weeks_increased_allowance := increased_savings / increased_allowance in
initial_allowance_weeks = 8 ∧
initial_allowance = 5 ∧
increased_allowance = 6 ∧
(video_game_cost + remaining_money) * 2 = total_savings ∧
total_savings = 76 ∧
(weeks_increased_allowance = 6)

theorem prove_francie_weeks : francie_weeks_increased_allowance 8 5 6 35 3 76 :=
by
  sorry

end prove_francie_weeks_l154_154220


namespace triangle_ratio_ABC_l154_154450

noncomputable def AC_CD_ratio (A B C D : Point) (angle_ABC : Angle) (angle_CBD : Angle) (CD_AB : Real) : Real :=
  let ∠B := 90 * π / 180
  let ∠CBD := 30 * π / 180
  let CD = AB
  (AC / CD)

theorem triangle_ratio_ABC (A B C D : Point) (CD_AB CD : CD_AB) (angle_ABC : angle_ABC = 90°) (angle_CBD : angle_CBD = 30°)
  (CD_eq_AB : CD = AB)
  : AC_CD_ratio A B C D angle_ABC angle_CBD CD_AB = 2 :=
begin
  sorry
end

end triangle_ratio_ABC_l154_154450


namespace expected_value_of_uniform_distribution_l154_154194

noncomputable def expected_value_uniform (a b : ℝ) (X : ℝ → ℝ)
  (h : X ∈ set.Icc a b) : ℝ :=
∫ (x : ℝ) in set.Icc a b, x * (1 / (b - a))

theorem expected_value_of_uniform_distribution (a b : ℝ) :
  expected_value_uniform a b (λ x, if a ≤ x ∧ x ≤ b then x else 0) (by sorry) = (a + b) / 2 :=
sorry

end expected_value_of_uniform_distribution_l154_154194


namespace igor_is_5_l154_154541

-- Define the initial lineup
def initial_lineup := [9, 11, 10, 6, 8, 5, 4, 1]

-- Define the condition on when players leave
def should_run_off (lineup : List ℕ) (idx : ℕ) : Prop :=
  (idx = 0 ∨ idx = lineup.length - 1 ∨ lineup.getD idx 0 < (lineup.getD (idx - 1) 0)) ∨
  (idx < lineup.length - 1 ∧ lineup.getD idx 0 < (lineup.getD (idx + 1) 0))

-- Define the condition when Igor leaves
def igor_left_when_three_players_remain (lineup : List ℕ) (igor_number : ℕ) : Prop :=
  ∃ rest_of_lineup, length rest_of_lineup = 3 ∧ igor_number ∈ initial_lineup ∧ 
  should_run_off lineup (lineup.indexOf' igor_number)  

-- Define the theorem statement
theorem igor_is_5 : igor_left_when_three_players_remain initial_lineup 5 :=
  sorry

end igor_is_5_l154_154541


namespace positive_integer_solution_lcm_eq_sum_l154_154180

def is_lcm (x y z m : Nat) : Prop :=
  ∃ (d : Nat), x = d * (Nat.gcd y z) ∧ y = d * (Nat.gcd x z) ∧ z = d * (Nat.gcd x y) ∧
  x * y * z / Nat.gcd x (Nat.gcd y z) = m

theorem positive_integer_solution_lcm_eq_sum :
  ∀ (a b c : Nat), 0 < a → 0 < b → 0 < c → is_lcm a b c (a + b + c) → (a, b, c) = (a, 2 * a, 3 * a) := by
    sorry

end positive_integer_solution_lcm_eq_sum_l154_154180


namespace intersection_of_sets_example_l154_154360

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l154_154360


namespace percentage_vanaspati_after_adding_ghee_l154_154719

theorem percentage_vanaspati_after_adding_ghee :
  ∀ (original_quantity new_pure_ghee percentage_ghee percentage_vanaspati : ℝ),
    original_quantity = 30 →
    percentage_ghee = 0.5 →
    percentage_vanaspati = 0.5 →
    new_pure_ghee = 20 →
    (percentage_vanaspati * original_quantity) /
    (original_quantity + new_pure_ghee) * 100 = 30 :=
by
  intros original_quantity new_pure_ghee percentage_ghee percentage_vanaspati
  sorry

end percentage_vanaspati_after_adding_ghee_l154_154719


namespace vehicles_traveled_l154_154749

theorem vehicles_traveled (A : ℕ) (V : ℕ) (ratio : ℕ → ℕ → Prop) (h_ratio : ratio 1 1000000) (h_accidents : A = 2000) : V = 2000000000 :=
by
  have h1 : ratio 2000 V, from sorry
  have h2 : V = 2000000000, from sorry
  exact h2

end vehicles_traveled_l154_154749


namespace min_value_frac_l154_154985

theorem min_value_frac (x y : ℝ) (h₁ : x + y = 1) (h₂ : x > 0) (h₃ : y > 0) : 
  ∃ c, (∀ (a b : ℝ), (a + b = 1) → (a > 0) → (b > 0) → (1/a + 4/b) ≥ c) ∧ c = 9 :=
by
  sorry

end min_value_frac_l154_154985


namespace ending_number_correct_l154_154838

def does_not_contain_digit_1 (n : ℕ) : Prop :=
  ∀ (d ∈ (Nat.digits 10 n)), d ≠ 1

noncomputable def ending_number : ℕ :=
  {n : ℕ // (∃ (count : ℕ), count = 728 ∧ ∀ i, 0 ≤ i ∧ i ≤ n → does_not_contain_digit_1 i)}

theorem ending_number_correct : ending_number = 989 :=
  sorry

end ending_number_correct_l154_154838


namespace time_for_first_train_to_cross_second_train_is_80_seconds_l154_154492

-- Define given conditions
def length_first_train : ℝ := 300 -- meters
def speed_first_train_kmph : ℝ := 72 -- km per hour
def length_second_train : ℝ := 500 -- meters
def speed_second_train_kmph : ℝ := 36 -- km per hour

-- Convert speeds from kmph to m/s
def speed_first_train : ℝ := speed_first_train_kmph * 1000 / 3600 -- meters per second
def speed_second_train : ℝ := speed_second_train_kmph * 1000 / 3600 -- meters per second

-- Define relative speed
def relative_speed : ℝ := speed_first_train - speed_second_train -- meters per second

-- Define total distance to be covered
def total_distance : ℝ := length_first_train + length_second_train -- meters

-- Time to cross
def time_to_cross : ℝ := total_distance / relative_speed 

-- The proof problem statement
theorem time_for_first_train_to_cross_second_train_is_80_seconds :
  time_to_cross = 80 := by
  sorry

end time_for_first_train_to_cross_second_train_is_80_seconds_l154_154492


namespace friendships_structure_count_l154_154317

/-- In a group of 8 individuals, where each person has exactly 3 friends within the group,
there are 420 different ways to structure these friendships. -/
theorem friendships_structure_count : 
  ∃ (structure_count : ℕ), 
    structure_count = 420 ∧ 
    (∀ (G : Fin 8 → Fin 8 → Prop), 
      (∀ i, ∃! (j₁ j₂ j₃ : Fin 8), G i j₁ ∧ G i j₂ ∧ G i j₃) ∧ 
      (∀ i j, G i j → G j i) ∧ 
      (structure_count = 420)) := 
by
  sorry

end friendships_structure_count_l154_154317


namespace height_isosceles_trapezoid_l154_154420

variables (S α : ℝ)

noncomputable def height_of_isosceles_trapezoid (S : ℝ) (α : ℝ) : ℝ :=
  sqrt (S * tan (α / 2))

theorem height_isosceles_trapezoid :
  ∀ (S α : ℝ), height_of_isosceles_trapezoid S α = sqrt (S * tan (α / 2)) :=
by
  intro S α
  -- proof skipped
  sorry

end height_isosceles_trapezoid_l154_154420


namespace solve_math_problem_l154_154947

def is_true_proposition_1 (m l : Set Point) (α : Set Set Point) (A : Point) : Prop :=
  m ⊆ α ∧ (l ∩ α) = {A} ∧ A ∉ m → (l ∪ m).Order > 2  -- meaning l and m are skew lines.

def is_true_proposition_2 (l m n : Set Point) (α : Set Set Point) : Prop :=
  l.Order = 2 ∧ m.Order = 2 ∧ l.Union ∩ m = ∅ ∧ l ∪ m ⊆ α ∧
  (n ∩ l = ∅ ∧ n ∩ m = ∅) ∧ (l ∩ α).Empty ∧ (m ∩ α).Empty → n ∩ α = ∅

def is_true_proposition_3 (l m : Set Point) (α β : Set Set Point) : Prop :=
  l.Union ∩ α = l ∧ m.Union ∩ β = m ∧ (α.Union ∩ β = α) → l ∩ m = ∅

def is_true_proposition_4 (l m : Set Point) (α β : Set Set Point) : Prop :=
  l.Union ∩ α = l ∧ m.Union ∩ α = m ∧ (l ∩ m).Empty ∧ (l.Union ∩ β = l) ∧ 
  (m.Union ∩ β = m) → α.Union ∩ β = α  -- meaning α is parallel to β.

def math_problem : Prop :=
  let p1 := is_true_proposition_1 m l α A
  let p2 := is_true_proposition_2 l m n α
  let p3 := is_true_proposition_3 l m α β
  let p4 := is_true_proposition_4 l m α β
  {p1, p2, p3, p4}.count(λ p, p = true) = 3

theorem solve_math_problem : math_problem :=
by
  sorry

end solve_math_problem_l154_154947


namespace fx_monotonic_increasing_l154_154433

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem fx_monotonic_increasing : ∀ x : ℝ, x > 1 → (f' x) > 0 := by
  sorry

end fx_monotonic_increasing_l154_154433


namespace cube_edge_sums_not_distinct_l154_154734

def vertex_labels : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def edges : Finset (ℕ × ℕ) :=
  {(1, 2), (2, 3), (3, 4), (4, 1),
   (5, 6), (6, 7), (7, 8), (8, 5),
   (1, 5), (2, 6), (3, 7), (4, 8)}

def edge_sums (labeling : ℕ → ℕ) : Finset ℕ :=
  edges.image (λ e, labeling e.1 + labeling e.2)

theorem cube_edge_sums_not_distinct (labeling : ℕ → ℕ) (h : ∀ v ∈ vertex_labels, labeling v ∈ vertex_labels) :
  ∃ (e₁ e₂ : ℕ × ℕ), e₁ ∈ edges ∧ e₂ ∈ edges ∧ e₁ ≠ e₂ ∧ labeling e₁.1 + labeling e₁.2 = labeling e₂.1 + labeling e₂.2 :=
by {
  sorry
}

end cube_edge_sums_not_distinct_l154_154734


namespace max_value_sin_expression_l154_154998

def max_of_sin_expression (x y z : ℝ) : ℝ :=
  Real.sin (x - y) + Real.sin (y - z) + Real.sin (z - x)

theorem max_value_sin_expression : ∀ (x y z : ℝ),
  (0 ≤ x ∧ x ≤ Real.pi) →
  (0 ≤ y ∧ y ≤ Real.pi) →
  (0 ≤ z ∧ z ≤ Real.pi) →
  (max_of_sin_expression x y z ≤ 2) ∧ (∃ x y z : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ 0 ≤ y ∧ y ≤ Real.pi ∧ 0 ≤ z ∧ z ≤ Real.pi ∧ max_of_sin_expression x y z = 2) :=
by
  sorry

end max_value_sin_expression_l154_154998


namespace infinitely_many_tuples_l154_154407

theorem infinitely_many_tuples :
  ∃ infinitely_many (a b c d : ℕ), a^3 + b^4 + c^5 = d^7 := sorry

end infinitely_many_tuples_l154_154407


namespace greatest_b_for_no_minus_nine_in_range_l154_154473

theorem greatest_b_for_no_minus_nine_in_range :
  ∃ b_max : ℤ, (b_max = 16) ∧ (∀ b : ℤ, (b^2 < 288) ↔ (b ≤ 16)) :=
by
  sorry

end greatest_b_for_no_minus_nine_in_range_l154_154473


namespace find_a_l154_154659

theorem find_a (a : ℝ) :
  let x := (1 + a * complex.I) * (2 + complex.I) in
  x.re = x.im → a = 1/3 :=
by
  sorry

end find_a_l154_154659


namespace line_parallel_perpendicular_plane_implies_plane_perpendicular_l154_154674

variables {l m : Line} {α β : Plane}

def parallel (l : Line) (α : Plane) : Prop :=
  ∃ (v : Vector), (v ≠ 0) ∧ (∀ p ∈ l, ∀ q ∈ α, v ⬝ (q - p) = 0)

def perpendicular (l : Line) (β : Plane) : Prop :=
  ∃ (v w : Vector), (v ≠ 0) ∧ (w ≠ 0) ∧ (∃ p ∈ l, v = p) ∧ (∃ p ∈ β, w = p) ∧ (v ⬝ w = 0)

theorem line_parallel_perpendicular_plane_implies_plane_perpendicular
  (h1 : parallel l α)
  (h2 : perpendicular l β) :
  perpendicular α β :=
sorry

end line_parallel_perpendicular_plane_implies_plane_perpendicular_l154_154674


namespace constant_term_binomial_expansion_l154_154962

theorem constant_term_binomial_expansion : 
  (x^2 + 1/Real.sqrt x)^5.coeff 0 = 5 :=
sorry

end constant_term_binomial_expansion_l154_154962


namespace tan_alpha_beta_sum_l154_154293

theorem tan_alpha_beta_sum (α β : ℝ) 
  (h1 : tan α + tan β - tan α * tan β + 1 = 0)
  (h2 : α ∈ Ioo (π / 2) π)
  (h3 : β ∈ Ioo (π / 2) π) : 
  α + β = 7 * π / 4 :=
by
  sorry

end tan_alpha_beta_sum_l154_154293


namespace min_value_expression_l154_154229

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 :=
by
  sorry

end min_value_expression_l154_154229


namespace cos_double_angle_l154_154254

theorem cos_double_angle (α : ℝ) (h : Real.cos (π - α) = -3/5) : Real.cos (2 * α) = -7/25 :=
  sorry

end cos_double_angle_l154_154254


namespace sqrt_product_l154_154591

theorem sqrt_product (h1 : Real.sqrt 81 = 9) 
                     (h2 : Real.sqrt 16 = 4) 
                     (h3 : Real.sqrt (Real.sqrt (Real.sqrt 64)) = 2 * Real.sqrt 2) : 
                     Real.sqrt 81 * Real.sqrt 16 * Real.sqrt (Real.sqrt (Real.sqrt 64)) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l154_154591


namespace value_of_expression_eq_33_l154_154834

theorem value_of_expression_eq_33 : (3^2 + 7^2 - 5^2 = 33) := by
  sorry

end value_of_expression_eq_33_l154_154834


namespace number_of_six_digit_flippy_divisible_by_15_l154_154136

def is_flippy (n : ℕ) : Prop :=
  let d := n.digits in
  d.length = 6 ∧
  d.length > 0 ∧
  ∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ list.filter (λ a, a = x) d = [x, x]
  ∧ list.filter (λ a, a = y) d = [y, y] ∧ list.filter (λ a, a = z) d = [z, z].

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

def six_digit_flippy_divisible_by_15_count : ℕ :=
  finset.card (finset.filter (λ n, is_flippy n ∧ divisible_by_15 n) (finset.range (10 ^ 6)))

theorem number_of_six_digit_flippy_divisible_by_15 :
  six_digit_flippy_divisible_by_15_count = 5 :=
sorry

end number_of_six_digit_flippy_divisible_by_15_l154_154136


namespace intersection_M_N_l154_154364

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l154_154364


namespace can_all_be_black_l154_154852

-- Define a graph with vertices and a finite set of edges
structure Graph :=
  (V : Type) -- the type of vertices
  [finite : Fintype V]
  (E : V → V → Prop) -- edge relation

-- Condition: No loops (no vertex has an edge to itself)
def no_loops (G : Graph) : Prop :=
  ∀ v : G.V, ¬ G.E v v

-- Condition: No multiple edges (edge relation is symmetric)
def no_multiple_edges (G : Graph) : Prop :=
  ∀ u v : G.V, G.E u v → G.E v u

-- State representation of vertices (true = white, false = black)
def state (G : Graph) := G.V → bool

-- Initial state: All vertices are white
def initial_state (G : Graph) : state G :=
  λ _, true

-- Operation: Switch a vertex and its neighbors
def switch (G : Graph) (s : state G) (v : G.V) : state G :=
  λ u, if u = v ∨ G.E v u then bnot (s u) else s u

-- Main theorem: Prove that we can switch vertices to make all vertices black
theorem can_all_be_black (G : Graph) [Fintype G.V] :
  no_loops G → no_multiple_edges G → 
  ∃ seq_of_ops : list G.V, 
    (initial_state G, seq_of_ops.foldl (λ s v, switch G s v) (initial_state G)) = (initial_state G, λ _, false) :=
begin
  intros hnl hme,
  -- Proof to be completed
  sorry
end

end can_all_be_black_l154_154852


namespace option_d_correct_l154_154294

theorem option_d_correct (a b : ℝ) (h : a > b) : -b > -a :=
sorry

end option_d_correct_l154_154294


namespace find_a_l154_154263

namespace math_proof

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ln (a * x - 1)

theorem find_a (a : ℝ) (h₁ : ∀ x, deriv (f a) x = a / (a * x - 1))
  (h₂ : deriv (f a) 2 = 2) : a = 2 / 3 :=
by
  sorry

end math_proof

end find_a_l154_154263


namespace partition_pos_int_as_sum_of_three_l154_154718

theorem partition_pos_int_as_sum_of_three (n : ℕ) (h : n > 0) :
  (∑ x in finset.range (n-2), (n - x - 1)) = (n-1)*(n-2)/2 :=
sorry

end partition_pos_int_as_sum_of_three_l154_154718


namespace smallest_integer_with_16_divisors_l154_154067

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (∀ m : ℕ, (has_16_divisors m → m ≥ n)) ∧ n = 384 :=
by
  def has_16_divisors (n : ℕ) : Prop :=
    ∃ a₁ a₂ a₃ k₁ k₂ k₃, n = k₁ ^ a₁ * k₂ ^ a₂ * k₃ ^ a₃ ∧ (a₁ + 1) * (a₂ + 1) * (a₃ + 1) = 16
  -- Proof will go here
  sorry

end smallest_integer_with_16_divisors_l154_154067


namespace calculate_second_rate_l154_154555

noncomputable def second_certificate_interest_rate (investment : ℝ) (first_rate : ℝ) (final_amount : ℝ) : ℝ :=
  let intermediate_amount := investment * (1 + (first_rate / 100) * (9 / 12))
  let s := ((final_amount / intermediate_amount - 1) / (9 / 12)) * 100
  s

theorem calculate_second_rate
  (initial_investment : ℝ := 15000)
  (first_annual_interest_rate : ℝ := 8)
  (final_amount : ℝ := 16620)
  (second_annual_interest_rate : ℝ := second_certificate_interest_rate initial_investment first_annual_interest_rate final_amount)
  : second_annual_interest_rate ≈ 6.04 := 
sorry

end calculate_second_rate_l154_154555


namespace remainder_783245_div_7_l154_154859

theorem remainder_783245_div_7 :
  783245 % 7 = 1 :=
sorry

end remainder_783245_div_7_l154_154859


namespace solution_l154_154496

noncomputable def inequality_prove (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5)

noncomputable def equality_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5) ↔ (x = 2 ∧ y = 2)

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : 
  inequality_prove x y h1 h2 h3 ∧ equality_condition x y h1 h2 h3 := by
  sorry

end solution_l154_154496


namespace smallest_integer_with_16_divisors_l154_154072

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (number_of_divisors m = 16) → (m ≥ n)) ∧ (n = 120) :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154072


namespace integers_with_product_72_and_difference_4_have_sum_20_l154_154441

theorem integers_with_product_72_and_difference_4_have_sum_20 :
  ∃ (x y : ℕ), (x * y = 72) ∧ (x - y = 4) ∧ (x + y = 20) :=
sorry

end integers_with_product_72_and_difference_4_have_sum_20_l154_154441


namespace consistency_condition_l154_154961

variable 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)

theorem consistency_condition :
  (∃ x y : ℝ, 
    a1 * x + b1 * y = c1 ∧ 
    a2 * x + b2 * y = c2 ∧ 
    a3 * x + b3 * y = c3) ↔ 
  (a1 * (b2 * c3 - b3 * c2) + 
   a2 * (b3 * c1 - c3 * b1) + 
   a3 * (b1 * c2 - b2 * c1) = 0) :=
by sorry

end consistency_condition_l154_154961


namespace locus_of_K_is_circle_l154_154994

noncomputable def point_is_on_circle (A B C K : Point) (r1 r2 : ℝ) :=
  let circle1 := Circle A r1
  let circle2 := Circle A r2
  B ∈ circle1 ∧ C ∈ circle2 ∧
  ∃ M N : Point, 
    (∃ (r : ℝ), is_tangent_circle M r B ∧ is_tangent_circle N r C ∧
    midpoint(M, N) = K ) →
    (distance A K = constant)

theorem locus_of_K_is_circle (A B C K : Point) (r1 r2 : ℝ) :
  point_is_on_circle A B C K r1 r2 → (∃ r : ℝ, Circle A r ∋ K) :=
begin
  -- proof
  sorry
end

end locus_of_K_is_circle_l154_154994


namespace coefficient_of_x3_in_expansion_l154_154811

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def expansion_coefficient_x3 : ℤ :=
  let term1 := (-1 : ℤ) ^ 3 * binomial_coefficient 6 3
  let term2 := (1 : ℤ) * binomial_coefficient 6 2
  term1 + term2

theorem coefficient_of_x3_in_expansion :
  expansion_coefficient_x3 = -5 := by
  sorry

end coefficient_of_x3_in_expansion_l154_154811


namespace intersection_M_N_l154_154349

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l154_154349


namespace part1_part2_l154_154260

-- Definitions
def f (x : ℝ) (a b : ℝ) : ℝ := Real.exp x + a * x + b

-- Given conditions
variable (a b : ℝ) (x : ℝ) (h0 : f 0 a b = 1) (h1 : f' 0 a b = 2) (h2 : ∀ x > 0, f x 1 0 ≥ x^2 + m * x + 1)

-- Theorem statement
theorem part1 : f x 1 0 = Real.exp x + x := by
  sorry

theorem part2 (m : ℝ) : (∀ x > 0, f x 1 0 ≥ x^2 + m * x + 1) → m ≤ Real.exp 1 - 1 := by
  sorry

end part1_part2_l154_154260


namespace total_tiles_needed_l154_154171

-- Define the dimensions of the dining room
def dining_room_length : ℕ := 15
def dining_room_width : ℕ := 20

-- Define the width of the border
def border_width : ℕ := 2

-- Areas for one-foot by one-foot border tiles
def one_foot_tile_border_tiles : ℕ :=
  2 * (dining_room_width + (dining_room_width - 2 * border_width)) + 
  2 * ((dining_room_length - 2) + (dining_room_length - 2 * border_width))

-- Dimensions of the inner area
def inner_length : ℕ := dining_room_length - 2 * border_width
def inner_width : ℕ := dining_room_width - 2 * border_width

-- Area for two-foot by two-foot tiles
def inner_area : ℕ := inner_length * inner_width
def two_foot_tile_inner_tiles : ℕ := inner_area / 4

-- Total number of tiles
def total_tiles : ℕ := one_foot_tile_border_tiles + two_foot_tile_inner_tiles

-- Prove that the total number of tiles needed is 168
theorem total_tiles_needed : total_tiles = 168 := sorry

end total_tiles_needed_l154_154171


namespace not_always_correct_l154_154286

theorem not_always_correct (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : y < x) (hz : z ≠ 0) :
  ¬ ∀ x y z, (0 < x → 0 < y → y < x → z ≠ 0 → |x + z| > |y + z|) :=
by
  intro h
  sorry

end not_always_correct_l154_154286


namespace monotonic_intervals_max_min_values_l154_154271

noncomputable def f : ℝ → ℝ := λ x => (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonic_intervals :
  (∀ x, x < -3 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x > 0) ∧
  (∀ x, -3 < x ∧ x < 1 → deriv f x < 0) :=
by
  sorry

theorem max_min_values :
  f 2 = 5 / 3 ∧ f 1 = -2 / 3 :=
by
  sorry

end monotonic_intervals_max_min_values_l154_154271


namespace complement_of_set_A_in_U_l154_154283

open Set Real

theorem complement_of_set_A_in_U :
  let U := {x : ℝ | - sqrt 3 < x}
  let A := {x : ℝ | 2^x > sqrt 2}
  (U \ A) = {x : ℝ | - sqrt 3 ≤ x ∧ x ≤ 1 / 2} :=
by
  let U := {x : ℝ | - sqrt 3 < x}
  let A := {x : ℝ | 2^x > sqrt 2}
  have step1 : 2^(1/2) = sqrt 2 := sorry
  have step2 : 2^x > sqrt 2 ↔ x > 1/2 := sorry
  have A_def : A = {x : ℝ | x > 1/2} := sorry
  have step3 : U \ A = {x : ℝ | - sqrt 3 < x ∧ x ≤ 1/2} := sorry
  exact set.ext (λ x, iff.intro (λ hx, ⟨hx.left, hx.right⟩) (λ hx, ⟨hx.left, hx.right⟩))
  sorry

end complement_of_set_A_in_U_l154_154283


namespace WalterWorksDaysAWeek_l154_154466

theorem WalterWorksDaysAWeek (hourlyEarning : ℕ) (hoursPerDay : ℕ) (schoolAllocationFraction : ℚ) (schoolAllocation : ℕ) 
  (dailyEarning : ℕ) (weeklyEarning : ℕ) (daysWorked : ℕ) :
  hourlyEarning = 5 →
  hoursPerDay = 4 →
  schoolAllocationFraction = 3 / 4 →
  schoolAllocation = 75 →
  dailyEarning = hourlyEarning * hoursPerDay →
  weeklyEarning = (schoolAllocation : ℚ) / schoolAllocationFraction →
  daysWorked = weeklyEarning / dailyEarning →
  daysWorked = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end WalterWorksDaysAWeek_l154_154466


namespace binary_add_mul_l154_154586

def x : ℕ := 0b101010
def y : ℕ := 0b11010
def z : ℕ := 0b1110
def result : ℕ := 0b11000000000

theorem binary_add_mul : ((x + y) * z) = result := by
  sorry

end binary_add_mul_l154_154586


namespace tangent_parallel_to_line_l154_154448

noncomputable def f : ℝ → ℝ := λ x, x^3 + x - 2

theorem tangent_parallel_to_line (x y : ℝ) :
  ∃ x y, (f x = y) ∧ (3 * x^2 + 1 = 4) ∧ (x = 1 ∧ y = 0) :=
by
  sorry

end tangent_parallel_to_line_l154_154448


namespace problem_l154_154805

-- Definitions and conditions
def Center (A B C D O : Point) : Prop :=
  midpoint (A, C) = O ↔ midpoint (B, D) = O

def Square (A B C D : Point): Prop :=
  perpendicular (A, B, D) ∧ perpendicular (A, D, C) ∧ eq_dist (A, B, D, C)

def AngleEOF (E O F : Point) : Prop :=
  angle E O F = 30

def LineSegment (A B E F: Point) : Prop :=
  is_between A E F

-- Given Statements
variables (A B C D O E F : Point)

theorem problem :
  Square A B C D →
  Center A B C D O →
  distance A B = 1200 →
  LineSegment A B E F →
  distance E F = 500 →
  AngleEOF E O F → 
  ∃ (p q r: ℕ), BF = p + q * sqrt r ∧ r ∣ r ∧ is_not_divisible_by_square_of_any_prime r ∧ p + q + r = 503 :=
by
  sorry

end problem_l154_154805


namespace apples_after_operations_l154_154828

-- Define the initial conditions
def initial_apples : ℕ := 38
def used_apples : ℕ := 20
def bought_apples : ℕ := 28

-- State the theorem we want to prove
theorem apples_after_operations : initial_apples - used_apples + bought_apples = 46 :=
by
  sorry

end apples_after_operations_l154_154828


namespace smallest_portion_bread_l154_154111

theorem smallest_portion_bread (a d : ℚ) (h1 : 5 * a = 100) (h2 : 24 * d = 11 * a) :
  a - 2 * d = 5 / 3 :=
by
  -- Solution proof goes here...
  sorry -- placeholder for the proof

end smallest_portion_bread_l154_154111


namespace sum_of_reciprocals_eq_l154_154933

def triangular_number (n : ℕ) : ℚ :=
  n * (n + 1) / 2

def sum_of_reciprocals_first_100_triangulars : ℚ :=
  ∑ n in Finset.range 100, 1 / triangular_number (n + 1)

theorem sum_of_reciprocals_eq : 
  sum_of_reciprocals_first_100_triangulars = 200 / 101 :=
by
  sorry

end sum_of_reciprocals_eq_l154_154933


namespace number_of_perfect_square_multiples_le_2000_l154_154970

theorem number_of_perfect_square_multiples_le_2000 :
  {n : ℕ | n ≤ 2000 ∧ ∃ k : ℕ, 10 * n = k^2}.finite.card = 14 := by
sorry

end number_of_perfect_square_multiples_le_2000_l154_154970


namespace problem_solution_l154_154902

noncomputable def polynomial_property (p : ℝ → ℝ) := ∀ x : ℝ, p(x + 1) - p(x) = x^100

theorem problem_solution (p : ℝ → ℝ) (h1 : polynomial_property p) :
  ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 / 2 → p(1 - t) ≥ p(t) := 
sorry

end problem_solution_l154_154902


namespace exists_triangle_with_conditions_l154_154604

theorem exists_triangle_with_conditions :
  ∃ (T : Triangle), (∀ h ∈ altitudes T, h < 0.01) ∧ (area T > 1) :=
sorry

end exists_triangle_with_conditions_l154_154604


namespace volume_of_pyramid_in_cone_l154_154523

noncomputable def volume_of_inscribed_pyramid (K : ℝ) (alpha beta gamma : ℝ ) : ℝ :=
  (2 * K / Real.pi) * (Real.sin alpha) * (Real.sin beta) * (Real.sin gamma)

theorem volume_of_pyramid_in_cone :
  volume_of_inscribed_pyramid 20 (25 + 44 / 60 + 12 / 3600).toRadians (82 + 12 / 60 + 40 / 3600).toRadians (72 + 3 / 60 + 8 / 3600).toRadians = 5.21 :=
by
  sorry

end volume_of_pyramid_in_cone_l154_154523


namespace problem_statement_l154_154927

theorem problem_statement : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3) ^ 2 = 4 - 2 * Real.sqrt 6 := 
by 
  sorry

end problem_statement_l154_154927


namespace lattice_points_count_is_correct_l154_154200

-- Define the coordinates of the endpoints
def x1 : ℤ := 5
def y1 : ℤ := 23
def x2 : ℤ := 65
def y2 : ℤ := 407

-- The main theorem to prove the count of lattice points is 13
theorem lattice_points_count_is_correct : 
  let dx := x2 - x1,
      dy := y2 - y1 in
  -- gcd computation to find the number of lattice points
  let gcd := Int.gcd dx dy in
  -- Number of lattice points on the segment (dx/gcd) + 1 considering both endpoints
  gcd + 1 = 13 :=
sorry

end lattice_points_count_is_correct_l154_154200


namespace smallest_integer_with_16_divisors_l154_154080

theorem smallest_integer_with_16_divisors : 
  ∃ (n : ℕ), (∃ (p_1 p_2 p_3 : ℕ) (a_1 a_2 a_3 : ℕ), 
  (p_1 = 2 ∧ p_2 = 3 ∧ a_1 = 3 ∧ a_2 = 3 ∧ n = p_1 ^ a_1 * p_2 ^ a_2) ∧
  (∀ m, m > 0 → (∃ b1 b2 ..., m has exactly 16 positive divisors) → 216 ≤ m)) := 
sorry

end smallest_integer_with_16_divisors_l154_154080


namespace initial_money_is_correct_l154_154401

-- Given conditions
def spend_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12
def money_left_after_year : ℕ := 104

-- Define the initial amount of money
def initial_amount_of_money (spend_per_trip trips_per_month months_per_year money_left_after_year : ℕ) : ℕ :=
  money_left_after_year + (spend_per_trip * trips_per_month * months_per_year)

-- Theorem stating that under the given conditions, the initial amount of money is 200
theorem initial_money_is_correct :
  initial_amount_of_money spend_per_trip trips_per_month months_per_year money_left_after_year = 200 :=
  sorry

end initial_money_is_correct_l154_154401


namespace problem_part1_problem_part2_l154_154266

open Real

noncomputable def f (x : ℝ) := (sqrt 3) * sin x * cos x - (1 / 2) * cos (2 * x)

theorem problem_part1 : 
  (∀ x : ℝ, -1 ≤ f x) ∧ 
  (∃ T : ℝ, (T > 0) ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = π) := 
sorry

theorem problem_part2 (C A B c : ℝ) :
  (f C = 1) → 
  (B = π / 6) → 
  (c = 2 * sqrt 3) → 
  ∃ b : ℝ, ∃ area : ℝ, b = 2 ∧ area = (1 / 2) * b * c * sin A ∧ area = 2 * sqrt 3 := 
sorry

end problem_part1_problem_part2_l154_154266


namespace union_complement_eq_l154_154778

def U := {1, 2, 3, 4, 5}
def M := {1, 4}
def N := {2, 5}

def complement (univ : Set ℕ) (s : Set ℕ) : Set ℕ :=
  {x ∈ univ | x ∉ s}

theorem union_complement_eq :
  N ∪ (complement U M) = {2, 3, 5} :=
by sorry

end union_complement_eq_l154_154778


namespace complex_quadrant_condition_l154_154422

-- Definitions and Theorems used to declare the problem
noncomputable def quadrant (z : ℂ) : string :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "boundary"

theorem complex_quadrant_condition (z : ℂ) (h : z / (z - complex.i) = complex.i) : 
  quadrant z = "first quadrant" := by
sory -- proof is omitted

end complex_quadrant_condition_l154_154422


namespace volume_ratio_of_cubes_l154_154596

theorem volume_ratio_of_cubes 
  (P_A P_B : ℕ) 
  (h_A : P_A = 40) 
  (h_B : P_B = 64) : 
  (∃ s_A s_B V_A V_B, 
    s_A = P_A / 4 ∧ 
    s_B = P_B / 4 ∧ 
    V_A = s_A^3 ∧ 
    V_B = s_B^3 ∧ 
    (V_A : ℚ) / V_B = 125 / 512) := 
by
  sorry

end volume_ratio_of_cubes_l154_154596


namespace max_min_g_l154_154279

open Real

noncomputable def f (t : ℝ) : ℝ := (t^2 + 2 * t - 1) / (t^2 + 1)

noncomputable def g (x : ℝ) : ℝ := f(x) * f(1 - x)

theorem max_min_g :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → g(x) ≤ 1 / 25) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g(x) = 1 / 25) ∧
  (∀ x, -1 ≤ x ∧ x ≤ 1 → g(x) ≥ 4 - sqrt 34) ∧
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g(x) = 4 - sqrt 34) :=
by
  sorry

end max_min_g_l154_154279


namespace probability_p1n_sum_probabilities_l154_154235

theorem probability_p1n (n m : ℕ) (h1 : 4 ≤ n) (h2 : 2 ≤ m) (h3 : m ≤ n - 2) :
  P_{1n} = 4 / (m * (n - m)) :=
sorry

theorem sum_probabilities (n m : ℕ) (h1 : 4 ≤ n) (h2 : 2 ≤ m) (h3 : m ≤ n - 2) :
  ∑_{1 ≤ i < j ≤ n} P_{ij} = 6 :=
sorry

end probability_p1n_sum_probabilities_l154_154235


namespace smallest_integer_with_16_divisors_l154_154060

-- Define the condition for the number of divisors of an integer
def num_divisors (n : ℕ) : ℕ :=
  (n.factorization.map (λ p a, a + 1)).prod

-- Define the problem statement as a theorem
theorem smallest_integer_with_16_divisors : ∃ n : ℕ, num_divisors n = 16 ∧ (∀ m : ℕ, num_divisors m = 16 → n ≤ m) :=
by
  -- Placeholder to skip proof
  sorry

end smallest_integer_with_16_divisors_l154_154060


namespace Aimee_escalator_time_l154_154188

theorem Aimee_escalator_time (d : ℝ) (v_esc : ℝ) (v_walk : ℝ) :
  v_esc = d / 60 → v_walk = d / 90 → (d / (v_esc + v_walk)) = 36 :=
by
  intros h1 h2
  sorry

end Aimee_escalator_time_l154_154188


namespace same_heads_probability_l154_154747

theorem same_heads_probability : 
  let keiko_outcomes := [{hh := 1, ht := 1, th := 1, tt := 1}],
      ephraim_outcomes := [
        {hhh := 1, hht := 1, hth := 1, thh := 1, htt := 1, tht := 1, tth := 1, ttt := 1}
      ] in
  (probability_same_heads keiko_outcomes ephraim_outcomes = (1 / 4)) :=
sorry

end same_heads_probability_l154_154747


namespace regression_analysis_correct_statements_l154_154096

theorem regression_analysis_correct_statements :
  (∀ (r : ℝ), abs r ≤ 1 → abs r = 1 → ∀ (x y : ℤ), 
    has_stronger_linear_correlation x y) ∧
  (∀ (x y : ℤ) (b a : ℝ), linear_regression_line (λ x, b * x + a) (average x) (average y)) ∧
  (∀ (model : ℤ → ℝ), (sum_of_squared_residuals model < ε → 
    model_is_better_fit model)) ∧
  ¬ (∀ (X Y : Type) (obs_val_k : ℝ), categorical_variables_relationship X Y obs_val_k → 
    obs_val_k_indicates_stronger_relationship obs_val_k) :=
by sorry

end regression_analysis_correct_statements_l154_154096


namespace option_C_cannot_form_right_triangle_l154_154865

def is_right_triangle_sides (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem option_C_cannot_form_right_triangle :
  ¬ (is_right_triangle_sides 1.5 2 3) :=
by
  -- This is intentionally left incomplete as per instructions
  sorry

end option_C_cannot_form_right_triangle_l154_154865


namespace find_difference_l154_154795

theorem find_difference (L S : ℕ) (h1: L = 2 * S + 3) (h2: L + S = 27) (h3: L = 19) : L - 2 * S = 3 :=
by
  sorry

end find_difference_l154_154795


namespace path_is_epicycloid_l154_154882

-- Define the conditions given in the problem
variable (r : ℝ) -- radius of the smaller circle
variable (R : ℝ) -- radius of the larger circle
variable (fixed_point : ℝ × ℝ) -- fixed point on the circumference of the smaller circle

-- Assume the relationship between the radii
def two_times_r : Prop := R = 2 * r

-- State the problem of proving the path traced is an epicycloid
theorem path_is_epicycloid (h : two_times_r r R) : 
(proof (rolling_path_traced fixed_point r R)) :=
sorry

end path_is_epicycloid_l154_154882


namespace coin_flip_probability_l154_154018

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154018


namespace missing_digits_in_leontinas_anniversary_l154_154110

noncomputable def tens_digit : ℕ
noncomputable def units_digit : ℕ

axiom age_condition : (10 * tens_digit + units_digit) / 2 = 2 * (tens_digit + units_digit)

theorem missing_digits_in_leontinas_anniversary:
  tens_digit = 6 ∧ units_digit = 2 :=
by
  sorry

end missing_digits_in_leontinas_anniversary_l154_154110


namespace proof_equivalence_l154_154258

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables {α β γ δ : ℝ} -- angles are real numbers

-- Definition of cyclic quadrilateral
def cyclic_quadrilateral (α β γ δ : ℝ) : Prop :=
α + γ = 180 ∧ β + δ = 180

-- Definition of the problem statements
def statement1 (α γ : ℝ) : Prop :=
α = γ → α = 90

def statement3 (α γ : ℝ) : Prop :=
180 - α + 180 - γ = 180

def statement2 (α β : ℝ) (ψ χ : ℝ) : Prop := 
α = β → cyclic_quadrilateral α β ψ χ → ψ = χ ∨ (α = β ∧ α = ψ ∧ α = χ)

def statement4 (α β γ δ : ℝ) : Prop :=
1*α + 2*β + 3*γ + 4*δ = 360

-- Theorem statement
theorem proof_equivalence (α β γ δ : ℝ) :
  cyclic_quadrilateral α β γ δ →
  (statement1 α γ) ∧ (statement3 α γ) ∧ ¬(statement2 α β γ δ) ∧ ¬(statement4 α β γ δ) :=
by
  sorry

end proof_equivalence_l154_154258


namespace intersection_of_sets_example_l154_154356

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l154_154356


namespace isla_field_is_larger_by_425_square_feet_l154_154414

theorem isla_field_is_larger_by_425_square_feet:
  let area_stan := 30 * 50 in
  let area_isla := 35 * 55 in
  area_isla - area_stan = 425 :=
by
  sorry

end isla_field_is_larger_by_425_square_feet_l154_154414


namespace distance_between_cars_l154_154313

theorem distance_between_cars (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) :
  t = 1 ∧ v_kmh = 180 ∧ v_ms = v_kmh * 1000 / 3600 → 
  v_ms * t = 50 := 
by 
  sorry

end distance_between_cars_l154_154313


namespace range_of_m_l154_154381

noncomputable def f : ℝ → ℝ
| x => if x >= 0 then (1/3)^(-x) - 2 else 2 * Real.log (-x) / Real.log 3

theorem range_of_m (m : ℝ) : f m > 1 ↔ m ∈ set.Ioo (-∞) (-Real.sqrt 3) ∪ set.Ioo 1 ∞ :=
sorry

end range_of_m_l154_154381


namespace number_of_true_propositions_is_2_l154_154174

-- Definitions for the propositions
def original_proposition (x : ℝ) : Prop := x > -3 → x > -6
def converse_proposition (x : ℝ) : Prop := x > -6 → x > -3
def inverse_proposition (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive_proposition (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The theorem we need to prove
theorem number_of_true_propositions_is_2 :
  (∀ x, original_proposition x) ∧ (∀ x, contrapositive_proposition x) ∧ 
  ¬ (∀ x, converse_proposition x) ∧ ¬ (∀ x, inverse_proposition x) → 2 = 2 := 
sorry

end number_of_true_propositions_is_2_l154_154174


namespace tan_double_angle_l154_154225

theorem tan_double_angle (α : ℝ) (h : sin α + √3 * cos α = 0) : tan (2 * α) = √3 := by
  sorry

end tan_double_angle_l154_154225


namespace vertex_x_coord_func_value_at_neg_p_a_vertex_a_div_2_x_coord_l154_154427

noncomputable def quadratic_func (a p q : ℝ) (ha : 0 < a) (hp : 0 < p) : ℝ → ℝ :=
  λ x, a * x^2 + p * x + q

theorem vertex_x_coord (a p q : ℝ) (ha : 0 < a) (hp : 0 < p) :
  - p / (2 * a) = argmin (quadratic_func a p q ha hp) :=
sorry

theorem func_value_at_neg_p_a (a p q : ℝ) (ha : 0 < a) (hp : 0 < p) :
  quadratic_func a p q ha hp (- p / a) = q :=
sorry

theorem vertex_a_div_2_x_coord (a p q : ℝ) (ha : 0 < a) (hp : 0 < p) :
  - p / a = argmin (quadratic_func (a / 2) p q (by linarith [ha]) hp) :=
sorry

end vertex_x_coord_func_value_at_neg_p_a_vertex_a_div_2_x_coord_l154_154427


namespace abs_difference_l154_154940

noncomputable def A : ℕ :=
  (List.range' 1 40).map (λ n => if n % 2 = 1 then n * (n + 1) else n).sum + 41

noncomputable def B : ℕ :=
  (List.range' 1 40).map (λ n => if n % 2 = 0 then n * (n + 1) else n).sum + 1 * 42

theorem abs_difference : abs (A - B) = 800 := by
  sorry

end abs_difference_l154_154940


namespace constant_term_expansion_l154_154725

theorem constant_term_expansion :
  let general_term (r : ℕ) := (binom 6 r) * (2 ^ r) * (x ^ (6 - (3 * r) / 2))
  let constant_term_in_expansion := general_term 4
  constant_term_in_expansion = 240 := by
  sorry

end constant_term_expansion_l154_154725


namespace carol_rectangle_length_l154_154588

theorem carol_rectangle_length :
  ∃ (L : ℝ), ∃ (width_carol_jordan : ℝ),
  let area_jordan := 8 * 45 in
  let area_carol := L * width_carol_jordan in
  area_carol = area_jordan ∧ width_carol_jordan = 24 ∧ width_carol_jordan = 24 → L = 15 :=
by
  sorry

end carol_rectangle_length_l154_154588


namespace intersection_of_M_and_N_l154_154351

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l154_154351


namespace jason_tip_correct_l154_154741

def jason_tip : ℝ := 
  let c := 15.00
  let t := 0.20
  let a := 20.00
  let total := c + t * c
  a - total

theorem jason_tip_correct : jason_tip = 2.00 := by
  unfold jason_tip
  -- Definitions from conditions
  let c := 15.00
  let t := 0.20
  let a := 20.00
  let total := c + t * c
  -- Calculation
  have h1 : t * c = 3.00 := by norm_num
  have h2 : total = c + 3.00 := by rw [h1]
  have h3 : total = 18.00 := by norm_num
  have h4 : a - total = 2.00 := by norm_num
  -- Result
  exact h4

end jason_tip_correct_l154_154741


namespace correct_option_C_l154_154147

theorem correct_option_C : 
  ∃ (A B C D : Prop), 
    (A ↔ (Real.sqrt 16 = 4 ∨ Real.sqrt 16 = -4)) ∧
    (B ↔ (4 = Real.sqrt 16 ∨ -4 = Real.sqrt 16)) ∧
    (C ↔ Real.cbrt (-27) = -3) ∧
    (D ↔ Real.sqrt ((-4)^2) = -4) ∧
    C :=
by
  let A := (Real.sqrt 16 = 4 ∨ Real.sqrt 16 = -4)
  let B := (4 = Real.sqrt 16 ∨ -4 = Real.sqrt 16)
  let C := Real.cbrt (-27) = -3
  let D := Real.sqrt ((-4)^2) = -4
  have hA : ¬A := by sorry
  have hB : ¬B := by sorry
  have hC : C := by sorry
  have hD : ¬D := by sorry
  exact ⟨A, B, C, D, ⟨hA, hB, hC, hD, hC⟩⟩

end correct_option_C_l154_154147


namespace geometric_sequence_product_l154_154226

variable {a : ℕ → ℝ}
variable {r : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_product 
  (h : is_geometric_sequence a r)
  (h_cond : a 4 * a 6 = 10) :
  a 2 * a 8 = 10 := 
sorry

end geometric_sequence_product_l154_154226


namespace intersection_of_sets_example_l154_154357

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l154_154357


namespace non_decreasing_function_evaluation_l154_154653

-- Definition of non-decreasing function on [0,1]
def non_decreasing (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → x₁ < x₂ → f x₁ ≤ f x₂

-- Given conditions and the final result as a theorem statement
theorem non_decreasing_function_evaluation {f : ℝ → ℝ} :
  (non_decreasing f {x | 0 ≤ x ∧ x ≤ 1}) →
  (f 0 = 0) →
  (∀ x ∈ {x | 0 ≤ x ∧ x ≤ 1}, f (x / 3) = 1/2 * f x) →
  (∀ x ∈ {x | 0 ≤ x ∧ x ≤ 1}, f (1 - x) = 1 - f x) →
  f 1 + f (1/2) + f (1/3) + f (1/6) + f (1/7) + f (1/8) = 11/4 :=
by
  -- Proof goes here
  sorry

end non_decreasing_function_evaluation_l154_154653


namespace magnitude_of_z_l154_154756

theorem magnitude_of_z : 
  let i := complex.I in
  let z := 2 * i - (5 / (2 - i)) in
  complex.abs z = real.sqrt 5 :=
by
  let i := complex.I
  let z := 2 * i - (5 / (2 - i))
  have hz : z = -2 + i := sorry
  have abs_z : complex.abs z = complex.abs (-2 + i) := by rw hz
  rw [complex.abs, complex.abs_mk]
  have h_sqrt : real.sqrt ((-2)^2 + 1^2) = real.sqrt 5 := by norm_num
  rw h_sqrt
  triv리
  
end magnitude_of_z_l154_154756


namespace distances_diff_less_than_ten_cm_l154_154337

noncomputable def sum_distances_to_vertices (P : Point) (vertices : Fin 12 → Point) : ℝ :=
  ∑ i : Fin 12, distance P (vertices i)

theorem distances_diff_less_than_ten_cm
  (dodecagon : ConvexPolygon 12)
  (O O' : Point)
  (h_dist : distance O O' = 10) :
  |sum_distances_to_vertices O dodecagon.vertices - sum_distances_to_vertices O' dodecagon.vertices| < 10 :=
sorry

end distances_diff_less_than_ten_cm_l154_154337


namespace number_of_pairs_lcm_2000_l154_154689

theorem number_of_pairs_lcm_2000 :
  let n := 2000
  in ∃ (pairs : List (ℕ × ℕ)), (∀ p ∈ pairs, p.1 > 0 ∧ p.2 > 0 ∧ Nat.lcm p.1 p.2 = n) ∧ pairs.length = 32 :=
by
  let n := 2000
  sorry

end number_of_pairs_lcm_2000_l154_154689


namespace find_g_three_l154_154430

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_three (h : ∀ x : ℝ, g (3^x) + (x + 1) * g (3^(-x)) = 3) : g 3 = -3 :=
sorry

end find_g_three_l154_154430


namespace find_f_two_l154_154270

noncomputable def f (x : ℝ) : ℝ := sorry

lemma functional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  f(x) + f((x - 1) / x) = 1 + x := sorry

theorem find_f_two : f(2) = 1.5 := 
  sorry

end find_f_two_l154_154270


namespace compute_fraction_sum_l154_154754

theorem compute_fraction_sum
  (a b c : ℝ)
  (h : a^3 - 6 * a^2 + 11 * a = 12)
  (h : b^3 - 6 * b^2 + 11 * b = 12)
  (h : c^3 - 6 * c^2 + 11 * c = 12) :
  (ab : ℝ) / c + (bc : ℝ) / a + (ca : ℝ) / b = -23 / 12 := by
  sorry

end compute_fraction_sum_l154_154754


namespace relationship_between_line_and_circle_l154_154657

noncomputable def circle (r : ℝ) : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = r^2 }
def line (m b : ℝ) : set (ℝ × ℝ) := { p | p.2 = m * p.1 + b }

theorem relationship_between_line_and_circle (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) (l : set (ℝ × ℝ)) :
  (P ∈ l) → (dist P O = r) → (circle O r) ∩ l = {q | q = tangent O r l ∨ intersects O r l} :=
sorry

end relationship_between_line_and_circle_l154_154657


namespace quadratic_trinomials_coincide_l154_154330

variable {R : Type*} [LinearOrderedField R]
variable {f1 f2 : R → R}
variable {l1 l2 : AffineSubspace ℝ (R × R)}

-- Definitions of the quadratic trinomials
def is_quadratic_trinomial (f : R → R) : Prop :=
  ∃ a b c : R, ∀ x, f x = a*x^2 + b*x + c

-- Definition of parabolas related to the quadratic trinomials
def parabola (f : R → R) : Set (R × R) :=
  { p : R × R | ∃ x, p = (x, f x) }

-- Definitions representing the conditions of the problem
def segments_are_equal_on_line (f1 f2 : R → R) (l : AffineSubspace ℝ (R × R)) : Prop :=
  ∃ A1 B1 A2 B2 : R × R,
    A1 ∈ parabola f1 ∧ B1 ∈ parabola f1 ∧
    A2 ∈ parabola f2 ∧ B2 ∈ parabola f2 ∧
    A1 -ᵥ B1 = A2 -ᵥ B2 ∧
    A1 ∈ l ∧ B1 ∈ l ∧ A2 ∈ l ∧ B2 ∈ l

theorem quadratic_trinomials_coincide
  (h1 : is_quadratic_trinomial f1)
  (h2 : is_quadratic_trinomial f2)
  (hl1 : ∃ p1 p2 : R × R, p1 ≠ p2 ∧ p1 ∈ l1 ∧ p2 ∈ l1) -- l1 is a line
  (hl2 : ∃ p1 p2 : R × R, p1 ≠ p2 ∧ p1 ∈ l2 ∧ p2 ∈ l2) -- l2 is a line
  (hnp : ∃ p1 p2 : R × R, p1 ∈ l1 ∧ p2 ∈ l2 ∧ p1 ≠ p2) -- l1 and l2 are non-parallel
  (hs1 : segments_are_equal_on_line f1 f2 l1)
  (hs2 : segments_are_equal_on_line f1 f2 l2) :
  parabola f1 = parabola f2 := sorry

end quadratic_trinomials_coincide_l154_154330


namespace tan_of_angle_in_third_quadrant_l154_154257

theorem tan_of_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : α < -π / 2 ∧ α > -π) 
  (h2 : Real.sin α = -Real.sqrt 5 / 5) :
  Real.tan α = 1 / 2 := 
sorry

end tan_of_angle_in_third_quadrant_l154_154257


namespace median_of_six_data_points_l154_154830

theorem median_of_six_data_points :
  let data := [172, 169, 180, 182, 175, 176]
  let sorted_data := List.sort data
  let median := (sorted_data[2] + sorted_data[3]) / 2
  median = 175.5 :=
by
  let data := [172, 169, 180, 182, 175, 176]
  let sorted_data := List.sort data
  have h1 : sorted_data = [169, 172, 175, 176, 180, 182] := by sorry
  let median := (sorted_data[2] + sorted_data[3]) / 2
  have h2 : sorted_data[2] = 175 := by sorry
  have h3 : sorted_data[3] = 176 := by sorry
  have h4 : (175 + 176) / 2 = 175.5 := by sorry
  show 175.5 = 175.5 by sorry

end median_of_six_data_points_l154_154830


namespace parabola_equation_l154_154989

theorem parabola_equation (p : ℝ) (h0 : 0 < p)
  (h1 : ∃ (A B : ℝ × ℝ), ∃ (F : ℝ × ℝ), F.1 = p / 2 ∧ F.2 = 0 ∧ (A, B ∈ set_of (λ (x : ℝ × ℝ), x.2^2 = 2 * p * x.1)) ∧ 
         (A.1 - F.1, A.2 - F.2) = (3 * ((B.1 - F.1), (B.2 - F.2))) ∧ 
         dist ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (set_of (λ (x : ℝ × ℝ), x.1 = -p / 2)) = 16 / 3) : 
  y ^ 2 = 8 * x :=
sorry

end parabola_equation_l154_154989


namespace part1_part2_part3_l154_154545

-- Part 1
theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem part2 (a m n : ℤ) (h1 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) (h2 : 0 < a) (h3 : 0 < m) (h4 : 0 < n) : 
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem part3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end part1_part2_part3_l154_154545


namespace lines_intersection_l154_154041

theorem lines_intersection :
  ∃ (x y : ℝ), 
  (3 * y = -2 * x + 6) ∧ 
  (-4 * y = 3 * x + 4) ∧ 
  (x = -36) ∧ 
  (y = 26) :=
sorry

end lines_intersection_l154_154041


namespace range_of_solutions_l154_154577

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l154_154577


namespace max_value_of_f_l154_154628

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∀ x : ℝ, x > 0 → f x ≤ (Real.log (Real.exp 1)) / (Real.exp 1) :=
by
  sorry

end max_value_of_f_l154_154628


namespace AC_equals_neg_three_halves_BC_l154_154638

-- Definitions of A, B, and C as points in Euclidean space
variables {V : Type*} [inner_product_space ℝ V]
variables (A B C : V)

-- Given conditions
def on_segment (A B C : V) : Prop := ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ C = A + t • (B - A)
axiom h_on_segment : on_segment A B C
axiom h_AC_eq : (C - A) = (3 / 5) • (B - A)

-- The theorem to be proven
theorem AC_equals_neg_three_halves_BC (h_on_segment : on_segment A B C) (h_AC_eq : (C - A) = (3 / 5) • (B - A)) :
  (C - A) = - (3 / 2) • (B - C) :=
sorry

end AC_equals_neg_three_halves_BC_l154_154638


namespace natasha_average_speed_l154_154870

theorem natasha_average_speed :
  (∀ t_up t_down avg_speed_total (dist : ℝ),
    t_up = 4 ∧ t_down = 2 ∧ avg_speed_total = 3 ∧
    dist = (avg_speed_total * (t_up + t_down) / 2) →
    dist / t_up = 2.25) :=
begin
  intros t_up t_down avg_speed_total dist,
  rintro ⟨h1, h2, h3, h4⟩,
  rw [h1, h2, h3, h4],
  simp,
  norm_num,
end

end natasha_average_speed_l154_154870


namespace smallest_integer_with_16_divisors_l154_154062

-- Define the condition for the number of divisors of an integer
def num_divisors (n : ℕ) : ℕ :=
  (n.factorization.map (λ p a, a + 1)).prod

-- Define the problem statement as a theorem
theorem smallest_integer_with_16_divisors : ∃ n : ℕ, num_divisors n = 16 ∧ (∀ m : ℕ, num_divisors m = 16 → n ≤ m) :=
by
  -- Placeholder to skip proof
  sorry

end smallest_integer_with_16_divisors_l154_154062


namespace solution_interval_l154_154574

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end solution_interval_l154_154574


namespace prime_divisors_of_exponential_sum_l154_154958

-- Definitions and conditions as stated in the problem
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def divides (a b : ℕ) := ∃ k : ℕ, b = a * k

def solution_set := {(2, 2), (2, 3), (3, 2)}

-- Main theorem statement
theorem prime_divisors_of_exponential_sum :
  ∀ (p q : ℕ), is_prime p → is_prime q → divides (p * q) (2^p + 2^q) ↔ (p, q) ∈ solution_set :=
begin
  sorry
end

end prime_divisors_of_exponential_sum_l154_154958


namespace marked_box_in_second_row_l154_154606

theorem marked_box_in_second_row:
  ∀ a b c d e f g h : ℕ, 
  (e = a + b) → 
  (f = b + c) →
  (g = c + d) →
  (h = a + 2 * b + c) →
  ((a = 5) ∧ (d = 6)) →
  ((a = 3) ∨ (b = 3) ∨ (c = 3) ∨ (d = 3)) →
  (f = 3) :=
by
  sorry

end marked_box_in_second_row_l154_154606


namespace bounded_area_l154_154190

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (x^2 + 1))^(1/3) + (x - Real.sqrt (x^2 + 1))^(1/3)

def g (y : ℝ) : ℝ := y + 1

theorem bounded_area : 
  (∫ y in (0:ℝ)..(1:ℝ), (g y - f (g y))) = (5/8 : ℝ) := by
  sorry

end bounded_area_l154_154190


namespace train_speed_is_30_kmh_l154_154504

noncomputable def speed_of_train (train_length : ℝ) (cross_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let train_speed_ms := relative_speed + man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_is_30_kmh :
  speed_of_train 400 59.99520038396929 6 = 30 :=
by
  -- Using the approximation mentioned in the solution, hence no computation proof required.
  sorry

end train_speed_is_30_kmh_l154_154504


namespace smallest_integer_with_16_divisors_l154_154058

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ :=
  n.factors.to_finset.prod fun p => (n.factorization p + 1)

-- Define the positive integer n which we need to prove has 16 divisors
def smallest_positive_integer_with_16_divisors (n : ℕ) : Prop :=
  num_divisors n = 16

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, smallest_positive_integer_with_16_divisors n ∧ ∀ m : ℕ, m < n → ¬smallest_positive_integer_with_16_divisors m :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154058


namespace scientific_notation_example_l154_154910

theorem scientific_notation_example :
  ∃ (a : ℝ) (b : ℤ), 1300000 = a * 10 ^ b ∧ a = 1.3 ∧ b = 6 :=
sorry

end scientific_notation_example_l154_154910


namespace remainder_degrees_l154_154862

theorem remainder_degrees (p q : Polynomial ℝ) (h_deg_q : degree q = 7) :
  ∃ r, degree r < 7 :=
sorry

end remainder_degrees_l154_154862


namespace one_thirds_of_nine_halfs_l154_154685

theorem one_thirds_of_nine_halfs : (9 / 2) / (1 / 3) = 27 / 2 := 
by sorry

end one_thirds_of_nine_halfs_l154_154685


namespace meal_combinations_l154_154866

theorem meal_combinations (MenuA_items : ℕ) (MenuB_items : ℕ) : MenuA_items = 15 ∧ MenuB_items = 12 → MenuA_items * MenuB_items = 180 :=
by
  sorry

end meal_combinations_l154_154866


namespace minimum_distance_to_C_prime_l154_154333

-- Definition of unit cube face
structure Point3D where
  x : ℝ 
  y : ℝ
  z : ℝ

-- Function to calculate the distance between two points in 3D space
def distance (p1 p2 : Point3D) : ℝ := 
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

-- Point M lies on the face AA'B'B of the unit cube
def is_on_face_AA_prime_B_prime_B (M : Point3D) : Prop := 
  (0 ≤ M.x ∧ M.x ≤ 1) ∧ (0 ≤ M.y ∧ M.y ≤ 1) ∧ (M.z = 0)

-- The distance from M to line AB is equal to the distance from M to line B'C'
def equidistant_from_AB_and_B_prime_C_prime (M : Point3D) (A B B_prime C_prime : Point3D) : Prop := 
  let line_AB := {p : Point3D | ∃ t : ℝ, p = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩}
  let line_B_prime_C_prime := {p : Point3D | ∃ t : ℝ, p = ⟨B_prime.x + t * (C_prime.x - B_prime.x), B_prime.y + t * (C_prime.y - B_prime.y), B_prime.z + t * (C_prime.z - B_prime.z)⟩}
  ∀ p1 p2 : Point3D, p1 ∈ line_AB → p2 ∈ line_B_prime_C_prime → distance M p1 = distance M p2

-- The minimum distance from a point on the trajectory of M to C' is sqrt(5)/2
theorem minimum_distance_to_C_prime :
  ∀ (A B C D A_prime B_prime C_prime D_prime M : Point3D),
  A = ⟨0, 0, 0⟩ →
  B = ⟨1, 0, 0⟩ →
  C = ⟨1, 1, 0⟩ →
  D = ⟨0, 1, 0⟩ →
  A_prime = ⟨0, 0, 1⟩ →
  B_prime = ⟨1, 0, 1⟩ →
  C_prime = ⟨1, 1, 1⟩ →
  D_prime = ⟨0, 1, 1⟩ →
  is_on_face_AA_prime_B_prime_B M →
  equidistant_from_AB_and_B_prime_C_prime M A B B_prime C_prime →
  distance M C_prime = Real.sqrt 5 / 2 := 
  sorry

end minimum_distance_to_C_prime_l154_154333


namespace seq_periodic_l154_154277

noncomputable def seq (a₁ : ℝ) : ℕ → ℝ
| 0     := a₁
| (n+1) := 1 / (1 - seq n)

theorem seq_periodic (a₁ := 1 / 2) : seq a₁ 2013 = a₁ :=
by sorry

end seq_periodic_l154_154277


namespace katie_total_earnings_l154_154746

-- Define the conditions
def bead_necklaces := 4
def gem_necklaces := 3
def price_per_necklace := 3

-- The total money earned
def total_money_earned := bead_necklaces + gem_necklaces * price_per_necklace = 21

-- The statement to prove
theorem katie_total_earnings : total_money_earned :=
by
  sorry

end katie_total_earnings_l154_154746


namespace incorrect_statement_b_l154_154097

def million := 1000000
def ten_thousand := 10000
def hundred := 100
def thousandth := 10^(-3)
def ten_million := 10000000

theorem incorrect_statement_b :
  ¬ (2.13 * 10^3).round = (2.13 * 10^3 / hundred).round * hundred → 
  (2.13 * 10^3).round ≠ 2130 :=
by
  sorry

end incorrect_statement_b_l154_154097


namespace train_length_is_360_l154_154533

-- Definitions based on conditions
def train_speed_kmph := 45 -- speed of the train in km/hr
def platform_length := 180 -- length of the platform in meters
def passing_time := 43.2 -- time taken to pass the platform in seconds

-- Conversion factor
def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Equivalent Proof Statement
theorem train_length_is_360 :
  let train_speed_mps := kmph_to_mps train_speed_kmph in
  let total_distance := train_speed_mps * passing_time in
  let train_length := total_distance - platform_length in
  train_length = 360 := by
begin
  --lean code for the solution
  sorry
end

end train_length_is_360_l154_154533


namespace probability_heads_at_least_9_of_12_flips_l154_154009

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l154_154009


namespace distinct_values_f_x_l154_154763

open Int

def f (x : ℝ) : ℤ :=
  ∑ k in Finset.range 11 \ Finset.singleton 0, (Int.floor (k * x) - k * Int.floor x)

def euler_totient (n : ℕ) : ℕ := λ n, (Finset.range n).filter (nat.coprime n).card

noncomputable def distinct_values_count (k_max : ℕ) : ℕ :=
  (Finset.range (k_max + 1)).erase 1 |>.sum euler_totient

theorem distinct_values_f_x : distinct_values_count 12 + 1 = 46 := by
  sorry

end distinct_values_f_x_l154_154763


namespace area_AOC_eq_3a_l154_154377

-- Define points X, Y, and Z on sides BC, AC, and AB of triangle ABC
variables {A B C O X Y Z : Type*} [Simplex A B C] [Segment B C X] [Segment A C Y] [Segment A B Z]

-- Define the given ratios BX : XC = 2 : 3 and CY : YA = 1 : 2
variables (hx : ratio (B, X, C) = 2 / 5) (hy : ratio (C, Y, A) = 1 / 3)

-- Define the given condition that AX, BY, and CZ concur at O
variable (hconcur : concurrent A X B Y C Z O)

-- Define the area of triangle BOC
parameter (a : ℝ)
variable (harea : area B O C = a)

-- Prove the area of triangle AOC is 3a
theorem area_AOC_eq_3a : area A O C = 3 * a :=
sorry

end area_AOC_eq_3a_l154_154377


namespace find_fraction_l154_154753

noncomputable def distinct_real_numbers (a b : ℝ) : Prop :=
  a ≠ b

noncomputable def equation_condition (a b : ℝ) : Prop :=
  (2 * a / (3 * b)) + ((a + 12 * b) / (3 * b + 12 * a)) = (5 / 3)

theorem find_fraction (a b : ℝ) (h1 : distinct_real_numbers a b) (h2 : equation_condition a b) : a / b = -93 / 49 :=
by
  sorry

end find_fraction_l154_154753


namespace samantha_routes_eq_320_l154_154800

-- Define Samantha's starting point and destinations
def starting_point := (3, -2)   -- (west, south from the southwest corner of City Park)
def school := (3, 3)            -- (east, north from the northeast corner of City Park)
def library := (1, 1)           -- (east, north of the northeast corner of City Park)

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Total number of routes
def total_routes : ℕ :=
  let to_park := binomial 5 2 in
  let across_park := 1 in
  let without_library := binomial 6 3 in
  let to_library := binomial 2 1 in
  let from_library_to_school := binomial 4 2 in
  let with_library := to_library * from_library_to_school in
  (to_park * across_park * without_library) + (to_park * across_park * with_library)

-- Theorem to check the total number of routes
theorem samantha_routes_eq_320 : total_routes = 320 :=
by
  sorry

end samantha_routes_eq_320_l154_154800


namespace greatest_b_value_l154_154195

theorem greatest_b_value : ∃ b, (b^2 - 14 * b + 45 ≤ 0) ∧ (∀ c, (c^2 - 14 * c + 45 ≤ 0) → c ≤ b) ∧ b = 9 :=
by
  use 9
  split
  sorry
  split
  sorry
  rfl

end greatest_b_value_l154_154195


namespace find_A_find_bc_l154_154646

variables (a b c A B C : Real)

-- Given condition for part 1
axiom cos_condition : cos B * cos C - sin B * sin C = -1 / 2

-- Prove A = π / 3
theorem find_A (h : cos B * cos C - sin B * sin C = -1 / 2) : A = π / 3 :=
sorry

-- Given conditions for part 2
axiom side_a : a = 2
axiom area_condition : (1 / 2) * b * c * sin A = √3

-- Prove b = 2 and c = 2
theorem find_bc (ha : a = 2) (h_area : (1 / 2) * b * c * sin A = √3) : b = 2 ∧ c = 2 :=
sorry

end find_A_find_bc_l154_154646


namespace max_value_sin_expression_l154_154999

def max_of_sin_expression (x y z : ℝ) : ℝ :=
  Real.sin (x - y) + Real.sin (y - z) + Real.sin (z - x)

theorem max_value_sin_expression : ∀ (x y z : ℝ),
  (0 ≤ x ∧ x ≤ Real.pi) →
  (0 ≤ y ∧ y ≤ Real.pi) →
  (0 ≤ z ∧ z ≤ Real.pi) →
  (max_of_sin_expression x y z ≤ 2) ∧ (∃ x y z : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ 0 ≤ y ∧ y ≤ Real.pi ∧ 0 ≤ z ∧ z ≤ Real.pi ∧ max_of_sin_expression x y z = 2) :=
by
  sorry

end max_value_sin_expression_l154_154999


namespace find_value_of_a4_plus_a5_l154_154366

variables {S_n : ℕ → ℕ} {a_n : ℕ → ℕ} {d : ℤ} 

-- Conditions
def arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (d : ℤ) : Prop :=
∀ n : ℕ, S_n n = n * a_n 1 + (n * (n - 1) / 2) * d

def a_3_S_3_condition (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) : Prop := 
a_n 3 = 3 ∧ S_n 3 = 3

-- Question
theorem find_value_of_a4_plus_a5 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℤ):
  arithmetic_sequence_sum S_n a_n d →
  a_3_S_3_condition a_n S_n →
  a_n 4 + a_n 5 = 12 :=
by
  sorry

end find_value_of_a4_plus_a5_l154_154366


namespace quadratic_equation_real_roots_k_value_l154_154704

theorem quadratic_equation_real_roots_k_value :
  (∀ k : ℕ, (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) <-> k = 1) :=
by
  sorry
  
end quadratic_equation_real_roots_k_value_l154_154704


namespace new_refrigerator_cost_l154_154748

theorem new_refrigerator_cost :
  ∀ (x : ℝ),
  (∀ (old_cost : ℝ) (savings : ℝ) (days : ℝ),
    old_cost = 0.85 →
    savings = 12 →
    days = 30 →
    savings = days * (old_cost - x)
  ) → x = 0.45 :=
by
  intros x h
  apply h
  repeat { sorry }

end new_refrigerator_cost_l154_154748


namespace perpendicular_lines_slope_l154_154302

-- Definitions
def line1_equation (m : ℝ) : ℝ → ℝ := λ x, m * x + 1
def line2_equation : ℝ → ℝ := λ x, 4 * x - 8

def is_perpendicular (slope1 slope2 : ℝ) : Prop := slope1 * slope2 = -1

-- Proof statement
theorem perpendicular_lines_slope (m : ℝ) (h : is_perpendicular m 4) : m = -1 / 4 := by
  sorry

end perpendicular_lines_slope_l154_154302


namespace hexagon_perimeter_sum_l154_154935

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def perimeter : ℝ := 
  distance 0 0 1 2 +
  distance 1 2 3 3 +
  distance 3 3 5 3 +
  distance 5 3 6 1 +
  distance 6 1 4 (-1) +
  distance 4 (-1) 0 0

theorem hexagon_perimeter_sum :
  perimeter = 3 * Real.sqrt 5 + 2 + 2 * Real.sqrt 2 + Real.sqrt 17 := 
sorry

end hexagon_perimeter_sum_l154_154935


namespace largest_among_four_numbers_l154_154548

theorem largest_among_four_numbers :
  ∀ (x ∈ {-1.5, -3, -1, -5}), -1 ≥ x :=
by sorry

end largest_among_four_numbers_l154_154548


namespace dot_product_range_l154_154292

theorem dot_product_range (c d : ℝ^3) (φ : ℝ) (h_norm_c : ∥c∥ = 5) (h_norm_d : ∥d∥ = 13) (h_angle : 0 ≤ φ ∧ φ ≤ π / 3) :
  32.5 ≤ c.dot d ∧ c.dot d ≤ 65 :=
by sorry

end dot_product_range_l154_154292


namespace num_revolutions_proof_num_revolutions_special_case_l154_154493

-- Let n be the number of coins forming a convex polygon.
-- r is the radius of the rolling coin.
-- k is the ratio such that rolling coin's radius is k times the radius of each coin in the chain.

def num_revolutions (n : ℕ) (k : ℝ) : ℝ :=
  (k + 1) / (2 * k) * (n - (2 / π) * n * real.arccos (1 / (k + 1)) + 2)

theorem num_revolutions_proof (n : ℕ) (k : ℝ) (h_pos : 0 < k): num_revolutions n k = 
  (k + 1) / (2 * k) * (n - (2 / π) * n * real.arccos (1 / (k + 1)) + 2) :=
by 
  sorry

theorem num_revolutions_special_case (n : ℕ): num_revolutions n 1 = 
  (n + 2) / 3 + 2 :=
by 
  sorry

end num_revolutions_proof_num_revolutions_special_case_l154_154493


namespace complement_of_A_l154_154261

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

theorem complement_of_A :
  U \ A = {x | -3 < x ∧ x < 0} :=
sorry

end complement_of_A_l154_154261


namespace rectangle_area_l154_154713

theorem rectangle_area (a : ℕ) (h : 2 * (3 * a + 2 * a) = 160) : 3 * a * 2 * a = 1536 :=
by
  sorry

end rectangle_area_l154_154713


namespace sum_of_rational_roots_of_f_eq_zero_l154_154975

def f (x : ℚ) : ℚ := x^3 - 9*x^2 + 27*x - 8

theorem sum_of_rational_roots_of_f_eq_zero : 
  (∑ root in {x : ℚ | f x = 0}.to_finset, root) = 0 :=
by
  sorry

end sum_of_rational_roots_of_f_eq_zero_l154_154975


namespace intersection_M_N_l154_154348

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l154_154348


namespace intersection_in_quadrants_I_and_II_l154_154445

open Set

def in_quadrants_I_and_II (x y : ℝ) : Prop :=
  (0 ≤ x ∧ 0 ≤ y) ∨ (x ≤ 0 ∧ 0 ≤ y)

theorem intersection_in_quadrants_I_and_II :
  ∀ (x y : ℝ),
    y > 3 * x → y > -2 * x + 3 → in_quadrants_I_and_II x y :=
by
  intros x y h1 h2
  sorry

end intersection_in_quadrants_I_and_II_l154_154445


namespace polynomial_roots_are_minus_one_l154_154438

noncomputable def polynomial_roots (n : ℕ) (a_2 : ℝ) (a_1 ... a_0 : ℝ) : List ℝ :=
  let r := -1
  List.repeat r n

theorem polynomial_roots_are_minus_one (n : ℕ) (a_2 : ℝ) (a_1 ... a_0 : ℝ) :
  (∀ x, polynomial.eval x (polynomial_roots n a_2 a_1 ... a_0 x) = x^n + n * x^(n-1) + a_2 * x^(n-2) + ... + a_0) ↔
  (∀ r, r ∈ polynomial_roots n a_2 a_1 ... a_0 → r = -1) 
  :=
by
  sorry

end polynomial_roots_are_minus_one_l154_154438


namespace part1_part2_l154_154268

def f (x : ℝ) : ℝ := |x - 1|

theorem part1 (x : ℝ) : f(2*x) + f(x + 4) ≥ 8 ↔ (x ≤ -10/3 ∨ x ≥ 2) := sorry

theorem part2 (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (h0 : a ≠ 0) : 
  f(ab) / |a| > f(b / a) := sorry

end part1_part2_l154_154268


namespace tiled_rectangle_area_eq_l154_154814

theorem tiled_rectangle_area_eq (n : ℕ) :
  (∑ k in range 8, k * (k + 1)) = n * (n + 1) → n = 15 :=
by
  sorry

end tiled_rectangle_area_eq_l154_154814


namespace relationship_between_f_b_2_and_f_a_1_l154_154240

noncomputable def f (a b x : ℝ) : ℝ := log a (abs (x + b))

theorem relationship_between_f_b_2_and_f_a_1 (a : ℝ) (b : ℝ) : 
  (0 < a ∧ a < 1) ∧ f a b (-2) < f a b (a + 1) :=
by
  sorry

end relationship_between_f_b_2_and_f_a_1_l154_154240


namespace show_equal_ED_EC_l154_154369

noncomputable def circles_intersect (Γ1 Γ2 : Circle) : Prop :=
  ∃ M N : Point, M ≠ N ∧ M ∈ Γ1 ∧ M ∈ Γ2 ∧ N ∈ Γ1 ∧ N ∈ Γ2

noncomputable def tangent_intersects (Δ : Line) (Γ : Circle) : Prop :=
  ∃ P : Point, P ∈ Δ ∧ P ∈ Γ

noncomputable def parallel_to_tangent (l Δ : Line) (M : Point) : Prop :=
  l || Δ ∧ M ∈ l

noncomputable def second_intersection (l : Line) (Γ1 Γ2 : Circle) (M : Point) : Prop :=
  ∃ A B : Point, A ≠ M ∧ B ≠ M ∧ A ∈ l ∧ A ∈ Γ1 ∧ B ∈ l ∧ B ∈ Γ2

noncomputable def intersections {α : Type*} [LinearOrder α] (l : Line) (p q : Point) : Prop :=
  ∃ C D : Point, C ≠ D ∧ C ∈ l ∧ liesOnLine (mk_segment p q) C ∧ 
                          D ∈ l ∧ liesOnLine (mk_segment q p) D

noncomputable def intersection_point (AP PQ : Line) : Prop :=
  ∃ E : Point, E ∈ AP ∧ E ∈ PQ

theorem show_equal_ED_EC
  (Γ1 Γ2 : Circle) (M N : Point) (Δ : Line) (P Q A B C D E : Point) (l : Line)
  (h_intersect : circles_intersect Γ1 Γ2)
  (h_tangent_Γ1 : tangent_intersects Δ Γ1)
  (h_tangent_Γ2 : tangent_intersects Δ Γ2)
  (h_parallel : parallel_to_tangent l Δ M)
  (h_second_intersection : second_intersection l Γ1 Γ2 M)
  (h_intersections : intersections l P N ∧ intersections l Q N)
  (h_AP_PQ : intersection_point (mk_line A P) (mk_line P Q)) :
  dist E D = dist E C := 
by sorry

end show_equal_ED_EC_l154_154369


namespace inscribed_circle_diameter_l154_154854

theorem inscribed_circle_diameter (DE DF EF : ℝ) (hDE : DE = 10) (hDF : DF = 5) (hEF : EF = 9) : 
  let s := (DE + DF + EF) / 2 in
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  let r := K / s in
  let d := 2 * r in
  d = Real.sqrt 14 := by
  sorry

end inscribed_circle_diameter_l154_154854


namespace solve_for_x_l154_154691

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - 2 * y = 8) (h2 : x + 3 * y = 7) : x = 38 / 11 :=
by
  sorry

end solve_for_x_l154_154691


namespace proof_ellipse_equation_existence_lambda_l154_154992

open Real

noncomputable def ellipse := 
  {a b : ℝ // a = 2 ∧ b = sqrt 3 ∧ (∀ x y, (x/a)^2 + (y/b)^2 = 1) ∧ a > 0 ∧ b > 0}

theorem proof_ellipse_equation (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (R : ℝ × ℝ) (M N G : ℝ × ℝ) :
  P = (2 * sqrt 6 / 3, -1) →
  (F1, F2).1 = (0, a) →
  (F1, F2).2 = (0, -a) →
  abs (dist P F1 + dist P F2) = 4 →
  R = (4, 0) →
  M ∈ ({ x y | (x/a)^2 + (y/b)^2 = 1 }) →
  N ∈ ({ x y | (x/a)^2 + (y/b)^2 = 1 }) →
  G ∈ ({ x (-y) | (x/a)^2 + (y/b)^2 = 1 }) →
  by let equation_C := ellipse; 
     sorry

theorem existence_lambda (M N G F2 : ℝ × ℝ) :
  ∃ λ : ℝ, dist G F2 = λ * dist F2 N :=
sorry

end proof_ellipse_equation_existence_lambda_l154_154992


namespace calculate_principal_l154_154491

-- Principal calculation under given conditions
theorem calculate_principal (r t A : ℝ) (h_r : r = 0.05) (h_t : t = 2 + 2/5) (h_A : A = 1232) : 
  let P := A / (1 + r * t) in 
  P = 1100 :=
by 
suffices P_def : P = A / (1 + r * t), from
  calc P = A / (1 + r * t) : by exact P_def
      ... = 1100 : by norm_num [h_r, h_t, h_A]
sorry

end calculate_principal_l154_154491


namespace wuyang_airlines_min_flights_l154_154099

theorem wuyang_airlines_min_flights :
  ∃ (n : ℕ), (WuyangAirlines_flights needs_opening) → n = 6 → (∀ S ⊆ (finset.range 6), S.card = 3 → ∃ (a b : ℕ), a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ connected a b) → n = 15 :=
by
  sorry

end wuyang_airlines_min_flights_l154_154099


namespace equilateral_triangle_on_parabola_length_l154_154234

-- Definitions for the conditions
def point_on_curve (x y : ℝ) := y^2 = 4 * x

def circle_passes_through_point (x y : ℝ) :=
  ∃ (r : ℝ), (x - 1)^2 + y^2 = r^2

def circle_tangent_to_line (r : ℝ) := r = 2

-- Statement of the proof problem
theorem equilateral_triangle_on_parabola_length :
  (∀ (B C : ℝ × ℝ), point_on_curve B.1 B.2 ∧ point_on_curve C.1 C.2 ∧
  ∃ (lengths : ℝ),
    let A := (4, 0) in 
    (∃ (triangle : set (ℝ × ℝ)), triangle = {A, B, C} ∧ 
    is_equilateral_triangle triangle ∧ lengths ∈ {4 * (real.sqrt 3 + real.sqrt 7), 
                                                  4 * (real.sqrt 7 - real.sqrt 3), 
                                                  (8 * real.sqrt 2) / 3})) :=
sorry

end equilateral_triangle_on_parabola_length_l154_154234


namespace increased_area_l154_154827

variable (r : ℝ)

theorem increased_area (r : ℝ) : 
  let initial_area : ℝ := π * r^2
  let final_area : ℝ := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π := by
sorry

end increased_area_l154_154827


namespace sin_cos_identity_l154_154626

theorem sin_cos_identity (α : ℝ) (h : sin α - 3 * cos α = 0) : sin α ^ 2 + sin α * cos α = 6 / 5 := 
by
  sorry

end sin_cos_identity_l154_154626


namespace smallest_integer_with_16_divisors_l154_154079

theorem smallest_integer_with_16_divisors : 
  ∃ (n : ℕ), (∃ (p_1 p_2 p_3 : ℕ) (a_1 a_2 a_3 : ℕ), 
  (p_1 = 2 ∧ p_2 = 3 ∧ a_1 = 3 ∧ a_2 = 3 ∧ n = p_1 ^ a_1 * p_2 ^ a_2) ∧
  (∀ m, m > 0 → (∃ b1 b2 ..., m has exactly 16 positive divisors) → 216 ≤ m)) := 
sorry

end smallest_integer_with_16_divisors_l154_154079


namespace raft_travel_time_l154_154877

variable (v_b v_c D : ℝ)

def downstream_time : ℝ := 8
def upstream_time : ℝ := 10

axiom downstream_condition : D = downstream_time * (v_b + v_c)
axiom upstream_condition : D = upstream_time * (v_b - v_c)

theorem raft_travel_time : ∀ v_b v_c D, downstream_condition v_b v_c D → upstream_condition v_b v_c D → D / v_c = 80 := 
by 
  intros
  sorry

end raft_travel_time_l154_154877


namespace sum_of_areas_of_DFG_and_AGE_l154_154921

-- Definitions based on conditions
def point := ℝ × ℝ
def line (p1 p2 : point) : Set point := {q | ∃ t : ℝ, q = (p1.1 + t * (p2.1 - p1.1), p1.2 + t * (p2.2 - p1.2))}

-- Given points and segments
noncomputable def A : point := (0, 0)
noncomputable def B : point := (0, 3)
noncomputable def C : point := (6, 0)
noncomputable def D : point := (11, 0)
noncomputable def E : point := (B.1 + 3, B.2) -- midpoint (0, 3) to (3, 0)
noncomputable def F : point := (C.1 - 3, C.2)
noncomputable def G : point := line A F ∩ line D E

theorem sum_of_areas_of_DFG_and_AGE :
  let DEF_area := (|D.1 - F.1| * |D.2 - F.2|) / 2;
  let AEG_area := (|A.1 - E.1| * |A.2 - E.2|) / 2;
  DEF_area + AEG_area = 49 / 8 := 
by sorry

end sum_of_areas_of_DFG_and_AGE_l154_154921


namespace train_speed_30_kmph_l154_154536

def train_speed (time_to_cross: ℝ) (length_of_train: ℝ) : ℝ :=
  (length_of_train / time_to_cross) * (3600 / 1000)

theorem train_speed_30_kmph : 
  train_speed 9 75 = 30 :=
by 
  sorry

end train_speed_30_kmph_l154_154536


namespace solve_for_x_l154_154640

theorem solve_for_x (x : ℝ) (h : (x^2 + 4*x - 5)^0 = 1) : x^2 - 5*x + 5 = 1 → x = 4 := 
by
  intro h2
  have : ∀ x, (x^2 + 4*x - 5 = 0) ↔ false := sorry
  exact sorry

end solve_for_x_l154_154640


namespace smallest_integer_with_16_divisors_l154_154053

-- Define prime factorization and the function to count divisors
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def number_of_divisors (n : ℕ) : ℕ :=
  (prime_factorization n).foldr (λ ⟨_, a⟩ acc, acc * (a + 1)) 1

-- Main theorem stating the smallest positive integer with exactly 16 divisors is 210
theorem smallest_integer_with_16_divisors : 
  ∃ n, n > 0 ∧ number_of_divisors n = 16 ∧ ∀ m, m > 0 ∧ number_of_divisors m = 16 → n ≤ m :=
begin
  use 210,
  split,
  { -- Prove 210 > 0
    exact nat.zero_lt_succ _,
  },
  split,
  { -- Prove number_of_divisors 210 = 16
    sorry,
  },
  { -- Prove minimality
    intros m hm1 hm2,
    sorry,
  }
end

end smallest_integer_with_16_divisors_l154_154053


namespace tangent_line_at_zero_range_of_a_l154_154661

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

theorem tangent_line_at_zero (h : ∀ x, f 1 x = Real.exp x - Real.sin x - 1) :
  ∀ x, Real.exp x - Real.sin x - 1 = f 1 x :=
by
  sorry

theorem range_of_a (h : ∀ x, f a x ≥ 0) : a ∈ Set.Iic 1 :=
by
  sorry

end tangent_line_at_zero_range_of_a_l154_154661


namespace p_evaluation_l154_154766

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else 2 * x + 2 * y

theorem p_evaluation : p (p 3 (-4)) (p (-7) 0) = 40 := by
  sorry

end p_evaluation_l154_154766


namespace find_AC_BC_l154_154758

-- Defining the problem with the given conditions and the statement to prove
noncomputable def triangle_lengths (A B C D : Point) (length_AB : ℝ) (incenter_centroid_property : (incenter (triangle B C D) = centroid (triangle A B C))) : ℝ :=
√(5 / 2)

theorem find_AC_BC (A B C D : Point) (length_AB : ℝ) (incenter_centroid_property : (incenter (triangle B C D) = centroid (triangle A B C)))
  (AB_eq : length_AB = 1) 
  : length (segment A C) = √(5 / 2) ∧ length (segment B C) = √(5 / 2) :=
begin
  sorry
end

end find_AC_BC_l154_154758


namespace units_digit_Of_FF15_l154_154415

noncomputable def Fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := Fibonacci(n+1) + Fibonacci(n)

theorem units_digit_Of_FF15 : (Fibonacci (Fibonacci 15)) % 10 = 5 :=
by sorry

end units_digit_Of_FF15_l154_154415


namespace calc_moles_CaCO3_formed_l154_154167

constant CaOH2 CO2 H2SO4 CaCO3 H2O CaSO4 : Type
constant primary_reaction : (CaOH2 × CO2) → (CaCO3 × H2O)
constant side_reaction : (CaOH2 × H2SO4) → (CaSO4 × H2O)

constant moles_CaOH2 : ℕ
constant moles_CO2 : ℕ
constant moles_H2SO4 : ℕ

axiom primary_reaction_completion : ∀ (n_CaOH2 n_CO2 : ℕ), n_CaOH2 = n_CO2 → primary_reaction (n_CaOH2, n_CO2) = (n_CaOH2, 1)
axiom side_reaction_completion : ∀ (n_CaOH2 n_H2SO4 : ℕ), n_CaOH2 ≤ n_H2SO4 → side_reaction (n_CaOH2, n_H2SO4) = (n_CaOH2, H2O)

theorem calc_moles_CaCO3_formed :
  moles_CaOH2 = 3 →
  moles_CO2 = 3 →
  moles_H2SO4 = 2 →
  primary_reaction (moles_CaOH2, moles_CO2) = (3, H2O) →
  ∃ moles_CaCO3, moles_CaCO3 = 3 :=
by
  intros h1 h2 h3 h4
  exists 3
  exact h4

end calc_moles_CaCO3_formed_l154_154167


namespace hikers_rate_l154_154129

noncomputable def rate_up (rate_down := 15) : ℝ := 5

theorem hikers_rate :
  let R := rate_up
  let distance_down := rate_down
  let time := 2
  let rate_down := 1.5 * R
  distance_down = rate_down * time → R = 5 :=
by
  intro h
  sorry

end hikers_rate_l154_154129


namespace total_cost_of_toppings_after_james_share_l154_154740

def topping_cost_one : ℝ := 1.50
def topping_cost_two : ℝ := 2.00
def topping_cost_three : ℝ := 1.25

def num_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 6
def fraction_eaten_by_james : ℝ := 2/3

def total_topping_cost_each_pizza : ℝ := topping_cost_one + topping_cost_two + topping_cost_three

def total_topping_cost_all_pizzas : ℝ := num_pizzas * total_topping_cost_each_pizza

def total_slices : ℕ := num_pizzas * slices_per_pizza
def james_slices : ℝ := fraction_eaten_by_james * total_slices

def cost_james_share : ℝ := fraction_eaten_by_james * total_topping_cost_all_pizzas

theorem total_cost_of_toppings_after_james_share : cost_james_share = 6.33 :=
by
  sorry

end total_cost_of_toppings_after_james_share_l154_154740


namespace union_of_sets_l154_154987

noncomputable def A (a : ℝ) : Set ℝ := {abs (a + 1), 3, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2 * a + 1, a^(2 * a + 2), a^2 + 2 * a - 1}

theorem union_of_sets (a : ℝ) (h : A a ∩ B a = {2, 3}) : 
  A a ∪ B a = {1, 2, 3, 5} :=
sorry

end union_of_sets_l154_154987


namespace chemical_solution_percentage_l154_154140

theorem chemical_solution_percentage (original_concentration replaced_portion new_concentration total_units : ℕ)
  (h_original_concentration : original_concentration = 80)
  (h_replaced_portion : replaced_portion = 0.5 * total_units)
  (h_new_concentration : new_concentration = 20)
  (total_units = 100) :
  let original_pure = original_concentration * total_units / 100
  let removed_pure = original_concentration * replaced_portion / 100
  let added_pure = new_concentration * replaced_portion / 100
  let resulting_pure = original_pure - removed_pure + added_pure
  let resulting_percentage = resulting_pure * 100 / total_units
  resulting_percentage = 50 :=
by
  sorry

end chemical_solution_percentage_l154_154140


namespace angle_between_a_and_b_l154_154641

-- Variables and definitions
variable {a b m : ℝ → ℝ → ℝ → ℝ}
variable {t : ℝ}

-- Hypotheses
def nonzero_vectors (a b : ℝ → ℝ → ℝ → ℝ) : Prop := 
  (|a| ≠ 0) ∧ (|b| ≠ 0)

def vector_m (a b : ℝ → ℝ → ℝ → ℝ) (t : ℝ) : ℝ := 
  a + t * b

def unit_norm_a (a : ℝ → ℝ → ℝ → ℝ) : Prop := 
  |a| = 1

def double_norm_b (b : ℝ → ℝ → ℝ → ℝ) : Prop := 
  |b| = 2

def minimum_norm_m (a b : ℝ → ℝ → ℝ → ℝ) (t : ℝ) : Prop := 
  ∀ (t : ℝ), |a + t * b| has_minimum_value_at t = 1/4

-- Lean statement of the proof problem
theorem angle_between_a_and_b (a b : ℝ → ℝ → ℝ → ℝ) (t : ℝ) 
  (h1 : nonzero_vectors a b)
  (h2 : vector_m a b t = a + t * b)
  (h3 : unit_norm_a a)
  (h4 : double_norm_b b)
  (h5 : minimum_norm_m a b t) :
  angle_with_minimum_norm a b = (2 * pi) / 3 := 
  sorry

end angle_between_a_and_b_l154_154641


namespace sum_of_exterior_angles_of_regular_polygon_measure_of_each_exterior_angle_of_regular_polygon_l154_154520

-- Defining a regular polygon with 14 sides
def regular_polygon_sides := 14

-- Proving the sum of all exterior angles of this polygon is 360 degrees
theorem sum_of_exterior_angles_of_regular_polygon : 
  ∀ (n : ℕ), n = regular_polygon_sides → (∑ i in (Finset.range n), 360 / n) = 360 :=
by
  intros n h
  simp [h]
  sorry

-- Proving the measure of each exterior angle is 360 degrees divided by the number of sides
theorem measure_of_each_exterior_angle_of_regular_polygon :
  ∀ (n : ℕ), n = regular_polygon_sides → (360 : ℚ) / n = (360 : ℚ) / regular_polygon_sides :=
by
  intros n h
  simp [h]
  sorry

end sum_of_exterior_angles_of_regular_polygon_measure_of_each_exterior_angle_of_regular_polygon_l154_154520


namespace countEquilateralTriangles_l154_154813

-- Define the problem conditions
def numSmallTriangles := 18  -- The number of small equilateral triangles
def includesMarkedTriangle: Prop := True  -- All counted triangles include the marked triangle "**"

-- Define the main question as a proposition
def totalEquilateralTriangles : Prop :=
  (numSmallTriangles = 18 ∧ includesMarkedTriangle) → (1 + 4 + 1 = 6)

-- The theorem stating the number of equilateral triangles containing the marked triangle
theorem countEquilateralTriangles : totalEquilateralTriangles :=
  by
    sorry

end countEquilateralTriangles_l154_154813


namespace total_bricks_required_l154_154487

def courtyard_length : ℕ := 24 * 100  -- convert meters to cm
def courtyard_width : ℕ := 14 * 100  -- convert meters to cm
def brick_length : ℕ := 25
def brick_width : ℕ := 15

-- Calculate the area of the courtyard in square centimeters
def courtyard_area : ℕ := courtyard_length * courtyard_width

-- Calculate the area of one brick in square centimeters
def brick_area : ℕ := brick_length * brick_width

theorem total_bricks_required :  courtyard_area / brick_area = 8960 := by
  -- This part will have the proof, for now, we use sorry to skip it
  sorry

end total_bricks_required_l154_154487


namespace slope_of_line_slope_is_correct_l154_154476

theorem slope_of_line (a b c : ℝ) (h : 4 * b = 5 * a - c) : b = (5 / 4) * a - 2 := 
by sorry

theorem slope_is_correct :
    slope_of_line 1 1 8 = (5 / 4) := 
by sorry

end slope_of_line_slope_is_correct_l154_154476


namespace flip_coin_probability_l154_154029

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154029


namespace evaluate_f_1990_l154_154648

def binom (n k : ℕ) : ℕ := Nat.choose n k  -- Define the binomial coefficient function

theorem evaluate_f_1990 :
  let f (n : ℕ) := ∑ k in Finset.range n, (-1) ^ k * (binom n k) ^ 2
  f 1990 = -binom 1990 995 :=
by
  sorry

end evaluate_f_1990_l154_154648


namespace range_of_a_condition_l154_154303

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem range_of_a_condition :
  range_of_a a → -1 < a ∧ a < 3 := sorry

end range_of_a_condition_l154_154303


namespace coin_flip_probability_l154_154019

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154019


namespace sufficient_but_not_necessary_for_parallel_planes_l154_154757

theorem sufficient_but_not_necessary_for_parallel_planes
  (m n : Plane → Line)
  (l1 l2 : Plane → Line)
  (α β : Plane)
  (Hm : ∀ (p : Plane), m p ≠ n p ∧ m α ∧ n α)
  (Hl : ∀ (p : Plane), l1 p ≠ l2 p ∧ l1 β ∧ l2 β ∧ (∃ (x : Point), l1 β ∧ l2 β ∧ x ∈ l1 ∧ x ∈ l2))
  (Hparallel1 : ∀ (p1 p2 : Plane), α p1 ∧ β p2 ∧ (m α ∧ l1 β) → parallel_planes α β)
  (Hparallel2 : ∀ (p1 p2 : Plane), α p1 ∧ β p2 ∧ (n α ∧ l2 β) → parallel_planes α β) :
  (parallel_planes α β ↔ (m α ∧ l1 β) ∧ (n α ∧ l2 β)) :=
  sorry

end sufficient_but_not_necessary_for_parallel_planes_l154_154757


namespace fish_offspring_base10_l154_154905

def convert_base_7_to_10 (n : ℕ) : ℕ :=
  let d2 := n / 49
  let r2 := n % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d2 * 49 + d1 * 7 + d0

theorem fish_offspring_base10 :
  convert_base_7_to_10 265 = 145 :=
by
  sorry

end fish_offspring_base10_l154_154905


namespace directrix_equation_l154_154967

def parabola_directrix (x : ℝ) : ℝ :=
  (x^2 - 8*x + 12) / 16

theorem directrix_equation :
  ∀ x, parabola_directrix x = y → y = -5/4 :=
sorry

end directrix_equation_l154_154967


namespace product_is_even_if_n_is_odd_l154_154380

theorem product_is_even_if_n_is_odd : 
  ∀ (n : ℕ), n ≥ 2 → (∀ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → ∃ j, 1 ≤ j ∧ j ≤ n ∧ i ≠ j ∧ a i = j) → 
  (∃ k, ∃ p, 1 ≤ k ∧ k ≤ n ∧ (n + 1).bodd ≠ 0 → (∃ m, (1 + k + a k) = (2 * m + 1) ∨ (product (list.range n).map (λ i, i + a i + 1)).odd)) := sorry

end product_is_even_if_n_is_odd_l154_154380


namespace symmetric_line_equation_l154_154995

theorem symmetric_line_equation (x y : ℝ) :
  (∃ l₂ : ℝ → ℝ, (∀ x, l₂ x = 11 * x - 21 / 2)) ↔
  (∃ P : ℝ × ℝ, ∀ l : ℝ → ℝ, (∀ x, l x = 3 * x + 3)) ∧
  (∀ l₁ : ℝ → ℝ, (∀ x, l₁ x = 2 * x)) ∧
  ∀ y, (y l) = x :=
sorry

end symmetric_line_equation_l154_154995


namespace intersection_complement_M_N_l154_154284

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 ≤ x }
def N : Set ℝ := { x | 2^x < 1 }
def complement_N : Set ℝ := { x | x ≥ 0 }
def solution : Set ℝ := [0, 1]

theorem intersection_complement_M_N :
  (M ∩ complement_N) = solution := by
  {
    sorry
  }

end intersection_complement_M_N_l154_154284


namespace flip_coin_probability_l154_154037

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154037


namespace greatest_perpendicular_faces_l154_154474

def max_perpendicular_faces (n : ℕ) : ℕ :=
  if even n then n / 2 else (n + 1) / 2

theorem greatest_perpendicular_faces (n : ℕ) :
  max_perpendicular_faces n = 
  if even n then n / 2 else (n + 1) / 2 := 
by
  sorry

end greatest_perpendicular_faces_l154_154474


namespace summer_camp_total_kids_l154_154457

def summer_camp_kids_total (T : ℕ) : Prop :=
  (1/2 * T > 0) ∧                     -- Condition: Half of the kids are going to soccer camp.
  (1/3 * (1/2 * T) > 0) ∧             -- Condition: 1/3 of the kids going to soccer camp are going in the morning.
  (2/3 * (1/2 * T) = 750) ∧           -- Condition: 750 kids are going to soccer camp in the afternoon.
  (1/5 * (1/2 * T) > 0) ∧             -- Condition: 1/5 of the other half of kids in the camp are going to basketball camp.
  (1/3 * (1/5 * (1/2 * T)) = 200)     -- Condition: 200 kids are going to the morning basketball camp.

theorem summer_camp_total_kids : ∃ T : ℕ, summer_camp_kids_total T ∧ T = 6000 :=
begin
  use 6000,
  unfold summer_camp_kids_total,
  split,
  { norm_num },       -- 1/2 * 6000
  split,
  { norm_num },       -- 1/3 * (1/2 * 6000)
  split,
  { norm_num },       -- 2/3 * (1/2 * 6000)
  split,
  { norm_num },       -- 1/5 * (1/2 * 6000)
  { norm_num },       -- 1/3 * (1/5 * (1/2 * 6000))
end

end summer_camp_total_kids_l154_154457


namespace percentage_increase_40_percent_l154_154524

-- Definitions for the conditions
def total_capacity := 1312.5
def added_water := 300
def new_fill_ratio := 0.8
def percentage_increase (increase original: ℝ) := (increase / original) * 100

-- Theorem statement
theorem percentage_increase_40_percent :
  let current_amount := new_fill_ratio * total_capacity - added_water in
  percentage_increase added_water current_amount = 40 :=
by
  let current_amount := new_fill_ratio * total_capacity - added_water
  sorry

end percentage_increase_40_percent_l154_154524


namespace angela_height_l154_154162

def height_of_Amy : ℕ := 150
def height_of_Helen : ℕ := height_of_Amy + 3
def height_of_Angela : ℕ := height_of_Helen + 4

theorem angela_height : height_of_Angela = 157 := by
  sorry

end angela_height_l154_154162


namespace area_of_circle_l154_154040

theorem area_of_circle :
  (∃ x y : ℝ, x^2 + y^2 - 6*x + 8*y = 0) →
  ∃ r : ℝ, r = 5 ∧ real.pi * r^2 = 25 * real.pi :=
by
  sorry

end area_of_circle_l154_154040


namespace percentage_error_in_area_l154_154552

theorem percentage_error_in_area (S : ℝ) (h : S > 0) :
  let S' := S * 1.06
  let A := S^2
  let A' := (S')^2
  (A' - A) / A * 100 = 12.36 := by
  sorry

end percentage_error_in_area_l154_154552


namespace inequality_solution_set_l154_154974

theorem inequality_solution_set :
  {x : ℝ | (x - 5) * (x + 1) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 5} :=
by
  sorry

end inequality_solution_set_l154_154974


namespace determine_de_length_l154_154806

open real

structure RightTriangle (a b c : Type) :=
  (right_angle_at : a)

variable (D E F : Type)
variable [RightTriangle D E F]

def length_of_median (x y : D) := sqrt (x + y)

def midpoint (x y : D) := (x + y) / 2

noncomputable def DE_length : ℝ := sqrt 56

theorem determine_de_length (PD_length EQ_length : ℝ) (h1 : PD_length = 3 * sqrt 5) (h2 : EQ_length = 5) : 
  sqrt 56 = DE_length := 
begin
  sorry -- proof skipped 
end

end determine_de_length_l154_154806


namespace number_of_factors_of_1320_l154_154288

theorem number_of_factors_of_1320 : 
  (nat.factors_count_exists 1320 ((2, 3), (3, 1), (5, 1), (11, 1)) → ∃ n, n = 32) := 
begin
  sorry
end

end number_of_factors_of_1320_l154_154288


namespace inequality_proof_l154_154376

noncomputable def p (n : ℕ) (x : ℕ → ℝ) : ℝ := ∑ i in Finset.range n, x i

noncomputable def q (n : ℕ) (x : ℕ → ℝ) : ℝ := ∑ i in Finset.range n, ∑ j in Finset.range i, x i * x j

theorem inequality_proof (n : ℕ) (x : ℕ → ℝ) (hn : n ≥ 3) :
  (n - 1) * (p n x)^2 / n - 2 * q n x ≥ 0 := sorry

end inequality_proof_l154_154376


namespace number_of_1s_in_black_cells_even_l154_154314

-- Define the problem conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1

def chessboard_condition (a : ℕ → ℕ → ℕ) : Prop :=
  (∀ i < 10, is_odd (∑ j in finset.range 14, a i j)) ∧
  (∀ j < 14, is_odd (∑ i in finset.range 10, a i j))

def is_black_cell (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the main theorem 
theorem number_of_1s_in_black_cells_even (a : ℕ → ℕ → ℕ) 
  (h1 : ∀ i j, a i j = 0 ∨ a i j = 1)
  (h2 : chessboard_condition a) : (∑ i in finset.range 10, ∑ j in finset.range 14, if is_black_cell i j then a i j else 0) % 2 = 0 := 
sorry

end number_of_1s_in_black_cells_even_l154_154314


namespace lending_period_C_l154_154515

theorem lending_period_C (P_B P_C : ℝ) (R : ℝ) (T_B I_total : ℝ) (T_C_months : ℝ) :
  P_B = 5000 ∧ P_C = 3000 ∧ R = 0.10 ∧ T_B = 2 ∧ I_total = 2200 ∧ 
  T_C_months = (2 / 3) * 12 → T_C_months = 8 := by
  intros h
  sorry

end lending_period_C_l154_154515


namespace atomic_weight_Br_correct_l154_154198

def atomic_weight_Ba : ℝ := 137.33
def molecular_weight_compound : ℝ := 297
def atomic_weight_Br : ℝ := 79.835

theorem atomic_weight_Br_correct :
  molecular_weight_compound = atomic_weight_Ba + 2 * atomic_weight_Br :=
by
  sorry

end atomic_weight_Br_correct_l154_154198


namespace l1_l2_intersect_l1_l2_parallel_l1_l2_coincide_l1_l2_perpendicular_l154_154637

noncomputable def l1 (m : ℚ) : ℚ × ℚ → Prop :=
  λ p, (m + 3) * p.1 + 2 * p.2 = 5 - 3 * m

noncomputable def l2 (m : ℚ) : ℚ × ℚ → Prop :=
  λ p, 4 * p.1 + (5 + m) * p.2 = 16

-- Proving intersection condition
theorem l1_l2_intersect (m : ℚ) (h : m ≠ -1 ∧ m ≠ -7) :
  ∃ p, l1 m p ∧ l2 m p :=
sorry

-- Proving parallel condition
theorem l1_l2_parallel (m : ℚ) (h : m = -7) :
  ∀ p, l1 m p → l2 m p :=
sorry

-- Proving coincidence condition
theorem l1_l2_coincide (m : ℚ) (h : m = -1) :
  ∀ p, l1 m p ↔ l2 m p :=
sorry

-- Proving perpendicular condition
theorem l1_l2_perpendicular (m : ℚ) (h : m = -11/3) :
  ∃ p, l1 m p ∧ l2 m p ∧ (m + 3) / 4 * (5 + m) / 2 = -1 :=
sorry

end l1_l2_intersect_l1_l2_parallel_l1_l2_coincide_l1_l2_perpendicular_l154_154637


namespace man_older_than_son_l154_154893

theorem man_older_than_son : 
  ∀ (M S : ℕ), S = 25 → (M + 2 = 2 * (S + 2)) → (M - S = 27) :=
by 
  intros M S hS hM,
  subst hS,
  simp at hM,
  sorry

end man_older_than_son_l154_154893


namespace monotonic_intervals_l154_154199

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_intervals :
  (∀ x (h : 0 < x ∧ x < Real.exp 1), 0 < f x) ∧
  (∀ x (h : Real.exp 1 < x), f x < 0) :=
by
  sorry

end monotonic_intervals_l154_154199


namespace johns_climb_height_correct_l154_154745

noncomputable def johns_total_height : ℝ :=
  let stair1_height := 4 * 15
  let stair2_height := 5 * 12.5
  let total_stair_height := stair1_height + stair2_height
  let rope1_height := (2 / 3) * stair1_height
  let rope2_height := (3 / 5) * stair2_height
  let total_rope_height := rope1_height + rope2_height
  let rope1_height_m := rope1_height / 3.281
  let rope2_height_m := rope2_height / 3.281
  let total_rope_height_m := rope1_height_m + rope2_height_m
  let ladder_height := 1.5 * total_rope_height_m * 3.281
  let rock_wall_height := (2 / 3) * ladder_height
  let total_pre_tree := total_stair_height + total_rope_height + ladder_height + rock_wall_height
  let tree_height := (3 / 4) * total_pre_tree - 10
  total_stair_height + total_rope_height + ladder_height + rock_wall_height + tree_height

theorem johns_climb_height_correct : johns_total_height = 679.115 := by
  sorry

end johns_climb_height_correct_l154_154745


namespace value_of_b_l154_154307

-- Definitions
def A := 45  -- in degrees
def B := 60  -- in degrees
def a := 10  -- length of side a

-- Assertion
theorem value_of_b : (b : ℝ) = 5 * Real.sqrt 6 :=
by
  -- Definitions used in previous problem conditions
  let sin_A := Real.sin (Real.pi * A / 180)
  let sin_B := Real.sin (Real.pi * B / 180)
  -- Applying the Law of Sines
  have law_of_sines := (a / sin_A) = (b / sin_B)
  -- Simplified calculation of b (not provided here; proof required later)
  sorry

end value_of_b_l154_154307


namespace floor_log_sum_l154_154214

theorem floor_log_sum :
  (∑ k in Finset.range 2010, Real.floor (Real.log10 (k + 1))) = 4920 :=
sorry

end floor_log_sum_l154_154214


namespace intersection_of_M_and_N_l154_154355

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l154_154355


namespace z_conjugate_second_quadrant_l154_154223

def z_conjugate (z : ℂ) : ℂ := complex.conj z

theorem z_conjugate_second_quadrant (z : ℂ) :
  (3 + real.sqrt 3 * complex.I) * z = -2 * real.sqrt 3 * complex.I →
  let conj_z := z_conjugate z in -1/2 < conj_z.re ∧ conj_z.re < 0 ∧ 0 < conj_z.im ∧ conj_z.im < real.sqrt 3/2 :=
begin
  sorry
end

end z_conjugate_second_quadrant_l154_154223


namespace smallest_M_for_polynomial_l154_154184

theorem smallest_M_for_polynomial (M : ℕ) :
  (∀ a b c : ℤ, ∃ (P : ℤ[X]), P.eval 1 = a * M ∧ P.eval 2 = b * M ∧ P.eval 4 = c * M) ↔ M = 6 :=
by
  sorry

end smallest_M_for_polynomial_l154_154184


namespace cos_2x_min_value_l154_154265

-- Define the function f
def f (x : ℝ) : ℝ := 9 / (8 * cos (2 * x) + 16) - (sin x) ^ 2

-- Define the Lean statement for the proof problem
theorem cos_2x_min_value : ∃ x : ℝ, cos (2 * x) = -1/2 ∧ ∀ y : ℝ, f y ≥ f x := 
sorry

end cos_2x_min_value_l154_154265


namespace restore_original_order_after_3_repetitions_l154_154950

theorem restore_original_order_after_3_repetitions : 
  ∀ (cards : List ℕ), cards.length = 8 → 
  (restore_order_after_n_repetitions cards 3 = cards) := by
  sorry

def distribute_and_stack (cards : List α) : List α :=
  let n := cards.length / 2
  let left := list.filteri (λ i _, i % 2 = 0) cards
  let right := list.filteri (λ i _, i % 2 = 1) cards
  left ++ right 

def restore_order_after_n_repetitions (cards : List α) (n : Nat) : List α :=
  if n = 0 then cards
  else restore_order_after_n_repetitions (distribute_and_stack cards) (n - 1)

# Check the function to ensures it matches the hypothesis
# def distribute_and_stack
-- function signature should be: 
-- distribute_and_stack : List α → List α

end restore_original_order_after_3_repetitions_l154_154950


namespace exponent_inequality_l154_154639

theorem exponent_inequality (a b c : ℝ) (h1 : a ≠ 1) (h2 : b ≠ 1) (h3 : c ≠ 1) (h4 : a > b) (h5 : b > c) (h6 : c > 0) : a ^ b > c ^ b :=
  sorry

end exponent_inequality_l154_154639


namespace intersection_M_N_l154_154365

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end intersection_M_N_l154_154365


namespace largestInteresting_is_6310972_interesting_property_largestInteresting_l154_154135

-- Definition to check if a digit is interesting
def isInterestingPair (d1 d2 : ℕ) : Prop :=
  (d1 + d2 = 1 ∨ d1 + d2 = 4 ∨ d1 + d2 = 9)

-- Check if all digits in a number are distinct
def allDistinctDigits (n : ℕ) : Prop :=
  let digits := n.digits in
  ∀ i j, i ≠ j → digits.get i ≠ digits.get j

-- Check if all adjacent pairs are interesting
def allPairsInteresting (n : ℕ) : Prop :=
  let digits := n.digits in
  ∀ i, i < digits.length - 1 → isInterestingPair (digits.get i) (digits.get (i + 1))

-- Definition of an interesting number
def isInterestingNumber (n : ℕ) : Prop :=
  allDistinctDigits n ∧ allPairsInteresting n

-- Largest interesting number
noncomputable def largestInteresting : ℕ :=
  6310972

-- Theorem that the largest interesting number is 6310972
theorem largestInteresting_is_6310972 :
  largestInteresting = 6310972 :=
by rfl

-- Theorem stating that 6310972 is indeed an interesting number
theorem interesting_property_largestInteresting :
  isInterestingNumber 6310972 :=
sorry

end largestInteresting_is_6310972_interesting_property_largestInteresting_l154_154135


namespace a_n_general_term_b_n_general_term_T_n_sum_l154_154658

open Nat

def S_n (n : ℕ) : ℕ := n^2

def a_n (n : ℕ) : ℕ := 
  if n = 1 then 1
  else S_n n - S_n (n - 1)

def b_n : ℕ → ℚ
| 1       := 1
| (n + 1) := (b_n n) / 2

def T_n (n : ℕ) :=
  (List.range n).sum (λ k, (2 * (k + 1) - 1) * 2^k)

theorem a_n_general_term (n : ℕ) : a_n n = 2 * n - 1 := sorry

theorem b_n_general_term (n : ℕ) : b_n n = (1 / 2)^(n - 1) := sorry

theorem T_n_sum (n : ℕ) : 
  T_n n = (2 * n - 1) * 2^n + 3 - 2^(n + 1) := sorry

end a_n_general_term_b_n_general_term_T_n_sum_l154_154658


namespace square_area_percentage_error_l154_154551

theorem square_area_percentage_error (s : ℝ) (h : s > 0) : 
  let measured_s := 1.06 * s in
  let actual_area := s^2 in
  let calculated_area := measured_s^2 in
  let error_area := calculated_area - actual_area in
  let percentage_error := (error_area / actual_area) * 100 in
  percentage_error = 12.36 := 
by
  sorry

end square_area_percentage_error_l154_154551


namespace positive_difference_R_coordinates_l154_154846

def point := ℝ × ℝ

variables A B C R S : point

-- Conditions
-- Vertices of triangle ABC
def A : point := (0, 10)
def B : point := (2, 0)
def C : point := (10, 0)

-- A vertical line intersects AC at R
-- and BC at S forming triangle RSC
-- Coordinates of R and S
variable x1 : ℝ
def R : point := (x1, 10 - x1)
def S : point := (x1, 0)

-- Area of triangle RSC is 20
axiom area_RSC : 0.5 * x1 * (10 - x1) = 20

-- Goal: positive difference of the x and y coordinates of R
theorem positive_difference_R_coordinates : |(R.1 - R.2)| = 6 :=
sorry

end positive_difference_R_coordinates_l154_154846


namespace cathy_wins_probability_l154_154158

theorem cathy_wins_probability : 
  -- Definitions of the problem conditions
  let p_win := (1 : ℚ) / 6
  let p_not_win := (5 : ℚ) / 6
  -- The probability that Cathy wins
  (p_not_win ^ 2 * p_win) / (1 - p_not_win ^ 3) = 25 / 91 :=
by
  sorry

end cathy_wins_probability_l154_154158


namespace smallest_integer_with_16_divisors_l154_154052

-- Define prime factorization and the function to count divisors
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def number_of_divisors (n : ℕ) : ℕ :=
  (prime_factorization n).foldr (λ ⟨_, a⟩ acc, acc * (a + 1)) 1

-- Main theorem stating the smallest positive integer with exactly 16 divisors is 210
theorem smallest_integer_with_16_divisors : 
  ∃ n, n > 0 ∧ number_of_divisors n = 16 ∧ ∀ m, m > 0 ∧ number_of_divisors m = 16 → n ≤ m :=
begin
  use 210,
  split,
  { -- Prove 210 > 0
    exact nat.zero_lt_succ _,
  },
  split,
  { -- Prove number_of_divisors 210 = 16
    sorry,
  },
  { -- Prove minimality
    intros m hm1 hm2,
    sorry,
  }
end

end smallest_integer_with_16_divisors_l154_154052


namespace smaller_than_neg3_l154_154152

theorem smaller_than_neg3 :
  (∃ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3) ∧ ∀ x ∈ ({0, -1, -5, -1/2} : Set ℚ), x < -3 → x = -5 :=
by
  sorry

end smaller_than_neg3_l154_154152


namespace place_mat_length_l154_154124

noncomputable def matLength (R : ℝ) (n : ℕ) (arcLength : ℝ) : ℝ :=
  2 * R * Real.sin (arcLength / (2 * Real.pi * R) * Real.pi / (n : ℝ))

theorem place_mat_length :
  let R := 5
  let n := 8
  let circumference := 2 * Real.pi * R
  let arcLength := circumference / n
  matLength R n arcLength = 3.83 :=
by
  sorry

end place_mat_length_l154_154124


namespace remainder_t100_mod_4_l154_154598

def T : ℕ → ℕ
| 0 := 3
| (n + 1) := 3 ^ T n

theorem remainder_t100_mod_4 : T 100 % 4 = 3 := sorry

end remainder_t100_mod_4_l154_154598


namespace principal_value_range_l154_154629

noncomputable theory

-- Define the principal value range of argument
def principal_value_range_arg {z : ℂ} (h : |z| = 1) : set ℝ :=
  {θ : ℝ | ∃ k ∈ {0, 1}, kπ - arccos ≤ θ ∧ θ ≤ kπ + arccos}

-- Theorem stating the equivalent proof problem
theorem principal_value_range (z : ℂ) (h : |z| = 1) :
  principal_value_range_arg h = {θ : ℝ | ∃ k ∈ {0, 1}, kπ - arccos ≤ θ ∧ θ ≤ kπ + arccos} :=
sorry

end principal_value_range_l154_154629


namespace five_coins_all_heads_or_tails_l154_154210

theorem five_coins_all_heads_or_tails : 
  (1 / 2) ^ 5 + (1 / 2) ^ 5 = 1 / 16 := 
by 
  sorry

end five_coins_all_heads_or_tails_l154_154210


namespace eventually_one_knows_other_number_l154_154464

variable (a b x y : ℕ)
variable (h1 : 0 < x)
variable (h2 : x < y)
variable (h3 : x = a + b ∨ y = a + b)

theorem eventually_one_knows_other_number :
  ∃ k : ℕ, k > 0 → k * (y - x) < b → b < y - k * (y - x) ∨ (∃ s : ℕ, s <= k ∧ knows_other s) :=
sorry

end eventually_one_knows_other_number_l154_154464


namespace spadesuit_value_l154_154216

def spadesuit (x y : ℝ) : ℝ := x - 1 / y

theorem spadesuit_value : spadesuit 2 (spadesuit 2 2) = 4 / 3 := by
  sorry

end spadesuit_value_l154_154216


namespace one_thirds_in_nine_halves_l154_154679

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end one_thirds_in_nine_halves_l154_154679


namespace tenth_term_arithmetic_sequence_l154_154085

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ : ℚ) (d : ℚ), 
  (a₁ = 3/4) → (d = 1/2) →
  (a₁ + 9 * d) = 21/4 :=
by
  intro a₁ d ha₁ hd
  rw [ha₁, hd]
  sorry

end tenth_term_arithmetic_sequence_l154_154085


namespace remainder_eq_zero_l154_154948

-- Define the polynomials
def P := polynomial ℝ -- You could choose ℂ or another field

open polynomial

-- Polynomial definitions
def f := X^5 - 1
def g := X^3 - 1
def h := X^2 + X + 1

-- The Remainder Theorem
theorem remainder_eq_zero : (f * g) % h = 0 := 
by sorry

end remainder_eq_zero_l154_154948


namespace sum_of_integer_solutions_l154_154477

theorem sum_of_integer_solutions :
  (∑ x in Finset.filter (λ x, 1 < (x - 3)^2 ∧ (x - 3)^2 < 25) (Finset.Icc (-100) 100), x) = 18 := by
  -- The range -100 to 100 is chosen arbitrarily to cover all potential integer solutions.
  sorry

end sum_of_integer_solutions_l154_154477


namespace determine_range_of_m_l154_154668

theorem determine_range_of_m (m : ℝ) :
  (∀ x ∈ Icc (-2 : ℝ) (2 : ℝ), (deriv (λ x, 9^x + m * 3^x - 3)) x ≤ 0) →
  m ≤ -18 :=
begin
  intro h,
  -- Function derivative calculation and monotonicity check will be part of the proof
  sorry
end

end determine_range_of_m_l154_154668


namespace part1_part2_part3_l154_154544

-- Part 1
theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) : 
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem part2 (a m n : ℤ) (h1 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) (h2 : 0 < a) (h3 : 0 < m) (h4 : 0 < n) : 
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem part3 : Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end part1_part2_part3_l154_154544


namespace cost_per_box_per_month_l154_154122

-- Define the conditions
def box_length : ℝ := 15
def box_width : ℝ := 12
def box_height : ℝ := 10
def total_volume : ℝ := 1_080_000 -- in cubic inches
def total_monthly_cost : ℝ := 360 -- in dollars

-- The main statement we want to prove
theorem cost_per_box_per_month : 
  (total_monthly_cost / (total_volume / (box_length * box_width * box_height))) = 0.60 :=
by 
  sorry

end cost_per_box_per_month_l154_154122


namespace compute_series_sum_l154_154929

noncomputable def term (n : ℕ) : ℝ := (5 * n - 2) / (3 ^ n)

theorem compute_series_sum : 
  ∑' n, term n = 11 / 4 := 
sorry

end compute_series_sum_l154_154929


namespace prob_all_heads_or_tails_five_coins_l154_154206

theorem prob_all_heads_or_tails_five_coins :
  (number_of_favorable_outcomes : ℕ) (total_number_of_outcomes : ℕ) (probability : ℚ) 
  (h_favorable : number_of_favorable_outcomes = 2)
  (h_total : total_number_of_outcomes = 32)
  (h_probability : probability = number_of_favorable_outcomes / total_number_of_outcomes) :
  probability = 1 / 16 :=
by
  sorry

end prob_all_heads_or_tails_five_coins_l154_154206


namespace exists_root_f_in_interval_l154_154262

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 3 * x + 2) * Real.log x + 2008 * x - 2009

theorem exists_root_f_in_interval : ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
  by
  have f1 : f 1 = -1 := by
    rw [f, Real.log_one, zero_mul, add_zero, sub_sub, one_mul, sub_self, sub_zero, neg_one_eq_add_neg]
    linarith
  have f2 : f 2 = 2007 := by
    rw [f, sq, Real.log_two, sub_sub, two_mul, sub_self, sub_self, add_sub]
    norm_num
  sorry

end exists_root_f_in_interval_l154_154262


namespace number_of_n_l154_154972

theorem number_of_n (h1: n > 0) (h2: n ≤ 2000) (h3: ∃ m, 10 * n = m^2) : n = 14 :=
by sorry

end number_of_n_l154_154972


namespace prove_c_eq_2_or_all_x_equal_l154_154246

theorem prove_c_eq_2_or_all_x_equal (c : ℝ) (n : ℕ) (x : fin n → ℝ)
  (h1 : c > -2)
  (h2 : ∀ i, 0 < x i)
  (h3 : (∑ i : fin n, Real.sqrt ((x i)^2 + c * (x i) * (x ((i.1 + 1) % n)) + (x ((i.1 + 1) % n))^2)) = Real.sqrt (c + 2) * ∑ i : fin n, x i) :
  c = 2 ∨ ∀ i, x i = x 0 :=
sorry

end prove_c_eq_2_or_all_x_equal_l154_154246


namespace probability_even_product_l154_154413

-- Define the spinners
def spinner_C : List ℕ := [1, 1, 2, 3, 3, 4]
def spinner_D : List ℕ := [1, 2, 2, 3, 4, 4]

-- Calculate the required probability
theorem probability_even_product :
  (let odd_C := (spinner_C.filter (λ n, n % 2 = 1)).length / spinner_C.length,
       odd_D := (spinner_D.filter (λ n, n % 2 = 1)).length / spinner_D.length,
       prob_odd_product := odd_C * odd_D,
       prob_even_product := 1 - prob_odd_product
   in prob_even_product) = 7 / 9 := 
sorry

end probability_even_product_l154_154413


namespace Jim_distance_in_24_steps_l154_154170

theorem Jim_distance_in_24_steps 
    (Carly_steps_per_metre : ℝ)
    (Jim_steps_per_Carly_steps : ℝ)
    (Carly_distance_per_step : ℝ) :
    (Carly_steps_per_metre = 3) →
    (Jim_steps_per_Carly_steps = 4) →
    (Carly_distance_per_step = 0.5) →
    (Jim_distance_24_steps = 24 * Carly_steps_per_metre * Carly_distance_per_step / Jim_steps_per_Carly_steps) →
    Jim_distance_24_steps = 9 :=
begin
  sorry
end

end Jim_distance_in_24_steps_l154_154170


namespace problem_statement_l154_154497
open ProbabilityTheory

-- Define the sequence of independent standard normal random variables
noncomputable def xi : ℕ → MeasureTheory.Measure (ℝ)
| i := MeasureTheory.Measure.map (λ x, x) MeasureTheory.MeasureSpace.ProbabilitySpace (MeasureTheory.Normal 0 1)

-- Define X_n as the sum of first n xi's
noncomputable def X (n : ℕ) : ℝ := ∑ i in finset.range n, xi i

-- Define S_n as the sum of first n X_i's
noncomputable def S (n : ℕ) : ℝ := ∑ i in finset.range n, X i

-- Statement of the problem
theorem problem_statement :
  ∀ n, distribution_converges (λ n, (S n) / n^(3/2)) (MeasureTheory.Normal 0 (1/3)) as n → ∞ :=
sorry

end problem_statement_l154_154497


namespace values_f_comparison_f_l154_154236

noncomputable def a (n : ℕ) : ℚ := 1 / n
noncomputable def S (n : ℕ) : ℚ := ∑ i in finset.range n, a (i + 1)
noncomputable def f : ℕ → ℚ
| 1     := S (2 * 1)
| (n+1) := S (2 * (n + 1)) - S n

theorem values_f :
  f 1 = 3 / 2 ∧ f 2 = 13 / 12 ∧ f 3 = 19 / 20 :=
by sorry

theorem comparison_f (n : ℕ) :
  (n = 1 ∨ n = 2 → f n > 1) ∧ (n ≥ 3 → f n < 1) :=
by sorry

end values_f_comparison_f_l154_154236


namespace limsup_le_epsilon_almost_surely_l154_154796

noncomputable def problem :=
  (sum_prob : ∀ (ξ : ℕ → ℝ), ε > 0 → (∑' n, ℙ (ξ n > ε)) < ∞) →
  (eventually_le : ∀ (ξ : ℕ → ℝ) (ε : ℝ), 0 < ε → ℙ (∀ᶠ n in at_top, ξ n ≤ ε) = 1)

def limsup_le_epsilon (ξ : ℕ → ℝ) (ε : ℝ) (h1 : ∑' n, ℙ (ξ n > ε) < ∞) (h2 : ε > 0) : Prop :=
  ∃ N, ∀ n ≥ N, ξ n ≤ ε
  
theorem limsup_le_epsilon_almost_surely (ξ : ℕ → ℝ) (ε : ℝ)
  (h1 : ∑' n, ℙ (ξ n > ε) < ∞) (h2 : ε > 0) : 
  ℙ (∀ᶠ n in at_top, ξ n ≤ ε) = 1 :=
by
  sorry

end limsup_le_epsilon_almost_surely_l154_154796


namespace translate_point_left_l154_154843

def initial_point : ℝ × ℝ := (-2, -1)
def translation_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1 - units, p.2)

theorem translate_point_left :
  translation_left initial_point 2 = (-4, -1) :=
by
  -- By definition and calculation
  -- Let p = initial_point
  -- x' = p.1 - 2,
  -- y' = p.2
  -- translation_left (-2, -1) 2 = (-4, -1)
  sorry

end translate_point_left_l154_154843


namespace constant_sum_sequence_a18_S21_l154_154944

-- Define the sequence as a constant sum sequence
noncomputable def a : ℕ → ℝ
| 0       := 2
| (n + 1) := 5 - a n

-- Define the sum of the first n terms
noncomputable def sum_seq (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

-- Theorem statement for a18 and S21
theorem constant_sum_sequence_a18_S21 :
  a 17 = 3 ∧ sum_seq 21 = 52 := sorry

end constant_sum_sequence_a18_S21_l154_154944


namespace range_of_solutions_l154_154575

open Real

theorem range_of_solutions (b : ℝ) :
  (∀ x : ℝ, 
    (x = -3 → x^2 - b*x - 5 = 13)  ∧
    (x = -2 → x^2 - b*x - 5 = 5)   ∧
    (x = -1 → x^2 - b*x - 5 = -1)  ∧
    (x = 4 → x^2 - b*x - 5 = -1)   ∧
    (x = 5 → x^2 - b*x - 5 = 5)    ∧
    (x = 6 → x^2 - b*x - 5 = 13)) →
  (∀ x : ℝ,
    (x^2 - b*x - 5 = 0 → (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) :=
by
  intros h x hx
  sorry

end range_of_solutions_l154_154575


namespace compute_expression_l154_154172

theorem compute_expression :
  20 * (150 / 3 + 40 / 5 + 16 / 25 + 2) = 1212.8 :=
by
  -- skipping the proof steps
  sorry

end compute_expression_l154_154172


namespace unique_function_l154_154946

theorem unique_function (f : ℝ → ℝ) 
  (H : ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y) : 
  ∀ x : ℝ, f x = 3 * x :=
by 
  sorry

end unique_function_l154_154946


namespace problem_statement_l154_154941

variable (C : ℝ → ℝ → Prop)
variable (F : ℝ × ℝ)
variable (M N : ℝ × ℝ)
variable (k a n : ℝ)
variable (l : ℝ → ℝ → Prop)

def parabola_c (x y : ℝ) : Prop := y^2 = 4 * x

def midpoint (M N : ℝ × ℝ) : ℝ × ℝ :=
  ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

def distance (p q : ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt

def vertical_bisector (M N : ℝ × ℝ) : ℝ → ℝ :=
  let m := midpoint M N
  let k_inv := -1 / k
  fun x => k_inv * (x - m.1) + m.2

theorem problem_statement :
  (∀ x y, parabola_c x y ↔ C x y) →
  F = (1, 0) →
  (∀ x y, l x y ↔ y = k * x) →
  (∀ x y, C x y → l x y → True) → -- Intersection condition
  (∃ M N, C M.1 M.2 ∧ C N.1 N.2 ∧ l M.1 M.2 ∧ l N.1 N.2) →
  let m := midpoint M N in
  m.2 = 2 / k →
  let a := k * m.2 + m.1 in
  n = 2 * m.1 + 2 →
  2 * a - n = 2 := by sorry

end problem_statement_l154_154941


namespace sin_double_angle_identity_l154_154982

theorem sin_double_angle_identity (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : Real.sin (2 * x) = -7/25 := 
by 
  sorry

end sin_double_angle_identity_l154_154982


namespace sequence_bound_l154_154444

theorem sequence_bound (a b c : ℕ → ℝ) :
  (a 0 = 1) ∧ (b 0 = 0) ∧ (c 0 = 0) ∧
  (∀ n, n ≥ 1 → a n = a (n-1) + c (n-1) / n) ∧
  (∀ n, n ≥ 1 → b n = b (n-1) + a (n-1) / n) ∧
  (∀ n, n ≥ 1 → c n = c (n-1) + b (n-1) / n) →
  ∀ n, n ≥ 1 → |a n - (n + 1) / 3| < 2 / Real.sqrt (3 * n) :=
by sorry

end sequence_bound_l154_154444


namespace count_valid_B_l154_154802

open Finset

def valid_B (B : Finset ℕ) : Prop :=
  B ⊆ {1, 2, 3, 4, 5, 6, 7, 8} ∧
  B.card = 3 ∧
  ∀ a b ∈ B, a ≠ b → a + b ≠ 9

theorem count_valid_B : (univ.filter valid_B).card = 10 :=
sorry

end count_valid_B_l154_154802


namespace triangle_ratio_sin_find_BC_length_l154_154707

theorem triangle_ratio_sin (A B C D : Point) (k : ℝ)
  (h1 : collinear A B D)
  (h2 : dist A D / dist D B = 1 / 3)
  (h3 : ∡ A C D = α)
  (h4 : ∡ B C D = β) :
  dist A C / dist B C = sin β / (3 * sin α) := sorry

theorem find_BC_length (A B C D : Point) (k : ℝ)
  (h1 : collinear A B D)
  (h2 : dist A D / dist D B = 1 / 3)
  (h3 : ∡ A C D = π / 6)
  (h4 : ∡ B C D = π / 2)
  (h5 : dist A B = real.sqrt 19) :
  dist B C = 3 := sorry

end triangle_ratio_sin_find_BC_length_l154_154707


namespace intersection_M_N_l154_154347

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l154_154347


namespace max_value_A_is_n_squared_over_2_l154_154248

noncomputable def max_value_A (n : ℕ) : ℝ :=
  ∀ (x : Fin n → ℝ), let A := (∑ i in Finset.univ, Real.sin (x i)) * (∑ i in Finset.univ, Real.cos (x i))
  in A ≤ n^2 / 2

theorem max_value_A_is_n_squared_over_2 (n : ℕ) : 
  (∃ (x : Fin n → ℝ), (let A := (∑ i in Finset.univ, Real.sin (x i)) * (∑ i in Finset.univ, Real.cos (x i))
  in A = n^2 / 2)) :=
sorry

end max_value_A_is_n_squared_over_2_l154_154248


namespace problem_solution_l154_154861

open Nat

def sum_odd (n : ℕ) : ℕ :=
  n ^ 2

def sum_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem problem_solution : 
  sum_odd 1010 - sum_even 1009 = 1010 :=
by
  -- Here the proof would go
  sorry

end problem_solution_l154_154861


namespace cosineFunction_properties_l154_154202

noncomputable def cosineFunction (x : ℝ) : ℝ := 5 * Real.cos (x + (Real.pi / 4)) + 2

theorem cosineFunction_properties :
  (abs 5 = 5) ∧ (-(Real.pi / 4) = - (Real.pi / 4)) :=
by
  unfold cosineFunction
  split
  . sorry
  . sorry

end cosineFunction_properties_l154_154202


namespace find_rectangle_length_l154_154316

-- Define the problem constants and conditions
def square_perimeter : ℝ := 64
def rectangle_width : ℝ := 8
def triangle_height : ℝ := 64

-- Define the length of the rectangle
def y : ℝ := 8

-- Use the conditions to state the theorem that the length of the rectangle is 8
theorem find_rectangle_length 
  (square_perimeter = 64) 
  (rectangle_width = 8)
  (triangle_height = 64)
  (square_side : ℝ := square_perimeter / 4) 
  (square_area : ℝ := square_side * square_side) 
  (triangle_area : ℝ := 1/2 * triangle_height * y)
  (square_area = triangle_area) : y = 8 := 
by 
  sorry

end find_rectangle_length_l154_154316


namespace find_central_angle_of_sector_l154_154810

noncomputable def sector_angle (r : ℝ) (A : ℝ) : ℝ :=
  (A * 360) / (Real.pi * r^2)

theorem find_central_angle_of_sector :
  let r := 12
  let A := 52.8
  let θ := sector_angle r A
  θ ≈ 42.04 :=
by sorry

end find_central_angle_of_sector_l154_154810


namespace flip_coin_probability_l154_154033

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154033


namespace find_scalar_product_l154_154851

open Real

variables (a b : ℝ^2)

def condition1 : Prop := ‖a - 2 • b‖ = 1
def condition2 : Prop := ‖2 • a + 3 • b‖ = 1 / 3

theorem find_scalar_product (h1 : condition1 a b) (h2 : condition2 a b) :
  (5 • a - 3 • b) ⬝ (a - 9 • b) = 80 / 9 :=
sorry

end find_scalar_product_l154_154851


namespace joan_spent_on_thursday_l154_154791

theorem joan_spent_on_thursday : 
  ∀ (n : ℕ), 
  2 * (4 + n) = 18 → 
  n = 14 := 
by 
  sorry

end joan_spent_on_thursday_l154_154791


namespace solution_range_l154_154582

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l154_154582


namespace min_period_cos_sin_quot_l154_154434

noncomputable def min_positive_period (f : ℝ → ℝ) : ℝ :=
Inf {T : ℝ | 0 < T ∧ ∀ x, f (x + T) = f x}

theorem min_period_cos_sin_quot (x : ℝ) :
  min_positive_period (λ x, (Real.cos (2 * x) + Real.sin (2 * x)) / (Real.cos (2 * x) - Real.sin (2 * x))) = π / 2 :=
sorry

end min_period_cos_sin_quot_l154_154434


namespace range_of_c_minus_b_l154_154636

theorem range_of_c_minus_b (A B C a b c : ℝ) (h_triangle : A + B + C = π)
  (h_a : a = 1) (h_C_minus_B : C - B = π / 2) :
  ∃ l u : ℝ, l = sqrt 2 / 2 ∧ u = 1 ∧ l < c - b ∧ c - b < u :=
by {
  sorry
}

end range_of_c_minus_b_l154_154636


namespace correct_option_C_l154_154146

theorem correct_option_C : 
  ∃ (A B C D : Prop), 
    (A ↔ (Real.sqrt 16 = 4 ∨ Real.sqrt 16 = -4)) ∧
    (B ↔ (4 = Real.sqrt 16 ∨ -4 = Real.sqrt 16)) ∧
    (C ↔ Real.cbrt (-27) = -3) ∧
    (D ↔ Real.sqrt ((-4)^2) = -4) ∧
    C :=
by
  let A := (Real.sqrt 16 = 4 ∨ Real.sqrt 16 = -4)
  let B := (4 = Real.sqrt 16 ∨ -4 = Real.sqrt 16)
  let C := Real.cbrt (-27) = -3
  let D := Real.sqrt ((-4)^2) = -4
  have hA : ¬A := by sorry
  have hB : ¬B := by sorry
  have hC : C := by sorry
  have hD : ¬D := by sorry
  exact ⟨A, B, C, D, ⟨hA, hB, hC, hD, hC⟩⟩

end correct_option_C_l154_154146


namespace second_typist_time_for_third_chapter_l154_154465

-- Definitions of typing speeds and times
variables (x y : ℝ) -- typing speeds of the first and second typist, respectively.
variables (W : ℝ)   -- total work done, equivalently, the number of pages.

-- Conditions from the problem
def chapter1_time := 3 + 36 / 60
def chapter2_time := 8
def chapter1_work := (x + y) * 3.6
def chapter2_work := 2 * x + 6 * (x + y)

-- Mathematical relationships
def total_work := 1
def speed_sum := 1 / 3.6

-- Lean statement for the proof problem
theorem second_typist_time_for_third_chapter :
  (chapter1_time = 3.6) ∧
  (chapter1_work = W) ∧
  (chapter2_work = 1) ∧
  (speed_sum = (x + y)) →
  (W / (3 * y) = 3) :=
begin
  intros,
  sorry
end

end second_typist_time_for_third_chapter_l154_154465


namespace problem_statement_l154_154374

variables {R : Type*} [linear_ordered_field R] 

def odd_function (f : R → R) : Prop := ∀ x, f (-x) = -f x
def even_function (g : R → R) : Prop := ∀ x, g (-x) = g x
def strictly_increasing (f : R → R) : Prop := ∀ x y, x < y → f x < f y

noncomputable def f := sorry -- Define the function f somewhere
noncomputable def g := sorry -- Define the function g somewhere

theorem problem_statement 
  (a b : R) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (f_odd : odd_function f) (f_increasing : strictly_increasing f) 
  (g_even : even_function g) 
  (coincide_on_pos : ∀ x, 0 ≤ x → g x = f x) : 
  (f b - f (-a) > g a - g (-b)) ∧ (f b - f (-a) < g a - g (-b)) :=
begin
  sorry
end

end problem_statement_l154_154374


namespace machine_b_time_l154_154098

theorem machine_b_time (A_rate B_rate: ℝ)
  (A_copies_20min: A_rate * 20 = 100)
  (AB_copies_30min: A_rate * 30 + B_rate * 30 = 600) :
  150 / B_rate = 10 :=
by
  -- Given conditions
  have A_time := 20
  have total_time := 30
  let A_copies := 100
  let total_copies := 600

  -- Machine A's rate: A_rate = A_copies / A_time
  have A_rate_calculation : A_rate = 5 := A_copies / A_time

  -- Total copies made by both machines in 30 minutes
  have copies_A_30 := A_rate * 30
  have B_copies := total_copies - copies_A_30

  -- Machine B's rate: B_rate = B_copies / total_time
  have B_rate_calculation : B_rate = 450 / total_time := by
    field_simp [B_copies]

  -- Time for machine B to make 150 copies: time_B = 150 / B_rate
  have time_B: 150 / B_rate = 10 := by
    rw [B_rate_calculation]
    norm_num

  exact time_B

end machine_b_time_l154_154098


namespace sum_of_x_values_l154_154411

theorem sum_of_x_values :
  (2^(x^2 + 6*x + 9) = 16^(x + 3)) → ∃ x1 x2 : ℝ, x1 + x2 = -2 :=
by
  sorry

end sum_of_x_values_l154_154411


namespace math_problems_not_a_set_l154_154091

-- Define the conditions in Lean
def is_well_defined (α : Type) : Prop := sorry

-- Type definitions for the groups of objects
def table_tennis_players : Type := sorry
def positive_integers_less_than_5 : Type := sorry
def irrational_numbers : Type := sorry
def math_problems_2023_college_exam : Type := sorry

-- Defining specific properties of each group
def well_defined_table_tennis_players : is_well_defined table_tennis_players := sorry
def well_defined_positive_integers_less_than_5 : is_well_defined positive_integers_less_than_5 := sorry
def well_defined_irrational_numbers : is_well_defined irrational_numbers := sorry

-- The key property that math problems from 2023 college entrance examination cannot form a set.
theorem math_problems_not_a_set : ¬ is_well_defined math_problems_2023_college_exam := sorry

end math_problems_not_a_set_l154_154091


namespace no_bounded_function_exists_l154_154185

theorem no_bounded_function_exists (f : ℝ → ℝ) : (bounded (set.univ : set ℝ) f) ∧ f 1 > 0 ∧ (∀ x y : ℝ, f(x+y)^2 ≥ f(x)^2 + 2*f(x*y) + f(y)^2) → false := 
sorry

end no_bounded_function_exists_l154_154185


namespace max_sales_volume_after_6_th_day_max_profit_l154_154708

def sales_volume_first_4_days (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 4 ∧ (∃ y, y = 20 * x + 80)

def sales_volume_after_6_th_day (x t : ℕ) : Prop :=
  6 ≤ x ∧ x ≤ 20 ∧ t = (-x^2 + 50 * x - 100)

def selling_price_per_set := 28
def cost_price_per_set := 22

def profit_function (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ 5 then
    40 * x^2 + 280 * x + 480
  else if 6 ≤ x ∧ x ≤ 20 then
    -6 * (x - 25)^2 + 3150
  else
    0

theorem max_sales_volume_after_6_th_day :
  ∀ x, sales_volume_after_6_th_day x 500 → x = 20 := sorry

theorem max_profit :
  ∀ x, profit_function x = 3000 → x = 20 := sorry

end max_sales_volume_after_6_th_day_max_profit_l154_154708


namespace solution_range_l154_154579

-- Define the polynomial function and given values at specific points
def polynomial (x : ℝ) (b : ℝ) : ℝ := x^2 - b * x - 5

-- Given conditions as values of the polynomial at specific points
axiom h1 : ∀ b : ℝ, polynomial (-2) b = 5
axiom h2 : ∀ b : ℝ, polynomial (-1) b = -1
axiom h3 : ∀ b : ℝ, polynomial 4 b = -1
axiom h4 : ∀ b : ℝ, polynomial 5 b = 5

-- The range of solutions for the polynomial equation
theorem solution_range (b : ℝ) : 
  (∃ x : ℝ, -2 < x ∧ x < -1 ∧ polynomial x b = 0) ∨ 
  (∃ x : ℝ, 4 < x ∧ x < 5 ∧ polynomial x b = 0) :=
sorry

end solution_range_l154_154579


namespace approx_nine_ninety_eight_power_five_l154_154923

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem approx_nine_ninety_eight_power_five :
  let x := 10 - 0.02 in
  let k := 5 in
  let term0 := binom k 0 * x^5 in
  let term1 := binom k 1 * x^4 * (-0.02) in
  let term2 := binom k 2 * x^3 * (-0.02)^2 in
  (term0 + term1 + term2).toInt = 99004 := by
  sorry

end approx_nine_ninety_eight_power_five_l154_154923


namespace train_speed_30_kmph_l154_154537

def train_speed (time_to_cross: ℝ) (length_of_train: ℝ) : ℝ :=
  (length_of_train / time_to_cross) * (3600 / 1000)

theorem train_speed_30_kmph : 
  train_speed 9 75 = 30 :=
by 
  sorry

end train_speed_30_kmph_l154_154537


namespace integer_solution_for_system_l154_154410

theorem integer_solution_for_system 
    (x y z : ℕ) 
    (h1 : 3 * x - 4 * y + 5 * z = 10) 
    (h2 : 7 * y + 8 * x - 3 * z = 13) : 
    x = 1 ∧ y = 2 ∧ z = 3 :=
by 
  sorry

end integer_solution_for_system_l154_154410


namespace snow_probability_l154_154439

theorem snow_probability 
  (p_snow : ℚ)
  (h_psnow : p_snow = 2 / 3) :
  let p_no_snow := 1 - p_snow in
  p_no_snow ^ 4 = 1 / 81 :=
by
  sorry

end snow_probability_l154_154439


namespace geometric_series_nonnegative_l154_154397

theorem geometric_series_nonnegative (x y : ℝ) (n : ℕ) (hn : n > 0) : 
  x^(2 * n) + x^(2 * n - 1) * y + x^(2 * n - 2) * y^2 + ... + y^(2 * n) ≥ 0 :=
sorry

end geometric_series_nonnegative_l154_154397


namespace number_of_elements_in_intersection_l154_154280

def A : Set ℤ := {x | abs (x - 1) ≤ 2}
def B : Set ℤ := {x | 1 < x ∧ x ≤ 3}

theorem number_of_elements_in_intersection :
  (A ∩ B).card = 2 := by
  sorry

end number_of_elements_in_intersection_l154_154280


namespace directrix_of_parabola_l154_154965

theorem directrix_of_parabola : ∀ (x : ℝ), y = (x^2 - 8*x + 12) / 16 → ∃ (d : ℝ), d = -1/2 := 
sorry

end directrix_of_parabola_l154_154965


namespace part1_part2_l154_154269

open Real

def f (x : ℝ) : ℝ := abs (x - 2)

theorem part1 : {x : ℝ | f (x + 1) + f (x + 2) < 4} = set.Ioo (-3 / 2 : ℝ) (5 / 2 : ℝ) :=
by
  sorry

theorem part2 {a : ℝ} (h : 2 < a) : ∀ x : ℝ, f (a * x) + a * f x > 2 :=
by
  sorry

end part1_part2_l154_154269


namespace jason_tip_correct_l154_154742

def jason_tip : ℝ := 
  let c := 15.00
  let t := 0.20
  let a := 20.00
  let total := c + t * c
  a - total

theorem jason_tip_correct : jason_tip = 2.00 := by
  unfold jason_tip
  -- Definitions from conditions
  let c := 15.00
  let t := 0.20
  let a := 20.00
  let total := c + t * c
  -- Calculation
  have h1 : t * c = 3.00 := by norm_num
  have h2 : total = c + 3.00 := by rw [h1]
  have h3 : total = 18.00 := by norm_num
  have h4 : a - total = 2.00 := by norm_num
  -- Result
  exact h4

end jason_tip_correct_l154_154742


namespace part1_part2_l154_154986

variable {a b : ℝ}
def quadratic (x : ℝ) : ℝ := x^2 - (a + 2) * x + 4

theorem part1 (h1 : 1 < x ∧ x < b ∧ b > 1 ∧ quadratic x < 0 → True) :
  a = 3 ∧ b = 4 := by sorry

theorem part2 (h2 : ∀ x, 1 ≤ x ∧ x ≤ 4 → quadratic x ≥ -a - 1 → True) :
  a ≤ 4 := by sorry

end part1_part2_l154_154986


namespace cos_60_eq_half_l154_154500

theorem cos_60_eq_half :
  cos (60 * (π / 180)) = 1 / 2 :=
by
  -- the proof is omitted as per the instruction
  sorry

end cos_60_eq_half_l154_154500


namespace count_sequences_from_a_to_z_l154_154585

theorem count_sequences_from_a_to_z :
  let alphabet := list.range 26
  let is_upcase (c : ℕ) : bool := c % 2 = 1
  let next_lowercase (c : ℕ) : ℕ := if c = 0 then 0 else c - 1
  let next_uppercase (c : ℕ) : ℕ := (c + 2) % 52
  ∃ (count : ℕ), 
    count = 376 ∧ 
    ∀ (sequence : list ℕ), 
      list.length sequence = 32 → 
      list.head sequence = alphabet.head → 
      list.last sequence = alphabet.last → 
      (∀ t (ht : t ∈ list.tail sequence), 
        (is_upcase (sequence.nth_le t ht) = true →
          (sequence.nth_le (t + 1) sorry = next_lowercase (sequence.nth_le t ht)) ∨ 
          (sequence.nth_le (t + 1) sorry = next_uppercase (sequence.nth_le t ht))) ∧ 
        (is_upcase (sequence.nth_le t ht) = false →
          (sequence.nth_le (t + 1) sorry = sequence.nth_le t ht) ∨ 
          (sequence.nth_le (t + 1) sorry = next_uppercase (sequence.nth_le (t - 1) sorry))))
sorry

end count_sequences_from_a_to_z_l154_154585


namespace smallest_integer_with_16_divisors_l154_154051

-- Define prime factorization and the function to count divisors
def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def number_of_divisors (n : ℕ) : ℕ :=
  (prime_factorization n).foldr (λ ⟨_, a⟩ acc, acc * (a + 1)) 1

-- Main theorem stating the smallest positive integer with exactly 16 divisors is 210
theorem smallest_integer_with_16_divisors : 
  ∃ n, n > 0 ∧ number_of_divisors n = 16 ∧ ∀ m, m > 0 ∧ number_of_divisors m = 16 → n ≤ m :=
begin
  use 210,
  split,
  { -- Prove 210 > 0
    exact nat.zero_lt_succ _,
  },
  split,
  { -- Prove number_of_divisors 210 = 16
    sorry,
  },
  { -- Prove minimality
    intros m hm1 hm2,
    sorry,
  }
end

end smallest_integer_with_16_divisors_l154_154051


namespace solitaire_game_win_l154_154554

variables (N : ℕ → ℤ) (m : ℤ)

-- Define the vertices
def sum_vertices (N : ℕ → ℤ) : ℤ := N 0 + N 1 + N 2 + N 3 + N 4

-- Define the invariant quantity 
def S (N : ℕ → ℤ) : ℤ := (N 1 + 2 * N 2 + 3 * N 3 + 4 * N 4) % 5

-- Initial sum assumption
axiom h_sum : sum_vertices N = 2011

-- Define a turn of the solitaire game
def turn (N : ℕ → ℤ) (i j k : ℕ) (m : ℤ) : ℕ → ℤ :=
  λ n, if n = i then N i - m
    else if n = j then N j - m
    else if n = k then N k + 2 * m
    else N n

-- Define the winning condition
def win_condition (N : ℕ → ℤ) : Prop :=
  ∃ (k : ℕ), N k = 2011 ∧ ∀ (n : ℕ), n ≠ k → N n = 0

-- Define the problem statement
theorem solitaire_game_win :
  ∀ (N : ℕ → ℤ), sum_vertices N = 2011 → ∃! (k : ℕ), win_condition (turn N) k :=
sorry

end solitaire_game_win_l154_154554


namespace gasoline_used_by_car_l154_154879

noncomputable def total_gasoline_used (gasoline_per_km : ℝ) (duration_hours : ℝ) (speed_kmh : ℝ) : ℝ :=
  gasoline_per_km * duration_hours * speed_kmh

theorem gasoline_used_by_car :
  total_gasoline_used 0.14 (2 + 0.5) 93.6 = 32.76 := sorry

end gasoline_used_by_car_l154_154879


namespace urn_contains_three_balls_each_after_iterations_l154_154558

open ProbabilityTheory

-- Defining an urn operation where we draw a ball and replace it with another of the same color
inductive BallColor where
  | red | blue

def urn_operation (urn : List BallColor) (box : List BallColor) : Probability (List BallColor) :=
  let n := urn.length
  let red_count := urn.count BallColor.red
  let blue_count := urn.count BallColor.blue
  fun u => urn ++ [if u < (red_count : ℚ) / n then BallColor.red else BallColor.blue]

noncomputable def probability_same_color_distribution (initial_urn : List BallColor) (n : ℕ) : Probability (List BallColor) :=
  ProbabilityTheory.iterate urn_operation initial_urn n

def is_final_state_correct (urn : List BallColor) : Prop :=
  (urn.count BallColor.red = 3) ∧ (urn.count BallColor.blue = 3)

theorem urn_contains_three_balls_each_after_iterations :
  let initial_urn := [BallColor.red, BallColor.blue]
  let n := 4
  let final_urn_prob := probability_same_color_distribution initial_urn n
  final_urn_probT (is_final_state_correct) = 1/5
:= sorry

end urn_contains_three_balls_each_after_iterations_l154_154558


namespace range_of_m_l154_154673

def setA : Set ℝ := { x | (1/2 : ℝ) < (2 ^ x) ∧ (2 ^ x) < 8 }
def setB (m : ℝ) : Set ℝ := { x | -1 < x ∧ x < m + 1 }

theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ setB m → x ∈ setA) → m > 2 := sorry

end range_of_m_l154_154673


namespace molecular_weight_of_compound_l154_154856

def hydrogen_atomic_weight : ℝ := 1.008
def chromium_atomic_weight : ℝ := 51.996
def oxygen_atomic_weight : ℝ := 15.999

def compound_molecular_weight (h_atoms : ℕ) (cr_atoms : ℕ) (o_atoms : ℕ) : ℝ :=
  h_atoms * hydrogen_atomic_weight + cr_atoms * chromium_atomic_weight + o_atoms * oxygen_atomic_weight

theorem molecular_weight_of_compound :
  compound_molecular_weight 2 1 4 = 118.008 :=
by
  sorry

end molecular_weight_of_compound_l154_154856


namespace geometric_sequence_properties_l154_154833

noncomputable def geometric_sequence_sum (a : ℝ) (r : ℝ := a / 120) : ℝ :=
  120 + a + (a * r)

theorem geometric_sequence_properties (a : ℝ) (h_pos : 0 < a) (h_third_term : 120 * r = a)
  (h_third_term_is : a * r = 45 / 28) : 
  a = 5 * Real.sqrt (135 / 7) ∧ geometric_sequence_sum a = 184 :=
by sorry

end geometric_sequence_properties_l154_154833


namespace lake_depth_proof_l154_154886

-- Define the height of the volcano
def height_volcano : ℝ := 6000

-- Define the volume ratio above water
def volume_ratio_above : ℝ := 1 / 6

-- Define the height of the submerged part (h')
noncomputable def submerged_height : ℝ := (5 / 6)^(1 / 3) * height_volcano

-- Define the depth of the lake at the base
noncomputable def depth_lake : ℝ := height_volcano - submerged_height

-- The theorem stating the depth of the lake
theorem lake_depth_proof : depth_lake = 390 :=
by
  sorry -- Proof to be inserted here

end lake_depth_proof_l154_154886


namespace equilateral_triangle_distinct_lines_l154_154319

theorem equilateral_triangle_distinct_lines :
  ∀ (E : Type) [equilateral_triangle E], 
    let count_lines := 
      (λ (triangle : E), 
        (∑ vertex in triangle.vertices, 1)) in
    count_lines E = 3 :=
by 
  intros,
  sorry

end equilateral_triangle_distinct_lines_l154_154319


namespace rearrange_digits_divisible_by_7_l154_154723

open Function

-- Define the set of digits
def digits : Set ℕ := {1, 3, 7, 9}

-- Define a predicate for permutations of the digits set
def isPermOfDigits (d : ℕ) : Prop :=
  ∃ (perm : List ℕ), perm.perm (digits.toList) ∧ (perm.toNat = d)

theorem rearrange_digits_divisible_by_7 (a : ℕ) (ha : ∀ d ∈ digits, 0 ≤ d ∧ d < 10) : 
  ∃ d ∈ digits.toList.perms, 7 ∣ (a + d.toNat) :=
sorry

end rearrange_digits_divisible_by_7_l154_154723


namespace wash_time_difference_l154_154788

def C := 30
def T := 2 * C
def total_time := 135

theorem wash_time_difference :
  ∃ S, C + T + S = total_time ∧ T - S = 15 :=
by
  sorry

end wash_time_difference_l154_154788


namespace beth_red_pill_cost_l154_154583

noncomputable def red_pill_cost (blue_pill_cost : ℝ) : ℝ := blue_pill_cost + 3

theorem beth_red_pill_cost :
  ∃ (blue_pill_cost : ℝ), 
  (21 * (red_pill_cost blue_pill_cost + blue_pill_cost) = 966) 
  → 
  red_pill_cost blue_pill_cost = 24.5 :=
by
  sorry

end beth_red_pill_cost_l154_154583


namespace one_thirds_in_nine_halves_l154_154683

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l154_154683


namespace train_speed_in_kmph_l154_154534

theorem train_speed_in_kmph (time_to_cross: ℕ) (train_length: ℕ) (h_time : time_to_cross = 9) (h_length : train_length = 75) : 
  (train_length / time_to_cross) * 3.6 = 30 :=
by 
  rw [h_time, h_length]
  sorry

end train_speed_in_kmph_l154_154534


namespace range_of_solutions_l154_154568

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l154_154568


namespace problem_statement_l154_154772

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l154_154772


namespace midpoint_iff_equal_area_triangles_l154_154651

variable {A B C D E : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ D] [InnerProductSpace ℝ E]

def is_equal_area (P Q R S : Type) [InnerMetricSpace P]
  (h1 : Triangle Area A P Q = Triangle Area A P R) (h2 : Triangle Area A P R = Triangle Area A P S) 
  (h3 : Triangle Area A P Q = Triangle Area A P S) : Prop := 
  sorry  -- Placeholder for actual definition of equal-area property

theorem midpoint_iff_equal_area_triangles {A B C D E : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ E] 
  (h1 : is_equal_area A B E C) (h2 : is_equal_area B C E D) 
  (h3 : is_equal_area C D E A) (h4 : is_equal_area D A E B) : 
  E ∈ midpoint [A, C] ∨ E ∈ midpoint [B, D] := 
  sorry  -- Placeholder for proof

end midpoint_iff_equal_area_triangles_l154_154651


namespace pipes_volume_ratio_l154_154137

theorem pipes_volume_ratio (h : ℝ) (D1 D2 : ℝ) (V1 V2 : ℝ)
  (H1 : D1 = 12)
  (H2 : D2 = 3)
  (H3 : V1 = 36 * π * h)
  (H4 : V2 = 2.25 * π * h) :
  V1 / V2 = 16 :=
by
  rw [H3, H4]
  field_simp
  exact div_eq_of_eq_mul (by norm_num) (eq_refl _)

end pipes_volume_ratio_l154_154137


namespace correct_calculation_l154_154864

theorem correct_calculation {a x y : ℝ} :
  (a^2 * a^4 ≠ a^8) ∧
  (x ≠ 0 → -x - x ≠ 0) ∧
  ((-2 * x * y) ^ 2 = 4 * x^2 * y^2) ∧
  ((-a^3)^4 ≠ a^7) :=
by
  split
  · intro h
    -- Proof for a^2 * a^4 ≠ a^8 (left as an exercise)
    sorry
  split
  · intro hx h
    -- Proof for -x - x ≠ 0 if x ≠ 0 (left as an exercise)
    sorry
  split
  · 
    -- Proof for (-2 * x * y)^2 = 4 * x^2 * y^2 (left as an exercise)
    sorry
  · 
    -- Proof for (-a^3)^4 ≠ a^7 (left as an exercise)
    sorry

end correct_calculation_l154_154864


namespace total_wrappers_collected_l154_154560

def andy := 34
def max := 15
def zoe := 25
def mia := 19

theorem total_wrappers_collected:
  andy + max + zoe + mia = 93 :=
by
  sorry

end total_wrappers_collected_l154_154560


namespace shyam_weight_increase_l154_154495

theorem shyam_weight_increase (x : ℝ) (h1 : ∀ x, Ram = 4 * x ∧ Shyam = 5 * x) 
  (h2 : ∀ x, Ram_new = 4.4 * x)
  (h3 : ∀ x Combined_new = 82.8)
  (h4 : ∀ x Original_combined = 9 * x)
  (h5 : ∀ x Combined_new = Original_combined * 1.15)
  (hx : x = 82.8 / 10.35)
  : 
    let shyam_new := Combined_new - Ram_new in
    let shyam_old := 5 * x in
    let percent_increase := (shyam_new - shyam_old) / shyam_old * 100 in
    percent_increase = 19 :=
by
  sorry

end shyam_weight_increase_l154_154495


namespace flip_coin_probability_l154_154032

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154032


namespace find_number_l154_154894

-- Define the conditions
def satisfies_condition (x : ℝ) : Prop := x * 4 * 25 = 812

-- The main theorem stating that the number satisfying the condition is 8.12
theorem find_number (x : ℝ) (h : satisfies_condition x) : x = 8.12 :=
by
  sorry

end find_number_l154_154894


namespace polygon_perimeters_l154_154750

noncomputable def perimeter_P1 := 1985 * (2 * c / (2 * real.pi)) * real.tan (real.pi / 1985)
noncomputable def perimeter_P2 := 1985 * (2 * (c / (2 * real.pi)) * real.sin (real.pi / 1985))

theorem polygon_perimeters (h1 : ∀ θ, 0 ≤ θ ∧ θ < real.pi / 2 → real.tan θ ≥ θ) (c : ℝ) :
  perimeter_P1 + perimeter_P2 ≥ 2 * c :=
sorry

end polygon_perimeters_l154_154750


namespace intersection_of_M_and_N_l154_154354

variable {α : Type*} [LinearOrder α] [OrderBot α] [OrderTop α]

def M (x : α) : Prop := 0 < x ∧ x < 4

def N (x : α) : Prop := (1 / 3 : α) ≤ x ∧ x ≤ 5

theorem intersection_of_M_and_N {x : α} : (M x ∧ N x) ↔ ((1 / 3 : α) ≤ x ∧ x < 4) :=
by 
  sorry

end intersection_of_M_and_N_l154_154354


namespace inverse_of_i_is_negative_i_l154_154587

theorem inverse_of_i_is_negative_i : 1 / (complex.I) = - (complex.I) := by
  sorry

end inverse_of_i_is_negative_i_l154_154587


namespace coin_black_region_prob_l154_154907

structure SquareCoinDrop :=
  (side_length : ℕ := 10)
  (triangle_leg : ℕ := 3)
  (diamond_side : ℝ := 3 * Real.sqrt 2)
  (coin_diameter : ℕ := 2)

def blackRegionProbability (S : SquareCoinDrop) : ℝ :=
  let total_area := (S.side_length - S.coin_diameter) ^ 2
  let triangle_area := 4 * (1 / 2 * S.triangle_leg ^ 2)
  let diamond_area := (S.diamond_side ^ 2) / 2
  let buffer_zone_diamond := 4 * (Real.pi / 4 * 1 ^ 2 + 3 * Real.sqrt 2 * 1)
  let buffer_zone_triangle := 4 * (Real.pi / 4 * 1 ^ 2 + 3 * 1)
  let affected_area := (triangle_area + diamond_area + buffer_zone_diamond + buffer_zone_triangle)
  affected_area / total_area

theorem coin_black_region_prob (S : SquareCoinDrop) :
  blackRegionProbability S = 1/100 * (48 + 12 * Real.sqrt 2 + 2 * Real.pi) :=
sorry

end coin_black_region_prob_l154_154907


namespace intersection_M_N_l154_154346

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l154_154346


namespace largest_possible_percent_error_l154_154417

theorem largest_possible_percent_error
  (d : ℝ) (error_percent : ℝ) (actual_area : ℝ)
  (h_d : d = 30) (h_error_percent : error_percent = 0.1)
  (h_actual_area : actual_area = 225 * Real.pi) :
  ∃ max_error_percent : ℝ,
    (max_error_percent = 21) :=
by
  sorry

end largest_possible_percent_error_l154_154417


namespace find_ellipse_equation_l154_154276

theorem find_ellipse_equation (a b : ℕ) (h1 : a > b) (h2 : b > 0) (h3 : (a ^ 2) ∈ (Set.Ici 1)) (h4 : b ∈ (Set.Ici 1)) (c : ℤ) :
  (6 * c - 28 = 0) ∧ (5 * b = 20) ∧ (18 * c + 5 * b = 56) ∧ (5 * b * c = 2 * a ^ 2) ->
  (a^2 = 20) ∧ (b = 4) ∧ c = 2 ∧ (b = 4) ∧ (c = 2) ∧
  (x y : ℝ) ->
  \frac{x^2}{20} + \frac{y^2}{16} = 1 :=
begin
  sorry
end

end find_ellipse_equation_l154_154276


namespace T_n_sum_l154_154285

noncomputable def a_n (n : ℕ) : ℤ := -1 + (n - 1 : ℕ)
noncomputable def b_n (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c_n (n : ℕ) : ℕ := (n - 2) * (n - 1)

noncomputable def T_n (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), c_n i

theorem T_n_sum (n : ℕ) : T_n n = n*(n - 1)*(n - 2)/3 :=
by
  sorry

end T_n_sum_l154_154285


namespace balance_equation_l154_154841

theorem balance_equation : 
  ∀ (x y z : ℝ), 
    3 * x + 2 * y = 18 * z ∧ x = 2 * y + 3 * z → 
    4 * y = (32 / 9) * z :=
by intros x y z h
   cases h with h1 h2
   sorry

end balance_equation_l154_154841


namespace region_area_l154_154959

-- Definitions based on the problem conditions
def region (x y : ℝ) : Prop := x^6 - x^2 + y^2 ≤ 0

-- Statement proving the area of the region
theorem region_area : ∫ x in 0..1, 4 * x * sqrt(1 - x^4) = π / 2 :=
by
  sorry

end region_area_l154_154959


namespace obtuse_angle_implication_l154_154981

-- Define the vectors a and b.
def a : ℝ × ℝ × ℝ := (2, -1, 3)
def b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

-- Define the dot product of two vectors.
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- The condition for the angle between a and b being obtuse.
def angle_obtuse (x : ℝ) : Prop :=
  dot_product a (b x) < 0

-- The condition for a and b not being collinear with opposite direction.
def not_collinear (x : ℝ) : Prop :=
  3 ≠ -2 * x

-- The main theorem to be proven: for the values of x that make the angle between a and b obtuse.
theorem obtuse_angle_implication (x : ℝ) (h : angle_obtuse x) :
  x < 10 / 3 ∧ not_collinear x :=
sorry

end obtuse_angle_implication_l154_154981


namespace correct_choice_l154_154912

def shortest_path_axiom := ∀ (P Q : Type) [metric_space P], ∀ (p q : P), ∃ (path : P → P), ∀ r, (path r = p) ∨ (path r = q) → (metric.dist p q ≤ metric.dist p r + metric.dist r q)

-- Definitions for conditions
def condition_A : Prop := ∀ (N : Type) (n1 n2 : N), fix_wooden_strip_on_wall n1 n2
def condition_B : Prop := ∀ (R : Type) (r1 r2 : R), straighten_curved_road r1 r2 → shorten_distance r1 r2
def condition_C : Prop := ∀ (T : Type) (t1 t2 : T), determine_positions t1 t2 → determine_straight_line t1 t2
def condition_D : Prop := ∀ (W : Type) (w1 w2 : W), install_door_frame w1 w2

-- The problem: prove that the correct answer is B given the axiom
theorem correct_choice :
  shortest_path_axiom → condition_B :=
begin
  intro h,
  -- Proof would go here, skipped by sorry
  sorry
end

end correct_choice_l154_154912


namespace ratio_five_to_one_is_60_l154_154871

theorem ratio_five_to_one_is_60 (x : ℕ) : (5 : ℕ) * (12 : ℕ) = 1 * x → x = 60 :=
by
  intro h
  rw [mul_comm 1 x] at h
  exact h

end ratio_five_to_one_is_60_l154_154871


namespace urn_contains_three_balls_each_after_iterations_l154_154559

open ProbabilityTheory

-- Defining an urn operation where we draw a ball and replace it with another of the same color
inductive BallColor where
  | red | blue

def urn_operation (urn : List BallColor) (box : List BallColor) : Probability (List BallColor) :=
  let n := urn.length
  let red_count := urn.count BallColor.red
  let blue_count := urn.count BallColor.blue
  fun u => urn ++ [if u < (red_count : ℚ) / n then BallColor.red else BallColor.blue]

noncomputable def probability_same_color_distribution (initial_urn : List BallColor) (n : ℕ) : Probability (List BallColor) :=
  ProbabilityTheory.iterate urn_operation initial_urn n

def is_final_state_correct (urn : List BallColor) : Prop :=
  (urn.count BallColor.red = 3) ∧ (urn.count BallColor.blue = 3)

theorem urn_contains_three_balls_each_after_iterations :
  let initial_urn := [BallColor.red, BallColor.blue]
  let n := 4
  let final_urn_prob := probability_same_color_distribution initial_urn n
  final_urn_probT (is_final_state_correct) = 1/5
:= sorry

end urn_contains_three_balls_each_after_iterations_l154_154559


namespace unique_solution_count_l154_154614

theorem unique_solution_count : 
  ∃! (x y : ℝ), 32^(x^2 + y) + 32^(x + y^2) = 1 :=
begin
  sorry
end

end unique_solution_count_l154_154614


namespace max_value_among_sequence_S_n_div_2_pow_n_minus1_l154_154239

-- Define the arithmetic sequence conditions
def arithmetic_sequence_condition1 (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (2 * n) = 2 * a n - 3

def arithmetic_sequence_condition2 (a : ℕ → ℝ) :=
  a 6 ^ 2 = a 1 * a 21

-- Define the sequence Sn
def S_n (a : ℕ → ℝ) (n : ℕ) :=
  a 1 * n + (n * (n - 1) / 2) * (a 2 - a 1)

def sequence_S_n_div_2_pow_n_minus1 (S_n : ℕ → ℝ) (n : ℕ) :=
  S_n n / 2 ^ (n - 1)

-- Prove the maximum value of the term in the sequence
theorem max_value_among_sequence_S_n_div_2_pow_n_minus1 :
  ∀ (a : ℕ → ℝ), (arithmetic_sequence_condition1 a) →
  (arithmetic_sequence_condition2 a) →
  ∃ n : ℕ, sequence_S_n_div_2_pow_n_minus1 (S_n a) n = 6 :=
by
  intros a h1 h2
  use 2
  sorry  

end max_value_among_sequence_S_n_div_2_pow_n_minus1_l154_154239


namespace greatest_divisor_l154_154475

theorem greatest_divisor (d : ℕ) :
  (690 % d = 10) ∧ (875 % d = 25) ∧ ∀ e : ℕ, (690 % e = 10) ∧ (875 % e = 25) → (e ≤ d) :=
  sorry

end greatest_divisor_l154_154475


namespace union_complement_eq_l154_154781

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l154_154781


namespace more_larger_boxes_l154_154164

theorem more_larger_boxes (S L : ℕ) 
  (h1 : 12 * S + 16 * L = 480)
  (h2 : S + L = 32)
  (h3 : L > S) : L - S = 16 := 
sorry

end more_larger_boxes_l154_154164


namespace amy_files_l154_154914

theorem amy_files (music_files : ℝ) (video_files : ℝ) (picture_files : ℝ)
    (h_music : music_files = 4.0) (h_video : video_files = 21.0) (h_picture : picture_files = 23.0) :
    (music_files + video_files + picture_files) = 48.0 :=
by 
  rw [h_music, h_video, h_picture]
  exact rfl

end amy_files_l154_154914


namespace group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l154_154518

-- Question 1
theorem group_photo_arrangements {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ arrangements : ℕ, arrangements = 14400 := 
sorry

-- Question 2
theorem grouping_methods {N : ℕ} (hN : N = 8) :
  ∃ methods : ℕ, methods = 2520 := 
sorry

-- Question 3
theorem selection_methods_with_at_least_one_male {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ methods : ℕ, methods = 1560 := 
sorry

end group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l154_154518


namespace mean_minus_median_is_two_ninths_l154_154622

theorem mean_minus_median_is_two_ninths :
  let missed_days := [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6]
  let n := missed_days.length
  let mean_days := (missed_days.sum : ℚ) / n
  let median_days := list.nth_le (missed_days.sort) (n / 2) (by linarith) 
  mean_days - median_days = 2 / 9 := by
  sorry

end mean_minus_median_is_two_ninths_l154_154622


namespace units_drawn_from_model_C_l154_154888

def total_units (A B C D : ℕ) : ℕ :=
  A + B + C + D

def sampling_ratio (sample_size total_size : ℝ) : ℝ :=
  sample_size / total_size

def units_to_draw (model_units ratio : ℝ) : ℝ :=
  model_units * ratio

-- Conditions
def A_units := 200
def B_units := 400
def C_units := 300
def D_units := 100
def total_production := total_units A_units B_units C_units D_units
def sample_size := 60
def ratio := sampling_ratio sample_size total_production

-- Question and correct answer
theorem units_drawn_from_model_C : units_to_draw C_units ratio = 18 := by 
  sorry

end units_drawn_from_model_C_l154_154888


namespace molecular_weight_1_mole_l154_154045

-- Define the molecular weight of 3 moles
def molecular_weight_3_moles : ℕ := 222

-- Prove that the molecular weight of 1 mole is 74 given the molecular weight of 3 moles
theorem molecular_weight_1_mole (mw3 : ℕ) (h : mw3 = 222) : mw3 / 3 = 74 :=
by
  sorry

end molecular_weight_1_mole_l154_154045


namespace determine_a_l154_154873

theorem determine_a (a : ℝ) (x1 x2 : ℝ) :
  (x1 * x1 + (2 * a - 1) * x1 + a * a = 0) ∧
  (x2 * x2 + (2 * a - 1) * x2 + a * a = 0) ∧
  ((x1 + 2) * (x2 + 2) = 11) →
  a = -1 :=
by
  sorry

end determine_a_l154_154873


namespace last_two_non_zero_digits_of_75_factorial_l154_154603

theorem last_two_non_zero_digits_of_75_factorial : 
  ∃ (d : ℕ), d = 32 := sorry

end last_two_non_zero_digits_of_75_factorial_l154_154603


namespace trigonometric_identity_l154_154296

theorem trigonometric_identity
  (α : Real)
  (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l154_154296


namespace solve_quadratic_eq_l154_154412

theorem solve_quadratic_eq (x : ℝ) : x^2 - 2 * x - 15 = 0 ↔ (x = -3 ∨ x = 5) :=
by
  sorry

end solve_quadratic_eq_l154_154412


namespace b_1001_value_l154_154755

theorem b_1001_value (b : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)) 
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 888 = 17 + Real.sqrt 11) : 
  b 1001 = 7 * Real.sqrt 11 - 20 := sorry

end b_1001_value_l154_154755


namespace quadrilateral_is_parallelogram_l154_154635

noncomputable def circles_intersect (O₁ O₂ A B : Point) (c₁ c₂ : Circle) : Prop :=
c₁.center = O₁ ∧ O₁ ∈ c₂ ∧ c₁.contains A ∧ c₁.contains B ∧ c₂.contains A ∧ c₂.contains B

noncomputable def arbitrary_point_on_circle (P : Point) (c : Circle) : Prop :=
c.contains P

noncomputable def second_intersection (L : Line) (c : Circle) : Point := sorry

theorem quadrilateral_is_parallelogram
  (O₁ O₂ A B P X Y C : Point)
  (c₁ c₂ : Circle)
  (hl₁ : Line)
  (hl₂ : Line)
  (hl₃ : Line)
  (H₁ : circles_intersect O₁ O₂ A B c₁ c₂)
  (H₂ : arbitrary_point_on_circle P c₁)
  (H₃ : X = second_intersection (line_through B P) c₂)
  (H₄ : Y = second_intersection (line_through A P) c₂)
  (H₅ : C = second_intersection (line_through O₁ O₂) c₂) :
  is_parallelogram (X, P, Y, C) :=
sorry

end quadrilateral_is_parallelogram_l154_154635


namespace triangle_side_length_l154_154311

theorem triangle_side_length
  (A B : Real)
  (a : Real)
  (hA : A = 30 * Real.pi / 180)
  (hB : B = 105 * Real.pi / 180)
  (ha : a = 2) :
  let C := Real.pi - A - B in
  let sin_A := Real.sin A in
  let sin_C := Real.sin C in
  let c := a * sin_C / sin_A in
  c = 2 * Real.sqrt 2 :=
by
  let C := Real.pi - A - B
  let sin_A := Real.sin A
  let sin_C := Real.sin C
  let c := a * sin_C / sin_A
  sorry

end triangle_side_length_l154_154311


namespace people_after_Yoongi_l154_154116

theorem people_after_Yoongi (n k : Nat) (hk : k = 11) (hn : n = 20) :
  ∃ m : Nat, m = n - (k - 1) - 2 ∧ m = 9 :=
by
  use 9
  split
  · sorry
  · sorry

end people_after_Yoongi_l154_154116


namespace measure_of_angle_XYZ_l154_154928

section TriangleProof

-- Definitions of angles given
def angleA : ℝ := 50
def angleB : ℝ := 70
def angleC : ℝ := 60

-- Assumptions for the triangle ABC and XYZ
variables (Ω : Type) [circle Ω]
variables (A B C X Y Z : Type) [point A] [point B] [point C] [point X] [point Y] [point Z]

-- Conditions
axiom circle_condition : circumcircle Ω (triangle.mk A B C) ∧ incircle Ω (triangle.mk X Y Z)
axiom X_on_BC : lies_on X (segment.mk B C)
axiom Y_on_AB : lies_on Y (segment.mk A B)
axiom Z_on_AC : lies_on Z (segment.mk A C)

-- Triangle angle sum condition
axiom angles_sum_ABC : angleA + angleB + angleC = 180

-- Proof Goal: Prove the measure of ∠XYZ is 60°
theorem measure_of_angle_XYZ : measure_of_angle (angle.mk Y X Z) = 60 :=
by
  sorry

end TriangleProof

end measure_of_angle_XYZ_l154_154928


namespace domino_covering_l154_154594

theorem domino_covering (m n : ℕ) (m_eq : (m, n) ∈ [(5, 5), (4, 6), (3, 7), (5, 6), (3, 8)]) :
  (m * n % 2 = 1) ↔ (m = 5 ∧ n = 5) ∨ (m = 3 ∧ n = 7) :=
by
  sorry

end domino_covering_l154_154594


namespace find_k_in_expression_l154_154451

theorem find_k_in_expression :
  (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 :=
by
  sorry

end find_k_in_expression_l154_154451


namespace smallest_integer_with_16_divisors_l154_154056

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ :=
  n.factors.to_finset.prod fun p => (n.factorization p + 1)

-- Define the positive integer n which we need to prove has 16 divisors
def smallest_positive_integer_with_16_divisors (n : ℕ) : Prop :=
  num_divisors n = 16

theorem smallest_integer_with_16_divisors : ∃ n : ℕ, smallest_positive_integer_with_16_divisors n ∧ ∀ m : ℕ, m < n → ¬smallest_positive_integer_with_16_divisors m :=
by
  sorry

end smallest_integer_with_16_divisors_l154_154056


namespace y_payment_calc_l154_154460

noncomputable def weekly_payment_y (t : ℝ) : ℝ := 135 * t^2

theorem y_payment_calc :
  ∃ t : ℝ, (1500 = 130 * t^2 + 135 * t^2 + 120 * t^2) ∧ (weekly_payment_y t ≈ 526.47) := 
by
  sorry

end y_payment_calc_l154_154460


namespace total_arrangements_6_students_non_adjacent_ABC_arrangements_l154_154454

theorem total_arrangements_6_students : 
  let n := 6 in n.factorial = 720 := by
  sorry

theorem non_adjacent_ABC_arrangements :
  let total_students := 6
  let fixed_students := 3
  let total_positions := 4
  (fixed_students.factorial * (total_positions.choose fixed_students)) = 144 := by
  sorry

end total_arrangements_6_students_non_adjacent_ABC_arrangements_l154_154454


namespace union_complement_eq_l154_154785

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l154_154785


namespace N_eq_M_union_P_l154_154281

def M : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n}
def N : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n / 2}
def P : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n + 1 / 2}

theorem N_eq_M_union_P : N = M ∪ P :=
  sorry

end N_eq_M_union_P_l154_154281


namespace angle_in_quadrant_l154_154298

-- Definition of the fourth quadrant angle condition
def inFourthQuadrant (θ : ℝ) := -2 * π < θ ∧ θ < -π / 2

-- Definition of the first quadrant angle condition
def inFirstQuadrant (α : ℝ) := 0 < α ∧ α < π / 2

-- The theorem to be proved: if θ is in the fourth quadrant, then π/2 + θ is in the first quadrant
theorem angle_in_quadrant (θ : ℝ) (h : inFourthQuadrant θ) : inFirstQuadrant (π / 2 + θ) :=
by
  sorry

end angle_in_quadrant_l154_154298


namespace unique_positive_solution_exists_l154_154973

theorem unique_positive_solution_exists :
  ∃! z : ℝ, 0 < z ∧ z ≤ 1 ∧ cos (arctan (sin (arccos z))) = z := 
begin
  sorry,
end

end unique_positive_solution_exists_l154_154973


namespace unique_zero_function_l154_154378

theorem unique_zero_function {f : ℝ → ℝ} :
  (∀ x y : ℝ, f x + f y = f (f x * f y)) →
  (∀ x : ℝ, f x = 0) :=
begin
  sorry
end

end unique_zero_function_l154_154378


namespace correct_expression_l154_154149

theorem correct_expression : 
  (\sqrt[3](-27) = -3) ∧ ¬(\sqrt{16} = ± 4) ∧ ¬(±\sqrt{16} = 4) ∧ ¬(\sqrt{(-4)^2} = -4) :=
by
  -- rest of the proof here
  sorry

end correct_expression_l154_154149


namespace f_value_at_4_l154_154428

def f : ℝ → ℝ := sorry  -- Define f as a function from ℝ to ℝ

-- Specify the condition that f satisfies for all real numbers x
axiom f_condition (x : ℝ) : f (2^x) + x * f (2^(-x)) = 3

-- Statement to be proven: f(4) = -3
theorem f_value_at_4 : f 4 = -3 :=
by {
  -- Proof goes here
  sorry
}

end f_value_at_4_l154_154428


namespace coin_flip_probability_l154_154016

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l154_154016


namespace sum_of_proper_divisors_720_l154_154168

theorem sum_of_proper_divisors_720 : 
  let n := 720
  let proper_divisors_sum := (∑ d in (finset.filter (λ d, d ≠ n) (nat.divisors n)), d)
  proper_divisors_sum = 1698 :=
by
  let n := 720
  let proper_divisors_sum := (∑ d in (finset.filter (λ d, d ≠ n) (nat.divisors n)), d)
  show proper_divisors_sum = 1698
  sorry

end sum_of_proper_divisors_720_l154_154168


namespace Sarka_age_l154_154872

-- Definitions
def age_of_Sarka : ℝ := x
def age_of_Liba (x : ℝ) : ℝ := x + 3
def age_of_Eliska (x : ℝ) : ℝ := x + 8
def age_of_mother (x : ℝ) : ℝ := x + 29

-- Proposition
theorem Sarka_age (x : ℝ) (h : (x + (x + 3) + (x + 8) + (x + 29)) / 4 = 21) : x = 11 :=
by
  sorry

end Sarka_age_l154_154872


namespace graph_of_f_minus_1_is_C_l154_154431

def f (x : ℝ) : ℝ :=
if x >= -3 ∧ x <= 0 then -2 - x
else if x >= 0 ∧ x <= 2 then real.sqrt(4 - (x - 2) ^ 2) - 2
else if x >= 2 ∧ x <= 3 then 2 * (x - 2)
else 0

theorem graph_of_f_minus_1_is_C :
  (∀ x, f(x) - 1 = 
    (if x >= -3 ∧ x <= 0 then -3 - x
    else if x >= 0 ∧ x <= 2 then real.sqrt(4 - (x - 2) ^ 2) - 3
    else if x >= 2 ∧ x <= 3 then 2 * (x - 2) - 1
    else 0)) :=
by
  intros x
  simp [f]
  split_ifs
  all_goals { sorry }

end graph_of_f_minus_1_is_C_l154_154431


namespace resisting_arrest_years_l154_154388

theorem resisting_arrest_years (base_sentence_per_5000 : ℕ) (goods_stolen : ℕ) 
   (third_offense_increase_percent : ℝ) (total_sentence_years : ℕ) 
   (years_for_5000_stolen : ℕ) (years_increased : ℕ) (years_without_resist : ℕ) 
   (add_years_resist : ℕ) : 
   base_sentence_per_5000 = 1 →
   goods_stolen = 40000 →
   third_offense_increase_percent = 0.25 →
   total_sentence_years = 12 →
   years_for_5000_stolen = goods_stolen / 5000 →
   years_increased = nat.floor (third_offense_increase_percent * years_for_5000_stolen) →
   years_without_resist = years_for_5000_stolen + years_increased →
   add_years_resist = total_sentence_years - years_without_resist →
   add_years_resist = 2 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  simp at *,
  sorry
end

end resisting_arrest_years_l154_154388


namespace area_of_given_triangle_l154_154850

def line1 (x : ℝ) : ℝ := (3 / 4) * x + (3 / 4)
def line2 (x : ℝ) : ℝ := (1 / 3) * x + 2
def line3 (x y : ℝ) : Prop := x + y = 8

def pointA : ℝ × ℝ := (3, 3)
def pointB : ℝ × ℝ := (4.5, 3.5)
def pointC : ℝ × ℝ := (4.14, 3.86)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_given_triangle : area_of_triangle pointA pointB pointC = 0.36 :=
  sorry

end area_of_given_triangle_l154_154850


namespace students_with_exactly_two_talents_l154_154712

-- Variables representing the different sets of students
variable (U : Type) -- The universal set of students
variables (S D A N : Set U) -- Sets for students who can sing (S), dance (D), act (A), and no talents (N)

-- Given conditions
axiom h_total_students : Fintype.card U = 150
axiom h_no_all_three : ∀ u, ¬(u ∈ S ∧ u ∈ D ∧ u ∈ A)
axiom h_cannot_sing : Fintype.card (U \ S) = 75
axiom h_cannot_dance : Fintype.card (U \ D) = 95
axiom h_cannot_act : Fintype.card (U \ A) = 40
axiom h_no_talents : Fintype.card N = 20
axiom h_no_talents_def : N = U \ (S ∪ D ∪ A)

-- The goal to be proved
theorem students_with_exactly_two_talents : Fintype.card ((S ∩ D ∪ S ∩ A ∪ D ∩ A) \ (S ∩ D ∩ A)) = 90 := 
sorry

end students_with_exactly_two_talents_l154_154712


namespace sum_of_x_solutions_eq_twelve_l154_154618

theorem sum_of_x_solutions_eq_twelve :
  let f1 := fun (x : ℝ) => abs (x^2 - 4*x + 3)
  let f2 := fun (x : ℝ) => 2*x + 1
  let solutions := {x : ℝ | f1 x = f2 x}
  ∑ x in solutions, x = 12 :=
by
  sorry

end sum_of_x_solutions_eq_twelve_l154_154618


namespace final_board_configurations_when_Carl_wins_l154_154936

-- Definitions based on the problem's conditions
def board_size := 4
def total_moves := 8
def x_count := 4
def o_count := 4

-- The main theorem
theorem final_board_configurations_when_Carl_wins :
  ∃ (configs : Finset (Fin board_size → Fin board_size → Char)), 
    configs.card = 4950 ∧
    ∀ b ∈ configs, 
      (∃ c, (∀ i j, b(i)(j) = if c(i,j) then 'O' else 'X')) ∧
      ((count_O b = o_count) ∧ (count_X b = x_count)) ∧
      (Carl_wins b) := 
sorry

-- Definitions for counting 'X' and 'O'
def count_X (b : Fin board_size → Fin board_size → Char) : Nat := sorry 
def count_O (b : Fin board_size → Fin board_size → Char) : Nat := sorry 

-- Condition for Carl winning
def Carl_wins (b : Fin board_size → Fin board_size → Char) : Prop := sorry

end final_board_configurations_when_Carl_wins_l154_154936


namespace arithmetic_sequence_general_geometric_sequence_sum_l154_154502

theorem arithmetic_sequence_general (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d) 
  (h_a3 : a 3 = -6) 
  (h_a6 : a 6 = 0) :
  ∀ n, a n = 2 * n - 12 := 
sorry

theorem geometric_sequence_sum (a b : ℕ → ℤ) 
  (r : ℤ) 
  (S : ℕ → ℤ)
  (h_geom : ∀ n : ℕ, b (n + 1) = b n * r) 
  (h_b1 : b 1 = -8) 
  (h_b2 : b 2 = a 0 + a 1 + a 2) 
  (h_a1 : a 0 = -10) 
  (h_a2 : a 1 = -8) 
  (h_a3 : a 2 = -6) :
  ∀ n, S n = 4 * (1 - 3 ^ n) := 
sorry

end arithmetic_sequence_general_geometric_sequence_sum_l154_154502


namespace y_coordinate_of_equidistant_point_l154_154039

/-
  Definitions from the conditions:
  1. Coordinates of points C and D.
  2. The distance formula.
  3. The condition that the distances are equal.
-/

noncomputable def distance (p q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem y_coordinate_of_equidistant_point :
  ∃ y : ℝ, distance (0, y) (1, 0) = distance (0, y) (4, 3) ∧ y = 4 :=
by
  sorry

end y_coordinate_of_equidistant_point_l154_154039


namespace equation_of_hyperbola_point_on_circle_area_of_triangle_l154_154652

-- Step 1: Define initial conditions
def center_of_hyperbola := (0, 0 : ℝ)
def focal_distance_on_axes (F1 F2 : ℝ × ℝ) :=
  F1.1 = 2 * Real.sqrt 3 ∧ F1.2 = 0 ∧ F2.1 = -2 * Real.sqrt 3 ∧ F2.2 = 0
def eccentricity := Real.sqrt 2
def passes_through_point (P : ℝ × ℝ) := P = (4, -Real.sqrt 10)
def hyperbola (x y λ : ℝ) := x^2 - y^2 = λ

-- Step 2: Lean statement for part (I)
theorem equation_of_hyperbola (λ : ℝ) : 
  (passes_through_point (4, -Real.sqrt 10)) → 
  (hyperbola 4 (-Real.sqrt 10) λ) → 
  λ = 6 := 
sorry

-- Step 3: Lean statement for part (II)
theorem point_on_circle 
  (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
    M = (3, Real.sqrt 3) →
    F1.1 = 2 * Real.sqrt 3 ∧ F1.2 = 0 ∧ F2.1 = -2 * Real.sqrt 3 ∧ F2.2 = 0 →
    ((2 * Real.sqrt 3) - 3, -Real.sqrt 3) ∙ ((-2 * Real.sqrt 3) - 3, -Real.sqrt 3) = 0 :=
sorry

-- Step 4: Lean statement for part (III)
theorem area_of_triangle 
  (F1 F2 : ℝ × ℝ)
  (M : ℝ × ℝ) :
    F1 = (2 * Real.sqrt 3, 0) ∧ F2 = (-2 * Real.sqrt 3, 0) ∧ (passes_through_point M) →
    let C := 2 * Real.sqrt 3 in
    let |M| := Real.sqrt 3 in
    ∃ S : ℝ, S = (1/2) * F1.1 * F2.1 * |M| ∧ S = 6 :=
sorry

end equation_of_hyperbola_point_on_circle_area_of_triangle_l154_154652


namespace problem_statement_l154_154773

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l154_154773


namespace number_smaller_than_neg3_exists_l154_154150

def numbers := [0, -1, -5, -1/2]

theorem number_smaller_than_neg3_exists : ∃ x ∈ numbers, x < -3 :=
by
  let x := -5
  have h : x ∈ numbers := by simp [numbers]
  have h_lt : x < -3 := by norm_num
  exact ⟨x, h, h_lt⟩ -- show that -5 meets the criteria

end number_smaller_than_neg3_exists_l154_154150


namespace smallest_abs_difference_of_root_products_l154_154372

theorem smallest_abs_difference_of_root_products :
  let g (x : ℂ) := x^4 + 8 * x^3 + 18 * x^2 + 8 * x + 1 in
  ∃ (w : Fin 4 → ℂ), (∀ i, g (w i) = 0) ∧
  (∃ {a b c d : Fin 4}, {a, b, c, d} = {0, 1, 2, 3} ∧ |w a * w b - w c * w d| = 0) := by
sorry

end smallest_abs_difference_of_root_products_l154_154372


namespace g_does_not_pass_second_quadrant_l154_154667

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+5) + 4

def M : ℝ × ℝ := (-5, 5)

noncomputable def g (x : ℝ) : ℝ := -5 + (5 : ℝ)^(x)

theorem g_does_not_pass_second_quadrant (a : ℝ) (x : ℝ) 
  (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hM : f a (-5) = 5) : 
  ∀ x < 0, g x < 0 :=
by
  sorry

end g_does_not_pass_second_quadrant_l154_154667


namespace find_x_plus_y_l154_154692

theorem find_x_plus_y :
  ∀ (x y : ℝ), (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 :=
by
  intros x y h
  sorry

end find_x_plus_y_l154_154692


namespace max_contestants_l154_154416

theorem max_contestants (n : ℕ) (h1 : n = 55) (h2 : ∀ (i j : ℕ), i < j → j < n → (j - i) % 5 ≠ 4) : ∃(k : ℕ), k = 30 := 
  sorry

end max_contestants_l154_154416


namespace sum_of_g_max_min_l154_154654

-- definitions of properties based on conditions a)
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def has_max_min (f : ℝ → ℝ) (a M m : ℝ) := M = max_on (-a) a f ∧ m = min_on (-a) a f

-- statement of the theorem in Lean 4
theorem sum_of_g_max_min (f : ℝ → ℝ) (a M m : ℝ) (h1 : 0 < a) (h2 : is_odd f) (h3 : has_max_min f a M m) :
  M + m = 0 → (max_on (-a) a (λ x, f x + 2)) + (min_on (-a) a (λ x, f x + 2)) = 4 :=
by
  sorry

end sum_of_g_max_min_l154_154654


namespace final_number_is_odd_l154_154817

theorem final_number_is_odd :
  (∀ l : List ℕ, l = List.range' 1 2018 → 
   ∃ (final_num : ℕ), 
     (∀ (a b : ℕ) (l' : List ℕ), 
       a ∈ l → b ∈ l → l' = (l.erase a).erase b ++ [|a - b|] → 
       l' = [final_num]) 
     ∧ Odd final_num) :=
by
  sorry

end final_number_is_odd_l154_154817


namespace gcd_360_128_is_8_l154_154472

def gcd_360_128 : ℕ :=
  gcd 360 128

theorem gcd_360_128_is_8 : gcd_360_128 = 8 :=
  by
    -- Proof goes here (use sorry for now)
    sorry

end gcd_360_128_is_8_l154_154472


namespace find_FH_in_quadrilateral_l154_154324

noncomputable def integer_FH (FH : ℕ) : Prop :=
  ∃ (EF FG GH HE : ℕ), EF = 7 ∧ FG = 19 ∧ GH = 7 ∧ HE = 11 ∧ 12 < FH ∧ FH < 18

theorem find_FH_in_quadrilateral (EF FG GH HE : ℕ) (FH : ℕ) : EF = 7 → FG = 19 → GH = 7 → HE = 11 → 12 < FH → FH < 18 → integer_FH FH :=
by
  intros hEF hFG hGH hHE hFH_lower hFH_upper
  use [EF, FG, GH, HE]
  simp [hEF, hFG, hGH, hHE, hFH_lower, hFH_upper]
  sorry

end find_FH_in_quadrilateral_l154_154324


namespace minimum_final_percentage_to_pass_l154_154953

-- Conditions
def problem_sets : ℝ := 100
def midterm_worth : ℝ := 100
def final_worth : ℝ := 300
def perfect_problem_sets_score : ℝ := 100
def midterm1_score : ℝ := 0.60 * midterm_worth
def midterm2_score : ℝ := 0.70 * midterm_worth
def midterm3_score : ℝ := 0.80 * midterm_worth
def passing_percentage : ℝ := 0.70

-- Derived Values
def total_points_available : ℝ := problem_sets + 3 * midterm_worth + final_worth
def required_points_to_pass : ℝ := passing_percentage * total_points_available
def total_points_before_final : ℝ := perfect_problem_sets_score + midterm1_score + midterm2_score + midterm3_score
def points_needed_from_final : ℝ := required_points_to_pass - total_points_before_final

-- Proof Statement
theorem minimum_final_percentage_to_pass : 
  ∃ (final_score : ℝ), (final_score / final_worth * 100) ≥ 60 :=
by
  -- Calculations for proof
  let required_final_percentage := (points_needed_from_final / final_worth) * 100
  -- We need to show that the required percentage is at least 60%
  have : required_final_percentage = 60 := sorry
  exact Exists.intro 180 sorry

end minimum_final_percentage_to_pass_l154_154953


namespace signup_ways_l154_154874

theorem signup_ways (students groups : ℕ) (h_students : students = 5) (h_groups : groups = 3) :
  (groups ^ students = 243) :=
by
  have calculation : 3 ^ 5 = 243 := by norm_num
  rwa [h_students, h_groups]

end signup_ways_l154_154874


namespace impossible_grid_arrangement_l154_154107

theorem impossible_grid_arrangement :
  ¬ (∃ (f : ℕ → ℕ → ℕ), ∀ (m n : ℕ), m > 100 → n > 100 → (∑ i in finset.range m, ∑ j in finset.range n, f i j) % (m + n) = 0) := 
sorry

end impossible_grid_arrangement_l154_154107


namespace probability_heads_9_or_more_12_flips_l154_154021

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l154_154021


namespace range_of_m_l154_154649

def complex_in_fourth_quadrant (z: ℂ) : Prop := (z.re > 0) ∧ (z.im < 0)

theorem range_of_m (m : ℝ) (h : complex_in_fourth_quadrant ((2 : ℂ)*m^2 + (1 : ℂ)*m - (1+2*ℂ.i))) : 
  -2 < m ∧ m < -1/2 :=
by
  sorry

end range_of_m_l154_154649


namespace roots_real_and_equal_l154_154968

theorem roots_real_and_equal (a b c : ℝ) (h_eq : a = 1) (h_b : b = -4 * Real.sqrt 2) (h_c : c = 8) :
  ∃ x : ℝ, (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c = 0) :=
by
  have h_a : a = 1 := h_eq;
  have h_b : b = -4 * Real.sqrt 2 := h_b;
  have h_c : c = 8 := h_c;
  sorry

end roots_real_and_equal_l154_154968


namespace minewaska_state_park_l154_154306

variable (B H : Nat)

theorem minewaska_state_park (hikers_bike_riders_sum : H + B = 676) (hikers_more_than_bike_riders : H = B + 178) : H = 427 :=
sorry

end minewaska_state_park_l154_154306


namespace find_abc_l154_154511

theorem find_abc (t : ℝ) : 
    let x := 3 * Real.cos t + 2 * Real.sin t
    let y := 5 * Real.sin t
    let a := 1 / 9
    let b := -4 / 15
    let c := 19 / 375
    in a * x^2 + b * x * y + c * y^2 = 1 := 
by {
    sorry
}

end find_abc_l154_154511


namespace sum_series_eq_l154_154931

noncomputable def sum_series : ℕ → ℚ :=
λ n, ∑' n, (5 * n - 2) / 3^n

theorem sum_series_eq : sum_series 1 = 11 / 4 := 
sorry

end sum_series_eq_l154_154931


namespace handshakes_count_l154_154977

theorem handshakes_count (n : ℕ) (h : n = 4) : 
  ∃ k : ℕ, k = 2 ∧ 
    (∀ (A B C D : Type), set_ordered_distinct [A, B, C, D] ∧ circular (A, B, C, D)) :=
begin
  -- Proof omitted
  sorry
end

end handshakes_count_l154_154977


namespace seq_fifth_term_l154_154237

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ (a 2 = 6) ∧ (∀ n : ℕ, a (n + 2) = a (n + 1) - a n)

theorem seq_fifth_term (a : ℕ → ℤ) (h : seq a) : a 5 = -6 :=
by
  sorry

end seq_fifth_term_l154_154237


namespace flip_coin_probability_l154_154034

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l154_154034


namespace part1_solution_set_A_part2_inequality_l154_154666

def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 2)

theorem part1_solution_set_A :
  {x : ℝ | -2 < f x ∧ f x < 0} = set.Ioo (-1/2 : ℝ) (1/2 : ℝ) := sorry

theorem part2_inequality (m n : ℝ) (h1 : m ∈ set.Ioo (-1/2 : ℝ) (1/2 : ℝ)) (h2 : n ∈ set.Ioo (-1/2 : ℝ) (1/2 : ℝ)) :
  abs (1 - 4 * m * n) > 2 * abs (m - n) := sorry

end part1_solution_set_A_part2_inequality_l154_154666


namespace parallel_lines_constant_l154_154675

theorem parallel_lines_constant (a : ℝ) : 
  (∀ x y : ℝ, (a - 1) * x + 2 * y + 3 = 0 → x + a * y + 3 = 0) → a = -1 :=
by sorry

end parallel_lines_constant_l154_154675


namespace novel_corona_high_students_l154_154458

theorem novel_corona_high_students (students_know_it_all students_karen_high total_students students_novel_corona : ℕ)
  (h1 : students_know_it_all = 50)
  (h2 : students_karen_high = 3 / 5 * students_know_it_all)
  (h3 : total_students = 240)
  (h4 : students_novel_corona = total_students - (students_know_it_all + students_karen_high))
  : students_novel_corona = 160 :=
sorry

end novel_corona_high_students_l154_154458


namespace tetrahedron_cd_lengths_l154_154939

theorem tetrahedron_cd_lengths:
  ∀ (A B C D : ℝ^3) (radius : ℝ),
  dist A B = 4 ∧
  dist A C = 7 ∧ dist B C = 7 ∧
  dist A D = 8 ∧ dist B D = 8 ∧
  (on_cylinder A B C D radius) ∧
  (CD_parallel_to_axis C D) →
  ∃ (CD_length : ℝ),
  CD_length = 2 * real.sqrt(14) - real.sqrt(41) ∨
  CD_length = 2 * real.sqrt(14) + real.sqrt(41) := 
  sorry

end tetrahedron_cd_lengths_l154_154939
