import Mathlib

namespace max_servings_l472_472027

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l472_472027


namespace probability_at_least_one_each_color_in_bag_l472_472915

open BigOperators

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def prob_at_least_one_each_color : ℚ :=
  let total_ways := num_combinations 9 5
  let favorable_ways := 27 + 27 + 27 -- 3 scenarios (2R+1B+2G, 2B+1R+2G, 2G+1R+2B)
  favorable_ways / total_ways

theorem probability_at_least_one_each_color_in_bag :
  prob_at_least_one_each_color = 9 / 14 :=
by
  sorry

end probability_at_least_one_each_color_in_bag_l472_472915


namespace find_value_expression_l472_472808

theorem find_value_expression : 
  (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = (27 / 89) := 
by	sorry

end find_value_expression_l472_472808


namespace libby_quarters_left_l472_472313

theorem libby_quarters_left (initial_quarters : ℕ) (dress_cost_dollars : ℕ) (quarters_per_dollar : ℕ) 
  (h1 : initial_quarters = 160) (h2 : dress_cost_dollars = 35) (h3 : quarters_per_dollar = 4) : 
  initial_quarters - (dress_cost_dollars * quarters_per_dollar) = 20 := by
  sorry

end libby_quarters_left_l472_472313


namespace relationship_x_l472_472613

-- Define the conditions
variables (a x1 x2 x3 : ℝ)

-- The function defining the relationship y = k / x
def inverse_proportion := ∀ x : ℝ, x ≠ 0 → ∃ k : ℝ, y = k / x

-- Points A, B, and C on the given function
def on_graph (x y : ℝ) := y = ((a + 1)^2) / x

-- Points with specific y-values
def points_A := on_graph x1 (-3)
def points_B := on_graph x2 2
def points_C := on_graph x3 6

-- Proving the relationship between x1, x3, and x2
theorem relationship_x (hx1 : points_A) (hx2 : points_B) (hx3 : points_C) :
  x1 < x3 ∧ x3 < x2 :=
sorry

end relationship_x_l472_472613


namespace compute_a_plus_b_l472_472725

-- Define the volume formula for a sphere
def volume_of_sphere (r : ℝ) : ℝ := (4/3) * π * r^3

-- Given conditions
def radius_small_sphere : ℝ := 6
def volume_small_sphere := volume_of_sphere radius_small_sphere
def volume_large_sphere := 3 * volume_small_sphere

-- Radius of the larger sphere
def radius_large_sphere := (volume_large_sphere * 3 / (4 * π))^(1/3)
def diameter_large_sphere := 2 * radius_large_sphere

-- Express diameter in the form a*root(3, b)
def a : ℕ := 12
def b : ℕ := 3

-- The mathematically equivalent proof problem
theorem compute_a_plus_b : (a + b) = 15 := by
  sorry

end compute_a_plus_b_l472_472725


namespace max_servings_l472_472011

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l472_472011


namespace two_planes_perpendicular_to_same_plane_not_parallel_distance_from_point_to_line_l472_472652

/-- Math problem 1: Two planes perpendicular to the same plane are not necessarily parallel -/
theorem two_planes_perpendicular_to_same_plane_not_parallel
  (P1 P2 P3 : Plane)
  (h1 : P1 ⊥ P3)
  (h2 : P2 ⊥ P3) :
  ¬ ∀ ⦃P1 P2 : Plane⦄, (P1 ⊥ P3 ∧ P2 ⊥ P3) → P1 ∥ P2 :=
sorry

/-- Math problem 2: Distance from point A1 to line l for given angle φ between planes -/
theorem distance_from_point_to_line
  (a: ℝ)
  (phi: ℝ)
  (hphi: phi ≠ π / 2) :
  dist (point A1) (line l) = a * cot phi :=
sorry

end two_planes_perpendicular_to_same_plane_not_parallel_distance_from_point_to_line_l472_472652


namespace num_positive_divisors_not_divisible_by_3_l472_472594

theorem num_positive_divisors_not_divisible_by_3 (n : ℕ) (h : n = 180) : 
  (∃ (divisors : finset ℕ), (∀ d ∈ divisors, d ∣ n ∧ ¬ (3 ∣ d)) ∧ finset.card divisors = 6) := 
by
  have prime_factors : (n = 2^2 * 3^2 * 5) := by norm_num [h]
  sorry

end num_positive_divisors_not_divisible_by_3_l472_472594


namespace problem_1_problem_2_l472_472951

theorem problem_1 (α : ℝ) (h : sin α - 3 * cos α = 0) :
  (3 * sin α + 2 * cos α) / (4 * cos α - sin α) = 11 := 
sorry

theorem problem_2 (α : ℝ) (h : sin α - 3 * cos α = 0) :
  sin α ^ 2 + 2 * sin α * cos α + 4 = 11 / 2 := 
sorry

end problem_1_problem_2_l472_472951


namespace problem_l472_472945

open Real

theorem problem (p q : Prop)
  (hp : p ↔ ∀ x : ℝ, 2^x > 1)
  (hq : q ↔ ∃ x : ℝ, sin x = cos x) :
  ¬ p ∧ q :=
by
  sorry

end problem_l472_472945


namespace parabola_translation_vertex_l472_472767

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation of the parabola
def translated_parabola (x : ℝ) : ℝ := (x + 3)^2 - 4*(x + 3) + 2 - 2 -- Adjust x + 3 for shift left and subtract 2 for shift down

-- The vertex coordinates function
def vertex_coords (f : ℝ → ℝ) (x_vertex : ℝ) : ℝ × ℝ := (x_vertex, f x_vertex)

-- Define the original vertex
def original_vertex : ℝ × ℝ := vertex_coords original_parabola 2

-- Define the translated vertex we expect
def expected_translated_vertex : ℝ × ℝ := vertex_coords translated_parabola (-1)

-- Statement of the problem
theorem parabola_translation_vertex :
  expected_translated_vertex = (-1, -4) :=
  sorry

end parabola_translation_vertex_l472_472767


namespace find_a_value_l472_472685

noncomputable def prob_sum_equals_one (a : ℝ) : Prop :=
  a * (1/2 + 1/4 + 1/8 + 1/16) = 1

theorem find_a_value (a : ℝ) (h : prob_sum_equals_one a) : a = 16/15 :=
sorry

end find_a_value_l472_472685


namespace construct_equilateral_triangle_projection_l472_472150

-- Variables k and s
variables {k s : ℝ}

-- Set up base definitions for points and line
structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (a : α) (b : α) (c : α)

-- Given point P and line l
variables {P : Point ℝ} {l : Line ℝ}

-- Condition which ensures the base is on the line and distance from the vertex to the base is equal to the altitude of the triangle
def is_base_on_line (A B : Point ℝ) (l : Line ℝ) : Prop :=
  l.a * A.x + l.b * A.y + l.c = 0 ∧
  l.a * B.x + l.b * B.y + l.c = 0

def is_equilateral_triangle (P A B : Point ℝ) (s : ℝ) : Prop :=
  (P.x - A.x)^2 + (P.y - A.y)^2 = s^2 ∧
  (P.x - B.x)^2 + (P.y - B.y)^2 = s^2 ∧
  (A.x - B.x)^2 + (A.y - B.y)^2 = s^2

def is_valid_projection (P : Point ℝ) (l : Line ℝ) (s : ℝ) : Prop :=
  ∃ (A B : Point ℝ), is_base_on_line A B l ∧
  is_equilateral_triangle P A B s ∧
  (l.a * P.x + l.b * P.y + l.c) = -(s * (real.sqrt 3) / 2)

-- Final statement
theorem construct_equilateral_triangle_projection (P : Point ℝ) (l : Line ℝ) (s : ℝ) :
  is_valid_projection P l s :=
sorry -- Proof omitted

end construct_equilateral_triangle_projection_l472_472150


namespace third_coin_value_l472_472081

theorem third_coin_value (n : ℕ) (a b c : ℕ) (x y z : ℝ)
  (h1 : n = 20)
  (h2 : a = 1)
  (h3 : b = 0.50)
  (h4 : x + y + z = 35)
  (h5 : z = 20 * c)
  (h6 : x = 20 * a)
  (h7 : y = 20 * b) :
  c = 0.25 :=
by sorry

end third_coin_value_l472_472081


namespace Walter_allocates_75_for_school_l472_472408

/-- Walter's conditions -/
variables (days_per_week hours_per_day earnings_per_hour : ℕ) (allocation_fraction : ℝ)

/-- Given values from the problem -/
def Walter_conditions : Prop :=
  days_per_week = 5 ∧
  hours_per_day = 4 ∧
  earnings_per_hour = 5 ∧
  allocation_fraction = 3 / 4

/-- Walter's weekly earnings calculation -/
def weekly_earnings (days_per_week hours_per_day earnings_per_hour : ℕ) : ℕ :=
  days_per_week * hours_per_day * earnings_per_hour

/-- Amount allocated for school -/
def allocated_for_school (weekly_earnings : ℕ) (allocation_fraction : ℝ) : ℝ :=
  weekly_earnings * allocation_fraction

/-- Main Theorem: Walter allocates $75 for his school --/
theorem Walter_allocates_75_for_school :
  Walter_conditions days_per_week hours_per_day earnings_per_hour allocation_fraction →
  allocated_for_school (weekly_earnings days_per_week hours_per_day earnings_per_hour) allocation_fraction = 75 :=
begin
  sorry
end

end Walter_allocates_75_for_school_l472_472408


namespace arithmetic_sequence_a1_equals_20_l472_472292

theorem arithmetic_sequence_a1_equals_20
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) = a n + (-2))
  (h2 : ∀ n, S n = ∑ i in Finset.range n, a i)
  (h3 : S 10 = S 11) :
  a 0 = 20 :=
sorry

end arithmetic_sequence_a1_equals_20_l472_472292


namespace sum_of_valid_x_in_degrees_l472_472515

def sum_of_solutions (x : ℝ) : ℝ :=
  ∑ x in {x | sin (3 * x) ^ 3 + sin (5 * x) ^ 3 = 8 * sin (4 * x) ^ 3 * sin (x) ^ 3 
          ∧ 100 < x 
          ∧ x < 200}, x

theorem sum_of_valid_x_in_degrees :
  sum_of_solutions 1876 = 687 :=
sorry

end sum_of_valid_x_in_degrees_l472_472515


namespace dan_picked_9_apples_l472_472865

theorem dan_picked_9_apples (benny_apples : ℕ) (total_apples : ℕ) (dan_apples : ℕ) 
  (h1 : benny_apples = 2) (h2 : total_apples = 11) (h3 : total_apples = benny_apples + dan_apples) : 
  dan_apples = 9 :=
by
  have : 11 = 2 + dan_apples := by rw [h2, h1, h3]
  have : dan_apples = 11 - 2 := by linarith
  exact this

end dan_picked_9_apples_l472_472865


namespace price_of_small_bags_l472_472316

theorem price_of_small_bags (price_medium_bag : ℤ) (price_large_bag : ℤ) 
  (money_mark_has : ℤ) (balloons_in_small_bag : ℤ) 
  (balloons_in_medium_bag : ℤ) (balloons_in_large_bag : ℤ) 
  (total_balloons : ℤ) : 
  price_medium_bag = 6 → 
  price_large_bag = 12 → 
  money_mark_has = 24 → 
  balloons_in_small_bag = 50 → 
  balloons_in_medium_bag = 75 → 
  balloons_in_large_bag = 200 → 
  total_balloons = 400 → 
  (money_mark_has / (total_balloons / balloons_in_small_bag)) = 3 :=
by 
  sorry

end price_of_small_bags_l472_472316


namespace A_share_is_9000_l472_472111

noncomputable def A_share_in_gain (x : ℝ) : ℝ :=
  let total_gain := 27000
  let A_investment_time := 12 * x
  let B_investment_time := 6 * 2 * x
  let C_investment_time := 4 * 3 * x
  let total_investment_time := A_investment_time + B_investment_time + C_investment_time
  total_gain * A_investment_time / total_investment_time

theorem A_share_is_9000 (x : ℝ) : A_share_in_gain x = 27000 / 3 :=
by
  sorry

end A_share_is_9000_l472_472111


namespace problem_lean_statement_l472_472310

open Real

noncomputable def f (ω ϕ x : ℝ) : ℝ := 2 * sin (ω * x + ϕ)

theorem problem_lean_statement (ω ϕ : ℝ) (hω : 0 < ω) (hϕ : abs ϕ < π)
  (h1 : f ω ϕ (5 * π / 8) = 2)
  (h2 : f ω ϕ (11 * π / 8) = 0)
  (h3 : 2 * π / ω > 2 * π) : ω = 2 / 3 ∧ ϕ = π / 12 :=
by
  sorry

end problem_lean_statement_l472_472310


namespace max_servings_l472_472023

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l472_472023


namespace unique_triplet_l472_472337

-- Define the sets of numbers in each row
def row1 := {6, 8, 12, 18, 24}
def row2 := {14, 20, 28, 44, 56}
def row3 := {5, 15, 18, 27, 42}

-- Define a function that checks whether the found triplet (x, y, z) is unique
theorem unique_triplet :
  ∃ (a b c : ℕ), 
  (gcd a b ∈ row1) ∧ 
  (gcd b c ∈ row2) ∧ 
  (gcd c a ∈ row3) ∧
  (gcd a b = 8) ∧ 
  (gcd b c = 14) ∧ 
  (gcd c a = 18) ∧
  (∀ (x' y' z' : ℕ),
    (gcd x' y' ∈ row1) ∧ 
    (gcd y' z' ∈ row2) ∧ 
    (gcd z' x' ∈ row3) →
    (gcd x' y' = 8) ∧ 
    (gcd y' z' = 14) ∧ 
    (gcd z' x' = 18)) :=
by 
  sorry

end unique_triplet_l472_472337


namespace solve_integral_equation_l472_472353

noncomputable def integral_equation_solution (λ : ℂ) (h : |λ| ≠ 1) : (ℝ → ℂ) :=
  fun x => x * complex.exp (-x) + λ * ∫ t in set.Ici (0:ℝ), besselJ 0 (2 * real.sqrt (x * t)) * integral_equation_solution λ h t

theorem solve_integral_equation (λ : ℂ) (h : |λ| ≠ 1) :
  integral_equation_solution λ h = 
    fun x => complex.exp (-x) * ((x / (1 + λ)) + (λ / (1 - λ^2))) :=
sorry

end solve_integral_equation_l472_472353


namespace ellipse_condition_l472_472615

theorem ellipse_condition (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m) = 1) →
  (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_l472_472615


namespace polynomial_degree_example_l472_472780

theorem polynomial_degree_example :
  ∀ (x: ℝ), degree ((5 * x^3 + 7) ^ 10) = 30 :=
by
  sorry

end polynomial_degree_example_l472_472780


namespace medical_bills_value_l472_472128

variable (M : ℝ)
variable (property_damage : ℝ := 40000)
variable (insurance_coverage : ℝ := 0.80)
variable (carl_coverage : ℝ := 0.20)
variable (carl_owes : ℝ := 22000)

theorem medical_bills_value : 0.20 * (property_damage + M) = carl_owes → M = 70000 := 
by
  intro h
  sorry

end medical_bills_value_l472_472128


namespace money_last_weeks_l472_472687

theorem money_last_weeks (mowing_earning : ℕ) (weeding_earning : ℕ) (spending_per_week : ℕ) 
  (total_amount : ℕ) (weeks : ℕ) :
  mowing_earning = 9 →
  weeding_earning = 18 →
  spending_per_week = 3 →
  total_amount = mowing_earning + weeding_earning →
  weeks = total_amount / spending_per_week →
  weeks = 9 :=
by
  intros
  sorry

end money_last_weeks_l472_472687


namespace cosine_54_deg_l472_472136

theorem cosine_54_deg : ∃ c : ℝ, c = cos (54 : ℝ) ∧ c = 1 / 2 :=
  by 
    let c := cos (54 : ℝ)
    let d := cos (108 : ℝ)
    have h1 : d = 2 * c^2 - 1 := sorry
    have h2 : d = -c := sorry
    have h3 : 2 * c^2 + c - 1 = 0 := sorry
    use 1 / 2 
    have h4 : c = 1 / 2 := sorry
    exact ⟨cos_eq_cos_of_eq_rad 54 1, h4⟩

end cosine_54_deg_l472_472136


namespace hyperbola_eccentricity_l472_472982

-- Define the parabola and hyperbola
def parabola (x y : ℝ) : Prop :=
  y^2 = 20 * x

def hyperbola (x y a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the distance between the focus of the parabola and the asymptote
def distance (a b : ℝ) : Prop :=
  (5 * b) / (Real.sqrt (a^2 + b^2)) = 4

-- Define the eccentricity of the hyperbola
def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity (a b : ℝ) (h1 : hyperbola 0 0 a b)
    (h2 : distance a b) : eccentricity a b = 5 / 3 :=
by { sorry }

end hyperbola_eccentricity_l472_472982


namespace max_distance_l472_472932

theorem max_distance (α β : ℝ) : 
  let P := (Real.cos α, Real.sin α) in
  let Q := (Real.cos β, Real.sin β) in
  ∃ d, (∀ x y, |x - y| ≤ d) ∧ ∀ d', (∀ x y, |x - y| ≤ d') → d ≤ d' :=
begin
  let P := (Real.cos α, Real.sin α),
  let Q := (Real.cos β, Real.sin β),
  let distance := Real.sqrt ((Real.cos α - Real.cos β) ^ 2 + (Real.sin α - Real.sin β) ^ 2),
  use 2,
  split,
  { intros x y,
    rw abs_le,
    split,
    { sorry },
    { sorry }
  },
  { intros d' h,
    sorry }
end

end max_distance_l472_472932


namespace a_and_b_work_together_days_l472_472067

theorem a_and_b_work_together_days (W : ℝ) : 
  let a_rate := W / 20
  let a_and_b_rate := W / 40
  let work_by_a_in_15_days := 15 * a_rate
  ∀ (x : ℝ),
  (x * a_and_b_rate + work_by_a_in_15_days = W) → (x = 10) :=
by
  intros
  rw [← W / 40 * x, ← 15 * (W / 20)] at *
  sorry

end a_and_b_work_together_days_l472_472067


namespace villager_travel_by_motorcycle_fraction_l472_472108

theorem villager_travel_by_motorcycle_fraction {v : ℝ} (h : v > 0) :
  ∃ (x : ℝ), 
    x = 1 / 6 ∧ 
    (1 - x) / 1 = 5 / 6 :=
by {
  use (1 / 6),
  split,
  { norm_num },
  simp,
  sorry
}

end villager_travel_by_motorcycle_fraction_l472_472108


namespace libby_quarters_left_after_payment_l472_472315

noncomputable def quarters_needed (usd_target : ℝ) (usd_per_quarter : ℝ) : ℝ := 
  usd_target / usd_per_quarter

noncomputable def quarters_left (initial_quarters : ℝ) (used_quarters : ℝ) : ℝ := 
  initial_quarters - used_quarters

theorem libby_quarters_left_after_payment
  (initial_quarters : ℝ) (usd_target : ℝ) (usd_per_quarter : ℝ) 
  (h_initial : initial_quarters = 160) 
  (h_usd_target : usd_target = 35) 
  (h_usd_per_quarter : usd_per_quarter = 0.25) : 
  quarters_left initial_quarters (quarters_needed usd_target usd_per_quarter) = 20 := 
by
  sorry

end libby_quarters_left_after_payment_l472_472315


namespace alpha_beta_sum_equal_two_l472_472571

theorem alpha_beta_sum_equal_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0) 
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := 
sorry

end alpha_beta_sum_equal_two_l472_472571


namespace max_servings_l472_472015

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l472_472015


namespace ratio_of_radii_l472_472458

noncomputable def ratio_radius_truncated_cone (R r : ℝ) : ℝ :=
  (5 + Real.sqrt 21) / 2

theorem ratio_of_radii (R r s H : ℝ) :
  (s = Real.sqrt (R * r)) →
  (R^2 + R * r + r^2) * H = 12 * (R * r) * (R^0.5 * r^0.5) →
  (H = 12 * (R * r) / (R^2 + R * r + r^2)) →
  (R^2 - 5 * R * r + r^2 = 0) →
  (R / r = ratio_radius_truncated_cone R r) :=
by 
  intros h1 h2 h3 h4
  sorry

end ratio_of_radii_l472_472458


namespace polynomial_degree_example_l472_472779

theorem polynomial_degree_example :
  ∀ (x: ℝ), degree ((5 * x^3 + 7) ^ 10) = 30 :=
by
  sorry

end polynomial_degree_example_l472_472779


namespace max_sitting_people_l472_472397

/-- 
Twelve chairs are arranged in a row. A person can sit on an empty chair. 
When a person sits on a chair, exactly one of their neighbors (if any) 
stands up and leaves. Prove that the maximum number of people that can 
be sitting simultaneously is 11.
-/
theorem max_sitting_people (arrangement : Fin 12 → bool) :
  (∀ i, arrangement i = tt → (arrangement (i - 1) = ff ∨ arrangement (i + 1) = ff)) →
  ∃ n, (∃ f : Fin n → Fin 12, ∀ i, arrangement (f i) = tt) ∧ n = 11 :=
sorry

end max_sitting_people_l472_472397


namespace valid_pairs_count_is_32_l472_472236

def valid_pairs (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m^2 + n^2 < 50

def valid_pair_count : ℕ :=
  Finset.card (Finset.filter (λ p : ℕ × ℕ, valid_pairs p.1 p.2) (Finset.cartesianProduct (Finset.range 8) (Finset.range 8)))

theorem valid_pairs_count_is_32 : valid_pair_count = 32 := by
  sorry

end valid_pairs_count_is_32_l472_472236


namespace sequence_composite_l472_472338

noncomputable def sequence (n : ℕ) : ℕ := 10^(3*n) + 10^(2*n) + 10^n + 1

theorem sequence_composite (n : ℕ) : ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = sequence n :=
by
  sorry

end sequence_composite_l472_472338


namespace proof_arithmetic_sequence_proof_sum_terms_l472_472550

variable (a : ℕ → ℕ) (T : ℕ → ℚ)
variable (S_4 : ℕ) (d : ℕ) 

-- Conditions
def condition1 : Prop := S_4 = 14
def condition2 : Prop := ∃ a1 a3 a7, a1 = a 1 ∧ a3 = a 3 ∧ a7 = a 7 ∧ a3^2 = a1 * a7

-- General formula
def general_formula (n : ℕ) : Prop := a n = n + 1

-- Sum of the first n terms
def sum_terms (n : ℕ) : Prop := T n = n / (2 * (n + 2))

-- Proof statements
theorem proof_arithmetic_sequence : condition1 → condition2 → (∀ n, general_formula n) :=
by sorry

theorem proof_sum_terms (h : ∀ n, general_formula n) : ∀ n, sum_terms n :=
by sorry

end proof_arithmetic_sequence_proof_sum_terms_l472_472550


namespace initial_number_of_men_l472_472717

theorem initial_number_of_men (M A : ℕ) 
  (h1 : ((M * A) - 22 + 42 = M * (A + 2))) : M = 10 :=
by
  sorry

end initial_number_of_men_l472_472717


namespace closest_point_is_correct_l472_472909

noncomputable def line_eq (x : ℝ) : ℝ := (x + 3) / 2

def point_to_check : ℝ × ℝ := (4, 0)

-- Define the point we want to find: the closest point on the line to (4, 0)
def closest_point : ℝ × ℝ := (13 / 5, 14 / 5)

theorem closest_point_is_correct :
  ∃ x : ℝ, ∃ y : ℝ, y = line_eq x ∧ ∃ min_dist : ℝ, 
  ∀ (a b : ℝ), b = line_eq a → ((a - fst point_to_check)^2 + (b - snd point_to_check)^2)^0.5 ≥ min_dist ∧
  min_dist = ((fst closest_point - fst point_to_check)^2 + (snd closest_point - snd point_to_check)^2)^0.5 :=
sorry

end closest_point_is_correct_l472_472909


namespace pete_books_ratio_l472_472336

theorem pete_books_ratio 
  (M_last : ℝ) (P_last P_this_year M_this_year : ℝ)
  (h1 : P_last = 2 * M_last)
  (h2 : M_this_year = 1.5 * M_last)
  (h3 : P_last + P_this_year = 300)
  (h4 : M_this_year = 75) :
  P_this_year / P_last = 2 :=
by
  sorry

end pete_books_ratio_l472_472336


namespace max_diff_interior_angles_l472_472358

theorem max_diff_interior_angles
  (n : ℕ) (a : ℕ → ℕ)
  (convex_polygon : 0 < n)
  (interior_angles_int : ∀ i, 0 < i ∧ i < n → a i ∈ ℤ)
  (consecutive_diff_one : ∀ i, 0 ≤ i ∧ i < n - 1 → (a (i + 1) - a i = 1))
  (sum_exterior_angles : ∑ i in range n, (180 - a i) = 360) :
  ∃ M m : ℕ, (M = max (map a (range n))) ∧ (m = min (map a (range n))) ∧ (M - m = 18) :=
sorry

end max_diff_interior_angles_l472_472358


namespace undefined_values_count_l472_472921

theorem undefined_values_count : 
  let f : ℝ → ℝ := λ x, (x^2 - 9) / ((x^2 - 5 * x + 6) * (x + 1)) in 
  (∃ x, (x^2 - 5 * x + 6) * (x + 1) = 0) →
  { x : ℝ | (x^2 - 5 * x + 6) * (x + 1) = 0 }.finite.to_finset.card = 3 :=
by
  let D := { x : ℝ | (x^2 - 5 * x + 6) * (x + 1) = 0 }
  have hD : D = {2, 3, -1}, from sorry  -- This should be proven
  rw hD
  norm_num
  sorry

end undefined_values_count_l472_472921


namespace proper_subsets_count_of_union_l472_472931

noncomputable def a : ℕ := by sorry
noncomputable def b : ℕ := by sorry
def M : Set ℕ := {3, 2^a}
def N : Set ℕ := {a, b}

theorem proper_subsets_count_of_union (h : M ∩ N = {2}) : Finset.card (Finset.powerset (M ∪ N)) - 1 = 7 := by
  sorry

end proper_subsets_count_of_union_l472_472931


namespace find_m_l472_472213

variable (α : ℝ) (m : ℝ)

noncomputable def sqrt_m := Real.sqrt m
noncomputable def cbrt_m := Real.cbrt m

axiom terminal_side_condition : ∃ (α m : ℝ), α = 7 * π / 3 ∧ (Real.sqrt m, Real.cbrt m) = (sqrt_m, cbrt_m)

theorem find_m (α : ℝ) (m : ℝ) (h₁ : α = 7 * π / 3) 
  (h₂ : (Real.sqrt m, Real.cbrt m) = (sqrt_m, cbrt_m)) : 
  m = 1 / 27 :=
begin
  sorry
end

end find_m_l472_472213


namespace emily_olivia_books_l472_472165

theorem emily_olivia_books (shared_books total_books_emily books_olivia_not_in_emily : ℕ)
  (h1 : shared_books = 15)
  (h2 : total_books_emily = 23)
  (h3 : books_olivia_not_in_emily = 8) : (total_books_emily - shared_books + books_olivia_not_in_emily = 16) :=
by
  sorry

end emily_olivia_books_l472_472165


namespace polynomial_degree_l472_472793

def polynomial := 5 * X ^ 3 + 7
def exponent := 10
def degree_of_polynomial := 3
def final_degree := 30

theorem polynomial_degree : degree (polynomial ^ exponent) = final_degree :=
by
  sorry

end polynomial_degree_l472_472793


namespace probability_function_has_zero_point_l472_472243

noncomputable def probability_of_zero_point : ℚ :=
by
  let S := ({-1, 1, 2} : Finset ℤ).product ({-1, 1, 2} : Finset ℤ)
  let zero_point_pairs := S.filter (λ p => (p.1 * p.2 ≤ 1))
  let favorable_outcomes := zero_point_pairs.card
  let total_outcomes := S.card
  exact favorable_outcomes / total_outcomes

theorem probability_function_has_zero_point :
  probability_of_zero_point = (2 / 3 : ℚ) :=
  sorry

end probability_function_has_zero_point_l472_472243


namespace circumcircle_eqn_l472_472991

def point := ℝ × ℝ

def A : point := (-1, 5)
def B : point := (5, 5)
def C : point := (6, -2)

def circ_eq (D E F : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D * x + E * y + F = 0

theorem circumcircle_eqn :
  ∃ D E F : ℝ, (∀ (p : point), p ∈ [A, B, C] → circ_eq D E F p.1 p.2) ∧
              circ_eq (-4) (-2) (-20) = circ_eq D E F := by
  sorry

end circumcircle_eqn_l472_472991


namespace mean_of_other_two_numbers_is_2321_l472_472350

theorem mean_of_other_two_numbers_is_2321 
  (s : finset ℕ) 
  (h1 : s = {2179, 2231, 2307, 2375, 2419, 2433})
  (h2 : ∃ s₄ : finset ℕ, s₄.card = 4 ∧ (s₄.sum / s₄.card) = 2323) :
  ∃ s₂ : finset ℕ, s₂.card = 2 ∧ (s₂.sum / s₂.card) = 2321 := sorry

end mean_of_other_two_numbers_is_2321_l472_472350


namespace no_A_with_integer_roots_l472_472161

theorem no_A_with_integer_roots 
  (A : ℕ) 
  (h1 : A > 0) 
  (h2 : A < 10) 
  : ¬ ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ p + q = 10 + A ∧ p * q = 10 * A + A :=
by sorry

end no_A_with_integer_roots_l472_472161


namespace exist_secants_l472_472227

noncomputable def secant_sums_to_length (C1 C2 : Type) [circle C1] [circle C2] (O O' : Point) (AB : Line) (a : Real) (D E: Point) (F: Point) (CD EF : Line): Prop :=
  -- Conditions:
  -- C1, C2 are circles with centers O and O'
  -- AB is a given line
  -- A secant parallel to AB such that the sum of the chords cut from the circles is equal to a.
  is_parallel AB CD ∧ is_parallel AB EF ∧ 
  -- Sum of the lengths of the chords equals a
  length_of_chord C1 CD + length_of_chord C2 EF = a

-- The statement to prove the existence of such secants
theorem exist_secants
  (C1 C2 : Type) [circle C1] [circle C2] (O O' : Point) (AB : Line) (a : Real) : 
  ∃ (D E F: Point) (CD EF : Line), 
    secant_sums_to_length C1 C2 O O' AB a D E F CD EF :=
by {
  -- We will need to fill in the proof details
  sorry
}

end exist_secants_l472_472227


namespace sum_of_k_is_zero_l472_472197

noncomputable def point (x y : ℝ) := (x, y)

def on_parabola (x y : ℝ) : Prop := x^2 = 4 * y

def vector_add_zero (v1 v2 v3 : ℝ × ℝ) : Prop := v1 + v2 + v3 = (0, 0)

theorem sum_of_k_is_zero
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (hA : on_parabola x1 y1)
  (hB : on_parabola x2 y2)
  (hC : on_parabola x3 y3)
  (hF : vector_add_zero (point x1 (y1 - 1)) (point x2 (y2 - 1)) (point x3 (y3 - 1))) :
  let k_AB := √((x2 - x1)^2 + (y2 - y1)^2)
  let k_AC := √((x3 - x1)^2 + (y3 - y1)^2)
  let k_BC := √((x3 - x2)^2 + (y3 - y2)^2)
  in k_AB + k_AC + k_BC = 0 :=
sorry

end sum_of_k_is_zero_l472_472197


namespace sequence_probability_sum_prime_l472_472841

/-- 
A sequence of twelve 0s, 1s, and/or 2s is randomly generated with the condition that no two consecutive characters are the same.
If the probability that the generated sequence meets this condition can be expressed as m / n, where m, n are relatively prime positive integers,
then m + n = 179195.
-/
theorem sequence_probability_sum_prime (m n : ℕ) (hmn : Nat.coprime m n) (h : m / n = 2048 / 177147) : m + n = 179195 :=
sorry

end sequence_probability_sum_prime_l472_472841


namespace max_probability_at_k_l472_472630

noncomputable def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p) ^ (n - k))

theorem max_probability_at_k :
  ∃ k : ℕ, max (binomialProbability 5 k (1 / 4)) = binomialProbability 5 1 (1 / 4) :=
by {
  sorry
}

end max_probability_at_k_l472_472630


namespace polynomial_factor_value_l472_472160

theorem polynomial_factor_value (k : ℚ) :
  x + 6 | (k * x^3 + 21 * x^2 - 5 * k * x + 42) ↔ k = 133 / 31 :=
sorry

end polynomial_factor_value_l472_472160


namespace probability_both_in_picture_l472_472849

namespace CircularTrack

def alice_period : ℕ := 120
def bob_period : ℕ := 75
def bob_start_delay : ℕ := 15
def picture_time_window_start : ℕ := 720
def picture_time_window_end : ℕ := 780
def track_fraction : ℚ := 1 / 3

theorem probability_both_in_picture : 
  ∃ p : ℚ, p = 11 / 1200 ∧ 
  (∀ t, picture_time_window_start ≤ t ∧ t ≤ picture_time_window_end → 
    (alice_in_picture t) ∧ (bob_in_picture t)) → 
  (random_picture p) :=
sorry

def alice_position (t : ℕ) : ℕ := t % alice_period
def bob_position (t : ℕ) : ℕ := (t - bob_start_delay) % bob_period

def alice_in_picture (t : ℕ) : Prop :=
  let pos := alice_position t
  pos ≤ (track_fraction * alice_period) ∨ pos ≥ (alice_period - track_fraction * alice_period)

def bob_in_picture (t : ℕ) : Prop :=
  let pos := bob_position t
  pos ≤ (track_fraction * bob_period) ∨ pos ≥ (bob_period - track_fraction * bob_period)

noncomputable theory
def random_picture (p : ℚ) :=
  p = 11 / 1200

end CircularTrack

end probability_both_in_picture_l472_472849


namespace max_servings_l472_472030

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l472_472030


namespace problem_1_problem_2_monotonicity_problem_2_extreme_problem_3_l472_472976

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem problem_1 (x : ℝ) (h : x > Real.exp (Real.exp 1 - 1)) : f' x > Real.exp 1 := 
sorry

theorem problem_2_monotonicity (x : ℝ) :
  (∀ x, x > 1 / Real.exp 1 → ∃ ε > 0, ∀ δ, δ ∈ (x - ε, x + ε) → f' δ > 0) ∧
  (∀ x, 0 < x ∧ x < 1 / Real.exp 1 → ∃ ε > 0, ∀ δ, δ ∈ (x - ε, x + ε) → f' δ < 0) := 
sorry

theorem problem_2_extreme : 
  (∀ x, x = 1 / Real.exp 1 → f x = -1 / Real.exp 1) := 
sorry

theorem problem_3 (x1 x2 : ℝ) (h : x1 < x2) : 
  (f x2 - f x1) / (x2 - x1) < f' ((x1 + x2) / 2) := 
sorry

end problem_1_problem_2_monotonicity_problem_2_extreme_problem_3_l472_472976


namespace circles_intersecting_l472_472984

def r1 := 2
def r2 := 3
def d := ℝ

theorem circles_intersecting (h : d^2 - 6 * d + 5 < 0) : 
  (abs(r1 - r2) < d) ∧ (d < r1 + r2) :=
sorry

end circles_intersecting_l472_472984


namespace sequence_general_formula_proof_smallest_n_proof_l472_472531

def sequence_sum_condition (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 2 * a n - 2

def sequence_general_formula (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 2^n

def b (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  a n * n

def T (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, b a (i + 1)

def condition_for_n (a : ℕ → ℕ) (n : ℕ) : Prop :=
  T a n - n * 2^(n + 1) + 50 < 0

theorem sequence_general_formula_proof (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_condition : sequence_sum_condition a S) : sequence_general_formula a :=
sorry

theorem smallest_n_proof (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h_condition : sequence_sum_condition a S) : ∃ n : ℕ, condition_for_n a n ∧ ∀ m : ℕ, m < n → ¬ condition_for_n a m :=
⟨5, by sorry, by sorry⟩

end sequence_general_formula_proof_smallest_n_proof_l472_472531


namespace inequality_x2_8_over_xy_y2_l472_472705

open Real

theorem inequality_x2_8_over_xy_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^2 + 8 / (x * y) + y^2 ≥ 8 := 
sorry

end inequality_x2_8_over_xy_y2_l472_472705


namespace matching_units_digits_pages_reverse_numbering_l472_472084

theorem matching_units_digits_pages_reverse_numbering :
  (finset.univ.filter (λ x : ℕ, x % 10 = (54 - x) % 10)).card = 11 :=
sorry

end matching_units_digits_pages_reverse_numbering_l472_472084


namespace statement_II_and_IV_true_l472_472379

-- Definitions based on the problem's conditions
def AllNewEditions (P : Type) (books : P → Prop) := ∀ x, books x

-- Condition that the statement "All books in the library are new editions." is false
def NotAllNewEditions (P : Type) (books : P → Prop) := ¬ (AllNewEditions P books)

-- Statements to analyze
def SomeBookNotNewEdition (P : Type) (books : P → Prop) := ∃ x, ¬ books x
def NotAllBooksNewEditions (P : Type) (books : P → Prop) := ∃ x, ¬ books x

-- The theorem to prove
theorem statement_II_and_IV_true 
  (P : Type) 
  (books : P → Prop) 
  (h : NotAllNewEditions P books) : 
  SomeBookNotNewEdition P books ∧ NotAllBooksNewEditions P books :=
  by
    sorry

end statement_II_and_IV_true_l472_472379


namespace C_one_subject_l472_472465

variable (A B C : Prop)
variable (ExcellentInChineseA ExcellentInMathA ExcellentInEnglishA : Prop)
variable (ExcellentInChineseB ExcellentInMathB ExcellentInEnglishB : Prop)
variable (ExcellentInChineseC ExcellentInMathC ExcellentInEnglishC : Prop)

-- A's statement
axiom A_statement : ExcellentInChineseA ∨ ExcellentInMathA ∨ ExcellentInEnglishA ∧
                     ExcellentInChineseB ∨ ExcellentInMathB ∨ ExcellentInEnglishB ∧
                     ExcellentInChineseC ∨ ExcellentInMathC ∨ ExcellentInEnglishC

-- B's statement
axiom B_statement : ¬ ExcellentInEnglishB

-- C's statement
axiom C_statement : (if ExcellentInChineseB then 1 else 0) + (if ExcellentInMathB then 1 else 0) + (if ExcellentInEnglishB then 1 else 0) >
                    (if ExcellentInChineseC then 1 else 0) + (if ExcellentInMathC then 1 else 0) + (if ExcellentInEnglishC then 1 else 0)

theorem C_one_subject : ((if ExcellentInChineseB then 1 else 0) + (if ExcellentInMathB then 1 else 0) + (if ExcellentInEnglishB then 1 else 0) = 2) →
                       ((if ExcellentInChineseC then 1 else 0) + (if ExcellentInMathC then 1 else 0) + (if ExcellentInEnglishC then 1 else 0) = 1) := 
by
  intros
  -- Proof is skipped with sorry
  sorry

end C_one_subject_l472_472465


namespace percentage_increase_in_expenditure_l472_472331

-- Definitions
def original_income (I : ℝ) := I
def expenditure (I : ℝ) := 0.75 * I
def savings (I E : ℝ) := I - E
def new_income (I : ℝ) := 1.2 * I
def new_expenditure (E P : ℝ) := E * (1 + P / 100)
def new_savings (I E P : ℝ) := new_income I - new_expenditure E P

-- Theorem to prove
theorem percentage_increase_in_expenditure (I : ℝ) (P : ℝ) :
  savings I (expenditure I) * 1.5 = new_savings I (expenditure I) P →
  P = 10 :=
by
  intros h
  simp [savings, expenditure, new_income, new_expenditure, new_savings] at h
  sorry

end percentage_increase_in_expenditure_l472_472331


namespace sphere_volume_max_l472_472063

-- Definitions of the conditions
def AB : ℝ := 6
def AC : ℝ := 10
def AA₁ : ℝ := 3
def r : ℝ := (AB + sqrt (AC^2 - AB^2) - AC) / 2
def r_s : ℝ := min r (AA₁ / 2)

-- Statement of the proof problem
theorem sphere_volume_max (AB_perp_BC : True)
  (hAB : AB = 6) (hAC : AC = 10) (hAA₁ : AA₁ = 3) :
  (4 / 3 * Real.pi * r_s^3) = 9 * Real.pi / 2 :=
by sorry

end sphere_volume_max_l472_472063


namespace mat_finds_x_l472_472326

theorem mat_finds_x :
  ∃ x : ℤ, (589 + x) + (544 - x) + 80 * x = 2013 ∧ x = 11 :=
by
  have h₁ : (589 + 544) = 1133 := by norm_num
  have h₂ : 80 * 11 = 880 := by norm_num
  use 11
  split
  { calc
      (589 + 11) + (544 - 11) + 80 * 11 = 589 + 11 + 544 - 11 + 80 * 11 : by ring
      ... = 589 + 544 + 80 * 11 : by ring
      ... = 1133 + 880 : by rw [h₁, h₂]
      ... = 2013 : by norm_num }
  { refl }

end mat_finds_x_l472_472326


namespace sum_of_reciprocals_of_distances_l472_472535

theorem sum_of_reciprocals_of_distances 
  (a b : ℝ) (n : ℕ) (h1 : a > b) (h2 : b > 0) 
  (F : ℝ × ℝ) (hF : F = (-real.sqrt (a^2 - b^2), 0)) 
  (P : ℕ → ℝ × ℝ) 
  (hP : ∀i : ℕ, i < n → ((P i).1^2 / a^2 + (P i).2^2 / b^2 = 1)) 
  (angles : ℕ → ℝ)
  (hangles : ∀i : ℕ, i < n → angles i = (2 * real.pi / n) * i) : 
  (∑ i in finset.range n, 1 / ((P i).1 + a^2 / real.sqrt (a^2 - b^2))) = n * real.sqrt (a^2 - b^2) / b^2 :=
sorry

end sum_of_reciprocals_of_distances_l472_472535


namespace equilibrium_wage_increases_equilibrium_price_decreases_l472_472147

-- Define the conditions
def minimum_work_requirement (P : Prop) := P -- P represents the policy condition

-- Define the statements to be proved
theorem equilibrium_wage_increases (P : Prop) (HS : P) :
  (Supply_of_Public_Teachers < Supply_of_Public_Teachers_before) →
  (Equilibrium_Wage > Equilibrium_Wage_before) :=
by sorry

theorem equilibrium_price_decreases (P : Prop) (HS : P) :
  (Supply_of_Teachers_in_Commercial_Sector > Supply_of_Teachers_in_Commercial_Sector_before) →
  (Equilibrium_Price_of_Commercial_Education < Equilibrium_Price_of_Commercial_Education_before) :=
by sorry

end equilibrium_wage_increases_equilibrium_price_decreases_l472_472147


namespace emily_sees_ethan_total_time_l472_472166

-- Define parameters for Emily and Ethan's speeds (in meters per minute).
def EmilySpeed : ℝ := 20 * 1000 / 60
def EthanSpeed : ℝ := 15 * 1000 / 60

-- Define the initial distance between Emily and Ethan (in meters).
def initial_distance : ℝ := 600
-- Define the final distance between Emily and Ethan after overtaking (in meters).
def final_distance : ℝ := 600

-- Prove the total time Emily can see Ethan is 14 minutes (840 seconds).
theorem emily_sees_ethan_total_time :
  let relative_speed := EmilySpeed - EthanSpeed in
  let time_to_catch_up := initial_distance / relative_speed in
  let time_to_be_ahead := final_distance / relative_speed in
  time_to_catch_up + time_to_be_ahead = 14 :=
by
  -- Proof goes here.
  sorry

end emily_sees_ethan_total_time_l472_472166


namespace max_salad_servings_l472_472010

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l472_472010


namespace proof_l472_472708

variables (Cat Chicken Crab Bear Goat : ℕ)

-- Conditions:
def condition1 : Prop := (5 * Crab = 10)
def condition2 : Prop := (4 * Crab + Goat = 11)
def condition3 : Prop := (2 * Goat + Crab + 2 * Bear = 16)
def condition4 : Prop := (Cat + Bear + 2 * Goat + Crab = 13)
def condition5 : Prop := (2 * Crab + 2 * Chicken + Goat = 17)

-- The goal (Number formed):
def goal : Prop := 
  Cat = 1 ∧ Chicken = 5 ∧ Crab = 2 ∧ Bear = 4 ∧ Goat = 3

-- Proof Statement
theorem proof (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) : goal :=
by {
  sorry
}

end proof_l472_472708


namespace inequality_holds_for_all_x_l472_472418

theorem inequality_holds_for_all_x (x: ℝ) : exp(x) ≥ x * exp(1) :=
by {
  sorry
}

end inequality_holds_for_all_x_l472_472418


namespace Q_is_circumcenter_CDE_l472_472826

variables {l1 l2 : Line}
variables {ω ω1 ω2 : Circle}
variables {A B C D E Q : Point}

-- Conditions
axiom tangent_ω_l1 : tangent ω l1
axiom tangent_ω_l2 : tangent ω l2
axiom tangent_ω1_l1_at_A : tangent_at ω1 l1 A
axiom tangent_ω1_ω_at_C : tangent_at ω1 ω C
axiom tangent_ω2_l2_at_B : tangent_at ω2 l2 B
axiom tangent_ω2_ω_at_D : tangent_at ω2 ω D
axiom tangent_ω2_ω1_at_E : tangent_at ω2 ω1 E
axiom intersect_AD_BC_at_Q : intersection (line_through A D) (line_through B C) Q

-- Question
theorem Q_is_circumcenter_CDE : is_circumcenter Q C D E :=
begin
  sorry
end

end Q_is_circumcenter_CDE_l472_472826


namespace find_m_l472_472583

def vec (α : Type*) := (α × α)
def dot_product (v1 v2 : vec ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) :
  let a : vec ℝ := (1, 3)
  let b : vec ℝ := (-2, m)
  let c : vec ℝ := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  dot_product a c = 0 → m = -1 :=
by
  sorry

end find_m_l472_472583


namespace coefficient_of_x_in_first_equation_is_one_l472_472218

theorem coefficient_of_x_in_first_equation_is_one
  (x y z : ℝ)
  (h1 : x - 5 * y + 3 * z = 22 / 6)
  (h2 : 4 * x + 8 * y - 11 * z = 7)
  (h3 : 5 * x - 6 * y + 2 * z = 12)
  (h4 : x + y + z = 10) :
  (1 : ℝ) = 1 := 
by 
  sorry

end coefficient_of_x_in_first_equation_is_one_l472_472218


namespace certain_number_l472_472401

theorem certain_number (x y z : ℕ) 
  (h1 : x + y = 15) 
  (h2 : y = 7) 
  (h3 : 3 * x = z * y - 11) : 
  z = 5 :=
by sorry

end certain_number_l472_472401


namespace percentage_increase_in_expenditure_l472_472332

-- Definitions
def original_income (I : ℝ) := I
def expenditure (I : ℝ) := 0.75 * I
def savings (I E : ℝ) := I - E
def new_income (I : ℝ) := 1.2 * I
def new_expenditure (E P : ℝ) := E * (1 + P / 100)
def new_savings (I E P : ℝ) := new_income I - new_expenditure E P

-- Theorem to prove
theorem percentage_increase_in_expenditure (I : ℝ) (P : ℝ) :
  savings I (expenditure I) * 1.5 = new_savings I (expenditure I) P →
  P = 10 :=
by
  intros h
  simp [savings, expenditure, new_income, new_expenditure, new_savings] at h
  sorry

end percentage_increase_in_expenditure_l472_472332


namespace mei_fruit_basket_l472_472690

theorem mei_fruit_basket (a b c: ℕ) (h₁: a = 15) (h₂: b = 9) (h₃: c = 18) : Nat.gcd (Nat.gcd a b) c = 3 := by
  rw [h₁, h₂, h₃]
  exact Nat.gcd_assoc 15 9 18
  sorry

end mei_fruit_basket_l472_472690


namespace cos_54_eq_3_sub_sqrt_5_div_8_l472_472137

theorem cos_54_eq_3_sub_sqrt_5_div_8 :
  let x := Real.cos (Real.pi / 10) in
  let y := Real.cos (3 * Real.pi / 10) in
  y = (3 - Real.sqrt 5) / 8 :=
by
  -- Proof of the statement is omitted.
  sorry

end cos_54_eq_3_sub_sqrt_5_div_8_l472_472137


namespace max_tower_height_l472_472447

theorem max_tower_height (r0 r1 r2 r3 : ℝ) (hr0 : r0 = 100) (h0 : 0 ≤ r1 ∧ r1 ≤ r0)
  (h1 : 0 ≤ r2 ∧ r2 ≤ r1) (h2 : 0 ≤ r3 ∧ r3 ≤ r2)
  (h3 : r0 * r0 = 10000):

  r0 + sqrt (r0 * r0 - r1 * r1) + sqrt (r1 * r1 - r2 * r2) + sqrt (r2 * r2 - r3 * r3) + r3 ≤ 300 := 
begin
  sorry
end

end max_tower_height_l472_472447


namespace problem_1_problem_2_problem_3_l472_472207

-- Problem statement 1: Prove n = 8 such that the first three coefficients are in arithmetic sequence
theorem problem_1 : 
  (∃ n : ℕ, 
   ∀ k : ℕ, k < 3 
       → abs ((nat.choose n k) * ((sqrt (3 : ℚ))^(n-k)) * (1/(2 * (sqrt (3 : ℚ)))^(k))) 
       = abs ((nat.choose n (k + 1)) * ((sqrt (3 : ℚ))^(n-k-1)) * (1/(2 * (sqrt (3 : ℚ)))^(k + 1)))) 
  → n = 8 := sorry

-- Problem statement 2: Identify the term with the largest coefficient in the expansion
theorem problem_2 : 
  (∃ n : ℕ, 
   n = 8
   → ∃ r : ℕ, 
       r = nat.choose 8 4 
       → r = 4) := sorry

-- Problem statement 3: Identify the term with the largest coefficient considering the factors
theorem problem_3 : 
  (∃ n : ℕ,
   n = 8
   → ∃ r : ℕ,
       2 ≤ r 
       ∧ r ≤ 3) := sorry

end problem_1_problem_2_problem_3_l472_472207


namespace frank_has_3_cookies_l472_472926

-- The definitions and conditions based on the problem statement
def num_cookies_millie : ℕ := 4
def num_cookies_mike : ℕ := 3 * num_cookies_millie
def num_cookies_frank : ℕ := (num_cookies_mike / 2) - 3

-- The theorem stating the question and the correct answer
theorem frank_has_3_cookies : num_cookies_frank = 3 :=
by 
  -- This is where the proof steps would go, but for now we use sorry
  sorry

end frank_has_3_cookies_l472_472926


namespace find_N_l472_472508

noncomputable def M1 : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, -5], ![4, -3]]

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[-20, 5], ![16, -4]]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[4, -1], ![-4, 1]]

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ :=
  ![[16 / 7, -36 / 7], ![-12 / 7, 27 / 7]]

theorem find_N :
    (N ⬝ M1) = (A + B) := by
  sorry

end find_N_l472_472508


namespace standard_equation_of_circle_l472_472965

theorem standard_equation_of_circle
    (r : ℝ)
    (symmetric_center : ℝ × ℝ → ℝ × ℝ)
    (center : ℝ × ℝ)
    (h_sym : symmetric_center (1, 0) = center)
    (r_eq : r = 1)
    (center_eq : center = (0, 1)) :
    ∀ x y, x^2 + (y - 1)^2 = 1 :=
by
  intros x y
  have h1 : center = symmetric_center (1, 0) := h_sym
  have h2 : r = 1 := r_eq
  have h3 : center = (0, 1) := center_eq
  rw h3 at h1
  rw h2
  sorry

end standard_equation_of_circle_l472_472965


namespace correct_calculation_l472_472056

theorem correct_calculation : (sqrt 5 + sqrt 3) * (sqrt 5 - sqrt 3) = 2 :=
by {
  sorry
}

end correct_calculation_l472_472056


namespace grid_game_winner_l472_472813

theorem grid_game_winner {m n : ℕ} :
  (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") = (if (m + n) % 2 = 0 then "Second player wins" else "First player wins") := by
  sorry

end grid_game_winner_l472_472813


namespace circle_chains_positive_sum_count_l472_472439

theorem circle_chains_positive_sum_count:
  ∀ (ints : Fin 100 → ℤ), 
    (∑ i, ints i) = 1 → 
    (count_chains_positive_sum ints) = 4951 :=
by
  sorry

-- Define count_chains_positive_sum as a placeholder.
def count_chains_positive_sum (ints : Fin 100 → ℤ) : ℕ := sorry

end circle_chains_positive_sum_count_l472_472439


namespace closest_point_l472_472162

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem closest_point : 
  ∀ (points : List (ℝ × ℝ)) (target : ℝ × ℝ), 
    points = [(1,5), (2,1), (4,-3), (7,0), (-2,-1)] ->
    target = (3,3) ->
    (2,1) = List.argmin (distance target) points :=
by
  intro points target hpoints htarget
  -- We would proceed with the proof here
  sorry

end closest_point_l472_472162


namespace trapezoid_is_proposition_l472_472805

-- Define what it means to be a proposition
def is_proposition (s : String) : Prop := ∃ b : Bool, (s = "A trapezoid is a quadrilateral" ∨ s = "Construct line AB" ∨ s = "x is an integer" ∨ s = "Will it snow today?") ∧ 
  (b → s = "A trapezoid is a quadrilateral") 

-- Main proof statement
theorem trapezoid_is_proposition : is_proposition "A trapezoid is a quadrilateral" :=
  sorry

end trapezoid_is_proposition_l472_472805


namespace eccentricity_of_hyperbola_l472_472981

variable {a b : ℝ} [h₁ : a > 0] [h₂ : b > 0]

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def asymptote (x y : ℝ) : Prop :=
  b * x - a * y = 0

def line_through_focus (c : ℝ) (x y : ℝ) : Prop :=
  y = - (a / b) * (x - c)

def area_triangle (a : ℝ) (F1 F2 H : ℝ × ℝ) : Prop :=
  let (x1, y1) := F1; let (x2, y2) := F2; let (x3, y3) := H in
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2 = a^2

theorem eccentricity_of_hyperbola (F1 F2 H : ℝ × ℝ) (h₃ : hyperbola F1.1 F1.2)
    (h₄ : hyperbola F2.1 F2.2) (h₅ : asymptote H.1 H.2)
    (h₆ : line_through_focus (sqrt (a^2 + b^2)) H.1 H.2)
    (h₇ : area_triangle a F1 F2 H) : sqrt (a^2 + b^2) / a = sqrt 2 := by
  sorry

end eccentricity_of_hyperbola_l472_472981


namespace interest_rate_last_year_l472_472427

noncomputable def annualInterestRateLastYear : ℝ :=
  (9 / 100) / 1.10

theorem interest_rate_last_year (r : ℝ) :
  (9 / 100) = r * 1.10 → r = annualInterestRateLastYear :=
by
  intros h
  symmetry
  exact h
  sorry

end interest_rate_last_year_l472_472427


namespace total_turtles_rabbits_l472_472389

-- Number of turtles and rabbits on Happy Island
def turtles_happy : ℕ := 120
def rabbits_happy : ℕ := 80

-- Number of turtles and rabbits on Lonely Island
def turtles_lonely : ℕ := turtles_happy / 3
def rabbits_lonely : ℕ := turtles_lonely

-- Number of turtles and rabbits on Serene Island
def rabbits_serene : ℕ := 2 * rabbits_lonely
def turtles_serene : ℕ := (3 * rabbits_lonely) / 4

-- Number of turtles and rabbits on Tranquil Island
def turtles_tranquil : ℕ := (turtles_happy - turtles_serene) + 5
def rabbits_tranquil : ℕ := turtles_tranquil

-- Proving the total numbers
theorem total_turtles_rabbits :
    turtles_happy = 120 ∧ rabbits_happy = 80 ∧
    turtles_lonely = 40 ∧ rabbits_lonely = 40 ∧
    turtles_serene = 30 ∧ rabbits_serene = 80 ∧
    turtles_tranquil = 95 ∧ rabbits_tranquil = 95 ∧
    (turtles_happy + turtles_lonely + turtles_serene + turtles_tranquil = 285) ∧
    (rabbits_happy + rabbits_lonely + rabbits_serene + rabbits_tranquil = 295) := 
    by
        -- Here we prove each part step by step using the definitions and conditions provided above
        sorry

end total_turtles_rabbits_l472_472389


namespace power_of_m_l472_472250

theorem power_of_m (m : ℕ) (h₁ : ∀ k : ℕ, m^k % 24 = 0) (h₂ : ∀ d : ℕ, d ∣ m → d ≤ 8) : ∃ k : ℕ, m^k = 24 :=
sorry

end power_of_m_l472_472250


namespace probability_of_error_is_0_05_l472_472393

noncomputable def chi_square : ℝ :=
  50 * ((13 * 20 - 10 * 7) ^ 2 : ℕ) / ((23 : ℕ) * (27 : ℕ) * (20 : ℕ) * (30 : ℕ))

theorem probability_of_error_is_0_05 :
  (3.841 < chi_square) ∧ (chi_square < 6.635) → 0.05 = 0.05 :=
by
  have chi_square := chi_square
  sorry

end probability_of_error_is_0_05_l472_472393


namespace xy_uv_zero_l472_472246

theorem xy_uv_zero (x y u v : ℝ) (h1 : x^2 + y^2 = 1) (h2 : u^2 + v^2 = 1) (h3 : x * u + y * v = 0) : x * y + u * v = 0 :=
by
  sorry

end xy_uv_zero_l472_472246


namespace sum_a4_to_a12_l472_472291

variable (a : Nat → ℝ)
variable (S : Nat → ℝ)
variable (r : ℝ)

-- Conditions
def geometric_sequence : Prop := 
  ∀ (n : Nat), S (n+1) = S n + a (n+1)

def S3_eq_2 : Prop := S 3 = 2
def S6_eq_6 : Prop := S 6 = 6

-- Problem statement (theorem)
theorem sum_a4_to_a12 : S3_eq_2 a S ∧ S6_eq_6 a S ∧ geometric_sequence a S →
  (∑ i in Finset.range 9, a (i+4) = 28) := sorry

end sum_a4_to_a12_l472_472291


namespace division_problem_l472_472778

theorem division_problem : 0.05 / 0.0025 = 20 := 
sorry

end division_problem_l472_472778


namespace range_of_m_l472_472946

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 1 > m

def proposition_q (m : ℝ) : Prop :=
  3 - m > 1

theorem range_of_m (m : ℝ) (p_false : ¬proposition_p m) (q_true : proposition_q m) (pq_false : ¬(proposition_p m ∧ proposition_q m)) (porq_true : proposition_p m ∨ proposition_q m) : 
  1 ≤ m ∧ m < 2 := 
sorry

end range_of_m_l472_472946


namespace find_first_odd_number_l472_472701

theorem find_first_odd_number (x : ℤ)
  (h : 8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) : x = 7 :=
by
  sorry

end find_first_odd_number_l472_472701


namespace minimum_value_condition_l472_472547

variable {x y z : ℝ}

theorem minimum_value_condition :
  x + y + z = x * y + y * z + z * x →
  ∃ c, (c = -1/2) ∧ (∀ x y z : ℝ, x + y + z = xy + yz + zx → 
  (\frac{x}{x^2 + 1} + \frac{y}{y^2 + 1} + \frac{z}{z^2 + 1}) ≥ c) :=
sorry

end minimum_value_condition_l472_472547


namespace cost_of_apples_l472_472749

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l472_472749


namespace total_charge_for_trip_l472_472283

def initial_fee : ℝ := 2.25
def rate_per_mile_weekend_night : ℝ := 0.55
def time_increment_distance : ℝ := 2 / 5
def waiting_time_charge_per_minute : ℝ := 0.5
def discount_for_three_passengers : ℝ := 0.10

-- Definitions for the problem scenario
def calculate_distance_fee (distance : ℝ) : ℝ :=
  let increments := distance / time_increment_distance in
  increments * rate_per_mile_weekend_night

def calculate_waiting_fee (waiting_time_minutes : ℕ) : ℝ :=
  waiting_time_minutes * waiting_time_charge_per_minute

def calculate_discount (total : ℝ) : ℝ :=
  total * discount_for_three_passengers

def calculate_total_charge (distance : ℝ) (waiting_time_minutes : ℕ) (num_passengers : ℕ) : ℝ :=
  let distance_fee := calculate_distance_fee distance
  let waiting_fee := calculate_waiting_fee waiting_time_minutes
  let total_before_discount := initial_fee + distance_fee + waiting_fee
  let discount := calculate_discount total_before_discount
  total_before_discount - discount

-- Proof statement
theorem total_charge_for_trip : 
  calculate_total_charge 3.6 7 3 = 9.63 :=
by
  intros
  rw calculate_total_charge
  rw calculate_distance_fee
  rw calculate_waiting_fee
  rw calculate_discount
  sorry

end total_charge_for_trip_l472_472283


namespace degree_of_polynomial_power_l472_472050

open Polynomial

-- Define the polynomial and the proof problem
def polynomial : Polynomial ℤ := 5 * X^3 - 2 * X + 7

theorem degree_of_polynomial_power : degree (polynomial^10) = 30 :=
by sorry

end degree_of_polynomial_power_l472_472050


namespace jessa_needs_470_cupcakes_l472_472658

def total_cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade_class : ℕ) (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_afterschool_club : ℕ) : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) + pe_class_students + (afterschool_clubs * students_per_afterschool_club)

theorem jessa_needs_470_cupcakes :
  total_cupcakes_needed 8 40 80 2 35 = 470 :=
by
  sorry

end jessa_needs_470_cupcakes_l472_472658


namespace Walter_allocates_75_for_school_l472_472407

/-- Walter's conditions -/
variables (days_per_week hours_per_day earnings_per_hour : ℕ) (allocation_fraction : ℝ)

/-- Given values from the problem -/
def Walter_conditions : Prop :=
  days_per_week = 5 ∧
  hours_per_day = 4 ∧
  earnings_per_hour = 5 ∧
  allocation_fraction = 3 / 4

/-- Walter's weekly earnings calculation -/
def weekly_earnings (days_per_week hours_per_day earnings_per_hour : ℕ) : ℕ :=
  days_per_week * hours_per_day * earnings_per_hour

/-- Amount allocated for school -/
def allocated_for_school (weekly_earnings : ℕ) (allocation_fraction : ℝ) : ℝ :=
  weekly_earnings * allocation_fraction

/-- Main Theorem: Walter allocates $75 for his school --/
theorem Walter_allocates_75_for_school :
  Walter_conditions days_per_week hours_per_day earnings_per_hour allocation_fraction →
  allocated_for_school (weekly_earnings days_per_week hours_per_day earnings_per_hour) allocation_fraction = 75 :=
begin
  sorry
end

end Walter_allocates_75_for_school_l472_472407


namespace num_valid_pairs_l472_472172

open Nat

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define gcd and lcm conditions
def areValidPairs (a b : ℕ) : Prop :=
  gcd a b = fact 50 ∧ lcm a b = (fact 50) ^ 2

-- Number of valid pairs theorem
theorem num_valid_pairs :
  {p : ℕ × ℕ // areValidPairs p.1 p.2}.size = 32768 :=
by
  sorry

end num_valid_pairs_l472_472172


namespace round_trip_speed_ratio_l472_472083

noncomputable def boat_speed_in_still_water : ℝ := 20
noncomputable def current_speed : ℝ := 4
noncomputable def distance_each_way : ℝ := 2

theorem round_trip_speed_ratio : 
  let downstream_speed := boat_speed_in_still_water + current_speed in
  let upstream_speed := boat_speed_in_still_water - current_speed in
  let time_downstream := distance_each_way / downstream_speed in
  let time_upstream := distance_each_way / upstream_speed in
  let total_time := time_downstream + time_upstream in
  let total_distance := 2 * distance_each_way in
  let average_speed := total_distance / total_time in
  (average_speed / boat_speed_in_still_water = 24 / 25) := by
  sorry

end round_trip_speed_ratio_l472_472083


namespace circumcircles_tangent_l472_472679

-- Define the geometrical setup and conditions
variables {A B C X₁ Y₁ Z₁ X₂ Y₂ Z₂ : Point}
variables (l₁ l₂ : Line)
variables (ABC : Triangle A B C)
variables (Δ₁ Δ₂ : Triangle (Foot X₁ BC) (Foot Y₁ CA) (Foot Z₁ AB) (Foot X₂ BC) (Foot Y₂ CA) (Foot Z₂ AB))
variables [line_parallel l₁ l₂]
variables [non_degenerate Δ₁ Δ₂ : Triangle]

-- Define the circumcircles
noncomputable def ω₁ := circumcircle Δ₁
noncomputable def ω₂ := circumcircle Δ₂

-- Prove that the circumcircles are tangent to each other
theorem circumcircles_tangent :
  ∃ H : Point, H ∈ ω₁ ∧ H ∈ ω₂ :=
sorry

end circumcircles_tangent_l472_472679


namespace maximize_det_l472_472416

theorem maximize_det (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  (Matrix.det ![
    ![a, 1],
    ![1, b]
  ]) ≤ 0 :=
sorry

end maximize_det_l472_472416


namespace system_nonzero_solution_l472_472392

-- Definition of the game setup and conditions
def initial_equations (a b c : ℤ) (x y z : ℤ) : Prop :=
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0) ∧
  (a * x + b * y + c * z = 0)

-- The main proposition statement in Lean
theorem system_nonzero_solution :
  ∀ (a b c : ℤ), ∃ (x y z : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧ initial_equations a b c x y z :=
by
  sorry

end system_nonzero_solution_l472_472392


namespace trajectory_area_fixed_points_l472_472577

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem trajectory_area_fixed_points :
  ∀ (P : ℝ × ℝ), distance P (-2, 0) = 2 * distance P (1, 0) →
  ∃ (C : ℝ × ℝ) (r : ℝ), C = (2, 0) ∧ r = 2 ∧ π * r^2 = 4 * π :=
by
  intros P h
  sorry

end trajectory_area_fixed_points_l472_472577


namespace sum_of_values_l472_472513

theorem sum_of_values (x_set : Finset ℝ) (h1 : ∀ x ∈ x_set, 100 < x ∧ x < 200)
  (h2 : ∀ x ∈ x_set, real.sin ((3 * x) * (π / 180)) ^ 3 + real.sin ((5 * x) * (π / 180)) ^ 3 = 
                     8 * (real.sin ((4 * x) * (π / 180))) ^ 3 * (real.sin (x * (π / 180))) ^ 3)
  : x_set.sum id = 687 := 
sorry

end sum_of_values_l472_472513


namespace frank_cookies_l472_472927

theorem frank_cookies (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (h1 : Millie_cookies = 4)
  (h2 : Mike_cookies = 3 * Millie_cookies)
  (h3 : Frank_cookies = Mike_cookies / 2 - 3)
  : Frank_cookies = 3 := by
  sorry

end frank_cookies_l472_472927


namespace diameter_of_tripled_volume_sphere_l472_472724

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem diameter_of_tripled_volume_sphere :
  let r1 := 6
  let V1 := volume_sphere r1
  let V2 := 3 * V1
  let r2 := (V2 * 3 / (4 * Real.pi))^(1 / 3)
  let D := 2 * r2
  ∃ (a b : ℕ), (D = a * (b:ℝ)^(1 / 3) ∧ b ≠ 0 ∧ ∀ n : ℕ, n^3 ∣ b → n = 1) ∧ a + b = 15 :=
by
  sorry

end diameter_of_tripled_volume_sphere_l472_472724


namespace max_servings_l472_472032

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l472_472032


namespace log_equation_solution_l472_472712

theorem log_equation_solution {x : ℝ} (h : log 8 (2*x) + log 2 (x^3) = 13) : x = 2^3.8 := 
by sorry

end log_equation_solution_l472_472712


namespace max_value_expression_l472_472611

theorem max_value_expression (θ : ℝ) : 
  2 ≤ 5 + 3 * Real.sin θ ∧ 5 + 3 * Real.sin θ ≤ 8 → 
  (∃ θ, (14 / (5 + 3 * Real.sin θ)) = 7) := 
sorry

end max_value_expression_l472_472611


namespace tangent_line_monotonicity_range_of_a_l472_472970

noncomputable def f (x a : ℝ) := x - 1 - a * Real.log x

theorem tangent_line (a : ℝ) (h : a = 2) : 
  let f_x := f 1 2 in 
  ∃ (m : ℝ), m = -1 ∧ 
    (∀ (x : ℝ), x + f_x - 1 = 0) := 
sorry

theorem monotonicity (a : ℝ) (x : ℝ) (hx : 0 < x) : 
  ((a <= 0) → (∀ t, f t a ≥ f t 0)) ∧ 
  ((a > 0) → ( ∀ t ∈ Ioo 0 a, f t a < 0 ∧  ∀ t > a, f t a >0)) :=
sorry

theorem range_of_a (a : ℝ) (x1 x2 : ℝ)
  (hx1 : 0 < x1 ∧ x1 <= 1)
  (hx2 : 0 < x2 ∧ x2 <= 1)
  (habs : |f x1 a - f x2 a| ≤ 4 * |1 / x1 - 1 / x2|) :
  -3 ≤ a ∧ a < 0 :=
sorry

end tangent_line_monotonicity_range_of_a_l472_472970


namespace additional_days_to_complete_l472_472700

/-- 
Originally, it took 20 men working steadily 5 days to dig the foundation for a building. 
However, due to unforeseen circumstances, a new team had to take over after the foundation 
was partially dug. The new team, consisting of 30 men, works at a rate that is 80% as effective 
as the original team. How many additional days would it take the new team to complete the digging 
of the foundation?
-/

theorem additional_days_to_complete (total_man_days : ℕ) 
  (new_team_size : ℕ) 
  (new_team_effectiveness : ℝ) : 
  real := 
  let effective_man_days_per_day := new_team_size * new_team_effectiveness in 
  total_man_days / effective_man_days_per_day := 4.2 :=
sorry

end additional_days_to_complete_l472_472700


namespace range_of_m_l472_472621

-- Define the conditions
theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, (m-1) * x^2 + 2 * x + 1 = 0 → 
     (m-1 ≠ 0) ∧ 
     (4 - 4 * (m - 1) > 0)) ↔ 
    (m < 2 ∧ m ≠ 1) :=
sorry

end range_of_m_l472_472621


namespace max_servings_possible_l472_472018

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l472_472018


namespace rhombus_diagonal_l472_472769

theorem rhombus_diagonal (d1 d2 : ℝ) (area_tri : ℝ) (h1 : d1 = 15) (h2 : area_tri = 75) :
  (d1 * d2) / 2 = 2 * area_tri → d2 = 20 :=
by
  sorry

end rhombus_diagonal_l472_472769


namespace scientific_notation_willow_catkin_l472_472072

-- Definition of 0.00105
def willow_catkin_diameter : ℝ := 0.00105

-- Theorem stating that the scientific notation of 0.00105 is 1.05 x 10^(-3)
theorem scientific_notation_willow_catkin :
  willow_catkin_diameter = 1.05 * 10^(-3) :=
by
  -- proof is expected to be written here
  sorry

end scientific_notation_willow_catkin_l472_472072


namespace degree_of_polynomial10_l472_472785

-- Definition of the degree function for polynomials.
def degree (p : Polynomial ℝ) : ℕ := p.natDegree

-- Given condition: the degree of the polynomial 5x^3 + 7 is 3.
def polynomial1 := (Polynomial.C 5) * (Polynomial.X ^ 3) + (Polynomial.C 7)
axiom degree_poly1 : degree polynomial1 = 3

-- Statement to prove:
theorem degree_of_polynomial10 : degree (polynomial1 ^ 10) = 30 :=
by
  sorry

end degree_of_polynomial10_l472_472785


namespace find_m_n_l472_472675

variable {α : Type*} [AddCommGroup α] [VectorSpace ℝ α]

noncomputable def midpoint (A C : α) : α :=
  (1 / 2 : ℝ) • (A + C)

theorem find_m_n (A B C E : α) (hE : E = midpoint A C)
  (h : E - B = m • (B - A) + n • (C - A)) :
  m = -1 ∧ n = 1 / 2 := by
  sorry

end find_m_n_l472_472675


namespace degree_of_poly_l472_472788

-- Define the polynomial and its degree
def inner_poly := (5 : ℝ) * (X ^ 3) + (7 : ℝ)
def poly := inner_poly ^ 10

-- Statement to prove
theorem degree_of_poly : polynomial.degree poly = 30 :=
sorry

end degree_of_poly_l472_472788


namespace find_expression_value_l472_472526

theorem find_expression_value (x : ℝ) : 
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  have h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 := sorry
  exact h

end find_expression_value_l472_472526


namespace sum_even_natural_numbers_condition_l472_472913

def is_even (n : ℕ) : Prop := n % 2 = 0

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d : ℕ, d > 0 ∧ n % d = 0).card

theorem sum_even_natural_numbers_condition :
  ∑ n in Finset.filter (λ n, is_even n ∧ number_of_divisors n = n / 2) (Finset.range 100), n = 20 :=
sorry

end sum_even_natural_numbers_condition_l472_472913


namespace longest_side_of_triangle_l472_472106

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem longest_side_of_triangle :
  let A := (1, 3)
  let B := (4, 7)
  let C := (7, 3)
  let d1 := distance A B
  let d2 := distance A C
  let d3 := distance B C
  d1 ≤ 6 ∧ d2 ≤ 6 ∧ d3 ≤ 6 ∧ max d1 (max d2 d3) = 6 :=
by
  let A := (1, 3)
  let B := (4, 7)
  let C := (7, 3)
  let d1 := distance A B
  let d2 := distance A C
  let d3 := distance B C
  sorry

end longest_side_of_triangle_l472_472106


namespace second_new_player_weight_l472_472759

theorem second_new_player_weight :
  ∀ (X : ℕ), 7 * 94 + 110 + X = 9 * 92 → X = 60 :=
by {
  intros X h,
  have h1 : 7 * 94 = 658 := by norm_num,
  have h2 : 9 * 92 = 828 := by norm_num,
  rw [h1, h2] at h,
  calc X = 828 - 768 : by linarith
     ... = 60       : by norm_num
}

end second_new_player_weight_l472_472759


namespace num_positive_divisors_not_divisible_by_3_l472_472595

theorem num_positive_divisors_not_divisible_by_3 (n : ℕ) (h : n = 180) : 
  (∃ (divisors : finset ℕ), (∀ d ∈ divisors, d ∣ n ∧ ¬ (3 ∣ d)) ∧ finset.card divisors = 6) := 
by
  have prime_factors : (n = 2^2 * 3^2 * 5) := by norm_num [h]
  sorry

end num_positive_divisors_not_divisible_by_3_l472_472595


namespace find_a_l472_472968

theorem find_a (a b : ℝ) (h_curve : ∀ x : ℝ, (deriv (λ (x : ℝ), x^4 + a*x + 3) x) = 4*x^3 + a)
  (h_tangent : ∀ y : ℝ, (1 : ℝ) * y + b = x + b) : a = -3 := 
by
  sorry

end find_a_l472_472968


namespace sum_simplified_l472_472666

theorem sum_simplified : 
  let T := ∑ n in Finset.range 10000 \ Finset.range 2, (1 / Real.sqrt (n + 2 + Real.sqrt ((n + 2)^2 - 4)))
  in T = 139 + 59 * Real.sqrt 2 :=
by
  sorry

end sum_simplified_l472_472666


namespace inscribed_square_side_length_l472_472459

theorem inscribed_square_side_length (AC BC : ℝ) (h₀ : AC = 6) (h₁ : BC = 8) :
  ∃ x : ℝ, x = 24 / 7 :=
by
  sorry

end inscribed_square_side_length_l472_472459


namespace cos_54_deg_l472_472143

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end cos_54_deg_l472_472143


namespace find_m_l472_472979

theorem find_m (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 < 2) ∧ (m^2 + m = 2) → m = -2 :=
by
  sorry

end find_m_l472_472979


namespace q_at_two_l472_472123

def q (x : ℝ) : ℝ :=
  Real.sign (3 * x - 3) * (abs (3 * x - 3))^(1 / 4) +
  3 * Real.sign (3 * x - 3) * (abs (3 * x - 3))^(1 / 6) +
  (abs (3 * x - 3))^(1 / 8)

theorem q_at_two : q 2 = 6 := by
  sorry

end q_at_two_l472_472123


namespace circles_are_externally_tangent_l472_472570

theorem circles_are_externally_tangent
  (R : ℝ) (r : ℝ) (d : ℝ)
  (hR : R = 3) (hr : r = 1) (hd : d = 4) :
  d = R + r :=
by
  rw [hR, hr, hd]
  simp
  sorry

end circles_are_externally_tangent_l472_472570


namespace sum_x_y_eq_11_l472_472555

   variables {A B C D E : Type*} 
   variable [inner_product_space ℝ A]

   open_locale affine

   -- Define the points
   namespace vector_spaces

   variables {A B C D E : A} 

   -- Given conditions as hypotheses
   -- D is the midpoint of BC
   def is_midpoint (D B C : A) : Prop := (D : ℝ) = (B + C) / 2

   -- Given vector equation CD = 3*CE - 2*CA
   def vector_eq (C D E A : A) : Prop :=
     (D - C : ℝ) = 3 * (E - C) - 2 * (A - C)

   -- To prove x + y = 11
   theorem sum_x_y_eq_11 (h1 : is_midpoint D B C) 
                         (h2 : vector_eq C D E A) 
                         (h3 : (A - C : ℝ) = x * (B - C) + y * (E - B)) :
                         x + y = 11 :=
     sorry

   end vector_spaces
   
end sum_x_y_eq_11_l472_472555


namespace colinear_vectors_x_value_l472_472579

theorem colinear_vectors_x_value :
  let a1 := 1
  let a2 := Real.sqrt (1 + Real.sin 20)
  let b1 := 1 / Real.sin 55
  ∀ (x : ℝ), (a1, a2) = (k * b1, k * x) for some k  =>  x = Real.sqrt 2 :=
sorry

end colinear_vectors_x_value_l472_472579


namespace car_speed_is_80_l472_472068

theorem car_speed_is_80 
  (d : ℝ) (t_delay : ℝ) (v_train_factor : ℝ)
  (t_car t_train : ℝ) (v : ℝ) :
  ((d = 75) ∧ (t_delay = 12.5 / 60) ∧ (v_train_factor = 1.5) ∧ 
   (d = v * t_car) ∧ (d = v_train_factor * v * (t_car - t_delay))) →
  v = 80 := 
sorry

end car_speed_is_80_l472_472068


namespace find_k_values_l472_472167

theorem find_k_values :
    ∀ (k : ℚ),
    (∀ (a b : ℚ), (5 * a^2 + 7 * a + k = 0) ∧ (5 * b^2 + 7 * b + k = 0) ∧ |a - b| = a^2 + b^2 → k = 21 / 25 ∨ k = -21 / 25) :=
by
  sorry

end find_k_values_l472_472167


namespace prime_not_fourth_power_l472_472895

theorem prime_not_fourth_power (p : ℕ) (hp : p > 5) (prime : Prime p) : 
  ¬ ∃ a : ℕ, p = a^4 + 4 :=
by
  sorry

end prime_not_fourth_power_l472_472895


namespace two_element_intersection_three_element_intersection_l472_472987

noncomputable def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def A (a : ℝ) : Set (ℝ × ℝ) := {p | a * p.1 + p.2 = 1}

def B (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 + a * p.2 = 1}

def A_union_B (a : ℝ) : Set (ℝ × ℝ) := A a ∪ B a

def intersection (a : ℝ) : Set (ℝ × ℝ) := A_union_B a ∩ C

theorem two_element_intersection (a : ℝ) :
  ∃ p q : (ℝ × ℝ), p ≠ q ∧ intersection a = {p, q} ↔ a = 0 ∨ a = 1 := 
sorry

theorem three_element_intersection (a : ℝ) :
  ∃ p q r : (ℝ × ℝ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ intersection a = {p, q, r} ↔ (a ≠ 0 ∧ a ≠ 1 ∧ (a = -1 + sqrt 2 ∨ a = -1 - sqrt 2)) :=
sorry

end two_element_intersection_three_element_intersection_l472_472987


namespace find_original_number_l472_472423

theorem find_original_number (x : ℤ) 
  (h1 : odd (3 * x)) 
  (h2 : 9 ∣ (3 * x)) 
  (h3 : 4 * x = 108) : 
  x = 27 := by
  sorry

end find_original_number_l472_472423


namespace magnitude_of_angle_B_area_of_triangle_l472_472626

-- Definitions based on given conditions
variable {A B C α β γ : ℝ}
variable {a b c : ℝ}
variable (ΔABC : Triangle)
variable (h1 : b = Real.sqrt 13)
variable (h2 : a + c = 4)
variable (h3 : cos B / b + cos C / (2 * a + c) = 0)

-- Proof problems based on questions and answers derived:
theorem magnitude_of_angle_B (h3 : cos B / b + cos C / (2 * a + c) = 0) : 
  B = 2 * Real.pi / 3 := 
sorry

theorem area_of_triangle (h1 : b = Real.sqrt 13) (h2 : a + c = 4) (h3 : B = 2 * Real.pi / 3) 
  : (½ * a * c * sin B) = 3 * Real.sqrt 3 / 4 :=
sorry

end magnitude_of_angle_B_area_of_triangle_l472_472626


namespace compute_fraction_sum_l472_472874

theorem compute_fraction_sum :
  8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 :=
by
  sorry

end compute_fraction_sum_l472_472874


namespace proof_l472_472653

section

variable (f : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 

-- Condition 1: The quadratic function passes through the origin
def quad_at_origin : Prop := f 0 = 0

-- Condition 2: The derivative is given
def quad_derivative (x : ℝ) : Prop := deriv f x = 6 * x - 2

-- Condition 3: Sum of first n terms of the sequence {a_n} is s_n
def sum_first_n_terms (s : ℕ → ℝ) (n : ℕ) : Prop := 
s n = ∑ k in range n.succ, a k

-- Given each (n, s_n) lies on the graph of y = f(x)
def point_on_graph (s : ℕ → ℝ) (n : ℕ) : Prop := s n = f n

-- Question 1: f(x), a_n
def fx (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x
def an (n : ℕ) : Prop := a n = 6 * n - 5

-- Question 2: bn and Tn
def bn (n : ℕ) : ℝ := 3 / (a n * a (n + 1))
def tn_formula : ℕ → ℝ := λ n, (1 / 2 : ℝ) * (1 - 1 / (6 * n + 1))

-- The final inequality for Tn
def tn_bound : ℕ → Prop := λ n, (3 / 7 : ℝ) ≤ T n ∧ T n < (1 / 2 : ℝ)

-- Lean statement for the proof
theorem proof (s : ℕ → ℝ) (n : ℕ) :
  quad_at_origin f →
  quad_derivative f →
  sum_first_n_terms s n →
  point_on_graph f s n →
  (∀ n, an n) →
  (∀ n, b n = bn n) →
  (∀ n, T n = ∑ k in range n.succ, b k) →
  (∀ n, T n = tn_formula n) →
  tn_bound n :=
by
  intros
  sorry

end

end proof_l472_472653


namespace ad_squared_ag_ab_ad_equals_ae_l472_472473

-- Define the setup of the triangle and circle as stated in the problem
variable {ABC : Type*} [nio_triangle : acute_triangle ABC]
variable (O : Type*) [circle : Circle (diameter BC)]
variable (AD : Tangent (Circumcircle O) (Point D (Circle O)))
variable (E : Point AB)
variable (F : Perpendicular_to_line (Extend AC) E)
variable (G : Intersection (Line AB) (Circle O))
variable (H : Intersection (AF ExtendAC))

-- The first part of the problem
theorem ad_squared_ag_ab : 
  AD^2 = AG * AB := by sorry

-- The second part of the problem
theorem ad_equals_ae (h_ac_af : AB * AC = AE * AF) : 
  AD = AE := by sorry

end ad_squared_ag_ab_ad_equals_ae_l472_472473


namespace correct_propositions_l472_472200

variable (a : ℕ → ℝ) -- the arithmetic sequence
variable (S : ℕ → ℝ) -- the sum of the first n terms
variable (d a₁ : ℝ)  -- common difference and first term of the sequence

-- given conditions
axiom sum_def (n : ℕ) : S n = n * (2 * a₁ + (n - 1) * d) / 2
axiom S6_S7_S5 : S 6 > S 7 ∧ S 7 > S 5

-- propositions to prove
def prop1 : Prop := d < 0
def prop2 : Prop := S 11 > 0
def prop3 : Prop := ∀ n, S n > 0 →  n ≤ 12
def prop4 : Prop := ∀ n, S n = S 12 → S n ≤ S 6
def prop5 : Prop := |a 6| > |a 7|

theorem correct_propositions : prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4 ∧ prop5 :=
by
  sorry

end correct_propositions_l472_472200


namespace vasya_correct_l472_472088

-- Define the condition of a convex quadrilateral
def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a < 180 ∧ b < 180 ∧ c < 180 ∧ d < 180

-- Define the properties of forming two types of triangles from a quadrilateral
def can_form_two_acute_triangles (a b c d : ℝ) : Prop :=
  a < 90 ∧ b < 90 ∧ c < 90 ∧ d < 90

def can_form_two_right_triangles (a b c d : ℝ) : Prop :=
  (a = 90 ∧ b = 90) ∨ (b = 90 ∧ c = 90) ∨ (c = 90 ∧ d = 90) ∨ (d = 90 ∧ a = 90)

def can_form_two_obtuse_triangles (a b c d : ℝ) : Prop :=
  ∃ x y z w, (x > 90 ∧ y < 90 ∧ z < 90 ∧ w < 90 ∧ (x + y + z + w = 360)) ∧
             (x > 90 ∨ y > 90 ∨ z > 90 ∨ w > 90)

-- Prove that Vasya's claim is definitively correct
theorem vasya_correct (a b c d : ℝ) (h : convex_quadrilateral a b c d) :
  can_form_two_obtuse_triangles a b c d ∧
  ¬(can_form_two_acute_triangles a b c d) ∧
  ¬(can_form_two_right_triangles a b c d) ∨
  can_form_two_right_triangles a b c d ∧
  can_form_two_obtuse_triangles a b c d := sorry

end vasya_correct_l472_472088


namespace number_of_polynomial_expressions_l472_472851

def is_polynomial (expr : String) : Prop :=
  match expr with
  | "x^2 + 2"    => true
  | "1/a + 4"    => false
  | "3ab^2 / 7"  => true
  | "ab / c"     => false
  | "-5x"        => true
  | "0"          => true
  | _            => false

theorem number_of_polynomial_expressions : 
  ∃ n : ℕ, n = 4 ∧ 
  (is_polynomial "x^2 + 2") ∧ 
  ¬(is_polynomial "1/a + 4") ∧
  (is_polynomial "3ab^2 / 7") ∧ 
  ¬(is_polynomial "ab / c") ∧ 
  (is_polynomial "-5x") ∧ 
  (is_polynomial "0") :=
begin
  use 4,
  split,
  { refl },
  repeat { split },
  { exact true.intro },
  { exact false.elim },
  { exact true.intro },
  { exact false.elim },
  { exact true.intro },
  { exact true.intro }
end

end number_of_polynomial_expressions_l472_472851


namespace find_p_and_q_find_probability_lt_l472_472402

section Probability

variables {p q : ℝ}

-- Conditions for part 1
def conditions_part1 (p q : ℝ) :=
  (1 - p) * (1 - q) = 1 / 6 ∧
  p * q = 1 / 3 ∧
  p > q

-- Conditions for part 2
def conditions_part2 (p q : ℝ) :=
  p = 2 / 3 ∧ q = 1 / 2

-- Part 1: Prove values of p and q
theorem find_p_and_q : ∃ (p q : ℝ), conditions_part1 p q ∧ p = 2 / 3 ∧ q = 1 / 2 :=
by {
  sorry
}

-- Part 2: Prove the probability calculation
theorem find_probability_lt : ∀ {p q : ℝ}, conditions_part2 p q → 
  let P := (1 - p)^2 * 2 * (1 - q) * q + (1 - p)^2 * q^2 + 2 * (1 - p) * p * q^2 in 
  P = 7 / 36 :=
by {
  assume h : conditions_part2 p q,
  sorry
}

end Probability

end find_p_and_q_find_probability_lt_l472_472402


namespace matrix_multiplication_zero_l472_472127

variable (a b c d e f : ℝ)

def matA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]
  ]

def matB : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![a^3, a^2 * b, a^2 * c],
    ![a * b^2, b^3, b^2 * c],
    ![a * c^2, b * c^2, c^3]
  ]

theorem matrix_multiplication_zero : matA a b c d e f * matB a b c d e f = 0 :=
  by
    sorry

end matrix_multiplication_zero_l472_472127


namespace correct_enrollment_statistics_l472_472404

theorem correct_enrollment_statistics 
    (enrollments : List ℕ)
    (h : enrollments = [1340, 1470, 1960, 1780, 1610]) :
    let largest := List.maximum enrollments
    let smallest := List.minimum enrollments
    let positive_difference := largest - smallest
    let total_enrollment := List.sum enrollments
    let average_enrollment := total_enrollment / enrollments.length
    positive_difference = 620 ∧ average_enrollment = 1632 :=
by
  let largest := List.maximum enrollments
  let smallest := List.minimum enrollments
  let positive_difference := largest - smallest
  let total_enrollment := List.sum enrollments
  let average_enrollment := total_enrollment / enrollments.length
  sorry

end correct_enrollment_statistics_l472_472404


namespace sphere_diameter_triple_volume_l472_472153

constant π : ℝ

def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

constant a b : ℕ

theorem sphere_diameter_triple_volume :
  ∃ (d : ℝ), d = 12 * real.cbrt 3 ∧ d = 2 * (6 * real.cbrt 3) :=
by
  let r := 6
  let original_volume := volume_sphere r
  let triple_volume := 3 * original_volume
  let r_new_cubed := triple_volume / ((4 / 3) * π)
  sorry

end sphere_diameter_triple_volume_l472_472153


namespace doesNotHoldForAnyFourLatticePoints_l472_472290

-- Definition of a lattice point
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Predicate that checks if a point is a lattice point
def isLatticePoint (p : LatticePoint) : Prop := True

-- Main theorem statement
theorem doesNotHoldForAnyFourLatticePoints
  (L : Set LatticePoint)
  (hL : ∀ p : LatticePoint, isLatticePoint p)
  (hExist : ∀ (A B C : LatticePoint) (hA : isLatticePoint A) (hB : isLatticePoint B) (hC : isLatticePoint C),
    ∃ D : LatticePoint, D ≠ A ∧ D ≠ B ∧ D ≠ C ∧ 
    ∀ E : LatticePoint, (E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D) →
    ¬ (onSegment A D E ∨ onSegment B D E ∨ onSegment C D E))
  : ¬ ∀ (A B C D : LatticePoint) (hA : isLatticePoint A) (hB : isLatticePoint B) (hC : isLatticePoint C) (hD : isLatticePoint D),
    ∃ E : LatticePoint, E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D ∧
    ∀ F : LatticePoint, (F ≠ A ∧ F ≠ B ∧ F ≠ C ∧ F ≠ D ∧ F ≠ E) →
    ¬ (onSegment A E F ∨ onSegment B E F ∨ onSegment C E F ∨ onSegment D E F) := sorry

end doesNotHoldForAnyFourLatticePoints_l472_472290


namespace exist_positive_n_with_prime_factors_l472_472346

def is_prime (n : ℕ) := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ := 
  if n = 1 then 1 else
  Classical.some (Nat.exists_prime_and_dvd (Nat.pos_of_ne_zero (ne_of_gt (Nat.succ_pos _)))).some_spec

theorem exist_positive_n_with_prime_factors (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) (h_diff : p ≠ q) :
  ∃ n : ℕ, n > 0 ∧ {smallest_prime_factor n, smallest_prime_factor (n + 2)} = {p, q} :=
  sorry

end exist_positive_n_with_prime_factors_l472_472346


namespace area_of_regular_octagon_is_correct_l472_472542

-- Conditions
def is_square (BDEF : set (ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ × ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    dist a b = dist b c ∧ dist b c = dist c d ∧ dist c d = dist d a ∧
    dist a c = dist b d

def AB_eq_BC_eq_1 (A B C : ℝ × ℝ) : Prop :=
  dist A B = 1 ∧ dist B C = 1

-- Goal
theorem area_of_regular_octagon_is_correct (A B C D E F G H : ℝ × ℝ)
  (h_square : is_square ({D, E, F, G} : set (ℝ × ℝ)))
  (h_eq : AB_eq_BC_eq_1 A B C) : 
  ∃ octagon : set (ℝ × ℝ),
    set.card octagon = 8 ∧
    (∃ a b c d e f g h : ℝ × ℝ, octagon = {a, b, c, d, e, f, g, h}) ∧
    specific_area octagon = 4 + 4 * real.sqrt 2 := 
sorry

end area_of_regular_octagon_is_correct_l472_472542


namespace unique_function_solution_l472_472897

theorem unique_function_solution :
  ∀ f : ℕ+ → ℕ+, (∀ x y : ℕ+, f (x + y * f x) = x * f (y + 1)) → (∀ x : ℕ+, f x = x) :=
by
  sorry

end unique_function_solution_l472_472897


namespace JamieEarnings_l472_472655

theorem JamieEarnings :
  ∀ (rate_per_hour : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) (weeks : ℕ),
    rate_per_hour = 10 →
    days_per_week = 2 →
    hours_per_day = 3 →
    weeks = 6 →
    rate_per_hour * days_per_week * hours_per_day * weeks = 360 :=
by
  intros rate_per_hour days_per_week hours_per_day weeks
  intros hrate hdays hhours hweeks
  rw [hrate, hdays, hhours, hweeks]
  norm_num
  sorry

end JamieEarnings_l472_472655


namespace solve_for_s_l472_472245

theorem solve_for_s (s t : ℚ) (h1 : 15 * s + 7 * t = 210) (h2 : t = 3 * s) : s = 35 / 6 := 
by
  sorry

end solve_for_s_l472_472245


namespace find_xyz_integer_solutions_l472_472078

theorem find_xyz_integer_solutions (x y z : ℕ) (h_pos: x > 0 ∧ y > 0 ∧ z > 0) :
  (∃ k : ℕ, (sqrt (2006 / (x + y)) + sqrt (2006 / (y + z)) + sqrt (2006 / (z + x))) = k) ↔ 
  (x = 2006 ∧ y = 2006 ∧ z = 2006 ∨ x = 1003 ∧ y = 1003 ∧ z = 7021 ∨ x = 9027 ∧ y = 9027 ∧ z = 9027) :=
sorry

end find_xyz_integer_solutions_l472_472078


namespace impossible_trailing_zeros_l472_472442

theorem impossible_trailing_zeros (n : ℕ) : ¬ ∃ (k : ℝ), k = 123.75999999999999 ∧ k = ∑ i in (List.range (n + 1)).filter (λ x, 5 ^ x ≤ n), n / 5 ^ i := 
by
  sorry

end impossible_trailing_zeros_l472_472442


namespace selection_structure_count_is_three_l472_472219

def requiresSelectionStructure (problem : ℕ) : Bool :=
  match problem with
  | 1 => true
  | 2 => false
  | 3 => true
  | 4 => true
  | _ => false

def countSelectionStructure : ℕ :=
  (if requiresSelectionStructure 1 then 1 else 0) +
  (if requiresSelectionStructure 2 then 1 else 0) +
  (if requiresSelectionStructure 3 then 1 else 0) +
  (if requiresSelectionStructure 4 then 1 else 0)

theorem selection_structure_count_is_three : countSelectionStructure = 3 :=
  by
    sorry

end selection_structure_count_is_three_l472_472219


namespace cost_apples_l472_472752

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l472_472752


namespace dryer_cost_l472_472110

theorem dryer_cost (W D : ℕ) (h1 : W + D = 600) (h2 : W = 3 * D) : D = 150 :=
by
  sorry

end dryer_cost_l472_472110


namespace number_of_tables_large_meeting_l472_472285

-- Conditions
def table_length : ℕ := 2
def table_width : ℕ := 1
def side_length_large_meeting : ℕ := 7

-- To be proved: number of tables needed for a large meeting is 12.
theorem number_of_tables_large_meeting : 
  let tables_per_side := side_length_large_meeting / (table_length + table_width)
  ∃ total_tables, total_tables = 4 * tables_per_side ∧ total_tables = 12 :=
by
  sorry

end number_of_tables_large_meeting_l472_472285


namespace circle_area_ratios_third_circle_area_ratio_l472_472762

variable {O P X : Type} [MetricSpace X]

def divides (O P X : X) (r : ℝ) : Prop :=
  dist O X = r * dist O P

theorem circle_area_ratios
  (h : divides O P X (1 / 4)) :
  (π * (dist O X) ^ 2) / (π * (dist O P) ^ 2) = 1 / 16 :=
by
  sorry

theorem third_circle_area_ratio
  (h : divides O P X (1 / 4)) :
  (π * (dist O (2 • X)) ^ 2) / (π * (dist O P) ^ 2) = 1 / 4 :=
by
  sorry

end circle_area_ratios_third_circle_area_ratio_l472_472762


namespace max_S_n_l472_472190

/-- Arithmetic sequence proof problem -/
theorem max_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 + a 3 + a 5 = 15)
  (h2 : a 2 + a 4 + a 6 = 0)
  (d : ℝ) (h3 : ∀ n, a (n + 1) = a n + d) :
  (∃ n, S n = 30) :=
sorry

end max_S_n_l472_472190


namespace age_of_twin_brothers_l472_472116

theorem age_of_twin_brothers (x : Nat) : (x + 1) * (x + 1) = x * x + 11 ↔ x = 5 :=
by
  sorry  -- Proof omitted.

end age_of_twin_brothers_l472_472116


namespace valid_pairs_count_is_32_l472_472237

def valid_pairs (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m^2 + n^2 < 50

def valid_pair_count : ℕ :=
  Finset.card (Finset.filter (λ p : ℕ × ℕ, valid_pairs p.1 p.2) (Finset.cartesianProduct (Finset.range 8) (Finset.range 8)))

theorem valid_pairs_count_is_32 : valid_pair_count = 32 := by
  sorry

end valid_pairs_count_is_32_l472_472237


namespace shell_count_l472_472164

theorem shell_count (initial_shells : ℕ) (ed_limpet : ℕ) (ed_oyster : ℕ) (ed_conch : ℕ) (jacob_extra : ℕ)
  (h1 : initial_shells = 2)
  (h2 : ed_limpet = 7) 
  (h3 : ed_oyster = 2) 
  (h4 : ed_conch = 4) 
  (h5 : jacob_extra = 2) : 
  (initial_shells + ed_limpet + ed_oyster + ed_conch + (ed_limpet + ed_oyster + ed_conch + jacob_extra)) = 30 := 
by 
  sorry

end shell_count_l472_472164


namespace cos_54_deg_l472_472144

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end cos_54_deg_l472_472144


namespace train_length_eq_l472_472102

theorem train_length_eq 
  (speed_kmh : ℝ) (time_sec : ℝ) 
  (h_speed_kmh : speed_kmh = 126)
  (h_time_sec : time_sec = 6.856594329596489) : 
  ((speed_kmh * 1000 / 3600) * time_sec) = 239.9808045358781 :=
by
  -- We skip the proof with sorry, as per instructions
  sorry

end train_length_eq_l472_472102


namespace cosine_54_deg_l472_472133

theorem cosine_54_deg : ∃ c : ℝ, c = cos (54 : ℝ) ∧ c = 1 / 2 :=
  by 
    let c := cos (54 : ℝ)
    let d := cos (108 : ℝ)
    have h1 : d = 2 * c^2 - 1 := sorry
    have h2 : d = -c := sorry
    have h3 : 2 * c^2 + c - 1 = 0 := sorry
    use 1 / 2 
    have h4 : c = 1 / 2 := sorry
    exact ⟨cos_eq_cos_of_eq_rad 54 1, h4⟩

end cosine_54_deg_l472_472133


namespace evaluate_fraction_l472_472244

theorem evaluate_fraction (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x - 1 / y ≠ 0) :
  (y - 1 / x) / (x - 1 / y) + y / x = 2 * y / x :=
by sorry

end evaluate_fraction_l472_472244


namespace number_of_divisors_not_divisible_by_3_l472_472588

def prime_factorization (n : ℕ) : Prop :=
  n = 2 ^ 2 * 3 ^ 2 * 5

def is_not_divisible_by (n d : ℕ) : Prop :=
  ¬ (d ∣ n)

def positive_divisors_not_divisible_by_3 (n : ℕ) : ℕ :=
  (finset.range (2 + 1)).filter (λ a, ∀ d : ℕ, is_not_divisible_by (2 ^ a * d) 3).card

theorem number_of_divisors_not_divisible_by_3 :
  prime_factorization 180 → positive_divisors_not_divisible_by_3 180 = 6 :=
by
  intro h
  sorry

end number_of_divisors_not_divisible_by_3_l472_472588


namespace max_servings_l472_472034

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l472_472034


namespace compare_star_l472_472882

def star (m n : ℤ) : ℤ := (m + 2) * 3 - n

theorem compare_star : star 2 (-2) > star (-2) 2 := 
by sorry

end compare_star_l472_472882


namespace sum_of_consecutive_primes_l472_472318

theorem sum_of_consecutive_primes :
  (∃ (a b c d : ℕ), (a < b ∧ b < c ∧ c < d) ∧ prime a ∧ prime b ∧ prime c ∧ prime d ∧ 27433619 = a * b * c * d) →
  (∃ (p q r s : ℕ), (p < q ∧ q < r ∧ r < s ∧ 27433619 = p * q * r * s ∧ prime p ∧ prime q ∧ prime r ∧ prime s) ∧ p + q + r + s = 290) :=
by 
  sorry

end sum_of_consecutive_primes_l472_472318


namespace tiffany_total_lives_l472_472002

-- Define the conditions
def initial_lives : Float := 43.0
def hard_part_won : Float := 14.0
def next_level_won : Float := 27.0

-- State the theorem
theorem tiffany_total_lives : 
  initial_lives + hard_part_won + next_level_won = 84.0 :=
by 
  sorry

end tiffany_total_lives_l472_472002


namespace quadratic_function_properties_l472_472730

/-- The graph of the quadratic function y = x^2 - 4x - 1 opens upwards,
    and the vertex is at (2, -5). -/
theorem quadratic_function_properties :
  ∀ (x : ℝ), let y := x^2 - 4*x - 1 in
  (∃ (h k : ℝ), y = (x - h)^2 + k ∧ h = 2 ∧ k = -5) ∧ (∃ a : ℝ, a > 0 ∧ (y = a * (x - 2)^2 - 5)) :=
by
  sorry

end quadratic_function_properties_l472_472730


namespace remainder_2027_div_28_l472_472052

theorem remainder_2027_div_28 : 2027 % 28 = 3 :=
by
  sorry

end remainder_2027_div_28_l472_472052


namespace n_div_p_eq_27_l472_472739

theorem n_div_p_eq_27 (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0)
    (h4 : ∃ r1 r2 : ℝ, r1 * r2 = m ∧ r1 + r2 = -p ∧ (3 * r1) * (3 * r2) = n ∧ 3 * (r1 + r2) = -m)
    : n / p = 27 := sorry

end n_div_p_eq_27_l472_472739


namespace largest_fraction_l472_472803

theorem largest_fraction :
  let fA : ℚ := 2 / 5
  let fB : ℚ := 3 / 7
  let fC : ℚ := 4 / 9
  let fD : ℚ := 7 / 15
  let fE : ℚ := 9 / 20
  let fF : ℚ := 11 / 25
  in fD > fA ∧ fD > fB ∧ fD > fC ∧ fD > fE ∧ fD > fF :=
by
  -- Definitions of each fraction
  let fA : ℚ := 2 / 5
  let fB : ℚ := 3 / 7
  let fC : ℚ := 4 / 9
  let fD : ℚ := 7 / 15
  let fE : ℚ := 9 / 20
  let fF : ℚ := 11 / 25
  -- Proof requirement
  sorry

end largest_fraction_l472_472803


namespace f_neg_2_plus_f_0_eq_neg_1_l472_472551

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 3 else if x = 0 then 0 else -f (-x)

theorem f_neg_2_plus_f_0_eq_neg_1 (h_odd : is_odd_function f)
  (h_pos : ∀ x, 0 < x → f x = 2^x - 3) : f (-2) + f 0 = -1 :=
by
  sorry

end f_neg_2_plus_f_0_eq_neg_1_l472_472551


namespace total_oil_leaked_correct_l472_472115

-- Definitions of given conditions.
def initial_leak_A : ℕ := 6522
def leak_rate_A : ℕ := 257
def time_A : ℕ := 20

def initial_leak_B : ℕ := 3894
def leak_rate_B : ℕ := 182
def time_B : ℕ := 15

def initial_leak_C : ℕ := 1421
def leak_rate_C : ℕ := 97
def time_C : ℕ := 12

-- Total additional leaks calculation.
def additional_leak (rate time : ℕ) : ℕ := rate * time
def additional_leak_A : ℕ := additional_leak leak_rate_A time_A
def additional_leak_B : ℕ := additional_leak leak_rate_B time_B
def additional_leak_C : ℕ := additional_leak leak_rate_C time_C

-- Total leaks from each pipe.
def total_leak_A : ℕ := initial_leak_A + additional_leak_A
def total_leak_B : ℕ := initial_leak_B + additional_leak_B
def total_leak_C : ℕ := initial_leak_C + additional_leak_C

-- Total oil leaked.
def total_oil_leaked : ℕ := total_leak_A + total_leak_B + total_leak_C

-- The proof problem statement.
theorem total_oil_leaked_correct : total_oil_leaked = 20871 := by
  sorry

end total_oil_leaked_correct_l472_472115


namespace total_valid_selection_methods_correct_l472_472850

def people : List String := ["a", "b", "c", "d", "e"]

def valid_selections : List (String × String) :=
  [(l, d) | l ← people, d ← people, l ≠ d ∧ d ≠ "a"]

theorem total_valid_selection_methods_correct :
  valid_selections.length = 16 := by
  sorry

end total_valid_selection_methods_correct_l472_472850


namespace complement_of_A_in_U_l472_472532

def U := {0, 2, 4, 6, 8, 10}
def A := {2, 4, 6}

theorem complement_of_A_in_U : (U \ A) = {0, 8, 10} := by
  sorry

end complement_of_A_in_U_l472_472532


namespace ratio_of_XYZ_ABC_l472_472279

theorem ratio_of_XYZ_ABC (A B C G H I X Y Z : Type) 
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ G] [InnerProductSpace ℝ H] [InnerProductSpace ℝ I]
  [InnerProductSpace ℝ X] [InnerProductSpace ℝ Y] [InnerProductSpace ℝ Z]
  (BG_GC CH_HA AI_IB : ℝ)
  (h1 : BG_GC = 2 / 1) (h2 : CH_HA = 2 / 1) (h3 : AI_IB = 2 / 1)
  (h4 : ArealsIntersect (A, G, X) (B, H, Y) (C, I, Z))
  : area(X, Y, Z) / area(A, B, C) = 1 / 36 := 
begin
  sorry
end

end ratio_of_XYZ_ABC_l472_472279


namespace fraction_a_b_l472_472254

variables {a b x y : ℝ}

theorem fraction_a_b (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (hb : b ≠ 0) :
  a / b = -1/3 := 
sorry

end fraction_a_b_l472_472254


namespace smallest_d_for_divisibility_by_9_l472_472511

theorem smallest_d_for_divisibility_by_9 : ∃ d : ℕ, 0 ≤ d ∧ d < 10 ∧ (437003 + d * 100) % 9 = 0 ∧ ∀ d', 0 ≤ d' ∧ d' < d → ((437003 + d' * 100) % 9 ≠ 0) :=
by
  sorry

end smallest_d_for_divisibility_by_9_l472_472511


namespace find_third_number_l472_472816

theorem find_third_number (x : ℕ) (h : 3 * 16 + 3 * 17 + 3 * x + 11 = 170) : x = 20 := by
  sorry

end find_third_number_l472_472816


namespace max_servings_possible_l472_472022

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l472_472022


namespace equivalent_proof_problem_l472_472802

def option_A : ℚ := 14 / 10
def option_B : ℚ := 1 + 2 / 5
def option_C : ℚ := 1 + 6 / 15
def option_D : ℚ := 1 + 3 / 8
def option_E : ℚ := 1 + 28 / 20
def target : ℚ := 7 / 5

theorem equivalent_proof_problem : option_D ≠ target :=
by {
  sorry
}

end equivalent_proof_problem_l472_472802


namespace morgan_hula_hoop_time_l472_472321

theorem morgan_hula_hoop_time :
  let nancy_time := 10
  let casey_time := nancy_time - 3
  let morgan_time := 3 * casey_time
  morgan_time = 21 := by
{
  let nancy_time := 10
  let casey_time := nancy_time - 3
  let morgan_time := 3 * casey_time
  show morgan_time = 21 from sorry
}

end morgan_hula_hoop_time_l472_472321


namespace cost_apples_l472_472751

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l472_472751


namespace poly_roots_l472_472301

open Complex

theorem poly_roots (a b : ℝ) (h : ∃ c : ℂ, c ≠ (2 + complex.I) ∧ c ≠ (2 - complex.I) ∧ (root (x^3 + (a*x) + b) 2 + complex.I) (root (x^3 + (a*x) + b) 2 - I) (root (x^3 + (a*x) + b) c)) : a + b = 9 := 
sorry

end poly_roots_l472_472301


namespace correct_conclusions_l472_472564

open Real

def f (ω x : ℝ) : ℝ := sin (ω * x) + sqrt 3 * cos (ω * x)

theorem correct_conclusions (ω > 0) (x1 x2 : ℝ) (f_x1 : f ω x1 = 2) (f_x2 : f ω x2 = 2)
  (min_dist : abs (x1 - x2) = 2) : 
  let ω := π in
  (f ω 0 ≠ sqrt 3) ∧ 
  (∃ x ∈ Ioo (0 : ℝ) 1, f ω x = 2) ∧ 
  (∀ x, f ω (x + 1 / 6) = 2 * cos (π * x)) ∧ 
  ¬ (∀ x ∈ Icc (-1 : ℝ) 0, (π * x + π / 3) ∈ Icc (-π / 2) (π / 2) → 0 ≤ sin (π * x + π / 3)) := sorry

end correct_conclusions_l472_472564


namespace expression_evaluation_l472_472777

theorem expression_evaluation :
  (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 :=
by
  sorry

end expression_evaluation_l472_472777


namespace paula_remaining_money_l472_472702

-- Define the given conditions
def given_amount : ℕ := 109
def cost_shirt : ℕ := 11
def number_shirts : ℕ := 2
def cost_pants : ℕ := 13

-- Calculate total spending
def total_spent : ℕ := (cost_shirt * number_shirts) + cost_pants

-- Define the remaining amount Paula has
def remaining_amount : ℕ := given_amount - total_spent

-- State the theorem
theorem paula_remaining_money : remaining_amount = 74 := by
  -- Proof goes here
  sorry

end paula_remaining_money_l472_472702


namespace number_of_correct_statements_l472_472639

theorem number_of_correct_statements :
  let statements := ["Each line has both point-slope and slope-intercept equations",
                     "The slope of a line with an obtuse angle is negative",
                     "The equations k = (y + 1) / (x - 2) and y + 1 = k * (x - 2) can represent the same line",
                     "If line l passes through point P(x_0, y_0) with a right angle (90°), its equation is x = x_0"]
  in 
  (statements_correct count) == 2 :=
by
  sorry

end number_of_correct_statements_l472_472639


namespace sum_M_n_eq_half_l472_472676

noncomputable def M_n (n : ℕ) : Set (ℕ × ℕ) :=
  {pq | ∃ p q : ℕ, 1 ≤ p ∧ p < q ∧ q ≤ n ∧ p + q > n ∧ Nat.gcd p q = 1}

theorem sum_M_n_eq_half (n : ℕ) (h : n > 1) :
  (∑ (pq : ℕ × ℕ) in M_n n, 1 / (pq.snd : ℚ)) = 1 / 2 := 
sorry

end sum_M_n_eq_half_l472_472676


namespace cos_54_deg_l472_472141

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end cos_54_deg_l472_472141


namespace find_new_person_weight_l472_472812

-- Define the conditions
def average_weight_increases_by (avg_increase : ℝ) (total_persons : ℕ) (replaced_person_weight : ℝ) : Prop :=
  ∃ new_person_weight : ℝ, new_person_weight = replaced_person_weight + total_persons * avg_increase

-- Define the statement
theorem find_new_person_weight :
  average_weight_increases_by 6.2 7 76 :=
begin
  use 119.4,
  sorry,
end

end find_new_person_weight_l472_472812


namespace problem_statement_l472_472306

variables (u v w : ℝ)

theorem problem_statement (h₁: u + v + w = 3) : 
  (1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8) :=
sorry

end problem_statement_l472_472306


namespace orthocenter_AXY_on_BD_l472_472267

open EuclideanGeometry

variables {A B C D X Y : Point}
variables [nondegerate_quad : nondegenerate_quadrilateral A B C D]

-- Conditions:
-- In quadrilateral ABCD, AC is the angle bisector of ∠A.
-- ∠ADC = ∠ACB.
-- X and Y are the feet of the perpendiculars from A to BC and CD, respectively.
def AC_angle_bisector (A B C D : Point) : Prop := is_angle_bisector A C
def angle_equality (A B C D : Point) : Prop := ∠ADC = ∠ACB
def feet_of_perpendiculars (A B C D X Y : Point) : Prop :=
  is_perpendicular_from_point A BC X ∧ is_perpendicular_from_point A CD Y

-- Prove that the orthocenter of triangle AXY is on BD.
theorem orthocenter_AXY_on_BD : 
  AC_angle_bisector A B C D ∧ angle_equality A B C D ∧ feet_of_perpendiculars A B C D X Y → 
  orthocenter_triangle_AXY_on_BD A B C D X Y :=
begin
  sorry
end

end orthocenter_AXY_on_BD_l472_472267


namespace parallel_HB0_PQ_l472_472261

open EuclideanGeometry

def scalene_triangle (A B C : Point) : Prop :=
  ¬ collinear A B C ∧ (∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a)

def orthocenter (A B C H : Point) : Prop :=
  altitude A B C H ∧ altitude B C A H ∧ altitude C A B H

def circumcenter (A B C O : Point) : Prop :=
  ∀ P : Point, dist O P = dist O A ↔ dist O P = dist O B ∧ dist O P = dist O C

def midpoint (X Y M : Point) : Prop :=
  dist X M = dist Y M

def intersect_at (L1 L2 : Line) (P : Point) : Prop :=
  P ∈ L1 ∧ P ∈ L2

def parallel (L1 L2 : Line) : Prop :=
  ∀ (P1 P2 Q1 Q2 : Point), (P1 ∈ L1 ∧ P2 ∈ L1 ∧ Q1 ∈ L2 ∧ Q2 ∈ L2) → slope P1 P2 = slope Q1 Q2

theorem parallel_HB0_PQ
  (A B C H O P Q A1 B0 C1 : Point)
  (h_scalene : scalene_triangle A B C)
  (h_ortho : orthocenter A B C H)
  (h_circum : circumcenter A B C O)
  (h_midpoint : midpoint A C B0)
  (h_inter_BO_AC : intersect_at (line_through B O) (line_through A C) P)
  (h_inter_BH_A1C1 : intersect_at (line_through B H) (line_through A1 C1) Q) :
  parallel (line_through H B0) (line_through P Q) :=
sorry

end parallel_HB0_PQ_l472_472261


namespace initial_balloons_l472_472394

-- Definitions based on conditions
def balloons_given_away : ℕ := 16
def balloons_left : ℕ := 14

-- Statement of the problem
theorem initial_balloons : ∃ (initial_balloons : ℕ), initial_balloons = balloons_given_away + balloons_left := 
by
  use 30
  calc 
    30 = 16 + 14 : by rfl 
    sorry

end initial_balloons_l472_472394


namespace g_g_g_g_3_eq_101_l472_472494

def g (m : ℕ) : ℕ :=
  if m < 5 then m^2 + 1 else 2 * m + 3

theorem g_g_g_g_3_eq_101 : g (g (g (g 3))) = 101 :=
  by {
    -- the proof goes here
    sorry
  }

end g_g_g_g_3_eq_101_l472_472494


namespace hex_B25_to_dec_l472_472151

theorem hex_B25_to_dec : 
  let B := 11 in
  let digit2 := 2 in
  let digit5 := 5 in
  B * 16^2 + digit2 * 16^1 + digit5 * 16^0 = 2853 :=
by
  sorry

end hex_B25_to_dec_l472_472151


namespace Dasha_purchases_l472_472054

-- Definitions
def strawberries_price_per_kg : ℕ := 300
def sugar_price_per_kg : ℕ := 30
def discount_threshold : ℕ := 1000
def discount_rate : ℕ := 50 -- representing 50%

-- Dasha’s conditions
def total_money : ℕ := 1200
def required_strawberries : ℕ := 4
def required_sugar : ℕ := 6

-- Theorem statement
theorem Dasha_purchases
  (strawberries_price : ℕ := strawberries_price_per_kg)
  (sugar_price : ℕ := sugar_price_per_kg)
  (discount_threshold : ℕ := discount_threshold)
  (discount_rate : ℕ := discount_rate)
  (total_money : ℕ := total_money)
  (required_strawberries : ℕ := required_strawberries)
  (required_sugar : ℕ := required_sugar):
  ∃ first_purchase second_purchase, 
  let cost_first_purchase := (3 * strawberries_price + 4 * sugar_price),
      cost_second_purchase := (strawberries_price + 2 * sugar_price) * (100 - discount_rate) / 100,
      total_cost := cost_first_purchase + cost_second_purchase in
  cost_first_purchase ≥ discount_threshold ∧ total_cost = total_money := 
sorry

end Dasha_purchases_l472_472054


namespace particle_probability_l472_472452

theorem particle_probability 
  (P : ℕ → ℝ) (n : ℕ)
  (h1 : P 0 = 1)
  (h2 : P 1 = 2 / 3)
  (h3 : ∀ n ≥ 3, P n = 2 / 3 * P (n-1) + 1 / 3 * P (n-2)) :
  P n = 2 / 3 + 1 / 12 * (1 - (-1 / 3)^(n-1)) := 
sorry

end particle_probability_l472_472452


namespace number_of_girls_not_playing_soccer_l472_472426

axiom total_students : ℕ := 500
axiom boys : ℕ := 350
axiom soccer_players : ℕ := 250
axiom percent_boys_playing_soccer : Real := 0.86

noncomputable def boys_playing_soccer : ℕ := (percent_boys_playing_soccer * soccer_players).toUInt
noncomputable def boys_not_playing_soccer : ℕ := boys - boys_playing_soccer
noncomputable def girls : ℕ := total_students - boys
noncomputable def girls_playing_soccer : ℕ := soccer_players - boys_playing_soccer
noncomputable def girls_not_playing_soccer : ℕ := girls - girls_playing_soccer

theorem number_of_girls_not_playing_soccer : girls_not_playing_soccer = 115 := by
  sorry

end number_of_girls_not_playing_soccer_l472_472426


namespace max_l_shapes_in_5x10_max_l_shapes_in_5x9_l472_472796

-- Define the problem conditions for part (a)
def max_l_shapes_5x10 := ∀ (n : ℕ), (n = 16) ↔ 
  ∃ (arrangement : fin 5 × fin 10 → ℕ), 
    (∀ i j, arrangement (⟨i, by linarith⟩, ⟨j, by linarith⟩) ≤ 1) ∧
    (∃ l_shapes, ∀ k < l_shapes, is_l_shape k arrangement) ∧
    l_shapes = n

-- Define the problem conditions for part (b)
def max_l_shapes_5x9 := ∀ (n : ℕ), (n = 15) ↔ 
  ∃ (arrangement : fin 5 × fin 9 → ℕ), 
    (∀ i j, arrangement (⟨i, by linarith⟩, ⟨j, by linarith⟩) ≤ 1) ∧
    (∃ l_shapes, ∀ k < l_shapes, is_l_shape k arrangement) ∧
    l_shapes = n

-- A predicate to encapsulate the L-shaped arrangement
-- We assume there exists an is_l_shape function, 
-- which checks if a given index corresponds to an L-shaped arrangement in the given grid
constant is_l_shape : ℕ → (fin 5 × fin 10 → ℕ) → Prop

-- Main statements without proofs
theorem max_l_shapes_in_5x10 : max_l_shapes_5x10 := sorry

theorem max_l_shapes_in_5x9 : max_l_shapes_5x9 := sorry

end max_l_shapes_in_5x10_max_l_shapes_in_5x9_l472_472796


namespace arithmetic_sequence_sum_condition_l472_472189

theorem arithmetic_sequence_sum_condition (a1 d : ℤ) 
  (h1 : a1 + d + (a1 + 3 * d) = 4) 
  (h2 : a1 + 2 * d + (a1 + 4 * d) = 10) :
  let S10 := 10 / 2 * (2 * a1 + (10 - 1) * d) in
  S10 = 65 :=
by
  sorry

end arithmetic_sequence_sum_condition_l472_472189


namespace sin_transformation_l472_472370

theorem sin_transformation (w : ℝ) (φ : ℝ) (h_w : w > 0) (h_φ : abs φ < real.pi) 
    (h_shift : ∀ x : ℝ, sin (x) = sin (2 * (w * (x - real.pi / 6) + φ))) :
    w = 2 ∧ φ = -real.pi / 3 :=
by {
  sorry -- Proof steps are to be filled here
}

end sin_transformation_l472_472370


namespace temperature_conversion_l472_472409

def F_to_C (F: ℝ) : ℝ := (F - 32) * 5 / 9

theorem temperature_conversion :
  F_to_C 140 = 60 := by
  sorry

end temperature_conversion_l472_472409


namespace find_x_y_values_l472_472500

noncomputable def x_and_y_conditions (x y : ℝ) :=
  (x ∈ set.Icc 1 2) ∧ (y ∈ set.Icc 1 2) ∧ (1 - real.sqrt (x - 1) = real.sqrt (y - 1))

theorem find_x_y_values (x y : ℝ) (h : x_and_y_conditions x y) :
  5 / 2 ≤ x + y ∧ x + y ≤ 3 :=
sorry

end find_x_y_values_l472_472500


namespace cos_54_eq_3_sub_sqrt_5_div_8_l472_472140

theorem cos_54_eq_3_sub_sqrt_5_div_8 :
  let x := Real.cos (Real.pi / 10) in
  let y := Real.cos (3 * Real.pi / 10) in
  y = (3 - Real.sqrt 5) / 8 :=
by
  -- Proof of the statement is omitted.
  sorry

end cos_54_eq_3_sub_sqrt_5_div_8_l472_472140


namespace inequality_proof_l472_472184

noncomputable def a := (Real.log 0.3) / (Real.log 0.4)
def b := 0.3 ^ 0.4
def c := 0.4 ^ 0.3

theorem inequality_proof : a > c ∧ c > b := 
by
  sorry

end inequality_proof_l472_472184


namespace induction_factor_increase_l472_472775

theorem induction_factor_increase (k : ℕ) (hk : 0 < k) :
  let lhs_k := (finset.range(k).map (λ i, k + 1 + i)).prod
      lhs_k1 := (finset.range(k+1).map (λ i, k + 2 + i)).prod
  in lhs_k1 = lhs_k * (2 * k + 1) * (2 * k + 2) / (k + 1) :=
by sorry

end induction_factor_increase_l472_472775


namespace line_intersects_x_axis_between_A_and_B_l472_472964

theorem line_intersects_x_axis_between_A_and_B (a : ℝ) :
  (∀ x, (x = 1 ∨ x = 3) → (2 * x + (3 - a) = 0)) ↔ 5 ≤ a ∧ a ≤ 9 :=
by
  sorry

end line_intersects_x_axis_between_A_and_B_l472_472964


namespace gcd_expression_l472_472519

theorem gcd_expression (n : ℕ) (h : n > 2) : Nat.gcd (n^5 - 5 * n^3 + 4 * n) 120 = 120 :=
by
  sorry

end gcd_expression_l472_472519


namespace rhombus_perimeter_and_triangle_area_l472_472722

-- Given conditions
variables (d1 d2 : ℕ)
variable (h₁ : d1 = 18)
variable (h₂ : d2 = 24)

-- Definitions derived from conditions
def half1 := d1 / 2
def half2 := d2 / 2
def side_length := Real.sqrt (half1 ^ 2 + half2 ^ 2)
def perimeter := 4 * side_length
def triangle_area := (1 / 2) * side_length * half1

theorem rhombus_perimeter_and_triangle_area
  (h₁ : d1 = 18)
  (h₂ : d2 = 24) :
  perimeter d1 d2 = 60 ∧ triangle_area d1 d2 = 67.5 :=
  by {
    sorry -- the proof is omitted
  }

end rhombus_perimeter_and_triangle_area_l472_472722


namespace find_b1_and_general_term_sum_S_n_of_first_n_terms_l472_472994

variable {a b c : ℕ → ℝ}

-- Conditions
axiom a1 (n : ℕ) : a 1 = b 2
axiom a2 (n : ℕ) : a 2 = b 6
axiom a3 (n : ℕ) : 2 * a n = b n * b (n + 1)
axiom a4 (n : ℕ) : a n + a (n + 1) = (b (n + 1))^2
axiom a5 : b 1 = 2
axiom a6 (n : ℕ) : b n = 2 * n
axiom a7 : c 1 = -1 / 3
axiom a8 (n : ℕ) : c n + c (n + 1) = (sqrt 2)^b n

-- Questions
theorem find_b1_and_general_term (n : ℕ) : b 1 = 2 ∧ b n = 2 * n :=
by sorry

theorem sum_S_n_of_first_n_terms (n : ℕ) : 
    let c2n := λ n : ℕ, c (2 * n) in 
    S_n = ∑ (i : ℕ) in finset.range n, c2n i :=
by sorry

end find_b1_and_general_term_sum_S_n_of_first_n_terms_l472_472994


namespace profit_relationship_profit_difference_max_total_profit_l472_472086

variables (x : ℕ) (W : ℕ)

def y1 (x : ℕ) : ℤ := 1950 - 30 * x
def y2 (x : ℕ) : ℤ := 120 * x - 2 * x^2

theorem profit_relationship (x : ℕ) : y1 x = 1950 - 30 * x ∧ y2 x = 120 * x - 2 * x^2 :=
by sorry

theorem profit_difference (x : ℕ) : y1 x = y2 x + 1250 → x = 5 :=
by sorry

theorem max_total_profit : ∃ x, (W = -2 * x^2 + 90 * x + 1950) ∧ (W = 2962) ∧ (x = 22 ∨ x = 23) :=
by sorry

end profit_relationship_profit_difference_max_total_profit_l472_472086


namespace find_a_l472_472983

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := real.log (a * x^2 - x + a / 16)
noncomputable def g (x : ℝ) : ℝ := 3^x - 9^x

def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f a x = y
def q (a : ℝ) : Prop := ∀ x : ℝ, g x < a

theorem find_a (a : ℝ) (h_false : ¬(p a ∧ q a)) : a > 2 ∨ a ≤ 1 / 4 :=
sorry

end find_a_l472_472983


namespace find_angle_A_find_function_range_l472_472636

namespace TriangleProblem

variables {A B C a b c : ℝ}

-- Given that ∆ABC is an acute triangle with sides a, b, and c opposite to angles A, B, and C, respectively
-- And given the condition (2b - c) / a = cos C / cos A
-- Define these conditions
def acute_triangle (A B C : ℝ) := A < π / 2 ∧ B < π / 2 ∧ C < π / 2

-- Define the function
def function_y (B C : ℝ) : ℝ := sqrt 3 * sin B + sin (C - π / 6)

-- Theorem 1: Prove that angle A = π / 3 given the conditions
theorem find_angle_A (h1 : acute_triangle A B C) (h2 : (2 * b - c) / a = cos C / cos A) : A = π / 3 :=
sorry

-- Theorem 2: Prove the range of the function y = sqrt 3 * sin B + sin (C - π / 6) is (sqrt 3, 2]
theorem find_function_range (h1 : acute_triangle A B C) (h2 : (2 * b - c) / a = cos C / cos A) :
  ∀ y, y = function_y B C → sqrt 3 < y ∧ y ≤ 2 :=
sorry

end TriangleProblem

end find_angle_A_find_function_range_l472_472636


namespace problem_propositions_C_D_l472_472195

/-- Given plane vectors a = (1, sqrt(3)) and b = (s, t). 
    We need to check the correctness of propositions C and D. --/

structure Vector2D where
  x : ℝ
  y : ℝ

def a : Vector2D := { x := 1, y := Real.sqrt 3 }

def b (s t : ℝ) : Vector2D := { x := s, y := t }

def magnitude (v : Vector2D) : ℝ := Real.sqrt (v.x ^ 2 + v.y ^ 2)

def angle_between (v1 v2 : Vector2D) : ℝ :=
  Real.acos ((v1.x * v2.x + v1.y * v2.y) / (magnitude v1 * magnitude v2))

theorem problem_propositions_C_D (s t : ℝ) :
  (magnitude a = magnitude (b s t) → -2 ≤ t ∧ t ≤ 2) ∧
  (s = Real.sqrt 3 → angle_between a (b s t) = Real.pi / 6 → t = 1) :=
by
  sorry

end problem_propositions_C_D_l472_472195


namespace log_sum_ineq_l472_472561

/-- Prove that for any n ∈ ℕ*, ∑_{k=1}^{n} 1 / log(n + k) > 1 / (2 * n). -/
theorem log_sum_ineq (n : ℕ) (h : n > 0) : 
  (∑ k in Finset.range n, 1 / Real.log (n + k + 1)) > 1 / (2 * n) :=
sorry

end log_sum_ineq_l472_472561


namespace initial_ratio_of_liquids_l472_472085

theorem initial_ratio_of_liquids (A B : ℕ) (H1 : A = 21)
  (H2 : 9 * A = 7 * (B + 9)) :
  A / B = 7 / 6 :=
sorry

end initial_ratio_of_liquids_l472_472085


namespace find_a_perpendicular_l472_472554

theorem find_a_perpendicular (a : ℝ) : 
  let l1 := (a - 4) * x + y + 1 = 0 
  let l2 := 2 * x + 3 * y - 5 = 0 in 
  ((∃ x y : ℝ, l1 x y ∧ l2 x y) → (2 * (a - 4) + 3 = 0)) → a = 5 / 2 :=
by
  have H := (2 * (a - 4) + 3 = 0)
  sorry

end find_a_perpendicular_l472_472554


namespace number_of_valid_pairs_l472_472908

def is_prime (n : ℕ) : Bool := n > 1 ∧ (∀ d, d ∣ n → d = 1 ∨ d = n)

def count_valid_pairs : ℕ :=
  let valid_a_b_pairs := 
    List.range' 1 50 |>.filter λ a =>
      is_prime (Nat.abs (1 - a)) || is_prime (Nat.abs (-1 - a))
  valid_a_b_pairs.length

theorem number_of_valid_pairs : count_valid_pairs = N := by
  sorry

end number_of_valid_pairs_l472_472908


namespace max_radius_of_tangent_circle_l472_472469

theorem max_radius_of_tangent_circle (a b : ℝ) (h_a : a = 5) (h_b : b = 4) :
  ∀ (r : ℝ), ∃ (c : ℝ), c = sqrt (a^2 - b^2) ∧ 
  (∀ x : ℝ, (x - c)^2 + y^2 ≤ r^2) ∧ (c = 3) → r = 2 :=
begin
  intros r,
  use sqrt (a^2 - b^2),
  split,
  { exact h_eq },
  { 
    split,
    { intros x,
      exact sorry },
    { exact sorry }
  }
end

end max_radius_of_tangent_circle_l472_472469


namespace trains_cross_time_l472_472044

noncomputable def time_to_cross (len1 len2 speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * (5 / 18)
  let speed2_ms := speed2_kmh * (5 / 18)
  let relative_speed_ms := speed1_ms + speed2_ms
  let total_distance := len1 + len2
  total_distance / relative_speed_ms

theorem trains_cross_time :
  time_to_cross 1500 1000 90 75 = 54.55 := by
  sorry

end trains_cross_time_l472_472044


namespace partition_exists_l472_472918

def r_S (S : set ℕ) (n : ℕ) : ℕ := 
  {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2 ∧ p.1 + p.2 = n}.to_finset.card

theorem partition_exists (A B : set ℕ)
  (hA : A.partition (λ n, even (nat.popcount n)))
  (hB : B.partition (λ n, ¬ even (nat.popcount n)))
  (hU : ∀ n, n ∈ A ∨ n ∈ B)
  (hD : ∀ n, ¬ (n ∈ A ∧ n ∈ B)) :
  ∀ n, r_S A n = r_S B n := 
  sorry

end partition_exists_l472_472918


namespace length_perpendicular_segment_l472_472710

-- Definitions and theorems
variable {α : Type*} [EuclideanSpace α]

-- Consider A, B, C are points on the Euclidean space.
variables {A B C G D E F : α}

-- The perpendicular distances
variables {AD BE CF : ℝ}

-- Given conditions
def conditions (AD BE CF : ℝ) := AD = 12 ∧ BE = 8 ∧ CF = 20

-- The centroid of the triangle
def centroid (A B C : α) : α := 
  (1 / 3 : ℝ) • (A + B + C)

-- The theorem statement
theorem length_perpendicular_segment (AD BE CF : ℝ) (h : conditions AD BE CF) :
  let G := centroid A B C in
  ∃ x : ℝ, x = 40 / 3 :=
by
  sorry

end length_perpendicular_segment_l472_472710


namespace total_bill_after_90_days_l472_472478

theorem total_bill_after_90_days (initial_bill : ℝ) (late_charge_rate : ℝ) (days_late : ℕ) (final_bill : ℝ) : 
  initial_bill = 600 → 
  late_charge_rate = 0.02 → 
  days_late = 90 → 
  final_bill = 636.53 → 
  final_bill = initial_bill * (1 + late_charge_rate) ^ (days_late / 30) := 
by 
  intro h1 h2 h3 h4 
  rw [h1, h2] 
  have h5 : 636.53 = 600 * (1 + 0.02) ^ (90 / 30) := sorry 
  rw h5 
  exact h4

end total_bill_after_90_days_l472_472478


namespace num_positive_divisors_not_divisible_by_3_l472_472593

theorem num_positive_divisors_not_divisible_by_3 (n : ℕ) (h : n = 180) : 
  (∃ (divisors : finset ℕ), (∀ d ∈ divisors, d ∣ n ∧ ¬ (3 ∣ d)) ∧ finset.card divisors = 6) := 
by
  have prime_factors : (n = 2^2 * 3^2 * 5) := by norm_num [h]
  sorry

end num_positive_divisors_not_divisible_by_3_l472_472593


namespace min_value_of_f_l472_472497

noncomputable def f : ℝ → ℝ := λ x, 3*x^2 - 6*x + 9

theorem min_value_of_f : ∃ x₀, ∀ x, f x₀ ≤ f x :=
begin
  use 1,  -- x₀ = 1, where the minimum occurs
  intro x,
  -- The quadratic function opens upwards since the leading coefficient (3) is positive.
  -- The minimum value is at the vertex, and the function is f(x) = 3x² - 6x + 9.
  -- At x = 1, f(1) = 3*1² - 6*1 + 9 = 6.
  -- Thus, for all x, f(1) = 6 is the minimum value.
  sorry
end

end min_value_of_f_l472_472497


namespace select_pencils_l472_472256

theorem select_pencils (boxes : Fin 10 → ℕ) (colors : ∀ (i : Fin 10), Fin (boxes i) → Fin 10) :
  (∀ i : Fin 10, 1 ≤ boxes i) → -- Each box is non-empty
  (∀ i j : Fin 10, i ≠ j → boxes i ≠ boxes j) → -- Different number of pencils in each box
  ∃ (selection : Fin 10 → Fin 10), -- Function to select a pencil color from each box
  Function.Injective selection := -- All selected pencils have different colors
sorry

end select_pencils_l472_472256


namespace ellipse_properties_l472_472534

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a c : ℝ) : Prop :=
  c / a = 1 / 2

noncomputable def distance_from_focus_to_line (a b c d : ℝ) : Prop :=
  d = (abs (b * c - a * b)) / sqrt (a^2 + b^2)

noncomputable def distance_O_to_AB_constant (d : ℝ) : Prop :=
  d = 2 * sqrt(21) / 7

noncomputable def min_length_chord_AB (AB_min : ℝ) : Prop :=
  AB_min = 4 * sqrt(21) / 7

theorem ellipse_properties 
  (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse_equation a b) 
  (h4 : eccentricity a c) 
  (d : ℝ) (h5 : distance_from_focus_to_line a b c d)
  (const_d : ℝ) (h6 : distance_O_to_AB_constant const_d)
  (min_AB : ℝ) (h7 : min_length_chord_AB min_AB) :
  (a = 2) ∧ (b = sqrt 3) ∧ (ellipse_equation a b) ∧
  (distance_O_to_AB_constant const_d) ∧
  (min_length_chord_AB min_AB) 
:= by
  sorry

end ellipse_properties_l472_472534


namespace only_rational_root_is_one_l472_472898

-- Define the polynomial
def polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 (x : ℚ) : ℚ :=
  3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

-- The main theorem stating that 1 is the only rational root
theorem only_rational_root_is_one : 
  ∀ x : ℚ, polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 x = 0 ↔ x = 1 :=
by
  sorry

end only_rational_root_is_one_l472_472898


namespace arithmetic_sequence_a_11_l472_472940

theorem arithmetic_sequence_a_11 :
  (∃ a : ℕ → ℤ, a 1 = 1 ∧ (∀ n : ℕ, a (n+2) - a n = 6)) → a 11 = 61 :=
by
  sorry

end arithmetic_sequence_a_11_l472_472940


namespace find_a_l472_472959

-- Define the circle equation and the line equation as conditions
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 1
def line_eq (x y a : ℝ) : Prop := y = x + a
def chord_length (l : ℝ) : Prop := l = 2

-- State the main problem
theorem find_a (a : ℝ) (h1 : ∀ x y : ℝ, circle_eq x y → ∃ y', line_eq x y' a ∧ chord_length 2) :
  a = -2 :=
sorry

end find_a_l472_472959


namespace maximize_probability_remove_15_l472_472049

def integer_list : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def remove_element (l : List ℤ) (x : ℤ) : List ℤ :=
  l.erase x

def sum_to_15 (l : List ℤ) : ℕ :=
  (l.product l).count (λ (pair : ℤ × ℤ), pair.fst ≠ pair.snd ∧ pair.fst + pair.snd = 15)

theorem maximize_probability_remove_15 :
  ∀ x ∈ integer_list, sum_to_15 (remove_element integer_list x) ≤ sum_to_15 (remove_element integer_list 15) := 
sorry

end maximize_probability_remove_15_l472_472049


namespace min_abs_phi_eq_pi_div_2_l472_472510

noncomputable def min_abs_phi : ℝ :=
  if h : ∃ φ : ℝ, ∀ x : ℝ, 3 * cos (2 * x + φ) = -3 * cos (2 * x + φ) then
    classical.some h
  else 0

theorem min_abs_phi_eq_pi_div_2 :
  min_abs_phi = π / 2 :=
by
  admit -- proof goes here, we're omitting it.

end min_abs_phi_eq_pi_div_2_l472_472510


namespace polynomial_degree_example_l472_472781

theorem polynomial_degree_example :
  ∀ (x: ℝ), degree ((5 * x^3 + 7) ^ 10) = 30 :=
by
  sorry

end polynomial_degree_example_l472_472781


namespace find_N_value_l472_472216

variable (a b N : ℚ)
variable (h1 : a + 2 * b = N)
variable (h2 : a * b = 4)
variable (h3 : 2 / a + 1 / b = 1.5)

theorem find_N_value : N = 6 :=
by
  sorry

end find_N_value_l472_472216


namespace sqrt_sub_inequality_l472_472132

theorem sqrt_sub_inequality : sqrt 7 - sqrt 6 < sqrt 6 - sqrt 5 :=
sorry

end sqrt_sub_inequality_l472_472132


namespace correct_choice_l472_472852

-- Define the functions as stated in the options
def f_A (x : ℝ) : ℝ := Real.cos(2 * x + Real.pi / 2)
def f_B (x : ℝ) : ℝ := Real.sin(2 * x + Real.pi / 2)
def f_C (x : ℝ) : ℝ := Real.sin(2 * x) + Real.cos(2 * x)
def f_D (x : ℝ) : ℝ := Real.sin(x) + Real.cos(x)

-- Define the requirements: smallest positive period is π and the function is odd
def has_period_pi (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + Real.pi) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- The main statement: Proving that f_A meets the stated requirements
theorem correct_choice : has_period_pi f_A ∧ is_odd f_A :=
by
  -- Here we would provide the proof
  sorry

end correct_choice_l472_472852


namespace liars_count_l472_472386

-- Definitions for problem conditions
def Person := { knight : Bool // knight = tt ∨ knight = ff } -- Represents a person (knight or liar)
def Table := Array Person -- Represents a circular table of persons

def statement1 (person : Person) (right_neigh : Person) : Prop :=
  person.val = tt → right_neigh.val = tt

def statement2 (person : Person) (second_right_neigh : Person) : Prop :=
  person.val = tt → second_right_neigh.val = tt

def valid_table (table : Table) : Prop :=
  ∀ i, (statement1 table[i] table[(i + 1) % 120]) ∨ (statement2 table[i] table[(i + 2) % 120])

-- Question rephrased in Lean
theorem liars_count : ∃ n : Nat, n = 0 ∨ n = 60 ∨ n = 120 ∧
  valid_table (Array.init 120 (λ i, ⟨if i < n then false else true, sorry⟩)) := sorry

end liars_count_l472_472386


namespace Condition_甲_is_necessary_but_not_sufficient_l472_472359

noncomputable def Condition_甲 (a : ℝ) : Prop := a > 0
noncomputable def Condition_乙 (a b : ℝ) : Prop := a > b ∧ a⁻¹ > b⁻¹

theorem Condition_甲_is_necessary_but_not_sufficient (a b : ℝ) :
  (Condition_甲 a → Condition_乙 a b) ∧ (Condition_乙 a b → Condition_甲 a) :=
by
  sorry

end Condition_甲_is_necessary_but_not_sufficient_l472_472359


namespace largest_three_digit_integer_divisible_by_8_and_non_zero_digits_l472_472507

theorem largest_three_digit_integer_divisible_by_8_and_non_zero_digits :
  ∃ n : ℕ, (n ≤ 999) ∧ (n ≥ 800) ∧ (n % 8 = 0) ∧ (∀ d : ℕ, (d ∈ [8] ++ list.digits (n % 100)) → (d ≠ 0 → n % d = 0)) ∧ (n = 888) :=
by 
  sorry

end largest_three_digit_integer_divisible_by_8_and_non_zero_digits_l472_472507


namespace problem1_problem2_l472_472714

-- Problem (1) Statement
theorem problem1 (m n : ℕ) (hm : 2^m = 32) (hn : 3^n = 81) : 5^(m - n) = 5 := by
  sorry

-- Problem (2) Statement
theorem problem2 (x y : ℤ) (h : 3 * x + 2 * y + 1 = 3) : 27^x * 9^y * 3 = 27 := by
  sorry

end problem1_problem2_l472_472714


namespace dice_sum_probability_15_l472_472415
open Nat

theorem dice_sum_probability_15 (n : ℕ) (h : n = 3432) : 
  ∃ d1 d2 d3 d4 d5 d6 d7 d8 : ℕ,
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ 
  (1 ≤ d3 ∧ d3 ≤ 6) ∧ (1 ≤ d4 ∧ d4 ≤ 6) ∧ 
  (1 ≤ d5 ∧ d5 ≤ 6) ∧ (1 ≤ d6 ∧ d6 ≤ 6) ∧ 
  (1 ≤ d7 ∧ d7 ≤ 6) ∧ (1 ≤ d8 ∧ d8 ≤ 6) ∧ 
  (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 = 15) :=
by
  sorry

end dice_sum_probability_15_l472_472415


namespace teacher_engineer_ratio_l472_472716

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers and e is the number of engineers.
  (h1 : (40 * t + 55 * e) / (t + e) = 45)
  : t = 2 * e :=
by
  sorry

end teacher_engineer_ratio_l472_472716


namespace relationship_between_x_y_l472_472746

theorem relationship_between_x_y :
  ∀ x y : ℤ, (x = 0 ∧ y = 240) ∨ (x = 1 ∧ y = 180) ∨ (x = 2 ∧ y = 120) ∨ (x = 3 ∧ y = 60) ∨ (x = 4 ∧ y = 0) →
  y = 240 - 60 * x :=
by
  intros x y h
  cases h;
  {
    -- x = 0 and y = 240
    cases h,
    rw [h.left, h.right],
    rw mul_zero,
    simp
  };
  {
    -- x = 1 and y = 180
    cases h,
    rw [h.left, h.right],
    simp
  };
  {
    -- x = 2 and y = 120
    cases h,
    rw [h.left, h.right],
    simp
  };
  {
    -- x = 3 and y = 60
    cases h,
    rw [h.left, h.right],
    simp
  };
  {
    -- x = 4 and y = 0
    cases h,
    rw [h.left, h.right],
    simp
  }

end relationship_between_x_y_l472_472746


namespace correct_calculations_count_l472_472364

theorem correct_calculations_count : 
  ( ¬ (a ^ 12 - a ^ 6 = a ^ 6) ∧
    ¬ (x ^ 2 * y ^ 3 * x ^ 4 * y = x ^ 8 * y ^ 3) ∧
    (a ^ 3 * b ^ 3 = (a * b) ^ 3) ∧
    ¬ ((-3 * a ^ 3) ^ 2 / (-a ^ 8) = 9 * a ^ (-2)) ∧
    ¬ ((2 * x + 3) ^ 2 = 4 * x ^ 2 + 9) ∧
    (3 - x) / (x ^ 2 - 9) = -1 / (x + 3)) → 
  (2 = 2) :=
by sorry

end correct_calculations_count_l472_472364


namespace T_is_x_plus_3_to_the_4_l472_472299

variable (x : ℝ)

def T : ℝ := (x + 2)^4 + 4 * (x + 2)^3 + 6 * (x + 2)^2 + 4 * (x + 2) + 1

theorem T_is_x_plus_3_to_the_4 : T x = (x + 3)^4 := by
  -- Proof would go here
  sorry

end T_is_x_plus_3_to_the_4_l472_472299


namespace different_distance_from_P_l472_472640

-- Define the coordinates of the points
def A := (2, 3)
def B := (4, 5)
def C := (6, 5)
def D := (7, 4)
def E := (8, 1)
def P := (5, 2)

-- Define the Euclidean distance function
def dist (p1 p2 : ℕ × ℕ) : ℝ :=
  real.sqrt (↑(p2.1 - p1.1) ^ 2 + ↑(p2.2 - p1.2) ^ 2)

-- The theorem to prove
theorem different_distance_from_P :
  dist P D ≠ dist P A ∧ dist P D ≠ dist P B ∧ dist P D ≠ dist P C ∧ dist P D ≠ dist P E :=
by sorry

end different_distance_from_P_l472_472640


namespace addison_sold_tickets_l472_472694

noncomputable def number_of_tickets_sold_friday (F : ℕ) : Prop :=
  let saturday_tickets := 2 * F
  let sunday_tickets := 78
  let extra_tickets := 284
  saturday_tickets = sunday_tickets + extra_tickets

theorem addison_sold_tickets {F : ℕ} (h : number_of_tickets_sold_friday F) : F = 181 :=
by
  unfold number_of_tickets_sold_friday at h
  have h1 : 2 * F = 78 + 284 := h
  have h2 : 2 * F = 362 := by rw [Nat.add_comm] at h1; exact h1
  have h3 : F = 362 / 2 := Nat.eq_of_mul_eq_mul_left (by norm_num) h2
  exact h3.symm

end addison_sold_tickets_l472_472694


namespace num_divisors_not_div_by_3_l472_472599

theorem num_divisors_not_div_by_3 : 
  let n := 180 in
  let prime_factorization_180 := factorization 180 in
  (prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180) →
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)} in
  divisors_not_div_by_3.card = 6 :=
by 
  let n := 180
  let prime_factorization_180 := factorization n
  have h_factorization : prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180 := -- proof ommitted
    sorry
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)}
  have h_card : divisors_not_div_by_3.card = 6 := -- proof ommitted
    sorry
  exact h_card

end num_divisors_not_div_by_3_l472_472599


namespace square_adjacent_to_multiple_of_5_l472_472706

theorem square_adjacent_to_multiple_of_5 (n : ℤ) (h : n % 5 ≠ 0) : (∃ k : ℤ, n^2 = 5 * k + 1) ∨ (∃ k : ℤ, n^2 = 5 * k - 1) := 
by
  sorry

end square_adjacent_to_multiple_of_5_l472_472706


namespace probability_white_ball_l472_472823

theorem probability_white_ball
  (white_balls : ℕ)
  (black_balls : ℕ)
  (locked_balls : ℕ)
  (total_balls := white_balls + black_balls)
  (accessible_balls := total_balls - locked_balls)
  (accessible_white_balls := white_balls)
  (white_probability := accessible_white_balls / accessible_balls) :  
  white_balls = 7 → 
  black_balls = 10 →
  locked_balls = 3 → 
  white_probability = 1 / 2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.add_sub_assoc (by decide : 3 ≤ 17)]
  unfold accessible_balls accessible_white_balls white_probability
  norm_num
  sorry

end probability_white_ball_l472_472823


namespace download_time_l472_472727

def first_segment_size : ℝ := 30
def first_segment_rate : ℝ := 5
def second_segment_size : ℝ := 40
def second_segment_rate1 : ℝ := 10
def second_segment_rate2 : ℝ := 2
def third_segment_size : ℝ := 20
def third_segment_rate1 : ℝ := 8
def third_segment_rate2 : ℝ := 4

theorem download_time :
  let time_first := first_segment_size / first_segment_rate
  let time_second := (10 / second_segment_rate1) + (10 / second_segment_rate2) + (10 / second_segment_rate1) + (10 / second_segment_rate2)
  let time_third := (10 / third_segment_rate1) + (10 / third_segment_rate2)
  time_first + time_second + time_third = 21.75 :=
by
  sorry

end download_time_l472_472727


namespace sum_of_all_possible_p_values_l472_472287

noncomputable def sum_of_possible_primes : ℕ :=
  let p_values := {p : ℕ | prime p ∧ ∃ q : ℕ, p ∣ (q - 1) ∧ (p + q) ∣ (p^2 + 2020 * q^2)};
  p_values.to_finset.sum id

theorem sum_of_all_possible_p_values : sum_of_possible_primes = 35 := 
by
  sorry

end sum_of_all_possible_p_values_l472_472287


namespace paper_area_ratio_l472_472454

-- Define the problem conditions
theorem paper_area_ratio (w : ℝ) (A : ℝ) (B : ℝ) (h1 : A = 2 * w^2) (h2 : B = A - (sqrt 2 / 4 + sqrt 1.25 / 4)) :
  B / A = 1 - (sqrt 2 + sqrt 1.25) / 8 :=
by
  sorry

end paper_area_ratio_l472_472454


namespace minimum_possible_n_l472_472449

theorem minimum_possible_n (n p : ℕ) (h1: p > 0) (h2: 15 * n - 45 = 105) : n = 10 :=
sorry

end minimum_possible_n_l472_472449


namespace max_sequence_length_l472_472303

-- The maximum number of terms in the sequence \( x_k \) is 20 given the conditions.
theorem max_sequence_length (M : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : ∀ k, 1 ≤ k → x k > 0)
  (h2 : ∀ k, 1 ≤ k → x k ≤ M)
  (h3 : ∀ k, 3 ≤ k → x k = (x (k - 1) - x (k - 2)).natAbs) :
  n ≤ 20 :=
begin
  sorry
end

end max_sequence_length_l472_472303


namespace intersection_M_N_l472_472988

def M (x : ℝ) : Prop := abs (x - 1) ≥ 2

def N (x : ℝ) : Prop := x^2 - 4 * x ≥ 0

def P (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4

theorem intersection_M_N (x : ℝ) : (M x ∧ N x) → P x :=
by
  sorry

end intersection_M_N_l472_472988


namespace distance_between_lines_AE_BF_l472_472632

-- Definitions of points and conditions in the problem
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A := Point3D.mk 0 0 0
def B := Point3D.mk 48 0 0
def D := Point3D.mk 0 24 0
def A1 := Point3D.mk 0 0 12
def B1 := Point3D.mk 48 0 12
def C1 := Point3D.mk 48 24 12

def E := Point3D.mk ((A1.x + B1.x) / 2) ((A1.y + B1.y) / 2) ((A1.z + B1.z) / 2)
def F := Point3D.mk (B1.x + C1.x) / 2 (B1.y + C1.y) / 2 (B1.z + C1.z) / 2

def vector3D := Point3D → Point3D → Point3D
def cross_product3D := Point3D → Point3D → Point3D
def dot_product3D := Point3D → Point3D → ℝ
def magnitude3D := Point3D → ℝ

axiom calculate_distance_between_lines (A B D A1 B1 C1 E F : Point3D)
  (AB_len AD_len AA1_len : ℝ)
  (E_midpoint_B1 C1_midpoint : Prop) :
  dist := 16

-- Use the axiom to define the theorem to be proved
theorem distance_between_lines_AE_BF :
  calculate_distance_between_lines A B D A1 B1 C1 E F 48 24 12
    (E = Point3D.mk ((A1.x + B1.x) / 2) ((A1.y + B1.y) / 2) ((A1.z + B1.z) / 2))
    (F = Point3D.mk (B1.x + C1.x) / 2 (B1.y + C1.y) / 2 (B1.z + C1.z) / 2) :=
by
  sorry

end distance_between_lines_AE_BF_l472_472632


namespace triangle_area_sol_y_range_sol_l472_472277

noncomputable def B_angle_sol (a b c : ℝ) (h : b = sqrt 13) (hsum : a + c = 4)
  (hm_perp : (cos (2*π/3), cos C) ⊥ (2*a + c, b)) : ℝ := by
  have hB : cos (2*π/3) = -1/2 := by sorry
  have B : 2*π/3 = arccos (-1/2) := by sorry
  exact 2*π/3

theorem triangle_area_sol (a b c : ℝ) (h : b = sqrt 13) (h_ac_sum : a + c = 4)
  (h_ac_prod : a * c = 3) : ℝ := by
  have h_area : (1 / 2) * a * c * (sqrt 3 / 2) = 3 * sqrt 3 / 4 := by sorry
  exact 3 * sqrt 3 / 4

theorem y_range_sol (A C : ℝ) (h_range : -π/3 < A - C ∧ A - C < π/3) :
  ℝ := by
  have h_y : sqrt 3 / 2 < sqrt 3 * cos (A - C) ∧ sqrt 3 * cos (A - C) ≤ sqrt 3 := by sorry
  exact sqrt 3 * cos (A - C)

# A clearer and detailed proof with correct symbolic links and logical consistency should be included.

end triangle_area_sol_y_range_sol_l472_472277


namespace ticket_cost_increase_l472_472891

theorem ticket_cost_increase (last_year_cost : ℝ) (increase_percentage : ℝ) :
  last_year_cost = 85 ∧ increase_percentage = 0.20 → (last_year_cost * (1 + increase_percentage) = 102) :=
by
  intros h
  cases h with h1 h2
  rw h1
  rw h2
  norm_num
  sorry

end ticket_cost_increase_l472_472891


namespace OC_squared_l472_472992

-- Definitions of points O, A, B and the point C on the line AB
variable (O A B : Type) [inner_product_space ℝ O] [inner_product_space ℝ A] [inner_product_space ℝ B]

-- Defining the specific point C with the condition
variable C : O -- C is a point in the same space

-- Conditions
variable h1 : C ∈ line A B
variable h2 : (dist O C) ^ 2 = (dist O A) ^ 2 + (dist O B) ^ 2

-- The theorem we want to prove
theorem OC_squared (h : C) : (dist O C) ^ 2 = 2 * (dist O A) ^ 2 - (dist O B) ^ 2 :=
sorry

end OC_squared_l472_472992


namespace cost_of_apples_l472_472755

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l472_472755


namespace derangement_516_l472_472817

theorem derangement_516 : 
  ∃ a : ℕ, a = (516.factorial) * (∑ k in finset.range 515, (-1)^(k+2) / ((k+2).factorial)) :=
sorry

end derangement_516_l472_472817


namespace outlet_pipe_emptying_time_l472_472772

noncomputable def fill_rate_pipe1 : ℝ := 1 / 18
noncomputable def fill_rate_pipe2 : ℝ := 1 / 30
noncomputable def empty_rate_outlet_pipe (x : ℝ) : ℝ := 1 / x
noncomputable def combined_rate (x : ℝ) : ℝ := fill_rate_pipe1 + fill_rate_pipe2 - empty_rate_outlet_pipe x
noncomputable def total_fill_time : ℝ := 0.06666666666666665

theorem outlet_pipe_emptying_time : ∃ x : ℝ, combined_rate x = 1 / total_fill_time ∧ x = 45 :=
by
  sorry

end outlet_pipe_emptying_time_l472_472772


namespace matrix_power_vector_eq_l472_472225

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 5], ![0, -2]]
def vector_beta : Fin 2 → ℝ := ![1, -1]

theorem matrix_power_vector_eq :
  (matrix_A ^ 2016) ⬝ vector_beta = ![2^2016, -2^2016] :=
sorry

end matrix_power_vector_eq_l472_472225


namespace kendall_tau_correct_l472_472815

-- Base Lean setup and list of dependencies might go here

structure TestScores :=
  (A : List ℚ)
  (B : List ℚ)

-- Constants from the problem
def scores : TestScores :=
  { A := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
  , B := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70] }

-- Function to calculate the Kendall rank correlation coefficient
noncomputable def kendall_tau (scores : TestScores) : ℚ :=
  -- the method of calculating Kendall tau could be very complex
  -- hence we assume the correct coefficient directly for the example
  0.51

-- The proof problem
theorem kendall_tau_correct : kendall_tau scores = 0.51 :=
by
  sorry

end kendall_tau_correct_l472_472815


namespace base10_to_base2_l472_472880

def num_zeros (b : ℕ) : ℕ :=
  (b.digits 2).count 0

def num_ones (b : ℕ) : ℕ :=
  (b.digits 2).count 1

theorem base10_to_base2 (x y : ℕ) : 173.digits 2 = [1, 0, 1, 0, 1, 1, 0, 1]
  ∧ x = num_zeros 173
  ∧ y = num_ones 173
  → y - x = 4 :=
by
  sorry

end base10_to_base2_l472_472880


namespace route_down_distance_l472_472092

-- Definitions
def rate_up : ℝ := 7
def time_up : ℝ := 2
def distance_up : ℝ := rate_up * time_up
def rate_down : ℝ := 1.5 * rate_up
def time_down : ℝ := time_up
def distance_down : ℝ := rate_down * time_down

-- Theorem
theorem route_down_distance : distance_down = 21 := by
  sorry

end route_down_distance_l472_472092


namespace inequality_sum_reciprocal_log_l472_472565

theorem inequality_sum_reciprocal_log (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
    (∑ i in Finset.range n, 1 / Real.log (m + (i + 1))) > (n : ℝ) / (m * (m + n)) := by
  sorry

end inequality_sum_reciprocal_log_l472_472565


namespace worker_net_salary_change_l472_472845

theorem worker_net_salary_change (S : ℝ) :
  let final_salary := S * 1.15 * 0.90 * 1.20 * 0.95
  let net_change := final_salary - S
  net_change = 0.0355 * S := by
  -- Proof goes here
  sorry

end worker_net_salary_change_l472_472845


namespace increasing_function_implies_a_nonpositive_l472_472204

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem increasing_function_implies_a_nonpositive (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → a ≤ 0 :=
by
  sorry

end increasing_function_implies_a_nonpositive_l472_472204


namespace inequality_l472_472935

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.pi)
noncomputable def c : ℝ := Real.log 0.5 / Real.log 2

theorem inequality (h1: a = Real.sqrt 2) (h2: b = Real.log 3 / Real.log Real.pi) (h3: c = Real.log 0.5 / Real.log 2) : a > b ∧ b > c := 
by 
  sorry

end inequality_l472_472935


namespace activity_popularity_order_l472_472476

-- Definitions for the fractions representing activity popularity
def dodgeball_popularity : Rat := 9 / 24
def magic_show_popularity : Rat := 4 / 12
def singing_contest_popularity : Rat := 1 / 3

-- Theorem stating the order of activities based on popularity
theorem activity_popularity_order :
  dodgeball_popularity > magic_show_popularity ∧ magic_show_popularity = singing_contest_popularity :=
by 
  sorry

end activity_popularity_order_l472_472476


namespace angle_PED_is_120_l472_472489

-- Definitions of points, angles, and triangles based on conditions.
variables (P Q R D E F : Type*)
variables (angle_P angle_Q angle_R angle_PED : ℝ)

-- Given conditions
def conditions :=
  (angle_P = 50) ∧
  (angle_Q = 70) ∧
  (angle_R = 60) ∧
  (D ∈ set.range (λ t : ℝ, t • (Q - R))) ∧
  (E ∈ set.range (λ t : ℝ, t • (P - Q))) ∧
  (F ∈ set.range (λ t : ℝ, t • (P - R)))

-- Theorem stating what we need to prove
theorem angle_PED_is_120 (h : conditions) : angle_PED = 120 := 
sorry

end angle_PED_is_120_l472_472489


namespace num_satisfying_integers_l472_472604

/-- Define the greatest integer not exceeding x -/
def floor (x : ℝ) := ⌊x⌋

/-- Define the integer sequence satisfying the conditions -/
def is_pos_int_satisfy (n : ℕ) : Prop :=
  (n + 1500) % 90 = 0 ∧
  (n > 0) ∧
  ((n + 1500) / 90) = floor (Real.sqrt n)

/-- Prove that the number of positive integers n 
that satisfy the given conditions is exactly 5 -/
theorem num_satisfying_integers : 
  (Finset.filter is_pos_int_satisfy (Finset.range 10000)).card = 5 :=
by
  sorry

end num_satisfying_integers_l472_472604


namespace symmetrical_points_l472_472963

def f (x : ℝ) : ℝ := x^2 + Real.exp x - (1/2)
def g (x a : ℝ) : ℝ := x^2 + Real.log (x + a)

theorem symmetrical_points (a : ℝ) :
  (∃ x < 0, f x = g (-x) a) ↔ a ∈ Set.Iio (Real.sqrt Real.exp 1) := 
by
  sorry

end symmetrical_points_l472_472963


namespace problem_statement_l472_472622

noncomputable def a : ℚ := 18 / 11
noncomputable def c : ℚ := -30 / 11

theorem problem_statement (a b c : ℚ) (h1 : b / a = 4)
    (h2 : b = 18 - 7 * a) (h3 : c = 2 * a - 6):
    a = 18 / 11 ∧ c = -30 / 11 :=
by
  sorry

end problem_statement_l472_472622


namespace number_of_divisors_not_divisible_by_3_l472_472591

def prime_factorization (n : ℕ) : Prop :=
  n = 2 ^ 2 * 3 ^ 2 * 5

def is_not_divisible_by (n d : ℕ) : Prop :=
  ¬ (d ∣ n)

def positive_divisors_not_divisible_by_3 (n : ℕ) : ℕ :=
  (finset.range (2 + 1)).filter (λ a, ∀ d : ℕ, is_not_divisible_by (2 ^ a * d) 3).card

theorem number_of_divisors_not_divisible_by_3 :
  prime_factorization 180 → positive_divisors_not_divisible_by_3 180 = 6 :=
by
  intro h
  sorry

end number_of_divisors_not_divisible_by_3_l472_472591


namespace sum_of_valid_x_in_degrees_l472_472514

def sum_of_solutions (x : ℝ) : ℝ :=
  ∑ x in {x | sin (3 * x) ^ 3 + sin (5 * x) ^ 3 = 8 * sin (4 * x) ^ 3 * sin (x) ^ 3 
          ∧ 100 < x 
          ∧ x < 200}, x

theorem sum_of_valid_x_in_degrees :
  sum_of_solutions 1876 = 687 :=
sorry

end sum_of_valid_x_in_degrees_l472_472514


namespace original_time_to_cover_distance_l472_472807

theorem original_time_to_cover_distance (S : ℝ) (T : ℝ) (D : ℝ) :
  (0.8 * S) * (T + 10 / 60) = S * T → T = 2 / 3 :=
  by sorry

end original_time_to_cover_distance_l472_472807


namespace max_sum_ab_bc_cd_de_ea_l472_472294

theorem max_sum_ab_bc_cd_de_ea (a b c d e : ℕ) (h1 : {a, b, c, d, e} = {1, 2, 3, 4, 5}) :
  ab + bc + cd + de + ea ≤ 47 := 
sorry

end max_sum_ab_bc_cd_de_ea_l472_472294


namespace surface_area_of_inscribed_cube_l472_472624

-- Variables and conditions
variables {R a : ℝ}
constant volume_of_sphere : ℝ := 256 * π / 3
constant radius_of_sphere : R = 4
constant edge_length_of_cube : a = 8 / Real.sqrt 3

-- Statement to prove
theorem surface_area_of_inscribed_cube : 6 * a^2 = 128 :=
by
  sorry

end surface_area_of_inscribed_cube_l472_472624


namespace odd_prime_2wy_factors_l472_472251

theorem odd_prime_2wy_factors (w y : ℕ) (h1 : Nat.Prime w) (h2 : Nat.Prime y) (h3 : ¬ Even w) (h4 : ¬ Even y) (h5 : w < y) (h6 : Nat.totient (2 * w * y) = 8) :
  w = 3 :=
sorry

end odd_prime_2wy_factors_l472_472251


namespace hyperbola_enclosed_area_is_correct_l472_472643

noncomputable def hyperbola_area_enclosed_by_asymptotes : ℝ :=
let a := 2 in 
let b := 3 in 
let c := Real.sqrt (a^2 + b^2) in 
let x_asymptote := a^2 / c in
let y1_asymptote := b/a * x_asymptote in
let y2_asymptote := -b/a * x_asymptote in
let vertex1 := (x_asymptote, y1_asymptote) in
let vertex2 := (x_asymptote, y2_asymptote) in 
1/2 * x_asymptote * (y1_asymptote - y2_asymptote)

theorem hyperbola_enclosed_area_is_correct : hyperbola_area_enclosed_by_asymptotes = 24 / 13 := 
sorry

end hyperbola_enclosed_area_is_correct_l472_472643


namespace smallest_positive_period_f_is_pi_f_not_symmetric_about_minus_pi_six_zeros_of_f_f_not_monotonically_increasing_l472_472975

open Real

-- Define the function sin2x_plus_cos2x_pi_six
def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x + π / 6)

-- Prove the smallest positive period of f(x) is π
theorem smallest_positive_period_f_is_pi :
  ∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x := 
  sorry

-- Prove that f(x) is not symmetric about x = -π/6
theorem f_not_symmetric_about_minus_pi_six : 
  ∀ x, f x ≠ f (- x - π / 6) := 
  sorry

-- Prove the zeros of f(x) are {x | x = k * π / 2 - π / 6, k ∈ ℤ}
theorem zeros_of_f : 
  ∀ x, (∃ k : ℤ, x = k * π / 2 - π / 6) ↔ f x = 0 := 
  sorry

-- Prove intervals where f(x) is monotonically increasing are NOT [-5π/6 + kπ, π/6 + kπ] (k ∈ ℤ)
theorem f_not_monotonically_increasing :
  ∀ k : ℤ, ¬ ∀ x, -5 * π / 6 + k * π ≤ x ∧ x ≤ π / 6 + k * π →
    (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ < f x₂) :=
  sorry

end smallest_positive_period_f_is_pi_f_not_symmetric_about_minus_pi_six_zeros_of_f_f_not_monotonically_increasing_l472_472975


namespace distance_to_other_focus_l472_472211

-- Define the standard form of the ellipse and the distances to the foci.
def ellipse (x y : ℝ) := (x^2/100) + (y^2/81) = 1

-- The problem is to prove the distance from P to the other focus
theorem distance_to_other_focus
  (P : ℝ × ℝ)
  (hx : ellipse P.1 P.2)
  (d1 : ℝ)
  (hd1 : d1 = 6)
  (a : ℝ)
  (ha : a = 10) :
  ∃ d2 : ℝ, d1 + d2 = 2 * a ∧ d2 = 14 :=
by 
  use 14
  split
  · linarith
  · rfl

end distance_to_other_focus_l472_472211


namespace degree_of_polynomial10_l472_472783

-- Definition of the degree function for polynomials.
def degree (p : Polynomial ℝ) : ℕ := p.natDegree

-- Given condition: the degree of the polynomial 5x^3 + 7 is 3.
def polynomial1 := (Polynomial.C 5) * (Polynomial.X ^ 3) + (Polynomial.C 7)
axiom degree_poly1 : degree polynomial1 = 3

-- Statement to prove:
theorem degree_of_polynomial10 : degree (polynomial1 ^ 10) = 30 :=
by
  sorry

end degree_of_polynomial10_l472_472783


namespace twice_total_credits_is_190_l472_472854

def credits (Emily Spencer Aria Hannah : ℕ) : Prop :=
  Spencer = Emily / 2 ∧
  Aria = 5 * Emily / 2 ∧
  Hannah = 3 * Spencer / 2 ∧
  Emily = 20

theorem twice_total_credits_is_190 (Emily Spencer Aria Hannah : ℕ) 
  (h : credits Emily Spencer Aria Hannah) : 
  2 * (Aria + Emily + Spencer + Hannah) = 190 :=
by
  simp [credits] at h
  cases h with hSpencer h1
  cases h1 with hAria h2
  cases h2 with hHannah hEmily
  -- Each individual condition case is dealt with.
  sorry

end twice_total_credits_is_190_l472_472854


namespace inning_is_31_l472_472822

noncomputable def inning_number (s: ℕ) (i: ℕ) (a: ℕ) : ℕ := s - a + i

theorem inning_is_31
  (batsman_runs: ℕ)
  (increase_average: ℕ)
  (final_average: ℕ) 
  (n: ℕ) 
  (h1: batsman_runs = 92)
  (h2: increase_average = 3)
  (h3: final_average = 44)
  (h4: 44 * n - 92 = 41 * n): 
  inning_number 44 1 3 = 31 := 
by 
  sorry

end inning_is_31_l472_472822


namespace systematic_sampling_l472_472765

theorem systematic_sampling (N : ℕ) (k : ℕ) (interval : ℕ) (seq : List ℕ) : 
  N = 70 → k = 7 → interval = 10 → 
  seq = [3, 13, 23, 33, 43, 53, 63] := 
by 
  intros hN hk hInt;
  sorry

end systematic_sampling_l472_472765


namespace log_eq_three_l472_472075

theorem log_eq_three :
  log 2 (1 / 4) + log 2 32 = log 2 8 := by
sorry

end log_eq_three_l472_472075


namespace Jamie_earnings_l472_472656

theorem Jamie_earnings
  (earn_per_hour : ℕ)
  (days_per_week : ℕ)
  (hours_per_day : ℕ)
  (weeks : ℕ)
  (earnings : ℕ) :
  earn_per_hour = 10 →
  days_per_week = 2 →
  hours_per_day = 3 →
  weeks = 6 →
  earnings = earn_per_hour * days_per_week * hours_per_day * weeks →
  earnings = 360 :=
  by
    intros
    rw [H, H_1, H_2, H_3]
    have h1 : 10 * 2 = 20 := by norm_num
    have h2 : 20 * 3 = 60 := by norm_num
    have h3 : 60 * 6 = 360 := by norm_num
    rw [h1, h2, h3]
    exact H_4

end Jamie_earnings_l472_472656


namespace equal_integrals_segments_l472_472163

-- Define the intervals for the black and white segments as per the solution
def interval1_black : set ℝ := {x | -1 ≤ x ∧ x ≤ -1/2 ∨ 1/2 ≤ x ∧ x ≤ 1}
def interval1_white : set ℝ := {x | -1/2 < x ∧ x < 1/2}

def interval2_white : set ℝ := {x | -1 ≤ x ∧ x ≤ -3/4 ∨ -1/4 ≤ x ∧ x ≤ 0 ∨ 1/4 ≤ x ∧ x ≤ 3/4}
def interval2_black : set ℝ := {x | -3/4 < x ∧ x < -1/4 ∨ 0 < x ∧ x < 1/4 ∨ 3/4 < x ∧ x < 1}

-- Integral equality condition for linear functions
def linear_integral_equal (f : ℝ → ℝ) (a b c d : ℝ) : Prop :=
  ∫ x in a..b, f x = ∫ x in c..d, f x

-- Integral equality condition for quadratic polynomials
def quadratic_integral_equal (f : ℝ → ℝ) (a b c d e f' g h i : ℝ) : Prop :=
  ∫ x in a..b, f x + ∫ x in c..d, f x + ∫ x in e..f', f x =
  ∫ x in g..h, f x + ∫ x in i, f x

-- The main theorem statement
theorem equal_integrals_segments :
  (∀ f : ℝ → ℝ, (is_linear f ∧ linear_integral_equal f (-1) (-1/2) (1/2) 1) ∧
                 (is_quadratic f ∧ quadratic_integral_equal f (-1) (-3/4) (-1/4) 0 (1/4) (3/4))) :=
sorry

end equal_integrals_segments_l472_472163


namespace tangent_intersection_x_l472_472827

theorem tangent_intersection_x :
  ∃ x : ℝ, 
    0 < x ∧ (∃ r1 r2 : ℝ, 
     (r1 = 3) ∧ 
     (r2 = 8) ∧ 
     (0, 0) = (0, 0) ∧ 
     (18, 0) = (18, 0) ∧
     (∀ t : ℝ, t > 0 → t = x / (18 - x) → t = r1 / r2) ∧ 
      x = 54 / 11) := 
sorry

end tangent_intersection_x_l472_472827


namespace find_m_l472_472572

noncomputable def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
noncomputable def B (m : ℝ) : Set ℝ := {3, m^2}

theorem find_m (m : ℝ) : B m ⊆ A m ↔ m = 1 ∨ m = 3 :=
by
  sorry

end find_m_l472_472572


namespace increase_expenditure_by_10_percent_l472_472333

variable (I : ℝ) (P : ℝ)
def E := 0.75 * I
def I_new := 1.20 * I
def S_new := 1.50 * (I - E)
def E_new := E * (1 + P / 100)

theorem increase_expenditure_by_10_percent :
  (E_new = 0.75 * I * (1 + P / 100)) → P = 10 :=
by
  sorry

end increase_expenditure_by_10_percent_l472_472333


namespace minimum_value_of_x_plus_y_l472_472545

theorem minimum_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (x - 1) * (y - 1) = 1) : x + y = 4 :=
sorry

end minimum_value_of_x_plus_y_l472_472545


namespace max_neg_integers_l472_472809

theorem max_neg_integers (a b c d e f : ℤ) (h : ab + cdef < 0) : 
  ∃ w ≤ 6, w = 4 ∧ (∃ l : List ℤ, l.length = 6 ∧ List.countp (· < 0) l = w) :=
sorry

end max_neg_integers_l472_472809


namespace angle_between_vectors_acute_l472_472541

open Real

noncomputable def vector_angle {p q : ℝ × ℝ} : ℝ :=
  let dp := (p.1 * q.1 + p.2 * q.2)
  let np := (sqrt ((p.1 ^ 2 + p.2 ^ 2) * (q.1 ^ 2 + q.2 ^ 2)))
  arccos (dp / np)

theorem angle_between_vectors_acute (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (hSum : A + B + C = π)
  (p := (1 + sin A, 1 + cos A))
  (q := (1 + sin B, -1 - cos B)) :
  0 < vector_angle p q ∧ vector_angle p q < π / 2 :=
sorry

end angle_between_vectors_acute_l472_472541


namespace statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_correct_l472_472804

open Complex

/-!
  Define the propositions to reflect the correctness of statements given in problem 
-/

-- Prove that A is incorrect
theorem statement_A_incorrect : ¬ (∀ z : ℂ, (z^2 ∈ ℝ) → (z ∈ ℝ)) :=
by sorry

-- Prove that B is correct
theorem statement_B_correct : ∀ z : ℂ, (↑(Complex.i / z) ∈ ℝ) → (∃ b : ℝ, z = 0 + b * Complex.i) :=
by sorry

-- Prove that C is incorrect
theorem statement_C_incorrect : ¬ (∀ z1 z2 : ℂ, (|z1| = |z2|) → (z1 = z2 ∨ z1 = -z2)) :=
by sorry

-- Prove that D is correct
theorem statement_D_correct : ∀ z1 z2 : ℂ, (z1 * z2 = |z1|^2 ∧ z1 ≠ 0) → |z1| = |z2| :=
by sorry

end statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_correct_l472_472804


namespace parallel_lines_distance_l472_472770

noncomputable def distance_between_parallel_lines (a₁ b₁ c₁ : ℝ) (a₂ b₂ c₂ : ℝ) (h : a₁ / a₂ = b₁ / b₂) : ℝ :=
  abs (c₂ - c₁) / sqrt (a₁ ^ 2 + b₁ ^ 2)

theorem parallel_lines_distance :
  distance_between_parallel_lines 2 3 1 4 6 7 (by norm_num) = 9 / (2 * sqrt 13) :=
by
  sorry

end parallel_lines_distance_l472_472770


namespace extreme_value_at_one_zero_interval_l472_472220

-- Definitions used for the conditions in problem a)
def f_a (a x : ℝ) := (Real.log x + a) / x - 1

-- Statement for the first proof problem
theorem extreme_value_at_one (x : ℝ) (h : f_a 1 x = 0) : x = 1 := sorry

-- Definitions for the second problem's conditions and statement
def has_zero_in_interval (a : ℝ) := ∃ x, 0 < x ∧ x ≤ Real.exp 1 ∧ f_a a x = 0

-- Statement for the second proof problem
theorem zero_interval (a : ℝ) (h : has_zero_in_interval a) : 1 ≤ a := sorry

end extreme_value_at_one_zero_interval_l472_472220


namespace num_divisors_not_div_by_3_l472_472597

theorem num_divisors_not_div_by_3 : 
  let n := 180 in
  let prime_factorization_180 := factorization 180 in
  (prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180) →
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)} in
  divisors_not_div_by_3.card = 6 :=
by 
  let n := 180
  let prime_factorization_180 := factorization n
  have h_factorization : prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180 := -- proof ommitted
    sorry
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)}
  have h_card : divisors_not_div_by_3.card = 6 := -- proof ommitted
    sorry
  exact h_card

end num_divisors_not_div_by_3_l472_472597


namespace general_term_formula_sum_inverse_S_n_l472_472941

-- Let {a_n} be an arithmetic sequence
variable (a: ℕ → ℕ)
variable (S: ℕ → ℕ)

-- Given conditions
axiom h1 : a 1 + a 3 = 10
axiom h2 : S 4 = 24

-- Definition of sequences in terms of common difference and first term
def a_n_term (n d: ℕ) (a1: ℕ) := a1 + (n - 1) * d

-- Definition of sum of first n terms of arithmetic sequence
def sum_arith_seq (n d: ℕ) (a1: ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2

-- Main goals to prove
theorem general_term_formula (a1 d : ℕ) (ha1 : a1 = 3) (hd : d = 2):
  ∀ n, a n = 2 * n + 1 :=
sorry

theorem sum_inverse_S_n (a1: ℕ) (d: ℕ) (ha1 : a1 = 3) (hd : d = 2):
  ∀ n, (∑ k in Finset.range n, 1 / S (k + 1)) = 3/4 - 1/2 * (1 / (n + 1) + 1 / (n + 2)) :=
sorry

end general_term_formula_sum_inverse_S_n_l472_472941


namespace cards_in_unfilled_box_l472_472659

theorem cards_in_unfilled_box : ∀ (total_cards boxes_capacity : ℕ), 
  total_cards = 94 → boxes_capacity = 8 → 
  (∃ (unfilled_cards : ℕ), unfilled_cards = total_cards % boxes_capacity ∧ unfilled_cards = 6) :=
by
  intros total_cards boxes_capacity h1 h2
  use total_cards % boxes_capacity
  split
  · rfl
  · sorry

end cards_in_unfilled_box_l472_472659


namespace gcd_a_b_eq_1023_l472_472154

def a : ℕ := 2^1010 - 1
def b : ℕ := 2^1000 - 1

theorem gcd_a_b_eq_1023 : Nat.gcd a b = 1023 := 
by
  sorry

end gcd_a_b_eq_1023_l472_472154


namespace problem_m_value_l472_472616

noncomputable def find_m (m : ℝ) : Prop :=
  let a : ℝ := real.sqrt (10 - m)
  let b : ℝ := real.sqrt (m - 2)
  (2 * real.sqrt (a^2 - b^2) = 4) ∧ (10 - m > m - 2) ∧ (m - 2 > 0) ∧ (10 - m > 0)

theorem problem_m_value (m : ℝ) : find_m m → m = 4 := by
  sorry

end problem_m_value_l472_472616


namespace total_candles_this_year_l472_472319

-- Defining the ages of the children this year
def age_child (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | 5 => 6  -- since they are twins
  | 6 => 6  -- same as above
  | _ => 0  -- no other children

-- Sum of candles this year
def candles_this_year : ℕ :=
  (List.range 7).map age_child |>.sum

-- Sum of candles two years ago
def candles_two_years_ago : ℕ :=
  6 + (List.range 6).map (λ n, age_child (n+1) - 2) |>.sum

-- Total candles condition
def total_candles_condition : Prop :=
  candles_this_year = 2 * candles_two_years_ago

-- Theorem statement for the number of candles this year
theorem total_candles_this_year (h : total_candles_condition) : candles_this_year = 26 :=
by sorry

end total_candles_this_year_l472_472319


namespace overlapping_circumference_l472_472400

-- Let's define the conditions
def disk_radius := 1
def distance_between_centers := 1

-- The goal is to compute the circumference of the overlapping region
theorem overlapping_circumference:
  (r₁ r₂: ℝ) (r₁ = disk_radius ∧ r₂ = disk_radius)
  (dist_centers: ℝ) (dist_centers = distance_between_centers):
  (overlapping_circumference: ℝ) 
  overlapping_circumference = 4 * Real.pi / 3 := 
  sorry

end overlapping_circumference_l472_472400


namespace frank_cookies_l472_472928

theorem frank_cookies (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (h1 : Millie_cookies = 4)
  (h2 : Mike_cookies = 3 * Millie_cookies)
  (h3 : Frank_cookies = Mike_cookies / 2 - 3)
  : Frank_cookies = 3 := by
  sorry

end frank_cookies_l472_472928


namespace f_at_6_eq_8_l472_472309

def f : ℝ → ℝ
| x => if x ≥ 10 then x - 2 else f (f (x + 6))

theorem f_at_6_eq_8 : f 6 = 8 := by
  sorry

end f_at_6_eq_8_l472_472309


namespace bottom_row_correct_l472_472896

theorem bottom_row_correct (A : Fin 4 → Fin 4 → Fin 4) :
  (∀ i j, A i j ∈ {1, 2, 3, 4}) ∧
  (∀ i, ∀ j1 j2, j1 ≠ j2 → A i j1 ≠ A i j2) ∧
  (∀ j, ∀ i1 i2, i1 ≠ i2 → A i1 j ≠ A i2 j) ∧
  (A 0 0 + A 0 1 = 3) ∧
  (A 1 0 + A 1 1 = 6) ∧
  (A 2 0 + A 2 1 = 5) →
  (A 3 0 = 2) ∧ (A 3 1 = 1) ∧ (A 3 2 = 4) ∧ (A 3 3 = 3) →
  A 3 0 * 1000 + A 3 1 * 100 + A 3 2 * 10 + A 3 3 = 2143 :=
by
  sorry

end bottom_row_correct_l472_472896


namespace villager_travel_by_motorcycle_fraction_l472_472109

theorem villager_travel_by_motorcycle_fraction {v : ℝ} (h : v > 0) :
  ∃ (x : ℝ), 
    x = 1 / 6 ∧ 
    (1 - x) / 1 = 5 / 6 :=
by {
  use (1 / 6),
  split,
  { norm_num },
  simp,
  sorry
}

end villager_travel_by_motorcycle_fraction_l472_472109


namespace max_salad_servings_l472_472007

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l472_472007


namespace number_of_true_propositions_l472_472498

open Real

-- Define the conditions
def discriminant (m : ℝ) : ℝ := 1 + 4 * m

-- Define the original statement
def original_statement (m : ℝ) : Prop := m > 0 → discriminant m > 0

-- Define the converse of the statement
def converse_statement (m : ℝ) : Prop := (discriminant m > 0) → (m > 0)

-- Define the inverse of the statement
def inverse_statement (m : ℝ) : Prop := (m ≤ 0) → (discriminant m ≤ 0)

-- Define the contrapositive of the statement
def contrapositive_statement (m : ℝ) : Prop := (discriminant m ≤ 0) → (m ≤ 0)

-- Lean statement to prove the number of true propositions
theorem number_of_true_propositions (m : ℝ) :
  (if original_statement m then 1 else 0) +
  (if converse_statement m then 1 else 0) +
  (if inverse_statement m then 1 else 0) +
  (if contrapositive_statement m then 1 else 0) = 2 :=
sorry

end number_of_true_propositions_l472_472498


namespace good_subset_count_l472_472741

def S : Finset ℕ := Finset.range 1991

def is_good_subset (A : Finset ℕ) : Prop :=
  A.card = 31 ∧ (A.sum id) % 5 = 0

noncomputable def number_of_good_subsets : ℕ :=
  (Finset.choose 1990 31 / 5)

theorem good_subset_count :
  (Finset.filter is_good_subset (Finset.powersetLen 31 S)).card = number_of_good_subsets :=
sorry

end good_subset_count_l472_472741


namespace mary_initial_amount_l472_472466

theorem mary_initial_amount (current_amount pie_cost mary_after_pie : ℕ) 
  (h1 : pie_cost = 6) 
  (h2 : mary_after_pie = 52) :
  current_amount = pie_cost + mary_after_pie → 
  current_amount = 58 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end mary_initial_amount_l472_472466


namespace probability_divisible_by_5_l472_472381

def is_three_digit_integer (M : ℕ) : Prop :=
  100 ≤ M ∧ M < 1000

def ones_digit_is_4 (M : ℕ) : Prop :=
  (M % 10) = 4

theorem probability_divisible_by_5 (M : ℕ) (h1 : is_three_digit_integer M) (h2 : ones_digit_is_4 M) :
  (∃ p : ℚ, p = 0) :=
by
  sorry

end probability_divisible_by_5_l472_472381


namespace valid_outfit_combinations_l472_472240

-- Definitions for the conditions
def num_shirts := 7
def num_pants := 5
def num_hats := 7
def colors_pants := {'tan', 'black', 'blue', 'gray', 'red'}
def colors_shirts := colors_pants ∪ {'white', 'yellow'}
def colors_hats := colors_pants ∪ {'white', 'yellow'}

-- Proof statement
theorem valid_outfit_combinations : 
  let total_outfits := num_shirts * num_pants * num_hats in
  let matching_outfits := colors_pants.size in
  total_outfits - matching_outfits = 240 :=
by
  sorry

end valid_outfit_combinations_l472_472240


namespace correct_propositions_l472_472058

open Real -- Access real number functions and properties

-- Definitions for propositions
def propositionA (x : ℝ) : Prop := (x > 1) → (x^2 > 1) ∧ ¬((x^2 > 1) → (x > 1))
def propositionB : Prop := ¬(∀ x ∈ Ioo 0 (π / 2), sin x > cos x) = ∃ x₀ ∈ Ioo 0 (π / 2), sin x₀ ≤ cos x₀
def propositionC : Prop := ∀ x, sin (x + (π / 2)) = -cos x
def propositionD : Prop := cos 1 < cos (1 * (π / 180)) -- 1 degree to radians

-- The set of true propositions should be { A, B, D }
def correctOptions : Set (Prop) := { propositionA 2, propositionB, propositionD } -- Proposition A is true for x = 2 > 1

theorem correct_propositions : correctOptions = { true } :=
by sorry

end correct_propositions_l472_472058


namespace minimum_phrases_to_study_for_90_percent_score_l472_472607

theorem minimum_phrases_to_study_for_90_percent_score (total_phrases : ℕ) (score_percentage : ℕ) :
  total_phrases = 800 → score_percentage = 90 → (∃ (min_phrases : ℕ), min_phrases = 720) :=
by
  -- Definitions for the conditions from a)
  assume h1 : total_phrases = 800
  assume h2 : score_percentage = 90
  
  -- The proof will reside here, but we use sorry as a placeholder
  use 720
  sorry

end minimum_phrases_to_study_for_90_percent_score_l472_472607


namespace degree_of_poly_l472_472790

-- Define the polynomial and its degree
def inner_poly := (5 : ℝ) * (X ^ 3) + (7 : ℝ)
def poly := inner_poly ^ 10

-- Statement to prove
theorem degree_of_poly : polynomial.degree poly = 30 :=
sorry

end degree_of_poly_l472_472790


namespace algebraic_expression_value_l472_472241

theorem algebraic_expression_value (a b : ℝ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3 / 2 := by
  sorry

end algebraic_expression_value_l472_472241


namespace greatest_possible_average_speed_l472_472585

theorem greatest_possible_average_speed
  (start_palindrome : Nat)
  (end_palindrome : Nat)
  (time : Nat)
  (speed_limit : Nat)
  (max_distance : Nat)
  (reader_reads_palindrome : Nat → Bool)
  (d1 d2 d3 : Nat)
  (p1 : reader_reads_palindrome start_palindrome = true)
  (p2 : reader_reads_palindrome end_palindrome = true)
  (t : time = 2)
  (sl : speed_limit = 65)
  (md : max_distance = speed_limit * time)
  (d1_dist : end_palindrome - start_palindrome = d1)
  (d2_dist : end_palindrome - start_palindrome = d2)
  (d3_dist : end_palindrome - start_palindrome = d3)
  (max_palindrome : d1 <= max_distance ∨ d2 <= max_distance ∨ d3 <= max_distance) :
  (max (d1, d2, d3) / time = 50) :=
by 
  sorry

end greatest_possible_average_speed_l472_472585


namespace Walter_allocates_for_school_l472_472405

open Nat

def Walter_works_5_days_a_week := 5
def Walter_earns_per_hour := 5
def Walter_works_per_day := 4
def Proportion_for_school := 3/4

theorem Walter_allocates_for_school :
  let daily_earnings := Walter_works_per_day * Walter_earns_per_hour
  let weekly_earnings := daily_earnings * Walter_works_5_days_a_week
  let school_allocation := weekly_earnings * Proportion_for_school
  school_allocation = 75 := by
  sorry

end Walter_allocates_for_school_l472_472405


namespace sigma_condition_iff_powers_same_prime_l472_472919

def σ (N : ℕ) : ℕ := (Finset.range (N+1)).filter (λ d, N % d = 0).sum id

theorem sigma_condition_iff_powers_same_prime (m n : ℕ) (h_ge : m ≥ n) (h2 : n ≥ 2)
  (h_cond : (σ m - 1) / (m - 1) = (σ n - 1) / (n - 1) ∧ (σ m - 1) / (m - 1) = (σ (m * n) - 1) / (m * n - 1)) :
  ∃ p e f : ℕ, p.prime ∧ m = p ^ e ∧ n = p ^ f :=
sorry

end sigma_condition_iff_powers_same_prime_l472_472919


namespace number_of_divisors_not_divisible_by_3_l472_472589

def prime_factorization (n : ℕ) : Prop :=
  n = 2 ^ 2 * 3 ^ 2 * 5

def is_not_divisible_by (n d : ℕ) : Prop :=
  ¬ (d ∣ n)

def positive_divisors_not_divisible_by_3 (n : ℕ) : ℕ :=
  (finset.range (2 + 1)).filter (λ a, ∀ d : ℕ, is_not_divisible_by (2 ^ a * d) 3).card

theorem number_of_divisors_not_divisible_by_3 :
  prime_factorization 180 → positive_divisors_not_divisible_by_3 180 = 6 :=
by
  intro h
  sorry

end number_of_divisors_not_divisible_by_3_l472_472589


namespace max_salad_servings_l472_472008

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l472_472008


namespace simplify_and_evaluate_l472_472711

theorem simplify_and_evaluate (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 2) (h₃ : x ≠ -2) :
  ((x - 1 - 3 / (x + 1)) / ((x^2 - 4) / (x^2 + 2 * x + 1))) = x + 1 ∧ ((x = 1) → (x + 1 = 2)) :=
by
  sorry

end simplify_and_evaluate_l472_472711


namespace cos_double_angle_l472_472242

theorem cos_double_angle (x : Real) (h : sin (-x) = sqrt 3 / 2) : cos (2 * x) = -1 / 2 :=
sorry

end cos_double_angle_l472_472242


namespace imo_34_l472_472308

-- Define the input conditions
variables (R r ρ : ℝ)

-- The main theorem we need to prove
theorem imo_34 { R r ρ : ℝ } (hR : R = 1) : 
  ρ ≤ 1 - (1/3) * (1 + r)^2 :=
sorry

end imo_34_l472_472308


namespace zoey_finished_on_monday_l472_472065

def total_days_read (n : ℕ) : ℕ :=
  2 * ((2^n) - 1)

def day_of_week_finished (start_day : ℕ) (total_days : ℕ) : ℕ :=
  (start_day + total_days) % 7

theorem zoey_finished_on_monday :
  day_of_week_finished 1 (total_days_read 18) = 1 :=
by
  sorry

end zoey_finished_on_monday_l472_472065


namespace count_congruent_to_9_mod_14_lt_500_l472_472238

theorem count_congruent_to_9_mod_14_lt_500 : 
  ∃ n : ℕ, n = 36 ∧ ∀ k : ℕ, (9 + 14 * k) < 500 → 1 ≤ 9 + 14 * k ∧ 9 + 14 * k < 500 :=
by
  let n := 36
  use n
  split
  { refl }
  { intro k
    intro h
    split
    sorry -- We will need to prove that each of the numbers is positive
    sorry -- We will need to prove that each of the numbers less than 500
}

end count_congruent_to_9_mod_14_lt_500_l472_472238


namespace determine_integer_n_l472_472887

theorem determine_integer_n (n : ℤ) :
  (n + 15 ≥ 16) ∧ (-5 * n < -10) → n = 3 :=
by
  sorry

end determine_integer_n_l472_472887


namespace num_positive_divisors_not_divisible_by_3_l472_472592

theorem num_positive_divisors_not_divisible_by_3 (n : ℕ) (h : n = 180) : 
  (∃ (divisors : finset ℕ), (∀ d ∈ divisors, d ∣ n ∧ ¬ (3 ∣ d)) ∧ finset.card divisors = 6) := 
by
  have prime_factors : (n = 2^2 * 3^2 * 5) := by norm_num [h]
  sorry

end num_positive_divisors_not_divisible_by_3_l472_472592


namespace sequence_term_500_l472_472262

theorem sequence_term_500 :
  ∃ (a : ℕ → ℤ), 
  a 1 = 1001 ∧
  a 2 = 1005 ∧
  (∀ n, 1 ≤ n → (a n + a (n+1) + a (n+2)) = 2 * n) → 
  a 500 = 1334 := 
sorry

end sequence_term_500_l472_472262


namespace unicorn_rope_problem_l472_472107

theorem unicorn_rope_problem
  (d e f : ℕ)
  (h_prime_f : Prime f)
  (h_d : d = 75)
  (h_e : e = 450)
  (h_f : f = 3)
  : d + e + f = 528 := by
  sorry

end unicorn_rope_problem_l472_472107


namespace problem_part_one_problem_part_two_l472_472990

-- Step 1: Define vectors m and n, and state the condition that they are parallel
def vec_m (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)
def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, -1 / 2)

def are_parallel (m n : ℝ × ℝ) : Prop := 
  m.1 / n.1 = m.2 / n.2

-- Step 2: Define the condition related to the triangle and the function f
variables {a b c A B : ℝ}

axiom acute_angle_triangle (h : a * Real.sin A / c = 1) 
    (h1 : Real.sin B > 0) : 0 < A ∧ A < Real.pi / 2 ∧ 
                            0 < B ∧ B < Real.pi / 2 ∧ 
                            0 < (Real.pi / 2 - A - B)

lemma triangle_condition 
    (sqrt3_c_eq : sqrt 3 * c = 2 * a * sin (A + B)) : 
    sin A = sqrt 3 / 2 ↔ A = pi / 3 := sorry 

noncomputable def f (B : ℝ) (k : ℝ) : ℝ := 
  let m := vec_m B
  let n := vec_n k
  (m.1 + n.1) * m.1 + (m.2 + n.2) * m.2

-- Proof Problem 1: Prove the given fraction equals -3sqrt(3)/2
theorem problem_part_one (x : ℝ) (h1 : are_parallel (vec_m x) (vec_n x)) :
  (sqrt 3 * sin x + cos x) / (sin x - sqrt 3 * cos x) = -3 * sqrt 3 / 2 := 
sorry
  
-- Proof Problem 2: Prove the range of f(B)
theorem problem_part_two (h1 : sqrt 3 * c = 2 * a * sin (A + B)) 
    (h2 : acute_angle_triangle (a * sin A / c) (sin B)) :
  (3 / 2 : ℝ) < f B A ∧ f B A < (3 : ℝ) := sorry

end problem_part_one_problem_part_two_l472_472990


namespace volume_of_S_l472_472878

theorem volume_of_S' :
  let S' := {p : ℝ × ℝ × ℝ |
              (p.1 + 2 * p.2 ≤ 1) ∧ (2 * p.1 + p.3 ≤ 1) ∧
              (p.2 + 2 * p.3 ≤ 1) ∧ (p.1 ≥ 0) ∧ (p.2 ≥ 0) ∧ (p.3 ≥ 0)} in
  (∃! v : ℝ, v = 1 / 48 ∧
    v = (1/6) * ∫∫∫ (λ x y z, 1) (set.prod (set.prod {x | 0 ≤ x} {y | 0 ≤ y}) {z | 0 ≤ z}) (S' x y z)) :=
by sorry

end volume_of_S_l472_472878


namespace convert_kmph_to_mps_l472_472424

theorem convert_kmph_to_mps (speed_kmph : ℝ) (km_to_m : ℝ) (hr_to_s : ℝ) : 
  speed_kmph = 56 → km_to_m = 1000 → hr_to_s = 3600 → 
  (speed_kmph * (km_to_m / hr_to_s) : ℝ) = 15.56 :=
by
  intros
  sorry

end convert_kmph_to_mps_l472_472424


namespace correct_option_is_D_l472_472525

noncomputable def expression1 (a b : ℝ) : Prop := a + b > 2 * b^2
noncomputable def expression2 (a b : ℝ) : Prop := a^5 + b^5 > a^3 * b^2 + a^2 * b^3
noncomputable def expression3 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * (a - b - 1)
noncomputable def expression4 (a b : ℝ) : Prop := (b / a) + (a / b) > 2

theorem correct_option_is_D (a b : ℝ) (h : a ≠ b) : 
  (expression3 a b ∧ ¬expression1 a b ∧ ¬expression2 a b ∧ ¬expression4 a b) :=
by
  sorry

end correct_option_is_D_l472_472525


namespace line_tangent_to_ellipse_l472_472620

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1 → m^2 = 35/9) := 
sorry

end line_tangent_to_ellipse_l472_472620


namespace pair_a_n_uniq_l472_472496

theorem pair_a_n_uniq (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_eq : 3^n = a^2 - 16) : a = 5 ∧ n = 2 := 
by 
  sorry

end pair_a_n_uniq_l472_472496


namespace even_number_of_odd_degree_vertices_l472_472737

variables {V : Type*} [Fintype V] (G : SimpleGraph V)

def odd_degree_set (G : SimpleGraph V) : Finset V :=
  Finset.filter (λ v, Finset.card (G.neighborFinset v) % 2 = 1) Finset.univ

theorem even_number_of_odd_degree_vertices (G : SimpleGraph V) :
  (odd_degree_set G).card % 2 = 0 :=
sorry

end even_number_of_odd_degree_vertices_l472_472737


namespace total_dots_not_visible_l472_472178

theorem total_dots_not_visible (visible : list ℕ) (total_dots_per_die : ℕ) (number_of_dice : ℕ) :
  (visible = [2, 2, 3, 4, 4, 5, 6, 6]) →
  (total_dots_per_die = 21) →
  (number_of_dice = 4) →
  ((total_dots_per_die * number_of_dice) - visible.sum = 52) :=
by
  sorry

end total_dots_not_visible_l472_472178


namespace mashed_potatoes_suggestions_l472_472351

-- Define the number of students who suggested tomatoes
def T : ℕ := 79

-- Define the equation relating the students who suggested mashed potatoes and tomatoes
def M : ℕ := T + 65

-- Proof statement
theorem mashed_potatoes_suggestions : M = 144 :=
by
  have h1 : T = 79 := rfl
  have h2 : M = T + 65 := rfl
  rw [h1, h2]
  norm_num
  done

end mashed_potatoes_suggestions_l472_472351


namespace age_difference_l472_472077

theorem age_difference (J P : ℕ) 
  (h1 : P = 16 - 10) 
  (h2 : P = (1 / 3) * J) : 
  (J + 10) - 16 = 12 := 
by 
  sorry

end age_difference_l472_472077


namespace find_unit_price_B_l472_472324

/-- Definitions based on the conditions --/
def total_cost_A := 7500
def total_cost_B := 4800
def quantity_difference := 30
def price_ratio : ℝ := 2.5

/-- Define the variable x as the unit price of B type soccer balls --/
def unit_price_B (x : ℝ) : Prop :=
  (total_cost_A / (price_ratio * x)) + 30 = (total_cost_B / x) ∧
  total_cost_A > 0 ∧ total_cost_B > 0 ∧ x > 0

/-- The main statement to prove --/
theorem find_unit_price_B (x : ℝ) : unit_price_B x ↔ x = 60 :=
by
  sorry

end find_unit_price_B_l472_472324


namespace find_a_and_use_it_l472_472563

def f (x a : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) + a

theorem find_a_and_use_it (a : ℝ) :
  (f 0 a = 1) →
  (f (Real.log 2) a + f (Real.log (1 / 2)) a = 2) :=
by
  sorry

end find_a_and_use_it_l472_472563


namespace matthew_crackers_left_l472_472689

theorem matthew_crackers_left (total_crackers crackers_per_friend : ℕ) :
  total_crackers = 24 → crackers_per_friend = 7 → 3 * crackers_per_friend = 21 → total_crackers - 3 * crackers_per_friend = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end matthew_crackers_left_l472_472689


namespace pq_r_sum_l472_472677

theorem pq_r_sum (p q r : ℝ) (h1 : p^3 - 18 * p^2 + 27 * p - 72 = 0) 
                 (h2 : 27 * q^3 - 243 * q^2 + 729 * q - 972 = 0)
                 (h3 : 3 * r = 9) : p + q + r = 18 :=
by
  sorry

end pq_r_sum_l472_472677


namespace simplify_expression_l472_472502

variable (a b : ℝ)

theorem simplify_expression :
  (a^3 - b^3) / (a * b) - (ab - b^2) / (ab - a^3) = (a^2 + ab + b^2) / b :=
by
  sorry

end simplify_expression_l472_472502


namespace parallel_planes_conditions_l472_472933

-- Define the problem
section parallel_planes

variables {Plane : Type} [linear_order Plane]
variables {line : Plane → Plane → Prop}
variables (α β : Plane) (a b : Plane) 

-- Conditions as definitions
def condition_1 : Prop :=
∃ (a : Plane), (line a α) ∧ (line a β)

def condition_2 : Prop :=
∃ (γ : Plane), (line γ α) ∧ (line γ β)

def condition_3 : Prop :=
∃ (a b : Plane), (line a α) ∧ (line b β) ∧ (parallel a β) ∧ (parallel b α)

def condition_4 : Prop :=
∃ (a b : Plane), (line a α) ∧ (line b β) ∧ (parallel a β) ∧ (skew b α)

-- Sufficient condition statement
theorem parallel_planes_conditions (α β : Plane) :
  (condition_1 α β) ∨ (condition_4 α β) → α ∥ β :=
sorry

end parallel_planes

end parallel_planes_conditions_l472_472933


namespace probability_relatively_prime_pairs_l472_472399

open Real

-- Define the set of natural numbers in question
def S : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define a function to calculate the greatest common factor (gcd)
noncomputable def gcd (a b : ℕ) : ℕ := nat.gcd a b

-- Define the count of gcd being 1 pairs within the set
noncomputable def relatively_prime_pairs_count : ℕ :=
  (S.product S).filter (λ (p : ℕ × ℕ), p.1 < p.2 ∧ gcd p.1 p.2 = 1).card

-- Define the total two-element subsets (pairs)
noncomputable def total_pairs_count : ℕ := (S.product S).filter (λ (p : ℕ × ℕ), p.1 < p.2).card

-- Define the probability as a common fraction
theorem probability_relatively_prime_pairs : 
  (relatively_prime_pairs_count : ℝ) / (total_pairs_count : ℝ) = 3 / 4 :=
by
  sorry

end probability_relatively_prime_pairs_l472_472399


namespace expression_meaningful_l472_472004

theorem expression_meaningful (x : ℝ) : 1 / sqrt (x - 3) ∈ ℝ → x > 3 := by
  intro h
  sorry

end expression_meaningful_l472_472004


namespace fair_collection_l472_472361

theorem fair_collection 
  (children : ℕ) (fee_child : ℝ) (adults : ℕ) (fee_adult : ℝ) 
  (total_people : ℕ) (count_children : ℕ) (count_adults : ℕ)
  (total_collected: ℝ) :
  children = 700 →
  fee_child = 1.5 →
  adults = 1500 →
  fee_adult = 4.0 →
  total_people = children + adults →
  count_children = 700 →
  count_adults = 1500 →
  total_collected = (count_children * fee_child) + (count_adults * fee_adult) →
  total_collected = 7050 :=
by
  intros
  sorry

end fair_collection_l472_472361


namespace path_area_and_cost_l472_472099

-- Define the initial conditions
def field_length : ℝ := 65
def field_width : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2

-- Define the extended dimensions including the path
def extended_length := field_length + 2 * path_width
def extended_width := field_width + 2 * path_width

-- Define the areas
def area_with_path := extended_length * extended_width
def area_of_field := field_length * field_width
def area_of_path := area_with_path - area_of_field

-- Define the cost
def cost_of_constructing_path := area_of_path * cost_per_sq_m

theorem path_area_and_cost :
  area_of_path = 625 ∧ cost_of_constructing_path = 1250 :=
by
  sorry

end path_area_and_cost_l472_472099


namespace train_length_l472_472429

theorem train_length (speed_kmh : ℕ) (time_min : ℕ) (train_platform_equal : Prop) :
  speed_kmh = 126 →
  time_min = 1 →
  train_platform_equal →
  (∃ L : ℕ, L = 1050) :=
by 
  intros h1 h2 h3
  have speed_ms : ℚ := 126 * 1000 / 3600
  have time_s : ℕ := 1 * 60
  have distance : ℚ := speed_ms * time_s
  have two_L : ℚ := distance
  have L := two_L / 2
  have L_int : ℕ := trunc L
  use L_int
  sorry

end train_length_l472_472429


namespace cosine_series_sum_l472_472265

noncomputable def cosine_sum (n : ℕ) (θ : ℝ) : ℝ :=
  ∑ k in Finset.range (n + 1), if k = 0 then (0 : ℝ) else Real.cos (k * θ)

theorem cosine_series_sum {n : ℕ} {θ : ℝ} (h : θ ≠ 0) :
  cosine_sum n θ = (Real.sin (((n + 1) * θ) / 2) * Real.cos ((n * θ) / 2)) / Real.sin (θ / 2) :=
sorry

end cosine_series_sum_l472_472265


namespace minimum_surface_area_l472_472079

theorem minimum_surface_area (init_surface_area : ℕ) 
  (total_cubes : ℕ) (removed_cubes : ℕ) (min_surface_area : ℕ) :
  init_surface_area = 54 →
  total_cubes = 27 →
  removed_cubes = 5 →
  min_surface_area = 50 :=
begin
  sorry
end

end minimum_surface_area_l472_472079


namespace find_f_lg_lg2_l472_472562

def f (a b x : ℝ) : ℝ := a * x^3 + b * Real.sin x + 4

axiom lg : ℝ → ℝ
axiom log2 : ℝ → ℝ
variable (a : ℝ) (b : ℝ)
variable (h_cond : f a b (lg (log2 10)) = 5)

theorem find_f_lg_lg2 :
  f a b (lg (lg 2)) = 3 :=
sorry

end find_f_lg_lg2_l472_472562


namespace probability_three_dice_show_prime_l472_472860

theorem probability_three_dice_show_prime :
  let dice_faces := 20
  let number_of_primes := 8
  let number_of_rolls := 5
  let probability_prime := number_of_primes / dice_faces.to_rat
  let probability_non_prime := (dice_faces - number_of_primes) / dice_faces.to_rat
  ((number_of_primes / dice_faces.to_rat)^3 * 
   ((dice_faces-number_of_primes) / dice_faces.to_rat)^2 *
   Nat.choose number_of_rolls 3) = 720 / 3125 := 
by sorry

end probability_three_dice_show_prime_l472_472860


namespace num_true_propositions_eq_two_l472_472229

open Classical

theorem num_true_propositions_eq_two (p q : Prop) :
  (if (p ∧ q) then 1 else 0) + (if (p ∨ q) then 1 else 0) + (if (¬p) then 1 else 0) + (if (¬q) then 1 else 0) = 2 :=
by sorry

end num_true_propositions_eq_two_l472_472229


namespace max_salad_servings_l472_472006

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l472_472006


namespace negation_example_l472_472199

theorem negation_example :
  (¬ (∃ n : ℕ, n^2 ≥ 2^n)) → (∀ n : ℕ, n^2 < 2^n) :=
by
  sorry

end negation_example_l472_472199


namespace number_of_B_students_l472_472260

theorem number_of_B_students (total_students : ℕ) 
  (h1 : ∀ y, y * 0.8 + y + y * 1.2 = total_students) 
  (h2 : total_students = 25) : ∃ y : ℕ, y = 8 := 
by 
  sorry

end number_of_B_students_l472_472260


namespace no_possible_PS_80_l472_472757

-- Definitions based on provided conditions
def PR : ℕ := 75
def QS : ℕ := 45
def QR : ℕ := 20

-- Theorem stating the conclusion
theorem no_possible_PS_80 (dPS : ℕ) (h1 : dPS ≠ 80) : ∀ dPS = 10 ∨ dPS = 50 ∨ dPS = 100 ∨ dPS = 140 :=
by
  sorry

end no_possible_PS_80_l472_472757


namespace inequality_B_l472_472954

-- Given Conditions
variable {f : ℝ → ℝ}
variable (h_incr : ∀ x y : ℝ, x < y → f(x) ≤ f(y))
variable (a b : ℝ)
variable (h_cond : a + b ≤ 0)

-- Statement of the proof problem
theorem inequality_B :
  f(a) + f(b) ≤ f(-a) + f(-b) := 
begin
  sorry
end

end inequality_B_l472_472954


namespace max_servings_possible_l472_472021

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l472_472021


namespace tangent_line_at_five_l472_472967

variable {f : ℝ → ℝ}

theorem tangent_line_at_five 
  (h_tangent : ∀ x, f x = -x + 8)
  (h_tangent_deriv : deriv f 5 = -1) :
  f 5 = 3 ∧ deriv f 5 = -1 :=
by sorry

end tangent_line_at_five_l472_472967


namespace quadratic_unique_solution_ordered_pair_l472_472738

theorem quadratic_unique_solution_ordered_pair (a c : ℝ) 
  (h1: (∀ x: ℝ, (a * x^2 + 30 * x + c = 0) → x = -15/a)) 
  (h2: a + c = 35) 
  (h3: a < c) : (a, c) = ( (35 - 5 * real.sqrt 13) / 2 , (35 + 5 * real.sqrt 13) / 2 ) := 
sorry

end quadratic_unique_solution_ordered_pair_l472_472738


namespace red_pens_count_l472_472390

theorem red_pens_count (R : ℕ) : 
  (∃ (black_pens blue_pens : ℕ), 
  black_pens = R + 10 ∧ 
  blue_pens = R + 7 ∧ 
  R + black_pens + blue_pens = 41) → 
  R = 8 := by
  sorry

end red_pens_count_l472_472390


namespace find_m_value_l472_472618

theorem find_m_value
  (m : ℝ)
  (h1 : 10 - m > 0)
  (h2 : m - 2 > 0)
  (h3 : 2 * Real.sqrt (10 - m - (m - 2)) = 4) :
  m = 4 := by
sorry

end find_m_value_l472_472618


namespace trigonometric_identity_l472_472548

variables {α : Real.Angle}

/-- Given that α is an angle in the third quadrant, and tan α = 3/4, show that sin α = -3/5. -/
theorem trigonometric_identity (h_tan : Real.tan α = 3 / 4) (h_quadrant : α > π ∧ α < 3 * π / 2) : 
  Real.sin α = -3 / 5 :=
sorry

end trigonometric_identity_l472_472548


namespace solid_is_sphere_l472_472612

theorem solid_is_sphere (S : Type) (solid : S)
  (cross_sections_are_circles : ∀ (P : Type) (plane : P), is_cross_section_circle solid plane) :
  is_sphere solid :=
sorry

end solid_is_sphere_l472_472612


namespace value_of_first_equation_l472_472217

theorem value_of_first_equation (x y : ℚ) 
  (h1 : 5 * x + 6 * y = 7) 
  (h2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 :=
sorry

end value_of_first_equation_l472_472217


namespace ellipse_properties_l472_472191

theorem ellipse_properties
  (a b: ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (e: ℝ := 1/2)
  (ecc: e = c / a) 
  (c: ℝ)
  (h3: e = 1 / 2) 
  (foci: |PQ| = 3)
  (h4: M : (2, 0)) 
  (h5: ∀ A B, A ≠ M ∧ B ≠ M ∧ slope(M, A) * slope(M, B) = 1/4) :
  (C: ellipse := {x // x^2 / a^2 + y^2 / b^2 = 1})
  (h6: equation_C: C = ({x // x^2 / 4 + y^2 / 3 = 1}))
  (fixed_point : ∃ P: (ℝ × ℝ), on_line_AB (A, B) → P = (-4, 0)) : Prop := sorry

end ellipse_properties_l472_472191


namespace sum_alternating_series_l472_472868

theorem sum_alternating_series :
  (∑ k in Finset.range 100, (-1)^(k+1) * (k + 1)) = -50 :=
by
  sorry

end sum_alternating_series_l472_472868


namespace cosine_54_deg_l472_472135

theorem cosine_54_deg : ∃ c : ℝ, c = cos (54 : ℝ) ∧ c = 1 / 2 :=
  by 
    let c := cos (54 : ℝ)
    let d := cos (108 : ℝ)
    have h1 : d = 2 * c^2 - 1 := sorry
    have h2 : d = -c := sorry
    have h3 : 2 * c^2 + c - 1 = 0 := sorry
    use 1 / 2 
    have h4 : c = 1 / 2 := sorry
    exact ⟨cos_eq_cos_of_eq_rad 54 1, h4⟩

end cosine_54_deg_l472_472135


namespace order_of_abc_l472_472953

def a := Real.log 5 / Real.log (1/2)
def b := (1/3) ^ 0.3
def c := 2 ^ (1/5)

theorem order_of_abc : a < b ∧ b < c := by
  sorry

end order_of_abc_l472_472953


namespace parabola_touches_x_axis_at_one_point_l472_472736

theorem parabola_touches_x_axis_at_one_point (c : ℝ) :
  let y := (x : ℝ) -> x^2 + x + c
  (∃ x : ℝ, y x = 0) ∧ (∀ x1 x2 : ℝ, y x1 = 0 → y x2 = 0 → x1 = x2) ↔ c = 1/4 :=
by
  sorry

end parabola_touches_x_axis_at_one_point_l472_472736


namespace num_divisible_by_105_l472_472996

def sequence_term : ℕ → ℕ := λ n, 10^n + 5

theorem num_divisible_by_105 :
  ∀ n ≤ 2023, (∃ k ≤ n, sequence_term k % 105 = 0) → ∃ m, m = 1011 :=
by sorry

end num_divisible_by_105_l472_472996


namespace part1_solution_set_part2_range_of_a_l472_472978

-- Part (1): If a = 1, find the solution set of f(x) ≤ 7
theorem part1_solution_set (x : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |x - 1| + x) :
  f x ≤ 7 ↔ x ∈ set.Iic (4 : ℝ) := by
sorry

-- Part (2): If f(x) ≥ 2a + 1, find the range of values for a
theorem part2_range_of_a (x : ℝ) (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x a, f x = |x - a| + x) :
  (∀ x, f x ≥ 2 * a + 1) → a ≤ -1 := by
sorry

end part1_solution_set_part2_range_of_a_l472_472978


namespace triangle_ABC_no_common_factor_l472_472742

theorem triangle_ABC_no_common_factor (a b c : ℕ) (h_coprime: Nat.gcd (Nat.gcd a b) c = 1)
  (h_angleB_eq_2angleC : True) (h_b_lt_600 : b < 600) : False :=
by
  sorry

end triangle_ABC_no_common_factor_l472_472742


namespace right_triangle_count_l472_472695

theorem right_triangle_count :
  let x0_points := {(0, y) | y ∈ (finset.range 62).map (lambda i: i + 1)},
      x2_points := {(2, y) | y ∈ (finset.range 62).map (lambda i: i + 1)} in
  (count_right_triangles (x0_points ∪ x2_points)) = 7908 :=
by sorry

end right_triangle_count_l472_472695


namespace problem_solution_l472_472986

-- Define the sequence
def a (n : ℕ) : ℝ := 1 / (Real.sqrt (n + 1) + Real.sqrt n)

-- Define S_n
def S (n : ℕ) : ℝ := (Nat.range n).sum (λ k, a k)

theorem problem_solution : ∃ n : ℕ, S n = Real.sqrt 101 - 1 ∧ n = 100 :=
by
  sorry

end problem_solution_l472_472986


namespace polynomial_degree_l472_472794

def polynomial := 5 * X ^ 3 + 7
def exponent := 10
def degree_of_polynomial := 3
def final_degree := 30

theorem polynomial_degree : degree (polynomial ^ exponent) = final_degree :=
by
  sorry

end polynomial_degree_l472_472794


namespace circle_tangent_line_chord_length_intercepted_tangent_lines_through_point_l472_472209

theorem circle_tangent_line (r : ℝ) : 
  ∃ r, (∀ (x y : ℝ), x^2 + y^2 = r^2) ∧
  (∀ R, (∀ (x y : ℝ), x^2 + y^2 = R^2) →
       (∀ p q : ℝ, p - q - 2 * real.sqrt 2 = 0 → (R = 2)))
   ∧  (r = 2) ∧ 
  (radius : 2^2 = 4) := 
   sorry

theorem chord_length_intercepted (d : ℝ) : 
  ∃ d, (∀ (x y : ℝ), x^2 + y^2 = 4 ∧ 4 * x - 3 * y + 5 = 0) ∧ 
      (∀ d, (d = |5| / real.sqrt (16 + 9) → (d = 1))) ∧
  (chord_length : d = 2 * real.sqrt (4 - 1) = 2 * real.sqrt 3) := 
  sorry

theorem tangent_lines_through_point (r : ℝ) :
  ∃ r, (∀ (x y : ℝ), x^2 + y^2 = 4 ∧ (∀ p q : ℝ, (p = 1) ∧ (q = 3))) ∧ 
    (tangency_points : ∀ (M N : Point), M ≠ N ∧ (M ∈ TangentPoints) ∧ (N ∈ TangentPoints)) ∧
    (equation_of_MN : x + 3 * y - 4 = 0) := 
  sorry

end circle_tangent_line_chord_length_intercepted_tangent_lines_through_point_l472_472209


namespace number_of_valid_integers_l472_472239

theorem number_of_valid_integers :
  let valid_values := List.filter (λ c, (10 * c + 3) % 7 = 0) (List.range' 10 100)
  valid_values.length = 12 :=
by
  let valid_values : List Nat := List.filter (λ c, (10 * c + 3) % 7 = 0) (List.range' 10 90)
  have : valid_values.length = 12 := sorry

end number_of_valid_integers_l472_472239


namespace max_star_player_salary_l472_472094

-- Define the constants given in the problem
def num_players : Nat := 12
def min_salary : Nat := 20000
def total_salary_cap : Nat := 1000000

-- Define the statement we want to prove
theorem max_star_player_salary :
  (∃ star_player_salary : Nat, 
    star_player_salary ≤ total_salary_cap - (num_players - 1) * min_salary ∧
    star_player_salary = 780000) :=
sorry

end max_star_player_salary_l472_472094


namespace count_numbers_1000_to_5000_l472_472232

def countFourDigitNumbersInRange (lower upper : ℕ) : ℕ :=
  if lower <= upper then upper - lower + 1 else 0

theorem count_numbers_1000_to_5000 : countFourDigitNumbersInRange 1000 5000 = 4001 :=
by
  sorry

end count_numbers_1000_to_5000_l472_472232


namespace increasing_sequence_range_of_lambda_l472_472966

theorem increasing_sequence_range_of_lambda (a_n : ℕ → ℝ) (λ : ℝ)
  (h_def : ∀ n, a_n n = n ^ 2 + λ * n)
  (h_increasing : ∀ n, a_n n ≤ a_n (n + 1)) : λ > -3 :=
sorry

end increasing_sequence_range_of_lambda_l472_472966


namespace arithmetic_mean_of_three_digit_multiples_of_7_l472_472411

theorem arithmetic_mean_of_three_digit_multiples_of_7 :
  let first_term := 105
  let last_term := 994
  let num_terms := 128
  let sum := (num_terms / 2) * (first_term + last_term)
  let mean := sum / num_terms
  mean = 549.5 :=
by
  let first_term := 105
  let last_term := 994
  let num_terms := 128
  let sum := (num_terms / 2) * (first_term + last_term)
  let mean := sum / num_terms
  have h1 : mean = 549.5 := sorry
  exact h1

end arithmetic_mean_of_three_digit_multiples_of_7_l472_472411


namespace f_difference_l472_472681

noncomputable def f (n : ℕ) : ℝ :=
  (6 + 4 * Real.sqrt 3) / 12 * ((1 + Real.sqrt 3) / 2)^n + 
  (6 - 4 * Real.sqrt 3) / 12 * ((1 - Real.sqrt 3) / 2)^n

theorem f_difference (n : ℕ) : f (n + 1) - f n = (Real.sqrt 3 - 3) / 4 * f n :=
  sorry

end f_difference_l472_472681


namespace geometric_sum_S4_l472_472938

theorem geometric_sum_S4 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) = 2 * a n)
  (h_pos : ∀ n, a n > 0) 
  (h_sum : ∀ n, S n = ∑ i in finset.range n, a (i + 1))
  (h_arith_seq : 4 * a 3 + 2 * a 4 = 2 * a 5)
  (h_a1 : a 1 = 1) : 
  S 4 = 15 :=
by
  sorry

end geometric_sum_S4_l472_472938


namespace sum_exterior_angles_triangle_and_dodecagon_l472_472468

-- Definitions derived from conditions
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Conditions
def is_polygon (n : ℕ) : Prop := n ≥ 3

-- Proof problem statement
theorem sum_exterior_angles_triangle_and_dodecagon :
  is_polygon 3 ∧ is_polygon 12 → sum_exterior_angles 3 + sum_exterior_angles 12 = 720 :=
by
  sorry

end sum_exterior_angles_triangle_and_dodecagon_l472_472468


namespace degree_of_polynomial_l472_472876

open Polynomial

theorem degree_of_polynomial : degree ((X^3 + 1)^5 * (X^4 + 1)^2) = 23 :=
by sorry

end degree_of_polynomial_l472_472876


namespace acute_triangle_on_perpendicular_lines_l472_472575

theorem acute_triangle_on_perpendicular_lines :
  ∀ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) →
  ∃ (x y z : ℝ), (x^2 = (b^2 + c^2 - a^2) / 2) ∧ (y^2 = (a^2 + c^2 - b^2) / 2) ∧ (z^2 = (a^2 + b^2 - c^2) / 2) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) :=
by
  sorry

end acute_triangle_on_perpendicular_lines_l472_472575


namespace largest_prime_factor_150_choose_75_l472_472413

-- Define the problem conditions and required results in Lean
def factorial (n : Nat) : Nat :=
if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : Nat) : Nat :=
factorial n / (factorial k * factorial (n - k))

def largest_prime_factor_in_range (n : Nat) (lower upper : Nat) : Nat :=
  -- this function would typically compute the largest prime factor of n within the given range
  -- leaving it as sorry for now since we are not asked to implement it
  sorry 

theorem largest_prime_factor_150_choose_75 :
  let n := binomial 150 75 in
  largest_prime_factor_in_range n 10 100 = 47 :=
by
  -- leaving the proof as a sorry as we are only asked to state the theorem
  sorry

end largest_prime_factor_150_choose_75_l472_472413


namespace cost_of_apples_l472_472754

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l472_472754


namespace coefficient_of_x3_in_expansion_l472_472641

theorem coefficient_of_x3_in_expansion :
  -- Given condition
  let expr := (2 * x - 1 / x^(1/2))^6
  -- Conclusion (what we need to prove)
  (coefficient_of_x3_in_binomial_expansion expr) = 240 :=
sorry

-- Auxiliary definition for binomial expansion coefficient, based on the problem condition
noncomputable def coefficient_of_x3_in_binomial_expansion (expr : ℕ → ℕ) : ℕ :=
sorry

end coefficient_of_x3_in_expansion_l472_472641


namespace solutions_shifted_quadratic_l472_472556

theorem solutions_shifted_quadratic (a h k : ℝ) (x1 x2: ℝ)
  (h1 : a * (-1 - h)^2 + k = 0)
  (h2 : a * (3 - h)^2 + k = 0) :
  a * (0 - (h + 1))^2 + k = 0 ∧ a * (4 - (h + 1))^2 + k = 0 :=
by
  sorry

end solutions_shifted_quadratic_l472_472556


namespace count_five_digit_numbers_divisible_by_6_7_8_9_l472_472231

theorem count_five_digit_numbers_divisible_by_6_7_8_9 :
  let lcm := Nat.lcm (Nat.lcm (Nat.lcm 6 7) 8) 9 in
  let lower_bound := Nat.ceil (10000 : ℤ) lcm in
  let upper_bound := Nat.floor (99999 : ℤ) lcm in
  upper_bound - lower_bound + 1 = 179 :=
by
  sorry

end count_five_digit_numbers_divisible_by_6_7_8_9_l472_472231


namespace relationship_among_a_b_c_l472_472952

def a : ℝ := 2 ^ 2.5
def b : ℝ := log 10 2.5
def c : ℝ := 1

theorem relationship_among_a_b_c : a > c ∧ c > b := by
  have h1 : a = 2 ^ 2.5 := rfl
  have h2 : b = log 10 2.5 := rfl
  have h3 : c = 1 := rfl
  -- Using these definitions, we state the exact relationship
  sorry

end relationship_among_a_b_c_l472_472952


namespace square_pyramid_intersection_area_l472_472101

theorem square_pyramid_intersection_area (a b c d e : ℝ) (h_midpoints : a = 2 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4) : 
  ∃ p : ℝ, (p = 80) :=
by
  sorry

end square_pyramid_intersection_area_l472_472101


namespace number_of_false_propositions_is_three_l472_472703

theorem number_of_false_propositions_is_three : 
  ∀ (p1 p2 p3 p4 : Prop), 
    (¬ p1) ∧ (¬ p2) ∧ p3 ∧ (¬ p4) → 
    (p1 = "The converse of vertical angles are equal") ∧ 
    (p2 = "Two lines perpendicular to the same line are parallel") ∧ 
    (p3 = "If in a triangle, the sum of the squares of two sides equals the square of the third side, then the triangle is a right triangle") ∧ 
    (p4 = "Corresponding angles are equal") → 
  (1 + 1 + 0 + 1 = 3) :=
by 
  sorry

end number_of_false_propositions_is_three_l472_472703


namespace hexagon_sides_l472_472881

-- Definitions of the side lengths according to the problem conditions
def sides (PQRSTU : list ℕ) : Prop :=
  PQRSTU.length = 6 ∧ -- Six sides forming the hexagon
  (PQRSTU.count 7) = 2 ∧ -- Two sides measure 7 units
  (PQRSTU.count 8) = 2 ∧ -- Two sides measure 8 units
  (PQRSTU.count 9) = 1 ∧ -- One side measures 9 units
  (PQRSTU.count 6) = 1 ∧ -- One side measures 6 units
  PQRSTU.sum = 45 -- The perimeter is 45 units

-- Main theorem statement
theorem hexagon_sides :
  ∃ (PQRSTU : list ℕ), PQRSTU.length = 6 ∧ sides PQRSTU :=
by
  sorry

end hexagon_sides_l472_472881


namespace min_liars_required_l472_472122

-- Define the presidium configuration and properties
def person : Type := ℕ  -- We can identify each person by a natural number

def rows : fin 4 := -- 4 rows
sorry

def cols : fin 8 := -- 8 columns
sorry

def is_neighbor (p q: person) : Prop := -- Define when two persons are neighbors
sorry 

structure presidium :=
(is_liar : person → Prop) -- Define a property representing if someone is a liar

-- Conditions:
-- Each member claims to have both liars and truth-tellers among their neighbors
-- Liars always lie (their statements about neighbors are false), truth-tellers always tell the truth (their statements are true)
def valid_claim (p : person) (pres : presidium) : Prop :=
  (pres.is_liar p ↔ ¬(∃ neighbor, is_neighbor p neighbor ∧ pres.is_liar neighbor ∧
                      ∃ neighbor, is_neighbor p neighbor ∧ ¬pres.is_liar neighbor ))

-- Main theorem to state the minimum number of liars required
theorem min_liars_required : ∃ pres : presidium, 
  ∀ p : person, valid_claim p pres ∧ 
  (∀ q, ¬pres.is_liar q) = 8 :=
sorry

end min_liars_required_l472_472122


namespace increase_expenditure_by_10_percent_l472_472335

variable (I : ℝ) (P : ℝ)
def E := 0.75 * I
def I_new := 1.20 * I
def S_new := 1.50 * (I - E)
def E_new := E * (1 + P / 100)

theorem increase_expenditure_by_10_percent :
  (E_new = 0.75 * I * (1 + P / 100)) → P = 10 :=
by
  sorry

end increase_expenditure_by_10_percent_l472_472335


namespace ice_cream_total_sum_l472_472848

noncomputable def totalIceCream (friday saturday sunday monday tuesday : ℝ) : ℝ :=
  friday + saturday + sunday + monday + tuesday

theorem ice_cream_total_sum : 
  let friday := 3.25
  let saturday := 2.5
  let sunday := 1.75
  let monday := 0.5
  let tuesday := 2 * monday
  totalIceCream friday saturday sunday monday tuesday = 9 := by
    sorry

end ice_cream_total_sum_l472_472848


namespace bobs_walking_rate_l472_472064

theorem bobs_walking_rate
  (distance_XY : ℕ)
  (yolanda_rate : ℕ)
  (bob_start_delay : ℕ)
  (bob_meet_distance : ℕ) :
  distance_XY = 31 →
  yolanda_rate = 3 →
  bob_start_delay = 1 →
  bob_meet_distance = 16 →
  let yolanda_walked_distance := distance_XY - bob_meet_distance in
  let yolanda_time := yolanda_walked_distance / yolanda_rate in
  let bob_time := yolanda_time - bob_start_delay in
  let bob_rate := bob_meet_distance / bob_time in
  bob_rate = 4 :=
begin
  intros,
  sorry
end

end bobs_walking_rate_l472_472064


namespace x_coord_of_Q_after_rotation_l472_472557

variable {α : ℝ}
def point_P := (4/5 : ℝ, -3/5 : ℝ)
def cos_alpha := 4/5
def sin_alpha := -3/5
def rotation_angle := π / 3

theorem x_coord_of_Q_after_rotation :
  let Qx := cos_alpha * (cos rotation_angle) - sin_alpha * (sin rotation_angle)
  Qx = (4 + 3 * sqrt 3) / 10 :=
by
  sorry

end x_coord_of_Q_after_rotation_l472_472557


namespace operation_not_possible_l472_472889

theorem operation_not_possible 
  (A B C : ℕ) 
  (k : ℕ) 
  (star : ℕ → ℕ → ℕ)
  (h1 : ∀ A B, A ≠ B → star A B = star (|A - B|) (A + B))
  (h2 : ∀ A B C, star (star A C) (star B C) = star (star A B) (star C C)) 
  (h3 : ∀ k, star (2 * k + 1) (2 * k + 1) = 2 * k + 1) : 
  ¬(∃ star, (∀ A B, A ≠ B → star A B = star (|A - B|) (A + B)) ∧ 
             (∀ A B C, star (star A C) (star B C) = star (star A B) (star C C)) ∧ 
             (∀ k, star (2 * k + 1) (2 * k + 1) = 2 * k + 1)) := 
sorry

end operation_not_possible_l472_472889


namespace polynomial_degree_example_l472_472782

theorem polynomial_degree_example :
  ∀ (x: ℝ), degree ((5 * x^3 + 7) ^ 10) = 30 :=
by
  sorry

end polynomial_degree_example_l472_472782


namespace angle_between_vectors_l472_472578

open Real
open InnerProductSpace

variables {α : Type*} [InnerProductSpace ℝ α]

theorem angle_between_vectors 
  (a b : α) 
  (h1 : ∥a∥ = 1) 
  (h2 : ∥b∥ = sqrt 2) 
  (h3 : ⟪a - b, a⟫ = 0) : 
  real.angle a b = π / 4 := 
sorry

end angle_between_vectors_l472_472578


namespace asian_population_west_percentage_l472_472857

theorem asian_population_west_percentage (Asian_NE Asian_MW Asian_South Asian_West : ℕ)
  (h1 : Asian_NE = 2)
  (h2 : Asian_MW = 2)
  (h3 : Asian_South = 2)
  (h4 : Asian_West = 5)
  (total_Asian : ℕ := Asian_NE + Asian_MW + Asian_South + Asian_West) :
  (Asian_West * 100 / total_Asian : ℝ) ≈ (45 : ℝ) :=
by
  have h_total : total_Asian = 11 := by 
     rw [h1, h2, h3, h4]
     norm_num
  have h_west_percentage : (Asian_West * 100 / total_Asian : ℝ) = 
    ↑Asian_West * 100 / ↑total_Asian := by norm_cast
   
  rw [h_total, h_west_percentage, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
  norm_num
  sorry

end asian_population_west_percentage_l472_472857


namespace max_f_area_ABC_l472_472185

-- Problem 1
def vector_m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 2 * Real.sqrt 3 * Real.sin x)
def vector_n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
def f (x : ℝ) : ℝ := vector_m x.1 * vector_n x.1 + vector_m x.2 * vector_n x.2

theorem max_f (x : ℝ) : ∀ x : ℝ, f x = 2 * Real.sin(2*x + Real.pi / 6) + 1 ∧ (∀ y : ℝ, f y ≤ 3) := 
sorry

-- Problem 2
noncomputable def triangle_area (a b c A : ℝ) : ℝ := 
  1 / 2 * b * c * Real.sin A

theorem area_ABC (a b c : ℝ) (A : ℝ) : c = 2 * Real.sqrt 3 ∧ (∀ b' : ℝ, b = b' → 
  b' = 2 ∨ b' = 4) → triangle_area a b c A = Real.sqrt 3 ∨ triangle_area a b c A = 2 * Real.sqrt 3 := sorry

end max_f_area_ABC_l472_472185


namespace max_servings_l472_472039

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l472_472039


namespace min_words_to_score_90_l472_472605

theorem min_words_to_score_90 {x : ℕ} : 
  (x - 0.25 * (800 - x) ≥ 0.90 * 800) → x ≥ 736 := 
by 
  sorry

end min_words_to_score_90_l472_472605


namespace find_x_l472_472249

theorem find_x (h1 : log 0.318 = 0.3364) (h2 : log 0.317 = 0.33320) (x : ℝ) (hx : log x = 0.3396307322929171) : x = 2.186 :=
sorry

end find_x_l472_472249


namespace area_of_R_proof_l472_472715

noncomputable def area_of_R (A B C D : Π (x y : ℝ), x^2 ≤ 2 * x * A ∧ y^2 ≤ 2 * y * D) : ℝ :=
  if Region_R : Π (P : ℝ × ℝ), (dist P A < dist P B) ∧ (dist P A < dist P C) ∧ (dist P A < dist P D) 
  then 1 / 2 else 0

theorem area_of_R_proof (A B C D : ℝ × ℝ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) :
     (∃ (ABCD is a square with side length 2) 
     ∧ (angle A == 90) 
     ∧ (Region_R : Π (P : ℝ × ℝ), 
        (dist P A < dist P B) 
        ∧ (dist P A < dist P C) 
        ∧ (dist P A < dist P D)) 
    → area_of_R A B C D = 1 / 2 := 
sorry

end area_of_R_proof_l472_472715


namespace sum_arith_seq_l472_472295

theorem sum_arith_seq (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h₁ : ∀ n, S n = n * a 1 + (n * (n - 1)) * d / 2)
    (h₂ : S 10 = S 20)
    (h₃ : d > 0) :
    a 10 + a 22 > 0 := 
sorry

end sum_arith_seq_l472_472295


namespace max_dist_vasya_l472_472698

theorem max_dist_vasya (d : ℕ → ℕ → ℕ) (P V : ℕ) (friends : Fin 100) :
  (∑ i in (Finset.filter (λ x, x ≠ P) (Finset.univ : Finset (Fin 100))), d P i) = 1000 →
  (∃ x, x = 99 * 1000) →
  (∑ i in  (Finset.filter (λ x, x ≠ V) (Finset.univ : Finset (Fin 100))), d V i ≤ 99000) :=
by
  sorry

end max_dist_vasya_l472_472698


namespace tangent_sum_eq_l472_472993

theorem tangent_sum_eq (C1 C2 : Circle) (A B C : Point) 
  (h1 : C1.internal_tangent C2) 
  (h2 : equilateral_triangle C1 A B C) 
  (t_a t_b t_c : ℝ) 
  (h3 : tangent_length A C2 t_a)
  (h4 : tangent_length B C2 t_b)
  (h5 : tangent_length C C2 t_c) : 
  t_a = t_b + t_c := sorry

end tangent_sum_eq_l472_472993


namespace sum_smallest_and_third_smallest_l472_472053

theorem sum_smallest_and_third_smallest (d1 d2 d3 : ℕ) (h1 : d1 = 1) (h2 : d2 = 6) (h3 : d3 = 8) :
  let nums := [d1, d2, d3].perms.map (λ l, l.foldl (λ x y, 10 * x + y) 0)
  let sorted_nums := nums.sort
  sorted_nums[0] + sorted_nums[2] = 786 := 
by {
  sorry
}

end sum_smallest_and_third_smallest_l472_472053


namespace assumption_for_contradiction_l472_472704

theorem assumption_for_contradiction (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (h : 5 ∣ a * b) : 
  ¬ (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) := 
sorry

end assumption_for_contradiction_l472_472704


namespace cars_people_count_l472_472073

-- Define the problem conditions
def cars_people_conditions (x y : ℕ) : Prop :=
  y = 3 * (x - 2) ∧ y = 2 * x + 9

-- Define the theorem stating that there exist numbers of cars and people that satisfy the conditions
theorem cars_people_count (x y : ℕ) : cars_people_conditions x y ↔ (y = 3 * (x - 2) ∧ y = 2 * x + 9) := by
  -- skip the proof
  sorry

end cars_people_count_l472_472073


namespace simplify_and_ratio_l472_472349

theorem simplify_and_ratio (k : ℤ) : 
  let a := 1
  let b := 2
  (∀ (k : ℤ), (6 * k + 12) / 6 = a * k + b) →
  (a / b = 1 / 2) :=
by
  intros
  sorry
  
end simplify_and_ratio_l472_472349


namespace problem_min_x_plus_2y_l472_472947

theorem problem_min_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) : 
  x + 2 * y ≥ -2 * Real.sqrt 2 - 1 :=
sorry

end problem_min_x_plus_2y_l472_472947


namespace compute_decimal_expression_l472_472866

theorem compute_decimal_expression :
  (0.3 * 0.7) + (0.5 * 0.4) = 0.41 :=
by
  have h1 : 0.3 = 3 * 10⁻¹ := by norm_num
  have h2 : 0.7 = 7 * 10⁻¹ := by norm_num
  have h3 : 0.5 = 5 * 10⁻¹ := by norm_num
  have h4 : 0.4 = 4 * 10⁻¹ := by norm_num
  sorry

end compute_decimal_expression_l472_472866


namespace petya_lost_remaining_games_l472_472258

theorem petya_lost_remaining_games (participants : ℕ) (games_played : ℕ) 
  (max_scores : ℕ) (petya_score : ℕ) 
  (score_win : ℕ) (score_draw : ℕ) (score_loss : ℕ) 
  (h1 : participants = 12)
  (h2 : games_played = (participants * (participants - 1)) / 2)
  (h3 : ∀ x, x ≤ max_scores)
  (h4 : max_scores = 4)
  (h5 : petya_score = 9)
  (h6 : score_win = 1)
  (h7 : score_draw = 0.5)
  (h8 : score_loss = 0) :
  ∀remaining_games, 
    (let total_games_petya = participants - 1 in
    remaining_games = 2 
    ∧ total_games_petya - 9 = remaining_games 
    ∧ petya_score = 9) → 
    (∀remaining_game_score, remaining_game_score = 0) :=
by
  sorry

end petya_lost_remaining_games_l472_472258


namespace verify_incorrect_option_l472_472670

variable (a : ℕ → ℝ) -- The sequence a_n
variable (S : ℕ → ℝ) -- The sum of the first n terms S_n

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def condition_1 (S : ℕ → ℝ) : Prop := S 5 < S 6

def condition_2 (S : ℕ → ℝ) : Prop := S 6 = S 7 ∧ S 7 > S 8

theorem verify_incorrect_option (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond1 : condition_1 S)
  (h_cond2 : condition_2 S) :
  S 9 ≤ S 5 :=
sorry

end verify_incorrect_option_l472_472670


namespace compute_a_plus_b_l472_472726

-- Define the volume formula for a sphere
def volume_of_sphere (r : ℝ) : ℝ := (4/3) * π * r^3

-- Given conditions
def radius_small_sphere : ℝ := 6
def volume_small_sphere := volume_of_sphere radius_small_sphere
def volume_large_sphere := 3 * volume_small_sphere

-- Radius of the larger sphere
def radius_large_sphere := (volume_large_sphere * 3 / (4 * π))^(1/3)
def diameter_large_sphere := 2 * radius_large_sphere

-- Express diameter in the form a*root(3, b)
def a : ℕ := 12
def b : ℕ := 3

-- The mathematically equivalent proof problem
theorem compute_a_plus_b : (a + b) = 15 := by
  sorry

end compute_a_plus_b_l472_472726


namespace kangaroo_sequence_2017th_letter_l472_472646

theorem kangaroo_sequence_2017th_letter : ∀ sequence, 
  (sequence = "KANGAROO" * (2017 / 8).natAbs + "KANGAROO".take (2017 % 8).natAbs) → 
  (sequence[2016] = 'K') :=
by
  intros sequence h_sequence
  have h_length := 8 -- length of the sequence "KANGAROO"
  have h_mod := 2017 % h_length -- remainder of 2017 divided by 8
  have h_pos := 1 -- position in the sequence (since 2017 mod 8 = 1)
  rw h_sequence -- substitute the sequence definition
  sorry

end kangaroo_sequence_2017th_letter_l472_472646


namespace train_leave_tunnel_l472_472103

noncomputable def train_leave_time 
  (train_speed : ℝ) 
  (tunnel_length : ℝ) 
  (train_length : ℝ) 
  (enter_time : ℝ × ℝ) : ℝ × ℝ :=
  let speed_km_min := train_speed / 60
  let total_distance := train_length + tunnel_length
  let time_to_pass := total_distance / speed_km_min
  let enter_minutes := enter_time.1 * 60 + enter_time.2
  let leave_minutes := enter_minutes + time_to_pass
  let leave_hours := leave_minutes / 60
  let leave_remainder_minutes := leave_minutes % 60
  (leave_hours, leave_remainder_minutes)

theorem train_leave_tunnel : 
  train_leave_time 80 70 1 (5, 12) = (6, 5.25) := 
sorry

end train_leave_tunnel_l472_472103


namespace circumference_of_circle_l472_472614

noncomputable def π : ℝ := Real.pi

def area (r : ℝ) : ℝ := π * r * r

def radius_from_area (A : ℝ) : ℝ := Real.sqrt (A / π)

def circumference (r : ℝ) : ℝ := 2 * π * r

theorem circumference_of_circle (A : ℝ) (h : A = 616) : 
    (circumference (radius_from_area A)) ≈ 88 := 
by
  -- Let Lean handle the values' approximations internally.
  sorry

end circumference_of_circle_l472_472614


namespace smallest_m_l472_472157

theorem smallest_m (a b c : ℝ) (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  ∃ m, (∀ (a b c : ℝ), a + b + c = 1 → 0 < a → 0 < b → 0 < c → m * (a ^ 3 + b ^ 3 + c ^ 3) ≥ 6 * (a ^ 2 + b ^ 2 + c ^ 2) + 1) ↔ m = 27 :=
by
  sorry

end smallest_m_l472_472157


namespace count_integers_last_digit_n_pow_20_eq_1_l472_472900

theorem count_integers_last_digit_n_pow_20_eq_1 :
  (∃ S : finset ℕ, (∀ n ∈ S, (n ≤ 2009 ∧ (n^20 % 10 = 1)) ∧ S.card = 804)) :=
sorry

end count_integers_last_digit_n_pow_20_eq_1_l472_472900


namespace vectors_perpendicular_l472_472995

-- Definitions of vector components
def a (α: ℝ): ℝ × ℝ := (Real.cos α, Real.sin α)
def b (β: ℝ): ℝ × ℝ := (Real.cos β, Real.sin β)

-- Definition of dot product for 2D vectors
def dot_product (u v: ℝ × ℝ): ℝ := u.1 * v.1 + u.2 * v.2

-- Definition of perpendicular vectors
def perpendicular (u v: ℝ × ℝ): Prop := dot_product u v = 0

-- Proof statement of the problem
theorem vectors_perpendicular (α β: ℝ): 
  perpendicular (a α + b β) (a α - b β) :=
  sorry

end vectors_perpendicular_l472_472995


namespace jasmine_cookies_l472_472317

theorem jasmine_cookies (J : ℕ) (h1 : 20 + J + (J + 10) = 60) : J = 15 :=
sorry

end jasmine_cookies_l472_472317


namespace area_of_region_occupied_by_P_l472_472957

-- Definition of the point P satisfying the given equation
def satisfies_circle_eq (P : ℝ × ℝ) (θ : ℝ) : Prop :=
  let (x, y) := P in
  (x - 4 * Real.cos θ) ^ 2 + (y - 4 * Real.sin θ) ^ 2 = 4

-- The main proof statement: Given the condition, prove the area of the region where P lies
theorem area_of_region_occupied_by_P :
  (∃ P : ℝ × ℝ, ∀ θ : ℝ, satisfies_circle_eq P θ) →
  ∃ area : ℝ, area = 32 * Real.pi :=
by
  sorry

end area_of_region_occupied_by_P_l472_472957


namespace problem1_answer_problem2_answer_l472_472483

-- Definition for Problem 1
def problem1 := 27^(2/3) - 2^(Real.logBase 2 3) * Real.logBase 2 (1/8) + Real.logBase 2 3 * Real.logBase 3 4

theorem problem1_answer : problem1 = 20 :=
by
  sorry

-- Definitions and conditions for Problem 2
variables (x : ℝ) (hx1 : 0 < x) (hx2 : x < 1) (hx3 : x + x⁻¹ = 3)

theorem problem2_answer (hx1 : 0 < x) (hx2 : x < 1) (hx3 : x + x⁻¹ = 3) : 
  x^(1/2) - x^(-1/2) = -1 :=
by
  sorry

end problem1_answer_problem2_answer_l472_472483


namespace sum_vectors_correct_l472_472883

def vector2 : Type := (ℝ × ℝ)

-- Definitions of initial vectors
def v0 : vector2 := (2, 2)
def w0 : vector2 := (3, 1)
def u0 : vector2 := (1, -1)

-- Projection function
def proj (a b : vector2) : vector2 :=
  let dot_product : ℝ := a.fst * b.fst + a.snd * b.snd
  let norm_sq : ℝ := b.fst * b.fst + b.snd * b.snd
  let scalar : ℝ := dot_product / norm_sq
  (scalar * b.fst, scalar * b.snd)

-- Initialize sequences
noncomputable def vn : ℕ → vector2
| 0 := v0
| n + 1 := proj (wn n) v0

noncomputable def wn : ℕ → vector2
| 0 := w0
| n + 1 := proj (vn (n + 1)) w0

noncomputable def un : ℕ → vector2
| 0 := u0
| n + 1 := proj (wn (n + 1)) u0

-- Infinite sum of sequences vn, wn, un
noncomputable def sum_vectors : vector2 :=
  let v_sum := (v0.fst + 0.8 * w0.fst - 1.6 * u0.fst, v0.snd + 0.8 * w0.snd - 1.6 * u0.snd)
  v_sum

-- Theorem statement
theorem sum_vectors_correct :
  sum_vectors = (5.0, 5.6) :=
sorry

end sum_vectors_correct_l472_472883


namespace plane_intersects_24_unit_cubes_l472_472448

-- Definitions based on conditions
def is_unit_cube (x y z : ℕ) : Prop := x < 4 ∧ y < 4 ∧ z < 4

-- The proof goal
theorem plane_intersects_24_unit_cubes :
  let cubes : Finset (ℕ × ℕ × ℕ) := 
    { c | ∃ x y z, is_unit_cube x y z ∧ plane_intersects_cube c bisecting_diagonal } 
  in cubes.card = 24 :=
sorry

-- To run this proof, more specific definitions would be needed for:
-- 1. The definition and properties of the plane (plane_intersects_cube).
-- 2. How the plane is positioned relative to the cube (perpendicular to diagonal, bisects diagonal).
-- These definitions are abstracted here for process demonstration.

end plane_intersects_24_unit_cubes_l472_472448


namespace perp_if_and_only_if_collinear_l472_472474

-- Definitions of the entities and conditions in the problem
variables (O O1 O2 : Type) [circle O] [circle O1] [circle O2]
variables (M N S T : point)
variables (r r1 r2 : ℝ) -- radii of the circles O, O1, O2 respectively
variables (intersects_at : ∀ {C1 C2 : Type} [circle C1] [circle C2] (p1 p2 : point), Prop)
variables (tangent_at : ∀ {C1 C2 : Type} [circle C1] [circle C2] (p : point), Prop)

-- The given conditions
axiom intersect_cond : intersects_at O1 O2 M ∧ intersects_at O1 O2 N
axiom tangent_cond1 : tangent_at O O1 S
axiom tangent_cond2 : tangent_at O O2 T

-- The main statement to be proved
theorem perp_if_and_only_if_collinear : 
  (⊥ (line_through O M) (line_through M N)) ↔ collinear {S, N, T} :=
sorry

end perp_if_and_only_if_collinear_l472_472474


namespace no_integer_in_interval_l472_472347

theorem no_integer_in_interval (n : ℕ) : ¬ ∃ k : ℤ, 
  (n ≠ 0 ∧ (n * Real.sqrt 2 - 1 / (3 * n) < k) ∧ (k < n * Real.sqrt 2 + 1 / (3 * n))) := 
sorry

end no_integer_in_interval_l472_472347


namespace ball_distribution_probability_l472_472761

theorem ball_distribution_probability :
  (exists (p q : ℕ), (Nat.gcd p q = 1) ∧ (p : ℚ / q) = 6 / 49) ∧ (p + q = 55) := 
  sorry

end ball_distribution_probability_l472_472761


namespace find_diamond_value_l472_472536

variable {x y : ℝ}

def diamond (x y : ℝ) : ℝ := x y -- Assuming some fixed rule for definition
axiom diamond_rule1 : ∀ {x y : ℝ}, x > 0 → y > 0 → (x * y) ∪ diamond y = x ∪ (diamond y y)
axiom diamond_rule2 : ∀ {x : ℝ}, x > 0 → (diamond x 1) ∪ diamond x = diamond x x
axiom diamond_value : diamond 1 1 = 2

theorem find_diamond_value : diamond 19 98 = 3724 := 
by 
    sorry

end find_diamond_value_l472_472536


namespace find_b_l472_472668

def a := ![2, 1, 5]
def b := ![-3.5, 3.75, 1.75]

def dot_product (u v : Fin 3 → ℝ) : ℝ :=
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def cross_product (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0
  ]

theorem find_b : dot_product a b = 15 ∧ cross_product a b = ![-17, -11, 11] :=
by
  sorry

end find_b_l472_472668


namespace gas_pressure_inversely_proportional_l472_472477

theorem gas_pressure_inversely_proportional
  (p v k : ℝ)
  (v_i v_f : ℝ)
  (p_i p_f : ℝ)
  (h1 : v_i = 3.5)
  (h2 : p_i = 8)
  (h3 : v_f = 7)
  (h4 : p * v = k)
  (h5 : p_i * v_i = k)
  (h6 : p_f * v_f = k) : p_f = 4 := by
  sorry

end gas_pressure_inversely_proportional_l472_472477


namespace log_sum_property_l472_472560

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x / Real.log a
noncomputable def f_inv (a : ℝ) (y : ℝ) := a ^ y

theorem log_sum_property (a : ℝ) (h1 : f_inv a 2 = 9) (h2 : f a 9 = 2) : f a 9 + f a 6 = 1 :=
by
  sorry

end log_sum_property_l472_472560


namespace range_of_a_l472_472224

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (x + 1 - a) >= 0 → x ≠ 1) ↔ (1 < a ∧ a < 2) := 
sorry

end range_of_a_l472_472224


namespace freshmen_assignment_l472_472609

theorem freshmen_assignment (freshmen : ℕ) (classes : ℕ) : 
  freshmen = 5 ∧ classes = 3 → (Σ (dist : Multiset (Fin 3) → Fin 5 → ℕ), 
      (∀ x, dist x > 0) ∧ (∑ i, dist i = freshmen) ∧
      (multiset.count 0 dist = 1 ∧ multiset.count 1 dist = 1 ∧ multiset.count 2 dist = 3 ∨
       multiset.count 0 dist = 1 ∧ multiset.count 1 dist = 2 ∧ multiset.count 2 dist = 2)) = 150 :=
by
  sorry

end freshmen_assignment_l472_472609


namespace verify_incorrect_option_l472_472671

variable (a : ℕ → ℝ) -- The sequence a_n
variable (S : ℕ → ℝ) -- The sum of the first n terms S_n

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

def condition_1 (S : ℕ → ℝ) : Prop := S 5 < S 6

def condition_2 (S : ℕ → ℝ) : Prop := S 6 = S 7 ∧ S 7 > S 8

theorem verify_incorrect_option (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond1 : condition_1 S)
  (h_cond2 : condition_2 S) :
  S 9 ≤ S 5 :=
sorry

end verify_incorrect_option_l472_472671


namespace sum_of_squares_l472_472645

variable (a : ℕ → ℕ)

def sequence_cond (n : ℕ) : Prop :=
  (finset.range (n + 1)).sum a = 2 ^ (n + 1)

theorem sum_of_squares (h : ∀ n, sequence_cond a n) :
  ∀ n, (finset.range (n + 1)).sum (λ k, (a k)^2) = (1 / 3) * (4 ^ (n + 1) + 8) :=
sorry

end sum_of_squares_l472_472645


namespace cube_volume_problem_mnp_sum_l472_472149
noncomputable def set_volume : ℕ :=
let side_length : ℕ := 6 in
let m := 432 in
let n := 19 in
let p := 1 in
if (n.gcd p = 1) then m + n + p else sorry

theorem cube_volume_problem_mnp_sum :
  set_volume = 452 :=
by
  -- We are skipping the proof, hence using 'sorry.'
  sorry

end cube_volume_problem_mnp_sum_l472_472149


namespace bruce_remaining_eggs_l472_472124

theorem bruce_remaining_eggs :
  (bruce_initial_eggs : ℕ) →
  (bruce_lost_eggs : ℕ) →
  bruce_initial_eggs = 215 →
  bruce_lost_eggs = 137 →
  bruce_initial_eggs - bruce_lost_eggs = 78 := 
by
  intros bruce_initial_eggs bruce_lost_eggs
  intros h_initial h_lost
  rw [h_initial, h_lost]
  exact Nat.sub_self sorry

end bruce_remaining_eggs_l472_472124


namespace max_servings_l472_472012

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l472_472012


namespace max_servings_l472_472024

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l472_472024


namespace not_monotonic_interval_l472_472973

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x ^ 2 + 6 * x - 8 * Real.log x

theorem not_monotonic_interval (m : ℝ) : 
  ¬ (∀ x y ∈ set.Icc m (m + 1), x ≤ y → f x ≤ f y ∨ f x ≥ f y) ↔ m ∈ set.Ioo 1 2 ∪ set.Ioo 3 4 := 
sorry

end not_monotonic_interval_l472_472973


namespace sum_of_values_l472_472512

theorem sum_of_values (x_set : Finset ℝ) (h1 : ∀ x ∈ x_set, 100 < x ∧ x < 200)
  (h2 : ∀ x ∈ x_set, real.sin ((3 * x) * (π / 180)) ^ 3 + real.sin ((5 * x) * (π / 180)) ^ 3 = 
                     8 * (real.sin ((4 * x) * (π / 180))) ^ 3 * (real.sin (x * (π / 180))) ^ 3)
  : x_set.sum id = 687 := 
sorry

end sum_of_values_l472_472512


namespace sum_of_solutions_l472_472798

theorem sum_of_solutions : 
  let solutions := {x | 0 < x ∧ x ≤ 30 ∧ 15 * (5 * x - 3) % 12 = 45 % 12} in
  (solutions.sum id) = 54 := by
  sorry

end sum_of_solutions_l472_472798


namespace fair_tickets_sold_l472_472373

theorem fair_tickets_sold (F : ℕ) (number_of_baseball_game_tickets : ℕ) 
  (h1 : F = 2 * number_of_baseball_game_tickets + 6) (h2 : number_of_baseball_game_tickets = 56) :
  F = 118 :=
by
  sorry

end fair_tickets_sold_l472_472373


namespace queenie_total_earnings_l472_472340

-- Define the conditions
def daily_wage : ℕ := 150
def overtime_wage_per_hour : ℕ := 5
def days_worked : ℕ := 5
def overtime_hours : ℕ := 4

-- Define the main problem
theorem queenie_total_earnings : 
  (daily_wage * days_worked + overtime_wage_per_hour * overtime_hours) = 770 :=
by
  sorry

end queenie_total_earnings_l472_472340


namespace sin_cos_mixed_l472_472669

theorem sin_cos_mixed (θ : ℝ) (h : sin (2 * θ) = 1 / 4) : (sin θ)^6 + (cos θ)^6 = 61 / 64 :=
by
  have h1 : sin (2 * θ) = 2 * sin θ * cos θ := by congr sorry
  rw [h] at h1
  have h2 : 2 * sin θ * cos θ = 1 / 4 := h1
  have h3 : sin θ * cos θ = 1 / 8 := by linarith
  have h4 : (cos θ)^2 + (sin θ)^2 = 1 := by congr sorry
  ring_exp at h3 h4
  sorry

end sin_cos_mixed_l472_472669


namespace tangent_equation_at_origin_l472_472368

noncomputable def f (x : ℝ) := Real.exp (1 - x)

theorem tangent_equation_at_origin :
  ∃ k, (∀ x y : ℝ, y = f x → y = -Real.exp 2 * x) ∧ (0 = k * 0) :=
begin
  sorry
end

end tangent_equation_at_origin_l472_472368


namespace triangle_acute_and_inequality_l472_472674

variables (A B C : Type) 
variables [is_triangle ABC]
variables (r r_a r_b r_c a b c: ℝ) 

-- Condition that describes the ex-radii and sides of the triangle
variables (ha : a > r_a) (hb : b > r_b) (hc : c > r_c)

theorem triangle_acute_and_inequality (h : is_triangle ABC) (hr : inradius ABC = r) (hra : exradius ABC A = r_a)
  (hrb : exradius ABC B = r_b) (hrc : exradius ABC C = r_c) (ha : a > r_a) (hb : b > r_b) (hc : c > r_c) :
(∀ (A B C : ℝ), is_acute ABC) ∧ (a + b + c > r + r_a + r_b + r_c) :=
sorry

end triangle_acute_and_inequality_l472_472674


namespace sum_D_E_F_l472_472729

theorem sum_D_E_F (D E F : ℤ) (h : ∀ x, x^3 + D * x^2 + E * x + F = (x + 3) * x * (x - 4)) : 
  D + E + F = -13 :=
by
  sorry

end sum_D_E_F_l472_472729


namespace f_decreasing_in_interval_l472_472344

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

noncomputable def shifted_g (x : ℝ) : ℝ := g (x + Real.pi / 6)

noncomputable def f (x : ℝ) : ℝ := shifted_g (2 * x)

theorem f_decreasing_in_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 4 → f y < f x :=
by
  sorry

end f_decreasing_in_interval_l472_472344


namespace tiles_not_interchangeable_l472_472097

def tiling (m n : ℕ) : Type :=
Π (i j : fin m) (j : fin n), (bool × bool)

def broken_tile_replacement_possible (m n : ℕ) (til : tiling m n) : Prop :=
  ∃ (b : bool), (∀ i j, tiling.1 i j ≠ tiling.2 i j) →
  ¬ (∃ (x y : ∀ (i j : fin m) (j : fin n) tiling), (tiling m n) with broken tile (x, y))

theorem tiles_not_interchangeable {m n : ℕ} (til : tiling m n) (b: bool) :
  ¬ broken_tile_replacement_possible m n til := sorry

end tiles_not_interchangeable_l472_472097


namespace no_base_6_digit_divisible_by_7_l472_472522

theorem no_base_6_digit_divisible_by_7 :
  ∀ (d : ℕ), d < 6 → ¬ (7 ∣ (652 + 42 * d)) :=
by
  intros d hd
  sorry

end no_base_6_digit_divisible_by_7_l472_472522


namespace number_of_true_propositions_is_2_l472_472527

variables (m n : Line) (α β : Plane)

-- Condition: m and n are non-coincident lines
axiom non_coincident_lines : m ≠ n

-- Condition: α and β are non-coincident planes
axiom non_coincident_planes : α ≠ β

-- Proposition 1
def proposition1 (m n : Line) (α β : Plane) : Prop :=
  (m ⊂ α ∧ n ⊂ α ∧ m ∥ β ∧ n ∥ β) → α ∥ β

-- Proposition 2
def proposition2 (m n : Line) (α β : Plane) : Prop :=
  (m ⊥ α ∧ n ⊥ β ∧ m ∥ n) → α ∥ β

-- Proposition 3
def proposition3 (m n : Line) (α β : Plane) : Prop :=
  (α ⊥ β ∧ m ⊂ α ∧ n ⊂ β) → m ⊥ n

-- Proposition 4
def proposition4 (m n : Line) (α β : Plane) : Prop :=
  (skew_lines m n ∧ m ⊂ α ∧ m ∥ β ∧ n ⊂ β ∧ n ∥ α) → α ∥ β

-- True propositions count
def true_propositions_count (m n : Line) (α β : Plane) : Nat :=
  (if proposition2 m n α β then 1 else 0) + (if proposition4 m n α β then 1 else 0)

theorem number_of_true_propositions_is_2 :
  true_propositions_count m n α β = 2 := sorry

end number_of_true_propositions_is_2_l472_472527


namespace range_of_a_l472_472625

open Real

theorem range_of_a (k a : ℝ) : 
  (∀ k : ℝ, ∀ x y : ℝ, k * x - y - k + 2 = 0 → x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-7 : ℝ) (-2) ∪ Set.Ioi 1) := 
sorry

end range_of_a_l472_472625


namespace inequality_proof_l472_472923

noncomputable theory
open Real

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 2) :
  1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) ≥ 27 / 13 :=
sorry

end inequality_proof_l472_472923


namespace cone_volume_with_inscribed_sphere_l472_472457

theorem cone_volume_with_inscribed_sphere (r α : ℝ) : 
  let V := (π * r^3 * (Real.cot ((π / 4 : ℝ) - (α / 2)))^3) / 
            (3 * (Real.cos α)^2 * Real.sin α)
  in V = (π * r^3 * (Real.cot ((π / 4 : ℝ) - (α / 2)))^3) / 
         (3 * (Real.cos α)^2 * Real.sin α) :=
by
  sorry

end cone_volume_with_inscribed_sphere_l472_472457


namespace smallest_positive_period_of_tan_func_l472_472911

theorem smallest_positive_period_of_tan_func
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = tan (x / 3 + π / 4)):
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = 3 * π :=
sorry

end smallest_positive_period_of_tan_func_l472_472911


namespace find_angle_ACB_l472_472720

variable (A B C D E F : Type) [HasCos A] [HasCos B] [HasCos C] [HasCos D]

def is_midpoint (p1 p2 : Type) [HasNorm p1] [HasNorm p2] (m : Type) :=
  dist m p1 = dist m p2

variables (AB CD EF : Type) [HasNorm AB] [HasNorm CD] [HasNorm EF]

variable (cos_phi_AB_CD : ℝ)
variable (AB_length : ℝ)
variable (CD_length : ℝ)
variable (EF_length : ℝ)

noncomputable def angle_ACB : ℝ :=
  real.arccos (5 / 8)

theorem find_angle_ACB
  (h1 : cos_phi_AB_CD = (√35)/10)
  (h2 : 2 * √5 = AB_length)
  (h3 : 2 * √7 = CD_length)
  (h4 : EF_length = √13)
  (h5 : is_midpoint A B E)
  (h6 : is_midpoint C D F)
  (h7 : ∀ (u : Type) [HasNorm u], dist E F = EF_length → dist u E = dist u F → dist A B = AB_length → dist C D = CD_length → orthogonal E F u) :
  angle_ACB = real.arccos (5 / 8) := by sorry

end find_angle_ACB_l472_472720


namespace angle_bac_iff_cyclic_xbyc_l472_472297

-- Define the necessary geometrical entities
variables {A B C M X Y : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace X] [MetricSpace Y]

-- Define the triangle properties
def is_triangle (A B C : Type*) : Prop :=
(¬(A = B) ∧ ¬(B = C) ∧ ¬(C = A))

-- Define the conditions for the problem
variables [Midpoint M B C] [OnLine X A B] [OnLine Y A C] [Perpendicular L A M]

-- Define the cyclic quadrilateral property
def cyclic_quad (X Y B C : Type*) : Prop :=
∃ (omega : Type*), OnCircle omega X Y B C

-- Define the angle at A
def right_angle_at_A (A B C : Type*) : Prop :=
angle B A C = 90

-- Main theorem statement
theorem angle_bac_iff_cyclic_xbyc (h_triangle_ABC : is_triangle A B C)
    (h_midpoint_M : Midpoint M B C)
    (h_line_l_perp_AM : Perpendicular L A M)
    (h_line_l_intersects_AB_at_X : OnLine X A B)
    (h_line_l_intersects_AC_at_Y : OnLine Y A C) :
    right_angle_at_A A B C ↔ cyclic_quad X Y B C :=
sorry

end angle_bac_iff_cyclic_xbyc_l472_472297


namespace vector_perpendicular_iff_norm_equality_l472_472664

variables {V : Type*} [inner_product_space ℝ V]

theorem vector_perpendicular_iff_norm_equality (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + 2 • b).norm = (a - 2 • b).norm ↔ ⟪a, b⟫ = 0 :=
by 
-- Proof goes here
sorry

end vector_perpendicular_iff_norm_equality_l472_472664


namespace range_abs_diff_geq_four_l472_472293

open Set Real

theorem range_abs_diff_geq_four (a b : ℝ)
    (hA : ∀ x, x ∈ Ioo (a - 1) (a + 1) → x ∈ ℝ)
    (hB : ∀ x, x ∈ (Iio (b - 3) ∪ Ioi (b + 3)) → x ∈ ℝ)
    (h : Ioo (a - 1) (a + 1) ⊆ Iio (b - 3) ∪ Ioi (b + 3)) :
    |a - b| ≥ 4 := by
  sorry

end range_abs_diff_geq_four_l472_472293


namespace min_common_perimeter_isosceles_triangles_l472_472771

theorem min_common_perimeter_isosceles_triangles
  (x y k : ℤ)
  (h1 : 2 * x + 5 * k = 2 * y + 4 * k)
  (h2 : 5 * (real.sqrt (x^2 - (5 * k / 2)^2)) = 4 * (real.sqrt (y^2 - (2 * k)^2)))
  (h3 : y = x + 1)
  (h4 : k = 2)
  : 2 * x + 5 * k = 34 :=
by
  sorry

end min_common_perimeter_isosceles_triangles_l472_472771


namespace solve_for_x_l472_472914

theorem solve_for_x:
  ∀ (x : ℝ), (sqrt (2 - 5 * x + x^2) = 9) ↔ (x = (5 + sqrt 341) / 2 ∨ x = (5 - sqrt 341) / 2) :=
by
  sorry

end solve_for_x_l472_472914


namespace limit_x_cot_l472_472907

theorem limit_x_cot (f : ℝ → ℝ) (h_f : ∀ x, f x = x * (Real.cos (x / 3) / Real.sin (x / 3))) :
  filter.tendsto f (nhds 0) (nhds 3) :=
sorry

end limit_x_cot_l472_472907


namespace gumballs_problem_l472_472871

theorem gumballs_problem 
  (L x : ℕ)
  (h1 : 19 ≤ (17 + L + x) / 3 ∧ (17 + L + x) / 3 ≤ 25)
  (h2 : ∃ x_min x_max, x_max - x_min = 18 ∧ x_min = 19 ∧ x = x_min ∨ x = x_max) : 
  L = 21 :=
sorry

end gumballs_problem_l472_472871


namespace bob_max_candies_l472_472398

theorem bob_max_candies (b : ℕ) (h : b + 2 * b = 30) : b = 10 := 
sorry

end bob_max_candies_l472_472398


namespace mark_owes_joanna_l472_472284

def dollars_per_room : ℚ := 12 / 3
def rooms_cleaned : ℚ := 9 / 4
def total_amount_owed : ℚ := 9

theorem mark_owes_joanna :
  dollars_per_room * rooms_cleaned = total_amount_owed :=
by
  sorry

end mark_owes_joanna_l472_472284


namespace sin_double_angle_l472_472183

theorem sin_double_angle (x : ℝ) (h : Real.sin (x - π / 4) = 3 / 5) : Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_l472_472183


namespace inequality_solution_set_l472_472744

variable {x : ℝ}

theorem inequality_solution_set (h : 1 / x > 1) : 0 < x ∧ x < 1 := 
begin
  sorry
end

end inequality_solution_set_l472_472744


namespace probability_of_negative_m_l472_472252

theorem probability_of_negative_m (m : ℤ) (h₁ : -2 ≤ m) (h₂ : m < (9 : ℤ) / 4) :
  ∃ (neg_count total_count : ℤ), 
    (neg_count = 2) ∧ (total_count = 5) ∧ (m ∈ {i : ℤ | -2 ≤ i ∧ i < 2 ∧ i < 9 / 4}) → 
    (neg_count / total_count = 2 / 5) :=
sorry

end probability_of_negative_m_l472_472252


namespace distances_inequality_l472_472176

theorem distances_inequality (x y : ℝ) :
  Real.sqrt ((x + 4)^2 + (y + 2)^2) + 
  Real.sqrt ((x - 5)^2 + (y + 4)^2) ≤ 
  Real.sqrt ((x - 2)^2 + (y - 6)^2) + 
  Real.sqrt ((x - 5)^2 + (y - 6)^2) + 20 :=
  sorry

end distances_inequality_l472_472176


namespace parametric_to_standard_max_value_after_scaling_l472_472152

theorem parametric_to_standard (θ : ℝ) : 
    let x := cos θ 
    let y := sin θ in 
    x^2 + y^2 = 1 := 
by
  sorry

theorem max_value_after_scaling (θ : ℝ) :
    let x := cos θ 
    let y := sin θ 
    let x' := 3 * cos θ 
    let y' := 2 * sin θ in 
    abs (x' * y') ≤ 3 := 
by
  sorry

end parametric_to_standard_max_value_after_scaling_l472_472152


namespace range_of_a_l472_472558

noncomputable def func_derivative (a x : ℝ) : ℝ := a * (x + 1) * (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, func_derivative a x = a * (x + 1) * (x - a)) →
  (∀ f : ℝ → ℝ, (∀ x : ℝ, f' = func_derivative a x) → (∃ x = a, is_local_max f x)) →
  -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l472_472558


namespace find_a_at_1_find_a_range_l472_472568

noncomputable def f (x a : ℝ) := log x - a * sin (x - 1)

-- Question I: Prove that if the slope of the tangent line to y = f(x) at x = 1 is -1, then a = 2.
theorem find_a_at_1 (a : ℝ) (h : deriv (λ x, log x - a * sin (x - 1)) 1 = -1) : a = 2 :=
sorry

-- Question II: Prove that if f(x) is increasing on (0,1), then a ≤ 1.
theorem find_a_range (a : ℝ) (h_inc : ∀ x, 0 < x ∧ x < 1 → deriv (λ x, log x - a * sin (x - 1)) x ≥ 0) : a ≤ 1 :=
sorry

end find_a_at_1_find_a_range_l472_472568


namespace distance_CM_l472_472289

def point := (ℝ × ℝ × ℝ)

def A : point := (3, 3, 1)
def B : point := (1, 0, 5)
def C : point := (0, 1, 0)

def midpoint (P Q : point) : point := ( (P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2 )

def distance (P Q : point) : ℝ := real.sqrt ( (P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2 )

def M : point := midpoint A B

theorem distance_CM : distance M C = (real.sqrt 53) / 2 :=
by
  sorry

end distance_CM_l472_472289


namespace train_crossing_time_l472_472460

theorem train_crossing_time (length_of_train : ℕ) (speed_kmph : ℕ) (time_in_seconds : ℕ) :
  length_of_train = 2500 ∧ speed_kmph = 90 ∧ time_in_seconds = 100 → length_of_train / (speed_kmph * 1000 / 3600) = time_in_seconds :=
by
  intros
  rw [← (eq.subst rfl (eq.symm (eq.symm (eq.symm rfl))))]
  sorry

end train_crossing_time_l472_472460


namespace minimum_z_value_l472_472540

theorem minimum_z_value (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : x^2 + y^2 ≥ 14 - 2 * Real.sqrt 13 :=
sorry

end minimum_z_value_l472_472540


namespace trigonometric_values_l472_472949

theorem trigonometric_values (α β : ℝ) (h1 : real.cos α = 1 / 7)
  (h2 : real.cos (α - β) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < real.pi / 2) :
  real.tan (2 * α) = -8 * real.sqrt 3 / 47 ∧ β = real.pi / 3 :=
by { sorry }

end trigonometric_values_l472_472949


namespace solution_I_solution_II_l472_472222

noncomputable def f (a x : ℝ) : ℝ := |2 * x - a| + |2 * x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 3

theorem solution_I (x : ℝ) : IsSubset (SetOf (λ x, g x < 5)) (Set.Ioo (-1 : ℝ) 3) :=
sorry

theorem solution_II (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f a x1 = g x2) ↔
  (a ≤ -6 ∨ 0 ≤ a) :=
begin
  split,
  {
    intro h,
    have h_range := λ x, h x,
    replace h_range : SetOf (λ y, ∃ x, f a x = y) ⊆ SetOf (λ y, ∃ x, g x = y),
    {
      exact h_range,
    },
    sorry
  },
  {
    intro h,
    cases h,
    { sorry },
    { sorry }
  }
end

end solution_I_solution_II_l472_472222


namespace necklace_count_five_white_two_black_l472_472233

theorem necklace_count_five_white_two_black : 
  ∃ n, n = 3 ∧ ∀ (w b : ℕ) (arrangement : list char) (h₁ : w = 5) (h₂ : b = 2)
  (h₃ : arrangement.count 'W' = w) (h₄ : arrangement.count 'B' = b) (cyclic : ∀ (r : list char), list.perm r arrangement ↔ list.perm r (list.rotate arrangement 1)) 
  (flip : ∀ (r : list char), list.perm r arrangement ↔ list.perm r (list.reverse arrangement)), 
  ∃ (unique_arrangements : list (list char)), unique_arrangements.length = n :=
begin
  sorry
end

end necklace_count_five_white_two_black_l472_472233


namespace convex_subsets_count_98_l472_472456

def is_convex (pts : list (ℤ × ℤ)) : Prop :=
  ∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ pts → p2 ∈ pts → p3 ∈ pts →
    collinear p1 p2 p3 → 
    (p2 = p1 ∨ p2 = p3)

def S : set (ℤ × ℤ) := { p | ∃ x y, 1 ≤ x ∧ x ≤ 26 ∧ 1 ≤ y ∧ y ≤ 26 ∧ p = (x, y) }

noncomputable def num_convex_subsets_of_size (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 26 then
    -- This is a placeholder, the actual logic to count convex subsets should be implemented
    sorry
  else 0

theorem convex_subsets_count_98 :
  num_convex_subsets_of_size 98 = 4958 := 
sorry

end convex_subsets_count_98_l472_472456


namespace sum_x_coords_at_f_eq_1_l472_472877

noncomputable def f : ℝ → ℝ 
| x => if -4 ≤ x ∧ x ≤ -2 then -2 * x - 3
       else if -2 < x ∧ x ≤ -1 then -x - 3
       else if -1 < x ∧ x ≤ 1 then 2 * x
       else if 1 < x ∧ x ≤ 2 then -x + 3
       else if 2 < x ∧ x ≤ 4 then x + 3
       else 0

theorem sum_x_coords_at_f_eq_1.5 : (∑ x in {x : ℝ | f x = 1.5}.toFinset, x) = 2.25 := by
  sorry

end sum_x_coords_at_f_eq_1_l472_472877


namespace exists_root_in_interval_l472_472112

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

-- Conditions given in the problem
variables {a b c : ℝ}
variable  (h_a_nonzero : a ≠ 0)
variable  (h_neg_value : quadratic a b c 3.24 = -0.02)
variable  (h_pos_value : quadratic a b c 3.25 = 0.01)

-- Problem statement to be proved
theorem exists_root_in_interval : ∃ x : ℝ, 3.24 < x ∧ x < 3.25 ∧ quadratic a b c x = 0 :=
sorry

end exists_root_in_interval_l472_472112


namespace max_servings_l472_472025

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l472_472025


namespace alice_paper_cranes_l472_472113

theorem alice_paper_cranes : 
  ∀ (total : ℕ) (half : ℕ) (one_fifth : ℕ) (thirty_percent : ℕ),
    total = 1000 →
    half = total / 2 →
    one_fifth = (total - half) / 5 →
    thirty_percent = ((total - half) - one_fifth) * 3 / 10 →
    total - (half + one_fifth + thirty_percent) = 280 :=
by
  intros total half one_fifth thirty_percent h_total h_half h_one_fifth h_thirty_percent
  sorry

end alice_paper_cranes_l472_472113


namespace circle_tangent_radius_l472_472186

-- Definitions of points and distances
variables {A F B C O : Point}
variables {r x : Real}
variables {AF FB : Real}

-- Given conditions
def given_conditions : Prop := (AF = FB) ∧ (AF + FB = 2 * AB) ∧
  (distance A O = 4 * r - x) ∧ (distance B O = 4 * r - x) ∧
  (distance F O = (r + x) ^ 2 - r ^ 2) ∧
  (distance F O = (4 * r - x) ^ 2 - (2 * r) ^ 2)

-- The theorem statement
theorem circle_tangent_radius : given_conditions → x = 6 / 5 * r :=
by
  -- Assume given conditions
  assume h : given_conditions,
  -- Proof steps (skipped)
  sorry

end circle_tangent_radius_l472_472186


namespace problem1_problem2_l472_472924

def count_good_subsets (n : ℕ) : ℕ := 
if n % 2 = 1 then 2^(n - 1) 
else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2)

def sum_f_good_subsets (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2)

theorem problem1 (n : ℕ)  :
  (count_good_subsets n = (if n % 2 = 1 then 2^(n - 1) else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2))) :=
sorry

theorem problem2 (n : ℕ) :
  (sum_f_good_subsets n = (if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
  else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2))) := 
sorry

end problem1_problem2_l472_472924


namespace find_k_l472_472903

theorem find_k (k : ℝ) :
  ∃ k, ∀ x : ℝ, (3 * x^3 + k * x^2 - 8 * x + 52) % (3 * x + 4) = 7 :=
by
-- The proof would go here, we insert sorry to acknowledge the missing proof
sorry

end find_k_l472_472903


namespace max_value_of_a_l472_472569

noncomputable def max_a_value (a b : ℝ) (e : ℝ) : ℝ :=
  if h : (e ∈ (Set.Icc (1/2) (Real.sqrt 3 / 2))) then
    let m := 1/2 * (1 + 1 / (1 - e^2))
    if (m > 0) then 
      let a_sq_max := Real.sqrt 10 / 2
      a_sq_max
    else 0
  else 0

theorem max_value_of_a :
  ∀ a b e : ℝ,
  (a > b) → (b > 0) →
  (e ∈ (Set.Icc (1/2) (Real.sqrt 3 / 2))) →
  (∀ x, (y = -x + 1) → (x^2 / a^2 + y^2 / b^2 = 1) → (OA ⊥ OB)) →
  (max_a_value a b e = Real.sqrt 10 / 2) := 
sorry

end max_value_of_a_l472_472569


namespace midpoint_coordinates_l472_472198

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (A B : Point3D) : Point3D :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2, z := (A.z + B.z) / 2 }

theorem midpoint_coordinates :
  let A := Point3D.mk 3 4 1
  let B := Point3D.mk 1 0 5
  midpoint A B = Point3D.mk 2 2 3 :=
by
  let A := Point3D.mk 3 4 1
  let B := Point3D.mk 1 0 5
  let M := midpoint A B
  have hx : M.x = 2 := by
    simp [midpoint, Point3D.mk, A, B]
  have hy : M.y = 2 := by
    simp [midpoint, Point3D.mk, A, B]
  have hz : M.z = 3 := by
    simp [midpoint, Point3D.mk, A, B]
  exact Point.ext (eq.trans (Eq.refl _) hx) (eq.trans (Eq.refl _) hy) (eq.trans (Eq.refl _) hz)

end midpoint_coordinates_l472_472198


namespace unique_zero_of_function_l472_472552

theorem unique_zero_of_function (a : ℝ) :
  (∃! x : ℝ, e^(abs x) + 2 * a - 1 = 0) ↔ a = 0 := 
by 
  sorry

end unique_zero_of_function_l472_472552


namespace appropriate_chart_for_body_temperature_changes_over_week_l472_472041

-- Define characteristics of chart types
def bar_chart_characteristics : Prop := 
  ∀ (data: Type), (bar_chart data).shows_amount_only (data)

def line_chart_characteristics : Prop := 
  ∀ (data: Type), (line_chart data).shows_amount_and_changes (data)

-- Define the problem statement to prove that the appropriate chart for the given context is a line chart
theorem appropriate_chart_for_body_temperature_changes_over_week : 
  (∀ (data: Type), (bar_chart data).shows_amount_only (data)) →
  (∀ (data: Type), (line_chart data).shows_amount_and_changes (data)) → 
  (∃ chart_type: Type, chart_type = line_chart) :=
sorry

end appropriate_chart_for_body_temperature_changes_over_week_l472_472041


namespace stable_k_digit_number_l472_472264

def is_stable (a k : ℕ) : Prop :=
  ∀ m n : ℕ, (10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a))

theorem stable_k_digit_number (k : ℕ) (h_pos : k > 0) : ∃ (a : ℕ) (h : ∀ m n : ℕ, 10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a)), (10^(k-1)) ≤ a ∧ a < 10^k ∧ ∀ b : ℕ, (∀ m n : ℕ, 10^k ∣ ((m * 10^k + b) * (n * 10^k + b) - b)) → (10^(k-1)) ≤ b ∧ b < 10^k → a = b :=
by
  sorry

end stable_k_digit_number_l472_472264


namespace max_elements_in_union_l472_472377

-- Definitions based on conditions given in the problem
def is_pos_int (n : ℕ) : Prop := n > 0

def in_A (A : set ℕ) (x : ℕ) : Prop := x ∈ A ∧ is_pos_int x

def in_B (B : set ℕ) (y : ℕ) : Prop := y ∈ B ∧ is_pos_int y

def sum_in_B (A B : set ℕ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x ≠ y → x + y ∈ B

def quotient_in_A (A B : set ℕ) : Prop :=
  ∀ x y, x ∈ B → y ∈ B → x > y → x / y ∈ A

-- Main theorem statement
theorem max_elements_in_union (A B : set ℕ) (hA : ∀ x, x ∈ A → is_pos_int x) (hB : ∀ y, y ∈ B → is_pos_int y)
    (h_sum : sum_in_B A B) (h_quotient : quotient_in_A A B) :
    (|A ∪ B| <= 5) :=
sorry

end max_elements_in_union_l472_472377


namespace condition_for_extreme_values_l472_472974

theorem condition_for_extreme_values (a : ℝ) (h : a > 0) : (\frac{1}{3} < a ∧ a < 1) ↔ ∃ x ∈ Ioo (a, a + 2/3), deriv (λ x, (1 + log x) / x) x = 0 :=
sorry

end condition_for_extreme_values_l472_472974


namespace savings_percentage_correct_l472_472873

-- Definitions based on conditions
def food_per_week : ℕ := 100
def num_weeks : ℕ := 4
def rent : ℕ := 1500
def video_streaming : ℕ := 30
def cell_phone : ℕ := 50
def savings : ℕ := 198

-- Total spending calculations based on the conditions
def food_total : ℕ := food_per_week * num_weeks
def total_spending : ℕ := food_total + rent + video_streaming + cell_phone

-- Calculation of the percentage
def savings_percentage (savings total_spending : ℕ) : ℕ :=
  (savings * 100) / total_spending

-- The statement to prove
theorem savings_percentage_correct : savings_percentage savings total_spending = 10 := by
  sorry

end savings_percentage_correct_l472_472873


namespace correct_average_l472_472811

theorem correct_average (avg: ℕ) (n: ℕ) (incorrect: ℕ) (correct: ℕ) 
  (h_avg : avg = 16) (h_n : n = 10) (h_incorrect : incorrect = 25) (h_correct : correct = 35) :
  (avg * n + (correct - incorrect)) / n = 17 := 
by
  sorry

end correct_average_l472_472811


namespace bug_position_after_jumps_l472_472343

def circle_points := {1, 2, 3, 4, 5, 6, 7}
def start_point := 7
def jump_rule (point : ℕ) : ℕ :=
  if point % 2 = 1 then (point + 2) % 7
  else (point + 3) % 7

theorem bug_position_after_jumps :
  let final_position := (nat.iterate jump_rule 3000 start_point : ℕ) in
  final_position = 2 :=
by
  -- proof steps would go here
  sorry

end bug_position_after_jumps_l472_472343


namespace all_statements_true_l472_472693

variable (b h r : ℝ) (a divisor numerator : ℝ) (circumference: ℝ)

-- Definitions for each statement
def statement_A : Prop := ∀ (b h : ℝ), 2 * (b * h) = 2 * h * b
def statement_B : Prop := ∀ (b h : ℝ), 2 * (1/2 * b * h) = b * h
def statement_C : Prop := ∀ (r : ℝ), 2 * (2 * real.pi * r) = 4 * real.pi * r
def statement_D : Prop := ∀ (a divisor : ℝ), 2 * a / (2 * divisor) = a / divisor
def statement_E : Prop := ∀ x : ℝ, x < 0 → 3 * x < x

-- The proof problem that all statements are true given the mathematical conditions
theorem all_statements_true : 
  statement_A ∧ 
  statement_B ∧ 
  statement_C ∧ 
  statement_D ∧ 
  statement_E := 
by 
  sorry

end all_statements_true_l472_472693


namespace hyperbola_center_l472_472886

theorem hyperbola_center : 
  (∃ x y : ℝ, (4 * y + 6)^2 / 16 - (5 * x - 3)^2 / 9 = 1) →
  (∃ h k : ℝ, h = 3 / 5 ∧ k = -3 / 2 ∧ 
    (∀ x' y', (4 * y' + 6)^2 / 16 - (5 * x' - 3)^2 / 9 = 1 → x' = h ∧ y' = k)) :=
sorry

end hyperbola_center_l472_472886


namespace intersection_point_l472_472051

def L1 (x y : ℚ) : Prop := y = -3 * x
def L2 (x y : ℚ) : Prop := y + 4 = 9 * x

theorem intersection_point : ∃ x y : ℚ, L1 x y ∧ L2 x y ∧ x = 1/3 ∧ y = -1 := sorry

end intersection_point_l472_472051


namespace bill_new_profit_percentage_l472_472479

theorem bill_new_profit_percentage 
  (original_SP : ℝ)
  (profit_percent : ℝ)
  (increment : ℝ)
  (CP : ℝ)
  (CP_new : ℝ)
  (SP_new : ℝ)
  (Profit_new : ℝ)
  (new_profit_percent : ℝ) :
  original_SP = 439.99999999999966 →
  profit_percent = 0.10 →
  increment = 28 →
  CP = original_SP / (1 + profit_percent) →
  CP_new = CP * (1 - profit_percent) →
  SP_new = original_SP + increment →
  Profit_new = SP_new - CP_new →
  new_profit_percent = (Profit_new / CP_new) * 100 →
  new_profit_percent = 30 :=
by
  -- sorry to skip the proof
  sorry

end bill_new_profit_percentage_l472_472479


namespace prob_a_wins_match_l472_472773

-- Define the probability of A winning a single game
def prob_win_a_single_game : ℚ := 1 / 3

-- Define the probability of A winning two consecutive games
def prob_win_a_two_consec_games : ℚ := prob_win_a_single_game * prob_win_a_single_game

-- Define the probability of A winning two games with one loss in between
def prob_win_a_two_wins_one_loss_first : ℚ := prob_win_a_single_game * (1 - prob_win_a_single_game) * prob_win_a_single_game
def prob_win_a_two_wins_one_loss_second : ℚ := (1 - prob_win_a_single_game) * prob_win_a_single_game * prob_win_a_single_game

-- Define the total probability of A winning the match
def prob_a_winning_match : ℚ := prob_win_a_two_consec_games + prob_win_a_two_wins_one_loss_first + prob_win_a_two_wins_one_loss_second

-- The theorem to be proved
theorem prob_a_wins_match : prob_a_winning_match = 7 / 27 :=
by sorry

end prob_a_wins_match_l472_472773


namespace alpha_beta_cubes_sum_l472_472356

noncomputable def alpha_beta_equation : polynomial ℝ :=
  polynomial.C (sqrt (sqrt 2 + 1)) +
  polynomial.C (2 * sqrt (sqrt 2 + 1)) * polynomial.X +
  polynomial.X ^ 2

theorem alpha_beta_cubes_sum :
  ∃ α β : ℝ, 
  (alpha_beta_equation.eval α = 0 ∧ alpha_beta_equation.eval β = 0 ) →
  (α ≠ β ∧ 
  (1 / α ^ 3) + (1 / β ^ 3) = 6 * sqrt (sqrt 2 + 1) * (sqrt 2 - 1) - 8) :=
by
  sorry

end alpha_beta_cubes_sum_l472_472356


namespace find_sequences_sum_first_n_terms_range_of_a_l472_472380

section ArithmeticGeometricSequences

variables {a_n b_n S_n T_n : ℕ → ℕ} {a : ℝ}

-- Definitions and conditions given in the problem
def arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) := ∀ n, a_n n = 3 + (n - 1) * d
def geometric_sequence (b_n : ℕ → ℕ) (q : ℕ) := ∀ n, b_n n = 2 * q^(n - 1)
def S (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) := ∀ n, S_n n = finset.sum (finset.range n) (λ k, a_n (k + 1))
def T (T_n : ℕ → ℕ) (a_n b_n : ℕ → ℕ) := ∀ n, T_n n = finset.sum (finset.range n) (λ k, a_n (k + 1) * b_n (k + 1))

-- Proof problem 1: Finding a_n and b_n
theorem find_sequences (d q : ℕ) (a_n b_n : ℕ → ℕ) (S_n : ℕ → ℕ) (h_arithmetic : arithmetic_sequence a_n d) (h_geometric : geometric_sequence b_n q) 
  (h_cond_1 : b_n 2 * S_n 2 = 32) (h_cond_2 : b_n 3 * S_n 3 = 120) :
  (a_n = λ n, 2 * n + 1) ∧ (b_n = λ n, 2^n) :=
sorry

-- Proof problem 2: Finding T_n
theorem sum_first_n_terms (a_n b_n T_n : ℕ → ℕ) :
  (T_n = λ n, (2 * n - 1) * 2^(n + 1) + 2) :=
sorry

-- Proof problem 3: Range of a
theorem range_of_a {S_n : ℕ → ℕ} (H : ∀ n, S_n n = n * (n + 2)) :
  (∀ n x, (finset.sum (finset.range n) (λ k, 1 / S_n (k + 1)) ≤ x^2 + a * x + 1) → -1 ≤ a ∧ a ≤ 1) :=
sorry

end ArithmeticGeometricSequences

end find_sequences_sum_first_n_terms_range_of_a_l472_472380


namespace cycling_journey_l472_472859

theorem cycling_journey :
  ∃ y : ℚ, 0 < y ∧ y <= 12 ∧ (15 * y + 10 * (12 - y) = 150) ∧ y = 6 :=
by
  sorry

end cycling_journey_l472_472859


namespace conditional_statements_needed_l472_472559

theorem conditional_statements_needed (P1 P2 P3 : Prop) 
  (H1 : ∃ x : ℝ, P1) -- Problem ①: Input a number x, output its absolute value.
  (H2 : P2)          -- Problem ②: Find the volume of a cube with a surface area of 6.
  (H3 : ∃ x : ℝ, P3) -- Problem ③: Calculate the value of function f(x) as given.  
  (C1 : ∃ x : ℝ, x < 0 ∨ 0 ≤ x → P1) -- This is equivalent to needing a conditional statement for problem ①.
  (NC2 : P2)       -- Problem ② does not need a conditional statement.
  (C3 : ∃ x : ℝ, x < 0 ∨ 0 ≤ x → P3) -- This is equivalent to needing a conditional statement for problem ③.
  : (C1 ∧ ∼NC2 ∧ C3) := 
sorry

end conditional_statements_needed_l472_472559


namespace max_servings_possible_l472_472020

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l472_472020


namespace problem1_problem2_l472_472181

noncomputable def vec2 := (ℝ × ℝ)

def OA : vec2 := (-1, 3)
def OB : vec2 := (3, -1)
def OC (m : ℝ) : vec2 := (m, 1)

-- first problem
theorem problem1 (m : ℝ) : 
  let AB := (OB.1 - OA.1, OB.2 - OA.2)
  ∃ k : ℝ, AB = (k * OC(m).1, k * OC(m).2) → m = -1 :=
by
  sorry

-- second problem
theorem problem2 (m : ℝ) : 
  let AC := (m + 1, -2)
  let BC := (m - 3, 2)
  (AC.1 * BC.1 + AC.2 * BC.2) = 0 → (m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2) :=
by
  sorry

end problem1_problem2_l472_472181


namespace problem_solution_l472_472348

-- Definitions
def condition (a b : ℝ) : Prop := (a - 2)^2 + sqrt (b + 1) = 0
def expression (a b : ℝ) : ℝ := 
  (a^2 - 2 * a * b + b^2) / (a^2 - b^2) / ((a^2 - a * b) / a) - (2 / (a + b))

-- Theorem
theorem problem_solution (a b : ℝ) (h : condition a b) : expression a b = -1 :=
by
  -- the proof would go here
  sorry

end problem_solution_l472_472348


namespace cylindrical_hole_area_l472_472829

variables (R r : ℝ) (h : r ≤ R)
def m : ℝ := r / R
noncomputable def K : ℝ := ∫ v in 0..1, (1 / real.sqrt ((1 - v^2) * (1 - (m r R)^2 * v^2)))
noncomputable def E : ℝ := ∫ v in 0..1, real.sqrt ((1 - (m r R)^2 * v^2) / (1 - v^2))
noncomputable def S1 : ℝ := 8 * r^2 * ∫ v in 0..1, ((1 - v^2) / real.sqrt ((1 - v^2) * (1 - (m r R)^2 * v^2)))
noncomputable def S2 : ℝ := 8 * (R^2 * E R r - (R^2 - r^2) * K R r)

theorem cylindrical_hole_area :
  S1 R r = S2 R r :=
sorry

end cylindrical_hole_area_l472_472829


namespace cost_apples_l472_472750

def total_cost := 42
def cost_bananas := 12
def cost_bread := 9
def cost_milk := 7

theorem cost_apples:
  total_cost - (cost_bananas + cost_bread + cost_milk) = 14 :=
by
  sorry

end cost_apples_l472_472750


namespace libby_quarters_left_l472_472312

theorem libby_quarters_left (initial_quarters : ℕ) (dress_cost_dollars : ℕ) (quarters_per_dollar : ℕ) 
  (h1 : initial_quarters = 160) (h2 : dress_cost_dollars = 35) (h3 : quarters_per_dollar = 4) : 
  initial_quarters - (dress_cost_dollars * quarters_per_dollar) = 20 := by
  sorry

end libby_quarters_left_l472_472312


namespace find_n_l472_472202

theorem find_n (x : ℝ) (n : ℝ) 
    (h1 : log 10 (sin x) + log 10 (cos x) = -2) 
    (h2 : log 10 ((sin x + cos x)^2) = log 10 n - 1) : 
    n = 10.2 :=
sorry

end find_n_l472_472202


namespace polynomial_degree_l472_472792

def polynomial := 5 * X ^ 3 + 7
def exponent := 10
def degree_of_polynomial := 3
def final_degree := 30

theorem polynomial_degree : degree (polynomial ^ exponent) = final_degree :=
by
  sorry

end polynomial_degree_l472_472792


namespace problem_solution_l472_472893

theorem problem_solution : 
  (Int.floor ((Real.ceil ((15 / 8 : Real) ^ 2) + (19 / 5 : Real) - (3 / 2 : Real))) = 6) :=
by
  sorry

end problem_solution_l472_472893


namespace savings_percentage_first_year_l472_472806

noncomputable def savings_percentage (I S : ℝ) : ℝ := (S / I) * 100

theorem savings_percentage_first_year (I S : ℝ) (h1 : S = 0.20 * I) :
  savings_percentage I S = 20 :=
by
  unfold savings_percentage
  rw [h1]
  field_simp
  norm_num
  sorry

end savings_percentage_first_year_l472_472806


namespace cuboid_on_sphere_surface_area_l472_472089

-- Definitions based on conditions
def cuboid_edge_lengths : ℝ × ℝ × ℝ := (1, 2, 3)
def space_diagonal (a b c : ℝ) : ℝ := Real.sqrt (a^2 + b^2 + c^2)

-- Theorem statement
theorem cuboid_on_sphere_surface_area :
  let d := space_diagonal 1 2 3 in
  let r := d / 2 in
  4 * Real.pi * r^2 = 14 * Real.pi :=
by sorry

end cuboid_on_sphere_surface_area_l472_472089


namespace vasya_cuts_larger_area_l472_472121

noncomputable def E_Vasya_square_area : ℝ :=
  (1/6) * (1^2) + (1/6) * (2^2) + (1/6) * (3^2) + (1/6) * (4^2) + (1/6) * (5^2) + (1/6) * (6^2)

noncomputable def E_Asya_rectangle_area : ℝ :=
  (3.5 * 3.5)

theorem vasya_cuts_larger_area :
  E_Vasya_square_area > E_Asya_rectangle_area :=
  by
    sorry

end vasya_cuts_larger_area_l472_472121


namespace trig_eq_solution_l472_472713

open Real

theorem trig_eq_solution (x : ℝ) :
    (∃ k : ℤ, x = -arccos ((sqrt 13 - 1) / 4) + 2 * k * π) ∨ 
    (∃ k : ℤ, x = -arccos ((1 - sqrt 13) / 4) + 2 * k * π) ↔ 
    (cos 5 * x - cos 7 * x) / (sin 4 * x + sin 2 * x) = 2 * abs (sin 2 * x) := by
  sorry

end trig_eq_solution_l472_472713


namespace cistern_width_l472_472828

theorem cistern_width (w : ℝ) (h : 1.25) (length : 7) (A : 55.5) : w = 4 := 
by
  have bottom_area : ℝ := length * w
  have longer_side_area : ℝ := length * h
  have shorter_side_area : ℝ := w * h
  have total_area : ℝ := bottom_area + 2 * longer_side_area + 2 * shorter_side_area
  have formula : total_area = A
  sorry

end cistern_width_l472_472828


namespace staircase_perimeter_l472_472274

theorem staircase_perimeter :
  let rect_base := 13
  let right_triangle_area := (3 * 4) / 2
  let square_area := 12 * (2 * 2)
  let region_area := 150
  let x := (region_area + square_area - right_triangle_area) / rect_base
  let perimeter := x + 13 + 2 * 10 + 3 + 2 * 8 + 1 + 2 * 6 + 1 + 2 * 4 + 1 + 2 * 2 + 4
  perimeter ≈  81.77 :=
begin
  sorry
end

end staircase_perimeter_l472_472274


namespace Bennett_sales_l472_472861

-- Define the variables for the number of screens sold in each month.
variables (J F M : ℕ)

-- State the given conditions.
theorem Bennett_sales (h1: F = 2 * J) (h2: F = M / 4) (h3: M = 8800) :
  J + F + M = 12100 := by
sorry

end Bennett_sales_l472_472861


namespace second_catch_approx_l472_472257

noncomputable def fish_caught_in_second_catch : ℕ := 80

theorem second_catch_approx (N x : ℕ) (hN : N ≈ 3200) (tagged_fish : ℕ) (h_tagged_fish : tagged_fish = 80) 
  (tagged_in_second_catch : ℕ) (h_tagged_in_second_catch : tagged_in_second_catch = 2) :
  tagged_in_second_catch / ↑x ≈ tagged_fish / ↑N → x ≈ fish_caught_in_second_catch :=
by
  sorry

end second_catch_approx_l472_472257


namespace cost_of_apples_l472_472748

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l472_472748


namespace width_of_rectangle_l472_472660

-- Define the side length of the square and the length of the rectangle.
def side_length_square : ℝ := 12
def length_rectangle : ℝ := 18

-- Calculate the perimeter of the square.
def perimeter_square : ℝ := 4 * side_length_square

-- This definition represents the perimeter of the rectangle made from the same wire.
def perimeter_rectangle : ℝ := perimeter_square

-- Show that the width of the rectangle is 6 cm.
theorem width_of_rectangle : ∃ W : ℝ, 2 * (length_rectangle + W) = perimeter_rectangle ∧ W = 6 :=
by
  use 6
  simp [length_rectangle, perimeter_rectangle, side_length_square]
  norm_num
  sorry

end width_of_rectangle_l472_472660


namespace Cody_total_bill_l472_472129

-- Definitions for the problem
def cost_per_child : ℝ := 7.5
def cost_per_adult : ℝ := 12.0

variables (A C : ℕ)

-- Conditions
def condition1 : Prop := C = A + 8
def condition2 : Prop := A + C = 12

-- Total bill
def total_cost := (A * cost_per_adult) + (C * cost_per_child)

-- The proof statement
theorem Cody_total_bill (h1 : condition1 A C) (h2 : condition2 A C) : total_cost A C = 99.0 := by
  sorry

end Cody_total_bill_l472_472129


namespace max_servings_l472_472035

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l472_472035


namespace value_of_f_at_sqrt2_l472_472684

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then log (x + 1) / log (1 / 2) else f (x - 1)

theorem value_of_f_at_sqrt2 : f (sqrt 2) = -1 / 2 :=
by
  sorry

end value_of_f_at_sqrt2_l472_472684


namespace total_right_handed_players_is_correct_l472_472322

variable (total_players : ℕ)
variable (throwers : ℕ)
variable (left_handed_non_throwers_ratio : ℕ)
variable (total_right_handed_players : ℕ)

theorem total_right_handed_players_is_correct
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers_ratio = 1 / 3)
  (h4 : total_right_handed_players = 53) :
  total_right_handed_players = throwers + (total_players - throwers) -
    left_handed_non_throwers_ratio * (total_players - throwers) :=
by
  sorry

end total_right_handed_players_is_correct_l472_472322


namespace largest_number_divisible_by_48_is_9984_l472_472795

def largest_divisible_by_48 (n : ℕ) := ∀ m ≥ n, m % 48 = 0 → m ≤ 9999

theorem largest_number_divisible_by_48_is_9984 :
  largest_divisible_by_48 9984 ∧ 9999 / 10^3 = 9 ∧ 48 ∣ 9984 ∧ 9984 < 10000 :=
by
  sorry

end largest_number_divisible_by_48_is_9984_l472_472795


namespace T_sum_is_correct_l472_472516

-- Define the geometric series sum T(r)
def T (r : ℝ) : ℝ := 18 / (1 - r)

-- State the problem as a Lean theorem
theorem T_sum_is_correct (b : ℝ) (h_b_range : -1 < b ∧ b < 1) (h_T_equation : T b * T (-b) = 3024) : T b + T (-b) = 337.5 :=
sorry

end T_sum_is_correct_l472_472516


namespace find_m_value_l472_472619

theorem find_m_value
  (m : ℝ)
  (h1 : 10 - m > 0)
  (h2 : m - 2 > 0)
  (h3 : 2 * Real.sqrt (10 - m - (m - 2)) = 4) :
  m = 4 := by
sorry

end find_m_value_l472_472619


namespace camille_saw_31_birds_l472_472484

def num_cardinals : ℕ := 3
def num_robins : ℕ := 4 * num_cardinals
def num_blue_jays : ℕ := 2 * num_cardinals
def num_sparrows : ℕ := 3 * num_cardinals + 1
def total_birds : ℕ := num_cardinals + num_robins + num_blue_jays + num_sparrows

theorem camille_saw_31_birds : total_birds = 31 := by
  sorry

end camille_saw_31_birds_l472_472484


namespace population_reaches_target_l472_472125

def initial_year : ℕ := 2020
def initial_population : ℕ := 450
def growth_period : ℕ := 25
def growth_factor : ℕ := 3
def target_population : ℕ := 10800

theorem population_reaches_target : ∃ (year : ℕ), year - initial_year = 3 * growth_period ∧ (initial_population * growth_factor ^ 3) >= target_population := by
  sorry

end population_reaches_target_l472_472125


namespace union_of_sets_l472_472683

def A : Set ℝ := {x | ∃ y : ℝ, y = log 2 (x - 2)}

def B : Set ℝ := {x | x^2 - 5 * x + 4 < 0}

theorem union_of_sets : A ∪ B = {x | 1 < x} :=
by
  sorry

end union_of_sets_l472_472683


namespace choose_five_points_from_35gon_l472_472529

def points : Type := fin 35 → (ℝ × ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def convex_35gon (P : points) : Prop := sorry

def all_dist_at_least_sqrt3 (P : points) : Prop :=
  ∀ i j : fin 35, i ≠ j → distance (P i) (P j) ≥ real.sqrt 3

def exists_five_points (P : points) : Prop :=
  ∃ Q : fin 5 → fin 35,
  ∀ i j : fin 5, i ≠ j → distance (P (Q i)) (P (Q j)) ≥ 3

theorem choose_five_points_from_35gon (P : points) :
  convex_35gon P → all_dist_at_least_sqrt3 P → exists_five_points P :=
sorry

end choose_five_points_from_35gon_l472_472529


namespace Menelaus_proof_Ceva_proof_l472_472194

-- Definitions for the problem
structure TrihedralAngle where
  vertex : Point
  edge_a : Line
  edge_b : Line
  edge_c : Line 
  ray_alpha : Ray
  ray_beta : Ray
  ray_gamma : Ray

def MenelausFirstTheorem (T : TrihedralAngle) : Prop :=
  sin (angle T.edge_a T.ray_gamma) / sin (angle T.edge_b T.ray_gamma) * 
  sin (angle T.edge_b T.ray_alpha) / sin (angle T.edge_c T.ray_alpha) * 
  sin (angle T.edge_c T.ray_beta) / sin (angle T.edge_a T.ray_beta) = 1

def CevaFirstTheorem (T : TrihedralAngle) : Prop :=
  sin (angle T.edge_a T.ray_gamma) / sin (angle T.edge_b T.ray_gamma) * 
  sin (angle T.edge_b T.ray_alpha) / sin (angle T.edge_c T.ray_alpha) * 
  sin (angle T.edge_c T.ray_beta) / sin (angle T.edge_a T.ray_beta) = -1

-- The statement of part (a)
theorem Menelaus_proof {T : TrihedralAngle} : 
  Planar (Set.of [T.ray_alpha, T.ray_beta, T.ray_gamma]) ↔ MenelausFirstTheorem T := 
sorry

-- The statement of part (b)
theorem Ceva_proof {T : TrihedralAngle} : 
  ¬ SingleLineIntersection (Set.of [(T.edge_a, T.ray_alpha), (T.edge_b, T.ray_beta), (T.edge_c, T.ray_gamma)]) ↔ CevaFirstTheorem T := 
sorry

end Menelaus_proof_Ceva_proof_l472_472194


namespace underpaid_wages_per_worker_l472_472444

theorem underpaid_wages_per_worker (hourly_rate : ℝ) (daily_hours : ℝ) 
(minute_hand_coincide_time : ℝ) (actual_coincide_time : ℝ) :
  hourly_rate = 6 →
  daily_hours = 8 →
  actual_coincide_time = 69 →
  minute_hand_coincide_time = (60 + 60 / 11) →
  let actual_working_hours := daily_hours * (actual_coincide_time / minute_hand_coincide_time) in
  let excess_hours := actual_working_hours - daily_hours in
  let underpaid_wages := hourly_rate * excess_hours in
  underpaid_wages = 2.60 :=
begin
  intros h1 h2 h3 h4,
  have h5 := h1,
  have h6 := h2,
  have h7 := h3,
  have h8 := h4,
  sorry
end

end underpaid_wages_per_worker_l472_472444


namespace camille_birds_count_l472_472486

theorem camille_birds_count : 
  let cardinals := 3 in
  let robins := 4 * cardinals in
  let blue_jays := 2 * cardinals in
  let sparrows := 3 * cardinals + 1 in
  cardinals + robins + blue_jays + sparrows = 31 :=
by 
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  show cardinals + robins + blue_jays + sparrows = 31
  calc 
    cardinals + robins + blue_jays + sparrows = 3 + (4 * 3) + (2 * 3) + (3 * 3 + 1) : by rfl
    ... = 3 + 12 + 6 + 10 : by rfl
    ... = 31 : by rfl

end camille_birds_count_l472_472486


namespace hyperbola_eccentricity_l472_472980

theorem hyperbola_eccentricity (m n : ℝ) (h : m * n < 0)
    (h_asymp_tangent : ∃ (x y : ℝ), x^2 + y^2 - 6*x - 2*y + 9 = 0 ∧
                      (mx^2 + ny^2 = 1 → (√|m| * x ± √|n| * y = 0) ⊂ tangent_line (x, y))) :
    (eccentricity (hyperbola C) = 5/3 ∨ eccentricity (hyperbola C) = 5/4) :=
sorry

end hyperbola_eccentricity_l472_472980


namespace orthodiagonal_quadrilateral_l472_472662

-- Define the quadrilateral sides and their relationships
variables (AB BC CD DA : ℝ)
variables (h1 : AB = 20) (h2 : BC = 70) (h3 : CD = 90)
theorem orthodiagonal_quadrilateral : AB^2 + CD^2 = BC^2 + DA^2 → DA = 60 :=
by
  sorry

end orthodiagonal_quadrilateral_l472_472662


namespace no_infinite_lines_satisfying_conditions_l472_472890

theorem no_infinite_lines_satisfying_conditions :
  ¬ ∃ (l : ℕ → ℝ → ℝ → Prop)
      (k : ℕ → ℝ)
      (a b : ℕ → ℝ),
    (∀ n, l n 1 1) ∧
    (∀ n, k (n + 1) = a n - b n) ∧
    (∀ n, k n * k (n + 1) ≥ 0) := 
sorry

end no_infinite_lines_satisfying_conditions_l472_472890


namespace problem1_problem2_problem3_l472_472307

-- Definitions of sets A, B, and C as per given conditions
def set_A (a : ℝ) : Set ℝ :=
  {x | x^2 - a * x + a^2 - 19 = 0}

def set_B : Set ℝ :=
  {x | x^2 - 5 * x + 6 = 0}

def set_C : Set ℝ :=
  {x | x^2 + 2 * x - 8 = 0}

-- Questions reformulated as proof problems
theorem problem1 (a : ℝ) (h : set_A a = set_B) : a = 5 :=
sorry

theorem problem2 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : ∀ x, x ∈ set_A a → x ∉ set_C) : a = -2 :=
sorry

theorem problem3 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : set_A a ∩ set_B = set_A a ∩ set_C) : a = -3 :=
sorry

end problem1_problem2_problem3_l472_472307


namespace smallest_initial_cells_to_become_unbounded_l472_472455

theorem smallest_initial_cells_to_become_unbounded (N : ℕ) :
  (∀ t : ℕ, (if t % 2 = 1 then 1 else 0) * (N - 30) > N / 2 → False → 61 ≤ N) :=
begin 
  sorry
end

end smallest_initial_cells_to_become_unbounded_l472_472455


namespace percentage_increase_expenditure_l472_472329

variable (I : ℝ) -- original income
variable (E : ℝ) -- original expenditure
variable (I_new : ℝ) -- new income
variable (S : ℝ) -- original savings
variable (S_new : ℝ) -- new savings

-- a) Conditions
def initial_spend (I : ℝ) : ℝ := 0.75 * I
def income_increased (I : ℝ) : ℝ := 1.20 * I
def savings_increased (S : ℝ) : ℝ := 1.4999999999999996 * S

-- b) Definitions relating formulated conditions
def new_expenditure (I : ℝ) : ℝ := 1.20 * I - 0.3749999999999999 * I
def original_expenditure (I : ℝ) : ℝ := 0.75 * I

-- c) Proof statement
theorem percentage_increase_expenditure :
  initial_spend I = E →
  income_increased I = I_new →
  savings_increased (0.25 * I) = S_new →
  ((new_expenditure I - original_expenditure I) / original_expenditure I) * 100 = 10 := 
by 
  intros h1 h2 h3
  sorry

end percentage_increase_expenditure_l472_472329


namespace problem1_l472_472438

theorem problem1 (a b c : ℝ) (h : a * c + b * c + c^2 < 0) : b^2 > 4 * a * c := sorry

end problem1_l472_472438


namespace limits_of_ratios_l472_472187

noncomputable def x_n (n : ℕ) : ℝ := (1 + Real.sqrt 2 + Real.sqrt 3) ^ n

noncomputable def q_n (n : ℕ) : ℤ := -- definition derived from x_n
noncomputable def r_n (n : ℕ) : ℤ := -- definition derived from x_n
noncomputable def s_n (n : ℕ) : ℤ := -- definition derived from x_n
noncomputable def t_n (n : ℕ) : ℤ := -- definition derived from x_n

theorem limits_of_ratios :
  (∀ n, x_n n = (q_n n : ℝ) + (r_n n : ℝ) * Real.sqrt 2 + (s_n n : ℝ) * Real.sqrt 3 + (t_n n : ℝ) * Real.sqrt 6) →
  (∀ k, 2 ≤ k → |1 - Real.sqrt 2 + Real.sqrt 3| < |1 + Real.sqrt 2 + Real.sqrt 3| ∧
               |1 + Real.sqrt 2 - Real.sqrt 3| < |1 + Real.sqrt 2 + Real.sqrt 3| ∧
               |1 - Real.sqrt 2 - Real.sqrt 3| < |1 + Real.sqrt 2 + Real.sqrt 3|) →
  (lim n → ∞, (r_n n : ℝ) / (q_n n : ℝ)) = 1 / Real.sqrt 2 ∧
  (lim n → ∞, (s_n n : ℝ) / (q_n n : ℝ)) = 1 / Real.sqrt 3 ∧
  (lim n → ∞, (t_n n : ℝ) / (q_n n : ℝ)) = 1 / Real.sqrt 6 :=
by
  intros; sorry

end limits_of_ratios_l472_472187


namespace find_k_l472_472969

def f (x : ℝ) : ℝ := 2^x + x - 8

theorem find_k (k : ℤ) (h : ∃ x : ℝ, f x = 0 ∧ x ∈ (k : ℝ, k + 1)) : k = 2 :=
sorry

end find_k_l472_472969


namespace tan_pi_six_expr_is_tenth_root_l472_472480

noncomputable def tan_pi_six : ℂ := real.tan (real.pi / 6)

def tenth_root_of_unity (n : ℤ) : ℂ := complex.exp (2 * real.pi * complex.I * n / 10)

theorem tan_pi_six_expr_is_tenth_root : 
  (tan_pi_six + complex.I) / (tan_pi_six - complex.I) = tenth_root_of_unity 1 := by
  sorry

end tan_pi_six_expr_is_tenth_root_l472_472480


namespace quadratic_root_difference_l472_472491

theorem quadratic_root_difference 
  (a b c : ℝ) (h_a : a = 5 + 3 * Real.sqrt 5) (h_b : b = 5 + Real.sqrt 5) (h_c : c = -3) :
  let x₁ := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a),
      x₂ := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  in abs (x₁ - x₂) = 2 * Real.sqrt 5 + 1 / 2 :=
by 
  sorry

end quadratic_root_difference_l472_472491


namespace sum_of_digits_concat_1_to_999_l472_472420

theorem sum_of_digits_concat_1_to_999 : 
  (∑ i in list.range 999, (i + 1).digits.sum) = 13500 := by
  sorry

end sum_of_digits_concat_1_to_999_l472_472420


namespace reporters_cover_local_politics_percentage_l472_472824

theorem reporters_cover_local_politics_percentage (
  total_reporters : ℕ,
  reporters_cover_politics : ℕ,
  reporters_cover_no_local_politics : ℕ
  (h1 : reporters_cover_politics = 0.25 * total_reporters)
  (h2 : reporters_cover_no_local_politics = 0.20 * reporters_cover_politics)
  (h3 : 0.75 * total_reporters = total_reporters - reporters_cover_politics)
) : reporters_cover_local_politics = 0.20 * total_reporters :=
by
  let local_politics_reporters := reporters_cover_politics - reporters_cover_no_local_politics
  have h4 : local_politics_reporters = reporters_cover_local_politics
  -- Prove the necessary steps
  sorry

end reporters_cover_local_politics_percentage_l472_472824


namespace width_of_room_is_3_75_l472_472731

/-- Define the length of the room --/
def length_of_room : ℝ := 5.5

/-- Define the total cost of paving --/
def total_cost : ℝ := 16500

/-- Define the cost per square meter --/
def cost_per_sq_meter : ℝ := 800

/-- Calculated area of the room from the given costs --/
def area_of_room : ℝ := total_cost / cost_per_sq_meter

/-- Definition of the width of the room using the area and the length --/
def width_of_room : ℝ := area_of_room / length_of_room

/-- The theorem to prove the width of the room --/
theorem width_of_room_is_3_75 : width_of_room = 3.75 :=
by
  /-- Proof is required here --/
  sorry

end width_of_room_is_3_75_l472_472731


namespace percentage_increase_expenditure_l472_472328

variable (I : ℝ) -- original income
variable (E : ℝ) -- original expenditure
variable (I_new : ℝ) -- new income
variable (S : ℝ) -- original savings
variable (S_new : ℝ) -- new savings

-- a) Conditions
def initial_spend (I : ℝ) : ℝ := 0.75 * I
def income_increased (I : ℝ) : ℝ := 1.20 * I
def savings_increased (S : ℝ) : ℝ := 1.4999999999999996 * S

-- b) Definitions relating formulated conditions
def new_expenditure (I : ℝ) : ℝ := 1.20 * I - 0.3749999999999999 * I
def original_expenditure (I : ℝ) : ℝ := 0.75 * I

-- c) Proof statement
theorem percentage_increase_expenditure :
  initial_spend I = E →
  income_increased I = I_new →
  savings_increased (0.25 * I) = S_new →
  ((new_expenditure I - original_expenditure I) / original_expenditure I) * 100 = 10 := 
by 
  intros h1 h2 h3
  sorry

end percentage_increase_expenditure_l472_472328


namespace max_servings_l472_472028

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l472_472028


namespace speed_limit_correct_l472_472100

def speed_limit_statement (v : ℝ) : Prop :=
  v ≤ 70

theorem speed_limit_correct (v : ℝ) (h : v ≤ 70) : speed_limit_statement v :=
by
  exact h

#print axioms speed_limit_correct

end speed_limit_correct_l472_472100


namespace curve_C_equation_l472_472582

noncomputable def vector_m1 (x : ℝ) : ℝ × ℝ := (0, x)
noncomputable def vector_n1 : ℝ × ℝ := (1, 1)
noncomputable def vector_m2 (x : ℝ) : ℝ × ℝ := (x, 0)
noncomputable def vector_n2 (y : ℝ) : ℝ × ℝ := (y^2, 1)
noncomputable def vector_m (x y : ℝ) : ℝ × ℝ := (sqrt 2 * y^2, x + sqrt 2)
noncomputable def vector_n (x : ℝ) : ℝ × ℝ := (x - sqrt 2, -sqrt 2)

theorem curve_C_equation (x y : ℝ) :
  (vector_m x y) = (fst (vector_n x)) • (vector_n x) ↔
  (x^2 / 2) + (y^2) = 1 :=
by sorry

end curve_C_equation_l472_472582


namespace radius_of_inscribed_box_l472_472098

theorem radius_of_inscribed_box (a b c : ℝ) (r : ℝ) 
  (h1 : a + b + c = 42) 
  (h2 : 2 * (a * b + b * c + c * a) = 672) 
  (h3 : a^2 + b^2 + c^2 = (a + b + c)^2 - 2 * (a * b + b * c + c * a)) :
  r = Real.sqrt 273 :=
by
  have h4 : 4 * r^2 = 1092, from 
    calc 4 * r^2 = a^2 + b^2 + c^2 : by sorry
    ... = 42^2 - 2 * (a * b + b * c + c * a) : by rw h3
    ... = 1764 - 672 : by simp [h2]
    ... = 1092 : by norm_num
  have h5 : r^2 = 273, from (by norm_num : 4 * 273 = 1092).symm ▸ h4
  exact (Real.sqrt_eq_iff_eq_sq_left r 273).mpr h5

end radius_of_inscribed_box_l472_472098


namespace find_m_value_l472_472581

variable (m : ℝ)
noncomputable def a : ℝ × ℝ := (2 * Real.sqrt 2, 2)
noncomputable def b : ℝ × ℝ := (0, 2)
noncomputable def c (m : ℝ) : ℝ × ℝ := (m, Real.sqrt 2)

theorem find_m_value (h : (a.1 + 2 * b.1) * (m) + (a.2 + 2 * b.2) * (Real.sqrt 2) = 0) : m = -3 :=
by
  sorry

end find_m_value_l472_472581


namespace max_servings_l472_472014

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l472_472014


namespace percentage_increase_in_expenditure_l472_472330

-- Definitions
def original_income (I : ℝ) := I
def expenditure (I : ℝ) := 0.75 * I
def savings (I E : ℝ) := I - E
def new_income (I : ℝ) := 1.2 * I
def new_expenditure (E P : ℝ) := E * (1 + P / 100)
def new_savings (I E P : ℝ) := new_income I - new_expenditure E P

-- Theorem to prove
theorem percentage_increase_in_expenditure (I : ℝ) (P : ℝ) :
  savings I (expenditure I) * 1.5 = new_savings I (expenditure I) P →
  P = 10 :=
by
  intros h
  simp [savings, expenditure, new_income, new_expenditure, new_savings] at h
  sorry

end percentage_increase_in_expenditure_l472_472330


namespace coeff_x70_in_expansion_l472_472902

theorem coeff_x70_in_expansion :
  let p := (∏ i in finset.range 13, polynomial.C (i+1) * polynomial.X ^ (i+1) - polynomial.C (i+1))
  polynomial.coeff p 70 = 4 :=
begin
  -- sorry, this is the proof placeholder
  sorry
end

end coeff_x70_in_expansion_l472_472902


namespace find_wrongly_read_number_l472_472718

def initial_average : ℕ := 14
def correct_average : ℕ := 15
def number_wrongly_read : ℕ := 36

theorem find_wrongly_read_number :
  ∃ (wrongly_read : ℕ),
    let initial_sum := 10 * initial_average,
        correct_sum := 10 * correct_average,
        difference := correct_sum - initial_sum in
      number_wrongly_read - difference = wrongly_read :=
sorry

end find_wrongly_read_number_l472_472718


namespace first_pair_weight_l472_472488

variable (total_weight : ℕ) (second_pair_weight : ℕ) (third_pair_weight : ℕ)

theorem first_pair_weight (h : total_weight = 32) (h_second : second_pair_weight = 5) (h_third : third_pair_weight = 8) : 
    total_weight - 2 * (second_pair_weight + third_pair_weight) = 6 :=
by
  sorry

end first_pair_weight_l472_472488


namespace initial_amount_of_water_l472_472419

theorem initial_amount_of_water 
  (W : ℚ) 
  (h1 : W - (7/15) * W - (5/8) * (W - (7/15) * W) - (2/3) * (W - (7/15) * W - (5/8) * (W - (7/15) * W)) = 2.6) 
  : W = 39 := 
sorry

end initial_amount_of_water_l472_472419


namespace equation_of_ellipse_max_area_l472_472943

noncomputable theory

variables {a b x y : ℝ}

def ellipse (a b : ℝ) := ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1
def eccentricity := a > b ∧ a > 0 ∧ b > 0 ∧ (2 * real.sqrt 2) / 3 = 2 * real.sqrt 2 / 3
def vertex_B : ℝ × ℝ := 0, 1
def perpendicular (P Q : ℝ × ℝ) : Prop := ∀ B : ℝ × ℝ, B = (0,1) → (P.1 - B.1) * (P.2 - B.2) + (Q.1 - B.1) * (Q.2 - B.2) = 0

theorem equation_of_ellipse (h1 : 0 < b) (h2 : b < a) :
  (ellipse 3 1) :=
sorry

theorem max_area (P Q : ℝ × ℝ) (h_perp : perpendicular P Q) :
  (∀ (k : ℝ), ∃ (k : ℝ), real.sqrt (7 / 225) = k ) → 
  (27 / 8) = (27 / 8) :=
sorry

end equation_of_ellipse_max_area_l472_472943


namespace area_of_triangle_ABC_l472_472432

-- Definitions and Conditions of the problem
variables {A B C P Q R S P1 Q1 R1 S1 : Type} [LinearOrderedField A]
variables {PS P1S1 : A} (H_ps : PS = 3) (H_p1s1 : P1S1 = 9) (tri_ABC : triangle A B C)
variables (rect_PQRS : inscribed_rectangle PQRS tri_ABC) (rect_P1Q1R1S1 : inscribed_rectangle P1Q1R1S1 tri_ABC)

theorem area_of_triangle_ABC :
  area tri_ABC = 72 :=
by sorry

end area_of_triangle_ABC_l472_472432


namespace clay_blocks_needed_l472_472450

theorem clay_blocks_needed 
  (block_length : ℝ)
  (block_width : ℝ)
  (block_height : ℝ)
  (cylinder_height : ℝ)
  (cylinder_diameter : ℝ)
  (block_volume := block_length * block_width * block_height)
  (cylinder_radius := cylinder_diameter / 2)
  (cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height)
  (required_blocks := (cylinder_volume / block_volume).ceil) :
  block_length = 8 →
  block_width = 3 →
  block_height = 2 →
  cylinder_height = 9 →
  cylinder_diameter = 6 →
  required_blocks = 6 := 
by
  intros
  -- proof goes here
  sorry

end clay_blocks_needed_l472_472450


namespace determine_h_l472_472495

theorem determine_h (x : ℝ) : 
  ∃ h : ℝ → ℝ, (4*x^4 + 11*x^3 + h x = 10*x^3 - x^2 + 4*x - 7) ↔ (h x = -4*x^4 - x^3 - x^2 + 4*x - 7) :=
by
  sorry

end determine_h_l472_472495


namespace max_servings_l472_472013

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l472_472013


namespace camille_saw_31_birds_l472_472485

def num_cardinals : ℕ := 3
def num_robins : ℕ := 4 * num_cardinals
def num_blue_jays : ℕ := 2 * num_cardinals
def num_sparrows : ℕ := 3 * num_cardinals + 1
def total_birds : ℕ := num_cardinals + num_robins + num_blue_jays + num_sparrows

theorem camille_saw_31_birds : total_birds = 31 := by
  sorry

end camille_saw_31_birds_l472_472485


namespace symmetry_probability_is_two_fifths_l472_472472

-- Define the grid and center point P
def P := (5, 5 : ℕ × ℕ)

-- Define symmetry axes and count favorable points excluding the center
def count_points_on_symmetry_axes : ℕ :=
  let vert_points := 8
  let horiz_points := 8
  let leading_diag_points := 8
  let anti_diag_points := 8
  vert_points + horiz_points + leading_diag_points + anti_diag_points

-- Total number of possible points excluding the center
def total_possible_points := 80

-- Calculate probability
def symmetry_probability : ℚ :=
  count_points_on_symmetry_axes / total_possible_points

-- Final theorem statement
theorem symmetry_probability_is_two_fifths : symmetry_probability = 2 / 5 := by sorry

end symmetry_probability_is_two_fifths_l472_472472


namespace ellipse_properties_l472_472942

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_properties :
  ∃ a : ℝ, a > 0 ∧ a^2 = 3 ∧
  (ellipse_equation a 1) ∧
  (∀ k m : ℝ, k ≠ 0 → 
    (∀ x y : ℝ, ellipse_equation a 1 x y → y = k * x + m → ∃ M N, M ≠ N ∧ |A(0, -1) - M| = |A(0, -1) - N|) →
    (1/2 < m ∧ m < 2)) :=
sorry

end ellipse_properties_l472_472942


namespace proof_equation_of_line_l472_472506

noncomputable def equation_of_line_par_tangent (P : ℝ × ℝ) (M : ℝ × ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  P = (-1, 2) ∧ M = (1, 1) ∧ 
  f x = 3 * x * x - 4 * x + 2 ∧ 
  f' x = 6 * x - 4 ∧ 
  let m := f' 1 in 
  m = 2 ∧ 
  ∀ x y : ℝ, (y - P.2) = m * (x - P.1) ↔ (2 * x - y + 4 = 0)

theorem proof_equation_of_line : equation_of_line_par_tangent (-1,2) (1,1) (λ x, 3 * x^2 - 4 * x + 2) (λ x, 6 * x - 4) := 
begin 
  sorry 
end

end proof_equation_of_line_l472_472506


namespace angle_between_a_b_cos_length_a_add_b_m_value_l472_472528

variables (a b : Vector ℝ) (m : ℝ)

axiom a_norm_eq_two : ∥a∥ = 2
axiom b_norm_eq_one : ∥b∥ = 1
axiom dot_product_eq_seventeen : (2 • a - 3 • b) ⬝ (2 • a + b) = 17

theorem angle_between_a_b_cos :
  Real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)) = 2 * Real.pi / 3 :=
by sorry

theorem length_a_add_b : ∥a + b∥ = Real.sqrt 3 :=
by sorry

axiom c_collinear_d : m • a + 2 • b = λ * (2 • a - b)

theorem m_value : m = -4 :=
by sorry

end angle_between_a_b_cos_length_a_add_b_m_value_l472_472528


namespace problem_m_value_l472_472617

noncomputable def find_m (m : ℝ) : Prop :=
  let a : ℝ := real.sqrt (10 - m)
  let b : ℝ := real.sqrt (m - 2)
  (2 * real.sqrt (a^2 - b^2) = 4) ∧ (10 - m > m - 2) ∧ (m - 2 > 0) ∧ (10 - m > 0)

theorem problem_m_value (m : ℝ) : find_m m → m = 4 := by
  sorry

end problem_m_value_l472_472617


namespace perfect_squares_digits_l472_472247

theorem perfect_squares_digits 
  (a b : ℕ) 
  (ha : ∃ m : ℕ, a = m * m) 
  (hb : ∃ n : ℕ, b = n * n) 
  (a_units_digit_1 : a % 10 = 1) 
  (b_units_digit_6 : b % 10 = 6) 
  (a_tens_digit : ∃ x : ℕ, (a / 10) % 10 = x) 
  (b_tens_digit : ∃ y : ℕ, (b / 10) % 10 = y) : 
  ∃ x y : ℕ, (x % 2 = 0) ∧ (y % 2 = 1) := 
sorry

end perfect_squares_digits_l472_472247


namespace field_trip_probability_field_trip_probability_proof_l472_472837

-- Definition of combinations
def combinations (n k : ℕ) : ℕ := nat.choose n k

-- Definition of the problem conditions and what needs to be proven
theorem field_trip_probability (n_grades : ℕ) (n_museums : ℕ) (MuseumA : ℕ) 
  (chooseMuseumA : ℕ) (remainingMuseums : ℕ) (k : ℕ) 
  (combinations_formula : ℕ) : Prop :=
  (n_grades = 6) →
  (n_museums = 6) →
  (MuseumA = 1) →
  (chooseMuseumA = 2) →
  (remainingMuseums = 5) →
  (k = 4) →
  (combinations_formula = combinations 6 2 * 5^4)

-- Proof placeholder
theorem field_trip_probability_proof : field_trip_probability 6 6 1 2 5 4 (nat.choose 6 2 * 5^4) :=
by
sorry

end field_trip_probability_field_trip_probability_proof_l472_472837


namespace expected_value_fair_8_sided_die_l472_472831

theorem expected_value_fair_8_sided_die :
  ∑ i in (finset.range 8).map (λ x, x + 1), 
  ((if odd x then x^2 else 2*x^2) * (1 / 8 : ℚ)) = 70.5 := 
by
  sorry

end expected_value_fair_8_sided_die_l472_472831


namespace tan_beta_rational_iff_square_l472_472357

theorem tan_beta_rational_iff_square (p q : ℤ) (h : q ≠ 0) :
  (∃ β : ℚ, tan (2 * β) = tan (3 * atan (p / q : ℚ))) ↔ ∃ k : ℤ, p^2 + q^2 = k^2 := 
sorry

end tan_beta_rational_iff_square_l472_472357


namespace volume_frustum_l472_472463

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1/3) * (base_edge ^ 2) * height

theorem volume_frustum (original_base_edge original_height small_base_edge small_height : ℝ)
  (h_orig : original_base_edge = 10) (h_orig_height : original_height = 10)
  (h_small : small_base_edge = 5) (h_small_height : small_height = 5) :
  volume_pyramid original_base_edge original_height - volume_pyramid small_base_edge small_height
  = 875 / 3 := by
    simp [volume_pyramid, h_orig, h_orig_height, h_small, h_small_height]
    sorry

end volume_frustum_l472_472463


namespace smallest_consecutive_divisible_by_17_l472_472060

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_consecutive_divisible_by_17 :
  ∃ (n m : ℕ), 
    (m = n + 1) ∧
    sum_digits n % 17 = 0 ∧ 
    sum_digits m % 17 = 0 ∧ 
    n = 8899 ∧ 
    m = 8900 := 
by
  sorry

end smallest_consecutive_divisible_by_17_l472_472060


namespace cost_of_book_sold_at_loss_l472_472606

noncomputable def C1_and_C2_solution : ℝ :=
  let C1 := 274.17
  let C2 := 470 - C1
  if C1 + C2 = 470 ∧ C1 * 0.85 = C2 * 1.19 then C1 else 0

theorem cost_of_book_sold_at_loss :
    ∃ C1 : ℝ, C1 + (470 - C1) = 470 ∧ C1 * 0.85 = (470 - C1) * 1.19 ∧ C1 = 274.17 :=
by
  use 274.17
  have C2 : ℝ := 470 - 274.17
  split_ifs
  have eq1 : 274.17 + C2 = 470 := by sorry
  have eq2 : 274.17 * 0.85 = C2 * 1.19 := by sorry
  exact And.intro eq1 eq2
  sorry

end cost_of_book_sold_at_loss_l472_472606


namespace cirrus_to_cumulus_is_four_l472_472743

noncomputable def cirrus_to_cumulus_ratio (Ci Cu Cb : ℕ) : ℕ :=
  Ci / Cu

theorem cirrus_to_cumulus_is_four :
  ∀ (Ci Cu Cb : ℕ), (Cb = 3) → (Cu = 12 * Cb) → (Ci = 144) → cirrus_to_cumulus_ratio Ci Cu Cb = 4 :=
by
  intros Ci Cu Cb hCb hCu hCi
  sorry

end cirrus_to_cumulus_is_four_l472_472743


namespace range_of_independent_variable_l472_472273

-- Define the function y
def y (x : ℝ) : ℝ := (3 / (x - 2)) - (real.sqrt (x + 1))

-- Define the range condition
def range_condition (x : ℝ) : Prop := x ≥ -1 ∧ x ≠ 2

-- State the proof problem
theorem range_of_independent_variable :
  ∀ x : ℝ, (∃ y : ℝ, y = (3 / (x - 2)) - (real.sqrt (x + 1))) → (x ≥ -1 ∧ x ≠ 2) :=
by
  sorry

end range_of_independent_variable_l472_472273


namespace functions_are_even_l472_472057

noncomputable def f_A (x : ℝ) : ℝ := -|x| + 2
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3
noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

theorem functions_are_even :
  (∀ x : ℝ, f_A x = f_A (-x)) ∧
  (∀ x : ℝ, f_B x = f_B (-x)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_C x = f_C (-x)) :=
by
  sorry

end functions_are_even_l472_472057


namespace solve_for_y_l472_472425

theorem solve_for_y (x y : ℝ) (h1 : x * y = 25) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 5 / 6 := 
by
  sorry

end solve_for_y_l472_472425


namespace distance_from_circle_center_to_line_l472_472904

-- Define the circle and the line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 2 * y = 0
def line_eq (x y : ℝ) : Prop := y = x + 1

-- Define the distance formula between a point and a line
def distance_point_line (x y A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / sqrt (A^2 + B^2)

-- Prove the distance from the center of the circle to the line is the specified value
theorem distance_from_circle_center_to_line :
  let x0 := 1 in
  let y0 := -1 in
  distance_point_line x0 y0 (-1) 1 (-1) = 3 * sqrt 2 / 2 :=
by
  sorry

end distance_from_circle_center_to_line_l472_472904


namespace probability_divisible_by_4_l472_472066

theorem probability_divisible_by_4 : 
  let digits := [2, 45, 68]
  let form_number (d1 d2 d3 d4 d5 : Nat) := d1 * 10000 + d2 * 1000 + (d3 * 100 + d4 * 10 + d5)
  ( ∑(d1 in digits), ∑(d2 in digits.filter (≠ d1)), ∑(d3 in digits.filter (≠ d1 && ≠ d2)),
    ∑(d4 in digits.filter (≠ d1 && ≠ d2 && ≠ d3)), ∑(d5 in digits.filter (≠ d1 && ≠ d2 && ≠ d3 && ≠ d4)),
      id (form_number d1 d2 d3 d4 d5) ) = 2 :=
  by
     let numbers := [
       24568,
       24658,
       45268,
       45628,
       68245,
       68452
     ]
     let favorable := numbers.filter (λ n => n % 4 = 0)
     have h_fav_len : favorable.length = 2 := by sorry
     have h_total_len : numbers.length = 6 := by sorry
     have : (favorable.length / numbers.length) = (1/3) := by sorry
     exact this

end probability_divisible_by_4_l472_472066


namespace final_solution_l472_472948

-- We are stating the conditions for the sequences
variables (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ)

-- Given conditions and definitions based on them:
-- 1. \( 2S_n = 3a_n - 2 \) for \( n \in \mathbb{N}_+ \)
def cond1 := ∀ n, 2 * S n = 3 * a n - 2

-- Derived sequences from the conditions:
-- 2. \( a_n = 2 \times 3^{n-1} \)
def an_formula := ∀ n, a n = 2 * 3^(n-1)

-- 3. \( S_n = 3^n - 1 \)
def Sn_formula := ∀ n, S n = 3^n - 1

-- 4. Sequences \( b_n = \log_3(S_n + 1) \) such that \( T_n \) is sum of first \( n \) terms of \( \{b_{2n}\} \)
def bn_formula := ∀ n, b n = nat.log 3 (S n + 1)
def bn2_formula := ∀ n, b (2 * n) = 2 * n

-- 5. Sum of first \( n \) terms of \( \{b_{2n}\} \)
def Tn_formula := ∀ n, T n = n^2 + n

-- Theorem statement taking the conditions and proving the final results
theorem final_solution (cond1_valid : cond1) :
  (an_formula) ∧ (Sn_formula) ∧ (bn_formula) ∧ (bn2_formula) ∧ (Tn_formula) :=
by sorry

end final_solution_l472_472948


namespace exists_unique_triangle_l472_472879

-- Define the conditions
variables (base : ℝ) (height : ℝ) (angle_diff : ℝ)

-- Define the triangle ABC given the conditions
noncomputable def triangle_ABC (A B C : Point) :=
  side AB = base ∧ height_from_base base height ∧ angle_diff_at_base A B angle_diff

-- State the theorem to prove the existence and uniqueness of triangle ABC
theorem exists_unique_triangle :
  ∃! (A B C : Point), triangle_ABC base height angle_diff A B C :=
sorry

end exists_unique_triangle_l472_472879


namespace determine_polynomial_value_l472_472884

theorem determine_polynomial_value
  {p q r : ℚ}
  (h : ∀ x : ℚ, (x^4 + 5*x^3 + 8*p*x^2 + 6*q*x + r) = (x^3 + 4*x^2 + 16*x + 4) * (x + k) for some k : ℚ) :
  (2*p + q)*r = 76 / 3 := 
  sorry

end determine_polynomial_value_l472_472884


namespace line_through_intersection_of_circles_l472_472576

theorem line_through_intersection_of_circles :
  ∀ (x y : ℝ),
  (x^2 + y^2 = 10 ∧ (x-1)^2 + (y-3)^2 = 10) →
  x + 3 * y - 5 = 0 :=
by
  intros x y h,
  sorry

end line_through_intersection_of_circles_l472_472576


namespace rope_touching_tower_length_l472_472830

-- Definitions according to conditions
def tower_radius : ℝ := 10
def rope_length : ℝ := 30
def attachment_height : ℝ := 6
def horizontal_distance : ℝ := 6

-- The goal is to prove the length of the rope touching the tower
theorem rope_touching_tower_length :
  (let R := tower_radius in
   let L := rope_length in
   let h := attachment_height in
   let x := horizontal_distance in
   let effective_horizontal_length := R + x in
   let y := sqrt (effective_horizontal_length^2 - h^2) in
   let y_half := y / 2 in
   let θ := 2 * real.arccos (R / y_half) in
   R * θ) ≈ 12.28 :=
by
  let R := tower_radius
  let L := rope_length
  let h := attachment_height
  let x := horizontal_distance
  let effective_horizontal_length := R + x
  let y := real.sqrt (effective_horizontal_length^2 - h^2)
  let y_half := y / 2
  let θ := 2 * real.arccos (R / y_half)
  have hR : R = tower_radius := rfl
  have hL : L = rope_length := rfl
  have hh : h = attachment_height := rfl
  have hx : x = horizontal_distance := rfl
  have hex : effective_horizontal_length = R + x := rfl
  have hy : y = real.sqrt (effective_horizontal_length^2 - h^2) := rfl
  have hyh : y_half = y / 2 := rfl
  have hθ : θ = 2 * real.arccos (R / y_half) := rfl
  have := calc
    _ = tower_radius * θ : by rw [hR, hθ]
    _ = 10 * θ : by sorry -- Continue proving the actual computation
  exact sorry -- Skip the proof

end rope_touching_tower_length_l472_472830


namespace marbles_total_l472_472080

-- Conditions
variables (T : ℕ) -- Total number of marbles
variables (h_red : T ≥ 12) -- At least 12 red marbles
variables (h_blue : T ≥ 8) -- At least 8 blue marbles
variables (h_prob : (T - 12 : ℚ) / T = (3 / 4 : ℚ)) -- Probability condition

-- Proof statement
theorem marbles_total : T = 48 :=
by
  -- Proof here
  sorry

end marbles_total_l472_472080


namespace degree_of_polynomial10_l472_472784

-- Definition of the degree function for polynomials.
def degree (p : Polynomial ℝ) : ℕ := p.natDegree

-- Given condition: the degree of the polynomial 5x^3 + 7 is 3.
def polynomial1 := (Polynomial.C 5) * (Polynomial.X ^ 3) + (Polynomial.C 7)
axiom degree_poly1 : degree polynomial1 = 3

-- Statement to prove:
theorem degree_of_polynomial10 : degree (polynomial1 ^ 10) = 30 :=
by
  sorry

end degree_of_polynomial10_l472_472784


namespace sin_series_converges_absolutely_l472_472651

theorem sin_series_converges_absolutely : 
  summable (λ n : ℕ, |sin n / n^2|) :=
sorry

end sin_series_converges_absolutely_l472_472651


namespace max_servings_l472_472037

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l472_472037


namespace avg_annual_decrease_rate_30_predicted_sales_2020_is_68600_l472_472062

-- Define the sales in 2017 and 2019
def initial_sales : ℝ := 200000
def sales_2019 : ℝ := 98000

-- Define the number of years
def years : ℕ := 2

-- Define the average annual decrease rate
def avg_annual_decrease_rate (initial_sales sales_2019 : ℝ) (years : ℕ) : ℝ :=
  1 - (sales_2019 / initial_sales)^(1 / years)

-- Define the predicted sales for 2020
def predicted_sales_2020 (sales_2019 avg_annual_decrease_rate : ℝ) : ℝ :=
  sales_2019 * (1 - avg_annual_decrease_rate)

-- Prove the average annual decrease rate and the predicted sales
theorem avg_annual_decrease_rate_30 :
  avg_annual_decrease_rate initial_sales sales_2019 years = 0.3 :=
sorry

theorem predicted_sales_2020_is_68600 :
  predicted_sales_2020 sales_2019 0.3 = 68600 :=
sorry

end avg_annual_decrease_rate_30_predicted_sales_2020_is_68600_l472_472062


namespace constant_term_f_f_x_l472_472971

def f : ℝ → ℝ := λ x, if x < 0 then (x - 1/x)^8 else -sqrt x

theorem constant_term_f_f_x (x : ℝ) (hx : x > 0) : 
    (let t := (sqrt x - 1/sqrt x)^8 in t.coeff 0 = 70) :=
by
  sorry

end constant_term_f_f_x_l472_472971


namespace find_constant_l472_472371

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem find_constant : ∃ c : ℝ, (∀ x : ℝ, x = 0.4 → (3 * f (x - c)) / f 0 + 4 = f (2 * x + 1)) ∧ c = 2 :=
by
  sorry

end find_constant_l472_472371


namespace geometric_sequence_tan_cos_sin_l472_472998

theorem geometric_sequence_tan_cos_sin (x : ℝ) (h : (cos x ≠ 0) ∧ (sin x ≠ 0) ∧ (cos x * sin x ≠ 0) ∧ (tan x)^2 = cos x * sin x) : 
  (tan x)^6 - (tan x)^2 = - (cos x) * (sin x) :=
by
  sorry

end geometric_sequence_tan_cos_sin_l472_472998


namespace total_oak_trees_after_planting_l472_472387

-- Definitions based on conditions
def initial_oak_trees : ℕ := 5
def new_oak_trees : ℕ := 4

-- Statement of the problem and solution
theorem total_oak_trees_after_planting : initial_oak_trees + new_oak_trees = 9 := by
  sorry

end total_oak_trees_after_planting_l472_472387


namespace sqrt_b_minus_a_l472_472208

theorem sqrt_b_minus_a :
  ∀ (a b : ℝ),
    sqrt (2 * a - 1) = 3 →
    real.cbrt (3 * a + b - 1) = 3 →
    sqrt (b - a) = real.sqrt 8 ∨ sqrt (b - a) = -real.sqrt 8 :=
by
  sorry

end sqrt_b_minus_a_l472_472208


namespace part_one_part_two_l472_472437

def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Question (1)
theorem part_one (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := 
sorry

-- Question (2)
theorem part_two (a : ℝ) (h : a ≥ 1) : ∀ (y : ℝ), (∃ x : ℝ, f x a = y) ↔ (∃ b : ℝ, y = b + 2 ∧ b ≥ a) := 
sorry

end part_one_part_two_l472_472437


namespace eccentricity_of_ellipse_l472_472543

theorem eccentricity_of_ellipse (a b c : ℝ) (e : ℝ) (P : ℝ × ℝ) 
  (h_ellipse : a > b ∧ b > 0 ∧ a^2 = b^2 + c^2)
  (h_conditions : ∀ (F1 F2 : ℝ × ℝ),
    F1 ≠ F2 ∧ 
    |P.1 - F1.1| + |P.2 - F1.2| = 2c ∧ 
    |P.1 - F1.1| = 2c ∧ 
    |P.2 - (c/F1.2)| = b) :
  e = 5 / 7 :=
sorry

end eccentricity_of_ellipse_l472_472543


namespace max_f_value_2006_l472_472922

def f : ℕ → ℕ
| 0       := 0
| (x + 1) := 
    let x_div_10 := Nat.div x 10 in
    let x_mod_1 := x - 10 * x_div_10 in
    f x_div_10 + Nat.log10 (10 / x_mod_1)

theorem max_f_value_2006 : 
  ∃ x, 0 ≤ x ∧ x ≤ 2006 ∧ (∀ y, 0 ≤ y ∧ y ≤ 2006 → f y ≤ f x) ∧ x = 1111 :=
by
  sorry

end max_f_value_2006_l472_472922


namespace tangent_line_at_origin_max_min_on_interval_l472_472567

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x^2 + x + 1

theorem tangent_line_at_origin :
  let m := deriv f 0 in
  let y0 := f 0 in
  m = 1 ∧ y0 = 1 ∧ ∀ x y : ℝ, (y - y0) = m * (x - 0) ↔ x - y + 1 = 0 :=
sorry

theorem max_min_on_interval :
  ∃ (x_max x_min : ℝ), x_min ∈ set.Icc (-2 : ℝ) 0 ∧ x_max ∈ set.Icc (-2 : ℝ) 0 ∧ 
  is_local_min_on f (set.Icc (-2 : ℝ) 0) x_min ∧ f x_max = 1 ∧ f x_min = -1 :=
sorry

end tangent_line_at_origin_max_min_on_interval_l472_472567


namespace cylinder_radius_l472_472840

theorem cylinder_radius
  (r₁ r₂ : ℝ)
  (rounds₁ rounds₂ : ℕ)
  (H₁ : r₁ = 14)
  (H₂ : rounds₁ = 70)
  (H₃ : rounds₂ = 49)
  (L₁ : rounds₁ * 2 * Real.pi * r₁ = rounds₂ * 2 * Real.pi * r₂) :
  r₂ = 20 := 
sorry

end cylinder_radius_l472_472840


namespace circles_intersect_l472_472499

noncomputable def positional_relationship (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : String :=
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  if radius1 + radius2 > d ∧ d > abs (radius1 - radius2) then "Intersecting"
  else if radius1 + radius2 = d then "Externally tangent"
  else if abs (radius1 - radius2) = d then "Internally tangent"
  else "Separate"

theorem circles_intersect :
  positional_relationship (0, 1) (1, 2) 1 2 = "Intersecting" :=
by
  sorry

end circles_intersect_l472_472499


namespace sphere_radius_that_touches_plane_and_three_given_spheres_l472_472001

-- Definitions and conditions from problem a
def Sphere (R : ℝ) := {center : ℝ × ℝ × ℝ // sqrt (center.1^2 + center.2^2 + center.3^2) = R}

def TangentToPlane (s : Sphere R) := s.val.3 = R

def TangentToEachOther (s₁ s₂ s₃ : Sphere R) : Prop :=
  dist s₁.val s₂.val = 2 * R ∧
  dist s₂.val s₃.val = 2 * R ∧
  dist s₃.val s₁.val = 2 * R

-- The statement to be proved
theorem sphere_radius_that_touches_plane_and_three_given_spheres
  (s₁ s₂ s₃ : Sphere R)
  (h₁ : TangentToPlane s₁)
  (h₂ : TangentToPlane s₂)
  (h₃ : TangentToPlane s₃)
  (h₄ : TangentToEachOther s₁ s₂ s₃) :
  let r := (1 / 3) * R in
  ∃ s' : Sphere r, TangentToPlane s' ∧
  dist s'.val s₁.val = R + r ∧
  dist s'.val s₂.val = R + r ∧
  dist s'.val s₃.val = R + r :=
by
  sorry

end sphere_radius_that_touches_plane_and_three_given_spheres_l472_472001


namespace plate_and_rollers_acceleration_l472_472048

-- Definitions for conditions
def roller_radii := (1 : ℝ, 0.4 : ℝ)
def plate_mass := 150 -- kg
def inclination_angle := Real.arccos 0.68
def gravity_acceleration := 10 -- m/s^2

-- Theorem statement for the problem
theorem plate_and_rollers_acceleration :
  let R := roller_radii.1,
      r := roller_radii.2,
      m := plate_mass,
      α := inclination_angle,
      g := gravity_acceleration in
  ∃ (a_plate a_rollers : ℝ), a_plate = a_rollers ∧ a_plate = 4 :=
  sorry

end plate_and_rollers_acceleration_l472_472048


namespace original_radius_l472_472493

noncomputable def volume_change_radius (r : ℝ) : ℝ :=
  3 * (Mathlib.pi * (r - 4)^2) - 3 * (Mathlib.pi * r^2)

noncomputable def volume_change_height (r : ℝ) : ℝ :=
  Mathlib.pi * r^2 - 3 * (Mathlib.pi * r^2)

noncomputable def quadratic (r : ℝ) : ℝ :=
  r^2 - 12 * r + 24

theorem original_radius (r : ℝ) :
  volume_change_radius r = x ∧ volume_change_height r = x ∧ 3 > 0 →
  r = 6 + 2 * Real.sqrt 3 ∨ r = 6 - 2 * Real.sqrt 3 :=
by
  sorry

end original_radius_l472_472493


namespace number_of_values_m_n_k_l472_472956

theorem number_of_values_m_n_k :
  ∀ (m n k : ℕ) (h_m : 0 < m) (h_n : 0 < n) (h_k : 0 < k),
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (1 + a) * n^2 - 4 * (m + a) * n + 4 * m^2 + 4 * a + b * (k - 1)^2 < 3) →
  ↔ 4 :=
sorry

end number_of_values_m_n_k_l472_472956


namespace angles_satisfying_cotg_eq_l472_472055

theorem angles_satisfying_cotg_eq :
  ∃ k : ℤ, (x = 48 + k * 180) ∨ (x = -48 + k * 180) → 
  (cot x ^ 2 - sin x ^ 2 = 1/4) :=
begin
  sorry
end

end angles_satisfying_cotg_eq_l472_472055


namespace problem1_problem2_problem3_problem4_l472_472870

-- Problem 1
theorem problem1 : (-3 / 8) + ((-5 / 8) * (-6)) = 27 / 8 :=
by sorry

-- Problem 2
theorem problem2 : 12 + (7 * (-3)) - (18 / (-3)) = -3 :=
by sorry

-- Problem 3
theorem problem3 : -((2:ℤ)^2) - (4 / 7) * (2:ℚ) - (-((3:ℤ)^2:ℤ) : ℤ) = -99 / 7 :=
by sorry

-- Problem 4
theorem problem4 : -(((-1) ^ 2020 : ℤ)) + ((6 : ℚ) / (-(2 : ℤ) ^ 3)) * (-1 / 3) = -3 / 4 :=
by sorry

end problem1_problem2_problem3_problem4_l472_472870


namespace sale_in_second_month_l472_472445

theorem sale_in_second_month :
  ∀ (sales_1 sales_3 sales_4 sales_5 sales_6 avg_sale num_months total_sales sales_2 : ℕ),
  avg_sale = 5600 →
  num_months = 6 →
  sales_1 = 5266 →
  sales_3 = 5678 →
  sales_4 = 6029 →
  sales_5 = 4937 →
  sales_6 = 4937 →
  total_sales = avg_sale * num_months →
  sales_2 = total_sales - (sales_1 + sales_3 + sales_4 + sales_5 + sales_6) →
  sales_2 = 11690 :=
by
  intros sales_1 sales_3 sales_4 sales_5 sales_6 avg_sale num_months total_sales sales_2
  intro h_avg_sale h_num_months h_sales_1 h_sales_3 h_sales_4 h_sales_5 h_sales_6 h_total_sales h_sales_2
  rw [h_avg_sale, h_num_months, h_sales_1, h_sales_3, h_sales_4, h_sales_5, h_sales_6] at h_total_sales h_sales_2
  exact h_sales_2

end sale_in_second_month_l472_472445


namespace melissa_coupe_sale_l472_472691

theorem melissa_coupe_sale :
  ∃ x : ℝ, (0.02 * x + 0.02 * 2 * x = 1800) ∧ x = 30000 :=
by
  sorry

end melissa_coupe_sale_l472_472691


namespace compare_y1_y2_l472_472196

theorem compare_y1_y2 : 
  let line := λ x : ℝ, - (1 / 2) * x + 2 in
  let y1 := line (-4) in
  let y2 := line 2 in
  y1 > y2 :=
by 
  -- definitions of y1 and y2 based on the line equation
  let y1 := - (1 / 2) * (-4) + 2
  let y2 := - (1 / 2) * 2 + 2
  sorry

end compare_y1_y2_l472_472196


namespace max_value_frac_x1_x2_et_l472_472944

theorem max_value_frac_x1_x2_et (f g : ℝ → ℝ)
  (hf : ∀ x, f x = x * Real.exp x)
  (hg : ∀ x, g x = - (Real.log x) / x)
  (x1 x2 t : ℝ)
  (hx1 : f x1 = t)
  (hx2 : g x2 = t)
  (ht_pos : t > 0) :
  ∃ x1 x2, (f x1 = t ∧ g x2 = t) ∧ (∀ u v, (f u = t ∧ g v = t → u / (v * Real.exp t) ≤ 1 / Real.exp 1)) :=
by
  sorry

end max_value_frac_x1_x2_et_l472_472944


namespace inscribed_quadrilateral_sides_l472_472345

theorem inscribed_quadrilateral_sides (A B C D : ℝ) (O : ℝ) (r : ℝ) (h_convex: convex_quadrilateral A B C D) (h_unit_circle: inscribed_in_unit_circle A B C D O r) :
  ∀ (x y : ℝ), side_length x y ≤ Real.sqrt 2 :=
  sorry

end inscribed_quadrilateral_sides_l472_472345


namespace bus_problem_l472_472627

-- Define the participants in 2005
def participants_2005 (k : ℕ) : ℕ := 27 * k + 19

-- Define the participants in 2006
def participants_2006 (k : ℕ) : ℕ := participants_2005 k + 53

-- Define the total number of buses needed in 2006
def buses_needed_2006 (k : ℕ) : ℕ := (participants_2006 k) / 27 + if (participants_2006 k) % 27 = 0 then 0 else 1

-- Define the total number of buses needed in 2005
def buses_needed_2005 (k : ℕ) : ℕ := k + 1

-- Define the additional buses needed in 2006 compared to 2005
def additional_buses_2006 (k : ℕ) := buses_needed_2006 k - buses_needed_2005 k

-- Define the number of people in the incomplete bus in 2006
def people_in_incomplete_bus_2006 (k : ℕ) := (participants_2006 k) % 27

-- The proof statement to be proved
theorem bus_problem (k : ℕ) : additional_buses_2006 k = 2 ∧ people_in_incomplete_bus_2006 k = 9 := by
  sorry

end bus_problem_l472_472627


namespace trajectory_of_M_l472_472363

theorem trajectory_of_M (x y : ℝ) :
  (∃ α : ℝ, ∡((2, 0), (x, y), (-1, 0)) = 2 * ∡((x, y), (-1, 0), (2, 0))) →
  ((3 * x^2 - y^2 = 3 ∧ x ≥ 1) ∨ (y = 0 ∧ -1 < x ∧ x < 2)) :=
by
  sorry

end trajectory_of_M_l472_472363


namespace cos_54_deg_l472_472142

-- Define cosine function
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

-- The main theorem statement
theorem cos_54_deg : cos_deg 54 = (-1 + Real.sqrt 5) / 4 :=
  sorry

end cos_54_deg_l472_472142


namespace max_servings_l472_472016

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l472_472016


namespace billiard_ball_trajectory_passes_through_circumcenter_l472_472431

theorem billiard_ball_trajectory_passes_through_circumcenter
  (A B C : Point)
  (ABC_is_acute_angled : acute_triangle A B C)
  (angle_A_eq_60_deg : ∠ BAC = 60)
  (bisects_angle : bisector A (∠ BAC))
  (reflects_off_BC : reflects_off_side A BC bisector)
  : passes_through_circumcenter (trajectory A BC bisector) := 
sorry

end billiard_ball_trajectory_passes_through_circumcenter_l472_472431


namespace part1_answer1_part1_answer2_part2_answer1_part2_answer2_l472_472665

open Set

def A : Set ℕ := {x | 1 ≤ x ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem part1_answer1 : A ∩ C = {3, 4, 5, 6, 7} :=
by
  sorry

theorem part1_answer2 : A \ B = {5, 6, 7, 8, 9, 10} :=
by
  sorry

theorem part2_answer1 : A \ (B ∪ C) = {8, 9, 10} :=
by 
  sorry

theorem part2_answer2 : A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} :=
by 
  sorry

end part1_answer1_part1_answer2_part2_answer1_part2_answer2_l472_472665


namespace num_divisors_not_divisible_by_3_l472_472601

-- Define the prime factorization of 180
def prime_factorization_180 : Nat → Nat :=
  λ n, if n = 2 then 2 else if n = 3 then 2 else if n = 5 then 1 else 0

-- Define the conditions for valid exponents a, b, c
def valid_exponent_a (a : Nat) : Prop := 0 ≤ a ∧ a ≤ 2
def valid_exponent_b (b : Nat) : Prop := b = 0
def valid_exponent_c (c : Nat) : Prop := 0 ≤ c ∧ c ≤ 1

-- Define the problem statement in Lean
theorem num_divisors_not_divisible_by_3 : 
  (∑ a in Finset.range 3, ∑ c in Finset.range 2, 
     if valid_exponent_a a ∧ valid_exponent_b 0 ∧ valid_exponent_c c 
     then 1 else 0) = 6 := 
  by
    sorry

end num_divisors_not_divisible_by_3_l472_472601


namespace problem_proof_l472_472421

theorem problem_proof :
  1.25 * 67.875 + 125 * 6.7875 + 1250 * 0.053375 = 1000 :=
by
  sorry

end problem_proof_l472_472421


namespace correct_solution_l472_472800

theorem correct_solution :
  ∀ x : ℝ, x = -8 ↔ (∃ a : ℝ, a = 2 / 3 ∧ ∀ x : ℝ, (2 * (2x - 1) = 3 * (x + a) - 2 * 6) → (x = 2)) → 
  (2x - 1) / 3 = (x + 2 / 3) / 2 - 2 :=
by
  sorry

end correct_solution_l472_472800


namespace num_divisors_not_divisible_by_3_l472_472602

-- Define the prime factorization of 180
def prime_factorization_180 : Nat → Nat :=
  λ n, if n = 2 then 2 else if n = 3 then 2 else if n = 5 then 1 else 0

-- Define the conditions for valid exponents a, b, c
def valid_exponent_a (a : Nat) : Prop := 0 ≤ a ∧ a ≤ 2
def valid_exponent_b (b : Nat) : Prop := b = 0
def valid_exponent_c (c : Nat) : Prop := 0 ≤ c ∧ c ≤ 1

-- Define the problem statement in Lean
theorem num_divisors_not_divisible_by_3 : 
  (∑ a in Finset.range 3, ∑ c in Finset.range 2, 
     if valid_exponent_a a ∧ valid_exponent_b 0 ∧ valid_exponent_c c 
     then 1 else 0) = 6 := 
  by
    sorry

end num_divisors_not_divisible_by_3_l472_472602


namespace find_y_l472_472215

theorem find_y (x y : ℕ) (h1 : x = 2407) (h2 : x^y + y^x = 2408) : y = 1 :=
sorry

end find_y_l472_472215


namespace problem_A_l472_472436

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem problem_A (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_mono : is_monotonically_increasing_on_nonneg f) :
  f (-2) > f (1) :=
begin
  -- concise form:
  have h1 : f (-2) = f (2), from h_even (-2),
  have h2 : f (2) > f (1), from h_mono 1 2 (by linarith) (by linarith),
  rw h1,
  exact h2,
end

end problem_A_l472_472436


namespace general_term_formula_l472_472644

noncomputable def a_sequence (n : ℕ) : ℚ :=
  if n = 0 then 0  -- as a0 is not defined in the problem, let's define it as 0 to avoid errors
  else
    match n with
    | 1       => 1
    | _ + 2 => sorry  -- This matches our recurrence relation and leaves the proof open for definition

theorem general_term_formula (a : ℕ → ℚ) (n : ℕ) :
  (a 1 = 1) ∧ (∀ n ∈ ℕ, (n^2 + 2*n) * ((a (n+1)) - (a n)) = 1) → 
  a n = (7/4 : ℚ) - ((2*n + 1 : ℚ) / (2*n*(n+1))) :=
begin
  intros h,
  cases h with h_a1 h_recurrence,
  sorry
end

end general_term_formula_l472_472644


namespace cos_of_sin_given_l472_472950

theorem cos_of_sin_given (θ : ℝ) (h : Real.sin (88 * Real.pi / 180 + θ) = 2 / 3) :
  Real.cos (178 * Real.pi / 180 + θ) = - (2 / 3) :=
by
  sorry

end cos_of_sin_given_l472_472950


namespace binomial_expansion_properties_l472_472206

theorem binomial_expansion_properties :
  ∀ (x : ℝ), 
  (∃ n : ℕ, n = 7 ∧ (∀ i, 0 ≤ i ≤ n → coefficients_maximized (3*x - 1) n 4 5)
  → (let a : ℕ → ℝ := λ i, binomial_coeff (3*x - 1) n i in
      (∃ (a_0 a_1 ... a_n : ℝ), sum_abs_coefficients_sum (a) = 4^7 - 1 ∧ 
      sum_abs_weighted_coefficients_sum (a) = 21 * 4^6 ∧ 
      ∃ k, largest_abs_coefficient (a k) = 5103))) := 
begin
  intro x,
  intros n h,
  cases h with h_n h_coeffs,
  -- Proof would go here, but we are just writing the statement
  sorry,
end

end binomial_expansion_properties_l472_472206


namespace eccentricity_of_hyperbola_l472_472223

-- Define the conditions of the hyperbola and parabola
def hyperbola (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1
def parabola : Prop := ∀ x y : ℝ, y^2 = 4 * x

-- Define the conditions for the points A and B and the area of triangle AOB
def area_triangle_AOB (a b : ℝ) : Prop := ∃ A B : (ℝ × ℝ), 
  A.1 = -1 ∧ B.1 = -1 ∧ (1 / 2) * 1 * (2 * b / a) = 2 * real.sqrt 3

-- Define the eccentricity of the hyperbola
def eccentricity (a b e : ℝ) : Prop := a > 0 ∧ b > 0 ∧ e = real.sqrt ((a^2 + b^2) / a^2)

-- Main statement: prove the eccentricity given the conditions
theorem eccentricity_of_hyperbola (a b : ℝ) (h_hyperbola : hyperbola a b) (h_parabola : parabola) 
  (h_area : area_triangle_AOB a b) : eccentricity a b (real.sqrt 13) :=
begin
  sorry
end

end eccentricity_of_hyperbola_l472_472223


namespace number_of_squares_l472_472667

open Classical

theorem number_of_squares (A B C : Type) [is_isosceles_triangle A B C]
  (h1 : dist A B = dist A C) (h2 : dist A B > dist B C) : 
  number_of_squares_sharing_vertices A B C = 4 :=
sorry

end number_of_squares_l472_472667


namespace median_group_l472_472825

theorem median_group (group1_freq group2_freq group3_freq group4_freq group5_freq : ℕ)
  (h_total : group1_freq = 12 ∧ group2_freq = 24 ∧ group3_freq = 18 ∧ group4_freq = 10 ∧ group5_freq = 6) :
  let total : ℕ := group1_freq + group2_freq + group3_freq + group4_freq + group5_freq in
  total = 70 →
  12 + 24 ≥ 35 ∧ 12 < 35 :=
by
  intro h_total_sum
  let h_group1_cum := 12
  let h_group2_cum := 12 + 24
  have h_median_in_group2 : 12 < 35 ∧ 35 ≤ 12 + 24, from sorry
  exact h_median_in_group2

end median_group_l472_472825


namespace area_of_AEF_l472_472118

variables {A B C D E F : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variables (AD CF : Line)
variables (area : Triangle → ℝ)
variables [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E] [DecidableEq F]

-- conditions
def is_intersection_point (P : Point) (l1 l2 : Line) : Prop := l1.contains P ∧ l2.contains P
def is_midpoint (P : Point) (A B : Point) : Prop := dist A P = dist B P

-- problem statement
theorem area_of_AEF
  (h1 : is_intersection_point E AD CF) 
  (h2 : is_midpoint E A D) 
  (h3 : area (△ A B C) = 1) 
  (h4 : area (△ B E F) = 1 / 10) : 
  area (△ A E F) = 1 / 15 := 
sorry

end area_of_AEF_l472_472118


namespace proof_least_area_l472_472839

def min_area_rectangle (x y : ℕ) (h1 : 2 * x + 2 * y = 200) : ℕ :=
  x * y

theorem proof_least_area :
  ∃ (x y : ℕ), 2 * x + 2 * y = 200 ∧ x * y = 99 :=
by
  use 1, 99
  split
  { simp }
  { sorry }

end proof_least_area_l472_472839


namespace place_coins_ways_l472_472856

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem place_coins_ways : 
  let num_coins := 5 in 
  let ways_per_row := factorial num_coins in
  let ways_per_col := factorial num_coins in
  ways_per_row * ways_per_col = 14400 :=
by
  let num_coins := 5
  let ways_per_row := factorial num_coins
  let ways_per_col := factorial num_coins
  show ways_per_row * ways_per_col = 14400
  sorry

end place_coins_ways_l472_472856


namespace degree_of_poly_l472_472789

-- Define the polynomial and its degree
def inner_poly := (5 : ℝ) * (X ^ 3) + (7 : ℝ)
def poly := inner_poly ^ 10

-- Statement to prove
theorem degree_of_poly : polynomial.degree poly = 30 :=
sorry

end degree_of_poly_l472_472789


namespace log2_final_value_l472_472728

-- Define the function f
def f (a b : ℕ) : ℕ := (a * b - 1) / (a + b + 2)

-- State the conditions and the goal as a theorem
theorem log2_final_value:
  ∀ (p q : ℕ), (p = 1) ∧ (q = 2^100 - 1) → log (p + q) / log 2 = 100 :=
begin
  intros p q h,
  sorry -- Proof goes here
end

end log2_final_value_l472_472728


namespace necessary_AB_neq_BC_l472_472650

variables {A B C O L K : Type}
variables [is_point A] [is_point B] [is_point C] [is_point O] [is_point L] [is_point K]
variables [triangle A B C] [segment AO CO] [angle_eq (angle O A C) (angle O C A)]
variables [extension (AO) (BC) (L)] [extension (CO) (AB) (K)]
variables [segment_len_eq (AK) (CL)]

theorem necessary_AB_neq_BC : ¬ (AB = BC) :=
by
  sorry

end necessary_AB_neq_BC_l472_472650


namespace angle_EFC_eq_angle_GFD_l472_472939

open Real

variable {A B C : Point}
variable {D E F G : Point}
variable [Triangle ABC]
variable [Incircle ABC D E F]
variable (G : Point) [Midpoint G D E]

theorem angle_EFC_eq_angle_GFD :
  ∠ EFC  = ∠ GFD :=
sorry

end angle_EFC_eq_angle_GFD_l472_472939


namespace find_n_from_binomial_term_l472_472960

noncomputable def binomial_coefficient (n r : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem find_n_from_binomial_term :
  (∃ n : ℕ, 3^2 * binomial_coefficient n 2 = 54) ↔ n = 4 :=
by
  sorry

end find_n_from_binomial_term_l472_472960


namespace find_m_l472_472584

theorem find_m (m : ℤ) (a := (3, m)) (b := (1, -2)) (h : a.1 * b.1 + a.2 * b.2 = b.1^2 + b.2^2) : m = -1 :=
sorry

end find_m_l472_472584


namespace max_servings_possible_l472_472019

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l472_472019


namespace num_valid_four_digit_numbers_l472_472586

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def is_non_prime_odd (n : ℕ) : Prop := n = 1 ∨ n = 9
def all_unique_digits (d1 d2 d3 d4 : ℕ) : Prop := d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

theorem num_valid_four_digit_numbers : 
  ∃ n : ℕ, n = 448 ∧
    ∃ d1 d2 d3 d4 : ℕ, 
      1 ≤ d1 ∧ d1 ≤ 9 ∧ is_prime d1 ∧ 
      0 ≤ d2 ∧ d2 ≤ 9 ∧ is_non_prime_odd d2 ∧ 
      0 ≤ d3 ∧ d3 ≤ 9 ∧ 
      0 ≤ d4 ∧ d4 ≤ 9 ∧ 
      all_unique_digits d1 d2 d3 d4 := 
begin
  sorry,
end

end num_valid_four_digit_numbers_l472_472586


namespace composition_of_even_number_of_central_symmetries_is_translation_composition_of_odd_number_of_central_symmetries_is_central_symmetry_l472_472799

-- Definitions of central symmetry and translations
def central_symmetry (O : Point) (P: Point) : Point := sorry -- placeholder definition

def translation (v : Vector) (P: Point) : Point := sorry -- placeholder definition

-- Definitions related to the nature of composition of symmetries
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the transformation results
def transformation (symmetries : ℕ → Point → Point) (n : ℕ) : Point → Point :=
  if is_even n then translation sorry else central_symmetry sorry

theorem composition_of_even_number_of_central_symmetries_is_translation 
  (symmetries : ℕ → Point → Point) (n : ℕ) (h : is_even n) :
  ∃ v, transformation symmetries n = translation v :=
sorry

theorem composition_of_odd_number_of_central_symmetries_is_central_symmetry 
  (symmetries : ℕ → Point → Point) (n : ℕ) (h : is_odd n) :
  ∃ O, transformation symmetries n = central_symmetry O :=
sorry

end composition_of_even_number_of_central_symmetries_is_translation_composition_of_odd_number_of_central_symmetries_is_central_symmetry_l472_472799


namespace length_of_BC_segment_l472_472228

noncomputable def cos_func (x : ℝ) : ℝ := Real.cos x
noncomputable def sin_func (x : ℝ) : ℝ := Real.sin x
noncomputable def g_func (x : ℝ) : ℝ := Real.sqrt 3 * sin_func x

theorem length_of_BC_segment :
  ∃ A B C : ℝ × ℝ,
    (cos_func A.1 = A.2 ∧ g_func A.1 = A.2 ∧ A.1 ∈ Ioo 0 (Real.pi / 2)) ∧
    (B.2 = 0 ∧ (by sorry)) ∧
    (C.2 = 0 ∧ (by sorry)) ∧
    |B.1 - C.1| = (4 * Real.sqrt 3) / 3 :=
sorry  -- Proof to be filled

end length_of_BC_segment_l472_472228


namespace plate_and_roller_acceleration_l472_472045

noncomputable def m : ℝ := 150
noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def alpha : ℝ := Real.arccos 0.68

theorem plate_and_roller_acceleration :
  let sin_alpha_half := Real.sin (alpha / 2)
  sin_alpha_half = 0.4 →
  plate_acceleration == 4 ∧ direction == Real.arcsin 0.4 ∧ rollers_acceleration == 4 :=
by
  sorry

end plate_and_roller_acceleration_l472_472045


namespace find_m_l472_472989

-- Definitions of the given vectors and their properties
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Condition that vectors a and b are parallel
def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 = 0

-- Goal: Find the value of m such that vectors a and b are parallel
theorem find_m (m : ℝ) : 
  are_parallel a (b m) → m = 6 :=
by
  sorry

end find_m_l472_472989


namespace smallest_lcm_of_quadruplets_l472_472760

theorem smallest_lcm_of_quadruplets (a b c d n : ℕ) (h_gcd : Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 60)
  (h_count : ∃ F : Finset (ℕ × ℕ × ℕ × ℕ), F.card = 60000 ∧
    ∀ x ∈ F, Nat.gcd (Nat.gcd (Nat.gcd x.1.1 x.1.2) x.2.1) x.2.2 = 60 ∧ 
    Nat.lcm (Nat.lcm (Nat.lcm x.1.1 x.1.2) x.2.1) x.2.2 = n):
  n = 6480 := 
sorry

end smallest_lcm_of_quadruplets_l472_472760


namespace other_divisors_of_y_l472_472093

theorem other_divisors_of_y (y : ℕ) (h₀ : y = 20) (h₁ : y % 5 = 0) (h₂ : ¬ (y % 8 = 0)) :
  ∃ z ∈ {2, 4, 10}, y % z = 0 :=
by {
  sorry
}

end other_divisors_of_y_l472_472093


namespace dot_product_equilateral_l472_472192

-- Define the conditions for the equilateral triangle ABC
variable {A B C : ℝ}

noncomputable def equilateral_triangle (A B C : ℝ) := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ |A - B| = 1 ∧ |B - C| = 1 ∧ |C - A| = 1

-- Define the dot product of the vectors AB and BC
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_equilateral (A B C : ℝ) (h : equilateral_triangle A B C) : 
  dot_product (B - A, 0) (C - B, 0) = -1 / 2 :=
sorry

end dot_product_equilateral_l472_472192


namespace trapezoid_perimeter_l472_472105

theorem trapezoid_perimeter (AB CD AD BC h : ℝ)
  (AB_eq : AB = 40)
  (CD_eq : CD = 70)
  (AD_eq_BC : AD = BC)
  (h_eq : h = 24)
  : AB + BC + CD + AD = 110 + 2 * Real.sqrt 801 :=
by
  -- Proof goes here, you can replace this comment with actual proof.
  sorry

end trapezoid_perimeter_l472_472105


namespace determine_sum_of_squares_l472_472888

theorem determine_sum_of_squares
  (x y z : ℝ)
  (h1 : x + y + z = 13)
  (h2 : x * y * z = 72)
  (h3 : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := 
sorry

end determine_sum_of_squares_l472_472888


namespace problem_proof_l472_472255

variables {A B C : ℝ} {a b c : ℝ}

-- Define the conditions of the problem
def conditions (A B C a b c : ℝ) :=
  A = Real.pi / 4 ∧ 
  b^2 - a^2 = 1/2 * c^2 ∧ 
  (1/2 * a * b * Real.sin C = 3)

-- Define the assertions we want to prove
def assertions (A B C a b c : ℝ) :=
  Real.tan C = 2 ∧ 
  b = 3 ∧ 
  (circumcircle_circumference a b c = Real.sqrt 10 * Real.pi)

-- Define the function that computes the circumference of the circumcircle
noncomputable def circumcircle_circumference (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  2 * Real.arcsin ( (s * (s - a) * (s - b) * (s - c)).sqrt / (a * b * c) )

-- State the theorem to be proved
theorem problem_proof (A B C a b c : ℝ) : 
  conditions A B C a b c → 
  assertions A B C a b c :=
by 
  intros h
  sorry

end problem_proof_l472_472255


namespace total_screens_sold_l472_472863

variable (J F M : ℕ)
variable (feb_eq_fourth_of_march : F = M / 4)
variable (feb_eq_double_of_jan : F = 2 * J)
variable (march_sales : M = 8800)

theorem total_screens_sold (J F M : ℕ)
  (feb_eq_fourth_of_march : F = M / 4)
  (feb_eq_double_of_jan : F = 2 * J)
  (march_sales : M = 8800) :
  J + F + M = 12100 :=
by
  sorry

end total_screens_sold_l472_472863


namespace negation_of_universal_proposition_l472_472735

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 3 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2 * x + 3 < 0 := 
sorry

end negation_of_universal_proposition_l472_472735


namespace exist_similar_ngons_of_same_color_l472_472130

theorem exist_similar_ngons_of_same_color 
  (color : ℝ² → bool) 
  (n : ℕ) 
  (k : ℝ) 
  (hn : n ≥ 3) 
  (hk : k > 0 ∧ k ≠ 1)
  (coloring : ∀ P: ℝ², color P = true ∨ color P = false) :
  ∃ (p₁ p₂ : fin n → ℝ²),
    (∀ i j : fin n, (p₁ i = p₁ j ↔ p₂ i = p₂ j)) ∧
    (∀ i : fin n, color (p₁ i) = color (p₁ 0)) ∧
    (∀ i : fin n, color (p₂ i) = color (p₂ 0)) ∧
    (∀ i j : fin n, dist (p₁ i) (p₁ j) = k * dist (p₂ i) (p₂ j)) :=
sorry

end exist_similar_ngons_of_same_color_l472_472130


namespace cartesian_equation_l_distances_from_P_to_l_l472_472269

-- Define the polar equation of line l
def polar_equation_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + π / 4) = 2 * Real.sqrt 2

-- Define point P on curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (sqrt 3 * Real.cos θ, Real.sin θ)

-- (I) Convert the polar equation to Cartesian coordinate equation
theorem cartesian_equation_l (ρ θ : ℝ) : 
  polar_equation_l ρ θ -> (ρ * Real.cos θ + ρ * Real.sin θ - 4 = 0) := 
sorry

-- (II) Maximum and minimum distances from point P to line l
theorem distances_from_P_to_l (θ : ℝ) : 
  let P := curve_C θ in 
  let d := (|P.1 + P.2 - 1| / Real.sqrt 2) in 
  ∃ d_max d_min, 
    (d_max = 3 * Real.sqrt 2 / 2) ∧ 
    (d_min = Real.sqrt 2 / 2) := 
sorry

end cartesian_equation_l_distances_from_P_to_l_l472_472269


namespace train_lengths_equal_l472_472069

theorem train_lengths_equal (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ)  
  (h1 : v_fast = 46) 
  (h2 : v_slow = 36) 
  (h3 : t = 36.00001) : 
  2 * L = (v_fast - v_slow) / 3600 * t → L = 1800.0005 := 
by
  sorry

end train_lengths_equal_l472_472069


namespace problem1_problem2_l472_472263

-- Definitions for the conditions
variables (A B C a b c : ℝ)
variables (sinA sinB sinC : ℝ)
variable (area : ℝ)

-- Conditions
def acute_triangle := ∀ {A B C : ℝ}, A < π / 2 ∧ B < π / 2 ∧ C < π / 2 ∧ A + B + C = π
def sin_A := sinA = 3 * sqrt 10 / 10
def main_equation := a * sinA + b * sinB = c * sinC + 2 * sqrt 5 / 5 * a * sinB

-- Proof that B = π / 4 given conditions
theorem problem1 (h1 : acute_triangle) (h2 : sin_A) (h3 : main_equation) : B = π / 4 :=
sorry

-- Further condition for second problem
def b_value := b = sqrt 5

-- Proof that the area is 3 given conditions and b = sqrt 5
theorem problem2 (h1 : acute_triangle) (h2 : sin_A) (h3 : main_equation) (h4 : b_value) : area = 3 :=
sorry

end problem1_problem2_l472_472263


namespace distance_problem_l472_472170

noncomputable def distance_from_point_to_line (p : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) : ℝ :=
  let direction_vector := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let t := (direction_vector.1 * (p.1 - a.1) + direction_vector.2 * (p.2 - a.2) + direction_vector.3 * (p.3 - a.3)) /
           (direction_vector.1^2 + direction_vector.2^2 + direction_vector.3^2)
  let closest_point := (a.1 + t * direction_vector.1, a.2 + t * direction_vector.2, a.3 + t * direction_vector.3)
  Real.sqrt ((p.1 - closest_point.1)^2 + (p.2 - closest_point.2)^2 + (p.3 - closest_point.3)^2)

theorem distance_problem : distance_from_point_to_line (2, 1, -1) (1, -2, 4) (3, 1, 6) = (Real.sqrt 10394) / 17 :=
by
  sorry

end distance_problem_l472_472170


namespace min_connections_for_fault_tolerant_network_l472_472797

theorem min_connections_for_fault_tolerant_network :
  ∀ (G : SimpleGraph (Fin 10)), (∀ (u v : Fin 10), u ≠ v → (G.degree u ≥ 3)) ∧ (∀ v w : Fin 10, ∃ p : G.Path v w, v ≠ w)
  → G.edge_card := 15 :=
sorry

end min_connections_for_fault_tolerant_network_l472_472797


namespace total_green_and_yellow_peaches_in_basket_l472_472388

def num_red_peaches := 5
def num_yellow_peaches := 14
def num_green_peaches := 6

theorem total_green_and_yellow_peaches_in_basket :
  num_yellow_peaches + num_green_peaches = 20 :=
by
  sorry

end total_green_and_yellow_peaches_in_basket_l472_472388


namespace geometric_sequence_ratio_l472_472623

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def condition_1 (n : ℕ) : Prop := S n = 2 * a n - 2

theorem geometric_sequence_ratio (h : ∀ n, condition_1 a S n) : (a 8 / a 6 = 4) :=
sorry

end geometric_sequence_ratio_l472_472623


namespace new_container_volume_l472_472091

-- Define the original volume of the container 
def original_volume : ℝ := 4

-- Define the scale factor of each dimension (quadrupled)
def scale_factor : ℝ := 4

-- Define the new volume, which is original volume * (scale factor ^ 3)
def new_volume (orig_vol : ℝ) (scale : ℝ) : ℝ := orig_vol * (scale ^ 3)

-- The theorem we want to prove
theorem new_container_volume : new_volume original_volume scale_factor = 256 :=
by
  sorry

end new_container_volume_l472_472091


namespace polynomial_degree_l472_472791

def polynomial := 5 * X ^ 3 + 7
def exponent := 10
def degree_of_polynomial := 3
def final_degree := 30

theorem polynomial_degree : degree (polynomial ^ exponent) = final_degree :=
by
  sorry

end polynomial_degree_l472_472791


namespace median_free_throws_is_15_l472_472821

def basketball_free_throws : List ℕ := [6, 18, 15, 14, 19, 12, 19, 15, 22, 11]

theorem median_free_throws_is_15
  (sorted_free_throws : List ℕ := basketball_free_throws
    |> List.qsort (λ a b => a ≤ b)) :
  (sorted_free_throws.nth 4 + sorted_free_throws.nth 5) / 2 = 15 := by
  sorry

end median_free_throws_is_15_l472_472821


namespace distinct_odd_numbers_between_100_and_999_with_even_hundreds_l472_472587

theorem distinct_odd_numbers_between_100_and_999_with_even_hundreds :
  (finstinct_count_odd_numbers : { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ odd n 
    ∧ ∀ (d1 d2 d3 : ℕ), (n = d1 * 100 + d2 * 10 + d3 → d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3 ∧ even d1) } 
    = 160) :=
by
  sorry

end distinct_odd_numbers_between_100_and_999_with_even_hundreds_l472_472587


namespace trapezoid_height_l472_472464

-- Given conditions
def base1 : ℝ := 3
def base2 : ℝ := 5
def area : ℝ := 21

-- Prove that the height is 5.25 feet
theorem trapezoid_height : 
  ∃ height : ℝ, (1 / 2) * (base1 + base2) * height = area ∧ height = 5.25 := 
by {
  use 5.25,
  split,
  { sorry },
  { refl }
}

end trapezoid_height_l472_472464


namespace maximal_sum_of_million_numbers_l472_472376

theorem maximal_sum_of_million_numbers (a : Fin 1000000 → ℕ) (h_prod : (∏ i, a i) = 1000000) :
  (∑ i, a i) ≤ 1999999 :=
sorry

end maximal_sum_of_million_numbers_l472_472376


namespace coefficient_x21_l472_472272

theorem coefficient_x21 : 
  let f := (λ x : ℕ, 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^(10) + x^(11) + x^(12) + x^(13) + x^(14) + x^(15) + x^(16) + x^(17) + x^(18) + x^(19) + x^(20))
  let g := (λ x : ℕ, (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^(10)))
  let h := (λ x : ℕ, 1 + x^2 + x^4 + x^6 + x^8 + x^(10) + x^(12) + x^(14) + x^(16) + x^(18) + x^(20))
  let product := (f x) * (g x)^2 * (h x)
  polynomial.coeff product 21 = 385
:= sorry

end coefficient_x21_l472_472272


namespace largest_angle_l472_472635

theorem largest_angle (y : ℝ) (h : 40 + 70 + y = 180) : y = 70 :=
by
  sorry

end largest_angle_l472_472635


namespace div_remainder_l472_472835

theorem div_remainder (B x : ℕ) (h1 : B = 301) (h2 : B % 7 = 0) : x = 3 :=
  sorry

end div_remainder_l472_472835


namespace at_least_two_mayors_visit_same_city_l472_472721

variables {City : Type}
variables (n : ℕ) (dist : City → City → ℝ)
variables (mayor_city : ℕ → City)
variables [finite City] [inhabited City]

-- n is an odd integer greater than 3
def valid_cities (n : ℕ) : Prop := n > 3 ∧ n % 2 = 1

-- distances between cities are distinct
def distinct_distances (n : ℕ) (dist : City → City → ℝ) : Prop :=
  ∀ (M N P Q : City), M ≠ N ∧ P ≠ Q ∧ (M = P ∨ N = Q ∨ M ≠ P ∨ N ≠ Q) → dist M N ≠ dist P Q

-- each mayor travels to the closest city
def closest_city (n : ℕ) (dist : City → City → ℝ) (mayor_city : ℕ → City) : Prop :=
  ∀ M, (∃ N, N ≠ M ∧ dist (mayor_city M) (mayor_city N) = 
    min {d (mayor_city M) (mayor_city i) | i ≠ M}) 

theorem at_least_two_mayors_visit_same_city 
  (n : ℕ) (dist : City → City → ℝ) (mayor_city : ℕ → City) 
  [finite City] [inhabited City] : 
  valid_cities n → distinct_distances n dist → closest_city n dist mayor_city → 
  ∃ c : City, ∃ M N : ℕ, M ≠ N ∧ mayor_city M = c ∧ mayor_city N = c :=
begin
  sorry
end

end at_least_two_mayors_visit_same_city_l472_472721


namespace total_parts_in_batch_l472_472082

/-- A proof statement asserting the total number of parts in a batch given specific working conditions -/
theorem total_parts_in_batch :
  (∀ t : ℕ, t > 0 → ∀ a_parts b_parts : ℕ, 
    (a_parts = t / 10) → (b_parts = t / 12) → 
    ((a_parts * 1 - b_parts * 1 = 40) → (t = 2400))) :=
begin
  sorry
end

end total_parts_in_batch_l472_472082


namespace part_a_1_part_a_2_part_b_1_planar_part_b_1_spatial_part_b_2_line_part_b_3_l472_472298

noncomputable def diameter (F : Type) [metric_space F] (s : set F) : ℝ := sorry

variable (F : Type) [metric_space F] (L : set F) (a : ℝ)

def is_planar (F : Type) : Prop := sorry
def is_spatial (F : Type) : Prop := sorry
def central_figure (F : Type) : set F := sorry
def a_central_figure (a : ℝ) (F : Type) : set F := sorry

theorem part_a_1 
  (cond : is_planar F) 
  (d : diameter F (central_figure F) ≤ 1) :
  diameter F (central_figure F) ≤ 1 / 2 := sorry

theorem part_a_2 
  (cond : is_spatial F) 
  (d : diameter F (central_figure F) ≤ 1) :
  diameter F (central_figure F) ≤ real.sqrt 2 / 2 := sorry

theorem part_b_1_planar 
  (cond : is_planar F) 
  (d : diameter F (a_central_figure a F) ≤ 1) :
  diameter F (a_central_figure a F) ≤ 1 - (a^2 / 2) := sorry

theorem part_b_1_spatial 
  (cond : is_spatial F) 
  (d : diameter F (a_central_figure a F) ≤ 1) :
  diameter F (a_central_figure a F) ≤ real.sqrt (1 - (a^2 / 2)) := sorry

theorem part_b_2_line
  (cond : diameter F (central_figure F) = 0) :
  diameter F (a_central_figure a F) = 1 - a := sorry

theorem part_b_3 
  (k_dim : ∀ k, k ≥ 3 → diameter F (a_central_figure a F) ≤ real.sqrt 1/2) :
  ∀ k, k ≥ 3 → diameter F (a_central_figure a F) = real.sqrt (1 - (a^2 / 2)) := sorry

end part_a_1_part_a_2_part_b_1_planar_part_b_1_spatial_part_b_2_line_part_b_3_l472_472298


namespace inequality_proof_l472_472707

theorem inequality_proof (a b x : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : -1 ≤ sin x ∧ sin x ≤ 1) :
  (b - a) / (b + a) ≤ (b + a * sin x) / (b - a * sin x) ∧
  (b + a * sin x) / (b - a * sin x) ≤ (b + a) / (b - a) :=
sorry

end inequality_proof_l472_472707


namespace cos_54_eq_3_sub_sqrt_5_div_8_l472_472138

theorem cos_54_eq_3_sub_sqrt_5_div_8 :
  let x := Real.cos (Real.pi / 10) in
  let y := Real.cos (3 * Real.pi / 10) in
  y = (3 - Real.sqrt 5) / 8 :=
by
  -- Proof of the statement is omitted.
  sorry

end cos_54_eq_3_sub_sqrt_5_div_8_l472_472138


namespace geometric_sequence_problem_l472_472530

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = q^(n-1)

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 1)
  (h_q_ne_one : q ≠ 1)
  (h_prod : a (46) = (∏ i in finset.range 10, a (i+1))) :
  46 = 46 :=
by sorry

end geometric_sequence_problem_l472_472530


namespace triangle_area_is_12_l472_472276

noncomputable def angle_ABC (α β : ℝ) : ℝ := sorry
noncomputable def altitude_BD : ℝ := sorry
noncomputable def side_AB : ℝ := sorry
noncomputable def side_BC : ℝ := sorry
noncomputable def side_AC : ℝ := sorry
noncomputable def area_ABC : ℝ := 
  1 / 2 * side_AB * altitude_BD

theorem triangle_area_is_12 :
  (α β : ℝ) (hab : side_AB = side_BC) (hbe : sorry) 
  (hge : sorry) (hap : sorry) :
  area_ABC = 12 := sorry

end triangle_area_is_12_l472_472276


namespace periodic_function_l472_472212

variable {α : Type*} [AddGroup α] {f : α → α} {a b : α}

def symmetric_around (c : α) (f : α → α) : Prop := ∀ x, f (c - x) = f (c + x)

theorem periodic_function (h1 : symmetric_around a f) (h2 : symmetric_around b f) (h_ab : a ≠ b) : ∃ T, (∀ x, f (x + T) = f x) := 
sorry

end periodic_function_l472_472212


namespace max_cylinders_fit_l472_472414

noncomputable def volume_of_cube (s : ℝ) : ℝ := s^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

def max_cylinders_in_cube (s r h : ℝ) : ℕ :=
  ⌊volume_of_cube s / volume_of_cylinder r h⌋.to_nat

theorem max_cylinders_fit (s r h : ℝ) :
  s = 9 → r = 2 → h = 5 → max_cylinders_in_cube s r h = 11 :=
by
  intros h1 h2 h3
  rw [←h1, ←h2, ←h3]
  norm_cast
  have : volume_of_cube 9 = 729 := by norm_num [volume_of_cube]
  have : volume_of_cylinder 2 5 = 20 * π := by norm_num [volume_of_cylinder]
  suffices 729 / (20 * π) = 36.45 / π by
    have hp : π ≈ 3.14 := by norm_num
    norm_cast at hp
    have := congr_arg to_nat (floor_eq_iff_eq.symm.mp _)
    exact this
  suffices : 729 / 20 ≈ 36.45 := by norm_num
  suffices : 1 / π ≈ 1 / 3.14 by norm_num [π]
  exact congr_arg floor this
sorry

end max_cylinders_fit_l472_472414


namespace weekend_rain_probability_l472_472175

-- Let P_Sat denote the probability of rain on Saturday, which is 0.60.
-- Let P_Sun denote the probability of rain on Sunday, which is 0.70.
-- These probabilities are independent.
-- We need to show that the probability of raining over the weekend is 0.88.

theorem weekend_rain_probability :
  let P_Sat := 0.60 in
  let P_Sun := 0.70 in
  let P_no_rain_weekend := (1 - P_Sat) * (1 - P_Sun) in
  (1 - P_no_rain_weekend) = 0.88 :=
by
  let P_Sat := 0.60
  let P_Sun := 0.70
  let P_no_rain_weekend := (1 - P_Sat) * (1 - P_Sun)
  show (1 - P_no_rain_weekend) = 0.88
  sorry

end weekend_rain_probability_l472_472175


namespace mean_equals_median_diff_l472_472631

-- Define the problem conditions
def students := 50
def score_distribution : List (ℕ × ℕ) := [(60, 15), (75, 20), (85, 25), (90, 30), (100, 10)]

-- Define the median score based on the problem's conditions
def median_score := 85

-- Define the computed mean score based on the problem's conditions
def mean_score := 83.7

theorem mean_equals_median_diff : 
  |mean_score - median_score| = 1 :=
by
  -- Implementation details will be provided here.
  sorry

end mean_equals_median_diff_l472_472631


namespace quarters_in_jar_l472_472834

noncomputable def total_value_of_coins_excluding_quarters : ℝ :=
  123 * 0.01 + 85 * 0.05 + 35 * 0.10 + 15 * 0.50 + 5 * 1.00

noncomputable def total_cost_of_ice_cream : ℝ :=
  8 * 4.50

noncomputable def total_money_spent : ℝ :=
  total_cost_of_ice_cream - 0.97

noncomputable def total_money_in_jar_before_trip (quarters_value : ℝ) : ℝ :=
  total_value_of_coins_excluding_quarters + quarters_value

noncomputable def number_of_quarters (quarters_value : ℝ) : ℝ :=
  quarters_value / 0.25

theorem quarters_in_jar :
  ∃ (quarters_value : ℝ), total_money_in_jar_before_trip quarters_value = total_money_spent + total_value_of_coins_excluding_quarters ∧ number_of_quarters quarters_value = 140 :=
begin
  use 35.00,  -- This is the value of quarters required
  split,
  { simp [total_money_in_jar_before_trip, total_value_of_coins_excluding_quarters, total_money_spent, total_cost_of_ice_cream],
    norm_num,
    exact 56.51, },
  { simp [number_of_quarters],
    norm_num },
  sorry -- Skipping the detailed proof here
end

end quarters_in_jar_l472_472834


namespace square_of_1037_l472_472490

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := 
by {
  -- Proof omitted
  sorry
}

end square_of_1037_l472_472490


namespace lemonade_volume_water_l472_472764

theorem lemonade_volume_water (r_w r_l : ℕ) (V_total_gallons : ℕ) (q_g : ℝ) (l_q : ℝ) 
  (h_ratio : r_w = 8) (h_ratio2 : r_l = 2) 
  (h_total_gallons : V_total_gallons = 2) 
  (h_q_g : q_g = 4) 
  (h_l_q : l_q = 0.95) : 
  let total_parts := r_w + r_l in
  let total_quarts := V_total_gallons * q_g in
  let volume_per_part_quarts := total_quarts / total_parts in
  let volume_water_quarts := volume_per_part_quarts * r_w in
  let volume_water_liters := volume_water_quarts * l_q in
  volume_water_liters = 6.08 := 
by 
  sorry

end lemonade_volume_water_l472_472764


namespace shortest_path_length_l472_472275

/-- 
Prove that the length of the shortest path from (0,0) to (16,12) 
that does not go inside the circle (x-8)^2 + (y-6)^2 = 36 is 16 + 3π.
-/
theorem shortest_path_length :
  let A := (0, 0)
  let D := (16, 12)
  let O := (8, 6)
  let r := 6
  let path_length := 16 + 3 * Real.pi
  (dist O A) = 10 → (dist O D) = 10 → path_length = 16 + 3 * Real.pi :=
by
  intros
  let circle (O : ℝ × ℝ) (r : ℝ) := (λ x y, (x - O.1)^2 + (y - O.2)^2 - r^2 = 0)
  have h1 : circle O r (8, 6) = 0 := sorry
  have h2 : dist (0, 0) (16, 12) = 20 := sorry
  have h3 : dist (0, 0) (8, 6) = 10 := sorry
  sorry

end shortest_path_length_l472_472275


namespace range_of_a_l472_472253

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a^2 ≤ 0 → false) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l472_472253


namespace total_screens_sold_l472_472864

variable (J F M : ℕ)
variable (feb_eq_fourth_of_march : F = M / 4)
variable (feb_eq_double_of_jan : F = 2 * J)
variable (march_sales : M = 8800)

theorem total_screens_sold (J F M : ℕ)
  (feb_eq_fourth_of_march : F = M / 4)
  (feb_eq_double_of_jan : F = 2 * J)
  (march_sales : M = 8800) :
  J + F + M = 12100 :=
by
  sorry

end total_screens_sold_l472_472864


namespace find_number_x_l472_472440

theorem find_number_x (x : ℝ) (h : 2500 - x / 20.04 = 2450) : x = 1002 :=
by
  -- Proof can be written here, but skipped by using sorry
  sorry

end find_number_x_l472_472440


namespace alice_bob_carol_probability_l472_472467

noncomputable def probability_after_100_rings (initial_money : ℕ → ℕ) (rings : ℕ) : ℚ :=
sorry

def money_update : ℕ → (ℕ → ℚ) :=
sorry

theorem alice_bob_carol_probability :
  let initial_money := (fun x => 3 : ℕ → ℕ) in
  initial_money 0 = 3 ∧ initial_money 1 = 3 ∧ initial_money 2 = 3 →
  let rings := 100 in
  money_update rings (probability_after_100_rings initial_money rings) = 8 / 13 :=
sorry

end alice_bob_carol_probability_l472_472467


namespace age_difference_l472_472628

variable (A B : ℕ)

-- Given conditions
def B_is_95 : Prop := B = 95
def A_after_30_years : Prop := A + 30 = 2 * (B - 30)

-- Theorem to prove
theorem age_difference (h1 : B_is_95 B) (h2 : A_after_30_years A B) : A - B = 5 := 
by
  sorry

end age_difference_l472_472628


namespace max_servings_l472_472031

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l472_472031


namespace probability_odd_sum_die_rolls_l472_472763

theorem probability_odd_sum_die_rolls : 
  let coin_tosses := 3 in
  let number_of_heads_to_dice_rolls := (heads : ℕ) → 2 * heads in
  let p := (1 / 2 : ℚ) in
  let toss_probability := (tosses : ℕ) → (p ^ tosses : ℚ) in
  (toss_probability coin_tosses) / 2  = (7 / 16 : ℚ) :=
sorry

end probability_odd_sum_die_rolls_l472_472763


namespace expected_stand_ups_expected_never_stand_l472_472692

-- Part (a)
theorem expected_stand_ups (n : ℕ) : 
  ∀ girls_with_tickets (politeness_condition : ∀ (i j : ℕ), i ≠ j ∧ i ≤ n ∧ j ≤ n → Prop),
  (expected_stand_ups n girls_with_tickets politeness_condition) = n * (n - 1) / 4 :=
sorry

-- Part (b)
theorem expected_never_stand (n : ℕ) : 
  ∀ girls_with_tickets (politeness_condition : ∀ (i j : ℕ), i ≠ j ∧ i ≤ n ∧ j ≤ n → Prop),
  (expected_never_stand n girls_with_tickets politeness_condition) = 
  (1 : ℚ) + (2 : ℚ)⁻¹ + (3 : ℚ)⁻¹ + ... + (n : ℚ)⁻¹ :=
sorry

end expected_stand_ups_expected_never_stand_l472_472692


namespace Zachary_game_price_l472_472801

noncomputable def price_per_game (R J Z : ℝ) : ℝ :=
  let total := R + J + Z
  let price := Z / 40
  if (R = J + 50) ∧ (J = 1.30 * Z) ∧ (total = 770) then price else -1

theorem Zachary_game_price :
  ∃ R J Z : ℝ,
  let price := price_per_game R J Z in
  R = J + 50 ∧ J = 1.30 * Z ∧ R + J + Z = 770 ∧ price = 5 :=
sorry

end Zachary_game_price_l472_472801


namespace trigonometric_inequality_l472_472682

noncomputable def a : Real := (1/2) * Real.cos (8 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (8 * Real.pi / 180)
noncomputable def b : Real := (2 * Real.tan (14 * Real.pi / 180)) / (1 - (Real.tan (14 * Real.pi / 180))^2)
noncomputable def c : Real := Real.sqrt ((1 - Real.cos (48 * Real.pi / 180)) / 2)

theorem trigonometric_inequality :
  a < c ∧ c < b := by
  sorry

end trigonometric_inequality_l472_472682


namespace remainder_of_division_l472_472365

theorem remainder_of_division : 
  ∀ (L x : ℕ), (L = 1430) → 
               (L - x = 1311) → 
               (L = 11 * x + (L % x)) → 
               (L % x = 121) :=
by
  intros L x L_value diff quotient
  sorry

end remainder_of_division_l472_472365


namespace maria_ends_up_with_22_towels_l472_472061

-- Define the number of green towels Maria bought
def green_towels : Nat := 35

-- Define the number of white towels Maria bought
def white_towels : Nat := 21

-- Define the number of towels Maria gave to her mother
def given_towels : Nat := 34

-- Total towels Maria initially bought
def total_towels := green_towels + white_towels

-- Towels Maria ended up with
def remaining_towels := total_towels - given_towels

theorem maria_ends_up_with_22_towels :
  remaining_towels = 22 :=
by
  sorry

end maria_ends_up_with_22_towels_l472_472061


namespace num_divisors_not_divisible_by_3_l472_472603

-- Define the prime factorization of 180
def prime_factorization_180 : Nat → Nat :=
  λ n, if n = 2 then 2 else if n = 3 then 2 else if n = 5 then 1 else 0

-- Define the conditions for valid exponents a, b, c
def valid_exponent_a (a : Nat) : Prop := 0 ≤ a ∧ a ≤ 2
def valid_exponent_b (b : Nat) : Prop := b = 0
def valid_exponent_c (c : Nat) : Prop := 0 ≤ c ∧ c ≤ 1

-- Define the problem statement in Lean
theorem num_divisors_not_divisible_by_3 : 
  (∑ a in Finset.range 3, ∑ c in Finset.range 2, 
     if valid_exponent_a a ∧ valid_exponent_b 0 ∧ valid_exponent_c c 
     then 1 else 0) = 6 := 
  by
    sorry

end num_divisors_not_divisible_by_3_l472_472603


namespace binomial_prob_1_l472_472985

noncomputable def bernoulli_trial (n : ℕ) (p : ℚ) : Measure ℕ :=
  Measure.dirac (binomial_distribution n p)

theorem binomial_prob_1 : (Probability := bernoulli_trial 3 (1/3)) (ProbabilityEvent.Exactly 1) = 4/9 :=
sorry

end binomial_prob_1_l472_472985


namespace complex_point_quadrant_l472_472961

-- Definitions of the conditions and the final statement
def complex_plane_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First"
  else if z.re < 0 ∧ z.im > 0 then "Second"
  else if z.re < 0 ∧ z.im < 0 then "Third"
  else if z.re > 0 ∧ z.im < 0 then "Fourth"
  else "Axis or Origin"

theorem complex_point_quadrant :
  ∀ z : ℂ, (1 - complex.I) * z = 2 * complex.I → complex_plane_quadrant z = "Second" :=
by
  intro z h
  sorry

end complex_point_quadrant_l472_472961


namespace probability_not_power_of_2_l472_472847

theorem probability_not_power_of_2 :
  let S := {2, 4, 6, 8, 10}
  let choices := λ x : ℕ, x ∈ S
  let P := λ x : ℕ, ∃ n : ℕ, x = 2^n
  let power_of_2_product := 
    (choices abigail ∧ choices bill ∧ choices charlie) ∧
    P (abigail * bill * charlie)
  let count_powers_of_2 := 3
  let total_elements := 5
  (1 - (count_powers_of_2.toRat / total_elements.toRat)^3) =
  98 / 125 := sorry

end probability_not_power_of_2_l472_472847


namespace find_BC_length_l472_472433

noncomputable def median_of_triangle (A B C D : Point) : Prop :=
  segment D A = segment D C

theorem find_BC_length {A B C D : Point}
  (hmedian : median_of_triangle A B C D)
  (hangle : ∠A B D = 90)
  (hAB : segment A B = 2)
  (hAC : segment A C = 6) : 
  segment B C = 2 * sqrt 14 := sorry

end find_BC_length_l472_472433


namespace sqrt_square_eq_14_l472_472070

theorem sqrt_square_eq_14 : Real.sqrt (14 ^ 2) = 14 :=
by
  sorry

end sqrt_square_eq_14_l472_472070


namespace inequality_holds_l472_472814

noncomputable def positive_real_numbers := { x : ℝ // 0 < x }

theorem inequality_holds (a b c : positive_real_numbers) (h : (a.val * b.val + b.val * c.val + c.val * a.val) = 1) :
    (a.val / b.val + b.val / c.val + c.val / a.val) ≥ (a.val^2 + b.val^2 + c.val^2 + 2) :=
by
  sorry

end inequality_holds_l472_472814


namespace vector_addition_problem_l472_472673

variables {x y : ℝ}
def a := (x, 1 : ℝ)
def b := (1, y : ℝ)
def c := (2, -4 : ℝ)

theorem vector_addition_problem (h₁ : 2 * x + 1 * (-4) = 0) (h₂ : 1 * (-4) = 2 * y) :
  (a.1 + b.1, a.2 + b.2) = (3, -1) :=
by
  sorry

end vector_addition_problem_l472_472673


namespace triangle_angles_ratios_l472_472354

def angles_of_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

theorem triangle_angles_ratios (α β γ : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : β = 2 * α)
  (h3 : γ = 3 * α) : 
  angles_of_triangle 60 45 75 ∨ angles_of_triangle 45 22.5 112.5 :=
by
  sorry

end triangle_angles_ratios_l472_472354


namespace cosine_54_deg_l472_472134

theorem cosine_54_deg : ∃ c : ℝ, c = cos (54 : ℝ) ∧ c = 1 / 2 :=
  by 
    let c := cos (54 : ℝ)
    let d := cos (108 : ℝ)
    have h1 : d = 2 * c^2 - 1 := sorry
    have h2 : d = -c := sorry
    have h3 : 2 * c^2 + c - 1 = 0 := sorry
    use 1 / 2 
    have h4 : c = 1 / 2 := sorry
    exact ⟨cos_eq_cos_of_eq_rad 54 1, h4⟩

end cosine_54_deg_l472_472134


namespace lines_parallel_or_intersect_l472_472188

theorem lines_parallel_or_intersect
  (triangle : Type)
  (vertices : finset ℝ)
  (lines : finset (set ℝ)) 
  (h_vertices : vertices.card = 3) 
  (h_lines : lines.card = 10) 
  (h_eq_dist : ∀ line ∈ lines, ∃ v1 v2 ∈ vertices, line = {x : ℝ | dist x v1 = dist x v2}) :
  (∃ l1 l2 ∈ lines, parallel l1 l2) ∨ (∃ l1 l2 l3 ∈ lines, ∃ p ∈ set.univ, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3) := 
sorry

end lines_parallel_or_intersect_l472_472188


namespace max_salad_servings_l472_472005

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l472_472005


namespace prob_xi_eq_12_l472_472629

noncomputable def prob_of_draws (total_draws red_draws : ℕ) (prob_red prob_white : ℚ) : ℚ :=
    (Nat.choose (total_draws - 1) (red_draws - 1)) * (prob_red ^ (red_draws - 1)) * (prob_white ^ (total_draws - red_draws)) * prob_red

theorem prob_xi_eq_12 :
    prob_of_draws 12 10 (3 / 8) (5 / 8) = 
    (Nat.choose 11 9) * (3 / 8)^9 * (5 / 8)^2 * (3 / 8) :=
by sorry

end prob_xi_eq_12_l472_472629


namespace p_necessary_but_not_sufficient_for_q_l472_472936

/- Definitions and conditions -/
def p (x : ℝ) : Prop := x - 1 = real.sqrt (x - 1)
def q (x : ℝ) : Prop := x = 2

/- The theorem to prove that p is a necessary but not sufficient condition for q -/
theorem p_necessary_but_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) :=
by
  sorry

end p_necessary_but_not_sufficient_for_q_l472_472936


namespace diameter_of_tripled_volume_sphere_l472_472723

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem diameter_of_tripled_volume_sphere :
  let r1 := 6
  let V1 := volume_sphere r1
  let V2 := 3 * V1
  let r2 := (V2 * 3 / (4 * Real.pi))^(1 / 3)
  let D := 2 * r2
  ∃ (a b : ℕ), (D = a * (b:ℝ)^(1 / 3) ∧ b ≠ 0 ∧ ∀ n : ℕ, n^3 ∣ b → n = 1) ∧ a + b = 15 :=
by
  sorry

end diameter_of_tripled_volume_sphere_l472_472723


namespace max_servings_l472_472036

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l472_472036


namespace card_probability_l472_472000

/-- Three cards are drawn at random from a standard deck of 52 cards.
What is the probability that the first card is an Ace, the second card is a Diamond, 
and the third card is a King?
-/
theorem card_probability :
  let p := (3 / 52) * (12 / 51) * (4 / 50) +
            (3 / 52) * (1 / 51) * (3 / 50) +
            (1 / 52) * (11 / 51) * (4 / 50) +
            (1 / 52) * (1 / 51) * (3 / 50)
  in p = 1 / 663 :=
begin
  sorry
end

end card_probability_l472_472000


namespace max_servings_l472_472040

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l472_472040


namespace parabola_focus_coordinates_l472_472226

theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), y^2 = 6 * x → (x, y) = (3 / 2, 0) :=
begin
  sorry
end

end parabola_focus_coordinates_l472_472226


namespace frank_has_3_cookies_l472_472925

-- The definitions and conditions based on the problem statement
def num_cookies_millie : ℕ := 4
def num_cookies_mike : ℕ := 3 * num_cookies_millie
def num_cookies_frank : ℕ := (num_cookies_mike / 2) - 3

-- The theorem stating the question and the correct answer
theorem frank_has_3_cookies : num_cookies_frank = 3 :=
by 
  -- This is where the proof steps would go, but for now we use sorry
  sorry

end frank_has_3_cookies_l472_472925


namespace rotated_Q_coordinates_l472_472875

-- Assign points O, P, and Q
def O := (0, 0)
def P := (4, 0)
def Q := (0, 4)

-- Define the rotation
def rotate (θ : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ)

-- Prove the rotated coordinates
theorem rotated_Q_coordinates :
  rotate (Real.pi / 4) 0 4 = (-2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
begin
  sorry
end

end rotated_Q_coordinates_l472_472875


namespace intersection_area_eq_pi_l472_472221

def f (x : ℝ) : ℝ := x^2 - 1
def M : set (ℝ × ℝ) := { p | f p.1 + f p.2 ≤ 0 }
def N : set (ℝ × ℝ) := { p | f p.1 - f p.2 ≥ 0 }

theorem intersection_area_eq_pi :
  let region : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 ≤ 2 } ∩ { p | |p.2| ≤ |p.1| } in
  measure_theory.measure_union_inter (measure_theory.measure_space.measurable_set.region)
  .measure (region) = real.pi :=
sorry

end intersection_area_eq_pi_l472_472221


namespace smallest_digit_divisibility_l472_472910

theorem smallest_digit_divisibility : 
  ∃ d : ℕ, (d < 10) ∧ (∃ k1 k2 : ℤ, 5 + 2 + 8 + d + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d + 7 + 4 = 3 * k2) ∧ (∀ d' : ℕ, (d' < 10) ∧ 
  (∃ k1 k2 : ℤ, 5 + 2 + 8 + d' + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d' + 7 + 4 = 3 * k2) → d ≤ d') :=
by
  sorry

end smallest_digit_divisibility_l472_472910


namespace triangle_area_FGH_l472_472647

open Set

-- Definitions of points D, E, and F
noncomputable def D (A B : Point) : Point := 
  -- Definition for trisection point not explicitly given in Lean mathlib
  sorry 

noncomputable def E (A B : Point) : Point := 
  -- Definition for trisection point not explicitly given in Lean mathlib
  sorry

noncomputable def F (A C : Point) : Point := 
  -- Definition for midpoint not explicitly given in Lean mathlib
  sorry

-- Definition of triangles using the points
noncomputable def AreaOfTriangle (A B C : Point) (area : ℝ) : Prop :=
  -- Function to calculate the area of a triangle not explicitly available in Lean mathlib
  sorry 

-- Given conditions
variables {A B C D E F G H : Point}
--here D E and F are defined in lean above. G and H are not
noncomputable def G : Point :=
  -- F E intersect CB at G
  sorry

noncomputable def H : Point :=
  -- F D intersect CB at H
  sorry

-- Main theorem statement
theorem triangle_area_FGH :
  ∃ (A B C D E F G H : Point), 
  AreaOfTriangle A B C 1 ∧ 
  D = D A B ∧
  E = E A B ∧
  F = F A C ∧
  (∃ l1 l2 : Line, l1 = LineThrough F E ∧ l2 = LineThrough C B ∧ G = Intersection l1 l2 ∧ G ∈ l2) ∧
  (∃ l3 l4 : Line, l3 = LineThrough F D ∧ l4 = LineThrough C B ∧ H = Intersection l3 l4 ∧ H ∈ l4) → 
  AreaOfTriangle F G H (1/6)
:=
begin
  sorry
end

end triangle_area_FGH_l472_472647


namespace maximum_sum_of_distances_l472_472697

theorem maximum_sum_of_distances (P V : Point ℝ) (d : ℕ → Point ℝ) (hP : ∑ i in range 100, dist P (d i) = 1000) :
  (∑ i in range 100, dist V (d i)) ≤ 99000 :=
sorry

end maximum_sum_of_distances_l472_472697


namespace no_upper_bound_on_c_l472_472171

-- Definition of the average of a list of 100 real numbers
def average (lst : List ℝ) : ℝ := (lst.sum) / (lst.length)

-- Theorem stating that there is no upper bound for the largest c
theorem no_upper_bound_on_c :
  ∀ (c : ℝ),
  ∃ (x : Fin 100 → ℝ),
  (Vector.sum (fun i => x i) = 0) ∧
  (Vector.sum (fun i => x i ^ 2) ≥ c * (average (Vector.to_list x))^2) :=
by
  intro c
  -- Construct the required example to show the inequality.
  -- The proof steps are omitted as they are not required.
  sorry

end no_upper_bound_on_c_l472_472171


namespace zero_extreme_points_l472_472374

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

theorem zero_extreme_points : ∀ x : ℝ, 
  ∃! (y : ℝ), deriv f y = 0 → y = x :=
by
  sorry

end zero_extreme_points_l472_472374


namespace num_values_of_n_l472_472146

open Nat

theorem num_values_of_n : 
  (∃ (n : ℕ), ∃ (a b c : ℕ), 7*a + 77*b + 777*c = 3100 ∧ n = a + 2*b + 3*c) → 
  ((∃ n : ℕ, true) ∧ (card {n | ∃ (a b c : ℕ), 7*a + 77*b + 777*c = 3100 ∧ n = a + 2*b + 3*c}) = 50) := sorry

end num_values_of_n_l472_472146


namespace magnitude_z1_pure_imaginary_l472_472214

open Complex

theorem magnitude_z1_pure_imaginary 
  (a : ℝ)
  (z1 : ℂ := a + 2 * I)
  (z2 : ℂ := 3 - 4 * I)
  (h : (z1 / z2).re = 0) :
  Complex.abs z1 = 10 / 3 := 
sorry

end magnitude_z1_pure_imaginary_l472_472214


namespace coin_collection_problem_l472_472395

variable (n d q : ℚ)

theorem coin_collection_problem 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 20 * q = 340)
  (h3 : d = 2 * n) :
  q - n = 2 / 7 := by
  sorry

end coin_collection_problem_l472_472395


namespace number_of_paths_l472_472501

open Finset

structure Tetrahedron := (vertices : Finset ℕ) 
  (edges : vertices → vertices → Prop)

/-- Example of a Tetrahedron with vertices {0, 1, 2, 3} -/
def example_tetrahedron : Tetrahedron :=
{ vertices := {0, 1, 2, 3},
  edges := by {
    assume (x y : ℕ), 
    exact (x ≠ y) -- This should be refined to actual edges if necessary.
  }
}

def valid_paths (t : Tetrahedron) (start : ℕ) : Finset (List ℕ) :=
  { l | l ∈ t.vertices.to_list.permutations ∧ l.head = start ∧ l.length = 4 
       ∧ ∀ (i : ℕ), i < 3 → t.edges (l.nth_le i sorry) (l.nth_le (i + 1) sorry)}

theorem number_of_paths (t : Tetrahedron) (start : ℕ) 
  (h_t_edges : ∀ (x y z w : ℕ), 
    x ∈ t.vertices → y ∈ t.vertices → z ∈ t.vertices → w ∈ t.vertices → 
    t.edges x y → t.edges y z → t.edges z w → ¬ t.edges w x) : 
  (valid_paths t start).card = 6 := 
  sorry

end number_of_paths_l472_472501


namespace proof_equation_of_line_l472_472505

noncomputable def equation_of_line_par_tangent (P : ℝ × ℝ) (M : ℝ × ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  P = (-1, 2) ∧ M = (1, 1) ∧ 
  f x = 3 * x * x - 4 * x + 2 ∧ 
  f' x = 6 * x - 4 ∧ 
  let m := f' 1 in 
  m = 2 ∧ 
  ∀ x y : ℝ, (y - P.2) = m * (x - P.1) ↔ (2 * x - y + 4 = 0)

theorem proof_equation_of_line : equation_of_line_par_tangent (-1,2) (1,1) (λ x, 3 * x^2 - 4 * x + 2) (λ x, 6 * x - 4) := 
begin 
  sorry 
end

end proof_equation_of_line_l472_472505


namespace Jamie_earnings_l472_472657

theorem Jamie_earnings
  (earn_per_hour : ℕ)
  (days_per_week : ℕ)
  (hours_per_day : ℕ)
  (weeks : ℕ)
  (earnings : ℕ) :
  earn_per_hour = 10 →
  days_per_week = 2 →
  hours_per_day = 3 →
  weeks = 6 →
  earnings = earn_per_hour * days_per_week * hours_per_day * weeks →
  earnings = 360 :=
  by
    intros
    rw [H, H_1, H_2, H_3]
    have h1 : 10 * 2 = 20 := by norm_num
    have h2 : 20 * 3 = 60 := by norm_num
    have h3 : 60 * 6 = 360 := by norm_num
    rw [h1, h2, h3]
    exact H_4

end Jamie_earnings_l472_472657


namespace find_xy_solution_l472_472177

theorem find_xy_solution (x y : ℕ) (hx : x > 0) (hy : y > 0) 
    (h : 3^x + x^4 = y.factorial + 2019) : 
    (x = 6 ∧ y = 3) :=
by {
  sorry
}

end find_xy_solution_l472_472177


namespace max_servings_l472_472026

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l472_472026


namespace tan_half_angle_product_l472_472955

variables {a b x y e : Real}
variables {α β : Real}

theorem tan_half_angle_product (P_on_ellipse : x^2 / a^2 + y^2 / b^2 = 1) 
                               (foci_F1_F2 : True)  -- Placeholder, we don't model foci directly
                               (eccentricity : True) -- Placeholder, we don't model eccentricity directly
                               (angle_α_beta : True) -- Placeholder, we deal with angles as given
                               : (tan (α / 2) * tan (β / 2)) = (1 - e) / (1 + e) :=
sorry

end tan_half_angle_product_l472_472955


namespace unit_digit_of_fourth_number_l472_472383

theorem unit_digit_of_fourth_number
  (n1 n2 n3 n4 : ℕ)
  (h1 : n1 % 10 = 4)
  (h2 : n2 % 10 = 8)
  (h3 : n3 % 10 = 3)
  (h4 : (n1 * n2 * n3 * n4) % 10 = 8) : 
  n4 % 10 = 3 :=
sorry

end unit_digit_of_fourth_number_l472_472383


namespace equivalent_expression_for_a_five_l472_472114

theorem equivalent_expression_for_a_five (a : ℕ) :
  (a^2 * a^3 = a^5) ∧ ¬(a^2 + a^3 = a^5) ∧ ¬(a^5 / a = a^5) ∧ ¬((a^2)^3 = a^5) :=
by {
  apply and.intro,
  { have h : a^2 * a^3 = a^(2 + 3) := nat.pow_add,
    exact h },
  apply and.intro,
  { intro h,
    exact nat.one_ne_zero (nat.pow_sub_of_div (nat.div_add_div_same_left a^2 a^3)) },
  apply and.intro,
  { intro h,
    exact nat.one_ne_zero (nat.pow_sub_of_div (nat.div_one a^5)) },
  { intro h,
    exact nat.one_ne_zero (nat.pow_mul (a^2) 3) }
}

end equivalent_expression_for_a_five_l472_472114


namespace orchestra_members_l472_472375

theorem orchestra_members (n : ℕ) (h₀ : 100 ≤ n) (h₁ : n ≤ 300)
    (h₂ : n % 4 = 3) (h₃ : n % 5 = 1) (h₄ : n % 7 = 5) : n = 231 := by
  sorry

end orchestra_members_l472_472375


namespace num_divisors_not_divisible_by_3_l472_472600

-- Define the prime factorization of 180
def prime_factorization_180 : Nat → Nat :=
  λ n, if n = 2 then 2 else if n = 3 then 2 else if n = 5 then 1 else 0

-- Define the conditions for valid exponents a, b, c
def valid_exponent_a (a : Nat) : Prop := 0 ≤ a ∧ a ≤ 2
def valid_exponent_b (b : Nat) : Prop := b = 0
def valid_exponent_c (c : Nat) : Prop := 0 ≤ c ∧ c ≤ 1

-- Define the problem statement in Lean
theorem num_divisors_not_divisible_by_3 : 
  (∑ a in Finset.range 3, ∑ c in Finset.range 2, 
     if valid_exponent_a a ∧ valid_exponent_b 0 ∧ valid_exponent_c c 
     then 1 else 0) = 6 := 
  by
    sorry

end num_divisors_not_divisible_by_3_l472_472600


namespace libby_quarters_left_after_payment_l472_472314

noncomputable def quarters_needed (usd_target : ℝ) (usd_per_quarter : ℝ) : ℝ := 
  usd_target / usd_per_quarter

noncomputable def quarters_left (initial_quarters : ℝ) (used_quarters : ℝ) : ℝ := 
  initial_quarters - used_quarters

theorem libby_quarters_left_after_payment
  (initial_quarters : ℝ) (usd_target : ℝ) (usd_per_quarter : ℝ) 
  (h_initial : initial_quarters = 160) 
  (h_usd_target : usd_target = 35) 
  (h_usd_per_quarter : usd_per_quarter = 0.25) : 
  quarters_left initial_quarters (quarters_needed usd_target usd_per_quarter) = 20 := 
by
  sorry

end libby_quarters_left_after_payment_l472_472314


namespace smallest_of_four_numbers_l472_472131

def A : ℝ := -2^2
def B : ℝ := -|(-3)|
def C : ℝ := -(-1.5)
def D : ℝ := -3 / 2

theorem smallest_of_four_numbers : min (min A B) (min C D) = A := by
  sorry

end smallest_of_four_numbers_l472_472131


namespace distance_from_point_to_line_l472_472504

def point := ℝ × ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

def line_param (p1 p2 : point) (t : ℝ) : point :=
  (p1.1 + t * (p2.1 - p1.1), p1.2 + t * (p2.2 - p1.2), p1.3 + t * (p2.3 - p1.3))

def point_line_distance (a p1 p2 : point) : ℝ :=
  let d := (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3) in
  let t := ((a.1 - p1.1) * (p2.1 - p1.1) + (a.2 - p1.2) * (p2.2 - p1.2) + (a.3 - p1.3) * (p2.3 - p1.3)) /
           ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2) in
  distance a (line_param p1 p2 t)

theorem distance_from_point_to_line :
  point_line_distance (2, -2, 1) (1, 1, -1) (3, 0, 0) = 13 / 6 := by
  sorry

end distance_from_point_to_line_l472_472504


namespace parametric_curve_length_formula_l472_472482

noncomputable def parametric_curve_length : ℝ :=
  ∫ t in 0..(Real.pi / 2), Real.sqrt (9 * Real.cos t ^ 2 + 4 * Real.sin t ^ 2)

theorem parametric_curve_length_formula :
  parametric_curve_length = ∫ t in 0..(Real.pi / 2), Real.sqrt (9 * Real.cos t ^ 2 + 4 * Real.sin t ^ 2) :=
by
  -- Proof is omitted
  sorry

end parametric_curve_length_formula_l472_472482


namespace intersection_point_sum_l472_472304

theorem intersection_point_sum (A B C D : ℝ × ℝ) (p q r s : ℕ) 
  (hA : A = (0, 0)) (hB : B = (2, 1)) (hC : C = (5, 4)) (hD : D = (6, 0)) 
  (hLine : ∃ m b, ∀ x y, y = m * x + b ∧ y = 0 → y = -4 * x + 24)
  (hIntersect : ∃ x y, y = -4 * x + 24 ∧ (y - 0) = -4 * (x - 6) + 4 ∧ (x = p/q ∧ y = r/s)) :
  p + q + r + s = 58 := 
by 
  sorry

end intersection_point_sum_l472_472304


namespace second_team_odd_approximation_l472_472633

noncomputable def first_team_odd : ℝ := 1.28
noncomputable def third_team_odd : ℝ := 3.25
noncomputable def fourth_team_odd : ℝ := 2.05
noncomputable def amount_bet : ℝ := 5.00
noncomputable def expected_winnings : ℝ := 223.0072

theorem second_team_odd_approximation :
  ∃ (second_team_odd : ℝ), 
    second_team_odd = 44.60144 / (first_team_odd * third_team_odd * fourth_team_odd) 
    ∧ ∀ ε > 0, abs (second_team_odd - 5.23) < ε :=
begin
  sorry
end

end second_team_odd_approximation_l472_472633


namespace integer_solutions_of_inequality_l472_472745

/-- 
Prove that the number of integer values of x such that the inequality 
4 < sqrt(2 * x) < 5 is satisfied is 4. 
-/
theorem integer_solutions_of_inequality :
  ∃ (n : ℕ), (∀ (x : ℤ), 8 < x ∧ x < 12.5 → (x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12)) ∧ n = 4 :=
by
  -- The proof itself is not needed, we only state the theorem
  sorry

end integer_solutions_of_inequality_l472_472745


namespace parabola_translation_vertex_l472_472766

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation of the parabola
def translated_parabola (x : ℝ) : ℝ := (x + 3)^2 - 4*(x + 3) + 2 - 2 -- Adjust x + 3 for shift left and subtract 2 for shift down

-- The vertex coordinates function
def vertex_coords (f : ℝ → ℝ) (x_vertex : ℝ) : ℝ × ℝ := (x_vertex, f x_vertex)

-- Define the original vertex
def original_vertex : ℝ × ℝ := vertex_coords original_parabola 2

-- Define the translated vertex we expect
def expected_translated_vertex : ℝ × ℝ := vertex_coords translated_parabola (-1)

-- Statement of the problem
theorem parabola_translation_vertex :
  expected_translated_vertex = (-1, -4) :=
  sorry

end parabola_translation_vertex_l472_472766


namespace count_valid_integers_l472_472521

def D (n : ℕ) : ℕ := 
  let bin_list := n.to_digits 2
  (bin_list.zip (List.tail bin_list)).count (λ p => p.fst ≠ p.snd)

theorem count_valid_integers : {m : ℕ // m ≤ 127 ∧ D m = 2}.card = 30 :=
by sorry

end count_valid_integers_l472_472521


namespace find_a_if_odd_l472_472962

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 1) * (x + a)

theorem find_a_if_odd (a : ℝ) : (∀ x : ℝ, f (-x) a = -f x a) → a = 0 := by
  intro h
  have h0 : f 0 a = 0 := by
    simp [f]
    specialize h 0
    simp [f] at h
    exact h
  sorry

end find_a_if_odd_l472_472962


namespace train_length_l472_472810

theorem train_length (speed_kmph : ℕ) (time_sec : ℕ) (length_meters : ℕ) : speed_kmph = 90 → time_sec = 4 → length_meters = 100 :=
by
  intros h₁ h₂
  have speed_mps : ℕ := speed_kmph * 1000 / 3600
  have speed_mps_val : speed_mps = 25 := sorry
  have distance : ℕ := speed_mps * time_sec
  have distance_val : distance = 100 := sorry
  exact sorry

end train_length_l472_472810


namespace nat_power_of_p_iff_only_prime_factor_l472_472296

theorem nat_power_of_p_iff_only_prime_factor (p n : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, n = p^k) ↔ (∀ q : ℕ, Nat.Prime q → q ∣ n → q = p) := 
sorry

end nat_power_of_p_iff_only_prime_factor_l472_472296


namespace mag_a_plus_b_mag_a_minus_b_angle_a_plus_b_a_minus_b_l472_472205

variables {a b : ℝ}
-- Given conditions
def mag_a : ℝ := 6
def mag_b : ℝ := 6
def cos_theta : ℝ := Real.cos (π / 3)
def dot_product_ab : ℝ := mag_a * mag_b * cos_theta

-- Prove that |a + b| = 6 * sqrt 3
theorem mag_a_plus_b : Real.sqrt (mag_a^2 + 2 * dot_product_ab + mag_b^2) = 6 * Real.sqrt 3 := sorry

-- Prove that |a - b| = 6
theorem mag_a_minus_b : Real.sqrt (mag_a^2 - 2 * dot_product_ab + mag_b^2) = 6 := sorry

-- Prove that the angle between a + b and a - b is 90 degrees
theorem angle_a_plus_b_a_minus_b :
  (mag_a^2 - mag_b^2) = 0 := sorry

end mag_a_plus_b_mag_a_minus_b_angle_a_plus_b_a_minus_b_l472_472205


namespace find_circle2_l472_472210

/-- Define the first circle C1 -/
def circle1 (x y : ℝ) := (x-1)^2 + (y-1)^2 = 1

/-- Circle C1 and C2 have the coordinate axes as common tangents -/
def common_tangents (C1 C2 : (ℝ × ℝ) → Prop) := 
  ∀ (x y : ℝ), (C1 (x, y) → (x = 0 ∨ y = 0)) ∧ (C2 (x, y) → (x = 0 ∨ y = 0))

/-- Distance between the centers of C1 and C2 is 3*sqrt(2) -/
def distance_between_centers (c1 c2 : ℝ × ℝ) := 
  let (x1, y1) := c1 in
  let (x2, y2) := c2 in
  (x2 - x1)^2 + (y2 - y1)^2 = (3 * Real.sqrt 2)^2

/-- Define circle -/
def circle (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) := (x - c.1)^2 + (y - c.2)^2 = r^2

/-- Main theorem proving the equation of circle C2 -/
theorem find_circle2 (c2 : ℝ × ℝ) (r : ℝ) :
  common_tangents circle1 (circle c2 r) ∧ 
  distance_between_centers (1, 1) c2 →
  circle c2 r = 
  (circle (4, 4) 4 ∨
   circle (-2, -2) 2 ∨
   circle (2 * Real.sqrt 2, -2 * Real.sqrt 2) 4 ∨
   circle (-2 * Real.sqrt 2, 2 * Real.sqrt 2) 4) := 
by 
  -- The proof can fill here later
  sorry

end find_circle2_l472_472210


namespace possible_values_of_c_l472_472686

theorem possible_values_of_c :
  ∃ (a b d e f g c : ℕ),
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
  (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
  (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
  (e ≠ f) ∧ (e ≠ g) ∧
  (f ≠ g) ∧
  (1 ≤ a) ∧ (a ≤ 7) ∧
  (1 ≤ b) ∧ (b ≤ 7) ∧
  (1 ≤ c) ∧ (c ≤ 7) ∧
  (1 ≤ d) ∧ (d ≤ 7) ∧
  (1 ≤ e) ∧ (e ≤ 7) ∧
  (1 ≤ f) ∧ (f ≤ 7) ∧
  (1 ≤ g) ∧ (g ≤ 7) ∧
  (a + b + c = c + d + e) ∧
  (a + b + c = c + f + g) ∧
  ∃ (c1 c2 c3 : ℕ),
  c1 = 1 ∧ c2 = 4 ∧ c3 = 7 ∧ 
  fintype.card {x : ℕ // x = c1 ∨ x = c2 ∨ x = c3} = 3
  :=
begin
  sorry
end

end possible_values_of_c_l472_472686


namespace some_magical_beings_are_mystical_creatures_l472_472248

variable (Dragon MagicalBeing MysticalCreature : Type)
variable (isDragon : Dragon → MagicalBeing)
variable (isMysticalCreature : MysticalCreature → Prop)
variable (someMysticalCreaturesAreDragons : ∃ d : MysticalCreature, ∃ (a : Dragon), isMysticalCreature d ∧ d = ↑a)

theorem some_magical_beings_are_mystical_creatures :
  ∃ (m : MagicalBeing), ∃ (x : MysticalCreature), isMysticalCreature x :=
sorry

end some_magical_beings_are_mystical_creatures_l472_472248


namespace cost_of_apples_l472_472753

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l472_472753


namespace average_of_divisible_by_two_l472_472901

theorem average_of_divisible_by_two (A : Set ℕ) 
  (hA : A = {12, 14, 16, 18, 20}) : 
  let sum := (∑ x in A, x)
  let count := A.card
  (sum / count : ℕ) = 16 := by
  sorry

end average_of_divisible_by_two_l472_472901


namespace polar_coordinates_of_point_l472_472492

theorem polar_coordinates_of_point (x y : ℝ) (h : (x, y) = (3, -3)) :
  ∃ r θ : ℝ, r = 3 * Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by {
    use [3 * Real.sqrt 2, 7 * Real.pi / 4],
    split,
    { refl },
    split,
    { refl },
    split,
    { exact Real.sqrt_two_pos },
    split,
    { exact zero_le_of_re },
    { exact lt_two_pi_of_re }
}

end polar_coordinates_of_point_l472_472492


namespace tory_needs_to_raise_more_l472_472917

variable (goal : ℕ) (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
variable (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ)

def remainingAmount (goal : ℕ) 
                    (pricePerChocolateChip pricePerOatmealRaisin pricePerSugarCookie : ℕ)
                    (soldChocolateChip soldOatmealRaisin soldSugarCookie : ℕ) : ℕ :=
  let profitFromChocolateChip := soldChocolateChip * pricePerChocolateChip
  let profitFromOatmealRaisin := soldOatmealRaisin * pricePerOatmealRaisin
  let profitFromSugarCookie := soldSugarCookie * pricePerSugarCookie
  let totalProfit := profitFromChocolateChip + profitFromOatmealRaisin + profitFromSugarCookie
  goal - totalProfit

theorem tory_needs_to_raise_more : 
  remainingAmount 250 6 5 4 5 10 15 = 110 :=
by
  -- Proof omitted 
  sorry

end tory_needs_to_raise_more_l472_472917


namespace correct_statement_is_C_l472_472059

theorem correct_statement_is_C :
  (sqrt 4 = 2 ∧ sqrt (real.pow 4 (1/2)) = 2) ∧
  (real.cbrt 27 = 3 ∧ ¬ (real.cbrt 27 = -3)) ∧
  (sqrt 16 = 4 ∧ (sqrt 4 = 2 ∨ sqrt 4 = -2)) ∧
  (sqrt 9 = 3 ∧ (¬ (sqrt (sqrt 9) = 3) ∧ ¬ (sqrt (sqrt 9) = -3))) →
  true := sorry

end correct_statement_is_C_l472_472059


namespace stratified_sampling_l472_472461

noncomputable def total_employees : ℕ := 800
noncomputable def senior_titles : ℕ := 160
noncomputable def intermediate_titles : ℕ := 320
noncomputable def junior_titles : ℕ := 200
noncomputable def remaining_employees : ℕ := 120
noncomputable def sample_size : ℕ := 40

def sample_ratio := sample_size / total_employees

def senior_samples := senior_titles / 20
def intermediate_samples := intermediate_titles / 20
def junior_samples := junior_titles / 20
def remaining_samples := remaining_employees / 20

theorem stratified_sampling :
  (senior_samples, intermediate_samples, junior_samples, remaining_samples) = (8, 16, 10, 6) :=
by sorry

end stratified_sampling_l472_472461


namespace max_val_frac_leq_27_l472_472305

theorem max_val_frac_leq_27 {x y : ℝ} (h1 : 3 ≤ xy^2) (h2 : xy^2 ≤ 8) 
    (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) 
    : ∃ x y, 3 ≤ xy^2 ∧ xy^2 ≤ 8 ∧ 4 ≤ x^2 / y ∧ x^2 / y ≤ 9 ∧ ∀ z, (∃ x y, 3 ≤ xy^2 ∧ xy^2 ≤ 8 ∧ 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) → z ≤ x^3 / y^4 :=
begin
    sorry
end

end max_val_frac_leq_27_l472_472305


namespace teams_of_four_from_seven_l472_472342

theorem teams_of_four_from_seven : 
  (nat.choose 7 4) = 35 := 
by
  sorry

end teams_of_four_from_seven_l472_472342


namespace Bennett_sales_l472_472862

-- Define the variables for the number of screens sold in each month.
variables (J F M : ℕ)

-- State the given conditions.
theorem Bennett_sales (h1: F = 2 * J) (h2: F = M / 4) (h3: M = 8800) :
  J + F + M = 12100 := by
sorry

end Bennett_sales_l472_472862


namespace initial_men_in_garrison_l472_472832

variable (x : ℕ)

theorem initial_men_in_garrison (h1 : x * 65 = x * 50 + (x + 3000) * 20) : x = 2000 :=
  sorry

end initial_men_in_garrison_l472_472832


namespace intersection_of_four_convex_sets_nonempty_intersection_of_n_convex_sets_nonempty_l472_472776

-- Define what it means for a set to be convex
variable {α : Type*} [LinearOrderedField α] {X : Type*} [TopologicalSpace X] [BorelSpace X]

def is_convex (s : Set X) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → ∀ ⦃t : α⦄, 0 ≤ t → t ≤ 1 → t • x + (1 - t) • y ∈ s

-- Assume we have four convex sets
variables (C1 C2 C3 C4 : Set X)
variables (h1 : is_convex C1)
variables (h2 : is_convex C2)
variables (h3 : is_convex C3)
variables (h4 : is_convex C4)

-- Condition: the intersection of any three of them is non-empty
variables (h123 : (C1 ∩ C2 ∩ C3).Nonempty)
variables (h124 : (C1 ∩ C2 ∩ C4).Nonempty)
variables (h134 : (C1 ∩ C3 ∩ C4).Nonempty)
variables (h234 : (C2 ∩ C3 ∩ C4).Nonempty)

-- Part (a) statement: Show that the intersection of all four convex sets is non-empty
theorem intersection_of_four_convex_sets_nonempty :
  (C1 ∩ C2 ∩ C3 ∩ C4).Nonempty := sorry

-- Part (b) statement: Show that the theorem remains true if 4 is replaced by n ≥ 4
def is_intersected_by_any_subset_of_size (S : Finset (Set X)) (k : ℕ) : Prop :=
  ∀ (T : Finset (Set X)), T ⊆ S → T.card = k → (⋂₀ T).Nonempty

theorem intersection_of_n_convex_sets_nonempty {n : ℕ} (hn : n ≥ 4)
  (S : Finset (Set X)) (hS_len : S.card = n)
  (h_convex : ∀ s ∈ S, is_convex s)
  (h_intersection_condition : is_intersected_by_any_subset_of_size S (n - 1)) :
  (⋂₀ S).Nonempty := sorry

end intersection_of_four_convex_sets_nonempty_intersection_of_n_convex_sets_nonempty_l472_472776


namespace point_P_range_a_l472_472977

noncomputable def f (a x : ℝ) : ℝ := log a (x - 1) + 2

theorem point_P (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) : f a 2 = 2 :=
by
  unfold f
  have hlog : log a 1 = 0 := log_self a
  rw [hlog]
  simp

theorem range_a (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  (∀ x ∈ set.Icc 2 4, f a x ≤ -x + 8) ↔ (a ∈ set.Icc 0 1 ∨ a > real.sqrt 3) :=
by
  sorry

end point_P_range_a_l472_472977


namespace largest_number_l472_472391

theorem largest_number 
  (a b c : ℝ) (h1 : a = 0.8) (h2 : b = 1/2) (h3 : c = 0.9) (h4 : a ≤ 2) (h5 : b ≤ 2) (h6 : c ≤ 2) :
  max (max a b) c = 0.9 :=
by
  sorry

end largest_number_l472_472391


namespace solution_set_l472_472544

def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 4)

theorem solution_set (k : ℤ) :
  {x : ℝ | f x ≥ Real.sqrt 3} =
  ⋃ k : ℤ, Set.Icc (Real.pi / 24 + k * Real.pi / 2) (Real.pi / 8 + k * Real.pi / 2) := 
sorry

end solution_set_l472_472544


namespace find_m_l472_472539

noncomputable def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
(nat.choose n k) * p^k * (1 - p)^(n - k)

noncomputable def normal_pdf (μ σ x : ℝ) : ℝ :=
(1 / (σ * sqrt (2 * π))) * exp (- (x - μ)^2 / (2 * σ^2))

variable (X Y : Type)

axiom h_X : ∀ n p, X = binomial_pmf 4 2 (1/2)
axiom h_Y : ∀ μ σ², Y = μ ∧ μ = 3
axiom h_EY : ∀ E, E Y = 8 * (binomial_pmf 4 2 (1/2))
axiom h_prob_sym : ∀ P, P (Y ≤ 0) = P (Y ≥ m^2 + 2)

theorem find_m (m : ℝ) : m = 2 ∨ m = -2 :=
sorry

end find_m_l472_472539


namespace maximum_chord_line_eq_l472_472367

theorem maximum_chord_line_eq (x y : ℝ) :
  let P := (2, 1)
  let C := (1, -2)
  let circle := ∀ p : ℝ × ℝ, (p.1 - 1)^2 + (p.2 + 2)^2 = 5 → True
  (P.1 ≠ C.1 ∧ P.2 ≠ C.2 ∧ ∀ (x y : ℝ), (y + 2) / (x - 1) = (P.2 + 2) / (P.1 - 1) → 3 * x - y - 5 = 0) :=
by
  intro P C circle
  have h1 : P.1 = 2 := rfl
  have h2 : P.2 = 1 := rfl
  have h3 : C.1 = 1 := rfl
  have h4 : C.2 = -2 := rfl
  have h_circle : ∀ (x y : ℝ), (x - C.1)^2 + (y + C.2)^2 = 5 → True := circle
  have h_line : ∀ (x y : ℝ), (y + C.2) / (x - C.1) = (P.2 + C.2) / (P.1 - C.1) → 3 * x - y - 5 = 0 := sorry
  exact ⟨ne_of_lt (by linarith), ne_of_lt (by linarith), h_line⟩

end maximum_chord_line_eq_l472_472367


namespace max_value_expr_l472_472509

theorem max_value_expr (x y : ℝ) : (2 * x + 3 * y + 4) / (Real.sqrt (x^4 + y^2 + 1)) ≤ Real.sqrt 29 := sorry

end max_value_expr_l472_472509


namespace num_divisors_not_div_by_3_l472_472596

theorem num_divisors_not_div_by_3 : 
  let n := 180 in
  let prime_factorization_180 := factorization 180 in
  (prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180) →
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)} in
  divisors_not_div_by_3.card = 6 :=
by 
  let n := 180
  let prime_factorization_180 := factorization n
  have h_factorization : prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180 := -- proof ommitted
    sorry
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)}
  have h_card : divisors_not_div_by_3.card = 6 := -- proof ommitted
    sorry
  exact h_card

end num_divisors_not_div_by_3_l472_472596


namespace position_after_2010_transformations_l472_472369

-- Define the initial position of the square
def init_position := "ABCD"

-- Define the transformation function
def transform (position : String) (steps : Nat) : String :=
  match steps % 8 with
  | 0 => "ABCD"
  | 1 => "CABD"
  | 2 => "DACB"
  | 3 => "BCAD"
  | 4 => "ADCB"
  | 5 => "CBDA"
  | 6 => "BADC"
  | 7 => "CDAB"
  | _ => "ABCD"  -- Default case, should never happen

-- The theorem to prove the correct position after 2010 transformations
theorem position_after_2010_transformations : transform init_position 2010 = "CABD" := 
by
  sorry

end position_after_2010_transformations_l472_472369


namespace guards_per_team_l472_472758

theorem guards_per_team (forwards guards : ℕ) (h_forwards : forwards = 32) (h_guards : guards = 80)
: let gcd_value := Nat.gcd forwards guards in
  gcd_value = 16 → (guards / gcd_value) = 5 :=
by
  intros
  sorry

end guards_per_team_l472_472758


namespace maximize_xyplusxzplusyzplusy2_l472_472678

theorem maximize_xyplusxzplusyzplusy2 (x y z : ℝ) (h1 : x + 2 * y + z = 7) (h2 : y ≥ 0) :
  xy + xz + yz + y^2 ≤ 10.5 :=
sorry

end maximize_xyplusxzplusyzplusy2_l472_472678


namespace maximum_omega_l472_472566

noncomputable def f (omega varphi : ℝ) (x : ℝ) : ℝ :=
  Real.cos (omega * x + varphi)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f y ≤ f x

theorem maximum_omega (omega varphi : ℝ)
    (h0 : omega > 0)
    (h1 : 0 < varphi ∧ varphi < π)
    (h2 : is_odd_function (f omega varphi))
    (h3 : is_monotonically_decreasing (f omega varphi) (-π/3) (π/6)) :
  omega ≤ 3/2 :=
sorry

end maximum_omega_l472_472566


namespace transformed_series_telescoping_series_l472_472912

noncomputable def transformed_term (k : ℕ) : ℝ :=
  (∑ k in Nat.range (k+1), (3^k / (3^(2*k) - 2)))

theorem transformed_series :
  (∀ k : ℕ, (3^k / (3^(2*k) - 2)) = (1 / (3^k - 1)) - (2 / (3^(2*k) - 2))) :=
by
  sorry

theorem telescoping_series :
  ∑ k in Nat.range(n), (1 / (3^k - 1) - 2 / (3^(2*k) - 2)) = (1 - 2 / (3^((n+1)*2)-2)) :=
by
  sorry

-- Here we formally encode the problem's structure in Lean 4,
-- but proving the final exact numerical result is noted as non-trivial
-- given the setup described in the initial problem.

end transformed_series_telescoping_series_l472_472912


namespace number_of_roots_l472_472156

-- Define the equation as a function
def equation (x : ℝ) : Prop := sqrt (9 - x) = x^2 * sqrt (9 - x)

-- Prove that the number of roots of the equation is 3
theorem number_of_roots : (finset.filter equation { x : ℝ | true}.finset).card = 3 := 
sorry

end number_of_roots_l472_472156


namespace real_solution_exists_l472_472899

theorem real_solution_exists : ∃ x : ℝ, x^3 + (x+1)^4 + (x+2)^3 = (x+3)^4 :=
sorry

end real_solution_exists_l472_472899


namespace problem_statement_l472_472443

-- Definitions
def heightsA : List ℝ := [178, 177, 179, 179, 178, 178, 177, 178, 177, 179]
def heightsB : List ℝ := [176, 177, 178, 178, 176, 178, 178, 179, 180, 180]

-- Proving the properties
theorem problem_statement : 
  median heightsB = 178 ∧
  mode heightsA = 178 ∧
  variance heightsB = 1.9 ∧
  better_team = TeamA := by
  sorry

-- Auxiliary Definitions (if needed)
def median (l : List ℝ) : ℝ := sorry -- Implementation of median
def mode (l : List ℝ) : ℝ := sorry -- Implementation of mode
def variance (l : List ℝ) : ℝ := sorry -- Implementation of variance
def better_team : ℝ := sorry -- Implementation comparing variances and concluding the best team

end problem_statement_l472_472443


namespace inequalities_l472_472355

variable {a b c : ℝ}

theorem inequalities (ha : a < 0) (hab : a < b) (hbc : b < c) :
  a^2 * b < b^2 * c ∧ a^2 * c < b^2 * c ∧ a^2 * b < a^2 * c :=
by
  sorry

end inequalities_l472_472355


namespace degree_of_poly_l472_472787

-- Define the polynomial and its degree
def inner_poly := (5 : ℝ) * (X ^ 3) + (7 : ℝ)
def poly := inner_poly ^ 10

-- Statement to prove
theorem degree_of_poly : polynomial.degree poly = 30 :=
sorry

end degree_of_poly_l472_472787


namespace find_n_range_n_p_l472_472573

-- Definition of vectors m, n, q, and p
def m : ℝ × ℝ := (1, 1)
def angle_m_n : ℝ := 3 * Real.pi / 4
def dot_m_n := -1
def q : ℝ × ℝ := (1, 0)
def n_condition (x y : ℝ) := x + y = -1 ∧ x^2 + y^2 = 1

-- Conditions A, B, C for triangle
def angle_sum := ∀ (A B C : ℝ), A + C = 2 * B

-- Vectors n and p in the given context
def p (A C : ℝ) : ℝ × ℝ := (Real.cos A, 2 * (Real.cos (C / 2))^2)
def n := [(0, -1), (-1, 0)] -- Possible values for n
def potential_range : Set ℝ := Set.Icc (Real.sqrt 2 / 2) (Real.sqrt 5 / 2)

-- Lean statements proving the conditions given:
theorem find_n (x y : ℝ) : n_condition x y → (x, y) ∈ n := sorry

theorem range_n_p (A B C : ℝ) :
  angle_sum A B C → 
  B = Real.pi / 3 → 
  ∃ n ∈ [(0, -1)], 
  let p := p A C in
  let n_p := (0, 1) + p in  -- since β is equal to α
  ∃ r, r ∈ potential_range := 
  sorry

end find_n_range_n_p_l472_472573


namespace probability_of_selection_l472_472180

theorem probability_of_selection (total_students : ℕ) (eliminated_students : ℕ) (groups : ℕ) (selected_students : ℕ)
(h1 : total_students = 1003) 
(h2 : eliminated_students = 3)
(h3 : groups = 20)
(h4 : selected_students = 50) : 
(selected_students : ℝ) / (total_students : ℝ) = 50 / 1003 :=
by
  sorry

end probability_of_selection_l472_472180


namespace percentage_increase_expenditure_l472_472327

variable (I : ℝ) -- original income
variable (E : ℝ) -- original expenditure
variable (I_new : ℝ) -- new income
variable (S : ℝ) -- original savings
variable (S_new : ℝ) -- new savings

-- a) Conditions
def initial_spend (I : ℝ) : ℝ := 0.75 * I
def income_increased (I : ℝ) : ℝ := 1.20 * I
def savings_increased (S : ℝ) : ℝ := 1.4999999999999996 * S

-- b) Definitions relating formulated conditions
def new_expenditure (I : ℝ) : ℝ := 1.20 * I - 0.3749999999999999 * I
def original_expenditure (I : ℝ) : ℝ := 0.75 * I

-- c) Proof statement
theorem percentage_increase_expenditure :
  initial_spend I = E →
  income_increased I = I_new →
  savings_increased (0.25 * I) = S_new →
  ((new_expenditure I - original_expenditure I) / original_expenditure I) * 100 = 10 := 
by 
  intros h1 h2 h3
  sorry

end percentage_increase_expenditure_l472_472327


namespace floor_factorial_div_is_even_l472_472517

def is_even (n : ℤ) : Prop :=
  ∃ k, n = 2 * k

theorem floor_factorial_div_is_even (n : ℕ) (hn : 0 < n) :
  is_even ⌊((n - 1)!) / (n * (n + 1))⌋ :=
  sorry

end floor_factorial_div_is_even_l472_472517


namespace limit_of_expression_l472_472481

noncomputable def limit_expression : ℕ → ℝ :=
  λ n, ∑ k in Finset.range (n+1), (k^2 + 3 * k + 1) / (Nat.factorial (k+2))

theorem limit_of_expression :
  filter.tendsto limit_expression filter.at_top (nhds 2) :=
  sorry

end limit_of_expression_l472_472481


namespace calories_in_lemonade_l472_472282

noncomputable def caloric_content_lemonade_mixture
  (lemon_grams : ℕ)
  (sugar_grams : ℕ)
  (water_grams : ℕ)
  (lemon_cal_per_100g : ℕ)
  (sugar_cal_per_100g : ℕ)
  (water_cal_per_100g : ℕ)
  (mixture_grams : ℕ) : ℝ :=
(lemon_grams * (lemon_cal_per_100g / 100) + sugar_grams * (sugar_cal_per_100g / 100) + water_grams * (water_cal_per_100g / 100)) / 
(real.of_nat (lemon_grams + sugar_grams + water_grams)) * real.of_nat mixture_grams

theorem calories_in_lemonade :
  caloric_content_lemonade_mixture 
    150 -- grams of lemon juice 
    100 -- grams of sugar
    600 -- grams of water
    30 -- calories per 100 grams of lemon juice
    386 -- calories per 100 grams of sugar
    0 -- calories per 100 grams of water
    300 -- grams of lemonade to check
    = 152.1 := sorry

end calories_in_lemonade_l472_472282


namespace max_salad_servings_l472_472009

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l472_472009


namespace camille_birds_count_l472_472487

theorem camille_birds_count : 
  let cardinals := 3 in
  let robins := 4 * cardinals in
  let blue_jays := 2 * cardinals in
  let sparrows := 3 * cardinals + 1 in
  cardinals + robins + blue_jays + sparrows = 31 :=
by 
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  show cardinals + robins + blue_jays + sparrows = 31
  calc 
    cardinals + robins + blue_jays + sparrows = 3 + (4 * 3) + (2 * 3) + (3 * 3 + 1) : by rfl
    ... = 3 + 12 + 6 + 10 : by rfl
    ... = 31 : by rfl

end camille_birds_count_l472_472487


namespace movie_of_the_year_condition_l472_472003

noncomputable def smallest_needed_lists : Nat :=
  let total_lists := 765
  let required_fraction := 1 / 4
  Nat.ceil (total_lists * required_fraction)

theorem movie_of_the_year_condition :
  smallest_needed_lists = 192 := by
  sorry

end movie_of_the_year_condition_l472_472003


namespace circle_area_l472_472042

-- Let PQR be an equilateral triangle with side length 8
variable (P Q R M : Point)
variable (hEquilateral : Equilateral P Q R)
variable (hSideLength : SideLength P Q R = 8)
variable (hIncenter : Incenter P Q R M)

theorem circle_area {
  (circle_area : Area (Circle P M R) = (64 * π) / 3) :
  sorry

end circle_area_l472_472042


namespace sqrt_neg3_squared_l472_472867

theorem sqrt_neg3_squared : Real.sqrt ((-3)^2) = 3 :=
by sorry

end sqrt_neg3_squared_l472_472867


namespace cos_54_eq_3_sub_sqrt_5_div_8_l472_472139

theorem cos_54_eq_3_sub_sqrt_5_div_8 :
  let x := Real.cos (Real.pi / 10) in
  let y := Real.cos (3 * Real.pi / 10) in
  y = (3 - Real.sqrt 5) / 8 :=
by
  -- Proof of the statement is omitted.
  sorry

end cos_54_eq_3_sub_sqrt_5_div_8_l472_472139


namespace equal_lengths_l472_472642

axiom cyclic {A B C D : Type} (E : Type) (circ : cyclic A B C D) (interior : E inside quadrilateral A B C D) :
  angle E A B = angle E C D ∧ angle E B A = angle E D C

axiom bisector {A B C D : Type} (E : Type) (circ : cyclic A B C D) (F G : Type) :
  line_passes E F G ∧ bisects (angle B E C) E F G ∧ intersects_circle F G

theorem equal_lengths {A B C D E F G : Type} (circ : cyclic A B C D) (interior : E inside quadrilateral A B C D) 
  (angles : angle E A B = angle E C D ∧ angle E B A = angle E D C) (line : line_passes E F G) 
  (bisect : bisects (angle B E C) E F G) (intersect : intersects_circle F G) : 
  EF = EG := 
sorry

end equal_lengths_l472_472642


namespace solve_system_part1_solve_system_part3_l472_472071

noncomputable def solution_part1 : Prop :=
  ∃ (x y : ℝ), (x + y = 2) ∧ (5 * x - 2 * (x + y) = 6) ∧ (x = 2) ∧ (y = 0)

-- Part (1) Statement
theorem solve_system_part1 : solution_part1 := sorry

noncomputable def solution_part3 : Prop :=
  ∃ (a b c : ℝ), (a + b = 3) ∧ (5 * a + 3 * c = 1) ∧ (a + b + c = 0) ∧ (a = 2) ∧ (b = 1) ∧ (c = -3)

-- Part (3) Statement
theorem solve_system_part3 : solution_part3 := sorry

end solve_system_part1_solve_system_part3_l472_472071


namespace decimal_145th_digit_of_17_div_270_l472_472410

theorem decimal_145th_digit_of_17_div_270 :
  let r := 17 / 270
  (decimal_digit r 145) = 9 :=
begin
  -- To be proved
  sorry
end

end decimal_145th_digit_of_17_div_270_l472_472410


namespace train_speed_is_144_kmph_l472_472104

-- Define the conditions
def length_of_train := 1200 -- meters
def length_of_platform := 1200 -- meters
def time_to_cross := 1 -- minute
def meter_to_km := 0.001 -- conversion factor from meters to kilometers
def minute_to_hour := 1 / 60 -- conversion factor from minutes to hours

-- Hypothesis based on conditions
def speed_of_train (v : ℝ) :=
  (length_of_train + length_of_platform) / time_to_cross = v

-- Proposition to prove
theorem train_speed_is_144_kmph (v : ℝ) 
  (h : speed_of_train v) : 
  (v * meter_to_km * 60) = 144 := 
by
  sorry

end train_speed_is_144_kmph_l472_472104


namespace lcm_1230_924_l472_472155

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 :=
by
  /- Proof steps skipped -/
  sorry

end lcm_1230_924_l472_472155


namespace discount_difference_proof_l472_472842

variables (original_price : ℝ)
def first_discounted_price : ℝ := original_price * 0.60
def second_discounted_price : ℝ := first_discounted_price * 0.90
def actual_discount_percentage : ℝ := 100 - (second_discounted_price / original_price * 100)
def claimed_discount_percentage : ℝ := 55
def discount_difference : ℝ := claimed_discount_percentage - actual_discount_percentage

theorem discount_difference_proof : discount_difference = 9 := by
  sorry

end discount_difference_proof_l472_472842


namespace polynomial_eval_at_3_is_290_l472_472403

noncomputable def polynomial_eval : Polynomial ℤ :=
  Polynomial.C 2 * Polynomial.X ^ 4 +
  Polynomial.C 3 * Polynomial.X ^ 3 +
  Polynomial.C 4 * Polynomial.X ^ 2 +
  Polynomial.C 5 * Polynomial.X - Polynomial.C 4

theorem polynomial_eval_at_3_is_290 :
  polynomial_eval.eval 3 = 290 :=
by
  sorry

end polynomial_eval_at_3_is_290_l472_472403


namespace log_sum_of_geometric_sequence_l472_472259

theorem log_sum_of_geometric_sequence (a : ℕ → ℝ) (h : ∀ n, a n > 0) 
  (H : a 3 * a 8 = 9) : 
  Real.logBase 3 (a 1) + Real.logBase 3 (a 10) = 2 := 
sorry

end log_sum_of_geometric_sequence_l472_472259


namespace num_valid_m_values_l472_472920

theorem num_valid_m_values (n : ℕ) : 
  (∃ S : finset ℕ, ∀ m ∈ S, ∃ d ∈ (finset.divisors 1806), m^2 = d + 2 ∧ nat.sqrt (d + 2) ^ 2 = d + 2) →
  (S.card = 2) := 
sorry

end num_valid_m_values_l472_472920


namespace min_value_of_sin_double_angle_l472_472733

theorem min_value_of_sin_double_angle : 
  ∃ x : ℝ, 2 * Real.sin x * Real.cos x = -1 :=
begin
  sorry
end

end min_value_of_sin_double_angle_l472_472733


namespace plate_and_rollers_acceleration_l472_472047

-- Definitions for conditions
def roller_radii := (1 : ℝ, 0.4 : ℝ)
def plate_mass := 150 -- kg
def inclination_angle := Real.arccos 0.68
def gravity_acceleration := 10 -- m/s^2

-- Theorem statement for the problem
theorem plate_and_rollers_acceleration :
  let R := roller_radii.1,
      r := roller_radii.2,
      m := plate_mass,
      α := inclination_angle,
      g := gravity_acceleration in
  ∃ (a_plate a_rollers : ℝ), a_plate = a_rollers ∧ a_plate = 4 :=
  sorry

end plate_and_rollers_acceleration_l472_472047


namespace determinant_product_l472_472999

theorem determinant_product {A B C : Matrix} 
  (hA : Matrix.det A = 3) 
  (hB : Matrix.det B = 8) 
  (hC : Matrix.det C = 5) : 
  Matrix.det (A * B * C) = 120 :=
by
  -- Proof would go here
  sorry

end determinant_product_l472_472999


namespace hyperbola_equation_l472_472074

theorem hyperbola_equation (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (h_ab_sqrt3 : a * b = real.sqrt 3) (theta : ℝ)
  (h_tan_theta : real.tan theta = real.sqrt 21 / 2)
  (Q P : ℝ × ℝ) (F2 : ℝ × ℝ) (h_QP_PF2 : dist P F2 / dist Q P = 2) :
  ∃ (x y : ℝ), (3 * x ^ 2 - y ^ 2 = 3) :=
by
  sorry

end hyperbola_equation_l472_472074


namespace max_area_rectangle_l472_472096

theorem max_area_rectangle (l w : ℕ) (h : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
sorry

end max_area_rectangle_l472_472096


namespace approx_pi_to_thousandth_l472_472362

theorem approx_pi_to_thousandth (pi_approx : Real) : π ≈ 3.142 := 
by
  sorry

end approx_pi_to_thousandth_l472_472362


namespace number_of_regions_split_by_lines_l472_472148

theorem number_of_regions_split_by_lines : 
  ∀ (x y : ℝ), (y = 3 * x) ∨ (y = (1 / 3) * x) → 
  (number_of_regions_by_lines y x) = 4 :=
by
  sorry

noncomputable def number_of_regions_by_lines (y x : ℝ) : ℕ :=
  if y = 3 * x ∨ y = (1 / 3) * x then 4 else 1

end number_of_regions_split_by_lines_l472_472148


namespace equation_of_line_l472_472732

theorem equation_of_line
  (a b : ℝ)
  (h1 : a - 4 * b - 1 = 0)
  (h2 : distance (2, 3) (2 * a - 5 * b + 9) = distance (2, 3) (2 * a - 5 * b - 7))
  (h3 : a = 4 * b + 1)
  : ∃ k m n : ℝ, k * 2 + m * 3 + n = 0 ∧ k * (-3) + m * (-1) + n = 0 ∧ k = 4 ∧ m = -5 ∧ n = 7 :=
by
  sorry

end equation_of_line_l472_472732


namespace overlapping_segments_length_l472_472740

theorem overlapping_segments_length 
    (total_length : ℝ) 
    (actual_distance : ℝ) 
    (num_overlaps : ℕ) 
    (h1 : total_length = 98) 
    (h2 : actual_distance = 83)
    (h3 : num_overlaps = 6) :
    (total_length - actual_distance) / num_overlaps = 2.5 :=
by
  sorry

end overlapping_segments_length_l472_472740


namespace correct_statement_D_l472_472972

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) * Real.cos x

theorem correct_statement_D : 
  ∃ g (x : ℝ), 
    g x = f (x - π/8) - 1/2 ∧ 
    g (-x) = -g x :=
sorry

end correct_statement_D_l472_472972


namespace cost_of_apples_l472_472747

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h1 : total_cost = 42)
  (h2 : cost_bananas = 12)
  (h3 : cost_bread = 9)
  (h4 : cost_milk = 7)
  (h5 : total_cost = cost_bananas + cost_bread + cost_milk + cost_apples) :
  cost_apples = 14 :=
by
  sorry

end cost_of_apples_l472_472747


namespace oranges_in_pyramid_l472_472446

def oranges_in_layer (n : Nat) : Nat := n * (n + 1) / 2

def total_oranges_in_pyramid (base_layer_size : Nat) : Nat :=
  Nat.recOn base_layer_size 0 (λ n acc, acc + oranges_in_layer (n + 1))

theorem oranges_in_pyramid : total_oranges_in_pyramid 6 = 56 := by
  sorry

end oranges_in_pyramid_l472_472446


namespace hyperbola_properties_l472_472169

noncomputable def hyperbola_c := (4: ℝ)
noncomputable def a := (2: ℝ)
noncomputable def b := (√12: ℝ)
noncomputable def e := hyperbola_c / a

theorem hyperbola_properties : 
  let foci := ((-4, 0), (4, 0)),
      eccentricity := 2 in
  (∀ x y: ℝ, (x^2) / 4 - (y^2) / 12 = 1 → foci = ((-hyperbola_c, 0), (hyperbola_c, 0)) ∧ e = eccentricity) :=
by
  sorry

end hyperbola_properties_l472_472169


namespace heartsuit_sum_l472_472518

namespace HeartsuitProblem

-- Define the function ⧫(x)
def Heartsuit (x : ℝ) : ℝ := (x + x^2 + x^3) / 3

theorem heartsuit_sum : Heartsuit 1 + Heartsuit (-1) + Heartsuit 2 = 16 / 3 := by
  sorry

end HeartsuitProblem

end heartsuit_sum_l472_472518


namespace problem_180_l472_472470

variables (P Q : Prop)

theorem problem_180 (h : P → Q) : ¬ (P ∨ ¬Q) :=
sorry

end problem_180_l472_472470


namespace six_digit_is_zero_l472_472819

theorem six_digit_is_zero
  (A B C D E F G : ℕ)  -- The digits of the 7-digit number.
  (hA : A = 3)  -- There are 3 zeros.
  (hB : B = 2)  -- There are 2 ones.
  (hC : C = 2)  -- There are 2 twos.
  (hD : D = 1)  -- There is 1 three.
  (hE : E = 1)  -- There is 1 four.
  (hNumber : ∀ n : ℕ, n < 7 → (match n with
    | 0 => A
    | 1 => B
    | 2 => C
    | 3 => D
    | 4 => E
    | 5 => F
    | 6 => G
    | _ => 0 end) = n)
  : F = 0 :=
sorry

end six_digit_is_zero_l472_472819


namespace number_of_integers_satisfying_inequalities_l472_472885

theorem number_of_integers_satisfying_inequalities :
  ∃ (count : ℕ), count = 3 ∧
    (∀ x : ℤ, -4 * x ≥ x + 10 → -3 * x ≤ 15 → -5 * x ≥ 3 * x + 24 → 2 * x ≤ 18 →
      x = -5 ∨ x = -4 ∨ x = -3) :=
sorry

end number_of_integers_satisfying_inequalities_l472_472885


namespace emma_missing_coins_l472_472892

theorem emma_missing_coins (x : ℤ) (h₁ : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  let missing := x - remaining
  missing / x = 1 / 9 :=
by
  sorry

end emma_missing_coins_l472_472892


namespace a_pow_5_mod_11_l472_472302

theorem a_pow_5_mod_11 (a : ℕ) : (a^5) % 11 = 0 ∨ (a^5) % 11 = 1 ∨ (a^5) % 11 = 10 :=
sorry

end a_pow_5_mod_11_l472_472302


namespace bill_sunday_miles_l472_472323

noncomputable def miles_run (B : ℕ) : ℕ :=
  B + (B + 4) + 2 * (B + 4)

theorem bill_sunday_miles : ∃ (B : ℕ), miles_run B = 28 → (B + 4) = 8 :=
begin
  use 4,
  intro h,
  dsimp [miles_run] at h,
  linarith,
end

end bill_sunday_miles_l472_472323


namespace cages_needed_l472_472434

theorem cages_needed (initial_puppies sold_puppies puppies_per_cage : ℕ) (h1 : initial_puppies = 13) (h2 : sold_puppies = 7) (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := 
by
  sorry

end cages_needed_l472_472434


namespace non_student_ticket_price_l472_472843

theorem non_student_ticket_price 
  (total_tickets : ℕ) (student_tickets : ℕ) (non_student_tickets : ℕ) 
  (student_ticket_price : ℕ) (total_revenue : ℕ): 
  total_tickets = 150 ∧ student_tickets = 90 ∧ student_ticket_price = 5 ∧
  non_student_tickets = 60 ∧ total_revenue = 930 → 
  ∃ x : ℕ, (student_tickets * student_ticket_price + non_student_tickets * x = total_revenue) ∧ x = 8 :=
by
  intros h
  rcases h with ⟨ht, hst, hstp, hnst, htr⟩
  use 8
  split
  {
    rw [hst, hstp, hnst]
    norm_num
  }
  {
    refl
  }

end non_student_ticket_price_l472_472843


namespace total_days_2003_2008_l472_472997

theorem total_days_2003_2008 : 
  let years := [2003, 2004, 2005, 2006, 2007, 2008],
  let is_leap_year (y : Nat) : Prop := (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0),
  let days_in_year (y : Nat) : Nat := if is_leap_year y then 366 else 365
  in
  (years.map days_in_year).sum = 2192 :=
by
  let years := [2003, 2004, 2005, 2006, 2007, 2008]
  let is_leap_year (y : Nat) : Prop := (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)
  let days_in_year (y : Nat) : Nat := if is_leap_year y then 366 else 365
  show (years.map days_in_year).sum = 2192
  sorry

end total_days_2003_2008_l472_472997


namespace distinct_triangles_l472_472855

theorem distinct_triangles (n1 n2 n3 : ℕ) :
  (∃ pointsAB pointsBC pointsAC: Finset ℕ,
    pointsAB.card = n1 ∧ pointsBC.card = n2 ∧ pointsAC.card = n3 ∧
    ∀ (pAB ∈ pointsAB) (pBC ∈ pointsBC) (pAC ∈ pointsAC), 
    ({pAB, pBC, pAC}.card = 3)) →
  n1 * n2 * n3 = n1 * n2 * n3 :=
by
  sorry

end distinct_triangles_l472_472855


namespace smallest_angle_in_triangle_l472_472634

theorem smallest_angle_in_triangle (x : ℝ) 
  (h_ratio : 4 * x < 5 * x ∧ 5 * x < 9 * x) 
  (h_sum : 4 * x + 5 * x + 9 * x = 180) : 
  4 * x = 40 :=
by
  sorry

end smallest_angle_in_triangle_l472_472634


namespace angle_EOA_65_degrees_l472_472271

-- Define angles and equalities
variables (A B C D E O : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited O]
variables (angle : Type) [inhabited angle]
variables (degree : angle → Type) [inhabited (degree angle)]

-- Define entities and conditions
variable (convex_quadrilateral : A → B → C → D → Prop)

-- Define circle and other relevant points
noncomputable def circle (ω : Type) [inhabited ω] (circumscribed : B → C → D → ω) (center : ω → O) : Prop := sorry

-- Define line intersections
variable (line_intersects : D → A → E → Prop)

-- Main proof statement
theorem angle_EOA_65_degrees :
  ∀ (ω : Type) [inhabited ω],
    convex_quadrilateral A B C D →
    circle ω (λ B C D => ...) (λ ω => O) →
    AB = BC ∧ BC = CD ∧ CD = DA →
    ∠ACD = 10 →
    line_intersects D A E →
    ∠EOA = 65 :=
by
  -- Skipping the actual proof
  sorry

end angle_EOA_65_degrees_l472_472271


namespace prove_trig_values_l472_472537

/-- Given angles A and B, where both are acute angles,
  and their sine values are known,
  we aim to prove the cosine of (A + B) and the measure
  of angle C in triangle ABC. -/
theorem prove_trig_values (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2)
  (sin_A_eq : Real.sin A = (Real.sqrt 5) / 5)
  (sin_B_eq : Real.sin B = (Real.sqrt 10) / 10) :
  Real.cos (A + B) = (Real.sqrt 2) / 2 ∧ (π - (A + B)) = 3 * π / 4 := by
sorry

end prove_trig_values_l472_472537


namespace tangents_converge_with_EF_common_point_l472_472546

open Geometry

theorem tangents_converge_with_EF_common_point (
  AB: Segment,
  semicircle: Circle,
  C D: Point,
  E F: Point,
  h1: ∃ (C D : Point), on_semicircle semicircle AB C ∧ on_semicircle semicircle AB D,
  h2: ∃ (E : Point), intersect_at_line (line_through A C) (line_through B D) E,
  h3: ∃ (F : Point), intersect_at_line (line_through A D) (line_through B C) F 
):
  ∃ G : Point, intersect_at_point (tangent_to semicircle C) (line_through E F) G ∧
               intersect_at_point (tangent_to semicircle D) (line_through E F) G :=
sorry

end tangents_converge_with_EF_common_point_l472_472546


namespace max_sum_first_n_terms_l472_472076

variable {a : ℕ → ℝ} -- Assume a is an arithmetic sequence where ℕ → ℝ

-- Conditions from the problem
variables (h1 : a 7 + a 8 + a 9 > 0) (h2 : a 7 + a 10 < 0)

-- Main problem statement: prove that n = 8 maximizes the sum of the first n terms
theorem max_sum_first_n_terms : 
  ∃ (n : ℕ), (n = 8) ∧ (∀ m : ℕ, m ≠ 8 → (sum (range m) a) ≤ (sum (range 8) a)) :=
by
  sorry

end max_sum_first_n_terms_l472_472076


namespace find_k_l472_472549

theorem find_k (x y k : ℝ) (h1 : x = 1) (h2 : y = 4) (h3 : k * x + y = 3) : k = -1 :=
by
  sorry

end find_k_l472_472549


namespace sum_of_two_equals_third_l472_472649

theorem sum_of_two_equals_third
  (A B C D E F D' E' F' : Type)
  [triangle : Triangle A B C]
  (sides : Sides A B C)
  (angle_bisectors : AngleBisectors A B C D E F)
  (circle_intersects : CircleIntersection D E F D' E' F') :
  (DD' + EE' = FF') ∧ (EE' + FF' = DD') ∧ (FF' + DD' = EE') :=
sorry

end sum_of_two_equals_third_l472_472649


namespace paint_mixture_green_tint_percentage_l472_472451

theorem paint_mixture_green_tint_percentage :
  ∀ (initial_volume initial_green_tint_percent additional_green_paint : ℝ),
  initial_volume = 20 ∧ 
  initial_green_tint_percent = 40 ∧ 
  additional_green_paint = 3 →
    let original_green_tint := initial_green_tint_percent / 100 * initial_volume in
    let new_green_tint := original_green_tint + additional_green_paint in
    let new_volume := initial_volume + additional_green_paint in
    abs (new_green_tint / new_volume * 100 - 48) < 1 :=
by
  intro initial_volume initial_green_tint_percent additional_green_paint
  intro h
  cases h with h_volume h_rest
  cases h_rest with h_percent h_additional
  -- Definitions and calculations can be added here for the proof.
  sorry

end paint_mixture_green_tint_percentage_l472_472451


namespace combination_identity_l472_472929

theorem combination_identity (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ m) (h3 : m ≤ n) :
  ∑ i in finset.range (k + 1), nat.choose k i * nat.choose n (m - i) = nat.choose (n + k) m :=
by
  sorry

end combination_identity_l472_472929


namespace series_sum_equals_one_l472_472174

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 : ℝ)^(2 * (k + 1)) / ((3 : ℝ)^(2 * (k + 1)) - 1)

theorem series_sum_equals_one :
  series_sum = 1 :=
sorry

end series_sum_equals_one_l472_472174


namespace num_pairs_lt_50_l472_472234

def num_pairs (ub : ℕ) : ℕ :=
  (List.finRange (ub + 1)).sum (λ m =>
    (List.finRange (ub + 1)).count (λ n => m > 0 ∧ n > 0 ∧ m * m + n * n < 50))

-- The upper bound of 7 is derived from the condition of m^2 < 50.
theorem num_pairs_lt_50 : num_pairs 7 = 32 := by
  sorry

end num_pairs_lt_50_l472_472234


namespace domain_log_function_l472_472412

noncomputable def domain_of_function := {x : ℝ | x > 64}

theorem domain_log_function :
  ∀ x : ℝ, (64 < x) ↔ (log 5 (log 3 (log 4 x))).domain :=
by
  sorry

end domain_log_function_l472_472412


namespace negation_ln_tan_l472_472734

theorem negation_ln_tan :
  (¬ ∃ x_0 : ℝ, x_0 ∈ set.Ioo 0 (π / 2) ∧ ln x_0 + tan x_0 < 0) ↔
  (∀ x : ℝ, x ∈ set.Ioo 0 (π / 2) → ln x + tan x ≥ 0) :=
sorry

end negation_ln_tan_l472_472734


namespace sum_series_even_sum_series_odd_l472_472158

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem sum_series_even (n : ℕ) (h : is_even n) : 
  (list.sum (list.map (λx, if x % 2 = 1 then x * (x + 1) else -x * (x + 1)) (list.range (n+1)))) 
  = - (n / 2) * (n + 2) := 
sorry

theorem sum_series_odd (n : ℕ) (h : is_odd n) : 
  (list.sum (list.map (λx, if x % 2 = 1 then x * (x + 1) else -x * (x + 1)) (list.range (n+1)))) 
  = ((n + 1) * (n + 1)) / 2 := 
sorry

end sum_series_even_sum_series_odd_l472_472158


namespace Walter_allocates_for_school_l472_472406

open Nat

def Walter_works_5_days_a_week := 5
def Walter_earns_per_hour := 5
def Walter_works_per_day := 4
def Proportion_for_school := 3/4

theorem Walter_allocates_for_school :
  let daily_earnings := Walter_works_per_day * Walter_earns_per_hour
  let weekly_earnings := daily_earnings * Walter_works_5_days_a_week
  let school_allocation := weekly_earnings * Proportion_for_school
  school_allocation = 75 := by
  sorry

end Walter_allocates_for_school_l472_472406


namespace regular_tetrahedron_exists_on_planes_l472_472193

variable {R : Type*} [LinearOrder R] [AddCommGroup R] [Module ℝ R]
variable {d1 d2 d3 : R}

noncomputable def exists_tetrahedron_on_planes (S1 S2 S3 S4 : R) 
  (planes_parallel : ∀ {X Y : R}, X ≠ Y → S1 = S2 ∧ S2 = S3 ∧ S3 = S4) 
  (distances : S1 + d1 = S2 ∧ S2 + d2 = S3 ∧ S3 + d3 = S4) : Prop :=
  ∃ (A B C D : R), 
    (A ∈ S1) ∧ 
    (B ∈ S2) ∧ 
    (C ∈ S3) ∧ 
    (D ∈ S4) ∧ 
    (∃ (a b c d : R), 
      a = A ∧ b = B ∧ c = C ∧ d = D ∧ 
      is_regular_tetrahedron a b c d)

theorem regular_tetrahedron_exists_on_planes {S1 S2 S3 S4 : R} 
  (planes_parallel : ∀ {X Y : R}, X ≠ Y → S1 = S2 ∧ S2 = S3 ∧ S3 = S4) 
  (distances : S1 + d1 = S2 ∧ S2 + d2 = S3 ∧ S3 + d3 = S4) : 
  exists_tetrahedron_on_planes S1 S2 S3 S4 planes_parallel distances :=
  sorry

end regular_tetrahedron_exists_on_planes_l472_472193


namespace apple_percentage_after_adding_fruits_l472_472311

theorem apple_percentage_after_adding_fruits
  (x y z w : ℕ)
  (h₁ : x + y = 30) 
  (h₂ : z + w = 12) 
  (h₃ : x = 2 * y) 
  (h₄ : w = 3 * z) :
  (29 : ℚ) / (42 : ℚ) * (100 : ℚ) ≈ 69.05 := by
sorr_CODEC_MISSING

end apple_percentage_after_adding_fruits_l472_472311


namespace area_triangle_BOC_l472_472648

-- Definitions based on conditions in the problem
variables (A B C O K : Point)
variables (AC AB : ℝ)

-- Given conditions
def triangle_ABC := triangle A B C
def side_AC := AC = 14
def side_AB := AB = 6
def circle_with_diameter_AC := Circle O (AC / 2) 
def intersection_K := intersects circle_with_diameter_AC (line B C) K
def angle_condition := angle B A K = angle A C B

theorem area_triangle_BOC (h1 : triangle_ABC)
                         (h2 : side_AC)
                         (h3 : side_AB)
                         (h4 : circle_with_diameter_AC)
                         (h5 : intersection_K)
                         (h6 : angle_condition) :
                         area (triangle B O C) = 21 := 
sorry

end area_triangle_BOC_l472_472648


namespace number_of_female_students_selected_is_20_l472_472087

noncomputable def number_of_female_students_to_be_selected
(total_students : ℕ) (female_students : ℕ) (students_to_be_selected : ℕ) : ℕ :=
students_to_be_selected * female_students / total_students

theorem number_of_female_students_selected_is_20 :
  number_of_female_students_to_be_selected 2000 800 50 = 20 := 
by
  sorry

end number_of_female_students_selected_is_20_l472_472087


namespace tangent_line_at_1_1_of_x_pow_x_l472_472930

theorem tangent_line_at_1_1_of_x_pow_x :
  ∀ x : ℝ, 0 < x →
  let y := x^x in
  let f := λ x : ℝ, x in
  let φ := λ x : ℝ, x in
  deriv y 1 = 1 ∧ (y = x := f 1, φ 1) :=
by
  sorry

end tangent_line_at_1_1_of_x_pow_x_l472_472930


namespace transform_curve_point_coordinates_l472_472396

theorem transform_curve_point_coordinates :
  (∃ C : ℝ → ℝ × ℝ,  
      (∀ x y, x^2 + (1/4) * y^2 = 1 → (let p := C (real.sqrt (x^2 + y^2)) in p.1 = x ∧ p.2 = (1/2) * y)) ∧
      (∀ α, let ⟨x, y⟩ := C α in x = real.cos α ∧ y = real.sin α) ∧
      (∀ D : ℝ × ℝ, let ⟨dx, dy⟩ := D in 
        ((∃ α, dx = real.cos α ∧ dy = real.sin α) ∧
         ((dx = real.sqrt(2) / 2 ∧ dy = real.sqrt(2) / 2) ∨
          (dx = -real.sqrt(2) / 2 ∧ dy = -real.sqrt(2) / 2))))) := 
sorry

end transform_curve_point_coordinates_l472_472396


namespace dot_product_solution_l472_472580

namespace VectorProof

def vector_a : ℤ × ℤ := (1, 2)
def vector_b : ℤ × ℤ := (1, 1)

theorem dot_product_solution :
  let c := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
  let d := (vector_a.1 - 2 * vector_b.1, vector_a.2 - 2 * vector_b.2)
  c.1 * d.1 + c.2 * d.2 = -2 :=
by
  intro c d
  have h1 : c = (2, 3) := by simp [c, vector_a, vector_b]
  have h2 : d = (-1, 0) := by simp [d, vector_a, vector_b]
  rw [h1, h2]
  simp
  sorry

end VectorProof

end dot_product_solution_l472_472580


namespace carter_made_shots_in_fifth_game_l472_472872

-- Definitions based on conditions
def initial_made : ℕ := 15
def initial_attempted : ℕ := 45
def next_attempted : ℕ := 15
def new_shooting_average : ℚ := 0.4

-- The theorem stating Carter made 9 shots in his fifth game
theorem carter_made_shots_in_fifth_game :
  (initial_made + (next_attempted * new_shooting_average).to_nat - initial_made) = 9 :=
by
  sorry

end carter_made_shots_in_fifth_game_l472_472872


namespace max_servings_l472_472038

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l472_472038


namespace platform_length_is_225_l472_472774

def length_of_platform (length_trainA : ℕ) (speed_trainA_kmph : ℕ) (time_trainA_cross : ℕ) : ℕ :=
let speed_trainA_mps := speed_trainA_kmph * 1000 / 3600 in
let distance_covered_trainA := speed_trainA_mps * time_trainA_cross in
distance_covered_trainA - length_trainA

theorem platform_length_is_225 :
  length_of_platform 175 36 40 = 225 :=
by
  /- Proof would be placed here. -/
  sorry

end platform_length_is_225_l472_472774


namespace jenna_tanning_time_l472_472281

theorem jenna_tanning_time (max_minutes_per_month : ℕ) (first_week_monday_friday_minutes : ℕ) (first_week_wednesday_minutes : ℕ) (second_week_monday_friday_minutes : ℕ) : 
  max_minutes_per_month = 200 →
  first_week_monday_friday_minutes = 30 →
  first_week_wednesday_minutes = 15 →
  second_week_monday_friday_minutes = 40 →
  let first_week_total := 2 * first_week_monday_friday_minutes + first_week_wednesday_minutes in
  let second_week_total := 2 * second_week_monday_friday_minutes in
  let total_first_two_weeks := first_week_total + second_week_total in
  max_minutes_per_month - total_first_two_weeks = 45 :=
by
  intro h_max h_monday_friday h_wednesday h_monday_friday_2
  let first_week_total := 2 * first_week_monday_friday_minutes + first_week_wednesday_minutes
  let second_week_total := 2 * second_week_monday_friday_minutes
  let total_first_two_weeks := first_week_total + second_week_total
  have h_first_week : first_week_total = 75 := by sorry
  have h_second_week : second_week_total = 80 := by sorry
  have h_total_first_two_weeks : total_first_two_weeks = 155 := by sorry
  have h_remaining_minutes : max_minutes_per_month - total_first_two_weeks = 45 := by sorry
  exact h_remaining_minutes

end jenna_tanning_time_l472_472281


namespace gypsum_element_weights_l472_472836

noncomputable def gypsum_composition := 
  let Ca := 20
  let O := 8
  let S := 16
  let H := 1
  let total_weight := 100
  let molecular_weight_CaO := Ca + O
  let molecular_weight_SO3 := S + 3 * O
  let molecular_weight_2H2O := 2 * (2 * H + O)
  let total_molecular_weight := molecular_weight_CaO + molecular_weight_SO3 + molecular_weight_2H2O
  
  let weight_Ca := (Ca / total_molecular_weight) * total_weight
  let weight_O := ((O + 3 * O + 2 * 8) / total_molecular_weight) * total_weight
  let weight_S := (S / total_molecular_weight) * total_weight
  let weight_H := (2 * H / total_molecular_weight) * total_weight

  (weight_Ca, weight_O, weight_S, weight_H)

theorem gypsum_element_weights : gypsum_composition = (22.727, 54.545, 18.182, 2.273) :=
  by sorry

end gypsum_element_weights_l472_472836


namespace other_asymptote_l472_472325

theorem other_asymptote (h1 : ∃ c, ∀ y, (y, 3) ∈ hyperbola -> (c, y) ∈ line y = 4x - 3 ∧ 
                  ∃ h: y = 4x - 3, ∃ center: (3, 9) ∈ hyperbola ∧ 
                  ∀ animouto h, (∀ asym: y = -4x + 21, asym)) :
  other_asymptote(hyperbola, animouto) :=
sorry

end other_asymptote_l472_472325


namespace max_servings_possible_l472_472017

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l472_472017


namespace probability_squares_overlap_l472_472768

theorem probability_squares_overlap :
  (∃ sq1x sq1y sq2x sq2y,
     sq1x ∈ {0, 1, 2, 3, 4} ∧ sq1y ∈ {0, 1, 2, 3, 4} ∧
     sq2x ∈ {0, 1, 2, 3, 4} ∧ sq2y ∈ {0, 1, 2, 3, 4} ∧
     (sq1x ≠ sq2x ∨ sq1y ≠ sq2y)) →
  ∃ n d, (n = 529) ∧ (d = 625) ∧ (n / d = (529 : ℚ) / 625) := 
sorry

end probability_squares_overlap_l472_472768


namespace gcd_182_98_l472_472905

theorem gcd_182_98 : Nat.gcd 182 98 = 14 :=
by
  -- Provide the proof here, but as per instructions, we'll use sorry to skip it.
  sorry

end gcd_182_98_l472_472905


namespace knicks_advance_seven_games_l472_472360

noncomputable def binom : ℕ → ℕ → ℕ 
| 0, _ := 1
| _, 0 := 1
| n, k := binom (n - 1) (k - 1) * n / k

noncomputable def knicks_win_probability : ℚ :=
  let knicks_win := (1/4 : ℚ)
  let bulls_win := (3/4 : ℚ)
  let tied_after_six_games := binom 6 3 * (knicks_win^3) * (bulls_win^3)
  let win_seventh_game := knicks_win
  tied_after_six_games * win_seventh_game

theorem knicks_advance_seven_games :
  knicks_win_probability = 135 / 4096 :=
by
  sorry

end knicks_advance_seven_games_l472_472360


namespace solve_for_ABC_l472_472958

noncomputable def problem_statement (A B C : ℝ) (a b c : ℝ) : Prop :=
  (a = 7) ∧ 
  (∃ (bc : ℝ), bc = 40) ∧
  (∃ (area : ℝ), area = 10 * Real.sqrt 3) ∧
  (sin A)^2 = (sin B)^2 + (sin C)^2 - (sin B * sin C)

theorem solve_for_ABC:
  ∃ (A b c : ℝ), 
  A = Real.pi / 3 ∧ 
  (b = 8 ∧ c = 5 ∨ b = 5 ∧ c = 8) ∧ 
  problem_statement A _ _ 7 b c :=
by
  sorry

end solve_for_ABC_l472_472958


namespace additional_amount_l472_472320

noncomputable def additional_amount_borrowed (P₁ : ℝ) (R : ℝ) (T₁ : ℝ) (A : ℝ) : ℝ :=
  let SI₁ := P₁ * R * T₁ / 100
  let P₂ := P₁ + SI₁
  let SI₂ := (P₂ + X) * R * 3 / 100
  let Total := P₂ + X + SI₂
  in  X

theorem additional_amount (P₁ R T₁ A : ℝ) (h₁ : P₁ = 10000) (h₂ : R = 0.06) (h₃ : T₁ = 2) (h₄ : A = 27160) :
  additional_amount_borrowed P₁ R T₁ A = 11817.29 :=
by
  sorry

end additional_amount_l472_472320


namespace projection_correct_l472_472838

open Matrix

def proj (u v : Vector 2 ℝ) := (u ⬝ v / u ⬝ u) • u

theorem projection_correct :
  proj (λ i, ![1, 0] i) (λ i, ![3, -3] i) = (λ i, ![3, 0] i) :=
by
  let u := λ i, ![1, 0] i
  let v := λ i, ![3, -3] i
  let result := λ i, ![3, 0] i
  have u_dot_u : u ⬝ u = 1 := sorry
  have u_dot_v : u ⬝ v = 3 := sorry
  show proj u v = result
  rw [proj, u_dot_u, u_dot_v]
  simp
  sorry

end projection_correct_l472_472838


namespace bisects_segment_BC_l472_472475

-- Definitions:
variables (A B C M K D T : Point)
variable [incircle : Circle Ω (Triangle A B C)]
variable [is_midpoint_of_arc : MidpointOfArc M (Arc (Circumcircle Ω) B C)]
variable [external_angle_bisector : ExternalAngleBisector BAC K (Line B C)]
variable [perpendicular_point : PerpendicularPoint A (Line B C) D]
variable [equal_segments : Segment D M = Segment A M]
variable [circumcircle_ADK : Circle (Circumcircle (Triangle A D K)) (Points A K D T)]
variable [intersection_with_omega : IntersectionPoint (Circumcircle Ω) (Circumcircle (Triangle A D K)) T]

-- The conclusion statement to be proved:
theorem bisects_segment_BC : Bisects (Segment A T) (Segment B C) :=
sorry

end bisects_segment_BC_l472_472475


namespace JamieEarnings_l472_472654

theorem JamieEarnings :
  ∀ (rate_per_hour : ℕ) (days_per_week : ℕ) (hours_per_day : ℕ) (weeks : ℕ),
    rate_per_hour = 10 →
    days_per_week = 2 →
    hours_per_day = 3 →
    weeks = 6 →
    rate_per_hour * days_per_week * hours_per_day * weeks = 360 :=
by
  intros rate_per_hour days_per_week hours_per_day weeks
  intros hrate hdays hhours hweeks
  rw [hrate, hdays, hhours, hweeks]
  norm_num
  sorry

end JamieEarnings_l472_472654


namespace sum_ij_eq_zero_l472_472916

variables {a : Fin m → ℝ} {v : Fin m → ℝ^n}

/-- m, n are positive integers -/ 
variable (m n : ℕ)

/-- The vectors v_i are pairwise distinct -/ 
variable (h_dist : ∀ i j, i ≠ j → v i ≠ v j)

/-- The given condition about the sum being zero -/ 
variable (h_condition : ∀ i, ∑ j in {j | j ≠ i}, a j * ((v j - v i) / ((‖v j - v i‖ : ℝ) ^ 3)) = 0)

theorem sum_ij_eq_zero (h1 : m > 0) (h2 : n > 0) : 
    ∑ i in Finset.range m, ∑ j in Finset.range m, 
    if i < j then (a i * a j / ‖v j - v i‖) else 0 = 0 := 
sorry

end sum_ij_eq_zero_l472_472916


namespace whole_number_M_l472_472462

theorem whole_number_M (M : ℤ) (hM : 9 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) : M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end whole_number_M_l472_472462


namespace divide_into_three_groups_l472_472858

-- Define a delegate as a vertex in a graph
structure Symposium (V : Type) where
  acquainted : V → V → Prop
  acquainted_symm : ∀ {a b}, acquainted a b → acquainted b a
  acquainted_irrefl : ∀ {a}, ¬acquainted a a
  exists_acquainted : ∀ (a : V), ∃ (b : V), acquainted a b
  exists_third_delegates : ∀ (a b : V), ∃ (c : V), ¬(acquainted c a ∧ acquainted c b)

-- State the problem in lean
theorem divide_into_three_groups (V : Type) [finite V] (S : Symposium V) :
  ∃ (G1 G2 G3 : set V), 
  (∀ (v : V), v ∈ G1 ∨ v ∈ G2 ∨ v ∈ G3) ∧
  (∀ (v ∈ G1), ∃ (w ∈ G1), S.acquainted v w) ∧
  (∀ (v ∈ G2), ∃ (w ∈ G2), S.acquainted v w) ∧
  (∀ (v ∈ G3), ∃ (w ∈ G3), S.acquainted v w) := 
sorry

end divide_into_three_groups_l472_472858


namespace intersection_M_N_union_M_N_l472_472230

-- Conditions
def M : Set ℚ := {x | -2 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- Statements to prove
theorem intersection_M_N :
  M ∩ N = {x : ℚ | -1 ≤ x ∧ x ≤ 1} :=
sorry

theorem union_M_N :
  M ∪ N = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
sorry

end intersection_M_N_union_M_N_l472_472230


namespace cos_sum_of_angles_l472_472182

theorem cos_sum_of_angles (α β : Real) (h1 : Real.sin α = 4/5) (h2 : (π/2) < α ∧ α < π) 
(h3 : Real.cos β = -5/13) (h4 : 0 < β ∧ β < π/2) : 
  Real.cos (α + β) = -33/65 := 
by
  sorry

end cos_sum_of_angles_l472_472182


namespace determine_sum_with_one_cell_l472_472719

-- Definition of the problem conditions
def table : Type := matrix (fin 5) (fin 7) ℤ

def zero_sum_rectangles (t : table) : Prop :=
  ∀ (i j : fin 4) (k l : fin 2), 
    (t i j + t (i.succ) j + t i (j.succ) + t (i.succ) (j.succ) + 
     t i (j.succ.succ) + t (i.succ) (j.succ.succ)) = 0 ∧
    (t i j + t i (j.succ) + t (i.succ) j + t (i.succ) (j.succ) + 
     t (i.succ.succ) j + t (i.succ.succ) (j.succ)) = 0

-- The problem statement in Lean 4
theorem determine_sum_with_one_cell (t : table) (zero_sum : zero_sum_rectangles t) :
  ∃ (cell_value : ℤ), 
    ∀ i j, t i j = cell_value → 
    ∀ S, S = cell_value * 35 := 
sorry

end determine_sum_with_one_cell_l472_472719


namespace expression_eval_l472_472126

theorem expression_eval :
    (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
    (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * 5040 = 
    (5^128 - 4^128) * 5040 := by
  sorry

end expression_eval_l472_472126


namespace sufficient_but_not_necessary_condition_l472_472538

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, |x + 1| + |x - 1| ≥ m
def proposition_q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - 2 * m * x₀ + m^2 + m - 3 = 0

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (proposition_p m → proposition_q m) ∧ ¬ (proposition_q m → proposition_p m) :=
sorry

end sufficient_but_not_necessary_condition_l472_472538


namespace maximum_sum_of_distances_l472_472696

theorem maximum_sum_of_distances (P V : Point ℝ) (d : ℕ → Point ℝ) (hP : ∑ i in range 100, dist P (d i) = 1000) :
  (∑ i in range 100, dist V (d i)) ≤ 99000 :=
sorry

end maximum_sum_of_distances_l472_472696


namespace angle_of_inclination_of_line_l472_472168

noncomputable def angle_of_inclination (x y : ℝ → ℝ) : ℝ :=
  real.arctan ((deriv y) 0 / (deriv x) 0)

theorem angle_of_inclination_of_line
  (t : ℝ) (x y : ℝ → ℝ)
  (hx : ∀ t, x t = 5 - 3 * t)
  (hy : ∀ t, y t = 3 + (√3) * t) :
  angle_of_inclination x y = 150 :=
by
  -- conditions as facts
  have hx' : (deriv x) 0 = -3 := by sorry
  have hy' : (deriv y) 0 = √3 := by sorry
  -- using arc tangent formula for angle of inclination
  have slope : (deriv y) 0 / (deriv x) 0 = - (√3 / 3) := by sorry
  have angle : real.arctan (- (√3 / 3)) = 150 := by sorry
  -- final proof
  rw [angle_of_inclination, hx', hy', slope, angle]
  exact rfl

end angle_of_inclination_of_line_l472_472168


namespace max_area_rectangle_l472_472095

theorem max_area_rectangle (l w : ℕ) (h : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
sorry

end max_area_rectangle_l472_472095


namespace increase_expenditure_by_10_percent_l472_472334

variable (I : ℝ) (P : ℝ)
def E := 0.75 * I
def I_new := 1.20 * I
def S_new := 1.50 * (I - E)
def E_new := E * (1 + P / 100)

theorem increase_expenditure_by_10_percent :
  (E_new = 0.75 * I * (1 + P / 100)) → P = 10 :=
by
  sorry

end increase_expenditure_by_10_percent_l472_472334


namespace part1_part2_l472_472672

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a+1)*x + a

theorem part1 (a x : ℝ) :
  (a < 1 ∧ f x a < 0 ↔ a < x ∧ x < 1) ∧
  (a = 1 ∧ ¬(f x a < 0)) ∧
  (a > 1 ∧ f x a < 0 ↔ 1 < x ∧ x < a) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 < x → f x a ≥ -1) → a ≤ 3 :=
sorry

end part1_part2_l472_472672


namespace simplify_expression_l472_472435

-- Define the mathematical terms needed
def two_pow_two := (2: ℝ)^2
def cube_root_neg_one := real.cbrt (-1: ℝ)
def abs_neg_sqrt_two := abs (-real.sqrt 2)
def sqrt_nine := real.sqrt 9

-- The main statement to prove the given expression equals the simplified result
theorem simplify_expression : two_pow_two + cube_root_neg_one - abs_neg_sqrt_two + sqrt_nine = 6 - real.sqrt 2 :=
by
  sorry

end simplify_expression_l472_472435


namespace probability_of_drawing_3_red_1_blue_l472_472441

open_locale big_operators

/-
We will define the necessary elements and express the problem as a theorem in Lean.
-/

-- Define the number of red and blue balls and the total number of balls
def number_of_red_balls : ℕ := 10
def number_of_blue_balls : ℕ := 5
def total_number_of_balls : ℕ := number_of_red_balls + number_of_blue_balls

-- Define the number of balls drawn
def number_of_draws : ℕ := 4

-- Define the specific event we are interested in (3 red balls and 1 blue ball)
def favorable_outcomes : ℕ :=
  (nat.choose 10 3) * (nat.choose 5 1)

-- Define the total possible outcomes
def total_outcomes : ℕ :=
  nat.choose 15 4

-- The probability of the specific event
def probability_of_event : ℚ :=
  favorable_outcomes / total_outcomes

-- The theorem which states that this probability is indeed 40/91
theorem probability_of_drawing_3_red_1_blue :
  probability_of_event = 40 / 91 :=
by
  sorry

end probability_of_drawing_3_red_1_blue_l472_472441


namespace num_points_P_l472_472372

noncomputable def point_on_ellipse (P : Point ℝ) : Prop :=
  (P.x^2 / 16) + (P.y^2 / 9) = 1

noncomputable def line_eq (P : Point ℝ) : Prop :=
  (P.x / 4) + (P.y / 3) = 1

def intersect_points (A B : Point ℝ) : Prop :=
  line_eq A ∧ line_eq B ∧ point_on_ellipse A ∧ point_on_ellipse B

theorem num_points_P (A B : Point ℝ) (area : ℝ) :
  intersect_points A B ∧ (∃ P : Point ℝ, point_on_ellipse P ∧ triangle_area A B P = area)
  → area = 12 → 
     card {P : Point ℝ | point_on_ellipse P ∧ triangle_area A B P = 12} = 2 :=
sorry

end num_points_P_l472_472372


namespace point_outside_circle_l472_472553

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) : a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l472_472553


namespace red_paint_quarts_l472_472523

theorem red_paint_quarts (r g w : ℕ) (ratio_rw : r * 5 = w * 4) (w_quarts : w = 15) : r = 12 :=
by 
  -- We provide the skeleton of the proof here: the detailed steps are skipped (as instructed).
  sorry

end red_paint_quarts_l472_472523


namespace num_pairs_lt_50_l472_472235

def num_pairs (ub : ℕ) : ℕ :=
  (List.finRange (ub + 1)).sum (λ m =>
    (List.finRange (ub + 1)).count (λ n => m > 0 ∧ n > 0 ∧ m * m + n * n < 50))

-- The upper bound of 7 is derived from the condition of m^2 < 50.
theorem num_pairs_lt_50 : num_pairs 7 = 32 := by
  sorry

end num_pairs_lt_50_l472_472235


namespace seating_arrangements_l472_472266

theorem seating_arrangements (n : ℕ) (h : n = 6) : 
  ∃ k, k = 48 ∧ (∀ (x y : fin n), x ≠ y → ∃ s : finset (fin (n-1) * 2), 
  (s.card = (n-2)! * 2) ∧ (∀ (i : fin (n-1) * 2), i ∈ s)) :=
by {
  intros,
  have h₁ : fact (n-1) = factorial (n-2) := sorry,
  have h₂ : fact (n-2) * 2 = 48 := sorry,
  use 48,
  split,
  { exact h₂, },
  { intros x y hxy,
    use finset.range ((n-1) * 2),
    split,
    { rw finset.card_range,
      exact h₁, },
    { intros i hi,
      exact finset.mem_range.mpr hi, }, },
  sorry
}

end seating_arrangements_l472_472266


namespace prism_cut_out_l472_472119

theorem prism_cut_out (x y : ℕ)
  (H1 : 15 * 5 * 4 - y * 5 * x = 120)
  (H2 : x < 4) :
  x = 3 ∧ y = 12 :=
sorry

end prism_cut_out_l472_472119


namespace solution_set_2_exp_2x_minus_1_lt_2_l472_472173

theorem solution_set_2_exp_2x_minus_1_lt_2 (x : ℝ) : 2^(2*x - 1) < 2 ↔ x < 1 :=
by
  sorry

end solution_set_2_exp_2x_minus_1_lt_2_l472_472173


namespace min_value_of_2x_plus_y_l472_472610

theorem min_value_of_2x_plus_y (x y : ℝ) (h : log 2 x + log 2 y = 3) : 2 * x + y = 8 := by
  sorry

end min_value_of_2x_plus_y_l472_472610


namespace symmetric_points_ratio_l472_472638

-- Define the problem conditions
variables {m n : ℝ}
def point_A := (m + 4, -1 : ℝ × ℝ)
def point_B := (1, n - 3 : ℝ × ℝ)
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- Lean 4 statement of the equivalence
theorem symmetric_points_ratio (h : symmetric_about_origin point_A point_B) :
  m / n = -5 / 4 :=
by sorry

end symmetric_points_ratio_l472_472638


namespace vertex_on_y_axis_l472_472384

theorem vertex_on_y_axis (d : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -10 in
  let h : ℝ := -b / (2 * a) in
  h = 0 → d = 25 :=
by
  let a : ℝ := 1
  let b : ℝ := -10
  let h : ℝ := -b / (2 * a)
  assume h_eq_zero : h = 0
  sorry

end vertex_on_y_axis_l472_472384


namespace range_reciprocal_sum_l472_472637

theorem range_reciprocal_sum {α : ℝ} (cos : ℝ) (sin : ℝ) 
(P : ℝ × ℝ) (hP : P = (√3 / 2, 3 / 2)) 
(hline : ∀ t : ℝ, P = (√3 / 2 + t * cos α, 3 / 2 + t * sin α)) 
(hcurve : ∀ x y : ℝ, (x^2 + y^2 = 1)) :
  ∃ (PM PN : ℝ), ∀ M N : ℝ × ℝ, (M ≠ N ∧ (M ∈ (x, y)) ∧ (N ∈ (x, y)) →
  (sqrt(2) < (1/PM) + (1/PN) ∧ (1/PM) + (1/PN) ≤ sqrt(3))) :=
by
  sorry

end range_reciprocal_sum_l472_472637


namespace y_intercept_of_line_l472_472378

theorem y_intercept_of_line (m : ℝ) (x₀ : ℝ) (y₀ : ℝ) (h_slope : m = -3) (h_intercept : (x₀, y₀) = (7, 0)) : (0, 21) = (0, (y₀ - m * x₀)) :=
by
  sorry

end y_intercept_of_line_l472_472378


namespace solve_for_n_l472_472352

theorem solve_for_n (n : ℝ) : 9^n * 9^n * 9^n = 81^4 → n = 8 / 3 :=
by
  intro h,
  sorry

end solve_for_n_l472_472352


namespace third_consecutive_even_sum_52_l472_472430

theorem third_consecutive_even_sum_52
  (x : ℤ)
  (h : x + (x + 2) + (x + 4) + (x + 6) = 52) :
  x + 4 = 14 :=
by
  sorry

end third_consecutive_even_sum_52_l472_472430


namespace inequality_proof_l472_472608

theorem inequality_proof (x y : ℝ) (h : 3^x - 3^y < 4^(-x) - 4^(-y)) : x < y ∧ 2^(-y) < 2^(-x) :=
by
  sorry

end inequality_proof_l472_472608


namespace triangle_area_is_rational_l472_472844

-- Definition of the area of a triangle given vertices with integer coordinates
def triangle_area (x1 x2 x3 y1 y2 y3 : ℤ) : ℚ :=
0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The theorem stating that the area of a triangle formed by points with integer coordinates is rational
theorem triangle_area_is_rational (x1 x2 x3 y1 y2 y3 : ℤ) :
  ∃ (area : ℚ), area = triangle_area x1 x2 x3 y1 y2 y3 :=
by
  sorry

end triangle_area_is_rational_l472_472844


namespace series_fraction_reduction_l472_472853

noncomputable def series_sum : ℚ :=
  (999/1999) + (999/1999) * (998/1998) + (999/1999) * (998/1998) * (997/1997) + 
  -- continuing the pattern
  ∑ (x : ℕ) in finset.range (999), (999/1999) * ∏ (k : ℕ) in finset.range x, (1 - k/999).

theorem series_fraction_reduction (a b : ℕ) : 
  (a : ℚ) / (b : ℚ) = series_sum → (a, b) = (999, 1001) :=
by
  sorry

end series_fraction_reduction_l472_472853


namespace log15_12_eq_l472_472201

-- Goal: Define the constants and statement per the identified conditions and goal
variable (a b : ℝ)
#check Real.log
#check Real.logb

-- Math conditions
def lg2_eq_a := Real.log 2 = a
def lg3_eq_b := Real.log 3 = b

-- Math proof problem statement
theorem log15_12_eq : lg2_eq_a a → lg3_eq_b b → Real.logb 15 12 = (2 * a + b) / (1 - a + b) :=
by intros h1 h2; sorry

end log15_12_eq_l472_472201


namespace pairwise_perpendicular_not_coplanar_l472_472574

-- Define pairwise perpendicular condition for three lines
structure Lines (α : Type) :=
  (l1 l2 l3 : α)
  (perp_l1_l2 : l1 ⊥ l2)
  (perp_l1_l3 : l1 ⊥ l3)
  (perp_l2_l3 : l2 ⊥ l3)

-- Theorem statement to prove that three pairwise perpendicular lines cannot all be coplanar.
theorem pairwise_perpendicular_not_coplanar {α : Type} [euclidean_space α] (lines : Lines α) :
  ¬ coplanar lines.l1 lines.l2 lines.l3 :=
sorry

end pairwise_perpendicular_not_coplanar_l472_472574


namespace number_of_acceptable_ages_l472_472428

theorem number_of_acceptable_ages (avg_age : ℤ) (std_dev : ℤ) (a b : ℤ) (h_avg : avg_age = 10) (h_std : std_dev = 8)
    (h1 : a = avg_age - std_dev) (h2 : b = avg_age + std_dev) :
    b - a + 1 = 17 :=
by {
    sorry
}

end number_of_acceptable_ages_l472_472428


namespace max_servings_l472_472033

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l472_472033


namespace pascal_triangle_entries_l472_472869

theorem pascal_triangle_entries :
  (∑ n in Finset.range (26 - 6 + 1), (6 + n)) = 336 := by
  sorry

end pascal_triangle_entries_l472_472869


namespace circle_inscribed_isosceles_trapezoid_l472_472117

theorem circle_inscribed_isosceles_trapezoid (r a c : ℝ) : 
  (∃ base1 base2 : ℝ,  2 * a = base1 ∧ 2 * c = base2) →
  (∃ O : ℝ, O = r) →
  r^2 = a * c :=
by
  sorry

end circle_inscribed_isosceles_trapezoid_l472_472117


namespace counterexample_exists_l472_472043

theorem counterexample_exists : ∃ (a b c : ℤ), (a ∣ (b * c) ∧ ¬ a ∣ b ∧ ¬ a ∣ c) :=
by {
  use 4,
  use 2,
  use 2,
  split,
  { exact dvd.intro 1 rfl },
  split,
  { intro h,
    have : 2 / 4 = 0.5 := rfl,
    exact this, -- contradiction
    },
  { intro h,
    have : 2 / 4 = 0.5 := rfl,
    exact this } -- contradiction
}

end counterexample_exists_l472_472043


namespace relationship_m_n_l472_472937

variables {a b : ℝ}

theorem relationship_m_n (h1 : |a| ≠ |b|) (m : ℝ) (n : ℝ)
  (hm : m = (|a| - |b|) / |a - b|)
  (hn : n = (|a| + |b|) / |a + b|) :
  m ≤ n :=
by sorry

end relationship_m_n_l472_472937


namespace all_equal_l472_472286

variable {n : ℕ}
variable {a : Fin (2 * n + 1) → ℤ}
variable H : ∀ i, (∃ (s t : Finset (Fin (2 * n + 1))) (h : s ∩ t = ∅) (h' : s ∪ t = Finset.univ \ {i}), (∑ x in s, a x) = (∑ x in t, a x))

theorem all_equal : ∀ i j, a i = a j := by
  sorry

end all_equal_l472_472286


namespace right_triangle_hypotenuse_length_l472_472268

theorem right_triangle_hypotenuse_length 
    (AB AC x y : ℝ) 
    (P : AB = x) (Q : AC = y) 
    (ratio_AP_PB : AP / PB = 1 / 3) 
    (ratio_AQ_QC : AQ / QC = 2 / 1) 
    (BQ_length : BQ = 18) 
    (CP_length : CP = 24) : 
    BC = 24 := 
by 
  sorry

end right_triangle_hypotenuse_length_l472_472268


namespace rhind_papyrus_denominator_l472_472270

theorem rhind_papyrus_denominator :
  ∃ x : ℕ, (2:ℚ) / 73 = 1 / 60 + 1 / 219 + 1 / 292 + 1 / x ∧ x = 365 :=
by
  use 365
  by simp [←eq_div_iff_mul_eq, show x = 365 by sorry]

end rhind_papyrus_denominator_l472_472270


namespace projection_of_m_onto_n_l472_472203

variables (A B C : Type) [AddCommGroup A] [Module ℝ A]
variable (side_length : ℝ)
variable (m n : A)
variable (equilateral_triangle : ∀ (u v : A), ‖ u ‖ = side_length ∧ ‖ v ‖ = side_length ∧ ⟨ u, v ⟩  = ‖ u ‖ * ‖ v ‖ * real.cos (2 * real.pi / 3))

theorem projection_of_m_onto_n (h_triangle : equilateral_triangle) : 
  vector_proj m n = - (1/2) • n :=
by {
  sorry
}

end projection_of_m_onto_n_l472_472203


namespace carrots_per_bundle_l472_472090

theorem carrots_per_bundle (potatoes_total: ℕ) (potatoes_in_bundle: ℕ) (price_per_potato_bundle: ℝ) 
(carrot_total: ℕ) (price_per_carrot_bundle: ℝ) (total_revenue: ℝ) (carrots_per_bundle : ℕ) :
potatoes_total = 250 → potatoes_in_bundle = 25 → price_per_potato_bundle = 1.90 → 
carrot_total = 320 → price_per_carrot_bundle = 2 → total_revenue = 51 →
((carrots_per_bundle = carrot_total / ((total_revenue - (potatoes_total / potatoes_in_bundle) 
    * price_per_potato_bundle) / price_per_carrot_bundle))  ↔ carrots_per_bundle = 20) := by
  sorry

end carrots_per_bundle_l472_472090


namespace max_dist_vasya_l472_472699

theorem max_dist_vasya (d : ℕ → ℕ → ℕ) (P V : ℕ) (friends : Fin 100) :
  (∑ i in (Finset.filter (λ x, x ≠ P) (Finset.univ : Finset (Fin 100))), d P i) = 1000 →
  (∃ x, x = 99 * 1000) →
  (∑ i in  (Finset.filter (λ x, x ≠ V) (Finset.univ : Finset (Fin 100))), d V i ≤ 99000) :=
by
  sorry

end max_dist_vasya_l472_472699


namespace find_leftmost_x_l472_472385

open Real

noncomputable def shoelace_area (m : ℕ) : ℝ :=
  let A := 1 / 2 * abs (
    exp m * (m + 1) + exp (m + 1) * (m + 2) + exp (m + 2) * (m + 3) + exp (m + 3) * m -
    (exp (m + 1) * m + exp (m + 2) * (m + 1) + exp (m + 3) * (m + 2) + exp m * (m + 3))
  )
  in A

theorem find_leftmost_x :
  ∃ m : ℕ, shoelace_area m = exp 1 ∧ m = 3 := sorry

end find_leftmost_x_l472_472385


namespace tom_weekly_fluid_intake_l472_472503

-- Definitions based on the conditions.
def soda_cans_per_day : ℕ := 5
def ounces_per_can : ℕ := 12
def water_ounces_per_day : ℕ := 64
def days_per_week : ℕ := 7

-- The mathematical proof problem statement.
theorem tom_weekly_fluid_intake :
  (soda_cans_per_day * ounces_per_can + water_ounces_per_day) * days_per_week = 868 := 
by
  sorry

end tom_weekly_fluid_intake_l472_472503


namespace minimize_total_distance_l472_472366

-- Definitions based on conditions
def distance_between_villages : ℝ := 3
def students_in_A : ℕ := 300
def students_in_B : ℕ := 200
def distance_A_to_school (x : ℝ) : ℝ := x
def distance_B_to_school (x : ℝ) : ℝ := distance_between_villages - x

-- The total distance function f(x)
def total_distance (x : ℝ) : ℝ :=
  students_in_A * distance_A_to_school(x) + students_in_B * distance_B_to_school(x)

-- The Lean statement to minimize the total distance traveled by the students
theorem minimize_total_distance :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ distance_between_villages ∧ (∀ y : ℝ, 0 ≤ y ∧ y ≤ distance_between_villages → total_distance(x) ≤ total_distance(y))) :=
begin
  use 0,
  split,
  { linarith },
  split,
  { linarith },
  { intros y hy,
    simp [total_distance, distance_A_to_school, distance_B_to_school],
    linarith }
end

end minimize_total_distance_l472_472366


namespace geometric_sum_first_k_terms_l472_472145

theorem geometric_sum_first_k_terms (k : ℕ) : 
  let a := k^3 in
  let r := 2 in
  (a * ((r^k - 1) / (r - 1))) = k^3 * (2^k - 1) := by
sorry

end geometric_sum_first_k_terms_l472_472145


namespace rewrite_neg_multiplication_as_exponent_l472_472709

theorem rewrite_neg_multiplication_as_exponent :
  -2 * 2 * 2 * 2 = - (2^4) :=
by
  sorry

end rewrite_neg_multiplication_as_exponent_l472_472709


namespace lean_proof_statement_l472_472680

variables {A B C D E F G : Point}
variable {Γ : Circle}
variable {triangle_ABC : Triangle}
variables (P Q R M N O : Point)

noncomputable def acute_triangle (A B C : Point) : Prop := ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90
noncomputable def circumcircle (ABC : Triangle) : Circle := sorry -- definition of circumcircle
noncomputable def perp_bisector (p1 p2 : Point) : Line := sorry -- definition of perpendicular bisector
noncomputable def minor_arc (Γ : Circle) (A B : Point) : Arc := sorry -- definition of minor arc

theorem lean_proof_statement :
  acute_triangle A B C →
  let Γ := circumcircle triangle_ABC in
  point_on_segment D A B →
  point_on_segment E A C →
  D = E →
  perp_bisector B D = perp_bisector (circumcircle triangle_ABC) (minor_arc Γ A B) →
  perp_bisector C E = perp_bisector (circumcircle triangle_ABC) (minor_arc Γ A C) →
  parallel D E F G :=
begin
  sorry
end

end lean_proof_statement_l472_472680


namespace y_coord_of_tangents_intersection_l472_472288

noncomputable def point (x : ℝ) : ℝ × ℝ :=
  (x, x^2 + x)

def tangent_slope (x : ℝ) : ℝ :=
  2 * x + 1

def tangent_at (x : ℝ) : ℝ → ℝ :=
  let m := tangent_slope x
  λ y, m * (y - x) + x^2 + x

theorem y_coord_of_tangents_intersection {a b: ℝ} 
  (ha: point a ∈ {p : ℝ × ℝ | p.snd = p.fst ^ 2 + p.fst}) 
  (hb: point b ∈ {p : ℝ × ℝ | p.snd = p.fst ^ 2 + p.fst}) 
  (m_perp: tangent_slope a * tangent_slope b = -1):
  let x := a + b
  let y := tangent_at a x
  y = -1 / 4 :=
sorry

end y_coord_of_tangents_intersection_l472_472288


namespace bottles_difference_l472_472833

theorem bottles_difference :
  let regular_soda := 81
  let diet_soda := 60
  in regular_soda - diet_soda = 21 := by
  sorry

end bottles_difference_l472_472833


namespace concurrency_of_lines_l472_472533

noncomputable def Triangle (A B C : Point) : Prop :=  A ≠ B ∧ B ≠ C ∧ C ≠ A

noncomputable def Circumcircle (A B C : Point) : Circle := sorry

noncomputable def InnerCircleTangent (A B C : Point) (Γ : Circle) (side1 side2 : Line) : Circle := sorry

noncomputable def TangencyPoint (incircle : Circle) (circumcircle: Circle) : Point := sorry

theorem concurrency_of_lines {A B C : Point} 
  (h_triangle : Triangle A B C) 
  (Γ : Circle) 
  (h_circumcircle : Circumcircle A B C = Γ)
  (Γ_A Γ_B Γ_C : Circle)
  (h_inner_circleA : Γ_A = InnerCircleTangent A B C Γ (Line.mk A B) (Line.mk A C))
  (h_inner_circleB : Γ_B = InnerCircleTangent A B C Γ (Line.mk B A) (Line.mk B C))
  (h_inner_circleC : Γ_C = InnerCircleTangent A B C Γ (Line.mk C A) (Line.mk C B)) 
  (A' B' C' : Point)
  (h_tangentA : A' = TangencyPoint Γ_A Γ)
  (h_tangentB : B' = TangencyPoint Γ_B Γ)
  (h_tangentC : C' = TangencyPoint Γ_C Γ) : 
  ∃ X : Point, 
    (Line.mk A A') = (Line.mk X X) ∧
    (Line.mk B B') = (Line.mk X X) ∧
    (Line.mk C C') = (Line.mk X X) := 
sorry

end concurrency_of_lines_l472_472533


namespace sin_identity_l472_472934

variable (α : ℝ)
axiom alpha_def : α = Real.pi / 7

theorem sin_identity : (Real.sin (3 * α)) ^ 2 - (Real.sin α) ^ 2 = Real.sin (2 * α) * Real.sin (3 * α) := 
by 
  sorry

end sin_identity_l472_472934


namespace travel_ratio_l472_472661

theorem travel_ratio
    (d_s : ℕ) (d_e : ℕ) (x : ℕ) (total_distance : ℕ)
    (south : d_s = 40)
    (east : d_e = 60)
    (total : d_s + d_e + d_e * x = total_distance)
    (journey : total_distance = 220) :
    x = 2 :=
by {
    rw [south, east, journey] at total,
    linarith,
}

end travel_ratio_l472_472661


namespace parallelogram_diagonals_equal_rectangle_l472_472339

theorem parallelogram_diagonals_equal_rectangle 
  {A B C D O : Type} 
  [IsParallelogram A B C D]
  (h1 : AC = BD)
  : IsRectangle A B C D :=
sorry

end parallelogram_diagonals_equal_rectangle_l472_472339


namespace s_neq_t_if_Q_on_DE_l472_472300

-- Conditions and Definitions
noncomputable def DQ (x : ℝ) := x
noncomputable def QE (x : ℝ) := 10 - x
noncomputable def FQ := 5 * Real.sqrt 3
noncomputable def s (x : ℝ) := (DQ x) ^ 2 + (QE x) ^ 2
noncomputable def t := 2 * FQ ^ 2

-- Lean 4 Statement
theorem s_neq_t_if_Q_on_DE (x : ℝ) : s x ≠ t :=
by
  sorry -- Provided proof step to be filled in

end s_neq_t_if_Q_on_DE_l472_472300


namespace integral_solution_l472_472906

noncomputable def integral_expression : ℝ → ℝ :=
  λ x, (4 * x^3 + 24 * x^2 + 20 * x - 28) / ((x + 3)^2 * (x^2 + 2 * x + 2))

noncomputable def indefinite_integral := 
  λ x, (-4 / (x + 3) + 2 * Real.log (x^2 + 2 * x + 2) - 8 * Real.arctan (x + 1))

theorem integral_solution :
  ∀ x : ℝ, ∃ C : ℝ, ∫ u in 0..x, integral_expression u = indefinite_integral x + C :=
by
  sorry

end integral_solution_l472_472906


namespace max_soap_boxes_in_carton_l472_472422

theorem max_soap_boxes_in_carton
  (L_carton W_carton H_carton : ℕ)
  (L_soap_box W_soap_box H_soap_box : ℕ)
  (vol_carton := L_carton * W_carton * H_carton)
  (vol_soap_box := L_soap_box * W_soap_box * H_soap_box)
  (max_soap_boxes := vol_carton / vol_soap_box) :
  L_carton = 25 → W_carton = 42 → H_carton = 60 →
  L_soap_box = 7 → W_soap_box = 6 → H_soap_box = 5 →
  max_soap_boxes = 300 :=
by
  intros hL hW hH hLs hWs hHs
  sorry

end max_soap_boxes_in_carton_l472_472422


namespace vectors_perpendicular_l472_472520

variables {V : Type*} [inner_product_space ℝ V]

theorem vectors_perpendicular {a b : V} (ha : a ≠ 0) (hb : b ≠ 0) (h : ∥a + b∥ = ∥a - b∥) : ⟪a, b⟫ = 0 :=
sorry

end vectors_perpendicular_l472_472520


namespace lower_limit_of_arun_weight_l472_472471

-- Given conditions for Arun's weight
variables (W : ℝ)
variables (avg_val : ℝ)

-- Define the conditions
def arun_weight_condition_1 := W < 72
def arun_weight_condition_2 := 60 < W ∧ W < 70
def arun_weight_condition_3 := W ≤ 67
def arun_weight_avg := avg_val = 66

-- The math proof problem statement
theorem lower_limit_of_arun_weight 
  (h1: arun_weight_condition_1 W) 
  (h2: arun_weight_condition_2 W) 
  (h3: arun_weight_condition_3 W) 
  (h4: arun_weight_avg avg_val) :
  ∃ (lower_limit : ℝ), lower_limit = 65 :=
sorry

end lower_limit_of_arun_weight_l472_472471


namespace number_of_divisors_not_divisible_by_3_l472_472590

def prime_factorization (n : ℕ) : Prop :=
  n = 2 ^ 2 * 3 ^ 2 * 5

def is_not_divisible_by (n d : ℕ) : Prop :=
  ¬ (d ∣ n)

def positive_divisors_not_divisible_by_3 (n : ℕ) : ℕ :=
  (finset.range (2 + 1)).filter (λ a, ∀ d : ℕ, is_not_divisible_by (2 ^ a * d) 3).card

theorem number_of_divisors_not_divisible_by_3 :
  prime_factorization 180 → positive_divisors_not_divisible_by_3 180 = 6 :=
by
  intro h
  sorry

end number_of_divisors_not_divisible_by_3_l472_472590


namespace num_divisors_not_div_by_3_l472_472598

theorem num_divisors_not_div_by_3 : 
  let n := 180 in
  let prime_factorization_180 := factorization 180 in
  (prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180) →
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)} in
  divisors_not_div_by_3.card = 6 :=
by 
  let n := 180
  let prime_factorization_180 := factorization n
  have h_factorization : prime_factorization_180.factors = [2, 2, 3, 3, 5] ∧ prime_factorization_180.prod = 180 := -- proof ommitted
    sorry
  let divisors_not_div_by_3 := {d in divisors n | ¬(3 ∣ d)}
  have h_card : divisors_not_div_by_3.card = 6 := -- proof ommitted
    sorry
  exact h_card

end num_divisors_not_div_by_3_l472_472598


namespace impossible_to_transform_circle_to_square_l472_472280

theorem impossible_to_transform_circle_to_square 
  (r : ℝ) 
  (circle : { x // x ∈ metric.sphere (0 : ℂ) r } ) :
  ¬ ( ∃ (cut_positions : Finset (ℝ × ℝ)),
      ∃ (square : { y // y ∈ (set.range (λ (x : ℝ), (x, 0)) ∪ set.range (λ (y : ℝ), (1, y)) ∪ set.range (λ (x : ℝ), (x, 1)) ∪ set.range (λ (y : ℝ), (0, y)) )}),
      (∀ (cut1 cut2 ∈ cut_positions), cut1 ≠ cut2) ∧
      (area_of_square square = π * r ^ 2) ∧
      reassemble_circle_into_square_with_cuts circle cut_positions square ) :=
sorry

end impossible_to_transform_circle_to_square_l472_472280


namespace degree_of_polynomial10_l472_472786

-- Definition of the degree function for polynomials.
def degree (p : Polynomial ℝ) : ℕ := p.natDegree

-- Given condition: the degree of the polynomial 5x^3 + 7 is 3.
def polynomial1 := (Polynomial.C 5) * (Polynomial.X ^ 3) + (Polynomial.C 7)
axiom degree_poly1 : degree polynomial1 = 3

-- Statement to prove:
theorem degree_of_polynomial10 : degree (polynomial1 ^ 10) = 30 :=
by
  sorry

end degree_of_polynomial10_l472_472786


namespace tan_alpha_eq_neg_3_4_l472_472524

theorem tan_alpha_eq_neg_3_4 (α : ℝ) (h₁ : sin α + cos α = -1 / 5) (h₂ : 0 < α ∧ α < π) : tan α = -3 / 4 :=
sorry

end tan_alpha_eq_neg_3_4_l472_472524


namespace trigonometric_identity_l472_472756

theorem trigonometric_identity :
  sin (160 * Real.pi / 180) * sin (10 * Real.pi / 180) - cos (20 * Real.pi / 180) * cos (10 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end trigonometric_identity_l472_472756


namespace find_the_number_l472_472818

theorem find_the_number :
  ∃ X : ℝ, (66.2 = (6.620000000000001 / 100) * X) ∧ X = 1000 :=
by
  sorry

end find_the_number_l472_472818


namespace symmetry_axis_l472_472159

noncomputable def y_func (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

theorem symmetry_axis : ∃ a : ℝ, (∀ x : ℝ, y_func (a - x) = y_func (a + x)) ∧ a = Real.pi / 8 :=
by
  sorry

end symmetry_axis_l472_472159


namespace correct_conclusions_l472_472417

def chi_square_test : Prop :=
  ∀ χ² : ℝ, ∀ P : ℝ, (χ² ≥ 6.635) ∧ (P ≥ 6.635) → P ≈ 0.01 → 
    (99% confidence that the two categorical variables are related)

def linear_regression_correlation : Prop :=
  ∀ r : ℝ, (abs r → abs closer r → 1 → stronger correlation) ∧ 
    (smaller abs r → weaker correlation)

def regression_line_through_point : Prop :=
  ∀ b x a : ℝ, ∀ (x̄ ȳ : ℝ), (regression equation passes through (x̄, ȳ)) → 
    (regression line equation  ^ bx + a passes through point (A (x̄, ȳ)))

def regression_line_calculation_incorrect : Prop :=
  ∀ x : ℝ, (0.5 * 200 - 85 = 15) → (estimation not definite)

theorem correct_conclusions :
  chi_square_test ∧ linear_regression_correlation ∧ regression_line_through_point :=
by
  sorry

end correct_conclusions_l472_472417


namespace circular_seating_arrangements_l472_472820

theorem circular_seating_arrangements :
  let n := 10 in
  let reference_point := 1 in
  (n - reference_point)! = 362880 :=
by
  -- proof goes here
  sorry

end circular_seating_arrangements_l472_472820


namespace area_outside_circles_l472_472341

theorem area_outside_circles : 
  ∀ (EF FG : ℝ) (radius_E radius_F radius_G radius_H : ℝ),
    EF = 4 -> FG = 6 -> 
    radius_E = 2 -> radius_F = 3 -> radius_G = 1 -> radius_H = 1.5 -> 
    (24 - (radius_E^2 * π + radius_F^2 * π + radius_G^2 * π + radius_H^2 * π) / 4) ≈ 11.24 :=
by
  intros EF FG radius_E radius_F radius_G radius_H 
  intros hEF hFG hE hF hG hH
  subst hEF
  subst hFG
  subst hE
  subst hF
  subst hG
  subst hH
  simp
  linarith [(4 ≈ 3.14) -- approx pi with 3.14
            (radius_E^2 * 3.14 ≈ 12.56), -- approx circle area
            (radius_F^2 * 3.14 ≈ 28.26),
            (radius_G^2 * 3.14 ≈ 3.14),
            (radius_H^2 * 3.14 ≈ 7.065)]
  sorry -- proof needed for numerical approximation

end area_outside_circles_l472_472341


namespace excircles_angle_bisector_l472_472278

theorem excircles_angle_bisector (A B C M N : Type) [Inhabited A] 
  (triangle : Triangle A B C) 
  (ωB ωC ωB' ωC': Circle)
  (tangent_AC: Tangent ωB AC) 
  (tangent_AB: Tangent ωC AB)
  (reflection_ωB: Reflection ωB M AC = ωB')
  (reflection_ωC: Reflection ωC N AB = ωC') :
  Line (IntersectionPoints ωB' ωC') ∠bisects perimeter triangle :=
sorry

end excircles_angle_bisector_l472_472278


namespace fraction_of_phones_l472_472120

-- The total number of valid 8-digit phone numbers (b)
def valid_phone_numbers_total : ℕ := 5 * 10^7

-- The number of valid phone numbers that begin with 5 and end with 2 (a)
def valid_phone_numbers_special : ℕ := 10^6

-- The fraction of phone numbers that begin with 5 and end with 2
def fraction_phone_numbers_special : ℚ := valid_phone_numbers_special / valid_phone_numbers_total

-- Prove that the fraction of such phone numbers is 1/50
theorem fraction_of_phones : fraction_phone_numbers_special = 1 / 50 := by
  sorry

end fraction_of_phones_l472_472120


namespace c_investment_period_l472_472846

variable (x m : ℝ)

def total_profit : ℝ := 18900

def a_share : ℝ := 6300

def a_investment : ℝ := x * 12

def b_investment : ℝ := 2 * x * 6

def c_investment (m : ℝ) : ℝ := 3 * x * (12 - m)

def total_investment (x m : ℝ) : ℝ := a_investment x + b_investment x + c_investment x m

def a_share_fraction : ℝ := a_share / total_profit

theorem c_investment_period : 
  ∀ (x : ℝ), a_share_fraction = 1 / 3 → total_investment x m = 3 * a_investment x → m = 4 := sorry

end c_investment_period_l472_472846


namespace log_eval_l472_472894

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval :
  let a := log_base 5 625
  let b := log_base 5 25
  let c := log_base 5 5
  a - b + c = 3 :=
by
  let a := log_base 5 625
  let b := log_base 5 25
  let c := log_base 5 5
  have h1 : 5 ^ a = 625 := by sorry
  have h2 : 5 ^ b = 25 := by sorry
  have h3 : 5 ^ c = 5 := by sorry
  have ha : a = 4 := by sorry
  have hb : b = 2 := by sorry
  have hc : c = 1 := by sorry
  show a - b + c = 3, from by sorry

end log_eval_l472_472894


namespace probability_divisible_by_5_l472_472382

def is_three_digit_integer (M : ℕ) : Prop :=
  100 ≤ M ∧ M < 1000

def ones_digit_is_4 (M : ℕ) : Prop :=
  (M % 10) = 4

theorem probability_divisible_by_5 (M : ℕ) (h1 : is_three_digit_integer M) (h2 : ones_digit_is_4 M) :
  (∃ p : ℚ, p = 0) :=
by
  sorry

end probability_divisible_by_5_l472_472382


namespace max_servings_l472_472029

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l472_472029


namespace exists_k_l472_472663

-- Define P as a non-constant homogeneous polynomial with real coefficients
def homogeneous_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ (a b : ℝ), P (a * a) (b * b) = (a * a) ^ n * (b * b) ^ n

-- Define the main problem
theorem exists_k (P : ℝ → ℝ → ℝ) (hP : ∃ n : ℕ, homogeneous_polynomial n P)
  (h : ∀ t : ℝ, P (Real.sin t) (Real.cos t) = 1) :
  ∃ k : ℕ, ∀ x y : ℝ, P x y = (x^2 + y^2) ^ k :=
sorry

end exists_k_l472_472663


namespace plate_and_roller_acceleration_l472_472046

noncomputable def m : ℝ := 150
noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def alpha : ℝ := Real.arccos 0.68

theorem plate_and_roller_acceleration :
  let sin_alpha_half := Real.sin (alpha / 2)
  sin_alpha_half = 0.4 →
  plate_acceleration == 4 ∧ direction == Real.arcsin 0.4 ∧ rollers_acceleration == 4 :=
by
  sorry

end plate_and_roller_acceleration_l472_472046


namespace ratio_areas_of_triangles_l472_472453

theorem ratio_areas_of_triangles 
  (a b : ℝ) (n m : ℕ) :
  (n > 0) → (m > 0) →
  let area_A := (1 / 2) * (b / n) * (a / 2)
  let area_B := (1 / 2) * (b / m) * (a / 2)
  ratio := area_A / area_B
  ratio = (m : ℝ) / n :=
  sorry

end ratio_areas_of_triangles_l472_472453


namespace price_roger_cookie_l472_472179

-- Define the conditions given in the problem
def radius_art := 2
def price_per_art_cookie := 50
def num_art_cookies := 10
def total_earnings_art := num_art_cookies * price_per_art_cookie

-- Define the areas
def area_art_cookie := Real.pi * radius_art ^ 2
def total_dough_used_art := num_art_cookies * area_art_cookie
def num_roger_cookies := 8
def area_roger_cookie := total_dough_used_art / num_roger_cookies

-- Define Roger's cookie side and price
def side_roger_cookie := Real.sqrt area_roger_cookie
def price_per_roger_cookie := total_earnings_art / num_roger_cookies

-- State the theorem
theorem price_roger_cookie : round_above price_per_roger_cookie = 63 := 
by sorry

end price_roger_cookie_l472_472179


namespace matthew_points_l472_472688

theorem matthew_points:
  ∀ (n_baskets total_baskets: ℕ) (points_per_basket shawn_points: ℕ),
    points_per_basket = 3 
    → total_baskets = 5
    → shawn_points = 6 
    → n_baskets * points_per_basket = total_baskets * points_per_basket - shawn_points 
    → n_baskets = 9 := 
by
  intros n_baskets total_baskets points_per_basket shawn_points h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  have : n_baskets * 3 = 5 * 3 - 6 := h4
  linarith

end matthew_points_l472_472688
