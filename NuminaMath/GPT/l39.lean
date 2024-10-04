import Mathlib

namespace employees_six_years_or_more_percentage_l39_39490

theorem employees_six_years_or_more_percentage 
  (Y : ℕ)
  (Total : ℝ := (3 * Y:ℝ) + (4 * Y:ℝ) + (7 * Y:ℝ) - (2 * Y:ℝ) + (6 * Y:ℝ) + (1 * Y:ℝ))
  (Employees_Six_Years : ℝ := (6 * Y:ℝ) + (1 * Y:ℝ))
  : Employees_Six_Years / Total * 100 = 36.84 :=
by
  sorry

end employees_six_years_or_more_percentage_l39_39490


namespace steven_ships_boxes_l39_39684

-- Translate the conditions into Lean definitions and state the theorem
def truck_weight_limit : ℕ := 2000
def truck_count : ℕ := 3
def pair_weight : ℕ := 10 + 40
def boxes_per_pair : ℕ := 2

theorem steven_ships_boxes :
  ((truck_weight_limit / pair_weight) * boxes_per_pair * truck_count) = 240 := by
  sorry

end steven_ships_boxes_l39_39684


namespace count_valid_n_in_range_l39_39115

theorem count_valid_n_in_range :
  (∃ count : Nat, count = List.length (List.filter (λ n, (2^(2*n) + 2^n + 5) % 7 = 0) (List.range (2011 - 2000)).map (λ x, x + 2000)))
  ∧ count = 4 :=
by
  sorry

end count_valid_n_in_range_l39_39115


namespace fat_per_serving_l39_39276

theorem fat_per_serving (servings : ℕ) (half_cup_fat : ℝ) (total_fat : ℝ) (fat_per_serving : ℝ) :
  servings = 4 →
  half_cup_fat = 88 / 2 →
  total_fat = half_cup_fat →
  fat_per_serving = total_fat / servings →
  fat_per_serving = 11 :=
by
  intro h1 h2 h3 h4
  rw [h2, h3, h4]
  norm_num
  sorry

end fat_per_serving_l39_39276


namespace relationship_between_a_and_b_l39_39536

theorem relationship_between_a_and_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
    (h₃ : ∀ x : ℝ, |(3 * x + 1) - 4| < a → |x - 1| < b) : a ≥ 3 * b :=
by
  -- Applying the given conditions, we want to demonstrate that a ≥ 3b.
  sorry

end relationship_between_a_and_b_l39_39536


namespace soft_drink_cost_in_euros_l39_39706

theorem soft_drink_cost_in_euros :
  let price_per_12pack := 2.99
  let tax_rate := 0.075
  let deposit_fee_per_can := 0.05
  let exchange_rate := 0.85
  let cost_per_12pack := price_per_12pack * (1 + tax_rate) + 12 * deposit_fee_per_can
  let cost_per_can_usd := cost_per_12pack / 12
  let cost_per_can_eur := cost_per_can_usd * exchange_rate
  (Real.round(cost_per_can_eur * 100) / 100 = 0.27)
:= sorry

end soft_drink_cost_in_euros_l39_39706


namespace necessary_but_not_sufficient_condition_l39_39953

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient_condition (a : ℝ) : (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) := 
  by 
    sorry

end necessary_but_not_sufficient_condition_l39_39953


namespace cone_volume_filled_88_8900_percent_l39_39023

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39023


namespace find_matrix_N_l39_39102

def eq_matrix_2x2 (N : Matrix (Fin 2) (Fin 2) ℤ) : Prop :=
  N.mulVec (λ _ => [2, 0]) = (λ _ => [4, 14]) ∧
  N.mulVec (λ _ => [-2, 10]) = (λ _ => [6, -34])

theorem find_matrix_N :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℤ), eq_matrix_2x2 N ∧ N = (λ _ => [ [2, 1], [7, -2] ]) :=
by
  sorry

end find_matrix_N_l39_39102


namespace hairstylist_earnings_per_week_l39_39411

theorem hairstylist_earnings_per_week :
  let cost_normal := 5
  let cost_special := 6
  let cost_trendy := 8
  let haircuts_normal := 5
  let haircuts_special := 3
  let haircuts_trendy := 2
  let days_per_week := 7
  let daily_earnings := cost_normal * haircuts_normal + cost_special * haircuts_special + cost_trendy * haircuts_trendy
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 413 := sorry

end hairstylist_earnings_per_week_l39_39411


namespace angle_BIC_120_l39_39548

theorem angle_BIC_120 (A B C I : Type*) [Incenter I A B C] (h1 : ∠ABC = 2 * ∠ACB) 
                       (h2 : dist A B = dist C I) : ∠BIC = 120 :=
by sorry

end angle_BIC_120_l39_39548


namespace yard_fraction_occupied_by_flower_beds_l39_39047

theorem yard_fraction_occupied_by_flower_beds (w : ℝ) (h_w : w = 5) :
  let a := 10 / 2 in
  let triangle_area := (1 / 2) * a^2 in
  let total_flower_beds_area := 2 * triangle_area in
  let yard_area := 30 * w in
  total_flower_beds_area / yard_area = 1 / 6 :=
by
  sorry

end yard_fraction_occupied_by_flower_beds_l39_39047


namespace hairstylist_earnings_per_week_l39_39412

theorem hairstylist_earnings_per_week :
  let cost_normal := 5
  let cost_special := 6
  let cost_trendy := 8
  let haircuts_normal := 5
  let haircuts_special := 3
  let haircuts_trendy := 2
  let days_per_week := 7
  let daily_earnings := cost_normal * haircuts_normal + cost_special * haircuts_special + cost_trendy * haircuts_trendy
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 413 := sorry

end hairstylist_earnings_per_week_l39_39412


namespace find_range_m_l39_39939

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x + 2

noncomputable def g (x : ℝ) (a : ℝ) (m : ℝ) : ℝ := f x a + m * x

theorem find_range_m (m : ℝ) (a : ℝ) (interval : set ℝ)
  (h1: interval = Ioo (-3) (a - 1)) 
  (f_has_maxval4 : ∃ x, f x a = 4) 
  (g_has_minleq_m_sub_1 : ∀ x ∈ interval, g x a m ≤ m - 1) :
  m ∈ Icc (-9) (-15/4) := 
  sorry

end find_range_m_l39_39939


namespace sequence_converges_to_sqrt_c_l39_39182

noncomputable def recurrence_relation (x_n c : ℝ) : ℝ := (x_n^2 + c) / (2 * x_n)

theorem sequence_converges_to_sqrt_c (c x1 : ℝ) (h_initial : abs (x1 + real.sqrt c) < 1)
  (h_c_pos : 0 < c) (h_x1_pos : 0 < x1) :
  ∀ (x : ℕ → ℝ) (h_recurrence : ∀ n, x (n+1) = recurrence_relation (x n) c),
  filter.tendsto x filter.at_top (𝓝 (real.sqrt c)) :=
sorry

end sequence_converges_to_sqrt_c_l39_39182


namespace finite_squares_cover_black_cells_l39_39389

-- Define the problem setting: N black cells on an infinite grid
def infinite_grid := ℕ → ℕ → Prop -- a function to represent an infinite grid, where cells are either black or not

variable (N : ℕ)
variable (black_cells : finite_set ℕ ℕ) -- finite set of black cells on the grid

-- Define the two conditions described
def all_black_cells_within_squares (squares : finite_set (set (ℕ × ℕ))) : Prop :=
  ∀ (cell ∈ black_cells), ∃ (K ∈ squares), cell ∈ K

def valid_area_fraction (square : set (ℕ × ℕ)) : Prop :=
  let area := square.card in
  let black_area := (black_cells ∩ square).card in
  (1 / 5 : ℝ) ≤ (black_area : ℝ) / (area : ℝ) ∧ (black_area : ℝ) / (area : ℝ) ≤ 4 / 5

def all_squares_valid_area_fraction (squares : finite_set (set (ℕ × ℕ))) : Prop :=
  ∀ (K ∈ squares), valid_area_fraction K

-- The main theorem stating the existence of these squares
theorem finite_squares_cover_black_cells : ∃ (squares : finite_set (set (ℕ × ℕ))),
  all_black_cells_within_squares black_cells squares ∧ all_squares_valid_area_fraction squares :=
sorry

end finite_squares_cover_black_cells_l39_39389


namespace min_internal_sides_l39_39488

open Nat

theorem min_internal_sides (n : ℕ) (h1 : 3 ∣ n) :
  ∀ (grid : Fin n × Fin n → Fin 3),
  (∀ color : Fin 3, (∑ i in Fin n, ∑ j in Fin n, if grid (i, j) = color then 1 else 0) = (n * n) / 3) →
  ∃ (internal_sides : ℕ),
  internal_sides = 66 :=
by
  sorry

end min_internal_sides_l39_39488


namespace find_a_l39_39916

def quadratic_function (y : ℝ → ℝ) := ∃ a b c : ℝ, ∀ x : ℝ, y x = a * x^2 + b * x + c

def coordinates := [(-2, 11), (-1, 6), (0, 3), (1, 2), (2, 3), (3, 6), (4, 11)]

theorem find_a (y : ℝ → ℝ) (h_quad : quadratic_function y) :
  coordinates.nth 1 = some (-1, 6) :=
begin
  sorry -- Proof not required
end

end find_a_l39_39916


namespace integer_points_distribution_l39_39425

open Nat

theorem integer_points_distribution (p : ℕ) (hp : Prime p) : 
  ∀ (A B O : Point) (hA : A = (p, 0)) (hB : B = (0, p)) (hO : O = (0, 0)), 
  (∀ i, 1 ≤ i ∧ i ≤ p - 1 → (i, p - i) ∈ LineSegment A B)
  → ∀ C ∈ (PointsInsideTriangle O A B),
    ∃! k : ℕ, 0 < k ∧ k < p - 1 ∧ 
    ∀s ∈ SubTriangles (O A B) k, 
      IntegerPointsInTriangle C = IntegerPointsInTriangle (SubTriangle k) := 
sorry

end integer_points_distribution_l39_39425


namespace product_of_primes_l39_39294

theorem product_of_primes (n : ℕ) (h : n > 0) : 
  ∃ (k : ℕ) (p : Fin k → ℕ), (∀ i : Fin k, Prime (p i)) ∧ (∏ i in Finset.univ, p i) = n := by 
  sorry

end product_of_primes_l39_39294


namespace hairstylist_weekly_earnings_l39_39409

-- Definition of conditions as given in part a)
def cost_normal_haircut := 5
def cost_special_haircut := 6
def cost_trendy_haircut := 8

def number_normal_haircuts_per_day := 5
def number_special_haircuts_per_day := 3
def number_trendy_haircuts_per_day := 2

def working_days_per_week := 7

-- The goal is to prove that the hairstylist's weekly earnings equal to 413 dollars
theorem hairstylist_weekly_earnings : 
  (number_normal_haircuts_per_day * cost_normal_haircut +
  number_special_haircuts_per_day * cost_special_haircut +
  number_trendy_haircuts_per_day * cost_trendy_haircut) * 
  working_days_per_week = 413 := 
by sorry -- We use "by sorry" to skip the proof

end hairstylist_weekly_earnings_l39_39409


namespace men_absent_l39_39036

theorem men_absent (n : ℕ) (d1 d2 : ℕ) (x : ℕ) 
  (h1 : n = 22) 
  (h2 : d1 = 20) 
  (h3 : d2 = 22) 
  (hc : n * d1 = (n - x) * d2) : 
  x = 2 := 
by {
  sorry
}

end men_absent_l39_39036


namespace midpoint_distance_l39_39861

-- Define the endpoints of the segment
def pointA : ℝ × ℝ := (10, -3)
def pointB : ℝ × ℝ := (-4, 7)

-- Define the formula to compute the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the formula to compute the Euclidean distance between two points
def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the midpoint of the segment with endpoints (10, -3) and (-4, 7)
def M : ℝ × ℝ := midpoint pointA pointB

-- State the theorem to prove that the distance from the midpoint M to endpoint (10, -3) is sqrt 74
theorem midpoint_distance : dist M pointA = Real.sqrt 74 := 
by
  sorry

end midpoint_distance_l39_39861


namespace last_two_digits_7_pow_2012_l39_39267

theorem last_two_digits_7_pow_2012 :
  (7^2012) % 100 = 01 :=
by 
  -- Exploration shows that the pattern for last two digits repeats every 4
  have H1 : (7^2) % 100 = 49 := by norm_num,
  have H2 : (7^3) % 100 = 43 := by norm_num,
  have H3 : (7^4) % 100 = 01 := by norm_num,
  have H4 : (7^5) % 100 = 07 := by norm_num,
  -- The general pattern found: 
  -- (7^(4k-2)) % 100 = 49
  -- (7^(4k-1)) % 100 = 43
  -- (7^(4k)) % 100 = 01
  -- (7^(4k+1)) % 100 = 07
  -- Since 2012 = 4 * 503, we fit k = 503 exactly,
  -- thus giving us the case (7^(4*503)) % 100 which must be 01
  sorry

end last_two_digits_7_pow_2012_l39_39267


namespace solve_sqrt_equation_l39_39082

theorem solve_sqrt_equation (x : ℝ) :
  (sqrt (3 * x ^ 2 + 2 * x + 1) = 3) ↔ (x = 4 / 3 ∨ x = -2) :=
by
  sorry

end solve_sqrt_equation_l39_39082


namespace endpoints_undetermined_l39_39260

theorem endpoints_undetermined (m : ℝ → ℝ) :
  (∀ x, m x = x - 2) ∧ (∃ mid : ℝ × ℝ, ∃ (x1 x2 y1 y2 : ℝ), 
    mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m mid.1 = mid.2) → 
  ¬ (∃ (x1 x2 y1 y2 : ℝ), mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m ((x1 + x2) / 2) = (y1 + y2) / 2 ∧
    x1 = the_exact_endpoint ∧ x2 = the_exact_other_endpoint) :=
by sorry

end endpoints_undetermined_l39_39260


namespace jason_age_at_end_of_2004_l39_39805

noncomputable def jason_age_in_1997 (y : ℚ) (g : ℚ) : Prop :=
  y = g / 3 

noncomputable def birth_years_sum (y : ℚ) (g : ℚ) : Prop :=
  (1997 - y) + (1997 - g) = 3852

theorem jason_age_at_end_of_2004
  (y g : ℚ)
  (h1 : jason_age_in_1997 y g)
  (h2 : birth_years_sum y g) :
  y + 7 = 42.5 :=
by
  sorry

end jason_age_at_end_of_2004_l39_39805


namespace percent_volume_filled_with_water_l39_39017

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39017


namespace problem_1_problem_2_l39_39178

theorem problem_1 {m : ℝ} (h₁ : 0 < m) (h₂ : ∀ x : ℝ, (m - |x + 2| ≥ 0) ↔ (-3 ≤ x ∧ x ≤ -1)) :
  m = 1 :=
sorry

theorem problem_2 {a b c : ℝ} (h₃ : 0 < a ∧ 0 < b ∧ 0 < c) (h₄ : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1)
  : a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_1_problem_2_l39_39178


namespace percent_volume_filled_with_water_l39_39013

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39013


namespace num_solutions_eq_three_l39_39704

theorem num_solutions_eq_three :
  (∃ n : Nat, (x : ℝ) → (x^2 - 4) * (x^2 - 1) = (x^2 + 3 * x + 2) * (x^2 - 8 * x + 7) → n = 3) :=
sorry

end num_solutions_eq_three_l39_39704


namespace largest_prime_divisor_of_factorial_sum_l39_39109

theorem largest_prime_divisor_of_factorial_sum {n : ℕ} (h1 : n = 13) : 
  Nat.gcd (Nat.factorial 13) 15 = 1 ∧ Nat.gcd (Nat.factorial 13 * 15) 13 = 13 :=
by
  sorry

end largest_prime_divisor_of_factorial_sum_l39_39109


namespace orchids_initially_l39_39331

-- Definitions and Conditions
def initial_orchids (current_orchids: ℕ) (cut_orchids: ℕ) : ℕ :=
  current_orchids + cut_orchids

-- Proof statement
theorem orchids_initially (current_orchids: ℕ) (cut_orchids: ℕ) : initial_orchids current_orchids cut_orchids = 3 :=
by 
  have h1 : current_orchids = 7 := sorry
  have h2 : cut_orchids = 4 := sorry
  have h3 : initial_orchids current_orchids cut_orchids = 7 + 4 := sorry
  have h4 : initial_orchids current_orchids cut_orchids = 3 := sorry
  sorry

end orchids_initially_l39_39331


namespace sum_fraction_series_eq_l39_39837

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l39_39837


namespace quadratic_properties_l39_39929

theorem quadratic_properties (a b c : ℝ) (h₀ : a < 0) (h₁ : a - b + c = 0) :
  (am² b c - 4 a) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ x1 x2 : ℝ, (∃ h : a * x1^2 + b * x1 + c + 1 = 0, ∃ h2 : a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39929


namespace jobApplicants_l39_39661

theorem jobApplicants (total : ℕ) (exp : ℕ) (deg : ℕ) (noExpNoDeg : ℕ)
  (h_total : total = 30) 
  (h_exp : exp = 10) 
  (h_deg : deg = 18)
  (h_noExpNoDeg : noExpNoDeg = 3) :
  ∃ (expAndDeg : ℕ), expAndDeg = 1 :=
by
  have h1 : total = exp + deg - expAndDeg + noExpNoDeg := sorry
  have h2 : expAndDeg =  total - (exp + deg - noExpNoDeg) := sorry
  have h3 : expAndDeg = 30 - (10 + 18 - 3) := sorry
  exact ⟨1, h3⟩

end jobApplicants_l39_39661


namespace union_area_of_five_triangles_l39_39875

theorem union_area_of_five_triangles :
  let s := 4 in
  let num_triangles := 5 in
  let single_triangle_area := (sqrt 3 / 4) * s ^ 2 in
  let total_area_without_overlap := num_triangles * single_triangle_area in
  let overlap_area_per_pair := (sqrt 3 / 4) * (s / 2) ^ 2 in
  let num_overlaps := num_triangles - 1 in
  let total_overlap_area := num_overlaps * overlap_area_per_pair in
  let net_area := total_area_without_overlap - total_overlap_area in
  net_area = 16 * sqrt 3 :=
by 
  sorry

end union_area_of_five_triangles_l39_39875


namespace sum_f_eq_sqrt3_div3_l39_39177

def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem sum_f_eq_sqrt3_div3 (x₁ x₂ : ℝ) (h : x₁ + x₂ = 1) : f x₁ + f x₂ = Real.sqrt 3 / 3 :=
by {
  unfold f,
  sorry
}

end sum_f_eq_sqrt3_div3_l39_39177


namespace Jamie_pays_30_in_taxes_l39_39630

def progressive_tax (income : ℝ) : ℝ :=
  if income ≤ 150 then 0
  else if income ≤ 300 then (income - 150) * 0.10
  else (150 * 0.10) + (income - 300) * 0.15

noncomputable def calculate_tax (gross_income deduction : ℝ) : ℝ :=
  let taxable_income := gross_income - deduction
  progressive_tax taxable_income

theorem Jamie_pays_30_in_taxes :
  calculate_tax 450 50 = 30 :=
by
  sorry

end Jamie_pays_30_in_taxes_l39_39630


namespace trigonometric_identity_eq_neg_one_l39_39825

theorem trigonometric_identity_eq_neg_one :
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180)) = -1 :=
by
  -- Variables needed for hypotheses
  have h₁ : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₂ : Real.sin (60 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₃ : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h₄ : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  -- Main proof
  sorry

end trigonometric_identity_eq_neg_one_l39_39825


namespace unit_vector_is_orthogonal_and_unit_l39_39512

open Real
open Matrix

noncomputable def is_unit_vector (v : Vector ℝ) : Prop :=
  ∥v∥ = 1

def is_orthogonal (v w : Vector ℝ) : Prop :=
  v ⬝ w = 0

def given_vector1 := ![2, 1, 1]
def given_vector2 := ![3, 0, 2]
def unit_vector := ![2 / Real.sqrt 14, -1 / Real.sqrt 14, -3 / Real.sqrt 14]

theorem unit_vector_is_orthogonal_and_unit :
  is_unit_vector unit_vector ∧ is_orthogonal unit_vector given_vector1 ∧ is_orthogonal unit_vector given_vector2 :=
sorry

end unit_vector_is_orthogonal_and_unit_l39_39512


namespace at_most_two_sides_equal_to_longest_diagonal_l39_39994

variables {n : ℕ} (P : convex_polygon n)
  (h1 : n > 3)
  (longest_diagonal : diagonal P)
  (equal_length_sides : finset (side P))
  (h2 : ∀ s ∈ equal_length_sides, side.length s = diagonal.length longest_diagonal)

theorem at_most_two_sides_equal_to_longest_diagonal :
  equal_length_sides.card ≤ 2 :=
sorry

end at_most_two_sides_equal_to_longest_diagonal_l39_39994


namespace milk_leftover_l39_39475

-- Definitions of the conditions
def milk_per_day := 16
def kids_consumption_percent := 0.75
def cooking_usage_percent := 0.50

-- The theorem to prove
theorem milk_leftover : 
  let milk_consumed_by_kids := milk_per_day * kids_consumption_percent in
  let remaining_milk_after_kids := milk_per_day - milk_consumed_by_kids in
  let milk_used_for_cooking := remaining_milk_after_kids * cooking_usage_percent in
  let milk_leftover := remaining_milk_after_kids - milk_used_for_cooking in
  milk_leftover = 2 := 
sorry

end milk_leftover_l39_39475


namespace sum_of_coeffs_of_integer_powers_x_l39_39615

theorem sum_of_coeffs_of_integer_powers_x (n : ℕ) (h : n > 0) :
  let f := λ x : ℝ, (x + sqrt x + 1) ^ (2 * n + 1) in
  (∀ t : ℝ, ∀ k : ℕ, (t^2 + t + 1) ^ (2 * n + 1) = ∑ i in range (4 * n + 3), f i * t ^ i) →
  (sum_of_coeffs_of_integer_powers_x (x + sqrt x + 1) ^ (2 * n + 1)) = (3 ^ (2 * n + 1) + 1) / 2 :=
sorry

end sum_of_coeffs_of_integer_powers_x_l39_39615


namespace simplify_expression_l39_39646

variable (a b c d x : ℝ)
variable (hab : a ≠ b)
variable (hac : a ≠ c)
variable (had : a ≠ d)
variable (hbc : b ≠ c)
variable (hbd : b ≠ d)
variable (hcd : c ≠ d)

theorem simplify_expression :
  ( ( (x + a)^4 / ((a - b)*(a - c)*(a - d)) )
  + ( (x + b)^4 / ((b - a)*(b - c)*(b - d)) )
  + ( (x + c)^4 / ((c - a)*(c - b)*(c - d)) )
  + ( (x + d)^4 / ((d - a)*(d - b)*(d - c)) ) = a + b + c + d + 4*x ) :=
  sorry

end simplify_expression_l39_39646


namespace quadratic_properties_l39_39927

theorem quadratic_properties (a b c : ℝ) (h₀ : a < 0) (h₁ : a - b + c = 0) :
  (am² b c - 4 a) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ x1 x2 : ℝ, (∃ h : a * x1^2 + b * x1 + c + 1 = 0, ∃ h2 : a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39927


namespace select_participants_l39_39292

theorem select_participants (female male : ℕ) (total : ℕ) (at_least_one_female : ℕ) :
  female = 2 → male = 4 → total = 3 → at_least_one_female = 16 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end select_participants_l39_39292


namespace sum_of_odd_digits_upto_321_l39_39647

-- Define the sum of odd digits function
def s (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.filter (λ d => d % 2 = 1).sum

-- The specific problem statement
theorem sum_of_odd_digits_upto_321 :
  (∑ n in Finset.range 322, s n) = 1727 :=
by
  sorry

end sum_of_odd_digits_upto_321_l39_39647


namespace beth_twice_sister_age_l39_39590

theorem beth_twice_sister_age (beth_age sister_age : ℕ) (h_beth : beth_age = 18) (h_sister : sister_age = 5) :
  ∃ n : ℕ, beth_age + n = 2 * (sister_age + n) ∧ n = 8 :=
by
  use 8
  split
  { calc
    beth_age + 8 = 18 + 8 : by rw [h_beth]
             ... = 26 : by ring
    2 * (sister_age + 8) = 2 * (5 + 8) : by rw [h_sister]
                  ... = 2 * 13 : by ring
                  ... = 26 : by ring
  }
  { exact rfl }
  sorry

end beth_twice_sister_age_l39_39590


namespace sum_of_decimals_as_common_fraction_l39_39504

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l39_39504


namespace ctg_double_angle_l39_39858

open Real

theorem ctg_double_angle (α : ℝ) 
  (h1 : sin (α - π / 2) = -2 / 3) 
  (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  cotan (2 * α) = sqrt 5 / 20 := 
by sorry

end ctg_double_angle_l39_39858


namespace B_inter_C_cardinality_l39_39184

open Set

def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 99}
def B : Set ℕ := {x | ∃ (y : ℕ), y ∈ A ∧ x = 2 * y}
def C : Set ℕ := {x | 2 * x ∈ A}

theorem B_inter_C_cardinality : (B ∩ C).card = 24 := by
  sorry

end B_inter_C_cardinality_l39_39184


namespace simplify_absolute_values_l39_39137

theorem simplify_absolute_values (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a + 2| = 2 :=
sorry

end simplify_absolute_values_l39_39137


namespace george_monthly_income_l39_39131

theorem george_monthly_income (I : ℝ) (h : I / 2 - 20 = 100) : I = 240 := 
by sorry

end george_monthly_income_l39_39131


namespace coefficient_x4_expansion_l39_39481

theorem coefficient_x4_expansion :
  let f := (x-1) * (3*x^2 + 1)^3
  in nat_degree f = 4 ->
  coefficient f 4 = -27 :=
by
  sorry

end coefficient_x4_expansion_l39_39481


namespace horner_method_v3_correct_l39_39728

-- Define the polynomial function according to Horner's method
def horner (x : ℝ) : ℝ :=
  (((((3 * x - 2) * x + 2) * x - 4) * x) * x - 7)

-- Given the value of x
def x_val : ℝ := 2

-- Define v_3 based on the polynomial evaluated at x = 2 using Horner's method
def v3 : ℝ := horner x_val

-- Theorem stating what we need to prove
theorem horner_method_v3_correct : v3 = 16 :=
  by
    sorry

end horner_method_v3_correct_l39_39728


namespace number_of_positive_divisors_of_60_l39_39962

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l39_39962


namespace number_of_solutions_l39_39865

theorem number_of_solutions :
  card { (x, y, z) : ℕ × ℕ × ℕ // x^2 + y^2 = 2 * z^2 ∧ z < y ∧ y ≤ z + 50 } = 131 :=
sorry

end number_of_solutions_l39_39865


namespace students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l39_39664

noncomputable def numStudentsKnowingSecret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem students_on_seventh_day :
  (numStudentsKnowingSecret 7) = 3280 :=
by
  sorry

theorem day_of_week (n : ℕ) : String :=
  if n % 7 = 0 then "Monday" else
  if n % 7 = 1 then "Tuesday" else
  if n % 7 = 2 then "Wednesday" else
  if n % 7 = 3 then "Thursday" else
  if n % 7 = 4 then "Friday" else
  if n % 7 = 5 then "Saturday" else
  "Sunday"

theorem day_when_3280_students_know_secret :
  day_of_week 7 = "Sunday" :=
by
  sorry

end students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l39_39664


namespace cone_volume_filled_88_8900_percent_l39_39025

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39025


namespace perpendicular_vectors_lambda_l39_39884

theorem perpendicular_vectors_lambda (λ : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (-1, λ)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  λ = 1 / 2 := by 
  sorry

end perpendicular_vectors_lambda_l39_39884


namespace gasoline_added_correct_l39_39401

def tank_capacity := 48
def initial_fraction := 3 / 4
def final_fraction := 9 / 10

def gasoline_at_initial_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_at_final_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_added (initial: ℝ) (final: ℝ) : ℝ := final - initial

theorem gasoline_added_correct (capacity: ℝ) (initial_fraction: ℝ) (final_fraction: ℝ)
  (h_capacity : capacity = 48) (h_initial : initial_fraction = 3 / 4) (h_final : final_fraction = 9 / 10) :
  gasoline_added (gasoline_at_initial_fraction capacity initial_fraction) (gasoline_at_final_fraction capacity final_fraction) = 7.2 :=
by
  sorry

end gasoline_added_correct_l39_39401


namespace sum_of_floor_sqrt_l39_39454

theorem sum_of_floor_sqrt :
  (∑ n in Finset.range 26, Int.floor (Real.sqrt n)) = 75 :=
by
  -- skipping proof details
  sorry

end sum_of_floor_sqrt_l39_39454


namespace find_second_sum_l39_39745

theorem find_second_sum (S : ℤ) (x : ℤ) (h_S : S = 2678)
  (h_eq_interest : x * 3 * 8 = (S - x) * 5 * 3) : (S - x) = 1648 :=
by {
  sorry
}

end find_second_sum_l39_39745


namespace kangaroo_jumps_to_E_l39_39441

noncomputable def a_n (n : ℕ) : ℝ :=
  if n % 2 ≠ 0 then 0 else let m := n / 2 in 
  (1 / Real.sqrt 2) * (2 + Real.sqrt 2) ^ (m - 1) - (1 / Real.sqrt 2) * (2 - Real.sqrt 2) ^ (m - 1)

theorem kangaroo_jumps_to_E (n : ℕ) :
  a_n n = if n % 2 ≠ 0 then 0 else 
          (1 / Real.sqrt 2) * (2 + Real.sqrt 2) ^ (n / 2 - 1) - 
          (1 / Real.sqrt 2) * (2 - Real.sqrt 2) ^ (n / 2 - 1) :=
by sorry

end kangaroo_jumps_to_E_l39_39441


namespace number_of_positive_divisors_of_60_l39_39964

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l39_39964


namespace geometric_to_arithmetic_l39_39188

theorem geometric_to_arithmetic (a_1 a_2 a_3 b_1 b_2 b_3: ℝ) (ha: a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0 ∧ b_1 > 0 ∧ b_2 > 0 ∧ b_3 > 0)
  (h_geometric_a : ∃ q : ℝ, a_2 = a_1 * q ∧ a_3 = a_1 * q^2)
  (h_geometric_b : ∃ q₁ : ℝ, b_2 = b_1 * q₁ ∧ b_3 = b_1 * q₁^2)
  (h_sum : a_1 + a_2 + a_3 = b_1 + b_2 + b_3)
  (h_arithmetic : 2 * a_2 * b_2 = a_1 * b_1 + a_3 * b_3) : 
  a_2 = b_2 :=
by
  sorry

end geometric_to_arithmetic_l39_39188


namespace solve_equation_l39_39859

-- Definitions based on the conditions
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10) = 0

-- Theorem stating that the solutions of the given equation are the expected values
theorem solve_equation :
  {x : ℝ | equation x} = {-2 + 2 * Real.sqrt 14, -2 - 2 * Real.sqrt 14, (7 + Real.sqrt 89) / 2, (7 - Real.sqrt 89) / 2} :=
by
  sorry

end solve_equation_l39_39859


namespace find_y_values_l39_39100

theorem find_y_values (y : ℝ) : 
  (∃ y, y ∈ [-1, -4/3) ∨ y ∈ (-4/3, 0) ∨ y ∈ (0, 1) ∨ y ∈ (1, ∞)) ↔ 
  (∃ y, y \neq 0 ∧ y \neq 1 ∧ y \neq -4/3 ∧ (y + y^2 - 3y^3) ≠ 0 ∧ 
  (y^2 + y^3 - 3y^4) / (y + y^2 - 3y^3) ≥ -1) :=
begin
  sorry
end

end find_y_values_l39_39100


namespace problem_statement_l39_39160

-- Definition of the conditions
def cond1 (z1 : ℂ) : Prop := (z1 - 2) * (1 + complex.i) = 1 - complex.i
def cond2 (z2 : ℂ) : Prop := z2.im = 2
def cond3 (z1 z2 : ℂ) : Prop := (z1 * z2).im = 0

-- Statement to be proved
theorem problem_statement (z1 z2 : ℂ)
  (h1 : cond1 z1) 
  (h2 : cond2 z2)
  (h3 : cond3 z1 z2) : 
  z2 = 4 + 2 * complex.i ∧ abs z2 = 2 * real.sqrt 5 :=
sorry

end problem_statement_l39_39160


namespace sec_225_eq_neg_sqrt2_l39_39091

theorem sec_225_eq_neg_sqrt2 : Real.sec (225 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end sec_225_eq_neg_sqrt2_l39_39091


namespace sum_of_possible_m_values_l39_39157

theorem sum_of_possible_m_values : 
  (∑ m in (Finset.filter (λ m, 0 < m ∧ m < 9) (Finset.range 10)), m) = 36 :=
by
  sorry

end sum_of_possible_m_values_l39_39157


namespace pasha_mistake_l39_39390

theorem pasha_mistake :
  ¬ (∃ (K R O S C T P : ℕ), K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ P < 10 ∧
    K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ P ∧
    R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ P ∧
    O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ P ∧
    S ≠ C ∧ S ≠ T ∧ S ≠ P ∧
    C ≠ T ∧ C ≠ P ∧ T ≠ P ∧
    10000 * K + 1000 * R + 100 * O + 10 * S + S + 2011 = 10000 * C + 1000 * T + 100 * A + 10 * P + T) :=
sorry

end pasha_mistake_l39_39390


namespace positive_divisors_60_l39_39987

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l39_39987


namespace find_A_coordinates_l39_39910

-- Conditions
def O : Prod ℝ ℝ := (0, 0)
def F : Prod ℝ ℝ := (1, 0)

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def on_parabola (A : Prod ℝ ℝ) : Prop := parabola A.1 A.2

def dot_product (v w : Prod ℝ ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Given conditions in Lean
axiom OA_AF_dot_product (A : Prod ℝ ℝ) : on_parabola A → 
  dot_product A (1 - A.1, -A.2) = -4

-- Proof statement
theorem find_A_coordinates (A : Prod ℝ ℝ) 
  (h_on_parabola : on_parabola A) 
  (h_dot_product : dot_product A (1 - A.1, -A.2) = -4) :
  A = (1, 2) ∨ A = (1, -2) := by
sorrey

end find_A_coordinates_l39_39910


namespace position_of_2020_is_31_l39_39293

-- Define the set of digits
def digits := [0, 1, 2, 3, 4]

-- Function to check if the given number has each digit used at most 3 times
def valid_number (n : ℕ) : Prop :=
  let digits := nat.digits 10 n in
  ∀ (d ∈ digits), digits.count d ≤ 3

-- Function to generate four-digit numbers using two chosen digits
def generate_numbers (d1 d2 : ℕ) : List ℕ :=
  let digits := [d1, d2] in 
  [d1 * 1000 + d1 * 100 + d1 * 10 + d1,
   d1 * 1000 + d1 * 100 + d1 * 10 + d2,
   d1 * 1000 + d1 * 100 + d2 * 10 + d1,
   d1 * 1000 + d1 * 100 + d2 * 10 + d2,
   d1 * 1000 + d2 * 100 + d1 * 10 + d1,
   d1 * 1000 + d2 * 100 + d1 * 10 + d2,
   d1 * 1000 + d2 * 100 + d2 * 10 + d1,
   d1 * 1000 + d2 * 100 + d2 * 10 + d2,
   d2 * 1000 + d1 * 100 + d1 * 10 + d1,
   d2 * 1000 + d1 * 100 + d1 * 10 + d2,
   d2 * 1000 + d1 * 100 + d2 * 10 + d1,
   d2 * 1000 + d1 * 100 + d2 * 10 + d2,
   d2 * 1000 + d2 * 100 + d1 * 10 + d1,
   d2 * 1000 + d2 * 100 + d1 * 10 + d2,
   d2 * 1000 + d2 * 100 + d2 * 10 + d1,
   d2 * 1000 + d2 * 100 + d2 * 10 + d2]

-- Generate all valid four-digit numbers from any pair of digits
def all_numbers : List ℕ :=
  (List.bind (finset.powerset_len 2 {0, 1, 2, 3, 4}.val) (λ pr, match pr with
   | [d1, d2] => generate_numbers d1 d2
   | _ => []
   end)).filter (λ n, n >= 1000) -- valid four-digit numbers

-- Sort and get the position of 2020
def position_2020 : ℕ :=
  match List.indexOf 2020 (all_numbers.sort (≤)) with
  | none => 0 -- Should never be none according to our problem
  | some n => n + 1  -- position in 1-based indexing

-- Lean statement to prove the position of 2020 is 31
theorem position_of_2020_is_31 :
  position_2020 = 31 :=
  by sorry

end position_of_2020_is_31_l39_39293


namespace koala_fiber_intake_l39_39238

theorem koala_fiber_intake (x : ℝ) (h : 0.30 * x = 12) : x = 40 := 
sorry

end koala_fiber_intake_l39_39238


namespace chicago_bulls_heat_games_total_l39_39212

-- Statement of the problem in Lean 4
theorem chicago_bulls_heat_games_total :
  ∀ (bulls_games : ℕ) (heat_games : ℕ),
    bulls_games = 70 →
    heat_games = bulls_games + 5 →
    bulls_games + heat_games = 145 :=
by
  intros bulls_games heat_games h_bulls h_heat
  rw [h_bulls, h_heat]
  exact sorry

end chicago_bulls_heat_games_total_l39_39212


namespace painting_time_l39_39523

-- Define types for experienced and inexperienced workers
structure Worker := 
  (experienced : Bool)

-- Define the work rates
def work_rate (w : Worker) (x : ℝ) : ℝ := 
  if w.experienced then 2 * x else x

-- Define the collective work rate for a group of workers
def collective_work_rate (group : List Worker) (x : ℝ) : ℝ :=
  group.foldl (λ acc w => acc + work_rate w x) 0

-- Given conditions
def condition1 (x : ℝ) : Prop :=
  let five_people : List Worker := [⟨true⟩, ⟨true⟩, ⟨true⟩, ⟨false⟩, ⟨false⟩]
  collective_work_rate five_people x = 8 * x

def condition2 : Prop :=
  ∀ x, collective_work_rate [⟨true⟩, ⟨false⟩] x = 3 * x

-- Prove the time required by two experienced and three inexperienced workers equals 8/7 hours
theorem painting_time (x : ℝ) (h1 : condition1 x) :
  let group := [⟨true⟩, ⟨true⟩, ⟨false⟩, ⟨false⟩, ⟨false⟩]
  let rate := collective_work_rate group x
  rate ≠ 0 →
  1 / rate = 8 / 7 :=
by
  let two_experienced_three_inexperienced_group := [⟨true⟩, ⟨true⟩, ⟨false⟩, ⟨false⟩, ⟨false⟩]
  have key_work_rate : collective_work_rate two_experienced_three_inexperienced_group x = 7 * x,
  { sorry },  -- Proof of the work rate equivalency
  have rate_nonzero : 7 * x ≠ 0,
  { sorry },  -- Proof that the rate is nonzero
  have time_eq : 1 / (7 * x) = 8 / 7,
  { sorry },  -- Proof of time equivalency
  exact time_eq

end painting_time_l39_39523


namespace trajectory_eq_PA_PB_value_l39_39619

-- Define the parametric equation of line l.
def line_l (t : ℝ) : ℝ × ℝ :=
  ( ( √ 2 / 2 ) * t, 2 + ( √ 2 / 2 ) * t )

-- Define the polar coordinate equation of circle C.
def circle_C_eq (theta : ℝ) : ℝ :=
  4 * Real.cos theta

-- Define a transformation from polar to rectangular coordinates.
def polar_to_rect (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

-- Define the trajectory equations condition.
def Q_trajectory (x y : ℝ) : Prop :=
  (x - 6)^2 + y^2 = 36

-- Main theorem: The rectangular coordinate equation of the trajectory of point Q.
theorem trajectory_eq :
  ∀ (rho theta : ℝ), 
  ρ = 12 * Real.cos θ →
  Q_trajectory (ρ * Real.cos θ) (ρ * Real.sin θ) :=
by
  sorry

-- Main theorem: The value of |PA| + |PB|.
theorem PA_PB_value (P Q : ℝ × ℝ) :
  P = (0, 2) →
  ∀ t, line_l t ∈ set_of (λ q => (q.1 - 6)^2 + q.2^2 = 36) →
  |PA| + |PB| = 4 * √ 2 :=
by
  sorry

end trajectory_eq_PA_PB_value_l39_39619


namespace remainder_of_3_pow_101_plus_4_mod_5_l39_39370

theorem remainder_of_3_pow_101_plus_4_mod_5 :
  (3^101 + 4) % 5 = 2 :=
by
  have h1 : 3 % 5 = 3 := by sorry
  have h2 : (3^2) % 5 = 4 := by sorry
  have h3 : (3^3) % 5 = 2 := by sorry
  have h4 : (3^4) % 5 = 1 := by sorry
  -- more steps to show the pattern and use it to prove the final statement
  sorry

end remainder_of_3_pow_101_plus_4_mod_5_l39_39370


namespace distance_from_B_to_AC_l39_39635

variable (A B C : Type)

def is_right_triangle (ABC : Triangle) : Prop :=
  ∃ (a b c : ℝ), a^2 + b^2 = c^2

def is_inscribed_in_semicircle (ABC : Triangle) (AC: ℝ) : Prop :=
  ∀ (M : Point),  M = midpoint A C → (B ∈ semicircle_with_diameter A C)

def median_is_geometric_mean (ABC : Triangle) (AB BC BM : ℝ) : Prop :=
  BM = sqrt (AB * BC)

def distance_vertex_to_side (A B C : Point) : Prop :=
  ∃ BD : ℝ, BD = (5 / 2) 

theorem distance_from_B_to_AC (A B C : Point) (ABC : Triangle) (AC : ℝ)
  (h1 : is_right_triangle ABC)
  (h2 : is_inscribed_in_semicircle ABC AC)
  (h3 : median_is_geometric_mean ABC AB BC BM) :
  distance_vertex_to_side A B C :=
sorry

end distance_from_B_to_AC_l39_39635


namespace a_2013_value_l39_39225

noncomputable def seq : ℕ → ℝ
| 0 := 0
| (n + 1) := (real.sqrt 3 + seq n) / (1 - real.sqrt 3 * seq n)

theorem a_2013_value : seq 2013 = -real.sqrt 3 :=
sorry

end a_2013_value_l39_39225


namespace students_remaining_on_bus_l39_39424

theorem students_remaining_on_bus :
  ∀ (initial : ℕ) (f1 f2 f3 : ℚ),
    initial = 60 →
    f1 = 1 / 3 →
    f2 = 1 / 2 →
    f3 = 1 / 4 →
    let after_first_stop := initial - initial * f1 in
    let after_second_stop := after_first_stop - after_first_stop * f2 in
    let after_third_stop := after_second_stop - after_second_stop * f3 in
    after_third_stop = 15 :=
by
  intros initial f1 f2 f3 hinit hf1 hf2 hf3
  let after_first_stop := initial - initial * f1
  let after_second_stop := after_first_stop - after_first_stop * f2
  let after_third_stop := after_second_stop - after_second_stop * f3
  have : initial = 60 := hinit
  have : f1 = 1 / 3 := hf1
  have : f2 = 1 / 2 := hf2
  have : f3 = 1 / 4 := hf3
  sorry

end students_remaining_on_bus_l39_39424


namespace arithmetic_mean_same_color_coloring_exists_if_and_only_if_l39_39495

/-
We introduce the concepts and assumptions from the initial problem:

1. Every positive integer is colored with one of n colors.
2. There are infinitely many numbers of each color.
3. The arithmetic mean of any two different numbers with the same parity has a color determined uniquely by the colors of the two numbers.
-/

noncomputable def ColoringExists (n : ℕ) : Prop :=
∃ (color : ℕ → Fin n),
  (∀ k : Fin n, ∃∞m, color m = k) ∧
  (∀ a b : ℕ, a % 2 = b % 2 → a ≠ b → color ((a + b) / 2) = some_color color a b)

-- The first statement to prove:
theorem arithmetic_mean_same_color (n : ℕ) :
  ColoringExists n →
  ∀ (color : ℕ → Fin n) (a b : ℕ),
  a % 2 = b % 2 →
  color a = color b →
  color ((a + b) / 2) = color a :=
sorry

-- The second statement to prove:
theorem coloring_exists_if_and_only_if (n : ℕ) :
  ColoringExists n ↔ odd n :=
sorry

end arithmetic_mean_same_color_coloring_exists_if_and_only_if_l39_39495


namespace value_of_m_l39_39596

-- Define the condition of the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m

-- State the equivalence to be proved
theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 1 ∧ quadratic_equation x m = 0) → m = 1 :=
by
  sorry

end value_of_m_l39_39596


namespace valid_division_l39_39289

theorem valid_division (A B C E F G H K : ℕ) (hA : A = 7) (hB : B = 1) (hC : C = 2)
    (hE : E = 6) (hF : F = 8) (hG : G = 5) (hH : H = 4) (hK : K = 9) :
    (A * 10 + B) / ((C * 100 + A * 10 + B) / 100 + E + B * F * D) = 71 / 271 :=
by {
  sorry
}

end valid_division_l39_39289


namespace find_a_if_A_cap_B_not_empty_l39_39153

theorem find_a_if_A_cap_B_not_empty (a : ℝ) :
  let A := {-1, 0, a}
  let B := {x : ℤ | 1 < real.exp (x * real.log 3) ∧ real.exp (x * real.log 3) < 9}
  (¬A ∩ B = ∅) → a = 1 :=
by
  sorry

end find_a_if_A_cap_B_not_empty_l39_39153


namespace work_together_time_l39_39744

theorem work_together_time :
  let man_work_rate := 1/20
  let father_work_rate := 1/20
  let son_work_rate := 1/25
  let combined_work_rate := man_work_rate + father_work_rate + son_work_rate
  let days_to_complete := 1 / combined_work_rate 
  (days_to_complete - 100 / 14).abs < 0.01 :=
by
  -- Definitions of individual work rates
  let man_work_rate : ℚ := 1 / 20
  let father_work_rate : ℚ := 1 / 20
  let son_work_rate : ℚ := 1 / 25
  
  -- Combined work rate calculation
  let combined_work_rate := man_work_rate + father_work_rate + son_work_rate
  let combined_work_rate_simplified : ℚ := (1 / 20) + (1 / 20) + (1 / 25)
  
  -- Days to complete the job calculation
  let days_to_complete := 1 / combined_work_rate
  
  -- Compared with the expected value
  have h : (days_to_complete - 100 / 14).abs < 0.01 := 
    by sorry
  exact h

end work_together_time_l39_39744


namespace common_tangent_x_eq_neg1_l39_39545
open Real

-- Definitions of circles C₁ and C₂
def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
def circle2 := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Statement of the problem
theorem common_tangent_x_eq_neg1 :
  ∀ (x : ℝ) (y : ℝ),
    (x, y) ∈ circle1 ∧ (x, y) ∈ circle2 → x = -1 :=
sorry

end common_tangent_x_eq_neg1_l39_39545


namespace solution_set_inequality_l39_39872

theorem solution_set_inequality (a x : ℝ) :
  (12 * x^2 - a * x > a^2) ↔
  ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
   (a = 0 ∧ x ≠ 0) ∨
   (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
sorry


end solution_set_inequality_l39_39872


namespace proof_problem_l39_39161

variable {R : Type*} [Real R]

noncomputable def f : R → R := sorry

theorem proof_problem 
  (h_domain : ∀ (x : R), Continuous (f x))
  (h_functional_eq : ∀ (x y : R), f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end proof_problem_l39_39161


namespace max_colored_nodes_without_cycle_in_convex_polygon_l39_39143

def convex_polygon (n : ℕ) : Prop := n ≥ 3

def valid_diagonals (n : ℕ) : Prop := n = 2019

def no_three_diagonals_intersect_at_single_point (x : Type*) : Prop :=
  sorry -- You can provide a formal definition here based on combinatorial geometry.

def no_loops (n : ℕ) (k : ℕ) : Prop :=
  k ≤ (n * (n - 3)) / 2 - 1

theorem max_colored_nodes_without_cycle_in_convex_polygon :
  convex_polygon 2019 →
  valid_diagonals 2019 →
  no_three_diagonals_intersect_at_single_point ℝ →
  ∃ k, k = 2035151 ∧ no_loops 2019 k := 
by
  -- The proof would be constructed here.
  sorry

end max_colored_nodes_without_cycle_in_convex_polygon_l39_39143


namespace find_x_l39_39421

theorem find_x (x : ℝ) (hx : x > 0) (condition : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 :=
by
  sorry

end find_x_l39_39421


namespace sum_of_areas_of_triangles_l39_39483

theorem sum_of_areas_of_triangles (m n p : ℕ) 
  (h1 : m = 48) 
  (h2 : n = 2304) 
  (h3 : p = 3072) :
  let sum_areas := m + sqrt n + sqrt p
  in sum_areas = m + sqrt n + sqrt p → m + n + p = 5424 := 
by 
  sorry

end sum_of_areas_of_triangles_l39_39483


namespace problem_p17_l39_39759

def orderings_count (n : ℕ) : ℕ :=
  if n = 1 then 1 else 
  (Nat.floor (n / 2) + 1) * orderings_count (n - 1)

noncomputable def is_perfect_square (x : ℕ) : Prop :=
  ∃ m : ℕ, m * m = x

theorem problem_p17 : (Finset.filter (λ k, is_perfect_square (orderings_count k)) (Finset.range 51)).card = 29 :=
by
  sorry

end problem_p17_l39_39759


namespace meal_cost_l39_39803

theorem meal_cost (total_people total_bill : ℕ) (h1 : total_people = 2 + 5) (h2 : total_bill = 21) :
  total_bill / total_people = 3 := by
  sorry

end meal_cost_l39_39803


namespace percent_volume_filled_with_water_l39_39009

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39009


namespace gcd_lcm_product_l39_39120

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end gcd_lcm_product_l39_39120


namespace four_digit_integers_count_l39_39194

def is_valid_digit (d : ℕ) : Prop := d >= 1 ∧ d <= 9
  
def no_adjacent_duplicates (x : ℕ) : Prop :=
  let digits := [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10]
  ∀ (i : ℕ), i < 3 → digits[i] ≠ digits[i + 1]
  
def count_valid_4digit_numbers : ℕ :=
  (List.range (9 * 10^3 + 9)).filter (λ n, no_adjacent_duplicates n ∧ is_valid_digit (n / 1000 % 10) ∧ is_valid_digit (n / 100 % 10) ∧  is_valid_digit (n / 10 % 10) ∧ is_valid_digit (n % 10)).length

theorem four_digit_integers_count : count_valid_4digit_numbers = 3528 :=
by
  sorry

end four_digit_integers_count_l39_39194


namespace parallelogram_angle_ratio_l39_39218

theorem parallelogram_angle_ratio (ABCD : Type*)
  [Parallelogram ABCD]
  (O : Intersection Diagonal_Intersection ABCD)
  (r : ℝ)
  (h1 : ∀ (A B C D O: ABCD),
        Angle(CAB) = 2 * Angle(DBA)
      ∧ Angle(DBC) = 2 * Angle(DBA)
      ∧ Angle(ACB) = r * Angle(AOB))
  (h2 : ∀ (x : ℝ), 
        x = 15) :
  let r := 7 / 9 in
  ∀ (r : ℝ),
  floor (1000 * r) = 777 := by
  let r := 7 / 9
  have : 1000 * r = 777 + (1000 * r - 777), from sorry
  have : floor (1000 * r) = 777, from sorry
  assumption
  

end parallelogram_angle_ratio_l39_39218


namespace nth_monomial_sequence_l39_39426

theorem nth_monomial_sequence (n : ℕ) : 
  ∃ (f : ℕ → ℤ × ℕ), f n = if n % 2 = 1 then (1, 2 * n + 1) else (-1, 2 * n + 1) → 
  (if n % 2 = 1 then 1 else -1) * x ^ (2 * n + 1) = (-1)^(n-1) * x^(2n+1) :=
by
  sorry

end nth_monomial_sequence_l39_39426


namespace relationship_among_abc_l39_39034

open Real

def f : ℝ → ℝ := sorry -- Defining a general function f

lemma symmetry_about_one (h : ℝ) : f (1 + h) = f (1 - h) :=
sorry  -- Symmetry of the function about x = 1

lemma decreasing_on_interval (x1 x2 : ℝ) (hx1 : 1 < x1) (hx2 : x1 < x2) :
  [f x2 - f x1] * [x2 - x1] < 0 :=
sorry -- f is decreasing on (1, ∞)

def a : ℝ := f (-1 / 2)
def b : ℝ := f 2
def c : ℝ := f (exp 1)

theorem relationship_among_abc :
  b > a ∧ a > c := 
sorry -- The relationship among a, b, c is b > a > c

end relationship_among_abc_l39_39034


namespace sum_x_coordinates_l39_39670

theorem sum_x_coordinates (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a * b = 42) :
  let x := -7 / a in
  let x_sum := 
      List.sum [ -7, -7 / 2, -7 / 3, -7 / 6, -1, -1 / 2, -1 / 3, 0 ] in
  x_sum = -33 / 2 :=
by
  sorry

end sum_x_coordinates_l39_39670


namespace no_fewer_than_60_circles_l39_39665

theorem no_fewer_than_60_circles
  (side_length : ℝ := 300)  -- The side length of the square in mm
  (circle_diameter : ℝ := 5) -- The diameter of each circle in mm
  (no_intersect_path : ∀ (p1 p2 : ℝ), 
    (p1 ∈ set.Icc 0 side_length) ∧ (p2 ∈ set.Icc 0 side_length) → 
    (LineSegment p1 p2).intersect_circle_count ≥ 1 := sorry) : 
  ∃ n : ℕ, n ≥ 60 :=
begin
  have h : side_length / circle_diameter = 60 := by sorry,
  existsi 60,
  exact le_refl 60,
end

end no_fewer_than_60_circles_l39_39665


namespace race_distance_difference_l39_39384

theorem race_distance_difference
  (d : ℕ) (tA tB : ℕ)
  (h_d: d = 80) 
  (h_tA: tA = 20) 
  (h_tB: tB = 25) :
  (d / tA) * tA = d ∧ (d - (d / tB) * tA) = 16 := 
by
  sorry

end race_distance_difference_l39_39384


namespace prism_distance_to_plane_l39_39159

theorem prism_distance_to_plane
  (side_length : ℝ)
  (volume : ℝ)
  (h : ℝ)
  (base_is_square : side_length = 6)
  (volume_formula : volume = (1 / 3) * h * (side_length ^ 2)) :
  h = 8 := 
  by sorry

end prism_distance_to_plane_l39_39159


namespace distance_between_parallel_lines_l39_39343

theorem distance_between_parallel_lines :
  let line1 := 3 * x + 4 * y - 12
  let line2 := 6 * x + 8 * y + 11
  let distance := fun (a b c1 c2 : ℝ) => abs(c2 - c1) / sqrt(a^2 + b^2)
  distance 6 8 (-24) 11 = 7 / 2 :=
by
  sorry

end distance_between_parallel_lines_l39_39343


namespace sum_of_101_terms_geometric_sequence_l39_39896

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ (a1 q : ℝ), a 0 = a1 ∧ ∀ n, a (n + 1) = a1 * q ^ n

noncomputable def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (finset.range n).sum a

theorem sum_of_101_terms_geometric_sequence {a S : ℕ → ℝ} 
  (h_geo : geometric_sequence a) 
  (h_sum : sum_of_first_n_terms S a)
  (h_a3 : a 2 = 3)
  (h_geom_cond : a 2015 + a 2016 = 0) :
  S 101 = 3 :=
by
  sorry

end sum_of_101_terms_geometric_sequence_l39_39896


namespace analytical_expression_max_min_values_l39_39958

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x + Real.sin x)
noncomputable def f (x : ℝ) : ℝ := a x.1 * b x.1 + a x.2 * b x.2

theorem analytical_expression (x : ℝ) : f x = (Real.sqrt 2 / 2) * Real.sin (2 * x + Real.pi / 4) + 3 / 2 :=
  sorry

theorem max_min_values (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (1 ≤ f x ∧ f x ≤ (3 + Real.sqrt 2) / 2) :=
  sorry

end analytical_expression_max_min_values_l39_39958


namespace prob1_prob2_l39_39907

-- Define lines l1 and l2
def l1 (x y m : ℝ) : Prop := x + m * y + 1 = 0
def l2 (x y m : ℝ) : Prop := (m - 3) * x - 2 * y + (13 - 7 * m) = 0

-- Perpendicular condition
def perp_cond (m : ℝ) : Prop := 1 * (m - 3) - 2 * m = 0

-- Parallel condition
def parallel_cond (m : ℝ) : Prop := m * (m - 3) + 2 = 0

-- Distance between parallel lines when m = 1
def distance_between_parallel_lines (d : ℝ) : Prop := d = 2 * Real.sqrt 2

-- Problem 1: Prove that if l1 ⊥ l2, then m = -3
theorem prob1 (m : ℝ) (h : perp_cond m) : m = -3 := sorry

-- Problem 2: Prove that if l1 ∥ l2, the distance d is 2√2
theorem prob2 (m : ℝ) (h1 : parallel_cond m) (d : ℝ) (h2 : m = 1 ∨ m = -2) (h3 : m = 1) (h4 : distance_between_parallel_lines d) : d = 2 * Real.sqrt 2 := sorry

end prob1_prob2_l39_39907


namespace tank_overflows_after_24_minutes_l39_39751

theorem tank_overflows_after_24_minutes 
  (rateA : ℝ) (rateB : ℝ) (t : ℝ) 
  (hA : rateA = 1) 
  (hB : rateB = 4) :
  t - 1/4 * rateB + t * rateA = 1 → t = 2/5 :=
by 
  intros h
  -- the proof steps go here
  sorry

end tank_overflows_after_24_minutes_l39_39751


namespace circle_intersects_line_when_m_is_1_25_no_common_points_of_circle_line_circle_with_diameter_PQ_passes_through_origin_l39_39892

-- Define the circle and line
def circle (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + x - 6 * y + m = 0
def line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- 1. Prove that when m = 1.25, the circle intersects the line
theorem circle_intersects_line_when_m_is_1_25 :
  ∃ x y : ℝ, circle (5 / 4) x y ∧ line x y :=
sorry

-- 2. Prove the range of values for m so that there are no common points
theorem no_common_points_of_circle_line (m : ℝ) :
  (∀ x y : ℝ, ¬ (circle m x y ∧ line x y)) ↔ 8 < m ∧ m < 37 / 4 :=
sorry

-- 3. Prove that if the circle with diameter PQ passes through the origin then m = 3
theorem circle_with_diameter_PQ_passes_through_origin (m : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, circle m x1 y1 ∧ circle m x2 y2 ∧ line x1 y1 ∧ line x2 y2 ∧
    (x1 * x2 + y1 * y2 = 0) ∧ (x1 + x2 = -2) ∧ ((4 * m - 27) / 5 = x1 * x2) ∧
    ((m + 12) / 5 = (9 - 3 * (x1 + x2) + x1 * x2) / 4)) ↔ m = 3 :=
sorry

end circle_intersects_line_when_m_is_1_25_no_common_points_of_circle_line_circle_with_diameter_PQ_passes_through_origin_l39_39892


namespace sum_s_1_to_321_l39_39650

-- Define the function s(n) to compute the sum of all odd digits of n
def s (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (λ d, d % 2 = 1) |>.sum

-- Now state the problem
theorem sum_s_1_to_321 : 
  (∑ n in Finset.range 322, s n) = 1727 :=
sorry

end sum_s_1_to_321_l39_39650


namespace intervals_of_monotonicity_range_of_m_l39_39533

open Real

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 2 * (x - a) / (x ^ 2 + b * x + 1)

theorem intervals_of_monotonicity (a b : ℝ) (h_odd : ∀ x, f a b x = - f a b (-x)) (h_a_b_zero : a = 0 ∧ b = 0) :
  (∀ x, f 0 0 x = 2 * x / (x ^ 2 + 1)) → 
  (∀ x, (f 0 0 x > f 0 0 (x - 1) ∧ -1 < x ∧ x < 1) ∨ (f 0 0 ↑x < f 0 0 (x - 1) ∧ (x > 1 ∨ x < -1))) :=
by sorry

theorem range_of_m (a b : ℝ) (h_odd : ∀ x, f a b x = - f a b (-x)) (h_a_b_zero : a = 0 ∧ b = 0) :
  (∀ m, (∃ x, 2 * m - 1 > f 0 0 x) ↔ m > 0) :=
by sorry

end intervals_of_monotonicity_range_of_m_l39_39533


namespace num_solutions_f_comp_eq_6_l39_39255

def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x + 4 else 3 * x - 6

theorem num_solutions_f_comp_eq_6 : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f(f(x1)) = 6 ∧ f(f(x2)) = 6) :=
sorry

end num_solutions_f_comp_eq_6_l39_39255


namespace apples_in_basket_l39_39716

-- Define the initial number of apples.
def initial_apples : Nat := 8

-- Define the number of apples added by Jungkook.
def added_apples : Nat := 7

-- The total number of apples in the basket.
def total_apples : Nat := initial_apples + added_apples

-- Prove that total_apples equals 15.
theorem apples_in_basket : total_apples = 15 := by
  dix "proof".
end by 
  -- Add the assumptions and calculate the total
  have h_initial : initial_apples = 8 := rfl
  have h_added : added_apples = 7 := rfl
  calc total_apples
         = initial_apples + added_apples : rfl
     ... = 8 + 7 : by rw [h_initial, h_added]
     ... = 15 : rfl

end apples_in_basket_l39_39716


namespace max_value_min_value_l39_39126

def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem max_value (k : ℤ) : f (2 * k * Real.pi + Real.pi / 3) = 1 := 
  sorry

theorem min_value (k : ℤ) : f (2 * k * Real.pi - 2 * Real.pi / 3) = -1 :=
  sorry

end max_value_min_value_l39_39126


namespace kat_third_test_score_l39_39632

theorem kat_third_test_score
  (score1 : ℕ)
  (score2 : ℕ)
  (desired_avg : ℕ)
  (h_score1 : score1 = 95)
  (h_score2 : score2 = 80)
  (h_desired_avg : desired_avg = 90) :
  ∃ x : ℕ, (score1 + score2 + x) / 3 = desired_avg :=
by {
  use 95,
  rw [h_score1, h_score2, h_desired_avg],
  norm_num,
  sorry,
}

end kat_third_test_score_l39_39632


namespace sum_of_decimals_as_fraction_l39_39497

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l39_39497


namespace dogwood_trees_total_is_100_l39_39715

def initial_dogwood_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20
def total_dogwood_trees : ℕ := initial_dogwood_trees + trees_planted_today + trees_planted_tomorrow

theorem dogwood_trees_total_is_100 : total_dogwood_trees = 100 := by
  sorry  -- Proof goes here

end dogwood_trees_total_is_100_l39_39715


namespace number_of_lines_intersecting_circle_l39_39943

theorem number_of_lines_intersecting_circle : 
  ∃ l : ℕ, 
  (∀ a b x y : ℤ, (x^2 + y^2 = 50 ∧ (x / a + y / b = 1))) → 
  (∃ n : ℕ, n = 60) :=
sorry

end number_of_lines_intersecting_circle_l39_39943


namespace weight_of_rectangle_l39_39431

noncomputable def area_triangle (s : ℝ) : ℝ := (s^2 * real.sqrt 3) / 4

noncomputable def area_rectangle (l w : ℝ) : ℝ := l * w

noncomputable def weight_from_area (initial_weight initial_area target_area : ℝ) : ℝ :=
  (initial_weight * target_area) / initial_area

theorem weight_of_rectangle :
  weight_from_area 16 (area_triangle 4) (area_rectangle 6 4) = 55.4 :=
by
  sorry

end weight_of_rectangle_l39_39431


namespace Billy_Reads_3_Books_l39_39810

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end Billy_Reads_3_Books_l39_39810


namespace camp_children_count_l39_39617

theorem camp_children_count (C : ℕ) (h : 0.1 * C = 0.05 * (C + 50)) : C = 50 :=
  sorry

end camp_children_count_l39_39617


namespace problem_solution_l39_39081

-- Define ceiling function
def ceiling (m : ℝ) : ℤ := ⌈m⌉

-- Define floor function
def floor (m : ℝ) : ℤ := ⌊m⌋

-- Variables
variables (x y : ℝ)

-- Equations from the problem
def equation1 : Prop := 3 * (floor x) + 2 * (ceiling y) = 2011
def equation2 : Prop := 2 * (ceiling x) - (floor y) = 2

-- The theorem to prove
theorem problem_solution (hx : equation1) (hy : equation2) : x + y = 861 :=
sorry

end problem_solution_l39_39081


namespace min_val_of_q_l39_39651

theorem min_val_of_q (p q : ℕ) (h1 : 72 / 487 < p / q) (h2 : p / q < 18 / 121) : 
  ∃ p q : ℕ, (72 / 487 < p / q) ∧ (p / q < 18 / 121) ∧ q = 27 :=
sorry

end min_val_of_q_l39_39651


namespace number_of_divisors_60_l39_39978

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l39_39978


namespace solution_is_correct_l39_39741

-- Initial conditions
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.40
def target_concentration : ℝ := 0.50

-- Given that we start with 2.4 liters of pure alcohol in a 6-liter solution
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Expected result after adding x liters of pure alcohol
def final_solution_volume (x : ℝ) : ℝ := initial_volume + x
def final_pure_alcohol (x : ℝ) : ℝ := initial_pure_alcohol + x

-- Equation to prove
theorem solution_is_correct (x : ℝ) :
  (final_pure_alcohol x) / (final_solution_volume x) = target_concentration ↔ 
  x = 1.2 := 
sorry

end solution_is_correct_l39_39741


namespace f_prime_e_value_l39_39595

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x * (deriv f e) + Real.log x

-- Define the condition that f is differentiable on (0, +∞)
def differentiable_on_f : differentiable_on ℝ f (Ioi 0) := by sorry

theorem f_prime_e_value : deriv f e = -1 / e :=
by sorry

end f_prime_e_value_l39_39595


namespace percent_volume_filled_with_water_l39_39011

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39011


namespace find_phi_l39_39941

variable (φ : ℝ) -- Define φ as a real number

-- Conditions
def f (x : ℝ) : ℝ := Real.sin (2 * x + φ)
def ϕ_condition : Prop := 0 < φ ∧ φ < π
def f_shifted (x : ℝ) := f (x + π / 6)
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 3 + φ)
def g_even : Prop := ∀ x : ℝ, g x = g (-x)

-- Theorem statement
theorem find_phi : ϕ_condition φ → g_even φ → φ = π / 6 :=
by
  intro hϕ h_even_g
  sorry -- Proof to be filled in

end find_phi_l39_39941


namespace min_a_for_monotonic_increase_l39_39174

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x ^ 3 + 2 * a * x ^ 2 + 2

theorem min_a_for_monotonic_increase :
  ∀ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → x^2 + 4 * a * x ≥ 0) ↔ a ≥ -1/4 := sorry

end min_a_for_monotonic_increase_l39_39174


namespace largest_prime_divisor_of_factorial_sum_l39_39106

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n : ℕ), 13 ≤ n → (n ∣ 13! + 14!) → is_prime n → n = 13 :=
by sorry

end largest_prime_divisor_of_factorial_sum_l39_39106


namespace number_of_odd_and_increasing_powers_l39_39951

theorem number_of_odd_and_increasing_powers (s : Set ℝ) :
  s = {-2, -1, 1/2, 1, 2, 3} →
  { α ∈ s | (∀ x : ℝ, x > 0 → mon_incr (λ y, y^α) x) ∧ odd (λ y, y^α) }.card = 2 :=
by
  intro h_s
  sorry

end number_of_odd_and_increasing_powers_l39_39951


namespace area_of_WIN_sector_l39_39406

-- Define the radius of the circle
def r : ℝ := 7

-- Define the probability of winning
def p_WIN : ℝ := 3 / 8

-- Define the total area of the circle
def total_area : ℝ := π * r^2

-- The goal is to prove the area of the WIN sector given the probability
theorem area_of_WIN_sector :
  ∃ (A_WIN : ℝ), A_WIN = p_WIN * total_area ∧ A_WIN = 147 * π / 8 :=
by
  sorry

end area_of_WIN_sector_l39_39406


namespace simplify_and_evaluate_expr_l39_39295

theorem simplify_and_evaluate_expr (a : ℝ) (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a = 2) :
  (a - (a^2 / (a^2 - 1))) / (a^2 / (a^2 - 1)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expr_l39_39295


namespace calc_expression_l39_39817

theorem calc_expression : | -7 | + Real.sqrt 16 - (-3)^2 = 2 := by
  have h1 : | -7 | = 7 := by sorry -- Absolute value calculation
  have h2 : Real.sqrt 16 = 4 := by sorry -- Square root calculation
  have h3 : (-3)^2 = 9 := by sorry -- Square calculation
  rw [h1, h2, h3]
  simp
  norm_num

end calc_expression_l39_39817


namespace pow_mod_eq_l39_39369

theorem pow_mod_eq (h : 101 % 100 = 1) : (101 ^ 50) % 100 = 1 :=
by
  -- Proof omitted
  sorry

end pow_mod_eq_l39_39369


namespace range_of_a_for_one_common_point_l39_39187

theorem range_of_a_for_one_common_point (a : ℝ) (a_ne_zero : a ≠ 0) :
  (∃ x : ℝ, logb 4 (a * 2^x - (4 / 3) * a) = logb 4 (4^x + 1) - (1 / 2) * x) ↔ (a > 1 ∨ a = -3) :=
by 
  sorry

end range_of_a_for_one_common_point_l39_39187


namespace point_M_lies_on_line_AB_l39_39339

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry

-- Definitions and assumptions from the conditions
variable (ω1 ω2 : Circle α) (A B M : Point α)
variable (tangent1 tangent2 : α)
variable (Ha : ω1 ∩ ω2 = {A, B})
variable (Ht1 : Tangent M ω1 tangent1)
variable (Ht2 : Tangent M ω2 tangent2)
variable (Hlen : TangentLength M ω1 tangent1 = TangentLength M ω2 tangent2)

-- Claim that needs to be proved 
theorem point_M_lies_on_line_AB 
    (Ha : ω1 ∩ ω2 = {A, B})
    (Ht1 : Tangent M ω1 tangent1)
    (Ht2 : Tangent M ω2 tangent2)
    (Hlen : TangentLength M ω1 tangent1 = TangentLength M ω2 tangent2) :
    LiesOnLine M (Line_through A B) :=
begin
  sorry
end

end point_M_lies_on_line_AB_l39_39339


namespace series_sum_equals_seven_ninths_l39_39844

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l39_39844


namespace missile_selection_systematic_sampling_l39_39881

theorem missile_selection_systematic_sampling :
  ∃ seq : List ℕ, seq = [3, 13, 23, 33, 43] ∧
  (∀ i, i ∈ seq → 1 ≤ i ∧ i ≤ 50) ∧
  (∀ i j, i < j → i ∈ seq → j ∈ seq → i + 10 = j) :=
by
  use [3, 13, 23, 33, 43]
  split
  . refl
  split
  . intro i hi
    repeat
      { cases hi
        simp [hi]
        norm_num }
  . intros i j hij hi hj
    repeat
      { cases hi,
        cases hj,
        simp [hi, hj],
        try { exact rfl } }
    norm_num

end missile_selection_systematic_sampling_l39_39881


namespace polynomial_multiplication_equiv_l39_39264

theorem polynomial_multiplication_equiv (x : ℝ) : 
  (x^4 + 50*x^2 + 625)*(x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := 
by 
  sorry

end polynomial_multiplication_equiv_l39_39264


namespace vector_scalar_operations_l39_39451

-- Define the vectors
def v1 : ℤ × ℤ := (2, -9)
def v2 : ℤ × ℤ := (-1, -6)

-- Define the scalars
def c1 : ℤ := 4
def c2 : ℤ := 3

-- Define the scalar multiplication of vectors
def scale (c : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (c * v.1, c * v.2)

-- Define the vector subtraction
def sub (v w : ℤ × ℤ) : ℤ × ℤ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem vector_scalar_operations :
  sub (scale c1 v1) (scale c2 v2) = (11, -18) :=
by
  sorry

end vector_scalar_operations_l39_39451


namespace fixed_point_of_transformed_exponential_l39_39700

variable (a : ℝ)
variable (h_pos : 0 < a)
variable (h_ne_one : a ≠ 1)

theorem fixed_point_of_transformed_exponential :
    (∃ x y : ℝ, (y = a^(x-2) + 2) ∧ (y = x) ∧ (x = 2) ∧ (y = 3)) :=
by {
    sorry -- Proof goes here
}

end fixed_point_of_transformed_exponential_l39_39700


namespace all_six_can_sit_l39_39605

def knows (people : Finset ℕ) (a b : ℕ) : Prop := sorry

-- Given that any subset of 5 people out of 6 can sit around a table with the knowing condition met
axiom subset_condition (group : Finset ℕ) (h₁ : group.card = 6) :
  ∀ (five_people : Finset ℕ), five_people ⊆ group → five_people.card = 5 →
  ∃ (arrangement : (Fin ℕ → ℕ)), (∀ i : ℕ, (knows group (arrangement i) (arrangement (i+1) % 5)) ∧ (arrangement i ∈ five_people) ) 

 -- Prove that all six of them can sit around a table with the knowing condition met
theorem all_six_can_sit (group : Finset ℕ) (h₁ : group.card = 6) :
  ∃ (arrangement : (Fin ℕ → ℕ)), ∀ i : ℕ, (knows group (arrangement i) (arrangement (i+1) % 6)) ∧ (arrangement i ∈ group) :=
sorry

end all_six_can_sit_l39_39605


namespace bounded_regions_l39_39618

noncomputable def regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => regions n + n + 1

theorem bounded_regions (n : ℕ) :
  (regions n = n * (n + 1) / 2 + 1) := by
  sorry

end bounded_regions_l39_39618


namespace largest_amount_received_back_l39_39743

theorem largest_amount_received_back 
  (x y x_lost y_lost : ℕ) 
  (h1 : 20 * x + 100 * y = 3000) 
  (h2 : x_lost + y_lost = 16) 
  (h3 : x_lost = y_lost + 2 ∨ x_lost = y_lost - 2) 
  : (3000 - (20 * x_lost + 100 * y_lost) = 2120) :=
sorry

end largest_amount_received_back_l39_39743


namespace solution_set_f_greater_than_4_l39_39170

def f (x: ℝ) : ℝ :=
  if x < 0 then 2 * Real.exp x else Real.log (x + 1) / Real.log 2 + 2

theorem solution_set_f_greater_than_4 :
  {x : ℝ | f x > 4} = {x : ℝ | 3 < x} :=
by
  sorry

end solution_set_f_greater_than_4_l39_39170


namespace sum_of_decimals_as_fraction_l39_39499

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l39_39499


namespace part_a_part_b_l39_39239

noncomputable def f : ℕ → ℝ → ℝ
| 1, x := x
| (n + 1), x := sqrt (f n x) - 1 / 4

theorem part_a (n : ℕ) (n_ge_1 : n ≥ 1) (x : ℝ) : 
  f (n + 1) x ≤ f n x := 
sorry

theorem part_b (n : ℕ) (n_ge_1 : n ≥ 1) : 
  f n (1 / 4) = 1 / 4 := 
sorry

end part_a_part_b_l39_39239


namespace cone_water_volume_percentage_l39_39001

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39001


namespace solution_set_l39_39173

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then Real.log (x+1) else -2 * x^2

theorem solution_set :
  {x : ℝ | f(x+2) < f(x^2 + 2*x)} = (set.Iio (-2)) ∪ (set.Ioi 1) :=
by 
  sorry

end solution_set_l39_39173


namespace general_term_formula_geometric_sequence_and_sum_l39_39166

-- Given conditions from the problem
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom a2 : a 2 = 1
axiom S11 : S 11 = 33

-- Define the general term and sum formula for the arithmetic sequence
noncomputable def a_n (n : ℕ) : ℝ := n / 2
noncomputable def S_n (n : ℕ) : ℝ := n * (a 1 + a n) / 2

theorem general_term_formula :
  (∀ n : ℕ, a n = n / 2) :=
begin
  sorry
end

variables (b : ℕ → ℝ)
noncomputable def b_n (n : ℕ) : ℝ := (1 / 4) ^ (a n)

-- Statement for the geometric sequence and its sum
theorem geometric_sequence_and_sum (n : ℕ) :
  (∀ m : ℕ, b (m + 1) / b m = 1 / 2) ∧ (∀ n : ℕ, T_n n = 1 - 1 / 2^n) :=
begin
  sorry
end

end general_term_formula_geometric_sequence_and_sum_l39_39166


namespace simple_interest_rate_calculation_l39_39584

theorem simple_interest_rate_calculation :
  ∃ r_simple : ℝ,
    let P := 5000 in
    let r_compound := 0.12 in
    let n := 2 in
    let t := 1 in
    let A_compound := P * (1 + r_compound / n)^(n * t) in
    let A_simple := A_compound - 12 in
    A_simple = P * (1 + r_simple * t) ∧
    r_simple = 0.1212 :=
begin
  sorry
end

end simple_interest_rate_calculation_l39_39584


namespace consecutive_number_other_17_l39_39714

theorem consecutive_number_other_17 (a b : ℕ) (h1 : b = 17) (h2 : a + b = 35) (h3 : a + b % 5 = 0) : a = 18 :=
sorry

end consecutive_number_other_17_l39_39714


namespace hairstylist_weekly_earnings_l39_39410

-- Definition of conditions as given in part a)
def cost_normal_haircut := 5
def cost_special_haircut := 6
def cost_trendy_haircut := 8

def number_normal_haircuts_per_day := 5
def number_special_haircuts_per_day := 3
def number_trendy_haircuts_per_day := 2

def working_days_per_week := 7

-- The goal is to prove that the hairstylist's weekly earnings equal to 413 dollars
theorem hairstylist_weekly_earnings : 
  (number_normal_haircuts_per_day * cost_normal_haircut +
  number_special_haircuts_per_day * cost_special_haircut +
  number_trendy_haircuts_per_day * cost_trendy_haircut) * 
  working_days_per_week = 413 := 
by sorry -- We use "by sorry" to skip the proof

end hairstylist_weekly_earnings_l39_39410


namespace ring_arrangement_digits_l39_39154

theorem ring_arrangement_digits :
  let m := (Nat.choose 10 6) * ((Nat.choose 9 3) - 4 * (Nat.choose 6 3)) * Nat.factorial 6 in
  (m / 10^⌊Math.log 10 m⌋.to_nat) % 10^3 = 604 :=
by
  sorry

end ring_arrangement_digits_l39_39154


namespace largest_lambda_condition_l39_39539

theorem largest_lambda_condition 
  (k : ℝ) (hk : k > 2) 
  (n : ℕ) (hn : n ≥ 3)
  (a : Fin n.succ → ℝ) (ha_pos : ∀ i, 0 < a i) 
  (lambda := (Real.sqrt (k + 4 / k + 5) + n - 3) ^ 2): 
  (∑ i in Finset.range n, a i) * (∑ i in Finset.range n, (1 / a i)) < λ → 
  a 0 + a 1 < k * a 2 :=
by sorry

end largest_lambda_condition_l39_39539


namespace combined_area_correct_l39_39530

def popsicle_stick_length_gino : ℚ := 9 / 2
def popsicle_stick_width_gino : ℚ := 2 / 5
def popsicle_stick_length_me : ℚ := 6
def popsicle_stick_width_me : ℚ := 3 / 5

def number_of_sticks_gino : ℕ := 63
def number_of_sticks_me : ℕ := 50

def side_length_square : ℚ := number_of_sticks_gino / 4 * popsicle_stick_length_gino
def area_square : ℚ := side_length_square ^ 2

def length_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_length_me
def width_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_width_me
def area_rectangle : ℚ := length_rectangle * width_rectangle

def combined_area : ℚ := area_square + area_rectangle

theorem combined_area_correct : combined_area = 6806.25 := by
  sorry

end combined_area_correct_l39_39530


namespace soccer_team_matches_impossible_l39_39448

theorem soccer_team_matches_impossible
  (teams : Fin 5 → ℕ) -- teams is a function mapping each of the 5 teams to the number of matches they participate in
  (h_match_even : ∑ i, teams i % 2 = 1) 
  : False := 
sorry

end soccer_team_matches_impossible_l39_39448


namespace find_f3_l39_39247

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 - c * x + 2

theorem find_f3 (a b c : ℝ)
  (h1 : f a b c (-3) = 9) :
  f a b c 3 = -5 :=
by
  sorry

end find_f3_l39_39247


namespace ab_plus_ad_gt_cd_l39_39277

-- Define the geometrical and angular conditions.
variable (O A B C D: Type)
variable [circumscribed quadrilateral O A B C D]
variable [central_angle A O C = 110]
variable [angle B A D = 110]
variable [angle B A C > angle A D C]

-- Define the proof statement.
theorem ab_plus_ad_gt_cd :
  AB + AD > CD :=
by
  sorry

end ab_plus_ad_gt_cd_l39_39277


namespace is_monotonically_decreasing_on_intervals_l39_39287

def f (x : ℝ) : ℝ := x ^ (-1 / 3)

theorem is_monotonically_decreasing_on_intervals :
  (∀ x y : ℝ, x < y ∧ x ∈ (-∞, 0) ∧ y ∈ (-∞, 0) → f y < f x) ∧
  (∀ x y : ℝ, x < y ∧ x ∈ (0, ∞) ∧ y ∈ (0, ∞) → f y < f x) :=
sorry

end is_monotonically_decreasing_on_intervals_l39_39287


namespace drum_roll_distance_l39_39780

open Real

structure RodDrumRollProblem where
  diameter : ℝ
  midpoint_welding : bool
  angle_A : ℝ

theorem drum_roll_distance (problem : RodDrumRollProblem) (h1 : problem.diameter = 12) (h2 : problem.midpoint_welding = true) (h3 : problem.angle_A = π / 6) : 
problem.diameter * π / 6 = 2 * π := sorry

end drum_roll_distance_l39_39780


namespace fifteenth_number_is_266_l39_39795

/-- Define the property of a number having digits that sum up to 14 -/
def sum_digits_eq_14 (n : ℕ) : Prop :=
  (n.digits 10).sum = 14

/-- Define the sequence of numbers whose digits sum up to 14 -/
def nums_with_sum_digits_14 := list.filter sum_digits_eq_14 (list.range 1000)

/-- The fifteenth number in the sequence of numbers whose digits sum to 14 is 266 -/
theorem fifteenth_number_is_266 :
  list.nth_le nums_with_sum_digits_14 14 _ = 266 := sorry

end fifteenth_number_is_266_l39_39795


namespace smallest_number_of_students_l39_39060

theorem smallest_number_of_students (n : ℕ) (mean_score : ℕ) 
  (score_90 : ℕ) (score_min : ℕ) (students_90 : ℕ) 
  (h1 : students_90 = 4)
  (h2 : score_90 = 90)
  (h3 : score_min = 70)
  (h4 : mean_score = 80)
  (minimum_score_condition : ∀ i, i < n - 4 → (score_min <= score_70[i]))
  : ∃ n ≥ 8, ∀ n', n' < n → mean_score * n' ≥ total_score  := 
begin
  sorry
end

end smallest_number_of_students_l39_39060


namespace sphere_volume_of_hexagonal_prism_l39_39037

noncomputable def volume_of_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem sphere_volume_of_hexagonal_prism
  (a h : ℝ)
  (volume : ℝ)
  (base_perimeter : ℝ)
  (vertices_on_sphere : ∀ (x y : ℝ) (hx : x^2 + y^2 = a^2) (hy : y = h / 2), x^2 + y^2 = 1) :
  volume = 9 / 8 ∧ base_perimeter = 3 →
  volume_of_sphere 1 = 4 * Real.pi / 3 :=
by
  sorry

end sphere_volume_of_hexagonal_prism_l39_39037


namespace max_alligators_in_days_l39_39399

noncomputable def days := 616
noncomputable def weeks := 88  -- derived from 616 / 7
noncomputable def alligators_per_week := 1

theorem max_alligators_in_days
  (h1 : weeks = days / 7)
  (h2 : ∀ (w : ℕ), alligators_per_week = 1) :
  weeks * alligators_per_week = 88 := by
  sorry

end max_alligators_in_days_l39_39399


namespace floor_square_sum_approx_l39_39071

noncomputable def T : ℝ :=
  ∑ i in Finset.range 3000, real.sqrt (1 + 1/(i + 1)^3 + 1/(i + 2)^3)

theorem floor_square_sum_approx :
  ∃ (T : ℝ), ∑ i in Finset.range 3000, real.sqrt (1 + 1/(i + 1)^3 + 1/(i + 2)^3) = T ∧ ⌊T^2⌋ = 9000000 :=
by
  sorry

end floor_square_sum_approx_l39_39071


namespace cone_water_volume_percentage_l39_39006

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39006


namespace distance_center_circle_to_line_l39_39726

theorem distance_center_circle_to_line
  (x a : ℝ)
  (h1 : ∃ (ABCD KLMN : ℝ), ∀ (B C K N : ℝ), B = C ∧ C = K ∧ K = N ∧ B ≠ K ∧ C ≠ N) -- condition about alignment of B, C, K, N
  (h2 : ∀ (O R : ℝ), ∃ (O : ℝ), (∀ (P : ℝ), P ≠ O) ∧ (∃ P1 P2, (P1^2 + P2^2 = (O + R)^2 )) -- condition about vertices on a circle and Pythagorean theorem
  (h3 : ∃ (s1 s2 : ℝ), s1 = x ∧ s2 = x + 1) -- condition about side lengths
  (h4 : (a + x)^2 + (x / 2)^2 = ((x + 1) / 2)^2 + (x + 1 - a)^2) :  -- equation derived from Pythagorean 
(a = 5 / 8) := sorry

end distance_center_circle_to_line_l39_39726


namespace geometric_sequence_a_equals_minus_four_l39_39587

theorem geometric_sequence_a_equals_minus_four (a : ℝ) 
(h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : a = -4 :=
sorry

end geometric_sequence_a_equals_minus_four_l39_39587


namespace proof_problem_l39_39259

noncomputable def f (x : ℝ) := 3 * Real.sin x + 2 * Real.cos x + 1

theorem proof_problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  (b * Real.cos c / a) = -1 :=
sorry

end proof_problem_l39_39259


namespace cone_volume_filled_88_8900_percent_l39_39027

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39027


namespace minimum_n_l39_39558

noncomputable def a : ℕ → ℕ
| 1       := 1
| 2       := 2
| (n + 1) := if n % 2 = 1 then a (n + 1 - 1) * 2 else a (n-1) * 2

def S : ℕ → ℕ
| 0       := 0
| (n + 1) := S n + a (n + 1)

theorem minimum_n (n : ℕ) (x : ℝ) (h : 0 < x ∧ x ≤ 2023): 
  (S (2 * n + 1) + 2) / (4 * a (2 * n)) > (x - 1) / x → 9 ≤ n := 
sorry

end minimum_n_l39_39558


namespace amount_of_bales_left_l39_39777

noncomputable def bales_left_at_year_end : ℕ :=
  let historical_harvest := 560 * 6 in
  let cooler_growth_rate := 0.75 in
  let total_cooler_bales := (cooler_growth_rate * 560 * 6).toInt in
  let weather_loss := 0.2 in
  let total_cooler_bales_with_losses := total_cooler_bales * (1 - weather_loss) in
  let additional_acres := 7 in
  let total_acres := 5 + additional_acres in
  let increased_bales_main_period := (additional_acres / 5) * historical_harvest in
  let increased_bales_cooler_period := (additional_acres / 5) * total_cooler_bales_with_losses in
  let total_main_period_bales := 3360 + 4704 in
  let total_cooler_period_bales := 2016 + 2822.4 in
  let total_harvest := total_main_period_bales + (total_cooler_period_bales / 2) in
  let total_hay_used := 
    (9 * 3 * 30) + 
    (12 * 3 * 30) + 
    (12 * 4 * 31) in
  total_harvest - total_hay_used

theorem amount_of_bales_left : bales_left_at_year_end = 7105 :=
  by
    sorry

end amount_of_bales_left_l39_39777


namespace percent_volume_filled_with_water_l39_39012

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39012


namespace sample_size_is_100_l39_39720

-- Conditions:
def scores_from_students := 100
def sampling_method := "simple random sampling"
def goal := "statistical analysis of senior three students' exam performance"

-- Problem statement:
theorem sample_size_is_100 :
  scores_from_students = 100 →
  sampling_method = "simple random sampling" →
  goal = "statistical analysis of senior three students' exam performance" →
  scores_from_students = 100 := by
sorry

end sample_size_is_100_l39_39720


namespace number_of_divisors_60_l39_39979

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l39_39979


namespace find_n_satisfies_314n_divisible_by_18_l39_39129

variable n : ℕ

def divisible_by_2 (m : ℕ) : Prop := m % 2 = 0
def sum_of_digits_314n (n : ℕ) : ℕ := 3 + 1 + 4 + n
def divisible_by_9 (m : ℕ) : Prop := m % 9 = 0
def divisible_by_18 (m : ℕ) : Prop := divisible_by_2 m ∧ divisible_by_9 m

theorem find_n_satisfies_314n_divisible_by_18 :
  (∃ n : ℕ, divisible_by_2 n ∧ divisible_by_9 (sum_of_digits_314n n) ∧ 3140 + n = 3144) :=
by
  exists 4
  sorry

end find_n_satisfies_314n_divisible_by_18_l39_39129


namespace term_position_in_sequence_l39_39947

theorem term_position_in_sequence (n : ℕ) (h : n^2 / (n^2 + 1) = 0.98) : n = 7 :=
sorry

end term_position_in_sequence_l39_39947


namespace sides_equal_max_diagonal_at_most_two_l39_39995

variable {n : ℕ}
variable (P : Polygon n)
variable (is_convex : P.IsConvex)
variable (max_diagonal : ℝ)
variable (sides_equal_max_diagonal : list ℝ)
variable (length_sides_equal_max_diagonal : sides_equal_max_diagonal.length)

-- Here we assume the basic conditions given in the problem:
-- 1. The polygon P is convex.
-- 2. The number of sides equal to the longest diagonal are stored in sides_equal_max_diagonal.

theorem sides_equal_max_diagonal_at_most_two :
  is_convex → length_sides_equal_max_diagonal ≤ 2 :=
by
  sorry

end sides_equal_max_diagonal_at_most_two_l39_39995


namespace probability_two_maths_teachers_l39_39318

theorem probability_two_maths_teachers :
  let total_teachers := 3 + 4 + 2
  let total_combinations := Nat.choose total_teachers 2
  let math_teachers := 4
  let math_combinations := Nat.choose math_teachers 2
  total_combinations = 36 → math_combinations = 6 →
  (math_combinations : ℚ) / (total_combinations : ℚ) = 1 / 6 :=
by
  intros total_teachers total_combinations math_teachers math_combinations h1 h2
  rw [← h1, ← h2]
  norm_num
  sorry

end probability_two_maths_teachers_l39_39318


namespace identify_true_statements_l39_39938

-- Definitions of the given statements
def statement1 (a x y : ℝ) : Prop := a * (x + y) = a * x + a * y
def statement2 (a x y : ℝ) : Prop := a ^ (x + y) = a ^ x + a ^ y
def statement3 (x y : ℝ) : Prop := (x + y) ^ 2 = x ^ 2 + y ^ 2
def statement4 (a b : ℝ) : Prop := Real.sqrt (a ^ 2 + b ^ 2) = a + b
def statement5 (a b c : ℝ) : Prop := a * (b / c) = (a * b) / c

-- The statement to prove
theorem identify_true_statements (a x y b c : ℝ) :
  statement1 a x y ∧ statement5 a b c ∧
  ¬ statement2 a x y ∧ ¬ statement3 x y ∧ ¬ statement4 a b :=
sorry

end identify_true_statements_l39_39938


namespace expected_value_of_coins_is_48_l39_39417

theorem expected_value_of_coins_is_48 :
  let penny := 1
      nickel := 5
      dime := 10
      quarter := 25
      half_dollar := 50
      total_value := penny + nickel + nickel + dime + quarter + half_dollar
      probability_of_heads := 1 / 2
  in (probability_of_heads * total_value = 48.0) := 
by 
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let total_value := penny + nickel + nickel + dime + quarter + half_dollar
  let probability_of_heads := 1 / 2
  have : total_value = 96 := by norm_num
  have : probability_of_heads * total_value = 48.0 := 
    by rw [mul_comm, this, mul_div_assoc, div_self, mul_one]
    exact div_ne_zero one_ne_zero two_ne_zero
  exact this

end expected_value_of_coins_is_48_l39_39417


namespace trigonometric_identity_eq_neg_one_l39_39826

theorem trigonometric_identity_eq_neg_one :
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180)) = -1 :=
by
  -- Variables needed for hypotheses
  have h₁ : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₂ : Real.sin (60 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₃ : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h₄ : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  -- Main proof
  sorry

end trigonometric_identity_eq_neg_one_l39_39826


namespace sum_of_decimals_as_fraction_l39_39498

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l39_39498


namespace percentage_decrease_in_area_l39_39432

theorem percentage_decrease_in_area {L B : ℝ} : 
  (L > 0) → (B > 0) → 
  let A_original := L * B;
  let L_new := L * 0.90;
  let B_new := B * 0.80;
  let A_new := L_new * B_new;
  ((A_original - A_new) / A_original) * 100 = 28 := 
by {
  intros;
  let A_original := L * B;
  let L_new := L * 0.90;
  let B_new := B * 0.80;
  let A_new := L_new * B_new;
  have h1 : A_new = A_original * 0.72 := by {
    calc A_new = (L * 0.90) * (B * 0.80) : by sorry
      ... = L * B * 0.90 * 0.80 : by sorry
      ... = A_original * 0.72 : by sorry,
  };
  calc ((A_original - A_new) / A_original) * 100
   = ((A_original - A_original * 0.72) / A_original) * 100 : by rw h1
  ... = ((1 - 0.72) * A_original / A_original) * 100 : by sorry
  ... = (0.28 * A_original / A_original) * 100 : by sorry
  ... = 0.28 * 100 : by sorry
  ... = 28 : by sorry,
}

end percentage_decrease_in_area_l39_39432


namespace wise_number_2022_eq_2699_l39_39203

def is_wise_number (n : ℕ) : Prop :=
  ∃ a b : ℕ, (a > 0) ∧ (b > 0) ∧ (n = a^2 - b^2)

def wise_number_sequence : List ℕ :=
  [3, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, ...]

def nth_wise_number (n : ℕ) : ℕ :=
  wise_number_sequence.get_or_else (n - 1) 0

theorem wise_number_2022_eq_2699 : nth_wise_number 2022 = 2699 :=
by {
  sorry
}

end wise_number_2022_eq_2699_l39_39203


namespace number_of_girls_in_circle_l39_39775

theorem number_of_girls_in_circle (n : ℕ) : 
  ∃ (girls : ℕ), 
  (n = 5) ∧ 
  (∀ k, (k = n → holds_hands_with_either_two_boys_or_two_girls k)) → 
  (girls = 5) :=
begin
  sorry
end

def holds_hands_with_either_two_boys_or_two_girls (n : ℕ) : Prop :=
  -- Prop specifying that each person in the circle holds hands with either two boys or two girls
  sorry

end number_of_girls_in_circle_l39_39775


namespace trigonometric_identity_l39_39832

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l39_39832


namespace solution_function_form_l39_39240

noncomputable def find_function (P : ℝ → ℝ) (t : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f(x + t) - f(x) = P(x)

theorem solution_function_form (P : ℝ → ℝ) (t : ℝ) : (∃ f : ℝ → ℝ, ∀ x : ℝ, f(x + t) - f(x) = P(x)) →
  (∃ g : ℝ → ℝ, (∀ x : ℝ, 0 ≤ x ∧ x < t → g(x) = f(x)) ∧
    ∀ x : ℝ, f(x) = g(x) + ∑ i in finset.range (⌊ x / t ⌋₊), P(x - i * t)) :=
begin
  sorry
end

end solution_function_form_l39_39240


namespace H1_H2_bisect_AC_l39_39068

-- Define the setup of the problem
variables {A B C H H1 H2 : Type} 
predicate orthocenter {A B C H : Type} : Prop :=
  orthocenter A B C H

predicate perpendicular_from {X Y : Type} (Z : Type) : Prop :=
  perpendicular X Y Z

predicate bisection {X Y Z W : Type} : Prop :=
  bisects X Y Z W

-- Problem statement in Lean
theorem H1_H2_bisect_AC
  (h_orthocenter : orthocenter A B C H)
  (h_perpendicular_H1 : perpendicular_from H (angle_bisector_of B) H1)
  (h_perpendicular_H2 : perpendicular_from H (external_angle_bisector_of B) H2) :
  bisection H1 H2 A C :=
begin
  sorry
end

end H1_H2_bisect_AC_l39_39068


namespace fractions_ordered_l39_39364

theorem fractions_ordered :
  (2 / 5 : ℚ) < (3 / 5) ∧ (3 / 5) < (4 / 6) ∧ (4 / 6) < (4 / 5) ∧ (4 / 5) < (6 / 5) ∧ (6 / 5) < (4 / 3) :=
by
  sorry

end fractions_ordered_l39_39364


namespace tom_books_after_transactions_l39_39758

-- Define the initial conditions as variables
def initial_books : ℕ := 5
def sold_books : ℕ := 4
def new_books : ℕ := 38

-- Define the property we need to prove
theorem tom_books_after_transactions : initial_books - sold_books + new_books = 39 := by
  sorry

end tom_books_after_transactions_l39_39758


namespace at_least_two_shoes_do_not_form_a_pair_l39_39882

variable (S : Type) (shoes : Finset (Finset S)) (selected : Finset S)
  [decidable_eq S]

-- Conditions
def four_different_pairs_of_shoes : Prop :=
  shoes.card = 4 ∧ ∀ s ∈ shoes, s.card = 2

def four_shoes_randomly_selected : Prop :=
  selected.card = 4

-- Question (translated as a proposition)
def complementary_event_all_four_shoes_form_pairs : Prop :=
  ∃ s1 s2 ∈ shoes, s1 ≠ s2 ∧
  ∃ a b ∈ s1, a ∉ selected ∧ b ∉ selected ∨
  ∃ c d ∈ s2, c ∉ selected ∧ d ∉ selected

-- Equivalent proof
theorem at_least_two_shoes_do_not_form_a_pair
  (h1 : four_different_pairs_of_shoes shoes)
  (h2 : four_shoes_randomly_selected selected) :
  complementary_event_all_four_shoes_form_pairs shoes selected := by
  sorry

end at_least_two_shoes_do_not_form_a_pair_l39_39882


namespace bobby_position_after_100_turns_l39_39815

def movement_pattern (start_pos : ℤ × ℤ) (n : ℕ) : (ℤ × ℤ) :=
  let x := start_pos.1 - ((2 * (n / 4 + 1) + 3 * (n / 4)) * ((n + 1) / 4))
  let y := start_pos.2 + ((2 * (n / 4 + 1) + 2 * (n / 4)) * ((n + 1) / 4))
  if n % 4 == 0 then (x, y)
  else if n % 4 == 1 then (x, y + 2 * ((n + 3) / 4) + 1)
  else if n % 4 == 2 then (x - 3 * ((n + 5) / 4), y + 2 * ((n + 3) / 4) + 1)
  else (x - 3 * ((n + 5) / 4) + 3, y + 2 * ((n + 3) / 4) - 2)

theorem bobby_position_after_100_turns :
  movement_pattern (10, -10) 100 = (-667, 640) :=
sorry

end bobby_position_after_100_turns_l39_39815


namespace volume_of_region_l39_39522

theorem volume_of_region : 
  ∀ (x y z : ℝ),
  abs (x + y + z) + abs (x - y + z) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
  ∃ V : ℝ, V = 62.5 :=
  sorry

end volume_of_region_l39_39522


namespace profit_percentage_l39_39439

theorem profit_percentage (SP CP : ℕ) (h₁ : SP = 800) (h₂ : CP = 640) : (SP - CP) / CP * 100 = 25 :=
by 
  sorry

end profit_percentage_l39_39439


namespace find_price_per_gallon_sour_cream_l39_39070
noncomputable theory

def milk_gallons := 16
def fraction_milk_to_sour_cream := 1 / 4
def fraction_milk_to_butter := 1 / 4
def milk_to_butter_ratio := 4
def milk_to_sour_cream_ratio := 2
def price_per_gallon_butter := 5
def price_per_gallon_whole_milk := 3
def total_revenue := 41

def price_per_gallon_sour_cream := 6

theorem find_price_per_gallon_sour_cream:
  let sour_cream := fraction_milk_to_sour_cream * milk_gallons,
      butter := fraction_milk_to_butter * milk_gallons,
      whole_milk := milk_gallons - (sour_cream + butter),
      butter_made := butter / milk_to_butter_ratio,
      sour_cream_made := sour_cream / milk_to_sour_cream_ratio,
      revenue_from_butter := butter_made * price_per_gallon_butter,
      revenue_from_whole_milk := whole_milk * price_per_gallon_whole_milk in
  total_revenue - (revenue_from_butter + revenue_from_whole_milk) = price_per_gallon_sour_cream * sour_cream_made :=
by 
  sorry

end find_price_per_gallon_sour_cream_l39_39070


namespace sum_of_digits_A_plus_B_l39_39329

-- Defining a function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_A_plus_B
  (A B : ℕ)
  (h1 : sum_of_digits A = 19)
  (h2 : sum_of_digits B = 20)
  (h3 : (A + B).digits 10.filter (λ d, d = 0).length = 2) :
  sum_of_digits (A + B) = 21 :=
    sorry

end sum_of_digits_A_plus_B_l39_39329


namespace distance_between_parallel_lines_l39_39341

theorem distance_between_parallel_lines 
  (x y : ℝ) 
  (line1 : 3 * x + 4 * y - 12 = 0)
  (line2 : 6 * x + 8 * y + 11 = 0) :
  let a := 6 in 
  let b := 8 in
  let c1 := -24 in
  let c2 := 11 in
  abs (c1 - c2) / sqrt (a^2 + b^2) = 7 / 2 :=
by 
  sorry

end distance_between_parallel_lines_l39_39341


namespace floor_sqrt_sum_l39_39470

open Real

theorem floor_sqrt_sum : (∑ k in Finset.range 25, ⌊sqrt (k + 1)⌋) = 75 := 
by
  sorry

end floor_sqrt_sum_l39_39470


namespace largest_possible_difference_l39_39086

/-- Estimates suggest that Charles attends a baseball game in Chicago, estimating 80,000 fans, 
and Dale attends a game in Detroit with an estimate of 95,000 fans. 
Conditions:
1. The actual attendance in Chicago is within 5% of Charles' estimate.
2. The actual attendance in Detroit is within 15% of Dale's estimate.
/-- Prove that the largest possible difference between the attendances at the two games, 
rounded to the nearest 1,000, is 36,000.
-/
theorem largest_possible_difference 
    (C_est : ℝ) (D_est : ℝ) 
    (C_actual_min : ℝ) (C_actual_max : ℝ) 
    (D_actual_min : ℝ) (D_actual_max : ℝ)
    (h1 : C_est = 80000)
    (h2 : D_est = 95000)
    (h3 : C_actual_min = C_est * 0.95)
    (h4 : C_actual_max = C_est * 1.05)
    (h5 : D_actual_min = D_est / 1.15)
    (h6 : D_actual_max = D_est / 0.85) :
    round ((D_actual_max - C_actual_min) / 1000) * 1000 = 36000 := 
sorry

end largest_possible_difference_l39_39086


namespace seq_formula_l39_39557

noncomputable def seq {a : Nat → ℝ} (h1 : a 2 - a 1 = 1) (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1) : Nat → ℝ :=
sorry

theorem seq_formula {a : Nat → ℝ} 
  (h1 : a 2 - a 1 = 1)
  (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1)
  (n : Nat) : a n = 2 ^ n - 1 :=
sorry

end seq_formula_l39_39557


namespace print_shop_cost_difference_l39_39128

-- Definitions of the conditions
def cost_per_copy_X := 1.20
def cost_per_copy_Y := 1.70
def number_of_copies := 40

-- Total cost calculations
def total_cost_X := cost_per_copy_X * number_of_copies
def total_cost_Y := cost_per_copy_Y * number_of_copies

-- Proof statement
theorem print_shop_cost_difference :
  total_cost_Y - total_cost_X = 20.00 :=
by
  -- The following line is a placeholder to ensure the code builds successfully.
  -- The actual proof steps are skipped (using 'sorry').
  sorry

end print_shop_cost_difference_l39_39128


namespace wendy_third_day_miles_l39_39667

theorem wendy_third_day_miles (miles_day1 miles_day2 total_miles : ℕ)
  (h1 : miles_day1 = 125)
  (h2 : miles_day2 = 223)
  (h3 : total_miles = 493) :
  total_miles - (miles_day1 + miles_day2) = 145 :=
by sorry

end wendy_third_day_miles_l39_39667


namespace sum_of_floor_sqrt_l39_39458

theorem sum_of_floor_sqrt :
  (∑ i in Finset.range 25, (Nat.sqrt (i + 1))) = 75 := by
  sorry

end sum_of_floor_sqrt_l39_39458


namespace ratio_of_numbers_l39_39711

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : 0 < a) (h3 : b < a) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
sorry

end ratio_of_numbers_l39_39711


namespace quadratic_conclusions_l39_39931

variables {a b c : ℝ} (h1 : a < 0) (h2 : a - b + c = 0)

theorem quadratic_conclusions
    (h_intersect : ∃ x, a * x ^ 2 + b * x + c = 0 ∧ x = -1)
    (h_symmetry : ∀ x, x = 1 → a * (x - 1) ^ 2 + b * (x - 1) + c = a * (x + 1) ^ 2 + b * (x + 1) + c) :
    a - b + c = 0 ∧ 
    (∀ m : ℝ, a * m ^ 2 + b * m + c ≤ -4 * a) ∧ 
    (∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c + 1 = 0 ∧ a * x2 ^ 2 + b * x2 + c + 1 = 0 ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
begin
    sorry
end

end quadratic_conclusions_l39_39931


namespace painting_cubes_probability_l39_39723

def number_of_faces := 6

def number_of_colors := 3

def total_paintings_per_cube := number_of_colors ^ number_of_faces

def probability := (exact_division : ℚ) (32, 9841)

theorem painting_cubes_probability :
  let valid_paintings := 132  -- based on detailed symmetry and distribution.
  (valid_paintings / total_paintings_per_cube) = probability :=
by
  sorry

end painting_cubes_probability_l39_39723


namespace not_construct_frame_without_breaking_wire_minimum_breaks_to_construct_frame_l39_39747

-- Definition of the problem conditions
def length_of_wire : ℕ := 120
def edge_length_of_cube : ℕ := 10
def number_of_edges_of_cube : ℕ := 12

-- Part (a): It is impossible to construct the frame without breaking the wire.

theorem not_construct_frame_without_breaking_wire (length_of_wire : ℕ) (edge_length_of_cube : ℕ) (number_of_edges_of_cube: ℕ) :
(length_of_wire = edge_length_of_cube * number_of_edges_of_cube) → ¬(∃ (trace_path : bool), trace_path = true) :=
by
  sorry

-- Part (b): Minimum number of breaks required to construct the frame

theorem minimum_breaks_to_construct_frame (odd_vertices : ℕ) :
(odd_vertices = 8) → ∃ (breaks : ℕ), breaks = 3 :=
by
  sorry

end not_construct_frame_without_breaking_wire_minimum_breaks_to_construct_frame_l39_39747


namespace probability_home_appliance_correct_maximum_m_profitable_l39_39085

namespace Promotions

open Nat

def probability_at_least_one_home_appliance : ℝ :=
  1 - (choose 6 3 : ℕ) / (choose 8 3 : ℕ)

theorem probability_home_appliance_correct : probability_at_least_one_home_appliance = 9 / 14 :=
  sorry

def expected_value_of_lottery (m : ℝ) : ℝ :=
  0 * (8 / 27) +
  m * (4 / 9) +
  3 * m * (2 / 9) +
  6 * m * (1 / 27)

theorem maximum_m_profitable (m : ℝ) (h : expected_value_of_lottery m ≤ 100) : m ≤ 75 :=
  sorry

end Promotions

end probability_home_appliance_correct_maximum_m_profitable_l39_39085


namespace sum_of_possible_values_l39_39708

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) : (∃ x y : ℝ, x * (x - 4) = -21 ∧ y * (y - 4) = -21 ∧ x + y = 4) :=
sorry

end sum_of_possible_values_l39_39708


namespace trigonometric_expression_value_l39_39823

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l39_39823


namespace magnitude_of_c_l39_39893

-- Define the polynomial Q(x)
def Q (x c : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - c*x + 1) * (x^2 - 2*x + 5)

-- Provided conditions as hypotheses
variables (c : ℂ)
hypothesis (h : ∃ (r : list ℂ), r.length = 5 ∧ ∀ x ∈ r, Q x c = 0)

-- Proof statement
theorem magnitude_of_c (c : ℂ) (h : ∃ (r : list ℂ), r.length = 5 ∧ ∀ x ∈ r, Q x c = 0) : |c| = 5 :=
sorry

end magnitude_of_c_l39_39893


namespace smallest_positive_period_and_intervals_of_increase_range_of_f_on_interval_l39_39565

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + 2 * sqrt 3 * sin x * cos x + 3 * cos x ^ 2 - 2

-- Statement for Proof Problem 1
theorem smallest_positive_period_and_intervals_of_increase (k : ℤ) :
  (∃ T > 0, ∀ x : ℝ, f (x + T) = f x) ∧ (∀ k : ℤ, ∀ x ∈ (Icc (-π/3 + k * π) (π/6 + k * π)), ∃ δ > 0, monotone_on f (x + 0 .. x + δ)) :=
sorry

-- Statement for Proof Problem 2
theorem range_of_f_on_interval :
  ∀ x ∈ (Icc (-π/6) (π/3)), f x ∈ (Icc (-1/2) (1)) :=
sorry

end smallest_positive_period_and_intervals_of_increase_range_of_f_on_interval_l39_39565


namespace slope_of_tangent_line_at_x_equals_two_l39_39323

noncomputable def slope_of_tangent_at_two : ℝ := -1 / 4

theorem slope_of_tangent_line_at_x_equals_two (x : ℝ) (hx : x = 2) :
  (deriv (λ x : ℝ, 1 / x)) x = slope_of_tangent_at_two :=
by
  rw hx
  sorry

end slope_of_tangent_line_at_x_equals_two_l39_39323


namespace roots_in_arithmetic_progression_l39_39729

theorem roots_in_arithmetic_progression (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x2 = (x1 + x3) / 2) ∧ (x1 + x2 + x3 = -a) ∧ (x1 * x3 + x2 * (x1 + x3) = b) ∧ (x1 * x2 * x3 = -c)) ↔ 
  (27 * c = 3 * a * b - 2 * a^3 ∧ 3 * b ≤ a^2) :=
sorry

end roots_in_arithmetic_progression_l39_39729


namespace verify_z_relationship_l39_39721

variable {x y z : ℝ}

theorem verify_z_relationship (h1 : x > y) (h2 : y > 1) :
  z = (x + 3) - 2 * (y - 5) → z = x - 2 * y + 13 :=
by
  intros
  sorry

end verify_z_relationship_l39_39721


namespace sum_s_1_to_321_l39_39649

-- Define the function s(n) to compute the sum of all odd digits of n
def s (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (λ d, d % 2 = 1) |>.sum

-- Now state the problem
theorem sum_s_1_to_321 : 
  (∑ n in Finset.range 322, s n) = 1727 :=
sorry

end sum_s_1_to_321_l39_39649


namespace prob_red_or_blue_l39_39330

open Nat

noncomputable def total_marbles : Nat := 90
noncomputable def prob_white : (ℚ) := 1 / 6
noncomputable def prob_green : (ℚ) := 1 / 5

theorem prob_red_or_blue :
  let prob_total := 1
  let prob_white_or_green := prob_white + prob_green
  let prob_red_blue := prob_total - prob_white_or_green
  prob_red_blue = 19 / 30 := by
    sorry

end prob_red_or_blue_l39_39330


namespace profit_on_next_sales_approx_l39_39414

-- Define the conditions as given in problem
def initial_sales : ℝ := 1000000
def initial_profit : ℝ := 125000
def next_sales : ℝ := 2000000
def approx_decrease_ratio : ℝ := 2.0 -- Approximately 200%

-- Define the original ratio of profit to sales for the first $1 million
def original_ratio : ℝ := initial_profit / initial_sales

-- Estimate the new ratio assuming a significant decrease
def estimated_new_ratio : ℝ := original_ratio * 0.01

-- Define the proof goal: the profit on the next $2 million in sales is approximately $2,500
theorem profit_on_next_sales_approx :
  let profit := next_sales * estimated_new_ratio in abs (profit - 2500) < 1 :=
by
  sorry

end profit_on_next_sales_approx_l39_39414


namespace milk_leftover_l39_39474

-- Definitions of the conditions
def milk_per_day := 16
def kids_consumption_percent := 0.75
def cooking_usage_percent := 0.50

-- The theorem to prove
theorem milk_leftover : 
  let milk_consumed_by_kids := milk_per_day * kids_consumption_percent in
  let remaining_milk_after_kids := milk_per_day - milk_consumed_by_kids in
  let milk_used_for_cooking := remaining_milk_after_kids * cooking_usage_percent in
  let milk_leftover := remaining_milk_after_kids - milk_used_for_cooking in
  milk_leftover = 2 := 
sorry

end milk_leftover_l39_39474


namespace probability_laurent_greater_chloe_l39_39074

noncomputable def chloe_and_laurent_probability : ℝ :=
  let chloe_dist := set.Icc (0 : ℝ) 1000
  let laurent_dist := set.Icc (500 : ℝ) 3000
  let total_area := 1000 * 2500
  let favorable_area := (1/2 : ℝ) * 500 * 500 + 1000 * 2000
  favorable_area / total_area

theorem probability_laurent_greater_chloe : chloe_and_laurent_probability = 0.85 := 
  by
    sorry

end probability_laurent_greater_chloe_l39_39074


namespace quadratic_properties_l39_39930

theorem quadratic_properties (a b c : ℝ) (h₀ : a < 0) (h₁ : a - b + c = 0) :
  (am² b c - 4 a) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ x1 x2 : ℝ, (∃ h : a * x1^2 + b * x1 + c + 1 = 0, ∃ h2 : a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39930


namespace sum_of_floor_sqrt_l39_39455

theorem sum_of_floor_sqrt :
  (∑ n in Finset.range 26, Int.floor (Real.sqrt n)) = 75 :=
by
  -- skipping proof details
  sorry

end sum_of_floor_sqrt_l39_39455


namespace pairwise_coprime_max_not_representable_ge_sqrt2abc_l39_39257

-- Definitions and conditions
def is_representable (a b c : ℕ) (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x * a + y * b + z * c ∈ set.univ

def g (a b c : ℕ) : ℕ :=
  @Classical.some (ℕ → Prop) Classical.prop_decidable sorry

-- Main theorem to prove
theorem pairwise_coprime_max_not_representable_ge_sqrt2abc
  (a b c : ℕ)
  (h_pairwise_coprime : Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1)
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_c_pos : c > 0) :
  g(a, b, c) ≥ Nat.sqrt (2 * a * b * c) := 
sorry

end pairwise_coprime_max_not_representable_ge_sqrt2abc_l39_39257


namespace sufficient_condition_necessary_condition_l39_39242

variables {α β γ : ℝ}
variable {ABC : Type}
variables {BC XY : ℝ}

def perpendicular_bisector (AB : ℝ) : ℝ := sorry   -- Definition placeholder
def intersects (a b : ℝ) : ℝ := sorry  -- Definition placeholder

theorem sufficient_condition (h₁ : α + β + γ = π)
                             (h₂ : intersects (perpendicular_bisector AB) BC = X)
                             (h₃ : intersects (perpendicular_bisector AC) BC = Y)
                             (h₄ : tan β * tan γ = 3) : BC = XY :=
by sorry

theorem necessary_condition (h₁ : α + β + γ = π)
                            (h₂ : intersects (perpendicular_bisector AB) BC = X)
                            (h₃ : intersects (perpendicular_bisector AC) BC = Y) : 
                            BC = XY ↔ tan β * tan γ ∈ ({-1, 3} : Set ℝ) :=
by sorry

end sufficient_condition_necessary_condition_l39_39242


namespace distribute_positions_l39_39770

theorem distribute_positions :
  let positions := 11
  let classes := 6
  ∃ total_ways : ℕ, total_ways = Nat.choose (positions - 1) (classes - 1) ∧ total_ways = 252 :=
by
  let positions := 11
  let classes := 6
  have : Nat.choose (positions - 1) (classes - 1) = 252 := by sorry
  exact ⟨Nat.choose (positions - 1) (classes - 1), this, this⟩

end distribute_positions_l39_39770


namespace num_divisors_60_l39_39969

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l39_39969


namespace max_length_OB_is_sqrt2_l39_39346

noncomputable def max_length_OB : ℝ :=
  let θ : ℝ := real.pi / 4 in
  let AB : ℝ := 1 in
  max (AB / real.sin θ * real.sin (real.pi / 2))

theorem max_length_OB_is_sqrt2 : max_length_OB = real.sqrt 2 := 
sorry

end max_length_OB_is_sqrt2_l39_39346


namespace convex_pentagon_impossible_l39_39073

theorem convex_pentagon_impossible (lengths : List ℕ) (h : lengths = [2, 3, 5, 7, 8, 9, 10, 11, 13, 15]) :
  ¬∃ (s₁ s₂ s₃ s₄ s₅ d₁ d₂ d₃ d₄ d₅ : ℕ), 
    lengths = [s₁, s₂, s₃, s₄, s₅, d₁, d₂, d₃, d₄, d₅] ∧ 
    s₁ + s₂ + s₃ + s₄ + s₅ + d₁ + d₂ + d₃ + d₄ + d₅ = 83 ∧
    (∀ (a b c : ℕ), List.mem a lengths ∧ List.mem b lengths ∧ List.mem c lengths → a + b > c ∧ b + c > a ∧ a + c > b) := 
sorry

end convex_pentagon_impossible_l39_39073


namespace beth_time_more_minutes_l39_39235

-- Definitions for John's trip
def john_speed : ℝ := 40
def john_time_hours : ℝ := 0.5
def john_distance : ℝ := john_speed * john_time_hours

-- Definitions for Beth's trip
def beth_distance : ℝ := john_distance + 5
def beth_speed : ℝ := 30
def beth_time_hours : ℝ := beth_distance / beth_speed

-- Conversion to minutes
def john_time_minutes : ℝ := john_time_hours * 60
def beth_time_minutes : ℝ := beth_time_hours * 60

-- Prove the time difference
theorem beth_time_more_minutes : beth_time_minutes - john_time_minutes = 20 := 
  sorry

end beth_time_more_minutes_l39_39235


namespace product_divisible_by_14_l39_39636

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7 * a + 8 * b = 14 * c + 28 * d) : 14 ∣ a * b := 
sorry

end product_divisible_by_14_l39_39636


namespace cone_volume_filled_88_8900_percent_l39_39024

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39024


namespace exists_x_y_infinitely_many_pairs_l39_39396

noncomputable theory

open Nat Real Rational

-- Part (a)
theorem exists_x_y (y : ℚ) : ∃ x : ℕ, sqrt (x + sqrt x) = y := sorry

-- Part (b)
theorem infinitely_many_pairs : ∃ᶠ (xy : ℚ × ℚ) in filter.univ, sqrt (xy.1 + sqrt xy.1) = xy.2 := sorry

end exists_x_y_infinitely_many_pairs_l39_39396


namespace regular_octagon_area_l39_39362

theorem regular_octagon_area (a : ℝ) : 
  let A := 2 * a^2 * (1 + Real.sqrt 2)
  in A = 2 * a^2 * (1 + Real.sqrt 2) :=
by
  sorry

end regular_octagon_area_l39_39362


namespace max_two_way_airline_number_l39_39299

-- Definitions and conditions
variables (n : ℕ) (C : Type) (cities : fin n → C)
variable (d_A : C → ℕ)
axiom (forall_A : ∀ (A : C), A ≠ C → d_A A ≤ n)
axiom (forall_AB : ∀ (A B : C), A ≠ C ∧ B ≠ C ∧ (A, B) ∉ two_way_connections → d_A A + d_A B ≤ n)

-- Prove the maximum number of two-way airlines
def max_two_way_airlines : ℕ := (n * (n + 1)) / 2

theorem max_two_way_airline_number :
  ∃ (configuration : C → C → Prop), is_valid_configuration configuration ∧
  two_way_airline_count configuration = max_two_way_airlines n :=
sorry

end max_two_way_airline_number_l39_39299


namespace red_hood_equations_correct_l39_39261

noncomputable def red_hood_system (x y : ℝ) : Prop :=
  (2 / 60 * x + 3 / 60 * y = 1.5) ∧ (x + y = 18)

theorem red_hood_equations_correct :
  ∃ x y : ℝ, red_hood_system x y :=
begin
  use 9, -- assumed values
  use 9,
  dsimp [red_hood_system],
  split;
  norm_num,
  sorry
end

end red_hood_equations_correct_l39_39261


namespace probability_density_function_condition_l39_39179

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (-Real.abs x)

theorem probability_density_function_condition (a : ℝ) :
  (∫ x in -∞..∞, f a x) = 1 → a = 1 / 2 :=
by
  sorry

end probability_density_function_condition_l39_39179


namespace trigonometric_identity_l39_39830

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l39_39830


namespace solve_linear_system_l39_39297

/-- Let x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈ be real numbers that satisfy the following system of equations:
1. x₁ + x₂ + x₃ = 6
2. x₂ + x₃ + x₄ = 9
3. x₃ + x₄ + x₅ = 3
4. x₄ + x₅ + x₆ = -3
5. x₅ + x₆ + x₇ = -9
6. x₆ + x₇ + x₈ = -6
7. x₇ + x₈ + x₁ = -2
8. x₈ + x₁ + x₂ = 2
Prove that the solution is
  x₁ = 1, x₂ = 2, x₃ = 3, x₄ = 4, x₅ = -4, x₆ = -3, x₇ = -2, x₈ = -1
-/
theorem solve_linear_system :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ),
  x₁ + x₂ + x₃ = 6 →
  x₂ + x₃ + x₄ = 9 →
  x₃ + x₄ + x₅ = 3 →
  x₄ + x₅ + x₆ = -3 →
  x₅ + x₆ + x₇ = -9 →
  x₆ + x₇ + x₈ = -6 →
  x₇ + x₈ + x₁ = -2 →
  x₈ + x₁ + x₂ = 2 →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by
  intros x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, the proof steps would go
  sorry

end solve_linear_system_l39_39297


namespace floor_sqrt_sum_l39_39466

theorem floor_sqrt_sum : (∑ x in Finset.range 25 + 1, (Nat.floor (Real.sqrt x : ℝ))) = 75 := 
begin
  sorry
end

end floor_sqrt_sum_l39_39466


namespace find_angle_C_l39_39626

theorem find_angle_C
  (A B C D E : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  (ABC : Triangle) 
  (hB : angle ABC.B = 90)
  (hMedians : medians_perpendicular ABC.AD ABC.BE) :
  angle ABC.C = arctan (1 / sqrt 2) := 
sorry

end find_angle_C_l39_39626


namespace sin_value_of_sum_angle_l39_39912

theorem sin_value_of_sum_angle 
  (α : ℝ) (h0 : α ∈ Ioo 0 (π / 3)) 
  (h1 : cos (α - π / 6) + sin α = (4 / 5) * sqrt 3) : 
  sin (α + 5 / 12 * π) = (7 * sqrt 2) / 10 :=
by
  -- Proof required here
  sorry

end sin_value_of_sum_angle_l39_39912


namespace number_of_divisors_60_l39_39976

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l39_39976


namespace trigonometric_expression_value_l39_39822

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l39_39822


namespace evaluate_power_l39_39087

open Real

theorem evaluate_power : (-32 : ℝ)^(5/3) = -256 * cbrt 2 := 
by
  sorry

end evaluate_power_l39_39087


namespace correct_sample_size_of_survey_l39_39335

theorem correct_sample_size_of_survey 
    (total_students : ℕ)
    (selected_students : ℕ)
    (survey_is_comprehensive : Prop)
    (students_are_population : Prop)
    (student_is_individual : Prop)
    (correct_statement : selected_students = 200) 
    (total_students_val : total_students = 2000)
    (selected_students_val : selected_students = 200) :
    correct_statement :=
by
    sorry

end correct_sample_size_of_survey_l39_39335


namespace billy_books_read_l39_39813

def hours_per_day : ℕ := 8
def days_per_weekend : ℕ := 2
def reading_percentage : ℚ := 0.25
def pages_per_hour : ℕ := 60
def pages_per_book : ℕ := 80

theorem billy_books_read :
  let total_hours := hours_per_day * days_per_weekend in
  let reading_hours := total_hours * reading_percentage in
  let total_pages := reading_hours * pages_per_hour in
  let books_read := total_pages / pages_per_book in
  books_read = 3 :=
by
  sorry

end billy_books_read_l39_39813


namespace num_divisors_sixty_l39_39984

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l39_39984


namespace common_tangent_curves_l39_39207

theorem common_tangent_curves (s t a : ℝ) (e : ℝ) (he : e > 0) :
  (t = (1 / (2 * e)) * s^2) →
  (t = a * Real.log s) →
  (s / e = a / s) →
  a = 1 :=
by
  intro h1 h2 h3
  sorry

end common_tangent_curves_l39_39207


namespace gcd_lcm_product_l39_39118

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end gcd_lcm_product_l39_39118


namespace quadratic_properties_l39_39923

def quadratic_function (a b c : ℝ) := λ x, a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h0 : a < 0) (h1 : quadratic_function a b c (-1) = 0)
  (h2 : ∀ m : ℝ, let f := quadratic_function a b c in f(m) ≤ -4 * a)
  (h3 : b = -2 * a) (h4: c = -3 * a):
  (a - b + c = 0) ∧ 
  (∀ x1 x2 : ℝ, quadratic_function a b (c + 1) x1 = 0 → quadratic_function a b (c + 1) x2 = 0 →
    x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39923


namespace sequence_root_formula_l39_39541

theorem sequence_root_formula {a : ℕ → ℝ} 
    (h1 : ∀ n, (a (n + 1))^2 = (a n)^2 + 4)
    (h2 : a 1 = 1)
    (h3 : ∀ n, a n > 0) :
    ∀ n, a n = Real.sqrt (4 * n - 3) := 
sorry

end sequence_root_formula_l39_39541


namespace find_w_squared_l39_39583

theorem find_w_squared (w : ℝ) :
  (w + 15)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = ((-21 + Real.sqrt 7965) / 22)^2 ∨ 
        w^2 = ((-21 - Real.sqrt 7965) / 22)^2 :=
by sorry

end find_w_squared_l39_39583


namespace degree_of_h_is_3_l39_39304

-- Definitions for the polynomials f(z) and g(z)
def f (a3 a2 a1 a0 : ℝ) (z : ℝ) := a3 * z^3 + a2 * z^2 + a1 * z + a0
def g (b2 b1 b0 : ℝ) (z : ℝ) := b2 * z^2 + b1 * z + b0

-- Assumptions
variables {a3 a2 a1 a0 b2 b1 b0 : ℝ}
variable (z : ℝ)
axiom h_a3_nonzero : a3 ≠ 0
axiom h_b2_nonzero : b2 ≠ 0

-- Definition of the sum of the polynomials f and g
def h (a3 a2 a1 a0 b2 b1 b0 : ℝ) (z : ℝ) := f a3 a2 a1 a0 z + g b2 b1 b0 z

-- Degree of the polynomial h(z)
theorem degree_of_h_is_3 : nat_degree (h a3 a2 a1 a0 b2 b1 b0 z) = 3 :=
by 
  sorry

end degree_of_h_is_3_l39_39304


namespace circle_inscribed_ratio_proof_l39_39340

noncomputable def r1 := Real.sqrt 2
noncomputable def r2 := 2
noncomputable def d := Real.sqrt 3 + 1
noncomputable def s := (π * (3 + r1 - Real.sqrt 3 - Real.sqrt (6)) / 4)
noncomputable def S0 := (π * 7 - 6 * (Real.sqrt 3 + 1)) / 6
noncomputable def ratio := (3 * s) / S0

theorem circle_inscribed_ratio_proof :
  ratio = (3 * π * (3 + Real.sqrt 2 - Real.sqrt 3 - Real.sqrt 6) / (7 * π - 6 * (Real.sqrt 3 + 1))) := sorry

end circle_inscribed_ratio_proof_l39_39340


namespace sum_of_integers_l39_39320

theorem sum_of_integers (m n : ℕ) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by
  sorry

end sum_of_integers_l39_39320


namespace total_pages_in_sci_fi_books_l39_39315

theorem total_pages_in_sci_fi_books 
  (books : ℕ) 
  (pages_per_book : ℕ) 
  (total_pages : ℕ) 
  (h_books : books = 8) 
  (h_pages_per_book : pages_per_book = 478)
  (h_total_pages : total_pages = books * pages_per_book) : 
  total_pages = 3824 :=
by {
  rw [h_books, h_pages_per_book] at h_total_pages,
  exact h_total_pages,
  sorry
}

end total_pages_in_sci_fi_books_l39_39315


namespace sum_of_odd_digits_upto_321_l39_39648

-- Define the sum of odd digits function
def s (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.filter (λ d => d % 2 = 1).sum

-- The specific problem statement
theorem sum_of_odd_digits_upto_321 :
  (∑ n in Finset.range 322, s n) = 1727 :=
by
  sorry

end sum_of_odd_digits_upto_321_l39_39648


namespace smallest_number_with_5_zeros_7_ones_divisible_by_11_l39_39866

-- Define the set of all natural numbers with exactly 5 zeros and 7 ones.
def specific_number (n : ℕ) : Prop :=
  let digits := nat.digits 10 n in
  digits.count 0 = 5 ∧ digits.count 1 = 7

-- Define the divisibility rule for 11.
def divisible_by_11 (n : ℕ) : Prop :=
  (let digits := nat.digits 10 n in
  (digits.enum_from 0).sum (λ ⟨i, d⟩, if i % 2 = 0 then d else -d)) % 11 = 0

-- The main proof problem statement
theorem smallest_number_with_5_zeros_7_ones_divisible_by_11 :
  ∃ n : ℕ, specific_number n ∧ divisible_by_11 n ∧ (∀ m : ℕ, (specific_number m ∧ divisible_by_11 m) → n ≤ m) ∧ n = 1000001111131 :=
begin
  sorry
end

end smallest_number_with_5_zeros_7_ones_divisible_by_11_l39_39866


namespace measure_of_angle_B_l39_39599

-- Define our variables and context
variables (A B C a b c : ℝ) (q : ℝ)

-- Given conditions
def triangle_angles_in_geometric_progression : Prop :=
  A = B / q ∧ C = q * B

def side_relation : Prop := 
  b^2 - a^2 = a * c

def angle_sum : Prop :=
  A + B + C = Real.pi

-- The theorem we need to prove
theorem measure_of_angle_B :
  triangle_angles_in_geometric_progression A B C q →
  side_relation a b c →
  angle_sum A B C →
  B = (2 * Real.pi) / 7 :=
begin
  intros,
  sorry
end

end measure_of_angle_B_l39_39599


namespace smallest_positive_integer_l39_39372

def is_prime_gt_60 (n : ℕ) : Prop :=
  n > 60 ∧ Prime n

def smallest_integer_condition (k : ℕ) : Prop :=
  ¬ Prime k ∧ ¬ (∃ m : ℕ, m * m = k) ∧ 
  ∀ p : ℕ, Prime p → p ∣ k → p > 60

theorem smallest_positive_integer : ∃ k : ℕ, k = 4087 ∧ smallest_integer_condition k := by
  sorry

end smallest_positive_integer_l39_39372


namespace removed_term_a11_l39_39067

def geo_mean_11 (a₁ q : ℝ) : ℝ := ((a₁ : ℕ) * (q : ℕ)) // sorry 

-- Define the arithmetic series
def arithmetic_series (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

-- First term a₁ = 2⁻⁵
noncomputable def a₁ := (2:ℝ)^(-5)

-- Geometric mean of first 11 terms is 2⁵
def geom_mean_11_terms (a₁ q : ℝ) : Prop :=
  (∀ q, (∀ q,
  (2:ℝ)^5 = Real.geom_mean (range 1 12) (λ n, arith_seq a₁ q n))

-- After removing one term, new geometric mean is 2⁴
def geom_mean_after_removal (a₁ q : ℝ) (n : ℕ) : Prop :=
  (∀ q, (2:ℝ)^4 = Real.geom_mean (range 1 12).erase n (λ k, arith_seq a₁ q k))

theorem removed_term_a11 :
  ∃ (n : ℕ), 
    geom_mean_11_terms a₁ q →
    geom_mean_after_removal a₁ q n →
    n = 11 := by sorry

end removed_term_a11_l39_39067


namespace surface_area_increases_l39_39787

-- Type declaration for our problem's specific constants
variables (a b c : ℝ) -- lengths defining the parallelepiped and pyramid base

-- Define the points and lengths for the problem
def L1_vertices : Type := (ℝ × ℝ × ℝ)
def L2_vertices : Type := (ℝ × ℝ × ℝ)

-- Define the relationship of vertices.
def point_A1 : L1_vertices := (0, 0, 0)
def point_B1 : L1_vertices := (a, 0, 0)
def point_C1 : L1_vertices := (a, b, 0)
def point_D1 : L1_vertices := (0, b, 0)
def point_A2 : L2_vertices := (0, 0, c)
def point_E : L2_vertices := (0, b/2, c) -- Midpoint of A2 D2
def point_F : L2_vertices := (a/2, 0, c) -- Midpoint of A2 B2

-- Define how the surface area changes as M moves around the perimeter of L2
noncomputable def surface_area_of_pyramid_changing_with_apex 
(M : L2_vertices) : ℝ := 
2 * b * (dist point_A1 M + dist M point_B1) 

-- The theorem to prove equivalence
theorem surface_area_increases (M F : L2_vertices) 
(hM : (M.1, M.2, M.3) = (x, y, c) ∧ x ≠ a/2 ∧ x > 0 ∧ y > 0 ) 
: surface_area_of_pyramid_changing_with_apex M > surface_area_of_pyramid_changing_with_apex F :=
sorry

end surface_area_increases_l39_39787


namespace minimum_number_of_integers_l39_39739

theorem minimum_number_of_integers :
  (∀ (s : Set ℤ), (card s ≥ 4) → 
  (∃ a : ℤ, (∀ b c : ℤ, b ∈ s → c ∈ s → 
  (abs a ≤ abs b ∧ abs a ≤ abs c → abs (b * c) ≥ abs (a ^ 2))) → False))
  ∧ (∀ (s : Set ℤ), (card s = 3) → 
  (∃ a : ℤ, (∀ b c : ℤ, b ∈ s → c ∈ s → 
  (abs a ≤ abs b ∧ abs a ≤ abs c → abs (b * c) < abs (a ^ 2))))) :=
by
  sorry

end minimum_number_of_integers_l39_39739


namespace induction_goal_l39_39546

variables (n k : ℕ)
variables (a : ℕ → ℕ)
variable [decidable_eq ℕ]

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b

-- Given Conditions
axiom decreasing_sequence (h1 : n ≥ a 1) (h2 : ∀ i, 1 ≤ i → i < k → a i > a (i + 1)) (h3 : ∀ i, 1 ≤ i → i ≤ k → a i > 0)
axiom lcm_condition (h4 : ∀ i j, 1 ≤ i → i ≤ k → 1 ≤ j → j ≤ k → lcm (a i) (a j) ≤ n)

-- Proof Goal
theorem induction_goal : (∀ i, 1 ≤ i → i ≤ k → i * a i ≤ n) :=
by
  sorry

end induction_goal_l39_39546


namespace wechat_balance_l39_39263

def transaction1 : ℤ := 48
def transaction2 : ℤ := -30
def transaction3 : ℤ := -50

theorem wechat_balance :
  transaction1 + transaction2 + transaction3 = -32 :=
by
  -- placeholder for proof
  sorry

end wechat_balance_l39_39263


namespace cone_volume_filled_88_8900_percent_l39_39026

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39026


namespace cone_volume_filled_88_8900_percent_l39_39019

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39019


namespace shortest_distance_explanation_l39_39377

theorem shortest_distance_explanation
  (A : Prop) (B : Prop) (C : Prop) (D : Prop)
  (hA : A = (∀ p1 p2, ∃! l, line(l) ∧ passes_through(p1, l) ∧ passes_through(p2, l))) 
  (hB : B = (∀ p1 p2, shortest_path(p1, p2) = straight_line(p1, p2)))
  (hC : C = (∀ p, infinite_lines_through(p)))
  (hD : D = (∀ l1 l2, line_segment(l1) ∧ line_segment(l2) → comparable_length(l1, l2)))) :
  B :=
by
  sorry

end shortest_distance_explanation_l39_39377


namespace sum_inradii_constant_l39_39798

theorem sum_inradii_constant (n : ℕ) (R : ℝ) (O : ℝ) (P : List ℝ) (T : Finset (Finset ℕ)) :
  (O ∈ P) → (T.card = n - 2) →
  (∀ t ∈ T, ∃ r : ℝ, incircle_radius (O :: t.to_list) r) →
  ∃ C : ℝ, ∀ T' : Finset (Finset ℕ), 
    T'.card = n - 2 →
    (∀ t' ∈ T', ∃ r' : ℝ, incircle_radius (O :: t'.to_list) r') →
    ∑ t in T', incircle_radius (O :: t.to_list) = C :=
by sorry

def incircle_radius (vertices : List ℝ) (r : ℝ) : Prop := sorry

end sum_inradii_constant_l39_39798


namespace answered_both_l39_39588

variables (A B : Type)
variables {test_takers : Type}

-- Defining the conditions
def pa : ℝ := 0.80  -- 80% answered first question correctly
def pb : ℝ := 0.75  -- 75% answered second question correctly
def pnone : ℝ := 0.05 -- 5% answered neither question correctly

-- Formal problem statement
theorem answered_both (test_takers: Type) : 
  (pa + pb - (1 - pnone)) = 0.60 :=
by
  sorry

end answered_both_l39_39588


namespace trigonometric_identity_l39_39829

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l39_39829


namespace equivalent_terminal_angle_l39_39690

theorem equivalent_terminal_angle :
  ∃ n : ℤ, 660 = n * 360 - 420 := 
by
  sorry

end equivalent_terminal_angle_l39_39690


namespace max_length_OB_l39_39349

-- Given conditions
variables {O A B : Type} [MetricSpace O] [MetricSpace A] [MetricSpace B]
variables (OA : dist O A) (OB : dist O B)
variables (angle : ℝ) (AB : ℝ)

-- Conditions from the problem
def angle_OAB : ℝ := O.angle A B -- the angle ∠OAB
def angle_AOB : ℝ := 45 -- the angle ∠AOB in degrees
def length_AB : ℝ := 1 -- the length of AB

-- Statement of the theorem
theorem max_length_OB (h1: angle_AOB = 45) (h2: length_AB = 1):
  ∃ OB_max : ℝ, OB_max = sqrt 2 := 
sorry

end max_length_OB_l39_39349


namespace c_neg1_sufficient_not_necessary_l39_39567

def f (x : ℝ) (c : ℝ) : ℝ :=
  if x ≥ 1 then log 2 x else x + c

theorem c_neg1_sufficient_not_necessary (c : ℝ) :
  (∀ x y : ℝ, x < y → f x (-1) ≤ f y (-1)) ∧ ¬ (∀ x y : ℝ, f x c ≤ f y c → c = -1) :=
begin
  sorry
end

end c_neg1_sufficient_not_necessary_l39_39567


namespace b_n_arithmetic_c_n_sum_l39_39707

open Real Nat

-- Definitions of the sequences
def a_n : ℕ+ → ℝ
| ⟨n, hn⟩ := 2^n

def b_n : ℕ+ → ℝ
| ⟨n, hn⟩ := 3 * log 2 (2^n) - 2

def c_n : ℕ+ → ℝ
| ⟨n, hn⟩ := (3*n - 2) * (2^n)

noncomputable def S_n (n : ℕ+) : ℝ :=
  ∑ i in (Finset.range n), c_n ⟨i + 1, Nat.succ_pos i⟩

-- Proof problem rewritten in Lean 4 statement
theorem b_n_arithmetic (n : ℕ+) : 
  ∃ (b_1 : ℝ) (d : ℝ), b_n ⟨1, zero_lt_one⟩ = 1 ∧ (∀ m : ℕ+, b_n ⟨m + 1, by exact m.2⟩ - b_n m = d) :=
sorry

theorem c_n_sum (n : ℕ+) :
  S_n n = - (3 * n + 6) * 2^(n + 1) :=
sorry

end b_n_arithmetic_c_n_sum_l39_39707


namespace beth_twice_sister_age_l39_39591

theorem beth_twice_sister_age :
  ∃ (x : ℕ), 18 + x = 2 * (5 + x) :=
by
  use 8
  sorry

end beth_twice_sister_age_l39_39591


namespace expression_has_no_meaning_from_given_choices_l39_39392

def evaluate_expression : ℤ :=
  6 * 4 * ((-1 : ℤ) ^ ((-1 : ℤ) ^ (-1 : ℤ)))

theorem expression_has_no_meaning_from_given_choices :
  evaluate_expression = -24 ∧ ∀ result ∈ {1, -1, (1 : ℤ) ∨ (-1 : ℤ)}, evaluate_expression ≠ result :=
  by 
    sorry

end expression_has_no_meaning_from_given_choices_l39_39392


namespace distance_to_taylor_l39_39603

theorem distance_to_taylor (A J T : ℝ × ℝ)
  (hA : A = (8, -15))
  (hJ : J = (-4, 22))
  (hT : T = (2, 10)) :
  let M := ((A.1 + J.1) / 2, (A.2 + J.2) / 2) in
  T.2 - M.2 = 6.5 :=
by
  sorry

end distance_to_taylor_l39_39603


namespace problem1_problem2_l39_39563

noncomputable def α_in_range (α : ℝ) : Prop := α ∈ set.Ioo (-π) 0

noncomputable def point_A := (4 : ℝ, 0 : ℝ)
noncomputable def point_B := (0 : ℝ, 4 : ℝ)
noncomputable def point_C (α : ℝ) := (3 * Real.cos α, 3 * Real.sin α)

noncomputable def vector_AC (α : ℝ) := (3 * Real.cos α - 4, 3 * Real.sin α)
noncomputable def vector_BC (α : ℝ) := (3 * Real.cos α, 3 * Real.sin α - 4)

noncomputable def norm_squared (v : ℝ × ℝ) := v.1 * v.1 + v.2 * v.2

noncomputable def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

theorem problem1 (α : ℝ) (hα : α_in_range α) (h : norm_squared (vector_AC α) = norm_squared (vector_BC α)) :
  α = -3 * π / 4 :=
sorry

theorem problem2 (α : ℝ) (hα : α_in_range α) (h : dot_product (vector_AC α) (vector_BC α) = 0) :
  (2 * (Real.sin α)^2 + 2 * Real.sin α * Real.cos α) / (1 + Real.tan α) = -7 / 16 :=
sorry

end problem1_problem2_l39_39563


namespace ratio_of_ages_is_six_l39_39450

-- Definitions of ages
def Cody_age : ℕ := 14
def Grandmother_age : ℕ := 84

-- The ratio we want to prove
def age_ratio : ℕ := Grandmother_age / Cody_age

-- The theorem stating the ratio is 6
theorem ratio_of_ages_is_six : age_ratio = 6 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_ages_is_six_l39_39450


namespace max_min_sum_eq_two_l39_39209

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + x + 1) / (x^2 + 1)

theorem max_min_sum_eq_two : 
  let M := RealSup (set.range f)
  let N := RealInf (set.range f)
  M + N = 2 :=
by
  sorry

end max_min_sum_eq_two_l39_39209


namespace slant_asymptote_sum_l39_39053

theorem slant_asymptote_sum (x : ℝ) :
  (∃ m b : ℝ, (∀ x : ℝ, y = (2*x^2 + 3*x - 7)/(x-3) ∧ x ≠ 3 → y = m*x + b + o(1/x)) ∧ (m + b = 11)) :=
sorry

end slant_asymptote_sum_l39_39053


namespace quadratic_properties_l39_39921

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39921


namespace max_yes_men_l39_39272

-- Define types of inhabitants
inductive Inhabitant
| Knight
| Liar
| YesMan

-- Main theorem stating the problem
theorem max_yes_men (total inhabitants: ℕ) (yes_answers: ℕ)
  (K L S: ℕ)
  (Hcondition: K + L + S = 2018)
  (Hyes: yes_answers = 1009)
  (Hbehaviour: ∀ x, (x = Inhabitant.Knight → is_true x) ∧ 
                     (x = Inhabitant.Liar → ¬is_true x) ∧ 
                     (x = Inhabitant.YesMan → (majority_so_far x → is_true x) ∨ (majority_so_far x → ¬is_true x))):
  S ≤ 1009 := sorry

end max_yes_men_l39_39272


namespace find_plane_equation_l39_39103

def point3D := (ℝ × ℝ × ℝ)

def plane_equation (A B C D : ℝ) (p : point3D) : Prop :=
  A * p.1 + B * p.2 + C * p.3 + D = 0

theorem find_plane_equation :
  ∃ (A B C D : ℤ), ∀ (p : point3D), 
  (p = (1, 0, 2) ∨ p = (5, 0, 4) ∨ p = (7, -2, 3)) → plane_equation A B C D (p.1, p.2, p.3) ∧
  A > 0 ∧ Int.gcd (abs A) (Int.gcd (abs B) (Int.gcd (abs C) (abs D))) = 1 :=
by
  use 1, 1, -1, 1
  sorry

end find_plane_equation_l39_39103


namespace initial_amount_spent_l39_39380

theorem initial_amount_spent
    (X : ℕ) -- initial amount of money to spend
    (sets_purchased : ℕ := 250) -- total sets purchased
    (sets_cost_20 : ℕ := 178) -- sets that cost $20 each
    (price_per_set : ℕ := 20) -- price of each set that cost $20
    (remaining_sets : ℕ := sets_purchased - sets_cost_20) -- remaining sets
    (spent_all : (X = sets_cost_20 * price_per_set + remaining_sets * 0)) -- spent all money, remaining sets assumed free to simplify as the exact price is not given or necessary
    : X = 3560 :=
    by
    sorry

end initial_amount_spent_l39_39380


namespace num_divisors_sixty_l39_39980

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l39_39980


namespace max_length_OB_l39_39350

-- Given conditions
variables {O A B : Type} [MetricSpace O] [MetricSpace A] [MetricSpace B]
variables (OA : dist O A) (OB : dist O B)
variables (angle : ℝ) (AB : ℝ)

-- Conditions from the problem
def angle_OAB : ℝ := O.angle A B -- the angle ∠OAB
def angle_AOB : ℝ := 45 -- the angle ∠AOB in degrees
def length_AB : ℝ := 1 -- the length of AB

-- Statement of the theorem
theorem max_length_OB (h1: angle_AOB = 45) (h2: length_AB = 1):
  ∃ OB_max : ℝ, OB_max = sqrt 2 := 
sorry

end max_length_OB_l39_39350


namespace watch_correction_l39_39058

theorem watch_correction :
  let daily_loss := (13 / 4 : ℚ) in
  let hourly_loss := daily_loss / 24 in
  let total_hours := (11 * 24 + 18 : ℚ) in
  let total_loss := hourly_loss * total_hours in
  total_loss ≈ 38.1875 :=
by
  use daily_loss, hourly_loss, total_hours, total_loss
  sorry

end watch_correction_l39_39058


namespace x2000_equals_4_l39_39949

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 1     := 9
| 2     := 4
| 3     := 7
| (n+4) := sequence n

lemma sum_consecutive (n : ℕ) :
  sequence n + sequence (n+1) + sequence (n+2) = 20 := sorry

theorem x2000_equals_4 : sequence 2000 = 4 :=
by
  -- use periodicity in the sequence to conclude the proof
  sorry

end x2000_equals_4_l39_39949


namespace chicago_bulls_heat_games_total_l39_39211

-- Statement of the problem in Lean 4
theorem chicago_bulls_heat_games_total :
  ∀ (bulls_games : ℕ) (heat_games : ℕ),
    bulls_games = 70 →
    heat_games = bulls_games + 5 →
    bulls_games + heat_games = 145 :=
by
  intros bulls_games heat_games h_bulls h_heat
  rw [h_bulls, h_heat]
  exact sorry

end chicago_bulls_heat_games_total_l39_39211


namespace cone_water_volume_percentage_l39_39003

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39003


namespace two_positive_numbers_inequality_three_positive_numbers_inequality_l39_39681

theorem two_positive_numbers_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
by sorry

theorem three_positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
by sorry

end two_positive_numbers_inequality_three_positive_numbers_inequality_l39_39681


namespace equal_white_black_balls_l39_39816

theorem equal_white_black_balls (b w n x : ℕ) 
(h1 : x = n - x)
: (x = b + w - n + x - w) := sorry

end equal_white_black_balls_l39_39816


namespace find_x_l39_39305

theorem find_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : 
  x = 7 :=
sorry

end find_x_l39_39305


namespace max_prime_subset_sums_l39_39361

noncomputable def is_prime(n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def subset_sums (a b c : ℕ) : List ℕ := [a, b, c, a+b, a+c, b+c, a+b+c]

def prime_subset_sums_count (a b c : ℕ) : ℕ :=
  (subset_sums a b c).countp is_prime

theorem max_prime_subset_sums (a b c : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) : 
  prime_subset_sums_count a b c ≤ 5 :=
sorry


end max_prime_subset_sums_l39_39361


namespace contrapositive_prop_l39_39311

theorem contrapositive_prop {α : Type} [Mul α] [Zero α] (a b : α) : 
  (a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0) :=
by sorry

end contrapositive_prop_l39_39311


namespace solve_eq_n_fact_plus_n_eq_n_pow_k_l39_39479

theorem solve_eq_n_fact_plus_n_eq_n_pow_k :
  ∀ (n k : ℕ), 0 < n → 0 < k → (n! + n = n^k ↔ (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3)) :=
by
  sorry

end solve_eq_n_fact_plus_n_eq_n_pow_k_l39_39479


namespace problem1_problem2_l39_39915

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k
def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 1 < m ∧ m < n → ¬ (m ∣ n)
def sum_of_digits (n : ℕ) : ℕ := (n.toString.data.map (λ c, c.toNat - '0'.toNat)).sum
def derived_prime (k : ℕ) : Prop := is_prime (sum_of_digits k)

-- Part (1)
theorem problem1 (k : ℕ) (h1 : is_composite k) (h2 : 1 < k) (h3 : k < 100) (h4 : derived_prime k) :
  sum_of_digits k = 2 → k = 20 :=
sorry

-- Part (2)
theorem problem2 : finset.card (finset.union {3, 5, 7, 2, 11, 13, 17} {12, 14, 16, 20, 21, 25, 30, 32, 34, 38, 49, 50, 52, 56, 58, 65, 70, 74, 76, 85, 92, 94, 98}) = 30 :=
sorry

end problem1_problem2_l39_39915


namespace reflection_through_orthocenter_l39_39800

noncomputable def reflection (P line : Point) : Point := sorry -- reflection definition placeholder
noncomputable def orthocenter (A B C : Point) : Point := sorry -- orthocenter definition placeholder

theorem reflection_through_orthocenter (A B C P : Point) (circumcircle : Circle A B C)
  (P_on_circumcircle : P ∈ circumcircle)
  (H : Point := orthocenter A B C)
  (P1 : Point := reflection P (Line B C))
  (P2 : Point := reflection P (Line A C)) :
  on_line (Line P1 P2) H :=
begin
  sorry
end

end reflection_through_orthocenter_l39_39800


namespace calculate_f3_l39_39914

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^7 + a * x^5 + b * x - 5

theorem calculate_f3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := 
by
  sorry

end calculate_f3_l39_39914


namespace angle_set_equality_l39_39093

open Real

theorem angle_set_equality (α : ℝ) :
  ({sin α, sin (2 * α), sin (3 * α)} = {cos α, cos (2 * α), cos (3 * α)}) ↔ 
  ∃ (k : ℤ), α = π / 8 + (k : ℝ) * (π / 2) :=
by
  sorry

end angle_set_equality_l39_39093


namespace value_of_x_l39_39200

theorem value_of_x : ∃ x : ℝ, x = 2 ∧ x = real.sqrt (2 + x) :=
by
  use 2
  split
  · rfl
  · rw [real.sqrt_eq_rfl]
    ring
  sorry

end value_of_x_l39_39200


namespace maxwell_speed_correct_l39_39262

noncomputable def maxwell_walking_speed
  (total_distance : ℕ)
  (brad_speed : ℕ)
  (maxwell_distance : ℕ)
  (meeting_distance : ℕ) :
  ℕ :=
    let brad_travelled := total_distance - maxwell_distance in
    let t := brad_travelled / brad_speed in
    maxwell_distance / t

theorem maxwell_speed_correct
  (total_distance : ℕ = 40) 
  (brad_speed : ℕ = 5) 
  (maxwell_distance : ℕ = 15) 
  : maxwell_walking_speed 40 5 15 = 3 :=
by
  simp [maxwell_walking_speed, total_distance, brad_speed, maxwell_distance]
  sorry

end maxwell_speed_correct_l39_39262


namespace magnitude_of_b_l39_39574

variable (m : ℝ)
def a : ℝ × ℝ := (-1, -2)
def b : ℝ × ℝ := (m, 2)

theorem magnitude_of_b (h : ‖a + 2 • b‖ = ‖a - 2 • b‖) : ‖b‖ = 2 * Real.sqrt 5 :=
sorry

end magnitude_of_b_l39_39574


namespace sum_of_floor_sqrt_l39_39460

theorem sum_of_floor_sqrt :
  (∑ i in Finset.range 25, (Nat.sqrt (i + 1))) = 75 := by
  sorry

end sum_of_floor_sqrt_l39_39460


namespace total_visitors_400_l39_39385

variables (V E U : ℕ)

def visitors_did_not_enjoy_understand (V : ℕ) := 3 * V / 4 + 100 = V
def visitors_enjoyed_equal_understood (E U : ℕ) := E = U
def total_visitors_satisfy_34 (V E : ℕ) := 3 * V / 4 = E

theorem total_visitors_400
  (h1 : ∀ V, visitors_did_not_enjoy_understand V)
  (h2 : ∀ E U, visitors_enjoyed_equal_understood E U)
  (h3 : ∀ V E, total_visitors_satisfy_34 V E) :
  V = 400 :=
by { sorry }

end total_visitors_400_l39_39385


namespace cone_to_sphere_volume_ratio_l39_39422

variable (r : ℝ) (π : ℝ := Real.pi)

-- Definitions of the conditions
def cone_height (r : ℝ) : ℝ := 2 * r
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * (r^3)
def volume_cone (r : ℝ) : ℝ := (1 / 3) * π * (r^2) * cone_height r

-- The theorem we want to prove
theorem cone_to_sphere_volume_ratio (h_eq : ∀ r, cone_height r = 2 * r) :
  ∀ r, (volume_cone r) / (volume_sphere r) = 1 / 2 := 
by
  sorry

end cone_to_sphere_volume_ratio_l39_39422


namespace sum_of_floor_sqrt_l39_39453

theorem sum_of_floor_sqrt :
  (∑ n in Finset.range 26, Int.floor (Real.sqrt n)) = 75 :=
by
  -- skipping proof details
  sorry

end sum_of_floor_sqrt_l39_39453


namespace min_distance_curve_C2_to_line_l39_39944

noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  (1/2 * Real.cos θ, √3 / 2 * Real.sin θ)

def distance_from_line (x y : ℝ) : ℝ :=
  |x - y - 2| / √2

theorem min_distance_curve_C2_to_line :
  ∃ θ : ℝ, distance_from_line (1/2 * Real.cos θ) (√3 / 2 * Real.sin θ) = √2 / 2 :=
sorry

end min_distance_curve_C2_to_line_l39_39944


namespace avg_age_team_proof_l39_39029

-- Defining the known constants
def members : ℕ := 15
def avg_age_team : ℕ := 28
def captain_age : ℕ := avg_age_team + 4
def remaining_players : ℕ := members - 2
def avg_age_remaining : ℕ := avg_age_team - 2

-- Stating the problem to prove the average age remains 28
theorem avg_age_team_proof (W : ℕ) :
  28 = avg_age_team ∧
  members = 15 ∧
  captain_age = avg_age_team + 4 ∧
  remaining_players = members - 2 ∧
  avg_age_remaining = avg_age_team - 2 ∧
  28 * 15 = 26 * 13 + captain_age + W :=
sorry

end avg_age_team_proof_l39_39029


namespace find_slope_l39_39232

-- Defining the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Defining the line l passing through the point (3, 0) with slope k
def line_through_P (x y k : ℝ) : Prop := y = k * (x - 3)

-- Defining the condition that line l intersects hyperbola C at points A and B
def intersects (l : ℝ → ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop :=
  ∃ x1 y1 x2 y2, l x1 y1 k ∧ l x2 y2 k ∧ c x1 y1 ∧ c x2 y2

-- Defining the condition that the sum of distances from the foci to points A and B is 16
def sum_of_distances_eq_16 (x1 x2 k : ℝ) : Prop :=
  let e : ℝ := 2 in -- eccentricity
  let a : ℝ := 1 in -- semi-major axis length
  e * (x1 + x2) - 2 * a = 16

-- Stating the main theorem to be proved
theorem find_slope (k : ℝ) (H1 : ∃ (x1 x2 : ℝ), 
  intersects line_through_P hyperbola ∧ sum_of_distances_eq_16 x1 x2 k):
  k = 3 ∨ k = -3 := sorry

end find_slope_l39_39232


namespace even_k_l39_39637

theorem even_k :
  ∀ (a b n k : ℕ),
  1 ≤ a → 1 ≤ b → 0 < n →
  2^n - 1 = a * b →
  (a * b + a - b - 1) % 2^k = 0 →
  (a * b + a - b - 1) % 2^(k+1) ≠ 0 →
  Even k :=
by
  intros a b n k ha hb hn h1 h2 h3
  sorry

end even_k_l39_39637


namespace sample_data_correlation_is_one_l39_39215

section
variables {n : ℕ} {x y : Fin n → ℝ}
hypothesis (h₀ : 2 ≤ n) -- n ≥ 2
hypothesis (h₁ : ∃ i j, i ≠ j ∧ x i ≠ x j) -- x_1, x_2, ..., x_n are not all equal
hypothesis (h₂ : ∀ i, y i = 2 * x i + 1) -- all sample points (x_i, y_i) are on the line y = 2x + 1

theorem sample_data_correlation_is_one : sample_correlation x y = 1 := 
sorry
end

end sample_data_correlation_is_one_l39_39215


namespace sum_infinite_series_l39_39839

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l39_39839


namespace leo_total_points_l39_39600

theorem leo_total_points (x y : ℕ) (h1 : x + y = 50) :
  0.4 * (x : ℝ) * 3 + 0.5 * (y : ℝ) * 2 = 0.2 * (x : ℝ) + 50 :=
by sorry

end leo_total_points_l39_39600


namespace find_ellipse_equation_and_midpoint_l39_39151

-- Definition of the ellipse and conditions
def is_ellipse_passing_through (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)

def has_eccentricity (a b e : ℝ) : Prop :=
  e = sqrt (1 - (b^2 / a^2))

def is_line_through_with_slope (P : ℝ × ℝ) (m : ℝ) : Prop :=
  ∀ x, y = m * (x - P.1)

noncomputable def ellipse_midpoint (a b : ℝ) (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

theorem find_ellipse_equation_and_midpoint :
  ∀ (a b : ℝ),
  is_ellipse_passing_through a b (0, 4) ∧ has_eccentricity a b (3/5) →
  (a = 5 ∧ b = 4 ∧ ∀ (m : ℝ) (line : ℝ → ℝ),
  is_line_through_with_slope (3, 0) (4/5) →
  ∀ (A B : ℝ × ℝ),
  ellipse_midpoint a b A B = (3 / 2, -6 / 5)) :=
by sorry

end find_ellipse_equation_and_midpoint_l39_39151


namespace exponent_division_l39_39730

-- We need to reformulate the given condition into Lean definitions
def twenty_seven_is_three_cubed : Prop := 27 = 3^3

-- Using the condition to state the problem
theorem exponent_division (h : twenty_seven_is_three_cubed) : 
  3^15 / 27^3 = 729 :=
by
  sorry

end exponent_division_l39_39730


namespace number_of_non_empty_proper_subsets_l39_39950

def non_empty_proper_subsets_count (A : Finset ℕ) : ℕ :=
  ((2 ^ A.card) - 2)

theorem number_of_non_empty_proper_subsets (A : Finset ℕ) (hA : A.card = 3) : 
  non_empty_proper_subsets_count A = 6 :=
by
  rw [non_empty_proper_subsets_count, hA]
  simp

end number_of_non_empty_proper_subsets_l39_39950


namespace steven_shipment_boxes_l39_39686

theorem steven_shipment_boxes (num_trucks : ℕ) (truck_capacity : ℕ)
  (light_box_weight heavy_box_weight : ℕ) :
  num_trucks = 3 →
  truck_capacity = 2000 →
  light_box_weight = 10 →
  heavy_box_weight = 40 →
  ∃ (total_boxes : ℕ), total_boxes = 240 :=
by 
  intros h_num_trucks h_truck_capacity h_light_box_weight h_heavy_box_weight
  use 240
  sorry

end steven_shipment_boxes_l39_39686


namespace group_size_is_eight_l39_39308

/-- Theorem: The number of people in the group is 8 if the 
average weight increases by 6 kg when a new person replaces 
one weighing 45 kg, and the weight of the new person is 93 kg. -/
theorem group_size_is_eight
    (n : ℕ)
    (H₁ : 6 * n = 48)
    (H₂ : 93 - 45 = 48) :
    n = 8 :=
by
  sorry

end group_size_is_eight_l39_39308


namespace ellipse_standard_eq_hyperbola_standard_eq_l39_39125

-- Definition for the Ellipse problem conditions
def ellipse_conditions (a b : ℝ) (c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : c = a * (Real.sqrt 3 / 2)) (h5 : 2 * b = 4) (h6 : a^2 = b^2 + c^2) : Prop :=
  ellipse (x y : ℝ) ⟨a, b⟩ := x^2 / a^2 + y^2 / b^2 = 1

-- Proof statement for the Ellipse problem
theorem ellipse_standard_eq (x y : ℝ) :
  ellipse_conditions 4 2 (2 * (Real.sqrt 3 / 2)) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) =
  (x^2 / 16 + y^2 / 4 = 1) :=
sorry

-- Definition for the Hyperbola problem conditions
def hyperbola_conditions (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 2 * c = 16) (h4 : c = 8) (h5 : a = 6) (h6 : b = Real.sqrt 7 * 2) : Prop :=
  hyperbola (x y : ℝ) ⟨a, b⟩ := y^2 / a^2 - x^2 / b^2 = 1

-- Proof statement for the Hyperbola problem
theorem hyperbola_standard_eq (x y : ℝ) :
  hyperbola_conditions 6 (2 * Real.sqrt 7) 8 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) =
  (y^2 / 36 - x^2 / 28 = 1) :=
sorry

end ellipse_standard_eq_hyperbola_standard_eq_l39_39125


namespace crossword_solution_correct_l39_39683

noncomputable def vertical_2 := "счет"
noncomputable def vertical_3 := "евро"
noncomputable def vertical_4 := "доллар"
noncomputable def vertical_5 := "вклад"
noncomputable def vertical_6 := "золото"
noncomputable def vertical_7 := "ломбард"

noncomputable def horizontal_1 := "обмен"
noncomputable def horizontal_2 := "система"
noncomputable def horizontal_3 := "ломбард"

theorem crossword_solution_correct :
  (vertical_2 = "счет") ∧
  (vertical_3 = "евро") ∧
  (vertical_4 = "доллар") ∧
  (vertical_5 = "вклад") ∧
  (vertical_6 = "золото") ∧
  (vertical_7 = "ломбард") ∧
  (horizontal_1 = "обмен") ∧
  (horizontal_2 = "система") ∧
  (horizontal_3 = "ломбард") :=
by
  sorry

end crossword_solution_correct_l39_39683


namespace max_x_of_conditions_l39_39249

theorem max_x_of_conditions (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 11) : x ≤ 2 :=
by
  -- Placeholder for the actual proof
  sorry

end max_x_of_conditions_l39_39249


namespace steven_shipment_boxes_l39_39687

theorem steven_shipment_boxes (num_trucks : ℕ) (truck_capacity : ℕ)
  (light_box_weight heavy_box_weight : ℕ) :
  num_trucks = 3 →
  truck_capacity = 2000 →
  light_box_weight = 10 →
  heavy_box_weight = 40 →
  ∃ (total_boxes : ℕ), total_boxes = 240 :=
by 
  intros h_num_trucks h_truck_capacity h_light_box_weight h_heavy_box_weight
  use 240
  sorry

end steven_shipment_boxes_l39_39687


namespace cone_volume_filled_88_8900_percent_l39_39028

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39028


namespace find_angle_B_l39_39148

noncomputable def triangle_angles (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π

noncomputable def sides_opposite (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  triangle_angles A B C ∧
  A + B + C = π

theorem find_angle_B
  (a b c A B C : ℝ)
  (h1: sides_opposite a b c A B C)
  (h2 : 2 * b * (Real.cos B) = a * (Real.cos C) + c * (Real.cos A)) :
  B = π / 3 :=
begin
  sorry,
end

end find_angle_B_l39_39148


namespace incorrect_statement₁_l39_39532
-- Import the necessary Lean library to cover geometry

-- Define types for Planes and Lines
axiom Plane : Type
axiom Line  : Type

-- Define the relevant relations: perpendicular, parallel, subset, and angle formation
axiom perp : Line → Plane → Prop
axiom par  : Line → Plane → Prop
axiom l_perp_l : Line → Line → Prop
axiom l_par_l : Line → Line → Prop
axiom angle : Line → Plane → Real

-- Define the entities: two planes α and β, and two lines m and n
variables (α β : Plane) (m n : Line)

-- The statement to be proven
theorem incorrect_statement₁ : (l_perp_l m n) → (perp m α) → (par n β) → ¬ (l_perp_l α β) :=
begin
  intros h1 h2 h3,
  sorry, -- Proof omitted
end

end incorrect_statement₁_l39_39532


namespace optimal_cylinder_dimensions_l39_39863

variable (R : ℝ)

noncomputable def optimal_cylinder_height : ℝ := (2 * R) / Real.sqrt 3
noncomputable def optimal_cylinder_radius : ℝ := R * Real.sqrt (2 / 3)

theorem optimal_cylinder_dimensions :
  ∃ (h r : ℝ), 
    (h = optimal_cylinder_height R ∧ r = optimal_cylinder_radius R) ∧
    ∀ (h' r' : ℝ), (4 * R^2 = 4 * r'^2 + h'^2) → 
      (h' = optimal_cylinder_height R ∧ r' = optimal_cylinder_radius R) → 
      (π * r' ^ 2 * h' ≤ π * r ^ 2 * h) :=
by
  -- Proof omitted
  sorry

end optimal_cylinder_dimensions_l39_39863


namespace max_length_OB_is_sqrt2_l39_39348

noncomputable def max_length_OB : ℝ :=
  let θ : ℝ := real.pi / 4 in
  let AB : ℝ := 1 in
  max (AB / real.sin θ * real.sin (real.pi / 2))

theorem max_length_OB_is_sqrt2 : max_length_OB = real.sqrt 2 := 
sorry

end max_length_OB_is_sqrt2_l39_39348


namespace find_angle_B_l39_39228

variables {A B C : ℝ} {a b c : ℝ}

def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def arithmetic_progression (a b c : ℝ) : Prop := 2 * b = a + c

def geometric_progression (x y z : ℝ) : Prop := y^2 = x * z

theorem find_angle_B 
  (h_triangle : is_triangle a b c)
  (h_ap : arithmetic_progression a b c)
  (h_gp : geometric_progression (sin A) (sin B) (sin C)) :
  B = π / 3 :=
sorry

end find_angle_B_l39_39228


namespace win_sector_area_l39_39405

noncomputable def radius : ℝ := 7
noncomputable def probability_win : ℝ := 3 / 8
noncomputable def area_circle : ℝ := π * radius^2
noncomputable def area_win_sector : ℝ := probability_win * area_circle

theorem win_sector_area :
  area_win_sector = (147 * π) / 8 :=
by
  unfold area_win_sector
  unfold probability_win
  unfold area_circle
  sorry

end win_sector_area_l39_39405


namespace triangle_angles_l39_39560

-- Define the problem and the conditions as Lean statements.
theorem triangle_angles (x y z : ℝ) 
  (h1 : y + 150 + 160 = 360)
  (h2 : z + 150 + 160 = 360)
  (h3 : x + y + z = 180) : 
  x = 80 ∧ y = 50 ∧ z = 50 := 
by 
  sorry

end triangle_angles_l39_39560


namespace composite_proposition_l39_39181

theorem composite_proposition :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬ (1 < 0) :=
by
  sorry

end composite_proposition_l39_39181


namespace sum_of_decimals_as_common_fraction_l39_39507

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l39_39507


namespace total_students_accommodated_l39_39051

def num_columns : ℕ := 4
def num_rows : ℕ := 10
def num_buses : ℕ := 6

theorem total_students_accommodated : num_columns * num_rows * num_buses = 240 := by
  sorry

end total_students_accommodated_l39_39051


namespace m_zero_sufficient_but_not_necessary_l39_39204

-- Define the sequence a_n
variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the condition for equal difference of squares sequence
def equal_diff_of_squares_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, (a (n+1))^2 - (a n)^2 = d

-- Define the sequence b_n as an arithmetic sequence with common difference m
variable (b : ℕ → ℝ)
variable (m : ℝ)

def arithmetic_sequence (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n+1) - b n = m

-- Prove "m = 0" is a sufficient but not necessary condition for {b_n} to be an equal difference of squares sequence
theorem m_zero_sufficient_but_not_necessary (a b : ℕ → ℝ) (d m : ℝ) :
  equal_diff_of_squares_sequence a d → arithmetic_sequence b m → (m = 0 → equal_diff_of_squares_sequence b d) ∧ (¬(m ≠ 0) → equal_diff_of_squares_sequence b d) :=
sorry


end m_zero_sufficient_but_not_necessary_l39_39204


namespace K_time_correct_l39_39393

variables (x : ℝ) (tK tM : ℝ)

def speed_K : ℝ := x
def speed_M : ℝ := x - 2/3
def distance : ℝ := 45

def time_K : ℝ := distance / speed_K
def time_M : ℝ := distance / speed_M

theorem K_time_correct (h : time_M - time_K = 0.75) : time_K = 45 / x := by
  unfold time_K
  unfold speed_K
  sorry

end K_time_correct_l39_39393


namespace infinitely_many_n_divisible_by_2018_l39_39283

theorem infinitely_many_n_divisible_by_2018 :
  ∃ᶠ n : ℕ in Filter.atTop, 2018 ∣ (1 + 2^n + 3^n + 4^n) :=
sorry

end infinitely_many_n_divisible_by_2018_l39_39283


namespace cone_water_volume_percentage_l39_39007

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39007


namespace probability_different_numbers_and_colors_l39_39883

def total_ways : ℕ := Nat.choose 5 2

def valid_pairs : ℕ := 4

def probability_valid_pairs : ℚ := valid_pairs / total_ways

theorem probability_different_numbers_and_colors :
  probability_valid_pairs = 2 / 5 := by
  sorry

end probability_different_numbers_and_colors_l39_39883


namespace lines_parallel_l39_39526

theorem lines_parallel (a : ℝ) : 
  let l₁ := (λ x y : ℝ, (3 + a) * x + 4 * y = 5 - 3a)
      l₂ := (λ x y : ℝ, 2 * x + (5 + a) * y = 8)
  in (∃ m₁ m₂ : ℝ, ∀ x y : ℝ, l₁ x y → l₂ x y → m₁ = m₂) ∧ a ≠ -1 → a = -7 :=
by sorry

end lines_parallel_l39_39526


namespace solve_pair_l39_39416

theorem solve_pair (x y : ℕ) (h₁ : x = 12785 ∧ y = 12768 ∨ x = 11888 ∧ y = 11893 ∨ x = 12784 ∧ y = 12770 ∨ x = 1947 ∧ y = 1945) :
  1983 = 1982 * 11888 - 1981 * 11893 :=
by {
  sorry
}

end solve_pair_l39_39416


namespace slope_of_parallel_line_l39_39371

theorem slope_of_parallel_line (a b c : ℝ) (h: 3*a + 6*b = -24) :
  ∃ m : ℝ, (a * 3 + b * 6 = c) → m = -1/2 :=
by
  sorry

end slope_of_parallel_line_l39_39371


namespace grid_covered_l39_39717

-- Define the configuration of the grid and the properties of the coins
variables (grid : ℕ → ℕ → Prop) (centers : ℕ → ℕ → Prop) 
constant (cm : ℝ := 1.0) (r : ℝ := 1.3)

-- Initial assumptions
axiom grid_structure : ∀ (x y : ℕ), cm * √(x^2 + y^2) ≤ r
axiom coin_coverage : ∀ (x y : ℕ), (centers x y) → ∀ (vx vy : ℕ), grid vx vy → (cm * √((vx - x)^2 + (vy - y)^2)) ≤ r

-- Proof statement: the grid can be covered without overlapping coins
theorem grid_covered : ∀ (vx vy : ℕ), grid vx vy → ∃ (x y : ℕ), centers x y ∧ (cm * √((vx - x)^2 + (vy - y)^2)) ≤ r :=
begin
  sorry
end

end grid_covered_l39_39717


namespace factor_polynomial_l39_39857

theorem factor_polynomial :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x ^ 2 =
  (3 * x ^ 2 + 59 * x + 231) * (3 * x ^ 2 + 53 * x + 231) := by
  sorry

end factor_polynomial_l39_39857


namespace otimes_example_l39_39849

def otimes (a b : ℤ) : ℤ := a^2 - a * b

theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end otimes_example_l39_39849


namespace doughnut_machine_completion_l39_39764

noncomputable def completion_time (start_time : ℕ) (partial_duration : ℕ) : ℕ :=
  start_time + 4 * partial_duration

theorem doughnut_machine_completion :
  let start_time := 8 * 60  -- 8:00 AM in minutes
  let partial_completion_time := 11 * 60 + 40  -- 11:40 AM in minutes
  let one_fourth_duration := partial_completion_time - start_time
  completion_time start_time one_fourth_duration = (22 * 60 + 40) := -- 10:40 PM in minutes
by
  sorry

end doughnut_machine_completion_l39_39764


namespace sum_to_common_fraction_l39_39503

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l39_39503


namespace Aiden_has_19_goats_l39_39434

/-
Adam, Andrew, Ahmed, Alice, and Aiden all raise goats.
Adam has 7 goats.
Andrew has 5 more than twice as many goats as Adam.
Ahmed has 6 fewer goats than Andrew.
Alice has the average amount of goats between Adam, Andrew, and Ahmed, rounded up to the nearest whole number.
Aiden has 1.5 times the amount of goats Alice has, rounded down to the nearest whole number.
Prove that Aiden has 19 goats.
-/

theorem Aiden_has_19_goats :
  let Adam_goats := 7 in
  let Andrew_goats := 2 * Adam_goats + 5 in
  let Ahmed_goats := Andrew_goats - 6 in
  let Alice_goats := Int.ceil ((Adam_goats + Andrew_goats + Ahmed_goats : ℚ) / 3) in
  let Aiden_goats := Int.floor (1.5 * (Alice_goats : ℚ)) in
  Aiden_goats = 19 :=
by
  sorry

end Aiden_has_19_goats_l39_39434


namespace painted_cubes_count_l39_39382

/-- A theorem to prove the number of painted small cubes in a larger cube. -/
theorem painted_cubes_count (total_cubes unpainted_cubes : ℕ) (a b : ℕ) :
  total_cubes = a * a * a →
  unpainted_cubes = (a - 2) * (a - 2) * (a - 2) →
  22 = unpainted_cubes →
  64 = total_cubes →
  ∃ m, m = total_cubes - unpainted_cubes ∧ m = 42 :=
by
  sorry

end painted_cubes_count_l39_39382


namespace trigonometric_expression_value_l39_39821

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l39_39821


namespace eliot_account_balance_l39_39676

variable (A E F : ℝ)

theorem eliot_account_balance
  (h1 : A > E)
  (h2 : F > A)
  (h3 : A - E = (1 : ℝ) / 12 * (A + E))
  (h4 : F - A = (1 : ℝ) / 8 * (F + A))
  (h5 : 1.1 * A = 1.2 * E + 21)
  (h6 : 1.05 * F = 1.1 * A + 40) :
  E = 210 := 
sorry

end eliot_account_balance_l39_39676


namespace angles_set_equality_solution_l39_39095

theorem angles_set_equality_solution (α : ℝ) :
  ({Real.sin α, Real.sin (2 * α), Real.sin (3 * α)} = {Real.cos α, Real.cos (2 * α), Real.cos (3 * α)}) ↔ 
  (∃ (k : ℤ), 0 ≤ k ∧ k ≤ 7 ∧ α = (k * Real.pi / 2) + (Real.pi / 8)) := 
by
  sorry

end angles_set_equality_solution_l39_39095


namespace quadratic_conclusions_l39_39934

variables {a b c : ℝ} (h1 : a < 0) (h2 : a - b + c = 0)

theorem quadratic_conclusions
    (h_intersect : ∃ x, a * x ^ 2 + b * x + c = 0 ∧ x = -1)
    (h_symmetry : ∀ x, x = 1 → a * (x - 1) ^ 2 + b * (x - 1) + c = a * (x + 1) ^ 2 + b * (x + 1) + c) :
    a - b + c = 0 ∧ 
    (∀ m : ℝ, a * m ^ 2 + b * m + c ≤ -4 * a) ∧ 
    (∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c + 1 = 0 ∧ a * x2 ^ 2 + b * x2 + c + 1 = 0 ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
begin
    sorry
end

end quadratic_conclusions_l39_39934


namespace quadratic_max_value_l39_39660

theorem quadratic_max_value 
  (s r1 p : ℝ) 
  (h1 : s = 0) 
  (h2 : r1 ≠ 0) 
  (h3 : r1 * (-r1) = p) 
  (h4 : 0 < p)
  (h5 : r1^2 = -p)
  : max (1 / r1^2006 + 1 / (-r1)^2006) 2 := 
sorry

end quadratic_max_value_l39_39660


namespace max_length_OB_is_sqrt2_l39_39345

noncomputable def max_length_OB : ℝ :=
  let θ : ℝ := real.pi / 4 in
  let AB : ℝ := 1 in
  max (AB / real.sin θ * real.sin (real.pi / 2))

theorem max_length_OB_is_sqrt2 : max_length_OB = real.sqrt 2 := 
sorry

end max_length_OB_is_sqrt2_l39_39345


namespace max_length_OB_l39_39353

theorem max_length_OB (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B]
  (ray1 : Set (Line O)) (ray2 : Set (Line O)) (rays_angle : ∠ O = 45) 
  (A_on_ray1 : A ∈ ray1) (B_on_ray2 : B ∈ ray2) (AB_len : dist A B = 1) : 

  ∃ (OB : ℝ), OB = √2 := 
sorry

end max_length_OB_l39_39353


namespace simplify_expression_l39_39856

theorem simplify_expression :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90 - 18) * (90 + 18)) / ((120 - 9) * (120 + 9)) = 1 := by
  sorry

end simplify_expression_l39_39856


namespace find_integers_l39_39757

theorem find_integers (f : ℝ → ℝ) (h : ∀ x, f x = x^3 + 3*x^2 - x + 1) :
  ∃ (A B : ℤ), B = A + 1 ∧ ∃ (c : ℝ), A < c ∧ c < B ∧ f c = 0 :=
begin
  use -4,
  use -3,
  split,
  { norm_num },
  { use (-7 / 2),
    split,
    { norm_num },
    split,
    { norm_num },
    { rw h,
      norm_num } }
end

end find_integers_l39_39757


namespace f_monotonicity_and_max_value_c_range_l39_39258

noncomputable def f (x : ℝ) : ℝ := (x - Real.exp 1) / Real.exp x

theorem f_monotonicity_and_max_value :
  (∀ x, (x < Real.exp 1 + 1) → (f x) < (f (Real.exp 1 + 1))) ∧
  (∀ x, (x > Real.exp 1 + 1) → (f x) < (f (Real.exp 1 + 1))) ∧
  (f (Real.exp 1 + 1) = Real.exp (- (Real.exp 1 + 1))) :=
sorry

theorem c_range (c : ℝ) : 
  (∀ x, (0 < x ∧ x < +∞) → 2 * Real.abs (Real.log x - Real.log 2) ≥ (f x + c - 1 / Real.exp 2)) ↔
  c ≤ (Real.exp 1 - 1) / Real.exp 2 :=
sorry

end f_monotonicity_and_max_value_c_range_l39_39258


namespace b_10_is_110_l39_39244

noncomputable def sequence_b (n : ℕ) : ℕ
| 0       := 0
| 1       := 2
| (n + 1) := sequence_b n + sequence_b 1 + 2 * n * 1

theorem b_10_is_110 : sequence_b 10 = 110 := by
  sorry

end b_10_is_110_l39_39244


namespace fat_per_serving_l39_39275

theorem fat_per_serving (servings : ℕ) (half_cup_fat : ℝ) (total_fat : ℝ) (fat_per_serving : ℝ) :
  servings = 4 →
  half_cup_fat = 88 / 2 →
  total_fat = half_cup_fat →
  fat_per_serving = total_fat / servings →
  fat_per_serving = 11 :=
by
  intro h1 h2 h3 h4
  rw [h2, h3, h4]
  norm_num
  sorry

end fat_per_serving_l39_39275


namespace minimum_sum_is_40_l39_39880

noncomputable def minimum_sum_four_numbers : ℕ :=
  if h : ∃ (s : Finset ℕ), s.card = 4 ∧ 1 ∈ s
    ∧ (∀ x y ∈ s, x + y ≠ x → x + y ≡ 0 [MOD 2])
    ∧ (∀ x y z ∈ s, x + y + z ≠ x + y → x + y + z ≡ 0 [MOD 3])
    ∧ (s.sum id ≡ 0 [MOD 4]) then s.sum id else 0

theorem minimum_sum_is_40 :
  minimum_sum_four_numbers = 40 :=
by
  sorry

end minimum_sum_is_40_l39_39880


namespace rita_bought_5_dresses_l39_39677

def pants_cost := 3 * 12
def jackets_cost := 4 * 30
def total_cost_pants_jackets := pants_cost + jackets_cost
def amount_spent := 400 - 139
def total_cost_dresses := amount_spent - total_cost_pants_jackets - 5
def number_of_dresses := total_cost_dresses / 20

theorem rita_bought_5_dresses : number_of_dresses = 5 :=
by sorry

end rita_bought_5_dresses_l39_39677


namespace number_of_positive_divisors_of_60_l39_39963

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l39_39963


namespace complex_conjugate_is_2_minus_i_l39_39894

theorem complex_conjugate_is_2_minus_i (a : ℝ) :
  (z : ℂ) = a + complex.i →
  z + conj z = 4 →
  conj z = 2 - complex.i :=
by
  intro h1 h2
  -- proof omitted
  sorry

end complex_conjugate_is_2_minus_i_l39_39894


namespace triangle_repetition_equilateral_l39_39792

theorem triangle_repetition_equilateral {a b c : ℝ} (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : a + c > b) 
  (h₄ : a > 0) (h₅ : b > 0) (h₆ : c > 0) : 
  (∀ n : ℕ, 
     let T := λ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y > z) ∧ (y + z > x) ∧ (x + z > y) in 
     T ((-a + b + c) / 2) ((a - b + c) / 2) ((a + b - c) / 2) → T a b c) → 
  a = b ∧ b = c :=
by
  sorry

end triangle_repetition_equilateral_l39_39792


namespace fraction_of_second_year_students_not_declared_major_l39_39662

theorem fraction_of_second_year_students_not_declared_major (T : ℕ) :
  (1 / 2 : ℝ) * (1 - (1 / 3 * (1 / 5))) = 7 / 15 :=
by
  sorry

end fraction_of_second_year_students_not_declared_major_l39_39662


namespace ratio_of_votes_l39_39433

theorem ratio_of_votes (up_votes down_votes : ℕ) (h_up : up_votes = 18) (h_down : down_votes = 4) : (up_votes / Nat.gcd up_votes down_votes) = 9 ∧ (down_votes / Nat.gcd up_votes down_votes) = 2 :=
by
  sorry

end ratio_of_votes_l39_39433


namespace cosine_value_l39_39084

-- Define the angle theta
def theta := (2017 * Real.pi) / 6

-- State the theorem to prove
theorem cosine_value : Real.cos theta = sqrt(3) / 2 :=
by
  sorry

end cosine_value_l39_39084


namespace max_length_OB_l39_39354

theorem max_length_OB (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B]
  (ray1 : Set (Line O)) (ray2 : Set (Line O)) (rays_angle : ∠ O = 45) 
  (A_on_ray1 : A ∈ ray1) (B_on_ray2 : B ∈ ray2) (AB_len : dist A B = 1) : 

  ∃ (OB : ℝ), OB = √2 := 
sorry

end max_length_OB_l39_39354


namespace four_digit_number_with_conditions_l39_39065

theorem four_digit_number_with_conditions :
  ∃ n : ℕ,
    (1000 ≤ n ∧ n < 10000) ∧ 
    (∀ d, d ∣ n → d % 2 = 0) ∧ 
    (∃ p1 p2 p3 : ℕ, 
      Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 
      {d : ℕ | d ∣ n}.card = 42 ∧ 
      {d : ℕ | d ∣ n ∧ Prime d}.card = 3 ∧ 
      {d : ℕ | d ∣ n ∧ ¬Prime d}.card = 39 ∧ 
      n = 6336 ) :=
sorry

end four_digit_number_with_conditions_l39_39065


namespace smallest_three_digit_common_multiple_of_3_and_5_l39_39736

theorem smallest_three_digit_common_multiple_of_3_and_5 : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m) :=
by 
  sorry

end smallest_three_digit_common_multiple_of_3_and_5_l39_39736


namespace ratio_is_two_l39_39691

noncomputable def ratio_of_altitude_to_base (area base : ℕ) : ℕ :=
  have h : ℕ := area / base
  h / base

theorem ratio_is_two (area base : ℕ) (h : ℕ)  (h_area : area = 288) (h_base : base = 12) (h_altitude : h = area / base) : ratio_of_altitude_to_base area base = 2 :=
  by
    sorry 

end ratio_is_two_l39_39691


namespace gcd_lcm_product_l39_39117

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end gcd_lcm_product_l39_39117


namespace problem_134a_problem_134b_problem_134c_l39_39746

theorem problem_134a (A : Finset ℕ) (hA₁ : ∀ a ∈ A, a ≤ 200) (hA₂ : A.card = 101) :
  ∃ a b ∈ A, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
sorry

theorem problem_134b : ∃ B ⊆ (Finset.range 201), B.card = 100 ∧ ∀ a b ∈ B, a ≠ b → ¬ (a ∣ b ∨ b ∣ a) :=
sorry

theorem problem_134c (C : Finset ℕ) (hC₁ : ∀ c ∈ C, c ≤ 200) (hC₂ : C.card = 100) (hC₃ : ∃ x ∈ C, x < 16) :
  ∃ a b ∈ C, a ≠ b ∧ (a ∣ b ∨ b ∣ a) :=
sorry

end problem_134a_problem_134b_problem_134c_l39_39746


namespace length_of_segment_BO_l39_39403

variables {O A B C M N K T : Type*} [EuclideanGeometry O A B C M N K T]
variables (r a : ℝ)
variables (circle_center: O)
variables (radius_condition: ∀ {X : Type*} (P : EuclideanGeometry.X), dist O P = r)
variables (touches_BA: dist M A = 0)
variables (touches_BC: dist N C = 0)
variables (parallel_condition: ∀ {X Y : Type*}, parallel (line M X) (line O Y))
variables (KT_condition: dist K T = a)
variables (angle_condition: ∀ {X Y Z W : Type*}, angle X Y Z = 1/2 * angle W A B C)

noncomputable def find_BO : ℝ :=
√(r * (a + r))

theorem length_of_segment_BO
  (h1: touches_BA)
  (h2: touches_BC)
  (h3: parallel_condition)
  (h4: KT_condition)
  (h5: angle_condition):
  dist O B = find_BO r a :=
sorry

end length_of_segment_BO_l39_39403


namespace sum_of_squares_l39_39673

-- Define the proposition as a universal statement 
theorem sum_of_squares (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by
  sorry

end sum_of_squares_l39_39673


namespace positive_divisors_60_l39_39988

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l39_39988


namespace positive_divisors_60_l39_39990

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l39_39990


namespace necessary_and_sufficient_condition_l39_39253

theorem necessary_and_sufficient_condition (t : ℝ) :
  ((t + 1) * (1 - |t|) > 0) ↔ (t < 1 ∧ t ≠ -1) :=
by
  sorry

end necessary_and_sufficient_condition_l39_39253


namespace dice_product_divisible_by_8_l39_39358

theorem dice_product_divisible_by_8 :
  (∑ k in (Finset.range (8^8)), if (∏ i in (Finset.range 8), (k / 6^i) % 6 + 1) % 8 = 0 then 1 else 0).toRat / 8^8 = 199 / 256 :=
by
  sorry

end dice_product_divisible_by_8_l39_39358


namespace boat_license_combinations_l39_39428

theorem boat_license_combinations : 
  let letter_choices := 4 in
  let first_digit_choices := 8 in
  let remaining_digit_choices := 10 in
  letter_choices * first_digit_choices * (remaining_digit_choices ^ 6) = 32000000 :=
by
  sorry

end boat_license_combinations_l39_39428


namespace sec_225_eq_neg_sqrt_2_l39_39090

theorem sec_225_eq_neg_sqrt_2 :
  (sec (225 : ℝ)) = -Real.sqrt 2 :=
by
  have h_cos_225 : (cos (225 : ℝ)) = cos (180 + 45), from rfl,
  have h_angle_subtraction_identity : (cos (180 + 45 : ℝ)) = -(cos 45), from sorry,
  have h_cos_45 : (cos (45 : ℝ)) = 1 / Real.sqrt 2, from sorry,
  have h_cos_225_value : (cos (225 : ℝ)) = - (1 / Real.sqrt 2), from 
    calc (cos (225 : ℝ)) 
          = cos (180 + 45 : ℝ) : by simp [h_cos_225]
      ... = - (cos 45 : ℝ)     : by simp [h_angle_subtraction_identity]
      ... = - (1 / Real.sqrt 2) : by simp [h_cos_45],
  have h_sec_def : (sec (225 : ℝ)) = (1 / cos (225 : ℝ)), from sorry,
  have h_sec_final : (1 / (- (1 / Real.sqrt 2))) = - Real.sqrt 2, from sorry,
  show (sec (225 : ℝ)) = - Real.sqrt 2, from 
    calc (sec (225 : ℝ))
          = 1 / (cos (225 : ℝ)) : by simp [h_sec_def]
      ... = 1 / (- (1 / Real.sqrt 2)) : by simp [h_cos_225_value]
      ... = - Real.sqrt 2 : by simp [h_sec_final]

end sec_225_eq_neg_sqrt_2_l39_39090


namespace first_player_winning_strategy_l39_39725

theorem first_player_winning_strategy (m n : ℕ) (r : ℝ) (r_eq_1 : r = 1) (h₁ : m ≥ 2) (h₂ : n ≥ 2) :
  ∃ strategy : (ℕ × ℕ) → option (ℕ × ℕ), winning_strategy m n r strategy :=
sorry

end first_player_winning_strategy_l39_39725


namespace num_true_statements_l39_39593

theorem num_true_statements :
  (if (2 : ℝ) = 2 then (2 : ℝ)^2 - 4 = 0 else false) ∧
  ((∀ (x : ℝ), x^2 - 4 = 0 → x = 2) ∨ (∃ (x : ℝ), x^2 - 4 = 0 ∧ x ≠ 2)) ∧
  ((∀ (x : ℝ), x ≠ 2 → x^2 - 4 ≠ 0) ∨ (∃ (x : ℝ), x ≠ 2 ∧ x^2 - 4 = 0)) ∧
  ((∀ (x : ℝ), x^2 - 4 ≠ 0 → x ≠ 2) ∨ (∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ x = 2)) :=
sorry

end num_true_statements_l39_39593


namespace movie_channels_cost_l39_39233

variable (M : ℝ) -- Cost of the movie channels per month
variable (basic_cost : ℝ) -- Cost of the basic cable service
variable (sports_diff : ℝ) -- Difference in cost between sports and movie channels
variable (total_cost : ℝ) -- Total monthly payment with movie and sports channels

-- Conditions
def basic_cost_def : basic_cost = 15 := by sorry
def sports_diff_def : sports_diff = 3 := by sorry
def total_cost_def : total_cost = 36 := by sorry

-- The cost of sports channels is M - sports_diff
def sports_cost := M - sports_diff

-- Equation representing the total cost of movie and sports channels added to basic cost
def total_cost_eq : M + (M - sports_diff) + basic_cost = total_cost := by sorry

-- Goal: Prove that the cost of the movie channels is $12
theorem movie_channels_cost : M = 12 := by
  -- Assume the conditions
  assume h1 : basic_cost = 15
  assume h2 : sports_diff = 3
  assume h3 : total_cost = 36
  -- Using the conditions to reach the conclusion
  sorry

end movie_channels_cost_l39_39233


namespace sum_of_decimals_is_fraction_l39_39509

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l39_39509


namespace sides_equal_max_diagonal_at_most_two_l39_39996

variable {n : ℕ}
variable (P : Polygon n)
variable (is_convex : P.IsConvex)
variable (max_diagonal : ℝ)
variable (sides_equal_max_diagonal : list ℝ)
variable (length_sides_equal_max_diagonal : sides_equal_max_diagonal.length)

-- Here we assume the basic conditions given in the problem:
-- 1. The polygon P is convex.
-- 2. The number of sides equal to the longest diagonal are stored in sides_equal_max_diagonal.

theorem sides_equal_max_diagonal_at_most_two :
  is_convex → length_sides_equal_max_diagonal ≤ 2 :=
by
  sorry

end sides_equal_max_diagonal_at_most_two_l39_39996


namespace KM_perp_LN_l39_39696

variables {A B C D E K M L N : Type}

-- Assuming we have a cyclic quadrilateral ABCD
parameters (cyclic_ABCD : cyclic quadrilateral ABCD)
-- Points of intersection of diagonals at E
(parameters (E_intersection : E = intersection (diagonal AC) (diagonal BD)))
-- Midpoints K and M
(parameters (K_mid_AB : is_midpoint K A B) (M_mid_CD : is_midpoint M C D))
-- Points L on BC and N on AD, with perpendicular conditions
(parameters (L_on_BC : on_line L B C) (N_on_AD : on_line N A D))
(parameters (EL_perp_BC : perpendicular E L B C) (EN_perp_AD : perpendicular E N A D))

-- Statement to be proved
theorem KM_perp_LN :
  is_perpendicular KM LN :=
by
  sorry

end KM_perp_LN_l39_39696


namespace george_income_l39_39135

def half (x: ℝ) : ℝ := x / 2

theorem george_income (income : ℝ) (H1 : half income - 20 = 100) : income = 240 := 
by 
  sorry

end george_income_l39_39135


namespace length_of_AB_is_two_l39_39538

-- Declare the conditions
def line_equation (x : ℝ) : ℝ := (real.sqrt 3) * x

def circle_equation (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

-- Define the points of intersection A and B
def is_intersection (x y : ℝ) : Prop := line_equation x = y ∧ circle_equation x y

-- Define the length of segment AB
def length_AB : Prop :=
  ∃ A B : ℝ × ℝ,
    is_intersection A.1 A.2 ∧
    is_intersection B.1 B.2 ∧
    norm (A.1 - B.1, A.2 - B.2) = 2

-- Proof statement
theorem length_of_AB_is_two :
  length_AB :=
sorry

end length_of_AB_is_two_l39_39538


namespace P_plus_Q_l39_39642

theorem P_plus_Q (P Q : ℝ) (h : ∀ x, x ≠ 3 → (P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3))) : P + Q = 46 :=
sorry

end P_plus_Q_l39_39642


namespace num_quadrilaterals_including_A_l39_39722

def choose : ℕ → ℕ → ℕ
| n, k := if k > n then 0 else Finset.card ((Finset.range n).powersetLen k)

noncomputable def num_convex_quadrilaterals (total_points : ℕ) (required_point : ℕ) : ℕ :=
  choose (total_points - 1) 3

theorem num_quadrilaterals_including_A (points : Finset ℕ) (A : ℕ) (hA : A ∈ points) (h_card : points.card = 12) :
  num_convex_quadrilaterals points.card A = 165 :=
by {
  rw [num_convex_quadrilaterals, h_card],
  norm_num,
  rw [choose, if_neg, Finset.card_powerset_len],
  norm_num,
  sorry
}

end num_quadrilaterals_including_A_l39_39722


namespace quadratic_properties_l39_39925

def quadratic_function (a b c : ℝ) := λ x, a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h0 : a < 0) (h1 : quadratic_function a b c (-1) = 0)
  (h2 : ∀ m : ℝ, let f := quadratic_function a b c in f(m) ≤ -4 * a)
  (h3 : b = -2 * a) (h4: c = -3 * a):
  (a - b + c = 0) ∧ 
  (∀ x1 x2 : ℝ, quadratic_function a b (c + 1) x1 = 0 → quadratic_function a b (c + 1) x2 = 0 →
    x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39925


namespace number_of_positive_divisors_l39_39992

theorem number_of_positive_divisors (n : ℕ) (h : n = 2^2 * 3^2 * 5) : ∃ d : ℕ, d = 18 ∧ 
(count_factors n = d) := sorry

end number_of_positive_divisors_l39_39992


namespace sin_alpha_cos_alpha_l39_39913

theorem sin_alpha_cos_alpha {α : ℝ} (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l39_39913


namespace team_winning_percentage_l39_39489

theorem team_winning_percentage :
  let first_games := 100
  let remaining_games := 125 - first_games
  let won_first_games := 75
  let percentage_won := 50
  let won_remaining_games := Nat.ceil ((percentage_won : ℝ) / 100 * remaining_games)
  let total_won_games := won_first_games + won_remaining_games
  let total_games := 125
  let winning_percentage := (total_won_games : ℝ) / total_games * 100
  winning_percentage = 70.4 :=
by sorry

end team_winning_percentage_l39_39489


namespace sum_of_digits_n_plus_1_l39_39644

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def satisfies_conditions (n : ℕ) : Prop :=
  sum_of_digits n = 1274 ∧ n % 9 = sum_of_digits n % 9

theorem sum_of_digits_n_plus_1 (n : ℕ) (h1 : satisfies_conditions n) :
  sum_of_digits (n + 1) = 1239 :=
sorry

end sum_of_digits_n_plus_1_l39_39644


namespace proof_of_problem_l39_39136

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, (x + 2) ^ (x + 3) = 1 ↔ (x = -1 ∨ x = -3)

theorem proof_of_problem : proof_problem :=
by
  sorry

end proof_of_problem_l39_39136


namespace angle_bisector_intersection_l39_39689

theorem angle_bisector_intersection
  (A B C A1 C1 I C2 A2 : Type)
  (triangle_ABC : Triangle A B C)
  (angle_bisectors_AA1_CC1 : AngleBisectors A A1 C C1)
  (circumcircles_AIC1_CIA1 : Circumcircles A I C1 C I A1)
  (intersections_C2_A2 : IntersectionsOnArcs A I C1 C I A1 C2 A2)
  (circumcircle_ABC : Circumcircle A B C) :
  intersects_on_circumcircle (Line A1 A2) (Line C1 C2) (circumcircle_ABC) :=
sorry

end angle_bisector_intersection_l39_39689


namespace correct_average_price_correct_white_cat_amount_correct_black_cat_amount_l39_39793

-- Defining the initial conditions
def white_cat_fish := 5
def black_cat_fish := 3
def calico_cat_fish := 0
def total_fish := white_cat_fish + black_cat_fish

def fish_shared_equally := total_fish / 3
def calico_cat_payment := 0.8

-- Correct answers from the solution
def average_price_per_fish := 2.4
def white_cat_amount := white_cat_fish * average_price_per_fish - fish_shared_equally * average_price_per_fish
def black_cat_amount := black_cat_fish * average_price_per_fish - fish_shared_equally * average_price_per_fish

-- Prove the correctness of the solution
theorem correct_average_price :
  calico_cat_payment * 3 = average_price_per_fish := sorry

theorem correct_white_cat_amount :
  (white_cat_fish - fish_shared_equally) * average_price_per_fish = white_cat_amount := sorry

theorem correct_black_cat_amount :
  (black_cat_fish - fish_shared_equally) * average_price_per_fish = black_cat_amount := sorry

end correct_average_price_correct_white_cat_amount_correct_black_cat_amount_l39_39793


namespace find_a9_l39_39900

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 2
  | n + 1 => 2 * sequence n

theorem find_a9 :
  let S_n := λ (n : ℕ), 2 * (sequence n - 1) in
  sequence 9 = 512 :=
by
  -- Proof steps would go here
  sorry

end find_a9_l39_39900


namespace xy_value_l39_39156

theorem xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 3 / x = y + 3 / y) (hxy : x ≠ y) : x * y = 3 :=
sorry

end xy_value_l39_39156


namespace probability_cover_black_region_l39_39055

noncomputable def area_black_regions : ℝ :=
  let triangle_area := 4 * (1.125 + 2.285) in
  let central_circle_area := 19.625 in
  triangle_area + central_circle_area

noncomputable def restricted_area : ℝ :=
  8 * 8

theorem probability_cover_black_region : 
  area_black_regions / restricted_area = 0.5197 := 
sorry

end probability_cover_black_region_l39_39055


namespace sum_of_floor_sqrt_l39_39457

theorem sum_of_floor_sqrt :
  (∑ i in Finset.range 25, (Nat.sqrt (i + 1))) = 75 := by
  sorry

end sum_of_floor_sqrt_l39_39457


namespace calories_burned_exercise_calories_burned_l39_39314

def total_trips : Nat := 60
def stairs_per_trip : Nat := 45
def calories_per_stair : Nat := 3

theorem calories_burned (total_trips stairs_per_trip calories_per_stair : Nat) : Nat :=
  let stairs_per_round_trip := 2 * stairs_per_trip
  let total_stairs := total_trips * stairs_per_round_trip
  total_stairs * calories_per_stair

theorem exercise_calories_burned : 
  calories_burned total_trips stairs_per_trip calories_per_stair = 16200 :=
by
  -- definition of calories_burned
  let stairs_per_round_trip := 2 * stairs_per_trip
  let total_stairs := total_trips * stairs_per_round_trip
  let total_calories := total_stairs * calories_per_stair
  -- substitution:
  have h1 : stairs_per_round_trip = 90 := by rfl
  have h2 : total_stairs = 5400 := by rw [← h1]; rfl
  have h3 : total_calories = 16200 := by rw [← h2]; rfl
  -- final result:
  show total_calories = 16200 from h3

end calories_burned_exercise_calories_burned_l39_39314


namespace cone_water_volume_percentage_l39_39004

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39004


namespace sqrt_eq_seven_iff_l39_39101

theorem sqrt_eq_seven_iff (x y : ℝ) : sqrt (10 + 3 * x - y) = 7 ↔ y = 3 * x - 39 :=
by
  sorry

end sqrt_eq_seven_iff_l39_39101


namespace salary_percentage_gain_l39_39678

theorem salary_percentage_gain (S : ℝ) :
  let S_increased := S + 0.50 * S in
  let S_final := S_increased - 0.10 * S_increased in
  ((S_final - S) / S) * 100 = 35 :=
by
  let S_increased := S + 0.50 * S
  let S_final := S_increased - 0.10 * S_increased
  sorry

end salary_percentage_gain_l39_39678


namespace widget_production_difference_l39_39663

variable (w t : ℕ)
variable (h_wt : w = 2 * t)

theorem widget_production_difference (w t : ℕ)
    (h_wt : w = 2 * t) :
  (w * t) - ((w + 5) * (t - 3)) = t + 15 :=
by 
  sorry

end widget_production_difference_l39_39663


namespace time_to_fill_pool_l39_39784

-- Definitions based on the conditions
def poolVolume : ℕ := 4000
def fillRate : ℕ := 20
def leakRate : ℕ := 2
def leakStartTime : ℕ := 20

-- Main goal statement
theorem time_to_fill_pool : 
  let initial_fill := 20 * 20 in
  let remaining_volume := poolVolume - initial_fill in
  let net_fill_rate := fillRate - leakRate in
  220 = 
    let time_to_fill_remaining := remaining_volume / net_fill_rate in
    20 + time_to_fill_remaining 
:=
sorry

end time_to_fill_pool_l39_39784


namespace distance_between_circumcenter_and_incenter_distance_between_circumcenter_and_excenter_l39_39254

variables (A B C O I I_a : Point) (R r r_a d d_a : ℝ)
variables (triangle_ABC : triangle A B C)
variables [Circumcenter O triangle_ABC]
variables [Incenter I triangle_ABC]
variables [Excenter I_a triangle_ABC A]

-- part (a)
theorem distance_between_circumcenter_and_incenter :
  d = dist O I → d^2 = R^2 - 2 * R * r := 
sorry

-- part (b)
theorem distance_between_circumcenter_and_excenter :
  d_a = dist O I_a → d_a^2 = R^2 + 2 * R * r_a := 
sorry

end distance_between_circumcenter_and_incenter_distance_between_circumcenter_and_excenter_l39_39254


namespace john_pushups_less_l39_39738

theorem john_pushups_less (zachary david john : ℕ) 
  (h1 : zachary = 19)
  (h2 : david = zachary + 39)
  (h3 : david = 58)
  (h4 : john < david) : 
  david - john = 0 :=
sorry

end john_pushups_less_l39_39738


namespace decreasing_order_l39_39890

noncomputable def m : ℝ := (0.3 : ℝ) ^ (0.2 : ℝ)
noncomputable def n : ℝ := Real.logBase (0.2 : ℝ) 3
noncomputable def p : ℝ := Real.sin 1 + Real.cos 1

theorem decreasing_order : p > m ∧ m > n := by
  sorry

end decreasing_order_l39_39890


namespace find_sets_that_tile_l39_39516

open Set

def tiles (S A : Set ℕ) : Prop :=
  ∀ x ∈ A, ∃ l : List ℕ, (∀ y ∈ l, y ∈ S ∧ y ≠ 0) ∧ l.Nodup ∧ l.sum = x

def correct_answer : Set (Set ℕ) :=
  {{1}, {1, 2}, {1, 3}, {1, 4}, {1, 7}, {1, 2, 3}, {1, 3, 5}, {1, 5, 9}, {1, 2, 3, 4}, {1, 2, 7, 8}, {1, 4, 7, 10}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 7, 8, 9}, {1, 2, 5, 6, 9, 10}, {1, 3, 5, 7, 9, 11}, {1, 2, 4, 8}}

theorem find_sets_that_tile : ∀ S, (∀ s ∈ S, ∃ s' ∈ {s}, 1 ∈ s' ∧ tiles s' (range 13)) ↔ S ∈ correct_answer :=
by
  sorry

end find_sets_that_tile_l39_39516


namespace solution_exists_for_100_100_l39_39440

def exists_positive_integers_sum_of_cubes (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a^3 + b^3 + c^3 + d^3 = x

theorem solution_exists_for_100_100 : exists_positive_integers_sum_of_cubes (100 ^ 100) :=
by
  sorry

end solution_exists_for_100_100_l39_39440


namespace smaller_angle_at_5_30_l39_39195

-- Condition definitions
def hours_on_clock := 12
def full_circle_degrees := 360
def degrees_per_hour := full_circle_degrees / hours_on_clock
def minute_hand_position := 180 -- in degrees, when the minute hand is pointing at the 6
def hour_hand_position := 5 * degrees_per_hour + 0.5 * degrees_per_hour -- Position of the hour hand at 5:30 in degrees

-- Theorem statement
theorem smaller_angle_at_5_30 : abs (minute_hand_position - hour_hand_position) = 15 :=
by
  -- using the fact that abs (180 - 165) = 15
  sorry

end smaller_angle_at_5_30_l39_39195


namespace binary_to_decimal_11011_l39_39848

-- Statement of the theorem
theorem binary_to_decimal_11011 : Nat.ofDigits 2 [1, 1, 0, 1, 1] = 27 := sorry

end binary_to_decimal_11011_l39_39848


namespace problem_part1_problem_part2_l39_39657

def a_seq (n : ℕ) : ℝ := if n = 0 then 1 else (1/2)^(n-1)

def S (n : ℕ) : ℝ := ∑ i in range n, a_seq i

theorem problem_part1 :
  ∀ n : ℕ, n > 0 → a_seq n = (1 / 2) ^ (n - 1) := 
by 
  intros n hn
  simp [a_seq]

def b_seq (n : ℕ) : ℕ → ℝ
| 0     := 0
| (n+1) := (n+1) * (a_seq (n+1))^2

def T (n : ℕ) : ℝ := ∑ i in range n, b_seq i

theorem problem_part2 :
  ∀ n : ℕ, T n < 16 / 9 :=
by
  intro n
  -- Proof to be filled in later
  sorry

end problem_part1_problem_part2_l39_39657


namespace problem_1_problem_2_problem_3_l39_39193

noncomputable def f (x : ℝ) : ℝ := (2 / 3) * x^3 - x

theorem problem_1 :
  f'(1) = 1 ∧ f(1) = -1 / 3 :=
by sorry

theorem problem_2 :
  ∃ k : ℕ, (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ k - 1993) ∧ k = 2008 :=
by sorry

theorem problem_3 (x : ℝ) (t : ℝ) (ht : 0 < t) :
  |f (Real.sin x) + f (Real.cos x)| ≤ 2 * f (t + 1 / (2 * t)) :=
by sorry

end problem_1_problem_2_problem_3_l39_39193


namespace recreation_spending_l39_39749

theorem recreation_spending : 
  ∀ (W : ℝ), 
  (last_week_spent : ℝ) -> last_week_spent = 0.20 * W →
  (this_week_wages : ℝ) -> this_week_wages = 0.80 * W →
  (this_week_spent : ℝ) -> this_week_spent = 0.40 * this_week_wages →
  this_week_spent / last_week_spent * 100 = 160 :=
by
  sorry

end recreation_spending_l39_39749


namespace quadratic_conclusions_l39_39933

variables {a b c : ℝ} (h1 : a < 0) (h2 : a - b + c = 0)

theorem quadratic_conclusions
    (h_intersect : ∃ x, a * x ^ 2 + b * x + c = 0 ∧ x = -1)
    (h_symmetry : ∀ x, x = 1 → a * (x - 1) ^ 2 + b * (x - 1) + c = a * (x + 1) ^ 2 + b * (x + 1) + c) :
    a - b + c = 0 ∧ 
    (∀ m : ℝ, a * m ^ 2 + b * m + c ≤ -4 * a) ∧ 
    (∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c + 1 = 0 ∧ a * x2 ^ 2 + b * x2 + c + 1 = 0 ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
begin
    sorry
end

end quadratic_conclusions_l39_39933


namespace diff_faces_mod_four_l39_39491

-- Define the concept of a face with labeled edges and its orientation
structure Face :=
  (edges : (char ∧ char ∧ char)) -- Labels on the edges e.g., (a, b, c)

inductive Orientation
| front | back

-- Define the convex polyhedron with triangular faces
structure Polyhedron :=
  (faces : List Face)
  (is_convex : Bool)

-- Function to determine the orientation of a face
def determine_orientation (face : Face) : Orientation :=
  sorry -- Based on the order of labels in edges

-- Prove the key theorem
theorem diff_faces_mod_four (P : Polyhedron) 
  (h : P.is_convex) 
  (h_each_face_triangle : ∀ face ∈ P.faces, true) -- Each face is a triangle
  (h_each_edge_label : ∀ face ∈ P.faces, face.edges.1 ≠ face.edges.2 ∧ face.edges.2 ≠ face.edges.3 ∧ face.edges.1 ≠ face.edges.3) -- Edges labeled with a, b, c exactly once
  (h_orientation : ∀ face ∈ P.faces, determine_orientation face = Orientation.front ∨ determine_orientation face = Orientation.back) -- Orientation criteria
  : (List.count (λ f => determine_orientation f = Orientation.front) P.faces - List.count (λ f => determine_orientation f = Orientation.back) P.faces) % 4 = 0 :=
sorry

end diff_faces_mod_four_l39_39491


namespace solve_equation_l39_39099

open Real

theorem solve_equation :
  ∀ x : ℝ, (
    (1 / ((x - 2) * (x - 3))) +
    (1 / ((x - 3) * (x - 4))) +
    (1 / ((x - 4) * (x - 5))) = (1 / 12)
  ) ↔ (x = 5 + sqrt 19 ∨ x = 5 - sqrt 19) := 
by 
  sorry

end solve_equation_l39_39099


namespace hexagon_area_is_88_l39_39076

-- Define variables for the sides of the hexagon ABCDEF
variables (AB BC CD DE EF FA : ℝ)
axiom AB_val : AB = 8
axiom BC_val : BC = 10
axiom CD_val : CD = 7
axiom DE_val : DE = 8
axiom EF_val : EF = 5
axiom FA_val : FA = 6

-- Intersection point G and triangle ABG being isosceles with AB = BG
variable G : Type
axiom G_intersection : ∃ G, G ∈ (line_segment (8:ℝ) (6:ℝ)) ∩ (line_segment (7:ℝ) (8:ℝ))
axiom triangle_isosceles : ∃ A B G, AB = 8 ∧ BG = 8 ∧ is_isosceles_triangle ABC G

-- Rectangle CDEF
axiom CDEF_rectangle : is_rectangle CD DE EF FA

-- Define the area of the hexagon ABCDEF
def area_hexagon_ABCDEF : ℝ :=
  let area_CDEF := CD * DE
  let area_triangl_ABG := 1/2 * AB * AB * 1 -- assuming theta = 90 degrees
  in area_CDEF + area_triangl_ABG

-- Prove the area of hexagon ABCDEF is 88 square units
theorem hexagon_area_is_88 : area_hexagon_ABCDEF AB BC CD DE EF FA = 88 :=
by
  simp
  -- From conditions: CD * DE = 7 * 8 = 56
  have area_CDEF: CD * DE = 56 := by rewrite [CD_val, DE_val]
  
  -- For triangle ABG (isosceles triangle with AB = BG and assuming theta = 90 degrees)
  have area_ABG : 1/2 * 8 * 8 * 1 = 32 := by norm_num

  -- Add areas of CDEF and ABG
  have final_area: 56 + 32 = 88 := by norm_num
  exact final_area

end hexagon_area_is_88_l39_39076


namespace initial_bananas_l39_39268

theorem initial_bananas (x B : ℕ) (h1 : 840 * x = B) (h2 : 420 * (x + 2) = B) : x = 2 :=
by
  sorry

end initial_bananas_l39_39268


namespace final_answer_l39_39437

def isKnight (i: ℕ) : Prop := sorry
def isLiar (i: ℕ) : Prop := sorry

axiom five_islanders : ∀ i : ℕ, i < 5 → isKnight i ∨ isLiar i 

axiom first_islander_answer : isLiar 0

axiom second_and_third_islander_answer : isLiar 1 ∧ isLiar 2

theorem final_answer :
  isKnight 3 ∧ isKnight 4 :=
begin
  sorry
end

end final_answer_l39_39437


namespace sum_sequence_11000_l39_39485

def sequence_value (k : ℕ) : ℤ :=
  if ∃ n, k = n^2 then 
    (-1)^n * k 
  else
    (if k % 2 = 1 then 1 else -1) * k

def sum_to_n (n : ℕ) : ℤ :=
  ∑ k in finset.range n, sequence_value (k + 1)

theorem sum_sequence_11000 :
  sum_to_n 12100 = 1100000 := by
  sorry

end sum_sequence_11000_l39_39485


namespace max_length_OB_l39_39355

theorem max_length_OB (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B]
  (ray1 : Set (Line O)) (ray2 : Set (Line O)) (rays_angle : ∠ O = 45) 
  (A_on_ray1 : A ∈ ray1) (B_on_ray2 : B ∈ ray2) (AB_len : dist A B = 1) : 

  ∃ (OB : ℝ), OB = √2 := 
sorry

end max_length_OB_l39_39355


namespace scytale_decode_l39_39611

theorem scytale_decode (α : ℝ) (d : ℝ) (h : ℝ) : 
  ∀ (n : ℕ), n = ⌈d / (h * cos α)⌉ :=
sorry

end scytale_decode_l39_39611


namespace winner_exceeds_second_opponent_l39_39804

theorem winner_exceeds_second_opponent
  (total_votes : ℕ)
  (votes_winner : ℕ)
  (votes_second : ℕ)
  (votes_third : ℕ)
  (votes_fourth : ℕ) 
  (h_votes_sum : total_votes = votes_winner + votes_second + votes_third + votes_fourth)
  (h_total_votes : total_votes = 963) 
  (h_winner_votes : votes_winner = 195) 
  (h_second_votes : votes_second = 142) 
  (h_third_votes : votes_third = 116) 
  (h_fourth_votes : votes_fourth = 90) :
  votes_winner - votes_second = 53 := by
  sorry

end winner_exceeds_second_opponent_l39_39804


namespace area_of_triangle_PQR_l39_39623

/-- In triangle PQR with PQ = QR, PS is an altitude,
such that point T is on the extension of QR with PT = 8.
The values of tan(angle QPT), tan(angle PST), and tan(angle PSQ) 
form a geometric progression, and the values of cot(angle PST), 
cot(angle QPT), and cot(angle PSQ) form an arithmetic progression.
Prove that the area of triangle PQR is 32 / 3. -/
theorem area_of_triangle_PQR :
  ∃ P Q R S T : Type,
  (PQ : ℝ) = (QR : ℝ) ∧
  (PS : ℝ) > 0 ∧ 
  (PT : ℝ) = 8 ∧
  (tan (∠ QPT) * tan (∠ PST) * tan (∠ PSQ) = tan^2(∠PST)) ∧
  (cot (∠ PST) + cot (∠ QPT) + cot (∠ PSQ) = 
   (cot (∠ PSQ) + 2 * cot(∠ PST)) / 3) ⟹
  area (triangle P Q R) = 32 / 3 :=
sorry

end area_of_triangle_PQR_l39_39623


namespace milk_leftover_l39_39477

def total_milk := 16
def kids_percentage := 0.75
def cooking_percentage := 0.50

theorem milk_leftover : 
  let consumed_by_kids := kids_percentage * total_milk in
  let remaining_after_kids := total_milk - consumed_by_kids in
  let used_for_cooking := cooking_percentage * remaining_after_kids in
  let milk_left := remaining_after_kids - used_for_cooking in
  milk_left = 2 := 
by
  sorry

end milk_leftover_l39_39477


namespace rectangle_length_width_l39_39786

theorem rectangle_length_width (x y : ℝ) 
  (h1 : 2 * x + 2 * y = 16) 
  (h2 : x - y = 1) : 
  x = 4.5 ∧ y = 3.5 :=
by {
  sorry
}

end rectangle_length_width_l39_39786


namespace num_digits_of_x_l39_39586

theorem num_digits_of_x (x : ℝ) (h : log 2 (log 2 (log 2 (log 2 x))) = 2) : 
  ⌊65536 * log 10 2⌋ + 1 = 19732 :=
by
  have h1 : log 2 (log 2 (log 2 x)) = 2^2, from sorry,
  have h2 : log 2 (log 2 x) = 2^4, from sorry,
  have h3 : log 2 x = 2^16, from sorry,
  have h4 : x = 2^(2^16), from sorry,
  have h_digits : log 10 (2^(2^16)) = 2^16 * log 10 2, from sorry,
  have h_digits_floor : ⌊65536 * log 10 2⌋ = 19731, from sorry,
  exact h_digits_floor + 1 = 19732

end num_digits_of_x_l39_39586


namespace probability_of_at_least_two_but_fewer_than_five_heads_l39_39032

open Finset

def favorable_outcomes : ℕ :=
  choose 8 2 + choose 8 3 + choose 8 4

def total_outcomes : ℕ :=
  2 ^ 8

theorem probability_of_at_least_two_but_fewer_than_five_heads :
  let p := favorable_outcomes / total_outcomes in
  p = (77 : ℚ) / 128 :=
by
  sorry

end probability_of_at_least_two_but_fewer_than_five_heads_l39_39032


namespace charlie_dana_coinciding_rest_days_1500_l39_39449

noncomputable def charlie_rest_days (d : ℕ) : Prop :=
  (d % 6 = 5) ∨ (d % 6 = 0)

noncomputable def dana_rest_days (d : ℕ) : Prop :=
  d % 7 = 6

noncomputable def coinciding_rest_days (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ d, charlie_rest_days d ∧ dana_rest_days d).card

theorem charlie_dana_coinciding_rest_days_1500 :
  coinciding_rest_days 1500 = 70 :=
by
  sorry

end charlie_dana_coinciding_rest_days_1500_l39_39449


namespace sum_of_squares_eq_expansion_l39_39674

theorem sum_of_squares_eq_expansion (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 :=
sorry

end sum_of_squares_eq_expansion_l39_39674


namespace abc_sum_is_17_l39_39794

noncomputable def A := 3
noncomputable def B := 5
noncomputable def C := 9

theorem abc_sum_is_17 (A B C : ℕ) (h1 : 100 * A + 10 * B + C = 359) (h2 : 4 * (100 * A + 10 * B + C) = 1436)
  (h3 : A ≠ B) (h4 : B ≠ C) (h5 : A ≠ C) : A + B + C = 17 :=
by
  sorry

end abc_sum_is_17_l39_39794


namespace max_length_OB_l39_39356

theorem max_length_OB (O A B : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B]
  (ray1 : Set (Line O)) (ray2 : Set (Line O)) (rays_angle : ∠ O = 45) 
  (A_on_ray1 : A ∈ ray1) (B_on_ray2 : B ∈ ray2) (AB_len : dist A B = 1) : 

  ∃ (OB : ℝ), OB = √2 := 
sorry

end max_length_OB_l39_39356


namespace floor_sqrt_sum_l39_39469

open Real

theorem floor_sqrt_sum : (∑ k in Finset.range 25, ⌊sqrt (k + 1)⌋) = 75 := 
by
  sorry

end floor_sqrt_sum_l39_39469


namespace initial_coloring_books_eq_l39_39056

variables (initial_remaining : ℕ) (sold remaining_on_shelves : ℕ) (shelves books_per_shelf : ℕ)

-- Define the given constants
-- The number of books the store got rid of
def sold := 33
-- The number of shelves and books per shelf
def shelves := 9
def books_per_shelf := 6
-- Calculate the number of remaining books on shelves
def remaining_on_shelves := shelves * books_per_shelf

-- Theorem: Calculate the initial number of coloring books
theorem initial_coloring_books_eq : 
  initial_remaining = remaining_on_shelves + sold → initial_remaining = 87 :=
by
  sorry

end initial_coloring_books_eq_l39_39056


namespace inequality_one_inequality_two_l39_39285

theorem inequality_one (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + a * c + b * c :=
sorry

theorem inequality_two : sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 :=
sorry

end inequality_one_inequality_two_l39_39285


namespace M_even_comp_M_composite_comp_M_prime_not_div_l39_39762

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_composite (n : ℕ) : Prop :=  ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n
def M (n : ℕ) : ℕ := 2^n - 1

theorem M_even_comp (n : ℕ) (h1 : n ≠ 2) (h2 : is_even n) : is_composite (M n) :=
sorry

theorem M_composite_comp (n : ℕ) (h : is_composite n) : is_composite (M n) :=
sorry

theorem M_prime_not_div (p : ℕ) (h : Nat.Prime p) : ¬ (p ∣ M p) :=
sorry

end M_even_comp_M_composite_comp_M_prime_not_div_l39_39762


namespace half_angle_in_third_quadrant_l39_39553

theorem half_angle_in_third_quadrant 
  (α : ℝ) 
  (h1 : 180 < α ∧ α < 270) 
  (h2 : |cos (α / 2)| = cos (α / 2)) : 
  270 < α / 2 ∧ α / 2 < 360 :=
by 
-- The proof goes here
sorry

end half_angle_in_third_quadrant_l39_39553


namespace maximize_inscribed_polygons_l39_39326

theorem maximize_inscribed_polygons : 
  ∃ (n : ℕ) (m : ℕ → ℕ), 
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → m i < m j) ∧ 
    (∑ i in Finset.range n, m i = 1996) ∧ 
    (n = 61) ∧ 
    (∀ k, 0 ≤ k ∧ k < n → m k = k + 2) :=
by
  sorry

end maximize_inscribed_polygons_l39_39326


namespace number_of_common_points_l39_39482

-- Define the first curve as a predicate
def curve1 (x y : ℝ) : Prop := 9 * x ^ 2 + y ^ 2 = 9

-- Define the second curve as a predicate
def curve2 (x y : ℝ) : Prop := x ^ 2 + 9 * y ^ 2 = 1

-- Define the set of points that lie on both curves
def common_points : set (ℝ × ℝ) := {p | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

-- State the theorem
theorem number_of_common_points : set.finite common_points ∧ set.card common_points = 2 := 
by {
  -- Proof body to be filled
  sorry
}

end number_of_common_points_l39_39482


namespace line_equation_through_point_perpendicular_l39_39862

def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

def line_through_point_slope (x1 y1 m : ℝ) : (ℝ → ℝ) :=
  λ x, m * (x - x1) + y1

theorem line_equation_through_point_perpendicular
  (x1 y1 : ℝ) (a b c : ℝ)
  (h_perpendicular : is_perpendicular (-a/b) 1)
  (h_point : (x1, y1) = (2, -1))
  : ∃ (A B C : ℝ), (A, B, C) = (1, -1, -3) :=
by
  exists 1, -1, -3
  sorry

end line_equation_through_point_perpendicular_l39_39862


namespace bowling_average_decrease_l39_39040

/-- Represents data about the bowler's performance. -/
structure BowlerPerformance :=
(old_average : ℚ)
(last_match_runs : ℚ)
(last_match_wickets : ℕ)
(previous_wickets : ℕ)

/-- Calculates the new total runs given. -/
def new_total_runs (perf : BowlerPerformance) : ℚ :=
  perf.old_average * ↑perf.previous_wickets + perf.last_match_runs

/-- Calculates the new total number of wickets. -/
def new_total_wickets (perf : BowlerPerformance) : ℕ :=
  perf.previous_wickets + perf.last_match_wickets

/-- Calculates the new bowling average. -/
def new_average (perf : BowlerPerformance) : ℚ :=
  new_total_runs perf / ↑(new_total_wickets perf)

/-- Calculates the decrease in the bowling average. -/
def decrease_in_average (perf : BowlerPerformance) : ℚ :=
  perf.old_average - new_average perf

/-- The proof statement to be verified. -/
theorem bowling_average_decrease :
  ∀ (perf : BowlerPerformance),
    perf.old_average = 12.4 →
    perf.last_match_runs = 26 →
    perf.last_match_wickets = 6 →
    perf.previous_wickets = 115 →
    decrease_in_average perf = 0.4 :=
by
  intros
  sorry

end bowling_average_decrease_l39_39040


namespace sweater_markup_l39_39754

-- Conditions
variables (W R : ℝ)
axiom h1 : 0.40 * R = 1.20 * W

-- Theorem statement
theorem sweater_markup (W R : ℝ) (h1 : 0.40 * R = 1.20 * W) : (R - W) / W * 100 = 200 :=
sorry

end sweater_markup_l39_39754


namespace problem_b_amount_l39_39742

theorem problem_b_amount (a b : ℝ) (h1 : a + b = 1210) (h2 : (4/5) * a = (2/3) * b) : b = 453.75 :=
sorry

end problem_b_amount_l39_39742


namespace sum_infinite_series_l39_39840

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l39_39840


namespace quadratic_properties_l39_39922

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39922


namespace non_acute_triangle_exists_l39_39906

theorem non_acute_triangle_exists (A B C D : Point) (h_non_collinear : ¬Collinear A B C D) :
  ∃ T : Triangle, ¬Acute T ∧ (T.vertex1 = A ∨ T.vertex1 = B ∨ T.vertex1 = C ∨ T.vertex1 = D) ∧
                     (T.vertex2 = A ∨ T.vertex2 = B ∨ T.vertex2 = C ∨ T.vertex2 = D) ∧
                     (T.vertex3 = A ∨ T.vertex3 = B ∨ T.vertex3 = C ∨ T.vertex3 = D) :=
by
  sorry

end non_acute_triangle_exists_l39_39906


namespace distance_between_parallel_lines_l39_39342

theorem distance_between_parallel_lines 
  (x y : ℝ) 
  (line1 : 3 * x + 4 * y - 12 = 0)
  (line2 : 6 * x + 8 * y + 11 = 0) :
  let a := 6 in 
  let b := 8 in
  let c1 := -24 in
  let c2 := 11 in
  abs (c1 - c2) / sqrt (a^2 + b^2) = 7 / 2 :=
by 
  sorry

end distance_between_parallel_lines_l39_39342


namespace trigonometric_identity_l39_39831

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l39_39831


namespace max_yes_men_l39_39271

-- Define types of inhabitants
inductive Inhabitant
| Knight
| Liar
| YesMan

-- Main theorem stating the problem
theorem max_yes_men (total inhabitants: ℕ) (yes_answers: ℕ)
  (K L S: ℕ)
  (Hcondition: K + L + S = 2018)
  (Hyes: yes_answers = 1009)
  (Hbehaviour: ∀ x, (x = Inhabitant.Knight → is_true x) ∧ 
                     (x = Inhabitant.Liar → ¬is_true x) ∧ 
                     (x = Inhabitant.YesMan → (majority_so_far x → is_true x) ∨ (majority_so_far x → ¬is_true x))):
  S ≤ 1009 := sorry

end max_yes_men_l39_39271


namespace probability_subset_l39_39281

variable {Ω : Type*} [MeasurableSpace Ω]
variable {P : MeasureTheory.ProbabilityMeasure Ω}
variable {A B : Set Ω}

theorem probability_subset (h : A ⊆ B) :
  P.measure_of A ≤ P.measure_of B :=
by sorry

end probability_subset_l39_39281


namespace largest_prime_divisor_of_factorial_sum_l39_39108

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n : ℕ), 13 ≤ n → (n ∣ 13! + 14!) → is_prime n → n = 13 :=
by sorry

end largest_prime_divisor_of_factorial_sum_l39_39108


namespace smaller_angle_at_910_l39_39442

theorem smaller_angle_at_910
  (minute_hand_angle : ℕ := 60)
  (hour_hand_angle : ℕ := 275)
  (full_circle : ℕ := 360) :
  |hour_hand_angle - minute_hand_angle| = 215 ∧
  (full_circle - |hour_hand_angle - minute_hand_angle| = 145) :=
by 
  sorry

end smaller_angle_at_910_l39_39442


namespace sum_every_second_term_3015_l39_39077

theorem sum_every_second_term_3015 (seq : ℕ → ℝ)
  (h_length : ∀ n, n ≥ 1 → n ≤ 3015 → seq n + 1 = seq (n+1))
  (h_sum : (∑ n in Finset.range 3015, seq n.succ) = 8010) :
  (∑ k in Finset.range 0 1508, seq (2*k + 1)) = 3251.5 :=
sorry

end sum_every_second_term_3015_l39_39077


namespace bob_has_no_winning_strategy_l39_39436

def canWinIfAliceStartsAndPlaysOptimally (n : Nat) (k : Nat) : Prop :=
  (n % 4 = 0) ∨ (n % 4 = 1) ∨ (n % 4 = 2) ∨ (n % 4 = 3) 
  
theorem bob_has_no_winning_strategy (matches : Nat := 101) (max_take : Nat := 3) : 
  ¬ ∃ (bob_strategy : ∀ n, n % 4 ≠ 0 → ∃ k, 1 ≤ k ∧ k ≤ max_take ∧ canWinIfAliceStartsAndPlaysOptimally(n - k) max_take), 
  canWinIfAliceStartsAndPlaysOptimally matches max_take :=
by 
  sorry

end bob_has_no_winning_strategy_l39_39436


namespace money_collected_l39_39659

theorem money_collected
  (households_per_day : ℕ)
  (days : ℕ)
  (half_give_money : ℕ → ℕ)
  (total_money_collected : ℕ)
  (households_give_money : ℕ) :
  households_per_day = 20 →  
  days = 5 →
  total_money_collected = 2000 →
  half_give_money (households_per_day * days) = (households_per_day * days) / 2 →
  households_give_money = (households_per_day * days) / 2 →
  total_money_collected / households_give_money = 40
:= sorry

end money_collected_l39_39659


namespace sum_odd_sequence_l39_39559

theorem sum_odd_sequence : 
  (∑ i in finset.range 51, (2 * i + 1)) = 2601 :=
sorry

end sum_odd_sequence_l39_39559


namespace number_of_valid_permutations_l39_39952

-- Define the set S and the permutations A
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2022}
def A : ℕ → ℕ := sorry -- Since A is a permutation of S

-- Given condition
def condition (a : ℕ → ℕ) (n m : ℕ) : Prop :=
  (n ∈ S ∧ m ∈ S ∧ gcd n m ∣ (a n + a m)) 

-- Prove that there are exactly 2 permutations of A meeting the condition
theorem number_of_valid_permutations : 
  ∃ (valid_permutations : Finset (ℕ → ℕ)), 
  (∀ a ∈ valid_permutations, ∀ n m, condition a n m) ∧ valid_permutations.card = 2 := 
sorry

end number_of_valid_permutations_l39_39952


namespace area_to_be_painted_l39_39608

def wall_height : ℕ := 8
def wall_length : ℕ := 15
def glass_painting_height : ℕ := 3
def glass_painting_length : ℕ := 5

theorem area_to_be_painted :
  (wall_height * wall_length) - (glass_painting_height * glass_painting_length) = 105 := by
  sorry

end area_to_be_painted_l39_39608


namespace probability_correct_l39_39774

noncomputable def probability_all_even_before_odd (die_sides : ℕ) (evens : finset ℕ) (odds : finset ℕ) : ℚ :=
  if (die_sides = 10) ∧ (evens = {2,4,6,8,10}) ∧ (odds = {1,3,5,7,9}) then
    1 / 252
  else
    0

theorem probability_correct:
  probability_all_even_before_odd 10 ({2,4,6,8,10} : finset ℕ) ({1,3,5,7,9} : finset ℕ) = 1 / 252 :=
by { sorry }

end probability_correct_l39_39774


namespace sin_smallest_angle_approx_l39_39487

theorem sin_smallest_angle_approx (α β γ : ℝ) (a b c : ℝ) (h_angle_ratio : α / β = 1 / 2 ∧ β / γ = 2 / 4) (h_side_ratio : a / c = 4 / 9) :
  sin α ≈ 0.4330 := 
sorry

end sin_smallest_angle_approx_l39_39487


namespace relationship_a_b_c_l39_39889

noncomputable def e : ℝ := Real.exp 1

def a (x : ℝ) : ℝ := (1 / e)^x
def b (x : ℝ) : ℝ := x ^ 2
def c (x : ℝ) : ℝ := Real.log x

theorem relationship_a_b_c : a e < c e ∧ c e < b e := by
  sorry

end relationship_a_b_c_l39_39889


namespace total_area_of_frequency_histogram_l39_39325

theorem total_area_of_frequency_histogram (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ f x ∧ f x ≤ 1) (integral_f_one : ∫ x, f x = 1) :
  ∫ x, f x = 1 := 
sorry

end total_area_of_frequency_histogram_l39_39325


namespace sum_to_common_fraction_l39_39500

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l39_39500


namespace time_for_Robert_to_traverse_two_mile_stretch_l39_39057

noncomputable def time_to_traverse (highway_length_miles : ℝ) (highway_width_feet : ℝ) (speed_mph : ℝ) : ℝ :=
  let highway_length_feet := highway_length_miles * 5280
  let radius_feet := highway_width_feet / 2
  let num_quarter_circles := (4 * highway_length_feet) / highway_width_feet
  let distance_per_quarter_circle := (radius_feet * 2 * π) / 4
  let total_distance_feet := num_quarter_circles * distance_per_quarter_circle
  let total_distance_miles := total_distance_feet / 5280
  let time_hours := total_distance_miles / speed_mph
  in time_hours

theorem time_for_Robert_to_traverse_two_mile_stretch :
  time_to_traverse 2 50 8 = 2.49 * π :=
by
  sorry

end time_for_Robert_to_traverse_two_mile_stretch_l39_39057


namespace num_divisors_sixty_l39_39985

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l39_39985


namespace intersecting_lines_l39_39316

theorem intersecting_lines (a b : ℝ) :
  (3 : ℝ) = (16 / 5) * (-2 : ℝ) + a ∧
  (-2 : ℝ) = (8 / 15) * (3 : ℝ) + b → 
  a + b = 5.8 := 
by
  intros h
  cases h with h₁ h₂
  sorry

end intersecting_lines_l39_39316


namespace cone_volume_filled_88_8900_percent_l39_39020

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39020


namespace evaluate_expression_l39_39494

theorem evaluate_expression :
  2 ^ (Real.log 2 (1 / 4)) - (8 / 27) ^ (-2 / 3) + Real.log10 (1 / 100) +
  (Real.sqrt 2 - 1) ^ (Real.log10 1) = -3 :=
by
  sorry

end evaluate_expression_l39_39494


namespace problem_statement_l39_39189

noncomputable def a_n (n : ℕ) : ℝ := 2 ^ n
noncomputable def b_n (n : ℕ) : ℝ := n * (n + 1)
noncomputable def c_n (n : ℕ) : ℝ := (1 / a_n n) - (1 / b_n n)
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), c_n i

theorem problem_statement (n : ℕ) (k : ℕ) : 
  (∀ k, k = 4 ↔ (∀ m, S_n k ≥ S_n m)) ∧ (a_n n = 2 ^ n) ∧ (b_n n = n * (n + 1)) ∧ (S_n n = (1 / (n + 1)) - (1 / (2 ^ n))) :=
by sorry

end problem_statement_l39_39189


namespace part1_part2_l39_39150

variable {α : Type*} [LinearOrderedField α]

-- Function definitions for the sequences
def an (n : ℕ) : α := 3 * (n : α) - 4
def Sn (n : ℕ) : α := n * (an 1 + an n) / 2
def bn (n : ℕ) : α := -1 * (-2)^(n - 1)
def sum_b {n : ℕ} : α := (if -2 = 1 then n else bn 1 * (1 - (-2)^n) / (1 - -2)).abs

-- Conditions
axiom a3_eq_5 : an 3 = 5
axiom S4_eq_14 : Sn 4 = 14
axiom b1_eq_a1 : bn 1 = an 1
axiom b4_eq_a4 : bn 4 = an 4

-- Theorem statements
theorem part1 : ∀ n : ℕ, an n = 3 * (n : α) - 4 := 
by
  intro n
  sorry

theorem part2 : sum_b 6 = 21 :=
by
  sorry

end part1_part2_l39_39150


namespace find_angle_A_l39_39229

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hB : B = Real.pi / 3) : 
  A = Real.pi / 4 := 
sorry

end find_angle_A_l39_39229


namespace line_AB_passes_through_fixed_point_ab_passes_through_fixed_point_l39_39904

-- Given definitions
def a : ℝ := 8^.sqrt  -- a = sqrt(8)
def b : ℝ := 2        -- b = 2
def c : ℝ := 2        -- c = 2

def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / (a^2)) + (y^2 / (b^2)) = 1

def F : ℝ × ℝ := (2, 0)
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (0, b)
def isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
  let AC := (A.1 - C.1)^2 + (A.2 - C.2)^2 in
  let BC := (B.1 - C.1)^2 + (B.2 - C.2)^2 in
  AB = AC ∧ 2 * AB = BC ∧ BC ≠ 0

def fixed_point : ℝ × ℝ := (-1/2, -2)

-- Main statement to prove
theorem line_AB_passes_through_fixed_point (k1 k2 : ℝ) (h1 : k1 + k2 = 8) :
  (∃ x1 y1 x2 y2, (x1 ≠ x2 ∨ y1 ≠ y2) ∧ ellipse_eq x1 y1 ∧ ellipse_eq x2 y2 
   ∧ (y1 - M.2) / (x1 - M.1) = k1 
   ∧ (y2 - M.2) / (x2 - M.1) = k2 
   ∧ ((∃ y m, y = k1 * fixed_point.1 + m ∧ y = k2 * fixed_point.1 + m)
   ∧ ((fixed_point.2 - m1) / (fixed_point.1 - x1) = k1) ∨ ((fixed_point.2 - m2) / (fixed_point.1 - x2) = k2)))

noncomputable def line_AB (x : ℝ) : ℝ := sorry

def passes_through_fixed_point : Prop :=
  line_AB fixed_point.1 = fixed_point.2

theorem ab_passes_through_fixed_point (hx : ellipse_eq 0 b) (hf : F = (2, 0)) (ho : O = (0, 0)) :
  passes_through_fixed_point :=
sorry

end line_AB_passes_through_fixed_point_ab_passes_through_fixed_point_l39_39904


namespace find_f3_plus_f4_l39_39556

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom symmetric_about_1 : ∀ x : ℝ, f (2 - x) = f (x + 2) = f (2 - x)
axiom f_at_1 : f 1 = 2

theorem find_f3_plus_f4 : f 3 + f 4 = -2 :=
by
  sorry

end find_f3_plus_f4_l39_39556


namespace area_of_gray_region_l39_39363

theorem area_of_gray_region :
  (radius_smaller = (2 : ℝ) / 2) →
  (radius_larger = 4 * radius_smaller) →
  (gray_area = π * radius_larger ^ 2 - π * radius_smaller ^ 2) →
  gray_area = 15 * π :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  sorry

end area_of_gray_region_l39_39363


namespace sqrt_x_plus_y_eq_two_l39_39887

theorem sqrt_x_plus_y_eq_two (x y : ℝ) (h : sqrt (3 - x) + sqrt (x - 3) + 1 = y) : sqrt (x + y) = 2 :=
by
  sorry

end sqrt_x_plus_y_eq_two_l39_39887


namespace floor_sqrt_sum_l39_39471

open Real

theorem floor_sqrt_sum : (∑ k in Finset.range 25, ⌊sqrt (k + 1)⌋) = 75 := 
by
  sorry

end floor_sqrt_sum_l39_39471


namespace range_f_mul_g_and_h_l39_39303

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := f x + g x

theorem range_f_mul_g_and_h :
  (∀ x, f x ∈ Set.Icc (-7 : ℝ) 2) →
  (∀ x, g x ∈ Set.Icc (-3 : ℝ) 4) →
  (∃ (a b : ℝ), a = -28 ∧ b = 21 ∧ (∀ x, f x * g x ∈ Set.Icc a b)) ∧
  (∃ (c d : ℝ), c = -10 ∧ d = 6 ∧ (∀ x, h x ∈ Set.Icc c d)) :=
by
  intros hf hg
  split
  · use [-28, 21]
    split; try { refl }
    sorry
  · use [-10, 6]
    split; try { refl }
    sorry

end range_f_mul_g_and_h_l39_39303


namespace theorem_statements_l39_39524

variables 
  (O A B C P : Point)
  (vector_space : Type u)
  [add_comm_group vector_space]
  [vector_space A]
  [vector_space B]
  [vector_space C]
  [vector_space P]

-- Definitions of the vector relationships
def statement_a := (A B C P : point) (O : point → (vector_space)) (OP : point) := 
  (def OP := (1/2) * (OA) - OB + (1/2) * OC) → ¬coplanar {P, A, B, C}

def statement_b := (A B C P : point) (O : point → (vector_space)) (OP : point) := 
  (def OP := (-1/3) * (OA) + 2 * OB - (2/3) * OC) → coplanar {P, A, B, C}

def statement_c := (A B P : point) (O : point → (vector_space)) (OP : point) := 
  (def OP := (-1/3) * (OA) + (4/3) * OB) → collinear {P, A, B}

def statement_d := (A B P : point) (O : point → (vector_space)) (OP : point) := 
  (def OP := OA + 2 * (AB)) → midpoint B (segment A P)

-- The final theorem statement
theorem theorem_statements : 
  (statement_a) ∧ 
  (statement_b) ∧ 
  (statement_c) ∧ 
  (statement_d) :=
by sorry

end theorem_statements_l39_39524


namespace ortho_center_on_g_l39_39639

-- Noncomputable theory needed if the incenters are treated as non-explicit constructions.
noncomputable theory

-- Definitions and theorem for the geometrical problem.
open EuclideanGeometry

variables {A B C D M N : Point}
variables g : Line
variables (h1 : On A g) (h2 : On M g) (h3 : M ≠ A)
variables (h4 : On M (Line.through B C)) (h5 : On N g) (h6 : On N (Line.through C D))
variables (h7 : CirumscribedQuadrilateral A B C D)
variables (I1 I2 I3 : Point)
variables (h8 : Incenter I1 (Triangle.mk A B M))
variables (h9 : Incenter I2 (Triangle.mk M N C))
variables (h10 : Incenter I3 (Triangle.mk N D A))

theorem ortho_center_on_g : Orthocenter (Triangle.mk I1 I2 I3) ∈ g :=
sorry

end ortho_center_on_g_l39_39639


namespace area_of_given_polygon_l39_39270

def point := (ℝ × ℝ)

def vertices : List point := [(0,0), (5,0), (5,2), (3,2), (3,3), (2,3), (2,2), (0,2), (0,0)]

def polygon_area (vertices : List point) : ℝ := 
  -- Function to compute the area of the given polygon
  -- Implementation of the area computation is assumed to be correct
  sorry

theorem area_of_given_polygon : polygon_area vertices = 11 :=
sorry

end area_of_given_polygon_l39_39270


namespace max_length_OB_l39_39352

-- Given conditions
variables {O A B : Type} [MetricSpace O] [MetricSpace A] [MetricSpace B]
variables (OA : dist O A) (OB : dist O B)
variables (angle : ℝ) (AB : ℝ)

-- Conditions from the problem
def angle_OAB : ℝ := O.angle A B -- the angle ∠OAB
def angle_AOB : ℝ := 45 -- the angle ∠AOB in degrees
def length_AB : ℝ := 1 -- the length of AB

-- Statement of the theorem
theorem max_length_OB (h1: angle_AOB = 45) (h2: length_AB = 1):
  ∃ OB_max : ℝ, OB_max = sqrt 2 := 
sorry

end max_length_OB_l39_39352


namespace billy_reads_books_l39_39806

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end billy_reads_books_l39_39806


namespace series_sum_equals_seven_ninths_l39_39843

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l39_39843


namespace incenter_inequality_1_incenter_inequality_2_l39_39149

variables {A B C I A1 B1 C1 : Point}
variable {r R : ℝ}
variables (h1 : isIncenter I A B C) (h2 : incircleTouches A1 B1 C1 A B C)

theorem incenter_inequality_1:
  IA1 + IB1 + IC1 >= IA + IB + IC := 
  by
  sorry

theorem incenter_inequality_2:
  IA1 * IB1 * IC1 >= IA * IB * IC :=
  by
  sorry

end incenter_inequality_1_incenter_inequality_2_l39_39149


namespace no_solution_exists_l39_39851

theorem no_solution_exists (t s : ℝ) (k : ℝ) : (1 : ℝ) = 0 → (∃ k, k = 18 / 5) ↔
  ∀ t s : ℝ, (1, 4) + t • (5, -9) = (0, 1) + s • (-2, k) → false :=
sorry

end no_solution_exists_l39_39851


namespace strictly_increasing_intervals_extreme_values_l39_39140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2

theorem strictly_increasing_intervals (a : ℝ) (x : ℝ) : 
  (a ≤ 0 ∧ x > 0 → ∀ x > 0, deriv (f a x) > 0) ∧
  (a > 0 ∧ 0 < x < 1 / a → deriv (f a x) > 0) := 
sorry

theorem extreme_values (a : ℝ) (x : ℝ) : 
  (a ≥ 0 → ¬ ∃ x > 0, deriv (F a x) = 0) ∧
  (a < 0 → ∃ x = real.sqrt (-1 / (2 * a)), 
    deriv (F a x) = 0 ∧ 
    ∀ (h : x > 0), F a (real.sqrt (-1 / (2 * a))) = log (real.sqrt (-1 / (2 * a))) - 1 / 2) :=
sorry

end strictly_increasing_intervals_extreme_values_l39_39140


namespace catch_up_time_l39_39198

-- Definitions of the conditions
def my_time_to_school : ℕ := 30 
def brother_time_to_school : ℕ := 40 
def brother_head_start : ℕ := 5 

-- Statement of the theorem
theorem catch_up_time : 
  let my_speed := (1 / my_time_to_school : ℚ), 
      brother_speed := (1 / brother_time_to_school : ℚ),
      relative_speed := my_speed - brother_speed,
      head_start_distance := brother_speed * brother_head_start in
  (head_start_distance / relative_speed = 15) := sorry

end catch_up_time_l39_39198


namespace sums_ratio_l39_39429

theorem sums_ratio (total_sums : ℕ) (sums_right : ℕ) (sums_wrong: ℕ) (h1 : total_sums = 24) (h2 : sums_right = 8) (h3 : sums_wrong = total_sums - sums_right) :
  sums_wrong / Nat.gcd sums_wrong sums_right = 2 ∧ sums_right / Nat.gcd sums_wrong sums_right = 1 := by
  sorry

end sums_ratio_l39_39429


namespace pentagon_area_l39_39652

theorem pentagon_area (A B C D E : Type) [convex_pentagon A B C D E]
  (h1 : dist A B = 1) (h2 : dist A E = 1) (h3 : dist C D = 1) 
  (h4 : dist B C + dist D E = 1) (h5 : angle B A C = 90) (h6 : angle D E A = 90) 
  : area A B C D E = 1 :=
sorry

end pentagon_area_l39_39652


namespace painted_cubes_l39_39772

/-- 
  Given a cube of side 9 painted red and cut into smaller cubes of side 3,
  prove the number of smaller cubes with paint on exactly 2 sides is 12.
-/
theorem painted_cubes (l : ℕ) (s : ℕ) (n : ℕ) (edges : ℕ) (faces : ℕ)
  (hcube_dimension : l = 9) (hsmaller_cubes_dimension : s = 3) 
  (hedges : edges = 12) (hfaces : faces * edges = 12) 
  (htotal_cubes : n = (l^3) / (s^3)) : 
  n * faces = 12 :=
sorry

end painted_cubes_l39_39772


namespace jillian_distance_l39_39234

theorem jillian_distance : 
  ∀ (x y z : ℝ),
  (1 / 63) * x + (1 / 77) * y + (1 / 99) * z = 11 / 3 →
  (1 / 63) * z + (1 / 77) * y + (1 / 99) * x = 13 / 3 →
  x + y + z = 308 :=
by
  sorry

end jillian_distance_l39_39234


namespace sum_of_decimals_is_fraction_l39_39510

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l39_39510


namespace quadratic_properties_l39_39924

def quadratic_function (a b c : ℝ) := λ x, a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h0 : a < 0) (h1 : quadratic_function a b c (-1) = 0)
  (h2 : ∀ m : ℝ, let f := quadratic_function a b c in f(m) ≤ -4 * a)
  (h3 : b = -2 * a) (h4: c = -3 * a):
  (a - b + c = 0) ∧ 
  (∀ x1 x2 : ℝ, quadratic_function a b (c + 1) x1 = 0 → quadratic_function a b (c + 1) x2 = 0 →
    x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39924


namespace nina_total_money_l39_39750

def original_cost_widget (C : ℝ) : ℝ := C
def num_widgets_nina_can_buy_original (C : ℝ) : ℝ := 6
def num_widgets_nina_can_buy_reduced (C : ℝ) : ℝ := 8
def cost_reduction : ℝ := 1.5

theorem nina_total_money (C : ℝ) (hc : 6 * C = 8 * (C - cost_reduction)) : 
  6 * C = 36 :=
by
  sorry

end nina_total_money_l39_39750


namespace sticker_distribution_l39_39576

theorem sticker_distribution :
  ∃! d : fin 4 → ℕ, (Σ i, d i = 10) ∧ (∀ i, d i ≤ 5) := sorry

end sticker_distribution_l39_39576


namespace general_operation_rule_l39_39737

theorem general_operation_rule (n : ℕ) (hn : n > 0): sqrt (n + 1/(n + 2)) = (n + 1) * sqrt (1/(n + 2)) :=
by sorry

example : sqrt (2022 + 1/2024) * sqrt 4048 = 2023 * sqrt 2 :=
by {
  have h := general_operation_rule 2022 (by norm_num),
  rw h,
  sorry
}

end general_operation_rule_l39_39737


namespace contrapositive_divisibility_l39_39359

-- Definitions corresponding to conditions from part a)
def is_divisible_by (n k : ℕ) : Prop := ∃ m : ℕ, k = n * m

def not_divisible_by_three (n : ℕ) : Prop := ¬ is_divisible_by 3 n

-- Statement based on conditions, questions, and answer
theorem contrapositive_divisibility (a b : ℕ) : 
  not_divisible_by_three a ∧ not_divisible_by_three b → 
  not is_divisible_by 3 (a * b) :=
begin
  sorry
end

end contrapositive_divisibility_l39_39359


namespace probability_number_is_odd_l39_39697

theorem probability_number_is_odd : 
  let digits := [2, 4, 5, 7]
  let total_permutations := Nat.factorial 4
  let odd_permutations := (Nat.factorial 3 + Nat.factorial 3)
  let probability := (odd_permutations : ℚ) / (total_permutations : ℚ)
  in probability = 1 / 2 :=
begin
  sorry
end

end probability_number_is_odd_l39_39697


namespace table_height_kmn_sum_l39_39621

-- Definition of the given triangle sides.
def BC : ℕ := 23
def CA : ℕ := 27
def AB : ℕ := 30

-- Parallel conditions of segments after the folds.
def UV_parallel_BC := true
def WX_parallel_AB := true
def YZ_parallel_CA := true

-- Definition of height h in the given form k * sqrt(m) / n.
def height := 40 * (real.sqrt 221) / 57

-- The goal is to prove that given the conditions, the expression for h leads to k + m + n = 318.
theorem table_height_kmn_sum :
  UV_parallel_BC ∧ WX_parallel_AB ∧ YZ_parallel_CA →
  (40 + 221 + 57 = 318) :=
by
  intro conditions
  sorry

end table_height_kmn_sum_l39_39621


namespace pump_A_time_to_empty_pool_l39_39378

theorem pump_A_time_to_empty_pool :
  ∃ (A : ℝ), (1/A + 1/9 = 1/3.6) ∧ A = 6 :=
sorry

end pump_A_time_to_empty_pool_l39_39378


namespace invested_sum_is_5000_l39_39771

-- Definitions based on conditions
def interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := (P * R * T) / 100

def P := 5000
def T := 2
def R1 := 18
def R2 := 12

-- Interest calculations
def SI1 := interest P R1 T
def SI2 := interest P R2 T
def interest_diff := SI1 - SI2

theorem invested_sum_is_5000 : P = 5000 ∧ interest_diff = 600 := by
  sorry

end invested_sum_is_5000_l39_39771


namespace integral_arctan_sqrt_l39_39444

-- Define the integral problem and the solution equivalence to the correct answer 
theorem integral_arctan_sqrt (C : ℝ) :
  ∫ x in 0..(λ x, arctan (sqrt (4 * x - 1))), dx = 
  λ x, x * (arctan (sqrt (4 * x - 1))) - (1/4) * sqrt (4 * x - 1) + C :=
by
  sorry

end integral_arctan_sqrt_l39_39444


namespace matrix_power_101_l39_39634

noncomputable def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_power_101 :
  (matrix_B ^ 101) = ![![0, 0, 1], ![1, 0, 0], ![0, 1, 0]] :=
  sorry

end matrix_power_101_l39_39634


namespace exact_two_visits_l39_39079

theorem exact_two_visits (days_period : ℕ) (alice_freq : ℕ) (beatrix_freq : ℕ) (claire_freq : ℕ) :
  days_period = 365 ∧ alice_freq = 4 ∧ beatrix_freq = 6 ∧ claire_freq = 8 →
  (exact_two_friends_visits days_period alice_freq beatrix_freq claire_freq) = 60 :=
by
  sorry

end exact_two_visits_l39_39079


namespace circle_radius_ratio_l39_39620

section GeometryProblem

variables (A O B C : Type) [RightAngledSector A O B]
variables (OC : Arc O C) (BO : Real := arcRadius OC) (ω ω' : Circle)
variables (tangent_AB_OA : TangentTo ω.arc_AB ω.line_OA)
variables (tangent_OC_ω'_OA : TangentTo ω'.arc_OC ω'.line_OA)

theorem circle_radius_ratio :
  let r : Real := radius ω in
  let r' : Real := radius ω' in
  (r / r') = (7 ± 2 * Real.sqrt 6) / 6 :=
sorry

end GeometryProblem

end circle_radius_ratio_l39_39620


namespace benny_comic_books_l39_39443

theorem benny_comic_books (x : ℕ) (h : (1 / 3 : ℝ) * x + 15 = 45) : x = 90 := by
  suffices h₁ : (1 : ℝ) / 3 = 1 / 3 from sorry
  haveI : noncomputable_instance := by
    apply_instance
  sorry

end benny_comic_books_l39_39443


namespace george_income_l39_39134

def half (x: ℝ) : ℝ := x / 2

theorem george_income (income : ℝ) (H1 : half income - 20 = 100) : income = 240 := 
by 
  sorry

end george_income_l39_39134


namespace sum_of_decimals_as_fraction_l39_39496

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l39_39496


namespace two_digit_appearance_l39_39415

theorem two_digit_appearance (numbers : List (Fin 1000000)) (h_arbitrary_order : Permuted 1.toList numbers) :
  (∀ n : Fin 100000, ∃ (d1 d2 : Fin 10), (d1, d2) ∈ two_digit_segments numbers) :=
by
  sorry

-- Helper definition for extracting two-digit segments
def two_digit_segments (numbers : List Nat) : List (Fin 100 × Fin 10) :=
  numbers.bind fun n => (nat_to_string_pairs n.toString)

-- Auxiliary function that converts a string of digits to pairs of two-digit numbers
def nat_to_string_pairs (s : String) : List (Fin 100 × Fin 10) :=
  (s.toList.zip (s.toList.drop 1)).map (fun (a, b) => (Fin 100.ofNat! (val_digit a * 10 + val_digit b), Fin 10.ofNat! (val_digit b)))

-- Converts a single character digit to its numeric value
def val_digit (c : Char) : Nat :=
  c.cVal.to_nat - '0'.cVal.to_nat

end two_digit_appearance_l39_39415


namespace infinite_primes_dividing_S_l39_39241

noncomputable def infinite_set_of_pos_integers (S : Set ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ m ∈ S) ∧ ∀ n : ℕ, n ∈ S → n > 0

def set_of_sums (S : Set ℕ) : Set ℕ :=
  {t | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ t = x + y}

noncomputable def finitely_many_primes_condition (S : Set ℕ) (T : Set ℕ) : Prop :=
  {p : ℕ | Prime p ∧ p % 4 = 1 ∧ (∃ t ∈ T, p ∣ t)}.Finite

theorem infinite_primes_dividing_S (S : Set ℕ) (T := set_of_sums S)
  (hS : infinite_set_of_pos_integers S)
  (hT : finitely_many_primes_condition S T) :
  {p : ℕ | Prime p ∧ ∃ s ∈ S, p ∣ s}.Infinite := 
sorry

end infinite_primes_dividing_S_l39_39241


namespace region_area_sum_l39_39066

theorem region_area_sum (h1 : ∃ (r1 r2 r3 : ℕ), r1 = 3 ∧ r2 = 2 ∧ r3 = 1) :
  let π := Real.pi in
  let A := λ r : ℕ, π * r * r in
  let Region1 := A 3 - A 2 in
  let Region3 := A 1 in
  Region1 + Region3 = 6 * π := by
    sorry

end region_area_sum_l39_39066


namespace average_height_of_60_students_l39_39307

theorem average_height_of_60_students :
  (35 * 22 + 25 * 18) / 60 = 20.33 := 
sorry

end average_height_of_60_students_l39_39307


namespace original_proposition_converse_false_l39_39570

theorem original_proposition (a b : ℝ) (h : a + b ≥ 2) : a ≥ 1 ∨ b ≥ 1 := sorry

theorem converse_false :
  ¬ (∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
begin
  intro h,
  specialize h 3 (-3),
  linarith,
end

end original_proposition_converse_false_l39_39570


namespace condo_total_units_l39_39782

-- Definitions from conditions
def total_floors := 23
def regular_units_per_floor := 12
def penthouse_units_per_floor := 2
def penthouse_floors := 2
def regular_floors := total_floors - penthouse_floors

-- Definition for total units
def total_units := (regular_floors * regular_units_per_floor) + (penthouse_floors * penthouse_units_per_floor)

-- Theorem statement: prove total units is 256
theorem condo_total_units : total_units = 256 :=
by
  sorry

end condo_total_units_l39_39782


namespace number_of_positive_divisors_of_60_l39_39965

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l39_39965


namespace collinear_PAB_l39_39551

variables (P A B C : Type) [AddCommGroup P] [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]

axiom distinct_points : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C
axiom vector_eq : (P + A + B = A + C)

theorem collinear_PAB : (P + A + B = A + C) → collinear P A B :=
by
  assume h : P + A + B = A + C
  sorry

end collinear_PAB_l39_39551


namespace angles_set_equality_solution_l39_39096

theorem angles_set_equality_solution (α : ℝ) :
  ({Real.sin α, Real.sin (2 * α), Real.sin (3 * α)} = {Real.cos α, Real.cos (2 * α), Real.cos (3 * α)}) ↔ 
  (∃ (k : ℤ), 0 ≤ k ∧ k ≤ 7 ∧ α = (k * Real.pi / 2) + (Real.pi / 8)) := 
by
  sorry

end angles_set_equality_solution_l39_39096


namespace proof_equivalent_problem_l39_39719

/- Definitions for the given conditions -/
variables 
  (a b : ℕ) -- excavation rates for type A and B excavators
  (m : ℕ)  -- number of type A excavators

/- Conditions given in the problem -/
def condition1 : Prop := 3 * a + 5 * b = 165
def condition2 : Prop := 4 * a + 7 * b = 225
def condition3 : Prop := 4 * (30 * m + 15 * (12 - m)) ≥ 1080
def condition4 : Prop := 480 * m + 8640 ≤ 12960
def condition5 : Prop := m + (12 - m) = 12 -- total of 12 excavators

/- Correct answers to prove -/
def correct_answers : Prop := 
  a = 30 ∧ b = 15 ∧ 
  (m = 7 ∨ m = 8 ∨ m = 9) ∧ 
  480 * 7 + 8640 = 12000

/- Lean statement combining everything -/
theorem proof_equivalent_problem :
  condition1 →
  condition2 →
  condition3 →
  condition4 →
  condition5 →
  correct_answers := 
by
  sorry

end proof_equivalent_problem_l39_39719


namespace simplify_expression_l39_39733

theorem simplify_expression : (2468 * 2468) / (2468 + 2468) = 1234 :=
by
  sorry

end simplify_expression_l39_39733


namespace square_garden_tiles_l39_39427

theorem square_garden_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_garden_tiles_l39_39427


namespace sum_of_numbers_taking_4_steps_to_palindrome_l39_39527

def is_palindrome (n : ℕ) : Prop := 
  let rev (n : ℕ) := n.digits 10.reverse.foldl (λ a i, 10*a + i) 0
  n = rev n

def reverse_and_add (n : ℕ) : ℕ :=
  let rev (n : ℕ) := n.digits 10.reverse.foldl (λ a i, 10*a + i) 0
  n + rev n

def takes_exactly_four_steps_to_palindrome (n : ℕ) : Prop :=
  let step1 := reverse_and_add n
  let step2 := reverse_and_add step1
  let step3 := reverse_and_add step2
  let step4 := reverse_and_add step3
  is_palindrome step4 ∧ ¬ is_palindrome step3

def sum_of_special_numbers : ℕ :=
  (Finset.range 200).filter (λ n, 100 ≤ n ∧ n < 200 ∧ ¬ is_palindrome n ∧ takes_exactly_four_steps_to_palindrome n).sum id

theorem sum_of_numbers_taking_4_steps_to_palindrome :
  sum_of_special_numbers = 356 :=
by sorry

end sum_of_numbers_taking_4_steps_to_palindrome_l39_39527


namespace win_sector_area_l39_39404

noncomputable def radius : ℝ := 7
noncomputable def probability_win : ℝ := 3 / 8
noncomputable def area_circle : ℝ := π * radius^2
noncomputable def area_win_sector : ℝ := probability_win * area_circle

theorem win_sector_area :
  area_win_sector = (147 * π) / 8 :=
by
  unfold area_win_sector
  unfold probability_win
  unfold area_circle
  sorry

end win_sector_area_l39_39404


namespace sqrt_x_plus_y_eq_two_l39_39888

theorem sqrt_x_plus_y_eq_two (x y : ℝ) (h : sqrt (3 - x) + sqrt (x - 3) + 1 = y) : sqrt (x + y) = 2 :=
by
  sorry

end sqrt_x_plus_y_eq_two_l39_39888


namespace prove_all_propositions_l39_39796

noncomputable def proposition_1 (a b c : ℝ) (h : a + b + c = 3) : Prop :=
  a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1

noncomputable def proposition_2 (z : ℂ) (h : |z| = 1) : Prop :=
  |z - complex.I| ≤ 2

noncomputable def proposition_3 (x : ℝ) (h : 0 < x) : Prop :=
  x > real.sin x

noncomputable def proposition_4 : Prop :=
  ∫ x in 0..real.sqrt real.pi, real.sqrt (real.pi - x^2) = real.pi^2 / 4

theorem prove_all_propositions
  (a b c : ℝ) (h₁ : a + b + c = 3)
  (z : ℂ) (h₂ : |z| = 1)
  (x : ℝ) (h₃ : 0 < x) :
  proposition_1 a b c h₁ ∧ proposition_2 z h₂ ∧ proposition_3 x h₃ ∧ proposition_4 :=
by
  split
  · sorry
  split
  · sorry
  split
  · sorry
  · sorry

end prove_all_propositions_l39_39796


namespace max_dot_product_ellipse_l39_39641

theorem max_dot_product_ellipse :
  let ellipse := λ x y : ℝ, x^2 + y^2 / 4 = 1
  let F1 := (0, -Real.sqrt 3)
  let O := (0, 0)
  let P_f (x y : ℝ) := y^2 / 4 + x^2 = 1
    ∨ -2 ≤ y ∧ y ≤ 2 → Real.sqrt 3 * y + 4 + y^2 ≤ 4 + 2 * Real.sqrt 3 → (x, y) = (x, 2) → 4 + 2 * Real.sqrt 3 ≤ ((P_f x y) -^) (F1,P,O)
(P : ℝ × ℝ) : max (y : ℝ) = 4 + 2 * Real.sqrt 3 :=
  
begin
  -- Define the ellipse
  have ellipse_eq : ∀ {x y : ℝ}, ellipse x y = x^2 + y^2 / 4 = 1 := by sorry,
  
  -- Define point F1
  let F1 := (0, -Real.sqrt 3),
  
  -- Define the origin
  let O := (0, 0),
  
  -- Define the point P
  let P := (x,y),
  
  
  -- Prove
  show 4 + 2 * Real.sqrt 3 ≤ max (y : ℝ) ((λ y : ℝ, ( ∣ Real.sqrt 3 ≤ F y))
end

end max_dot_product_ellipse_l39_39641


namespace number_of_topologies_l39_39997

def A_2 : Set ℕ := {0, 1, 4}

def is_topology (τ : Set (Set ℕ)) : Prop :=
  ∅ ∈ τ ∧ A_2 ∈ τ ∧
  (∀ s ∈ τ, ∀ t ∈ τ, s ∪ t ∈ τ) ∧
  (∀ s ∈ τ, ∀ t ∈ τ, s ∩ t ∈ τ)

def topologies_containing_4_elements : Set (Set (Set ℕ)) :=
  {τ | is_topology τ ∧ τ.finite ∧ τ.card = 4}

theorem number_of_topologies : topologies_containing_4_elements.finite ∧
  topologies_containing_4_elements.card = 9 :=
sorry

end number_of_topologies_l39_39997


namespace intersection_A_B_l39_39655

def A : Set ℝ := {x | 1 < x}
def B : Set ℝ := {y | y ≤ 2}
def expected_intersection : Set ℝ := {z | 1 < z ∧ z ≤ 2}

theorem intersection_A_B : (A ∩ B) = expected_intersection :=
by
  -- Proof to be completed
  sorry

end intersection_A_B_l39_39655


namespace count_correct_propositions_l39_39064

open Classical

theorem count_correct_propositions :
  let P1 := ∀ x : ℝ, x^2 ≥ x
  let P2 := ∃ x : ℝ, x^2 ≥ x
  let P3 := 4 ≥ 3
  let P4 := ∀ x : ℝ, (x^2 ≠ 1 ↔ (x ≠ 1 ∧ x ≠ -1))
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) → 
  (number_of_correct := 2) :=
by
  sorry

end count_correct_propositions_l39_39064


namespace area_of_triangle_formed_by_tangents_l39_39956

theorem area_of_triangle_formed_by_tangents (R r : ℝ) (h : R > 0 ∧ r > 0) (perp_tangents : ∀ (d : ℝ), d > R + r → d = sqrt(R^2 + r^2)) :
  let S := R * r in
  S = R * r :=
by
  sorry

end area_of_triangle_formed_by_tangents_l39_39956


namespace x_axis_intercept_of_line_l39_39702

theorem x_axis_intercept_of_line (x : ℝ) : (∃ x, 2*x + 1 = 0) → x = - 1 / 2 :=
  by
    intro h
    obtain ⟨x, h1⟩ := h
    have : 2 * x + 1 = 0 := h1
    linarith [this]

end x_axis_intercept_of_line_l39_39702


namespace find_a_given_conditions_l39_39571

theorem find_a_given_conditions (a : ℤ)
  (hA : ∃ (x : ℤ), x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)
  (hA_contains_minus3 : ∃ (x : ℤ), (-3 = x) ∧ (x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)) : a = -3 := 
by
  sorry

end find_a_given_conditions_l39_39571


namespace average_income_eq_58_l39_39766

def income_day1 : ℕ := 45
def income_day2 : ℕ := 50
def income_day3 : ℕ := 60
def income_day4 : ℕ := 65
def income_day5 : ℕ := 70
def number_of_days : ℕ := 5

theorem average_income_eq_58 :
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / number_of_days = 58 := by
  sorry

end average_income_eq_58_l39_39766


namespace derivative_of_e_x_cos_x_l39_39312

theorem derivative_of_e_x_cos_x :
  ∀ (x : ℝ), (deriv (λ x, (Real.exp x) * (Real.cos x))) x = (Real.exp x) * (Real.cos x - Real.sin x) :=
by
  intro x
  sorry

end derivative_of_e_x_cos_x_l39_39312


namespace arrangement_in_circular_holder_l39_39033

/-- 
  Given:
  - 5 black pencils
  - 3 blue pens
  - 1 red pen
  - 1 green pen
  
  Prove that the number of ways to arrange these writing utensils in a circular holder,
  such that no two blue pens are adjacent, is 168.
-/
theorem arrangement_in_circular_holder :
  let total_arrangements := (9.fact) / (5.fact * 3.fact * 1.fact * 1.fact),
      blue_adjacent_arrangements := (8.fact) / (5.fact * 1.fact * 1.fact * 1.fact * 1.fact),
      non_adjacent_arrangements := total_arrangements - blue_adjacent_arrangements
  in  non_adjacent_arrangements = 168 :=
by
  sorry

end arrangement_in_circular_holder_l39_39033


namespace quadratic_conclusions_l39_39932

variables {a b c : ℝ} (h1 : a < 0) (h2 : a - b + c = 0)

theorem quadratic_conclusions
    (h_intersect : ∃ x, a * x ^ 2 + b * x + c = 0 ∧ x = -1)
    (h_symmetry : ∀ x, x = 1 → a * (x - 1) ^ 2 + b * (x - 1) + c = a * (x + 1) ^ 2 + b * (x + 1) + c) :
    a - b + c = 0 ∧ 
    (∀ m : ℝ, a * m ^ 2 + b * m + c ≤ -4 * a) ∧ 
    (∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c + 1 = 0 ∧ a * x2 ^ 2 + b * x2 + c + 1 = 0 ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
begin
    sorry
end

end quadratic_conclusions_l39_39932


namespace distance_between_parallel_lines_l39_39344

theorem distance_between_parallel_lines :
  let line1 := 3 * x + 4 * y - 12
  let line2 := 6 * x + 8 * y + 11
  let distance := fun (a b c1 c2 : ℝ) => abs(c2 - c1) / sqrt(a^2 + b^2)
  distance 6 8 (-24) 11 = 7 / 2 :=
by
  sorry

end distance_between_parallel_lines_l39_39344


namespace tangent_line_value_l39_39167

theorem tangent_line_value (f : ℝ) (f' : ℝ) (h : tangent_line_at_point (λ x, f) (2, f) = (λ x, x + 4)) : f + f' = 7 :=
by sorry

end tangent_line_value_l39_39167


namespace sum_of_floor_sqrt_l39_39459

theorem sum_of_floor_sqrt :
  (∑ i in Finset.range 25, (Nat.sqrt (i + 1))) = 75 := by
  sorry

end sum_of_floor_sqrt_l39_39459


namespace arithmetic_sequence_sum_l39_39155

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sequence_sum (a d : ℝ) (n : ℕ) : ℝ :=
  (n * (a + arithmetic_sequence a d n)) / 2

-- Given assumptions
variables (a d : ℝ)

theorem arithmetic_sequence_sum :
  (arithmetic_sequence a d 1) + (arithmetic_sequence a d 3) + (arithmetic_sequence a d 5) = 3 →
  sequence_sum a d 5 = 5 :=
begin
  sorry
end

end arithmetic_sequence_sum_l39_39155


namespace total_number_of_flowers_l39_39327

theorem total_number_of_flowers (pots : ℕ) (flowers_per_pot : ℕ) (h_pots : pots = 544) (h_flowers_per_pot : flowers_per_pot = 32) : 
  pots * flowers_per_pot = 17408 := by
  sorry

end total_number_of_flowers_l39_39327


namespace work_days_in_week_l39_39031

theorem work_days_in_week (total_toys_per_week : ℕ) (toys_produced_each_day : ℕ) (h1 : total_toys_per_week = 6500) (h2 : toys_produced_each_day = 1300) : 
  total_toys_per_week / toys_produced_each_day = 5 :=
by
  sorry

end work_days_in_week_l39_39031


namespace intersection_of_lines_l39_39864

theorem intersection_of_lines :
  ∃ x y : ℚ, 8 * x - 5 * y = 40 ∧ 6 * x + 2 * y = 14 ∧ x = 75 / 23 ∧ y = -64 / 23 :=
by
  use 75 / 23, -64 / 23
  split
  { calc 
      8 * (75 / 23) - 5 * (-64 / 23) = 8 * (75 / 23) + 5 * (64 / 23) : by ring
      ... = (8 * 75 + 5 * 64) / 23 : by ring
      ... = 40 : by norm_num },
  split
  { calc 
      6 * (75 / 23) + 2 * (-64 / 23) = 6 * (75 / 23) - 2 * (64 / 23) : by ring
      ... = (6 * 75 - 2 * 64) / 23 : by ring
      ... = 14 : by norm_num },
  split
  { norm_num },
  { norm_num }

end intersection_of_lines_l39_39864


namespace exponential_sum_sequence_l39_39936

noncomputable def Sn (n : ℕ) : ℝ :=
  Real.log (1 + 1 / n)

theorem exponential_sum_sequence : 
  e^(Sn 9 - Sn 6) = (20 : ℝ) / 21 := by
  sorry

end exponential_sum_sequence_l39_39936


namespace final_apples_count_l39_39313

def initial_apples : ℝ := 5708
def apples_given_away : ℝ := 2347.5
def additional_apples_harvested : ℝ := 1526.75

theorem final_apples_count :
  initial_apples - apples_given_away + additional_apples_harvested = 4887.25 :=
by
  sorry

end final_apples_count_l39_39313


namespace time_after_1876_minutes_l39_39324

-- Define the structure for Time
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
noncomputable def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.minute + m
  let additional_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let new_hour := (t.hour + additional_hours) % 24
  { hour := new_hour, minute := remaining_minutes }

-- Definition of the starting time
def three_pm : Time := { hour := 15, minute := 0 }

-- The main theorem statement
theorem time_after_1876_minutes : add_minutes three_pm 1876 = { hour := 10, minute := 16 } :=
  sorry

end time_after_1876_minutes_l39_39324


namespace curve_to_polar_l39_39709

noncomputable def polar_eq_of_curve (x y : ℝ) (ρ θ : ℝ) : Prop :=
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ (x ^ 2 + y ^ 2 - 2 * x = 0) → (ρ = 2 * Real.cos θ)

theorem curve_to_polar (x y ρ θ : ℝ) :
  polar_eq_of_curve x y ρ θ :=
sorry

end curve_to_polar_l39_39709


namespace stock_problem_l39_39387

variable (p0 p1 p2 p3 p4 p5 : ℝ)
variable (n : ℕ)
variable (Delta : ℕ → ℝ)

-- Initial conditions
axiom h1 : p0 = 25
axiom h2 : n = 1000
axiom h3 : Delta 0 = 2
axiom h4 : Delta 1 = -0.5
axiom h5 : Delta 2 = 1.5
axiom h6 : Delta 3 = -1.8
axiom h7 : Delta 4 = 0.8

-- Closing prices definitions
def p1 := p0 + Delta 0
def p2 := p1 + Delta 1
def p3 := p2 + Delta 2
def p4 := p3 + Delta 3
def p5 := p4 + Delta 4

-- Proof statements
axiom a1 : p1 = 27
axiom a2 : p3 = 28
axiom a3 : p4 = 26.2
axiom a4 : n * p5 - n * p0 = 2000

theorem stock_problem (h1 h2 h3 h4 h5 h6 h7 : Prop) : (p1 = 27) ∧ (n * p5 - n * p0 = 2000) ∧ 
    (∃ (maxp), maxp = 28) ∧ (∃ (minp), minp = 26.2) :=
  by
    exact ⟨a1, a4, ⟨28, a2⟩, ⟨26.2, a3⟩⟩

end stock_problem_l39_39387


namespace george_monthly_income_l39_39132

theorem george_monthly_income (I : ℝ) (h : I / 2 - 20 = 100) : I = 240 := 
by sorry

end george_monthly_income_l39_39132


namespace floor_sqrt_sum_l39_39463

theorem floor_sqrt_sum : (∑ x in Finset.range 25 + 1, (Nat.floor (Real.sqrt x : ℝ))) = 75 := 
begin
  sorry
end

end floor_sqrt_sum_l39_39463


namespace range_of_a_if_two_is_solution_l39_39288

variable (a : ℝ)

-- Define the inequality condition with x = 2
def inequality_condition : Prop := (2 : ℝ) ∈ { x | 2 * x^2 + a * x - a^2 > 0 }

-- State the theorem
theorem range_of_a_if_two_is_solution : inequality_condition a → -2 < a ∧ a < 4 :=
begin
  sorry
end

end range_of_a_if_two_is_solution_l39_39288


namespace maintenance_check_time_l39_39402

/-- A certain protective additive increases the time between required maintenance checks 
on an industrial vehicle from 30 days to some days. The time between maintenance checks 
is increased by 100% by using the additive. -/
theorem maintenance_check_time (t_original : ℕ) (percentage_increase : ℕ) 
  (doubled : percentage_increase = 100) (h_t_original : t_original = 30) : 
  let t_new := t_original * 2 
  in t_new = 60 := 
by
  rw [h_t_original]
  rw [doubled]
  dsimp [t_new]
  norm_num
  sorry

end maintenance_check_time_l39_39402


namespace steel_strength_value_l39_39855

theorem steel_strength_value 
  (s : ℝ) 
  (condition: s = 4.6 * 10^8) : 
  s = 460000000 := 
by sorry

end steel_strength_value_l39_39855


namespace toys_produced_on_sunday_l39_39773

-- Given conditions
def factory_production (day: ℕ) : ℕ :=
  2500 + 25 * day

theorem toys_produced_on_sunday : factory_production 6 = 2650 :=
by {
  -- The proof steps are omitted as they are not required.
  sorry
}

end toys_produced_on_sunday_l39_39773


namespace floor_sqrt_sum_l39_39462

theorem floor_sqrt_sum : (∑ x in Finset.range 25 + 1, (Nat.floor (Real.sqrt x : ℝ))) = 75 := 
begin
  sorry
end

end floor_sqrt_sum_l39_39462


namespace integral_solution_eq_l39_39756

open Real

noncomputable def integral_expression (x : ℝ) : ℝ :=
(2 * x ^ 3 - 6 * x ^ 2 + 7 * x - 4) / ((x - 2) * (x - 1) ^ 3)

theorem integral_solution_eq (C : ℝ) :
  ∫ x in ℝ, integral_expression x = 2 * ln |x - 2| - 1 / (2 * (x - 1)^2) + C :=
sorry

end integral_solution_eq_l39_39756


namespace p_implies_q_q_not_implies_p_l39_39248

variable {a : ℝ}
def p : Prop := a^2 + a ≠ 0
def q : Prop := a ≠ 0

theorem p_implies_q : p → q := by
  sorry

theorem q_not_implies_p : ¬ (q → p) := by
  sorry

end p_implies_q_q_not_implies_p_l39_39248


namespace trigonometric_expression_value_l39_39820

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l39_39820


namespace trajectory_of_P_l39_39152

noncomputable def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def F1 : point := (2, 0)
def F2 : point := (-2, 0)

theorem trajectory_of_P (a : ℝ) (a_pos : a > 0) (P : point)
  (h : distance P F1 + distance P F2 = 4 * a + 1 / a) :
  ∃ T, (T = "Ellipse" ∨ T = "Line Segment") :=
sorry

end trajectory_of_P_l39_39152


namespace number_of_elements_in_C_l39_39638

def A := {1, 2, 3, 4, 5}
def B := {1, 2, 3}
def C := { z | ∃ x ∈ A, ∃ y ∈ B, z = x * y }

theorem number_of_elements_in_C : Fintype.card C = 11 :=
by
  sorry

end number_of_elements_in_C_l39_39638


namespace boy_work_completion_days_l39_39041

theorem boy_work_completion_days (M W B : ℚ) (D : ℚ)
  (h1 : M + W + B = 1 / 4)
  (h2 : M = 1 / 6)
  (h3 : W = 1 / 36)
  (h4 : B = 1 / D) :
  D = 18 := by
  sorry

end boy_work_completion_days_l39_39041


namespace A_alone_completes_in_25_days_l39_39778

-- Define the conditions as constants
constant W : ℕ -- Work done by B per hour
constant A_workman_factor : ℕ := 2
constant C_workman_factor : ℕ := 3
constant A_daily_hours : ℕ := 6
constant B_daily_hours : ℕ := 4
constant C_daily_hours : ℕ := 3
constant total_days_together : ℕ := 12

-- Define the derived values for daily work
def A_daily_work := A_daily_hours * (A_workman_factor * W)
def B_daily_work := B_daily_hours * W
def C_daily_work := C_daily_hours * (C_workman_factor * W)

-- Total work done in one day by A, B, and C together
def daily_total_work := A_daily_work + B_daily_work + C_daily_work

-- Total work done over total_days_together days
def total_work_done := daily_total_work * total_days_together

-- Daily work done by A alone
def A_daily_work_alone := A_daily_work

-- Number of days it would take for A alone to complete the work
def days_for_A_alone := total_work_done / A_daily_work_alone

-- The statement to prove
theorem A_alone_completes_in_25_days : days_for_A_alone = 25 := by
  sorry

end A_alone_completes_in_25_days_l39_39778


namespace secants_ratio_constant_l39_39306

-- Definitions for points and secant lines passing through a circle's center
noncomputable def circle_center {k : Type*} [field k] (O : k × k) (r : k) := {p : k × k // (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

noncomputable def is_secant {k : Type*} [field k] (O : k × k) (p1 p2 : k × k) :=
  (∃ (M : k × k) (N : k × k), (M ∈ circle_center O 1) ∧ (N ∈ circle_center O 1) ∧ M ≠ N ∧ 
   collinear {p1, p2, M, N})

variables {k : Type*} [field k]
  
theorem secants_ratio_constant
  (O : k × k) (P M N A B : k × k)
  (hPMN_center : collinear {P, M, N, O} ∧ M ≠ N)
  (hPAB_secant : is_secant O P A B)
  : ∃ k : k, AM.1 * BM.1 / (AN.1 * BN.1) = k :=
  sorry

end secants_ratio_constant_l39_39306


namespace cone_water_volume_percentage_l39_39008

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39008


namespace complex_number_problem_l39_39169

noncomputable def z_i : ℂ := ( (complex.I + 1) / (complex.I - 1) )^(2016)

theorem complex_number_problem : z_i = -complex.I := 
sorry

end complex_number_problem_l39_39169


namespace f_at_neg2_l39_39918

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - Real.log (x^2 - 3*x + 5) / Real.log 3 
else -2^(-x) + Real.log ((-x)^2 + 3*(-x) + 5) / Real.log 3 

theorem f_at_neg2 : f (-2) = -3 := by
  sorry

end f_at_neg2_l39_39918


namespace radius_of_arch_bridge_l39_39561

theorem radius_of_arch_bridge :
  ∀ (AB CD AD r : ℝ),
    AB = 12 →
    CD = 4 →
    AD = AB / 2 →
    r^2 = AD^2 + (r - CD)^2 →
    r = 6.5 :=
by
  intros AB CD AD r hAB hCD hAD h_eq
  sorry

end radius_of_arch_bridge_l39_39561


namespace leaves_problem_l39_39616

noncomputable def leaves_dropped_last_day (L : ℕ) (n : ℕ) : ℕ :=
  L - n * (L / 10)

theorem leaves_problem (L : ℕ) (n : ℕ) (h1 : L = 340) (h2 : leaves_dropped_last_day L n = 204) :
  n = 4 :=
by {
  sorry
}

end leaves_problem_l39_39616


namespace jessica_more_than_rodney_l39_39291

-- Definitions based on the conditions
def rodney_more_than_ian := 35
def jessica_money := 100

-- Defining Ian's money in terms of Jessica's money
def ian_money := jessica_money / 2

-- Proving that Jessica has 15 dollars more than Rodney
theorem jessica_more_than_rodney :
  jessica_money - (ian_money + rodney_more_than_ian) = 15 := 
by {
  have ian_value : ian_money = 50, by simp [ian_money, jessica_money],
  have rodney_value : (ian_money + rodney_more_than_ian) = 85, by simp [ian_value, rodney_more_than_ian],
  simp [jessica_money, rodney_value],
  -- remaining steps would be the detailed mathematical proof
  simp,
  sorry
}

end jessica_more_than_rodney_l39_39291


namespace inequality_holds_l39_39480

theorem inequality_holds (x : ℝ) : (∀ y : ℝ, y > 0 → (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y) / (x + y) > 3 * x^2 * y)) ↔ x > 0 := 
sorry

end inequality_holds_l39_39480


namespace quadratic_properties_l39_39926

def quadratic_function (a b c : ℝ) := λ x, a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h0 : a < 0) (h1 : quadratic_function a b c (-1) = 0)
  (h2 : ∀ m : ℝ, let f := quadratic_function a b c in f(m) ≤ -4 * a)
  (h3 : b = -2 * a) (h4: c = -3 * a):
  (a - b + c = 0) ∧ 
  (∀ x1 x2 : ℝ, quadratic_function a b (c + 1) x1 = 0 → quadratic_function a b (c + 1) x2 = 0 →
    x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39926


namespace number_of_polynomials_l39_39473

-- Define the polynomial and its coefficients
def polynomial (coeffs : Fin 10 → ℕ) : Polynomial ℤ :=
  Polynomial.sum (Fin 10) (fun i => Polynomial.C (coeffs i) * Polynomial.X ^ i)

-- Condition: Each coefficient is either 0 or 1
def coeff_valid (coeffs : Fin 10 → ℕ) : Prop :=
  ∀ i, coeffs i = 0 ∨ coeffs i = 1

-- Condition: The polynomial must have exactly two different integer roots
def has_exactly_two_integer_roots (p : Polynomial ℤ) : Prop :=
  ∃ a b : ℤ, a ≠ b ∧ p.eval a = 0 ∧ p.eval b = 0 ∧ (∀ x, p.eval x = 0 → x = a ∨ x = b)

-- Main theorem to prove the number of such valid polynomials
theorem number_of_polynomials : 
  (Finset.univ.filter (fun coeffs => coeff_valid coeffs ∧ has_exactly_two_integer_roots (polynomial coeffs))).card = 146 := 
sorry

end number_of_polynomials_l39_39473


namespace range_of_m_l39_39594

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) (hineq : 4 / (x + 1) + 1 / y < m^2 + (3 / 2) * m) :
  m < -3 ∨ m > 3 / 2 :=
by sorry

end range_of_m_l39_39594


namespace relationship_between_a_b_c_l39_39575

noncomputable def a : ℝ := (1/2)^(-3)
noncomputable def b : ℝ := (-2)^2
noncomputable def c : ℝ := (real.pi - 2015)^0

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by
  have ha : a = 8 := by sorry
  have hb : b = 4 := by sorry
  have hc : c = 1 := by sorry
  rw [ha, hb, hc]
  exact and.intro (by norm_num) (by norm_num)

end relationship_between_a_b_c_l39_39575


namespace parallelogram_area_correct_l39_39279

noncomputable def parallelogram_area (s1 s2 : ℝ) (a : ℝ) : ℝ :=
s2 * (2 * s2 * Real.sin a)

theorem parallelogram_area_correct (s2 a : ℝ) (h_pos_s2 : 0 < s2) :
  parallelogram_area (2 * s2) s2 a = 2 * s2^2 * Real.sin a :=
by
  unfold parallelogram_area
  sorry

end parallelogram_area_correct_l39_39279


namespace range_of_a_l39_39367

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a < |x - 4| + |x + 3|) → a < 7 :=
by
  sorry

end range_of_a_l39_39367


namespace ellipse_properties_l39_39945

section Problem

variables {a b c x0 y0 : ℝ}
-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line equation
def line (x : ℝ) : ℝ := -x + 1

-- Define the midpoint condition for segment AB
def midpoint_condition (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the condition for the right focus of the ellipse
def right_focus_cond (b : ℝ) : Prop := (3/5 * b)^2 + (4/5 * b)^2 = 4

-- Define the property expressing the eccentricity
def eccentricity (a b : ℝ) : ℝ := (1 - (b^2 / a^2))^0.5

-- The main theorem that we need to prove
theorem ellipse_properties 
  (h1 : a > b > 0)
  (h2 : ∀ (x y : ℝ), ellipse x y → y = line x)
  (h3 : ∀ (x y : ℝ), midpoint_condition (a^2 / (a^2 + b^2)) (b^2 / (a^2 + b^2)))
  (h4 : ∀ (x0 y0 : ℝ), right_focus_cond b)
  : eccentricity a b = (1 / 2)^0.5 ∧ ((a^2 = 8) ∧ (b^2 = 4) ∧ ∀ x y, ellipse x y → x^2 / 8 + y^2 / 4 = 1) := 
sorry

end Problem

end ellipse_properties_l39_39945


namespace median_length_KE_l39_39613

-- Given definitions
variable {Point : Type} [MetricSpace Point]
variable (A B C D K : Point)
variable (a b : ℝ)
variable (S_ABCD S_AKD : ℝ)
variable [AddGroup Point]
variable [Module ℝ Point]

-- Given conditions
axiom midpoint_BC (hK : MetricSpace.Point.distance B K = MetricSpace.Point.distance K C)
axiom twice_area_AKD (hArea : S_ABCD = 2 * S_AKD)
axiom length_AB (hAB_eq : MetricSpace.Point.distance A B = a)
axiom length_CD (hCD_eq : MetricSpace.Point.distance C D = b)

-- Derived definition
noncomputable def length_KE : ℝ :=
  ((MetricSpace.Point.distance A B) + (MetricSpace.Point.distance C D)) / 2

-- Proof goal
theorem median_length_KE
    (hK : MetricSpace.Point.distance B K = MetricSpace.Point.distance K C)
    (hArea : S_ABCD = 2 * S_AKD)
    (hAB_eq : MetricSpace.Point.distance A B = a)
    (hCD_eq : MetricSpace.Point.distance C D = b) :
  MetricSpace.Point.distance K (Midpoint A D) = (a + b) / 2 := by
  sorry

end median_length_KE_l39_39613


namespace max_distance_between_spheres_l39_39366

open Real EuclideanGeometry 

def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ × ℝ) :=
  {point | dist point center = radius}

theorem max_distance_between_spheres :
  ∀ (E F : ℝ × ℝ × ℝ),
    E ∈ sphere (0, 0, 0) 25 →
    F ∈ sphere (20, 15, -25) 60 →
    dist E F ≤ 85 + 25 * Real.sqrt 2 :=
by
  intros E F hE hF
  sorry

end max_distance_between_spheres_l39_39366


namespace number_of_benches_l39_39042

-- Define the conditions
def bench_capacity : ℕ := 4
def people_sitting : ℕ := 80
def available_spaces : ℕ := 120
def total_capacity : ℕ := people_sitting + available_spaces -- this equals 200

-- The theorem to prove the number of benches
theorem number_of_benches (B : ℕ) : bench_capacity * B = total_capacity → B = 50 :=
by
  intro h
  exact sorry

end number_of_benches_l39_39042


namespace not_always_true_inequality_l39_39063

theorem not_always_true_inequality (x : ℝ) (hx : x > 0) : 2^x ≤ x^2 := sorry

end not_always_true_inequality_l39_39063


namespace equivalent_expression_l39_39088

theorem equivalent_expression : 8^8 * 4^4 / 2^28 = 16 := by
  -- Here, we're stating the equivalency directly
  sorry

end equivalent_expression_l39_39088


namespace house_rent_fraction_l39_39039

def salary : ℝ := 170000
def food_fraction : ℝ := 1/5
def clothes_fraction : ℝ := 3/5
def left_over : ℝ := 17000

theorem house_rent_fraction :
  ∃ H : ℝ, (food_fraction * salary + H * salary + clothes_fraction * salary + left_over = salary) ∧ H = 1/10 :=
by
  sorry

end house_rent_fraction_l39_39039


namespace sum_formula_sum_bound_l39_39543

noncomputable def arithmetic_sequence (a d : ℕ → ℕ) := ∀ n, a (n + 1) = a n + d

def sequence_properties (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  S 3 = 15 ∧ a 4 = 9

theorem sum_formula (a : ℕ → ℕ) (S : ℕ → ℕ)
  (H_seq : arithmetic_sequence a 2)
  (H_prop : sequence_properties a S) :
  ∀ n, S n = n * (n + 2) :=
sorry

def T_seq (S: ℕ → ℕ) (T : ℕ → ℕ) :=
  T n = Σ i in range n, 1 / S i

theorem sum_bound (S : ℕ → ℕ) (T : ℕ → ℕ)
  (H_seq : arithmetic_sequence a 2)
  (H_prop : sequence_properties a S)
  (H_sum : ∀ n, S n = n * (n + 2)) :
  ∀ n, T n < 3 / 4 :=
sorry

end sum_formula_sum_bound_l39_39543


namespace odd_factors_of_240_l39_39581

theorem odd_factors_of_240 : 
    let n : ℕ := 240
    let p1 : ℕ := 2
    let p2 : ℕ := 3
    let p3 : ℕ := 5
    let e1 : ℕ := 4
    let e2 : ℕ := 1
    let e3 : ℕ := 1
    let prime_factorization : n = p1^e1 * p2^e2 * p3^e3
    let odd_prime_factorization : ℕ := p2^e2 * p3^e3
    let odd_factors_count : ℕ := (e2 + 1) * (e3 + 1)
    odd_factors_count = 4 := by
  sorry

end odd_factors_of_240_l39_39581


namespace envelope_reflected_rays_cardioid_l39_39472

-- Define the unit circle S
def S : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = 1 }

-- Define the point A on the unit circle S
def A : ℝ × ℝ := (-1, 0)

-- Define the property we need to prove: the envelope of reflected rays forms a cardioid
theorem envelope_reflected_rays_cardioid (S A : set (ℝ × ℝ)) (h : A ∈ S): 
  ∃ C : set (ℝ × ℝ), is_cardioid C :=
sorry

end envelope_reflected_rays_cardioid_l39_39472


namespace symmetric_line_about_y_axis_l39_39180

theorem symmetric_line_about_y_axis (x y : ℝ) :
  (∀ x y : ℝ, y = 2 * x + 3 ∧ y = -2 * x + 3) :=
begin
  sorry
end

end symmetric_line_about_y_axis_l39_39180


namespace jonathan_distance_l39_39236

theorem jonathan_distance :
  ∀ (J : ℝ),
  let M := 2 * J,
  let D := 2 * J + 2,
  M + D = 32 → J = 7.5 :=
by
  intros J M D hM hD hTotal,
  sorry

end jonathan_distance_l39_39236


namespace coefficient_x2_in_expansion_l39_39246

noncomputable def a : ℝ := ∫ x in 0..3, (2 * x - 1)

theorem coefficient_x2_in_expansion :
  let expr := (x - (a / (2 * x)))^6
  let term := λ (r : ℕ), (Nat.choose 6 r) * (-3)^r * x^(6 - 2 * r)
  ∃ r : ℕ, (6 - 2 * r = 2) ∧ (term r = 135) :=
by
  sorry

end coefficient_x2_in_expansion_l39_39246


namespace tangent_line_eq_monotonicity_l39_39176
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a + a * real.log x

theorem tangent_line_eq (a : ℝ) (ha_pos : 0 < a) :
    f 1 1 = -0.5 ∧
    ∀ (x : ℝ), f' x 1 = x + 1 / x ∧ f' 1 1 = 2 ∧
    ∀ (y : ℝ), y - (-0.5) = 2 * (1 - 1) → y = 2 - 2.5 := by
  sorry

theorem monotonicity (a : ℝ) (ha_pos : 0 < a) :
    ∀ x > 0, x^2 + a > 0 → f' x a > 0 → strict_mono_on (λ x, f x a) (set.Ioi 0) := by
  sorry

end tangent_line_eq_monotonicity_l39_39176


namespace smallest_M_exists_l39_39852

theorem smallest_M_exists :
  ∃ M, M = 4044 ∧ ∀ (n : ℕ) (Hn : n > 0) (x : Fin n → ℝ) (Hx : ∀ i j, i < j → x i < x j) (H2023 : ∀ i, x i ≤ 2023),
    (∑ i in Finset.range n, ∑ j in Finset.Ico (i+1) n, if x j - x i ≥ 1 then 2^(i-j) else 0) ≤ M := sorry

end smallest_M_exists_l39_39852


namespace parallel_vectors_y_value_l39_39192

theorem parallel_vectors_y_value (y : ℝ) : 
  let a := (2, 4, 5) in
  let b := (3, 6, y) in
  (∃ k : ℝ, b = k • a) → y = 7.5 :=
by
  sorry

end parallel_vectors_y_value_l39_39192


namespace width_of_grass_field_l39_39045

theorem width_of_grass_field (length : ℕ) (path_width : ℕ) (cost_per_sqm : ℕ) (total_cost : ℕ) : 
  ∃ (w : ℕ), w = 55 :=
by
  -- Given conditions
  assume h_length : length = 75,
  assume h_path_width : path_width = 2.5,
  assume h_cost_per_sqm : cost_per_sqm = 2,
  assume h_total_cost : total_cost = 1350,
  -- Placeholder for proof
  sorry

end width_of_grass_field_l39_39045


namespace richard_twice_scott_years_l39_39769

-- Definitions of the conditions given in the problem.
def richard_age (david_age : ℕ) : ℕ := david_age + 6
def scott_age (david_age : ℕ) : ℕ := david_age - 8
def david_age_current : ℕ := 8 + 6

-- The theorem we need to prove: in how many years will Richard be twice as old as Scott?
theorem richard_twice_scott_years
  (david_age : ℕ)
  (richard_age : ℕ := richard_age david_age)
  (scott_age : ℕ := scott_age david_age)
  (david_current : david_age = david_age_current := rfl) :
  ∃ (x : ℕ), (richard_age + x = 2 * (scott_age + x)) ∧ x = 8 :=
by
  sorry

end richard_twice_scott_years_l39_39769


namespace largest_prime_divisor_of_factorial_sum_l39_39110

theorem largest_prime_divisor_of_factorial_sum {n : ℕ} (h1 : n = 13) : 
  Nat.gcd (Nat.factorial 13) 15 = 1 ∧ Nat.gcd (Nat.factorial 13 * 15) 13 = 13 :=
by
  sorry

end largest_prime_divisor_of_factorial_sum_l39_39110


namespace problem_angle_between_vectors_is_90_degrees_l39_39573

def vector_angle_is_90_degrees (θ : ℝ) : Prop :=
  let a : ℝ × ℝ × ℝ := (Real.cos θ, 1, Real.sin θ)
  let b : ℝ × ℝ × ℝ := (Real.sin θ, 1, Real.cos θ)
  let ab_sum : ℝ × ℝ × ℝ := (a.1 + b.1, a.2 + b.2, a.3 + b.3)
  let ab_diff : ℝ × ℝ × ℝ := (a.1 - b.1, a.2 - b.2, a.3 - b.3)
  let dot_product := ab_sum.1 * ab_diff.1 + ab_sum.2 * ab_diff.2 + ab_sum.3 * ab_diff.3
  dot_product = 0

theorem problem_angle_between_vectors_is_90_degrees (θ : ℝ) : vector_angle_is_90_degrees θ := by
  -- implementation proof here
  sorry

end problem_angle_between_vectors_is_90_degrees_l39_39573


namespace number_of_correct_relations_l39_39705

def isReal (x : ℝ) : Prop := True
def isIrrational (x : ℝ) : Prop := ∀ q : ℚ, x ≠ q
def isPositiveNat (x : ℕ) : Prop := x > 0

theorem number_of_correct_relations :
  (isReal π) ∧ (isIrrational (real.sqrt 3)) ∧ ¬ (isPositiveNat 0) ∧ isPositiveNat (abs (-4)) → 2 = 2 :=
by
  intro _
  apply rfl

end number_of_correct_relations_l39_39705


namespace seating_arrangement_fixed_pairs_l39_39609

theorem seating_arrangement_fixed_pairs 
  (total_chairs : ℕ) 
  (total_people : ℕ) 
  (specific_pair_adjacent : Prop)
  (comb : ℕ) 
  (four_factorial : ℕ) 
  (two_factorial : ℕ) 
  : total_chairs = 6 → total_people = 5 → specific_pair_adjacent → comb = Nat.choose 6 4 → 
    four_factorial = Nat.factorial 4 → two_factorial = Nat.factorial 2 → 
    Nat.choose 6 4 * Nat.factorial 4 * Nat.factorial 2 = 720 
  := by
  intros
  sorry

end seating_arrangement_fixed_pairs_l39_39609


namespace rectangle_area_l39_39338

theorem rectangle_area (x y : ℝ) (hx : 3 * y = 7 * x) (hp : 2 * (x + y) = 40) :
  x * y = 84 := by
  sorry

end rectangle_area_l39_39338


namespace parallelogram_contains_points_l39_39251

theorem parallelogram_contains_points (P : Set (ℝ × ℝ))
  (hP : ∃ a b c d : ℝ × ℝ, P = { t • a + u • b | t u : ℝ })
  (area_P : P.area = 1990) :
  ∃ p1 p2 ∈ P, p1 ≠ p2 ∧ (∃ (x1 y1 x2 y2 : ℤ), 
    p1 = (41 * x1 + 2 * y1, 59 * x1 + 15 * y1) ∧ 
    p2 = (41 * x2 + 2 * y2, 59 * x2 + 15 * y2)) :=
sorry

end parallelogram_contains_points_l39_39251


namespace at_least_one_angle_not_greater_than_60_l39_39360

def triangle (α β γ : ℝ) := α + β + γ = 180 ∧ α > 0 ∧ β > 0 ∧ γ > 0

theorem at_least_one_angle_not_greater_than_60 (α β γ : ℝ) (h : triangle α β γ) :
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60 :=
begin
  by_contra h_contra,
  have h1 : α > 60 ∧ β > 60 ∧ γ > 60 := by exact h_contra,
  sorry
end

end at_least_one_angle_not_greater_than_60_l39_39360


namespace positive_divisors_60_l39_39991

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l39_39991


namespace probability_of_sequence_l39_39763

namespace MarbleBagProblem

def initial_bag := {total := 12, red := 4, white := 6, blue := 2}

def first_draw_red_event (bag : initial_bag) : Prop :=
  4 / 12 = 1 / 3

def second_draw_white_event (bag : initial_bag) (first_red_drawn : Prop) : Prop :=
  6 / 11 = 6 / 11

def third_draw_blue_event (bag : initial_bag) (first_red_drawn second_white_drawn : Prop) : Prop :=
  2 / 10 = 1 / 5

theorem probability_of_sequence :
  first_draw_red_event initial_bag ∧
  second_draw_white_event initial_bag (first_draw_red_event initial_bag) ∧
  third_draw_blue_event initial_bag (first_draw_red_event initial_bag) (second_draw_white_event initial_bag (first_draw_red_event initial_bag)) →
  ∃ p : ℚ, p = 2 / 55 :=
by
  intros h
  sorry

end MarbleBagProblem

end probability_of_sequence_l39_39763


namespace transformed_center_l39_39818

def initial_point := (-2, 6)

def reflected_point (p : Int × Int) : Int × Int :=
  (p.1, -p.2)

def translated_point (p : Int × Int) : Int × Int :=
  (p.1 + 5, p.2)

theorem transformed_center :
  translated_point (reflected_point initial_point) = (3, -6) := by
  sorry

end transformed_center_l39_39818


namespace gcd_lcm_product_l39_39119

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end gcd_lcm_product_l39_39119


namespace find_first_remainder_l39_39870

theorem find_first_remainder (N : ℕ) (R₁ R₂ : ℕ) (h1 : N = 184) (h2 : N % 15 = R₂) (h3 : R₂ = 4) : 
  N % 13 = 2 :=
by
  sorry

end find_first_remainder_l39_39870


namespace prove_proposition_false_l39_39534

def proposition (a : ℝ) := ∃ x : ℝ, x^2 - 4*a*x + 3 < 0

theorem prove_proposition_false : proposition 0 = False :=
by
sorry

end prove_proposition_false_l39_39534


namespace constant_term_of_p_is_neg_two_sevenths_l39_39310

noncomputable def p (x : ℝ) := (5 * x^2 - 2) / 7

theorem constant_term_of_p_is_neg_two_sevenths :
  ∀ (x : ℝ), p(x) = (5 / 7) * x^2 - (2 / 7) → ∃ c : ℝ, c = - (2 / 7) := by
sorry

end constant_term_of_p_is_neg_two_sevenths_l39_39310


namespace linear_combination_solution_l39_39083

theorem linear_combination_solution :
  ∃ a b c : ℚ, 
    a • (⟨1, -2, 3⟩ : ℚ × ℚ × ℚ) + b • (⟨4, 1, -1⟩ : ℚ × ℚ × ℚ) + c • (⟨-3, 2, 1⟩ : ℚ × ℚ × ℚ) = ⟨0, 1, 4⟩ ∧
    a = -491/342 ∧
    b = 233/342 ∧
    c = 49/38 :=
by
  sorry

end linear_combination_solution_l39_39083


namespace max_length_OB_is_sqrt2_l39_39347

noncomputable def max_length_OB : ℝ :=
  let θ : ℝ := real.pi / 4 in
  let AB : ℝ := 1 in
  max (AB / real.sin θ * real.sin (real.pi / 2))

theorem max_length_OB_is_sqrt2 : max_length_OB = real.sqrt 2 := 
sorry

end max_length_OB_is_sqrt2_l39_39347


namespace number_of_positive_divisors_of_60_l39_39967

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l39_39967


namespace floor_sqrt_sum_l39_39464

theorem floor_sqrt_sum : (∑ x in Finset.range 25 + 1, (Nat.floor (Real.sqrt x : ℝ))) = 75 := 
begin
  sorry
end

end floor_sqrt_sum_l39_39464


namespace sum_fraction_series_eq_l39_39838

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l39_39838


namespace midpoint_of_PQ_is_incenter_l39_39230

-- Define the isosceles triangle with AB = AC
variables {A B C P Q : Point}
variables {AB AC : Real}
variables {K : Circle}
variables [incircle_tangent_cond : InscribedInCircle K (Triangle A B C) P Q]
variables [isosceles_cond : IsIsoscelesTriangle (Triangle A B C) A B AC]

-- Definition of midpoint of segment PQ being the incenter of triangle ABC
def midpoint_is_incenter (ABC : Triangle) (PQ : Segment) : Point :=
  incenter ABC

-- Theorem statement
theorem midpoint_of_PQ_is_incenter
  (hiso : isosceles_cond)
  (hincircle : incircle_tangent_cond) :
  midpoint_is_incenter (Triangle A B C) (Segment P Q) 
  = incenter (Triangle A B C) := by 
  sorry

end midpoint_of_PQ_is_incenter_l39_39230


namespace ratio_of_areas_of_inscribed_squares_l39_39368

theorem ratio_of_areas_of_inscribed_squares (R : ℝ) (hR : R > 0) :
  let s1 := sqrt (4 * R^2 / 5),
      s2 := sqrt (4 * (2 * R)^2) in
  (s1^2 / s2^2) = 1 / 5 :=
by
  let s1 := sqrt (4 * R^2 / 5)
  let s2 := sqrt (4 * (2 * R)^2)
  have : s1^2 = 4 * R^2 / 5 := by
    sorry
  have : s2^2 = 4 * (2 * R)^2 := by
    sorry
  show (s1^2 / s2^2) = 1 / 5
  sorry

end ratio_of_areas_of_inscribed_squares_l39_39368


namespace max_n_odd_group_l39_39052

def is_odd_power_prime_factors (n : ℕ) : Prop :=
  ∀ p k : ℕ, nat.prime p → (n = p ^ k) → odd k

def is_n_odd_group (seq : list ℕ) : Prop :=
  seq.length = seq.last' - seq.head' + 1 ∧ (∀ x ∈ seq, is_odd_power_prime_factors x)

theorem max_n_odd_group :
  ∀ seq : list ℕ, is_n_odd_group seq → seq.length ≤ 7 :=
begin
  sorry
end

end max_n_odd_group_l39_39052


namespace eccentricity_range_l39_39905

theorem eccentricity_range (a b : ℝ) (h₁ : a > b) (h₂ : b > 0)
  (h₃ : ∃ (P : ℝ × ℝ), ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
        ((x - a/2)^2 + y^2 = (a/2)^2)) :
  ∃ e : ℝ, a^2 = b^2 + (e * a)^2 ∧ (sqrt 2) / 2 < e ∧ e < 1 :=
sorry

end eccentricity_range_l39_39905


namespace number_of_true_propositions_l39_39127

def manhattan_distance (A B : ℝ × ℝ) : ℝ :=
  |B.1 - A.1| + |B.2 - A.2|

def on_segment (A B C : ℝ × ℝ) : Prop :=
  A.1 ≤ C.1 ∧ C.1 ≤ B.1 ∧ A.2 ≤ C.2 ∧ C.2 ≤ B.2

theorem number_of_true_propositions (A B C : ℝ × ℝ) :
  let d := manhattan_distance in
  ((on_segment A B C → d A C + d C B = d A B) ∧
   (C.2 ^ 2 + C.1 ^ 2 = (B.2 - A.2) ^ 2 + (B.1 - A.1) ^ 2 → d A C ^ 2 + d C B ^ 2 = d A B ^ 2) ∧
   (d A C + d C B > d A B → false)) →
  1 := sorry

end number_of_true_propositions_l39_39127


namespace prove_q_and_sn_l39_39903

noncomputable def arithmetic_sequence {a : ℕ → ℝ} (q : ℝ) :=
∀ n, a (n + 1) = a n * q

noncomputable def specific_conditions (a : ℕ → ℝ) :=
2 * a 1 + a 3 = 3 * a 2 ∧
(a 3 + 2) = (a 2 + a 4) / 2

noncomputable def b_seq_sum_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range(n), a i * real.log (a i) / real.log 2

theorem prove_q_and_sn :
∀ {a : ℕ → ℝ} (q : ℝ) (n : ℕ),
  arithmetic_sequence q a →
  specific_conditions a →
  q = 2 ∧ b_seq_sum_s_n a n = 2 + (n - 1) * 2^(n+1) :=
by
  intros a q n h_arith h_spec
  sorry

end prove_q_and_sn_l39_39903


namespace percentage_increase_surface_area_40_percent_growth_l39_39753

variable (L : ℝ) (hL : L > 0)

def original_surface_area := 6 * (L ^ 2)
def new_edge_length := 1.40 * L
def new_surface_area := 6 * (new_edge_length ^ 2)
def percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100

theorem percentage_increase_surface_area_40_percent_growth :
  percentage_increase L = 96 := by
  sorry

end percentage_increase_surface_area_40_percent_growth_l39_39753


namespace cone_water_volume_percentage_l39_39000

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39000


namespace roots_of_polynomial_l39_39123

theorem roots_of_polynomial : ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ x = 2 ∨ x = 3 ∨ x = -2 :=
by sorry

end roots_of_polynomial_l39_39123


namespace nine_pow_x_eq_243_l39_39585

theorem nine_pow_x_eq_243 (x : ℝ) (h : 9^x = 243) : x = 5 / 2 :=
by sorry

end nine_pow_x_eq_243_l39_39585


namespace hyperbolas_same_asymptote_l39_39853

variables (M : ℝ)

def hyperbola1_asymptote : ℝ := 4 / 3
def hyperbola2_asymptote (M : ℝ) : ℝ := 5 / Real.sqrt M

theorem hyperbolas_same_asymptote (M : ℝ) :
  hyperbola1_asymptote = hyperbola2_asymptote M →
  M = 144 / 25 :=
begin
  intro h,
  simp [hyperbola1_asymptote, hyperbola2_asymptote] at h,
  have h1 : Real.sqrt M = 12 / 5,
  { rw [eq_comm, ← mul_eq_mul_left_iff, ← Real.sqrt_mul_self_eq_abs, abs_of_pos, ← h, mul_comm],
    norm_num,
    linarith, },
  exact (Real.sqrt_eq_iff_sq_eq ⟨0, by norm_num⟩).1 h1,
  linarith,
end

end hyperbolas_same_asymptote_l39_39853


namespace billy_books_read_l39_39814

def hours_per_day : ℕ := 8
def days_per_weekend : ℕ := 2
def reading_percentage : ℚ := 0.25
def pages_per_hour : ℕ := 60
def pages_per_book : ℕ := 80

theorem billy_books_read :
  let total_hours := hours_per_day * days_per_weekend in
  let reading_hours := total_hours * reading_percentage in
  let total_pages := reading_hours * pages_per_hour in
  let books_read := total_pages / pages_per_book in
  books_read = 3 :=
by
  sorry

end billy_books_read_l39_39814


namespace farthest_distance_traveled_total_fuel_consumption_proved_l39_39791

-- Define the patrol record as a list of integers
def patrol_record : List Int := [15, -3, 14, -11, 10, 4, -26]

-- Define the absolute values of the patrol record
def abs_patrol_record : List Int := patrol_record.map Int.natAbs

-- Define the fuel consumption rate
def fuel_consumption_rate : Float := 0.1

-- Define the total distance traveled as the sum of the absolute values of the patrol record
def total_distance_traveled : Nat := abs_patrol_record.foldl Nat.add 0

-- Define the total fuel consumption
noncomputable def total_fuel_consumption : Float := fuel_consumption_rate * total_distance_traveled

-- Proof statement: The segment where the officer traveled the farthest distance in the given patrol record
theorem farthest_distance_traveled : (max (List.map Int.natAbs patrol_record)) = 26 := by
  sorry

-- Proof statement: The total fuel consumption for the given patrol record and fuel consumption rate
theorem total_fuel_consumption_proved : total_fuel_consumption = 8.3 := by
  sorry

end farthest_distance_traveled_total_fuel_consumption_proved_l39_39791


namespace gcd_of_462_and_330_l39_39365

theorem gcd_of_462_and_330 :
  Nat.gcd 462 330 = 66 :=
sorry

end gcd_of_462_and_330_l39_39365


namespace find_x_l39_39748

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the proof goal
theorem find_x (x : ℝ) : 2 * f x - 19 = f (x - 4) → x = 4 :=
by
  sorry

end find_x_l39_39748


namespace problem1_problem2_problem3_l39_39214

def a_n1 (n : ℕ) : ℕ → ℝ := λ n, -2 * n + 19

theorem problem1 : ∀ n, 1 ≤ n ∧ n ≤ 100 → inversion_number (list.map a_n1 (list.range 100)) = 4950 :=
by sorry

def a_n2 (k : ℕ) : ℕ → ℝ := λ n, if n % 2 = 1 then (1/3) ^ n else -n / (n + 1)

theorem problem2 : ∀ k, 1 ≤ k →
  (if k % 2 = 1 then inversion_number (list.map a_n2 (list.range k)) = (3 * k^2 - 4 * k + 1) / 8 
  else inversion_number (list.map a_n2 (list.range k)) = (3 * k^2 - 2 * k) / 8) :=
by sorry

variable {a : ℕ}

theorem problem3 (n : ℕ) : inversion_number (list.range n) = a →
  inversion_number (list.reverse (list.range n)) = n * (n - 1) / 2 - a :=
by sorry

end problem1_problem2_problem3_l39_39214


namespace right_triangle_infinite_non_congruent_l39_39577

theorem right_triangle_infinite_non_congruent (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : a + b + Math.sqrt (a^2 + b^2) = (1/2) * a * b) :
    ∃ (a b : ℝ), a ≠ 4 ∧ (a + b + Math.sqrt (a^2 + b^2) = (1/2) * a * b) ∧ (a^2 + b^2 ≥ 0) :=
sorry

end right_triangle_infinite_non_congruent_l39_39577


namespace max_min_values_on_interval_l39_39519

noncomputable def f (x : ℝ) : ℝ := 4 * x - x ^ 3

theorem max_min_values_on_interval :
  (∀ x ∈ Icc (0 : ℝ) 2, f x ≤ f (2 * Real.sqrt 3 / 3)) ∧ (f 0 = 0) ∧ (f 2 = 0) :=
by
  sorry

end max_min_values_on_interval_l39_39519


namespace num_five_digit_numbers_l39_39960

def valid_middle_triplets : List (ℕ × ℕ × ℕ) :=
  [(1,1,1), (1,1,2), (1,2,1), (2,1,1), (1,1,3), (1,3,1), (3,1,1), 
   (1,1,4), (1,4,1), (4,1,1), (1,2,2), (2,1,2), (2,2,1), 
   (1,1,5), (1,5,1), (5,1,1), (1,1,6), (1,6,1), (6,1,1), 
   (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1), 
   (1,1,7), (1,7,1), (7,1,1), (1,1,8), (1,8,1), (8,1,1), 
   (1,2,4), (1,4,2), (2,1,4), (2,4,1), (4,1,2), (4,2,1), 
   (1,1,9), (1,9,1), (9,1,1), (1,3,3), (3,1,3), (3,3,1)]

theorem num_five_digit_numbers : 
  let first_digit_choices := [4, 5, 6, 7, 8, 9]
      last_digit_choices := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      middle_triplet_count := valid_middle_triplets.length in
  2580 = first_digit_choices.length * middle_triplet_count * last_digit_choices.length :=
by
  unfold valid_middle_triplets
  sorry

end num_five_digit_numbers_l39_39960


namespace percent_volume_filled_with_water_l39_39010

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39010


namespace num_divisors_sixty_l39_39983

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l39_39983


namespace largest_prime_divisor_13_plus_14_fact_l39_39113

theorem largest_prime_divisor_13_plus_14_fact : 
  ∀ p : ℕ, prime p ∧ p ∣ 13! + 14! → p ≤ 13 := 
sorry

end largest_prime_divisor_13_plus_14_fact_l39_39113


namespace trigonometric_identity_l39_39835

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l39_39835


namespace triangle_BPQ_perimeter_greater_one_l39_39319

variable {A B C P Q : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q]
variables [dist : MetricSpace.dist A B C P Q]
variables (AP AB QC BC : ℝ)

theorem triangle_BPQ_perimeter_greater_one
  (h1 : ∀ (a b c : ℝ), a + b + c = 2)
  (h2 : ∀ (AP AB : ℝ), 2 * AP = AB)
  (h3 : ∀ (QC BC : ℝ), 2 * QC = BC)
  : ∑ BP BQ PQ > 1 :=
by
  sorry

end triangle_BPQ_perimeter_greater_one_l39_39319


namespace trip_length_l39_39679

theorem trip_length (d : ℕ) 
  (h_battery_miles : d ≥ 60)
  (h_gas_consumption : ∀ (x : ℕ), x = d - 60 → 0.03 * x = 0.03 * (d - 60))
  (h_avg_mileage : 50 = d / (0.03 * (d - 60))) :
  d = 180 :=
sorry

end trip_length_l39_39679


namespace sophia_ate_pie_l39_39301

theorem sophia_ate_pie (weight_fridge weight_total weight_ate : ℕ)
  (h1 : weight_fridge = 1200) 
  (h2 : weight_fridge = 5 * weight_total / 6) :
  weight_ate = weight_total / 6 :=
by
  have weight_total_formula : weight_total = 6 * weight_fridge / 5 := by
    sorry
  have weight_ate_formula : weight_ate = weight_total / 6 := by
    sorry
  sorry

end sophia_ate_pie_l39_39301


namespace center_of_circle_l39_39695

theorem center_of_circle : ∃ c : ℝ × ℝ, 
  (∃ r : ℝ, ∀ x y : ℝ, (x - c.1) * (x - c.1) + (y - c.2) * (y - c.2) = r ↔ x^2 + y^2 - 6*x - 2*y - 15 = 0) → c = (3, 1) :=
by 
  sorry

end center_of_circle_l39_39695


namespace find_C_l39_39381

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 :=
by
  sorry

end find_C_l39_39381


namespace problem_one_problem_two_l39_39221

variables (a₁ a₂ a₃ : ℤ) (n : ℕ)
def arith_sequence : Prop :=
  a₁ + a₂ + a₃ = 21 ∧ a₁ * a₂ * a₃ = 231

theorem problem_one (h : arith_sequence a₁ a₂ a₃) : a₂ = 7 :=
sorry

theorem problem_two (h : arith_sequence a₁ a₂ a₃) :
  (∃ d : ℤ, (d = -4 ∨ d = 4) ∧ (a_n = a₁ + (n - 1) * d ∨ a_n = a₃ + (n - 1) * d)) :=
sorry

end problem_one_problem_two_l39_39221


namespace janet_tulips_l39_39631

theorem janet_tulips (T : ℕ) (roses : ℕ) (used : ℕ) (extra : ℕ) (total_flowers : ℕ) :
  (roses = 11) →
  (used = 11) →
  (extra = 4) →
  (total_flowers = T + roses) →
  (total_flowers - used = extra) →
  T = 4 :=
by
  intros hroses hused hextra htotal hleftover
  rw hroses at htotal
  rw htotal at hleftover
  rw hused at hleftover
  simp at hleftover
  assumption

end janet_tulips_l39_39631


namespace sum_of_decimals_is_fraction_l39_39508

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l39_39508


namespace determinant_evaluation_l39_39493

theorem determinant_evaluation (x z : ℝ) :
  (Matrix.det ![
    ![1, x, z],
    ![1, x + z, z],
    ![1, x, x + z]
  ]) = x * z - z * z := 
sorry

end determinant_evaluation_l39_39493


namespace floor_sqrt_sum_l39_39467

open Real

theorem floor_sqrt_sum : (∑ k in Finset.range 25, ⌊sqrt (k + 1)⌋) = 75 := 
by
  sorry

end floor_sqrt_sum_l39_39467


namespace find_x_l39_39256

def exp (m n : ℤ) : ℤ := m ^ n

theorem find_x (x : ℤ) (h : exp 10 x = 25 * exp 2 2) : x = 2 :=
by 
  -- problem setup
  have h1 : exp 2 2 = 4 := by simp [exp]
  -- substituting exp values into equation
  have h2 : exp 10 x = 25 * 4 := by rw [h1] at h
  rw [←h2, exp, mul_comm 25 4] at h
  sorry

end find_x_l39_39256


namespace arithmetic_sequence_first_term_range_l39_39902

theorem arithmetic_sequence_first_term_range (a_1 : ℝ) (d : ℝ) (a_10 : ℝ) (a_11 : ℝ) :
  d = (Real.pi / 8) → 
  (a_1 + 9 * d ≤ 0) → 
  (a_1 + 10 * d ≥ 0) → 
  - (5 * Real.pi / 4) ≤ a_1 ∧ a_1 ≤ - (9 * Real.pi / 8) :=
by
  sorry

end arithmetic_sequence_first_term_range_l39_39902


namespace find_difference_of_max_and_min_values_l39_39727

noncomputable def v (a b : Int) : Int := a * (-4) + b

theorem find_difference_of_max_and_min_values :
  let v0 := 3
  let v1 := v v0 12
  let v2 := v v1 6
  let v3 := v v2 10
  let v4 := v v3 (-8)
  (max (max (max (max v0 v1) v2) v3) v4) - (min (min (min (min v0 v1) v2) v3) v4) = 62 :=
by
  sorry

end find_difference_of_max_and_min_values_l39_39727


namespace angle_in_second_quadrant_l39_39139

theorem angle_in_second_quadrant (x : ℝ) (hx1 : Real.tan x < 0) (hx2 : Real.sin x - Real.cos x > 0) : 
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 2 ∨ x = 2 * k * Real.pi + 3 * Real.pi / 2) :=
sorry

end angle_in_second_quadrant_l39_39139


namespace find_sum_reciprocal_a_n_l39_39146

noncomputable def a : Nat → ℕ 
| 1 => 1
| n => if h : 1 ≤ n then 
         let ⟨m, k, hmk⟩ := nat.exists_eq_add_of_le h in
         a m + a k + m * k
       else 
         0  -- Just a base case to satisfy Lean, won't be triggered since n > 1

theorem find_sum_reciprocal_a_n :
  (∑ i in finset.range 2017, (1 : ℚ) / a (i + 1)) = 4034 / 2018 := 
sorry

end find_sum_reciprocal_a_n_l39_39146


namespace stability_of_2k_times_n_is_2k_2008_stable_l39_39785

def is_stable (N : ℕ) : Prop :=
  ∃ (A B : finset ℕ), A ∩ B = ∅ ∧ A ∪ B = (finset.range (N + 1)).filter (λ d, N % d = 0) ∧ A.sum id = B.sum id

theorem stability_of_2k_times_n (k : ℕ) (h_n : ℕ -> ℕ):
  is_stable (2^k * h_n) := 
sorry

theorem is_2k_2008_stable : is_stable (2^2008 * 2008) :=
stability_of_2k_times_n 2008 (λ n, 2008)

end stability_of_2k_times_n_is_2k_2008_stable_l39_39785


namespace intersection_of_M_and_N_l39_39572

open Set Real

def M := {y : ℝ | ∃ x : ℝ, y = 2^x}
def N := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem intersection_of_M_and_N :
  M ∩ N = (λ y, 0 < y) '' Ioo 0 (1/0) :=
by
  sorry

end intersection_of_M_and_N_l39_39572


namespace postage_stamp_problem_l39_39873

theorem postage_stamp_problem :
  ∃! (n : ℕ), (∃ (a b c : ℕ), 9 * a + n * b + (n + 2) * c = 120 = false) ∧
              (∀ k ∈ [121, 122, 123, 124, 125], ∃ (a b c : ℕ), 9 * a + n * b + (n + 2) * c = k) ∧
              (Σ' (sol : ℕ), sol = n) = 17 := 
sorry

end postage_stamp_problem_l39_39873


namespace solution_to_problem_l39_39164

def f (x : ℝ) : ℝ := sorry

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem solution_to_problem
  (f : ℝ → ℝ)
  (h : functional_equation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end solution_to_problem_l39_39164


namespace correct_system_of_equations_l39_39220

theorem correct_system_of_equations (x y : ℝ) :
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔
  (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by sorry

end correct_system_of_equations_l39_39220


namespace sum_of_floor_sqrt_l39_39461

theorem sum_of_floor_sqrt :
  (∑ i in Finset.range 25, (Nat.sqrt (i + 1))) = 75 := by
  sorry

end sum_of_floor_sqrt_l39_39461


namespace smallest_N_l39_39357

theorem smallest_N (n : ℕ) (h1 : n > 0) (h2 : n ≤ 2023) 
    (id1 id2 : ℕ) (h3 : id1 ≠ id2) (h4 : 1 ≤ id1) (h5 : id1 ≤ n) (h6 : 1 ≤ id2) (h7 : id2 ≤ n) :
    ∃ N, (∀ IDs : finset ℕ, (∀ id ∈ IDs, 1 ≤ id ∧ id ≤ n) → ∃ t : ℕ, (t ≤ N) ∧ ∀ (strategy : ℕ → ℕ) (id_1 id_2 : ℕ), (id_1 ∈ IDs) → (id_2 ∈ IDs) → id_1 ≠ id_2 → (strategy id_1 = 0 ∨ strategy id_2 = 0 ∨ (strategy id_1 ≤ t ∧ strategy id_2 ≤ t))
        ∧ (strategy id_1 > strategy id_2 → id_1 > id_2) ∧ (strategy id_1 < strategy id_2 → id_2 > id_1)) ∧ N = nat.log2 n - 1 :=
begin
  sorry
end

end smallest_N_l39_39357


namespace circle_area_l39_39597

theorem circle_area (r : ℝ) (h : 2 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 2 := 
by 
  sorry

end circle_area_l39_39597


namespace hexagon_area_eq_l39_39788

theorem hexagon_area_eq (s t : ℝ) (hs : s^2 = 16) (heq : 4 * s = 6 * t) :
  6 * (t^2 * (Real.sqrt 3) / 4) = 32 * (Real.sqrt 3) / 3 := by
  sorry

end hexagon_area_eq_l39_39788


namespace ratio_CP_PA_l39_39622

-- Definitions for the problem conditions
variables {A B C D M P : Type} [MetricSpace A B C D M P]

-- Assumptions based on the problem conditions
variables 
  (hAB : Distance A B = 30) 
  (hAC : Distance A C = 18)
  (angle_bisector_A : IsAngleBisector A B C D) 
  (midpoint_M : M = Midpoint A D)
  (intersection_P : P = Intersection AC BM)

-- The goal: Prove that the ratio CP/PA is 3/5 and thus the sum m + n = 8
theorem ratio_CP_PA (hAB : Distance A B = 30) (hAC : Distance A C = 18)
  (angle_bisector_A : IsAngleBisector A B C D) (midpoint_M : M = Midpoint A D)
  (intersection_P : P = Intersection AC BM) :
  Ratio CP PA = 3 / 5 ∧ GCD 3 5 = 1 → 3 + 5 = 8 :=
by sorry

end ratio_CP_PA_l39_39622


namespace bacteria_colonies_simultaneous_growth_l39_39767

theorem bacteria_colonies_simultaneous_growth 
  (doub: ∀ (n : ℕ), n < 25 → (∃ (growth : ℕ), growth = 2 ^ n)) :
  (∀ (N : ℕ), N * 2^24 = 2^25 → N = 2) :=
by
  intros N h
  have h1 : 2^25 = 2 * 2^24 := by norm_num
  rwa [h1] at h
  sorry

end bacteria_colonies_simultaneous_growth_l39_39767


namespace cosine_of_angle_between_a_b_l39_39191

variable (a b : ℝ × ℝ)

def a_add_b_eq : a + b = (2, -8) := sorry
def a_sub_b_eq : a - b = (-8, 16) := sorry

theorem cosine_of_angle_between_a_b :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = -63 / 65 :=
  by
  sorry

end cosine_of_angle_between_a_b_l39_39191


namespace derivative_of_f_l39_39564

noncomputable def f (x : ℝ) : ℝ := x / exp x

theorem derivative_of_f : deriv f x = (1 - x) / exp x := by
  sorry

end derivative_of_f_l39_39564


namespace area_section_plane_A1D_M_l39_39540

-- Initial conditions and definitions
variables (AB AD AA1 : ℝ)
variables (M : (ℝ × ℝ × ℝ))

axiom parallelepiped_dimensions : AB = 4 ∧ AD = 14 ∧ AA1 = 14 
axiom midpoint_M : M = ((CC + C1) / 2)

theorem area_section_plane_A1D_M : 
  AB = 4 ∧ AD = 14 ∧ AA1 = 14 → M = ((CC + C1) / 2) → 
  let A1 : ℝ := 42 in sorry

end area_section_plane_A1D_M_l39_39540


namespace men_in_second_group_l39_39201

variables (M B : ℝ) (x : ℝ)

-- Define conditions
def condition1 : Prop := ∀ (M B : ℝ), (12 * M + 16 * B) * 5 = (x * M + 24 * B) * 4
def condition2 : Prop := M = 2 * B

-- Prove the statement
theorem men_in_second_group :
  (condition1 M B) → (condition2 M B) → x = 13 :=
by
  intros h1 h2
  -- Proof steps skipped, as they are not required
  sorry

end men_in_second_group_l39_39201


namespace find_x_l39_39138

def delta (x : ℝ) : ℝ := 4 * x + 9
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x (x : ℝ) (h : delta (phi x) = 10) : x = -23 / 36 := 
by 
  sorry

end find_x_l39_39138


namespace prime_sum_of_squares_l39_39280

theorem prime_sum_of_squares (p : ℕ) (k : ℕ) (x y : ℤ) :
  prime p → p = 4 * k + 1 →
  (∃ x y : ℤ, x^2 + y^2 = p) :=
begin
  assume hp prime_property,
  -- The proof goes here
  sorry
end

end prime_sum_of_squares_l39_39280


namespace sum_to_common_fraction_l39_39502

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l39_39502


namespace range_of_k_l39_39250

variables {A B C O : Type} [AddGroup A] [InnerProductSpace ℝ A]
variables (AB AC AO : A)
variables (k : ℝ)

def circumcenter_condition (O A B C : A)
  (h1 : (AO = AB + k • AC)) : Prop :=
    ∃ (k : ℝ) (hk_pos : 0 < k), AO = AB + k • AC

theorem range_of_k
  (h : circumcenter_condition O A B C)
  (h1 : (O - A) • (B - A) = (1 / 2) * ∥B - A∥^2)
  (h2 : (O - A) • (C - A) = (1 / 2) * ∥C - A∥^2) :
  k > (1 / 2) :=
sorry

end range_of_k_l39_39250


namespace sum_pqr_l39_39819

/-- Define the circles with their centers and radii. -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

def C1 : Circle := ⟨(0, 0), 1⟩
def C2 : Circle := ⟨(12, 0), 2⟩
def C3 : Circle := ⟨(24, 0), 4⟩

/-- Define the properties of the tangents. -/
def tangent1_slope : ℝ := 1 / Real.sqrt 143
def tangent2_slope : ℝ := -2 / Real.sqrt 70

/-- Define the intersection point x in terms of p, q, and r. -/
def x := 19 - 3 * Real.sqrt 5

/-- Define the integers p, q, and r. -/
def p : ℝ := 19
def q : ℝ := 3
def r : ℝ := 5

/-- The main statement to be proved: p + q + r = 27. -/
theorem sum_pqr : p + q + r = 27 :=
by
  sorry

#lint

end sum_pqr_l39_39819


namespace find_AB_length_l39_39224

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
(t, -real.sqrt 3 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
(real.cos θ, 1 + real.sin θ)

noncomputable def curve_C2_polar (θ : ℝ) : ℝ :=
4 * real.sin (θ - real.pi / 6)

noncomputable def curve_C1_polar (θ : ℝ) : ℝ :=
2 * real.sin θ

noncomputable def curve_C2_rect (x y : ℝ) : Prop :=
(x + 1)^2 + (y - real.sqrt 3)^2 = 4

theorem find_AB_length :
  ∀ (t θ : ℝ),
    curve_C1_polar (2 * real.pi / 3) = real.sqrt 3 →
    curve_C2_polar (2 * real.pi / 3) = 4 →
    |4 - real.sqrt 3| = 4 - real.sqrt 3 :=
by
  intros t θ hC1 hC2
  sorry

end find_AB_length_l39_39224


namespace hockey_games_in_season_l39_39332

theorem hockey_games_in_season
  (games_per_month : ℤ)
  (months_in_season : ℤ)
  (h1 : games_per_month = 25)
  (h2 : months_in_season = 18) :
  games_per_month * months_in_season = 450 :=
by
  sorry

end hockey_games_in_season_l39_39332


namespace central_angle_radian_l39_39145

-- Define the radius of the sector
variable (R : ℝ)

-- Define the area of the sector
def area_sector (α : ℝ) : ℝ := (1 / 2) * α * R^2

theorem central_angle_radian {R : ℝ} (h : area_sector R 4 = 2 * R^2) : 4 = 4 :=
by
  sorry

end central_angle_radian_l39_39145


namespace distribute_positions_l39_39486

theorem distribute_positions : 
  ∃ (ways : ℕ), distribute 24 3 (λ n, n ≥ 1) (λ p1 p2 p3, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1) ways ∧ ways = 222 :=
sorry

end distribute_positions_l39_39486


namespace num_divisors_sixty_l39_39982

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l39_39982


namespace sum_of_floor_sqrt_l39_39456

theorem sum_of_floor_sqrt :
  (∑ n in Finset.range 26, Int.floor (Real.sqrt n)) = 75 :=
by
  -- skipping proof details
  sorry

end sum_of_floor_sqrt_l39_39456


namespace pond_area_percentage_l39_39789

theorem pond_area_percentage :
  let radius := 6
  let side := 16
  let A_pond := Real.pi * (radius ^ 2)
  let A_garden := side ^ 2
  let K := (A_pond / A_garden) * 100
  real.floor (K + 0.5) = 44 :=
by
  sorry

end pond_area_percentage_l39_39789


namespace quadratic_properties_l39_39920

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39920


namespace quadratic_properties_l39_39928

theorem quadratic_properties (a b c : ℝ) (h₀ : a < 0) (h₁ : a - b + c = 0) :
  (am² b c - 4 a) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ x1 x2 : ℝ, (∃ h : a * x1^2 + b * x1 + c + 1 = 0, ∃ h2 : a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39928


namespace inverse_of_A_matrix_X_satisfies_AX_eq_B_curve_transformation_l39_39908

variable (A : Matrix (Fin 2) (Fin 2) ℤ) (B : Matrix (Fin 2) (Fin 2) ℤ)
variable (x y x' y' : ℝ)

/- Given matrices -/
def A := ![![1, 2], ![-2, -3]]
def B := ![![0, 1], ![1, -2]]

/- Inverse of A -/
theorem inverse_of_A :
  (Matrix.det A ≠ 0) ∧ (A⁻¹ = ![![-3, -2], ![2, 1]]) :=
by
  sorry

/- Matrix X such that AX = B -/
theorem matrix_X_satisfies_AX_eq_B :
  let X := ![![-2, 1], ![1, 0]] in (A ⬝ X = B) :=
by
  sorry

/- Linear transformation and curve transformation -/
theorem curve_transformation :
  (x' = y) → (y' = x - 2*y) →
  let C_eq : x^2 - 4*x*y + y^2 = 1 in
  (3*(x')^2 - (y')^2 = -1) :=
by
  sorry

end inverse_of_A_matrix_X_satisfies_AX_eq_B_curve_transformation_l39_39908


namespace find_m_l39_39549

variable (m : ℝ)

def vector_oa : ℝ × ℝ := (-1, 2)
def vector_ob : ℝ × ℝ := (3, m)

def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_m
  (h : orthogonal (vector_oa) (vector_ob m)) :
  m = 3 / 2 := by
  sorry

end find_m_l39_39549


namespace trigonometric_identity_l39_39833

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l39_39833


namespace bijective_functions_to_real_eq_one_l39_39658

variable {ι : Type} [DecidableEq ι] [Fintype ι]

def is_bijective (f : ι → ι) : Prop :=
  Function.Bijective f

noncomputable def F (f : (ι → ι)) : ℝ := sorry

theorem bijective_functions_to_real_eq_one (F : (ι → ι) → ℝ) (h : ∀ (p q : ι → ι), is_bijective p → is_bijective q → 
  (F p + F q) ^ 2 = F (p ∘ p) + F (p ∘ q) + F (q ∘ p) + F (q ∘ q)) :
  ∀ (p : ι → ι) (hp : is_bijective p), F p = 1 := 
begin
  sorry
end

end bijective_functions_to_real_eq_one_l39_39658


namespace num_divisors_60_l39_39968

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l39_39968


namespace parametric_eq_of_line_cartesian_eq_of_circle_distance_center_to_line_l39_39897

noncomputable def line_parametric_eq (t : ℝ) : ℝ × ℝ :=
  (-1 + (4/5) * t, 1 + (3/5) * t)

def circle_cartesian_eq (x y : ℝ) : Prop :=
  (x - 1/2) ^ 2 + (y - 1/2) ^ 2 = 1/2

def distance_from_center_to_line : ℝ :=
  abs ((3 * (1/2) - 4 * (1/2) + 7) / real.sqrt (3^2 + (-4)^2))

theorem parametric_eq_of_line (t : ℝ) : 
  line_parametric_eq t = (-1 + (4/5) * t, 1 + (3/5) * t) :=
sorry

theorem cartesian_eq_of_circle (x y : ℝ) : 
  circle_cartesian_eq x y ↔ (x - 1/2) ^ 2 + (y - 1/2) ^ 2 = 1/2 :=
sorry

theorem distance_center_to_line :
  distance_from_center_to_line = 13/10 :=
sorry

end parametric_eq_of_line_cartesian_eq_of_circle_distance_center_to_line_l39_39897


namespace arrangement_books_l39_39430

theorem arrangement_books : 
  let n_geom := 4
  let n_numtheory := 5
  let gaps := n_numtheory + 1
  let comb := Nat.choose gaps n_geom
  gaps >= n_geom → comb = 15 :=
by
  intros
  let n_geom := 4
  let n_numtheory := 5
  let gaps := n_numtheory + 1
  let comb := Nat.choose gaps n_geom
  have : gaps = 6 := by rfl
  have : n_geom = 4 := by rfl
  rw [this]
  have : Nat.choose 6 4 = 15 := by sorry
  exact this


end arrangement_books_l39_39430


namespace sqrt_9_minus_1_eq_2_l39_39447

theorem sqrt_9_minus_1_eq_2 : Real.sqrt 9 - 1 = 2 := by
  sorry

end sqrt_9_minus_1_eq_2_l39_39447


namespace cos_product_pi_seventh_equals_neg_one_eighth_l39_39446

theorem cos_product_pi_seventh_equals_neg_one_eighth : 
  (cos (Real.pi / 7) * cos (2 * Real.pi / 7) * cos (4 * Real.pi / 7)) = -1 / 8 :=
by 
  sorry

end cos_product_pi_seventh_equals_neg_one_eighth_l39_39446


namespace parallel_vectors_mn_eq_neg3_l39_39190

theorem parallel_vectors_mn_eq_neg3 (m n : ℝ) 
    (h1 : ∃ (λ : ℝ), (2, m, 3) = (λ * 4, λ * -1, λ * n)) : 
    m * n = -3 := 
by
  sorry

end parallel_vectors_mn_eq_neg3_l39_39190


namespace gcd_lcm_product_l39_39116

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end gcd_lcm_product_l39_39116


namespace best_buy_order_ranking_order_l39_39408

noncomputable def cost_per_ounce (cost quantity : ℝ) : ℝ := cost / quantity

theorem best_buy_order (cS cM cL qS qM qL : ℝ) (h1 : cM = 1.5 * cS)
  (h2 : cL = 1.3 * cM) (h3 : qL = 2 * qS) (h4 : qM = 0.8 * qL) :
  cost_per_ounce cM qM < cost_per_ounce cS qS ∧ cost_per_ounce cS qS < cost_per_ounce cL qL :=
by
  sorry

-- Combine the logical statement to express the ranking order
theorem ranking_order : ∃ (cS cM cL qS qM qL : ℝ), 
  cM = 1.5 * cS ∧ cL = 1.3 * cM ∧ qL = 2 * qS ∧ qM = 0.8 * qL ∧ 
  cost_per_ounce cM qM < cost_per_ounce cS qS ∧ cost_per_ounce cS qS < cost_per_ounce cL qL :=
by
  use 1, 1.5, 1.95, 5, 8, 10
  split
  repeat { split,
    { exact rfl },
    sorry
  }
  
-- Sorry to skip part of proof steps

end best_buy_order_ranking_order_l39_39408


namespace problem_solution_l39_39878

def c_n (n : ℕ) : ℕ :=
  if h : ∃ k : ℕ, (n^k - 1) % 210 = 0 then
    Nat.find h
  else 0

theorem problem_solution : (Finset.range 210).sum c_n = 329 := 
by 
  sorry

end problem_solution_l39_39878


namespace riya_speed_l39_39290

theorem riya_speed (riya_speed : ℝ) (priya_speed : ℝ) (distance : ℝ) (time : ℝ) 
    (h1 : priya_speed = 35)
    (h2 : distance = 44.25)
    (h3 : time = 0.75) :
    riya_speed = 24 :=
by
  -- Using the conditions provided
  have relative_speed := riya_speed + priya_speed
  have distance_eq := relative_speed * time

  -- Now substitute values from h1, h2, and h3 into the equation
  rw [h1, h2, h3] at distance_eq

  -- Simplify the resulting equation to find riya_speed
  sorry

end riya_speed_l39_39290


namespace problem_statement_l39_39175

noncomputable def A : ℝ := 10
noncomputable def ω : ℝ := 2
noncomputable def φ : ℝ := π / 6

def f (x : ℝ) : ℝ := A * sin (ω * x + φ)
def g (x : ℝ) : ℝ := A * sin (ω * (x - π / 6))

theorem problem_statement :
  (∀ x : ℝ, f x = 10 * sin (2 * x + π / 6)) ∧ 
  (∀ k : ℤ, -π / 6 + k * π ≤ x ∧ x ≤ π / 3 + k * π → increasing_on g (set.Icc (-π / 6 + k * π) (π / 3 + k * π))) :=
sorry

end problem_statement_l39_39175


namespace density_ξ_eta_density_ξ_div_eta_l39_39243

-- Definitions of the densities and their properties
variable {Ξ : Type*} [MeasureSpace Ξ] [HasPDF Ξ]
variable {Η : Type*} [MeasureSpace Η] [HasPDF Η]

-- Define the densities
variable (f_ξ : ℝ → ℝ) (f_η : ℝ → ℝ)
variable (I_01 : ℝ → ℝ) := λ y, if 0 ≤ y ∧ y ≤ 1 then 1 else 0

-- Assumptions
variable (independent : Independent Ξ Η)
variable (ξ_density : ∀ x, f_ξ x ≥ 0)
variable (η_uniform : f_η = I_01)

-- Theorem statements
theorem density_ξ_eta (z : ℝ) :
  ∀ z, (∃ (f_Z : ℝ → ℝ), true) :=
  sorry

theorem density_ξ_div_eta (z : ℝ) :
  ∀ z, (∃ (f_W : ℝ → ℝ), true) :=
  sorry


end density_ξ_eta_density_ξ_div_eta_l39_39243


namespace missed_angle_l39_39043

theorem missed_angle (sum_calculated : ℕ) (missed_angle_target : ℕ) 
  (h1 : sum_calculated = 2843) 
  (h2 : missed_angle_target = 37) : 
  ∃ n : ℕ, (sum_calculated + missed_angle_target = n * 180) :=
by {
  sorry
}

end missed_angle_l39_39043


namespace triangle_third_side_l39_39668

variables (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
variables {AB AC BC : ℝ}
variables (angle_A angle_B angle_C : ℝ)

def triangle_satisfies_conditions : Prop :=
  (angle_B = 2 * angle_C) ∧ (AB = 9) ∧ (AC = 15)

theorem triangle_third_side (h : triangle_satisfies_conditions A B C) :
  BC = 16 :=
by 
  sorry

end triangle_third_side_l39_39668


namespace largest_prime_divisor_13_plus_14_fact_l39_39112

theorem largest_prime_divisor_13_plus_14_fact : 
  ∀ p : ℕ, prime p ∧ p ∣ 13! + 14! → p ≤ 13 := 
sorry

end largest_prime_divisor_13_plus_14_fact_l39_39112


namespace constant_term_expansion_l39_39614

theorem constant_term_expansion : 
  ∀ (x : Real), (∀ (r : ℕ), (r = 9) → 
  (binom 12 r * (-1)^r * (2^(12-r)) * (x^(12 - (4 * r/ 3))) = binom 12 9 * (-1)^9 * (2^3) * 1)) → 
  ( - (binom 12 9 * 8) = -1760 ) := 
by sorry


end constant_term_expansion_l39_39614


namespace sigma_has_large_prime_l39_39252

open Nat

theorem sigma_has_large_prime (k : ℕ) (h_k : 0 < k) : 
  let n := factorial (2^k) in 
  ∃ p : ℕ, p > 2^k ∧ p ∣ σ n :=
by
  sorry

end sigma_has_large_prime_l39_39252


namespace fraction_problem_l39_39375

theorem fraction_problem : 
  (  (1/4 - 1/5) / (1/3 - 1/4)  ) = 3/5 :=
by
  sorry

end fraction_problem_l39_39375


namespace pairs_of_integers_l39_39097

-- The main theorem to prove:
theorem pairs_of_integers (x y : ℤ) :
  y ^ 2 = x ^ 3 + 16 ↔ (x = 0 ∧ (y = 4 ∨ y = -4)) :=
by sorry

end pairs_of_integers_l39_39097


namespace intersection_A_B_l39_39185

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : 
  A ∩ B = {x | 0 < x ∧ x ≤ 2} :=
  sorry

end intersection_A_B_l39_39185


namespace find_a_l39_39909

theorem find_a (a : ℝ) : 
  let A := {0, 1, a^2}
  let B := {1, 0, 2*a + 3}
  (A ∩ B = A ∪ B) → a = 3 := 
by
  intros h
  -- Using h and the given conditions prove that a = 3
  sorry

end find_a_l39_39909


namespace smallest_number_div_by_11_with_zeros_and_ones_l39_39868

/-- Smallest Natural Number with 5 zeros and 7 ones divisible by 11. --/
theorem smallest_number_div_by_11_with_zeros_and_ones :
  ∃ n : ℕ, (nat.mod n 11 = 0) ∧ (string.length (nat.digits 10 n)) = 13 ∧ 
            (nat.countP (λ c, c = '0') (nat.digits 10 n) = 5) ∧
            (nat.countP (λ c, c = '1') (nat.digits 10 n) = 7) ∧
            n = 1000001111131 :=
sorry

end smallest_number_div_by_11_with_zeros_and_ones_l39_39868


namespace area_of_convex_quadrilateral_l39_39217

theorem area_of_convex_quadrilateral (W X Y Z : Type)
  (WZ XY YZ WY XZ : ℝ)
  (angle_YZX : ℝ)
  (h1 : WZ = 10)
  (h2 : XY = 6)
  (h3 : YZ = 8)
  (h4 : WY = 7)
  (h5 : XZ = 7)
  (h6 : angle_YZX = 45) :
  ∃ p q r : ℝ,
    (p = 773) ∧ 
    (q = 49) ∧ 
    (r = 2) ∧ 
    (∃ area : ℝ, area = sqrt p + (q * sqrt r) / 4) ∧ 
    (p+q+r = 824) := 
sorry

end area_of_convex_quadrilateral_l39_39217


namespace ratio_volume_second_largest_to_largest_l39_39049

theorem ratio_volume_second_largest_to_largest :
  ∀ (r h : ℝ),
  let V_E := (1 / 3) * π * (5 * r) ^ 2 * (5 * h),
      V_D := (1 / 3) * π * (4 * r) ^ 2 * (4 * h),
      V_C := (1 / 3) * π * (3 * r) ^ 2 * (3 * h) in
  (V_D - V_C) / (V_E - V_D) = 37 / 61 :=
by
  intros r h
  let V_E := (1 / 3) * π * (5 * r) ^ 2 * (5 * h)
  let V_D := (1 / 3) * π * (4 * r) ^ 2 * (4 * h)
  let V_C := (1 / 3) * π * (3 * r) ^ 2 * (3 * h)
  sorry

end ratio_volume_second_largest_to_largest_l39_39049


namespace positive_divisors_60_l39_39986

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l39_39986


namespace hypergeometric_event_l39_39438

theorem hypergeometric_event:
  ∀ (X₁ X₂ X₃ X₄ : ℕ) (H₁ : Binomial 3 0.5 X₁)
    (H₂ : Hypergeometric 10 3 5 X₂)   -- Condition (2)
    (H₃ : Bernoulli 0.8 X₃)
    (H₄ : NegativeHypergeometric 7 3 1 X₄),
    X₁ ≠ 2 ∧ X₃ ≠ 2 ∧ X₄ ≠ 2 ∧ X₂ = 2 :=
begin
  sorry
end

end hypergeometric_event_l39_39438


namespace sum_of_squares_eq_expansion_l39_39675

theorem sum_of_squares_eq_expansion (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 :=
sorry

end sum_of_squares_eq_expansion_l39_39675


namespace distinct_units_digits_mod_16_l39_39959

theorem distinct_units_digits_mod_16 : 
  let squares_mod_16 := finset.image (λ d : fin 16, (d * d) % 16) (finset.range 16) in
  finset.card squares_mod_16 = 4 :=
by {
  sorry
}

end distinct_units_digits_mod_16_l39_39959


namespace percent_volume_filled_with_water_l39_39015

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39015


namespace count_integer_points_in_circle_l39_39850

theorem count_integer_points_in_circle : 
  let circle := λ x y : ℝ, (x - 2)^2 + (y + 2)^2 ≤ 36
  let integer_point := λ x : ℤ, circle x (2 * x)
  1 = (Finset.filter integer_point (Finset.Icc (-6) 6)).card := 
sorry

end count_integer_points_in_circle_l39_39850


namespace smallest_number_with_5_zeros_7_ones_divisible_by_11_l39_39867

-- Define the set of all natural numbers with exactly 5 zeros and 7 ones.
def specific_number (n : ℕ) : Prop :=
  let digits := nat.digits 10 n in
  digits.count 0 = 5 ∧ digits.count 1 = 7

-- Define the divisibility rule for 11.
def divisible_by_11 (n : ℕ) : Prop :=
  (let digits := nat.digits 10 n in
  (digits.enum_from 0).sum (λ ⟨i, d⟩, if i % 2 = 0 then d else -d)) % 11 = 0

-- The main proof problem statement
theorem smallest_number_with_5_zeros_7_ones_divisible_by_11 :
  ∃ n : ℕ, specific_number n ∧ divisible_by_11 n ∧ (∀ m : ℕ, (specific_number m ∧ divisible_by_11 m) → n ≤ m) ∧ n = 1000001111131 :=
begin
  sorry
end

end smallest_number_with_5_zeros_7_ones_divisible_by_11_l39_39867


namespace floor_sqrt_sum_l39_39468

open Real

theorem floor_sqrt_sum : (∑ k in Finset.range 25, ⌊sqrt (k + 1)⌋) = 75 := 
by
  sorry

end floor_sqrt_sum_l39_39468


namespace proof_problem_l39_39162

variable {R : Type*} [Real R]

noncomputable def f : R → R := sorry

theorem proof_problem 
  (h_domain : ∀ (x : R), Continuous (f x))
  (h_functional_eq : ∀ (x y : R), f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end proof_problem_l39_39162


namespace initial_candies_l39_39273

-- Define initial variables and conditions
variable (x : ℕ)
variable (remaining_candies_after_first_day : ℕ)
variable (remaining_candies_after_second_day : ℕ)

-- Conditions as per given problem
def condition1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
def condition2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
def final_condition : remaining_candies_after_second_day = 10 := sorry

-- Goal: Prove that initially, Liam had 52 candies
theorem initial_candies : x = 52 := by
  have h1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
  have h2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
  have h3 : remaining_candies_after_second_day = 10 := sorry
    
  -- Combine conditions to solve for x
  sorry

end initial_candies_l39_39273


namespace kids_waiting_for_swings_l39_39328

theorem kids_waiting_for_swings (x : ℕ) (h1 : 2 * 60 = 120) 
  (h2 : ∀ y, y = 2 → (y * x = 2 * x)) 
  (h3 : 15 * (2 * x) = 30 * x)
  (h4 : 120 * x - 30 * x = 270) : x = 3 :=
sorry

end kids_waiting_for_swings_l39_39328


namespace smallest_positive_period_monotonicity_interval_l39_39171

section
variable (x : Real)

def f (x : Real) : Real := (sin x) * (2 * Real.sqrt 3 * (cos x) - sin x) + 1

theorem smallest_positive_period :
  ∃ T > 0, T = π ∧ ∀ x, f (x + T) = f x := by
  sorry

theorem monotonicity_interval :
  ∀ x ∈ (Set.Icc (-π / 4) (π / 4)), (x ≤ π / 6 → f' x > 0) ∧ (x > π / 6 → f' x < 0) := by
  sorry
end

end smallest_positive_period_monotonicity_interval_l39_39171


namespace george_income_l39_39133

def half (x: ℝ) : ℝ := x / 2

theorem george_income (income : ℝ) (H1 : half income - 20 = 100) : income = 240 := 
by 
  sorry

end george_income_l39_39133


namespace ST_length_proof_l39_39337

-- Define the geometry and necessary properties
noncomputable def triangle_PQR (PQ PR QR : ℝ) (S T : ℝ → ℝ) (tr : TrianglePoint) :=
  PQ = 13 ∧ PR = 14 ∧ QR = 15 ∧ parallel (S T) QR ∧ contains_incenter (S T) tr

-- Define the Lean 4 statement for the proof problem
theorem ST_length_proof (PQ PR QR : ℝ) (S T : ℝ → ℝ) (tr : TrianglePoint) :
  triangle_PQR PQ PR QR S T tr → ∃ ST : ℝ, ST = 135 / 14 :=
by
  intro h,
  have PQ_eq := h.1,
  have PR_eq := h.2,
  have QR_eq := h.3,
  sorry

end ST_length_proof_l39_39337


namespace largest_prime_divisor_of_factorial_sum_l39_39107

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n : ℕ), 13 ≤ n → (n ∣ 13! + 14!) → is_prime n → n = 13 :=
by sorry

end largest_prime_divisor_of_factorial_sum_l39_39107


namespace precious_stones_total_l39_39205

variable (agate olivine sapphire diamond amethyst ruby garnet topaz : ℕ)

theorem precious_stones_total
  (h1 : agate = 24)
  (h2 : olivine = agate + 5)
  (h3 : sapphire = 2 * olivine)
  (h4 : diamond = olivine + 11)
  (h5 : amethyst = sapphire + diamond)
  (h6 : ruby = 2.5 * olivine)
  (h7 : garnet = amethyst - ruby - 5)
  (h8 : topaz = garnet / 2) :
  agate + olivine + sapphire + diamond + amethyst + ruby + garnet + topaz = 352 := 
sorry

end precious_stones_total_l39_39205


namespace additional_money_needed_for_free_shipping_l39_39075

-- Define the prices of the books
def price_book1 : ℝ := 13.00
def price_book2 : ℝ := 15.00
def price_book3 : ℝ := 10.00
def price_book4 : ℝ := 10.00

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Calculate the discounted prices
def discounted_price_book1 : ℝ := price_book1 * (1 - discount_rate)
def discounted_price_book2 : ℝ := price_book2 * (1 - discount_rate)

-- Sum of discounted prices of books
def total_cost : ℝ := discounted_price_book1 + discounted_price_book2 + price_book3 + price_book4

-- Free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- The proof statement
theorem additional_money_needed_for_free_shipping : additional_amount = 9.00 := by
  -- calculation steps omitted
  sorry

end additional_money_needed_for_free_shipping_l39_39075


namespace sum_to_common_fraction_l39_39501

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l39_39501


namespace problem1_problem2_l39_39937

def z1 : ℂ := 1 - 3 * complex.I
def z2_2 : ℂ := 2 - 3 * complex.I
def z2_a (a : ℝ) : ℂ := a - 3 * complex.I

theorem problem1 : z1 * complex.conj (z2_2) = -7 - 3 * complex.I := 
  by sorry

theorem problem2 (a : ℝ) (h : (z1 / z2_a a).im ≠ 0) : a = -9 := 
  by 
    rw complex.add_im_div at h
    have : (z1 / z2_a a).re = 0 := sorry
    exact sorry

end problem1_problem2_l39_39937


namespace parallelogram_line_equations_l39_39898

/-- Given points A, B, C as defined and the properties of the parallelogram ABCD,
    prove the equation of the line on which side AD lies and the equation of the
    line containing the altitude from C to side CD. -/
theorem parallelogram_line_equations :
  let A := (-1 : ℝ, 4 : ℝ),
      B := (-2 : ℝ, -1 : ℝ),
      C := (2 : ℝ, 3 : ℝ),
      D := (x, y : ℝ) -- D is the point we need to find
  in (AD_line_eq : ∃ x y : ℝ, y = x - 1) ∧ 
     (altitude_line_eq : ∃ x y : ℝ, y = -1/5 * x + (4 + 1/5)) :=
by
  sorry

end parallelogram_line_equations_l39_39898


namespace weight_loss_percentage_l39_39379

theorem weight_loss_percentage {W : ℝ} (hW : 0 < W) :
  (((W - ((1 - 0.13 + 0.02 * (1 - 0.13)) * W)) / W) * 100) = 11.26 :=
by
  sorry

end weight_loss_percentage_l39_39379


namespace quadratic_properties_l39_39919

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l39_39919


namespace maxSundaysInFirst45Days_l39_39731

-- Definitions based on the problem's conditions
def daysInWeek := 7
def totalDays := 45

-- The main theorem statement
theorem maxSundaysInFirst45Days (startDay : Nat) (h : startDay ∈ {0, 1, 2, 3, 4, 5, 6}) :
  (if startDay = 1 ∨ startDay = 2 ∨ startDay = 3 
   then 7 
   else 6) = 7 
:=
sorry

end maxSundaysInFirst45Days_l39_39731


namespace cone_lateral_surface_development_diagram_l39_39321

theorem cone_lateral_surface_development_diagram (r s : ℝ) (h_r : r = 1) (h_s : s = 3) : 
  let C := 2 * Real.pi * r in
  let L := C in
  let n := (2 * 180 * Real.pi) / (s * Real.pi) in
  n = 120 := by
  sorry

end cone_lateral_surface_development_diagram_l39_39321


namespace gold_stickers_for_second_student_l39_39712

theorem gold_stickers_for_second_student :
  (exists f : ℕ → ℕ,
      f 1 = 29 ∧
      f 3 = 41 ∧
      f 4 = 47 ∧
      f 5 = 53 ∧
      f 6 = 59 ∧
      (∀ n, f (n + 1) - f n = 6 ∨ f (n + 2) - f n = 12)) →
  (∃ f : ℕ → ℕ, f 2 = 35) :=
by
  sorry

end gold_stickers_for_second_student_l39_39712


namespace coefficient_of_x_in_expansion_l39_39309

theorem coefficient_of_x_in_expansion : 
  ∀ (x : ℝ), (∃ (coeff : ℝ), coeff = 10 ∧ 
    (∀ r : ℕ, r = 3 → (binomial 5 r) * x^(10 - 3 * r) = coeff * x)) :=
by
  sorry

end coefficient_of_x_in_expansion_l39_39309


namespace at_most_two_sides_equal_to_longest_diagonal_l39_39993

variables {n : ℕ} (P : convex_polygon n)
  (h1 : n > 3)
  (longest_diagonal : diagonal P)
  (equal_length_sides : finset (side P))
  (h2 : ∀ s ∈ equal_length_sides, side.length s = diagonal.length longest_diagonal)

theorem at_most_two_sides_equal_to_longest_diagonal :
  equal_length_sides.card ≤ 2 :=
sorry

end at_most_two_sides_equal_to_longest_diagonal_l39_39993


namespace wendy_third_day_miles_l39_39666

theorem wendy_third_day_miles (miles_day1 miles_day2 total_miles : ℕ)
  (h1 : miles_day1 = 125)
  (h2 : miles_day2 = 223)
  (h3 : total_miles = 493) :
  total_miles - (miles_day1 + miles_day2) = 145 :=
by sorry

end wendy_third_day_miles_l39_39666


namespace circumcircles_tangent_l39_39542

-- Definitions based on the given conditions
variables {A B C M T K P : Type} 
variables [triangle_ABC : triangle A B C] [point_M_on_BC : on_line_segment B C M]
variables {Γ : circle}
variables [tangent_Γ_AB : tangent_to Γ A B T] [tangent_Γ_BM : tangent_to Γ B M K]
variables [tangent_Γ_circumcircle_AMC : tangent_to Γ (circumcircle A M C) P]

-- Given condition that TK is parallel to AM
def TK_parallel_AM (TK : line_segment T K) (AM : line_segment A M) : Prop :=
  parallel TK AM

-- The proof goal is to show that circumcircles of triangles APT and KPC are tangent to each other
theorem circumcircles_tangent
  (h1 : triangle A B C)
  (h2 : on_line_segment B C M)
  (h3 : tangent_to Γ A B T)
  (h4 : tangent_to Γ B M K)
  (h5 : tangent_to Γ (circumcircle A M C) P)
  (h6 : TK_parallel_AM ⟨T, K⟩ ⟨A, M⟩) :
  tangent (circumcircle A P T) (circumcircle K P C) :=
sorry

end circumcircles_tangent_l39_39542


namespace yellow_tint_percentage_l39_39781

theorem yellow_tint_percentage {V₀ V₁ V_t red_pct yellow_pct : ℝ} 
  (hV₀ : V₀ = 40)
  (hRed : red_pct = 0.20)
  (hYellow : yellow_pct = 0.25)
  (hAdd : V₁ = 10) :
  (yellow_pct * V₀ + V₁) / (V₀ + V₁) = 0.40 :=
by
  sorry

end yellow_tint_percentage_l39_39781


namespace largest_prime_divisor_of_factorial_sum_l39_39111

theorem largest_prime_divisor_of_factorial_sum {n : ℕ} (h1 : n = 13) : 
  Nat.gcd (Nat.factorial 13) 15 = 1 ∧ Nat.gcd (Nat.factorial 13 * 15) 13 = 13 :=
by
  sorry

end largest_prime_divisor_of_factorial_sum_l39_39111


namespace determinant_scaled_columns_l39_39640

variables {α : Type*} [linear_ordered_field α] {V : Type*} [add_comm_group V] [module α V]

-- Define the vectors a, b, c as variables
variables (a b c : V)

-- Define the determinant D using the given condition
def D := a • (b ⨯ c)

-- The theorem statement
theorem determinant_scaled_columns (a b c : V) :
  det (matrix.matrix_det (λ i j, [3*a, 2*b, c].nth j)) = 6 * D :=
by sorry

end determinant_scaled_columns_l39_39640


namespace distinct_bad_arrangements_l39_39701

-- Defining a bad arrangement
def is_bad_arrangement (arr : list ℕ) : Prop :=
  ¬ ∀ n ∈ list.range 1 22, ∃ (sub : list ℕ), sub ⊆ list.blocks_nth n arr ∧ list.sum sub = n

-- Set of all rotations of an arrangement
def rotations (arr : list ℕ) : set (list ℕ) :=
  set.of_list (list.map (λ i, list.rotate i arr) (list.range arr.length))

-- Checking for reflection (assuming the list is halved and then reversed)
def reflections (arr : list ℕ) : set (list ℕ) :=
  set.of_list [arr, list.reverse arr]

-- Counting distinct bad arrangements under rotation and reflection
theorem distinct_bad_arrangements : 
  ∃ (s : set (list ℕ)), (∀ arr ∈ s, is_bad_arrangement arr) ∧ s.card = 3 := sorry

end distinct_bad_arrangements_l39_39701


namespace clock_operation_difference_l39_39400

theorem clock_operation_difference
    (QA QB : ℕ)
    (h1 : QA = 6 * QB)
    (h2 : ∀ t : ℕ, 4 * QA * t = 3 * QB * t) -- Same drain rate property
    (h3 : 3 * QB * 2 = 3 * QB) -- Clock 2 operates for 2 months 
    : (4 * QA * 2) / (3 * QB) - 2 = 14 :=
by
  -- Definitions for operational time difference are derived from the problem conditions.
  have t1 : 4 * QA = 24 * QB := by rw [h1]; ring
  have t2 : (24 * QB) / (3 * QB) = 8 := by rw [←mul_div_assoc, div_self, mul_one]; norm_num
  have t3 : 8 * 2 = 16 := by norm_num
  rw [t3, sub_eq_add_neg]
  norm_num
  sorry

end clock_operation_difference_l39_39400


namespace main_relationship_l39_39624

-- Define the angles involved
variables {X Y Z T A : Type}
variables (x y m t : ℝ)

-- Specify the conditions of the problem
-- Line segment ZT bisects $\angle Z$ meaning ∠XZT = ∠TZY = t
-- Segment XY is extended to point A such that ∠TAZ is a straight angle
def bisector_bisects_angle : Prop := 2 * t = y ∧ (m + t = 180)

-- Triangle angle sum property
def triangle_angle_sum : Prop := x + y + 2 * t = 180

-- State the main theorem to prove
theorem main_relationship
    (h1 : bisector_bisects_angle)
    (h2 : triangle_angle_sum) :
    m = (x + y + 180) / 2 :=
sorry

end main_relationship_l39_39624


namespace remainder_a_power_series_mod_10_l39_39141
noncomputable def integral_ln_x : ℝ :=
  ∫ (x : ℝ) in 1..Real.exp 1, Real.log x

theorem remainder_a_power_series_mod_10 :
  let a := integral_ln_x in
  (a ^ 100 + ∑ i in List.range 100, (2 ^ i) * Nat.binomial 100 (i + 1) * a ^ (99 - i)) % 10 = 1 :=
by
  let a := integral_ln_x
  have h : a = 1 := sorry  -- Use given \( a = \int_{1}^{e} \ln x \, dx = 1 \)
  rw [h]
  have series_sum : (1 + 2) ^ 100 % 10 = 1 := sorry  -- Evaluate the series modulo 10
  exact series_sum

end remainder_a_power_series_mod_10_l39_39141


namespace find_angles_of_triangle_ABC_l39_39710

-- Definitions for involved conditions
variables {A A1 B B1 C C1 : Type}
variables [Triangle ABC : Type]
variables [Triangle A1B1C1 : Type]

-- Angles of triangles and assumptions
variables (α β γ : ℝ) -- Angles in triangle ABC
variables (α1 β1 γ1 : ℝ) -- Angles in triangle A1B1C1

-- The segments considered are altitudes
variables (AA1_alt BB1_alt CC1_alt : Segment)

-- Consideration that AA1, BB1, CC1 are altitudes of triangle ABC
variables (is_altitude : ∀ P Q R, is_altitude_segment ABC P Q R)

-- Triangle ABC is similar to A1B1C1
variables (similarity : similar ABC A1B1C1)

-- Prove the angles given provided conditions
theorem find_angles_of_triangle_ABC (h_ABC_similar_A1B1C1: similarity ABC A1B1C1) 
    (h1 : ∀P Q R, is_altitude_segment ABC P Q R) :
    (α = 60 ∧ β = 60 ∧ γ = 60) ∨
    (α = 720/7 ∧ β = 360/7 ∧ γ = 180/7) :=
sorry

end find_angles_of_triangle_ABC_l39_39710


namespace odd_factors_of_240_l39_39580

theorem odd_factors_of_240 : 
    let n : ℕ := 240
    let p1 : ℕ := 2
    let p2 : ℕ := 3
    let p3 : ℕ := 5
    let e1 : ℕ := 4
    let e2 : ℕ := 1
    let e3 : ℕ := 1
    let prime_factorization : n = p1^e1 * p2^e2 * p3^e3
    let odd_prime_factorization : ℕ := p2^e2 * p3^e3
    let odd_factors_count : ℕ := (e2 + 1) * (e3 + 1)
    odd_factors_count = 4 := by
  sorry

end odd_factors_of_240_l39_39580


namespace transformed_data_properties_l39_39317

variables {x : ℕ → ℝ}
variable (n : ℕ)

-- The conditions
def mean (x : ℕ → ℝ) (n : ℕ) := (∑ i in finset.range n, x i) / n
def variance (x : ℕ → ℝ) (n : ℕ) := 
  (∑ i in finset.range n, (x i - mean x n)^2) / n
def stddev (x : ℕ → ℝ) (n : ℕ) := real.sqrt (variance x n)

axiom mean_x1_to_x8_equals_6 : mean x 8 = 6
axiom stddev_x1_to_x8_equals_2 : stddev x 8 = 2

-- The transformed data
def transformed_x (i : ℕ) := 2 * x i - 6

-- The statement
theorem transformed_data_properties :
  mean transformed_x 8 = 6 ∧ variance transformed_x 8 = 16 :=
by sorry

end transformed_data_properties_l39_39317


namespace find_abc_l39_39245

theorem find_abc (a b c : ℕ) (h1 : c = b^2) (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 3 := 
by
  sorry

end find_abc_l39_39245


namespace average_weight_14_children_l39_39694

theorem average_weight_14_children 
  (average_weight_boys : ℕ → ℤ → ℤ)
  (average_weight_girls : ℕ → ℤ → ℤ)
  (total_children : ℕ)
  (total_weight : ℤ)
  (total_average_weight : ℤ)
  (boys_count : ℕ)
  (girls_count : ℕ)
  (boys_average : ℤ)
  (girls_average : ℤ) :
  boys_count = 8 →
  girls_count = 6 →
  boys_average = 160 →
  girls_average = 130 →
  total_children = boys_count + girls_count →
  total_weight = average_weight_boys boys_count boys_average + average_weight_girls girls_count girls_average →
  average_weight_boys boys_count boys_average = boys_count * boys_average →
  average_weight_girls girls_count girls_average = girls_count * girls_average →
  total_average_weight = total_weight / total_children →
  total_average_weight = 147 :=
by
  sorry

end average_weight_14_children_l39_39694


namespace number_of_positive_divisors_of_60_l39_39966

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l39_39966


namespace three_points_integer_centroid_l39_39891

/--  
  Given 19 points in the plane with integer coordinates, no three collinear, 
  show that we can always find three points whose centroid has integer coordinates.
-/
theorem three_points_integer_centroid 
  (points: fin 19 → ℤ × ℤ)
  (h_no_three_collinear: ∀ (a b c : fin 19), a ≠ b → b ≠ c → a ≠ c → ¬ collinear [points a, points b, points c]) :
  ∃ (i j k : fin 19), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    let (x₁, y₁) := points i,
        (x₂, y₂) := points j,
        (x₃, y₃) := points k in
      (x₁ + x₂ + x₃) % 3 = 0 ∧ (y₁ + y₂ + y₃) % 3 = 0 :=
by
  sorry

end three_points_integer_centroid_l39_39891


namespace percent_volume_filled_with_water_l39_39014

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39014


namespace num_divisors_60_l39_39970

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l39_39970


namespace cannot_form_triangle_l39_39735

theorem cannot_form_triangle {a b c : ℝ} (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) : 
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end cannot_form_triangle_l39_39735


namespace total_amount_saved_is_40_percent_l39_39038

-- Define the original prices of the jacket and shirt
def jacket_price := 100
def shirt_price := 50

-- Define the discounts on the jacket and shirt
def jacket_discount := 0.30
def shirt_discount := 0.60

-- Calculate the total savings
def total_savings := (jacket_price * jacket_discount) + (shirt_price * shirt_discount)

-- Calculate the total original cost
def total_original_cost := jacket_price + shirt_price

-- Calculate the percentage saved
def percentage_saved := (total_savings / total_original_cost) * 100

theorem total_amount_saved_is_40_percent : percentage_saved = 40 := by
    sorry

end total_amount_saved_is_40_percent_l39_39038


namespace Hr_iff_not_Contain_Krplus1_l39_39895

open Set

variable {α : Type*} (A : Finset (Finset α)) (r : ℕ)

def H_r_family (A : Finset (Finset α)) (r : ℕ) : Prop :=
  ∀ S ∈ A, S.card ≤ r ∧
  ∀ T ⊆ A, T.card = r + 1 → ⋂₀ T = ∅

def K_(r+1) (A : Finset (Finset α)) (r : ℕ) : Prop :=
  ∃ (B : Finset (Finset α)), (∀ i ∈ B, i.card = r) ∧ B.card = r + 1 ∧
  ∀ S ⊂ B, S.card = r → ⋂₀ S ≠ ∅

theorem Hr_iff_not_Contain_Krplus1 :
  H_r_family A r ↔ ¬ K_(r + 1) A r :=
by
  sorry

end Hr_iff_not_Contain_Krplus1_l39_39895


namespace trigonometric_identity_l39_39828

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l39_39828


namespace pythagorean_triple_divisibility_l39_39671

theorem pythagorean_triple_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (∃ k₃, k₃ ∣ a ∨ k₃ ∣ b) ∧
  (∃ k₄, k₄ ∣ a ∨ k₄ ∣ b ∧ 2 ∣ k₄) ∧
  (∃ k₅, k₅ ∣ a ∨ k₅ ∣ b ∨ k₅ ∣ c) :=
by
  sorry

end pythagorean_triple_divisibility_l39_39671


namespace monotonicity_and_solution_l39_39940

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (x^2) / 2 - k * Real.log x

theorem monotonicity_and_solution (k : ℝ) :
  (k ≤ 0 → ∀ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < +∞ → f x1 k ≤ f x2 k) ∧
  (0 < k → (∀ x, 0 < x ∧ x < sqrt k → deriv (f x k) < 0) ∧
           (∀ x, sqrt k < x ∧ x < +∞ → deriv (f x k) > 0)) ∧
  ((k ≤ Real.exp 1 → ¬∃ x ∈ Ioo 1 (Real.sqrt (Real.exp 1)), f x k = 0) ∧
   (k > Real.exp 1 → ∃! x ∈ Ioo 1 (Real.sqrt (Real.exp 1)), f x k = 0)) :=
by sorry

end monotonicity_and_solution_l39_39940


namespace limit_proof_l39_39755

open Real

-- Given function definition
def f (x : ℝ) : ℝ :=
  (tg (exp (x + 2) - exp (x^2 - 4))) / (tg x + tg 2)

-- Statement of the theorem to be proved in Lean
theorem limit_proof : 
  tendsto f (𝓝 (-2)) (𝓝 (5 * (cos 2)^2)) :=
sorry

end limit_proof_l39_39755


namespace range_of_f_l39_39172

noncomputable def f (x : ℝ) := x^2 - 4 * x + 2

theorem range_of_f : set.range (λ x : ℝ, f x) ∩ (set.Icc 1 4) = set.Icc (-2 : ℝ) 2 := 
sorry

end range_of_f_l39_39172


namespace find_length_of_rod_l39_39210

-- Constants representing the given conditions
def weight_6m_rod : ℝ := 6.1
def length_6m_rod : ℝ := 6
def weight_unknown_rod : ℝ := 12.2

-- Proof statement ensuring the length of the rod that weighs 12.2 kg is 12 meters
theorem find_length_of_rod (L : ℝ) (h : weight_6m_rod / length_6m_rod = weight_unknown_rod / L) : 
  L = 12 := by
  sorry

end find_length_of_rod_l39_39210


namespace problem_sum_of_roots_l39_39374
noncomputable def sum_of_roots (a b c : ℤ) : ℤ :=
let Δ := b^2 - 4 * a * c
in if h : 0 ≠ a then
  (-b + Δ.sqrt) / (2 * a) + (-b - Δ.sqrt) / (2 * a)
else 0

theorem problem_sum_of_roots :
  (sum_of_roots (-1) (-15) 54) = -15 :=
by simp [sum_of_roots]; sorry

end problem_sum_of_roots_l39_39374


namespace valid_m_values_l39_39492

theorem valid_m_values :
  ∃ (m : ℕ), (m ∣ 720) ∧ (m ≠ 1) ∧ (m ≠ 720) ∧ ((720 / m) > 1) ∧ ((30 - 2) = 28) := 
sorry

end valid_m_values_l39_39492


namespace water_added_l39_39765

def container_capacity : ℕ := 80
def initial_fill_percentage : ℝ := 0.5
def final_fill_percentage : ℝ := 0.75
def initial_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity
def final_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity

theorem water_added (capacity : ℕ) (initial_percentage final_percentage : ℝ) :
  final_fill_amount capacity final_percentage - initial_fill_amount capacity initial_percentage = 20 :=
by {
  sorry
}

end water_added_l39_39765


namespace inequality_implication_l39_39322

theorem inequality_implication (x : ℝ) : 3 * x + 4 < 5 * x - 6 → x > 5 := 
by {
  sorry
}

end inequality_implication_l39_39322


namespace greatest_value_b_l39_39104

-- Define the polynomial and the inequality condition
def polynomial (b : ℝ) : ℝ := -b^2 + 8*b - 12
#check polynomial
-- State the main theorem with the given condition and the result
theorem greatest_value_b (b : ℝ) : -b^2 + 8*b - 12 ≥ 0 → b ≤ 6 :=
sorry

end greatest_value_b_l39_39104


namespace solve_problem_l39_39876

theorem solve_problem (a b c d e : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) 
  (h6 : a * b * c * d * e = 9!)
  (h7 : a * b + a + b = 728)
  (h8 : b * c + b + c = 342)
  (h9 : c * d + c + d = 464)
  (h10 : d * e + d + e = 780) : 
  a - e = 172 :=
sorry

end solve_problem_l39_39876


namespace tangent_line_eq_l39_39206

theorem tangent_line_eq (x y : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : x^2 + y^2 = r^2) :
  ∃ (L : ℝ), (L = x + 2 * y - 5) :=
by
  use x + 2 * y - 5
  sorry

end tangent_line_eq_l39_39206


namespace average_age_remains_l39_39693

theorem average_age_remains (total_age : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) (initial_people_avg : ℕ) 
                            (total_age_eq : total_age = initial_people_avg * 8) 
                            (new_total_age : ℕ := total_age - leaving_age)
                            (new_avg : ℝ := new_total_age / remaining_people) :
  (initial_people_avg = 25) ∧ (leaving_age = 20) ∧ (remaining_people = 7) → new_avg = 180 / 7 := 
by
  sorry

end average_age_remains_l39_39693


namespace intersection_sum_l39_39222

-- Definitions for points A, B, C, and midpoints D, E
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def D : ℝ × ℝ := midpoint A B
def E : ℝ × ℝ := midpoint B C

-- Equations of the lines AE and CD
def line (p1 p2 : ℝ × ℝ) (x : ℝ) : ℝ :=
  let slope := (p2.2 - p1.2) / (p2.1 - p1.1)
  p1.2 + slope * (x - p1.1)

def line_AE (x : ℝ) : ℝ := line A E x
def line_CD (x : ℝ) : ℝ := line C D x

-- Intersection point F of the lines AE and CD
def F : ℝ × ℝ :=
  let x := ((line C D 0) - (line A E 0)) / ((- (A.2 - E.2)/(A.1 - E.1)) - (- (C.2 - D.2)/(C.1 - D.1)))
  let y := line_AE x
  (x, y)

-- The proof problem statement
theorem intersection_sum : F.1 + F.2 = 6 :=
  sorry

end intersection_sum_l39_39222


namespace coordinates_A2023_l39_39917

theorem coordinates_A2023 (m n : ℝ) : 
    let A_1 := (m, n)
    let A_2 := (m, -n)
    let A_3 := (-m, -n)
    let A_4 := (-m, n)
    let A_5 := (m, n)
    -- cycle repeats every 4 steps
    -- thus A_{2023} will be the same as A_3
    (2023 % 4 = 3) → (A_1 = (m, n)) → (A_2023 = A_3) := 
by {
  sorry
}

end coordinates_A2023_l39_39917


namespace jubilant_2009th_is_4019_l39_39044

def is_jubilant (n : ℕ) : Prop :=
  nat.bitcount 1 n % 2 = 0

def nth_jubilant (n : ℕ) : ℕ :=
  (list.filter is_jubilant (list.range ((2 * n) + 1))).nth (n - 1)

theorem jubilant_2009th_is_4019 :
  nth_jubilant 2009 = 4019 :=
sorry

end jubilant_2009th_is_4019_l39_39044


namespace floor_sqrt_sum_l39_39465

theorem floor_sqrt_sum : (∑ x in Finset.range 25 + 1, (Nat.floor (Real.sqrt x : ℝ))) = 75 := 
begin
  sorry
end

end floor_sqrt_sum_l39_39465


namespace max_length_OB_l39_39351

-- Given conditions
variables {O A B : Type} [MetricSpace O] [MetricSpace A] [MetricSpace B]
variables (OA : dist O A) (OB : dist O B)
variables (angle : ℝ) (AB : ℝ)

-- Conditions from the problem
def angle_OAB : ℝ := O.angle A B -- the angle ∠OAB
def angle_AOB : ℝ := 45 -- the angle ∠AOB in degrees
def length_AB : ℝ := 1 -- the length of AB

-- Statement of the theorem
theorem max_length_OB (h1: angle_AOB = 45) (h2: length_AB = 1):
  ∃ OB_max : ℝ, OB_max = sqrt 2 := 
sorry

end max_length_OB_l39_39351


namespace isabel_uploaded_pictures_l39_39231

theorem isabel_uploaded_pictures :
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  total_pictures = 25 :=
by
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  show total_pictures = 25
  sorry

end isabel_uploaded_pictures_l39_39231


namespace regular_ngon_sum_zero_l39_39282

-- Definition of the problem
def vertices_on_regular_ngon (n : ℕ) (k : ℕ) (x : Fin n → ℝ) :=
  ∀ (s : Finset (Fin n)), s.card = k → (∑ i in s, x i) = 0

-- Proof Statement
theorem regular_ngon_sum_zero (n k : ℕ) (h₁ : k ∣ n) (h₂ : ∀ i : Fin n, x i ≠ 0) :
  ∃ (x : Fin n → ℝ), vertices_on_regular_ngon n k x :=
sorry

end regular_ngon_sum_zero_l39_39282


namespace sum_of_floor_sqrt_l39_39452

theorem sum_of_floor_sqrt :
  (∑ n in Finset.range 26, Int.floor (Real.sqrt n)) = 75 :=
by
  -- skipping proof details
  sorry

end sum_of_floor_sqrt_l39_39452


namespace smooth_numbers_classification_l39_39419

def is_smooth (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℤ), (∑ i in finset.range n, a i) = n ∧ (∏ i in finset.range n, a i) = n

theorem smooth_numbers_classification (n : ℕ) :
  is_smooth n ↔ ((n % 4 = 0 ∨ n % 4 = 1) ∧ n ≠ 4) := 
sorry

end smooth_numbers_classification_l39_39419


namespace expected_area_of_nth_circle_l39_39050

variable {X : ℕ → ℝ}
variable {a d : ℝ}
variable (n : ℕ)

def mean (X : ℕ → ℝ) (a : ℝ) : Prop :=
  ∀ i, i > 0 → i ≤ n → E[X[i]] = a

def variance (X : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ i, i > 0 → i ≤ n → Var[X[i]] = d

theorem expected_area_of_nth_circle
  (h_mean : mean X a)
  (h_variance : variance X d) :
  E[π * (Σ i in range n, X[i]^2)] = π * n * (d + a^2) := 
sorry

end expected_area_of_nth_circle_l39_39050


namespace mutual_acquaintances_or_strangers_l39_39213

theorem mutual_acquaintances_or_strangers (n : ℕ) (h : n = 18) :
  (∃ (s : Finset (Fin n)), s.card = 4 ∧
    (∀ (x y ∈ s), x ≠ y → (familiar x y ∨ unfamiliar x y))) :=
sorry

end mutual_acquaintances_or_strangers_l39_39213


namespace prob_more_heads_proof_l39_39724

noncomputable def prob_more_heads : ℝ :=
by sorry

theorem prob_more_heads_proof :
  (∀ (num_flips_1 num_flips_2 : ℕ) (p : ℝ), 
    num_flips_1 = 10 → 
    num_flips_2 = 11 → 
    p = 0.5 → 
    prob_more_heads = 1 / 2) :=
by
  intros num_flips_1 num_flips_2 p h1 h2 h3
  replace h1 : num_flips_1 := 10
  replace h2 : num_flips_2 := 11
  replace h3 : p := 0.5
  exact sorry

end prob_more_heads_proof_l39_39724


namespace sandwiches_consumption_difference_l39_39269

theorem sandwiches_consumption_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let combined_monday_tuesday := monday_total + tuesday_total

  combined_monday_tuesday - wednesday_total = -5 :=
by
  sorry

end sandwiches_consumption_difference_l39_39269


namespace solve_for_M_plus_N_l39_39199

theorem solve_for_M_plus_N (M N : ℕ) (h1 : 4 * N = 588) (h2 : 4 * 63 = 7 * M) : M + N = 183 := by
  sorry

end solve_for_M_plus_N_l39_39199


namespace ellipse_properties_l39_39654

noncomputable def ellipse_conditions (a b : ℝ) (ha : a > b) (hb : b > 0) : Prop :=
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → 
  (∃ (F1 F2 P Q : ℝ × ℝ), 
    -- Line l passing through F2 intersects at P and Q
    (∃ l : ℝ, l = 1 → 
      -- Perimeter of triangle PQF1 is 2sqrt(3) times minor axis
      (let length_minor_axis := 2 * b in 
        (P ≠ Q) → 
        (abs P + abs Q + abs F1) = 2 * sqrt(3) * length_minor_axis)))))

theorem ellipse_properties {a b : ℝ} (ha : a > b) (hb : b > 0) :
  ellipse_conditions a b ha hb →
  (let e := sqrt(1 - (b / a)^2) in e = sqrt(6) / 3) ∧
  (∀ (F1 F2 P Q M : ℝ × ℝ), 
    ¬ (F2.1 = a ∧ F2.2 = 0) → 
    (P ≠ Q) → 
    (slope_l := 1) → 
    (¬ (M.1 = 2 * P.1 + Q.1 ∧ M.2 = 2 * P.2 + Q.2))) :=
sorry

end ellipse_properties_l39_39654


namespace cone_volume_filled_88_8900_percent_l39_39021

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39021


namespace find_f_2012_l39_39555

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1 / 4
axiom f_condition2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem find_f_2012 : f 2012 = -1 / 4 := 
sorry

end find_f_2012_l39_39555


namespace find_x_l39_39957

def vector2 := (ℝ × ℝ)

def dot_product (v1 v2 : vector2) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def norm_squared (v : vector2) : ℝ := v.1 * v.1 + v.2 * v.2

def projection (a b : vector2) : vector2 :=
  let factor := dot_product a b / norm_squared a
  (factor * a.1, factor * a.2)

theorem find_x (x : ℝ) :
  let a : vector2 := (2, 1)
  let b : vector2 := (2, x)
  projection a b = a → x = 1 :=
by
  sorry

end find_x_l39_39957


namespace joyce_apples_l39_39237

theorem joyce_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 75 → 
    given_apples = 52 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 23 :=
by 
  intros initial_apples given_apples remaining_apples h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end joyce_apples_l39_39237


namespace sum_squares_distances_l39_39413

theorem sum_squares_distances (n : ℕ) (h_n : 2 ≤ n):
  ∃ sum_of_squares : ℝ, 
    (∀ (vertices : ℕ → ℂ) (line : ℂ) (center : ℂ), 
        (∀ i, abs (vertices i) = 1) ∧ (abs center = 0) ∧ (line = 0) → 
        sum_of_squares = ∑ i in Finset.range n, (vertices i).im * (vertices i).im) :=
  sorry

end sum_squares_distances_l39_39413


namespace sample_size_divided_into_six_groups_l39_39854

theorem sample_size_divided_into_six_groups
  (n : ℕ)
  (c1 c2 c3 : ℕ)
  (k : ℚ)
  (h1 : c1 + c2 + c3 = 36)
  (h2 : 20 * k = 1)
  (h3 : 2 * k * n = c1)
  (h4 : 3 * k * n = c2)
  (h5 : 4 * k * n = c3) :
  n = 80 :=
by
  sorry

end sample_size_divided_into_six_groups_l39_39854


namespace infinite_product_equals_nine_l39_39445

theorem infinite_product_equals_nine : 
  (∏ n : ℕ in Finset.range (Nat.succ n), (3^n)^(1/2^n)) = 9 := sorry

end infinite_product_equals_nine_l39_39445


namespace range_of_expression_l39_39547

theorem range_of_expression (x y : ℝ) (h : x^2 + (y-2)^2 ≤ 1) : 
    1 ≤ (x + real.sqrt 3 * y) / real.sqrt (x^2 + y^2) ∧ 
    (x + real.sqrt 3 * y) / real.sqrt (x^2 + y^2) ≤ 2 :=
sorry

end range_of_expression_l39_39547


namespace constants_partial_fractions_l39_39517

theorem constants_partial_fractions :
  ∃ P Q R : ℝ, 
    (∀ x : ℝ, x ≠ 0 → x^2 + 2 ≠ 0 → 
    (-2*x^2 + 5*x - 7) / (x^3 + 2*x) = (P / x) + ((Q * x + R) / (x^2 + 2))) → 
    P = -2 ∧ Q = 0 ∧ R = 5 :=
by
  use -2, 0, 5
  intro h x hx hxx
  sorry

end constants_partial_fractions_l39_39517


namespace num_divisors_60_l39_39973

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l39_39973


namespace arithmetic_square_root_of_x_plus_y_l39_39885

theorem arithmetic_square_root_of_x_plus_y (x y : ℝ) (h1 : 3 - x ≥ 0) (h2 : x - 3 ≥ 0) (h3 : sqrt (3 - x) + sqrt (x - 3) + 1 = y) :
  sqrt (x + y) = 2 :=
by
  sorry

end arithmetic_square_root_of_x_plus_y_l39_39885


namespace binary_to_decimal_equiv_l39_39078

-- Define the binary number 11011_2
def binary_number : list ℕ := [1, 1, 0, 1, 1]

-- Function to convert binary to decimal
def binary_to_decimal (b : list ℕ) : ℕ :=
  b.reverse.enum.map (λ ⟨i, d⟩ => d * 2^i).sum

-- Theorem statement
theorem binary_to_decimal_equiv : binary_to_decimal binary_number = 27 := by
  sorry

end binary_to_decimal_equiv_l39_39078


namespace find_X_l39_39521

theorem find_X (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 :=
sorry

end find_X_l39_39521


namespace num_solutions_abs_ineq_l39_39197

theorem num_solutions_abs_ineq :
  {x : ℤ | abs (3 * x^2 + 5 * x - 4) ≤ 12}.finite.to_finset.card = 8 :=
by
  sorry

end num_solutions_abs_ineq_l39_39197


namespace inequality_no_solution_l39_39208

theorem inequality_no_solution (m : ℝ) : (∀ x : ℝ, ¬ (x ≥ m ∧ x ≤ 2023)) ↔ m > 2023 :=
begin
  sorry
end

end inequality_no_solution_l39_39208


namespace smallest_number_div_by_11_with_zeros_and_ones_l39_39869

/-- Smallest Natural Number with 5 zeros and 7 ones divisible by 11. --/
theorem smallest_number_div_by_11_with_zeros_and_ones :
  ∃ n : ℕ, (nat.mod n 11 = 0) ∧ (string.length (nat.digits 10 n)) = 13 ∧ 
            (nat.countP (λ c, c = '0') (nat.digits 10 n) = 5) ∧
            (nat.countP (λ c, c = '1') (nat.digits 10 n) = 7) ∧
            n = 1000001111131 :=
sorry

end smallest_number_div_by_11_with_zeros_and_ones_l39_39869


namespace largest_prime_divisor_13_plus_14_fact_l39_39114

theorem largest_prime_divisor_13_plus_14_fact : 
  ∀ p : ℕ, prime p ∧ p ∣ 13! + 14! → p ≤ 13 := 
sorry

end largest_prime_divisor_13_plus_14_fact_l39_39114


namespace sum_infinite_series_l39_39841

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l39_39841


namespace smallest_n_exists_l39_39871

theorem smallest_n_exists :
  ∃ n ∈ ℕ , n > 0 ∧ ∃∞ (a : tuple ℚ n), 
    (a.to_list.sum ∈ ℤ ∧ (a.to_list.map (λ x, x⁻¹)).sum ∈ ℤ) → n = 3 :=
sorry

end smallest_n_exists_l39_39871


namespace beth_twice_sister_age_l39_39589

theorem beth_twice_sister_age (beth_age sister_age : ℕ) (h_beth : beth_age = 18) (h_sister : sister_age = 5) :
  ∃ n : ℕ, beth_age + n = 2 * (sister_age + n) ∧ n = 8 :=
by
  use 8
  split
  { calc
    beth_age + 8 = 18 + 8 : by rw [h_beth]
             ... = 26 : by ring
    2 * (sister_age + 8) = 2 * (5 + 8) : by rw [h_sister]
                  ... = 2 * 13 : by ring
                  ... = 26 : by ring
  }
  { exact rfl }
  sorry

end beth_twice_sister_age_l39_39589


namespace solve_problem_l39_39098

open Nat

theorem solve_problem :
  ∃ (n p : ℕ), p.Prime ∧ n > 0 ∧ ∃ k : ℤ, p^2 + 7^n = k^2 ∧ (n, p) = (1, 3) := 
by
  sorry

end solve_problem_l39_39098


namespace sequence_general_term_l39_39183

noncomputable def sequence (n : ℕ) : ℚ :=
if h : n > 0 then -1 / n else 0

theorem sequence_general_term (n : ℕ) (hn : n > 0) :
  let a : ℕ → ℚ := λ k, if k = 1 then -1 else sequence k
  in a n = -1 / n :=
by
  let a : ℕ → ℚ := λ k, if k = 1 then -1 else sequence k
  have h1 : a 1 = -1, by simp [a]
  sorry

end sequence_general_term_l39_39183


namespace min_diff_factorial_expr_l39_39703

theorem min_diff_factorial_expr (a b : ℕ) 
  (h1 : a ≥ b) 
  (h2 : 2500 = a.factorial * (∏ i in (finset.range m).erase a, (a - i).factorial) / (b.factorial * (∏ i in (finset.range n).erase b, (b - i).factorial))) 
  (h3 : ∃ a1 b1, (a = a1) ∧ (b = b1) ∧ (a + b) = (a1 + b1)) :
  |a - b| = 1 :=
begin
  sorry
end

end min_diff_factorial_expr_l39_39703


namespace largest_common_multiple_of_7_8_l39_39105

noncomputable def largest_common_multiple_of_7_8_sub_2 (n : ℕ) : ℕ :=
  if n <= 100 then n else 0

theorem largest_common_multiple_of_7_8 :
  ∃ x : ℕ, x <= 100 ∧ (x - 2) % Nat.lcm 7 8 = 0 ∧ x = 58 :=
by
  let x := 58
  use x
  have h1 : x <= 100 := by norm_num
  have h2 : (x - 2) % Nat.lcm 7 8 = 0 := by norm_num
  have h3 : x = 58 := by norm_num
  exact ⟨h1, h2, h3⟩

end largest_common_multiple_of_7_8_l39_39105


namespace angle_set_equality_l39_39094

open Real

theorem angle_set_equality (α : ℝ) :
  ({sin α, sin (2 * α), sin (3 * α)} = {cos α, cos (2 * α), cos (3 * α)}) ↔ 
  ∃ (k : ℤ), α = π / 8 + (k : ℝ) * (π / 2) :=
by
  sorry

end angle_set_equality_l39_39094


namespace cone_water_volume_percentage_l39_39005

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39005


namespace championship_positions_l39_39226

def positions_valid : Prop :=
  ∃ (pos_A pos_B pos_D pos_E pos_V pos_G : ℕ),
  (pos_A = pos_B + 3) ∧
  (pos_D < pos_E ∧ pos_E < pos_B) ∧
  (pos_V < pos_G) ∧
  (pos_D = 1) ∧
  (pos_E = 2) ∧
  (pos_B = 3) ∧
  (pos_V = 4) ∧
  (pos_G = 5) ∧
  (pos_A = 6)

theorem championship_positions : positions_valid :=
by
  sorry

end championship_positions_l39_39226


namespace steel_pipe_cutting_time_l39_39790

theorem steel_pipe_cutting_time (length_of_pipe : ℕ) (num_sections : ℕ) (time_per_cut : ℕ) : 
  length_of_pipe = 15 → num_sections = 5 → time_per_cut = 6 → 
  (num_sections - 1) * time_per_cut = 24 := 
by
  intros h1 h2 h3
  rw [h2, h3]
  norm_num
  exact rfl

end steel_pipe_cutting_time_l39_39790


namespace milk_leftover_l39_39476

def total_milk := 16
def kids_percentage := 0.75
def cooking_percentage := 0.50

theorem milk_leftover : 
  let consumed_by_kids := kids_percentage * total_milk in
  let remaining_after_kids := total_milk - consumed_by_kids in
  let used_for_cooking := cooking_percentage * remaining_after_kids in
  let milk_left := remaining_after_kids - used_for_cooking in
  milk_left = 2 := 
by
  sorry

end milk_leftover_l39_39476


namespace find_C_l39_39740

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 360) : C = 60 := by
  sorry

end find_C_l39_39740


namespace sum_of_decimals_as_common_fraction_l39_39505

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l39_39505


namespace sequence_a_formula_sequence_b_sum_l39_39645

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S T : ℕ → ℕ)

noncomputable def a_1 : ℕ := 3
noncomputable def a_next (n : ℕ) : ℕ := 2 * S n + 3
noncomputable def S_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k, a k)
noncomputable def b_n (n : ℕ) : ℕ := (2 * n - 1) * a n
noncomputable def T_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k, b k)

theorem sequence_a_formula :
  ∀ n, a 1 = 3 ∧ (∀ n, a (n+1) = 2 * (Finset.range n).sum (λ k, a k) + 3) → a n = 3^n :=
begin
  sorry
end

theorem sequence_b_sum :
  ∀ n, (∀ k, b k = (2 * k - 1) * a k) → (∀ n, T n = (Finset.range n).sum (λ k, b k)) → T n = (n-1) * 3^(n+1) + 3 :=
begin
  sorry
end

end sequence_a_formula_sequence_b_sum_l39_39645


namespace geometric_body_not_cylinder_l39_39035

-- Define the shape and size condition
def has_three_same_views (G : Type) : Prop :=
  ∃ V : G → Prop, (∀ g1 g2 g3 : G, V g1 ∧ V g2 ∧ V g3 → g1 = g2 ∧ g2 = g3)

-- Define specific geometric shapes
inductive GeometricBody
| sphere : GeometricBody
| triangular_pyramid : GeometricBody
| cube : GeometricBody
| cylinder : GeometricBody

open GeometricBody

-- The theorem statement
theorem geometric_body_not_cylinder : 
  has_three_same_views GeometricBody → ¬has_three_same_views cylinder := 
sorry

end geometric_body_not_cylinder_l39_39035


namespace renaming_not_unnoticeable_l39_39388

-- Define the conditions as necessary structures for cities and connections
structure City := (name : String)
structure Connection := (city1 city2 : City)

-- Definition of the king's list of connections
def kingList : List Connection := sorry  -- The complete list of connections

-- The renaming function represented generically
def rename (c1 c2 : City) : City := sorry  -- The renaming function which is unspecified here

-- The main theorem statement
noncomputable def renaming_condition (c1 c2 : City) : Prop :=
  -- This condition represents that renaming preserves the king's perception of connections
  ∀ c : City, sorry  -- The specific condition needs full details of renaming logic

-- The theorem to prove, which states that the renaming is not always unnoticeable
theorem renaming_not_unnoticeable : ∃ c1 c2 : City, ¬ renaming_condition c1 c2 := sorry

end renaming_not_unnoticeable_l39_39388


namespace symmetric_point_on_side_l39_39625

theorem symmetric_point_on_side
  (A B C B₁ C₁ : Type)
  [EuclideanGeometry A B C B₁ C₁]
  (hA : is_triangle A B C)
  (hA_angle : ∠A = 60°)
  (hB₁_bisector : is_angle_bisector B B₁)
  (hC₁_bisector : is_angle_bisector C C₁)
  (symmetric_point : point_symmetric A B₁ C₁) :
  lies_on_side symmetric_point B C :=
sorry

end symmetric_point_on_side_l39_39625


namespace find_a_l39_39158

theorem find_a
  (x y : ℝ)
  (a : ℝ)
  (h1 : (x^2 + (y + 2)^2 = 1/4))
  (h2 : (a > 0 ∧ -1 ≤ x ∧ x ≤ 2))
  (h3 : (∀ P Q : ℝ × ℝ, P = (x, y) ∧ Q = (a * x ^ 2, y) → |PQ| ≤ 9/2)) :
  a = (Real.sqrt 3 - 1) / 2 :=
by
  sorry

end find_a_l39_39158


namespace integer_values_n_satisfy_inequality_l39_39961

theorem integer_values_n_satisfy_inequality :
  {n : ℤ | -100 < n^4 ∧ n^4 < 100}.finite.card = 7 :=
by
  sorry

end integer_values_n_satisfy_inequality_l39_39961


namespace length_of_platform_l39_39761

-- Definitions for the given problem conditions
def train_length : ℝ := 300
def time_cross_signal_pole : ℝ := 8
def time_cross_platform : ℝ := 39

-- Additional necessary definition for the problem
def train_speed : ℝ := train_length / time_cross_signal_pole

-- The proof statement
theorem length_of_platform : ∃ (P : ℝ), P = 1162.5 ∧ train_speed * time_cross_platform = train_length + P :=
by
  sorry

end length_of_platform_l39_39761


namespace solution_to_problem_l39_39163

def f (x : ℝ) : ℝ := sorry

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem solution_to_problem
  (f : ℝ → ℝ)
  (h : functional_equation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end solution_to_problem_l39_39163


namespace sequence_properties_l39_39144

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
a1 + d * (n - 1)

theorem sequence_properties (d a1 : ℤ) (h_d_ne_zero : d ≠ 0)
(h1 : arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 10)
(h2 : (arithmetic_sequence a1 d 2)^2 = (arithmetic_sequence a1 d 1) * (arithmetic_sequence a1 d 5)) :
a1 = 1 ∧ ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end sequence_properties_l39_39144


namespace find_width_of_rectangle_l39_39046

variable (L : ℕ) (W : ℕ) (P : ℕ)

def is_rectangle (len : ℕ) (wid : ℕ) (peri : ℕ) : Prop :=
  peri = 2 * (len + wid)

theorem find_width_of_rectangle 
  (h₁ : L = 19) 
  (h₂ : P = 70) 
  (h₃ : is_rectangle L W P) : 
  W = 16 := 
  by 
    have h4 : 2 * (L + W) = 2 * (19 + W) := by rw h₁
    rw [is_rectangle, h₂, h4] at h₃
    have h5 : 70 = 2 * (19 + W) := by rw h₃
    linarith

end find_width_of_rectangle_l39_39046


namespace prob_correct_l39_39274

-- Define percentages as ratio values
def prob_beginner_excel : ℝ := 0.35
def prob_intermediate_excel : ℝ := 0.25
def prob_advanced_excel : ℝ := 0.20
def prob_no_excel : ℝ := 0.20

def prob_day_shift : ℝ := 0.70
def prob_night_shift : ℝ := 0.30

def prob_weekend : ℝ := 0.40
def prob_not_weekend : ℝ := 0.60

-- Define the target probability calculation
def prob_intermediate_or_advanced_excel : ℝ := prob_intermediate_excel + prob_advanced_excel
def prob_combined : ℝ := prob_intermediate_or_advanced_excel * prob_night_shift * prob_not_weekend

-- The proof problem statement
theorem prob_correct : prob_combined = 0.081 :=
by
  sorry

end prob_correct_l39_39274


namespace positive_divisors_60_l39_39989

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l39_39989


namespace negation_of_proposition_l39_39569

theorem negation_of_proposition {c : ℝ} (h : ∃ (c : ℝ), c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) :
  ∀ (c : ℝ), c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0 :=
by
  sorry

end negation_of_proposition_l39_39569


namespace num_divisors_60_l39_39972

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l39_39972


namespace trigonometric_identity_eq_neg_one_l39_39824

theorem trigonometric_identity_eq_neg_one :
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180)) = -1 :=
by
  -- Variables needed for hypotheses
  have h₁ : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₂ : Real.sin (60 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₃ : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h₄ : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  -- Main proof
  sorry

end trigonometric_identity_eq_neg_one_l39_39824


namespace polynomial_value_l39_39142

theorem polynomial_value (a b : ℝ) : 
  (|a - 2| + (b + 1/2)^2 = 0) → (2 * a * b^2 + a^2 * b) - (3 * a * b^2 + a^2 * b - 1) = 1/2 :=
by
  sorry

end polynomial_value_l39_39142


namespace number_of_divisors_60_l39_39977

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l39_39977


namespace sum_of_squares_l39_39672

-- Define the proposition as a universal statement 
theorem sum_of_squares (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by
  sorry

end sum_of_squares_l39_39672


namespace t_20_property_mul_12_smallest_k_t_20_15_property_l39_39420

-- Define the t-m-property
def t_m_property (k m : ℕ) : Prop :=
  ∀ a : ℕ, ∃ n : ℕ, (∑ i in Finset.range (n + 1), i^k) % m = a % m

-- Problem (a): k has t-20-property if and only if k is a multiple of 12
theorem t_20_property_mul_12 (k : ℕ) : t_m_property k 20 ↔ ∃ m : ℕ, k = 12 * m :=
sorry

-- Problem (b): The smallest k that has t-20^15-property is 4
theorem smallest_k_t_20_15_property : ∀ k : ℕ, t_m_property k (20^15) -> k = 4 :=
sorry

end t_20_property_mul_12_smallest_k_t_20_15_property_l39_39420


namespace number_of_odd_factors_of_240_l39_39578

theorem number_of_odd_factors_of_240 : 
  let n := 240
  let odd_factors_of_15 := (\[1, 3, 5, 15\])
  in (n = 2^4 * 15) →
     15 = 3^1 * 5^1 →
     (length (filter (λ x, x % 2 = 1) (divisors n)) = 4) sorry

end number_of_odd_factors_of_240_l39_39578


namespace parabola_directrix_l39_39699

theorem parabola_directrix (x y : ℝ) (h : y = 2 * x^2) : y = - (1 / 8) :=
sorry

end parabola_directrix_l39_39699


namespace men_in_first_group_l39_39298

theorem men_in_first_group (M : ℕ) 
  (h1 : (M * 25 : ℝ) = (15 * 26.666666666666668 : ℝ)) : 
  M = 16 := 
by 
  sorry

end men_in_first_group_l39_39298


namespace sqrt_simplification_l39_39376

variable (a b : ℝ)

theorem sqrt_simplification (h : a < 0) : sqrt ((a^2 * b) / 2) = - (a / 2) * sqrt (2 * b) := 
sorry

end sqrt_simplification_l39_39376


namespace number_of_green_balls_l39_39768

theorem number_of_green_balls
  (total_balls white_balls yellow_balls red_balls purple_balls : ℕ)
  (prob : ℚ)
  (H_total : total_balls = 100)
  (H_white : white_balls = 50)
  (H_yellow : yellow_balls = 10)
  (H_red : red_balls = 7)
  (H_purple : purple_balls = 3)
  (H_prob : prob = 0.9) :
  ∃ (green_balls : ℕ), 
    (white_balls + green_balls + yellow_balls) / total_balls = prob ∧ green_balls = 30 := by
  sorry

end number_of_green_balls_l39_39768


namespace rectangle_side_length_l39_39219

theorem rectangle_side_length (ABCD : Rectangle) (AB AD : ℝ)
  (h_AB : AB = 120)
  (E : Point) (h_E_midpoint : midpoint E A D)
  (h_perpendicular : perpendicular AC BE) :
  AD = 120 * sqrt 2 :=
sorry

end rectangle_side_length_l39_39219


namespace impossible_all_plus_1_for_15_impossible_all_plus_1_for_30_impossible_all_plus_1_for_general_n_max_K_for_general_n_max_K_for_200_l39_39606

-- (1) Statement for n = 15
theorem impossible_all_plus_1_for_15 (n : ℕ) (h15 : n = 15) :
  ∃ (arrangement : Fin n → bool), 
  (∀ (k : ℕ) (h₁ : k ∣ n) (h₂ : k ≥ 2), 
    ∃ (f : Fin k → Fin n) (h₃ : Function.Bijective f), 
    ∃ (transformation : (Fin k → bool) → (Fin k → bool)), 
    (transformation (λ x, arrangement (f x)) = λ x, bnot (arrangement (f x))) ∨ 
    (transformation (λ x, arrangement (f x)) = arrangement)) → 
  (arrangement ≠ (λ _, tt)) := sorry

-- (2) Statement for n = 30
theorem impossible_all_plus_1_for_30 (n : ℕ) (h30 : n = 30) :
  ∃ (arrangement : Fin n → bool), 
  (∀ (k : ℕ) (h₁ : k ∣ n) (h₂ : k ≥ 2), 
    ∃ (f : Fin k → Fin n) (h₃ : Function.Bijective f), 
    ∃ (transformation : (Fin k → bool) → (Fin k → bool)), 
    (transformation (λ x, arrangement (f x)) = λ x, bnot (arrangement (f x))) ∨ 
    (transformation (λ x, arrangement (f x)) = arrangement)) → 
  (arrangement ≠ (λ _, tt)) := sorry

-- (3) Statement for n > 2
theorem impossible_all_plus_1_for_general_n (n : ℕ) (hn : 2 < n) :
  ∃ (arrangement : Fin n → bool),
  (∀ (k : ℕ) (h₁ : k ∣ n) (h₂ : k ≥ 2),
    ∃ (f : Fin k → Fin n) (h₃ : Function.Bijective f),
    ∃ (transformation : (Fin k → bool) → (Fin k → bool)),
    (transformation (λ x, arrangement (f x)) = λ x, bnot (arrangement (f x))) ∨
    (transformation (λ x, arrangement (f x)) = arrangement)) →
  (arrangement ≠ (λ _, tt)) := sorry

-- (4) Statement for finding maximum K(n)
theorem max_K_for_general_n (n : ℕ) :
  ∃ (K : ℕ), K = 2 ^ ((Nat.div8 n) - 1 / 2) := sorry 

theorem max_K_for_200 :
  ∃ (K : ℕ), K = 2 ^ 80 := sorry

end impossible_all_plus_1_for_15_impossible_all_plus_1_for_30_impossible_all_plus_1_for_general_n_max_K_for_general_n_max_K_for_200_l39_39606


namespace integer_solution_count_l39_39196

theorem integer_solution_count :
  {x : ℤ | (x^2 - 3*x + 2)^(x + 3) = 1}.to_finset.card = 3 := 
sorry

end integer_solution_count_l39_39196


namespace an_correct_Tn_correct_l39_39948

-- Splitting the statements for clarity
open Nat

section problem
variables {a b : ℕ → ℚ}

-- Assumption about sum of first n terms
def Sn (n : ℕ) : ℚ := 2n^2 + 3n

-- General term of a_n sequence
def an (n : ℕ) : ℚ := 4n + 1

-- Definition of b_n sequence
def bn (n : ℕ) : ℚ := 1 / (an n * an (n + 1))

-- Sum of first n terms of b_n sequence
def Tn (n : ℕ) : ℚ := Finset.sum (Finset.range n) (λ i, bn i)

-- Theorem statements
theorem an_correct (n : ℕ) : Sn n - Sn (n - 1) = an n :=
  sorry

theorem Tn_correct (n : ℕ) : Tn n = n / (5 * (4n + 5)) :=
  sorry

end problem

end an_correct_Tn_correct_l39_39948


namespace distance_origin_to_z_l39_39612

/-- A structure representing the complex unit -/
noncomputable def imaginary_unit : ℂ := complex.I

/-- Definition of the complex number z -/
noncomputable def z : ℂ := (2 * imaginary_unit) / (1 + imaginary_unit)

noncomputable def distance_from_origin (w : ℂ) : ℝ := complex.abs w

/-- Theorem: The distance from the origin to the complex number z is √2 -/
theorem distance_origin_to_z : distance_from_origin z = real.sqrt 2 := sorry

end distance_origin_to_z_l39_39612


namespace shortest_path_length_l39_39845

-- Define the dimensions of the rectangular prism
def dimensions : ℕ × ℕ × ℕ := (1, 2, 3)

-- Define the length of the shortest path on the surface between opposite corners A and B
theorem shortest_path_length (d : ℕ × ℕ × ℕ) : d = dimensions → 
  let AB_length := 3 * Real.sqrt 2 in
  AB_length = 3 * (Real.sqrt 2) :=
by
  sorry

end shortest_path_length_l39_39845


namespace blue_length_of_pencil_l39_39783

theorem blue_length_of_pencil (total_length purple_length black_length blue_length : ℝ)
  (h1 : total_length = 6)
  (h2 : purple_length = 3)
  (h3 : black_length = 2)
  (h4 : total_length = purple_length + black_length + blue_length)
  : blue_length = 1 :=
by
  sorry

end blue_length_of_pencil_l39_39783


namespace shadedAreaValue_l39_39302

noncomputable
def equilateralTriangleLength := 4

noncomputable
def circleRadius := 2

theorem shadedAreaValue (a : ℝ) :
  let circleArea := π * circleRadius^2
  let sectorArea := (1/6) * circleArea
  let totalShadedArea := 3 * sectorArea
  totalShadedArea = a * π → a = 2 :=
by
  sorry

end shadedAreaValue_l39_39302


namespace ratio_area_triangle_circle_l39_39423

open Real

theorem ratio_area_triangle_circle
  (l r : ℝ)
  (h : ℝ := sqrt 2 * l)
  (h_eq_perimeter : 2 * l + h = 2 * π * r) :
  (1 / 2 * l^2) / (π * r^2) = (π * (3 - 2 * sqrt 2)) / 2 :=
by
  sorry

end ratio_area_triangle_circle_l39_39423


namespace measure_of_B_max_value_of_a_plus_c_l39_39186

variable {a b c A B C : Real}

-- Given the angle B in a triangle \triangle ABC.
axiom triangle_B (h : b * cos C = (2 * a - c) * cos B) :
  B = π / 3

-- Given the side length b = sqrt(3) in a triangle \triangle ABC.
axiom max_a_plus_c (h : b = sqrt 3) :
  a + c ≤ 2 * sqrt 3

theorem measure_of_B (h : b * cos C = (2 * a - c) * cos B) : B = π / 3 :=
  triangle_B h

theorem max_value_of_a_plus_c (h : b = sqrt 3) : a + c ≤ 2 * sqrt 3 :=
  max_a_plus_c h

end measure_of_B_max_value_of_a_plus_c_l39_39186


namespace steiner_double_system_mod_6_l39_39284

theorem steiner_double_system_mod_6 (n : ℕ) : exists (S : set (set ℕ)), (∀ s ∈ S, s = 2) ∧ (∀ s1 s2 ∈ S, s1 ≠ s2 → s1 ∩ s2 = ∅) → (n % 6 = 1 ∨ n % 6 = 3) :=
sorry

end steiner_double_system_mod_6_l39_39284


namespace billy_books_read_l39_39812

def hours_per_day : ℕ := 8
def days_per_weekend : ℕ := 2
def reading_percentage : ℚ := 0.25
def pages_per_hour : ℕ := 60
def pages_per_book : ℕ := 80

theorem billy_books_read :
  let total_hours := hours_per_day * days_per_weekend in
  let reading_hours := total_hours * reading_percentage in
  let total_pages := reading_hours * pages_per_hour in
  let books_read := total_pages / pages_per_book in
  books_read = 3 :=
by
  sorry

end billy_books_read_l39_39812


namespace infinite_grid_points_impossible_l39_39525

theorem infinite_grid_points_impossible :
  ∀ (n : ℕ) (H : n > 1),
    ¬∃ (S : set (ℤ × ℤ)), infinite S ∧
      (∀ (T : finset (ℤ × ℤ)), T ⊆ S → ∃ (x y : ℤ),
        (T.card > 0) → x = (T.sum (λ p, p.1) / T.card) ∧ y = (T.sum (λ p, p.2) / T.card) ∧ (x, y) ∈ T) :=
by
  sorry

end infinite_grid_points_impossible_l39_39525


namespace percentage_profits_to_revenues_l39_39602

variable {R P : ℝ}

-- Conditions from the given problem.
def revenues_fell_by_20_percent (R1999 : ℝ) :=
  R1999 = 0.8 * R

def profits_12_percent_of_revenues (P1999 R1999 : ℝ) :=
  P1999 = 0.12 * R1999

def profits_96_percent_of_previous_year (P1999 : ℝ) :=
  P1999 = 0.96 * P

-- The theorem we need to prove.
theorem percentage_profits_to_revenues :
  ∀ (R1999 P1999 : ℝ),
    revenues_fell_by_20_percent R1999 →
    profits_12_percent_of_revenues P1999 R1999 →
    profits_96_percent_of_previous_year P1999 →
    (P / R * 100 = 10) :=
by
  intros R1999 P1999 h1 h2 h3
  sorry

end percentage_profits_to_revenues_l39_39602


namespace beth_twice_sister_age_l39_39592

theorem beth_twice_sister_age :
  ∃ (x : ℕ), 18 + x = 2 * (5 + x) :=
by
  use 8
  sorry

end beth_twice_sister_age_l39_39592


namespace odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l39_39515

theorem odd_solutions_eq_iff_a_le_neg3_or_a_ge3 (a : ℝ) :
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 5 ∧ (a - 3 * x^2 + Real.cos (9 * Real.pi * x / 2)) * Real.sqrt (3 - a * x) = 0) ↔ (a ≤ -3 ∨ a ≥ 3) := 
by
  sorry

end odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l39_39515


namespace ab_plus_ac_plus_bc_value_l39_39653

open Real

theorem ab_plus_ac_plus_bc_value (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 1)
  (h2 : a + b + c = 0) :
  ab + ac + bc = -1/2 := 
by
  sorry

end ab_plus_ac_plus_bc_value_l39_39653


namespace george_monthly_income_l39_39130

theorem george_monthly_income (I : ℝ) (h : I / 2 - 20 = 100) : I = 240 := 
by sorry

end george_monthly_income_l39_39130


namespace solution_set_l39_39802

variable {f : ℝ → ℝ}

-- Assume f is differentiable on (-∞, 0)
axiom differentiable_on_f : ∀ x < 0, differentiable_at ℝ f x

-- Given condition: 2 * f(x) + x * deriv f x > x^2 for all x < 0
axiom given_inequality : ∀ x < 0, 2 * f x + x * deriv f x > x^2

-- Proof that the solution set for (x + 2017) ^ 2 * f(x + 2017) - 4 * f(-2) > 0 is x ∈ (-∞, -2019)
theorem solution_set :
  {x : ℝ | (x + 2017) ^ 2 * f (x + 2017) - 4 * f (-2) > 0} = set.Iio (-2019) :=
by
  sorry

end solution_set_l39_39802


namespace intersection_P_Q_l39_39954

open Set

def P := {x : ℝ | x > 0}
def Q := {x : ℤ | (x + 1) * (x - 4) < 0}

theorem intersection_P_Q : P ∩ (Q : Set ℝ) = {1, 2, 3} := by
  sorry

end intersection_P_Q_l39_39954


namespace sum_sequence_11000_l39_39484

def sequence_value (k : ℕ) : ℤ :=
  if ∃ n, k = n^2 then 
    (-1)^n * k 
  else
    (if k % 2 = 1 then 1 else -1) * k

def sum_to_n (n : ℕ) : ℤ :=
  ∑ k in finset.range n, sequence_value (k + 1)

theorem sum_sequence_11000 :
  sum_to_n 12100 = 1100000 := by
  sorry

end sum_sequence_11000_l39_39484


namespace All_Yarns_are_Zorps_and_Xings_l39_39601

-- Define the basic properties
variables {α : Type}
variable (Zorp Xing Yarn Wit Vamp : α → Prop)

-- Given conditions
axiom all_Zorps_are_Xings : ∀ z, Zorp z → Xing z
axiom all_Yarns_are_Xings : ∀ y, Yarn y → Xing y
axiom all_Wits_are_Zorps : ∀ w, Wit w → Zorp w
axiom all_Yarns_are_Wits : ∀ y, Yarn y → Wit y
axiom all_Yarns_are_Vamps : ∀ y, Yarn y → Vamp y

-- Proof problem
theorem All_Yarns_are_Zorps_and_Xings : 
  ∀ y, Yarn y → (Zorp y ∧ Xing y) :=
sorry

end All_Yarns_are_Zorps_and_Xings_l39_39601


namespace sec_225_eq_neg_sqrt2_l39_39092

theorem sec_225_eq_neg_sqrt2 : Real.sec (225 * Real.pi / 180) = -Real.sqrt 2 := by
  sorry

end sec_225_eq_neg_sqrt2_l39_39092


namespace minimize_glue_length_l39_39054

def cube (A B C D E F G H : Type) : Prop := 
  -- Define the properties of the cube with base ABCD and edges AE, BF, CG, DH
  sorry

def remove_pyramids (A B C D E F G H : Type) : Prop := 
  -- Define the removal of pyramids CBEF and ADGH
  sorry

theorem minimize_glue_length (A B C D E F G H : Type) [cube A B C D E F G H] [remove_pyramids A B C D E F G H] :
  (∃ edges : set (A × B × C × D × E × F × G × H), edges ⊆ original_cube_edges A B C D E F G H ∧ edges.card = 7 ∧ total_glue_length edges = minimum) :=
sorry

end minimize_glue_length_l39_39054


namespace num_divisors_60_l39_39971

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l39_39971


namespace domain_sqrt_sin_minus_cos_l39_39518

noncomputable def domain_function (x : ℝ) : Prop := ∃ k : ℤ, (2 * k * Real.pi + Real.pi / 4) ≤ x ∧ x ≤ (2 * k * Real.pi + 5 * Real.pi / 4)

theorem domain_sqrt_sin_minus_cos :
  (∀ x : ℝ, 0 ≤ sin x - cos x → domain_function x) := sorry

end domain_sqrt_sin_minus_cos_l39_39518


namespace find_satisfying_functions_l39_39513

-- Define the problem conditions
def satisfies_condition (f : ℚ → ℝ) : Prop :=
∀ x y : ℚ, f x ^ 2 - f y ^ 2 = f (x + y) * f (x - y)

-- State the theorem to prove all such functions
theorem find_satisfying_functions (f : ℚ → ℝ) (h : satisfies_condition f) :
  (∃ k a : ℝ, k > 0 ∧ ∀ x : ℚ, f x = k * (real.exp (real.log a * (x : ℝ)) - real.exp (real.log a * -(x : ℝ)))) ∨ (∀ x : ℚ, f x = 0) :=
sorry

end find_satisfying_functions_l39_39513


namespace division_of_right_triangle_into_three_equal_parts_l39_39266

-- Given definitions
variables {A B C D E : Point}
variable (ABC : Triangle)
variable [RightTriangle ABC]
variable [AngleEq (angle A) (30 : degrees)]
variable [Midpoint D A B]
variable [Perpendicular (lineSegment D E) (lineSegment A B)]
variable [OnLineSegment E A C]

-- Statement to prove equal division
theorem division_of_right_triangle_into_three_equal_parts :
  Congruent (Triangle ADE) (Triangle DEB) ∧ Congruent (Triangle DEB) (Triangle EBC) :=
sorry

end division_of_right_triangle_into_three_equal_parts_l39_39266


namespace number_of_ways_to_divide_day_l39_39030

theorem number_of_ways_to_divide_day (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h : n * m = 1440) : 
  ∃ (pairs : List (ℕ × ℕ)), (pairs.length = 36) ∧
  (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 * p.2 = 1440)) :=
sorry

end number_of_ways_to_divide_day_l39_39030


namespace train_crossing_pole_time_l39_39627

-- Definitions from conditions
def train_length : ℝ := 500 -- in meters
def train_speed_kmph : ℝ := 200 -- in km/hr

-- Conversion factor from km/hr to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Speed in m/s
def train_speed_mps : ℝ := train_speed_kmph * kmph_to_mps

-- The theorem we need to prove
theorem train_crossing_pole_time :
  train_length / train_speed_mps ≈ 9 := sorry

end train_crossing_pole_time_l39_39627


namespace distance_between_stones_l39_39061

variable (d : ℝ)

-- Conditions extracted from the problem
def num_stones := 31
def total_distance := 4.8
def arithmetic_series_sum (n : ℕ) (a₁ aₙ : ℝ) : ℝ :=
  (n / 2) * (a₁ + aₙ)

-- Summation for the distances moved for stones on one side
def half_sum :=
  arithmetic_series_sum 15 d (15 * d)

-- Total distance to account for both sides
def total_movement_distance :=
  2 * half_sum

-- Theorem to prove the distance between each stone
theorem distance_between_stones 
  (h : total_movement_distance d = total_distance) :
  d = 0.02 :=
by
  rw [total_movement_distance, half_sum, arithmetic_series_sum] at h
  simp at h
  field_simp at h
  linarith

end distance_between_stones_l39_39061


namespace ratio_of_plane_to_total_distance_l39_39779

def total_distance : ℝ := 900
def distance_by_bus : ℝ := 360
def distance_by_train : ℝ := (2/3) * distance_by_bus
def distance_by_plane : ℝ := total_distance - (distance_by_bus + distance_by_train)

theorem ratio_of_plane_to_total_distance : distance_by_plane / total_distance = 1 / 3 :=
by
  sorry

end ratio_of_plane_to_total_distance_l39_39779


namespace Billy_Reads_3_Books_l39_39809

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end Billy_Reads_3_Books_l39_39809


namespace imaginary_part_div_conjugate_l39_39535

-- Definitions based on the given conditions
def z1 : ℂ := 1 - 3 * complex.i
def z2 : ℂ := 3 + complex.i

-- Theorem statement corresponding to the proof problem
theorem imaginary_part_div_conjugate : 
  complex.imag (complex.conj z1 / z2) = 4 / 5 :=
by
  sorry

end imaginary_part_div_conjugate_l39_39535


namespace y₁_y₂_ge_0_l39_39946

variables {a b : ℝ} {m n x : ℝ}
noncomputable def y₁ (x : ℝ) := a * x^2 + 4 * x + b
noncomputable def y₂ (x : ℝ) := b * x^2 + 4 * x + a
noncomputable def min_y₁ := (a * (-2/a)^2 + 4 * (-2/a) + b)
noncomputable def min_y₂ := (b * (-2/b)^2 + 4 * (-2/b) + a)

theorem y₁_y₂_ge_0 (h1 : min_y₁ + min_y₂ = 0) : y₁ x + y₂ x ≥ 0 :=
sorry

end y₁_y₂_ge_0_l39_39946


namespace locus_of_intersection_points_of_equal_circles_l39_39899

-- Define the segment and point condition
variables (A B C : ℝ^2)
variables (h_segment : C ∈ segment ℝ (A, B))

-- Define the circles and their properties
def circle_through (P Q : ℝ^2) := { M : ℝ^2 | dist M P = dist M Q }

-- Define the condition of equal circles passing through given points
def equal_circles_condition (P Q R S : ℝ^2) := 
  ∃ R, circle_through P Q = circle_through R S

-- Define the perpendicular bisector of segment AB excluding midpoint
def perpendicular_bisector_excluding_midpoint (A B : ℝ^2) := 
  { M : ℝ^2 | dist M A = dist M B ∧ M ≠ (A + B) / 2 }

-- State the theorem as a Lean statement
theorem locus_of_intersection_points_of_equal_circles 
  (A B C : ℝ^2)
  (h_segment : C ∈ segment ℝ (A, B))
  (h_circles : equal_circles_condition A C C B) :
  locus_of_intersection_points_of_equal_circles A C B = perpendicular_bisector_excluding_midpoint A B ∪ {C} :=
sorry

end locus_of_intersection_points_of_equal_circles_l39_39899


namespace sum_fraction_series_eq_l39_39836

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l39_39836


namespace distance_between_foci_of_hyperbola_l39_39692

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem distance_between_foci_of_hyperbola (x y : ℝ) : 
  let as1 := y = 2 * x - 1
  let as2 := y = -2 * x + 7
  let hyperbola_contains := (5, 5)
  in distance_between_foci as1 as2 hyperbola_contains = sqrt 34 := 
sorry

end distance_between_foci_of_hyperbola_l39_39692


namespace sum_of_arithmetic_sequence_l39_39550

-- Define the arithmetic sequence and its properties
variable {a : ℕ → ℤ}

-- Hypothesis that the sequence is arithmetic with common difference -2
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n - 2

-- Hypothesis that a_7 is the geometric mean of a_3 and a_9
def geometric_mean_condition (a : ℕ → ℤ) : Prop :=
  a 7 * a 7 = a 3 * a 9

-- Sum of the first n terms of the arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  let S : ℕ → ℤ := λ n, ∑ i in finset.range n, a i in S n

-- Given an arithmetic sequence with the above properties, prove S_10 = 110
theorem sum_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_mean_condition a) :
  sum_of_first_n_terms a 10 = 110 := 
sorry

end sum_of_arithmetic_sequence_l39_39550


namespace gcd_lcm_product_l39_39121

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end gcd_lcm_product_l39_39121


namespace min_distance_fly_l39_39776

noncomputable def min_travel_distance (a : ℝ) : ℝ :=
4

theorem min_distance_fly (a : ℝ) : 
  a > 0 →
  (∃ P Q R E F G H : Point, 
    Tetrahedron a P Q R E ∧ Tetrahedron a E F G H ∧ 
    minimal_path P Q R E ∧ minimal_path E F G H ∧
    visit_faces P {A, B, C, D}) → 
  min_travel_distance a = 4 := 
sorry

end min_distance_fly_l39_39776


namespace AngleBisectorTheorem_l39_39227

namespace TriangleProblem

variables {P Q R A B C I S : Type*}

-- Given conditions in the problem
def isTriangle (P Q R : Type*) : Prop := True  -- Assuming non-degenerate triangles
def is_Bisector (PA : Type*) (I : Type*) (PQR : Type*) : Prop := True  -- Angle bisectors intersect at incenter
def commonSide (PQ : Type*) (PQS : Type*) : Prop := True  -- PQ is a common side

-- Angles in conditions
def angle_QPS : ℝ := 28
def angle_QRP : ℝ := 68

-- The proof goal
def measure_BIA : ℝ := 79

theorem AngleBisectorTheorem 
  (h1 : isTriangle P Q R)
  (h2 : isTriangle P Q S)
  (h3 : is_Bisector PA I PQR)
  (h4 : is_Bisector QB I PQR)
  (h5 : is_Bisector RC I PQR)
  (h6 : commonSide PQ PQS)
  (h7 : angle_QPS = 28)
  (h8 : angle_QRP = 68) :
  ∃ (BIA : ℝ), BIA = measure_BIA := by
  sorry

end TriangleProblem

end AngleBisectorTheorem_l39_39227


namespace Billy_Reads_3_Books_l39_39811

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end Billy_Reads_3_Books_l39_39811


namespace problem_l39_39544

noncomputable def a_n (n : ℕ) : ℝ := 2 * n - 7

def S_n (n : ℕ) : ℝ := (n * (2 * n - 7 + (-5))) / 2

def T_n (n : ℕ) : ℝ := (finset.range (n + 1)).sum (λ i, abs (a_n (i + 1)))

theorem problem (a₃ a₆ S₅ T₅: ℝ) (ha₃₊₆: a₃ + a₆ = 4) (hS₅: S₅ = -5) :
  a_n 3 + a_n 6 = 4 ∧ S_n 5 = -5 ∧ T_n 5 = 13 := 
by
  sorry

end problem_l39_39544


namespace together_work_days_l39_39386

/-- 
  X does the work in 10 days and Y does the same work in 15 days.
  Together, they will complete the work in 6 days.
 -/
theorem together_work_days (hx : ℝ) (hy : ℝ) : 
  (hx = 10) → (hy = 15) → (1 / (1 / hx + 1 / hy) = 6) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end together_work_days_l39_39386


namespace find_omega_find_intervals_of_increase_l39_39566

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * sin (ω * x) * cos (ω * x) + cos (2 * ω * x)

noncomputable def monotonic_increase_intervals (k : ℤ) : Set ℝ :=
  {x | - (3 * Real.pi) / 8 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 8 + k * Real.pi}

-- We will state the theorem we want in Lean, without providing a proof.
theorem find_omega (ω : ℝ) (h : ω > 0) (T : ℝ) (hf_periodic: ∀ x, f ω (x + T) = f ω x) :
  T = Real.pi → ω = 1 := sorry

theorem find_intervals_of_increase :
  (∀ x, f 1 (x + Real.pi) = f 1 x) → (∀ k : ℤ, ∃ I, I = monotonic_increase_intervals k) := sorry

end find_omega_find_intervals_of_increase_l39_39566


namespace hyperbola_eccentricity_is_sqrt2_l39_39537

variable (a b c : ℝ)

def hyperbola_asymptotes_perpendicular (a b : ℝ) : Prop :=
  b / a = 1

def focal_length (c : ℝ) : Prop :=
  2 * c = 8

def eccentricity (a c : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity_is_sqrt2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = 1) (h4 : 2 * c = 8) (h5 : c^2 = a^2 + b^2) :
  eccentricity a c = Real.sqrt 2 := by
    sorry

end hyperbola_eccentricity_is_sqrt2_l39_39537


namespace trigonometric_identity_l39_39834

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l39_39834


namespace student_weekly_allowance_l39_39435

-- Define the conditions
def saves_10_percent (A : ℝ) : ℝ := 0.10 * A
def spends_arcade (remaining : ℝ) : ℝ := (3/5) * remaining
def saves_again_10_percent (remaining : ℝ) : ℝ := 0.10 * remaining
def spends_toy_store (remaining : ℝ) : ℝ := (1/3) * remaining
def spends_last (remaining : ℝ) : ℝ := remaining

-- Define the theorem to be proven
theorem student_weekly_allowance (A : ℝ) 
    (a1 : remaining_1 = A - saves_10_percent A)
    (a2 : remaining_2 = remaining_1 - spends_arcade remaining_1)
    (a3 : remaining_3 = remaining_2 - saves_again_10_percent remaining_2)
    (a4 : remaining_4 = remaining_3 - spends_toy_store remaining_3)
    (a5 : remaining_5 = remaining_4 - saves_again_10_percent remaining_4)
    (a6 : remaining_5 = 0.60) : 
    A ≈ 3.09 := 
by
  sorry

end student_weekly_allowance_l39_39435


namespace sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l39_39682

noncomputable def calculate_time (distance1 distance2 speed1 speed2 : ℕ) : ℕ := 
  (distance1 / speed1) + (distance2 / speed2)

noncomputable def total_time_per_lap := calculate_time 200 100 4 6

theorem sofia_total_time_for_5_laps : total_time_per_lap * 5 = 335 := 
  by sorry

def converted_time (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem sofia_total_time_in_minutes_and_seconds :
  converted_time (total_time_per_lap * 5) = (5, 35) :=
  by sorry

end sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l39_39682


namespace planes_1_and_6_adjacent_prob_l39_39334

noncomputable def probability_planes_adjacent (total_planes: ℕ) : ℚ :=
  if total_planes = 6 then 1/3 else 0

theorem planes_1_and_6_adjacent_prob :
  probability_planes_adjacent 6 = 1/3 := 
by
  sorry

end planes_1_and_6_adjacent_prob_l39_39334


namespace cone_water_volume_percentage_l39_39002

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l39_39002


namespace percent_volume_filled_with_water_l39_39016

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39016


namespace smallest_c_such_that_one_in_range_l39_39124

theorem smallest_c_such_that_one_in_range :
  ∃ c : ℝ, (∀ x : ℝ, ∃ y : ℝ, y =  x^2 - 2 * x + c ∧ y = 1) ∧ c = 2 :=
by
  sorry

end smallest_c_such_that_one_in_range_l39_39124


namespace Toby_pull_time_l39_39336

theorem Toby_pull_time :
  let fully_loaded_snow_speed := 10 * 0.70,
      half_loaded_ice_speed := ((20 + 10) / 2) * 0.80,
      unloaded_normal_speed := 20,
      half_loaded_snow_speed := ((20 + 10) / 2) * 0.70,
      fully_loaded_ice_speed := 10 * 0.80,
      unloaded_normal_speed2 := 20,
      time1 := 150 / fully_loaded_snow_speed,
      time2 := 100 / half_loaded_ice_speed,
      time3 := 120 / unloaded_normal_speed,
      time4 := 90 / half_loaded_snow_speed,
      time5 := 60 / fully_loaded_ice_speed,
      time6 := 180 / unloaded_normal_speed2
  in time1 + time2 + time3 + time4 + time5 + time6 = 60.83 
:= 
by {
  let fully_loaded_snow_speed := 10 * 0.70,
  let half_loaded_ice_speed := ((20 + 10) / 2) * 0.80,
  let unloaded_normal_speed := 20,
  let half_loaded_snow_speed := ((20 + 10) / 2) * 0.70,
  let fully_loaded_ice_speed := 10 * 0.80,
  let unloaded_normal_speed2 := 20,
  let time1 := 150 / fully_loaded_snow_speed,
  let time2 := 100 / half_loaded_ice_speed,
  let time3 := 120 / unloaded_normal_speed,
  let time4 := 90 / half_loaded_snow_speed,
  let time5 := 60 / fully_loaded_ice_speed,
  let time6 := 180 / unloaded_normal_speed2,
  have : time1 + time2 + time3 + time4 + time5 + time6 = 60.83, { sorry },
  exact this,
}

end Toby_pull_time_l39_39336


namespace billy_reads_books_l39_39807

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end billy_reads_books_l39_39807


namespace ramu_profit_percent_is_correct_l39_39286

noncomputable def car_profit_percent (initial_cost : ℕ) (repair1 : ℕ) (repair2 : ℕ) (repair3 : ℕ) (sale_price : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_repair := repair1 + repair2 + repair3
  let total_cost := initial_cost + total_repair
  let discount_amount := (discount_rate / 100) * sale_price
  let discounted_price := sale_price - discount_amount.toNat
  let profit := discounted_price - total_cost
  (profit.toNat * 100) / total_cost.toNat

theorem ramu_profit_percent_is_correct :
  car_profit_percent 42000 2500 1750 2850 64900 5 = 25.56 :=
by
  sorry

end ramu_profit_percent_is_correct_l39_39286


namespace find_triplet_solution_l39_39860

theorem find_triplet_solution :
  ∃ m n p : ℕ, 
    0 < m ∧ 0 < n ∧ 0 < p ∧
    (Nat.Prime p) ∧
    (2 ^ m * p ^ 2 + 1 = n ^ 5) ∧
    (m = 1 ∧ n = 3 ∧ p = 11) :=
begin
  sorry
end

end find_triplet_solution_l39_39860


namespace quadratic_vertex_ordinate_l39_39528

theorem quadratic_vertex_ordinate :
  let a := 2
  let b := -4
  let c := -1
  let vertex_x := -b / (2 * a)
  let vertex_y := a * vertex_x ^ 2 + b * vertex_x + c
  vertex_y = -3 :=
by
  sorry

end quadratic_vertex_ordinate_l39_39528


namespace perpendicular_lines_exist_l39_39628

theorem perpendicular_lines_exist : ∃ A B C D : ℝ × ℝ, 
  (vector_angle (B.1 - A.1, B.2 - A.2) (D.1 - C.1, D.2 - C.2) = π / 2) ∧
  (vector_angle (C.1 - A.1, C.2 - A.2) (D.1 - B.1, D.2 - B.2) = π / 2) ∧
  (vector_angle (D.1 - A.1, D.2 - A.2) (C.1 - B.1, C.2 - B.2) = π / 2) :=
begin
  sorry
end

end perpendicular_lines_exist_l39_39628


namespace fastest_increasing_function_l39_39062

def A (x : ℝ) : ℝ := 100
def B (x : ℝ) : ℝ := 10
def C (x : ℝ) : ℝ := Real.log x
def D (x : ℝ) : ℝ := Real.exp x

theorem fastest_increasing_function :
  ∀ x : ℝ, x > 0 → (D x = Real.exp x) ∧ (∀ f ∈ {A, B, C}, ∃ δ > 0, ∀ ε > 0, x > δ + ε → D x > f x) :=
by
  sorry

end fastest_increasing_function_l39_39062


namespace sec_225_eq_neg_sqrt_2_l39_39089

theorem sec_225_eq_neg_sqrt_2 :
  (sec (225 : ℝ)) = -Real.sqrt 2 :=
by
  have h_cos_225 : (cos (225 : ℝ)) = cos (180 + 45), from rfl,
  have h_angle_subtraction_identity : (cos (180 + 45 : ℝ)) = -(cos 45), from sorry,
  have h_cos_45 : (cos (45 : ℝ)) = 1 / Real.sqrt 2, from sorry,
  have h_cos_225_value : (cos (225 : ℝ)) = - (1 / Real.sqrt 2), from 
    calc (cos (225 : ℝ)) 
          = cos (180 + 45 : ℝ) : by simp [h_cos_225]
      ... = - (cos 45 : ℝ)     : by simp [h_angle_subtraction_identity]
      ... = - (1 / Real.sqrt 2) : by simp [h_cos_45],
  have h_sec_def : (sec (225 : ℝ)) = (1 / cos (225 : ℝ)), from sorry,
  have h_sec_final : (1 / (- (1 / Real.sqrt 2))) = - Real.sqrt 2, from sorry,
  show (sec (225 : ℝ)) = - Real.sqrt 2, from 
    calc (sec (225 : ℝ))
          = 1 / (cos (225 : ℝ)) : by simp [h_sec_def]
      ... = 1 / (- (1 / Real.sqrt 2)) : by simp [h_cos_225_value]
      ... = - Real.sqrt 2 : by simp [h_sec_final]

end sec_225_eq_neg_sqrt_2_l39_39089


namespace construct_triangle_l39_39955

variables {A1 B2 C3 : Point}

-- Define the triangle and its properties.
structure Triangle (A B C : Point) :=
  (midpoint_BC : Midpoint A1 B C)
  (foot_of_altitude_from_B : FootAltitude B2 B C)

-- Define midpoint of the orthocenter and vertex C
def OrthocenterMidpoint (A B C : Point) : Prop :=
  ∃ M : Point, Midpoint M B C ∧ Midpoint C3 B M

theorem construct_triangle (A1 B2 C3 : Point) :
  ∃ A B C : Point, Triangle A B C ∧ OrthocenterMidpoint A B C :=
sorry

end construct_triangle_l39_39955


namespace ratio_proof_l39_39069

variable (A B C D E F M S O : Point)
variable (circleO : Circle)
variable (quadrilateralABCD : Quadrilateral)
variable (circumCircle : ∀ p : Point, p ∈ quadrilateralABCD.vertices → p ∈ circleO)
variable (incircleExists : ∃ incircle : Circle, (∀ p : Point, p ∈ quadrilateralABCD.edges → tangent p incircle))
variable (EFdiameter : diameter circleO E F)
variable (A_same_side_E_BD : same_side BD A E)
variable (perpendicularEF_BD : perpendicular EF BD)
variable (intersectBD_EF_at_M : intersects BD EF M)
variable (intersectBD_AC_at_S : intersects BD AC S)

theorem ratio_proof : (lengthSegment A S) / (lengthSegment S C) = (lengthSegment E M) / (lengthSegment M F) := sorry

end ratio_proof_l39_39069


namespace mosquitoes_required_l39_39048

theorem mosquitoes_required
  (blood_loss_to_cause_death : Nat)
  (drops_per_mosquito_A : Nat)
  (drops_per_mosquito_B : Nat)
  (drops_per_mosquito_C : Nat)
  (n : Nat) :
  blood_loss_to_cause_death = 15000 →
  drops_per_mosquito_A = 20 →
  drops_per_mosquito_B = 25 →
  drops_per_mosquito_C = 30 →
  75 * n = blood_loss_to_cause_death →
  n = 200 := by
  sorry

end mosquitoes_required_l39_39048


namespace mariela_cards_received_l39_39391

theorem mariela_cards_received (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403) (h2 : cards_at_home = 287) : 
  cards_in_hospital + cards_at_home = 690 := 
by 
  sorry

end mariela_cards_received_l39_39391


namespace range_m_div_n_l39_39478

noncomputable def f (x : ℝ) : ℝ := sorry -- Odd and decreasing function on ℝ

theorem range_m_div_n (f_odd : ∀ x, f (-x) = - f x)
  (f_decreasing : ∀ x y, x ≤ y → f y ≤ f x)
  (f_inequality : ∀ m n, f (m^2 - 2 * m) + f (2 * n - n^2) ≤ 0)
  (n_range : ∀ n, 1 ≤ n ∧ n ≤ 3 / 2) :
  ∀ m n, (1 ≤ n) ∧ (n ≤ 3/2) → (f (m^2 - 2 * m) + f (2 * n - n^2) ≤ 0) →
  (m / n ∈ set.Icc (1 / 3 : ℝ) 1) :=
by
  sorry

end range_m_div_n_l39_39478


namespace isosceles_triangle_rectangle_l39_39223

theorem isosceles_triangle_rectangle (PQ PR QR RT QT : ℝ)
  (h1 : PQ = PR)                   -- condition 1: isosceles triangle PQ = PR
  (h2 : angle Q P R = 70)          -- condition 3: ∠QPR = 70°
  (h3 : angle P Q R = x)
  (h4 : angle R Q T = y)
  (h5 : angle Q R T = 90)          -- condition 5: ∠RQT = 90°
  (h6 : QR = RT) (h7 : RT = QT)    -- conditions ensuring QRST is a rectangle
  : x + y = 145 :=
  sorry

end isosceles_triangle_rectangle_l39_39223


namespace mark_height_from_tree_ratio_l39_39398

theorem mark_height_from_tree_ratio
    (tree_height : ℝ) (tree_shadow : ℝ)
    (mark_shadow : ℝ) (tree_ratio_eq_mark_ratio : tree_height / tree_shadow = mark_height / mark_shadow)
    (tree_height_eq : tree_height = 50)
    (tree_shadow_eq : tree_shadow = 25)
    (mark_shadow_eq : mark_shadow = 20) :
  mark_height = 40 :=
by
  let tree_ratio := tree_height / tree_shadow
  have tree_ratio_eq_two : tree_ratio = 2 := by
    calc
      tree_ratio = 50 / 25 : by rw [tree_height_eq, tree_shadow_eq]
              ... = 2       : by norm_num
  have mark_height_eq := tree_ratio_eq_mark_ratio ▸ (tree_ratio_eq_two ▸ by rw [mark_shadow_eq, mul_comm 2 20]; norm_num)
  sorry

end mark_height_from_tree_ratio_l39_39398


namespace proof_problem_l39_39531

-- Given definitions
def A := { y : ℝ | ∃ x : ℝ, y = x^2 + 1 }
def B := { p : ℝ × ℝ | ∃ x : ℝ, p.snd = x^2 + 1 }

-- Theorem to prove 1 ∉ B and 2 ∈ A
theorem proof_problem : 1 ∉ B ∧ 2 ∈ A :=
by
  sorry

end proof_problem_l39_39531


namespace fifteenth_entry_correct_l39_39877

def r_7(n : ℕ) := n % 7

def satisfies_condition (n : ℕ) : Prop := r_7 (3 * n) ≤ 3

noncomputable def fifteenth_entry : ℕ :=
  Nat.find_greatest (λ (n : ℕ), satisfies_condition n) 21

theorem fifteenth_entry_correct : fifteenth_entry = 21 :=
  sorry -- Proof goes here

end fifteenth_entry_correct_l39_39877


namespace trigonometric_ratios_of_alpha_l39_39168

noncomputable def terminal_side_on_line_y_eq_sqrt3x (α : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (
    (k > 0 ∧ (sin α = sqrt 3 / 2 ∧ cos α = 1 / 2)) ∨ 
    (k < 0 ∧ (sin α = -sqrt 3 / 2 ∧ cos α = -1 / 2))
  )

theorem trigonometric_ratios_of_alpha (α : ℝ) (h : terminal_side_on_line_y_eq_sqrt3x α) : 
  (α = if α > 0 then (sin α = sqrt 3 / 2 ∧ cos α = 1 / 2) else (sin α = -sqrt 3 / 2 ∧ cos α = -1 / 2)) :=
sorry

end trigonometric_ratios_of_alpha_l39_39168


namespace problem_conditions_and_conclusions_l39_39656

noncomputable def f : ℝ → ℝ := sorry

theorem problem_conditions_and_conclusions :
  (∀ x : ℝ, f(x + 1) = f(x - 1)) →
  (∀ x : ℝ, f(-x) = f(x)) →
  (∀ x : set.Icc (0 : ℝ) 1, f(x) = (1/2)^(1 - x)) →
  (f(2 + x) = f(x)) ∧
  (∀ x : set.Ioo (1 : ℝ) 2, ∀ (d : deriv f x), d < 0) ∧
  (∀ x : set.Ioo (2 : ℝ) 3, ∀ (d : deriv f x), d > 0) ∧
  ((∀ x : ℝ, f(x) ≤ 1) ∧ (∃ x : ℝ, f(x) = 1) ∧ (∃ y : set.Icc (0 : ℝ) 1, y ≠ 0 → f(0) ≠ 1/2)) ∧
  (∀ x : set.Ioo (3 : ℝ) 4, f(x) = (1/2)^(x - 3)) :=
by sorry

end problem_conditions_and_conclusions_l39_39656


namespace trigonometric_identities_l39_39998

variable (α : ℝ) -- Define the variable α

-- Given condition
def tan_alpha_plus_pi_div_3 (α : ℝ) := tan (α + π / 3) = 2 * sqrt 3

-- Theorem statement
theorem trigonometric_identities (h : tan (α + π / 3) = 2 * sqrt 3) :
  tan (α - 2 * π / 3) = 2 * sqrt 3 ∧ 2 * sin α ^ 2 - cos α ^ 2 = -43 / 52 := by
  sorry

end trigonometric_identities_l39_39998


namespace steven_ships_boxes_l39_39685

-- Translate the conditions into Lean definitions and state the theorem
def truck_weight_limit : ℕ := 2000
def truck_count : ℕ := 3
def pair_weight : ℕ := 10 + 40
def boxes_per_pair : ℕ := 2

theorem steven_ships_boxes :
  ((truck_weight_limit / pair_weight) * boxes_per_pair * truck_count) = 240 := by
  sorry

end steven_ships_boxes_l39_39685


namespace area_of_WIN_sector_l39_39407

-- Define the radius of the circle
def r : ℝ := 7

-- Define the probability of winning
def p_WIN : ℝ := 3 / 8

-- Define the total area of the circle
def total_area : ℝ := π * r^2

-- The goal is to prove the area of the WIN sector given the probability
theorem area_of_WIN_sector :
  ∃ (A_WIN : ℝ), A_WIN = p_WIN * total_area ∧ A_WIN = 147 * π / 8 :=
by
  sorry

end area_of_WIN_sector_l39_39407


namespace child_B_total_value_l39_39418

theorem child_B_total_value (total_money : ℝ) (ratio_A : ℝ) (ratio_B : ℝ) (ratio_C : ℝ)
                            (rate_A : ℝ) (rate_B : ℝ) (rate_C : ℝ) (time : ℝ)
                            (correct_total_value_B : ℝ) :
  let total_parts := ratio_A + ratio_B + ratio_C in
  let share_B := (ratio_B / total_parts) * total_money in
  let interest_B := share_B * rate_B * time in
  let total_value_B := share_B + interest_B in
  total_value_B = correct_total_value_B :=
by
  sorry

end child_B_total_value_l39_39418


namespace Linda_mean_score_eq_91_l39_39333

def Jake_scores := {80, 86, 90, 92, 95, 97}

def Jake_mean : ℕ := 89

def Linda_mean (s : Set ℕ) (total_score : ℕ) (Jake_mean : ℕ) : ℕ :=
  (total_score - 3 * Jake_mean) / 3

theorem Linda_mean_score_eq_91 :
  Linda_mean Jake_scores 540 Jake_mean = 91 :=
by
  sorry

end Linda_mean_score_eq_91_l39_39333


namespace minimum_value_when_a_eq_1_range_of_minimum_value_l39_39942

def f (x a : ℝ) : ℝ := x^2 + |x - a| + 1

def g (a : ℝ) : ℝ :=
  if a ≥ 1/2 then 3/4 + a
  else if -1/2 < a ∧ a < 1/2 then a^2 + 1
  else 3/4 - a

theorem minimum_value_when_a_eq_1 : ∀ x : ℝ, f x 1 ≥ 7/4 :=
sorry

theorem range_of_minimum_value (m : ℝ) : (∃ a : ℝ, g a = m) ↔ (1 ≤ m) :=
sorry

end minimum_value_when_a_eq_1_range_of_minimum_value_l39_39942


namespace tiling_possible_values_of_n_l39_39397

-- Define the sizes of the grid and the tiles
def grid_size : ℕ × ℕ := (9, 7)
def l_tile_size : ℕ := 3  -- L-shaped tile composed of three unit squares
def square_tile_size : ℕ := 4  -- square tile composed of four unit squares

-- Formalize the properties of the grid and the constraints for the tiling
def total_squares : ℕ := grid_size.1 * grid_size.2
def white_squares (n : ℕ) : ℕ := 3 * n
def black_squares (n : ℕ) : ℕ := n
def total_black_squares : ℕ := 20
def total_white_squares : ℕ := total_squares - total_black_squares

-- The main theorem statement
theorem tiling_possible_values_of_n (n : ℕ) : 
  (n = 2 ∨ n = 5 ∨ n = 8 ∨ n = 11 ∨ n = 14 ∨ n = 17 ∨ n = 20) ↔
  (3 * (total_white_squares - 2 * (20 - n)) / 3 + n = 23 ∧ n + (total_black_squares - n) = 20) :=
sorry

end tiling_possible_values_of_n_l39_39397


namespace triangle_possible_side_lengths_l39_39846

theorem triangle_possible_side_lengths (x : ℕ) (hx : x > 0) (h1 : x^2 + 9 > 12) (h2 : x^2 + 12 > 9) (h3 : 9 + 12 > x^2) : x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end triangle_possible_side_lengths_l39_39846


namespace ellipse_eccentricity_half_l39_39797

-- Definitions and assumptions
variable (a b c e : ℝ)
variable (h₁ : a = 2 * c)
variable (h₂ : b = sqrt 3 * c)
variable (eccentricity_def : e = c / a)

-- Theorem statement
theorem ellipse_eccentricity_half : e = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_half_l39_39797


namespace circle_intersection_zero_l39_39582

theorem circle_intersection_zero :
  (∀ θ : ℝ, ∀ r1 : ℝ, r1 = 3 * Real.cos θ → ∀ r2 : ℝ, r2 = 6 * Real.sin (2 * θ) → False) :=
by 
  sorry

end circle_intersection_zero_l39_39582


namespace billy_reads_books_l39_39808

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end billy_reads_books_l39_39808


namespace parabola_symmetry_range_l39_39165

theorem parabola_symmetry_range (p : ℝ) (hp : 0 < p) :
    (0 < p ∧ p < 2 / 3) ↔ (∃ A B : ℝ × ℝ, A ≠ B ∧ A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧
    (A.1 + A.2 = 1 ∨ B.1 + B.2 = 1) ∧ 
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = 1 / 2 ∧ 
        (A.1 + B.1) / 2 + (A.2 + B.2) / 2 = 1)) :=
begin
  sorry
end

end parabola_symmetry_range_l39_39165


namespace probability_x_gt_3y_l39_39669

noncomputable def rect_region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3020 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3010}

theorem probability_x_gt_3y : 
  (∫ p in rect_region, if p.1 > 3 * p.2 then 1 else (0:ℝ)) / 
  (∫ p in rect_region, (1:ℝ)) = 1007 / 6020 := sorry

end probability_x_gt_3y_l39_39669


namespace circumcircle_radius_AMB_l39_39278

noncomputable def circumradius_AMB (A B O C D : Point) (M : Point) : Real :=
  let alpha := ArcSin (5 / 13)
  let beta := ArcSin (5 / 7)
  let sin_alpha_plus_beta := (sin(alpha) * cos(beta)) + (sin(beta) * cos(alpha))
  let AB := 20
  let R := AB / (2 * sin_alpha_plus_beta)
  R

theorem circumcircle_radius_AMB (A B O C D : Point) (M : Point)
  (AO CO OD OB : Real)
  (h1 : AO = 13)
  (h2 : OB = 7)
  (h3 : CO = 5)
  (h4 : OD = 5)
  (h5 : C ≠ D)
  (h6 : C.1 > O.1)
  (h7 : D.1 > O.1)
  : circumradius_AMB A B O C D M = (91 * (6 - sqrt 6)) / 30 :=
by
  sorry

end circumcircle_radius_AMB_l39_39278


namespace sum_of_decimals_as_common_fraction_l39_39506

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l39_39506


namespace circle_equation_line_equation_l39_39554

-- Definitions and conditions for the first proof
def center (O: Point) [Inhabited Circular] : Bool :=
  O = ⟨0, 0⟩

def tangent_slope (ℓ: Line) (m: ℝ) : Bool :=
  tangent ℓ ∧ slope ℓ = m

def passes_through (ℓ: Line) (M: Point) : Bool :=
  ∃ x y, M = ⟨sqrt 2, 3 * sqrt 2⟩ ∧ Line.passes_through M ℓ

-- Proof for the first question
theorem circle_equation (O: Point) (ℓ: Line) (M: Point) :
  center O → passes_through ℓ M → tangent_slope ℓ 1 → equation_circle O = "x^2 + y^2 = 4" :=
by sorry

-- Definitions and conditions for the second proof
def distance (O M A B: Point) {p: ℝ} : Bool :=
  area_triangle O A B = p

-- Proof for the second question
theorem line_equation (O: Point) (ℓ: Line) (M A B: Point) (p r: ℝ) :
  center O → passes_through ℓ M → distance O M A B 2 →
  equation_line ℓ = ("x = " ++ sqrt 2 ∨ "y = " ++ (4 / 3) * x + (5 * sqrt 2) / 3) :=
by sorry

end circle_equation_line_equation_l39_39554


namespace minimum_value_l39_39598

variable {A B C : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]

noncomputable def given_triangle (a : A) (b : A := 1) (c : A := (Real.sqrt 3) / 2) : Prop := 
  ∃ (A B C : A), 
    0 < B ∧ 
    B ≤ π ∧ 
    0 < C ∧ 
    C ≤ π / 3

noncomputable def expression (C : A) : A := 4 * Real.sin C * Real.cos (C + π / 6)

theorem minimum_value (C : A) (hC : 0 < C ∧ C ≤ π / 3) : 
  ∀ C ∈ Icc (0 : A) (π / 3), expression C ≥ 0 :=
begin
  sorry
end

end minimum_value_l39_39598


namespace sum_of_real_solutions_l39_39520

open Real Polynomial

theorem sum_of_real_solutions :
  let p := (x-3)/(x^2 + 5*x + 2) = (x-6)/(x^2 - 9*x + 2)
  ∑ (x : ℝ) in (roots p), x = 57 / 11 :=
by sorry

end sum_of_real_solutions_l39_39520


namespace both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l39_39216

variables (p1 p2 : Prop)

theorem both_shots_hit (p1 p2 : Prop) : (p1 ∧ p2) ↔ (p1 ∧ p2) :=
by sorry

theorem both_shots_missed (p1 p2 : Prop) : (¬p1 ∧ ¬p2) ↔ (¬p1 ∧ ¬p2) :=
by sorry

theorem exactly_one_shot_hit (p1 p2 : Prop) : ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) ↔ ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) :=
by sorry

theorem at_least_one_shot_hit (p1 p2 : Prop) : (p1 ∨ p2) ↔ (p1 ∨ p2) :=
by sorry

end both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l39_39216


namespace sum_of_decimals_is_fraction_l39_39511

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l39_39511


namespace bed_to_frame_ratio_l39_39629

def price_of_bed_frame : ℝ := 75
def discount_rate : ℝ := 0.80
def amount_paid : ℝ := 660

theorem bed_to_frame_ratio :
  let x := (amount_paid / discount_rate - price_of_bed_frame) / price_of_bed_frame
  in x = 10 := by
  sorry

end bed_to_frame_ratio_l39_39629


namespace tank_A_is_64_percent_of_tank_B_l39_39752

-- Define the initial conditions
def height_A : ℝ := 8
def height_B : ℝ := 8
def circumference_A : ℝ := 8
def circumference_B : ℝ := 10

-- Define the radii of the tanks
def radius_A := circumference_A / (2 * π)
def radius_B := circumference_B / (2 * π)

-- Define the volumes of the tanks
def volume_A := π * radius_A^2 * height_A
def volume_B := π * radius_B^2 * height_B

-- Define the percentage of the capacity of Tank A with respect to Tank B
noncomputable def percentage_capacity := (volume_A / volume_B) * 100

-- Prove that the capacity of Tank A is 64% of the capacity of Tank B
theorem tank_A_is_64_percent_of_tank_B : percentage_capacity = 64 := by
  sorry

end tank_A_is_64_percent_of_tank_B_l39_39752


namespace sum_nine_smallest_even_multiples_of_7_l39_39373

theorem sum_nine_smallest_even_multiples_of_7 : 
  ∑ i in finset.range 9, 14 * (i + 1) = 630 :=
by
  sorry

end sum_nine_smallest_even_multiples_of_7_l39_39373


namespace trigonometric_identity_eq_neg_one_l39_39827

theorem trigonometric_identity_eq_neg_one :
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180)) = -1 :=
by
  -- Variables needed for hypotheses
  have h₁ : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₂ : Real.sin (60 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₃ : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h₄ : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  -- Main proof
  sorry

end trigonometric_identity_eq_neg_one_l39_39827


namespace number_of_divisors_60_l39_39975

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l39_39975


namespace sum_of_squares_l39_39552

theorem sum_of_squares (a b c : ℝ) (h1 : ab + bc + ca = 4) (h2 : a + b + c = 17) : a^2 + b^2 + c^2 = 281 :=
by
  sorry

end sum_of_squares_l39_39552


namespace card_distribution_count_l39_39529

def card_distribution_ways : Nat := sorry

theorem card_distribution_count :
  card_distribution_ways = 9 := sorry

end card_distribution_count_l39_39529


namespace percent_volume_filled_with_water_l39_39018

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l39_39018


namespace units_digit_F_F10_l39_39688

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem units_digit_F_F10 : units_digit (fibonacci (fibonacci 10)) = 5 := 
by sorry

end units_digit_F_F10_l39_39688


namespace trigonometric_identity_trigonometric_expression_l39_39395

noncomputable def point_P := (3 : ℝ, -4 : ℝ)
def alpha := real.angle
def sinα : ℝ := -4/5
def cosα : ℝ := 3/5
def tanα : ℝ := -4/3

theorem trigonometric_identity (α : real.angle) 
  (hpt : (cos α, sin α) = (3 / 5, -4 / 5))
  (h1 : sin α * cos α = 1 / 8)
  (h2 : π < α ∧ α < 5 * π / 4) :
  sin α = -4 / 5 ∧ cos α = 3 / 5 ∧ tan α = -4 / 3 :=
sorry

theorem trigonometric_expression (α : real.angle) 
  (hpt : (cos α, sin α) = (3 / 5, -4 / 5))
  (h1 : sin α * cos α = 1 / 8)
  (h2 : π < α ∧ α < 5 * π / 4) :
  cos α - sin α = -sqrt 3 / 12 :=
sorry

end trigonometric_identity_trigonometric_expression_l39_39395


namespace number_of_odd_factors_of_240_l39_39579

theorem number_of_odd_factors_of_240 : 
  let n := 240
  let odd_factors_of_15 := (\[1, 3, 5, 15\])
  in (n = 2^4 * 15) →
     15 = 3^1 * 5^1 →
     (length (filter (λ x, x % 2 = 1) (divisors n)) = 4) sorry

end number_of_odd_factors_of_240_l39_39579


namespace f_t_when_g_ln_x_range_of_a_l39_39879

noncomputable def num_real_roots (g : ℝ → ℝ) (t : ℝ) : ℝ := sorry

def g1 (x : ℝ) : ℝ := Real.log x

def g2 (x : ℝ) (a : ℝ) : ℝ :=
if h : x ≤ 0 then x else (-x^2 + 2*a*x + a)

theorem f_t_when_g_ln_x (t : ℝ) : num_real_roots g1 t = 1 :=
sorry

theorem range_of_a (a : ℝ) (t : ℝ) (h : num_real_roots (λ x, g2 x a) (t + 2) > num_real_roots (λ x, g2 x a) t) : a > 1 :=
sorry

end f_t_when_g_ln_x_range_of_a_l39_39879


namespace sum_of_m_and_n_l39_39760

noncomputable def number_of_rectangles (n : ℕ) : ℕ :=
Nat.choose (n + 1) 2 * Nat.choose (n + 1) 2

def number_of_squares (n : ℕ) : ℕ :=
(n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_m_and_n (n : ℕ) (h1 : number_of_squares 7 = 140) (h2 : number_of_rectangles 7 = 784) :
  let s := number_of_squares n,
      r := number_of_rectangles n,
      m := 5,
      k := 28 
  in r = 784 → s = 140 → (s / r = 5 / 28) → (m + k = 33) := 
by
  sorry

end sum_of_m_and_n_l39_39760


namespace time_spent_on_spelling_l39_39080

-- Define the given conditions
def total_time : Nat := 60
def math_time : Nat := 15
def reading_time : Nat := 27

-- Define the question as a Lean theorem statement
theorem time_spent_on_spelling : total_time - math_time - reading_time = 18 := sorry

end time_spent_on_spelling_l39_39080


namespace sophia_ate_pie_l39_39300

theorem sophia_ate_pie (weight_fridge weight_total weight_ate : ℕ)
  (h1 : weight_fridge = 1200) 
  (h2 : weight_fridge = 5 * weight_total / 6) :
  weight_ate = weight_total / 6 :=
by
  have weight_total_formula : weight_total = 6 * weight_fridge / 5 := by
    sorry
  have weight_ate_formula : weight_ate = weight_total / 6 := by
    sorry
  sorry

end sophia_ate_pie_l39_39300


namespace inequality_unequal_positive_numbers_l39_39999

theorem inequality_unequal_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > (2 * a * b) / (a + b) :=
by
sorry

end inequality_unequal_positive_numbers_l39_39999


namespace remainder_x1002_div_x2_minus_1_mul_x_plus_1_l39_39122

noncomputable def polynomial_div_remainder (a b : Polynomial ℝ) : Polynomial ℝ := sorry

theorem remainder_x1002_div_x2_minus_1_mul_x_plus_1 :
  polynomial_div_remainder (Polynomial.X ^ 1002) ((Polynomial.X ^ 2 - 1) * (Polynomial.X + 1)) = 1 :=
by sorry

end remainder_x1002_div_x2_minus_1_mul_x_plus_1_l39_39122


namespace cone_volume_filled_88_8900_percent_l39_39022

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l39_39022


namespace max_value_of_operation_l39_39059

theorem max_value_of_operation : 
  ∃ (n : ℤ), (10 ≤ n ∧ n ≤ 99) ∧ 4 * (300 - n) = 1160 := by
  sorry

end max_value_of_operation_l39_39059


namespace triangle_tan_identity_l39_39901

variable (A B C : ℝ)

def x := Real.tan ((B - C) / 2) * Real.tan (A / 2)
def y := Real.tan ((C - A) / 2) * Real.tan (B / 2)
def z := Real.tan ((A - B) / 2) * Real.tan (C / 2)

theorem triangle_tan_identity : x + y + z + x * y * z = 0 :=
by
  sorry

end triangle_tan_identity_l39_39901


namespace determinant_value_l39_39713

noncomputable def determinant_cos_sin : ℝ :=
  let a := real.cos (20 * real.pi / 180)
  let b := real.sin (40 * real.pi / 180)
  let c := real.sin (20 * real.pi / 180)
  let d := real.cos (40 * real.pi / 180)
  a * d - b * c

theorem determinant_value : determinant_cos_sin = 1 / 2 :=
sorry

end determinant_value_l39_39713


namespace part_one_part_two_l39_39911

noncomputable def g (t : ℝ) : ℝ := (8 * sqrt (t^2 + 1) * (2 * t^2 + 5)) / (16 * t^2 + 25)

theorem part_one (t : ℝ) (α β : ℝ) (h_distinct : α ≠ β) 
  (h_roots : 4 * α^2 - 4 * t * α - 1 = 0 ∧ 4 * β^2 - 4 * t * β - 1 = 0) 
  (h_bounds : α < β) : 
  (let f (x : ℝ) := (2 * x - t) / (x^2 + 1) in
  (f β - f α) = g t) := 
sorry

theorem part_two (u₁ u₂ u₃ : ℝ) (h_range : ∀ i ∈ [u₁, u₂, u₃], 0 < i ∧ i < π / 2)
  (h_sum : sin u₁ + sin u₂ + sin u₃ = 1) : 
  (1 / g (tan u₁) + 1 / g (tan u₂) + 1 / g (tan u₃) < (3 / 4) * sqrt 6) := 
sorry

end part_one_part_two_l39_39911


namespace sum_and_product_of_midpoint_l39_39072

-- Define the endpoints
def point1 : ℝ × ℝ × ℝ := (3, -2, 4)
def point2 : ℝ × ℝ × ℝ := (-1, 6, -8)

-- Define the midpoint calculation
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the sum and product of the coordinates of the midpoint
def sum_of_midpoint (m : ℝ × ℝ × ℝ) : ℝ := m.1 + m.2 + m.3
def product_of_midpoint (m : ℝ × ℝ × ℝ) : ℝ := m.1 * m.2 * m.3

-- Now state the theorem that we'll prove
theorem sum_and_product_of_midpoint :
  let m := midpoint point1 point2 in
  sum_of_midpoint m = 1 ∧ product_of_midpoint m = -4 :=
by
  let m := midpoint point1 point2
  exact And.intro sorry sorry

end sum_and_product_of_midpoint_l39_39072


namespace find_b_from_fixed_point_l39_39568

variable (a b : ℝ)
variable (H1 : 0 < a) (H2 : a ≠ 1)
variable (f : ℝ → ℝ := λ x, a^(x + b) + 3)

theorem find_b_from_fixed_point :
  f (-1) = 4 → b = 1 := by
  sorry

end find_b_from_fixed_point_l39_39568


namespace max_mass_range_l39_39633

theorem max_mass_range (H M : ℝ) (g : ℝ := 10) (r : ℝ := 0.25) (Δr : ℝ := 0.05)
  (tolerance : ℝ := 160) (M_nominal : ℝ := 800) :
  let M_min := M_nominal - tolerance,
      M_max := M_nominal + tolerance in
  640 ≤ M_min ∧ M_max ≤ 960 :=
by
  let M_max_formula : ℝ :=
    2000 * H * M / g / (r + Δr)
  let M_min_formula : ℝ :=
    2000 * H * M / g / (r - Δr)
  have hM_nominal : M_nominal = 2000 * H * M / g / r := sorry -- based on provided condition
  have h_formula : M_min_formula ≤ M_nominal ∧ M_nominal ≤ M_max_formula := sorry
  have h_tolerance : 640 = M_min ∧ 960 = M_max := sorry
  exact ⟨le_refl M_min, le_refl M_max⟩

end max_mass_range_l39_39633


namespace sum_of_positive_real_solutions_l39_39874

theorem sum_of_positive_real_solutions :
  ∑ x in { x : ℝ | x > 0 ∧ 2 * sin (2 * x) * (sin (2 * x) - sin (4028 * π ^ 2 / x)) = sin (4 * x) - 1 }, x = 1008 * π :=
by sorry

end sum_of_positive_real_solutions_l39_39874


namespace probability_diff_digits_l39_39799

open Finset

def two_digit_same_digit (n : ℕ) : Prop :=
  n / 10 = n % 10

def three_digit_same_digit (n : ℕ) : Prop :=
  (n % 100) / 10 = n / 100 ∧ (n / 100) = (n % 10)

def same_digit (n : ℕ) : Prop :=
  two_digit_same_digit n ∨ three_digit_same_digit n

def total_numbers : ℕ :=
  (199 - 10 + 1)

def same_digit_count : ℕ :=
  9 + 9

theorem probability_diff_digits : 
  ((total_numbers - same_digit_count) / total_numbers : ℚ) = 86 / 95 :=
by
  sorry

end probability_diff_digits_l39_39799


namespace arithmetic_square_root_of_x_plus_y_l39_39886

theorem arithmetic_square_root_of_x_plus_y (x y : ℝ) (h1 : 3 - x ≥ 0) (h2 : x - 3 ≥ 0) (h3 : sqrt (3 - x) + sqrt (x - 3) + 1 = y) :
  sqrt (x + y) = 2 :=
by
  sorry

end arithmetic_square_root_of_x_plus_y_l39_39886


namespace find_lambda_l39_39610

open Real

variables (λ : ℝ) (m n : ℝ)
def OA : ℝ × ℝ := (3, -1)
def OB : ℝ × ℝ := (0, 2)
def AB : ℝ × ℝ := (0, 2) - (3, -1) -- corresponds to (-3, 3)
def OC : ℝ × ℝ := (m, n)
def AC : ℝ × ℝ := OC - OA -- corresponds to (m - 3, n + 1)

def vectors_perpendicular : Prop := (OC.1 - 3) * AB.1 + (OC.2 + 1) * AB.2 = 0 -- corresponds to OC · AB = 0
def AC_parallel_to_OB : Prop := AC = λ • OB -- corresponds to AC = λ OB

theorem find_lambda
  (h1 : vectors_perpendicular λ m n)
  (h2 : AC_parallel_to_OB λ m n) :
  λ = -1 :=
sorry

end find_lambda_l39_39610


namespace units_digit_squares_eq_l39_39680

theorem units_digit_squares_eq (x y : ℕ) (hx : x % 10 + y % 10 = 10) :
  (x * x) % 10 = (y * y) % 10 :=
by
  sorry

end units_digit_squares_eq_l39_39680


namespace even_positive_integers_l39_39514

theorem even_positive_integers (n : ℕ) (h : 0 < n) :
  (∃ (x : Fin 2n → ℝ), Function.Injective x ∧ 
    ∀ s : Finset (Fin 2n), s.card = n → ∑ i in s, x i = ∏ i in (Finset.univ \ s), x i) ↔ 
  ∃ m : ℕ, n = 2 * m := 
sorry

end even_positive_integers_l39_39514


namespace sum_is_expected_sum_l39_39202

-- Define the initial assumptions and conditions
def n_is_multiple_of_3 (n : ℕ) : Prop := ∃ m : ℕ, n = 3 * m
def i : ℂ := complex.I

-- Define the sum expression
noncomputable def s (n : ℕ) : ℂ :=
  finset.sum (finset.range (n + 1)) (λ k, (k + 1) * i^k)

-- Define the expected result
def expected_sum (n : ℕ) : ℂ := (n / 3 : ℕ) + 1 + (n / 3 : ℕ) * i

-- Prove the equivalence
theorem sum_is_expected_sum (n : ℕ) (h : n_is_multiple_of_3 n) : s n = expected_sum n :=
by {
  sorry
}

end sum_is_expected_sum_l39_39202


namespace projection_on_P_l39_39643

noncomputable def projection (v : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let dot_prod := v.1 * n.1 + v.2 * n.2 + v.3 * n.3 in
let n_norm_sq := n.1 * n.1 + n.2 * n.2 + n.3 * n.3 in
let scale := dot_prod / n_norm_sq in
(v.1 - scale * n.1, v.2 - scale * n.2, v.3 - scale * n.3)

noncomputable def plane_projection := 
  projection (3, 1, 8) (1, -1, 1)

theorem projection_on_P :
  plane_projection = (5/3, 7/3, 20/3) :=
by
  sorry

end projection_on_P_l39_39643


namespace correct_operation_l39_39734

theorem correct_operation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = - (a^2 * b) :=
by
  sorry

end correct_operation_l39_39734


namespace base_length_of_isosceles_triangle_triangle_l39_39607

section Geometry

variable {b m x : ℝ}

-- Define the conditions
def isosceles_triangle (b : ℝ) : Prop :=
∀ {A B C : ℝ}, A = b ∧ B = b -- representing an isosceles triangle with two equal sides

def segment_length (m : ℝ) : Prop :=
∀ {D E : ℝ}, D - E = m -- the segment length between points where bisectors intersect sides is m

-- The theorem we want to prove
theorem base_length_of_isosceles_triangle_triangle (h1 : isosceles_triangle b) (h2 : segment_length m) : x = b * m / (b - m) :=
sorry

end Geometry

end base_length_of_isosceles_triangle_triangle_l39_39607


namespace tangent_line_slope_l39_39732

theorem tangent_line_slope (c p : ℝ × ℝ) (h : c = (2, 1) ∧ p = (6, 3)) 
: let m_radius := ((p.2 - c.2) / (p.1 - c.1))
  in -1 / m_radius = -2 := 
by 
  -- Extract the center and point
  cases h with hc hp;
  -- Substitute the values of c and p
  rw [hc, hp];
  -- Simplify the slope of the radius
  let m_radius := (3 - 1) / (6 - 2);
  -- Show that the slope of the tangent line is -2
  calc
  -1 / m_radius = -1 / (2 / 4) : by refl
  ... = -1 / (1 / 2) : by norm_num
  ... = -2 : by norm_num

end tangent_line_slope_l39_39732


namespace units_digit_base8_l39_39265

theorem units_digit_base8 (a b : ℕ) (h_a : a = 123) (h_b : b = 57) :
  let product := a * b
  let units_digit := product % 8
  units_digit = 7 := by
  sorry

end units_digit_base8_l39_39265


namespace num_divisors_sixty_l39_39981

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l39_39981


namespace problem1_problem2_l39_39394

namespace ProofProblems

-- Proof Problem (1) Translation
theorem problem1 : 0.064^(-1/3) - (-1/8)^0 + 7^(Real.logBase 7 2) + 0.25^(5/2) * 0.5^(-4) = 4 :=
by
  sorry

-- Definitions for Problem (2)
def a : ℝ := Real.log 2
def b : ℝ := Real.log 3 / Real.log 10 

-- Proof Problem (2) Translation
theorem problem2 : Real.logBase 6 (sqrt 30) = (b + 1) / (2 * (a + b)) :=
by
  sorry

end ProofProblems

end problem1_problem2_l39_39394


namespace difference_in_ages_27_l39_39698

def conditions (a b : ℕ) : Prop :=
  10 * b + a = (1 / 2) * (10 * a + b) + 6 ∧
  10 * a + b + 2 = 5 * (10 * b + a - 4)

theorem difference_in_ages_27 {a b : ℕ} (h : conditions a b) :
  (10 * a + b) - (10 * b + a) = 27 :=
sorry

end difference_in_ages_27_l39_39698


namespace number_of_divisors_60_l39_39974

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l39_39974


namespace complex_number_quadrant_l39_39562

-- Define the complex number z
def z : ℂ := (2 - Complex.i) / (1 + Complex.i)

-- Define the condition that we need to prove: z is in the fourth quadrant
theorem complex_number_quadrant :
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_quadrant_l39_39562


namespace inequality_solution_l39_39296

theorem inequality_solution (x : ℝ) : (x < -4 ∨ x > -4) → (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) :=
by
  intro h
  sorry

end inequality_solution_l39_39296


namespace hour_hand_angle_is_correct_l39_39801

noncomputable def hour_hand_angle_when_next_coincide : ℝ :=
let hour_circle_degrees := 360
let hour_hand_circles_per_minute_hand_circle := 1 / 9
let time_for_next_coincidence := hour_hand_circles_per_minute_hand_circle / 8 in
time_for_next_coincidence * hour_circle_degrees

theorem hour_hand_angle_is_correct :
  hour_hand_angle_when_next_coincide = 45 :=
by
  sorry

end hour_hand_angle_is_correct_l39_39801


namespace solve_angles_l39_39847

def triangle : Type := ℝ × ℝ × ℝ  -- Represents the sides of a triangle

def angles (a b c : ℝ) : Prop :=
  ∃ α β γ : ℝ, 
  cos α = (4 * a^2 / sqrt 3 + 8 * a^2 / π - a^2) / (8 * a^2 * sqrt 2 / (sqrt 3 ^ (1/4) * sqrt π)) ∧
  α ≈ 37.34 ∧
  cos β = (a^2 + 8 * a^2 / π - 4 * a^2 / sqrt 3) / (4 * a * 2 * a * sqrt 2 / sqrt π) ∧
  β ≈ 67.19 ∧
  γ = 180 - (α + β) ∧
  γ ≈ 75.47

def problem (a b c : ℝ) (h1 : a^2 = (sqrt 3 / 4) * b^2) (h2 : a^2 = (π / 8) * c^2) : Prop :=
  angles a b c

theorem solve_angles (a b c : ℝ) (h1 : a^2 = (sqrt 3 / 4) * b^2) (h2 : a^2 = (π / 8) * c^2) : problem a b c h1 h2 :=
  sorry

end solve_angles_l39_39847


namespace quadratic_inequality_solution_l39_39935

theorem quadratic_inequality_solution (a b c : ℝ) :
  (∀ x : ℝ, (x > -2 ∧ x < 1) ↔ (ax^2 + bx + c > 0)) →
  (∀ x : ℝ, (x ∈ (Set.Ioo (-∞ : ℝ) (-1) ∪ Set.Ioo (1 / 2) (∞ : ℝ))) ↔ (cx^2 - bx + a < 0)) :=
by 
  intros h x
  sorry

end quadratic_inequality_solution_l39_39935


namespace different_tangent_lines_count_is_5_l39_39604

open Set Real

/-- Definition of two circles in a coordinate plane and problem of counting different numbers
of lines that are simultaneously tangent to both circles. -/
def circle_tangent_lines_count (radius1 radius2 : ℝ) (center1 center2 : ℝ × ℝ) : Set ℝ :=
  let dist := dist center1 center2 in
  if dist = 0 then
    {0}  -- Concentric circles
  else if dist = abs (radius2 - radius1) then
    {1}  -- Internally tangent circles
  else if dist = radius1 + radius2 then
    {3}  -- Externally tangent circles
  else if dist < radius1 + radius2 ∧ dist > abs (radius2 - radius1) then
    {4}  -- Non-overlapping and not externally tangent circles
  else if dist < abs (radius2 - radius1) then
    {2} -- Overlapping circles
  else
    ∅   -- Otherwise (logically should not happen in well-defined configuration)

/-- The number of different possible values for the count of tangent lines between two circles
with given conditions above is 5. -/
theorem different_tangent_lines_count_is_5 :
  let k := circle_tangent_lines_count 4 5 (0,0) (a,0)
  ∃ s : Set ℝ, s = k ∧ (#s = 5) :=
by
  sorry

end different_tangent_lines_count_is_5_l39_39604


namespace expression_approx_4028_l39_39383

noncomputable def expression := (45 + 23 / 89) * 89

theorem expression_approx_4028 : Real.floor expression = 4028 := by
  sorry

end expression_approx_4028_l39_39383


namespace P_zero_is_64_div_127_l39_39718

noncomputable def P : ℕ → ℚ := 
  λ n, if n = 0 then 64/127 else if 1 ≤ n ∧ n ≤ 6 then (1/2)^n * (64/127) else 0

theorem P_zero_is_64_div_127 :
  P 0 = 64/127 :=
by
  have h : ∑ n in Finset.range 7, P n = 1 := sorry
  have p0 := P 0
  have p1 := P 1
  have p2 := P 2
  have p3 := P 3
  have p4 := P 4
  have p5 := P 5
  have p6 := P 6
  calc P 0 + (1/2) * P 0 + (1/4) * P 0 + (1/8) * P 0 + (1/16) * P 0 + (1/32) * P 0 + (1/64) * P 0
        = P 0 * (1 + (1/2) + (1/4) + (1/8) + (1/16) + (1/32) + (1/64)) : by sorry
    ... = P 0 * (127/64) : by sorry
    ... = 1 : by sorry
  exact sorry

end P_zero_is_64_div_127_l39_39718


namespace part1_part2_l39_39147

-- Define the sequence and its sum condition
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
axiom seq_condition : ∀ n : ℕ, a n = (3 / 4) * S n + 2

-- Part 1: Prove that the sequence {log base 2 of a_n} is arithmetic
theorem part1 (n : ℕ) : let b := λ n, Real.logb 2 (a n) in ∀ n : ℕ, b (n + 1) - b n = 2 :=
by sorry

-- Part 2: Prove the bounds on T_n under the given conditions
def c_n (n : ℕ) (b : ℕ → ℝ) := (-1)^(n+1) * (n + 1) / (b n * b (n + 1))
variables {b : ℕ → ℝ} (b_arith : ∀ n : ℕ, b (n + 1) = b n + 2)

theorem part2 (n : ℕ) 
  (T : ℕ → ℝ) 
  (T_def : T n = ∑ i in Finset.range (n + 1), c_n i b) : 
  1/21 ≤ T n ∧ T n ≤ 2/15 :=
by sorry

end part1_part2_l39_39147


namespace series_sum_equals_seven_ninths_l39_39842

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l39_39842
