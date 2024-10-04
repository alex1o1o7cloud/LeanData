import Mathlib

namespace polynomial_sequence_symmetric_l25_25672

def P : ℕ → ℝ → ℝ → ℝ → ℝ 
| 0, x, y, z => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_sequence_symmetric (m : ℕ) (x y z : ℝ) (σ : ℝ × ℝ × ℝ): 
  P m x y z = P m σ.1 σ.2.1 σ.2.2 :=
sorry

end polynomial_sequence_symmetric_l25_25672


namespace system_no_five_distinct_solutions_system_four_distinct_solutions_l25_25633

theorem system_no_five_distinct_solutions (a : ℤ) :
  ¬ ∃ x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ z₁ z₂ z₃ z₄ z₅ : ℤ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅) ∧
    (y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₁ ≠ y₄ ∧ y₁ ≠ y₅ ∧ y₂ ≠ y₃ ∧ y₂ ≠ y₄ ∧ y₂ ≠ y₅ ∧ y₃ ≠ y₄ ∧ y₃ ≠ y₅ ∧ y₄ ≠ y₅) ∧
    (z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ z₄ ≠ z₅) ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a) ∧
    (2 * y₃ * z₃ + x₃ - y₃ - z₃ = a) ∧ (2 * x₃ * z₃ - x₃ + y₃ - z₃ = a) ∧ (2 * x₃ * y₃ - x₃ - y₃ + z₃ = a) ∧
    (2 * y₄ * z₄ + x₄ - y₄ - z₄ = a) ∧ (2 * x₄ * z₄ - x₄ + y₄ - z₄ = a) ∧ (2 * x₄ * y₄ - x₄ - y₄ + z₄ = a) ∧
    (2 * y₅ * z₅ + x₅ - y₅ - z₅ = a) ∧ (2 * x₅ * z₅ - x₅ + y₅ - z₅ = a) ∧ (2 * x₅ * y₅ - x₅ - y₅ + z₅ = a) :=
sorry

theorem system_four_distinct_solutions (a : ℤ) :
  (∃ x₁ x₂ y₁ y₂ z₁ z₂ : ℤ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ z₁ ≠ z₂ ∧
    (2 * y₁ * z₁ + x₁ - y₁ - z₁ = a) ∧ (2 * x₁ * z₁ - x₁ + y₁ - z₁ = a) ∧ (2 * x₁ * y₁ - x₁ - y₁ + z₁ = a) ∧
    (2 * y₂ * z₂ + x₂ - y₂ - z₂ = a) ∧ (2 * x₂ * z₂ - x₂ + y₂ - z₂ = a) ∧ (2 * x₂ * y₂ - x₂ - y₂ + z₂ = a)) ↔
  ∃ k : ℤ, k % 2 = 1 ∧ a = (k^2 - 1) / 8 :=
sorry

end system_no_five_distinct_solutions_system_four_distinct_solutions_l25_25633


namespace determine_l_l25_25835

theorem determine_l :
  ∃ l : ℤ, (2^2000 - 2^1999 - 3 * 2^1998 + 2^1997 = l * 2^1997) ∧ l = -1 :=
by
  sorry

end determine_l_l25_25835


namespace find_subtracted_value_l25_25670

theorem find_subtracted_value (N : ℕ) (V : ℕ) (hN : N = 2976) (h : (N / 12) - V = 8) : V = 240 := by
  sorry

end find_subtracted_value_l25_25670


namespace length_fraction_of_radius_l25_25756

noncomputable def side_of_square_area (A : ℕ) : ℕ := Nat.sqrt A
noncomputable def radius_of_circle_from_square_area (A : ℕ) : ℕ := side_of_square_area A

noncomputable def length_of_rectangle_from_area_breadth (A b : ℕ) : ℕ := A / b
noncomputable def fraction_of_radius (len rad : ℕ) : ℚ := len / rad

theorem length_fraction_of_radius 
  (A_square A_rect breadth : ℕ) 
  (h_square_area : A_square = 1296)
  (h_rect_area : A_rect = 360)
  (h_breadth : breadth = 10) : 
  fraction_of_radius 
    (length_of_rectangle_from_area_breadth A_rect breadth)
    (radius_of_circle_from_square_area A_square) = 1 := 
by
  sorry

end length_fraction_of_radius_l25_25756


namespace book_prices_l25_25526

theorem book_prices (x : ℝ) (y : ℝ) (h1 : y = 2.5 * x) (h2 : 800 / x - 800 / y = 24) : (x = 20 ∧ y = 50) :=
by
  sorry

end book_prices_l25_25526


namespace john_bought_more_than_ray_l25_25489

variable (R_c R_d M_c M_d J_c J_d : ℕ)

-- Define the conditions
def conditions : Prop :=
  (R_c = 10) ∧
  (R_d = 3) ∧
  (M_c = R_c + 6) ∧
  (M_d = R_d + 1) ∧
  (J_c = M_c + 5) ∧
  (J_d = M_d + 2)

-- Define the question
def john_more_chickens_and_ducks (J_c R_c J_d R_d : ℕ) : ℕ :=
  (J_c - R_c) + (J_d - R_d)

-- The proof problem statement
theorem john_bought_more_than_ray :
  conditions R_c R_d M_c M_d J_c J_d → john_more_chickens_and_ducks J_c R_c J_d R_d = 14 :=
by
  intro h
  sorry

end john_bought_more_than_ray_l25_25489


namespace line_intersects_circle_l25_25758

theorem line_intersects_circle (k : ℝ) : ∀ (x y : ℝ),
  (x + y) ^ 2 = x ^ 2 + y ^ 2 →
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2)) ∧ 
  ((-1/2)^2 + (0)^2 < 1) →
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2) := 
by
  intro x y h₁ h₂
  sorry

end line_intersects_circle_l25_25758


namespace annual_growth_rate_l25_25529

theorem annual_growth_rate (p : ℝ) : 
  let S1 := (1 + p) ^ 12 - 1 / p
  let S2 := ((1 + p) ^ 12 * ((1 + p) ^ 12 - 1)) / p
  let annual_growth := (S2 - S1) / S1
  annual_growth = (1 + p) ^ 12 - 1 :=
by
  sorry

end annual_growth_rate_l25_25529


namespace ellipse_hyperbola_foci_l25_25493

theorem ellipse_hyperbola_foci {a b : ℝ} (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) :
  a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 :=
by sorry

end ellipse_hyperbola_foci_l25_25493


namespace volleyball_team_lineup_l25_25415

theorem volleyball_team_lineup :
  ∃ (choices : ℕ), choices = 18 * (Nat.choose 17 7) ∧ choices = 350064 :=
by
  use 18 * (Nat.choose 17 7)
  split
  · rfl
  · sorry

end volleyball_team_lineup_l25_25415


namespace nine_sided_polygon_diagonals_count_l25_25024

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25024


namespace batsman_average_l25_25659

/-- The average after 12 innings given that the batsman makes a score of 115 in his 12th innings,
     increases his average by 3 runs, and he had never been 'not out'. -/
theorem batsman_average (A : ℕ) (h1 : 11 * A + 115 = 12 * (A + 3)) : A + 3 = 82 := 
by
  sorry

end batsman_average_l25_25659


namespace find_a_b_l25_25320

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l25_25320


namespace smallest_solution_for_quartic_eq_l25_25897

theorem smallest_solution_for_quartic_eq :
  let f (x : ℝ) := x^4 - 40*x^2 + 144
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y :=
sorry

end smallest_solution_for_quartic_eq_l25_25897


namespace min_regions_l25_25748

namespace CircleDivision

def k := 12

-- Theorem statement: Given exactly 12 points where at least two circles intersect,
-- the minimum number of regions into which these circles divide the plane is 14.
theorem min_regions (k := 12) : ∃ R, R = 14 :=
by
  let R := 14
  existsi R
  exact rfl

end min_regions_l25_25748


namespace travel_routes_l25_25586

theorem travel_routes (S N : Finset ℕ) (hS : S.card = 4) (hN : N.card = 5) :
  ∃ (routes : ℕ), routes = 3! * 5^4 := by
  sorry

end travel_routes_l25_25586


namespace number_of_diagonals_l25_25087

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25087


namespace olaf_total_toy_cars_l25_25863

def olaf_initial_collection : ℕ := 150
def uncle_toy_cars : ℕ := 5
def auntie_toy_cars : ℕ := uncle_toy_cars + 1 -- 6 toy cars
def grandpa_toy_cars : ℕ := 2 * uncle_toy_cars -- 10 toy cars
def dad_toy_cars : ℕ := 10
def mum_toy_cars : ℕ := dad_toy_cars + 5 -- 15 toy cars
def toy_cars_received : ℕ := grandpa_toy_cars + uncle_toy_cars + dad_toy_cars + mum_toy_cars + auntie_toy_cars -- total toy cars received
def olaf_total_collection : ℕ := olaf_initial_collection + toy_cars_received

theorem olaf_total_toy_cars : olaf_total_collection = 196 := by
  sorry

end olaf_total_toy_cars_l25_25863


namespace sam_initial_puppies_l25_25483

theorem sam_initial_puppies (gave_away : ℝ) (now_has : ℝ) (initially : ℝ) 
    (h1 : gave_away = 2.0) (h2 : now_has = 4.0) : initially = 6.0 :=
by
  sorry

end sam_initial_puppies_l25_25483


namespace incorrect_variance_l25_25821

noncomputable def normal_pdf (x : ℝ) : ℝ :=
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (- (x - 1)^2 / 2)

theorem incorrect_variance :
  (∫ x, normal_pdf x * x^2) - (∫ x, normal_pdf x * x)^2 ≠ 2 := 
sorry

end incorrect_variance_l25_25821


namespace geom_seq_common_ratio_l25_25606

-- We define a geometric sequence and the condition provided in the problem.
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Condition for geometric sequence: a_n = a * q^(n-1)
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^(n-1)

-- Given condition: 2a_4 = a_6 - a_5
def given_condition (a : ℕ → ℝ) : Prop := 
  2 * a 4 = a 6 - a 5

-- Proof statement
theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_seq a q) (h_cond : given_condition a) : 
    q = 2 ∨ q = -1 :=
sorry

end geom_seq_common_ratio_l25_25606


namespace tan_passing_through_point_l25_25446

theorem tan_passing_through_point :
  (∃ ϕ : ℝ, (∀ x : ℝ, y = Real.tan (2 * x + ϕ)) ∧ (Real.tan (2 * (π / 12) + ϕ) = 0)) →
  ϕ = - (π / 6) :=
by
  sorry

end tan_passing_through_point_l25_25446


namespace divide_condition_l25_25427

theorem divide_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ n : ℕ, 0 < n ∧ a ∣ (b^n - n) :=
by
  sorry

end divide_condition_l25_25427


namespace find_numbers_l25_25491

theorem find_numbers (a b : ℝ) (h1 : a - b = 7.02) (h2 : a = 10 * b) : a = 7.8 ∧ b = 0.78 :=
by
  sorry

end find_numbers_l25_25491


namespace Amy_gets_fewest_cookies_l25_25523

theorem Amy_gets_fewest_cookies:
  let area_Amy := 4 * Real.pi
  let area_Ben := 9
  let area_Carl := 8
  let area_Dana := (9 / 2) * Real.pi
  let num_cookies_Amy := 1 / area_Amy
  let num_cookies_Ben := 1 / area_Ben
  let num_cookies_Carl := 1 / area_Carl
  let num_cookies_Dana := 1 / area_Dana
  num_cookies_Amy < num_cookies_Ben ∧ num_cookies_Amy < num_cookies_Carl ∧ num_cookies_Amy < num_cookies_Dana :=
by
  sorry

end Amy_gets_fewest_cookies_l25_25523


namespace total_cost_of_hats_l25_25172

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l25_25172


namespace eliminate_denominators_l25_25514

theorem eliminate_denominators (x : ℝ) :
  (4 * (2 * x - 1) - 3 * (3 * x - 4) = 12) ↔ ((2 * x - 1) / 3 - (3 * x - 4) / 4 = 1) := 
by
  sorry

end eliminate_denominators_l25_25514


namespace f_is_odd_l25_25380

open Real

def f (x : ℝ) : ℝ := x^3 + x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end f_is_odd_l25_25380


namespace max_value_of_f_l25_25206

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : 
  ∃ x : ℝ, f x = 6 / 5 ∧ ∀ y : ℝ, f y ≤ 6 / 5 :=
sorry

end max_value_of_f_l25_25206


namespace muffins_division_l25_25462

theorem muffins_division (total_muffins total_people muffins_per_person : ℕ) 
  (h1 : total_muffins = 20) (h2 : total_people = 5) (h3 : muffins_per_person = total_muffins / total_people) : 
  muffins_per_person = 4 := 
by
  sorry

end muffins_division_l25_25462


namespace sum_of_cubes_of_three_consecutive_integers_l25_25496

theorem sum_of_cubes_of_three_consecutive_integers (a : ℕ) (h : (a * a) + (a + 1) * (a + 1) + (a + 2) * (a + 2) = 2450) : a * a * a + (a + 1) * (a + 1) * (a + 1) + (a + 2) * (a + 2) * (a + 2) = 73341 :=
by
  sorry

end sum_of_cubes_of_three_consecutive_integers_l25_25496


namespace max_temperature_when_80_l25_25627

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10 * t + 60

-- State the theorem
theorem max_temperature_when_80 : ∃ t : ℝ, temperature t = 80 ∧ t = 5 + Real.sqrt 5 := 
by {
  -- Theorem proof is skipped with sorry
  sorry
}

end max_temperature_when_80_l25_25627


namespace fraction_value_l25_25395

theorem fraction_value (a b c : ℕ) (h1 : a = 2200) (h2 : b = 2096) (h3 : c = 121) :
    (a - b)^2 / c = 89 := by
  sorry

end fraction_value_l25_25395


namespace find_a_value_l25_25442

theorem find_a_value (a : ℝ) (m : ℝ) (f g : ℝ → ℝ)
  (f_def : ∀ x, f x = Real.log x / Real.log a)
  (g_def : ∀ x, g x = (2 + m) * Real.sqrt x)
  (a_pos : 0 < a) (a_neq_one : a ≠ 1)
  (max_f : ∀ x ∈ Set.Icc (1 / 2) 16, f x ≤ 4)
  (min_f : ∀ x ∈ Set.Icc (1 / 2) 16, m ≤ f x)
  (g_increasing : ∀ x y, 0 < x → x < y → g x < g y):
  a = 2 :=
sorry

end find_a_value_l25_25442


namespace odd_function_characterization_l25_25304

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l25_25304


namespace max_k_subset_l25_25177

theorem max_k_subset :
  ∃ (k : ℕ), k ≤ 16 ∧ (∀ A : Finset ℕ, A.card = k → A ⊆ {n | n ∈ Finset.range 17 \ {-1}} → 
  (∀ B ⊆ A, ∃ S : Finset (ℕ → ℕ), S.card = 2^k - 1 ∧ 
  (∀ s1 s2 : ℕ, s1 ∈ S → s2 ∈ S → s1 ≠ s2)) ∧ k = 5 
   ) := 
   sorry

end max_k_subset_l25_25177


namespace nine_sided_polygon_diagonals_l25_25050

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25050


namespace max_proj_area_of_regular_tetrahedron_l25_25512

theorem max_proj_area_of_regular_tetrahedron (a : ℝ) (h_a : a > 0) : 
    ∃ max_area : ℝ, max_area = a^2 / 2 :=
by
  existsi (a^2 / 2)
  sorry

end max_proj_area_of_regular_tetrahedron_l25_25512


namespace daughter_weight_l25_25247

def main : IO Unit :=
  IO.println s!"The weight of the daughter is 50 kg."

theorem daughter_weight :
  ∀ (G D C : ℝ), G + D + C = 110 → D + C = 60 → C = (1/5) * G → D = 50 :=
by
  intros G D C h1 h2 h3
  sorry

end daughter_weight_l25_25247


namespace bounded_f_l25_25620

theorem bounded_f (f : ℝ → ℝ) (h1 : ∀ x1 x2, |x1 - x2| ≤ 1 → |f x2 - f x1| ≤ 1)
  (h2 : f 0 = 1) : ∀ x, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end bounded_f_l25_25620


namespace find_x_l25_25459

theorem find_x (x : ℚ) (h1 : 3 * x + (4 * x - 10) = 90) : x = 100 / 7 :=
by {
  sorry
}

end find_x_l25_25459


namespace distance_Reims_to_Chaumont_l25_25645

noncomputable def distance_Chalons_Vitry : ℝ := 30
noncomputable def distance_Vitry_Chaumont : ℝ := 80
noncomputable def distance_Chaumont_SaintQuentin : ℝ := 236
noncomputable def distance_SaintQuentin_Reims : ℝ := 86
noncomputable def distance_Reims_Chalons : ℝ := 40

theorem distance_Reims_to_Chaumont :
  distance_Reims_Chalons + 
  distance_Chalons_Vitry + 
  distance_Vitry_Chaumont = 150 :=
sorry

end distance_Reims_to_Chaumont_l25_25645


namespace attraction_ticket_cost_for_parents_l25_25605

noncomputable def total_cost (children parents adults: ℕ) (entrance_cost child_attraction_cost adult_attraction_cost: ℕ) : ℕ :=
  (children + parents + adults) * entrance_cost + children * child_attraction_cost + adults * (adult_attraction_cost)

theorem attraction_ticket_cost_for_parents
  (children parents adults: ℕ) 
  (entrance_cost child_attraction_cost total_cost_of_family: ℕ) 
  (h_children: children = 4)
  (h_parents: parents = 2)
  (h_adults: adults = 1)
  (h_entrance_cost: entrance_cost = 5)
  (h_child_attraction_cost: child_attraction_cost = 2)
  (h_total_cost_of_family: total_cost_of_family = 55)
  : (total_cost children parents adults entrance_cost child_attraction_cost 4 / 3) = total_cost_of_family - (children + parents + adults) * entrance_cost - children * child_attraction_cost := 
sorry

end attraction_ticket_cost_for_parents_l25_25605


namespace initial_profit_percentage_l25_25249

-- Definitions of conditions
variables {x y : ℝ} (h1 : y > x) (h2 : 2 * y - x = 1.4 * x)

-- Proof statement in Lean
theorem initial_profit_percentage (x y : ℝ) (h1 : y > x) (h2 : 2 * y - x = 1.4 * x) :
  ((y - x) / x) * 100 = 20 :=
by sorry

end initial_profit_percentage_l25_25249


namespace maximize_product_numbers_l25_25679

theorem maximize_product_numbers (a b : ℕ) (ha : a = 96420) (hb : b = 87531) (cond: a * b = 96420 * 87531):
  b = 87531 := 
by sorry

end maximize_product_numbers_l25_25679


namespace side_length_of_square_l25_25912

theorem side_length_of_square (A : ℝ) (h : A = 81) : ∃ s : ℝ, s^2 = A ∧ s = 9 :=
by
  sorry

end side_length_of_square_l25_25912


namespace evaluate_expression_l25_25790

theorem evaluate_expression : 
  3 * (-3)^4 + 2 * (-3)^3 + (-3)^2 + 3^2 + 2 * 3^3 + 3 * 3^4 = 504 :=
by
  sorry

end evaluate_expression_l25_25790


namespace johns_hats_cost_l25_25168

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l25_25168


namespace M_diff_N_eq_l25_25728

noncomputable def A_diff_B (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

noncomputable def M : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

noncomputable def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem M_diff_N_eq : A_diff_B M N = { x | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end M_diff_N_eq_l25_25728


namespace jacoby_needs_l25_25157

-- Given conditions
def total_goal : ℤ := 5000
def job_earnings_per_hour : ℤ := 20
def total_job_hours : ℤ := 10
def cookie_price_each : ℤ := 4
def total_cookies_sold : ℤ := 24
def lottery_ticket_cost : ℤ := 10
def lottery_winning : ℤ := 500
def gift_from_sister_one : ℤ := 500
def gift_from_sister_two : ℤ := 500

-- Total money Jacoby has so far
def current_total_money : ℤ := 
  job_earnings_per_hour * total_job_hours +
  cookie_price_each * total_cookies_sold +
  lottery_winning +
  gift_from_sister_one + gift_from_sister_two -
  lottery_ticket_cost

-- The amount Jacoby needs to reach his goal
def amount_needed : ℤ := total_goal - current_total_money

-- The main statement to be proved
theorem jacoby_needs : amount_needed = 3214 := by
  -- The proof is skipped
  sorry

end jacoby_needs_l25_25157


namespace owl_cost_in_gold_l25_25828

-- Definitions for conditions
def spellbook_cost_gold := 5
def potionkit_cost_silver := 20
def num_spellbooks := 5
def num_potionkits := 3
def silver_per_gold := 9
def total_payment_silver := 537

-- Function to convert gold to silver
def gold_to_silver (gold : ℕ) : ℕ := gold * silver_per_gold

-- Function to compute total cost in silver for spellbooks and potion kits
def total_spellbook_cost_silver : ℕ :=
  gold_to_silver spellbook_cost_gold * num_spellbooks

def total_potionkit_cost_silver : ℕ :=
  potionkit_cost_silver * num_potionkits

-- Function to calculate the cost of the owl in silver
def owl_cost_silver : ℕ :=
  total_payment_silver - (total_spellbook_cost_silver + total_potionkit_cost_silver)

-- Function to convert the owl's cost from silver to gold
def owl_cost_gold : ℕ :=
  owl_cost_silver / silver_per_gold

-- The proof statement
theorem owl_cost_in_gold : owl_cost_gold = 28 :=
  by
    sorry

end owl_cost_in_gold_l25_25828


namespace calc_sum_of_digits_l25_25703

theorem calc_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10) 
(hm : 10 * 3 + x = 34) (hmy : 34 * (10 * y + 4) = 136) : x + y = 7 :=
sorry

end calc_sum_of_digits_l25_25703


namespace train_length_l25_25538

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def crossing_time : ℝ := 20
noncomputable def platform_length : ℝ := 220.032
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time

theorem train_length :
  total_distance - platform_length = 179.968 := by
  sorry

end train_length_l25_25538


namespace john_growth_l25_25466

theorem john_growth 
  (InitialHeight : ℤ)
  (GrowthRate : ℤ)
  (FinalHeight : ℤ)
  (h1 : InitialHeight = 66)
  (h2 : GrowthRate = 2)
  (h3 : FinalHeight = 72) :
  (FinalHeight - InitialHeight) / GrowthRate = 3 :=
by
  sorry

end john_growth_l25_25466


namespace curves_intersect_four_points_l25_25766

theorem curves_intersect_four_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = 4 * a^2 ∧ y = x^2 - 2 * a) → (a > 1/3)) :=
sorry

end curves_intersect_four_points_l25_25766


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25017

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25017


namespace sum_of_digits_base2_345_l25_25227

open Nat -- open natural numbers namespace

theorem sum_of_digits_base2_345 : (Nat.digits 2 345).sum = 5 := by
  sorry -- proof to be filled in later

end sum_of_digits_base2_345_l25_25227


namespace odd_function_values_l25_25332

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l25_25332


namespace diagonals_in_nonagon_l25_25136

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25136


namespace diagonals_in_nine_sided_polygon_l25_25077

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25077


namespace walk_two_dogs_for_7_minutes_l25_25452

variable (x : ℕ)

def charge_per_dog : ℕ := 20
def charge_per_minute_per_dog : ℕ := 1
def total_earnings : ℕ := 171

def charge_one_dog := charge_per_dog + charge_per_minute_per_dog * 10
def charge_three_dogs := charge_per_dog * 3 + charge_per_minute_per_dog * 9 * 3
def charge_two_dogs (x : ℕ) := charge_per_dog * 2 + charge_per_minute_per_dog * x * 2

theorem walk_two_dogs_for_7_minutes 
  (h1 : charge_one_dog = 30)
  (h2 : charge_three_dogs = 87)
  (h3 : charge_one_dog + charge_three_dogs + charge_two_dogs x = total_earnings) : 
  x = 7 :=
by
  unfold charge_one_dog charge_three_dogs charge_per_dog charge_per_minute_per_dog total_earnings at *
  sorry

end walk_two_dogs_for_7_minutes_l25_25452


namespace circumcircle_radius_min_cosA_l25_25450

noncomputable def circumcircle_radius (a b c : ℝ) (A B C : ℝ) :=
  a / (2 * (Real.sin A))

theorem circumcircle_radius_min_cosA
  (a b c A B C : ℝ)
  (h1 : a = 2)
  (h2 : Real.sin C + Real.sin B = 4 * Real.sin A)
  (h3 : a^2 + b^2 - 2 * a * b * (Real.cos A) = c^2)
  (h4 : a^2 + c^2 - 2 * a * c * (Real.cos B) = b^2)
  (h5 : b^2 + c^2 - 2 * b * c * (Real.cos C) = a^2) :
  circumcircle_radius a b c A B C = 8 * Real.sqrt 15 / 15 :=
sorry

end circumcircle_radius_min_cosA_l25_25450


namespace number_of_diagonals_l25_25088

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25088


namespace total_cost_of_hats_l25_25170

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l25_25170


namespace largest_value_among_expressions_l25_25232

def expA : ℕ := 3 + 1 + 2 + 4
def expB : ℕ := 3 * 1 + 2 + 4
def expC : ℕ := 3 + 1 * 2 + 4
def expD : ℕ := 3 + 1 + 2 * 4
def expE : ℕ := 3 * 1 * 2 * 4

theorem largest_value_among_expressions :
  expE > expA ∧ expE > expB ∧ expE > expC ∧ expE > expD :=
by
  -- Proof will go here
  sorry

end largest_value_among_expressions_l25_25232


namespace tens_digit_36_pow_12_l25_25401

theorem tens_digit_36_pow_12 : ((36 ^ 12) % 100) / 10 % 10 = 1 := 
by 
sorry

end tens_digit_36_pow_12_l25_25401


namespace amount_each_girl_receives_l25_25371

theorem amount_each_girl_receives (total_amount : ℕ) (total_children : ℕ) (amount_per_boy : ℕ) (num_boys : ℕ) (remaining_amount : ℕ) (num_girls : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460) 
  (h2 : total_children = 41)
  (h3 : amount_per_boy = 12)
  (h4 : num_boys = 33)
  (h5 : remaining_amount = total_amount - num_boys * amount_per_boy)
  (h6 : num_girls = total_children - num_boys)
  (h7 : amount_per_girl = remaining_amount / num_girls) :
  amount_per_girl = 8 := 
sorry

end amount_each_girl_receives_l25_25371


namespace find_numerical_value_l25_25348

-- Define the conditions
variables {x y z : ℝ}
axiom h1 : 3 * x - 4 * y - 2 * z = 0
axiom h2 : x + 4 * y - 20 * z = 0
axiom h3 : z ≠ 0

-- State the goal
theorem find_numerical_value : (x^2 + 4 * x * y) / (y^2 + z^2) = 2.933 :=
by
  sorry

end find_numerical_value_l25_25348


namespace distinct_real_solutions_l25_25617

theorem distinct_real_solutions
  (a b c d e : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x₁ x₂ x₃ x₄ : ℝ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - d) +
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - e) +
    (x₁ - a) * (x₁ - b) * (x₁ - d) * (x₁ - e) +
    (x₁ - a) * (x₁ - c) * (x₁ - d) * (x₁ - e) +
    (x₁ - b) * (x₁ - c) * (x₁ - d) * (x₁ - e) = 0 ∧
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - d) +
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - e) +
    (x₂ - a) * (x₂ - b) * (x₂ - d) * (x₂ - e) +
    (x₂ - a) * (x₂ - c) * (x₂ - d) * (x₂ - e) +
    (x₂ - b) * (x₂ - c) * (x₂ - d) * (x₂ - e) = 0 ∧
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - d) +
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - e) +
    (x₃ - a) * (x₃ - b) * (x₃ - d) * (x₃ - e) +
    (x₃ - a) * (x₃ - c) * (x₃ - d) * (x₃ - e) +
    (x₃ - b) * (x₃ - c) * (x₃ - d) * (x₃ - e) = 0 ∧
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - d) +
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - e) +
    (x₄ - a) * (x₄ - b) * (x₄ - d) * (x₄ - e) +
    (x₄ - a) * (x₄ - c) * (x₄ - d) * (x₄ - e) +
    (x₄ - b) * (x₄ - c) * (x₄ - d) * (x₄ - e) = 0 :=
  sorry

end distinct_real_solutions_l25_25617


namespace symmetric_points_sum_l25_25705

theorem symmetric_points_sum (n m : ℤ) 
  (h₁ : (3 : ℤ) = m)
  (h₂ : n = (-5 : ℤ)) : 
  m + n = (-2 : ℤ) := 
by 
  sorry

end symmetric_points_sum_l25_25705


namespace area_of_region_l25_25219

-- Define the condition: the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 10 * x - 4 * y + 9 = 0

-- State the theorem: the area of the region defined by the equation is 20π
theorem area_of_region : ∀ x y : ℝ, region_equation x y → ∃ A : ℝ, A = 20 * Real.pi :=
by sorry

end area_of_region_l25_25219


namespace multiply_difference_of_cubes_l25_25187

def multiply_and_simplify (x : ℝ) : ℝ :=
  (x^4 + 25 * x^2 + 625) * (x^2 - 25)

theorem multiply_difference_of_cubes (x : ℝ) :
  multiply_and_simplify x = x^6 - 15625 :=
by
  sorry

end multiply_difference_of_cubes_l25_25187


namespace students_taking_neither_l25_25668

def total_students : ℕ := 1200
def music_students : ℕ := 60
def art_students : ℕ := 80
def sports_students : ℕ := 30
def music_and_art_students : ℕ := 25
def music_and_sports_students : ℕ := 15
def art_and_sports_students : ℕ := 20
def all_three_students : ℕ := 10

theorem students_taking_neither :
  total_students - (music_students + art_students + sports_students 
  - music_and_art_students - music_and_sports_students - art_and_sports_students 
  + all_three_students) = 1080 := sorry

end students_taking_neither_l25_25668


namespace impossible_partition_10x10_square_l25_25608

theorem impossible_partition_10x10_square :
  ¬ ∃ (x y : ℝ), (x - y = 1) ∧ (x * y = 1) ∧ (∃ (n m : ℕ), 10 = n * x + m * y ∧ n + m = 100) :=
by
  sorry

end impossible_partition_10x10_square_l25_25608


namespace phil_quarters_collection_l25_25370

theorem phil_quarters_collection
    (initial_quarters : ℕ)
    (doubled_quarters : ℕ)
    (additional_quarters_per_month : ℕ)
    (total_quarters_end_of_second_year : ℕ)
    (quarters_collected_every_third_month : ℕ)
    (total_quarters_end_of_third_year : ℕ)
    (remaining_quarters_after_loss : ℕ)
    (quarters_left : ℕ) :
    initial_quarters = 50 →
    doubled_quarters = 2 * initial_quarters →
    additional_quarters_per_month = 3 →
    total_quarters_end_of_second_year = doubled_quarters + 12 * additional_quarters_per_month →
    total_quarters_end_of_third_year = total_quarters_end_of_second_year + 4 * quarters_collected_every_third_month →
    remaining_quarters_after_loss = (3 / 4 : ℚ) * total_quarters_end_of_third_year → 
    quarters_left = 105 →
    quarters_collected_every_third_month = 1 := 
by
  sorry

end phil_quarters_collection_l25_25370


namespace systematic_sampling_interval_people_l25_25252

theorem systematic_sampling_interval_people (total_employees : ℕ) (selected_employees : ℕ) (start_interval : ℕ) (end_interval : ℕ)
  (h_total : total_employees = 420)
  (h_selected : selected_employees = 21)
  (h_start_end : start_interval = 281)
  (h_end : end_interval = 420)
  : (end_interval - start_interval + 1) / (total_employees / selected_employees) = 7 := 
by
  -- sorry placeholder for proof
  sorry

end systematic_sampling_interval_people_l25_25252


namespace evaluate_expression_l25_25680

theorem evaluate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  ((a^2 + b + c)^2 - (a^2 - b - c)^2) = 108 :=
by
  sorry

end evaluate_expression_l25_25680


namespace sum_a2_a4_a6_l25_25382

theorem sum_a2_a4_a6 : ∀ {a : ℕ → ℕ}, (∀ i, a (i+1) = (1 / 2 : ℝ) * a i) → a 2 = 32 → a 2 + a 4 + a 6 = 42 :=
by
  intros a ha h2
  sorry

end sum_a2_a4_a6_l25_25382


namespace cost_of_bananas_is_two_l25_25355

variable (B : ℝ)

theorem cost_of_bananas_is_two (h : 1.20 * (3 + B) = 6) : B = 2 :=
by
  sorry

end cost_of_bananas_is_two_l25_25355


namespace sin_cos_alpha_frac_l25_25692

theorem sin_cos_alpha_frac (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 := 
by
  sorry

end sin_cos_alpha_frac_l25_25692


namespace nine_sided_polygon_diagonals_l25_25121

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25121


namespace determine_a_b_odd_function_l25_25326

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l25_25326


namespace union_sets_l25_25293

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

theorem union_sets : A ∪ B = {x | x ≥ 1} :=
  by
    sorry

end union_sets_l25_25293


namespace minimum_distance_l25_25880

noncomputable def minDistPointOnLineToPointOnCircle : Real := 
  let line := λy, y = 2
  let circle := λx y, (x - 1)^2 + y^2 = 1
  -- The minimum distance we want to prove
  1

-- The statement of the problem
theorem minimum_distance : ∃ (p q : ℝ × ℝ), 
  (p.snd = 2) ∧ ((q.fst - 1)^2 + q.snd^2 = 1) ∧ 
  ∀(r s : ℝ × ℝ), (r.snd = 2) ∧ ((s.fst - 1)^2 + s.snd^2 = 1) -> 
    dist ℝ ℝ p q ≤ dist ℝ ℝ r s ∧ dist ℝ ℝ p q = 1 :=
by
  -- Proof will be provided
  sorry

end minimum_distance_l25_25880


namespace balloon_highest_elevation_l25_25162

theorem balloon_highest_elevation
  (time_rise1 time_rise2 time_descent : ℕ)
  (rate_rise rate_descent : ℕ)
  (t1 : time_rise1 = 15)
  (t2 : time_rise2 = 15)
  (t3 : time_descent = 10)
  (rr : rate_rise = 50)
  (rd : rate_descent = 10)
  : (time_rise1 * rate_rise - time_descent * rate_descent + time_rise2 * rate_rise) = 1400 := 
by
  sorry

end balloon_highest_elevation_l25_25162


namespace proof_problem_l25_25946

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem proof_problem
  (a : ℝ) (b : ℝ) (x : ℝ)
  (h₀ : 0 ≤ a)
  (h₁ : a ≤ 1 / 2)
  (h₂ : b = 1)
  (h₃ : 0 ≤ x) :
  (1 / f x) + (x / g x a b) ≥ 1 := by
    sorry

end proof_problem_l25_25946


namespace union_S_T_l25_25364

def S : Set ℝ := { x | 3 < x ∧ x ≤ 6 }
def T : Set ℝ := { x | x^2 - 4*x - 5 ≤ 0 }

theorem union_S_T : S ∪ T = { x | -1 ≤ x ∧ x ≤ 6 } := 
by 
  sorry

end union_S_T_l25_25364


namespace f_periodic_with_period_4a_l25_25739

-- Definitions 'f' and 'g' (functions on real numbers), and the given conditions:
variables {a : ℝ} (f g : ℝ → ℝ)
-- Condition on a: a ≠ 0
variable (ha : a ≠ 0)

-- Given conditions
variable (hf0 : f 0 = 1) (hga : g a = 1) (h_odd_g : ∀ x : ℝ, g x = -g (-x))

-- Functional equation
variable (h_func_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y)

-- The theorem stating that f is periodic with period 4a
theorem f_periodic_with_period_4a : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_periodic_with_period_4a_l25_25739


namespace probability_last_passenger_own_seat_is_half_l25_25419

open Classical

-- Define the behavior and probability question:

noncomputable def probability_last_passenger_own_seat (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / 2

-- The main theorem stating the probability for an arbitrary number of passengers n
-- The theorem that needs to be proved:
theorem probability_last_passenger_own_seat_is_half (n : ℕ) (h : n > 0) : 
  probability_last_passenger_own_seat n = 1 / 2 :=
by sorry

end probability_last_passenger_own_seat_is_half_l25_25419


namespace find_n_value_l25_25460

theorem find_n_value (m n k : ℝ) (h1 : n = k / m) (h2 : m = k / 2) (h3 : k ≠ 0): n = 2 :=
sorry

end find_n_value_l25_25460


namespace distinct_lengths_from_E_to_DF_l25_25482

noncomputable def distinct_integer_lengths (DE EF: ℕ) : ℕ :=
if h : DE = 15 ∧ EF = 36 then 24 else 0

theorem distinct_lengths_from_E_to_DF :
  distinct_integer_lengths 15 36 = 24 :=
by {
  sorry
}

end distinct_lengths_from_E_to_DF_l25_25482


namespace parabola_perpendicular_bisector_intersects_x_axis_l25_25948

theorem parabola_perpendicular_bisector_intersects_x_axis
  (x1 y1 x2 y2 : ℝ) 
  (A_on_parabola : y1^2 = 2 * x1)
  (B_on_parabola : y2^2 = 2 * x2) 
  (k m : ℝ) 
  (AB_line : ∀ x y, y = k * x + m)
  (k_not_zero : k ≠ 0) 
  (k_m_condition : (1 / k^2) - (m / k) > 0) :
  ∃ x0 : ℝ, x0 = (1 / k^2) - (m / k) + 1 ∧ x0 > 1 :=
by
  sorry

end parabola_perpendicular_bisector_intersects_x_axis_l25_25948


namespace nine_sided_polygon_diagonals_l25_25029

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25029


namespace two_truth_tellers_are_B_and_C_l25_25488

-- Definitions of students and their statements
def A_statement_false (A_said : Prop) (A_truth_teller : Prop) := ¬A_said = A_truth_teller
def B_statement_true (B_said : Prop) (B_truth_teller : Prop) := B_said = B_truth_teller
def C_statement_true (C_said : Prop) (C_truth_teller : Prop) := C_said = C_truth_teller
def D_statement_false (D_said : Prop) (D_truth_teller : Prop) := ¬D_said = D_truth_teller

-- Given statements
def A_said := ¬ (False : Prop)
def B_said := True
def C_said := B_said ∨ D_statement_false True True
def D_said := False

-- Define who is telling the truth
def A_truth_teller := False
def B_truth_teller := True
def C_truth_teller := True
def D_truth_teller := False

-- Proof problem statement
theorem two_truth_tellers_are_B_and_C :
  (A_statement_false A_said A_truth_teller) ∧
  (B_statement_true B_said B_truth_teller) ∧
  (C_statement_true C_said C_truth_teller) ∧
  (D_statement_false D_said D_truth_teller) →
  ((A_truth_teller = False) ∧
  (B_truth_teller = True) ∧
  (C_truth_teller = True) ∧
  (D_truth_teller = False)) := 
by {
  sorry
}

end two_truth_tellers_are_B_and_C_l25_25488


namespace diagonals_in_regular_nine_sided_polygon_l25_25067

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25067


namespace determine_b_value_l25_25945

theorem determine_b_value 
  (a : ℝ) 
  (b : ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1) 
  (h₂ : 2 * a^(2 - b) + 1 = 3) : 
  b = 2 := 
by 
  sorry

end determine_b_value_l25_25945


namespace simplify_sum_l25_25798

theorem simplify_sum :
  -2^2004 + (-2)^2005 + 2^2006 - 2^2007 = -2^2004 - 2^2005 + 2^2006 - 2^2007 :=
by
  sorry

end simplify_sum_l25_25798


namespace joyful_not_blue_l25_25387

variables {Snakes : Type} 
variables (isJoyful : Snakes → Prop) (isBlue : Snakes → Prop)
variables (canMultiply : Snakes → Prop) (canDivide : Snakes → Prop)

-- Conditions
axiom H1 : ∀ s : Snakes, isJoyful s → canMultiply s
axiom H2 : ∀ s : Snakes, isBlue s → ¬ canDivide s
axiom H3 : ∀ s : Snakes, ¬ canDivide s → ¬ canMultiply s

theorem joyful_not_blue (s : Snakes) : isJoyful s → ¬ isBlue s :=
by sorry

end joyful_not_blue_l25_25387


namespace regular_nine_sided_polygon_diagonals_l25_25099

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25099


namespace probability_same_fruits_l25_25443

theorem probability_same_fruits :
  let fruits : Finset ℕ := {1, 2, 3, 4},
      n := fruits.card,
      k := 2
  in (∑ _ in (fruits.subsets k), 1 : ℚ) / ((∑ _ in (fruits.subsets k), 1 : ℚ) * (∑ _ in (fruits.subsets k), 1 : ℚ)) = 1 / 6 :=
by
  sorry

end probability_same_fruits_l25_25443


namespace problem_l25_25809

open Set

theorem problem (a b : ℝ) :
  (A = {x | x^2 - 2*x - 3 > 0}) →
  (B = {x | x^2 + a*x + b ≤ 0}) →
  (A ∪ B = univ) →
  (A ∩ B = Ioo 3 4) →
  a + b = -7 :=
by
  intros hA hB hUnion hIntersection
  sorry

end problem_l25_25809


namespace number_of_committees_l25_25418

theorem number_of_committees (physics_men : Finset ℕ) (physics_women : Finset ℕ)
                             (chemistry_men : Finset ℕ) (chemistry_women : Finset ℕ)
                             (biology_men : Finset ℕ) (biology_women : Finset ℕ)
                             (h_pm : physics_men.card = 3) (h_pw : physics_women.card = 3)
                             (h_cm : chemistry_men.card = 3) (h_cw : chemistry_women.card = 3)
                             (h_bm : biology_men.card = 3) (h_bw : biology_women.card = 3) :
  (number_of_ways physics_men physics_women) * (number_of_ways chemistry_men chemistry_women) * (number_of_ways biology_men biology_women) +
  6 * (special_case_num physics_men physics_women chemistry_men chemistry_women biology_men biology_women) = 1215 :=
sorry

end number_of_committees_l25_25418


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25014

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25014


namespace bounded_infinite_sequence_l25_25812

noncomputable def sequence_x (n : ℕ) : ℝ :=
  4 * (Real.sqrt 2 * n - ⌊Real.sqrt 2 * n⌋)

theorem bounded_infinite_sequence (a : ℝ) (h : a > 1) :
  ∀ i j : ℕ, i ≠ j → (|sequence_x i - sequence_x j| * |(i - j : ℝ)|^a) ≥ 1 := 
by
  intros i j h_ij
  sorry

end bounded_infinite_sequence_l25_25812


namespace find_a_and_b_to_make_f_odd_l25_25347

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l25_25347


namespace sara_staircase_l25_25197

theorem sara_staircase (n : ℕ) (h : 2 * n * (n + 1) = 360) : n = 13 :=
sorry

end sara_staircase_l25_25197


namespace inequality_for_positive_n_and_x_l25_25745

theorem inequality_for_positive_n_and_x (n : ℕ) (x : ℝ) (hn : n > 0) (hx : x > 0) :
  (x^(2 * n - 1) - 1) / (2 * n - 1) ≤ (x^(2 * n) - 1) / (2 * n) :=
by sorry

end inequality_for_positive_n_and_x_l25_25745


namespace inequality_proof_l25_25474

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (h_condition : a * b + b * c + c * d + d * a = 1) :
    (a ^ 3 / (b + c + d)) + (b ^ 3 / (c + d + a)) + (c ^ 3 / (a + b + d)) + (d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l25_25474


namespace nine_sided_polygon_diagonals_l25_25125

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25125


namespace simplify_expression_l25_25984

theorem simplify_expression (z : ℝ) : (5 - 2*z^2) - (4*z^2 - 7) = 12 - 6*z^2 :=
by
  sorry

end simplify_expression_l25_25984


namespace candles_time_l25_25389

/-- Prove that if two candles of equal length are lit at a certain time,
and by 6 PM one of the stubs is three times the length of the other,
the correct time to light the candles is 4:00 PM. -/

theorem candles_time :
  ∀ (ℓ : ℝ) (t : ℝ),
  (∀ t1 t2 : ℝ, t = t1 + t2 → 
    (180 - t1) = 3 * (300 - t2) / 3 → 
    18 <= 6 ∧ 0 <= t → ℓ / 180 * (180 - (t - 180)) = 3 * (ℓ / 300 * (300 - (6 - t))) →
    t = 4
  ) := 
by 
  sorry

end candles_time_l25_25389


namespace probability_of_D_l25_25240

theorem probability_of_D (pA pB pC pD : ℚ)
  (hA : pA = 1/4)
  (hB : pB = 1/3)
  (hC : pC = 1/6)
  (hTotal : pA + pB + pC + pD = 1) : pD = 1/4 :=
by
  have hTotal_before_D : pD = 1 - (pA + pB + pC) := by sorry
  sorry

end probability_of_D_l25_25240


namespace fraction_simplification_l25_25954

theorem fraction_simplification (x y : ℚ) (h1 : x = 4) (h2 : y = 5) : 
  (1 / y) / (1 / x) = 4 / 5 :=
by
  sorry

end fraction_simplification_l25_25954


namespace cos_sub_sin_alpha_l25_25295

theorem cos_sub_sin_alpha (alpha : ℝ) (h1 : π / 4 < alpha) (h2 : alpha < π / 2)
    (h3 : Real.sin (2 * alpha) = 24 / 25) : Real.cos alpha - Real.sin alpha = -1 / 5 :=
by
  sorry

end cos_sub_sin_alpha_l25_25295


namespace tan_sum_pi_over_12_l25_25993

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l25_25993


namespace diagonals_in_regular_nine_sided_polygon_l25_25007

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25007


namespace coordinates_of_N_l25_25687

-- Define the given conditions
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)
def minusThreeA : ℝ × ℝ := (-3, 6)
def vectorMN (N : ℝ × ℝ) : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define the required goal
theorem coordinates_of_N (N : ℝ × ℝ) : vectorMN N = minusThreeA → N = (2, 0) :=
by
  sorry

end coordinates_of_N_l25_25687


namespace isosceles_triangle_base_l25_25752

theorem isosceles_triangle_base (b : ℝ) (h1 : 7 + 7 + b = 20) : b = 6 :=
by {
    sorry
}

end isosceles_triangle_base_l25_25752


namespace negation_of_proposition_l25_25878

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end negation_of_proposition_l25_25878


namespace muffins_per_person_l25_25464

-- Definitions based on conditions
def total_friends : ℕ := 4
def total_people : ℕ := 1 + total_friends
def total_muffins : ℕ := 20

-- Theorem statement for the proof
theorem muffins_per_person : total_muffins / total_people = 4 := by
  sorry

end muffins_per_person_l25_25464


namespace triangle_at_most_one_obtuse_l25_25397

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90 → B + C < 90) (h3 : B > 90 → A + C < 90) (h4 : C > 90 → A + B < 90) :
  ¬ (A > 90 ∧ B > 90 ∨ B > 90 ∧ C > 90 ∨ A > 90 ∧ C > 90) :=
by
  sorry

end triangle_at_most_one_obtuse_l25_25397


namespace top_card_probability_l25_25216

theorem top_card_probability :
  let total_cards := 104
  let favorable_outcomes := 4
  let probability := (favorable_outcomes : ℚ) / total_cards
  probability = 1 / 26 :=
by
  sorry

end top_card_probability_l25_25216


namespace f_g_2_eq_neg_19_l25_25947

def f (x : ℝ) : ℝ := 5 - 4 * x

def g (x : ℝ) : ℝ := x^2 + 2

theorem f_g_2_eq_neg_19 : f (g 2) = -19 := 
by
  -- The proof is omitted
  sorry

end f_g_2_eq_neg_19_l25_25947


namespace smallest_number_of_students_l25_25599

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l25_25599


namespace arctan_sum_pi_over_two_l25_25793

theorem arctan_sum_pi_over_two : 
  Real.arctan (3 / 7) + Real.arctan (7 / 3) = Real.pi / 2 := 
by sorry

end arctan_sum_pi_over_two_l25_25793


namespace nine_sided_polygon_diagonals_l25_25124

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25124


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l25_25998

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l25_25998


namespace slope_points_eq_l25_25568

theorem slope_points_eq (m : ℚ) (h : ((m + 2) / (3 - m) = 2)) : m = 4 / 3 :=
sorry

end slope_points_eq_l25_25568


namespace sequence_properties_sum_Tn_l25_25179

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 2^(n - 1)
noncomputable def c_n (n : ℕ) : ℤ := (2 * n - 1) / 2^(n - 1)
noncomputable def T_n (n : ℕ) : ℤ := 6 - (2 * n + 3) / 2^(n - 1)

theorem sequence_properties : (d = 2) → (S₁₀ = 100) → 
  (∀ n : ℕ, a_n n = 2 * n - 1) ∧ (∀ n : ℕ, b_n n = 2^(n - 1)) := by
  sorry

theorem sum_Tn : (d > 1) → 
  (∀ n : ℕ, T_n n = 6 - (2 * n + 3) / 2^(n - 1)) := by
  sorry

end sequence_properties_sum_Tn_l25_25179


namespace class_student_count_l25_25200

-- Statement: Prove that under the given conditions, the number of students in the class is 19.
theorem class_student_count (n : ℕ) (avg_students_age : ℕ) (teacher_age : ℕ) (avg_with_teacher : ℕ):
  avg_students_age = 20 → 
  teacher_age = 40 → 
  avg_with_teacher = 21 → 
  21 * (n + 1) = 20 * n + 40 → 
  n = 19 := 
by 
  intros h1 h2 h3 h4 
  sorry

end class_student_count_l25_25200


namespace simplify_and_calculate_expression_l25_25373

variable (a b : ℤ)

theorem simplify_and_calculate_expression (h_a : a = -3) (h_b : b = -2) :
  (a + b) * (b - a) + (2 * a^2 * b - a^3) / (-a) = -8 :=
by
  -- We include the proof steps here to achieve the final result.
  sorry

end simplify_and_calculate_expression_l25_25373


namespace dave_time_correct_l25_25678

-- Definitions for the given conditions
def chuck_time (dave_time : ℕ) := 5 * dave_time
def erica_time (chuck_time : ℕ) := chuck_time + (3 * chuck_time / 10)
def erica_fixed_time := 65

-- Statement to prove
theorem dave_time_correct : ∃ (dave_time : ℕ), erica_time (chuck_time dave_time) = erica_fixed_time ∧ dave_time = 10 := by
  sorry

end dave_time_correct_l25_25678


namespace geom_seq_sum_is_15_l25_25365

theorem geom_seq_sum_is_15 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 1) (hq : q = -2) (h_geom : ∀ n, a (n + 1) = a n * q) :
  a 1 + |a 2| + a 3 + |a 4| = 15 :=
by
  sorry

end geom_seq_sum_is_15_l25_25365


namespace car_rental_cost_l25_25801

theorem car_rental_cost (daily_rent : ℕ) (rent_duration : ℕ) (mileage_rate : ℚ) (mileage : ℕ) (total_cost : ℕ) :
  daily_rent = 30 → rent_duration = 5 → mileage_rate = 0.25 → mileage = 500 → total_cost = 275 :=
by
  intros hd hr hm hl
  sorry

end car_rental_cost_l25_25801


namespace nine_sided_polygon_diagonals_l25_25032

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25032


namespace value_of_a_minus_b_l25_25143

theorem value_of_a_minus_b (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l25_25143


namespace triangle_acute_l25_25582

theorem triangle_acute (A B C : ℝ) (h1 : A = 2 * (180 / 9)) (h2 : B = 3 * (180 / 9)) (h3 : C = 4 * (180 / 9)) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_acute_l25_25582


namespace problem_statement_l25_25736

theorem problem_statement (w x y z : ℕ) (h : 2^w * 3^x * 5^y * 7^z = 882) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
sorry

end problem_statement_l25_25736


namespace sweater_markup_percentage_l25_25231

variables (W R : ℝ)
variables (h1 : 0.30 * R = 1.40 * W)

theorem sweater_markup_percentage :
  (R = (1.40 / 0.30) * W) →
  (R - W) / W * 100 = 367 := 
by
  intro hR
  sorry

end sweater_markup_percentage_l25_25231


namespace average_score_l25_25400

theorem average_score 
  (total_students : ℕ)
  (assigned_day_students_pct : ℝ)
  (makeup_day_students_pct : ℝ)
  (assigned_day_avg_score : ℝ)
  (makeup_day_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_students_pct = 0.70)
  (h3 : makeup_day_students_pct = 0.30)
  (h4 : assigned_day_avg_score = 0.60)
  (h5 : makeup_day_avg_score = 0.90) :
  (0.70 * 100 * 0.60 + 0.30 * 100 * 0.90) / 100 = 0.69 := 
sorry


end average_score_l25_25400


namespace range_of_a_l25_25146

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 0 then (x - a) ^ 2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ f 0 a) ↔ 0 ≤ a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l25_25146


namespace chocolate_bars_per_box_is_25_l25_25244

-- Define the conditions
def total_chocolate_bars : Nat := 400
def total_small_boxes : Nat := 16

-- Define the statement to be proved
def chocolate_bars_per_small_box : Nat := total_chocolate_bars / total_small_boxes

theorem chocolate_bars_per_box_is_25
  (h1 : total_chocolate_bars = 400)
  (h2 : total_small_boxes = 16) :
  chocolate_bars_per_small_box = 25 :=
by
  -- proof will go here
  sorry

end chocolate_bars_per_box_is_25_l25_25244


namespace diagonals_in_nonagon_l25_25128

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25128


namespace disproves_proposition_l25_25285

theorem disproves_proposition (a b : ℤ) (h₁ : a = -4) (h₂ : b = 3) : (a^2 > b^2) ∧ ¬ (a > b) :=
by
  sorry

end disproves_proposition_l25_25285


namespace platform_length_is_500_l25_25665

-- Define the length of the train, the time to cross a tree, and the time to cross a platform as given conditions
def train_length := 1500 -- in meters
def time_to_cross_tree := 120 -- in seconds
def time_to_cross_platform := 160 -- in seconds

-- Define the speed based on the train crossing the tree
def train_speed := train_length / time_to_cross_tree -- in meters/second

-- Define the total distance covered when crossing the platform
def total_distance_crossing_platform (platform_length : ℝ) := train_length + platform_length

-- State the main theorem to prove the platform length is 500 meters
theorem platform_length_is_500 (platform_length : ℝ) :
  (train_speed * time_to_cross_platform = total_distance_crossing_platform platform_length) → platform_length = 500 :=
by
  sorry

end platform_length_is_500_l25_25665


namespace hats_cost_l25_25166

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l25_25166


namespace local_minimum_point_l25_25288

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_point (a : ℝ) (h : ∃ δ > 0, ∀ x, abs (x - a) < δ → f x ≥ f a) : a = 2 :=
by
  sorry

end local_minimum_point_l25_25288


namespace solve_pears_and_fruits_l25_25843

noncomputable def pears_and_fruits_problem : Prop :=
  ∃ (x y : ℕ), x + y = 1000 ∧ (11 * x) * (1/9 : ℚ) + (4 * y) * (1/7 : ℚ) = 999

theorem solve_pears_and_fruits :
  pears_and_fruits_problem := by
  sorry

end solve_pears_and_fruits_l25_25843


namespace solution_value_a_l25_25834

theorem solution_value_a (x a : ℝ) (h₁ : x = 2) (h₂ : 2 * x + a = 3) : a = -1 :=
by
  -- Proof goes here
  sorry

end solution_value_a_l25_25834


namespace diagonals_in_regular_nine_sided_polygon_l25_25073

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25073


namespace smallest_solution_of_quartic_l25_25895

theorem smallest_solution_of_quartic :
  ∃ x : ℝ, x^4 - 40*x^2 + 144 = 0 ∧ ∀ y : ℝ, (y^4 - 40*y^2 + 144 = 0) → x ≤ y :=
sorry

end smallest_solution_of_quartic_l25_25895


namespace first_car_departure_time_l25_25916

variable (leave_time : Nat) -- in minutes past 8:00 am

def speed : Nat := 60 -- km/h
def firstCarTimeAt32 : Nat := 32 -- minutes since 8:00 am
def secondCarFactorAt32 : Nat := 3
def firstCarTimeAt39 : Nat := 39 -- minutes since 8:00 am
def secondCarFactorAt39 : Nat := 2

theorem first_car_departure_time :
  let firstCarSpeed := (60 / 60 : Nat) -- km/min
  let d1_32 := firstCarSpeed * firstCarTimeAt32
  let d2_32 := firstCarSpeed * (firstCarTimeAt32 - leave_time)
  let d1_39 := firstCarSpeed * firstCarTimeAt39
  let d2_39 := firstCarSpeed * (firstCarTimeAt39 - leave_time)
  d1_32 = secondCarFactorAt32 * d2_32 →
  d1_39 = secondCarFactorAt39 * d2_39 →
  leave_time = 11 :=
by
  intros h1 h2
  sorry

end first_car_departure_time_l25_25916


namespace solve_for_y_l25_25767

theorem solve_for_y (y : ℝ) (h : 3 / y + 4 / y / (6 / y) = 1.5) : y = 3.6 :=
sorry

end solve_for_y_l25_25767


namespace diagonals_in_regular_nine_sided_polygon_l25_25004

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25004


namespace nine_sided_polygon_diagonals_count_l25_25026

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25026


namespace jacoby_needs_l25_25158

-- Given conditions
def total_goal : ℤ := 5000
def job_earnings_per_hour : ℤ := 20
def total_job_hours : ℤ := 10
def cookie_price_each : ℤ := 4
def total_cookies_sold : ℤ := 24
def lottery_ticket_cost : ℤ := 10
def lottery_winning : ℤ := 500
def gift_from_sister_one : ℤ := 500
def gift_from_sister_two : ℤ := 500

-- Total money Jacoby has so far
def current_total_money : ℤ := 
  job_earnings_per_hour * total_job_hours +
  cookie_price_each * total_cookies_sold +
  lottery_winning +
  gift_from_sister_one + gift_from_sister_two -
  lottery_ticket_cost

-- The amount Jacoby needs to reach his goal
def amount_needed : ℤ := total_goal - current_total_money

-- The main statement to be proved
theorem jacoby_needs : amount_needed = 3214 := by
  -- The proof is skipped
  sorry

end jacoby_needs_l25_25158


namespace handshaking_pairs_l25_25961

-- Definition of the problem: Given 8 people, pair them up uniquely and count the ways modulo 1000
theorem handshaking_pairs (N : ℕ) (H : N=105) : (N % 1000) = 105 :=
by {
  -- The proof is omitted.
  sorry
}

end handshaking_pairs_l25_25961


namespace rectangular_prism_volume_l25_25780

theorem rectangular_prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : a * c = 6) (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := 
sorry

end rectangular_prism_volume_l25_25780


namespace John_Anna_total_eBooks_l25_25969

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end John_Anna_total_eBooks_l25_25969


namespace nine_sided_polygon_diagonals_l25_25108

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25108


namespace solve_for_x_l25_25749

noncomputable def proof (x : ℚ) : Prop :=
  (x + 6) / (x - 4) = (x - 7) / (x + 2)

theorem solve_for_x (x : ℚ) (h : proof x) : x = 16 / 19 :=
by
  sorry

end solve_for_x_l25_25749


namespace car_count_l25_25208

theorem car_count (x y : ℕ) (h1 : x + y = 36) (h2 : 6 * x + 4 * y = 176) :
  x = 16 ∧ y = 20 :=
by
  sorry

end car_count_l25_25208


namespace part1_part2_part3_l25_25181

def A (x : ℝ) : Prop := x^2 - x - 2 > 0
def B (x : ℝ) : Prop := 3 - |x| ≥ 0
def C (x : ℝ) (p : ℝ) : Prop := 4 * x + p < 0

theorem part1 : 
  {x : ℝ | A x} ∩ {x | B x} = {x | -3 ≤ x ∧ x < -1 ∨ 2 < x ∧ x ≤ 3} :=
sorry

theorem part2 : 
  {x : ℝ | A x} ∪ {x | B x} = set.univ :=
sorry

theorem part3 (p : ℝ) : 
  (∀ x, C x p → A x) → p ≥ 4 :=
sorry

end part1_part2_part3_l25_25181


namespace probability_of_white_ball_l25_25908

theorem probability_of_white_ball (red_balls white_balls : ℕ) (draws : ℕ)
    (h_red : red_balls = 4) (h_white : white_balls = 2) (h_draws : draws = 2) :
    ((4 * 2 + 1) / 15 : ℚ) = 3 / 5 := by sorry

end probability_of_white_ball_l25_25908


namespace profit_percentage_l25_25706

/-- If the cost price is 81% of the selling price, then the profit percentage is approximately 23.46%. -/
theorem profit_percentage (SP CP: ℝ) (h : CP = 0.81 * SP) : 
  (SP - CP) / CP * 100 = 23.46 := 
sorry

end profit_percentage_l25_25706


namespace find_ab_l25_25194

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, (3 * x - a) * (2 * x + 5) - x = 6 * x^2 + 2 * (5 * x - b)) → a = 2 ∧ b = 5 :=
by
  intro h
  -- We assume the condition holds for all x
  sorry -- Proof not needed as per instructions

end find_ab_l25_25194


namespace Karlsson_eats_more_than_half_l25_25726

open Real

theorem Karlsson_eats_more_than_half
  (D : ℝ) (S : ℕ → ℝ)
  (a b : ℕ → ℝ)
  (cut_and_eat : ∀ n, S (n + 1) = S n - (S n * a n) / (a n + b n))
  (side_conditions : ∀ n, max (a n) (b n) ≤ D) :
  ∃ n, S n < (S 0) / 2 := sorry

end Karlsson_eats_more_than_half_l25_25726


namespace John_height_l25_25614

open Real

variable (John Mary Tom Angela Helen Amy Becky Carl : ℝ)

axiom h1 : John = 1.5 * Mary
axiom h2 : Mary = 2 * Tom
axiom h3 : Tom = Angela - 70
axiom h4 : Angela = Helen + 4
axiom h5 : Helen = Amy + 3
axiom h6 : Amy = 1.2 * Becky
axiom h7 : Becky = 2 * Carl
axiom h8 : Carl = 120

theorem John_height : John = 675 := by
  sorry

end John_height_l25_25614


namespace nine_sided_polygon_diagonals_l25_25049

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25049


namespace find_a_and_b_to_make_f_odd_l25_25346

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l25_25346


namespace sum_of_ages_l25_25760

-- Define Henry's and Jill's present ages
def Henry_age : ℕ := 23
def Jill_age : ℕ := 17

-- Define the condition that 11 years ago, Henry was twice the age of Jill
def condition_11_years_ago : Prop := (Henry_age - 11) = 2 * (Jill_age - 11)

-- Theorem statement: sum of Henry's and Jill's present ages is 40
theorem sum_of_ages : Henry_age + Jill_age = 40 :=
by
  -- Placeholder for proof
  sorry

end sum_of_ages_l25_25760


namespace parabola_c_value_l25_25751

theorem parabola_c_value :
  ∃ a b c : ℝ, (∀ y : ℝ, 4 = a * (3 : ℝ)^2 + b * 3 + c ∧ 2 = a * 5^2 + b * 5 + c ∧ c = -1 / 2) :=
by
  sorry

end parabola_c_value_l25_25751


namespace parity_equiv_l25_25737

open Nat

theorem parity_equiv (p q : ℕ) : (Even (p^3 - q^3) ↔ Even (p + q)) :=
by
  sorry

end parity_equiv_l25_25737


namespace range_of_a_l25_25838

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) ↔ 1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l25_25838


namespace triangle_area_l25_25782

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  a^2 + b^2 = c^2 ∧ 0.5 * a * b = 24 :=
by {
  sorry
}

end triangle_area_l25_25782


namespace cost_of_each_soda_l25_25725

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l25_25725


namespace odd_function_a_b_l25_25331

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l25_25331


namespace relationship_between_a_and_b_l25_25289

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
  sorry

end relationship_between_a_and_b_l25_25289


namespace intersection_polar_coords_l25_25154

noncomputable def polar_coord_intersection (rho theta : ℝ) : Prop :=
  (rho * (Real.sqrt 3 * Real.cos theta - Real.sin theta) = 2) ∧ (rho = 4 * Real.sin theta)

theorem intersection_polar_coords :
  ∃ (rho theta : ℝ), polar_coord_intersection rho theta ∧ rho = 2 ∧ theta = (Real.pi / 6) := 
sorry

end intersection_polar_coords_l25_25154


namespace nine_sided_polygon_diagonals_l25_25102

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25102


namespace count_multiples_5_or_10_l25_25699

theorem count_multiples_5_or_10 (n : ℕ) (hn : n = 999) : 
  ∃ k : ℕ, k = 199 ∧ (∀ i : ℕ, i < 1000 → (i % 5 = 0 ∨ i % 10 = 0) → i = k) := 
by {
  sorry
}

end count_multiples_5_or_10_l25_25699


namespace carlos_wins_one_game_l25_25849

def games_Won_Laura : ℕ := 5
def games_Lost_Laura : ℕ := 4
def games_Won_Mike : ℕ := 7
def games_Lost_Mike : ℕ := 2
def games_Lost_Carlos : ℕ := 5
variable (C : ℕ) -- Carlos's wins

theorem carlos_wins_one_game :
  games_Won_Laura + games_Won_Mike + C = (games_Won_Laura + games_Lost_Laura + games_Won_Mike + games_Lost_Mike + C + games_Lost_Carlos) / 2 →
  C = 1 :=
by
  sorry

end carlos_wins_one_game_l25_25849


namespace regular_nine_sided_polygon_diagonals_l25_25095

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25095


namespace probability_top_two_same_suit_l25_25414

theorem probability_top_two_same_suit :
  let deck_size := 52
  let suits := 4
  let cards_per_suit := 13
  let first_card_prob := (13 / 52 : ℚ)
  let remaining_cards := 51
  let second_card_same_suit_prob := (12 / 51 : ℚ)
  first_card_prob * second_card_same_suit_prob = (1 / 17 : ℚ) :=
by
  sorry

end probability_top_two_same_suit_l25_25414


namespace arithmetic_geometric_common_ratio_l25_25439

theorem arithmetic_geometric_common_ratio (a₁ r : ℝ) 
  (h₁ : a₁ + a₁ * r^2 = 10) 
  (h₂ : a₁ * (1 + r + r^2 + r^3) = 15) : 
  r = 1/2 ∨ r = -1/2 :=
by {
  sorry
}

end arithmetic_geometric_common_ratio_l25_25439


namespace number_of_triangles_l25_25794

theorem number_of_triangles (x y : ℕ) (P Q : ℕ × ℕ) (O : ℕ × ℕ := (0,0)) (area : ℕ) :
  (P ≠ Q) ∧ (P.1 * 31 + P.2 = 2023) ∧ (Q.1 * 31 + Q.2 = 2023) ∧ 
  (P.1 ≠ Q.1 → P.1 - Q.1 = n ∧ 2023 * n % 6 = 0) → area = 165 :=
sorry

end number_of_triangles_l25_25794


namespace total_cars_l25_25860

-- Conditions
def initial_cars : ℕ := 150
def uncle_cars : ℕ := 5
def grandpa_cars : ℕ := 2 * uncle_cars
def dad_cars : ℕ := 10
def mum_cars : ℕ := dad_cars + 5
def auntie_cars : ℕ := 6

-- Proof statement (theorem)
theorem total_cars : initial_cars + (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars) = 196 :=
by
  sorry

end total_cars_l25_25860


namespace cost_of_second_batch_l25_25764

theorem cost_of_second_batch
  (C_1 C_2 : ℕ)
  (quantity_ratio cost_increase: ℕ) 
  (H1 : C_1 = 3000) 
  (H2 : C_2 = 9600) 
  (H3 : quantity_ratio = 3) 
  (H4 : cost_increase = 1)
  : (∃ x : ℕ, C_1 / x = C_2 / (x + cost_increase) / quantity_ratio) ∧ 
    (C_2 / (C_1 / 15 + cost_increase) / 3 = 16) :=
by
  sorry

end cost_of_second_batch_l25_25764


namespace school_competition_l25_25591

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l25_25591


namespace more_than_four_numbers_make_polynomial_prime_l25_25434

def polynomial (n : ℕ) : ℤ := n^3 - 10 * n^2 + 31 * n - 17

def is_prime (k : ℤ) : Prop :=
  k > 1 ∧ ∀ m : ℤ, m > 1 ∧ m < k → ¬ (m ∣ k)

theorem more_than_four_numbers_make_polynomial_prime :
  (∃ n1 n2 n3 n4 n5 : ℕ, 
    n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧ n5 > 0 ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧ 
    n3 ≠ n4 ∧ n3 ≠ n5 ∧ 
    n4 ≠ n5 ∧ 
    is_prime (polynomial n1) ∧
    is_prime (polynomial n2) ∧
    is_prime (polynomial n3) ∧
    is_prime (polynomial n4) ∧
    is_prime (polynomial n5)) :=
sorry

end more_than_four_numbers_make_polynomial_prime_l25_25434


namespace find_a_b_l25_25336

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l25_25336


namespace set_C_cannot_form_right_triangle_l25_25888

theorem set_C_cannot_form_right_triangle :
  ¬(5^2 + 2^2 = 5^2) :=
by
  sorry

end set_C_cannot_form_right_triangle_l25_25888


namespace odd_function_characterization_l25_25306

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l25_25306


namespace diagonals_in_nine_sided_polygon_l25_25081

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25081


namespace find_x_l25_25950

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (3, 5)
def vec_b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define what it means for two vectors to be parallel
def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (a.1 = k * b.1) ∧ (a.2 = k * b.2)

-- Given condition: vectors a and b are parallel
theorem find_x (x : ℝ) (h : vectors_parallel vec_a (vec_b x)) : x = 5 / 3 :=
by
  sorry

end find_x_l25_25950


namespace quadratic_residue_one_mod_p_l25_25979

theorem quadratic_residue_one_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ) :
  (a^2 % p = 1 % p) ↔ (a % p = 1 % p ∨ a % p = (p-1) % p) :=
sorry

end quadratic_residue_one_mod_p_l25_25979


namespace arithmetic_geometric_sequence_l25_25785

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_d : d ≠ 0)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d)
  (h_geo : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)) :
  (S 4 - S 2) / (S 5 - S 3) = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l25_25785


namespace catherine_pencils_per_friend_l25_25792

theorem catherine_pencils_per_friend :
  ∀ (pencils pens given_pens : ℕ), 
  pencils = pens ∧ pens = 60 ∧ given_pens = 8 ∧ 
  (∃ remaining_items : ℕ, remaining_items = 22 ∧ 
    ∀ friends : ℕ, friends = 7 → 
    remaining_items = (pens - (given_pens * friends)) + (pencils - (given_pens * friends * (pencils / pens)))) →
  ((pencils - (given_pens * friends * (pencils / pens))) / friends) = 6 :=
by 
  sorry

end catherine_pencils_per_friend_l25_25792


namespace regular_nonagon_diagonals_correct_l25_25111

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25111


namespace sum_mod_17_l25_25279

/--
Given the sum of the numbers 82, 83, 84, 85, 86, 87, 88, and 89, and the divisor 17,
prove that the remainder when dividing this sum by 17 is 11.
-/
theorem sum_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 :=
by
  sorry

end sum_mod_17_l25_25279


namespace find_smaller_number_l25_25520

theorem find_smaller_number (a b : ℕ) (h_ratio : 11 * a = 7 * b) (h_diff : b = a + 16) : a = 28 :=
by
  sorry

end find_smaller_number_l25_25520


namespace regular_nine_sided_polygon_diagonals_l25_25097

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25097


namespace each_serving_requires_1_5_apples_l25_25719

theorem each_serving_requires_1_5_apples 
  (guest_count : ℕ) (pie_count : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℝ) 
  (h_guest_count : guest_count = 12)
  (h_pie_count : pie_count = 3)
  (h_servings_per_pie : servings_per_pie = 8)
  (h_apples_per_guest : apples_per_guest = 3) :
  (apples_per_guest * guest_count) / (pie_count * servings_per_pie) = 1.5 :=
by
  sorry

end each_serving_requires_1_5_apples_l25_25719


namespace find_books_second_purchase_profit_l25_25536

-- For part (1)
theorem find_books (x y : ℕ) (h₁ : 12 * x + 10 * y = 1200) (h₂ : 3 * x + 2 * y = 270) :
  x = 50 ∧ y = 60 :=
by 
  sorry

-- For part (2)
theorem second_purchase_profit (m : ℕ) (h₃ : 50 * (m - 12) + 2 * 60 * (12 - 10) ≥ 340) :
  m ≥ 14 :=
by 
  sorry

end find_books_second_purchase_profit_l25_25536


namespace no_day_income_is_36_l25_25781

theorem no_day_income_is_36 : ∀ (n : ℕ), 3 * 3^(n-1) ≠ 36 :=
by
  intro n
  sorry

end no_day_income_is_36_l25_25781


namespace neg_root_sufficient_not_necessary_l25_25403

theorem neg_root_sufficient_not_necessary (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (a < 0) :=
sorry

end neg_root_sufficient_not_necessary_l25_25403


namespace coin_value_permutations_l25_25528

theorem coin_value_permutations : 
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540 := by
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  show 3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540
  
  -- Steps for the proof can be filled in
  -- sorry in place to indicate incomplete proof steps
  sorry

end coin_value_permutations_l25_25528


namespace focus_coordinates_of_parabola_l25_25375

def parabola_focus_coordinates (x y : ℝ) : Prop :=
  x^2 + y = 0 ∧ (0, -1/4) = (0, y)

theorem focus_coordinates_of_parabola (x y : ℝ) :
  parabola_focus_coordinates x y →
  (0, y) = (0, -1/4) := by
  sorry

end focus_coordinates_of_parabola_l25_25375


namespace number_of_routes_l25_25585

open Nat

theorem number_of_routes (south_cities north_cities : ℕ) 
  (connections : south_cities = 4 ∧ north_cities = 5) : 
  ∃ routes, routes = (factorial 3) * (5 ^ 4) := 
by
  sorry

end number_of_routes_l25_25585


namespace f_log_sum_l25_25825

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 2

theorem f_log_sum (x : ℝ) : f (Real.log 5) + f (Real.log (1 / 5)) = 4 :=
by
  sorry

end f_log_sum_l25_25825


namespace f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l25_25377

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem f_monotonically_increasing_intervals:
  ∀ (k : ℤ), ∀ x y, (-Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ y ∧ y ≤ (k * Real.pi + Real.pi / 3) → f x ≤ f y :=
sorry

theorem f_max_min_in_range:
  ∀ x, (-Real.pi / 12) ≤ x ∧ x ≤ (5 * Real.pi / 12) → 
  (f x ≤ 2 ∧ f x ≥ -Real.sqrt 3) :=
sorry

theorem f_max_at_pi_over_3:
  f (Real.pi / 3) = 2 :=
sorry

theorem f_min_at_neg_pi_over_12:
  f (-Real.pi / 12) = -Real.sqrt 3 :=
sorry

end f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l25_25377


namespace hours_per_shift_l25_25967

def hourlyWage : ℝ := 4.0
def tipRate : ℝ := 0.15
def shiftsWorked : ℕ := 3
def averageOrdersPerHour : ℝ := 40.0
def totalEarnings : ℝ := 240.0

theorem hours_per_shift :
  (hourlyWage + averageOrdersPerHour * tipRate) * (8 * shiftsWorked) = totalEarnings := 
sorry

end hours_per_shift_l25_25967


namespace ceil_sqrt_of_900_l25_25925

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem ceil_sqrt_of_900 :
  isPerfectSquare 36 ∧ isPerfectSquare 25 ∧ (36 * 25 = 900) → 
  Int.ceil (Real.sqrt 900) = 30 :=
by
  intro h
  sorry

end ceil_sqrt_of_900_l25_25925


namespace shaded_area_l25_25931

def radius (R : ℝ) : Prop := R > 0
def angle (α : ℝ) : Prop := α = 20 * (Real.pi / 180)

theorem shaded_area (R : ℝ) (hR : radius R) (hα : angle (20 * (Real.pi / 180))) :
  let S0 := Real.pi * R^2 / 2
  let sector_radius := 2 * R
  let sector_angle := 20 * (Real.pi / 180)
  (2 * sector_radius * sector_radius * sector_angle / 2) / sector_angle = 2 * Real.pi * R^2 / 9 :=
by
  sorry

end shaded_area_l25_25931


namespace value_of_a_2008_l25_25178

open Nat

def is_rel_prime_75 (n : ℕ) : Prop :=
  gcd n 75 = 1

def rel_prime_75_seq : ℕ → ℕ
| n := (75 * (n / 40) + (Finset.filter is_rel_prime_75 (Finset.range (75 + 1))).toList.get! (n % 40))

theorem value_of_a_2008 : rel_prime_75_seq 2008 = 3764 :=
sorry

end value_of_a_2008_l25_25178


namespace reflected_light_eq_l25_25666

theorem reflected_light_eq
  (incident_light : ∀ x y : ℝ, 2 * x - y + 6 = 0)
  (reflection_line : ∀ x y : ℝ, y = x) :
  ∃ x y : ℝ, x + 2 * y + 18 = 0 :=
sorry

end reflected_light_eq_l25_25666


namespace xy_value_l25_25852

theorem xy_value :
  ∃ a b c x y : ℝ,
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧
    3 * a + 2 * b + c = 5 ∧
    2 * a + b - 3 * c = 1 ∧
    (∀ m, m = 3 * a + b - 7 * c → (m = x ∨ m = y)) ∧
    x = -5 / 7 ∧
    y = -1 / 11 ∧
    x * y = 5 / 77 :=
sorry

end xy_value_l25_25852


namespace hyperbola_standard_equation_l25_25820

theorem hyperbola_standard_equation (a b : ℝ) (x y : ℝ)
  (H₁ : 2 * a = 2) -- length of the real axis is 2
  (H₂ : y = 2 * x) -- one of its asymptote equations
  : y^2 - 4 * x^2 = 1 :=
sorry

end hyperbola_standard_equation_l25_25820


namespace triple_solution_unique_l25_25430

theorem triple_solution_unique (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  (a^2 + b^2 = n * Nat.lcm a b + n^2) ∧
  (b^2 + c^2 = n * Nat.lcm b c + n^2) ∧
  (c^2 + a^2 = n * Nat.lcm c a + n^2) →
  (a = n ∧ b = n ∧ c = n) :=
by
  sorry

end triple_solution_unique_l25_25430


namespace shaded_area_correct_l25_25423

def diameter := 3 -- inches
def pattern_length := 18 -- inches equivalent to 1.5 feet

def radius := diameter / 2 -- radius calculation

noncomputable def area_of_one_circle := Real.pi * (radius ^ 2)
def number_of_circles := pattern_length / diameter
noncomputable def total_shaded_area := number_of_circles * area_of_one_circle

theorem shaded_area_correct :
  total_shaded_area = 13.5 * Real.pi :=
  by
  sorry

end shaded_area_correct_l25_25423


namespace point_in_plane_region_l25_25515

theorem point_in_plane_region :
  (2 * 0 + 1 - 6 < 0) ∧ ¬(2 * 5 + 0 - 6 < 0) ∧ ¬(2 * 0 + 7 - 6 < 0) ∧ ¬(2 * 2 + 3 - 6 < 0) :=
by
  -- Proof detail goes here.
  sorry

end point_in_plane_region_l25_25515


namespace parallel_lines_condition_l25_25576

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, (a * x + 2 * y - 1 = 0) → (x + (a + 1) * y + 4 = 0) → a = 1) ↔
  (∀ x y : ℝ, (a = 1 ∧ a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0) ∨
   (a ≠ 1 ∧ a = -2 ∧ a * x + 2 * y - 1 ≠ 0 → x + (a + 1) * y + 4 ≠ 0)) :=
by
  sorry

end parallel_lines_condition_l25_25576


namespace will_initial_money_l25_25769

theorem will_initial_money (spent_game : ℕ) (number_of_toys : ℕ) (cost_per_toy : ℕ) (initial_money : ℕ) :
  spent_game = 27 →
  number_of_toys = 5 →
  cost_per_toy = 6 →
  initial_money = spent_game + number_of_toys * cost_per_toy →
  initial_money = 57 :=
by
  intros
  sorry

end will_initial_money_l25_25769


namespace find_constants_for_odd_function_l25_25342

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l25_25342


namespace johns_hats_cost_l25_25169

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l25_25169


namespace diagonals_in_regular_nine_sided_polygon_l25_25071

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25071


namespace no_perfect_squares_l25_25261

theorem no_perfect_squares (x y : ℕ) : ¬ (∃ a b : ℕ, x^2 + y = a^2 ∧ x + y^2 = b^2) :=
sorry

end no_perfect_squares_l25_25261


namespace smallest_number_of_students_l25_25597

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l25_25597


namespace total_distinguishable_triangles_l25_25655

-- Define number of colors
def numColors : Nat := 8

-- Define center colors
def centerColors : Nat := 3

-- Prove the total number of distinguishable large equilateral triangles
theorem total_distinguishable_triangles : 
  numColors * (numColors + numColors * (numColors - 1) + (numColors.choose 3)) * centerColors = 360 := by
  sorry

end total_distinguishable_triangles_l25_25655


namespace nine_sided_polygon_diagonals_l25_25033

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25033


namespace a_sub_b_eq_2_l25_25141

theorem a_sub_b_eq_2 (a b : ℝ)
  (h : (a - 5) ^ 2 + |b ^ 3 - 27| = 0) : a - b = 2 :=
by
  sorry

end a_sub_b_eq_2_l25_25141


namespace value_of_k_l25_25147

theorem value_of_k (k : ℝ) (x : ℝ) (h : (k - 3) * x^2 + 6 * x + k^2 - k = 0) (r : x = -1) : 
  k = -3 := 
by
  sorry

end value_of_k_l25_25147


namespace elsa_emma_spending_ratio_l25_25803

theorem elsa_emma_spending_ratio
  (E : ℝ)
  (h_emma : ∃ (x : ℝ), x = 58)
  (h_elizabeth : ∃ (y : ℝ), y = 4 * E)
  (h_total : 58 + E + 4 * E = 638) :
  E / 58 = 2 :=
by
  sorry

end elsa_emma_spending_ratio_l25_25803


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l25_25999

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l25_25999


namespace quadratic_factorization_l25_25822

theorem quadratic_factorization (p q x_1 x_2 : ℝ) (h1 : x_1 = 2) (h2 : x_2 = -3) 
    (h3 : x_1 + x_2 = -p) (h4 : x_1 * x_2 = q) : 
    (x - 2) * (x + 3) = x^2 + p * x + q :=
by
  sorry

end quadratic_factorization_l25_25822


namespace abs_neg_is_2_l25_25642

theorem abs_neg_is_2 (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 :=
by sorry

end abs_neg_is_2_l25_25642


namespace fans_received_all_items_l25_25937

theorem fans_received_all_items (n : ℕ) (h1 : (∀ k : ℕ, k * 45 ≤ n → (k * 45) ∣ n))
                                (h2 : (∀ k : ℕ, k * 50 ≤ n → (k * 50) ∣ n))
                                (h3 : (∀ k : ℕ, k * 100 ≤ n → (k * 100) ∣ n))
                                (capacity_full : n = 5000) :
  n / Nat.lcm 45 (Nat.lcm 50 100) = 5 :=
by
  sorry

end fans_received_all_items_l25_25937


namespace regular_nonagon_diagonals_correct_l25_25116

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25116


namespace HeatherIsHeavier_l25_25573

-- Definitions
def HeatherWeight : ℕ := 87
def EmilyWeight : ℕ := 9

-- Theorem statement
theorem HeatherIsHeavier : HeatherWeight - EmilyWeight = 78 := by
  sorry

end HeatherIsHeavier_l25_25573


namespace nine_sided_polygon_diagonals_l25_25055

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25055


namespace diagonals_in_nonagon_l25_25134

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25134


namespace heather_heavier_than_emily_l25_25572

theorem heather_heavier_than_emily :
  let Heather_weight := 87
  let Emily_weight := 9
  Heather_weight - Emily_weight = 78 :=
by
  -- Proof here
  sorry

end heather_heavier_than_emily_l25_25572


namespace find_a_b_l25_25323

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l25_25323


namespace sequence_term_and_k_value_l25_25813

/-- Given a sequence {a_n} whose sum of the first n terms is S_n = n^2 - 9n,
    prove the sequence term a_n = 2n - 10, and if 5 < a_k < 8, then k = 8. -/
theorem sequence_term_and_k_value (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 9 * n) :
  (∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) →
  (∀ n, a n = 2 * n - 10) ∧ (∀ k, 5 < a k ∧ a k < 8 → k = 8) :=
by {
  -- Given S_n = n^2 - 9n, we need to show a_n = 2n - 10 and verify when 5 < a_k < 8, then k = 8
  sorry
}

end sequence_term_and_k_value_l25_25813


namespace arithmetic_sequence_l25_25716

theorem arithmetic_sequence {a b : ℤ} :
  (-1 < a ∧ a < b ∧ b < 8) ∧
  (8 - (-1) = 9) ∧
  (a + b = 7) →
  (a = 2 ∧ b = 5) :=
by
  sorry

end arithmetic_sequence_l25_25716


namespace odd_function_values_l25_25333

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l25_25333


namespace smallest_total_students_l25_25600

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l25_25600


namespace evaluate_expression_l25_25926

theorem evaluate_expression (x : ℝ) (h : |7 - 8 * (x - 12)| - |5 - 11| = 73) : x = 3 :=
  sorry

end evaluate_expression_l25_25926


namespace simplify_tangent_sum_l25_25988

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l25_25988


namespace value_when_x_is_neg1_l25_25899

theorem value_when_x_is_neg1 (p q : ℝ) (h : p + q = 2022) : 
  (p * (-1)^3 + q * (-1) + 1) = -2021 := by
  sorry

end value_when_x_is_neg1_l25_25899


namespace step_of_induction_l25_25892

theorem step_of_induction (k : ℕ) (h : ∃ m : ℕ, 5^k - 2^k = 3 * m) :
  5^(k+1) - 2^(k+1) = 5 * (5^k - 2^k) + 3 * 2^k := 
by
  sorry

end step_of_induction_l25_25892


namespace diagonals_in_regular_nine_sided_polygon_l25_25068

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25068


namespace total_lunch_bill_l25_25640

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h_hd : cost_hotdog = 5.36) (h_sd : cost_salad = 5.10) :
  cost_hotdog + cost_salad = 10.46 :=
by
  sorry

end total_lunch_bill_l25_25640


namespace nine_sided_polygon_diagonals_l25_25052

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25052


namespace find_constants_for_odd_function_l25_25341

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l25_25341


namespace digit_B_divisibility_l25_25731

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧
    (∃ n : ℕ, 658274 * 10 + B = 2 * n) ∧
    (∃ m : ℕ, 6582740 + B = 4 * m) ∧
    (B = 0 ∨ B = 5) ∧
    (∃ k : ℕ, 658274 * 10 + B = 7 * k) ∧
    (∃ p : ℕ, 6582740 + B = 8 * p) :=
sorry

end digit_B_divisibility_l25_25731


namespace tan_half_angle_l25_25691

theorem tan_half_angle (α : ℝ) (h1 : Real.sin α + Real.cos α = 1 / 5)
  (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -1 / 3 :=
sorry

end tan_half_angle_l25_25691


namespace dot_product_a_b_l25_25584

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (-1, 2)

theorem dot_product_a_b : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 := by
  sorry

end dot_product_a_b_l25_25584


namespace minimum_value_expression_equality_case_l25_25853

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c) ≥ 343 :=
sorry

theorem equality_case :
  (a b c : ℝ) (h : a = 1 ∧ b = 1 ∧ c = 1) →
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c) = 343 :=
sorry

end minimum_value_expression_equality_case_l25_25853


namespace triangle_is_equilateral_l25_25839

-- Define a triangle with angles A, B, and C
variables (A B C : ℝ)

-- The conditions of the problem
def log_sin_arithmetic_sequence : Prop :=
  Real.log (Real.sin A) + Real.log (Real.sin C) = 2 * Real.log (Real.sin B)

def angles_arithmetic_sequence : Prop :=
  2 * B = A + C

-- The theorem that the triangle is equilateral given these conditions
theorem triangle_is_equilateral :
  log_sin_arithmetic_sequence A B C → angles_arithmetic_sequence A B C → 
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_is_equilateral_l25_25839


namespace solve_inequality_inequality_proof_l25_25905

-- Problem 1: Solve the inequality |2x+1| - |x-4| > 2
theorem solve_inequality (x : ℝ) :
  (|2 * x + 1| - |x - 4| > 2) ↔ (x < -7 ∨ x > (5/3)) :=
sorry

-- Problem 2: Prove the inequality given a > 0 and b > 0
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
sorry

end solve_inequality_inequality_proof_l25_25905


namespace odd_function_characterization_l25_25307

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l25_25307


namespace examination_total_students_l25_25661

theorem examination_total_students (T : ℝ) :
  (0.35 * T + 520) = T ↔ T = 800 :=
by 
  sorry

end examination_total_students_l25_25661


namespace sales_tax_difference_l25_25421

theorem sales_tax_difference
  (price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.075)
  (h_rate2 : rate2 = 0.07)
  (h_price : price = 30) :
  (price * rate1 - price * rate2 = 0.15) :=
by
  sorry

end sales_tax_difference_l25_25421


namespace pieces_bound_l25_25964

open Finset

variable {n : ℕ} (B W : ℕ)

theorem pieces_bound (n : ℕ) (B W : ℕ) (hB : B ≤ n^2) (hW : W ≤ n^2) :
    B ≤ n^2 ∨ W ≤ n^2 := 
by
  sorry

end pieces_bound_l25_25964


namespace sum_of_series_eq_5_over_16_l25_25876

theorem sum_of_series_eq_5_over_16 :
  ∑' n : ℕ, (n + 1 : ℝ) / (5 : ℝ)^(n + 1) = 5 / 16 := by
  sorry

end sum_of_series_eq_5_over_16_l25_25876


namespace regular_nine_sided_polygon_diagonals_l25_25092

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25092


namespace tan_eq_tan_x2_sol_count_l25_25303

noncomputable def arctan1000 := Real.arctan 1000

theorem tan_eq_tan_x2_sol_count :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, 
    0 ≤ x ∧ x ≤ arctan1000 ∧ Real.tan x = Real.tan (x^2) →
    ∃ k : ℕ, k < n ∧ x = Real.sqrt (k * Real.pi + x) :=
sorry

end tan_eq_tan_x2_sol_count_l25_25303


namespace avg_cost_is_7000_l25_25717

open ProbabilityTheory

noncomputable def avg_cost_of_testing (n m : ℕ) (cost: ℕ) : ℝ := 
  let prob_X_4000 := (2/5) * (1/4)
  let prob_X_6000 := (2/5) * (3/4) * (1/3) + (3/5) * (2/4) * (1/3) + (3/5) * (2/4) * (1/3)
  let prob_X_8000 := 1 - prob_X_4000 - prob_X_6000
  4000 * prob_X_4000 + 6000 * prob_X_6000 + 8000 * prob_X_8000

theorem avg_cost_is_7000 : 
  avg_cost_of_testing 5 2 2000 = 7000 :=
sorry

end avg_cost_is_7000_l25_25717


namespace number_of_correct_inequalities_l25_25685

variable {a b : ℝ}

theorem number_of_correct_inequalities (h₁ : a > 0) (h₂ : 0 > b) (h₃ : a + b > 0) :
  (ite (a^2 > b^2) 1 0) + (ite (1/a > 1/b) 1 0) + (ite (a^3 < ab^2) 1 0) + (ite (a^2 * b < b^3) 1 0) = 3 := 
sorry

end number_of_correct_inequalities_l25_25685


namespace sum_of_roots_l25_25700

theorem sum_of_roots : ∀ x : ℝ, ((x + 3) * (x - 4) = 20) → (∃ a b : ℝ, a = x + 3 ∧ b = x - 4 ∧ a * b = 20 ∧ (∑ x in {a, b}, x) = 1) :=
begin
  intros x h,
  sorry
end

end sum_of_roots_l25_25700


namespace hats_cost_l25_25165

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l25_25165


namespace part_I_part_II_l25_25944

def t1 (x : ℕ) : ℕ → ℕ := λ r => binomial 5 r * 2^(5 - r) * x^(10 - 3*r)
def t2 (x : ℕ) (n : ℕ) : ℕ → ℕ := λ r => binomial n r * 4 * x^(n/2 - 3)

-- term containing 1/x^2 in expansion of (2*x^2 + 1/x)^5
theorem part_I (x : ℕ) : t1 x 4 = 10 / x^2 :=
  sorry

-- value of n when sum of binomial coefficients is 28 less than coefficient of third term in (sqrt(x) + 2/x)^n
theorem part_II (x : ℕ) (n : ℕ) : 2^5 = binomial n 2 * 4 - 28 → n = 6 :=
  sorry

end part_I_part_II_l25_25944


namespace find_interest_rate_l25_25753

-- Given conditions
def P : ℝ := 4099.999999999999
def t : ℕ := 2
def CI_minus_SI : ℝ := 41

-- Formulas for Simple Interest and Compound Interest
def SI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * (t : ℝ)
def CI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * ((1 + r) ^ t) - P

-- Main theorem to prove: the interest rate r is 0.1 (i.e., 10%)
theorem find_interest_rate (r : ℝ) : 
  (CI P r t - SI P r t = CI_minus_SI) → r = 0.1 :=
by
  sorry

end find_interest_rate_l25_25753


namespace jessica_initial_withdrawal_fraction_l25_25611

variable {B : ℝ} -- this is the initial balance

noncomputable def initial_withdrawal_fraction (B : ℝ) : Prop :=
  let remaining_balance := B - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 → (400 / B) = 2 / 5

-- Our goal is to prove the statement given conditions.
theorem jessica_initial_withdrawal_fraction : 
  ∃ B : ℝ, initial_withdrawal_fraction B :=
sorry

end jessica_initial_withdrawal_fraction_l25_25611


namespace min_surface_area_base_edge_length_l25_25823

noncomputable def min_base_edge_length (V : ℝ) : ℝ :=
  2 * (V / (2 * Real.pi))^(1/3)

theorem min_surface_area_base_edge_length (V : ℝ) : 
  min_base_edge_length V = (4 * V)^(1/3) :=
by
  sorry

end min_surface_area_base_edge_length_l25_25823


namespace distance_from_M_to_x_axis_l25_25604

-- Define the point M and its coordinates.
def point_M : ℤ × ℤ := (-9, 12)

-- Define the distance to the x-axis is simply the absolute value of the y-coordinate.
def distance_to_x_axis (p : ℤ × ℤ) : ℤ := Int.natAbs p.snd

-- Theorem stating the distance from point M to the x-axis is 12.
theorem distance_from_M_to_x_axis : distance_to_x_axis point_M = 12 := by
  sorry

end distance_from_M_to_x_axis_l25_25604


namespace diagonals_in_nonagon_l25_25132

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25132


namespace diagonals_in_nine_sided_polygon_l25_25078

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25078


namespace cost_of_each_soda_l25_25721

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l25_25721


namespace original_price_correct_l25_25530

noncomputable def original_price (selling_price : ℝ) (gain_percent : ℝ) : ℝ :=
  selling_price / (1 + gain_percent / 100)

theorem original_price_correct :
  original_price 35 75 = 20 :=
by
  sorry

end original_price_correct_l25_25530


namespace poly_divisibility_implies_C_D_l25_25426

noncomputable def poly_condition : Prop :=
  ∃ (C D : ℤ), ∀ (α : ℂ), α^2 - α + 1 = 0 → α^103 + C * α^2 + D * α + 1 = 0

/- The translated proof problem -/
theorem poly_divisibility_implies_C_D (C D : ℤ) :
  (poly_condition) → (C = -1 ∧ D = 0) :=
by
  intro h
  sorry

end poly_divisibility_implies_C_D_l25_25426


namespace solution_set_l25_25274

theorem solution_set (x : ℝ) : floor (floor (3 * x) - 1 / 2) = floor (x + 3) ↔ x ∈ set.Ico (5 / 3) (7 / 3) :=
by
  sorry

end solution_set_l25_25274


namespace mother_sold_rings_correct_l25_25802

noncomputable def motherSellsRings (initial_bought_rings mother_bought_rings remaining_rings final_stock : ℤ) : ℤ :=
  let initial_stock := initial_bought_rings / 2
  let total_stock := initial_bought_rings + initial_stock
  let sold_by_eliza := (3 * total_stock) / 4
  let remaining_after_eliza := total_stock - sold_by_eliza
  let new_total_stock := remaining_after_eliza + mother_bought_rings
  new_total_stock - final_stock

theorem mother_sold_rings_correct :
  motherSellsRings 200 300 225 300 = 150 :=
by
  sorry

end mother_sold_rings_correct_l25_25802


namespace problem_l25_25451

open Nat

theorem problem (m n : ℕ) (M : Finset ℕ) (N : Finset ℕ) 
  (hyp_M : M = {1, 2, 3, m}) 
  (hyp_N : N = {4, 7, n^4, n^2 + 3 * n})
  (hyp_f : ∀ x ∈ M, (3 * x + 1) ∈ N) : m - n = 3 := 
by 
  sorry

end problem_l25_25451


namespace HeatherIsHeavier_l25_25574

-- Definitions
def HeatherWeight : ℕ := 87
def EmilyWeight : ℕ := 9

-- Theorem statement
theorem HeatherIsHeavier : HeatherWeight - EmilyWeight = 78 := by
  sorry

end HeatherIsHeavier_l25_25574


namespace nine_sided_polygon_diagonals_l25_25127

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25127


namespace balloon_highest_elevation_l25_25160

theorem balloon_highest_elevation 
  (lift_rate : ℕ)
  (descend_rate : ℕ)
  (pull_time1 : ℕ)
  (release_time : ℕ)
  (pull_time2 : ℕ) :
  lift_rate = 50 →
  descend_rate = 10 →
  pull_time1 = 15 →
  release_time = 10 →
  pull_time2 = 15 →
  (lift_rate * pull_time1 - descend_rate * release_time + lift_rate * pull_time2) = 1400 :=
by
  sorry

end balloon_highest_elevation_l25_25160


namespace pencil_cost_is_4_l25_25248

variables (pencils pens : ℕ) (pen_cost total_cost : ℕ)

def total_pencils := 15 * 80
def total_pens := (2 * total_pencils) + 300
def total_pen_cost := total_pens * pen_cost
def total_pencil_cost := total_cost - total_pen_cost
def pencil_cost := total_pencil_cost / total_pencils

theorem pencil_cost_is_4
  (pen_cost_eq_5 : pen_cost = 5)
  (total_cost_eq_18300 : total_cost = 18300)
  : pencil_cost = 4 :=
by
  sorry

end pencil_cost_is_4_l25_25248


namespace percent_brandA_in_mix_l25_25778

theorem percent_brandA_in_mix (x : Real) :
  (0.60 * x + 0.35 * (100 - x) = 50) → x = 60 :=
by
  intro h
  sorry

end percent_brandA_in_mix_l25_25778


namespace product_of_roots_l25_25453

theorem product_of_roots (x : ℝ) :
  (x+3) * (x-4) = 22 →
  let a := 1 in
  let b := -1 in
  let c := -34 in
  a * x^2 + b * x + c = 0 → 
  (c / a) = -34 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end product_of_roots_l25_25453


namespace students_more_than_rabbits_l25_25800

-- Definitions of conditions
def classrooms : ℕ := 5
def students_per_classroom : ℕ := 22
def rabbits_per_classroom : ℕ := 2

-- Statement of the theorem
theorem students_more_than_rabbits :
  classrooms * students_per_classroom - classrooms * rabbits_per_classroom = 100 := 
  by
    sorry

end students_more_than_rabbits_l25_25800


namespace least_common_multiple_l25_25875

open Int

theorem least_common_multiple {a b c : ℕ} 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : Nat.lcm a c = 90 := 
sorry

end least_common_multiple_l25_25875


namespace speed_difference_l25_25612

theorem speed_difference :
  let distance : ℝ := 8
  let zoe_time_hours : ℝ := 2 / 3
  let john_time_hours : ℝ := 1
  let zoe_speed : ℝ := distance / zoe_time_hours
  let john_speed : ℝ := distance / john_time_hours
  zoe_speed - john_speed = 4 :=
by
  sorry

end speed_difference_l25_25612


namespace smallest_number_of_students_l25_25589

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l25_25589


namespace number_of_green_hats_l25_25510

theorem number_of_green_hats 
  (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) 
  : G = 40 :=
sorry

end number_of_green_hats_l25_25510


namespace total_students_correct_l25_25983

-- Define the number of students who play football, cricket, both and neither.
def play_football : ℕ := 325
def play_cricket : ℕ := 175
def play_both : ℕ := 90
def play_neither : ℕ := 50

-- Define the total number of students
def total_students : ℕ := play_football + play_cricket - play_both + play_neither

-- Prove that the total number of students is 460 given the conditions
theorem total_students_correct : total_students = 460 := by
  sorry

end total_students_correct_l25_25983


namespace nine_sided_polygon_diagonals_l25_25036

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25036


namespace minimum_value_of_z_l25_25268

def z (x y : ℝ) : ℝ := 3 * x ^ 2 + 4 * y ^ 2 + 12 * x - 8 * y + 3 * x * y + 30

theorem minimum_value_of_z : ∃ (x y : ℝ), z x y = 8 := 
sorry

end minimum_value_of_z_l25_25268


namespace sam_dimes_now_l25_25635

-- Define the initial number of dimes Sam had
def initial_dimes : ℕ := 9

-- Define the number of dimes Sam gave away
def dimes_given : ℕ := 7

-- State the theorem: The number of dimes Sam has now is 2
theorem sam_dimes_now : initial_dimes - dimes_given = 2 := by
  sorry

end sam_dimes_now_l25_25635


namespace cost_of_each_soda_l25_25724

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l25_25724


namespace percentage_of_remaining_cats_kittens_is_67_l25_25368

noncomputable def percentage_of_kittens : ℕ :=
  let total_cats := 6 in
  let female_cats := total_cats / 2 in
  let kittens_per_female_cat := 7 in
  let total_kittens := female_cats * kittens_per_female_cat in
  let sold_kittens := 9 in
  let remaining_kittens := total_kittens - sold_kittens in
  let remaining_total_cats := total_cats + remaining_kittens in
  let percentage := (remaining_kittens * 100) / remaining_total_cats in
  percentage

theorem percentage_of_remaining_cats_kittens_is_67 :
  percentage_of_kittens = 67 :=
by
  sorry

end percentage_of_remaining_cats_kittens_is_67_l25_25368


namespace sequence_problem_l25_25814

variable {n : ℕ}

-- We define the arithmetic sequence conditions
noncomputable def a_n : ℕ → ℕ
| n => 2 * n + 1

-- Conditions that the sequence must satisfy
axiom a_3_eq_7 : a_n 3 = 7
axiom a_5_a_7_eq_26 : a_n 5 + a_n 7 = 26

-- Define the sum of the sequence
noncomputable def S_n (n : ℕ) := n^2 + 2 * n

-- Define the sequence b_n
noncomputable def b_n (n : ℕ) := 1 / (a_n n ^ 2 - 1 : ℝ)

-- Define the sum of the sequence b_n
noncomputable def T_n (n : ℕ) := (n / (4 * (n + 1)) : ℝ)

-- The main theorem to prove
theorem sequence_problem :
  (a_n n = 2 * n + 1) ∧ (S_n n = n^2 + 2 * n) ∧ (T_n n = n / (4 * (n + 1))) :=
  sorry

end sequence_problem_l25_25814


namespace hyperbola_center_l25_25779

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 3) (h3 : x2 = 10) (h4 : y2 = 7) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (8, 5) :=
by
  rw [h1, h2, h3, h4]
  simp
  -- Proof steps demonstrating the calculation
  -- simplify the arithmetic expressions
  sorry

end hyperbola_center_l25_25779


namespace smallest_possible_number_of_students_l25_25596

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l25_25596


namespace nine_sided_polygon_diagonals_l25_25035

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25035


namespace find_r_l25_25808

theorem find_r 
  (r s : ℝ)
  (h1 : 9 * (r * r) * s = -6)
  (h2 : r * r + 2 * r * s = -16 / 3)
  (h3 : 2 * r + s = 2 / 3)
  (polynomial_condition : ∀ x : ℝ, 9 * x^3 - 6 * x^2 - 48 * x + 54 = 9 * (x - r)^2 * (x - s)) 
: r = -2 / 3 :=
sorry

end find_r_l25_25808


namespace sets_are_equal_l25_25540

-- Define sets according to the given options
def option_a_M : Set (ℕ × ℕ) := {(3, 2)}
def option_a_N : Set (ℕ × ℕ) := {(2, 3)}

def option_b_M : Set ℕ := {3, 2}
def option_b_N : Set (ℕ × ℕ) := {(3, 2)}

def option_c_M : Set (ℕ × ℕ) := {(x, y) | x + y = 1}
def option_c_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_d_M : Set ℕ := {3, 2}
def option_d_N : Set ℕ := {2, 3}

-- Proof goal
theorem sets_are_equal : option_d_M = option_d_N :=
sorry

end sets_are_equal_l25_25540


namespace nine_sided_polygon_diagonals_l25_25048

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25048


namespace xiaoming_grade_is_89_l25_25669

noncomputable def xiaoming_physical_education_grade
  (extra_activity_score : ℕ) (midterm_score : ℕ) (final_exam_score : ℕ)
  (ratio_extra : ℕ) (ratio_mid : ℕ) (ratio_final : ℕ) : ℝ :=
  (extra_activity_score * ratio_extra + midterm_score * ratio_mid + final_exam_score * ratio_final) / (ratio_extra + ratio_mid + ratio_final)

theorem xiaoming_grade_is_89 :
  xiaoming_physical_education_grade 95 90 85 2 4 4 = 89 := by
    sorry

end xiaoming_grade_is_89_l25_25669


namespace product_173_240_l25_25209

theorem product_173_240 :
  ∃ n : ℕ, n = 3460 ∧ n * 12 = 173 * 240 ∧ 173 * 240 = 41520 :=
by
  sorry

end product_173_240_l25_25209


namespace school_competition_l25_25593

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l25_25593


namespace jane_20_cent_items_l25_25356

theorem jane_20_cent_items {x y z : ℕ} (h1 : x + y + z = 50) (h2 : 20 * x + 150 * y + 250 * z = 5000) : x = 31 :=
by
  -- The formal proof would go here
  sorry

end jane_20_cent_items_l25_25356


namespace regular_triangular_pyramid_volume_l25_25283

noncomputable def pyramid_volume (a h γ : ℝ) : ℝ :=
  (Real.sqrt 3 * a^2 * h) / 12

theorem regular_triangular_pyramid_volume
  (a h γ : ℝ) (h_nonneg : 0 ≤ h) (γ_nonneg : 0 ≤ γ) :
  pyramid_volume a h γ = (Real.sqrt 3 * a^2 * h) / 12 :=
by
  sorry

end regular_triangular_pyramid_volume_l25_25283


namespace diagonals_in_nine_sided_polygon_l25_25064

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25064


namespace lines_parallel_m_value_l25_25302

theorem lines_parallel_m_value (m : ℝ) : 
  (∀ (x y : ℝ), (x + 2 * m * y - 1 = 0) → ((m - 2) * x - m * y + 2 = 0)) → m = 3 / 2 :=
by
  -- placeholder for mathematical proof
  sorry

end lines_parallel_m_value_l25_25302


namespace cricket_team_members_l25_25212

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℚ) (wk_keeper_age : ℚ) 
  (avg_whole_team : ℚ) (avg_remaining_players : ℚ)
  (h1 : captain_age = 25)
  (h2 : wk_keeper_age = 28)
  (h3 : avg_whole_team = 22)
  (h4 : avg_remaining_players = 21)
  (h5 : 22 * n = 25 + 28 + 21 * (n - 2)) :
  n = 11 :=
by sorry

end cricket_team_members_l25_25212


namespace prime_square_minus_five_not_div_by_eight_l25_25746

theorem prime_square_minus_five_not_div_by_eight (p : ℕ) (prime_p : Prime p) (p_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) :=
sorry

end prime_square_minus_five_not_div_by_eight_l25_25746


namespace find_x_l25_25513

theorem find_x (x y : ℤ) (some_number : ℤ) (h1 : y = 2) (h2 : some_number = 14) (h3 : 2 * x - y = some_number) : x = 8 :=
by 
  sorry

end find_x_l25_25513


namespace height_of_building_l25_25243

-- Define the conditions as hypotheses
def height_of_flagstaff : ℝ := 17.5
def shadow_length_of_flagstaff : ℝ := 40.25
def shadow_length_of_building : ℝ := 28.75

-- Define the height ratio based on similar triangles
theorem height_of_building :
  (height_of_flagstaff / shadow_length_of_flagstaff = 12.47 / shadow_length_of_building) :=
by
  sorry

end height_of_building_l25_25243


namespace percent_exceed_not_ticketed_l25_25478

-- Defining the given conditions
def total_motorists : ℕ := 100
def percent_exceed_limit : ℕ := 50
def percent_with_tickets : ℕ := 40

-- Calculate the number of motorists exceeding the limit and receiving tickets
def motorists_exceed_limit := total_motorists * percent_exceed_limit / 100
def motorists_with_tickets := total_motorists * percent_with_tickets / 100

-- Theorem: Percentage of motorists exceeding the limit but not receiving tickets
theorem percent_exceed_not_ticketed : 
  (motorists_exceed_limit - motorists_with_tickets) * 100 / motorists_exceed_limit = 20 := 
by
  sorry

end percent_exceed_not_ticketed_l25_25478


namespace remainder_of_19_pow_60_mod_7_l25_25893

theorem remainder_of_19_pow_60_mod_7 : (19 ^ 60) % 7 = 1 := 
by {
  sorry
}

end remainder_of_19_pow_60_mod_7_l25_25893


namespace quadratic_roots_product_sum_l25_25738

theorem quadratic_roots_product_sum :
  (∀ d e : ℝ, 3 * d^2 + 4 * d - 7 = 0 ∧ 3 * e^2 + 4 * e - 7 = 0 →
   (d + 1) * (e + 1) = - 8 / 3) := by
sorry

end quadratic_roots_product_sum_l25_25738


namespace percentage_of_stock_l25_25757

noncomputable def investment_amount : ℝ := 6000
noncomputable def income_derived : ℝ := 756
noncomputable def brokerage_percentage : ℝ := 0.25
noncomputable def brokerage_fee : ℝ := investment_amount * (brokerage_percentage / 100)
noncomputable def net_investment_amount : ℝ := investment_amount - brokerage_fee
noncomputable def dividend_yield : ℝ := (income_derived / net_investment_amount) * 100

theorem percentage_of_stock :
  ∃ (percentage_of_stock : ℝ), percentage_of_stock = dividend_yield := by
  sorry

end percentage_of_stock_l25_25757


namespace distance_to_school_l25_25399

variable (T D : ℕ)

/-- Given the conditions, prove the distance from the child's home to the school is 630 meters --/
theorem distance_to_school :
  (5 * (T + 6) = D) →
  (7 * (T - 30) = D) →
  D = 630 :=
by
  intros h1 h2
  sorry

end distance_to_school_l25_25399


namespace mariel_dogs_count_l25_25182

theorem mariel_dogs_count
  (num_dogs_other: Nat)
  (num_legs_tangled: Nat)
  (num_legs_per_dog: Nat)
  (num_legs_per_human: Nat)
  (num_dog_walkers: Nat)
  (num_dogs_mariel: Nat):
  num_dogs_other = 3 →
  num_legs_tangled = 36 →
  num_legs_per_dog = 4 →
  num_legs_per_human = 2 →
  num_dog_walkers = 2 →
  4*num_dogs_mariel + 4*num_dogs_other + 2*num_dog_walkers = num_legs_tangled →
  num_dogs_mariel = 5 :=
by 
  intros h_other h_tangled h_legs_dog h_legs_human h_walkers h_eq
  sorry

end mariel_dogs_count_l25_25182


namespace integer_representation_l25_25744

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 :=
by sorry

end integer_representation_l25_25744


namespace quadratic_eq_is_general_form_l25_25755

def quadratic_eq_general_form (x : ℝ) : Prop :=
  x^2 - 2 * (3 * x - 2) + (x + 1) = x^2 - 5 * x + 5

theorem quadratic_eq_is_general_form :
  quadratic_eq_general_form x :=
sorry

end quadratic_eq_is_general_form_l25_25755


namespace a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l25_25557

theorem a1_minus_2a2_plus_3a3_minus_4a4_eq_48:
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (∀ x : ℝ, (1 + 2 * x) ^ 4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 = 48 :=
by
  sorry

end a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l25_25557


namespace bounded_expression_l25_25856

theorem bounded_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := 
sorry

end bounded_expression_l25_25856


namespace chatterboxes_total_jokes_l25_25239

theorem chatterboxes_total_jokes :
  let num_chatterboxes := 10
  let jokes_increasing := (100 * (100 + 1)) / 2
  let jokes_decreasing := (99 * (99 + 1)) / 2
  (jokes_increasing + jokes_decreasing) / num_chatterboxes = 1000 :=
by
  sorry

end chatterboxes_total_jokes_l25_25239


namespace find_f_neg_log3_5_l25_25693

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x >= 0 then 3^x + m else -f (-x)

theorem find_f_neg_log3_5 (m : ℝ) (h_odd : ∀ x : ℝ, f x m = -f (-x) m)
  (hx_geq_0 : ∀ x : ℝ, x ≥ 0 → f x m = 3^x + m) :
  ∃ m : ℝ, f (-Real.logb 5 3) m = -4 :=
sorry

end find_f_neg_log3_5_l25_25693


namespace no_seq_for_lambda_le_e_l25_25429

open Real

theorem no_seq_for_lambda_le_e (λ : ℝ) (hλ : λ ∈ Ioo 0 (exp 1)) :
    ∀ (a : ℕ → ℝ), (∀ n ≥ 2, 0 < a n) → ¬ (∀ n ≥ 2, a n + 1 ≤ Real.sqrt n λ * a (n-1)) := by
    sorry

end no_seq_for_lambda_le_e_l25_25429


namespace problem1_problem2_l25_25543

noncomputable section

theorem problem1 :
  (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 :=
  sorry

theorem problem2 :
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1 / 2) = -6 * Real.sqrt 5 :=
  sorry

end problem1_problem2_l25_25543


namespace constructible_triangle_and_area_bound_l25_25850

noncomputable def triangle_inequality_sine (α β γ : ℝ) : Prop :=
  (Real.sin α + Real.sin β > Real.sin γ) ∧
  (Real.sin β + Real.sin γ > Real.sin α) ∧
  (Real.sin γ + Real.sin α > Real.sin β)

theorem constructible_triangle_and_area_bound 
  (α β γ : ℝ) (h_pos : 0 < α) (h_pos_β : 0 < β) (h_pos_γ : 0 < γ)
  (h_sum : α + β + γ < Real.pi)
  (h_ineq1 : α + β > γ)
  (h_ineq2 : β + γ > α)
  (h_ineq3 : γ + α > β) :
  triangle_inequality_sine α β γ ∧
  (Real.sin α * Real.sin β * Real.sin γ) / 4 ≤ (1 / 8) * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) :=
sorry

end constructible_triangle_and_area_bound_l25_25850


namespace complement_union_eq_l25_25741

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {3, 5}

theorem complement_union_eq :
  (U \ (M ∪ N)) = {2, 4} := by
  sorry

end complement_union_eq_l25_25741


namespace mushrooms_on_log_l25_25408

theorem mushrooms_on_log :
  ∃ (G : ℕ), ∃ (S : ℕ), S = 9 * G ∧ G + S = 30 ∧ G = 3 :=
by
  sorry

end mushrooms_on_log_l25_25408


namespace non_degenerate_ellipse_l25_25425

theorem non_degenerate_ellipse (x y k : ℝ) : (∃ k, (2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k) → k > -135 / 4) := sorry

end non_degenerate_ellipse_l25_25425


namespace square_side_length_l25_25201

theorem square_side_length (radius : ℝ) (s1 s2 : ℝ) (h1 : s1 = s2) (h2 : radius = 2 - Real.sqrt 2):
  s1 = 1 :=
  sorry

end square_side_length_l25_25201


namespace number_of_diagonals_l25_25084

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25084


namespace number_of_diagonals_l25_25083

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25083


namespace diagonals_in_regular_nine_sided_polygon_l25_25072

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25072


namespace notebook_cost_l25_25186

theorem notebook_cost (total_spent ruler_cost pencil_count pencil_cost: ℕ)
  (h1 : total_spent = 74)
  (h2 : ruler_cost = 18)
  (h3 : pencil_count = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (ruler_cost + pencil_count * pencil_cost) = 35 := 
by 
  sorry

end notebook_cost_l25_25186


namespace diagonals_in_nine_sided_polygon_l25_25056

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25056


namespace find_a_b_for_odd_function_l25_25309

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l25_25309


namespace diagonals_in_regular_nine_sided_polygon_l25_25009

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25009


namespace candidates_count_l25_25913

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
by sorry

end candidates_count_l25_25913


namespace smallest_solution_of_quartic_l25_25894

theorem smallest_solution_of_quartic :
  ∃ x : ℝ, x^4 - 40*x^2 + 144 = 0 ∧ ∀ y : ℝ, (y^4 - 40*y^2 + 144 = 0) → x ≤ y :=
sorry

end smallest_solution_of_quartic_l25_25894


namespace greatest_possible_sum_xy_l25_25220

noncomputable def greatest_possible_xy (x y : ℝ) :=
  x^2 + y^2 = 100 ∧ xy = 40 → x + y = 6 * Real.sqrt 5

theorem greatest_possible_sum_xy {x y : ℝ} (h1 : x^2 + y^2 = 100) (h2 : xy = 40) :
  x + y ≤ 6 * Real.sqrt 5 :=
sorry

end greatest_possible_sum_xy_l25_25220


namespace who_finished_in_7th_place_l25_25456

theorem who_finished_in_7th_place:
  ∀ (Alex Ben Charlie David Ethan : ℕ),
  (Ethan + 4 = Alex) →
  (David + 1 = Ben) →
  (Charlie = Ben + 3) →
  (Alex = Ben + 2) →
  (Ethan + 2 = David) →
  (Ben = 5) →
  Alex = 7 :=
by
  intros Alex Ben Charlie David Ethan h1 h2 h3 h4 h5 h6
  sorry

end who_finished_in_7th_place_l25_25456


namespace isabella_original_hair_length_l25_25846

-- Define conditions from the problem
def isabella_current_hair_length : ℕ := 9
def hair_cut_length : ℕ := 9

-- The proof problem to show original hair length equals 18 inches
theorem isabella_original_hair_length 
  (hc : isabella_current_hair_length = 9)
  (ht : hair_cut_length = 9) : 
  isabella_current_hhair_length + hair_cut_length = 18 := 
sorry

end isabella_original_hair_length_l25_25846


namespace integer_distances_implies_vertex_l25_25632

theorem integer_distances_implies_vertex (M A B C D : ℝ × ℝ × ℝ)
  (a b c d : ℕ)
  (h_tetrahedron: 
    dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2 ∧ 
    dist A C = 2 ∧ dist B D = 2)
  (h_distances: 
    dist M A = a ∧ dist M B = b ∧ dist M C = c ∧ dist M D = d) :
  M = A ∨ M = B ∨ M = C ∨ M = D := 
  sorry

end integer_distances_implies_vertex_l25_25632


namespace operation_three_six_l25_25953

theorem operation_three_six : (3 * 3 * 6) / (3 + 6) = 6 :=
by
  calc (3 * 3 * 6) / (3 + 6) = 6 := sorry

end operation_three_six_l25_25953


namespace margaret_speed_on_time_l25_25743
-- Import the necessary libraries from Mathlib

-- Define the problem conditions and state the theorem
theorem margaret_speed_on_time :
  ∃ r : ℝ, (∀ d t : ℝ,
    d = 50 * (t - 1/12) ∧
    d = 30 * (t + 1/12) →
    r = d / t) ∧
  r = 37.5 := 
sorry

end margaret_speed_on_time_l25_25743


namespace scrap_metal_collected_l25_25879

theorem scrap_metal_collected (a b : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9)
  (h₂ : 900 + 10 * a + b - (100 * a + 10 * b + 9) = 216) :
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by
  sorry

end scrap_metal_collected_l25_25879


namespace find_z_value_l25_25296

theorem find_z_value (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1/w = 15) (h2 : w^2 + 1/z = 3) : z = 44/3 := 
by 
  sorry

end find_z_value_l25_25296


namespace rate_of_interest_l25_25193

-- Define the conditions
def P : ℝ := 1200
def SI : ℝ := 432
def T (R : ℝ) : ℝ := R

-- Define the statement to be proven
theorem rate_of_interest (R : ℝ) (h : SI = (P * R * T R) / 100) : R = 6 :=
by sorry

end rate_of_interest_l25_25193


namespace evaluate_expression_l25_25218

theorem evaluate_expression :
  54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 :=
by
  sorry

end evaluate_expression_l25_25218


namespace plane_intercept_equation_l25_25351

-- Define the conditions in Lean 4
variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- State the main theorem
theorem plane_intercept_equation :
  ∃ (p : ℝ → ℝ → ℝ → ℝ), (∀ x y z, p x y z = x / a + y / b + z / c) :=
sorry

end plane_intercept_equation_l25_25351


namespace roots_of_quadratic_sum_of_sixth_powers_l25_25176

theorem roots_of_quadratic_sum_of_sixth_powers {u v : ℝ} 
  (h₀ : u^2 - 2*u*Real.sqrt 3 + 1 = 0)
  (h₁ : v^2 - 2*v*Real.sqrt 3 + 1 = 0)
  : u^6 + v^6 = 970 := 
by 
  sorry

end roots_of_quadratic_sum_of_sixth_powers_l25_25176


namespace complex_imaginary_part_l25_25349

theorem complex_imaginary_part (z : ℂ) (h : z + (3 - 4 * I) = 1) : z.im = 4 :=
  sorry

end complex_imaginary_part_l25_25349


namespace determine_a_b_odd_function_l25_25324

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l25_25324


namespace nine_sided_polygon_diagonals_l25_25053

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25053


namespace find_a_b_l25_25338

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l25_25338


namespace minimum_value_of_expression_l25_25473

noncomputable def min_value (p q r s t u : ℝ) : ℝ :=
  (1 / p) + (9 / q) + (25 / r) + (49 / s) + (81 / t) + (121 / u)

theorem minimum_value_of_expression (p q r s t u : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hs : 0 < s) (ht : 0 < t) (hu : 0 < u) (h_sum : p + q + r + s + t + u = 11) :
  min_value p q r s t u ≥ 1296 / 11 :=
by sorry

end minimum_value_of_expression_l25_25473


namespace hyperbola_eccentricity_correct_l25_25267

noncomputable def hyperbola_eccentricity : Real :=
  let a := 5
  let b := 4
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  c / a

theorem hyperbola_eccentricity_correct :
  hyperbola_eccentricity = Real.sqrt 41 / 5 :=
by
  sorry

end hyperbola_eccentricity_correct_l25_25267


namespace regular_nonagon_diagonals_correct_l25_25113

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25113


namespace decreasing_population_density_l25_25900

def Population (t : Type) : Type := t

variable (stable_period: Prop)
variable (infertility: Prop)
variable (death_rate_exceeds_birth_rate: Prop)
variable (complex_structure: Prop)

theorem decreasing_population_density :
  death_rate_exceeds_birth_rate → true := sorry

end decreasing_population_density_l25_25900


namespace number_100_in_row_15_l25_25353

theorem number_100_in_row_15 (A : ℕ) (H1 : 1 ≤ A)
  (H2 : ∀ n : ℕ, n > 0 → n ≤ 100 * A)
  (H3 : ∃ k : ℕ, 4 * A + 1 ≤ 31 ∧ 31 ≤ 5 * A ∧ k = 5):
  ∃ r : ℕ, (14 * A + 1 ≤ 100 ∧ 100 ≤ 15 * A ∧ r = 15) :=
by {
  sorry
}

end number_100_in_row_15_l25_25353


namespace min_value_at_x_eq_2_l25_25651

theorem min_value_at_x_eq_2 (x : ℝ) (h : x > 1) : 
  x + 1/(x-1) = 3 ↔ x = 2 :=
by sorry

end min_value_at_x_eq_2_l25_25651


namespace problem_l25_25578

variable (a b c d : ℕ)

theorem problem (h1 : a + b = 12) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 6 :=
sorry

end problem_l25_25578


namespace odd_function_a_b_l25_25328

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l25_25328


namespace find_A_minus_B_l25_25257

variables (A B : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + B = 814.8
def condition2 : Prop := B = A / 10

-- Statement to prove
theorem find_A_minus_B (h1 : condition1 A B) (h2 : condition2 A B) : A - B = 611.1 :=
sorry

end find_A_minus_B_l25_25257


namespace same_color_combination_probability_184_323_gcd_184_323_prime_m_plus_n_l25_25241

noncomputable def same_color_combination_probability (red : ℕ) (blue : ℕ) : ℚ :=
  let total_combinations := (red + blue).choose 2
  let lucy_red := red.choose 2 / total_combinations
  let john_red := (red - 2).choose 2 / (red + blue - 2).choose 2
  let both_red := lucy_red * john_red
  let lucy_blue := blue.choose 2 / total_combinations
  let john_blue := (blue - 2).choose 2 / (red + blue - 2).choose 2
  let both_blue := lucy_blue * john_blue
  let different_colors := ((red * blue).choose 2 * 2) / total_combinations
  (both_red + both_blue + different_colors)

theorem same_color_combination_probability_184_323 :
  ∀ (red blue : ℕ), red = 12 → blue = 8 → 
  same_color_combination_probability red blue = 184 / 323 :=
by
  intros red blue hred hblue
  rw [hred, hblue]
  sorry

theorem gcd_184_323_prime :
  Nat.gcd 184 323 = 1 :=
by
  sorry

theorem m_plus_n :
  ∀ (red blue : ℕ), red = 12 → blue = 8 → 
  ∑ m n, m = 184 → n = 323 → m + n = 507 :=
by
  intros red blue hred hblue m n hm hn
  rw [hm, hn]
  sorry

end same_color_combination_probability_184_323_gcd_184_323_prime_m_plus_n_l25_25241


namespace odd_function_a_b_l25_25329

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l25_25329


namespace distinct_ordered_pairs_eq_49_l25_25298

theorem distinct_ordered_pairs_eq_49 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 49) (hy : 1 ≤ y ∧ y ≤ 49) (h_eq : x + y = 50) :
  ∃ xs : List (ℕ × ℕ), (∀ p ∈ xs, p.1 + p.2 = 50 ∧ 1 ≤ p.1 ∧ p.1 ≤ 49 ∧ 1 ≤ p.2 ∧ p.2 ≤ 49) ∧ xs.length = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l25_25298


namespace sachin_is_younger_by_8_years_l25_25196

variable (S R : ℕ)

-- Conditions
axiom age_of_sachin : S = 28
axiom ratio_of_ages : S * 9 = R * 7

-- Goal
theorem sachin_is_younger_by_8_years (S R : ℕ) (h1 : S = 28) (h2 : S * 9 = R * 7) : R - S = 8 :=
by
  sorry

end sachin_is_younger_by_8_years_l25_25196


namespace eval_expression_in_second_quadrant_l25_25297

theorem eval_expression_in_second_quadrant (α : ℝ) (h1 : π/2 < α ∧ α < π) (h2 : Real.sin α > 0) (h3 : Real.cos α < 0) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / (Real.sin α) ^ 2 - 1) = -1 :=
by
  sorry

end eval_expression_in_second_quadrant_l25_25297


namespace geometric_series_sum_l25_25789

theorem geometric_series_sum :
  let a := (1 : ℝ) / 5
  let r := -(1 : ℝ) / 5
  let n := 5
  let S_n := (a * (1 - r ^ n)) / (1 - r)
  S_n = 521 / 3125 := by
  sorry

end geometric_series_sum_l25_25789


namespace perpendicular_lines_condition_l25_25287

theorem perpendicular_lines_condition (m : ℝ) :
    (m = 1 → (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m * x + y - 1) = 0 → d * (x - m * y - 1) = 0 → (c * m + d / m) ^ 2 = 1))) ∧ (∀ (m' : ℝ), m' ≠ 1 → ¬ (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m' * x + y - 1) = 0 → d * (x - m' * y - 1) = 0 → (c * m' + d / m') ^ 2 = 1))) :=
by
  sorry

end perpendicular_lines_condition_l25_25287


namespace mariel_dogs_count_l25_25185

theorem mariel_dogs_count (total_legs : ℤ) (num_dog_walkers : ℤ) (legs_per_walker : ℤ) 
  (other_dogs_count : ℤ) (legs_per_dog : ℤ) (mariel_dogs : ℤ) :
  total_legs = 36 →
  num_dog_walkers = 2 →
  legs_per_walker = 2 →
  other_dogs_count = 3 →
  legs_per_dog = 4 →
  mariel_dogs = (total_legs - (num_dog_walkers * legs_per_walker + other_dogs_count * legs_per_dog)) / legs_per_dog →
  mariel_dogs = 5 :=
by
  intros
  sorry

end mariel_dogs_count_l25_25185


namespace question1_question2_application_l25_25508

theorem question1: (-4)^2 - (-3) * (-5) = 1 := by
  sorry

theorem question2 (a : ℝ) (h : a = -4) : a^2 - (a + 1) * (a - 1) = 1 := by
  sorry

theorem application (a : ℝ) (h : a = 1.35) : a * (a - 1) * 2 * a - a^3 - a * (a - 1)^2 = -1.35 := by
  sorry

end question1_question2_application_l25_25508


namespace meaningful_fraction_l25_25763

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 5 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end meaningful_fraction_l25_25763


namespace problem_statement_l25_25560

noncomputable def min_expression_value (θ1 θ2 θ3 θ4 : ℝ) : ℝ :=
  (2 * (Real.sin θ1)^2 + 1 / (Real.sin θ1)^2) *
  (2 * (Real.sin θ2)^2 + 1 / (Real.sin θ2)^2) *
  (2 * (Real.sin θ3)^2 + 1 / (Real.sin θ3)^2) *
  (2 * (Real.sin θ4)^2 + 1 / (Real.sin θ4)^2)

theorem problem_statement (θ1 θ2 θ3 θ4 : ℝ) (h_pos: θ1 > 0 ∧ θ2 > 0 ∧ θ3 > 0 ∧ θ4 > 0) (h_sum: θ1 + θ2 + θ3 + θ4 = Real.pi) :
  min_expression_value θ1 θ2 θ3 θ4 = 81 :=
sorry

end problem_statement_l25_25560


namespace find_f_neg1_l25_25942

-- Definition of odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f) (h_f1 : f 1 = 2)

-- Theorem stating the necessary proof
theorem find_f_neg1 : f (-1) = -2 :=
by
  sorry

end find_f_neg1_l25_25942


namespace samuel_initial_speed_l25_25870

/-
Samuel is driving to San Francisco’s Comic-Con in his car and he needs to travel 600 miles to the hotel where he made a reservation. 
He drives at a certain speed for 3 hours straight, then he speeds up to 80 miles/hour for 4 hours. 
Now, he is 130 miles away from the hotel. What was his initial speed?
-/

theorem samuel_initial_speed : 
  ∃ v : ℝ, (3 * v + 320 = 470) ↔ (v = 50) :=
by
  use 50
  /- detailed proof goes here -/
  sorry

end samuel_initial_speed_l25_25870


namespace product_of_roots_l25_25225

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 9 * x + 20

-- The main statement for the Lean theorem
theorem product_of_roots : (∃ x₁ x₂ : ℝ, quadratic x₁ = 0 ∧ quadratic x₂ = 0 ∧ x₁ * x₂ = 20) :=
by
  sorry

end product_of_roots_l25_25225


namespace david_marks_in_biology_l25_25923

theorem david_marks_in_biology (english: ℕ) (math: ℕ) (physics: ℕ) (chemistry: ℕ) (average: ℕ) (biology: ℕ) :
  english = 81 ∧ math = 65 ∧ physics = 82 ∧ chemistry = 67 ∧ average = 76 → (biology = 85) :=
by
  sorry

end david_marks_in_biology_l25_25923


namespace time_away_is_43point64_minutes_l25_25531

theorem time_away_is_43point64_minutes :
  ∃ (n1 n2 : ℝ), 
    (195 + n1 / 2 - 6 * n1 = 120 ∨ 195 + n1 / 2 - 6 * n1 = -120) ∧
    (195 + n2 / 2 - 6 * n2 = 120 ∨ 195 + n2 / 2 - 6 * n2 = -120) ∧
    n1 ≠ n2 ∧
    n1 < 60 ∧
    n2 < 60 ∧
    |n2 - n1| = 43.64 :=
sorry

end time_away_is_43point64_minutes_l25_25531


namespace probability_of_circle_l25_25867

theorem probability_of_circle :
  let numCircles := 4
  let numSquares := 3
  let numTriangles := 3
  let totalFigures := numCircles + numSquares + numTriangles
  let probability := numCircles / totalFigures
  probability = 2 / 5 :=
by
  sorry

end probability_of_circle_l25_25867


namespace find_n_l25_25927

theorem find_n (n : ℕ) (h₁ : 2^6 * 3^3 * n = factorial 10) : n = 2100 :=
sorry

end find_n_l25_25927


namespace y_decreases_as_x_increases_l25_25713

-- Define the function y = 7 - x
def my_function (x : ℝ) : ℝ := 7 - x

-- Prove that y decreases as x increases
theorem y_decreases_as_x_increases : ∀ x1 x2 : ℝ, x1 < x2 → my_function x1 > my_function x2 := by
  intro x1 x2 h
  unfold my_function
  sorry

end y_decreases_as_x_increases_l25_25713


namespace cost_of_60_tulips_l25_25256

-- Definition of conditions
def cost_of_bouquet (n : ℕ) : ℝ :=
  if n ≤ 40 then n * 2
  else 40 * 2 + (n - 40) * 3

-- The main statement
theorem cost_of_60_tulips : cost_of_bouquet 60 = 140 := by
  sorry

end cost_of_60_tulips_l25_25256


namespace train_speed_is_144_l25_25251

-- Definitions for the conditions
def length_of_train_passing_pole (S : ℝ) := S * 8
def length_of_train_passing_stationary_train (S : ℝ) := S * 18 - 400

-- The main theorem to prove the speed of the train
theorem train_speed_is_144 (S : ℝ) :
  (length_of_train_passing_pole S = length_of_train_passing_stationary_train S) →
  (S * 3.6 = 144) :=
by
  sorry

end train_speed_is_144_l25_25251


namespace school_competition_l25_25592

theorem school_competition :
  (∃ n : ℕ, 
    n > 0 ∧ 
    75% students did not attend the competition ∧
    10% of those who attended participated in both competitions ∧
    ∃ y z : ℕ, y = 3 / 2 * z ∧ 
    y + z - (1 / 10) * (n / 4) = n / 4
  ) → n = 200 :=
sorry

end school_competition_l25_25592


namespace packs_of_yellow_balls_l25_25973

theorem packs_of_yellow_balls (Y : ℕ) : 
  3 * 19 + Y * 19 + 8 * 19 = 399 → Y = 10 :=
by sorry

end packs_of_yellow_balls_l25_25973


namespace solution_set_of_log_inequality_l25_25468

noncomputable def log_a (a x : ℝ) : ℝ := sorry -- The precise definition of the log base 'a' is skipped for brevity.

theorem solution_set_of_log_inequality (a x : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (h_max : ∃ y, log_a a (y^2 - 2*y + 3) = y):
  log_a a (x - 1) > 0 ↔ (1 < x ∧ x < 2) :=
sorry

end solution_set_of_log_inequality_l25_25468


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25011

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25011


namespace find_alpha_l25_25811

theorem find_alpha (α : Real) (hα : 0 < α ∧ α < π) :
  (∃ x : Real, (|2 * x - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * x| = Real.sin α) ∧ 
  ∀ y : Real, (|2 * y - 1 / 2| + |(Real.sqrt 6 - Real.sqrt 2) * y| = Real.sin α) → y = x) →
  α = π / 12 ∨ α = 11 * π / 12 :=
by
  sorry

end find_alpha_l25_25811


namespace sqrt_condition_l25_25707

theorem sqrt_condition (x : ℝ) : (3 * x - 5 ≥ 0) → (x ≥ 5 / 3) :=
by
  intros h
  have h1 : 3 * x ≥ 5 := by linarith
  have h2 : x ≥ 5 / 3 := by linarith
  exact h2

end sqrt_condition_l25_25707


namespace composite_10201_base_n_composite_10101_base_n_l25_25484

-- 1. Prove that 10201_n is composite given n > 2
theorem composite_10201_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + 2*n^2 + 1 := 
sorry

-- 2. Prove that 10101_n is composite given n > 2.
theorem composite_10101_base_n (n : ℕ) (h : n > 2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n^4 + n^2 + 1 := 
sorry

end composite_10201_base_n_composite_10101_base_n_l25_25484


namespace equation_of_parabola_l25_25933

def parabola_vertex_form_vertex (a x y : ℝ) := y = a * (x - 3)^2 - 2
def parabola_passes_through_point (a : ℝ) := 1 = a * (0 - 3)^2 - 2
def parabola_equation (y x : ℝ) := y = (1/3) * x^2 - 2 * x + 1

theorem equation_of_parabola :
  ∃ a : ℝ,
    ∀ x y : ℝ,
      parabola_vertex_form_vertex a x y ∧
      parabola_passes_through_point a →
      parabola_equation y x :=
by
  sorry

end equation_of_parabola_l25_25933


namespace find_constants_for_odd_function_l25_25343

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l25_25343


namespace total_lunch_bill_l25_25639

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h_hd : cost_hotdog = 5.36) (h_sd : cost_salad = 5.10) :
  cost_hotdog + cost_salad = 10.46 :=
by
  sorry

end total_lunch_bill_l25_25639


namespace find_specific_n_l25_25684

theorem find_specific_n :
  ∀ (n : ℕ), (∃ (a b : ℤ), n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by {
  sorry
}

end find_specific_n_l25_25684


namespace part1_part2_l25_25236

def custom_operation (a b : ℝ) : ℝ := a^2 + 2*a*b

theorem part1 : custom_operation 2 3 = 16 :=
by sorry

theorem part2 (x : ℝ) (h : custom_operation (-2) x = -2 + x) : x = 6 / 5 :=
by sorry

end part1_part2_l25_25236


namespace calculate_expression_l25_25676

-- Defining the main theorem to prove
theorem calculate_expression (a b : ℝ) : 
  3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by 
  sorry

end calculate_expression_l25_25676


namespace interest_rate_correct_l25_25864

noncomputable def annual_interest_rate : ℝ :=
  4^(1/10) - 1

theorem interest_rate_correct (P A₁₀ A₁₅ : ℝ) (h₁ : P = 6000) (h₂ : A₁₀ = 24000) (h₃ : A₁₅ = 48000) :
  (P * (1 + annual_interest_rate)^10 = A₁₀) ∧ (P * (1 + annual_interest_rate)^15 = A₁₅) :=
by
  sorry

end interest_rate_correct_l25_25864


namespace number_of_diagonals_l25_25089

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25089


namespace diagonals_in_nine_sided_polygon_l25_25061

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25061


namespace area_of_inscribed_square_l25_25374

theorem area_of_inscribed_square :
  let parabola := λ x => x^2 - 10 * x + 21
  ∃ (t : ℝ), parabola (5 + t) = -2 * t ∧ (2 * t)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end area_of_inscribed_square_l25_25374


namespace find_a_b_l25_25339

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l25_25339


namespace diagonals_in_regular_nine_sided_polygon_l25_25005

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25005


namespace average_score_10_students_l25_25704

theorem average_score_10_students (x : ℝ)
  (h1 : 15 * 70 = 1050)
  (h2 : 25 * 78 = 1950)
  (h3 : 15 * 70 + 10 * x = 25 * 78) :
  x = 90 :=
sorry

end average_score_10_students_l25_25704


namespace tan_sum_simplification_l25_25986
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l25_25986


namespace triangle_solid_revolution_correct_l25_25570

noncomputable def triangle_solid_revolution (t : ℝ) (alpha beta gamma : ℝ) (longest_side : string) : ℝ × ℝ :=
  let pi := Real.pi;
  let sin := Real.sin;
  let cos := Real.cos;
  let sqrt := Real.sqrt;
  let to_rad (x : ℝ) : ℝ := x * pi / 180;
  let alpha_rad := to_rad alpha;
  let beta_rad := to_rad beta;
  let gamma_rad := to_rad gamma;
  let a := sqrt (2 * t * sin alpha_rad / (sin beta_rad * sin gamma_rad));
  let b := sqrt (2 * t * sin beta_rad / (sin gamma_rad * sin alpha_rad));
  let m_c := sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  let F := 2 * pi * t * cos ((alpha_rad - beta_rad) / 2) / sin (gamma_rad / 2);
  let K := 2 * pi / 3 * t * sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  (F, K)

theorem triangle_solid_revolution_correct :
  triangle_solid_revolution 80.362 (39 + 34/60 + 30/3600) (60 : ℝ) (80 + 25/60 + 30/3600) "c" = (769.3, 1595.3) :=
sorry

end triangle_solid_revolution_correct_l25_25570


namespace total_cars_l25_25861

-- Conditions
def initial_cars : ℕ := 150
def uncle_cars : ℕ := 5
def grandpa_cars : ℕ := 2 * uncle_cars
def dad_cars : ℕ := 10
def mum_cars : ℕ := dad_cars + 5
def auntie_cars : ℕ := 6

-- Proof statement (theorem)
theorem total_cars : initial_cars + (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars) = 196 :=
by
  sorry

end total_cars_l25_25861


namespace jennifer_marbles_l25_25718

noncomputable def choose_ways (total special non_special choose_total choose_special choose_non_special : ℕ) : ℕ :=
  let ways_special := choose_special * choose_non_special
  ways_special

theorem jennifer_marbles :
  let total := 20
  let red := 3
  let green := 3
  let blue := 2
  let special := red + green + blue
  let non_special := total - special
  let choose_total := 5
  let choose_special := 2
  let choose_non_special := choose_total - choose_special
  let ways_special :=
    (Nat.choose red 2) + (Nat.choose green 2) + (Nat.choose blue 2) +
    ((Nat.choose red 1) * (Nat.choose green 1)) +
    ((Nat.choose red 1) * (Nat.choose blue 1)) +
    ((Nat.choose green 1) * (Nat.choose blue 1))
  let ways_non_special := Nat.choose non_special 3
  choose_ways total special non_special choose_total choose_special choose_non_special = 6160 :=
by
  simp only [choose_ways]
  exact sorry

end jennifer_marbles_l25_25718


namespace quadratic_vertex_transform_l25_25759

theorem quadratic_vertex_transform {p q r m k : ℝ} (h : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - h)^2 + k) →
  h = -3 :=
by
  intros h1 h2
  -- The actual proof goes here
  sorry

end quadratic_vertex_transform_l25_25759


namespace find_ks_l25_25358

theorem find_ks (n : ℕ) (h_pos : 0 < n) :
  ∀ k, k ∈ (Finset.range (2 * n * n + 1)).erase 0 ↔ (n^2 - n + 1 ≤ k ∧ k ≤ n^2) ∨ (2*n ∣ k ∧ k ≥ n^2 - n + 1) :=
sorry

end find_ks_l25_25358


namespace students_neither_l25_25477

-- Define the conditions
def total_students : ℕ := 60
def students_math : ℕ := 40
def students_physics : ℕ := 35
def students_both : ℕ := 25

-- Define the problem statement
theorem students_neither : total_students - ((students_math - students_both) + (students_physics - students_both) + students_both) = 10 :=
by
  sorry

end students_neither_l25_25477


namespace regular_nine_sided_polygon_diagonals_l25_25043

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25043


namespace diagonals_in_nine_sided_polygon_l25_25057

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25057


namespace restore_catalogue_numbers_impossible_l25_25784

theorem restore_catalogue_numbers_impossible :
  ¬ (∀ (f : ℕ → ℕ), (∀ (x y : ℕ), 2 ≤ x ∧ x ≤ 2000 ∧ 2 ≤ y ∧ y ≤ 2000 → f (x, y) = Nat.gcd x y) → (∀ a b, 2 ≤ a ∧ a ≤ 2000 ∧ 2 ≤ b ∧ b ≤ 2000 → a = b)) :=
by
  sorry

end restore_catalogue_numbers_impossible_l25_25784


namespace Oo_remains_stationary_l25_25417

-- Definitions and conditions of the problem
structure Point :=
(x : ℝ)
(y : ℝ)

-- Fixed points A and B
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}

-- Moving point O with coordinates (ox, oy)
variable (ox oy : ℝ)
def O : Point := {x := ox, y := oy}

-- Define points A' and B' with given conditions
def A' : Point := {x := ox, y := oy}
def B' : Point := {x := ox, y := oy}

-- Midpoint O' of A'B'
def O' : Point := {x := (A'.x + B'.x) / 2, y := (A'.y + B'.y) / 2}

-- Given conditions on angles and distances
axiom angle_OAA' : ∃ θ : ℝ, θ = 90
axiom angle_OBB' : ∃ θ : ℝ, θ = 90
axiom distance_AA' : ∀ (A O : Point), A'.x - A.x = O.x - A.x ∧ A'.y - A.y = O.y - A.y
axiom distance_BB' : ∀ (B O : Point), B'.x - B.x = O.x - B.x ∧ B'.y - B.y = O.y - B.y

-- Theorem to be proved
theorem Oo_remains_stationary :
  ∀ (ox oy : ℝ), O' = {x := (A'.x + B'.x) / 2, y := (A'.y + B'.y) / 2} :=
by
  sorry

end Oo_remains_stationary_l25_25417


namespace copies_made_in_half_hour_l25_25770

theorem copies_made_in_half_hour :
  let copies_per_minute_machine1 := 40
  let copies_per_minute_machine2 := 55
  let time_minutes := 30
  (copies_per_minute_machine1 * time_minutes) + (copies_per_minute_machine2 * time_minutes) = 2850 := by
    sorry

end copies_made_in_half_hour_l25_25770


namespace coefficient_a_must_be_zero_l25_25631

noncomputable def all_real_and_positive_roots (a b c : ℝ) : Prop :=
∀ p : ℝ, p > 0 → ∀ x : ℝ, (a * x^2 + b * x + c + p = 0) → x > 0

theorem coefficient_a_must_be_zero (a b c : ℝ) :
  (all_real_and_positive_roots a b c) → (a = 0) :=
by sorry

end coefficient_a_must_be_zero_l25_25631


namespace total_modules_in_stock_l25_25152

-- Given conditions
def module_cost_high : ℝ := 10
def module_cost_low : ℝ := 3.5
def total_stock_value : ℝ := 45
def low_module_count : ℕ := 10

-- To be proved: total number of modules in stock
theorem total_modules_in_stock (x : ℕ) (y : ℕ) (h1 : y = low_module_count) 
  (h2 : module_cost_high * x + module_cost_low * y = total_stock_value) : 
  x + y = 11 := 
sorry

end total_modules_in_stock_l25_25152


namespace sum_of_interior_angles_of_octagon_l25_25650

theorem sum_of_interior_angles_of_octagon (n : ℕ) (h : n = 8) : (n - 2) * 180 = 1080 := by
  sorry

end sum_of_interior_angles_of_octagon_l25_25650


namespace tan_sum_pi_over_12_l25_25992

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l25_25992


namespace concert_songs_l25_25958

def total_songs (g : ℕ) : ℕ := (9 + 3 + 9 + g) / 3

theorem concert_songs 
  (g : ℕ) 
  (h1 : 9 + 3 + 9 + g = 3 * total_songs g) 
  (h2 : 3 + g % 4 = 0) 
  (h3 : 4 ≤ g ∧ g ≤ 9) 
  : total_songs g = 9 ∨ total_songs g = 10 := 
sorry

end concert_songs_l25_25958


namespace find_f91_plus_fm91_l25_25472

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

theorem find_f91_plus_fm91 (a b c : ℝ) (h : f 91 a b c = 1) : f 91 a b c + f (-91) a b c = 2 := by
  sorry

end find_f91_plus_fm91_l25_25472


namespace fifth_equation_sum_first_17_even_sum_even_28_to_50_l25_25858

-- Define a function to sum the first n even numbers
def sum_even (n : ℕ) : ℕ := n * (n + 1)

-- Part (1) According to the pattern, write down the ⑤th equation
theorem fifth_equation : sum_even 5 = 30 := by
  sorry

-- Part (2) Calculate according to this pattern:
-- ① Sum of first 17 even numbers
theorem sum_first_17_even : sum_even 17 = 306 := by
  sorry

-- ② Sum of even numbers from 28 to 50
theorem sum_even_28_to_50 : 
  let sum_even_50 := sum_even 25
  let sum_even_26 := sum_even 13
  sum_even_50 - sum_even_26 = 468 := by
  sorry

end fifth_equation_sum_first_17_even_sum_even_28_to_50_l25_25858


namespace order_of_means_l25_25949

variables (a b : ℝ)
-- a and b are positive and unequal
axiom h1 : 0 < a
axiom h2 : 0 < b
axiom h3 : a ≠ b

-- Definitions of the means
noncomputable def AM : ℝ := (a + b) / 2
noncomputable def GM : ℝ := Real.sqrt (a * b)
noncomputable def HM : ℝ := (2 * a * b) / (a + b)
noncomputable def QM : ℝ := Real.sqrt ((a^2 + b^2) / 2)

-- The theorem to prove the order of the means
theorem order_of_means (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  QM a b > AM a b ∧ AM a b > GM a b ∧ GM a b > HM a b :=
sorry

end order_of_means_l25_25949


namespace negation_of_exists_lt_l25_25877

theorem negation_of_exists_lt :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 3 < 0) = (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0) :=
by sorry

end negation_of_exists_lt_l25_25877


namespace age_ratio_in_six_years_l25_25505

-- Definitions for Claire's and Pete's current ages
variables (c p : ℕ)

-- Conditions given in the problem
def condition1 : Prop := c - 3 = 2 * (p - 3)
def condition2 : Prop := p - 7 = (1 / 4) * (c - 7)

-- The proof problem statement
theorem age_ratio_in_six_years (c p : ℕ) (h1 : condition1 c p) (h2 : condition2 c p) : 
  (c + 6) = 3 * (p + 6) :=
sorry

end age_ratio_in_six_years_l25_25505


namespace max_x2y_l25_25938

noncomputable def maximum_value_x_squared_y (x y : ℝ) : ℝ :=
  if x ∈ Set.Ici 0 ∧ y ∈ Set.Ici 0 ∧ x^3 + y^3 + 3*x*y = 1 then x^2 * y else 0

theorem max_x2y (x y : ℝ) (h1 : x ∈ Set.Ici 0) (h2 : y ∈ Set.Ici 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  maximum_value_x_squared_y x y = 4 / 27 :=
sorry

end max_x2y_l25_25938


namespace a_sub_b_eq_2_l25_25140

theorem a_sub_b_eq_2 (a b : ℝ)
  (h : (a - 5) ^ 2 + |b ^ 3 - 27| = 0) : a - b = 2 :=
by
  sorry

end a_sub_b_eq_2_l25_25140


namespace probability_light_change_l25_25914

noncomputable def total_cycle_duration : ℕ := 45 + 5 + 50
def change_intervals : ℕ := 15

theorem probability_light_change :
  (15 : ℚ) / total_cycle_duration = 3 / 20 :=
by
  sorry

end probability_light_change_l25_25914


namespace sum_of_fractions_l25_25561

theorem sum_of_fractions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/2 + x) + f (1/2 - x) = 2) :
  f (1 / 8) + f (2 / 8) + f (3 / 8) + f (4 / 8) + 
  f (5 / 8) + f (6 / 8) + f (7 / 8) = 7 :=
by 
  sorry

end sum_of_fractions_l25_25561


namespace certain_number_calculation_l25_25955

theorem certain_number_calculation (x : ℝ) (h : (15 * x) / 100 = 0.04863) : x = 0.3242 :=
by
  sorry

end certain_number_calculation_l25_25955


namespace prime_square_mod_24_l25_25740

theorem prime_square_mod_24 (p q : ℕ) (k : ℤ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : p > 5) (hq_gt_5 : q > 5) 
  (h_diff : p ≠ q)
  (h_eq : p^2 - q^2 = 6 * k) : (p^2 - q^2) % 24 = 0 := by
sorry

end prime_square_mod_24_l25_25740


namespace geometric_sequence_general_formula_sum_of_sequence_l25_25204

noncomputable def a (n : ℕ) : ℝ := (1 / 2) ^ n

def b (n : ℕ) : ℝ := 3 * n - 2

def c (n : ℕ) : ℝ := (3 * n - 2) * (1 / 2) ^ n

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, (c i)

theorem geometric_sequence_general_formula (a_1 a_2 a_3 a_6 : ℝ) (h1 : a_1 + 2 * a_2 = 1) 
  (h2 : a_3 * a_3 = 4 * a_2 * a_6) :
  (∀ n, a n = (1 / 2) ^ n) :=
sorry

theorem sum_of_sequence (n : ℕ) :
  S(n) = 4 - (3 * n + 4) / 2^n :=
sorry

end geometric_sequence_general_formula_sum_of_sequence_l25_25204


namespace intersection_points_zero_l25_25144

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem intersection_points_zero
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h_gp : geometric_sequence a b c)
  (h_ac_pos : a * c > 0) :
  ∃ x : ℝ, quadratic_function a b c x = 0 → false :=
by
  -- Proof to be completed
  sorry

end intersection_points_zero_l25_25144


namespace percent_decrease_l25_25519

theorem percent_decrease (original_price sale_price : ℝ) 
  (h_original: original_price = 100) 
  (h_sale: sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 :=
by
  sorry

end percent_decrease_l25_25519


namespace find_a_and_b_to_make_f_odd_l25_25344

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l25_25344


namespace tan_sum_simplification_l25_25987
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l25_25987


namespace diagonals_in_regular_nine_sided_polygon_l25_25066

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25066


namespace inequality_squares_l25_25363

theorem inequality_squares (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h : a + b + c = 1) :
    (3 / 16) ≤ ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ∧
    ( (a / (1 + a))^2 + (b / (1 + b))^2 + (c / (1 + c))^2 ) ≤ 1 / 4 :=
by
  sorry

end inequality_squares_l25_25363


namespace sum_of_first_six_terms_of_geometric_series_l25_25259

-- Definitions for the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 6

-- Define the formula for the sum of the first n terms of a geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The equivalent Lean 4 statement
theorem sum_of_first_six_terms_of_geometric_series :
  geometric_series_sum a r n = 4095 / 12288 :=
by
  sorry

end sum_of_first_six_terms_of_geometric_series_l25_25259


namespace cost_of_small_bonsai_l25_25623

variable (cost_small_bonsai cost_big_bonsai : ℝ)

theorem cost_of_small_bonsai : 
  cost_big_bonsai = 20 → 
  3 * cost_small_bonsai + 5 * cost_big_bonsai = 190 → 
  cost_small_bonsai = 30 := 
by
  intros h1 h2 
  sorry

end cost_of_small_bonsai_l25_25623


namespace max_value_of_f_l25_25730

noncomputable def f (x : ℝ) : ℝ := min (2^x) (min (x + 2) (10 - x))

theorem max_value_of_f : ∃ M, (∀ x ≥ 0, f x ≤ M) ∧ (∃ x ≥ 0, f x = M) ∧ M = 6 :=
by
  sorry

end max_value_of_f_l25_25730


namespace paint_room_alone_l25_25517

theorem paint_room_alone (x : ℝ) (hx : (1 / x) + (1 / 4) = 1 / 1.714) : x = 3 :=
by sorry

end paint_room_alone_l25_25517


namespace problem_solution_l25_25577

theorem problem_solution (x y : ℚ) (h1 : |x| + x + y - 2 = 14) (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := 
by
  -- It remains to prove
  sorry

end problem_solution_l25_25577


namespace fifth_term_of_geometric_sequence_l25_25497

theorem fifth_term_of_geometric_sequence
  (a r : ℝ)
  (h1 : a * r^2 = 16)
  (h2 : a * r^6 = 2) : a * r^4 = 8 :=
sorry

end fifth_term_of_geometric_sequence_l25_25497


namespace diagonals_in_nine_sided_polygon_l25_25060

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25060


namespace geom_sequence_a1_value_l25_25378

-- Define the conditions and the statement
theorem geom_sequence_a1_value (a_1 a_6 : ℚ) (a_3 a_4 : ℚ)
  (h1 : a_1 + a_6 = 11)
  (h2 : a_3 * a_4 = 32 / 9) :
  (a_1 = 32 / 3 ∨ a_1 = 1 / 3) :=
by 
-- We will prove the theorem here (skipped with sorry)
sorry

end geom_sequence_a1_value_l25_25378


namespace john_anna_ebook_readers_l25_25971

-- Definitions based on conditions
def anna_bought : ℕ := 50
def john_buy_diff : ℕ := 15
def john_lost : ℕ := 3

-- Main statement
theorem john_anna_ebook_readers :
  let john_bought := anna_bought - john_buy_diff in
  let john_remaining := john_bought - john_lost in
  john_remaining + anna_bought = 82 :=
by
  sorry

end john_anna_ebook_readers_l25_25971


namespace fill_pool_with_B_only_l25_25413

theorem fill_pool_with_B_only
    (time_AB : ℝ)
    (R_AB : time_AB = 30)
    (time_A_B_then_B : ℝ)
    (R_A_B_then_B : (10 / 30 + (time_A_B_then_B - 10) / time_A_B_then_B) = 1)
    (only_B_time : ℝ)
    (R_B : only_B_time = 60) :
    only_B_time = 60 :=
by
    sorry

end fill_pool_with_B_only_l25_25413


namespace balloon_highest_elevation_l25_25161

theorem balloon_highest_elevation
  (time_rise1 time_rise2 time_descent : ℕ)
  (rate_rise rate_descent : ℕ)
  (t1 : time_rise1 = 15)
  (t2 : time_rise2 = 15)
  (t3 : time_descent = 10)
  (rr : rate_rise = 50)
  (rd : rate_descent = 10)
  : (time_rise1 * rate_rise - time_descent * rate_descent + time_rise2 * rate_rise) = 1400 := 
by
  sorry

end balloon_highest_elevation_l25_25161


namespace region_of_inequality_l25_25881

theorem region_of_inequality (x y : ℝ) : (x + y - 6 < 0) → y < -x + 6 := by
  sorry

end region_of_inequality_l25_25881


namespace max_intersections_l25_25507

theorem max_intersections (X Y : Type) [Fintype X] [Fintype Y]
  (hX : Fintype.card X = 20) (hY : Fintype.card Y = 10) : 
  ∃ (m : ℕ), m = 8550 := by
  sorry

end max_intersections_l25_25507


namespace abs_inequality_solution_l25_25192

theorem abs_inequality_solution (x : ℝ) : 
  (|2 * x + 1| > 3) ↔ (x > 1 ∨ x < -2) :=
sorry

end abs_inequality_solution_l25_25192


namespace diagonals_in_nonagon_l25_25129

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25129


namespace minimum_value_of_f_l25_25494

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem minimum_value_of_f : ∃ y, (∀ x, f x ≥ y) ∧ y = 3 := 
by
  sorry

end minimum_value_of_f_l25_25494


namespace diagonals_in_regular_nine_sided_polygon_l25_25010

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25010


namespace find_base_solve_inequality_case1_solve_inequality_case2_l25_25300

noncomputable def log_function (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_base (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : log_function a 8 = 3 → a = 2 :=
by sorry

theorem solve_inequality_case1 (a : ℝ) (h₁ : 1 < a) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 0 < x ∧ x ≤ 1 / 2 :=
by sorry

theorem solve_inequality_case2 (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 1 / 2 ≤ x ∧ x < 2 / 3 :=
by sorry

end find_base_solve_inequality_case1_solve_inequality_case2_l25_25300


namespace water_added_16_l25_25407

theorem water_added_16 (W : ℝ) 
  (h1 : ∃ W, 24 * 0.90 = 0.54 * (24 + W)) : 
  W = 16 := 
by {
  sorry
}

end water_added_16_l25_25407


namespace eight_p_plus_one_composite_l25_25234

theorem eight_p_plus_one_composite 
  (p : ℕ) 
  (hp : Nat.Prime p) 
  (h8p_minus_one : Nat.Prime (8 * p - 1))
  : ¬ (Nat.Prime (8 * p + 1)) :=
sorry

end eight_p_plus_one_composite_l25_25234


namespace tan_sum_simplification_l25_25985
-- We start by importing the relevant Lean libraries that contain trigonometric functions and basic real analysis.

-- Define the statement to be proved in Lean.
theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12) = 4 * Real.sqrt 2 - 4) :=
by
  sorry

end tan_sum_simplification_l25_25985


namespace exists_g_l25_25975

variable {R : Type} [Field R]

-- Define the function f with the given condition
def f (x y : R) : R := sorry

-- The main theorem to prove the existence of g
theorem exists_g (f_condition: ∀ x y z : R, f x y + f y z + f z x = 0) : ∃ g : R → R, ∀ x y : R, f x y = g x - g y := 
by 
  sorry

end exists_g_l25_25975


namespace product_of_roots_abs_eq_l25_25872

theorem product_of_roots_abs_eq (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  x = 5 ∨ x = -5 ∧ ((5 : ℝ) * (-5 : ℝ) = -25) := 
sorry

end product_of_roots_abs_eq_l25_25872


namespace triangle_vertex_y_coordinate_l25_25388

theorem triangle_vertex_y_coordinate (h : ℝ) :
  let A := (0, 0)
  let C := (8, 0)
  let B := (4, h)
  (1/2) * (8) * h = 32 → h = 8 :=
by
  intro h
  intro H
  sorry

end triangle_vertex_y_coordinate_l25_25388


namespace max_log2_x_2log2_y_l25_25441

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem max_log2_x_2log2_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y^2 = 2) :
  log2 x + 2 * log2 y ≤ 0 :=
sorry

end max_log2_x_2log2_y_l25_25441


namespace area_of_quadrilateral_l25_25551

theorem area_of_quadrilateral (d a b : ℝ) (h₀ : d = 28) (h₁ : a = 9) (h₂ : b = 6) :
  (1 / 2 * d * a) + (1 / 2 * d * b) = 210 :=
by
  -- Provided proof steps are skipped
  sorry

end area_of_quadrilateral_l25_25551


namespace polynomial_evaluation_l25_25805

theorem polynomial_evaluation (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) : y^3 - 3 * y^2 - 9 * y + 7 = 7 := 
  sorry

end polynomial_evaluation_l25_25805


namespace haman_dropped_trays_l25_25951

def initial_trays_to_collect : ℕ := 10
def additional_trays : ℕ := 7
def eggs_sold : ℕ := 540
def eggs_per_tray : ℕ := 30

theorem haman_dropped_trays :
  ∃ dropped_trays : ℕ,
  (initial_trays_to_collect + additional_trays - dropped_trays)*eggs_per_tray = eggs_sold → dropped_trays = 8 :=
sorry

end haman_dropped_trays_l25_25951


namespace nine_sided_polygon_diagonals_l25_25106

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25106


namespace total_stamps_l25_25457

def num_foreign_stamps : ℕ := 90
def num_old_stamps : ℕ := 70
def num_both_foreign_old_stamps : ℕ := 20
def num_neither_stamps : ℕ := 60

theorem total_stamps :
  (num_foreign_stamps + num_old_stamps - num_both_foreign_old_stamps + num_neither_stamps) = 220 :=
  by
    sorry

end total_stamps_l25_25457


namespace quadratic_equation_of_list_l25_25768

def is_quadratic (eq : Polynomial ℝ) : Prop :=
  eq.degree = 2

def equations : List (Polynomial ℝ) :=
  [3 * Polynomial.x + Polynomial.C 1,
   Polynomial.x - 2 * Polynomial.x ^ 3 - Polynomial.C 3,
   Polynomial.x ^ 2 - Polynomial.C 5,
   2 * Polynomial.x + Polynomial.C 1 / Polynomial.x - Polynomial.C 3]

theorem quadratic_equation_of_list : 
  ∃ (eq : Polynomial ℝ), 
    eq ∈ equations ∧ is_quadratic eq ∧ 
    ∀ eq' ∈ equations, eq' ≠ eq → ¬ is_quadratic eq' :=
by
  sorry

end quadratic_equation_of_list_l25_25768


namespace work_completion_time_l25_25667

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 6
def work_rate_C : ℚ := 1 / 6

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the total work to be done (1 represents the whole job)
def total_work : ℚ := 1

-- Calculate the number of days to complete the work together
def days_to_complete_work : ℚ := total_work / combined_work_rate

theorem work_completion_time :
  work_rate_A = 1 / 6 ∧
  work_rate_B = 1 / 6 ∧
  work_rate_C = 1 / 6 →
  combined_work_rate = (work_rate_A + work_rate_B + work_rate_C) →
  days_to_complete_work = 2 :=
by
  intros
  sorry

end work_completion_time_l25_25667


namespace equality_of_costs_l25_25911

variable (x : ℝ)
def C1 : ℝ := 50 + 0.35 * (x - 500)
def C2 : ℝ := 75 + 0.45 * (x - 1000)

theorem equality_of_costs : C1 x = C2 x → x = 2500 :=
by
  intro h
  sorry

end equality_of_costs_l25_25911


namespace nine_sided_polygon_diagonals_l25_25120

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25120


namespace simplify_expression_l25_25485

-- Definitions derived from the problem statement
variable (x : ℝ)

-- Theorem statement
theorem simplify_expression : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x :=
sorry

end simplify_expression_l25_25485


namespace minimum_xy_l25_25564

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/y = 1/2) : x * y ≥ 16 :=
sorry

end minimum_xy_l25_25564


namespace smallest_solution_for_quartic_eq_l25_25896

theorem smallest_solution_for_quartic_eq :
  let f (x : ℝ) := x^4 - 40*x^2 + 144
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y :=
sorry

end smallest_solution_for_quartic_eq_l25_25896


namespace num_lines_satisfying_conditions_l25_25818

-- Define the entities line, angle, and perpendicularity in a geometric framework
variable (Point Line : Type)
variable (P : Point)
variable (a b l : Line)

-- Define geometrical predicates
variable (Perpendicular : Line → Line → Prop)
variable (Passes_Through : Line → Point → Prop)
variable (Forms_Angle : Line → Line → ℝ → Prop)

-- Given conditions
axiom perp_ab : Perpendicular a b
axiom passes_through_P : Passes_Through l P
axiom angle_la_30 : Forms_Angle l a (30 : ℝ)
axiom angle_lb_90 : Forms_Angle l b (90 : ℝ)

-- The statement to prove
theorem num_lines_satisfying_conditions : ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
  Passes_Through l1 P ∧ Forms_Angle l1 a (30 : ℝ) ∧ Forms_Angle l1 b (90 : ℝ) ∧
  Passes_Through l2 P ∧ Forms_Angle l2 a (30 : ℝ) ∧ Forms_Angle l2 b (90 : ℝ) ∧
  (∀ l', Passes_Through l' P ∧ Forms_Angle l' a (30 : ℝ) ∧ Forms_Angle l' b (90 : ℝ) → l' = l1 ∨ l' = l2) := sorry

end num_lines_satisfying_conditions_l25_25818


namespace base_7_to_base_10_equiv_l25_25391

theorem base_7_to_base_10_equiv (digits : List ℕ) 
  (h : digits = [5, 4, 3, 2, 1]) : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 13539 := 
by 
  sorry

end base_7_to_base_10_equiv_l25_25391


namespace positive_expression_l25_25559

variable (a b c d : ℝ)

theorem positive_expression (ha : a < b) (hb : b < 0) (hc : 0 < c) (hd : c < d) : d - c - b - a > 0 := 
sorry

end positive_expression_l25_25559


namespace leila_savings_l25_25174

theorem leila_savings (S : ℝ) (h : (1 / 4) * S = 20) : S = 80 :=
by
  sorry

end leila_savings_l25_25174


namespace arithmetic_sequence_ratio_l25_25438

-- Definitions and conditions from the problem
variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
variable (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
variable (h3 : ∀ n, S n / T n = (3 * n - 1) / (n + 3))

-- The theorem that will give us the required answer
theorem arithmetic_sequence_ratio : 
  (a 8) / (b 5 + b 11) = 11 / 9 := by 
  have h4 := h3 15
  sorry

end arithmetic_sequence_ratio_l25_25438


namespace geom_seq_sum_l25_25291

theorem geom_seq_sum (q : ℝ) (a₃ a₄ a₅ : ℝ) : 
  0 < q ∧ 3 * (1 - q^3) / (1 - q) = 21 ∧ a₃ = 3 * q^2 ∧ a₄ = 3 * q^3 ∧ a₅ = 3 * q^4 
  -> a₃ + a₄ + a₅ = 84 := 
by 
  sorry

end geom_seq_sum_l25_25291


namespace three_w_seven_l25_25266

def operation_w (a b : ℤ) : ℤ := b + 5 * a - 3 * a^2

theorem three_w_seven : operation_w 3 7 = -5 :=
by
  sorry

end three_w_seven_l25_25266


namespace solution1_solution2_l25_25791

noncomputable def Problem1 : ℝ :=
  4 + (-2)^3 * 5 - (-0.28) / 4

theorem solution1 : Problem1 = -35.93 := by
  sorry

noncomputable def Problem2 : ℚ :=
  -1^4 - (1/6) * (2 - (-3)^2)

theorem solution2 : Problem2 = 1/6 := by
  sorry

end solution1_solution2_l25_25791


namespace range_of_a_l25_25138

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - 2 * a * x + 2 < 0) → a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
  sorry

end range_of_a_l25_25138


namespace range_of_a_for_false_proposition_l25_25581

theorem range_of_a_for_false_proposition :
  ∀ a : ℝ, (¬ ∃ x : ℝ, a * x ^ 2 + a * x + 1 ≤ 0) ↔ (0 ≤ a ∧ a < 4) :=
by sorry

end range_of_a_for_false_proposition_l25_25581


namespace circle_equation_l25_25565

theorem circle_equation : ∃ (x y : ℝ), (x - 2)^2 + y^2 = 2 :=
by
  sorry

end circle_equation_l25_25565


namespace karen_wrong_questions_l25_25173

theorem karen_wrong_questions (k l n : ℕ) (h1 : k + l = 6 + n) (h2 : k + n = l + 9) : k = 6 := 
by
  sorry

end karen_wrong_questions_l25_25173


namespace largest_n_divisibility_l25_25224

theorem largest_n_divisibility (n : ℕ) (h : n + 12 ∣ n^3 + 144) : n ≤ 132 :=
  sorry

end largest_n_divisibility_l25_25224


namespace change_received_correct_l25_25624

-- Define the conditions
def apples := 5
def cost_per_apple_cents := 80
def paid_dollars := 10

-- Convert the cost per apple to dollars
def cost_per_apple_dollars := (cost_per_apple_cents : ℚ) / 100

-- Calculate the total cost for 5 apples
def total_cost_dollars := apples * cost_per_apple_dollars

-- Calculate the change received
def change_received := paid_dollars - total_cost_dollars

-- Prove that the change received by Margie
theorem change_received_correct : change_received = 6 := by
  sorry

end change_received_correct_l25_25624


namespace John_Anna_total_eBooks_l25_25968

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end John_Anna_total_eBooks_l25_25968


namespace problem_integer_pairs_l25_25689

theorem problem_integer_pairs (a b q r : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : q^2 + r = 1977) :
    (a, b) = (50, 7) ∨ (a, b) = (50, 37) ∨ (a, b) = (7, 50) ∨ (a, b) = (37, 50) :=
sorry

end problem_integer_pairs_l25_25689


namespace find_radius_of_sphere_l25_25762

def radius_of_sphere_equal_to_cylinder_area (r : ℝ) (h : ℝ) (d : ℝ) : Prop :=
  (4 * Real.pi * r^2 = 2 * Real.pi * ((d / 2) * h))

theorem find_radius_of_sphere : ∃ r : ℝ, radius_of_sphere_equal_to_cylinder_area r 6 6 ∧ r = 3 :=
by
  sorry

end find_radius_of_sphere_l25_25762


namespace simplify_tangent_sum_l25_25990

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l25_25990


namespace compare_logarithms_l25_25471

noncomputable def a : ℝ := Real.log 3 / Real.log 4 -- log base 4 of 3
noncomputable def b : ℝ := Real.log 4 / Real.log 3 -- log base 3 of 4
noncomputable def c : ℝ := Real.log 3 / Real.log 5 -- log base 5 of 3

theorem compare_logarithms : b > a ∧ a > c := sorry

end compare_logarithms_l25_25471


namespace nine_sided_polygon_diagonals_count_l25_25027

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25027


namespace line_through_points_l25_25647

theorem line_through_points (m b : ℝ)
  (h_slope : m = (-1 - 3) / (-3 - 1))
  (h_point : 3 = m * 1 + b) :
  m + b = 3 :=
sorry

end line_through_points_l25_25647


namespace distance_between_trees_correct_l25_25524

-- Define the given conditions
def yard_length : ℕ := 300
def tree_count : ℕ := 26
def interval_count : ℕ := tree_count - 1

-- Define the target distance between two consecutive trees
def target_distance : ℕ := 12

-- Prove that the distance between two consecutive trees is correct
theorem distance_between_trees_correct :
  yard_length / interval_count = target_distance := 
by
  sorry

end distance_between_trees_correct_l25_25524


namespace cost_of_each_soda_l25_25720

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l25_25720


namespace abs_eq_of_unique_solution_l25_25188

theorem abs_eq_of_unique_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
    (unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) :
    |a| = |b| :=
sorry

end abs_eq_of_unique_solution_l25_25188


namespace shaded_regions_area_l25_25674

/-- Given a grid of 1x1 squares with 2015 shaded regions where boundaries are either:
    - Horizontal line segments
    - Vertical line segments
    - Segments connecting the midpoints of adjacent sides of 1x1 squares
    - Diagonals of 1x1 squares

    Prove that the total area of these 2015 shaded regions is 47.5.
-/
theorem shaded_regions_area (n : ℕ) (h1 : n = 2015) : 
  ∃ (area : ℝ), area = 47.5 :=
by sorry

end shaded_regions_area_l25_25674


namespace survived_trees_difference_l25_25697

theorem survived_trees_difference {original_trees died_trees survived_trees: ℕ} 
  (h1 : original_trees = 13) 
  (h2 : died_trees = 6)
  (h3 : survived_trees = original_trees - died_trees) :
  survived_trees - died_trees = 1 :=
by
  sorry

end survived_trees_difference_l25_25697


namespace value_of_a_minus_b_l25_25142

theorem value_of_a_minus_b (a b : ℝ) (h : (a - 5)^2 + |b^3 - 27| = 0) : a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l25_25142


namespace birds_find_more_than_half_millet_on_sunday_l25_25626

noncomputable def seed_millet_fraction : ℕ → ℚ
| 0 => 2 * 0.2 -- initial amount on Day 1 (Monday)
| (n+1) => 0.7 * seed_millet_fraction n + 0.4

theorem birds_find_more_than_half_millet_on_sunday :
  let dayMillets : ℕ := 7
  let total_seeds : ℚ := 2
  let half_seeds : ℚ := total_seeds / 2
  (seed_millet_fraction dayMillets > half_seeds) := by
    sorry

end birds_find_more_than_half_millet_on_sunday_l25_25626


namespace flea_never_lands_on_all_points_l25_25889

noncomputable def a_n (n : ℕ) : ℕ := (n * (n + 1) / 2) % 300

theorem flea_never_lands_on_all_points :
  ∃ k : ℕ, k < 300 ∧ ∀ n : ℕ, a_n n ≠ k :=
sorry

end flea_never_lands_on_all_points_l25_25889


namespace find_ab_l25_25318

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l25_25318


namespace nine_sided_polygon_diagonals_l25_25105

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25105


namespace kendra_minivans_l25_25615

theorem kendra_minivans (afternoon: ℕ) (evening: ℕ) (h1: afternoon = 4) (h2: evening = 1) : afternoon + evening = 5 :=
by sorry

end kendra_minivans_l25_25615


namespace minimum_questionnaires_l25_25902

theorem minimum_questionnaires (p : ℝ) (r : ℝ) (n_min : ℕ) (h1 : p = 0.65) (h2 : r = 300) :
  n_min = ⌈r / p⌉ ∧ n_min = 462 := 
by
  sorry

end minimum_questionnaires_l25_25902


namespace arrange_order_l25_25361

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := Real.log 2 / Real.log 3
noncomputable def c : Real := Real.cos 2

theorem arrange_order : c < b ∧ b < a :=
by
  sorry

end arrange_order_l25_25361


namespace monomial_properties_l25_25490

def coefficient (m : String) : ℤ := 
  if m = "-2xy^3" then -2 
  else sorry

def degree (m : String) : ℕ := 
  if m = "-2xy^3" then 4 
  else sorry

theorem monomial_properties : coefficient "-2xy^3" = -2 ∧ degree "-2xy^3" = 4 := 
by 
  exact ⟨rfl, rfl⟩

end monomial_properties_l25_25490


namespace regular_nine_sided_polygon_diagonals_l25_25038

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25038


namespace least_number_of_stamps_l25_25787

theorem least_number_of_stamps : ∃ c f : ℕ, 3 * c + 4 * f = 50 ∧ c + f = 13 :=
by
  sorry

end least_number_of_stamps_l25_25787


namespace range_of_m_l25_25567

open Real

noncomputable def f (x m : ℝ) : ℝ := log x / log 2 + x - m

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x m = 0) → 1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l25_25567


namespace mariel_dogs_count_l25_25183

theorem mariel_dogs_count
  (num_dogs_other: Nat)
  (num_legs_tangled: Nat)
  (num_legs_per_dog: Nat)
  (num_legs_per_human: Nat)
  (num_dog_walkers: Nat)
  (num_dogs_mariel: Nat):
  num_dogs_other = 3 →
  num_legs_tangled = 36 →
  num_legs_per_dog = 4 →
  num_legs_per_human = 2 →
  num_dog_walkers = 2 →
  4*num_dogs_mariel + 4*num_dogs_other + 2*num_dog_walkers = num_legs_tangled →
  num_dogs_mariel = 5 :=
by 
  intros h_other h_tangled h_legs_dog h_legs_human h_walkers h_eq
  sorry

end mariel_dogs_count_l25_25183


namespace binary_to_decimal_l25_25796

theorem binary_to_decimal (x : ℕ) (h : x = 0b110010) : x = 50 := by
  sorry

end binary_to_decimal_l25_25796


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25012

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25012


namespace nine_sided_polygon_diagonals_count_l25_25020

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25020


namespace diagonals_in_nine_sided_polygon_l25_25058

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25058


namespace smallest_c_minus_a_l25_25890

theorem smallest_c_minus_a (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prod : a * b * c = 362880) (h_ineq : a < b ∧ b < c) : 
  c - a = 109 :=
sorry

end smallest_c_minus_a_l25_25890


namespace remainder_of_poly_div_l25_25657

theorem remainder_of_poly_div (n : ℕ) (h : n > 2) : (n^3 + 3) % (n + 1) = 2 :=
by 
  sorry

end remainder_of_poly_div_l25_25657


namespace diagonals_in_nonagon_l25_25133

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25133


namespace smallest_number_of_beads_l25_25230

theorem smallest_number_of_beads (M : ℕ) (h1 : ∃ d : ℕ, M = 5 * d + 2) (h2 : ∃ e : ℕ, M = 7 * e + 2) (h3 : ∃ f : ℕ, M = 9 * f + 2) (h4 : M > 1) : M = 317 := sorry

end smallest_number_of_beads_l25_25230


namespace minimum_value_of_quadratic_expression_l25_25432

def quadratic_expr (x y : ℝ) : ℝ := x^2 - x * y + y^2

def constraint (x y : ℝ) : Prop := x + y = 5

theorem minimum_value_of_quadratic_expression :
  ∃ m, ∀ x y, constraint x y → quadratic_expr x y ≥ m ∧ (∃ x y, constraint x y ∧ quadratic_expr x y = m) :=
sorry

end minimum_value_of_quadratic_expression_l25_25432


namespace monkey_total_distance_l25_25525

theorem monkey_total_distance :
  let speedRunning := 15
  let timeRunning := 5
  let speedSwinging := 10
  let timeSwinging := 10
  let distanceRunning := speedRunning * timeRunning
  let distanceSwinging := speedSwinging * timeSwinging
  let totalDistance := distanceRunning + distanceSwinging
  totalDistance = 175 :=
by
  sorry

end monkey_total_distance_l25_25525


namespace optimal_chalk_length_l25_25607

theorem optimal_chalk_length (l : ℝ) (h₁: 10 ≤ l) (h₂: l ≤ 15) (h₃: l = 12) : l = 12 :=
by
  sorry

end optimal_chalk_length_l25_25607


namespace tickets_spent_on_hat_l25_25917

def tickets_won_whack_a_mole := 32
def tickets_won_skee_ball := 25
def tickets_left := 50
def total_tickets := tickets_won_whack_a_mole + tickets_won_skee_ball

theorem tickets_spent_on_hat : 
  total_tickets - tickets_left = 7 :=
by
  sorry

end tickets_spent_on_hat_l25_25917


namespace expression_value_is_241_l25_25229

noncomputable def expression_value : ℕ :=
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2

theorem expression_value_is_241 : expression_value = 241 := 
by
  sorry

end expression_value_is_241_l25_25229


namespace perpendicular_lines_iff_l25_25404

theorem perpendicular_lines_iff (a : ℝ) : 
  (∀ b₁ b₂ : ℝ, b₁ ≠ b₂ → ¬ (∀ x : ℝ, a * x + b₁ = (a - 2) * x + b₂) ∧ 
   (a * (a - 2) = -1)) ↔ a = 1 :=
by
  sorry

end perpendicular_lines_iff_l25_25404


namespace cyclic_sum_inequality_l25_25936

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) ≥ (2 / 3) * (a^2 + b^2 + c^2) :=
  sorry

end cyclic_sum_inequality_l25_25936


namespace vector_BC_is_correct_l25_25688

-- Given points B(1,2) and C(4,5)
def point_B := (1, 2)
def point_C := (4, 5)

-- Define the vector BC
def vector_BC (B C : ℕ × ℕ) : ℕ × ℕ :=
  (C.1 - B.1, C.2 - B.2)

-- Prove that the vector BC is (3, 3)
theorem vector_BC_is_correct : vector_BC point_B point_C = (3, 3) :=
  sorry

end vector_BC_is_correct_l25_25688


namespace rate_of_interest_l25_25394

variable (P SI T R : ℝ)
variable (hP : P = 400)
variable (hSI : SI = 160)
variable (hT : T = 2)

theorem rate_of_interest :
  (SI = (P * R * T) / 100) → R = 20 :=
by
  intro h
  have h1 : P = 400 := hP
  have h2 : SI = 160 := hSI
  have h3 : T = 2 := hT
  sorry

end rate_of_interest_l25_25394


namespace total_books_in_library_l25_25383

theorem total_books_in_library :
  ∃ (total_books : ℕ),
  (∀ (books_per_floor : ℕ), books_per_floor - 2 = 20 → 
  total_books = (28 * 6 * books_per_floor)) ∧ total_books = 3696 :=
by
  sorry

end total_books_in_library_l25_25383


namespace men_who_wore_glasses_l25_25476

theorem men_who_wore_glasses (total_people : ℕ) (women_ratio men_with_glasses_ratio : ℚ)  
  (h_total : total_people = 1260) 
  (h_women_ratio : women_ratio = 7 / 18)
  (h_men_with_glasses_ratio : men_with_glasses_ratio = 6 / 11)
  : ∃ (men_with_glasses : ℕ), men_with_glasses = 420 := 
by
  sorry

end men_who_wore_glasses_l25_25476


namespace regular_nonagon_diagonals_correct_l25_25110

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25110


namespace triangles_with_two_white_vertices_l25_25437

theorem triangles_with_two_white_vertices (p f z : ℕ) 
    (h1 : p * f + p * z + f * z = 213)
    (h2 : (p * (p - 1) / 2) + (f * (f - 1) / 2) + (z * (z - 1) / 2) = 112)
    (h3 : p * f * z = 540)
    (h4 : (p * (p - 1) / 2) * (f + z) = 612) :
    (f * (f - 1) / 2) * (p + z) = 210 ∨ (f * (f - 1) / 2) * (p + z) = 924 := 
  sorry

end triangles_with_two_white_vertices_l25_25437


namespace find_product_stu_l25_25646

-- Define hypotheses
variables (a x y c : ℕ)
variables (s t u : ℕ)
variable (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2))

-- Statement to prove the equivalent form and stu product
theorem find_product_stu (h_eq : a^8 * x * y - a^7 * x - a^6 * y = a^5 * (c^5 - 2)) :
  ∃ s t u : ℕ, (a^s * x - a^t) * (a^u * y - a^3) = a^5 * c^5 ∧ s * t * u = 12 :=
sorry

end find_product_stu_l25_25646


namespace nine_sided_polygon_diagonals_count_l25_25025

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25025


namespace solve_quadratic_eq_l25_25486

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 * x = 2 ↔ (x = 2 + Real.sqrt 6) ∨ (x = 2 - Real.sqrt 6) :=
by
  sorry

end solve_quadratic_eq_l25_25486


namespace twice_a_plus_one_non_negative_l25_25270

theorem twice_a_plus_one_non_negative (a : ℝ) : 2 * a + 1 ≥ 0 :=
sorry

end twice_a_plus_one_non_negative_l25_25270


namespace calculate_decimal_l25_25788

theorem calculate_decimal : 3.59 + 2.4 - 1.67 = 4.32 := 
  by
  sorry

end calculate_decimal_l25_25788


namespace houses_without_features_l25_25774

-- Definitions for the given conditions
def N : ℕ := 70
def G : ℕ := 50
def P : ℕ := 40
def GP : ℕ := 35

-- The statement of the proof problem
theorem houses_without_features : N - (G + P - GP) = 15 := by
  sorry

end houses_without_features_l25_25774


namespace smallest_total_students_l25_25602

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l25_25602


namespace diagonals_in_nine_sided_polygon_l25_25082

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25082


namespace solution_set_of_inequality_l25_25836

theorem solution_set_of_inequality (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ x2 → f x2 ≤ f x1) →
  (f 1 = 0) →
  {x : ℝ | f (x - 3) ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
by
  intros h_even h_mono h_f1
  sorry

end solution_set_of_inequality_l25_25836


namespace jim_total_weight_per_hour_l25_25357

theorem jim_total_weight_per_hour :
  let hours := 8
  let gold_chest := 100
  let gold_bag := 50
  let gold_extra := 30 + 20 + 10
  let silver := 30
  let bronze := 50
  let weight_gold := 10
  let weight_silver := 5
  let weight_bronze := 2
  let total_gold := gold_chest + 2 * gold_bag + gold_extra
  let total_weight := total_gold * weight_gold + silver * weight_silver + bronze * weight_bronze
  total_weight / hours = 356.25 := by
  sorry

end jim_total_weight_per_hour_l25_25357


namespace determine_a_b_odd_function_l25_25327

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l25_25327


namespace prove_p_l25_25235

variables {m n p : ℝ}

/-- Given points (m, n) and (m + p, n + 4) lie on the line 
   x = y / 2 - 2 / 5, prove p = 2.
-/
theorem prove_p (hmn : m = n / 2 - 2 / 5)
                (hmpn4 : m + p = (n + 4) / 2 - 2 / 5) : p = 2 := 
by
  sorry

end prove_p_l25_25235


namespace darius_drive_miles_l25_25922

theorem darius_drive_miles (total_miles : ℕ) (julia_miles : ℕ) (darius_miles : ℕ) 
  (h1 : total_miles = 1677) (h2 : julia_miles = 998) (h3 : total_miles = darius_miles + julia_miles) : 
  darius_miles = 679 :=
by
  sorry

end darius_drive_miles_l25_25922


namespace union_of_A_and_B_l25_25690

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

-- The theorem we aim to prove
theorem union_of_A_and_B : A ∪ B = { x | -3 ≤ x ∧ x ≤ 4 } :=
sorry

end union_of_A_and_B_l25_25690


namespace part1_part2_l25_25694

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin ((1 / 3) * x - (Real.pi / 6))

theorem part1 : f (5 * Real.pi / 4) = Real.sqrt 2 :=
by sorry

theorem part2 (α β : ℝ) (hαβ : 0 ≤ α ∧ α ≤ Real.pi / 2 ∧ 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h1: f (3 * α + Real.pi / 2) = 10 / 13) (h2: f (3 * β + 2 * Real.pi) = 6 / 5) :
  Real.cos (α + β) = 16 / 65 :=
by sorry

end part1_part2_l25_25694


namespace nine_sided_polygon_diagonals_l25_25109

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25109


namespace coefficient_A_l25_25393

-- Definitions from the conditions
variable (A c₀ d : ℝ)
variable (h₁ : c₀ = 47)
variable (h₂ : A * c₀ + (d - 12) ^ 2 = 235)

-- The theorem to prove
theorem coefficient_A (h₁ : c₀ = 47) (h₂ : A * c₀ + (d - 12) ^ 2 = 235) : A = 5 :=
by sorry

end coefficient_A_l25_25393


namespace diagonals_in_nonagon_l25_25131

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25131


namespace balloon_highest_elevation_l25_25159

theorem balloon_highest_elevation 
  (lift_rate : ℕ)
  (descend_rate : ℕ)
  (pull_time1 : ℕ)
  (release_time : ℕ)
  (pull_time2 : ℕ) :
  lift_rate = 50 →
  descend_rate = 10 →
  pull_time1 = 15 →
  release_time = 10 →
  pull_time2 = 15 →
  (lift_rate * pull_time1 - descend_rate * release_time + lift_rate * pull_time2) = 1400 :=
by
  sorry

end balloon_highest_elevation_l25_25159


namespace find_f7_l25_25734

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x^7 + b * x^3 + c * x - 5

theorem find_f7 (a b c : ℝ) (h : f (-7) a b c = 7) : f 7 a b c = -17 :=
by
  sorry

end find_f7_l25_25734


namespace sum_xyz_l25_25832

variables {x y z : ℝ}

theorem sum_xyz (hx : x * y = 30) (hy : x * z = 60) (hz : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_l25_25832


namespace andrew_paid_total_l25_25518

-- Define the quantities and rates
def quantity_grapes : ℕ := 14
def rate_grapes : ℕ := 54
def quantity_mangoes : ℕ := 10
def rate_mangoes : ℕ := 62

-- Define the cost calculations
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes
def total_cost : ℕ := cost_grapes + cost_mangoes

-- Prove the total amount paid is as expected
theorem andrew_paid_total : total_cost = 1376 := by
  sorry 

end andrew_paid_total_l25_25518


namespace diagonals_in_regular_nine_sided_polygon_l25_25003

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25003


namespace minimum_sum_of_nine_consecutive_integers_l25_25475

-- We will define the consecutive sequence and the conditions as described.
structure ConsecutiveIntegers (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ) : Prop :=
(seq : a1 + 1 = a2 ∧ a2 + 1 = a3 ∧ a3 + 1 = a4 ∧ a4 + 1 = a5 ∧ a5 + 1 = a6 ∧ a6 + 1 = a7 ∧ a7 + 1 = a8 ∧ a8 + 1 = a9)
(sq_cond : ∃ k : ℕ, (a1 + a3 + a5 + a7 + a9) = k * k)
(cube_cond : ∃ l : ℕ, (a2 + a4 + a6 + a8) = l * l * l)

theorem minimum_sum_of_nine_consecutive_integers :
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ,
  ConsecutiveIntegers a1 a2 a3 a4 a5 a6 a7 a8 a9 ∧ (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 = 18000) :=
  sorry

end minimum_sum_of_nine_consecutive_integers_l25_25475


namespace sqrt_one_over_four_eq_pm_half_l25_25226

theorem sqrt_one_over_four_eq_pm_half : Real.sqrt (1 / 4) = 1 / 2 ∨ Real.sqrt (1 / 4) = - (1 / 2) := by
  sorry

end sqrt_one_over_four_eq_pm_half_l25_25226


namespace isosceles_triangle_third_vertex_y_coord_l25_25541

theorem isosceles_triangle_third_vertex_y_coord :
  ∀ (A B : ℝ × ℝ) (θ : ℝ), 
  A = (0, 5) → B = (8, 5) → θ = 60 → 
  ∃ (C : ℝ × ℝ), C.fst > 0 ∧ C.snd > 5 ∧ C.snd = 5 + 4 * Real.sqrt 3 :=
by
  intros A B θ hA hB hθ
  use (4, 5 + 4 * Real.sqrt 3)
  sorry

end isosceles_triangle_third_vertex_y_coord_l25_25541


namespace min_value_proof_l25_25854

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c)

theorem min_value_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ≥ 343 :=
sorry

end min_value_proof_l25_25854


namespace total_lunch_bill_l25_25638

def cost_of_hotdog : ℝ := 5.36
def cost_of_salad : ℝ := 5.10

theorem total_lunch_bill : cost_of_hotdog + cost_of_salad = 10.46 := 
by
  sorry

end total_lunch_bill_l25_25638


namespace combined_age_of_staff_l25_25711

/--
In a school, the average age of a class of 50 students is 25 years. 
The average age increased by 2 years when the ages of 5 additional 
staff members, including the teacher, are also taken into account. 
Prove that the combined age of these 5 staff members is 235 years.
-/
theorem combined_age_of_staff 
    (n_students : ℕ) (avg_age_students : ℕ) (n_staff : ℕ) (avg_age_total : ℕ)
    (h1 : n_students = 50) 
    (h2 : avg_age_students = 25) 
    (h3 : n_staff = 5) 
    (h4 : avg_age_total = 27) :
  n_students * avg_age_students + (n_students + n_staff) * avg_age_total - 
  n_students * avg_age_students = 235 :=
by
  sorry

end combined_age_of_staff_l25_25711


namespace sqrt_product_simplification_l25_25545

theorem sqrt_product_simplification (p : ℝ) : 
  (Real.sqrt (42 * p)) * (Real.sqrt (14 * p)) * (Real.sqrt (7 * p)) = 14 * p * (Real.sqrt (21 * p)) := 
  sorry

end sqrt_product_simplification_l25_25545


namespace john_age_multiple_of_james_age_l25_25613

-- Define variables for the problem conditions
def john_current_age : ℕ := 39
def john_age_3_years_ago : ℕ := john_current_age - 3

def james_brother_age : ℕ := 16
def james_brother_older : ℕ := 4

def james_current_age : ℕ := james_brother_age - james_brother_older
def james_age_in_6_years : ℕ := james_current_age + 6

-- The goal is to prove the multiple relationship
theorem john_age_multiple_of_james_age :
  john_age_3_years_ago = 2 * james_age_in_6_years :=
by {
  -- Skip the proof
  sorry
}

end john_age_multiple_of_james_age_l25_25613


namespace largest_possible_s_l25_25729

theorem largest_possible_s :
  ∃ s r : ℕ, (r ≥ s) ∧ (s ≥ 5) ∧ (122 * r - 120 * s = r * s) ∧ (s = 121) :=
by sorry

end largest_possible_s_l25_25729


namespace sqrt_expression_meaningful_domain_l25_25837

theorem sqrt_expression_meaningful_domain {x : ℝ} (h : 3 - x ≥ 0) : x ≤ 3 := by
  sorry

end sqrt_expression_meaningful_domain_l25_25837


namespace athletes_leave_rate_l25_25909

theorem athletes_leave_rate (R : ℝ) (h : 300 - 4 * R + 105 = 307) : R = 24.5 :=
  sorry

end athletes_leave_rate_l25_25909


namespace no_solution_for_inequalities_l25_25431

theorem no_solution_for_inequalities (x : ℝ) : ¬ ((6 * x - 2 < (x + 2) ^ 2) ∧ ((x + 2) ^ 2 < 9 * x - 5)) :=
by sorry

end no_solution_for_inequalities_l25_25431


namespace required_circle_properties_l25_25932

-- Define the two given circles' equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def line (x y : ℝ) : Prop :=
  x - y - 4 = 0

-- The equation of the required circle
def required_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - x + 7*y - 32 = 0

-- Prove that the required circle satisfies the conditions
theorem required_circle_properties (x y : ℝ) (hx : required_circle x y) :
  (∃ x y, circle1 x y ∧ circle2 x y ∧ required_circle x y) ∧
  (∃ x y, required_circle x y ∧ line x y) :=
by
  sorry

end required_circle_properties_l25_25932


namespace max_homework_time_l25_25625

theorem max_homework_time :
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  biology + history + geography = 180 :=
by
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  show biology + history + geography = 180
  sorry

end max_homework_time_l25_25625


namespace number_of_diagonals_l25_25090

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25090


namespace nine_sided_polygon_diagonals_l25_25122

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25122


namespace unique_digit_for_prime_l25_25883

theorem unique_digit_for_prime (B : ℕ) (hB : B < 10) (hprime : Nat.Prime (30420 * 10 + B)) : B = 1 :=
sorry

end unique_digit_for_prime_l25_25883


namespace cost_of_each_soda_l25_25723

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l25_25723


namespace orlando_weight_gain_l25_25918

def weight_gain_statement (x J F : ℝ) : Prop :=
  J = 2 * x + 2 ∧ F = 1/2 * J - 3 ∧ x + J + F = 20

theorem orlando_weight_gain :
  ∃ x J F : ℝ, weight_gain_statement x J F ∧ x = 5 :=
by {
  sorry
}

end orlando_weight_gain_l25_25918


namespace find_g_at_3_l25_25701

theorem find_g_at_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 4 * x + 1) : g 3 = 23 / 3 :=
by
  sorry

end find_g_at_3_l25_25701


namespace abhay_speed_l25_25903

-- Definitions of the problem's conditions
def condition1 (A S : ℝ) : Prop := 42 / A = 42 / S + 2
def condition2 (A S : ℝ) : Prop := 42 / (2 * A) = 42 / S - 1

-- Define Abhay and Sameer's speeds and declare the main theorem
theorem abhay_speed (A S : ℝ) (h1 : condition1 A S) (h2 : condition2 A S) : A = 10.5 :=
by
  sorry

end abhay_speed_l25_25903


namespace remainder_of_sum_div_17_l25_25280

-- Definitions based on the conditions from the problem
def numbers : List ℕ := [82, 83, 84, 85, 86, 87, 88, 89]
def divisor : ℕ := 17

-- The theorem statement proving the result
theorem remainder_of_sum_div_17 : List.sum numbers % divisor = 0 := by
  sorry

end remainder_of_sum_div_17_l25_25280


namespace nine_sided_polygon_diagonals_count_l25_25022

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25022


namespace arithmetic_sequence_product_l25_25175

theorem arithmetic_sequence_product (b : ℕ → ℤ) (h1 : ∀ n, b (n + 1) = b n + d) 
  (h2 : b 5 * b 6 = 35) : b 4 * b 7 = 27 :=
sorry

end arithmetic_sequence_product_l25_25175


namespace find_a_b_l25_25321

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l25_25321


namespace h_at_3_l25_25467

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (f x) + 1
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_3 : h 3 = 74 + 28 * Real.sqrt 2 :=
by
  sorry

end h_at_3_l25_25467


namespace sharks_in_Cape_May_August_l25_25354

section
variable {D_J C_J D_A C_A : ℕ}

-- Given conditions
theorem sharks_in_Cape_May_August 
  (h1 : C_J = 2 * D_J) 
  (h2 : C_A = 5 + 3 * D_A) 
  (h3 : D_J = 23) 
  (h4 : D_A = D_J) : 
  C_A = 74 := 
by 
  -- Skipped the proof steps 
  sorry
end

end sharks_in_Cape_May_August_l25_25354


namespace sufficient_material_for_box_l25_25966

theorem sufficient_material_for_box :
  ∃ (l w h : ℕ), l * w * h ≥ 1995 ∧ 2 * (l * w + w * h + h * l) ≤ 958 :=
  sorry

end sufficient_material_for_box_l25_25966


namespace unicorn_journey_length_l25_25384

theorem unicorn_journey_length (num_unicorns : ℕ) (flowers_per_step : ℕ) (total_flowers : ℕ) (step_length_meters : ℕ) : (num_unicorns = 6) → (flowers_per_step = 4) → (total_flowers = 72000) → (step_length_meters = 3) → 
(total_flowers / flowers_per_step / num_unicorns * step_length_meters / 1000 = 9) :=
by
  intros h1 h2 h3 h4
  sorry

end unicorn_journey_length_l25_25384


namespace diagonals_in_regular_nine_sided_polygon_l25_25008

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25008


namespace max_mean_weight_BC_l25_25360

theorem max_mean_weight_BC
  (A_n B_n C_n : ℕ)
  (w_A w_B : ℕ)
  (mean_A mean_B mean_AB mean_AC : ℤ)
  (hA : mean_A = 30)
  (hB : mean_B = 55)
  (hAB : mean_AB = 35)
  (hAC : mean_AC = 32)
  (h1 : mean_A * A_n + mean_B * B_n = mean_AB * (A_n + B_n))
  (h2 : mean_A * A_n + mean_AC * C_n = mean_AC * (A_n + C_n)) :
  ∃ n : ℕ, n ≤ 62 ∧ (mean_B * B_n + w_A * C_n) / (B_n + C_n) = n := 
sorry

end max_mean_weight_BC_l25_25360


namespace no_solution_system_of_equations_l25_25641

theorem no_solution_system_of_equations :
  ¬ (∃ (x y : ℝ),
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0) :=
    by
      sorry

end no_solution_system_of_equations_l25_25641


namespace nine_sided_polygon_diagonals_l25_25123

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25123


namespace intersection_of_A_and_B_l25_25449

def A : Set ℝ := { x | x < 3 }
def B : Set ℝ := { x | Real.log (x - 1) / Real.log 3 > 0 }

theorem intersection_of_A_and_B :
  (A ∩ B) = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l25_25449


namespace number_of_arrangements_l25_25435

-- Define the number of boys and girls
def boys : ℕ := 6
def girls : ℕ := 2

-- Define the total number of students to be selected
def total_selected : ℕ := 4

-- State the problem
theorem number_of_arrangements (B G T : ℕ) (H_boys : B = boys) (H_girls : G = girls) (H_total : T = total_selected) :
  ∃ P : ℕ, P = 240 :=
sorry

end number_of_arrangements_l25_25435


namespace find_family_ages_l25_25709

theorem find_family_ages :
  ∃ (a b father_age mother_age : ℕ), 
    (a < 21) ∧
    (b < 21) ∧
    (a^3 + b^2 > 1900) ∧
    (a^3 + b^2 < 1978) ∧
    (father_age = 1978 - (a^3 + b^2)) ∧
    (mother_age = father_age - 8) ∧
    (a = 12) ∧
    (b = 14) ∧
    (father_age = 54) ∧
    (mother_age = 46) := 
by 
  use 12, 14, 54, 46
  sorry

end find_family_ages_l25_25709


namespace turnover_june_l25_25150

variable (TurnoverApril TurnoverMay : ℝ)

theorem turnover_june (h1 : TurnoverApril = 10) (h2 : TurnoverMay = 12) :
  TurnoverMay * (1 + (TurnoverMay - TurnoverApril) / TurnoverApril) = 14.4 := by
  sorry

end turnover_june_l25_25150


namespace milk_purchase_maximum_l25_25255

theorem milk_purchase_maximum :
  let num_1_liter_bottles := 6
  let num_half_liter_bottles := 6
  let value_per_1_liter_bottle := 20
  let value_per_half_liter_bottle := 15
  let price_per_liter := 22
  let total_value := num_1_liter_bottles * value_per_1_liter_bottle + num_half_liter_bottles * value_per_half_liter_bottle
  total_value / price_per_liter = 5 :=
by
  sorry

end milk_purchase_maximum_l25_25255


namespace black_white_difference_l25_25264

theorem black_white_difference (m n : ℕ) (h_dim : m = 7 ∧ n = 9) (h_first_black : m % 2 = 1 ∧ n % 2 = 1) :
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  black_count - white_count = 1 := 
by
  -- We start with known dimensions and conditions
  let ⟨hm, hn⟩ := h_dim
  have : m = 7 := by rw [hm]
  have : n = 9 := by rw [hn]
  
  -- Calculate the number of black and white squares 
  let black_count := (5 * 4 + 4 * 3)
  let white_count := (4 * 4 + 5 * 3)
  
  -- Use given formulas to calculate the difference
  have diff : black_count - white_count = 1 := by
    sorry -- proof to be provided
  
  exact diff

end black_white_difference_l25_25264


namespace ratio_Lisa_Claire_l25_25866

-- Definitions
def Claire_photos : ℕ := 6
def Robert_photos : ℕ := Claire_photos + 12
def Lisa_photos : ℕ := Robert_photos

-- Theorem statement
theorem ratio_Lisa_Claire : (Lisa_photos : ℚ) / (Claire_photos : ℚ) = 3 / 1 :=
by
  sorry

end ratio_Lisa_Claire_l25_25866


namespace Diane_age_l25_25262

variable (C D E : ℝ)

def Carla_age_is_four_times_Diane_age : Prop := C = 4 * D
def Emma_is_eight_years_older_than_Diane : Prop := E = D + 8
def Carla_and_Emma_are_twins : Prop := C = E

theorem Diane_age : Carla_age_is_four_times_Diane_age C D → 
                    Emma_is_eight_years_older_than_Diane D E → 
                    Carla_and_Emma_are_twins C E → 
                    D = 8 / 3 :=
by
  intros hC hE hTwins
  have h1 : C = 4 * D := hC
  have h2 : E = D + 8 := hE
  have h3 : C = E := hTwins
  sorry

end Diane_age_l25_25262


namespace find_ab_integer_l25_25930

theorem find_ab_integer (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a ≠ b) :
    ∃ n : ℤ, (a^b + b^a) = n * (a^a - b^b) ↔ (a = 2 ∧ b = 1) ∨ (a = 1 ∧ b = 2) := 
sorry

end find_ab_integer_l25_25930


namespace probability_abs_x_le_one_l25_25411

noncomputable def geometric_probability (a b c d : ℝ) : ℝ := (b - a) / (d - c)

theorem probability_abs_x_le_one : 
  ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 3 →  
  geometric_probability (-1) 1 (-1) 3 = 1 / 2 := 
by
  sorry

end probability_abs_x_le_one_l25_25411


namespace regular_nine_sided_polygon_diagonals_l25_25093

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25093


namespace cos2x_quadratic_eq_specific_values_l25_25857

variable (a b c x : ℝ)

axiom eqn1 : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0

noncomputable def quadratic_equation_cos2x 
  (a b c : ℝ) : ℝ × ℝ × ℝ := 
  (a^2, 2*a^2 + 2*a*c - b^2, a^2 + 2*a*c - b^2 + 4*c^2)

theorem cos2x_quadratic_eq 
  (a b c x : ℝ) 
  (h: a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0) :
  (a^2) * (Real.cos (2*x))^2 + 
  (2*a^2 + 2*a*c - b^2) * Real.cos (2*x) + 
  (a^2 + 2*a*c - b^2 + 4*c^2) = 0 :=
sorry

theorem specific_values : 
  quadratic_equation_cos2x 4 2 (-1) = (4, 2, -1) :=
by
  unfold quadratic_equation_cos2x
  simp
  sorry

end cos2x_quadratic_eq_specific_values_l25_25857


namespace weight_of_lighter_boxes_l25_25352

theorem weight_of_lighter_boxes :
  ∃ (x : ℝ),
  (∀ (w : ℝ), w = 20 ∨ w = x) ∧
  (20 * 18 = 360) ∧
  (∃ (n : ℕ), n = 15 → 15 * 20 = 300) ∧
  (∃ (m : ℕ), m = 5 → 5 * 12 = 60) ∧
  (360 - 300 = 60) ∧
  (∀ (l : ℝ), l = 60 / 5 → l = x) →
  x = 12 :=
by
  sorry

end weight_of_lighter_boxes_l25_25352


namespace prob_XYZ_wins_l25_25710

-- Define probabilities as given in the conditions
def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_Z : ℚ := 1 / 12

-- Define the probability that one of X, Y, or Z wins, assuming events are mutually exclusive
def P_XYZ_wins : ℚ := P_X + P_Y + P_Z

theorem prob_XYZ_wins : P_XYZ_wins = 11 / 24 := by
  -- sorry is used to skip the proof
  sorry

end prob_XYZ_wins_l25_25710


namespace dice_product_divisible_by_8_probability_l25_25217

open ProbabilityTheory

/-- Representation of a single roll of a 6-sided die -/
def roll := Finset.range 1 7

/-- Definition of the event that the product of 8 dice rolls is divisible by 8 -/
def event_product_divisible_by_8 : Event (roll ^ 8) :=
  { ω | 8 ∣ List.prod ω.toList }

/-- Calculate the probability of the above event -/
theorem dice_product_divisible_by_8_probability:
  (event_product_divisible_by_8).prob = 277 / 288 := sorry

end dice_product_divisible_by_8_probability_l25_25217


namespace definite_integral_ln_squared_l25_25402

noncomputable def integralFun : ℝ → ℝ := λ x => x * (Real.log x) ^ 2

theorem definite_integral_ln_squared (f : ℝ → ℝ) (a b : ℝ):
  (f = integralFun) → 
  (a = 1) → 
  (b = 2) → 
  ∫ x in a..b, f x = 2 * (Real.log 2) ^ 2 - 2 * Real.log 2 + 3 / 4 :=
by
  intros hfa hao hbo
  rw [hfa, hao, hbo]
  sorry

end definite_integral_ln_squared_l25_25402


namespace eccentricity_of_hyperbola_l25_25299

open Real

-- Hyperbola parameters and conditions
variables (a b c e : ℝ)
-- Ensure a > 0, b > 0
axiom h_a_pos : a > 0
axiom h_b_pos : b > 0
-- Hyperbola equation
axiom hyperbola_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
-- Coincidence of right focus and center of circle
axiom circle_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 3 = 0 → (x, y) = (2, 0)
-- Distance from focus to asymptote is 1
axiom distance_focus_to_asymptote : b = 1

-- Prove the eccentricity e of the hyperbola is 2sqrt(3)/3
theorem eccentricity_of_hyperbola : e = 2 * sqrt 3 / 3 := sorry

end eccentricity_of_hyperbola_l25_25299


namespace calculate_expression_l25_25260

theorem calculate_expression :
  (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end calculate_expression_l25_25260


namespace optionA_optionC_l25_25284

noncomputable def f (x : ℝ) : ℝ := Real.log (|x - 2| + 1)

theorem optionA : ∀ x : ℝ, f (x + 2) = f (-x + 2) := 
by sorry

theorem optionC : (∀ x : ℝ, x < 2 → f x > f (x + 0.01)) ∧ (∀ x : ℝ, x > 2 → f x < f (x - 0.01)) := 
by sorry

end optionA_optionC_l25_25284


namespace compute_fraction_eq_2410_l25_25921

theorem compute_fraction_eq_2410 (x : ℕ) (hx : x = 7) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 2410 := 
by
  -- proof steps go here
  sorry

end compute_fraction_eq_2410_l25_25921


namespace lex_coins_total_l25_25622

def value_of_coins (dimes quarters : ℕ) : ℕ :=
  10 * dimes + 25 * quarters

def more_quarters_than_dimes (dimes quarters : ℕ) : Prop :=
  quarters > dimes

theorem lex_coins_total (dimes quarters : ℕ) (h : value_of_coins dimes quarters = 265) (h_more : more_quarters_than_dimes dimes quarters) : dimes + quarters = 13 :=
sorry

end lex_coins_total_l25_25622


namespace canoe_row_probability_l25_25660

theorem canoe_row_probability :
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_can_still_row := (p_left_works * p_right_works) + (p_left_works * p_right_breaks) + (p_left_breaks * p_right_works)
  p_can_still_row = 21 / 25 :=
by
  sorry

end canoe_row_probability_l25_25660


namespace f_2016_value_l25_25941

def f : ℝ → ℝ := sorry

axiom f_prop₁ : ∀ x : ℝ, (x + 6) + f x = 0
axiom f_symmetry : ∀ x : ℝ, f (-x) = -f x ∧ f 0 = 0

theorem f_2016_value : f 2016 = 0 :=
by
  sorry

end f_2016_value_l25_25941


namespace weight_of_8_moles_CCl4_correct_l25_25830

/-- The problem states that carbon tetrachloride (CCl4) is given, and we are to determine the weight of 8 moles of CCl4 based on its molar mass calculations. -/
noncomputable def weight_of_8_moles_CCl4 (molar_mass_C : ℝ) (molar_mass_Cl : ℝ) : ℝ :=
  let molar_mass_CCl4 := molar_mass_C + 4 * molar_mass_Cl
  8 * molar_mass_CCl4

/-- Given the molar masses of Carbon (C) and Chlorine (Cl), prove that the calculated weight of 8 moles of CCl4 matches the expected weight. -/
theorem weight_of_8_moles_CCl4_correct :
  let molar_mass_C := 12.01
  let molar_mass_Cl := 35.45
  weight_of_8_moles_CCl4 molar_mass_C molar_mass_Cl = 1230.48 := by
  sorry

end weight_of_8_moles_CCl4_correct_l25_25830


namespace verify_total_amount_l25_25198

noncomputable def total_withdrawable_amount (a r : ℝ) : ℝ :=
  a / r * ((1 + r) ^ 5 - (1 + r))

theorem verify_total_amount (a r : ℝ) (h_r_nonzero : r ≠ 0) :
  total_withdrawable_amount a r = a / r * ((1 + r)^5 - (1 + r)) :=
by
  sorry

end verify_total_amount_l25_25198


namespace fish_caught_by_dad_l25_25919

def total_fish_both : ℕ := 23
def fish_caught_morning : ℕ := 8
def fish_thrown_back : ℕ := 3
def fish_caught_afternoon : ℕ := 5
def fish_kept_brendan : ℕ := fish_caught_morning - fish_thrown_back + fish_caught_afternoon

theorem fish_caught_by_dad : total_fish_both - fish_kept_brendan = 13 := by
  sorry

end fish_caught_by_dad_l25_25919


namespace max_books_borrowed_l25_25773

noncomputable def max_books_per_student : ℕ := 14

theorem max_books_borrowed (students_borrowed_0 : ℕ)
                           (students_borrowed_1 : ℕ)
                           (students_borrowed_2 : ℕ)
                           (total_students : ℕ)
                           (average_books : ℕ)
                           (remaining_students_borrowed_at_least_3 : ℕ)
                           (total_books : ℕ)
                           (max_books : ℕ) 
  (h1 : students_borrowed_0 = 2)
  (h2 : students_borrowed_1 = 10)
  (h3 : students_borrowed_2 = 5)
  (h4 : total_students = 20)
  (h5 : average_books = 2)
  (h6 : remaining_students_borrowed_at_least_3 = total_students - students_borrowed_0 - students_borrowed_1 - students_borrowed_2)
  (h7 : total_books = total_students * average_books)
  (h8 : total_books = (students_borrowed_1 * 1 + students_borrowed_2 * 2) + remaining_students_borrowed_at_least_3 * 3 + (max_books - 6))
  (h_max : max_books = max_books_per_student) :
  max_books ≤ max_books_per_student := 
sorry

end max_books_borrowed_l25_25773


namespace measure_8_cm_measure_5_cm_1_measure_5_cm_2_l25_25865

theorem measure_8_cm:
  ∃ n : ℕ, n * (11 - 7) = 8 := by
  sorry

theorem measure_5_cm_1:
  ∃ x : ℕ, ∃ y : ℕ, x * ((11 - 7) * 2) - y * 7 = 5 := by
  sorry

theorem measure_5_cm_2:
  3 * 11 - 4 * 7 = 5 := by
  sorry

end measure_8_cm_measure_5_cm_1_measure_5_cm_2_l25_25865


namespace num_values_between_l25_25189

theorem num_values_between (x y : ℕ) (h1 : x + y ≥ 200) (h2 : x + y ≤ 1000) 
  (h3 : (x * (x - 1) + y * (y - 1)) * 2 = (x + y) * (x + y - 1)) : 
  ∃ n : ℕ, n - 1 = 17 := by
  sorry

end num_values_between_l25_25189


namespace alternating_sign_max_pos_l25_25359

theorem alternating_sign_max_pos (x : ℕ → ℝ) 
  (h_nonzero : ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n ≠ 0)
  (h_condition : ∀ k, 1 ≤ k ∧ k ≤ 2022 → x k + (1 / x (k + 1)) < 0)
  (h_periodic : x 2023 = x 1) :
  ∃ m, m = 1011 ∧ ( ∀ n, 1 ≤ n ∧ n ≤ 2022 → x n > 0 → n ≤ m ∧ m ≤ 2022 ) := 
sorry

end alternating_sign_max_pos_l25_25359


namespace floor_equation_solution_l25_25272

/-- Given the problem's conditions and simplifications, prove that the solution x must 
    be in the interval [5/3, 7/3). -/
theorem floor_equation_solution (x : ℝ) :
  (Real.floor (Real.floor (3 * x) - 1 / 2) = Real.floor (x + 3)) →
  x ∈ Set.Ico (5 / 3 : ℝ) (7 / 3 : ℝ) :=
by
  sorry

end floor_equation_solution_l25_25272


namespace angle_B_of_right_triangle_l25_25213

theorem angle_B_of_right_triangle (B C : ℝ) (hA : A = 90) (hC : C = 3 * B) (h_sum : A + B + C = 180) : B = 22.5 :=
sorry

end angle_B_of_right_triangle_l25_25213


namespace entire_show_length_l25_25265

def first_segment (S T : ℕ) : ℕ := 2 * (S + T)
def second_segment (T : ℕ) : ℕ := 2 * T
def third_segment : ℕ := 10

theorem entire_show_length : 
  first_segment (second_segment third_segment) third_segment + 
  second_segment third_segment + 
  third_segment = 90 :=
by
  sorry

end entire_show_length_l25_25265


namespace inverse_B2_l25_25815

def matrix_B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 7; -2, -4]

def matrix_B2_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-5, -7; 2, 2]

theorem inverse_B2 (B : Matrix (Fin 2) (Fin 2) ℝ) (hB_inv : B⁻¹ = matrix_B_inv) :
  (B^2)⁻¹ = matrix_B2_inv :=
sorry

end inverse_B2_l25_25815


namespace part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l25_25695

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1_min_value : ∀ (x : ℝ), x > 0 → f x ≥ -1 / Real.exp 1 := 
by sorry

noncomputable def g (x k : ℝ) : ℝ := f x - k * (x - 1)

theorem part2_max_value_k_lt : ∀ (k : ℝ), k < Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ Real.exp 1 - k * Real.exp 1 + k :=
by sorry

theorem part2_max_value_k_geq : ∀ (k : ℝ), k ≥ Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ 0 :=
by sorry

end part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l25_25695


namespace mariel_dogs_count_l25_25184

theorem mariel_dogs_count (total_legs : ℤ) (num_dog_walkers : ℤ) (legs_per_walker : ℤ) 
  (other_dogs_count : ℤ) (legs_per_dog : ℤ) (mariel_dogs : ℤ) :
  total_legs = 36 →
  num_dog_walkers = 2 →
  legs_per_walker = 2 →
  other_dogs_count = 3 →
  legs_per_dog = 4 →
  mariel_dogs = (total_legs - (num_dog_walkers * legs_per_walker + other_dogs_count * legs_per_dog)) / legs_per_dog →
  mariel_dogs = 5 :=
by
  intros
  sorry

end mariel_dogs_count_l25_25184


namespace intersection_A_B_find_a_b_l25_25558

noncomputable def A : Set ℝ := { x | x^2 - 5 * x + 6 > 0 }
noncomputable def B : Set ℝ := { x | Real.log (x + 1) / Real.log 2 < 2 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 2 } :=
by
  -- Proof will be provided
  sorry

theorem find_a_b :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + a * x - b < 0 ↔ -1 < x ∧ x < 2) ∧ a = -1 ∧ b = 2 :=
by
  -- Proof will be provided
  sorry

end intersection_A_B_find_a_b_l25_25558


namespace tan_alpha_value_cos2_minus_sin2_l25_25940

variable (α : Real) 

axiom is_internal_angle (angle : Real) : angle ∈ Set.Ico 0 Real.pi 

axiom sin_cos_sum (α : Real) : α ∈ Set.Ico 0 Real.pi → Real.sin α + Real.cos α = 1 / 5

theorem tan_alpha_value (h : α ∈ Set.Ico 0 Real.pi) : Real.tan α = -4 / 3 := by 
  sorry

theorem cos2_minus_sin2 (h : Real.tan α = -4 / 3) : 1 / (Real.cos α^2 - Real.sin α^2) = -25 / 7 := by 
  sorry

end tan_alpha_value_cos2_minus_sin2_l25_25940


namespace sum_of_reciprocals_of_roots_l25_25555

open Real

-- Define the polynomial and its properties using Vieta's formulas
theorem sum_of_reciprocals_of_roots :
  ∀ p q : ℝ, 
  (p + q = 16) ∧ (p * q = 9) → 
  (1 / p + 1 / q = 16 / 9) :=
by
  intros p q h
  let ⟨h1, h2⟩ := h
  sorry

end sum_of_reciprocals_of_roots_l25_25555


namespace solve_inequality_l25_25885

-- Declare the necessary conditions as variables in Lean
variables (a c : ℝ)

-- State the Lean theorem
theorem solve_inequality :
  (∀ x : ℝ, (ax^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) →
  a < 0 →
  a = -6 ∧ c = -1 :=
  sorry

end solve_inequality_l25_25885


namespace find_rate_of_current_l25_25886

-- Given speed of the boat in still water (km/hr)
def boat_speed : ℤ := 20

-- Given time of travel downstream (hours)
def time_downstream : ℚ := 24 / 60

-- Given distance travelled downstream (km)
def distance_downstream : ℤ := 10

-- To find: rate of the current (km/hr)
theorem find_rate_of_current (c : ℚ) 
  (h1 : distance_downstream = (boat_speed + c) * time_downstream) : 
  c = 5 := 
by sorry

end find_rate_of_current_l25_25886


namespace number_of_unsold_items_l25_25258

theorem number_of_unsold_items (v k : ℕ) (hv : v ≤ 53) (havg_int : ∃ n : ℕ, k = n * v)
  (hk_eq : k = 130*v - 1595) 
  (hnew_avg : (k + 2505) / (v + 7) = 130) :
  60 - (v + 7) = 24 :=
by
  sorry

end number_of_unsold_items_l25_25258


namespace area_of_square_is_25_l25_25663

-- Define side length of the square
def sideLength : ℝ := 5

-- Define the area of the square
def area_of_square (side : ℝ) : ℝ := side * side

-- Prove the area of the square with side length 5 is 25 square meters
theorem area_of_square_is_25 : area_of_square sideLength = 25 := by
  sorry

end area_of_square_is_25_l25_25663


namespace odd_function_a_b_l25_25330

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l25_25330


namespace region_in_quadrants_l25_25282

theorem region_in_quadrants (x y : ℝ) :
  (y > 3 * x) → (y > 5 - 2 * x) → (x > 0 ∧ y > 0) :=
by
  intros h₁ h₂
  sorry

end region_in_quadrants_l25_25282


namespace sum_of_ages_is_24_l25_25616

def age_problem :=
  ∃ (x y z : ℕ), 2 * x^2 + y^2 + z^2 = 194 ∧ (x + x + y + z = 24)

theorem sum_of_ages_is_24 : age_problem :=
by
  sorry

end sum_of_ages_is_24_l25_25616


namespace find_sum_l25_25898

variables (a b c d : ℕ)

axiom h1 : 6 * a + 2 * b = 3848
axiom h2 : 6 * c + 3 * d = 4410
axiom h3 : a + 3 * b + 2 * d = 3080

theorem find_sum : a + b + c + d = 1986 :=
by
  sorry

end find_sum_l25_25898


namespace regular_nonagon_diagonals_correct_l25_25118

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25118


namespace original_number_is_0_02_l25_25396

theorem original_number_is_0_02 (x : ℝ) (h : 10000 * x = 4 / x) : x = 0.02 :=
by
  sorry

end original_number_is_0_02_l25_25396


namespace nine_sided_polygon_diagonals_l25_25051

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25051


namespace smallest_number_of_students_l25_25598

theorem smallest_number_of_students
  (n : ℕ)
  (students_attended : ℕ)
  (students_both_competitions : ℕ)
  (students_hinting : ℕ)
  (students_cheating : ℕ)
  (attended_fraction : Real := 0.25)
  (both_competitions_fraction : Real := 0.1)
  (hinting_ratio : Real := 1.5)
  (h_attended : students_attended = (attended_fraction * n).to_nat)
  (h_both : students_both_competitions = (both_competitions_fraction * students_attended).to_nat)
  (h_hinting : students_hinting = (hinting_ratio * students_cheating).to_nat)
  (h_total_attended : students_attended = students_hinting + students_cheating - students_both_competitions)
  : n = 200 :=
sorry

end smallest_number_of_students_l25_25598


namespace simplify_tangent_sum_l25_25989

theorem simplify_tangent_sum :
  tan (Real.pi / 12) + tan (5 * Real.pi / 12) = Real.sqrt 6 - Real.sqrt 2 := 
sorry

end simplify_tangent_sum_l25_25989


namespace combined_rate_of_mpg_l25_25191

-- Defining the average rate of Ray's car
def ray_car_mpg : ℚ := 50

-- Defining the average rate of Tom's car
def tom_car_mpg : ℚ := 20

-- Defining the distance driven by Ray
def ray_distance : ℚ := 150

-- Defining the distance driven by Tom
def tom_distance : ℚ := 300

-- Calculating the gasoline used by Ray
def ray_gallons_used : ℚ := ray_distance / ray_car_mpg

-- Calculating the gasoline used by Tom
def tom_gallons_used : ℚ := tom_distance / tom_car_mpg

-- Calculating the total gasoline used
def total_gallons_used : ℚ := ray_gallons_used + tom_gallons_used

-- Calculating the total distance driven
def total_distance_driven : ℚ := ray_distance + tom_distance

-- Calculating the combined miles per gallon
def combined_mpg : ℚ := total_distance_driven / total_gallons_used

-- The proof statement
theorem combined_rate_of_mpg : combined_mpg = 25 :=  
by 
  /- Proof goes here -/
  sorry

end combined_rate_of_mpg_l25_25191


namespace diagonals_in_regular_nine_sided_polygon_l25_25002

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25002


namespace total_sheep_l25_25242

theorem total_sheep (n : ℕ) 
  (h1 : 3 ∣ n)
  (h2 : 5 ∣ n)
  (h3 : 6 ∣ n)
  (h4 : 8 ∣ n)
  (h5 : n * 7 / 40 = 12) : 
  n = 68 :=
by
  sorry

end total_sheep_l25_25242


namespace bulb_works_longer_than_4000_hours_l25_25683

noncomputable def P_X := 0.5
noncomputable def P_Y := 0.3
noncomputable def P_Z := 0.2

noncomputable def P_4000_given_X := 0.59
noncomputable def P_4000_given_Y := 0.65
noncomputable def P_4000_given_Z := 0.70

noncomputable def P_4000 := 
  P_X * P_4000_given_X + P_Y * P_4000_given_Y + P_Z * P_4000_given_Z

theorem bulb_works_longer_than_4000_hours : P_4000 = 0.63 :=
by
  sorry

end bulb_works_longer_than_4000_hours_l25_25683


namespace smallest_number_of_students_l25_25588

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l25_25588


namespace correct_multiplicand_l25_25673

theorem correct_multiplicand (x : ℕ) (h1 : x * 467 = 1925817) : 
  ∃ n : ℕ, n * 467 = 1325813 :=
by
  sorry

end correct_multiplicand_l25_25673


namespace john_anna_ebook_readers_l25_25970

-- Definitions based on conditions
def anna_bought : ℕ := 50
def john_buy_diff : ℕ := 15
def john_lost : ℕ := 3

-- Main statement
theorem john_anna_ebook_readers :
  let john_bought := anna_bought - john_buy_diff in
  let john_remaining := john_bought - john_lost in
  john_remaining + anna_bought = 82 :=
by
  sorry

end john_anna_ebook_readers_l25_25970


namespace min_value_96_l25_25855

noncomputable def min_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) : ℝ :=
x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

theorem min_value_96 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) :
  min_value x y z h_pos h_xyz = 96 :=
sorry

end min_value_96_l25_25855


namespace fraction_calculation_l25_25544

theorem fraction_calculation : 
  ( (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) ) = (864 / 1505) := 
by
  sorry

end fraction_calculation_l25_25544


namespace probability_xiaoming_l25_25887

variable (win_probability : ℚ) 
          (xiaoming_goal : ℕ)
          (xiaojie_goal : ℕ)
          (rounds_needed_xiaoming : ℕ)
          (rounds_needed_xiaojie : ℕ)

def probability_xiaoming_wins_2_consecutive_rounds
   (win_probability : ℚ) 
   (rounds_needed_xiaoming : ℕ) : ℚ :=
  (win_probability ^ 2) + 
  2 * win_probability ^ 3 * (1 - win_probability) + 
  win_probability ^ 4

theorem probability_xiaoming :
    win_probability = (1/2) ∧ 
    rounds_needed_xiaoming = 2 ∧
    rounds_needed_xiaojie = 3 →
    probability_xiaoming_wins_2_consecutive_rounds (1 / 2) 2 = 7 / 16 :=
by
  -- Proof steps placeholder
  sorry

end probability_xiaoming_l25_25887


namespace balloon_count_l25_25910

theorem balloon_count (total_balloons red_balloons blue_balloons black_balloons : ℕ) 
  (h_total : total_balloons = 180)
  (h_red : red_balloons = 3 * blue_balloons)
  (h_black : black_balloons = 2 * blue_balloons) :
  red_balloons = 90 ∧ blue_balloons = 30 ∧ black_balloons = 60 :=
by
  sorry

end balloon_count_l25_25910


namespace baskets_of_peaches_l25_25654

theorem baskets_of_peaches (n : ℕ) :
  (∀ x : ℕ, (n * 2 = 14) → (n = x)) := by
  sorry

end baskets_of_peaches_l25_25654


namespace number_of_diagonals_l25_25091

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25091


namespace nine_sided_polygon_diagonals_l25_25107

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25107


namespace find_n_l25_25379

-- Define the original and new parabola conditions
def original_parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
noncomputable def new_parabola (x n : ℝ) : ℝ := (x - n + 2)^2 - 1

-- Define the conditions for points A and B lying on the new parabola
def point_A (n : ℝ) : Prop := ∃ y₁ : ℝ, new_parabola 2 n = y₁
def point_B (n : ℝ) : Prop := ∃ y₂ : ℝ, new_parabola 4 n = y₂

-- Define the condition that y1 > y2
def points_condition (n : ℝ) : Prop := ∃ y₁ y₂ : ℝ, new_parabola 2 n = y₁ ∧ new_parabola 4 n = y₂ ∧ y₁ > y₂

-- Prove that n = 6 is the necessary value given the conditions
theorem find_n : ∀ n, (0 < n) → point_A n ∧ point_B n ∧ points_condition n → n = 6 :=
  by
    sorry

end find_n_l25_25379


namespace sum_of_coordinates_of_B_l25_25630

theorem sum_of_coordinates_of_B (x : ℝ) (y : ℝ) 
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0,0)) 
  (hB : B = (x, 3))
  (hslope : (3 - 0) / (x - 0) = 4 / 5) :
  x + 3 = 6.75 := 
by
  sorry

end sum_of_coordinates_of_B_l25_25630


namespace loss_record_l25_25455

-- Conditions: a profit of 25 yuan is recorded as +25 yuan.
def profit_record (profit : Int) : Int :=
  profit

-- Statement we need to prove: A loss of 30 yuan is recorded as -30 yuan.
theorem loss_record : profit_record (-30) = -30 :=
by
  sorry

end loss_record_l25_25455


namespace power_mod_l25_25797

theorem power_mod (n : ℕ) : 3^100 % 7 = 4 := by
  sorry

end power_mod_l25_25797


namespace nine_sided_polygon_diagonals_l25_25030

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25030


namespace product_of_areas_eq_square_of_volume_l25_25643

variable (x y z : ℝ)

def area_xy : ℝ := x * y
def area_yz : ℝ := y * z
def area_zx : ℝ := z * x

theorem product_of_areas_eq_square_of_volume :
  (area_xy x y) * (area_yz y z) * (area_zx z x) = (x * y * z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l25_25643


namespace find_two_sets_l25_25920

theorem find_two_sets :
  ∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 : ℕ),
    a1 + a2 + a3 + a4 + a5 = a1 * a2 * a3 * a4 * a5 ∧
    b1 + b2 + b3 + b4 + b5 = b1 * b2 * b3 * b4 * b5 ∧
    (a1, a2, a3, a4, a5) ≠ (b1, b2, b3, b4, b5) := by
  sorry

end find_two_sets_l25_25920


namespace middle_of_7_consecutive_nat_sum_63_l25_25649

theorem middle_of_7_consecutive_nat_sum_63 (x : ℕ) (h : 7 * x = 63) : x = 9 :=
by
  sorry

end middle_of_7_consecutive_nat_sum_63_l25_25649


namespace johns_hats_cost_l25_25167

theorem johns_hats_cost 
  (weeks : ℕ)
  (days_in_week : ℕ)
  (cost_per_hat : ℕ) 
  (h : weeks = 2 ∧ days_in_week = 7 ∧ cost_per_hat = 50) 
  : (weeks * days_in_week * cost_per_hat) = 700 :=
by
  sorry

end johns_hats_cost_l25_25167


namespace cost_per_unit_range_of_type_A_purchases_maximum_profit_l25_25956

-- Definitions of the problem conditions
def cost_type_A : ℕ := 15
def cost_type_B : ℕ := 20

def profit_type_A : ℕ := 3
def profit_type_B : ℕ := 4

def budget_min : ℕ := 2750
def budget_max : ℕ := 2850

def total_units : ℕ := 150
def profit_min : ℕ := 565

-- Main proof statements as Lean theorems
theorem cost_per_unit : 
  ∃ (x y : ℕ), 
    2 * x + 3 * y = 90 ∧ 
    3 * x + y = 65 ∧ 
    x = cost_type_A ∧ 
    y = cost_type_B := 
sorry

theorem range_of_type_A_purchases : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 50 ∧ 
    budget_min ≤ cost_type_A * a + cost_type_B * (total_units - a) ∧ 
    cost_type_A * a + cost_type_B * (total_units - a) ≤ budget_max := 
sorry

theorem maximum_profit : 
  ∃ (a : ℕ), 
    30 ≤ a ∧ 
    a ≤ 35 ∧ 
    profit_min ≤ profit_type_A * a + profit_type_B * (total_units - a) ∧ 
    ¬∃ (b : ℕ), 
      30 ≤ b ∧ 
      b ≤ 35 ∧ 
      b ≠ a ∧ 
      profit_type_A * b + profit_type_B * (total_units - b) > profit_type_A * a + profit_type_B * (total_units - a) :=
sorry

end cost_per_unit_range_of_type_A_purchases_maximum_profit_l25_25956


namespace driers_drying_time_l25_25628

noncomputable def drying_time (r1 r2 r3 : ℝ) : ℝ := 1 / (r1 + r2 + r3)

theorem driers_drying_time (Q : ℝ) (r1 r2 r3 : ℝ)
  (h1 : r1 = Q / 24) 
  (h2 : r2 = Q / 2) 
  (h3 : r3 = Q / 8) : 
  drying_time r1 r2 r3 = 1.5 :=
by
  sorry

end driers_drying_time_l25_25628


namespace charlotte_one_way_journey_time_l25_25263

def charlotte_distance : ℕ := 60
def charlotte_speed : ℕ := 10

theorem charlotte_one_way_journey_time :
  charlotte_distance / charlotte_speed = 6 :=
by
  sorry

end charlotte_one_way_journey_time_l25_25263


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25016

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25016


namespace young_people_sampled_l25_25841

def num_young_people := 800
def num_middle_aged_people := 1600
def num_elderly_people := 1400
def sampled_elderly_people := 70

-- Lean statement to prove the number of young people sampled
theorem young_people_sampled : 
  (sampled_elderly_people:ℝ) / num_elderly_people = (1 / 20:ℝ) ->
  num_young_people * (1 / 20:ℝ) = 40 := by
  sorry

end young_people_sampled_l25_25841


namespace regular_nine_sided_polygon_diagonals_l25_25039

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25039


namespace range_of_quadratic_expression_l25_25924

theorem range_of_quadratic_expression :
  (∃ x : ℝ, y = 2 * x^2 - 4 * x + 12) ↔ (y ≥ 10) :=
by
  sorry

end range_of_quadratic_expression_l25_25924


namespace ordered_pairs_unique_solution_l25_25935

theorem ordered_pairs_unique_solution :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 :=
by
  sorry

end ordered_pairs_unique_solution_l25_25935


namespace total_cost_of_hats_l25_25171

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l25_25171


namespace scientific_notation_650000_l25_25634

theorem scientific_notation_650000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 650000 = a * 10 ^ n ∧ a = 6.5 ∧ n = 5 :=
  sorry

end scientific_notation_650000_l25_25634


namespace expression_value_l25_25708

-- Define the given condition as an assumption
variable (x : ℝ)
variable (h : 2 * x^2 + 3 * x - 1 = 7)

-- Define the target expression and the required result
theorem expression_value :
  4 * x^2 + 6 * x + 9 = 25 :=
by
  sorry

end expression_value_l25_25708


namespace lottery_prob_correct_l25_25963

def possibleMegaBalls : ℕ := 30
def possibleWinnerBalls : ℕ := 49
def drawnWinnerBalls : ℕ := 6

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def winningProbability : ℚ :=
  (1 : ℚ) / possibleMegaBalls * (1 : ℚ) / combination possibleWinnerBalls drawnWinnerBalls

theorem lottery_prob_correct :
  winningProbability = 1 / 419514480 := by
  sorry

end lottery_prob_correct_l25_25963


namespace remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l25_25554

theorem remainder_7_times_10_pow_20_plus_1_pow_20_mod_9 :
  (7 * 10 ^ 20 + 1 ^ 20) % 9 = 8 :=
by
  -- need to note down the known conditions to help guide proof writing.
  -- condition: 1 ^ 20 = 1
  -- condition: 10 % 9 = 1

  sorry

end remainder_7_times_10_pow_20_plus_1_pow_20_mod_9_l25_25554


namespace problem_a_b_squared_l25_25733

theorem problem_a_b_squared {a b : ℝ} (h1 : a + 3 = (b-1)^2) (h2 : b + 3 = (a-1)^2) (h3 : a ≠ b) : a^2 + b^2 = 10 :=
by
  sorry

end problem_a_b_squared_l25_25733


namespace number_below_267_is_301_l25_25416

-- Define the row number function
def rowNumber (n : ℕ) : ℕ :=
  Nat.sqrt n + 1

-- Define the starting number of a row
def rowStart (k : ℕ) : ℕ :=
  (k - 1) * (k - 1) + 1

-- Define the number in the row below given a number and its position in the row
def numberBelow (n : ℕ) : ℕ :=
  let k := rowNumber n
  let startK := rowStart k
  let position := n - startK
  let startNext := rowStart (k + 1)
  startNext + position

-- Prove that the number below 267 is 301
theorem number_below_267_is_301 : numberBelow 267 = 301 :=
by
  -- skip proof details, just the statement is needed
  sorry

end number_below_267_is_301_l25_25416


namespace packages_per_truck_l25_25211

theorem packages_per_truck (total_packages : ℕ) (number_of_trucks : ℕ) (h1 : total_packages = 490) (h2 : number_of_trucks = 7) :
  (total_packages / number_of_trucks) = 70 := by
  sorry

end packages_per_truck_l25_25211


namespace train_length_correct_l25_25537

-- Define the conditions
def train_speed : ℝ := 63
def time_crossing : ℝ := 40
def expected_length : ℝ := 2520

-- The statement to prove
theorem train_length_correct : train_speed * time_crossing = expected_length :=
by
  exact sorry

end train_length_correct_l25_25537


namespace select_students_for_competitions_l25_25502

theorem select_students_for_competitions : 
  let total_students := 9
  let only_chess := 2
  let only_go := 3
  let both := 4
  total_students = only_chess + only_go + both → 
  (only_chess * only_go) + (both * only_go) + (both * only_chess) + Nat.choose both 2 = 32 := by
  intros
  sorry

end select_students_for_competitions_l25_25502


namespace max_cake_pieces_l25_25527

theorem max_cake_pieces (m n : ℕ) (h₁ : m ≥ 4) (h₂ : n ≥ 4)
    (h : (m-4)*(n-4) = m * n) :
    m * n = 72 :=
by
  sorry

end max_cake_pieces_l25_25527


namespace intersection_A_B_union_A_B_subset_C_A_l25_25180

def set_A : Set ℝ := { x | x^2 - x - 2 > 0 }
def set_B : Set ℝ := { x | 3 - abs x ≥ 0 }
def set_C (p : ℝ) : Set ℝ := { x | 4 * x + p < 0 }

theorem intersection_A_B : set_A ∩ set_B = { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) } :=
sorry

theorem union_A_B : set_A ∪ set_B = Set.univ :=
sorry

theorem subset_C_A (p : ℝ) : set_C p ⊆ set_A → p ≥ 4 :=
sorry

end intersection_A_B_union_A_B_subset_C_A_l25_25180


namespace total_teams_l25_25207

theorem total_teams (m n : ℕ) (hmn : m > n) : 
  (m - n) + 1 = m - n + 1 := 
by sorry

end total_teams_l25_25207


namespace diagonals_in_nine_sided_polygon_l25_25075

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25075


namespace find_a_b_for_odd_function_l25_25310

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l25_25310


namespace find_a_b_l25_25322

theorem find_a_b (f : ℝ → ℝ)
  (h : ∀ x, f x = log (abs (a + 1 / (1 - x))) + b)
  (hf_odd : ∀ x, f (-x) = -f x) : 
  a = -1/2 ∧ b = log 2 :=
by 
  sorry

end find_a_b_l25_25322


namespace equation_of_l_l25_25145

-- Defining the equations of the circles
def circle_O (x y : ℝ) := x^2 + y^2 = 4
def circle_C (x y : ℝ) := x^2 + y^2 + 4 * x - 4 * y + 4 = 0

-- Assuming the line l makes circles O and C symmetric
def symmetric (l : ℝ → ℝ → Prop) := ∀ (x y : ℝ), l x y → 
  (∃ (x' y' : ℝ), circle_O x y ∧ circle_C x' y' ∧ (x + x') / 2 = x' ∧ (y + y') / 2 = y')

-- Stating the theorem to be proven
theorem equation_of_l :
  ∀ l : ℝ → ℝ → Prop, symmetric l → (∀ x y : ℝ, l x y ↔ x - y + 2 = 0) :=
by
  sorry

end equation_of_l_l25_25145


namespace smallest_total_students_l25_25601

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end smallest_total_students_l25_25601


namespace scientific_notation_of_604800_l25_25539

theorem scientific_notation_of_604800 : 604800 = 6.048 * 10^5 := 
sorry

end scientific_notation_of_604800_l25_25539


namespace color_crafter_secret_codes_l25_25842

theorem color_crafter_secret_codes :
  8^5 = 32768 := by
  sorry

end color_crafter_secret_codes_l25_25842


namespace libraryRoomNumber_l25_25366

-- Define the conditions
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def isPrime (n : ℕ) : Prop := Nat.Prime n
def isEven (n : ℕ) : Prop := n % 2 = 0
def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0
def hasDigit7 (n : ℕ) : Prop := n / 10 = 7 ∨ n % 10 = 7

-- Main theorem
theorem libraryRoomNumber (n : ℕ) (h1 : isTwoDigit n)
  (h2 : (isPrime n ∧ isEven n ∧ isDivisibleBy5 n ∧ hasDigit7 n) ↔ false)
  : n % 10 = 0 := 
sorry

end libraryRoomNumber_l25_25366


namespace kittens_percentage_rounded_l25_25369

theorem kittens_percentage_rounded (total_cats female_ratio kittens_per_female cats_sold : ℕ) (h1 : total_cats = 6)
  (h2 : female_ratio = 2)
  (h3 : kittens_per_female = 7)
  (h4 : cats_sold = 9) : 
  ((12 : ℤ) * 100 / (18 : ℤ)).toNat = 67 := by
  -- Historical reference and problem specific values involved 
  sorry

end kittens_percentage_rounded_l25_25369


namespace total_games_is_272_l25_25776

-- Define the number of players
def n : ℕ := 17

-- Define the formula for the number of games played
def total_games (n : ℕ) : ℕ := n * (n - 1)

-- Define a theorem stating that the total games played is 272
theorem total_games_is_272 : total_games n = 272 := by
  -- Proof omitted
  sorry

end total_games_is_272_l25_25776


namespace regular_nine_sided_polygon_diagonals_l25_25094

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25094


namespace isosceles_triangle_largest_angle_l25_25603

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_isosceles : A = B) (h_angles : A = 60 ∧ B = 60) :
  max A (max B C) = 60 :=
by
  sorry

end isosceles_triangle_largest_angle_l25_25603


namespace diagonals_in_nine_sided_polygon_l25_25063

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25063


namespace remainder_of_product_modulo_12_l25_25662

theorem remainder_of_product_modulo_12 : (1625 * 1627 * 1629) % 12 = 3 := by
  sorry

end remainder_of_product_modulo_12_l25_25662


namespace distance_after_3rd_turn_l25_25406

theorem distance_after_3rd_turn (d1 d2 d4 total_distance : ℕ) 
  (h1 : d1 = 5) 
  (h2 : d2 = 8) 
  (h4 : d4 = 0) 
  (h_total : total_distance = 23) : 
  total_distance - (d1 + d2 + d4) = 10 := 
  sorry

end distance_after_3rd_turn_l25_25406


namespace midpoint_trajectory_l25_25566

   -- Defining the given conditions
   def P_moves_on_circle (x1 y1 : ℝ) : Prop :=
     (x1 + 1)^2 + y1^2 = 4

   def Q_coordinates : (ℝ × ℝ) := (4, 3)

   -- Defining the midpoint relationship
   def midpoint_relation (x y x1 y1 : ℝ) : Prop :=
     x1 + Q_coordinates.1 = 2 * x ∧ y1 + Q_coordinates.2 = 2 * y

   -- Proving the trajectory equation of the midpoint M
   theorem midpoint_trajectory (x y : ℝ) : 
     (∃ x1 y1 : ℝ, midpoint_relation x y x1 y1 ∧ P_moves_on_circle x1 y1) →
     (x - 3/2)^2 + (y - 3/2)^2 = 1 :=
   by
     intros h
     sorry
   
end midpoint_trajectory_l25_25566


namespace product_of_p_r_s_l25_25139

theorem product_of_p_r_s
  (p r s : ℕ)
  (h1 : 3^p + 3^4 = 90)
  (h2 : 2^r + 44 = 76)
  (h3 : 5^3 + 6^s = 1421) :
  p * r * s = 40 := 
sorry

end product_of_p_r_s_l25_25139


namespace sin2theta_plus_cos2theta_l25_25831

theorem sin2theta_plus_cos2theta (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end sin2theta_plus_cos2theta_l25_25831


namespace polygon_sides_l25_25671

theorem polygon_sides (n : ℕ) (h : 144 * n = 180 * (n - 2)) : n = 10 :=
by { sorry }

end polygon_sides_l25_25671


namespace consecutive_odd_sum_l25_25381

theorem consecutive_odd_sum (n : ℤ) (h : n + 2 = 9) : 
  let a := n
  let b := n + 2
  let c := n + 4
  (a + b + c) = a + 20 := by
  sorry

end consecutive_odd_sum_l25_25381


namespace ratio_is_l25_25562

noncomputable def volume_dodecahedron (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) / 4 * s ^ 3

noncomputable def volume_tetrahedron (s : ℝ) : ℝ := Real.sqrt 2 / 12 * ((Real.sqrt 3 / 2) * s) ^ 3

noncomputable def ratio_volumes (s : ℝ) : ℝ := volume_dodecahedron s / volume_tetrahedron s

theorem ratio_is (s : ℝ) : ratio_volumes s = (60 + 28 * Real.sqrt 5) / Real.sqrt 6 :=
by
  sorry

end ratio_is_l25_25562


namespace smallest_n_l25_25424

theorem smallest_n (n : ℕ) : 
  (n > 0 ∧ ((n^2 + n + 1)^2 > 1999) ∧ ∀ m : ℕ, (m > 0 ∧ (m^2 + m + 1)^2 > 1999) → m ≥ n) → n = 7 :=
sorry

end smallest_n_l25_25424


namespace K_time_9_hours_l25_25848

theorem K_time_9_hours
  (x : ℝ) -- x is the speed of K
  (hx : 45 / x = 9) -- K's time for 45 miles is 9 hours
  (y : ℝ) -- y is the speed of M
  (h₁ : x = y + 0.5) -- K travels 0.5 mph faster than M
  (h₂ : 45 / y - 45 / x = 3 / 4) -- K takes 3/4 hour less than M
  : 45 / x = 9 :=
by
  sorry

end K_time_9_hours_l25_25848


namespace diagonals_in_nine_sided_polygon_l25_25000

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l25_25000


namespace muffins_per_person_l25_25465

-- Definitions based on conditions
def total_friends : ℕ := 4
def total_people : ℕ := 1 + total_friends
def total_muffins : ℕ := 20

-- Theorem statement for the proof
theorem muffins_per_person : total_muffins / total_people = 4 := by
  sorry

end muffins_per_person_l25_25465


namespace f_of_g_of_3_l25_25619

def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := (x + 2)^2
theorem f_of_g_of_3 : f (g 3) = 95 := by
  sorry

end f_of_g_of_3_l25_25619


namespace train_length_is_150_l25_25250

-- Let length_of_train be the length of the train in meters
def length_of_train (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_s

theorem train_length_is_150 (speed_kmh time_s : ℕ) (h_speed : speed_kmh = 180) (h_time : time_s = 3) :
  length_of_train speed_kmh time_s = 150 := by
  sorry

end train_length_is_150_l25_25250


namespace brownie_count_l25_25254

noncomputable def initial_brownies : ℕ := 20
noncomputable def to_school_administrator (n : ℕ) : ℕ := n / 2
noncomputable def remaining_after_administrator (n : ℕ) : ℕ := n - to_school_administrator n
noncomputable def to_best_friend (n : ℕ) : ℕ := remaining_after_administrator n / 2
noncomputable def remaining_after_best_friend (n : ℕ) : ℕ := remaining_after_administrator n - to_best_friend n
noncomputable def to_friend_simon : ℕ := 2
noncomputable def final_brownies : ℕ := remaining_after_best_friend initial_brownies - to_friend_simon

theorem brownie_count : final_brownies = 3 := by
  sorry

end brownie_count_l25_25254


namespace maximum_area_right_triangle_hypotenuse_8_l25_25205

theorem maximum_area_right_triangle_hypotenuse_8 :
  ∃ a b : ℝ, (a^2 + b^2 = 64) ∧ (a * b) / 2 = 16 :=
by
  sorry

end maximum_area_right_triangle_hypotenuse_8_l25_25205


namespace union_of_A_and_B_l25_25907

namespace SetUnionProof

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | x ≤ 2 }
def C : Set ℝ := { x | x ≤ 2 }

theorem union_of_A_and_B : A ∪ B = C := by
  -- proof goes here
  sorry

end SetUnionProof

end union_of_A_and_B_l25_25907


namespace number_of_cheeses_per_pack_l25_25847

-- Definitions based on the conditions
def packs : ℕ := 3
def cost_per_cheese : ℝ := 0.10
def total_amount_paid : ℝ := 6

-- Theorem statement to prove the number of string cheeses in each pack
theorem number_of_cheeses_per_pack : 
  (total_amount_paid / (packs : ℝ)) / cost_per_cheese = 20 :=
sorry

end number_of_cheeses_per_pack_l25_25847


namespace regular_2020_gon_isosceles_probability_l25_25656

theorem regular_2020_gon_isosceles_probability :
  let n := 2020
  let totalTriangles := (n * (n - 1) * (n - 2)) / 6
  let isoscelesTriangles := n * ((n - 2) / 2)
  let probability := isoscelesTriangles * 6 / totalTriangles
  let (a, b) := (1, 673)
  100 * a + b = 773 := by
    sorry

end regular_2020_gon_isosceles_probability_l25_25656


namespace regular_nine_sided_polygon_diagonals_l25_25046

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25046


namespace min_value_fraction_l25_25436

theorem min_value_fraction (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : a + 2 * b = 2) :
  (a + b) / (a * b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_fraction_l25_25436


namespace value_of_abs_div_sum_l25_25816

theorem value_of_abs_div_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (|a| / a + |b| / b = 2) ∨ (|a| / a + |b| / b = -2) ∨ (|a| / a + |b| / b = 0) := 
by
  sorry

end value_of_abs_div_sum_l25_25816


namespace find_a_and_b_l25_25314

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l25_25314


namespace find_n_l25_25929

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 10) : n = 2100 :=
by
sorry

end find_n_l25_25929


namespace evaluate_expression_l25_25681

def numerator : ℤ :=
  (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1)

def denominator : ℤ :=
  (2 - 3) + (4 - 5) + (6 - 7) + (8 - 9) + (10 - 11) + 12

theorem evaluate_expression : numerator / denominator = 6 / 7 := by
  sorry

end evaluate_expression_l25_25681


namespace find_ab_l25_25316

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l25_25316


namespace diagonals_in_nine_sided_polygon_l25_25076

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25076


namespace floor_problem_solution_l25_25273

noncomputable def floor_problem (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋

theorem floor_problem_solution :
  { x : ℝ | floor_problem x } = { x : ℝ | 2 ≤ x ∧ x < 7 / 3 } :=
by sorry

end floor_problem_solution_l25_25273


namespace time_to_fill_pool_l25_25487

theorem time_to_fill_pool (V : ℕ) (n : ℕ) (r : ℕ) (fill_rate_per_hour : ℕ) :
  V = 24000 → 
  n = 4 →
  r = 25 → -- 2.5 gallons per minute expressed as 25/10 gallons
  fill_rate_per_hour = (n * r * 6) → -- since 6 * 10 = 60 (to convert per minute rate to per hour, we divide so r is 25 instead of 2.5)
  V / fill_rate_per_hour = 40 :=
by
  sorry

end time_to_fill_pool_l25_25487


namespace project_completion_time_l25_25233

-- Definitions for conditions
def a_rate : ℚ := 1 / 20
def b_rate : ℚ := 1 / 30
def combined_rate : ℚ := a_rate + b_rate

-- Total days to complete the project
def total_days (x : ℚ) : Prop :=
  (x - 5) * a_rate + x * b_rate = 1

-- The theorem to be proven
theorem project_completion_time : ∃ (x : ℚ), total_days x ∧ x = 15 := by
  sorry

end project_completion_time_l25_25233


namespace probability_change_needed_l25_25385

noncomputable def toy_prices : List ℝ := List.range' 1 11 |>.map (λ n => n * 0.25)

def favorite_toy_price : ℝ := 2.25

def total_quarters : ℕ := 12

def total_toy_count : ℕ := 10

def total_orders : ℕ := Nat.factorial total_toy_count

def ways_to_buy_without_change : ℕ :=
  (Nat.factorial (total_toy_count - 1)) + 2 * (Nat.factorial (total_toy_count - 2))

def probability_without_change : ℚ :=
  ↑ways_to_buy_without_change / ↑total_orders

def probability_with_change : ℚ :=
  1 - probability_without_change

theorem probability_change_needed : probability_with_change = 79 / 90 :=
  sorry

end probability_change_needed_l25_25385


namespace angle_B_l25_25844

theorem angle_B (a b c A B : ℝ) (h : a * Real.cos B - b * Real.cos A = c) (C : ℝ) (hC : C = Real.pi / 5) (h_triangle : A + B + C = Real.pi) : B = 3 * Real.pi / 10 :=
sorry

end angle_B_l25_25844


namespace smallest_possible_number_of_students_l25_25595

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l25_25595


namespace right_triangle_median_to_hypotenuse_l25_25153

theorem right_triangle_median_to_hypotenuse 
    {DEF : Type} [MetricSpace DEF] 
    (D E F M : DEF) 
    (h_triangle : dist D E = 15 ∧ dist D F = 20 ∧ dist E F = 25) 
    (h_midpoint : dist D M = dist E M ∧ dist D E = 2 * dist D M ∧ dist E F * dist E F = dist E D * dist E D + dist D F * dist D F) :
    dist F M = 12.5 :=
by sorry

end right_triangle_median_to_hypotenuse_l25_25153


namespace new_price_of_computer_l25_25148

theorem new_price_of_computer (d : ℝ) (h : 2 * d = 520) : d * 1.3 = 338 := 
sorry

end new_price_of_computer_l25_25148


namespace find_constants_for_odd_function_l25_25340

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def f (a b : ℝ) (x : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

theorem find_constants_for_odd_function :
  ∃ a b : ℝ, a = -1/2 ∧ b = Real.log 2 ∧ is_odd_function (f a b) :=
by
  sorry

end find_constants_for_odd_function_l25_25340


namespace odd_function_values_l25_25334

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l25_25334


namespace jason_current_cards_l25_25610

-- Define Jason's initial number of Pokemon cards
def jason_initial_cards : ℕ := 1342

-- Define the number of Pokemon cards Alyssa bought
def alyssa_bought_cards : ℕ := 536

-- Define the number of Pokemon cards Jason has now
def jason_final_cards (initial_cards bought_cards : ℕ) : ℕ :=
  initial_cards - bought_cards

-- Theorem statement verifying the final number of Pokemon cards Jason has
theorem jason_current_cards : jason_final_cards jason_initial_cards alyssa_bought_cards = 806 :=
by
  -- Proof goes here
  sorry

end jason_current_cards_l25_25610


namespace root_product_identity_l25_25977

theorem root_product_identity (a b c : ℝ) (h1 : a * b * c = -8) (h2 : a * b + b * c + c * a = 20) (h3 : a + b + c = 15) :
    (1 + a) * (1 + b) * (1 + c) = 28 :=
by
  sorry

end root_product_identity_l25_25977


namespace heather_heavier_than_emily_l25_25571

theorem heather_heavier_than_emily :
  let Heather_weight := 87
  let Emily_weight := 9
  Heather_weight - Emily_weight = 78 :=
by
  -- Proof here
  sorry

end heather_heavier_than_emily_l25_25571


namespace chlorine_needed_l25_25934

variable (Methane moles_HCl moles_Cl₂ : ℕ)

-- Given conditions
def reaction_started_with_one_mole_of_methane : Prop :=
  Methane = 1

def reaction_produces_two_moles_of_HCl : Prop :=
  moles_HCl = 2

-- Question to be proved
def number_of_moles_of_Chlorine_combined : Prop :=
  moles_Cl₂ = 2

theorem chlorine_needed
  (h1 : reaction_started_with_one_mole_of_methane Methane)
  (h2 : reaction_produces_two_moles_of_HCl moles_HCl)
  : number_of_moles_of_Chlorine_combined moles_Cl₂ :=
sorry

end chlorine_needed_l25_25934


namespace count_multiples_of_4_between_300_and_700_l25_25829

noncomputable def num_multiples_of_4_in_range (a b : ℕ) : ℕ :=
  (b - (b % 4) - (a - (a % 4) + 4)) / 4 + 1

theorem count_multiples_of_4_between_300_and_700 : 
  num_multiples_of_4_in_range 301 699 = 99 := by
  sorry

end count_multiples_of_4_between_300_and_700_l25_25829


namespace z_is_greater_by_50_percent_of_w_l25_25350

variable (w q y z : ℝ)

def w_is_60_percent_q : Prop := w = 0.60 * q
def q_is_60_percent_y : Prop := q = 0.60 * y
def z_is_54_percent_y : Prop := z = 0.54 * y

theorem z_is_greater_by_50_percent_of_w (h1 : w_is_60_percent_q w q) 
                                        (h2 : q_is_60_percent_y q y) 
                                        (h3 : z_is_54_percent_y z y) : 
  ((z - w) / w) * 100 = 50 :=
sorry

end z_is_greater_by_50_percent_of_w_l25_25350


namespace number_of_triangles_l25_25799

-- Define a structure representing a triangle with integer angles.
structure Triangle :=
  (A B C : ℕ) -- angles in integer degrees
  (angle_sum : A + B + C = 180)
  (obtuse_A : A > 90)

-- Define a structure representing point D on side BC of triangle ABC such that triangle ABD is right-angled
-- and triangle ADC is isosceles.
structure PointOnBC (ABC : Triangle) :=
  (D : ℕ) -- angle at D in triangle ABC
  (right_ABD : ABC.A = 90 ∨ ABC.B = 90 ∨ ABC.C = 90)
  (isosceles_ADC : ABC.A = ABC.B ∨ ABC.A = ABC.C ∨ ABC.B = ABC.C)

-- Problem Statement:
theorem number_of_triangles (t : Triangle) (d : PointOnBC t): ∃ n : ℕ, n = 88 :=
by
  sorry

end number_of_triangles_l25_25799


namespace problem_statement_l25_25732
  
noncomputable def isosceles_triangle_area_prob : ℝ :=
  let DEF := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1} in
  let region := {(x, y) : ℝ × ℝ | x ∈ Icc 0 1 ∧ y ∈ Icc 0 1 ∧ y > x ∧ y > (x + y) / 2} in
  (set.finite.to_finset DEF).card⁻¹ * (set.finite.to_finset region).card

theorem problem_statement :
  isosceles_triangle_area_prob = 1 / 4 :=
sorry

end problem_statement_l25_25732


namespace sum_of_three_largest_ge_50_l25_25290

theorem sum_of_three_largest_ge_50 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) :
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
  a₆ ≠ a₇ ∧
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0 ∧ a₅ > 0 ∧ a₆ > 0 ∧ a₇ > 0 ∧
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 →
  ∃ (x y z : ℕ), (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧ (x > 0 ∧ y > 0 ∧ z > 0) ∧ (x + y + z ≥ 50) :=
by sorry

end sum_of_three_largest_ge_50_l25_25290


namespace solution_value_a_l25_25833

theorem solution_value_a (x a : ℝ) (h₁ : x = 2) (h₂ : 2 * x + a = 3) : a = -1 :=
by
  -- Proof goes here
  sorry

end solution_value_a_l25_25833


namespace max_sum_x_y_l25_25223

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end max_sum_x_y_l25_25223


namespace right_triangle_integer_segments_count_l25_25481

theorem right_triangle_integer_segments_count :
  ∀ (DE EF : ℕ), DE = 15 → EF = 36 → 
  let DF := Real.sqrt (DE^2 + EF^2) in
  let area := (DE * EF) / 2 in
  ∃ (integer_segment_count : ℕ),
  (integer_segment_count = 24) := 
by
  intros DE EF hDE hEF
  let DF := Real.sqrt (DE^2 + EF^2)
  let area := (DE * EF) / 2
  use 24
  sorry

end right_triangle_integer_segments_count_l25_25481


namespace arranging_six_letters_example_l25_25906

open Multiset 

theorem arranging_six_letters_example (A B C D E F  : ℕ) :
  ∃ (arrangements : ℕ), 
    (A, B, C, D, E, F) = (1, 2, 3, 4, 5, 6) →
    (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) →
    (A < C ∧ B < C ∨ A > C ∧ B > C) →
    arrangements = 480 := 
by 
  sorry

end arranging_six_letters_example_l25_25906


namespace lines_intersect_at_l25_25548

noncomputable def line1 (x : ℚ) : ℚ := (-2 / 3) * x + 2
noncomputable def line2 (x : ℚ) : ℚ := -2 * x + (3 / 2)

theorem lines_intersect_at :
  ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = (3 / 8) ∧ y = (7 / 4) :=
sorry

end lines_intersect_at_l25_25548


namespace Jace_post_break_time_correct_l25_25609

noncomputable def Jace_post_break_time (total_distance : ℝ) (speed : ℝ) (pre_break_time : ℝ) : ℝ :=
  (total_distance - (speed * pre_break_time)) / speed

theorem Jace_post_break_time_correct :
  Jace_post_break_time 780 60 4 = 9 :=
by
  sorry

end Jace_post_break_time_correct_l25_25609


namespace ellipses_same_eccentricity_l25_25943

theorem ellipses_same_eccentricity 
  (a b : ℝ) (k : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : k > 0)
  (e1_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / (a^2)) + (y^2 / (b^2)) = 1)
  (e2_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = k ↔ (x^2 / (ka^2)) + (y^2 / (kb^2)) = 1) :
  1 - (b^2 / a^2) = 1 - (b^2 / (ka^2)) :=
by
  sorry

end ellipses_same_eccentricity_l25_25943


namespace regular_nonagon_diagonals_correct_l25_25112

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25112


namespace range_of_2x_plus_y_l25_25810

theorem range_of_2x_plus_y {x y: ℝ} (h: x^2 / 4 + y^2 = 1) : -Real.sqrt 17 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 17 :=
sorry

end range_of_2x_plus_y_l25_25810


namespace lengths_of_angle_bisectors_areas_of_triangles_l25_25390

-- Given conditions
variables (x y : ℝ) (S1 S2 : ℝ)
variables (hx1 : x + y = 15) (hx2 : x / y = 3 / 2)
variables (hS1 : S1 / S2 = 9 / 4) (hS2 : S1 - S2 = 6)

-- Prove the lengths of the angle bisectors
theorem lengths_of_angle_bisectors :
  x = 9 ∧ y = 6 :=
by sorry

-- Prove the areas of the triangles
theorem areas_of_triangles :
  S1 = 54 / 5 ∧ S2 = 24 / 5 :=
by sorry

end lengths_of_angle_bisectors_areas_of_triangles_l25_25390


namespace quadrilateral_area_l25_25214

noncomputable def AreaOfQuadrilateral (AB AC AD : ℝ) : ℝ :=
  let BC := Real.sqrt (AC^2 - AB^2)
  let CD := Real.sqrt (AC^2 - AD^2)
  let AreaABC := (1 / 2) * AB * BC
  let AreaACD := (1 / 2) * AD * CD
  AreaABC + AreaACD

theorem quadrilateral_area :
  AreaOfQuadrilateral 5 13 12 = 60 :=
by
  sorry

end quadrilateral_area_l25_25214


namespace ratio_doctors_lawyers_l25_25960

theorem ratio_doctors_lawyers (d l : ℕ) (h1 : (45 * d + 60 * l) / (d + l) = 50) (h2 : d + l = 50) : d = 2 * l :=
by
  sorry

end ratio_doctors_lawyers_l25_25960


namespace Toms_out_of_pocket_cost_l25_25765

theorem Toms_out_of_pocket_cost (visit_cost cast_cost insurance_percent : ℝ) 
  (h1 : visit_cost = 300) 
  (h2 : cast_cost = 200) 
  (h3 : insurance_percent = 0.6) : 
  (visit_cost + cast_cost) - ((visit_cost + cast_cost) * insurance_percent) = 200 :=
by
  sorry

end Toms_out_of_pocket_cost_l25_25765


namespace tan_sum_pi_over_12_l25_25991

theorem tan_sum_pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (5 * Real.pi / 12)) = 4 := 
sorry

end tan_sum_pi_over_12_l25_25991


namespace irreducible_fraction_unique_l25_25428

theorem irreducible_fraction_unique :
  ∃ (a b : ℕ), a = 5 ∧ b = 2 ∧ gcd a b = 1 ∧ (∃ n : ℕ, 10^n = a * b) :=
by
  sorry

end irreducible_fraction_unique_l25_25428


namespace father_l25_25245

-- Conditions definitions
def man's_current_age (F : ℕ) : ℕ := (2 / 5) * F
def man_after_5_years (M F : ℕ) : Prop := M + 5 = (1 / 2) * (F + 5)

-- Main statement to prove
theorem father's_age (F : ℕ) (h₁ : man's_current_age F = (2 / 5) * F)
  (h₂ : ∀ M, man_after_5_years M F → M = (2 / 5) * F + 5): F = 25 :=
sorry

end father_l25_25245


namespace probability_same_color_probability_different_color_l25_25500

def count_combinations {α : Type*} (s : Finset α) (k : ℕ) : ℕ :=
  Nat.choose s.card k

noncomputable def count_ways_same_color : ℕ :=
  (count_combinations (Finset.range 3) 2) * 2

noncomputable def count_ways_diff_color : ℕ :=
  (Finset.range 3).card * (Finset.range 3).card

noncomputable def total_ways : ℕ :=
  count_combinations (Finset.range 6) 2

noncomputable def prob_same_color : ℚ :=
  count_ways_same_color / total_ways

noncomputable def prob_diff_color : ℚ :=
  count_ways_diff_color / total_ways

theorem probability_same_color :
  prob_same_color = 2 / 5 := by
  sorry

theorem probability_different_color :
  prob_diff_color = 3 / 5 := by
  sorry

end probability_same_color_probability_different_color_l25_25500


namespace odd_function_values_l25_25335

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_values_l25_25335


namespace jacob_additional_money_needed_l25_25155

/-- Jacob's total trip cost -/
def trip_cost : ℕ := 5000

/-- Jacob's hourly wage -/
def hourly_wage : ℕ := 20

/-- Jacob's working hours -/
def working_hours : ℕ := 10

/-- Income from job -/
def job_income : ℕ := hourly_wage * working_hours

/-- Price per cookie -/
def cookie_price : ℕ := 4

/-- Number of cookies sold -/
def cookies_sold : ℕ := 24

/-- Income from cookies -/
def cookie_income : ℕ := cookie_price * cookies_sold

/-- Lottery ticket cost -/
def lottery_ticket_cost : ℕ := 10

/-- Lottery win amount -/
def lottery_win : ℕ := 500

/-- Money received from each sister -/
def sister_gift : ℕ := 500

/-- Total income from job and cookies -/
def income_without_expenses : ℕ := job_income + cookie_income

/-- Income after lottery ticket purchase -/
def income_after_ticket : ℕ := income_without_expenses - lottery_ticket_cost

/-- Total income after lottery win -/
def income_with_lottery : ℕ := income_after_ticket + lottery_win

/-- Total gift from sisters -/
def total_sisters_gift : ℕ := 2 * sister_gift

/-- Total money Jacob has -/
def total_money : ℕ := income_with_lottery + total_sisters_gift

/-- Additional amount needed by Jacob -/
def additional_needed : ℕ := trip_cost - total_money

theorem jacob_additional_money_needed : additional_needed = 3214 := by
  sorry

end jacob_additional_money_needed_l25_25155


namespace tan_sum_pi_over_12_eq_4_l25_25996

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l25_25996


namespace number_of_roses_picked_later_l25_25409

-- Given definitions
def initial_roses : ℕ := 50
def sold_roses : ℕ := 15
def final_roses : ℕ := 56

-- Compute the number of roses left after selling.
def roses_left := initial_roses - sold_roses

-- Define the final goal: number of roses picked later.
def picked_roses_later := final_roses - roses_left

-- State the theorem
theorem number_of_roses_picked_later : picked_roses_later = 21 :=
by
  sorry

end number_of_roses_picked_later_l25_25409


namespace wall_width_l25_25648

theorem wall_width (w h l V : ℝ) (h_eq : h = 4 * w) (l_eq : l = 3 * h) (V_eq : V = w * h * l) (v_val : V = 10368) : w = 6 :=
  sorry

end wall_width_l25_25648


namespace value_range_sin_neg_l25_25498

theorem value_range_sin_neg (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) : 
  Set.Icc (-1) (Real.sqrt 2 / 2) ( - (Real.sin x) ) :=
sorry

end value_range_sin_neg_l25_25498


namespace diagonals_in_regular_nine_sided_polygon_l25_25070

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25070


namespace y_coord_intersection_with_y_axis_l25_25499

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 11

-- Define the point P
def P : ℝ × ℝ := (1, curve 1)

-- Define the derivative of the curve
def derivative (x : ℝ) : ℝ := 3 * x^2

-- Define the tangent line at point P (1, 12)
def tangent_line (x : ℝ) : ℝ := 3 * (x - 1) + 12

-- Proof statement
theorem y_coord_intersection_with_y_axis : 
  tangent_line 0 = 9 :=
by
  -- proof goes here
  sorry

end y_coord_intersection_with_y_axis_l25_25499


namespace fraction_r_over_b_l25_25195

-- Definition of the conditions
def initial_expression (k : ℝ) : ℝ := 8 * k^2 - 12 * k + 20

-- Proposition statement
theorem fraction_r_over_b : ∃ a b r : ℝ, 
  (∀ k : ℝ, initial_expression k = a * (k + b)^2 + r) ∧ 
  r / b = -47.33 :=
sorry

end fraction_r_over_b_l25_25195


namespace real_number_identity_l25_25292

theorem real_number_identity (a : ℝ) (h : a^2 - a - 1 = 0) : a^8 + 7 * a^(-(4:ℝ)) = 48 := by
  sorry

end real_number_identity_l25_25292


namespace commentator_mistake_l25_25151

def round_robin_tournament : Prop :=
  ∀ (x y : ℝ),
    x + 2 * x + 13 * y = 105 ∧ x < y ∧ y < 2 * x → False

theorem commentator_mistake : round_robin_tournament :=
  by {
    sorry
  }

end commentator_mistake_l25_25151


namespace min_area_triangle_l25_25522

-- Define the points and line equation
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (30, 10)
def line (x : ℤ) : ℤ := 2 * x - 5

-- Define a function to calculate the area using Shoelace formula
noncomputable def area (C : ℤ × ℤ) : ℝ :=
  (1 / 2) * |(A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1)|

-- Prove that the minimum area of the triangle with the given conditions is 15
theorem min_area_triangle : ∃ (C : ℤ × ℤ), C.2 = line C.1 ∧ area C = 15 := sorry

end min_area_triangle_l25_25522


namespace real_solution_count_eq_31_l25_25275

theorem real_solution_count_eq_31 :
  (∃ (S : set ℝ), S = {x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ (x / 50 = real.cos x)} ∧ S.card = 31) :=
sorry

end real_solution_count_eq_31_l25_25275


namespace total_five_digit_odd_and_multiples_of_5_l25_25976

def count_odd_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 5
  choices

def count_multiples_of_5_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 2
  choices

theorem total_five_digit_odd_and_multiples_of_5 : count_odd_five_digit_numbers + count_multiples_of_5_five_digit_numbers = 63000 :=
by
  -- Proof Placeholder
  sorry

end total_five_digit_odd_and_multiples_of_5_l25_25976


namespace olaf_total_toy_cars_l25_25862

def olaf_initial_collection : ℕ := 150
def uncle_toy_cars : ℕ := 5
def auntie_toy_cars : ℕ := uncle_toy_cars + 1 -- 6 toy cars
def grandpa_toy_cars : ℕ := 2 * uncle_toy_cars -- 10 toy cars
def dad_toy_cars : ℕ := 10
def mum_toy_cars : ℕ := dad_toy_cars + 5 -- 15 toy cars
def toy_cars_received : ℕ := grandpa_toy_cars + uncle_toy_cars + dad_toy_cars + mum_toy_cars + auntie_toy_cars -- total toy cars received
def olaf_total_collection : ℕ := olaf_initial_collection + toy_cars_received

theorem olaf_total_toy_cars : olaf_total_collection = 196 := by
  sorry

end olaf_total_toy_cars_l25_25862


namespace find_parabola_equation_l25_25552

-- Define the problem conditions
def parabola_vertex_at_origin (f : ℝ → ℝ) : Prop :=
  f 0 = 0

def axis_of_symmetry_x_or_y (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ y, f 0 = y)

def passes_through_point (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop :=
  f pt.1 = pt.2

-- Define the specific forms we expect the equations of the parabola to take
def equation1 (x y : ℝ) : Prop :=
  y^2 = - (9 / 2) * x

def equation2 (x y : ℝ) : Prop :=
  x^2 = (4 / 3) * y

-- state the main theorem
theorem find_parabola_equation :
  ∃ f : ℝ → ℝ, parabola_vertex_at_origin f ∧ axis_of_symmetry_x_or_y f ∧ passes_through_point f (-2, 3) ∧
  (equation1 (-2) (f (-2)) ∨ equation2 (-2) (f (-2))) :=
sorry

end find_parabola_equation_l25_25552


namespace different_routes_calculation_l25_25636

-- Definitions for the conditions
def west_blocks := 3
def south_blocks := 2
def east_blocks := 3
def north_blocks := 3

-- Calculation of combinations for the number of sequences
def house_to_sw_corner_routes := Nat.choose (west_blocks + south_blocks) south_blocks
def ne_corner_to_school_routes := Nat.choose (east_blocks + north_blocks) east_blocks

-- Proving the total number of routes
theorem different_routes_calculation : 
  house_to_sw_corner_routes * 1 * ne_corner_to_school_routes = 200 :=
by
  -- Mathematical proof steps (to be filled)
  sorry

end different_routes_calculation_l25_25636


namespace find_ab_l25_25317

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l25_25317


namespace regular_nonagon_diagonals_correct_l25_25115

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25115


namespace jacob_additional_money_needed_l25_25156

/-- Jacob's total trip cost -/
def trip_cost : ℕ := 5000

/-- Jacob's hourly wage -/
def hourly_wage : ℕ := 20

/-- Jacob's working hours -/
def working_hours : ℕ := 10

/-- Income from job -/
def job_income : ℕ := hourly_wage * working_hours

/-- Price per cookie -/
def cookie_price : ℕ := 4

/-- Number of cookies sold -/
def cookies_sold : ℕ := 24

/-- Income from cookies -/
def cookie_income : ℕ := cookie_price * cookies_sold

/-- Lottery ticket cost -/
def lottery_ticket_cost : ℕ := 10

/-- Lottery win amount -/
def lottery_win : ℕ := 500

/-- Money received from each sister -/
def sister_gift : ℕ := 500

/-- Total income from job and cookies -/
def income_without_expenses : ℕ := job_income + cookie_income

/-- Income after lottery ticket purchase -/
def income_after_ticket : ℕ := income_without_expenses - lottery_ticket_cost

/-- Total income after lottery win -/
def income_with_lottery : ℕ := income_after_ticket + lottery_win

/-- Total gift from sisters -/
def total_sisters_gift : ℕ := 2 * sister_gift

/-- Total money Jacob has -/
def total_money : ℕ := income_with_lottery + total_sisters_gift

/-- Additional amount needed by Jacob -/
def additional_needed : ℕ := trip_cost - total_money

theorem jacob_additional_money_needed : additional_needed = 3214 := by
  sorry

end jacob_additional_money_needed_l25_25156


namespace regular_nine_sided_polygon_diagonals_l25_25040

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25040


namespace quotient_three_l25_25469

theorem quotient_three (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a * b ∣ a^2 + b^2 + 1) :
  (a^2 + b^2 + 1) / (a * b) = 3 :=
sorry

end quotient_three_l25_25469


namespace regular_nine_sided_polygon_diagonals_l25_25042

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25042


namespace book_distribution_l25_25503

theorem book_distribution (a b : ℕ) (h1 : a + b = 282) (h2 : (3 / 4) * a = (5 / 9) * b) : a = 120 ∧ b = 162 := by
  sorry

end book_distribution_l25_25503


namespace max_value_of_f_l25_25362

noncomputable def f (x : ℝ) : ℝ :=
  2022 * x ^ 2 * Real.log (x + 2022) / ((Real.log (x + 2022)) ^ 3 + 2 * x ^ 3)

theorem max_value_of_f : ∃ x : ℝ, 0 < x ∧ f x ≤ 674 :=
by
  sorry

end max_value_of_f_l25_25362


namespace perfect_square_expression_l25_25754
open Real

theorem perfect_square_expression (x : ℝ) :
  (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = (12.86 + 0.14)^2) → x = 0.28 :=
by
  sorry

end perfect_square_expression_l25_25754


namespace muffins_division_l25_25463

theorem muffins_division (total_muffins total_people muffins_per_person : ℕ) 
  (h1 : total_muffins = 20) (h2 : total_people = 5) (h3 : muffins_per_person = total_muffins / total_people) : 
  muffins_per_person = 4 := 
by
  sorry

end muffins_division_l25_25463


namespace smallest_b_l25_25199

theorem smallest_b (a b : ℕ) (hp : a > 0) (hq : b > 0) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 8) : b = 4 :=
sorry

end smallest_b_l25_25199


namespace diagonals_in_regular_nine_sided_polygon_l25_25065

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25065


namespace probability_of_odd_sum_greater_than_36_l25_25506

theorem probability_of_odd_sum_greater_than_36 :
  let tiles := finset.range 12 in
  let configurations := finset.powerset_len 3 tiles in
  let num_configurations := configurations.card.instances in
  let valid_configurations :=
    configurations.filter (λ c, (c.card = 3) ∧ (c.sum % 2 = 1) ∧ (c.sum > 6)) in
  let num_valid_configurations := valid_configurations.card.instances in
  (valid_configurations.card * valid_configurations.card * valid_configurations.card)
  / num_configurations.choose 3 = 1 / 205 :=
sorry

end probability_of_odd_sum_greater_than_36_l25_25506


namespace train_speed_incl_stoppages_l25_25682

theorem train_speed_incl_stoppages
  (speed_excl_stoppages : ℝ)
  (stoppage_time_minutes : ℝ)
  (h1 : speed_excl_stoppages = 42)
  (h2 : stoppage_time_minutes = 21.428571428571423)
  : ∃ speed_incl_stoppages, speed_incl_stoppages = 27 := 
sorry

end train_speed_incl_stoppages_l25_25682


namespace solve_system_of_equations_l25_25420

theorem solve_system_of_equations :
  ∃ x y : ℤ, (2 * x + 7 * y = -6) ∧ (2 * x - 5 * y = 18) ∧ (x = 4) ∧ (y = -2) := 
by
  -- Proof will go here
  sorry

end solve_system_of_equations_l25_25420


namespace total_food_each_day_l25_25804

-- Conditions
def num_dogs : ℕ := 2
def food_per_dog : ℝ := 0.125
def total_food : ℝ := num_dogs * food_per_dog

-- Proof statement
theorem total_food_each_day : total_food = 0.25 :=
by
  sorry

end total_food_each_day_l25_25804


namespace waiter_income_fraction_l25_25901

theorem waiter_income_fraction (S T : ℝ) (hT : T = 5/4 * S) :
  T / (S + T) = 5 / 9 :=
by
  sorry

end waiter_income_fraction_l25_25901


namespace sum_of_binary_digits_345_l25_25228

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else List.reverse (List.unfold (λ n, if n = 0 then none else some (n % 2, n / 2)) n)

def sum_of_digits (digits : List ℕ) : ℕ :=
  digits.foldr (· + ·) 0
  
-- Define the specific example
def digits_of_345 : List ℕ := decimal_to_binary 345

def sum_of_digits_of_345 : ℕ := sum_of_digits digits_of_345

theorem sum_of_binary_digits_345 : sum_of_digits_of_345 = 5 :=
by 
  sorry

end sum_of_binary_digits_345_l25_25228


namespace max_sum_x_y_l25_25222

theorem max_sum_x_y 
  (x y : ℝ)
  (h1 : x^2 + y^2 = 100)
  (h2 : x * y = 40) :
  x + y = 6 * Real.sqrt 5 :=
sorry

end max_sum_x_y_l25_25222


namespace compound_interest_principal_l25_25882

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (Real.exp (T * Real.log (1 + R / 100)) - 1)

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem compound_interest_principal :
  let P_SI := 2800.0000000000027
  let R_SI := 5
  let T_SI := 3
  let P_CI := 4000
  let R_CI := 10
  let T_CI := 2
  let SI := simple_interest P_SI R_SI T_SI
  let CI := 2 * SI
  CI = compound_interest P_CI R_CI T_CI → P_CI = 4000 :=
by
  intros
  sorry

end compound_interest_principal_l25_25882


namespace sum_mod_17_l25_25278

/--
Given the sum of the numbers 82, 83, 84, 85, 86, 87, 88, and 89, and the divisor 17,
prove that the remainder when dividing this sum by 17 is 11.
-/
theorem sum_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 :=
by
  sorry

end sum_mod_17_l25_25278


namespace paint_faces_l25_25952

def cuboid_faces : ℕ := 6
def number_of_cuboids : ℕ := 8 
def total_faces_painted : ℕ := cuboid_faces * number_of_cuboids

theorem paint_faces (h1 : cuboid_faces = 6) (h2 : number_of_cuboids = 8) : total_faces_painted = 48 := by
  -- conditions are defined above
  sorry

end paint_faces_l25_25952


namespace find_a_b_for_odd_function_l25_25308

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l25_25308


namespace nine_sided_polygon_diagonals_l25_25054

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25054


namespace regular_nonagon_diagonals_correct_l25_25117

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25117


namespace new_salary_correct_l25_25534

-- Define the initial salary and percentage increase as given in the conditions
def initial_salary : ℝ := 10000
def percentage_increase : ℝ := 0.02

-- Define the function that calculates the new salary after a percentage increase
def new_salary (initial_salary : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_salary + (initial_salary * percentage_increase)

-- The theorem statement that proves the new salary is €10,200
theorem new_salary_correct :
  new_salary initial_salary percentage_increase = 10200 := by
  sorry

end new_salary_correct_l25_25534


namespace range_of_a_range_of_m_l25_25904

-- Problem (1)
theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, ¬(2 * x^2 - 3 * x + 1 ≤ 0) → ¬(x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) ∧ ¬(x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0) → ¬(2 * x^2 - 3 * x + 1 ≤ 0)) :
  0 ≤ a ∧ a ≤ 0.5 :=
sorry

-- Problem (2)
theorem range_of_m (m : ℝ) (h1 : ∃ x ∈ (0, 1), ∃ y ∈ (2, 3), x^2 + (m - 3) * x + m = 0 ∧ y^2 + (m - 3) * y + m = 0 ∨ ∀ x : ℝ, ln (m * x^2 - 2 * x + 1) :=
  (0 < m ∧ m < 2 / 3) ∨ 1 < m :=
sorry

end range_of_a_range_of_m_l25_25904


namespace tan_sum_pi_over_12_eq_4_l25_25995

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l25_25995


namespace totalSandwiches_l25_25664

def numberOfPeople : ℝ := 219.0
def sandwichesPerPerson : ℝ := 3.0

theorem totalSandwiches : numberOfPeople * sandwichesPerPerson = 657.0 := by
  -- Proof goes here
  sorry

end totalSandwiches_l25_25664


namespace cost_of_each_soda_l25_25722

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l25_25722


namespace find_ab_average_l25_25712

variable (a b c k : ℝ)

-- Conditions
def sum_condition : Prop := (4 + 6 + 8 + 12 + a + b + c) / 7 = 20
def abc_condition : Prop := a + b + c = 3 * ((4 + 6 + 8) / 3)

-- Theorem
theorem find_ab_average 
  (sum_cond : sum_condition a b c) 
  (abc_cond : abc_condition a b c) 
  (c_eq_k : c = k) : 
  (a + b) / 2 = (18 - k) / 2 :=
sorry  -- Proof is omitted


end find_ab_average_l25_25712


namespace greatest_possible_sum_xy_l25_25221

noncomputable def greatest_possible_xy (x y : ℝ) :=
  x^2 + y^2 = 100 ∧ xy = 40 → x + y = 6 * Real.sqrt 5

theorem greatest_possible_sum_xy {x y : ℝ} (h1 : x^2 + y^2 = 100) (h2 : xy = 40) :
  x + y ≤ 6 * Real.sqrt 5 :=
sorry

end greatest_possible_sum_xy_l25_25221


namespace no_solution_outside_intervals_l25_25550

theorem no_solution_outside_intervals (x a : ℝ) :
  (a < 0 ∨ a > 10) → 3 * |x + 3 * a| + |x + a^2| + 2 * x ≠ a :=
by {
  sorry
}

end no_solution_outside_intervals_l25_25550


namespace k_value_for_root_multiplicity_l25_25556

theorem k_value_for_root_multiplicity (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 3) = k / (x - 3) ∧ (x-3 = 0)) → k = 2 :=
by
  sorry

end k_value_for_root_multiplicity_l25_25556


namespace nine_sided_polygon_diagonals_l25_25101

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25101


namespace cos_squared_value_l25_25563

theorem cos_squared_value (α : ℝ) (h : Real.tan (α + π/4) = 3/4) : Real.cos (π/4 - α) ^ 2 = 9 / 25 :=
sorry

end cos_squared_value_l25_25563


namespace T_shaped_area_l25_25271

theorem T_shaped_area (a b c d : ℕ) (side1 side2 side3 large_side : ℕ)
  (h_side1: side1 = 2)
  (h_side2: side2 = 2)
  (h_side3: side3 = 4)
  (h_large_side: large_side = 6)
  (h_area_large_square : a = large_side * large_side)
  (h_area_square1 : b = side1 * side1)
  (h_area_square2 : c = side2 * side2)
  (h_area_square3 : d = side3 * side3) :
  a - (b + c + d) = 12 := by
  sorry

end T_shaped_area_l25_25271


namespace total_lunch_bill_l25_25637

def cost_of_hotdog : ℝ := 5.36
def cost_of_salad : ℝ := 5.10

theorem total_lunch_bill : cost_of_hotdog + cost_of_salad = 10.46 := 
by
  sorry

end total_lunch_bill_l25_25637


namespace total_voters_in_districts_l25_25501

theorem total_voters_in_districts :
  let D1 := 322
  let D2 := (D1 / 2) - 19
  let D3 := 2 * D1
  let D4 := D2 + 45
  let D5 := (3 * D3) - 150
  let D6 := (D1 + D4) + (1 / 5) * (D1 + D4)
  let D7 := D2 + (D5 - D2) / 2
  D1 + D2 + D3 + D4 + D5 + D6 + D7 = 4650 := 
by
  sorry

end total_voters_in_districts_l25_25501


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25013

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25013


namespace prob_product_less_than_36_is_15_over_16_l25_25479

noncomputable def prob_product_less_than_36 : ℚ := sorry

theorem prob_product_less_than_36_is_15_over_16 :
  prob_product_less_than_36 = 15 / 16 := 
sorry

end prob_product_less_than_36_is_15_over_16_l25_25479


namespace victory_saved_less_l25_25509

-- Definitions based on conditions
def total_savings : ℕ := 1900
def sam_savings : ℕ := 1000
def victory_savings : ℕ := total_savings - sam_savings

-- Prove that Victory saved $100 less than Sam
theorem victory_saved_less : sam_savings - victory_savings = 100 := by
  -- placeholder for the proof
  sorry

end victory_saved_less_l25_25509


namespace license_plate_combinations_l25_25675

theorem license_plate_combinations :
  let letters := 26
  let choose (n k : ℕ) := nat.choose n k
  let factorial := nat.factorial
  let digits := 10
  26 * choose 25 3 * choose 5 2 * factorial 3 * 10 * 9 * 8 = 44,400,000 :=
by
  let letters := 26
  let choose (n k : ℕ) := nat.choose
  let factorial := nat.factorial
  let digits := 10
  calc
    26 * choose 25 3 * choose 5 2 * factorial 3 * 10 * 9 * 8 = 26 * 2300 * 10 * 6 * 720 : by
      -- Expand and verify intermediate steps if needed
      sorry
    ... = 26 * 25440000 : by
      -- Intermediate calculation
      sorry
    ... = 44400000 : by
      -- Final multiplication
      sorry

end license_plate_combinations_l25_25675


namespace find_a_b_l25_25337

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x, f (-x) = -f (x)

theorem find_a_b 
  (f : ℝ → ℝ := λ x, Real.log (abs (a + 1 / (1 - x))) + b)
  (h_odd : is_odd_function f) 
  (h_domain : ∀ x, x ≠ 1) :
  a = -1 / 2 ∧ b = Real.log 2 :=
sorry

end find_a_b_l25_25337


namespace determine_a_b_odd_function_l25_25325

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def func (a b : ℝ) (x : ℝ) : ℝ :=
  Real.log (|a + (1 / (1 - x))|) + b

theorem determine_a_b_odd_function :
  ∃ (a b : ℝ), (∀ x, func a b (-x) = -func a b x) ↔ (a = -1/2 ∧ b = Real.log 2) :=
sorry

end determine_a_b_odd_function_l25_25325


namespace divisor_exists_l25_25210

theorem divisor_exists (n : ℕ) : (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) →
                                (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) ∧
                                (n = 3) :=
by
  sorry

end divisor_exists_l25_25210


namespace certain_number_is_3_l25_25454

-- Given conditions
variables (z x : ℤ)
variable (k : ℤ)
variable (n : ℤ)

-- Conditions
-- Remainder when z is divided by 9 is 6
def is_remainder_6 (z : ℤ) := ∃ k : ℤ, z = 9 * k + 6
-- (z + x) / 9 is an integer
def is_integer_division (z x : ℤ) := ∃ m : ℤ, (z + x) / 9 = m

-- Proof to show that x must be 3
theorem certain_number_is_3 (z : ℤ) (h1 : is_remainder_6 z) (h2 : is_integer_division z x) : x = 3 :=
sorry

end certain_number_is_3_l25_25454


namespace nonneg_int_solution_coprime_l25_25190

theorem nonneg_int_solution_coprime (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : c ≥ (a - 1) * (b - 1)) :
  ∃ (x y : ℕ), c = a * x + b * y :=
sorry

end nonneg_int_solution_coprime_l25_25190


namespace number_of_diagonals_l25_25086

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25086


namespace diagonals_in_nine_sided_polygon_l25_25080

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25080


namespace regular_nine_sided_polygon_diagonals_l25_25098

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25098


namespace soft_lenses_more_than_hard_l25_25786

-- Define the problem conditions as Lean definitions
def total_sales (S H : ℕ) : Prop := 150 * S + 85 * H = 1455
def total_pairs (S H : ℕ) : Prop := S + H = 11

-- The theorem we need to prove
theorem soft_lenses_more_than_hard (S H : ℕ) (h1 : total_sales S H) (h2 : total_pairs S H) : S - H = 5 :=
by
  sorry

end soft_lenses_more_than_hard_l25_25786


namespace part1_intersection_1_part1_union_1_part2_range_a_l25_25742

open Set

def U := ℝ
def A (x : ℝ) := -1 < x ∧ x < 3
def B (a x : ℝ) := a - 1 ≤ x ∧ x ≤ a + 6

noncomputable def part1_a : ℝ → Prop := sorry
noncomputable def part1_b : ℝ → Prop := sorry

-- part (1)
theorem part1_intersection_1 (a : ℝ) : A x ∧ B a x := sorry

theorem part1_union_1 (a : ℝ) : A x ∨ B a x := sorry

-- part (2)
theorem part2_range_a : {a : ℝ | -3 ≤ a ∧ a ≤ 0} := sorry

end part1_intersection_1_part1_union_1_part2_range_a_l25_25742


namespace y_expression_value_l25_25495

theorem y_expression_value
  (y : ℝ)
  (h : y + 2 / y = 2) :
  y^6 + 3 * y^4 - 4 * y^2 + 2 = 2 := sorry

end y_expression_value_l25_25495


namespace last_two_videos_length_l25_25974

noncomputable def ad1 : ℕ := 45
noncomputable def ad2 : ℕ := 30
noncomputable def pause1 : ℕ := 45
noncomputable def pause2 : ℕ := 30
noncomputable def video1 : ℕ := 120
noncomputable def video2 : ℕ := 270
noncomputable def total_time : ℕ := 960

theorem last_two_videos_length : 
    ∃ v : ℕ, 
    v = 210 ∧ 
    total_time = ad1 + ad2 + video1 + video2 + pause1 + pause2 + 2 * v :=
by
  sorry

end last_two_videos_length_l25_25974


namespace diagonals_in_nine_sided_polygon_l25_25074

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25074


namespace largest_number_l25_25398

-- Define the given numbers
def A : ℝ := 0.986
def B : ℝ := 0.9859
def C : ℝ := 0.98609
def D : ℝ := 0.896
def E : ℝ := 0.8979
def F : ℝ := 0.987

-- State the theorem that F is the largest number among A, B, C, D, and E
theorem largest_number : F > A ∧ F > B ∧ F > C ∧ F > D ∧ F > E := by
  sorry

end largest_number_l25_25398


namespace polygon_sides_count_l25_25795

-- Definitions for each polygon and their sides
def pentagon_sides := 5
def square_sides := 4
def hexagon_sides := 6
def heptagon_sides := 7
def nonagon_sides := 9

-- Compute the total number of sides
def total_exposed_sides :=
  (pentagon_sides + nonagon_sides - 2) + (square_sides + hexagon_sides + heptagon_sides - 6)

theorem polygon_sides_count : total_exposed_sides = 23 :=
by
  -- Mathematical proof steps can be detailed here
  -- For now, let's assume it is correctly given as a single number
  sorry

end polygon_sides_count_l25_25795


namespace happy_boys_count_l25_25859

def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := total_children - happy_children - sad_children

def total_boys := 19
def total_girls := 41
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

def sad_boys := sad_children - sad_girls

theorem happy_boys_count :
  total_boys - sad_boys - neither_happy_nor_sad_boys = 6 :=
by
  sorry

end happy_boys_count_l25_25859


namespace sqrt_multiplication_and_subtraction_l25_25422

theorem sqrt_multiplication_and_subtraction :
  (Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3) = 6 * Real.sqrt 3 := 
by
  sorry

end sqrt_multiplication_and_subtraction_l25_25422


namespace cos_A_minus_B_l25_25939

variable {A B : ℝ}

-- Conditions
def cos_conditions (A B : ℝ) : Prop :=
  (Real.cos A + Real.cos B = 1 / 2)

def sin_conditions (A B : ℝ) : Prop :=
  (Real.sin A + Real.sin B = 3 / 2)

-- Mathematically equivalent proof problem
theorem cos_A_minus_B (h1 : cos_conditions A B) (h2 : sin_conditions A B) :
  Real.cos (A - B) = 1 / 4 := 
sorry

end cos_A_minus_B_l25_25939


namespace find_g50_l25_25203

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g50 (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x * y) = y * g x)
  (h1 : g 1 = 10) : g 50 = 50 * 10 :=
by
  -- The proof sketch here; the detailed proof is omitted
  sorry

end find_g50_l25_25203


namespace diagonal_of_rectangle_l25_25629

theorem diagonal_of_rectangle (l w d : ℝ) (h_length : l = 15) (h_area : l * w = 120) (h_diagonal : d^2 = l^2 + w^2) : d = 17 :=
by
  sorry

end diagonal_of_rectangle_l25_25629


namespace train_passes_man_in_12_seconds_l25_25771

noncomputable def time_to_pass_man (train_length: ℝ) (train_speed_kmph: ℝ) (man_speed_kmph: ℝ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18)
  train_length / relative_speed_mps

theorem train_passes_man_in_12_seconds :
  time_to_pass_man 220 60 6 = 12 := by
 sorry

end train_passes_man_in_12_seconds_l25_25771


namespace y_values_relation_l25_25575

theorem y_values_relation :
  ∀ y1 y2 y3 : ℝ,
    (y1 = (-3 + 1) ^ 2 + 1) →
    (y2 = (0 + 1) ^ 2 + 1) →
    (y3 = (2 + 1) ^ 2 + 1) →
    y2 < y1 ∧ y1 < y3 :=
by
  sorry

end y_values_relation_l25_25575


namespace arrange_descending_order_l25_25851

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 3) / Real.log 2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem arrange_descending_order : c > a ∧ a > b := by
  sorry

end arrange_descending_order_l25_25851


namespace proof1_l25_25677

def prob1 : Prop :=
  (1 : ℝ) * (Real.sqrt 45 + Real.sqrt 18) - (Real.sqrt 8 - Real.sqrt 125) = 8 * Real.sqrt 5 + Real.sqrt 2

theorem proof1 : prob1 :=
by
  sorry

end proof1_l25_25677


namespace diagonals_in_regular_nine_sided_polygon_l25_25006

theorem diagonals_in_regular_nine_sided_polygon : 
  ∀ (n : ℕ), n = 9 → (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) - n = 27 :=
begin
  intros n hn,
  have : (∑ i in (finset.range n).powerset.filter (λ s, s.card = 2), 1) = nat.choose n 2,
  {
    rw nat.choose,
    rw hn,
  },
  rw this,
  simp [nat.choose, hn],
  sorry
end

end diagonals_in_regular_nine_sided_polygon_l25_25006


namespace inscribed_circle_radius_l25_25215

noncomputable def radius_inscribed_circle (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ) :=
  if (r1 = 2 ∧ r2 = 6) ∧ ((O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) then
    2 * (Real.sqrt 3 - 1)
  else
    0

theorem inscribed_circle_radius (O1 O2 D : ℝ × ℝ) (r1 r2 : ℝ)
  (h1 : r1 = 2) (h2 : r2 = 6)
  (h3 : (O1.fst - O2.fst)^2 + (O1.snd - O2.snd)^2 = 64) :
  radius_inscribed_circle O1 O2 D r1 r2 = 2 * (Real.sqrt 3 - 1) :=
by
  sorry

end inscribed_circle_radius_l25_25215


namespace composite_evaluation_at_two_l25_25727

-- Define that P(x) is a polynomial with coefficients in {0, 1}
def is_binary_coefficient_polynomial (P : Polynomial ℤ) : Prop :=
  ∀ (n : ℕ), P.coeff n = 0 ∨ P.coeff n = 1

-- Define that P(x) can be factored into two nonconstant polynomials with integer coefficients
def is_reducible_to_nonconstant_polynomials (P : Polynomial ℤ) : Prop :=
  ∃ (f g : Polynomial ℤ), f.degree > 0 ∧ g.degree > 0 ∧ P = f * g

theorem composite_evaluation_at_two {P : Polynomial ℤ}
  (h1 : is_binary_coefficient_polynomial P)
  (h2 : is_reducible_to_nonconstant_polynomials P) :
  ∃ (m n : ℤ), m > 1 ∧ n > 1 ∧ P.eval 2 = m * n := sorry

end composite_evaluation_at_two_l25_25727


namespace number_of_real_solutions_l25_25277

-- Definition of the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- The main theorem stating the number of solutions
theorem number_of_real_solutions : 
  ∃ (n : ℕ), n = 32 ∧ ∀ x : ℝ, equation x → -50 ≤ x ∧ x ≤ 50 :=
sorry

end number_of_real_solutions_l25_25277


namespace number_of_pairs_lcm_600_l25_25553

theorem number_of_pairs_lcm_600 :
  ∃ n, n = 53 ∧ (∀ m n : ℕ, (m ≤ n ∧ m > 0 ∧ n > 0 ∧ Nat.lcm m n = 600) ↔ n = 53) := sorry

end number_of_pairs_lcm_600_l25_25553


namespace find_m_l25_25735

open Nat

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (a : ℕ := Nat.choose (2 * m) m) 
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 :=
by
  sorry

end find_m_l25_25735


namespace range_of_m_l25_25826

variable (m : ℝ)

def hyperbola (m : ℝ) := (x y : ℝ) → (x^2 / (1 + m)) - (y^2 / (3 - m)) = 1

def eccentricity_condition (m : ℝ) := (2 / (Real.sqrt (1 + m)) > Real.sqrt 2)

theorem range_of_m (m : ℝ) (h1 : 1 + m > 0) (h2 : 3 - m > 0) (h3 : eccentricity_condition m) :
 -1 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l25_25826


namespace first_shaded_square_in_each_column_l25_25532

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem first_shaded_square_in_each_column : 
  ∃ n, triangular_number n = 120 ∧ ∀ m < n, ¬ ∀ k < 8, ∃ j ≤ m, ((triangular_number j) % 8) = k := 
by
  sorry

end first_shaded_square_in_each_column_l25_25532


namespace diagonals_in_nonagon_l25_25135

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25135


namespace contrapositive_quadratic_roots_l25_25547

theorem contrapositive_quadratic_roots (m : ℝ) (h_discriminant : 1 + 4 * m < 0) : m ≤ 0 :=
sorry

end contrapositive_quadratic_roots_l25_25547


namespace smallest_number_of_students_l25_25590

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l25_25590


namespace magician_can_always_determine_hidden_pair_l25_25521

-- Define the cards as an enumeration
inductive Card
| one | two | three | four | five

-- Define a pair of cards
structure CardPair where
  first : Card
  second : Card

-- Define the function the magician uses to decode the hidden pair 
-- based on the two cards the assistant points out, encoded as a pentagon
noncomputable def magician_decodes (assistant_cards spectator_announced: CardPair) : CardPair := sorry

-- Theorem statement: given the conditions, the magician can always determine the hidden pair.
theorem magician_can_always_determine_hidden_pair 
  (hidden_cards assistant_cards spectator_announced : CardPair)
  (assistant_strategy : CardPair → CardPair)
  (h : assistant_strategy assistant_cards = spectator_announced)
  : magician_decodes assistant_cards spectator_announced = hidden_cards := sorry

end magician_can_always_determine_hidden_pair_l25_25521


namespace nine_sided_polygon_diagonals_count_l25_25021

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25021


namespace diagonals_in_nine_sided_polygon_l25_25059

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25059


namespace readers_in_group_l25_25962

theorem readers_in_group (S L B T : ℕ) (hS : S = 120) (hL : L = 90) (hB : B = 60) :
  T = S + L - B → T = 150 :=
by
  intro h₁
  rw [hS, hL, hB] at h₁
  linarith

end readers_in_group_l25_25962


namespace diagonals_in_regular_nine_sided_polygon_l25_25069

theorem diagonals_in_regular_nine_sided_polygon : 
  ∃ n d : ℕ, n = 9 ∧ d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end diagonals_in_regular_nine_sided_polygon_l25_25069


namespace find_n_l25_25928

theorem find_n (n : ℕ) (h : 2^6 * 3^3 * n = nat.factorial 10) : n = 2100 :=
sorry

end find_n_l25_25928


namespace movie_box_office_revenue_l25_25982

variable (x : ℝ)

theorem movie_box_office_revenue (h : 300 + 300 * (1 + x) + 300 * (1 + x)^2 = 1000) :
  3 + 3 * (1 + x) + 3 * (1 + x)^2 = 10 :=
by
  sorry

end movie_box_office_revenue_l25_25982


namespace find_a_and_b_l25_25315

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l25_25315


namespace find_a_and_b_l25_25313

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l25_25313


namespace machine_P_additional_hours_unknown_l25_25367

noncomputable def machine_A_rate : ℝ := 1.0000000000000013

noncomputable def machine_Q_rate : ℝ := machine_A_rate + 0.10 * machine_A_rate

noncomputable def total_sprockets : ℝ := 110

noncomputable def machine_Q_hours : ℝ := total_sprockets / machine_Q_rate

variable (x : ℝ) -- additional hours taken by Machine P

theorem machine_P_additional_hours_unknown :
  ∃ x, total_sprockets / machine_Q_rate + x = total_sprockets / ((total_sprockets + total_sprockets / machine_Q_rate * x) / total_sprockets) :=
sorry

end machine_P_additional_hours_unknown_l25_25367


namespace range_of_m_l25_25827

noncomputable def set_A (x : ℝ) : ℝ := x^2 - (3 / 2) * x + 1

def A : Set ℝ := {y | ∃ (x : ℝ), x ∈ (Set.Icc (-1/2 : ℝ) 2) ∧ y = set_A x}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m + 1 ∨ x ≤ m - 1}

def sufficient_condition (m : ℝ) : Prop := A ⊆ B m

theorem range_of_m :
  {m : ℝ | sufficient_condition m} = {m | m ≤ -(9 / 16) ∨ m ≥ 3} :=
sorry

end range_of_m_l25_25827


namespace garden_perimeter_l25_25775

theorem garden_perimeter (width_garden length_playground width_playground : ℕ) 
  (h1 : width_garden = 12) 
  (h2 : length_playground = 16) 
  (h3 : width_playground = 12) 
  (area_playground : ℕ)
  (h4 : area_playground = length_playground * width_playground) 
  (area_garden : ℕ) 
  (h5 : area_garden = area_playground) 
  (length_garden : ℕ) 
  (h6 : area_garden = length_garden * width_garden) :
  2 * length_garden + 2 * width_garden = 56 := by
  sorry

end garden_perimeter_l25_25775


namespace cost_of_shoes_l25_25386

   theorem cost_of_shoes (initial_budget remaining_budget : ℝ) (H_initial : initial_budget = 999) (H_remaining : remaining_budget = 834) : 
   initial_budget - remaining_budget = 165 := by
     sorry
   
end cost_of_shoes_l25_25386


namespace w12_plus_inv_w12_l25_25817

open Complex

-- Given conditions
def w_plus_inv_w_eq_two_cos_45 (w : ℂ) : Prop :=
  w + (1 / w) = 2 * Real.cos (Real.pi / 4)

-- Statement of the theorem to prove
theorem w12_plus_inv_w12 {w : ℂ} (h : w_plus_inv_w_eq_two_cos_45 w) : 
  w^12 + (1 / (w^12)) = -2 :=
sorry

end w12_plus_inv_w12_l25_25817


namespace nine_sided_polygon_diagonals_l25_25103

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25103


namespace find_a_b_for_odd_function_l25_25311

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_a_b_for_odd_function (a b : ℝ) :
  is_odd (λ x : ℝ, Real.log (abs (a + 1 / (1 - x))) + b) ↔
  a = -1/2 ∧ b = Real.log 2 :=
sorry

end find_a_b_for_odd_function_l25_25311


namespace average_income_l25_25644

-- Lean statement to express the given mathematical problem
theorem average_income (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : A = 3000) :
  (A + C) / 2 = 4200 :=
by
  sorry

end average_income_l25_25644


namespace sincos_terminal_side_l25_25569

noncomputable def sincos_expr (α : ℝ) :=
  let P : ℝ × ℝ := (-4, 3)
  let r := Real.sqrt (P.1 ^ 2 + P.2 ^ 2)
  let sinα := P.2 / r
  let cosα := P.1 / r
  sinα + 2 * cosα = -1

theorem sincos_terminal_side :
  sincos_expr α :=
by
  sorry

end sincos_terminal_side_l25_25569


namespace B_work_time_alone_l25_25405

theorem B_work_time_alone
  (A_rate : ℝ := 1 / 8)
  (together_rate : ℝ := 3 / 16) :
  ∃ (B_days : ℝ), B_days = 16 :=
by
  sorry

end B_work_time_alone_l25_25405


namespace first_number_in_expression_l25_25237

theorem first_number_in_expression (a b c d e : ℝ)
  (h_expr : (a * b * c) / d + e = 2229) :
  a = 26.3 :=
  sorry

end first_number_in_expression_l25_25237


namespace cos_A_eq_a_eq_l25_25714

-- Defining the problem conditions:
variables {A B C a b c : ℝ}
variable (sin_eq : Real.sin (B + C) = 3 * Real.sin (A / 2) ^ 2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 6)
variable (sum_eq : b + c = 8)
variable (bc_prod_eq : b * c = 13)
variable (cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)

-- Proving the statements:
theorem cos_A_eq : Real.cos A = 5 / 13 :=
sorry

theorem a_eq : a = 3 * Real.sqrt 2 :=
sorry

end cos_A_eq_a_eq_l25_25714


namespace insufficient_data_to_compare_l25_25246

variable (M P O : ℝ)

theorem insufficient_data_to_compare (h1 : M < P) (h2 : O > M) : ¬(P > O) ∧ ¬(O > P) :=
sorry

end insufficient_data_to_compare_l25_25246


namespace units_digit_l25_25807

noncomputable def C := 20 + Real.sqrt 153
noncomputable def D := 20 - Real.sqrt 153

theorem units_digit (h : ∀ n ≥ 1, 20 ^ n % 10 = 0) :
  (C ^ 12 + D ^ 12) % 10 = 0 :=
by
  -- Proof will be provided based on the outlined solution
  sorry

end units_digit_l25_25807


namespace solving_linear_equations_problems_l25_25372

def num_total_math_problems : ℕ := 140
def percent_algebra_problems : ℝ := 0.40
def fraction_solving_linear_equations : ℝ := 0.50

theorem solving_linear_equations_problems :
  let num_algebra_problems := percent_algebra_problems * num_total_math_problems
  let num_solving_linear_equations := fraction_solving_linear_equations * num_algebra_problems
  num_solving_linear_equations = 28 :=
by
  sorry

end solving_linear_equations_problems_l25_25372


namespace Jill_age_l25_25747

variable (J R : ℕ) -- representing Jill's current age and Roger's current age

theorem Jill_age :
  (R = 2 * J + 5) →
  (R - J = 25) →
  J = 20 :=
by
  intros h1 h2
  sorry

end Jill_age_l25_25747


namespace no_tiling_with_seven_sided_convex_l25_25868

noncomputable def Polygon := {n : ℕ // 3 ≤ n}

def convex (M : Polygon) : Prop := sorry

def tiles_plane (M : Polygon) : Prop := sorry

theorem no_tiling_with_seven_sided_convex (M : Polygon) (h_convex : convex M) (h_sides : 7 ≤ M.1) : ¬ tiles_plane M := sorry

end no_tiling_with_seven_sided_convex_l25_25868


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25015

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25015


namespace find_A_l25_25618

def is_divisible (n : ℕ) (d : ℕ) : Prop := d ∣ n

noncomputable def valid_digit (A : ℕ) : Prop :=
  A < 10

noncomputable def digit_7_number := 653802 * 10

theorem find_A (A : ℕ) (h : valid_digit A) :
  is_divisible (digit_7_number + A) 2 ∧
  is_divisible (digit_7_number + A) 3 ∧
  is_divisible (digit_7_number + A) 4 ∧
  is_divisible (digit_7_number + A) 6 ∧
  is_divisible (digit_7_number + A) 8 ∧
  is_divisible (digit_7_number + A) 9 ∧
  is_divisible (digit_7_number + A) 25 →
  A = 0 :=
sorry

end find_A_l25_25618


namespace multiplication_solution_l25_25433

theorem multiplication_solution 
  (x : ℤ) 
  (h : 72517 * x = 724807415) : 
  x = 9999 := 
sorry

end multiplication_solution_l25_25433


namespace regular_nine_sided_polygon_diagonals_l25_25041

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25041


namespace derivative_of_function_l25_25376

theorem derivative_of_function
  (y : ℝ → ℝ)
  (h : ∀ x, y x = (1/2) * (Real.exp x + Real.exp (-x))) :
  ∀ x, deriv y x = (1/2) * (Real.exp x - Real.exp (-x)) :=
by
  sorry

end derivative_of_function_l25_25376


namespace coin_draws_expected_value_l25_25511

theorem coin_draws_expected_value :
  ∃ f : ℕ → ℝ, (∀ (n : ℕ), n ≥ 4 → f n = (3 : ℝ)) := sorry

end coin_draws_expected_value_l25_25511


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25019

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25019


namespace sum_of_coefficients_of_poly_is_neg_1_l25_25294

noncomputable def evaluate_poly_sum (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) : ℂ :=
  α^2005 + β^2005

theorem sum_of_coefficients_of_poly_is_neg_1 (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  evaluate_poly_sum α β h1 h2 = -1 := by
  sorry

end sum_of_coefficients_of_poly_is_neg_1_l25_25294


namespace days_worked_per_week_l25_25783

theorem days_worked_per_week
  (hourly_wage : ℕ) (hours_per_day : ℕ) (total_earnings : ℕ) (weeks : ℕ)
  (H_wage : hourly_wage = 12) (H_hours : hours_per_day = 9) (H_earnings : total_earnings = 3780) (H_weeks : weeks = 7) :
  (total_earnings / weeks) / (hourly_wage * hours_per_day) = 5 :=
by 
  sorry

end days_worked_per_week_l25_25783


namespace nine_sided_polygon_diagonals_l25_25031

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25031


namespace george_change_sum_l25_25286

theorem george_change_sum :
  ∃ n m : ℕ,
    0 ≤ n ∧ n < 19 ∧
    0 ≤ m ∧ m < 10 ∧
    (7 + 5 * n) = (4 + 10 * m) ∧
    (7 + 5 * 14) + (4 + 10 * 7) = 144 :=
by
  -- We declare the problem stating that there exist natural numbers n and m within
  -- the given ranges such that the sums of valid change amounts add up to 144 cents.
  sorry

end george_change_sum_l25_25286


namespace train_length_l25_25772

theorem train_length (S L : ℝ)
  (h1 : L = S * 11)
  (h2 : L + 120 = S * 22) : 
  L = 120 := 
by
  -- proof goes here
  sorry

end train_length_l25_25772


namespace problem1_problem2_l25_25546

theorem problem1 : ((- (5 : ℚ) / 6) + 2 / 3) / (- (7 / 12)) * (7 / 2) = 1 := 
sorry

theorem problem2 : ((1 - 1 / 6) * (-3) - (- (11 / 6)) / (- (22 / 3))) = - (11 / 4) := 
sorry

end problem1_problem2_l25_25546


namespace find_ab_l25_25319

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := log (abs (a + 1 / (1 - x))) + b

theorem find_ab (a b : ℝ) :
  (∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) →
  a = -1/2 ∧ b = log 2 :=
by
  sorry

end find_ab_l25_25319


namespace jeff_total_run_is_290_l25_25461

variables (monday_to_wednesday_run : ℕ)
variables (thursday_run : ℕ)
variables (friday_run : ℕ)

def jeff_weekly_run_total : ℕ :=
  monday_to_wednesday_run + thursday_run + friday_run

theorem jeff_total_run_is_290 :
  (60 * 3) + (60 - 20) + (60 + 10) = 290 :=
by
  sorry

end jeff_total_run_is_290_l25_25461


namespace correct_set_of_equations_l25_25761

-- Define the digits x and y as integers
def digits (x y : ℕ) := x + y = 8

-- Conditions
def condition_1 (x y : ℕ) := 10*y + x + 18 = 10*x + y

theorem correct_set_of_equations : 
  ∃ (x y : ℕ), digits x y ∧ condition_1 x y :=
sorry

end correct_set_of_equations_l25_25761


namespace parallel_lines_of_equation_l25_25869

theorem parallel_lines_of_equation (y : Real) :
  (y - 2) * (y + 3) = 0 → (y = 2 ∨ y = -3) :=
by
  sorry

end parallel_lines_of_equation_l25_25869


namespace no_integer_roots_l25_25978

theorem no_integer_roots (n : ℕ) (p : Fin (2*n + 1) → ℤ)
  (non_zero : ∀ i, p i ≠ 0)
  (sum_non_zero : (Finset.univ.sum (λ i => p i)) ≠ 0) :
  ∃ P : ℤ → ℤ, ∀ x : ℤ, P x ≠ 0 → x > 1 ∨ x < -1 := sorry

end no_integer_roots_l25_25978


namespace regular_nine_sided_polygon_diagonals_l25_25100

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25100


namespace nine_sided_polygon_diagonals_count_l25_25023

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25023


namespace smallest_possible_number_of_students_l25_25594

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l25_25594


namespace sequence_a_n_a5_eq_21_l25_25202

theorem sequence_a_n_a5_eq_21 
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  a 5 = 21 :=
by
  sorry

end sequence_a_n_a5_eq_21_l25_25202


namespace student_adjustment_l25_25652

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem student_adjustment : 
  let front_row_size := 4
  let back_row_size := 8
  let total_students := 12
  let num_to_select := 2
  let ways_to_select := binomial back_row_size num_to_select
  let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
  ways_to_select * ways_to_permute = 840 :=
  by {
    let front_row_size := 4
    let back_row_size := 8
    let total_students := 12
    let num_to_select := 2
    let ways_to_select := binomial back_row_size num_to_select
    let ways_to_permute := permutation (front_row_size + num_to_select) num_to_select
    exact sorry
  }

end student_adjustment_l25_25652


namespace regular_nine_sided_polygon_diagonals_l25_25044

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25044


namespace new_salary_calculation_l25_25535

-- Define the initial conditions of the problem
def current_salary : ℝ := 10000
def percentage_increase : ℝ := 2
def increase := current_salary * (percentage_increase / 100)
def new_salary := current_salary + increase

-- Define the theorem to check the new salary
theorem new_salary_calculation : new_salary = 10200 := by
  -- Lean would check the proof here, but it's being skipped with 'sorry'
  sorry

end new_salary_calculation_l25_25535


namespace limit_of_sequence_l25_25480

noncomputable def limit_problem := 
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |((2 * n - 3) / (n + 2) : ℝ) - 2| < ε

theorem limit_of_sequence : limit_problem :=
sorry

end limit_of_sequence_l25_25480


namespace distribution_of_learning_machines_l25_25715

theorem distribution_of_learning_machines :
  let machines := 6
  let people := 4
  let total_arrangements := 1560
  (∃ distrib : Vector Nat people, 
    sum distrib = machines ∧
    ∀ n, n ∈ distrib → n ≥ 1) → 
  num_possible_arrangements machines people = total_arrangements := 
by
  sorry

end distribution_of_learning_machines_l25_25715


namespace right_triangle_hypotenuse_l25_25533

noncomputable def hypotenuse_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem right_triangle_hypotenuse :
  ∀ (a b : ℝ),
  (1/3) * Real.pi * b^2 * a = 675 * Real.pi →
  (1/3) * Real.pi * a^2 * b = 1215 * Real.pi →
  hypotenuse_length a b = 3 * Real.sqrt 106 :=
  by
  intros a b h1 h2
  sorry

end right_triangle_hypotenuse_l25_25533


namespace tan_sum_pi_over_12_eq_4_l25_25994

theorem tan_sum_pi_over_12_eq_4 :
  tan (π / 12) + tan (5 * π / 12) = 4 := 
by
  have cos_pi_over_12 : cos (π / 12) = (real.sqrt 6 + real.sqrt 2) / 4 := sorry
  have cos_5pi_over_12 : cos (5 * π / 12) = (real.sqrt 6 - real.sqrt 2) / 4 := sorry
  have sin_pi_over_2 : sin (π / 2) = 1 := by
    exact real.sin_pi_div_two
  sorry

end tan_sum_pi_over_12_eq_4_l25_25994


namespace jill_spent_30_percent_on_food_l25_25981

variables (T F : ℝ)

theorem jill_spent_30_percent_on_food
  (h1 : 0.04 * T = 0.016 * T + 0.024 * T)
  (h2 : 0.40 + 0.30 + F = 1) :
  F = 0.30 :=
by 
  sorry

end jill_spent_30_percent_on_food_l25_25981


namespace urn_probability_l25_25253

/-- Given an urn with 2 red and 2 blue balls, and a sequence of 5 operations where a ball is drawn and replaced with another of the same color from an external box, prove that the probability of having 6 red and 6 blue balls after all operations is 6/7. -/
theorem urn_probability:
  let urn := {red := 2, blue := 2}
  let operations := 5
  let final_state := {red := 6, blue := 6}
  let total_balls := 12
  (probability (λ outcome, outcome.red = 6 ∧ outcome.blue = 6 | draw_replace urn operations) = 6 / 7) :=
sorry

end urn_probability_l25_25253


namespace incorrect_reciprocal_quotient_l25_25516

-- Definitions based on problem conditions
def identity_property (x : ℚ) : x * 1 = x := by sorry
def division_property (a b : ℚ) (h : b ≠ 0) : a / b = 0 → a = 0 := by sorry
def additive_inverse_property (x : ℚ) : x * (-1) = -x := by sorry

-- Statement that needs to be proved
theorem incorrect_reciprocal_quotient (a b : ℚ) (h1 : a ≠ 0) (h2 : b = 1 / a) : a / b ≠ 1 :=
by sorry

end incorrect_reciprocal_quotient_l25_25516


namespace find_a_l25_25884

theorem find_a (a : ℝ) (x : ℝ) (h : ∀ (x : ℝ), 2 * x - a ≤ -1 ↔ x ≤ 1) : a = 3 :=
sorry

end find_a_l25_25884


namespace toads_max_l25_25587

theorem toads_max (n : ℕ) (h₁ : n ≥ 3) : 
  ∃ k : ℕ, k = ⌈ (n : ℝ) / 2 ⌉ ∧ ∀ (labels : Fin n → Fin n) (jumps : Fin n → ℕ), 
  (∀ i, jumps (labels i) = labels i) → ¬ ∃ f : Fin k → Fin n, ∀ i₁ i₂, i₁ ≠ i₂ → f i₁ ≠ f i₂ :=
sorry

end toads_max_l25_25587


namespace part1_part2_l25_25696

def f (x : ℝ) : ℝ := abs (x + 2) - 2 * abs (x - 1)

theorem part1 : { x : ℝ | f x ≥ -2 } = { x : ℝ | -2/3 ≤ x ∧ x ≤ 6 } :=
by
  sorry

theorem part2 (a : ℝ) :
  (∀ x ≥ a, f x ≤ x - a) ↔ a ≤ -2 ∨ a ≥ 4 :=
by
  sorry

end part1_part2_l25_25696


namespace triangle_sine_inequality_l25_25845

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2) ≤
  1 + (1 / 2) * Real.cos ((A - B) / 4) ^ 2 :=
by
  sorry

end triangle_sine_inequality_l25_25845


namespace milk_price_increase_day_l25_25492

theorem milk_price_increase_day (total_cost : ℕ) (old_price : ℕ) (new_price : ℕ) (days : ℕ) (x : ℕ)
    (h1 : old_price = 1500)
    (h2 : new_price = 1600)
    (h3 : days = 30)
    (h4 : total_cost = 46200)
    (h5 : (x - 1) * old_price + (days + 1 - x) * new_price = total_cost) :
  x = 19 :=
by
  sorry

end milk_price_increase_day_l25_25492


namespace alex_class_size_l25_25149

theorem alex_class_size 
  (n : ℕ) 
  (h_top : 30 ≤ n)
  (h_bottom : 30 ≤ n) 
  (h_better : n - 30 > 0)
  (h_worse : n - 30 > 0)
  : n = 59 := 
sorry

end alex_class_size_l25_25149


namespace regular_nonagon_diagonals_correct_l25_25114

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l25_25114


namespace probability_of_2_points_for_question_11_probability_of_total_7_points_l25_25980

-- Define the probabilities for selecting options
def prob_select_one := (1 : ℚ) / 3
def prob_select_two := (1 : ℚ) / 3
def prob_select_three := (1 : ℚ) / 3

-- Define the events and their probabilities
def prob_correct_selection_one := (1 : ℚ) / 2  -- Given one option is selected, the probability it is correct
def prob_correct_selection_two := (1 : ℚ) / 6  -- Probability of selecting the two correct options out of possible pairs

-- Part 1: Probability of getting 2 points for question 11
theorem probability_of_2_points_for_question_11 : 
  (prob_select_one * prob_correct_selection_one) = (1 : ℚ) / 6 := 
sorry

-- Part 2: Probability of scoring a total of 7 points for questions 11 and 12
theorem probability_of_total_7_points : 
  2 * ((prob_select_one * prob_correct_selection_one) * (prob_select_two * prob_correct_selection_two)) = (1 : ℚ) / 54 := 
sorry

end probability_of_2_points_for_question_11_probability_of_total_7_points_l25_25980


namespace triangle_inequality_range_isosceles_triangle_perimeter_l25_25965

-- Define the parameters for the triangle
variables (AB BC AC a : ℝ)
variables (h_AB : AB = 8) (h_BC : BC = 2 * a + 2) (h_AC : AC = 22)

-- Define the lean proof problem for the given conditions
theorem triangle_inequality_range (h_triangle : AB = 8 ∧ BC = 2 * a + 2 ∧ AC = 22) :
  6 < a ∧ a < 14 := sorry

-- Define the isosceles condition and perimeter calculation
theorem isosceles_triangle_perimeter (h_isosceles : BC = AC) :
  perimeter = 52 := sorry

end triangle_inequality_range_isosceles_triangle_perimeter_l25_25965


namespace bruce_paid_correct_amount_l25_25542

-- Define the conditions
def kg_grapes : ℕ := 8
def cost_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 8
def cost_per_kg_mangoes : ℕ := 55

-- Calculate partial costs
def cost_grapes := kg_grapes * cost_per_kg_grapes
def cost_mangoes := kg_mangoes * cost_per_kg_mangoes
def total_paid := cost_grapes + cost_mangoes

-- The theorem to prove
theorem bruce_paid_correct_amount : total_paid = 1000 := 
by 
  -- Merge several logical steps into one
  -- sorry can be used for incomplete proof
  sorry

end bruce_paid_correct_amount_l25_25542


namespace expression_evaluate_l25_25269

theorem expression_evaluate (a b c : ℤ) (h1 : b = a + 2) (h2 : c = b - 10) (ha : a = 4)
(h3 : a ≠ -1) (h4 : b ≠ 2) (h5 : b ≠ -4) (h6 : c ≠ -6) : (a + 2) / (a + 1) * (b - 1) / (b - 2) * (c + 8) / (c + 6) = 3 :=
by
  sorry

end expression_evaluate_l25_25269


namespace real_solution_count_eq_14_l25_25276

theorem real_solution_count_eq_14 :
  { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.finite ∧
  finset.card { x : ℝ | -50 ≤ x ∧ x ≤ 50 ∧ x / 50 = real.cos x }.to_finset = 14 :=
sorry

end real_solution_count_eq_14_l25_25276


namespace calculate_distance_l25_25891

theorem calculate_distance (t : ℕ) (h_t : t = 4) : 5 * t^2 + 2 * t = 88 :=
by
  rw [h_t]
  norm_num

end calculate_distance_l25_25891


namespace common_chord_equation_l25_25301

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_equation :
  ∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧
                     ∀ (x y : ℝ), (x - 2*y + 4 = 0) ↔ ((x, y) = A ∨ (x, y) = B) :=
by
  sorry

end common_chord_equation_l25_25301


namespace nine_sided_polygon_diagonals_l25_25119

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25119


namespace number_of_diagonals_l25_25085

-- Define the number of vertices and parameter of combination
def num_vertices : ℕ := 9
def num_edges : ℕ := nat.choose num_vertices 2
def num_sides : ℕ := num_vertices

-- Define the theorem to prove the number of diagonals is 27
theorem number_of_diagonals (n : ℕ) (h : n = 9) : (num_edges - num_sides) = 27 := by
  -- Unfold the definitions
  unfold num_edges num_sides
  -- Apply the substitution for specific values
  rw h
  -- Simplify the combination and subtraction
  sorry

end number_of_diagonals_l25_25085


namespace regular_nine_sided_polygon_diagonals_l25_25096

theorem regular_nine_sided_polygon_diagonals : 
  ∃ d : ℕ, d = 27 ∧ 
  let n := 9 in 
  let combinations := n * (n - 1) / 2 in 
  d = combinations - n :=
by
  sorry

end regular_nine_sided_polygon_diagonals_l25_25096


namespace nine_sided_polygon_diagonals_l25_25126

def number_of_diagonals (n : ℕ) : ℕ := nat.choose n 2 - n

theorem nine_sided_polygon_diagonals :
  number_of_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l25_25126


namespace find_valid_tax_range_l25_25957

noncomputable def valid_tax_range (t : ℝ) : Prop :=
  let initial_consumption := 200000
  let price_per_cubic_meter := 240
  let consumption_reduction := 2.5 * t * 10^4
  let tax_revenue := (initial_consumption - consumption_reduction) * price_per_cubic_meter * (t / 100)
  tax_revenue >= 900000

theorem find_valid_tax_range (t : ℝ) : 3 ≤ t ∧ t ≤ 5 ↔ valid_tax_range t :=
sorry

end find_valid_tax_range_l25_25957


namespace find_range_a_l25_25448

-- Define the parabola equation y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line equation y = (√3/3) * (x - a)
def line (x y a : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - a)

-- Define the focus of the parabola
def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Define the condition that F is outside the circle with diameter CD
def F_outside_circle_CD (x1 y1 x2 y2 a : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 > 0

-- Define the parabola-line intersection points and the related Vieta's formulas
def intersection_points (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = 2 * a + 12 ∧ x1 * x2 = a^2

-- Define the final condition for a
def range_a (a : ℝ) : Prop :=
  -3 < a ∧ a < -2 * Real.sqrt 5 + 3

-- Main theorem statement
theorem find_range_a (a : ℝ) (hneg : a < 0)
  (x1 x2 y1 y2 : ℝ)
  (hparabola1 : parabola x1 y1)
  (hparabola2 : parabola x2 y2)
  (hline1 : line x1 y1 a)
  (hline2 : line x2 y2 a)
  (hfocus : focus 1 0)
  (hF_out : F_outside_circle_CD x1 y1 x2 y2 a)
  (hintersect : intersection_points a x1 x2) :
  range_a a := 
sorry

end find_range_a_l25_25448


namespace find_solutions_l25_25549

theorem find_solutions (x y : ℕ) : 33 ^ x + 31 = 2 ^ y → (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := 
by
  sorry

end find_solutions_l25_25549


namespace regular_nine_sided_polygon_has_27_diagonals_l25_25018

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end regular_nine_sided_polygon_has_27_diagonals_l25_25018


namespace range_of_t_l25_25447

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + t ≤ 0 ∧ x ≤ t) ↔ (0 ≤ t ∧ t ≤ 9 / 4) := 
sorry

end range_of_t_l25_25447


namespace setB_can_form_triangle_l25_25658

theorem setB_can_form_triangle : 
  let a := 8
  let b := 6
  let c := 4
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  let a := 8
  let b := 6
  let c := 4
  have h1 : a + b > c := by sorry
  have h2 : a + c > b := by sorry
  have h3 : b + c > a := by sorry
  exact ⟨h1, h2, h3⟩

end setB_can_form_triangle_l25_25658


namespace diagonals_in_nine_sided_polygon_l25_25079

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l25_25079


namespace somu_present_age_l25_25750

variable (S F : ℕ)

-- Conditions from the problem
def condition1 : Prop := S = F / 3
def condition2 : Prop := S - 10 = (F - 10) / 5

-- The statement we need to prove
theorem somu_present_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 20 := 
by sorry

end somu_present_age_l25_25750


namespace product_of_hypotenuse_segments_eq_area_l25_25874

theorem product_of_hypotenuse_segments_eq_area (x y c t : ℝ) : 
  -- Conditions
  (c = x + y) → 
  (t = x * y) →
  -- Conclusion
  x * y = t :=
by
  intros
  sorry

end product_of_hypotenuse_segments_eq_area_l25_25874


namespace diagonals_in_nonagon_l25_25130

theorem diagonals_in_nonagon : 
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  diagonals = 27 :=
by
  let vertices := 9
  let total_segments := vertices * (vertices - 1) / 2
  let sides := vertices
  let diagonals := total_segments - sides
  have h : total_segments = 36 := by sorry
  have h2 : sides = 9 := by sorry
  have h3 : diagonals = total_segments - sides := by sorry
  show diagonals = 27 from by
    rw [h, h2, h3]
    exact rfl

end diagonals_in_nonagon_l25_25130


namespace find_a_and_b_l25_25312

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l25_25312


namespace hats_cost_l25_25164

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l25_25164


namespace sock_pairs_l25_25959

open Nat

theorem sock_pairs (r g y : ℕ) (hr : r = 5) (hg : g = 6) (hy : y = 4) :
  (choose r 2) + (choose g 2) + (choose y 2) = 31 :=
by
  rw [hr, hg, hy]
  norm_num
  sorry

end sock_pairs_l25_25959


namespace original_price_calculation_l25_25410

variable (P : ℝ)
variable (selling_price : ℝ := 1040)
variable (loss_percentage : ℝ := 20)

theorem original_price_calculation :
  P = 1300 :=
by
  have sell_percent := 100 - loss_percentage
  have SP_eq := selling_price = (sell_percent / 100) * P
  sorry

end original_price_calculation_l25_25410


namespace simplified_fraction_sum_l25_25871

theorem simplified_fraction_sum (n d : ℕ) (h_n : n = 144) (h_d : d = 256) : (9 + 16 = 25) := by
  have h1 : n = 2^4 * 3^2 := by sorry
  have h2 : d = 2^8 := by sorry
  have h3 : (n / gcd n d) = 9 := by sorry
  have h4 : (d / gcd n d) = 16 := by sorry
  exact rfl

end simplified_fraction_sum_l25_25871


namespace part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l25_25819

-- Conditions
def quadratic (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * a * x + 2 * a
def point_A (a : ℝ) : ℝ × ℝ := (-1, quadratic a (-1))
def point_B (a : ℝ) : ℝ × ℝ := (3, quadratic a 3)
def line_EF (a : ℝ) : ℝ × ℝ × ℝ × ℝ := ((a - 1), -1, (2 * a + 3), -1)

-- Statements based on solution
theorem part_1 (a : ℝ) :
  (quadratic a (-1)) = -1 := sorry

theorem part_2_max_min (a : ℝ) : 
  a = 1 → 
  (∀ x, -2 ≤ x ∧ x ≤ 3 → 
    (quadratic 1 1 = 3 ∧ 
     quadratic 1 (-2) = -6 ∧ 
     quadratic 1 3 = -1)) := sorry

theorem part_3_length_AC (a : ℝ) (h : a > -1) :
  abs ((2 * a + 1) - (-1)) = abs ((2 * a + 2)) := sorry

theorem part_4_range_a (a : ℝ) : 
  quadratic a (a-1) = -1 ∧ quadratic a (2 * a + 3) = -1 → 
  a ∈ ({-2, -1} ∪ {b : ℝ | b ≥ 0}) := sorry

end part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l25_25819


namespace nine_sided_polygon_diagonals_l25_25104

def num_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_of_diagonals 9 = 27 :=
by
  -- The formula for the number of diagonals in a polygon with n sides is:
  -- num_of_diagonals(n) = (n * (n - 3)) / 2
  
  -- For a nine-sided polygon:
  -- num_of_diagonals(9) = 9 * (9 - 3) / 2
  --                      = 9 * 6 / 2
  --                      = 54 / 2
  --                      = 27
  sorry

end nine_sided_polygon_diagonals_l25_25104


namespace diagonals_in_nine_sided_polygon_l25_25062

-- Given a regular polygon with 9 sides
def regular_polygon_sides : ℕ := 9

-- To find the number of diagonals in a polygon, we use the formula
noncomputable def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- We need to prove this particular instance where the number of sides is 9
theorem diagonals_in_nine_sided_polygon : number_of_diagonals regular_polygon_sides = 27 := 
by sorry

end diagonals_in_nine_sided_polygon_l25_25062


namespace tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l25_25997

-- Definitions for given conditions
def cos_pi_over_12 : ℝ := (Real.sqrt 6 + Real.sqrt 2) / 4
def cos_5pi_over_12 : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

-- The theorem to be proved
theorem tan_pi_over_12_plus_tan_5pi_over_12_eq_4 : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 :=
by sorry

end tan_pi_over_12_plus_tan_5pi_over_12_eq_4_l25_25997


namespace car_dealership_sales_l25_25840

theorem car_dealership_sales (trucks_ratio suvs_ratio trucks_expected suvs_expected : ℕ)
  (h_ratio : trucks_ratio = 5 ∧ suvs_ratio = 8)
  (h_expected : trucks_expected = 35 ∧ suvs_expected = 56) :
  (trucks_ratio : ℚ) / suvs_ratio = (trucks_expected : ℚ) / suvs_expected :=
by
  sorry

end car_dealership_sales_l25_25840


namespace volume_of_truncated_triangular_pyramid_l25_25873

variable {a b H α : ℝ} (h1 : H = Real.sqrt (a * b))

theorem volume_of_truncated_triangular_pyramid
  (h2 : H = Real.sqrt (a * b))
  (h3 : 0 < a)
  (h4 : 0 < b)
  (h5 : 0 < H)
  (h6 : 0 < α) :
  (volume : ℝ) = H^3 * Real.sqrt 3 / (4 * (Real.sin α)^2) := sorry

end volume_of_truncated_triangular_pyramid_l25_25873


namespace parabola_standard_eq_l25_25579

theorem parabola_standard_eq (h : ∃ (x y : ℝ), x - 2 * y - 4 = 0 ∧ (
                         (y = 0 ∧ x = 4 ∧ y^2 = 16 * x) ∨ 
                         (x = 0 ∧ y = -2 ∧ x^2 = -8 * y))
                         ) :
                         (y^2 = 16 * x) ∨ (x^2 = -8 * y) :=
by 
  sorry

end parabola_standard_eq_l25_25579


namespace number_of_integers_l25_25698

theorem number_of_integers (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2020) (h3 : ∃ k : ℕ, n^n = k^2) : n = 1032 :=
sorry

end number_of_integers_l25_25698


namespace solution_set_ln_inequality_l25_25621

noncomputable def f (x : ℝ) := Real.cos x - 4 * x^2

theorem solution_set_ln_inequality :
  {x : ℝ | 0 < x ∧ x < Real.exp (-Real.pi / 2)} ∪ {x : ℝ | x > Real.exp (Real.pi / 2)} =
  {x : ℝ | f (Real.log x) + Real.pi^2 > 0} :=
by
  sorry

end solution_set_ln_inequality_l25_25621


namespace total_land_l25_25163

variable (land_house : ℕ) (land_expansion : ℕ) (land_cattle : ℕ) (land_crop : ℕ)

theorem total_land (h1 : land_house = 25) 
                   (h2 : land_expansion = 15) 
                   (h3 : land_cattle = 40) 
                   (h4 : land_crop = 70) : 
  land_house + land_expansion + land_cattle + land_crop = 150 := 
by 
  sorry

end total_land_l25_25163


namespace fraction_eq_zero_iff_l25_25583

theorem fraction_eq_zero_iff (x : ℝ) : (3 * x - 1) / (x ^ 2 + 1) = 0 ↔ x = 1 / 3 := by
  sorry

end fraction_eq_zero_iff_l25_25583


namespace percentage_less_than_l25_25412

variable (x y z n : ℝ)
variable (hx : x = 8 * y)
variable (hy : y = 2 * |z - n|)
variable (hz : z = 1.1 * n)

theorem percentage_less_than (hx : x = 8 * y) (hy : y = 2 * |z - n|) (hz : z = 1.1 * n) :
  ((x - y) / x) * 100 = 87.5 := sorry

end percentage_less_than_l25_25412


namespace find_acid_percentage_l25_25137

theorem find_acid_percentage (P : ℕ) (x : ℕ) (h1 : 4 + x = 20) 
  (h2 : x = 20 - 4) 
  (h3 : (P : ℝ)/100 * 4 + 0.75 * 16 = 0.72 * 20) : P = 60 :=
by
  sorry

end find_acid_percentage_l25_25137


namespace value_of_three_inch_cube_l25_25915

theorem value_of_three_inch_cube (value_two_inch: ℝ) (volume_two_inch: ℝ) (volume_three_inch: ℝ) (cost_two_inch: ℝ):
  value_two_inch = cost_two_inch * ((volume_three_inch / volume_two_inch): ℝ) := 
by
  have volume_two_inch := 2^3 -- Volume of two-inch cube
  have volume_three_inch := 3^3 -- Volume of three-inch cube
  let volume_ratio := (volume_three_inch / volume_two_inch: ℝ)
  have := cost_two_inch * volume_ratio
  norm_num
  sorry

end value_of_three_inch_cube_l25_25915


namespace handshake_problem_l25_25504

theorem handshake_problem :
  ∃ (a b : ℕ), a + b = 20 ∧ (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 ∧ a * b = 84 :=
by
  sorry

end handshake_problem_l25_25504


namespace find_value_of_z_l25_25824

open Complex

-- Define the given complex number z and imaginary unit i
def z : ℂ := sorry
def i : ℂ := Complex.I

-- Given condition
axiom condition : z / (1 - i) = i ^ 2019

-- Proof that z equals -1 - i
theorem find_value_of_z : z = -1 - i :=
by
  sorry

end find_value_of_z_l25_25824


namespace volume_of_sphere_l25_25238

theorem volume_of_sphere (V : ℝ) (r : ℝ) : r = 1 / 3 → (2 * r) = (16 / 9 * V)^(1/3) → V = 1 / 6 :=
by
  intro h_radius h_diameter
  sorry

end volume_of_sphere_l25_25238


namespace solution_set_l25_25580

noncomputable def f : ℝ → ℝ := sorry

axiom f_cond1 : ∀ x : ℝ, f x + deriv f x > 1
axiom f_cond2 : f 0 = 4

theorem solution_set (x : ℝ) : e^x * f x > e^x + 3 ↔ x > 0 :=
by sorry

end solution_set_l25_25580


namespace diagonals_in_nine_sided_polygon_l25_25001

theorem diagonals_in_nine_sided_polygon : ∀ (n : ℕ), n = 9 → (n * (n - 3) / 2) = 27 :=
by
  intro n hn
  rw hn
  norm_num
  sorry

end diagonals_in_nine_sided_polygon_l25_25001


namespace part1_part2_part3_l25_25686

-- Define the sequence and conditions
variable {a : ℕ → ℕ}
axiom sequence_def (n : ℕ) : a n = max (a (n + 1)) (a (n + 2)) - min (a (n + 1)) (a (n + 2))

-- Part (1)
axiom a1_def : a 1 = 1
axiom a2_def : a 2 = 2
theorem part1 : a 4 = 1 ∨ a 4 = 3 ∨ a 4 = 5 :=
  sorry

-- Part (2)
axiom has_max (M : ℕ) : ∀ n, a n ≤ M
theorem part2 : ∃ n, a n = 0 :=
  sorry

-- Part (3)
axiom positive_seq : ∀ n, a n > 0
theorem part3 : ¬∃ M : ℝ, ∀ n, a n ≤ M :=
  sorry

end part1_part2_part3_l25_25686


namespace nine_sided_polygon_diagonals_l25_25047

theorem nine_sided_polygon_diagonals : 
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let sides := n in
  let diagonals := total_pairs - sides in
  diagonals = 27 :=
by
  let n := 9
  let total_pairs := Nat.choose n 2
  let sides := n
  let diagonals := total_pairs - sides
  have : total_pairs = 36 := by sorry
  have : sides = 9 := by sorry
  have : diagonals = 36 - 9 := by sorry
  exact Eq.trans this rfl

end nine_sided_polygon_diagonals_l25_25047


namespace nine_sided_polygon_diagonals_count_l25_25028

theorem nine_sided_polygon_diagonals_count :
  ∃ (n : ℕ), n = 9 → (nat.choose n 2 - n = 36) :=
by
  sorry

end nine_sided_polygon_diagonals_count_l25_25028


namespace new_fish_received_l25_25972

def initial_fish := 14
def added_fish := 2
def eaten_fish := 6
def final_fish := 11

def current_fish := initial_fish + added_fish - eaten_fish
def returned_fish := 2
def exchanged_fish := final_fish - current_fish

theorem new_fish_received : exchanged_fish = 1 := by
  sorry

end new_fish_received_l25_25972


namespace nine_sided_polygon_diagonals_l25_25037

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25037


namespace odd_function_characterization_l25_25305

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l25_25305


namespace nine_sided_polygon_diagonals_l25_25034

def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by
  -- Place proof here
  sorry

end nine_sided_polygon_diagonals_l25_25034


namespace conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l25_25440

variable {r a b x1 y1 x2 y2 : ℝ} -- variables used in the problem

-- conditions
def circle1 : x1^2 + y1^2 = r^2 := sorry -- Circle C1 equation
def circle2 : (x1 + a)^2 + (y1 + b)^2 = r^2 := sorry -- Circle C2 equation
def r_positive : r > 0 := sorry -- r > 0
def not_both_zero : ¬ (a = 0 ∧ b = 0) := sorry -- a, b are not both zero
def distinct_points : x1 ≠ x2 ∧ y1 ≠ y2 := sorry -- A(x1, y1) and B(x2, y2) are distinct

-- Proofs to be provided for each of the conclusions
theorem conclusion_A : 2 * a * x1 + 2 * b * y1 + a^2 + b^2 = 0 := sorry
theorem conclusion_B : a * (x1 - x2) + b * (y1 - y2) = 0 := sorry
theorem conclusion_C1 : x1 + x2 = -a := sorry
theorem conclusion_C2 : y1 + y2 = -b := sorry

end conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l25_25440


namespace incorrect_proposition3_l25_25444

open Real

-- Definitions from the problem
def prop1 (x : ℝ) := 2 * sin (2 * x - π / 3) = 2
def prop2 (x y : ℝ) := tan x + tan (π - x) = 0
def prop3 (x1 x2 : ℝ) (k : ℤ) := x1 - x2 = (k : ℝ) * π → k % 2 = 1
def prop4 (x : ℝ) := cos x ^ 2 + sin x >= -1

-- Incorrect proposition proof
theorem incorrect_proposition3 (x1 x2 : ℝ) (k : ℤ) :
  sin (2 * x1 - π / 4) = 0 →
  sin (2 * x2 - π / 4) = 0 →
  x1 - x2 ≠ (k : ℝ) * π := sorry

end incorrect_proposition3_l25_25444


namespace distance_center_to_plane_l25_25458

theorem distance_center_to_plane (r : ℝ) (a b : ℝ) (h : a ^ 2 + b ^ 2 = 10 ^ 2) (d : ℝ) : 
  r = 13 → a = 6 → b = 8 → d = 12 := 
by 
  sorry

end distance_center_to_plane_l25_25458


namespace remainder_of_sum_div_17_l25_25281

-- Definitions based on the conditions from the problem
def numbers : List ℕ := [82, 83, 84, 85, 86, 87, 88, 89]
def divisor : ℕ := 17

-- The theorem statement proving the result
theorem remainder_of_sum_div_17 : List.sum numbers % divisor = 0 := by
  sorry

end remainder_of_sum_div_17_l25_25281


namespace min_value_expression_l25_25702

theorem min_value_expression (x : ℝ) (h : x > 1) : x + 9 / x - 2 ≥ 4 :=
sorry

end min_value_expression_l25_25702


namespace smallest_b_theorem_l25_25470

open Real

noncomputable def smallest_b (a b c: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) : Prop :=
  b = 5

theorem smallest_b_theorem (a b c: ℝ) (r: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) :
  smallest_b a b c h1 h2 h3 h4 :=
by {
  sorry
}

end smallest_b_theorem_l25_25470


namespace dogwood_trees_l25_25653

/-- There are 7 dogwood trees currently in the park. 
Park workers will plant 5 dogwood trees today. 
The park will have 16 dogwood trees when the workers are finished.
Prove that 4 dogwood trees will be planted tomorrow. --/
theorem dogwood_trees (x : ℕ) : 7 + 5 + x = 16 → x = 4 :=
by
  sorry

end dogwood_trees_l25_25653


namespace regular_nine_sided_polygon_diagonals_l25_25045

theorem regular_nine_sided_polygon_diagonals : ∀ (P : Type) [Fintype P] [Fintype (finset.univ : finset P)],
  (P → Prop)
  (regular_polygon : ∀ (x y : P), x ≠ y → Prop)
  (nine_sided : Fintype.card P = 9) :
  finsupp.sum (λ (xy : P × P), if xy.1 ≠ xy.2 then 1 else 0) = 27 :=
sorry

end regular_nine_sided_polygon_diagonals_l25_25045


namespace peaches_eaten_correct_l25_25777

-- Given conditions
def total_peaches : ℕ := 18
def initial_ripe_peaches : ℕ := 4
def peaches_ripen_per_day : ℕ := 2
def days_passed : ℕ := 5
def ripe_unripe_difference : ℕ := 7

-- Definitions derived from conditions
def ripe_peaches_after_days := initial_ripe_peaches + peaches_ripen_per_day * days_passed
def unripe_peaches_initial := total_peaches - initial_ripe_peaches
def unripe_peaches_after_days := unripe_peaches_initial - peaches_ripen_per_day * days_passed
def actual_ripe_peaches_needed := unripe_peaches_after_days + ripe_unripe_difference
def peaches_eaten := ripe_peaches_after_days - actual_ripe_peaches_needed

-- Prove that the number of peaches eaten is equal to 3
theorem peaches_eaten_correct : peaches_eaten = 3 := by
  sorry

end peaches_eaten_correct_l25_25777


namespace base_seven_to_ten_l25_25392

theorem base_seven_to_ten :
  let a := 54321
  let b := 5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0
  a = b :=
by
  unfold a b
  exact rfl

end base_seven_to_ten_l25_25392


namespace sum_of_max_values_l25_25445

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem sum_of_max_values : (f π + f (3 * π)) = (Real.exp π + Real.exp (3 * π)) := 
by sorry

end sum_of_max_values_l25_25445


namespace cone_cylinder_volume_ratio_l25_25806

theorem cone_cylinder_volume_ratio (h r : ℝ) (hc_pos : h > 0) (r_pos : r > 0) :
  let V_cylinder := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * (3 / 4 * h)
  (V_cone / V_cylinder) = 1 / 4 := 
by 
  sorry

end cone_cylinder_volume_ratio_l25_25806


namespace find_a_and_b_to_make_f_odd_l25_25345

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := ln (abs (a + 1 / (1 - x))) + b

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

theorem find_a_and_b_to_make_f_odd :
  (a b : ℝ) (h : a = -1/2 ∧ b = ln 2) :
  is_odd_function (f a b) := 
by
  sorry

end find_a_and_b_to_make_f_odd_l25_25345
