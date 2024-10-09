import Mathlib

namespace shape_is_plane_l240_24059

noncomputable
def cylindrical_coordinates_shape (r θ z c : ℝ) := θ = 2 * c

theorem shape_is_plane (c : ℝ) : 
  ∀ (r : ℝ) (θ : ℝ) (z : ℝ), cylindrical_coordinates_shape r θ z c → (θ = 2 * c) :=
by
  sorry

end shape_is_plane_l240_24059


namespace intersection_A_B_l240_24057

def A (x : ℝ) : Prop := (2 * x - 1 > 0)
def B (x : ℝ) : Prop := (x * (x - 2) < 0)

theorem intersection_A_B :
  {x : ℝ | A x ∧ B x} = {x : ℝ | 1 / 2 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l240_24057


namespace basketball_players_taking_chemistry_l240_24081

variable (total_players : ℕ) (taking_biology : ℕ) (taking_both : ℕ)

theorem basketball_players_taking_chemistry (h1 : total_players = 20) 
                                           (h2 : taking_biology = 8) 
                                           (h3 : taking_both = 4) 
                                           (h4 : ∀p, p ≤ total_players) :
  total_players - taking_biology + taking_both = 16 :=
by sorry

end basketball_players_taking_chemistry_l240_24081


namespace ned_initially_had_games_l240_24047

variable (G : ℕ)

theorem ned_initially_had_games (h1 : (3 / 4) * (2 / 3) * G = 6) : G = 12 := by
  sorry

end ned_initially_had_games_l240_24047


namespace transport_cost_l240_24038

theorem transport_cost (mass_g: ℕ) (cost_per_kg : ℕ) (mass_kg : ℝ) 
  (h1 : mass_g = 300) (h2 : mass_kg = (mass_g : ℝ) / 1000) 
  (h3: cost_per_kg = 18000)
  : mass_kg * cost_per_kg = 5400 := by
  sorry

end transport_cost_l240_24038


namespace initial_ratio_milk_water_l240_24088

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 60) 
  (h2 : ∀ k, k = M → M * 2 = W + 60) : (M:ℚ) / (W:ℚ) = 4 / 1 :=
by
  sorry

end initial_ratio_milk_water_l240_24088


namespace fraction_of_managers_l240_24025

theorem fraction_of_managers (female_managers : ℕ) (total_female_employees : ℕ)
  (total_employees: ℕ) (male_employees: ℕ) (f: ℝ) :
  female_managers = 200 →
  total_female_employees = 500 →
  total_employees = total_female_employees + male_employees →
  (f * total_employees) = female_managers + (f * male_employees) →
  f = 0.4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_managers_l240_24025


namespace simplify_144_over_1296_times_36_l240_24087

theorem simplify_144_over_1296_times_36 :
  (144 / 1296) * 36 = 4 :=
by
  sorry

end simplify_144_over_1296_times_36_l240_24087


namespace convex_polyhedron_theorems_l240_24013

-- Definitions for convex polyhedron and symmetric properties
structure ConvexSymmetricPolyhedron (α : Type*) :=
  (isConvex : Bool)
  (isCentrallySymmetric : Bool)
  (crossSection : α → α → α)
  (center : α)

-- Definitions for proofs required
def largest_cross_section_area
  (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ∀ (p : ℝ), P.crossSection p P.center ≤ P.crossSection P.center P.center

def largest_radius_circle (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ¬∀ (p : ℝ), P.crossSection p P.center = P.crossSection P.center P.center

-- The theorem combining both statements
theorem convex_polyhedron_theorems
  (P : ConvexSymmetricPolyhedron ℝ) :
  P.isConvex = true ∧ 
  P.isCentrallySymmetric = true →
  (largest_cross_section_area P) ∧ (largest_radius_circle P) :=
by 
  sorry

end convex_polyhedron_theorems_l240_24013


namespace birds_left_after_a_week_l240_24097

def initial_chickens := 300
def initial_turkeys := 200
def initial_guinea_fowls := 80
def daily_chicken_loss := 20
def daily_turkey_loss := 8
def daily_guinea_fowl_loss := 5
def days_in_a_week := 7

def remaining_chickens := initial_chickens - daily_chicken_loss * days_in_a_week
def remaining_turkeys := initial_turkeys - daily_turkey_loss * days_in_a_week
def remaining_guinea_fowls := initial_guinea_fowls - daily_guinea_fowl_loss * days_in_a_week

def total_remaining_birds := remaining_chickens + remaining_turkeys + remaining_guinea_fowls

theorem birds_left_after_a_week : total_remaining_birds = 349 := by
  sorry

end birds_left_after_a_week_l240_24097


namespace number_of_terms_in_arithmetic_sequence_l240_24078

-- Definitions and conditions
def a : ℤ := -58  -- First term
def d : ℤ := 7   -- Common difference
def l : ℤ := 78  -- Last term

-- Statement of the problem
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 20 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l240_24078


namespace card_giving_ratio_l240_24035

theorem card_giving_ratio (initial_cards cards_to_Bob cards_left : ℕ) 
  (h1 : initial_cards = 18) 
  (h2 : cards_to_Bob = 3)
  (h3 : cards_left = 9) : 
  (initial_cards - cards_left - cards_to_Bob) / gcd (initial_cards - cards_left - cards_to_Bob) cards_to_Bob = 2 / 1 :=
by sorry

end card_giving_ratio_l240_24035


namespace twenty_four_game_l240_24028

-- Definition of the cards' values
def card2 : ℕ := 2
def card5 : ℕ := 5
def cardJ : ℕ := 11
def cardQ : ℕ := 12

-- Theorem stating the proof
theorem twenty_four_game : card2 * (cardJ - card5) + cardQ = 24 :=
by
  sorry

end twenty_four_game_l240_24028


namespace inequality_problem_l240_24085

theorem inequality_problem (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by {
  sorry
}

end inequality_problem_l240_24085


namespace even_function_a_is_0_l240_24074

def f (a : ℝ) (x : ℝ) : ℝ := (a+1) * x^2 + 3 * a * x + 1

theorem even_function_a_is_0 (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end even_function_a_is_0_l240_24074


namespace person_picking_number_who_announced_6_is_1_l240_24092

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

end person_picking_number_who_announced_6_is_1_l240_24092


namespace cost_price_of_one_toy_l240_24002

-- Definitions translating the conditions into Lean
def total_revenue (toys_sold : ℕ) (price_per_toy : ℕ) : ℕ := toys_sold * price_per_toy
def gain (cost_per_toy : ℕ) (toys_gained : ℕ) : ℕ := cost_per_toy * toys_gained

-- Given the conditions in the problem
def total_cost_price_of_sold_toys := 18 * (1300 : ℕ)
def gain_from_sale := 3 * (1300 : ℕ)
def selling_price := total_cost_price_of_sold_toys + gain_from_sale

-- The target theorem we want to prove
theorem cost_price_of_one_toy : (selling_price = 27300) → (1300 = 27300 / 21) :=
by
  intro h
  sorry

end cost_price_of_one_toy_l240_24002


namespace sufficient_but_not_necessary_l240_24022

theorem sufficient_but_not_necessary (a b : ℝ) : (a > b ∧ b > 0) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > b ∧ b > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l240_24022


namespace find_K_l240_24015

noncomputable def cylinder_paint (r h : ℝ) : ℝ := 2 * Real.pi * r * h
noncomputable def cube_surface_area (s : ℝ) : ℝ := 6 * s^2
noncomputable def cube_volume (s : ℝ) : ℝ := s^3

theorem find_K :
  (cylinder_paint 3 4 = 24 * Real.pi) →
  (∃ s, cube_surface_area s = 24 * Real.pi ∧ cube_volume s = 48 / Real.sqrt K) →
  K = 36 / Real.pi^3 :=
by
  sorry

end find_K_l240_24015


namespace sin_315_degree_l240_24027

-- Definitions based on given conditions
def angle : ℝ := 315
def unit_circle_radius : ℝ := 1

-- Theorem statement to prove
theorem sin_315_degree : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
sorry

end sin_315_degree_l240_24027


namespace total_embroidery_time_l240_24086

-- Defining the constants as given in the problem
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1 -- Implicitly from the problem statement

-- Total time calculation as a Lean theorem
theorem total_embroidery_time : 
  (stitches_per_godzilla * num_godzillas + 
   stitches_per_unicorn * num_unicorns + 
   stitches_per_flower * num_flowers) / stitches_per_minute = 1085 := 
by
  sorry

end total_embroidery_time_l240_24086


namespace garden_perimeter_l240_24036

theorem garden_perimeter
  (a b : ℝ)
  (h1 : a^2 + b^2 = 1156)
  (h2 : a * b = 240) :
  2 * (a + b) = 80 :=
sorry

end garden_perimeter_l240_24036


namespace number_of_pages_500_l240_24046

-- Define the conditions as separate constants
def cost_per_page : ℕ := 3 -- cents
def total_cents : ℕ := 1500 

-- Define the number of pages calculation
noncomputable def number_of_pages := total_cents / cost_per_page

-- Statement we want to prove
theorem number_of_pages_500 : number_of_pages = 500 :=
sorry

end number_of_pages_500_l240_24046


namespace lineD_intersects_line1_l240_24066

-- Define the lines based on the conditions
def line1 (x y : ℝ) := x + y - 1 = 0
def lineA (x y : ℝ) := 2 * x + 2 * y = 6
def lineB (x y : ℝ) := x + y = 0
def lineC (x y : ℝ) := y = -x - 3
def lineD (x y : ℝ) := y = x - 1

-- Define the statement that line D intersects with line1
theorem lineD_intersects_line1 : ∃ (x y : ℝ), line1 x y ∧ lineD x y :=
by
  sorry

end lineD_intersects_line1_l240_24066


namespace john_took_11_more_l240_24070

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end john_took_11_more_l240_24070


namespace division_remainder_l240_24045

theorem division_remainder : 1234567 % 112 = 0 := 
by 
  sorry

end division_remainder_l240_24045


namespace m_range_l240_24054

theorem m_range (m : ℝ) :
  (∀ x : ℝ, 1 < x → 2 * x + m + 2 / (x - 1) > 0) ↔ m > -6 :=
by
  -- The proof will be provided later
  sorry

end m_range_l240_24054


namespace binomial_coefficient_30_3_l240_24055

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_coefficient_30_3_l240_24055


namespace math_problem_l240_24050

theorem math_problem
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a^3 + b^3 = 2) :
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := 
by
  sorry

end math_problem_l240_24050


namespace intersection_of_M_N_equals_0_1_open_interval_l240_24093

def M : Set ℝ := { x | x ≥ 0 }
def N : Set ℝ := { x | x^2 < 1 }

theorem intersection_of_M_N_equals_0_1_open_interval :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } := 
sorry

end intersection_of_M_N_equals_0_1_open_interval_l240_24093


namespace area_of_rectangle_l240_24079

theorem area_of_rectangle (length : ℝ) (width : ℝ) (h_length : length = 47.3) (h_width : width = 24) : 
  length * width = 1135.2 := 
by 
  sorry

end area_of_rectangle_l240_24079


namespace expand_expression_l240_24089

theorem expand_expression (x y : ℤ) : (x + 7) * (3 * y + 8) = 3 * x * y + 8 * x + 21 * y + 56 :=
by
  sorry

end expand_expression_l240_24089


namespace apples_left_total_l240_24005

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end apples_left_total_l240_24005


namespace max_m_n_l240_24004

theorem max_m_n (m n: ℕ) (h: m + 3*n - 5 = 2 * Nat.lcm m n - 11 * Nat.gcd m n) : 
  m + n ≤ 70 :=
sorry

end max_m_n_l240_24004


namespace boy_lap_time_l240_24030

noncomputable def total_time_needed
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) : ℝ :=
(side_lengths.zip running_speeds).foldl (λ (acc : ℝ) ⟨len, speed⟩ => acc + (len / (speed / 60))) 0
+ obstacle_time

theorem boy_lap_time
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) :
  side_lengths = [80, 120, 140, 100, 60] →
  running_speeds = [250, 200, 300, 166.67, 266.67] →
  obstacle_time = 5 →
  total_time_needed side_lengths running_speeds obstacle_time = 7.212 := by
  intros h_lengths h_speeds h_obstacle_time
  rw [h_lengths, h_speeds, h_obstacle_time]
  sorry

end boy_lap_time_l240_24030


namespace solve_inequality_l240_24051

theorem solve_inequality (a : ℝ) : (6 * x^2 + a * x - a^2 < 0) ↔
  ((a > 0) ∧ (-a / 2 < x ∧ x < a / 3)) ∨
  ((a < 0) ∧ (a / 3 < x ∧ x < -a / 2)) ∨
  ((a = 0) ∧ false) :=
by 
  sorry

end solve_inequality_l240_24051


namespace max_4x_3y_l240_24082

theorem max_4x_3y (x y : ℝ) (h : x^2 + y^2 = 16 * x + 8 * y + 8) : 4 * x + 3 * y ≤ 63 :=
sorry

end max_4x_3y_l240_24082


namespace equivalent_fraction_l240_24098

theorem equivalent_fraction : (8 / (5 * 46)) = (0.8 / 23) := 
by sorry

end equivalent_fraction_l240_24098


namespace total_coins_is_16_l240_24024

theorem total_coins_is_16 (x y : ℕ) (h₁ : x ≠ y) (h₂ : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := 
sorry

end total_coins_is_16_l240_24024


namespace ratio_w_y_l240_24067

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 4 / 3)
  (h3 : z / x = 1 / 8) : 
  w / y = 15 / 2 := 
by
  sorry

end ratio_w_y_l240_24067


namespace portrait_in_silver_box_l240_24019

-- Definitions for the first trial
def gold_box_1 : Prop := false
def gold_box_2 : Prop := true
def silver_box_1 : Prop := true
def silver_box_2 : Prop := false
def lead_box_1 : Prop := false
def lead_box_2 : Prop := true

-- Definitions for the second trial
def gold_box_3 : Prop := false
def gold_box_4 : Prop := true
def silver_box_3 : Prop := true
def silver_box_4 : Prop := false
def lead_box_3 : Prop := false
def lead_box_4 : Prop := true

-- The main theorem statement
theorem portrait_in_silver_box
  (gold_b1 : gold_box_1 = false)
  (gold_b2 : gold_box_2 = true)
  (silver_b1 : silver_box_1 = true)
  (silver_b2 : silver_box_2 = false)
  (lead_b1 : lead_box_1 = false)
  (lead_b2 : lead_box_2 = true)
  (gold_b3 : gold_box_3 = false)
  (gold_b4 : gold_box_4 = true)
  (silver_b3 : silver_box_3 = true)
  (silver_b4 : silver_box_4 = false)
  (lead_b3 : lead_box_3 = false)
  (lead_b4 : lead_box_4 = true) : 
  (silver_box_1 ∧ ¬lead_box_2) ∧ (silver_box_3 ∧ ¬lead_box_4) :=
sorry

end portrait_in_silver_box_l240_24019


namespace man_l240_24061

theorem man's_age_ratio_father (M F : ℕ) (hF : F = 60)
  (h_age_relationship : M + 12 = (F + 12) / 2) :
  M / F = 2 / 5 :=
by
  sorry

end man_l240_24061


namespace n_gon_angle_condition_l240_24076

theorem n_gon_angle_condition (n : ℕ) (h1 : 150 * (n-1) + (30 * n - 210) = 180 * (n-2)) (h2 : 30 * n - 210 < 150) (h3 : 30 * n - 210 > 0) :
  n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 :=
by
  sorry

end n_gon_angle_condition_l240_24076


namespace inradius_of_triangle_area_twice_perimeter_l240_24021

theorem inradius_of_triangle_area_twice_perimeter (A p r s : ℝ) (hA : A = 2 * p) (hs : p = 2 * s)
  (hA_formula : A = r * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_area_twice_perimeter_l240_24021


namespace tan_sum_identity_l240_24094

theorem tan_sum_identity (theta : Real) (h : Real.tan theta = 1 / 3) :
  Real.tan (theta + Real.pi / 4) = 2 :=
by
  sorry

end tan_sum_identity_l240_24094


namespace chi_square_association_l240_24009

theorem chi_square_association (k : ℝ) :
  (k > 3.841 → (∃ A B, A ∧ B)) ∧ (k ≤ 2.076 → (∃ A B, ¬(A ∧ B))) :=
by
  sorry

end chi_square_association_l240_24009


namespace sequence_bound_l240_24072

theorem sequence_bound (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n) ^ 2 ≤ a (n + 1)) :
  ∀ n, a n < 1 / n :=
by
  intros
  sorry

end sequence_bound_l240_24072


namespace find_a_of_exp_function_l240_24040

theorem find_a_of_exp_function (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a ^ 2 = 9) : a = 3 :=
sorry

end find_a_of_exp_function_l240_24040


namespace problem_1_problem_2_l240_24099

namespace ProofProblems

def U : Set ℝ := {y | true}

def E : Set ℝ := {y | y > 2}

def F : Set ℝ := {y | ∃ (x : ℝ), (-1 < x ∧ x < 2 ∧ y = x^2 - 2*x)}

def complement (A : Set ℝ) : Set ℝ := {y | y ∉ A}

theorem problem_1 : 
  (complement E ∩ F) = {y | -1 ≤ y ∧ y ≤ 2} := 
  sorry

def G (a : ℝ) : Set ℝ := {y | ∃ (x : ℝ), (0 < x ∧ x < a ∧ y = Real.log x / Real.log 2)}

theorem problem_2 (a : ℝ) :
  (∀ y, (y ∈ G a → y < 3)) → a ≥ 8 :=
  sorry

end ProofProblems

end problem_1_problem_2_l240_24099


namespace construction_better_than_logistics_l240_24095

theorem construction_better_than_logistics 
  (applications_computer : ℕ := 215830)
  (applications_mechanical : ℕ := 200250)
  (applications_marketing : ℕ := 154676)
  (applications_logistics : ℕ := 74570)
  (applications_trade : ℕ := 65280)
  (recruitments_computer : ℕ := 124620)
  (recruitments_marketing : ℕ := 102935)
  (recruitments_mechanical : ℕ := 89115)
  (recruitments_construction : ℕ := 76516)
  (recruitments_chemical : ℕ := 70436) :
  applications_construction / recruitments_construction < applications_logistics / recruitments_logistics→ 
  (applications_computer / recruitments_computer < applications_chemical / recruitments_chemical) :=
sorry

end construction_better_than_logistics_l240_24095


namespace max_liters_of_water_heated_l240_24058

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end max_liters_of_water_heated_l240_24058


namespace line_through_point_with_equal_intercepts_l240_24034

theorem line_through_point_with_equal_intercepts
  (P : ℝ × ℝ) (hP : P = (1, 3))
  (intercepts_equal : ∃ a : ℝ, a ≠ 0 ∧ (∀ x y : ℝ, (x/a) + (y/a) = 1 → x + y = 4 ∨ 3*x - y = 0)) :
  ∃ a b c : ℝ, (a, b, c) = (3, -1, 0) ∨ (a, b, c) = (1, 1, -4) ∧ (∀ x y : ℝ, a*x + b*y + c = 0 → (x + y = 4 ∨ 3*x - y = 0)) := 
by
  sorry

end line_through_point_with_equal_intercepts_l240_24034


namespace sum_of_ages_of_alex_and_allison_is_47_l240_24041

theorem sum_of_ages_of_alex_and_allison_is_47 (diane_age_now : ℕ)
  (diane_age_at_30_alex_relation : diane_age_now + 14 = 30 ∧ diane_age_now + 14 = 60 / 2)
  (diane_age_at_30_allison_relation : diane_age_now + 14 = 30 ∧ 30 = 2 * (diane_age_now + 14 - (30 - 15)))
  : (60 - (30 - 16)) + (15 - (30 - 16)) = 47 :=
by
  sorry

end sum_of_ages_of_alex_and_allison_is_47_l240_24041


namespace negation_proposition_l240_24032

-- Define the proposition as a Lean function
def quadratic_non_negative (x : ℝ) : Prop := x^2 - 2*x + 1 ≥ 0

-- State the theorem that we need to prove
theorem negation_proposition : ∀ x : ℝ, quadratic_non_negative x :=
by 
  sorry

end negation_proposition_l240_24032


namespace range_of_x_l240_24001

theorem range_of_x (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x / (Real.sqrt (x + 2)) :=
by {
  sorry
}

end range_of_x_l240_24001


namespace count_not_divisible_by_5_or_7_l240_24008

theorem count_not_divisible_by_5_or_7 :
  let n := 1000
  let count_divisible_by (m : ℕ) := Nat.floor (999 / m)
  (999 - count_divisible_by 5 - count_divisible_by 7 + count_divisible_by 35) = 686 :=
by
  sorry

end count_not_divisible_by_5_or_7_l240_24008


namespace find_K_l240_24084

def satisfies_conditions (K m n h : ℕ) : Prop :=
  K ∣ (m^h - 1) ∧ K ∣ (n ^ ((m^h - 1) / K) + 1)

def odd (n : ℕ) : Prop := n % 2 = 1

theorem find_K (r : ℕ) (h : ℕ := 2^r) :
    ∀ K : ℕ, (∃ (m : ℕ), odd m ∧ m > 1 ∧ ∃ (n : ℕ), satisfies_conditions K m n h) ↔
    (∃ s t : ℕ, K = 2^(r + s) * t ∧ 2 ∣ t) := sorry

end find_K_l240_24084


namespace booth_earnings_after_5_days_l240_24017

def booth_daily_popcorn_earnings := 50
def booth_daily_cotton_candy_earnings := 3 * booth_daily_popcorn_earnings
def booth_total_daily_earnings := booth_daily_popcorn_earnings + booth_daily_cotton_candy_earnings
def booth_total_expenses := 30 + 75

theorem booth_earnings_after_5_days :
  5 * booth_total_daily_earnings - booth_total_expenses = 895 :=
by
  sorry

end booth_earnings_after_5_days_l240_24017


namespace gcd_of_A_B_l240_24043

noncomputable def A (k : ℕ) := 2 * k
noncomputable def B (k : ℕ) := 5 * k

theorem gcd_of_A_B (k : ℕ) (h_lcm : Nat.lcm (A k) (B k) = 180) : Nat.gcd (A k) (B k) = 18 :=
by
  sorry

end gcd_of_A_B_l240_24043


namespace natasha_time_reach_top_l240_24075

variable (t : ℝ) (d_up d_total T : ℝ)

def time_to_reach_top (T d_up d_total t : ℝ) : Prop :=
  d_total = 2 * d_up ∧
  d_up = 1.5 * t ∧
  T = t + 2 ∧
  2 = d_total / T

theorem natasha_time_reach_top (T : ℝ) (h : time_to_reach_top T (1.5 * 4) (3 * 4) 4) : T = 4 :=
by
  sorry

end natasha_time_reach_top_l240_24075


namespace set_intersection_is_correct_l240_24003

def setA : Set ℝ := {x | x^2 - 4 * x > 0}
def setB : Set ℝ := {x | abs (x - 1) ≤ 2}
def setIntersection : Set ℝ := {x | -1 ≤ x ∧ x < 0}

theorem set_intersection_is_correct :
  setA ∩ setB = setIntersection := 
by
  sorry

end set_intersection_is_correct_l240_24003


namespace find_sum_of_cubes_l240_24039

theorem find_sum_of_cubes (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : (a^3 + 9) / a = (b^3 + 9) / b)
  (h₅ : (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
by
  sorry

end find_sum_of_cubes_l240_24039


namespace boat_capacity_problem_l240_24071

variables (L S : ℕ)

theorem boat_capacity_problem
  (h1 : L + 4 * S = 46)
  (h2 : 2 * L + 3 * S = 57) :
  3 * L + 6 * S = 96 :=
sorry

end boat_capacity_problem_l240_24071


namespace line_hyperbola_unique_intersection_l240_24096

theorem line_hyperbola_unique_intersection (k : ℝ) :
  (∃ (x y : ℝ), k * x - y - 2 * k = 0 ∧ x^2 - y^2 = 2 ∧ 
  ∀ y₁, y₁ ≠ y → k * x - y₁ - 2 * k ≠ 0 ∧ x^2 - y₁^2 ≠ 2) ↔ (k = 1 ∨ k = -1) :=
by
  sorry

end line_hyperbola_unique_intersection_l240_24096


namespace sum_of_primes_between_1_and_20_l240_24006

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l240_24006


namespace mrs_hilt_total_distance_l240_24049

def total_distance_walked (d n : ℕ) : ℕ := 2 * d * n

theorem mrs_hilt_total_distance :
  total_distance_walked 30 4 = 240 :=
by
  -- Proof goes here
  sorry

end mrs_hilt_total_distance_l240_24049


namespace problem1_problem2_l240_24018

section ArithmeticSequence

variable {a : ℕ → ℤ} {a1 a5 a8 a6 a4 d : ℤ}

-- Problem 1: Prove that if a_5 = -1 and a_8 = 2, then a_1 = -5 and d = 1
theorem problem1 
  (h1 : a 5 = -1) 
  (h2 : a 8 = 2)
  (h3 : ∀ n, a n = a1 + n * d) : 
  a1 = -5 ∧ d = 1 := 
sorry 

-- Problem 2: Prove that if a_1 + a_6 = 12 and a_4 = 7, then a_9 = 17
theorem problem2 
  (h1 : a1 + a 6 = 12) 
  (h2 : a 4 = 7)
  (h3 : ∀ n, a n = a1 + n * d) 
  (h4 : ∀ m (hm : m ≠ 0), a1 = a 1): 
   a 9 = 17 := 
sorry

end ArithmeticSequence

end problem1_problem2_l240_24018


namespace find_a_l240_24007

-- Given conditions as definitions.
def f (a x : ℝ) := a * x^3
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 3 * x + a - 3

-- Problem statement in Lean 4.
theorem find_a (a : ℝ) (h_tangent : ∀ x : ℝ, f a 1 = 1 ∧ f a 1 = tangent_line a 1) : a = 1 := 
by sorry

end find_a_l240_24007


namespace arithmetic_sequence_a10_l240_24062

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_positive : ∀ n, a n > 0) 
  (h_sum : a 1 + a 2 + a 3 = 15) 
  (h_geo : (a 1 + 2) * (a 3 + 13) = (a 2 + 5) * (a 2 + 5))  
  : a 10 = 21 := sorry

end arithmetic_sequence_a10_l240_24062


namespace swap_columns_produce_B_l240_24029

variables {n : ℕ} (A : Matrix (Fin n) (Fin n) (Fin n))

def K (B : Matrix (Fin n) (Fin n) (Fin n)) : ℕ :=
  Fintype.card {ij : (Fin n) × (Fin n) // B ij.1 ij.2 = ij.2}

theorem swap_columns_produce_B (A : Matrix (Fin n) (Fin n) (Fin n)) :
  ∃ (B : Matrix (Fin n) (Fin n) (Fin n)), (∀ i, ∃ j, B i j = A i j) ∧ K B ≤ n :=
sorry

end swap_columns_produce_B_l240_24029


namespace real_roots_exactly_three_l240_24077

theorem real_roots_exactly_three (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * |x| + 2 = m) → (∃ a b c : ℝ, 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (a^2 - 2 * |a| + 2 = m) ∧ 
  (b^2 - 2 * |b| + 2 = m) ∧ 
  (c^2 - 2 * |c| + 2 = m)) → 
  m = 2 := 
sorry

end real_roots_exactly_three_l240_24077


namespace Seojun_apples_decimal_l240_24010

theorem Seojun_apples_decimal :
  let total_apples := 100
  let seojun_apples := 11
  seojun_apples / total_apples = 0.11 :=
by
  let total_apples := 100
  let seojun_apples := 11
  sorry

end Seojun_apples_decimal_l240_24010


namespace total_gems_in_chest_l240_24048

theorem total_gems_in_chest (diamonds rubies : ℕ) 
  (h_diamonds : diamonds = 45)
  (h_rubies : rubies = 5110) : 
  diamonds + rubies = 5155 := 
by 
  sorry

end total_gems_in_chest_l240_24048


namespace least_whole_number_subtracted_l240_24060

theorem least_whole_number_subtracted (x : ℕ) :
  ((6 - x) / (7 - x) < (16 / 21)) → x = 3 :=
by
  sorry

end least_whole_number_subtracted_l240_24060


namespace fraction_torn_off_l240_24000

theorem fraction_torn_off (P: ℝ) (A_remaining: ℝ) (fraction: ℝ):
  P = 32 → 
  A_remaining = 48 → 
  fraction = 1 / 4 :=
by 
  sorry

end fraction_torn_off_l240_24000


namespace paul_spent_81_90_l240_24016

-- Define the original price of each racket
def originalPrice : ℝ := 60

-- Define the discount rates
def firstDiscount : ℝ := 0.20
def secondDiscount : ℝ := 0.50

-- Define the sales tax rate
def salesTax : ℝ := 0.05

-- Define the prices after discount
def firstRacketPrice : ℝ := originalPrice * (1 - firstDiscount)
def secondRacketPrice : ℝ := originalPrice * (1 - secondDiscount)

-- Define the total price before tax
def totalPriceBeforeTax : ℝ := firstRacketPrice + secondRacketPrice

-- Define the total sales tax
def totalSalesTax : ℝ := totalPriceBeforeTax * salesTax

-- Define the total amount spent
def totalAmountSpent : ℝ := totalPriceBeforeTax + totalSalesTax

-- The statement to prove
theorem paul_spent_81_90 : totalAmountSpent = 81.90 := 
by
  sorry

end paul_spent_81_90_l240_24016


namespace annual_profit_function_correct_maximum_annual_profit_l240_24031

noncomputable def fixed_cost : ℝ := 60

noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 12 then 
    0.5 * x^2 + 4 * x 
  else 
    11 * x + 100 / x - 39

noncomputable def selling_price_per_thousand : ℝ := 10

noncomputable def sales_revenue (x : ℝ) : ℝ := selling_price_per_thousand * x

noncomputable def annual_profit (x : ℝ) : ℝ := sales_revenue x - fixed_cost - variable_cost x

theorem annual_profit_function_correct : 
∀ x : ℝ, (0 < x ∧ x < 12 → annual_profit x = -0.5 * x^2 + 6 * x - fixed_cost) ∧ 
        (x ≥ 12 → annual_profit x = -x - 100 / x + 33) :=
sorry

theorem maximum_annual_profit : 
∃ x : ℝ, x = 12 ∧ annual_profit x = 38 / 3 :=
sorry

end annual_profit_function_correct_maximum_annual_profit_l240_24031


namespace maximum_contribution_l240_24080

theorem maximum_contribution (total_contribution : ℕ) (num_people : ℕ) (individual_min_contribution : ℕ) :
  total_contribution = 20 → num_people = 10 → individual_min_contribution = 1 → 
  ∃ (max_contribution : ℕ), max_contribution = 11 := by
  intro h1 h2 h3
  existsi 11
  sorry

end maximum_contribution_l240_24080


namespace visitor_increase_l240_24069

variable (x : ℝ) -- The percentage increase each day

theorem visitor_increase (h1 : 1.2 * (1 + x)^2 = 2.5) : 1.2 * (1 + x)^2 = 2.5 :=
by exact h1

end visitor_increase_l240_24069


namespace ln_of_x_sq_sub_2x_monotonic_l240_24052

noncomputable def ln_of_x_sq_sub_2x : ℝ → ℝ := fun x => Real.log (x^2 - 2*x)

theorem ln_of_x_sq_sub_2x_monotonic : ∀ x y : ℝ, (2 < x ∧ 2 < y ∧ x ≤ y) → ln_of_x_sq_sub_2x x ≤ ln_of_x_sq_sub_2x y :=
by
    intros x y h
    sorry

end ln_of_x_sq_sub_2x_monotonic_l240_24052


namespace obtuse_angle_at_515_l240_24090

-- Definitions derived from conditions
def minuteHandDegrees (minute: ℕ) : ℝ := minute * 6.0
def hourHandDegrees (hour: ℕ) (minute: ℕ) : ℝ := hour * 30.0 + (minute * 0.5)

-- Main statement to be proved
theorem obtuse_angle_at_515 : 
  let hour := 5
  let minute := 15
  let minute_pos := minuteHandDegrees minute
  let hour_pos := hourHandDegrees hour minute
  let angle := abs (minute_pos - hour_pos)
  angle = 67.5 :=
by
  sorry

end obtuse_angle_at_515_l240_24090


namespace arithmetic_sum_l240_24056

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
  sorry

end arithmetic_sum_l240_24056


namespace statement_A_statement_C_statement_D_statement_B_l240_24014

variable (a b : ℝ)

theorem statement_A :
  4 * a^2 - a * b + b^2 = 1 → |a| ≤ 2 * Real.sqrt 15 / 15 :=
sorry

theorem statement_C :
  (4 * a^2 - a * b + b^2 = 1) → 4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3 :=
sorry

theorem statement_D :
  4 * a^2 - a * b + b^2 = 1 → |2 * a - b| ≤ 2 * Real.sqrt 10 / 5 :=
sorry

theorem statement_B :
  4 * a^2 - a * b + b^2 = 1 → ¬(|a + b| < 1) :=
sorry

end statement_A_statement_C_statement_D_statement_B_l240_24014


namespace find_values_general_formula_l240_24053

variable (a_n S_n : ℕ → ℝ)

-- Conditions
axiom sum_sequence (n : ℕ) (hn : n > 0) :  S_n n = (1 / 3) * (a_n n - 1)

-- Questions
theorem find_values :
  (a_n 1 = 2) ∧ (a_n 2 = 5) ∧ (a_n 3 = 8) := sorry

theorem general_formula (n : ℕ) :
  n > 0 → a_n n = n + 1 := sorry

end find_values_general_formula_l240_24053


namespace line_intersects_circle_l240_24023

theorem line_intersects_circle (k : ℝ) : ∀ (x y : ℝ),
  (x + y) ^ 2 = x ^ 2 + y ^ 2 →
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2)) ∧ 
  ((-1/2)^2 + (0)^2 < 1) →
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2) := 
by
  intro x y h₁ h₂
  sorry

end line_intersects_circle_l240_24023


namespace determine_b_perpendicular_l240_24083

theorem determine_b_perpendicular :
  ∀ (b : ℝ),
  (b * 2 + (-3) * (-1) + 2 * 4 = 0) → 
  b = -11/2 :=
by
  intros b h
  sorry

end determine_b_perpendicular_l240_24083


namespace coefficients_equality_l240_24042

theorem coefficients_equality (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : a_1 * (x-1)^4 + a_2 * (x-1)^3 + a_3 * (x-1)^2 + a_4 * (x-1) + a_5 = x^4)
  (h1 : a_1 = 1)
  (h2 : a_5 = 1)
  (h3 : 1 - a_2 + a_3 - a_4 + 1 = 0) :
  a_2 - a_3 + a_4 = 2 :=
sorry

end coefficients_equality_l240_24042


namespace symmetric_point_y_axis_l240_24033

theorem symmetric_point_y_axis (B : ℝ × ℝ) (hB : B = (-3, 4)) : 
  ∃ A : ℝ × ℝ, A = (3, 4) ∧ A.2 = B.2 ∧ A.1 = -B.1 :=
by
  use (3, 4)
  sorry

end symmetric_point_y_axis_l240_24033


namespace no_32_people_class_exists_30_people_class_l240_24091

-- Definition of the conditions: relationship between boys and girls
def friends_condition (B G : ℕ) : Prop :=
  3 * B = 2 * G

-- The first problem statement: No 32 people class
theorem no_32_people_class : ¬ ∃ (B G : ℕ), friends_condition B G ∧ B + G = 32 := 
sorry

-- The second problem statement: There is a 30 people class
theorem exists_30_people_class : ∃ (B G : ℕ), friends_condition B G ∧ B + G = 30 := 
sorry

end no_32_people_class_exists_30_people_class_l240_24091


namespace LiFangOutfitChoices_l240_24026

variable (shirts skirts dresses : Nat) 

theorem LiFangOutfitChoices (h_shirts : shirts = 4) (h_skirts : skirts = 3) (h_dresses : dresses = 2) :
  shirts * skirts + dresses = 14 :=
by 
  -- Given the conditions and the calculations, the expected result follows.
  sorry

end LiFangOutfitChoices_l240_24026


namespace room_length_l240_24011

theorem room_length (length width rate cost : ℝ)
    (h_width : width = 3.75)
    (h_rate : rate = 1000)
    (h_cost : cost = 20625)
    (h_eq : cost = length * width * rate) :
    length = 5.5 :=
by
  -- the proof will go here
  sorry

end room_length_l240_24011


namespace not_prime_expression_l240_24065

theorem not_prime_expression (x y : ℕ) : ¬ Prime (x^8 - x^7 * y + x^6 * y^2 - x^5 * y^3 + x^4 * y^4 
  - x^3 * y^5 + x^2 * y^6 - x * y^7 + y^8) :=
sorry

end not_prime_expression_l240_24065


namespace Jessie_points_l240_24064

theorem Jessie_points (total_points team_points : ℕ) (players_points : ℕ) (P Q R : ℕ) (eq1 : total_points = 311) (eq2 : players_points = 188) (eq3 : team_points - players_points = 3 * P) (eq4 : P = Q) (eq5 : Q = R) : Q = 41 :=
by
  sorry

end Jessie_points_l240_24064


namespace rectangle_area_proof_l240_24012

def rectangle_width : ℕ := 5

def rectangle_length (width : ℕ) : ℕ := 3 * width

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem rectangle_area_proof : rectangle_area (rectangle_length rectangle_width) rectangle_width = 75 := by
  sorry -- Proof can be added later

end rectangle_area_proof_l240_24012


namespace find_a2_l240_24063

noncomputable def a_sequence (k : ℕ+) (n : ℕ) : ℚ :=
  -(1 / 2 : ℚ) * n^2 + k * n

theorem find_a2
  (k : ℕ+)
  (max_S : ∀ n : ℕ, a_sequence k n ≤ 8)
  (max_reached : ∃ n : ℕ, a_sequence k n = 8) :
  a_sequence 4 2 - a_sequence 4 1 = 5 / 2 :=
by
  -- To be proved, insert appropriate steps here
  sorry

end find_a2_l240_24063


namespace garage_sale_items_l240_24068

theorem garage_sale_items (h : 34 = 13 + n + 1 + 14 - 14) : n = 22 := by
  sorry

end garage_sale_items_l240_24068


namespace container_volumes_l240_24020

theorem container_volumes (a r : ℝ) (h1 : (2 * a)^3 = (4 / 3) * Real.pi * r^3) :
  ((2 * a + 2)^3 > (4 / 3) * Real.pi * (r + 1)^3) :=
by sorry

end container_volumes_l240_24020


namespace decimal_difference_l240_24073

theorem decimal_difference : (0.650 : ℝ) - (1 / 8 : ℝ) = 0.525 := by
  sorry

end decimal_difference_l240_24073


namespace max_lambda_inequality_l240_24037

theorem max_lambda_inequality 
  (a b x y : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : a + b = 27) : 
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 4 * (a * x^2 * y + b * x * y^2)^2 :=
sorry

end max_lambda_inequality_l240_24037


namespace positive_solution_l240_24044

theorem positive_solution (x : ℝ) (h : (1 / 2) * (3 * x^2 - 1) = (x^2 - 50 * x - 10) * (x^2 + 25 * x + 5)) : x = 25 + Real.sqrt 159 :=
sorry

end positive_solution_l240_24044
