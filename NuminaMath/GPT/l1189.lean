import Mathlib

namespace max_paths_from_A_to_F_l1189_118945

-- Define the points and line segments.
inductive Point
| A | B | C | D | E | F

-- Define the edges of the graph as pairs of points.
def edges : List (Point × Point) :=
  [(Point.A, Point.B), (Point.A, Point.E), (Point.A, Point.D),
   (Point.B, Point.C), (Point.B, Point.E),
   (Point.C, Point.F),
   (Point.D, Point.E), (Point.D, Point.F),
   (Point.E, Point.F)]

-- A path is valid if it passes through each point and line segment only once.
def valid_path (path : List (Point × Point)) : Bool :=
  -- Check that each edge in the path is unique and forms a sequence from A to F.
  sorry

-- Calculate the maximum number of different valid paths from point A to point F.
def max_paths : Nat :=
  List.length (List.filter valid_path (List.permutations edges))

theorem max_paths_from_A_to_F : max_paths = 9 :=
by sorry

end max_paths_from_A_to_F_l1189_118945


namespace common_ratio_is_4_l1189_118976

theorem common_ratio_is_4 
  (a : ℕ → ℝ) -- The geometric sequence
  (r : ℝ) -- The common ratio
  (h_geo_seq : ∀ n, a (n + 1) = r * a n) -- Definition of geometric sequence
  (h_condition : ∀ n, a n * a (n + 1) = 16 ^ n) -- Given condition
  : r = 4 := 
  sorry

end common_ratio_is_4_l1189_118976


namespace henry_collection_cost_l1189_118998

def initial_figures : ℕ := 3
def total_needed : ℕ := 8
def cost_per_figure : ℕ := 6

theorem henry_collection_cost : 
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  total_cost = 30 := 
by
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  sorry

end henry_collection_cost_l1189_118998


namespace evaluate_f_at_2_l1189_118987

def f (x : ℕ) : ℕ := 5 * x + 2

theorem evaluate_f_at_2 : f 2 = 12 := by
  sorry

end evaluate_f_at_2_l1189_118987


namespace simone_fraction_per_day_l1189_118935

theorem simone_fraction_per_day 
  (x : ℚ) -- Define the fraction of an apple Simone ate each day as x.
  (h1 : 16 * x + 15 * (1/3) = 13) -- Condition: Simone and Lauri together ate 13 apples.
  : x = 1/2 := 
 by 
  sorry

end simone_fraction_per_day_l1189_118935


namespace pqrs_sum_l1189_118951

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l1189_118951


namespace max_value_of_expression_l1189_118915

noncomputable def max_value (x : ℝ) : ℝ :=
  x * (1 + x) * (3 - x)

theorem max_value_of_expression :
  ∃ x : ℝ, 0 < x ∧ max_value x = (70 + 26 * Real.sqrt 13) / 27 :=
sorry

end max_value_of_expression_l1189_118915


namespace ratio_mara_janet_l1189_118974

variables {B J M : ℕ}

/-- Janet has 9 cards more than Brenda --/
def janet_cards (B : ℕ) : ℕ := B + 9

/-- Mara has 40 cards less than 150 --/
def mara_cards : ℕ := 150 - 40

/-- They have a total of 211 cards --/
axiom total_cards_eq (B : ℕ) : B + janet_cards B + mara_cards = 211

/-- Mara has a multiple of Janet's number of cards --/
axiom multiples_cards (J M : ℕ) : J * 2 = M

theorem ratio_mara_janet (B J M : ℕ) (h1 : janet_cards B = J)
  (h2 : mara_cards = M) (h3 : J * 2 = M) :
  (M / J : ℕ) = 2 :=
sorry

end ratio_mara_janet_l1189_118974


namespace pedestrian_wait_probability_l1189_118932

-- Define the duration of the red light
def red_light_duration := 45

-- Define the favorable time window for the pedestrian to wait at least 20 seconds
def favorable_window := 25

-- The probability that the pedestrian has to wait at least 20 seconds
def probability_wait_at_least_20 : ℚ := favorable_window / red_light_duration

theorem pedestrian_wait_probability : probability_wait_at_least_20 = 5 / 9 := by
  sorry

end pedestrian_wait_probability_l1189_118932


namespace bernoulli_inequality_l1189_118918

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : 1 + n * x ≤ (1 + x) ^ n :=
sorry

end bernoulli_inequality_l1189_118918


namespace f_divisible_by_8_l1189_118927

-- Define the function f
def f (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

-- Theorem statement
theorem f_divisible_by_8 (n : ℕ) (hn : n > 0) : 8 ∣ f n := sorry

end f_divisible_by_8_l1189_118927


namespace quadrilateral_area_BEIH_l1189_118920

-- Define the necessary points in the problem
structure Point :=
(x : ℚ)
(y : ℚ)

-- Definitions of given points and midpoints
def B : Point := ⟨0, 0⟩
def E : Point := ⟨0, 1.5⟩
def F : Point := ⟨1.5, 0⟩

-- Definitions of line equations from points
def line_DE (p : Point) : Prop := p.y = - (1 / 2) * p.x + 1.5
def line_AF (p : Point) : Prop := p.y = -2 * p.x + 3

-- Intersection points
def I : Point := ⟨3 / 5, 9 / 5⟩
def H : Point := ⟨3 / 4, 3 / 4⟩

-- Function to calculate the area using the Shoelace Theorem
def shoelace_area (a b c d : Point) : ℚ :=
  (1 / 2) * ((a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y) - (a.y * b.x + b.y * c.x + c.y * d.x + d.y * a.x))

-- The proof statement
theorem quadrilateral_area_BEIH :
  shoelace_area B E I H = 9 / 16 :=
sorry

end quadrilateral_area_BEIH_l1189_118920


namespace shoe_size_15_is_9point25_l1189_118973

noncomputable def smallest_shoe_length (L : ℝ) := L
noncomputable def largest_shoe_length (L : ℝ) := L + 9 * (1/4 : ℝ)
noncomputable def length_ratio_condition (L : ℝ) := largest_shoe_length L = 1.30 * smallest_shoe_length L
noncomputable def shoe_length_size_15 (L : ℝ) := L + 7 * (1/4 : ℝ)

theorem shoe_size_15_is_9point25 : ∃ L : ℝ, length_ratio_condition L → shoe_length_size_15 L = 9.25 :=
by
  sorry

end shoe_size_15_is_9point25_l1189_118973


namespace second_train_further_l1189_118988

-- Define the speeds of the two trains
def speed_train1 : ℝ := 50
def speed_train2 : ℝ := 60

-- Define the total distance between points A and B
def total_distance : ℝ := 1100

-- Define the distances traveled by the two trains when they meet
def distance_train1 (t: ℝ) : ℝ := speed_train1 * t
def distance_train2 (t: ℝ) : ℝ := speed_train2 * t

-- Define the meeting condition
def meeting_condition (t: ℝ) : Prop := distance_train1 t + distance_train2 t = total_distance

-- Prove the distance difference
theorem second_train_further (t: ℝ) (h: meeting_condition t) : distance_train2 t - distance_train1 t = 100 :=
sorry

end second_train_further_l1189_118988


namespace differences_multiple_of_nine_l1189_118970

theorem differences_multiple_of_nine (S : Finset ℕ) (hS : S.card = 10) (h_unique : ∀ {x y : ℕ}, x ∈ S → y ∈ S → x ≠ y → x ≠ y) : 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 9 = 0 :=
by
  sorry

end differences_multiple_of_nine_l1189_118970


namespace problem_l1189_118966

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x ^ 2 - x - a * Real.log (x - a)

def monotonicity_f (a : ℝ) : Prop :=
  if a = 0 then
    ∀ x : ℝ, 0 < x → (x < 1 → f x 0 < f (x + 1) 0) ∧ (x > 1 → f x 0 > f (x + 1) 0)
  else if a > 0 then
    ∀ x : ℝ, a < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f x a > f (x + 1) a)
  else if -1 < a ∧ a < 0 then
    ∀ x : ℝ, 0 < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f (x + 1) a > f x a)
  else if a = -1 then
    ∀ x : ℝ, -1 < x → f x (-1) < f (x + 1) (-1)
  else
    ∀ x : ℝ, a < x → (x < 0 → f (x + 1) a > f x a) ∧ (0 < x → f x a > f (x + 1) a)

noncomputable def g (x a : ℝ) : ℝ := f (x + a) a - a * (x + (1/2) * a - 1)

def extreme_points (x₁ x₂ a : ℝ) : Prop :=
  x₁ < x₂ ∧ ∀ x : ℝ, 0 < x → x < 1 → g x a = 0

theorem problem (a : ℝ) (x₁ x₂ : ℝ) (hx : extreme_points x₁ x₂ a) (h_dom : -1/4 < a ∧ a < 0) :
  0 < f x₁ a - f x₂ a ∧ f x₁ a - f x₂ a < 1/2 := sorry

end problem_l1189_118966


namespace problem_statement_l1189_118982

def f (x : ℝ) : ℝ := sorry

theorem problem_statement
  (cond1 : ∀ {x y w : ℝ}, x > y → f x + x ≥ w → w ≥ f y + y → ∃ (z : ℝ), z ∈ Set.Icc y x ∧ f z = w - z)
  (cond2 : ∃ (u : ℝ), 0 ∈ Set.range f ∧ ∀ a ∈ Set.range f, u ≤ a)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 := sorry

end problem_statement_l1189_118982


namespace crates_lost_l1189_118901

theorem crates_lost (total_crates : ℕ) (total_cost : ℕ) (desired_profit_percent : ℕ) 
(lost_crates remaining_crates : ℕ) (price_per_crate : ℕ) 
(h1 : total_crates = 10) (h2 : total_cost = 160) (h3 : desired_profit_percent = 25) 
(h4 : price_per_crate = 25) (h5 : remaining_crates = total_crates - lost_crates)
(h6 : price_per_crate * remaining_crates = total_cost + total_cost * desired_profit_percent / 100) :
  lost_crates = 2 :=
by
  sorry

end crates_lost_l1189_118901


namespace find_shift_b_l1189_118931

-- Define the periodic function f
variable (f : ℝ → ℝ)
-- Define the condition on f
axiom f_periodic : ∀ x, f (x - 30) = f x

-- The theorem we want to prove
theorem find_shift_b : ∃ b > 0, (∀ x, f ((x - b) / 3) = f (x / 3)) ∧ b = 90 := 
by
  sorry

end find_shift_b_l1189_118931


namespace mushroom_mass_decrease_l1189_118938

theorem mushroom_mass_decrease :
  ∀ (initial_mass water_content_fresh water_content_dry : ℝ),
  water_content_fresh = 0.8 →
  water_content_dry = 0.2 →
  (initial_mass * (1 - water_content_fresh) / (1 - water_content_dry) = initial_mass * 0.25) →
  (initial_mass - initial_mass * 0.25) / initial_mass = 0.75 :=
by
  intros initial_mass water_content_fresh water_content_dry h_fresh h_dry h_dry_mass
  sorry

end mushroom_mass_decrease_l1189_118938


namespace resulting_polygon_sides_l1189_118950

theorem resulting_polygon_sides :
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let decagon_sides := 10
  let shared_square_decagon := 2
  let shared_between_others := 2 * 5 -- 2 sides shared for pentagon to nonagon
  let total_shared_sides := shared_square_decagon + shared_between_others
  let total_unshared_sides := 
    square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides + nonagon_sides + decagon_sides
  total_unshared_sides - total_shared_sides = 37 := by
  sorry

end resulting_polygon_sides_l1189_118950


namespace potassium_bromate_molecular_weight_l1189_118958

def potassium_atomic_weight : Real := 39.10
def bromine_atomic_weight : Real := 79.90
def oxygen_atomic_weight : Real := 16.00
def oxygen_atoms : Nat := 3

theorem potassium_bromate_molecular_weight :
  potassium_atomic_weight + bromine_atomic_weight + oxygen_atoms * oxygen_atomic_weight = 167.00 :=
by
  sorry

end potassium_bromate_molecular_weight_l1189_118958


namespace quadratic_roots_l1189_118942

theorem quadratic_roots {x y : ℝ} (h1 : x + y = 8) (h2 : |x - y| = 10) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (x^2 - 8*x - 9 = 0) ∧ (y^2 - 8*y - 9 = 0) :=
by
  sorry

end quadratic_roots_l1189_118942


namespace speed_conversion_l1189_118964

theorem speed_conversion (speed_mps: ℝ) (conversion_factor: ℝ) (expected_speed_kmph: ℝ):
  speed_mps * conversion_factor = expected_speed_kmph :=
by
  let speed_mps := 115.00919999999999
  let conversion_factor := 3.6
  let expected_speed_kmph := 414.03312
  sorry

end speed_conversion_l1189_118964


namespace hillary_descending_rate_l1189_118984

def baseCampDistance : ℕ := 4700
def hillaryClimbingRate : ℕ := 800
def eddyClimbingRate : ℕ := 500
def hillaryStopShort : ℕ := 700
def departTime : ℕ := 6 -- time is represented in hours from midnight
def passTime : ℕ := 12 -- time is represented in hours from midnight

theorem hillary_descending_rate :
  ∃ r : ℕ, r = 1000 := by
  sorry

end hillary_descending_rate_l1189_118984


namespace length_of_AE_l1189_118900

noncomputable def AE_calculation (AB AC AD : ℝ) (h : ℝ) (AE : ℝ) : Prop :=
  AB = 3.6 ∧ AC = 3.6 ∧ AD = 1.2 ∧ 
  (0.5 * AC * h = 0.5 * AE * (1/3) * h) →
  AE = 10.8

theorem length_of_AE {h : ℝ} : AE_calculation 3.6 3.6 1.2 h 10.8 :=
sorry

end length_of_AE_l1189_118900


namespace remainder_of_a_squared_l1189_118917

theorem remainder_of_a_squared (n : ℕ) (a : ℤ) (h : a % n * a % n % n = 1) : (a * a) % n = 1 := by
  sorry

end remainder_of_a_squared_l1189_118917


namespace complement_intersection_l1189_118975

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) : (U \ (M ∩ N)) = {1, 4} := 
by
  sorry

end complement_intersection_l1189_118975


namespace green_tractor_price_l1189_118943

-- Define the conditions
def salary_based_on_sales (r_ct : Nat) (r_price : ℝ) (g_ct : Nat) (g_price : ℝ) : ℝ :=
  0.1 * r_ct * r_price + 0.2 * g_ct * g_price

-- Define the problem's Lean statement
theorem green_tractor_price
  (r_ct : Nat) (g_ct : Nat)
  (r_price : ℝ) (total_salary : ℝ)
  (h_rct : r_ct = 2)
  (h_gct : g_ct = 3)
  (h_rprice : r_price = 20000)
  (h_salary : total_salary = 7000) :
  ∃ g_price : ℝ, salary_based_on_sales r_ct r_price g_ct g_price = total_salary ∧ g_price = 5000 :=
by
  sorry

end green_tractor_price_l1189_118943


namespace sandra_coffee_l1189_118991

theorem sandra_coffee (S : ℕ) (H1 : 2 + S = 8) : S = 6 :=
by
  sorry

end sandra_coffee_l1189_118991


namespace like_terms_sum_l1189_118959

theorem like_terms_sum (m n : ℕ) (h1 : m + 1 = 1) (h2 : 3 = n) : m + n = 3 :=
by sorry

end like_terms_sum_l1189_118959


namespace inequality_problem_l1189_118981

-- Define a and the condition that expresses the given problem as an inequality
variable (a : ℝ)

-- The inequality to prove
theorem inequality_problem : a - 5 > 2 * a := sorry

end inequality_problem_l1189_118981


namespace ratios_of_PQR_and_XYZ_l1189_118937

-- Define triangle sides
def sides_PQR : ℕ × ℕ × ℕ := (7, 24, 25)
def sides_XYZ : ℕ × ℕ × ℕ := (9, 40, 41)

-- Perimeter calculation functions
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Area calculation functions for right triangles
def area (a b : ℕ) : ℕ := (a * b) / 2

-- Required proof statement
theorem ratios_of_PQR_and_XYZ :
  let (a₁, b₁, c₁) := sides_PQR
  let (a₂, b₂, c₂) := sides_XYZ
  area a₁ b₁ * 15 = 7 * area a₂ b₂ ∧ perimeter a₁ b₁ c₁ * 45 = 28 * perimeter a₂ b₂ c₂ :=
sorry

end ratios_of_PQR_and_XYZ_l1189_118937


namespace cost_of_horse_l1189_118919

theorem cost_of_horse (H C : ℝ) 
  (h1 : 4 * H + 9 * C = 13400)
  (h2 : 0.4 * H + 1.8 * C = 1880) :
  H = 2000 :=
by
  sorry

end cost_of_horse_l1189_118919


namespace solve_inequality_l1189_118979

def solution_set_of_inequality : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

theorem solve_inequality (x : ℝ) (h : (2 - x) / (x + 4) > 0) : x ∈ solution_set_of_inequality :=
by
  sorry

end solve_inequality_l1189_118979


namespace sum_of_six_angles_l1189_118965

theorem sum_of_six_angles (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 + a3 + a5 = 180)
                           (h2 : a2 + a4 + a6 = 180) : 
                           a1 + a2 + a3 + a4 + a5 + a6 = 360 := 
by
  -- omitted proof
  sorry

end sum_of_six_angles_l1189_118965


namespace find_2023rd_digit_of_11_div_13_l1189_118989

noncomputable def decimal_expansion_repeating (n d : Nat) : List Nat := sorry

noncomputable def decimal_expansion_digit (n d pos : Nat) : Nat :=
  let repeating_block := decimal_expansion_repeating n d
  repeating_block.get! ((pos - 1) % repeating_block.length)

theorem find_2023rd_digit_of_11_div_13 :
  decimal_expansion_digit 11 13 2023 = 8 := by
  sorry

end find_2023rd_digit_of_11_div_13_l1189_118989


namespace gum_boxes_l1189_118944

theorem gum_boxes (c s t g : ℕ) (h1 : c = 2) (h2 : s = 5) (h3 : t = 9) (h4 : c + s + g = t) : g = 2 := by
  sorry

end gum_boxes_l1189_118944


namespace parts_per_day_l1189_118902

noncomputable def total_parts : ℕ := 400
noncomputable def unfinished_parts_after_3_days : ℕ := 60
noncomputable def excess_parts_after_3_days : ℕ := 20

variables (x y : ℕ)

noncomputable def condition1 : Prop := (3 * x + 2 * y = total_parts - unfinished_parts_after_3_days)
noncomputable def condition2 : Prop := (3 * x + 3 * y = total_parts + excess_parts_after_3_days)

theorem parts_per_day (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 60 ∧ y = 80 :=
by {
  sorry
}

end parts_per_day_l1189_118902


namespace not_perfect_square_9n_squared_minus_9n_plus_9_l1189_118925

theorem not_perfect_square_9n_squared_minus_9n_plus_9
  (n : ℕ) (h : n > 1) : ¬ (∃ k : ℕ, 9 * n^2 - 9 * n + 9 = k * k) := sorry

end not_perfect_square_9n_squared_minus_9n_plus_9_l1189_118925


namespace jack_jill_same_speed_l1189_118921

theorem jack_jill_same_speed (x : ℝ) (h : x^2 - 8*x - 10 = 0) :
  (x^2 - 7*x - 18) = 2 := 
sorry

end jack_jill_same_speed_l1189_118921


namespace unique_real_solution_l1189_118978

theorem unique_real_solution :
  ∃! (a b : ℝ), 2 * (a^2 + 1) * (b^2 + 1) = (a + 1)^2 * (ab + 1) ∧ a = 1 ∧ b = 1 :=
by
  sorry

end unique_real_solution_l1189_118978


namespace exponential_inequality_l1189_118930

theorem exponential_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (Real.exp a * Real.exp c > Real.exp b * Real.exp d) :=
by sorry

end exponential_inequality_l1189_118930


namespace pieces_to_same_point_l1189_118922

theorem pieces_to_same_point :
  ∀ (x y z : ℤ), (∃ (final_pos : ℤ), (x = final_pos ∧ y = final_pos ∧ z = final_pos)) ↔ 
  (x, y, z) = (1, 2009, 2010) ∨ 
  (x, y, z) = (0, 2009, 2010) ∨ 
  (x, y, z) = (2, 2009, 2010) ∨ 
  (x, y, z) = (3, 2009, 2010) := 
by {
  sorry
}

end pieces_to_same_point_l1189_118922


namespace arithmetic_sequence_75th_term_l1189_118954

theorem arithmetic_sequence_75th_term (a d n : ℕ) (h1 : a = 2) (h2 : d = 4) (h3 : n = 75) : 
  a + (n - 1) * d = 298 :=
by 
  sorry

end arithmetic_sequence_75th_term_l1189_118954


namespace tim_drinks_amount_l1189_118934

theorem tim_drinks_amount (H : ℚ := 2/7) (T : ℚ := 5/8) : 
  (T * H) = 5/28 :=
by sorry

end tim_drinks_amount_l1189_118934


namespace common_fraction_proof_l1189_118913

def expr_as_common_fraction : Prop :=
  let numerator := (3 / 6) + (4 / 5)
  let denominator := (5 / 12) + (1 / 4)
  (numerator / denominator) = (39 / 20)

theorem common_fraction_proof : expr_as_common_fraction :=
by
  sorry

end common_fraction_proof_l1189_118913


namespace unique_integer_triplet_solution_l1189_118933

theorem unique_integer_triplet_solution (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : 
    (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end unique_integer_triplet_solution_l1189_118933


namespace point_of_tangency_l1189_118972

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)
noncomputable def f_deriv (x a : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem point_of_tangency (a : ℝ) (h1 : ∀ x, f_deriv (-x) a = -f_deriv x a)
  (h2 : ∃ x0, f_deriv x0 1 = 3/2) :
  ∃ x0 y0, x0 = Real.log 2 ∧ y0 = f (Real.log 2) 1 ∧ y0 = 5/2 :=
by
  sorry

end point_of_tangency_l1189_118972


namespace bucket_full_weight_l1189_118912

variable (p q r : ℚ)
variable (x y : ℚ)

-- Define the conditions
def condition1 : Prop := p = r + (3 / 4) * y
def condition2 : Prop := q = r + (1 / 3) * y
def condition3 : Prop := x = r

-- Define the conclusion
def conclusion : Prop := x + y = (4 * p - r) / 3

-- The theorem stating that the conclusion follows from the conditions
theorem bucket_full_weight (h1 : condition1 p r y) (h2 : condition2 q r y) (h3 : condition3 x r) : conclusion x y p r :=
by
  sorry

end bucket_full_weight_l1189_118912


namespace Chicago_White_Sox_loss_l1189_118952

theorem Chicago_White_Sox_loss :
  ∃ (L : ℕ), (99 = L + 36) ∧ (L = 63) :=
by
  sorry

end Chicago_White_Sox_loss_l1189_118952


namespace max_value_of_function_l1189_118968

noncomputable def max_value (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem max_value_of_function : 
  ∀ x : ℝ, (- (Real.pi / 2)) ≤ x ∧ x ≤ 0 → max_value x ≤ 2 :=
sorry

end max_value_of_function_l1189_118968


namespace leaf_raking_earnings_l1189_118908

variable {S M L P : ℕ}

theorem leaf_raking_earnings (h1 : 5 * 4 + 7 * 2 + 10 * 1 + 3 * 1 = 47)
                             (h2 : 5 * 2 + 3 * 1 + 7 * 1 + 10 * 2 = 40)
                             (h3 : 163 - 87 = 76) :
  5 * S + 7 * M + 10 * L + 3 * P = 76 :=
by
  sorry

end leaf_raking_earnings_l1189_118908


namespace max_distance_convoy_l1189_118995

structure Vehicle :=
  (mpg : ℝ) (min_gallons : ℝ)

def SUV : Vehicle := ⟨12.2, 10⟩
def Sedan : Vehicle := ⟨52, 5⟩
def Motorcycle : Vehicle := ⟨70, 2⟩

def total_gallons : ℝ := 21

def total_distance (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) : ℝ :=
  SUV.mpg * SUV_gallons + Sedan.mpg * Sedan_gallons + Motorcycle.mpg * Motorcycle_gallons

theorem max_distance_convoy (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) :
  SUV_gallons + Sedan_gallons + Motorcycle_gallons = total_gallons →
  SUV_gallons >= SUV.min_gallons →
  Sedan_gallons >= Sedan.min_gallons →
  Motorcycle_gallons >= Motorcycle.min_gallons →
  total_distance SUV_gallons Sedan_gallons Motorcycle_gallons = 802 :=
sorry

end max_distance_convoy_l1189_118995


namespace simplify_and_evaluate_l1189_118949

-- Definitions of given conditions
def a := 1
def b := 2

-- Statement of the theorem
theorem simplify_and_evaluate : (a * b + (a^2 - a * b) - (a^2 - 2 * a * b) = 4) :=
by
  -- Using sorry to indicate the proof is to be completed
  sorry

end simplify_and_evaluate_l1189_118949


namespace sum_of_solutions_l1189_118996

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l1189_118996


namespace abs_expression_eq_five_l1189_118904

theorem abs_expression_eq_five : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry -- proof omitted

end abs_expression_eq_five_l1189_118904


namespace speed_of_stream_l1189_118999

theorem speed_of_stream (vs : ℝ) (h : ∀ (d : ℝ), d / (57 - vs) = 2 * (d / (57 + vs))) : vs = 19 :=
by
  sorry

end speed_of_stream_l1189_118999


namespace count_perfect_squares_diff_two_consecutive_squares_l1189_118955

theorem count_perfect_squares_diff_two_consecutive_squares:
  (∃ n : ℕ, n = 71 ∧ 
            ∀ a : ℕ, (a < 20000 → 
            (∃ b : ℕ, a^2 = (b+1)^2 - b^2))) :=
sorry

end count_perfect_squares_diff_two_consecutive_squares_l1189_118955


namespace vectors_parallel_l1189_118923

theorem vectors_parallel (m n : ℝ) (k : ℝ) (h1 : 2 = k * 1) (h2 : -1 = k * m) (h3 : 2 = k * n) : 
  m + n = 1 / 2 := 
by
  sorry

end vectors_parallel_l1189_118923


namespace odd_function_neg_value_l1189_118977

theorem odd_function_neg_value
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  ∀ x, x < 0 → f x = -x^2 + 2 * x :=
by
  intros x hx
  -- The proof would go here
  sorry

end odd_function_neg_value_l1189_118977


namespace is_decreasing_on_interval_l1189_118992

open Set Real

def f (x : ℝ) : ℝ := x^3 - x^2 - x

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem is_decreasing_on_interval :
  ∀ x ∈ Ioo (-1 / 3 : ℝ) 1, f' x < 0 :=
by
  intro x hx
  sorry

end is_decreasing_on_interval_l1189_118992


namespace average_age_choir_l1189_118947

theorem average_age_choir (S_f S_m S_total : ℕ) (avg_f : ℕ) (avg_m : ℕ) (females males total : ℕ)
  (h1 : females = 8) (h2 : males = 12) (h3 : total = 20)
  (h4 : avg_f = 25) (h5 : avg_m = 40)
  (h6 : S_f = avg_f * females) 
  (h7 : S_m = avg_m * males) 
  (h8 : S_total = S_f + S_m) :
  (S_total / total) = 34 := by
  sorry

end average_age_choir_l1189_118947


namespace difference_divisible_l1189_118939

theorem difference_divisible (a b n : ℕ) (h : n % 2 = 0) (hab : a + b = 61) :
  (47^100 - 14^100) % 61 = 0 := by
  sorry

end difference_divisible_l1189_118939


namespace largest_k_l1189_118983

-- Define the system of equations and conditions
def system_valid (x y k : ℝ) : Prop := 
  2 * x + y = k ∧ 
  3 * x + y = 3 ∧ 
  x - 2 * y ≥ 1

-- Define the proof problem as a theorem in Lean
theorem largest_k (x y : ℝ) :
  ∀ k : ℝ, system_valid x y k → k ≤ 2 := 
sorry

end largest_k_l1189_118983


namespace jane_book_pages_l1189_118971

theorem jane_book_pages (x : ℝ) :
  (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20) - (1 / 2 * (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20)) + 25) = 75) → x = 380 :=
by
  sorry

end jane_book_pages_l1189_118971


namespace tenth_number_in_twentieth_row_l1189_118967

def arrangement : ∀ n : ℕ, ℕ := -- A function defining the nth number in the sequence.
  sorry

-- A function to get the nth number in the mth row, respecting the arithmetic sequence property.
def number_in_row (m n : ℕ) : ℕ := 
  sorry

theorem tenth_number_in_twentieth_row : number_in_row 20 10 = 426 :=
  sorry

end tenth_number_in_twentieth_row_l1189_118967


namespace solve_quadratic_solve_inequalities_l1189_118928
open Classical

-- Define the equation for Part 1
theorem solve_quadratic (x : ℝ) : x^2 - 6 * x + 5 = 0 → (x = 1 ∨ x = 5) :=
by
  sorry

-- Define the inequalities for Part 2
theorem solve_inequalities (x : ℝ) : (x + 3 > 0) ∧ (2 * (x - 1) < 4) → (-3 < x ∧ x < 3) :=
by
  sorry

end solve_quadratic_solve_inequalities_l1189_118928


namespace probability_of_selection_l1189_118941

theorem probability_of_selection : 
  ∀ (n k : ℕ), n = 121 ∧ k = 20 → (P : ℚ) = 20 / 121 :=
by
  intros n k h
  sorry

end probability_of_selection_l1189_118941


namespace solve_for_x_l1189_118997

theorem solve_for_x (x : ℝ) : 45 - 5 = 3 * x + 10 → x = 10 :=
by
  sorry

end solve_for_x_l1189_118997


namespace conic_sections_parabolas_l1189_118962

theorem conic_sections_parabolas (x y : ℝ) :
  (y^6 - 9*x^6 = 3*y^3 - 1) → 
  ((y^3 = 3*x^3 + 1) ∨ (y^3 = -3*x^3 + 1)) := 
by 
  sorry

end conic_sections_parabolas_l1189_118962


namespace winner_collected_l1189_118911

variable (M : ℕ)
variable (last_year_rate this_year_rate : ℝ)
variable (extra_miles : ℕ)
variable (money_collected_last_year money_collected_this_year : ℝ)

axiom rate_last_year : last_year_rate = 4
axiom rate_this_year : this_year_rate = 2.75
axiom extra_miles_eq : extra_miles = 5

noncomputable def money_eq (M : ℕ) : ℝ :=
  last_year_rate * M

theorem winner_collected :
  ∃ M : ℕ, money_eq M = 44 :=
by
  sorry

end winner_collected_l1189_118911


namespace sqrt_product_simplification_l1189_118953

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end sqrt_product_simplification_l1189_118953


namespace zero_points_product_l1189_118926

noncomputable def f (a x : ℝ) : ℝ := abs (Real.log x / Real.log a) - (1 / 2) ^ x

theorem zero_points_product (a x1 x2 : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (hx1_zero : f a x1 = 0) (hx2_zero : f a x2 = 0) : 0 < x1 * x2 ∧ x1 * x2 < 1 :=
by
  sorry

end zero_points_product_l1189_118926


namespace parallel_lines_condition_l1189_118914

theorem parallel_lines_condition (a : ℝ) : 
  (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → y = 1 + x) := 
sorry

end parallel_lines_condition_l1189_118914


namespace problem_l1189_118906

theorem problem (k : ℕ) (h1 : 30^k ∣ 929260) : 3^k - k^3 = 2 :=
sorry

end problem_l1189_118906


namespace number_of_bags_of_chips_l1189_118957

theorem number_of_bags_of_chips (friends : ℕ) (amount_per_friend : ℕ) (cost_per_bag : ℕ) (total_amount : ℕ) (number_of_bags : ℕ) : 
  friends = 3 → amount_per_friend = 5 → cost_per_bag = 3 → total_amount = friends * amount_per_friend → number_of_bags = total_amount / cost_per_bag → number_of_bags = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_bags_of_chips_l1189_118957


namespace shorter_leg_of_right_triangle_l1189_118907

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l1189_118907


namespace find_divided_number_l1189_118993

-- Declare the constants and assumptions
variables (d q r : ℕ)
variables (n : ℕ)
variables (h_d : d = 20)
variables (h_q : q = 6)
variables (h_r : r = 2)
variables (h_def : n = d * q + r)

-- State the theorem we want to prove
theorem find_divided_number : n = 122 :=
by
  sorry

end find_divided_number_l1189_118993


namespace football_basketball_problem_l1189_118946

theorem football_basketball_problem :
  ∃ (football_cost basketball_cost : ℕ),
    (3 * football_cost + basketball_cost = 230) ∧
    (2 * football_cost + 3 * basketball_cost = 340) ∧
    football_cost = 50 ∧
    basketball_cost = 80 ∧
    ∃ (basketballs footballs : ℕ),
      (basketballs + footballs = 20) ∧
      (footballs < basketballs) ∧
      (80 * basketballs + 50 * footballs ≤ 1400) ∧
      ((basketballs = 11 ∧ footballs = 9) ∨
       (basketballs = 12 ∧ footballs = 8) ∨
       (basketballs = 13 ∧ footballs = 7)) :=
by
  sorry

end football_basketball_problem_l1189_118946


namespace apples_fell_out_l1189_118936

theorem apples_fell_out (initial_apples stolen_apples remaining_apples : ℕ) 
  (h₁ : initial_apples = 79) 
  (h₂ : stolen_apples = 45) 
  (h₃ : remaining_apples = 8) 
  : initial_apples - stolen_apples - remaining_apples = 26 := by
  sorry

end apples_fell_out_l1189_118936


namespace smallest_m_l1189_118969

theorem smallest_m (m : ℤ) (h : 2 * m + 1 ≥ 0) : m ≥ 0 :=
sorry

end smallest_m_l1189_118969


namespace fraction_spent_on_food_l1189_118909

theorem fraction_spent_on_food (r c f : ℝ) (l s : ℝ)
  (hr : r = 1/10)
  (hc : c = 3/5)
  (hl : l = 16000)
  (hs : s = 160000)
  (heq : f * s + r * s + c * s + l = s) :
  f = 1/5 :=
by
  sorry

end fraction_spent_on_food_l1189_118909


namespace f_12_16_plus_f_16_12_l1189_118948

noncomputable def f : ℕ × ℕ → ℕ :=
sorry

axiom ax1 : ∀ (x : ℕ), f (x, x) = x
axiom ax2 : ∀ (x y : ℕ), f (x, y) = f (y, x)
axiom ax3 : ∀ (x y : ℕ), (x + y) * f (x, y) = y * f (x, x + y)

theorem f_12_16_plus_f_16_12 : f (12, 16) + f (16, 12) = 96 :=
by sorry

end f_12_16_plus_f_16_12_l1189_118948


namespace solve_for_b_l1189_118963

noncomputable def P (x a b d c : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + d * x + c

theorem solve_for_b (a b d c : ℝ) (h1 : -a = d) (h2 : d = 1 + a + b + d + c) (h3 : c = 8) :
    b = -17 :=
by
  sorry

end solve_for_b_l1189_118963


namespace wrestler_teams_possible_l1189_118980

theorem wrestler_teams_possible :
  ∃ (team1 team2 team3 : Finset ℕ),
  (team1 ∪ team2 ∪ team3 = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (team1 ∩ team2 = ∅) ∧ (team1 ∩ team3 = ∅) ∧ (team2 ∩ team3 = ∅) ∧
  (team1.card = 3) ∧ (team2.card = 3) ∧ (team3.card = 3) ∧
  (team1.sum id = 15) ∧ (team2.sum id = 15) ∧ (team3.sum id = 15) ∧
  (∀ x ∈ team1, ∀ y ∈ team2, x > y) ∧
  (∀ x ∈ team2, ∀ y ∈ team3, x > y) ∧
  (∀ x ∈ team3, ∀ y ∈ team1, x > y) := sorry

end wrestler_teams_possible_l1189_118980


namespace not_always_divisible_by_40_l1189_118960

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem not_always_divisible_by_40 (p : ℕ) (hp_prime : is_prime p) (hp_geq7 : p ≥ 7) : ¬ (∀ p : ℕ, is_prime p ∧ p ≥ 7 → 40 ∣ (p^2 - 1)) := 
sorry

end not_always_divisible_by_40_l1189_118960


namespace distance_swam_against_current_l1189_118916

def swimming_speed_in_still_water : ℝ := 4
def speed_of_current : ℝ := 2
def time_taken_against_current : ℝ := 5

theorem distance_swam_against_current : ∀ distance : ℝ,
  (distance = (swimming_speed_in_still_water - speed_of_current) * time_taken_against_current) → distance = 10 :=
by
  intros distance h
  sorry

end distance_swam_against_current_l1189_118916


namespace day_100_M_minus_1_is_Tuesday_l1189_118956

variable {M : ℕ}

-- Given conditions
def day_200_M_is_Monday (M : ℕ) : Prop :=
  ((200 % 7) = 6)

def day_300_M_plus_2_is_Monday (M : ℕ) : Prop :=
  ((300 % 7) = 6)

-- Statement to prove
theorem day_100_M_minus_1_is_Tuesday (M : ℕ) 
  (h1 : day_200_M_is_Monday M) 
  (h2 : day_300_M_plus_2_is_Monday M) 
  : (((100 + (365 - 200)) % 7 + 7 - 1) % 7 = 2) :=
sorry

end day_100_M_minus_1_is_Tuesday_l1189_118956


namespace no_perfect_square_integers_l1189_118986

open Nat

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 10 * x^2 + 4 * x + 29

theorem no_perfect_square_integers : ∀ x : ℤ, ¬∃ a : ℤ, Q x = a^2 :=
by
  sorry

end no_perfect_square_integers_l1189_118986


namespace original_number_q_l1189_118905

variables (q : ℝ) (a b c : ℝ)
 
theorem original_number_q : 
  (a = 1.125 * q) → (b = 0.75 * q) → (c = 30) → (a - b = c) → q = 80 :=
by
  sorry

end original_number_q_l1189_118905


namespace inequality_l1189_118903

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.pi)
noncomputable def c : ℝ := Real.log 0.5 / Real.log 2

theorem inequality (h1: a = Real.sqrt 2) (h2: b = Real.log 3 / Real.log Real.pi) (h3: c = Real.log 0.5 / Real.log 2) : a > b ∧ b > c := 
by 
  sorry

end inequality_l1189_118903


namespace ratio_of_area_to_perimeter_l1189_118990

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l1189_118990


namespace quadratic_complete_square_l1189_118940

theorem quadratic_complete_square :
  ∃ a b c : ℤ, (8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) ∧ (a + b + c = -387) :=
sorry

end quadratic_complete_square_l1189_118940


namespace value_of_k_l1189_118985

def f (x : ℝ) := 4 * x ^ 2 - 5 * x + 6
def g (x : ℝ) (k : ℝ) := 2 * x ^ 2 - k * x + 1

theorem value_of_k :
  (f 5 - g 5 k = 30) → k = -10 := 
by 
  sorry

end value_of_k_l1189_118985


namespace sufficient_but_not_necessary_condition_l1189_118961

theorem sufficient_but_not_necessary_condition (b c : ℝ) :
  (∃ x0 : ℝ, (x0^2 + b * x0 + c) < 0) ↔ (c < 0) ∨ true :=
sorry

end sufficient_but_not_necessary_condition_l1189_118961


namespace range_of_b_l1189_118929

open Real

theorem range_of_b (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → abs (y - (x + b)) = 1) ↔ -sqrt 2 < b ∧ b < sqrt 2 := 
by sorry

end range_of_b_l1189_118929


namespace contrapositive_example_l1189_118924

theorem contrapositive_example (x : ℝ) : (x > 2 → x^2 > 4) → (x^2 ≤ 4 → x ≤ 2) :=
by
  sorry

end contrapositive_example_l1189_118924


namespace sin_theta_value_l1189_118994

theorem sin_theta_value (f : ℝ → ℝ)
  (hx : ∀ x, f x = 3 * Real.sin x - 8 * Real.cos (x / 2) ^ 2)
  (h_cond : ∀ x, f x ≤ f θ) : Real.sin θ = 3 / 5 := 
sorry

end sin_theta_value_l1189_118994


namespace general_term_defines_sequence_l1189_118910

/-- Sequence definition -/
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (2 * a n + 6) / (a n + 1)

/-- General term formula -/
def general_term (n : ℕ) : ℚ :=
  (3 * 4 ^ n + 2 * (-1) ^ n) / (4 ^ n - (-1) ^ n)

/-- Theorem stating that the general term formula defines the sequence -/
theorem general_term_defines_sequence : ∀ (a : ℕ → ℚ), seq a → ∀ n, a n = general_term n :=
by
  intros a h_seq n
  sorry

end general_term_defines_sequence_l1189_118910
