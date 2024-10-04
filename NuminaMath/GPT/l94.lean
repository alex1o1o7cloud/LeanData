import Mathlib

namespace f_periodic_f_central_sym_f_defined_for_interval_find_f_2011_l94_94039

noncomputable def f : ℝ → ℝ :=
sorry

theorem f_periodic : ∀ x, f(4 - x) = f(x) :=
sorry

theorem f_central_sym : ∀ x, f(2 - x) = -f(x) :=
sorry

theorem f_defined_for_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f(x) = x - 1 :=
sorry

theorem find_f_2011 : f 2011 = 0 :=
by
  -- Use the given properties to derive the result
  sorry

end f_periodic_f_central_sym_f_defined_for_interval_find_f_2011_l94_94039


namespace find_line_equation_l94_94377

-- Define slope of given line
def slope_given : ℝ := (√2) / 2

-- Define point through which the new line passes
def point_through : ℝ × ℝ := (-1, 1)

-- Define desired slope as twice the slope of the given line
def desired_slope : ℝ := 2 * slope_given

-- Define the desired line's equation
def desired_line (x y : ℝ) : Prop :=
  y - 1 = desired_slope * (x + 1)

-- The theorem statement to prove
theorem find_line_equation : ∃ (x y : ℝ), desired_line x y :=
  exists.intro (-1) (1) sorry

end find_line_equation_l94_94377


namespace cos_angle_BAC_proof_l94_94234

open EuclideanGeometry

variables {A B C O M : Point}
variables [EuclideanSpace V]

-- Define the condition that O is the center of the circumcircle of triangle ABC
def is_circumcenter (O A B C : Point) : Prop := 
  dist O A = dist O B ∧ dist O B = dist O C

-- Define the condition involving vector relationships
def vector_condition (A O B C : Point) : Prop := 
  (2 : ℚ) • (A -ᵥ O) = 4 / 5 • ((B -ᵥ A) + (C -ᵥ A))

-- Define the cosine of the angle BAC
noncomputable def cos_angle_BAC (A B C O : Point) : ℚ :=
cos (∠ B A C)

-- The final Lean 4 Statement
theorem cos_angle_BAC_proof (A B C O : Point) (h_circumcenter : is_circumcenter O A B C) 
  (h_vector : vector_condition A O B C) : cos_angle_BAC A B C O = 1 / 4 := 
by sorry

end cos_angle_BAC_proof_l94_94234


namespace exists_route_within_republic_l94_94924

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l94_94924


namespace airline_route_within_republic_l94_94905

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l94_94905


namespace smallest_multiple_of_84_with_6_and_7_l94_94452

variable (N : Nat)

def is_multiple_of_84 (N : Nat) : Prop :=
  N % 84 = 0

def consists_of_6_and_7 (N : Nat) : Prop :=
  ∀ d ∈ N.digits 10, d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  ∃ N, is_multiple_of_84 N ∧ consists_of_6_and_7 N ∧ ∀ M, is_multiple_of_84 M ∧ consists_of_6_and_7 M → N ≤ M := 
sorry

end smallest_multiple_of_84_with_6_and_7_l94_94452


namespace abs_expression_evaluation_l94_94366

theorem abs_expression_evaluation (x : ℝ) (hx : x > 2) : 
  |2 - |x^2 - 3 * x + 1| | = x^2 - 3 * x - 1 := 
sorry

end abs_expression_evaluation_l94_94366


namespace village_connection_possible_l94_94271

variable (V : Type) -- Type of villages
variable (Villages : List V) -- List of 26 villages
variable (connected_by_tractor connected_by_train : V → V → Prop) -- Connections

-- Define the hypothesis
variable (bidirectional_connections : ∀ (v1 v2 : V), v1 ≠ v2 → (connected_by_tractor v1 v2 ∨ connected_by_train v1 v2))

-- Main theorem statement
theorem village_connection_possible :
  ∃ (mode : V → V → Prop), (∀ v1 v2 : V, v1 ≠ v2 → v1 ∈ Villages → v2 ∈ Villages → mode v1 v2) ∧
  (∀ v1 v2 : V, v1 ∈ Villages → v2 ∈ Villages → ∃ (path : List (V × V)), (∀ edge ∈ path, mode edge.fst edge.snd) ∧ path ≠ []) :=
by
  sorry

end village_connection_possible_l94_94271


namespace proof_problem_l94_94871

noncomputable def neither_prime_nor_composite_probability : ℝ := 0.01

def total_pieces_of_paper : ℕ := 100

def neither_prime_nor_composite_number : ℕ := 1

theorem proof_problem (pieces : fin total_pieces_of_paper → ℕ)
  (h : (∃ i : fin total_pieces_of_paper, pieces i = neither_prime_nor_composite_number) ∧ 
       ∀ n, n ≠ neither_prime_nor_composite_number → 
            (Prime n ∨ Composite n)) :
  (neither_prime_nor_composite_probability = 1 / total_pieces_of_paper.toReal) := 
sorry

end proof_problem_l94_94871


namespace t_over_s_possible_values_l94_94972

-- Define the initial conditions
variables (n : ℕ) (h : n ≥ 3)

-- The theorem statement
theorem t_over_s_possible_values (s t : ℕ) (h_s : s > 0) (h_t : t > 0) : 
  (∃ r : ℚ, r = t / s ∧ 1 ≤ r ∧ r < (n - 1)) :=
sorry

end t_over_s_possible_values_l94_94972


namespace length_of_smallest_repeating_block_in_decimal_expansion_of_11_over_13_l94_94612

theorem length_of_smallest_repeating_block_in_decimal_expansion_of_11_over_13 : 
  ∀ (d : ℚ), d = 11 / 13 → repeating_decimal_length d = 6 :=
by
  intro d h
  -- proof goes here
  sorry

end length_of_smallest_repeating_block_in_decimal_expansion_of_11_over_13_l94_94612


namespace range_of_alpha_l94_94104

noncomputable def f (x : ℝ) (α : ℝ) : ℝ :=
  if x ≤ -3/2 * Real.sin α then -2 * x - 4 * Real.sin α
  else if -3/2 * Real.sin α < x ∧ x < -1/2 * Real.sin α then -Real.sin α
  else if -1/2 * Real.sin α ≤ x ∧ x ≤ 0 then 2 * x
  else 
  |x + 1/2 * Real.sin α| + |x + 3/2 * Real.sin α| - 2 * Real.sin α

theorem range_of_alpha (α : ℝ) (h1 : α ∈ Ioo (-π) π) (h2 : ∀ x, f x α ≥ f (x + 2) α) :
  α ∈ Icc 0 (π / 6) ∪ Ico (5 * π / 6) π :=
sorry

end range_of_alpha_l94_94104


namespace log_inequality_region_l94_94059

theorem log_inequality_region (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x ≠ 1) (hx2 : x ≠ y) :
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) 
  ∨ (1 < x ∧ y > x) ↔ (Real.log y / Real.log x ≥ Real.log (x * y) / Real.log (x / y)) :=
  sorry

end log_inequality_region_l94_94059


namespace visible_black_area_ratio_l94_94991

-- Definitions for circle areas as nonnegative real numbers
variables (A_b A_g A_w : ℝ) (hA_b : 0 ≤ A_b) (hA_g : 0 ≤ A_g) (hA_w : 0 ≤ A_w)
-- Condition: Initial visible black area is 7 times the white area
axiom initial_visible_black_area : 7 * A_w = A_b

-- Definition of new visible black area after movement
def new_visible_black_area := A_b - A_w

-- Prove the ratio of the visible black regions before and after moving the circles
theorem visible_black_area_ratio :
  (7 * A_w) / ((7 * A_w) - A_w) = 7 / 6 :=
by { sorry }

end visible_black_area_ratio_l94_94991


namespace sasha_can_get_sqrt_l94_94552

noncomputable def canGetSqrt (s : ℝ) (hs : s > 0) : Prop :=
  ∃ (seq : List ℝ), (seq.head = s) ∧ (seq.contains (Real.sqrt s)) ∧
  (∀ a b, (seq.contains a) → (seq.contains b) → 
     (seq.contains (a + 1)) ∧ ∃ x1 x2, (seq.contains x1) ∧ (seq.contains x2) ∧
       (x1 * x1 + a * x1 + b = 0) ∧ (x2 * x2 + a * x2 + b = 0))

theorem sasha_can_get_sqrt (s : ℝ) (hs : s > 0) : canGetSqrt s hs :=
sorry

end sasha_can_get_sqrt_l94_94552


namespace value_of_m_range_of_n_value_of_k_l94_94436

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m+1)^2 * x^(m^2 - m - 4)

-- Part 1: Prove the value of m 
theorem value_of_m (h : ∀ x : ℝ, x > 0 → f m x ≥ 0) : m = -2 := 
sorry

-- Part 2: Prove the range of n
def g (x n : ℝ) : ℝ := 2 * x + n
def A : set ℝ := {y : ℝ | ∃ x : ℝ, x ∈ Icc (-1:ℝ) (3:ℝ) → y = x^2}
def B (n : ℝ) : set ℝ := {y : ℝ | ∃ x : ℝ, x ∈ Icc (-1:ℝ) (3:ℝ) → y = g x n}

theorem range_of_n (p q : set ℝ → Prop) (hA : p A) (hB : q B) (hyp : ∀ x, p {x ∈ A} → q {x ∈ B}) : n ∈ Icc 2 3 :=
sorry

-- Part 3: Prove the value of k
noncomputable def F (k x : ℝ) : ℝ := x^2 - k*x + (1-k)*(1+k)

theorem value_of_k (h : ∀ x ∈ Icc 0 2, F k x ≥ -2) : k = -real.sqrt 3 ∨ k = (2 * real.sqrt 15) / 5 :=
sorry

end value_of_m_range_of_n_value_of_k_l94_94436


namespace perimeter_triangle_ABF2_is_6_l94_94089
open Real

noncomputable def ellipse_perimeter_of_triangle_ABF2
  (b : ℝ) (e : ℝ) (F1 F2 : Point) (A B : Point)
  (h_minor_axis : b = sqrt 5 / 2)
  (h_eccentricity : e = 2 / 3)
  (h_foci : F1 ≠ F2)
  (h_A_on_ellipse : (A = ellipse_point F1 F2 b) ∨ (A = ellipse_point F2 F1 b))
  (h_B_on_ellipse : (B = ellipse_point F1 F2 b) ∨ (B = ellipse_point F2 F1 b))
  : ℝ :=
  (|dist A F1 + dist A F2| + |dist B F1 + dist B F2|)

theorem perimeter_triangle_ABF2_is_6 :
  ellipse_perimeter_of_triangle_ABF2 (sqrt 5 / 2) (2 / 3) F1 F2 A B h_minor_axis h_eccentricity h_foci h_A_on_ellipse h_B_on_ellipse = 6 :=
sorry

end perimeter_triangle_ABF2_is_6_l94_94089


namespace small_bottles_needed_l94_94327

noncomputable def small_bottle_capacity := 40 -- in milliliters
noncomputable def large_bottle_capacity := 540 -- in milliliters
noncomputable def worst_case_small_bottle_capacity := 38 -- in milliliters

theorem small_bottles_needed :
  let n_bottles := Int.ceil (large_bottle_capacity / worst_case_small_bottle_capacity : ℚ)
  n_bottles = 15 :=
by
  sorry

end small_bottles_needed_l94_94327


namespace airline_route_within_republic_l94_94908

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l94_94908


namespace greatest_product_from_sum_2004_l94_94600

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l94_94600


namespace mad_hatter_wait_time_l94_94981

theorem mad_hatter_wait_time 
  (mad_hatter_fast_rate : ℝ := 15 / 60)
  (march_hare_slow_rate : ℝ := 10 / 60)
  (meeting_time : ℝ := 5) :
  let mad_hatter_real_time := meeting_time * (60 / (60 + mad_hatter_fast_rate * 60)),
      march_hare_real_time := meeting_time * (60 / (60 - march_hare_slow_rate * 60)),
      waiting_time := march_hare_real_time - mad_hatter_real_time
  in waiting_time = 2 :=
by 
  sorry

end mad_hatter_wait_time_l94_94981


namespace area_of_sector_l94_94821

def central_angle := 120 * Real.pi / 180
def radius := Real.sqrt 3

theorem area_of_sector : (1 / 2) * central_angle * radius * radius = Real.pi :=
by
  sorry

end area_of_sector_l94_94821


namespace binary_multiplication_correct_l94_94381

theorem binary_multiplication_correct :
  (110101₂ * 11101₂) = 10101110101₂ := 
sorry

end binary_multiplication_correct_l94_94381


namespace exists_route_within_same_republic_l94_94886

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l94_94886


namespace train_passing_time_correct_l94_94632

-- Definitions of the conditions
def length_of_train : ℕ := 180  -- Length of the train in meters
def speed_of_train_km_hr : ℕ := 54  -- Speed of the train in kilometers per hour

-- Known conversion factors
def km_per_hour_to_m_per_sec (v : ℕ) : ℚ := (v * 1000) / 3600

-- Define the speed of the train in meters per second
def speed_of_train_m_per_sec : ℚ := km_per_hour_to_m_per_sec speed_of_train_km_hr

-- Define the time to pass the oak tree
def time_to_pass_oak_tree (d : ℕ) (v : ℚ) : ℚ := d / v

-- The statement to prove
theorem train_passing_time_correct :
  time_to_pass_oak_tree length_of_train speed_of_train_m_per_sec = 12 := 
by
  sorry

end train_passing_time_correct_l94_94632


namespace greatest_product_two_integers_sum_2004_l94_94605

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l94_94605


namespace middle_number_of_ratio_l94_94152

theorem middle_number_of_ratio (x : ℝ) (h : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862) : 2 * x = 14 :=
sorry

end middle_number_of_ratio_l94_94152


namespace Coles_payment_l94_94029

-- Conditions
def side_length : ℕ := 9
def back_length : ℕ := 18
def cost_per_foot : ℕ := 3
def back_neighbor_contrib (length : ℕ) : ℕ := (length / 2) * cost_per_foot
def left_neighbor_contrib (length : ℕ) : ℕ := (length / 3) * cost_per_foot

-- Main problem statement
theorem Coles_payment : 
  let total_fence_length := 2 * side_length + back_length,
      total_cost := total_fence_length * cost_per_foot,
      contribution_back_neighbor := back_neighbor_contrib back_length,
      contribution_left_neighbor := left_neighbor_contrib side_length,
      coles_contribution := total_cost - (contribution_back_neighbor + contribution_left_neighbor)
  in coles_contribution = 72 := by
  sorry

end Coles_payment_l94_94029


namespace find_five_numbers_l94_94200

theorem find_five_numbers :
  ∃ (a d : ℚ), 
    let n₁ := a - d,
        n₂ := a,
        n₃ := a + d,
        n₄ := a + 2*d,
        n₅ := (a + 2*d)^2 / (a + d) in
    -- Condition 1: Arithmetic progression sum equals 40
    n₁ + n₂ + n₃ + n₄ = 40 ∧
    -- Condition 2: Geometric progression property
    n₃ * n₅ = 32 * n₂ ∧
    -- Verify the numbers found
    (n₁ = 4 ∧ n₂ = 8 ∧ n₃ = 12 ∧ n₄ = 16 ∧ n₅ = 64 / 3) :=
begin
  -- proof is not required
  sorry
end

end find_five_numbers_l94_94200


namespace f_2017_of_9_eq_8_l94_94520

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_k (k n : ℕ) : ℕ :=
  if k = 0 then n else f (f_k (k-1) n)

theorem f_2017_of_9_eq_8 : f_k 2017 9 = 8 := by
  sorry

end f_2017_of_9_eq_8_l94_94520


namespace even_product_probability_l94_94699

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a function to check if a product of two numbers is even
def is_even_product (a b : ℕ) : Prop :=
  (a * b) % 2 = 0

-- Define the main theorem
theorem even_product_probability :
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, p.fst ≠ p.snd ∧ is_even_product p.fst p.snd) (S.product S))) / (Finset.card (Finset.filter (λ p : ℕ × ℕ, p.fst ≠ p.snd) (S.product S))) = 4 / 5 := 
sorry

end even_product_probability_l94_94699


namespace boat_speed_in_still_water_l94_94667

-- Problem Definitions
def V_s : ℕ := 16
def t : ℕ := sorry -- t is arbitrary positive value
def V_b : ℕ := 48

-- Conditions
def upstream_time := 2 * t
def downstream_time := t
def upstream_distance := (V_b - V_s) * upstream_time
def downstream_distance := (V_b + V_s) * downstream_time

-- Proof Problem
theorem boat_speed_in_still_water :
  upstream_distance = downstream_distance → V_b = 48 :=
by sorry

end boat_speed_in_still_water_l94_94667


namespace fraction_of_girls_l94_94486

theorem fraction_of_girls (G B : ℕ) (h1 : G + B = 800) (h2 : 7 * G + 4 * B = 4700) : (G : ℚ) / (G + B) = 5 / 8 :=
by
  sorry

end fraction_of_girls_l94_94486


namespace nh4i_required_l94_94375

theorem nh4i_required (KOH NH4I NH3 KI H2O : ℕ) (h_eq : 1 * NH4I + 1 * KOH = 1 * NH3 + 1 * KI + 1 * H2O)
  (h_KOH : KOH = 3) : NH4I = 3 := 
by
  sorry

end nh4i_required_l94_94375


namespace length_of_park_l94_94332

variable (L : ℝ)
variable (width_park : ℝ) (area_lawn : ℝ) (width_road : ℝ)
variable (roads : ℕ)

#check width_park
#check area_lawn

theorem length_of_park
  (h_width_park : width_park = 40)
  (h_area_lawn : area_lawn = 2109)
  (h_width_road : width_road = 3)
  (h_roads : roads = 2)
  (h_condition : L * width_park = area_lawn + real.of_nat roads * L) :
  L = 55.5 :=
  sorry

end length_of_park_l94_94332


namespace simplify_expression_l94_94219

variable {a : ℝ}

theorem simplify_expression (h₁ : a ≠ 0) (h₂ : a ≠ -1) (h₃ : a ≠ 1) :
  ( ( (a^2 + 1) / a - 2 ) / ( (a^2 - 1) / (a^2 + a) ) ) = a - 1 :=
sorry

end simplify_expression_l94_94219


namespace gcf_factorial_seven_eight_l94_94760

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l94_94760


namespace sum_f_to_2015_l94_94652

noncomputable def f : ℝ → ℝ
| x if -3 ≤ x ∧ x < -1 := -(x+2)^2
| x if -1 ≤ x ∧ x < 3 := x
| x := f (x - 6 * (⌊x / 6⌋ + 1))

theorem sum_f_to_2015 :
  (∑ k in finset.range 2015, f (k + 1)) = 336 :=
sorry

end sum_f_to_2015_l94_94652


namespace coin_exchange_proof_l94_94017

/-- Prove the coin combination that Petya initially had -/
theorem coin_exchange_proof (x y z : ℕ) (hx : 20 * x + 15 * y + 10 * z = 125) : x = 0 ∧ y = 1 ∧ z = 11 :=
by
  sorry

end coin_exchange_proof_l94_94017


namespace man_speed_still_water_l94_94326

theorem man_speed_still_water :
  ∀ (v_m v_s : ℝ),
    v_m + v_s = 10 ∧ v_m - v_s = 6 →
    v_m = 8 :=
by
  intros v_m v_s h
  cases h with h1 h2
  have h3 : 2 * v_m = 16 := by linarith
  have h4 : v_m = 8 := by
    linarith
  exact h4

end man_speed_still_water_l94_94326


namespace good_order_number_reverse_l94_94086

-- Sequence is given and its reverse is required
variable {α : Type} [LinearOrder α]

def good_order_number (seq : List α) : Nat :=
  seq.enum.filter (λ ⟨i, ai⟩, seq.drop i.succ.enum.any (λ ⟨j, aj⟩, ai > aj)).length

theorem good_order_number_reverse
  (seq : List α) (h : good_order_number seq = 3) :
  good_order_number seq.reverse = 18 := 
sorry

end good_order_number_reverse_l94_94086


namespace valid_propositions_l94_94815

example (a b c : ℝ) : ¬(a > b → a * c^2 > b * c^2) :=
begin
  intro h,
  have hc0 : c = 0 := rfl,
  rw [hc0, mul_zero, mul_zero] at h,
  exact not_lt_of_le (le_refl 0) (h (by linarith)),
end

example (a b c : ℝ) : (a * c^2 > b * c^2) → a > b :=
by intro h; linarith [real.mul_pos hc hc, h]

example (a b c : ℝ) : (a > b) → (a * 2^c > b * 2^c) :=
by intro h; exact mul_lt_mul_of_pos_right h (real.rpow_pos_of_pos (by norm_num) c) 

theorem valid_propositions (a b c : ℝ) : ¬(a > b → a * c^2 > b * c^2) ∧ ((a * c^2 > b * c^2) → a > b) ∧ ((a > b) → (a * 2^c > b * 2^c)) :=
by split; sorry 

end valid_propositions_l94_94815


namespace initial_miles_correct_l94_94019

-- Definitions and conditions
def miles_per_gallon : ℕ := 30
def gallons_per_tank : ℕ := 20
def current_miles : ℕ := 2928
def tanks_filled : ℕ := 2

-- Question: How many miles were on the car before the road trip?
def initial_miles : ℕ := current_miles - (miles_per_gallon * gallons_per_tank * tanks_filled)

-- Proof problem statement
theorem initial_miles_correct : initial_miles = 1728 :=
by
  -- Here we expect the proof, but are skipping it with 'sorry'
  sorry

end initial_miles_correct_l94_94019


namespace temperature_increase_per_century_l94_94502

def total_temperature_change_over_1600_years : ℕ := 64
def years_in_a_century : ℕ := 100
def years_overall : ℕ := 1600

theorem temperature_increase_per_century :
  total_temperature_change_over_1600_years / (years_overall / years_in_a_century) = 4 := by
  sorry

end temperature_increase_per_century_l94_94502


namespace rationalize_t_l94_94453

theorem rationalize_t (t : ℝ) (a : ℝ) (b : ℝ)
  (h : t = a / (1 - b)) (b_eq : b = real.cbrt 3) 
  (a_eq : a = 1) : t = - (1 + b) * (1 + real.sqrt 3) := 
by 
  sorry

end rationalize_t_l94_94453


namespace abel_overtakes_kelly_l94_94154

theorem abel_overtakes_kelly
  (head_start : ℝ)
  (distance_lost_by : ℝ)
  (finish_line : ℝ)
  (kelly_distance_ran : ℝ := finish_line - head_start + distance_lost_by)
  (abel_distance_ran : ℝ := finish_line)
  (required_distance_to_overtake : ℝ := kelly_distance_ran + distance_lost_by) :
  abel_distance_ran - required_distance_to_overtake = 2 :=
by sorry

#eval abel_overtakes_kelly 3 0.5 100 -- Outputs: true (when proved)

end abel_overtakes_kelly_l94_94154


namespace integral_1_value_integral_2_value_l94_94694

noncomputable def integral_1 := ∫ x in -4..3, |x + 2|
noncomputable def integral_2 := ∫ x in 2..Real.exp 1 + 1, 1 / (x - 1)

theorem integral_1_value : integral_1 = 29 / 2 :=
by
  sorry

theorem integral_2_value : integral_2 = 1 :=
by
  sorry

end integral_1_value_integral_2_value_l94_94694


namespace gcd_of_8_and_12_l94_94228

theorem gcd_of_8_and_12 :
  let a := 8
  let b := 12
  let lcm_ab := 24
  Nat.lcm a b = lcm_ab → Nat.gcd a b = 4 :=
by
  intros
  sorry

end gcd_of_8_and_12_l94_94228


namespace number_of_valid_pairs_l94_94128

def S := {0, 1, 2, 3}

def op (i j : Nat) : Nat :=
  (i + j) % 4

def valid_pairs : Nat :=
  (S.to_finset.filter (λ i => 
    S.to_finset.filter (λ j => 
      op (op i i) j = 0)).card).sum

theorem number_of_valid_pairs :
  valid_pairs = 4 :=
by
  sorry

end number_of_valid_pairs_l94_94128


namespace coin_flip_probability_l94_94450

open Classical

noncomputable section

theorem coin_flip_probability :
  let total_outcomes := 2^10
  let exactly_five_heads_tails := Nat.choose 10 5 / total_outcomes
  let even_heads_probability := 1/2
  (even_heads_probability * (1 - exactly_five_heads_tails) / 2 = 193 / 512) :=
by
  sorry

end coin_flip_probability_l94_94450


namespace find_angle_A_find_max_perimeter_l94_94419

noncomputable def triangle (a b c : ℝ) := 
  ∃ (A B C : ℝ), 
    A + B + C = π ∧
    a = 2 * sin A ∧
    b = 2 * sin B ∧
    c = 2 * sin C

theorem find_angle_A 
  (a b c : ℝ) (A B C : ℝ)
  (h₁ : A + B + C = π)
  (h₂ : a = 2 * sin A)
  (h₃ : b = 2 * sin B)
  (h₄ : c = 2 * sin C)
  (h₅ : (sin A - sin B + sin C) / sin C = b / (a + b - c)) :
  A = π / 3 :=
sorry

theorem find_max_perimeter 
  (a b c : ℝ) (A B C : ℝ)
  (h₀ : a + b + c ≤ 6)
  (h₁ : a = b = c) 
  (h₂ : a = 2 * sin A)
  (h₃ : circumradius 1 a A) :
  a + b + c ≤ 3 * sqrt 3 :=
sorry

end find_angle_A_find_max_perimeter_l94_94419


namespace prob_three_tails_one_head_in_four_tosses_l94_94471

open Nat

-- Definitions
def coin_toss_prob (n : ℕ) (k : ℕ) : ℚ :=
  (real.to_rat (choose n k) / real.to_rat (2^n))

-- Parameters: n = 4 (number of coin tosses), k = 3 (number of tails)
def four_coins_three_tails_prob : ℚ := coin_toss_prob 4 3

-- Theorem: The probability of getting exactly three tails and one head in four coin tosses is 1/4.
theorem prob_three_tails_one_head_in_four_tosses : four_coins_three_tails_prob = 1/4 := by
  sorry

end prob_three_tails_one_head_in_four_tosses_l94_94471


namespace range_of_a_for_monotonic_f_l94_94473

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a^2 * x^2 + a * x

theorem range_of_a_for_monotonic_f (a : ℝ) : 
  (∀ x, 1 < x → f a x ≤ f a (1 : ℝ)) ↔ (a ≤ -1 / 2 ∨ 1 ≤ a) := 
by
  sorry

end range_of_a_for_monotonic_f_l94_94473


namespace rational_lines_intersect_all_sets_largest_r_for_rational_lines_l94_94641

-- Part 1: 
theorem rational_lines_intersect_all_sets :
  ∃ (A : set (set (ℚ × ℚ))), 
    (∀ a ∈ A, infinite a) ∧ 
    (pairwise (disjoint : set (ℚ × ℚ) → set (ℚ × ℚ) → Prop) A) ∧
    (∃ (L : set (ℚ × ℚ)), 
      (∃ (p1 p2 : ℚ × ℚ), p1 ∈ L ∧ p2 ∈ L ∧ p1 ≠ p2) →
      ∀ a ∈ A, ∃ p ∈ a, p ∈ L) := sorry

-- Part 2: 
theorem largest_r_for_rational_lines : 
  ∃ (r : ℕ), r = 3 ∧ 
    (∀ (A : set (set (ℚ × ℚ))),
      (∀ a ∈ A, infinite a) ∧ 
      (pairwise (disjoint : set (ℚ × ℚ) → set (ℚ × ℚ) → Prop) A) ∧
      (card A = 100) →
      (∃ (L : set (ℚ × ℚ)), ∃ (s : finset (set (ℚ × ℚ))), 
        (s ⊆ A) ∧ (finset.card s ≥ r) ∧ ∀ a ∈ s, ∃ p ∈ a, p ∈ L)) := sorry

end rational_lines_intersect_all_sets_largest_r_for_rational_lines_l94_94641


namespace factorial_ratio_integer_l94_94549

theorem factorial_ratio_integer (m n : ℕ) : 
    (m ≥ 0) → (n ≥ 0) → ∃ k : ℤ, k = (2 * m).factorial * (2 * n).factorial / ((m.factorial * n.factorial * (m + n).factorial) : ℝ) :=
by
  sorry

end factorial_ratio_integer_l94_94549


namespace total_points_correct_l94_94203

variable (H Q T : ℕ)

-- Given conditions
def hw_points : ℕ := 40
def quiz_points := hw_points + 5
def test_points := 4 * quiz_points

-- Question: Prove the total points assigned are 265
theorem total_points_correct :
  H = hw_points →
  Q = quiz_points →
  T = test_points →
  H + Q + T = 265 :=
by
  intros h_hw h_quiz h_test
  rw [h_hw, h_quiz, h_test]
  exact sorry

end total_points_correct_l94_94203


namespace james_recovery_time_l94_94506

theorem james_recovery_time :
  ∀ (initial_healing_time graft_healing_factor : ℕ),
  initial_healing_time = 4 →
  graft_healing_factor = 3/2 → -- 50% longer represented as multiplying by 1.5
  let graft_healing_time := initial_healing_time * (3/2)
  in let total_recovery_time := initial_healing_time + graft_healing_time
  in total_recovery_time = 10 :=
by
  intros initial_healing_time graft_healing_factor h₁ h₂,
  rw [h₁, h₂],
  let graft_healing_time := 4 * 3/2,
  have h₃ : graft_healing_time = 6, by sorry,
  let total_recovery_time := 4 + 6,
  exact eq.refl 10

end james_recovery_time_l94_94506


namespace probability_of_selecting_specific_letters_l94_94684

theorem probability_of_selecting_specific_letters :
  let total_cards := 15
  let amanda_cards := 6
  let chloe_or_ethan_cards := 9
  let prob_amanda_then_chloe_or_ethan := (amanda_cards / total_cards) * (chloe_or_ethan_cards / (total_cards - 1))
  let prob_chloe_or_ethan_then_amanda := (chloe_or_ethan_cards / total_cards) * (amanda_cards / (total_cards - 1))
  let total_prob := prob_amanda_then_chloe_or_ethan + prob_chloe_or_ethan_then_amanda
  total_prob = 18 / 35 :=
by
  sorry

end probability_of_selecting_specific_letters_l94_94684


namespace min_value_2x_minus_y_l94_94150

theorem min_value_2x_minus_y :
  ∃ (x y : ℝ), (y = abs (x - 1) ∨ y = 2) ∧ (y ≤ 2) ∧ (2 * x - y = -4) :=
by
  sorry

end min_value_2x_minus_y_l94_94150


namespace probability_sum_is_five_l94_94394
open Finset

-- Definitions for the set and conditions
def S : Finset ℕ := {0, 1, 2, 3}
def non_empty_subsets (S : Finset ℕ) : Finset (Finset ℕ) := S.powerset.filter (λ s, ¬ s.isEmpty)

-- The main statement we want to prove
theorem probability_sum_is_five :
  (non_empty_subsets S).card = 15 → 
  (filter (λ s : Finset ℕ, s.sum id = 5) (non_empty_subsets S)).card = 2 → 
  (filter (λ s : Finset ℕ, s.sum id = 5) (non_empty_subsets S)).card * 1 / (non_empty_subsets S).card = (2 : ℚ) / 15 :=
by sorry

end probability_sum_is_five_l94_94394


namespace smallest_positive_integer_cube_ends_544_l94_94382

theorem smallest_positive_integer_cube_ends_544 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 544 → m ≥ n :=
by
  sorry

end smallest_positive_integer_cube_ends_544_l94_94382


namespace tom_sara_age_problem_l94_94589

-- Define the given conditions as hypotheses and variables
variables (t s : ℝ)
variables (h1 : t - 3 = 2 * (s - 3))
variables (h2 : t - 8 = 3 * (s - 8))

-- Lean statement of the problem
theorem tom_sara_age_problem :
  ∃ x : ℝ, (t + x) / (s + x) = 3 / 2 ∧ x = 7 :=
by
  sorry

end tom_sara_age_problem_l94_94589


namespace find_quadrant_l94_94820

theorem find_quadrant (α : ℝ) (h1 : tan α < 0) (h2 : cos α < 0) : α ∈ (II) :=
sorry

end find_quadrant_l94_94820


namespace math_problem_equivalent_l94_94300

noncomputable def f (x : ℝ) : ℝ := sin (x + Real.pi / 4) + cos (x - Real.pi / 4)
def alpha_beta_conditions (α β : ℝ) : Prop :=
  (cos (β - α) = 1 / 2) ∧
  (cos (β + α) = -1 / 2) ∧
  (0 < α) ∧ (α < β) ∧ (β ≤ (Real.pi / 2))

theorem math_problem_equivalent (α β : ℝ) (h : alpha_beta_conditions α β) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f(x + T) = f(x) ∧ T = 2 * Real.pi) ∧
  (∀ x : ℝ, f(x) ≥ -2) ∧
  (β = Real.pi / 2 → f(β) = -Real.sqrt 2) :=
sorry

end math_problem_equivalent_l94_94300


namespace graph_opens_downward_axis_of_symmetry_vertex_coordinates_range_on_interval_l94_94849

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 6 * x - 1 / 2

theorem graph_opens_downward : ∀ x : ℝ, f x < -2 * x^2 + 6 * x - 1 / 2 :=
by
  intros
  sorry

theorem axis_of_symmetry : ∃ c : ℝ, (f c) = 4 ∧ c = 3 / 2 :=
by
  use 3 / 2
  sorry

theorem vertex_coordinates : ∃ v : ℝ × ℝ, v = (3 / 2, 4) ∧ f (3 / 2) = 4 :=
by
  use (3 / 2, 4)
  sorry

theorem range_on_interval : ∃ y_min y_max : ℝ, (∀ x ∈ Icc 0 2, y_min ≤ f x ∧ f x ≤ y_max) ∧ y_min = -1 / 2 ∧ y_max = 4 :=
by
  use (-1 / 2), 4
  sorry

end graph_opens_downward_axis_of_symmetry_vertex_coordinates_range_on_interval_l94_94849


namespace range_of_g_for_abs_x_lt_1_is_nonneg_l94_94185

def quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ f = λ x, a * x^2 + b * x + c

def range_of_fg (f g : ℝ → ℝ) : Prop :=
  ∀ y, y ≥ 0 → ∃ x, f (g x) = y

def range_of_g_for_abs_x_lt_1 (g : ℝ → ℝ) : Set ℝ :=
  {y | ∃ x, |x| < 1 ∧ g x = y}

theorem range_of_g_for_abs_x_lt_1_is_nonneg (f g : ℝ → ℝ) 
  (hf : quadratic f) (hg : quadratic g) (hrange : range_of_fg f g) :
  range_of_g_for_abs_x_lt_1 g = {y | y ≥ 0} :=
sorry

end range_of_g_for_abs_x_lt_1_is_nonneg_l94_94185


namespace percentage_democrats_for_candidate_X_l94_94881

variable (x: ℝ) -- x as a common factor
variable (D: ℝ) -- Percentage of Democrats voting for candidate X

-- Definitions
def republican_ratio : ℝ := 3 / 2
def total_republicans : ℝ := 3 * x
def total_democrats : ℝ := 2 * x
def votes_x_republicans : ℝ := 0.9 * total_republicans
def votes_x_democrats : ℝ := (D / 100) * total_democrats
def total_votes : ℝ := total_republicans + total_democrats
def ratio_votes_x : ℝ := 3 / 5 -- 60% of total votes means X wins by 20%
def votes_x : ℝ := ratio_votes_x * total_votes

-- Assertion to prove
theorem percentage_democrats_for_candidate_X
    (h1 : republican_ratio = 3 / 2)
    (h2 : votes_x = votes_x_republicans + votes_x_democrats) :
    D = 15 :=
by
  -- use proof steps based on the conditions and transformations above
  sorry

end percentage_democrats_for_candidate_X_l94_94881


namespace gertrude_fleas_l94_94070

variables (G M O : ℕ)

def fleas_maud := M = 5 * O
def fleas_olive := O = G / 2
def total_fleas := G + M + O = 40

theorem gertrude_fleas
  (h_maud : fleas_maud M O)
  (h_olive : fleas_olive G O)
  (h_total : total_fleas G M O) :
  G = 10 :=
sorry

end gertrude_fleas_l94_94070


namespace max_expression_on_ellipse_l94_94806

theorem max_expression_on_ellipse :
  ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1) → (∃ (M : ℝ), M = 7 ∧ ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1) → (M ≥ (3/4 * x^2 + 2 * x - y^2))) :=
begin
  sorry
end

end max_expression_on_ellipse_l94_94806


namespace max_points_for_top_teams_l94_94481

-- Definitions based on the problem conditions
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def points_for_loss : ℕ := 0
def number_of_teams : ℕ := 8
def number_of_games_between_each_pair : ℕ := 2
def total_games : ℕ := (number_of_teams * (number_of_teams - 1) / 2) * number_of_games_between_each_pair
def total_points_in_tournament : ℕ := total_games * points_for_win
def top_teams : ℕ := 4

-- Theorem stating the correct answer
theorem max_points_for_top_teams : (total_points_in_tournament / number_of_teams = 33) :=
sorry

end max_points_for_top_teams_l94_94481


namespace number_of_ordered_pairs_lcm_232848_l94_94187

theorem number_of_ordered_pairs_lcm_232848 :
  let count_pairs :=
    let pairs_1 := 9
    let pairs_2 := 7
    let pairs_3 := 5
    let pairs_4 := 3
    pairs_1 * pairs_2 * pairs_3 * pairs_4
  count_pairs = 945 :=
by
  sorry

end number_of_ordered_pairs_lcm_232848_l94_94187


namespace complex_coords_l94_94527

-- Define the complex number z
def z : ℂ := 2 + Complex.i

-- Define the transformed complex number
def transformed_z : ℂ := z^2 - 1

-- Assertion about the coordinates
theorem complex_coords : transformed_z = 2 + 4 * Complex.i :=
  sorry

end complex_coords_l94_94527


namespace toy_swords_count_l94_94508

variable (s : ℕ)

def cost_lego := 250
def cost_toy_sword := 120
def cost_play_dough := 35

def total_cost (s : ℕ) :=
  3 * cost_lego + s * cost_toy_sword + 10 * cost_play_dough

theorem toy_swords_count : total_cost s = 1940 → s = 7 := by
  sorry

end toy_swords_count_l94_94508


namespace probability_of_stopping_on_H_l94_94679

theorem probability_of_stopping_on_H (y : ℚ)
  (h1 : (1 / 5) + (1 / 4) + y + y + (1 / 10) = 1)
  : y = 9 / 40 :=
sorry

end probability_of_stopping_on_H_l94_94679


namespace sec_120_eq_neg_2_l94_94739

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_120_eq_neg_2 : sec 120 = -2 := by
  sorry

end sec_120_eq_neg_2_l94_94739


namespace exists_route_within_republic_l94_94919

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l94_94919


namespace pipe_cistern_l94_94330

theorem pipe_cistern (rate: ℚ) (duration: ℚ) (portion: ℚ) : 
  rate = (2/3) / 10 → duration = 8 → portion = 8/15 →
  portion = duration * rate := 
by 
  intros h1 h2 h3
  sorry

end pipe_cistern_l94_94330


namespace mean_study_hours_l94_94531

theorem mean_study_hours :
  let students := [3, 6, 8, 5, 4, 2, 2]
  let hours := [0, 2, 4, 6, 8, 10, 12]
  (0 * 3 + 2 * 6 + 4 * 8 + 6 * 5 + 8 * 4 + 10 * 2 + 12 * 2) / (3 + 6 + 8 + 5 + 4 + 2 + 2) = 5 :=
by
  sorry

end mean_study_hours_l94_94531


namespace find_angle_measure_l94_94495

noncomputable def isosceles_triangle (A B C : Type) [metric_space B] (a b : B) (h : B) : Prop :=
  dist a b = dist a h

noncomputable def point_on_line_segment (A B C : Type) [metric_space B] (x y z : B) (h : B) : Prop :=
  dist x h = dist h z ∧ dist h z = dist y z

noncomputable def angle_measure (A B : Type) [metric_space B] (a b c : B) [angle_space B] : Type :=
  measure_of_angle a b c

theorem find_angle_measure {A B C : Type} [metric_space B] (x y z w : B)
  (isosceles_XYZ : isosceles_triangle B x y z)
  (isosceles_XW : point_on_line_segment B x w z y) :
  angle_measure B x y w = 36 :=
  sorry

end find_angle_measure_l94_94495


namespace solution_set_of_abs_x_plus_one_gt_one_l94_94580

theorem solution_set_of_abs_x_plus_one_gt_one :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} :=
sorry

end solution_set_of_abs_x_plus_one_gt_one_l94_94580


namespace youngest_person_age_l94_94261

theorem youngest_person_age (n : ℕ) (average_age : ℕ) (average_age_when_youngest_born : ℕ) 
    (h1 : n = 7) (h2 : average_age = 30) (h3 : average_age_when_youngest_born = 24) :
    ∃ Y : ℚ, Y = 66 / 7 :=
by
  sorry

end youngest_person_age_l94_94261


namespace missing_dollar_problem_l94_94266

theorem missing_dollar_problem:
    ∀ (payment_per_person : ℕ) (total_people : ℕ) (overcharge : ℕ) (refund_per_person : ℕ) (kept_by_assistant : ℕ),
    total_people = 3 →
    payment_per_person = 10 →
    overcharge = 5 →
    refund_per_person = 1 →
    kept_by_assistant = 2 →
    (payment_per_person * total_people) - overcharge = 25 ∧ 
    (total_people * (payment_per_person - refund_per_person)) + kept_by_assistant = 27 ∧ 
    25 + (total_people * refund_per_person) + kept_by_assistant = 30 :=
begin
    intros payment_per_person total_people overcharge refund_per_person kept_by_assistant h1 h2 h3 h4 h5,
    split,
    { simp [h1, h2, h3] },
    split,
    { simp [h1, h2, h4, h5] },
    { simp [h1, h2, h4, h5] },
    sorry
end

end missing_dollar_problem_l94_94266


namespace hyperbola_equation_l94_94083

-- Define the conditions

structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0
  eqn : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1 

def circle_intersects_asymptote (d : ℝ) : Prop :=
  let c := Real.sqrt (3^2 + 4^2) 
  (c = d / 2)

def asymptote_slope (b a : ℝ) : Prop := 
  b / a = 4 / 3

-- The theorem to prove the hyperbola equation
theorem hyperbola_equation 
  (a b : ℝ)
  (h : Hyperbola a b)
  (circle_intersection : circle_intersects_asymptote (Real.sqrt (3^2 + 4^2) * 2))
  (asymptote_condition : asymptote_slope b a)
  : (Hyperbola 3 4).eqn = (Hyperbola 9 16).eqn := sorry

end hyperbola_equation_l94_94083


namespace eggs_leftover_l94_94341

theorem eggs_leftover :
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  total_eggs % 10 = 0 := by
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  exact Nat.mod_eq_zero_of_dvd (show 10 ∣ total_eggs from by norm_num)

end eggs_leftover_l94_94341


namespace envelopes_left_l94_94703

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end envelopes_left_l94_94703


namespace M_subset_N_l94_94721

def M : set ℝ := {x | x^2 = x}
def N : set ℝ := {x | x ≤ 1}

theorem M_subset_N : M ⊆ N := 
begin
  sorry
end

end M_subset_N_l94_94721


namespace max_product_of_two_integers_with_sum_2004_l94_94602

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l94_94602


namespace geometric_sequence_properties_l94_94829

-- Define sequence sum as provided in the conditions
def sum_geometric_sequence (x : ℝ) (n : ℕ) : ℝ := x * 3^n - 1/2

-- Define sequence terms in terms of the sum
def a_1 (x : ℝ) : ℝ := sum_geometric_sequence x 1
def a_2 (x : ℝ) : ℝ := sum_geometric_sequence x 2 - sum_geometric_sequence x 1
def a_3 (x : ℝ) : ℝ := sum_geometric_sequence x 3 - sum_geometric_sequence x 2

-- State the main theorem to prove
theorem geometric_sequence_properties (x : ℝ) : x = 1/2 ∧ ∀ n : ℕ, n > 0 → a_1 x * 3^(n-1) = 3^(n-1) :=
by
  sorry

end geometric_sequence_properties_l94_94829


namespace cannot_be_zero_can_be_one_all_odd_values_l94_94497

-- Definition of the list from 1 to 9
def seq : List Int := List.range' 1 10

-- Definition of the sum of the sequence
def sum_seq := List.sum seq

-- Statements to prove
theorem cannot_be_zero : ∀ (signs : List Int), (signs.filter (λ x => x = 1)).length + (signs.filter (λ x => x = 0)).length = 9 → signs.sum = 45 → 
                          signs.map (λ x => if x = 1 then 1 else -1).sum ≠ 0 := sorry

theorem can_be_one : ∃ (signs : List Int), (signs.filter (λ x => x = 1)).length + (signs.filter (λ x => x = 0)).length = 9 ∧ signs.sum = 45 ∧ 
                      signs.map (λ x => if x = 1 then 1 else -1).sum = 1 := sorry

theorem all_odd_values : ∀ (k : Int), -45 ≤ k ∧ k ≤ 45 ∧ k % 2 = 1 → ∃ (signs : List Int), (signs.filter (λ x => x = 1)).length + (signs.filter (λ x => x = 0)).length = 9 ∧ 
                                   signs.map (λ x => if x = 1 then 1 else -1).sum = k := sorry

end cannot_be_zero_can_be_one_all_odd_values_l94_94497


namespace mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l94_94980

-- Definitions
def mad_hatter_clock_rate := 5 / 4
def march_hare_clock_rate := 5 / 6
def time_at_dormouse_clock := 5 -- 5:00 PM

-- Real time calculation based on clock rates
def real_time (clock_rate : ℚ) (clock_time : ℚ) : ℚ := clock_time * (1 / clock_rate)

-- Mad Hatter's and March Hare's arrival times in real time
def mad_hatter_real_time := real_time mad_hatter_clock_rate time_at_dormouse_clock
def march_hare_real_time := real_time march_hare_clock_rate time_at_dormouse_clock

-- Theorems to be proved
theorem mad_hatter_waiting_time : mad_hatter_real_time = 4 := sorry
theorem march_hare_waiting_time : march_hare_real_time = 6 := sorry

-- Main theorem
theorem waiting_time : march_hare_real_time - mad_hatter_real_time = 2 := sorry

end mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l94_94980


namespace prism_height_l94_94232

theorem prism_height (α β b : ℝ) (hα: 0 < α ∧ α < π/2) (hβ: 0 < β ∧ β < π/2) :
  ∃ (h: ℝ), 
    h = b / cos (α/2) * sqrt (sin (β + α/2) * sin (β - α/2)) :=
sorry

end prism_height_l94_94232


namespace trigonometric_identity_l94_94992

theorem trigonometric_identity 
  (α β γ : ℝ)
  (h : (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ)) :
  (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) ∧
  (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ) = 
  abs (Real.cos α * Real.cos β * Real.cos γ) := by
  sorry

end trigonometric_identity_l94_94992


namespace jared_should_order_21_servings_l94_94260

def pieces_per_serving := 50
def jared_pieces := 120
def five_friends_pieces := 5 * 90
def three_other_friends_pieces := 3 * 150

def total_pieces := jared_pieces + five_friends_pieces + three_other_friends_pieces
def servings_needed := total_pieces / pieces_per_serving

theorem jared_should_order_21_servings :
  servings_needed.ceil = 21 :=
by
  sorry

end jared_should_order_21_servings_l94_94260


namespace triangle_converse_inverse_false_l94_94438

variables {T : Type} (p q : T → Prop)

-- Condition: If a triangle is equilateral, then it is isosceles
axiom h : ∀ t, p t → q t

-- Conclusion: Neither the converse nor the inverse is true
theorem triangle_converse_inverse_false : 
  (∃ t, q t ∧ ¬ p t) ∧ (∃ t, ¬ p t ∧ q t) :=
sorry

end triangle_converse_inverse_false_l94_94438


namespace total_students_l94_94883

variables (F G B N : ℕ)
variables (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6)

theorem total_students (F G B N : ℕ) (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6) : 
  F + G - B + N = 60 := by
sorry

end total_students_l94_94883


namespace find_acute_angles_l94_94233

-- Definitions of given conditions 
def right_triangle_base (ABC : Triangle) : Prop := ABC.angleC = 90
def lateral_perpendicular (AD : Line) (base : Triangle) : Prop := AD ⊥ base
def angles_condition (alpha beta : ℝ) : Prop := alpha < beta

-- Proof problem statement
theorem find_acute_angles (ABC : Triangle) 
    (AD : Line) (alpha beta : ℝ) 
    (h_base : right_triangle_base ABC) 
    (h_lateral : lateral_perpendicular AD ABC)
    (h_angles : angles_condition alpha beta) :
    (ABC.angleA = arcsin (cos beta / cos alpha) ∧ ABC.angleB = arccos (cos beta / cos alpha)) :=
sorry

end find_acute_angles_l94_94233


namespace sufficient_but_not_necessary_condition_l94_94408

def p (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def q (x a : ℝ) : Prop := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬ p x) ↔ a ≤ -1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l94_94408


namespace probability_sum_is_odd_l94_94068
open scoped BigOperators

theorem probability_sum_is_odd : 
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  let T := set.powersetLen 4 S
  ∃ p : ℚ, p = 16 / 35 ∧ 
  (∃ A ∈ T, (∃ odd_count, odd_count = A.count odd ∧ (odd_count = 1 ∨ odd_count = 3))) :=
sorry

end probability_sum_is_odd_l94_94068


namespace gcf_7fact_8fact_l94_94756

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l94_94756


namespace sum_of_four_selected_numbers_is_odd_probability_l94_94371

theorem sum_of_four_selected_numbers_is_odd_probability :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let n := 15
  let k := 4
  let comb (n k : ℕ) : ℕ := nat.choose n k
  let total_combinations := comb n k
  let valid_combinations := comb (n - 1) (k - 1) -- excluding one spot for the prime number 2
  total_combinations ≠ 0 → (valid_combinations / total_combinations : ℚ) = 4 / 15 :=
by
  sorry

end sum_of_four_selected_numbers_is_odd_probability_l94_94371


namespace sum_of_new_avg_and_sd_is_99_l94_94317

noncomputable def problem := 
  let student_scores := (x : ℝ) (hx : x ∈ {x | x = 1..54}).to_real in
  let avg := 90 in
  let sd := 4 in
  let updated_scores := (x + 5) in
  ∃ (new_avg new_sd : ℝ), new_avg = avg + 5 ∧ new_sd = sd ∧ new_avg + new_sd = 99

theorem sum_of_new_avg_and_sd_is_99 :
  problem :=
  by
  sorry

end sum_of_new_avg_and_sd_is_99_l94_94317


namespace largest_power_of_3_l94_94288

noncomputable def q : ℝ :=
  ∑ k in Finset.range 7, (k + 1)^2 * Real.log (k + 2)

theorem largest_power_of_3 (e_q_is_integer : ∃ n : ℕ, Real.exp q = n) : ∃ m : ℕ, 3^29 ∣ m ∧ ∀ p : ℕ, 3^p ∣ m → p ≤ 29 :=
by sorry

end largest_power_of_3_l94_94288


namespace parallel_lines_iff_a_eq_3_l94_94411

theorem parallel_lines_iff_a_eq_3 (a : ℝ) :
  (∀ x y : ℝ, (6 * x - 4 * y + 1 = 0) ↔ (a * x - 2 * y - 1 = 0)) ↔ (a = 3) := 
sorry

end parallel_lines_iff_a_eq_3_l94_94411


namespace probability_three_tails_one_head_l94_94461

theorem probability_three_tails_one_head :
  let outcome_probability := (1 / 2) ^ 4 in
  let combinations := Finset.card (Finset.filter (λ (s : Finset.Fin 4 × Bool), s.2).Finset (Finset.product (Finset.range 4) (Finset.singleton (false, true, true, true)))) in
  outcome_probability * combinations = 1 / 4 :=
sorry

end probability_three_tails_one_head_l94_94461


namespace Psi_gcd_eq_gcd_Psi_l94_94513

-- Let p be a prime number
variable (p : ℕ) [Fact (Nat.Prime p)]

-- Define the finite field F_p
def F_p : Type := ZMod p

-- Define the polynomials over F_p
def F_pX := Polynomial F_p

-- Define the Ψ function
noncomputable def Psi (f : F_pX) : F_pX :=
  f.sum (λ n a, Polynomial.monomial (p^n) a)

-- Define the gcd of polynomials in F_pX
noncomputable def gcd_F_pX (P Q : F_pX) : F_pX := Polynomial.gcd P Q

-- The main statement to be proven
theorem Psi_gcd_eq_gcd_Psi (F G : F_pX) (F_nonzero : F ≠ 0) (G_nonzero : G ≠ 0) :
  Psi (gcd_F_pX F G) = gcd_F_pX (Psi F) (Psi G) :=
sorry

end Psi_gcd_eq_gcd_Psi_l94_94513


namespace stamp_problem_l94_94984

open Nat

theorem stamp_problem :
  (∑ n in (Finset.filter (λ n, n % 6 = 4 ∧ n % 8 = 2) (Finset.range 100)), id n) = 68 := by
sorry

end stamp_problem_l94_94984


namespace solve_modulo_example_l94_94272

theorem solve_modulo_example :
  ∃ (n : ℕ), 0 ≤ n ∧ n < 19 ∧ 42568 % 19 = n := by
  use 3
  split
  { apply Nat.zero_le }
  split
  { exact Nat.lt_of_succ_le (by norm_num) }
  { norm_num }

end solve_modulo_example_l94_94272


namespace school_colors_percentage_l94_94532

-- Definitions
variable {N : ℕ} -- Total number of students
variable {G : ℕ} -- Number of girls
variable {B : ℕ} -- Number of boys
variable {P : ℝ} -- Percentage of students wearing school colors

-- Given Conditions
def girls_fraction : ℝ := 0.45
def boys_fraction : ℝ := 0.55
def girls_in_colors_fraction : ℝ := 0.60
def boys_in_colors_fraction : ℝ := 0.80

-- Proof target
theorem school_colors_percentage (hG : G = (girls_fraction * N).to_nat)
  (hB : B = (boys_fraction * N).to_nat)
  (hGirlsInColors : (girls_in_colors_fraction * G).to_nat + 
                    (boys_in_colors_fraction * B).to_nat = 
                    (0.71 * N).to_nat):
  P = 71 := 
by 
  sorry

end school_colors_percentage_l94_94532


namespace prob_three_tails_one_head_in_four_tosses_l94_94470

open Nat

-- Definitions
def coin_toss_prob (n : ℕ) (k : ℕ) : ℚ :=
  (real.to_rat (choose n k) / real.to_rat (2^n))

-- Parameters: n = 4 (number of coin tosses), k = 3 (number of tails)
def four_coins_three_tails_prob : ℚ := coin_toss_prob 4 3

-- Theorem: The probability of getting exactly three tails and one head in four coin tosses is 1/4.
theorem prob_three_tails_one_head_in_four_tosses : four_coins_three_tails_prob = 1/4 := by
  sorry

end prob_three_tails_one_head_in_four_tosses_l94_94470


namespace maximal_product_at_12_l94_94480

noncomputable def geometric_sequence (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
a₁ * q^(n - 1)

noncomputable def product_first_n_terms (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
(a₁ ^ n) * (q ^ ((n - 1) * n / 2))

theorem maximal_product_at_12 :
  ∀ (a₁ : ℕ) (q : ℚ), 
  a₁ = 1536 → 
  q = -1/2 → 
  ∀ (n : ℕ), n ≠ 12 → 
  (product_first_n_terms a₁ q 12) > (product_first_n_terms a₁ q n) :=
by
  sorry

end maximal_product_at_12_l94_94480


namespace fraction_to_decimal_zeros_l94_94281

theorem fraction_to_decimal_zeros (a b : ℕ) (h₁ : a = 5) (h₂ : b = 1600) : 
  (let dec_rep := (a : ℝ) / b;
       num_zeros := (String.mk (dec_rep.to_decimal_string.to_list) - '0')
  in num_zeros = 3) := 
  sorry

end fraction_to_decimal_zeros_l94_94281


namespace points_concyclic_l94_94270

variables {A B C D E F H : Type} [EuclideanGeometry A B C D E F H]

-- Conditions
-- 1. AB and AC are diameters of semi-circles constructed outside the acute triangle ABC
-- 2. AH ⊥ BC at H
-- 3. D is any point on side BC (excluding B and C)
-- 4. DE ‖ AC and DF ‖ AB with E and F on the two semi-circles respectively

theorem points_concyclic 
  (h1: IsDiameter A B) (h2: IsDiameter A C) 
  (h3: Perpendicular AH BC)
  (h4: OnLine D BC) (h5: D ≠ B ∧ D ≠ C)
  (h6: Parallel DE AC)
  (h7: Parallel DF AB)
  (h8: OnSemiCircle E A B)
  (h9: OnSemiCircle F A C) : 
  Concyclic D E F H := 
sorry

end points_concyclic_l94_94270


namespace mean_greater_than_median_by_two_l94_94391

theorem mean_greater_than_median_by_two (x : ℕ) (h : x > 0) :
  ((x + (x + 2) + (x + 4) + (x + 7) + (x + 17)) / 5 - (x + 4)) = 2 :=
sorry

end mean_greater_than_median_by_two_l94_94391


namespace determine_k_and_angle_l94_94723

variables {V : Type*} [inner_product_space ℝ V] (a b : V) (k : ℝ)

theorem determine_k_and_angle 
  (h : ∥a + k • b∥ = ∥a - b∥) (hk : k ≠ 0) :
  (k = -1 → inner_product ℝ a b = 0) ∧ 
  (k ≠ -1 → cos (angle a b) = - 1 / (k + 1)) :=
by
  sorry

end determine_k_and_angle_l94_94723


namespace problem1_problem2_l94_94441

noncomputable def a (x : ℝ) : (ℝ × ℝ) := ((Real.sqrt 3) * Real.cos x - (Real.sqrt 3), Real.sin x)
noncomputable def b (x : ℝ) : (ℝ × ℝ) := (1 + Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let ⟨a1, a2⟩ := a x
  let ⟨b1, b2⟩ := b x
  a1 * b1 + a2 * b2

theorem problem1 : f (25 * Real.pi / 6) = 0 := 
sorry

theorem problem2 : 
  ∀ x, x ∈ set.Icc (-Real.pi / 3) (Real.pi / 6) → 
  f x ∈ set.Icc (-Real.sqrt 3) (1 - Real.sqrt 3 / 2) := 
sorry

end problem1_problem2_l94_94441


namespace solution_set_l94_94965

open Real

variables {f : ℝ → ℝ}

-- Conditions
axiom cond1 : ∀ x ∈ ℝ, deriv f x > 2 * f x
axiom cond2 : f (1/2) = exp 1

-- Theorem to prove
theorem solution_set (x : ℝ) : f (log x) < x^2 ↔ 0 < x ∧ x < Real.sqrt (exp 1) :=
by
  sorry

end solution_set_l94_94965


namespace range_of_reciprocal_tan_l94_94878

theorem range_of_reciprocal_tan (A B C a b c : ℝ) 
  (h_triangle: 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_sides: b^2 - a^2 = a * c)
  (h_sum_angles: A + B + C = π)
  (ha: a > 0) (hb: b > 0) (hc: c > 0) 
  (h_acute: A < π / 4 ∧ A > π / 6) :
  0 < 1 / (tan A * tan B) ∧ 1 / (tan A * tan B) < 1 :=
sorry

end range_of_reciprocal_tan_l94_94878


namespace length_of_chord_l94_94435

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (-1 + 4 * t, 3 * t)

theorem length_of_chord :
  let circle := λ (x y : ℝ), x^2 + y^2 = 1,
      line := λ (x y : ℝ), 3 * x - 4 * y + 3 = 0,
      d := 3 / 5
  in 2 * real.sqrt (1 - d^2) = 8 / 5 :=
by
  -- Note: Proof is skipped here, using sorry
  sorry

end length_of_chord_l94_94435


namespace slowest_bailing_rate_l94_94293

noncomputable theory

def distance_to_shore : ℝ := 1.5 -- miles
def leak_rate : ℕ := 15 -- gallons per minute
def max_capacity_before_sinking : ℕ := 60 -- gallons
def rowing_speed : ℝ := 3 -- miles per hour

def required_bailing_rate (distance rowing_speed : ℝ) 
                           (leak_rate max_capacity_before_sinking : ℕ) : ℝ :=
  let time_to_shore := distance / rowing_speed -- hours
  let time_to_shore_minutes := time_to_shore * 60 -- minutes
  let total_leak := leak_rate * time_to_shore_minutes.to_nat -- gallons
  let required_bailing := max_capacity_before_sinking - total_leak
  required_bailing / time_to_shore_minutes

theorem slowest_bailing_rate :
  required_bailing_rate distance_to_shore rowing_speed leak_rate max_capacity_before_sinking.to_real = 13 :=
by
  sorry

end slowest_bailing_rate_l94_94293


namespace range_of_logs_function_l94_94258

theorem range_of_logs_function (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, log (3^x + 3^(-x) - a) = y) ↔ a ≥ 2 :=
by
  sorry

end range_of_logs_function_l94_94258


namespace product_f_geq_one_third_l94_94138

theorem product_f_geq_one_third (n : ℕ) (x : Fin n → ℝ) (h_sum : (∑ i, x i) = 1 / 2) (h_pos : ∀ i, 0 < x i) :
  (∏ i, (1 - x i) / (1 + x i)) ≥ 1 / 3 :=
sorry

end product_f_geq_one_third_l94_94138


namespace analogical_reasoning_max_area_volume_l94_94395

axiom rect_area_max (R : ℝ) :
  ∀ (a b : ℝ), a * b = π * R^2 → max_area_rect := a = b

axiom rect_solid_vol_max (R : ℝ) :
  ∀ (a b c : ℝ), a * b * c = (4/3) * π * R^3 → max_vol_rect_solid := a = b = c

theorem analogical_reasoning_max_area_volume (R : ℝ) :
  rect_area_max R → rect_solid_vol_max R :=
  sorry

end analogical_reasoning_max_area_volume_l94_94395


namespace shortest_wire_length_l94_94591

theorem shortest_wire_length
  (d1 d2 : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 30) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_sections := 2 * (r2 - r1)
  let curved_sections := 2 * Real.pi * r1 + 2 * Real.pi * r2
  let total_wire_length := straight_sections + curved_sections
  total_wire_length = 20 + 40 * Real.pi :=
by
  sorry

end shortest_wire_length_l94_94591


namespace symmetric_lines_a_b_l94_94244

theorem symmetric_lines_a_b (x y a b : ℝ) (A : ℝ × ℝ) (hA : A = (1, 0))
  (h1 : x + 2 * y - 3 = 0)
  (h2 : a * x + 4 * y + b = 0)
  (h_slope : -1 / 2 = -a / 4)
  (h_point : a * 1 + 4 * 0 + b = 0) :
  a + b = 0 :=
sorry

end symmetric_lines_a_b_l94_94244


namespace num_valid_row_lengths_l94_94313

theorem num_valid_row_lengths :
  let n := 96 in
  let valid_lengths := { x | 6 ≤ x ∧ x ≤ 18 ∧ n % x = 0 } in
  valid_lengths.card = 4 :=
by
  sorry

end num_valid_row_lengths_l94_94313


namespace problem_statement_l94_94850

noncomputable def S (n : ℕ) : ℕ := 
  if n % 2 = 1 then
    1 + 4 * ((n - 1) / 2)
  else
    2

def sum_S (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ k, S k)

theorem problem_statement : sum_S 2019 = 2019 * 1011 - 1 :=
sorry

end problem_statement_l94_94850


namespace total_chess_games_l94_94636

/-- Given 6 chess amateurs where each plays with exactly 4 other amateurs,
prove that the total number of unique chess games played is 10. -/
theorem total_chess_games (A B C D E F : Type) [Finite A] [Finite B] [Finite C] [Finite D] [Finite E] [Finite F] 
  (total_amateurs : (Finset Type) := {A, B, C, D, E, F})
  (total_players : card total_amateurs = 6)
  (games_matrix : matrix (Fin 6) (Fin 6) Bool)
  (games_played : ∀ (i : Fin 6), card {j | games_matrix i j} = 4) :
  ∑ i in finset.range 6, ∑ j in finset.range i, games_matrix i j = 10 :=
by 
  sorry

end total_chess_games_l94_94636


namespace total_cost_is_correct_l94_94196

def hip_hop_classes_per_week := 3
def salsa_classes_per_two_weeks := 1
def ballet_classes_per_week := 2
def jazz_classes_per_week := 1
def contemporary_classes_per_three_weeks := 1

def hip_hop_cost_per_class := 10.50
def salsa_cost_per_class := 15
def ballet_cost_per_class := 12.25
def jazz_cost_per_class := 8.75
def contemporary_cost_per_class := 10

def hip_hop_classes_four_weeks := 4 * hip_hop_classes_per_week
def salsa_classes_four_weeks := 4 / 2 * salsa_classes_per_two_weeks
def ballet_classes_four_weeks := 4 * ballet_classes_per_week
def jazz_classes_four_weeks := 4 * jazz_classes_per_week
def contemporary_classes_four_weeks := 4 / 3 * contemporary_classes_per_three_weeks

def total_hip_hop_cost := hip_hop_classes_four_weeks * hip_hop_cost_per_class
def total_salsa_cost := salsa_classes_four_weeks * salsa_cost_per_class
def total_ballet_cost := ballet_classes_four_weeks * ballet_cost_per_class
def total_jazz_cost := jazz_classes_four_weeks * jazz_cost_per_class
def total_contemporary_cost := contemporary_classes_four_weeks * contemporary_cost_per_class

def total_dance_class_cost := total_hip_hop_cost + total_salsa_cost + total_ballet_cost + total_jazz_cost + total_contemporary_cost

theorem total_cost_is_correct : total_dance_class_cost = 299 := by
  sorry

end total_cost_is_correct_l94_94196


namespace smallest_possible_sum_of_two_products_is_2448_l94_94990

-- Define the set of digits used
def digits := {3, 4, 5, 6, 7}

-- Condition: use each digit exactly once
def all_digits_used (l : List ℕ) : Prop :=
  l.erase_dup.length = digits.toList.length ∧ ∀ x, x ∈ digits ↔ x ∈ l

-- Define the function to calculate the sum of the two products
def sum_of_two_products (l : List ℕ) : ℕ :=
  if h : all_digits_used l ∧ l.length = 5 then
    let a := l.nthLe 0 h.2,
        b := l.nthLe 1 h.2,
        c := l.nthLe 2 h.2,
        d := l.nthLe 3 h.2,
        e := l.nthLe 4 h.2 in
    ((10 * a + b) * (10 * c + d)) + (5 * (10 * e + b))
  else
    0

-- The main theorem statement: the smallest possible sum is 2448
theorem smallest_possible_sum_of_two_products_is_2448 :
  ∃ l : List ℕ, all_digits_used l ∧ (l.length = 5) ∧ sum_of_two_products l = 2448 := sorry

end smallest_possible_sum_of_two_products_is_2448_l94_94990


namespace interval_of_monotonic_decrease_l94_94360

-- Definitions based on the conditions given in part a)
def f (x : ℝ) : ℝ := 2^(x^2 - 2*x - 3)

theorem interval_of_monotonic_decrease :
  (∀ x : ℝ, x ∈ (-∞ : ℝ, 0] → f (x + 1) > f (x + 2)) :=
by
  sorry

end interval_of_monotonic_decrease_l94_94360


namespace prove_square_ratio_l94_94483
noncomputable section

-- Definitions from given conditions
variables (a b : ℝ) (d : ℝ := Real.sqrt (a^2 + b^2))

-- Condition from the problem
def ratio_condition : Prop := a / b = (a + 2 * b) / d

-- The theorem we need to prove
theorem prove_square_ratio (h : ratio_condition a b d) : 
  ∃ k : ℝ, k = a / b ∧ k^4 - 3*k^2 - 4*k - 4 = 0 := 
by
  sorry

end prove_square_ratio_l94_94483


namespace correct_statements_in_problem_conditions_l94_94008

structure ProblemConditions :=
  (cond1 : "The negation of the proposition “For all $x \in \mathbb{R}$, $x^2 - 3x - 2 \geq 0$” 
           is “There exists an $x_0 \in \mathbb{R}$ such that $x_0^2 - 3x_0 - 2 \leq 0”")
  (cond2 : "If line a is parallel to line b, and line b is parallel to plane β, 
           then line a is parallel to plane β")
  (cond3 : "There exists an $m \in \mathbb{R}$, such that the function $f(x) = mx^{m^2 + 2m}$ 
           is a power function and is monotonically increasing on the interval $(0, +\infty)$")
  (cond4 : "Any line passing through the points $(x_1, y_1)$ and $(x_2, y_2)$ 
           can be represented by the equation 
           $(x_2 - x_1)(y - y_1) - (y_2 - y_1)(x - x_1) = 0$")

theorem correct_statements_in_problem_conditions (pc : ProblemConditions) :
  number_of_correct_statements pc = 2 := 
sorry

end correct_statements_in_problem_conditions_l94_94008


namespace problem_statement_l94_94126

variable (p q : Prop)
variable (h_p : ∃ x : ℝ, x^2 - x + 1 ≥ 0)
variable (h_q : ∀ (a b : ℝ), a^2 < b^2 → a < b)

theorem problem_statement : (h_p ∧ ¬ h_q) ↔ (p ∧ ¬ q) := by
  sorry

end problem_statement_l94_94126


namespace grandpa_total_distance_l94_94132

def distance_grandpa_walks (n : ℕ) (interval : ℕ) : ℕ :=
  2 * interval * (∑ i in finset.range n, i)

theorem grandpa_total_distance (n : ℕ) (interval : ℕ) :
  n = 18 → interval = 3 → distance_grandpa_walks n interval = 1734 :=
begin
  intros hn hi,
  subst hn,
  subst hi,
  simp [distance_grandpa_walks], 
  rw [finset.sum_range, nat.mul_sum],
  simp only [mul_assoc, nat.smul_eq_mul],
  norm_num,
  sorry
end

end grandpa_total_distance_l94_94132


namespace predict_HCl_formed_l94_94127

-- Define the initial conditions and chemical reaction constants
def initial_moles_CH4 : ℝ := 3
def initial_moles_Cl2 : ℝ := 6
def volume : ℝ := 2

-- Define the reaction stoichiometry constants
def stoich_CH4_to_HCl : ℝ := 2
def stoich_CH4 : ℝ := 1
def stoich_Cl2 : ℝ := 2

-- Declare the hypothesis that reaction goes to completion
axiom reaction_goes_to_completion : Prop

-- Define the function to calculate the moles of HCl formed
def moles_HCl_formed : ℝ :=
  initial_moles_CH4 * stoich_CH4_to_HCl

-- Prove the predicted amount of HCl formed is 6 moles under the given conditions
theorem predict_HCl_formed : reaction_goes_to_completion → moles_HCl_formed = 6 := by
  sorry

end predict_HCl_formed_l94_94127


namespace problem_statement_l94_94141

/-!
The problem states:
If |a-2| and |m+n+3| are opposite numbers, then a + m + n = -1.
-/

theorem problem_statement (a m n : ℤ) (h : |a - 2| = -|m + n + 3|) : a + m + n = -1 :=
by {
  sorry
}

end problem_statement_l94_94141


namespace q1_q2_l94_94399

def setM (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def setN (a x : ℝ) : Prop := a + 1 ≤ x ∧ x ≤ 2 * a - 1
def compRN (a x : ℝ) : Prop := x < a + 1 ∨ x > 2 * a - 1

-- Question 1
theorem q1 (a : ℝ) (x : ℝ) (h_a : a = 3) : setM x ∨ compRN a x ↔ True :=
by trivial

-- Question 2
theorem q2 (a : ℝ) : 
  (∀ x : ℝ, setN a x → setM x) ↔ a ≤ 3 :=
by sorry

end q1_q2_l94_94399


namespace mary_remaining_money_l94_94528

theorem mary_remaining_money (p : ℝ) : 
  let initial_money := 50
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let num_drinks := 5
  let num_medium_pizzas := 2
  let num_large_pizza := 1
  let total_spent := (num_drinks * drink_cost) + (num_medium_pizzas * medium_pizza_cost) + (num_large_pizza * large_pizza_cost)
  let remaining_money := initial_money - total_spent
  in remaining_money = 50 - 15 * p :=
by
  sorry

end mary_remaining_money_l94_94528


namespace lengths_of_triangle_sides_l94_94100

open Real

noncomputable def triangle_side_lengths (a b c : ℝ) (A B C : ℝ) :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ A + B + C = π ∧ A = 60 * π / 180 ∧
  10 * sqrt 3 = 0.5 * a * b * sin A ∧
  a + b = 13 ∧
  c = sqrt (a^2 + b^2 - 2 * a * b * cos A)

theorem lengths_of_triangle_sides
  (a b c : ℝ) (A B C : ℝ)
  (h : triangle_side_lengths a b c A B C) :
  (a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7) :=
sorry

end lengths_of_triangle_sides_l94_94100


namespace find_y_l94_94324

theorem find_y : ∀ (x1 y1 x2 : ℝ) (d : ℝ), x1 = 3 → y1 = -2 → x2 = 1 → d = 15 → 
  x1 < x2 → ∃ y : ℝ, sqrt ((x1 - x2)^2 + (y1 - y)^2) = d ∧ y < 0 → y = -2 - sqrt 221 :=
by
  intros x1 y1 x2 d hx1 hy1 hx2 hd hlt
  use -2 - sqrt 221
  split
  { sorry },
  { sorry }

end find_y_l94_94324


namespace find_y_coordinate_of_third_vertex_l94_94011

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def is_equilateral (p1 p2 p3 : ℝ × ℝ) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1 ∧ distance p3 p1 = distance p1 p2

noncomputable def altitude (side : ℝ) : ℝ := (side / 2) * real.sqrt 3

theorem find_y_coordinate_of_third_vertex 
  (p1 p2 : ℝ × ℝ) (h1 : p1 = (2, 3)) (h2 : p2 = (10, 3)) (third_vertex : ℝ × ℝ) 
  (h3 : is_equilateral p1 p2 third_vertex) (h4 : third_vertex.1 > 0) (h5 : third_vertex.2 > 0) :
  third_vertex.2 = 3 + 4 * real.sqrt 3 :=
sorry

end find_y_coordinate_of_third_vertex_l94_94011


namespace simplify_and_evaluate_expr_l94_94218

theorem simplify_and_evaluate_expr (a b : ℝ) (h1 : a = 1 / 2) (h2 : b = -4) :
  5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (-a * b ^ 2 + 3 * a ^ 2 * b) = -11 :=
by
  sorry

end simplify_and_evaluate_expr_l94_94218


namespace min_value_of_inverse_sum_l94_94398

noncomputable def minimumValue (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) : ℝ :=
  9 + 6 * Real.sqrt 2

theorem min_value_of_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 / 3 ∧ (1/x + 1/y) = 9 + 6 * Real.sqrt 2 := by
  sorry

end min_value_of_inverse_sum_l94_94398


namespace local_extrema_f_inequality_range_m_l94_94844

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3 * x + Real.log x

-- Part (1) : Local extremum proofs
theorem local_extrema_f :
  (∀ x : ℝ, f 1 = -2 ∧ differentiable f) ∧
  (∀ x : ℝ, (f 0.5 = - (5/4) - Real.log 2) ∧ differentiable f) := sorry

-- Part (2) : Inequality and range of m
theorem inequality_range_m (m : ℝ) :
  (∀ (x1 x2 : ℝ), x1 ∈ Icc 1 2 → x2 ∈ Icc 1 2 → x1 < x2 →
     x1 * x2 * (f x1 - f x2) - m * (x1 - x2) > 0)
  ↔ (m ∈ Iic (-6)) := sorry

end local_extrema_f_inequality_range_m_l94_94844


namespace quadratic_polynomials_equal_l94_94999

def integer_part (a : ℝ) : ℤ := ⌊a⌋

theorem quadratic_polynomials_equal 
  (f g : ℝ → ℝ)
  (hf : ∀ x, ∃ a1 b1 c1, f x = a1 * x^2 + b1 * x + c1)
  (hg : ∀ x, ∃ a2 b2 c2, g x = a2 * x^2 + b2 * x + c2)
  (H : ∀ x, integer_part (f x) = integer_part (g x)) : 
  ∀ x, f x = g x :=
sorry

end quadratic_polynomials_equal_l94_94999


namespace distance_from_vertex_to_asymptote_of_hyperbola_l94_94570

noncomputable def hyperbola_distance := 
  let hyperbola := ∀ x y, x^2 / 4 - y^2 = 1
  let vertex := (2, 0)
  let asymptote1 := ∀ x y, x - 2 * y = 0
  let asymptote2 := ∀ x y, x + 2 * y = 0
  let distance := (|2| : ℝ) / real.sqrt (1^2 + 2^2)
  distance = 2 * real.sqrt 5 / 5
  
theorem distance_from_vertex_to_asymptote_of_hyperbola
  (x y : ℝ) 
  (hxy : x^2 / 4 - y^2 = 1) 
  (vertex : x = 2 ∧ y = 0)
  (asymptote1 : x - 2 * y = 0 ∨ x + 2 * y = 0) : 
  (2 : ℝ) / real.sqrt (1^2 + 2^2) = 2 * real.sqrt 5 / 5 := 
  by
    sorry

end distance_from_vertex_to_asymptote_of_hyperbola_l94_94570


namespace find_number_that_gives_200_9_when_8_036_divided_by_it_l94_94655

theorem find_number_that_gives_200_9_when_8_036_divided_by_it (
  x : ℝ
) : (8.036 / x = 200.9) → (x = 0.04) :=
by
  intro h
  sorry

end find_number_that_gives_200_9_when_8_036_divided_by_it_l94_94655


namespace initial_price_of_TV_l94_94689

theorem initial_price_of_TV (T : ℤ) (phone_price_increase : ℤ) (total_amount : ℤ) 
    (h1 : phone_price_increase = (400: ℤ) + (40 * 400 / 100)) 
    (h2 : total_amount = T + (2 * T / 5) + phone_price_increase) 
    (h3 : total_amount = 1260) : 
    T = 500 := by
  sorry

end initial_price_of_TV_l94_94689


namespace complex_mod_sum_inverse_l94_94970

theorem complex_mod_sum_inverse
  (z w : ℂ)
  (hz : complex.abs z = 2)
  (hw : complex.abs w = 4)
  (hzw : complex.abs (z + w) = 3) :
  complex.abs (1 / z + 1 / w) = 3 / 8 :=
by
  sorry

end complex_mod_sum_inverse_l94_94970


namespace correct_statements_for_line_and_circle_l94_94434

theorem correct_statements_for_line_and_circle :
  (∀ k : ℝ, ∃ x y : ℝ, k * x - y - k = 0 ∧ (1, 0) = (x, y)) ∧
  (∃ c r : ℝ, ∀ x y : ℝ, (x - 2) * (x - 2) + (y - 1) * (y - 1) = 4 ∧ r = 2 := 
begin
  sorry
end

end correct_statements_for_line_and_circle_l94_94434


namespace angle_ROS_eq_angle_EGF_l94_94940

-- Define the quadrilateral and related points and conditions
variables {A B C D E F G P Q R S T O : Point}
variables (AB_intersect_CD_at_EF : Intersect (Line A B) (Line C D) E F)
variables (diagonals_intersect_at_G : Intersect (Line A C) (Line B D) G)
variables (P_interior_point: Interior P (Quadrilateral A B C D))
variables (perpendiculars_drawn: Perpendicular (Line P Q) (Line A B) ∧
                                  Perpendicular (Line P R) (Line B C) ∧
                                  Perpendicular (Line P S) (Line C D) ∧
                                  Perpendicular (Line P T) (Line D A))
variables (lines_intersect_O: Intersect (Line Q S) (Line R T) O)
variables (QRST_parallelogram: Parallelogram (Quadrilateral Q R S T))

-- Prove the required angle equality
theorem angle_ROS_eq_angle_EGF :
  ∠ROS = ∠EGF :=
sorry

end angle_ROS_eq_angle_EGF_l94_94940


namespace f_5_eq_8_l94_94823

noncomputable def f : ℕ → ℕ := sorry

-- Conditions
variable (f : ℕ → ℕ)

axiom monotone_f : ∀ a b : ℕ, a ≤ b → f(a) ≤ f(b)
axiom f_n_in_nat_star : ∀ n : ℕ, n > 0 → f(n) > 0
axiom ff_eq_3n : ∀ n : ℕ, n > 0 → f(f(n)) = 3 * n

-- Goal
theorem f_5_eq_8 : f(5) = 8 :=
sorry

end f_5_eq_8_l94_94823


namespace is_root_of_monic_deg4_poly_l94_94050

noncomputable def monic_deg4_poly_having_root : Polynomial ℚ :=
  x^4 - 16 * x^2 + 4

theorem is_root_of_monic_deg4_poly (a : ℝ) (b : ℝ) (r : ℝ)
  (h_sqrt3 : a = Real.sqrt 3)
  (h_sqrt5 : b = Real.sqrt 5)
  (h_root : r = a + b) :
  Polynomial.eval r (monic_deg4_poly_having_root) = 0 := by
  sorry

end is_root_of_monic_deg4_poly_l94_94050


namespace arithmetic_sequence_b3b7_l94_94184

theorem arithmetic_sequence_b3b7 (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_cond : b 4 * b 6 = 17) : 
  b 3 * b 7 = -175 :=
sorry

end arithmetic_sequence_b3b7_l94_94184


namespace gcf_factorial_seven_eight_l94_94763

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l94_94763


namespace andrew_game_night_expenses_l94_94957

theorem andrew_game_night_expenses : 
  let cost_per_game := 9 
  let number_of_games := 5 
  total_money_spent = cost_per_game * number_of_games 
→ total_money_spent = 45 := 
by
  intro cost_per_game number_of_games total_money_spent
  sorry

end andrew_game_night_expenses_l94_94957


namespace find_overline_abc_l94_94389

noncomputable def overline (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc d => acc * 10 + d) 0

theorem find_overline_abc 
  (a b c : ℕ)
  (h1 : a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h2 : b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h3 : c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h4 : overline [a, b] = b^2)
  (h5 : overline [a, c, b, c] = (overline [b, a])^2) :
  overline [a, b, c] = 369 :=
  sorry

end find_overline_abc_l94_94389


namespace number_of_red_notes_each_row_l94_94173

-- Definitions for the conditions
variable (R : ℕ) -- Number of red notes in each row
variable (total_notes : ℕ := 100) -- Total number of notes

-- Derived quantities
def total_red_notes := 5 * R
def total_blue_notes := 2 * total_red_notes + 10

-- Statement of the theorem
theorem number_of_red_notes_each_row 
  (h : total_red_notes + total_blue_notes = total_notes) : 
  R = 6 :=
by
  sorry

end number_of_red_notes_each_row_l94_94173


namespace x_one_minus_f_eq_one_l94_94522

noncomputable def x : ℝ := (1 + Real.sqrt 2) ^ 500
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem x_one_minus_f_eq_one : x * (1 - f) = 1 :=
by
  sorry

end x_one_minus_f_eq_one_l94_94522


namespace binomial_coeff_ratio_l94_94874

theorem binomial_coeff_ratio (n : ℕ) :
  (∑ k in finset.range (n + 1), (nat.choose n k) * (real.sqrt 1)^(n - k) * (3 * (1)⁻¹)^k) /
  (∑ k in finset.range (n + 1), nat.choose n k) = 64 ↔ n = 6 :=
by sorry

end binomial_coeff_ratio_l94_94874


namespace eliana_refill_l94_94451

theorem eliana_refill (total_spent cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) : (total_spent / cost_per_refill) = 3 :=
sorry

end eliana_refill_l94_94451


namespace complement_of_union_l94_94814

def is_in_Real_Set (x : ℝ) (A : set ℝ) : Prop := x ∈ A

def setA : set ℝ := { x | 3 ≤ x ∧ x < 7 }
def setB : set ℝ := { x | 2 < x ∧ x < 10 }

theorem complement_of_union (x : ℝ) : 
  (x ∉ setA ∪ setB) ↔ (x ≤ 2 ∨ x ≥ 10) :=
by
  sorry

end complement_of_union_l94_94814


namespace probability_three_tails_one_head_l94_94458

noncomputable def probability_of_three_tails_one_head : ℚ :=
  if H : 1/2 ∈ ℚ then 4 * ((1 / 2)^4 : ℚ)
  else 0

theorem probability_three_tails_one_head :
  probability_of_three_tails_one_head = 1 / 4 :=
by {
  have h : (1 / 2 : ℚ) ∈ ℚ := by norm_cast; norm_num,
  rw probability_of_three_tails_one_head,
  split_ifs,
  { field_simp [h],
    norm_cast,
    norm_num }
}

end probability_three_tails_one_head_l94_94458


namespace six_digit_number_l94_94674

theorem six_digit_number : ∃ x : ℕ, 100000 ≤ x ∧ x < 1000000 ∧ 3 * x = (x - 300000) * 10 + 3 ∧ x = 428571 :=
by
sorry

end six_digit_number_l94_94674


namespace train_speed_l94_94001

theorem train_speed
  (train_length : Real := 460)
  (bridge_length : Real := 140)
  (time_seconds : Real := 48) :
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 :=
by
  sorry

end train_speed_l94_94001


namespace initial_amount_of_money_l94_94194

variable (X : ℕ) -- Initial amount of money Lily had in her account

-- Conditions
def spent_on_shirt : ℕ := 7
def spent_in_second_shop : ℕ := 3 * spent_on_shirt
def remaining_after_purchases : ℕ := 27

-- Proof problem: prove that the initial amount of money X is 55 given the conditions
theorem initial_amount_of_money (h : X - spent_on_shirt - spent_in_second_shop = remaining_after_purchases) : X = 55 :=
by
  -- Placeholder to indicate that steps will be worked out in Lean
  sorry

end initial_amount_of_money_l94_94194


namespace sum_of_coefficients_l94_94095

theorem sum_of_coefficients :
  (∑ k in Finset.range 8, (1 : ℕ) + (k + 1) = 8) →
  (∑ k in Finset.range 8, (k + 1) * ((8.choose k -1) * k.succ + ∑ i in Finset.range k, (k.choose i - 1) * i.succ) = 502) :=
sorry

end sum_of_coefficients_l94_94095


namespace average_cost_per_fruit_l94_94983

theorem average_cost_per_fruit :
  let apple_price := 2
  let banana_price := 1
  let orange_price := 3
  let grape_price := 1.5
  let kiwi_price := 1.75
  let apple_quantity := 12
  let banana_quantity := 4
  let orange_quantity := 4
  let grape_quantity := 10
  let kiwi_quantity := 10
  let apple_discount_price := (apple_quantity - 2) * apple_price   -- Buy 10 get 2 free
  let orange_discount_price := (orange_quantity - 1) * orange_price -- Buy 3 get 1 free
  let grape_discount_price := if grape_price * grape_quantity > 10 then 0.8 * (grape_price * grape_quantity) else grape_price * grape_quantity  -- 20% discount over $10
  let kiwi_discount_price := if kiwi_quantity >= 10 then 0.85 * (kiwi_price * kiwi_quantity) else kiwi_price * kiwi_quantity  -- 15% discount for 10 or more
  let total_cost :=
    apple_discount_price +
    (banana_quantity * banana_price) +
    orange_discount_price +
    grape_discount_price +
    kiwi_discount_price
  total_cost / (apple_quantity + banana_quantity + orange_quantity + grape_quantity + kiwi_quantity) = 1.5 :=
begin
  sorry
end

end average_cost_per_fruit_l94_94983


namespace find_m_value_l94_94475

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.logBase m (m - x)

theorem find_m_value :
  ∀ (m : ℝ), (5 < m) → 
  let f_max := f m 3
      f_min := f m 5 in
  (f_max - f_min = 1) → 
  m = 3 + Real.sqrt 6 :=
begin
  intros m m_gt_5 f_max f_min h,
  sorry
end

end find_m_value_l94_94475


namespace necessary_but_not_sufficient_l94_94803

def angle_of_inclination (α : ℝ) : Prop :=
  α > Real.pi / 4

def slope_of_line (k : ℝ) : Prop :=
  k > 1

theorem necessary_but_not_sufficient (α k : ℝ) :
  angle_of_inclination α → (slope_of_line k → (k = Real.tan α)) → (angle_of_inclination α → slope_of_line k) ∧ ¬(slope_of_line k → angle_of_inclination α) :=
by
  sorry

end necessary_but_not_sufficient_l94_94803


namespace find_sum_of_products_l94_94988

-- Let ABC be a triangle with BC as one of its sides
variable (A B C X Y : Point) (r : ℝ) (radius : r = 20) (BC : Segment B C) (circleCenter : M : Point) 
variable (AB AC BX CY : ℝ)

-- Define the variables
def circle : Circle M (2 * r) := sorry -- A circle with center M and radius 20

-- Assume that the circle intersects AB at X and AC at Y
axiom intersects_AB_X : intersects circle AB X
axiom intersects_AC_Y : intersects circle AC Y

-- Given the conditions, we want to prove the final relationship
theorem find_sum_of_products 
    (hBC_diameter : BC = 2 * r) 
    (hBX_AB : BX = AB - (AM^2 - r^2) / AB)
    (hCY_AC : CY = AC - (AM^2 - r^2) / AC) :
    BX * AB + CY * AC = 1600 := 
sorry

end find_sum_of_products_l94_94988


namespace matrix_vector_addition_l94_94717

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -2], ![-5, 6]]
def v : Fin 2 → ℤ := ![5, -2]
def w : Fin 2 → ℤ := ![1, -1]

theorem matrix_vector_addition :
  (A.mulVec v + w) = ![25, -38] :=
by
  sorry

end matrix_vector_addition_l94_94717


namespace Eve_spend_l94_94736

noncomputable def hand_mitts := 14.00
noncomputable def apron := 16.00
noncomputable def utensils_set := 10.00
noncomputable def small_knife := 2 * utensils_set
noncomputable def total_cost_for_one_niece := hand_mitts + apron + utensils_set + small_knife
noncomputable def total_cost_for_three_nieces := 3 * total_cost_for_one_niece
noncomputable def discount := 0.25 * total_cost_for_three_nieces
noncomputable def final_cost := total_cost_for_three_nieces - discount

theorem Eve_spend : final_cost = 135.00 :=
by sorry

end Eve_spend_l94_94736


namespace conjugate_system_solution_l94_94504

theorem conjugate_system_solution (a b : ℝ) :
  (∀ x y : ℝ,
    (x + (2-a) * y = b + 1) ∧ ((2*a-7) * x + y = -5 - b)
    ↔ x + (2*a-7) * y = -5 - b ∧ (x + (2-a) * y = b + 1))
  ↔ a = 3 ∧ b = -3 := by
  sorry

end conjugate_system_solution_l94_94504


namespace pipe_A_rate_l94_94269

theorem pipe_A_rate (A : ℝ) : 
  let cycle_time := 5 
  let total_cycles := 100 / cycle_time
  let pipe_B_fill := 50 * 2
  let pipe_C_drain := 25 * 2
  let net_fill_per_cycle := A + pipe_B_fill - pipe_C_drain
  let total_fill := total_cycles * net_fill_per_cycle

  (total_fill = 5000) ↔ (A = 200) :=
by {
  rw [total_cycles],
  rw [pipe_B_fill],
  rw [pipe_C_drain],
  rw [net_fill_per_cycle],
  rw [total_fill],
  split;
  intro h,
  { linarith },
  { symmetry, linarith }
}

end pipe_A_rate_l94_94269


namespace range_of_a_l94_94541

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (1 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
sorry

end range_of_a_l94_94541


namespace number_of_polynomials_Q_l94_94514

def P (x : ℂ) : ℂ := (x - 2) * (x - 3) * (x - 4)

theorem number_of_polynomials_Q :
  ∃ (Q : ℂ[X]), ∃ (R : ℂ[X]), degree R = 3 ∧ P (Q) = P * R ∧ degree Q = 2 ∧
  (finset.univ.image (polynomial.eval 2) Q).card * (finset.univ.image (polynomial.eval 3) Q).card * (finset.univ.image (polynomial.eval 4) Q).card = 22 := sorry

end number_of_polynomials_Q_l94_94514


namespace park_length_l94_94334

noncomputable def length_of_park (L : ℝ) : Prop :=
  let width := 40 
  let road_width_each := 3 
  let lawn_area := 2109
  let total_width_of_roads := 2 * road_width_each
  let total_area_of_park := L * width
  let area_of_roads := L * total_width_of_roads
  let area_of_lawn := total_area_of_park - area_of_roads
  area_of_lawn = lawn_area

theorem park_length : ∃ L : ℝ, length_of_park L ∧ L = 62 :=
by {
  use 62,
  unfold length_of_park,
  simp,
  sorry
}

end park_length_l94_94334


namespace counterexample_to_proposition_l94_94583

theorem counterexample_to_proposition : ∃ a : ℝ, a > -2 ∧ a^2 ≤ 4 ∧ a = 0 :=
by {
  use 0,
  split,
  {
    linarith,
  },
  split,
  {
    norm_num,
  },
  {
    reflexivity,
  }
}

end counterexample_to_proposition_l94_94583


namespace median_of_scores_is_17point5_l94_94337

theorem median_of_scores_is_17point5 : 
  let scores : List ℕ := [16, 18, 20, 17, 16, 18]
  List.median scores = 17.5 :=
by
  sorry

end median_of_scores_is_17point5_l94_94337


namespace base_subtraction_correct_l94_94692

theorem base_subtraction_correct : 
  let n1 := 3 * (9^2) + 2 * (9^1) + 5 * (9^0),
      n2 := 2 * (6^2) + 4 * (6^1) + 5 * (6^0) in
  n1 - n2 = 165 :=
by
  -- The declaration of constants and the expression evaluation skipped
  sorry

end base_subtraction_correct_l94_94692


namespace airline_route_same_republic_exists_l94_94932

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l94_94932


namespace airline_route_same_republic_exists_l94_94931

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l94_94931


namespace conclusion_correct_l94_94784

theorem conclusion_correct :
  (∃ (x : ℝ), x ∈ (0 : ℝ, Real.pi / 2) ∧ Real.sin x + Real.cos x = 1/3) = false ∧
  (∃ (a b : ℝ), ∀ (x : ℝ), a < x ∧ x < b → Real.cos x < Real.cos (x + 1)) = false ∧
  (∀ (x : ℝ), Real.tan (x + 1) ≥ Real.tan x) = false ∧
  (∀ (x : ℝ), Real.cos (2 * x) + Real.sin (Real.pi / 2 - x) = Real.cos (2 * x) + Real.cos x ∧
              ∃ (m M : ℝ), ∀ (y : ℝ), m ≤ y ∧ y ≤ M) ∧
  Real.even 2 ∧ ∃ (n : ℝ), ∀ (k : ℝ), k > 0 → Real.abs (Real.sin (2 * k + Real.pi / 6)) = Real.abs (Real.sin (2 * k + Real.pi / 6)) = 0. :=
sorry

end conclusion_correct_l94_94784


namespace HACK_translation_is_correct_l94_94582

-- Define the mapping from the "QUICK MATH" code to digits
def letter_to_digit : Char → ℕ 
| 'Q' := 0
| 'U' := 1
| 'I' := 2
| 'C' := 3
| 'K' := 4
| 'M' := 5
| 'A' := 6
| 'T' := 7
| 'H' := 8
| 'S' := 9
| _   := 0  -- Default case, should not be reached with valid input

-- Define the input string "HACK" and its expected digit translation
def HACK_code : List Char := ['H', 'A', 'C', 'K']

-- Define the translated number from "HACK"
def HACK_number : List ℕ := HACK_code.map letter_to_digit

-- State the theorem
theorem HACK_translation_is_correct : HACK_number = [8, 6, 3, 4] :=
by
  -- Proof goes here
  sorry

end HACK_translation_is_correct_l94_94582


namespace ratio_of_sides_l94_94676

theorem ratio_of_sides (s r : ℝ) (h : s^2 = 2 * r^2 * Real.sqrt 2) : r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := 
by
  sorry

end ratio_of_sides_l94_94676


namespace C_start_time_l94_94593

noncomputable def M_to_N_distance : ℝ := 15
noncomputable def walking_speed : ℝ := 6
noncomputable def bicycling_speed : ℝ := 15
noncomputable def start_time_difference : ℝ := 3 / 11

theorem C_start_time {x : ℝ} (hx1 : 0 ≤ x) (hx2 : x ≤ M_to_N_distance) :
  let t_A := x / walking_speed + (M_to_N_distance - x) / bicycling_speed in
  let t_B := ((M_to_N_distance - x) / bicycling_speed + x / walking_speed) / 2 in
  let t_C := x / walking_speed in
  t_A = t_B ∧ ∀ t : ℝ, t_C = (x / (bicycling_speed / walking_speed)) - start_time_difference :=
sorry

end C_start_time_l94_94593


namespace remainder_500th_increasing_sequence_7_ones_l94_94182

theorem remainder_500th_increasing_sequence_7_ones : 
  (let S := { n : ℕ | n.binary_repr.count 1 = 7 }.to_list.sort (≤),
       N := S.get (by linarith) 
   in N % 500) = 375 := 
sorry

end remainder_500th_increasing_sequence_7_ones_l94_94182


namespace cupcakes_per_package_l94_94857

theorem cupcakes_per_package 
  (total_cupcakes : ℕ)
  (eaten_cupcakes : ℕ)
  (packages : ℕ)
  (remaining_cupcakes : ℕ := total_cupcakes - eaten_cupcakes)
  (cupcakes_per_package : ℕ := remaining_cupcakes / packages) :
  total_cupcakes = 20 →
  eaten_cupcakes = 11 →
  packages = 3 →
  cupcakes_per_package = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end cupcakes_per_package_l94_94857


namespace domain_function_y_range_function_y_intervals_of_monotonicity_l94_94056

def function_y (x : ℝ) : ℝ := 3 ^ (-x^2 + 2 * x + 3)

theorem domain_function_y : ∀ x : ℝ, function_y x = 3 ^ (-x^2 + 2 * x + 3) := 
by sorry

theorem range_function_y : ∃ y : ℝ, 0 < y ∧ y ≤ 81 ∧ ∀ x : ℝ, function_y x = y := 
by sorry

theorem intervals_of_monotonicity : 
  (∀ x1 x2 : ℝ, x1 < x2 → x1 ≤ 1 → x2 ≤ 1 → function_y x1 < function_y x2) ∧ 
  (∀ x1 x2 : ℝ, x1 < x2 → x1 ≥ 1 → x2 ≥ 1 → function_y x1 > function_y x2) := 
by sorry

end domain_function_y_range_function_y_intervals_of_monotonicity_l94_94056


namespace part1_part2_l94_94837

section

variable (a : ℝ) {x : ℝ}
variable (h₀ : a > 0) (h₁ : a ≠ 1)

noncomputable def f (x : ℝ) : ℝ := Real.log_base a (a * x^2 - x + 1)

-- Part (1)
theorem part1 : (∀ {x : ℝ}, f (x) ≤ 1) ↔ a = 1/2 := sorry

-- Part (2)
theorem part2 : 
  (∀ {x : ℝ}, x ∈ Set.Icc (1/4 : ℝ) (3/2 : ℝ) → f (x) < f (x + 1)) ↔ 
  a ∈ Set.Ioc (2/9 : ℝ) (1/3 : ℝ) ∪ Set.Ici (2 : ℝ) := sorry

end

end part1_part2_l94_94837


namespace gcf_factorial_seven_eight_l94_94761

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l94_94761


namespace distance_from_circle_center_to_line_l94_94833

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the equation of the line
def line_eq (x y : ℝ) : ℝ := 2 * x + y - 5

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a ^ 2 + b ^ 2)

-- Define the actual proof problem
theorem distance_from_circle_center_to_line : 
  distance_to_line circle_center 2 1 (-5) = Real.sqrt 5 :=
by
  sorry

end distance_from_circle_center_to_line_l94_94833


namespace ratio_of_perimeters_l94_94225

-- Given conditions
variables (s : ℝ) (h : 0 < s)

-- Definitions based on folding steps
def small_rectangle_perimeter := 2 * (s / 2 + s / 2)
def large_rectangle_perimeter := 2 * (s / 2 + s)

-- Proving the required ratio
theorem ratio_of_perimeters : 
  small_rectangle_perimeter s h / large_rectangle_perimeter s h = 2 / 3 :=
by
  -- Calculation placeholder
  sorry

end ratio_of_perimeters_l94_94225


namespace least_upper_bound_ant_distance_l94_94345

noncomputable def ant_position (n : ℕ) : ℝ × ℝ := 
  let θ := asin(3/5) * n
  let x := real.cos θ
  let y := real.sin θ
  (x * n, y * n)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem least_upper_bound_ant_distance :
  ∀ n : ℕ, distance (0, 0) (ant_position n) ≤ sqrt 10 := 
sorry

end least_upper_bound_ant_distance_l94_94345


namespace rectangle_enclosure_combinations_l94_94220

theorem rectangle_enclosure_combinations :
  (finset.card (finset.powersetLen 2 (finset.range 6)) *
   finset.card (finset.powersetLen 2 (finset.range 5)) = 150) :=
by
  sorry

end rectangle_enclosure_combinations_l94_94220


namespace triangle_area_l94_94963

def vec2 := ℝ × ℝ

def area_of_triangle (a b : vec2) : ℝ :=
  0.5 * |a.1 * b.2 - a.2 * b.1|

def a : vec2 := (2, -3)
def b : vec2 := (4, -1)

theorem triangle_area : area_of_triangle a b = 5 := by
  sorry

end triangle_area_l94_94963


namespace prime_sum_difference_bounded_prime_sum_difference_converges_to_beta_l94_94310

noncomputable def prime_sum_difference (N : ℕ) : ℝ :=
  ((List.filter Prime (List.range (N+1))).map (λ p, 1 / (p:ℝ))).sum - Real.log (Real.log N)

theorem prime_sum_difference_bounded (N : ℕ) (T : ℝ) 
  (hT : T > 15): 
  ∃ T, ∀ N ≥ 2, abs (prime_sum_difference N) < T :=
sorry

theorem prime_sum_difference_converges_to_beta (β : ℝ) : 
  Tendsto (λ N, prime_sum_difference N) atTop (𝓝 β) :=
sorry

end prime_sum_difference_bounded_prime_sum_difference_converges_to_beta_l94_94310


namespace undefined_expression_real_val_l94_94795

theorem undefined_expression_real_val (a : ℝ) :
  a = 2 → (a^3 - 8 = 0) :=
by
  intros
  sorry

end undefined_expression_real_val_l94_94795


namespace problem_inequality_l94_94119

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

noncomputable def x_i (i : ℕ) : ℝ := (2 * i - 1) * Real.pi / 2

theorem problem_inequality (n : ℕ) (h : 2 ≤ n) : 
  (∑ i in Finset.range (n + 1), if 2 ≤ i then 1 / (x_i i)^2 else 0) < 1 / 9 :=
by sorry

end problem_inequality_l94_94119


namespace constants_A_B_C_l94_94041

theorem constants_A_B_C (A B C : ℝ) (h₁ : ∀ x : ℝ, (x^2 + 5 * x - 6) / (x^4 + x^2) = A / x^2 + (B * x + C) / (x^2 + 1)) :
  A = -6 ∧ B = 0 ∧ C = 7 :=
by
  sorry

end constants_A_B_C_l94_94041


namespace unique_identification_impossible_l94_94695

theorem unique_identification_impossible (mw : ℝ) :
  (∃ c : string, c = "Compound" ∧ molecular_weight c = mw) → (⦿ ((a : ℝ) ∈ set_of_atoms(mw)).fintype):
sorry

def molecular_weight (compound : string) : ℝ := sorry

def set_of_atoms (mw : ℝ) : set ℝ := sorry

end unique_identification_impossible_l94_94695


namespace odd_function_f_neg3_l94_94824

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2 ^ x else -2 ^ (-x)

theorem odd_function_f_neg3 :
  f (-3) = -8 :=
by
  have h1 : ∀ x : ℝ, f(-x) = -f(x), from
    λ x, by
      simp only [f]
      split_ifs with h1 h2
      { exact h2.elim (show -x > 0 from neg_pos.mpr h1) }
      { split_ifs with h3
        { rw [pow_neg (2 : ℝ), neg_neg] }
        { exact h3.elim (show x > 0 from neg_neg h2) } }
  have h2 : ∀ x : ℝ, x > 0 → f(x) = 2^x, from
    λ x hx, by
      rw [f]
      split_ifs with h
      { exact if_true_true.mpr hx }
      { exact hx.elim h }
  have h3 : f(3) = 2 ^ 3 := h2 3 (by norm_num)
  calc
    f (-3) = -f (3) : h1 3
         ... = -8    : by rw [h3]

end odd_function_f_neg3_l94_94824


namespace infinite_arith_prog_contains_infinite_nth_powers_l94_94555

theorem infinite_arith_prog_contains_infinite_nth_powers
  (a d : ℕ) (n : ℕ) 
  (h_pos: 0 < d) 
  (h_power: ∃ k : ℕ, ∃ m : ℕ, a + k * d = m^n) :
  ∃ infinitely_many k : ℕ, ∃ m : ℕ, a + k * d = m^n :=
sorry

end infinite_arith_prog_contains_infinite_nth_powers_l94_94555


namespace integer_part_of_expression_l94_94379

noncomputable def integer_part (x : ℝ) : ℝ := real.floor x

theorem integer_part_of_expression : 
  let z := sqrt (76 - 42 * sqrt 3)
  let a := real.floor z
  let b := z - a
  integer_part (a + 9 / b) = 12 :=
by
  let z := sqrt (76 - 42 * sqrt 3)
  let a := real.floor z
  let b := z - a
  have : integer_part (a + 9 / b) = 12 := sorry
  exact this

end integer_part_of_expression_l94_94379


namespace blue_pill_cost_l94_94353

theorem blue_pill_cost
  (days : Int := 10)
  (total_expenditure : Int := 430)
  (daily_cost : Int := total_expenditure / days) :
  ∃ (y : Int), y + (y - 3) = daily_cost ∧ y = 23 := by
  sorry

end blue_pill_cost_l94_94353


namespace sin_gt_cos_interval_l94_94498

theorem sin_gt_cos_interval (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x > Real.cos x) : 
  Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4) :=
by
  sorry

end sin_gt_cos_interval_l94_94498


namespace total_girls_in_school_l94_94227

-- Definitions based on conditions
def total_students := 2000
def sample_students := 200
def sample_girls := 103
def sampling_ratio := (sample_students : ℝ) / total_students

-- Defining the target proof statement
theorem total_girls_in_school :
  ∃ (n : ℕ), n = inferred_total_girls ∧ inferred_total_girls = 970
where 
  inferred_total_girls := (sample_girls * 10 : ℕ)
:= sorry

end total_girls_in_school_l94_94227


namespace annual_income_of_a_l94_94299

-- Definitions based on the conditions
def monthly_income_ratio (a_income b_income : ℝ) : Prop := a_income / b_income = 5 / 2
def income_percentage (part whole : ℝ) : Prop := part / whole = 12 / 100
def c_monthly_income : ℝ := 15000
def b_monthly_income (c_income : ℝ) := c_income + 0.12 * c_income

-- The theorem to prove
theorem annual_income_of_a : ∀ (a_income b_income c_income : ℝ),
  monthly_income_ratio a_income b_income ∧
  b_income = b_monthly_income c_income ∧
  c_income = c_monthly_income →
  (a_income * 12) = 504000 :=
by
  -- Here we do not need to fill out the proof, so we use sorry
  sorry

end annual_income_of_a_l94_94299


namespace part1_part2_acute_part2_obtuse_l94_94155

-- Definition of the vectors a and b
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-2, 6)

-- Definitions involving vector c for part (1)
def c (x y : ℝ) : ℝ × ℝ := (x, y)
def scalar_c (k : ℝ) : ℝ × ℝ := (0, 2 * k)

-- Part 1: Prove the existence of c with given properties
theorem part1 : (c 0 3 = scalar_c k ∨ c 0 (-3) = scalar_c k) ∧ (c 0 3 = 3 ∨ c 0 (-3) = 3) :=
by sorry

-- Part 2: Definitions for λ ranges for acute/obtuse angle
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def lambda_range_acute : set ℝ :=
  { λ : ℝ | dot_product a (a.1 + λ * b.1, a.2 + λ * b.2) > 0 }

def lambda_range_obtuse : set ℝ :=
  { λ : ℝ | dot_product a (a.1 + λ * b.1, a.2 + λ * b.2) < 0 }

-- Proving the ranges for λ
theorem part2_acute : lambda_range_acute = {x | x < 0 ∨ (0 < x ∧ x < 5/14)} :=
by sorry

theorem part2_obtuse : lambda_range_obtuse = {x | x > 5/14} :=
by sorry

end part1_part2_acute_part2_obtuse_l94_94155


namespace minimum_candy_swaps_l94_94952

structure CandyCount :=
  (chocolates gummies caramels lollipops : Nat)

def initialCandies : CandyCount × CandyCount × CandyCount × CandyCount × CandyCount :=
  (⟨4, 3, 2, 1⟩, ⟨1, 2, 1, 1⟩, ⟨8, 6, 4, 2⟩, ⟨2, 1, 1, 1⟩, ⟨3, 4, 2, 1⟩)

def desiredCandies : CandyCount × CandyCount × CandyCount × CandyCount × CandyCount :=
  (⟨5, 3, 2, 1⟩, ⟨2, 3, 2, 1⟩, ⟨4, 4, 4, 2⟩, ⟨3, 2, 2, 1⟩, ⟨3, 4, 2, 2⟩)

theorem minimum_candy_swaps : ∃ (swaps : Nat), swaps = 7 ∧ ¬ ∃ swaps', swaps' < 7 ∧ can_meet_quotas initialCandies desiredCandies swaps' :=
by
  sorry

end minimum_candy_swaps_l94_94952


namespace common_area_of_overlapping_circles_l94_94235

theorem common_area_of_overlapping_circles (R : ℝ) (hR : 0 < R) :
  let area := (R^2 * (4 * Real.pi - 3 * Real.sqrt 3)) / 6
  in area = (R^2 * (4 * Real.pi - 3 * Real.sqrt 3)) / 6 := by
sorry

end common_area_of_overlapping_circles_l94_94235


namespace largest_possible_b_l94_94807

theorem largest_possible_b (a b c : ℤ) (h1 : a > b) (h2 : b > c) (h3 : c > 2) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l94_94807


namespace jasper_sold_31_drinks_l94_94175

def chips := 27
def hot_dogs := chips - 8
def drinks := hot_dogs + 12

theorem jasper_sold_31_drinks : drinks = 31 := by
  sorry

end jasper_sold_31_drinks_l94_94175


namespace nina_money_l94_94985

variable (C : ℝ)

def original_widget_count : ℕ := 6
def new_widget_count : ℕ := 8
def price_reduction : ℝ := 1.5

theorem nina_money (h : original_widget_count * C = new_widget_count * (C - price_reduction)) :
  original_widget_count * C = 36 := by
  sorry

end nina_money_l94_94985


namespace a_greater_than_b_for_n_ge_2_l94_94387

theorem a_greater_than_b_for_n_ge_2 
  (n : ℕ) 
  (hn : n ≥ 2) 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a^n = a + 1) 
  (h2 : b^(2 * n) = b + 3 * a) : 
  a > b := 
  sorry

end a_greater_than_b_for_n_ge_2_l94_94387


namespace Steven_has_16_apples_l94_94505

variable (Jake_Peaches Steven_Peaches Jake_Apples Steven_Apples : ℕ)

theorem Steven_has_16_apples
  (h1 : Jake_Peaches = Steven_Peaches - 6)
  (h2 : Steven_Peaches = 17)
  (h3 : Steven_Peaches = Steven_Apples + 1)
  (h4 : Jake_Apples = Steven_Apples + 8) :
  Steven_Apples = 16 := by
  sorry

end Steven_has_16_apples_l94_94505


namespace num_routes_l94_94859

def move := ℕ × ℕ

def A : move := (0, 4)
def B : move := (4, 0)

def valid_move (m1 m2 : move) : Prop :=
  (m1.1 ≤ m2.1 ∨ m1.1 = m2.1 ∧ m1.2 ≥ m2.2) ∧
  (m2.1 - m1.1 = 1 ∨ m1.1 = m2.1 ∧ m1.2 - m2.2 = 1)

def up_move_prohibited (moves : list move) : Prop :=
  ∀ (i : ℕ), i < moves.length - 1 →
    (moves.nth_le i _).2 > (moves.nth_le (i + 1) _) ∨
    (moves.nth_le i _).2 < (moves.nth_le (i + 1) _) →
    moves.nth_le (i + 1) _ = (moves.nth_le i _) ∧ 
    (moves.nth_le i _).2 = (moves.nth_le (i + 1) _).2 + 1
  
def valid_route (r : list move) : Prop :=
  r.head = A ∧ r.last = B ∧
  (∀ (i: ℕ), i < r.length - 1 → valid_move (r.nth_le i _) (r.nth_le (i + 1) _)) ∧
  up_move_prohibited r
  
def routes := {r : list move // valid_route r}

theorem num_routes : Fintype.card routes = 15 :=
  sorry

end num_routes_l94_94859


namespace train_speed_kmh_l94_94003

theorem train_speed_kmh 
  (L_train : ℝ) (L_bridge : ℝ) (time : ℝ)
  (h_train : L_train = 460)
  (h_bridge : L_bridge = 140)
  (h_time : time = 48) : 
  (L_train + L_bridge) / time * 3.6 = 45 := 
by
  -- Definitions and conditions
  have h_total_dist : L_train + L_bridge = 600 := by sorry
  have h_speed_mps : (L_train + L_bridge) / time = 600 / 48 := by sorry
  have h_speed_mps_simplified : 600 / 48 = 12.5 := by sorry
  have h_speed_kmh : 12.5 * 3.6 = 45 := by sorry
  sorry

end train_speed_kmh_l94_94003


namespace vector_magnitude_l94_94854

variables {V : Type*} [inner_product_space ℝ V]

-- Conditions
variables (a b : V)
hypothesis norm_a : ∥a∥ = 2
hypothesis norm_b : ∥b∥ = 1
hypothesis ortho_ab : ⟪a, b⟫ = 0

-- Prove |2a - b| = sqrt(17)
theorem vector_magnitude : ∥2 • a - b∥ = real.sqrt 17 :=
sorry

end vector_magnitude_l94_94854


namespace total_coins_l94_94137

-- Definitions for the conditions
def number_of_nickels := 13
def number_of_quarters := 8

-- Statement of the proof problem
theorem total_coins : number_of_nickels + number_of_quarters = 21 :=
by
  sorry

end total_coins_l94_94137


namespace max_elements_set_A_l94_94509

open Set

theorem max_elements_set_A :
  ∃ A ⊆ {1, 2, ..., 50}, (∀ x y ∈ A, x ≠ y → |1/x - 1/y| > 1/1000) 
  ∧ ∀ B ⊆ {1, 2, ..., 50}, (∀ x y ∈ B, x ≠ y → |1/x - 1/y| > 1/1000) → |B| ≤ 40 :=
sorry

end max_elements_set_A_l94_94509


namespace airline_route_within_republic_l94_94910

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l94_94910


namespace minimize_variance_l94_94831

theorem minimize_variance (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
    (h_sorted : [2, 3, 3, 7, a, b, 12, 15, 18, 20].sorted) 
    (h_median : (a + b) / 2 = 10) : a = 10 ∧ b = 10 :=
sorry

end minimize_variance_l94_94831


namespace red_given_red_l94_94637

def p_i (i : ℕ) : ℚ := sorry
axiom lights_probs_eq : p_i 1 + p_i 2 = 2 / 3
axiom lights_probs_eq2 : p_i 1 + p_i 3 = 2 / 3
axiom green_given_green : p_i 1 / (p_i 1 + p_i 2) = 3 / 4
axiom total_prob : p_i 1 + p_i 2 + p_i 3 + p_i 4 = 1

theorem red_given_red : (p_i 4 / (p_i 3 + p_i 4)) = 1 / 2 := 
sorry

end red_given_red_l94_94637


namespace range_of_m_l94_94804

variable (m : ℝ)
def p (x : ℝ) : Prop := x^2 + 2*x - m > 0

theorem range_of_m (h1 : ¬ p 1) (h2 : p 2) : 3 ≤ m ∧ m < 8 :=
by {
  sorry
}

end range_of_m_l94_94804


namespace solve_diophantine_eq_4x_7y_600_l94_94730

theorem solve_diophantine_eq_4x_7y_600 :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (x y : ℕ), ((4 * x + 7 * y = 600) ↔ ((x, y) ∈ S)) ∧ S.card = 22 :=
begin
  sorry
end

end solve_diophantine_eq_4x_7y_600_l94_94730


namespace no_nine_letter_good_words_l94_94358

-- Definition of a good word according to the given conditions
def is_good_word (s : String) : Prop :=
  s.data.all (λ c, c = 'A' ∨ c = 'B' ∨ c = 'C') ∧
  ∀ i, i < s.length - 1 → (s.data.nth i = 'A' → s.data.nth (i + 1) ≠ 'B' ∧ s.data.nth (i + 1) ≠ 'C') ∧
           (s.data.nth i = 'B' → s.data.nth (i + 1) ≠ 'C' ∧ s.data.nth (i + 1) ≠ 'A') ∧
           (s.data.nth i = 'C' → s.data.nth (i + 1) ≠ 'A' ∧ s.data.nth (i + 1) ≠ 'B')

-- The length of the word we are considering
def good_word_length := 9

-- The statement to prove
theorem no_nine_letter_good_words : ∀ s : String, s.length = good_word_length → ¬ is_good_word s :=
by
  intro s h
  sorry

end no_nine_letter_good_words_l94_94358


namespace flagpole_break_height_l94_94664

theorem flagpole_break_height (AB : ℝ) (BC : ℝ) (x : ℝ) : 
  AB = 5 ∧ BC = 1 ∧ AC = Real.sqrt(AB ^ 2 + BC ^ 2) → x = 2.6 :=
by
  intro h
  sorry

end flagpole_break_height_l94_94664


namespace equal_right_angles_l94_94401

theorem equal_right_angles (n : ℕ) (a b : ℕ) (h_angles : n^2 + n = a + b) 
                           (h_type_A : ∀ k, k < a → type_of_right_angle (grid_angle k) = A)
                           (h_type_B : ∀ k, k < b → type_of_right_angle (grid_angle k) = B) :
     a = b := 
     sorry

end equal_right_angles_l94_94401


namespace clinton_shoes_count_l94_94710

theorem clinton_shoes_count : 
  let hats := 5
  let belts := hats + 2
  let shoes := 2 * belts
  shoes = 14 := 
by
  -- Define the number of hats
  let hats := 5
  -- Define the number of belts
  let belts := hats + 2
  -- Define the number of shoes
  let shoes := 2 * belts
  -- Assert that the number of shoes is 14
  show shoes = 14 from sorry

end clinton_shoes_count_l94_94710


namespace cos_A_given_sin_A_eq_sin_C_area_triangle_given_conditions_l94_94877

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)

-- Problem (I)
theorem cos_A_given_sin_A_eq_sin_C (h₁ : a^2 = 3 * b * c) (h₂ : sin A = sin C) : cos A = 1 / 6 :=
by
  sorry

-- Problem (II)
theorem area_triangle_given_conditions (h₁ : a^2 = 3 * b * c) (h₂ : A = π / 4) (h₃ : a = 3) : S = (3 / 4) * sqrt 2 :=
by
  sorry

end cos_A_given_sin_A_eq_sin_C_area_triangle_given_conditions_l94_94877


namespace efficiency_percentage_l94_94666

-- Define the conditions
def E_A : ℚ := 1 / 23
noncomputable def days_B : ℚ := 299 / 10
def E_B : ℚ := 1 / days_B
def combined_efficiency : ℚ := 1 / 13

-- Define the main theorem
theorem efficiency_percentage : 
  E_A + E_B = combined_efficiency → 
  ((E_A / E_B) * 100) ≈ 1300 :=
by
  sorry

end efficiency_percentage_l94_94666


namespace cosine_of_angle_through_point_l94_94830

theorem cosine_of_angle_through_point (α : ℝ) (A : ℝ × ℝ)
  (hA : A = (-3/5, 4/5))
  (hα : ∃ θ, (cos θ = A.1 ∧ sin θ = A.2) ∧ α = θ) :
  cos α = -3/5 :=
by
  sorry

end cosine_of_angle_through_point_l94_94830


namespace factorization_problem_l94_94040

theorem factorization_problem (a b c : ℝ) :
  let E := a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3)
  let P := -(a^2 + ab + b^2 + bc + c^2 + ac)
  E = (a - b) * (b - c) * (c - a) * P :=
by
  sorry

end factorization_problem_l94_94040


namespace octahedron_hamiltonian_path_exists_l94_94321

def octahedron : SimpleGraph (Fin 6) :=
  ⟦0⟧ -- A has index 0
  ⟦1⟧ -- B has index 1
  ⟦2⟧ -- C has index 2
  ⟦3⟧ -- A_1 has index 3
  ⟦4⟧ -- B_1 has index 4
  ⟦5⟧ -- C_1 has index 5

def octahedron_edges (v1 v2 : Fin 6) : Prop :=
  v1 ≠ v2 ∧ (v1 + v2) % 3 ≠ 0

theorem octahedron_hamiltonian_path_exists :
  ∃ p : List (Fin 6), octahedron.isHamiltonianCycle p :=
begin
  sorry
end

end octahedron_hamiltonian_path_exists_l94_94321


namespace subset_of_primes_is_all_primes_l94_94306

theorem subset_of_primes_is_all_primes
  (P : Set ℕ)
  (M : Set ℕ)
  (hP : ∀ n, n ∈ P ↔ Nat.Prime n)
  (hM : ∀ S : Finset ℕ, (∀ p ∈ S, p ∈ M) → ∀ p, p ∣ (Finset.prod S id + 1) → p ∈ M) :
  M = P :=
sorry

end subset_of_primes_is_all_primes_l94_94306


namespace simplify_and_evaluate_expression_l94_94556

theorem simplify_and_evaluate_expression (x : ℝ) 
  (h1 : 5 - 2 * x ≥ 1)
  (h2 : x + 3 > 0)
  (h3 : x ≠ -1)
  (h4 : x ≠ 2)
  (h5 : x ≠ -2) : 
  (let expr := (x^2 - 4 * x + 4) / (x + 1) / (3 / (x + 1) - x + 1) in expr) = 1 := 
by
  have hx0 : x = 0, 
  { sorry },  -- Using the range from inequalities and constraints
  have simp_expr := ((0^2 - 4 * 0 + 4) / (0 + 1)) / (3 / (0 + 1) - 0 + 1),
  calc
    simp_expr = (0^2 - 4 * 0 + 4) / (0 + 1) / (3 / (0 + 1) - 0 + 1) : by sorry
            ... = 1 : by sorry

end simplify_and_evaluate_expression_l94_94556


namespace gcf_7_8_fact_l94_94749

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l94_94749


namespace a_alone_finishes_work_in_60_days_l94_94298

noncomputable def work_done_per_day_a_b_c : ℚ := 1 / 10
noncomputable def work_done_per_day_b : ℚ := 1 / 20
noncomputable def work_done_per_day_c : ℚ := 1 / 30

theorem a_alone_finishes_work_in_60_days :
  ∀ (A B C : ℚ), A + B + C = work_done_per_day_a_b_c → B = work_done_per_day_b → C = work_done_per_day_c → 
  1 / A = 60 :=
by
  intros A B C h1 h2 h3
  -- proof omitted
  sorry

end a_alone_finishes_work_in_60_days_l94_94298


namespace length_AB_is_sqrt3_max_OC_over_OD_l94_94834
noncomputable theory

-- Defining the curves C1 and C2
def curve1 (θ : ℝ) : ℝ × ℝ :=
  (1 + cos θ, sin θ)

def curve2 (t : ℝ) : ℝ × ℝ :=
  (- (sqrt 3) / 2 * t, (2 * sqrt 3) / 3 + t / 2)

-- Task (1): Proving the length of line segment AB
theorem length_AB_is_sqrt3 : ∀ {A B : ℝ × ℝ},
  (A = curve1 θ ∧ B = curve2 t) →
  ∥A - B∥ = sqrt 3 :=
sorry

-- Task (2): Polar coordinate system, find the maximum value of |OC| / |OD|
def polar_curve1 (θ : ℝ) : ℝ :=
  2 * cos θ

theorem max_OC_over_OD : ∀ (α : ℝ),
  ∀ {C D : ℝ},
  ∃ (OC OD : ℝ),
  OC = 2 * cos α ∧ OD = 2 / sin α →
  max ((OC) / (OD)) = 1 / 2 :=
sorry

end length_AB_is_sqrt3_max_OC_over_OD_l94_94834


namespace expression_zero_l94_94190

theorem expression_zero (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x / |y| - |x| / y) * (y / |z| - |y| / z) * (z / |x| - |z| / x) = 0 :=
begin
  sorry
end

end expression_zero_l94_94190


namespace problem_I_problem_II_l94_94812

-- (I) Monotonicity and range of function f(θ)
theorem problem_I (θ : ℝ) (hθ : θ ∈ Ioo (-π/2 : ℝ) (π/2 : ℝ)) :
    let P := (2 * (Real.cos θ) - Real.sin θ, -1)
    let BP := (Real.sin θ - Real.cos θ, 1)
    let CA := (2 * (Real.sin θ), -1)
    let f := BP.1 * CA.1 + BP.2 * CA.2
    ( (θ ∈ Ioo (-π/8) (π/8) → StrictMono (λ θ : ℝ, - Real.sqrt 2 * Real.sin(2 * θ + π / 4))) ∧
      (θ ∈ Ioo (π/8) (π/2) → StrictMono (λ θ : ℝ, Real.sqrt 2 * Real.sin(2 * θ + π / 4))) ) ∧
    Icc (- Real.sqrt 2) 1 (λ θ, f)
:= sorry

-- (II) Value of the magnitude of vector sum of OA and OB
theorem problem_II (θ : ℝ) (hθ : θ ∈ Ioo (-π/2 : ℝ) (π/2 : ℝ)) 
    (h_tan : Real.tan θ = 4 / 3) :
    let OA := ((Real.sin θ), 1)
    let OB := ((Real.cos θ), 0)
    let magnitude_sum := Real.sqrt ((OA.1 + OB.1)^2 + (OA.2 + OB.2)^2)
    magnitude_sum = sqrt(74) / 5
:= sorry

end problem_I_problem_II_l94_94812


namespace cube_edge_percentage_growth_l94_94621

theorem cube_edge_percentage_growth (p : ℝ) 
  (h : (1 + p / 100) ^ 2 - 1 = 0.96) : p = 40 :=
by
  sorry

end cube_edge_percentage_growth_l94_94621


namespace fencing_required_l94_94630

theorem fencing_required (L : ℝ) (W : ℝ) (A : ℝ) (H1 : L = 20) (H2 : A = 720) 
  (H3 : A = L * W) : L + 2 * W = 92 := by 
{
  sorry
}

end fencing_required_l94_94630


namespace probability_three_tails_one_head_l94_94459

noncomputable def probability_of_three_tails_one_head : ℚ :=
  if H : 1/2 ∈ ℚ then 4 * ((1 / 2)^4 : ℚ)
  else 0

theorem probability_three_tails_one_head :
  probability_of_three_tails_one_head = 1 / 4 :=
by {
  have h : (1 / 2 : ℚ) ∈ ℚ := by norm_cast; norm_num,
  rw probability_of_three_tails_one_head,
  split_ifs,
  { field_simp [h],
    norm_cast,
    norm_num }
}

end probability_three_tails_one_head_l94_94459


namespace solution_set_of_inequality_l94_94254

theorem solution_set_of_inequality (x : ℝ) : 
  {|x - 1| - |x - 5| < 2 ↔ x < 4} :=
sorry

end solution_set_of_inequality_l94_94254


namespace train_speed_l94_94338

def train_length : ℕ := 120
def bridge_length : ℕ := 255
def crossing_time : ℕ := 30

theorem train_speed :
  (375 / 30) * 3.6 = 45 := 
by 
  have dist : ℕ := train_length + bridge_length
  have speed_ms : ℕ := dist / crossing_time
  have speed_kmh := speed_ms * 3.6
  show speed_kmh = 45
  sorry

end train_speed_l94_94338


namespace gcf_7fact_8fact_l94_94752

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l94_94752


namespace probability_three_tails_one_head_l94_94466

theorem probability_three_tails_one_head :
  (nat.choose 4 1) * (1/2)^4 = 1/4 :=
by
  sorry

end probability_three_tails_one_head_l94_94466


namespace max_value_f_min_value_l94_94649

-- (1) Problem statement
def f (x : ℝ) : ℝ := 4 / (x - 3) + x

theorem max_value_f (h : x < 3) : 
  ∃ x : ℝ, max_value f x = -1 :=
sorry

-- (2) Problem statement
theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  ∃ x y : ℝ, min_value (1 / x + 3 / y) = 1 + (Real.sqrt 3) / 2 :=
sorry

end max_value_f_min_value_l94_94649


namespace brainiacs_survey_l94_94678

/-- 
A survey was taken among some brainiacs. Of those surveyed, twice as many brainiacs like rebus
teasers as math teasers. If 18 brainiacs like both rebus teasers and math teasers, 
4 like neither kind of teaser, and 20 brainiacs like math teasers but not rebus teasers, 
then the total number of brainiacs surveyed is 100.
-/
theorem brainiacs_survey (R M : ℕ) (h1 : R = 2 * M) (h2 : 18 ≤ R ∧ 18 ≤ M)
    (h3 : 20 ≤ M) (h4 : 4 > 0) :
    let total := (R - 18) + 20 + 18 + 4
    in total = 100 := 
sorry

end brainiacs_survey_l94_94678


namespace numberOfTrucks_l94_94320

-- Conditions
def numberOfTanksPerTruck : ℕ := 3
def capacityPerTank : ℕ := 150
def totalWaterCapacity : ℕ := 1350

-- Question and proof goal
theorem numberOfTrucks : 
  (totalWaterCapacity / (numberOfTanksPerTruck * capacityPerTank) = 3) := 
by 
  sorry

end numberOfTrucks_l94_94320


namespace price_increase_for_desired_profit_l94_94478

/--
In Xianyou Yonghui Supermarket, the profit from selling Pomelos is 10 yuan per kilogram.
They can sell 500 kilograms per day. Market research has found that, with a constant cost price, if the price per kilogram increases by 1 yuan, the daily sales volume will decrease by 20 kilograms.
Now, the supermarket wants to ensure a daily profit of 6000 yuan while also offering the best deal to the customers.
-/
theorem price_increase_for_desired_profit :
  ∃ x : ℝ, (10 + x) * (500 - 20 * x) = 6000 ∧ x = 5 :=
sorry

end price_increase_for_desired_profit_l94_94478


namespace ellipse_line_intersection_points_l94_94242

theorem ellipse_line_intersection_points {a b m c x y : ℝ}
    (h1 : a ≠ 0) 
    (h2 : b ≠ 0) 
    (h3 : (x^2) / (a^2) + (y^2) / (b^2) = 1) 
    (h4 : y = m * x + c) : 
    (x, y) satisfies the intersection points of the ellipse and the line :=
sorry

end ellipse_line_intersection_points_l94_94242


namespace log_sum_less_neg_two_l94_94800

theorem log_sum_less_neg_two (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) : 
  log 2 a + log 2 b < -2 := 
sorry

end log_sum_less_neg_two_l94_94800


namespace infinite_triples_exists_l94_94210

/-- There are infinitely many ordered triples (a, b, c) of positive integers such that 
the greatest common divisor of a, b, and c is 1, and the sum a^2b^2 + b^2c^2 + c^2a^2 
is the square of an integer. -/
theorem infinite_triples_exists : ∃ (a b c : ℕ), (∀ p q : ℕ, p ≠ q ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ 2 < p ∧ 2 < q →
  let a := p * q
  let b := 2 * p^2
  let c := q^2
  gcd (gcd a b) c = 1 ∧
  ∃ k : ℕ, a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = k^2) :=
sorry

end infinite_triples_exists_l94_94210


namespace smallest_possible_perimeter_l94_94243

theorem smallest_possible_perimeter (l m n : ℕ) (hlm : l > m) (hmn : m > n)
  (h : (frac (3^l / 10^4) = frac (3^m / 10^4)) ∧ (frac (3^m / 10^4) = frac (3^n / 10^4))) : 
  l + m + n = 3003 :=
sorry

end smallest_possible_perimeter_l94_94243


namespace maximum_integer_solutions_l94_94671

def is_polynomial_with_int_coeff (p : ℤ[X]) : Prop := ∀ n, p.coeff n ∈ ℤ

def is_self_centered (p : ℤ[X]) : Prop := is_polynomial_with_int_coeff p ∧ p.eval 100 = 100

theorem maximum_integer_solutions (p : ℤ[X]) (h_self_centered: is_self_centered p) : 
  ∃ k: ℕ, k = 11 :=
sorry

end maximum_integer_solutions_l94_94671


namespace tangent_line_equation_at_x_zero_l94_94571

noncomputable def curve (x : ℝ) : ℝ := x + Real.exp (2 * x)

theorem tangent_line_equation_at_x_zero :
  ∃ (k b : ℝ), (∀ x : ℝ, curve x = k * x + b) :=
by
  let df := fun (x : ℝ) => (deriv curve x)
  have k : ℝ := df 0
  have b : ℝ := curve 0 - k * 0
  use k, b
  sorry

end tangent_line_equation_at_x_zero_l94_94571


namespace gcf_7_8_fact_l94_94747

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l94_94747


namespace jelly_beans_ratio_l94_94265

theorem jelly_beans_ratio :
  let bagA := 24
  let bagB := 32
  let bagC := 34
  let yellowA := 24 * 0.40
  let yellowB := 32 * 0.30
  let yellowC := 34 * 0.25
  let totalYellow := yellowA + yellowB + yellowC
  let totalBeans := bagA + bagB + bagC
  (totalYellow / totalBeans * 100) ≈ 32.22 :=
by
  let bagA := 24
  let bagB := 32
  let bagC := 34
  let yellowA := 24 * 0.40
  let yellowB := 32 * 0.30
  let yellowC := 34 * 0.25
  let totalYellow := yellowA + yellowB + yellowC
  let totalBeans := bagA + bagB + bagC
  have ratio := (totalYellow / totalBeans) * 100
  have expected := 32.22
  exact ratio ≈ expected

end jelly_beans_ratio_l94_94265


namespace first_day_pushups_l94_94798

-- Define the conditions as lean definitions and constants
constant Geli_first_day_pushups : ℕ
constant Geli_second_day_pushups : ℕ := Geli_first_day_pushups + 5
constant Geli_third_day_pushups : ℕ := Geli_first_day_pushups + 10
constant Geli_total_week_pushups : ℕ := Geli_first_day_pushups + Geli_second_day_pushups + Geli_third_day_pushups

-- Condition that the total number of push-ups in the first week is 45
axiom total_pushups_45 : Geli_total_week_pushups = 45

-- The theorem to be proved
theorem first_day_pushups : Geli_first_day_pushups = 10 :=
by
  sorry

end first_day_pushups_l94_94798


namespace problem_f_g_l94_94521

noncomputable def f : ℝ → ℝ := λ x, (3 * x^2 + 5 * x + 8) / (x^2 - x + 4)
def g : ℝ → ℝ := λ x, x - 1

theorem problem_f_g : (f (g 1)) + (g (f 1)) = 5 := by
  sorry

end problem_f_g_l94_94521


namespace recreation_percentage_this_week_l94_94507

variable (W : ℝ) -- David's last week wages
variable (R_last_week : ℝ) -- Recreation spending last week
variable (W_this_week : ℝ) -- This week's wages
variable (R_this_week : ℝ) -- Recreation spending this week

-- Conditions
def wages_last_week : R_last_week = 0.4 * W := sorry
def wages_this_week : W_this_week = 0.95 * W := sorry
def recreation_spending_this_week : R_this_week = 1.1875 * R_last_week := sorry

-- Theorem to prove
theorem recreation_percentage_this_week :
  (R_this_week / W_this_week) = 0.5 := sorry

end recreation_percentage_this_week_l94_94507


namespace positive_integers_solution_l94_94052

theorem positive_integers_solution :
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ (3 ^ m - 2 ^ n = -1 ∨ 3 ^ m - 2 ^ n = 5 ∨ 3 ^ m - 2 ^ n = 7) ↔
  (m, n) = (0, 1) ∨ (m, n) = (2, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (2, 2) :=
by
  sorry

end positive_integers_solution_l94_94052


namespace max_f_g_lt_one_l94_94839

def f (x : ℝ) : ℝ := (1 - x) * Real.exp(x) - 1
def g (x : ℝ) : ℝ := f(x) / x

theorem max_f : (∀ x : ℝ, f(x) ≤ 0) ∧ (∃ x : ℝ, f(x) = 0) := by
  sorry

theorem g_lt_one (x : ℝ) (hx1 : x > -1) (hx2 : x ≠ 0) : g(x) < 1 := by
  sorry

end max_f_g_lt_one_l94_94839


namespace arithmetic_sequence_2007th_term_l94_94488

theorem arithmetic_sequence_2007th_term :
  ∃ (a d : ℤ),
  (∀ (n : ℤ), a_d = a + (n-1) * d) ∧
  (a_3 = a + 2 * d) ∧
  (a_5 = a + 4 * d) ∧
  (a_{11} = a + 10 * d) ∧
  (a_4 = 6) ∧
  (a_5 / a_3 = a_{11} / a_5) ∧
  (a + (2006 : ℕ) * d = 6015) :=
sorry

end arithmetic_sequence_2007th_term_l94_94488


namespace vector_calculation_l94_94125

namespace VectorProof

-- Given vectors a and b
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

-- Define the target vector operation
def target_vector : ℝ × ℝ := (1/2 * a.1 - 3/2 * b.1, 1/2 * a.2 - 3/2 * b.2)

-- Prove that the target vector is equal to (-1, 2)
theorem vector_calculation : target_vector = (-1, 2) := by
  -- Calculation steps wrapped
  sorry

end VectorProof

end vector_calculation_l94_94125


namespace cyclic_quadrilateral_l94_94810

-- Define the conditions
structure EquilateralTriangle (α : Type*) [MetricSpace α] :=
(A B C O : α)
(is_equilateral : dist A B = dist B C ∧ dist B C = dist C A)
(center_O : ∀ P, dist O A = dist O B ∧ dist O B = dist O C)

-- Define the problem
theorem cyclic_quadrilateral (α : Type*) [MetricSpace α] 
  {A B C O D E: α} 
  (eq_triangle : EquilateralTriangle α)
  (line_C : ∃ (p : Line α), p ∩ {C} = {C} ∧ 
                            ∃ (readQ : intersects_circumcircle A O B p).points = {D, E}) :
  let D' := midpoint D B,
  let E' := midpoint E B in
  circle (Set α) {A, O, D', E'} :=
  sorry

end cyclic_quadrilateral_l94_94810


namespace systematic_sampling_5_of_50_l94_94596

theorem systematic_sampling_5_of_50 :
  ∃ S : Finset ℕ, (∀ x ∈ S, x ∈ Finset.range 51) ∧ S.card = 5 ∧ 
  (∀ (x y ∈ S), x ≠ y → abs (x - y) = 10) ∧ S = {5, 15, 25, 35, 45} :=
by
  sorry

end systematic_sampling_5_of_50_l94_94596


namespace chocolate_ticket_fraction_l94_94563

theorem chocolate_ticket_fraction (box_cost : ℝ) (ticket_count_per_free_box : ℕ) (ticket_count_included : ℕ) :
  ticket_count_per_free_box = 10 →
  ticket_count_included = 1 →
  (1 / 9 : ℝ) * box_cost =
  box_cost / ticket_count_per_free_box + box_cost / (ticket_count_per_free_box - ticket_count_included + 1) :=
by 
  intros h1 h2 
  have h : ticket_count_per_free_box = 10 := h1 
  have h' : ticket_count_included = 1 := h2 
  sorry

end chocolate_ticket_fraction_l94_94563


namespace product_arithmetic_sequence_mod_100_l94_94774

def is_arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ → Prop) : Prop :=
  ∀ k, n k → k = a + d * (k / d)

theorem product_arithmetic_sequence_mod_100 :
  ∀ P : ℕ,
    (∀ k, 7 ≤ k ∧ k ≤ 1999 ∧ ((k - 7) % 12 = 0) → P = k) →
    (P % 100 = 75) :=
by {
  sorry
}

end product_arithmetic_sequence_mod_100_l94_94774


namespace maximum_value_of_expression_l94_94409

theorem maximum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 := sorry

end maximum_value_of_expression_l94_94409


namespace distribution_ways_l94_94668

-- Define the conditions given in the problem
def apples : ℕ := 2
def pears : ℕ := 3
def days : ℕ := 5

-- The statement that needs to be proven
theorem distribution_ways : nat.choose days apples = 10 :=
by sorry

end distribution_ways_l94_94668


namespace integral_inequality_l94_94207

noncomputable def F (x : ℝ) : ℝ :=
  (∫ y in 0..x, sqrt (1 + (cos y)^2)) - (sqrt (x^2 + (sin x)^2))

theorem integral_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) :
  (∫ y in 0..1, sqrt (1 + (cos y)^2)) ≥ sqrt (x^2 + (sin x)^2) :=
begin
  -- Proof would go here
  sorry
end

end integral_inequality_l94_94207


namespace last_two_nonzero_digits_of_factorial_90_l94_94574

theorem last_two_nonzero_digits_of_factorial_90 :
  let n := 90!
  ∃ (x : ℕ), x < 100 ∧ (x % 10 ≠ 0) ∧ ((0 ≤ x ∧ x ≤ 99) → x = 12) :=
begin
  sorry
end

end last_two_nonzero_digits_of_factorial_90_l94_94574


namespace yanni_paintings_l94_94628

theorem yanni_paintings
  (total_area : ℤ)
  (painting1 : ℕ → ℤ × ℤ)
  (painting2 : ℤ × ℤ)
  (painting3 : ℤ × ℤ)
  (num_paintings : ℕ) :
  total_area = 200
  → painting1 1 = (5, 5)
  → painting1 2 = (5, 5)
  → painting1 3 = (5, 5)
  → painting2 = (10, 8)
  → painting3 = (5, 9)
  → num_paintings = 5 := 
by
  sorry

end yanni_paintings_l94_94628


namespace number_of_routes_l94_94675

structure RailwayStation :=
  (A B C D E F G H I J K L M : ℕ)

def initialize_station : RailwayStation :=
  ⟨1, 1, 1, 1, 2, 2, 3, 3, 3, 6, 9, 9, 18⟩

theorem number_of_routes (station : RailwayStation) : station.M = 18 :=
  by sorry

end number_of_routes_l94_94675


namespace exists_route_within_republic_l94_94925

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l94_94925


namespace min_value_arith_geo_seq_l94_94250

theorem min_value_arith_geo_seq (A B C D : ℕ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : 0 < D)
  (h_arith : C - B = B - A) (h_geo : C * C = B * D) (h_frac : 4 * C = 7 * B) :
  A + B + C + D = 97 :=
sorry

end min_value_arith_geo_seq_l94_94250


namespace sum_of_intersections_l94_94722

theorem sum_of_intersections :
  (∃ x1 y1 x2 y2 x3 y3 x4 y4, 
    y1 = (x1 - 1)^2 ∧ y2 = (x2 - 1)^2 ∧ y3 = (x3 - 1)^2 ∧ y4 = (x4 - 1)^2 ∧
    x1 - 2 = (y1 + 1)^2 ∧ x2 - 2 = (y2 + 1)^2 ∧ x3 - 2 = (y3 + 1)^2 ∧ x4 - 2 = (y4 + 1)^2 ∧
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4) = 2) :=
sorry

end sum_of_intersections_l94_94722


namespace geometric_mean_problem_l94_94951

theorem geometric_mean_problem
  (a : Nat) (a1 : Nat) (a8 : Nat) (r : Rat) 
  (h1 : a1 = 6) (h2 : a8 = 186624) 
  (h3 : a8 = a1 * r^7) 
  : a = a1 * r^3 → a = 1296 := 
by
  sorry

end geometric_mean_problem_l94_94951


namespace find_number_l94_94289

theorem find_number :
  ∃ x : ℕ, 3 * x = (26 - x) + 10 ∧ x = 9 :=
by
  exists 9
  split
  · simp
  · simp

end find_number_l94_94289


namespace club_members_count_l94_94318

theorem club_members_count :
  ∃ (n : ℕ), n = 10 ∧
  (∀ (C : Fin 5 → Fin 2 → Prop),
    (∀ a b c, a ≠ b → C a = C b → c ∈ C a ∩ C b) ∧
    (∀ a b, ∃! x, C a b x) → true) :=
begin
  -- The proof will go here.
  sorry
end

end club_members_count_l94_94318


namespace min_sum_bn_l94_94809

theorem min_sum_bn (b : ℕ → ℤ) 
  (h₁ : ∀ n, b n = 2 * n - 31) 
  (h₂ : ∀ n, b n ∈ Int) :
  ∃ n : ℕ, n = 15 ∧ (∀ m : ℕ, m ≠ 15 → (∑ i in Finset.range (m + 1), b i) > (∑ i in Finset.range (15 + 1), b i)) :=
sorry

end min_sum_bn_l94_94809


namespace three_digit_sum_9_l94_94791

theorem three_digit_sum_9 : 
  {abc : ℕ // 100 ≤ abc ∧ abc < 1000 ∧ (abc.digits 10).sum = 9}.card = 45 := 
by
  sorry

end three_digit_sum_9_l94_94791


namespace minimal_f_n_l94_94388

theorem minimal_f_n (n : ℤ) (hn : n ≥ 4) : 
  ∃ f : ℤ, (∀ m : ℤ, ∃ s : Finset ℤ, s ⊆ Finset.range (m + n + 2) ∧ s.card = f ∧ (∀ t ⊆ s, t.card = 3 → t.prod (λ x, x.gcd = 1))) ∧ 
  f = (⌊(n + 1)/2⌋ + ⌊(n + 1)/3⌋ - ⌊(n + 1)/6⌋ + 1 : ℤ) := sorry


end minimal_f_n_l94_94388


namespace find_legs_l94_94124

-- Given conditions
variables {c S : ℝ} (h : S^2 ≥ 2 * c^2)

-- Definitions of the legs a and b based on the given conditions
def leg_a := (S + real.sqrt (2 * c^2 - S^2)) / 2
def leg_b := (S - real.sqrt (2 * c^2 - S^2)) / 2

-- The theorem statement we want to prove
theorem find_legs (h₀ : 0 < c)(h₁ : S > 0):
  ∃ (a b : ℝ), a + b = S ∧ a^2 + b^2 = c^2 ∧ a = leg_a ∧ b = leg_b :=
sorry

end find_legs_l94_94124


namespace largest_consecutive_integer_product_2520_l94_94743

theorem largest_consecutive_integer_product_2520 :
  ∃ (n : ℕ), n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧ (n + 3) = 8 :=
by {
  sorry
}

end largest_consecutive_integer_product_2520_l94_94743


namespace tournament_cycle_count_l94_94156

-- Tournament setup
def teams := {t : ℕ | t < 19} -- 19 teams

-- Each team plays each other exactly once
def played_exactly_once (X Y : ℕ) (hX : X ∈ teams) (hY : Y ∈ teams) : Prop := 
  X ≠ Y -- No team plays against itself and every pair meets exactly once

-- Each team won 9 games and lost 9 games
def team_record (X : ℕ) (hX : X ∈ teams) : Prop := 
  ∃ wins losses : ℕ,
    wins = 9 ∧ losses = 9

-- Definitions for beat relationship (assuming beat is transitive and exactly defined)
def beat (A B : ℕ) (hA : A ∈ teams) (hB : B ∈ teams) : Prop := sorry

noncomputable def number_of_sets_ABC :=
  let total_teams := 19 in
  let total_sets := nat.choose total_teams 3 in
  let dominant_sets_per_team := nat.choose 9 2 in
  let total_dominant_sets := total_teams * dominant_sets_per_team in
  total_sets - total_dominant_sets

theorem tournament_cycle_count : 
  number_of_sets_ABC = 285 := sorry

end tournament_cycle_count_l94_94156


namespace intersection_sets_l94_94977

theorem intersection_sets :
  (let S := {x : ℝ | (x + 5) / (5 - x) > 0} in
   let T := {x : ℝ | x^2 + 4 * x - 21 < 0} in
   S ∩ T = {x : ℝ | -5 < x ∧ x < 3}) :=
by
  let S := {x : ℝ | (x + 5) / (5 - x) > 0}
  let T := {x : ℝ | x^2 + 4 * x - 21 < 0}
  show S ∩ T = {x : ℝ | -5 < x ∧ x < 3}
sorry

end intersection_sets_l94_94977


namespace exists_route_within_same_republic_l94_94889

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l94_94889


namespace largest_integer_in_interval_l94_94772

theorem largest_integer_in_interval :
  (∃ x : ℤ, (1 : ℚ) / 4 < x / 6 ∧ x / 6 < 7 / 11 ∧ ∀ y : ℤ, (1 : ℚ) / 4 < y / 6 → y / 6 < 7 / 11 → y ≤ x) :=
begin
  use 3,
  split,
  { norm_num,
    split,
    { norm_num },
    { intros y hy1 hy2,
      norm_num at hy1 hy2,
      linarith }
  },
  sorry
end

end largest_integer_in_interval_l94_94772


namespace ratio_of_gerald_to_polly_speed_l94_94539

noncomputable def ratio_ger_spec_to_pol_speed : ℕ := 
  let track_length : ℝ := 0.25
  let laps_by_polly := 12
  let time_polly_hours := 0.5
  let dist_polly := laps_by_polly * track_length
  let speed_polly := dist_polly / time_polly_hours
  let speed_gerald := 3 
  let ratio := speed_gerald / speed_polly
  ratio 

theorem ratio_of_gerald_to_polly_speed : ratio_ger_spec_to_pol_speed = 1/2 := 
by sorry

end ratio_of_gerald_to_polly_speed_l94_94539


namespace exists_airline_route_within_same_republic_l94_94915

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l94_94915


namespace proof_AXDY_eq_BXCY_l94_94251

noncomputable def are_points_collinear (a b c : Point) : Prop :=
a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) = 0

noncomputable def quadrilateral_inscribed (A B C D I : Point) : Prop :=
are_points_collinear A B C ∧ are_points_collinear A D C ∧ dist AK = dist IL

theorem proof_AXDY_eq_BXCY
  {A B C D I X Y : Point}
  (h1 : quadrilateral_inscribed A B C D I)
  (h2 : ∠ BAD + ∠ ADC > π)
  (h3 : are_points_collinear I X Y)
  (h4 : dist I X = dist I Y) :
  dist A X * dist D Y = dist B X * dist C Y :=
sorry

end proof_AXDY_eq_BXCY_l94_94251


namespace throws_to_return_to_ben_is_13_l94_94066

def throws_to_return_to_ben (num_boys : ℕ) (skip : ℕ) : ℕ :=
  if h : 0 < num_boys then
    let rec helper : ℕ → ℕ → ℕ
      | curr_pos, count :=
        if curr_pos = 1 ∧ count > 0 then
          count
        else
          helper ((curr_pos + skip) % num_boys + 1) (count + 1)
    in helper 1 0
  else
    0

theorem throws_to_return_to_ben_is_13 : throws_to_return_to_ben 14 5 = 13 := 
by
  -- Here, we would include the proof.
  sorry

end throws_to_return_to_ben_is_13_l94_94066


namespace sum_of_exponents_l94_94868

theorem sum_of_exponents (n : ℕ) (h : n = 2^11 + 2^10 + 2^5 + 2^4 + 2^2) : 11 + 10 + 5 + 4 + 2 = 32 :=
by {
  -- The proof could be written here
  sorry
}

end sum_of_exponents_l94_94868


namespace train_crossing_time_l94_94594

noncomputable def km_per_hr_to_m_per_s (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 5 / 18

noncomputable def crossing_time (length1 length2 speed_kmh1 speed_kmh2 : ℝ) : ℝ :=
  let relative_speed := km_per_hr_to_m_per_s (speed_kmh1 + speed_kmh2)
  let combined_length := length1 + length2
  combined_length / relative_speed

theorem train_crossing_time :
  crossing_time 300 450 60 40 ≈ 27 := sorry

end train_crossing_time_l94_94594


namespace shaded_area_percentage_l94_94305

variables (A B C D M N : Type) -- Points on the rectangle
variables [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq M] [decidable_eq N]
variables (rectangle : Π (x y : Type), Prop) -- definition of rectangle
variables (midpoint : Π (x y z : Type), Prop) -- definition of midpoint
variables (distance : Π (x y : Type), ℝ) -- distance function

variables (CM : ℝ) (NC : ℝ) (BC : ℝ) (CD : ℝ) (area_rectangle : ℝ) (area_triangle : ℝ) (shaded_area : ℝ) (percentage_shaded : ℝ)

axiom rectangle_ABCD : rectangle A B C D -- ABCD is a rectangle
axiom midpoint_M : midpoint B C M -- M is the midpoint of BC
axiom midpoint_N : midpoint C D N -- N is the midpoint of CD

axiom CM_value : distance C M = 4 -- CM = 4
axiom NC_value : distance N C = 5 -- NC = 5

-- Calculate BC and CD
noncomputable def BC_calc : ℝ := 2 * distance C M
noncomputable def CD_calc : ℝ := 2 * distance N C

axiom BC_value : BC = BC_calc -- BC = 8
axiom CD_value : CD = CD_calc -- CD = 10

-- Calculate the area of the rectangle
noncomputable def area_rectangle_calc : ℝ := distance B C * distance C D
axiom area_rectangle_value : area_rectangle = area_rectangle_calc -- Area of ABCD = 80

-- Calculate the area of triangle NCM
noncomputable def area_triangle_calc : ℝ := 0.5 * distance N C * distance C M
axiom area_triangle_value : area_triangle = area_triangle_calc -- Area of triangle NCM = 10

-- Calculate the shaded area
noncomputable def shaded_area_calc : ℝ := area_rectangle - area_triangle
axiom shaded_area_value : shaded_area = shaded_area_calc -- Shaded area = 70

-- Calculate the percentage of shaded area
noncomputable def percentage_shaded_calc : ℝ := (shaded_area / area_rectangle) * 100
axiom percentage_shaded_value : percentage_shaded = percentage_shaded_calc -- Percentage shaded = 87.5

-- Proof statement
theorem shaded_area_percentage : percentage_shaded = 87.5 := 
begin
  rw [percentage_shaded_value],
  rw [shaded_area_value],
  rw [area_triangle_value],
  rw [area_rectangle_value],
  rw [CD_value],
  rw [BC_value],
  rw [NC_value],
  rw [CM_value],
  sorry
end

end shaded_area_percentage_l94_94305


namespace determine_2002_tuples_l94_94510

theorem determine_2002_tuples (b : ℕ) (a : fin 2002 → ℕ) (H1 : 0 < b)
  (H2 : ∑ j, (a j) ^ (a j) = 2002 * b ^ b) :
  ∃ (c : fin 2002 → ℕ), 
    (∀ i, i < b → a i = b) ∧
    (∀ i, b ≤ i → a i = 0) :=
by sorry

end determine_2002_tuples_l94_94510


namespace range_a_l94_94421

noncomputable def cond1 (x : ℝ) : Prop := 6 + 5 * x - x^2 > 0
def A : set ℝ := { x | -1 < x ∧ x < 6 }

def cond2 (x a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≥ 0
def B (a : ℝ) : set ℝ := { x | cond2 x a }

def p (x : ℝ) : Prop := x ∈ A
def q (x a : ℝ) : Prop := x ∈ B a

theorem range_a (a : ℝ) (h : 0 < a) (H : ∀ x, ¬p x → q x a) : 0 < a ∧ a ≤ 2 :=
sorry

end range_a_l94_94421


namespace right_triangle_angle_ratios_l94_94597

theorem right_triangle_angle_ratios (a b c : ℝ)
  (h : a^2 + b^2 = c^2) -- The Pythagorean theorem for a right triangle
  (h_ratio : (a / b)^2 = 1 / 5) : 
  ∃ γ β : ℝ, γ = Real.arctan (1 / Real.sqrt 5) ∧ β = π / 2 - γ := 
begin
  sorry
end

end right_triangle_angle_ratios_l94_94597


namespace problem1_solution_problem2_solution_l94_94414

open Real

noncomputable def ellipse_foci (a b : ℝ) (h_b : 0 < b ∧ b < 10) : Prop :=
  ∃ (c : ℝ), c = sqrt (a^2 - b^2) ∧ 2 * a = 20

noncomputable def condition1 (F₁ F₂ P : ℝ × ℝ) (a b : ℝ) (h_b : 0 < b ∧ b < 10) : Prop :=
  ∃ (t₁ t₂ : ℝ), (t₁ + t₂ = 20) ∧ (t₁ * t₂ ≤ 100)

noncomputable def condition2 (F₁ F₂ P : ℝ × ℝ) (a b : ℝ) (h_b : 0 < b ∧ b < 10) : Prop :=
  ∃ (c : ℝ), c = sqrt (a^2 - b^2) ∧ 
             let t₁ t₂ := (
               dist P F₁,
               dist P F₂
             ) in ( t₁ + t₂ = 20) ∧ (∠ F₁ P F₂ = π / 3) ∧ 
                 (abs (1/2 * t₁ * t₂ * sin (π / 3)) = 64 * sqrt 3 / 3) ∧ 
                 (b = 8)

theorem problem1_solution (F₁ F₂ P : ℝ × ℝ) (a b : ℝ) (h_b : 0 < b ∧ b < 10) :
  ellipse_foci a b h_b → condition1 F₁ F₂ P a b h_b :=
  sorry

theorem problem2_solution (F₁ F₂ P : ℝ × ℝ) (a b : ℝ) (h_b : 0 < b ∧ b < 10) :
  ellipse_foci a b h_b → condition2 F₁ F₂ P a b h_b :=
  sorry

end problem1_solution_problem2_solution_l94_94414


namespace number_of_valid_permutations_l94_94779

theorem number_of_valid_permutations : 
  let valid_permutation (a : ℕ → ℕ) := ∀ i, 1 ≤ i ∧ i < 10 → a (i + 1) ≥ a i - 1
  in {a : ℕ → ℕ | valid_permutation a}.card = 512 :=
by
  sorry

end number_of_valid_permutations_l94_94779


namespace correct_statement_l94_94292

theorem correct_statement (a b c : ℝ) (h : c ≠ 0) : 
  (a / c = b / c) ↔ (a = b) := 
by 
  intros h₁ 
  calc 
    a = b : by sorry

end correct_statement_l94_94292


namespace range_of_a_l94_94112

theorem range_of_a (a : ℝ) :
  (∃ x ∈ Icc 0 π, cos x - sin x + a = 0) ↔ a ∈ Icc (-1) (sqrt 2) := 
sorry

end range_of_a_l94_94112


namespace perimeter_of_square_l94_94564

theorem perimeter_of_square (area : ℝ) (h : area = 520) : ∃ s : ℝ, s^2 = area ∧ 4 * s = 8 * real.sqrt 130 :=
by
  sorry

end perimeter_of_square_l94_94564


namespace eggs_in_seven_boxes_l94_94584

-- define the conditions
def eggs_per_box : Nat := 15
def number_of_boxes : Nat := 7

-- state the main theorem to prove
theorem eggs_in_seven_boxes : eggs_per_box * number_of_boxes = 105 := by
  sorry

end eggs_in_seven_boxes_l94_94584


namespace the_ship_is_safe_l94_94651

def shipNotAffectedByTyphoon : Prop :=
  ∀ (ship typhoon port : ℝ×ℝ),
    (typhoon = (-70, 0)) →
    (port = (0, 40)) →
    (dist (typhoon, ship) > 30) →
    ∃ (safeDistance : ℝ), safeDistance > 30 ∧ dist (ship, typhoon) > safeDistance

theorem the_ship_is_safe :
  shipNotAffectedByTyphoon :=
by 
  sorry

end the_ship_is_safe_l94_94651


namespace infinite_solutions_l94_94511

noncomputable def f : (ℕ → ℝ) → (ℝ → ℝ)
| _ f := f

theorem infinite_solutions {f : ℝ → ℝ} 
  (h1 : ∀ a > 0, ∃ x ≥ 1, f x = a * x) 
  (h2 : continuous_on f (set.Ici 1)) : 
  ∀ a > 0, set.infinite {x | f x = a * x ∧ x ≥ 1} :=
by
  intros a ha
  sorry

end infinite_solutions_l94_94511


namespace solution_set_inequality_l94_94731

theorem solution_set_inequality (x : ℝ) : (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1 / 3) :=
by
  sorry

end solution_set_inequality_l94_94731


namespace total_dollars_is_correct_l94_94176

-- Definitions for the fractions owned by John and Alice.
def johnDollars : ℚ := 5 / 8
def aliceDollars : ℚ := 7 / 20

-- Definition for the total amount of dollars.
def totalDollars : ℚ := johnDollars + aliceDollars

-- Statement of the theorem to prove.
theorem total_dollars_is_correct : totalDollars = 39 / 40 := 
by
  /- proof omitted -/
  sorry

end total_dollars_is_correct_l94_94176


namespace characteristic_triangle_smallest_angle_l94_94623

theorem characteristic_triangle_smallest_angle 
  (α β : ℝ)
  (h1 : α = 2 * β)
  (h2 : α = 100)
  (h3 : β + α + γ = 180) : 
  min α (min β γ) = 30 := 
by 
  sorry

end characteristic_triangle_smallest_angle_l94_94623


namespace min_colors_collinear_points_l94_94364

theorem min_colors_collinear_points (n : ℕ) (h : n > 1) : 
  ∃ (c : ℕ), ∀ (S : set $fin n$),
  (∀ (X Y : $fin n$), 
    X ≠ Y → 
    ∃ (circle_XY : set point),
    (diameter circle_XY = XY) ∧
    (∀ (c1 c2 : set point), 
      (intersect_at_two_points c1 c2 → same_color c1 c2 = false))) → 
  c = if n = 3 then 1 else nat.ceil (n / 2 : ℝ) :=
begin
  sorry, -- proof not required
end

end min_colors_collinear_points_l94_94364


namespace max_possible_value_l94_94973

noncomputable def max_value_expr (x : ℝ) : ℝ :=
  (3 * x^2 + 2 - real.sqrt (9 * x^4 + 4)) / x

theorem max_possible_value (x : ℝ) (hx : 0 < x) : 
  ∃ y : ℝ, y = max_value_expr x ∧ y ≤ 12 / (5 + 3 * real.sqrt 3) := 
by
  sorry

end max_possible_value_l94_94973


namespace problem_l94_94788

variable (ℝ : Type) [LinearOrderedField ℝ]

def op (r s : ℝ) : ℝ :=
  if s = 0 then r else if r = 0 then s else sorry

def condition1 (r : ℝ) : op ℝ r 0 = r := sorry
def condition2 (r s : ℝ) : op ℝ r s = op ℝ s r := sorry
def condition3 (r s : ℝ) : op ℝ (r + 1) s = (op ℝ r s) + s + 2 := sorry

theorem problem : (op ℝ 10 4 = 58) :=
  by
    rw [condition2 10 4]
    sorry

end problem_l94_94788


namespace circles_are_tangent_externally_l94_94144

theorem circles_are_tangent_externally :
  let O1_center := (2 : ℝ, 0 : ℝ)
  let O2_center := (-1 : ℝ, 0 : ℝ)
  let R1 := (1 : ℝ)
  let R2 := (3 : ℝ)
  dist O1_center O2_center = R1 + R2 :=
by
  sorry

end circles_are_tangent_externally_l94_94144


namespace no_real_solution_log_eqn_l94_94361

theorem no_real_solution_log_eqn (x : ℝ) :
  (0 < x + 5) ∧ (0 < x - 3) ∧ (0 < x^2 - 4x - 15) → 
  ¬ (log (x + 5) + log (x - 3) = log (x^2 - 4x - 15)) :=
by 
  intro h,
  cases h with h1 h,
  cases h with h2 h3,
  sorry

end no_real_solution_log_eqn_l94_94361


namespace incorrect_statement_E_l94_94870

variable (x : ℝ)

-- Definitions based on given conditions
def log_b_x (b x : ℝ) := Real.log x / Real.log b
def y := log_b_x 10 x
def statement_A : Prop := x = 1 → y = 0
def statement_B : Prop := x = 10 → y = 1
def statement_C : Prop := x = -1 → ∃ (z: ℂ), z = log_b_x 10 x 
def statement_D : Prop := 0 < x ∧ x < 1 → y < 0 ∧ ∀ ε > 0, 0 < x ∧ x < ε → y < -Real.log ε / Real.log 10
def statement_E : Prop := ¬(statement_A ∧ statement_B ∧ statement_C ∧ statement_D)

-- Theorem stating the proof problem
theorem incorrect_statement_E :
  statement_A ∧ statement_B ∧ statement_C ∧ statement_D ∧ statement_E = false :=
sorry

end incorrect_statement_E_l94_94870


namespace max_subset_cardinality_l94_94129

variable {n : ℕ}

def S (n : ℕ) : Set ℕ :=
  {x | 1 ≤ x ∧ x ≤ 2 * n + 1}

def validSubset (T : Set ℕ) : Prop :=
  ∀ x y z ∈ T, x ≠ y → y ≠ z → x ≠ z → x + y ≠ z

theorem max_subset_cardinality (T : Set ℕ) (hT₁ : T ⊆ S n) (hT₂ : validSubset T) :
  T.card ≤ n + 1 := 
  sorry

end max_subset_cardinality_l94_94129


namespace even_perfect_square_factors_count_l94_94861

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_factor (a b c : ℕ) : Prop := 0 ≤ a ∧ a ≤ 6 ∧ 0 ≤ b ∧ b ≤ 10 ∧ 0 ≤ c ∧ c ≤ 2

theorem even_perfect_square_factors_count :
  (∑ a in {n | n ≤ 6 ∧ is_even n ∧ n ≥ 2}, 
   ∑ b in {m | m ≤ 10 ∧ is_even m}, 
   ∑ c in {p | p ≤ 2 ∧ is_even p}, 1) = 36 := 
by 
  sorry

end even_perfect_square_factors_count_l94_94861


namespace constant_term_of_expansion_l94_94418

theorem constant_term_of_expansion (n : ℕ) (h : n = 6) :
  constant_term (x * (1 - (2 / sqrt x))^n) = 60 :=
by
  sorry

end constant_term_of_expansion_l94_94418


namespace sum_of_fourth_powers_l94_94445

theorem sum_of_fourth_powers (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 37 / 6 := 
sorry

end sum_of_fourth_powers_l94_94445


namespace probability_three_tails_one_head_l94_94464

theorem probability_three_tails_one_head :
  (nat.choose 4 1) * (1/2)^4 = 1/4 :=
by
  sorry

end probability_three_tails_one_head_l94_94464


namespace smallest_prime_factor_of_54_l94_94217

def has_prime_factor (n m : ℕ) : Prop :=
  ∀ p: ℕ, Nat.Prime p → p ∣ n → p ≥ m

theorem smallest_prime_factor_of_54 (C : Set ℕ) (hC : C = {51, 53, 54, 55, 57}) :
  ∃ n ∈ C, has_prime_factor 54 n := by
synth

end smallest_prime_factor_of_54_l94_94217


namespace pits_dug_by_six_in_five_hours_l94_94587

theorem pits_dug_by_six_in_five_hours : 
  (∀ (pits diggers hours : ℕ), 
    (diggers = 3 ∧ pits = 3 ∧ hours = 2) → 
    (diggers * hours = 6 ∧ pits = 3)) → 
  ∃ (pits_in_five_hours : ℕ), pits_in_five_hours = 15 :=
by
  assume h : ∀ (pits diggers hours : ℕ), (diggers = 3 ∧ pits = 3 ∧ hours = 2) → (diggers * hours = 6 ∧ pits = 3)
  sorry

end pits_dug_by_six_in_five_hours_l94_94587


namespace minimize_PQ_of_perpendicular_foot_l94_94067

-- Definitions and conditions
variables {A B C M P Q : Type}
variables [triangle A B C] [point_on_side M A B]
variables [perpendicular MP M B C] [perpendicular MQ M A C]

-- The main theorem
theorem minimize_PQ_of_perpendicular_foot :
  (∀ M : point_on_side M A B,
   ∀ P Q : Type,
   perpendicular MP M B C → 
   perpendicular MQ M A C →
   length_segment PQ ≥ length_segment PQ_min) :=
begin
  sorry
end

end minimize_PQ_of_perpendicular_foot_l94_94067


namespace cole_fence_cost_l94_94032

def side_length : ℝ := 9
def back_length : ℝ := 18
def cost_per_foot : ℝ := 3

def total_cost : ℝ :=
  let side_cost := cost_per_foot * side_length
  let cole_left_cost := side_cost * (2/3)
  let cole_back_cost := (cost_per_foot * back_length) / 2
  cole_left_cost + side_cost + cole_back_cost

theorem cole_fence_cost : total_cost = 72 := by sorry

end cole_fence_cost_l94_94032


namespace tire_mileage_l94_94658

theorem tire_mileage (n m total_miles : ℕ) (h1 : n = 6) (h2 : m = 5) (h3 : total_miles = 42000) :
  let total_tire_miles := total_miles * m
  in total_tire_miles / n = 35000 :=
by {
  sorry
}

end tire_mileage_l94_94658


namespace det_C2_Dinv_l94_94447

variable (C D : Matrix n n ℝ)

-- Given conditions
axiom det_C : det C = 3
axiom det_D : det D = 7

-- Proof statement
theorem det_C2_Dinv : det (C^2 * D⁻¹) = 9 / 7 := by
  sorry

end det_C2_Dinv_l94_94447


namespace clinton_shoes_count_l94_94712

def num_hats : ℕ := 5
def num_belts : ℕ := num_hats + 2
def num_shoes : ℕ := 2 * num_belts

theorem clinton_shoes_count : num_shoes = 14 := by
  -- proof goes here
  sorry

end clinton_shoes_count_l94_94712


namespace find_m_l94_94099

theorem find_m (x : ℝ) (m : ℝ) (h1 : log10 (tan x + cot x) = 1) (h2 : log10 (tan x * cot x) = log10 m - 2) :
  m = 100 :=
sorry

end find_m_l94_94099


namespace black_white_area_ratio_l94_94393

-- Define the radii of the circles
def radii : List ℝ := [2, 4, 6, 8]

-- Define areas of the concentric circles
def areas (r : ℝ) : ℝ := π * r^2

-- Define the black and white areas
def black_areas := [areas (radii[0]), areas (radii[2]) - areas (radii[1])]
def white_areas := [areas (radii[1]) - areas (radii[0]), areas (radii[3]) - areas (radii[2])]

-- Total areas painted black and white
def total_black_area := black_areas.sum
def total_white_area := white_areas.sum

-- Define the expected ratio
def expected_ratio : ℝ := 3 / 5

-- The theorem we need to prove
theorem black_white_area_ratio :
  total_black_area / total_white_area = expected_ratio := 
by
  sorry

end black_white_area_ratio_l94_94393


namespace factorial_fraction_is_integer_l94_94546

theorem factorial_fraction_is_integer (m n : ℕ) : ∃ k : ℤ, ((2 * m)!.toRational * (2 * n)!.toRational) / (m!.toRational * n!.toRational * (m + n)!.toRational) = k :=
by {
  sorry
}

end factorial_fraction_is_integer_l94_94546


namespace correct_option_B_incorrect_option_A_incorrect_option_C_incorrect_option_D_l94_94624

-- Define the conditions as hypotheses
variables {a : ℝ}

-- Statements of each option based on the conditions
def option_A := (a ^ 2) ^ 3 = a ^ 5
def option_B := a ^ 4 / a = a ^ 3
def option_C := a ^ 2 * a ^ 3 = a ^ 6
def option_D := a ^ 2 + a ^ 3 = a ^ 5

-- Prove the correct option (option B)
theorem correct_option_B : option_B :=
by sorry

-- Prove other options are incorrect
theorem incorrect_option_A : ¬ option_A :=
by sorry

theorem incorrect_option_C : ¬ option_C :=
by sorry

theorem incorrect_option_D : ¬ option_D :=
by sorry

end correct_option_B_incorrect_option_A_incorrect_option_C_incorrect_option_D_l94_94624


namespace intersection_distances_sum_l94_94494

theorem intersection_distances_sum : 
  let C := { p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4 }
  let l := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  let M : ℝ × ℝ := (0, 1)
  let intersection_points := C ∩ l
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ abs (dist M A + dist M B) = 3 * real.sqrt 2
:=
by
  -- definitions and problem setup
  let C : set (ℝ × ℝ) := { p | (p.1 - 2)^2 + p.2^2 = 4 }
  let l : set (ℝ × ℝ) := { p | p.1 + p.2 = 1 }
  let M : ℝ × ℝ := (0, 1)
  let intersection_points : set (ℝ × ℝ) := C ∩ l

  sorry

end intersection_distances_sum_l94_94494


namespace james_total_cost_l94_94954

/-- 
  Price of 15 dollars per pound.
  James buys 10 pounds of steak A and 14 pounds of steak B.
  Buy two get one free offer.
  12% discount on the total price of steak A.
  Service fee of 7.5 dollars.
  The total cost paid by James should be 249.90 dollars.
-/
def price_per_pound : ℝ := 15
def steak_A_pounds : ℝ := 10
def steak_B_pounds : ℝ := 14
def discount_rate : ℝ := 0.12
def service_fee : ℝ := 7.5
def expected_total_cost : ℝ := 249.90

theorem james_total_cost :
  let total_cost := (steak_A_pounds * price_per_pound * (2/3) * (1 - discount_rate)) +
                    (steak_B_pounds * price_per_pound * (2/3)) + service_fee
  in total_cost = expected_total_cost :=
by
  -- Proof goes here
  sorry

end james_total_cost_l94_94954


namespace solution_set_m5_range_m_sufficient_condition_l94_94431

theorem solution_set_m5 (x : ℝ) : 
  (|x + 1| + |x - 2| > 5) ↔ (x < -2 ∨ x > 3) := 
sorry

theorem range_m_sufficient_condition (x m : ℝ) (h : ∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) : 
  m ≤ 1 := 
sorry

end solution_set_m5_range_m_sufficient_condition_l94_94431


namespace max_radius_of_circle_l94_94315

theorem max_radius_of_circle : 
  ∃ (a b R : ℝ), (∀ x y : ℝ, (x = 0 ∧ y = 11) ∨ (x = 0 ∧ y = -11) → sqrt ((a - x)^2 + (b - y)^2) = R) ∧
                 (∀ x y : ℝ, x^2 + y^2 < 1 → (a - x)^2 + (b - y)^2 < R^2) ∧
                 R = sqrt 122 :=
sorry

end max_radius_of_circle_l94_94315


namespace number_of_motorcycles_l94_94159

theorem number_of_motorcycles (M : ℕ) (H1 : ∀c : ℕ, number_of_cars = 19 → c = 5) (H2 : ∀m : ℕ, number_of_motorcycles = M → m = 2) 
    (H3 : number_of_cars = 19) (H4 : total_wheels = 117) : M = 11 :=
by
    sorry

end number_of_motorcycles_l94_94159


namespace tan_alpha_frac_simplification_l94_94396

theorem tan_alpha_frac_simplification (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 4 / 3 :=
by sorry

end tan_alpha_frac_simplification_l94_94396


namespace hotel_fee_original_flat_fee_l94_94323

theorem hotel_fee_original_flat_fee
  (f n : ℝ)
  (H1 : 0.85 * (f + 3 * n) = 210)
  (H2 : f + 6 * n = 400) :
  f = 94.12 :=
by
  -- Sorry is used to indicate that the proof is not provided
  sorry

end hotel_fee_original_flat_fee_l94_94323


namespace proof_problem_l94_94847

theorem proof_problem (a b : ℝ) (H1 : ∀ x : ℝ, (ax^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, (ax^2 - (a*c + b)*x + b*c < 0) ↔ 
   (if c > 2 then 2 < x ∧ x < c
    else if c < 2 then c < x ∧ x < 2
    else false)) :=
by
  sorry

end proof_problem_l94_94847


namespace problem1_problem2_l94_94638

noncomputable section

open Real EuclideanSpace

variables {V : Type*} [AddCommGroup V] [Module ℝ V] [InnerProductSpace ℝ V]

variables (A B C D P Q R S O : V)
-- given conditions
variables (h1 : convex ℝ (set.insert A (set.insert B (set.insert C (set.insert D ∅)))))
variables (hP : P ∈ line_segment ℝ A B)
variables (hQ : Q ∈ line_segment ℝ B C)
variables (hR : R ∈ line_segment ℝ C D)
variables (hS : S ∈ line_segment ℝ D A)
variables (hO : O ∈ affine_span ℝ {P, R} ∩ affine_span ℝ {Q, S})
variables (hCond1 : ∃ (α : ℝ), α * (B - A) = P - A ∧ α = ∥P - A∥ / ∥B - A∥)
variables (hCond2 : ∃ (β : ℝ), β * (D - A) = S - A ∧ β = ∥S - A∥ / ∥D - A∥)
variables (hCond3 : α = ∥D - R∥ / ∥C - D∥)
variables (hCond4 : β = ∥B - Q∥ / ∥C - B∥)

theorem problem1 : ∀ (α β : ℝ), α = ∥P - A∥ / ∥B - A∥ ∧ β = ∥S - A∥ / ∥D - A∥ → ∥SO∥ / ∥SQ∥ = α :=
sorry

theorem problem2 : ∀ (α β : ℝ), α = ∥P - A∥ / ∥B - A∥ ∧ β = ∥S - A∥ / ∥D - A∥ → ∥PQ∥ / ∥PR∥ = β :=
sorry

end problem1_problem2_l94_94638


namespace triangle_area_eq_4sqrt3_over_3_l94_94525

def ellipse (a : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ (x^2 / a^2) + (y^2 / 4) = 1}

def is_focus_left (a : ℝ) (p : ℝ × ℝ) : Prop := p = (-sqrt(a^2 - 4), 0)
def is_focus_right (a : ℝ) (p : ℝ × ℝ) : Prop := p = (sqrt(a^2 - 4), 0)

def angle_eq_60 (P F1 F2 : ℝ × ℝ) : Prop :=
  let v1 := (P.1 - F1.1, P.2 - F1.2)
  let v2 := (P.1 - F2.1, P.2 - F2.2)
  let dot := v1.1 * v2.1 + v1.2 * v2.2
  let norm_v1 := real.sqrt (v1.1^2 + v1.2^2)
  let norm_v2 := real.sqrt (v2.1^2 + v2.2^2)
  let cos_theta := dot / (norm_v1 * norm_v2)
  real.arccos cos_theta = real.pi / 3

theorem triangle_area_eq_4sqrt3_over_3
  (a : ℝ) (h : 2 < a)
  (P : ℝ × ℝ) (P_on_ellipse : P ∈ ellipse a)
  (F1 F2 : ℝ × ℝ) (F1_is_focus_left : is_focus_left a F1) (F2_is_focus_right : is_focus_right a F2)
  (angle_is_60 : angle_eq_60 P F1 F2) :
  let d1 := real.sqrt((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
  let d2 := real.sqrt((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  0.5 * d1 * d2 * real.sin (real.pi / 3) = 4 * real.sqrt 3 / 3 :=
sorry

end triangle_area_eq_4sqrt3_over_3_l94_94525


namespace exists_divisible_by_sum_of_digits_l94_94543

def sum_of_digits (n : ℕ) : ℕ := 
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3

theorem exists_divisible_by_sum_of_digits :
  ∀ n : ℕ, 100 ≤ n → n + 17 ≤ 999 →
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % sum_of_digits k = 0 :=
begin
  sorry
end

end exists_divisible_by_sum_of_digits_l94_94543


namespace combined_tennis_percentage_l94_94349

/-- Definitions for the number of students and the percentage preferring tennis -/
def NorthHighSchool_students : ℕ := 1800
def SouthHighSchool_students : ℕ := 2700
def NorthHighSchool_tennis_percentage : ℕ := 25
def SouthHighSchool_tennis_percentage : ℕ := 35

/-- The theorem stating the combined percentage of students preferring tennis -/
theorem combined_tennis_percentage (n_students_north : ℕ)
       (n_students_south : ℕ)
       (perc_tennis_north : ℕ)
       (perc_tennis_south : ℕ) :
  n_students_north = NorthHighSchool_students →
  n_students_south = SouthHighSchool_students →
  perc_tennis_north = NorthHighSchool_tennis_percentage →
  perc_tennis_south = SouthHighSchool_tennis_percentage →
  let students_tennis_north := n_students_north * perc_tennis_north / 100 in
  let students_tennis_south := n_students_south * perc_tennis_south / 100 in
  let combined_students := n_students_north + n_students_south in
  let combined_tennis := students_tennis_north + students_tennis_south in
  combined_tennis * 100 / combined_students = 31 :=
begin
  intros hn_students_south hn_students_north hperc_tennis_north hperc_tennis_south,
  have students_tennis_north := NorthHighSchool_students * NorthHighSchool_tennis_percentage / 100,
  have students_tennis_south := SouthHighSchool_students * SouthHighSchool_tennis_percentage / 100,
  have combined_students := NorthHighSchool_students + SouthHighSchool_students,
  have combined_tennis := students_tennis_north + students_tennis_south,
  rw [hn_students_south, hn_students_north, hperc_tennis_north, hperc_tennis_south],
  have : combined_tennis * 100 / combined_students = (1395 * 100 / 4500 : ℕ), by sorry,
  rw this,
  norm_num,
end

#eval combined_tennis_percentage 1800 2700 25 35 sorry sorry sorry sorry

end combined_tennis_percentage_l94_94349


namespace exists_airline_route_within_same_republic_l94_94901

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l94_94901


namespace avg_height_girls_l94_94572

-- Given conditions
def num_boys := 12
def num_girls := 10
def avg_height_all_students := 103
def avg_height_boys := 108

-- To be proved: The average height of the girls is 97 cm
theorem avg_height_girls :
  let total_height_all_students := (num_boys + num_girls) * avg_height_all_students in
  let total_height_boys := num_boys * avg_height_boys in
  let total_height_girls := total_height_all_students - total_height_boys in
  total_height_girls / num_girls = 97 :=
by
  sorry

end avg_height_girls_l94_94572


namespace a_invertible_d_invertible_f_invertible_g_invertible_h_invertible_l94_94290

-- Definitions of the functions with their domains
def a (x : ℝ) : ℝ := real.sqrt (3 - x)
def b (x : ℝ) : ℝ := x^3 + 3 * x
def c (x : ℝ) : ℝ := 2 * x + 1 / x
def d (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 8
def e (x : ℝ) : ℝ := abs(x - 3) + abs(x + 2)
def f (x : ℝ) : ℝ := 2^x + 5^x
def g (x : ℝ) : ℝ := x + 2 / x
def h (x : ℝ) : ℝ := x / 3

-- Statements that certain functions have inverses
theorem a_invertible : ∀ x ∈ Iic (3 : ℝ), ∃ φ, ∀ y, φ (a y) = y :=
by sorry

theorem d_invertible : ∀ x ∈ Ici (1 : ℝ), ∃ φ, ∀ y, φ (d y) = y :=
by sorry

theorem f_invertible : ∀ x ∈ (set.univ : set ℝ), ∃ φ, ∀ y, φ (f y) = y :=
by sorry

theorem g_invertible : ∀ x ∈ Ici (0 : ℝ) − {0}, ∃ φ, ∀ y, φ (g y) = y :=
by sorry

theorem h_invertible : ∀ x ∈ Ico (-3) 6, ∃ φ, ∀ y, φ (h y) = y :=
by sorry

end a_invertible_d_invertible_f_invertible_g_invertible_h_invertible_l94_94290


namespace problem1_problem2_l94_94304
noncomputable section

-- Problem (1) Lean Statement
theorem problem1 : |-4| - (2021 - Real.pi)^0 + (Real.cos (Real.pi / 3))⁻¹ - (-Real.sqrt 3)^2 = 2 :=
by 
  sorry

-- Problem (2) Lean Statement
theorem problem2 (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) : 
  (1 + 4 / (a^2 - 4)) / (a / (a + 2)) = a / (a - 2) := 
by 
  sorry

end problem1_problem2_l94_94304


namespace distance_traveled_center_is_181_5pi_l94_94662

def cylinder_diameter := 6 -- inches
def cylinder_radius := cylinder_diameter / 2 -- inches

def arc1_radius := 150 -- inches
def arc2_radius := 90 -- inches
def arc3_radius := 120 -- inches

def adjusted_arc1_radius := arc1_radius + cylinder_radius -- inches
def adjusted_arc2_radius := arc2_radius - cylinder_radius -- inches
def adjusted_arc3_radius := arc3_radius + cylinder_radius -- inches

def quarter_circle_distance (r : ℝ) : ℝ := (1/4) * 2 * Real.pi * r -- distance for one quarter of the circumference

def distance_traveled_by_center : ℝ :=
  quarter_circle_distance adjusted_arc1_radius + 
  quarter_circle_distance adjusted_arc2_radius + 
  quarter_circle_distance adjusted_arc3_radius

theorem distance_traveled_center_is_181_5pi : distance_traveled_by_center = 181.5 * Real.pi :=
  sorry

end distance_traveled_center_is_181_5pi_l94_94662


namespace ratio_planes_bisect_volume_l94_94193

-- Definitions
def n : ℕ := 6
def m : ℕ := 20

-- Statement to prove
theorem ratio_planes_bisect_volume : (n / m : ℚ) = 3 / 10 := by
  sorry

end ratio_planes_bisect_volume_l94_94193


namespace part_a_part_b_l94_94297

-- Define the variables
variables {R r p ra rb rc a b c S : ℝ}

-- Condition definitions
def condition1 : Prop := 4 * R + r = ra + rb + rc
def condition2 : Prop := R - 2 * r ≥ 0
def condition3 : Prop := ra + rb + rc = p * (r / (p - a) + r / (p - b) + r / (p - c))
def condition4 : Prop := p * (2 * (a * b + b * c + c * a) - a^2 - b^2 - c^2) / (4 * S) ≥ sqrt 3 * p

def condition5 : Prop := 4 * R - ra = rb + rc - r
def condition6 : Prop := rb + rc - r = (p - a) * r / S
def condition7 : Prop := p^2 - a * c ≥ a^2 + b^2 + c^2 - 2 * a * c
def condition8 : Prop := 2 * (a * b + b * c + c * a) - a^2 - b^2 - c^2 + 2 * (a^2 + b^2 + c^2 - 2 * a * c) ≥ 4 * sqrt 3 * S + 2 * (a^2 + (b - c)^2)

-- Part (a) Proof Problem
theorem part_a (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 5 * R - r ≥ sqrt 3 * p :=
by sorry

-- Part (b) Proof Problem
theorem part_b (h5 : condition5) (h6 : condition6) (h7 : condition7) (h8 : condition8) : 4 * R - ra ≥ (p - a) * (sqrt 3 + (a^2 + (b - c)^2) / (2 * S)) :=
by sorry

end part_a_part_b_l94_94297


namespace volume_ratio_l94_94204

def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem volume_ratio (r1 h1 r2 h2 : ℝ) (V1 V2 : ℝ) (hc1 : r1 = 6) (hc2 : h1 = 12) (hc3 : r2 = 12) (hc4 : h2 = 6) (hv1 : V1 = volume_cylinder r1 h1) (hv2 : V2 = volume_cylinder r2 h2) :
  V1 / V2 = 1 / 2 :=
by
  -- The proof is omitted
  sorry

end volume_ratio_l94_94204


namespace ellipse_equation_hyperbola_equation_l94_94384

theorem ellipse_equation (b : ℝ) (e : ℝ) (a : ℝ) (c : ℝ) (h_minor_axis : 2 * b = 6) (h_eccentricity : e = 2 * Real.sqrt 2 / 3) (h_a : a^2 = b^2 + c^2) :
  (b = 3 ∧ a = 9 ∧ c = 6 * Real.sqrt 2) → (∀ x y : ℝ, (x^2 / 81 + y^2 / 9 = 1) ∨ (y^2 / 81 + x^2 / 9 = 1)) := sorry

theorem hyperbola_equation (a b c : ℝ) (h_ellipse : ∀ x y : ℝ, 4 * x^2 + 9 * y^2 = 36 → x^2 / 9 + y^2 / 4 = 1)
  (h_foci : c = Real.sqrt 5) (h_point: (x : ℝ) (y : ℝ), x = 4 → y = Real.sqrt 3)
  (h_system : a^2 + b^2 = c^2 ∧ 16 / a^2 - 3 / b^2 = 1) :
  (a = 2 ∧ b = 1) → (∀ x y : ℝ, x^2 / 4 - y^2 = 1) := sorry

end ellipse_equation_hyperbola_equation_l94_94384


namespace mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l94_94979

-- Definitions
def mad_hatter_clock_rate := 5 / 4
def march_hare_clock_rate := 5 / 6
def time_at_dormouse_clock := 5 -- 5:00 PM

-- Real time calculation based on clock rates
def real_time (clock_rate : ℚ) (clock_time : ℚ) : ℚ := clock_time * (1 / clock_rate)

-- Mad Hatter's and March Hare's arrival times in real time
def mad_hatter_real_time := real_time mad_hatter_clock_rate time_at_dormouse_clock
def march_hare_real_time := real_time march_hare_clock_rate time_at_dormouse_clock

-- Theorems to be proved
theorem mad_hatter_waiting_time : mad_hatter_real_time = 4 := sorry
theorem march_hare_waiting_time : march_hare_real_time = 6 := sorry

-- Main theorem
theorem waiting_time : march_hare_real_time - mad_hatter_real_time = 2 := sorry

end mad_hatter_waiting_time_march_hare_waiting_time_waiting_time_l94_94979


namespace bisection_approximation_interval_l94_94595

noncomputable def bisection_accuracy (a b : ℝ) (n : ℕ) : ℝ := (b - a) / 2^n

theorem bisection_approximation_interval 
  (a b : ℝ) (n : ℕ) (accuracy : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : accuracy = 0.01) 
  (h4 : 2^n ≥ 100) : bisection_accuracy a b n ≤ accuracy :=
sorry

end bisection_approximation_interval_l94_94595


namespace add_three_to_operation_l94_94248

def operation (a b : ℝ) : ℝ := a + (5 * a) / (3 * b)

theorem add_three_to_operation : (operation 12 9) + 3 = 17 + 2 / 9 := by
  sorry

end add_three_to_operation_l94_94248


namespace triangle_proposition_A_triangle_proposition_B_triangle_proposition_C_triangle_proposition_D_l94_94171

/--
Prove the correctness of the following propositions for a triangle ABC:
1. Given \( b^2 = ac \) and \( B = 60^\circ \), prove \( \triangle ABC \) is an equilateral triangle.
2. Given \( \sin^2(2A) = \sin^2(2B) \), prove \( \triangle ABC \) is not necessarily an isosceles triangle.
3. Given \( \cos^2(A) + \sin^2(B) + \sin^2(C) < 1 \), prove \( \triangle ABC \) is an obtuse triangle.
4. Given \( a = 4 \), \( b = 2 \), and \( B = 25^\circ \), prove there are 2 solutions for \( \triangle ABC \).
-/
theorem triangle_proposition_A {a b c : ℝ} (B : ℝ) 
  (h1 : b^2 = a * c) (h2 : B = 60) : 
  a = c ∧ B = 60 := 
sorry

theorem triangle_proposition_B {A B : ℝ} 
  (h : sin A^2 = sin B^2) : 
  ¬ (cos A + cos B = 2 * cos (A - B) / 2) := 
sorry

theorem triangle_proposition_C {A B C : ℝ} 
  (h : cos A^2 + sin B^2 + sin C^2 < 1) : 
  A > π / 2 := 
sorry

theorem triangle_proposition_D (a b B : ℝ) 
  (h1 : a = 4) (h2 : b = 2) (h3 : B = 25) : 
  exists (A C : ℝ), A + B + C = 180 ∧ a * sin(B) < b := 
sorry

end triangle_proposition_A_triangle_proposition_B_triangle_proposition_C_triangle_proposition_D_l94_94171


namespace total_volume_all_cubes_l94_94698

def volume_of_cube (s : ℕ) : ℕ := s ^ 3

def total_volume_cubes (count : ℕ) (side_length : ℕ) : ℕ :=
  count * volume_of_cube side_length

def total_volume := 216 + 500 -- precomputed values for lean efficiency

theorem total_volume_all_cubes (V_carl_cube: ℕ) (V_kate_cube: ℕ) (total_V_carl: ℕ) (total_V_kate: ℕ) : 
  V_carl_cube = volume_of_cube 3 → 
  V_kate_cube = volume_of_cube 5 → 
  total_V_carl = total_volume_cubes 8 3 → 
  total_V_kate = total_volume_cubes 4 5 → 
  (total_V_carl + total_V_kate) = 716 := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have H1 : V_carl_cube = 27 := by simp [volume_of_cube]
  have H2 : V_kate_cube = 125 := by simp [volume_of_cube]
  have H3 : total_V_carl = 216 := by simp [total_volume_cubes]
  have H4 : total_V_kate = 500 := by simp [total_volume_cubes]
  simp [H1, H2, H3, H4]
  sorry

end total_volume_all_cubes_l94_94698


namespace range_of_a_l94_94114

noncomputable def f (a x : ℝ) : ℝ := log 4 (a * x^2 - 4 * x + a)

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, ∃ y : ℝ, f a x = y) : 0 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l94_94114


namespace length_BC_l94_94592

noncomputable def circle_radius_A : ℝ := 7
noncomputable def circle_radius_B : ℝ := 3
noncomputable def distance_between_centers : ℝ := 10

theorem length_BC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (rA rB dAB : ℝ) (h_radii_A : rA = circle_radius_A) 
  (h_radii_B : rB = circle_radius_B) 
  (h_distance_AB : dAB = distance_between_centers) : 
  ∃ BC : ℝ, BC = 7.5 :=
by
  sorry

end length_BC_l94_94592


namespace scientific_notation_400000000_l94_94681

theorem scientific_notation_400000000 : 400000000 = 4 * 10^8 :=
by
  sorry

end scientific_notation_400000000_l94_94681


namespace find_k_l94_94430

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x + cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (x - π / 6) + 1 / 2

theorem find_k :
  ∀ k : ℝ, (∃ a b : ℝ, a ≠ b ∧ 0 ≤ a ∧ a ≤ π ∧ 0 ≤ b ∧ b ≤ π ∧ g a = k ∧ g b = k) ↔ k ∈ set.Ico 1 (3 / 2) :=
begin
  sorry
end

end find_k_l94_94430


namespace park_needs_minimum_37_nests_l94_94158

-- Defining the number of different birds
def num_sparrows : ℕ := 5
def num_pigeons : ℕ := 3
def num_starlings : ℕ := 6
def num_robins : ℕ := 2

-- Defining the nesting requirements for each bird species
def nests_per_sparrow : ℕ := 1
def nests_per_pigeon : ℕ := 2
def nests_per_starling : ℕ := 3
def nests_per_robin : ℕ := 4

-- Definition of total minimum nests required
def min_nests_required : ℕ :=
  (num_sparrows * nests_per_sparrow) +
  (num_pigeons * nests_per_pigeon) +
  (num_starlings * nests_per_starling) +
  (num_robins * nests_per_robin)

-- Proof Statement
theorem park_needs_minimum_37_nests :
  min_nests_required = 37 :=
sorry

end park_needs_minimum_37_nests_l94_94158


namespace part1_part2_l94_94117

-- Define the function f(x) and the conditions
def f (a x : ℝ) := (2 * x - 4) * Real.exp(x) + a * (x + 2)^2

-- Part 1: Prove that if f(x) is monotonically increasing on (0, +∞), then a ≥ 1/2
theorem part1 (a : ℝ) : (∀ x > 0, (2 * x - 2) * Real.exp(x) + 2 * a * (x + 2) ≥ 0) ↔ a ≥ 1/2 := 
sorry

-- Part 2: Prove that when 0 < a < 1/2, f(x) has a minimum value and find its range
theorem part2 (a : ℝ) (h : 0 < a ∧ a < 1/2) : 
  ∃ t ∈ Ioo(0:ℝ, 1), 
  f a t = (2 * t - 4) * Real.exp(t) - (t - 1) * (t + 2) * Real.exp(t) ∧ 
  f a t ∈ Ioo(-2 * Real.exp(1), -2) := 
sorry

end part1_part2_l94_94117


namespace hyperbola_equation_l94_94122

-- Definitions from the conditions
def hyperbola (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
def eccentricity (e a c : ℝ) : Prop := e = c / a
def semiMajorAxis (a : ℝ) : Prop := a = 4

-- Main theorem statement
theorem hyperbola_equation (e a b c : ℝ) (h_ecc : eccentricity e a c) (h_a : semiMajorAxis a) (h_hyp : hyperbola a b) :
  e = 5/4 → a = 4 → ∃ b, ∑ [x] / 16 - y^2 / 9 = 1 := 
begin
  sorry
end

end hyperbola_equation_l94_94122


namespace airline_route_same_republic_exists_l94_94927

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l94_94927


namespace PXQY_is_rectangle_l94_94642

-- Definitions
variables {A B C D X Y M N P Q : Type}

-- Preconditions
variables (is_cyclic_quadrilateral : CyclicQuadrilateral A B C D)
variables (AD_meets_BC_at_X : Intersect (Line A D) (Line B C) X)
variables (AB_meets_CD_at_Y : Intersect (Line A B) (Line C D) Y)
variables (M_midpoint_AC : Midpoint M A C)
variables (N_midpoint_BD : Midpoint N B D)
variables (MN_meets_internal_bisector_AXB_at_P : InternalBisectorIntersect P M N (Angle A X B))
variables (MN_meets_external_bisector_BYC_at_Q : ExternalBisectorIntersect Q M N (Angle B Y C))

-- Theorem to prove
theorem PXQY_is_rectangle : IsRectangle P X Q Y :=
sorry

end PXQY_is_rectangle_l94_94642


namespace find_page_words_l94_94657
open Nat

-- Define the conditions
def condition1 : Nat := 150
def condition2 : Nat := 221
def total_words_modulo : Nat := 220
def upper_bound_words : Nat := 120

-- Define properties
def is_solution (p : Nat) : Prop :=
  Nat.Prime p ∧ p ≤ upper_bound_words ∧ (condition1 * p) % condition2 = total_words_modulo

-- The theorem to prove
theorem find_page_words (p : Nat) (hp : is_solution p) : p = 67 :=
by
  sorry

end find_page_words_l94_94657


namespace cindy_envelopes_left_l94_94700

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end cindy_envelopes_left_l94_94700


namespace total_games_in_soccer_league_l94_94259

-- Definition of the problem conditions
def num_teams : ℕ := 10

-- Combination function
def combination (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- The proof problem statement
theorem total_games_in_soccer_league : combination num_teams 2 = 45 :=
by
  -- Proof goes here
  sorry

end total_games_in_soccer_league_l94_94259


namespace max_kings_8x8_min_kings_8x8_l94_94565

-- Definition for the problem conditions
def no_kings_attacking (board : ℕ → ℕ → bool) : Prop :=
  ∀ i j, board i j → ∀ di dj, (abs di ≤ 1 ∧ abs dj ≤ 1) → 
    (i ≠ i + di ∨ j ≠ j + dj) → ¬board (i + di) (j + dj)

def all_squares_covered (board : ℕ → ℕ → bool) : Prop :=
  ∀ i j, (board i j ∨ ∃ di dj, abs di ≤ 1 ∧ abs dj ≤ 1 ∧ board (i + di) (j + dj))

-- Proving the maximum number of kings
theorem max_kings_8x8 : ∃ (board : ℕ → ℕ → bool), no_kings_attacking board ∧ all_squares_covered board ∧ 
  (∑ i in finset.range 8, ∑ j in finset.range 8, if board i j then 1 else 0) = 16 := 
sorry

-- Proving the minimum number of kings
theorem min_kings_8x8 : ∃ (board : ℕ → ℕ → bool), no_kings_attacking board ∧ all_squares_covered board ∧ 
  (∑ i in finset.range 8, ∑ j in finset.range 8, if board i j then 1 else 0) = 9 := 
sorry

end max_kings_8x8_min_kings_8x8_l94_94565


namespace incorrect_conclusions_l94_94075

-- Define the vectors
def a : ℝ × ℝ × ℝ := (2, 3, -1)
def b : ℝ × ℝ × ℝ := (2, 0, 4)
def c : ℝ × ℝ × ℝ := (-4, -6, 2)

-- Helper functions
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def is_parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2, λ * v2.3)

def is_perpendicular (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The proof problem
theorem incorrect_conclusions :
  ¬ is_parallel b c ∧ ¬ is_parallel a b ∧ ¬ is_perpendicular a c :=
by {
  -- Remember that the proof steps are omitted and replaced with sorry
  sorry
}

end incorrect_conclusions_l94_94075


namespace smallest_possible_value_l94_94518

theorem smallest_possible_value 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 9 / 2 :=
sorry

end smallest_possible_value_l94_94518


namespace number_of_odd_functions_l94_94308

-- Define the functions
def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := 2^x
def f3 (x : ℝ) : ℝ := x^2 + 1
def f4 (x : ℝ) : ℝ := 2 * sin x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f(x)

-- State the problem
theorem number_of_odd_functions :
  (is_odd f1 → true) ∧
  (is_odd f2 → false) ∧
  (is_odd f3 → false) ∧
  (is_odd f4 → true) →
  (1 + 0 + 0 + 1 = 2) := by sorry


end number_of_odd_functions_l94_94308


namespace problem_l94_94867

open Complex

-- Given condition: smallest positive integer n greater than 3
def smallest_n_gt_3 (n : ℕ) : Prop :=
  n > 3 ∧ ∀ m : ℕ, m > 3 → m < n → False

-- Given condition: equation holds for complex numbers
def equation_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b * I)^n + a = (a - b * I)^n + b

-- Proof problem: Given conditions, prove b / a = 1
theorem problem (n : ℕ) (a b : ℝ)
  (h1 : smallest_n_gt_3 n)
  (h2 : 0 < a) (h3 : 0 < b)
  (h4 : equation_holds a b n) :
  b / a = 1 :=
by
  sorry

end problem_l94_94867


namespace range_of_a_l94_94472

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 9^x + (a + 4) * 3^x + 4 = 0) ↔ a ∈ Iic (-8) :=
by 
  sorry

end range_of_a_l94_94472


namespace exists_points_with_small_distance_l94_94209

noncomputable def f : ℝ → ℝ := λ x, x^3
noncomputable def g : ℝ → ℝ := λ x, x^3 + |x| + 1

theorem exists_points_with_small_distance : 
  ∃ x : ℝ, |f x - g x| ≤ 1 / 100 := 
by 
  sorry

end exists_points_with_small_distance_l94_94209


namespace intersection_points_count_l94_94247

-- Define the parametric equations for the ellipse
def ellipse_param (α : ℝ) : ℝ × ℝ := (6 * Real.cos α, 4 * Real.sin α)

-- Define the parametric equations for the circle
def circle_param (θ : ℝ) : ℝ × ℝ := (4 * Real.sqrt 2 * Real.cos θ, 4 * Real.sqrt 2 * Real.sin θ)

-- Prove the number of intersection points
theorem intersection_points_count :
  ∃ P : Set (ℝ × ℝ), P = {p : ℝ × ℝ | ∃ α θ : ℝ, p = ellipse_param α ∧ p = circle_param θ} ∧ P.finite ∧ P.card = 4 :=
by
  sorry

end intersection_points_count_l94_94247


namespace sum_distances_eq_14_l94_94111

noncomputable def C : ℝ → ℝ :=
  λ x, real.sqrt (-x^2 + 16 * x - 15)

structure Point (α : Type*) :=
  (x : α) (y : α)

def A : Point ℝ := ⟨1, 0⟩

def on_circle (p : Point ℝ) : Prop :=
  (p.x - 8)^2 + p.y^2 = 1 ∧ p.y ≥ 0

def on_parabola (p : Point ℝ) : Prop :=
 p.y^2 = 4 * p.x 

def distance (p q : Point ℝ) : ℝ :=
  real.sqrt ((p.x - q.x) ^ 2 + (p.y - q.y) ^ 2)

theorem sum_distances_eq_14 :
  ∃ B C : Point ℝ, on_circle B ∧ on_circle C ∧ on_parabola B ∧ on_parabola C ∧ (B ≠ C) ∧
  let dAB := distance A B in
  let dAC := distance A C in
  dAB + dAC = 14 :=
sorry

end sum_distances_eq_14_l94_94111


namespace combine_terms_implies_mn_l94_94866

theorem combine_terms_implies_mn {m n : ℕ} (h1 : m = 2) (h2 : n = 3) : m ^ n = 8 :=
by
  -- We will skip the proof here
  sorry

end combine_terms_implies_mn_l94_94866


namespace total_time_approx_l94_94581

noncomputable def downstream_speed (boat_speed stream_speed : ℝ) : ℝ :=
  boat_speed + stream_speed

noncomputable def upstream_speed (boat_speed stream_speed : ℝ) : ℝ :=
  boat_speed - stream_speed

noncomputable def time_taken (distance speed : ℝ) : ℝ :=
  distance / speed

noncomputable def total_time_taken (downstream_time upstream_time : ℝ) : ℝ :=
  downstream_time + upstream_time

theorem total_time_approx (boat_speed stream_speed distance : ℝ) :
  boat_speed = 20 →
  stream_speed = 2.5 →
  distance = 6523 →
  total_time_taken (time_taken distance (downstream_speed boat_speed stream_speed))
                    (time_taken distance (upstream_speed boat_speed stream_speed))
  ≈ 662.654 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have ds := downstream_speed 20 2.5
  have us := upstream_speed 20 2.5
  have t_down := time_taken 6523 ds
  have t_up := time_taken 6523 us
  have t_total := total_time_taken t_down t_up
  sorry

end total_time_approx_l94_94581


namespace problem_solution_l94_94309

-- Definitions of given constants
variables {A ω φ : ℝ}
variables {k : ℤ}

-- Given conditions
def A_pos : Prop := A > 0
def omega_pos : Prop := ω > 0
def phi_in_range : Prop := 0 ≤ φ ∧ φ ≤ 2 * Real.pi
def max_condition : Prop := (sin (Real.pi + φ) = 1)
def min_condition : Prop := (sin (6 * Real.pi + φ) = -1)

-- Definitions of our expected outcomes
def expected_function (x : ℝ) : ℝ := 3 * sin (x - Real.pi)
def monotonic_increase_interval (x : ℝ) : Prop := -4 * Real.pi + 10 * k * Real.pi ≤ x ∧ x ≤ Real.pi + 10 * k * Real.pi

-- Lean statement
theorem problem_solution :
  A_pos ∧ omega_pos ∧ phi_in_range ∧ max_condition ∧ min_condition →
  (∀ x : ℝ, expected_function x = 3 * sin (x - Real.pi)) ∧
  (∀ x : ℝ, monotonic_increase_interval x) :=
sorry

end problem_solution_l94_94309


namespace pictures_taken_at_zoo_l94_94626

-- Define the conditions
def pictures_at_zoo (Z : ℕ) : ℕ := Z
def pictures_at_museum : ℕ := 12
def deleted_pictures : ℕ := 14
def remaining_pictures : ℕ := 22

-- Define the proof problem
theorem pictures_taken_at_zoo (Z : ℕ) 
  (h : (pictures_at_zoo Z + pictures_at_museum) - deleted_pictures = remaining_pictures) : 
  Z = 24 :=
begin
  sorry
end

end pictures_taken_at_zoo_l94_94626


namespace fiftieth_digit_of_decimal_representation_of_five_fourteenth_is_five_l94_94598

theorem fiftieth_digit_of_decimal_representation_of_five_fourteenth_is_five :
  let d := (5 : ℚ) / 14 in
  let decDigits := [3, 5, 7, 1, 4, 2] in
  (decDigits : List ℕ)[(50 % 6) - 1] = 5 := 
by
  sorry

end fiftieth_digit_of_decimal_representation_of_five_fourteenth_is_five_l94_94598


namespace exists_route_within_republic_l94_94920

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l94_94920


namespace isosceles_right_triangle_area_l94_94013

-- Define a structure for the properties of the isosceles right triangle.
structure IsoscelesRightTriangleOnHyperbola where
  A B C : ℝ × ℝ
  hyp : A.1 * A.2 = 1 ∧ B.1 * B.2 = 1 ∧ C.1 * C.2 = 1
  centroid_origin : A.1 + B.1 + C.1 = 0 ∧ A.2 + B.2 + C.2 = 0
  isosceles_right : B.1 * B.1 + B.2 * B.2 = C.1 * C.1 + C.2 * C.2

noncomputable def square_of_area (triangle : IsoscelesRightTriangleOnHyperbola) : ℝ :=
  4 * (triangle.B.1^2 + (1/triangle.B.1)^2) ^ 2

-- Statement of the theorem
theorem isosceles_right_triangle_area (triangle : IsoscelesRightTriangleOnHyperbola) :
  square_of_area triangle = 4 * (triangle.B.1^2 + (1/triangle.B.1)^2) ^ 2 :=
by
  sorry

end isosceles_right_triangle_area_l94_94013


namespace acres_used_for_corn_l94_94629

def total_land : ℕ := 1034
def beans_ratio : ℕ := 5
def wheat_ratio : ℕ := 2
def corn_ratio : ℕ := 4
def total_ratio_parts : ℕ := beans_ratio + wheat_ratio + corn_ratio

theorem acres_used_for_corn : 
  let total_land_used := total_land
  let beans := beans_ratio
  let wheat := wheat_ratio
  let corn := corn_ratio
  let total_parts := total_ratio_parts
  let acres_per_part := total_land / total_parts
  let corn_acres := acres_per_part * corn
  in corn_acres = 376 := 
by
  sorry

end acres_used_for_corn_l94_94629


namespace no_real_solution_log2_eq_2_l94_94139

theorem no_real_solution_log2_eq_2 :
  ∀ x : ℝ, ¬ (log 2 (x^2 - 5 * x + 14) = 2) :=
begin
  sorry
end

end no_real_solution_log2_eq_2_l94_94139


namespace rearrange_coins_in_circle_l94_94987

theorem rearrange_coins_in_circle (n : ℕ) (initial_configuration : Fin n → Fin n) :
  ∃ final_configuration : Fin n → Fin n, (∀ i : Fin n, final_configuration (⟨i.1 + 1, Nat.mod_lt (i.1+1) n⟩ : Fin n) 
    = (⟨final_configuration i.1 + 1, sorry⟩ : Fin n)) :=
sorry

end rearrange_coins_in_circle_l94_94987


namespace santa_can_give_each_child_gift_l94_94215

theorem santa_can_give_each_child_gift (n : ℕ) (x : Fin n → ℕ) 
  (hx_pos : ∀ i, x i > 0)
  (hx_sum : (∑ i, (1 : ℝ) / x i) ≤ 1) :
  ∃ (f : Fin n → Fin n), ∀ i, f i < n ∧ x i > 0 :=
by
  sorry

end santa_can_give_each_child_gift_l94_94215


namespace cindy_envelopes_left_l94_94702

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end cindy_envelopes_left_l94_94702


namespace dorothy_annual_earnings_correct_l94_94363

-- Define the conditions
def dorothyEarnings (X : ℝ) : Prop :=
  X - 0.18 * X = 49200

-- Define the amount Dorothy earns a year
def dorothyAnnualEarnings : ℝ := 60000

-- State the theorem
theorem dorothy_annual_earnings_correct : dorothyEarnings dorothyAnnualEarnings :=
by
-- The proof will be inserted here
sorry

end dorothy_annual_earnings_correct_l94_94363


namespace zero_count_between_decimal_and_first_non_zero_digit_l94_94284

theorem zero_count_between_decimal_and_first_non_zero_digit :
  let frac := (5 : ℚ) / 1600
  let dec_form := 3125 / 10^6
  (frac = dec_form) → 
  ∃ n, (frac * 10^6 = n) ∧ (3 = String.length (n.to_digits 10).takeWhile (λ c, c = '0')) :=
by
  intros frac dec_form h
  sorry

end zero_count_between_decimal_and_first_non_zero_digit_l94_94284


namespace inequality_solution_set_minimum_m2_4n2_l94_94432

-- Part 1
theorem inequality_solution_set (x : ℝ) : 
  (∀ x, abs (x - 2) ≥ 3 - abs (x - 1) → x ∈ (set.Iic 0) ∪ (set.Ici 3)) :=
by { sorry }

-- Part 2
theorem minimum_m2_4n2 (a : ℝ) (h₁ : ∀ x, abs (x - a) ≤ 1 ↔ 2 ≤ x ∧ x ≤ 4) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h₂ : m + 2 * n = a) :
  m^2 + 4 * n^2 = 9 / 2 :=
by { sorry }

end inequality_solution_set_minimum_m2_4n2_l94_94432


namespace volume_of_defined_region_l94_94783

noncomputable def volume_of_region (x y z : ℝ) : ℝ :=
if x + y ≤ 5 ∧ z ≤ 5 ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x ≤ 2 then 15 else 0

theorem volume_of_defined_region :
  ∀ (x y z : ℝ),
  (0 ≤ x) → (0 ≤ y) → (0 ≤ z) → (x ≤ 2) →
  (|x + y + z| + |x + y - z| ≤ 10) →
  volume_of_region x y z = 15 :=
sorry

end volume_of_defined_region_l94_94783


namespace parabola_c_value_l94_94329

theorem parabola_c_value (b c : ℝ)
  (h1 : 3 = 2^2 + b * 2 + c)
  (h2 : 6 = 5^2 + b * 5 + c) :
  c = -13 :=
by
  -- Proof would follow here
  sorry

end parabola_c_value_l94_94329


namespace wedding_cost_correct_l94_94177

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def john_guests : ℕ := 50
def wife_guest_increase : ℕ := john_guests * 60 / 100
def total_wedding_cost : ℕ := venue_cost + cost_per_guest * (john_guests + wife_guest_increase)

theorem wedding_cost_correct : total_wedding_cost = 50000 :=
by
  sorry

end wedding_cost_correct_l94_94177


namespace problem_equiv_proof_l94_94962

noncomputable def prob_b1_div_b2_div_b3 : ℚ :=
  let T := { d | ∃ (e1 : ℕ) (e2 : ℕ), d = 2^e1 * 3^e2 ∧ e1 ≤ 5 ∧ e2 ≤ 10 }
  let choices := (T.to_finset.card : ℚ) ^ 3
  let valid_pairs := (nat.ascents 3 5).card * (nat.ascents 3 10).card
  valid_pairs / choices

theorem problem_equiv_proof :
  prob_b1_div_b2_div_b3 = 77 / 1387 := by
    sorry

end problem_equiv_proof_l94_94962


namespace gcf_7fact_8fact_l94_94751

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l94_94751


namespace simplify_and_evaluate_at_3_l94_94558

noncomputable def expression (x : ℝ) : ℝ := 
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1))

theorem simplify_and_evaluate_at_3 : expression 3 = -5 := 
  sorry

end simplify_and_evaluate_at_3_l94_94558


namespace log_expression_simplify_l94_94351

theorem log_expression_simplify :
  log 14 - 2 * log (7 / 3) + log 7 - log 18 = 0 := 
sorry

end log_expression_simplify_l94_94351


namespace find_eccentricity_l94_94426

-- Ellipse definition and its properties
variable (E : Type*) -- Type for the ellipse

-- Points F1 and F2 are the foci of the ellipse
variables (F1 F2 R Q : E)
-- slope function
variable (slope : E → E → ℝ) 

-- Definitions of conditions from the problem
-- Condition 1: Line passing through F1 with slope 2 intersects at R and Q
axiom slope_F1_R : slope F1 R = 2
axiom slope_F1_Q : slope F1 Q = 2

-- Condition 2: Triangle P F1 F2 is a right triangle
axiom right_triangle : ∀ (P : E), angle F1 F2 P = π/2 ∨ angle F1 P F2 = π/2 ∨ angle P F1 F2 = π/2

-- Eccentricity of the ellipse
def eccentricity (e : ℝ) : Prop := e = real.sqrt 5 - 2 ∨ e = real.sqrt 5 / 3

-- Statement of the problem
theorem find_eccentricity {e : ℝ} (E : Ellipse E) (F1 F2 R Q : E) 
  (h1 : slope F1 R = 2) (h2 : slope F1 Q = 2)
  (h3 : ∀ (P : E), angle F1 F2 P = π/2 ∨ angle F1 P F2 = π/2 ∨ angle P F1 F2 = π/2)
  : eccentricity e := 
sorry

end find_eccentricity_l94_94426


namespace hyperbola_vertices_distance_l94_94055

theorem hyperbola_vertices_distance :
  let hyperbola_eq : ℝ → ℝ → Prop := λ x y, 4 * x^2 - 16 * x - y^2 + 6 * y - 11 = 0
  ∃ (a : ℝ), ∀ (h k : ℝ), 
    (h = 2 → k = 3 → a = √4.5 → 
      (distance (h - a, k) (h + a, k) = 2 * √4.5)) :=
by 
  sorry

end hyperbola_vertices_distance_l94_94055


namespace hand_mitts_cost_l94_94367

/-- Given the conditions and total expenditure, prove the cost of hand mitts --/
theorem hand_mitts_cost (M : ℝ) (h1 : ∀ (M : ℝ), M > 0) (h2 : ∀ (costs : ℝ), costs = 135) :
  M = 14 :=
by
-- Define costs
let apron_cost : ℝ := 16
let utensil_cost : ℝ := 10
let knife_cost : ℝ := 2 * utensil_cost
let total_cost_per_niece := M + apron_cost + utensil_cost + knife_cost
let total_cost := 3 * total_cost_per_niece
let discounted_total_cost := 0.75 * total_cost

-- Use discounted_total_cost to calculate the cost of hand mitts
have h := (0.75 * total_cost = 135),
calc
  0.75 * (3 * (M + 16 + 10 + 20)) = 135 : by sorry
  2.25 * (M + 46) = 135                  : by sorry
  M + 46 = 60                            : by sorry
  M = 14                                 : by sorry

end hand_mitts_cost_l94_94367


namespace distinct_prime_sets_count_l94_94562

theorem distinct_prime_sets_count :
  (∃ q r s : ℕ, 2206 = q * r * s ∧ prime q ∧ prime r ∧ prime s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  (∀ p1 p2 p3 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 + p2 + p3 = (q + r + s + 1)) →
  ((p1 = q ∧ p2 = r ∧ p3 = s) ∨ (p1 = q ∧ p2 = s ∧ p3 = r) ∨ (p1 = r ∧ p2 = q ∧ p3 = s) ∨ (p1 = r ∧ p2 = s ∧ p3 = q) ∨ (p1 = s ∧ p2 = q ∧ p3 = r) ∨ (p1 = s ∧ p2 = r ∧ p3 = q)) →
  1 := 
sorry

end distinct_prime_sets_count_l94_94562


namespace sin_alpha_plus_pi_over_4_l94_94799

theorem sin_alpha_plus_pi_over_4 
  (α β : ℝ)
  (h_alpha : (3 * Real.pi / 4) < α ∧ α < Real.pi)
  (h_beta : (3 * Real.pi / 4) < β ∧ β < Real.pi)
  (h1 : Real.sin (α + β) = -7/25)
  (h2 : Real.sin (β - Real.pi / 4) = 4/5) :
  Real.sin (α + Real.pi / 4) = -3/5 :=
begin
  sorry
end

end sin_alpha_plus_pi_over_4_l94_94799


namespace Rachel_wins_probability_l94_94551

-- Definitions: Position, movement rules, and optimal play conditions
structure Position where
  x : ℤ
  y : ℤ

def initial_Sarah : Position := ⟨0, 0⟩
def initial_Rachel : Position := ⟨6, 8⟩

def move_Sarah (p : Position) : set Position := 
  { p' | (p'.x = p.x + 1 ∧ p'.y = p.y) ∨ (p'.x = p.x ∧ p'.y = p.y + 1) }

def move_Rachel (p : Position) : set Position := 
  { p' | (p'.x = p.x - 1 ∧ p'.y = p.y) ∨ (p'.x = p.x ∧ p'.y = p.y - 1) }

def distance (p1 p2 : Position) : ℤ :=
  abs (p1.x - p2.x) + abs (p1.y - p2.y)

def optimal_play_probability (start_Sarah start_Rachel : Position) : ℝ :=
  if start_Sarah = ⟨0, 0⟩ ∧ start_Rachel = ⟨6, 8⟩ then 1 - (1/2)^6 else sorry

theorem Rachel_wins_probability : optimal_play_probability initial_Sarah initial_Rachel = 63/64 :=
sorry

end Rachel_wins_probability_l94_94551


namespace cube_root_x_plus_y_l94_94442

theorem cube_root_x_plus_y (x y : ℝ) (h : sqrt(x - 1) + (y + 2)^2 = 0) : real.cbrt (x + y) = -1 := by
  sorry

end cube_root_x_plus_y_l94_94442


namespace find_z_solutions_l94_94062

theorem find_z_solutions (r : ℚ) (z : ℤ) (h : 2^z + 2 = r^2) : 
  (r = 2 ∧ z = 1) ∨ (r = -2 ∧ z = 1) ∨ (r = 3/2 ∧ z = -2) ∨ (r = -3/2 ∧ z = -2) :=
sorry

end find_z_solutions_l94_94062


namespace area_of_polygon_l94_94352

-- Define the vertices as given in the conditions
def vertices : List (ℝ × ℝ) := [(1, -1), (4, 2), (6, 1), (3, 4), (2, 0)]

-- Define the Shoelace Theorem area calculation
def shoelace_area (vertices : List (ℝ × ℝ)) : ℝ :=
  let cyclicPairs := (vertices.zip vertices.tail).append [(vertices.last!, vertices.head!)]
  let sum := cyclicPairs.map (λ ((x1, y1), (x2, y2)) => (x1 * y2) - (y1 * x2)).sum
  (1 / 2) * abs sum

-- State the theorem
theorem area_of_polygon : shoelace_area vertices = 4.5 :=
by
  sorry

end area_of_polygon_l94_94352


namespace minimum_value_sincsc_cossec_l94_94777

theorem minimum_value_sincsc_cossec (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (\sin x + 1 / sin x + 1) ^ 2 + (\cos x + 1 / cos x + 1) ^ 2 = 10 :=
sorry

end minimum_value_sincsc_cossec_l94_94777


namespace gcf_7_8_fact_l94_94750

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l94_94750


namespace extreme_value_and_tangent_line_l94_94843

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2 - 3 * x

theorem extreme_value_and_tangent_line (a b : ℝ) (h1 : f a b 1 = 0) (h2 : f a b (-1) = 0) :
  (f 1 0 (-1) = 2) ∧ (f 1 0 1 = -2) ∧ (∀ x : ℝ, x = -2 → (9 * x - (x^3 - 3 * x) + 16 = 0)) :=
by
  sorry

end extreme_value_and_tangent_line_l94_94843


namespace set_of_all_possible_values_m_plus_n_l94_94192

theorem set_of_all_possible_values_m_plus_n : 
  ∃ m n : ℝ, 
    (∀ x ∈ set.Icc m n, (-x^2 + 4 * x) ∈ set.Icc (-5 : ℝ) 4) ∧ 
    (m + n) ∈ set.Icc 1 5 :=
sorry

end set_of_all_possible_values_m_plus_n_l94_94192


namespace relationship_exists_l94_94018

noncomputable def contingency_table : Prop := 
  let total_individuals := 124
  let females := 70
  let males := 54
  let females_tv := 43
  let females_sports := 27
  let males_tv := 21
  let males_sports := 33
  let chi_square_confidence := 0.975
  ∃ (relationship : bool), relationship = true

theorem relationship_exists : contingency_table :=
by
  let total_individuals := 124
  let females := 70
  let males := 54
  let females_tv := 43
  let females_sports := 27
  let males_tv := 21
  let males_sports := 33
  let chi_square_confidence := 0.975
  let relationship := true
  use relationship
  sorry

end relationship_exists_l94_94018


namespace ratio_circumcircle_radii_constant_minimize_radii_position_l94_94172

-- Conditions
variables {A B C D : Type} [Point A] [Point B] [Point C] [Point D]
variables {a b : ℝ} -- lengths of AC and BC
variables {φ : ℝ} -- angle ∠ACD
variables h0 : 0 < a
variables h1 : 0 < b

-- Radius definition for circumscribed circles
def R (b : ℝ) (φ : ℝ) := b / (2 * sin φ)
def R1 (a : ℝ) (φ : ℝ) := a / (2 * sin φ)

-- Theorem statements
theorem ratio_circumcircle_radii_constant :
  R b φ / R1 a φ = b / a := sorry

theorem minimize_radii_position :
  (exists (D : Type), is_perpendicular D (AC : b) (BC : a)) :=
    φ = π / 2 := sorry

end ratio_circumcircle_radii_constant_minimize_radii_position_l94_94172


namespace ap_digit_sum_repeat_l94_94545

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.toString.foldl (λ sum digit_char, sum + (digit_char.toNat - '0'.toNat)) 0

theorem ap_digit_sum_repeat (a d : ℕ) (h_a : 0 < a) (h_d : 0 < d) :
  ∃ n m : ℕ, n ≠ m ∧ digit_sum (a + (n-1)*d) = digit_sum (a + (m-1)*d) :=
sorry

end ap_digit_sum_repeat_l94_94545


namespace valid_sequences_count_l94_94725

theorem valid_sequences_count :
  (count (λ (seq : Fin 5 → Int), 
    (∀ i, seq i ≤ 1) ∧ -- Condition 1: all elements are <= 1
    (∀ j, (0 ≤ ∑ k in Finset.range j.succ, seq k))) -- Condition 2: all partial sums are non-negative
  = 132 := 
sorry

end valid_sequences_count_l94_94725


namespace circle_equation_l94_94106

theorem circle_equation 
  (C : ℝ × ℝ → Prop)
  (h1 : C (1, 0))
  (h2 : C (3, 0))
  (h3 : ∃ c : ℝ × ℝ, ∀ x : ℝ × ℝ, C x ↔ (x.1 - c.1)^2 + (x.2 - c.2)^2 = 4 ∧ c.1 = 2 ∧ c.2 = 2 ∨ c.2 = -2) :
  ∃ k : ℝ, k = ∓ √3 ∧ C (x - 2)^2 + (y - k)^2 = 4 := sorry

end circle_equation_l94_94106


namespace surface_area_inscribed_sphere_l94_94021

noncomputable def tetrahedron_edge_length (a : ℝ) :=
  ∀ (T : Type), is_tetrahedron T → ∀ (e : edge T), length e = a

noncomputable def surface_area_of_inscribed_sphere (a : ℝ) : ℝ :=
  4 * Real.pi * (a / (2 * Real.sqrt 6))^2

theorem surface_area_inscribed_sphere (a : ℝ) :
  tetrahedron_edge_length a → 
  surface_area_of_inscribed_sphere a = Real.pi * a^2 / 6 :=
by
  sorry

end surface_area_inscribed_sphere_l94_94021


namespace dartboard_partitions_l94_94734

theorem dartboard_partitions (darts boards : ℕ) (h : darts = 6) (b : boards = 5) :
  (list.partitions darts).filter (λ l, l.length ≤ boards).length = 11 :=
by sorry

end dartboard_partitions_l94_94734


namespace sum_odd_indexed_terms_l94_94943

-- Define the arithmetic sequence and its properties.
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first 100 terms.
def sum_first_100_terms (a : ℕ → ℝ) :=
  ∑ n in finset.range 100, a n

-- The problem statement to prove.
theorem sum_odd_indexed_terms (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_d : d = 1 / 2)
  (h_sum_100 : sum_first_100_terms a = 45) :
  ∑ k in finset.range 50, a (2 * k + 1) = 10 :=
sorry

end sum_odd_indexed_terms_l94_94943


namespace equilateral_triangle_area_l94_94489

theorem equilateral_triangle_area (ABC : Type) (A B C P : ABC)
  [equilateral_triangle ABC A B C] (hAP : dist A P = 10) (hBP : dist B P = 8)
  (hCP : dist C P = 6) :
  area ABC = 36 + 25 * sqrt 3 :=
sorry

end equilateral_triangle_area_l94_94489


namespace suitable_material_for_observing_meiosis_l94_94267

-- Definitions of the conditions
def fertilized_eggs_of_ascaris_undergo_mitosis : Prop :=
  ∀ eggs : AscarisEggs, undergoesMitosis eggs

def testes_of_mice_undergo_meiosis : Prop :=
  ∀ testes : MouseTestes, undergoesVigorousMeiosis testes

def sperm_of_locusts_are_post_meiosis : Prop :=
  ∀ sperm : LocustSperm, postMeiosis sperm

def blood_of_chickens_do_not_undergo_meiosis : Prop :=
  ∀ blood : ChickenBlood, ¬undergoesMeiosis blood

-- The main theorem
theorem suitable_material_for_observing_meiosis :
  fertilized_eggs_of_ascaris_undergo_mitosis →
  testes_of_mice_undergo_meiosis →
  sperm_of_locusts_are_post_meiosis →
  blood_of_chickens_do_not_undergo_meiosis →
  suitableMaterialForMeiosis = MouseTestes :=
sorry

end suitable_material_for_observing_meiosis_l94_94267


namespace greatest_product_from_sum_2004_l94_94601

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l94_94601


namespace clock_angle_at_330_l94_94613

/--
At 3:00, the hour hand is at 90 degrees from the 12 o'clock position.
The minute hand at 3:30 is at 180 degrees from the 12 o'clock position.
The hour hand at 3:30 has moved an additional 15 degrees (0.5 degrees per minute).
Prove that the smaller angle formed by the hour and minute hands of a clock at 3:30 is 75.0 degrees.
-/
theorem clock_angle_at_330 : 
  let hour_pos_at_3 := 90
  let min_pos_at_330 := 180
  let hour_additional := 15
  (min_pos_at_330 - (hour_pos_at_3 + hour_additional) = 75)
  :=
  by
  sorry

end clock_angle_at_330_l94_94613


namespace eccentricity_of_hyperbola_l94_94096

-- This ensures we are dealing with non-negative reals
noncomputable def sqrt (x : ℝ) : ℝ := x ^ (1 / 2)

variables (a b c : ℝ) (F1 F2 : ℝ × ℝ)
variables (M : ℝ × ℝ)

-- Given conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def a_pos : Prop := a > 0
def b_pos : Prop := b > 0
def asymptote (x y : ℝ) : Prop := y = (b/a) * x
def f2_coordinates : F2 = (c, 0)
def symmetric_across_asymptote : Prop :=
  M.1 = -(b^2 / c) ∧ M.2 = (a * b / c) ∧ M ∈ {p | hyperbola p.1 p.2}

-- Eccentricity
def eccentricity (e : ℝ) : Prop :=
  e = c / a

-- Prove that the eccentricity e is equal to sqrt(5)
theorem eccentricity_of_hyperbola : (∃ e : ℝ, eccentricity e ∧ e = sqrt 5) :=
sorry

end eccentricity_of_hyperbola_l94_94096


namespace triangle_area_ratio_trapezoid_area_l94_94948

variables (A B C D F : Type*)
variables [trapezoid ABCD]
variables (AB CD : ℝ) (l_AB l_CD : segment A B) (l_F_extends_A l_F_extends_B : point_extends F AB)

def area_ratio := (area (triangle F A B)) / (area (trapezoid A B C D))

theorem triangle_area_ratio_trapezoid_area
  (hAB : l_AB.length = 5) 
  (hCD : l_CD.length = 20)
  (h_extend_A : extends F l_AB)
  (h_extend_B : extends F l_CD) :
  area_ratio = 1 / 15 := 
sorry

end triangle_area_ratio_trapezoid_area_l94_94948


namespace two_digit_squares_divisible_by_4_l94_94864

theorem two_digit_squares_divisible_by_4 : 
  ∃ n : ℕ, n = 3 ∧ (n = (count (λ x : ℕ, 10 ≤ x ∧ x ≤ 99 ∧ (∃ k : ℕ, x = k * k ∧ x % 4 = 0)))) :=
by 
  sorry

end two_digit_squares_divisible_by_4_l94_94864


namespace factorial_fraction_is_integer_l94_94547

theorem factorial_fraction_is_integer (m n : ℕ) : ∃ k : ℤ, ((2 * m)!.toRational * (2 * n)!.toRational) / (m!.toRational * n!.toRational * (m + n)!.toRational) = k :=
by {
  sorry
}

end factorial_fraction_is_integer_l94_94547


namespace y_coord_third_vertex_l94_94009

theorem y_coord_third_vertex (a b : ℝ) (h1 : a = (2:ℝ)) (h2 : b = (10:ℝ)) : 
  (∃ y : ℝ, y = 3 + 4 * Real.sqrt 3) :=
by
  have h3 : a < b := by linarith [h1, h2]
  use 3 + 4 * Real.sqrt 3
  split
  assumption
  sorry

end y_coord_third_vertex_l94_94009


namespace cannot_achieve_equal_chips_l94_94986

theorem cannot_achieve_equal_chips (initial_distribution : ℕ → ℕ → ℕ) :
  (∃ cells, |cells| = 100 ∧ cells = { (i,j) | 0 ≤ i < 10 ∧ 0 ≤ j < 10 }) →
  (∀ i j, initial_distribution i j ∈ ℕ) →
  (∑ i j in (finset.range 10).product (finset.range 10), initial_distribution i j = 400) →
  ¬(∃ n, ∀ i j, initial_distribution i j = n) :=
by
  sorry

end cannot_achieve_equal_chips_l94_94986


namespace phi_periodic_and_bounded_l94_94971

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_iter (n : ℕ) : ℝ → ℝ := sorry

def non_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y
def phi (n : ℕ) (x : ℝ) : ℝ := f_iter n x - x

axiom f_non_decreasing : non_decreasing f
axiom f_shift : ∀ x, f (x + 1) = f x + 1
axiom f_iter_def : ∀ n x, f_iter (n + 1) x = f (f_iter n x)

theorem phi_periodic_and_bounded {n : ℕ} {x y : ℝ} :
  |phi n x - phi n y| < 1 := sorry

end phi_periodic_and_bounded_l94_94971


namespace solution_set_of_exponential_inequality_l94_94579

theorem solution_set_of_exponential_inequality :
  {x : ℝ | 2^(x^2 - x) < 4} = {x : ℝ | -1 < x ∧ x < 2} :=
by {
  sorry
}

end solution_set_of_exponential_inequality_l94_94579


namespace even_rows_pascal_triangle_l94_94042

def is_even (n : ℕ) : Prop := n % 2 = 0

def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

theorem even_rows_pascal_triangle : ∃ (n : ℕ), n = 4 ∧
  ∀ k : ℕ, k < 31 →
  (∀ m : ℕ, 0 < m ∧ m < k → is_even (pascal k m)) :=
by
  sorry

end even_rows_pascal_triangle_l94_94042


namespace ellipse_range_x_plus_y_l94_94097

/-- The problem conditions:
Given any point P(x, y) on the ellipse x^2 / 144 + y^2 / 25 = 1,
prove that the range of values for x + y is [-13, 13].
-/
theorem ellipse_range_x_plus_y (x y : ℝ) (h : (x^2 / 144) + (y^2 / 25) = 1) : 
  -13 ≤ x + y ∧ x + y ≤ 13 := sorry

end ellipse_range_x_plus_y_l94_94097


namespace matrix_sum_correct_l94_94020

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 3], ![-2, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-1, 5], ![8, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![6, -2]]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l94_94020


namespace trajectory_of_midpoint_l94_94091

theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) :
  (y₀ = 2 * x₀ ^ 2 + 1) ∧ (x = (x₀ + 0) / 2) ∧ (y = (y₀ + 1) / 2) →
  y = 4 * x ^ 2 + 1 :=
by sorry

end trajectory_of_midpoint_l94_94091


namespace repeating_block_length_of_11_over_13_l94_94609

-- Given definitions and conditions
def repeatingBlockLength (n d : ℕ) : ℕ :=
  let rec find_period (remainder modulo_count : ℕ) (seen_rem : Finset ℕ) : ℕ :=
    if seen_rem.contains remainder then
      modulo_count
    else
      find_period ((remainder * 10) % d) (modulo_count + 1) (seen_rem.insert remainder)
  find_period (n % d) 0 ∅

-- Problem statement
theorem repeating_block_length_of_11_over_13 : repeatingBlockLength 11 13 = 6 := by
  sorry

end repeating_block_length_of_11_over_13_l94_94609


namespace robert_books_l94_94550

/-- Given that Robert reads at a speed of 75 pages per hour, books have 300 pages, and Robert reads for 9 hours,
    he can read 2 complete 300-page books in that time. -/
theorem robert_books (reading_speed : ℤ) (pages_per_book : ℤ) (hours_available : ℤ) 
(h1 : reading_speed = 75) 
(h2 : pages_per_book = 300) 
(h3 : hours_available = 9) : 
  hours_available / (pages_per_book / reading_speed) = 2 := 
by {
  -- adding placeholder for proof
  sorry
}

end robert_books_l94_94550


namespace gcf_7fact_8fact_l94_94757

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l94_94757


namespace problem_solution_l94_94238

variable (X : Type) [Distrib X] (m n : ℝ) (P : X → ℝ) [HasMem X ℝ]
variable (h1 : P 0 = m)
variable (h2 : P 1 = 1 / 2)
variable (h3 : P 2 = n)
variable (h_sum : m + 1 / 2 + n = 1)
variable (E : (X → ℝ) → ℝ)
variable (h_E : E id = 1)

theorem problem_solution :
  n = 1 / 4 ∧ D X = 1 / 2 ∧ E (λ x, 2 * x + 1) = 3 :=
by
  -- Proof not required
  sorry

end problem_solution_l94_94238


namespace airline_route_same_republic_exists_l94_94930

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l94_94930


namespace problem1_problem2_l94_94116

-- Problem 1: If \( f(x) \) is defined as:
-- \[
-- f(x)= 
-- \begin{cases} 
-- x^2 + x, & \text{for } -2 \leq x \leq 0 \\
-- \frac{1}{x}, & \text{for } 0 < x \leq 3 
-- \end{cases}
-- \]
-- then the range of \( f(x) \) is \( \left[-\frac{1}{4}, +\infty \right) \).

theorem problem1 {x : ℝ} : 
  (f : ℝ → ℝ)
  (h1 : ∀ x, -2 ≤ x → x ≤ 0 → f(x) = x^2 + x)
  (h2 : ∀ x, 0 < x → x ≤ 3 → f(x) = 1 / x) 
  → (∀ y, ∃ x, f(x) = y ↔ y ∈ set.Ici (-1/4))
:= sorry

-- Problem 2: If \( f(x) \) is defined as:
-- \[
-- f(x)= 
-- \begin{cases} 
-- x^2 + x, & \text{for } -2 \leq x \leq c \\
-- \frac{1}{x}, & \text{for } c < x \leq 3 
-- \end{cases}
-- \]
-- and the range of \( f(x) \) is \( \left[-\frac{1}{4}, 2 \right] \), then \( c \in \left[\frac{1}{2}, 1 \right] \).

theorem problem2 {c : ℝ} :
  (f : ℝ → ℝ)
  (h1 : ∀ x, -2 ≤ x → x ≤ c → f(x) = x^2 + x)
  (h2 : ∀ x, c < x → x ≤ 3 → f(x) = 1 / x) 
  (hrange : ∀ y, ∃ x, f(x) = y ↔ y ∈ set.Icc (-1/4) 2)
  → c ∈ set.Icc (1/2:ℝ) 1
:= sorry

end problem1_problem2_l94_94116


namespace complex_in_fourth_quadrant_l94_94946

noncomputable def Z : ℂ := (2 / (3 - I)) + (I ^ 3)

theorem complex_in_fourth_quadrant : (0 < Z.re) ∧ (Z.im < 0) :=
by 
  -- Steps given in the solution
  have Z_def : Z = (2 / (3 - I)) + (I ^ 3) by sorry,
  have simpZ : Z = (3 / 5) - (4 / 5) * I by sorry,
  sorry -- This would be the proof, which we are skipping

end complex_in_fourth_quadrant_l94_94946


namespace probability_face_heart_ace_l94_94586

-- Defining the probability of drawing sequential cards in a specific order from a deck
theorem probability_face_heart_ace :
  let face_card_prob := 12 / 52 in
  let heart_after_face_prob := 1 / 4 in
  let ace_after_face_and_heart_prob := 2 / 25 in
  face_card_prob * heart_after_face_prob * ace_after_face_and_heart_prob = 3 / 650 :=
by
  let face_card_prob := 12 / 52
  let heart_after_face_prob := 1 / 4
  let ace_after_face_and_heart_prob := 2 / 25
  calc 
    face_card_prob * heart_after_face_prob * ace_after_face_and_heart_prob
      = (12 / 52) * (1 / 4) * (2 / 25) : by sorry
      ... = 3 / 650 : by sorry

end probability_face_heart_ace_l94_94586


namespace greatest_product_two_integers_sum_2004_l94_94607

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l94_94607


namespace odd_function_properties_l94_94149

def f : ℝ → ℝ := sorry

theorem odd_function_properties 
  (H1 : ∀ x, f (-x) = -f x) -- f is odd
  (H2 : ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → f x ≤ f y) -- f is increasing on [1, 3]
  (H3 : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≥ 7) -- f has a minimum value of 7 on [1, 3]
  : (∀ x y, -3 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) -- f is increasing on [-3, -1]
    ∧ (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) -- f has a maximum value of -7 on [-3, -1]
:= sorry

end odd_function_properties_l94_94149


namespace incorrect_option_D_l94_94660

theorem incorrect_option_D (μ σ : ℝ) (h : σ > 0) : 
  (prob_less_than_10_3 : probability (x : ℝ) in normal (μ, σ^2) x < 10.3) ≠ 
  (prob_between_9_9_and_10_2 : probability (x : ℝ) in normal (μ, σ^2) (9.9 < x ∧ x < 10.2)) :=
by
  -- necessary parameter settings based on given problem
  let normal_dist := normal (10, σ^2)
  have h1 : prob_less_than_10_3 = probability (x : ℝ) in normal_dist x < 10.3 := sorry
  have h2 : prob_between_9_9_and_10_2 = probability (x : ℝ) in normal_dist (9.9 < x ∧ x < 10.2) := sorry
  sorry

end incorrect_option_D_l94_94660


namespace probability_sum_equals_15_l94_94046

-- Conditions and definitions
def erika_age : ℕ := 15

def coin_faces : set ℕ := {5, 15}

def die_faces : set ℕ := {1, 2, 3, 4, 5, 6}

-- Helper function to compute the probability of a specific sum
def rolling_sum_probability (target_sum : ℕ) : ℚ := 
  let coin_pairs := { (a, b) | a ∈ coin_faces ∧ b ∈ coin_faces }
  let die_face_values := die_faces
  let favorable_outcomes := ∑ c in coin_faces, ∑ d in coin_faces, 
    if c + d + die_face_values = target_sum then 1 else 0
  let total_outcomes := card coin_pairs * card die_faces
  favorable_outcomes / total_outcomes

theorem probability_sum_equals_15 : rolling_sum_probability 15 = 1/8 :=
sorry

end probability_sum_equals_15_l94_94046


namespace distance_from_O_is_450_l94_94365

noncomputable def find_distance_d (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ) : ℝ :=
    if h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
           dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
           -- condition of 120 degree dihedral angle translates to specific geometric constraints
           true -- placeholder for the actual geometrical configuration that proves the problem
    then 450
    else 0 -- default or indication of inconsistency in conditions

-- Assuming all conditions hold true
theorem distance_from_O_is_450 (A B C P Q O : Type) 
  (side_length : ℝ) (PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ : ℝ)
  (h : PA = PB ∧ PB = PC ∧ QA = QB ∧ QB = QC ∧ side_length = 600 ∧
       dist_OA = dist_OB ∧ dist_OB = dist_OC ∧ dist_OC = dist_OP ∧ dist_OP = dist_OQ ∧
       -- adding condition of 120 degree dihedral angle
       true) -- true is a placeholder, the required proof to be filled in
  : find_distance_d A B C P Q O side_length PA PB PC QA QB QC dist_OA dist_OB dist_OC dist_OP dist_OQ = 450 :=
by
  -- proof goes here
  sorry

end distance_from_O_is_450_l94_94365


namespace length_CD_correct_l94_94578

noncomputable def length_CD : ℝ :=
  let radius := 4 in
  let volume := 400 * Real.pi in
  let cylinder_volume (L : ℝ) := 16 * Real.pi * L in 
  let hemisphere_volume := (2 * (128 / 3) * Real.pi) in
  let total_volume (L : ℝ) := cylinder_volume L + hemisphere_volume in
  Classical.some (Exists.intro 20 (by
    have h : total_volume 20 = volume :=
      by calc
      total_volume 20 = 16 * Real.pi * 20 + hemisphere_volume : by rfl
      ... = 16 * Real.pi * 20 + (256 / 3) * Real.pi : by rfl
      ... = (320 * Real.pi + (256 / 3) * Real.pi) : by rfl
      ... = ((960 / 3) * Real.pi + (256 / 3) * Real.pi) : by rw [(mul_div_assoc _ _ 3), (mul_div_assoc _ _ 3)]
      ... = (1216 / 3) * Real.pi : by rw [add_div]
      ... = 400 * Real.pi : by norm_num
    exact h))

theorem length_CD_correct : length_CD = 20 := sorry

end length_CD_correct_l94_94578


namespace cannot_be_zero_l94_94240

noncomputable def P (x : ℝ) (a b c d e : ℝ) := x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e

theorem cannot_be_zero (a b c d e : ℝ) (p q r s : ℝ) :
  e = 0 ∧ c = 0 ∧ (∀ x, P x a b c d e = x * (x - p) * (x - q) * (x - r) * (x - s)) ∧ 
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  d ≠ 0 := 
by {
  sorry
}

end cannot_be_zero_l94_94240


namespace quadratic_solution_property_l94_94969

theorem quadratic_solution_property :
  (∃ p q : ℝ, 3 * p^2 + 7 * p - 6 = 0 ∧ 3 * q^2 + 7 * q - 6 = 0 ∧ (p - 2) * (q - 2) = 6) :=
by
  sorry

end quadratic_solution_property_l94_94969


namespace clinton_shoes_count_l94_94709

theorem clinton_shoes_count : 
  let hats := 5
  let belts := hats + 2
  let shoes := 2 * belts
  shoes = 14 := 
by
  -- Define the number of hats
  let hats := 5
  -- Define the number of belts
  let belts := hats + 2
  -- Define the number of shoes
  let shoes := 2 * belts
  -- Assert that the number of shoes is 14
  show shoes = 14 from sorry

end clinton_shoes_count_l94_94709


namespace fish_in_exceptional_table_l94_94566

theorem fish_in_exceptional_table
  (tables : ℕ)
  (fish_per_table : ℕ)
  (total_fish : ℕ)
  (standard_tables : ℕ)
  (extra_fish : ℕ)
  (exceptional_tables : ℕ)
  (extra_fish_count : ℕ)
  : tables = 32 → fish_per_table = 2 → total_fish = 65 → standard_tables = 31 → exceptional_tables = 1 → extra_fish = 1 → total_fish = (standard_tables * fish_per_table + exceptional_tables * (fish_per_table + extra_fish_count)) → extra_fish_count = 1 → (fish_per_table + extra_fish_count) = 3 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  sorry

end fish_in_exceptional_table_l94_94566


namespace min_value_three_l94_94523

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (1 / ((1 - x) * (1 - y) * (1 - z))) +
  (1 / ((1 + x) * (1 + y) * (1 + z))) +
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)))

theorem min_value_three (x y z : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  min_value_expression x y z = 3 :=
by
  sorry

end min_value_three_l94_94523


namespace min_distance_PQ_l94_94526

theorem min_distance_PQ :
  ∃ P Q : ℝ × ℝ, 
     (P.1 = P.2 ∧ Q.2 = log Q.1 ∧ |dist P Q| = (Real.sqrt 2) / 2) :=
begin
  sorry -- Proof to be provided
end

end min_distance_PQ_l94_94526


namespace smallest_positive_period_max_value_and_set_values_of_alpha_l94_94517

namespace ProofProblem

def a (x : ℝ) : ℝ × ℝ := (3, -Real.sin (2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sqrt 3)
def f (x : ℝ) := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- (Ⅰ) Prove that the smallest positive period of f(x) is π.
theorem smallest_positive_period : ∀ x, f (x + π) = f x :=
by
  sorry

-- (Ⅱ) Prove that the maximum value of f(x) is 2√3 and set of x when the maximum value is achieved.
theorem max_value_and_set : 
  ∃ k : ℤ, ∀ x, (2 * x + π/6) = 2 * k * π → f x = 2 * Real.sqrt 3 :=
by
  sorry

-- (Ⅲ) Prove that the values of α satisfying f(α) = -√3 and 0 < α < π are {π/4, 7π/12}.
theorem values_of_alpha : 
  ∀ α, f α = -Real.sqrt 3 ∧ 0 < α ∧ α < π → α = π/4 ∨ α = 7 * π/12 :=
by
  sorry

end ProofProblem

end smallest_positive_period_max_value_and_set_values_of_alpha_l94_94517


namespace integrate_even_function_l94_94966

open Real

theorem integrate_even_function (f : ℝ → ℝ) (a : ℝ) (h_cont : Continuous f)
  (h_even : ∀ x : ℝ, f(-x) = f(x)) :
  ∫ x in -a..a, f x = 2 * ∫ x in -a..0, f x :=
sorry

end integrate_even_function_l94_94966


namespace max_area_of_quadrilateral_ABCD_l94_94947

-- Define the quadrilateral ABCD with specified properties
variables {A B C D G L M : Type}

-- Condition: Length of side BC is 2
axiom length_BC : Real := 2

-- Condition: Length of side CD is 6
axiom length_CD : Real := 6

-- Condition: Points of intersection of the medians form an equilateral triangle GLM
axiom GLM_is_equilateral : equilateral_triangle G L M

-- Prove that the maximum area of quadrilateral ABCD is 29.32
theorem max_area_of_quadrilateral_ABCD :
  maximum_area_quadrilateral A B C D length_BC length_CD GLM_is_equilateral = 29.32 :=
begin
  sorry
end

end max_area_of_quadrilateral_ABCD_l94_94947


namespace triangle_AF_AT_ratio_l94_94169

theorem triangle_AF_AT_ratio (A B C D E F T : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace T]
  (AD DB AE EC : ℝ)
  (h_AD : AD = 1) (h_DB : DB = 3)
  (h_AE : AE = 2) (h_EC : EC = 4)
  (h_bisector : is_angle_bisector A T B C)
  (h_intersect : line_intersects DE AT F) :
  \(\frac{AF}{AT} = \frac{5}{18}) :=
sorry

end triangle_AF_AT_ratio_l94_94169


namespace solution_l94_94405

noncomputable def A := ℝ × ℝ
noncomputable def B := ℝ × ℝ
variables (a : ℝ × ℝ) 

/-- The function f mapping from set A to set B. --/
def f (x : A) : B := 
  (x.1 - 2 * (x.1 * a.1 + x.2 * a.2) * a.1, x.2 - 2 * (x.1 * a.1 + x.2 * a.2) * a.2)

/-- The condition: f(x) · f(y) = x · y for any vectors x, y in A. --/
def condition : Prop :=
  ∀ (x y : A), let fx := f a x, fy := f a y in
    fx.1 * fy.1 + fx.2 * fy.2 = x.1 * y.1 + x.2 * y.2

/-- The proof goal is to show that (a.1, a.2) = (-1/2, sqrt(3)/2) is a valid solution. --/
theorem solution : condition a →
  a = (-1/2, real.sqrt(3)/2) :=
sorry

end solution_l94_94405


namespace X_is_degenerate_l94_94189

-- Definitions for the sequence of random variables and their convergence
variable {Ω : Type*} -- the sample space
variable (X_n : ℕ → Ω → ℝ) -- sequence of random variables
variable (X : Ω → ℝ) -- random variable

-- Assumptions
variable (pairwise_independent : ∀ i j, i ≠ j → independent (X_n i) (X_n j))
variable (converges_in_probability : ∀ ε > 0, ∀ δ > 0, ∃ N, ∀ n ≥ N, (λ ω, abs (X_n n ω - X ω)) < δ)

-- Theorem Statement
theorem X_is_degenerate : ∀ x, (P (λ ω, X ω = x) = 1) :=
sorry

end X_is_degenerate_l94_94189


namespace smallest_n_1987_zeros_l94_94060

def h (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 + n / 15625 + n / 78125 + n / 390625 + n / 1953125 + n / 9765625

theorem smallest_n_1987_zeros :
  ∃ n : ℕ, h(n) = 1987 ∧ ∀ m : ℕ, m < n → h(m) ≠ 1987 :=
  by
  existsi 7960
  split
  · sorry
  · intro m hm
    sorry

end smallest_n_1987_zeros_l94_94060


namespace max_product_of_two_integers_with_sum_2004_l94_94603

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l94_94603


namespace yi_successful_shots_l94_94538

-- Defining the basic conditions
variables {x y : ℕ} -- Number of successful shots made by Jia and Yi respectively

-- Each hit gains 20 points and each miss deducts 12 points.
-- Both person A (Jia) and person B (Yi) made 10 shots each.
def total_shots (x y : ℕ) : Prop := 
  (20 * x - 12 * (10 - x)) + (20 * y - 12 * (10 - y)) = 208 ∧ x + y = 14 ∧ x - y = 2

theorem yi_successful_shots (x y : ℕ) (h : total_shots x y) : y = 6 := 
  by sorry

end yi_successful_shots_l94_94538


namespace exists_airline_route_within_same_republic_l94_94904

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l94_94904


namespace number_of_employees_l94_94231

-- Defining the conditions
variables (n : ℕ) -- number of employees excluding the manager
variables (employee_avg_salary : ℕ) -- average salary excluding the manager
variables (increased_avg_salary : ℕ) -- increased average salary including the manager
variables (manager_salary : ℕ) -- manager's salary

-- Setting the conditions as variables
def conditions :=
  employee_avg_salary = 2000 ∧
  increased_avg_salary = 2200 ∧
  manager_salary = 5800

-- The statement to prove
theorem number_of_employees (h : conditions) : n = 18 :=
by sorry

end number_of_employees_l94_94231


namespace zero_count_between_decimal_and_first_non_zero_digit_l94_94282

theorem zero_count_between_decimal_and_first_non_zero_digit :
  let frac := (5 : ℚ) / 1600
  let dec_form := 3125 / 10^6
  (frac = dec_form) → 
  ∃ n, (frac * 10^6 = n) ∧ (3 = String.length (n.to_digits 10).takeWhile (λ c, c = '0')) :=
by
  intros frac dec_form h
  sorry

end zero_count_between_decimal_and_first_non_zero_digit_l94_94282


namespace negation_of_proposition_l94_94245

namespace NegationProp

theorem negation_of_proposition :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - x < 0) ↔
  (∃ x0 : ℝ, 0 < x0 ∧ x0 < 1 ∧ x0^2 - x0 ≥ 0) := by sorry

end NegationProp

end negation_of_proposition_l94_94245


namespace geom_seq_decreasing_l94_94121

theorem geom_seq_decreasing :
  (∀ n : ℕ, (4 : ℝ) * 3^(1 - (n + 1) : ℤ) < (4 : ℝ) * 3^(1 - n : ℤ)) :=
sorry

end geom_seq_decreasing_l94_94121


namespace math_problem_proof_l94_94168

-- Define the parameters for circle C1
def parametric_eq_C1 (phi : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos phi, 2 * Real.sin phi)

-- Define the parameters for circle C2
def parametric_eq_C2 (phi : ℝ) : ℝ × ℝ := (Real.cos phi, 1 + Real.sin phi)

-- Define the polar equations of the circles
def polar_eq_C1 (theta : ℝ) : ℝ := 4 * Real.cos theta
def polar_eq_C2 (theta : ℝ) : ℝ := 2 * Real.sin theta

-- Main theorem to prove the polar equations and the maximum value of |OP| * |OQ|
theorem math_problem_proof (alpha : ℝ) :
  (∀ phi, let ⟨x, y⟩ := parametric_eq_C1 phi in x^2 + y^2 - 4*x = 0 → polar_eq_C1 alpha = 4 * Real.cos alpha) ∧
  (∀ phi, let ⟨x, y⟩ := parametric_eq_C2 phi in x^2 + y^2 - 2*y = 0 → polar_eq_C2 alpha = 2 * Real.sin alpha) ∧
  (Real.sqrt (8 + 8 * Real.cos alpha) * Real.sqrt (2 + 2 * Real.sin alpha) ≤ 4 + 2 * Real.sqrt 2) :=
by
  sorry

end math_problem_proof_l94_94168


namespace sinusoidal_extrema_l94_94776

noncomputable def max_value : ℝ := 2
noncomputable def min_value : ℝ := -2

def max_set (k : ℤ) : ℝ := (2 * k * Real.pi) / 3 + Real.pi / 18
def min_set (k : ℤ) : ℝ := (2 * k * Real.pi) / 3 - 5 * Real.pi / 18

theorem sinusoidal_extrema :
  ∀ x : ℝ, 
    (∃ k : ℤ, x = max_set k) ↔ 2 * Real.sin (3 * x + Real.pi / 3) = max_value ∧
    (∃ k : ℤ, x = min_set k) ↔ 2 * Real.sin (3 * x + Real.pi / 3) = min_value :=
by sorry

end sinusoidal_extrema_l94_94776


namespace rice_and_grain_separation_l94_94230

theorem rice_and_grain_separation (total_weight : ℕ) (sample_size : ℕ) (non_rice_sample : ℕ) (non_rice_in_batch : ℕ) :
  total_weight = 1524 →
  sample_size = 254 →
  non_rice_sample = 28 →
  non_rice_in_batch = total_weight * non_rice_sample / sample_size →
  non_rice_in_batch = 168 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end rice_and_grain_separation_l94_94230


namespace max_boxes_l94_94268

theorem max_boxes (budget : ℕ) (p1 p2 : ℕ) : budget = 105 → p1 = 5 → p2 = 7 → 
  ∃ (x y : ℕ), 5 * x + 7 * y = budget ∧ (∀ z w : ℕ, 5 * z + 7 * w = budget → z + w ≤ x + y) → 
  x + y = 19 :=
by {
  intros,
  -- The proof part which is skipped for this task
  sorry
}

end max_boxes_l94_94268


namespace triangle_is_isosceles_given_condition_l94_94811

noncomputable def is_isosceles_triangle {A B C D : EuclideanPlane.Point} 
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (cond : (A - B) ⬝ ((2 : ℝ) • D - B - C) = 0) : Prop :=
(triangle A B C).is_isosceles

theorem triangle_is_isosceles_given_condition 
  {A B C D : EuclideanPlane.Point}
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (cond : (A - B) ⬝ ((2 : ℝ) • D - B - C) = 0) : 
  (triangle A B C).is_isosceles :=
sorry

end triangle_is_isosceles_given_condition_l94_94811


namespace union_A_B_intersection_complement_A_B_range_of_a_l94_94403

noncomputable def A := {x : ℝ | 2^x > 1}
noncomputable def B := {x : ℝ | -1 < x ∧ x < 1}

theorem union_A_B : A ∪ B = {x : ℝ | x > -1} := 
by sorry

theorem intersection_complement_A_B : (set.univ \ A) ∩ B = {x : ℝ | -1 < x ∧ x ≤ 0} :=
by sorry

noncomputable def C (a : ℝ) := {x : ℝ | x < a}

theorem range_of_a (a : ℝ) : (B ∪ C a = C a) ↔ 1 ≤ a :=
by sorry

end union_A_B_intersection_complement_A_B_range_of_a_l94_94403


namespace no_zero_terms_l94_94786

noncomputable def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem no_zero_terms {a : ℕ → ℝ} {d : ℝ} (h_arith_prog : arithmetic_progression a d)
  (h_non_const : d ≠ 0)
  (h_exists_n : ∃ n > 1, a n + a (n + 1) = ∑ i in finset.range (3 * n - 1), a (i + 1)) :
  ∀ n, a n ≠ 0 :=
  sorry

end no_zero_terms_l94_94786


namespace line_passing_through_intersection_and_midpoint_l94_94853

-- Define the two points A and B
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (4, 3)

-- Define the midpoint function
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the two lines
def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 2 * y - 1 = 0

-- Prove that the line passing through the intersection of line1 and line2
-- and the midpoint of segment AB is 7x - 4y + 1 = 0
theorem line_passing_through_intersection_and_midpoint :
  ∃ l : ℝ → ℝ → Prop, l = (λ x y, 7 * x - 4 * y + 1 = 0) ∧
    (∃ x y, line1 x y ∧ line2 x y ∧ l x y) ∧ 
    l (midpoint A B).1 (midpoint A B).2 :=
sorry

end line_passing_through_intersection_and_midpoint_l94_94853


namespace max_t_value_l94_94967

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := |(Real.log (x) / Real.log(2) + a*x + b)|

noncomputable def M_t (a : ℝ) (b : ℝ) (t : ℝ) : ℝ :=
max (f t a b) (f (t + 1) a b)

theorem max_t_value (a : ℝ) (h_a : 0 < a) (t : ℝ) (h_t : 0 < t) : 
  (∀ (b : ℝ), M_t a b t ≥ a + 1) → t ≤ 1/(2^(a + 2) - 1) := by
sorry

end max_t_value_l94_94967


namespace solution_to_problem_l94_94142

noncomputable def find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (c1 : x * y = 24 * real.root 3 4) (c2 : x * z = 42 * real.root 3 4) 
  (c3 : y * z = 21 * real.root 3 4) : ℝ :=
  real.sqrt 63504

theorem solution_to_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (c1 : x * y = 24 * real.root 3 4) (c2 : x * z = 42 * real.root 3 4)
  (c3 : y * z = 21 * real.root 3 4) : 
  x * y * z = find_xyz x y z h1 h2 h3 c1 c2 c3 :=
sorry

end solution_to_problem_l94_94142


namespace sum_of_digits_n_plus_2_l94_94183

def T (n : ℕ) : ℕ := -- Define the sum of the digits function (As an example, skipping the detail)
  sorry

theorem sum_of_digits_n_plus_2 (n : ℕ) (h₁ : T(n) = 1598) : T(n + 2) = 1600 := by
  sorry

end sum_of_digits_n_plus_2_l94_94183


namespace first_to_receive_10_pieces_l94_94262

-- Definitions and conditions
def children := [1, 2, 3, 4, 5, 6, 7, 8]
def distribution_cycle := [1, 3, 6, 8, 3, 5, 8, 2, 5, 7, 2, 4, 7, 1, 4, 6]

def count_occurrences (n : ℕ) (lst : List ℕ) : ℕ :=
  lst.count n

-- Theorem
theorem first_to_receive_10_pieces : ∃ k, k = 3 ∧ count_occurrences k distribution_cycle = 2 :=
by
  sorry

end first_to_receive_10_pieces_l94_94262


namespace exists_airline_route_within_same_republic_l94_94914

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l94_94914


namespace teacherLinConcernedWithVariance_l94_94627

def xiaoWangScores : List ℝ := [86, 78, 80, 85, 92]

noncomputable def concernedMeasure (scores : List ℝ) : String :=
  if variance scores ≥ 0 then "Variance" else "Unknown"

-- We need to show that given the scores, Teacher Lin should be concerned about the variance
theorem teacherLinConcernedWithVariance:
  concernedMeasure xiaoWangScores = "Variance" :=
by
  sorry

end teacherLinConcernedWithVariance_l94_94627


namespace quadratic_to_vertex_form_l94_94576

noncomputable def quadratic : ℝ → ℝ
| x := 6 * x^2 + 36 * x + 216

theorem quadratic_to_vertex_form :
  ∃ a b c, ∀ x, quadratic x = a * (x + b) ^ 2 + c ∧ a + b + c = 171 :=
by
  use [6, 3, 162]
  -- Proof to be filled in.
  sorry

end quadratic_to_vertex_form_l94_94576


namespace gcf_7fact_8fact_l94_94755

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l94_94755


namespace exists_m_l94_94960

def cyclic_distance (p : ℕ) (x : ℕ) : ℕ :=
if x < p / 2 then x else p - x

def almost_additive {p : ℕ} (f : ℕ → ℕ) : Prop :=
∀ x y : ℕ, x < p → y < p → cyclic_distance p (f (x + y) % p - (f x + f y) % p) < 100

theorem exists_m {p : ℕ} [fact (nat.prime p)] {f : ℕ → ℕ} 
  (h_f : almost_additive f) : 
  ∃ m ∈ finset.range p, ∀ x < p, cyclic_distance p (f x % p - m * x % p) < 1000 :=
sorry

end exists_m_l94_94960


namespace problem_part_I_problem_part_II_l94_94406

open Real

-- Given the moving circle C and external circle E
def C (m : ℝ) : set (ℝ × ℝ) := {p | (p.1 - m)^2 + (p.2 - 2 * m)^2 = m^2}
def E : set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 16}

-- The conditions
axiom m_pos (m : ℝ) : m > 0

-- Problem Part I: Find the tangents when m = 2
theorem problem_part_I : (C 2).tangent (0, 0) = sorry

-- Problem Part II: Tangency condition between C and E
theorem problem_part_II (m : ℝ) :
  ((∃ p ∈ C m, p ∈ E) → m = (sqrt 29 - 1) / 4) ∧ (m = (sqrt 29 - 1) / 4 → (∃ p ∈ C m, p ∈ E)) := sorry

end problem_part_I_problem_part_II_l94_94406


namespace exists_airline_route_within_same_republic_l94_94903

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l94_94903


namespace calculate_y_l94_94677

theorem calculate_y (x : ℤ) (y : ℤ) (h1 : x = 121) (h2 : 2 * x - y = 102) : y = 140 :=
by
  -- Placeholder proof
  sorry

end calculate_y_l94_94677


namespace BURN_maps_to_8615_l94_94157

open List Function

def tenLetterMapping : List (Char × Nat) := 
  [('G', 0), ('R', 1), ('E', 2), ('A', 3), ('T', 4), ('N', 5), ('U', 6), ('M', 7), ('B', 8), ('S', 9)]

def charToDigit (c : Char) : Option Nat :=
  tenLetterMapping.lookup c

def wordToNumber (word : List Char) : Option (List Nat) :=
  word.mapM charToDigit 

theorem BURN_maps_to_8615 :
  wordToNumber ['B', 'U', 'R', 'N'] = some [8, 6, 1, 5] :=
by
  sorry

end BURN_maps_to_8615_l94_94157


namespace prop_2_prop_4_l94_94102

noncomputable theory

variables {Line Plane : Type} [LinearSpace Line Plane]
variables {m n l : Line} {α β : Plane}
variables (h1 : m ≠ n) (h2 : n ≠ l) (h3 : l ≠ m)
variables (α ≠ β)

-- Proposition ②: If m ⊄ α, n ⊂ α, m ∥ n, then m ∥ α.
theorem prop_2 (h4 : ¬m ⊂ α) (h5 : n ⊂ α) (h6 : m ∥ n) : m ∥ α := sorry

-- Proposition ④: If m ⊂ α, m ⊥ β, then α ⊥ β.
theorem prop_4 (h7 : m ⊂ α) (h8 : m ⊥ β) : α ⊥ β := sorry

end prop_2_prop_4_l94_94102


namespace proof_problem_l94_94802

theorem proof_problem (x y : ℝ) (h₁ : 4^x = 16^(y + 1)) (h₂ : 27^y = 9^(x - 6)) : x + y = 32 := 
sorry

end proof_problem_l94_94802


namespace total_books_l94_94590

theorem total_books (Tim_books Sam_books Emma_books : ℕ) 
  (hTim : Tim_books = 44) 
  (hSam : Sam_books = 52) 
  (hEmma : Emma_books = 37) : 
  Tim_books + Sam_books + Emma_books = 133 := 
by
  rw [hTim, hSam, hEmma]
  norm_num
  sorry

end total_books_l94_94590


namespace range_of_m_l94_94151

-- Define the conditions of the problem
def is_obtuse_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ max α (max β γ) > 90

def angles_arithmetic_sequence (α β γ : ℝ) : Prop :=
  β - α = γ - β

-- Define the main theorem to be proved
theorem range_of_m (α β γ : ℝ) (m : ℝ) (h_seq : angles_arithmetic_sequence α β γ)
  (h_obtuse : is_obtuse_triangle α β γ) (h_m : m = (sin (γ.to_real) / sin (α.to_real))) :
  m > 2 :=
sorry -- Proof to be provided

end range_of_m_l94_94151


namespace equation_of_line_l_l94_94822

-- Definition of the centers P and Q
def P : (ℝ × ℝ) := (7, -4)
def Q : (ℝ × ℝ) := (-5, 6)

-- Define the midpoint M of line segment PQ
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the slope of a line passing through points A and B
def slope (A B : ℝ × ℝ) : ℝ :=
  if B.1 - A.1 = 0 then 0 else (B.2 - A.2) / (B.1 - A.1)

-- Negative reciprocal of slope to define perpendicular line
def neg_reciprocal (k : ℝ) : ℝ :=
  if k = 0 then 0 else -1 / k

-- Theorem statement
theorem equation_of_line_l :
  let M := midpoint P Q in
  let k_PQ := slope P Q in
  let k_l := neg_reciprocal k_PQ in
  M = (1, 1) →
  k_l = 6 / 5 →
  (∀ x y: ℝ, y - 1 = (6 / 5) * (x - 1) → 6 * x - 5 * y - 1 = 0) →
  ∀ x y, (6 * x - 5 * y - 1 = 0) := 
by
  intros M_eq k_l_eq line_eq
  sorry

end equation_of_line_l_l94_94822


namespace discount_percentage_is_25_l94_94014

-- Defining the cost price as C
variable (C : ℝ)

-- Defining the marked price M
def marked_price := 1.60 * C

-- Defining the actual selling price A
def actual_selling_price := 1.20000000000000018 * C

-- Defining the discount D
def discount := marked_price C - actual_selling_price C

-- Defining the discount percentage d%
def discount_percentage := (discount C / marked_price C) * 100

-- Proof statement that the discount percentage is 25%
theorem discount_percentage_is_25 : discount_percentage C = 25 := by
  sorry

end discount_percentage_is_25_l94_94014


namespace gcf_7fact_8fact_l94_94753

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l94_94753


namespace tangent_line_of_ellipse_l94_94412

variable {a b x y x₀ y₀ : ℝ}

theorem tangent_line_of_ellipse
    (h1 : 0 < a)
    (h2 : a > b)
    (h3 : b > 0)
    (h4 : (x₀, y₀) ∈ { p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_of_ellipse_l94_94412


namespace find_point_P_l94_94092

theorem find_point_P :
  ∃ (x y : ℝ), 
    let M := (-2, 7)
    let N := (10, -2)
    let P := (x, y)
    (10 - x, -2 - y) = -2 * (-2 - x, 7 - y) ∧
    P = (2, 4) :=
begin
  sorry
end

end find_point_P_l94_94092


namespace parametric_circle_l94_94249

theorem parametric_circle (θ : ℝ) : ∃ (x y : ℝ), x = Real.cos (2 * θ) ∧ y = Real.sin (2 * θ) ∧ x^2 + y^2 = 1 :=
by
  exists Real.cos (2 * θ)
  exists Real.sin (2 * θ)
  split
  · rfl
  split
  · rfl
  calc
    Real.cos (2 * θ) ^ 2 + Real.sin (2 * θ) ^ 2
        = 1 : by rw [Real.cos_sq_add_sin_sq (2 * θ)]

end parametric_circle_l94_94249


namespace cube_face_sum_l94_94221

theorem cube_face_sum (a b c d e f : ℕ) (h1 : e = b) (h2 : 2 * (a * b * c + a * b * f + d * b * c + d * b * f) = 1332) :
  a + b + c + d + e + f = 47 :=
sorry

end cube_face_sum_l94_94221


namespace series_equality_l94_94994

theorem series_equality :
  (∑ n in Finset.range 200, (-1)^(n+1) * (1:ℚ) / (n+1)) = (∑ n in Finset.range 100, 1 / (100 + n + 1)) :=
by sorry

end series_equality_l94_94994


namespace L_of_A1_L_of_arithmetic_progression_l94_94087

-- Define the set A = {2, 4, 6, 8}
def A1 : Set ℕ := {2, 4, 6, 8}

-- Define L which counts the number of distinct sums a_i + a_j in set A
def L (A : Set ℕ) : ℕ := (A.powerset.filter (λ s, s.card = 2)).image (λ s, s.to_finset.sum).card

-- Theorem for the first part where A = {2, 4, 6, 8}
theorem L_of_A1 : L A1 = 5 := sorry

-- Define a function to generate an arithmetic progression
def arithmetic_progression (a d m : ℕ) : List ℕ := List.range m |>.map (λ i, a + i * d)

-- Convert list to set
def list_to_set (l : List ℕ) : Set ℕ := l.to_finset.to_set

-- Theorem for the second part where A is an arithmetic progression
theorem L_of_arithmetic_progression (a d m : ℕ) (hm : 3 ≤ m) :
  let A := list_to_set (arithmetic_progression a d m) in
  L A = 2 * m - 3 := sorry

end L_of_A1_L_of_arithmetic_progression_l94_94087


namespace exists_airline_route_within_same_republic_l94_94899

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l94_94899


namespace necessary_but_not_sufficient_l94_94840

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  1 / (x - 1) + a / (x + a - 1) + 1 / (x + 1)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

def condition_a_zero (a : ℝ) : Prop := a = 0

def proof_problem (a : ℝ) : Prop :=
  (is_odd_function (λ x => f x a)) → condition_a_zero a

theorem necessary_but_not_sufficient : proof_problem a ↔ (condition_a_zero a ∧ (∃ b : ℝ, b ≠ 0 ∧ is_odd_function (λ x => f x b))) :=
sorry

end necessary_but_not_sufficient_l94_94840


namespace exists_airline_route_within_same_republic_l94_94913

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l94_94913


namespace gcf_7_8_fact_l94_94746

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l94_94746


namespace remainder_1492_mul_1999_mod_500_l94_94616

theorem remainder_1492_mul_1999_mod_500 : (1492 * 1999) % 500 = 8 := by
  have h1: 1492 % 500 = 492 := by 
    norm_num
  have h2: 1999 % 500 = 499 := by 
    norm_num
  have h3: 492 * 499 % 500 = 8 := by 
    norm_num
  rw [←h1, ←h2]
  exact h3

end remainder_1492_mul_1999_mod_500_l94_94616


namespace gcf_7_8_fact_l94_94748

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l94_94748


namespace shopkeepers_count_l94_94325

def initial_amount := 8.75
def final_amount := 0
def total_spend (n : ℕ) : ℝ := 10 * (2^n - 1)

theorem shopkeepers_count (n : ℕ) :
  (2^n * initial_amount - total_spend n = final_amount) -> n = 3 :=
by
  sorry

end shopkeepers_count_l94_94325


namespace solution1_solution2_l94_94838

def f (x : ℝ) : ℝ := |2 * x - 3| + |2 * x - 1|

theorem solution1 (x : ℝ) : (f x ≥ 3) ↔ (x ≤ 1/4 ∨ x ≥ 7/4) := by
  sorry

theorem solution2 (m n x : ℝ) (h1 : m + n = 1) (h2 : f x ≥ 2) : 
  (sqrt (2 * m + 1) + sqrt (2 * n + 1) ≤ 2 * sqrt (f x)) := by
  sorry

end solution1_solution2_l94_94838


namespace M_coordinates_l94_94813

noncomputable def point_on_z_axis (z : ℝ) := (0, 0, z)

def dist (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

def point_equidistant (M A B : ℝ × ℝ × ℝ) : Prop :=
  dist M A = dist M B

theorem M_coordinates :
  ∃ z : ℝ, let M := point_on_z_axis z in point_equidistant M (1, 0, 2) (1, -3, 1) ∧ M = (0, 0, 3) :=
sorry

end M_coordinates_l94_94813


namespace two_digit_squares_divisible_by_4_l94_94865

theorem two_digit_squares_divisible_by_4 : 
  ∃ n : ℕ, n = 3 ∧ (n = (count (λ x : ℕ, 10 ≤ x ∧ x ≤ 99 ∧ (∃ k : ℕ, x = k * k ∧ x % 4 = 0)))) :=
by 
  sorry

end two_digit_squares_divisible_by_4_l94_94865


namespace part_i_part_ii_l94_94720

-- Sequence definition
def seq (n : ℕ) : ℕ := 3^(2^n) + 1

-- Part (i): Infinite primes that don't divide any term of the sequence
theorem part_i : ∃ᶠ p in (λ n, nat.prime (seq n)), true :=
begin
  sorry
end

-- Part (ii): Infinite primes that divide some term of the sequence
theorem part_ii : ∃ᶠ p in (λ p, ∃ n, p.prime ∧ p ∣ seq n), true :=
begin
  sorry
end

end part_i_part_ii_l94_94720


namespace range_of_m_l94_94079

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y - x * y = 0) :
    (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y - x * y = 0 → x + 2 * y > m^2 + 2 * m) ↔ (-4 : ℝ) < m ∧ m < 2 :=
by
  sorry

end range_of_m_l94_94079


namespace number_of_men_at_beginning_l94_94198

theorem number_of_men_at_beginning :
  ∀ (M C : ℕ), W + M + C = 60 ∧ W = 30 ∧ (2 * M / 3) + C = 25 ∧ M + C = 30 → M = 15 :=
by
  intros M C h,
  cases' h with h1 h2,
  cases' h2 with h3 h4,
  cases' h4 with h5 h6,
  sorry

end number_of_men_at_beginning_l94_94198


namespace smallest_prime_factor_of_2310_l94_94619

theorem smallest_prime_factor_of_2310 : ∃ p : ℕ, nat.prime p ∧ p ∣ 2310 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 2310 → p ≤ q :=
by {
  -- Sorry is used to indicate proof has been omitted.
  sorry
}

end smallest_prime_factor_of_2310_l94_94619


namespace gcf_factorial_seven_eight_l94_94759

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l94_94759


namespace alternatingsum_sum_equals_1024_l94_94064

-- Define the set {1, 2, 3, 4, 5, 6, 7, 8}
def full_set := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the function to compute alternating sum
def alternating_sum (s : Finset ℕ) : ℤ :=
  if s.nonempty then (s.sort (· > ·)).val.enum.map (λ ⟨i, n⟩, if i % 2 = 0 then n else -n).sum
  else 0

-- Define the theorem that sum of all alternating sums equals 1024
theorem alternatingsum_sum_equals_1024 :
  (Finset.powerset full_set).filter (λ s, s.nonempty).sum (λ s, alternating_sum s) = 1024 :=
by 
  sorry

end alternatingsum_sum_equals_1024_l94_94064


namespace exist_partition_of_delegates_l94_94016

/-- At a symposium, each delegate is acquainted with at least one of the other participants but not with everyone. 
    Prove that all delegates can be divided into two groups so that each participant in the symposium is acquainted with at least one person in their group. -/
theorem exist_partition_of_delegates 
  (G : SimpleGraph V)
  [DecidableRel G.Adj]
  (h1 : ∀ v : V, ∃ (w : V), G.Adj v w) 
  (h2 : ∀ v : V, ∃ (w : V), v ≠ w ∧ ¬G.Adj v w) : 
  ∃ (V₁ V₂ : Set V), 
    (V₁ ∪ V₂ = Set.univ) ∧ 
    (V₁ ∩ V₂ = ∅) ∧ 
    (∀ v ∈ V₁, ∃ w ∈ V₁, G.Adj v w) ∧ 
    (∀ v ∈ V₂, ∃ w ∈ V₂, G.Adj v w) :=
by
  sorry

end exist_partition_of_delegates_l94_94016


namespace cross_section_area_l94_94934

variables (A B C D A1 B1 C1 D1 : Type) [InnerProductSpace ℝ A]
variables {AB AD BD AA1 : ℝ}

def dimensions (AB AD BD AA1 : ℝ) : Prop :=
  AB = 29 ∧ AD = 36 ∧ BD = 25 ∧ AA1 = 48

theorem cross_section_area (h : dimensions AB AD BD AA1) :
  ∃A B C D A1 B1 C1 D1, area_of_cross_section AB AD BD AA1 = 1872 :=
by { sorry }

end cross_section_area_l94_94934


namespace arithmetic_sequence_a3_l94_94944

theorem arithmetic_sequence_a3 :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
    (∀ n, a n = 2 + (n - 1) * d) ∧
    (a 1 = 2) ∧
    (a 5 = a 4 + 2) →
    a 3 = 6 :=
sorry

end arithmetic_sequence_a3_l94_94944


namespace range_of_a_l94_94789

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), 0 < x → (x - a + log (x / a)) * (-2 * x^2 + a * x + 10) ≤ 0) →
  a = real.sqrt 10 :=
sorry

end range_of_a_l94_94789


namespace units_digit_of_power_435_l94_94061

def units_digit_cycle (n : ℕ) : ℕ :=
  n % 2

def units_digit_of_four_powers (cycle : ℕ) : ℕ :=
  if cycle = 0 then 6 else 4

theorem units_digit_of_power_435 : 
  units_digit_of_four_powers (units_digit_cycle (3^5)) = 4 :=
by
  sorry

end units_digit_of_power_435_l94_94061


namespace distance_to_hole_l94_94682

-- Define the variables from the problem
variables (distance_first_turn distance_second_turn beyond_hole total_distance hole_distance : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first_turn = 180 ∧
  distance_second_turn = distance_first_turn / 2 ∧
  beyond_hole = 20 ∧
  total_distance = distance_first_turn + distance_second_turn

-- The main statement we need to prove
theorem distance_to_hole : conditions →
  hole_distance = total_distance - beyond_hole → hole_distance = 250 :=
by
  sorry

end distance_to_hole_l94_94682


namespace hanna_has_money_l94_94858

variable (total_roses money_spent : ℕ)
variable (rose_price : ℕ := 2)

def hanna_gives_roses (total_roses : ℕ) : Bool :=
  (1 / 3 * total_roses + 1 / 2 * total_roses) = 125

theorem hanna_has_money (H : hanna_gives_roses total_roses) : money_spent = 300 := sorry

end hanna_has_money_l94_94858


namespace weight_of_NH4Cl_l94_94044

/--
Given:
- temperature T = 1500 K
- pressure P = 1200 mmHg
- ideal gas law: PV = nRT
- number of moles n = 8

Prove:
The weight of 8 moles of NH4Cl is 428 grams.
-/
theorem weight_of_NH4Cl (T : ℝ) (P : ℝ) (n : ℝ) (R : ℝ) : (T = 1500) → (P = 1200) → (n = 8) → (R = 0.0821) → 
  let molar_mass_NH4Cl := 53.5
  let atm_conversion := 1 / 760
  let P_atm := P * atm_conversion
  let V := (n * R * T) / P_atm
  let weight := n * molar_mass_NH4Cl
  weight = 428 :=
begin
  intros hT hP hn hR,
  rw [hT, hP, hn, hR],
  let molar_mass_NH4Cl := 53.5,
  let atm_conversion := 1 / 760,
  let P_atm := 1200 * atm_conversion,
  have P_atm_val : P_atm = 1.5789473684210527 := by norm_num [atm_conversion],
  let V := (8 * 0.0821 * 1500) / P_atm,
  have V_val : V = 51.15894039735158 := by norm_num [P_atm_val],
  let weight := 8 * molar_mass_NH4Cl,
  have weight_val : weight = 428 := by norm_num [molar_mass_NH4Cl],
  exact weight_val,
end

end weight_of_NH4Cl_l94_94044


namespace arrangement_with_tallest_in_middle_and_symmetry_arrangements_in_2x3_grid_l94_94585

-- We define the problem with the necessary conditions and translate it to Lean statements.

theorem arrangement_with_tallest_in_middle_and_symmetry (students : Finset ℕ) (h : students.card = 7):
  let tallest_student := students.max' sorry in
  let remaining_students := students.erase tallest_student in
  (∃! arrangement : Vector ℕ 7, 
    arrangement.nth 3 = tallest_student ∧
    (∀ i, i < 3 → arrangement.nth i > arrangement.nth (i + 1)) ∧
    (∀ i, 3 < i → arrangement.nth i > arrangement.nth (i - 1))) → 20 := 
sorry

theorem arrangements_in_2x3_grid (students : Finset ℕ) (h : students.card = 6):
  (∃! arrangement : Matrix ℕ (Fin 2) (Fin 3), 
    (∀ i, arrangement 0 i < arrangement 1 i) ∧
    (arrangement.to_list.nodup)) → 630 := 
sorry

end arrangement_with_tallest_in_middle_and_symmetry_arrangements_in_2x3_grid_l94_94585


namespace intersecting_diameter_circle_l94_94427

theorem intersecting_diameter_circle (a : ℝ) :
  (∀ x y : ℝ, ax + y - 2 = 0 → (x - 1)^2 + (y - a)^2 = 4) →
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    ax + (snd A) - 2 = 0 ∧ (fst A - 1)^2 + (snd A - a)^2 = 4 ∧
    ax + (snd B) - 2 = 0 ∧ (fst B - 1)^2 + (snd B - a)^2 = 4 ∧
    dist A B = 4) →
  a = 1 := 
by 
  sorry

end intersecting_diameter_circle_l94_94427


namespace prove_f_expr_g_range_l94_94845

def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 3)

def a (x : ℝ) : ℝ × ℝ := (f (x - Real.pi / 6), 1)
def b (x : ℝ) : ℝ × ℝ := (1 / 2, -2 * Real.cos x)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def g (x : ℝ) : ℝ :=
  dot_product (a x) (b x) + 1 / 2

theorem prove_f_expr : f = (λ x, 2 * Real.cos (2 * x + Real.pi / 3)) :=
by sorry

theorem g_range : ∀ x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 2), 
  g x ∈ Set.Icc (-1 : ℝ) (Math.sqrt 2 + 1 / 2) :=
by sorry

end prove_f_expr_g_range_l94_94845


namespace length_of_park_l94_94333

variable (L : ℝ)
variable (width_park : ℝ) (area_lawn : ℝ) (width_road : ℝ)
variable (roads : ℕ)

#check width_park
#check area_lawn

theorem length_of_park
  (h_width_park : width_park = 40)
  (h_area_lawn : area_lawn = 2109)
  (h_width_road : width_road = 3)
  (h_roads : roads = 2)
  (h_condition : L * width_park = area_lawn + real.of_nat roads * L) :
  L = 55.5 :=
  sorry

end length_of_park_l94_94333


namespace number_of_valid_triangles_l94_94058

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c = 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2

def count_valid_triangles : ℕ :=
  (Finset.filter (λ (xyz : ℕ × ℕ × ℕ), is_valid_triangle xyz.1 xyz.2.1 xyz.2.2)
    ((Finset.range 20).product (Finset.range 20).product (Finset.range 20))).card

theorem number_of_valid_triangles : count_valid_triangles = 8 :=
  sorry

end number_of_valid_triangles_l94_94058


namespace find_a_l94_94569

-- Definitions based on the given condition
def point (x y : ℝ) := (x, y)
def parabola (a : ℝ) (x : ℝ) := a * x^2
def directrix (a : ℝ) := - (1 / (4 * a))

-- Distance from a point to a line y = mx + b
def distance_to_line (p : ℝ × ℝ) (m b : ℝ) : ℝ :=
  abs (snd p - m * fst p - b)

-- Given point M and distance condition
def M : ℝ × ℝ := point 1 1
def distance := 2

-- Lean statement to prove
theorem find_a (a : ℝ) :
  distance_to_line M 0 (directrix a) = distance →
  a = 1/4 ∨ a = -1/12 :=
by sorry

end find_a_l94_94569


namespace dave_initial_files_l94_94726

theorem dave_initial_files 
  (initial_apps : ℕ) (apps_left : ℕ) (files_left : ℕ) (more_apps_than_files : ℕ)
  (initial_files : ℕ) :
  initial_apps = 15 →
  apps_left = 21 →
  files_left = 4 →
  more_apps_than_files = 17 →
  initial_files - (apps_left - initial_apps) = files_left →
  initial_files = 10 :=
begin
  intros h1 h2 h3 h4 h5,
  simp at *,
  sorry
end

end dave_initial_files_l94_94726


namespace sum_squares_eq_l94_94826

-- Define the sum of the first n terms of a geometric sequence
def S_n (n : ℕ) : ℕ := 2^n - 1

-- Define the terms of the geometric sequence
def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2^(n-1)

-- Define the sum of squares of the terms of the geometric sequence
def sum_squares (n : ℕ) : ℕ :=
  (List.range n).foldr (λ i acc, (a (i+1))^2 + acc) 0

-- Prove the main theorem
theorem sum_squares_eq : ∀ n : ℕ, sum_squares n = (4^n - 1) / 3 :=
by
  intro n
  sorry

end sum_squares_eq_l94_94826


namespace original_number_of_percentage_l94_94328

theorem original_number_of_percentage:
  ∀ n : ℝ, n = 501.99999999999994 → n / 100 = 5.0199999999999994 := by
  intros n hn
  rw hn
  sorry

end original_number_of_percentage_l94_94328


namespace tan_theta_negative_one_max_magnitude_sum_vecs_l94_94131

variables {θ : ℝ}

-- Proof Problem (I)
theorem tan_theta_negative_one (h₁ : - (π / 2) < θ) (h₂ : θ < π / 2) (h₃ : (sin θ, 1) ⬝ (1, cos θ) = 0) :
  tan θ = -1 :=
by sorry

-- Proof Problem (II)
theorem max_magnitude_sum_vecs (h₁ : - (π / 2) < θ) (h₂ : θ < π / 2) :
  ∥(sin θ + 1, cos θ + 1)∥ ≤ 1 + real.sqrt 2 :=
by sorry

end tan_theta_negative_one_max_magnitude_sum_vecs_l94_94131


namespace problem_part1_problem_part2_l94_94816

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem_part1 (h1 : ∀ x : ℝ, f (-x) = -f x) 
                      (h2 : ∀ x : ℝ, f (x + 2) = f x + 2) : 
                      f 1 = 1 := 
by
  sorry

theorem problem_part2 (h1 : ∀ x : ℝ, f (-x) = -f x) 
                      (h2 : ∀ x : ℝ, f (x + 2) = f x + 2) :
                      ∑ k in Finset.range 20, f (k + 1) = 210 := 
by
  sorry

end problem_part1_problem_part2_l94_94816


namespace airline_route_within_republic_l94_94907

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l94_94907


namespace room_number_unit_digit_is_8_l94_94197

def is_room_number (n : ℕ) : Prop :=
  n > 9 ∧ n < 100 ∧
  (prime n ∨ even n ∨ n % 7 = 0 ∨ (9 ∈ [n / 10, n % 10])) ∧
  (∃ cond_true : Fin 4 → Bool,
    (cond_true 0 = true ∧ ¬ prime n ∨ cond_true 0 = false ∧ prime n) ∧
    (cond_true 1 = true ∧ even n ∨ cond_true 1 = false ∧ ¬ even n) ∧
    (cond_true 2 = true ∧ n % 7 = 0 ∨ cond_true 2 = false ∧ n % 7 ≠ 0) ∧
    (cond_true 3 = true ∧ (9 ∈ [n / 10, n % 10]) ∨ cond_true 3 = false ∧ ¬ (9 ∈ [n / 10, n % 10])) ∧
    (cond_true 0 + cond_true 1 + cond_true 2 + cond_true 3 = 3))

theorem room_number_unit_digit_is_8 :
  ∃ (n : ℕ), is_room_number n ∧ n % 10 = 8 :=
by
  sorry

end room_number_unit_digit_is_8_l94_94197


namespace complex_cubed_l94_94404

theorem complex_cubed (z : ℂ) (h1 : |z - 2| = 2) (h2 : |z| = 2) : z ^ 3 = -8 :=
sorry

end complex_cubed_l94_94404


namespace area_of_right_triangle_l94_94978

-- Given the conditions
variables {P Q R : ℝ × ℝ} -- Points in the xy-plane
variable {PQ : ℝ} (median_P : (ℝ → ℝ)) (median_Q : (ℝ → ℝ))
variable (right_angle_R : Bool) (length_hypotenuse_PQ : ℝ)
variable (angle_PRQR : Bool)

-- Defining the conditions
def is_right_triangle (R : ℝ × ℝ) : Bool := right_angle_R
def hypotenuse_length : ℝ := length_hypotenuse_PQ
def median_through_P : (ℝ → ℝ) := median_P
def median_through_Q : (ℝ → ℝ) := median_Q

-- Prove that the area is 125/3 given the conditions
theorem area_of_right_triangle 
  (h1: is_right_triangle R)
  (h2: hypotenuse_length = 50)
  (h3: median_through_P = λ x, x + 2)
  (h4: median_through_Q = λ x, 3 * x + 5) :
  ∃ area: ℝ, area = 125 / 3 := 
by 
  sorry  -- Placeholder for the proof

end area_of_right_triangle_l94_94978


namespace distinct_payment_amounts_excludes_no_payment_l94_94113

-- Define the problem conditions
def coins : List ℕ := [1, 5, 5, 1, 1, 1, 1, 1, 5, 5] -- coins in jiao (1 jiao = 1, 1 yuan = 10 jiao, 5 yuan = 50 jiao)

-- The theorem to be proven
theorem distinct_payment_amounts_excludes_no_payment (coins : List ℕ) : 
  let amounts := calc_distinct_amounts coins in  -- Calculate distinct payment amounts
  amounts.card = 127 :=  -- The number of unique possible amounts is 127
sorry

-- Helper function to calculate distinct payment amounts
noncomputable def calc_distinct_amounts (coins : List ℕ) : Finset ℕ :=
  coins.powerset.map (λ s, s.sum).erase 0 -- Generate all possible sums, eliminate 0

end distinct_payment_amounts_excludes_no_payment_l94_94113


namespace gcf_7_factorial_8_factorial_l94_94768

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l94_94768


namespace airline_route_same_republic_exists_l94_94928

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l94_94928


namespace max_profit_at_nine_l94_94422

noncomputable def profit_function (x : ℝ) : ℝ :=
  -(1/3) * x ^ 3 + 81 * x - 234

theorem max_profit_at_nine :
  ∃ x, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function 9 :=
by
  sorry

end max_profit_at_nine_l94_94422


namespace general_term_formula_l94_94110

noncomputable def geom_seq_general_term (n : ℕ) : ℕ → ℚ
| 0     := 0
| 1     := 1
| (n+1) := (1 / 2 : ℚ) * geom_seq_general_term n

theorem general_term_formula (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h1 : ∀ n, S n = 2 - a n) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
by
  sorry

end general_term_formula_l94_94110


namespace exists_route_within_republic_l94_94922

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l94_94922


namespace ellipse_eqn_and_max_area_l94_94835

-- Given conditions and setup
variables {a b c : ℝ} (x y k : ℝ)
variables (D : ℝ × ℝ) 
variables {O A B N : ℝ × ℝ}

-- Condition: the ellipse equation and eccentricity constraint
def is_ellipse (x y a b : ℝ) := (x / a)^2 + (y / b)^2 = 1
def eccentricity_constraint (a c : ℝ) := c / a = real.sqrt 3 / 2
def length_major_axis (a c : ℝ) := a + c = 2 + real.sqrt 3

-- Condition: point on the ellipse other than endpoints of the major axis
def valid_point (M : ℝ × ℝ) := ∃ x y, x ≠ a ∧ x ≠ -a ∧ is_ellipse x y a b

-- Condition: perimeter of triangle
def perimeter_triangle (M F1 F2 : ℝ × ℝ) := sorry -- express the perimeter formula

-- Condition: line passing through D and intersects ellipse at A and B
def line_through_D (D : ℝ × ℝ) (k : ℝ) := λ x, k * x - 2
def line_intersects_ellipse (l : ℝ → ℝ) (A B : ℝ × ℝ) := 
  ∃ x1 x2 y1 y2, l x1 = y1 ∧ l x2 = y2 ∧ (is_ellipse x1 y1 a b) ∧ (is_ellipse x2 y2 a b)

-- Condition: point N setup
def point_N (O A B N : ℝ × ℝ) := (N = (fst A + fst B, snd A + snd B))

-- Prove the equation of the ellipse is canonical and find max area of quadrilateral OANB
theorem ellipse_eqn_and_max_area :
  valid_point M ∧ length_major_axis a c ∧ eccentricity_constraint a c ∧ 
  (perimeter_triangle M F1 F2) = 4 + 2 * real.sqrt 3 →
  (line_through_D (0, -2) k) → (line_intersects_ellipse (line_through_D (0, -2) k) A B) →
  point_N O A B N →
  is_ellipse x y 2 1 ∧
  ∃ l : ℝ → ℝ, l = (λ x, (real.sqrt 7 / 2) * x - 2) ∧
  quadrilateral_max_area O A N B 2 :=
sorry

end ellipse_eqn_and_max_area_l94_94835


namespace value_of_p_plus_q_l94_94413

theorem value_of_p_plus_q (p q : ℝ) (h1 : ∀ x : ℂ, (2 * x^2 + p * x + q = 0) → (x = (-3 + 2 * complex.I) ∨ x = (-3 - 2 * complex.I))) : 
  p + q = 38 := 
sorry

end value_of_p_plus_q_l94_94413


namespace train_length_l94_94340

theorem train_length
  (speed_kmh : ℕ) (time_s : ℕ) (speed_conversion_factor : ℚ) :
  speed_kmh = 72 →
  time_s = 16 →
  speed_conversion_factor = 5 / 18 →
  ((speed_kmh * speed_conversion_factor : ℚ) * time_s) = 320 :=
by
  intros h_speed h_time h_conversion
  rw [h_speed, h_time, h_conversion]
  norm_num
  sorry

end train_length_l94_94340


namespace fraction_to_decimal_zeros_l94_94280

theorem fraction_to_decimal_zeros (a b : ℕ) (h₁ : a = 5) (h₂ : b = 1600) : 
  (let dec_rep := (a : ℝ) / b;
       num_zeros := (String.mk (dec_rep.to_decimal_string.to_list) - '0')
  in num_zeros = 3) := 
  sorry

end fraction_to_decimal_zeros_l94_94280


namespace probability_three_tails_one_head_l94_94462

theorem probability_three_tails_one_head :
  let outcome_probability := (1 / 2) ^ 4 in
  let combinations := Finset.card (Finset.filter (λ (s : Finset.Fin 4 × Bool), s.2).Finset (Finset.product (Finset.range 4) (Finset.singleton (false, true, true, true)))) in
  outcome_probability * combinations = 1 / 4 :=
sorry

end probability_three_tails_one_head_l94_94462


namespace sin_alpha_cos_alpha_l94_94076

theorem sin_alpha_cos_alpha (α : ℝ) (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l94_94076


namespace find_k_value_l94_94385

theorem find_k_value (k : ℝ) (h₁ : ∀ x, k * x^2 - 5 * x - 12 = 0 → (x = 3 ∨ x = -4 / 3)) : k = 3 :=
sorry

end find_k_value_l94_94385


namespace mad_hatter_wait_time_l94_94982

theorem mad_hatter_wait_time 
  (mad_hatter_fast_rate : ℝ := 15 / 60)
  (march_hare_slow_rate : ℝ := 10 / 60)
  (meeting_time : ℝ := 5) :
  let mad_hatter_real_time := meeting_time * (60 / (60 + mad_hatter_fast_rate * 60)),
      march_hare_real_time := meeting_time * (60 / (60 - march_hare_slow_rate * 60)),
      waiting_time := march_hare_real_time - mad_hatter_real_time
  in waiting_time = 2 :=
by 
  sorry

end mad_hatter_wait_time_l94_94982


namespace alpha_value_l94_94420

noncomputable def find_alpha (α : ℝ) : Prop :=
  ∃ α, (α > π / 2) ∧ (α < 3 * π / 2) ∧
    ∃ A B C : ℝ × ℝ, 
      A = (3, 0) ∧ 
      B = (0, 3) ∧ 
      C = (Real.cos α, Real.sin α) ∧ 
      ∃ k : ℝ, 
        ((Real.cos α) / (-3) = k) ∧ 
        ((Real.sin α) / (3) = k) ∧ 
        α = 3 * π / 4

theorem alpha_value : find_alpha (3 * π / 4) :=
by
  sorry

end alpha_value_l94_94420


namespace day_of_twentieth_l94_94880

theorem day_of_twentieth 
  (h1 : ∃ n : ℕ, n ≤ 3 ∧ n > 0 ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → is_sunday (d ⟨2 * i, d<31⟩))
  : is_thursday (d 20) := sorry

end day_of_twentieth_l94_94880


namespace problem_2022_integer_part_l94_94253

-- Define the sequence {a_n}
def a : ℕ → ℝ
| 0       := 3 / 2
| (n + 1) := (a n) ^ 2 - a n + 1

-- Define the sum function for the first 2022 terms
def sum_a (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, 1 / a (k + 1)

-- Define the main theorem
theorem problem_2022_integer_part :
  ∀ n : ℕ, n = 2022 → ⌊sum_a n⌋ = 1 :=
by {
  sorry
}

end problem_2022_integer_part_l94_94253


namespace lemons_needed_l94_94643

theorem lemons_needed (lemons32 : ℕ) (lemons4 : ℕ) (h1 : lemons32 = 24) (h2 : (24 : ℕ) / 32 = (lemons4 : ℕ) / 4) : lemons4 = 3 := 
sorry

end lemons_needed_l94_94643


namespace conjugate_of_z_l94_94107

-- Define the conditions
def z : ℂ := 1 - 2 * Complex.I

-- State the problem and correct answer
theorem conjugate_of_z : Complex.conjugate z = 1 + 2 * Complex.I :=
  by
  sorry

end conjugate_of_z_l94_94107


namespace compare_neg_frac_compare_abs_l94_94713

theorem compare_neg_frac : - (7 / 8) < - (6 / 7) :=
sorry

theorem compare_abs : |(-0.1)| > -0.2 :=
sorry

end compare_neg_frac_compare_abs_l94_94713


namespace range_of_a_l94_94852

theorem range_of_a (x a : ℝ) :
  (x - 1 ≥ a^2) ∧ (x - 4 < 2a) → (-1 < a ∧ a < 3) :=
by 
  sorry

end range_of_a_l94_94852


namespace exist_odd_sum_multiple_l94_94084

def sum_of_digits (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c - '0'.toNat)).sum

theorem exist_odd_sum_multiple (M : ℕ) (hM : M > 0) : 
  ∃ k : ℕ, k % M = 0 ∧ odd (sum_of_digits k) :=
begin
  sorry
end

end exist_odd_sum_multiple_l94_94084


namespace length_of_base_AD_l94_94490

-- Definitions based on the conditions
def isosceles_trapezoid (A B C D : Type) : Prop := sorry -- Implementation of an isosceles trapezoid
def length_of_lateral_side (A B C D : Type) : ℝ := 40 -- The lateral side is 40 cm
def angle_BAC (A B C D : Type) : ℝ := 45 -- The angle ∠BAC is 45 degrees
def bisector_O_center (O A B D M : Type) : Prop := sorry -- Implementation that O is the center of circumscribed circle and lies on bisector

-- Main theorem based on the derived problem statement
theorem length_of_base_AD (A B C D O M : Type) 
  (h_iso_trapezoid : isosceles_trapezoid A B C D)
  (h_length_lateral : length_of_lateral_side A B C D = 40)
  (h_angle_BAC : angle_BAC A B C D = 45)
  (h_O_center_bisector : bisector_O_center O A B D M)
  : ℝ :=
  20 * (Real.sqrt 6 + Real.sqrt 2)

end length_of_base_AD_l94_94490


namespace find_prices_max_basketballs_l94_94224

-- Define price of basketballs and soccer balls
def basketball_price : ℕ := 80
def soccer_ball_price : ℕ := 50

-- Define the equations given in the problem
theorem find_prices (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 310)
  (h2 : 5 * x + 2 * y = 500) : 
  x = basketball_price ∧ y = soccer_ball_price :=
sorry

-- Define the maximum number of basketballs given the cost constraints
theorem max_basketballs (m : ℕ)
  (htotal : m + (60 - m) = 60)
  (hcost : 80 * m + 50 * (60 - m) ≤ 4000) : 
  m ≤ 33 :=
sorry

end find_prices_max_basketballs_l94_94224


namespace find_final_position_calculate_fuel_consumption_l94_94536

-- Part 1: Final position and direction from the guard post
theorem find_final_position (travel_records : List Int) (total_distance : Int)
    (hp1 : travel_records = [+2, -3, +5, -4, +6, -2, +4, -2])
    (hp2 : total_distance = travel_records.sum) : total_distance = 6 :=
by
  sorry

-- Part 2: Total fuel consumption for the entire journey
theorem calculate_fuel_consumption (travel_records : List Int) (fuel_per_10km : Float)
    (hp1 : travel_records = [+2, -3, +5, -4, +6, -2, +4, -2])
    (hp2 : fuel_per_10km = 0.8)
    (total_distance_abs : Int)
    (total_fuel : Float)
    (hp3 : total_distance_abs = travel_records.foldl (λ acc x => acc + x.natAbs) 0)
    (hp4 : total_fuel = (Float.ofNat total_distance_abs / 10.0) * fuel_per_10km) : total_fuel = 2.24 :=
by
  sorry

end find_final_position_calculate_fuel_consumption_l94_94536


namespace imo_1993_q34_l94_94188

def f (x : ℤ) (n : ℤ) : ℤ := x^n + 5 * x^(n - 1) + 3

theorem imo_1993_q34 (n : ℤ) (hn : n > 1) :
  ¬ ∃ (g h : ℤ[X]), degree g ≥ 1 ∧ degree h ≥ 1 ∧ f 0 n = (g * h 0)  := 
by 
  sorry

end imo_1993_q34_l94_94188


namespace sum_reciprocal_not_integer_l94_94993

theorem sum_reciprocal_not_integer : 
  ¬ ∃ (z : ℤ), z = ∑ m in finset.range 1986 \+ 1, ∑ n in finset.range 1986 \+ 1, (1:ℚ) / (m * n) := 
  sorry

end sum_reciprocal_not_integer_l94_94993


namespace age_product_difference_l94_94685

theorem age_product_difference (age_today : ℕ) (product_today : ℕ) (product_next_year : ℕ) :
  age_today = 7 →
  product_today = age_today * age_today →
  product_next_year = (age_today + 1) * (age_today + 1) →
  product_next_year - product_today = 15 :=
by
  sorry

end age_product_difference_l94_94685


namespace rooks_arrangement_count_l94_94491

theorem rooks_arrangement_count : 
  let total_squares := 64
  let threatened_by_rook := 14
  let remaining_squares := total_squares - 1 - threatened_by_rook
  total_squares * remaining_squares = 3136 :=
by
  let total_squares := 64
  let threatened_by_rook := 14
  let remaining_squares := total_squares - 1 - threatened_by_rook
  have h : total_squares * remaining_squares = 3136 := by
    simp [total_squares, threatened_by_rook, remaining_squares]
    sorry
  exact h

end rooks_arrangement_count_l94_94491


namespace monotonicity_of_f_max_value_g_diff_l94_94646

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x - a * x - 3 / (a * x)
noncomputable def g (x a : ℝ) : ℝ := f x a + x^2 + 3 / (a * x)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 3 / a → 0 < deriv (f x a) x) ∧
  (∀ x : ℝ, x > 3 / a → deriv (f x a) x < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < -1 / a → 0 > deriv (f x a) x) ∧
  (∀ x : ℝ, x > -1 / a → 0 < deriv (f x a) x) :=
sorry

theorem max_value_g_diff (a : ℝ) (x1 x2 : ℝ) (h : a > 4) (hx : x1 < x2) :
  g x2 a - 2 * g x1 a = 3 * Real.log 2 + 1 :=
sorry

end monotonicity_of_f_max_value_g_diff_l94_94646


namespace monotonicity_of_f_max_value_g_diff_l94_94645

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.log x - a * x - 3 / (a * x)
noncomputable def g (x a : ℝ) : ℝ := f x a + x^2 + 3 / (a * x)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 3 / a → 0 < deriv (f x a) x) ∧
  (∀ x : ℝ, x > 3 / a → deriv (f x a) x < 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < -1 / a → 0 > deriv (f x a) x) ∧
  (∀ x : ℝ, x > -1 / a → 0 < deriv (f x a) x) :=
sorry

theorem max_value_g_diff (a : ℝ) (x1 x2 : ℝ) (h : a > 4) (hx : x1 < x2) :
  g x2 a - 2 * g x1 a = 3 * Real.log 2 + 1 :=
sorry

end monotonicity_of_f_max_value_g_diff_l94_94645


namespace election_result_l94_94939

theorem election_result (total_votes : ℕ) (invalid_vote_percentage valid_vote_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (hv: valid_vote_percentage = 1 - invalid_vote_percentage) 
  (ht: total_votes = 560000) 
  (hi: invalid_vote_percentage = 0.15) 
  (hc: candidate_A_percentage = 0.80) : 
  (candidate_A_percentage * valid_vote_percentage * total_votes = 380800) :=
by 
  sorry

end election_result_l94_94939


namespace gcf_7_factorial_8_factorial_l94_94769

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l94_94769


namespace four_digit_numbers_sum_even_l94_94796

theorem four_digit_numbers_sum_even : 
  ∃ N : ℕ, 
    (∀ (digits : Finset ℕ) (thousands hundreds tens units : ℕ), 
      digits = {1, 2, 3, 4, 5, 6} ∧ 
      ∀ n ∈ digits, (0 < n ∧ n < 10) ∧ 
      (thousands ∈ digits ∧ hundreds ∈ digits ∧ tens ∈ digits ∧ units ∈ digits) ∧ 
      (thousands ≠ hundreds ∧ thousands ≠ tens ∧ thousands ≠ units ∧ 
       hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units) ∧ 
      (tens + units) % 2 = 0 → N = 324) :=
sorry

end four_digit_numbers_sum_even_l94_94796


namespace exists_route_within_same_republic_l94_94884

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l94_94884


namespace area_of_triangle_ABC_sinA_value_l94_94477

noncomputable def cosC := 3 / 4
noncomputable def sinC := Real.sqrt (1 - cosC ^ 2)
noncomputable def a := 1
noncomputable def b := 2
noncomputable def c := Real.sqrt (a ^ 2 + b ^ 2 - 2 * a * b * cosC)
noncomputable def area := (1 / 2) * a * b * sinC
noncomputable def sinA := (a * sinC) / c

theorem area_of_triangle_ABC : area = Real.sqrt 7 / 4 :=
by sorry

theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
by sorry

end area_of_triangle_ABC_sinA_value_l94_94477


namespace repeating_block_length_of_11_over_13_l94_94610

-- Given definitions and conditions
def repeatingBlockLength (n d : ℕ) : ℕ :=
  let rec find_period (remainder modulo_count : ℕ) (seen_rem : Finset ℕ) : ℕ :=
    if seen_rem.contains remainder then
      modulo_count
    else
      find_period ((remainder * 10) % d) (modulo_count + 1) (seen_rem.insert remainder)
  find_period (n % d) 0 ∅

-- Problem statement
theorem repeating_block_length_of_11_over_13 : repeatingBlockLength 11 13 = 6 := by
  sorry

end repeating_block_length_of_11_over_13_l94_94610


namespace coeff_x2_in_expansion_l94_94496

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x2_in_expansion : 
  (1 + x) * (1 - 2 * x)^5.coeff 2 = 30 := by
  sorry

end coeff_x2_in_expansion_l94_94496


namespace lincoln_high_students_club_overlap_l94_94348

theorem lincoln_high_students_club_overlap (total_students : ℕ)
  (drama_club_students science_club_students both_or_either_club_students : ℕ)
  (h1 : total_students = 500)
  (h2 : drama_club_students = 150)
  (h3 : science_club_students = 200)
  (h4 : both_or_either_club_students = 300) :
  drama_club_students + science_club_students - both_or_either_club_students = 50 :=
by
  sorry

end lincoln_high_students_club_overlap_l94_94348


namespace sum_of_arithmetic_sequence_l94_94276

theorem sum_of_arithmetic_sequence :
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 240 := by {
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  sorry
}

end sum_of_arithmetic_sequence_l94_94276


namespace largest_lcm_value_l94_94273

open Nat

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_value_l94_94273


namespace spherical_coordinates_standard_form_l94_94492

theorem spherical_coordinates_standard_form :
  ∀ (ρ θ φ : ℝ), 
  ρ = 5 → θ = 3 * Real.pi / 5 → φ = 9 * Real.pi / 5 →
  ρ > 0 → (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ (0 ≤ φ ∧ φ ≤ Real.pi) →
  ∃ θ' φ',
  (θ' = 8 * Real.pi / 5 ∧ φ' = Real.pi / 5) :=
by
  intros ρ θ φ hρ hθ hφ ρ_pos h_theta_range h_phi_range
  use (8 * Real.pi / 5, Real.pi / 5)
  constructor 
  sorry

end spherical_coordinates_standard_form_l94_94492


namespace DIAMOND_paths_l94_94719

-- Define the grid as an abstract structure (simplified for this statement)
constant D : Type
constant I : Type
constant A : Type
constant M : Type
constant O : Type
constant N : Type

-- Define the function specifying the number of paths
/-- The number of paths spelling "DIAMOND" in the given grid is 64 -/
def number_of_paths_DIAMOND : Nat := 64

-- State the theorem we want to prove
theorem DIAMOND_paths : number_of_paths_DIAMOND = 64 := 
by 
  -- Proof goes here
  sorry

end DIAMOND_paths_l94_94719


namespace digital_sum_condition_l94_94790

def digital_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digital_sum_condition (M : ℕ) (hM : M > 0) :
  (∀ k : ℕ, 1 ≤ k → k ≤ M → digital_sum (M * k) = digital_sum M) ↔ 
  (∃ (l : ℕ), M = 10^l - 1) :=
begin
  sorry
end

end digital_sum_condition_l94_94790


namespace total_balloons_l94_94955

-- Define the conditions
def joan_balloons : ℕ := 9
def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2

-- The statement we want to prove
theorem total_balloons : joan_balloons + sally_balloons + jessica_balloons = 16 :=
by
  sorry

end total_balloons_l94_94955


namespace log_sqrt_sin_range_l94_94614

noncomputable def range_of_log_sqrt_sin (x : ℝ) : Set ℝ :=
  {y | 0 < x ∧ x < 90 ∧ ∃ a, a = log 10 (sqrt (sin x)) ∧ y = a }

theorem log_sqrt_sin_range : (range_of_log_sqrt_sin 0 < x ∧ x < 90) = Set.Icc (-(∞ : ℝ)) 0 := by
  sorry

end log_sqrt_sin_range_l94_94614


namespace polynomials_equal_if_integer_parts_equal_l94_94997

theorem polynomials_equal_if_integer_parts_equal (f g : ℚ[X])
  (hf : f.degree = 2)
  (hg : g.degree = 2)
  (h : ∀ x : ℝ, ⌊f.eval x⌋ = ⌊g.eval x⌋) : f = g :=
sorry

end polynomials_equal_if_integer_parts_equal_l94_94997


namespace ratio_of_x_to_y_l94_94615

theorem ratio_of_x_to_y (x y : ℤ) (h : (7 * x - 4 * y) * 9 = (20 * x - 3 * y) * 4) : x * 17 = y * -24 :=
by {
  sorry
}

end ratio_of_x_to_y_l94_94615


namespace range_of_a_for_domain_of_f_l94_94846

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sqrt (-5 / (a * x^2 + a * x - 3))

theorem range_of_a_for_domain_of_f :
  {a : ℝ | ∀ x : ℝ, a * x^2 + a * x - 3 < 0} = {a : ℝ | -12 < a ∧ a ≤ 0} :=
by
  sorry

end range_of_a_for_domain_of_f_l94_94846


namespace distance_to_hole_l94_94683

-- Define the variables from the problem
variables (distance_first_turn distance_second_turn beyond_hole total_distance hole_distance : ℝ)

-- Given conditions
def conditions : Prop :=
  distance_first_turn = 180 ∧
  distance_second_turn = distance_first_turn / 2 ∧
  beyond_hole = 20 ∧
  total_distance = distance_first_turn + distance_second_turn

-- The main statement we need to prove
theorem distance_to_hole : conditions →
  hole_distance = total_distance - beyond_hole → hole_distance = 250 :=
by
  sorry

end distance_to_hole_l94_94683


namespace sum_binom_equiv_l94_94277

theorem sum_binom_equiv (T : ℂ) (h : T = ∑ k in finset.range 50, (-1)^k * (nat.choose 100 (2*k))) : 
  T = 2^50 :=
sorry

end sum_binom_equiv_l94_94277


namespace exists_route_within_republic_l94_94923

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l94_94923


namespace catch_up_time_l94_94656

def A_departure_time : ℕ := 8 * 60 -- in minutes
def B_departure_time : ℕ := 6 * 60 -- in minutes
def relative_speed (v : ℕ) : ℕ := 5 * v / 4 -- (2.5v effective) converted to integer math
def initial_distance (v : ℕ) : ℕ := 2 * v * 2 -- 4v distance (B's 2 hours lead)

theorem catch_up_time (v : ℕ) :  A_departure_time + ((initial_distance v * 4) / (relative_speed v - v)) = 1080 :=
by
  sorry

end catch_up_time_l94_94656


namespace square_101_l94_94716

theorem square_101:
  (101 : ℕ)^2 = 10201 :=
by
  sorry

end square_101_l94_94716


namespace monotonicity_of_f_l94_94648

def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x - 3 / (a * x)

theorem monotonicity_of_f (a : ℝ) (x : ℝ) (hx : x > 0) : 
    (if a > 0 then (∀ y, 0 < y ∧ y < 3/a → f y a = increasing) ∧ 
            (∀ y, y > 3/a → f y a = decreasing)
     else if a < 0 then (∀ y, 0 < y ∧ y < -1/a → f y a = decreasing) ∧ 
            (∀ y, y > -1/a → f y a = increasing)
     else false
    ) sorry

end monotonicity_of_f_l94_94648


namespace division_problem_l94_94618

theorem division_problem :
  (0.25 / 0.005) / 0.1 = 500 := by
  sorry

end division_problem_l94_94618


namespace find_m_value_l94_94848

theorem find_m_value (m : ℝ) :
  let l := ∀ x y : ℝ, mx + y + 3m - sqrt 3 = 0,
      circle := ∀ x y : ℝ, x^2 + y^2 = 12,
      AB := 2 * sqrt 3
  in
  (∃ A B : ℝ × ℝ, l A.1 A.2 ∧ l B.1 B.2 ∧ circle A.1 A.2 ∧ circle B.1 B.2 ∧ AB = dist A B) →
  (m = - sqrt 3 / 3) :=
begin
  intros,
  sorry
end

end find_m_value_l94_94848


namespace binomial_factorial_binomial_theorem_binomial_sum_l94_94832

-- Define the recursive binomial coefficient
def binomial (n k : ℕ) : ℕ
  | n, 0 => 1
  | n, m + 1 => if n = m + 1 then 1 else binomial n m + binomial (n - 1) m

-- Statement 1: Prove the equality of binomial coefficient and factorial formula
theorem binomial_factorial (n k : ℕ) : binomial n k = n.factorial / (k.factorial * (n - k).factorial) :=
sorry

-- Statement 2: Prove the binomial theorem
theorem binomial_theorem (a b : ℝ) (n : ℕ) :
  (a + b) ^ n = ∑ k in Finset.range(n + 1), binomial n k * a ^ k * b ^ (n - k) :=
sorry

-- Statement 3: Calculate the given binomial sum
theorem binomial_sum (n : ℕ) :
  ∑ k in Finset.range(n + 1), binomial (2 * n) (2 * k) = 2 ^ (2 * n - 1) :=
sorry

end binomial_factorial_binomial_theorem_binomial_sum_l94_94832


namespace sum_of_roots_of_quadratic_l94_94620

theorem sum_of_roots_of_quadratic :
  ∀ {x : ℝ}, x^2 - 7*x + 10 = 0 → ∑ roots (x^2 - 7*x + 10) = 7 :=
sorry

end sum_of_roots_of_quadratic_l94_94620


namespace non_periodic_cos_add_cos_l94_94208

noncomputable def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f(x + T) = f x

theorem non_periodic_cos_add_cos (α a : ℝ) (h_irrational : irrational α) (h_positive : 0 < a) :
  ¬ ∃ T ≠ 0, is_periodic (λ x, Real.cos x + a * Real.cos (α * x)) T :=
sorry

end non_periodic_cos_add_cos_l94_94208


namespace bowling_ball_weight_l94_94785

-- Definitions based on given conditions
variable (k b : ℕ)

-- Condition 1: one kayak weighs 35 pounds
def kayak_weight : Prop := k = 35

-- Condition 2: four kayaks weigh the same as five bowling balls
def balance_equation : Prop := 4 * k = 5 * b

-- Goal: prove the weight of one bowling ball is 28 pounds
theorem bowling_ball_weight (hk : kayak_weight k) (hb : balance_equation k b) : b = 28 :=
by
  sorry

end bowling_ball_weight_l94_94785


namespace billy_restaurant_total_payment_l94_94688

noncomputable def cost_of_meal
  (adult_count child_count : ℕ)
  (adult_cost child_cost : ℕ) : ℕ :=
  adult_count * adult_cost + child_count * child_cost

noncomputable def cost_of_dessert
  (total_people : ℕ)
  (dessert_cost : ℕ) : ℕ :=
  total_people * dessert_cost

noncomputable def total_cost_before_discount
  (adult_count child_count : ℕ)
  (adult_cost child_cost dessert_cost : ℕ) : ℕ :=
  (cost_of_meal adult_count child_count adult_cost child_cost) +
  (cost_of_dessert (adult_count + child_count) dessert_cost)

noncomputable def discount_amount
  (total : ℕ)
  (discount_rate : ℝ) : ℝ :=
  total * discount_rate

noncomputable def total_amount_to_pay
  (total : ℕ)
  (discount : ℝ) : ℝ :=
  total - discount

theorem billy_restaurant_total_payment :
  total_amount_to_pay
  (total_cost_before_discount 2 5 7 3 2)
  (discount_amount (total_cost_before_discount 2 5 7 3 2) 0.15) = 36.55 := by
  sorry

end billy_restaurant_total_payment_l94_94688


namespace calculate_siding_cost_l94_94214

noncomputable def wall_area : ℕ := 8 * 6
noncomputable def roof_area_per_part : ℕ := 5 * 8
noncomputable def total_roof_area : ℕ := 2 * roof_area_per_part
noncomputable def total_area : ℕ := wall_area + total_roof_area
noncomputable def sheet_area : ℕ := 10 * 10
noncomputable def num_sheets : ℕ := ⌈(total_area : ℝ) / (sheet_area : ℝ)⌉.to_nat  -- Ceiling function to get the integer number of sheets
noncomputable def sheet_cost : ℕ := 30
noncomputable def total_cost : ℕ := num_sheets * sheet_cost

theorem calculate_siding_cost : total_cost = 60 := by
  sorry

end calculate_siding_cost_l94_94214


namespace exists_airline_route_within_same_republic_l94_94902

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l94_94902


namespace digits_of_expression_l94_94729

theorem digits_of_expression (a b : ℕ) : nat.log10 (8 ^ a * 3 ^ b) + 1 = 12 :=
by 
  -- Increase the a and b with the actual problem conditions
  have ha : a = 10 := sorry,
  have hb : b = 15 := sorry,
  rw [ha, hb],
  -- the problem converted into Lean proof context with conditions
  sorry

end digits_of_expression_l94_94729


namespace slope_l3_l94_94976

-- Define the coordinates of points A, B, and C
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (4, 2)

-- Define lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def l2 (y : ℝ) : Prop := y = 2

-- Define the area of triangle ABC
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (((B.1 - A.1) * (C.2 - A.2)) - ((C.1 - A.1) * (B.2 - A.2)))

-- Define the slope of a line passing through two points
def slope (P Q : ℝ × ℝ) : ℝ :=
  if P.1 = Q.1 then 0 -- Avoid division by zero
  else (Q.2 - P.2) / (Q.1 - P.1)

-- Given conditions to solve the problem
theorem slope_l3 :
  l1 3 0 → l2 2 → l1 0 2 →
  area_triangle A B C = 6 →
  slope A C = 2 :=
by
  intros hA hB hC areaABC
  sorry

#eval slope (3, 0) (4, 2) -- This should return 2 for verification

end slope_l3_l94_94976


namespace number_of_ways_to_turn_off_lamps_l94_94201

theorem number_of_ways_to_turn_off_lamps :
  (let n := 10 in  -- number of lamps
  let m := 4 in   -- number of lamps to turn off
  let eligible_spaces := 6 in -- number of eligible spaces to insert the turned-off lamps
  let ways_to_choose_spaces := (eligible_spaces.choose 3) in -- choosing 3 out of 6 spaces
  ways_to_choose_spaces = 20) := 
sorry

end number_of_ways_to_turn_off_lamps_l94_94201


namespace angle_EGD_108_degrees_l94_94170

theorem angle_EGD_108_degrees
  (D E F G : Type)
  [is_triangle DEF]
  (DE_eq_DF : DE = DF)
  (G_on_EF : G ∈ EF)
  (EG_eq_GF : EG = GF)
  (angle_EDF : angle EDF = 2 * x)
  (DG_bisect_EDF : bisects DG (angle EDF)) :
  angle EGD = 108 := 
sorry

end angle_EGD_108_degrees_l94_94170


namespace triangle_construction_number_of_solutions_l94_94724

/-- Given the following conditions for constructing a triangle:
1. \(AB = c\), \(CC_1 = h\), \(\angle BAC = \alpha\).
2. \(AB = c\), \(CC_1 = h\), \(\angle BCA = \gamma\).
3. \(AB = c\), \(AA_1 = h\), \(\angle BAC = \alpha\).
4. \(AB = c\), \(AA_1 = h\), \(\angle ABC = \beta\).
5. \(AB = c\), \(AA_1 = h\), \(\angle ACB = \gamma\).
Prove that the number of unique solutions for constructing the triangle \(ABC\) is 11. -/
theorem triangle_construction_number_of_solutions
  (c h α γ β : ℝ) : 
  number_of_solutions_for_triangle_construction c h α γ β = 11 := 
sorry

end triangle_construction_number_of_solutions_l94_94724


namespace probability_three_tails_one_head_l94_94463

theorem probability_three_tails_one_head :
  let outcome_probability := (1 / 2) ^ 4 in
  let combinations := Finset.card (Finset.filter (λ (s : Finset.Fin 4 × Bool), s.2).Finset (Finset.product (Finset.range 4) (Finset.singleton (false, true, true, true)))) in
  outcome_probability * combinations = 1 / 4 :=
sorry

end probability_three_tails_one_head_l94_94463


namespace inequality_for_positive_reals_l94_94206

theorem inequality_for_positive_reals 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a^3 + b^3 + a * b * c)) + (1 / (b^3 + c^3 + a * b * c)) + 
  (1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) := 
sorry

end inequality_for_positive_reals_l94_94206


namespace cos_angle_correct_l94_94855

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (2, 2)

noncomputable def cos_angle : ℝ :=
  let ab_sum := (a.1 + b.1, a.2 + b.2)
  let ab_diff := (a.1 - b.1, a.2 - b.2)
  let dot_product := ab_sum.1 * ab_diff.1 + ab_sum.2 * ab_diff.2
  let ab_sum_magnitude := real.sqrt (ab_sum.1 ^ 2 + ab_sum.2 ^ 2)
  let ab_diff_magnitude := real.sqrt (ab_diff.1 ^ 2 + ab_diff.2 ^ 2)
  dot_product / (ab_sum_magnitude * ab_diff_magnitude)

theorem cos_angle_correct :
  cos_angle = real.sqrt 17 / 17 :=
by
  sorry

end cos_angle_correct_l94_94855


namespace greatest_product_from_sum_2004_l94_94599

theorem greatest_product_from_sum_2004 : ∃ (x y : ℤ), x + y = 2004 ∧ x * y = 1004004 :=
by
  sorry

end greatest_product_from_sum_2004_l94_94599


namespace dog_total_distance_l94_94568

-- constant s represents distance from home to work
def s : ℝ := 6

-- total distances by Ivan and the dog based on given conditions
theorem dog_total_distance (s : ℝ) (h : s = 6) :  
  ∀ v_Ivan v_Dog : ℝ, (v_Dog = 2 * v_Ivan) → 
    let d_total_Ivan := s in
    let d_total_dog := 2 * d_total_Ivan in 
    d_total_dog = 12 :=
by 
  intros v_Ivan v_Dog hv_Dog
  have d_total_Ivan := s
  have d_total_dog := 2 * d_total_Ivan
  have hdog : d_total_dog = 12 := by 
    rw [←h, ←d_total_Ivan, ←d_total_dog]
    simp
  exact hdog


end dog_total_distance_l94_94568


namespace constant_term_binomial_expansion_l94_94236

theorem constant_term_binomial_expansion :
  let binomial_expansion := (sqrt[3] x + (1 / (2 * x))) ^ 8 in
  ∃ r : ℕ, 
    r = 2 ∧ 
    (choose 8 r) * (1 / 2) ^ r * x^(8 - 4 * r) / 3 = 7 :=
by {
  sorry
}

end constant_term_binomial_expansion_l94_94236


namespace find_B_value_l94_94665

theorem find_B_value (A C B : ℕ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 :=
by {
  -- Proof would go here
  sorry
}

end find_B_value_l94_94665


namespace gary_money_shortfall_l94_94069

/-- 
Given the initial amount of money Gary had and the costs of his purchases including the discount, 
prove that Gary is short by 23.75 dollars.
--/
theorem gary_money_shortfall : 
  let initial_amount := 73 in
  let cost_pet_snake := 55 in
  let cost_snake_food := 12 in
  let original_cost_habitat := 35 in
  let discount_rate := 0.15 in
  let total_spent := 
    let spent_on_snake_and_food := cost_pet_snake + cost_snake_food in
    let discount_amount := discount_rate * original_cost_habitat in
    let discounted_habitat_cost := original_cost_habitat - discount_amount in
    spent_on_snake_and_food + discounted_habitat_cost in
  (total_spent - initial_amount = 23.75) := 
by
  sorry

end gary_money_shortfall_l94_94069


namespace simplify_and_evaluate_at_3_l94_94557

noncomputable def expression (x : ℝ) : ℝ := 
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1))

theorem simplify_and_evaluate_at_3 : expression 3 = -5 := 
  sorry

end simplify_and_evaluate_at_3_l94_94557


namespace integer_k_condition_l94_94373

theorem integer_k_condition (k x : ℤ) :
  (∃ x : ℤ, sqrt (39 - 6 * sqrt 12) + sqrt (k * x * (k * x + sqrt 12) + 3) = 2 * k) →
  (k = 3 ∨ k = 6) :=
by
  sorry

end integer_k_condition_l94_94373


namespace find_rs_l94_94205

noncomputable def rs_value (r s : ℝ) (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15 / 8) : ℝ :=
  r * s

theorem find_rs (r s : ℝ) (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15 / 8) :
  rs_value r s h1 h2 = sqrt 17 / 4 :=
sorry

end find_rs_l94_94205


namespace age_of_golden_retriever_l94_94136

def golden_retriever (gain_per_year current_weight : ℕ) (age : ℕ) :=
  gain_per_year * age = current_weight

theorem age_of_golden_retriever :
  golden_retriever 11 88 8 :=
by
  unfold golden_retriever
  simp
  sorry

end age_of_golden_retriever_l94_94136


namespace find_number_of_girls_l94_94635

variable (B G : ℕ)

theorem find_number_of_girls
  (h1 : B = G / 2)
  (h2 : B + G = 90)
  : G = 60 :=
sorry

end find_number_of_girls_l94_94635


namespace sports_club_membership_l94_94631

theorem sports_club_membership
  (N B T Neither : ℕ)
  (hN : N = 28)
  (hB : B = 17)
  (hT : T = 19)
  (hNeither : Neither = 2) :
  (B + T - N + Neither = 10) :=
by
  rw [hN, hB, hT, hNeither]
  sorry

end sports_club_membership_l94_94631


namespace journey_distance_l94_94296

theorem journey_distance
  (total_time : ℝ)
  (speed1 speed2 : ℝ)
  (journey_time : total_time = 10)
  (speed1_val : speed1 = 21)
  (speed2_val : speed2 = 24) :
  ∃ D : ℝ, (D / 2 / speed1 + D / 2 / speed2 = total_time) ∧ D = 224 :=
by
  sorry

end journey_distance_l94_94296


namespace arc_length_of_f_l94_94693

noncomputable def f (x : ℝ) : ℝ := 2 - Real.exp x

theorem arc_length_of_f :
  ∫ x in Real.log (Real.sqrt 3)..Real.log (Real.sqrt 8), Real.sqrt (1 + (Real.exp x)^2) = 1 + 1/2 * Real.log (3 / 2) :=
by
  sorry

end arc_length_of_f_l94_94693


namespace A_on_y_axis_B_coordinates_AB_parallel_x_axis_a_value_l94_94942

theorem A_on_y_axis_B_coordinates (a : ℝ) :
  (a + 1 = 0) → (2 * a + 1 = -1) :=
begin
  sorry,
end

theorem AB_parallel_x_axis_a_value (a : ℝ) :
  (-3 = 2 * a + 1) → (a = -2) :=
begin
  sorry,
end

end A_on_y_axis_B_coordinates_AB_parallel_x_axis_a_value_l94_94942


namespace symmetry_props_of_cosine_function_l94_94054

noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ :=
  (real.pi / 8 + k * real.pi / 2, 0)

noncomputable def axis_of_symmetry (k : ℤ) : ℝ :=
  -real.pi / 8 + k * real.pi / 2

noncomputable def decreasing_interval (k : ℤ) : set ℝ :=
  set.Icc (-real.pi / 8 + k * real.pi) (3 * real.pi / 8 + k * real.pi)

noncomputable def smallest_positive_period : ℝ :=
  real.pi

theorem symmetry_props_of_cosine_function :
  ∀ k : ℤ,
    center_of_symmetry k = (real.pi / 8 + k * real.pi / 2, 0) ∧
    axis_of_symmetry k = -real.pi / 8 + k * real.pi / 2 ∧
    decreasing_interval k = set.Icc (-real.pi / 8 + k * real.pi) (3 * real.pi / 8 + k * real.pi) ∧
    smallest_positive_period = real.pi :=
by
  sorry

end symmetry_props_of_cosine_function_l94_94054


namespace find_rate_of_grapes_l94_94350

def rate_per_kg_of_grapes (G : ℝ) : Prop :=
  let cost_of_grapes := 8 * G
  let cost_of_mangoes := 10 * 55
  let total_paid := 1110
  cost_of_grapes + cost_of_mangoes = total_paid

theorem find_rate_of_grapes : rate_per_kg_of_grapes 70 :=
by
  unfold rate_per_kg_of_grapes
  sorry

end find_rate_of_grapes_l94_94350


namespace ratio_of_m1_m2_l94_94968

open Real

theorem ratio_of_m1_m2 :
  ∀ (m : ℝ) (p q : ℝ), p ≠ 0 ∧ q ≠ 0 ∧ m ≠ 0 ∧
    (p + q = -((3 - 2 * m) / m)) ∧ 
    (p * q = 4 / m) ∧ 
    (p / q + q / p = 2) → 
   ∃ (m1 m2 : ℝ), 
    (4 * m1^2 - 28 * m1 + 9 = 0) ∧
    (4 * m2^2 - 28 * m2 + 9 = 0) ∧ 
    (m1 ≠ m2) ∧ 
    (m1 + m2 = 7) ∧ 
    (m1 * m2 = 9 / 4) ∧ 
    (m1 / m2 + m2 / m1 = 178 / 9) :=
by sorry

end ratio_of_m1_m2_l94_94968


namespace Patricia_money_l94_94354

theorem Patricia_money 
(P L C : ℝ)
(h1 : L = 5 * P)
(h2 : L = 2 * C)
(h3 : P + L + C = 51) :
P = 6.8 := 
by 
  sorry

end Patricia_money_l94_94354


namespace least_number_of_coins_l94_94622

theorem least_number_of_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 5 = 4) ∧ (∀ m : ℕ, (m % 7 = 3) ∧ (m % 5 = 4) → n ≤ m) → n = 24 :=
by
  sorry

end least_number_of_coins_l94_94622


namespace find_value_of_g1_l94_94416

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g x

theorem find_value_of_g1 (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2)
  (h4 : f 1 + g (-1) = 4) : 
  g 1 = 3 :=
sorry

end find_value_of_g1_l94_94416


namespace clinton_shoes_count_l94_94711

def num_hats : ℕ := 5
def num_belts : ℕ := num_hats + 2
def num_shoes : ℕ := 2 * num_belts

theorem clinton_shoes_count : num_shoes = 14 := by
  -- proof goes here
  sorry

end clinton_shoes_count_l94_94711


namespace finalSalary_l94_94680

noncomputable def initialSalary : ℝ := 5000
noncomputable def increaseBy30Percent (salary : ℝ) : ℝ := salary * 1.30
noncomputable def apply7PercentTax (salary : ℝ) : ℝ := salary * 0.93
noncomputable def decreaseBy20Percent (salary : ℝ) : ℝ := salary * 0.80
noncomputable def deduct100dollar (salary : ℝ) : ℝ := salary - 100
noncomputable def increaseBy10Percent (salary : ℝ) : ℝ := salary * 1.10
noncomputable def apply10PercentTax (salary : ℝ) : ℝ := salary * 0.90
noncomputable def reduceBy25Percent (salary : ℝ) : ℝ := salary * 0.75

theorem finalSalary (initialSalary : ℝ) 
                    (increaseBy30Percent : ℝ -> ℝ)
                    (apply7PercentTax : ℝ -> ℝ)
                    (decreaseBy20Percent : ℝ -> ℝ)
                    (deduct100dollar : ℝ -> ℝ)
                    (increaseBy10Percent : ℝ -> ℝ)
                    (apply10PercentTax : ℝ -> ℝ)
                    (reduceBy25Percent : ℝ -> ℝ) : 
                    (reduceBy25Percent 
                      (apply10PercentTax 
                        (increaseBy10Percent 
                          (deduct100dollar 
                            (decreaseBy20Percent 
                              (apply7PercentTax 
                                (increaseBy30Percent initialSalary))))))
                    = 3516.48 := 
by
  sorry

end finalSalary_l94_94680


namespace sum_of_n_satisfying_lcm_gcd_condition_l94_94732

theorem sum_of_n_satisfying_lcm_gcd_condition :
  (∑ n in {n : ℕ | 0 < n ∧ Nat.lcm n 120 = Nat.gcd n 120 + 300}, n) = 180 :=
by
  sorry

end sum_of_n_satisfying_lcm_gcd_condition_l94_94732


namespace airline_route_within_republic_l94_94906

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l94_94906


namespace cheenu_time_difference_l94_94026

theorem cheenu_time_difference :
  let time_per_mile_cycle := 120.0 / 18.0 in
  let time_per_mile_jog := 300.0 / 15.0 in
  (time_per_mile_jog - time_per_mile_cycle) = 13.3 :=
by
  sorry

end cheenu_time_difference_l94_94026


namespace range_of_4a_minus_2b_l94_94072

theorem range_of_4a_minus_2b (a b : ℝ) (h1 : 0 ≤ a - b) (h2 : a - b ≤ 1) (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) :
  2 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 7 := 
sorry

end range_of_4a_minus_2b_l94_94072


namespace max_S_n_l94_94088

noncomputable def S (n : ℕ) : ℝ := sorry  -- Definition of the sum of the first n terms

theorem max_S_n (S : ℕ → ℝ) (h16 : S 16 > 0) (h17 : S 17 < 0) : ∃ n, S n = S 8 :=
sorry

end max_S_n_l94_94088


namespace AC_solutions_l94_94950

noncomputable def find_AC (a : ℝ) : Set ℝ :=
  if a < sqrt(3) / 2 then ∅
  else if a = sqrt(3) / 2 then {1 / 2}
  else if a < 1 then { (1 + sqrt (4 * a^2 - 3)) / 2, (1 - sqrt (4 * a^2 - 3)) / 2 }
  else { (1 + sqrt (4 * a^2 - 3)) / 2 }

theorem AC_solutions (a : ℝ) : Set ℝ :=
find_AC a

end AC_solutions_l94_94950


namespace periodicity_of_f_l94_94425

variable {ℝ : Type*} [RealField ℝ]

def periodicity_problem (f : ℝ → ℝ) :=
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y) →
  (∀ x : ℝ, f (x + 2 * Real.pi) = f x)

theorem periodicity_of_f (f : ℝ → ℝ) : periodicity_problem f :=
  sorry

end periodicity_of_f_l94_94425


namespace find_angle_C_find_cos_l94_94476

open Real

-- Definitions for the conditions
def triangle_sides (a b c: ℝ) : Prop := 
  -- Add any relevant conditions for a, b, c to be sides of a triangle if needed
  True

def condition1 (a b c : ℝ) := 
  2 * c * cos (b : ℝ) = 2 * a + b

def condition2 (f : ℝ → ℝ) (alpha : ℝ) :=
  f (α / 2) = 6 / 5

noncomputable def f (x : ℝ) (m : ℝ) := 2 * sin (2 * x + π / 6) + m * cos (2 * x)

-- Proof problems
theorem find_angle_C (a b c : ℝ) (h : condition1 a b c) : 
  ∃ C : ℝ, C = 2 * π / 3 :=
sorry

theorem find_cos (m alpha C : ℝ) (hc : C = 2 * π / 3) (hf_sym : ∀ x : ℝ, f(x, m) = f(C / 2 - x, m)) (hf_alpha : condition2 (λ x, f(x, m)) alpha) : 
  cos(2 * alpha + C) = -7 / 25 :=
sorry

end find_angle_C_find_cos_l94_94476


namespace shopkeeper_profit_l94_94673

/-- A shopkeeper sells goods at cost price using a faulty meter that weighs 940 grams instead of 1000 grams. -/
theorem shopkeeper_profit (cost_per_kg : ℕ → ℝ) :
  let actual_weight := 1000
      faulty_weight := 940
      profit_weight := actual_weight - faulty_weight in
  cost_per_kg actual_weight = 1 →
  ((cost_per_kg profit_weight) / (cost_per_kg faulty_weight)) * 100 = 6.38 :=
by
  intros
  let actual_weight := 1000
  let faulty_weight := 940
  let profit_weight := actual_weight - faulty_weight
  rw [← cost_per_kg_eq_zero, cost_per_kg]
  sorry

end shopkeeper_profit_l94_94673


namespace determine_weights_l94_94071

def Ball := { name : String := "", weight : ℕ := 0 }

variables (W_1 W_2 G_1 G_2 R_1 R_2 : Ball)

-- Conditions
axiom cond1 : W_1.name = "White 1" ∧ W_2.name = "White 2" ∧ G_1.name = "Green 1" ∧ G_2.name = "Green 2" ∧ R_1.name = "Red 1" ∧ R_2.name = "Red 2"
axiom cond2 : (W_1.weight = 99 ∨ W_1.weight = 101) ∧ (W_2.weight = 99 ∨ W_2.weight = 101) ∧ (G_1.weight = 99 ∨ G_1.weight = 101) ∧ (G_2.weight = 99 ∨ G_2.weight = 101) ∧ (R_1.weight = 99 ∨ R_1.weight = 101) ∧ (R_2.weight = 99 ∨ R_2.weight = 101)
axiom cond3 : (∃ (b : Ball), b.weight = 99 ∧ b.name.contains "White") ∧ (∃ (b : Ball), b.weight = 99 ∧ b.name.contains "Green") ∧ (∃ (b : Ball), b.weight = 99 ∧ b.name.contains "Red")
axiom cond4 : (∃ (b : Ball), b.weight = 101 ∧ b.name.contains "White") ∧ (∃ (b : Ball), b.weight = 101 ∧ b.name.contains "Green") ∧ (∃ (b : Ball), b.weight = 101 ∧ b.name.contains "Red")

theorem determine_weights : 
  W_1.weight = 101 ∧ W_2.weight = 99 ∧ G_1.weight = 99 ∧ G_2.weight = 101 ∧ R_1.weight = 101 ∧ R_2.weight = 99 :=
  sorry

end determine_weights_l94_94071


namespace parabola_hyperbola_focus_l94_94146

theorem parabola_hyperbola_focus {p : ℝ} :
  let focus_parabola := (p / 2, 0)
  let focus_hyperbola := (2, 0)
  focus_parabola = focus_hyperbola -> p = 4 :=
by
  intro h
  sorry

end parabola_hyperbola_focus_l94_94146


namespace factorial_ratio_integer_l94_94548

theorem factorial_ratio_integer (m n : ℕ) : 
    (m ≥ 0) → (n ≥ 0) → ∃ k : ℤ, k = (2 * m).factorial * (2 * n).factorial / ((m.factorial * n.factorial * (m + n).factorial) : ℝ) :=
by
  sorry

end factorial_ratio_integer_l94_94548


namespace initial_pencils_sold_l94_94691

theorem initial_pencils_sold (C : ℝ) (N : ℕ) (H1 : N * 0.65 * C = 1) (H2 : 10 * 1.30 * C = 1) : N = 20 := by
  have hC : C = 1 / 13 := by
    have h : 13 * C = 1 := by
      linarith
    linarith
  rw [hC] at H1
  have hN : N * (0.65 / 13) = 1 := by
    have h : 0.65 * (1 / 13) = 0.65 / 13 := by
      field_simp
    rw [← h] at H1
    exact H1
  have hN_eq_20 : N * (1 / 20) = 1 := by
    have h : 0.65 / 13 = 1 / 20 := by
      norm_num
    rw [h] at hN
    exact hN
  linarith

end initial_pencils_sold_l94_94691


namespace exist_six_subsets_of_six_elements_l94_94524

theorem exist_six_subsets_of_six_elements (n m : ℕ) (X : Finset ℕ) (A : Fin m → Finset ℕ) :
    n > 6 →
    X.card = n →
    (∀ i, (A i).card = 5 ∧ (A i ⊆ X)) →
    m > (n * (n-1) * (n-2) * (n-3) * (4*n-15)) / 600 →
    ∃ i1 i2 i3 i4 i5 i6 : Fin m,
      i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧
      (A i1 ∪ A i2 ∪ A i3 ∪ A i4 ∪ A i5 ∪ A i6).card = 6 := 
sorry

end exist_six_subsets_of_six_elements_l94_94524


namespace gcf_7_8_fact_l94_94744

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l94_94744


namespace gcf_7_factorial_8_factorial_l94_94765

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l94_94765


namespace tangent_slope_at_neg5_l94_94423

-- Definitions for the given conditions
variable {f : ℝ → ℝ}

-- Conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def differentiable_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x, DifferentiableAt ℝ f x

def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f (x - 2)

-- Lean statement encapsulating the proof
theorem tangent_slope_at_neg5 
  (h_even : even_function f)
  (h_diff : differentiable_on_reals f)
  (h_slope : f' 1 = 1)
  (h_periodic : periodic_4 f) :
  (f' (-5) = -1) :=
sorry


end tangent_slope_at_neg5_l94_94423


namespace probability_three_tails_one_head_l94_94457

noncomputable def probability_of_three_tails_one_head : ℚ :=
  if H : 1/2 ∈ ℚ then 4 * ((1 / 2)^4 : ℚ)
  else 0

theorem probability_three_tails_one_head :
  probability_of_three_tails_one_head = 1 / 4 :=
by {
  have h : (1 / 2 : ℚ) ∈ ℚ := by norm_cast; norm_num,
  rw probability_of_three_tails_one_head,
  split_ifs,
  { field_simp [h],
    norm_cast,
    norm_num }
}

end probability_three_tails_one_head_l94_94457


namespace additional_flour_cost_l94_94038

def cost_per_pound := 1.50
def discount_rate := 0.10
def pounds_to_buy := 4
def cost_without_discount := pounds_to_buy * cost_per_pound
def discount := cost_without_discount * discount_rate
def final_cost := cost_without_discount - discount

theorem additional_flour_cost : final_cost = 5.40 := by
  sorry

end additional_flour_cost_l94_94038


namespace zero_count_between_decimal_and_first_non_zero_digit_l94_94283

theorem zero_count_between_decimal_and_first_non_zero_digit :
  let frac := (5 : ℚ) / 1600
  let dec_form := 3125 / 10^6
  (frac = dec_form) → 
  ∃ n, (frac * 10^6 = n) ∧ (3 = String.length (n.to_digits 10).takeWhile (λ c, c = '0')) :=
by
  intros frac dec_form h
  sorry

end zero_count_between_decimal_and_first_non_zero_digit_l94_94283


namespace max_value_f_on_interval_l94_94573

def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem max_value_f_on_interval : ∃ x ∈ set.Icc ( -2:ℝ ) ( 4:ℝ ) , f x = 16 := by
  sorry

end max_value_f_on_interval_l94_94573


namespace num_correct_statements_l94_94801

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ q : ℚ, x = q

theorem num_correct_statements (α β : ℝ) (h_irr_α : is_irrational α) (h_irr_β : is_irrational β) (h_distinct : α ≠ β) :
  ((¬ is_irrational (α * β + α - β)) ∧ 
   (¬ is_irrational ((α - β) / (α + β))) ∧ 
   (is_irrational (Real.sqrt α + Real.cbrt β))) → 
  (1 = 1) := 
sorry

end num_correct_statements_l94_94801


namespace moles_CO2_formed_l94_94057

-- Define the conditions based on the problem statement
def moles_HCl := 1
def moles_NaHCO3 := 1

-- Define the reaction equation in equivalence terms
def chemical_equation (hcl : Nat) (nahco3 : Nat) : Nat :=
  if hcl = 1 ∧ nahco3 = 1 then 1 else 0

-- State the proof problem
theorem moles_CO2_formed : chemical_equation moles_HCl moles_NaHCO3 = 1 :=
by
  -- The proof goes here
  sorry

end moles_CO2_formed_l94_94057


namespace probability_three_tails_one_head_l94_94465

theorem probability_three_tails_one_head :
  (nat.choose 4 1) * (1/2)^4 = 1/4 :=
by
  sorry

end probability_three_tails_one_head_l94_94465


namespace matrix_not_invertible_value_l94_94793

theorem matrix_not_invertible_value (a b c : ℝ)
  (h : det (matrix (ℤ × ℤ → ℝ)
    ![![a^2, b^2, c^2],
      ![b^2, c^2, a^2],
      ![c^2, a^2, b^2]]) = 0) :
  (a = b) ∧ (b = c) → 
  (∃ q : ℝ, q = (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2))) → q = 3 / 2 := 
by
  sorry

end matrix_not_invertible_value_l94_94793


namespace number_of_elements_is_one_l94_94246

/-- 
Prove that the number of elements in the set 
{ (x, y) | log (x^3 + 1/3 * y^3 + 1/9) = log x + log y } is 1. 
-/
theorem number_of_elements_is_one :
  { p : (ℝ × ℝ) | (Real.log (p.1^3 + 1/3 * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2) }.card = 1 :=
sorry

end number_of_elements_is_one_l94_94246


namespace valid_placements_count_l94_94134

-- Define the problem setup
def grid := (Fin 4) × (Fin 4)
def color := {r, b, g} -- red, blue, green

def no_adjacent (f : grid → color) : Prop :=
  ∀ x y, 
    (dist x y = 1 ∧ f x = f y) →
    false

def no_uninterrupted_line (f : grid → color) : Prop :=
  ∀ (c : color) (row col : Fin 4),
    (∀ (i : Fin 4), f (row, i) = c ∨ f (i, col) = c ∨ f (i, i) = c ∨ f (i, 3 - i) = c) →
    false

def placement_valid (f : grid → color) : Prop :=
  no_adjacent f ∧ no_uninterrupted_line f

-- Prove there are 24 valid placements
theorem valid_placements_count : 
  ∃ (count : Nat), count = 24 ∧ (
    (∃ (f : grid → color), placement_valid f) ->
    count = 24
  ) := 
sorry

end valid_placements_count_l94_94134


namespace park_length_l94_94335

noncomputable def length_of_park (L : ℝ) : Prop :=
  let width := 40 
  let road_width_each := 3 
  let lawn_area := 2109
  let total_width_of_roads := 2 * road_width_each
  let total_area_of_park := L * width
  let area_of_roads := L * total_width_of_roads
  let area_of_lawn := total_area_of_park - area_of_roads
  area_of_lawn = lawn_area

theorem park_length : ∃ L : ℝ, length_of_park L ∧ L = 62 :=
by {
  use 62,
  unfold length_of_park,
  simp,
  sorry
}

end park_length_l94_94335


namespace exists_route_within_same_republic_l94_94890

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l94_94890


namespace x_squared_plus_y_squared_l94_94446

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 :=
sorry

end x_squared_plus_y_squared_l94_94446


namespace no_zero_points_l94_94433

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x
noncomputable def g (x a : ℝ) : ℝ := (f x a) - x + Real.exp (x - 1)

theorem no_zero_points (a : ℝ) (h : 0 ≤ a ∧ a ≤ Real.exp 1) : 
  ¬ ∃ x : ℝ, g x a = 0 :=
by
  sorry

end no_zero_points_l94_94433


namespace max_product_of_two_integers_with_sum_2004_l94_94604

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l94_94604


namespace general_formula_b_S3_values_l94_94808

variable (a : ℕ → ℤ)
variable (b : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (T : ℕ → ℤ)
variable d : ℤ
variable q : ℤ

-- Initial conditions
axiom h1 : a 1 = -1
axiom h2 : b 1 = 1
axiom h3 : a 2 + b 2 = 2
axiom h4 : a 3 + b 3 = 5
axiom h5 : T 3 = 21

-- Define arithmetic sequence assumption 
axiom h_arith : ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence assumption
axiom h_geom : ∀ n : ℕ, b (n + 1) = b n * q

-- Define sum of first n terms
axiom h_sum_arith : ∀ n : ℕ, S n = (n * (2 * a 1 + (n - 1) * d)) / 2
axiom h_sum_geom : ∀ n : ℕ, T n = b 1 * (q ^ n - 1) / (q - 1)

-- Proving the general formula for {b_n} is b_n = 2^(n-1)
theorem general_formula_b : b n = 2 ^ (n - 1) :=
sorry

-- Given T_3 = 21, prove that S_3 = -6 or S_3 = 21.
theorem S3_values : S 3 = -6 ∨ S 3 = 21 :=
sorry

end general_formula_b_S3_values_l94_94808


namespace gcf_7_factorial_8_factorial_l94_94770

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l94_94770


namespace envelopes_left_l94_94704

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end envelopes_left_l94_94704


namespace max_intersections_three_lines_one_circle_l94_94588

/-- There are three distinct lines and one circle on a plane. 
The largest possible number of points of intersection is 9. -/
theorem max_intersections_three_lines_one_circle : 
  ∃ (lines : set (set ℝ^2)) (c : set ℝ^2), 
    lines.card = 3 ∧ ∀ l ∈ lines, ∃ p₁ p₂ ∈ c, (p₁ ∈ l ∧ p₂ ∈ l) ∧ ∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → (l₁ ∩ l₂).card ≤ 1 ∧ 
    ((lines.card * 2) + (finset.card (finset.filter (λ (s : set ℝ^2 × set ℝ^2), s.1 ∩ s.2 ≠ ∅) (lines.ssubsets 2)))) = 9 :=
sorry

end max_intersections_three_lines_one_circle_l94_94588


namespace label_cube_faces_l94_94860

/-- 
Proof problem: 
There are 30 unique ways (up to rotation) to label the faces of a cube with the numbers {1, 2, 3, 4, 5, 6}. 
-/
theorem label_cube_faces : 
  ∃ (n : ℕ), n = 30 ∧ (∀ (labelling : (ℕ → ℕ)), 
  (∀ i, 1 ≤ labelling i ∧ labelling i ≤ 6) ∧ 
  (∀ i j, i ≠ j → labelling i ≠ labelling j) 
  → n = 30) :=
begin
  use 30,
  split,
  { refl, },
  { intros labelling h,
    sorry
  }
end

end label_cube_faces_l94_94860


namespace find_f2012_value_l94_94108

noncomputable def f : ℝ → ℝ
| x => if x % 3 ∈ (0, 3) then (Real.logBase 2 (3 * (x % 3) + 1)) else 0

theorem find_f2012_value :
  (∀ x : ℝ, f (x + 3) = f x) ∧ 
  (∀ x : ℝ, 0 < x ∧ x < 3 → f x = Real.logBase 2 (3 * x + 1)) → 
  f 2012 = Real.logBase 2 7 :=
by
  sorry

end find_f2012_value_l94_94108


namespace simplest_quadratic_radical_l94_94291

-- Definitions for each option as conditions
def optionA (a : ℝ) : ℝ := Real.sqrt (a^2 + 1)
def optionB : ℝ := Real.sqrt 8
def optionC : ℝ := 1 / Real.sqrt 3
def optionD : ℝ := Real.sqrt 0.5

-- The main theorem stating optionA is the simplest quadratic radical
theorem simplest_quadratic_radical (a : ℝ) : 
  optionA a = Real.sqrt (a^2 + 1) ∧ 
  optionB = 2 * Real.sqrt 2 ∧ 
  optionC = Real.sqrt 3 / 3 ∧ 
  optionD = Real.sqrt 2 / 2 →
  (∀ x : ℝ, x = Real.sqrt (a^2 + 1) ∧ x ≠ 2 * Real.sqrt 2 ∧ x ≠ Real.sqrt 3 / 3 ∧ x ≠ Real.sqrt 2 / 2 ) := 
by sorry

end simplest_quadratic_radical_l94_94291


namespace find_matrix_C_l94_94090

/-- Define the given matrices A and B. --/
def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, 1], ![1, 3]]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![1, -1]]

/-- State the equivalent proof problem: finding matrix C such that AC = B --/
theorem find_matrix_C :
  ∃ C : Matrix (Fin 2) (Fin 2) ℚ, A ⬝ C = B :=
begin
  -- Using inverse matrix and the condition that A and B are given,
  -- we know that C = A⁻¹ ⬝ B
  let C := ![![3/5, 4/5], ![-1/5, -3/5]],
  use C,
  sorry -- The proof steps should go here, but we are skipping them as per the instructions.
end

end find_matrix_C_l94_94090


namespace exists_route_same_republic_l94_94894

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l94_94894


namespace volume_of_common_part_of_cylinders_l94_94256

/-- The volume of the common part of two right circular cylinders intersecting at a right angle,
each with a diameter of 2 cm, is 16/3 cm³. -/
theorem volume_of_common_part_of_cylinders :
  ∀ (d : ℝ), (d = 2) → 
  let r := d / 2 in
  let volume := (16 * r ^ 3) / 3 in
  volume = 16 / 3 :=
by {
  intros d h,
  let r := d / 2,
  have hr : r = 1 := by {
    rw [h, div_self],
    linarith,
  },
  rw [hr],
  let volume := (16 * r ^ 3) / 3,
  have hvol : volume = (16 * 1 ^ 3) / 3 := by {
    congr,
    exact hr,
  },
  rw [pow_one, one_pow, one_mul] at hvol,
  exact hvol,
  sorry
}

end volume_of_common_part_of_cylinders_l94_94256


namespace average_less_than_or_equal_six_l94_94742

def odd_numbers (n : ℕ) : Prop := n % 2 = 1
def within_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10
def less_than_or_equal_six (n : ℕ) : Prop := n ≤ 6

theorem average_less_than_or_equal_six :
  let numbers := {n | odd_numbers n ∧ within_range n ∧ less_than_or_equal_six n},
      num_list := [1, 3, 5],
      sum_numbers := num_list.sum,
      count_numbers := num_list.length in
  sum_numbers / count_numbers = 3 :=
by {
  sorry
}

end average_less_than_or_equal_six_l94_94742


namespace find_AC_find_angle_A_l94_94153

noncomputable def triangle_AC (AB BC : ℝ) (sinC_over_sinB : ℝ) : ℝ :=
  if h : sinC_over_sinB = 3 / 5 ∧ AB = 3 ∧ BC = 7 then 5 else 0

noncomputable def triangle_angle_A (AB AC BC : ℝ) : ℝ :=
  if h : AB = 3 ∧ AC = 5 ∧ BC = 7 then 120 else 0

theorem find_AC (BC AB : ℝ) (sinC_over_sinB : ℝ) (h : BC = 7 ∧ AB = 3 ∧ sinC_over_sinB = 3 / 5) : 
  triangle_AC AB BC sinC_over_sinB = 5 := by
  sorry

theorem find_angle_A (BC AB AC : ℝ) (h : BC = 7 ∧ AB = 3 ∧ AC = 5) : 
  triangle_angle_A AB AC BC = 120 := by
  sorry

end find_AC_find_angle_A_l94_94153


namespace decagon_quadrilateral_area_l94_94356

noncomputable def area_of_ABDN (A B D N E G : Point) (length : ℝ) : ℝ :=
  12

theorem decagon_quadrilateral_area :
  ∀ A B C D E F G H I J N: Point,
  (∀ P Q : Point, (distance P Q = 4)) →
  (∀ P Q R : Point, (∠ P Q R = 90) → P = A ∧ Q = B ∧ R = C ∨
                                     P = B ∧ Q = C ∧ R = D ∨
                                     P = C ∧ Q = D ∧ R = E ∨
                                     P = D ∧ Q = E ∧ R = F ∨
                                     P = E ∧ Q = F ∧ R = G ∨
                                     P = F ∧ Q = G ∧ R = H ∨
                                     P = G ∧ Q = H ∧ R = I ∨
                                     P = H ∧ Q = I ∧ R = J ∨
                                     P = I ∧ Q = J ∧ R = A ∨
                                     P = J ∧ Q = A ∧ R = B) →
  (∃ N : Point, line_through A E ∧ line_through C G) →
  area_of_ABDN A B D N E G 4 = 12 :=
by
  sorry

end decagon_quadrilateral_area_l94_94356


namespace f_g_of_neg2_l94_94448

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x - 1)^2

theorem f_g_of_neg2 : f (g (-2)) = 29 := by
  -- We need to show f(g(-2)) = 29 given the definitions of f and g
  sorry

end f_g_of_neg2_l94_94448


namespace tricycles_count_l94_94294

theorem tricycles_count (B T : ℕ) (hB : B = 50) (hW : 2 * B + 3 * T = 160) : T = 20 :=
by
  sorry

end tricycles_count_l94_94294


namespace hyperbola_standard_eq_l94_94109

-- Define the conditions of the problem for the existence of the hyperbola
def hyperbola_condition (x y : ℝ) : Prop :=
  (y = sqrt 3 * x ∨ y = -sqrt 3 * x) ∧ (x = 2 ∧ y = 3)

-- Define the target standard equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 3 = 1

-- Formal statement to prove that the given conditions imply the target equation
theorem hyperbola_standard_eq :
  ∃ (x y : ℝ), hyperbola_condition x y → hyperbola_eq x y :=
by
  sorry

end hyperbola_standard_eq_l94_94109


namespace line_MI_passes_through_midpoint_arc_l94_94935

-- Definitions based on the problem statement
variables {A B C D I Q P M: Type*}
variables [Incenter I T so that they can be used to define I]
variable scalene_triangle : Triangle ABC
variable circumcircle : Circle ABC
variable tangent_point : tangency_point circumcircle C D
variable midpoint_arc : midpoint (arc circumcircle ACB)

-- Main statement
theorem line_MI_passes_through_midpoint_arc :
  (scalene_triangle ABC) ∧
  (tangent_point circumcircle C D intersects AB D) ∧
  (Incenter I ABC) ∧
  (segment AI intersects_bisector_angle_CDB Q) ∧
  (segment BI intersects_bisector_angle_CDB P) ∧
  (midpoint PQ M) →
  (line MI passes_through midpoint_arc) :=
sorry

end line_MI_passes_through_midpoint_arc_l94_94935


namespace volume_of_regular_triangular_prism_l94_94105

noncomputable def inradius : ℝ := 2
noncomputable def sphere_radius : ℝ := 2
noncomputable def prism_height := 2 * sphere_radius
noncomputable def side_length := 4 * real.sqrt 3
noncomputable def volume_of_prism := 1 / 2 * side_length * side_length * real.sin (real.pi / 3) * prism_height

theorem volume_of_regular_triangular_prism 
  (inradius : ℝ := 2) 
  (sphere_radius : ℝ := 2)
  (prism_height := 2 * sphere_radius)
  (side_length := 4 * real.sqrt 3)
  (volume_of_prism := 1 / 2 * side_length * side_length * real.sin (real.pi / 3) * prism_height) :
  volume_of_prism = 48 * real.sqrt 3 :=
sorry

end volume_of_regular_triangular_prism_l94_94105


namespace equal_distances_l94_94180

def point := ℝ × ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def A : point := (10, 0, 0)
def B : point := (0, -6, 0)
def C : point := (0, 0, 8)
def D : point := (0, 0, 0)
def P : point := (5, -3, 4)

theorem equal_distances :
  dist P A = dist P B ∧
  dist P B = dist P C ∧
  dist P C = dist P D :=
by {
  sorry
}

end equal_distances_l94_94180


namespace largest_value_of_3k_l94_94773

theorem largest_value_of_3k (k x : ℝ) (h : sqrt (x^2 - k) + 2 * sqrt (x^3 - 1) = x) : 3 * k ≤ 4 :=
by
  sorry

end largest_value_of_3k_l94_94773


namespace Cindy_envelopes_left_l94_94706

theorem Cindy_envelopes_left :
  ∀ (initial_envelopes envelopes_per_friend friends : ℕ), 
    initial_envelopes = 37 →
    envelopes_per_friend = 3 →
    friends = 5 →
    initial_envelopes - envelopes_per_friend * friends = 22 :=
by
  intros initial_envelopes envelopes_per_friend friends h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Cindy_envelopes_left_l94_94706


namespace area_of_parallelogram_normal_vector_find_perpendicular_vector_l94_94440

open Real EuclideanSpace

def point := EuclideanSpace ℝ (Fin 3)

def A : point := ![0, 2, 3]
def B : point := ![-2, 1, 6]
def C : point := ![1, -1, 5]

def AB := B - A
def AC := C - A

def cross_product (u v : point) : point :=
  ![
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0]
  ]

def vector_length (v : point) : ℝ :=
  sqrt (v[0] ^ 2 + v[1] ^ 2 + v[2] ^ 2)

theorem area_of_parallelogram :
  vector_length (cross_product AB AC) = 7 * sqrt 3 := sorry

theorem normal_vector :
  cross_product AB AC = ![11, 1, -5] := sorry

def perpendicular_vector (a : point) :=
  AB ⬝ a = 0 ∧ AC ⬝ a = 0 ∧ vector_length a = sqrt 3

theorem find_perpendicular_vector (a : point) :
  perpendicular_vector a → a = (1 / sqrt 147) • ![11, 1, -5] := sorry

end area_of_parallelogram_normal_vector_find_perpendicular_vector_l94_94440


namespace probability_three_tails_one_head_l94_94467

theorem probability_three_tails_one_head :
  (nat.choose 4 1) * (1/2)^4 = 1/4 :=
by
  sorry

end probability_three_tails_one_head_l94_94467


namespace count_superbrazilian_4_digit_is_822_l94_94672

-- Define what it means for a number to be brazilian
def is_brazilian (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.head = digits.reverse.head

-- Define what it means for a number to be superbrazilian
def is_superbrazilian (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_brazilian a ∧ is_brazilian b ∧ n = a + b

-- Define the set of 4-digit numbers
def four_digit_numbers : Finset ℕ :=
  Finset.filter (λ n, 1000 ≤ n ∧ n ≤ 9999) (Finset.range 10000)

-- Count the number of 4-digit superbrazilian numbers
def count_superbrazilian_4_digit : ℕ :=
  Finset.card (Finset.filter is_superbrazilian four_digit_numbers)

-- The theorem to prove
theorem count_superbrazilian_4_digit_is_822 : count_superbrazilian_4_digit = 822 := by
  sorry

end count_superbrazilian_4_digit_is_822_l94_94672


namespace solution_validity_l94_94065

noncomputable def proof_problem (x : ℝ) : Prop :=
  x ≥ (-1 / 2) ∧ x ≠ 0 ∧ (4 * x^2) / (1 - real.sqrt (1 + 2 * x))^2 < 2 * x + 9

theorem solution_validity (x : ℝ) : proof_problem x ↔ (-1 / 2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 45 / 8) :=
  sorry

end solution_validity_l94_94065


namespace gcf_7_factorial_8_factorial_l94_94767

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l94_94767


namespace smallest_interval_for_p_l94_94575

theorem smallest_interval_for_p (P_A P_B p : ℝ) 
  (h1 : P_A = 3 / 4) 
  (h2 : P_B = 2 / 3) 
  (h3 : p ≥ P_A + P_B - 1) 
  (h4 : p ≤ min P_A P_B) : 
  (5 / 12) ≤ p ∧ p ≤ (2 / 3) :=
by
  have h5 : P_A + P_B - 1 = 5 / 12, by sorry
  exact ⟨h3.trans (by rw [h1, h2, h5]), h4.trans (min_le_iff.mpr (Or.inr h2))⟩

end smallest_interval_for_p_l94_94575


namespace gcf_7_factorial_8_factorial_l94_94771

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l94_94771


namespace smallest_possible_value_of_M_l94_94964

theorem smallest_possible_value_of_M :
  ∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  a + b + c + d + e + f = 4020 →
  (∃ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧
    (∀ (M' : ℕ), (∀ (a b c d e f : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
      a + b + c + d + e + f = 4020 →
      M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M' ≥ 804) → M = 804)) := by
  sorry

end smallest_possible_value_of_M_l94_94964


namespace partial_fraction_sum_zero_l94_94034

variable {A B C D E : ℝ}
variable {x : ℝ}

theorem partial_fraction_sum_zero (h : 
  (1:ℝ) / ((x-1)*x*(x+1)*(x+2)*(x+3)) = 
  A / (x-1) + B / x + C / (x+1) + D / (x+2) + E / (x+3)) : 
  A + B + C + D + E = 0 :=
by sorry

end partial_fraction_sum_zero_l94_94034


namespace mr_smith_grandchildren_prob_l94_94530

open Prob

-- Define the conditions
def children_count : ℕ := 12
def possibilities := 2 ^ children_count
def equal_gender_distribution := Nat.choose children_count (children_count / 2)
def probability_equal_gender := equal_gender_distribution.to_rat / possibilities.to_rat
def probability_unequal_gender := 1 - probability_equal_gender

-- The theorem statement to prove
theorem mr_smith_grandchildren_prob :
  probability_unequal_gender = (3172 : ℚ) / 4096 :=
by
  sorry

end mr_smith_grandchildren_prob_l94_94530


namespace no_base6_digit_for_divisibility_by_seven_l94_94392

theorem no_base6_digit_for_divisibility_by_seven :
    ∀ (d : ℕ), d < 6 → ¬ (7 ∣ (2 * 6^3 + d * 6^2 + d * 6 + 5)) :=
by
  intros d H
  by_contradiction H1
  sorry

end no_base6_digit_for_divisibility_by_seven_l94_94392


namespace difference_smallest_and_third_l94_94376

theorem difference_smallest_and_third (d1 d2 d3 : ℕ) :
  {d1, d2, d3} = {1, 6, 8} →
  let smallest := 100 * d1 + 10 * d2 + d3 in
  let numbers := [100 * d1 + 10 * d2 + d3,
                  100 * d1 + 10 * d3 + d2,
                  100 * d2 + 10 * d1 + d3,
                  100 * d2 + 10 * d3 + d1,
                  100 * d3 + 10 * d1 + d2,
                  100 * d3 + 10 * d2 + d1].sort(λ a b => a < b) in
  let third_smallest := numbers.nth 2 in
  third_smallest.get_or_else 0 - smallest = 450 :=
by
  sorry

end difference_smallest_and_third_l94_94376


namespace david_physics_marks_l94_94357

theorem david_physics_marks
  (marks_english : ℕ := 86)
  (marks_math : ℕ := 85)
  (marks_chemistry : ℕ := 87)
  (marks_biology : ℕ := 95)
  (average_marks : ℕ := 89)
  (number_of_subjects : ℕ := 5)
  (marks_sum : ℕ := average_marks * number_of_subjects := 445) : 
  marks_english + marks_math + marks_chemistry + marks_biology + ?marks_physics = marks_sum := 
sorry

end david_physics_marks_l94_94357


namespace cindy_envelopes_left_l94_94701

def total_envelopes : ℕ := 37
def envelopes_per_friend : ℕ := 3
def number_of_friends : ℕ := 5

theorem cindy_envelopes_left : total_envelopes - (envelopes_per_friend * number_of_friends) = 22 :=
by
  sorry

end cindy_envelopes_left_l94_94701


namespace airline_route_same_republic_exists_l94_94929

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l94_94929


namespace is_correct_function_l94_94344

theorem is_correct_function : ∀ (f : ℝ → ℝ),
  (∀ x, 0 < x ∧ x < π / 2 → f (x+π) = f x + π) ∧
  (∀ x, f(2π + x) = f x) ∧
  (∀ x, f (-x) = -f x) → 
  f = (λ x, Real.tan (x / 2)) := sorry

end is_correct_function_l94_94344


namespace coeff_x4_l94_94166

def poly1 : Polynomial ℚ := (1 - X)^4
def poly2 : Polynomial ℚ := X^3 * (1 + 3 * X)
def expr : Polynomial ℚ := poly1 - poly2

theorem coeff_x4 
  : (expr.coeff 4) = -2 :=
sorry

end coeff_x4_l94_94166


namespace cos_C_in_triangle_l94_94875

noncomputable def sin_A := 2 / 3
noncomputable def cos_B := 1 / 2
noncomputable def expected_cos_C := (2 * Real.sqrt 3 - Real.sqrt 5) / 6

theorem cos_C_in_triangle (A B C : ℝ) 
  (h1 : sin A = sin_A)
  (h2 : cos B = cos_B) : cos C = expected_cos_C :=
sorry

end cos_C_in_triangle_l94_94875


namespace squares_difference_l94_94696

theorem squares_difference : 435^2 - 365^2 = 56000 :=
  by
  -- Using the difference of squares formula
  have h : 435^2 - 365^2 = (435 + 365) * (435 - 365),
  {
    exact Int.SubSq (by norm_num : 435) (by norm_num : 365),
  },
  -- Calculating the results
  have h1 : 435 + 365 = 800 := by norm_num,
  have h2 : 435 - 365 = 70 := by norm_num,
  -- Combining the results
  rw [h, h1, h2],
  norm_num

end squares_difference_l94_94696


namespace fraction_to_decimal_zeros_l94_94279

theorem fraction_to_decimal_zeros (a b : ℕ) (h₁ : a = 5) (h₂ : b = 1600) : 
  (let dec_rep := (a : ℝ) / b;
       num_zeros := (String.mk (dec_rep.to_decimal_string.to_list) - '0')
  in num_zeros = 3) := 
  sorry

end fraction_to_decimal_zeros_l94_94279


namespace intersection_A_B_l94_94851

open Set Real

noncomputable def A := {x : ℝ | 0 < log x / log 4 ∧ log x / log 4 < 1 }
noncomputable def B := {x : ℝ | x ≤ 2}

theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 2} :=
by {
  simp_rw [A, B],
  ext,
  split;
  simp,
  sorry
}

end intersection_A_B_l94_94851


namespace penta_probability_l94_94239

noncomputable def penta_problem :
  ℕ := 5

theorem penta_probability {roles players choices : ℕ}
  (roles_eq : roles = 5)
  (players_eq : players = 5)
  (choices_eq : choices = 2)
  (probability_eq : \(\frac{51}{2500}\)) :
  True := sorry

end penta_probability_l94_94239


namespace a3_5a6_value_l94_94164

variable {a : ℕ → ℤ}

-- Conditions: The sequence {a_n} is an arithmetic sequence, and a_4 + a_7 = 19
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

axiom a_seq_arithmetic : is_arithmetic_sequence a
axiom a4_a7_sum : a 4 + a 7 = 19

-- Problem statement: Prove that a_3 + 5a_6 = 57
theorem a3_5a6_value : a 3 + 5 * a 6 = 57 :=
by
  -- Proof goes here
  sorry

end a3_5a6_value_l94_94164


namespace part1_arithmetic_part2_expression_l94_94828

def S (n : ℕ) : ℕ := n^2 

def a (n : ℕ) : ℕ := S n - S (n - 1)
  if h : n > 0 
  else 1  -- Base case: a_1 = 1 as derived in the solution 

theorem part1_arithmetic (n : ℕ) (h : n > 0) :
  a (n+1) - a n = 2 := by
  sorry

def T (n : ℕ) : ℚ := ∑ i in Finset.range n, 1 / (a (i+1) * a (i+2))

theorem part2_expression (n : ℕ) :
  T n = n / (2 * n + 1) := by
  sorry

end part1_arithmetic_part2_expression_l94_94828


namespace exists_route_same_republic_l94_94895

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l94_94895


namespace gcf_7_factorial_8_factorial_l94_94766

theorem gcf_7_factorial_8_factorial :
  let factorial (n : ℕ) := Nat.factorial n in
  let seven_factorial := factorial 7 in
  let eight_factorial := factorial 8 in
  ∃ (gcf : ℕ), gcf = Nat.gcd seven_factorial eight_factorial ∧ gcf = 5040 :=
by
  let factorial (n : ℕ) := Nat.factorial n
  let seven_factorial := factorial 7
  let eight_factorial := factorial 8
  have seven_factorial_eq : seven_factorial = 5040 := by sorry
  have gcf_eq_seven_factorial : Nat.gcd seven_factorial eight_factorial = seven_factorial := by sorry
  exact ⟨seven_factorial, gcf_eq_seven_factorial, seven_factorial_eq⟩

end gcf_7_factorial_8_factorial_l94_94766


namespace transformed_center_coordinates_l94_94028

-- Definitions for the problem
def initial_center : (ℝ × ℝ) := (3, -4)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

-- Proving the transformation results in the given coordinates
theorem transformed_center_coordinates :
  let reflected_center := reflect_y_axis initial_center in
  let final_center := translate_up reflected_center 12 in
  final_center = (-3, 8) :=
by
  sorry

end transformed_center_coordinates_l94_94028


namespace regions_in_n_gon_l94_94500

-- Given problem conditions
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2
def num_intersection_points (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) / 24

-- Proof problem
theorem regions_in_n_gon (n : ℕ) : 
  no_three_diagonals_intersect_at_one_point n →
  let R := 1 + num_diagonals n + num_intersection_points n in
  R = 1 + n * (n - 3) / 2 + n * (n - 1) * (n - 2) * (n - 3) / 24 := by
  sorry

-- Condition that no three diagonals intersect at one point
def no_three_diagonals_intersect_at_one_point (n : ℕ) : Prop := sorry

end regions_in_n_gon_l94_94500


namespace geometry_problem_l94_94081

-- Given conditions
variables (R a l : ℝ)
def diameter : ℝ := 2 * R
noncomputable def line_perpendicular_at_a (A B : ℝ) : Prop := A + a = B

-- Main theorem to be proven
theorem geometry_problem (A B C D : Point) (line s : Line)
  (h_diameter : distance A B = diameter R)
  (h_perpendicular : perpendicular s A B)
  (h_distance : distancefrompoint A s = a)
  (h_intersect_C : C ∈ circle_with_center A radius R ∧ C ∈ line_from_points A B)
  (h_intersect_D: D ∈ s ∧ D ∈ line_from_points A C)
  (h_product : distance A D * distance A C = 2 * R * a) :
  distance C D = l :=
sorry

end geometry_problem_l94_94081


namespace derivative_condition_l94_94147

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp(x) + c * x^2

theorem derivative_condition (c : ℝ) (h : (fun x => (f c x))' (-1) = 0) : c = 0 :=
  by
  sorry

end derivative_condition_l94_94147


namespace prove_A_plus_B_plus_1_l94_94165

theorem prove_A_plus_B_plus_1 (A B : ℤ) 
  (h1 : B = A + 2)
  (h2 : 2 * A^2 + A + 6 + 5 * B + 2 = 7 * (A + B + 1) + 5) :
  A + B + 1 = 15 :=
by 
  sorry

end prove_A_plus_B_plus_1_l94_94165


namespace zeros_in_fraction_decimal_l94_94286

noncomputable def decimal_representation_zeros (n d : ℕ) : ℕ :=
  let frac := n / d in
  let dec := Real.toRat (frac : ℝ) in
  let s := dec.toDigits 10 in
  let zeros := s.takeWhile (λ c => c = '0') in
  zeros.length

theorem zeros_in_fraction_decimal : 
  decimal_representation_zeros 5 1600 = 3 :=
by
  sorry

end zeros_in_fraction_decimal_l94_94286


namespace ordered_triples_count_l94_94778

theorem ordered_triples_count :
  (∃ (a b c: ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 6 * a + 10 * b + 15 * c = 3000) →
  (card (finset.filter (λ (t : ℕ × ℕ × ℕ), 6 * t.1 + 10 * t.2 + 15 * t.3 = 3000)
    (finset.product (finset.Icc 1 500) (finset.Icc 1 300) (finset.Icc 1 200))) = 4851) :=
begin
  sorry
end

end ordered_triples_count_l94_94778


namespace multiply_res_l94_94449

theorem multiply_res (
  h : 213 * 16 = 3408
) : 1.6 * 213 = 340.8 :=
sorry

end multiply_res_l94_94449


namespace gcf_factorial_seven_eight_l94_94764

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l94_94764


namespace airline_route_same_republic_exists_l94_94926

theorem airline_route_same_republic_exists
  (cities : Finset ℕ) (republics : Finset (Finset ℕ)) (routes : ℕ → ℕ → Prop)
  (H1 : cities.card = 100)
  (H2 : ∃ R1 R2 R3 : Finset ℕ, R1 ∈ republics ∧ R2 ∈ republics ∧ R3 ∈ republics ∧
        R1 ≠ R2 ∧ R2 ≠ R3 ∧ R1 ≠ R3 ∧ 
        (∀ (R : Finset ℕ), R ∈ republics → R.card ≤ 30) ∧ 
        R1 ∪ R2 ∪ R3 = cities)
  (H3 : ∃ (S : Finset ℕ), S ⊆ cities ∧ 70 ≤ S.card ∧ 
        (∀ x ∈ S, (routes x).filter (λ y, y ∈ cities).card ≥ 70)) :
  ∃ (x y : ℕ), x ∈ cities ∧ y ∈ cities ∧ (x = y ∨ ∃ R ∈ republics, x ∈ R ∧ y ∈ R) ∧ routes x y :=
begin
  sorry
end

end airline_route_same_republic_exists_l94_94926


namespace lemons_per_glass_l94_94174

theorem lemons_per_glass (lemons glasses : ℕ) (h : lemons = 18 ∧ glasses = 9) : lemons / glasses = 2 :=
by
  sorry

end lemons_per_glass_l94_94174


namespace determine_scores_l94_94938

variables {M Q S K : ℕ}

theorem determine_scores (h1 : Q > M ∨ K > M) 
                          (h2 : M ≠ K) 
                          (h3 : S ≠ Q) 
                          (h4 : S ≠ M) : 
  (Q, S, M) = (Q, S, M) :=
by
  -- We state the theorem as true
  sorry

end determine_scores_l94_94938


namespace log_sum_eq_two_l94_94644

/-- Let log be the base 10 logarithm. Prove that log 4 + 2 * log 5 equals 2. -/
theorem log_sum_eq_two : real.log10 4 + 2 * real.log10 5 = 2 :=
by
  sorry

end log_sum_eq_two_l94_94644


namespace solution_set_of_quadratic_inequality_l94_94255

namespace QuadraticInequality

variables {a b : ℝ}

def hasRoots (a b : ℝ) : Prop :=
  let x1 := -1 / 2
  let x2 := 1 / 3
  (- x1 + x2 = - b / a) ∧ (-x1 * x2 = 2 / a)

theorem solution_set_of_quadratic_inequality (h : hasRoots a b) : a + b = -14 :=
sorry

end QuadraticInequality

end solution_set_of_quadratic_inequality_l94_94255


namespace complex_number_imaginary_l94_94145

theorem complex_number_imaginary (m : ℝ) 
    (h : ∃ a : ℝ, m + 1 = 0 + a * I ∧ m - 1 = a) : m = -1 := by
  cases h with
  | intro a ha =>
    cases ha with
    | intro hre him =>
      have h1 : m + 1 = 0 := hre
      cases eq_add_of_eq_zero h1 with
      | left => 
        contradiction -- m + 1 cannot be zero for contradicting the given condition.
      | right =>
        exact eq_int_of_eq_eq_zero h1 -- m = - 1 hence proved

end complex_number_imaginary_l94_94145


namespace dice_probability_l94_94319

open Probability

theorem dice_probability (sum_gt_10 : ℕ) (w : ℕ) : 
  sum_gt_10 = 10 ∧ w = 48 → 
  P(sum (d6 d8) > 10) = 3 / 16 :=
by sorry

end dice_probability_l94_94319


namespace inequality_int_part_l94_94554

theorem inequality_int_part (a : ℝ) (n : ℕ) (h1 : 1 ≤ a) (h2 : (0 : ℝ) ≤ n ∧ (n : ℝ) ≤ a) : 
  ⌊a⌋ > (n / (n + 1 : ℝ)) * a := 
by 
  sorry

end inequality_int_part_l94_94554


namespace rubber_duck_charity_fundraiser_l94_94567

noncomputable def charity_raised (price_small price_medium price_large : ℕ) 
(bulk_discount_threshold_small bulk_discount_threshold_medium bulk_discount_threshold_large : ℕ)
(bulk_discount_rate_small bulk_discount_rate_medium bulk_discount_rate_large : ℝ)
(tax_rate_small tax_rate_medium tax_rate_large : ℝ)
(sold_small sold_medium sold_large : ℕ) : ℝ :=
  let cost_small := price_small * sold_small
  let cost_medium := price_medium * sold_medium
  let cost_large := price_large * sold_large

  let discount_small := if sold_small >= bulk_discount_threshold_small then 
                          (bulk_discount_rate_small * cost_small) else 0
  let discount_medium := if sold_medium >= bulk_discount_threshold_medium then 
                          (bulk_discount_rate_medium * cost_medium) else 0
  let discount_large := if sold_large >= bulk_discount_threshold_large then 
                          (bulk_discount_rate_large * cost_large) else 0

  let after_discount_small := cost_small - discount_small
  let after_discount_medium := cost_medium - discount_medium
  let after_discount_large := cost_large - discount_large

  let tax_small := tax_rate_small * after_discount_small
  let tax_medium := tax_rate_medium * after_discount_medium
  let tax_large := tax_rate_large * after_discount_large

  let total_small := after_discount_small + tax_small
  let total_medium := after_discount_medium + tax_medium
  let total_large := after_discount_large + tax_large

  total_small + total_medium + total_large

theorem rubber_duck_charity_fundraiser :
  charity_raised 2 3 5 10 15 20 0.1 0.15 0.2
  0.05 0.07 0.09 150 221 185 = 1693.10 :=
by 
  -- implementation of math corresponding to problem's solution
  sorry

end rubber_duck_charity_fundraiser_l94_94567


namespace volume_of_pyramid_base_isosceles_right_triangle_l94_94634

theorem volume_of_pyramid_base_isosceles_right_triangle (a h : ℝ) (ha : a = 3) (hh : h = 4) :
  (1 / 3) * (1 / 2) * a * a * h = 6 := by
  sorry

end volume_of_pyramid_base_isosceles_right_triangle_l94_94634


namespace truck_travel_distance_l94_94004

def distance_in_2r_seconds (a r : ℝ) : ℝ := a / 4
def time_in_5_minutes : ℝ := 5 * 60

theorem truck_travel_distance (a r : ℝ) :
  let rate := (distance_in_2r_seconds a r) / (2 * r)
  let total_time := time_in_5_minutes
  let total_distance := rate * total_time
  total_distance = 75 * a / (2 * r) := 
by
  sorry

end truck_travel_distance_l94_94004


namespace evaluate_mod_inverse_l94_94186

def pos_integer (n : ℕ) : Prop := n > 0

theorem evaluate_mod_inverse (m : ℕ) (hm : pos_integer m) (hm1 : m = 1) : 
  ((5^((2*m)) + 6)⁻¹ ≡ 5 [MOD 11]) :=
by
  -- This is the Lean statement.
  sorry

end evaluate_mod_inverse_l94_94186


namespace train_speed_l94_94339

theorem train_speed
  (train_length : ℕ)
  (man_speed_kmph : ℕ)
  (time_to_pass : ℕ)
  (speed_of_train : ℝ) :
  train_length = 180 →
  man_speed_kmph = 8 →
  time_to_pass = 4 →
  speed_of_train = 154 := 
by
  sorry

end train_speed_l94_94339


namespace exists_route_same_republic_l94_94893

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l94_94893


namespace grade11_paper_cutting_survey_l94_94314

theorem grade11_paper_cutting_survey (a b c x y z : ℕ)
  (h_total_students : a + b + c + x + y + z = 800)
  (h_clay_sculpture : a + b + c = 480)
  (h_paper_cutting : x + y + z = 320)
  (h_ratio : 5 * y = 3 * x ∧ 3 * z = 2 * y)
  (h_sample_size : 50):
  y * 50 / 800 = 6 :=
by {
  sorry
}

end grade11_paper_cutting_survey_l94_94314


namespace rebecca_mark_distance_difference_l94_94211

-- Statement of the problem
theorem rebecca_mark_distance_difference:
  let length := 3
  let width := 4
  let r := length + width
  let m := Real.sqrt (length^2 + width^2)
  (r - m) / r * 100 ≈ 28.57 :=
by
  sorry

end rebecca_mark_distance_difference_l94_94211


namespace prob_three_tails_one_head_in_four_tosses_l94_94469

open Nat

-- Definitions
def coin_toss_prob (n : ℕ) (k : ℕ) : ℚ :=
  (real.to_rat (choose n k) / real.to_rat (2^n))

-- Parameters: n = 4 (number of coin tosses), k = 3 (number of tails)
def four_coins_three_tails_prob : ℚ := coin_toss_prob 4 3

-- Theorem: The probability of getting exactly three tails and one head in four coin tosses is 1/4.
theorem prob_three_tails_one_head_in_four_tosses : four_coins_three_tails_prob = 1/4 := by
  sorry

end prob_three_tails_one_head_in_four_tosses_l94_94469


namespace arrangement_count_l94_94482

theorem arrangement_count (A B C D E : Type) :
  ∃ f : Fin 5 → (A ⊕ B ⊕ C ⊕ D ⊕ E), -- Bijective function representing arrangement
  (∀ i j, ¬((f i = A ∧ f j = B) ∨ (f i = B ∧ f j = A)) -- A and B are not adjacent
  ∧ ∃ i, (f i = C ∧ f (i+1) = D) ∨ (f i = D ∧ f (i+1) = C)) -- C and D are adjacent
  → cardinality ({ f : Fin 5 → (A ⊕ B ⊕ C ⊕ D ⊕ E) | -- Set of valid arrangements
    (∀ i j, ¬((f i = A ∧ f j = B) ∨ (f i = B ∧ f j = A))
    ∧ ∃ i, (f i = C ∧ f (i+1) = D) ∨ (f i = D ∧ f (i+1) = C)) }) = 24 := sorry

end arrangement_count_l94_94482


namespace smallest_n_terminating_decimal_contains_9_and_divisible_by_3_l94_94275

def contains_digit_9 (n : ℕ) : Prop :=
  "9".isIn (n.digits 10)

noncomputable def smallest_n : ℕ :=
  9000

theorem smallest_n_terminating_decimal_contains_9_and_divisible_by_3 :
  (∃ n : ℕ, n = smallest_n ∧ (1 / n : ℝ).den = 1 ∧ contains_digit_9 n ∧ n % 3 = 0) :=
by
  use smallest_n
  split
  rfl
  split
  sorry
  split
  sorry
  sorry

end smallest_n_terminating_decimal_contains_9_and_divisible_by_3_l94_94275


namespace regular_2n_gon_projection_l94_94501

theorem regular_2n_gon_projection (n : ℕ) (h : n > 0) :
  ∃ (P : Polyhedron), P.faces ≤ n + 2 ∧ regular_2n_gon n = P.projection := 
sorry

end regular_2n_gon_projection_l94_94501


namespace helen_hand_washing_time_l94_94369

theorem helen_hand_washing_time :
  (52 / 4) * 30 / 60 = 6.5 := by
  sorry

end helen_hand_washing_time_l94_94369


namespace Cindy_envelopes_left_l94_94707

theorem Cindy_envelopes_left :
  ∀ (initial_envelopes envelopes_per_friend friends : ℕ), 
    initial_envelopes = 37 →
    envelopes_per_friend = 3 →
    friends = 5 →
    initial_envelopes - envelopes_per_friend * friends = 22 :=
by
  intros initial_envelopes envelopes_per_friend friends h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Cindy_envelopes_left_l94_94707


namespace zeros_in_fraction_decimal_l94_94287

noncomputable def decimal_representation_zeros (n d : ℕ) : ℕ :=
  let frac := n / d in
  let dec := Real.toRat (frac : ℝ) in
  let s := dec.toDigits 10 in
  let zeros := s.takeWhile (λ c => c = '0') in
  zeros.length

theorem zeros_in_fraction_decimal : 
  decimal_representation_zeros 5 1600 = 3 :=
by
  sorry

end zeros_in_fraction_decimal_l94_94287


namespace eval_expression_l94_94049

def a : ℕ := 4 * 5 * 6
def b : ℚ := 1/4 + 1/5 - 1/10

theorem eval_expression : a * b = 42 := by
  sorry

end eval_expression_l94_94049


namespace onions_total_l94_94216

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ) 
  (h1: Sara_onions = 4) (h2: Sally_onions = 5) (h3: Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 :=
by
  sorry

end onions_total_l94_94216


namespace sum_of_digits_l94_94869

theorem sum_of_digits (k : ℕ) : 
  (k = 307) ∧ (303 = (String.length (Nat.toDigits 10 ((2^k) * (5^300))))) 
  → (Nat.sumDigits (128 * 10 ^ 300) = 11) :=
sorry

end sum_of_digits_l94_94869


namespace prob_fourth_term_integer_l94_94727
noncomputable theory

/-- The initial term of the sequence. -/
def a1 : ℤ := 5

/-- Function to compute the next term in the sequence based on the result of a fair coin flip.
If heads (true), the rule is 3a - 2.
If tails (false), the rule is a/3 - 2. -/
def next_term (a : ℤ) (heads : Bool) : ℤ :=
  if heads then 3 * a - 2 else a / 3 - 2

/--
The probability that the fourth term in Derek's sequence is an integer.
Given that Derek initiates the sequence with the first term as 5. For 
each subsequent term, he flips a fair coin. If the result is heads, 
he triples the previous term and then subtracts 2; if tails, he divides 
the previous term by 3 and then subtracts 2.
-/
theorem prob_fourth_term_integer : (1 : ℚ) / 2 = 1 / 2 :=
by sorry

end prob_fourth_term_integer_l94_94727


namespace airline_route_within_republic_l94_94911

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l94_94911


namespace gcf_7_8_fact_l94_94745

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem gcf_7_8_fact : Nat.gcd (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7_8_fact_l94_94745


namespace apollonius_circle_l94_94264

def competitive_areas_part_a (A B : Point ℝ) (c1 c2 : ℝ) (P : Point ℝ) (d : Point ℝ → Point ℝ → ℝ) :
  Prop :=
  d A P * c1 = d B P * c2

theorem apollonius_circle (A B : Point ℝ) (c1 c2 : ℝ) (P : Point ℝ) (d : Point ℝ → Point ℝ → ℝ) :
  competitive_areas_part_a A B c1 c2 P d → ApolloniusCircle A B c2 c1 P :=
by
  sorry

end apollonius_circle_l94_94264


namespace sum_of_digits_10_pow_93_minus_93_l94_94278

theorem sum_of_digits_10_pow_93_minus_93 : 
  let n := 93 in 
  (10^n - 93).digits.sum = 826 := 
by 
  let n := 93 
  sorry

end sum_of_digits_10_pow_93_minus_93_l94_94278


namespace CE_sq_plus_DE_sq_l94_94516

theorem CE_sq_plus_DE_sq (O A B C D E : Type) 
  [circle : circle O 8] 
  (diameter_AB : diameter O A B)
  (chord_CD : chord O C D)
  (intersect_E : E ∈ (line A B) ∧ E ∈ (line C D))
  (BE_4 : dist B E = 4)
  (angle_AEC_30 : ∠(A, E, C) = 30) : 
  CE^2 + DE^2 = 64 :=
sorry

end CE_sq_plus_DE_sq_l94_94516


namespace polar_equation_circle_l94_94167

noncomputable def centerC : ℝ × ℝ := (3, π / 6)

def radiusC : ℝ := 1

def ratioOQ_QP : ℝ := 2 / 3

theorem polar_equation_circle 
  (C : ℝ × ℝ) (r : ℝ) (OQ_QP_ratio : ℝ) 
  (hC : C = (3, π / 6)) 
  (hr : r = 1) 
  (hRatio : OQ_QP_ratio = 2 / 3) 
  : ( ∃ (ρ θ : ℝ), ρ^2 - 6 * ρ * cos (θ - (π / 6)) + 8 = 0 ) 
  ∧ ( ∃ (ρ θ : ℝ), ρ = 15 * cos (θ - (π / 6)) ) := 
  sorry

end polar_equation_circle_l94_94167


namespace gcf_factorial_seven_eight_l94_94758

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l94_94758


namespace part_I_a_and_b_part_II_area_l94_94876

theorem part_I_a_and_b (a b : ℝ) (h_c : c = 2) (h_C : C = π / 3) (h_area : 1/2 * a * b * sin C = sqrt 3) : 
  a = 2 ∧ b = 2 := sorry

theorem part_II_area (a b : ℝ) (A B C : ℝ) (h_c : c = 2) (h_C : C = π / 3)
  (h_sin_eq : sin C + sin (B - A) = 2 * sin (2 * A)) : 
  1/2 * a * b * sin C = 2 * sqrt 3 / 3 := sorry

end part_I_a_and_b_part_II_area_l94_94876


namespace minimum_value_l94_94415

variable {a : ℕ → ℝ}

-- Define \( \{a_n\} \) is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

-- Assert the conditions
def conditions (a : ℕ → ℝ) : Prop :=
  is_geometric_sequence a ∧ (∀ n, a n > 0) ∧ a 2018 = sqrt 2 / 2

-- Prove the main statement
theorem minimum_value (a : ℕ → ℝ) (h : conditions a) : 
  1 / a 2017 + 2 / a 2019 ≥ 4 :=
sorry

end minimum_value_l94_94415


namespace trigonometric_equation_solution_l94_94295

-- Lean 4 statement to prove the equivalent proof problem.
theorem trigonometric_equation_solution (t : ℝ) :
    (\sin (2 * t) - Real.arcsin (2 * t))^2 + (Real.arccos (2 * t) - \cos (2 * t))^2 = 1 ↔
    ∃ (k : ℤ), t = (Real.pi / 8) * (2 * k + 1) :=
by
    sorry

end trigonometric_equation_solution_l94_94295


namespace bologna_sandwiches_l94_94355

-- Conditions
def total_sandwiches : ℕ := 300
def ratio_cheese : ℝ := 2.5
def ratio_bologna : ℝ := 7.5
def ratio_peanut_butter : ℝ := 8.25
def ratio_tuna : ℝ := 3.5
def ratio_egg_salad : ℝ := 4.25

def total_ratio := ratio_cheese + ratio_bologna + ratio_peanut_butter + ratio_tuna + ratio_egg_salad

-- Question and Answer
theorem bologna_sandwiches :
  (total_sandwiches / total_ratio * ratio_bologna).round = 87 := 
by
  sorry

end bologna_sandwiches_l94_94355


namespace prob_three_tails_one_head_in_four_tosses_l94_94468

open Nat

-- Definitions
def coin_toss_prob (n : ℕ) (k : ℕ) : ℚ :=
  (real.to_rat (choose n k) / real.to_rat (2^n))

-- Parameters: n = 4 (number of coin tosses), k = 3 (number of tails)
def four_coins_three_tails_prob : ℚ := coin_toss_prob 4 3

-- Theorem: The probability of getting exactly three tails and one head in four coin tosses is 1/4.
theorem prob_three_tails_one_head_in_four_tosses : four_coins_three_tails_prob = 1/4 := by
  sorry

end prob_three_tails_one_head_in_four_tosses_l94_94468


namespace speed_first_half_trip_l94_94529

noncomputable def calculate_speed (d_total d_half t_ratio avg_speed : ℕ) : ℕ :=
let v := 2 * avg_speed in v

theorem speed_first_half_trip (d_total d_half t_ratio avg_speed v : ℕ) :
  d_total = 640 →
  d_half = 320 →
  t_ratio = 3 →
  avg_speed = 40 →
  v = 2 * avg_speed →
  calculate_speed d_total d_half t_ratio avg_speed = 80 :=
by
  intros
  simp [calculate_speed, *]
  sorry

end speed_first_half_trip_l94_94529


namespace zeros_in_fraction_decimal_l94_94285

noncomputable def decimal_representation_zeros (n d : ℕ) : ℕ :=
  let frac := n / d in
  let dec := Real.toRat (frac : ℝ) in
  let s := dec.toDigits 10 in
  let zeros := s.takeWhile (λ c => c = '0') in
  zeros.length

theorem zeros_in_fraction_decimal : 
  decimal_representation_zeros 5 1600 = 3 :=
by
  sorry

end zeros_in_fraction_decimal_l94_94285


namespace steven_peaches_calc_l94_94953

variables (Jake_peaches Steven_peaches : ℕ)

def condition1 : Prop := Jake_peaches = Steven_peaches - 12
def condition2 : Prop := Jake_peaches = 7

theorem steven_peaches_calc : condition1 ∧ condition2 → Steven_peaches = 19 :=
by
  -- The proof itself is not required here
  sorry

end steven_peaches_calc_l94_94953


namespace impossibility_of_partition_l94_94697

open Classical

noncomputable def grid := Fin 12 × Fin 12

def is_adjacent (cell1 cell2 : grid) : Prop :=
  abs (cell1.1 - cell2.1) + abs (cell1.2 - cell2.2) = 1

def is_L_shaped (shape : set grid) : Prop :=
  ∃ (a b c : grid), {a, b, c} = shape ∧ 
    is_adjacent a b ∧ is_adjacent b c ∧ a ≠ c

def partitions_into_L_shapes (partitions : set (set grid)) : Prop :=
  (∀ p ∈ partitions, is_L_shaped p) ∧ 
  (∀ (cell : grid), ∃! p, cell ∈ p ∧ p ∈ partitions)

def intersects_same_number_of_L_shapes (partitions : set (set grid)) : Prop :=
  let row_intersects (r : Fin 12) := {p | ∃ cell ∈ p, cell.1 = r} in
  let col_intersects (c : Fin 12) := {p | ∃ cell ∈ p, cell.2 = c} in
  ∀ (r r' : Fin 12), row_intersects r = row_intersects r' ∧ 
  ∀ (c c' : Fin 12), col_intersects c = col_intersects c'

theorem impossibility_of_partition 
  (partitions : set (set grid)) :
  partitions_into_L_shapes partitions →
  ¬ intersects_same_number_of_L_shapes partitions :=
  by sorry

end impossibility_of_partition_l94_94697


namespace volume_of_solid_of_revolution_l94_94386

noncomputable theory
open Real

def f (x : ℝ) : ℝ := 2^x - 1

theorem volume_of_solid_of_revolution :
  let V := π - (2 * π * (1 - log 2)^2) / (log 2)^2 in
  ∫ y in 0..1, (log (y + 1) / log 2)^2 * π = V :=
by
  let V := π - (2 * π * (1 - log 2)^2) / (log 2)^2
  sorry

end volume_of_solid_of_revolution_l94_94386


namespace find_a_if_symmetric_l94_94148

def f (x : ℝ) (a : ℝ) := sin (2 * x) + a * cos (2 * x)
def symmetric_line (x : ℝ) := x = π / 8

theorem find_a_if_symmetric :
  (∃ a : ℝ, (∀ x : ℝ, symmetric_line x → f x a = f x a)) → a = 1 := 
sorry

end find_a_if_symmetric_l94_94148


namespace right_triangle_largest_perimeter_l94_94160

theorem right_triangle_largest_perimeter : 
  ∃ (c : ℕ), (a = 8) → (b = 9) → (c = 12) → (8^2 + 9^2 = 145) →
  (∀ c < 12, a^2 + b^2 > c^2) → 
  8 + 9 + c = 29 :=
begin
  sorry
end

end right_triangle_largest_perimeter_l94_94160


namespace Q1_solution_Q2_solution_Q3_solution_l94_94439

universe u

variable (U : Set ℝ) (A B P : Set ℝ)

-- Define the universal set, sets A, B, and P
def U := {x : ℝ | True}
def A := {x : ℝ | -4 ≤ x ∧ x < 2}
def B := {x : ℝ | -1 < x ∧ x ≤ 3}
def P := {x : ℝ | x ≤ 0 ∨ x ≥ 5/2}

-- Define complements and intersections/unions of sets
def complement (U : Set ℝ) (S : Set ℝ) : Set ℝ := {x : ℝ | ¬ (S x)}

def Q1 := A ∩ B
def Q2 := complement U B ∪ P
def Q3 := (A ∩ B) ∩ complement U P

-- The proof statements
theorem Q1_solution : Q1 = { x : ℝ | -1 < x ∧ x < 2 } := by
  sorry

theorem Q2_solution : Q2 = { x : ℝ | x ≤ 0 ∨ x ≥ 5/2 } := by
  sorry

theorem Q3_solution : Q3 = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end Q1_solution_Q2_solution_Q3_solution_l94_94439


namespace percentage_students_C_grade_l94_94015

theorem percentage_students_C_grade :
  let scores := [94, 65, 59, 99, 82, 89, 90, 68, 79, 62, 85, 81, 64, 83, 91]
  let count_C := scores.countp (λ x => 78 ≤ x ∧ x ≤ 87)
  let total_students := scores.length
  (count_C : ℚ) / total_students * 100 = 33.33 :=
by
  let scores := [94, 65, 59, 99, 82, 89, 90, 68, 79, 62, 85, 81, 64, 83, 91]
  let count_C := scores.countp (λ x => 78 ≤ x ∧ x ≤ 87)
  let total_students := scores.length
  show ((count_C : ℚ) / total_students * 100 = 33.33) sorry

end percentage_students_C_grade_l94_94015


namespace total_marbles_count_l94_94479

variable (r b g : ℝ)
variable (h1 : r = 1.4 * b) (h2 : g = 1.5 * r)

theorem total_marbles_count (r b g : ℝ) (h1 : r = 1.4 * b) (h2 : g = 1.5 * r) :
  r + b + g = 3.21 * r :=
by
  sorry

end total_marbles_count_l94_94479


namespace petya_wins_second_race_and_leads_by_1_l94_94989

-- Definitions based on conditions
variables (v_p v_v : ℝ) -- speeds for Petya and Vasya
variables (h1 : v_v * 60 / v_p = 51) -- Vasya runs 51 meters when Petya finishes the 60-meter race
variables (h2 : ∀ t, v_p * t = 60 → v_v * t + 9 = 60) -- relationship between Petya and Vasya's running times and distances in the first race

-- Statement of the proof problem
theorem petya_wins_second_race_and_leads_by_1.35
  (h3 : ∀ t, v_p * t = 69 → ¬ (v_v * t = 60 + 9)) -- Petya starts 9 meters behind Vasya
  (h4 : v_p > 0 ∧ v_v > 0) -- Positive speeds for Petya and Vasya
  : true :=
begin
  sorry
end

end petya_wins_second_race_and_leads_by_1_l94_94989


namespace airline_route_within_republic_l94_94909

theorem airline_route_within_republic (cities : Finset α) (republics : Finset (Finset α))
  (routes : α → Finset α) (h_cities : cities.card = 100) (h_republics : republics.card = 3)
  (h_partition : ∀ r ∈ republics, disjoint r (Finset.univ \ r) ∧ Finset.univ = r ∪ (Finset.univ \ r))
  (h_millionaire : ∃ m ∈ cities, 70 ≤ (routes m).card) :
  ∃ c1 c2 ∈ cities, ∃ r ∈ republics, (routes c1).member c2 ∧ c1 ≠ c2 ∧ c1 ∈ r ∧ c2 ∈ r :=
by sorry

end airline_route_within_republic_l94_94909


namespace solve_system_of_equations_l94_94223

theorem solve_system_of_equations :
  ∃ x y z u : ℝ,
    x + y + z = 15 ∧
    x + y + u = 16 ∧
    x + z + u = 18 ∧
    y + z + u = 20 ∧
    x = 3 ∧
    y = 5 ∧
    z = 7 ∧
    u = 8 :=
by {
  use [3, 5, 7, 8],
  sorry
}

end solve_system_of_equations_l94_94223


namespace min_xy_l94_94873

variable {x y : ℝ}

theorem min_xy (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : x * y ≥ 180 := 
sorry

end min_xy_l94_94873


namespace inequality_sum_xi_squares_l94_94080

theorem inequality_sum_xi_squares (n : ℕ) (x : Fin n → ℝ) 
    (h0 : x 0 = 0) 
    (h_pos : ∀ i, 1 ≤ i → x i > 0) 
    (h_sum : ∑ i, x i = 1) : 
    1 ≤ ∑ i in Finset.range (n+1), x i / (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Icc i n, x j)) 
    ∧ ∑ i in Finset.range (n+1), x i / (Real.sqrt (1 + ∑ j in Finset.range i, x j) * Real.sqrt (∑ j in Finset.Icc i n, x j)) < Real.pi / 2 := 
by 
  sorry

end inequality_sum_xi_squares_l94_94080


namespace right_triangle_5_12_13_l94_94101

theorem right_triangle_5_12_13 (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) : a^2 + b^2 = c^2 := 
by 
   sorry

end right_triangle_5_12_13_l94_94101


namespace probability_three_tails_one_head_l94_94456

noncomputable def probability_of_three_tails_one_head : ℚ :=
  if H : 1/2 ∈ ℚ then 4 * ((1 / 2)^4 : ℚ)
  else 0

theorem probability_three_tails_one_head :
  probability_of_three_tails_one_head = 1 / 4 :=
by {
  have h : (1 / 2 : ℚ) ∈ ℚ := by norm_cast; norm_num,
  rw probability_of_three_tails_one_head,
  split_ifs,
  { field_simp [h],
    norm_cast,
    norm_num }
}

end probability_three_tails_one_head_l94_94456


namespace pirate_catch_up_time_l94_94331

/-
  Define the necessary constants for the problem.
  - initial_distance: Initial distance between ships.
  - pirate_initial_speed: Pirate ship's initial speed.
  - cargo_speed: Cargo ship's speed.
  - time_until_storm: Time until the storm hits.
  - pirate_reduced_speed_per_cargo: Pirate ship's reduced speed per cargo speed after the storm.
  - total_time: Total time from pursuit start to catch-up.
-/

def initial_distance : ℝ := 15
def pirate_initial_speed : ℝ := 12
def cargo_speed : ℝ := 9
def time_until_storm : ℝ := 3
def pirate_reduced_speed_per_cargo : ℝ := 10 / 9

theorem pirate_catch_up_time :
  let total_time := time_until_storm + ((initial_distance - (pirate_initial_speed - cargo_speed) * time_until_storm) / (10 * cargo_speed / 9 - cargo_speed))
  total_time = 9 :=
by sorry

end pirate_catch_up_time_l94_94331


namespace no_such_topology_exists_l94_94958

open Set

-- Define the interval partitions and characteristic functions
def I (m : ℕ) (k : ℕ) : Set ℝ :=
  Ioc ((k - 1) / (2 ^ m)) (k / (2 ^ m))

def char_fn (n : ℕ) (x : ℝ) : ℝ :=
  if x ∈ I (n / (2^m)) (n % (2^m)) then 1 else 0

noncomputable def E := ℝ → ℝ

-- Define the target statement
theorem no_such_topology_exists :
  ∀ ( f_n : ℕ → E) ( f : E),
    ¬ (∃ T : TopologicalSpace E, ∀ n : ℕ, f_n n ⟶ f ↔ f_n n →ᵐ[μ] f) :=
sorry

end no_such_topology_exists_l94_94958


namespace total_paint_l94_94794

theorem total_paint (liters_per_color : ℕ) (color_count : ℕ) (total : ℕ) : 
  liters_per_color = 5 → color_count = 3 → total = 15 →
  liters_per_color * color_count = total :=
by
  intros h1 h2 h3
  rw [h1, h2]
  exact h3

end total_paint_l94_94794


namespace no_such_function_exists_l94_94362

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l94_94362


namespace solve_k_l94_94301

theorem solve_k (t s : ℤ) : (∃ k m, 8 * k + 4 = 7 * m ∧ k = -4 + 7 * t ∧ m = -4 + 8 * t) →
  (∃ k m, 12 * k - 8 = 7 * m ∧ k = 3 + 7 * s ∧ m = 4 + 12 * s) →
  7 * t - 4 = 7 * s + 3 →
  ∃ k, k = 3 + 7 * s :=
by
  sorry

end solve_k_l94_94301


namespace range_of_a_l94_94474

open Set Int

noncomputable def f (x a : ℝ) : ℝ := x^3 + x^2 - a*x - 4

def has_extremum_in_interval (a : ℝ) : Prop :=
  let f_derivative := λ x, 3*x^2 + 2*x - a
  let f_prime_neg1 := f_derivative (-1)
  let f_prime_1 := f_derivative 1
  (f_prime_neg1 * f_prime_1 < 0) 

theorem range_of_a (a : ℝ) : has_extremum_in_interval a ↔ 1 ≤ a ∧ a < 5 :=
sorry

end range_of_a_l94_94474


namespace greatest_product_two_integers_sum_2004_l94_94606

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end greatest_product_two_integers_sum_2004_l94_94606


namespace train_speed_l94_94000

theorem train_speed
  (train_length : Real := 460)
  (bridge_length : Real := 140)
  (time_seconds : Real := 48) :
  ((train_length + bridge_length) / time_seconds) * 3.6 = 45 :=
by
  sorry

end train_speed_l94_94000


namespace alphabet_letter_count_l94_94503

def sequence_count : Nat :=
  let total_sequences := 2^7
  let sequences_per_letter := 1 + 7 -- 1 correct sequence + 7 single-bit alterations
  total_sequences / sequences_per_letter

theorem alphabet_letter_count : sequence_count = 16 :=
  by
    -- Proof placeholder
    sorry

end alphabet_letter_count_l94_94503


namespace initial_total_perimeter_l94_94263

theorem initial_total_perimeter (n : ℕ) (a : ℕ) (m : ℕ)
  (h1 : n = 2 * m)
  (h2 : 40 = 2 * a * m)
  (h3 : 4 * n - 6 * m = 4 * n - 40) :
  4 * n = 280 :=
by sorry

end initial_total_perimeter_l94_94263


namespace exists_route_within_republic_l94_94921

theorem exists_route_within_republic :
  ∃ (cities : Finset ℕ) (republics : Finset (Finset ℕ)),
    (Finset.card cities = 100) ∧
    (by ∃ (R1 R2 R3 : Finset ℕ), R1 ∪ R2 ∪ R3 = cities ∧ Finset.card R1 ≤ 30 ∧ Finset.card R2 ≤ 30 ∧ Finset.card R3 ≤ 30) ∧
    (∃ (millionaire_cities : Finset ℕ), Finset.card millionaire_cities ≥ 70 ∧ ∀ city ∈ millionaire_cities, ∃ routes : Finset ℕ, Finset.card routes ≥ 70 ∧ routes ⊆ cities) →
  ∃ (city1 city2 : ℕ) (republic : Finset ℕ), city1 ∈ republic ∧ city2 ∈ republic ∧
    city1 ≠ city2 ∧ (∃ route : ℕ, route = (city1, city2) ∨ route = (city2, city1)) :=
sorry

end exists_route_within_republic_l94_94921


namespace ratio_GH_HJ_l94_94949

-- Define the points and their relationships
variables {A B C G H J : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
[MetricSpace G] [MetricSpace H] [MetricSpace J]

-- Define the points A, B, C as vectors
variables (a b c : EuclideanSpace ℝ (Fin 3))

-- Define the points G and H with given ratios
def g := (3/5 : ℝ) • a + (2/5 : ℝ) • b
def h := (2/5 : ℝ) • b + (3/5 : ℝ) • c

-- Define the point J as the intersection of GH and AC
def j := (3/5 : ℝ) • h + (2/5 : ℝ) • g

-- Prove the ratio GH / HJ
theorem ratio_GH_HJ (a b c : EuclideanSpace ℝ (Fin 3)) : 
  let g := (3 / 5 : ℝ) • a + (2 / 5 : ℝ) • b,
      h := (2 / 5 : ℝ) • b + (3 / 5 : ℝ) • c,
      j := (3 / 5 : ℝ) • h + (2 / 5 : ℝ) • g in 
  dist h g / dist h j = 5 / 2 :=
by sorry

end ratio_GH_HJ_l94_94949


namespace scientific_notation_of_4_point_4_billion_l94_94303

theorem scientific_notation_of_4_point_4_billion :
  let billion := 10^9
  in 4.4 * billion = 4.4 * 10^9 :=
by
  sorry

end scientific_notation_of_4_point_4_billion_l94_94303


namespace function_increase_positive_interval_function_decrease_negative_interval_function_increase_negative_interval_l94_94036

noncomputable def nat_exponent_mono (x m n : Nat) : Prop :=
  x^(m/n)

theorem function_increase_positive_interval (m n : Nat): 
  ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 → nat_exponent_mono x2 m n > nat_exponent_mono x1 m n :=
sorry

theorem function_decrease_negative_interval (m n : Nat): 
  ∀ (x1 x2 : ℝ), x2 < 0 ∧ x1 < x2 ∧ (n % 2 = 1) ∧ (m % 2 = 0) → nat_exponent_mono x2 m n < nat_exponent_mono x1 m n :=
sorry

theorem function_increase_negative_interval (m n : Nat): 
  ∀ (x1 x2 : ℝ), x2 < 0 ∧ x1 < x2 ∧ (n % 2 = 1) ∧ (m % 2 = 1) → nat_exponent_mono x2 m n > nat_exponent_mono x1 m n :=
sorry

end function_increase_positive_interval_function_decrease_negative_interval_function_increase_negative_interval_l94_94036


namespace min_floodgates_to_reduce_level_l94_94484

-- Definitions for the conditions given in the problem
def num_floodgates : ℕ := 10
def a (v : ℝ) := 30 * v
def w (v : ℝ) := 2 * v

def time_one_gate : ℝ := 30
def time_two_gates : ℝ := 10
def time_target : ℝ := 3

-- Prove that the minimum number of floodgates \(n\) that must be opened to achieve the goal
theorem min_floodgates_to_reduce_level (v : ℝ) (n : ℕ) :
  (a v + time_target * v) ≤ (n * time_target * w v) → n ≥ 6 :=
by
  sorry

end min_floodgates_to_reduce_level_l94_94484


namespace sequence_formula_l94_94397

noncomputable def sequence (n : ℕ) : ℕ
| 0 => 1
| (n + 1) => (n + 1) * (sequence n - sequence (n + 1))

theorem sequence_formula (n : ℕ) (hn : n > 0) : sequence n = n :=
by
  sorry

end sequence_formula_l94_94397


namespace Cindy_envelopes_left_l94_94708

theorem Cindy_envelopes_left :
  ∀ (initial_envelopes envelopes_per_friend friends : ℕ), 
    initial_envelopes = 37 →
    envelopes_per_friend = 3 →
    friends = 5 →
    initial_envelopes - envelopes_per_friend * friends = 22 :=
by
  intros initial_envelopes envelopes_per_friend friends h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Cindy_envelopes_left_l94_94708


namespace solve_trig_log_equation_l94_94560

theorem solve_trig_log_equation :
  ∀ (x : ℝ), (∃ (k : ℤ), 2 * Real.log (Real.cot x) / Real.log 3 = Real.log (Real.cos x) / Real.log 2 ∧
                           x ∈ set.Ioo (2 * real.pi * k) (real.pi / 2 + 2 * real.pi * k) ∧
                           x ∈ set.Ioo (2 * real.pi * k - real.pi / 2) (2 * real.pi * k + real.pi / 2)) ↔
                           ∃ (k : ℤ), x = real.pi / 3 + 2 * real.pi * k := 
by
  sorry

end solve_trig_log_equation_l94_94560


namespace solve_card_trade_problem_l94_94202

def card_trade_problem : Prop :=
  ∃ V : ℕ, 
  (75 - V + 10 + 88 - 8 + V = 75 + 88 - 8 + 10 ∧ V + 15 = 35)

theorem solve_card_trade_problem : card_trade_problem :=
  sorry

end solve_card_trade_problem_l94_94202


namespace curve_equation_line_m_eq_l94_94961

/- Conditions definitions -/
def on_circle (x0 y0 : ℝ) : Prop := x0^2 + y0^2 = 4
def line_perpendicular (x0 : ℝ) : set (ℝ × ℝ) := {⟨x0, y⟩ | y ∈ set.univ}
def DM_eq (x0 y0 ym : ℝ) : Prop := ym = x0 * sqrt 3 / 2 * y0


/- Theorem Statements -/
theorem curve_equation (x y x0 y0 : ℝ) (h1 : on_circle x0 y0) (h2 : DM_eq x0 y0 y) :
  x^2 / 4 + y^2 / 3 = 1 :=
sorry

theorem line_m_eq (k : ℝ) :
  ∃ k : ℝ, k = (3 * real.sqrt 7 / 7) ∨ k = (-3 * real.sqrt 7 / 7) :=
sorry

end curve_equation_line_m_eq_l94_94961


namespace min_value_expr_l94_94380

theorem min_value_expr (x : ℝ) (hx : x > 0) : 3 * real.sqrt x + 4 / (x^2) = 4 * real.sqrt 2 :=
sorry

end min_value_expr_l94_94380


namespace nathan_ate_gumballs_l94_94856

theorem nathan_ate_gumballs (packages : ℝ) (gumballs_per_package : ℝ)
  (packages_eq : packages = 20)
  (gumballs_per_package_eq : gumballs_per_package = 5) :
  packages * gumballs_per_package = 100 :=
by {
  -- Introduce the given facts
  rw [packages_eq, gumballs_per_package_eq],
  -- Simplify the multiplication
  norm_num,
  -- Expected result
  exact rfl,
}

end nathan_ate_gumballs_l94_94856


namespace function_properties_l94_94390

noncomputable def f (x : ℝ) : ℝ := Real.sin ((13 * Real.pi / 2) - x)

theorem function_properties :
  (∀ x : ℝ, f x = Real.cos x) ∧
  (∀ x : ℝ, f (-x) = f x) ∧
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (forall t: ℝ, (∀ x : ℝ, f (x + t) = f x) → (t = 2 * Real.pi ∨ t = -2 * Real.pi)) :=
by
  sorry

end function_properties_l94_94390


namespace probability_same_length_segments_l94_94179

theorem probability_same_length_segments (T : Finset ℝ) (hT_card : T.card = 15)
  (num_sides : ℕ) (num_diagonals : ℕ)
  (h_sides : num_sides = 6) (h_diagonals : num_diagonals = 9)
  (h_T : T = (Finset.range num_sides).map ⟨λ _, 1, sorry⟩ ∪ (Finset.range num_diagonals).map ⟨λ _, 1 + (1 / 2), sorry⟩) :
  let prob_side := (num_sides : ℝ) / (num_sides + num_diagonals),
      prob_diagonal := (num_diagonals : ℝ) / (num_sides + num_diagonals),
      prob_same_side := (num_sides - 1 : ℝ) / (num_sides + num_diagonals - 1),
      prob_same_diagonal := (num_diagonals - 1 : ℝ) / (num_sides + num_diagonals - 1),
      total_prob := (prob_side * prob_same_side) + (prob_diagonal * prob_same_diagonal) in
  total_prob = 17 / 35 := by
  sorry

end probability_same_length_segments_l94_94179


namespace complete_the_square_eqn_l94_94561

theorem complete_the_square_eqn (x b c : ℤ) (h_eqn : x^2 - 10 * x + 15 = 0) (h_form : (x + b)^2 = c) : b + c = 5 := by
  sorry

end complete_the_square_eqn_l94_94561


namespace partition_three_sets_partition_four_sets_not_partition_three_sets_l94_94640

-- Problem (a)
theorem partition_three_sets {A B C : Set ℕ} :
  (∀ n : ℕ, n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ 
  (∀ m n : ℕ, |m - n| = 2 ∨ |m - n| = 5 → ¬(m ∈ A ∧ n ∈ A) ∧ ¬(m ∈ B ∧ n ∈ B) ∧ ¬(m ∈ C ∧ n ∈ C)) :=
sorry

-- Problem (b), part (1)
theorem partition_four_sets {A B C D : Set ℕ} :
  (∀ n : ℕ, n ∈ A ∨ n ∈ B ∨ n ∈ C ∨ n ∈ D) ∧ 
  (∀ m n : ℕ, |m - n| = 2 ∨ |m - n| = 3 ∨ |m - n| = 5 → ¬(m ∈ A ∧ n ∈ A) ∧ ¬(m ∈ B ∧ n ∈ B) ∧ ¬(m ∈ C ∧ n ∈ C) ∧ ¬(m ∈ D ∧ n ∈ D)) :=
sorry

-- Problem (b), part (2)
theorem not_partition_three_sets {A B C : Set ℕ} :
  ¬ ((∀ n : ℕ, n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ 
  (∀ m n : ℕ, |m - n| = 2 ∨ |m - n| = 3 ∨ |m - n| = 5 → ¬(m ∈ A ∧ n ∈ A) ∧ ¬(m ∈ B ∧ n ∈ B) ∧ ¬(m ∈ C ∧ n ∈ C))) :=
sorry

end partition_three_sets_partition_four_sets_not_partition_three_sets_l94_94640


namespace total_difference_proof_l94_94663

-- Definitions for the initial quantities
def initial_tomatoes : ℕ := 17
def initial_carrots : ℕ := 13
def initial_cucumbers : ℕ := 8

-- Definitions for the picked quantities
def picked_tomatoes : ℕ := 5
def picked_carrots : ℕ := 6

-- Definitions for the given away quantities
def given_away_tomatoes : ℕ := 3
def given_away_carrots : ℕ := 2

-- Definitions for the remaining quantities 
def remaining_tomatoes : ℕ := initial_tomatoes - (picked_tomatoes - given_away_tomatoes)
def remaining_carrots : ℕ := initial_carrots - (picked_carrots - given_away_carrots)

-- Definitions for the difference quantities
def difference_tomatoes : ℕ := initial_tomatoes - remaining_tomatoes
def difference_carrots : ℕ := initial_carrots - remaining_carrots

-- Definition for the total difference
def total_difference : ℕ := difference_tomatoes + difference_carrots

-- Lean Theorem Statement
theorem total_difference_proof : total_difference = 6 := by
  -- Proof is omitted
  sorry

end total_difference_proof_l94_94663


namespace polynomials_equal_if_integer_parts_equal_l94_94998

theorem polynomials_equal_if_integer_parts_equal (f g : ℚ[X])
  (hf : f.degree = 2)
  (hg : g.degree = 2)
  (h : ∀ x : ℝ, ⌊f.eval x⌋ = ⌊g.eval x⌋) : f = g :=
sorry

end polynomials_equal_if_integer_parts_equal_l94_94998


namespace focus_of_parabola_l94_94378

theorem focus_of_parabola (a k : ℝ) (h_eq : ∀ x : ℝ, k = 6 ∧ a = 9) :
  (0, (1 / (4 * a)) + k) = (0, 217 / 36) := sorry

end focus_of_parabola_l94_94378


namespace sum_of_possible_B_values_l94_94669

-- Definition: A number is divisible by 8 if the number formed by its last three digits is divisible by 8.
def divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

-- The problem condition states that "The number 2B7 is divisible by 8 for some single digit B"
def three_digit_number (b : ℕ) : ℕ :=
  200 + 10 * b + 7

-- Define the main proposition
theorem sum_of_possible_B_values :
  (∑ b in {b : ℕ | divisible_by_8 (three_digit_number b) ∧ b < 10}.to_finset, b) = 18 :=
sorry

end sum_of_possible_B_values_l94_94669


namespace limit_of_seq_l94_94775

-- Define the sequence
def seq (n : ℕ) : ℝ := (2 * n + 3) / (n + 1)

-- Theorem stating that the limit of the sequence is 2
theorem limit_of_seq : (Real.liminf at_top seq) = 2 := by
  sorry

end limit_of_seq_l94_94775


namespace simplify_to_quadratic_l94_94519

noncomputable def simplify_expression (a b c x : ℝ) : ℝ := 
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c + 2)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem simplify_to_quadratic {a b c x : ℝ} (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  simplify_expression a b c x = x^2 - (a + b + c) * x + sorry :=
sorry

end simplify_to_quadratic_l94_94519


namespace ellipse_eq_max_AB_dist_max_triangle_area_l94_94429

noncomputable def ellipse_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  (∃ c : ℝ, c / a = sqrt 2 / 2 ∧ sqrt (b^2 + c^2) = sqrt 2)

noncomputable def line_intersects_ellipse (m : ℝ) (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ y = x + m

theorem ellipse_eq : 
  ellipse_conditions (sqrt 2) 1 →
  (∃ (a b : ℝ), a = sqrt 2 ∧ b = 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

theorem max_AB_dist :
  (∃ (a b : ℝ), a = sqrt 2 ∧ b = 1) →
  ∀ m : ℝ, line_intersects_ellipse m (sqrt 2) 1 →
  ∃ (max_dist : ℝ), max_dist = 4 * sqrt 3 / 3 :=
by sorry

theorem max_triangle_area : 
  (∃ (a b : ℝ), a = sqrt 2 ∧ b = 1) →
  ∀ m : ℝ, line_intersects_ellipse m (sqrt 2) 1 →
  ∃ (max_area : ℝ), max_area = sqrt 2 / 2 :=
by sorry

end ellipse_eq_max_AB_dist_max_triangle_area_l94_94429


namespace triangle_side_ratio_l94_94407

variable (A B C : ℝ)  -- angles in radians
variable (a b c : ℝ)  -- sides of triangle

theorem triangle_side_ratio
  (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) :
  b / a = Real.sqrt 2 :=
by sorry

end triangle_side_ratio_l94_94407


namespace exists_route_same_republic_l94_94897

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l94_94897


namespace normal_prob_interval_l94_94825

open MeasureTheory

noncomputable def normalDist (μ σ : ℝ) : Measure ℝ :=
  Measure.map (λ x, μ + x * σ) (Measure.gaussian 0 1)

variable {σ : ℝ}

theorem normal_prob_interval {X : ℝ → Measure ℝ} (hX : X = normalDist 2 σ) (hP : X.measure {x | x < 0} = 0.1) :
  X.measure {x | 2 < x ∧ x < 4} = 0.4 :=
sorry

end normal_prob_interval_l94_94825


namespace exists_airline_route_within_same_republic_l94_94916

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l94_94916


namespace set_D_is_empty_l94_94625

-- Definitions based on the conditions from the original problem
def set_A : Set ℝ := {x | x + 3 = 3}
def set_B : Set (ℝ × ℝ) := {(x, y) | y^2 = -x^2}
def set_C : Set ℝ := {x | x^2 ≤ 0}
def set_D : Set ℝ := {x | x^2 - x + 1 = 0}

-- The theorem statement
theorem set_D_is_empty : set_D = ∅ :=
sorry

end set_D_is_empty_l94_94625


namespace area_of_triangle_length_of_hypotenuse_l94_94485

-- Define the legs of the right triangle
def leg1 : ℕ := 30
def leg2 : ℕ := 45

-- Prove the area of the triangle is 675 square inches
theorem area_of_triangle : (1 / 2 : ℝ) * (leg1 : ℝ) * (leg2 : ℝ) = 675 := sorry

-- Prove the length of the hypotenuse is 54 inches
theorem length_of_hypotenuse : real.sqrt ((leg1 : ℝ)^2 + (leg2 : ℝ)^2) = 54 := sorry

end area_of_triangle_length_of_hypotenuse_l94_94485


namespace y_coord_third_vertex_l94_94010

theorem y_coord_third_vertex (a b : ℝ) (h1 : a = (2:ℝ)) (h2 : b = (10:ℝ)) : 
  (∃ y : ℝ, y = 3 + 4 * Real.sqrt 3) :=
by
  have h3 : a < b := by linarith [h1, h2]
  use 3 + 4 * Real.sqrt 3
  split
  assumption
  sorry

end y_coord_third_vertex_l94_94010


namespace length_of_smallest_repeating_block_in_decimal_expansion_of_11_over_13_l94_94611

theorem length_of_smallest_repeating_block_in_decimal_expansion_of_11_over_13 : 
  ∀ (d : ℚ), d = 11 / 13 → repeating_decimal_length d = 6 :=
by
  intro d h
  -- proof goes here
  sorry

end length_of_smallest_repeating_block_in_decimal_expansion_of_11_over_13_l94_94611


namespace min_distance_curveC1_curveC2_l94_94941

-- Definitions of the conditions
def curveC1 (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 3 + Real.cos θ ∧ P.2 = 4 + Real.sin θ

def curveC2 (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

-- Proof statement
theorem min_distance_curveC1_curveC2 :
  (∀ A B : ℝ × ℝ,
    curveC1 A →
    curveC2 B →
    ∃ m : ℝ, m = 3 ∧ ∀ d : ℝ, (d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) → d ≥ m) := 
  sorry

end min_distance_curveC1_curveC2_l94_94941


namespace tangential_quadrilateral_is_cyclic_l94_94252

theorem tangential_quadrilateral_is_cyclic
  (a b c d : ℝ)
  (h : a + c = b + d)
  (area_eq : sqrt (a * b * c * d) = sqrt ((a + b + c + d) / 2 - a) * ((a + b + c + d) / 2 - b) * ((a + b + c + d) / 2 - c) * ((a + b + c + d) / 2 - d)) :
  cyclic_quadrilateral Q :=
sorry

end tangential_quadrilateral_is_cyclic_l94_94252


namespace fraction_from_condition_l94_94312

theorem fraction_from_condition (x f : ℝ) (h : 0.70 * x = f * x + 110) (hx : x = 300) : f = 1 / 3 :=
by
  sorry

end fraction_from_condition_l94_94312


namespace perpendicular_value_of_k_parallel_value_of_k_l94_94074

variables (a b : ℝ × ℝ) (k : ℝ)

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-3, 1)
def ka_plus_b (k : ℝ) : ℝ × ℝ := (2*k - 3, 3*k + 1)
def a_minus_3b : ℝ × ℝ := (11, 0)

theorem perpendicular_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  a - ka_plus_b k = a_minus_3b → k = (3 / 2) :=
sorry

theorem parallel_value_of_k 
  (h : a = vector_a ∧ b = vector_b ∧ (ka_plus_b k) = (2*k - 3, 3*k + 1) ∧ a_minus_3b = (11, 0)) :
  ∃ k, (ka_plus_b (-1/3)) = (-1/3 * 11, -1/3 * 0) ∧ k = -1 / 3 :=
sorry

end perpendicular_value_of_k_parallel_value_of_k_l94_94074


namespace minimum_value_f_when_b_gt_sqrt_a_l94_94836

theorem minimum_value_f_when_b_gt_sqrt_a (a b : ℝ) (h_a_pos : 0 < a) (h_b_gt_sqrt_a : b > real.sqrt a) :
  ∃ x ∈ Ioo (0 : ℝ) b, (∀ y ∈ Ioo (0 : ℝ) b, f a y ≥ f a x) ∧ f a x = 2 * real.sqrt a :=
begin
  sorry,
end

def f (a x : ℝ) := (a + x^2) / x

end minimum_value_f_when_b_gt_sqrt_a_l94_94836


namespace part1_part2_l94_94819

-- Setting up the conditions
variables {a b c A B C : ℝ}
variables (h1: c = sqrt 3)
variables (h2: (2 * b - a) * cos C = c * cos A)
variables (h3: a * b * sin C / 2 = sqrt 3 / 2)
variables (h4: a / sin A = b / sin B = c / sin C)

-- Proof for Part 1
theorem part1 : C = π / 3 :=
by {
  sorry
}

-- Proof for Part 2
theorem part2 (hC: C = π / 3) : a + b = 3 :=
by {
  sorry
}

end part1_part2_l94_94819


namespace failed_in_hindi_percentage_l94_94161

/-- In an examination, a specific percentage of students failed in Hindi (H%), 
45% failed in English, and 20% failed in both. We know that 40% passed in both subjects. 
Prove that 35% students failed in Hindi. --/
theorem failed_in_hindi_percentage : 
  ∀ (H E B P : ℕ),
    (E = 45) → (B = 20) → (P = 40) → (100 - P = H + E - B) → H = 35 := by
  intros H E B P hE hB hP h
  sorry

end failed_in_hindi_percentage_l94_94161


namespace evaluate_expression_l94_94048

theorem evaluate_expression : 
  ((-4 : ℤ) ^ 6) / (4 ^ 4) + (2 ^ 5) * (5 : ℤ) - (7 ^ 2) = 127 :=
by sorry

end evaluate_expression_l94_94048


namespace fraction_of_sand_is_one_third_l94_94659

noncomputable def total_weight : ℝ := 24
noncomputable def weight_of_water (total_weight : ℝ) : ℝ := total_weight / 4
noncomputable def weight_of_gravel : ℝ := 10
noncomputable def weight_of_sand (total_weight weight_of_water weight_of_gravel : ℝ) : ℝ :=
  total_weight - weight_of_water - weight_of_gravel
noncomputable def fraction_of_sand (weight_of_sand total_weight : ℝ) : ℝ :=
  weight_of_sand / total_weight

theorem fraction_of_sand_is_one_third :
  fraction_of_sand (weight_of_sand total_weight (weight_of_water total_weight) weight_of_gravel) total_weight
  = 1/3 := by
  sorry

end fraction_of_sand_is_one_third_l94_94659


namespace inequality_and_iff_arithmetic_sequence_l94_94512

variable (n : ℕ) (x : Fin n → ℝ)
-- Assume n > 0 and the sequence is sorted: x1 ≤ x2 ≤ ... ≤ xn
axiom (hn_pos : n > 0) (h_sorted : ∀ i j, i < j → x i ≤ x j)

theorem inequality_and_iff_arithmetic_sequence :
  let L := (∑ i j, |x i - x j|)^2
  let R := 2 * (n^2 - 1) / 3 * ∑ i j, (x i - x j)^2
  L ≤ R ∧ (L = R ↔ ∃ d, ∀ k, x k = x 0 + k * d) :=
by
  sorry

end inequality_and_iff_arithmetic_sequence_l94_94512


namespace exists_airline_route_within_same_republic_l94_94900

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l94_94900


namespace intersection_M_N_l94_94130

-- Define the sets M and N based on the given conditions.
def M : set ℝ := {x | 0 ≤ x ∧ x < 2}
def N : set ℝ := {y | ∃ x : ℝ, y = -x^2 + 3}

-- State the theorem we need to prove.
theorem intersection_M_N : M ∩ N = {z | 0 ≤ z ∧ z < 2} :=
sorry

end intersection_M_N_l94_94130


namespace sequence_log_sum_l94_94085

theorem sequence_log_sum :
  (∀ n: ℕ, 1 ≤ n → Real.log (x n.succ) = 1 + Real.log (x n)) →
  (∑ i in Finset.range 100, x (i + 1)) = 1 →
  Real.log (∑ i in Finset.range 100, x (i + 101)) = 100 := by
  sorry

end sequence_log_sum_l94_94085


namespace minimum_cost_l94_94213

variable (A1 A2 A3 A4 A5: ℕ)

def area_trapezoid (b1 b2 h : ℕ) : ℕ := (b1 + b2) * h / 2

def cost (areas : list ℕ) (prices : list ℕ) : ℕ :=
  (list.zip areas prices).map (λ x => x.1 * x.2).sum

theorem minimum_cost :
  let prices := [310, 280, 230, 180, 120]
  ∃ (A1 A2 A3 A4 A5 : ℕ),
  A1 = area_trapezoid 7 5 2 ∧
  A2 = area_trapezoid 5 3 4 ∧
  A3 = area_trapezoid 6 4 5 ∧
  A4 = area_trapezoid 8 6 3 ∧
  A5 = area_trapezoid 10 8 3 ∧
  cost [A1, A2, A3, A4, A5] prices = 20770 := sorry

end minimum_cost_l94_94213


namespace pictureArea_l94_94718

-- Define the variables and conditions
variables {x y : ℕ}

-- Ensure x and y are greater than zero
def positiveIntegers (x y : ℕ) : Prop := x > 0 ∧ y > 0

-- Define the dimensions of the frame including the picture
def frameArea (x y : ℕ) : ℕ := (2*x + 5)*(y + 3)

-- Define the condition that the total area of the frame is 60 square inches
def frameAreaCondition : Prop := frameArea x y = 60

-- The theorem to prove that the area of the picture is 27 square inches
theorem pictureArea : positiveIntegers x y → frameAreaCondition → x * y = 27 :=
by
  sorry

end pictureArea_l94_94718


namespace logan_usual_cartons_l94_94195

theorem logan_usual_cartons 
  (C : ℕ)
  (h1 : ∀ cartons, (∀ jars : ℕ, jars = 20 * cartons) → jars = 20 * C)
  (h2 : ∀ cartons, cartons = C - 20)
  (h3 : ∀ damaged_jars, (∀ cartons : ℕ, cartons = 5) → damaged_jars = 3 * 5)
  (h4 : ∀ completely_damaged_jars, completely_damaged_jars = 20)
  (h5 : ∀ good_jars, good_jars = 565) :
  C = 50 :=
by
  sorry

end logan_usual_cartons_l94_94195


namespace sum_q_p_values_is_neg42_l94_94120

def p (x : Int) : Int := 2 * Int.natAbs x - 1

def q (x : Int) : Int := -(Int.natAbs x) - 1

def values : List Int := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

def q_p_sum : Int :=
  let q_p_values := values.map (λ x => q (p x))
  q_p_values.sum

theorem sum_q_p_values_is_neg42 : q_p_sum = -42 :=
  by
    sorry

end sum_q_p_values_is_neg42_l94_94120


namespace area_of_gray_part_a_l94_94007

theorem area_of_gray_part_a
  (height width : ℝ)
  (square_side : ℝ)
  (num_squares : ℕ)
  (h₁ : height = 1)
  (h₂ : width = 4)
  (h₃ : square_side = 1)
  (h₄ : num_squares = 4) :
  (gray_area : ℝ) := 1.5 :=
sorry

end area_of_gray_part_a_l94_94007


namespace odd_function_neg_value_l94_94424

theorem odd_function_neg_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_value : f 1 = 1) : f (-1) = -1 :=
by
  sorry

end odd_function_neg_value_l94_94424


namespace exists_route_within_same_republic_l94_94887

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l94_94887


namespace leap_year_has_53_mondays_l94_94780

theorem leap_year_has_53_mondays 
  (days_in_leap_year : ℕ := 366) 
  (prob_53_mondays : ℚ := 0.2857142857142857) 
  (leap_year_extra_days : ℕ := 2) 
  (extra_days_combination : ℕ := 2) 
  (probability_of_53_mondays : prob_53_mondays = (2 / 7)) 
  : (days_in_leap_year = 366) → (leap_year_extra_days = 2) → probability_of_53_mondays = (2 / 7) → leap_year_extra_days * 7 / days_in_leap_year = prob_53_mondays → leap_year_extra_days = extra_days_combination := 
sorry

end leap_year_has_53_mondays_l94_94780


namespace circular_park_diameter_factor_l94_94316

theorem circular_park_diameter_factor (r : ℝ) :
  (π * (3 * r)^2) / (π * r^2) = 9 ∧ (2 * π * (3 * r)) / (2 * π * r) = 3 :=
by
  sorry

end circular_park_diameter_factor_l94_94316


namespace trapezoid_circle_ratio_l94_94347

variable (P R : ℝ)

def is_isosceles_trapezoid_inscribed_in_circle (P R : ℝ) : Prop :=
  ∃ m A, 
    m = P / 4 ∧
    A = m * 2 * R ∧
    A = (P * R) / 2

theorem trapezoid_circle_ratio (P R : ℝ) 
  (h : is_isosceles_trapezoid_inscribed_in_circle P R) :
  (P / 2 * π * R) = (P / 2 * π * R) :=
by
  -- Use the given condition to prove the statement
  sorry

end trapezoid_circle_ratio_l94_94347


namespace deal_or_no_deal_min_eliminations_l94_94162

theorem deal_or_no_deal_min_eliminations :
  (∃ (n : ℕ), n + 8 ≤ 16 ∧ 30 - n = 14) :=
by { 
  have h₁ : 30 - 14 = 16 := rfl,
  have h₂ : 14 + 8 = 22,
  have h₃ : 22 < 30,
  exact ⟨14, by simp, by simp⟩ }

end deal_or_no_deal_min_eliminations_l94_94162


namespace triangle_length_KL_5_l94_94499

theorem triangle_length_KL_5 (JL JK KL : ℝ) (h1 : ∠J = 90) (h2 : tan ∠K = 4 / 3) (h3 : JK = 3) 
  (right_triangle : is_right_triangle ∆JKL J) : KL = 5 :=
by
  -- Proof will go here.
  sorry

end triangle_length_KL_5_l94_94499


namespace shortest_of_tallest_vs_tallest_of_shortest_l94_94654

-- Definitions of students and properties
noncomputable def demo : Prop :=
  let students : ℕ := 200
  let rows_down : ℕ := 20
  let rows_across : ℕ := 10
  let A := λ (s : ℕ), ∃ t, shortest (tallest (longitudinal_row s)) 
  let B := λ (s : ℕ), ∃ t, tallest (shortest (transverse_row s))
  ∀ s, students = rows_down * rows_across → A s ≥ B s

theorem shortest_of_tallest_vs_tallest_of_shortest (students : ℕ) (rows_down : ℕ) (rows_across : ℕ) (A B : ℕ → ℕ) :
  200 = 20 * 10 →
  (∀ s, A s = shortest (tallest (longitudinal_row s))) →
  (∀ s, B s = tallest (shortest (transverse_row s))) →
  ∀ s, A s ≥ B s :=
by
  sorry

end shortest_of_tallest_vs_tallest_of_shortest_l94_94654


namespace polynomial_divisibility_l94_94782

theorem polynomial_divisibility (A B : ℝ)
  (h: ∀ (x : ℂ), x^2 + x + 1 = 0 → x^104 + A * x^3 + B * x = 0) :
  A + B = 0 :=
by
  sorry

end polynomial_divisibility_l94_94782


namespace car_speeds_l94_94237

theorem car_speeds (d x : ℝ) (small_car_speed large_car_speed : ℝ) 
  (h1 : d = 135) 
  (h2 : small_car_speed = 5 * x) 
  (h3 : large_car_speed = 2 * x) 
  (h4 : 135 / small_car_speed + (4 + 0.5) = 135 / large_car_speed)
  : small_car_speed = 45 ∧ large_car_speed = 18 := by
  sorry

end car_speeds_l94_94237


namespace calvins_initial_weight_l94_94023

/-- 
Calvin's initial weight problem
Given that Calvin loses 8 pounds every month and his weight after one year is 154 pounds, 
prove that his initial weight was 250 pounds.
-/
theorem calvins_initial_weight (loss_per_month : ℤ) (months : ℕ) (final_weight : ℤ) (initial_weight : ℤ) :
  loss_per_month = 8 → months = 12 → final_weight = 154 → initial_weight = final_weight + loss_per_month * months := 
by {
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  exact sorry
}

end calvins_initial_weight_l94_94023


namespace largest_class_students_l94_94936

theorem largest_class_students :
  ∃ (x : ℕ), 
    let c1 := x,
        c2 := x - 2,
        c3 := x - 4,
        c4 := x - 6,
        c5 := x - 8 in
    (c1 + c2 + c3 + c4 + c5 = 105) ∧ (x = 25) :=
begin
  sorry
end

end largest_class_students_l94_94936


namespace exists_route_within_same_republic_l94_94885

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l94_94885


namespace max_value_of_MN_l94_94872

noncomputable def f (x : ℝ) : ℝ := Math.sin x
noncomputable def g (x : ℝ) : ℝ := 2 * (Math.cos x) ^ 2 - 1
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem max_value_of_MN : 
  ∃ x : ℝ, (h x).abs = 2 :=
sorry

end max_value_of_MN_l94_94872


namespace radius_of_largest_circle_correct_l94_94033

noncomputable def radius_of_largest_circle_in_quadrilateral (AB BC CD DA : ℝ) (angle_BCD : ℝ) : ℝ :=
  if AB = 10 ∧ BC = 12 ∧ CD = 8 ∧ DA = 14 ∧ angle_BCD = 90
    then Real.sqrt 210
    else 0

theorem radius_of_largest_circle_correct :
  radius_of_largest_circle_in_quadrilateral 10 12 8 14 90 = Real.sqrt 210 :=
by
  sorry

end radius_of_largest_circle_correct_l94_94033


namespace cole_fence_cost_l94_94031

def side_length : ℝ := 9
def back_length : ℝ := 18
def cost_per_foot : ℝ := 3

def total_cost : ℝ :=
  let side_cost := cost_per_foot * side_length
  let cole_left_cost := side_cost * (2/3)
  let cole_back_cost := (cost_per_foot * back_length) / 2
  cole_left_cost + side_cost + cole_back_cost

theorem cole_fence_cost : total_cost = 72 := by sorry

end cole_fence_cost_l94_94031


namespace least_value_expression_l94_94608

noncomputable def expression (x y z : ℂ) : ℂ :=
  (2 * x - 3) ^ 2 * (y + 4) ^ 3 * (z - complex.sqrt 2) + 
  complex.exp (10 ^ (real.exp (complex.real_part x))) + 
  3 * complex.I * (x - 5) * (complex.cos y) * complex.log (z + 1)

theorem least_value_expression (x y z : ℂ) : 
  ∃ m : ℝ, m = ℂ(abs (expression x y z)) ∧ m ≥ 10 ^ real.exp (complex.real_part x) := sorry

end least_value_expression_l94_94608


namespace collinear_condition_l94_94093

theorem collinear_condition (A B O P : Type) [AddCommGroup A] [Module ℝ A] 
  (not_collinear : ¬collinear ℝ ![A, B, O])
  (h : ∃ (λ μ : ℝ), P = λ • O + μ • B ) : 
  (∃ (λ μ : ℝ), |λ| = |μ| ∧ P = λ • O + μ • B) ↔ collinear ℝ ![A, B, P] :=
begin
  sorry,
end

end collinear_condition_l94_94093


namespace delta_curve_l94_94302

noncomputable def is_d_curve {K : Type} (K_curve : ConvexCurve K) (O : Point) (r h : ℝ) : Prop :=
  let K' := rotate_curve K_curve O 120
  let K'' := rotate_curve K_curve O 240
  let M := add_curves [K_curve, K', K'']
  is_circle M r

theorem delta_curve {K : Type} (K_curve : ConvexCurve K) (O : Point) (h : ℝ)
  (H : is_d_curve K_curve O h h) : is_delta_curve K_curve :=
sorry

end delta_curve_l94_94302


namespace fourth_row_sequence_l94_94372

section FillGrid

variable (grid : Matrix (Fin 6) (Fin 6) (Fin 6))

-- Definitions based on given conditions
def unique_rows (grid : Matrix (Fin 6) (Fin 6) (Fin 6)) : Prop :=
  ∀ i : Fin 6, ∀ j k : Fin 6, j ≠ k → grid i j ≠ grid i k

def unique_columns (grid : Matrix (Fin 6) (Fin 6) (Fin 6)) : Prop :=
  ∀ j : Fin 6, ∀ i k : Fin 6, i ≠ k → grid i j ≠ grid i k

def black_dot (x y : Fin 6) : Prop :=
  x = 2 * y ∨ y = 2 * x

def white_dot (x y : Fin 6) : Prop :=
  x = y + 1 ∨ y = x + 1

-- Specific target fourth row
def fourth_row (grid : Matrix (Fin 6) (Fin 6) (Fin 6)) : Vector (Fin 6) 6 :=
  λ i, grid 3 i

-- Prove the specific sequence in the fourth row
theorem fourth_row_sequence (grid : Matrix (Fin 6) (Fin 6) (Fin 6)) :
  unique_rows grid ∧ unique_columns grid ∧
  (∀ i j, i ≠ j → 
    black_dot (grid i j) (grid (i + 1) j) ∨ 
    white_dot (grid i j) (grid (i + 1) j) ∨ 
    black_dot (grid i (j + 1)) (grid i j) ∨ 
    white_dot (grid i (j + 1)) (grid i j)
  ) →
  fourth_row grid = ![2, 1, 4, 3, 6] := by
  sorry

end FillGrid

end fourth_row_sequence_l94_94372


namespace rita_swimming_months_l94_94163

theorem rita_swimming_months
    (total_required_hours : ℕ := 1500)
    (backstroke_hours : ℕ := 50)
    (breaststroke_hours : ℕ := 9)
    (butterfly_hours : ℕ := 121)
    (monthly_hours : ℕ := 220) :
    (total_required_hours - (backstroke_hours + breaststroke_hours + butterfly_hours)) / monthly_hours = 6 := 
by
    -- Proof is omitted
    sorry

end rita_swimming_months_l94_94163


namespace problem1_problem2_l94_94022

-- Problem 1
theorem problem1 : (Real.sqrt 9 + (-2020)^(0:ℤ) - (1/4:ℚ)^(-1)) = 0 := sorry

-- Problem 2
variables (a b : ℚ)

theorem problem2 : (2 * a - b)^2 - (a + b) * (b - a) = 5 * a^2 - 4 * a * b := sorry

end problem1_problem2_l94_94022


namespace average_adjacent_boys_girls_l94_94639

theorem average_adjacent_boys_girls (B G : ℕ) (S : ℕ) 
  (hB : B = 7) (hG : G = 13)
  (hS : (S >= 1) ∧ (S <= 14)) :
  \(\bar S = 9 \)
 :=
begin
  sorry
end

end average_adjacent_boys_girls_l94_94639


namespace triangle_angles_l94_94359

theorem triangle_angles (a b c : ℝ) (ha : a = 4) (hb : b = 4) (hc : c = 2 * real.sqrt 6 - 2 * real.sqrt 2) :
  ∃ γ α β, α = 15 ∧ β = 15 ∧ γ = 150 ∧ α + β + γ = 180 :=
by
  sorry -- Proof steps go here

end triangle_angles_l94_94359


namespace domain_of_f_odd_function_inequality_solution_gt1_inequality_solution_lt1_l94_94841

variable (a : ℝ) (f : ℝ → ℝ)
noncomputable def f_def (x : ℝ) : ℝ := log a (2 + x) - log a (2 - x)

-- Assume conditions
variable (a_pos : 0 < a)
variable (a_ne_one : a ≠ 1)

-- 1. Prove that the domain of f(x) is -2 < x < 2
theorem domain_of_f : ∀ x, (f x = f_def a x) → (-2 < x) ∧ (x < 2) := 
sorry

-- 2. Prove that f(x) is an odd function
theorem odd_function : ∀ x, (f x = f_def a x) → f (-x) = -f x := 
sorry

-- 3. Prove the solutions for the inequality f(x) ≥ log_a(3x)
theorem inequality_solution_gt1 : (1 < a) → (∀ x, (f x = f_def a x) → (f x) ≥ log a (3 * x) → (2/3 ≤ x) ∧ (x ≤ 1)) := 
sorry

theorem inequality_solution_lt1 : (0 < a ∧ a < 1) → (∀ x, (f x = f_def a x) → (f x) ≥ log a (3 * x) → ((1 ≤ x ∧ x < 2) ∨ (0 < x ∧ x ≤ 2/3)) :=
sorry

end domain_of_f_odd_function_inequality_solution_gt1_inequality_solution_lt1_l94_94841


namespace log_of_n_geq_k_log_2_l94_94542

theorem log_of_n_geq_k_log_2
  (n : ℕ)
  (k : ℕ)
  (h : ∃ (prime_factors : Finset ℕ), (∀ p ∈ prime_factors, nat.prime p) ∧ prime_factors.card = k ∧ (∀ p ∈ prime_factors, p ∣ n)) :
  Real.log n ≥ k * Real.log 2 :=
by
  sorry

end log_of_n_geq_k_log_2_l94_94542


namespace f_value_neg_five_half_one_l94_94417

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom interval_definition : ∀ x, 0 < x ∧ x < 1 → f x = (4:ℝ) ^ x

-- The statement to prove
theorem f_value_neg_five_half_one : f (-5/2) + f 1 = -2 :=
by
  sorry

end f_value_neg_five_half_one_l94_94417


namespace even_digit_perfect_squares_odd_digit_perfect_squares_l94_94740

-- Define the property of being a four-digit number
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Define the property of having even digits
def is_even_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 0

-- Define the property of having odd digits
def is_odd_digit_number (n : ℕ) : Prop :=
  ∀ digit ∈ (n.digits 10), digit % 2 = 1

-- Part (a) statement
theorem even_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_even_digit_number n ∧ ∃ m : ℕ, n = m * m ↔ 
    n = 4624 ∨ n = 6084 ∨ n = 6400 ∨ n = 8464 :=
sorry

-- Part (b) statement
theorem odd_digit_perfect_squares :
  ∀ n : ℕ, is_four_digit n ∧ is_odd_digit_number n ∧ ∃ m : ℕ, n = m * m → false :=
sorry

end even_digit_perfect_squares_odd_digit_perfect_squares_l94_94740


namespace sum_of_four_squares_eq_20_l94_94343

variable (x y : ℕ)

-- Conditions based on the provided problem
def condition1 := 2 * x + 2 * y = 16
def condition2 := 2 * x + 3 * y = 19

-- Theorem to be proven
theorem sum_of_four_squares_eq_20 (h1 : condition1 x y) (h2 : condition2 x y) : 4 * x = 20 :=
by
  sorry

end sum_of_four_squares_eq_20_l94_94343


namespace isosceles_triangles_vertices_form_quadrilateral_with_equal_diagonals_l94_94535

variable (Q : Type) [quad : quadrilateral Q]
variables (A B C D : Q)

-- Conditions
axiom perpendicular_diagonals : ∃ P : Q, P ∈ (diagonals_perpendicular Q)
axiom isosceles_triangles_constructed : ∀ (X Y : Q), X-Y-isosceles-triangle-on-side X Y 

-- Question as proof goal
theorem isosceles_triangles_vertices_form_quadrilateral_with_equal_diagonals :
  ∃ R S T U : Q, isosceles_trian_vertices V W X Y ∧ (quadrilateral.has_equal_diagonals R S T U) :=
sorry

end isosceles_triangles_vertices_form_quadrilateral_with_equal_diagonals_l94_94535


namespace exists_route_within_same_republic_l94_94888

-- Conditions
def city := ℕ
def republic := ℕ
def airline_routes (c1 c2 : city) : Prop := sorry -- A predicate representing airline routes

constant n_cities : ℕ := 100
constant n_republics : ℕ := 3
constant cities_in_republic : city → republic
constant very_connected_city : city → Prop
axiom at_least_70_very_connected : ∃ S : set city, S.card ≥ 70 ∧ ∀ c ∈ S, (cardinal.mk {d : city | airline_routes c d}.to_finset) ≥ 70

-- Question
theorem exists_route_within_same_republic : ∃ c1 c2 : city, c1 ≠ c2 ∧ cities_in_republic c1 = cities_in_republic c2 ∧ airline_routes c1 c2 :=
sorry

end exists_route_within_same_republic_l94_94888


namespace functional_equation_solution_l94_94053

theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) : 
    ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l94_94053


namespace inequality_triangle_area_l94_94787

-- Define the triangles and their properties
variables {α β γ : Real} -- Internal angles of triangle ABC
variables {r : Real} -- Circumradius of triangle ABC
variables {P Q : Real} -- Areas of triangles ABC and A'B'C' respectively

-- Define the bisectors and intersect points
-- Note: For the purpose of this proof, we're not explicitly defining the geometry
-- of the inner bisectors and intersect points but working from the given conditions.

theorem inequality_triangle_area
  (h1 : P = r^2 * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) / 2)
  (h2 : Q = r^2 * (Real.sin (β + γ) + Real.sin (γ + α) + Real.sin (α + β)) / 2) :
  16 * Q^3 ≥ 27 * r^4 * P :=
sorry

end inequality_triangle_area_l94_94787


namespace min_tan_of_acute_angle_l94_94959

def is_ocular_ray (u : ℚ) (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ u = x / y

def acute_angle_tangent (u v : ℚ) : ℚ :=
  |(u - v) / (1 + u * v)|

theorem min_tan_of_acute_angle :
  ∃ θ : ℚ, (∀ u v : ℚ, (∃ x1 y1 x2 y2 : ℕ, is_ocular_ray u x1 y1 ∧ is_ocular_ray v x2 y2 ∧ u ≠ v) 
  → acute_angle_tangent u v ≥ θ) ∧ θ = 1 / 722 :=
sorry

end min_tan_of_acute_angle_l94_94959


namespace A_solution_B_solution_A_inter_B_A_union_B_l94_94437

open Set

-- Definitions
def A : Set ℝ := {x | x^2 + 2x - 3 < 0}
def B : Set ℝ := {x | x^2 - 4x - 5 < 0}

-- Theorems
theorem A_solution : A = Ioo (-3 : ℝ) 1 := by
  sorry

theorem B_solution : B = Ioo (-1 : ℝ) 5 := by
  sorry

theorem A_inter_B : A ∩ B = Ioo (-1 : ℝ) 1 := by
  sorry

theorem A_union_B : A ∪ B = Ioo (-3 : ℝ) 5 := by
  sorry

end A_solution_B_solution_A_inter_B_A_union_B_l94_94437


namespace part1_part2_l94_94115

-- Define the absolute value function
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

-- Given conditions
def condition1 : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6

def condition2 (a : ℝ) : Prop :=
  ∃ t m : ℝ, f (t / 2) a ≤ m - f (-t) a

-- Statements to prove
theorem part1 : ∃ a : ℝ, condition1 ∧ a = 1 := by
  sorry

theorem part2 : ∀ {a : ℝ}, a = 1 → ∃ m : ℝ, m ≥ 3.5 ∧ condition2 a := by
  sorry

end part1_part2_l94_94115


namespace susannah_swims_more_than_camden_l94_94024

-- Define the given conditions
def camden_total_swims : ℕ := 16
def susannah_total_swims : ℕ := 24
def number_of_weeks : ℕ := 4

-- State the theorem
theorem susannah_swims_more_than_camden :
  (susannah_total_swims / number_of_weeks) - (camden_total_swims / number_of_weeks) = 2 :=
by
  sorry

end susannah_swims_more_than_camden_l94_94024


namespace weight_each_Brandon_approx_l94_94956

noncomputable def weight_Jon_textbooks := 3.5 + 8 + 5 + 9
noncomputable def total_weight_Jon := weight_Jon_textbooks
noncomputable def total_weight_Brandon := total_weight_Jon / 3
noncomputable def number_of_Brandon_textbooks := 6
noncomputable def weight_each_Brandon := total_weight_Brandon / number_of_Brandon_textbooks

theorem weight_each_Brandon_approx : abs (weight_each_Brandon - 1.42) < 0.01 :=
by
  sorry

end weight_each_Brandon_approx_l94_94956


namespace circumcenter_AMN_on_AC_l94_94974

-- Definitions and Problem Statement
variables (A B C D M N : Point)
variable [Square ABCD]
variables [OnSeg BC M] [OnSeg CD N]
variables (h : ∠ (A, M, N) = 45)

-- Goal: The circumcenter of triangle AMN lies on AC
theorem circumcenter_AMN_on_AC : Circumcenter_Amn_on_AC A M N AC :=
sorry

end circumcenter_AMN_on_AC_l94_94974


namespace det_pow_eq_l94_94454

theorem det_pow_eq (M : Matrix n n ℝ) (h : det M = 3) : det (M^3) = 27 :=
by 
  sorry

end det_pow_eq_l94_94454


namespace josette_paid_correct_amount_l94_94178

-- Define the number of small and large bottles
def num_small_bottles : ℕ := 3
def num_large_bottles : ℕ := 2

-- Define the cost of each type of bottle
def cost_per_small_bottle : ℝ := 1.50
def cost_per_large_bottle : ℝ := 2.40

-- Define the total number of bottles purchased
def total_bottles : ℕ := num_small_bottles + num_large_bottles

-- Define the discount rate applicable when purchasing 5 or more bottles
def discount_rate : ℝ := 0.10

-- Calculate the initial total cost before any discount
def total_cost_before_discount : ℝ :=
  (num_small_bottles * cost_per_small_bottle) + 
  (num_large_bottles * cost_per_large_bottle)

-- Calculate the discount amount if applicable
def discount_amount : ℝ :=
  if total_bottles >= 5 then
    discount_rate * total_cost_before_discount
  else
    0

-- Calculate the final amount Josette paid after applying any discount
def final_amount_paid : ℝ :=
  total_cost_before_discount - discount_amount

-- Prove that the final amount paid is €8.37
theorem josette_paid_correct_amount :
  final_amount_paid = 8.37 :=
by
  sorry

end josette_paid_correct_amount_l94_94178


namespace compute_expression_l94_94715

theorem compute_expression : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by sorry

end compute_expression_l94_94715


namespace distribution_schemes_four_students_l94_94797

theorem distribution_schemes_four_students : 
  ∃ (unitA unitB : Finset ℕ),  -- Define two units, both sets of students
  card unitA > 0 ∧ card(unitB) > 0 ∧ -- Each unit must have at least one student
  (∀ x, x ∈ unitA ∨ x ∈ unitB) ∧ -- All students belong to either Unit A or Unit B
  (card unitA + card unitB = 4) ∧ -- Total number of students is 4
  (card (Finset.powerset (Finset.range 4)) = 14) -- The number of ways to partition them
:= by
  sorry

end distribution_schemes_four_students_l94_94797


namespace Coles_payment_l94_94030

-- Conditions
def side_length : ℕ := 9
def back_length : ℕ := 18
def cost_per_foot : ℕ := 3
def back_neighbor_contrib (length : ℕ) : ℕ := (length / 2) * cost_per_foot
def left_neighbor_contrib (length : ℕ) : ℕ := (length / 3) * cost_per_foot

-- Main problem statement
theorem Coles_payment : 
  let total_fence_length := 2 * side_length + back_length,
      total_cost := total_fence_length * cost_per_foot,
      contribution_back_neighbor := back_neighbor_contrib back_length,
      contribution_left_neighbor := left_neighbor_contrib side_length,
      coles_contribution := total_cost - (contribution_back_neighbor + contribution_left_neighbor)
  in coles_contribution = 72 := by
  sorry

end Coles_payment_l94_94030


namespace option_d_may_not_hold_l94_94077

theorem option_d_may_not_hold (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m^2 * a > m^2 * b) :=
sorry

end option_d_may_not_hold_l94_94077


namespace number_of_values_x0_eq_x5_l94_94400

open Nat

variable (x0 : ℝ)
variable (x : ℕ → ℝ)

axiom h0 : 0 ≤ x0 ∧ x0 < 1
axiom h1 : ∀ n > 0, (x n = if 2 * x (n - 1) < 1 then 2 * x (n - 1) else 2 * x (n - 1) - 1)
axiom h2 : x 0 = x0

theorem number_of_values_x0_eq_x5 : (finset.range 32).filter (λ n, x0 = x 5).card = 31 := 
sorry

end number_of_values_x0_eq_x5_l94_94400


namespace monotonicity_of_f_l94_94647

def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x - 3 / (a * x)

theorem monotonicity_of_f (a : ℝ) (x : ℝ) (hx : x > 0) : 
    (if a > 0 then (∀ y, 0 < y ∧ y < 3/a → f y a = increasing) ∧ 
            (∀ y, y > 3/a → f y a = decreasing)
     else if a < 0 then (∀ y, 0 < y ∧ y < -1/a → f y a = decreasing) ∧ 
            (∀ y, y > -1/a → f y a = increasing)
     else false
    ) sorry

end monotonicity_of_f_l94_94647


namespace exists_route_same_republic_l94_94896

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l94_94896


namespace remainder_of_7_pow_308_mod_11_l94_94617

theorem remainder_of_7_pow_308_mod_11 :
  (7 ^ 308) % 11 = 9 :=
by
  sorry

end remainder_of_7_pow_308_mod_11_l94_94617


namespace log_base_8_14_in_terms_of_a_l94_94975

theorem log_base_8_14_in_terms_of_a (a : ℝ) (h : Real.logBase 14 16 = a) : 
  Real.logBase 8 14 = 4 / (3 * a) :=
by
  sorry

end log_base_8_14_in_terms_of_a_l94_94975


namespace find_neighbors_of_6_l94_94199

-- Define the predicate for a "beautiful" face
def beautiful_face (a b c d : ℕ) : Prop :=
  a = b + c + d ∨ b = a + c + d ∨ c = a + b + d ∨ d = a + b + c

-- Define the problem condition: Set 1 to 8 are written on the cube's vertices
def cube_vertices := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the three numbers at the ends of the edges emanating from the vertex with number 6
def neighbors_of_6 (a b c : ℕ) : Prop :=
  (a, b, c).subseteq cube_vertices ∧ (beautiful_face 6 1 2 3) ∧ (beautiful_face 7 1 2 4) ∧ (beautiful_face 8 1 2 5 ∨ beautiful_face 8 1 3 4)

theorem find_neighbors_of_6 :
  ∃ a b c : ℕ, neighbors_of_6 a b c ∧ (a, b, c) = (2, 3, 5) :=
  sorry

end find_neighbors_of_6_l94_94199


namespace school_stats_l94_94487

-- Defining the conditions
def girls_grade6 := 315
def boys_grade6 := 309
def girls_grade7 := 375
def boys_grade7 := 341
def drama_club_members := 80
def drama_club_boys_percent := 30 / 100

-- Calculate the derived numbers
def students_grade6 := girls_grade6 + boys_grade6
def students_grade7 := girls_grade7 + boys_grade7
def total_students := students_grade6 + students_grade7
def drama_club_boys := drama_club_boys_percent * drama_club_members
def drama_club_girls := drama_club_members - drama_club_boys

-- Theorem
theorem school_stats :
  total_students = 1340 ∧
  drama_club_girls = 56 ∧
  boys_grade6 = 309 ∧
  boys_grade7 = 341 :=
by
  -- We provide the proof steps inline with sorry placeholders.
  -- In practice, these would be filled with appropriate proofs.
  sorry

end school_stats_l94_94487


namespace units_digit_largest_mersenne_prime_l94_94307

theorem units_digit_largest_mersenne_prime :
  ∀ (n : ℕ), n = 74207281 → 
  let m := 2^n - 1 in 
  m % 10 = 1 :=
by
  intro n h
  rw h
  let m := 2^74207281 - 1
  have : 74207281 % 4 = 1 := by sorry
  have : 2^74207281 % 10 = 2 := by sorry
  exact Nat.sub_mod_eq_sub_mod _ 1 _ -- ∵ 2^n % 10 = 2 ⇏ Mersenne prime's last digit is 1
  have : (2 - 1) % 10 = 1 := by sorry
  rw [mod_eq_sub_neg_mod] at this
  exact this

end units_digit_largest_mersenne_prime_l94_94307


namespace count_two_digit_perfect_squares_divisible_by_4_l94_94862

theorem count_two_digit_perfect_squares_divisible_by_4 :
  let nums := [4, 5, 6, 7, 8, 9]
  let squares := nums.map (λ n, n * n)
  let squares_div_by_4 := squares.filter (λ x, x % 4 = 0)
  squares_div_by_4.length = 3 :=
by
  let nums := [4, 5, 6, 7, 8, 9]
  let squares := nums.map (λ n, n * n)
  let squares_div_by_4 := squares.filter (λ x, x % 4 = 0)
  exact sorry

end count_two_digit_perfect_squares_divisible_by_4_l94_94862


namespace sum_of_digits_repeating_decimal_l94_94035

theorem sum_of_digits_repeating_decimal :
  let m := 400
  let digits : ℕ → ℕ := fun n => if n % 40 == 0 then 0 else (n % 10)
  let repeating_sequence := List.range m
  ∑ i in repeating_sequence, digits i = 450 :=
by
  sorry

end sum_of_digits_repeating_decimal_l94_94035


namespace susannah_swims_more_than_camden_l94_94025

-- Define the given conditions
def camden_total_swims : ℕ := 16
def susannah_total_swims : ℕ := 24
def number_of_weeks : ℕ := 4

-- State the theorem
theorem susannah_swims_more_than_camden :
  (susannah_total_swims / number_of_weeks) - (camden_total_swims / number_of_weeks) = 2 :=
by
  sorry

end susannah_swims_more_than_camden_l94_94025


namespace rhombus_area_l94_94933

theorem rhombus_area (A B C D : (ℝ × ℝ)) 
  (hA : A = (0, 3.5)) 
  (hB : B = (10, 0)) 
  (hC : C = (0, -3.5))
  (hD : D = (-10, 0)) :
  let d1 := dist A C in
  let d2 := dist B D in
  (d1 * d2) / 2 = 70 :=
by 
  dsimp at *,
  sorry

end rhombus_area_l94_94933


namespace range_g_l94_94686

open Real

def g (x : ℝ) : ℝ := sin x ^ 3 + cos x ^ 2

theorem range_g : ∀ y : ℝ, y ∈ set.range g → -1 ≤ y ∧ y ≤ 1 := by
  sorry

end range_g_l94_94686


namespace triangle_segment_ratio_l94_94082

theorem triangle_segment_ratio
  (A B C K L M N U : Point)
  (hK : midpoint A C K)
  (hU : midpoint B C U)
  (hL : on_segment L C K)
  (hM : on_segment M C U)
  (hLM_parallel_KU : parallel LM KU)
  (hN : ratio_segment A B N (3/10))
  (area_ratio : area_polygon [U, M, L, K] / area_polygon [M, L, K, N, U] = 3/7) :
  segment_ratio LM KU = 1/2 :=
sorry

end triangle_segment_ratio_l94_94082


namespace count_two_digit_perfect_squares_divisible_by_4_l94_94863

theorem count_two_digit_perfect_squares_divisible_by_4 :
  let nums := [4, 5, 6, 7, 8, 9]
  let squares := nums.map (λ n, n * n)
  let squares_div_by_4 := squares.filter (λ x, x % 4 = 0)
  squares_div_by_4.length = 3 :=
by
  let nums := [4, 5, 6, 7, 8, 9]
  let squares := nums.map (λ n, n * n)
  let squares_div_by_4 := squares.filter (λ x, x % 4 = 0)
  exact sorry

end count_two_digit_perfect_squares_divisible_by_4_l94_94863


namespace smallest_prime_with_prime_reverse_l94_94383

-- Define a predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define a predicate to reverse the digits of a number
def reverse_digits (n : ℕ) : ℕ :=
let digits := n.toString.data.rev;
return (digits.foldl (λ acc c => acc * 10 + c.toNat) 0)

theorem smallest_prime_with_prime_reverse :
    ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ is_prime n ∧ is_prime (reverse_digits n) ∧
    ∀ m, 100 ≤ m ∧ m < 1000 ∧ is_prime m ∧ is_prime (reverse_digits m) → n ≤ m :=
    -- Provide exact number and reasons
    by
    sorry

end smallest_prime_with_prime_reverse_l94_94383


namespace count_three_digit_even_numbers_with_4_and_5_l94_94443

-- Definitions
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def contains_digit (n digit : ℕ) : Prop :=
  ∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 = digit ∨ d2 = digit ∨ d3 = digit)

-- Main theorem
theorem count_three_digit_even_numbers_with_4_and_5 : 
  ∃! n, is_three_digit n ∧ is_even n ∧ contains_digit n 4 ∧ contains_digit n 5 := 94 :=
sorry

end count_three_digit_even_numbers_with_4_and_5_l94_94443


namespace find_interest_rate_for_second_year_l94_94741

theorem find_interest_rate_for_second_year 
  (P A : ℕ) (r1 r2 : ℚ) (hP : P = 5000) (hA : A = 6160) (hr1 : r1 = 10) :
  A = P * (1 + r1 / 100) * (1 + r2 / 100) → r2 = 12 := by
  intros h
  dsimp at h
  sorry

end find_interest_rate_for_second_year_l94_94741


namespace sum_cardinality_correct_l94_94181

def A_i (i : ℕ) : set (fin 2002) := sorry

noncomputable def F : set (fin 2002 → set (fin 2002)) := {f | ∀ i, f i ⊆ (fin 2002)}

def sum_cardinality (n : ℕ) : ℕ :=
  finset.sum (finset.univ : finset (fin 2002 → set (fin 2002)))
             (λ f, (f (λ i : fin n, A_i i)).card)

theorem sum_cardinality_correct (n : ℕ) :
  sum_cardinality n = 2002 * (2^(2002 * n) - 2^(2001 * n)) :=
sorry

end sum_cardinality_correct_l94_94181


namespace simplify_A_minus_B_value_of_A_minus_B_given_condition_l94_94073

variable (a b : ℝ)

def A := (a + b) ^ 2 - 3 * b ^ 2
def B := 2 * (a + b) * (a - b) - 3 * a * b

theorem simplify_A_minus_B :
  A a b - B a b = -a ^ 2 + 5 * a * b :=
by sorry

theorem value_of_A_minus_B_given_condition :
  (a - 3) ^ 2 + |b - 4| = 0 → A a b - B a b = 51 :=
by sorry

end simplify_A_minus_B_value_of_A_minus_B_given_condition_l94_94073


namespace initial_red_martians_l94_94212

/-- Red Martians always tell the truth, while Blue Martians lie and then turn red.
    In a group of 2018 Martians, they answered in the sequence 1, 2, 3, ..., 2018 to the question
    of how many of them were red at that moment. Prove that the initial number of red Martians was 0 or 1. -/
theorem initial_red_martians (N : ℕ) (answers : Fin (N+1) → ℕ) :
  (∀ i : Fin (N+1), answers i = i.succ) → N = 2018 → (initial_red_martians_count = 0 ∨ initial_red_martians_count = 1)
:= sorry

end initial_red_martians_l94_94212


namespace exists_airline_route_within_same_republic_l94_94912

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l94_94912


namespace minimal_bN_l94_94063

theorem minimal_bN (N : ℕ) (hN : 0 < N) : 
  ∃ b_N : ℝ, (∀ x : ℝ, (sqrtNthRoot N ((x^(2*N) + 1) / 2) ≤ b_N * (x - 1)^2 + x)) ∧ b_N = N / 2 :=
sorry

noncomputable def sqrtNthRoot (N : ℕ) (x : ℝ) : ℝ := 
  Real.sqrt (x^(1 / N : ℝ))

end minimal_bN_l94_94063


namespace min_students_all_right_l94_94882

/-- Problem statement: The minimum number of students who got all four questions right is calculated as follows:
There are 45 students in total. 
- 35 students got the first question right,
- 27 students got the second question right, 
- 41 students got the third question right,
- 38 students got the fourth question right.
Prove that the minimum number of students who got all four questions right is 6.
--/
theorem min_students_all_right
  (total_students : ℕ)
  (q1_right : ℕ)
  (q2_right : ℕ)
  (q3_right : ℕ)
  (q4_right : ℕ)
  (h_total : total_students = 45)
  (h_q1 : q1_right = 35)
  (h_q2 : q2_right = 27)
  (h_q3 : q3_right = 41)
  (h_q4 : q4_right = 38)
 : ∃ (s : ℕ), s = 6 :=
begin
  sorry
end

end min_students_all_right_l94_94882


namespace range_of_m_l94_94191

-- Define the conditions for p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
  (x₁^2 + 2 * m * x₁ + 1 = 0) ∧ (x₂^2 + 2 * m * x₂ + 1 = 0)

def q (m : ℝ) : Prop := ¬ ∃ x : ℝ, x^2 + 2 * (m-2) * x - 3 * m + 10 = 0

-- The main theorem
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ 
  (m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)) := 
by
  sorry

end range_of_m_l94_94191


namespace minh_trip_duration_l94_94533

-- Defining constants and given conditions
def distance_interstate : ℝ := 120
def distance_mountain : ℝ := 20
def distance_city : ℝ := 15
def ratio_interstate_mountain : ℝ := 4
def ratio_interstate_city : ℝ := 2
def time_mountain : ℝ := 40

-- Minh's speeds on different roads
def speed_mountain := distance_mountain / time_mountain
def speed_interstate := ratio_interstate_mountain * speed_mountain
def speed_city := speed_interstate / ratio_interstate_city

-- Calculating times based on speeds
def time_interstate := distance_interstate / speed_interstate
def time_city := distance_city / speed_city

-- Total time for the trip
def total_time := time_mountain + time_interstate + time_city

-- Statement to prove
theorem minh_trip_duration : total_time = 115 := sorry

end minh_trip_duration_l94_94533


namespace arcsin_cos_eq_neg_pi_14_l94_94735

theorem arcsin_cos_eq_neg_pi_14 : 
  ∀ (x : ℝ), x = (4 * Real.pi) / 7 → Real.arcsin (Real.cos x) = - (Real.pi / 14) :=
by
  intros x hx
  have hcos : Real.cos x = Real.sin (- (Real.pi / 14)) := by sorry
  have h_sin_neg : Real.sin (-(Real.pi / 14)) = - Real.sin (Real.pi / 14) := by sorry
  have : Real.sin (Real.arcsin (Real.cos x)) = Real.cos x := by sorry
  rw [hx, hcos, h_sin_neg] at this
  sorry

end arcsin_cos_eq_neg_pi_14_l94_94735


namespace find_constants_and_extrema_l94_94226

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem find_constants_and_extrema (a b c : ℝ) : 
  (∃ (f' : ℝ → ℝ), f' = fun x => 3 * a * x^2 + 2 * b * x + c) →
  (f' (-1) = 0) →
  (f' 1 = 0) →
  (f a b c (-1) = 1) →
  (∃ (a' b' c' : ℝ), a' = 1/2 ∧ b' = 0 ∧ c' = -3/2 ∧
  f a' b' c' (-1) = 1 ∧ f a' b' c' (1) = -1 ∧
  ∀ x : ℝ, 
    (x < -1 → ∀ y : ℝ, x < y → y < -1 → f a' b' c' x < f a' b' c' y) ∧
    (-1 < x ∧ x < 1 → ∀ y : ℝ, x < y → y < 1 → f a' b' c' x > f a' b' c' y) ∧
    (1 < x → ∀ y : ℝ, x < y → f a' b' c' x < f a' b' c' y))) :=
sorry

end find_constants_and_extrema_l94_94226


namespace number_of_ways_to_connect_gates_l94_94945

-- Defining the Catalan number
def catalan (n : ℕ) : ℕ := (2 * n).choose n / (n + 1)

-- Theorem stating the number of ways to connect 2n gates with non-intersecting paths
theorem number_of_ways_to_connect_gates (n : ℕ) :
  number_of_ways_to_connect_gates n = catalan n := 
sorry

end number_of_ways_to_connect_gates_l94_94945


namespace exists_airline_route_within_same_republic_l94_94917

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l94_94917


namespace students_with_all_three_pets_correct_l94_94937

noncomputable def students_with_all_three_pets (total_students dog_owners cat_owners bird_owners dog_and_cat_owners cat_and_bird_owners dog_and_bird_owners : ℕ) : ℕ :=
  total_students - (dog_owners + cat_owners + bird_owners - dog_and_cat_owners - cat_and_bird_owners - dog_and_bird_owners)

theorem students_with_all_three_pets_correct : 
  students_with_all_three_pets 50 30 35 10 8 5 3 = 7 :=
by
  rw [students_with_all_three_pets]
  norm_num
  sorry

end students_with_all_three_pets_correct_l94_94937


namespace number_of_integer_points_not_on_line_is_one_l94_94493

noncomputable def number_of_integer_points_not_on_line (θ : ℝ) : ℕ :=
  let line (x y : ℝ) := x * Real.cos θ + y * Real.sin θ = 1
  let is_integer_point (x y : ℤ) := ∀ (x : ℝ), ∀ (y : ℝ), ∃ n m : ℤ, x = n ∧ y = m
  let int_points_not_on_line := 
    { p : (ℤ × ℤ) | ¬ line (p.1 : ℝ) (p.2 : ℝ) }
      ∩ { p : (ℤ × ℤ) | (↑p.1)^2 + (↑p.2)^2 < 1 }
  in int_points_not_on_line.to_finset.card

theorem number_of_integer_points_not_on_line_is_one (θ : ℝ) :
  number_of_integer_points_not_on_line θ = 1 := 
sorry

end number_of_integer_points_not_on_line_is_one_l94_94493


namespace probability_three_tails_one_head_l94_94460

theorem probability_three_tails_one_head :
  let outcome_probability := (1 / 2) ^ 4 in
  let combinations := Finset.card (Finset.filter (λ (s : Finset.Fin 4 × Bool), s.2).Finset (Finset.product (Finset.range 4) (Finset.singleton (false, true, true, true)))) in
  outcome_probability * combinations = 1 / 4 :=
sorry

end probability_three_tails_one_head_l94_94460


namespace beautiful_function_conditions_l94_94322

noncomputable def isBeautifulFunction (f : ℝ → ℝ) (D : Set ℝ) :=
  (∀ x y ∈ D, x ≤ y → f x ≤ f y) ∧ ∃ a b : ℝ, ∃ I : Set ℝ, I ⊆ D ∧ I = Set.Icc (a / 2) (b / 2) ∧ ∀ x ∈ I, f x ∈ Set.Icc a b

def f (c x t : ℝ) : ℝ := Real.log c (c^x - t)

theorem beautiful_function_conditions
  (c t : ℝ) (h₀ : 0 < c) (h₁ : c ≠ 1) :
  (0 < t ∧ t < 1 / 4) ↔ isBeautifulFunction (f c · t) Set.univ :=
sorry

end beautiful_function_conditions_l94_94322


namespace solution_eq1_solution_eq2_l94_94222

theorem solution_eq1 (x : ℝ) : 
  2 * x^2 - 4 * x - 1 = 0 ↔ 
  (x = 1 + (Real.sqrt 6) / 2 ∨ x = 1 - (Real.sqrt 6) / 2) := by
sorry

theorem solution_eq2 (x : ℝ) :
  (x - 1) * (x + 2) = 28 ↔ 
  (x = -6 ∨ x = 5) := by
sorry

end solution_eq1_solution_eq2_l94_94222


namespace find_n_l94_94051

def is_geometric_sequence (seq : List ℕ) : Prop :=
  ∀ i j, i < j → seq.nth i * seq.nth (j - i) = (seq.nth j) ^ 2

def has_at_least_four_divisors (n : ℕ) : Prop :=
  (finset.divisors n).card ≥ 4

def satisfies_conditions (n : ℕ) : Prop :=
  has_at_least_four_divisors n ∧
  ∃ k (d : ℕ → ℕ), (∀ i, d i ∈ finset.divisors n) ∧ 
  sorted (≤) (seq.to_list d) ∧ 
  is_geometric_sequence (seq.to_list (λ i, d (i + 1) - d i))

theorem find_n (n : ℕ) (p : ℕ → Prop) (a : ℕ) :
  satisfies_conditions n →
  ∃ p a, (Prime p) ∧ (a ≥ 3) ∧ (n = p ^ a) :=
begin
  sorry
end

end find_n_l94_94051


namespace probability_Q_within_three_units_of_origin_l94_94670

noncomputable def probability_within_three_units_of_origin :=
  let radius := 3
  let square_side := 10
  let circle_area := Real.pi * radius^2
  let square_area := square_side^2
  circle_area / square_area

theorem probability_Q_within_three_units_of_origin :
  probability_within_three_units_of_origin = 9 * Real.pi / 100 :=
by
  -- Since this proof is not required, we skip it with sorry.
  sorry

end probability_Q_within_three_units_of_origin_l94_94670


namespace hyperbola_asymptotes_l94_94123

theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (e : ℝ) (he : e = Real.sqrt 3) (h_eq : e = Real.sqrt ((a^2 + b^2) / a^2)) :
  (∀ x : ℝ, y = x * Real.sqrt 2) :=
by
  sorry

end hyperbola_asymptotes_l94_94123


namespace razorback_shop_jersey_price_l94_94229

variable (J : ℕ)
variable (amount_from_jerseys : ℕ := 152)
variable (number_of_jerseys_sold : ℕ := 2)

theorem razorback_shop_jersey_price :
  number_of_jerseys_sold * J = amount_from_jerseys → J = 76 :=
by
  intro h
  have hJ : J = amount_from_jerseys / number_of_jerseys_sold := sorry
  rw [hJ]
  norm_num

end razorback_shop_jersey_price_l94_94229


namespace race_problem_l94_94879

theorem race_problem
  (total_distance : ℕ)
  (A_time : ℕ)
  (B_extra_time : ℕ)
  (A_speed B_speed : ℕ)
  (A_distance B_distance : ℕ)
  (H1 : total_distance = 120)
  (H2 : A_time = 8)
  (H3 : B_extra_time = 7)
  (H4 : A_speed = total_distance / A_time)
  (H5 : B_speed = total_distance / (A_time + B_extra_time))
  (H6 : A_distance = total_distance)
  (H7 : B_distance = B_speed * A_time) :
  A_distance - B_distance = 56 := 
sorry

end race_problem_l94_94879


namespace exists_route_same_republic_l94_94891

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l94_94891


namespace range_sin_cos_sum_l94_94818

theorem range_sin_cos_sum (θ : ℝ) (h1 : π / 2 < θ ∧ θ < π) (h2 : sin (θ / 2) < cos (θ / 2)) : 
  -sqrt 2 < sin (θ / 2) + cos (θ / 2) ∧ sin (θ / 2) + cos (θ / 2) < -1 := 
sorry

end range_sin_cos_sum_l94_94818


namespace expand_expression_l94_94370

theorem expand_expression : ∀ (x : ℝ), (20 * x - 25) * 3 * x = 60 * x^2 - 75 * x := 
by
  intro x
  sorry

end expand_expression_l94_94370


namespace count_three_digit_numbers_no_5_no_8_l94_94444

theorem count_three_digit_numbers_no_5_no_8 : 
  (number_of_valid_three_digit_numbers {d | d ≠ 0 ∧ d ≠ 5 ∧ d ≠ 8} {d | d ≠ 5 ∧ d ≠ 8} = 448) :=
sorry

def number_of_valid_three_digit_numbers (hundreds_set tens_units_set : set ℕ) : ℕ :=
  (hundreds_set.to_finset.card) * (tens_units_set.to_finset.card) * (tens_units_set.to_finset.card)

example : number_of_valid_three_digit_numbers {d | d ≠ 0 ∧ d ≠ 5 ∧ d ≠ 8} {d | d ≠ 5 ∧ d ≠ 8} = 448 :=
sorry

end count_three_digit_numbers_no_5_no_8_l94_94444


namespace find_a8_l94_94241

noncomputable def a_sequence : ℕ → ℕ
| 0       := 8
| 1       := 10
| (n + 2) := a_sequence (n + 1) + a_sequence n

theorem find_a8 :
  a_sequence 7 = 120 → a_sequence 8 = 194 :=
by
  sorry

end find_a8_l94_94241


namespace counterfeit_probability_correct_l94_94687

noncomputable def calc_probability (
  P_C : ℝ,
  P_R : ℝ,
  P_L : ℝ,
  P_L_C : ℝ,
  P_L_R : ℝ,
  P_T_counterfeit : ℝ,
  P_T_real : ℝ
) : ℝ :=
  let P_T := P_T_counterfeit * P_C * P_L_C + P_T_real * P_R * P_L_R in
  (P_T_counterfeit * P_C) / P_T

theorem counterfeit_probability_correct :
  calc_probability
    (1 / 100)     -- P(C)
    (99 / 100)    -- P(R)
    0.05          -- P(L)
    1             -- P(L | C)
    0.05          -- P(L | R)
    0.90          -- P(T | counterfeit)
    0.10          -- P(T | real)
  = 19 / 28 :=
sorry

end counterfeit_probability_correct_l94_94687


namespace sin_48_gt_cos_48_l94_94714

theorem sin_48_gt_cos_48 : real.sin (real.pi * 48 / 180) > real.cos (real.pi * 48 / 180) :=
by {
  have h1 : real.sin (real.pi * 48 / 180) = real.cos (real.pi * (90 - 48) / 180),
  { rw [real.sin_eq_cos_pi_div_two_sub], },
  have h2 : real.cos (real.pi * (90 - 48) / 180) > real.cos (real.pi * 48 / 180),
  { norm_num, },
  rw h1,
  exact h2,
}

end sin_48_gt_cos_48_l94_94714


namespace range_of_m_l94_94094

open Set

noncomputable def setA : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
noncomputable def setB (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ y = (1 / 3) * x + m}

theorem range_of_m {m : ℝ} (p q : Prop) :
  p ↔ ∃ x : ℝ, x ∈ setA →
  q ↔ ∃ x : ℝ, x ∈ setB m →
  ((p → q) ∧ ¬(q → p)) ↔ (1 / 3 < m ∧ m < 2 / 3) :=
by
  sorry

end range_of_m_l94_94094


namespace circle_radius_eq_one_l94_94781

theorem circle_radius_eq_one (x y : ℝ) : (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → (1 = 1) :=
by
  intros h
  sorry

end circle_radius_eq_one_l94_94781


namespace problem_statement_l94_94410

variable (a1 b1 c1 a2 b2 c2 : ℝ)

def P : Prop := ∀ x : ℝ, (a1 * x^2 + b1 * x + c1 > 0) ↔ (a2 * x^2 + b2 * x + c2 > 0)
def Q : Prop := a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2

theorem problem_statement : ¬ (Q ↔ P) :=
  sorry

end problem_statement_l94_94410


namespace cube_edges_after_cuts_l94_94733

theorem cube_edges_after_cuts (V E : ℕ) (hV : V = 8) (hE : E = 12) : 
  12 + 24 = 36 := by
  sorry

end cube_edges_after_cuts_l94_94733


namespace increase_by_20_percent_l94_94311

theorem increase_by_20_percent (initial : ℕ) (percentage : ℕ) (final : ℕ) 
    (h_initial : initial = 240) (h_percentage : percentage = 20) :
    final = initial + (initial * percentage / 100) := by
  have h1 : initial * percentage / 100 = 48 := by sorry
  have h2 : initial + 48 = 288 := by sorry
  show final = 288 from sorry

end increase_by_20_percent_l94_94311


namespace num_parallel_edges_rect_prism_l94_94133

-- Definition of a rectangular prism where each pair of opposite faces is unequal
structure RectangularPrism :=
  (length width height : ℕ)
  (hne1 : length ≠ width)
  (hne2 : width ≠ height)
  (hne3 : length ≠ height)

-- The theorem statement
theorem num_parallel_edges_rect_prism (rp : RectangularPrism) : 
  (fun num_pairs_parallel_edges(prism: RectangularPrism) =>
     2 * ((prism.length * prism.width) + (prism.width * prism.height) + (prism.height * prism.length))/2 = 12) rp := 
  sorry

end num_parallel_edges_rect_prism_l94_94133


namespace overall_gain_percentage_l94_94346

theorem overall_gain_percentage (cost_A cost_B cost_C sp_A sp_B sp_C : ℕ)
  (hA : cost_A = 1000)
  (hB : cost_B = 3000)
  (hC : cost_C = 6000)
  (hsA : sp_A = 2000)
  (hsB : sp_B = 4500)
  (hsC : sp_C = 8000) :
  ((sp_A + sp_B + sp_C - (cost_A + cost_B + cost_C) : ℝ) / (cost_A + cost_B + cost_C) * 100) = 45 :=
by sorry

end overall_gain_percentage_l94_94346


namespace people_dislike_both_radio_and_music_l94_94537

theorem people_dislike_both_radio_and_music (N : ℕ) (p_r p_rm : ℝ) (hN : N = 2000) (hp_r : p_r = 0.25) (hp_rm : p_rm = 0.15) : 
  N * p_r * p_rm = 75 :=
by {
  sorry
}

end people_dislike_both_radio_and_music_l94_94537


namespace find_y_coordinate_of_third_vertex_l94_94012

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def is_equilateral (p1 p2 p3 : ℝ × ℝ) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1 ∧ distance p3 p1 = distance p1 p2

noncomputable def altitude (side : ℝ) : ℝ := (side / 2) * real.sqrt 3

theorem find_y_coordinate_of_third_vertex 
  (p1 p2 : ℝ × ℝ) (h1 : p1 = (2, 3)) (h2 : p2 = (10, 3)) (third_vertex : ℝ × ℝ) 
  (h3 : is_equilateral p1 p2 third_vertex) (h4 : third_vertex.1 > 0) (h5 : third_vertex.2 > 0) :
  third_vertex.2 = 3 + 4 * real.sqrt 3 :=
sorry

end find_y_coordinate_of_third_vertex_l94_94012


namespace max_sum_of_two_numbers_in_set_min_sum_of_two_numbers_in_set_l94_94027

def number_set : Set ℤ := {-5, 1, -3, 6, -2}

theorem max_sum_of_two_numbers_in_set : 
  ∃ a b ∈ number_set, a + b = 7 := by 
  sorry

theorem min_sum_of_two_numbers_in_set : 
  ∃ a b ∈ number_set, a + b = -8 := by 
  sorry

end max_sum_of_two_numbers_in_set_min_sum_of_two_numbers_in_set_l94_94027


namespace postage_cost_l94_94792

theorem postage_cost (weight : ℝ) (h₁ : weight ≤ 100) (h₂ : weight ≥ 0) : 
  weight = 72.5 → postage weight = 2.4 := 
by 
  sorry

def postage (weight : ℝ) : ℝ :=
  if weight ≤ 20 then 0.6
  else if weight ≤ 40 then 1.2
  else (0.6 + (weight - 40) / 20 * 0.6)

# Examples to illustrate the function definition
-- postage 0  = 0.6
-- postage 20 = 0.6
-- postage 25 = 1.2
-- postage 45 = 1.5
-- postage 72.5 = 2.4

end postage_cost_l94_94792


namespace transform_v0_to_v2_l94_94515

noncomputable def projection_matrix (u : ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let norm_sq := u.1 * u.1 + u.2 * u.2 in
  (1 / norm_sq) • (matrix.col u ⬝ matrix.row u)

theorem transform_v0_to_v2 :
  let u1 := (4, 1) in
  let u2 := (2, 1) in
  let P1 := projection_matrix u1 in
  let P2 := projection_matrix u2 in
  P2 ⬝ P1 = (λ i j, [ [76 / 85, 20 / 85], [24 / 85, 8 / 85] ].transpose i j) :=
by sorry

end transform_v0_to_v2_l94_94515


namespace gcf_factorial_seven_eight_l94_94762

theorem gcf_factorial_seven_eight (a b : ℕ) (h : a = 7! ∧ b = 8!) : Nat.gcd a b = 7! := 
by 
  sorry

end gcf_factorial_seven_eight_l94_94762


namespace triangle_existence_l94_94402

theorem triangle_existence (n : ℕ) (h : 2 * n > 0) (segments : Finset (ℕ × ℕ))
  (h_segments : segments.card = n^2 + 1)
  (points_in_segment : ∀ {a b : ℕ}, (a, b) ∈ segments → a < 2 * n ∧ b < 2 * n) :
  ∃ x y z, x < 2 * n ∧ y < 2 * n ∧ z < 2 * n ∧ (x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  ((x, y) ∈ segments ∨ (y, x) ∈ segments) ∧
  ((y, z) ∈ segments ∨ (z, y) ∈ segments) ∧
  ((z, x) ∈ segments ∨ (x, z) ∈ segments) :=
by
  sorry

end triangle_existence_l94_94402


namespace simplify_fraction_l94_94047

theorem simplify_fraction (b x : ℝ) :
  (sqrt (b^2 + x^4) - (x^4 - b^2) / (2 * sqrt (b^2 + x^4))) / (b^2 + x^4)
  = (3 * b^2 + x^4) / (2 * (b^2 + x^4)^(3/2)) :=
by
  sorry

end simplify_fraction_l94_94047


namespace solve_for_x_l94_94140

theorem solve_for_x (x : ℝ) (hx : log 5 (2 * x^2 - 7 * x + 12) = 2) : 
  x = (7 + Real.sqrt 153) / 4 ∨ x = (7 - Real.sqrt 153) / 4 := 
  sorry

end solve_for_x_l94_94140


namespace triangle_inequality_l94_94078

theorem triangle_inequality
  (n q : ℕ)
  (h_no_four_coplanar : ∀ (S : Finset (EuclideanSpace ℝ (Fin n))), S.card = 4 → ¬(Geometry.coplanar ⊤ S.id)) :
  q ≥ 0 → 
  (∃ (segments : Finset (Fin n × Fin n)), segments.card = q) →
  ∃ (triangles : Finset (Fin n × Fin n × Fin n)), 
    triangles.card ≥ (4 * q / (3 * n)) * (q - n^2 / 4) :=
by
  sorry

end triangle_inequality_l94_94078


namespace polynomials_equal_if_integer_parts_equal_l94_94995

theorem polynomials_equal_if_integer_parts_equal (f g : ℚ[X])
  (hf : f.degree = 2)
  (hg : g.degree = 2)
  (h : ∀ x : ℝ, ⌊f.eval x⌋ = ⌊g.eval x⌋) : f = g :=
sorry

end polynomials_equal_if_integer_parts_equal_l94_94995


namespace min_sum_of_squares_l94_94805

variables {α : Type*}

theorem min_sum_of_squares (P : α → α → Prop) (a b : α) :
  P (3 + 2 * real.cos a) (4 + 2 * real.sin b) ∧
    ∀ x y, P x y ↔ (x - 3) ^ 2 + (y - 4) ^ 2 = 4 →
    (2 * x ^ 2 + 2 * y ^ 2 + 8) ≥ 26 :=
begin
  sorry -- proof skipped
end

end min_sum_of_squares_l94_94805


namespace cameron_books_ratio_l94_94690

theorem cameron_books_ratio (Boris_books : ℕ) (Cameron_books : ℕ)
  (Boris_after_donation : ℕ) (Cameron_after_donation : ℕ)
  (total_books_after_donation : ℕ) (ratio : ℚ) :
  Boris_books = 24 → 
  Cameron_books = 30 → 
  Boris_after_donation = Boris_books - (Boris_books / 4) →
  total_books_after_donation = 38 →
  Cameron_after_donation = total_books_after_donation - Boris_after_donation →
  ratio = (Cameron_books - Cameron_after_donation) / Cameron_books →
  ratio = 1 / 3 :=
by
  -- Proof goes here.
  sorry

end cameron_books_ratio_l94_94690


namespace sum_of_star_tip_angles_l94_94045

noncomputable def sum_star_tip_angles : ℝ :=
  let segment_angle := 360 / 8
  let subtended_arc := 3 * segment_angle
  let theta := subtended_arc / 2
  8 * theta

theorem sum_of_star_tip_angles:
  sum_star_tip_angles = 540 := by
  sorry

end sum_of_star_tip_angles_l94_94045


namespace sweets_distribution_l94_94534

theorem sweets_distribution (S : ℕ) (N : ℕ) (h1 : N - 70 > 0) (h2 : S = N * 24) (h3 : S = (N - 70) * 38) : N = 190 :=
by
  sorry

end sweets_distribution_l94_94534


namespace find_ab_find_min_max_l94_94842

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x + 4 * Real.log x
noncomputable def f' (a b : ℝ) (x : ℝ) := 2 * a * x + b + 4 / x

theorem find_ab (a b : ℝ) :
  (∀ x, (x = 1 ∨ x = 2) → f' a b x = 0) →
  a = 1 ∧ b = -6 := by
  sorry

theorem find_min_max (a b : ℝ) :
  a = 1 ∧ b = -6 →
  (∀ x, x ∈ Set.Ioi 0 → f (1 : ℝ) (-6 : ℝ) x) →
  (∃ max min, max = f (1 : ℝ) (-6 : ℝ) 1 ∧ max = -5 ∧ min = f (1:ℝ) (-6:ℝ) 2 ∧ min = -8 + 4 * Real.log 2) := by
  sorry

end find_ab_find_min_max_l94_94842


namespace age_of_golden_retriever_l94_94135

def golden_retriever (gain_per_year current_weight : ℕ) (age : ℕ) :=
  gain_per_year * age = current_weight

theorem age_of_golden_retriever :
  golden_retriever 11 88 8 :=
by
  unfold golden_retriever
  simp
  sorry

end age_of_golden_retriever_l94_94135


namespace exists_real_ys_l94_94544

theorem exists_real_ys (x : Fin (n+1) → ℝ) 
  (h : x 0 ^ 2 ≤ Finset.univ.filter (λ i, i ≠ 0) ∑ i, (x i ^ 2)) :
  ∃ y : Fin (n+1) → ℝ, (x 0 ^ 2 + y 0 ^ 2) = Finset.univ.filter (λ i, i ≠ 0) ∑ i, ((x i) ^ 2 + (y i) ^ 2) :=
by
  sorry

end exists_real_ys_l94_94544


namespace toy_truck_distribution_l94_94661

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem toy_truck_distribution :
  ∃ (y : ℕ), (y * (factorial 9) / 10^10) = (5670 : ℕ)  :=
begin
  sorry
end

end toy_truck_distribution_l94_94661


namespace taylor_one_basket_in_three_tries_l94_94737

theorem taylor_one_basket_in_three_tries (P_no_make : ℚ) (h : P_no_make = 1/3) : 
  (∃ P_make : ℚ, P_make = 1 - P_no_make ∧ P_make * P_no_make * P_no_make * 3 = 2/9) := 
by
  sorry

end taylor_one_basket_in_three_tries_l94_94737


namespace exists_airline_route_within_same_republic_l94_94918

theorem exists_airline_route_within_same_republic
  (C : Type) [Fintype C] [DecidableEq C]
  (R : Type) [Fintype R] [DecidableEq R]
  (belongs_to : C → R)
  (airline_route : C → C → Prop)
  (country_size : Fintype.card C = 100)
  (republics_size : Fintype.card R = 3)
  (millionaire_cities : {c : C // ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x) })
  (at_least_70_millionaire_cities : ∃ n : ℕ, n ≥ 70 ∧ (∃ S : Finset {c : C // ∃ n : ℕ, n ≥ 70 ∧ ( ∃ S : Finset C, S.card = n ∧ ∀ x ∈ S, airline_route c x )}, S.card = n)):
  ∃ (c1 c2 : C), airline_route c1 c2 ∧ belongs_to c1 = belongs_to c2 := 
sorry

end exists_airline_route_within_same_republic_l94_94918


namespace general_formula_a_maximum_value_t_l94_94827

noncomputable def S (n : ℕ) : ℚ := (3 * n^2 - n) / 2

def a (n : ℕ) : ℚ := 
  if n = 0 then 0
  else (S n) - (S (n-1))

def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

def T (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, b i

theorem general_formula_a (n : ℕ) (h : n > 0) :
  a n = 3 * n - 2 := sorry

theorem maximum_value_t (t : ℚ) : 
  (∀ n, t ≤ 4 * T n) → t ≤ 1 := sorry

end general_formula_a_maximum_value_t_l94_94827


namespace polynomial_remainder_l94_94653

theorem polynomial_remainder :
  let p := λ x : ℕ, 4 * x^6 - x^5 - 8 * x^4 + 3 * x^2 + 5 * x - 15 
  in p 3 = 2079 := 
by 
  sorry

end polynomial_remainder_l94_94653


namespace find_physics_marks_l94_94037

noncomputable def marks_in_physics {marks_english marks_math marks_chemistry marks_biology marks_physics : ℝ}
  (avg_marks : ℝ) (num_subjects : ℕ) : Prop :=
  avg_marks = 68.2 ∧ num_subjects = 5 ∧
  marks_english = 70 ∧ marks_math = 63 ∧ marks_chemistry = 63 ∧ marks_biology = 65 ∧
  marks_english + marks_math + marks_chemistry + marks_biology + marks_physics = avg_marks * num_subjects

theorem find_physics_marks (avg_marks : ℝ) (num_subjects : ℕ) (marks_english marks_math marks_chemistry marks_biology marks_physics : ℝ) :
  marks_in_physics avg_marks num_subjects marks_english marks_math marks_chemistry marks_biology marks_physics →
  marks_physics = 80 :=
by
  intro h,
  -- Proof of the theorem goes here, but we skip it with sorry
  sorry

end find_physics_marks_l94_94037


namespace fraction_irreducible_l94_94553

theorem fraction_irreducible (n : ℤ) : gcd (2 * n ^ 2 + 9 * n - 17) (n + 6) = 1 := by
  sorry

end fraction_irreducible_l94_94553


namespace exists_route_same_republic_l94_94892

noncomputable def cities := (Fin 100)
def republics := (Fin 3)
def connections (c : cities) : Finset cities := sorry
def owning_republic (c : cities) : republics := sorry

theorem exists_route_same_republic (H : ∃ (S : Finset cities), S.card ≥ 70 ∧ ∀ c ∈ S, (connections c).card ≥ 70) : 
  ∃ c₁ c₂ : cities, c₁ ≠ c₂ ∧ owning_republic c₁ = owning_republic c₂ ∧ connected_route c₁ c₂ :=
by
  sorry

end exists_route_same_republic_l94_94892


namespace helen_hand_washing_time_l94_94368

theorem helen_hand_washing_time :
  (52 / 4) * 30 / 60 = 6.5 := by
  sorry

end helen_hand_washing_time_l94_94368


namespace chord_length_intercepted_by_line_l94_94043

/-- Given a circle with equation x^2 + y^2 = 1 and a line x - y = 0, 
    the length of the chord intercepted by the line on the circle is 2. -/
theorem chord_length_intercepted_by_line {x y : ℝ} : 
  (x^2 + y^2 = 1 ∧ x - y = 0) → ∃ len : ℝ, len = 2 :=
begin
  sorry
end

end chord_length_intercepted_by_line_l94_94043


namespace smallest_positive_period_function_range_l94_94118

noncomputable def f (x : ℝ) : ℝ := 2 * sin(2 * x - π / 6)

theorem smallest_positive_period (x : ℝ) : 
  ∃ T > 0, ∀ t, f (x + T) = f x ∧ T = π := 
sorry

theorem function_range : 
  set.range (λ x, f x) ∩ Icc 0 (2 * π / 3) = set.Icc (-1) 2 := 
sorry

end smallest_positive_period_function_range_l94_94118


namespace envelopes_left_l94_94705

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end envelopes_left_l94_94705


namespace sufficient_but_not_necessary_condition_for_q_p_or_q_true_p_and_q_false_l94_94817

variable {x m : ℝ}

def p : Prop := -2 ≤ x ∧ x ≤ 4
def q (m : ℝ) : Prop := (2 - m) ≤ x ∧ x ≤ (2 + m)

/- First Proof Problem -/
theorem sufficient_but_not_necessary_condition_for_q (h : p → q m) (hn : ∃ x, ¬q m) (hpsubsetq : ∀ x, p → q m) : 4 ≤ m := sorry

/- Second Proof Problem -/
theorem p_or_q_true_p_and_q_false (hm : m = 5) (hp_or_q : p ∨ q m) (hp_and_q_false : ¬(p ∧ q m)) : -3 ≤ x ∧ x < -2 ∨ 4 < x ∧ x ≤ 7 := sorry

end sufficient_but_not_necessary_condition_for_q_p_or_q_true_p_and_q_false_l94_94817


namespace polynomial_factorization_l94_94738

theorem polynomial_factorization (x : ℝ) :
  x^6 + 6*x^5 + 15*x^4 + 20*x^3 + 15*x^2 + 6*x + 1 = (x + 1)^6 :=
by {
  -- proof goes here
  sorry
}

end polynomial_factorization_l94_94738


namespace lean_solution_l94_94006

example (ozPackage1Price : ℤ) (ozPackage2Price : ℤ) (ozPackage3Price : ℤ) (ozPackage3DiscountRate : ℤ) : Prop :=
  let costOption1 := 16 * ozPackage1Price
  let costOption2 := 8 * ozPackage2Price + 2 * (4 * ozPackage3Price * ozPackage3DiscountRate / 100)
  min costOption1 costOption2 = 6

def equivalent_math_problem
    (condition1 : ∀ ozPackage1Price, ozPackage1Price = 7)
    (condition2 : ∀ ozPackage2Price, ozPackage2Price = 4)
    (condition3 : ∀ ozPackage3Price, ozPackage3Price = 2)
    (condition4 : ∀ ozPackage3DiscountRate, ozPackage3DiscountRate = 50) : Prop :=
  example 7 4 2 50 = 6

theorem lean_solution : equivalent_math_problem sorry sorry sorry sorry :=
sorry

end lean_solution_l94_94006


namespace solve_for_x_l94_94559

noncomputable def x : ℚ := 45^2 / (7 - (3 / 4))

theorem solve_for_x : x = 324 := by
  sorry

end solve_for_x_l94_94559


namespace tan_half_angle_third_quadrant_l94_94098

theorem tan_half_angle_third_quadrant (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h : Real.sin α = -24/25) :
  Real.tan (α / 2) = -4/3 := 
by 
  sorry

end tan_half_angle_third_quadrant_l94_94098


namespace surface_area_of_prism_is_72_l94_94257

-- Defining the lengths of the legs, hypotenuse and height of the prism
def leg1 : ℝ := 3
def leg2 : ℝ := 4
def hypotenuse : ℝ := 5
def height : ℝ := 5

-- Defining the function to calculate the surface area of the prism
def surface_area (a b c h : ℝ) : ℝ :=
  a * b + (a + b + c) * h

-- Prove that the surface area of the prism is 72
theorem surface_area_of_prism_is_72 : surface_area leg1 leg2 hypotenuse height = 72 :=
by
  sorry

end surface_area_of_prism_is_72_l94_94257


namespace tan_sum_pi_over_four_l94_94103

open Real

theorem tan_sum_pi_over_four (θ : ℝ) (h1 : θ ∈ Ioo (π / 2) π) (h2 : cos (θ - π / 4) = 3 / 5) :
  tan (θ + π / 4) = -3 / 4 :=
sorry

end tan_sum_pi_over_four_l94_94103


namespace polynomial_degree_l94_94728

-- Let the polynomials be defined as follows:
-- p1 = (x^5 + ax^3 + bx^2 + c)
-- p2 = (x^6 + dx^4 + e)
-- p3 = (x + f)
-- where a, b, c, d, e, f are nonzero constants.

-- We are to prove the degree of the product of these polynomials is 12.

def degree (p : Polynomial ℂ) : ℤ := p.natDegree

theorem polynomial_degree :
  ∀ (a b c d e f : ℂ), a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → e ≠ 0 → f ≠ 0 →
    degree ((X^5 + C a * X^3 + C b * X^2 + C c) * (X^6 + C d * X^4 + C e) * (X + C f)) = 12 :=
by
  intros a b c d e f ha hb hc hd he hf
  sorry

end polynomial_degree_l94_94728


namespace train_speed_kmh_l94_94002

theorem train_speed_kmh 
  (L_train : ℝ) (L_bridge : ℝ) (time : ℝ)
  (h_train : L_train = 460)
  (h_bridge : L_bridge = 140)
  (h_time : time = 48) : 
  (L_train + L_bridge) / time * 3.6 = 45 := 
by
  -- Definitions and conditions
  have h_total_dist : L_train + L_bridge = 600 := by sorry
  have h_speed_mps : (L_train + L_bridge) / time = 600 / 48 := by sorry
  have h_speed_mps_simplified : 600 / 48 = 12.5 := by sorry
  have h_speed_kmh : 12.5 * 3.6 = 45 := by sorry
  sorry

end train_speed_kmh_l94_94002


namespace tangent_identity_l94_94577

theorem tangent_identity 
  (a b c d : ℕ) 
  (h₁ : a ≥ b) 
  (h₂ : b ≥ c)
  (h₃ : c ≥ d) 
  (h₄ : 0 < a) 
  (h₅ : 0 < b) 
  (h₆ : 0 < c) 
  (h₇ : 0 < d)
  (h₈ : tan (22.5 * (pi / 180)) = (real.sqrt a) - b + (real.sqrt c) - (real.sqrt d)) : 
  a + b + c + d = 3 := 
by 
  sorry

end tangent_identity_l94_94577


namespace clock_angle_at_6_48_l94_94274

theorem clock_angle_at_6_48 :
  let minute_position := (48 / 60) * 360
  let hour_position := (6 + (48 / 60)) * 30
  abs (hour_position - minute_position) = 84 := by
  sorry

end clock_angle_at_6_48_l94_94274


namespace find_common_difference_l94_94428

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

-- Conditions
def first_term (a_n : ℕ → ℕ) := a_n 1 = 1
def common_difference (d : ℕ) := d ≠ 0
def arithmetic_def (a_n : ℕ → ℕ) (d : ℕ) := ∀ n, a_n (n+1) = a_n n + d
def geom_mean_condition (a_n : ℕ → ℕ) := a_n 2 ^ 2 = a_n 1 * a_n 4

-- Proof statement
theorem find_common_difference
  (fa : first_term a_n)
  (cd : common_difference d)
  (ad : arithmetic_def a_n d)
  (gmc : geom_mean_condition a_n) :
  d = 1 := by
  sorry

end find_common_difference_l94_94428


namespace exists_airline_route_within_same_republic_l94_94898

-- Define the concept of a country with cities, republics, and airline routes
def City : Type := ℕ
def Republic : Type := ℕ
def country : Set City := {n | n < 100}
noncomputable def cities_in_republic (R : Set City) : Prop :=
  ∃ x : City, x ∈ R

-- Conditions
def connected_by_route (c1 c2 : City) : Prop := sorry -- Placeholder for being connected by a route
def is_millionaire_city (c : City) : Prop := ∃ (routes : Set City), routes.card ≥ 70 ∧ ∀ r ∈ routes, connected_by_route c r

-- Theorem to be proved
theorem exists_airline_route_within_same_republic :
  country.card = 100 →
  ∃ republics : Set (Set City), republics.card = 3 ∧
    (∀ R ∈ republics, R.nonempty ∧ R.card ≤ 30) →
  ∃ millionaire_cities : Set City, millionaire_cities.card ≥ 70 ∧
    (∀ m ∈ millionaire_cities, is_millionaire_city m) →
  ∃ c1 c2 : City, ∃ R : Set City, R ∈ republics ∧ c1 ∈ R ∧ c2 ∈ R ∧ connected_by_route c1 c2 :=
begin
  -- Proof outline
  exact sorry
end

end exists_airline_route_within_same_republic_l94_94898


namespace find_inequality_solution_set_l94_94374

noncomputable def inequality_solution_set : Set ℝ :=
  { x | (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < (1 / 4) }

theorem find_inequality_solution_set :
  inequality_solution_set = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end find_inequality_solution_set_l94_94374


namespace minimum_f_value_l94_94540

theorem minimum_f_value (d e f : ℕ) (h1 : d < e) (h2 : e < f)
(h3 : ∃ x y : ℝ, 3*x + y = 3005 ∧ y = |x - d| + |x - e| + |x - f| ∧ ∀ x' y' : ℝ, 3*x' + y' = 3005 → y' = |x' - d| + |x' - e| + |x' - f| → x' = x) :
 f = 1504 :=
by
suffices : d = 1 ∧ e = 1503 ∧ f = 1504, from this.2.2
sorry

end minimum_f_value_l94_94540


namespace find_x_l94_94455

theorem find_x : 
  (∃ x : ℝ, (cos 100 / (1 - 4 * sin 25 * cos 25 * cos 50)) = tan x) → 
  ∃ x : ℝ, x = 95 := 
by
  assume h,
  sorry

end find_x_l94_94455


namespace polynomials_equal_if_integer_parts_equal_l94_94996

theorem polynomials_equal_if_integer_parts_equal (f g : ℚ[X])
  (hf : f.degree = 2)
  (hg : g.degree = 2)
  (h : ∀ x : ℝ, ⌊f.eval x⌋ = ⌊g.eval x⌋) : f = g :=
sorry

end polynomials_equal_if_integer_parts_equal_l94_94996


namespace value_of_f_2014_l94_94143

def f : ℕ → ℕ := sorry

theorem value_of_f_2014 : (∀ n : ℕ, f (f n) + f n = 2 * n + 3) → (f 0 = 1) → (f 2014 = 2015) := by
  intro h₁ h₀
  have h₂ := h₀
  sorry

end value_of_f_2014_l94_94143


namespace Hari_contribution_l94_94633

theorem Hari_contribution (H : ℕ) (Praveen_capital : ℕ := 3500) (months_Praveen : ℕ := 12) 
                          (months_Hari : ℕ := 7) (profit_ratio_P : ℕ := 2) (profit_ratio_H : ℕ := 3) : 
                          (Praveen_capital * months_Praveen) * profit_ratio_H = (H * months_Hari) * profit_ratio_P → 
                          H = 9000 :=
by
  sorry

end Hari_contribution_l94_94633


namespace dolls_total_l94_94342

theorem dolls_total (V S A : ℕ) 
  (hV : V = 20) 
  (hS : S = 2 * V)
  (hA : A = 2 * S) 
  : A + S + V = 140 := 
by 
  sorry

end dolls_total_l94_94342


namespace smallest_range_l94_94336

theorem smallest_range {x1 x2 x3 x4 x5 : ℝ} 
  (h1 : (x1 + x2 + x3 + x4 + x5) = 100)
  (h2 : x3 = 18)
  (h3 : 2 * x1 + 2 * x5 + 18 = 100): 
  x5 - x1 = 19 :=
by {
  sorry
}

end smallest_range_l94_94336


namespace cat_average_weight_in_pounds_l94_94005

theorem cat_average_weight_in_pounds:
  let weights := [3.5, 7.2, 4.8, 6, 5.5, 9, 4, 7.5] in
  let number_of_cats := 8 in
  let conversion_factor := 2.20462 in
  let total_weight_kg := List.sum weights in
  let average_weight_kg := total_weight_kg / number_of_cats in
  let average_weight_pounds := average_weight_kg * conversion_factor in
  average_weight_pounds = 13.0925 :=
by
  sorry

end cat_average_weight_in_pounds_l94_94005


namespace problem1_l94_94650

theorem problem1
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l94_94650


namespace gcf_7fact_8fact_l94_94754

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end gcf_7fact_8fact_l94_94754
