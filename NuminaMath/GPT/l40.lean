import Mathlib

namespace sum_faces_edges_vertices_eq_26_l40_40667

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l40_40667


namespace decreasing_range_of_a_l40_40574

noncomputable def f (a x : ℝ) : ℝ := (Real.sqrt (2 - a * x)) / (a - 1)

theorem decreasing_range_of_a (a : ℝ) :
    (∀ x y : ℝ, 0 ≤ x → x ≤ 1/2 → 0 ≤ y → y ≤ 1/2 → x < y → f a y < f a x) ↔ (a < 0 ∨ (1 < a ∧ a ≤ 4)) :=
by
  sorry

end decreasing_range_of_a_l40_40574


namespace side_length_square_correct_l40_40712

noncomputable def side_length_square (time_seconds : ℕ) (speed_kmph : ℕ) : ℕ := sorry

theorem side_length_square_correct (time_seconds : ℕ) (speed_kmph : ℕ) (h_time : time_seconds = 24) 
  (h_speed : speed_kmph = 12) : side_length_square time_seconds speed_kmph = 20 :=
sorry

end side_length_square_correct_l40_40712


namespace total_amount_withdrawn_l40_40370

def principal : ℤ := 20000
def interest_rate : ℚ := 3.33 / 100
def term : ℤ := 3

theorem total_amount_withdrawn :
  principal + (principal * interest_rate * term) = 21998 := by
  sorry

end total_amount_withdrawn_l40_40370


namespace brand_tangyuan_purchase_l40_40630

theorem brand_tangyuan_purchase (x y : ℕ) 
  (h1 : x + y = 1000) 
  (h2 : x = 2 * y + 20) : 
  x = 670 ∧ y = 330 := 
sorry

end brand_tangyuan_purchase_l40_40630


namespace megatek_manufacturing_percentage_l40_40352

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ) 
    (h1 : total_degrees = 360) 
    (h2 : manufacturing_degrees = 126) : 
    (manufacturing_degrees / total_degrees) * 100 = 35 := by
  sorry

end megatek_manufacturing_percentage_l40_40352


namespace exists_An_Bn_l40_40381

theorem exists_An_Bn (n : ℕ) : ∃ (A_n B_n : ℕ), (3 - Real.sqrt 7) ^ n = A_n - B_n * Real.sqrt 7 := by
  sorry

end exists_An_Bn_l40_40381


namespace prob_prime_sum_l40_40259

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := nat.prime n

def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (List.product l l).filter (λ p, p.1 < p.2)

def prime_sum_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  valid_pairs l |>.filter (λ p, is_prime (p.1 + p.2))

theorem prob_prime_sum :
  prime_sum_pairs primes.length / (valid_pairs primes).length = 1 / 9 :=
by
  sorry -- Proof is not required

end prob_prime_sum_l40_40259


namespace mobile_purchase_price_l40_40446

theorem mobile_purchase_price (M : ℝ) 
  (P_grinder : ℝ := 15000)
  (L_grinder : ℝ := 0.05 * P_grinder)
  (SP_grinder : ℝ := P_grinder - L_grinder)
  (SP_mobile : ℝ := 1.1 * M)
  (P_overall : ℝ := P_grinder + M)
  (SP_overall : ℝ := SP_grinder + SP_mobile)
  (profit : ℝ := 50)
  (h : SP_overall = P_overall + profit) :
  M = 8000 :=
by 
  sorry

end mobile_purchase_price_l40_40446


namespace length_of_one_string_l40_40695

theorem length_of_one_string (total_length : ℕ) (num_strings : ℕ) (h_total_length : total_length = 98) (h_num_strings : num_strings = 7) : total_length / num_strings = 14 := by
  sorry

end length_of_one_string_l40_40695


namespace min_a_plus_b_l40_40610

theorem min_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -145 := sorry

end min_a_plus_b_l40_40610


namespace sum_of_faces_edges_vertices_of_rect_prism_l40_40689

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l40_40689


namespace compute_xy_l40_40188

theorem compute_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end compute_xy_l40_40188


namespace remainder_of_modified_expression_l40_40580

theorem remainder_of_modified_expression (x y u v : ℕ) (h : x = u * y + v) (hy_pos : y > 0) (hv_bound : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y + 4) % y = v + 4 :=
by sorry

end remainder_of_modified_expression_l40_40580


namespace rectangular_prism_faces_edges_vertices_sum_l40_40678

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l40_40678


namespace initial_players_round_robin_l40_40904

-- Definitions of conditions
def num_matches_round_robin (x : ℕ) : ℕ := x * (x - 1) / 2
def num_matches_after_drop_out (x : ℕ) : ℕ := num_matches_round_robin x - 2 * (x - 4) + 1

-- The theorem statement
theorem initial_players_round_robin (x : ℕ) 
  (two_players_dropped : num_matches_after_drop_out x = 84) 
  (round_robin_condition : num_matches_round_robin x - 2 * (x - 4) + 1 = 84 ∨ num_matches_round_robin x - 2 * (x - 4) = 84) :
  x = 15 :=
sorry

end initial_players_round_robin_l40_40904


namespace triathlete_average_speed_is_approx_3_5_l40_40730

noncomputable def triathlete_average_speed : ℝ :=
  let x : ℝ := 1; -- This represents the distance of biking/running segment
  let swimming_speed := 2; -- km/h
  let biking_speed := 25; -- km/h
  let running_speed := 12; -- km/h
  let swimming_distance := 2 * x; -- 2x km
  let biking_distance := x; -- x km
  let running_distance := x; -- x km
  let total_distance := swimming_distance + biking_distance + running_distance; -- 4x km
  let swimming_time := swimming_distance / swimming_speed; -- x hours
  let biking_time := biking_distance / biking_speed; -- x/25 hours
  let running_time := running_distance / running_speed; -- x/12 hours
  let total_time := swimming_time + biking_time + running_time; -- 1.12333x hours
  total_distance / total_time -- This should be the average speed

theorem triathlete_average_speed_is_approx_3_5 :
  abs (triathlete_average_speed - 3.5) < 0.1 := 
by
  sorry

end triathlete_average_speed_is_approx_3_5_l40_40730


namespace inverse_proportionality_l40_40843

-- Define the functions as assumptions or constants
def A (x : ℝ) := 2 * x
def B (x : ℝ) := x / 2
def C (x : ℝ) := 2 / x
def D (x : ℝ) := 2 / (x - 1)

-- State that C is the one which represents inverse proportionality
theorem inverse_proportionality (x : ℝ) :
  (∃ y, y = C x ∧ ∀ (u v : ℝ), u * v = 2) →
  (∃ y, y = A x ∧ ∀ (u v : ℝ), u * v ≠ 2) ∧
  (∃ y, y = B x ∧ ∀ (u v : ℝ), u * v ≠ 2) ∧
  (∃ y, y = D x ∧ ∀ (u v : ℝ), u * v ≠ 2):=
sorry

end inverse_proportionality_l40_40843


namespace probability_prime_sum_of_two_draws_l40_40265

open Finset

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrime (n : ℕ) : Prop := Nat.Prime n

def validPairs (primes : List ℕ) : List (ℕ × ℕ) :=
  primes.bind (λ p1 => primes.filter (λ p2 => p1 ≠ p2 ∧ isPrime (p1 + p2)).map (λ p2 => (p1, p2)))

theorem probability_prime_sum_of_two_draws :
  (firstTenPrimes.length.choose 2 : ℚ) > 0 →
  let validPairs := validPairs firstTenPrimes
  let numValidPairs := validPairs.toFinset.card
  let totalPairs := 45
  (∀ (x ∈ validPairs), ∃ (a b : ℕ), a ∈ firstTenPrimes ∧ b ∈ firstTenPrimes ∧ a ≠ b ∧ isPrime (a + b) ∧ (a, b) = x) →
  (∀ p, p ∈ firstTenPrimes → isPrime p) →
  numValidPairs / totalPairs = (1 : ℚ) / 9 := by
  sorry

end probability_prime_sum_of_two_draws_l40_40265


namespace intersection_eq_l40_40092

-- Define the sets M and N using the given conditions
def M : Set ℝ := { x | x < 1 / 2 }
def N : Set ℝ := { x | x ≥ -4 }

-- The goal is to prove that the intersection of M and N is { x | -4 ≤ x < 1 / 2 }
theorem intersection_eq : M ∩ N = { x | -4 ≤ x ∧ x < (1 / 2) } :=
by
  sorry

end intersection_eq_l40_40092


namespace sum_of_legs_of_right_triangle_l40_40355

theorem sum_of_legs_of_right_triangle (y : ℤ) (hyodd : y % 2 = 1) (hyp : y ^ 2 + (y + 2) ^ 2 = 17 ^ 2) :
  y + (y + 2) = 24 :=
sorry

end sum_of_legs_of_right_triangle_l40_40355


namespace plane_equation_l40_40053

-- Define the point and the normal vector
def point : ℝ × ℝ × ℝ := (8, -2, 2)
def normal_vector : ℝ × ℝ × ℝ := (8, -2, 2)

-- Define integers A, B, C, D such that the plane equation satisfies the conditions
def A : ℤ := 4
def B : ℤ := -1
def C : ℤ := 1
def D : ℤ := -18

-- Prove the equation of the plane
theorem plane_equation (x y z : ℝ) :
  A * x + B * y + C * z + D = 0 ↔ 4 * x - y + z - 18 = 0 :=
by
  sorry

end plane_equation_l40_40053


namespace probability_sum_two_primes_is_prime_l40_40260

open Finset

-- First 10 prime numbers
def firstTenPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Function to sum of two primes
def sum_of_two_primes_is_prime (a b : ℕ) : Prop := is_prime (a + b)

-- Function to compute the probability
def probability_two_prime_sum_is_prime : ℚ :=
  let pairs := firstTenPrimes.pairs
  let prime_pairs := pairs.filter (λ pair, sum_of_two_primes_is_prime pair.1 pair.2)
  (prime_pairs.card : ℚ) / pairs.card

theorem probability_sum_two_primes_is_prime :
  probability_two_prime_sum_is_prime = 1 / 9 :=
by sorry

end probability_sum_two_primes_is_prime_l40_40260


namespace completing_square_l40_40810

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l40_40810


namespace min_cost_example_l40_40395

-- Define the numbers given in the problem
def num_students : Nat := 25
def num_vampire : Nat := 11
def num_pumpkin : Nat := 14
def pack_cost : Nat := 3
def individual_cost : Nat := 1
def pack_size : Nat := 5

-- Define the cost calculation function
def min_cost (num_v: Nat) (num_p: Nat) : Nat :=
  let num_v_packs := num_v / pack_size  -- number of packs needed for vampire bags
  let num_v_individual := num_v % pack_size  -- remaining vampire bags needed
  let num_v_cost := (num_v_packs * pack_cost) + (num_v_individual * individual_cost)
  let num_p_packs := num_p / pack_size  -- number of packs needed for pumpkin bags
  let num_p_individual := num_p % pack_size  -- remaining pumpkin bags needed
  let num_p_cost := (num_p_packs * pack_cost) + (num_p_individual * individual_cost)
  num_v_cost + num_p_cost

-- The statement to prove
theorem min_cost_example : min_cost num_vampire num_pumpkin = 17 :=
  by
  sorry

end min_cost_example_l40_40395


namespace find_height_of_box_l40_40388

-- Definitions for the problem conditions
def numCubes : ℕ := 24
def volumeCube : ℕ := 27
def lengthBox : ℕ := 8
def widthBox : ℕ := 9
def totalVolumeBox : ℕ := numCubes * volumeCube

-- Problem statement in Lean 4
theorem find_height_of_box : totalVolumeBox = lengthBox * widthBox * 9 :=
by sorry

end find_height_of_box_l40_40388


namespace sum_of_faces_edges_vertices_l40_40661

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l40_40661


namespace Oshea_needs_30_small_planters_l40_40168

theorem Oshea_needs_30_small_planters 
  (total_seeds : ℕ) 
  (large_planters : ℕ) 
  (capacity_large : ℕ) 
  (capacity_small : ℕ)
  (h1: total_seeds = 200) 
  (h2: large_planters = 4) 
  (h3: capacity_large = 20) 
  (h4: capacity_small = 4) : 
  (total_seeds - large_planters * capacity_large) / capacity_small = 30 :=
by 
  sorry

end Oshea_needs_30_small_planters_l40_40168


namespace z_max_plus_z_min_l40_40896

theorem z_max_plus_z_min {x y z : ℝ} 
  (h1 : x^2 + y^2 + z^2 = 3) 
  (h2 : x + 2 * y - 2 * z = 4) : 
  z + z = -4 :=
by 
  sorry

end z_max_plus_z_min_l40_40896


namespace pie_eating_contest_l40_40503

def a : ℚ := 7 / 8
def b : ℚ := 5 / 6
def difference : ℚ := 1 / 24

theorem pie_eating_contest : a - b = difference := 
sorry

end pie_eating_contest_l40_40503


namespace calculate_square_difference_l40_40094

theorem calculate_square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 :=
by
  sorry

end calculate_square_difference_l40_40094


namespace inverse_proportion_l40_40844

theorem inverse_proportion (x : ℝ) (y : ℝ) (f₁ f₂ f₃ f₄ : ℝ → ℝ) (h₁ : f₁ x = 2 * x) (h₂ : f₂ x = x / 2) (h₃ : f₃ x = 2 / x) (h₄ : f₄ x = 2 / (x - 1)) :
  f₃ x * x = 2 := sorry

end inverse_proportion_l40_40844


namespace circumscribed_sphere_surface_area_l40_40081

theorem circumscribed_sphere_surface_area
  (x y z : ℝ)
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  let l := Real.sqrt (x^2 + y^2 + z^2)
  let R := l / 2
  4 * Real.pi * R^2 = 6 * Real.pi :=
by sorry

end circumscribed_sphere_surface_area_l40_40081


namespace least_positive_integer_with_12_factors_is_72_l40_40006

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l40_40006


namespace value_range_of_f_l40_40646

noncomputable def f (x : ℝ) : ℝ := 4 / (x - 2)

theorem value_range_of_f : Set.range (fun x => f x) ∩ Set.Icc 3 6 = Set.Icc 1 4 :=
by
  sorry

end value_range_of_f_l40_40646


namespace time_to_cover_length_l40_40846

def speed_escalator : ℝ := 10
def speed_person : ℝ := 4
def length_escalator : ℝ := 112

theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person) = 8) :=
by
  sorry

end time_to_cover_length_l40_40846


namespace arithmetic_mean_of_reciprocals_of_first_five_primes_l40_40285

theorem arithmetic_mean_of_reciprocals_of_first_five_primes :
  let primes := [2, 3, 5, 7, 11]
  let reciprocals := primes.map (λ x, 1 / (x : ℚ))
  let mean := (reciprocals.sum / 5) 
  in mean = (2927 / 11550 : ℚ) :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_five_primes_l40_40285


namespace group_size_l40_40384

theorem group_size (boxes_per_man total_boxes : ℕ) (h1 : boxes_per_man = 2) (h2 : total_boxes = 14) :
  total_boxes / boxes_per_man = 7 := by
  -- Definitions and conditions from the problem
  have man_can_carry_2_boxes : boxes_per_man = 2 := h1
  have group_can_hold_14_boxes : total_boxes = 14 := h2
  -- Proof follows from these conditions
  sorry

end group_size_l40_40384


namespace taxi_fare_proof_l40_40949

/-- Given equations representing the taxi fare conditions:
1. x + 7y = 16.5 (Person A's fare)
2. x + 11y = 22.5 (Person B's fare)

And using the value of the initial fare and additional charge per kilometer conditions,
prove the initial fare and additional charge and calculate the fare for a 7-kilometer ride. -/
theorem taxi_fare_proof (x y : ℝ) 
  (h1 : x + 7 * y = 16.5)
  (h2 : x + 11 * y = 22.5)
  (h3 : x = 6)
  (h4 : y = 1.5) :
  x = 6 ∧ y = 1.5 ∧ (x + y * (7 - 3)) = 12 :=
by
  sorry

end taxi_fare_proof_l40_40949


namespace johns_previous_earnings_l40_40325

theorem johns_previous_earnings (new_earnings raise_percentage old_earnings : ℝ) 
  (h1 : new_earnings = 68) (h2 : raise_percentage = 0.1333333333333334)
  (h3 : new_earnings = old_earnings * (1 + raise_percentage)) : old_earnings = 60 :=
sorry

end johns_previous_earnings_l40_40325


namespace triangle_third_side_l40_40356

theorem triangle_third_side (a b c : ℝ) (h1 : a = 5) (h2 : b = 7) (h3 : 2 < c ∧ c < 12) : c = 6 :=
sorry

end triangle_third_side_l40_40356


namespace sam_pam_ratio_is_2_l40_40644

-- Definition of given conditions
def min_assigned_pages : ℕ := 25
def harrison_extra_read : ℕ := 10
def pam_extra_read : ℕ := 15
def sam_read : ℕ := 100

-- Calculations based on the given conditions
def harrison_read : ℕ := min_assigned_pages + harrison_extra_read
def pam_read : ℕ := harrison_read + pam_extra_read

-- Prove the ratio of the number of pages Sam read to the number of pages Pam read is 2
theorem sam_pam_ratio_is_2 : sam_read / pam_read = 2 := 
by
  sorry

end sam_pam_ratio_is_2_l40_40644


namespace tan_alpha_fraction_l40_40100

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l40_40100


namespace part1_part2_l40_40871

theorem part1 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a * b > 0) : a + b = 8 ∨ a + b = -8 :=
sorry

theorem part2 (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h4 : |a + b| = a + b) : a - b = 4 ∨ a - b = 8 :=
sorry

end part1_part2_l40_40871


namespace calculate_product_l40_40996

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l40_40996


namespace area_of_OBEC_is_25_l40_40978

noncomputable def area_OBEC : ℝ :=
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let E := (5, 5)
  let O := (0, 0)
  let area_triangle (P Q R : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))
  area_triangle O B E - area_triangle O E C

theorem area_of_OBEC_is_25 :
  area_OBEC = 25 := 
by
  sorry

end area_of_OBEC_is_25_l40_40978


namespace factorize_cubic_l40_40279

theorem factorize_cubic (x : ℝ) : 
  (4 * x^3 - 4 * x^2 + x) = x * (2 * x - 1)^2 := 
begin
  sorry
end

end factorize_cubic_l40_40279


namespace bus_ticket_probability_l40_40832

theorem bus_ticket_probability :
  let total_tickets := 10 ^ 6
  let choices := Nat.choose 10 6 * 2
  (choices : ℝ) / total_tickets = 0.00042 :=
by
  sorry

end bus_ticket_probability_l40_40832


namespace correct_cases_needed_l40_40343

noncomputable def cases_needed (boxes_sold : ℕ) (boxes_per_case : ℕ) : ℕ :=
  (boxes_sold + boxes_per_case - 1) / boxes_per_case

theorem correct_cases_needed :
  cases_needed 10 6 = 2 ∧ -- For trefoils
  cases_needed 15 5 = 3 ∧ -- For samoas
  cases_needed 20 10 = 2  -- For thin mints
:= by
  sorry

end correct_cases_needed_l40_40343


namespace find_remainder_l40_40192

-- Definitions based on given conditions
def dividend := 167
def divisor := 18
def quotient := 9

-- Statement to prove
theorem find_remainder : dividend = (divisor * quotient) + 5 :=
by
  -- Definitions used in the problem
  unfold dividend divisor quotient
  sorry

end find_remainder_l40_40192


namespace triangle_no_solution_l40_40424

def angleSumOfTriangle : ℝ := 180

def hasNoSolution (a b A : ℝ) : Prop :=
  A >= angleSumOfTriangle

theorem triangle_no_solution {a b A : ℝ} (ha : a = 181) (hb : b = 209) (hA : A = 121) :
  hasNoSolution a b A := sorry

end triangle_no_solution_l40_40424


namespace number_of_fiction_books_l40_40456

theorem number_of_fiction_books (F NF : ℕ) (h1 : F + NF = 52) (h2 : NF = 7 * F / 6) : F = 24 := 
by
  sorry

end number_of_fiction_books_l40_40456


namespace relationship_between_a_and_b_l40_40571

-- Define the objects and their relationships
noncomputable def α_parallel_β : Prop := sorry
noncomputable def a_parallel_α : Prop := sorry
noncomputable def b_perpendicular_β : Prop := sorry

-- Define the relationship we want to prove
noncomputable def a_perpendicular_b : Prop := sorry

-- The statement we want to prove
theorem relationship_between_a_and_b (h1 : α_parallel_β) (h2 : a_parallel_α) (h3 : b_perpendicular_β) : a_perpendicular_b :=
sorry

end relationship_between_a_and_b_l40_40571


namespace present_population_l40_40359

variable (P : ℝ)
variable (H1 : P * 1.20 = 2400)

theorem present_population (H1 : P * 1.20 = 2400) : P = 2000 :=
by {
  sorry
}

end present_population_l40_40359


namespace solve_for_k_l40_40799

theorem solve_for_k (k x : ℝ) (h₁ : 4 * k - 3 * x = 2) (h₂ : x = -1) : 
  k = -1 / 4 := 
by sorry

end solve_for_k_l40_40799


namespace seq_2011_l40_40873

-- Definition of the sequence
def seq (a : ℕ → ℤ) := (a 1 = a 201) ∧ a 201 = 2 ∧ ∀ n : ℕ, a n + a (n + 1) = 0

-- The main theorem to prove that a_2011 = 2
theorem seq_2011 : ∀ a : ℕ → ℤ, seq a → a 2011 = 2 :=
by
  intros a h
  let seq := h
  sorry

end seq_2011_l40_40873


namespace least_positive_integer_with_12_factors_l40_40026

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l40_40026


namespace tan_alpha_value_l40_40135

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l40_40135


namespace sum_of_squares_mod_13_l40_40508

theorem sum_of_squares_mod_13 :
  (∑ i in Finset.range 16, i^2) % 13 = 1 := sorry

end sum_of_squares_mod_13_l40_40508


namespace speed_faster_train_correct_l40_40806

noncomputable def speed_faster_train_proof
  (time_seconds : ℝ) 
  (speed_slower_train : ℝ)
  (train_length_meters : ℝ) :
  Prop :=
  let time_hours := time_seconds / 3600
  let train_length_km := train_length_meters / 1000
  let total_distance_km := train_length_km + train_length_km
  let relative_speed_km_hr := total_distance_km / time_hours
  let speed_faster_train := relative_speed_km_hr + speed_slower_train
  speed_faster_train = 46

theorem speed_faster_train_correct :
  speed_faster_train_proof 36.00001 36 50.000013888888894 :=
by 
  -- proof steps would go here
  sorry

end speed_faster_train_correct_l40_40806


namespace value_of_expression_l40_40982

def g (x : ℝ) (p q r s t : ℝ) : ℝ :=
  p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_of_expression (p q r s t : ℝ) (h : g (-1) p q r s t = 4) :
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 :=
sorry

end value_of_expression_l40_40982


namespace frustum_slant_height_l40_40981

theorem frustum_slant_height
  (ratio_area : ℝ)
  (slant_height_removed : ℝ)
  (sf_ratio : ratio_area = 1/16)
  (shr : slant_height_removed = 3) :
  ∃ (slant_height_frustum : ℝ), slant_height_frustum = 9 :=
by
  sorry

end frustum_slant_height_l40_40981


namespace velocity_zero_times_l40_40524

noncomputable def s (t : ℝ) : ℝ := (1 / 4) * t^4 - (5 / 3) * t^3 + 2 * t^2

theorem velocity_zero_times :
  {t : ℝ | deriv s t = 0} = {0, 1, 4} :=
by 
  sorry

end velocity_zero_times_l40_40524


namespace completing_square_l40_40814

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l40_40814


namespace solve_for_y_l40_40469

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l40_40469


namespace minimize_a2_b2_l40_40332

theorem minimize_a2_b2 (a b t : ℝ) (h : 2 * a + b = 2 * t) : ∃ a b, (2 * a + b = 2 * t) ∧ (a^2 + b^2 = 4 * t^2 / 5) :=
by
  sorry

end minimize_a2_b2_l40_40332


namespace smallest_possible_value_of_sum_l40_40703

theorem smallest_possible_value_of_sum (a b : ℤ) (h1 : a > 6) (h2 : ∃ a' b', a' - b' = 4) : a + b < 11 := 
sorry

end smallest_possible_value_of_sum_l40_40703


namespace point_coordinates_l40_40440

def point_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0 

theorem point_coordinates (m : ℝ) 
  (h1 : point_in_second_quadrant (-m-1) (2*m+1))
  (h2 : |2*m + 1| = 5) : (-m-1, 2*m+1) = (-3, 5) :=
sorry

end point_coordinates_l40_40440


namespace general_term_formula_sum_of_geometric_sequence_l40_40302

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 3

def conditions_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 4 = 14

-- Definitions for the geometric sequence
def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q, ∀ n, b (n + 1) = b n * q

def conditions_2 (a b : ℕ → ℤ) : Prop := 
  b 2 = a 2 ∧ 
  b 4 = a 6

-- The main theorem statements for part (I) and part (II)
theorem general_term_formula (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : conditions_1 a) : 
  ∀ n, a n = 3 * n - 2 := 
sorry

theorem sum_of_geometric_sequence (a b : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = 3)
  (h2 : a 2 + a 4 = 14)
  (h3 : b 2 = a 2)
  (h4 : b 4 = a 6)
  (h5 : geometric_sequence b) :
  ∃ (S7 : ℤ), S7 = 254 ∨ S7 = -86 :=
sorry

end general_term_formula_sum_of_geometric_sequence_l40_40302


namespace probability_sum_is_prime_l40_40244

-- Define the set of the first ten primes
def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define a function that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  ∃ p, p > 1 ∧ p ≤ n ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Define a function that checks if the sum of two primes is a prime
def is_sum_prime (a b : ℕ) : Prop :=
  is_prime (a + b)

-- Define the problem of calculating the required probability
theorem probability_sum_is_prime :
  let totalPairs := (firstTenPrimes.to_finset.powerset.card) / 2
  let validPairs := (({(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)}).card)
  validPairs / totalPairs = (1 : ℕ) / (9 : ℕ) := by
  sorry

end probability_sum_is_prime_l40_40244


namespace percentage_error_in_area_l40_40038

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := 1.02 * s
  let A := s ^ 2
  let A' := s' ^ 2
  let error := A' - A
  let percent_error := (error / A) * 100
  percent_error = 4.04 := by
  sorry

end percentage_error_in_area_l40_40038


namespace decimal_palindrome_multiple_l40_40386

def is_decimal_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem decimal_palindrome_multiple (n : ℕ) (h : ¬ (10 ∣ n)) : 
  ∃ m : ℕ, is_decimal_palindrome m ∧ m % n = 0 :=
by sorry

end decimal_palindrome_multiple_l40_40386


namespace arctan_tan_equiv_l40_40397

theorem arctan_tan_equiv (h1 : Real.tan (Real.pi / 4 + Real.pi / 12) = 1 / Real.tan (Real.pi / 4 - Real.pi / 3))
  (h2 : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3):
  Real.arctan (Real.tan (5 * Real.pi / 12) - 2 * Real.tan (Real.pi / 6)) = 5 * Real.pi / 12 := 
sorry

end arctan_tan_equiv_l40_40397


namespace winning_percentage_is_65_l40_40725

theorem winning_percentage_is_65 
  (total_games won_games : ℕ) 
  (h1 : total_games = 280) 
  (h2 : won_games = 182) :
  ((won_games : ℚ) / (total_games : ℚ)) * 100 = 65 :=
by
  sorry

end winning_percentage_is_65_l40_40725


namespace smaller_angle_at_3_15_l40_40957

-- Definitions from the conditions
def degree_per_hour := 30
def degree_per_minute := 6
def minute_hand_position (minutes: Int) := minutes * degree_per_minute
def hour_hand_position (hour: Int) (minutes: Int) := hour * degree_per_hour + (minutes * degree_per_hour) / 60

-- Conditions at 3:15
def minute_hand_3_15 := minute_hand_position 15
def hour_hand_3_15 := hour_hand_position 3 15

-- The proof goal: smaller angle at 3:15 is 7.5 degrees
theorem smaller_angle_at_3_15 : 
  abs (hour_hand_3_15 - minute_hand_3_15) = 7.5 := 
by
  sorry

end smaller_angle_at_3_15_l40_40957


namespace gcd_exponentiation_gcd_fermat_numbers_l40_40200

-- Part (a)
theorem gcd_exponentiation (m n : ℕ) (a : ℕ) (h1 : m ≠ n) (h2 : a > 1) : 
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 :=
by
sorry

-- Part (b)
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_fermat_numbers (m n : ℕ) (h1 : m ≠ n) : 
  Nat.gcd (fermat_number m) (fermat_number n) = 1 :=
by
sorry

end gcd_exponentiation_gcd_fermat_numbers_l40_40200


namespace quadratic_increasing_implies_m_gt_1_l40_40091

theorem quadratic_increasing_implies_m_gt_1 (m : ℝ) (x : ℝ) 
(h1 : x > 1) 
(h2 : ∀ x, (y = x^2 + (m-3) * x + m + 1) → (∀ z > x, y < z^2 + (m-3) * z + m + 1)) 
: m > 1 := 
sorry

end quadratic_increasing_implies_m_gt_1_l40_40091


namespace equilateral_triangle_in_ellipse_l40_40533

theorem equilateral_triangle_in_ellipse :
  (∃ (a b : ℝ) (AC F1F2 s : ℝ),
       (a ≠ 0 ∧ b ≠ 0) ∧
       (∀ x y : ℝ, (x/a)^2 + (y/b)^2 = 1) ∧
       (s > 0) ∧ 
       (F1F2 = 2) ∧
       (b = √3) ∧
       (vertex_B = (0, √3)) ∧ 
       (vertex_C = (s/2, y_M)) ∧
       (vertex_A = (-s/2, y_M)) ∧
       (2 * c = F1F2) ∧ 
       (c = 1) ∧
       let mid_AC = (0, y_M) in y_M = -√3/2 * (s-2)
       → 
       AC / F1F2 = 8/5) :=
begin
  sorry
end

end equilateral_triangle_in_ellipse_l40_40533


namespace calculate_product_l40_40995

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l40_40995


namespace number_of_integer_solutions_is_zero_l40_40071

-- Define the problem conditions
def eq1 (x y z : ℤ) : Prop := x^2 - 3 * x * y + 2 * y^2 - z^2 = 27
def eq2 (x y z : ℤ) : Prop := -x^2 + 6 * y * z + 2 * z^2 = 52
def eq3 (x y z : ℤ) : Prop := x^2 + x * y + 8 * z^2 = 110

-- State the theorem to be proved
theorem number_of_integer_solutions_is_zero :
  ∀ (x y z : ℤ), eq1 x y z → eq2 x y z → eq3 x y z → false :=
by
  sorry

end number_of_integer_solutions_is_zero_l40_40071


namespace negation_of_exists_leq_zero_l40_40358

theorem negation_of_exists_leq_zero (x : ℝ) : (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) :=
by
  sorry

end negation_of_exists_leq_zero_l40_40358


namespace lcm_48_180_value_l40_40552

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l40_40552


namespace rhombus_locus_l40_40082

-- Define the coordinates of the vertices of the rhombus
structure Point :=
(x : ℝ)
(y : ℝ)

def A (e : ℝ) : Point := ⟨e, 0⟩
def B (f : ℝ) : Point := ⟨0, f⟩
def C (e : ℝ) : Point := ⟨-e, 0⟩
def D (f : ℝ) : Point := ⟨0, -f⟩

-- Define the distance squared from a point P to a point Q
def dist_sq (P Q : Point) : ℝ := (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the geometric locus problem
theorem rhombus_locus (P : Point) (e f : ℝ) :
  dist_sq P (A e) = dist_sq P (B f) + dist_sq P (C e) + dist_sq P (D f) ↔
  (if e > f then
    (dist_sq P (A e) = (e^2 - f^2) ∨ dist_sq P (C e) = (e^2 - f^2))
   else if e = f then
    (P = A e ∨ P = B f ∨ P = C e ∨ P = D f)
   else
    false) :=
sorry

end rhombus_locus_l40_40082


namespace tan_alpha_value_l40_40130

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l40_40130


namespace harmonic_point_P_3_m_harmonic_point_hyperbola_l40_40402

-- Part (1)
theorem harmonic_point_P_3_m (t : ℝ) (m : ℝ) (P : ℝ × ℝ → Prop)
  (h₁ : P ⟨ 3, m ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  m = -7 :=
by sorry

-- Part (2)
theorem harmonic_point_hyperbola (k : ℝ) (P : ℝ × ℝ → Prop)
  (h_hb : ∀ x, -3 < x ∧ x < -1 → P ⟨ x, k / x ⟩)
  (h₂ : ∀ x y, P ⟨ x, y ⟩ ↔ (x^2 = 4*y + t ∧ y^2 = 4*x + t ∧ x ≠ y)) :
  3 < k ∧ k < 4 :=
by sorry

end harmonic_point_P_3_m_harmonic_point_hyperbola_l40_40402


namespace solve_for_y_l40_40468

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l40_40468


namespace sum_faces_edges_vertices_l40_40674

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l40_40674


namespace part_one_solution_set_part_two_range_of_m_l40_40304

noncomputable def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

/- Part I -/
theorem part_one_solution_set (x : ℝ) : 
  (f x (-1) <= 2) ↔ (0 <= x ∧ x <= 4 / 3) := 
sorry

/- Part II -/
theorem part_two_range_of_m (m : ℝ) : 
  (∀ x ∈ (Set.Icc 1 2), f x m <= |2 * x + 1|) ↔ (-3 <= m ∧ m <= 0) := 
sorry

end part_one_solution_set_part_two_range_of_m_l40_40304


namespace max_ab_real_positive_l40_40410

theorem max_ab_real_positive (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 2) : 
  ab ≤ 1 :=
sorry

end max_ab_real_positive_l40_40410


namespace sum_faces_edges_vertices_l40_40669

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l40_40669


namespace rectangle_diagonals_equal_rhombus_not_l40_40639

/-- Define the properties for a rectangle -/
structure Rectangle :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- Define the properties for a rhombus -/
structure Rhombus :=
  (sides_parallel : Prop)
  (diagonals_equal : Prop)
  (diagonals_bisect : Prop)
  (angles_equal : Prop)

/-- The property that distinguishes a rectangle from a rhombus is that the diagonals are equal. -/
theorem rectangle_diagonals_equal_rhombus_not
  (R : Rectangle)
  (H : Rhombus)
  (hR1 : R.sides_parallel)
  (hR2 : R.diagonals_equal)
  (hR3 : R.diagonals_bisect)
  (hR4 : R.angles_equal)
  (hH1 : H.sides_parallel)
  (hH2 : ¬H.diagonals_equal)
  (hH3 : H.diagonals_bisect)
  (hH4 : H.angles_equal) :
  (R.diagonals_equal) := by
  sorry

end rectangle_diagonals_equal_rhombus_not_l40_40639


namespace inequality_equality_condition_l40_40335

theorem inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end inequality_equality_condition_l40_40335


namespace probability_prime_sum_of_two_draws_l40_40267

open Finset

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrime (n : ℕ) : Prop := Nat.Prime n

def validPairs (primes : List ℕ) : List (ℕ × ℕ) :=
  primes.bind (λ p1 => primes.filter (λ p2 => p1 ≠ p2 ∧ isPrime (p1 + p2)).map (λ p2 => (p1, p2)))

theorem probability_prime_sum_of_two_draws :
  (firstTenPrimes.length.choose 2 : ℚ) > 0 →
  let validPairs := validPairs firstTenPrimes
  let numValidPairs := validPairs.toFinset.card
  let totalPairs := 45
  (∀ (x ∈ validPairs), ∃ (a b : ℕ), a ∈ firstTenPrimes ∧ b ∈ firstTenPrimes ∧ a ≠ b ∧ isPrime (a + b) ∧ (a, b) = x) →
  (∀ p, p ∈ firstTenPrimes → isPrime p) →
  numValidPairs / totalPairs = (1 : ℚ) / 9 := by
  sorry

end probability_prime_sum_of_two_draws_l40_40267


namespace arithmetic_mean_of_reciprocals_of_first_five_primes_l40_40284

theorem arithmetic_mean_of_reciprocals_of_first_five_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7 + 1 / 11) / 5 = 2927 / 11550 := 
sorry

end arithmetic_mean_of_reciprocals_of_first_five_primes_l40_40284


namespace probability_green_ball_l40_40399

noncomputable def P_g : ℚ := 7 / 15

def balls_in_X : ℕ × ℕ := (3, 7)  -- (red balls, green balls)
def balls_in_Y : ℕ × ℕ := (8, 2)  -- (red balls, green balls)
def balls_in_Z : ℕ × ℕ := (5, 5)  -- (red balls, green balls)

def containers : List (ℕ × ℕ) := [balls_in_X, balls_in_Y, balls_in_Z]

theorem probability_green_ball : 
  (1/3) * (7/10) + (1/3) * (1/5) + (1/3) * (1/2) = 7 / 15 := 
by sorry

end probability_green_ball_l40_40399


namespace new_number_of_groups_l40_40751

-- Define the number of students
def total_students : ℕ := 2808

-- Define the initial and new number of groups
def initial_groups (n : ℕ) : ℕ := n + 4
def new_groups (n : ℕ) : ℕ := n

-- Condition: Fewer than 30 students per new group
def fewer_than_30_students_per_group (n : ℕ) : Prop :=
  total_students / n < 30

-- Condition: n and n + 4 must be divisors of total_students
def is_divisor (d : ℕ) (a : ℕ) : Prop :=
  a % d = 0

def valid_group_numbers (n : ℕ) : Prop :=
  is_divisor n total_students ∧ is_divisor (n + 4) total_students ∧ n > 93

-- The main theorem
theorem new_number_of_groups : ∃ n : ℕ, valid_group_numbers n ∧ fewer_than_30_students_per_group n ∧ n = 104 :=
by
  sorry

end new_number_of_groups_l40_40751


namespace arithmetic_sequence_n_value_l40_40413

theorem arithmetic_sequence_n_value (a_1 d a_nm1 n : ℤ) (h1 : a_1 = -1) (h2 : d = 2) (h3 : a_nm1 = 15) :
    a_nm1 = a_1 + (n - 2) * d → n = 10 :=
by
  intros h
  sorry

end arithmetic_sequence_n_value_l40_40413


namespace fraction_historical_fiction_new_releases_l40_40219

-- Define constants for book categories and new releases
def historical_fiction_percentage : ℝ := 0.40
def science_fiction_percentage : ℝ := 0.25
def biographies_percentage : ℝ := 0.15
def mystery_novels_percentage : ℝ := 0.20

def historical_fiction_new_releases : ℝ := 0.45
def science_fiction_new_releases : ℝ := 0.30
def biographies_new_releases : ℝ := 0.50
def mystery_novels_new_releases : ℝ := 0.35

-- Statement of the problem to prove
theorem fraction_historical_fiction_new_releases :
  (historical_fiction_percentage * historical_fiction_new_releases) /
    (historical_fiction_percentage * historical_fiction_new_releases +
     science_fiction_percentage * science_fiction_new_releases +
     biographies_percentage * biographies_new_releases +
     mystery_novels_percentage * mystery_novels_new_releases) = 9 / 20 :=
by
  sorry

end fraction_historical_fiction_new_releases_l40_40219


namespace tens_digit_13_power_1987_l40_40404

theorem tens_digit_13_power_1987 : (13^1987)%100 / 10 = 1 :=
by
  sorry

end tens_digit_13_power_1987_l40_40404


namespace tan_alpha_sqrt_15_over_15_l40_40119

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l40_40119


namespace g_26_equals_125_l40_40299

noncomputable def g : ℕ → ℕ := sorry

axiom g_property : ∀ x, g (x + g x) = 5 * g x
axiom g_initial : g 1 = 5

theorem g_26_equals_125 : g 26 = 125 :=
by
  sorry

end g_26_equals_125_l40_40299


namespace sum_faces_edges_vertices_l40_40671

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l40_40671


namespace minimum_attacking_pairs_l40_40147

noncomputable theory

open_locale big_operators

-- Define the chessboard as an 8x8 grid
def Chessboard := Fin 8 × Fin 8

-- Define rooks' placement on the chessboard
def rooks_placement (placements : Finset Chessboard) := 
  placements.card = 16

-- Define the attacking pairs of rooks
def attacking_pairs (placements : Finset Chessboard) : ℕ :=
  let row_contrib := λ (r : Fin 8), (placements.filter (λ p, p.1 = r)).card - 1
  let col_contrib := λ (c : Fin 8), (placements.filter (λ p, p.2 = c)).card - 1
  ∑ r, row_contrib r + ∑ c, col_contrib c

-- The theorem states the minimum number of attacking pairs of rooks in a valid placement
theorem minimum_attacking_pairs (placements : Finset Chessboard)
  (h : rooks_placement placements) :
  attacking_pairs placements = 16 :=
sorry

end minimum_attacking_pairs_l40_40147


namespace smallest_y_value_l40_40655

theorem smallest_y_value (y : ℝ) : (12 * y^2 - 56 * y + 48 = 0) → y = 2 :=
by
  sorry

end smallest_y_value_l40_40655


namespace no_entangled_two_digit_numbers_l40_40529

theorem no_entangled_two_digit_numbers :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 → 10 * a + b ≠ 2 * (a + b ^ 3) :=
by
  intros a b h
  rcases h with ⟨ha1, ha9, hb9⟩
  sorry

end no_entangled_two_digit_numbers_l40_40529


namespace total_wheels_in_garage_l40_40490

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end total_wheels_in_garage_l40_40490


namespace girls_dropped_out_l40_40641

theorem girls_dropped_out (B_initial G_initial B_dropped G_remaining S_remaining : ℕ)
  (hB_initial : B_initial = 14)
  (hG_initial : G_initial = 10)
  (hB_dropped : B_dropped = 4)
  (hS_remaining : S_remaining = 17)
  (hB_remaining : B_initial - B_dropped = B_remaining)
  (hG_remaining : G_remaining = S_remaining - B_remaining) :
  (G_initial - G_remaining) = 3 := 
by 
  sorry

end girls_dropped_out_l40_40641


namespace arrangement_ways_l40_40390

def num_ways_arrange_boys_girls : Nat :=
  let boys := 2
  let girls := 3
  let ways_girls := Nat.factorial girls
  let ways_boys := Nat.factorial boys
  ways_girls * ways_boys

theorem arrangement_ways : num_ways_arrange_boys_girls = 12 :=
  by
    sorry

end arrangement_ways_l40_40390


namespace probability_sum_is_prime_l40_40240

-- Define the set of the first ten primes
def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define a function that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  ∃ p, p > 1 ∧ p ≤ n ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Define a function that checks if the sum of two primes is a prime
def is_sum_prime (a b : ℕ) : Prop :=
  is_prime (a + b)

-- Define the problem of calculating the required probability
theorem probability_sum_is_prime :
  let totalPairs := (firstTenPrimes.to_finset.powerset.card) / 2
  let validPairs := (({(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)}).card)
  validPairs / totalPairs = (1 : ℕ) / (9 : ℕ) := by
  sorry

end probability_sum_is_prime_l40_40240


namespace tan_alpha_solution_l40_40123

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l40_40123


namespace each_friend_paid_l40_40909

def cottage_cost_per_hour : ℕ := 5
def rental_duration_hours : ℕ := 8
def total_cost := cottage_cost_per_hour * rental_duration_hours
def cost_per_person := total_cost / 2

theorem each_friend_paid : cost_per_person = 20 :=
by 
  sorry

end each_friend_paid_l40_40909


namespace hcf_lcm_fraction_l40_40629

theorem hcf_lcm_fraction (m n : ℕ) (HCF : Nat.gcd m n = 6) (LCM : Nat.lcm m n = 210) (sum_mn : m + n = 72) : 
  (1 / m : ℚ) + (1 / n : ℚ) = 2 / 35 :=
by
  sorry

end hcf_lcm_fraction_l40_40629


namespace graphs_intersect_at_one_point_l40_40847

theorem graphs_intersect_at_one_point (a : ℝ) : 
  (∀ x : ℝ, (a * x^2 + 3 * x + 1 = -x - 1) ↔ a = 2) :=
by
  sorry

end graphs_intersect_at_one_point_l40_40847


namespace random_variable_is_zero_with_prob_one_l40_40450

universe u
open ProbabilityTheory

-- Define the Lean 4 statement
theorem random_variable_is_zero_with_prob_one {Ω : Type u} {X Y : Ω → ℝ}
(HX1 : integrable X) (HY1 : integrable Y) (HYX : ae_eq_fun (E[Y | X]) (fun ω => 0))
(HYX_Y : ae_eq_fun (E[Y | fun ω => X ω + Y ω]) (fun ω => 0)) :
  ae_eq_fun (fun ω => Y ω) (fun ω => 0) :=
sorry

end random_variable_is_zero_with_prob_one_l40_40450


namespace each_friend_pays_20_l40_40907

def rent_cottage_cost_per_hour : ℕ := 5
def rent_cottage_hours : ℕ := 8
def total_rent_cost := rent_cottage_cost_per_hour * rent_cottage_hours
def number_of_friends : ℕ := 2
def each_friend_pays := total_rent_cost / number_of_friends

theorem each_friend_pays_20 :
  each_friend_pays = 20 := by
  sorry

end each_friend_pays_20_l40_40907


namespace last_digit_base5_of_M_l40_40728

theorem last_digit_base5_of_M (d e f : ℕ) (hd : d < 5) (he : e < 5) (hf : f < 5)
  (h : 25 * d + 5 * e + f = 64 * f + 8 * e + d) : f = 0 :=
by
  sorry

end last_digit_base5_of_M_l40_40728


namespace waiting_time_probability_l40_40593

theorem waiting_time_probability :
  (∀ (t : ℝ), 0 ≤ t ∧ t < 30 → (1 / 30) * (if t < 25 then 5 else 5 - (t - 25)) = 1 / 6) :=
by
  sorry

end waiting_time_probability_l40_40593


namespace tan_alpha_solution_l40_40122

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l40_40122


namespace jim_total_cars_l40_40915

theorem jim_total_cars (B F C : ℕ) (h1 : B = 4 * F) (h2 : F = 2 * C + 3) (h3 : B = 220) :
  B + F + C = 301 :=
by
  sorry

end jim_total_cars_l40_40915


namespace number_of_ways_to_score_2018_l40_40899

theorem number_of_ways_to_score_2018 : 
  let combinations_count := nat.choose 2021 3
  in combinations_count = 1373734330 := 
by {
  -- This is the placeholder for the proof
  sorry
}

end number_of_ways_to_score_2018_l40_40899


namespace incorrect_option_D_l40_40586

variable {p q : Prop}

theorem incorrect_option_D (hp : ¬p) (hq : q) : ¬(¬q) := 
by 
  sorry  

end incorrect_option_D_l40_40586


namespace max_product_of_two_integers_with_sum_180_l40_40372

theorem max_product_of_two_integers_with_sum_180 :
  ∃ x y : ℤ, (x + y = 180) ∧ (x * y = 8100) := by
  sorry

end max_product_of_two_integers_with_sum_180_l40_40372


namespace least_positive_integer_with_12_factors_l40_40027

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l40_40027


namespace least_positive_integer_with_12_factors_l40_40012

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l40_40012


namespace prob_prime_sum_l40_40256

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := nat.prime n

def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (List.product l l).filter (λ p, p.1 < p.2)

def prime_sum_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  valid_pairs l |>.filter (λ p, is_prime (p.1 + p.2))

theorem prob_prime_sum :
  prime_sum_pairs primes.length / (valid_pairs primes).length = 1 / 9 :=
by
  sorry -- Proof is not required

end prob_prime_sum_l40_40256


namespace boxwoods_shaped_into_spheres_l40_40617

theorem boxwoods_shaped_into_spheres :
  ∀ (total_boxwoods : ℕ) (cost_trimming : ℕ) (cost_shaping : ℕ) (total_charge : ℕ) (x : ℕ),
    total_boxwoods = 30 →
    cost_trimming = 5 →
    cost_shaping = 15 →
    total_charge = 210 →
    30 * 5 + x * 15 = 210 →
    x = 4 :=
by
  intros total_boxwoods cost_trimming cost_shaping total_charge x
  rintro rfl rfl rfl rfl h
  sorry

end boxwoods_shaped_into_spheres_l40_40617


namespace tan_alpha_solution_l40_40124

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l40_40124


namespace total_distance_travelled_l40_40850

def speed_one_sail : ℕ := 25 -- knots
def speed_two_sails : ℕ := 50 -- knots
def conversion_factor : ℕ := 115 -- 1.15, in hundredths

def distance_in_nautical_miles : ℕ :=
  (2 * speed_one_sail) +      -- Two hours, one sail
  (3 * speed_two_sails) +     -- Three hours, two sails
  (1 * speed_one_sail) +      -- One hour, one sail, navigating around obstacles
  (2 * (speed_one_sail - speed_one_sail * 30 / 100)) -- Two hours, strong winds, 30% reduction in speed

def distance_in_land_miles : ℕ :=
  distance_in_nautical_miles * conversion_factor / 100 -- Convert to land miles

theorem total_distance_travelled : distance_in_land_miles = 299 := by
  sorry

end total_distance_travelled_l40_40850


namespace Oshea_needs_30_small_planters_l40_40167

theorem Oshea_needs_30_small_planters 
  (total_seeds : ℕ) 
  (large_planters : ℕ) 
  (capacity_large : ℕ) 
  (capacity_small : ℕ)
  (h1: total_seeds = 200) 
  (h2: large_planters = 4) 
  (h3: capacity_large = 20) 
  (h4: capacity_small = 4) : 
  (total_seeds - large_planters * capacity_large) / capacity_small = 30 :=
by 
  sorry

end Oshea_needs_30_small_planters_l40_40167


namespace polar_to_rectangular_l40_40069

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 3 * Real.sqrt 2) (h2 : θ = Real.pi / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (3, 3) :=
by
  -- Proof goes here
  sorry

end polar_to_rectangular_l40_40069


namespace average_students_per_bus_l40_40166

-- Definitions
def total_students : ℕ := 396
def students_in_cars : ℕ := 18
def number_of_buses : ℕ := 7

-- Proof problem statement
theorem average_students_per_bus : (total_students - students_in_cars) / number_of_buses = 54 := by
  sorry

end average_students_per_bus_l40_40166


namespace find_a_and_b_solve_inequality_l40_40080

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem find_a_and_b (a b : ℝ) (h : ∀ x : ℝ, f x a b > 0 ↔ x < 0 ∨ x > 2) : a = -2 ∧ b = 0 :=
by sorry

theorem solve_inequality (a b : ℝ) (m : ℝ) (h1 : a = -2) (h2 : b = 0) :
  (∀ x : ℝ, f x a b < m^2 - 1 ↔ 
    (m = 0 → ∀ x : ℝ, false) ∧
    (m > 0 → (1 - m < x ∧ x < 1 + m)) ∧
    (m < 0 → (1 + m < x ∧ x < 1 - m))) :=
by sorry

end find_a_and_b_solve_inequality_l40_40080


namespace initial_potatoes_count_l40_40977

theorem initial_potatoes_count (initial_tomatoes picked_tomatoes total_remaining : ℕ) 
    (h_initial_tomatoes : initial_tomatoes = 177)
    (h_picked_tomatoes : picked_tomatoes = 53)
    (h_total_remaining : total_remaining = 136) :
  (initial_tomatoes - picked_tomatoes + x = total_remaining) → 
  x = 12 :=
by 
  sorry

end initial_potatoes_count_l40_40977


namespace power_sum_zero_l40_40485

theorem power_sum_zero (n : ℕ) (h : 0 < n) : (-1:ℤ)^(2*n) + (-1:ℤ)^(2*n+1) = 0 := 
by 
  sorry

end power_sum_zero_l40_40485


namespace least_integer_with_twelve_factors_l40_40013

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l40_40013


namespace probability_sum_two_primes_is_prime_l40_40261

open Finset

-- First 10 prime numbers
def firstTenPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Function to sum of two primes
def sum_of_two_primes_is_prime (a b : ℕ) : Prop := is_prime (a + b)

-- Function to compute the probability
def probability_two_prime_sum_is_prime : ℚ :=
  let pairs := firstTenPrimes.pairs
  let prime_pairs := pairs.filter (λ pair, sum_of_two_primes_is_prime pair.1 pair.2)
  (prime_pairs.card : ℚ) / pairs.card

theorem probability_sum_two_primes_is_prime :
  probability_two_prime_sum_is_prime = 1 / 9 :=
by sorry

end probability_sum_two_primes_is_prime_l40_40261


namespace min_value_frac_l40_40761

open Real

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 1) :
  (1 / a + 2 / b) = 9 + 4 * sqrt 2 :=
sorry

end min_value_frac_l40_40761


namespace prob_prime_sum_l40_40258

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := nat.prime n

def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (List.product l l).filter (λ p, p.1 < p.2)

def prime_sum_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  valid_pairs l |>.filter (λ p, is_prime (p.1 + p.2))

theorem prob_prime_sum :
  prime_sum_pairs primes.length / (valid_pairs primes).length = 1 / 9 :=
by
  sorry -- Proof is not required

end prob_prime_sum_l40_40258


namespace rectangular_prism_faces_edges_vertices_sum_l40_40677

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l40_40677


namespace most_suitable_sampling_method_l40_40042

/-- A unit has 28 elderly people, 54 middle-aged people, and 81 young people. 
    A sample of 36 people needs to be drawn in a way that accounts for age.
    The most suitable method for drawing a sample is to exclude one elderly person first,
    then use stratified sampling. -/
theorem most_suitable_sampling_method 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (sample_size : ℕ) (suitable_method : String)
  (condition1 : elderly = 28) 
  (condition2 : middle_aged = 54) 
  (condition3 : young = 81) 
  (condition4 : sample_size = 36) 
  (condition5 : suitable_method = "Exclude one elderly person first, then stratify sampling") : 
  suitable_method = "Exclude one elderly person first, then stratify sampling" := 
by sorry

end most_suitable_sampling_method_l40_40042


namespace parallel_heater_time_l40_40502

theorem parallel_heater_time (t1 t2 : ℕ) (R1 R2 : ℝ) (t : ℕ) (I : ℝ) (Q : ℝ) (h₁ : t1 = 3) 
  (h₂ : t2 = 6) (hq1 : Q = I^2 * R1 * t1) (hq2 : Q = I^2 * R2 * t2) :
  t = (t1 * t2) / (t1 + t2) := by
  sorry

end parallel_heater_time_l40_40502


namespace sum_not_prime_l40_40929

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := 
sorry

end sum_not_prime_l40_40929


namespace distance_between_homes_l40_40452

theorem distance_between_homes (Maxwell_speed : ℝ) (Brad_speed : ℝ) (M_time : ℝ) (B_delay : ℝ) (D : ℝ) 
  (h1 : Maxwell_speed = 4) 
  (h2 : Brad_speed = 6)
  (h3 : M_time = 8)
  (h4 : B_delay = 1) :
  D = 74 :=
by
  sorry

end distance_between_homes_l40_40452


namespace problem1_solution_problem2_solution_l40_40970

-- Problem 1: f(x-2) = 3x - 5 implies f(x) = 3x + 1
def problem1 (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f (x - 2) = 3 * x - 5 → f x = 3 * x + 1

-- Problem 2: Quadratic function satisfying specific conditions
def is_quadratic (f : ℝ → ℝ) : Prop := 
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a*x^2 + b*x + c

def problem2 (f : ℝ → ℝ) : Prop :=
  is_quadratic f ∧
  (f 0 = 4) ∧
  (∀ x : ℝ, f (3 - x) = f x) ∧
  (∀ x : ℝ, f x ≥ 7/4) →
  (∀ x : ℝ, f x = x^2 - 3*x + 4)

-- Statements to be proved
theorem problem1_solution : ∀ f : ℝ → ℝ, problem1 x f := sorry
theorem problem2_solution : ∀ f : ℝ → ℝ, problem2 f := sorry

end problem1_solution_problem2_solution_l40_40970


namespace units_digit_of_square_l40_40336

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 :=
by
  sorry

end units_digit_of_square_l40_40336


namespace measure_angle_PQR_is_55_l40_40713

noncomputable def measure_angle_PQR (POQ QOR : ℝ) : ℝ :=
  let POQ := 120
  let QOR := 130
  let POR := 360 - (POQ + QOR)
  let OPR := (180 - POR) / 2
  let OPQ := (180 - POQ) / 2
  let OQR := (180 - QOR) / 2
  OPQ + OQR

theorem measure_angle_PQR_is_55 : measure_angle_PQR 120 130 = 55 := by
  sorry

end measure_angle_PQR_is_55_l40_40713


namespace perp_lines_solution_l40_40300

theorem perp_lines_solution (a : ℝ) :
  ((a+2) * (a-1) + (1-a) * (2*a + 3) = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end perp_lines_solution_l40_40300


namespace sum_of_two_digit_integers_l40_40656

theorem sum_of_two_digit_integers :
  let a := 10
  let l := 99
  let d := 1
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = 4905 :=
by
  sorry

end sum_of_two_digit_integers_l40_40656


namespace painted_cube_l40_40383

noncomputable def cube_side_length : ℕ :=
  7

theorem painted_cube (painted_faces: ℕ) (one_side_painted_cubes: ℕ) (orig_side_length: ℕ) :
    painted_faces = 6 ∧ one_side_painted_cubes = 54 ∧ (orig_side_length + 2) ^ 2 / 6 = 9 →
    orig_side_length = cube_side_length :=
by
  sorry

end painted_cube_l40_40383


namespace mailman_junk_mail_l40_40838

theorem mailman_junk_mail (total_mail : ℕ) (magazines : ℕ) (junk_mail : ℕ) 
  (h1 : total_mail = 11) (h2 : magazines = 5) (h3 : junk_mail = total_mail - magazines) : junk_mail = 6 := by
  sorry

end mailman_junk_mail_l40_40838


namespace XYZStockPriceIs75_l40_40224

/-- XYZ stock price model 
Starts at $50, increases by 200% in first year, 
then decreases by 50% in second year.
-/
def XYZStockPriceEndOfSecondYear : ℝ :=
  let initialPrice := 50
  let firstYearIncreaseRate := 2.0
  let secondYearDecreaseRate := 0.5
  let priceAfterFirstYear := initialPrice * (1 + firstYearIncreaseRate)
  let priceAfterSecondYear := priceAfterFirstYear * (1 - secondYearDecreaseRate)
  priceAfterSecondYear

theorem XYZStockPriceIs75 : XYZStockPriceEndOfSecondYear = 75 := by
  sorry

end XYZStockPriceIs75_l40_40224


namespace calc_product_eq_243_l40_40992

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l40_40992


namespace line_intersects_x_axis_at_point_l40_40836

theorem line_intersects_x_axis_at_point (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (7, -3))
  (h2 : (x2, y2) = (3, 1)) : 
  ∃ x, (x, 0) = (4, 0) :=
by
  -- sorry serves as a placeholder for the actual proof
  sorry

end line_intersects_x_axis_at_point_l40_40836


namespace range_of_m_l40_40408

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l40_40408


namespace sum_of_faces_edges_vertices_l40_40659

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l40_40659


namespace remaining_laps_l40_40216

def track_length : ℕ := 9
def initial_laps : ℕ := 6
def total_distance : ℕ := 99

theorem remaining_laps : (total_distance - (initial_laps * track_length)) / track_length = 5 := by
  sorry

end remaining_laps_l40_40216


namespace cylinder_in_sphere_volume_difference_is_correct_l40_40983

noncomputable def volume_difference (base_radius_cylinder : ℝ) (radius_sphere : ℝ) : ℝ :=
  let height_cylinder := Real.sqrt (radius_sphere^2 - base_radius_cylinder^2)
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere^3
  let volume_cylinder := Real.pi * base_radius_cylinder^2 * height_cylinder
  volume_sphere - volume_cylinder

theorem cylinder_in_sphere_volume_difference_is_correct :
  volume_difference 4 7 = (1372 - 48 * Real.sqrt 33) / 3 * Real.pi :=
by
  sorry

end cylinder_in_sphere_volume_difference_is_correct_l40_40983


namespace tom_final_payment_l40_40803

noncomputable def cost_of_fruit (kg: ℝ) (rate_per_kg: ℝ) := kg * rate_per_kg

noncomputable def total_bill := 
  cost_of_fruit 15.3 1.85 + cost_of_fruit 12.7 2.45 + cost_of_fruit 10.5 3.20 + cost_of_fruit 6.2 4.50

noncomputable def discount (bill: ℝ) := 0.10 * bill

noncomputable def discounted_total (bill: ℝ) := bill - discount bill

noncomputable def sales_tax (amount: ℝ) := 0.06 * amount

noncomputable def final_amount (bill: ℝ) := discounted_total bill + sales_tax (discounted_total bill)

theorem tom_final_payment : final_amount total_bill = 115.36 :=
  sorry

end tom_final_payment_l40_40803


namespace triangle_area_l40_40729

theorem triangle_area (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 54 := by
  -- conditions provided
  sorry

end triangle_area_l40_40729


namespace angle_in_third_quadrant_l40_40141

theorem angle_in_third_quadrant
  (α : ℝ) (hα : 270 < α ∧ α < 360) : 90 < 180 - α ∧ 180 - α < 180 :=
by
  sorry

end angle_in_third_quadrant_l40_40141


namespace tan_alpha_value_l40_40096

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l40_40096


namespace smallest_value_satisfies_equation_l40_40865

theorem smallest_value_satisfies_equation : ∃ x : ℝ, (|5 * x + 9| = 34) ∧ x = -8.6 :=
by
  sorry

end smallest_value_satisfies_equation_l40_40865


namespace factorial_sum_power_of_two_l40_40853

theorem factorial_sum_power_of_two (a b c : ℕ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) :
  a! + b! = 2 ^ c! ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) :=
by
  sorry

end factorial_sum_power_of_two_l40_40853


namespace length_CK_angle_BCA_l40_40604

variables {A B C O O₁ O₂ K K₁ K₂ K₃ : Point}
variables {r R : ℝ}
variables {AC CK AK₁ AK₂ : ℝ}

-- Definitions and conditions
def triangle_ABC (A B C : Point) : Prop := True
def incenter (A B C O : Point) : Prop := True
def in_radius_is_equal (O₁ O₂ : Point) (r : ℝ) : Prop := True
def circle_touches_side (circle_center : Point) (side_point : Point) (distance : ℝ) : Prop := True
def circumcenter (A C B O₁ : Point) : Prop := True
def angle (A B C : Point) (θ : ℝ) : Prop := True

-- Conditions from the problem
axiom cond1 : triangle_ABC A B C
axiom cond2 : in_radius_is_equal O₁ O₂ r
axiom cond3 : incenter A B C O
axiom cond4 : circle_touches_side O₁ K₁ 6
axiom cond5 : circle_touches_side O₂ K₂ 8
axiom cond6 : AC = 21
axiom cond7 : circle_touches_side O K 9
axiom cond8 : circumcenter O K₁ K₃ O₁

-- Statements to prove
theorem length_CK : CK = 9 := by
  sorry

theorem angle_BCA : angle B C A 60 := by
  sorry

end length_CK_angle_BCA_l40_40604


namespace tan_alpha_value_l40_40099

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l40_40099


namespace find_q_l40_40581

variable (p q : ℝ)

theorem find_q (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end find_q_l40_40581


namespace remaining_pages_after_a_week_l40_40185

-- Define the conditions
def total_pages : Nat := 381
def pages_read_initial : Nat := 149
def pages_per_day : Nat := 20
def days : Nat := 7

-- Define the final statement to prove
theorem remaining_pages_after_a_week :
  let pages_left_initial := total_pages - pages_read_initial
  let pages_read_week := pages_per_day * days
  let pages_remaining := pages_left_initial - pages_read_week
  pages_remaining = 92 := by
  sorry

end remaining_pages_after_a_week_l40_40185


namespace inequality_to_prove_l40_40337

variable (x y z : ℝ)

axiom h1 : 0 ≤ x
axiom h2 : 0 ≤ y
axiom h3 : 0 ≤ z
axiom h4 : y * z + z * x + x * y = 1

theorem inequality_to_prove : x * (1 - y)^2 * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4 / 9) * Real.sqrt 3 :=
by 
  -- The proof is omitted.
  sorry

end inequality_to_prove_l40_40337


namespace committee_count_l40_40438

theorem committee_count :
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  eligible_owners.choose committee_size = 65780 := by
  let total_owners := 30
  let not_willing := 3
  let eligible_owners := total_owners - not_willing
  let committee_size := 5
  have lean_theorem : eligible_owners.choose committee_size = 65780 := sorry
  exact lean_theorem

end committee_count_l40_40438


namespace tan_alpha_fraction_l40_40101

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l40_40101


namespace least_positive_integer_with_12_factors_l40_40025

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l40_40025


namespace smallest_M_l40_40329

def Q (M : ℕ) := (2 * M / 3 + 1) / (M + 1)

theorem smallest_M (M : ℕ) (h : M % 6 = 0) (h_pos : 0 < M) : 
  (∃ k, M = 6 * k ∧ Q M < 3 / 4) ↔ M = 6 := 
by 
  sorry

end smallest_M_l40_40329


namespace sufficient_conditions_for_positive_product_l40_40800

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1) → a * b > 0 :=
by sorry

end sufficient_conditions_for_positive_product_l40_40800


namespace minimize_distance_l40_40084

noncomputable def f (x : ℝ) := x^2 - 2 * x
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, f x)
def Q : ℝ × ℝ := (4, -1)

theorem minimize_distance : ∃ (x : ℝ), dist (P x) Q = Real.sqrt 5 := by
  sorry

end minimize_distance_l40_40084


namespace range_of_a_l40_40311

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 - a) * x^2 + 2 * (2 - a) * x + 4 ≥ 0) → (-2 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l40_40311


namespace angle_of_inclination_l40_40505

noncomputable def line_slope (a b : ℝ) : ℝ := 1  -- The slope of the line y = x + 1 is 1
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m -- angle of inclination is arctan of the slope

theorem angle_of_inclination (θ : ℝ) : 
  inclination_angle (line_slope 1 1) = 45 :=
by
  sorry

end angle_of_inclination_l40_40505


namespace strong_2013_l40_40526

theorem strong_2013 :
  ∃ x : ℕ, x > 0 ∧ (x ^ (2013 * x) + 1) % (2 ^ 2013) = 0 :=
sorry

end strong_2013_l40_40526


namespace lcm_48_180_l40_40563

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l40_40563


namespace tan_alpha_solution_l40_40114

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l40_40114


namespace tan_alpha_value_l40_40126

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l40_40126


namespace problem_part_I_problem_part_II_l40_40143

-- Problem (I)
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (h1 : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : 
  a + b = 2 * c -> (a + b) = 2 * c :=
by
  intros h
  sorry

-- Problem (II)
theorem problem_part_II (a b c : ℝ) (A B C : ℝ) 
  (h1 : C = Real.pi / 3) 
  (h2 : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) 
  (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
  (h4 : a + b = 2 * c) : c = 4 :=
by
  intros
  sorry

end problem_part_I_problem_part_II_l40_40143


namespace tan_alpha_solution_l40_40120

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l40_40120


namespace sum_base_49_l40_40613

-- Definitions of base b numbers and their base 10 conversion
def num_14_in_base (b : ℕ) : ℕ := b + 4
def num_17_in_base (b : ℕ) : ℕ := b + 7
def num_18_in_base (b : ℕ) : ℕ := b + 8
def num_6274_in_base (b : ℕ) : ℕ := 6 * b^3 + 2 * b^2 + 7 * b + 4

-- The question: Compute 14 + 17 + 18 in base b
def sum_in_base (b : ℕ) : ℕ := 14 + 17 + 18

-- The main statement to prove
theorem sum_base_49 (b : ℕ) (h : (num_14_in_base b) * (num_17_in_base b) * (num_18_in_base b) = num_6274_in_base (b)) :
  sum_in_base b = 49 :=
by sorry

end sum_base_49_l40_40613


namespace remaining_students_l40_40950

def students_remaining (n1 n2 n_leaving1 n_leaving2 : Nat) : Nat :=
  (n1 * 4 - n_leaving1) + (n2 * 2 - n_leaving2)

theorem remaining_students :
  students_remaining 15 18 8 5 = 83 := 
by
  sorry

end remaining_students_l40_40950


namespace range_of_k_l40_40875

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > k → (3 / (x + 1) < 1)) ↔ k ≥ 2 := sorry

end range_of_k_l40_40875


namespace crop_yield_growth_l40_40328

-- Definitions based on conditions
def initial_yield := 300
def final_yield := 363
def eqn (x : ℝ) : Prop := initial_yield * (1 + x)^2 = final_yield

-- The theorem we need to prove
theorem crop_yield_growth (x : ℝ) : eqn x :=
by
  sorry

end crop_yield_growth_l40_40328


namespace metro_earnings_in_6_minutes_l40_40459

theorem metro_earnings_in_6_minutes 
  (ticket_cost : ℕ) 
  (tickets_per_minute : ℕ) 
  (duration_minutes : ℕ) 
  (earnings_in_one_minute : ℕ) 
  (earnings_in_six_minutes : ℕ) 
  (h1 : ticket_cost = 3) 
  (h2 : tickets_per_minute = 5) 
  (h3 : duration_minutes = 6) 
  (h4 : earnings_in_one_minute = tickets_per_minute * ticket_cost) 
  (h5 : earnings_in_six_minutes = earnings_in_one_minute * duration_minutes) 
  : earnings_in_six_minutes = 90 := 
by 
  -- Proof goes here
  sorry

end metro_earnings_in_6_minutes_l40_40459


namespace maximum_n_for_sequence_l40_40412

theorem maximum_n_for_sequence :
  ∃ (n : ℕ), 
  (∀ a S : ℕ → ℝ, 
    a 1 = 1 → 
    (∀ n : ℕ, n > 0 → 2 * a (n + 1) + S n = 2) → 
    (1001 / 1000 < S (2 * n) / S n ∧ S (2 * n) / S n < 11 / 10)) →
  n = 9 :=
sorry

end maximum_n_for_sequence_l40_40412


namespace equation_of_plane_passing_through_points_l40_40864

/-
Let M1, M2, and M3 be points in three-dimensional space.
M1 = (1, 2, 0)
M2 = (1, -1, 2)
M3 = (0, 1, -1)
We need to prove that the plane passing through these points has the equation 5x - 2y - 3z - 1 = 0.
-/

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M1 : Point3D := ⟨1, 2, 0⟩
def M2 : Point3D := ⟨1, -1, 2⟩
def M3 : Point3D := ⟨0, 1, -1⟩

theorem equation_of_plane_passing_through_points :
  ∃ (a b c d : ℝ), (∀ (P : Point3D), 
  P = M1 ∨ P = M2 ∨ P = M3 → a * P.x + b * P.y + c * P.z + d = 0)
  ∧ a = 5 ∧ b = -2 ∧ c = -3 ∧ d = -1 :=
by
  sorry

end equation_of_plane_passing_through_points_l40_40864


namespace x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l40_40569

theorem x_is_sufficient_but_not_necessary_for_x_squared_eq_one : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
by
  sorry

end x_is_sufficient_but_not_necessary_for_x_squared_eq_one_l40_40569


namespace least_positive_integer_with_12_factors_is_96_l40_40020

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l40_40020


namespace possible_values_of_a_l40_40421

theorem possible_values_of_a (a : ℚ) : 
  (a^2 = 9 * 16) ∨ (16 * a = 81) ∨ (9 * a = 256) → 
  a = 12 ∨ a = -12 ∨ a = 81 / 16 ∨ a = 256 / 9 :=
by
  intros h
  sorry

end possible_values_of_a_l40_40421


namespace bathroom_length_l40_40709

theorem bathroom_length (A L W : ℝ) (h₁ : A = 8) (h₂ : W = 2) (h₃ : A = L * W) : L = 4 :=
by
  -- Skip the proof with sorry
  sorry

end bathroom_length_l40_40709


namespace prime_pair_probability_l40_40231

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrimeSum (a b : ℕ) : Prop := Nat.Prime (a + b)

theorem prime_pair_probability :
  let pairs := List.sublistsLen 2 firstTenPrimes
  let primePairs := pairs.filter (λ p, match p with
                                       | [a, b] => isPrimeSum a b
                                       | _ => false
                                       end)
  (primePairs.length : ℚ) / (pairs.length : ℚ) = 1 / 9 :=
by
  sorry

end prime_pair_probability_l40_40231


namespace proposition_D_correct_l40_40988

theorem proposition_D_correct :
  ∀ x : ℝ, x^2 + x + 2 > 0 :=
by
  sorry

end proposition_D_correct_l40_40988


namespace convert_to_cylindrical_l40_40400

theorem convert_to_cylindrical (x y z : ℝ) (r θ : ℝ) 
  (h₀ : x = 3) 
  (h₁ : y = -3 * Real.sqrt 3) 
  (h₂ : z = 2) 
  (h₃ : r > 0) 
  (h₄ : 0 ≤ θ) 
  (h₅ : θ < 2 * Real.pi) 
  (h₆ : r = Real.sqrt (x^2 + y^2)) 
  (h₇ : x = r * Real.cos θ) 
  (h₈ : y = r * Real.sin θ) : 
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) :=
by
  -- Proof goes here
  sorry

end convert_to_cylindrical_l40_40400


namespace trig_expression_l40_40083

theorem trig_expression (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 3) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by sorry

end trig_expression_l40_40083


namespace tan_alpha_proof_l40_40105

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l40_40105


namespace correct_operation_l40_40963

theorem correct_operation (a : ℝ) : a^4 / a^2 = a^2 :=
by sorry

end correct_operation_l40_40963


namespace prime_sum_probability_l40_40239

open Set

noncomputable def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pairs : Finset (ℕ × ℕ) :=
  first_ten_primes.product first_ten_primes.filter (λ p, is_prime (p.1 + p.2))

theorem prime_sum_probability :
  (valid_pairs.card = 6) → ((first_ten_primes.card.choose 2 : ℚ) = 45) →
  Rat.mk valid_pairs.card (first_ten_primes.card.choose 2) = 2 / 15 :=
sorry

end prime_sum_probability_l40_40239


namespace clock_angle_at_3_15_is_7_5_l40_40956

def degrees_per_hour : ℝ := 360 / 12
def degrees_per_minute : ℝ := 6
def hour_hand_position (h m : ℝ) : ℝ := h * degrees_per_hour + 0.5 * m
def minute_hand_position (m : ℝ) : ℝ := m * degrees_per_minute
def clock_angle (h m : ℝ) : ℝ := abs(hour_hand_position h m - minute_hand_position m)

theorem clock_angle_at_3_15_is_7_5 :
  clock_angle 3 15 = 7.5 :=
by
  sorry

end clock_angle_at_3_15_is_7_5_l40_40956


namespace calculate_m_plus_n_l40_40745

noncomputable def shoe_pairs_probability_condition (k : ℕ) (n : ℕ) : Prop := 
  ∀ m, (0 < m ∧ m < k) → ¬ ∃ pairs, (pairs.card = m ∧ pairs ⊆ {1..n})

theorem calculate_m_plus_n : 
  let numAdults := 8
  let goodProbability := 21/112
  ∃ (m n : ℕ), shoe_pairs_probability_condition 4 numAdults (goodProbability = m / n) ∧ Nat.coprime m n ∧ m + n = 133 :=
by
  sorry

end calculate_m_plus_n_l40_40745


namespace group_B_population_calculation_l40_40425

variable {total_population : ℕ}
variable {sample_size : ℕ}
variable {sample_A : ℕ}
variable {total_B : ℕ}

theorem group_B_population_calculation 
  (h_total : total_population = 200)
  (h_sample_size : sample_size = 40)
  (h_sample_A : sample_A = 16)
  (h_sample_B : sample_size - sample_A = 24) :
  total_B = 120 :=
sorry

end group_B_population_calculation_l40_40425


namespace geometric_series_inequality_l40_40411

variables {x y : ℝ}

theorem geometric_series_inequality 
  (hx : |x| < 1) 
  (hy : |y| < 1) :
  (1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y)) :=
sorry

end geometric_series_inequality_l40_40411


namespace equal_real_roots_implies_m_l40_40432

theorem equal_real_roots_implies_m (m : ℝ) : (∃ (x : ℝ), x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) → m = 1/4 :=
by
  sorry

end equal_real_roots_implies_m_l40_40432


namespace percentile_75_eq_95_l40_40902

def seventy_fifth_percentile (data : List ℕ) : ℕ := sorry

theorem percentile_75_eq_95 : seventy_fifth_percentile [92, 93, 88, 99, 89, 95] = 95 := 
sorry

end percentile_75_eq_95_l40_40902


namespace total_seats_value_l40_40182

noncomputable def students_per_bus : ℝ := 14.0
noncomputable def number_of_buses : ℝ := 2.0
noncomputable def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_value : total_seats = 28.0 :=
by
  sorry

end total_seats_value_l40_40182


namespace tan_alpha_value_l40_40125

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l40_40125


namespace sequence_75th_term_l40_40314

theorem sequence_75th_term :
  ∀ (a d n : ℕ), a = 2 → d = 4 → n = 75 → a + (n-1) * d = 298 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  simp
  sorry

end sequence_75th_term_l40_40314


namespace arithmetic_expression_l40_40221

theorem arithmetic_expression : (5 * 7 - 6 + 2 * 12 + 2 * 6 + 7 * 3) = 86 :=
by
  sorry

end arithmetic_expression_l40_40221


namespace alice_walking_speed_l40_40987

theorem alice_walking_speed:
  ∃ v : ℝ, 
  (∀ t : ℝ, t = 1 → ∀ d_a d_b : ℝ, d_a = 25 → d_b = 41 - d_a → 
  ∀ s_b : ℝ, s_b = 4 → 
  d_b / s_b + t = d_a / v) ∧ v = 5 :=
by
  sorry

end alice_walking_speed_l40_40987


namespace sum_faces_edges_vertices_eq_26_l40_40668

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l40_40668


namespace jonathan_weekly_deficit_correct_l40_40918

def daily_intake_non_saturday : ℕ := 2500
def daily_intake_saturday : ℕ := 3500
def daily_burn : ℕ := 3000
def weekly_caloric_deficit : ℕ :=
  (7 * daily_burn) - ((6 * daily_intake_non_saturday) + daily_intake_saturday)

theorem jonathan_weekly_deficit_correct :
  weekly_caloric_deficit = 2500 :=
by
  unfold weekly_caloric_deficit daily_intake_non_saturday daily_intake_saturday daily_burn
  sorry

end jonathan_weekly_deficit_correct_l40_40918


namespace quadratic_has_at_most_two_solutions_l40_40171

theorem quadratic_has_at_most_two_solutions (a b c : ℝ) (h : a ≠ 0) :
  ¬(∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧ 
    a * x3^2 + b * x3 + c = 0) := 
by {
  sorry
}

end quadratic_has_at_most_two_solutions_l40_40171


namespace simplified_value_l40_40344

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log (3) / Real.log (20) + 1) + 
  1 / (Real.log (4) / Real.log (15) + 1) + 
  1 / (Real.log (7) / Real.log (12) + 1)

theorem simplified_value : simplify_expression = 2 :=
by {
  sorry
}

end simplified_value_l40_40344


namespace lizard_problem_theorem_l40_40912

def lizard_problem : Prop :=
  ∃ (E W S : ℕ), 
  E = 3 ∧ 
  W = 3 * E ∧ 
  S = 7 * W ∧ 
  (S + W) - E = 69

theorem lizard_problem_theorem : lizard_problem :=
by
  sorry

end lizard_problem_theorem_l40_40912


namespace prime_power_sum_l40_40860

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem prime_power_sum (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  is_perfect_square (p^q + p^r) →
  (p = 2 ∧ ((q = 2 ∧ r = 5) ∨ (q = 5 ∧ r = 2) ∨ (q ≥ 3 ∧ is_prime q ∧ q = r)))
  ∨
  (p = 3 ∧ ((q = 2 ∧ r = 3) ∨ (q = 3 ∧ r = 2))) :=
sorry

end prime_power_sum_l40_40860


namespace total_selling_price_correct_l40_40060

def meters_sold : ℕ := 85
def cost_price_per_meter : ℕ := 80
def profit_per_meter : ℕ := 25

def selling_price_per_meter : ℕ :=
  cost_price_per_meter + profit_per_meter

def total_selling_price : ℕ :=
  selling_price_per_meter * meters_sold

theorem total_selling_price_correct :
  total_selling_price = 8925 := by
  sorry

end total_selling_price_correct_l40_40060


namespace rectangular_prism_faces_edges_vertices_sum_l40_40676

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l40_40676


namespace trapezoid_length_property_l40_40645

noncomputable def trapezoid_properties (A B C D X : ℝ) :=
  -- Defining the conditions
  let α₁ := 6 * π / 180 -- ∠DAB in radians
  let α₂ := 42 * π / 180 -- ∠ABC in radians
  let α₃ := 78 * π / 180 -- ∠AXD in radians
  let α₄ := 66 * π / 180 -- ∠CXB in radians
  let h := 1 -- Distance between AB and CD 
  ∃ (AD DX BC CX : ℝ),
    (AD = 1 / (Real.sin α₁)) ∧ 
    (DX = 1 / (Real.sin α₃)) ∧
    (BC = 1 / (Real.sin α₂)) ∧
    (CX = 1 / (Real.sin α₄)) ∧
    (AD + DX - (BC + CX) = 8)

-- Final theorem statement
theorem trapezoid_length_property : ∀ (A B C D X : ℝ), 
  trapezoid_properties A B C D X :=
begin
  intros A B C D X,
  unfold trapezoid_properties,
  -- Proof skipped
  sorry
end

end trapezoid_length_property_l40_40645


namespace smallest_n_cond_l40_40531

theorem smallest_n_cond (n : ℕ) (h1 : n >= 100 ∧ n < 1000) (h2 : n ≡ 3 [MOD 9]) (h3 : n ≡ 3 [MOD 4]) : n = 111 := 
sorry

end smallest_n_cond_l40_40531


namespace maximum_price_for_360_skewers_price_for_1920_profit_l40_40818

-- Define the number of skewers sold as a function of the price
def skewers_sold (price : ℝ) : ℝ := 300 + 60 * (10 - price)

-- Define the profit as a function of the price
def profit (price : ℝ) : ℝ := (skewers_sold price) * (price - 3)

-- Maximum price for selling at least 360 skewers per day
theorem maximum_price_for_360_skewers (price : ℝ) (h : skewers_sold price ≥ 360) : price ≤ 9 :=
by {
    sorry
}

-- Price to achieve a profit of 1920 yuan per day with price constraint
theorem price_for_1920_profit (price : ℝ) (h₁ : profit price = 1920) (h₂ : price ≤ 8) : price = 7 :=
by {
    sorry
}

end maximum_price_for_360_skewers_price_for_1920_profit_l40_40818


namespace total_pencils_l40_40914

def pencils_per_person : Nat := 15
def number_of_people : Nat := 5

theorem total_pencils : pencils_per_person * number_of_people = 75 := by
  sorry

end total_pencils_l40_40914


namespace circles_intersect_probability_l40_40367

noncomputable def probability_circles_intersect : ℝ :=
  sorry

theorem circles_intersect_probability :
  probability_circles_intersect = (5 * Real.sqrt 2 - 7) / 4 :=
  sorry

end circles_intersect_probability_l40_40367


namespace sally_earnings_proof_l40_40341

def sally_last_month_earnings : ℝ := 1000
def raise_percentage : ℝ := 0.10
def sally_this_month_earnings := sally_last_month_earnings * (1 + raise_percentage)
def sally_total_two_months_earnings := sally_last_month_earnings + sally_this_month_earnings

theorem sally_earnings_proof :
  sally_total_two_months_earnings = 2100 :=
by
  sorry

end sally_earnings_proof_l40_40341


namespace prime_sum_probability_l40_40236

open Set

noncomputable def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pairs : Finset (ℕ × ℕ) :=
  first_ten_primes.product first_ten_primes.filter (λ p, is_prime (p.1 + p.2))

theorem prime_sum_probability :
  (valid_pairs.card = 6) → ((first_ten_primes.card.choose 2 : ℚ) = 45) →
  Rat.mk valid_pairs.card (first_ten_primes.card.choose 2) = 2 / 15 :=
sorry

end prime_sum_probability_l40_40236


namespace sum_of_faces_edges_vertices_rectangular_prism_l40_40686

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l40_40686


namespace calc_product_eq_243_l40_40994

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l40_40994


namespace algebraic_expression_equality_l40_40697

variable {x : ℝ}

theorem algebraic_expression_equality (h : x^2 + 3*x + 8 = 7) : 3*x^2 + 9*x - 2 = -5 := 
by
  sorry

end algebraic_expression_equality_l40_40697


namespace michelle_january_cost_l40_40048

noncomputable def cell_phone_cost (base_cost : ℕ) (text_rate : ℕ) (extra_minute_rate : ℕ)
  (included_hours : ℕ) (texts_sent : ℕ) (talked_hours : ℕ) : ℕ :=
  let cost_base := base_cost
  let cost_texts := texts_sent * text_rate / 100
  let extra_minutes := (talked_hours - included_hours) * 60
  let cost_extra_minutes := extra_minutes * extra_minute_rate / 100
  cost_base + cost_texts + cost_extra_minutes

theorem michelle_january_cost : cell_phone_cost 20 5 10 30 100 30.5 = 28 :=
by
  sorry

end michelle_january_cost_l40_40048


namespace eq_rectangular_eq_of_polar_eq_max_m_value_l40_40319

def polar_to_rectangular (ρ θ : ℝ) : Prop := (ρ = 4 * Real.cos θ) → ∀ x y : ℝ, ρ^2 = x^2 + y^2

theorem eq_rectangular_eq_of_polar_eq (ρ θ : ℝ) :
  polar_to_rectangular ρ θ → ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
sorry

def max_m_condition (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 → |4 + 2 * m| / Real.sqrt 5 ≤ 2

theorem max_m_value :
  (max_m_condition (Real.sqrt 5 - 2)) :=
sorry

end eq_rectangular_eq_of_polar_eq_max_m_value_l40_40319


namespace arina_should_accept_anton_offer_l40_40733

noncomputable def total_shares : ℕ := 300000
noncomputable def arina_shares : ℕ := 90001
noncomputable def need_to_be_largest : ℕ := 104999 
noncomputable def shares_needed : ℕ := 14999
noncomputable def largest_shareholder_total : ℕ := 105000

noncomputable def maxim_shares : ℕ := 104999
noncomputable def inga_shares : ℕ := 30000
noncomputable def yuri_shares : ℕ := 30000
noncomputable def yulia_shares : ℕ := 30000
noncomputable def anton_shares : ℕ := 15000

noncomputable def maxim_price_per_share : ℕ := 11
noncomputable def inga_price_per_share : ℕ := 1250 / 100
noncomputable def yuri_price_per_share : ℕ := 1150 / 100
noncomputable def yulia_price_per_share : ℕ := 1300 / 100
noncomputable def anton_price_per_share : ℕ := 14

noncomputable def anton_total_cost : ℕ := anton_shares * anton_price_per_share
noncomputable def yuri_total_cost : ℕ := yuri_shares * yuri_price_per_share
noncomputable def inga_total_cost : ℕ := inga_shares * inga_price_per_share
noncomputable def yulia_total_cost : ℕ := yulia_shares * yulia_price_per_share

theorem arina_should_accept_anton_offer :
  anton_total_cost = 210000 := by
  sorry

end arina_should_accept_anton_offer_l40_40733


namespace geom_sequence_ratio_and_fifth_term_l40_40522

theorem geom_sequence_ratio_and_fifth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 10) 
  (h₂ : a₂ = -15) 
  (h₃ : a₃ = 22.5) 
  (h₄ : a₄ = -33.75) : 
  ∃ r a₅, r = -1.5 ∧ a₅ = 50.625 ∧ (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ (a₄ = r * a₃) ∧ (a₅ = r * a₄) := 
by
  sorry

end geom_sequence_ratio_and_fifth_term_l40_40522


namespace board_coloring_l40_40068

open Finset

theorem board_coloring :
  ∃ (configs : Finset (Matrix (Fin 8) (Fin 8) Bool)), 
    (∀ m ∈ configs, (card (filter (λ c, c = tt) m.entries)) = 31 ∧ 
                   ∀ i j, (m i j = tt → ∀ (di dj : Fin 8), 
                           (abs (i - di) + abs (j - dj) = 1 → m di dj ≠ tt))) ∧ 
    card configs = 68 :=
sorry

end board_coloring_l40_40068


namespace sum_of_squares_xy_l40_40190

theorem sum_of_squares_xy (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end sum_of_squares_xy_l40_40190


namespace root_exists_l40_40417

variable {R : Type} [LinearOrderedField R]
variables (a b c : R)

def f (x : R) : R := a * x^2 + b * x + c

theorem root_exists (h : f a b c ((a - b - c) / (2 * a)) = 0) : f a b c (-1) = 0 ∨ f a b c 1 = 0 := by
  sorry

end root_exists_l40_40417


namespace books_sold_in_january_l40_40711

theorem books_sold_in_january (J : ℕ) 
  (h_avg : (J + 16 + 17) / 3 = 16) : J = 15 :=
sorry

end books_sold_in_january_l40_40711


namespace lcm_48_180_eq_720_l40_40554

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l40_40554


namespace product_of_triangle_areas_not_end_2014_l40_40316

theorem product_of_triangle_areas_not_end_2014
  (T1 T2 T3 T4 : ℤ)
  (h1 : T1 > 0)
  (h2 : T2 > 0)
  (h3 : T3 > 0)
  (h4 : T4 > 0) :
  (T1 * T2 * T3 * T4) % 10000 ≠ 2014 := by
sorry

end product_of_triangle_areas_not_end_2014_l40_40316


namespace sale_price_is_correct_l40_40326

def original_price : ℝ := 100
def percentage_decrease : ℝ := 0.30
def sale_price : ℝ := original_price * (1 - percentage_decrease)

theorem sale_price_is_correct : sale_price = 70 := by
  sorry

end sale_price_is_correct_l40_40326


namespace solve_equation_l40_40348

theorem solve_equation : 
  ∀ x : ℝ, (x^2 + 2*x + 3)/(x + 2) = x + 4 → x = -(5/4) := by
  sorry

end solve_equation_l40_40348


namespace xiao_ming_polygon_l40_40378

theorem xiao_ming_polygon (n : ℕ) (h : (n - 2) * 180 = 2185) : n = 14 :=
by sorry

end xiao_ming_polygon_l40_40378


namespace part1_part2_l40_40210

noncomputable def problem1 (x y: ℕ) : Prop := 
  (2 * x + 3 * y = 44) ∧ (4 * x = 5 * y)

noncomputable def solution1 (x y: ℕ) : Prop :=
  (x = 10) ∧ (y = 8)

theorem part1 : ∃ x y: ℕ, problem1 x y → solution1 x y :=
by sorry

noncomputable def problem2 (a b: ℕ) : Prop := 
  25 * (10 * a + 8 * b) = 3500

noncomputable def solution2 (a b: ℕ) : Prop :=
  ((a = 2 ∧ b = 15) ∨ (a = 6 ∧ b = 10) ∨ (a = 10 ∧ b = 5))

theorem part2 : ∃ a b: ℕ, problem2 a b → solution2 a b :=
by sorry

end part1_part2_l40_40210


namespace line_parallel_eq_l40_40653

theorem line_parallel_eq (x y : ℝ) (h1 : 3 * x - y = 6) (h2 : x = -2 ∧ y = 3) :
  ∃ m b, m = 3 ∧ b = 9 ∧ y = m * x + b :=
by
  sorry

end line_parallel_eq_l40_40653


namespace possible_values_of_d_l40_40951

theorem possible_values_of_d :
  ∃ (e f d : ℤ), (e + 12) * (f + 12) = 1 ∧
  ∀ x, (x - d) * (x - 12) + 1 = (x + e) * (x + f) ↔ (d = 22 ∨ d = 26) :=
by
  sorry

end possible_values_of_d_l40_40951


namespace problem_statement_l40_40762

def f (x : ℝ) : ℝ :=
  if x < 1 then (x + 1)^2 else 4 - real.sqrt (x - 1)

theorem problem_statement {x : ℝ} : (f x) ≥ x ↔ x ∈ set.Iic (-2) ∪ set.Icc 0 10 :=
by
  sorry

end problem_statement_l40_40762


namespace no_such_polynomials_exists_l40_40989

theorem no_such_polynomials_exists :
  ¬ ∃ (f g : Polynomial ℚ), (∀ x y : ℚ, f.eval x * g.eval y = x^200 * y^200 + 1) := 
by 
  sorry

end no_such_polynomials_exists_l40_40989


namespace bob_spends_more_time_l40_40460

def pages := 760
def time_per_page_bob := 45
def time_per_page_chandra := 30
def total_time_bob := pages * time_per_page_bob
def total_time_chandra := pages * time_per_page_chandra
def time_difference := total_time_bob - total_time_chandra

theorem bob_spends_more_time : time_difference = 11400 :=
by
  sorry

end bob_spends_more_time_l40_40460


namespace total_pencils_l40_40913

theorem total_pencils (pencils_per_person : ℕ) (num_people : ℕ) (total_pencils : ℕ) :
  pencils_per_person = 15 ∧ num_people = 5 → total_pencils = pencils_per_person * num_people :=
by
  intros h
  cases h with h1 h2
  rw [h1, h2]
  exact sorry
  
end total_pencils_l40_40913


namespace tan_alpha_fraction_l40_40103

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l40_40103


namespace max_value_of_expression_l40_40868

theorem max_value_of_expression (x : Real) :
  (x^4 / (x^8 + 2 * x^6 - 3 * x^4 + 5 * x^3 + 8 * x^2 + 5 * x + 25)) ≤ (1 / 15) :=
sorry

end max_value_of_expression_l40_40868


namespace prime_sum_probability_l40_40271

/-- Statement for the proof problem -/
theorem prime_sum_probability:
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} in 
  (∃ (x y ∈ primes), (x ≠ y) ∧ (x + y) ∈ primes) →
  (∃ (probability : ℚ), probability = 1/9) :=
by sorry

end prime_sum_probability_l40_40271


namespace prime_sum_probability_l40_40272

/-- Statement for the proof problem -/
theorem prime_sum_probability:
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} in 
  (∃ (x y ∈ primes), (x ≠ y) ∧ (x + y) ∈ primes) →
  (∃ (probability : ℚ), probability = 1/9) :=
by sorry

end prime_sum_probability_l40_40272


namespace car_speed_l40_40819

theorem car_speed (d t : ℝ) (h_d : d = 624) (h_t : t = 3) : d / t = 208 := by
  sorry

end car_speed_l40_40819


namespace kevin_wings_record_l40_40448

-- Conditions
def alanWingsPerMinute : ℕ := 5
def additionalWingsNeeded : ℕ := 4
def kevinRecordDuration : ℕ := 8

-- Question and answer
theorem kevin_wings_record : 
  (alanWingsPerMinute + additionalWingsNeeded) * kevinRecordDuration = 72 :=
by
  sorry

end kevin_wings_record_l40_40448


namespace prob_prime_sum_l40_40255

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := nat.prime n

def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (List.product l l).filter (λ p, p.1 < p.2)

def prime_sum_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  valid_pairs l |>.filter (λ p, is_prime (p.1 + p.2))

theorem prob_prime_sum :
  prime_sum_pairs primes.length / (valid_pairs primes).length = 1 / 9 :=
by
  sorry -- Proof is not required

end prob_prime_sum_l40_40255


namespace factorization_count_is_correct_l40_40598

noncomputable def count_factorizations (n : Nat) (k : Nat) : Nat :=
  (Nat.choose (n + k - 1) (k - 1))

noncomputable def factor_count : Nat :=
  let alpha_count := count_factorizations 6 3
  let beta_count := count_factorizations 6 3
  let total_count := alpha_count * beta_count
  let unordered_factorizations := total_count - 15 * 3 - 1
  1 + 15 + unordered_factorizations / 6

theorem factorization_count_is_correct :
  factor_count = 139 := by
  sorry

end factorization_count_is_correct_l40_40598


namespace total_population_l40_40594

theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) : 
  b + g + t = 26 * t :=
by
  -- We state our theorem including assumptions and goal
  sorry -- placeholder for the proof

end total_population_l40_40594


namespace work_completion_l40_40967

theorem work_completion (a b : Type) (work_done_together work_done_by_a work_done_by_b : ℝ) 
  (h1 : work_done_together = 1 / 12) 
  (h2 : work_done_by_a = 1 / 20) 
  (h3 : work_done_by_b = work_done_together - work_done_by_a) : 
  work_done_by_b = 1 / 30 :=
by
  sorry

end work_completion_l40_40967


namespace percentage_of_second_discount_is_correct_l40_40360

def car_original_price : ℝ := 12000
def first_discount : ℝ := 0.20
def final_price_after_discounts : ℝ := 7752
def third_discount : ℝ := 0.05

def solve_percentage_second_discount : Prop := 
  ∃ (second_discount : ℝ), 
    (car_original_price * (1 - first_discount) * (1 - second_discount) * (1 - third_discount) = final_price_after_discounts) ∧ 
    (second_discount * 100 = 15)

theorem percentage_of_second_discount_is_correct : solve_percentage_second_discount :=
  sorry

end percentage_of_second_discount_is_correct_l40_40360


namespace fraction_multiplier_l40_40585

theorem fraction_multiplier (x y : ℝ) :
  (3 * x * 3 * y) / (3 * x + 3 * y) = 3 * (x * y) / (x + y) :=
by
  sorry

end fraction_multiplier_l40_40585


namespace find_x_l40_40512

theorem find_x (x y : ℤ) (h1 : y = 3) (h2 : x + 3 * y = 10) : x = 1 :=
by
  sorry

end find_x_l40_40512


namespace union_set_solution_l40_40590

theorem union_set_solution (M N : Set ℝ) 
    (hM : M = { x | 0 ≤ x ∧ x ≤ 3 }) 
    (hN : N = { x | x < 1 }) : 
    M ∪ N = { x | x ≤ 3 } := 
by 
    sorry

end union_set_solution_l40_40590


namespace amount_needed_is_72_l40_40739

-- Define the given conditions
def original_price : ℝ := 90
def discount_rate : ℝ := 20

-- The goal is to prove that the amount of money needed after the discount is $72
theorem amount_needed_is_72 (P : ℝ) (D : ℝ) (hP : P = original_price) (hD : D = discount_rate) : P - (D / 100 * P) = 72 := 
by sorry

end amount_needed_is_72_l40_40739


namespace lcm_48_180_value_l40_40549

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l40_40549


namespace num_zero_points_f_l40_40854

namespace ZeroPoints

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * Real.cos x)

def interval := Set.Icc 0 (2 * Real.pi)

def zero_points (f : ℝ → ℝ) (I : Set ℝ) : Set ℝ := {x ∈ I | f x = 0}

theorem num_zero_points_f : Finset.card (Finset.filter (λ x, x ∈ zero_points f interval) 
  (Finset.image (λ n, Real.arccos n) (Finset.range 3))) = 5 :=
sorry

end ZeroPoints

end num_zero_points_f_l40_40854


namespace left_square_side_length_l40_40180

theorem left_square_side_length (x : ℕ) (h1 : x + (x + 17) + (x + 11) = 52) : x = 8 :=
sorry

end left_square_side_length_l40_40180


namespace stratified_sampling_seniors_l40_40840

theorem stratified_sampling_seniors
  (total_students : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)
  (senior_sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : seniors = 1500)
  (h3 : sample_size = 300)
  (h4 : senior_sample_size = seniors * sample_size / total_students) :
  senior_sample_size = 100 :=
  sorry

end stratified_sampling_seniors_l40_40840


namespace hyperbola_center_l40_40719

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (f1 : x1 = 3) (f2 : y1 = -2) (f3 : x2 = 11) (f4 : y2 = 6) :
    (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 2 :=
by
  sorry

end hyperbola_center_l40_40719


namespace problem_statement_l40_40568

theorem problem_statement (x m : ℝ) :
  (¬ (x > m) → ¬ (x^2 + x - 2 > 0)) ∧ (¬ (x > m) ↔ ¬ (x^2 + x - 2 > 0)) → m ≥ 1 :=
sorry

end problem_statement_l40_40568


namespace pow_mod_remainder_l40_40193

theorem pow_mod_remainder (n : ℕ) (h : 9 ≡ 2 [MOD 7]) (h2 : 9^2 ≡ 4 [MOD 7]) (h3 : 9^3 ≡ 1 [MOD 7]) : 9^123 % 7 = 1 := by
  sorry

end pow_mod_remainder_l40_40193


namespace farmer_plant_beds_l40_40049

theorem farmer_plant_beds :
  ∀ (bean_seedlings pumpkin_seeds radishes seedlings_per_row_pumpkin seedlings_per_row_radish radish_rows_per_bed : ℕ),
    bean_seedlings = 64 →
    seedlings_per_row_pumpkin = 7 →
    pumpkin_seeds = 84 →
    seedlings_per_row_radish = 6 →
    radish_rows_per_bed = 2 →
    (bean_seedlings / 8 + pumpkin_seeds / seedlings_per_row_pumpkin + radishes / seedlings_per_row_radish) / radish_rows_per_bed = 14 :=
by
  -- sorry to skip the proof
  sorry

end farmer_plant_beds_l40_40049


namespace probability_prime_sum_of_two_draws_l40_40269

open Finset

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrime (n : ℕ) : Prop := Nat.Prime n

def validPairs (primes : List ℕ) : List (ℕ × ℕ) :=
  primes.bind (λ p1 => primes.filter (λ p2 => p1 ≠ p2 ∧ isPrime (p1 + p2)).map (λ p2 => (p1, p2)))

theorem probability_prime_sum_of_two_draws :
  (firstTenPrimes.length.choose 2 : ℚ) > 0 →
  let validPairs := validPairs firstTenPrimes
  let numValidPairs := validPairs.toFinset.card
  let totalPairs := 45
  (∀ (x ∈ validPairs), ∃ (a b : ℕ), a ∈ firstTenPrimes ∧ b ∈ firstTenPrimes ∧ a ≠ b ∧ isPrime (a + b) ∧ (a, b) = x) →
  (∀ p, p ∈ firstTenPrimes → isPrime p) →
  numValidPairs / totalPairs = (1 : ℚ) / 9 := by
  sorry

end probability_prime_sum_of_two_draws_l40_40269


namespace sqrt_of_expression_l40_40541

theorem sqrt_of_expression :
  Real.sqrt (4^4 * 9^2) = 144 :=
sorry

end sqrt_of_expression_l40_40541


namespace sum_of_faces_edges_vertices_rectangular_prism_l40_40685

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l40_40685


namespace tan_alpha_value_l40_40097

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l40_40097


namespace instantaneous_rate_of_change_of_area_with_respect_to_perimeter_at_3cm_l40_40587

open Real

theorem instantaneous_rate_of_change_of_area_with_respect_to_perimeter_at_3cm :
  let s := 3
  let P := 4 * s
  let A := s^2
  let dAdP := deriv (fun P => (1 / 16) * P^2) P
  P = 12 →
  dAdP = 3 / 2 :=
by
  intros
  rw [←h]
  have s_eq := show 3 = s by rfl
  have P_eq := show 12 = 4 * s by rw [s_eq]; norm_num
  rw [P_eq] at this
  exact this

end instantaneous_rate_of_change_of_area_with_respect_to_perimeter_at_3cm_l40_40587


namespace weighted_avg_surfers_per_day_l40_40744

theorem weighted_avg_surfers_per_day 
  (total_surfers : ℕ) 
  (ratio1_day1 ratio1_day2 ratio2_day3 ratio2_day4 : ℕ) 
  (h_total_surfers : total_surfers = 12000)
  (h_ratio_first_two_days : ratio1_day1 = 5 ∧ ratio1_day2 = 7)
  (h_ratio_last_two_days : ratio2_day3 = 3 ∧ ratio2_day4 = 2) 
  : (total_surfers / (ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4)) * 
    ((ratio1_day1 + ratio1_day2 + ratio2_day3 + ratio2_day4) / 4) = 3000 :=
by
  sorry

end weighted_avg_surfers_per_day_l40_40744


namespace pages_wed_calculation_l40_40070

def pages_mon : ℕ := 23
def pages_tue : ℕ := 38
def pages_thu : ℕ := 12
def pages_fri : ℕ := 2 * pages_thu
def total_pages : ℕ := 158

theorem pages_wed_calculation (pages_wed : ℕ) : 
  pages_mon + pages_tue + pages_wed + pages_thu + pages_fri = total_pages → pages_wed = 61 :=
by
  intros h
  sorry

end pages_wed_calculation_l40_40070


namespace min_a2_plus_b2_l40_40876

theorem min_a2_plus_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
sorry

end min_a2_plus_b2_l40_40876


namespace math_problem_solution_l40_40309

theorem math_problem_solution (x y : ℝ) : 
  abs x + x + 5 * y = 2 ∧ abs y - y + x = 7 → x + y + 2009 = 2012 :=
by {
  sorry
}

end math_problem_solution_l40_40309


namespace james_spends_90_dollars_per_week_l40_40442

structure PistachioPurchasing where
  can_cost : ℕ  -- cost in dollars per can
  can_weight : ℕ -- weight in ounces per can
  consumption_oz_per_5days : ℕ -- consumption in ounces per 5 days

def cost_per_week (p : PistachioPurchasing) : ℕ :=
  let daily_consumption := p.consumption_oz_per_5days / 5
  let weekly_consumption := daily_consumption * 7
  let cans_needed := (weekly_consumption + p.can_weight - 1) / p.can_weight -- round up
  cans_needed * p.can_cost

theorem james_spends_90_dollars_per_week :
  cost_per_week ⟨10, 5, 30⟩ = 90 :=
by
  sorry

end james_spends_90_dollars_per_week_l40_40442


namespace unique_ones_digits_divisible_by_8_l40_40616

/-- Carla likes numbers that are divisible by 8.
    We want to show that there are 5 unique ones digits for such numbers. -/
theorem unique_ones_digits_divisible_by_8 : 
  (Finset.card 
    (Finset.image (fun n => n % 10) 
                  (Finset.filter (fun n => n % 8 = 0) (Finset.range 100)))) = 5 := 
by
  sorry

end unique_ones_digits_divisible_by_8_l40_40616


namespace a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l40_40971

-- For Question 1
theorem a_squared_plus_b_squared_gt_one_over_four (a b : ℝ) (h : a + b = 1) : a^2 + b^2 > 1/4 :=
sorry

-- For Question 2
theorem sequence_is_arithmetic (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, S n = 2 * (n:ℝ)^2 - 3 * (n:ℝ) - 2) :
  ∃ d, ∀ n, (S n / (2 * (n:ℝ) + 1)) = (S (n + 1) / (2 * (n + 1:ℝ) + 1)) + d :=
sorry

end a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l40_40971


namespace solve_for_y_l40_40470

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l40_40470


namespace prime_sum_probability_l40_40273

/-- Statement for the proof problem -/
theorem prime_sum_probability:
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} in 
  (∃ (x y ∈ primes), (x ≠ y) ∧ (x + y) ∈ primes) →
  (∃ (probability : ℚ), probability = 1/9) :=
by sorry

end prime_sum_probability_l40_40273


namespace nancy_crystal_beads_l40_40830

-- Definitions of given conditions
def price_crystal : ℕ := 9
def price_metal : ℕ := 10
def sets_metal : ℕ := 2
def total_spent : ℕ := 29

-- Statement of the proof problem
theorem nancy_crystal_beads : ∃ x : ℕ, price_crystal * x + price_metal * sets_metal = total_spent ∧ x = 1 := by
  sorry

end nancy_crystal_beads_l40_40830


namespace sally_earnings_proof_l40_40340

def sally_last_month_earnings : ℝ := 1000
def raise_percentage : ℝ := 0.10
def sally_this_month_earnings := sally_last_month_earnings * (1 + raise_percentage)
def sally_total_two_months_earnings := sally_last_month_earnings + sally_this_month_earnings

theorem sally_earnings_proof :
  sally_total_two_months_earnings = 2100 :=
by
  sorry

end sally_earnings_proof_l40_40340


namespace complex_product_l40_40855

theorem complex_product (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  (6 - 7 * i) * (3 + 6 * i) = 60 + 15 * i :=
  by
    -- proof statements would go here
    sorry

end complex_product_l40_40855


namespace small_planters_needed_l40_40170

-- This states the conditions for the problem
def Oshea_seeds := 200
def large_planters := 4
def large_planter_capacity := 20
def small_planter_capacity := 4
def remaining_seeds := Oshea_seeds - (large_planters * large_planter_capacity) 

-- The target we aim to prove: the number of small planters required
theorem small_planters_needed :
  remaining_seeds / small_planter_capacity = 30 := by
  sorry

end small_planters_needed_l40_40170


namespace value_to_be_subtracted_l40_40588

theorem value_to_be_subtracted (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 24) / 10 = 3) : x = 5 := by
  sorry

end value_to_be_subtracted_l40_40588


namespace lengths_of_legs_l40_40597

def is_right_triangle (a b c : ℕ) := a^2 + b^2 = c^2

theorem lengths_of_legs (a b : ℕ) 
  (h1 : is_right_triangle a b 60)
  (h2 : a + b = 84) 
  : (a = 48 ∧ b = 36) ∨ (a = 36 ∧ b = 48) :=
  sorry

end lengths_of_legs_l40_40597


namespace num_solutions_eq_4_l40_40403

theorem num_solutions_eq_4 (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  ∃ n : ℕ, n = 4 ∧ (2 + 4 * Real.cos θ - 6 * Real.sin (2 * θ) + 3 * Real.tan θ = 0) :=
sorry

end num_solutions_eq_4_l40_40403


namespace fruit_prices_l40_40538

theorem fruit_prices (x y : ℝ) 
  (h₁ : 3 * x + 2 * y = 40) 
  (h₂ : 2 * x + 3 * y = 35) : 
  x = 10 ∧ y = 5 :=
by
  sorry

end fruit_prices_l40_40538


namespace tan_alpha_fraction_l40_40102

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l40_40102


namespace main_problem_l40_40603

def arithmetic_sequence (a : ℕ → ℕ) : Prop := ∃ a₁ d, ∀ n, a (n + 1) = a₁ + n * d

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2

def another_sequence (b : ℕ → ℕ) (a : ℕ → ℕ) : Prop := ∀ n, b n = 1 / (a n * a (n + 1))

theorem main_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) 
  (h1 : a_3 = 5) 
  (h2 : S_3 = 9) 
  (h3 : arithmetic_sequence a)
  (h4 : sequence_sum a S)
  (h5 : another_sequence b a) : 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, T n = n / (2 * n + 1)) := sorry

end main_problem_l40_40603


namespace tan_alpha_solution_l40_40121

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l40_40121


namespace sum_projections_l40_40778

universe u

variables {α : Type u} [Nonempty α] [MetricSpace α]

structure Triangle (α : Type u) [MetricSpace α] :=
(A B C : α)
(AB AC BC : ℝ)
(AB_pos : 0 < AB)
(AC_pos : 0 < AC)
(BC_pos : 0 < BC)
(equal_ab : dist A B = AB)
(equal_ac : dist A C = AC)
(equal_bc : dist B C = BC)

def centroid (A B C : α) : α := sorry -- centroid calculation goes here

def projection (P Q R : α) (G : α) : ℝ := sorry -- projection calculation goes here

theorem sum_projections {A B C: α} (t : Triangle α)
  (G : α) (P : α := projection t.B t.C G)
  (Q : α := projection t.A t.C G)
  (R : α := projection t.A t.B G) :
  t.AB = 4 → t.AC = 6 → t.BC = 5 → (projection t.B t.C G) + (projection t.A t.C G) + (projection t.A t.B G) = (5 * real.sqrt 7) / 7 :=
by
  sorry

end sum_projections_l40_40778


namespace pie_remaining_portion_l40_40540

theorem pie_remaining_portion (Carlos_share Maria_share remaining: ℝ)
  (hCarlos : Carlos_share = 0.65)
  (hRemainingAfterCarlos : remaining = 1 - Carlos_share)
  (hMaria : Maria_share = remaining / 2) :
  remaining - Maria_share = 0.175 :=
by
  sorry

end pie_remaining_portion_l40_40540


namespace number_of_therapy_hours_l40_40974

theorem number_of_therapy_hours (A F H : ℝ) (h1 : F = A + 35) 
  (h2 : F + (H - 1) * A = 350) (h3 : F + A = 161) :
  H = 5 :=
by
  sorry

end number_of_therapy_hours_l40_40974


namespace sqrt_floor_19992000_l40_40406

theorem sqrt_floor_19992000 : (Int.floor (Real.sqrt 19992000)) = 4471 := by
  sorry

end sqrt_floor_19992000_l40_40406


namespace hyperbola_center_l40_40718

variable {Point : Type}

structure coordinates (P : Point) :=
(x : ℝ)
(y : ℝ)

def center_of_hyperbola (P₁ P₂ : Point) := 
  coordinates.mk ((coordinates.x P₁ + coordinates.x P₂) / 2) ((coordinates.y P₁ + coordinates.y P₂) / 2)

theorem hyperbola_center (f1 f2 : Point) (h1 : coordinates f1) (h2 : coordinates f2) :
  h1 = coordinates.mk 3 (-2) → h2 = coordinates.mk 11 6 → center_of_hyperbola f1 f2 = coordinates.mk 7 2 :=
by
  intros
  sorry

end hyperbola_center_l40_40718


namespace simplify_fraction_l40_40347

theorem simplify_fraction : (45 / (7 - 3 / 4)) = (36 / 5) :=
by
  sorry

end simplify_fraction_l40_40347


namespace fewer_hours_l40_40368

noncomputable def distance : ℝ := 300
noncomputable def speed_T : ℝ := 20
noncomputable def speed_A : ℝ := speed_T + 5

theorem fewer_hours (d : ℝ) (V_T : ℝ) (V_A : ℝ) :
    V_T = 20 ∧ V_A = V_T + 5 ∧ d = 300 → (d / V_T) - (d / V_A) = 3 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end fewer_hours_l40_40368


namespace probability_blue_before_green_l40_40045

open Finset Nat

noncomputable def num_arrangements : ℕ := choose 9 2  -- total number of ways to arrange the chips

noncomputable def num_favorable_arrangements : ℕ := 2 * (choose 7 4)  -- favorable arrangements where blue chips are among the first 8

noncomputable def probability_all_blue_before_green : ℚ := num_favorable_arrangements / num_arrangements

theorem probability_blue_before_green : probability_all_blue_before_green = 17 / 36 :=
sorry

end probability_blue_before_green_l40_40045


namespace sum_of_triangles_l40_40178

def triangle (a b c : ℤ) : ℤ := a + b - c

theorem sum_of_triangles : triangle 1 3 4 + triangle 2 5 6 = 1 := by
  sorry

end sum_of_triangles_l40_40178


namespace differentiable_implies_continuous_l40_40464

-- Theorem: If a function f is differentiable at x0, then it is continuous at x0.
theorem differentiable_implies_continuous {f : ℝ → ℝ} {x₀ : ℝ} (h : DifferentiableAt ℝ f x₀) : 
  ContinuousAt f x₀ :=
sorry

end differentiable_implies_continuous_l40_40464


namespace inverse_proposition_l40_40480

theorem inverse_proposition (x : ℝ) : 
  (¬ (x > 2) → ¬ (x > 1)) ↔ ((x > 1) → (x > 2)) := 
by 
  sorry

end inverse_proposition_l40_40480


namespace problem_part1_problem_part2_l40_40159

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem problem_part1 (h : ∀ x : ℝ, f (-x) = -f x) : a = 1 :=
sorry

theorem problem_part2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

end problem_part1_problem_part2_l40_40159


namespace fare_collected_from_I_class_l40_40991

theorem fare_collected_from_I_class (x y : ℕ) 
  (h_ratio_passengers : 4 * x = 4 * x) -- ratio of passengers 1:4
  (h_ratio_fare : 3 * y = 3 * y) -- ratio of fares 3:1
  (h_total_fare : 7 * 3 * x * y = 224000) -- total fare Rs. 224000
  : 3 * x * y = 96000 := 
by
  sorry

end fare_collected_from_I_class_l40_40991


namespace least_positive_integer_with_12_factors_l40_40028

def has_exactly_12_factors (n : ℕ) : Prop :=
  (finset.range(n + 1).filter (λ d, n % d = 0)).card = 12

theorem least_positive_integer_with_12_factors : ∀ n : ℕ, has_exactly_12_factors n → n ≥ 150 :=
sorry

end least_positive_integer_with_12_factors_l40_40028


namespace simplify_tan_cot_l40_40932

theorem simplify_tan_cot :
  ∀ (tan cot : ℝ), tan 45 = 1 ∧ cot 45 = 1 →
  (tan 45)^3 + (cot 45)^3 / (tan 45 + cot 45) = 1 :=
by
  intros tan cot h
  have h_tan : tan 45 = 1 := h.1
  have h_cot : cot 45 = 1 := h.2
  sorry

end simplify_tan_cot_l40_40932


namespace roots_polynomial_identity_l40_40172

theorem roots_polynomial_identity (a b x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + b*x₁ + b^2 + a = 0) 
  (h₂ : x₂^2 + b*x₂ + b^2 + a = 0) : x₁^2 + x₁*x₂ + x₂^2 + a = 0 :=
by 
  sorry

end roots_polynomial_identity_l40_40172


namespace find_square_tiles_l40_40203

variables (t s p : ℕ)

theorem find_square_tiles
  (h1 : t + s + p = 30)
  (h2 : 3 * t + 4 * s + 5 * p = 120) :
  s = 10 :=
by
  sorry

end find_square_tiles_l40_40203


namespace min_value_of_expression_l40_40611

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x = a^2 + b^2 + (1 / (a + b)^2) + (1 / (a * b)) ∧ x = Real.sqrt 10 :=
sorry

end min_value_of_expression_l40_40611


namespace lcm_48_180_eq_720_l40_40555

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l40_40555


namespace coin_toss_sequences_l40_40774

theorem coin_toss_sequences :
  ∃ (seqs : list (char × char)), list.length seqs = 25 ∧
  (∃ (HH HT TH TT : ℕ), HH = 3 ∧ HT = 5 ∧ TH = 4 ∧ TT = 7 ∧ 
  (seqs.count (λ x, x = ('H','H')) = HH ∧ 
   seqs.count (λ x, x = ('H','T')) = HT ∧ 
   seqs.count (λ x, x = ('T','H')) = TH ∧ 
   seqs.count (λ x, x = ('T','T')) = TT)) ∧ 
  list.length (list.filter (λ x, x = ('H','T') ∨ x = ('T','H')) seqs) + 1 - 1 = 
  10 ∧ (nat.choose (5 + 3 - 1) 2) * (nat.choose (10 + 4 - 1) 3) = 1200 :=
sorry

end coin_toss_sequences_l40_40774


namespace simplify_fraction_tan_cot_45_l40_40935

theorem simplify_fraction_tan_cot_45 :
  (tan 45 * tan 45 * tan 45 + cot 45 * cot 45 * cot 45) / (tan 45 + cot 45) = 1 :=
by
  -- Conditions: tan 45 = 1, cot 45 = 1
  have h_tan_45 : tan 45 = 1 := sorry
  have h_cot_45 : cot 45 = 1 := sorry
  -- Proof: Using the conditions and simplification
  sorry

end simplify_fraction_tan_cot_45_l40_40935


namespace sum_of_legs_of_larger_triangle_l40_40955

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def similar_triangles {a1 b1 c1 a2 b2 c2 : ℝ} (h1 : right_triangle a1 b1 c1) (h2 : right_triangle a2 b2 c2) :=
  ∃ k : ℝ, k > 0 ∧ (a2 = k * a1 ∧ b2 = k * b1)

theorem sum_of_legs_of_larger_triangle 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : right_triangle a1 b1 c1)
  (h2 : right_triangle a2 b2 c2)
  (h_sim : similar_triangles h1 h2)
  (area1 : ℝ) (area2 : ℝ)
  (hyp1 : c1 = 6) 
  (area_cond1 : (a1 * b1) / 2 = 8)
  (area_cond2 : (a2 * b2) / 2 = 200) :
  a2 + b2 = 40 := by
  sorry

end sum_of_legs_of_larger_triangle_l40_40955


namespace find_c_l40_40849

theorem find_c (c d : ℝ) (h1 : c < 0) (h2 : d > 0)
    (max_min_condition : ∀ x, c * Real.cos (d * x) ≤ 3 ∧ c * Real.cos (d * x) ≥ -3) :
    c = -3 :=
by
  -- The statement says if c < 0, d > 0, and given the cosine function hitting max 3 and min -3, then c = -3.
  sorry

end find_c_l40_40849


namespace remainder_of_504_divided_by_100_is_4_l40_40743

theorem remainder_of_504_divided_by_100_is_4 :
  (504 % 100) = 4 :=
by
  sorry

end remainder_of_504_divided_by_100_is_4_l40_40743


namespace valid_pairs_l40_40547

-- Define the target function and condition
def satisfies_condition (k l : ℤ) : Prop :=
  (7 * k - 5) * (4 * l - 3) = (5 * k - 3) * (6 * l - 1)

-- The theorem stating the exact pairs that satisfy the condition
theorem valid_pairs :
  ∀ (k l : ℤ), satisfies_condition k l ↔
    (k = 0 ∧ l = 6) ∨
    (k = 1 ∧ l = -1) ∨
    (k = 6 ∧ l = -6) ∨
    (k = 13 ∧ l = -7) ∨
    (k = -2 ∧ l = -22) ∨
    (k = -3 ∧ l = -15) ∨
    (k = -8 ∧ l = -10) ∨
    (k = -15 ∧ l = -9) :=
by
  sorry

end valid_pairs_l40_40547


namespace find_n_l40_40288

theorem find_n (n : ℕ) (h : Nat.lcm n (n - 30) = n + 1320) : n = 165 := 
sorry

end find_n_l40_40288


namespace find_y_l40_40158

def star (a b : ℝ) : ℝ := 2 * a * b - 3 * b - a

theorem find_y (y : ℝ) (h : star 4 y = 80) : y = 16.8 :=
by
  sorry

end find_y_l40_40158


namespace probability_prime_sum_is_1_9_l40_40228

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end probability_prime_sum_is_1_9_l40_40228


namespace probability_of_king_then_ace_is_4_over_663_l40_40805

noncomputable def probability_king_and_ace {α : Type*} [ProbabilityTheory α] 
  (deck : Finset α) (is_king : α → Prop) (is_ace : α → Prop) 
  (h1 : deck.card = 52) (h2 : (deck.filter is_king).card = 4) 
  (h3 : (deck.filter is_ace).card = 4) : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_king_then_ace_is_4_over_663 
  {α : Type*} [ProbabilityTheory α] 
  (deck : Finset α) (is_king : α → Prop) (is_ace : α → Prop) 
  (h1 : deck.card = 52) (h2 : (deck.filter is_king).card = 4) 
  (h3 : (deck.filter is_ace).card = 4) :
  probability_king_and_ace deck is_king is_ace h1 h2 h3 = 4 / 663 := 
sorry

end probability_of_king_then_ace_is_4_over_663_l40_40805


namespace least_positive_integer_with_12_factors_is_72_l40_40007

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l40_40007


namespace tan_A_right_triangle_l40_40599

theorem tan_A_right_triangle (A B C : Type) [RealField A] [RealField B] [RealField C]
  (h1 : ∠ABC = 90) (h2 : sin B = 3 / 5) : tan A = 4 / 3 :=
sorry

end tan_A_right_triangle_l40_40599


namespace area_of_triangle_POF_l40_40757

noncomputable def origin : (ℝ × ℝ) := (0, 0)
noncomputable def focus : (ℝ × ℝ) := (Real.sqrt 2, 0)

noncomputable def parabola (x y : ℝ) : Prop :=
  y ^ 2 = 4 * Real.sqrt 2 * x

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  parabola x y

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

noncomputable def PF_eq_4sqrt2 (x y : ℝ) : Prop :=
  distance x y (Real.sqrt 2) 0 = 4 * Real.sqrt 2

theorem area_of_triangle_POF (x y : ℝ) 
  (h1: point_on_parabola x y)
  (h2: PF_eq_4sqrt2 x y) :
   1 / 2 * distance 0 0 (Real.sqrt 2) 0 * |y| = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_POF_l40_40757


namespace find_A_l40_40484

theorem find_A (A B : ℕ) (hcfAB lcmAB : ℕ)
  (hcf_cond : Nat.gcd A B = hcfAB)
  (lcm_cond : Nat.lcm A B = lcmAB)
  (B_val : B = 169)
  (hcf_val : hcfAB = 13)
  (lcm_val : lcmAB = 312) :
  A = 24 :=
by 
  sorry

end find_A_l40_40484


namespace range_of_a_l40_40461

theorem range_of_a (a : ℝ) : (a < 0 → (∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) ∧ 
                              (∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0) ↔ (x < -4 ∨ x ≥ -2)) ∧ 
                              ((¬(∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) 
                                → (¬(∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0))))
                            → (a ≤ -4 ∨ (a < 0 ∧ 3 * a >= -2)) :=
by
  intros h
  sorry

end range_of_a_l40_40461


namespace factorize_polynomial_l40_40280

theorem factorize_polynomial (a x : ℝ) : 
  (x^3 - 3*x^2 + (a + 2)*x - 2*a) = (x^2 - x + a)*(x - 2) :=
by
  sorry

end factorize_polynomial_l40_40280


namespace original_price_of_sarees_l40_40361

theorem original_price_of_sarees (P : ℝ) (h : 0.75 * 0.85 * P = 306) : P = 480 :=
by
  sorry

end original_price_of_sarees_l40_40361


namespace sum_faces_edges_vertices_eq_26_l40_40666

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l40_40666


namespace find_range_l40_40142

noncomputable def capricious_function_step_lower (k : ℝ) (x : ℝ) : Prop :=
  k ≥ x - 2

noncomputable def capricious_function_step_upper (k : ℝ) (x : ℝ) : Prop :=
  k ≤ (x + 1) * (Real.log x + 1) / x

noncomputable def capricious_function (k : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 Real.exp 1, capricious_function_step_lower k x ∧ capricious_function_step_upper k x

theorem find_range (k : ℝ) : capricious_function k ↔ k ∈ Set.Icc (Real.exp 1 - 2) 2 :=
by
  sorry

end find_range_l40_40142


namespace production_cost_percentage_l40_40946

theorem production_cost_percentage
    (initial_cost final_cost : ℝ)
    (final_cost_eq : final_cost = 48)
    (initial_cost_eq : initial_cost = 50)
    (h : (initial_cost + 0.5 * x) * (1 - x / 100) = final_cost) :
    x = 20 :=
by
  sorry

end production_cost_percentage_l40_40946


namespace f_F_same_monotonicity_min_phi_value_l40_40882

-- (1) Prove that the range of a such that f(x) and F(x) have the same monotonicity on the interval (0, ln 3)
theorem f_F_same_monotonicity (f F : ℝ → ℝ) (a : ℝ) (h_f_eq : ∀ x, f x = a * x - Real.log x)
(h_F_eq : ∀ x, F x = Real.exp x + a * x) (h_a : a < 0) :
  (∀ x : ℝ, 0 < x ∧ x < Real.log 3 → (f' x) * (F' x) > 0) ↔ a ∈ Set.Iic (-3) :=
sorry

-- (2) Prove that the minimum value of φ(a) for g(x) is 0
theorem min_phi_value (g φ : ℝ → ℝ) (a : ℝ) (h_g_eq : ∀ x, g x = x * Real.exp (a * x - 1) - 2 * a * x + a * x - Real.log x)
(h_φ_eq : φ a = min_g_val g) (h_a_cond : a ∈ Set.Iic (-1 / Real.exp 2)) :
  φ a = 0 :=
sorry

end f_F_same_monotonicity_min_phi_value_l40_40882


namespace composite_number_property_l40_40076

theorem composite_number_property (n : ℕ) 
  (h1 : n > 1) 
  (h2 : ¬ Prime n) 
  (h3 : ∀ (d : ℕ), d ∣ n → 1 ≤ d → d < n → n - 20 ≤ d ∧ d ≤ n - 12) :
  n = 21 ∨ n = 25 :=
by
  sorry

end composite_number_property_l40_40076


namespace find_x_l40_40867

theorem find_x :
  (12^3 * 6^3) / x = 864 → x = 432 :=
by
  sorry

end find_x_l40_40867


namespace smallest_possible_e_l40_40852

-- Define the polynomial with its roots and integer coefficients
def polynomial (x : ℝ) : ℝ := (x + 4) * (x - 6) * (x - 10) * (2 * x + 1)

-- Define e as the constant term
def e : ℝ := 200 -- based on the final expanded polynomial result

-- The theorem stating the smallest possible value of e
theorem smallest_possible_e : 
  ∃ (e : ℕ), e > 0 ∧ polynomial e = 200 := 
sorry

end smallest_possible_e_l40_40852


namespace greatest_odd_factors_l40_40453

theorem greatest_odd_factors (n : ℕ) (h1 : n < 1000) (h2 : ∀ k : ℕ, k * k = n → (k < 32)) :
  n = 31 * 31 :=
by
  sorry

end greatest_odd_factors_l40_40453


namespace total_stars_l40_40647

/-- Let n be the number of students, and s be the number of stars each student makes.
    We need to prove that the total number of stars is n * s. --/
theorem total_stars (n : ℕ) (s : ℕ) (h_n : n = 186) (h_s : s = 5) : n * s = 930 :=
by {
  sorry
}

end total_stars_l40_40647


namespace sum_faces_edges_vertices_l40_40673

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l40_40673


namespace prime_sum_probability_l40_40270

/-- Statement for the proof problem -/
theorem prime_sum_probability:
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} in 
  (∃ (x y ∈ primes), (x ≠ y) ∧ (x + y) ∈ primes) →
  (∃ (probability : ℚ), probability = 1/9) :=
by sorry

end prime_sum_probability_l40_40270


namespace caleb_hamburgers_total_l40_40222

def total_spent : ℝ := 66.50
def cost_single : ℝ := 1.00
def cost_double : ℝ := 1.50
def num_double : ℕ := 33

theorem caleb_hamburgers_total : 
  ∃ n : ℕ,  n = 17 + num_double ∧ 
            (num_double * cost_double) + (n - num_double) * cost_single = total_spent := by
sorry

end caleb_hamburgers_total_l40_40222


namespace problem_statement_l40_40087

theorem problem_statement (a b c m : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0) (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 :=
sorry

end problem_statement_l40_40087


namespace num_girls_on_playground_l40_40489

-- Definitions based on conditions
def total_students : ℕ := 20
def classroom_students := total_students / 4
def playground_students := total_students - classroom_students
def boys_playground := playground_students / 3
def girls_playground := playground_students - boys_playground

-- Theorem statement
theorem num_girls_on_playground : girls_playground = 10 :=
by
  -- Begin preparing proofs
  sorry

end num_girls_on_playground_l40_40489


namespace total_mass_of_individuals_l40_40197

def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def initial_sink_depth : Float := 0.018
def density_of_water : Float := 1000.0
def mass_of_second_person : Float := 75.0

theorem total_mass_of_individuals :
  let V1 := boat_length * boat_breadth * initial_sink_depth
  let m1 := V1 * density_of_water
  let total_mass := m1 + mass_of_second_person
  total_mass = 183 :=
by
  sorry

end total_mass_of_individuals_l40_40197


namespace quadratic_root_proof_l40_40433

noncomputable def root_condition (p q m n : ℝ) :=
  ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x ≠ 0 ∧ (1/x)^2 + m * (1/x) + n = 0

theorem quadratic_root_proof (p q m n : ℝ) (h : root_condition p q m n) :
  (pn - m) * (qm - p) = (qn - 1)^2 :=
sorry

end quadratic_root_proof_l40_40433


namespace train_crossing_time_l40_40519

theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_cross_platform : ℝ)
  (train_speed : ℝ := (train_length + platform_length) / time_to_cross_platform)
  (time_to_cross_signal_pole : ℝ := train_length / train_speed) :
  train_length = 300 ∧ platform_length = 1000 ∧ time_to_cross_platform = 39 → time_to_cross_signal_pole = 9 := by
  intro h
  cases h
  sorry

end train_crossing_time_l40_40519


namespace ax5_by5_eq_6200_div_29_l40_40583

variables (a b x y : ℝ)

-- Given conditions
axiom h1 : a * x + b * y = 5
axiom h2 : a * x^2 + b * y^2 = 11
axiom h3 : a * x^3 + b * y^3 = 30
axiom h4 : a * x^4 + b * y^4 = 80

-- Statement to prove
theorem ax5_by5_eq_6200_div_29 : a * x^5 + b * y^5 = 6200 / 29 :=
by
  sorry

end ax5_by5_eq_6200_div_29_l40_40583


namespace completing_square_l40_40813

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l40_40813


namespace mrs_hilt_bees_l40_40164

theorem mrs_hilt_bees (n : ℕ) (h : 3 * n = 432) : n = 144 := by
  sorry

end mrs_hilt_bees_l40_40164


namespace determine_a_l40_40543

theorem determine_a (a b c : ℤ) (h : (b + 11) * (c + 11) = 2) (hb : b + 11 = -2) (hc : c + 11 = -1) :
  a = 13 := by
  sorry

end determine_a_l40_40543


namespace find_constant_k_eq_l40_40548

theorem find_constant_k_eq : ∃ k : ℤ, (-x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4)) ↔ (k = -17) :=
by
  sorry

end find_constant_k_eq_l40_40548


namespace maximum_positive_numbers_l40_40785

theorem maximum_positive_numbers (a : ℕ → ℝ) (n : ℕ) (h₀ : n = 100)
  (h₁ : ∀ i : ℕ, 0 < a i) 
  (h₂ : ∀ i : ℕ, a i > a ((i + 1) % n) * a ((i + 2) % n)) : 
  ∃ m : ℕ, m ≤ 50 ∧ (∀ k : ℕ, k < m → (a k) > 0) :=
by sorry

end maximum_positive_numbers_l40_40785


namespace smallest_positive_value_l40_40750

theorem smallest_positive_value (a b c d e : ℝ) (h1 : a = 8 - 2 * Real.sqrt 14) 
  (h2 : b = 2 * Real.sqrt 14 - 8) 
  (h3 : c = 20 - 6 * Real.sqrt 10) 
  (h4 : d = 64 - 16 * Real.sqrt 4) 
  (h5 : e = 16 * Real.sqrt 4 - 64) :
  a = 8 - 2 * Real.sqrt 14 ∧ 0 < a ∧ a < c ∧ a < d :=
by
  sorry

end smallest_positive_value_l40_40750


namespace lcm_48_180_l40_40562

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l40_40562


namespace sum_of_faces_edges_vertices_of_rect_prism_l40_40691

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l40_40691


namespace earnings_in_six_minutes_l40_40458

def ticket_cost : ℕ := 3
def tickets_per_minute : ℕ := 5
def minutes : ℕ := 6

theorem earnings_in_six_minutes (ticket_cost : ℕ) (tickets_per_minute : ℕ) (minutes : ℕ) :
  ticket_cost = 3 → tickets_per_minute = 5 → minutes = 6 → (tickets_per_minute * ticket_cost * minutes = 90) :=
by
  intros h_cost h_tickets h_minutes
  rw [h_cost, h_tickets, h_minutes]
  norm_num

end earnings_in_six_minutes_l40_40458


namespace train_speed_in_kmh_l40_40972

theorem train_speed_in_kmh 
  (train_length : ℕ) 
  (crossing_time : ℕ) 
  (conversion_factor : ℕ) 
  (hl : train_length = 120) 
  (ht : crossing_time = 6) 
  (hc : conversion_factor = 36) :
  train_length / crossing_time * conversion_factor / 10 = 72 := by
  sorry

end train_speed_in_kmh_l40_40972


namespace posts_needed_l40_40389

-- Define the main properties
def length_of_side_W_stone_wall := 80
def short_side := 50
def intervals (metres: ℕ) := metres / 10 + 1 

-- Define the conditions
def posts_along_w_stone_wall := intervals length_of_side_W_stone_wall
def posts_along_short_sides := 2 * (intervals short_side - 1)

-- Calculate total posts
def total_posts := posts_along_w_stone_wall + posts_along_short_sides

-- Define the theorem
theorem posts_needed : total_posts = 19 := 
by
  sorry

end posts_needed_l40_40389


namespace width_of_box_l40_40207

theorem width_of_box (w : ℝ) (h1 : w > 0) 
    (length : ℝ) (h2 : length = 60) 
    (area_lawn : ℝ) (h3 : area_lawn = 2109) 
    (width_road : ℝ) (h4 : width_road = 3) 
    (crossroads : ℝ) (h5 : crossroads = 2 * (60 / 3 * 3)) :
    60 * w - 120 = 2109 → w = 37.15 := 
by 
  intro h6
  sorry

end width_of_box_l40_40207


namespace find_a_b_c_l40_40479

theorem find_a_b_c (a b c : ℝ) 
  (h_min : ∀ x, -9 * x^2 + 54 * x - 45 ≥ 36) 
  (h1 : 0 = a * (1 - 1) * (1 - 5)) 
  (h2 : 0 = a * (5 - 1) * (5 - 5)) :
  a + b + c = 36 :=
sorry

end find_a_b_c_l40_40479


namespace ratio_of_x_to_y_l40_40721

-- Given condition: The percentage that y is less than x is 83.33333333333334%.
def percentage_less_than (x y : ℝ) : Prop := (x - y) / x = 0.8333333333333334

-- Prove: The ratio R = x / y is 1/6.
theorem ratio_of_x_to_y (x y : ℝ) (h : percentage_less_than x y) : x / y = 6 := 
by sorry

end ratio_of_x_to_y_l40_40721


namespace sum_of_faces_edges_vertices_rectangular_prism_l40_40681

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l40_40681


namespace solve_linear_equation_one_variable_with_parentheses_l40_40584

/--
Theorem: Solving a linear equation in one variable that contains parentheses
is equivalent to the process of:
1. Removing the parentheses,
2. Moving terms,
3. Combining like terms, and
4. Making the coefficient of the unknown equal to 1.

Given: a linear equation in one variable that contains parentheses
Prove: The process of solving it is to remove the parentheses, move terms, combine like terms, and make the coefficient of the unknown equal to 1.
-/
theorem solve_linear_equation_one_variable_with_parentheses
  (eq : String) :
  ∃ instructions : String,
    instructions = "remove the parentheses; move terms; combine like terms; make the coefficient of the unknown equal to 1" :=
by
  sorry

end solve_linear_equation_one_variable_with_parentheses_l40_40584


namespace prime_probability_l40_40246

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def pairs : List (ℕ × ℕ) :=
(primes.erase 2).map (λ p => (2, p))

def prime_sums : List (ℕ × ℕ) :=
pairs.filter (λ ⟨a, b⟩ => is_prime (a + b))

theorem prime_probability :
  let total_pairs := Nat.choose 10 2
  let valid_pairs := prime_sums.length
  (valid_pairs : ℚ) / (total_pairs : ℚ) = 1 / 9 :=
by
  sorry

end prime_probability_l40_40246


namespace laps_remaining_eq_five_l40_40214

variable (total_distance : ℕ)
variable (distance_per_lap : ℕ)
variable (laps_already_run : ℕ)

theorem laps_remaining_eq_five 
  (h1 : total_distance = 99) 
  (h2 : distance_per_lap = 9) 
  (h3 : laps_already_run = 6) : 
  (total_distance / distance_per_lap - laps_already_run = 5) :=
by 
  sorry

end laps_remaining_eq_five_l40_40214


namespace factorization_of_polynomial_l40_40747

theorem factorization_of_polynomial :
  ∀ x : ℝ, x^2 + 6 * x + 9 - 64 * x^4 = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by
  intro x
  -- Sorry placeholder for the proof
  sorry

end factorization_of_polynomial_l40_40747


namespace first_train_left_time_l40_40213

-- Definitions for conditions
def speed_first_train := 45
def speed_second_train := 90
def meeting_distance := 90

-- Prove the statement
theorem first_train_left_time (T : ℝ) (time_meeting : ℝ) :
  (time_meeting - T = 2) →
  (∀ t, 0 ≤ t → t ≤ 1 → speed_first_train * t ≤ meeting_distance) →
  (∀ t, 1 ≤ t → speed_first_train * (T + t) + speed_second_train * (t - 1) = meeting_distance) →
  (time_meeting = 2 + T) :=
by
  sorry

end first_train_left_time_l40_40213


namespace max_area_of_triangle_ABC_l40_40900

noncomputable def max_triangle_area (a b c : ℝ) (A B C : ℝ) := 
  1 / 2 * b * c * Real.sin A

theorem max_area_of_triangle_ABC :
  ∀ (a b c A B C : ℝ)
  (ha : a = 2)
  (hTrig : a = Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A))
  (hCondition: 3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0),
  max_triangle_area a b c A B C ≤ 2 := 
by
  intros a b c A B C ha hTrig hCondition
  sorry

end max_area_of_triangle_ABC_l40_40900


namespace sum_of_faces_edges_vertices_of_rect_prism_l40_40688

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l40_40688


namespace max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l40_40767

-- Definitions and conditions related to the given problem
def unit_circle (r : ℝ) : Prop := r = 1

-- Maximum number of non-intersecting circles of radius 1 tangent to a unit circle.
theorem max_non_intersecting_circles_tangent (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 6 := sorry

-- Maximum number of circles of radius 1 intersecting a given unit circle without intersecting centers.
theorem max_intersecting_circles_without_center_containment (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 12 := sorry

-- Maximum number of circles of radius 1 intersecting a unit circle K without containing the center of K or any other circle's center.
theorem max_intersecting_circles_without_center_containment_2 (r : ℝ) (K : ℝ)
  (h_r : unit_circle r) (h_K : unit_circle K) :
  ∃ n, n = 18 := sorry

end max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l40_40767


namespace train_speed_is_126_kmh_l40_40528

noncomputable def train_speed_proof : Prop :=
  let length_meters := 560 / 1000           -- Convert length to kilometers
  let time_hours := 16 / 3600               -- Convert time to hours
  let speed := length_meters / time_hours   -- Calculate the speed
  speed = 126                               -- The speed should be 126 km/h

theorem train_speed_is_126_kmh : train_speed_proof := by 
  sorry

end train_speed_is_126_kmh_l40_40528


namespace ratio_w_to_y_l40_40422

variable (w x y z : ℚ)
variable (h1 : w / x = 5 / 4)
variable (h2 : y / z = 5 / 3)
variable (h3 : z / x = 1 / 5)

theorem ratio_w_to_y : w / y = 15 / 4 := sorry

end ratio_w_to_y_l40_40422


namespace opposite_face_of_X_is_Y_l40_40839

-- Define the labels for the cube faces
inductive Label
| X | V | Z | W | U | Y

-- Define adjacency relations
def adjacent (a b : Label) : Prop :=
  (a = Label.X ∧ (b = Label.V ∨ b = Label.Z ∨ b = Label.W ∨ b = Label.U)) ∨
  (b = Label.X ∧ (a = Label.V ∨ a = Label.Z ∨ a = Label.W ∨ a = Label.U))

-- Define the theorem to prove the face opposite to X
theorem opposite_face_of_X_is_Y : ∀ l1 l2 l3 l4 l5 l6 : Label,
  l1 = Label.X →
  l2 = Label.V →
  l3 = Label.Z →
  l4 = Label.W →
  l5 = Label.U →
  l6 = Label.Y →
  ¬ adjacent l1 l6 →
  ¬ adjacent l2 l6 →
  ¬ adjacent l3 l6 →
  ¬ adjacent l4 l6 →
  ¬ adjacent l5 l6 →
  ∃ (opposite : Label), opposite = Label.Y ∧ opposite = l6 :=
by sorry

end opposite_face_of_X_is_Y_l40_40839


namespace line_intercepts_l40_40636

-- Definitions
def point_on_axis (a b : ℝ) : Prop := a = b
def passes_through_point (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 1

theorem line_intercepts (a b x y : ℝ) (hx : x = -1) (hy : y = 2) (intercept_property : point_on_axis a b) (point_property : passes_through_point a b x y) :
  (2 * x + y = 0) ∨ (x + y - 1 = 0) :=
sorry

end line_intercepts_l40_40636


namespace remainder_13_pow_51_mod_5_l40_40654

theorem remainder_13_pow_51_mod_5 : 13^51 % 5 = 2 := by
  sorry

end remainder_13_pow_51_mod_5_l40_40654


namespace age_difference_l40_40144

noncomputable def years_older (A B : ℕ) : ℕ :=
A - B

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) :
  years_older A B = 9 :=
by
  rw [years_older]
  rw [h1] at h2
  sorry

end age_difference_l40_40144


namespace sqrt_x_plus_sqrt_inv_x_l40_40334

theorem sqrt_x_plus_sqrt_inv_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  (Real.sqrt x + 1 / Real.sqrt x) = Real.sqrt 52 := 
by
  sorry

end sqrt_x_plus_sqrt_inv_x_l40_40334


namespace intersection_of_circle_and_line_in_polar_coordinates_l40_40906

noncomputable section

def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

theorem intersection_of_circle_and_line_in_polar_coordinates :
  ∀ θ ρ, (0 < θ ∧ θ < Real.pi) →
  circle_polar_eq ρ θ →
  line_polar_eq ρ θ →
  ρ = 1 ∧ θ = Real.pi / 2 :=
by
  sorry

end intersection_of_circle_and_line_in_polar_coordinates_l40_40906


namespace probability_prime_sum_is_1_9_l40_40225

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end probability_prime_sum_is_1_9_l40_40225


namespace line_through_point_equal_intercepts_l40_40749

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (x y a : ℝ) (k : ℝ) 
  (hP : P = (2, 3))
  (hx : x / a + y / a = 1 ∨ (P.fst * k - P.snd = 0)) :
  (x + y - 5 = 0 ∨ 3 * P.fst - 2 * P.snd = 0) := by
  sorry

end line_through_point_equal_intercepts_l40_40749


namespace min_perimeter_l40_40649

theorem min_perimeter :
  ∃ (a b c : ℕ), 
  (2 * a + 18 * c = 2 * b + 20 * c) ∧ 
  (9 * Real.sqrt (a^2 - (9 * c)^2) = 10 * Real.sqrt (b^2 - (10 * c)^2)) ∧ 
  (10 * (a - b) = 9 * c) ∧ 
  2 * a + 18 * c = 362 := 
sorry

end min_perimeter_l40_40649


namespace polynomial_root_arithmetic_sequence_l40_40635

theorem polynomial_root_arithmetic_sequence :
  (∃ (a d : ℝ), 
    (64 * (a - d)^3 + 144 * (a - d)^2 + 92 * (a - d) + 15 = 0) ∧
    (64 * a^3 + 144 * a^2 + 92 * a + 15 = 0) ∧
    (64 * (a + d)^3 + 144 * (a + d)^2 + 92 * (a + d) + 15 = 0) ∧
    (2 * d = 1)) := sorry

end polynomial_root_arithmetic_sequence_l40_40635


namespace arithmetic_sequence_75th_term_l40_40315

theorem arithmetic_sequence_75th_term (a d n : ℕ) (h1 : a = 2) (h2 : d = 4) (h3 : n = 75) : 
  a + (n - 1) * d = 298 :=
by 
  sorry

end arithmetic_sequence_75th_term_l40_40315


namespace each_friend_paid_l40_40910

def cottage_cost_per_hour : ℕ := 5
def rental_duration_hours : ℕ := 8
def total_cost := cottage_cost_per_hour * rental_duration_hours
def cost_per_person := total_cost / 2

theorem each_friend_paid : cost_per_person = 20 :=
by 
  sorry

end each_friend_paid_l40_40910


namespace minimum_tickets_needed_l40_40969

noncomputable def min_tickets {α : Type*} (winning_permutation : Fin 50 → α) (tickets : List (Fin 50 → α)) : ℕ :=
  List.length tickets

theorem minimum_tickets_needed
  (winning_permutation : Fin 50 → ℕ)
  (tickets : List (Fin 50 → ℕ))
  (h_tickets_valid : ∀ t ∈ tickets, Function.Surjective t)
  (h_at_least_one_match : ∀ winning_permutation : Fin 50 → ℕ,
      ∃ t ∈ tickets, ∃ i : Fin 50, t i = winning_permutation i) : 
  min_tickets winning_permutation tickets ≥ 26 :=
sorry

end minimum_tickets_needed_l40_40969


namespace tan_alpha_solution_l40_40111

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l40_40111


namespace largest_consecutive_sum_55_l40_40374

theorem largest_consecutive_sum_55 : 
  ∃ k n : ℕ, k > 0 ∧ (∃ n : ℕ, 55 = k * n + (k * (k - 1)) / 2 ∧ k = 5) :=
sorry

end largest_consecutive_sum_55_l40_40374


namespace translation_coordinates_l40_40148

theorem translation_coordinates (A : ℝ × ℝ) (T : ℝ × ℝ) (A' : ℝ × ℝ) 
  (hA : A = (-4, 3)) (hT : T = (2, 0)) (hA' : A' = (A.1 + T.1, A.2 + T.2)) : 
  A' = (-2, 3) := sorry

end translation_coordinates_l40_40148


namespace evaluateExpression_at_1_l40_40694

noncomputable def evaluateExpression (x : ℝ) : ℝ :=
  (x^2 - 3 * x - 10) / (x - 5)

theorem evaluateExpression_at_1 : evaluateExpression 1 = 3 :=
by
  sorry

end evaluateExpression_at_1_l40_40694


namespace range_of_a_variance_Y_when_a_half_options_a_and_d_are_correct_l40_40419

open ProbabilityTheory

-- Definitions for the random variables X and Y
def X_pmf (a : ℝ) : ProbabilityMassFunction ℝ :=
  ⟨[
      (1/3, -1),
      ((2 - a) / 3, 0),
      (a / 3, 1)
    ], by norm_num [add_assoc]⟩

def Y_pmf (a : ℝ) : ProbabilityMassFunction ℝ :=
  ⟨[
     (1/2, 0),
     ((1 - a) / 2, 1),
     (a / 2, 2)
   ], by norm_num [add_assoc]⟩

-- Definition to check the range of a
theorem range_of_a (a : ℝ) : 0 ≤ a ∧ a ≤ 1 :=
begin
  split,
  { norm_num },
  { norm_num },
  sorry
end

-- Definition to check the variance of Y when a = 1/2
theorem variance_Y_when_a_half : (D(Y) = 11 / 16) ∧ (a = 1/2) :=
begin
  sorry
end

-- Stating the main question as a theorem  
theorem options_a_and_d_are_correct (a : ℝ) : 
  (0 ≤ a ∧ a ≤ 1) ∧ 
  (a = 1/2 → D(Y) = 11 / 16) :=
begin
  refine ⟨range_of_a a, _⟩,
  intro ha,
  rw ha,
  exact variance_Y_when_a_half
end

end range_of_a_variance_Y_when_a_half_options_a_and_d_are_correct_l40_40419


namespace find_angle_measure_l40_40431

def complement_more_condition (x : ℝ) : Prop :=
  90 - x = (1 / 7) * x + 26

theorem find_angle_measure (x : ℝ) (h : complement_more_condition x) : x = 56 :=
sorry

end find_angle_measure_l40_40431


namespace binomial_variance_l40_40884

-- Define the parameters of the binomial distribution
def n := 10
def p := 2 / 5

-- Statement of the proof problem
theorem binomial_variance (ξ : ℕ → ℕ) 
  (h : ∀ k, ξ k = binomial n p k) : 
  variance ξ = 12 / 5 :=
sorry

end binomial_variance_l40_40884


namespace water_depth_in_cylindrical_tub_l40_40976

theorem water_depth_in_cylindrical_tub
  (tub_diameter : ℝ) (tub_depth : ℝ) (pail_angle : ℝ)
  (h_diam : tub_diameter = 40)
  (h_depth : tub_depth = 50)
  (h_angle : pail_angle = 45) :
  ∃ water_depth : ℝ, water_depth = 30 :=
by
  sorry

end water_depth_in_cylindrical_tub_l40_40976


namespace dice_probability_even_sum_one_six_l40_40500

theorem dice_probability_even_sum_one_six :
  let outcomes := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                   (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)] in
  let favorable := [(6, 2), (6, 4), (6, 6), (2, 6), (4, 6)] in
  (favorable.length : ℚ) / (outcomes.length : ℚ) = 5 / 36 :=
by
  sorry

end dice_probability_even_sum_one_six_l40_40500


namespace function_inequality_l40_40763

noncomputable def f : ℝ → ℝ
| x => if x < 1 then (x + 1)^2 else 4 - Real.sqrt (x - 1)

theorem function_inequality : 
  {x : ℝ | f x ≥ x} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end function_inequality_l40_40763


namespace compute_fg_l40_40308

def g (x : ℕ) : ℕ := 2 * x + 6
def f (x : ℕ) : ℕ := 4 * x - 8
def x : ℕ := 10

theorem compute_fg : f (g x) = 96 := by
  sorry

end compute_fg_l40_40308


namespace theorem_227_l40_40742

theorem theorem_227 (a b c d : ℤ) (k : ℤ) (h : b ≡ c [ZMOD d]) :
  (a + b ≡ a + c [ZMOD d]) ∧
  (a - b ≡ a - c [ZMOD d]) ∧
  (a * b ≡ a * c [ZMOD d]) :=
by
  sorry

end theorem_227_l40_40742


namespace divide_talers_l40_40807

theorem divide_talers (loaves1 loaves2 : ℕ) (coins : ℕ) (loavesShared : ℕ) :
  loaves1 = 3 → loaves2 = 5 → coins = 8 → loavesShared = (loaves1 + loaves2) →
  (3 - loavesShared / 3) * coins / loavesShared = 1 ∧ (5 - loavesShared / 3) * coins / loavesShared = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end divide_talers_l40_40807


namespace trig_identity_l40_40889

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin (2 * α)) = 25 / 64 := 
by
  sorry

end trig_identity_l40_40889


namespace min_large_buses_proof_l40_40050

def large_bus_capacity : ℕ := 45
def small_bus_capacity : ℕ := 30
def total_students : ℕ := 523
def min_small_buses : ℕ := 5

def min_large_buses_required (large_capacity small_capacity total small_buses : ℕ) : ℕ :=
  let remaining_students := total - (small_buses * small_capacity)
  let buses_needed := remaining_students / large_capacity
  if remaining_students % large_capacity = 0 then buses_needed else buses_needed + 1

theorem min_large_buses_proof :
  min_large_buses_required large_bus_capacity small_bus_capacity total_students min_small_buses = 9 :=
by
  sorry

end min_large_buses_proof_l40_40050


namespace small_planters_needed_l40_40169

-- This states the conditions for the problem
def Oshea_seeds := 200
def large_planters := 4
def large_planter_capacity := 20
def small_planter_capacity := 4
def remaining_seeds := Oshea_seeds - (large_planters * large_planter_capacity) 

-- The target we aim to prove: the number of small planters required
theorem small_planters_needed :
  remaining_seeds / small_planter_capacity = 30 := by
  sorry

end small_planters_needed_l40_40169


namespace find_a_l40_40766

def A (a : ℤ) : Set ℤ := {-4, 2 * a - 1, a * a}
def B (a : ℤ) : Set ℤ := {a - 5, 1 - a, 9}

theorem find_a (a : ℤ) : (9 ∈ (A a ∩ B a)) ∧ (A a ∩ B a = {9}) ↔ a = -3 :=
by
  sorry

end find_a_l40_40766


namespace probability_prime_sum_is_1_9_l40_40227

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end probability_prime_sum_is_1_9_l40_40227


namespace trig_relation_l40_40160

theorem trig_relation (a b c : ℝ) 
  (h1 : a = Real.sin 2) 
  (h2 : b = Real.cos 2) 
  (h3 : c = Real.tan 2) : c < b ∧ b < a := 
by
  sorry

end trig_relation_l40_40160


namespace find_f_neg_2_l40_40085

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x: ℝ, f (-x) = -f x

-- Problem statement
theorem find_f_neg_2 (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_fx_pos : ∀ x : ℝ, x > 0 → f x = 2 * x ^ 2 - 7) : 
  f (-2) = -1 :=
by
  sorry

end find_f_neg_2_l40_40085


namespace smallest_number_of_packs_l40_40278

theorem smallest_number_of_packs (n b w : ℕ) (Hn : n = 13) (Hb : b = 8) (Hw : w = 17) :
  Nat.lcm (Nat.lcm n b) w = 1768 :=
by
  sorry

end smallest_number_of_packs_l40_40278


namespace tina_pink_pens_l40_40953

def number_pink_pens (P G B : ℕ) : Prop :=
  G = P - 9 ∧
  B = P - 6 ∧
  P + G + B = 21

theorem tina_pink_pens :
  ∃ (P G B : ℕ), number_pink_pens P G B ∧ P = 12 :=
by
  sorry

end tina_pink_pens_l40_40953


namespace arithmetic_sequence_problem_l40_40317

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :=
  ∀ n, a n = a1 + n * d

-- Given condition
variable (h1 : a 3 + a 4 + a 5 = 36)

-- The goal is to prove that a 0 + a 8 = 24
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :
  arithmetic_sequence a a1 d →
  a 3 + a 4 + a 5 = 36 →
  a 0 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_problem_l40_40317


namespace chocolate_eggs_weeks_l40_40338

theorem chocolate_eggs_weeks (e: ℕ) (d: ℕ) (w: ℕ) (total: ℕ) (weeks: ℕ) 
    (initialEggs : e = 40)
    (dailyEggs : d = 2)
    (schoolDays : w = 5)
    (totalWeeks : weeks = total):
    total = e / (d * w) := by
sorry

end chocolate_eggs_weeks_l40_40338


namespace find_point_on_line_l40_40282

theorem find_point_on_line (x y : ℝ) (h : 4 * x + 7 * y = 28) (hx : x = 3) : y = 16 / 7 :=
by
  sorry

end find_point_on_line_l40_40282


namespace sqrt_product_equals_l40_40398

noncomputable def sqrt128 : ℝ := Real.sqrt 128
noncomputable def sqrt50 : ℝ := Real.sqrt 50
noncomputable def sqrt18 : ℝ := Real.sqrt 18

theorem sqrt_product_equals : sqrt128 * sqrt50 * sqrt18 = 240 * Real.sqrt 2 := 
by
  sorry

end sqrt_product_equals_l40_40398


namespace train_cross_pole_in_time_l40_40700

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_pole_in_time :
  time_to_cross_pole 100 126 = 100 / (126 * (1000 / 3600)) :=
by
  -- this will unfold the calculation step-by-step
  unfold time_to_cross_pole
  sorry

end train_cross_pole_in_time_l40_40700


namespace length_of_boat_l40_40044

-- Definitions based on the conditions
def breadth : ℝ := 3
def sink_depth : ℝ := 0.01
def man_mass : ℝ := 120
def g : ℝ := 9.8 -- acceleration due to gravity

-- Derived from the conditions
def weight_man : ℝ := man_mass * g
def density_water : ℝ := 1000

-- Statement to be proved
theorem length_of_boat : ∃ L : ℝ, (breadth * sink_depth * L * density_water * g = weight_man) → L = 4 :=
by
  sorry

end length_of_boat_l40_40044


namespace probability_prime_sum_l40_40252

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end probability_prime_sum_l40_40252


namespace least_integer_with_twelve_factors_l40_40015

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l40_40015


namespace count_pairs_divisible_by_nine_l40_40416

open Nat

theorem count_pairs_divisible_by_nine (n : ℕ) (h : n = 528) :
  ∃ (count : ℕ), count = n ∧
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 100 ∧ (a^2 + b^2 + a * b) % 9 = 0 ↔
  count = 528 :=
by
  sorry

end count_pairs_divisible_by_nine_l40_40416


namespace find_a_plus_b_l40_40290

theorem find_a_plus_b (a b : ℤ) (h1 : a^2 = 16) (h2 : b^3 = -27) (h3 : |a - b| = a - b) : a + b = 1 := by
  sorry

end find_a_plus_b_l40_40290


namespace major_axis_length_l40_40576

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by sorry

end major_axis_length_l40_40576


namespace inv_sum_mod_l40_40851

theorem inv_sum_mod (x y : ℤ) (h1 : 5 * x ≡ 1 [ZMOD 23]) (h2 : 25 * y ≡ 1 [ZMOD 23]) : (x + y) ≡ 3 [ZMOD 23] := by
  sorry

end inv_sum_mod_l40_40851


namespace sum_of_squares_xy_l40_40189

theorem sum_of_squares_xy (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end sum_of_squares_xy_l40_40189


namespace largest_consecutive_sum_55_l40_40375

theorem largest_consecutive_sum_55 :
  ∃ n a : ℕ, (n * (a + (n - 1) / 2) = 55) ∧ (n = 10) ∧ (∀ m : ℕ, ∀ b : ℕ, (m * (b + (m - 1) / 2) = 55) → (m ≤ 10)) :=
by 
  sorry

end largest_consecutive_sum_55_l40_40375


namespace math_problem_solution_l40_40777

noncomputable def problem_statement : Prop :=
  let AB := 4
  let AC := 6
  let BC := 5
  let area_ABC := 9.9216 -- Using the approximated area directly for simplicity
  let K_div3 := area_ABC / 3
  let GP := (2 * K_div3) / BC
  let GQ := (2 * K_div3) / AC
  let GR := (2 * K_div3) / AB
  GP + GQ + GR = 4.08432

theorem math_problem_solution : problem_statement :=
by
  sorry

end math_problem_solution_l40_40777


namespace fangfang_travel_time_l40_40075

theorem fangfang_travel_time (time_1_to_5 : ℕ) (start_floor end_floor : ℕ) (floors_1_to_5 : ℕ) (floors_2_to_7 : ℕ) :
  time_1_to_5 = 40 →
  floors_1_to_5 = 5 - 1 →
  floors_2_to_7 = 7 - 2 →
  end_floor = 7 →
  start_floor = 2 →
  (end_floor - start_floor) * (time_1_to_5 / floors_1_to_5) = 50 :=
by 
  sorry

end fangfang_travel_time_l40_40075


namespace calc_product_eq_243_l40_40993

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l40_40993


namespace Tyler_needs_more_eggs_l40_40948

noncomputable def recipe_eggs : ℕ := 2
noncomputable def recipe_milk : ℕ := 4
noncomputable def num_people : ℕ := 8
noncomputable def eggs_in_fridge : ℕ := 3

theorem Tyler_needs_more_eggs (recipe_eggs recipe_milk num_people eggs_in_fridge : ℕ)
  (h1 : recipe_eggs = 2)
  (h2 : recipe_milk = 4)
  (h3 : num_people = 8)
  (h4 : eggs_in_fridge = 3) :
  (num_people / 4) * recipe_eggs - eggs_in_fridge = 1 :=
by
  sorry

end Tyler_needs_more_eggs_l40_40948


namespace transmission_time_calc_l40_40546

theorem transmission_time_calc
  (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) (time_in_minutes : ℕ)
  (h_blocks : blocks = 80)
  (h_chunks_per_block : chunks_per_block = 640)
  (h_transmission_rate : transmission_rate = 160) 
  (h_time_in_minutes : time_in_minutes = 5) : 
  (blocks * chunks_per_block / transmission_rate) / 60 = time_in_minutes := 
by
  sorry

end transmission_time_calc_l40_40546


namespace probability_prime_sum_is_1_9_l40_40229

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end probability_prime_sum_is_1_9_l40_40229


namespace tan_cot_expr_simplify_l40_40936

theorem tan_cot_expr_simplify :
  (∀ θ : ℝ, θ = π / 4 → tan θ = 1) →
  (∀ θ : ℝ, θ = π / 4 → cot θ = 1) →
  ( (tan (π / 4)) ^ 3 + (cot (π / 4)) ^ 3) / (tan (π / 4) + cot (π / 4)) = 1 :=
by
  intro h_tan h_cot
  -- The proof goes here, we'll use sorry to skip it
  sorry

end tan_cot_expr_simplify_l40_40936


namespace frog_jumps_further_l40_40794

-- Definitions according to conditions
def grasshopper_jump : ℕ := 36
def frog_jump : ℕ := 53

-- Theorem: The frog jumped 17 inches farther than the grasshopper
theorem frog_jumps_further (g_jump f_jump : ℕ) (h1 : g_jump = grasshopper_jump) (h2 : f_jump = frog_jump) :
  f_jump - g_jump = 17 :=
by
  -- Proof is skipped in this statement
  sorry

end frog_jumps_further_l40_40794


namespace number_of_cases_ordered_in_may_l40_40726

noncomputable def cases_ordered_in_may (ordered_in_april_cases : ℕ) (bottles_per_case : ℕ) (total_bottles : ℕ) : ℕ :=
  let bottles_in_april := ordered_in_april_cases * bottles_per_case
  let bottles_in_may := total_bottles - bottles_in_april
  bottles_in_may / bottles_per_case

theorem number_of_cases_ordered_in_may :
  ∀ (ordered_in_april_cases bottles_per_case total_bottles : ℕ),
  ordered_in_april_cases = 20 →
  bottles_per_case = 20 →
  total_bottles = 1000 →
  cases_ordered_in_may ordered_in_april_cases bottles_per_case total_bottles = 30 := by
  intros ordered_in_april_cases bottles_per_case total_bottles ha hbp htt
  sorry

end number_of_cases_ordered_in_may_l40_40726


namespace additional_pots_last_hour_l40_40034

theorem additional_pots_last_hour (h1 : 60 / 6 = 10) (h2 : 60 / 5 = 12) : 12 - 10 = 2 :=
by
  sorry

end additional_pots_last_hour_l40_40034


namespace b_product_l40_40298

variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- All terms in the arithmetic sequence \{aₙ\} are non-zero.
axiom a_nonzero : ∀ n, a n ≠ 0

-- The sequence satisfies the given condition.
axiom a_cond : a 3 - (a 7)^2 / 2 + a 11 = 0

-- The sequence \{bₙ\} is a geometric sequence with ratio r.
axiom b_geometric : ∃ r, ∀ n, b (n + 1) = r * b n

-- And b₇ = a₇
axiom b_7 : b 7 = a 7

-- Prove that b₁ * b₁₃ = 16
theorem b_product : b 1 * b 13 = 16 :=
sorry

end b_product_l40_40298


namespace units_digit_of_x_l40_40565

theorem units_digit_of_x 
  (a x : ℕ) 
  (h1 : a * x = 14^8) 
  (h2 : a % 10 = 9) : 
  x % 10 = 4 := 
by 
  sorry

end units_digit_of_x_l40_40565


namespace triangle_leg_ratio_l40_40392

theorem triangle_leg_ratio :
  ∀ (a b : ℝ) (h₁ : a = 4) (h₂ : b = 2 * Real.sqrt 5),
    ((a / b) = (2 * Real.sqrt 5) / 5) :=
by
  intros a b h₁ h₂
  sorry

end triangle_leg_ratio_l40_40392


namespace five_alpha_plus_two_beta_is_45_l40_40157

theorem five_alpha_plus_two_beta_is_45
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (tan_β : Real.tan β = 3 / 79) :
  5 * α + 2 * β = π / 4 :=
by
  sorry

end five_alpha_plus_two_beta_is_45_l40_40157


namespace value_of_a_plus_b_l40_40894

variables (a b : ℝ)

theorem value_of_a_plus_b (ha : abs a = 1) (hb : abs b = 4) (hab : a * b < 0) : a + b = 3 ∨ a + b = -3 := by
  sorry

end value_of_a_plus_b_l40_40894


namespace parabola_directrix_l40_40863

theorem parabola_directrix (x : ℝ) : 
  (6 * x^2 + 5 = y) → (y = 6 * x^2 + 5) → (y = 6 * 0^2 + 5) → (y = (119 : ℝ) / 24) := 
sorry

end parabola_directrix_l40_40863


namespace book_price_net_change_l40_40589

theorem book_price_net_change (P : ℝ) :
  let decreased_price := P * 0.70
  let increased_price := decreased_price * 1.20
  let net_change := (increased_price - P) / P * 100
  net_change = -16 := 
by
  sorry

end book_price_net_change_l40_40589


namespace prime_triples_eq_l40_40861

open Nat

/-- Proof problem statement: Prove that the set of tuples (p, q, r) such that p, q, r 
      are prime numbers and p^q + p^r is a perfect square is exactly 
      {(2,2,5), (2,5,2), (3,2,3), (3,3,2)} ∪ {(2, q, q) | q ≥ 3 ∧ Prime q}. --/
theorem prime_triples_eq:
  ∀ (p q r : ℕ), Prime p → Prime q → Prime r → (∃ n, n^2 = p^q + p^r) ↔ 
  {(p, q, r) | 
    p = 2 ∧ (q = q ∧ q ≥ 3 ∧ Prime q) ∨ 
    p = 2 ∧ ((q = 2 ∧ r = 5) ∨ (q = 5 ∧ r = 2)) ∨
    p = 3 ∧ ((q = 2 ∧ r = 3) ∨ (q = 3 ∧ r = 2))}. 

end prime_triples_eq_l40_40861


namespace solution_set_of_f_gt_7_minimum_value_of_m_n_l40_40420

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

theorem solution_set_of_f_gt_7 :
  { x : ℝ | f x > 7 } = { x | x > 4 ∨ x < -3 } :=
by
  ext x
  sorry

theorem minimum_value_of_m_n (m n : ℝ) (h : 0 < m ∧ 0 < n) (hfmin : ∀ x : ℝ, f x ≥ m + n) :
  m = n ∧ m = 3 / 2 ∧ m^2 + n^2 = 9 / 2 :=
by
  sorry

end solution_set_of_f_gt_7_minimum_value_of_m_n_l40_40420


namespace pyramid_volume_84sqrt10_l40_40054

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1/3) * a * b * c

theorem pyramid_volume_84sqrt10 :
  let height := 4 * (Real.sqrt 10)
  let area_base := 7 * 9
  (volume_of_pyramid area_base height) = 84 * (Real.sqrt 10) :=
by
  intros
  simp [volume_of_pyramid]
  sorry

end pyramid_volume_84sqrt10_l40_40054


namespace least_positive_integer_with_12_factors_l40_40003

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l40_40003


namespace willie_bananas_remain_same_l40_40377

variable (Willie_bananas Charles_bananas Charles_loses : ℕ)

theorem willie_bananas_remain_same (h_willie : Willie_bananas = 48) (h_charles_initial : Charles_bananas = 14) (h_charles_loses : Charles_loses = 35) :
  Willie_bananas = 48 :=
by
  sorry

end willie_bananas_remain_same_l40_40377


namespace solve_3_pow_n_plus_55_eq_m_squared_l40_40937

theorem solve_3_pow_n_plus_55_eq_m_squared :
  ∃ (n m : ℕ), 3^n + 55 = m^2 ∧ ((n = 2 ∧ m = 8) ∨ (n = 6 ∧ m = 28)) :=
by
  sorry

end solve_3_pow_n_plus_55_eq_m_squared_l40_40937


namespace carpet_area_l40_40984

-- Definitions
def Rectangle1 (length1 width1 : ℕ) : Prop :=
  length1 = 12 ∧ width1 = 9

def Rectangle2 (length2 width2 : ℕ) : Prop :=
  length2 = 6 ∧ width2 = 9

def feet_to_yards (feet : ℕ) : ℕ :=
  feet / 3

-- Statement to prove
theorem carpet_area (length1 width1 length2 width2 : ℕ) (h1 : Rectangle1 length1 width1) (h2 : Rectangle2 length2 width2) :
  feet_to_yards (length1 * width1) / 3 + feet_to_yards (length2 * width2) / 3 = 18 :=
by
  sorry

end carpet_area_l40_40984


namespace nancy_marks_home_economics_l40_40928

-- Definitions from conditions
def marks_american_lit := 66
def marks_history := 75
def marks_physical_ed := 68
def marks_art := 89
def average_marks := 70
def num_subjects := 5
def total_marks := average_marks * num_subjects
def marks_other_subjects := marks_american_lit + marks_history + marks_physical_ed + marks_art

-- Statement to prove
theorem nancy_marks_home_economics : 
  (total_marks - marks_other_subjects = 52) := by 
  sorry

end nancy_marks_home_economics_l40_40928


namespace inverse_proportion_k_value_l40_40312

theorem inverse_proportion_k_value (k : ℝ) (h₁ : k ≠ 0) (h₂ : (2, -1) ∈ {p : ℝ × ℝ | ∃ (k' : ℝ), k' = k ∧ p.snd = k' / p.fst}) :
  k = -2 := 
by
  sorry

end inverse_proportion_k_value_l40_40312


namespace triangle_sine_equality_l40_40173

theorem triangle_sine_equality {a b c : ℝ} {α β γ : ℝ} 
  (cos_rule : c^2 = a^2 + b^2 - 2 * a * b * Real.cos γ)
  (area : ∃ T : ℝ, T = (1 / 2) * a * b * Real.sin γ)
  (sin_addition_γ : Real.sin (γ + Real.pi / 6) = Real.sin γ * (Real.sqrt 3 / 2) + Real.cos γ * (1 / 2))
  (sin_addition_β : Real.sin (β + Real.pi / 6) = Real.sin β * (Real.sqrt 3 / 2) + Real.cos β * (1 / 2))
  (sin_addition_α : Real.sin (α + Real.pi / 6) = Real.sin α * (Real.sqrt 3 / 2) + Real.cos α * (1 / 2)) :
  c^2 + 2 * a * b * Real.sin (γ + Real.pi / 6) = b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) ∧
  b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) = a^2 + 2 * b * c * Real.sin (α + Real.pi / 6) :=
sorry

end triangle_sine_equality_l40_40173


namespace sufficient_but_not_necessary_condition_l40_40451

theorem sufficient_but_not_necessary_condition (A B : Set ℝ) :
  (A = {x : ℝ | 1 < x ∧ x < 3}) →
  (B = {x : ℝ | x > -1}) →
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) :=
by
  sorry

end sufficient_but_not_necessary_condition_l40_40451


namespace hyperbola_center_l40_40720

theorem hyperbola_center (F1 F2 : ℝ × ℝ) (F1_eq : F1 = (3, -2)) (F2_eq : F2 = (11, 6)) :
  let C := ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2) in C = (7, 2) :=
by
  sorry

end hyperbola_center_l40_40720


namespace proof_problem_l40_40769

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom condition1 : ∀ x : ℝ, f x + x * g x = x ^ 2 - 1
axiom condition2 : f 1 = 1

theorem proof_problem : deriv f 1 + deriv g 1 = 3 :=
by
  sorry

end proof_problem_l40_40769


namespace prime_pair_probability_l40_40234

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrimeSum (a b : ℕ) : Prop := Nat.Prime (a + b)

theorem prime_pair_probability :
  let pairs := List.sublistsLen 2 firstTenPrimes
  let primePairs := pairs.filter (λ p, match p with
                                       | [a, b] => isPrimeSum a b
                                       | _ => false
                                       end)
  (primePairs.length : ℚ) / (pairs.length : ℚ) = 1 / 9 :=
by
  sorry

end prime_pair_probability_l40_40234


namespace solve_for_a_b_c_l40_40570

-- Conditions and necessary context
def m_angle_A : ℝ := 60  -- In degrees
def BC_length : ℝ := 12  -- Length of BC in units
def angle_DBC_eq_three_times_angle_ECB (DBC ECB : ℝ) : Prop := DBC = 3 * ECB

-- Definitions for perpendicularity could be checked by defining angles
-- between lines, but we can assert these as properties.
axiom BD_perpendicular_AC : Prop
axiom CE_perpendicular_AB : Prop

-- The proof problem
theorem solve_for_a_b_c :
  ∃ (EC a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  b ≠ c ∧ 
  (∀ d, b ∣ d → d = b ∨ d = 1) ∧ 
  (∀ d, c ∣ d → d = c ∨ d = 1) ∧
  EC = a * (Real.sqrt b + Real.sqrt c) ∧ 
  a + b + c = 11 :=
by
  sorry

end solve_for_a_b_c_l40_40570


namespace boundary_length_of_divided_rectangle_l40_40387

/-- Suppose a rectangle is divided into three equal parts along its length and two equal parts along its width, 
creating semicircle arcs connecting points on adjacent sides. Given the rectangle has an area of 72 square units, 
we aim to prove that the total length of the boundary of the resulting figure is 36.0. -/
theorem boundary_length_of_divided_rectangle 
(area_of_rectangle : ℝ)
(length_divisions : ℕ)
(width_divisions : ℕ)
(semicircle_arcs_length : ℝ)
(straight_segments_length : ℝ) :
  area_of_rectangle = 72 →
  length_divisions = 3 →
  width_divisions = 2 →
  semicircle_arcs_length = 7 * Real.pi →
  straight_segments_length = 14 →
  semicircle_arcs_length + straight_segments_length = 36 :=
by
  intros h_area h_length_div h_width_div h_arc_length h_straight_length
  sorry

end boundary_length_of_divided_rectangle_l40_40387


namespace completing_square_l40_40811

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l40_40811


namespace calc_exponent_result_l40_40736

theorem calc_exponent_result (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := 
by
  sorry

end calc_exponent_result_l40_40736


namespace yuna_survey_l40_40696

theorem yuna_survey :
  let M := 27
  let K := 28
  let B := 22
  M + K - B = 33 :=
by
  sorry

end yuna_survey_l40_40696


namespace tracy_michelle_distance_ratio_l40_40954

theorem tracy_michelle_distance_ratio :
  ∀ (T M K : ℕ), 
  (M = 294) → 
  (M = 3 * K) → 
  (T + M + K = 1000) →
  ∃ x : ℕ, (T = x * M + 20) ∧ x = 2 :=
by
  intro T M K
  intro hM hMK hDistance
  use 2
  sorry

end tracy_michelle_distance_ratio_l40_40954


namespace probability_odd_and_less_than_5000_l40_40792

def isValidOdd (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 7

def isValidLeading (d : ℕ) : Prop := d = 1 ∨ d = 3

def possibleDig : finset ℕ := {1, 3, 7, 8}

def countValidNumbers : ℕ :=
  let valid_lead_digits := {1, 3}
  let valid_unit_digits := {1, 3, 7}
  valid_lead_digits.sum (λ ld, valid_unit_digits.sum (λ ud, if ld = ud then 2 else 2 * 2))

def countTotalNumbers : ℕ := 12

def probabilityOfValidNumbers : ℚ :=
  (countValidNumbers : ℚ) / (countTotalNumbers : ℚ)

theorem probability_odd_and_less_than_5000 :
  probabilityOfValidNumbers = 2 / 3 :=
by
  have h1 : countValidNumbers = 8 := by sorry
  have h2 : countTotalNumbers = 12 := by sorry
  unfold probabilityOfValidNumbers
  rw [h1, h2]
  norm_num

end probability_odd_and_less_than_5000_l40_40792


namespace square_side_length_l40_40960

theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s^2) (hA : A = 144) : s = 12 :=
by 
  -- sorry is used to skip the proof
  sorry

end square_side_length_l40_40960


namespace b_share_l40_40821

-- Definitions based on the conditions
def salary (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ d = 6 * x

def condition (d c : ℕ) : Prop :=
  d = c + 700

-- Proof problem based on the correct answer
theorem b_share (a b c d : ℕ) (x : ℕ) (salary_cond : salary a b c d) (cond : condition d c) :
  b = 1050 := by
  sorry

end b_share_l40_40821


namespace number_of_students_above_90_l40_40436

noncomputable def num_students_above_90 (students : ℕ) (mean : ℝ) (variance : ℝ) : ℕ :=
  let sigma : ℝ := Real.sqrt variance
  let prob_above_90 : ℝ := 0.5 * (1 - 0.9544)
  let expected_num : ℝ := students * prob_above_90
  Real.to_nat (Real.round expected_num)

theorem number_of_students_above_90 :
  num_students_above_90 1000 80 25 = 23 :=
sorry

end number_of_students_above_90_l40_40436


namespace sum_faces_edges_vertices_l40_40672

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l40_40672


namespace tan_alpha_value_l40_40138

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l40_40138


namespace minimum_value_hyperbola_l40_40353

noncomputable def min_value (a b : ℝ) (h : a > 0) (k : b > 0)
  (eccentricity_eq_two : (2:ℝ) = Real.sqrt (1 + (b/a)^2)) : ℝ :=
  (b^2 + 1) / (3 * a)

theorem minimum_value_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2:ℝ) = Real.sqrt (1 + (b/a)^2) ∧
  min_value a b (by sorry) (by sorry) (by sorry) = (2 * Real.sqrt 3) / 3 :=
sorry

end minimum_value_hyperbola_l40_40353


namespace community_group_loss_l40_40833

def cookies_bought : ℕ := 800
def cost_per_4_cookies : ℚ := 3 -- dollars per 4 cookies
def sell_per_3_cookies : ℚ := 2 -- dollars per 3 cookies

def cost_per_cookie : ℚ := cost_per_4_cookies / 4
def sell_per_cookie : ℚ := sell_per_3_cookies / 3

def total_cost (n : ℕ) (cost_per_cookie : ℚ) : ℚ := n * cost_per_cookie
def total_revenue (n : ℕ) (sell_per_cookie : ℚ) : ℚ := n * sell_per_cookie

def loss (n : ℕ) (cost_per_cookie sell_per_cookie : ℚ) : ℚ := 
  total_cost n cost_per_cookie - total_revenue n sell_per_cookie

theorem community_group_loss : loss cookies_bought cost_per_cookie sell_per_cookie = 64 := by
  sorry

end community_group_loss_l40_40833


namespace cylinder_ellipse_eccentricity_l40_40716

noncomputable def eccentricity_of_ellipse (diameter : ℝ) (angle : ℝ) : ℝ :=
  let r := diameter / 2
  let b := r
  let a := r / (Real.cos angle)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem cylinder_ellipse_eccentricity :
  eccentricity_of_ellipse 12 (Real.pi / 6) = 1 / 2 :=
by
  sorry

end cylinder_ellipse_eccentricity_l40_40716


namespace lcm_48_180_l40_40558

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l40_40558


namespace abs_eq_solution_l40_40194

theorem abs_eq_solution (x : ℚ) : |x - 2| = |x + 3| → x = -1 / 2 :=
by
  sorry

end abs_eq_solution_l40_40194


namespace max_radius_of_inner_spheres_l40_40310

theorem max_radius_of_inner_spheres (R : ℝ) : 
  ∃ r : ℝ, (2 * r ≤ R) ∧ (r ≤ (4 * Real.sqrt 2 - 1) / 4 * R) :=
sorry

end max_radius_of_inner_spheres_l40_40310


namespace fully_charge_tablet_time_l40_40525

def time_to_fully_charge_smartphone := 26 -- 26 minutes to fully charge a smartphone
def total_charge_time := 66 -- 66 minutes to charge tablet fully and phone halfway
def halfway_charge_time := time_to_fully_charge_smartphone / 2 -- 13 minutes to charge phone halfway

theorem fully_charge_tablet_time : 
  ∃ T : ℕ, T + halfway_charge_time = total_charge_time ∧ T = 53 := 
by
  sorry

end fully_charge_tablet_time_l40_40525


namespace probability_correct_l40_40289

-- Definitions of the problem components
def total_beads : Nat := 7
def red_beads : Nat := 4
def white_beads : Nat := 2
def green_bead : Nat := 1

-- The total number of permutations of the given multiset
def total_permutations : Nat :=
  Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial green_bead)

-- The number of valid permutations where no two neighboring beads are the same color
def valid_permutations : Nat := 14 -- As derived in the solution steps

-- The probability that no two neighboring beads are the same color
def probability_no_adjacent_same_color : Rat :=
  valid_permutations / total_permutations

-- The theorem to be proven
theorem probability_correct :
  probability_no_adjacent_same_color = 2 / 15 :=
by
  -- Proof omitted
  sorry

end probability_correct_l40_40289


namespace pipe_individual_empty_time_l40_40058

variable (a b c : ℝ)

noncomputable def timeToEmptyFirstPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * c + b * c - a * b)

noncomputable def timeToEmptySecondPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + b * c - a * c)

noncomputable def timeToEmptyThirdPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + a * c - b * c)

theorem pipe_individual_empty_time
  (x y z : ℝ)
  (h1 : 1 / x + 1 / y = 1 / a)
  (h2 : 1 / x + 1 / z = 1 / b)
  (h3 : 1 / y + 1 / z = 1 / c) :
  x = timeToEmptyFirstPipe a b c ∧ y = timeToEmptySecondPipe a b c ∧ z = timeToEmptyThirdPipe a b c :=
sorry

end pipe_individual_empty_time_l40_40058


namespace multiplier_is_three_l40_40465

theorem multiplier_is_three (n m : ℝ) (h₁ : n = 3) (h₂ : 7 * n = m * n + 12) : m = 3 := 
by
  -- Skipping the proof using sorry
  sorry 

end multiplier_is_three_l40_40465


namespace fg_evaluation_l40_40893

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_evaluation : f (g 3) = 97 := by
  sorry

end fg_evaluation_l40_40893


namespace jeff_makes_donuts_for_days_l40_40443

variable (d : ℕ) (boxes donuts_per_box : ℕ) (donuts_per_day eaten_per_day : ℕ) (chris_eaten total_donuts : ℕ)

theorem jeff_makes_donuts_for_days :
  (donuts_per_day = 10) →
  (eaten_per_day = 1) →
  (chris_eaten = 8) →
  (boxes = 10) →
  (donuts_per_box = 10) →
  (total_donuts = boxes * donuts_per_box) →
  (9 * d - chris_eaten = total_donuts) →
  d = 12 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end jeff_makes_donuts_for_days_l40_40443


namespace minimum_value_of_k_l40_40155

theorem minimum_value_of_k (m n a k : ℕ) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hk : 1 < k) (h : 5^m + 63 * n + 49 = a^k) : k = 5 :=
sorry

end minimum_value_of_k_l40_40155


namespace least_integer_with_twelve_factors_l40_40014

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l40_40014


namespace rectangular_prism_faces_edges_vertices_sum_l40_40679

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l40_40679


namespace least_positive_integer_with_12_factors_l40_40001

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l40_40001


namespace paint_rate_5_l40_40795
noncomputable def rate_per_sq_meter (L : ℝ) (total_cost : ℝ) (B : ℝ) : ℝ :=
  let Area := L * B
  total_cost / Area

theorem paint_rate_5 : 
  ∀ (L B total_cost rate : ℝ),
    L = 19.595917942265423 →
    total_cost = 640 →
    L = 3 * B →
    rate = rate_per_sq_meter L total_cost B →
    rate = 5 :=
by
  intros L B total_cost rate hL hC hR hRate
  -- Proof goes here
  sorry

end paint_rate_5_l40_40795


namespace geo_seq_bn_plus_2_general_formula_an_l40_40789

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 2 = 4
axiom h3 : ∀ n, b n = a (n + 1) - a n
axiom h4 : ∀ n, b (n + 1) = 2 * b n + 2

-- Proof goals
theorem geo_seq_bn_plus_2 : (∀ n, ∃ r : ℕ, b n + 2 = 4 * 2^n) :=
  sorry

theorem general_formula_an : (∀ n, a n = 2^(n + 1) - 2 * n) :=
  sorry

end geo_seq_bn_plus_2_general_formula_an_l40_40789


namespace total_wheels_in_garage_l40_40494

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end total_wheels_in_garage_l40_40494


namespace heavier_boxes_weight_l40_40437

theorem heavier_boxes_weight
  (x y : ℤ)
  (h1 : x ≥ 0)
  (h2 : x ≤ 30)
  (h3 : 10 * x + (30 - x) * y = 540)
  (h4 : 10 * x + (15 - x) * y = 240) :
  y = 20 :=
by
  sorry

end heavier_boxes_weight_l40_40437


namespace least_positive_integer_with_12_factors_l40_40022

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l40_40022


namespace least_positive_integer_with_12_factors_l40_40011

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l40_40011


namespace H_iterated_l40_40872

variable (H : ℝ → ℝ)

-- Conditions as hypotheses
axiom H_2 : H 2 = -4
axiom H_neg4 : H (-4) = 6
axiom H_6 : H 6 = 6

-- The theorem we want to prove
theorem H_iterated (H : ℝ → ℝ) (h1 : H 2 = -4) (h2 : H (-4) = 6) (h3 : H 6 = 6) : 
  H (H (H (H (H 2)))) = 6 := by
  sorry

end H_iterated_l40_40872


namespace solution_set_inequality_l40_40798

theorem solution_set_inequality (x : ℝ) :
  ((x^2 - 4) * (x - 6)^2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 2 ∨ x = 6) :=
  sorry

end solution_set_inequality_l40_40798


namespace austin_tax_l40_40220

theorem austin_tax 
  (number_of_robots : ℕ)
  (cost_per_robot change_left starting_amount : ℚ) 
  (h1 : number_of_robots = 7)
  (h2 : cost_per_robot = 8.75)
  (h3 : change_left = 11.53)
  (h4 : starting_amount = 80) : 
  ∃ tax : ℚ, tax = 7.22 :=
by
  sorry

end austin_tax_l40_40220


namespace largest_N_with_square_in_base_nine_l40_40156

theorem largest_N_with_square_in_base_nine:
  ∃ N: ℕ, (9^2 ≤ N^2 ∧ N^2 < 9^3) ∧ ∀ M: ℕ, (9^2 ≤ M^2 ∧ M^2 < 9^3) → M ≤ N ∧ N = 26 := 
sorry

end largest_N_with_square_in_base_nine_l40_40156


namespace sum_of_terms_in_sequence_is_215_l40_40067

theorem sum_of_terms_in_sequence_is_215 (a d : ℕ) (h1: Nat.Prime a) (h2: Nat.Prime d)
  (hAP : a + 50 = a + 50)
  (hGP : (a + d) * (a + 50) = (a + 2 * d) ^ 2) :
  (a + (a + d) + (a + 2 * d) + (a + 50)) = 215 := sorry

end sum_of_terms_in_sequence_is_215_l40_40067


namespace chocolateBarsPerBox_l40_40835

def numberOfSmallBoxes := 20
def totalChocolateBars := 500

theorem chocolateBarsPerBox : totalChocolateBars / numberOfSmallBoxes = 25 :=
by
  -- Skipping the proof here
  sorry

end chocolateBarsPerBox_l40_40835


namespace tan_alpha_solution_l40_40112

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l40_40112


namespace pyramid_volume_l40_40057

noncomputable def volume_of_pyramid (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let diagonal_length := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2)
  (1 / 3) * base_area * height

theorem pyramid_volume : volume_of_pyramid 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end pyramid_volume_l40_40057


namespace percentage_y_of_x_l40_40429

variable {x y : ℝ}

theorem percentage_y_of_x 
  (h : 0.15 * x = 0.20 * y) : y = 0.75 * x := 
sorry

end percentage_y_of_x_l40_40429


namespace union_P_Q_l40_40874

noncomputable def P : Set ℤ := {x | x^2 - x = 0}
noncomputable def Q : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

theorem union_P_Q : P ∪ Q = {-1, 0, 1} :=
by 
  sorry

end union_P_Q_l40_40874


namespace students_arrangement_l40_40175

theorem students_arrangement :
  ({s | s ⊆ (finset.range 5) ∧ finset.card s = 4}.card *
   ((finset.card {t | t ⊆ (finset.range 5) ∧ finset.card t = 2}) *
   ((finset.card {u | u ⊆ ((finset.range 5) \ finset.image fintype.some s) ∧ finset.card u = 1}) *
   (finset.card {v | v ⊆ ((finset.range 5) \ (finset.image fintype.some s ∪ finset.image fintype.some t)) ∧ finset.card v = 1}))) = 60 :=
sorry

end students_arrangement_l40_40175


namespace circle_circumference_ratio_l40_40631

theorem circle_circumference_ratio (A₁ A₂ : ℝ) (h : A₁ / A₂ = 16 / 25) :
  ∃ C₁ C₂ : ℝ, (C₁ / C₂ = 4 / 5) :=
by
  -- Definitions and calculations to be done here
  sorry

end circle_circumference_ratio_l40_40631


namespace probability_prime_sum_l40_40250

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end probability_prime_sum_l40_40250


namespace rotated_angle_new_measure_l40_40637

theorem rotated_angle_new_measure (initial_angle : ℝ) (rotation : ℝ) (final_angle : ℝ) :
  initial_angle = 60 ∧ rotation = 300 → final_angle = 120 :=
by
  intros h
  sorry

end rotated_angle_new_measure_l40_40637


namespace graph_is_empty_l40_40544

theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + y^2 - 9 * x - 4 * y + 17 ≠ 0 :=
by
  intros x y
  sorry

end graph_is_empty_l40_40544


namespace prime_sum_probability_l40_40238

open Set

noncomputable def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pairs : Finset (ℕ × ℕ) :=
  first_ten_primes.product first_ten_primes.filter (λ p, is_prime (p.1 + p.2))

theorem prime_sum_probability :
  (valid_pairs.card = 6) → ((first_ten_primes.card.choose 2 : ℚ) = 45) →
  Rat.mk valid_pairs.card (first_ten_primes.card.choose 2) = 2 / 15 :=
sorry

end prime_sum_probability_l40_40238


namespace parallelogram_area_increase_l40_40791
open Real

/-- The area of the parallelogram increases by 600 square meters when the base is increased by 20 meters. -/
theorem parallelogram_area_increase :
  ∀ (base height new_base : ℝ), 
    base = 65 → height = 30 → new_base = base + 20 → 
    (new_base * height - base * height) = 600 := 
by
  sorry

end parallelogram_area_increase_l40_40791


namespace expression_evaluation_l40_40567

theorem expression_evaluation (x y z : ℝ) (h : x = y + z) (h' : x = 2) :
  x^3 + 2 * y^3 + 2 * z^3 + 6 * x * y * z = 24 :=
by
  sorry

end expression_evaluation_l40_40567


namespace ab_minus_c_eq_six_l40_40073

theorem ab_minus_c_eq_six (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : 
  a * b - c = 6 := 
by
  sorry

end ab_minus_c_eq_six_l40_40073


namespace sheets_in_total_l40_40186

theorem sheets_in_total (boxes_needed : ℕ) (sheets_per_box : ℕ) (total_sheets : ℕ) 
  (h1 : boxes_needed = 7) (h2 : sheets_per_box = 100) : total_sheets = boxes_needed * sheets_per_box := by
  sorry

end sheets_in_total_l40_40186


namespace sample_freq_0_40_l40_40209

def total_sample_size : ℕ := 100
def freq_group_0_10 : ℕ := 12
def freq_group_10_20 : ℕ := 13
def freq_group_20_30 : ℕ := 24
def freq_group_30_40 : ℕ := 15
def freq_group_40_50 : ℕ := 16
def freq_group_50_60 : ℕ := 13
def freq_group_60_70 : ℕ := 7

theorem sample_freq_0_40 : (freq_group_0_10 + freq_group_10_20 + freq_group_20_30 + freq_group_30_40) / (total_sample_size : ℝ) = 0.64 := by
  sorry

end sample_freq_0_40_l40_40209


namespace fraction_expression_as_common_fraction_l40_40858

theorem fraction_expression_as_common_fraction :
  ((3 / 7 + 5 / 8) / (5 / 12 + 2 / 15)) = (295 / 154) := 
by
  sorry

end fraction_expression_as_common_fraction_l40_40858


namespace ratio_of_josh_to_brad_l40_40327

theorem ratio_of_josh_to_brad (J D B : ℝ) (h1 : J + D + B = 68) (h2 : J = (3 / 4) * D) (h3 : D = 32) :
  (J / B) = 2 :=
by
  sorry

end ratio_of_josh_to_brad_l40_40327


namespace sphere_surface_area_l40_40980

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_surface_area (r_circle r_distance : ℝ) :
  (Real.pi * r_circle^2 = 16 * Real.pi) →
  (r_distance = 3) →
  (surface_area_of_sphere (Real.sqrt (r_distance^2 + r_circle^2)) = 100 * Real.pi) := by
sorry

end sphere_surface_area_l40_40980


namespace temperature_difference_l40_40181

def lowest_temp : ℝ := -15
def highest_temp : ℝ := 3

theorem temperature_difference :
  highest_temp - lowest_temp = 18 :=
by
  sorry

end temperature_difference_l40_40181


namespace tan_A_in_right_triangle_l40_40600

theorem tan_A_in_right_triangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] (angle_A angle_B angle_C : ℝ) 
  (sin_B : ℚ) (tan_A : ℚ) :
  angle_C = 90 ∧ sin_B = 3 / 5 → tan_A = 4 / 3 := by
  sorry

end tan_A_in_right_triangle_l40_40600


namespace sum_of_faces_edges_vertices_l40_40662

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l40_40662


namespace iron_aluminum_weight_difference_l40_40620

theorem iron_aluminum_weight_difference :
  let iron_weight := 11.17
  let aluminum_weight := 0.83
  iron_weight - aluminum_weight = 10.34 :=
by
  sorry

end iron_aluminum_weight_difference_l40_40620


namespace max_rectangles_1x2_l40_40945

-- Define the problem conditions
def single_cell_squares : Type := sorry
def rectangles_1x2 (figure : single_cell_squares) : Prop := sorry

-- State the maximum number theorem
theorem max_rectangles_1x2 (figure : single_cell_squares) (h : rectangles_1x2 figure) :
  ∃ (n : ℕ), n ≤ 5 ∧ ∀ m : ℕ, rectangles_1x2 figure ∧ m ≤ 5 → m = 5 :=
sorry

end max_rectangles_1x2_l40_40945


namespace largest_of_seven_consecutive_l40_40486

theorem largest_of_seven_consecutive (n : ℕ) 
  (h1: n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 3010) :
  n + 6 = 433 :=
by 
  sorry

end largest_of_seven_consecutive_l40_40486


namespace part1_intersection_part1_union_complement_part2_range_of_m_l40_40781

open Set

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (2 * m + 1)}

theorem part1_intersection (x : ℝ) : x ∈ (A ∩ B 4) ↔ 3 ≤ x ∧ x ≤ 5 :=
by sorry

theorem part1_union_complement (x : ℝ) : x ∈ (compl A ∪ B 4) ↔ x < 2 ∨ x ≥ 3 :=
by sorry

theorem part2_range_of_m (m : ℝ) : (∀ x, (x ∈ A ↔ x ∈ B m)) ↔ (2 ≤ m ∧ m ≤ 3) :=
by sorry

end part1_intersection_part1_union_complement_part2_range_of_m_l40_40781


namespace sum_of_faces_edges_vertices_of_rect_prism_l40_40690

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l40_40690


namespace tan_alpha_sqrt_15_over_15_l40_40115

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l40_40115


namespace value_of_f_g_6_squared_l40_40090

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem value_of_f_g_6_squared : (f (g 6))^2 = 26569 :=
by
  -- Place your proof here
  sorry

end value_of_f_g_6_squared_l40_40090


namespace race_distance_l40_40435

theorem race_distance (Va Vb Vc : ℝ) (D : ℝ) :
    (Va / Vb = 10 / 9) →
    (Va / Vc = 80 / 63) →
    (Vb / Vc = 8 / 7) →
    (D - 100) / D = 7 / 8 → 
    D = 700 :=
by
  intros h1 h2 h3 h4 
  sorry

end race_distance_l40_40435


namespace sum_of_faces_edges_vertices_rectangular_prism_l40_40683

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l40_40683


namespace pool_capacity_l40_40036

variable (C : ℕ)

-- Conditions
def rate_first_valve := C / 120
def rate_second_valve := C / 120 + 50
def combined_rate := C / 48

-- Proof statement
theorem pool_capacity (C_pos : 0 < C) (h1 : rate_first_valve C + rate_second_valve C = combined_rate C) : C = 12000 := by
  sorry

end pool_capacity_l40_40036


namespace largest_integer_less_85_with_remainder_3_l40_40287

theorem largest_integer_less_85_with_remainder_3 (n : ℕ) : 
  n < 85 ∧ n % 9 = 3 → n ≤ 84 :=
by
  intro h
  sorry

end largest_integer_less_85_with_remainder_3_l40_40287


namespace range_of_g_l40_40079

noncomputable def g (x : ℝ) : ℝ :=
  (sin x ^ 3 + 8 * sin x ^ 2 + 2 * sin x + 2 * (cos x ^ 2) - 10) / (sin x - 1)

theorem range_of_g :
  ∀ x : ℝ, sin x ≠ 1 → 3 ≤ g x ∧ g x < 15 :=
by
  sorry

end range_of_g_l40_40079


namespace most_precise_value_l40_40773

def D := 3.27645
def error := 0.00518
def D_upper := D + error
def D_lower := D - error
def rounded_D_upper := Float.round (D_upper * 10) / 10
def rounded_D_lower := Float.round (D_lower * 10) / 10

theorem most_precise_value :
  rounded_D_upper = 3.3 ∧ rounded_D_lower = 3.3 → rounded_D_upper = 3.3 :=
by sorry

end most_precise_value_l40_40773


namespace stream_speed_l40_40831

theorem stream_speed (v : ℝ) (boat_speed : ℝ) (distance : ℝ) (time : ℝ) 
    (h1 : boat_speed = 10) 
    (h2 : distance = 54) 
    (h3 : time = 3) 
    (h4 : distance = (boat_speed + v) * time) : 
    v = 8 :=
by
  sorry

end stream_speed_l40_40831


namespace integer_solutions_count_l40_40897

theorem integer_solutions_count (x : ℤ) : 
  (x^2 - 3 * x + 2)^2 - 3 * (x^2 - 3 * x) - 4 = 0 ↔ 0 = 0 :=
by sorry

end integer_solutions_count_l40_40897


namespace simplify_and_evaluate_expression_l40_40176

-- Define the conditions
def a := 2
def b := -1

-- State the theorem
theorem simplify_and_evaluate_expression : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 3 * a * b) / (-b) = -12 := by
  -- Placeholder for the proof
  sorry

end simplify_and_evaluate_expression_l40_40176


namespace odd_expression_proof_l40_40428

theorem odd_expression_proof (n : ℤ) : Odd (n^2 + n + 5) :=
by 
  sorry

end odd_expression_proof_l40_40428


namespace remainder_of_sum_l40_40612

theorem remainder_of_sum (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 :=
by
  sorry

end remainder_of_sum_l40_40612


namespace pirates_total_distance_l40_40947

def adjusted_distance_1 (d: ℝ) : ℝ := d * 1.10
def adjusted_distance_2 (d: ℝ) : ℝ := d * 1.15
def adjusted_distance_3 (d: ℝ) : ℝ := d * 1.20
def adjusted_distance_4 (d: ℝ) : ℝ := d * 1.25

noncomputable def total_distance : ℝ := 
  let first_island := (adjusted_distance_1 10) + (adjusted_distance_1 15) + (adjusted_distance_1 20)
  let second_island := adjusted_distance_2 40
  let third_island := (adjusted_distance_3 25) + (adjusted_distance_3 20) + (adjusted_distance_3 25) + (adjusted_distance_3 20)
  let fourth_island := adjusted_distance_4 35
  first_island + second_island + third_island + fourth_island

theorem pirates_total_distance : total_distance = 247.25 := by
  sorry

end pirates_total_distance_l40_40947


namespace b_investment_calculation_l40_40037

noncomputable def total_profit : ℝ := 9600
noncomputable def A_investment : ℝ := 2000
noncomputable def A_management_fee : ℝ := 0.10 * total_profit
noncomputable def remaining_profit : ℝ := total_profit - A_management_fee
noncomputable def A_total_received : ℝ := 4416
noncomputable def B_investment : ℝ := 1000

theorem b_investment_calculation (B: ℝ) 
  (h_total_profit: total_profit = 9600)
  (h_A_investment: A_investment = 2000)
  (h_A_management_fee: A_management_fee = 0.10 * total_profit)
  (h_remaining_profit: remaining_profit = total_profit - A_management_fee)
  (h_A_total_received: A_total_received = 4416)
  (h_A_total_formula : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit) :
  B = 1000 :=
by
  have h1 : total_profit = 9600 := h_total_profit
  have h2 : A_investment = 2000 := h_A_investment
  have h3 : A_management_fee = 0.10 * total_profit := h_A_management_fee
  have h4 : remaining_profit = total_profit - A_management_fee := h_remaining_profit
  have h5 : A_total_received = 4416 := h_A_total_received
  have h6 : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit := h_A_total_formula
  
  sorry

end b_investment_calculation_l40_40037


namespace ratio_side_length_pentagon_square_l40_40211

noncomputable def perimeter_square (s : ℝ) : ℝ := 4 * s
noncomputable def perimeter_pentagon (p : ℝ) : ℝ := 5 * p

theorem ratio_side_length_pentagon_square (s p : ℝ) 
  (h_square : perimeter_square(s) = 20) 
  (h_pentagon : perimeter_pentagon(p) = 20) :
  p / s = 4 / 5 :=
  sorry

end ratio_side_length_pentagon_square_l40_40211


namespace count_valid_Q_l40_40922

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 5)

def Q_degree (Q : Polynomial ℝ) : Prop :=
  Q.degree = 2

def R_degree (R : Polynomial ℝ) : Prop :=
  R.degree = 3

def P_Q_relation (Q R : Polynomial ℝ) : Prop :=
  ∀ x, P (Q.eval x) = P x * R.eval x

theorem count_valid_Q : 
  (∃ Qs : Finset (Polynomial ℝ), ∀ Q ∈ Qs, Q_degree Q ∧ (∃ R, R_degree R ∧ P_Q_relation Q R) 
    ∧ Qs.card = 22) :=
sorry

end count_valid_Q_l40_40922


namespace pool_capacity_percentage_l40_40351

theorem pool_capacity_percentage :
  let width := 60 
  let length := 150 
  let depth := 10 
  let drain_rate := 60 
  let time := 1200 
  let total_volume := width * length * depth
  let water_removed := drain_rate * time
  let capacity_percentage := (water_removed / total_volume : ℚ) * 100
  capacity_percentage = 80 := by
  sorry

end pool_capacity_percentage_l40_40351


namespace rowing_problem_l40_40051

theorem rowing_problem (R S x y : ℝ) 
  (h1 : R = y + x) 
  (h2 : S = y - x) : 
  x = (R - S) / 2 ∧ y = (R + S) / 2 :=
by
  sorry

end rowing_problem_l40_40051


namespace salmon_trip_l40_40072

theorem salmon_trip (male_female_sum : 712261 + 259378 = 971639) : 
  712261 + 259378 = 971639 := 
by 
  exact male_female_sum

end salmon_trip_l40_40072


namespace letters_with_both_l40_40145

/-
In a certain alphabet, some letters contain a dot and a straight line. 
36 letters contain a straight line but do not contain a dot. 
The alphabet has 60 letters, all of which contain either a dot or a straight line or both. 
There are 4 letters that contain a dot but do not contain a straight line. 
-/
def L_no_D : ℕ := 36
def D_no_L : ℕ := 4
def total_letters : ℕ := 60

theorem letters_with_both (DL : ℕ) : 
  total_letters = D_no_L + L_no_D + DL → 
  DL = 20 :=
by
  intros h
  sorry

end letters_with_both_l40_40145


namespace perfect_square_trinomial_iff_l40_40888

theorem perfect_square_trinomial_iff (m : ℤ) :
  (∃ a b : ℤ, 4 = a^2 ∧ 121 = b^2 ∧ (4 = a^2 ∧ 121 = b^2) ∧ m = 2 * a * b ∨ m = -2 * a * b) ↔ (m = 44 ∨ m = -44) :=
by sorry

end perfect_square_trinomial_iff_l40_40888


namespace least_positive_integer_with_12_factors_l40_40024

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l40_40024


namespace probability_sum_is_prime_l40_40242

-- Define the set of the first ten primes
def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define a function that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  ∃ p, p > 1 ∧ p ≤ n ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Define a function that checks if the sum of two primes is a prime
def is_sum_prime (a b : ℕ) : Prop :=
  is_prime (a + b)

-- Define the problem of calculating the required probability
theorem probability_sum_is_prime :
  let totalPairs := (firstTenPrimes.to_finset.powerset.card) / 2
  let validPairs := (({(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)}).card)
  validPairs / totalPairs = (1 : ℕ) / (9 : ℕ) := by
  sorry

end probability_sum_is_prime_l40_40242


namespace line_b_parallel_or_in_plane_l40_40770

def Line : Type := sorry    -- Placeholder for the type of line
def Plane : Type := sorry   -- Placeholder for the type of plane

def is_parallel (a b : Line) : Prop := sorry             -- Predicate for parallel lines
def is_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry   -- Predicate for a line being parallel to a plane
def lies_in_plane (l : Line) (p : Plane) : Prop := sorry          -- Predicate for a line lying in a plane

theorem line_b_parallel_or_in_plane (a b : Line) (α : Plane) 
  (h1 : is_parallel a b) 
  (h2 : is_parallel_to_plane a α) : 
  is_parallel_to_plane b α ∨ lies_in_plane b α :=
sorry

end line_b_parallel_or_in_plane_l40_40770


namespace product_of_sequence_is_243_l40_40999

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l40_40999


namespace yan_distance_ratio_l40_40817

theorem yan_distance_ratio (w x y : ℝ) (h1 : w > 0) (h2 : x > 0) (h3 : y > 0)
(h4 : y / w = x / w + (x + y) / (5 * w)) : x / y = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l40_40817


namespace claire_speed_l40_40066

def distance := 2067
def time := 39

def speed (d : ℕ) (t : ℕ) : ℕ := d / t

theorem claire_speed : speed distance time = 53 := by
  sorry

end claire_speed_l40_40066


namespace incorrect_conclusion_l40_40427

theorem incorrect_conclusion
  (a b : ℝ) 
  (h₁ : 1/a < 1/b) 
  (h₂ : 1/b < 0) 
  (h₃ : a < 0) 
  (h₄ : b < 0) 
  (h₅ : a > b) : ¬ (|a| + |b| > |a + b|) := 
sorry

end incorrect_conclusion_l40_40427


namespace find_a_2_find_a_n_l40_40608

-- Define the problem conditions and questions as types
def S_3 (a_1 a_2 a_3 : ℝ) : Prop := a_1 + a_2 + a_3 = 7
def arithmetic_mean_condition (a_1 a_2 a_3 : ℝ) : Prop :=
  (a_1 + 3 + a_3 + 4) / 2 = 3 * a_2

-- Prove that a_2 = 2 given the conditions
theorem find_a_2 (a_1 a_2 a_3 : ℝ) (h1 : S_3 a_1 a_2 a_3) (h2: arithmetic_mean_condition a_1 a_2 a_3) :
  a_2 = 2 := 
sorry

-- Define the general term for a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Prove the formula for the general term of the geometric sequence given the conditions and a_2 found
theorem find_a_n (a : ℕ → ℝ) (q : ℝ) (h1 : S_3 (a 1) (a 2) (a 3)) (h2 : arithmetic_mean_condition (a 1) (a 2) (a 3)) (h3 : geometric_sequence a q) : 
  (q = (1/2) → ∀ n, a n = (1 / 2)^(n - 3))
  ∧ (q = 2 → ∀ n, a n = 2^(n - 1)) := 
sorry

end find_a_2_find_a_n_l40_40608


namespace simplify_tan_cot_expr_l40_40931

theorem simplify_tan_cot_expr :
  let tan_45 := 1
  let cot_45 := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 :=
by
  let tan_45 := 1
  let cot_45 := 1
  sorry

end simplify_tan_cot_expr_l40_40931


namespace pieces_after_cuts_l40_40281

theorem pieces_after_cuts (n : ℕ) : 
  (∃ n, (8 * n + 1 = 2009)) ↔ (n = 251) :=
by 
  sorry

end pieces_after_cuts_l40_40281


namespace weekly_earnings_before_rent_l40_40615

theorem weekly_earnings_before_rent (EarningsAfterRent : ℝ) (weeks : ℕ) (rentPerWeek : ℝ) :
  EarningsAfterRent = 93899 → weeks = 233 → rentPerWeek = 49 →
  ((EarningsAfterRent + rentPerWeek * weeks) / weeks) = 451.99 :=
by
  intros H1 H2 H3
  -- convert the assumptions to the required form
  rw [H1, H2, H3]
  -- provide the objective statement
  change ((93899 + 49 * 233) / 233) = 451.99
  -- leave the final proof details as a sorry for now
  sorry

end weekly_earnings_before_rent_l40_40615


namespace ratio_of_ticket_prices_l40_40518

-- Given conditions
def num_adults := 400
def num_children := 200
def adult_ticket_price : ℕ := 32
def total_amount : ℕ := 16000
def child_ticket_price (C : ℕ) : Prop := num_adults * adult_ticket_price + num_children * C = total_amount

theorem ratio_of_ticket_prices (C : ℕ) (hC : child_ticket_price C) :
  adult_ticket_price / C = 2 :=
by
  sorry

end ratio_of_ticket_prices_l40_40518


namespace middle_term_arithmetic_sequence_l40_40086

theorem middle_term_arithmetic_sequence (m : ℝ) (h : 2 * m = 1 + 5) : m = 3 :=
by
  sorry

end middle_term_arithmetic_sequence_l40_40086


namespace find_k_l40_40887

theorem find_k (x k : ℤ) (h : 2 * k - x = 2) (hx : x = -4) : k = -1 :=
by
  rw [hx] at h
  -- Substituting x = -4 into the equation
  sorry  -- Skipping further proof steps

end find_k_l40_40887


namespace least_positive_integer_with_12_factors_l40_40009

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l40_40009


namespace cats_new_total_weight_l40_40732

noncomputable def total_weight (weights : List ℚ) : ℚ :=
  weights.sum

noncomputable def remove_min_max_weight (weights : List ℚ) : ℚ :=
  let min_weight := weights.minimum?.getD 0
  let max_weight := weights.maximum?.getD 0
  weights.sum - min_weight - max_weight

theorem cats_new_total_weight :
  let weights := [3.5, 7.2, 4.8, 6, 5.5, 9, 4]
  remove_min_max_weight weights = 27.5 := by
  sorry

end cats_new_total_weight_l40_40732


namespace line_intersects_ellipse_max_chord_length_l40_40577

theorem line_intersects_ellipse (m : ℝ) :
  (-2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), (9 * x^2 + 6 * m * x + 2 * m^2 - 8 = 0) ∧ (y = (3 / 2) * x + m) ∧ (x^2 / 4 + y^2 / 9 = 1) :=
sorry

theorem max_chord_length (m : ℝ) :
  m = 0 → (∃ (A B : ℝ × ℝ),
  ((A.1^2 / 4 + A.2^2 / 9 = 1) ∧ (A.2 = (3 / 2) * A.1 + m)) ∧
  ((B.1^2 / 4 + B.2^2 / 9 = 1) ∧ (B.2 = (3 / 2) * B.1 + m)) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 26 / 3)) :=
sorry

end line_intersects_ellipse_max_chord_length_l40_40577


namespace neg_universal_to_existential_l40_40482

theorem neg_universal_to_existential :
  (¬ (∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by 
  sorry

end neg_universal_to_existential_l40_40482


namespace point_in_second_quadrant_l40_40776

def isInSecondQuadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : isInSecondQuadrant (-1) 1 :=
by
  sorry

end point_in_second_quadrant_l40_40776


namespace tan_alpha_value_l40_40133

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l40_40133


namespace max_possible_value_of_k_l40_40706

noncomputable def max_knights_saying_less : Nat :=
  let n := 2015
  let k := n - 2
  k

theorem max_possible_value_of_k : max_knights_saying_less = 2013 :=
by
  sorry

end max_possible_value_of_k_l40_40706


namespace interest_rate_first_year_l40_40407

theorem interest_rate_first_year (R : ℚ)
  (principal : ℚ := 7000)
  (final_amount : ℚ := 7644)
  (time_period_first_year : ℚ := 1)
  (time_period_second_year : ℚ := 1)
  (rate_second_year : ℚ := 5) :
  principal + (principal * R * time_period_first_year / 100) + 
  ((principal + (principal * R * time_period_first_year / 100)) * rate_second_year * time_period_second_year / 100) = final_amount →
  R = 4 := 
by {
  sorry
}

end interest_rate_first_year_l40_40407


namespace perpendicular_line_passing_point_l40_40748

theorem perpendicular_line_passing_point (x y : ℝ) (hx : 4 * x - 3 * y + 2 = 0) : 
  ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ (3 * x + 4 * y + 1 = 0) → l 1 2) :=
sorry

end perpendicular_line_passing_point_l40_40748


namespace tan_alpha_value_l40_40139

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l40_40139


namespace product_of_sequence_is_243_l40_40998

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l40_40998


namespace domain_of_function_l40_40477

def function_domain : Set ℝ := { x : ℝ | x + 1 ≥ 0 ∧ 2 - x ≠ 0 }

theorem domain_of_function :
  function_domain = { x : ℝ | x ≥ -1 ∧ x ≠ 2 } :=
sorry

end domain_of_function_l40_40477


namespace sum_faces_edges_vertices_eq_26_l40_40665

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l40_40665


namespace sum_of_faces_edges_vertices_l40_40658

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l40_40658


namespace least_integer_with_twelve_factors_l40_40016

open Classical

theorem least_integer_with_twelve_factors : ∃ k : ℕ, (∀ n : ℕ, (count_factors n = 12 → k ≤ n)) ∧ k = 72 :=
by
  sorry

-- Here count_factors is a function that returns the number of positive divisors of a given integer.

end least_integer_with_twelve_factors_l40_40016


namespace logarithmic_function_through_point_l40_40877

noncomputable def log_function_expression (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem logarithmic_function_through_point (f : ℝ → ℝ) :
  (∀ x a : ℝ, a > 0 ∧ a ≠ 1 → f x = log_function_expression a x) ∧ f 4 = 2 →
  ∃ g : ℝ → ℝ, ∀ x : ℝ, g x = log_function_expression 2 x :=
by {
  sorry
}

end logarithmic_function_through_point_l40_40877


namespace divisible_by_primes_l40_40786

theorem divisible_by_primes (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (100100 * x + 10010 * y + 1001 * z) % 7 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 11 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 13 = 0 := 
by
  sorry

end divisible_by_primes_l40_40786


namespace rectangular_field_length_l40_40481

theorem rectangular_field_length {w l : ℝ} (h1 : l = 2 * w) (h2 : (8 : ℝ) * 8 = 1 / 18 * (l * w)) : l = 48 :=
by sorry

end rectangular_field_length_l40_40481


namespace tan_alpha_value_l40_40129

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l40_40129


namespace max_band_members_l40_40829

theorem max_band_members (r x m : ℕ) (h1 : m < 150) (h2 : r * x + 3 = m) (h3 : (r - 3) * (x + 2) = m) : m = 147 := by
  sorry

end max_band_members_l40_40829


namespace solve_for_y_l40_40471

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l40_40471


namespace train_speed_l40_40061

theorem train_speed :
  ∀ (length : ℝ) (time : ℝ),
    length = 135 ∧ time = 3.4711508793582233 →
    (length / time) * 3.6 = 140.0004 :=
by
  sorry

end train_speed_l40_40061


namespace prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l40_40856

-- Definitions
def classes := 4
def students := 4
def total_distributions := classes ^ students

-- Problem 1
theorem prob_each_class_receives_one : 
  (A_4 ^ 4) / total_distributions = 3 / 32 := sorry

-- Problem 2
theorem prob_at_least_one_class_empty : 
  1 - (A_4 ^ 4) / total_distributions = 29 / 32 := sorry

-- Problem 3
theorem prob_exactly_one_class_empty :
  (C_4 ^ 1 * C_4 ^ 2 * C_3 ^ 1 * C_2 ^ 1) / total_distributions = 9 / 16 := sorry

end prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l40_40856


namespace part1_part2_part3_l40_40357

variable (a b c : ℝ) (f : ℝ → ℝ)
-- Defining the polynomial function f
def polynomial (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem part1 (h0 : polynomial a b 6 0 = 6) : c = 6 :=
by sorry

theorem part2 (h1 : polynomial a b (-2) 0 = -2) (h2 : polynomial a b (-2) 1 = 5) : polynomial a b (-2) (-1) = -9 :=
by sorry

theorem part3 (h3 : polynomial a b 3 5 + polynomial a b 3 (-5) = 6) (h4 : polynomial a b 3 2 = 8) : polynomial a b 3 (-2) = -2 :=
by sorry

end part1_part2_part3_l40_40357


namespace min_value_of_m_l40_40579

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_value_of_m {m : ℝ} (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ m) : m = 5 := 
sorry

end min_value_of_m_l40_40579


namespace students_play_both_sports_l40_40968

theorem students_play_both_sports 
  (total_students : ℕ) (students_play_football : ℕ) 
  (students_play_cricket : ℕ) (students_play_neither : ℕ) :
  total_students = 470 → students_play_football = 325 → 
  students_play_cricket = 175 → students_play_neither = 50 → 
  (students_play_football + students_play_cricket - 
    (total_students - students_play_neither)) = 80 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end students_play_both_sports_l40_40968


namespace investment_difference_l40_40447

noncomputable def future_value_semi_annual (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 2)^((years * 2))

noncomputable def future_value_monthly (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + annual_rate / 12)^((years * 12))

theorem investment_difference :
  let jose_investment := future_value_semi_annual 30000 0.03 3
  let patricia_investment := future_value_monthly 30000 0.025 3
  round (jose_investment) - round (patricia_investment) = 317 :=
by
  sorry

end investment_difference_l40_40447


namespace prime_pair_probability_l40_40233

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrimeSum (a b : ℕ) : Prop := Nat.Prime (a + b)

theorem prime_pair_probability :
  let pairs := List.sublistsLen 2 firstTenPrimes
  let primePairs := pairs.filter (λ p, match p with
                                       | [a, b] => isPrimeSum a b
                                       | _ => false
                                       end)
  (primePairs.length : ℚ) / (pairs.length : ℚ) = 1 / 9 :=
by
  sorry

end prime_pair_probability_l40_40233


namespace difference_of_fractions_l40_40702

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h1 : a = 700) (h2 : b = 7) : a - b = 693 :=
by
  rw [h1, h2]
  norm_num

end difference_of_fractions_l40_40702


namespace cell_phone_bill_l40_40047

-- Definitions
def base_cost : ℝ := 20
def cost_per_text : ℝ := 0.05
def cost_per_extra_minute : ℝ := 0.10
def texts_sent : ℕ := 100
def hours_talked : ℝ := 30.5
def included_hours : ℝ := 30

-- Calculate extra minutes used
def extra_minutes : ℝ := (hours_talked - included_hours) * 60

-- Total cost calculation
def total_cost : ℝ := 
  base_cost + 
  (texts_sent * cost_per_text) + 
  (extra_minutes * cost_per_extra_minute)

-- Proof problem statement
theorem cell_phone_bill : total_cost = 28 := by
  sorry

end cell_phone_bill_l40_40047


namespace sum_set_15_l40_40885

noncomputable def sum_nth_set (n : ℕ) : ℕ :=
  let first_element := 1 + (n - 1) * n / 2
  let last_element := first_element + n - 1
  n * (first_element + last_element) / 2

theorem sum_set_15 : sum_nth_set 15 = 1695 :=
  by sorry

end sum_set_15_l40_40885


namespace solve_for_y_l40_40467

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l40_40467


namespace laps_remaining_eq_five_l40_40215

variable (total_distance : ℕ)
variable (distance_per_lap : ℕ)
variable (laps_already_run : ℕ)

theorem laps_remaining_eq_five 
  (h1 : total_distance = 99) 
  (h2 : distance_per_lap = 9) 
  (h3 : laps_already_run = 6) : 
  (total_distance / distance_per_lap - laps_already_run = 5) :=
by 
  sorry

end laps_remaining_eq_five_l40_40215


namespace vehicle_count_expression_l40_40596

variable (C B M : ℕ)

-- Given conditions
axiom wheel_count : 4 * C + 2 * B + 2 * M = 196
axiom bike_to_motorcycle : B = 2 * M

-- Prove that the number of cars can be expressed in terms of the number of motorcycles
theorem vehicle_count_expression : C = (98 - 3 * M) / 2 :=
by
  sorry

end vehicle_count_expression_l40_40596


namespace num_primes_with_squares_in_range_l40_40886

/-- There are exactly 6 prime numbers whose squares are between 2500 and 5500. -/
theorem num_primes_with_squares_in_range : 
  ∃ primes : Finset ℕ, 
    (∀ p ∈ primes, Prime p) ∧
    (∀ p ∈ primes, 2500 < p^2 ∧ p^2 < 5500) ∧
    primes.card = 6 :=
by
  sorry

end num_primes_with_squares_in_range_l40_40886


namespace power_equivalence_l40_40693

theorem power_equivalence (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 :=
by sorry

end power_equivalence_l40_40693


namespace parakeets_per_cage_is_2_l40_40722

variables (cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

def number_of_parakeets_each_cage : ℕ :=
  (total_birds - cages * parrots_per_cage) / cages

theorem parakeets_per_cage_is_2
  (hcages : cages = 4)
  (hparrots_per_cage : parrots_per_cage = 8)
  (htotal_birds : total_birds = 40) :
  number_of_parakeets_each_cage cages parrots_per_cage total_birds = 2 := 
by
  sorry

end parakeets_per_cage_is_2_l40_40722


namespace determine_quadrant_l40_40602

def pointInWhichQuadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On axis or origin"

theorem determine_quadrant : pointInWhichQuadrant (-7) 3 = "Second quadrant" :=
by
  sorry

end determine_quadrant_l40_40602


namespace find_inlet_rate_l40_40727

-- definitions for the given conditions
def volume_cubic_feet : ℝ := 20
def conversion_factor : ℝ := 12^3
def volume_cubic_inches : ℝ := volume_cubic_feet * conversion_factor

def outlet_rate1 : ℝ := 9
def outlet_rate2 : ℝ := 8
def empty_time : ℕ := 2880

-- theorem that captures the proof problem
theorem find_inlet_rate (volume_cubic_inches : ℝ) (outlet_rate1 outlet_rate2 empty_time : ℝ) :
  ∃ (inlet_rate : ℝ), volume_cubic_inches = (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time ↔ inlet_rate = 5 := 
by
  sorry

end find_inlet_rate_l40_40727


namespace students_with_all_three_pets_l40_40901

variable (x y z : ℕ)
variable (total_students : ℕ := 40)
variable (dog_students : ℕ := total_students * 5 / 8)
variable (cat_students : ℕ := total_students * 1 / 4)
variable (other_students : ℕ := 8)
variable (no_pet_students : ℕ := 6)
variable (only_dog_students : ℕ := 12)
variable (only_other_students : ℕ := 3)
variable (cat_other_no_dog_students : ℕ := 10)

theorem students_with_all_three_pets :
  (x + y + z + 10 + 3 + 12 = total_students - no_pet_students) →
  (x + z + 10 = dog_students) →
  (10 + z = cat_students) →
  (y + z + 10 = other_students) →
  z = 0 :=
by
  -- Provide proof here
  sorry

end students_with_all_three_pets_l40_40901


namespace measure_45_minutes_l40_40379

-- Definitions of the conditions
structure Conditions where
  lighter : Prop
  strings : ℕ
  burn_time : ℕ → ℕ
  non_uniform_burn : Prop

-- We can now state the problem in Lean
theorem measure_45_minutes (c : Conditions) (h1 : c.lighter) (h2 : c.strings = 2)
  (h3 : ∀ s, s < 2 → c.burn_time s = 60) (h4 : c.non_uniform_burn) :
  ∃ t, t = 45 := 
sorry

end measure_45_minutes_l40_40379


namespace john_taller_than_lena_l40_40917

-- Define the heights of John, Lena, and Rebeca.
variables (J L R : ℕ)

-- Given conditions:
-- 1. John has a height of 152 cm
axiom john_height : J = 152

-- 2. John is 6 cm shorter than Rebeca
axiom john_shorter_rebeca : J = R - 6

-- 3. The height of Lena and Rebeca together is 295 cm
axiom lena_rebeca_together : L + R = 295

-- Prove that John is 15 cm taller than Lena
theorem john_taller_than_lena : (J - L) = 15 := by
  sorry

end john_taller_than_lena_l40_40917


namespace operation_value_l40_40638

variable (a b : ℤ)

theorem operation_value (h : (21 - 1) * (9 - 1) = 160) : a = 21 :=
by
  sorry

end operation_value_l40_40638


namespace work_completion_l40_40966

theorem work_completion (A B C : ℝ) (h₁ : A + B = 1 / 18) (h₂ : B + C = 1 / 24) (h₃ : A + C = 1 / 36) : 
  1 / (A + B + C) = 16 := 
by
  sorry

end work_completion_l40_40966


namespace smallest_two_digit_multiple_of_17_l40_40511

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ n % 17 = 0 ∧ ∀ m, (10 ≤ m ∧ m < n ∧ m % 17 = 0) → false := sorry

end smallest_two_digit_multiple_of_17_l40_40511


namespace go_stones_perimeter_count_l40_40628

def stones_per_side : ℕ := 6
def sides_of_square : ℕ := 4
def corner_stones : ℕ := 4

theorem go_stones_perimeter_count :
  (stones_per_side * sides_of_square) - corner_stones = 20 := 
by
  sorry

end go_stones_perimeter_count_l40_40628


namespace estimate_less_Exact_l40_40857

variables (a b c d : ℕ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

def round_up (x : ℕ) : ℕ := x + 1
def round_down (x : ℕ) : ℕ := x - 1

theorem estimate_less_Exact
  (h₁ : round_down a = a - 1)
  (h₂ : round_down b = b - 1)
  (h₃ : round_down c = c - 1)
  (h₄ : round_up d = d + 1) :
  (round_down a + round_down b) / round_down c - round_up d < 
  (a + b) / c - d :=
sorry

end estimate_less_Exact_l40_40857


namespace tangent_line_l40_40520

variable (a b x₀ y₀ x y : ℝ)
variable (h_ab : a > b)
variable (h_b0 : b > 0)

def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem tangent_line (h_el : ellipse a b x₀ y₀) : 
  (x₀ * x / a^2) + (y₀ * y / b^2) = 1 :=
sorry

end tangent_line_l40_40520


namespace largest_consecutive_sum_55_l40_40373

theorem largest_consecutive_sum_55 : 
  ∃ k n : ℕ, k > 0 ∧ (∃ n : ℕ, 55 = k * n + (k * (k - 1)) / 2 ∧ k = 5) :=
sorry

end largest_consecutive_sum_55_l40_40373


namespace infinite_quadruples_inequality_quadruple_l40_40797

theorem infinite_quadruples 
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  ∃ (a p q r : ℕ), 
    1 < p ∧ 1 < q ∧ 1 < r ∧
    p ∣ (a * q * r + 1) ∧
    q ∣ (a * p * r + 1) ∧
    r ∣ (a * p * q + 1) :=
sorry

theorem inequality_quadruple
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end infinite_quadruples_inequality_quadruple_l40_40797


namespace parabola_fixed_point_l40_40924

theorem parabola_fixed_point (t : ℝ) : ∃ y, y = 4 * 3^2 + 2 * t * 3 - 3 * t ∧ y = 36 :=
by
  exists 36
  sorry

end parabola_fixed_point_l40_40924


namespace developed_surface_condition_l40_40780

noncomputable def is_developable (F : Surface) : Prop :=
  ∀ (p : Point F), GaussianCurvature F p = 0

theorem developed_surface_condition (F : Surface)
  (simple_cover_geodesics : ∀ (p : Point F), ∃ (G₁ G₂ : GeodesicSystem F), 
     (G₁.lines ∩ G₂.lines ≠ ∅) ∧ ∀ (g₁ ∈ G₁.lines) (g₂ ∈ G₂.lines), 
       angle_between g₁ g₂ = constant_angle) :
  is_developable F :=
begin
  -- proof is not required
  sorry
end

end developed_surface_condition_l40_40780


namespace calculate_c_from_law_of_cosines_l40_40605

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : Prop :=
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B

theorem calculate_c_from_law_of_cosines 
  (a b c : ℝ) (B : ℝ)
  (ha : a = 8) (hb : b = 7) (hB : B = Real.pi / 3) : 
  (c = 3) ∨ (c = 5) :=
sorry

end calculate_c_from_law_of_cosines_l40_40605


namespace integer_values_of_b_for_polynomial_root_l40_40077

theorem integer_values_of_b_for_polynomial_root
    (b : ℤ) :
    (∃ x : ℤ, x^3 + 6 * x^2 + b * x + 12 = 0) ↔
    b = -217 ∨ b = -74 ∨ b = -43 ∨ b = -31 ∨ b = -22 ∨ b = -19 ∨
    b = 19 ∨ b = 22 ∨ b = 31 ∨ b = 43 ∨ b = 74 ∨ b = 217 :=
    sorry

end integer_values_of_b_for_polynomial_root_l40_40077


namespace sum_of_faces_edges_vertices_of_rect_prism_l40_40692

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l40_40692


namespace luke_money_last_weeks_l40_40826

theorem luke_money_last_weeks (earnings_mowing : ℕ) (earnings_weed_eating : ℕ) (weekly_spending : ℕ) 
  (h1 : earnings_mowing = 9) (h2 : earnings_weed_eating = 18) (h3 : weekly_spending = 3) :
  (earnings_mowing + earnings_weed_eating) / weekly_spending = 9 :=
by sorry

end luke_money_last_weeks_l40_40826


namespace prime_probability_l40_40247

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def pairs : List (ℕ × ℕ) :=
(primes.erase 2).map (λ p => (2, p))

def prime_sums : List (ℕ × ℕ) :=
pairs.filter (λ ⟨a, b⟩ => is_prime (a + b))

theorem prime_probability :
  let total_pairs := Nat.choose 10 2
  let valid_pairs := prime_sums.length
  (valid_pairs : ℚ) / (total_pairs : ℚ) = 1 / 9 :=
by
  sorry

end prime_probability_l40_40247


namespace rectangular_to_cylindrical_l40_40401

theorem rectangular_to_cylindrical (x y z r θ : ℝ) (h₁ : x = 3) (h₂ : y = -3 * Real.sqrt 3) (h₃ : z = 2)
  (h₄ : r = Real.sqrt (3^2 + (-3 * Real.sqrt 3)^2)) 
  (h₅ : r > 0) 
  (h₆ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₇ : θ = Float.pi * 5 / 3) : 
  (r, θ, z) = (6, 5 * Float.pi / 3, 2) :=
sorry

end rectangular_to_cylindrical_l40_40401


namespace correct_option_l40_40816

def condition_A (a : ℝ) : Prop := a^3 * a^4 = a^12
def condition_B (a b : ℝ) : Prop := (-3 * a * b^3)^2 = -6 * a * b^6
def condition_C (a : ℝ) : Prop := (a - 3)^2 = a^2 - 9
def condition_D (x y : ℝ) : Prop := (-x + y) * (x + y) = y^2 - x^2

theorem correct_option (x y : ℝ) : condition_D x y := by
  sorry

end correct_option_l40_40816


namespace sample_and_probability_l40_40648

-- (I) Calculation part
def total_parents (parents1 parents2 parents3 : ℕ) := parents1 + parents2 + parents3

def sample_ratio (total_samples : ℕ) (total_individuals : ℕ) := (total_samples : ℝ) / (total_individuals : ℝ)

def number_sampled (num_parents : ℕ) (ratio : ℝ) := (num_parents : ℝ) * ratio

-- (II) Probability part
def total_combinations (n k : ℕ) := Nat.choose n k

def favorable_combinations := [
  ("A_1", "C_1"), ("A_1", "C_2"), ("A_2", "C_1"), ("A_2", "C_2"),
  ("A_3", "C_1"), ("A_3", "C_2"), ("B_1", "C_1"), ("B_1", "C_2"),
  ("C_1", "C_2")
]

def probability (favorable total : ℕ) := (favorable : ℝ) / (total : ℝ)

theorem sample_and_probability :
  let parents1 := 54;
  let parents2 := 18;
  let parents3 := 36;
  let totalSamples := 6;
  let totalParents := total_parents parents1 parents2 parents3;
  let ratio := sample_ratio totalSamples totalParents;
  let sampled1 := number_sampled parents1 ratio;
  let sampled2 := number_sampled parents2 ratio;
  let sampled3 := number_sampled parents3 ratio;
  let totalComb := total_combinations 6 2;
  let favComb := favorable_combinations.length;
  let prob := probability favComb totalComb 
  in sampled1 = 3 ∧ sampled2 = 1 ∧ sampled3 = 2 ∧ prob = 3 / 5 :=
by
  sorry

end sample_and_probability_l40_40648


namespace prime_sum_probability_l40_40237

open Set

noncomputable def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pairs : Finset (ℕ × ℕ) :=
  first_ten_primes.product first_ten_primes.filter (λ p, is_prime (p.1 + p.2))

theorem prime_sum_probability :
  (valid_pairs.card = 6) → ((first_ten_primes.card.choose 2 : ℚ) = 45) →
  Rat.mk valid_pairs.card (first_ten_primes.card.choose 2) = 2 / 15 :=
sorry

end prime_sum_probability_l40_40237


namespace least_positive_integer_with_12_factors_l40_40000

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l40_40000


namespace solve_for_y_l40_40474

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l40_40474


namespace gold_bars_distribution_l40_40939

theorem gold_bars_distribution 
  (initial_gold : ℕ) 
  (lost_gold : ℕ) 
  (num_friends : ℕ) 
  (remaining_gold : ℕ)
  (each_friend_gets : ℕ) :
  initial_gold = 100 →
  lost_gold = 20 →
  num_friends = 4 →
  remaining_gold = initial_gold - lost_gold →
  each_friend_gets = remaining_gold / num_friends →
  each_friend_gets = 20 :=
by
  intros
  sorry

end gold_bars_distribution_l40_40939


namespace avg_growth_rate_equation_l40_40394

/-- This theorem formalizes the problem of finding the equation for the average growth rate of working hours.
    Given that the average working hours in the first week are 40 hours and in the third week are 48.4 hours,
    we need to show that the equation for the growth rate \(x\) satisfies \( 40(1 + x)^2 = 48.4 \). -/
theorem avg_growth_rate_equation (x : ℝ) (first_week_hours third_week_hours : ℝ) 
  (h1: first_week_hours = 40) (h2: third_week_hours = 48.4) :
  40 * (1 + x) ^ 2 = 48.4 :=
sorry

end avg_growth_rate_equation_l40_40394


namespace point_B_position_l40_40878

/-- Given points A and B on the same number line, with A at -2 and B 5 units away from A, prove 
    that B can be either -7 or 3. -/
theorem point_B_position (A B : ℤ) (hA : A = -2) (hB : (B = A + 5) ∨ (B = A - 5)) : 
  B = 3 ∨ B = -7 :=
sorry

end point_B_position_l40_40878


namespace largest_consecutive_sum_55_l40_40376

theorem largest_consecutive_sum_55 :
  ∃ n a : ℕ, (n * (a + (n - 1) / 2) = 55) ∧ (n = 10) ∧ (∀ m : ℕ, ∀ b : ℕ, (m * (b + (m - 1) / 2) = 55) → (m ≤ 10)) :=
by 
  sorry

end largest_consecutive_sum_55_l40_40376


namespace total_sales_correct_l40_40920

def maries_newspapers : ℝ := 275.0
def maries_magazines : ℝ := 150.0
def total_sales := maries_newspapers + maries_magazines

theorem total_sales_correct :
  total_sales = 425.0 :=
by
  -- Proof omitted
  sorry

end total_sales_correct_l40_40920


namespace total_wheels_in_garage_l40_40491

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end total_wheels_in_garage_l40_40491


namespace orthocenter_lines_intersect_l40_40321

theorem orthocenter_lines_intersect 
  (A B C D H_a H_b H_c H_d : Point)
  (h_orthocenters_a : is_orthocenter H_a B C D)
  (h_orthocenters_b : is_orthocenter H_b C D A)
  (h_orthocenters_c : is_orthocenter H_c A B D)
  (h_orthocenters_d : is_orthocenter H_d A B C)
  (h_conditions : A B^2 + C D^2 = A C^2 + B D^2 ∧ 
                  A C^2 + B D^2 = A D^2 + B C^2) :
  ∃ P : Point, lies_on_line A H_a P ∧ lies_on_line B H_b P ∧ lies_on_line C H_c P ∧ lies_on_line D H_d P :=
sorry

end orthocenter_lines_intersect_l40_40321


namespace intersection_of_A_and_B_l40_40765

namespace IntersectionProblem

def setA : Set ℝ := {0, 1, 2}
def setB : Set ℝ := {x | x^2 - x ≤ 0}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := sorry

end IntersectionProblem

end intersection_of_A_and_B_l40_40765


namespace Q_after_move_up_4_units_l40_40476

-- Define the initial coordinates.
def Q_initial : (ℤ × ℤ) := (-4, -6)

-- Define the transformation - moving up 4 units.
def move_up (P : ℤ × ℤ) (units : ℤ) : (ℤ × ℤ) := (P.1, P.2 + units)

-- State the theorem to be proved.
theorem Q_after_move_up_4_units : move_up Q_initial 4 = (-4, -2) :=
by 
  sorry

end Q_after_move_up_4_units_l40_40476


namespace sum_of_digits_625_base5_l40_40809

def sum_of_digits_base_5 (n : ℕ) : ℕ :=
  let rec sum_digits n :=
    if n = 0 then 0
    else (n % 5) + sum_digits (n / 5)
  sum_digits n

theorem sum_of_digits_625_base5 : sum_of_digits_base_5 625 = 5 := by
  sorry

end sum_of_digits_625_base5_l40_40809


namespace lcm_48_180_l40_40561

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l40_40561


namespace equal_share_payment_l40_40921

theorem equal_share_payment (A B C : ℝ) (h : A < B) (h2 : B < C) :
  (B + C + (A + C - 2 * B) / 3) + (A + C - 2 * B / 3) = 2 * C - A - B / 3 :=
sorry

end equal_share_payment_l40_40921


namespace ratio_of_sugar_to_flour_l40_40320

theorem ratio_of_sugar_to_flour
  (F B : ℕ)
  (h1 : F = 10 * B)
  (h2 : F = 8 * (B + 60))
  (sugar : ℕ)
  (hs : sugar = 2000) :
  sugar / F = 5 / 6 :=
by {
  sorry -- proof omitted
}

end ratio_of_sugar_to_flour_l40_40320


namespace xy_equals_nine_l40_40307

theorem xy_equals_nine (x y : ℝ) (h : x * (x + 2 * y) = x ^ 2 + 18) : x * y = 9 :=
by
  sorry

end xy_equals_nine_l40_40307


namespace man_late_minutes_l40_40652

theorem man_late_minutes (v t t' : ℝ) (hv : v' = 3 / 4 * v) (ht : t = 2) (ht' : t' = 4 / 3 * t) :
  t' * 60 - t * 60 = 40 :=
by
  sorry

end man_late_minutes_l40_40652


namespace amount_of_benzene_l40_40283

-- Definitions of the chemical entities involved
def Benzene := Type
def Methane := Type
def Toluene := Type
def Hydrogen := Type

-- The balanced chemical equation as a condition
axiom balanced_equation : ∀ (C6H6 CH4 C7H8 H2 : ℕ), C6H6 + CH4 = C7H8 + H2

-- The proof problem: Prove the amount of Benzene required
theorem amount_of_benzene (moles_methane : ℕ) (moles_toluene : ℕ) (moles_hydrogen : ℕ) :
  moles_methane = 2 → moles_toluene = 2 → moles_hydrogen = 2 → 
  ∃ moles_benzene : ℕ, moles_benzene = 2 := by
  sorry

end amount_of_benzene_l40_40283


namespace passengers_in_7_buses_l40_40828

theorem passengers_in_7_buses (passengers_total buses_total_given buses_required : ℕ) 
    (h1 : passengers_total = 456) 
    (h2 : buses_total_given = 12) 
    (h3 : buses_required = 7) :
    (passengers_total / buses_total_given) * buses_required = 266 := 
sorry

end passengers_in_7_buses_l40_40828


namespace find_some_value_l40_40825

theorem find_some_value (m n : ℝ) (some_value : ℝ) (p : ℝ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + some_value) / 6 - 2 / 5)
  (h3 : p = 3)
  : some_value = -12 / 5 :=
by
  sorry

end find_some_value_l40_40825


namespace solve_quadratic_l40_40349

theorem solve_quadratic : 
  (∀ x : ℚ, 2 * x^2 - x - 6 = 0 → x = -3 / 2 ∨ x = 2) ∧ 
  (∀ y : ℚ, (y - 2)^2 = 9 * y^2 → y = -1 ∨ y = 1 / 2) := 
by
  sorry

end solve_quadratic_l40_40349


namespace max_distance_proof_l40_40842

def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline_gallons : ℝ := 21
def maximum_distance : ℝ := highway_mpg * gasoline_gallons

theorem max_distance_proof : maximum_distance = 256.2 := by
  sorry

end max_distance_proof_l40_40842


namespace find_a_l40_40601

-- Define the lines as given
def line1 (x y : ℝ) := 2 * x + y - 5 = 0
def line2 (x y : ℝ) := x - y - 1 = 0
def line3 (a x y : ℝ) := a * x + y - 3 = 0

-- Define the condition that they intersect at a single point
def lines_intersect_at_point (x y a : ℝ) := line1 x y ∧ line2 x y ∧ line3 a x y

-- To prove: If lines intersect at a certain point, then a = 1
theorem find_a (a : ℝ) : (∃ x y, lines_intersect_at_point x y a) → a = 1 :=
by
  sorry

end find_a_l40_40601


namespace shortest_path_l40_40405

noncomputable def diameter : ℝ := 18
noncomputable def radius : ℝ := diameter / 2
noncomputable def AC : ℝ := 7
noncomputable def BD : ℝ := 7
noncomputable def CD : ℝ := diameter - AC - BD
noncomputable def CP : ℝ := Real.sqrt (radius ^ 2 - (CD / 2) ^ 2)
noncomputable def DP : ℝ := CP

theorem shortest_path (C P D : ℝ) :
  (C - 7) ^ 2 + (D - 7) ^ 2 = CD ^ 2 →
  (C = AC) ∧ (D = BD) →
  2 * CP = 2 * Real.sqrt 77 :=
by
  intros h1 h2
  sorry

end shortest_path_l40_40405


namespace optimal_messenger_strategy_l40_40787

theorem optimal_messenger_strategy (p : ℝ) (hp : 0 < p ∧ p < 1) :
  (p < 1/3 → ∃ n : ℕ, n = 4 ∧ ∀ (k : ℕ), k = 10) ∧ 
  (1/3 ≤ p → ∃ n : ℕ, n = 2 ∧ ∀ (m : ℕ), m = 20) :=
by
  sorry

end optimal_messenger_strategy_l40_40787


namespace max_k_l40_40707

-- Definitions of knight and liar predicates
def is_knight (p : ℕ → Prop) := ∀ i, p i
def is_liar (p : ℕ → Prop) := ∀ i, ¬ p i

-- Definitions for the conditions in the problem
def greater_than_neighbors (n : ℕ) (cards : ℕ → ℕ) :=
  ∀ i : ℕ, 0 < i ∧ i < n - 1 → cards i > cards (i - 1) ∧ cards i > cards (i + 1)

def less_than_neighbors (n : ℕ) (cards : ℕ → ℕ) :=
  ∀ i : ℕ, 0 < i ∧ i < n - 1 → cards i < cards (i - 1) ∧ cards i < cards (i + 1)

def maximum_possible_k (n : ℕ) (cards : ℕ → ℕ) (k : ℕ) :=
  ∀ n : 2015, ∃ k : ℕ, (greater_than_neighbors n cards → is_knight (less_than_neighbors n cards)) ∧
  k = 2013

-- The main theorem statement
theorem max_k (n : ℕ) (cards : ℕ → ℕ) : maximum_possible_k 2015 cards 2013 := sorry

end max_k_l40_40707


namespace tan_alpha_value_l40_40136

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l40_40136


namespace find_other_asymptote_l40_40930

-- Define the conditions
def one_asymptote (x : ℝ) : ℝ := 3 * x
def foci_x_coordinate : ℝ := 5

-- Define the expected answer
def other_asymptote (x : ℝ) : ℝ := -3 * x + 30

-- Theorem statement to prove the equation of the other asymptote
theorem find_other_asymptote :
  (∀ x, y = one_asymptote x) →
  (∀ _x, _x = foci_x_coordinate) →
  (∀ x, y = other_asymptote x) :=
by
  intros h_one_asymptote h_foci_x
  sorry

end find_other_asymptote_l40_40930


namespace tan_alpha_value_l40_40134

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l40_40134


namespace sufficient_not_necessary_condition_l40_40409

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 2) : a > 4 → x^2 - a ≤ 0 :=
by
  sorry

end sufficient_not_necessary_condition_l40_40409


namespace simplify_trig_expr_l40_40933

   theorem simplify_trig_expr :
     (tan (real.pi / 4))^3 + (cot (real.pi / 4))^3 / (tan (real.pi / 4) + cot (real.pi / 4)) = 1 :=
   by
     have h1 : tan (real.pi / 4) = 1 := by sorry
     have h2 : cot (real.pi / 4) = 1 := by sorry
     calc
     (tan (real.pi / 4))^3 + (cot (real.pi / 4))^3 / (tan (real.pi / 4) + cot (real.pi / 4))
         = (1)^3 + (1)^3 / (1 + 1) : by rw [h1, h2]
     ... = 1 : by norm_num
   
end simplify_trig_expr_l40_40933


namespace lcm_48_180_eq_720_l40_40556

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l40_40556


namespace const_if_ineq_forall_l40_40153

theorem const_if_ineq_forall {f : ℝ → ℝ}
    (H : ∀ x y : ℝ, (f x - f y)^2 ≤ |x - y|^3) :
    ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end const_if_ineq_forall_l40_40153


namespace probability_sum_two_primes_is_prime_l40_40263

open Finset

-- First 10 prime numbers
def firstTenPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Function to sum of two primes
def sum_of_two_primes_is_prime (a b : ℕ) : Prop := is_prime (a + b)

-- Function to compute the probability
def probability_two_prime_sum_is_prime : ℚ :=
  let pairs := firstTenPrimes.pairs
  let prime_pairs := pairs.filter (λ pair, sum_of_two_primes_is_prime pair.1 pair.2)
  (prime_pairs.card : ℚ) / pairs.card

theorem probability_sum_two_primes_is_prime :
  probability_two_prime_sum_is_prime = 1 / 9 :=
by sorry

end probability_sum_two_primes_is_prime_l40_40263


namespace tan_alpha_proof_l40_40107

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l40_40107


namespace stream_speed_l40_40824

theorem stream_speed (u v : ℝ) (h1 : 27 = 9 * (u - v)) (h2 : 81 = 9 * (u + v)) : v = 3 :=
by
  sorry

end stream_speed_l40_40824


namespace integer_sum_19_l40_40626

variable (p q r s : ℤ)

theorem integer_sum_19 (h1 : p - q + r = 4) 
                       (h2 : q - r + s = 5) 
                       (h3 : r - s + p = 7) 
                       (h4 : s - p + q = 3) :
                       p + q + r + s = 19 :=
by
  sorry

end integer_sum_19_l40_40626


namespace sequence_formula_l40_40183

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_sum: ∀ n : ℕ, n ≥ 2 → S n = n^2 * a n)
  (h_a1 : a 1 = 1) : ∀ n : ℕ, n ≥ 2 → a n = 2 / (n * (n + 1)) :=
by {
  sorry
}

end sequence_formula_l40_40183


namespace perpendiculars_intersect_at_symmetric_point_l40_40714

-- Conditions
variables {A B C A1 A2 B1 B2 C1 C2 O M : Point}
variable Circle : Circle
variable ABC : Triangle

-- Assume the circle intersects sides at given points
variable ha1a2 : Circle.IntersectsLine (Segment BC) A1 A2
variable hb1b2 : Circle.IntersectsLine (Segment CA) B1 B2
variable hc1c2 : Circle.IntersectsLine (Segment AB) C1 C2

-- Assume perpendiculars drawn through the points intersect at M
variable h_perpendiculars_intersect_at_M : 
  (Perpendicular (LineThrough A1 BC) (LineThrough M SideBC)) ∧ 
  (Perpendicular (LineThrough B1 CA) (LineThrough M SideCA)) ∧ 
  (Perpendicular (LineThrough C1 AB) (LineThrough M SideAB))

-- Symmetry requirements
variable h_symmetry :
  SymmetricWithRespectTo O A1 A2 ∧ 
  SymmetricWithRespectTo O B1 B2 ∧ 
  SymmetricWithRespectTo O C1 C2

-- Goal: Prove the perpendiculars through A2, B2, C2 intersect at a point symmetric to M with respect to O
theorem perpendiculars_intersect_at_symmetric_point :
  ∃ M', 
    SymmetricWithRespectTo O M M' ∧ 
    Perpendicular (LineThrough A2 BC) (LineThrough M' SideBC) ∧ 
    Perpendicular (LineThrough B2 CA) (LineThrough M' SideCA) ∧ 
    Perpendicular (LineThrough C2 AB) (LineThrough M' SideAB) :=
begin
  sorry
end

end perpendiculars_intersect_at_symmetric_point_l40_40714


namespace part1_part2_l40_40881

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |2 * x| + |2 * x - 3|

-- Part 1: Proving the inequality solution
theorem part1 (x : ℝ) (h : f x ≤ 5) :
  -1/2 ≤ x ∧ x ≤ 2 :=
sorry

-- Part 2: Proving the range of m
theorem part2 (x₀ m : ℝ) (h1 : x₀ ∈ Set.Ici 1)
  (h2 : f x₀ + m ≤ x₀ + 3/x₀) :
  m ≤ 1 :=
sorry

end part1_part2_l40_40881


namespace no_solution_intervals_l40_40078

theorem no_solution_intervals :
    ¬ ∃ x : ℝ, (2 / 3 < x ∧ x < 4 / 3) ∧ (1 / 5 < x ∧ x < 3 / 5) :=
by
  sorry

end no_solution_intervals_l40_40078


namespace prime_pair_probability_l40_40232

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrimeSum (a b : ℕ) : Prop := Nat.Prime (a + b)

theorem prime_pair_probability :
  let pairs := List.sublistsLen 2 firstTenPrimes
  let primePairs := pairs.filter (λ p, match p with
                                       | [a, b] => isPrimeSum a b
                                       | _ => false
                                       end)
  (primePairs.length : ℚ) / (pairs.length : ℚ) = 1 / 9 :=
by
  sorry

end prime_pair_probability_l40_40232


namespace percentage_meetings_correct_l40_40926

def work_day_hours : ℕ := 10
def minutes_in_hour : ℕ := 60
def total_work_day_minutes := work_day_hours * minutes_in_hour

def lunch_break_minutes : ℕ := 30
def effective_work_day_minutes := total_work_day_minutes - lunch_break_minutes

def first_meeting_minutes : ℕ := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

def percentage_of_day_spent_in_meetings := (total_meeting_minutes * 100) / effective_work_day_minutes

theorem percentage_meetings_correct : percentage_of_day_spent_in_meetings = 42 := 
by
  sorry

end percentage_meetings_correct_l40_40926


namespace speed_of_second_train_equivalent_l40_40651

noncomputable def relative_speed_in_m_per_s (time_seconds : ℝ) (total_distance_m : ℝ) : ℝ :=
total_distance_m / time_seconds

noncomputable def relative_speed_in_km_per_h (relative_speed_m_per_s : ℝ) : ℝ :=
relative_speed_m_per_s * 3.6

noncomputable def speed_of_second_train (relative_speed_km_per_h : ℝ) (speed_of_first_train_km_per_h : ℝ) : ℝ :=
relative_speed_km_per_h - speed_of_first_train_km_per_h

theorem speed_of_second_train_equivalent
  (length_of_first_train length_of_second_train : ℝ)
  (speed_of_first_train_km_per_h : ℝ)
  (time_of_crossing_seconds : ℝ) :
  speed_of_second_train
    (relative_speed_in_km_per_h (relative_speed_in_m_per_s time_of_crossing_seconds (length_of_first_train + length_of_second_train)))
    speed_of_first_train_km_per_h = 36 := by
  sorry

end speed_of_second_train_equivalent_l40_40651


namespace selling_price_of_cycle_l40_40820

theorem selling_price_of_cycle (cost_price : ℕ) (loss_percent : ℕ) (selling_price : ℕ) :
  cost_price = 1400 → loss_percent = 25 → selling_price = 1050 := by
  sorry

end selling_price_of_cycle_l40_40820


namespace question_1_question_2_l40_40303

def f (x a : ℝ) := |x - a|

theorem question_1 :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

theorem question_2 (a : ℝ) (h : a = 2) :
  (∀ x, f x a + f (x + 5) a ≥ m) → m ≤ 5 :=
by
  sorry

end question_1_question_2_l40_40303


namespace completing_square_l40_40815

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l40_40815


namespace bounce_ratio_l40_40366

theorem bounce_ratio (r : ℝ) (h₁ : 96 * r^4 = 3) : r = Real.sqrt 2 / 4 :=
by
  sorry

end bounce_ratio_l40_40366


namespace additional_pots_produced_l40_40031

theorem additional_pots_produced (first_hour_time_per_pot last_hour_time_per_pot : ℕ) :
  first_hour_time_per_pot = 6 →
  last_hour_time_per_pot = 5 →
  60 / last_hour_time_per_pot - 60 / first_hour_time_per_pot = 2 :=
by
  intros
  sorry

end additional_pots_produced_l40_40031


namespace max_cos_alpha_l40_40039

open Real

-- Define the condition as a hypothesis
def cos_sum_eq (α β : ℝ) : Prop :=
  cos (α + β) = cos α + cos β

-- State the maximum value theorem
theorem max_cos_alpha (α β : ℝ) (h : cos_sum_eq α β) : ∃ α, cos α = sqrt 3 - 1 :=
by
  sorry   -- Proof is omitted

#check max_cos_alpha

end max_cos_alpha_l40_40039


namespace first_term_of_infinite_geo_series_l40_40534

theorem first_term_of_infinite_geo_series (S r : ℝ) (hS : S = 80) (hr : r = 1/4) :
  let a := (S * (1 - r)) in a = 60 :=
by
  sorry

end first_term_of_infinite_geo_series_l40_40534


namespace additional_pots_last_hour_l40_40033

theorem additional_pots_last_hour (h1 : 60 / 6 = 10) (h2 : 60 / 5 = 12) : 12 - 10 = 2 :=
by
  sorry

end additional_pots_last_hour_l40_40033


namespace solve_for_y_l40_40473

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l40_40473


namespace relationship_correct_l40_40755

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem relationship_correct (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) :
  log_base a b < a^b ∧ a^b < b^a :=
by sorry

end relationship_correct_l40_40755


namespace stones_on_one_side_l40_40305

theorem stones_on_one_side (total_perimeter_stones : ℕ) (h : total_perimeter_stones = 84) :
  ∃ s : ℕ, 4 * s - 4 = total_perimeter_stones ∧ s = 22 :=
by
  use 22
  sorry

end stones_on_one_side_l40_40305


namespace least_positive_integer_with_12_factors_l40_40004

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l40_40004


namespace least_positive_integer_with_12_factors_l40_40021

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l40_40021


namespace ratio_of_poets_to_novelists_l40_40985

-- Define the conditions
def total_people : ℕ := 24
def novelists : ℕ := 15
def poets := total_people - novelists

-- Theorem asserting the ratio of poets to novelists
theorem ratio_of_poets_to_novelists (h1 : poets = total_people - novelists) : poets / novelists = 3 / 5 := by
  sorry

end ratio_of_poets_to_novelists_l40_40985


namespace simplify_fraction_l40_40346

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) = (5 * Real.sqrt 3) / 36 :=
by
  have h1 : Real.sqrt 75 = 5 * Real.sqrt 3 := by sorry
  have h2 : Real.sqrt 48 = 4 * Real.sqrt 3 := by sorry
  sorry

end simplify_fraction_l40_40346


namespace inverse_proportion_value_k_l40_40313

theorem inverse_proportion_value_k (k : ℝ) (h : k ≠ 0) (H : (2 : ℝ), -1 = (k : ℝ)/(2)) :
  k = -2 :=
by
  sorry

end inverse_proportion_value_k_l40_40313


namespace book_profit_percentage_l40_40710

noncomputable def profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let discount := discount_rate / 100 * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

theorem book_profit_percentage :
  profit_percentage 47.50 69.85 15 = 24.994736842105263 :=
by
  sorry

end book_profit_percentage_l40_40710


namespace fg_of_3_eq_97_l40_40890

def f (x : ℕ) : ℕ := 4 * x - 3
def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem fg_of_3_eq_97 : f (g 3) = 97 := by
  sorry

end fg_of_3_eq_97_l40_40890


namespace probability_prime_sum_of_two_draws_l40_40266

open Finset

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrime (n : ℕ) : Prop := Nat.Prime n

def validPairs (primes : List ℕ) : List (ℕ × ℕ) :=
  primes.bind (λ p1 => primes.filter (λ p2 => p1 ≠ p2 ∧ isPrime (p1 + p2)).map (λ p2 => (p1, p2)))

theorem probability_prime_sum_of_two_draws :
  (firstTenPrimes.length.choose 2 : ℚ) > 0 →
  let validPairs := validPairs firstTenPrimes
  let numValidPairs := validPairs.toFinset.card
  let totalPairs := 45
  (∀ (x ∈ validPairs), ∃ (a b : ℕ), a ∈ firstTenPrimes ∧ b ∈ firstTenPrimes ∧ a ≠ b ∧ isPrime (a + b) ∧ (a, b) = x) →
  (∀ p, p ∈ firstTenPrimes → isPrime p) →
  numValidPairs / totalPairs = (1 : ℚ) / 9 := by
  sorry

end probability_prime_sum_of_two_draws_l40_40266


namespace solve_speed_of_second_train_l40_40650

open Real

noncomputable def speed_of_second_train
  (L1 : ℝ) (L2 : ℝ) (S1 : ℝ) (T : ℝ) : ℝ :=
  let D := (L1 + L2) / 1000   -- Total distance in kilometers
  let H := T / 3600           -- Time in hours
  let relative_speed := D / H -- Relative speed in km/h
  relative_speed - S1         -- Speed of the second train

theorem solve_speed_of_second_train :
  speed_of_second_train 100 220 42 15.99872010239181 = 30 := by
  sorry

end solve_speed_of_second_train_l40_40650


namespace remainder_of_sum_of_squares_mod_l40_40509

-- Define the function to compute the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Define the specific sum for the first 15 natural numbers
def S : ℕ := sum_of_squares 15

-- State the theorem
theorem remainder_of_sum_of_squares_mod (n : ℕ) (h : n = 15) : 
  S % 13 = 5 := by
  sorry

end remainder_of_sum_of_squares_mod_l40_40509


namespace train_passing_time_l40_40152

theorem train_passing_time :
  ∀ (length : ℕ) (speed_kmph : ℕ),
    length = 120 →
    speed_kmph = 72 →
    ∃ (time : ℕ), time = 6 :=
by
  intro length speed_kmph hlength hspeed_kmph
  sorry

end train_passing_time_l40_40152


namespace find_z_l40_40879

theorem find_z (z : ℝ) 
    (cos_angle : (2 + 2 * z) / ((Real.sqrt (1 + z^2)) * 3) = 2 / 3) : 
    z = 0 := 
sorry

end find_z_l40_40879


namespace completing_square_l40_40812

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l40_40812


namespace truth_probability_l40_40391

variables (P_A P_B P_AB : ℝ)

theorem truth_probability (h1 : P_B = 0.60) (h2 : P_AB = 0.48) : P_A = 0.80 :=
by
  have h3 : P_AB = P_A * P_B := sorry  -- Placeholder for the rule: P(A and B) = P(A) * P(B)
  rw [h2, h1] at h3
  sorry

end truth_probability_l40_40391


namespace least_positive_integer_with_12_factors_l40_40023

/--
Prove that the smallest positive integer with exactly 12 positive factors is 60.
-/
theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (0 < n) ∧ (nat.factors_count n = 12) ∧ (∀ m : ℕ, (0 < m) ∧ (nat.factors_count m = 12) → n ≤ m) → n = 60 :=
sorry

end least_positive_integer_with_12_factors_l40_40023


namespace bars_cannot_form_triangle_l40_40365

theorem bars_cannot_form_triangle 
  (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 10) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) :=
by 
  rw [h1, h2, h3]
  sorry

end bars_cannot_form_triangle_l40_40365


namespace ac_length_l40_40822

theorem ac_length (a b c d e : ℝ)
  (h1 : b - a = 5)
  (h2 : c - b = 2 * (d - c))
  (h3 : e - d = 4)
  (h4 : e - a = 18) :
  d - a = 11 :=
by
  sorry

end ac_length_l40_40822


namespace line_intersects_y_axis_at_0_2_l40_40064

theorem line_intersects_y_axis_at_0_2 (P1 P2 : ℝ × ℝ) (h1 : P1 = (2, 8)) (h2 : P2 = (6, 20)) :
  ∃ y : ℝ, (0, y) = (0, 2) :=
by {
  sorry
}

end line_intersects_y_axis_at_0_2_l40_40064


namespace tan_alpha_value_l40_40127

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l40_40127


namespace simplify_fraction_when_b_equals_4_l40_40624

theorem simplify_fraction_when_b_equals_4 (b : ℕ) (h : b = 4) : (18 * b^4) / (27 * b^3) = 8 / 3 :=
by {
  -- we use the provided condition to state our theorem goals.
  sorry
}

end simplify_fraction_when_b_equals_4_l40_40624


namespace ned_initial_lives_l40_40035

-- Define the initial number of lives Ned had
def initial_lives (start_lives current_lives lost_lives : ℕ) : ℕ :=
  current_lives + lost_lives

-- Define the conditions
def current_lives := 70
def lost_lives := 13

-- State the theorem
theorem ned_initial_lives : initial_lives current_lives current_lives lost_lives = 83 := by
  sorry

end ned_initial_lives_l40_40035


namespace non_empty_solution_set_range_l40_40764

theorem non_empty_solution_set_range {a : ℝ} 
  (h : ∃ x : ℝ, |x + 2| + |x - 3| ≤ a) : 
  a ≥ 5 :=
sorry

end non_empty_solution_set_range_l40_40764


namespace malcolm_social_media_followers_l40_40927

theorem malcolm_social_media_followers :
  let instagram_initial := 240
  let facebook_initial := 500
  let twitter_initial := (instagram_initial + facebook_initial) / 2
  let tiktok_initial := 3 * twitter_initial
  let youtube_initial := tiktok_initial + 510
  let pinterest_initial := 120
  let snapchat_initial := pinterest_initial / 2

  let instagram_after := instagram_initial + (15 * instagram_initial / 100)
  let facebook_after := facebook_initial + (20 * facebook_initial / 100)
  let twitter_after := twitter_initial - 12
  let tiktok_after := tiktok_initial + (10 * tiktok_initial / 100)
  let youtube_after := youtube_initial + (8 * youtube_initial / 100)
  let pinterest_after := pinterest_initial + 20
  let snapchat_after := snapchat_initial - (5 * snapchat_initial / 100)

  instagram_after + facebook_after + twitter_after + tiktok_after + youtube_after + pinterest_after + snapchat_after = 4402 := sorry

end malcolm_social_media_followers_l40_40927


namespace salt_percentage_l40_40043

theorem salt_percentage :
  ∀ (salt water : ℝ), salt = 10 → water = 90 → 
  100 * (salt / (salt + water)) = 10 :=
by
  intros salt water h_salt h_water
  sorry

end salt_percentage_l40_40043


namespace quadratic_roots_solve_equation_l40_40938

theorem quadratic_roots (a b c : ℝ) (x1 x2 : ℝ) (h : a ≠ 0)
  (root_eq : x1 = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
            ∧ x2 = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h_eq : a*x^2 + b*x + c = 0) :
  ∀ x, a*x^2 + b*x + c = 0 → x = x1 ∨ x = x2 :=
by
  sorry -- Proof not given

theorem solve_equation (x : ℝ) :
  7*x*(5*x + 2) = 6*(5*x + 2) ↔ x = -2 / 5 ∨ x = 6 / 7 :=
by
  sorry -- Proof not given

end quadratic_roots_solve_equation_l40_40938


namespace car_speed_first_hour_l40_40643

theorem car_speed_first_hour (x : ℝ) (h_second_hour_speed : x + 80 / 2 = 85) : x = 90 :=
sorry

end car_speed_first_hour_l40_40643


namespace students_like_apple_and_chocolate_not_carrot_l40_40903

-- Definitions based on the conditions
def total_students : ℕ := 50
def apple_likers : ℕ := 23
def chocolate_likers : ℕ := 20
def carrot_likers : ℕ := 10
def non_likers : ℕ := 15

-- The main statement we need to prove: 
-- the number of students who liked both apple pie and chocolate cake but not carrot cake
theorem students_like_apple_and_chocolate_not_carrot : 
  ∃ (a b c d : ℕ), a + b + d = apple_likers ∧
                    a + c + d = chocolate_likers ∧
                    b + c + d = carrot_likers ∧
                    a + b + c + (50 - (35) - 15) = 35 ∧ 
                    a = 7 :=
by 
  sorry

end students_like_apple_and_chocolate_not_carrot_l40_40903


namespace problem_statement_l40_40218

-- Universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Definition of set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ (Real.sqrt (2 * x - x ^ 2 + 3)) }

-- Complement of M in U
def C_U_M : Set ℝ := { y | y < 1 ∨ y > 4 }

-- Definition of set N
def N : Set ℝ := { x | -3 < x ∧ x < 2 }

-- Theorem stating (C_U_M) ∩ N = (-3, 1)
theorem problem_statement : (C_U_M ∩ N) = { x | -3 < x ∧ x < 1 } :=
sorry

end problem_statement_l40_40218


namespace additional_pots_produced_l40_40032

theorem additional_pots_produced (first_hour_time_per_pot last_hour_time_per_pot : ℕ) :
  first_hour_time_per_pot = 6 →
  last_hour_time_per_pot = 5 →
  60 / last_hour_time_per_pot - 60 / first_hour_time_per_pot = 2 :=
by
  intros
  sorry

end additional_pots_produced_l40_40032


namespace lcm_48_180_l40_40560

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l40_40560


namespace no_solution_in_nat_for_xx_plus_2yy_eq_zz_l40_40174

theorem no_solution_in_nat_for_xx_plus_2yy_eq_zz :
  ¬∃ (x y z : ℕ), x^x + 2 * y^y = z^z := by
  sorry

end no_solution_in_nat_for_xx_plus_2yy_eq_zz_l40_40174


namespace calculate_product_l40_40997

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l40_40997


namespace tan_alpha_solution_l40_40113

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l40_40113


namespace tan_alpha_value_l40_40098

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l40_40098


namespace greatest_consecutive_sum_l40_40191

theorem greatest_consecutive_sum (S : ℤ) (hS : S = 105) : 
  ∃ N : ℤ, (∃ a : ℤ, (N * (2 * a + N - 1) = 2 * S)) ∧ 
  (∀ M : ℤ, (∃ b : ℤ, (M * (2 * b + M - 1) = 2 * S)) → M ≤ N) ∧ N = 210 := 
sorry

end greatest_consecutive_sum_l40_40191


namespace prime_probability_l40_40248

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def pairs : List (ℕ × ℕ) :=
(primes.erase 2).map (λ p => (2, p))

def prime_sums : List (ℕ × ℕ) :=
pairs.filter (λ ⟨a, b⟩ => is_prime (a + b))

theorem prime_probability :
  let total_pairs := Nat.choose 10 2
  let valid_pairs := prime_sums.length
  (valid_pairs : ℚ) / (total_pairs : ℚ) = 1 / 9 :=
by
  sorry

end prime_probability_l40_40248


namespace amc_inequality_l40_40782

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem amc_inequality : (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 3 / 2 :=
sorry

end amc_inequality_l40_40782


namespace minimum_value_of_f_l40_40507

-- Define the function
def f (x : ℝ) : ℝ := |x - 4| + |x + 8| + |x - 5|

-- State the theorem
theorem minimum_value_of_f : ∃ x, f x = -25 :=
by
  sorry

end minimum_value_of_f_l40_40507


namespace simplify_fraction_l40_40466

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : 
  (x / (x - y) - y / (x + y)) = (x^2 + y^2) / (x^2 - y^2) :=
sorry

end simplify_fraction_l40_40466


namespace smaller_angle_formed_by_hands_at_3_15_l40_40958

def degrees_per_hour : ℝ := 30
def degrees_per_minute : ℝ := 6
def hour_hand_degrees_per_minute : ℝ := 0.5

def minute_position (minute : ℕ) : ℝ :=
  minute * degrees_per_minute

def hour_position (hour : ℕ) (minute : ℕ) : ℝ :=
  hour * degrees_per_hour + minute * hour_hand_degrees_per_minute

theorem smaller_angle_formed_by_hands_at_3_15 : 
  minute_position 15 = 90 ∧ 
  hour_position 3 15 = 97.5 →
  abs (hour_position 3 15 - minute_position 15) = 7.5 :=
by
  intros h
  sorry

end smaller_angle_formed_by_hands_at_3_15_l40_40958


namespace dragon_poker_score_l40_40898

-- Define the scoring system
def score (card : Nat) : Int :=
  match card with
  | 1     => 1
  | 11    => -2
  | n     => -(2^n)

-- Define the possible scores a single card can have
def possible_scores : List Int := [1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024]

-- Scoring function for four suits
def ways_to_score (target : Int) : Nat :=
  Nat.choose (target + 4 - 1) (4 - 1)

-- Problem statement to prove
theorem dragon_poker_score : ways_to_score 2018 = 1373734330 := by
  sorry

end dragon_poker_score_l40_40898


namespace total_wheels_in_garage_l40_40493

theorem total_wheels_in_garage : 
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  in (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle) = 25 :=
by
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  have h_bicycles := bicycles * wheels_per_bicycle
  have h_tricycles := tricycles * wheels_per_tricycle
  have h_unicycles := unicycles * wheels_per_unicycle
  show (h_bicycles + h_tricycles + h_unicycles) = 25
  sorry

end total_wheels_in_garage_l40_40493


namespace simplify_expression_l40_40625

-- Define the conditions as parameters
variable (x y : ℕ)

-- State the theorem with the required conditions and proof goal
theorem simplify_expression (hx : x = 2) (hy : y = 3) :
  (8 * x * y^2) / (6 * x^2 * y) = 2 := by
  -- We'll provide the outline and leave the proof as sorry
  sorry

end simplify_expression_l40_40625


namespace isosceles_right_triangle_area_l40_40498

theorem isosceles_right_triangle_area (a : ℝ) (h : ℝ) (p : ℝ) 
  (h_triangle : h = a * Real.sqrt 2) 
  (hypotenuse_is_16 : h = 16) :
  (1 / 2) * a * a = 64 := 
by
  -- Skip the proof as per guidelines
  sorry

end isosceles_right_triangle_area_l40_40498


namespace cora_reading_ratio_l40_40542

variable (P : Nat) 
variable (M T W Th F : Nat)

-- Conditions
def conditions (P M T W Th F : Nat) : Prop := 
  P = 158 ∧ 
  M = 23 ∧ 
  T = 38 ∧ 
  W = 61 ∧ 
  Th = 12 ∧ 
  F = Th

-- The theorem statement
theorem cora_reading_ratio (h : conditions P M T W Th F) : F / Th = 1 / 1 :=
by
  -- We use the conditions to apply the proof
  obtain ⟨hp, hm, ht, hw, hth, hf⟩ := h
  rw [hf]
  norm_num
  sorry

end cora_reading_ratio_l40_40542


namespace least_positive_integer_with_12_factors_is_72_l40_40008

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l40_40008


namespace power_equality_l40_40140

theorem power_equality (x : ℝ) (n : ℕ) (h : x^(2 * n) = 3) : x^(4 * n) = 9 := 
by 
  sorry

end power_equality_l40_40140


namespace smallest_three_digit_divisible_by_4_and_5_l40_40961

-- Define the problem conditions and goal as a Lean theorem statement
theorem smallest_three_digit_divisible_by_4_and_5 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 4 = 0 ∧ m % 5 = 0 → n ≤ m) :=
sorry

end smallest_three_digit_divisible_by_4_and_5_l40_40961


namespace simplify_trig_expression_l40_40934

variable (θ : ℝ)
variable (h_tan : Real.tan θ = 1)
variable (h_cot : Real.cot θ = 1)

theorem simplify_trig_expression :
  (Real.tan θ) ^ 3 + (Real.cot θ) ^ 3 / 
  (Real.tan θ + Real.cot θ) = 1 :=
by
  sorry

end simplify_trig_expression_l40_40934


namespace lcm_48_180_l40_40559

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l40_40559


namespace cheryl_used_material_l40_40701

theorem cheryl_used_material
    (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
    (h1 : material1 = 5/9)
    (h2 : material2 = 1/3)
    (h_lf : leftover = 8/24) :
    material1 + material2 - leftover = 5/9 :=
by
  sorry

end cheryl_used_material_l40_40701


namespace cats_combined_weight_l40_40731

theorem cats_combined_weight :
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  cat1 + cat2 + cat3 = 13 := 
by
  let cat1 := 2
  let cat2 := 7
  let cat3 := 4
  sorry

end cats_combined_weight_l40_40731


namespace trail_length_is_20_km_l40_40444

-- Define the conditions and the question
def length_of_trail (L : ℝ) (hiked_percentage remaining_distance : ℝ) : Prop :=
  hiked_percentage = 0.60 ∧ remaining_distance = 8 ∧ 0.40 * L = remaining_distance

-- The statement: given the conditions, prove that length of trail is 20 km
theorem trail_length_is_20_km : ∃ L : ℝ, length_of_trail L 0.60 8 ∧ L = 20 := by
  -- Proof goes here
  sorry

end trail_length_is_20_km_l40_40444


namespace tan_alpha_proof_l40_40109

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l40_40109


namespace soccer_score_combinations_l40_40595

theorem soccer_score_combinations :
  ∃ (x y z : ℕ), x + y + z = 14 ∧ 3 * x + y = 19 ∧ x + y + z ≥ 0 ∧ 
    ({ (3, 10, 1), (4, 7, 3), (5, 4, 5), (6, 1, 7) } = 
      { (x, y, z) | x + y + z = 14 ∧ 3 * x + y = 19 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 }) :=
by 
  sorry

end soccer_score_combinations_l40_40595


namespace sufficient_but_not_necessary_l40_40331

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 2) (h2 : b > 1) : 
  (a + b > 3 ∧ a * b > 2) ∧ ∃ x y : ℝ, (x + y > 3 ∧ x * y > 2) ∧ (¬ (x > 2 ∧ y > 1)) :=
by 
  sorry

end sufficient_but_not_necessary_l40_40331


namespace gcd_437_323_eq_19_l40_40539

theorem gcd_437_323_eq_19 : Int.gcd 437 323 = 19 := 
by 
  sorry

end gcd_437_323_eq_19_l40_40539


namespace total_wheels_in_garage_l40_40495

def bicycles: Nat := 3
def tricycles: Nat := 4
def unicycles: Nat := 7

def wheels_per_bicycle: Nat := 2
def wheels_per_tricycle: Nat := 3
def wheels_per_unicycle: Nat := 1

theorem total_wheels_in_garage (bicycles tricycles unicycles wheels_per_bicycle wheels_per_tricycle wheels_per_unicycle : Nat) :
  bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle = 25 := by
  sorry

end total_wheels_in_garage_l40_40495


namespace rectangular_prism_faces_edges_vertices_sum_l40_40680

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l40_40680


namespace smallest_two_digit_multiple_of_17_l40_40510

theorem smallest_two_digit_multiple_of_17 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ 17 ∣ n ∧ ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ 17 ∣ m → n ≤ m :=
begin
  use 17,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  split,
  { use 1,
    norm_num },
  intros m h1 h2 h3,
  rw ← nat.dvd_iff_mod_eq_zero at *,
  have h4 := nat.mod_eq_zero_of_dvd h3,
  cases (nat.le_of_mod_eq_zero h4),
  { linarith [nat.le_of_dvd (dec_trivial) this] },
  { exfalso,
    linarith }
end

end smallest_two_digit_multiple_of_17_l40_40510


namespace extrema_range_of_m_l40_40088

def has_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, (∀ z : ℝ, z ≤ x → f z ≤ f x) ∧ (∀ z : ℝ, z ≥ y → f z ≤ f y)

noncomputable def f (m x : ℝ) : ℝ :=
  x^3 + m * x^2 + (m + 6) * x + 1

theorem extrema_range_of_m (m : ℝ) :
  has_extrema (f m) ↔ (m ∈ Set.Iic (-3) ∪ Set.Ici 6) :=
by
  sorry

end extrema_range_of_m_l40_40088


namespace taller_pot_shadow_length_l40_40516

theorem taller_pot_shadow_length
  (height1 shadow1 height2 : ℝ)
  (h1 : height1 = 20)
  (h2 : shadow1 = 10)
  (h3 : height2 = 40) :
  ∃ shadow2 : ℝ, height2 / shadow2 = height1 / shadow1 ∧ shadow2 = 20 :=
by
  -- Since Lean requires proofs for existential statements,
  -- we add "sorry" to skip the proof.
  sorry

end taller_pot_shadow_length_l40_40516


namespace ryan_learning_schedule_l40_40788

theorem ryan_learning_schedule
  (E1 E2 E3 S1 S2 S3 : ℕ)
  (hE1 : E1 = 7) (hE2 : E2 = 6) (hE3 : E3 = 8)
  (hS1 : S1 = 4) (hS2 : S2 = 5) (hS3 : S3 = 3):
  (E1 + E2 + E3) - (S1 + S2 + S3) = 9 :=
by
  sorry

end ryan_learning_schedule_l40_40788


namespace average_score_l40_40804

variable (u v A : ℝ)
variable (h1 : v / u = 1/3)
variable (h2 : A = (u + v) / 2)

theorem average_score : A = (2/3) * u := by
  sorry

end average_score_l40_40804


namespace total_wheels_in_garage_l40_40492

theorem total_wheels_in_garage : 
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  in (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle + unicycles * wheels_per_unicycle) = 25 :=
by
  let bicycles := 3
  let tricycles := 4
  let unicycles := 7
  let wheels_per_bicycle := 2
  let wheels_per_tricycle := 3
  let wheels_per_unicycle := 1
  have h_bicycles := bicycles * wheels_per_bicycle
  have h_tricycles := tricycles * wheels_per_tricycle
  have h_unicycles := unicycles * wheels_per_unicycle
  show (h_bicycles + h_tricycles + h_unicycles) = 25
  sorry

end total_wheels_in_garage_l40_40492


namespace people_per_car_l40_40708

theorem people_per_car (total_people : ℝ) (total_cars : ℝ) (h1 : total_people = 189) (h2 : total_cars = 3.0) : total_people / total_cars = 63 := 
by
  sorry

end people_per_car_l40_40708


namespace general_term_geometric_seq_sum_of_arithmetic_seq_maximum_sum_of_arithmetic_seq_l40_40318

noncomputable def geometric_seq (n : ℕ) : ℝ :=
  if n = 2 then 2 else
  if n = 5 then 16 else
  2^(n - 1)

theorem general_term_geometric_seq :
  (geometric_seq 2 = 2) → (geometric_seq 5 = 16) →
  ∀ n, geometric_seq n = 2^(n - 1) :=
by sorry

noncomputable def arithmetic_seq (n : ℕ) : ℝ :=
  16 + (n - 1) * (-2)

noncomputable def sum_arithmetic_seq (n : ℕ) : ℝ :=
  n * 16 + (n * (n - 1) / 2) * (-2)

theorem sum_of_arithmetic_seq (b1 b8: ℝ) 
  (hb1: b1 = 16) (hb8: b8 = 2) :
  ∀ n, sum_arithmetic_seq n = 17 * n - n^2 :=
by sorry

theorem maximum_sum_of_arithmetic_seq (b1 b8: ℝ) 
  (hb1: b1 = 16) (hb8: b8 = 2) :
  ∃ n, sum_arithmetic_seq n = 72 ∧ (n = 8 ∨ n = 9) :=
by sorry

end general_term_geometric_seq_sum_of_arithmetic_seq_maximum_sum_of_arithmetic_seq_l40_40318


namespace smallest_number_greater_300_with_remainder_24_l40_40202

theorem smallest_number_greater_300_with_remainder_24 :
  ∃ n : ℕ, n > 300 ∧ n % 25 = 24 ∧ ∀ k : ℕ, k > 300 ∧ k % 25 = 24 → n ≤ k :=
sorry

end smallest_number_greater_300_with_remainder_24_l40_40202


namespace store_earnings_correct_l40_40717

theorem store_earnings_correct :
  let graphics_cards_qty := 10
  let hard_drives_qty := 14
  let cpus_qty := 8
  let rams_qty := 4
  let psus_qty := 12
  let monitors_qty := 6
  let keyboards_qty := 18
  let mice_qty := 24

  let graphics_card_price := 600
  let hard_drive_price := 80
  let cpu_price := 200
  let ram_price := 60
  let psu_price := 90
  let monitor_price := 250
  let keyboard_price := 40
  let mouse_price := 20

  let total_earnings := graphics_cards_qty * graphics_card_price +
                        hard_drives_qty * hard_drive_price +
                        cpus_qty * cpu_price +
                        rams_qty * ram_price +
                        psus_qty * psu_price +
                        monitors_qty * monitor_price +
                        keyboards_qty * keyboard_price +
                        mice_qty * mouse_price
  total_earnings = 12740 :=
by
  -- definitions and calculations here
  sorry

end store_earnings_correct_l40_40717


namespace distance_to_school_l40_40323

def jerry_one_way_time : ℝ := 15  -- Jerry's one-way time in minutes
def carson_speed_mph : ℝ := 8  -- Carson's speed in miles per hour
def minutes_per_hour : ℝ := 60  -- Number of minutes in one hour

noncomputable def carson_speed_mpm : ℝ := carson_speed_mph / minutes_per_hour -- Carson's speed in miles per minute
def carson_one_way_time : ℝ := jerry_one_way_time -- Carson's one-way time is the same as Jerry's round trip time / 2

-- Prove that the distance to the school is 2 miles.
theorem distance_to_school : carson_speed_mpm * carson_one_way_time = 2 := by
  sorry

end distance_to_school_l40_40323


namespace symmetric_points_l40_40295

-- Definitions from conditions
def is_symmetric (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

-- The points A and B
def A : ℝ × ℝ := (a, -2)
def B : ℝ × ℝ := (4, b)

-- The Lean 4 statement
theorem symmetric_points (a b : ℝ) (h : is_symmetric (A a) (B b)) : a - b = -6 :=
  sorry  -- Proof is omitted

end symmetric_points_l40_40295


namespace lost_revenue_is_correct_l40_40206

-- Define the ticket prices
def general_admission_price : ℤ := 10
def children_price : ℤ := 6
def senior_price : ℤ := 8
def veteran_discount : ℤ := 2

-- Define the number of tickets sold
def general_tickets_sold : ℤ := 20
def children_tickets_sold : ℤ := 3
def senior_tickets_sold : ℤ := 4
def veteran_tickets_sold : ℤ := 2

-- Calculate the actual revenue from sold tickets
def actual_revenue := (general_tickets_sold * general_admission_price) + 
                      (children_tickets_sold * children_price) + 
                      (senior_tickets_sold * senior_price) + 
                      (veteran_tickets_sold * (general_admission_price - veteran_discount))

-- Define the maximum potential revenue assuming all tickets are sold at general admission price
def max_potential_revenue : ℤ := 50 * general_admission_price

-- Define the potential revenue lost
def potential_revenue_lost := max_potential_revenue - actual_revenue

-- The theorem to prove
theorem lost_revenue_is_correct : potential_revenue_lost = 234 := 
by
  -- Placeholder for proof
  sorry

end lost_revenue_is_correct_l40_40206


namespace sample_size_product_A_l40_40434

theorem sample_size_product_A 
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (total_ratio : ℕ)
  (sample_size : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5)
  (h_total_ratio : total_ratio = ratio_A + ratio_B + ratio_C)
  (h_sample_size : sample_size = 80) :
  (80 * (ratio_A : ℚ) / total_ratio) = 16 :=
by
  sorry

end sample_size_product_A_l40_40434


namespace cylinder_surface_area_l40_40291

theorem cylinder_surface_area
  (r : ℝ) (V : ℝ) (h_radius : r = 1) (h_volume : V = 4 * Real.pi) :
  ∃ S : ℝ, S = 10 * Real.pi :=
by
  let l := V / (Real.pi * r^2)
  have h_l : l = 4 := sorry
  let S := 2 * Real.pi * r^2 + 2 * Real.pi * r * l
  have h_S : S = 10 * Real.pi := sorry
  exact ⟨S, h_S⟩

end cylinder_surface_area_l40_40291


namespace probability_prime_sum_l40_40253

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end probability_prime_sum_l40_40253


namespace find_largest_divisor_l40_40414

def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

theorem find_largest_divisor :
  ∃ m : ℕ, (∀ n : ℕ, f n % m = 0) ∧ m = 36 :=
sorry

end find_largest_divisor_l40_40414


namespace prime_probability_l40_40245

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def pairs : List (ℕ × ℕ) :=
(primes.erase 2).map (λ p => (2, p))

def prime_sums : List (ℕ × ℕ) :=
pairs.filter (λ ⟨a, b⟩ => is_prime (a + b))

theorem prime_probability :
  let total_pairs := Nat.choose 10 2
  let valid_pairs := prime_sums.length
  (valid_pairs : ℚ) / (total_pairs : ℚ) = 1 / 9 :=
by
  sorry

end prime_probability_l40_40245


namespace pens_sales_consistency_books_left_indeterminate_l40_40619

-- The initial conditions
def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_left : ℕ := 19
def pens_sold : ℕ := 23

-- Prove the consistency of the number of pens sold
theorem pens_sales_consistency : initial_pens - pens_left = pens_sold := by
  sorry

-- Assert that the number of books left is indeterminate based on provided conditions
theorem books_left_indeterminate : ∃ b_left : ℕ, b_left ≤ initial_books ∧
    ∀ n_books_sold : ℕ, n_books_sold > 0 → b_left = initial_books - n_books_sold := by
  sorry

end pens_sales_consistency_books_left_indeterminate_l40_40619


namespace lcm_48_180_l40_40557

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l40_40557


namespace jessica_money_left_l40_40154

theorem jessica_money_left : 
  let initial_amount := 11.73
  let amount_spent := 10.22
  initial_amount - amount_spent = 1.51 :=
by
  sorry

end jessica_money_left_l40_40154


namespace tan_alpha_proof_l40_40108

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l40_40108


namespace cosine_value_l40_40758

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

noncomputable def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

noncomputable def magnitude (x : ℝ × ℝ) : ℝ :=
  (x.1 ^ 2 + x.2 ^ 2).sqrt

noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

theorem cosine_value :
  cos_angle a b = 2 * (5:ℝ).sqrt / 25 :=
by
  sorry

end cosine_value_l40_40758


namespace simplify_inverse_sum_l40_40093

variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem simplify_inverse_sum :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) :=
by sorry

end simplify_inverse_sum_l40_40093


namespace probability_prime_sum_of_two_draws_l40_40268

open Finset

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrime (n : ℕ) : Prop := Nat.Prime n

def validPairs (primes : List ℕ) : List (ℕ × ℕ) :=
  primes.bind (λ p1 => primes.filter (λ p2 => p1 ≠ p2 ∧ isPrime (p1 + p2)).map (λ p2 => (p1, p2)))

theorem probability_prime_sum_of_two_draws :
  (firstTenPrimes.length.choose 2 : ℚ) > 0 →
  let validPairs := validPairs firstTenPrimes
  let numValidPairs := validPairs.toFinset.card
  let totalPairs := 45
  (∀ (x ∈ validPairs), ∃ (a b : ℕ), a ∈ firstTenPrimes ∧ b ∈ firstTenPrimes ∧ a ≠ b ∧ isPrime (a + b) ∧ (a, b) = x) →
  (∀ p, p ∈ firstTenPrimes → isPrime p) →
  numValidPairs / totalPairs = (1 : ℚ) / 9 := by
  sorry

end probability_prime_sum_of_two_draws_l40_40268


namespace remaining_integers_count_l40_40184

open Finset

theorem remaining_integers_count :
  let T := range 101 \ {x | x % 4 = 0 ∧ x ≤ 100} ∪ {x | x % 5 = 0 ∧ x % 4 ≠ 0 ∧ x ≤ 100}
  in T.card = 60 :=
by
  let T := range 101 \ {x | x % 4 = 0 ∧ x ≤ 100} ∪ {x | x % 5 = 0 ∧ x % 4 ≠ 0 ∧ x ≤ 100}
  show T.card = 60
  sorry

end remaining_integers_count_l40_40184


namespace least_positive_integer_with_12_factors_is_96_l40_40018

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l40_40018


namespace expand_product_l40_40277

theorem expand_product : ∀ (x : ℝ), (x + 2) * (x^2 - 4 * x + 1) = x^3 - 2 * x^2 - 7 * x + 2 :=
by 
  intro x
  sorry

end expand_product_l40_40277


namespace sum_of_faces_edges_vertices_rectangular_prism_l40_40682

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l40_40682


namespace fg_evaluation_l40_40892

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_evaluation : f (g 3) = 97 := by
  sorry

end fg_evaluation_l40_40892


namespace positive_difference_of_squares_l40_40487

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l40_40487


namespace pounds_over_minimum_l40_40735

def cost_per_pound : ℕ := 3
def minimum_purchase : ℕ := 15
def total_spent : ℕ := 105

theorem pounds_over_minimum : 
  (total_spent / cost_per_pound) - minimum_purchase = 20 :=
by
  sorry

end pounds_over_minimum_l40_40735


namespace maximum_distance_to_line_l40_40301

noncomputable def polarToRectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def line_l_in_rectangular_coords (x y : ℝ) : Prop :=
  x - y + 10 = 0

noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 + 2 * Real.sin α)

noncomputable def curve_C_in_general_form (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

noncomputable def distance_from_point_to_line (x y a b c : ℝ) :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

theorem maximum_distance_to_line 
  (α ∈ [0, 2*Real.PI)) :
  ∃ d, d = 4 * Real.sqrt 2 + 2 :=
by
  let P := curve_C α
  let C := curve_C_in_general_form
  have h_center : (0, 2) := (0, 2)
  let line_l := line_l_in_rectangular_coords
  let d := distance_from_point_to_line 0 2 1 -1 10
  sorry

end maximum_distance_to_line_l40_40301


namespace sum_faces_edges_vertices_eq_26_l40_40663

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l40_40663


namespace problem1_l40_40201

theorem problem1 {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end problem1_l40_40201


namespace probability_coin_heads_and_die_two_l40_40582

namespace CoinDieProbability

def fair_coin := {head, tail} -- Define the possible outcomes for a fair coin
def die := {1, 2, 3, 4, 5, 6} -- Define the possible outcomes for a regular six-sided die

def successful_outcome := (head, 2) -- Define the successful outcome

def total_outcomes := (fair_coin × die) -- Define the sample space of total outcomes

def probability_successful_outcome (s : total_outcomes) : Prop :=
  (s = successful_outcome)

theorem probability_coin_heads_and_die_two :
  (MeasureTheory.Probability (s in total_outcomes, probability_successful_outcome s) = 1 / 12) :=
sorry

end CoinDieProbability

end probability_coin_heads_and_die_two_l40_40582


namespace tan_alpha_sqrt_15_over_15_l40_40118

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l40_40118


namespace clock_hands_angle_3_15_l40_40959

-- Define the context of the problem
def degreesPerHour := 360 / 12
def degreesPerMinute := 360 / 60
def minuteMarkAngle (minutes : ℕ) := minutes * degreesPerMinute
def hourMarkAngle (hours : ℕ) (minutes : ℕ) := (hours % 12) * degreesPerHour + (minutes * degreesPerHour / 60)

-- The target theorem to prove
theorem clock_hands_angle_3_15 : 
  let minuteHandAngle := minuteMarkAngle 15 in
  let hourHandAngle := hourMarkAngle 3 15 in
  |hourHandAngle - minuteHandAngle| = 7.5 :=
by
  -- The proof is omitted, but we state that this theorem is correct
  sorry

end clock_hands_angle_3_15_l40_40959


namespace negation_of_universal_prop_l40_40895

variable (P : ∀ x : ℝ, Real.cos x ≤ 1)

theorem negation_of_universal_prop : ∃ x₀ : ℝ, Real.cos x₀ > 1 :=
sorry

end negation_of_universal_prop_l40_40895


namespace min_value_of_f_range_of_a_l40_40880

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x - 1

theorem min_value_of_f : ∃ x ∈ Set.Ioi 0, ∀ y ∈ Set.Ioi 0, f y ≥ f x ∧ f x = -2 * Real.exp (-1) - 1 := 
  sorry

theorem range_of_a {a : ℝ} : (∀ x > 0, f x ≤ 3 * x^2 + 2 * a * x) ↔ a ∈ Set.Ici (-2) := 
  sorry

end min_value_of_f_range_of_a_l40_40880


namespace probability_prime_sum_is_1_9_l40_40226

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end probability_prime_sum_is_1_9_l40_40226


namespace find_theta_l40_40423

theorem find_theta
  (θ : ℝ)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (ha : ∃ k, (2 * Real.cos θ, 2 * Real.sin θ) = (k * 3, k * Real.sqrt 3)) :
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 :=
by
  sorry

end find_theta_l40_40423


namespace exists_another_nice_triple_l40_40737

noncomputable def is_nice_triple (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c ∧ (b - a) = (c - b)) ∧
  (Nat.gcd b a = 1 ∧ Nat.gcd b c = 1) ∧ 
  (∃ k, a * b * c = k^2)

theorem exists_another_nice_triple (a b c : ℕ) 
  (h : is_nice_triple a b c) : ∃ a' b' c', 
  (is_nice_triple a' b' c') ∧ 
  (a' = a ∨ a' = b ∨ a' = c ∨ 
   b' = a ∨ b' = b ∨ b' = c ∨ 
   c' = a ∨ c' = b ∨ c' = c) :=
by sorry

end exists_another_nice_triple_l40_40737


namespace part_one_part_two_part_three_l40_40986

-- Definition of the operation ⊕
def op⊕ (a b : ℚ) : ℚ := a * b + 2 * a

-- Part (1): Prove that 2 ⊕ (-1) = 2
theorem part_one : op⊕ 2 (-1) = 2 :=
by
  sorry

-- Part (2): Prove that -3 ⊕ (-4 ⊕ 1/2) = 24
theorem part_two : op⊕ (-3) (op⊕ (-4) (1 / 2)) = 24 :=
by
  sorry

-- Part (3): Prove that ⊕ is not commutative
theorem part_three : ∃ (a b : ℚ), op⊕ a b ≠ op⊕ b a :=
by
  use 2, -1
  sorry

end part_one_part_two_part_three_l40_40986


namespace area_of_fourth_square_l40_40369

open Real

theorem area_of_fourth_square
  (EF FG GH : ℝ)
  (hEF : EF = 5)
  (hFG : FG = 7)
  (hGH : GH = 8) :
  let EG := sqrt (EF^2 + FG^2)
  let EH := sqrt (EG^2 + GH^2)
  EH^2 = 138 :=
by
  sorry

end area_of_fourth_square_l40_40369


namespace point_not_on_transformed_plane_l40_40161

def point_A : ℝ × ℝ × ℝ := (4, 0, -3)

def plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - 1

def scale_factor : ℝ := 3

def transformed_plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - (scale_factor * 1)

theorem point_not_on_transformed_plane :
  transformed_plane_eq 4 0 (-3) ≠ 0 :=
by
  sorry

end point_not_on_transformed_plane_l40_40161


namespace most_convincing_method_for_relationship_l40_40975

-- Definitions from conditions
def car_owners : ℕ := 300
def car_owners_opposed_policy : ℕ := 116
def non_car_owners : ℕ := 200
def non_car_owners_opposed_policy : ℕ := 121

-- The theorem statement
theorem most_convincing_method_for_relationship : 
  (owning_a_car_related_to_opposing_policy : Bool) :=
by
  -- Proof of the statement
  sorry

end most_convincing_method_for_relationship_l40_40975


namespace range_of_a_plus_b_l40_40609

theorem range_of_a_plus_b (a b : ℝ) (h : |a| + |b| + |a - 1| + |b - 1| ≤ 2) : 
  0 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end range_of_a_plus_b_l40_40609


namespace prime_probability_l40_40249

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def pairs : List (ℕ × ℕ) :=
(primes.erase 2).map (λ p => (2, p))

def prime_sums : List (ℕ × ℕ) :=
pairs.filter (λ ⟨a, b⟩ => is_prime (a + b))

theorem prime_probability :
  let total_pairs := Nat.choose 10 2
  let valid_pairs := prime_sums.length
  (valid_pairs : ℚ) / (total_pairs : ℚ) = 1 / 9 :=
by
  sorry

end prime_probability_l40_40249


namespace inequality_problem_l40_40575

theorem inequality_problem (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by {
  sorry
}

end inequality_problem_l40_40575


namespace smallest_divisor_after_391_l40_40445

theorem smallest_divisor_after_391 (m : ℕ) (h₁ : 1000 ≤ m ∧ m < 10000) (h₂ : Even m) (h₃ : 391 ∣ m) : 
  ∃ d, d > 391 ∧ d ∣ m ∧ ∀ e, 391 < e ∧ e ∣ m → e ≥ d :=
by
  use 441
  sorry

end smallest_divisor_after_391_l40_40445


namespace tan_alpha_solution_l40_40110

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l40_40110


namespace box_cost_coffee_pods_l40_40342

theorem box_cost_coffee_pods :
  ∀ (days : ℕ) (cups_per_day : ℕ) (pods_per_box : ℕ) (total_cost : ℕ), 
  days = 40 → cups_per_day = 3 → pods_per_box = 30 → total_cost = 32 → 
  total_cost / ((days * cups_per_day) / pods_per_box) = 8 := 
by
  intros days cups_per_day pods_per_box total_cost hday hcup hpod hcost
  sorry

end box_cost_coffee_pods_l40_40342


namespace husk_estimation_l40_40350

-- Define the conditions: total rice, sample size, and number of husks in the sample
def total_rice : ℕ := 1520
def sample_size : ℕ := 144
def husks_in_sample : ℕ := 18

-- Define the expected amount of husks in the total batch of rice
def expected_husks : ℕ := 190

-- The theorem stating the problem
theorem husk_estimation 
  (h : (husks_in_sample / sample_size) * total_rice = expected_husks) :
  (18 / 144) * 1520 = 190 := 
sorry

end husk_estimation_l40_40350


namespace find_y_l40_40223

def operation (x y : ℝ) : ℝ := 5 * x - 4 * y + 3 * x * y

theorem find_y : ∃ y : ℝ, operation 4 y = 21 ∧ y = 1 / 8 := by
  sorry

end find_y_l40_40223


namespace number_of_rows_l40_40869

theorem number_of_rows (total_chairs : ℕ) (chairs_per_row : ℕ) (r : ℕ) 
  (h1 : total_chairs = 432) (h2 : chairs_per_row = 16) (h3 : total_chairs = chairs_per_row * r) : r = 27 :=
sorry

end number_of_rows_l40_40869


namespace least_positive_integer_with_12_factors_l40_40010

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l40_40010


namespace cube_inequality_l40_40430

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l40_40430


namespace sum_faces_edges_vertices_eq_26_l40_40664

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l40_40664


namespace fg_of_3_eq_97_l40_40891

def f (x : ℕ) : ℕ := 4 * x - 3
def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem fg_of_3_eq_97 : f (g 3) = 97 := by
  sorry

end fg_of_3_eq_97_l40_40891


namespace find_m_if_f_monotonic_l40_40784

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  4 * x^3 + m * x^2 + (m - 3) * x + n

def is_monotonically_increasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2

theorem find_m_if_f_monotonic (m n : ℝ)
  (h : is_monotonically_increasing_on_ℝ (f m n)) :
  m = 6 :=
sorry

end find_m_if_f_monotonic_l40_40784


namespace probability_sum_is_prime_l40_40241

-- Define the set of the first ten primes
def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define a function that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  ∃ p, p > 1 ∧ p ≤ n ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Define a function that checks if the sum of two primes is a prime
def is_sum_prime (a b : ℕ) : Prop :=
  is_prime (a + b)

-- Define the problem of calculating the required probability
theorem probability_sum_is_prime :
  let totalPairs := (firstTenPrimes.to_finset.powerset.card) / 2
  let validPairs := (({(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)}).card)
  validPairs / totalPairs = (1 : ℕ) / (9 : ℕ) := by
  sorry

end probability_sum_is_prime_l40_40241


namespace paper_plate_cup_cost_l40_40488

variables (P C : ℝ)

theorem paper_plate_cup_cost (h : 100 * P + 200 * C = 6) : 20 * P + 40 * C = 1.20 :=
by sorry

end paper_plate_cup_cost_l40_40488


namespace P_gt_Q_l40_40426

variable (x : ℝ)

def P := x^2 + 2
def Q := 2 * x

theorem P_gt_Q : P x > Q x := by
  sorry

end P_gt_Q_l40_40426


namespace sum_of_cosines_l40_40866

theorem sum_of_cosines :
  (Real.cos (2 * Real.pi / 7) + Real.cos (4 * Real.pi / 7) + Real.cos (6 * Real.pi / 7) = -1 / 2) := sorry

end sum_of_cosines_l40_40866


namespace find_wrongly_written_height_l40_40632

variable (n : ℕ := 35)
variable (average_height_incorrect : ℚ := 184)
variable (actual_height_one_boy : ℚ := 106)
variable (actual_average_height : ℚ := 182)
variable (x : ℚ)

theorem find_wrongly_written_height
  (h_incorrect_total : n * average_height_incorrect = 6440)
  (h_correct_total : n * actual_average_height = 6370) :
  6440 - x + actual_height_one_boy = 6370 ↔ x = 176 := by
  sorry

end find_wrongly_written_height_l40_40632


namespace tan_alpha_sqrt_15_over_15_l40_40117

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l40_40117


namespace part1_part2_l40_40089

-- Define the function y in Lean
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part (1)
theorem part1 (x : ℝ) : y (1/2) x < 0 ↔ -1 < x ∧ x < 2 :=
  sorry

-- Part (2)
theorem part2 (x m : ℝ) : y m x < (1 - m) * x - 1 ↔ 
  (m = 0 → x > 0) ∧ 
  (m > 0 → 0 < x ∧ x < 1 / m) ∧ 
  (m < 0 → x < 1 / m ∨ x > 0) :=
  sorry

end part1_part2_l40_40089


namespace jackson_difference_l40_40911

theorem jackson_difference :
  let Jackson_initial := 500
  let Brandon_initial := 500
  let Meagan_initial := 700
  let Jackson_final := Jackson_initial * 4
  let Brandon_final := Brandon_initial * 0.20
  let Meagan_final := Meagan_initial + (Meagan_initial * 0.50)
  Jackson_final - (Brandon_final + Meagan_final) = 850 :=
by
  sorry

end jackson_difference_l40_40911


namespace values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l40_40760

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 2 * b * x

theorem values_of_a_and_b (h : ∀ x, f x (1 / 3) (-1 / 2) ≤ f 1 (1 / 3) (-1 / 2)) :
  (∃ a b, a = 1 / 3 ∧ b = -1 / 2) :=
sorry

theorem intervals_of_monotonicity (a b : ℝ) (h : ∀ x, f x a b ≤ f 1 a b) :
  (∀ x, (f x a b ≥ 0 ↔ x ≤ -1 / 3 ∨ x ≥ 1) ∧ (f x a b ≤ 0 ↔ -1 / 3 ≤ x ∧ x ≤ 1)) :=
sorry

theorem range_of_a_for_three_roots :
  (∃ a, -1 < a ∧ a < 5 / 27) :=
sorry

end values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l40_40760


namespace width_rectangular_box_5_cm_l40_40530

theorem width_rectangular_box_5_cm 
  (W : ℕ)
  (h_dim_wooden_box : (8 * 10 * 6 * 100 ^ 3) = 480000000) -- dimensions of the wooden box in cm³
  (h_dim_rectangular_box : (4 * W * 6) = (24 * W)) -- dimensions of the rectangular box in cm³
  (h_max_boxes : 4000000 * (24 * W) = 480000000) -- max number of boxes that fit in the wooden box
: 
  W = 5 := 
by
  sorry

end width_rectangular_box_5_cm_l40_40530


namespace tan_alpha_value_l40_40132

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l40_40132


namespace original_stations_l40_40723

theorem original_stations (m n : ℕ) (h : n > 1) (h_equation : n * (2 * m + n - 1) = 58) : m = 14 :=
by
  -- proof omitted
  sorry

end original_stations_l40_40723


namespace total_weekly_pay_l40_40501

theorem total_weekly_pay (Y_pay: ℝ) (X_pay: ℝ) (Y_weekly: Y_pay = 150) (X_weekly: X_pay = 1.2 * Y_pay) : 
  X_pay + Y_pay = 330 :=
by sorry

end total_weekly_pay_l40_40501


namespace clocks_sync_again_in_lcm_days_l40_40536

-- Defining the given conditions based on the problem statement.

-- Arthur's clock gains 15 minutes per day, taking 48 days to gain 12 hours (720 minutes).
def arthur_days : ℕ := 48

-- Oleg's clock gains 12 minutes per day, taking 60 days to gain 12 hours (720 minutes).
def oleg_days : ℕ := 60

-- The problem asks to prove that the situation repeats after 240 days, which is the LCM of 48 and 60.
theorem clocks_sync_again_in_lcm_days : Nat.lcm arthur_days oleg_days = 240 := 
by 
  sorry

end clocks_sync_again_in_lcm_days_l40_40536


namespace each_friend_pays_20_l40_40908

def rent_cottage_cost_per_hour : ℕ := 5
def rent_cottage_hours : ℕ := 8
def total_rent_cost := rent_cottage_cost_per_hour * rent_cottage_hours
def number_of_friends : ℕ := 2
def each_friend_pays := total_rent_cost / number_of_friends

theorem each_friend_pays_20 :
  each_friend_pays = 20 := by
  sorry

end each_friend_pays_20_l40_40908


namespace four_numbers_divisible_by_2310_in_4_digit_range_l40_40532

/--
There exist exactly 4 numbers within the range of 1000 to 9999 that are divisible by 2310.
-/
theorem four_numbers_divisible_by_2310_in_4_digit_range :
  ∃ n₁ n₂ n₃ n₄,
    1000 ≤ n₁ ∧ n₁ ≤ 9999 ∧ n₁ % 2310 = 0 ∧
    1000 ≤ n₂ ∧ n₂ ≤ 9999 ∧ n₂ % 2310 = 0 ∧ n₁ < n₂ ∧
    1000 ≤ n₃ ∧ n₃ ≤ 9999 ∧ n₃ % 2310 = 0 ∧ n₂ < n₃ ∧
    1000 ≤ n₄ ∧ n₄ ≤ 9999 ∧ n₄ % 2310 = 0 ∧ n₃ < n₄ ∧
    ∀ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 2310 = 0 → (n = n₁ ∨ n = n₂ ∨ n = n₃ ∨ n = n₄) :=
by
  sorry

end four_numbers_divisible_by_2310_in_4_digit_range_l40_40532


namespace sin_theta_correct_l40_40572

noncomputable def sin_theta (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real :=
  -Real.sqrt 2 / 2

theorem sin_theta_correct (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan (Real.arctan (-a)) = -a) : sin_theta a h1 h2 = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_theta_correct_l40_40572


namespace drums_filled_per_day_l40_40848

-- Definition of given conditions
def pickers : ℕ := 266
def total_drums : ℕ := 90
def total_days : ℕ := 5

-- Statement to prove
theorem drums_filled_per_day : (total_drums / total_days) = 18 := by
  sorry

end drums_filled_per_day_l40_40848


namespace find_h_l40_40793

theorem find_h (h : ℝ) (j k : ℝ) 
  (y_eq1 : ∀ x : ℝ, (4 * (x - h)^2 + j) = 2030)
  (y_eq2 : ∀ x : ℝ, (5 * (x - h)^2 + k) = 2040)
  (int_xint1 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (4 * x1 * x2 = 2032) )
  (int_xint2 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (5 * x1 * x2 = 2040) ) :
  h = 20.5 :=
by
  sorry

end find_h_l40_40793


namespace evaluate_polynomial_l40_40276

theorem evaluate_polynomial
  (x : ℝ)
  (h1 : x^2 - 3 * x - 9 = 0)
  (h2 : 0 < x)
  : x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = 8 :=
sorry

end evaluate_polynomial_l40_40276


namespace total_sales_l40_40715

theorem total_sales (S : ℕ) (h1 : (1 / 3 : ℚ) * S + (1 / 4 : ℚ) * S = (1 - (1 / 3 + 1 / 4)) * S + 15) : S = 36 :=
by
  sorry

end total_sales_l40_40715


namespace lcm_48_180_eq_720_l40_40553

variable (a b : ℕ)

def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem lcm_48_180_eq_720 : lcm 48 180 = 720 := by
  -- The proof is omitted
  sorry

end lcm_48_180_eq_720_l40_40553


namespace workers_planted_33_walnut_trees_l40_40802

def initial_walnut_trees : ℕ := 22
def total_walnut_trees_after_planting : ℕ := 55
def walnut_trees_planted (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem workers_planted_33_walnut_trees :
  walnut_trees_planted initial_walnut_trees total_walnut_trees_after_planting = 33 :=
by
  unfold walnut_trees_planted
  rfl

end workers_planted_33_walnut_trees_l40_40802


namespace least_positive_integer_with_12_factors_is_96_l40_40017

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l40_40017


namespace lcm_48_180_l40_40564

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l40_40564


namespace min_value_2x_y_l40_40418

noncomputable def min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (heq : Real.log (x + 2 * y) = Real.log x + Real.log y) : ℝ :=
  2 * x + y

theorem min_value_2x_y : ∀ (x y : ℝ), 0 < x → 0 < y → Real.log (x + 2 * y) = Real.log x + Real.log y → 2 * x + y ≥ 9 :=
by
  intros x y hx hy heq
  sorry

end min_value_2x_y_l40_40418


namespace probability_of_blue_face_l40_40808

theorem probability_of_blue_face (total_faces blue_faces : ℕ) (h_total : total_faces = 8) (h_blue : blue_faces = 5) : 
  blue_faces / total_faces = 5 / 8 :=
by
  sorry

end probability_of_blue_face_l40_40808


namespace prob_prime_sum_l40_40257

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := nat.prime n

def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (List.product l l).filter (λ p, p.1 < p.2)

def prime_sum_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  valid_pairs l |>.filter (λ p, is_prime (p.1 + p.2))

theorem prob_prime_sum :
  prime_sum_pairs primes.length / (valid_pairs primes).length = 1 / 9 :=
by
  sorry -- Proof is not required

end prob_prime_sum_l40_40257


namespace sum_of_faces_edges_vertices_rectangular_prism_l40_40684

theorem sum_of_faces_edges_vertices_rectangular_prism :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  let faces := 6
  let edges := 12
  let vertices := 8
  sorry

end sum_of_faces_edges_vertices_rectangular_prism_l40_40684


namespace cos420_add_sin330_l40_40275

theorem cos420_add_sin330 : Real.cos (420 * Real.pi / 180) + Real.sin (330 * Real.pi / 180) = 0 := 
by
  sorry

end cos420_add_sin330_l40_40275


namespace min_value_of_f_l40_40578

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  2 * x^3 - 6 * x^2 + m

theorem min_value_of_f :
  ∀ (m : ℝ),
    f 0 m = 3 →
    ∃ x min, x ∈ Set.Icc (-2:ℝ) (2:ℝ) ∧ min = f x m ∧ min = -37 :=
by
  intros m h
  have h' : f 0 m = 3 := h
  -- Proof omitted.
  sorry

end min_value_of_f_l40_40578


namespace num_elements_in_set_S_l40_40566

theorem num_elements_in_set_S (n : ℕ) (hn : n ≥ 1) :
  let S (n : ℕ) := {k : ℕ | k > n ∧ k ∣ (30 * n - 1)}
  let S_union := ⋃ i : ℕ, S i
  (S_union.filter (< 2016)).card = 536 :=
sorry

end num_elements_in_set_S_l40_40566


namespace solve_for_y_l40_40472

theorem solve_for_y (y : ℕ) (h : 5 * (2^y) = 320) : y = 6 :=
by sorry

end solve_for_y_l40_40472


namespace projectile_reaches_45_feet_first_time_l40_40179

theorem projectile_reaches_45_feet_first_time :
  ∃ t : ℝ, (-20 * t^2 + 90 * t = 45) ∧ abs (t - 0.9) < 0.1 := sorry

end projectile_reaches_45_feet_first_time_l40_40179


namespace Jean_money_l40_40322

theorem Jean_money (x : ℝ) (h1 : 3 * x + x = 76): 
  3 * x = 57 := 
by
  sorry

end Jean_money_l40_40322


namespace minimum_seats_l40_40801

-- Condition: 150 seats in a row.
def seats : ℕ := 150

-- Assertion: The fewest number of seats that must be occupied so that any additional person seated must sit next to someone.
def minOccupiedSeats : ℕ := 50

theorem minimum_seats (s : ℕ) (m : ℕ) (h_seats : s = 150) (h_min : m = 50) :
  (∀ x, x = 150 → ∀ n, n ≥ 0 ∧ n ≤ m → 
    ∃ y, y ≥ 0 ∧ y ≤ x ∧ ∀ z, z = n + 1 → ∃ w, w ≥ 0 ∧ w ≤ x ∧ w = n ∨ w = n + 1) := 
sorry

end minimum_seats_l40_40801


namespace lcm_48_180_value_l40_40551

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l40_40551


namespace man_l40_40385

theorem man's_age_twice_son (S M Y : ℕ) (h1 : M = S + 26) (h2 : S = 24) (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  sorry

end man_l40_40385


namespace center_of_circle_l40_40286

theorem center_of_circle (x y : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 - 6 * y = 12) → ((x + 2)^2 + (y - 3)^2 = 25) :=
by
  sorry

end center_of_circle_l40_40286


namespace pentagon_square_ratio_l40_40212

theorem pentagon_square_ratio (s p : ℕ) (h1 : 4 * s = 20) (h2 : 5 * p = 20) :
  p / s = 4 / 5 :=
by
  sorry

end pentagon_square_ratio_l40_40212


namespace downstream_speed_l40_40979

variable (V_u V_s V_d : ℝ)

theorem downstream_speed (h1 : V_u = 22) (h2 : V_s = 32) (h3 : V_s = (V_u + V_d) / 2) : V_d = 42 :=
sorry

end downstream_speed_l40_40979


namespace total_tickets_sold_l40_40204

theorem total_tickets_sold 
(adult_ticket_price : ℕ) (child_ticket_price : ℕ) 
(total_revenue : ℕ) (adult_tickets_sold : ℕ) 
(child_tickets_sold : ℕ) (total_tickets : ℕ) : 
adult_ticket_price = 5 → 
child_ticket_price = 2 → 
total_revenue = 275 → 
adult_tickets_sold = 35 → 
(child_tickets_sold * child_ticket_price) + (adult_tickets_sold * adult_ticket_price) = total_revenue →
total_tickets = adult_tickets_sold + child_tickets_sold →
total_tickets = 85 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_tickets_sold_l40_40204


namespace stratified_sampling_grade12_l40_40724

theorem stratified_sampling_grade12 (total_students grade12_students sample_size : ℕ) 
  (h_total : total_students = 2000) 
  (h_grade12 : grade12_students = 700) 
  (h_sample : sample_size = 400) : 
  (sample_size * grade12_students) / total_students = 140 := 
by 
  sorry

end stratified_sampling_grade12_l40_40724


namespace two_a_minus_two_d_eq_zero_l40_40449

noncomputable def g (a b c d x : ℝ) : ℝ := (2 * a * x - b) / (c * x - 2 * d)

theorem two_a_minus_two_d_eq_zero (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : ∀ x : ℝ, (g a a c d (g a b c d x)) = x) : 2 * a - 2 * d = 0 :=
sorry

end two_a_minus_two_d_eq_zero_l40_40449


namespace tan_alpha_value_l40_40128

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l40_40128


namespace problem_solution_l40_40333

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.rpow 3 (1 / 3)
noncomputable def c : ℝ := Real.log 2 / Real.log 3

theorem problem_solution : c < a ∧ a < b := 
by
  sorry

end problem_solution_l40_40333


namespace symmetric_origin_a_minus_b_l40_40294

noncomputable def A (a : ℝ) := (a, -2)
noncomputable def B (b : ℝ) := (4, b)
def symmetric (p q : ℝ × ℝ) : Prop := (q.1 = -p.1) ∧ (q.2 = -p.2)

theorem symmetric_origin_a_minus_b (a b : ℝ) (hA : A a = (-4, -2)) (hB : B b = (4, 2)) :
  a - b = -6 := by
  sorry

end symmetric_origin_a_minus_b_l40_40294


namespace probability_xi_l40_40573

noncomputable def xi_distribution (k : ℕ) : ℚ :=
  if h : k > 0 then 1 / (2 : ℚ)^k else 0

theorem probability_xi (h : ∀ k : ℕ, k > 0 → xi_distribution k = 1 / (2 : ℚ)^k) :
  (xi_distribution 3 + xi_distribution 4) = 3 / 16 :=
by
  sorry

end probability_xi_l40_40573


namespace last_digit_is_zero_last_ten_digits_are_zero_l40_40198

-- Condition: The product includes a factor of 10
def includes_factor_of_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10

-- Conclusion: The last digit of the product must be 0
theorem last_digit_is_zero (n : ℕ) (h : includes_factor_of_10 n) : 
  n % 10 = 0 :=
sorry

-- Condition: The product includes the factors \(5^{10}\) and \(2^{10}\)
def includes_10_to_the_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10^10

-- Conclusion: The last ten digits of the product must be 0000000000
theorem last_ten_digits_are_zero (n : ℕ) (h : includes_10_to_the_10 n) : 
  n % 10^10 = 0 :=
sorry

end last_digit_is_zero_last_ten_digits_are_zero_l40_40198


namespace probability_product_divisible_by_four_l40_40790

open Finset

theorem probability_product_divisible_by_four :
  (∃ (favorable_pairs total_pairs : ℕ), favorable_pairs = 70 ∧ total_pairs = 190 ∧ favorable_pairs / total_pairs = 7 / 19) := 
sorry

end probability_product_divisible_by_four_l40_40790


namespace excluded_students_count_l40_40942

theorem excluded_students_count 
  (N : ℕ) 
  (x : ℕ) 
  (average_marks : ℕ) 
  (excluded_average_marks : ℕ) 
  (remaining_average_marks : ℕ) 
  (total_students : ℕ)
  (h1 : average_marks = 80)
  (h2 : excluded_average_marks = 70)
  (h3 : remaining_average_marks = 90)
  (h4 : total_students = 10)
  (h5 : N = total_students)
  (h6 : 80 * N = 70 * x + 90 * (N - x))
  : x = 5 :=
by
  sorry

end excluded_students_count_l40_40942


namespace kishore_savings_l40_40062

noncomputable def rent := 5000
noncomputable def milk := 1500
noncomputable def groceries := 4500
noncomputable def education := 2500
noncomputable def petrol := 2000
noncomputable def miscellaneous := 700
noncomputable def total_expenses := rent + milk + groceries + education + petrol + miscellaneous
noncomputable def salary : ℝ := total_expenses / 0.9 -- given that savings is 10% of salary

theorem kishore_savings : (salary * 0.1) = 1800 :=
by
  sorry

end kishore_savings_l40_40062


namespace sum_of_faces_edges_vertices_l40_40660

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l40_40660


namespace miles_per_hour_l40_40063

theorem miles_per_hour (total_distance : ℕ) (total_hours : ℕ) (h1 : total_distance = 81) (h2 : total_hours = 3) :
  total_distance / total_hours = 27 :=
by
  sorry

end miles_per_hour_l40_40063


namespace female_managers_count_l40_40592

variable (E M F FM : ℕ)

-- Conditions
def female_employees : Prop := F = 750
def fraction_managers : Prop := (2 / 5 : ℚ) * E = FM + (2 / 5 : ℚ) * M
def total_employees : Prop := E = M + F

-- Proof goal
theorem female_managers_count (h1 : female_employees F) 
                              (h2 : fraction_managers E M FM) 
                              (h3 : total_employees E M F) : 
  FM = 300 := 
sorry

end female_managers_count_l40_40592


namespace f_neg1_plus_f_2_l40_40756

def f (x : ℤ) : ℤ :=
  if x ≤ 0 then 4 * x else 2 * x

theorem f_neg1_plus_f_2 : f (-1) + f 2 = 0 := 
by
  -- Definition of f is provided above and conditions are met in that.
  sorry

end f_neg1_plus_f_2_l40_40756


namespace expression_equality_l40_40741

theorem expression_equality :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := 
  sorry

end expression_equality_l40_40741


namespace proof_M1M2_product_l40_40330

theorem proof_M1M2_product : 
  (∀ x, (45 * x - 34) / (x^2 - 4 * x + 3) = M_1 / (x - 1) + M_2 / (x - 3)) →
  M_1 * M_2 = -1111 / 4 := 
by
  sorry

end proof_M1M2_product_l40_40330


namespace find_angle_A_l40_40151

variable {A B C a b c : ℝ}
variable {triangle_ABC : Prop}

theorem find_angle_A
  (h1 : a^2 + c^2 = b^2 + 2 * a * c * Real.cos C)
  (h2 : a = 2 * b * Real.sin A)
  (h3 : Real.cos B = Real.cos C)
  (h_triangle_angles : triangle_ABC) : A = 2 * Real.pi / 3 := 
by
  sorry

end find_angle_A_l40_40151


namespace sum_faces_edges_vertices_l40_40670

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end sum_faces_edges_vertices_l40_40670


namespace gain_amount_l40_40768

theorem gain_amount (gain_percent : ℝ) (gain : ℝ) (amount : ℝ) 
  (h_gain_percent : gain_percent = 1) 
  (h_gain : gain = 0.70) 
  : amount = 70 :=
by
  sorry

end gain_amount_l40_40768


namespace probability_sum_is_prime_l40_40243

-- Define the set of the first ten primes
def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define a function that checks if a number is prime
def is_prime (n : ℕ) : Prop :=
  ∃ p, p > 1 ∧ p ≤ n ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Define a function that checks if the sum of two primes is a prime
def is_sum_prime (a b : ℕ) : Prop :=
  is_prime (a + b)

-- Define the problem of calculating the required probability
theorem probability_sum_is_prime :
  let totalPairs := (firstTenPrimes.to_finset.powerset.card) / 2
  let validPairs := (({(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)}).card)
  validPairs / totalPairs = (1 : ℕ) / (9 : ℕ) := by
  sorry

end probability_sum_is_prime_l40_40243


namespace political_exam_pass_l40_40845

-- Define the students' statements.
def A_statement (C_passed : Prop) : Prop := C_passed
def B_statement (B_passed : Prop) : Prop := ¬ B_passed
def C_statement (A_statement : Prop) : Prop := A_statement

-- Define the problem conditions.
def condition_1 (A_passed B_passed C_passed : Prop) : Prop := ¬A_passed ∨ ¬B_passed ∨ ¬C_passed
def condition_2 (A_passed B_passed C_passed : Prop) := A_statement C_passed
def condition_3 (A_passed B_passed C_passed : Prop) := B_statement B_passed
def condition_4 (A_passed B_passed C_passed : Prop) := C_statement (A_statement C_passed)
def condition_5 (A_statement_true B_statement_true C_statement_true : Prop) : Prop := 
  (¬A_statement_true ∧ B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ ¬B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ B_statement_true ∧ ¬C_statement_true)

-- Define the proof problem.
theorem political_exam_pass : 
  ∀ (A_passed B_passed C_passed : Prop),
  condition_1 A_passed B_passed C_passed →
  condition_2 A_passed B_passed C_passed →
  condition_3 A_passed B_passed C_passed →
  condition_4 A_passed B_passed C_passed →
  ∃ (A_statement_true B_statement_true C_statement_true : Prop), 
  condition_5 A_statement_true B_statement_true C_statement_true →
  ¬A_passed
:= by { sorry }

end political_exam_pass_l40_40845


namespace total_people_l40_40523

theorem total_people (N B : ℕ) (h1 : N = 4 * B + 10) (h2 : N = 5 * B + 1) : N = 46 := by
  -- The proof will follow from the conditions, but it is not required in this script.
  sorry

end total_people_l40_40523


namespace fence_perimeter_l40_40497

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end fence_perimeter_l40_40497


namespace abs_non_positive_eq_zero_l40_40029

theorem abs_non_positive_eq_zero (y : ℚ) (h : |4 * y - 7| ≤ 0) : y = 7 / 4 :=
by
  sorry

end abs_non_positive_eq_zero_l40_40029


namespace rhombus_diagonal_sum_l40_40371

theorem rhombus_diagonal_sum (e f : ℝ) (h1: e^2 + f^2 = 16) (h2: 0 < e ∧ 0 < f):
  e + f = 5 :=
by
  sorry

end rhombus_diagonal_sum_l40_40371


namespace range_of_x_l40_40150

noncomputable def y (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem range_of_x : ∀ x : ℝ, (y x ≠ 0) → x ≠ 1 := by
  intro x h
  sorry

end range_of_x_l40_40150


namespace distinct_names_impossible_l40_40441

-- Define the alphabet
inductive Letter
| a | u | o | e

-- Simplified form of words in the Mumbo-Jumbo language
def simplified_form : List Letter → List Letter
| [] => []
| (Letter.e :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.a :: xs) => simplified_form (Letter.a :: Letter.a :: xs)
| (Letter.o :: Letter.o :: Letter.o :: Letter.o :: xs) => simplified_form xs
| (Letter.a :: Letter.a :: Letter.a :: Letter.u :: xs) => simplified_form (Letter.u :: xs)
| (x :: xs) => x :: simplified_form xs

-- Number of possible names
def num_possible_names : ℕ := 343

-- Number of tribe members
def num_tribe_members : ℕ := 400

theorem distinct_names_impossible : num_possible_names < num_tribe_members :=
by
  -- Skipping the proof with 'sorry'
  sorry

end distinct_names_impossible_l40_40441


namespace exterior_angle_parallel_lines_l40_40149

theorem exterior_angle_parallel_lines
  (k l : Prop) 
  (triangle_has_angles : ∃ (a b c : ℝ), a = 40 ∧ b = 40 ∧ c = 100 ∧ a + b + c = 180)
  (exterior_angle_eq : ∀ (y : ℝ), y = 180 - 100) :
  ∃ (x : ℝ), x = 80 :=
by
  sorry

end exterior_angle_parallel_lines_l40_40149


namespace problem1_problem2_l40_40827

-- define problem 1 as a theorem
theorem problem1: 
  ((-0.4) * (-0.8) * (-1.25) * 2.5 = -1) :=
  sorry

-- define problem 2 as a theorem
theorem problem2: 
  ((- (5:ℚ) / 8) * (3 / 14) * ((-16) / 5) * ((-7) / 6) = -1 / 2) :=
  sorry

end problem1_problem2_l40_40827


namespace least_positive_integer_with_12_factors_is_72_l40_40005

-- Definition of having exactly 12 factors for a positive integer
def has_exactly_12_factors (n : ℕ) : Prop := n > 0 ∧ (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card = 12

-- Statement of the problem in Lean 4
theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, has_exactly_12_factors n ∧ ∀ m : ℕ, has_exactly_12_factors m → n ≤ m :=
begin
  use 72,
  split,
  { -- Prove that 72 has exactly 12 factors
    sorry },
  { -- Prove that 72 is the least such integer
    sorry }
end

end least_positive_integer_with_12_factors_is_72_l40_40005


namespace rafael_total_net_pay_is_878_l40_40623

noncomputable def rafaelNetPay 
  (mondayHours tuesdayHours wednesdayHours thursdayHours fridayHours: ℕ)
  (totalHoursBonus taxDeduction taxCredit: ℝ)
  (wagePerHour overtimeWagePerHour: ℝ): ℝ :=
let weeklyHours := mondayHours + tuesdayHours + wednesdayHours + thursdayHours + fridayHours 
in let regularHoursMonday := min mondayHours 8 
in let overtimeHoursMonday := max (mondayHours - 8) 0 
in let regularPayMonday := regularHoursMonday * wagePerHour 
in let overtimePayMonday := overtimeHoursMonday * overtimeWagePerHour 
in let regularPayTWF := tuesdayHours * wagePerHour + wednesdayHours * wagePerHour + thursdayHours * wagePerHour + fridayHours * wagePerHour 
in let totalPayPreBonus := regularPayMonday + overtimePayMonday + regularPayTWF 
in let totalPayWithBonus := totalPayPreBonus + totalHoursBonus 
in let taxOwed := totalPayWithBonus * taxDeduction 
in let taxAfterCredit := max (taxOwed - taxCredit) 0 
in totalPayWithBonus - taxAfterCredit

theorem rafael_total_net_pay_is_878:
rafaelNetPay 10 8 8 8 6 100 0.1 50 20 30 = 878 := 
sorry

end rafael_total_net_pay_is_878_l40_40623


namespace bananas_each_child_l40_40455

theorem bananas_each_child (x : ℕ) (B : ℕ) 
  (h1 : 660 * x = B)
  (h2 : 330 * (x + 2) = B) : 
  x = 2 := 
by 
  sorry

end bananas_each_child_l40_40455


namespace range_of_p_nonnegative_range_of_p_all_values_range_of_p_l40_40944

def p (x : ℝ) : ℝ := x^4 - 6 * x^2 + 9

theorem range_of_p_nonnegative (x : ℝ) (hx : 0 ≤ x) : 
  ∃ y, y = p x ∧ 0 ≤ y := 
sorry

theorem range_of_p_all_values (y : ℝ) : 
  0 ≤ y → (∃ x, 0 ≤ x ∧ p x = y) :=
sorry

theorem range_of_p (x : ℝ) (hx : 0 ≤ x) : 
  ∀ y, (∃ x, 0 ≤ x ∧ p x = y) ↔ (0 ≤ y) :=
sorry

end range_of_p_nonnegative_range_of_p_all_values_range_of_p_l40_40944


namespace problem_lean_statement_l40_40923

def P (x : ℝ) : ℝ := x^2 - 3*x - 9

theorem problem_lean_statement :
  let a := 61
  let b := 109
  let c := 621
  let d := 39
  let e := 20
  a + b + c + d + e = 850 := 
by
  sorry

end problem_lean_statement_l40_40923


namespace solve_system_of_equations_l40_40614

theorem solve_system_of_equations
  (x y : ℚ)
  (h1 : 5 * x - 3 * y = -7)
  (h2 : 4 * x + 6 * y = 34) :
  x = 10 / 7 ∧ y = 33 / 7 :=
by
  sorry

end solve_system_of_equations_l40_40614


namespace necessary_but_not_sufficient_condition_l40_40363

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (h : ¬p) : p ∨ q ↔ true :=
by
  sorry

end necessary_but_not_sufficient_condition_l40_40363


namespace numerator_of_fraction_l40_40943

/-- 
Given:
1. The denominator of a fraction is 7 less than 3 times the numerator.
2. The fraction is equivalent to 2/5.
Prove that the numerator of the fraction is 14.
-/
theorem numerator_of_fraction {x : ℕ} (h : x / (3 * x - 7) = 2 / 5) : x = 14 :=
  sorry

end numerator_of_fraction_l40_40943


namespace correct_histogram_height_representation_l40_40964

   def isCorrectHeightRepresentation (heightRep : String) : Prop :=
     heightRep = "ratio of the frequency of individuals in that group within the sample to the class interval"

   theorem correct_histogram_height_representation :
     isCorrectHeightRepresentation "ratio of the frequency of individuals in that group within the sample to the class interval" :=
   by 
     sorry
   
end correct_histogram_height_representation_l40_40964


namespace number_of_tickets_bought_l40_40052

noncomputable def ticketCost : ℕ := 5
noncomputable def popcornCost : ℕ := (80 * ticketCost) / 100
noncomputable def sodaCost : ℕ := (50 * popcornCost) / 100
noncomputable def totalSpent : ℕ := 36
noncomputable def numberOfPopcorns : ℕ := 2 
noncomputable def numberOfSodas : ℕ := 4

theorem number_of_tickets_bought : 
  (totalSpent - (numberOfPopcorns * popcornCost + numberOfSodas * sodaCost)) = 4 * ticketCost :=
by
  sorry

end number_of_tickets_bought_l40_40052


namespace no_sensor_in_option_B_l40_40965

/-- Define the technologies and whether they involve sensors --/
def technology_involves_sensor (opt : String) : Prop :=
  opt = "A" ∨ opt = "C" ∨ opt = "D"

theorem no_sensor_in_option_B :
  ¬ technology_involves_sensor "B" :=
by
  -- We assume the proof for the sake of this example.
  sorry

end no_sensor_in_option_B_l40_40965


namespace no_two_primes_sum_to_10003_l40_40905

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the specific numbers involved
def even_prime : ℕ := 2
def target_number : ℕ := 10003
def candidate : ℕ := target_number - even_prime

-- State the main proposition in question
theorem no_two_primes_sum_to_10003 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = target_number :=
sorry

end no_two_primes_sum_to_10003_l40_40905


namespace A_work_days_l40_40380

theorem A_work_days (x : ℝ) :
  (1 / x + 1 / 6 + 1 / 12 = 7 / 24) → x = 24 :=
by
  intro h
  sorry

end A_work_days_l40_40380


namespace find_n_l40_40752

theorem find_n (n : ℕ) 
    (h : 6 * 4 * 3 * n = Nat.factorial 8) : n = 560 := 
sorry

end find_n_l40_40752


namespace prime_sum_probability_l40_40274

/-- Statement for the proof problem -/
theorem prime_sum_probability:
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} in 
  (∃ (x y ∈ primes), (x ≠ y) ∧ (x + y) ∈ primes) →
  (∃ (probability : ℚ), probability = 1/9) :=
by sorry

end prime_sum_probability_l40_40274


namespace total_cupcakes_l40_40396

noncomputable def cupcakesForBonnie : ℕ := 24
noncomputable def cupcakesPerDay : ℕ := 60
noncomputable def days : ℕ := 2

theorem total_cupcakes : (cupcakesPerDay * days + cupcakesForBonnie) = 144 := 
by
  sorry

end total_cupcakes_l40_40396


namespace least_sum_of_factors_l40_40306

theorem least_sum_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2400) : a + b = 98 :=
sorry

end least_sum_of_factors_l40_40306


namespace minimize_fraction_sum_l40_40783

theorem minimize_fraction_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 :=
sorry

end minimize_fraction_sum_l40_40783


namespace probability_sum_two_primes_is_prime_l40_40262

open Finset

-- First 10 prime numbers
def firstTenPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Function to sum of two primes
def sum_of_two_primes_is_prime (a b : ℕ) : Prop := is_prime (a + b)

-- Function to compute the probability
def probability_two_prime_sum_is_prime : ℚ :=
  let pairs := firstTenPrimes.pairs
  let prime_pairs := pairs.filter (λ pair, sum_of_two_primes_is_prime pair.1 pair.2)
  (prime_pairs.card : ℚ) / pairs.card

theorem probability_sum_two_primes_is_prime :
  probability_two_prime_sum_is_prime = 1 / 9 :=
by sorry

end probability_sum_two_primes_is_prime_l40_40262


namespace find_b1_b7_b10_value_l40_40040

open Classical

theorem find_b1_b7_b10_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith_seq : ∀ n m : ℕ, a n + a m = 2 * a ((n + m) / 2))
  (h_geom_seq : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r)
  (a3_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (b6_a6_eq : b 6 = a 6)
  (non_zero_seq : ∀ n : ℕ, a n ≠ 0) :
  b 1 * b 7 * b 10 = 8 := 
by 
  sorry

end find_b1_b7_b10_value_l40_40040


namespace acme_profit_l40_40393

-- Define the given problem conditions
def initial_outlay : ℝ := 12450
def cost_per_set : ℝ := 20.75
def selling_price_per_set : ℝ := 50
def num_sets : ℝ := 950

-- Define the total revenue and total manufacturing costs
def total_revenue : ℝ := num_sets * selling_price_per_set
def total_cost : ℝ := initial_outlay + (cost_per_set * num_sets)

-- State the profit calculation and the expected result
def profit : ℝ := total_revenue - total_cost

theorem acme_profit : profit = 15337.50 := by
  -- Proof goes here
  sorry

end acme_profit_l40_40393


namespace prime_sum_probability_l40_40235

open Set

noncomputable def first_ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_pairs : Finset (ℕ × ℕ) :=
  first_ten_primes.product first_ten_primes.filter (λ p, is_prime (p.1 + p.2))

theorem prime_sum_probability :
  (valid_pairs.card = 6) → ((first_ten_primes.card.choose 2 : ℚ) = 45) →
  Rat.mk valid_pairs.card (first_ten_primes.card.choose 2) = 2 / 15 :=
sorry

end prime_sum_probability_l40_40235


namespace tallest_vs_shortest_height_difference_l40_40772

-- Define the heights of the trees
def pine_tree_height := 12 + 4/5
def birch_tree_height := 18 + 1/2
def maple_tree_height := 14 + 3/5

-- Calculate improper fractions
def pine_tree_improper := 64 / 5
def birch_tree_improper := 41 / 2  -- This is 82/4 but not simplified here
def maple_tree_improper := 73 / 5

-- Calculate height difference
def height_difference := (82 / 4) - (64 / 5)

-- The statement that needs to be proven
theorem tallest_vs_shortest_height_difference : height_difference = 7 + 7 / 10 :=
by 
  sorry

end tallest_vs_shortest_height_difference_l40_40772


namespace exists_multiple_2003_no_restricted_digits_l40_40622

theorem exists_multiple_2003_no_restricted_digits : ∃ n : ℕ, 
  (∃ k : ℕ, n = 2003 * k) ∧ 
  (n < 10^11) ∧ 
  (∀ d ∈ (n.digits 10), d ∈ {0, 1, 8, 9}) :=
by sorry

end exists_multiple_2003_no_restricted_digits_l40_40622


namespace solution_set_inequality_l40_40297

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_inequality :
  (∀ x : ℝ, deriv^[2] f x > f x) ∧ f 1 = real.exp 1 → 
  {x : ℝ | f x < real.exp x} = set.Iio 1 :=
by {
  sorry
}

end solution_set_inequality_l40_40297


namespace arrange_scores_l40_40146

variable {K Q M S : ℝ}

theorem arrange_scores (h1 : Q > K) (h2 : M > S) (h3 : S < max Q (max M K)) : S < M ∧ M < Q := by
  sorry

end arrange_scores_l40_40146


namespace slices_left_for_tomorrow_is_four_l40_40545

def initial_slices : ℕ := 12
def lunch_slices : ℕ := initial_slices / 2
def remaining_slices_after_lunch : ℕ := initial_slices - lunch_slices
def dinner_slices : ℕ := remaining_slices_after_lunch / 3
def slices_left_for_tomorrow : ℕ := remaining_slices_after_lunch - dinner_slices

theorem slices_left_for_tomorrow_is_four : slices_left_for_tomorrow = 4 := by
  sorry

end slices_left_for_tomorrow_is_four_l40_40545


namespace triangle_angle_ratio_l40_40925

theorem triangle_angle_ratio (A B C D : Type*) 
  (α β γ δ : ℝ) -- α = ∠BAC, β = ∠ABC, γ = ∠BCA, δ = external angles
  (h1 : α + β + γ = 180)
  (h2 : δ = α + γ)
  (h3 : δ = β + γ) : (2 * 180 - (α + β)) / (α + β) = 2 :=
by
  sorry

end triangle_angle_ratio_l40_40925


namespace cube_surface_area_l40_40941

theorem cube_surface_area (Q : ℝ) (a : ℝ) (H : (3 * a^2 * Real.sqrt 3) / 2 = Q) :
    (6 * (a * Real.sqrt 2) ^ 2) = (8 * Q * Real.sqrt 3) / 3 :=
by
  sorry

end cube_surface_area_l40_40941


namespace rectangle_length_l40_40607

theorem rectangle_length (side_of_square : ℕ) (width_of_rectangle : ℕ) (same_wire_length : ℕ) 
(side_eq : side_of_square = 12) (width_eq : width_of_rectangle = 6) 
(square_perimeter : same_wire_length = 4 * side_of_square) :
  ∃ (length_of_rectangle : ℕ), 2 * (length_of_rectangle + width_of_rectangle) = same_wire_length ∧ length_of_rectangle = 18 :=
by
  sorry

end rectangle_length_l40_40607


namespace least_positive_integer_with_12_factors_is_96_l40_40019

theorem least_positive_integer_with_12_factors_is_96 :
  ∃ n : ℕ, (n = 96 ∧ (∀ m : ℕ, (m > 0 ∧ ∀ d : ℕ, d ∣ m → d > 0 → d < m → (∃ k : ℕ, d = k ∨ k * d = m) → k > 1 → m > n))) ∧ (∃ m : ℕ, m = 12 ∧ (∀ d : ℕ, d ∣ m → d ∈ {1, m} ∨ ∀ k : ℕ, k * d = m → d = k ∨ d < k)).
sorry

end least_positive_integer_with_12_factors_is_96_l40_40019


namespace no_nat_solution_for_exp_eq_l40_40463

theorem no_nat_solution_for_exp_eq (n x y z : ℕ) (hn : n > 1) (hx : x ≤ n) (hy : y ≤ n) :
  ¬ (x^n + y^n = z^n) :=
by
  sorry

end no_nat_solution_for_exp_eq_l40_40463


namespace relationship_y1_y2_y3_l40_40339

-- Define the quadratic function
def quadratic (x : ℝ) (k : ℝ) : ℝ :=
  -(x - 2) ^ 2 + k

-- Define the points A, B, and C
def A (y1 k : ℝ) := ∃ y1, quadratic (-1 / 2) k = y1
def B (y2 k : ℝ) := ∃ y2, quadratic (1) k = y2
def C (y3 k : ℝ) := ∃ y3, quadratic (4) k = y3

theorem relationship_y1_y2_y3 (y1 y2 y3 k: ℝ)
  (hA : A y1 k)
  (hB : B y2 k)
  (hC : C y3 k) :
  y1 < y3 ∧ y3 < y2 :=
  sorry

end relationship_y1_y2_y3_l40_40339


namespace correct_option_is_B_l40_40513

-- Define the operations as hypotheses
def option_A (a : ℤ) : Prop := (a^2 + a^3 = a^5)
def option_B (a : ℤ) : Prop := ((a^2)^3 = a^6)
def option_C (a : ℤ) : Prop := (a^2 * a^3 = a^6)
def option_D (a : ℤ) : Prop := (6 * a^6 - 2 * a^3 = 3 * a^3)

-- Prove that option B is correct
theorem correct_option_is_B (a : ℤ) : option_B a :=
by
  unfold option_B
  sorry

end correct_option_is_B_l40_40513


namespace hamburgers_made_l40_40208

theorem hamburgers_made (initial_hamburgers additional_hamburgers total_hamburgers : ℝ)
    (h_initial : initial_hamburgers = 9.0)
    (h_additional : additional_hamburgers = 3.0)
    (h_total : total_hamburgers = initial_hamburgers + additional_hamburgers) :
    total_hamburgers = 12.0 :=
by
    sorry

end hamburgers_made_l40_40208


namespace lcm_48_180_value_l40_40550

def lcm_48_180 : ℕ := Nat.lcm 48 180

theorem lcm_48_180_value : lcm_48_180 = 720 :=
by
-- Proof not required, insert sorry
sorry

end lcm_48_180_value_l40_40550


namespace arina_largest_shareholder_min_cost_l40_40734

-- Defining the shares owned by each person.
def owns_shares : Type :=
  { arina: Nat // arina = 90001} ∧
  { maxim: Nat // maxim = 104999} ∧
  { inga: Nat // inga = 30000} ∧
  { yuri: Nat // yuri = 30000} ∧
  { yulia: Nat // yulia = 30000} ∧
  { anton: Nat // anton = 15000}

-- Defining the price per share each person wants for their shares with the yield.
def price_per_share : Type :=
  { maxim: Nat // maxim = 11} ∧ -- 10 * 1.10
  { inga: Nat // inga = 12.5} ∧ -- 10 * 1.25
  { yuri: Nat // yuri = 11.5} ∧ -- 10 * 1.15
  { yulia: Nat // yulia = 13} ∧ -- 10 * 1.30
  { anton: Nat // anton = 14} -- 10 * 1.40

-- Main theorem to prove the minimum cost for Arina to become the largest shareholder.
theorem arina_largest_shareholder_min_cost (ow: owns_shares) (pp: price_per_share) : Nat :=
  (∃ n, n = 210000) ∧ n = 15000 * 14 :=
  sorry

end arina_largest_shareholder_min_cost_l40_40734


namespace linear_equation_solution_l40_40591

theorem linear_equation_solution (x y b : ℝ) (h1 : x - 2*y + b = 0) (h2 : y = (1/2)*x + b - 1) :
  b = 2 :=
by
  sorry

end linear_equation_solution_l40_40591


namespace sum_of_faces_edges_vertices_l40_40657

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices : 
  rectangular_prism_faces + rectangular_prism_edges + rectangular_prism_vertices = 26 := 
by
  sorry

end sum_of_faces_edges_vertices_l40_40657


namespace solve_for_x_l40_40823

theorem solve_for_x (x : ℝ) (h : 0.60 * 500 = 0.50 * x) : x = 600 :=
  sorry

end solve_for_x_l40_40823


namespace ratio_difference_l40_40199

theorem ratio_difference (x : ℕ) (h : (2 * x + 4) * 7 = (3 * x + 4) * 5) : 3 * x - 2 * x = 8 := 
by sorry

end ratio_difference_l40_40199


namespace tan_alpha_value_l40_40137

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l40_40137


namespace james_twitch_income_l40_40779

theorem james_twitch_income :
  let tier1_base := 120
  let tier2_base := 50
  let tier3_base := 30
  let tier1_gifted := 10
  let tier2_gifted := 25
  let tier3_gifted := 15
  let tier1_new := tier1_base + tier1_gifted
  let tier2_new := tier2_base + tier2_gifted
  let tier3_new := tier3_base + tier3_gifted
  let tier1_income := tier1_new * 4.99
  let tier2_income := tier2_new * 9.99
  let tier3_income := tier3_new * 24.99
  let total_income := tier1_income + tier2_income + tier3_income
  total_income = 2522.50 :=
by
  sorry

end james_twitch_income_l40_40779


namespace percent_profit_l40_40841

-- Definitions based on given conditions
variables (P : ℝ) -- original price of the car

def discounted_price := 0.90 * P
def first_year_value := 0.945 * P
def second_year_value := 0.9828 * P
def third_year_value := 1.012284 * P
def selling_price := 1.62 * P

-- Theorem statement
theorem percent_profit : (selling_price P - P) / P * 100 = 62 := by
  sorry

end percent_profit_l40_40841


namespace total_skips_correct_l40_40065

def bob_skip_rate := 12
def jim_skip_rate := 15
def sally_skip_rate := 18

def bob_rocks := 10
def jim_rocks := 8
def sally_rocks := 12

theorem total_skips_correct : 
  (bob_skip_rate * bob_rocks) + (jim_skip_rate * jim_rocks) + (sally_skip_rate * sally_rocks) = 456 := by
  sorry

end total_skips_correct_l40_40065


namespace find_p_l40_40074

theorem find_p (a : ℕ) (ha : a = 2030) : 
  let p := 2 * a + 1;
  let q := a * (a + 1);
  p = 4061 ∧ Nat.gcd p q = 1 := by
  sorry

end find_p_l40_40074


namespace rectangular_prism_faces_edges_vertices_sum_l40_40675

theorem rectangular_prism_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 := by
  let faces : ℕ := 6
  let edges : ℕ := 12
  let vertices : ℕ := 8
  sorry

end rectangular_prism_faces_edges_vertices_sum_l40_40675


namespace simplify_fraction_l40_40345

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : ((x^2 - y^2) / (x - y)) = x + y :=
by
  -- This is a placeholder for the actual proof
  sorry

end simplify_fraction_l40_40345


namespace probability_sum_two_primes_is_prime_l40_40264

open Finset

-- First 10 prime numbers
def firstTenPrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Function to check if a number is prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Function to sum of two primes
def sum_of_two_primes_is_prime (a b : ℕ) : Prop := is_prime (a + b)

-- Function to compute the probability
def probability_two_prime_sum_is_prime : ℚ :=
  let pairs := firstTenPrimes.pairs
  let prime_pairs := pairs.filter (λ pair, sum_of_two_primes_is_prime pair.1 pair.2)
  (prime_pairs.card : ℚ) / pairs.card

theorem probability_sum_two_primes_is_prime :
  probability_two_prime_sum_is_prime = 1 / 9 :=
by sorry

end probability_sum_two_primes_is_prime_l40_40264


namespace middle_card_is_five_l40_40952

theorem middle_card_is_five 
    (a b c : ℕ) 
    (h1 : a ≠ b ∧ a ≠ c ∧ b ≠ c) 
    (h2 : a + b + c = 16)
    (h3 : a < b ∧ b < c)
    (casey : ¬(∃ y z, y ≠ z ∧ y + z + a = 16 ∧ a < y ∧ y < z))
    (tracy : ¬(∃ x y, x ≠ y ∧ x + y + c = 16 ∧ x < y ∧ y < c))
    (stacy : ¬(∃ x z, x ≠ z ∧ x + z + b = 16 ∧ x < b ∧ b < z)) 
    : b = 5 :=
sorry

end middle_card_is_five_l40_40952


namespace grandma_contribution_l40_40515

def trip_cost : ℝ := 485
def candy_bar_profit : ℝ := 1.25
def candy_bars_sold : ℕ := 188
def amount_earned_from_selling_candy_bars : ℝ := candy_bars_sold * candy_bar_profit
def amount_grandma_gave : ℝ := trip_cost - amount_earned_from_selling_candy_bars

theorem grandma_contribution :
  amount_grandma_gave = 250 := by
  sorry

end grandma_contribution_l40_40515


namespace magnitude_of_z_l40_40759

open Complex -- open the complex number namespace

theorem magnitude_of_z (z : ℂ) (h : z + I = 3) : Complex.abs z = Real.sqrt 10 :=
by
  sorry

end magnitude_of_z_l40_40759


namespace remaining_laps_l40_40217

def track_length : ℕ := 9
def initial_laps : ℕ := 6
def total_distance : ℕ := 99

theorem remaining_laps : (total_distance - (initial_laps * track_length)) / track_length = 5 := by
  sorry

end remaining_laps_l40_40217


namespace compute_xy_l40_40187

theorem compute_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end compute_xy_l40_40187


namespace find_valid_N_l40_40859

def is_divisible_by_10_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, (N % (List.prod (List.range' m 10)) = 0)

def is_not_divisible_by_11_consec (N : ℕ) : Prop :=
  ∀ m : ℕ, ¬ (N % (List.prod (List.range' m 11)) = 0)

theorem find_valid_N (N : ℕ) :
  (is_divisible_by_10_consec N ∧ is_not_divisible_by_11_consec N) ↔
  (∃ k : ℕ, (k > 0) ∧ ¬ (k % 11 = 0) ∧ N = k * Nat.factorial 10) :=
sorry

end find_valid_N_l40_40859


namespace Eunji_higher_than_Yoojung_l40_40196

-- Define floors for Yoojung and Eunji
def Yoojung_floor: ℕ := 17
def Eunji_floor: ℕ := 25

-- Assert that Eunji lives on a higher floor than Yoojung
theorem Eunji_higher_than_Yoojung : Eunji_floor > Yoojung_floor :=
  by
    sorry

end Eunji_higher_than_Yoojung_l40_40196


namespace solve_for_y_l40_40475

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end solve_for_y_l40_40475


namespace tan_alpha_sqrt_15_over_15_l40_40116

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l40_40116


namespace pyramid_volume_84sqrt10_l40_40055

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1/3) * a * b * c

theorem pyramid_volume_84sqrt10 :
  let height := 4 * (Real.sqrt 10)
  let area_base := 7 * 9
  (volume_of_pyramid area_base height) = 84 * (Real.sqrt 10) :=
by
  intros
  simp [volume_of_pyramid]
  sorry

end pyramid_volume_84sqrt10_l40_40055


namespace max_product_three_distinct_nats_sum_48_l40_40517

open Nat

theorem max_product_three_distinct_nats_sum_48
  (a b c : ℕ) (h_distinct: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_sum: a + b + c = 48) :
  a * b * c ≤ 4080 :=
sorry

end max_product_three_distinct_nats_sum_48_l40_40517


namespace maria_initial_cookies_l40_40163

theorem maria_initial_cookies (X : ℕ) 
  (h1: X - 5 = 2 * (5 + 2)) 
  (h2: X ≥ 5)
  : X = 19 := 
by
  sorry

end maria_initial_cookies_l40_40163


namespace greatest_odd_factors_lt_1000_l40_40454

theorem greatest_odd_factors_lt_1000 : ∃ n : ℕ, n < 1000 ∧ n.factors.count % 2 = 1 ∧ n = 961 := 
by {
  sorry
}

end greatest_odd_factors_lt_1000_l40_40454


namespace radius_correct_l40_40527

open Real

noncomputable def radius_of_circle
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop) : ℝ := sorry

theorem radius_correct
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop)
  (h1 : tangent_length = 12) 
  (h2 : secant_internal_segment = 10) 
  (h3 : tangent_secant_perpendicular) : radius_of_circle tangent_length secant_internal_segment tangent_secant_perpendicular = 13 := 
sorry

end radius_correct_l40_40527


namespace supply_duration_l40_40324

theorem supply_duration (pills : ℕ) (rate : ℚ) (supply : ℕ) (days_per_month : ℕ) :
  rate = 3/4 ∧ pills / rate * supply / pills = 360 ∧ days_per_month = 30 → (360 : ℚ) / days_per_month = 12 := 
by {
  intro h,
  cases h with h_rate h_rest,
  cases h_rest with h_days h_month,
  rw [←h_rate] at h_days,
  exact h_days.symm.trans h_month.symm,
}

#eval supply_duration 90 (3/4) 90 30

end supply_duration_l40_40324


namespace decreasing_function_inequality_l40_40354

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ)
  (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : f (3 * a) < f (-2 * a + 10)) :
  a > 2 :=
sorry

end decreasing_function_inequality_l40_40354


namespace tan_alpha_fraction_l40_40104

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l40_40104


namespace apples_used_l40_40640

theorem apples_used (x : ℕ) 
  (initial_apples : ℕ := 23) 
  (bought_apples : ℕ := 6) 
  (final_apples : ℕ := 9) 
  (h : (initial_apples - x) + bought_apples = final_apples) : 
  x = 20 :=
by
  sorry

end apples_used_l40_40640


namespace f_periodic_with_period_one_l40_40940

noncomputable def is_periodic (f : ℝ → ℝ) :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem f_periodic_with_period_one
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f := 
sorry

end f_periodic_with_period_one_l40_40940


namespace number_of_distinct_configurations_l40_40834

-- Definitions of the problem conditions
structure CubeConfig where
  white_cubes : Finset (Fin 8)
  blue_cubes : Finset (Fin 8)
  condition_1 : white_cubes.card = 5
  condition_2 : blue_cubes.card = 3
  condition_3 : ∀ x ∈ white_cubes, x ∉ blue_cubes

def distinctConfigCount (configs : Finset CubeConfig) : ℕ :=
  (configs.filter (λ config => 
    config.white_cubes.card = 5 ∧
    config.blue_cubes.card = 3 ∧
    (∀ x ∈ config.white_cubes, x ∉ config.blue_cubes)
  )).card

-- Theorem stating the correct number of distinct configurations
theorem number_of_distinct_configurations : distinctConfigCount ∅ = 5 := 
  sorry

end number_of_distinct_configurations_l40_40834


namespace find_value_of_expression_l40_40293

-- Conditions as provided
axiom given_condition : ∃ (x : ℕ), 3^x + 3^x + 3^x + 3^x = 2187

-- Proof statement
theorem find_value_of_expression : (exists (x : ℕ), (3^x + 3^x + 3^x + 3^x = 2187) ∧ ((x + 2) * (x - 2) = 21)) :=
sorry

end find_value_of_expression_l40_40293


namespace even_sine_function_phi_eq_pi_div_2_l40_40478
open Real

theorem even_sine_function_phi_eq_pi_div_2 (φ : ℝ) (h : 0 ≤ φ ∧ φ ≤ π)
    (even_f : ∀ x : ℝ, sin (x + φ) = sin (-x + φ)) : φ = π / 2 :=
sorry

end even_sine_function_phi_eq_pi_div_2_l40_40478


namespace fifth_flower_is_e_l40_40030

def flowers : List String := ["a", "b", "c", "d", "e", "f", "g"]

theorem fifth_flower_is_e : flowers.get! 4 = "e" := sorry

end fifth_flower_is_e_l40_40030


namespace minimum_buses_needed_l40_40046

theorem minimum_buses_needed (bus_capacity : ℕ) (students : ℕ) (h : bus_capacity = 38 ∧ students = 411) :
  ∃ n : ℕ, 38 * n ≥ students ∧ ∀ m : ℕ, 38 * m ≥ students → n ≤ m :=
by sorry

end minimum_buses_needed_l40_40046


namespace students_just_passed_l40_40699

theorem students_just_passed (total_students first_div_percent second_div_percent : ℝ)
  (h_total_students: total_students = 300)
  (h_first_div_percent: first_div_percent = 0.29)
  (h_second_div_percent: second_div_percent = 0.54)
  (h_no_failures : total_students = 300) :
  ∃ passed_students, passed_students = total_students - (first_div_percent * total_students + second_div_percent * total_students) ∧ passed_students = 51 :=
by
  sorry

end students_just_passed_l40_40699


namespace olivia_initial_quarters_l40_40165

theorem olivia_initial_quarters : 
  ∀ (spent_quarters left_quarters initial_quarters : ℕ),
  spent_quarters = 4 → left_quarters = 7 → initial_quarters = spent_quarters + left_quarters → initial_quarters = 11 :=
by
  intros spent_quarters left_quarters initial_quarters h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end olivia_initial_quarters_l40_40165


namespace max_min_of_f_in_M_l40_40041

noncomputable def domain (x : ℝ) : Prop := 3 - 4*x + x^2 > 0

def M : Set ℝ := { x | domain x }

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem max_min_of_f_in_M :
  ∃ (xₘ xₘₐₓ : ℝ), xₘ ∈ M ∧ xₘₐₓ ∈ M ∧ 
  (∀ x ∈ M, f xₘₐₓ ≥ f x) ∧ 
  (∀ x ∈ M, f xₘ ≠ f xₓₐₓ) :=
by
  sorry

end max_min_of_f_in_M_l40_40041


namespace probability_prime_sum_l40_40251

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end probability_prime_sum_l40_40251


namespace actual_number_of_children_l40_40618

theorem actual_number_of_children (N : ℕ) (B : ℕ) 
  (h1 : B = 2 * N)
  (h2 : ∀ k : ℕ, k = N - 330)
  (h3 : B = 4 * (N - 330)) : 
  N = 660 :=
by 
  sorry

end actual_number_of_children_l40_40618


namespace circle_has_greatest_symmetry_l40_40962

-- Definitions based on the conditions
def lines_of_symmetry (figure : String) : ℕ∞ := 
  match figure with
  | "regular pentagon" => 5
  | "isosceles triangle" => 1
  | "circle" => ⊤  -- Using the symbol ⊤ to represent infinity in Lean.
  | "rectangle" => 2
  | "parallelogram" => 0
  | _ => 0          -- default case

theorem circle_has_greatest_symmetry :
  ∃ fig, fig = "circle" ∧ ∀ other_fig, lines_of_symmetry fig ≥ lines_of_symmetry other_fig := 
by
  sorry

end circle_has_greatest_symmetry_l40_40962


namespace find_n_divisible_by_11_l40_40698

theorem find_n_divisible_by_11 : ∃ n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 :=
by
  use 1
  -- proof steps would go here, but we're only asked for the statement
  sorry

end find_n_divisible_by_11_l40_40698


namespace number_of_sophomores_l40_40990

-- Definition of the conditions
variables (J S P j s p : ℕ)

-- Condition: Equal number of students in debate team
def DebateTeam_Equal : Prop := j = s ∧ s = p

-- Condition: Total number of students
def TotalStudents : Prop := J + S + P = 45

-- Condition: Percentage relationships
def PercentRelations_J : Prop := j = J / 5
def PercentRelations_S : Prop := s = 3 * S / 20
def PercentRelations_P : Prop := p = P / 10

-- The main theorem to prove
theorem number_of_sophomores : DebateTeam_Equal j s p 
                               → TotalStudents J S P 
                               → PercentRelations_J J j 
                               → PercentRelations_S S s 
                               → PercentRelations_P P p 
                               → P = 21 :=
by 
  sorry

end number_of_sophomores_l40_40990


namespace probability_white_ball_l40_40973

def num_white_balls : ℕ := 5
def num_black_balls : ℕ := 6
def total_balls : ℕ := num_white_balls + num_black_balls

theorem probability_white_ball : (num_white_balls : ℚ) / total_balls = 5 / 11 := by
  sorry

end probability_white_ball_l40_40973


namespace solution_of_inequality_l40_40177

theorem solution_of_inequality (a : ℝ) :
  (a = 0 → ∀ x : ℝ, ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1) ∧
  (a < 0 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1 ∨ x < 1/a)) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (a > 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ∀ x : ℝ, ¬(ax^2 - (a + 1) * x + 1 < 0)) :=
by
  sorry

end solution_of_inequality_l40_40177


namespace rational_number_property_l40_40862

theorem rational_number_property 
  (x : ℚ) (a : ℤ) (ha : 1 ≤ a) : 
  (x ^ (⌊x⌋)) = a / 2 → (∃ k : ℤ, x = k) ∨ x = 3 / 2 :=
by
  sorry

end rational_number_property_l40_40862


namespace at_least_one_greater_than_one_l40_40462

open Classical

variable (x y : ℝ)

theorem at_least_one_greater_than_one (h : x + y > 2) : x > 1 ∨ y > 1 :=
by
  sorry

end at_least_one_greater_than_one_l40_40462


namespace tan_150_deg_l40_40740

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - Real.sqrt 3 / 3 := by
  sorry

end tan_150_deg_l40_40740


namespace algebraic_expression_value_l40_40296

variables (a b c d m : ℝ)

theorem algebraic_expression_value :
  a = -b → cd = 1 → m^2 = 1 →
  -(a + b) - cd / 2022 + m^2 / 2022 = 0 :=
by
  intros h1 h2 h3
  sorry

end algebraic_expression_value_l40_40296


namespace cost_price_of_cloth_l40_40059

theorem cost_price_of_cloth:
  ∀ (meters_sold profit_per_meter : ℕ) (selling_price : ℕ),
  meters_sold = 45 →
  profit_per_meter = 12 →
  selling_price = 4500 →
  (selling_price - (profit_per_meter * meters_sold)) / meters_sold = 88 :=
by
  intros meters_sold profit_per_meter selling_price h1 h2 h3
  sorry

end cost_price_of_cloth_l40_40059


namespace trigonometric_identity_l40_40870

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (π / 4 + θ) = 3) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
sorry

end trigonometric_identity_l40_40870


namespace joey_pills_l40_40606

-- Definitions for the initial conditions
def TypeA_initial := 2
def TypeA_increment := 1

def TypeB_initial := 3
def TypeB_increment := 2

def TypeC_initial := 4
def TypeC_increment := 3

def days := 42

-- Function to calculate the sum of an arithmetic series
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- The theorem to be proved
theorem joey_pills :
  arithmetic_sum TypeA_initial TypeA_increment days = 945 ∧
  arithmetic_sum TypeB_initial TypeB_increment days = 1848 ∧
  arithmetic_sum TypeC_initial TypeC_increment days = 2751 :=
by sorry

end joey_pills_l40_40606


namespace sam_possible_lunches_without_violation_l40_40537

def main_dishes := ["Burger", "Fish and Chips", "Pasta", "Vegetable Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Apple Pie", "Chocolate Cake"]

def valid_combinations := 
  (main_dishes.length * beverages.length * snacks.length) - 
  ((1 * if "Fish and Chips" ∈ main_dishes then 1 else 0) * if "Soda" ∈ beverages then 1 else 0 * snacks.length)

theorem sam_possible_lunches_without_violation : valid_combinations = 14 := by
  sorry

end sam_possible_lunches_without_violation_l40_40537


namespace tan_alpha_proof_l40_40106

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l40_40106


namespace tan_alpha_value_l40_40131

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l40_40131


namespace contractor_daily_amount_l40_40205

theorem contractor_daily_amount
  (days_worked : ℕ) (total_days : ℕ) (fine_per_absent_day : ℝ)
  (total_amount : ℝ) (days_absent : ℕ) (amount_received : ℝ) :
  days_worked = total_days - days_absent →
  (total_amount = (days_worked * amount_received - days_absent * fine_per_absent_day)) →
  total_days = 30 →
  fine_per_absent_day = 7.50 →
  total_amount = 685 →
  days_absent = 2 →
  amount_received = 25 :=
by
  sorry

end contractor_daily_amount_l40_40205


namespace garden_width_l40_40916

theorem garden_width (w : ℝ) (h : ℝ) 
  (h1 : w * h ≥ 150)
  (h2 : h = w + 20)
  (h3 : 2 * (w + h) ≤ 70) :
  w = -10 + 5 * Real.sqrt 10 :=
by sorry

end garden_width_l40_40916


namespace chocolates_bought_at_cost_price_l40_40634

variables (C S : ℝ) (n : ℕ)

-- Given conditions
def cost_eq_selling_50 := n * C = 50 * S
def gain_percent := (S - C) / C = 0.30

-- Question to prove
theorem chocolates_bought_at_cost_price (h1 : cost_eq_selling_50 C S n) (h2 : gain_percent C S) : n = 65 :=
sorry

end chocolates_bought_at_cost_price_l40_40634


namespace charlotte_should_bring_money_l40_40738

theorem charlotte_should_bring_money (p d a : ℝ) (h_p : p = 90) (h_d : d = 20) (h_a : a = 72) :
  a = p - (d / 100 * p) :=
by
  rw [h_p, h_d, h_a]
  norm_num
  sorry

end charlotte_should_bring_money_l40_40738


namespace pyramid_volume_l40_40056

noncomputable def volume_of_pyramid (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let diagonal_length := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2)
  (1 / 3) * base_area * height

theorem pyramid_volume : volume_of_pyramid 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end pyramid_volume_l40_40056


namespace probability_difference_multiple_six_l40_40499

theorem probability_difference_multiple_six :
  ∀ (S : Finset ℕ), S.card = 12 ∧ (∀ x ∈ S, 1 ≤ x ∧ x ≤ 4012) →
  ∃ (a b ∈ S), a ≠ b ∧ (a - b) % 6 = 0 := 
by
  intros S hS
  obtain ⟨hcard, hx⟩ := hS
  have pigeonhole := pigeonhole_principle_mod_6 S hcard hx
  exact pigeonhole

-- Placeholder for the proof by Pigeonhole principle
sorry

end probability_difference_multiple_six_l40_40499


namespace m_leq_nine_l40_40883

theorem m_leq_nine (m : ℝ) : (∀ x : ℝ, (x^2 - 4*x + 3 < 0) → (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + m < 0)) → m ≤ 9 :=
by
sorry

end m_leq_nine_l40_40883


namespace third_side_triangle_l40_40439

theorem third_side_triangle (a : ℝ) :
  (5 < a ∧ a < 13) → (a = 8) :=
sorry

end third_side_triangle_l40_40439


namespace geometric_first_term_l40_40535

-- Define the conditions
def is_geometric_series (first_term : ℝ) (r : ℝ) (sum : ℝ) : Prop :=
  sum = first_term / (1 - r)

-- Define the main theorem
theorem geometric_first_term (r : ℝ) (sum : ℝ) (first_term : ℝ) 
  (h_r : r = 1/4) (h_S : sum = 80) (h_sum_formula : is_geometric_series first_term r sum) : 
  first_term = 60 :=
by
  sorry

end geometric_first_term_l40_40535


namespace M_is_infinite_l40_40504

variable (M : Set ℝ)

def has_properties (M : Set ℝ) : Prop :=
  (∃ x y : ℝ, x ∈ M ∧ y ∈ M ∧ x ≠ y) ∧ ∀ x ∈ M, (3*x - 2 ∈ M ∨ -4*x + 5 ∈ M)

theorem M_is_infinite (M : Set ℝ) (h : has_properties M) : ¬Finite M := by
  sorry

end M_is_infinite_l40_40504


namespace tan_alpha_value_l40_40095

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l40_40095


namespace harriet_speed_l40_40195

/-- Harriet drove back from B-town to A-ville at a constant speed of 145 km/hr.
    The entire trip took 5 hours, and it took Harriet 2.9 hours to drive from A-ville to B-town.
    Prove that Harriet's speed while driving from A-ville to B-town was 105 km/hr. -/
theorem harriet_speed (v_return : ℝ) (T_total : ℝ) (t_AB : ℝ) (v_AB : ℝ) :
  v_return = 145 →
  T_total = 5 →
  t_AB = 2.9 →
  v_AB = 105 :=
by
  intros
  sorry

end harriet_speed_l40_40195


namespace total_bill_first_month_l40_40746

theorem total_bill_first_month (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) 
  (h3 : 2 * C = 2 * C) : 
  F + C = 50 := by
  sorry

end total_bill_first_month_l40_40746


namespace arithmetic_sequence_sum_l40_40642

theorem arithmetic_sequence_sum (a d x y : ℤ) 
  (h1 : a = 3) (h2 : d = 5) 
  (h3 : x = a + d) 
  (h4 : y = x + d) 
  (h5 : y = 18) 
  (h6 : x = 13) : x + y = 31 := by
  sorry

end arithmetic_sequence_sum_l40_40642


namespace total_operations_in_one_hour_l40_40521

theorem total_operations_in_one_hour :
  let additions_per_second := 12000
  let multiplications_per_second := 8000
  (additions_per_second + multiplications_per_second) * 3600 = 72000000 :=
by
  sorry

end total_operations_in_one_hour_l40_40521


namespace geometric_sequence_properties_l40_40754

/-- Given {a_n} is a geometric sequence, a_1 = 1 and a_4 = 1/8, 
the common ratio q of {a_n} is 1/2 and the sum of the first 5 terms of {1/a_n} is 31. -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (h1 : a 1 = 1) (h4 : a 4 = 1 / 8) : 
  (∃ q : ℝ, (∀ n : ℕ, a n = a 1 * q ^ (n - 1)) ∧ q = 1 / 2) ∧ 
  (∃ S : ℝ, S = 31 ∧ S = (1 - 2^5) / (1 - 2)) :=
by
  -- Skipping the proof
  sorry

end geometric_sequence_properties_l40_40754


namespace remaining_requests_after_7_days_l40_40162

-- Definitions based on the conditions
def dailyRequests : ℕ := 8
def dailyWork : ℕ := 4
def days : ℕ := 7

-- Theorem statement representing our final proof problem
theorem remaining_requests_after_7_days : 
  (dailyRequests * days - dailyWork * days) + dailyRequests * days = 84 := by
  sorry

end remaining_requests_after_7_days_l40_40162


namespace multiplication_correct_l40_40382

theorem multiplication_correct : 121 * 54 = 6534 := by
  sorry

end multiplication_correct_l40_40382


namespace cos_two_pi_over_three_l40_40704

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 := 
by
  sorry

end cos_two_pi_over_three_l40_40704


namespace sum_of_faces_edges_vertices_of_rect_prism_l40_40687

theorem sum_of_faces_edges_vertices_of_rect_prism
  (f e v : ℕ)
  (h_f : f = 6)
  (h_e : e = 12)
  (h_v : v = 8) :
  f + e + v = 26 :=
by
  rw [h_f, h_e, h_v]
  norm_num

end sum_of_faces_edges_vertices_of_rect_prism_l40_40687


namespace perimeter_of_fenced_square_field_l40_40496

-- Definitions for conditions
def num_posts : ℕ := 36
def spacing_between_posts : ℝ := 6 -- in feet
def post_width : ℝ := 1 / 2 -- 6 inches in feet

-- The statement to be proven
theorem perimeter_of_fenced_square_field :
  (4 * ((9 * spacing_between_posts) + (10 * post_width))) = 236 :=
by
  sorry

end perimeter_of_fenced_square_field_l40_40496


namespace negation_of_p_l40_40292

def proposition_p := ∃ x : ℝ, x ≥ 1 ∧ x^2 - x < 0

theorem negation_of_p : (∀ x : ℝ, x ≥ 1 → x^2 - x ≥ 0) :=
by
  sorry

end negation_of_p_l40_40292


namespace range_of_a_l40_40627

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0)
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l40_40627


namespace find_x0_l40_40705

-- Define a function f with domain [0, 3] and its inverse
variable {f : ℝ → ℝ}

-- Assume conditions for the inverse function
axiom f_inv_1 : ∀ x, 0 ≤ x ∧ x < 1 → 1 ≤ f x ∧ f x < 2
axiom f_inv_2 : ∀ x, 2 < x ∧ x ≤ 4 → 0 ≤ f x ∧ f x < 1

-- Domain condition
variables (x : ℝ) (hf_domain : 0 ≤ x ∧ x ≤ 3)

-- The main theorem
theorem find_x0 : (∃ x0: ℝ, f x0 = x0) → x = 2 :=
  sorry

end find_x0_l40_40705


namespace permutation_6_4_l40_40364

theorem permutation_6_4 : (Nat.factorial 6) / (Nat.factorial (6 - 4)) = 360 := by
  sorry

end permutation_6_4_l40_40364


namespace max_a_squared_b_squared_c_squared_l40_40771

theorem max_a_squared_b_squared_c_squared (a b c : ℤ)
  (h1 : a + b + c = 3)
  (h2 : a^3 + b^3 + c^3 = 3) :
  a^2 + b^2 + c^2 ≤ 57 :=
sorry

end max_a_squared_b_squared_c_squared_l40_40771


namespace probability_prime_sum_l40_40254

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end probability_prime_sum_l40_40254


namespace no_real_solution_l40_40621

theorem no_real_solution (x y : ℝ) : x^3 + y^2 = 2 → x^2 + x * y + y^2 - y = 0 → false := 
by 
  intro h1 h2
  sorry

end no_real_solution_l40_40621


namespace possible_values_2a_b_l40_40415

theorem possible_values_2a_b (a b x y z : ℕ) (h1: a^x = 1994^z) (h2: b^y = 1994^z) (h3: 1/x + 1/y = 1/z) : 
  (2 * a + b = 1001) ∨ (2 * a + b = 1996) :=
by
  sorry

end possible_values_2a_b_l40_40415


namespace correct_operation_l40_40514

theorem correct_operation (a : ℝ) : 
  ((a^2)^3 = a^6) ∧ ¬(a^2 + a^3 = a^5) ∧ ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a^6 - 2 * a^3 = 3 * a^3) :=
by
  -- Provide the four required conditions separately
  -- Option B is correct:
  show (a^2)^3 = a^6, from sorry,
  
  -- Option A is incorrect:
  show ¬(a^2 + a^3 = a^5), from sorry,
  
  -- Option C is incorrect:
  show ¬(a^2 * a^3 = a^6), from sorry,
  
  -- Option D is incorrect:
  show ¬(6 * a^6 - 2 * a^3 = 3 * a^3), from sorry


end correct_operation_l40_40514


namespace negate_proposition_l40_40796

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^2 + 2 > 6)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) :=
by sorry

end negate_proposition_l40_40796


namespace ball_rebound_percentage_l40_40457

theorem ball_rebound_percentage (P : ℝ) 
  (h₁ : 100 + 2 * 100 * P + 2 * 100 * P^2 = 250) : P = 0.5 := 
by 
  sorry

end ball_rebound_percentage_l40_40457


namespace cost_price_percentage_of_marked_price_l40_40633

theorem cost_price_percentage_of_marked_price (MP CP : ℝ) (discount gain_percent : ℝ) 
  (h_discount : discount = 0.12) (h_gain_percent : gain_percent = 0.375) 
  (h_SP_def : SP = MP * (1 - discount))
  (h_SP_gain : SP = CP * (1 + gain_percent)) :
  CP / MP = 0.64 :=
by
  sorry

end cost_price_percentage_of_marked_price_l40_40633


namespace sum_of_solutions_l40_40483

theorem sum_of_solutions : 
  let a := 1
  let b := -7
  let c := -30
  (a * x^2 + b * x + c = 0) → ((-b / a) = 7) :=
by
  sorry

end sum_of_solutions_l40_40483


namespace average_tree_height_l40_40919

def mixed_num_to_improper (whole: ℕ) (numerator: ℕ) (denominator: ℕ) : Rat :=
  whole + (numerator / denominator)

theorem average_tree_height 
  (elm : Rat := mixed_num_to_improper 11 2 3)
  (oak : Rat := mixed_num_to_improper 17 5 6)
  (pine : Rat := mixed_num_to_improper 15 1 2)
  (num_trees : ℕ := 3) :
  ((elm + oak + pine) / num_trees) = (15 : Rat) := 
  sorry

end average_tree_height_l40_40919


namespace prime_pair_probability_l40_40230

def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def isPrimeSum (a b : ℕ) : Prop := Nat.Prime (a + b)

theorem prime_pair_probability :
  let pairs := List.sublistsLen 2 firstTenPrimes
  let primePairs := pairs.filter (λ p, match p with
                                       | [a, b] => isPrimeSum a b
                                       | _ => false
                                       end)
  (primePairs.length : ℚ) / (pairs.length : ℚ) = 1 / 9 :=
by
  sorry

end prime_pair_probability_l40_40230


namespace goldbach_10000_l40_40775

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem goldbach_10000 :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p q : ℕ), (p, q) ∈ S → is_prime p ∧ is_prime q ∧ p + q = 10000) ∧ S.card > 3 :=
sorry

end goldbach_10000_l40_40775


namespace scaled_system_solution_l40_40362

theorem scaled_system_solution (a1 b1 c1 a2 b2 c2 x y : ℝ) 
  (h1 : a1 * 8 + b1 * 3 = c1) 
  (h2 : a2 * 8 + b2 * 3 = c2) : 
  4 * a1 * 10 + 3 * b1 * 5 = 5 * c1 ∧ 4 * a2 * 10 + 3 * b2 * 5 = 5 * c2 := 
by 
  sorry

end scaled_system_solution_l40_40362


namespace least_positive_integer_with_12_factors_l40_40002

theorem least_positive_integer_with_12_factors :
  ∃ m : ℕ, 0 < m ∧ (∀ n : ℕ, 0 < n → (n ≠ m →  m ≤ n)) ∧ (12 = (m.factors.toFinset.card)) := sorry

end least_positive_integer_with_12_factors_l40_40002


namespace find_k_l40_40837

theorem find_k (k : ℚ) (h : ∃ k : ℚ, (3 * (4 - k) = 2 * (-5 - 3))): k = -4 / 3 := by
  sorry

end find_k_l40_40837


namespace least_palindrome_divisible_by_25_l40_40506

theorem least_palindrome_divisible_by_25 : ∃ (n : ℕ), 
  (10^4 ≤ n ∧ n < 10^5) ∧
  (∀ (a b c : ℕ), n = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a) ∧
  n % 25 = 0 ∧
  n = 10201 :=
by
  sorry

end least_palindrome_divisible_by_25_l40_40506


namespace total_cost_meal_l40_40753

-- Define the initial conditions
variables (x : ℝ) -- x represents the total cost of the meal

-- Initial number of friends
def initial_friends : ℝ := 4

-- New number of friends after additional friends join
def new_friends : ℝ := 7

-- The decrease in cost per friend
def cost_decrease : ℝ := 15

-- Lean statement to assert our proof
theorem total_cost_meal : x / initial_friends - x / new_friends = cost_decrease → x = 140 :=
by
  sorry

end total_cost_meal_l40_40753
