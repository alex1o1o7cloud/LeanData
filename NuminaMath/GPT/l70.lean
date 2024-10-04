import Mathlib

namespace total_surface_area_of_box_l70_70044

-- Definitions
def sum_of_edges (a b c : ℝ) : Prop :=
  4 * a + 4 * b + 4 * c = 160

def distance_to_opposite_corner (a b c : ℝ) : Prop :=
  real.sqrt (a^2 + b^2 + c^2) = 25

-- Theorem statement
theorem total_surface_area_of_box (a b c : ℝ) (h_edges : sum_of_edges a b c) (h_distance : distance_to_opposite_corner a b c) :
  2 * (a * b + b * c + c * a) = 975 :=
by
  sorry

end total_surface_area_of_box_l70_70044


namespace expected_games_per_match_l70_70117

theorem expected_games_per_match 
  (P_F : ℝ) (P_J : ℝ) (n : ℕ) 
  (hF : P_F = 0.3) (hJ : P_J = 0.7) (hn : n = 21) : 
  expected_value P_F P_J n = 30 :=
sorry

end expected_games_per_match_l70_70117


namespace max_possible_cities_traversed_l70_70503

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l70_70503


namespace find_b_l70_70242

theorem find_b (b : ℤ) :
  ∃ (r₁ r₂ : ℤ), (r₁ = -9) ∧ (r₁ * r₂ = 36) ∧ (r₁ + r₂ = -b) → b = 13 :=
by {
  sorry
}

end find_b_l70_70242


namespace factor_difference_of_cubes_l70_70934

theorem factor_difference_of_cubes (t : ℝ) : 
  t^3 - 125 = (t - 5) * (t^2 + 5 * t + 25) :=
sorry

end factor_difference_of_cubes_l70_70934


namespace initial_apples_l70_70447

theorem initial_apples (C : ℝ) (h : C + 7.0 = 27) : C = 20.0 := by
  sorry

end initial_apples_l70_70447


namespace max_rectangle_area_max_rectangle_area_exists_l70_70590

theorem max_rectangle_area (l w : ℕ) (h : l + w = 20) : l * w ≤ 100 :=
by sorry

-- Alternatively, to also show the existence of the maximum value.
theorem max_rectangle_area_exists : ∃ l w : ℕ, l + w = 20 ∧ l * w = 100 :=
by sorry

end max_rectangle_area_max_rectangle_area_exists_l70_70590


namespace total_value_of_coins_l70_70775

theorem total_value_of_coins (h1 : ∀ (q d : ℕ), q + d = 23)
                             (h2 : ∀ q, q = 16)
                             (h3 : ∀ d, d = 23 - 16)
                             (h4 : ∀ q, q * 0.25 = 4.00)
                             (h5 : ∀ d, d * 0.10 = 0.70)
                             : 4.00 + 0.70 = 4.70 :=
by
  sorry

end total_value_of_coins_l70_70775


namespace minimum_number_of_girls_l70_70366

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l70_70366


namespace random_sampling_not_in_proving_methods_l70_70564

inductive Method
| Comparison
| RandomSampling
| SyntheticAndAnalytic
| ProofByContradictionAndScaling

open Method

def proving_methods : List Method :=
  [Comparison, SyntheticAndAnalytic, ProofByContradictionAndScaling]

theorem random_sampling_not_in_proving_methods : 
  RandomSampling ∉ proving_methods :=
sorry

end random_sampling_not_in_proving_methods_l70_70564


namespace coordinates_of_M_l70_70643

theorem coordinates_of_M :
  -- Given the function f(x) = 2x^2 + 1
  let f : Real → Real := λ x => 2 * x^2 + 1
  -- And its derivative
  let f' : Real → Real := λ x => 4 * x
  -- The coordinates of point M where the instantaneous rate of change is -8 are (-2, 9)
  (∃ x0 : Real, f' x0 = -8 ∧ f x0 = y0 ∧ x0 = -2 ∧ y0 = 9) := by
    sorry

end coordinates_of_M_l70_70643


namespace jackson_grade_increase_per_hour_l70_70157

-- Define the necessary variables
variables (v s p G : ℕ)

-- The conditions from the problem
def study_condition1 : v = 9 := sorry
def study_condition2 : s = v / 3 := sorry
def grade_starts_at_zero : G = s * p := sorry
def final_grade : G = 45 := sorry

-- The final problem statement to prove
theorem jackson_grade_increase_per_hour :
  p = 15 :=
by
  -- Add our sorry to indicate the partial proof
  sorry

end jackson_grade_increase_per_hour_l70_70157


namespace rose_age_l70_70863

variable {R M : ℝ}

theorem rose_age (h1 : R = (1/3) * M) (h2 : R + M = 100) : R = 25 :=
sorry

end rose_age_l70_70863


namespace cafeteria_extra_fruits_l70_70879

def red_apples_ordered : ℕ := 43
def green_apples_ordered : ℕ := 32
def oranges_ordered : ℕ := 25
def red_apples_chosen : ℕ := 7
def green_apples_chosen : ℕ := 5
def oranges_chosen : ℕ := 4

def extra_red_apples : ℕ := red_apples_ordered - red_apples_chosen
def extra_green_apples : ℕ := green_apples_ordered - green_apples_chosen
def extra_oranges : ℕ := oranges_ordered - oranges_chosen

def total_extra_fruits : ℕ := extra_red_apples + extra_green_apples + extra_oranges

theorem cafeteria_extra_fruits : total_extra_fruits = 84 := by
  sorry

end cafeteria_extra_fruits_l70_70879


namespace work_completion_l70_70762

theorem work_completion (x y : ℕ) : 
  (1 / (x + y) = 1 / 12) ∧ (1 / y = 1 / 24) → x = 24 :=
by
  sorry

end work_completion_l70_70762


namespace value_of_y_minus_x_l70_70409

theorem value_of_y_minus_x (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : x + y = 8) 
  (h3 : y - 3 * x + z = 9) : 
  y - x = 6.5 :=
by
  -- Proof steps would go here
  sorry

end value_of_y_minus_x_l70_70409


namespace Vasyuki_coloring_possible_l70_70692

theorem Vasyuki_coloring_possible (n : ℕ) (perm : Equiv.Perm (Fin n)) : 
  ∃ (colors : Fin n → Fin 3), ∀ i, colors i ≠ colors (perm i) :=
by
  sorry

end Vasyuki_coloring_possible_l70_70692


namespace compute_b_l70_70813

open Real

theorem compute_b
  (a : ℚ) 
  (b : ℚ) 
  (h₀ : (3 + sqrt 5) ^ 3 + a * (3 + sqrt 5) ^ 2 + b * (3 + sqrt 5) + 12 = 0) 
  : b = -14 :=
sorry

end compute_b_l70_70813


namespace FatherCandyCount_l70_70788

variables (a b c d e : ℕ)

-- Conditions
def BillyInitial := 6
def CalebInitial := 11
def AndyInitial := 9
def BillyReceived := 8
def CalebReceived := 11
def AndyHasMore := 4

-- Define number of candies Andy has now based on Caleb's candies
def AndyTotal (b c : ℕ) : ℕ := c + AndyHasMore

-- Define number of candies received by Andy
def AndyReceived (a b c d e : ℕ) : ℕ := (AndyTotal b c) - AndyInitial

-- Define total candies bought by father
def FatherBoughtCandies (d e f : ℕ) : ℕ := d + e + f

theorem FatherCandyCount : FatherBoughtCandies BillyReceived CalebReceived (AndyReceived BillyInitial CalebInitial AndyInitial BillyReceived CalebReceived)  = 36 :=
by
  sorry

end FatherCandyCount_l70_70788


namespace x_minus_y_possible_values_l70_70258

theorem x_minus_y_possible_values (x y : ℝ) (hx : x^2 = 9) (hy : |y| = 4) (hxy : x < y) : x - y = -1 ∨ x - y = -7 := 
sorry

end x_minus_y_possible_values_l70_70258


namespace original_inhabitants_proof_l70_70408

noncomputable def original_inhabitants (final_population : ℕ) : ℝ :=
  final_population / (0.75 * 0.9)

theorem original_inhabitants_proof :
  original_inhabitants 5265 = 7800 :=
by
  sorry

end original_inhabitants_proof_l70_70408


namespace probability_two_red_two_blue_l70_70213

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_two_red_two_blue :
  (choose 15 2 * choose 9 2 : ℚ) / (choose 24 4 : ℚ) = 108 / 361 :=
by
  sorry

end probability_two_red_two_blue_l70_70213


namespace function_behaviour_l70_70129

theorem function_behaviour (a : ℝ) (h : a ≠ 0) :
  ¬ ((a * (-2)^2 + 2 * a * (-2) + 1 > a * (-1)^2 + 2 * a * (-1) + 1) ∧
     (a * (-1)^2 + 2 * a * (-1) + 1 > a * 0^2 + 2 * a * 0 + 1)) :=
by
  sorry

end function_behaviour_l70_70129


namespace length_of_edge_l70_70886

-- Define all necessary conditions
def is_quadrangular_pyramid (e : ℝ) : Prop :=
  (8 * e = 14.8)

-- State the main theorem which is the equivalent proof problem
theorem length_of_edge (e : ℝ) (h : is_quadrangular_pyramid e) : e = 1.85 :=
by
  sorry

end length_of_edge_l70_70886


namespace cube_volume_l70_70042

theorem cube_volume (h : 12 * l = 72) : l^3 = 216 :=
sorry

end cube_volume_l70_70042


namespace john_initial_pairs_9_l70_70696

-- Definitions based on the conditions in the problem

def john_initial_pairs (x : ℕ) := 2 * x   -- Each pair consists of 2 socks

def john_remaining_socks (x : ℕ) := john_initial_pairs x - 5   -- John loses 5 individual socks

def john_max_pairs_left := 7
def john_minimum_socks_required := john_max_pairs_left * 2  -- 7 pairs mean he needs 14 socks

-- Theorem statement proving John initially had 9 pairs of socks
theorem john_initial_pairs_9 : 
  ∀ (x : ℕ), john_remaining_socks x ≥ john_minimum_socks_required → x = 9 := by
  sorry

end john_initial_pairs_9_l70_70696


namespace count_non_perfect_square_or_cube_l70_70659

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l70_70659


namespace monotonically_decreasing_iff_l70_70124

noncomputable def f (a x : ℝ) : ℝ := (x^2 - 2 * a * x) * Real.exp x

theorem monotonically_decreasing_iff (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ f a (-1) ∧ f a x ≤ f a 1) ↔ (a ≥ 3 / 4) :=
by
  sorry

end monotonically_decreasing_iff_l70_70124


namespace value_of_a_for_perfect_square_trinomial_l70_70495

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) (x y : ℝ) :
  (∃ b : ℝ, (x + b * y) ^ 2 = x^2 + a * x * y + y^2) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end value_of_a_for_perfect_square_trinomial_l70_70495


namespace _l70_70746

lemma triangle_angle_neq_side_neq (A B C : Type) [euclidean_geometry A B C] 
  (h1 : ∠ A ≠ ∠ B) : ¬ (AC = BC) :=
by 
  assume h2 : AC = BC
  have h3 : ∠ A = ∠ B := isosceles_triangle_theorem h2
  contradiction

end _l70_70746


namespace optimal_selection_method_uses_golden_ratio_l70_70352

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l70_70352


namespace sum_distances_saham_and_mother_l70_70719

theorem sum_distances_saham_and_mother :
  let saham_distance := 2.6
  let mother_distance := 5.98
  saham_distance + mother_distance = 8.58 :=
by
  sorry

end sum_distances_saham_and_mother_l70_70719


namespace proof_x_y_l70_70259

noncomputable def x_y_problem (x y : ℝ) : Prop :=
  (x^2 = 9) ∧ (|y| = 4) ∧ (x < y) → (x - y = -1 ∨ x - y = -7)

theorem proof_x_y (x y : ℝ) : x_y_problem x y :=
by
  sorry

end proof_x_y_l70_70259


namespace total_cost_paint_and_primer_l70_70921

def primer_cost_per_gallon := 30.00
def primer_discount := 0.20
def paint_cost_per_gallon := 25.00
def number_of_rooms := 5

def sale_price_primer : ℝ := primer_cost_per_gallon * (1 - primer_discount)
def total_cost_primer : ℝ := sale_price_primer * number_of_rooms
def total_cost_paint : ℝ := paint_cost_per_gallon * number_of_rooms

theorem total_cost_paint_and_primer :
  total_cost_primer + total_cost_paint = 245.00 :=
by
  sorry

end total_cost_paint_and_primer_l70_70921


namespace total_travel_ways_l70_70482

-- Define the number of car departures
def car_departures : ℕ := 3

-- Define the number of train departures
def train_departures : ℕ := 4

-- Define the number of ship departures
def ship_departures : ℕ := 2

-- The total number of ways to travel from location A to location B
def total_ways : ℕ := car_departures + train_departures + ship_departures

-- The theorem stating the total number of ways to travel given the conditions
theorem total_travel_ways :
  total_ways = 9 :=
by
  -- Proof goes here
  sorry

end total_travel_ways_l70_70482


namespace largest_common_remainder_l70_70424

theorem largest_common_remainder : 
  ∃ n r, 2013 ≤ n ∧ n ≤ 2156 ∧ (n % 5 = r) ∧ (n % 11 = r) ∧ (n % 13 = r) ∧ (r = 4) := 
by
  sorry

end largest_common_remainder_l70_70424


namespace cost_per_pouch_l70_70271

theorem cost_per_pouch (boxes : ℕ) (pouches_per_box : ℕ) (total_cost_dollars : ℕ) :
  boxes = 10 →
  pouches_per_box = 6 →
  total_cost_dollars = 12 →
  (total_cost_dollars * 100) / (boxes * pouches_per_box) = 20 :=
by
  intros,
  -- proof steps here
  sorry

end cost_per_pouch_l70_70271


namespace donny_total_cost_eq_45_l70_70443

-- Definitions for prices of each type of apple
def price_small : ℝ := 1.5
def price_medium : ℝ := 2
def price_big : ℝ := 3

-- Quantities purchased by Donny
def count_small : ℕ := 6
def count_medium : ℕ := 6
def count_big : ℕ := 8

-- Total cost calculation
def total_cost (count_small count_medium count_big : ℕ) : ℝ := 
  (count_small * price_small) + (count_medium * price_medium) + (count_big * price_big)

-- Theorem stating the total cost
theorem donny_total_cost_eq_45 : total_cost count_small count_medium count_big = 45 := by
  sorry

end donny_total_cost_eq_45_l70_70443


namespace expected_digits_fair_icosahedral_die_l70_70099

noncomputable def expected_number_of_digits : ℝ :=
  let one_digit_count := 9
  let two_digit_count := 11
  let total_faces := 20
  let prob_one_digit := one_digit_count / total_faces
  let prob_two_digit := two_digit_count / total_faces
  (prob_one_digit * 1) + (prob_two_digit * 2)

theorem expected_digits_fair_icosahedral_die :
  expected_number_of_digits = 1.55 :=
by
  sorry

end expected_digits_fair_icosahedral_die_l70_70099


namespace julia_baking_days_l70_70989

variable (bakes_per_day : ℕ)
variable (clifford_eats_per_two_days : ℕ)
variable (final_cakes : ℕ)

def number_of_baking_days : ℕ :=
  2 * (final_cakes / (bakes_per_day * 2 - clifford_eats_per_two_days))

theorem julia_baking_days (h1 : bakes_per_day = 4)
                        (h2 : clifford_eats_per_two_days = 1)
                        (h3 : final_cakes = 21) :
  number_of_baking_days bakes_per_day clifford_eats_per_two_days final_cakes = 6 :=
by {
  sorry
}

end julia_baking_days_l70_70989


namespace pow_ge_double_l70_70062

theorem pow_ge_double (n : ℕ) : 2^n ≥ 2 * n := sorry

end pow_ge_double_l70_70062


namespace ceiling_and_floor_calculation_l70_70459

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l70_70459


namespace number_of_stadiums_to_visit_l70_70050

def average_cost_per_stadium : ℕ := 900
def annual_savings : ℕ := 1500
def years_saving : ℕ := 18

theorem number_of_stadiums_to_visit (c : ℕ) (s : ℕ) (n : ℕ) (h1 : c = average_cost_per_stadium) (h2 : s = annual_savings) (h3 : n = years_saving) : n * s / c = 30 := 
by 
  rw [h1, h2, h3]
  exact sorry

end number_of_stadiums_to_visit_l70_70050


namespace minimum_number_of_girls_l70_70378

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l70_70378


namespace min_value_of_expression_l70_70488

theorem min_value_of_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 1 < b) (h₃ : a + b = 2) :
  4 / a + 1 / (b - 1) = 9 := 
sorry

end min_value_of_expression_l70_70488


namespace integral_of_2x2_cos3x_l70_70209

theorem integral_of_2x2_cos3x :
  ∫ x in (0 : ℝ)..(2 * Real.pi), (2 * x ^ 2 - 15) * Real.cos (3 * x) = (8 * Real.pi) / 9 :=
by
  sorry

end integral_of_2x2_cos3x_l70_70209


namespace evaluate_expression_l70_70927

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 4 - 2 * g (-2) = 47 :=
by
  sorry

end evaluate_expression_l70_70927


namespace closest_points_distance_l70_70790

theorem closest_points_distance :
  let center1 := (2, 2)
  let center2 := (17, 10)
  let radius1 := 2
  let radius2 := 10
  let distance_centers := Nat.sqrt ((center2.1 - center1.1) ^ 2 + (center2.2 - center1.2) ^ 2)
  distance_centers = 17 → (distance_centers - radius1 - radius2) = 5 := by
  sorry

end closest_points_distance_l70_70790


namespace eating_time_correct_l70_70007

-- Define the rates at which each individual eats cereal
def rate_fat : ℚ := 1 / 20
def rate_thin : ℚ := 1 / 30
def rate_medium : ℚ := 1 / 15

-- Define the combined rate of eating cereal together
def combined_rate : ℚ := rate_fat + rate_thin + rate_medium

-- Define the total pounds of cereal
def total_cereal : ℚ := 5

-- Define the time taken by everyone to eat the cereal
def time_taken : ℚ := total_cereal / combined_rate

-- Proof statement
theorem eating_time_correct :
  time_taken = 100 / 3 :=
by sorry

end eating_time_correct_l70_70007


namespace at_least_one_fuse_blows_l70_70687

theorem at_least_one_fuse_blows (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.74) (independent : ∀ (A B : Prop), A ∧ B → ¬(A ∨ B)) :
  1 - (1 - pA) * (1 - pB) = 0.961 :=
by
  sorry

end at_least_one_fuse_blows_l70_70687


namespace rotated_D_coords_l70_70015

-- Definitions of the points used in the problem
def point (x y : ℤ) : ℤ × ℤ := (x, y)

-- Definitions of the vertices of the triangle DEF
def D : ℤ × ℤ := point 2 (-3)
def E : ℤ × ℤ := point 2 0
def F : ℤ × ℤ := point 5 (-3)

-- Definition of the rotation center
def center : ℤ × ℤ := point 3 (-2)

-- Function to rotate a point (x, y) by 180 degrees around (h, k)
def rotate_180 (p c : ℤ × ℤ) : ℤ × ℤ := 
  let (x, y) := p
  let (h, k) := c
  (2 * h - x, 2 * k - y)

-- Statement to prove the required coordinates after rotation
theorem rotated_D_coords : rotate_180 D center = point 4 (-1) :=
  sorry

end rotated_D_coords_l70_70015


namespace triangle_area_is_15_l70_70891

def Point := (ℝ × ℝ)

def A : Point := (2, 2)
def B : Point := (7, 2)
def C : Point := (4, 8)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem triangle_area_is_15 : area_of_triangle A B C = 15 :=
by
  -- The proof goes here
  sorry

end triangle_area_is_15_l70_70891


namespace combined_pre_tax_and_pre_tip_cost_l70_70850

theorem combined_pre_tax_and_pre_tip_cost (x y : ℝ) 
  (hx : 1.28 * x = 35.20) 
  (hy : 1.19 * y = 22.00) : 
  x + y = 46 := 
by
  sorry

end combined_pre_tax_and_pre_tip_cost_l70_70850


namespace binomial_coeff_a8_l70_70638

theorem binomial_coeff_a8 :
  (∃ (a : ℕ → ℕ), (1 + x)^10 = ∑ k in range 11, a k * (1 - x)^k) → 
  a 8 = 180 :=
by
  sorry

end binomial_coeff_a8_l70_70638


namespace basic_full_fare_l70_70589

theorem basic_full_fare 
  (F R : ℝ)
  (h1 : F + R = 216)
  (h2 : (F + R) + (0.5 * F + R) = 327) :
  F = 210 :=
by
  sorry

end basic_full_fare_l70_70589


namespace total_surface_area_l70_70043

variable (a b c : ℝ)

-- Conditions
def condition1 : Prop := 4 * (a + b + c) = 160
def condition2 : Prop := real.sqrt (a^2 + b^2 + c^2) = 25

-- Prove the desired statement
theorem total_surface_area (h1 : condition1 a b c) (h2 : condition2 a b c) : 2 * (a * b + b * c + c * a) = 975 :=
sorry

end total_surface_area_l70_70043


namespace monotonic_decreasing_interval_l70_70544

noncomputable def f (x : ℝ) : ℝ :=
  x / 4 + 5 / (4 * x) - Real.log x

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), (a = 0) ∧ (b = 5) ∧ (∀ x, 0 < x ∧ x < 5 → (deriv f x < 0)) :=
by
  sorry

end monotonic_decreasing_interval_l70_70544


namespace intersection_of_M_and_complementN_l70_70968

def UniversalSet := Set ℝ
def setM : Set ℝ := {-1, 0, 1, 3}
def setN : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def complementSetN : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_complementN :
  setM ∩ complementSetN = {0, 1} :=
sorry

end intersection_of_M_and_complementN_l70_70968


namespace find_hyperbola_eccentricity_l70_70809

variables {a b : ℝ} (k1 k2 : ℝ)

-- Conditions
def hyperbola_eq (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def slopes_product_minimization (k1 k2 : ℝ) := (2 / (k1 * k2)) + Real.log (k1 * k2)
def is_minimum_value_minimized (x : ℝ) := x = 2

-- Prove that the eccentricity is sqrt(3)
theorem find_hyperbola_eccentricity
  (ha : 0 < a) (hb : 0 < b)
  (hC : hyperbola_eq a b)
  (hk_min : is_minimum_value_minimized 2)
  (hk_eq : k1 * k2 = (b^2 / a^2)) :
  (1 + (b^2 / a^2)) = 3 :=
by
  sorry

end find_hyperbola_eccentricity_l70_70809


namespace minutes_before_noon_l70_70065

theorem minutes_before_noon
    (x : ℕ)
    (h1 : 20 <= x)
    (h2 : 180 - (x - 20) = 3 * (x - 20)) :
    x = 65 := by
  sorry

end minutes_before_noon_l70_70065


namespace find_x_val_l70_70298

theorem find_x_val (x y : ℝ) (c : ℝ) (h1 : y = 1 → x = 8) (h2 : ∀ y, x * y^3 = c) : 
  (∀ (y : ℝ), y = 2 → x = 1) :=
by
  sorry

end find_x_val_l70_70298


namespace radius_of_larger_circle_l70_70738

theorem radius_of_larger_circle (r : ℝ) (radius_ratio : ℝ) (h1 : radius_ratio = 3) (AC_diameter : ℝ) (BC_chord : ℝ) (tangent_point : ℝ) (AB_length : ℝ) (h2 : AB_length = 140) :
  3 * (AB_length / 4) = 210 :=
by 
  sorry

end radius_of_larger_circle_l70_70738


namespace rain_difference_l70_70845

theorem rain_difference (r_m r_t : ℝ) (h_monday : r_m = 0.9) (h_tuesday : r_t = 0.2) : r_m - r_t = 0.7 :=
by sorry

end rain_difference_l70_70845


namespace sam_has_two_nickels_l70_70870

def average_value_initial (total_value : ℕ) (total_coins : ℕ) := total_value / total_coins = 15
def average_value_with_extra_dime (total_value : ℕ) (total_coins : ℕ) := (total_value + 10) / (total_coins + 1) = 16

theorem sam_has_two_nickels (total_value total_coins : ℕ) (h1 : average_value_initial total_value total_coins) (h2 : average_value_with_extra_dime total_value total_coins) : 
∃ (nickels : ℕ), nickels = 2 := 
by 
  sorry

end sam_has_two_nickels_l70_70870


namespace vanessa_savings_weeks_l70_70392

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end vanessa_savings_weeks_l70_70392


namespace distance_from_C_to_A_is_8_l70_70568

-- Define points A, B, and C as real numbers representing positions
def A : ℝ := 0  -- Starting point
def B : ℝ := A - 15  -- 15 meters west from A
def C : ℝ := B + 23  -- 23 meters east from B

-- Prove that the distance from point C to point A is 8 meters
theorem distance_from_C_to_A_is_8 : abs (C - A) = 8 :=
by
  sorry

end distance_from_C_to_A_is_8_l70_70568


namespace moon_speed_conversion_l70_70545

theorem moon_speed_conversion :
  ∀ (moon_speed_kps : ℝ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ),
  moon_speed_kps = 0.9 →
  seconds_in_minute = 60 →
  minutes_in_hour = 60 →
  (moon_speed_kps * (seconds_in_minute * minutes_in_hour) = 3240) := by
  sorry

end moon_speed_conversion_l70_70545


namespace graph_of_equation_is_two_lines_l70_70563

-- define the condition
def equation_condition (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

-- state the theorem
theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, equation_condition x y → (x = 0) ∨ (y = 0) :=
by
  intros x y h
  -- proof here
  sorry

end graph_of_equation_is_two_lines_l70_70563


namespace mark_eggs_supply_l70_70282

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end mark_eggs_supply_l70_70282


namespace inequality_solution_l70_70727

theorem inequality_solution (x : ℝ) : 
  (x^2 + 4 * x + 13 > 0) -> ((x - 4) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ≥ 4) :=
by
  intro h_pos
  sorry

end inequality_solution_l70_70727


namespace eric_age_l70_70556

theorem eric_age (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 :=
by
  sorry

end eric_age_l70_70556


namespace percentage_problem_l70_70833

theorem percentage_problem (x : ℝ) (h : 0.30 * 0.15 * x = 18) : 0.15 * 0.30 * x = 18 :=
by
  sorry

end percentage_problem_l70_70833


namespace distinct_values_count_l70_70929

noncomputable def f : ℕ → ℤ := sorry -- The actual function definition is not required

theorem distinct_values_count :
  ∃! n, n = 3 ∧ 
  (∀ x : ℕ, 
    (f x = f (x - 1) + f (x + 1) ∧ 
     (x = 1 → f x = 2009) ∧ 
     (x = 3 → f x = 0))) := 
sorry

end distinct_values_count_l70_70929


namespace num_from_1_to_200_not_squares_or_cubes_l70_70666

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l70_70666


namespace numbers_not_squares_nor_cubes_1_to_200_l70_70651

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l70_70651


namespace solve_rebus_l70_70948

-- Definitions for the conditions
def is_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

def distinct_digits (A B C D : Nat) : Prop := 
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Main Statement
theorem solve_rebus (A B C D : Nat) (h_distinct : distinct_digits A B C D) 
(h_eq : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
by
  sorry

end solve_rebus_l70_70948


namespace no_such_divisor_l70_70261

theorem no_such_divisor (n : ℕ) : 
  (n ∣ (823435 : ℕ)^15) ∧ (n^5 - n^n = 1) → false := 
by sorry

end no_such_divisor_l70_70261


namespace statement_A_l70_70757

theorem statement_A (x : ℝ) (h : x < -1) : x^2 > x :=
sorry

end statement_A_l70_70757


namespace machines_complete_order_l70_70208

theorem machines_complete_order (h1 : ℝ) (h2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ)
  (h1_def : h1 = 9)
  (h2_def : h2 = 8)
  (rate1_def : rate1 = 1 / h1)
  (rate2_def : rate2 = 1 / h2)
  (combined_rate : ℝ := rate1 + rate2) :
  time = 72 / 17 :=
by
  sorry

end machines_complete_order_l70_70208


namespace nesbitts_inequality_l70_70211

theorem nesbitts_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

end nesbitts_inequality_l70_70211


namespace matrix_solution_l70_70801

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, -3], ![4, -1]]
noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := ![![ -8,  5], ![ 11, -7]]

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![ -1.2, -1.4], ![1.7, 1.9]]

theorem matrix_solution : M * A = B :=
by sorry

end matrix_solution_l70_70801


namespace max_bio_homework_time_l70_70858

-- Define our variables as non-negative real numbers
variables (B H G : ℝ)

-- Given conditions
axiom h1 : H = 2 * B
axiom h2 : G = 6 * B
axiom h3 : B + H + G = 180

-- We need to prove that B = 20
theorem max_bio_homework_time : B = 20 :=
by
  sorry

end max_bio_homework_time_l70_70858


namespace ceiling_and_floor_calculation_l70_70461

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l70_70461


namespace find_intercept_l70_70426

theorem find_intercept (avg_height : ℝ) (avg_shoe_size : ℝ) (a : ℝ)
  (h1 : avg_height = 170)
  (h2 : avg_shoe_size = 40) 
  (h3 : 3 * avg_shoe_size + a = avg_height) : a = 50 := 
by
  sorry

end find_intercept_l70_70426


namespace find_M_l70_70881

theorem find_M (p q r s M : ℚ)
  (h1 : p + q + r + s = 100)
  (h2 : p + 10 = M)
  (h3 : q - 5 = M)
  (h4 : 10 * r = M)
  (h5 : s / 2 = M) :
  M = 1050 / 41 :=
by
  sorry

end find_M_l70_70881


namespace pencil_case_costs_l70_70769

variable {x y : ℝ}

theorem pencil_case_costs :
  (2 * x + 3 * y = 108) ∧ (5 * x = 6 * y) → 
  (x = 24) ∧ (y = 20) :=
by
  intros h
  obtain ⟨h1, h2⟩ := h
  sorry

end pencil_case_costs_l70_70769


namespace middle_and_oldest_son_ages_l70_70082

theorem middle_and_oldest_son_ages 
  (x y z : ℕ) 
  (father_age_current father_age_future : ℕ) 
  (youngest_age_increment : ℕ)
  (father_age_increment : ℕ) 
  (father_equals_sons_sum : father_age_future = (x + youngest_age_increment) + (y + father_age_increment) + (z + father_age_increment))
  (father_age_constraint : father_age_current + father_age_increment = father_age_future)
  (youngest_age_initial : x = 2)
  (father_age_current_value : father_age_current = 33)
  (youngest_age_increment_value : youngest_age_increment = 12)
  (father_age_increment_value : father_age_increment = 12) 
  :
  y = 3 ∧ z = 4 :=
begin
  sorry
end

end middle_and_oldest_son_ages_l70_70082


namespace increasing_interval_l70_70262

-- Define the function f(x) = x^2 + 2*(a - 1)*x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*(a - 1)*x

-- Define the condition for f(x) being increasing on [4, +∞)
def is_increasing_on_interval (a : ℝ) : Prop := 
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → 
    f x a ≤ f y a

-- Define the main theorem that we need to prove
theorem increasing_interval (a : ℝ) (h : is_increasing_on_interval a) : -3 ≤ a :=
by 
  sorry -- proof is required, but omitted as per the instruction.

end increasing_interval_l70_70262


namespace F_2_f_3_equals_341_l70_70446

def f (a : ℕ) : ℕ := a^2 - 2
def F (a b : ℕ) : ℕ := b^3 - a

theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end F_2_f_3_equals_341_l70_70446


namespace optimal_selection_method_uses_golden_ratio_l70_70353

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l70_70353


namespace minimum_number_of_girls_l70_70367

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l70_70367


namespace angela_january_additional_sleep_l70_70782

-- Definitions corresponding to conditions in part a)
def december_sleep_hours : ℝ := 6.5
def january_sleep_hours : ℝ := 8.5
def days_in_january : ℕ := 31

-- The proof statement, proving the January's additional sleep hours
theorem angela_january_additional_sleep :
  (january_sleep_hours - december_sleep_hours) * days_in_january = 62 :=
by
  -- Since the focus is only on the statement, we skip the actual proof.
  sorry

end angela_january_additional_sleep_l70_70782


namespace second_concert_attendance_l70_70172

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119

theorem second_concert_attendance : first_concert_attendance + additional_people = 66018 := 
by 
  -- Proof is not discussed here, only the statement is required.
sorry

end second_concert_attendance_l70_70172


namespace parallelogram_area_72_l70_70539

def parallelogram_area (base height : ℕ) : ℕ :=
  base * height

theorem parallelogram_area_72 :
  parallelogram_area 12 6 = 72 :=
by
  sorry

end parallelogram_area_72_l70_70539


namespace find_x_l70_70882

theorem find_x (x y z : ℝ) 
  (h1 : x + y + z = 150)
  (h2 : x + 10 = y - 10)
  (h3 : x + 10 = 3 * z) :
  x = 380 / 7 := 
  sorry

end find_x_l70_70882


namespace sin_double_angle_l70_70765

theorem sin_double_angle (h1 : Real.pi / 2 < β)
    (h2 : β < α)
    (h3 : α < 3 * Real.pi / 4)
    (h4 : Real.cos (α - β) = 12 / 13)
    (h5 : Real.sin (α + β) = -3 / 5) :
    Real.sin (2 * α) = -56 / 65 := 
by
  sorry

end sin_double_angle_l70_70765


namespace gcd_exponentiation_l70_70201

def m : ℕ := 2^2050 - 1
def n : ℕ := 2^2040 - 1

theorem gcd_exponentiation : Nat.gcd m n = 1023 := by
  sorry

end gcd_exponentiation_l70_70201


namespace watch_sticker_price_l70_70004

theorem watch_sticker_price (x : ℝ)
  (hx_X : 0.80 * x - 50 = y)
  (hx_Y : 0.90 * x = z)
  (savings : z - y = 25) : 
  x = 250 := by
  sorry

end watch_sticker_price_l70_70004


namespace count_not_squares_or_cubes_200_l70_70672

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l70_70672


namespace optimal_selection_uses_golden_ratio_l70_70356

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l70_70356


namespace optimal_selection_uses_golden_ratio_l70_70329

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l70_70329


namespace sum_remainder_mod_9_l70_70627

theorem sum_remainder_mod_9 : 
  (123456 + 123457 + 123458 + 123459 + 123460 + 123461) % 9 = 6 :=
by
  sorry

end sum_remainder_mod_9_l70_70627


namespace fred_spent_18_42_l70_70806

variable (football_price : ℝ) (pokemon_price : ℝ) (baseball_price : ℝ)
variable (football_packs : ℕ) (pokemon_packs : ℕ) (baseball_decks : ℕ)

def total_cost (football_price : ℝ) (football_packs : ℕ) (pokemon_price : ℝ) (pokemon_packs : ℕ) (baseball_price : ℝ) (baseball_decks : ℕ) : ℝ :=
  football_packs * football_price + pokemon_packs * pokemon_price + baseball_decks * baseball_price

theorem fred_spent_18_42 :
  total_cost 2.73 2 4.01 1 8.95 1 = 18.42 :=
by
  sorry

end fred_spent_18_42_l70_70806


namespace ceil_floor_diff_l70_70474

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l70_70474


namespace three_digit_number_is_11_times_sum_of_digits_l70_70621

theorem three_digit_number_is_11_times_sum_of_digits :
    ∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
        (100 * a + 10 * b + c = 11 * (a + b + c)) ↔ 
        (100 * 1 + 10 * 9 + 8 = 11 * (1 + 9 + 8)) := 
by
    sorry

end three_digit_number_is_11_times_sum_of_digits_l70_70621


namespace same_type_sqrt_l70_70756

theorem same_type_sqrt (x : ℝ) : (x = 2 * Real.sqrt 3) ↔
  (x = Real.sqrt (1/3)) ∨
  (¬(x = Real.sqrt 8) ∧ ¬(x = Real.sqrt 18) ∧ ¬(x = Real.sqrt 9)) :=
by
  sorry

end same_type_sqrt_l70_70756


namespace min_value_abs_ab_l70_70362

theorem min_value_abs_ab (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) 
(h_perpendicular : - 1 / (a^2) * (a^2 + 1) / b = -1) :
|a * b| = 2 :=
sorry

end min_value_abs_ab_l70_70362


namespace min_number_of_girls_l70_70370

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l70_70370


namespace evaluate_expression_l70_70399

theorem evaluate_expression : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end evaluate_expression_l70_70399


namespace find_A_l70_70429

noncomputable def telephone_number_satisfies_conditions (A B C D E F G H I J : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J ∧
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  E = D - 2 ∧ F = D - 4 ∧ -- Given D, E, F are consecutive even digits
  H = G - 2 ∧ I = G - 4 ∧ J = G - 6 ∧ -- Given G, H, I, J are consecutive odd digits
  A + B + C = 9

theorem find_A :
  ∃ (A B C D E F G H I J : ℕ), telephone_number_satisfies_conditions A B C D E F G H I J ∧ A = 8 :=
by {
  sorry
}

end find_A_l70_70429


namespace choir_members_total_l70_70415

theorem choir_members_total
  (first_group second_group third_group : ℕ)
  (h1 : first_group = 25)
  (h2 : second_group = 30)
  (h3 : third_group = 15) :
  first_group + second_group + third_group = 70 :=
by
  sorry

end choir_members_total_l70_70415


namespace intersection_eq_l70_70132

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem intersection_eq : M ∩ N = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_eq_l70_70132


namespace solve_for_x_l70_70019

theorem solve_for_x (h : 125 = 5 ^ 3) : ∃ x : ℕ, 125 ^ 4 = 5 ^ x ∧ x = 12 := by
  sorry

end solve_for_x_l70_70019


namespace ceil_floor_diff_l70_70470

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l70_70470


namespace sequence_inequality_l70_70364

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (m n : ℕ)
  (h1 : a 1 = 21/16)
  (h2 : ∀ n ≥ 2, 2 * a n - 3 * a (n - 1) = 3 / 2^(n + 1))
  (h3 : m ≥ 2)
  (h4 : n ≤ m) :
  (a n + 3 / 2^(n + 3))^(1 / m) * (m - (2 / 3)^(n * (m - 1) / m)) < (m^2 - 1) / (m - n + 1) :=
sorry

end sequence_inequality_l70_70364


namespace calculation_result_l70_70203

theorem calculation_result :
  3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 :=
sorry

end calculation_result_l70_70203


namespace four_identical_pairwise_differences_l70_70635

theorem four_identical_pairwise_differences (a : Fin 20 → ℕ) (h_distinct : Function.Injective a) (h_lt_70 : ∀ i, a i < 70) :
  ∃ d, ∃ (f g : Fin 20 × Fin 20), f ≠ g ∧ (a f.1 - a f.2 = d) ∧ (a g.1 - a g.2 = d) ∧
  ∃ (f1 f2 : Fin 20 × Fin 20), (f1 ≠ f ∧ f1 ≠ g) ∧ (f2 ≠ f ∧ f2 ≠ g) ∧ (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  (a f1.1 - a f1.2 = d) ∧ (a f2.1 - a f2.2 = d) ∧
  ∃ (f3 : Fin 20 × Fin 20), (f3 ≠ f ∧ f3 ≠ g ∧ f3 ≠ f1 ∧ f3 ≠ f2) ∧ (a f3.1 - a f3.2 = d) := 
sorry

end four_identical_pairwise_differences_l70_70635


namespace number_of_parents_who_volunteered_to_bring_refreshments_l70_70010

theorem number_of_parents_who_volunteered_to_bring_refreshments 
  (total : ℕ) (supervise : ℕ) (supervise_and_refreshments : ℕ) (N : ℕ) (R : ℕ)
  (h_total : total = 84)
  (h_supervise : supervise = 25)
  (h_supervise_and_refreshments : supervise_and_refreshments = 11)
  (h_R_eq_1_5N : R = 3 * N / 2)
  (h_eq : total = (supervise - supervise_and_refreshments) + (R - supervise_and_refreshments) + supervise_and_refreshments + N) :
  R = 42 :=
by
  sorry

end number_of_parents_who_volunteered_to_bring_refreshments_l70_70010


namespace game_winning_strategy_l70_70711

theorem game_winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 1) ∧ (n % 2 = 1 → ∃ strategy : ℕ → ℕ, ∀ m, strategy m = 2) :=
by
  sorry

end game_winning_strategy_l70_70711


namespace sum_of_fractions_l70_70616

theorem sum_of_fractions :
  (1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6) + 1 / (5 * 6 * 7) + 1 / (6 * 7 * 8)) = 3 / 16 := 
by
  sorry

end sum_of_fractions_l70_70616


namespace part1_part2_l70_70240

def A := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2 * x + 1 - m^2 ≤ 0}

theorem part1 (m : ℝ) (hm : m = 2) :
  A ∩ {x : ℝ | x < -1 ∨ 3 < x} = {x : ℝ | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} :=
sorry

theorem part2 :
  (∀ x, x ∈ A → x ∈ B (m : ℝ)) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end part1_part2_l70_70240


namespace optimal_selection_method_uses_golden_ratio_l70_70337

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l70_70337


namespace min_eq_one_implies_x_eq_one_l70_70804

open Real

theorem min_eq_one_implies_x_eq_one (x : ℝ) (h : min (1/2 + x) (x^2) = 1) : x = 1 := 
sorry

end min_eq_one_implies_x_eq_one_l70_70804


namespace calculate_N_l70_70970

theorem calculate_N (h : (25 / 100) * N = (55 / 100) * 3010) : N = 6622 :=
by
  sorry

end calculate_N_l70_70970


namespace snack_eaters_remaining_l70_70911

noncomputable def initial_snack_eaters := 5000 * 60 / 100
noncomputable def snack_eaters_after_1_hour := initial_snack_eaters + 25
noncomputable def snack_eaters_after_70_percent_left := snack_eaters_after_1_hour * 30 / 100
noncomputable def snack_eaters_after_2_hour := snack_eaters_after_70_percent_left + 50
noncomputable def snack_eaters_after_800_left := snack_eaters_after_2_hour - 800
noncomputable def snack_eaters_after_2_thirds_left := snack_eaters_after_800_left * 1 / 3
noncomputable def final_snack_eaters := snack_eaters_after_2_thirds_left + 100

theorem snack_eaters_remaining : final_snack_eaters = 153 :=
by
  have h1 : initial_snack_eaters = 3000 := by sorry
  have h2 : snack_eaters_after_1_hour = initial_snack_eaters + 25 := by sorry
  have h3 : snack_eaters_after_70_percent_left = snack_eaters_after_1_hour * 30 / 100 := by sorry
  have h4 : snack_eaters_after_2_hour = snack_eaters_after_70_percent_left + 50 := by sorry
  have h5 : snack_eaters_after_800_left = snack_eaters_after_2_hour - 800 := by sorry
  have h6 : snack_eaters_after_2_thirds_left = snack_eaters_after_800_left * 1 / 3 := by sorry
  have h7 : final_snack_eaters = snack_eaters_after_2_thirds_left + 100 := by sorry
  -- Prove that these equal 153 overall
  sorry

end snack_eaters_remaining_l70_70911


namespace sugar_for_third_layer_l70_70600

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end sugar_for_third_layer_l70_70600


namespace daps_equiv_dirps_l70_70682

noncomputable def dops_equiv_daps : ℝ := 5 / 4
noncomputable def dips_equiv_dops : ℝ := 3 / 10
noncomputable def dirps_equiv_dips : ℝ := 2

theorem daps_equiv_dirps (n : ℝ) : 20 = (dops_equiv_daps * dips_equiv_dops * dirps_equiv_dips) * n → n = 15 :=
by sorry

end daps_equiv_dirps_l70_70682


namespace odd_perfect_prime_form_n_is_seven_l70_70064

theorem odd_perfect_prime_form (n p s m : ℕ) (h₁ : n % 2 = 1) (h₂ : ∃ k : ℕ, p = 4 * k + 1) (h₃ : ∃ h : ℕ, s = 4 * h + 1) (h₄ : n = p^s * m^2) (h₅ : ¬ p ∣ m) :
  ∃ k h : ℕ, p = 4 * k + 1 ∧ s = 4 * h + 1 :=
sorry

theorem n_is_seven (n : ℕ) (h₁ : n > 1) (h₂ : ∃ k : ℕ, k * k = n -1) (h₃ : ∃ l : ℕ, l * l = (n * (n + 1)) / 2) :
  n = 7 :=
sorry

end odd_perfect_prime_form_n_is_seven_l70_70064


namespace sin_alpha_eq_three_fifths_l70_70817

theorem sin_alpha_eq_three_fifths (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan α = -3 / 4) 
  (h3 : Real.sin α > 0) 
  (h4 : Real.cos α < 0) 
  (h5 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) : 
  Real.sin α = 3 / 5 := 
sorry

end sin_alpha_eq_three_fifths_l70_70817


namespace find_number_l70_70066

theorem find_number (N : ℝ) (h : (1 / 2) * (3 / 5) * N = 36) : N = 120 :=
by
  sorry

end find_number_l70_70066


namespace train_speed_correct_l70_70072

-- Definitions for the given conditions
def train_length : ℝ := 320
def time_to_cross : ℝ := 6

-- The speed of the train
def train_speed : ℝ := 53.33

-- The proof statement
theorem train_speed_correct : train_speed = train_length / time_to_cross :=
by
  sorry

end train_speed_correct_l70_70072


namespace optimal_selection_method_uses_golden_ratio_l70_70306

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l70_70306


namespace exists_non_cool_graph_l70_70087

-- Define what it means for a graph to be "cool".
def cool_graph (n : ℕ) (G : Type*) [graph G] : Prop :=
  ∃ (label : G → ℕ) (D : finset ℕ),
  (∀ v1 v2 : G, label v1 ≠ label v2 → label v1 ∈ finset.range (n * n / 4) ∧ 
                label v2 ∈ finset.range (n * n / 4) ∧ 
                (v1 ≠ v2 ↔ (abs (label v1 - label v2) ∈ D)))

-- Main theorem stating the existence of a non-"cool" graph for sufficiently large n.
theorem exists_non_cool_graph (n : ℕ) (hn : n > 100): ∃ (G : Type*) [graph G], ¬ cool_graph n G :=
by
  sorry

end exists_non_cool_graph_l70_70087


namespace anderson_family_seating_l70_70008

def anderson_family_seating_arrangements : Prop :=
  ∃ (family : Fin 5 → String),
    (family 0 = "Mr. Anderson" ∨ family 0 = "Mrs. Anderson") ∧
    (∀ (i : Fin 5), i ≠ 0 → family i ≠ family 0) ∧
    family 1 ≠ family 0 ∧ (family 1 = "Mrs. Anderson" ∨ family 1 = "Child 1" ∨ family 1 = "Child 2") ∧
    family 2 = "Child 3" ∧
    (family 3 ≠ family 0 ∧ family 3 ≠ family 1 ∧ family 3 ≠ family 2) ∧
    (family 4 ≠ family 0 ∧ family 4 ≠ family 1 ∧ family 4 ≠ family 2 ∧ family 4 ≠ family 3) ∧
    (family 3 = "Child 1" ∨ family 3 = "Child 2") ∧
    (family 4 = "Child 1" ∨ family 4 = "Child 2") ∧
    family 3 ≠ family 4 → 
    (2 * 3 * 2 = 12)

theorem anderson_family_seating : anderson_family_seating_arrangements := 
  sorry

end anderson_family_seating_l70_70008


namespace unique_even_odd_decomposition_l70_70714

def is_symmetric (s : Set ℝ) : Prop := ∀ x ∈ s, -x ∈ s

def is_even (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = f x

def is_odd (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = -f x

theorem unique_even_odd_decomposition (s : Set ℝ) (hs : is_symmetric s) (f : ℝ → ℝ) (hf : ∀ x ∈ s, True) :
  ∃! g h : ℝ → ℝ, (is_even g s) ∧ (is_odd h s) ∧ (∀ x ∈ s, f x = g x + h x) :=
sorry

end unique_even_odd_decomposition_l70_70714


namespace quadratic_equation_proof_l70_70061

def is_quadratic_equation (eqn : String) : Prop :=
  eqn = "x^2 + 2x - 1 = 0"

theorem quadratic_equation_proof :
  is_quadratic_equation "x^2 + 2x - 1 = 0" :=
sorry

end quadratic_equation_proof_l70_70061


namespace zoe_bought_8_roses_l70_70919

-- Define the conditions
def each_flower_costs : ℕ := 3
def roses_bought (R : ℕ) : Prop := true
def daisies_bought : ℕ := 2
def total_spent : ℕ := 30

-- The main theorem to prove
theorem zoe_bought_8_roses (R : ℕ) (h1 : total_spent = 30) 
  (h2 : 3 * R + 3 * daisies_bought = total_spent) : R = 8 := by
  sorry

end zoe_bought_8_roses_l70_70919


namespace percentage_of_b_l70_70579

variable (a b c p : ℝ)

theorem percentage_of_b :
  (0.04 * a = 8) →
  (p * b = 4) →
  (c = b / a) →
  p = 1 / (50 * c) :=
by
  sorry

end percentage_of_b_l70_70579


namespace sculpture_paint_area_correct_l70_70224

def sculpture_exposed_area (edge_length : ℝ) (num_cubes_layer1 : ℕ) (num_cubes_layer2 : ℕ) (num_cubes_layer3 : ℕ) : ℝ :=
  let area_top_layer1 := num_cubes_layer1 * edge_length ^ 2
  let area_side_layer1 := 8 * 3 * edge_length ^ 2
  let area_top_layer2 := num_cubes_layer2 * edge_length ^ 2
  let area_side_layer2 := 10 * edge_length ^ 2
  let area_top_layer3 := num_cubes_layer3 * edge_length ^ 2
  let area_side_layer3 := num_cubes_layer3 * 4 * edge_length ^ 2
  area_top_layer1 + area_side_layer1 + area_top_layer2 + area_side_layer2 + area_top_layer3 + area_side_layer3

theorem sculpture_paint_area_correct :
  sculpture_exposed_area 1 12 6 2 = 62 := by
  sorry

end sculpture_paint_area_correct_l70_70224


namespace find_b1_over_b2_l70_70183

variable {a b k a1 a2 b1 b2 : ℝ}

-- Assuming a is inversely proportional to b
def inversely_proportional (a b : ℝ) (k : ℝ) : Prop :=
  a * b = k

-- Define that a_1 and a_2 are nonzero and their ratio is 3/4
def a1_a2_ratio (a1 a2 : ℝ) (ratio : ℝ) : Prop :=
  a1 / a2 = ratio

-- Define that b_1 and b_2 are nonzero
def nonzero (x : ℝ) : Prop :=
  x ≠ 0

theorem find_b1_over_b2 (a1 a2 b1 b2 : ℝ) (h1 : inversely_proportional a b k)
  (h2 : a1_a2_ratio a1 a2 (3 / 4))
  (h3 : nonzero a1) (h4 : nonzero a2) (h5 : nonzero b1) (h6 : nonzero b2) :
  b1 / b2 = 4 / 3 := 
sorry

end find_b1_over_b2_l70_70183


namespace previous_year_height_l70_70709

noncomputable def previous_height (H_current : ℝ) (g : ℝ) : ℝ :=
  H_current / (1 + g)

theorem previous_year_height :
  previous_height 147 0.05 = 140 :=
by
  unfold previous_height
  -- Proof steps would go here
  sorry

end previous_year_height_l70_70709


namespace complex_quadrant_l70_70641

open Complex

noncomputable def z : ℂ := (2 * I) / (1 - I)

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 2 * I) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_quadrant_l70_70641


namespace symmetric_point_x_axis_l70_70045

theorem symmetric_point_x_axis (x y : ℝ) (p : Prod ℝ ℝ) (hx : p = (x, y)) :
  (x, -y) = (1, -2) ↔ (x, y) = (1, 2) :=
by
  sorry

end symmetric_point_x_axis_l70_70045


namespace boxes_needed_l70_70006

-- Define Marilyn's total number of bananas
def num_bananas : Nat := 40

-- Define the number of bananas per box
def bananas_per_box : Nat := 5

-- Calculate the number of boxes required for the given number of bananas and bananas per box
def num_boxes (total_bananas : Nat) (bananas_each_box : Nat) : Nat :=
  total_bananas / bananas_each_box

-- Statement to be proved: given the specific conditions, the result should be 8
theorem boxes_needed : num_boxes num_bananas bananas_per_box = 8 :=
sorry

end boxes_needed_l70_70006


namespace matrix_vector_subtraction_l70_70698

open Matrix

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def matrix_mul_vector (M : Matrix (Fin 2) (Fin 2) ℝ) (v : Fin 2 → ℝ) : Fin 2 → ℝ :=
  M.mulVec v

theorem matrix_vector_subtraction (M : Matrix (Fin 2) (Fin 2) ℝ) (v w : Fin 2 → ℝ)
  (hv : matrix_mul_vector M v = ![4, 6])
  (hw : matrix_mul_vector M w = ![5, -4]) :
  matrix_mul_vector M (v - (2 : ℝ) • w) = ![-6, 14] :=
sorry

end matrix_vector_subtraction_l70_70698


namespace tax_calculation_l70_70275

theorem tax_calculation 
  (total_earnings : ℕ) 
  (deductions : ℕ) 
  (tax_paid : ℕ) 
  (tax_rate_10 : ℚ) 
  (tax_rate_20 : ℚ) 
  (taxable_income : ℕ)
  (X : ℕ)
  (h_total_earnings : total_earnings = 100000)
  (h_deductions : deductions = 30000)
  (h_tax_paid : tax_paid = 12000)
  (h_tax_rate_10 : tax_rate_10 = 10 / 100)
  (h_tax_rate_20 : tax_rate_20 = 20 / 100)
  (h_taxable_income : taxable_income = total_earnings - deductions)
  (h_tax_equation : tax_paid = (tax_rate_10 * X) + (tax_rate_20 * (taxable_income - X))) :
  X = 20000 := 
sorry

end tax_calculation_l70_70275


namespace geometric_sequence_sum_l70_70521

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : a 1 + a 3 = 8)
  (h2 : a 5 + a 7 = 4)
  (geometric_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 9 + a 11 + a 13 + a 15 = 3 :=
by
  sorry

end geometric_sequence_sum_l70_70521


namespace probability_three_out_of_four_odd_dice_l70_70435

theorem probability_three_out_of_four_odd_dice :
  let p_odd := (4 / 8 : ℚ) in
  let p_even := 1 - p_odd in
  let ways := Nat.choose 4 3 in
  (ways * (p_odd^3) * (p_even^1) = (1 / 4 : ℚ)) :=
by
  sorry

end probability_three_out_of_four_odd_dice_l70_70435


namespace masha_can_pay_with_5_ruble_coins_l70_70720

theorem masha_can_pay_with_5_ruble_coins (p c n : ℤ) (h : 2 * p + c + 7 * n = 100) : (p + 3 * c + n) % 5 = 0 :=
  sorry

end masha_can_pay_with_5_ruble_coins_l70_70720


namespace subtract_fifteen_result_l70_70498

theorem subtract_fifteen_result (x : ℕ) (h : x / 10 = 6) : x - 15 = 45 :=
by
  sorry

end subtract_fifteen_result_l70_70498


namespace usual_time_catch_bus_l70_70069

variable (S T T' : ℝ)

theorem usual_time_catch_bus (h1 : T' = T + 6)
  (h2 : S * T = (4 / 5) * S * T') : T = 24 := by
  sorry

end usual_time_catch_bus_l70_70069


namespace div_relation_l70_70835

variable {a b c : ℚ}

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 2/5) : c / a = 5/6 := by
  sorry

end div_relation_l70_70835


namespace sufficient_but_not_necessary_condition_l70_70857

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 2 ∧ y = -1) → (x + y - 1 = 0) ∧ ¬(∀ x y, x + y - 1 = 0 → (x = 2 ∧ y = -1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l70_70857


namespace min_number_of_girls_l70_70368

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l70_70368


namespace value_of_g_of_h_at_2_l70_70683

def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -5 * x^3 + 4

theorem value_of_g_of_h_at_2 : g (h 2) = 3890 := by
  sorry

end value_of_g_of_h_at_2_l70_70683


namespace sampling_methods_l70_70416
-- Import the necessary library

-- Definitions for the conditions of the problem:
def NumberOfFamilies := 500
def HighIncomeFamilies := 125
def MiddleIncomeFamilies := 280
def LowIncomeFamilies := 95
def SampleSize := 100

def FemaleStudentAthletes := 12
def NumberToChoose := 3

-- Define the appropriate sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Stating the proof problem in Lean 4
theorem sampling_methods :
  SamplingMethod.Stratified = SamplingMethod.Stratified ∧
  SamplingMethod.SimpleRandom = SamplingMethod.SimpleRandom :=
by
  -- Proof is omitted in this theorem statement
  sorry

end sampling_methods_l70_70416


namespace min_value_of_quadratic_l70_70130

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 8 * x + 15 → y ≥ -1) ∧ (∃ x₀ : ℝ, x₀ = 4 ∧ (x₀^2 - 8 * x₀ + 15 = -1)) :=
by
  sorry

end min_value_of_quadratic_l70_70130


namespace excess_calories_l70_70846

theorem excess_calories (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ)
  (run_minutes : ℕ) (calories_per_minute : ℕ)
  (h_bags : bags = 3) (h_ounces_per_bag : ounces_per_bag = 2)
  (h_calories_per_ounce : calories_per_ounce = 150)
  (h_run_minutes : run_minutes = 40)
  (h_calories_per_minute : calories_per_minute = 12) :
  (bags * ounces_per_bag * calories_per_ounce) - (run_minutes * calories_per_minute) = 420 := by
  sorry

end excess_calories_l70_70846


namespace roots_of_quadratic_eq_l70_70055

theorem roots_of_quadratic_eq {x y : ℝ} (h1 : x + y = 10) (h2 : (x - y) * (x + y) = 48) : 
    ∃ a b c : ℝ, (a ≠ 0) ∧ (x^2 - a*x + b = 0) ∧ (y^2 - a*y + b = 0) ∧ b = 19.24 := 
by
  sorry

end roots_of_quadratic_eq_l70_70055


namespace circle_and_chord_problem_l70_70636

open Real

-- Definitions of points and line
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (2, 2)
def l (x : ℝ) : ℝ := x - 1

-- Circle equation definition
def circle (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- The main theorem to prove
theorem circle_and_chord_problem :
  (∀ x y, circle 1 1 (sqrt 2) x y ↔ (x-1)^2 + (y-1)^2 = 2) ∧
  (let d := abs(1 - 1 - 1) / sqrt (1^2 + (-1)^2) in
   let chord_length := 2 * sqrt ((sqrt 2)^2 - d^2) in
   chord_length = sqrt 6) :=
by sorry

end circle_and_chord_problem_l70_70636


namespace force_required_for_bolt_b_20_inch_l70_70186

noncomputable def force_inversely_proportional (F L : ℝ) : ℝ := F * L

theorem force_required_for_bolt_b_20_inch (F L : ℝ) :
  let handle_length_10 := 10
  let force_length_product_bolt_a := 3000
  let force_length_product_bolt_b := 4000
  let new_handle_length := 20
  (F * handle_length_10 = 400)
  ∧ (F * new_handle_length = 200)
  → force_inversely_proportional 400 10 = 4000
  ∧ force_inversely_proportional 200 20 = 4000
:=
by
  sorry

end force_required_for_bolt_b_20_inch_l70_70186


namespace optimal_selection_method_uses_golden_ratio_l70_70314

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l70_70314


namespace find_abc_l70_70829

theorem find_abc :
  ∃ a b c : ℝ, 
    -- Conditions
    (a + b + c = 12) ∧ 
    (2 * b = a + c) ∧ 
    ((a + 2) * (c + 5) = (b + 2) * (b + 2)) ∧ 
    -- Correct answers
    ((a = 1 ∧ b = 4 ∧ c = 7) ∨ 
     (a = 10 ∧ b = 4 ∧ c = -2)) := 
  by 
    sorry

end find_abc_l70_70829


namespace volume_of_pyramid_is_one_third_l70_70302

def is_right_triangle (P A B: ℝ) := ∠ P A B = π/2
def isSquareBase (ABCD: ℝ) := ∃ a, ∀ (A B C D: ℝ),
  dist A B = a ∧ dist B C = a ∧ dist C D = a ∧ dist D A = a ∧ dist A C = dist B D

theorem volume_of_pyramid_is_one_third 
  (h1 : isSquareBase ℝ)
  (h2 : ∀ P A B, is_right_triangle P A B)
  (height_apex : ∀ P (ABCD: ℝ), dist P ABCD = 1)
  (dihedral_angle_apex : ∀ P A, ∠ P A = 2 * π / 3) :
  (∃ V, V = 1/3) :=
sorry

end volume_of_pyramid_is_one_third_l70_70302


namespace train_speed_l70_70075

theorem train_speed (d t s : ℝ) (h1 : d = 320) (h2 : t = 6) (h3 : s = 53.33) :
  s = d / t :=
by
  rw [h1, h2]
  sorry

end train_speed_l70_70075


namespace ceil_floor_diff_l70_70472

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l70_70472


namespace calories_in_250_grams_of_lemonade_l70_70278

theorem calories_in_250_grams_of_lemonade:
  ∀ (lemon_juice_grams sugar_grams water_grams total_grams: ℕ)
    (lemon_juice_cal_per_100 sugar_cal_per_100 total_cal: ℕ),
  lemon_juice_grams = 150 →
  sugar_grams = 150 →
  water_grams = 300 →
  total_grams = lemon_juice_grams + sugar_grams + water_grams →
  lemon_juice_cal_per_100 = 30 →
  sugar_cal_per_100 = 386 →
  total_cal = (lemon_juice_grams * lemon_juice_cal_per_100 / 100) + (sugar_grams * sugar_cal_per_100 / 100) →
  (250:ℕ) * total_cal / total_grams = 260 :=
by
  intros lemon_juice_grams sugar_grams water_grams total_grams lemon_juice_cal_per_100 sugar_cal_per_100 total_cal
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end calories_in_250_grams_of_lemonade_l70_70278


namespace total_children_l70_70610

-- Given the conditions
def toy_cars : Nat := 134
def dolls : Nat := 269

-- Prove that the total number of children is 403
theorem total_children (h_cars : toy_cars = 134) (h_dolls : dolls = 269) :
  toy_cars + dolls = 403 :=
by
  sorry

end total_children_l70_70610


namespace function_properties_l70_70164

open Real

noncomputable def f (x : ℝ) : ℝ := log (2 + x) + log (2 - x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ ⦃a b : ℝ⦄, 0 < a → a < b → b < 2 → f b < f a) := by
  sorry

end function_properties_l70_70164


namespace geometric_sequence_S_n_l70_70131

-- Definitions related to the sequence
def a_n (n : ℕ) : ℕ := sorry  -- Placeholder for the actual sequence

-- Sum of the first n terms
def S_n (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms

-- Given conditions
axiom a1 : a_n 1 = 1
axiom Sn_eq_2an_plus1 : ∀ (n : ℕ), S_n n = 2 * a_n (n + 1)

-- Theorem to be proved
theorem geometric_sequence_S_n 
    (n : ℕ) (h : n > 1) 
    : S_n n = (3/2)^(n-1) := 
by 
  sorry

end geometric_sequence_S_n_l70_70131


namespace ab_product_eq_2_l70_70973

theorem ab_product_eq_2 (a b : ℝ) (h : (a + 1)^2 + (b + 2)^2 = 0) : a * b = 2 :=
by sorry

end ab_product_eq_2_l70_70973


namespace complex_number_solution_l70_70235

theorem complex_number_solution (z : ℂ) (h : z * (2 - complex.i) = 3 + complex.i) : z = 1 + complex.i := 
sorry

end complex_number_solution_l70_70235


namespace limit_of_f_at_infinity_l70_70851

open Filter
open Topology

variable (f : ℝ → ℝ)
variable (h_continuous : Continuous f)
variable (h_seq_limit : ∀ α > 0, Tendsto (fun n : ℕ => f (n * α)) atTop (nhds 0))

theorem limit_of_f_at_infinity : Tendsto f atTop (nhds 0) := by
  sorry

end limit_of_f_at_infinity_l70_70851


namespace fly_least_distance_l70_70425

noncomputable def least_distance_fly_crawled (radius height dist_start dist_end : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let slant_height := Real.sqrt (radius^2 + height^2)
  let angle := circumference / slant_height
  let half_angle := angle / 2
  let start_x := dist_start
  let end_x := dist_end * Real.cos half_angle
  let end_y := dist_end * Real.sin half_angle
  Real.sqrt ((end_x - start_x)^2 + end_y^2)

theorem fly_least_distance : least_distance_fly_crawled 500 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 486.396 := by
  sorry

end fly_least_distance_l70_70425


namespace part1_part2_l70_70239

open Real

variables {a b c : ℝ}

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    a + 4 * b + 9 * c ≥ 36 :=
sorry

theorem part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1 / a + 1 / b + 1 / c = 1) :
    (b + c) / sqrt a + (a + c) / sqrt b + (a + b) / sqrt c ≥ 2 * sqrt (a * b * c) :=
sorry

end part1_part2_l70_70239


namespace vector_parallel_solution_l70_70839

theorem vector_parallel_solution 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (x, -9)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  x = -6 :=
by
  sorry

end vector_parallel_solution_l70_70839


namespace x_gt_1_sufficient_but_not_necessary_x_gt_0_l70_70410

theorem x_gt_1_sufficient_but_not_necessary_x_gt_0 (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬(x > 0 → x > 1) :=
by
  sorry

end x_gt_1_sufficient_but_not_necessary_x_gt_0_l70_70410


namespace expand_fraction_product_l70_70615

theorem expand_fraction_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := 
by 
  sorry

end expand_fraction_product_l70_70615


namespace hotel_people_per_room_l70_70586

theorem hotel_people_per_room
  (total_rooms : ℕ := 10)
  (towels_per_person : ℕ := 2)
  (total_towels : ℕ := 60) :
  (total_towels / towels_per_person) / total_rooms = 3 :=
by
  sorry

end hotel_people_per_room_l70_70586


namespace least_multiple_of_25_gt_390_l70_70753

theorem least_multiple_of_25_gt_390 : ∃ n : ℕ, n * 25 > 390 ∧ (∀ m : ℕ, m * 25 > 390 → m * 25 ≥ n * 25) ∧ n * 25 = 400 :=
by
  sorry

end least_multiple_of_25_gt_390_l70_70753


namespace min_girls_in_class_l70_70381

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l70_70381


namespace sum_of_roots_even_l70_70597

theorem sum_of_roots_even (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
    (h_distinct : ∃ x y : ℤ, x ≠ y ∧ (x^2 - 2 * p * x + (p * q) = 0) ∧ (y^2 - 2 * p * y + (p * q) = 0)) :
    Even (2 * p) :=
by 
  sorry

end sum_of_roots_even_l70_70597


namespace max_possible_intersections_l70_70555

theorem max_possible_intersections : 
  let num_x := 12
  let num_y := 6
  let intersections := (num_x * (num_x - 1) / 2) * (num_y * (num_y - 1) / 2)
  intersections = 990 := 
by 
  sorry

end max_possible_intersections_l70_70555


namespace log_expression_evaluation_l70_70925

theorem log_expression_evaluation (log2 log5 : ℝ) (h : log2 + log5 = 1) :
  log2 * (log5 + log10) + 2 * log5 - log5 * log20 = 1 := by
  sorry

end log_expression_evaluation_l70_70925


namespace arithmetic_sequence_sum_l70_70689

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℝ) (d : ℝ), (∀ n, a_n n = a_n 1 + d * (n - 1)) →
  (root1 root2 : ℝ), (root1 + root2 = 10) →
  (root1 = a_n 1) ∧ (root2 = a_n 2015) →
  (a_n 2 + a_n 1008 + a_n 2014 = 15) :=
begin
  intros a_n d a_n_def root1 root2 roots_sum roots_def,
  sorry
end

end arithmetic_sequence_sum_l70_70689


namespace M_in_fourth_quadrant_l70_70142

-- Define the conditions
variables (a b : ℝ)

/-- Condition that point A(a, 3) and B(2, b) are symmetric with respect to the x-axis -/
def symmetric_points : Prop :=
  a = 2 ∧ 3 = -b

-- Define the point M and quadrant check
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- The theorem stating that if A(a, 3) and B(2, b) are symmetric wrt x-axis, M is in the fourth quadrant
theorem M_in_fourth_quadrant (a b : ℝ) (h : symmetric_points a b) : in_fourth_quadrant a b :=
by {
  sorry
}

end M_in_fourth_quadrant_l70_70142


namespace clusters_of_oats_l70_70704

-- Define conditions:
def clusters_per_spoonful : Nat := 4
def spoonfuls_per_bowl : Nat := 25
def bowls_per_box : Nat := 5

-- Define the question and correct answer:
def clusters_per_box : Nat :=
  clusters_per_spoonful * spoonfuls_per_bowl * bowls_per_box

-- Theorem statement for the proof problem:
theorem clusters_of_oats:
  clusters_per_box = 500 :=
by
  sorry

end clusters_of_oats_l70_70704


namespace find_four_digit_number_l70_70936

theorem find_four_digit_number :
  ∃ A B C D : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
    (1001 * A + 100 * B + 10 * C + A) = 182 * (10 * C + D) ∧
    (1000 * A + 100 * B + 10 * C + D) = 2916 :=
by 
  sorry

end find_four_digit_number_l70_70936


namespace minimum_number_of_girls_l70_70377

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l70_70377


namespace minimum_value_l70_70856

theorem minimum_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_condition : (1 / a) + (1 / b) + (1 / c) = 9) : 
    a^3 * b^2 * c ≥ 64 / 729 :=
sorry

end minimum_value_l70_70856


namespace readers_both_l70_70572

-- Define the given conditions
def total_readers : ℕ := 250
def readers_S : ℕ := 180
def readers_L : ℕ := 88

-- Define the proof statement
theorem readers_both : (readers_S + readers_L - total_readers = 18) :=
by
  -- Proof is omitted
  sorry

end readers_both_l70_70572


namespace average_payment_l70_70215

theorem average_payment (total_payments : ℕ) (first_n_payments : ℕ)  (first_payment_amt : ℕ) (remaining_payment_amt : ℕ) 
  (H1 : total_payments = 104)
  (H2 : first_n_payments = 24)
  (H3 : first_payment_amt = 520)
  (H4 : remaining_payment_amt = 615)
  :
  (24 * 520 + 80 * 615) / 104 = 593.08 := 
  by 
    sorry

end average_payment_l70_70215


namespace complex_purely_imaginary_a_eq_3_l70_70144

theorem complex_purely_imaginary_a_eq_3 (a : ℝ) :
  (∀ (a : ℝ), (a^2 - 2*a - 3) + (a + 1)*I = 0 + (a + 1)*I → a = 3) :=
by
  sorry

end complex_purely_imaginary_a_eq_3_l70_70144


namespace rebus_solution_l70_70955

theorem rebus_solution (A B C D : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : D ≠ 0) 
  (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D) :
  1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D) → 
  A = 2 ∧ B = 9 ∧ C = 1 ∧ D = 6 :=
by
  intros h
  sorry

end rebus_solution_l70_70955


namespace binom_7_3_value_l70_70791

-- Define the binomial coefficient.
def binom (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

-- Prove that $\binom{7}{3} = 35$ given the conditions
theorem binom_7_3_value : binom 7 3 = 35 :=
by
  have fact_7 : 7.factorial = 5040 := rfl
  have fact_3 : 3.factorial = 6 := rfl
  have fact_4 : 4.factorial = 24 := rfl
  rw [binom, fact_7, fact_3, fact_4]
  norm_num
  sorry

end binom_7_3_value_l70_70791


namespace day_90_N_minus_1_is_Thursday_l70_70844

/-- 
    Given that the 150th day of year N is a Sunday, 
    and the 220th day of year N+2 is also a Sunday,
    prove that the 90th day of year N-1 is a Thursday.
-/
theorem day_90_N_minus_1_is_Thursday (N : ℕ)
    (h1 : (150 % 7 = 0))  -- 150th day of year N is Sunday
    (h2 : (220 % 7 = 0))  -- 220th day of year N + 2 is Sunday
    : ((90 + 366) % 7 = 4) := -- 366 days in a leap year (N-1), 90th day modulo 7 is Thursday
by
  sorry

end day_90_N_minus_1_is_Thursday_l70_70844


namespace expression_value_l70_70397

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end expression_value_l70_70397


namespace numbers_not_squares_or_cubes_in_200_l70_70668

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l70_70668


namespace range_of_f_l70_70036

open Set

noncomputable def f (x : ℝ) : ℝ := 3^x + 5

theorem range_of_f :
  range f = Ioi 5 :=
sorry

end range_of_f_l70_70036


namespace train_speed_correct_l70_70073

-- Definitions for the given conditions
def train_length : ℝ := 320
def time_to_cross : ℝ := 6

-- The speed of the train
def train_speed : ℝ := 53.33

-- The proof statement
theorem train_speed_correct : train_speed = train_length / time_to_cross :=
by
  sorry

end train_speed_correct_l70_70073


namespace problem_statement_l70_70519

noncomputable def distance_from_line_to_point (a b : ℝ) : ℝ :=
  abs (1 / 2) / (Real.sqrt (a ^ 2 + b ^ 2))

theorem problem_statement (a b : ℝ) (h1 : a = (1 - 2 * b) / 2) (h2 : b = 1 / 2 - a) :
  distance_from_line_to_point a b ≤ Real.sqrt 2 := 
sorry

end problem_statement_l70_70519


namespace mark_eggs_supply_l70_70279

theorem mark_eggs_supply (dozen_eggs: ℕ) (days_in_week: ℕ) : dozen_eggs = 12 → 
  days_in_week = 7 → (5 * dozen_eggs + 30) * days_in_week = 630 :=
by 
  intros h_dozen h_days;
  rw [h_dozen, h_days];
  simp;
  norm_num;
  exact rfl

end mark_eggs_supply_l70_70279


namespace optimal_selection_method_uses_golden_ratio_l70_70312

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l70_70312


namespace lastTwoNonZeroDigits_of_80_fact_is_8_l70_70735

-- Define the factorial function
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Define the function to find the last two nonzero digits of a factorial
def lastTwoNonZeroDigits (n : ℕ) : ℕ := sorry -- Placeholder logic for now

-- State the problem as a theorem
theorem lastTwoNonZeroDigits_of_80_fact_is_8 :
  lastTwoNonZeroDigits 80 = 8 :=
sorry

end lastTwoNonZeroDigits_of_80_fact_is_8_l70_70735


namespace dentist_age_is_32_l70_70009

-- Define the conditions
def one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence (x : ℕ) : Prop :=
  (x - 8) / 6 = (x + 8) / 10

-- State the theorem
theorem dentist_age_is_32 : ∃ x : ℕ, one_sixth_of_age_8_years_ago_eq_one_tenth_of_age_8_years_hence x ∧ x = 32 :=
by
  sorry

end dentist_age_is_32_l70_70009


namespace numbers_neither_square_nor_cube_l70_70656

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l70_70656


namespace scientific_notation_correct_l70_70998

noncomputable def significant_figures : ℝ := 274
noncomputable def decimal_places : ℝ := 8
noncomputable def scientific_notation_rep : ℝ := 2.74 * (10^8)

theorem scientific_notation_correct :
  274000000 = scientific_notation_rep :=
sorry

end scientific_notation_correct_l70_70998


namespace valerie_money_left_l70_70550

theorem valerie_money_left
  (small_bulb_cost : ℕ)
  (large_bulb_cost : ℕ)
  (num_small_bulbs : ℕ)
  (num_large_bulbs : ℕ)
  (initial_money : ℕ) :
  small_bulb_cost = 8 →
  large_bulb_cost = 12 →
  num_small_bulbs = 3 →
  num_large_bulbs = 1 →
  initial_money = 60 →
  initial_money - (num_small_bulbs * small_bulb_cost + num_large_bulbs * large_bulb_cost) = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end valerie_money_left_l70_70550


namespace solve_for_x_l70_70562

theorem solve_for_x (x : ℚ) : (3 * x / 7 - 2 = 12) → (x = 98 / 3) :=
by
  intro h
  sorry

end solve_for_x_l70_70562


namespace max_value_of_N_l70_70511

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l70_70511


namespace b_minus_a_l70_70496

theorem b_minus_a :
  ∃ (a b : ℝ), (2 + 4 = -a) ∧ (2 * 4 = b) ∧ (b - a = 14) :=
by
  use (-6 : ℝ)
  use (8 : ℝ)
  simp
  sorry

end b_minus_a_l70_70496


namespace total_fruit_punch_l70_70181

/-- Conditions -/
def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def apple_juice : ℝ := cherry_punch - 1.5
def pineapple_juice : ℝ := 3
def grape_punch : ℝ := 1.5 * apple_juice

/-- Proof that total fruit punch is 35.25 liters -/
theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end total_fruit_punch_l70_70181


namespace correct_statements_count_l70_70358

-- Definition of the statements
def statement1 := ∀ (q : ℚ), q > 0 ∨ q < 0
def statement2 (a : ℝ) := |a| = -a → a < 0
def statement3 := ∀ (x y : ℝ), 0 = 3
def statement4 := ∀ (q : ℚ), ∃ (p : ℝ), q = p
def statement5 := 7 = 7 ∧ 10 = 10 ∧ 15 = 15

-- Define what it means for each statement to be correct
def is_correct_statement1 := statement1 = false
def is_correct_statement2 := ∀ a : ℝ, statement2 a = false
def is_correct_statement3 := statement3 = false
def is_correct_statement4 := statement4 = true
def is_correct_statement5 := statement5 = true

-- Define the problem and its correct answer
def problem := is_correct_statement1 ∧ is_correct_statement2 ∧ is_correct_statement3 ∧ is_correct_statement4 ∧ is_correct_statement5

-- Prove that the number of correct statements is 2
theorem correct_statements_count : problem → (2 = 2) :=
by
  intro h
  sorry

end correct_statements_count_l70_70358


namespace solve_equation_l70_70396

theorem solve_equation (x : ℝ) :
  (3 / x - (1 / x * 6 / x) = -2.5) ↔ (x = (-3 + Real.sqrt 69) / 5 ∨ x = (-3 - Real.sqrt 69) / 5) :=
by {
  sorry
}

end solve_equation_l70_70396


namespace product_of_solutions_l70_70957

theorem product_of_solutions : 
  (∃ x1 x2 : ℝ, |5 * x1 - 1| + 4 = 54 ∧ |5 * x2 - 1| + 4 = 54 ∧ x1 * x2 = -99.96) :=
  by sorry

end product_of_solutions_l70_70957


namespace optimal_selection_method_uses_golden_ratio_l70_70349

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l70_70349


namespace product_of_three_digit_numbers_ends_with_four_zeros_l70_70608

/--
Theorem: There exist three three-digit numbers formed using nine different digits such that their product ends with four zeros.
-/
theorem product_of_three_digit_numbers_ends_with_four_zeros :
  ∃ (x y z : ℕ), 100 ≤ x ∧ x < 1000 ∧
                 100 ≤ y ∧ y < 1000 ∧
                 100 ≤ z ∧ z < 1000 ∧
                 (∀ d ∈ (list.digits x).union (list.digits y).union (list.digits z), 
                     list.count d ((list.digits x).union (list.digits y).union (list.digits z)) = 1) ∧
                 (x * y * z) % 10000 = 0 :=
sorry

end product_of_three_digit_numbers_ends_with_four_zeros_l70_70608


namespace trajectory_of_center_l70_70484

-- Define the fixed circle C as x^2 + (y + 3)^2 = 1
def fixed_circle (p : ℝ × ℝ) : Prop :=
  (p.1)^2 + (p.2 + 3)^2 = 1

-- Define the line y = 2
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2

-- The main theorem stating the trajectory of the center of circle M is x^2 = -12y
theorem trajectory_of_center :
  ∀ (M : ℝ × ℝ), 
  tangent_line M → (∃ r : ℝ, fixed_circle (M.1, M.2 - r) ∧ r > 0) →
  (M.1)^2 = -12 * M.2 :=
sorry

end trajectory_of_center_l70_70484


namespace not_speaking_hindi_is_32_l70_70533

-- Definitions and conditions
def total_diplomats : ℕ := 120
def spoke_french : ℕ := 20
def percent_neither : ℝ := 0.20
def percent_both : ℝ := 0.10

-- Number of diplomats who spoke neither French nor Hindi
def neither_french_nor_hindi := (percent_neither * total_diplomats : ℝ)

-- Number of diplomats who spoke both French and Hindi
def both_french_and_hindi := (percent_both * total_diplomats : ℝ)

-- Number of diplomats who spoke only French
def only_french := (spoke_french - both_french_and_hindi : ℝ)

-- Number of diplomats who did not speak Hindi
def not_speaking_hindi := (only_french + neither_french_nor_hindi : ℝ)

theorem not_speaking_hindi_is_32 :
  not_speaking_hindi = 32 :=
by
  -- Provide proof here
  sorry

end not_speaking_hindi_is_32_l70_70533


namespace find_common_difference_l70_70811

variable {aₙ : ℕ → ℝ}
variable {Sₙ : ℕ → ℝ}

-- Condition that the sum of the first n terms of the arithmetic sequence is S_n
def is_arith_seq (aₙ : ℕ → ℝ) (Sₙ : ℕ → ℝ) : Prop :=
  ∀ n, Sₙ n = (n * (aₙ 0 + (aₙ (n - 1))) / 2)

-- Condition given in the problem
def problem_condition (Sₙ : ℕ → ℝ) : Prop :=
  2 * Sₙ 3 - 3 * Sₙ 2 = 12

theorem find_common_difference (h₀ : is_arith_seq aₙ Sₙ) (h₁ : problem_condition Sₙ) : 
  ∃ d : ℝ, d = 4 := 
sorry

end find_common_difference_l70_70811


namespace all_statements_correct_l70_70534

theorem all_statements_correct :
  (∀ (b h : ℝ), (3 * b * h = 3 * (b * h))) ∧
  (∀ (b h : ℝ), (1/2 * b * (1/2 * h) = 1/2 * (1/2 * b * h))) ∧
  (∀ (r : ℝ), (π * (2 * r) ^ 2 = 4 * (π * r ^ 2))) ∧
  (∀ (r : ℝ), (π * (3 * r) ^ 2 = 9 * (π * r ^ 2))) ∧
  (∀ (s : ℝ), ((2 * s) ^ 2 = 4 * (s ^ 2)))
  → False := 
by 
  intros h
  sorry

end all_statements_correct_l70_70534


namespace vanessa_savings_weeks_l70_70391

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end vanessa_savings_weeks_l70_70391


namespace car_selection_proportion_l70_70182

def production_volume_emgrand : ℕ := 1600
def production_volume_king_kong : ℕ := 6000
def production_volume_freedom_ship : ℕ := 2000
def total_selected_cars : ℕ := 48

theorem car_selection_proportion :
  (8, 30, 10) = (
    total_selected_cars * production_volume_emgrand /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_king_kong /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship),
    total_selected_cars * production_volume_freedom_ship /
    (production_volume_emgrand + production_volume_king_kong + production_volume_freedom_ship)
  ) :=
by sorry

end car_selection_proportion_l70_70182


namespace lines_forming_angle_bamboo_pole_longest_shadow_angle_l70_70868

-- Define the angle between sunlight and ground
def angle_sunlight_ground : ℝ := 60

-- Proof problem 1 statement
theorem lines_forming_angle (A : ℝ) : 
  (A > angle_sunlight_ground → ∃ l : ℕ, l = 0) ∧ (A < angle_sunlight_ground → ∃ l : ℕ, ∀ n : ℕ, n > l) :=
  sorry

-- Proof problem 2 statement
theorem bamboo_pole_longest_shadow_angle : 
  ∀ bamboo_pole_angle ground_angle : ℝ, 
  (ground_angle = 60 → bamboo_pole_angle = 30) :=
  sorry

end lines_forming_angle_bamboo_pole_longest_shadow_angle_l70_70868


namespace number_of_factors_1320_l70_70931

/-- 
  Determine how many distinct, positive factors the number 1320 has.
-/
theorem number_of_factors_1320 : 
  (finset.range (bit0 (bit3 (bit0 (bit0 1))))) = 
  {n | ∃ a b c d : ℕ, (2 ^ a * 3 ^ b * 5 ^ c * 11 ^ d = n) ∧ (a ≤ 3) ∧ (b ≤ 1) ∧ (c ≤ 1) ∧ (d ≤ 1)}.card = 32 :=
sorry

end number_of_factors_1320_l70_70931


namespace find_a_l70_70978

-- Given conditions
def expand_term (a b : ℝ) (r : ℕ) : ℝ :=
  (Nat.choose 7 r) * (a ^ (7 - r)) * (b ^ r)

def coefficient_condition (a : ℝ) : Prop :=
  expand_term a 1 7 * 1 = 1

-- Main statement to prove
theorem find_a (a : ℝ) : coefficient_condition a → a = 1 / 7 :=
by
  intros h
  sorry

end find_a_l70_70978


namespace union_inter_example_l70_70699

noncomputable def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
noncomputable def B : Set ℕ := {4, 7, 8, 9}

theorem union_inter_example :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (A ∩ B = {4, 7, 8}) :=
by
  sorry

end union_inter_example_l70_70699


namespace num_real_roots_eq_two_l70_70546

theorem num_real_roots_eq_two : 
  ∀ x : ℝ, (∃ r : ℕ, r = 2 ∧ (abs (x^2 - 1) = 1/10 * (x + 9/10) → x = r)) := sorry

end num_real_roots_eq_two_l70_70546


namespace no_number_exists_decreasing_by_removing_digit_l70_70612

theorem no_number_exists_decreasing_by_removing_digit :
  ¬ ∃ (x y n : ℕ), x * 10^n + y = 58 * y :=
by
  sorry

end no_number_exists_decreasing_by_removing_digit_l70_70612


namespace girls_more_than_boys_l70_70500

-- Given conditions
def ratio_boys_girls : ℕ := 3
def ratio_girls_boys : ℕ := 4
def total_students : ℕ := 42

-- Theorem statement
theorem girls_more_than_boys : 
  let x := total_students / (ratio_boys_girls + ratio_girls_boys)
  let boys := ratio_boys_girls * x
  let girls := ratio_girls_boys * x
  girls - boys = 6 := by
  sorry

end girls_more_than_boys_l70_70500


namespace scientific_notation_of_274000000_l70_70995

theorem scientific_notation_of_274000000 :
  (274000000 : ℝ) = 2.74 * 10 ^ 8 :=
by
    sorry

end scientific_notation_of_274000000_l70_70995


namespace trig_sum_identity_l70_70105

theorem trig_sum_identity :
  Real.sin (47 * Real.pi / 180) * Real.cos (43 * Real.pi / 180) 
  + Real.sin (137 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) = 1 :=
by
  sorry

end trig_sum_identity_l70_70105


namespace inequality_amgm_l70_70722

theorem inequality_amgm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : a^3 + b^3 + a + b ≥ 4 * a * b :=
sorry

end inequality_amgm_l70_70722


namespace symmetric_angle_of_inclination_l70_70640

theorem symmetric_angle_of_inclination (α₁ : ℝ) (h : 0 ≤ α₁ ∧ α₁ < π) : 
  (∃ β₁ : ℝ, (α₁ = 0 ∧ β₁ = 0) ∨ (0 < α₁ ∧ α₁ < π ∧ β₁ = π - α₁)) :=
by
  sorry

end symmetric_angle_of_inclination_l70_70640


namespace hypotenuse_length_l70_70218

-- Define the conditions
def right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- State the theorem using the conditions and correct answer
theorem hypotenuse_length : right_triangle 20 21 29 :=
by
  -- To be filled in by proof steps
  sorry

end hypotenuse_length_l70_70218


namespace find_symmetric_point_l70_70233

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def M : Point := ⟨3, -3, -1⟩

def line (x y z : ℝ) : Prop := 
  (x - 6) / 5 = (y - 3.5) / 4 ∧ (x - 6) / 5 = (z + 0.5) / 0

theorem find_symmetric_point (M' : Point) :
  (line M.x M.y M.z) →
  M' = ⟨-1, 2, 0⟩ := by
  sorry

end find_symmetric_point_l70_70233


namespace unripe_oranges_per_day_l70_70969

/-
Problem: Prove that if after 6 days, they will have 390 sacks of unripe oranges, then the number of sacks of unripe oranges harvested per day is 65.
-/

theorem unripe_oranges_per_day (total_sacks : ℕ) (days : ℕ) (harvest_per_day : ℕ)
  (h1 : days = 6)
  (h2 : total_sacks = 390)
  (h3 : harvest_per_day = total_sacks / days) :
  harvest_per_day = 65 :=
by
  sorry

end unripe_oranges_per_day_l70_70969


namespace kite_AB_BC_ratio_l70_70771

-- Define the kite properties and necessary elements to state the problem
def kite_problem (AB BC: ℝ) (angleB angleD : ℝ) (MN'_parallel_AC : Prop) : Prop :=
  angleB = 90 ∧ angleD = 90 ∧ MN'_parallel_AC ∧ AB / BC = (1 + Real.sqrt 5) / 2

-- Define the main theorem to be proven
theorem kite_AB_BC_ratio (AB BC : ℝ) (angleB angleD : ℝ) (MN'_parallel_AC : Prop) :
  kite_problem AB BC angleB angleD MN'_parallel_AC :=
by
  sorry

-- Statement of the condition that need to be satisfied
axiom MN'_parallel_AC : Prop

-- Example instantiation of the problem
example : kite_problem 1 1 90 90 MN'_parallel_AC :=
by
  sorry

end kite_AB_BC_ratio_l70_70771


namespace dingding_minimum_correct_answers_l70_70611

theorem dingding_minimum_correct_answers (x : ℕ) :
  (5 * x - (30 - x) > 100) → x ≥ 22 :=
by
  sorry

end dingding_minimum_correct_answers_l70_70611


namespace find_slope_of_line_l_l70_70119

theorem find_slope_of_line_l :
  ∃ k : ℝ, (k = 3 * Real.sqrt 5 / 10 ∨ k = -3 * Real.sqrt 5 / 10) :=
by
  -- Given conditions
  let F1 : ℝ := 6 / 5 * Real.sqrt 5
  let PF : ℝ := 4 / 5 * Real.sqrt 5
  let slope_PQ : ℝ := 1
  let slope_RF1 : ℝ := sorry  -- we need to prove/extract this from the given
  let k := 3 / 2 * slope_RF1
  -- to prove this
  sorry

end find_slope_of_line_l_l70_70119


namespace problem_statement_l70_70493

theorem problem_statement (a : ℝ) (h : a^2 - 2 * a + 1 = 0) : 4 * a - 2 * a^2 + 2 = 4 := 
sorry

end problem_statement_l70_70493


namespace angle_bisectors_triangle_l70_70269

theorem angle_bisectors_triangle
  (A B C I D K E : Type)
  (triangle : ∀ (A B C : Type), Prop)
  (is_incenter : ∀ (I A B C : Type), Prop)
  (is_on_arc_centered_at : ∀ (X Y : Type), Prop)
  (is_altitude_intersection : ∀ (X Y : Type), Prop)
  (angle_BIC : ∀ (B C : Type), ℝ)
  (angle_DKE : ∀ (D K E : Type), ℝ)
  (α β γ : ℝ)
  (h_sum_ang : α + β + γ = 180) :
  is_incenter I A B C →
  is_on_arc_centered_at D A → is_on_arc_centered_at K A → is_on_arc_centered_at E A →
  is_altitude_intersection E A →
  angle_BIC B C = 180 - (β + γ) / 2 →
  angle_DKE D K E = (360 - α) / 2 →
  angle_BIC B C + angle_DKE D K E = 270 :=
by sorry

end angle_bisectors_triangle_l70_70269


namespace math_problem_l70_70465

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l70_70465


namespace geometric_sequence_a7_l70_70808

theorem geometric_sequence_a7 (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 2 + a 3 = 6)
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  a 7 = 64 := by
  sorry

end geometric_sequence_a7_l70_70808


namespace sum_sequence_formula_l70_70810

theorem sum_sequence_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) ∧ a 1 = 1 →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) :=
by sorry

end sum_sequence_formula_l70_70810


namespace ceil_floor_diff_l70_70469

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l70_70469


namespace compare_abc_l70_70125

noncomputable def a := Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def b := Real.cos (Real.pi / 6) ^ 2 - Real.sin (Real.pi / 6) ^ 2
noncomputable def c := Real.tan (30 * Real.pi / 180) / (1 - Real.tan (30 * Real.pi / 180) ^ 2)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l70_70125


namespace find_distance_l70_70739

-- Definitions based on the given conditions
def speed_of_boat := 16 -- in kmph
def speed_of_stream := 2 -- in kmph
def total_time := 960 -- in hours
def downstream_speed := speed_of_boat + speed_of_stream
def upstream_speed := speed_of_boat - speed_of_stream

-- Prove that the distance D is 7590 km given the total time and speeds
theorem find_distance (D : ℝ) :
  (D / downstream_speed + D / upstream_speed = total_time) → D = 7590 :=
by
  sorry

end find_distance_l70_70739


namespace calculate_savings_l70_70441

theorem calculate_savings :
  let income := 5 * (45000 + 35000 + 7000 + 10000 + 13000),
  let expenses := 5 * (30000 + 10000 + 5000 + 4500 + 9000),
  let initial_savings := 849400
in initial_savings + income - expenses = 1106900 := by sorry

end calculate_savings_l70_70441


namespace optimal_selection_golden_ratio_l70_70307

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l70_70307


namespace investment_amount_l70_70551

noncomputable def calculate_principal (A : ℕ) (r t : ℝ) (n : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_amount (A : ℕ) (r t : ℝ) (n P : ℕ) :
  A = 70000 → r = 0.08 → t = 5 → n = 12 →
  P = 46994 →
  calculate_principal A r t n = P :=
by
  intros hA hr ht hn hP
  rw [hA, hr, ht, hn, hP]
  sorry

end investment_amount_l70_70551


namespace optimal_selection_uses_golden_ratio_l70_70330

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l70_70330


namespace max_area_triangle_has_max_area_l70_70554

noncomputable def max_area_triangle (PQ QR_ratio RP_ratio : ℝ) : ℝ :=
  -- constants based on given problem
  let PQ := 15
  let QR_ratio := 5
  let RP_ratio := 9
  -- let y be a positive real representing segments QR and RP
  ∀ (y : ℝ), 3 < y ∧ y < 5 → 
  -- lengths of the triangle sides QR and RP
  let QR := QR_ratio * y
  let RP := RP_ratio * y
  -- semi-perimeter s
  let s := (PQ + QR + RP) / 2
  -- area calculation using Heron's formula
  let area_squared := s * (s - PQ) * (s - QR) * (s - RP)
  -- calculate upper bound
  max_area = 612.5

theorem max_area_triangle_has_max_area :
  max_area_triangle 15 5 9 = 612.5 := sorry  

end max_area_triangle_has_max_area_l70_70554


namespace percentage_exceed_l70_70497

theorem percentage_exceed (x y : ℝ) (h : y = x + (0.25 * x)) : (y - x) / x * 100 = 25 :=
by
  sorry

end percentage_exceed_l70_70497


namespace delaney_left_home_at_7_50_l70_70928

theorem delaney_left_home_at_7_50 :
  (bus_time = 8 * 60 ∧ travel_time = 30 ∧ miss_time = 20) →
  (delaney_leave_time = bus_time + miss_time - travel_time) →
  delaney_leave_time = 7 * 60 + 50 :=
by
  intros
  sorry

end delaney_left_home_at_7_50_l70_70928


namespace find_a_l70_70644

-- Define the sets A and B and the condition that A union B is a subset of A intersect B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) :
  A ∪ B a ⊆ A ∩ B a → a = 1 :=
sorry

end find_a_l70_70644


namespace hua_luogeng_optimal_selection_l70_70316

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l70_70316


namespace scientific_notation_correct_l70_70997

noncomputable def significant_figures : ℝ := 274
noncomputable def decimal_places : ℝ := 8
noncomputable def scientific_notation_rep : ℝ := 2.74 * (10^8)

theorem scientific_notation_correct :
  274000000 = scientific_notation_rep :=
sorry

end scientific_notation_correct_l70_70997


namespace optimal_selection_uses_golden_ratio_l70_70327

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l70_70327


namespace vanessa_weeks_to_wait_l70_70394

theorem vanessa_weeks_to_wait
  (dress_cost savings : ℕ)
  (weekly_allowance weekly_expense : ℕ)
  (h₀ : dress_cost = 80)
  (h₁ : savings = 20)
  (h₂ : weekly_allowance = 30)
  (h₃ : weekly_expense = 10) :
  let net_savings_per_week := weekly_allowance - weekly_expense,
      additional_amount_needed := dress_cost - savings in
  additional_amount_needed / net_savings_per_week = 3 :=
by
  sorry

end vanessa_weeks_to_wait_l70_70394


namespace set_A_range_l70_70160

def A := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ (-1 ≤ x ∧ x ≤ 2)}

theorem set_A_range :
  A = {y : ℝ | -4 ≤ y ∧ y ≤ 0} :=
sorry

end set_A_range_l70_70160


namespace ceil_floor_difference_l70_70453

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l70_70453


namespace sum_remainder_l70_70401

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 12) (h2 : b % 30 = 9) (h3 : c % 30 = 15) :
  (a + b + c) % 30 = 6 := 
sorry

end sum_remainder_l70_70401


namespace linear_avoid_third_quadrant_l70_70149

theorem linear_avoid_third_quadrant (k b : ℝ) (h : ∀ x : ℝ, k * x + b ≥ 0 → k * x + b > 0 → (k * x + b ≥ 0) ∧ (x ≥ 0)) :
  k < 0 ∧ b ≥ 0 :=
by
  sorry

end linear_avoid_third_quadrant_l70_70149


namespace ratio_of_triangles_in_octagon_l70_70773

-- Conditions
def regular_octagon_division : Prop := 
  let L := 1 -- Area of each small congruent right triangle
  let ABJ := 2 * L -- Area of triangle ABJ
  let ADE := 6 * L -- Area of triangle ADE
  (ABJ / ADE = (1:ℝ) / 3)

-- Statement
theorem ratio_of_triangles_in_octagon : regular_octagon_division := by
  sorry

end ratio_of_triangles_in_octagon_l70_70773


namespace leonardo_nap_duration_l70_70524

theorem leonardo_nap_duration (h : (1 : ℝ) / 5 * 60 = 12) : (1 / 5 : ℝ) * 60 = 12 :=
by 
  exact h

end leonardo_nap_duration_l70_70524


namespace linear_avoid_third_quadrant_l70_70150

theorem linear_avoid_third_quadrant (k b : ℝ) (h : ∀ x : ℝ, k * x + b ≥ 0 → k * x + b > 0 → (k * x + b ≥ 0) ∧ (x ≥ 0)) :
  k < 0 ∧ b ≥ 0 :=
by
  sorry

end linear_avoid_third_quadrant_l70_70150


namespace sum_in_base4_eq_in_base5_l70_70959

def base4_to_base5 (n : ℕ) : ℕ := sorry -- Placeholder for the conversion function

theorem sum_in_base4_eq_in_base5 :
  base4_to_base5 (203 + 112 + 321) = 2222 := 
sorry

end sum_in_base4_eq_in_base5_l70_70959


namespace jill_braids_dancers_l70_70848

def dancers_on_team (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) (total_time_seconds : ℕ) : ℕ :=
  total_time_seconds / seconds_per_braid / braids_per_dancer

theorem jill_braids_dancers (h1 : braids_per_dancer = 5) (h2 : seconds_per_braid = 30)
                             (h3 : total_time_seconds = 20 * 60) : 
  dancers_on_team braids_per_dancer seconds_per_braid total_time_seconds = 8 :=
by
  sorry

end jill_braids_dancers_l70_70848


namespace probability_of_spinner_landing_on_C_l70_70908

theorem probability_of_spinner_landing_on_C :
  ∀ (PA PB PD PC : ℚ),
  PA = 1/4 →
  PB = 1/3 →
  PD = 1/6 →
  PA + PB + PD + PC = 1 →
  PC = 1/4 :=
by
  intros PA PB PD PC hPA hPB hPD hsum
  rw [hPA, hPB, hPD] at hsum
  sorry

end probability_of_spinner_landing_on_C_l70_70908


namespace length_of_bridge_l70_70067

theorem length_of_bridge (train_length : ℕ) (train_speed : ℕ) (cross_time : ℕ) 
  (h1 : train_length = 150) 
  (h2 : train_speed = 45) 
  (h3 : cross_time = 30) : 
  ∃ bridge_length : ℕ, bridge_length = 225 := sorry

end length_of_bridge_l70_70067


namespace hotel_charge_comparison_l70_70303

def charge_R (R G : ℝ) (P : ℝ) : Prop :=
  P = 0.8 * R ∧ P = 0.9 * G

def discounted_charge_R (R2 : ℝ) (R : ℝ) : Prop :=
  R2 = 0.85 * R

theorem hotel_charge_comparison (R G P R2 : ℝ)
  (h1 : charge_R R G P)
  (h2 : discounted_charge_R R2 R)
  (h3 : R = 1.125 * G) :
  R2 = 0.95625 * G := by
  sorry

end hotel_charge_comparison_l70_70303


namespace chocolates_difference_l70_70900

theorem chocolates_difference (robert_chocolates : ℕ) (nickel_chocolates : ℕ) (h1 : robert_chocolates = 7) (h2 : nickel_chocolates = 5) : robert_chocolates - nickel_chocolates = 2 :=
by sorry

end chocolates_difference_l70_70900


namespace area_of_right_square_l70_70596

theorem area_of_right_square (side_length_left : ℕ) (side_length_left_eq : side_length_left = 10) : ∃ area_right, area_right = 68 := 
by
  sorry

end area_of_right_square_l70_70596


namespace cone_cylinder_volume_ratio_l70_70802

theorem cone_cylinder_volume_ratio :
  let π := Real.pi
  let Vcylinder := π * (3:ℝ)^2 * (15:ℝ)
  let Vcone := (1/3:ℝ) * π * (2:ℝ)^2 * (5:ℝ)
  (Vcone / Vcylinder) = (4 / 81) :=
by
  let π := Real.pi
  let r_cylinder := (3:ℝ)
  let h_cylinder := (15:ℝ)
  let r_cone := (2:ℝ)
  let h_cone := (5:ℝ)
  let Vcylinder := π * r_cylinder^2 * h_cylinder
  let Vcone := (1/3:ℝ) * π * r_cone^2 * h_cone
  have h1 : Vcylinder = 135 * π := by sorry
  have h2 : Vcone = (20 / 3) * π := by sorry
  have h3 : (Vcone / Vcylinder) = (4 / 81) := by sorry
  exact h3

end cone_cylinder_volume_ratio_l70_70802


namespace monotonic_increasing_iff_a_lt_neg_4_l70_70894

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a) / (x - 2)

theorem monotonic_increasing_iff_a_lt_neg_4 (a : ℝ) :
  (∀ x y, 2 < x → 2 < y → x < y → f x a < f y a) ↔ a < -4 := 
  sorry

end monotonic_increasing_iff_a_lt_neg_4_l70_70894


namespace regular_milk_cartons_l70_70077

variable (R C : ℕ)
variable (h1 : C + R = 24)
variable (h2 : C = 7 * R)

theorem regular_milk_cartons : R = 3 :=
by
  sorry

end regular_milk_cartons_l70_70077


namespace overlapping_area_of_congruent_isosceles_triangles_l70_70199

noncomputable def isosceles_right_triangle (hypotenuse : ℝ) := 
  {l : ℝ // l = hypotenuse / Real.sqrt 2}

theorem overlapping_area_of_congruent_isosceles_triangles (hypotenuse : ℝ) 
  (A₁ A₂ : isosceles_right_triangle hypotenuse) (h_congruent : A₁ = A₂) :
  hypotenuse = 10 → 
  let leg := hypotenuse / Real.sqrt 2 
  let area := (leg * leg) / 2 
  let shared_area := area / 2 
  shared_area = 12.5 :=
by
  sorry

end overlapping_area_of_congruent_isosceles_triangles_l70_70199


namespace beetle_crawls_100th_segment_in_1300_seconds_l70_70423

def segment_length (n : ℕ) : ℕ :=
  (n / 4) + 1

def total_length (s : ℕ) : ℕ :=
  (s / 4) * 4 * (segment_length (s - 1)) * (segment_length (s - 1) + 1) / 2

theorem beetle_crawls_100th_segment_in_1300_seconds :
  total_length 100 = 1300 :=
  sorry

end beetle_crawls_100th_segment_in_1300_seconds_l70_70423


namespace anne_wandering_time_l70_70140

theorem anne_wandering_time (distance speed : ℝ) (h_dist : distance = 3.0) (h_speed : speed = 2.0) : 
  distance / speed = 1.5 :=
by
  rw [h_dist, h_speed]
  norm_num

end anne_wandering_time_l70_70140


namespace cards_left_l70_70158
noncomputable section

def initial_cards : ℕ := 676
def bought_cards : ℕ := 224

theorem cards_left : initial_cards - bought_cards = 452 := 
by
  sorry

end cards_left_l70_70158


namespace mean_proportional_l70_70625

variable (a b c d : ℕ)
variable (x : ℕ)

def is_geometric_mean (a b : ℕ) (x : ℕ) := x = Int.sqrt (a * b)

theorem mean_proportional (h49 : a = 49) (h64 : b = 64) (h81 : d = 81)
  (h_geometric1 : x = 56) (h_geometric2 : c = 72) :
  c = 64 := sorry

end mean_proportional_l70_70625


namespace angela_sleep_difference_l70_70780

theorem angela_sleep_difference :
  let december_sleep_hours := 6.5
  let january_sleep_hours := 8.5
  let december_days := 31
  let january_days := 31
  (january_sleep_hours * january_days) - (december_sleep_hours * december_days) = 62 :=
by
  sorry

end angela_sleep_difference_l70_70780


namespace sum_of_integers_with_product_2720_l70_70876

theorem sum_of_integers_with_product_2720 (n : ℤ) (h1 : n > 0) (h2 : n * (n + 2) = 2720) : n + (n + 2) = 104 :=
by {
  sorry
}

end sum_of_integers_with_product_2720_l70_70876


namespace log_sum_equals_18084_l70_70764

theorem log_sum_equals_18084 : 
  (Finset.sum (Finset.range 2013) (λ x => (Int.floor (Real.log x / Real.log 2)))) = 18084 :=
by
  sorry

end log_sum_equals_18084_l70_70764


namespace round_robin_teams_l70_70552

theorem round_robin_teams (x : ℕ) (h : x ≠ 0) :
  (x * (x - 1)) / 2 = 15 → ∃ n : ℕ, x = n :=
by
  sorry

end round_robin_teams_l70_70552


namespace sum_of_Q_and_R_in_base_8_l70_70138

theorem sum_of_Q_and_R_in_base_8 (P Q R : ℕ) (hp : 1 ≤ P ∧ P < 8) (hq : 1 ≤ Q ∧ Q < 8) (hr : 1 ≤ R ∧ R < 8) 
  (hdistinct : P ≠ Q ∧ Q ≠ R ∧ P ≠ R) (H : 8^2 * P + 8 * Q + R + (8^2 * R + 8 * Q + P) + (8^2 * Q + 8 * P + R) 
  = 8^3 * P + 8^2 * P + 8 * P) : Q + R = 7 := 
sorry

end sum_of_Q_and_R_in_base_8_l70_70138


namespace optimalSelectionUsesGoldenRatio_l70_70335

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l70_70335


namespace arithmetic_sequence_problem_l70_70122

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ A, ∀ n : ℕ, a n = A * (q ^ (n - 1))

theorem arithmetic_sequence_problem
  (q : ℝ) 
  (h1 : q > 1)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h_seq : is_arithmetic_sequence a q) : 
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := 
by 
  sorry

end arithmetic_sequence_problem_l70_70122


namespace big_al_bananas_l70_70228

-- Define conditions for the arithmetic sequence and total consumption
theorem big_al_bananas (a : ℕ) : 
  (a + (a + 6) + (a + 12) + (a + 18) + (a + 24) = 100) → 
  (a + 24 = 32) :=
by
  sorry

end big_al_bananas_l70_70228


namespace numbers_not_perfect_squares_or_cubes_l70_70678

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l70_70678


namespace group_C_both_axis_and_central_l70_70918

def is_axisymmetric (shape : Type) : Prop := sorry
def is_centrally_symmetric (shape : Type) : Prop := sorry

def square : Type := sorry
def rhombus : Type := sorry
def rectangle : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def isosceles_triangle : Type := sorry

def group_A := [square, rhombus, rectangle, parallelogram]
def group_B := [equilateral_triangle, square, rhombus, rectangle]
def group_C := [square, rectangle, rhombus]
def group_D := [parallelogram, square, isosceles_triangle]

def all_axisymmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_axisymmetric shape

def all_centrally_symmetric (group : List Type) : Prop :=
  ∀ shape ∈ group, is_centrally_symmetric shape

theorem group_C_both_axis_and_central :
  (all_axisymmetric group_C ∧ all_centrally_symmetric group_C) ∧
  (∀ (group : List Type), (all_axisymmetric group ∧ all_centrally_symmetric group) →
    group = group_C) :=
by sorry

end group_C_both_axis_and_central_l70_70918


namespace smaller_denom_is_five_l70_70093

-- Define the conditions
def num_smaller_bills : ℕ := 4
def num_ten_dollar_bills : ℕ := 8
def total_bills : ℕ := num_smaller_bills + num_ten_dollar_bills
def ten_dollar_bill_value : ℕ := 10
def total_value : ℕ := 100

-- Define the smaller denomination value
def value_smaller_denom (x : ℕ) : Prop :=
  num_smaller_bills * x + num_ten_dollar_bills * ten_dollar_bill_value = total_value

-- Prove that the value of the smaller denomination bill is 5
theorem smaller_denom_is_five : value_smaller_denom 5 :=
by
  sorry

end smaller_denom_is_five_l70_70093


namespace average_students_is_12_l70_70265

-- Definitions based on the problem's conditions
variables (a b c : Nat)

-- Given conditions
axiom condition1 : a + b + c = 30
axiom condition2 : a + c = 19
axiom condition3 : b + c = 9

-- Prove that the number of average students (c) is 12
theorem average_students_is_12 : c = 12 := by 
  sorry

end average_students_is_12_l70_70265


namespace slope_intercept_form_correct_l70_70587

theorem slope_intercept_form_correct:
  ∀ (x y : ℝ), (2 * (x - 3) - 1 * (y + 4) = 0) → (∃ m b, y = m * x + b ∧ m = 2 ∧ b = -10) :=
by
  intro x y h
  use 2, -10
  sorry

end slope_intercept_form_correct_l70_70587


namespace taxi_distance_l70_70849

variable (initial_fee charge_per_2_5_mile total_charge : ℝ)
variable (d : ℝ)

theorem taxi_distance 
  (h_initial_fee : initial_fee = 2.35)
  (h_charge_per_2_5_mile : charge_per_2_5_mile = 0.35)
  (h_total_charge : total_charge = 5.50)
  (h_eq : total_charge = initial_fee + (charge_per_2_5_mile / (2/5)) * d) :
  d = 3.6 :=
sorry

end taxi_distance_l70_70849


namespace workers_complete_job_together_in_time_l70_70207

theorem workers_complete_job_together_in_time :
  let work_rate_A := 1 / 10 
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  time = 60 / 13 :=
by
  let work_rate_A := 1 / 10
  let work_rate_B := 1 / 15
  let work_rate_C := 1 / 20
  let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
  let time := 1 / combined_work_rate
  sorry

end workers_complete_job_together_in_time_l70_70207


namespace repeating_decimal_fraction_l70_70231

theorem repeating_decimal_fraction (x : ℚ) (h : x = 7.5656) : x = 749 / 99 :=
by
  sorry

end repeating_decimal_fraction_l70_70231


namespace train_constant_speed_is_48_l70_70859

theorem train_constant_speed_is_48 
  (d_12_00 d_12_15 d_12_45 : ℝ)
  (h1 : 72.5 ≤ d_12_00 ∧ d_12_00 < 73.5)
  (h2 : 61.5 ≤ d_12_15 ∧ d_12_15 < 62.5)
  (h3 : 36.5 ≤ d_12_45 ∧ d_12_45 < 37.5)
  (constant_speed : ℝ → ℝ): 
  (constant_speed d_12_15 - constant_speed d_12_00 = 48) ∧
  (constant_speed d_12_45 - constant_speed d_12_15 = 48) :=
by
  sorry

end train_constant_speed_is_48_l70_70859


namespace math_problem_l70_70467

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l70_70467


namespace angela_january_additional_sleep_l70_70783

-- Definitions corresponding to conditions in part a)
def december_sleep_hours : ℝ := 6.5
def january_sleep_hours : ℝ := 8.5
def days_in_january : ℕ := 31

-- The proof statement, proving the January's additional sleep hours
theorem angela_january_additional_sleep :
  (january_sleep_hours - december_sleep_hours) * days_in_january = 62 :=
by
  -- Since the focus is only on the statement, we skip the actual proof.
  sorry

end angela_january_additional_sleep_l70_70783


namespace bugs_max_contacts_l70_70574

theorem bugs_max_contacts :
  ∃ a b : ℕ, (a + b = 2016) ∧ (a * b = 1008^2) :=
by
  sorry

end bugs_max_contacts_l70_70574


namespace tan_of_angle_123_l70_70852

variable (a : ℝ)
variable (h : Real.sin 123 = a)

theorem tan_of_angle_123 : Real.tan 123 = a / Real.cos 123 :=
by
  sorry

end tan_of_angle_123_l70_70852


namespace fraction_value_l70_70097

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 3) = 9 / 8 := 
by sorry

end fraction_value_l70_70097


namespace optimal_selection_method_use_golden_ratio_l70_70348

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l70_70348


namespace optimal_selection_method_uses_golden_ratio_l70_70305

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l70_70305


namespace combine_like_terms_l70_70565

theorem combine_like_terms (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := 
  sorry

end combine_like_terms_l70_70565


namespace problem1_problem2_l70_70966

open Set Real

-- Given A and B
def A (a : ℝ) : Set ℝ := {x | x > a}
def B : Set ℝ := {y | y > -1}

-- Problem 1: If A = B, then a = -1
theorem problem1 (a : ℝ) (h : A a = B) : a = -1 := by
  sorry

-- Problem 2: If (complement of A) ∩ B ≠ ∅, find the range of a
theorem problem2 (a : ℝ) (h : (compl (A a)) ∩ B ≠ ∅) : a ∈ Ioi (-1) := by
  sorry

end problem1_problem2_l70_70966


namespace gcd_lcm_sum_l70_70754

-- Define the numbers and their prime factorizations
def a := 120
def b := 4620
def a_prime_factors := (2, 3) -- 2^3
def b_prime_factors := (2, 2) -- 2^2

-- Define gcd and lcm based on the problem statement
def gcd_ab := 60
def lcm_ab := 4620

-- The statement to be proved
theorem gcd_lcm_sum : gcd a b + lcm a b = 4680 :=
by sorry

end gcd_lcm_sum_l70_70754


namespace solution_for_4_minus_c_l70_70251

-- Define the conditions as Lean hypotheses
theorem solution_for_4_minus_c (c d : ℚ) (h1 : 4 + c = 5 - d) (h2 : 5 + d = 9 + c) : 4 - c = 11 / 2 :=
by
  sorry

end solution_for_4_minus_c_l70_70251


namespace fraction_zero_l70_70983

theorem fraction_zero (x : ℝ) (h : x ≠ -1) (h₀ : (x^2 - 1) / (x + 1) = 0) : x = 1 :=
by {
  sorry
}

end fraction_zero_l70_70983


namespace inequality_a_cube_less_b_cube_l70_70974

theorem inequality_a_cube_less_b_cube (a b : ℝ) (ha : a < 0) (hb : b > 0) : a^3 < b^3 :=
by
  sorry

end inequality_a_cube_less_b_cube_l70_70974


namespace optimal_selection_method_uses_golden_ratio_l70_70322

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l70_70322


namespace common_chord_properties_l70_70134

noncomputable def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 1 = 0

noncomputable def length_common_chord : ℝ := 2 * Real.sqrt 5

theorem common_chord_properties :
  (∀ x y : ℝ, 
    x^2 + y^2 + 2 * x + 8 * y - 8 = 0 ∧
    x^2 + y^2 - 4 * x - 4 * y - 2 = 0 →
    line_equation x y) ∧ 
  length_common_chord = 2 * Real.sqrt 5 :=
by
  sorry

end common_chord_properties_l70_70134


namespace bags_of_sugar_bought_l70_70701

-- Define the conditions as constants
def cups_at_home : ℕ := 3
def cups_per_bag : ℕ := 6
def cups_per_batter_dozen : ℕ := 1
def cups_per_frosting_dozen : ℕ := 2
def dozens_of_cupcakes : ℕ := 5

-- Prove that the number of bags of sugar Lillian bought is 2
theorem bags_of_sugar_bought : ∃ bags : ℕ, bags = 2 :=
by
  let total_cups_batter := dozens_of_cupcakes * cups_per_batter_dozen
  let total_cups_frosting := dozens_of_cupcakes * cups_per_frosting_dozen
  let total_cups_needed := total_cups_batter + total_cups_frosting
  let cups_to_buy := total_cups_needed - cups_at_home
  let bags := cups_to_buy / cups_per_bag
  have h : bags = 2 := sorry
  exact ⟨bags, h⟩

end bags_of_sugar_bought_l70_70701


namespace expression_value_l70_70398

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end expression_value_l70_70398


namespace max_students_l70_70068

theorem max_students (pens pencils : ℕ) (h1 : pens = 1008) (h2 : pencils = 928) : Nat.gcd pens pencils = 16 :=
by
  sorry

end max_students_l70_70068


namespace optimal_selection_method_uses_golden_ratio_l70_70338

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l70_70338


namespace ceil_floor_difference_l70_70451

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l70_70451


namespace optimal_selection_golden_ratio_l70_70308

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l70_70308


namespace special_op_2_4_5_l70_70700

def special_op (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

theorem special_op_2_4_5 : special_op 2 4 5 = -24 := by
  sorry

end special_op_2_4_5_l70_70700


namespace sum_of_squares_of_roots_l70_70792

-- Define the roots of the polynomial and Vieta's conditions
variables {p q r : ℝ}

-- Given conditions from Vieta's formulas
def vieta_conditions (p q r : ℝ) : Prop :=
  p + q + r = 7 / 3 ∧
  p * q + p * r + q * r = 2 / 3 ∧
  p * q * r = 4 / 3

-- Statement that sum of squares of roots equals to 37/9 given Vieta's conditions
theorem sum_of_squares_of_roots 
  (h : vieta_conditions p q r) : 
  p^2 + q^2 + r^2 = 37 / 9 := 
sorry

end sum_of_squares_of_roots_l70_70792


namespace calculation_result_l70_70889

theorem calculation_result : (4^2)^3 - 4 = 4092 :=
by
  sorry

end calculation_result_l70_70889


namespace mary_keep_warm_hours_l70_70167

-- Definitions based on the conditions
def sticks_from_chairs (chairs : ℕ) : ℕ := chairs * 6
def sticks_from_tables (tables : ℕ) : ℕ := tables * 9
def sticks_from_stools (stools : ℕ) : ℕ := stools * 2
def sticks_needed_per_hour : ℕ := 5

-- Given counts of furniture
def chairs : ℕ := 18
def tables : ℕ := 6
def stools : ℕ := 4

-- Total number of sticks
def total_sticks : ℕ := (sticks_from_chairs chairs) + (sticks_from_tables tables) + (sticks_from_stools stools)

-- Proving the number of hours Mary can keep warm
theorem mary_keep_warm_hours : total_sticks / sticks_needed_per_hour = 34 := by
  sorry

end mary_keep_warm_hours_l70_70167


namespace books_before_purchase_l70_70361

theorem books_before_purchase (x : ℕ) (h : x + 140 = (27 / 25 : ℚ) * x) : x = 1750 :=
sorry

end books_before_purchase_l70_70361


namespace probability_three_cards_l70_70742

theorem probability_three_cards (S : Type) [Fintype S]
  (deck : Finset S) (n : ℕ) (hn : n = 52)
  (hearts : Finset S) (spades : Finset S)
  (tens: Finset S)
  (hhearts_count : ∃ k, hearts.card = k ∧ k = 13)
  (hspades_count : ∃ k, spades.card = k ∧ k = 13)
  (htens_count : ∃ k, tens.card = k ∧ k = 4)
  (hdeck_partition : ∀ x ∈ deck, x ∈ hearts ∨ x ∈ spades ∨ x ∈ tens ∨ (x ∉ hearts ∧ x ∉ spades ∧ x ∉ tens)) :
  (12 / 52 * 13 / 51 * 4 / 50 + 1 / 52 * 13 / 51 * 3 / 50 = 221 / 44200) :=
by {
  sorry
}

end probability_three_cards_l70_70742


namespace contradiction_in_stock_price_l70_70767

noncomputable def stock_price_contradiction : Prop :=
  ∃ (P D : ℝ), (D = 0.20 * P) ∧ (0.10 = (D / P) * 100)

theorem contradiction_in_stock_price : ¬(stock_price_contradiction) := sorry

end contradiction_in_stock_price_l70_70767


namespace twentieth_century_years_as_powers_of_two_diff_l70_70205

theorem twentieth_century_years_as_powers_of_two_diff :
  ∀ (y : ℕ), (1900 ≤ y ∧ y < 2000) →
    ∃ (n k : ℕ), y = 2^n - 2^k ↔ y = 1984 ∨ y = 1920 := 
by
  sorry

end twentieth_century_years_as_powers_of_two_diff_l70_70205


namespace peregrine_falcon_dive_time_l70_70871

/-- Definition of the bald eagle's speed in miles per hour --/
def v_be : ℝ := 100

/-- Definition of the bald eagle's time to dive in seconds --/
def t_be : ℝ := 30

/-- Definition of the peregrine falcon's speed, which is twice the bald eagle's speed --/
def v_pf : ℝ := 2 * v_be

/-- Definition of the conversion factor from miles per hour to miles per second --/
def miles_per_hour_to_miles_per_second : ℝ := 1 / 3600

/-- Calculate the peregrine falcon's time to dive the same distance --/
def t_pf : ℝ := (v_be * miles_per_hour_to_miles_per_second * t_be) / (v_pf * miles_per_hour_to_miles_per_second)

theorem peregrine_falcon_dive_time :
  t_pf = 15 :=
sorry

end peregrine_falcon_dive_time_l70_70871


namespace Malou_first_quiz_score_l70_70703

variable (score1 score2 score3 : ℝ)

theorem Malou_first_quiz_score (h1 : score1 = 90) (h2 : score2 = 92) (h_avg : (score1 + score2 + score3) / 3 = 91) : score3 = 91 := by
  sorry

end Malou_first_quiz_score_l70_70703


namespace latte_price_l70_70933

theorem latte_price
  (almond_croissant_price salami_croissant_price plain_croissant_price focaccia_price total_spent : ℝ)
  (lattes_count : ℕ)
  (H1 : almond_croissant_price = 4.50)
  (H2 : salami_croissant_price = 4.50)
  (H3 : plain_croissant_price = 3.00)
  (H4 : focaccia_price = 4.00)
  (H5 : total_spent = 21.00)
  (H6 : lattes_count = 2) :
  (total_spent - (almond_croissant_price + salami_croissant_price + plain_croissant_price + focaccia_price)) / lattes_count = 2.50 :=
by
  -- skip the proof
  sorry

end latte_price_l70_70933


namespace need_to_work_24_hours_per_week_l70_70832

-- Definitions
def original_hours_per_week := 20
def total_weeks := 12
def target_income := 3000

def missed_weeks := 2
def remaining_weeks := total_weeks - missed_weeks

-- Calculation
def new_hours_per_week := (original_hours_per_week * total_weeks) / remaining_weeks

-- Statement of the theorem
theorem need_to_work_24_hours_per_week : new_hours_per_week = 24 := 
by 
  -- Adding sorry to skip the proof, focusing on the statement.
  sorry

end need_to_work_24_hours_per_week_l70_70832


namespace complement_intersection_l70_70046

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {1, 6} :=
by
  sorry

end complement_intersection_l70_70046


namespace solve_equation_l70_70296

theorem solve_equation (x : ℝ) (h : x ≠ 4) :
  (x - 3) / (4 - x) - 1 = 1 / (x - 4) → x = 3 :=
by
  sorry

end solve_equation_l70_70296


namespace packs_sold_by_Robyn_l70_70293

theorem packs_sold_by_Robyn (total_packs : ℕ) (lucy_packs : ℕ) (robyn_packs : ℕ) 
  (h1 : total_packs = 98) (h2 : lucy_packs = 43) (h3 : robyn_packs = total_packs - lucy_packs) :
  robyn_packs = 55 :=
by
  rw [h1, h2] at h3
  exact h3

end packs_sold_by_Robyn_l70_70293


namespace points_per_question_l70_70403

theorem points_per_question (first_half_correct : ℕ) (second_half_correct : ℕ) (final_score : ℕ) :
  first_half_correct = 5 → 
  second_half_correct = 5 → 
  final_score = 50 → 
  (final_score / (first_half_correct + second_half_correct) = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have h_total : first_half_correct + second_half_correct = 10 := by rw [h1, h2]
  rw h_total
  exact Nat.div_eq_of_eq_mul_right (by norm_num) h3

end points_per_question_l70_70403


namespace num_non_squares_cubes_1_to_200_l70_70674

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l70_70674


namespace part1_l70_70841

variable (A B C : ℝ)
variable (a b c S : ℝ)
variable (h1 : a * (1 + Real.cos C) + c * (1 + Real.cos A) = (5 / 2) * b)
variable (h2 : a * Real.cos C + c * Real.cos A = b)

theorem part1 : 2 * (a + c) = 3 * b := 
sorry

end part1_l70_70841


namespace sum_of_digits_is_8_l70_70012

theorem sum_of_digits_is_8 (d : ℤ) (h1 : d ≥ 0)
  (h2 : 8 * d / 5 - 80 = d) : (d / 100) + ((d % 100) / 10) + (d % 10) = 8 :=
by
  sorry

end sum_of_digits_is_8_l70_70012


namespace abs_triangle_inequality_l70_70053

theorem abs_triangle_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by sorry

end abs_triangle_inequality_l70_70053


namespace area_of_right_triangle_l70_70151

theorem area_of_right_triangle (a b c : ℝ) 
  (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 30 :=
by sorry

end area_of_right_triangle_l70_70151


namespace expression_evaluation_l70_70796

theorem expression_evaluation : 
  (50 - (2210 - 251)) + (2210 - (251 - 50)) = 100 := 
  by sorry

end expression_evaluation_l70_70796


namespace youseff_blocks_l70_70070

-- Definition of the conditions
def time_to_walk (x : ℕ) : ℕ := x
def time_to_ride (x : ℕ) : ℕ := (20 * x) / 60
def extra_time (x : ℕ) : ℕ := time_to_walk x - time_to_ride x

-- Statement of the problem in Lean
theorem youseff_blocks : ∃ x : ℕ, extra_time x = 6 ∧ x = 9 :=
by {
  sorry
}

end youseff_blocks_l70_70070


namespace equivalence_of_negation_l70_70033

-- Define the statement for the negation
def negation_stmt := ¬ ∃ x0 : ℝ, x0 ≤ 0 ∧ x0^2 ≥ 0

-- Define the equivalent statement after negation
def equivalent_stmt := ∀ x : ℝ, x ≤ 0 → x^2 < 0

-- The theorem stating that the negation_stmt is equivalent to equivalent_stmt
theorem equivalence_of_negation : negation_stmt ↔ equivalent_stmt := 
sorry

end equivalence_of_negation_l70_70033


namespace race_length_l70_70598

theorem race_length
  (B_s : ℕ := 50) -- Biff's speed in yards per minute
  (K_s : ℕ := 51) -- Kenneth's speed in yards per minute
  (D_above_finish : ℕ := 10) -- distance Kenneth is past the finish line when Biff finishes
  : {L : ℕ // L = 500} := -- the length of the race is 500 yards.
  sorry

end race_length_l70_70598


namespace statement_A_statement_B_statement_D_l70_70566

theorem statement_A (x : ℝ) (hx : x > 1) : 
  ∃(y : ℝ), y = 3 * x + 1 / (x - 1) ∧ y = 2 * Real.sqrt 3 + 3 := 
  sorry

theorem statement_B (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃(z : ℝ), z = 1 / (x + 1) + 2 / y ∧ z = 9 / 2 := 
  sorry

theorem statement_D (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃(k : ℝ), k = (x^2 + y^2 + z^2) / (3 * x * y + 4 * y * z) ∧ k = 2 / 5 := 
  sorry

end statement_A_statement_B_statement_D_l70_70566


namespace minimum_number_of_girls_l70_70365

theorem minimum_number_of_girls (students boys girls : ℕ) 
(h1 : students = 20) (h2 : boys = students - girls)
(h3 : ∀ d, 0 ≤ d → 2 * (d + 1) ≥ boys) 
: 6 ≤ girls :=
by 
  have h4 : 20 - girls = boys, from h2.symm
  have h5 : 2 * (girls + 1) ≥ boys, from h3 girls (nat.zero_le girls)
  linarith

#check minimum_number_of_girls

end minimum_number_of_girls_l70_70365


namespace Reema_loan_problem_l70_70291

-- Define problem parameters
def Principal : ℝ := 150000
def Interest : ℝ := 42000
def ProfitRate : ℝ := 0.1
def Profit : ℝ := 25000

-- State the problem as a Lean 4 theorem
theorem Reema_loan_problem (R : ℝ) (Investment : ℝ) : 
  Principal * (R / 100) * R = Interest ∧ 
  Profit = Investment * ProfitRate * R ∧ 
  R = 5 ∧ 
  Investment = 50000 :=
by
  sorry

end Reema_loan_problem_l70_70291


namespace angle_of_inclination_of_line_l70_70489

theorem angle_of_inclination_of_line (θ : ℝ) (m : ℝ) (h : |m| = 1) :
  θ = 45 ∨ θ = 135 :=
sorry

end angle_of_inclination_of_line_l70_70489


namespace longest_side_of_triangle_l70_70499

theorem longest_side_of_triangle :
  ∀ (A B C a b : ℝ),
    B = 2 * π / 3 →
    C = π / 6 →
    a = 5 →
    A = π - B - C →
    (b / (Real.sin B) = a / (Real.sin A)) →
    b = 5 * Real.sqrt 3 :=
by
  intros A B C a b hB hC ha hA h_sine_ratio
  sorry

end longest_side_of_triangle_l70_70499


namespace surface_area_LShape_l70_70179

-- Define the structures and conditions
structure UnitCube where
  x : ℕ
  y : ℕ
  z : ℕ

def LShape (cubes : List UnitCube) : Prop :=
  -- Condition 1: Exactly 7 unit cubes
  cubes.length = 7 ∧
  -- Condition 2: 4 cubes in a line along x-axis (bottom row)
  ∃ a b c d : UnitCube, 
    (a.x + 1 = b.x ∧ b.x + 1 = c.x ∧ c.x + 1 = d.x ∧
     a.y = b.y ∧ b.y = c.y ∧ c.y = d.y ∧
     a.z = b.z ∧ b.z = c.z ∧ c.z = d.z) ∧
  -- Condition 3: 3 cubes stacked along z-axis at one end of the row
  ∃ e f g : UnitCube,
    (d.x = e.x ∧ e.x = f.x ∧ f.x = g.x ∧
     d.y = e.y ∧ e.y = f.y ∧ f.y = g.y ∧
     e.z + 1 = f.z ∧ f.z + 1 = g.z)

-- Define the surface area function
def surfaceArea (cubes : List UnitCube) : ℕ :=
  4*7 - 2*3 + 4 -- correct answer calculation according to manual analysis of exposed faces

-- The theorem to be proven
theorem surface_area_LShape : 
  ∀ (cubes : List UnitCube), LShape cubes → surfaceArea cubes = 26 :=
by sorry

end surface_area_LShape_l70_70179


namespace linear_function_no_third_quadrant_l70_70147

theorem linear_function_no_third_quadrant (k b : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ k * x + b) : k < 0 ∧ 0 ≤ b :=
sorry

end linear_function_no_third_quadrant_l70_70147


namespace number_of_students_scoring_above_90_l70_70686

theorem number_of_students_scoring_above_90
  (total_students : ℕ)
  (mean : ℝ)
  (variance : ℝ)
  (students_scoring_at_least_60 : ℕ)
  (h1 : total_students = 1200)
  (h2 : mean = 75)
  (h3 : ∃ (σ : ℝ), variance = σ^2)
  (h4 : students_scoring_at_least_60 = 960)
  : ∃ n, n = total_students - students_scoring_at_least_60 ∧ n = 240 :=
by {
  sorry
}

end number_of_students_scoring_above_90_l70_70686


namespace probability_sqrt_2_add_sqrt_5_le_abs_v_add_w_zero_l70_70853

noncomputable def root_of_unity (n k : ℕ) : ℂ := complex.exp (2 * real.pi * complex.I * k / n)

def is_root_of_unity (n : ℕ) (z : ℂ) : Prop := z ^ n = 1

def distinct_roots_of_equation (n : ℕ) : set ℂ := {z | is_root_of_unity n z}

theorem probability_sqrt_2_add_sqrt_5_le_abs_v_add_w_zero:
  ∀ (n : ℕ) (hn : 1 < n),
  let roots := (distinct_roots_of_equation n) in
  ∀ (v w : ℂ) (hv : v ∈ roots) (hw : w ∈ roots) (hvw : v ≠ w),
  real.sqrt (2 + real.sqrt 5) ≤ complex.abs (v + w) → false :=
begin
  sorry
end

end probability_sqrt_2_add_sqrt_5_le_abs_v_add_w_zero_l70_70853


namespace problem_part1_problem_part2_l70_70126

theorem problem_part1 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 * y - x * y^2 = 4 * Real.sqrt 2 := 
  sorry

theorem problem_part2 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 - x * y + y^2 = 33 := 
  sorry

end problem_part1_problem_part2_l70_70126


namespace mark_egg_supply_in_a_week_l70_70283

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end mark_egg_supply_in_a_week_l70_70283


namespace tree_growth_period_l70_70755

theorem tree_growth_period (initial height growth_rate : ℕ) (H4 final_height years : ℕ) 
  (h_init : initial_height = 4) 
  (h_growth_rate : growth_rate = 1) 
  (h_H4 : H4 = initial_height + 4 * growth_rate)
  (h_final_height : final_height = H4 + H4 / 4) 
  (h_years : years = (final_height - initial_height) / growth_rate) :
  years = 6 :=
by
  sorry

end tree_growth_period_l70_70755


namespace probability_single_trial_l70_70736

-- Conditions
def prob_at_least_once (p : ℝ) : ℝ := 1 - (1 - p)^3

-- Main theorem
theorem probability_single_trial (h : prob_at_least_once p = 0.973) : p = 0.7 :=
by
  sorry

end probability_single_trial_l70_70736


namespace find_A_l70_70430

noncomputable def telephone_number_satisfies_conditions (A B C D E F G H I J : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J ∧
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  E = D - 2 ∧ F = D - 4 ∧ -- Given D, E, F are consecutive even digits
  H = G - 2 ∧ I = G - 4 ∧ J = G - 6 ∧ -- Given G, H, I, J are consecutive odd digits
  A + B + C = 9

theorem find_A :
  ∃ (A B C D E F G H I J : ℕ), telephone_number_satisfies_conditions A B C D E F G H I J ∧ A = 8 :=
by {
  sorry
}

end find_A_l70_70430


namespace M_gt_N_l70_70137

variable (a b : ℝ)

def M := 10 * a^2 + 2 * b^2 - 7 * a + 6
def N := a^2 + 2 * b^2 + 5 * a + 1

theorem M_gt_N : M a b > N a b := by
  sorry

end M_gt_N_l70_70137


namespace minimum_number_of_girls_l70_70379

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end minimum_number_of_girls_l70_70379


namespace find_y_l70_70254

theorem find_y (y : ℕ) 
  (h : (1/8) * 2^36 = 8^y) : y = 11 :=
sorry

end find_y_l70_70254


namespace julia_investment_l70_70159

-- Define the total investment and the relationship between the investments
theorem julia_investment:
  ∀ (m : ℕ), 
  m + 6 * m = 200000 → 6 * m = 171428 := 
by
  sorry

end julia_investment_l70_70159


namespace sandy_potatoes_l70_70707

theorem sandy_potatoes (n_total n_nancy n_sandy : ℕ) 
  (h_total : n_total = 13) 
  (h_nancy : n_nancy = 6) 
  (h_sum : n_total = n_nancy + n_sandy) : 
  n_sandy = 7 :=
by
  sorry

end sandy_potatoes_l70_70707


namespace total_feathers_needed_l70_70480

theorem total_feathers_needed 
  (animals_group1 : ℕ) (feathers_group1 : ℕ)
  (animals_group2 : ℕ) (feathers_group2 : ℕ) 
  (total_feathers : ℕ) :
  animals_group1 = 934 →
  feathers_group1 = 7 →
  animals_group2 = 425 →
  feathers_group2 = 12 →
  total_feathers = 11638 :=
by sorry

end total_feathers_needed_l70_70480


namespace matilda_smartphone_loss_percentage_l70_70285

theorem matilda_smartphone_loss_percentage :
  ∀ (initial_cost selling_price : ℝ),
  initial_cost = 300 →
  selling_price = 255 →
  let loss := initial_cost - selling_price in
  let percentage_loss := (loss / initial_cost) * 100 in
  percentage_loss = 15 :=
by
  intros initial_cost selling_price h₁ h₂
  let loss := initial_cost - selling_price
  let percentage_loss := (loss / initial_cost) * 100
  rw [h₁, h₂]
  sorry

end matilda_smartphone_loss_percentage_l70_70285


namespace problem1_problem2_l70_70575

theorem problem1 : -1 + (-6) - (-4) + 0 = -3 := by
  sorry

theorem problem2 : 24 * (-1 / 4) / (-3 / 2) = 4 := by
  sorry

end problem1_problem2_l70_70575


namespace a_eq_zero_l70_70028

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ x : ℤ, x^2 = 2^n * a + b) : a = 0 :=
sorry

end a_eq_zero_l70_70028


namespace range_of_a_l70_70123

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

theorem range_of_a (h : (∀ x a, p x a → q x) ∧ (∃ x a, q x ∧ ¬ p x a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry

end range_of_a_l70_70123


namespace probability_of_sum_19_l70_70390

def count_card_deck := 52
def count_nine_cards := 4
def count_ten_cards := 4
def total_number_cards := count_nine_cards + count_ten_cards
def first_card_probability := total_number_cards / count_card_deck
def second_card_probability := count_nine_cards / (count_card_deck - 1)

theorem probability_of_sum_19 : first_card_probability * second_card_probability = 8 / 663 := by
  /- We know total_number_cards is 8 which is sum of 4 nines and 4 tens -/
  have total_cards : total_number_cards = 4 + 4 := by norm_num
  /- Probability calculation -/
  rw [first_card_probability, second_card_probability],
  /- Substitute probabilities in terms of rational numbers -/
  change (8 / 52) * (4 / 51) = 8 / 663,
  /- Simplify the left hand side -/
  norm_num,
  sorry

end probability_of_sum_19_l70_70390


namespace xy_divides_x2_plus_2y_minus_1_l70_70106

theorem xy_divides_x2_plus_2y_minus_1 (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * y ∣ x^2 + 2 * y - 1) ↔ (∃ t : ℕ, t > 0 ∧ ((x = 1 ∧ y = t) ∨ (x = 2 * t - 1 ∧ y = t)
  ∨ (x = 3 ∧ y = 8) ∨ (x = 5 ∧ y = 8))) :=
by
  sorry

end xy_divides_x2_plus_2y_minus_1_l70_70106


namespace value_of_f_f_2_l70_70000

def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x - 1

theorem value_of_f_f_2 : f (f 2) = 164 := by
  sorry

end value_of_f_f_2_l70_70000


namespace minimum_girls_in_class_l70_70372

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l70_70372


namespace min_value_exp_l70_70815

theorem min_value_exp (a b : ℝ) (h_condition : a - 3 * b + 6 = 0) : 
  ∃ (m : ℝ), m = 2^a + 1 / 8^b ∧ m ≥ (1 / 4) :=
by
  sorry

end min_value_exp_l70_70815


namespace unique_root_in_interval_l70_70824

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 2

theorem unique_root_in_interval (n : ℤ) (h_root : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0) :
  n = 1 := 
sorry

end unique_root_in_interval_l70_70824


namespace find_f_13_l70_70875

noncomputable def f : ℕ → ℕ :=
  sorry

axiom condition1 (x : ℕ) : f (x + f x) = 3 * f x
axiom condition2 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
  sorry

end find_f_13_l70_70875


namespace complementary_angle_decrease_ratio_l70_70878

theorem complementary_angle_decrease_ratio :
  ∀ (x y : ℝ), (x + y = 90) ∧ (x / y = 2 / 3) → (x * 1.2 + (y - y * (0.1333)) = 90) :=
by
  intros x y h
  cases h with h_sum h_ratio
  -- sorry is a placeholder for the actual proof
  sorry

end complementary_angle_decrease_ratio_l70_70878


namespace domain_of_f_l70_70026

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 1 - x > 0
def condition2 (x : ℝ) : Prop := 3 * x + 1 > 0

-- Define the domain interval
def domain (x : ℝ) : Prop := -1 / 3 < x ∧ x < 1

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (Real.sqrt (1 - x)) + Real.log (3 * x + 1)

-- The main theorem to prove
theorem domain_of_f : 
  (∀ x : ℝ, condition1 x ∧ condition2 x ↔ domain x) :=
by {
  sorry
}

end domain_of_f_l70_70026


namespace repeating_decimal_product_l70_70059

theorem repeating_decimal_product 
  (x : ℚ) 
  (h1 : x = (0.0126 : ℚ)) 
  (h2 : 9999 * x = 126) 
  (h3 : x = 14 / 1111) : 
  14 * 1111 = 15554 := 
by
  sorry

end repeating_decimal_product_l70_70059


namespace seq_properties_l70_70485

-- Conditions for the sequence a_n
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * a n + 1

-- The statements to prove given the sequence definition
theorem seq_properties (a : ℕ → ℝ) (h : seq a) :
  (∀ n, a (n + 1) ≥ 2 * a n) ∧
  (∀ n, a (n + 1) / a n ≥ a n) ∧
  (∀ n, a n ≥ n * n - 2 * n + 2) :=
by
  sorry

end seq_properties_l70_70485


namespace thabo_books_220_l70_70022

def thabo_books_total (H PNF PF Total : ℕ) : Prop :=
  (H = 40) ∧
  (PNF = H + 20) ∧
  (PF = 2 * PNF) ∧
  (Total = H + PNF + PF)

theorem thabo_books_220 : ∃ H PNF PF Total : ℕ, thabo_books_total H PNF PF 220 :=
by {
  sorry
}

end thabo_books_220_l70_70022


namespace exponential_equality_l70_70016

open Complex

/--
Prove that \( e^{\pi i} = -1 \) and \( e^{2 \pi i} = 1 \)
using Euler's formula for complex exponentials, \( e^{ix} = \cos(x) + i \sin(x) \)
-/
theorem exponential_equality :
  exp (π * I) = -1 ∧ exp (2 * π * I) = 1 :=
by
  -- Use Euler's formula to verify the two statements
  sorry

end exponential_equality_l70_70016


namespace ab_a4_b4_divisible_by_30_l70_70721

theorem ab_a4_b4_divisible_by_30 (a b : Int) : 30 ∣ a * b * (a^4 - b^4) := 
by
  sorry

end ab_a4_b4_divisible_by_30_l70_70721


namespace lassis_from_mangoes_l70_70609

theorem lassis_from_mangoes (m l m' : ℕ) (h : m' = 18) (hlm : l / m = 8 / 3) : l / m' = 48 / 18 :=
by
  sorry

end lassis_from_mangoes_l70_70609


namespace distance_between_B_and_D_l70_70749

theorem distance_between_B_and_D (a b c d : ℝ) (h1 : |2 * a - 3 * c| = 1) (h2 : |2 * b - 3 * c| = 1) (h3 : |(2/3) * (d - a)| = 1) (h4 : a ≠ b) :
  |d - b| = 0.5 ∨ |d - b| = 2.5 :=
by
  sorry

end distance_between_B_and_D_l70_70749


namespace window_ratio_area_l70_70417

/-- Given a rectangle with semicircles at either end, if the ratio of AD to AB is 3:2,
    and AB is 30 inches, then the ratio of the area of the rectangle to the combined 
    area of the semicircles is 6 : π. -/
theorem window_ratio_area (AD AB r : ℝ) (h1 : AB = 30) (h2 : AD / AB = 3 / 2) (h3 : r = AB / 2) :
    (AD * AB) / (π * r^2) = 6 / π :=
by
  sorry

end window_ratio_area_l70_70417


namespace part1_part2_l70_70267

open Real

noncomputable def a_value := 2 * sqrt 2

noncomputable def line_cartesian_eqn (x y : ℝ) : Prop :=
  x + y - 4 = 0

noncomputable def point_on_line (ρ θ : ℝ) :=
  ρ * cos (θ - π / 4) = a_value

noncomputable def curve_param_eqns (θ : ℝ) : (ℝ × ℝ) :=
  (sqrt 3 * cos θ, sin θ)

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 4) / sqrt 2

theorem part1 (P : ℝ × ℝ) (ρ θ : ℝ) : 
  P = (4, π / 2) ∧ point_on_line ρ θ → 
  a_value = 2 * sqrt 2 ∧ line_cartesian_eqn 4 (4 * tan (π / 4)) :=
sorry

theorem part2 :
  (∀ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) ≤ 3 * sqrt 2) ∧
  (∃ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) = 3 * sqrt 2) :=
sorry

end part1_part2_l70_70267


namespace max_intersection_points_l70_70005

-- Definitions based on given conditions
def L : Type := ℕ -- Type to represent lines
def B : Type := ℕ -- Type to represent the point B

/-- Conditions -/
def condition_parallel (n : ℕ) : Prop :=
  n % 5 = 0

def condition_through_B (n : ℕ) : Prop :=
  (n - 4) % 5 = 0

-- Total number of lines
def number_of_lines : ℕ := 120

-- Set of all lines
def lines_set : Finset L := Finset.range (number_of_lines + 1)

-- Count of lines in each set
def count_P : ℕ := Finset.card (lines_set.filter condition_parallel)
def count_Q : ℕ := Finset.card (lines_set.filter condition_through_B)
def count_R : ℕ := number_of_lines - count_P - count_Q

-- Maximum number of intersection points
theorem max_intersection_points :
  1 + count_P * count_R + (count_R * (count_R - 1)) / 2 + count_P * count_Q + (count_Q * count_R)
  = 6589 :=
by {
  -- count_P = 24, count_Q = 24, count_R = 72
  have hP : count_P = 24 := sorry,
  have hQ : count_Q = 24 := sorry,
  have hR : count_R = 72 := sorry,
  rw [hP, hQ, hR],
  simp,
  norm_num
}

end max_intersection_points_l70_70005


namespace restroom_students_l70_70442

theorem restroom_students (R : ℕ) (h1 : 4 * 6 = 24) (h2 : (2/3 : ℚ) * 24 = 16)
  (h3 : 23 = 16 + (3 * R - 1) + R) : R = 2 :=
by
  sorry

end restroom_students_l70_70442


namespace angle_A_value_sin_BC_value_l70_70264

open Real

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ 
  A + B + C = π 

theorem angle_A_value (A B C : ℝ) (h : triangleABC a b c A B C) (h1 : cos 2 * A - 3 * cos (B + C) = 1) : 
  A = π / 3 :=
sorry

theorem sin_BC_value (A B C S b c : ℝ) (h : triangleABC a b c A B C)
  (hA : A = π / 3) (hS : S = 5 * sqrt 3) (hb : b = 5) : 
  sin B * sin C = 5 / 7 :=
sorry

end angle_A_value_sin_BC_value_l70_70264


namespace ceiling_and_floor_calculation_l70_70457

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l70_70457


namespace bounded_functions_sup_eq_max_l70_70290

noncomputable def g1 (s : ℝ) : ℝ := 2
noncomputable def g2 (s : ℝ) : ℝ := 1
noncomputable def h1 (t : ℝ) : ℝ := (Real.log 2) * 2^t
noncomputable def h2 (t : ℝ) : ℝ := (1 - t * Real.log 2) * 2^t

theorem bounded_functions_sup_eq_max :
  ∀ (x : ℝ), 
    (∀ s, 1 ≤ g1 s) → 
    (∀ s, 1 ≤ g2 s) → 
    (∀ s, g1 s < ∞) → 
    (∀ s, g2 s < ∞) → 
    sup (λ s, (g1 s)^x * (g2 s)) = max (λ t, x * (h1 t) + h2 t) :=
by {
  intros x h1_bound h2_bound _ _,
  -- Proof omitted
  sorry
}

end bounded_functions_sup_eq_max_l70_70290


namespace count_not_squares_or_cubes_200_l70_70671

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l70_70671


namespace maximum_N_value_l70_70510

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l70_70510


namespace mock_exam_girls_count_l70_70071

theorem mock_exam_girls_count
  (B G Bc Gc : ℕ)
  (h1: B + G = 400)
  (h2: Bc = 60 * B / 100)
  (h3: Gc = 80 * G / 100)
  (h4: Bc + Gc = 65 * 400 / 100)
  : G = 100 :=
sorry

end mock_exam_girls_count_l70_70071


namespace hua_luogeng_optimal_selection_method_l70_70332

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l70_70332


namespace optimal_selection_method_uses_golden_ratio_l70_70310

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l70_70310


namespace residue_11_pow_2021_mod_19_l70_70892

theorem residue_11_pow_2021_mod_19 : (11^2021) % 19 = 17 := 
by
  -- this is to ensure the theorem is syntactically correct in Lean but skips the proof for now
  sorry

end residue_11_pow_2021_mod_19_l70_70892


namespace platform_length_calc_l70_70770

noncomputable def length_of_platform (V : ℝ) (T : ℝ) (L_train : ℝ) : ℝ :=
  (V * 1000 / 3600) * T - L_train

theorem platform_length_calc (speed : ℝ) (time : ℝ) (length_train : ℝ):
  speed = 72 →
  time = 26 →
  length_train = 280.0416 →
  length_of_platform speed time length_train = 239.9584 := by
  intros
  unfold length_of_platform
  sorry

end platform_length_calc_l70_70770


namespace variance_of_data_set_l70_70143

noncomputable def variance (s : Finset ℝ) : ℝ :=
  let μ := (∑ x in s, x) / s.card
  (∑ x in s, (x - μ)^2) / s.card

theorem variance_of_data_set :
  ∀ (x y : ℝ), 
  (4 + x + 5 + y + 7 + 9) / 6 = 6 →
  (x = 5 ∨ y = 5) →
  (x + y = 11) →
  variance {4, x, 5, y, 7, 9} = 8 / 3 :=
by 
  intros x y avg mode_sum xy_sum; 
  sorry

end variance_of_data_set_l70_70143


namespace price_before_tax_l70_70136

theorem price_before_tax (P : ℝ) (h : 1.15 * P = 1955) : P = 1700 :=
by sorry

end price_before_tax_l70_70136


namespace find_n_l70_70690

variable {a_n : ℕ → ℤ}
variable (a2 : ℤ) (an : ℤ) (d : ℤ) (n : ℕ)

def arithmetic_sequence (a2 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a2 + (n - 2) * d

theorem find_n 
  (h1 : a2 = 12)
  (h2 : an = -20)
  (h3 : d = -2)
  : n = 18 := by
  sorry

end find_n_l70_70690


namespace inverse_function_l70_70800

variable (x : ℝ)

def f (x : ℝ) : ℝ := (x^(1 / 3)) + 1
def g (x : ℝ) : ℝ := (x - 1)^3

theorem inverse_function :
  ∀ x, f (g x) = x ∧ g (f x) = x :=
by
  -- Proof goes here
  sorry

end inverse_function_l70_70800


namespace money_distribution_l70_70221

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 310) (h3 : C = 10) : A + B + C = 500 :=
by
  sorry

end money_distribution_l70_70221


namespace count_valid_numbers_between_1_and_200_l70_70647

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l70_70647


namespace excess_calories_l70_70847

theorem excess_calories 
  (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ)
  (run_minutes : ℕ) (calories_burned_per_minute : ℕ)
  (h1 : bags = 3) 
  (h2 : ounces_per_bag = 2) 
  (h3 : calories_per_ounce = 150)
  (h4 : run_minutes = 40)
  (h5 : calories_burned_per_minute = 12)
  : (3 * (2 * 150)) - (40 * 12) = 420 := 
by
  -- Introducing hypotheses for clarity
  let total_calories_consumed := bags * (ounces_per_bag * calories_per_ounce)
  let total_calories_burned := run_minutes * calories_burned_per_minute
  
  -- Applying the hypotheses
  have h_total_consumed : total_calories_consumed = 3 * (2 * 150), from by
    rw [h1, h2, h3]

  have h_total_burned : total_calories_burned = 40 * 12, from by
    rw [h4, h5]

  -- Concluding the proof using the hypotheses
  calc
    (3 * (2 * 150)) - (40 * 12) = 900 - 480 : by rw [h_total_consumed, h_total_burned]
    ... = 420 : by norm_num

end excess_calories_l70_70847


namespace cost_of_fencing_per_meter_l70_70031

theorem cost_of_fencing_per_meter
  (length : ℕ) (breadth : ℕ) (total_cost : ℝ) (cost_per_meter : ℝ)
  (h1 : length = 64) 
  (h2 : length = breadth + 28)
  (h3 : total_cost = 5300)
  (h4 : cost_per_meter = total_cost / (2 * (length + breadth))) :
  cost_per_meter = 26.50 :=
by {
  sorry
}

end cost_of_fencing_per_meter_l70_70031


namespace hua_luogeng_optimal_selection_method_l70_70333

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l70_70333


namespace x_minus_y_possible_values_l70_70257

theorem x_minus_y_possible_values (x y : ℝ) (hx : x^2 = 9) (hy : |y| = 4) (hxy : x < y) : x - y = -1 ∨ x - y = -7 := 
sorry

end x_minus_y_possible_values_l70_70257


namespace find_y_l70_70252

theorem find_y (y : ℕ) : (1 / 8) * 2^36 = 8^y → y = 11 := by
  sorry

end find_y_l70_70252


namespace imaginary_part_of_z_l70_70256

theorem imaginary_part_of_z (z : ℂ) (h : (z / (1 - I)) = (3 + I)) : z.im = -2 :=
sorry

end imaginary_part_of_z_l70_70256


namespace first_player_winning_strategy_l70_70048

def game_strategy (S : ℕ) : Prop :=
  ∃ k, (1 ≤ k ∧ k ≤ 5 ∧ (S - k) % 6 = 1)

theorem first_player_winning_strategy : game_strategy 100 :=
sorry

end first_player_winning_strategy_l70_70048


namespace find_middle_and_oldest_sons_l70_70085

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end find_middle_and_oldest_sons_l70_70085


namespace probability_correct_l70_70985

-- Define the total number of bulbs, good quality bulbs, and inferior quality bulbs
def total_bulbs : ℕ := 6
def good_bulbs : ℕ := 4
def inferior_bulbs : ℕ := 2

-- Define the probability of drawing one good bulb and one inferior bulb with replacement
def probability_one_good_one_inferior : ℚ := (good_bulbs * inferior_bulbs * 2) / (total_bulbs ^ 2)

-- Theorem stating that the probability of drawing one good bulb and one inferior bulb is 4/9
theorem probability_correct : probability_one_good_one_inferior = 4 / 9 := 
by
  -- Proof is skipped here
  sorry

end probability_correct_l70_70985


namespace quadratic_decreases_after_vertex_l70_70178

theorem quadratic_decreases_after_vertex :
  ∀ x : ℝ, (x > 2) → (y = -(x - 2)^2 + 3) → ∃ k : ℝ, k < 0 :=
by
  sorry

end quadratic_decreases_after_vertex_l70_70178


namespace no_such_continuous_function_exists_l70_70176

theorem no_such_continuous_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (Continuous f) ∧ ∀ x : ℝ, ((∃ q : ℚ, f x = q) ↔ ∀ q' : ℚ, f (x + 1) ≠ q') :=
sorry

end no_such_continuous_function_exists_l70_70176


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l70_70320

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l70_70320


namespace trivia_game_points_per_question_l70_70404

theorem trivia_game_points_per_question (correct_first_half correct_second_half total_score points_per_question : ℕ) 
  (h1 : correct_first_half = 5) 
  (h2 : correct_second_half = 5) 
  (h3 : total_score = 50) 
  (h4 : correct_first_half + correct_second_half = 10) : 
  points_per_question = 5 :=
by 
  sorry

end trivia_game_points_per_question_l70_70404


namespace length_of_BC_is_eight_l70_70091

theorem length_of_BC_is_eight (a : ℝ) (h_area : (1 / 2) * (2 * a) * a^2 = 64) : 2 * a = 8 := 
by { sorry }

end length_of_BC_is_eight_l70_70091


namespace y_is_75_percent_of_x_l70_70976

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := 0.45 * z = 0.72 * y
def condition2 : Prop := z = 1.20 * x

-- Theorem to prove y = 0.75 * x
theorem y_is_75_percent_of_x (h1 : condition1 z y) (h2 : condition2 x z) : y = 0.75 * x :=
by sorry

end y_is_75_percent_of_x_l70_70976


namespace sugar_needed_for_third_layer_l70_70602

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end sugar_needed_for_third_layer_l70_70602


namespace ceil_floor_diff_l70_70473

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l70_70473


namespace time_in_still_water_l70_70219

-- Define the conditions
variable (S x y : ℝ)
axiom condition1 : S / (x + y) = 6
axiom condition2 : S / (x - y) = 8

-- Define the proof statement
theorem time_in_still_water : S / x = 48 / 7 :=
by
  -- The proof is omitted
  sorry

end time_in_still_water_l70_70219


namespace violet_has_27_nails_l70_70888

def nails_tickletoe : ℕ := 12  -- T
def nails_violet : ℕ := 2 * nails_tickletoe + 3

theorem violet_has_27_nails (h : nails_tickletoe + nails_violet = 39) : nails_violet = 27 :=
by
  sorry

end violet_has_27_nails_l70_70888


namespace correct_result_without_mistake_l70_70758

variable {R : Type*} [CommRing R] (a b c : R)
variable (A : R)

theorem correct_result_without_mistake :
  A + 2 * (ab + 2 * bc - 4 * ac) = (3 * ab - 2 * ac + 5 * bc) → 
  A - 2 * (ab + 2 * bc - 4 * ac) = -ab + 14 * ac - 3 * bc :=
by
  sorry

end correct_result_without_mistake_l70_70758


namespace sum_of_minimums_is_zero_l70_70860

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

-- Conditions: P(Q(x)) has zeros at -5, -3, -1, 1
lemma zeroes_PQ : 
  P.eval (Q.eval (-5)) = 0 ∧ 
  P.eval (Q.eval (-3)) = 0 ∧ 
  P.eval (Q.eval (-1)) = 0 ∧ 
  P.eval (Q.eval (1)) = 0 := 
  sorry

-- Conditions: Q(P(x)) has zeros at -7, -5, -1, 3
lemma zeroes_QP : 
  Q.eval (P.eval (-7)) = 0 ∧ 
  Q.eval (P.eval (-5)) = 0 ∧ 
  Q.eval (P.eval (-1)) = 0 ∧ 
  Q.eval (P.eval (3)) = 0 := 
  sorry

-- Definition to find the minimum value of a polynomial
noncomputable def min_value (P : Polynomial ℝ) : ℝ := sorry

-- Main theorem
theorem sum_of_minimums_is_zero :
  min_value P + min_value Q = 0 := 
  sorry

end sum_of_minimums_is_zero_l70_70860


namespace minimum_girls_in_class_l70_70373

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l70_70373


namespace percentage_deposit_paid_l70_70582

theorem percentage_deposit_paid (D R T : ℝ) (hd : D = 105) (hr : R = 945) (ht : T = D + R) : (D / T) * 100 = 10 := by
  sorry

end percentage_deposit_paid_l70_70582


namespace trajectory_midpoint_eq_C2_length_CD_l70_70819

theorem trajectory_midpoint_eq_C2 {x y x' y' : ℝ} :
  (x' - 0)^2 + (y' - 4)^2 = 16 →
  x = (x' + 4) / 2 →
  y = y' / 2 →
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  sorry

theorem length_CD {x y Cx Cy Dx Dy : ℝ} :
  ((x - 2)^2 + (y - 2)^2 = 4) →
  (x^2 + (y - 4)^2 = 16) →
  ((Cx - Dx)^2 + (Cy - Dy)^2 = 14) :=
by
  sorry

end trajectory_midpoint_eq_C2_length_CD_l70_70819


namespace remainder_division_by_8_is_6_l70_70708

theorem remainder_division_by_8_is_6 (N Q2 R1 : ℤ) (h1 : N = 64 + R1) (h2 : N % 5 = 4) : R1 = 6 :=
by
  sorry

end remainder_division_by_8_is_6_l70_70708


namespace maximum_value_l70_70734

theorem maximum_value (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a^2 * (b + c - a) = b^2 * (a + c - b) ∧ b^2 * (a + c - b) = c^2 * (b + a - c)) :
    (2 * b + 3 * c) / a = 5 := 
sorry

end maximum_value_l70_70734


namespace no_real_y_for_common_solution_l70_70628

theorem no_real_y_for_common_solution :
  ∀ (x y : ℝ), x^2 + y^2 = 25 → x^2 + 3 * y = 45 → false :=
by 
sorry

end no_real_y_for_common_solution_l70_70628


namespace infinite_perfect_squares_in_ap_l70_70237

open Nat

def is_arithmetic_progression (a d : ℕ) (an : ℕ → ℕ) : Prop :=
  ∀ n, an n = a + n * d

def is_perfect_square (x : ℕ) : Prop :=
  ∃ m, m * m = x

theorem infinite_perfect_squares_in_ap (a d : ℕ) (an : ℕ → ℕ) (m : ℕ)
  (h_arith_prog : is_arithmetic_progression a d an)
  (h_initial_square : a = m * m) :
  ∃ (f : ℕ → ℕ), ∀ n, is_perfect_square (an (f n)) :=
sorry

end infinite_perfect_squares_in_ap_l70_70237


namespace inequality_proof_l70_70177

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + 1) * (b + 1) * (a + c) * (b + c) > 16 * a * b * c :=
by
  sorry

end inequality_proof_l70_70177


namespace numbers_not_squares_or_cubes_in_200_l70_70667

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l70_70667


namespace sqrt_9_is_rational_l70_70778

theorem sqrt_9_is_rational : ∃ q : ℚ, (q : ℝ) = 3 := by
  sorry

end sqrt_9_is_rational_l70_70778


namespace jen_triple_flips_l70_70273

-- Definitions based on conditions
def tyler_double_flips : ℕ := 12
def flips_per_double_flip : ℕ := 2
def flips_by_tyler : ℕ := tyler_double_flips * flips_per_double_flip
def flips_ratio : ℕ := 2
def flips_per_triple_flip : ℕ := 3
def flips_by_jen : ℕ := flips_by_tyler * flips_ratio

-- Lean 4 statement
theorem jen_triple_flips : flips_by_jen / flips_per_triple_flip = 16 :=
by 
    -- Proof contents should go here. We only need the statement as per the instruction.
    sorry

end jen_triple_flips_l70_70273


namespace hua_luogeng_optimal_selection_method_l70_70331

theorem hua_luogeng_optimal_selection_method:
  ∀ (method : Type) (golden_ratio : method) (mean : method) (mode : method) (median : method),
  method = golden_ratio :=
  by
    intros
    sorry

end hua_luogeng_optimal_selection_method_l70_70331


namespace initial_average_score_l70_70301

theorem initial_average_score (A : ℝ) :
  (∃ (A : ℝ), (16 * A = 15 * 64 + 24)) → A = 61.5 := 
by 
  sorry 

end initial_average_score_l70_70301


namespace monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l70_70827

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem monotonic_intervals_of_f :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂) ∧ (∀ x₁ x₂ : ℝ, x₁ > x₂ → f x₁ ≥ f x₂) :=
sorry

theorem f_gt_x_ln_x_plus_1 (x : ℝ) (hx : x > 0) : f x > x * Real.log (x + 1) :=
sorry

end monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l70_70827


namespace maximum_possible_value_of_N_l70_70514

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l70_70514


namespace numbers_not_squares_nor_cubes_1_to_200_l70_70649

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l70_70649


namespace path_length_of_dot_l70_70584

-- Define the edge length of the cube
def edge_length : ℝ := 3

-- Define the conditions of the problem
def cube_condition (l : ℝ) (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) : Prop :=
  l = edge_length ∧ rolling_without_slipping ∧ at_least_two_vertices_touching ∧ dot_at_one_corner ∧ returns_to_original_position

-- Define the theorem to be proven
theorem path_length_of_dot (rolling_without_slipping : Prop) (at_least_two_vertices_touching : Prop) (dot_at_one_corner : Prop) (returns_to_original_position : Prop) :
  cube_condition edge_length rolling_without_slipping at_least_two_vertices_touching dot_at_one_corner returns_to_original_position →
  ∃ c : ℝ, c = 6 ∧ (c * Real.pi) = 6 * Real.pi :=
by
  intro h
  sorry

end path_length_of_dot_l70_70584


namespace percent_profit_is_25_percent_l70_70145

theorem percent_profit_is_25_percent
  (CP SP : ℝ)
  (h : 75 * (CP - 0.05 * CP) = 60 * SP) :
  let profit := SP - (0.95 * CP)
  let percent_profit := (profit / (0.95 * CP)) * 100
  percent_profit = 25 :=
by
  sorry

end percent_profit_is_25_percent_l70_70145


namespace triangle_inequality_sqrt_sum_three_l70_70250

theorem triangle_inequality_sqrt_sum_three
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  3 ≤ (Real.sqrt (a / (-a + b + c)) + 
       Real.sqrt (b / (a - b + c)) + 
       Real.sqrt (c / (a + b - c))) := 
sorry

end triangle_inequality_sqrt_sum_three_l70_70250


namespace find_b_l70_70733

theorem find_b (b : ℚ) : (∃ x y : ℚ, x = 3 ∧ y = -5 ∧ (b * x - (b + 2) * y = b - 3)) → b = -13 / 7 :=
sorry

end find_b_l70_70733


namespace profit_functions_properties_l70_70100

noncomputable def R (x : ℝ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℝ) : ℝ := 500 * x + 4000
noncomputable def P (x : ℝ) : ℝ := R x - C x
noncomputable def MP (x : ℝ) : ℝ := P (x + 1) - P x

theorem profit_functions_properties :
  (P x = -20 * x^2 + 2500 * x - 4000) ∧ 
  (MP x = -40 * x + 2480) ∧ 
  (∃ x_max₁, ∀ x, P x_max₁ ≥ P x) ∧ 
  (∃ x_max₂, ∀ x, MP x_max₂ ≥ MP x) ∧ 
  P x_max₁ ≠ MP x_max₂ := by
  sorry

end profit_functions_properties_l70_70100


namespace gcd_8994_13326_37566_l70_70116

-- Define the integers involved
def a := 8994
def b := 13326
def c := 37566

-- Assert the GCD relation
theorem gcd_8994_13326_37566 : Int.gcd a (Int.gcd b c) = 2 := by
  sorry

end gcd_8994_13326_37566_l70_70116


namespace incorrect_statement_B_l70_70961

noncomputable def y (x : ℝ) : ℝ := 2 / x 

theorem incorrect_statement_B :
  ¬ ∀ x > 0, ∀ y1 y2 : ℝ, x < y1 → y1 < y2 → y x < y y2 := sorry

end incorrect_statement_B_l70_70961


namespace length_of_train_l70_70777

theorem length_of_train
  (L : ℝ) 
  (h1 : ∀ S, S = L / 8)
  (h2 : L + 267 = (L / 8) * 20) :
  L = 178 :=
sorry

end length_of_train_l70_70777


namespace arithmetic_sequence_x_values_l70_70475

theorem arithmetic_sequence_x_values {x : ℝ} (h_nonzero : x ≠ 0) (h_arith_seq : ∃ (k : ℤ), x - k = 1/2 ∧ x + 1 - (k + 1) = (k + 1) - 1/2) (h_lt_four : x < 4) :
  x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5 :=
by
  sorry

end arithmetic_sequence_x_values_l70_70475


namespace perpendicular_condition_l70_70120

noncomputable def line := ℝ → (ℝ × ℝ × ℝ)
noncomputable def plane := (ℝ × ℝ × ℝ) → Prop

variable {l m : line}
variable {α : plane}

-- l and m are two different lines
axiom lines_are_different : l ≠ m

-- m is parallel to the plane α
axiom m_parallel_alpha : ∀ t : ℝ, α (m t)

-- Prove that l perpendicular to α is a sufficient but not necessary condition for l perpendicular to m
theorem perpendicular_condition :
  (∀ t : ℝ, ¬ α (l t)) → (∀ t₁ t₂ : ℝ, (l t₁) ≠ (m t₂)) ∧ ¬ (∀ t : ℝ, ¬ α (l t)) :=
by 
  sorry

end perpendicular_condition_l70_70120


namespace find_number_l70_70803

theorem find_number (n x : ℕ) (h1 : n * (x - 1) = 21) (h2 : x = 4) : n = 7 :=
by
  sorry

end find_number_l70_70803


namespace solve_equation_l70_70725

theorem solve_equation (x y z t : ℤ) (h : x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0) : x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 :=
by
  sorry

end solve_equation_l70_70725


namespace jessica_total_spent_l70_70274

noncomputable def catToyCost : ℝ := 10.22
noncomputable def cageCost : ℝ := 11.73
noncomputable def totalCost : ℝ := 21.95

theorem jessica_total_spent :
  catToyCost + cageCost = totalCost :=
sorry

end jessica_total_spent_l70_70274


namespace students_at_school_yy_l70_70902

theorem students_at_school_yy (X Y : ℝ) 
    (h1 : X + Y = 4000)
    (h2 : 0.07 * X - 0.03 * Y = 40) : 
    Y = 2400 :=
by
  sorry

end students_at_school_yy_l70_70902


namespace optimal_selection_method_uses_golden_ratio_l70_70339

def Hua_Luogeng := "famous Chinese mathematician"

def optimal_selection_method := function (concept : Type) : Prop :=
  concept = "golden_ratio"

theorem optimal_selection_method_uses_golden_ratio (concept : Type) :
  optimal_selection_method concept → concept = "golden_ratio" :=
by
  intro h
  exact h

end optimal_selection_method_uses_golden_ratio_l70_70339


namespace evaluate_expression_l70_70400

theorem evaluate_expression : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 :=
by
  sorry

end evaluate_expression_l70_70400


namespace largest_number_l70_70795

-- Define the set elements with b = -3
def neg_5b (b : ℤ) : ℤ := -5 * b
def pos_3b (b : ℤ) : ℤ := 3 * b
def frac_30_b (b : ℤ) : ℤ := 30 / b
def b_sq (b : ℤ) : ℤ := b * b

-- Prove that when b = -3, the largest element in the set {-5b, 3b, 30/b, b^2, 2} is 15
theorem largest_number (b : ℤ) (h : b = -3) : max (max (max (max (neg_5b b) (pos_3b b)) (frac_30_b b)) (b_sq b)) 2 = 15 :=
by {
  sorry
}

end largest_number_l70_70795


namespace mean_of_six_numbers_l70_70039

theorem mean_of_six_numbers (sum : ℚ) (h : sum = 1/3) : (sum / 6 = 1/18) :=
by
  sorry

end mean_of_six_numbers_l70_70039


namespace ceil_floor_difference_l70_70452

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l70_70452


namespace rebus_problem_l70_70946

-- Define non-zero digit type
def NonZeroDigit := {d : Fin 10 // d.val ≠ 0}

-- Define the problem
theorem rebus_problem (A B C D : NonZeroDigit) (h1 : A.1 ≠ B.1) (h2 : A.1 ≠ C.1) (h3 : A.1 ≠ D.1) (h4 : B.1 ≠ C.1) (h5 : B.1 ≠ D.1) (h6 : C.1 ≠ D.1):
  let ABCD := 1000 * A.1 + 100 * B.1 + 10 * C.1 + D.1
  let ABCA := 1001 * A.1 + 100 * B.1 + 10 * C.1 + A.1
  ∃ (n : ℕ), ABCA = 182 * (10 * C.1 + D.1) → ABCD = 2916 :=
begin
  intro h,
  use 51, -- 2916 is 51 * 182
  sorry
end

end rebus_problem_l70_70946


namespace flattest_ellipse_is_B_l70_70103

-- Definitions for the given ellipses
def ellipseA : Prop := ∀ (x y : ℝ), (x^2 / 16 + y^2 / 12 = 1)
def ellipseB : Prop := ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1)
def ellipseC : Prop := ∀ (x y : ℝ), (x^2 / 6 + y^2 / 3 = 1)
def ellipseD : Prop := ∀ (x y : ℝ), (x^2 / 9 + y^2 / 8 = 1)

-- The proof to show that ellipseB is the flattest
theorem flattest_ellipse_is_B : ellipseB := by
  sorry

end flattest_ellipse_is_B_l70_70103


namespace odd_function_periodicity_l70_70812

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_periodicity (f_odd : ∀ x, f (-x) = -f x)
  (f_periodic : ∀ x, f (x + 2) = -f x) (f_val : f 1 = 2) : f 2011 = -2 :=
by
  sorry

end odd_function_periodicity_l70_70812


namespace ratio_of_boys_to_girls_l70_70571

theorem ratio_of_boys_to_girls {T G B : ℕ} (h1 : (2/3 : ℚ) * G = (1/4 : ℚ) * T) (h2 : T = G + B) : (B : ℚ) / G = 5 / 3 :=
by
  sorry

end ratio_of_boys_to_girls_l70_70571


namespace point_on_y_axis_l70_70713

theorem point_on_y_axis (m : ℝ) (M : ℝ × ℝ) (hM : M = (m + 1, m + 3)) (h_on_y_axis : M.1 = 0) : M = (0, 2) :=
by
  -- Proof omitted
  sorry

end point_on_y_axis_l70_70713


namespace scientific_notation_correct_l70_70999

noncomputable def significant_figures : ℝ := 274
noncomputable def decimal_places : ℝ := 8
noncomputable def scientific_notation_rep : ℝ := 2.74 * (10^8)

theorem scientific_notation_correct :
  274000000 = scientific_notation_rep :=
sorry

end scientific_notation_correct_l70_70999


namespace sum_of_consecutive_integers_l70_70363

theorem sum_of_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 930) : x + (x + 1) = 61 :=
sorry

end sum_of_consecutive_integers_l70_70363


namespace other_root_of_quadratic_l70_70818

theorem other_root_of_quadratic (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 + k * x - 5 = 0 ∧ x = 3) →
  ∃ r : ℝ, 3 * r * 3 = -5 / 3 ∧ r = -5 / 9 :=
by
  sorry

end other_root_of_quadratic_l70_70818


namespace solution_set_of_quadratic_inequality_l70_70037

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 4 * x - 5 > 0} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_quadratic_inequality_l70_70037


namespace solve_for_x_values_for_matrix_l70_70449

def matrix_equals_neg_two (x : ℝ) : Prop :=
  let a := 3 * x
  let b := x
  let c := 4
  let d := 2 * x
  (a * b - c * d = -2)

theorem solve_for_x_values_for_matrix : 
  ∃ (x : ℝ), matrix_equals_neg_two x ↔ (x = (4 + Real.sqrt 10) / 3 ∨ x = (4 - Real.sqrt 10) / 3) :=
sorry

end solve_for_x_values_for_matrix_l70_70449


namespace train_speed_l70_70074

theorem train_speed (d t s : ℝ) (h1 : d = 320) (h2 : t = 6) (h3 : s = 53.33) :
  s = d / t :=
by
  rw [h1, h2]
  sorry

end train_speed_l70_70074


namespace non_perfect_powers_count_l70_70661

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l70_70661


namespace diesel_usage_l70_70705

theorem diesel_usage (weekly_expenditure : ℝ) (cost_per_gallon : ℝ)
  (h_expenditure : weekly_expenditure = 36)
  (h_cost : cost_per_gallon = 3) :
  let weekly_gallons := weekly_expenditure / cost_per_gallon in
  let two_weeks_gallons := 2 * weekly_gallons in
  two_weeks_gallons = 24 := 
by
  sorry

end diesel_usage_l70_70705


namespace minimum_value_l70_70634

noncomputable def f (x : ℝ) (a b : ℝ) := a^x - b
noncomputable def g (x : ℝ) := x + 1

theorem minimum_value (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f (0 : ℝ) a b * g 0 ≤ 0)
  (h4 : ∀ x : ℝ, f x a b * g x ≤ 0) : (1 / a + 4 / b) ≥ 4 :=
sorry

end minimum_value_l70_70634


namespace multiple_of_a_l70_70537

theorem multiple_of_a's_share (A B : ℝ) (x : ℝ) (h₁ : A + B + 260 = 585) (h₂ : x * A = 780) (h₃ : 6 * B = 780) : x = 4 :=
sorry

end multiple_of_a_l70_70537


namespace margaret_mean_score_l70_70805

def sum_of_scores (scores : List ℤ) : ℤ :=
  scores.sum

def mean_score (total_score : ℤ) (count : ℕ) : ℚ :=
  total_score / count

theorem margaret_mean_score :
  let scores := [85, 88, 90, 92, 94, 96, 100]
  let cyprian_mean := 92
  let cyprian_count := 4
  let total_score := sum_of_scores scores
  let cyprian_total_score := cyprian_mean * cyprian_count
  let margaret_total_score := total_score - cyprian_total_score
  let margaret_mean := mean_score margaret_total_score 3
  margaret_mean = 92.33 :=
by
  sorry

end margaret_mean_score_l70_70805


namespace identity_1_identity_2_identity_3_l70_70003

-- Variables and assumptions
variables (a b c : ℝ)
variables (h_different : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0)

-- Part 1
theorem identity_1 : 
  (1 / ((a - b) * (a - c))) + (1 / ((b - c) * (b - a))) + (1 / ((c - a) * (c - b))) = 0 := 
by sorry

-- Part 2
theorem identity_2 :
  (a / ((a - b) * (a - c))) + (b / ((b - c) * (b - a))) + (c / ((c - a) * (c - b))) = 0 :=
by sorry

-- Part 3
theorem identity_3 :
  (a^2 / ((a - b) * (a - c))) + (b^2 / ((b - c) * (b - a))) + (c^2 / ((c - a) * (c - b))) = 1 :=
by sorry

end identity_1_identity_2_identity_3_l70_70003


namespace mean_proportional_l70_70821

theorem mean_proportional (a c x : ℝ) (ha : a = 9) (hc : c = 4) (hx : x^2 = a * c) : x = 6 := by
  sorry

end mean_proportional_l70_70821


namespace peregrine_falcon_dive_time_l70_70872

theorem peregrine_falcon_dive_time 
  (bald_eagle_speed : ℝ := 100) 
  (peregrine_falcon_speed : ℝ := 2 * bald_eagle_speed) 
  (bald_eagle_time : ℝ := 30) : 
  peregrine_falcon_speed = 2 * bald_eagle_speed ∧ peregrine_falcon_speed / bald_eagle_speed = 2 →
  ∃ peregrine_falcon_time : ℝ, peregrine_falcon_time = 15 :=
by
  intro h
  use (bald_eagle_time / 2)
  sorry

end peregrine_falcon_dive_time_l70_70872


namespace area_inside_S_outside_R_l70_70013

theorem area_inside_S_outside_R (area_R area_S : ℝ) (h1: area_R = 1 + 3 * Real.sqrt 3) (h2: area_S = 6 * Real.sqrt 3) :
  area_S - area_R = 1 :=
by {
   sorry
}

end area_inside_S_outside_R_l70_70013


namespace mary_can_keep_warm_l70_70170

theorem mary_can_keep_warm :
  let chairs := 18
  let chairs_sticks := 6
  let tables := 6
  let tables_sticks := 9
  let stools := 4
  let stools_sticks := 2
  let sticks_per_hour := 5
  let total_sticks := (chairs * chairs_sticks) + (tables * tables_sticks) + (stools * stools_sticks)
  let hours := total_sticks / sticks_per_hour
  hours = 34 := by
{
  sorry
}

end mary_can_keep_warm_l70_70170


namespace abc_inequality_l70_70526

theorem abc_inequality (x y z : ℝ) (a b c : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a = (x * (y - z) ^ 2) ^ 2) (h2 : b = (y * (z - x) ^ 2) ^ 2) (h3 : c = (z * (x - y) ^ 2) ^ 2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) :=
by {
  sorry
}

end abc_inequality_l70_70526


namespace find_k_l70_70032

theorem find_k (k : ℚ) :
  (∃ (x y : ℚ), y = 4 * x + 5 ∧ y = -3 * x + 10 ∧ y = 2 * x + k) →
  k = 45 / 7 :=
by
  sorry

end find_k_l70_70032


namespace max_value_E_zero_l70_70276

noncomputable def E (a b c : ℝ) : ℝ :=
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2)

theorem max_value_E_zero (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≥ b * c^2) (h2 : b ≥ c * a^2) (h3 : c ≥ a * b^2) :
  E a b c ≤ 0 :=
by
  sorry

end max_value_E_zero_l70_70276


namespace divisible_by_5_l70_70716

theorem divisible_by_5 (n : ℕ) : (∃ k : ℕ, 2^n - 1 = 5 * k) ∨ (∃ k : ℕ, 2^n + 1 = 5 * k) ∨ (∃ k : ℕ, 2^(2*n) + 1 = 5 * k) :=
sorry

end divisible_by_5_l70_70716


namespace sqrt_defined_iff_le_l70_70836

theorem sqrt_defined_iff_le (x : ℝ) : (∃ y : ℝ, y^2 = 4 - x) ↔ (x ≤ 4) :=
by
  sorry

end sqrt_defined_iff_le_l70_70836


namespace third_side_of_triangle_l70_70174

theorem third_side_of_triangle (a b : ℝ) (γ : ℝ) (x : ℝ) 
  (ha : a = 6) (hb : b = 2 * Real.sqrt 7) (hγ : γ = Real.pi / 3) :
  x = 2 ∨ x = 4 :=
by 
  sorry

end third_side_of_triangle_l70_70174


namespace complex_magnitude_pow_eight_l70_70617

theorem complex_magnitude_pow_eight :
  (Complex.abs ((2/5 : ℂ) + (7/5 : ℂ) * Complex.I))^8 = 7890481 / 390625 := 
by
  sorry

end complex_magnitude_pow_eight_l70_70617


namespace water_in_bowl_after_adding_4_cups_l70_70063

def total_capacity_bowl := 20 -- Capacity of the bowl in cups

def initially_half_full (C : ℕ) : Prop :=
C = total_capacity_bowl / 2

def after_adding_4_cups (initial : ℕ) : ℕ :=
initial + 4

def seventy_percent_full (C : ℕ) : ℕ :=
7 * C / 10

theorem water_in_bowl_after_adding_4_cups :
  ∀ (C initial after_adding) (h1 : initially_half_full initial)
  (h2 : after_adding = after_adding_4_cups initial)
  (h3 : after_adding = seventy_percent_full C),
  after_adding = 14 := 
by
  intros C initial after_adding h1 h2 h3
  -- Proof goes here
  sorry

end water_in_bowl_after_adding_4_cups_l70_70063


namespace calculate_savings_l70_70437

def monthly_income : list ℕ := [45000, 35000, 7000, 10000, 13000]
def monthly_expenses : list ℕ := [30000, 10000, 5000, 4500, 9000]
def initial_savings : ℕ := 849400

def total_income : ℕ := 5 * monthly_income.sum
def total_expenses : ℕ := 5 * monthly_expenses.sum
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem calculate_savings :
  total_income = 550000 ∧
  total_expenses = 292500 ∧
  final_savings = 1106900 :=
by
  sorry

end calculate_savings_l70_70437


namespace slope_of_tangent_at_1_l70_70880

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem slope_of_tangent_at_1 : (deriv f 1) = 1 / 2 :=
  by
  sorry

end slope_of_tangent_at_1_l70_70880


namespace quadratic_discriminant_eq_l70_70549

theorem quadratic_discriminant_eq (a b c n : ℤ) (h_eq : a = 3) (h_b : b = -8) (h_c : c = -5)
  (h_discriminant : b^2 - 4 * a * c = n) : n = 124 := 
by
  -- proof skipped
  sorry

end quadratic_discriminant_eq_l70_70549


namespace linear_function_no_third_quadrant_l70_70148

theorem linear_function_no_third_quadrant (k b : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ k * x + b) : k < 0 ∧ 0 ≤ b :=
sorry

end linear_function_no_third_quadrant_l70_70148


namespace hua_luogeng_optimal_selection_method_l70_70341

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l70_70341


namespace shot_put_distance_l70_70420

theorem shot_put_distance :
  (∃ x : ℝ, (y = - 1 / 12 * x^2 + 2 / 3 * x + 5 / 3) ∧ y = 0) ↔ x = 10 := 
by
  sorry

end shot_put_distance_l70_70420


namespace prime_sum_product_l70_70883

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 :=
sorry

end prime_sum_product_l70_70883


namespace least_four_digit_perfect_square_and_cube_l70_70558

theorem least_four_digit_perfect_square_and_cube :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ m1 : ℕ, n = m1^2) ∧ (∃ m2 : ℕ, n = m2^3) ∧ n = 4096 := sorry

end least_four_digit_perfect_square_and_cube_l70_70558


namespace num_non_squares_cubes_1_to_200_l70_70673

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l70_70673


namespace ticket_distribution_l70_70916

theorem ticket_distribution 
    (A Ad C Cd S : ℕ) 
    (h1 : 25 * A + 20 * 50 + 15 * C + 10 * 30 + 20 * S = 7200) 
    (h2 : A + 50 + C + 30 + S = 400)
    (h3 : A + 50 = 2 * S)
    (h4 : Ad = 50)
    (h5 : Cd = 30) : 
    A = 102 ∧ Ad = 50 ∧ C = 142 ∧ Cd = 30 ∧ S = 76 := 
by 
    sorry

end ticket_distribution_l70_70916


namespace height_of_parallelogram_l70_70624

-- Define the problem statement
theorem height_of_parallelogram (A : ℝ) (b : ℝ) (h : ℝ) (h_eq : A = b * h) (A_val : A = 384) (b_val : b = 24) : h = 16 :=
by
  -- Skeleton proof, include the initial conditions and proof statement
  sorry

end height_of_parallelogram_l70_70624


namespace cot_sum_simplified_l70_70724

noncomputable def cot (x : ℝ) : ℝ := (Real.cos x) / (Real.sin x)

theorem cot_sum_simplified : cot (π / 24) + cot (π / 8) = 96 / (π^2) := 
by 
  sorry

end cot_sum_simplified_l70_70724


namespace min_number_of_girls_l70_70383

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l70_70383


namespace arithmetic_seq_general_formula_l70_70967

-- Definitions based on given conditions
def f (x : ℝ) := x^2 - 2*x + 4
def a (n : ℕ) (d : ℝ) := f (d + n - 1) 

-- The general term formula for the arithmetic sequence
theorem arithmetic_seq_general_formula (d : ℝ) :
  (a 1 d = f (d - 1)) →
  (a 3 d = f (d + 1)) →
  (∀ n : ℕ, a n d = 2*n + 1) :=
by
  intros h1 h3
  sorry

end arithmetic_seq_general_formula_l70_70967


namespace angle_of_squares_attached_l70_70743

-- Definition of the problem scenario:
-- Three squares attached as described, needing to prove x = 39 degrees.

open Real

theorem angle_of_squares_attached (x : ℝ) (h : 
  let angle1 := 30
  let angle2 := 126
  let angle3 := 75
  angle1 + angle2 + angle3 + x = 3 * 90) :
  x = 39 :=
by 
  -- This proof is omitted
  sorry

end angle_of_squares_attached_l70_70743


namespace product_of_four_consecutive_even_numbers_divisible_by_240_l70_70903

theorem product_of_four_consecutive_even_numbers_divisible_by_240 :
  ∀ (n : ℤ), (n % 2 = 0) →
    (n + 2) % 2 = 0 →
    (n + 4) % 2 = 0 →
    (n + 6) % 2 = 0 →
    ((n * (n + 2) * (n + 4) * (n + 6)) % 240 = 0) :=
by
  intro n hn hnp2 hnp4 hnp6
  sorry

end product_of_four_consecutive_even_numbers_divisible_by_240_l70_70903


namespace distance_problem_l70_70109

noncomputable def distance_point_to_plane 
  (x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ) : ℝ :=
  -- Equation of the plane passing through three points derived using determinants
  let a := x2 - x1
  let b := y2 - y1
  let c := z2 - z1
  let d := x3 - x1
  let e := y3 - y1
  let f := z3 - z1
  let A := b*f - c*e
  let B := c*d - a*f
  let C := a*e - b*d
  let D := -(A*x1 + B*y1 + C*z1)
  -- Distance from the given point to the above plane
  (|A*x0 + B*y0 + C*z0 + D|) / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_problem :
  distance_point_to_plane 
  3 6 68 
  (-3) (-5) 6 
  2 1 (-4) 
  0 (-3) (-1) 
  = Real.sqrt 573 :=
by sorry

end distance_problem_l70_70109


namespace latoya_initial_payment_l70_70523

variable (cost_per_minute : ℝ) (call_duration : ℝ) (remaining_credit : ℝ) 
variable (initial_credit : ℝ)

theorem latoya_initial_payment : 
  ∀ (cost_per_minute call_duration remaining_credit initial_credit : ℝ),
  cost_per_minute = 0.16 →
  call_duration = 22 →
  remaining_credit = 26.48 →
  initial_credit = (cost_per_minute * call_duration) + remaining_credit →
  initial_credit = 30 :=
by
  intros cost_per_minute call_duration remaining_credit initial_credit
  sorry

end latoya_initial_payment_l70_70523


namespace complete_the_square_l70_70492

theorem complete_the_square :
  ∀ x : ℝ, (x^2 - 2 * x - 2 = 0) → ((x - 1)^2 = 3) :=
by
  intros x h
  sorry

end complete_the_square_l70_70492


namespace hua_luogeng_optimal_selection_method_l70_70342

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l70_70342


namespace lizette_stamps_count_l70_70702

-- Conditions
def lizette_more : ℕ := 125
def minerva_stamps : ℕ := 688

-- Proof of Lizette's stamps count
theorem lizette_stamps_count : (minerva_stamps + lizette_more = 813) :=
by 
  sorry

end lizette_stamps_count_l70_70702


namespace mean_of_six_numbers_l70_70041

theorem mean_of_six_numbers (a b c d e f : ℚ) (h : a + b + c + d + e + f = 1 / 3) :
  (a + b + c + d + e + f) / 6 = 1 / 18 :=
by
  sorry

end mean_of_six_numbers_l70_70041


namespace mother_reaches_timothy_l70_70887

/--
Timothy leaves home for school, riding his bicycle at a rate of 6 miles per hour.
Fifteen minutes after he leaves, his mother sees Timothy's math homework lying on his bed and immediately leaves home to bring it to him.
If his mother drives at 36 miles per hour, prove that she must drive 1.8 miles to reach Timothy.
-/
theorem mother_reaches_timothy
  (timothy_speed : ℕ)
  (mother_speed : ℕ)
  (delay_minutes : ℕ)
  (distance_must_drive : ℕ)
  (h_speed_t : timothy_speed = 6)
  (h_speed_m : mother_speed = 36)
  (h_delay : delay_minutes = 15)
  (h_distance : distance_must_drive = 18 / 10 ) :
  ∃ t : ℚ, (timothy_speed * (delay_minutes / 60) + timothy_speed * t) = (mother_speed * t) := sorry

end mother_reaches_timothy_l70_70887


namespace min_girls_in_class_l70_70382

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l70_70382


namespace max_possible_value_l70_70508

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l70_70508


namespace plane_equation_l70_70924

variable (x y z : ℝ)

def pointA : ℝ × ℝ × ℝ := (3, 0, 0)
def normalVector : ℝ × ℝ × ℝ := (2, -3, 1)

theorem plane_equation : 
  ∃ a b c d, normalVector = (a, b, c) ∧ pointA = (x, y, z) ∧ a * (x - 3) + b * y + c * z = d ∧ d = -6 := 
  sorry

end plane_equation_l70_70924


namespace obtuse_angle_in_second_quadrant_l70_70779

-- Let θ be an angle in degrees
def angle_in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def angle_terminal_side_same (θ₁ θ₂ : ℝ) : Prop := θ₁ % 360 = θ₂ % 360

def angle_in_fourth_quadrant (θ : ℝ) : Prop := -360 < θ ∧ θ < 0 ∧ (θ + 360) > 270

def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Statement D: An obtuse angle is definitely in the second quadrant
theorem obtuse_angle_in_second_quadrant (θ : ℝ) (h : is_obtuse_angle θ) :
  90 < θ ∧ θ < 180 := by
    sorry

end obtuse_angle_in_second_quadrant_l70_70779


namespace rebus_problem_l70_70945

-- Define non-zero digit type
def NonZeroDigit := {d : Fin 10 // d.val ≠ 0}

-- Define the problem
theorem rebus_problem (A B C D : NonZeroDigit) (h1 : A.1 ≠ B.1) (h2 : A.1 ≠ C.1) (h3 : A.1 ≠ D.1) (h4 : B.1 ≠ C.1) (h5 : B.1 ≠ D.1) (h6 : C.1 ≠ D.1):
  let ABCD := 1000 * A.1 + 100 * B.1 + 10 * C.1 + D.1
  let ABCA := 1001 * A.1 + 100 * B.1 + 10 * C.1 + A.1
  ∃ (n : ℕ), ABCA = 182 * (10 * C.1 + D.1) → ABCD = 2916 :=
begin
  intro h,
  use 51, -- 2916 is 51 * 182
  sorry
end

end rebus_problem_l70_70945


namespace factorization_correctness_l70_70223

theorem factorization_correctness :
  (∀ x : ℝ, (x + 1) * (x - 1) = x^2 - 1 → false) ∧
  (∀ x : ℝ, x^2 - 4 * x + 4 = x * (x - 4) + 4 → false) ∧
  (∀ x : ℝ, (x + 3) * (x - 4) = x^2 - x - 12 → false) ∧
  (∀ x : ℝ, x^2 - 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correctness_l70_70223


namespace max_value_of_vector_dot_product_l70_70486

theorem max_value_of_vector_dot_product :
  ∀ (x y : ℝ), (-2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2) → (2 * x - y ≤ 4) :=
by
  intros x y h
  sorry

end max_value_of_vector_dot_product_l70_70486


namespace initial_albums_in_cart_l70_70171

theorem initial_albums_in_cart (total_songs : ℕ) (songs_per_album : ℕ) (removed_albums : ℕ) 
  (h_total: total_songs = 42) 
  (h_songs_per_album: songs_per_album = 7)
  (h_removed: removed_albums = 2): 
  (total_songs / songs_per_album) + removed_albums = 8 := 
by
  sorry

end initial_albums_in_cart_l70_70171


namespace teacher_students_and_ticket_cost_l70_70184

theorem teacher_students_and_ticket_cost 
    (C_s C_a : ℝ) 
    (n_k n_h : ℕ)
    (hk_total ht_total : ℝ) 
    (h_students : n_h = n_k + 3)
    (hk  : n_k * C_s + C_a = hk_total)
    (ht : n_h * C_s + C_a = ht_total)
    (hk_total_val : hk_total = 994)
    (ht_total_val : ht_total = 1120)
    (C_s_val : C_s = 42) : 
    (n_h = 25) ∧ (C_a = 70) := 
by
  -- Proof steps would be provided here
  sorry

end teacher_students_and_ticket_cost_l70_70184


namespace smallest_n_divisible_by_100_million_l70_70528

noncomputable def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

noncomputable def nth_term (a1 r : ℚ) (n : ℕ) : ℚ := a1 * r^(n - 1)

theorem smallest_n_divisible_by_100_million :
  ∀ (a1 a2 : ℚ), a1 = 5/6 → a2 = 25 → 
  ∃ n : ℕ, nth_term a1 (common_ratio a1 a2) n % 100000000 = 0 ∧ n = 9 :=
by
  intros a1 a2 h1 h2
  have r := common_ratio a1 a2
  have a9 := nth_term a1 r 9
  sorry

end smallest_n_divisible_by_100_million_l70_70528


namespace parts_sampling_l70_70580

theorem parts_sampling (first_grade second_grade third_grade : ℕ)
                       (total_sample drawn_third : ℕ)
                       (h_first_grade : first_grade = 24)
                       (h_second_grade : second_grade = 36)
                       (h_total_sample : total_sample = 20)
                       (h_drawn_third : drawn_third = 10)
                       (h_non_third : third_grade = 60 - (24 + 36))
                       (h_total : 2 * (24 + 36) = 120)
                       (h_proportion : 2 * third_grade = 2 * (24 + 36)) :
    (third_grade = 60 ∧ (second_grade * (total_sample - drawn_third) / (24 + 36) = 6)) := by
    simp [h_first_grade, h_second_grade, h_total_sample, h_drawn_third] at *
    sorry

end parts_sampling_l70_70580


namespace ratio_of_average_speeds_l70_70932

-- Definitions based on the conditions
def distance_AB := 600 -- km
def distance_AC := 300 -- km
def time_Eddy := 3 -- hours
def time_Freddy := 3 -- hours

def speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def speed_Eddy := speed distance_AB time_Eddy
def speed_Freddy := speed distance_AC time_Freddy

theorem ratio_of_average_speeds : (speed_Eddy / speed_Freddy) = 2 :=
by 
  -- Proof is skipped, so we use sorry
  sorry

end ratio_of_average_speeds_l70_70932


namespace exists_k_undecisive_tournament_l70_70002

-- Definitions based on conditions
def tournament (n : ℕ) := Finset (Finset (Fin n))

def k_undecisive_tournament (k n : ℕ) (T : tournament n) : Prop :=
  ∀ (A : Finset (Fin n)), A.card = k → 
  ∃ (x : Fin n), x ∉ A ∧ (∀ (y : Fin n), y ∈ A → (x, y) ∈ T)

-- Statement of the theorem
theorem exists_k_undecisive_tournament (k : ℕ) (hk : k > 0) :
  ∃ n, n > k ∧ ∃ T : tournament n, k_undecisive_tournament k n T :=
sorry

end exists_k_undecisive_tournament_l70_70002


namespace tom_found_seashells_l70_70197

theorem tom_found_seashells : ∀ (days : ℕ) (seashells_per_day : ℕ), days = 5 ∧ seashells_per_day = 7 → days * seashells_per_day = 35 := 
by
  intros days seashells_per_day h
  cases h with h_days h_seashells_per_day
  rw [h_days, h_seashells_per_day]
  exact nat.mul_comm 5 7 ▸ rfl

end tom_found_seashells_l70_70197


namespace solve_rebus_l70_70950

-- Definitions for the conditions
def is_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

def distinct_digits (A B C D : Nat) : Prop := 
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Main Statement
theorem solve_rebus (A B C D : Nat) (h_distinct : distinct_digits A B C D) 
(h_eq : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
by
  sorry

end solve_rebus_l70_70950


namespace irrational_roots_of_odd_coeffs_l70_70717

theorem irrational_roots_of_odd_coeffs (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := 
sorry

end irrational_roots_of_odd_coeffs_l70_70717


namespace Yvonne_laps_l70_70406

-- Definitions of the given conditions
def laps_swim_by_Yvonne (l_y : ℕ) : Prop := 
  ∃ l_s l_j, 
  l_s = l_y / 2 ∧ 
  l_j = 3 * l_s ∧ 
  l_j = 15

-- Theorem statement
theorem Yvonne_laps (l_y : ℕ) (h : laps_swim_by_Yvonne l_y) : l_y = 10 :=
sorry

end Yvonne_laps_l70_70406


namespace large_box_times_smaller_box_l70_70051

noncomputable def large_box_volume (width length height : ℕ) : ℕ := width * length * height

noncomputable def small_box_volume (width length height : ℕ) : ℕ := width * length * height

theorem large_box_times_smaller_box :
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  large_volume / small_volume = 125 :=
by
  let large_volume := large_box_volume 30 20 5
  let small_volume := small_box_volume 6 4 1
  show large_volume / small_volume = 125
  sorry

end large_box_times_smaller_box_l70_70051


namespace quadratic_sum_of_coefficients_l70_70433

theorem quadratic_sum_of_coefficients (x : ℝ) : 
  let a := 1
  let b := 1
  let c := -4
  a + b + c = -2 :=
by
  sorry

end quadratic_sum_of_coefficients_l70_70433


namespace total_nails_needed_l70_70481

-- Given conditions
def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

-- Prove the total number of nails required
theorem total_nails_needed : nails_per_plank * number_of_planks = 32 :=
by
  sorry

end total_nails_needed_l70_70481


namespace distance_to_destination_l70_70530

theorem distance_to_destination (x : ℕ) 
    (condition_1 : True)  -- Manex is a tour bus driver. Ignore in the proof.
    (condition_2 : True)  -- Ignores the fact that the return trip is using a different path.
    (condition_3 : x / 30 + (x + 10) / 30 + 2 = 6) : 
    x = 55 :=
sorry

end distance_to_destination_l70_70530


namespace greatest_number_of_consecutive_integers_whose_sum_is_36_l70_70202

/-- 
Given that the sum of N consecutive integers starting from a is 36, 
prove that the greatest possible value of N is 72.
-/
theorem greatest_number_of_consecutive_integers_whose_sum_is_36 :
  ∀ (N a : ℤ), (N > 0) → (N * (2 * a + N - 1)) = 72 → N ≤ 72 := 
by
  intros N a hN h
  sorry

end greatest_number_of_consecutive_integers_whose_sum_is_36_l70_70202


namespace problem_solution_l70_70477

theorem problem_solution (x : ℝ) :
    (x^2 / (x - 2) ≥ (3 / (x + 2)) + (7 / 5)) →
    (x ∈ Set.Ioo (-2 : ℝ) 2 ∪ Set.Ioi (2 : ℝ)) :=
by
  intro h
  sorry

end problem_solution_l70_70477


namespace find_b_l70_70244

theorem find_b (b : ℤ) :
  (∃ x : ℤ, x^2 + b * x - 36 = 0 ∧ x = -9) → b = 5 :=
by
  sorry

end find_b_l70_70244


namespace no_int_solutions_for_equation_l70_70862

theorem no_int_solutions_for_equation :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * y^2 = x^4 + x := 
sorry

end no_int_solutions_for_equation_l70_70862


namespace mark_egg_supply_in_a_week_l70_70284

def dozen := 12
def eggs_per_day_store1 := 5 * dozen
def eggs_per_day_store2 := 30
def daily_eggs_supplied := eggs_per_day_store1 + eggs_per_day_store2
def days_per_week := 7

theorem mark_egg_supply_in_a_week : daily_eggs_supplied * days_per_week = 630 := by
  sorry

end mark_egg_supply_in_a_week_l70_70284


namespace original_selling_price_l70_70774

theorem original_selling_price (CP SP_original SP_loss : ℝ)
  (h1 : SP_original = CP * 1.25)
  (h2 : SP_loss = CP * 0.85)
  (h3 : SP_loss = 544) : SP_original = 800 :=
by
  -- The proof goes here, but we are skipping it with sorry
  sorry

end original_selling_price_l70_70774


namespace snowdrift_depth_end_of_third_day_l70_70913

theorem snowdrift_depth_end_of_third_day :
  let depth_ninth_day := 40
  let d_before_eighth_night_snowfall := depth_ninth_day - 10
  let d_before_eighth_day_melting := d_before_eighth_night_snowfall * 4 / 3
  let depth_seventh_day := d_before_eighth_day_melting
  let d_before_sixth_day_snowfall := depth_seventh_day - 20
  let d_before_fifth_day_snowfall := d_before_sixth_day_snowfall - 15
  let d_before_fourth_day_melting := d_before_fifth_day_snowfall * 3 / 2
  depth_ninth_day = 40 →
  d_before_eighth_night_snowfall = depth_ninth_day - 10 →
  d_before_eighth_day_melting = d_before_eighth_night_snowfall * 4 / 3 →
  depth_seventh_day = d_before_eighth_day_melting →
  d_before_sixth_day_snowfall = depth_seventh_day - 20 →
  d_before_fifth_day_snowfall = d_before_sixth_day_snowfall - 15 →
  d_before_fourth_day_melting = d_before_fifth_day_snowfall * 3 / 2 →
  d_before_fourth_day_melting = 7.5 :=
by
  intros
  sorry

end snowdrift_depth_end_of_third_day_l70_70913


namespace math_problem_l70_70468

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l70_70468


namespace ceil_floor_difference_l70_70455

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l70_70455


namespace total_canoes_proof_l70_70095

def n_canoes_january : ℕ := 5
def n_canoes_february : ℕ := 3 * n_canoes_january
def n_canoes_march : ℕ := 3 * n_canoes_february
def n_canoes_april : ℕ := 3 * n_canoes_march

def total_canoes_built : ℕ :=
  n_canoes_january + n_canoes_february + n_canoes_march + n_canoes_april

theorem total_canoes_proof : total_canoes_built = 200 := 
  by
  sorry

end total_canoes_proof_l70_70095


namespace range_of_x_l70_70837

-- Define the condition where the expression sqrt(4 - x) is meaningful
def condition (x : ℝ) : Prop := sqrt (4 - x) ∈ ℝ

-- Proof that x ≤ 4 given the condition
theorem range_of_x (x : ℝ) (h : 4 - x ≥ 0) : x ≤ 4 :=
by
  sorry

end range_of_x_l70_70837


namespace optimal_selection_method_uses_golden_ratio_l70_70323

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l70_70323


namespace quadrilateral_area_inequality_equality_condition_l70_70525

theorem quadrilateral_area_inequality 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d) 
  : S ≤ 0.5 * (a * c + b * d) :=
sorry

theorem equality_condition 
  (a b c d S : ℝ) 
  (hS : S = 0.5 * a * c + 0.5 * b * d)
  (h_perpendicular : ∃ (α β : ℝ), α = 90 ∧ β = 90) 
  : S = 0.5 * (a * c + b * d) :=
sorry

end quadrilateral_area_inequality_equality_condition_l70_70525


namespace smallest_n_geometric_seq_l70_70236

noncomputable def geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

noncomputable def S_n (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r ^ n) / (1 - r)

theorem smallest_n_geometric_seq :
  (∃ n : ℕ, S_n (1/9) 3 n > 2018) ∧ ∀ m : ℕ, m < 10 → S_n (1/9) 3 m ≤ 2018 :=
by
  sorry

end smallest_n_geometric_seq_l70_70236


namespace diamond_evaluation_l70_70926

-- Define the diamond operation as a function using the given table
def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4 | (1, 2) => 1 | (1, 3) => 3 | (1, 4) => 2
  | (2, 1) => 1 | (2, 2) => 3 | (2, 3) => 2 | (2, 4) => 4
  | (3, 1) => 3 | (3, 2) => 2 | (3, 3) => 4 | (3, 4) => 1
  | (4, 1) => 2 | (4, 2) => 4 | (4, 3) => 1 | (4, 4) => 3
  | (_, _) => 0  -- default case (should not occur)

-- State the proof problem
theorem diamond_evaluation : diamond (diamond 3 1) (diamond 4 2) = 1 := by
  sorry

end diamond_evaluation_l70_70926


namespace cube_surface_area_l70_70729

-- Define the edge length of the cube
def edge_length : ℝ := 4

-- Define the formula for the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Prove that given the edge length is 4 cm, the surface area is 96 cm²
theorem cube_surface_area : surface_area edge_length = 96 := by
  -- Proof goes here
  sorry

end cube_surface_area_l70_70729


namespace min_fraction_value_l70_70360

theorem min_fraction_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_tangent : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 :=
by
  sorry

end min_fraction_value_l70_70360


namespace goods_purchase_solutions_l70_70865

theorem goods_purchase_solutions (a : ℕ) (h1 : 0 < a ∧ a ≤ 45) :
  ∃ x : ℝ, 45 - 20 * (x - 1) = a * x :=
by sorry

end goods_purchase_solutions_l70_70865


namespace non_perfect_powers_count_l70_70663

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l70_70663


namespace prove_x_plus_y_leq_zero_l70_70816

-- Definitions of the conditions
def valid_powers (a b : ℝ) (x y : ℝ) : Prop :=
  1 < a ∧ a < b ∧ a^x + b^y ≤ a^(-x) + b^(-y)

-- The theorem statement
theorem prove_x_plus_y_leq_zero (a b x y : ℝ) (h : valid_powers a b x y) : 
  x + y ≤ 0 :=
by
  sorry

end prove_x_plus_y_leq_zero_l70_70816


namespace solve_rebus_l70_70949

-- Definitions for the conditions
def is_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

def distinct_digits (A B C D : Nat) : Prop := 
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Main Statement
theorem solve_rebus (A B C D : Nat) (h_distinct : distinct_digits A B C D) 
(h_eq : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
by
  sorry

end solve_rebus_l70_70949


namespace f_neg_m_l70_70825

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the problem as a theorem
theorem f_neg_m (a b m : ℝ) (h : f a b m = 6) : f a b (-m) = -4 :=
by
  -- Proof is not required
  sorry

end f_neg_m_l70_70825


namespace problem_solution_l70_70799

noncomputable def f (A B : ℝ) (x : ℝ) : ℝ := A + B / x + x

theorem problem_solution (A B : ℝ) :
  ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 →
  (x * f A B (x + 1 / y) + y * f A B y + y / x = y * f A B (y + 1 / x) + x * f A B x + x / y) :=
by
  sorry

end problem_solution_l70_70799


namespace find_x_for_parallel_vectors_l70_70822

noncomputable def vector_m : (ℝ × ℝ) := (1, 2)
noncomputable def vector_n (x : ℝ) : (ℝ × ℝ) := (x, 2 - 2 * x)

theorem find_x_for_parallel_vectors :
  ∀ x : ℝ, (1, 2).fst * (2 - 2 * x) - (1, 2).snd * x = 0 → x = 1 / 2 :=
by
  intros
  exact sorry

end find_x_for_parallel_vectors_l70_70822


namespace min_forget_all_three_l70_70180

theorem min_forget_all_three (total_students students_forgot_gloves students_forgot_scarves students_forgot_hats : ℕ) (h_total : total_students = 60) (h_gloves : students_forgot_gloves = 55) (h_scarves : students_forgot_scarves = 52) (h_hats : students_forgot_hats = 50) :
  ∃ min_students_forget_three, min_students_forget_three = total_students - (total_students - students_forgot_gloves + total_students - students_forgot_scarves + total_students - students_forgot_hats) :=
by
  use 37
  sorry

end min_forget_all_three_l70_70180


namespace convex_polygon_num_sides_l70_70731

theorem convex_polygon_num_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 120 + i * 5 < 180) 
  (h2 : (n - 2) * 180 = n * (240 + (n - 1) * 5) / 2) : 
  n = 9 :=
sorry

end convex_polygon_num_sides_l70_70731


namespace productivity_increase_is_233_33_percent_l70_70272

noncomputable def productivity_increase :
  Real :=
  let B := 1 -- represents the base number of bears made per week
  let H := 1 -- represents the base number of hours worked per week
  let P := B / H -- base productivity in bears per hour

  let B1 := 1.80 * B -- bears per week with first assistant
  let H1 := 0.90 * H -- hours per week with first assistant
  let P1 := B1 / H1 -- productivity with first assistant

  let B2 := 1.60 * B -- bears per week with second assistant
  let H2 := 0.80 * H -- hours per week with second assistant
  let P2 := B2 / H2 -- productivity with second assistant

  let B_both := B1 + B2 - B -- total bears with both assistants
  let H_both := H1 * H2 / H -- total hours with both assistants
  let P_both := B_both / H_both -- productivity with both assistants

  (P_both / P - 1) * 100

theorem productivity_increase_is_233_33_percent :
  productivity_increase = 233.33 :=
by
  sorry

end productivity_increase_is_233_33_percent_l70_70272


namespace bear_cubs_count_l70_70076

theorem bear_cubs_count (total_meat : ℕ) (meat_per_cub : ℕ) (rabbits_per_day : ℕ) (weeks_days : ℕ) (meat_per_rabbit : ℕ)
  (mother_total_meat : ℕ) (number_of_cubs : ℕ) : 
  total_meat = 210 →
  meat_per_cub = 35 →
  rabbits_per_day = 10 →
  weeks_days = 7 →
  meat_per_rabbit = 5 →
  mother_total_meat = rabbits_per_day * weeks_days * meat_per_rabbit →
  meat_per_cub * number_of_cubs + mother_total_meat = total_meat →
  number_of_cubs = 4 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end bear_cubs_count_l70_70076


namespace monica_sees_121_individual_students_l70_70861

def students_count : ℕ :=
  let class1 := 20
  let class2 := 25
  let class3 := 25
  let class4 := class1 / 2
  let class5 := 28
  let class6 := 28
  let total_spots := class1 + class2 + class3 + class4 + class5 + class6
  let overlap12 := 5
  let overlap45 := 3
  let overlap36 := 7
  total_spots - overlap12 - overlap45 - overlap36

theorem monica_sees_121_individual_students : students_count = 121 := by
  sorry

end monica_sees_121_individual_students_l70_70861


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l70_70321

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l70_70321


namespace min_girls_in_class_l70_70376

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l70_70376


namespace solve_x_of_det_8_l70_70712

variable (x : ℝ)

def matrix_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem solve_x_of_det_8
  (h : matrix_det (x + 1) (1 - x) (1 - x) (x + 1) = 8) : x = 2 := by
  sorry

end solve_x_of_det_8_l70_70712


namespace math_problem_l70_70466

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l70_70466


namespace linemen_ounces_per_drink_l70_70086

-- Definitions corresponding to the conditions.
def linemen := 12
def skill_position_drink := 6
def skill_position_before_refill := 5
def cooler_capacity := 126

-- The theorem that requires proof.
theorem linemen_ounces_per_drink (L : ℕ) (h : 12 * L + 5 * skill_position_drink = cooler_capacity) : L = 8 :=
by
  sorry

end linemen_ounces_per_drink_l70_70086


namespace no_injective_function_l70_70570

theorem no_injective_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m * n) = f m + f n) : ¬ Function.Injective f := 
sorry

end no_injective_function_l70_70570


namespace problem_statement_l70_70560

theorem problem_statement (x y : ℤ) (h1 : x = 8) (h2 : y = 3) :
  (x - 2 * y) * (x + 2 * y) = 28 :=
by
  sorry

end problem_statement_l70_70560


namespace pie_charts_cannot_show_changes_l70_70014

def pie_chart_shows_part_whole (P : Type) := true
def bar_chart_shows_amount (B : Type) := true
def line_chart_shows_amount_and_changes (L : Type) := true

theorem pie_charts_cannot_show_changes (P B L : Type) :
  pie_chart_shows_part_whole P ∧ bar_chart_shows_amount B ∧ line_chart_shows_amount_and_changes L →
  ¬ (pie_chart_shows_part_whole P ∧ ¬ line_chart_shows_amount_and_changes P) :=
by sorry

end pie_charts_cannot_show_changes_l70_70014


namespace race_course_length_to_finish_at_same_time_l70_70413

variable (v : ℝ) -- speed of B
variable (d : ℝ) -- length of the race course

-- A's speed is 4 times B's speed and A gives B a 75-meter head start.
theorem race_course_length_to_finish_at_same_time (h1 : v > 0) (h2 : d > 75) : 
  (1 : ℝ) / 4 * (d / v) = ((d - 75) / v) ↔ d = 100 := 
sorry

end race_course_length_to_finish_at_same_time_l70_70413


namespace shoe_price_calculation_l70_70166

theorem shoe_price_calculation :
  let initialPrice : ℕ := 50
  let increasedPrice : ℕ := 60  -- initialPrice * 1.2
  let discountAmount : ℕ := 9    -- increasedPrice * 0.15
  increasedPrice - discountAmount = 51 := 
by
  sorry

end shoe_price_calculation_l70_70166


namespace find_sinα_and_tanα_l70_70135

open Real 

noncomputable def vectors (α : ℝ) := (Real.cos α, 1)

noncomputable def vectors_perpendicular (α : ℝ) := (Real.sin α, -2)

theorem find_sinα_and_tanα (α: ℝ) (hα: π < α ∧ α < 3 * π / 2)
  (h_perp: vectors_perpendicular α = (Real.sin α, -2) ∧ vectors α = (Real.cos α, 1) ∧ (vectors α).1 * (vectors_perpendicular α).1 + (vectors α).2 * (vectors_perpendicular α).2 = 0):
  (Real.sin α = - (2 * Real.sqrt 5) / 5) ∧ 
  (Real.tan (α + π / 4) = -3) := 
sorry 

end find_sinα_and_tanα_l70_70135


namespace weights_problem_l70_70387

theorem weights_problem (n : ℕ) (x : ℝ) (h_avg : ∀ (i : ℕ), i < n → ∃ (w : ℝ), w = x) 
  (h_heaviest : ∃ (w_max : ℝ), w_max = 5 * x) : n > 5 :=
by
  sorry

end weights_problem_l70_70387


namespace optimalSelectionUsesGoldenRatio_l70_70336

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l70_70336


namespace sugar_needed_for_third_layer_l70_70603

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end sugar_needed_for_third_layer_l70_70603


namespace morgan_hula_hooping_time_l70_70288

-- Definitions based on conditions
def nancy_can_hula_hoop : ℕ := 10
def casey_can_hula_hoop : ℕ := nancy_can_hula_hoop - 3
def morgan_can_hula_hoop : ℕ := 3 * casey_can_hula_hoop

-- Theorem statement to show the solution is correct
theorem morgan_hula_hooping_time : morgan_can_hula_hoop = 21 :=
by
  sorry

end morgan_hula_hooping_time_l70_70288


namespace perpendicular_lines_l70_70263

theorem perpendicular_lines (a : ℝ) : 
  (2 * (a + 1) * a + a * 2 = 0) ↔ (a = -2 ∨ a = 0) :=
by 
  sorry

end perpendicular_lines_l70_70263


namespace other_root_of_quadratic_l70_70245

theorem other_root_of_quadratic (a b : ℝ) (h : (1:ℝ) = 1) (h_root : (1:ℝ) ^ 2 + a * (1:ℝ) + 2 = 0): b = 2 :=
by
  sorry

end other_root_of_quadratic_l70_70245


namespace min_number_of_girls_l70_70369

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l70_70369


namespace math_problem_l70_70464

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l70_70464


namespace arithmetic_sequence_general_term_l70_70990

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 4) 
  (h2 : a 4 + a 7 = 15) : 
  ∃ d : ℝ, ∀ n : ℕ, a n = n + 2 := 
by
  sorry

end arithmetic_sequence_general_term_l70_70990


namespace blake_total_expenditure_l70_70922

noncomputable def total_cost (rooms : ℕ) (primer_cost : ℝ) (paint_cost : ℝ) (primer_discount : ℝ) : ℝ :=
  let primer_needed := rooms
  let paint_needed := rooms
  let discounted_primer_cost := primer_cost * (1 - primer_discount)
  let total_primer_cost := primer_needed * discounted_primer_cost
  let total_paint_cost := paint_needed * paint_cost
  total_primer_cost + total_paint_cost

theorem blake_total_expenditure :
  total_cost 5 30 25 0.20 = 245 := 
by
  sorry

end blake_total_expenditure_l70_70922


namespace systematic_sampling_interval_l70_70553

-- Define the total number of students and sample size
def N : ℕ := 1200
def n : ℕ := 40

-- Define the interval calculation for systematic sampling
def k : ℕ := N / n

-- Prove that the interval k is 30
theorem systematic_sampling_interval : k = 30 := by
sorry

end systematic_sampling_interval_l70_70553


namespace gross_profit_value_l70_70901

theorem gross_profit_value
  (sales_price : ℝ)
  (gross_profit_percentage : ℝ)
  (sales_price_eq : sales_price = 91)
  (gross_profit_percentage_eq : gross_profit_percentage = 1.6)
  (C : ℝ)
  (cost_eqn : sales_price = C + gross_profit_percentage * C) :
  gross_profit_percentage * C = 56 :=
by
  sorry

end gross_profit_value_l70_70901


namespace find_deepaks_age_l70_70548

variable (R D : ℕ)

theorem find_deepaks_age
  (h1 : R / D = 4 / 3)
  (h2 : R + 2 = 26) :
  D = 18 := by
  sorry

end find_deepaks_age_l70_70548


namespace infinite_coprime_pairs_l70_70715

theorem infinite_coprime_pairs (m : ℤ) : ∃ infinitely_many (x y : ℤ), Int.gcd x y = 1 ∧ y ∣ (x^2 + m) ∧ x ∣ (y^2 + m) :=
sorry

end infinite_coprime_pairs_l70_70715


namespace number_of_boxes_sold_on_saturday_l70_70294

theorem number_of_boxes_sold_on_saturday (S : ℝ) 
  (h : S + 1.5 * S + 1.95 * S + 2.34 * S + 2.574 * S = 720) : 
  S = 77 := 
sorry

end number_of_boxes_sold_on_saturday_l70_70294


namespace find_b_l70_70241

theorem find_b (b : ℤ) :
  ∃ (r₁ r₂ : ℤ), (r₁ = -9) ∧ (r₁ * r₂ = 36) ∧ (r₁ + r₂ = -b) → b = 13 :=
by {
  sorry
}

end find_b_l70_70241


namespace min_girls_in_class_l70_70374

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l70_70374


namespace part1_part2_l70_70491

open Real

noncomputable def part1_statement (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 2 * m * x - 1 < 0

noncomputable def part2_statement (x : ℝ) : Prop := 
  ∀ (m : ℝ), |m| ≤ 1 → (m * x^2 - 2 * m * x - 1 < 0)

theorem part1 : part1_statement m ↔ (-1 < m ∧ m ≤ 0) :=
sorry

theorem part2 : part2_statement x ↔ ((1 - sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + sqrt 2)) :=
sorry

end part1_part2_l70_70491


namespace infinite_pseudoprimes_l70_70450

-- Definition of a pseudoprime to base a
def isPseudoprime (a n : ℕ) : Prop := 
  ¬ n.prime ∧ a^(n-1) ≡ 1 [MOD n]

-- The statement to prove there are infinitely many pseudoprimes to base 2
theorem infinite_pseudoprimes : ∃ᶠ n in at_top, isPseudoprime 2 n :=
sorry

end infinite_pseudoprimes_l70_70450


namespace optimal_selection_golden_ratio_l70_70309

def optimal_selection_method_uses_golden_ratio : Prop :=
  ∃ (concept : Type), 
    (concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median")
     ∧ concept = "Golden ratio"

theorem optimal_selection_golden_ratio :
  optimal_selection_method_uses_golden_ratio :=
by
  sorry

end optimal_selection_golden_ratio_l70_70309


namespace LawOfCosines_triangle_l70_70984

theorem LawOfCosines_triangle {a b C : ℝ} (ha : a = 9) (hb : b = 2 * Real.sqrt 3) (hC : C = Real.pi / 6 * 5) :
  ∃ c, c = 2 * Real.sqrt 30 :=
by
  sorry

end LawOfCosines_triangle_l70_70984


namespace three_inequalities_true_l70_70289

variables {x y a b : ℝ}
-- Declare the conditions as hypotheses
axiom h₁ : 0 < x
axiom h₂ : 0 < y
axiom h₃ : 0 < a
axiom h₄ : 0 < b
axiom hx : x^2 < a^2
axiom hy : y^2 < b^2

theorem three_inequalities_true : 
  (x^2 + y^2 < a^2 + b^2) ∧ 
  (x^2 * y^2 < a^2 * b^2) ∧ 
  (x^2 / y^2 < a^2 / b^2) :=
sorry

end three_inequalities_true_l70_70289


namespace cost_of_dvd_player_l70_70190

/-- The ratio of the cost of a DVD player to the cost of a movie is 9:2.
    A DVD player costs $63 more than a movie.
    Prove that the cost of the DVD player is $81. -/
theorem cost_of_dvd_player 
(D M : ℝ)
(h1 : D = (9 / 2) * M)
(h2 : D = M + 63) : 
D = 81 := 
sorry

end cost_of_dvd_player_l70_70190


namespace optimal_selection_method_uses_golden_ratio_l70_70354

def Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio: Prop :=
  HuaLuogengUsesOptimalSelectionMethod ∧ 
  ExistsAnswerSuchThatGoldenRatioIsUsed

theorem optimal_selection_method_uses_golden_ratio :
  Hua_Luogeng_Optimal_Selection_Uses_Golden_Ratio → (HuaLuogengUsesOptimalSelectionMethod → GoldenRatioIsUsed) :=
by
  intros h h1
  cases h with _ h2
  exact h2

end optimal_selection_method_uses_golden_ratio_l70_70354


namespace find_a_l70_70127

theorem find_a (a : ℝ) : (∃ p : ℝ × ℝ, p = (2 - a, a - 3) ∧ p.fst = 0) → a = 2 := by
  sorry

end find_a_l70_70127


namespace students_arrangement_count_l70_70867

theorem students_arrangement_count : 
  let total_permutations := Nat.factorial 5
  let a_first_permutations := Nat.factorial 4
  let b_last_permutations := Nat.factorial 4
  let both_permutations := Nat.factorial 3
  total_permutations - a_first_permutations - b_last_permutations + both_permutations = 78 :=
by
  sorry

end students_arrangement_count_l70_70867


namespace product_of_abc_l70_70191

-- Define the constants and conditions
variables (a b c m : ℝ)
axiom h1 : a + b + c = 180
axiom h2 : 5 * a = m
axiom h3 : b = m + 12
axiom h4 : c = m - 6

-- Prove that the product of a, b, and c is 42184
theorem product_of_abc : a * b * c = 42184 :=
by {
  sorry
}

end product_of_abc_l70_70191


namespace pairs_symmetry_l70_70173

theorem pairs_symmetry (N : ℕ) (hN : N > 2) :
  ∃ f : {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 > 2} ≃ 
           {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 < 2}, 
  true :=
sorry

end pairs_symmetry_l70_70173


namespace liza_final_balance_l70_70011

theorem liza_final_balance :
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries (balance : ℝ) := 0.2 * balance
  let friday_interest (balance : ℝ) := 0.02 * balance
  let saturday_phone_bill := 70
  let saturday_additional_deposit := 300
  let tuesday_balance := monday_balance - tuesday_rent
  let wednesday_balance := tuesday_balance + wednesday_deposit
  let thursday_balance_before_groceries := wednesday_balance - thursday_electric_bill - thursday_internet_bill
  let thursday_balance_after_groceries := thursday_balance_before_groceries - thursday_groceries thursday_balance_before_groceries
  let friday_balance := thursday_balance_after_groceries + friday_interest thursday_balance_after_groceries
  let saturday_balance_after_phone := friday_balance - saturday_phone_bill
  let final_balance := saturday_balance_after_phone + saturday_additional_deposit
  final_balance = 1562.528 :=
by
  let monday_balance := 800
  let tuesday_rent := 450
  let wednesday_deposit := 1500
  let thursday_electric_bill := 117
  let thursday_internet_bill := 100
  let thursday_groceries := 0.2 * (800 - 450 + 1500 - 117 - 100)
  let friday_interest := 0.02 * (800 - 450 + 1500 - 117 - 100 - 0.2 * (800 - 450 + 1500 - 117 - 100))
  let final_balance := 800 - 450 + 1500 - 117 - 100 - thursday_groceries + friday_interest - 70 + 300
  sorry

end liza_final_balance_l70_70011


namespace find_four_digit_number_l70_70937

theorem find_four_digit_number :
  ∃ A B C D : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
    (1001 * A + 100 * B + 10 * C + A) = 182 * (10 * C + D) ∧
    (1000 * A + 100 * B + 10 * C + D) = 2916 :=
by 
  sorry

end find_four_digit_number_l70_70937


namespace weaving_sum_first_seven_days_l70_70763

noncomputable def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

theorem weaving_sum_first_seven_days
  (a_1 d : ℕ) :
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) = 9 →
  (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 6) = 15 →
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) +
  (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 5) +
  (arithmetic_sequence a_1 d 6) + (arithmetic_sequence a_1 d 7) = 35 := by
  sorry

end weaving_sum_first_seven_days_l70_70763


namespace inequality_amgm_l70_70723

theorem inequality_amgm (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : a^3 + b^3 + a + b ≥ 4 * a * b :=
sorry

end inequality_amgm_l70_70723


namespace necessary_not_sufficient_l70_70632

noncomputable def perpendicular_condition (α β : Set Point) (m : Set Point) : Prop := 
  (∀ pt ∈ m, pt ∈ α) ∧                     -- m is in α
  (∃ n : Set Point, Line n ∧ ∀ pt ∈ n, pt ∉ α ∧ pt ∉ β) ∧ -- α and β are distinct planes
  (∀ pt ∈ m, pt ⊥ β ↔ α ⊥ β)               -- proving necessary but not sufficient condition

theorem necessary_not_sufficient (α β : Set Point) (m : Set Point) (h1 : ∀ pt ∈ m, pt ∈ α)
  (h2: ∃ n : Set Point, Line n ∧ ∀ pt ∈ n, pt ∉ α ∧ pt ∉ β) : 
  ∀ pt ∈ m, pt ⊥ β ↔ α ⊥ β :=
begin
  sorry
end

end necessary_not_sufficient_l70_70632


namespace cone_height_ratio_l70_70217

theorem cone_height_ratio (circumference : ℝ) (orig_height : ℝ) (short_volume : ℝ)
  (h_circumference : circumference = 20 * Real.pi)
  (h_orig_height : orig_height = 40)
  (h_short_volume : short_volume = 400 * Real.pi) :
  let r := circumference / (2 * Real.pi)
  let h_short := (3 * short_volume) / (Real.pi * r^2)
  (h_short / orig_height) = 3 / 10 :=
by {
  sorry
}

end cone_height_ratio_l70_70217


namespace max_2a_b_2c_l70_70814

theorem max_2a_b_2c (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 2 * a + b + 2 * c ≤ 3 :=
sorry

end max_2a_b_2c_l70_70814


namespace find_A_in_phone_number_l70_70427

theorem find_A_in_phone_number
  (A B C D E F G H I J : ℕ)
  (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧ 
            B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧ 
            C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧ 
            D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧ 
            E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧ 
            F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧ 
            G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
            H ≠ I ∧ H ≠ J ∧
            I ≠ J)
  (h_dec_ABC : A > B ∧ B > C)
  (h_dec_DEF : D > E ∧ E > F)
  (h_dec_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consec_even_DEF : D % 2 = 0 ∧ E % 2 = 0 ∧ F % 2 = 0 ∧ E = D - 2 ∧ F = E - 2)
  (h_consec_odd_GHIJ : G % 2 = 1 ∧ H % 2 = 1 ∧ I % 2 = 1 ∧ J % 2 = 1 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) :
  A = 8 :=
sorry

end find_A_in_phone_number_l70_70427


namespace candy_total_cost_l70_70785

theorem candy_total_cost
    (grape_candies cherry_candies apple_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 3 * cherry_candies)
    (h2 : apple_candies = 2 * grape_candies)
    (h3 : cost_per_candy = 2.50)
    (h4 : grape_candies = 24) :
    (grape_candies + cherry_candies + apple_candies) * cost_per_candy = 200 := 
by
  sorry

end candy_total_cost_l70_70785


namespace number_of_correct_statements_l70_70359

def condition1 : Prop := ∀ q : ℚ, q > 0 ∨ q < 0  -- This omits zero, hence incorrect.
def condition2 : Prop := ∀ a : ℝ, |a| = -a → a < 0  -- This doesn't consider the case of a = 0.
def poly := 2 * x^3 - 3 * x * y + 3 * y
def condition3 : Prop := ∃ (c : ℝ), polynomial.coeff poly 2 = c  -- There is no x^2 term here.
def condition4 : Prop := ∀ q : ℚ, ∃ r : ℝ, r = q  -- All rational numbers can be represented on the number line.
def pentagonal_prism := (7, 10, 15)  -- number of faces, vertices, edges
def condition5 : Prop := pentagonal_prism = (7, 10, 15)  -- This is correct by definition.

theorem number_of_correct_statements : (if condition4 then 1 else 0) + (if condition5 then 1 else 0) = 2 := by sorry

end number_of_correct_statements_l70_70359


namespace smallest_units_C_union_D_l70_70295

-- Definitions for the sets C and D and their sizes
def C_units : ℝ := 25.5
def D_units : ℝ := 18.0

-- Definition stating the inclusion-exclusion principle for sets C and D
def C_union_D (C_units D_units C_intersection_units : ℝ) : ℝ :=
  C_units + D_units - C_intersection_units

-- Statement to prove the minimum units in C union D
theorem smallest_units_C_union_D : ∃ h, h ≤ C_union_D C_units D_units D_units ∧ h = 25.5 := by
  sorry

end smallest_units_C_union_D_l70_70295


namespace count_non_perfect_square_or_cube_l70_70658

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l70_70658


namespace numbers_not_squares_nor_cubes_1_to_200_l70_70650

theorem numbers_not_squares_nor_cubes_1_to_200 : 
  let num_squares := nat.sqrt 200
  let num_cubes := nat.cbrt 200
  let num_sixth_powers := nat.root 6 200
  let total := num_squares + num_cubes - num_sixth_powers
  200 - total = 182 := 
by
  sorry

end numbers_not_squares_nor_cubes_1_to_200_l70_70650


namespace quadratic_no_real_roots_m_l70_70980

-- Define the quadratic equation and the condition for no real roots
def quadratic_no_real_roots (m : ℝ) : Prop :=
  let a := m - 1
  let b := 2
  let c := -2
  let Δ := b^2 - 4 * a * c
  Δ < 0

-- The final theorem statement that we need to prove
theorem quadratic_no_real_roots_m (m : ℝ) : quadratic_no_real_roots m → m < 1/2 :=
sorry

end quadratic_no_real_roots_m_l70_70980


namespace optimal_selection_uses_golden_ratio_l70_70357

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l70_70357


namespace smallest_factor_l70_70681

theorem smallest_factor (x : ℕ) (h1 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : ∃ (x : ℕ), (936 * x) % 2^5 = 0 ∧ (936 * x) % 3^3 = 0 ∧ (936 * x) % 13^2 = 0) : x = 468 := 
sorry

end smallest_factor_l70_70681


namespace other_store_pools_l70_70175

variable (P A : ℕ)
variable (three_times : P = 3 * A)
variable (total_pools : P + A = 800)

theorem other_store_pools (three_times : P = 3 * A) (total_pools : P + A = 800) : A = 266 := 
by
  sorry

end other_store_pools_l70_70175


namespace combined_weight_l70_70684

theorem combined_weight (S R : ℝ) (h1 : S - 5 = 2 * R) (h2 : S = 75) : S + R = 110 :=
sorry

end combined_weight_l70_70684


namespace ensure_mixed_tablets_l70_70214

theorem ensure_mixed_tablets (A B : ℕ) (total : ℕ) (hA : A = 10) (hB : B = 16) (htotal : total = 18) :
  ∃ (a b : ℕ), a + b = total ∧ a ≤ A ∧ b ≤ B ∧ a > 0 ∧ b > 0 :=
by
  sorry

end ensure_mixed_tablets_l70_70214


namespace find_x_l70_70623

theorem find_x (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x = (1 - k) / 2) ↔ ((x^2 - 3 * x - 4) / (x - 4) = 3 * x + k) :=
by sorry

end find_x_l70_70623


namespace sets_equivalence_l70_70092

theorem sets_equivalence :
  (∀ M N, (M = {(3, 2)} ∧ N = {(2, 3)} → M ≠ N) ∧
          (M = {4, 5} ∧ N = {5, 4} → M = N) ∧
          (M = {1, 2} ∧ N = {(1, 2)} → M ≠ N) ∧
          (M = {(x, y) | x + y = 1} ∧ N = {y | ∃ x, x + y = 1} → M ≠ N)) :=
by sorry

end sets_equivalence_l70_70092


namespace container_holds_slices_l70_70583

theorem container_holds_slices (x : ℕ) 
  (h1 : x > 1) 
  (h2 : x ≠ 332) 
  (h3 : x ≠ 166) 
  (h4 : x ∣ 332) :
  x = 83 := 
sorry

end container_holds_slices_l70_70583


namespace vanessa_weeks_to_wait_l70_70393

theorem vanessa_weeks_to_wait
  (dress_cost savings : ℕ)
  (weekly_allowance weekly_expense : ℕ)
  (h₀ : dress_cost = 80)
  (h₁ : savings = 20)
  (h₂ : weekly_allowance = 30)
  (h₃ : weekly_expense = 10) :
  let net_savings_per_week := weekly_allowance - weekly_expense,
      additional_amount_needed := dress_cost - savings in
  additional_amount_needed / net_savings_per_week = 3 :=
by
  sorry

end vanessa_weeks_to_wait_l70_70393


namespace certain_number_l70_70200

theorem certain_number (x : ℤ) (h : 12 + x = 27) : x = 15 :=
by
  sorry

end certain_number_l70_70200


namespace probability_root_condition_l70_70854

theorem probability_root_condition :
  let roots := {z : ℂ | z^2023 = 1}
  let distinct_roots := { v ∈ roots | ∃ w ∈ roots, v ≠ w }
  (probability (v, w) ∈ distinct_roots, sqrt 2 + sqrt 5 ≤ abs (v + w)) = 675 / 2022 := 
sorry

end probability_root_condition_l70_70854


namespace circle_diameter_from_area_l70_70058

theorem circle_diameter_from_area (A : ℝ) (hA : A = 400 * Real.pi) :
    ∃ D : ℝ, D = 40 := 
by
  -- Consider the formula for the area of a circle with radius r.
  -- The area is given as A = π * r^2.
  let r := Real.sqrt 400 -- Solve for radius r.
  have hr : r = 20 := by sorry
  -- The diameter D is twice the radius.
  let D := 2 * r 
  existsi D
  have hD : D = 40 := by sorry
  exact hD

end circle_diameter_from_area_l70_70058


namespace part_one_costs_part_two_feasible_values_part_three_min_cost_l70_70054

noncomputable def cost_of_stationery (a b : ℕ) (cost_A_and_B₁ : 2 * a + b = 35) (cost_A_and_B₂ : a + 3 * b = 30): ℕ × ℕ :=
(a, b)

theorem part_one_costs (a b : ℕ) (h₁ : 2 * a + b = 35) (h₂ : a + 3 * b = 30): cost_of_stationery a b h₁ h₂ = (15, 5) :=
sorry

theorem part_two_feasible_values (x : ℕ) (h₁ : x + (120 - x) = 120) (h₂ : 975 ≤ 15 * x + 5 * (120 - x)) (h₃ : 15 * x + 5 * (120 - x) ≤ 1000):
  x = 38 ∨ x = 39 ∨ x = 40 :=
sorry

theorem part_three_min_cost (x : ℕ) (h₁ : x = 38 ∨ x = 39 ∨ x = 40):
  ∃ min_cost, (min_cost = 10 * 38 + 600 ∧ min_cost ≤ 10 * x + 600) :=
sorry

end part_one_costs_part_two_feasible_values_part_three_min_cost_l70_70054


namespace optimal_selection_uses_golden_ratio_l70_70328

theorem optimal_selection_uses_golden_ratio
  (H : HuaLuogeng_popularized_method)
  (G : uses_golden_ratio H) :
  (optimal_selection_method_uses H = golden_ratio) :=
sorry

end optimal_selection_uses_golden_ratio_l70_70328


namespace range_of_x_l70_70121

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the relevant conditions
axiom decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) < 0
axiom symmetry : ∀ x : ℝ, f (1 - x) = -f (1 + x)
axiom f_one : f 1 = -1

-- Define the statement to be proved
theorem range_of_x : ∀ x : ℝ, -1 ≤ f (0.5 * x - 1) ∧ f (0.5 * x - 1) ≤ 1 → 0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l70_70121


namespace numbers_not_squares_or_cubes_in_200_l70_70669

/-- There are 182 numbers between 1 and 200 that are neither perfect squares nor perfect cubes. -/
theorem numbers_not_squares_or_cubes_in_200 : (finset.range 200).filter (λ n, ¬(∃ m, m^2 = n) ∧ ¬(∃ k, k^3 = n)).card = 182 :=
by 
  /-
    The proof will involve constructing the set of numbers from 1 to 200, filtering out those that
    are perfect squares or perfect cubes, and counting the remaining numbers.
  -/
  sorry

end numbers_not_squares_or_cubes_in_200_l70_70669


namespace calculate_lego_set_cost_l70_70234

variable (total_revenue_after_tax : ℝ) (little_cars_base_price : ℝ)
  (discount_rate : ℝ) (tax_rate : ℝ) (num_little_cars : ℕ)
  (num_action_figures : ℕ) (num_board_games : ℕ)
  (lego_set_cost_before_tax : ℝ)

theorem calculate_lego_set_cost :
  total_revenue_after_tax = 136.50 →
  little_cars_base_price = 5 →
  discount_rate = 0.10 →
  tax_rate = 0.05 →
  num_little_cars = 3 →
  num_action_figures = 2 →
  num_board_games = 1 →
  lego_set_cost_before_tax = 85 :=
by
  sorry

end calculate_lego_set_cost_l70_70234


namespace distance_BC_400m_l70_70864

-- Define the hypotheses
variables
  (starting_from_same_time : Prop) -- Sam and Nik start from points A and B respectively at the same time
  (constant_speeds : Prop) -- They travel towards each other at constant speeds along the same route
  (meeting_point_C : Prop) -- They meet at point C, which is 600 m away from starting point A
  (speed_Sam : ℕ) (speed_Sam_value : speed_Sam = 50) -- The speed of Sam is 50 meters per minute
  (time_Sam : ℕ) (time_Sam_value : time_Sam = 20) -- It took Sam 20 minutes to cover the distance between A and B

-- Define the statement to be proven
theorem distance_BC_400m
  (d_AB : ℕ) (d_AB_value : d_AB = speed_Sam * time_Sam)
  (d_AC : ℕ) (d_AC_value : d_AC = 600)
  (d_BC : ℕ) (d_BC_value : d_BC = d_AB - d_AC) :
  d_BC = 400 := by
  sorry

end distance_BC_400m_l70_70864


namespace inequality_a_cube_less_b_cube_l70_70975

theorem inequality_a_cube_less_b_cube (a b : ℝ) (ha : a < 0) (hb : b > 0) : a^3 < b^3 :=
by
  sorry

end inequality_a_cube_less_b_cube_l70_70975


namespace circle_area_eq_25pi_l70_70102

theorem circle_area_eq_25pi :
  (∃ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y - 12 = 0) →
  (∃ (area : ℝ), area = 25 * Real.pi) :=
by
  sorry

end circle_area_eq_25pi_l70_70102


namespace M_greater_than_N_l70_70527

-- Definitions based on the problem's conditions
def M (x : ℝ) : ℝ := (x - 3) * (x - 7)
def N (x : ℝ) : ℝ := (x - 2) * (x - 8)

-- Statement to prove
theorem M_greater_than_N (x : ℝ) : M x > N x := by
  -- Proof is omitted
  sorry

end M_greater_than_N_l70_70527


namespace investment_years_l70_70025

def principal (P : ℝ) := P = 1200
def rate (r : ℝ) := r = 0.10
def interest_diff (P r : ℝ) (t : ℝ) :=
  let SI := P * r * t
  let CI := P * (1 + r)^t - P
  CI - SI = 12

theorem investment_years (P r : ℝ) (t : ℝ) 
  (h_principal : principal P) 
  (h_rate : rate r) 
  (h_diff : interest_diff P r t) : 
  t = 2 := 
sorry

end investment_years_l70_70025


namespace monotonic_increasing_implies_range_a_l70_70979

-- Definition of the function f(x) = ax^3 - x^2 + x - 5
def f (a x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

-- Derivative of f(x) with respect to x
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

-- The statement that proves the monotonicity condition implies the range for a
theorem monotonic_increasing_implies_range_a (a : ℝ) : 
  ( ∀ x, f_prime a x ≥ 0 ) → a ≥ (1:ℝ) / 3 := by
  sorry

end monotonic_increasing_implies_range_a_l70_70979


namespace Ron_four_times_Maurice_l70_70986

theorem Ron_four_times_Maurice
  (r m : ℕ) (x : ℕ) 
  (h_r : r = 43) 
  (h_m : m = 7) 
  (h_eq : r + x = 4 * (m + x)) : 
  x = 5 := 
by
  sorry

end Ron_four_times_Maurice_l70_70986


namespace mark_eggs_supply_l70_70281

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end mark_eggs_supply_l70_70281


namespace interval_monotonically_increasing_range_g_l70_70248

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sqrt 3) * Real.sin (x + (Real.pi / 4)) * Real.cos (x + (Real.pi / 4)) + Real.sin (2 * x) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + (2 * Real.pi / 3)) - 1

theorem interval_monotonically_increasing :
  ∃ (k : ℤ), ∀ (x : ℝ), (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) → 0 ≤ deriv f x :=
sorry

theorem range_g (m : ℝ) : 
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → g x = m ↔ -3 ≤ m ∧ m ≤ Real.sqrt 3 - 1 :=
sorry

end interval_monotonically_increasing_range_g_l70_70248


namespace simplify_sqrt_l70_70017

noncomputable def simplify_expression : ℝ :=
  Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)

theorem simplify_sqrt (h : simplify_expression = 2 * Real.sqrt 6) : 
    Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 :=
  by sorry

end simplify_sqrt_l70_70017


namespace lemons_needed_l70_70578

theorem lemons_needed (initial_lemons : ℝ) (initial_gallons : ℝ) 
  (reduced_ratio : ℝ) (first_gallons : ℝ) (total_gallons : ℝ) :
  initial_lemons / initial_gallons * first_gallons 
  + (initial_lemons / initial_gallons * reduced_ratio) * (total_gallons - first_gallons) = 56.25 :=
by 
  let initial_ratio := initial_lemons / initial_gallons
  let reduced_ratio_amount := initial_ratio * reduced_ratio 
  let lemons_first := initial_ratio * first_gallons
  let lemons_remaining := reduced_ratio_amount * (total_gallons - first_gallons)
  let total_lemons := lemons_first + lemons_remaining
  show total_lemons = 56.25
  sorry

end lemons_needed_l70_70578


namespace optimal_selection_method_is_golden_ratio_l70_70344

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l70_70344


namespace repeating_decimal_to_fraction_l70_70113

theorem repeating_decimal_to_fraction : (let a := (6 : Real) / 10 in
                                         let r := (1 : Real) / 10 in
                                         ∑' n : ℕ, a * r^n) = (2 : Real) / 3 :=
by
  sorry

end repeating_decimal_to_fraction_l70_70113


namespace basketball_tournament_l70_70018

theorem basketball_tournament (teams : Finset ℕ) (games_played : ℕ → ℕ → ℕ) (win_chance : ℕ → ℕ → Prop) 
(points : ℕ → ℕ) (X Y : ℕ) :
  teams.card = 6 → 
  (∀ t₁ t₂, t₁ ≠ t₂ → games_played t₁ t₂ = 1) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ ∨ win_chance t₂ t₁) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → win_chance t₁ t₂ → points t₁ = points t₁ + 1 ∧ points t₂ = points t₂) → 
  win_chance X Y →
  0.5 = 0.5 →
  0.5 * (1 - ((252 : ℚ) / 1024)) = (193 : ℚ) / 512 →
  ((63 : ℚ) / 256) + ((193 : ℚ) / 512) = (319 : ℚ) / 512 :=
by 
  sorry 

end basketball_tournament_l70_70018


namespace sqrt_two_between_one_and_two_l70_70192

theorem sqrt_two_between_one_and_two : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2 := 
by
  -- sorry placeholder
  sorry

end sqrt_two_between_one_and_two_l70_70192


namespace union_eq_set_l70_70161

noncomputable def M : Set ℤ := {x | |x| < 2}
noncomputable def N : Set ℤ := {-2, -1, 0}

theorem union_eq_set : M ∪ N = {-2, -1, 0, 1} := by
  sorry

end union_eq_set_l70_70161


namespace determine_m_even_function_l70_70838

theorem determine_m_even_function (m : ℤ) :
  (∀ x : ℤ, (x^2 + (m-1)*x) = (x^2 - (m-1)*x)) → m = 1 :=
by
    sorry

end determine_m_even_function_l70_70838


namespace inequality_solution_l70_70726

theorem inequality_solution (x : ℝ) :
  (x+3)/(x+4) > (4*x+5)/(3*x+10) ↔ x ∈ Set.Ioo (-4 : ℝ) (- (10 : ℝ) / 3) ∪ Set.Ioi 2 :=
by
  sorry

end inequality_solution_l70_70726


namespace Olga_paints_zero_boards_l70_70710

variable (t p q t' : ℝ)
variable (rv ro : ℝ)

-- Conditions
axiom Valera_solo_trip : 2 * t + p = 2
axiom Valera_and_Olga_painting_time : 2 * t' + q = 3
axiom Valera_painting_rate : rv = 11 / p
axiom Valera_Omega_painting_rate : rv * q + ro * q = 9
axiom Valera_walk_faster : t' > t

-- Question: How many boards will Olga be able to paint alone if she needs to return home 1 hour after leaving?
theorem Olga_paints_zero_boards :
  t' > 1 → 0 = 0 := 
by 
  sorry

end Olga_paints_zero_boards_l70_70710


namespace solve_rebus_l70_70951

-- Definitions for the conditions
def is_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

def distinct_digits (A B C D : Nat) : Prop := 
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Main Statement
theorem solve_rebus (A B C D : Nat) (h_distinct : distinct_digits A B C D) 
(h_eq : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
by
  sorry

end solve_rebus_l70_70951


namespace count_valid_numbers_between_1_and_200_l70_70646

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l70_70646


namespace eval_infinite_series_eq_4_l70_70230

open BigOperators

noncomputable def infinite_series_sum : ℝ :=
  ∑' k, (k^2) / (3^k)

theorem eval_infinite_series_eq_4 : infinite_series_sum = 4 := 
  sorry

end eval_infinite_series_eq_4_l70_70230


namespace boat_distance_downstream_l70_70988

-- Let v_s be the speed of the stream in km/h
-- Condition 1: In one hour, a boat goes 5 km against the stream.
-- Condition 2: The speed of the boat in still water is 8 km/h.

theorem boat_distance_downstream (v_s : ℝ) :
  (8 - v_s = 5) →
  (distance : ℝ) →
  8 + v_s = distance →
  distance = 11 := by
  sorry

end boat_distance_downstream_l70_70988


namespace optimal_selection_method_uses_golden_ratio_l70_70313

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l70_70313


namespace merchant_profit_l70_70760

noncomputable theory

def profit_percentage (CP : ℝ) (markup_rate : ℝ) (discount_rate : ℝ) : ℝ :=
let marked_price := CP * (1 + markup_rate) in
let selling_price := marked_price * (1 - discount_rate) in
((selling_price - CP) / CP) * 100

theorem merchant_profit :
  profit_percentage 100 0.4 0.2 = 12 := by sorry

end merchant_profit_l70_70760


namespace arithmetic_sequence_sum_l70_70691

theorem arithmetic_sequence_sum (a b d : ℕ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ)
  (h1 : a₁ + a₂ + a₃ = 39)
  (h2 : a₄ + a₅ + a₆ = 27)
  (h3 : a₄ = a₁ + 3 * d)
  (h4 : a₅ = a₂ + 3 * d)
  (h5 : a₆ = a₃ + 3 * d)
  (h6 : a₇ = a₄ + 3 * d)
  (h7 : a₈ = a₅ + 3 * d)
  (h8 : a₉ = a₆ + 3 * d) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 81 :=
sorry

end arithmetic_sequence_sum_l70_70691


namespace repeating_six_to_fraction_l70_70114

-- Define the infinite geometric series representation of 0.666...
def infinite_geometric_series (n : ℕ) : ℝ := 6 / (10 ^ n)

-- Define the sum of the infinite geometric series for 0.666...
def sum_infinite_geometric_series : ℝ :=
  ∑' n, infinite_geometric_series n

-- Formally state the problem to prove that 0.666... equals 2/3
theorem repeating_six_to_fraction : sum_infinite_geometric_series = 2 / 3 :=
by
  -- Proof goes here, but for now we use sorry to denote it will be completed later
  sorry

end repeating_six_to_fraction_l70_70114


namespace intersection_complement_l70_70766

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_complement :
  A ∩ ({x | x < -1 ∨ x > 3} : Set ℝ) = {x | 3 < x ∧ x < 4} :=
by
  sorry

end intersection_complement_l70_70766


namespace find_x_for_which_f_f_x_eq_f_x_l70_70001

noncomputable def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem find_x_for_which_f_f_x_eq_f_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end find_x_for_which_f_f_x_eq_f_x_l70_70001


namespace find_a_value_l70_70962

theorem find_a_value (a : ℝ) (f : ℝ → ℝ)
  (h_def : ∀ x, f x = (Real.exp (x - a) - 1) * Real.log (x + 2 * a - 1))
  (h_ge_0 : ∀ x, x > 1 - 2 * a → f x ≥ 0) : a = 2 / 3 :=
by
  -- Omitted proof
  sorry

end find_a_value_l70_70962


namespace perfect_square_trinomial_m_eq_6_or_neg6_l70_70981

theorem perfect_square_trinomial_m_eq_6_or_neg6
  (m : ℤ) :
  (∃ a : ℤ, x * x + m * x + 9 = (x + a) * (x + a)) → (m = 6 ∨ m = -6) :=
by
  sorry

end perfect_square_trinomial_m_eq_6_or_neg6_l70_70981


namespace cake_sugar_calculation_l70_70605

theorem cake_sugar_calculation (sugar_first_layer : ℕ) (sugar_second_layer : ℕ) (sugar_third_layer : ℕ) :
  sugar_first_layer = 2 →
  sugar_second_layer = 2 * sugar_first_layer →
  sugar_third_layer = 3 * sugar_second_layer →
  sugar_third_layer = 12 := 
by
  intros h1 h2 h3
  have h4 : 2 = sugar_first_layer, from h1.symm
  have h5 : sugar_second_layer = 2 * 2, by rw [h4, h2]
  have h6 : sugar_third_layer = 3 * 4, by rw [h5, h3]
  exact h6

end cake_sugar_calculation_l70_70605


namespace rebus_solution_l70_70943

theorem rebus_solution (A B C D : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (equation : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
sorry

end rebus_solution_l70_70943


namespace subcommittees_with_at_least_one_teacher_l70_70034

theorem subcommittees_with_at_least_one_teacher :
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  total_subcommittees - non_teacher_subcommittees = 460 :=
by
  -- Definitions and conditions based on the problem statement
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  sorry -- Proof goes here

end subcommittees_with_at_least_one_teacher_l70_70034


namespace initial_sum_l70_70776

theorem initial_sum (P : ℝ) (compound_interest : ℝ) (r1 r2 r3 r4 r5 : ℝ) 
  (h1 : r1 = 0.06) (h2 : r2 = 0.08) (h3 : r3 = 0.07) (h4 : r4 = 0.09) (h5 : r5 = 0.10)
  (interest_sum : compound_interest = 4016.25) :
  P = 4016.25 / ((1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5) - 1) :=
by
  sorry

end initial_sum_l70_70776


namespace profit_function_correct_l70_70414

-- Definitions based on Conditions
def selling_price {R : Type*} [LinearOrderedField R] : R := 45
def profit_max {R : Type*} [LinearOrderedField R] : R := 450
def price_no_sales {R : Type*} [LinearOrderedField R] : R := 60
def quadratic_profit {R : Type*} [LinearOrderedField R] (x : R) : R := -2 * (x - 30) * (x - 60)

-- The statement we need to prove.
theorem profit_function_correct {R : Type*} [LinearOrderedField R] :
  quadratic_profit (selling_price : R) = profit_max ∧ quadratic_profit (price_no_sales : R) = 0 := 
sorry

end profit_function_correct_l70_70414


namespace count_three_digit_numbers_increased_by_99_when_reversed_l70_70831

def countValidNumbers : Nat := 80

theorem count_three_digit_numbers_increased_by_99_when_reversed :
  ∃ (a b c : Nat), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧
   (100 * a + 10 * b + c + 99 = 100 * c + 10 * b + a) ∧
  (countValidNumbers = 80) :=
sorry

end count_three_digit_numbers_increased_by_99_when_reversed_l70_70831


namespace diesel_fuel_usage_l70_70706

theorem diesel_fuel_usage (weekly_spending : ℝ) (cost_per_gallon : ℝ) (weeks : ℝ) (result : ℝ): 
  weekly_spending = 36 → cost_per_gallon = 3 → weeks = 2 → result = 24 → 
  (weekly_spending / cost_per_gallon) * weeks = result :=
by
  intros
  sorry

end diesel_fuel_usage_l70_70706


namespace maximize_value_l70_70561

def f (x : ℝ) : ℝ := -3 * x^2 - 8 * x + 18

theorem maximize_value : ∀ x : ℝ, f x ≤ f (-4/3) :=
by sorry

end maximize_value_l70_70561


namespace optimal_selection_method_uses_golden_ratio_l70_70351

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l70_70351


namespace pictures_remaining_l70_70402

-- Define the initial number of pictures taken at the zoo and museum
def zoo_pictures : Nat := 50
def museum_pictures : Nat := 8
-- Define the number of pictures deleted
def deleted_pictures : Nat := 38

-- Define the total number of pictures taken initially and remaining after deletion
def total_pictures : Nat := zoo_pictures + museum_pictures
def remaining_pictures : Nat := total_pictures - deleted_pictures

theorem pictures_remaining : remaining_pictures = 20 := 
by 
  -- This theorem states that, given the conditions, the remaining pictures count must be 20
  sorry

end pictures_remaining_l70_70402


namespace proof_x_y_l70_70260

noncomputable def x_y_problem (x y : ℝ) : Prop :=
  (x^2 = 9) ∧ (|y| = 4) ∧ (x < y) → (x - y = -1 ∨ x - y = -7)

theorem proof_x_y (x y : ℝ) : x_y_problem x y :=
by
  sorry

end proof_x_y_l70_70260


namespace given_expression_equality_l70_70483

theorem given_expression_equality (x : ℝ) (A ω φ b : ℝ) (hA : 0 < A)
  (h : 2 * (Real.cos x)^2 + Real.sin (2 * x) = A * Real.sin (ω * x + φ) + b) :
  A = Real.sqrt 2 ∧ b = 1 :=
sorry

end given_expression_equality_l70_70483


namespace work_completion_time_l70_70412

theorem work_completion_time (A_works_in : ℕ) (A_works_days : ℕ) (B_works_remainder_in : ℕ) (total_days : ℕ) :
  (A_works_in = 60) → (A_works_days = 15) → (B_works_remainder_in = 30) → (total_days = 24) := 
by
  intros hA_work hA_days hB_work
  sorry

end work_completion_time_l70_70412


namespace paving_stones_needed_l70_70909

variables (length_courtyard width_courtyard num_paving_stones length_paving_stone area_courtyard area_paving_stone : ℝ)
noncomputable def width_paving_stone := 2

theorem paving_stones_needed : 
  length_courtyard = 60 → 
  width_courtyard = 14 → 
  num_paving_stones = 140 →
  length_paving_stone = 3 →
  area_courtyard = length_courtyard * width_courtyard →
  area_paving_stone = length_paving_stone * width_paving_stone →
  num_paving_stones = area_courtyard / area_paving_stone :=
by
  intros h_length_courtyard h_width_courtyard h_num_paving_stones h_length_paving_stone h_area_courtyard h_area_paving_stone
  rw [h_length_courtyard, h_width_courtyard, h_length_paving_stone] at *
  simp at *
  sorry

end paving_stones_needed_l70_70909


namespace ceiling_and_floor_calculation_l70_70458

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l70_70458


namespace pumpkin_pie_degrees_l70_70532

theorem pumpkin_pie_degrees (total_students : ℕ) (peach_pie : ℕ) (apple_pie : ℕ) (blueberry_pie : ℕ)
                               (pumpkin_pie : ℕ) (banana_pie : ℕ)
                               (h_total : total_students = 40)
                               (h_peach : peach_pie = 14)
                               (h_apple : apple_pie = 9)
                               (h_blueberry : blueberry_pie = 7)
                               (h_remaining : pumpkin_pie = banana_pie)
                               (h_half_remaining : 2 * pumpkin_pie = 40 - (peach_pie + apple_pie + blueberry_pie)) :
  (pumpkin_pie * 360) / total_students = 45 := by
sorry

end pumpkin_pie_degrees_l70_70532


namespace problem_proof_l70_70794

-- Define positive integers and the conditions given in the problem
variables {p q r s : ℕ}

-- The product of the four integers is 7!
axiom product_of_integers : p * q * r * s = 5040  -- 7! = 5040

-- The equations defining the relationships
axiom equation1 : p * q + p + q = 715
axiom equation2 : q * r + q + r = 209
axiom equation3 : r * s + r + s = 143

-- The goal is to prove p - s = 10
theorem problem_proof : p - s = 10 :=
sorry

end problem_proof_l70_70794


namespace total_selling_price_l70_70914

theorem total_selling_price
  (meters_cloth : ℕ)
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter)
  (total_selling_price : ℕ := selling_price_per_meter * meters_cloth)
  (h_mc : meters_cloth = 75)
  (h_ppm : profit_per_meter = 15)
  (h_cppm : cost_price_per_meter = 51)
  (h_spm : selling_price_per_meter = 66)
  (h_tsp : total_selling_price = 4950) : 
  total_selling_price = 4950 := 
  by
  -- Skipping the actual proof
  trivial

end total_selling_price_l70_70914


namespace necessary_but_not_sufficient_l70_70828

-- Define the function f(x)
def f (a x : ℝ) := |a - 3 * x|

-- Define the condition for the function to be monotonically increasing on [1, +∞)
def is_monotonically_increasing_on_interval (a : ℝ) : Prop :=
  ∀ (x y : ℝ), 1 ≤ x → x ≤ y → (f a x ≤ f a y)

-- Define the condition that a must be 3
def condition_a_eq_3 (a : ℝ) : Prop := (a = 3)

-- Prove that condition_a_eq_3 is a necessary but not sufficient condition
theorem necessary_but_not_sufficient (a : ℝ) :
  (is_monotonically_increasing_on_interval a) →
  condition_a_eq_3 a ↔ (∀ (b : ℝ), b ≠ a → is_monotonically_increasing_on_interval b → false) := 
sorry

end necessary_but_not_sufficient_l70_70828


namespace example_problem_l70_70680

def Z (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem example_problem :
  Z 4 3 = -11 := 
by
  -- proof goes here
  sorry

end example_problem_l70_70680


namespace even_function_value_for_negative_x_l70_70494

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_value_for_negative_x (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_pos : ∀ (x : ℝ), 0 < x → f x = 10^x) :
  ∀ x : ℝ, x < 0 → f x = 10^(-x) :=
by
  sorry

end even_function_value_for_negative_x_l70_70494


namespace optimal_selection_method_use_golden_ratio_l70_70346

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l70_70346


namespace matilda_percentage_loss_l70_70287

theorem matilda_percentage_loss (initial_cost selling_price : ℕ) (h_initial : initial_cost = 300) (h_selling : selling_price = 255) :
  ((initial_cost - selling_price) * 100) / initial_cost = 15 :=
by
  rw [h_initial, h_selling]
  -- Proceed with the proof
  sorry

end matilda_percentage_loss_l70_70287


namespace pythagorean_theorem_example_l70_70992

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 8
noncomputable def c : ℕ := 10

theorem pythagorean_theorem_example :
  c = Real.sqrt (a^2 + b^2) := 
by
  sorry

end pythagorean_theorem_example_l70_70992


namespace calculate_savings_l70_70438

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l70_70438


namespace chips_probability_l70_70907

/-- A bag contains 4 green, 3 orange, and 5 blue chips. If the 12 chips are randomly drawn from
    the bag, one at a time and without replacement, the probability that the chips are drawn such
    that the 4 green chips are drawn consecutively, the 3 orange chips are drawn consecutively,
    and the 5 blue chips are drawn consecutively, but not necessarily in the green-orange-blue
    order, is 1/4620. -/
theorem chips_probability :
  let total_chips := 12
  let factorial := Nat.factorial
  let favorable_outcomes := (factorial 3) * (factorial 4) * (factorial 3) * (factorial 5)
  let total_outcomes := factorial total_chips
  favorable_outcomes / total_outcomes = 1 / 4620 :=
by
  -- proof goes here, but we skip it
  sorry

end chips_probability_l70_70907


namespace chef_earns_less_than_manager_l70_70920

noncomputable def hourly_wage_manager : ℝ := 8.5
noncomputable def hourly_wage_dishwasher : ℝ := hourly_wage_manager / 2
noncomputable def hourly_wage_chef : ℝ := hourly_wage_dishwasher * 1.2
noncomputable def daily_bonus : ℝ := 5
noncomputable def overtime_multiplier : ℝ := 1.5
noncomputable def tax_rate : ℝ := 0.15

noncomputable def manager_hours : ℝ := 10
noncomputable def dishwasher_hours : ℝ := 6
noncomputable def chef_hours : ℝ := 12
noncomputable def standard_hours : ℝ := 8

noncomputable def compute_earnings (hourly_wage : ℝ) (hours_worked : ℝ) : ℝ :=
  let regular_hours := min standard_hours hours_worked
  let overtime_hours := max 0 (hours_worked - standard_hours)
  let regular_pay := regular_hours * hourly_wage
  let overtime_pay := overtime_hours * hourly_wage * overtime_multiplier
  let total_earnings_before_tax := regular_pay + overtime_pay + daily_bonus
  total_earnings_before_tax * (1 - tax_rate)

noncomputable def manager_earnings : ℝ := compute_earnings hourly_wage_manager manager_hours
noncomputable def dishwasher_earnings : ℝ := compute_earnings hourly_wage_dishwasher dishwasher_hours
noncomputable def chef_earnings : ℝ := compute_earnings hourly_wage_chef chef_hours

theorem chef_earns_less_than_manager : manager_earnings - chef_earnings = 18.78 := by
  sorry

end chef_earns_less_than_manager_l70_70920


namespace middle_and_oldest_son_ages_l70_70083

theorem middle_and_oldest_son_ages 
  (x y z : ℕ) 
  (father_age_current father_age_future : ℕ) 
  (youngest_age_increment : ℕ)
  (father_age_increment : ℕ) 
  (father_equals_sons_sum : father_age_future = (x + youngest_age_increment) + (y + father_age_increment) + (z + father_age_increment))
  (father_age_constraint : father_age_current + father_age_increment = father_age_future)
  (youngest_age_initial : x = 2)
  (father_age_current_value : father_age_current = 33)
  (youngest_age_increment_value : youngest_age_increment = 12)
  (father_age_increment_value : father_age_increment = 12) 
  :
  y = 3 ∧ z = 4 :=
begin
  sorry
end

end middle_and_oldest_son_ages_l70_70083


namespace smallest_integer_n_satisfying_inequality_l70_70479

theorem smallest_integer_n_satisfying_inequality 
  (x y z : ℝ) : 
  (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4) :=
sorry

end smallest_integer_n_satisfying_inequality_l70_70479


namespace candies_taken_away_per_incorrect_answer_eq_2_l70_70895

/-- Define constants and assumptions --/
def candy_per_correct := 3
def correct_answers := 7
def extra_correct_answers := 2
def total_candies_if_extra_correct := 31

/-- The number of candies taken away per incorrect answer --/
def x : ℤ := sorry

/-- Prove that the number of candies taken away for each incorrect answer is 2. --/
theorem candies_taken_away_per_incorrect_answer_eq_2 : 
  ∃ x : ℤ, ((correct_answers + extra_correct_answers) * candy_per_correct - total_candies_if_extra_correct = x + (extra_correct_answers * candy_per_correct - (total_candies_if_extra_correct - correct_answers * candy_per_correct))) ∧ x = 2 := 
by
  exists 2
  sorry

end candies_taken_away_per_incorrect_answer_eq_2_l70_70895


namespace smallest_whole_number_larger_than_triangle_perimeter_l70_70395

theorem smallest_whole_number_larger_than_triangle_perimeter
  (s : ℝ) (h1 : 5 + 19 > s) (h2 : 5 + s > 19) (h3 : 19 + s > 5) :
  ∃ P : ℝ, P = 5 + 19 + s ∧ P < 48 ∧ ∀ n : ℤ, n > P → n = 48 :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l70_70395


namespace cheese_cut_process_l70_70912

-- Definitions and conditions based on part (a)
def infinite_cut_possible (R : ℝ) : Prop :=
  R = 0.5 → ∀ (weights : list ℝ), ∃ (new_weights : list ℝ), 
  (∀ w ∈ new_weights, w > 0) ∧
  length new_weights > length weights ∧
  (∀ i j, i ≠ j → new_weights.get! i / new_weights.get! j > R ∨ new_weights.get! j / new_weights.get! i > R)

-- Definitions and conditions based on part (b)
def finite_cut_inevitable (R : ℝ) : Prop :=
  R > 0.5 → ∃ (N : ℕ), ∀ (current_size : ℕ) (weights : list ℝ), 
  current_size ≥ N → ∀ (new_weights : list ℝ), 
  (∀ w ∈ new_weights, w > 0) →
  length new_weights ≤ current_size

-- Definitions and conditions based on part (c)
def max_pieces (R : ℝ) (maxNo : ℕ) : Prop :=
  R = 0.6 → ∀ (weights : list ℝ), (∀ w ∈ weights, w > 0) →
  length weights ≤ maxNo ∧ ∀ i j, i ≠ j → weights.get! i / weights.get! j > R ∨ weights.get! j / weights.get! i > R

theorem cheese_cut_process :
  (infinite_cut_possible 0.5) ∧
  (finite_cut_inevitable 0.5) ∧
  (max_pieces 0.6 6) :=
by {
  sorry,
}

end cheese_cut_process_l70_70912


namespace find_GQ_in_triangle_XYZ_l70_70156

noncomputable def GQ_in_triangle_XYZ_centroid : ℝ :=
  let XY := 13
  let XZ := 15
  let YZ := 24
  let centroid_ratio := 1 / 3
  let semi_perimeter := (XY + XZ + YZ) / 2
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - XY) * (semi_perimeter - XZ) * (semi_perimeter - YZ))
  let heightXR := (2 * area) / YZ
  (heightXR * centroid_ratio)

theorem find_GQ_in_triangle_XYZ :
  GQ_in_triangle_XYZ_centroid = 2.4 :=
sorry

end find_GQ_in_triangle_XYZ_l70_70156


namespace problem_l70_70972

theorem problem (r : ℝ) (h : (r + 1/r)^4 = 17) : r^6 + 1/r^6 = 1 * Real.sqrt 17 - 6 :=
sorry

end problem_l70_70972


namespace percentage_increase_l70_70538

theorem percentage_increase (P Q R : ℝ) (x y : ℝ) 
  (h1 : P > 0) (h2 : Q > 0) (h3 : R > 0)
  (h4 : P = (1 + x / 100) * Q)
  (h5 : Q = (1 + y / 100) * R)
  (h6 : P = 2.4 * R) :
  x + y = 140 :=
sorry

end percentage_increase_l70_70538


namespace training_cost_per_month_correct_l70_70588

-- Define the conditions
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_duration : ℕ := 3
def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2 : ℕ := (45000 / 100) -- 1% of salary2 which is 450
def net_gain_diff : ℕ := 850

-- Define the monthly training cost for the first applicant
def monthly_training_cost : ℕ := 1786667 / 100

-- Prove that the monthly training cost for the first applicant is correct
theorem training_cost_per_month_correct :
  (revenue1 - (salary1 + 3 * monthly_training_cost) = revenue2 - (salary2 + bonus2) + net_gain_diff) :=
by
  sorry

end training_cost_per_month_correct_l70_70588


namespace alice_instructors_l70_70434

noncomputable def num_students : ℕ := 40
noncomputable def num_life_vests_Alice_has : ℕ := 20
noncomputable def percent_students_with_their_vests : ℕ := 20
noncomputable def num_additional_life_vests_needed : ℕ := 22

-- Constants based on calculated conditions
noncomputable def num_students_with_their_vests : ℕ := (percent_students_with_their_vests * num_students) / 100
noncomputable def num_students_without_their_vests : ℕ := num_students - num_students_with_their_vests
noncomputable def num_life_vests_needed_for_students : ℕ := num_students_without_their_vests - num_life_vests_Alice_has
noncomputable def num_life_vests_needed_for_instructors : ℕ := num_additional_life_vests_needed - num_life_vests_needed_for_students

theorem alice_instructors : num_life_vests_needed_for_instructors = 10 := 
by
  sorry

end alice_instructors_l70_70434


namespace find_four_digit_number_l70_70939

theorem find_four_digit_number :
  ∃ A B C D : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
    (1001 * A + 100 * B + 10 * C + A) = 182 * (10 * C + D) ∧
    (1000 * A + 100 * B + 10 * C + D) = 2916 :=
by 
  sorry

end find_four_digit_number_l70_70939


namespace stratified_sampling_correct_l70_70910

-- Defining the conditions
def total_students : ℕ := 900
def freshmen : ℕ := 300
def sophomores : ℕ := 200
def juniors : ℕ := 400
def sample_size : ℕ := 45

-- Defining the target sample numbers
def freshmen_sample : ℕ := 15
def sophomores_sample : ℕ := 10
def juniors_sample : ℕ := 20

-- The proof problem statement
theorem stratified_sampling_correct :
  freshmen_sample = (freshmen * sample_size / total_students) ∧
  sophomores_sample = (sophomores * sample_size / total_students) ∧
  juniors_sample = (juniors * sample_size / total_students) :=
by
  sorry

end stratified_sampling_correct_l70_70910


namespace probability_top_two_hearts_and_third_spade_l70_70444

open Classical

-- Definitions related to the standard deck of cards with the described conditions.
def deck : Finset (Fin 52) := Finset.univ

def heartsuits : Finset (Fin 52) := (Finset.range 13).image (λ n, n + 0)
def spadesuits : Finset (Fin 52) := (Finset.range 13).image (λ n, n + 39)

-- The main problem statement.
theorem probability_top_two_hearts_and_third_spade :
  (13 * 12 * 13 : ℚ) / (52 * 51 * 50) = 13 / 850 := by
  sorry

end probability_top_two_hearts_and_third_spade_l70_70444


namespace carol_first_roll_eight_is_49_over_169_l70_70917

noncomputable def probability_carol_first_roll_eight : ℚ :=
  let p_roll_eight := (1 : ℚ) / 8
  let p_not_roll_eight := (7 : ℚ) / 8
  let p_no_one_rolls_eight_first_cycle := p_not_roll_eight * p_not_roll_eight * p_not_roll_eight
  let p_carol_rolls_eight_first_cycle := p_not_roll_eight * p_not_roll_eight * p_roll_eight
  p_carol_rolls_eight_first_cycle / (1 - p_no_one_rolls_eight_first_cycle)

theorem carol_first_roll_eight_is_49_over_169 : probability_carol_first_roll_eight = (49 : ℚ) / 169 :=
by
  sorry

end carol_first_roll_eight_is_49_over_169_l70_70917


namespace no_three_distinct_integers_solving_polynomial_l70_70569

theorem no_three_distinct_integers_solving_polynomial (p : ℤ → ℤ) (hp : ∀ x, ∃ k : ℕ, p x = k • x + p 0) :
  ∀ a b c : ℤ, a ≠ b → b ≠ c → c ≠ a → p a = b → p b = c → p c = a → false :=
by
  intros a b c hab hbc hca hpa_hp pb_pc_pc
  sorry

end no_three_distinct_integers_solving_polynomial_l70_70569


namespace solve_for_x_l70_70679

theorem solve_for_x (x y : ℝ) (h₁ : x - y = 8) (h₂ : x + y = 16) (h₃ : x * y = 48) : x = 12 :=
sorry

end solve_for_x_l70_70679


namespace arithmetic_sequence_1000th_term_l70_70695

theorem arithmetic_sequence_1000th_term (a_1 : ℤ) (d : ℤ) (n : ℤ) (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 1000) : 
  a_1 + (n - 1) * d = 2998 := 
by
  sorry

end arithmetic_sequence_1000th_term_l70_70695


namespace fraction_simplification_l70_70060

theorem fraction_simplification (x : ℝ) (h : x = Real.sqrt 2) : 
  ( (x^2 - 1) / (x^2 - x) - 1) = Real.sqrt 2 / 2 :=
by 
  sorry

end fraction_simplification_l70_70060


namespace greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l70_70752

theorem greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30 :
  ∃ d, d ∣ 480 ∧ d < 60 ∧ d ∣ 90 ∧ (∀ e, e ∣ 480 → e < 60 → e ∣ 90 → e ≤ d) ∧ d = 30 :=
sorry

end greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l70_70752


namespace min_number_of_girls_l70_70385

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l70_70385


namespace negation_proposition_l70_70567

-- Define the proposition as a Lean function
def quadratic_non_negative (x : ℝ) : Prop := x^2 - 2*x + 1 ≥ 0

-- State the theorem that we need to prove
theorem negation_proposition : ∀ x : ℝ, quadratic_non_negative x :=
by 
  sorry

end negation_proposition_l70_70567


namespace fraction_problem_l70_70577

theorem fraction_problem (x : ℝ) (h : (3 / 4) * (1 / 2) * x * 5000 = 750.0000000000001) : 
  x = 0.4 :=
sorry

end fraction_problem_l70_70577


namespace function_properties_l70_70585

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (f (Real.pi / 3) = 1) ∧
  (∀ x y, -Real.pi / 6 ≤ x → x ≤ y → y ≤ Real.pi / 3 → f x ≤ f y) := by
  sorry

end function_properties_l70_70585


namespace side_length_of_cloth_l70_70898

namespace ClothProblem

def original_side_length (trimming_x_sides trimming_y_sides remaining_area : ℤ) :=
  let x : ℤ := 12
  x

theorem side_length_of_cloth (x_trim y_trim remaining_area : ℤ) (h_trim_x : x_trim = 4) 
                             (h_trim_y : y_trim = 3) (h_area : remaining_area = 120) :
  original_side_length x_trim y_trim remaining_area = 12 :=
by
  sorry

end ClothProblem

end side_length_of_cloth_l70_70898


namespace linear_condition_l70_70820

theorem linear_condition (a : ℝ) : a ≠ 0 ↔ ∃ (x y : ℝ), ax + y = -1 :=
by
  sorry

end linear_condition_l70_70820


namespace ratio_of_a_to_c_l70_70502

theorem ratio_of_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3) 
  (h3 : d / b = 1 / 5) : a / c = 75 / 16 := 
sorry

end ratio_of_a_to_c_l70_70502


namespace hua_luogeng_optimal_selection_l70_70318

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l70_70318


namespace find_y_l70_70255

theorem find_y (y : ℕ) 
  (h : (1/8) * 2^36 = 8^y) : y = 11 :=
sorry

end find_y_l70_70255


namespace intersection_of_A_and_B_l70_70165

-- Definitions representing the conditions
def setA : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := {x | x < 2}

-- Proof problem statement
theorem intersection_of_A_and_B : setA ∩ setB = {x | -1 < x ∧ x < 2} :=
sorry

end intersection_of_A_and_B_l70_70165


namespace tom_seashells_l70_70198

theorem tom_seashells 
  (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) (h3 : total_seashells = days_at_beach * seashells_per_day) : 
  total_seashells = 35 := 
by
  rw [h1, h2] at h3 
  exact h3

end tom_seashells_l70_70198


namespace ceiling_and_floor_calculation_l70_70462

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l70_70462


namespace inverse_variation_example_l70_70297

theorem inverse_variation_example (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x y, x * y^3 = k) (h2 : 8 * (1:ℝ)^3 = k) : 
  (∃ (x : ℝ), x * (2:ℝ)^3 = k ∧ x = 1) :=
by
  have hx : 8 = k := by
    rw [←h2, one_mul]
  
  use 1
  split
  . exact hx.symm
  . rfl

end inverse_variation_example_l70_70297


namespace ages_of_sons_l70_70081

variable (x y z : ℕ)

def father_age_current : ℕ := 33

def youngest_age_current : ℕ := 2

def father_age_in_12_years : ℕ := father_age_current + 12

def sum_of_ages_in_12_years : ℕ := youngest_age_current + 12 + y + 12 + z + 12

theorem ages_of_sons (x y z : ℕ) 
  (h1 : x = 2)
  (h2 : father_age_current = 33)
  (h3 : father_age_in_12_years = 45)
  (h4 : sum_of_ages_in_12_years = 45) :
  x = 2 ∧ y + z = 7 ∧ ((y = 3 ∧ z = 4) ∨ (y = 4 ∧ z = 3)) :=
by
  sorry

end ages_of_sons_l70_70081


namespace paint_floor_cost_l70_70732

theorem paint_floor_cost :
  ∀ (L : ℝ) (rate : ℝ)
  (condition1 : L = 3 * (L / 3))
  (condition2 : L = 19.595917942265423)
  (condition3 : rate = 5),
  rate * (L * (L / 3)) = 640 :=
by
  intros L rate condition1 condition2 condition3
  sorry

end paint_floor_cost_l70_70732


namespace rebus_solution_l70_70954

theorem rebus_solution (A B C D : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : D ≠ 0) 
  (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D) :
  1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D) → 
  A = 2 ∧ B = 9 ∧ C = 1 ∧ D = 6 :=
by
  intros h
  sorry

end rebus_solution_l70_70954


namespace swimmer_distance_l70_70593

noncomputable def effective_speed := 4.4 - 2.5
noncomputable def time := 3.684210526315789
noncomputable def distance := effective_speed * time

theorem swimmer_distance :
  distance = 7 := by
  sorry

end swimmer_distance_l70_70593


namespace asymptotes_of_hyperbola_l70_70185

theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 9 = 1) → (y = 3/2 * x ∨ y = -3/2 * x) :=
by
  intro x y h
  -- Proof would go here
  sorry

end asymptotes_of_hyperbola_l70_70185


namespace relationship_x2_ax_bx_l70_70163

variable {x a b : ℝ}

theorem relationship_x2_ax_bx (h1 : x < a) (h2 : a < 0) (h3 : b > 0) : x^2 > ax ∧ ax > bx :=
by
  sorry

end relationship_x2_ax_bx_l70_70163


namespace domain_of_log_function_l70_70542

theorem domain_of_log_function : 
  { x : ℝ | x < 1 ∨ x > 2 } = { x : ℝ | 0 < x^2 - 3 * x + 2 } :=
by sorry

end domain_of_log_function_l70_70542


namespace find_A_in_phone_number_l70_70428

theorem find_A_in_phone_number
  (A B C D E F G H I J : ℕ)
  (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧ 
            B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧ 
            C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧ 
            D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧ 
            E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧ 
            F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧ 
            G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
            H ≠ I ∧ H ≠ J ∧
            I ≠ J)
  (h_dec_ABC : A > B ∧ B > C)
  (h_dec_DEF : D > E ∧ E > F)
  (h_dec_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consec_even_DEF : D % 2 = 0 ∧ E % 2 = 0 ∧ F % 2 = 0 ∧ E = D - 2 ∧ F = E - 2)
  (h_consec_odd_GHIJ : G % 2 = 1 ∧ H % 2 = 1 ∧ I % 2 = 1 ∧ J % 2 = 1 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) :
  A = 8 :=
sorry

end find_A_in_phone_number_l70_70428


namespace two_pow_n_plus_one_square_or_cube_l70_70448

theorem two_pow_n_plus_one_square_or_cube (n : ℕ) :
  (∃ a : ℕ, 2^n + 1 = a^2) ∨ (∃ a : ℕ, 2^n + 1 = a^3) → n = 3 :=
by
  sorry

end two_pow_n_plus_one_square_or_cube_l70_70448


namespace symmetric_coordinates_l70_70540

structure Point :=
  (x : Int)
  (y : Int)

def symmetric_about_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_coordinates (P : Point) (h : P = Point.mk (-1) 2) :
  symmetric_about_origin P = Point.mk 1 (-2) :=
by
  sorry

end symmetric_coordinates_l70_70540


namespace Haley_has_25_necklaces_l70_70830

theorem Haley_has_25_necklaces (J H Q : ℕ) 
  (h1 : H = J + 5) 
  (h2 : Q = J / 2) 
  (h3 : H = Q + 15) : 
  H = 25 := 
sorry

end Haley_has_25_necklaces_l70_70830


namespace no_real_solutions_l70_70476

theorem no_real_solutions :
  ¬ ∃ (a b c d : ℝ), 
  (a^3 + c^3 = 2) ∧ 
  (a^2 * b + c^2 * d = 0) ∧ 
  (b^3 + d^3 = 1) ∧ 
  (a * b^2 + c * d^2 = -6) := 
by
  sorry

end no_real_solutions_l70_70476


namespace evaluate_expression_l70_70614

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end evaluate_expression_l70_70614


namespace sum_of_coefficients_l70_70206

def original_function (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 4

def transformed_function (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2 * (x + 2) + 4 + 5

theorem sum_of_coefficients : (3 : ℝ) + 10 + 17 = 30 :=
by
  sorry

end sum_of_coefficients_l70_70206


namespace three_digit_number_452_l70_70089

theorem three_digit_number_452 (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 1 ≤ b) (h4 : b ≤ 9) (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 100 * a + 10 * b + c % (a + b + c) = 1)
  (h8 : 100 * c + 10 * b + a % (a + b + c) = 1)
  (h9 : a ≠ b) (h10 : b ≠ c) (h11 : a ≠ c)
  (h12 : a > c) :
  100 * a + 10 * b + c = 452 :=
sorry

end three_digit_number_452_l70_70089


namespace solution_correct_l70_70905

def mixed_number_to_fraction (a b c : ℕ) : ℚ :=
  (a * b + c) / b

def percentage_to_decimal (fraction : ℚ) : ℚ :=
  fraction / 100

def evaluate_expression : ℚ :=
  let part1 := 63 * 5 + 4
  let part2 := 48 * 7 + 3
  let part3 := 17 * 3 + 2
  let term1 := (mixed_number_to_fraction 63 5 4) * 3150
  let term2 := (mixed_number_to_fraction 48 7 3) * 2800
  let term3 := (mixed_number_to_fraction 17 3 2) * 945 / 2
  term1 - term2 + term3

theorem solution_correct :
  (percentage_to_decimal (mixed_number_to_fraction 63 5 4) * 3150) -
  (percentage_to_decimal (mixed_number_to_fraction 48 7 3) * 2800) +
  (percentage_to_decimal (mixed_number_to_fraction 17 3 2) * 945 / 2) = 737.175 := 
sorry

end solution_correct_l70_70905


namespace area_of_triangle_ABC_l70_70993

theorem area_of_triangle_ABC (AB CD : ℝ) (height : ℝ) (h1 : CD = 3 * AB) (h2 : AB * height + CD * height = 48) :
  (1/2) * AB * height = 6 :=
by
  have trapezoid_area : AB * height + CD * height = 48 := h2
  have length_relation : CD = 3 * AB := h1
  have area_triangle_ABC := 6
  sorry

end area_of_triangle_ABC_l70_70993


namespace correct_growth_equation_l70_70501

-- Define the parameters
def initial_income : ℝ := 2.36
def final_income : ℝ := 2.7
def growth_period : ℕ := 2

-- Define the growth rate x
variable (x : ℝ)

-- The theorem we want to prove
theorem correct_growth_equation : initial_income * (1 + x)^growth_period = final_income :=
sorry

end correct_growth_equation_l70_70501


namespace evaluate_expression_l70_70797

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 := by
  sorry

end evaluate_expression_l70_70797


namespace evaluate_at_2_l70_70057

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem evaluate_at_2 : f 2 = 62 := 
by
  sorry

end evaluate_at_2_l70_70057


namespace cheesecake_total_calories_l70_70104

-- Define the conditions
def slice_calories : ℕ := 350

def percent_eaten : ℕ := 25
def slices_eaten : ℕ := 2

-- Define the total number of slices in a cheesecake
def total_slices (percent_eaten slices_eaten : ℕ) : ℕ :=
  slices_eaten * (100 / percent_eaten)

-- Define the total calories in a cheesecake given the above conditions
def total_calories (slice_calories slices : ℕ) : ℕ :=
  slice_calories * slices

-- State the theorem
theorem cheesecake_total_calories :
  total_calories slice_calories (total_slices percent_eaten slices_eaten) = 2800 :=
by
  sorry

end cheesecake_total_calories_l70_70104


namespace max_integer_in_form_3_x_3_sub_x_l70_70793

theorem max_integer_in_form_3_x_3_sub_x :
  ∃ x : ℝ, ∀ y : ℝ, y = 3^(x * (3 - x)) → ⌊y⌋ ≤ 11 := 
sorry

end max_integer_in_form_3_x_3_sub_x_l70_70793


namespace numbers_neither_square_nor_cube_l70_70655

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l70_70655


namespace triangle_area_correct_l70_70890

/-- Vertices of the triangle -/
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (7, 2)
def C : ℝ × ℝ := (4, 8)

/-- Function to calculate the triangle area given vertices -/
def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- The problem statement -/
theorem triangle_area_correct :
  triangle_area A B C = 15 :=
by
  sorry

end triangle_area_correct_l70_70890


namespace mary_can_keep_warm_l70_70169

theorem mary_can_keep_warm :
  let chairs := 18
  let chairs_sticks := 6
  let tables := 6
  let tables_sticks := 9
  let stools := 4
  let stools_sticks := 2
  let sticks_per_hour := 5
  let total_sticks := (chairs * chairs_sticks) + (tables * tables_sticks) + (stools * stools_sticks)
  let hours := total_sticks / sticks_per_hour
  hours = 34 := by
{
  sorry
}

end mary_can_keep_warm_l70_70169


namespace triangle_lines_l70_70642

/-- Given a triangle with vertices A(1, 2), B(-1, 4), and C(4, 5):
  1. The equation of the line l₁ containing the altitude from A to side BC is 5x + y - 7 = 0.
  2. The equation of the line l₂ passing through C such that the distances from A and B to l₂ are equal
     is either x + y - 9 = 0 or x - 2y + 6 = 0. -/
theorem triangle_lines (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (hB : B = (-1, 4))
  (hC : C = (4, 5)) :
  ∃ l₁ l₂ : ℝ × ℝ × ℝ,
  (l₁ = (5, 1, -7)) ∧
  ((l₂ = (1, 1, -9)) ∨ (l₂ = (1, -2, 6))) := by
  sorry

end triangle_lines_l70_70642


namespace annie_initial_money_l70_70226

theorem annie_initial_money (h_cost : ℕ) (m_cost : ℕ) (h_count : ℕ) (m_count : ℕ) (remaining_money : ℕ) :
  h_cost = 4 → m_cost = 5 → h_count = 8 → m_count = 6 → remaining_money = 70 →
  h_cost * h_count + m_cost * m_count + remaining_money = 132 :=
by
  intros h_cost_def m_cost_def h_count_def m_count_def remaining_money_def
  rw [h_cost_def, m_cost_def, h_count_def, m_count_def, remaining_money_def]
  sorry

end annie_initial_money_l70_70226


namespace probability_of_white_balls_from_both_boxes_l70_70266

theorem probability_of_white_balls_from_both_boxes :
  let P_white_A := 3 / (3 + 2)
  let P_white_B := 2 / (2 + 3)
  P_white_A * P_white_B = 6 / 25 :=
by
  sorry

end probability_of_white_balls_from_both_boxes_l70_70266


namespace max_playground_area_l70_70645

theorem max_playground_area
  (l w : ℝ)
  (h_fence : 2 * l + 2 * w = 400)
  (h_l_min : l ≥ 100)
  (h_w_min : w ≥ 50) :
  l * w ≤ 10000 :=
by
  sorry

end max_playground_area_l70_70645


namespace function_passes_through_fixed_point_l70_70128

noncomputable def f (a : ℝ) (x : ℝ) := 4 + Real.log (x + 1) / Real.log a

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  f a 0 = 4 := 
by
  sorry

end function_passes_through_fixed_point_l70_70128


namespace max_possible_value_l70_70507

-- Define the number of cities and the structure of roads.
def numCities : ℕ := 110

-- Condition: Each city has either a road or no road to another city
def Road (city1 city2 : ℕ) : Prop := sorry  -- A placeholder definition for the road relationship

-- Condition: Number of roads leading out of each city.
def numRoads (city : ℕ) : ℕ := sorry  -- A placeholder for the actual function counting the number of roads from a city

-- Condition: The driver starts at a city with exactly one road leading out.
def startCity : ℕ := sorry  -- A placeholder for the starting city

-- Main theorem statement to prove the maximum possible value of N is 107
theorem max_possible_value : ∃ N : ℕ, N ≤ 107 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ N → numRoads k = k) :=
by
  sorry  -- Actual proof is not required, hence we use sorry to indicate the proof step is skipped.

end max_possible_value_l70_70507


namespace rebus_solution_l70_70952

theorem rebus_solution (A B C D : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : D ≠ 0) 
  (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D) :
  1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D) → 
  A = 2 ∧ B = 9 ∧ C = 1 ∧ D = 6 :=
by
  intros h
  sorry

end rebus_solution_l70_70952


namespace point_locus_l70_70249

variables {A B C : Type} [MetricSpace A]

-- Definition of angle measure
noncomputable def angle (A B C : A) : ℝ := sorry

-- Definition of the condition: ∠ACB = 30 degrees
def angle_condition (A B C : A) : Prop := angle A C B = 30

-- The final proof statement in Lean
theorem point_locus (A B : A) :
  ∃ C : A, angle_condition A B C → C ∈ locus C :=
sorry

end point_locus_l70_70249


namespace ants_harvest_time_l70_70592

theorem ants_harvest_time :
  ∃ h : ℕ, (∀ h : ℕ, 24 - 4 * h = 12) ∧ h = 3 := sorry

end ants_harvest_time_l70_70592


namespace find_digits_l70_70107

def divisible_45z_by_8 (z : ℕ) : Prop :=
  45 * z % 8 = 0

def sum_digits_divisible_by_9 (x y z : ℕ) : Prop :=
  (1 + 3 + x + y + 4 + 5 + z) % 9 = 0

def alternating_sum_digits_divisible_by_11 (x y z : ℕ) : Prop :=
  (1 - 3 + x - y + 4 - 5 + z) % 11 = 0

theorem find_digits (x y z : ℕ) (h_div8 : divisible_45z_by_8 z) (h_div9 : sum_digits_divisible_by_9 x y z) (h_div11 : alternating_sum_digits_divisible_by_11 x y z) :
  x = 2 ∧ y = 3 ∧ z = 6 := 
sorry

end find_digits_l70_70107


namespace factorization_x3_minus_9xy2_l70_70210

theorem factorization_x3_minus_9xy2 (x y : ℝ) : x^3 - 9 * x * y^2 = x * (x + 3 * y) * (x - 3 * y) :=
by sorry

end factorization_x3_minus_9xy2_l70_70210


namespace maximum_ratio_is_2_plus_2_sqrt2_l70_70154

noncomputable def C1_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ * (Real.cos θ + Real.sin θ) = 1

noncomputable def C2_polar_eq (θ : ℝ) : Prop :=
  ∀ ρ : ℝ, ρ = 4 * Real.cos θ

theorem maximum_ratio_is_2_plus_2_sqrt2 (α : ℝ) (hα : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃ ρA ρB : ℝ, (ρA = 1 / (Real.cos α + Real.sin α)) ∧ (ρB = 4 * Real.cos α) ∧ 
  (4 * Real.cos α * (Real.cos α + Real.sin α) = 2 + 2 * Real.sqrt 2) :=
sorry

end maximum_ratio_is_2_plus_2_sqrt2_l70_70154


namespace repeating_six_as_fraction_l70_70112

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l70_70112


namespace hua_luogeng_optimal_selection_l70_70317

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l70_70317


namespace rebus_solution_l70_70953

theorem rebus_solution (A B C D : ℕ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : C ≠ 0) (h4 : D ≠ 0) 
  (h5 : A ≠ B) (h6 : A ≠ C) (h7 : A ≠ D) (h8 : B ≠ C) (h9 : B ≠ D) (h10 : C ≠ D) :
  1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D) → 
  A = 2 ∧ B = 9 ∧ C = 1 ∧ D = 6 :=
by
  intros h
  sorry

end rebus_solution_l70_70953


namespace arithmetic_geometric_problem_l70_70963

noncomputable def arithmetic_sequence (a b : ℤ) := ∃ d : ℤ, b - a = d
noncomputable def geometric_sequence (a b : ℤ) := ∃ r : ℤ, b = a * r

theorem arithmetic_geometric_problem
  (a b d : ℤ)
  (arith_seq: are_seq (-1) a b (-4))
  (geom_seq: geo_seq (-1) c d e (-4)): 
  \frac{b - a}{d} = \frac{1}{2} :=
by
  sorry

end arithmetic_geometric_problem_l70_70963


namespace three_digit_number_is_11_times_sum_of_digits_l70_70622

theorem three_digit_number_is_11_times_sum_of_digits :
    ∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
        (100 * a + 10 * b + c = 11 * (a + b + c)) ↔ 
        (100 * 1 + 10 * 9 + 8 = 11 * (1 + 9 + 8)) := 
by
    sorry

end three_digit_number_is_11_times_sum_of_digits_l70_70622


namespace no_integer_roots_l70_70108

theorem no_integer_roots (x : ℤ) : ¬ (x^3 - 5 * x^2 - 11 * x + 35 = 0) := 
sorry

end no_integer_roots_l70_70108


namespace mary_keep_warm_hours_l70_70168

-- Definitions based on the conditions
def sticks_from_chairs (chairs : ℕ) : ℕ := chairs * 6
def sticks_from_tables (tables : ℕ) : ℕ := tables * 9
def sticks_from_stools (stools : ℕ) : ℕ := stools * 2
def sticks_needed_per_hour : ℕ := 5

-- Given counts of furniture
def chairs : ℕ := 18
def tables : ℕ := 6
def stools : ℕ := 4

-- Total number of sticks
def total_sticks : ℕ := (sticks_from_chairs chairs) + (sticks_from_tables tables) + (sticks_from_stools stools)

-- Proving the number of hours Mary can keep warm
theorem mary_keep_warm_hours : total_sticks / sticks_needed_per_hour = 34 := by
  sorry

end mary_keep_warm_hours_l70_70168


namespace max_possible_N_in_cities_l70_70506

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l70_70506


namespace angela_sleep_difference_l70_70781

theorem angela_sleep_difference :
  let december_sleep_hours := 6.5
  let january_sleep_hours := 8.5
  let december_days := 31
  let january_days := 31
  (january_sleep_hours * january_days) - (december_sleep_hours * december_days) = 62 :=
by
  sorry

end angela_sleep_difference_l70_70781


namespace factors_of_1320_l70_70930

theorem factors_of_1320 : ∃ n : ℕ, n = 24 ∧ ∃ (a b c d : ℕ),
  1320 = 2^a * 3^b * 5^c * 11^d ∧ (a = 0 ∨ a = 1 ∨ a = 2) ∧ (b = 0 ∨ b = 1) ∧ (c = 0 ∨ c = 1) ∧ (d = 0 ∨ d = 1) :=
by {
  sorry
}

end factors_of_1320_l70_70930


namespace foci_of_ellipse_l70_70024

-- Define the ellipsis
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 25) = 1

-- Prove the coordinates of foci of the ellipse
theorem foci_of_ellipse :
  ∃ c : ℝ, c = 3 ∧ ((0, c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2} ∧ (0, -c) ∈ {p : ℝ × ℝ | ellipse p.1 p.2}) :=
by
  sorry

end foci_of_ellipse_l70_70024


namespace max_N_value_l70_70518

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l70_70518


namespace incorrect_inequality_l70_70807

theorem incorrect_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) : ¬ (a^2 < a * b) :=
by
  sorry

end incorrect_inequality_l70_70807


namespace find_daily_wage_of_c_l70_70896

noncomputable def daily_wage_c (a b c : ℕ) (days_a days_b days_c total_earning : ℕ) : ℕ :=
  if 3 * b = 4 * a ∧ 3 * c = 5 * a ∧ 
    total_earning = 6 * a + 9 * b + 4 * c then c else 0

theorem find_daily_wage_of_c (a b c : ℕ)
  (days_a days_b days_c total_earning : ℕ)
  (h1 : days_a = 6)
  (h2 : days_b = 9)
  (h3 : days_c = 4)
  (h4 : 3 * b = 4 * a)
  (h5 : 3 * c = 5 * a)
  (h6 : total_earning = 1554)
  (h7 : total_earning = 6 * a + 9 * b + 4 * c) : 
  daily_wage_c a b c days_a days_b days_c total_earning = 105 := 
by sorry

end find_daily_wage_of_c_l70_70896


namespace probability_is_one_twelfth_l70_70196

def probability_red_gt4_green_odd_blue_lt4 : ℚ :=
  let total_outcomes := 6 * 6 * 6
  let successful_outcomes := 2 * 3 * 3
  successful_outcomes / total_outcomes

theorem probability_is_one_twelfth :
  probability_red_gt4_green_odd_blue_lt4 = 1 / 12 :=
by
  -- proof here
  sorry

end probability_is_one_twelfth_l70_70196


namespace constant_term_q_l70_70855

theorem constant_term_q (p q r : Polynomial ℝ) 
  (hp_const : p.coeff 0 = 6) 
  (hr_const : (p * q).coeff 0 = -18) : q.coeff 0 = -3 :=
sorry

end constant_term_q_l70_70855


namespace sum_infinite_series_l70_70606

theorem sum_infinite_series :
  ∑' n : ℕ, (3 * (n+1) + 2) / ((n+1) * (n+2) * (n+4)) = 29 / 36 :=
by
  sorry

end sum_infinite_series_l70_70606


namespace percentage_employees_six_years_or_more_l70_70884

theorem percentage_employees_six_years_or_more:
  let marks : List ℕ := [6, 6, 7, 4, 3, 3, 3, 1, 1, 1]
  let total_employees (marks : List ℕ) (y : ℕ) := marks.foldl (λ acc m => acc + m * y) 0
  let employees_six_years_or_more (marks : List ℕ) (y : ℕ) := (marks.drop 6).foldl (λ acc m => acc + m * y) 0
  (employees_six_years_or_more marks 1 / total_employees marks 1 : ℚ) * 100 = 17.14 := by
  sorry

end percentage_employees_six_years_or_more_l70_70884


namespace bicycle_spokes_count_l70_70599

theorem bicycle_spokes_count (bicycles wheels spokes : ℕ) 
       (h1 : bicycles = 4) 
       (h2 : wheels = 2) 
       (h3 : spokes = 10) : 
       bicycles * (wheels * spokes) = 80 :=
by
  sorry

end bicycle_spokes_count_l70_70599


namespace hypotenuse_of_right_triangle_l70_70187

theorem hypotenuse_of_right_triangle (h : height_dropped_to_hypotenuse = 1) (a : acute_angle = 15) :
∃ (hypotenuse : ℝ), hypotenuse = 4 :=
sorry

end hypotenuse_of_right_triangle_l70_70187


namespace geometric_sequence_fifth_term_l70_70694

variable {a : ℕ → ℝ} (h1 : a 1 = 1) (h4 : a 4 = 8)

theorem geometric_sequence_fifth_term (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) :
  a 5 = 16 :=
sorry

end geometric_sequence_fifth_term_l70_70694


namespace optimal_selection_uses_golden_ratio_l70_70326

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l70_70326


namespace optimal_selection_method_uses_golden_ratio_l70_70311

theorem optimal_selection_method_uses_golden_ratio
  (Hua_Luogeng : ∀ (contribution : Type), contribution = "optimal selection method popularization")
  (method_uses : ∀ (concept : string), concept = "Golden ratio" ∨ concept = "Mean" ∨ concept = "Mode" ∨ concept = "Median") :
  method_uses "Golden ratio" :=
by
  sorry

end optimal_selection_method_uses_golden_ratio_l70_70311


namespace rebus_problem_l70_70944

-- Define non-zero digit type
def NonZeroDigit := {d : Fin 10 // d.val ≠ 0}

-- Define the problem
theorem rebus_problem (A B C D : NonZeroDigit) (h1 : A.1 ≠ B.1) (h2 : A.1 ≠ C.1) (h3 : A.1 ≠ D.1) (h4 : B.1 ≠ C.1) (h5 : B.1 ≠ D.1) (h6 : C.1 ≠ D.1):
  let ABCD := 1000 * A.1 + 100 * B.1 + 10 * C.1 + D.1
  let ABCA := 1001 * A.1 + 100 * B.1 + 10 * C.1 + A.1
  ∃ (n : ℕ), ABCA = 182 * (10 * C.1 + D.1) → ABCD = 2916 :=
begin
  intro h,
  use 51, -- 2916 is 51 * 182
  sorry
end

end rebus_problem_l70_70944


namespace mean_of_six_numbers_l70_70040

theorem mean_of_six_numbers (a b c d e f : ℚ) (h : a + b + c + d + e + f = 1 / 3) :
  (a + b + c + d + e + f) / 6 = 1 / 18 :=
by
  sorry

end mean_of_six_numbers_l70_70040


namespace xyz_equivalence_l70_70118

theorem xyz_equivalence (x y z a b : ℝ) (h₁ : 4^x = a) (h₂: 2^y = b) (h₃ : 8^z = a * b) : 3 * z = 2 * x + y :=
by
  -- Here, we leave the proof as an exercise
  sorry

end xyz_equivalence_l70_70118


namespace initial_tomatoes_count_l70_70740

-- Definitions and conditions
def birds_eat_fraction : ℚ := 1/3
def tomatoes_left : ℚ := 14
def fraction_tomatoes_left : ℚ := 2/3

-- We want to prove the initial number of tomatoes
theorem initial_tomatoes_count (initial_tomatoes : ℚ) 
  (h1 : tomatoes_left = fraction_tomatoes_left * initial_tomatoes) : 
  initial_tomatoes = 21 := 
by
  -- skipping the proof for now
  sorry

end initial_tomatoes_count_l70_70740


namespace employees_in_factory_l70_70987

theorem employees_in_factory (initial_total : ℕ) (init_prod : ℕ) (init_admin : ℕ)
  (increase_prod_frac : ℚ) (increase_admin_frac : ℚ) :
  initial_total = 1200 →
  init_prod = 800 →
  init_admin = 400 →
  increase_prod_frac = 0.35 →
  increase_admin_frac = 3 / 5 →
  init_prod + init_prod * increase_prod_frac +
  init_admin + init_admin * increase_admin_frac = 1720 := by
  intros h_total h_prod h_admin h_inc_prod h_inc_admin
  sorry

end employees_in_factory_l70_70987


namespace find_equation_of_line_midpoint_find_equation_of_line_vector_l70_70088

-- Definition for Problem 1
def equation_of_line_midpoint (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (x / (-6) + y / 2 = 1) ∧ l (-3) = 1)

-- Proof Statement for Problem 1
theorem find_equation_of_line_midpoint : equation_of_line_midpoint (-6) 2 :=
sorry

-- Definition for Problem 2
def equation_of_line_vector (x y : ℝ) : Prop :=
  ∃ l : ℝ → ℝ, (l x = 0 ∧ l 0 = y ∧ (y - 1) / (-1) = (x + 3) / (-6) ∧ l (-3) = 1)

-- Proof Statement for Problem 2
theorem find_equation_of_line_vector : equation_of_line_vector (-9) (3 / 2) :=
sorry

end find_equation_of_line_midpoint_find_equation_of_line_vector_l70_70088


namespace gray_areas_trees_count_l70_70418

noncomputable def totalTreesInGrayAreas (T : ℕ) (white1 white2 white3 : ℕ) : ℕ :=
  let gray2 := T - white2
  let gray3 := T - white3
  gray2 + gray3

theorem gray_areas_trees_count (T : ℕ) :
  T = 100 → totalTreesInGrayAreas T 100 82 90 = 26 :=
by sorry

end gray_areas_trees_count_l70_70418


namespace optimal_selection_method_uses_golden_ratio_l70_70324

theorem optimal_selection_method_uses_golden_ratio (H : True) :
  (∃ choice : String, choice = "Golden ratio") :=
by
  -- We just introduce assumptions to match the problem statement
  have options := ["Golden ratio", "Mean", "Mode", "Median"]
  have chosen_option := "Golden ratio"
  use chosen_option
  have golden_ratio_in_options : List.contains options "Golden ratio" := by
    simp [List.contains]
  simp
  have correct_choice : chosen_option = options.head := by
    simp
    sorry

-- The theorem assumes that Hua's method uses a concept
-- and we show that the choice "Golden ratio" is the concept used.

end optimal_selection_method_uses_golden_ratio_l70_70324


namespace total_bananas_eq_l70_70885

def groups_of_bananas : ℕ := 2
def bananas_per_group : ℕ := 145

theorem total_bananas_eq : groups_of_bananas * bananas_per_group = 290 :=
by
  sorry

end total_bananas_eq_l70_70885


namespace chocolate_truffles_sold_l70_70581

def fudge_sold_pounds : ℕ := 20
def price_per_pound_fudge : ℝ := 2.50
def price_per_truffle : ℝ := 1.50
def pretzels_sold_dozen : ℕ := 3
def price_per_pretzel : ℝ := 2.00
def total_revenue : ℝ := 212.00

theorem chocolate_truffles_sold (dozens_of_truffles_sold : ℕ) :
  let fudge_revenue := (fudge_sold_pounds : ℝ) * price_per_pound_fudge
  let pretzels_revenue := (pretzels_sold_dozen : ℝ) * 12 * price_per_pretzel
  let truffles_revenue := total_revenue - fudge_revenue - pretzels_revenue
  let num_truffles_sold := truffles_revenue / price_per_truffle
  let dozens_of_truffles_sold := num_truffles_sold / 12
  dozens_of_truffles_sold = 5 :=
by
  sorry

end chocolate_truffles_sold_l70_70581


namespace campers_rowing_morning_equals_41_l70_70866

def campers_went_rowing_morning (hiking_morning : ℕ) (rowing_afternoon : ℕ) (total : ℕ) : ℕ :=
  total - (hiking_morning + rowing_afternoon)

theorem campers_rowing_morning_equals_41 :
  ∀ (hiking_morning rowing_afternoon total : ℕ), hiking_morning = 4 → rowing_afternoon = 26 → total = 71 → campers_went_rowing_morning hiking_morning rowing_afternoon total = 41 := by
  intros hiking_morning rowing_afternoon total hiking_morning_cond rowing_afternoon_cond total_cond
  rw [hiking_morning_cond, rowing_afternoon_cond, total_cond]
  exact rfl

end campers_rowing_morning_equals_41_l70_70866


namespace num_non_squares_cubes_1_to_200_l70_70675

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k * k * k * k * k * k = n

def num_perfect_squares (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_square - (List.range (a)).countp is_perfect_square

def num_perfect_cubes (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_cube - (List.range (a)).countp is_perfect_cube

def num_perfect_sixth_powers (a b : ℕ) : ℕ :=
  (List.range (b + 1)).countp is_perfect_sixth_power - (List.range (a)).countp is_perfect_sixth_power

theorem num_non_squares_cubes_1_to_200 :
  let squares := num_perfect_squares 1 200 in
  let cubes := num_perfect_cubes 1 200 in
  let sixth_powers := num_perfect_sixth_powers 1 200 in
  200 - (squares + cubes - sixth_powers) = 182 :=
by
  let squares := num_perfect_squares 1 200
  let cubes := num_perfect_cubes 1 200
  let sixth_powers := num_perfect_sixth_powers 1 200
  have h : 200 - (squares + cubes - sixth_powers) = 182 := sorry
  exact h

end num_non_squares_cubes_1_to_200_l70_70675


namespace max_roads_city_condition_l70_70515

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l70_70515


namespace three_digit_number_multiple_of_eleven_l70_70620

theorem three_digit_number_multiple_of_eleven:
  ∃ (a b c : ℕ), (1 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧ (0 ≤ c) ∧ (c ≤ 9) ∧
                  (100 * a + 10 * b + c = 11 * (a + b + c) ∧ (100 * a + 10 * b + c = 198)) :=
by
  use 1
  use 9
  use 8
  sorry

end three_digit_number_multiple_of_eleven_l70_70620


namespace tangent_line_equation_l70_70027

theorem tangent_line_equation (x y : ℝ) :
  (y = Real.exp x + 2) →
  (x = 0) →
  (y = 3) →
  (Real.exp x = 1) →
  (x - y + 3 = 0) :=
by
  intros h_eq h_x h_y h_slope
  -- The following proof will use the conditions to show the tangent line equation.
  sorry

end tangent_line_equation_l70_70027


namespace count_subsets_l70_70923

theorem count_subsets (S T : Set ℕ) (h1 : S = {1, 2, 3}) (h2 : T = {1, 2, 3, 4, 5, 6, 7}) :
  (∃ n : ℕ, n = 16 ∧ ∀ X, S ⊆ X ∧ X ⊆ T ↔ X ∈ { X | ∃ m : ℕ, m = 16 }) := 
sorry

end count_subsets_l70_70923


namespace required_equation_l70_70110

-- Define the given lines
def line1 (x y : ℝ) : Prop := 2 * x - y = 0
def line2 (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the equation to be proven for the line through the intersection point and perpendicular to perp_line
def required_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Define the predicate that states a point (2, 4) lies on line1 and line2
def point_intersect (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The main theorem to be proven in Lean 4
theorem required_equation : 
  point_intersect 2 4 ∧ perp_line 2 4 → required_line 2 4 := by
  sorry

end required_equation_l70_70110


namespace optimal_selection_method_uses_golden_ratio_l70_70350

theorem optimal_selection_method_uses_golden_ratio :
  (one_of_the_methods_in_optimal_selection_method, optimal_selection_method_popularized_by_hua_luogeng) → uses_golden_ratio :=
sorry

end optimal_selection_method_uses_golden_ratio_l70_70350


namespace common_remainder_proof_l70_70204

def least_subtracted := 6
def original_number := 1439
def reduced_number := original_number - least_subtracted
def divisors := [5, 11, 13]
def common_remainder := 3

theorem common_remainder_proof :
  ∀ d ∈ divisors, reduced_number % d = common_remainder := by
  sorry

end common_remainder_proof_l70_70204


namespace ceil_floor_diff_l70_70471

theorem ceil_floor_diff : 
  (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in 
     ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋) = 2 :=
by
  let h1 : ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ = -15 := sorry
  let h2 : ⌊(-34 : ℤ) / 4⌋ = -9 := sorry
  let h3 : (15 : ℤ) / 8 * (-9 : ℤ) = (15 * (-9)) / (8) := sorry
  let h4 : ⌊(15 : ℤ) / 8 * (-9)⌋ = -17 := sorry
  calc
    (let x := (15 : ℤ) / 8 * (-34 : ℤ) / 4 in ⌈x⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋)
        = ⌈(15 : ℤ) / 8 * (-34 : ℤ) / 4⌉ - ⌊(15 : ℤ) / 8 * ⌊(-34 : ℤ) / 4⌋⌋  : by rfl
    ... = -15 - (-17) : by { rw [h1, h4] }
    ... = 2 : by simp

end ceil_floor_diff_l70_70471


namespace geom_progression_common_ratio_l70_70146

theorem geom_progression_common_ratio (x y z r : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : ∃ a, a ≠ 0 ∧ x * (2 * y - z) = a ∧ y * (2 * z - x) = a * r ∧ z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end geom_progression_common_ratio_l70_70146


namespace child_tickets_sold_l70_70744

theorem child_tickets_sold
  (A C : ℕ)
  (h1 : A + C = 130)
  (h2 : 12 * A + 4 * C = 840) : C = 90 :=
  by {
  -- Proof skipped
  sorry
}

end child_tickets_sold_l70_70744


namespace reporters_covering_local_politics_l70_70079

theorem reporters_covering_local_politics (R : ℕ) (P Q A B : ℕ)
  (h1 : P = 70)
  (h2 : Q = 100 - P)
  (h3 : A = 40)
  (h4 : B = 100 - A) :
  B % 30 = 18 :=
by
  sorry

end reporters_covering_local_politics_l70_70079


namespace prism_dimensions_l70_70300

theorem prism_dimensions (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 60) : 
  a = 7.2 ∧ b = 9.6 ∧ c = 14.4 :=
by {
  -- Proof skipped for now
  sorry
}

end prism_dimensions_l70_70300


namespace flagpole_height_l70_70419

theorem flagpole_height
  (AB : ℝ) (AD : ℝ) (BC : ℝ)
  (h1 : AB = 10)
  (h2 : BC = 3)
  (h3 : 2 * AD^2 = AB^2 + BC^2) :
  AD = Real.sqrt 54.5 :=
by 
  -- Proof omitted
  sorry

end flagpole_height_l70_70419


namespace rebus_problem_l70_70947

-- Define non-zero digit type
def NonZeroDigit := {d : Fin 10 // d.val ≠ 0}

-- Define the problem
theorem rebus_problem (A B C D : NonZeroDigit) (h1 : A.1 ≠ B.1) (h2 : A.1 ≠ C.1) (h3 : A.1 ≠ D.1) (h4 : B.1 ≠ C.1) (h5 : B.1 ≠ D.1) (h6 : C.1 ≠ D.1):
  let ABCD := 1000 * A.1 + 100 * B.1 + 10 * C.1 + D.1
  let ABCA := 1001 * A.1 + 100 * B.1 + 10 * C.1 + A.1
  ∃ (n : ℕ), ABCA = 182 * (10 * C.1 + D.1) → ABCD = 2916 :=
begin
  intro h,
  use 51, -- 2916 is 51 * 182
  sorry
end

end rebus_problem_l70_70947


namespace terminal_side_angle_is_in_fourth_quadrant_l70_70965

variable (α : ℝ)
variable (tan_alpha cos_alpha : ℝ)

-- Given conditions
def in_second_quadrant := tan_alpha < 0 ∧ cos_alpha > 0

-- Conclusion to prove
theorem terminal_side_angle_is_in_fourth_quadrant 
  (h : in_second_quadrant tan_alpha cos_alpha) : 
  -- Here we model the "fourth quadrant" in a proof-statement context:
  true := sorry

end terminal_side_angle_is_in_fourth_quadrant_l70_70965


namespace find_integers_a_l70_70618

theorem find_integers_a (a : ℤ) : 
  (∃ n : ℤ, (a^3 + 1 = (a - 1) * n)) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := 
sorry

end find_integers_a_l70_70618


namespace min_fraction_expression_l70_70277

theorem min_fraction_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 1 / b = 1) : 
  ∃ a b, ∃ (h : 1 / a + 1 / b = 1), a > 1 ∧ b > 1 ∧ (1 / (a - 1) + 4 / (b - 1)) = 4 := 
by 
  sorry

end min_fraction_expression_l70_70277


namespace games_draw_fraction_l70_70520

-- Definitions from the conditions in the problems
def ben_win_fraction : ℚ := 4 / 9
def tom_win_fraction : ℚ := 1 / 3

-- The theorem we want to prove
theorem games_draw_fraction : 1 - (ben_win_fraction + (1 / 3)) = 2 / 9 := by
  sorry

end games_draw_fraction_l70_70520


namespace third_vertex_y_coordinate_correct_l70_70225

noncomputable def third_vertex_y_coordinate (x1 y1 x2 y2 : ℝ) (h : y1 = y2) (h_dist : |x1 - x2| = 10) : ℝ :=
  y1 + 5 * Real.sqrt 3

theorem third_vertex_y_coordinate_correct : 
  third_vertex_y_coordinate 3 4 13 4 rfl (by norm_num) = 4 + 5 * Real.sqrt 3 :=
by
  sorry

end third_vertex_y_coordinate_correct_l70_70225


namespace intersection_point_of_lines_l70_70728

theorem intersection_point_of_lines : 
  ∃ (x y : ℝ), (x - 4 * y - 1 = 0) ∧ (2 * x + y - 2 = 0) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end intersection_point_of_lines_l70_70728


namespace linoleum_cut_rearrange_l70_70216

def linoleum : Type := sorry -- placeholder for the specific type of the linoleum piece

def A : linoleum := sorry -- define piece A
def B : linoleum := sorry -- define piece B

def cut_and_rearrange (L : linoleum) (A B : linoleum) : Prop :=
  -- Define the proposition that pieces A and B can be rearranged into an 8x8 square
  sorry

theorem linoleum_cut_rearrange (L : linoleum) (A B : linoleum) :
  cut_and_rearrange L A B :=
sorry

end linoleum_cut_rearrange_l70_70216


namespace minimum_value_expression_l70_70639

open Real

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 3) :
  (∃ (min_val : ℝ), min_val = (13 / 5) ∧ ∀ x y, x > 0 → y > 0 → 2 * x + y = 3 → (2 * x^2 + 1) / x + (y^2 - 2) / (y + 2) ≥ min_val) :=
by
  use (13 / 5)
  sorry

end minimum_value_expression_l70_70639


namespace general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l70_70246

-- Define the conditions
axiom condition1 (n : ℕ) (h : 2 ≤ n) : ∀ (a : ℕ → ℕ), a 1 = 1 → a n = n / (n-1) * a (n-1)
axiom condition2 (n : ℕ) : ∀ (S : ℕ → ℕ), 2 * S n = n^2 + n
axiom condition3 (n : ℕ) : ∀ (a : ℕ → ℕ), a 1 = 1 → a 3 = 3 → (a n + a (n+2)) = 2 * a (n+1)

-- Proof statements
theorem general_formula_condition1 : ∀ (n : ℕ) (a : ℕ → ℕ) (h : 2 ≤ n), (a 1 = 1) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition2 : ∀ (n : ℕ) (S a : ℕ → ℕ), (2 * S n = n^2 + n) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition3 : ∀ (n : ℕ) (a : ℕ → ℕ), (a 1 = 1) → (a 3 = 3) → (∀ n, a n + a (n+2) = 2 * a (n+1)) → (∀ n, a n = n) :=
by sorry

theorem sum_Tn : ∀ (b : ℕ → ℕ) (T : ℕ → ℝ), (b 1 = 2) → (b 2 + b 3 = 12) → (∀ n, T n = 2 * (1 - 1 / (n + 1))) :=
by sorry

end general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l70_70246


namespace max_roads_city_condition_l70_70516

theorem max_roads_city_condition :
  (∃ (cities : ℕ) (roads : Π (n : ℕ), fin n -> fin 110 -> Prop),
  cities = 110 ∧
  (∀ n, (n < 110) -> (∃ k, k < 110 ∧ (∀ i, i ∈ (fin n).val -> (roads n i = true -> (∀ j, j != i -> roads n j = false)) ->
  (n = 0 → ∀ k, k = 1)) ∧
  (N ≤ 107))) .

end max_roads_city_condition_l70_70516


namespace sugar_for_third_layer_l70_70601

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end sugar_for_third_layer_l70_70601


namespace son_l70_70897

theorem son's_age (S F : ℕ) (h1: F = S + 27) (h2: F + 2 = 2 * (S + 2)) : S = 25 := by
  sorry

end son_l70_70897


namespace optimal_selection_method_uses_golden_ratio_l70_70315

-- Conditions
def optimal_selection_method_popularized_by_Hua_Luogeng := true

-- Question translated to proof problem
theorem optimal_selection_method_uses_golden_ratio :
  optimal_selection_method_popularized_by_Hua_Luogeng → (optimal_selection_method_uses "Golden ratio") :=
by 
  intro h
  sorry

end optimal_selection_method_uses_golden_ratio_l70_70315


namespace count_valid_numbers_between_1_and_200_l70_70648

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

def count_perfect_squares (m : ℕ) : ℕ :=
  Nat.sqrt m

def count_perfect_cubes (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.to_nat

def count_perfect_sixth_powers (m : ℕ) : ℕ :=
  (m : ℝ).cbrt.sqrt.to_nat

def count_either_squares_or_cubes (m : ℕ) : ℕ :=
  let squares := count_perfect_squares m
  let cubes := count_perfect_cubes m
  let sixths := count_perfect_sixth_powers m
  squares + cubes - sixths

def count_neither_squares_nor_cubes (n : ℕ) : ℕ :=
  n - count_either_squares_or_cubes n

theorem count_valid_numbers_between_1_and_200 : count_neither_squares_nor_cubes 200 = 182 :=
by
  sorry

end count_valid_numbers_between_1_and_200_l70_70648


namespace least_value_of_x_l70_70899

theorem least_value_of_x (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) (h3 : x = 11 * p * 2) : x = 44 := 
by
  sorry

end least_value_of_x_l70_70899


namespace hua_luogeng_optimal_selection_method_l70_70340

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l70_70340


namespace no_such_triples_l70_70935

theorem no_such_triples : ¬ ∃ (x y z : ℤ), (xy + yz + zx ≠ 0) ∧ (x^2 + y^2 + z^2) / (xy + yz + zx) = 2016 :=
by
  sorry

end no_such_triples_l70_70935


namespace count_not_squares_or_cubes_200_l70_70670

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n
def is_sixth_power (n : ℕ) : Prop := ∃ k : ℕ, k^6 = n

theorem count_not_squares_or_cubes_200 :
  let count_squares := (Finset.range 201).filter is_perfect_square).card
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card
  (200 - (count_squares + count_cubes - count_sixth_powers)) = 182 := 
by {
  -- Define everything in terms of Finset and filter
  let count_squares := (Finset.range 201).filter is_perfect_square).card,
  let count_cubes := (Finset.range 201).filter is_perfect_cube).card,
  let count_sixth_powers := (Finset.range 201).filter is_sixth_power).card,
  exact sorry
}

end count_not_squares_or_cubes_200_l70_70670


namespace ceiling_and_floor_calculation_l70_70460

theorem ceiling_and_floor_calculation : 
  let a := (15 : ℚ) / 8
  let b := (-34 : ℚ) / 4
  Int.ceil (a * b) - Int.floor (a * Int.floor b) = 2 :=
by
  sorry

end ceiling_and_floor_calculation_l70_70460


namespace profit_percentage_with_discount_is_26_l70_70220

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_without_discount : ℝ := 31.25
noncomputable def discount_percentage : ℝ := 4

noncomputable def selling_price_without_discount : ℝ :=
  cost_price * (1 + profit_percentage_without_discount / 100)

noncomputable def discount : ℝ := 
  discount_percentage / 100 * selling_price_without_discount

noncomputable def selling_price_with_discount : ℝ :=
  selling_price_without_discount - discount

noncomputable def profit_with_discount : ℝ := 
  selling_price_with_discount - cost_price

noncomputable def profit_percentage_with_discount : ℝ := 
  (profit_with_discount / cost_price) * 100

theorem profit_percentage_with_discount_is_26 :
  profit_percentage_with_discount = 26 := by 
  sorry

end profit_percentage_with_discount_is_26_l70_70220


namespace smallest_sum_of_squares_l70_70541

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 217) : 
  x^2 + y^2 ≥ 505 :=
sorry

end smallest_sum_of_squares_l70_70541


namespace mark_eggs_supply_l70_70280

theorem mark_eggs_supply (dozen_eggs: ℕ) (days_in_week: ℕ) : dozen_eggs = 12 → 
  days_in_week = 7 → (5 * dozen_eggs + 30) * days_in_week = 630 :=
by 
  intros h_dozen h_days;
  rw [h_dozen, h_days];
  simp;
  norm_num;
  exact rfl

end mark_eggs_supply_l70_70280


namespace numbers_not_perfect_squares_or_cubes_l70_70676

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l70_70676


namespace complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l70_70098

theorem complete_even_square_diff_eqn : (10^2 - 8^2 = 4 * 9) :=
by sorry

theorem even_square_diff_multiple_of_four (n : ℕ) : (4 * (n + 1) * (n + 1) - 4 * n * n) % 4 = 0 :=
by sorry

theorem odd_square_diff_multiple_of_eight (m : ℕ) : ((2 * m + 1)^2 - (2 * m - 1)^2) % 8 = 0 :=
by sorry

end complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l70_70098


namespace optimalSelectionUsesGoldenRatio_l70_70334

-- Definitions translated from conditions
def optimalSelectionMethodConcept :=
  "The method used by Hua Luogeng in optimal selection method"

def goldenRatio :=
  "A well-known mathematical constant, approximately equal to 1.618033988749895"

-- The theorem statement formulated from the problem
theorem optimalSelectionUsesGoldenRatio :
  optimalSelectionMethodConcept = goldenRatio := 
sorry

end optimalSelectionUsesGoldenRatio_l70_70334


namespace coefficient_fifth_term_expansion_l70_70798

theorem coefficient_fifth_term_expansion :
  let a := (2 : ℝ)
  let b := -(1 : ℝ)
  let n := 6
  let k := 4
  Nat.choose n k * (a ^ (n - k)) * (b ^ k) = 60 := by
  -- We can assume x to be any nonzero real, but it is not needed in the theorem itself.
  sorry

end coefficient_fifth_term_expansion_l70_70798


namespace scientific_notation_of_274000000_l70_70996

theorem scientific_notation_of_274000000 :
  (274000000 : ℝ) = 2.74 * 10 ^ 8 :=
by
    sorry

end scientific_notation_of_274000000_l70_70996


namespace sum_three_consecutive_integers_divisible_by_three_l70_70162

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : 1 < a) :
  (a - 1) + a + (a + 1) % 3 = 0 :=
by
  sorry

end sum_three_consecutive_integers_divisible_by_three_l70_70162


namespace problem_l70_70139

noncomputable def g (x : ℝ) : ℝ := 3^x + 2

theorem problem (x : ℝ) : g (x + 1) - g x = 2 * g x - 2 := sorry

end problem_l70_70139


namespace geometric_sequence_from_second_term_l70_70904

open Nat

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- to handle the 0th term which is typically not used here
  | 1 => 1
  | 2 => 2
  | n + 3 => 3 * S (n + 2) - 2 * S (n + 1) -- given recurrence relation

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- Define a_0 as 0 since it's not used in the problem
  | 1 => 1 -- a1
  | n + 2 => S (n + 2) - S (n + 1) -- a_n = S_n - S_(n-1)

theorem geometric_sequence_from_second_term :
  ∀ n ≥ 2, a (n + 1) = 2 * a n := by
  -- Proof step not provided
  sorry

end geometric_sequence_from_second_term_l70_70904


namespace work_complete_in_15_days_l70_70407

theorem work_complete_in_15_days :
  let A_rate := (1 : ℚ) / 20
  let B_rate := (1 : ℚ) / 30
  let C_rate := (1 : ℚ) / 10
  let all_together_rate := A_rate + B_rate + C_rate
  let work_2_days := 2 * all_together_rate
  let B_C_rate := B_rate + C_rate
  let work_next_2_days := 2 * B_C_rate
  let total_work_4_days := work_2_days + work_next_2_days
  let remaining_work := 1 - total_work_4_days
  let B_time := remaining_work / B_rate

  2 + 2 + B_time = 15 :=
by
  sorry

end work_complete_in_15_days_l70_70407


namespace maximize_sqrt_expression_l70_70389

theorem maximize_sqrt_expression :
  let a := Real.sqrt 8
  let b := Real.sqrt 2
  (a + b) > max (max (a - b) (a * b)) (a / b) := by
  sorry

end maximize_sqrt_expression_l70_70389


namespace first_term_of_arithmetic_sequence_l70_70697

theorem first_term_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ)
  (h1 : a 3 = 3) (h2 : S 9 - S 6 = 27)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d)
  (h4 : ∀ n, S n = n * (a 1 + a n) / 2) : a 1 = 3 / 5 :=
by
  sorry

end first_term_of_arithmetic_sequence_l70_70697


namespace fraction_n_p_l70_70737

theorem fraction_n_p (m n p : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * r2 = m)
  (h2 : -(r1 + r2) = p)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0)
  (h5 : p ≠ 0)
  (h6 : m = - (r1 + r2) / 2)
  (h7 : n = r1 * r2 / 4) :
  n / p = 1 / 8 :=
by
  sorry

end fraction_n_p_l70_70737


namespace problem_l70_70229

-- Definitions based on the provided conditions
def frequency_varies (freq : Real) : Prop := true -- Placeholder definition
def probability_is_stable (prob : Real) : Prop := true -- Placeholder definition
def is_random_event (event : Type) : Prop := true -- Placeholder definition
def is_random_experiment (experiment : Type) : Prop := true -- Placeholder definition
def is_sum_of_events (event1 event2 : Prop) : Prop := event1 ∨ event2 -- Definition of sum of events
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B) -- Definition of mutually exclusive events
def complementary_events (A B : Prop) : Prop := A ↔ ¬B -- Definition of complementary events
def equally_likely_events (events : List Prop) : Prop := true -- Placeholder definition

-- Translation of the questions and correct answers
theorem problem (freq prob : Real) (event experiment : Type) (A B : Prop) (events : List Prop) :
  (¬(frequency_varies freq = probability_is_stable prob)) ∧ -- 1
  ((is_random_event event) ≠ (is_random_experiment experiment)) ∧ -- 2
  (probability_is_stable prob) ∧ -- 3
  (is_sum_of_events A B) ∧ -- 4
  (mutually_exclusive A B → ¬(probability_is_stable (1 - prob))) ∧ -- 5
  (¬(equally_likely_events events)) :=  -- 6
by
  sorry

end problem_l70_70229


namespace equilibrium_and_stability_l70_70991

def system_in_equilibrium (G Q m r : ℝ) : Prop :=
    -- Stability conditions for points A and B, instability at C
    (G < (m-r)/(m-2*r)) ∧ (G > (m-r)/m)

-- Create a theorem to prove the system's equilibrium and stability
theorem equilibrium_and_stability (G Q m r : ℝ) 
  (h_gt_zero : G > 0) 
  (Q_gt_zero : Q > 0) 
  (m_gt_r : m > r) 
  (r_gt_zero : r > 0) : system_in_equilibrium G Q m r :=
by
  sorry   -- Proof omitted

end equilibrium_and_stability_l70_70991


namespace mean_of_six_numbers_l70_70038

theorem mean_of_six_numbers (sum : ℚ) (h : sum = 1/3) : (sum / 6 = 1/18) :=
by
  sorry

end mean_of_six_numbers_l70_70038


namespace drink_costs_l70_70094

theorem drink_costs (cost_of_steak_per_person : ℝ) (total_tip_paid : ℝ) (tip_percentage : ℝ) (billy_tip_coverage_percentage : ℝ) (total_tip_percentage : ℝ) :
  cost_of_steak_per_person = 20 → 
  total_tip_paid = 8 → 
  tip_percentage = 0.20 → 
  billy_tip_coverage_percentage = 0.80 → 
  total_tip_percentage = 0.20 → 
  ∃ (cost_of_drink : ℝ), cost_of_drink = 1.60 :=
by
  intros
  sorry

end drink_costs_l70_70094


namespace annie_initial_money_l70_70227

def cost_of_hamburgers (n : Nat) : Nat := n * 4
def cost_of_milkshakes (m : Nat) : Nat := m * 5
def total_cost (n m : Nat) : Nat := cost_of_hamburgers n + cost_of_milkshakes m
def initial_money (n m left : Nat) : Nat := total_cost n m + left

theorem annie_initial_money : initial_money 8 6 70 = 132 := by
  sorry

end annie_initial_money_l70_70227


namespace complex_number_solution_l70_70490

-- Define that z is a complex number and the condition given in the problem.
theorem complex_number_solution (z : ℂ) (hz : (i / (z + i)) = 2 - i) : z = -1/5 - 3/5 * i :=
sorry

end complex_number_solution_l70_70490


namespace rebus_solution_l70_70940

theorem rebus_solution (A B C D : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (equation : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
sorry

end rebus_solution_l70_70940


namespace numbers_neither_square_nor_cube_l70_70657

theorem numbers_neither_square_nor_cube (n : ℕ) (h : n = 200) :
  let num_squares := 14 in
  let num_cubes := 5 in
  let num_sixth_powers := 2 in
  (n - (num_squares + num_cubes - num_sixth_powers)) = 183 :=
by
  -- Let the number of integers in the range
  let num_total := 200
  -- Define how we count perfect squares and cubes
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2

  -- Subtract the overlapping perfect sixth powers from the sum of squares and cubes
  let num_either_square_or_cube := num_squares + num_cubes - num_sixth_powers

  -- Calculate the final result
  let result := num_total - num_either_square_or_cube

  -- Prove by computation
  show result = 183 from
    calc
      result = num_total - num_either_square_or_cube := rfl
           ... = 200 - (14 + 5 - 2)                   := rfl
           ... = 200 - 17                             := rfl
           ... = 183                                  := rfl

end numbers_neither_square_nor_cube_l70_70657


namespace BKINGTON_appears_first_on_eighth_line_l70_70543

-- Define the cycle lengths for letters and digits
def cycle_letters : ℕ := 8
def cycle_digits : ℕ := 4

-- Define the problem statement
theorem BKINGTON_appears_first_on_eighth_line :
  Nat.lcm cycle_letters cycle_digits = 8 := by
  sorry

end BKINGTON_appears_first_on_eighth_line_l70_70543


namespace bug_total_distance_l70_70768

theorem bug_total_distance 
  (p₀ p₁ p₂ p₃ : ℤ) 
  (h₀ : p₀ = 0) 
  (h₁ : p₁ = 4) 
  (h₂ : p₂ = -3) 
  (h₃ : p₃ = 7) : 
  |p₁ - p₀| + |p₂ - p₁| + |p₃ - p₂| = 21 :=
by 
  sorry

end bug_total_distance_l70_70768


namespace min_girls_in_class_l70_70380

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end min_girls_in_class_l70_70380


namespace number_of_zeros_at_end_of_factorial_30_l70_70096

-- Lean statement for the equivalence proof problem
def count_factors_of (p n : Nat) : Nat :=
  n / p + n / (p * p) + n / (p * p * p) + n / (p * p * p * p) + n / (p * p * p * p * p)

def zeros_at_end_of_factorial (n : Nat) : Nat :=
  count_factors_of 5 n

theorem number_of_zeros_at_end_of_factorial_30 : zeros_at_end_of_factorial 30 = 7 :=
by 
  sorry

end number_of_zeros_at_end_of_factorial_30_l70_70096


namespace sqrt_meaningful_range_l70_70982

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 3) : 3 ≤ x :=
by
  linarith

end sqrt_meaningful_range_l70_70982


namespace independent_and_dependent_variables_l70_70047

variable (R V : ℝ)

theorem independent_and_dependent_variables (h : V = (4 / 3) * Real.pi * R^3) :
  (∃ R : ℝ, ∀ V : ℝ, V = (4 / 3) * Real.pi * R^3) ∧ (∃ V : ℝ, ∃ R' : ℝ, V = (4 / 3) * Real.pi * R'^3) :=
by
  sorry

end independent_and_dependent_variables_l70_70047


namespace books_withdrawn_is_15_l70_70194

-- Define the initial condition
def initial_books : ℕ := 250

-- Define the books taken out on Tuesday
def books_taken_out_tuesday : ℕ := 120

-- Define the books returned on Wednesday
def books_returned_wednesday : ℕ := 35

-- Define the books left in library on Thursday
def books_left_thursday : ℕ := 150

-- Define the problem: Determine the number of books withdrawn on Thursday
def books_withdrawn_thursday : ℕ :=
  (initial_books - books_taken_out_tuesday + books_returned_wednesday) - books_left_thursday

-- The statement we want to prove
theorem books_withdrawn_is_15 : books_withdrawn_thursday = 15 := by sorry

end books_withdrawn_is_15_l70_70194


namespace magician_ball_count_l70_70842

theorem magician_ball_count (k : ℕ) : ∃ k : ℕ, 6 * k + 7 = 1993 :=
by sorry

end magician_ball_count_l70_70842


namespace rebus_solution_l70_70941

theorem rebus_solution (A B C D : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (equation : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
sorry

end rebus_solution_l70_70941


namespace num_times_teams_face_each_other_l70_70193

-- Conditions
variable (teams games total_games : ℕ)
variable (k : ℕ)
variable (h1 : teams = 17)
variable (h2 : games = teams * (teams - 1) * k / 2)
variable (h3 : total_games = 1360)

-- Proof problem
theorem num_times_teams_face_each_other : k = 5 := 
by 
  sorry

end num_times_teams_face_each_other_l70_70193


namespace find_third_number_l70_70906

noncomputable def third_number := 9.110300000000005

theorem find_third_number :
  12.1212 + 17.0005 - third_number = 20.011399999999995 :=
sorry

end find_third_number_l70_70906


namespace sampling_method_sequential_is_systematic_l70_70688

def is_sequential_ids (ids : List Nat) : Prop :=
  ids = [5, 10, 15, 20, 25, 30, 35, 40]

def is_systematic_sampling (sampling_method : Prop) : Prop :=
  sampling_method

theorem sampling_method_sequential_is_systematic :
  ∀ ids, is_sequential_ids ids → 
    is_systematic_sampling (ids = [5, 10, 15, 20, 25, 30, 35, 40]) :=
by
  intros
  apply id
  sorry

end sampling_method_sequential_is_systematic_l70_70688


namespace scientific_notation_of_274000000_l70_70994

theorem scientific_notation_of_274000000 :
  (274000000 : ℝ) = 2.74 * 10 ^ 8 :=
by
    sorry

end scientific_notation_of_274000000_l70_70994


namespace find_middle_and_oldest_sons_l70_70084

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end find_middle_and_oldest_sons_l70_70084


namespace rectangular_garden_width_l70_70030

variable (w : ℕ)

/-- The length of a rectangular garden is three times its width.
Given that the area of the rectangular garden is 768 square meters,
prove that the width of the garden is 16 meters. -/
theorem rectangular_garden_width
  (h1 : 768 = w * (3 * w)) :
  w = 16 := by
  sorry

end rectangular_garden_width_l70_70030


namespace rebus_solution_l70_70942

theorem rebus_solution (A B C D : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (equation : 1001 * A + 100 * B + 10 * C + A = 182 * (10 * C + D)) :
  1000 * A + 100 * B + 10 * C + D = 2916 :=
sorry

end rebus_solution_l70_70942


namespace tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l70_70576

-- First proof problem
theorem tan_theta_eq2_simplifies_to_minus1 (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin (θ - 6 * Real.pi) + Real.sin (Real.pi / 2 - θ)) / 
  (2 * Real.sin (Real.pi + θ) + Real.cos (-θ)) = -1 := sorry

-- Second proof problem
theorem sin_cos_and_tan_relation (x : ℝ) (hx1 : - Real.pi / 2 < x) (hx2 : x < Real.pi / 2) 
  (h : Real.sin x + Real.cos x = 1 / 5) : Real.tan x = -3 / 4 := sorry

end tan_theta_eq2_simplifies_to_minus1_sin_cos_and_tan_relation_l70_70576


namespace are_naptime_l70_70784

def flight_duration := 11 * 60 + 20  -- in minutes

def time_spent_reading := 2 * 60      -- in minutes
def time_spent_watching_movies := 4 * 60  -- in minutes
def time_spent_eating_dinner := 30    -- in minutes
def time_spent_listening_to_radio := 40   -- in minutes
def time_spent_playing_games := 1 * 60 + 10   -- in minutes

def total_time_spent_on_activities := 
  time_spent_reading + 
  time_spent_watching_movies + 
  time_spent_eating_dinner + 
  time_spent_listening_to_radio + 
  time_spent_playing_games

def remaining_time := (flight_duration - total_time_spent_on_activities) / 60  -- in hours

theorem are_naptime : remaining_time = 3 := by
  sorry

end are_naptime_l70_70784


namespace area_of_given_triangle_l70_70431

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_given_triangle :
  area_of_triangle (-2) 3 7 (-3) 4 6 = 31.5 :=
by
  sorry

end area_of_given_triangle_l70_70431


namespace ice_cream_ratio_l70_70789

theorem ice_cream_ratio :
  ∃ (B C : ℕ), 
    C = 1 ∧
    (∃ (W D : ℕ), 
      D = 2 ∧
      W = B + 1 ∧
      B + W + C + D = 10 ∧
      B / C = 3
    ) := sorry

end ice_cream_ratio_l70_70789


namespace math_problem_l70_70463

theorem math_problem :
  (Int.ceil ((15: ℚ) / 8 * (-34: ℚ) / 4) - Int.floor ((15: ℚ) / 8 * Int.floor ((-34: ℚ) / 4))) = 2 :=
by
  sorry

end math_problem_l70_70463


namespace rows_seat_7_students_are_5_l70_70613

-- Definitions based on provided conditions
def total_students : Nat := 53
def total_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_students = 6 * six_seat_rows + 7 * seven_seat_rows

-- To prove the number of rows seating exactly 7 students is 5
def number_of_7_seat_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_rows six_seat_rows seven_seat_rows ∧ seven_seat_rows = 5

-- Statement to be proved
theorem rows_seat_7_students_are_5 : ∃ (six_seat_rows seven_seat_rows : Nat), number_of_7_seat_rows six_seat_rows seven_seat_rows := 
by
  -- Skipping the proof
  sorry

end rows_seat_7_students_are_5_l70_70613


namespace problem_l70_70826

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 4)

theorem problem
  (h1 : f (Real.pi / 8) = 2)
  (h2 : f (5 * Real.pi / 8) = -2) :
  (∀ x : ℝ, f x = 1 ↔ 
    (∃ k : ℤ, x = -Real.pi / 24 + k * Real.pi) ∨
    (∃ k : ℤ, x = 7 * Real.pi / 24 + k * Real.pi)) :=
by
  sorry

end problem_l70_70826


namespace min_attendees_l70_70741

-- Define the constants and conditions
def writers : ℕ := 35
def min_editors : ℕ := 39
def x_max : ℕ := 26

-- Define the total number of people formula based on inclusion-exclusion principle
-- and conditions provided
def total_people (x : ℕ) : ℕ := writers + min_editors - x + 2 * x

-- Theorem to prove that the minimum number of attendees is 126
theorem min_attendees : ∃ x, x ≤ x_max ∧ total_people x = 126 :=
by
  use x_max
  sorry

end min_attendees_l70_70741


namespace min_girls_in_class_l70_70375

theorem min_girls_in_class (n : ℕ) (d : ℕ) (h1 : n = 20) 
  (h2 : ∀ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
       ∀ f : ℕ → set ℕ, (card (f i) ≠ card (f j) ∨ card (f j) ≠ card (f k) ∨ card (f i) ≠ card (f k))) : 
d ≥ 6 :=
by
  sorry

end min_girls_in_class_l70_70375


namespace optimal_selection_method_uses_golden_ratio_l70_70304

def optimalSelectionUsesGoldenRatio : Prop := 
  "The optimal selection method uses the golden ratio."

theorem optimal_selection_method_uses_golden_ratio :
  optimalSelectionUsesGoldenRatio := 
begin
  sorry
end

end optimal_selection_method_uses_golden_ratio_l70_70304


namespace minimum_girls_in_class_l70_70371

theorem minimum_girls_in_class (n : ℕ) (d : ℕ) (boys : List (List ℕ)) 
  (h_students : n = 20)
  (h_boys : boys.length = n - d)
  (h_unique : ∀ i j k : ℕ, i < boys.length → j < boys.length → k < boys.length → i ≠ j → j ≠ k → i ≠ k → 
    (boys.nthLe i (by linarith)).length ≠ (boys.nthLe j (by linarith)).length ∨ 
    (boys.nthLe j (by linarith)).length ≠ (boys.nthLe k (by linarith)).length ∨ 
    (boys.nthLe k (by linarith)).length ≠ (boys.nthLe i (by linarith)).length) :
  d = 6 := 
begin
  sorry
end

end minimum_girls_in_class_l70_70371


namespace zeros_of_f_x_minus_1_l70_70247

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_f_x_minus_1 :
  (f (0 - 1) = 0) ∧ (f (2 - 1) = 0) :=
by
  sorry

end zeros_of_f_x_minus_1_l70_70247


namespace num_from_1_to_200_not_squares_or_cubes_l70_70665

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l70_70665


namespace parallel_vectors_implies_x_l70_70133

-- a definition of the vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

-- a definition for vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- a definition for scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- a definition for vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- the theorem statement
theorem parallel_vectors_implies_x (x : ℝ) (h : 
  vector_add vector_a (vector_b x) = ⟨3, 1 + x⟩ ∧
  vector_sub (scalar_mul 2 vector_a) (vector_b x) = ⟨3, 2 - x⟩ ∧
  ∃ k : ℝ, vector_add vector_a (vector_b x) = scalar_mul k (vector_sub (scalar_mul 2 vector_a) (vector_b x))
  ) : x = 1 / 2 :=
sorry

end parallel_vectors_implies_x_l70_70133


namespace fraction_transformation_l70_70915

theorem fraction_transformation (a b : ℝ) (h : a ≠ b) : 
  (-a) / (a - b) = a / (b - a) :=
sorry

end fraction_transformation_l70_70915


namespace sum_of_legs_of_larger_triangle_l70_70056

theorem sum_of_legs_of_larger_triangle 
  (area_small area_large : ℝ)
  (hypotenuse_small : ℝ)
  (A : area_small = 10)
  (B : area_large = 250)
  (C : hypotenuse_small = 13) : 
  ∃ a b : ℝ, (a + b = 35) := 
sorry

end sum_of_legs_of_larger_triangle_l70_70056


namespace distribution_schemes_count_l70_70299

def num_distribution_schemes (volunteers pavilions : ℕ) : ℕ :=
  if volunteers = 5 ∧ pavilions = 3 then
    3 * (Nat.choose 5 2)
  else
    0

theorem distribution_schemes_count :
  num_distribution_schemes 5 3 = 30 := by
  simp [num_distribution_schemes, Nat.choose]
  sorry

end distribution_schemes_count_l70_70299


namespace fractional_equation_solution_l70_70823

theorem fractional_equation_solution (m : ℝ) (x : ℝ) :
  (m + 3) / (x - 1) = 1 → x > 0 → m > -4 ∧ m ≠ -3 :=
by
  sorry

end fractional_equation_solution_l70_70823


namespace fraction_of_water_in_mixture_l70_70153

theorem fraction_of_water_in_mixture (r : ℚ) (h : r = 2 / 3) : (3 / (2 + 3) : ℚ) = 3 / 5 :=
by
  sorry

end fraction_of_water_in_mixture_l70_70153


namespace count_not_squares_or_cubes_l70_70653

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l70_70653


namespace count_not_squares_or_cubes_l70_70654

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l70_70654


namespace incircle_tangent_distance_l70_70522

theorem incircle_tangent_distance (a b c : ℝ) (M : ℝ) (BM : ℝ) (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : BM = y1 + z1)
  (h2 : BM = y2 + z2)
  (h3 : x1 + y1 = x2 + y2)
  (h4 : x1 + z1 = c)
  (h5 : x2 + z2 = a) :
  |y1 - y2| = |(a - c) / 2| := by 
  sorry

end incircle_tangent_distance_l70_70522


namespace numbers_not_perfect_squares_or_cubes_l70_70677

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end numbers_not_perfect_squares_or_cubes_l70_70677


namespace find_x7_l70_70189

-- Definitions for the conditions
def seq (x : ℕ → ℕ) : Prop :=
  (x 6 = 144) ∧ ∀ n, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → (x (n + 3) = x (n + 2) * (x (n + 1) + x n))

-- Theorem statement to prove x_7 = 3456
theorem find_x7 (x : ℕ → ℕ) (h : seq x) : x 7 = 3456 := sorry

end find_x7_l70_70189


namespace solve_diamond_l70_70971

theorem solve_diamond (d : ℕ) (h : d * 6 + 5 = d * 7 + 2) : d = 3 :=
by
  sorry

end solve_diamond_l70_70971


namespace diamond_more_olivine_l70_70595

theorem diamond_more_olivine :
  ∃ A O D : ℕ, A = 30 ∧ O = A + 5 ∧ A + O + D = 111 ∧ D - O = 11 :=
by
  sorry

end diamond_more_olivine_l70_70595


namespace max_possible_cities_traversed_l70_70504

theorem max_possible_cities_traversed
    (cities : Finset (Fin 110))
    (roads : Finset (Fin 110 × Fin 110))
    (degree : Fin 110 → ℕ)
    (h1 : ∀ c ∈ cities, (degree c) = (roads.filter (λ r, r.1 = c ∨ r.2 = c)).card)
    (h2 : ∃ start : Fin 110, (degree start) = 1)
    (h3 : ∀ (n : ℕ) (i : Fin 110), n > 1 → (degree i) = n → ∃ j : Fin 110, (degree j) = n + 1)
    : ∃ N : ℕ, N ≤ 107 :=
begin
  sorry
end

end max_possible_cities_traversed_l70_70504


namespace product_of_three_3_digits_has_four_zeros_l70_70607

noncomputable def has_four_zeros_product : Prop :=
  ∃ (a b c: ℕ),
    (100 ≤ a ∧ a < 1000) ∧
    (100 ≤ b ∧ b < 1000) ∧
    (100 ≤ c ∧ c < 1000) ∧
    (∃ (da db dc: Finset ℕ), (da ∪ db ∪ dc = Finset.range 10) ∧
    (∀ x ∈ da, x = a / 10^(x%10) % 10) ∧
    (∀ x ∈ db, x = b / 10^(x%10) % 10) ∧
    (∀ x ∈ dc, x = c / 10^(x%10) % 10)) ∧
    (a * b * c % 10000 = 0)

theorem product_of_three_3_digits_has_four_zeros : has_four_zeros_product := sorry

end product_of_three_3_digits_has_four_zeros_l70_70607


namespace ratio_of_products_l70_70141

variable (a b c d : ℚ) -- assuming a, b, c, d are rational numbers

theorem ratio_of_products (h1 : a = 3 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end ratio_of_products_l70_70141


namespace non_perfect_powers_count_l70_70662

theorem non_perfect_powers_count :
  let n := 200,
      num_perfect_squares := 14,
      num_perfect_cubes := 5,
      num_perfect_sixth_powers := 1,
      total_perfect_powers := num_perfect_squares + num_perfect_cubes - num_perfect_sixth_powers,
      total_non_perfect_powers := n - total_perfect_powers
  in
  total_non_perfect_powers = 182 :=
by
  sorry

end non_perfect_powers_count_l70_70662


namespace abs_ineq_solution_set_l70_70958

theorem abs_ineq_solution_set {x : ℝ} : |x + 1| - |x - 3| ≥ 2 ↔ x ≥ 2 :=
by
  sorry

end abs_ineq_solution_set_l70_70958


namespace number_of_soccer_balls_in_first_set_l70_70411

noncomputable def cost_of_soccer_ball : ℕ := 50
noncomputable def first_cost_condition (F c : ℕ) : Prop := 3 * F + c = 155
noncomputable def second_cost_condition (F : ℕ) : Prop := 2 * F + 3 * cost_of_soccer_ball = 220

theorem number_of_soccer_balls_in_first_set (F : ℕ) :
  (first_cost_condition F 50) ∧ (second_cost_condition F) → 1 = 1 :=
by
  sorry

end number_of_soccer_balls_in_first_set_l70_70411


namespace optimal_selection_uses_golden_ratio_l70_70355

-- Define the statement that Hua Luogeng made contributions to optimal selection methods
def hua_luogeng_contribution : Prop := 
  ∃ M : Type, ∃ f : M → (String × ℝ) → Prop, 
  (∀ x : M, f x ("Golden ratio", 1.618)) ∧
  (∀ x : M, f x ("Mean", _)) ∨ f x ("Mode", _) ∨ f x ("Median", _)

-- Define the theorem that proves the optimal selection method uses the Golden ratio
theorem optimal_selection_uses_golden_ratio : hua_luogeng_contribution → ∃ x : Type, ∃ y : x → (String × ℝ) → Prop, y x ("Golden ratio", 1.618) :=
by 
  sorry

end optimal_selection_uses_golden_ratio_l70_70355


namespace calories_burned_l70_70874

/-- 
  The football coach makes his players run up and down the bleachers 60 times. 
  Each time they run up and down, they encounter 45 stairs. 
  The first half of the staircase has 20 stairs and every stair burns 3 calories, 
  while the second half has 25 stairs burning 4 calories each. 
  Prove that each player burns 9600 calories during this exercise.
--/
theorem calories_burned (n_stairs_first_half : ℕ) (calories_first_half : ℕ) 
  (n_stairs_second_half : ℕ) (calories_second_half : ℕ) (n_trips : ℕ) 
  (total_calories : ℕ) :
  n_stairs_first_half = 20 → calories_first_half = 3 → 
  n_stairs_second_half = 25 → calories_second_half = 4 → 
  n_trips = 60 → total_calories = 
  (n_stairs_first_half * calories_first_half + n_stairs_second_half * calories_second_half) * n_trips →
  total_calories = 9600 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end calories_burned_l70_70874


namespace percentage_loss_is_correct_l70_70286

noncomputable def initial_cost : ℝ := 300
noncomputable def selling_price : ℝ := 255
noncomputable def loss : ℝ := initial_cost - selling_price
noncomputable def percentage_loss : ℝ := (loss / initial_cost) * 100

theorem percentage_loss_is_correct :
  percentage_loss = 15 :=
sorry

end percentage_loss_is_correct_l70_70286


namespace largest_constant_C_l70_70478

theorem largest_constant_C (C : ℝ) : C = 2 / Real.sqrt 3 ↔ ∀ (x y z : ℝ), x^2 + y^2 + 2 * z^2 + 1 ≥ C * (x + y + z) :=
by
  sorry

end largest_constant_C_l70_70478


namespace calculate_savings_l70_70439

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l70_70439


namespace binom_n_plus_1_n_minus_1_eq_l70_70751

theorem binom_n_plus_1_n_minus_1_eq (n : ℕ) (h : 0 < n) : (Nat.choose (n + 1) (n - 1)) = n * (n + 1) / 2 := 
by sorry

end binom_n_plus_1_n_minus_1_eq_l70_70751


namespace count_not_squares_or_cubes_l70_70652

theorem count_not_squares_or_cubes {A B : Finset ℕ} (h_range : ∀ x, x ∈ A ∪ B → x ≤ 200)
  (h_squares : A.card = 14) (h_cubes : B.card = 5) (h_inter : (A ∩ B).card = 1) :
  (200 - (A ∪ B).card) = 182 :=
by
  sorry

end count_not_squares_or_cubes_l70_70652


namespace zero_one_sequence_count_l70_70637

theorem zero_one_sequence_count :
  ∑ k in finset.range 6, (nat.choose 11 k) * (nat.choose (11 - k) (10 - 2 * k)) = 24068 := by
  sorry

end zero_one_sequence_count_l70_70637


namespace range_of_a_l70_70964

noncomputable def problem (x y z : ℝ) (a : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y + z = 1) ∧ 
  (a / (x * y * z) = 1/x + 1/y + 1/z - 2) 

theorem range_of_a (x y z a : ℝ) (h : problem x y z a) : 
  0 < a ∧ a ≤ 7/27 :=
sorry

end range_of_a_l70_70964


namespace cow_count_16_l70_70761

theorem cow_count_16 (D C : ℕ) 
  (h1 : ∃ (L H : ℕ), L = 2 * D + 4 * C ∧ H = D + C ∧ L = 2 * H + 32) : C = 16 :=
by
  obtain ⟨L, H, ⟨hL, hH, hCond⟩⟩ := h1
  sorry

end cow_count_16_l70_70761


namespace probability_of_successful_meeting_l70_70422

noncomputable def meeting_probability : ℚ := 7 / 64

theorem probability_of_successful_meeting :
  (∃ x y z : ℝ,
     0 ≤ x ∧ x ≤ 2 ∧
     0 ≤ y ∧ y ≤ 2 ∧
     0 ≤ z ∧ z ≤ 2 ∧
     abs (x - z) ≤ 0.75 ∧
     abs (y - z) ≤ 1.5 ∧
     z ≥ x ∧
     z ≥ y) →
  meeting_probability = 7 / 64 := by
  sorry

end probability_of_successful_meeting_l70_70422


namespace fourth_student_seat_number_l70_70843

theorem fourth_student_seat_number (n : ℕ) (pop_size sample_size : ℕ)
  (s1 s2 s3 : ℕ)
  (h_pop_size : pop_size = 52)
  (h_sample_size : sample_size = 4)
  (h_6_in_sample : s1 = 6)
  (h_32_in_sample : s2 = 32)
  (h_45_in_sample : s3 = 45)
  : ∃ s4 : ℕ, s4 = 19 :=
by
  sorry

end fourth_student_seat_number_l70_70843


namespace sufficient_condition_l70_70893

theorem sufficient_condition (a b c : ℤ) : (a = c + 1) → (b = a - 1) → a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  intros h1 h2
  sorry

end sufficient_condition_l70_70893


namespace ages_of_sons_l70_70080

variable (x y z : ℕ)

def father_age_current : ℕ := 33

def youngest_age_current : ℕ := 2

def father_age_in_12_years : ℕ := father_age_current + 12

def sum_of_ages_in_12_years : ℕ := youngest_age_current + 12 + y + 12 + z + 12

theorem ages_of_sons (x y z : ℕ) 
  (h1 : x = 2)
  (h2 : father_age_current = 33)
  (h3 : father_age_in_12_years = 45)
  (h4 : sum_of_ages_in_12_years = 45) :
  x = 2 ∧ y + z = 7 ∧ ((y = 3 ∧ z = 4) ∨ (y = 4 ∧ z = 3)) :=
by
  sorry

end ages_of_sons_l70_70080


namespace ceil_floor_difference_l70_70456

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l70_70456


namespace overhead_cost_calculation_l70_70020

-- Define the production cost per performance
def production_cost_performance : ℕ := 7000

-- Define the revenue per sold-out performance
def revenue_per_soldout_performance : ℕ := 16000

-- Define the number of performances needed to break even
def break_even_performances : ℕ := 9

-- Prove the overhead cost
theorem overhead_cost_calculation (O : ℕ) :
  (O + break_even_performances * production_cost_performance = break_even_performances * revenue_per_soldout_performance) →
  O = 81000 :=
by
  sorry

end overhead_cost_calculation_l70_70020


namespace probability_of_U_l70_70559

def pinyin : List Char := ['S', 'H', 'U', 'X', 'U', 'E']
def total_letters : Nat := 6
def u_count : Nat := 2

theorem probability_of_U :
  ((u_count : ℚ) / (total_letters : ℚ)) = (1 / 3) :=
by
  sorry

end probability_of_U_l70_70559


namespace profit_achieved_at_50_yuan_l70_70759

theorem profit_achieved_at_50_yuan :
  ∀ (x : ℝ), (30 ≤ x ∧ x ≤ 54) → 
  ((x - 30) * (80 - 2 * (x - 40)) = 1200) →
  x = 50 :=
by
  intros x h_range h_profit
  sorry

end profit_achieved_at_50_yuan_l70_70759


namespace quadratic_equation_terms_l70_70155

theorem quadratic_equation_terms (x : ℝ) :
  (∃ a b c : ℝ, a = 3 ∧ b = -6 ∧ c = -7 ∧ a * x^2 + b * x + c = 0) →
  (∃ (a : ℝ), a = 3 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = a * x^2 - 6 * x - 7) ∧
  (∃ (c : ℝ), c = -7 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = 3 * x^2 - 6 * x + c) :=
by
  sorry

end quadratic_equation_terms_l70_70155


namespace find_T_b_minus_T_neg_b_l70_70629

noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

theorem find_T_b_minus_T_neg_b (b : ℝ) (h1 : -1 < b ∧ b < 1) (h2 : T b * T (-b) = 3240) (h3 : 1 - b^2 = 100 / 810) :
  T b - T (-b) = 324 * b :=
by
  sorry

end find_T_b_minus_T_neg_b_l70_70629


namespace exponent_multiplication_l70_70786

-- Define the core condition: the base 625
def base := 625

-- Define the exponents
def exp1 := 0.08
def exp2 := 0.17
def combined_exp := exp1 + exp2

-- The mathematical goal to prove
theorem exponent_multiplication (b : ℝ) (e1 e2 : ℝ) (h1 : b = 625) (h2 : e1 = 0.08) (h3 : e2 = 0.17) :
  (b ^ e1 * b ^ e2) = 5 :=
by {
  -- Sorry is added to skip the actual proof steps.
  sorry
}

end exponent_multiplication_l70_70786


namespace calculate_savings_l70_70440

theorem calculate_savings :
  let income := 5 * (45000 + 35000 + 7000 + 10000 + 13000),
  let expenses := 5 * (30000 + 10000 + 5000 + 4500 + 9000),
  let initial_savings := 849400
in initial_savings + income - expenses = 1106900 := by sorry

end calculate_savings_l70_70440


namespace highway_length_is_105_l70_70557

-- Define the speeds of the two cars
def speed_car1 : ℝ := 15
def speed_car2 : ℝ := 20

-- Define the time they travel for
def time_travelled : ℝ := 3

-- Define the distances covered by the cars
def distance_car1 : ℝ := speed_car1 * time_travelled
def distance_car2 : ℝ := speed_car2 * time_travelled

-- Define the total length of the highway
def length_highway : ℝ := distance_car1 + distance_car2

-- The theorem statement
theorem highway_length_is_105 : length_highway = 105 :=
by
  -- Skipping the proof for now
  sorry

end highway_length_is_105_l70_70557


namespace min_sum_of_bases_l70_70212

theorem min_sum_of_bases (a b : ℕ) (h : 3 * a + 5 = 4 * b + 2) : a + b = 13 :=
sorry

end min_sum_of_bases_l70_70212


namespace max_N_value_l70_70517

-- Define the structure for the country with cities and roads.
structure City (n : ℕ) where
  num_roads : ℕ

-- Define the list of cities visited by the driver
def visit_cities (n : ℕ) : List (City n) :=
  List.range' 1 (n + 1) |>.map (λ k => ⟨k⟩)

-- Define the main property proving the maximum possible value of N
theorem max_N_value (n : ℕ) (cities : List (City n)) :
  (∀ (k : ℕ), 2 ≤ k → k ≤ n → City.num_roads ((visit_cities n).get (k - 1)) = k)
  → n ≤ 107 :=
by
  sorry

end max_N_value_l70_70517


namespace total_walnut_trees_in_park_l70_70049

theorem total_walnut_trees_in_park 
  (initial_trees planted_by_first planted_by_second planted_by_third removed_trees : ℕ)
  (h_initial : initial_trees = 22)
  (h_first : planted_by_first = 12)
  (h_second : planted_by_second = 15)
  (h_third : planted_by_third = 10)
  (h_removed : removed_trees = 4) :
  initial_trees + (planted_by_first + planted_by_second + planted_by_third - removed_trees) = 55 :=
by
  sorry

end total_walnut_trees_in_park_l70_70049


namespace f_diff_ineq_l70_70529

variable {f : ℝ → ℝ}
variable (deriv_f : ∀ x > 0, x * (deriv f x) > 1)

theorem f_diff_ineq (h : ∀ x > 0, x * (deriv f x) > 1) : f 2 - f 1 > Real.log 2 := by 
  sorry

end f_diff_ineq_l70_70529


namespace boys_without_pencils_l70_70685

variable (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)

theorem boys_without_pencils
  (h1 : total_boys = 18)
  (h2 : students_with_pencils = 25)
  (h3 : girls_with_pencils = 15)
  (h4 : total_students = 30) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by
  sorry

end boys_without_pencils_l70_70685


namespace optimal_selection_method_is_golden_ratio_l70_70345

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l70_70345


namespace expiry_time_correct_l70_70405

def factorial (n : Nat) : Nat := match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

def seconds_in_a_day : Nat := 86400
def seconds_in_an_hour : Nat := 3600
def donation_time_seconds : Nat := 8 * seconds_in_an_hour
def expiry_seconds : Nat := factorial 8

def time_of_expiry (donation_time : Nat) (expiry_time : Nat) : Nat :=
  (donation_time + expiry_time) % seconds_in_a_day

def time_to_HM (time_seconds : Nat) : Nat × Nat :=
  let hours := time_seconds / seconds_in_an_hour
  let minutes := (time_seconds % seconds_in_an_hour) / 60
  (hours, minutes)

def is_correct_expiry_time : Prop :=
  let (hours, minutes) := time_to_HM (time_of_expiry donation_time_seconds expiry_seconds)
  hours = 19 ∧ minutes = 12

theorem expiry_time_correct : is_correct_expiry_time := by
  sorry

end expiry_time_correct_l70_70405


namespace ceil_floor_difference_l70_70454

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l70_70454


namespace area_transformation_l70_70021

variable {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x, g x = 20) : ∫ x, -4 * g (x + 3) = 80 := by
  sorry

end area_transformation_l70_70021


namespace intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l70_70977

variable (P Q : Prop)

theorem intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false
  (h : P ∧ Q = False) : (P ∨ Q = False) ↔ (P ∧ Q = False) := 
by 
  sorry

end intersection_of_P_and_Q_is_false_iff_union_of_P_and_Q_is_false_l70_70977


namespace curve_is_circle_l70_70956

theorem curve_is_circle (r θ : ℝ) (h : r = 3 * Real.sin θ) : 
  ∃ c : ℝ × ℝ, c = (0, 3 / 2) ∧ ∀ p : ℝ × ℝ, ∃ R : ℝ, R = 3 / 2 ∧ 
  (p.1 - c.1)^2 + (p.2 - c.2)^2 = R^2 :=
sorry

end curve_is_circle_l70_70956


namespace max_geometric_sequence_sum_l70_70188

theorem max_geometric_sequence_sum (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a * b * c = 216) (h4 : ∃ r : ℕ, b = a * r ∧ c = b * r) : 
  a + b + c ≤ 43 :=
sorry

end max_geometric_sequence_sum_l70_70188


namespace optimal_selection_uses_golden_ratio_l70_70325

/-- The famous Chinese mathematician Hua Luogeng made important contributions to popularizing the optimal selection method.
    One of the methods in the optimal selection method uses the golden ratio. -/
theorem optimal_selection_uses_golden_ratio 
  (Hua_Luogeng_contributions : ∀ {method}, method ∈ optimal_selection_method → method = golden_ratio) :
  ∃ method ∈ optimal_selection_method, method = golden_ratio :=
by {
  existsi golden_ratio,
  split,
  sorry,
}

end optimal_selection_uses_golden_ratio_l70_70325


namespace intersection_of_sets_l70_70487

def setA : Set ℝ := {x | x^2 < 8}
def setB : Set ℝ := {x | 1 - x ≤ 0}
def setIntersection : Set ℝ := {x | x ∈ setA ∧ x ∈ setB}

theorem intersection_of_sets :
    setIntersection = {x | 1 ≤ x ∧ x < 2 * Real.sqrt 2} :=
by
  sorry

end intersection_of_sets_l70_70487


namespace cake_sugar_calculation_l70_70604

theorem cake_sugar_calculation (sugar_first_layer : ℕ) (sugar_second_layer : ℕ) (sugar_third_layer : ℕ) :
  sugar_first_layer = 2 →
  sugar_second_layer = 2 * sugar_first_layer →
  sugar_third_layer = 3 * sugar_second_layer →
  sugar_third_layer = 12 := 
by
  intros h1 h2 h3
  have h4 : 2 = sugar_first_layer, from h1.symm
  have h5 : sugar_second_layer = 2 * 2, by rw [h4, h2]
  have h6 : sugar_third_layer = 3 * 4, by rw [h5, h3]
  exact h6

end cake_sugar_calculation_l70_70604


namespace sum_coordinates_is_60_l70_70386

theorem sum_coordinates_is_60 :
  let points := [(5 + Real.sqrt 91, 13), (5 - Real.sqrt 91, 13), (5 + Real.sqrt 91, 7), (5 - Real.sqrt 91, 7)]
  let x_coords_sum := (5 + Real.sqrt 91) + (5 - Real.sqrt 91) + (5 + Real.sqrt 91) + (5 - Real.sqrt 91)
  let y_coords_sum := 13 + 13 + 7 + 7
  x_coords_sum + y_coords_sum = 60 :=
by
  sorry

end sum_coordinates_is_60_l70_70386


namespace max_value_of_N_l70_70512

theorem max_value_of_N (N : ℕ) (cities : Finset ℕ) (roads : ℕ → Finset ℕ → Prop)
  (initial_city : ℕ) (num_cities : cities.card = 110)
  (start_city_road : ∀ city ∈ cities, city = initial_city → (roads initial_city cities).card = 1)
  (nth_city_road : ∀ (k : ℕ), 2 ≤ k → k ≤ N → ∃ city ∈ cities, (roads city cities).card = k) :
  N ≤ 107 := sorry

end max_value_of_N_l70_70512


namespace house_paint_possible_l70_70693
open Function

def family_perms_exist (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] : Prop :=
  ∃ (perm : Perm families), ∀ f : families, f ≠ perm f

def colorable (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] : Prop :=
  ∃ (colors : houses → ℕ), (∀ h : houses, colors h = 0 ∨ colors h = 1 ∨ colors h = 2) ∧
  ∀ (perm : Perm houses), ∀ h : houses, colors h ≠ colors (perm h)

theorem house_paint_possible (families : Type) (houses : Type) [Fintype families] [Fintype houses] [DecidableEq houses] :
  family_perms_exist families houses → colorable families houses :=
by
  sorry

end house_paint_possible_l70_70693


namespace eight_diamond_five_l70_70547

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem eight_diamond_five : diamond 8 5 = 160 :=
by sorry

end eight_diamond_five_l70_70547


namespace complementary_angle_problem_l70_70877

theorem complementary_angle_problem 
  (A B : ℝ) 
  (h1 : A + B = 90) 
  (h2 : A / B = 2 / 3) 
  (increase : A' = A * 1.20) 
  (new_sum : A' + B' = 90) 
  (B' : ℝ)
  (h3 : B' = B - B * 0.1333) :
  true := 
sorry

end complementary_angle_problem_l70_70877


namespace num_from_1_to_200_not_squares_or_cubes_l70_70664

noncomputable def numNonPerfectSquaresAndCubes (n : ℕ) : ℕ :=
  let num_squares := 14
  let num_cubes := 5
  let num_sixth_powers := 2
  n - (num_squares + num_cubes - num_sixth_powers)

theorem num_from_1_to_200_not_squares_or_cubes : numNonPerfectSquaresAndCubes 200 = 183 := by
  sorry

end num_from_1_to_200_not_squares_or_cubes_l70_70664


namespace maximum_N_value_l70_70509

theorem maximum_N_value (N : ℕ) (cities : Fin 110 → List (Fin 110)) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ N → 
    List.length (cities ⟨k-1, by linarith⟩) = k) →
  (∀ i j : Fin 110, i ≠ j → (∃ r : ℕ, (r ∈ cities i) ∨ (r ∈ cities j) ∨ (r ≠ i ∧ r ≠ j))) →
  N ≤ 107 :=
sorry

end maximum_N_value_l70_70509


namespace qualified_weight_example_l70_70029

-- Define the range of qualified weights
def is_qualified_weight (w : ℝ) : Prop :=
  9.9 ≤ w ∧ w ≤ 10.1

-- State the problem: show that 10 kg is within the qualified range
theorem qualified_weight_example : is_qualified_weight 10 :=
  by
    sorry

end qualified_weight_example_l70_70029


namespace bus_speeds_l70_70748

theorem bus_speeds (d t : ℝ) (s₁ s₂ : ℝ)
  (h₀ : d = 48)
  (h₁ : t = 1 / 6) -- 10 minutes in hours
  (h₂ : s₂ = s₁ - 4)
  (h₃ : d / s₂ - d / s₁ = t) :
  s₁ = 36 ∧ s₂ = 32 := 
sorry

end bus_speeds_l70_70748


namespace max_possible_N_in_cities_l70_70505

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l70_70505


namespace tan_2x_is_odd_l70_70730

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_2x_is_odd (x : ℝ) (k : ℤ) (h : x ≠ (1/2) * k * Real.pi + Real.pi / 4) :
  f (-x) = -f x :=
by
  rw [f, f]
  simp
  have a : -2 * x = (-2) * x, ring
  rw [a, Real.tan_neg]
  refl

end tan_2x_is_odd_l70_70730


namespace robin_has_43_packages_of_gum_l70_70718

theorem robin_has_43_packages_of_gum (P : ℕ) (h1 : 23 * P + 8 = 997) : P = 43 :=
by
  sorry

end robin_has_43_packages_of_gum_l70_70718


namespace optimal_selection_method_is_golden_ratio_l70_70343

def optimal_selection_method_uses_golden_ratio 
  (popularized_by_hua_luogeng : Prop) 
  (uses_specific_concept_for_optimization : Prop) : 
  uses_specific_concept_for_optimization → Prop :=
  popularized_by_hua_luogeng → 
  uses_specific_concept_for_optimization = True

axiom hua_luogeng_contribution 
  : optimal_selection_method_uses_golden_ratio True True

theorem optimal_selection_method_is_golden_ratio
  : ∃ c : Prop, c = True ∧ hua_luogeng_contribution := 
by 
  exact ⟨True, by trivial, hua_luogeng_contribution⟩

end optimal_selection_method_is_golden_ratio_l70_70343


namespace number_of_correct_statements_l70_70292

-- Definitions of the conditions from the problem
def seq_is_graphical_points := true  -- Statement 1
def seq_is_finite (s : ℕ → ℝ) := ∀ n, s n = 0 -- Statement 2
def seq_decreasing_implies_finite (s : ℕ → ℝ) := (∀ n, s (n + 1) ≤ s n) → seq_is_finite s -- Statement 3

-- Prove the number of correct statements is 1
theorem number_of_correct_statements : (seq_is_graphical_points = true ∧ ¬(∃ s: ℕ → ℝ, ¬seq_is_finite s) ∧ ∃ s : ℕ → ℝ, ¬seq_decreasing_implies_finite s) → 1 = 1 :=
by
  sorry

end number_of_correct_statements_l70_70292


namespace find_angle_A_l70_70840

theorem find_angle_A (BC AC : ℝ) (B : ℝ) (A : ℝ) (h_cond : BC = Real.sqrt 3 ∧ AC = 1 ∧ B = Real.pi / 6) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_angle_A_l70_70840


namespace three_digit_number_multiple_of_eleven_l70_70619

theorem three_digit_number_multiple_of_eleven:
  ∃ (a b c : ℕ), (1 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧ (0 ≤ c) ∧ (c ≤ 9) ∧
                  (100 * a + 10 * b + c = 11 * (a + b + c) ∧ (100 * a + 10 * b + c = 198)) :=
by
  use 1
  use 9
  use 8
  sorry

end three_digit_number_multiple_of_eleven_l70_70619


namespace find_number_l70_70232

theorem find_number (n : ℕ) :
  (n % 12 = 11) ∧
  (n % 11 = 10) ∧
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1)
  → n = 27719 := 
sorry

end find_number_l70_70232


namespace no_girl_can_avoid_losing_bet_l70_70388

theorem no_girl_can_avoid_losing_bet
  (G1 G2 G3 : Prop)
  (h1 : G1 ↔ ¬G2)
  (h2 : G2 ↔ ¬G3)
  (h3 : G3 ↔ ¬G1)
  : G1 ∧ G2 ∧ G3 → False := by
  sorry

end no_girl_can_avoid_losing_bet_l70_70388


namespace calculate_expression_l70_70787

theorem calculate_expression :
  2 * (-1 / 4) - |1 - Real.sqrt 3| + (-2023)^0 = 3 / 2 - Real.sqrt 3 :=
by
  sorry

end calculate_expression_l70_70787


namespace crackers_initial_count_l70_70531

theorem crackers_initial_count (friends : ℕ) (crackers_per_friend : ℕ) (total_crackers : ℕ) :
  (friends = 4) → (crackers_per_friend = 2) → (total_crackers = friends * crackers_per_friend) → total_crackers = 8 :=
by intros h_friends h_crackers_per_friend h_total_crackers
   rw [h_friends, h_crackers_per_friend] at h_total_crackers
   exact h_total_crackers

end crackers_initial_count_l70_70531


namespace maximum_possible_value_of_N_l70_70513

-- Definitions to structure the condition and the problem statement
structure City (n : ℕ) :=
(roads_out : ℕ)

def satisfies_conditions (cities : Fin 110 → City) (N : ℕ) : Prop :=
N ≤ 110 ∧
(∀ i, 2 ≤ i → i ≤ N → cities i = { roads_out := i } ∧
  ∀ j, (j = 1 ∨ j = N) → cities j = { roads_out := j })

-- Problem statement to verify the conditions
theorem maximum_possible_value_of_N :
  ∃ N, satisfies_conditions cities N ∧ N = 107 := by
  sorry

end maximum_possible_value_of_N_l70_70513


namespace cost_per_pouch_is_20_l70_70270

theorem cost_per_pouch_is_20 :
  let boxes := 10
  let pouches_per_box := 6
  let dollars := 12
  let cents_per_dollar := 100
  let total_pouches := boxes * pouches_per_box
  let total_cents := dollars * cents_per_dollar
  let cost_per_pouch := total_cents / total_pouches
  cost_per_pouch = 20 :=
by
  sorry

end cost_per_pouch_is_20_l70_70270


namespace find_b_l70_70243

theorem find_b (b : ℤ) :
  (∃ x : ℤ, x^2 + b * x - 36 = 0 ∧ x = -9) → b = 5 :=
by
  sorry

end find_b_l70_70243


namespace books_sold_on_monday_l70_70222

def InitialStock : ℕ := 800
def BooksNotSold : ℕ := 600
def BooksSoldTuesday : ℕ := 10
def BooksSoldWednesday : ℕ := 20
def BooksSoldThursday : ℕ := 44
def BooksSoldFriday : ℕ := 66

def TotalBooksSold : ℕ := InitialStock - BooksNotSold
def BooksSoldAfterMonday : ℕ := BooksSoldTuesday + BooksSoldWednesday + BooksSoldThursday + BooksSoldFriday

theorem books_sold_on_monday : 
  TotalBooksSold - BooksSoldAfterMonday = 60 := by
  sorry

end books_sold_on_monday_l70_70222


namespace remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l70_70090

structure ArtCollection where
  medieval : ℕ
  renaissance : ℕ
  modern : ℕ

def AliciaArtCollection : ArtCollection := {
  medieval := 70,
  renaissance := 120,
  modern := 150
}

def donationPercentages : ArtCollection := {
  medieval := 65,
  renaissance := 30,
  modern := 45
}

def remainingArtPieces (initial : ℕ) (percent : ℕ) : ℕ :=
  initial - ((percent * initial) / 100)

theorem remaining_medieval_art_pieces :
  remainingArtPieces AliciaArtCollection.medieval donationPercentages.medieval = 25 := by
  sorry

theorem remaining_renaissance_art_pieces :
  remainingArtPieces AliciaArtCollection.renaissance donationPercentages.renaissance = 84 := by
  sorry

theorem remaining_modern_art_pieces :
  remainingArtPieces AliciaArtCollection.modern donationPercentages.modern = 83 := by
  sorry

end remaining_medieval_art_pieces_remaining_renaissance_art_pieces_remaining_modern_art_pieces_l70_70090


namespace car_production_total_l70_70078

theorem car_production_total (northAmericaCars europeCars : ℕ) (h1 : northAmericaCars = 3884) (h2 : europeCars = 2871) : northAmericaCars + europeCars = 6755 := by
  sorry

end car_production_total_l70_70078


namespace boxes_contain_fruits_l70_70195

-- Define the weights of the boxes
def box_weights : List ℕ := [15, 16, 18, 19, 20, 31]

-- Define the weight requirement for apples and pears
def weight_rel (apple_weight pear_weight : ℕ) : Prop := apple_weight = pear_weight / 2

-- Define the statement with the constraints, given conditions and assignments.
theorem boxes_contain_fruits (h1 : box_weights = [15, 16, 18, 19, 20, 31])
                             (h2 : ∃ apple_weight pear_weight, 
                                   weight_rel apple_weight pear_weight ∧ 
                                   pear_weight ∈ box_weights ∧ apple_weight ∈ box_weights)
                             (h3 : ∃ orange_weight, orange_weight ∈ box_weights ∧ 
                                   ∀ w, w ∈ box_weights → w ≠ orange_weight)
                             : (15 = 2 ∧ 19 = 3 ∧ 20 = 1 ∧ 31 = 3) := 
                             sorry

end boxes_contain_fruits_l70_70195


namespace bmw_length_l70_70023

theorem bmw_length : 
  let horiz1 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let horiz2 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let vert1  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert2  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert3  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert4  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert5  : ℝ := 2 -- Length of each vertical segment in 'W'
  let diag1  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  let diag2  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  (horiz1 + horiz2 + vert1 + vert2 + vert3 + vert4 + vert5 + diag1 + diag2) = 14 + 2 * Real.sqrt 2 :=
by
  sorry

end bmw_length_l70_70023


namespace calculate_color_cartridges_l70_70745

theorem calculate_color_cartridges (c b : ℕ) (h1 : 32 * c + 27 * b = 123) (h2 : b ≥ 1) : c = 3 :=
by
  sorry

end calculate_color_cartridges_l70_70745


namespace percentage_of_non_technicians_l70_70152

theorem percentage_of_non_technicians (total_workers technicians non_technicians permanent_technicians permanent_non_technicians temporary_workers : ℝ)
  (h1 : technicians = 0.5 * total_workers)
  (h2 : non_technicians = total_workers - technicians)
  (h3 : permanent_technicians = 0.5 * technicians)
  (h4 : permanent_non_technicians = 0.5 * non_technicians)
  (h5 : temporary_workers = 0.5 * total_workers) :
  (non_technicians / total_workers) * 100 = 50 :=
by
  -- Proof is omitted
  sorry

end percentage_of_non_technicians_l70_70152


namespace behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l70_70445

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 5 * x ^ 2 + 4

theorem behavior_of_g_as_x_approaches_infinity_and_negative_infinity :
  (∀ ε > 0, ∃ M > 0, ∀ x > M, g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x < -N, g x > ε) :=
by
  sorry

end behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l70_70445


namespace triangular_region_area_l70_70594

noncomputable def area_of_triangle (f g h : ℝ → ℝ) : ℝ :=
  let (x1, y1) := (-3, f (-3))
  let (x2, y2) := (7/3, g (7/3))
  let (x3, y3) := (15/11, f (15/11))
  let base := abs (x2 - x1)
  let height := abs (y3 - 2)
  (1/2) * base * height

theorem triangular_region_area :
  let f x := (2/3) * x + 4
  let g x := -3 * x + 9
  let h x := (2 : ℝ)
  area_of_triangle f g h = 256/33 :=  -- Given conditions
by
  sorry  -- Proof to be supplied

end triangular_region_area_l70_70594


namespace count_non_perfect_square_or_cube_l70_70660

-- Define perfect_square function
def perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

-- Define perfect_cube function
def perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

-- Define range from 1 to 200
def range_200 := {1 .. 200}

-- Define numbers either perfect square or perfect cube
def perfect_square_or_cube : Finset ℕ :=
  (range_200.filter perfect_square) ∪ (range_200.filter perfect_cube)

-- Define sixth powers in range
def perfect_sixth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k^6 = n

-- The final problem statement
theorem count_non_perfect_square_or_cube :
  (range_200.card - perfect_square_or_cube.card) = 182 :=
by
  sorry

end count_non_perfect_square_or_cube_l70_70660


namespace total_toothpicks_for_grid_l70_70747

-- Defining the conditions
def grid_height := 30
def grid_width := 15

-- Define the function that calculates the total number of toothpicks
def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := (width + 1) * height
  horizontal_toothpicks + vertical_toothpicks

-- The theorem stating the problem and its answer
theorem total_toothpicks_for_grid : total_toothpicks grid_height grid_width = 945 :=
by {
  -- Here we would write the proof steps. Using sorry for now.
  sorry
}

end total_toothpicks_for_grid_l70_70747


namespace hua_luogeng_optimal_selection_method_uses_golden_ratio_l70_70319

-- Define the conditions
def optimal_selection_method (method: String) : Prop :=
  method = "associated with Hua Luogeng"

def options : List String :=
  ["Golden ratio", "Mean", "Mode", "Median"]

-- Define the proof problem
theorem hua_luogeng_optimal_selection_method_uses_golden_ratio :
  (∀ method, optimal_selection_method method → method ∈ options) → ("Golden ratio" ∈ options) :=
by
  sorry -- Placeholder for the proof


end hua_luogeng_optimal_selection_method_uses_golden_ratio_l70_70319


namespace degree_of_monomial_neg2x2y_l70_70873

def monomial_degree (coeff : ℤ) (exp_x exp_y : ℕ) : ℕ :=
  exp_x + exp_y

theorem degree_of_monomial_neg2x2y :
  monomial_degree (-2) 2 1 = 3 :=
by
  -- Definition matching conditions given
  sorry

end degree_of_monomial_neg2x2y_l70_70873


namespace min_product_of_three_numbers_l70_70631

def SetOfNumbers : Set ℤ := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three_numbers : 
  ∃ (a b c : ℤ), a ∈ SetOfNumbers ∧ b ∈ SetOfNumbers ∧ c ∈ SetOfNumbers ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -360 :=
by {
  sorry
}

end min_product_of_three_numbers_l70_70631


namespace fraction_for_repeating_decimal_l70_70111

variable (a r S : ℚ)
variable (h1 : a = 3/5)
variable (h2 : r = 1/10)
variable (h3 : S = a / (1 - r))

theorem fraction_for_repeating_decimal : S = 2 / 3 :=
by
  have h4 : 1 - r = 9 / 10, from sorry
  have h5 : S = (3 / 5) / (9 / 10), from sorry
  have h6 : S = (3 * 10) / (5 * 9), from sorry
  have h7 : S = 30 / 45, from sorry
  have h8 : 30 / 45 = 2 / 3, from sorry
  exact h8

end fraction_for_repeating_decimal_l70_70111


namespace triangle_problem_solution_l70_70268

theorem triangle_problem_solution
  (A : ℝ)
  (b c a : ℝ)
  (sinA sinB sinC : ℝ)
  (cosA : ℝ)
  (SinRule : (a / sinA = b / sinB ∧ a / sinA = c / sinC))
  (S : sinA + sinB + sinC)
  (H_A : A = π / 3)
  (H_b : b = 1)
  (H_c : c = 4)
  (cos_A : cosA = 1 / 2)
  (sin_A : sinA = (real.sqrt 3) / 2)
  (cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * cosA): 
  (a + b + c) / (sinA + sinB + sinC) = (2 * real.sqrt 39) / 3 :=
by
  sorry

end triangle_problem_solution_l70_70268


namespace probability_XOXOX_l70_70052

theorem probability_XOXOX (n_X n_O n_total : ℕ) (h_total : n_X + n_O = n_total)
  (h_X : n_X = 3) (h_O : n_O = 2) (h_total' : n_total = 5) :
  (1 / ↑(Nat.choose n_total n_X)) = (1 / 10) :=
by
  sorry

end probability_XOXOX_l70_70052


namespace fraction_equality_x_eq_neg1_l70_70630

theorem fraction_equality_x_eq_neg1 (x : ℝ) (h : (5 + x) / (7 + x) = (3 + x) / (4 + x)) : x = -1 := by
  sorry

end fraction_equality_x_eq_neg1_l70_70630


namespace box_neg2_0_3_eq_10_div_9_l70_70960

def box (a b c : ℤ) : ℚ :=
  a^b - b^c + c^a

theorem box_neg2_0_3_eq_10_div_9 : box (-2) 0 3 = 10 / 9 :=
by
  sorry

end box_neg2_0_3_eq_10_div_9_l70_70960


namespace abs_opposite_numbers_l70_70834

theorem abs_opposite_numbers (m n : ℤ) (h : m + n = 0) : |m + n - 1| = 1 := by
  sorry

end abs_opposite_numbers_l70_70834


namespace business_fraction_l70_70421

theorem business_fraction (x : ℚ) (H1 : 3 / 4 * x * 60000 = 30000) : x = 2 / 3 :=
by sorry

end business_fraction_l70_70421


namespace repeating_six_equals_fraction_l70_70115

theorem repeating_six_equals_fraction : ∃ f : ℚ, (∀ n : ℕ, (n ≥ 1 → (6 * (10 : ℕ) ^ (-n) : ℚ) + (f - (6 * (10 : ℕ) ^ (-n) : ℚ)) = f)) ∧ f = 2 / 3 := sorry

end repeating_six_equals_fraction_l70_70115


namespace members_playing_both_sports_l70_70573

theorem members_playing_both_sports 
    (N : ℕ) (B : ℕ) (T : ℕ) (D : ℕ)
    (hN : N = 30) (hB : B = 18) (hT : T = 19) (hD : D = 2) :
    N - D = 28 ∧ B + T = 37 ∧ B + T - (N - D) = 9 :=
by
  sorry

end members_playing_both_sports_l70_70573


namespace min_number_of_girls_l70_70384

theorem min_number_of_girls (total_students boys_count girls_count : ℕ) (no_three_equal: ∀ m n : ℕ, m ≠ n → boys_count m ≠ boys_count n) (boys_at_most : boys_count ≤ 2 * (girls_count + 1)) : ∃ d : ℕ, d ≥ 6 ∧ total_students = 20 ∧ boys_count = total_students - girls_count :=
by
  let d := 6
  exists d
  sorry

end min_number_of_girls_l70_70384


namespace angle_A_and_area_of_triangle_l70_70238

theorem angle_A_and_area_of_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) (h1 : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) 
(h2 : R = 2) (h3 : b^2 + c^2 = 18) :
  A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 := 
by
  sorry

end angle_A_and_area_of_triangle_l70_70238


namespace distance_between_A_and_B_l70_70535

def scale : ℕ := 20000
def map_distance : ℕ := 6
def actual_distance_cm : ℕ := scale * map_distance
def actual_distance_m : ℕ := actual_distance_cm / 100

theorem distance_between_A_and_B : actual_distance_m = 1200 := by
  sorry

end distance_between_A_and_B_l70_70535


namespace repeated_three_digit_divisible_101_l70_70772

theorem repeated_three_digit_divisible_101 (abc : ℕ) (h1 : 100 ≤ abc) (h2 : abc < 1000) :
  (1000000 * abc + 1000 * abc + abc) % 101 = 0 :=
by
  sorry

end repeated_three_digit_divisible_101_l70_70772


namespace mod_equiv_inverse_sum_l70_70750

theorem mod_equiv_inverse_sum :
  (3^15 + 3^14 + 3^13 + 3^12) % 17 = 5 :=
by sorry

end mod_equiv_inverse_sum_l70_70750


namespace countTwoLeggedBirds_l70_70869

def countAnimals (x y : ℕ) : Prop :=
  x + y = 200 ∧ 2 * x + 4 * y = 522

theorem countTwoLeggedBirds (x y : ℕ) (h : countAnimals x y) : x = 139 :=
by
  sorry

end countTwoLeggedBirds_l70_70869


namespace calculate_savings_l70_70436

def monthly_income : list ℕ := [45000, 35000, 7000, 10000, 13000]
def monthly_expenses : list ℕ := [30000, 10000, 5000, 4500, 9000]
def initial_savings : ℕ := 849400

def total_income : ℕ := 5 * monthly_income.sum
def total_expenses : ℕ := 5 * monthly_expenses.sum
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem calculate_savings :
  total_income = 550000 ∧
  total_expenses = 292500 ∧
  final_savings = 1106900 :=
by
  sorry

end calculate_savings_l70_70436


namespace percent_increase_march_to_april_l70_70035

theorem percent_increase_march_to_april (P : ℝ) (X : ℝ) 
  (H1 : ∃ Y Z : ℝ, P * (1 + X / 100) * 0.8 * 1.5 = P * (1 + Y / 100) ∧ Y = 56.00000000000001)
  (H2 : P * (1 + X / 100) * 0.8 * 1.5 = P * 1.5600000000000001)
  (H3 : P ≠ 0) :
  X = 30 :=
by sorry

end percent_increase_march_to_april_l70_70035


namespace functional_eq_l70_70101

theorem functional_eq (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x * f y + y) = f (x * y) + f y) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end functional_eq_l70_70101


namespace perpendicular_planes_condition_l70_70633

variables (α β : Plane) (m : Line) 

-- Assuming the basic definitions:
def perpendicular (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

-- Conditions
axiom α_diff_β : α ≠ β
axiom m_in_α : in_plane m α

-- Proving the necessary but not sufficient condition
theorem perpendicular_planes_condition : 
  (perpendicular α β → perpendicular_to_plane m β) ∧ 
  (¬ perpendicular_to_plane m β → ¬ perpendicular α β) ∧ 
  ¬ (perpendicular_to_plane m β → perpendicular α β) :=
sorry

end perpendicular_planes_condition_l70_70633


namespace find_y_l70_70253

theorem find_y (y : ℕ) : (1 / 8) * 2^36 = 8^y → y = 11 := by
  sorry

end find_y_l70_70253


namespace find_four_digit_number_l70_70938

theorem find_four_digit_number :
  ∃ A B C D : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
    (1001 * A + 100 * B + 10 * C + A) = 182 * (10 * C + D) ∧
    (1000 * A + 100 * B + 10 * C + D) = 2916 :=
by 
  sorry

end find_four_digit_number_l70_70938


namespace area_union_example_l70_70591

noncomputable def area_union_square_circle (s r : ℝ) : ℝ :=
  let A_square := s ^ 2
  let A_circle := Real.pi * r ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_square + A_circle - A_overlap

theorem area_union_example : (area_union_square_circle 10 10) = 100 + 75 * Real.pi :=
by
  sorry

end area_union_example_l70_70591


namespace ratio_Sydney_to_Sherry_l70_70536

variable (Randolph_age Sydney_age Sherry_age : ℕ)

-- Conditions
axiom Randolph_older_than_Sydney : Randolph_age = Sydney_age + 5
axiom Sherry_age_is_25 : Sherry_age = 25
axiom Randolph_age_is_55 : Randolph_age = 55

-- Theorem to prove
theorem ratio_Sydney_to_Sherry : (Sydney_age : ℝ) / (Sherry_age : ℝ) = 2 := by
  sorry

end ratio_Sydney_to_Sherry_l70_70536


namespace first_worker_time_l70_70432

theorem first_worker_time
  (T : ℝ) 
  (hT : T ≠ 0)
  (h_comb : (T + 8) / (8 * T) = 1 / 3.428571428571429) :
  T = 8 / 7 :=
by
  sorry

end first_worker_time_l70_70432


namespace optimal_selection_method_use_golden_ratio_l70_70347

/-- 
  The method used in the optimal selection method popularized by Hua Luogeng is the 
  Golden ratio given the options (Golden ratio, Mean, Mode, Median).
-/
theorem optimal_selection_method_use_golden_ratio 
  (options : List String)
  (golden_ratio : String)
  (mean : String)
  (mode : String)
  (median : String)
  (hua_luogeng : String) :
  options = ["Golden ratio", "Mean", "Mode", "Median"] ∧ 
  hua_luogeng = "Hua Luogeng" →
  "The optimal selection method uses " ++ golden_ratio ++ " according to " ++ hua_luogeng :=
begin
  -- conditions
  intro h,
  -- define each option based on the common context given "h"
  have h_option_list : options = ["Golden ratio", "Mean", "Mode", "Median"], from h.1,
  have h_hua_luogeng : hua_luogeng = "Hua Luogeng", from h.2,
  -- the actual statement
  have h_answer : golden_ratio = "Golden ratio", from rfl,
  exact Eq.subst h_answer (by simp [h_option_list, h_hua_luogeng, golden_ratio]) sorry,
end

end optimal_selection_method_use_golden_ratio_l70_70347


namespace num_factors_34848_l70_70626

/-- Define the number 34848 and its prime factorization -/
def n : ℕ := 34848
def p_factors : List (ℕ × ℕ) := [(2, 5), (3, 2), (11, 2)]

/-- Helper function to calculate the number of divisors from prime factors -/
def num_divisors (factors : List (ℕ × ℕ)) : ℕ := 
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.2 + 1)) 1

/-- Formal statement of the problem -/
theorem num_factors_34848 : num_divisors p_factors = 54 :=
by
  -- Proof that 34848 has the prime factorization 3^2 * 2^5 * 11^2 
  -- and that the number of factors is 54 would go here.
  sorry

end num_factors_34848_l70_70626
