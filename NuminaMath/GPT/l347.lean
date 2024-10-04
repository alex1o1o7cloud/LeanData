import Mathlib

namespace ways_to_place_balls_in_boxes_l347_347500

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347500


namespace number_of_triangles_in_decagon_l347_347044

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347044


namespace distinguish_ball_box_ways_l347_347353

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347353


namespace area_AEFC_eq_2_area_ABCD_l347_347953

-- Definitions for Points and Parallelograms
structure Point :=
(x : ℝ)
(y : ℝ)

structure Parallelogram :=
(A B C D : Point)
(parallel_AB_CD : ∃ k, (B.y - A.y) = k * (C.y - D.y) ∧ (B.x - A.x) = k * (C.x - D.x))
(parallel_AD_BC : ∃ k, (D.y - A.y) = k * (C.y - B.y) ∧ (D.x - A.x) = k * (C.x - B.x))

variables (parallelogram_ABCD : Parallelogram)

-- Definitions for E and F points
def E := Point.mk (2 * parallelogram_ABCD.B.x - parallelogram_ABCD.D.x) (2 * parallelogram_ABCD.B.y - parallelogram_ABCD.D.y)
def F := Point.mk (parallelogram_ABCD.A.x + parallelogram_ABCD.C.x - parallelogram_ABCD.B.x) (parallelogram_ABCD.A.y + parallelogram_ABCD.C.y - parallelogram_ABCD.B.y)

-- Definitions for Quadrilaterals and Areas
def area_parallelogram (p : Parallelogram) : ℝ := abs((p.A.x * p.C.y + p.C.x * p.B.y + p.B.x * p.D.y + p.D.x * p.A.y)
                                                      - (p.C.x * p.A.y + p.B.x * p.C.y + p.D.x * p.B.y + p.A.x * p.D.y))

def area_quadrilateral (A B C D : Point) : ℝ := abs((A.x * C.y + C.x * B.y + B.x * D.y + D.x * A.y)
                                                   - (C.x * A.y + B.x * C.y + D.x * B.y + A.x * D.y ))

-- The proof problem
theorem area_AEFC_eq_2_area_ABCD :
  area_quadrilateral parallelogram_ABCD.A E F parallelogram_ABCD.C = 2 * area_parallelogram parallelogram_ABCD :=
sorry

end area_AEFC_eq_2_area_ABCD_l347_347953


namespace triangle_area_AB_AC_angle_B_l347_347549

theorem triangle_area_AB_AC_angle_B
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (AB AC : ℝ)
  (angle_B : ℝ)
  (h_AB : AB = sqrt 3)
  (h_AC : AC = 1)
  (h_angle_B : angle_B = 30) :
  ∃ (S : ℝ), (S = sqrt 3 / 2) ∨ (S = sqrt 3 / 4) :=
sorry

end triangle_area_AB_AC_angle_B_l347_347549


namespace oliver_kept_stickers_l347_347603

theorem oliver_kept_stickers (initial_stickers : ℕ) 
(used_fraction : ℚ) 
(gave_fraction : ℚ) 
(h_initial : initial_stickers = 135) 
(h_used : used_fraction = 1/3) 
(h_gave : gave_fraction = 2/5) : 
∃ (kept_stickers : ℕ), kept_stickers = 54 := 
by 
-- Skip the steps and just assert the existence of the proof.
sory

end oliver_kept_stickers_l347_347603


namespace equal_roots_quadratic_eq_l347_347869

theorem equal_roots_quadratic_eq (m n : ℝ) (h : m^2 - 4 * n = 0) : m = 2 ∧ n = 1 :=
by
  sorry

end equal_roots_quadratic_eq_l347_347869


namespace general_term_formula_geometric_sequence_sum_l347_347872

structure Sequences where
  a : ℕ → ℕ
  b : ℕ → ℕ

def arithmetic_sequence (seq : Sequences) : Prop :=
  seq.a 1 = 1 ∧ seq.a 2 + seq.a 4 = 10

def geometric_sequence (seq : Sequences) : Prop :=
  seq.b 1 = 1 ∧ seq.b 2 * seq.b 4 = seq.a 5

theorem general_term_formula (seq : Sequences) (h : arithmetic_sequence seq) : ∀ n, seq.a n = 2 * n - 1 := sorry

theorem geometric_sequence_sum (seq : Sequences) (h₁ : arithmetic_sequence seq) (h₂ : geometric_sequence seq) :
  ∀ n, (∑ i in Finset.range n, seq.b (2 * i + 1)) = (3^n - 1) / 2 := sorry

end general_term_formula_geometric_sequence_sum_l347_347872


namespace determine_min_guesses_l347_347749

def minimum_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem determine_min_guesses (n k : ℕ) (h : n > k) :
  (if n = 2 * k then 2 else 1) = minimum_guesses n k h := by
  sorry

end determine_min_guesses_l347_347749


namespace regular_decagon_triangle_count_l347_347038

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347038


namespace mrs_sheridan_fish_count_l347_347599

/-
  Problem statement: 
  Prove that the total number of fish Mrs. Sheridan has now is 69, 
  given that she initially had 22 fish and she received 47 more from her sister.
-/

theorem mrs_sheridan_fish_count :
  let initial_fish : ℕ := 22
  let additional_fish : ℕ := 47
  initial_fish + additional_fish = 69 := by
sorry

end mrs_sheridan_fish_count_l347_347599


namespace balls_into_boxes_l347_347403

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347403


namespace modulus_of_quotient_l347_347514

theorem modulus_of_quotient (z : ℂ) (hz : z = (1 - 2 * Complex.i) / (3 - Complex.i)) : Complex.abs z = Real.sqrt 2 / 2 :=
by
  rw [hz]
  sorry

end modulus_of_quotient_l347_347514


namespace ways_to_distribute_balls_l347_347423

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347423


namespace no_prime_three_digit_l347_347963

theorem no_prime_three_digit (n : ℕ) (h_digits : n.digits.size = 9) 
  (h_diff : n.digits.nodup) (h_no_7 : ¬7 ∈ n.digits) : 
  ∀ m : ℕ, (m ∈ (remove_six_digits n).filter (λ k, 100 ≤ k ∧ k < 1000)) → ¬prime m :=
sorry

end no_prime_three_digit_l347_347963


namespace minimize_sum_of_distances_l347_347880

theorem minimize_sum_of_distances (P : ℝ × ℝ) (A : ℝ × ℝ) (F : ℝ × ℝ) 
  (hP_on_parabola : P.2 ^ 2 = 2 * P.1)
  (hA : A = (3, 2)) 
  (hF : F = (1/2, 0)) : 
  |P - A| + |P - F| ≥ |(2, 2) - A| + |(2, 2) - F| :=
by sorry

end minimize_sum_of_distances_l347_347880


namespace nth_monomial_pattern_l347_347746

theorem nth_monomial_pattern (a : ℝ) (n : ℕ) (h_n_pos : 0 < n) : 
  exists (c : ℕ), (c = 2 * n - 1) ∧ (nth_monomial n a = c * a ^ n) :=
by {
  sorry
}

end nth_monomial_pattern_l347_347746


namespace number_of_power_functions_l347_347543

noncomputable def is_power_function (f : ℝ → ℝ) : Prop :=
∃ (k n : ℝ), ∀ x, f x = k * x^n

def f1 (x : ℝ) : ℝ := x^(-2)
def f2 (x : ℝ) : ℝ := 2 * x
def f3 (x : ℝ) : ℝ := x^2 + x
def f4 (x : ℝ) : ℝ := x^(5/3)

theorem number_of_power_functions :
  (is_power_function f1 ∧ is_power_function f4) ∧ ¬(is_power_function f2 ∨ is_power_function f3) →
  2 :=
by 
  sorry

end number_of_power_functions_l347_347543


namespace option_C_mutually_exclusive_l347_347182

def at_least_one_defective (batch : List Bool) : Prop :=
  ∃ x, x = false ∧ x ∈ batch

def all_good (batch : List Bool) : Prop :=
  ∀ x ∈ batch, x = true

def exactly_one_defective (batch : List Bool) : Prop :=
  (∃ x, x = false ∧ x ∈ batch) ∧
  (∀ y ≠ x, y = true ∨ y ∉ batch)

def exactly_two_defective (batch : List Bool) : Prop :=
  (∃ x y, x ≠ y ∧ x = false ∧ y = false ∧ x ∈ batch ∧ y ∈ batch) ∧
  (∀ z, z ≠ x ∧ z ≠ y → z = true ∨ z ∉ batch)

def at_least_one_good (batch : List Bool) : Prop :=
  ∃ x, x = true ∧ x ∈ batch

def mutually_exclusive (P Q : Prop) : Prop :=
  P ∧ Q → false

theorem option_C_mutually_exclusive (batch : List Bool) 
  (h_good: ∃ g1 g2, g1 ≠ g2 ∧ g1 = true ∧ g2 = true ∧ g1 ∈ batch ∧ g2 ∈ batch)
  (h_defective: ∃ d1 d2, d1 ≠ d2 ∧ d1 = false ∧ d2 = false ∧ d1 ∈ batch ∧ d2 ∈ batch) :
  mutually_exclusive (at_least_one_defective batch) (all_good batch) :=
sorry

end option_C_mutually_exclusive_l347_347182


namespace cristina_catches_up_l347_347600

theorem cristina_catches_up
  (t : ℝ)
  (cristina_speed : ℝ := 5)
  (nicky_speed : ℝ := 3)
  (nicky_head_start : ℝ := 54)
  (distance_cristina : ℝ := cristina_speed * t)
  (distance_nicky : ℝ := nicky_head_start + nicky_speed * t) :
  distance_cristina = distance_nicky → t = 27 :=
by
  intros h
  sorry

end cristina_catches_up_l347_347600


namespace problem1_problem2_l347_347244

def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1
def f (a b : ℝ) : ℝ := 3 * A a b + 6 * B a b

theorem problem1 (a b : ℝ) : f a b = 15 * a * b - 6 * a - 9 :=
by 
  sorry

theorem problem2 (b : ℝ) : (∀ a : ℝ, f a b = -9) → b = 2 / 5 :=
by 
  sorry

end problem1_problem2_l347_347244


namespace triangle_angles_l347_347534

theorem triangle_angles
  (A B C M : Type)
  (ortho_divides_height_A : ∀ (H_AA1 : ℝ), ∃ (H_AM : ℝ), H_AA1 = H_AM * 3 ∧ H_AM = 2 * H_AA1 / 3)
  (ortho_divides_height_B : ∀ (H_BB1 : ℝ), ∃ (H_BM : ℝ), H_BB1 = H_BM * 5 / 2 ∧ H_BM = 3 * H_BB1 / 5) :
  ∃ α β γ : ℝ, α = 60 + 40 / 60 ∧ β = 64 + 36 / 60 ∧ γ = 54 + 44 / 60 :=
by { 
  sorry 
}

end triangle_angles_l347_347534


namespace put_balls_in_boxes_l347_347301

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347301


namespace fort_blocks_count_l347_347594

noncomputable def volume_original := 15 * 12 * 6
noncomputable def interior_length := 15 - 3
noncomputable def interior_width := 12 - 3
noncomputable def interior_height := 6 - 1.5
noncomputable def volume_interior := interior_length * interior_width * interior_height
noncomputable def volume_blocks := volume_original - volume_interior

theorem fort_blocks_count : volume_blocks = 594 := 
by
  -- Show that the volume of the original fort is 1080
  have h1 : volume_original = 1080 := by
    -- Calculations for volume_original: 15 * 12 * 6 = 1080
    sorry
    
  -- Show that the interior dimensions are 12, 9, and 4.5
  have h2 : interior_length = 12 ∧ interior_width = 9 ∧ interior_height = 4.5 := by
    -- Calculations for interior dimensions: 15 - 3 = 12, 12 - 3 = 9, 6 - 1.5 = 4.5
    sorry
    
  -- Show that the volume of the interior space is 486
  have h3 : volume_interior = 486 := by
    -- Calculations for volume_interior: 12 * 9 * 4.5 = 486
    sorry
    
  -- Combine results to show that the number of blocks used is 594
  have h4 : volume_blocks = 1080 - 486 := by
    -- Calculation for volume_blocks: 1080 - 486 = 594
    sorry
    
  show volume_blocks = 594 from h4

end fort_blocks_count_l347_347594


namespace farmland_areas_avg_sprayed_area_per_sortie_l347_347732

-- Definitions for farmland in zones A and B.
def zoneB_farm_land : ℝ := 40000
def zoneA_farm_land : ℝ := zoneB_farm_land + 10000

-- Definitions of suitable farmland in zones A and B.
def suitable_farm_land_A := 0.8 * zoneA_farm_land
def suitable_farm_land_B := zoneB_farm_land

-- Conditions:
-- Suitable farmland in both zones are exactly the same.
def suitable_land_equal : Prop := suitable_farm_land_A = suitable_farm_land_B

-- Number of drone sorties and their performance.
def drone_sorties_ratio : ℝ := 1.2
def avg_sprayed_area_A : ℝ := 100
def avg_sprayed_area_B : ℝ := avg_sprayed_area_A - (50 / 3)

-- Proving the areas.
theorem farmland_areas :
  suitable_land_equal →
  zoneA_farm_land = 50000 ∧ zoneB_farm_land = 40000 :=
by
  sorry

-- Proving the average sprayed area per sortie.
theorem avg_sprayed_area_per_sortie :
  suitable_land_equal →
  (40_000 / avg_sprayed_area_B) = (40_000 / avg_sprayed_area_A) * drone_sorties_ratio →
  avg_sprayed_area_A = 100 :=
by
  sorry

end farmland_areas_avg_sprayed_area_per_sortie_l347_347732


namespace find_eccentricity_of_hyperbola_l347_347236

def hyperbola_eccentricity (a b : ℝ) (h₁ : a + b = 5) (h₂ : a * b = 6) (h₃ : a > 0) (h₄ : b > 0) (h₅ : a > b) : ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem find_eccentricity_of_hyperbola (a b : ℝ) 
  (h₁ : a + b = 5) 
  (h₂ : a * b = 6) 
  (h₃ : a > 0) 
  (h₄ : b > 0) 
  (h₅ : a > b) : 
  hyperbola_eccentricity a b h₁ h₂ h₃ h₄ h₅ = Real.sqrt 13 / 3 := 
by 
  sorry

end find_eccentricity_of_hyperbola_l347_347236


namespace slope_between_midpoints_l347_347694

-- Define the points
def A : ℝ × ℝ := (3, 5)
def B : ℝ × ℝ := (7, 12)
def C : ℝ × ℝ := (4, 1)
def D : ℝ × ℝ := (9, 6)

-- Midpoint function
def midpoint (X Y : ℝ × ℝ) : ℝ × ℝ :=
  ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- Calculate the midpoints of the segments
def M1 := midpoint A B
def M2 := midpoint C D

-- Function to calculate the slope between two points
def slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

-- Prove the slope between M1 and M2 is -10/3
theorem slope_between_midpoints :
  slope M1 M2 = -10 / 3 :=
by
  sorry

end slope_between_midpoints_l347_347694


namespace salary_for_May_l347_347994

variable (J F M A May : ℕ)

axiom condition1 : (J + F + M + A) / 4 = 8000
axiom condition2 : (F + M + A + May) / 4 = 8800
axiom condition3 : J = 3300

theorem salary_for_May : May = 6500 :=
by sorry

end salary_for_May_l347_347994


namespace ways_to_distribute_balls_l347_347366

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347366


namespace lillian_candies_addition_l347_347595

noncomputable def lillian_initial_candies : ℕ := 88
noncomputable def lillian_father_candies : ℕ := 5
noncomputable def lillian_total_candies : ℕ := 93

theorem lillian_candies_addition : lillian_initial_candies + lillian_father_candies = lillian_total_candies := by
  sorry

end lillian_candies_addition_l347_347595


namespace number_of_triangles_in_decagon_l347_347091

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347091


namespace gcd_of_168_56_224_l347_347687

theorem gcd_of_168_56_224 : (Nat.gcd 168 56 = 56) ∧ (Nat.gcd 56 224 = 56) ∧ (Nat.gcd 168 224 = 56) :=
by
  sorry

end gcd_of_168_56_224_l347_347687


namespace gabe_playlist_count_l347_347842

open Nat

theorem gabe_playlist_count :
  let best_day := 3
  let raise_the_roof := 2
  let rap_battle := 3
  let total_playlist_time := best_day + raise_the_roof + rap_battle
  let total_ride_time := 40
  total_ride_time / total_playlist_time = 5 := by
  have h_playlist_time : total_playlist_time = best_day + raise_the_roof + rap_battle := rfl
  have h_total_playlist_time : total_playlist_time = 8 := by
    rw [h_playlist_time]
    norm_num
  have h_total_ride_time : total_ride_time = 40 := rfl
  rw [h_total_playlist_time, h_total_ride_time]
  norm_num
  sorry

end gabe_playlist_count_l347_347842


namespace math_problem_l347_347706

-- Define the first part of the problem
def line_area_to_axes (line_eq : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  line_eq x y ∧ x = 4 ∧ y = -4

-- Define the second part of the problem
def line_through_fixed_point (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (m * x) + y + m = 0 ∧ x = -1 ∧ y = 0

-- Theorem combining both parts
theorem math_problem (line_eq : ℝ → ℝ → Prop) (m : ℝ) :
  (∃ x y, line_area_to_axes line_eq x y → 8 = (1 / 2) * 4 * 4) ∧ line_through_fixed_point m :=
sorry

end math_problem_l347_347706


namespace put_balls_in_boxes_l347_347304

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347304


namespace exists_small_triangle_area_l347_347938

theorem exists_small_triangle_area (A B C D E F : Point) (P : Fin 2556 → Point) :
  let S := [A, B, C, D, E, F].toFinset ∪ Finset.univ.image P,
  regular_hexagon A B C D E F ∧
  distinct_points S ∧ 
  ∀ (x y z : Point), x ∈ S → y ∈ S → z ∈ S → ¬collinear x y z →
  ∃ (T : triangle), T ⊆ S ∧ area T < (1/1700 : ℝ) :=
by
  sorry

end exists_small_triangle_area_l347_347938


namespace tree_count_correct_l347_347539

def apricot_trees : ℕ := 58

def peach_trees : ℕ := 3 * apricot_trees

def cherry_trees : ℕ := 5 * peach_trees

def total_trees : ℕ := apricot_trees + peach_trees + cherry_trees

theorem tree_count_correct : total_trees = 1102 := by
  unfold total_trees apricot_trees peach_trees cherry_trees
  calc
    58 + (3 * 58) + (5 * (3 * 58)) = 58 + 174 + (5 * 174)    : by rw [mul_assoc]
    ...                          =  58 + 174 + 870            : by rw [mul_comm, nat.mul_comm]
    ...                          =  1102                     : by norm_num

end tree_count_correct_l347_347539


namespace number_of_triangles_in_decagon_l347_347017

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347017


namespace median_possible_values_l347_347571

theorem median_possible_values (S : Finset ℤ)
  (h : S.card = 10)
  (h_contains : {5, 7, 12, 15, 18, 21} ⊆ S) :
  ∃! n : ℕ, n = 5 :=
by
   sorry

end median_possible_values_l347_347571


namespace midpoint_in_polar_coordinates_l347_347531

theorem midpoint_in_polar_coordinates :
  let A := (9, Real.pi / 3)
  let B := (9, 2 * Real.pi / 3)
  let mid := (Real.sqrt (3) * 9 / 2, Real.pi / 2)
  (mid = (Real.sqrt (3) * 9 / 2, Real.pi / 2)) :=
by 
  sorry

end midpoint_in_polar_coordinates_l347_347531


namespace distance_between_trees_l347_347728

theorem distance_between_trees
  (yard_length : ℝ)
  (number_of_trees : ℝ)
  (yard_length = 856) 
  (number_of_trees = 64) :
  let number_of_segments := number_of_trees - 1 in
  yard_length / number_of_segments = 13.5873 :=
sorry

end distance_between_trees_l347_347728


namespace simplify_sqrt2_add_1_simplify_n_add_1_and_sqrt_n_sqrt_n_sub_1_simplify_series_l347_347976

noncomputable def simplify1 : ℝ := 1 / (sqrt 2 + 1)

theorem simplify_sqrt2_add_1 :
  simplify1 = sqrt 2 - 1 :=
sorry

noncomputable def simplify2 (n : ℕ) : ℝ := 1 / (n + 1) + 1 / (sqrt n + sqrt (n - 1))

theorem simplify_n_add_1_and_sqrt_n_sqrt_n_sub_1 (n : ℕ) :
  simplify2 n = 1 / (n + 1) + sqrt n - sqrt (n - 1) :=
sorry

noncomputable def simplify_sum (n : ℕ) : ℝ :=
  ∑ i in finset.range (n - 1), 1 / (sqrt (i + 1) + sqrt i) 

theorem simplify_series :
  simplify_sum 100 = 9 :=
sorry

end simplify_sqrt2_add_1_simplify_n_add_1_and_sqrt_n_sqrt_n_sub_1_simplify_series_l347_347976


namespace distinct_balls_boxes_l347_347467

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347467


namespace balls_in_boxes_l347_347297

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347297


namespace floor_ceil_sum_l347_347797

open Real

theorem floor_ceil_sum : (⌊1.999⌋ : ℤ) + (⌈3.001⌉ : ℤ) = 5 := by
  sorry

end floor_ceil_sum_l347_347797


namespace heptagonal_prism_faces_and_vertices_l347_347741

structure HeptagonalPrism where
  heptagonal_basis : ℕ
  lateral_faces : ℕ
  basis_vertices : ℕ

noncomputable def faces (h : HeptagonalPrism) : ℕ :=
  2 + h.lateral_faces

noncomputable def vertices (h : HeptagonalPrism) : ℕ :=
  h.basis_vertices * 2

theorem heptagonal_prism_faces_and_vertices : ∀ h : HeptagonalPrism,
  (h.heptagonal_basis = 2) →
  (h.lateral_faces = 7) →
  (h.basis_vertices = 7) →
  faces h = 9 ∧ vertices h = 14 :=
by
  intros
  simp [faces, vertices]
  sorry

end heptagonal_prism_faces_and_vertices_l347_347741


namespace num_triangles_from_decagon_l347_347001

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347001


namespace ball_in_boxes_l347_347273

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347273


namespace probability_of_diff_families_is_correct_l347_347981

open Finset

noncomputable def probability_diff_families : ℚ :=
  let total_people : ℕ := 18
  let num_people_per_family : ℕ := 3
  let num_families : ℕ := 6
  let total_ways_choose_3 : ℕ := (total_people.choose 3)
  let ways_choose_3_families : ℕ := (num_families.choose 3)
  let ways_choose_1_person_per_family : ℕ := num_people_per_family ^ 3
  let favorable_outcomes : ℕ := ways_choose_3_families * ways_choose_1_person_per_family
  (favorable_outcomes : ℚ) / (total_ways_choose_3 : ℚ)

theorem probability_of_diff_families_is_correct :
  probability_diff_families = 45 / 68 :=
by sorry

end probability_of_diff_families_is_correct_l347_347981


namespace regular_decagon_triangle_count_l347_347033

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347033


namespace books_sold_l347_347615

-- Define the conditions
def initial_books : ℕ := 134
def books_given_away : ℕ := 39
def remaining_books : ℕ := 68

-- Define the intermediate calculation of books left after giving away
def books_after_giving_away : ℕ := initial_books - books_given_away

-- Prove the number of books sold
theorem books_sold (initial_books books_given_away remaining_books : ℕ) (h1 : books_after_giving_away = 95) (h2 : remaining_books = 68) :
  (books_after_giving_away - remaining_books) = 27 :=
by
  sorry

end books_sold_l347_347615


namespace regular_decagon_triangle_count_l347_347036

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347036


namespace number_of_triangles_in_regular_decagon_l347_347135

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347135


namespace num_triangles_in_decagon_l347_347077

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347077


namespace complex_number_quadrant_l347_347901

variable (z : ℂ)

theorem complex_number_quadrant (h : z * complex.I = -1 - complex.I) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l347_347901


namespace solve_for_a_l347_347860

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem solve_for_a (a : ℝ) (h : is_pure_imaginary ((a + Complex.i) / (1 + Complex.i))) :
  a = -1 :=
by
  sorry

end solve_for_a_l347_347860


namespace max_sqrt_sum_l347_347205

variable (x y z : ℝ)
variable (h_posx : 0 < x)
variable (h_posy : 0 < y)
variable (h_posz : 0 < z)
variable (h_sum : x + y + z = 3)

theorem max_sqrt_sum : sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1) ≤ 3 * sqrt 3 :=
by
  sorry

end max_sqrt_sum_l347_347205


namespace radius_of_cookie_l347_347725

theorem radius_of_cookie (x y : ℝ) : 
  (x^2 + y^2 + x - 5 * y = 10) → 
  ∃ r, (r = Real.sqrt (33 / 2)) :=
by
  sorry

end radius_of_cookie_l347_347725


namespace convex_quadrilateral_intersecting_bisectors_l347_347736

theorem convex_quadrilateral_intersecting_bisectors (
    ABCD : Quadrilateral,
    convex : ABCD.isConvex,
    bisectors_intersect_AC : ∃ P : Point, 
      P ∈ LineSegment AC ∧ 
      P ∈ bisector (angle ABC) ∧ 
      P ∈ bisector (angle ADC)
  ) : ∃ Q : Point, 
      Q ∈ LineSegment BD ∧ 
      Q ∈ bisector (angle BAD) ∧ 
      Q ∈ bisector (angle BCD) := 
sorry

end convex_quadrilateral_intersecting_bisectors_l347_347736


namespace solve_for_x_l347_347982

theorem solve_for_x (x : ℚ) (h : (x + 10) / (x - 4) = (x + 3) / (x - 6)) : x = 48 / 5 :=
sorry

end solve_for_x_l347_347982


namespace distinguish_ball_box_ways_l347_347348

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347348


namespace ways_to_distribute_balls_l347_347425

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347425


namespace binom_coeff_exists_l347_347586

theorem binom_coeff_exists (n : ℕ) (r : ℤ) (r_odd : r % 2 = 1) : 
  ∃ i, i < 2^n ∧ (nat.choose (2^n + i) i) % 2^(n+1) = r % 2^(n+1) := 
sorry

end binom_coeff_exists_l347_347586


namespace ways_to_distribute_balls_l347_347450

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347450


namespace radius_of_inscribed_circle_in_triangle_l347_347693

noncomputable def inscribed_circle_radius (PQ PR QR : ℝ) (hPQ : PQ = 8) (hPR : PR = 8) (hQR : QR = 10) : ℝ :=
  let s := (PQ + PR + QR) / 2
  let K := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  let r := K / s
  r

theorem radius_of_inscribed_circle_in_triangle : inscribed_circle_radius 8 8 10 8.refl 8.refl 10.refl = 5 * Real.sqrt 39 / 13 :=
  by
    -- proof goes here
    sorry

end radius_of_inscribed_circle_in_triangle_l347_347693


namespace balls_into_boxes_l347_347333

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347333


namespace remainder_of_sum_mod_13_l347_347698

theorem remainder_of_sum_mod_13 :
  ∀ (D : ℕ) (k1 k2 : ℕ),
    D = 13 →
    (242 = k1 * D + 8) →
    (698 = k2 * D + 9) →
    (242 + 698) % D = 4 :=
by
  intros D k1 k2 hD h242 h698
  sorry

end remainder_of_sum_mod_13_l347_347698


namespace laplace_operator_eigenvalues_and_multiplicities_l347_347820

theorem laplace_operator_eigenvalues_and_multiplicities 
  (R : ℝ) (n : ℕ) (k : ℕ) (k_pos : 0 ≤ k) 
  : eigenvalues (laplace_operator (sphere R n)) k 
  = -((k * (k + n - 2)) / (R^2)) 
  ∧ multiplicity (laplace_operator (sphere R n)) k 
  = (nat.choose (n + k - 1) k) - (nat.choose (n + k - 3) (k - 2)) :=
sorry

end laplace_operator_eigenvalues_and_multiplicities_l347_347820


namespace inequality_holds_for_all_x_in_interval_l347_347171

theorem inequality_holds_for_all_x_in_interval (a b : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^2 + a * x + b| ≤ 1 / 8) ↔ (a = -1 ∧ b = 1 / 8) :=
sorry

end inequality_holds_for_all_x_in_interval_l347_347171


namespace f_l347_347894

def f (x : ℝ) : ℝ := a * x ^ 4 + b * x ^ 2 + c

-- Given conditions
variables (a b c : ℝ)
axiom h : (4 * a + 2 * b = 2)

theorem f'_neg_one : derivative (f) (-1) = -2 :=
by
  sorry

end f_l347_347894


namespace num_triangles_from_decagon_l347_347114

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347114


namespace triangle_side_length_divisibility_l347_347587

-- Defining the problem statement
theorem triangle_side_length_divisibility
  (p : ℕ) (n : ℕ) (h_odd_prime : Nat.Prime p) (h_odd : p % 2 = 1) (h_pos : 0 < n) :
  ∃ (pts : Fin 8 → ℤ × ℤ), 
    (∀ i : Fin 8, let (x, y) := pts i in x^2 + y^2 = p^(2 * n)) ∧
    ∃ (i j k : Fin 8), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    let (xi, yi) := pts i in
    let (xj, yj) := pts j in
    let (xk, yk) := pts k in
    (xi - xj)^2 + (yi - yj)^2 ≡ 0 [MOD p^(n+1)] ∧
    (xi - xk)^2 + (yi - yk)^2 ≡ 0 [MOD p^(n+1)] ∧
    (xj - xk)^2 + (yj - yk)^2 ≡ 0 [MOD p^(n+1)] :=
sorry


end triangle_side_length_divisibility_l347_347587


namespace ways_to_distribute_balls_l347_347414

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347414


namespace quadratic_not_divisible_by_49_l347_347622

theorem quadratic_not_divisible_by_49 (n : ℤ) : ¬ (n^2 + 3 * n + 4) % 49 = 0 := 
by
  sorry

end quadratic_not_divisible_by_49_l347_347622


namespace cylinder_on_sphere_l347_347640

noncomputable def cylinder_volume (d_cyl : ℝ) (v_sph : ℝ) (d_sph_to_cyl : ℝ) : ℝ :=
  let r_sph := real.sqrt ((3 * v_sph) / (4 * π)) in
  let r_cyl := d_cyl / 2 in
  let h_cyl := 2 * real.sqrt (r_sph^2 - r_cyl^2) in
  π * r_cyl^2 * h_cyl

theorem cylinder_on_sphere (d_cyl : ℝ) (v_sph : ℝ) (volume : ℝ) :
  d_cyl = 8 → v_sph = (500 * π) / 3 → volume = 96 * π → cylinder_volume d_cyl v_sph volume = 96 * π :=
by
  intro h_d_cyl h_v_sph h_volume
  rw [h_d_cyl, h_v_sph, h_volume]
  sorry

end cylinder_on_sphere_l347_347640


namespace remainder_n_plus_2023_l347_347700

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 :=
sorry

end remainder_n_plus_2023_l347_347700


namespace regular_decagon_triangle_count_l347_347031

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347031


namespace balls_into_boxes_l347_347329

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347329


namespace balls_into_boxes_l347_347338

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347338


namespace lemonade_proportion_l347_347529

-- The problem statement rephrased as a Lean declaration
theorem lemonade_proportion (lemons_per_40_gal: ℚ) (sugar_per_40_gal: ℚ) (gallons: ℚ)
  (h_lemons: lemons_per_40_gal = 30) (h_sugar: sugar_per_40_gal = 5) (h_gallons: gallons = 10):
  ∃ x y, x = 7.5 ∧ y = 1.25 := 
by
  have h1 : (30 / 40 : ℚ) = (7.5 / 10 : ℚ) := sorry
  have h2 : (5 / 40 : ℚ) = (1.25 / 10 : ℚ) := sorry
  use [7.5, 1.25]
  split; assumption

end lemonade_proportion_l347_347529


namespace SUCCESS_arrangement_count_l347_347784

theorem SUCCESS_arrangement_count : 
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  ∃ (ways_to_arrange : ℕ), ways_to_arrange = Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_C) := 420 :=
by
  -- Problem Conditions
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  -- The Proof
  existsi 420
  sorry

end SUCCESS_arrangement_count_l347_347784


namespace num_triangles_from_decagon_l347_347012

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347012


namespace solution_is_correct_l347_347216

noncomputable def domain := ℝ
def f (x : domain) : domain

-- Conditions
axiom f_defined_on_R : ∀ x : domain, x ∈ domain
axiom f_graph_passes_through_point : (1, 1) ∈ ( { p : domain × domain | p.2 = f p.1 } )
axiom f_derivative_gt_neg2 : ∀ x : domain, deriv f x > -2

-- Question
def solution_set : Set domain := { x | f (Real.log 2 (3^x - 1)) < 3 - (Real.log (Real.sqrt 2) (3^x - 1)) }

-- Correct answer
def correct_solution_set : Set domain := { x | x < 1 ∧ x ≠ 0 }

-- Proof
theorem solution_is_correct :
  (sol_set = (Set.Ioo (-∞) 0 ∪ Set.Ioo 0 1)) :=
sorry

end solution_is_correct_l347_347216


namespace six_digit_palindromes_base8_count_l347_347825

noncomputable def count_six_digit_palindromes_base8 : ℕ :=
  let a_choices := 7 -- choices for a
  let b_choices := 8 -- choices for b
  let c_choices := 8 -- choices for c
  let d_choices := 8 -- choices for d
  in a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindromes_base8_count : count_six_digit_palindromes_base8 = 3584 := by
  sorry

end six_digit_palindromes_base8_count_l347_347825


namespace ways_to_distribute_balls_l347_347444

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347444


namespace triangles_from_decagon_l347_347102

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347102


namespace smallest_n_integer_quotient_l347_347178

theorem smallest_n_integer_quotient :
  ∃ n : ℕ, n ≥ 1 ∧ ( ∑ i in finset.range (n + 1), i) / ( ∑ i in finset.range (2 * n + 1), if i > n then i else 0) = 1 :=
sorry

end smallest_n_integer_quotient_l347_347178


namespace ways_to_distribute_balls_l347_347370

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347370


namespace distinct_real_roots_probability_l347_347575

theorem distinct_real_roots_probability :
  let outcomes := (finset.range 6).map (λ n, n.succ)
  let condition := λ (a : ℕ), a^2 - 8 > 0
  let favorable_outcomes := outcomes.filter condition
  (favorable_outcomes.card : ℚ) / outcomes.card = 2 / 3 :=
by
  sorry

end distinct_real_roots_probability_l347_347575


namespace ways_to_place_balls_in_boxes_l347_347503

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347503


namespace ball_in_boxes_l347_347277

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347277


namespace area_ratio_of_triangle_and_trapezoid_l347_347917

-- Define the geometric entities and the conditions
noncomputable def TrapezoidAreaRatio : Rat :=
  let AB := 3 * CD
  let BN := 3 * NC
  let AM := 2 * MB
  if h: ratio(AreaTriangleAPM, AreaTrapezoidABCD) = 4 / 25 then
    (ratio(AreaTriangleAPM, AreaTrapezoidABCD) = 4 / 25)
  else
    false

theorem area_ratio_of_triangle_and_trapezoid (AB CD BC P: ℝ) 
  (h1: AB / CD = 3 / 2)
  (BN NC AM MB: ℝ)
  (h2: BN = 3 * NC)
  (h3: AM = 2 * MB)
  (AreaTriangleAPM AreaTrapezoidABCD: ℝ): 
  ratio(AreaTriangleAPM, AreaTrapezoidABCD) = 4 / 25 := 
  by
  sorry

end area_ratio_of_triangle_and_trapezoid_l347_347917


namespace polygon_side_count_l347_347683

theorem polygon_side_count (s : ℝ) (hs : s ≠ 0) : 
  ∀ (side_length_ratio : ℝ) (sides_first sides_second : ℕ),
  sides_first = 50 ∧ side_length_ratio = 3 ∧ 
  sides_first * side_length_ratio * s = sides_second * s → sides_second = 150 :=
by
  sorry

end polygon_side_count_l347_347683


namespace problem_statement_l347_347833

-- Definition of sum of digits function
def S (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Definition of the function f₁
def f₁ (k : ℕ) : ℕ :=
  (S k) ^ 2

-- Definition of the function fₙ₊₁
def f : ℕ → ℕ → ℕ
| 0, k => k
| (n+1), k => f₁ (f n k)

-- Theorem stating the proof problem
theorem problem_statement : f 2005 (2 ^ 2006) = 169 :=
  sorry

end problem_statement_l347_347833


namespace divisibility_of_product_l347_347940

theorem divisibility_of_product (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a ∣ b^3) (h2 : b ∣ c^3) (h3 : c ∣ a^3) : abc ∣ (a + b + c) ^ 13 := by
  sorry

end divisibility_of_product_l347_347940


namespace suit_price_after_discount_l347_347654

-- Define the original price of the suit.
def original_price : ℝ := 150

-- Define the increase rate and the discount rate.
def increase_rate : ℝ := 0.20
def discount_rate : ℝ := 0.20

-- Define the increased price after the 20% increase.
def increased_price : ℝ := original_price * (1 + increase_rate)

-- Define the final price after applying the 20% discount.
def final_price : ℝ := increased_price * (1 - discount_rate)

-- Prove that the final price is $144.
theorem suit_price_after_discount : final_price = 144 := by
  sorry  -- Proof to be completed

end suit_price_after_discount_l347_347654


namespace initial_dimes_of_A_l347_347719

theorem initial_dimes_of_A (a b c : ℕ) (h1 : 4 * (a - b - c) = 36) (h2 : -2 * a + 6 * b - 2 * c = 36) (h3 : -3 * a - b + 9 * c = 36) :
  a = 36 :=
by
  sorry

end initial_dimes_of_A_l347_347719


namespace num_triangles_from_decagon_l347_347112

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347112


namespace midpoint_coordinates_l347_347816

theorem midpoint_coordinates (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 2) (hy1 : y1 = 9) (hx2 : x2 = 8) (hy2 : y2 = 3) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx, my) = (5, 6) :=
by
  rw [hx1, hy1, hx2, hy2]
  sorry

end midpoint_coordinates_l347_347816


namespace ball_in_boxes_l347_347272

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347272


namespace num_triangles_from_decagon_l347_347002

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347002


namespace construct_triangle_with_symmetric_orthocenter_points_l347_347146

-- Definition of point type and triangle type
structure Point : Type where
  x : ℝ
  y : ℝ

structure Triangle : Type where
  A : Point
  B : Point
  C : Point

-- Definition of orthocenter and symmetric points in an acute-angled triangle
def is_orthocenter (H : Point) (T : Triangle) : Prop := sorry

def is_symmetric_to_orthocenter (A' B' C' : Point) (H : Point) (T : Triangle) : Prop := sorry

def is_acute_triangle (T : Triangle) : Prop := sorry

-- The theorem statement
theorem construct_triangle_with_symmetric_orthocenter_points
  (A' B' C' : Point)
  (acute : ∀ T : Triangle, is_acute_triangle T) 
  (symmetric : ∀ T H : Triangle, is_symmetric_to_orthocenter A' B' C' H T) :
  ∃ (T : Triangle), acute T ∧ symmetric T (is_orthocenter A B C) :=
sorry

end construct_triangle_with_symmetric_orthocenter_points_l347_347146


namespace balls_in_boxes_l347_347294

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347294


namespace trains_clear_each_other_time_l347_347685

def length_first_train : ℝ := 431
def length_second_train : ℝ := 365
def speed_first_train_kmh : ℝ := 120
def speed_second_train_kmh : ℝ := 105

def speed_first_train_ms : ℝ := speed_first_train_kmh * (1000 / 3600)
def speed_second_train_ms : ℝ := speed_second_train_kmh * (1000 / 3600)

def total_length : ℝ := length_first_train + length_second_train
def relative_speed : ℝ := speed_first_train_ms + speed_second_train_ms

theorem trains_clear_each_other_time : 
  total_length / relative_speed = 12.736 := by
  sorry

end trains_clear_each_other_time_l347_347685


namespace midpoint_coordinates_l347_347817

theorem midpoint_coordinates (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 2) (hy1 : y1 = 9) (hx2 : x2 = 8) (hy2 : y2 = 3) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx, my) = (5, 6) :=
by
  rw [hx1, hy1, hx2, hy2]
  sorry

end midpoint_coordinates_l347_347817


namespace common_area_of_45_45_90_triangles_l347_347678

def is_45_45_90_triangle (leg hypotenuse : ℝ) : Prop :=
  hypotenuse = leg * sqrt 2

def before_sliding (hypotenuse : ℝ) : Prop := hypotenuse = 16

def slide_distance (distance : ℝ) : Prop := distance = 6

def remaining_hypotenuse (original_hypotenuse slide_distance : ℝ) : ℝ :=
  original_hypotenuse - slide_distance

theorem common_area_of_45_45_90_triangles
  (leg hypotenuse : ℝ) 
  (h1 : is_45_45_90_triangle leg hypotenuse) 
  (h2 : before_sliding hypotenuse)
  (d : ℝ)
  (h3 : slide_distance d) :
  let new_hypotenuse := remaining_hypotenuse hypotenuse d in 
  is_45_45_90_triangle (new_hypotenuse / sqrt 2) new_hypotenuse →
  (1/2) * (new_hypotenuse / sqrt 2) * (new_hypotenuse / sqrt 2) = 25 :=
by
  sorry

end common_area_of_45_45_90_triangles_l347_347678


namespace continuous_difference_ineq_l347_347638

open Real

noncomputable theory

variable (f g : ℝ → ℝ)
variables (a b : ℝ)

theorem continuous_difference_ineq
  (hf : ContinuousOn f (Icc a b))
  (hg : ContinuousOn g (Icc a b))
  (hfa : f a = g a)
  (hdf : ∀ x ∈ Ioo a b, DifferentiableAt ℝ f x)
  (hdg : ∀ x ∈ Ioo a b, DifferentiableAt ℝ g x)
  (hderiv : ∀ x ∈ Ioo a b, deriv f x > deriv g x) :
  ∀ x ∈ Ioo a b, f x > g x :=
by
  sorry

end continuous_difference_ineq_l347_347638


namespace number_of_triangles_in_regular_decagon_l347_347120

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347120


namespace leo_remaining_words_l347_347558

variables (words_per_line : ℕ) (lines_per_page : ℕ) (pages_written : ℚ) (total_words_required : ℕ)

theorem leo_remaining_words :
  words_per_line = 10 →
  lines_per_page = 20 →
  pages_written = 1.5 →
  total_words_required = 400 →
  total_words_required - (words_per_line * lines_per_page * pages_written).to_nat = 100 :=
by
  intro h_words_per_line h_lines_per_page h_pages_written h_total_words_required
  rw [h_words_per_line, h_lines_per_page, h_pages_written, h_total_words_required]
  norm_num
  sorry

end leo_remaining_words_l347_347558


namespace evaluate_floor_ceil_l347_347799

theorem evaluate_floor_ceil :
  (⌊1.999⌋ : ℤ) + (⌈3.001⌉ : ℤ) = 5 :=
by
  sorry

end evaluate_floor_ceil_l347_347799


namespace number_of_triangles_in_regular_decagon_l347_347133

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347133


namespace sack_flour_cost_l347_347686

theorem sack_flour_cost
  (x y : ℝ) 
  (h1 : 10 * x + 800 = 108 * y)
  (h2 : 4 * x - 800 = 36 * y) : x = 1600 := by
  -- Add your proof here
  sorry

end sack_flour_cost_l347_347686


namespace no_L_shaped_partition_of_10x10_board_l347_347552

-- Definitions
def board : Type := ℕ × ℕ -- Representing a 10 x 10 board as pairs of natural numbers.

def is_L_shape (shape : finset (ℕ × ℕ)) : Prop := 
  ∃ (a b c d : ℕ × ℕ), 
    shape = {a, b, c, d} ∧ 
    -- Assuming the coordinates a, b, c, d form an "L"
    ((a.1 = b.1 ∧ b.1 = c.1 ∧ a.2 + 1 = b.2 ∧ b.2 + 1 = c.2 ∧ c.2 = d.2 ∧ c.1 + 1 = d.1) ∨
    (a.2 = b.2 ∧ b.2 = c.2 ∧ a.1 + 1 = b.1 ∧ b.1 + 1 = c.1 ∧ c.1 = d.1 ∧ c.2 + 1 = d.2))

def valid_partition (partition : finset (finset (ℕ × ℕ))) : Prop :=
  partition.card = 25 ∧ 
  (∀ shape ∈ partition, is_L_shape shape) ∧ -- Each part must be an L-shape
  (finset.bUnion partition id = finset.univ.filter (λ ⟨x, y⟩, x < 10 ∧ y < 10)) -- Partition covers exactly the 10x10 board

-- Theorem statement
theorem no_L_shaped_partition_of_10x10_board : ¬∃ partition : finset (finset (ℕ × ℕ)), valid_partition partition :=
sorry

end no_L_shaped_partition_of_10x10_board_l347_347552


namespace expand_expression_l347_347166

def p(x : ℝ) : ℝ := 4 * x ^ 3 - 3 * x ^ 2 + 2 * x - 7

theorem expand_expression (x : ℝ) : 5 * p(x) = 20 * x ^ 3 - 15 * x ^ 2 + 10 * x - 35 := 
by 
  sorry

end expand_expression_l347_347166


namespace decagon_triangle_count_l347_347053

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347053


namespace intersect_lines_BE_CF_l347_347930

variable {V : Type} [AddCommGroup V] [Module ℝ V]

variables (A B C E F P : V)
variables (x y z : ℝ)
variables (wAE wEC wAF wFB : ℝ) -- ratios

-- Given conditions
def pt_E (A C : V) (wAE wEC : ℝ) : V := (wAE / (wAE + wEC)) • A + (wEC / (wAE + wEC)) • C
def pt_F (A B : V) (wAF wFB : ℝ) : V := (wAF / (wAF + wFB)) • A + (wFB / (wAF + wFB)) • B

-- Let E be a point on AC such that the ratio AE:EC = 3:2
def E_def : V := pt_E A C 3 2

-- Let F be a point on AB such that the ratio AF:FB = 2:3
def F_def : V := pt_F A B 2 3

-- Define the point P as x A + y B + z C where x + y + z = 1
def P_def : V := x • A + y • B + z • C

-- Lean statement to prove the coordinates of P
theorem intersect_lines_BE_CF 
  (hE : E = E_def)
  (hF : F = F_def)
  (hx : x = 9 / 35)
  (hy : y = 5 / 35)
  (hz : z = 6 / 35) 
  (hxyz : x + y + z = 1) :
  ∃ P, P = P_def := sorry

end intersect_lines_BE_CF_l347_347930


namespace quadratic_equation_solution_l347_347516

-- We want to prove that for the conditions given, the only possible value of m is 3
theorem quadratic_equation_solution (m : ℤ) (h1 : m^2 - 7 = 2) (h2 : m + 3 ≠ 0) : m = 3 :=
sorry

end quadratic_equation_solution_l347_347516


namespace sequence_solution_l347_347199

-- Defining the sequence and the condition
def sequence_condition (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 1

-- Defining the sequence formula we need to prove
def sequence_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 ^ (n - 1)

theorem sequence_solution (a S : ℕ → ℝ) (h : sequence_condition a S) :
  sequence_formula a :=
by 
  sorry

end sequence_solution_l347_347199


namespace decagon_triangle_count_l347_347064

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347064


namespace find_XY_l347_347168

theorem find_XY (XYZ : Triangle) (h1 : is_30_60_90 XYZ) (h2 : ∠XYZ.X = 30) (h3 : XYZ.hypotenuse = 10) : 
  XYZ.side_opposite_30 = 5 := 
sorry

end find_XY_l347_347168


namespace exists_indices_divisible_2019_l347_347589

theorem exists_indices_divisible_2019 (x : Fin 2020 → ℤ) : 
  ∃ (i j : Fin 2020), i ≠ j ∧ (x j - x i) % 2019 = 0 := 
  sorry

end exists_indices_divisible_2019_l347_347589


namespace curve_C_equation_points_M_G_N_H_are_concyclic_l347_347922

-- Defining the points A, B, and transformation conditions
def A : (ℝ × ℝ) := (-1, 0)
def B : (ℝ × ℝ) := (1, 0)
def P (x y : ℝ) : (ℝ × ℝ) := (x, y)
def Q (x y : ℝ) : (ℝ × ℝ) := (x, sqrt 2 * y)

-- Vector operations
def vec (p1 p2 : ℝ × ℝ) : (ℝ × ℝ) := (p2.1 - p1.1, p2.2 - p1.2)
def dotProd (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Condition for the dot product
axiom AQ_dot_BQ_eq_one (x y : ℝ) 
  : dotProd (vec A (Q x y)) (vec B (Q x y)) = 1

-- Statement for the problem part I
theorem curve_C_equation (x y : ℝ) 
  (h : dotProd (vec A (Q x y)) (vec B (Q x y)) = 1) 
  : x^2 / 2 + y^2 = 1 := sorry

-- Points and conditions for part II
def H : (ℝ × ℝ) := (-1, -sqrt 2 / 2)
def G : (ℝ × ℝ) := (1, sqrt 2 / 2)
def O₁ : (ℝ × ℝ) := (1 / 8, -sqrt 2 / 8)
def radius : ℝ := 3 * sqrt 11 / 8

-- Check if points M, G, N, H are concyclic
theorem points_M_G_N_H_are_concyclic (x₁ y₁ x₂ y₂ : ℝ) 
  (hM_on_C : x₁^2 / 2 + y₁^2 = 1) 
  (hN_on_C : x₂^2 / 2 + y₂^2 = 1) 
  (h_sum_x : x₁ + x₂ = 1) 
  (h_sum_y : y₁ + y₂ = sqrt 2 / 2) 
  : (vec O₁ H).1^2 + (vec O₁ H).2^2 = radius^2 
    ∧ (vec O₁ (x₁, y₁)).1^2 + (vec O₁ (x₁, y₁)).2^2 = radius^2 := sorry

end curve_C_equation_points_M_G_N_H_are_concyclic_l347_347922


namespace max_value_of_function_l347_347172

noncomputable def function_to_maximize (x : ℝ) : ℝ :=
  (Real.sin x)^4 + (Real.cos x)^4 + 1 / ((Real.sin x)^2 + (Real.cos x)^2 + 1)

theorem max_value_of_function :
  ∃ x : ℝ, function_to_maximize x = 7 / 4 :=
sorry

end max_value_of_function_l347_347172


namespace circumradius_AMD_eq_r_l347_347774

variables {A B C D M O1 O2 : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M]
variables {r : ℝ}

noncomputable def is_circumradius := sorry

axiom circle_O1 : ∀ (X : Type) [MetricSpace X], dist O1 X = r
axiom circle_O2 : ∀ (X : Type) [MetricSpace X], dist O2 X = r

axiom passes_through_O1 : dist O1 A = r ∧ dist O1 B = r
axiom passes_through_O2 : dist O2 B = r ∧ dist O2 C = r

axiom intersect_at_M : ∀ (P : Type) [MetricSpace P], dist O1 P = r ∧ dist O2 P = r → P = M

def parallelogram (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
  dist A B = dist C D ∧ dist A D = dist B C ∧ dist B D = dist A C ∧ angle A B C = angle D C B

axiom parallelogram_ABCD : parallelogram A B C D

theorem circumradius_AMD_eq_r :
  is_circumradius A M D r :=
begin
  sorry
end

end circumradius_AMD_eq_r_l347_347774


namespace power_calculation_l347_347689

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l347_347689


namespace permutation_sorted_l347_347838

theorem permutation_sorted (n : ℕ) (a : fin n → ℕ) (condition : ∀ (i j : fin n), i < j → a i > a j → (true)) :
  ∃ b : fin n → ℕ, (∀ (i j : fin n), i < j → b i ≤ b j) :=
begin
  sorry
end

end permutation_sorted_l347_347838


namespace ways_to_put_balls_in_boxes_l347_347488

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347488


namespace solve_equation_naturals_l347_347716

theorem solve_equation_naturals :
  ∀ (X Y Z : ℕ), X^Y + Y^Z = X * Y * Z ↔ 
    (X = 1 ∧ Y = 1 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 4) := 
by
  sorry

end solve_equation_naturals_l347_347716


namespace distinct_balls_boxes_l347_347380

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347380


namespace quadratic_if_and_only_if_l347_347903

theorem quadratic_if_and_only_if (m : ℤ) : 
  ((m - 4) * x ^ (abs (m - 2)) + 2 * x - 5 = 0) ∧ (abs (m - 2) = 2) ∧ (m ≠ 4) → m = 0 :=
by
  sorry

end quadratic_if_and_only_if_l347_347903


namespace number_of_triangles_in_regular_decagon_l347_347119

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347119


namespace six_units_away_has_two_solutions_l347_347904

-- Define point A and its position on the number line
def A_position : ℤ := -3

-- Define the condition for a point x being 6 units away from point A
def is_6_units_away (x : ℤ) : Prop := abs (x + 3) = 6

-- The theorem stating that if x is 6 units away from -3, then x must be either 3 or -9
theorem six_units_away_has_two_solutions (x : ℤ) (h : is_6_units_away x) : x = 3 ∨ x = -9 := by
  sorry

end six_units_away_has_two_solutions_l347_347904


namespace ways_to_put_balls_in_boxes_l347_347482

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347482


namespace ways_to_distribute_balls_l347_347412

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347412


namespace area_of_shape_enclosed_by_curves_l347_347991

noncomputable def area_enclosed_by_curves : ℝ :=
  2 * ∫ x in 0..1, (2 - 2*x^2 - (x^2 - 1))

theorem area_of_shape_enclosed_by_curves : area_enclosed_by_curves = 4 := by
  sorry

end area_of_shape_enclosed_by_curves_l347_347991


namespace number_of_triangles_in_regular_decagon_l347_347130

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347130


namespace eight_lines_no_parallel_no_concurrent_l347_347795

-- Define the number of regions into which n lines divide the plane
def regions (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 2
else n * (n - 1) / 2 + n + 1

theorem eight_lines_no_parallel_no_concurrent :
  regions 8 = 37 :=
by
  sorry

end eight_lines_no_parallel_no_concurrent_l347_347795


namespace balls_in_boxes_l347_347285

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347285


namespace johnny_net_income_is_correct_l347_347971

noncomputable def johnny_net_income : ℝ :=
let day1_income := 3 * 7 + 2 * 10 + 4 * 12,
    day2_income := 3 * 9 + 2 * 9 + 4 * 12,
    day3_income := 3 * 7 + 2 * 10 + 4 * 15,
    day4_income := 3 * 4 + 2 * 12 + 4 * 15,
    day5_income := 3 * 7 + 2 * 10 + 4 * 12,
    total_income := day1_income + day2_income + day3_income + day4_income + day5_income,
    expenses := 5 * 5 + 20,
    earnings_after_expenses := total_income - expenses,
    taxes := 0.15 * earnings_after_expenses
in earnings_after_expenses - taxes

theorem johnny_net_income_is_correct : johnny_net_income = 359.55 :=
by
  have h_day1_income : 3 * 7 + 2 * 10 + 4 * 12 = 89 := rfl
  have h_day2_income : 3 * 9 + 2 * 9 + 4 * 12 = 93 := rfl
  have h_day3_income : 3 * 7 + 2 * 10 + 4 * 15 = 101 := rfl
  have h_day4_income : 3 * 4 + 2 * 12 + 4 * 15 = 96 := rfl
  have h_day5_income : 3 * 7 + 2 * 10 + 4 * 12 = 89 := rfl
  have h_total_income : 89 + 93 + 101 + 96 + 89 = 468 := rfl
  have h_expenses : 5 * 5 + 20 = 45 := rfl
  have h_earnings_after_expenses : 468 - 45 = 423 := rfl
  have h_taxes : 0.15 * 423 = 63.45 := rfl
  have h_net_income : 423 - 63.45 = 359.55 := rfl
  exact h_net_income

end johnny_net_income_is_correct_l347_347971


namespace trig_identity_l347_347808

theorem trig_identity (a b c : ℝ) : 
  sin (a + b) - sin (a - c) = 2 * cos (a + (b - c) / 2) * sin ((b + c) / 2) :=
sorry

end trig_identity_l347_347808


namespace mom_chicken_cost_l347_347965

def cost_bananas : ℝ := 2 * 4 -- bananas cost
def cost_pears : ℝ := 2 -- pears cost
def cost_asparagus : ℝ := 6 -- asparagus cost
def total_expenses_other_than_chicken : ℝ := cost_bananas + cost_pears + cost_asparagus -- total cost of other items
def initial_money : ℝ := 55 -- initial amount of money
def remaining_money_after_other_purchases : ℝ := initial_money - total_expenses_other_than_chicken -- money left after covering other items

theorem mom_chicken_cost : 
  (remaining_money_after_other_purchases - 28 = 11) := 
by
  sorry

end mom_chicken_cost_l347_347965


namespace area_triangle_DEF_union_area_triangle_DEPrimeFPrimes_l347_347931

/-- Given the following conditions:
1. Triangle DEF with sides DE = 5, EF = 12, and DF = 13.
2. Point H is the centroid of triangle DEF.
3. Triangle DEF is rotated 180 degrees around centroid H to form triangle D'E'F'.

Prove that the area of the union of triangles DEF and D'E'F' is 60. -/
theorem area_triangle_DEF_union_area_triangle_DEPrimeFPrimes
  {D E F : Type} 
  (DE EF DF : ℝ)
  (H : Type)
  (H_centroid : H)
  (DEF_is_right_triangle : DE^2 + EF^2 = DF^2)
  (rotation_180_deg : is_rotation H 180 (D, E, F) = (D', E', F')) :
  area_union DEF D'E'F' = 60 := 
sorry

end area_triangle_DEF_union_area_triangle_DEPrimeFPrimes_l347_347931


namespace selection_methods_l347_347675

theorem selection_methods (n_classes n_spots : ℕ) (h_classes : n_classes = 3) (h_spots : n_spots = 5) :
  n_spots ^ n_classes = 5^3 :=
by
  rw [h_classes, h_spots]
  exact rfl

end selection_methods_l347_347675


namespace num_triangles_in_decagon_l347_347074

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347074


namespace transactions_Mabel_l347_347604

variable {M A C J : ℝ}

theorem transactions_Mabel (h1 : A = 1.10 * M)
                          (h2 : C = 2 / 3 * A)
                          (h3 : J = C + 18)
                          (h4 : J = 84) :
  M = 90 :=
by
  sorry

end transactions_Mabel_l347_347604


namespace solution_set_of_inequality_l347_347226

open Real

noncomputable def f : ℝ → ℝ := sorry

def is_solution_set (s : Set ℝ) : Prop :=
∀ x : ℝ, (x > 0) → (x ∈ s ↔ f x ≤ ln x)

theorem solution_set_of_inequality :
  f 1 = 0 ∧ (∀ x > 0, x * (deriv f x) > 1) → is_solution_set (Set.Ioc 0 1) 
:= by
  intro h
  sorry

end solution_set_of_inequality_l347_347226


namespace balls_into_boxes_l347_347325

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347325


namespace A_eq_B_l347_347583

-- Define what it means for a sequence to be of type A
def isSeqA (a : List ℕ) : Prop :=
  a.sorted (· ≥ ·) ∧
  (∀ i, i ∈ a → (∃ k, i + 1 = 2^k)) ∧
  (a.sum = n)

-- Define what it means for a sequence to be of type B
def isSeqB (b : List ℕ) : Prop :=
  b.sorted (· ≥ ·) ∧
  (∀ (j : ℕ), j < b.length - 1 → b[j] ≥ 2 * b[j+1]) ∧
  (b.sum = n)

-- Define the number of sequences A(n)
def A (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ l, isSeqA l).length

-- Define the number of sequences B(n)
def B (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ l, isSeqB l).length

-- State the theorem
theorem A_eq_B (n : ℕ) : A n = B n :=
  sorry

end A_eq_B_l347_347583


namespace f_100_value_l347_347565

noncomputable def sum_of_divisors (n : ℕ) (f : ℕ → ℕ) : ℕ :=
∑ d in (Finset.divisors n), f d 

theorem f_100_value : 
  (∀ n : ℕ, n = sum_of_divisors n f) → f 100 = 40 :=
by
  sorry

end f_100_value_l347_347565


namespace angle_between_given_vectors_l347_347243

noncomputable def angle_between_vectors 
  (a : ℝ × ℝ) (b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  let magnitude_a := real.sqrt (a.1 ^ 2 + a.2 ^ 2) in
  let magnitude_b := real.sqrt (b.1 ^ 2 + b.2 ^ 2) in
  real.arccos (dot_product / (magnitude_a * magnitude_b))

theorem angle_between_given_vectors :
  angle_between_vectors (⟨ √3 / 2, 1 / 2 ⟩) (⟨ √3, -1 ⟩) = π / 3 :=
by
  sorry

end angle_between_given_vectors_l347_347243


namespace balls_into_boxes_l347_347404

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347404


namespace number_of_triangles_in_decagon_l347_347018

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347018


namespace distinct_balls_boxes_l347_347460

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347460


namespace balls_into_boxes_l347_347341

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347341


namespace socks_different_colors_l347_347506

theorem socks_different_colors (n_white n_brown n_blue n_red : ℕ) 
  (h_white : n_white = 5) (h_brown : n_brown = 5) 
  (h_blue : n_blue = 2) (h_red : n_red = 1) : 
  (n_white * n_brown + n_white * n_blue + n_white * n_red + n_brown * n_blue + n_brown * n_red + n_blue * n_red) = 57 :=
by
  rw [h_white, h_brown, h_blue, h_red]
  show 5 * 5 + 5 * 2 + 5 * 1 + 5 * 2 + 5 * 1 + 2 * 1 = 57
  sorry

end socks_different_colors_l347_347506


namespace sequence_sum_S_n_l347_347217

theorem sequence_sum_S_n (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n, S 0 = 0) ∧ S 1 = -2/3 ∧
  (∀ n, n ≥ 2 → S n + (1 / S n) + 2 = a n) ∧
  a 1 = -2/3 →
  (∀ n, S n = -(n + 1) / (n + 2)) :=
by sorry

end sequence_sum_S_n_l347_347217


namespace cos_squared_alpha_minus_pi_over_4_l347_347844

theorem cos_squared_alpha_minus_pi_over_4 (α : ℝ) (h : sin (2 * α) = 1 / 3) : 
  cos^2 (α - π / 4) = 2 / 3 :=
by
  sorry

end cos_squared_alpha_minus_pi_over_4_l347_347844


namespace line_no_common_points_parallel_l347_347195

variable (m : Line) (α : Plane)

theorem line_no_common_points_parallel (h : ∀ p : Point, p ∉ α ∨ p ∉ m) : m // α :=
sorry

end line_no_common_points_parallel_l347_347195


namespace euler_line_parallel_bisector_angle_l347_347724

theorem euler_line_parallel_bisector_angle (A B C : Point) (h_non_isosceles : ¬is_isosceles A B C) 
(h_parallel : ∃ O H : Point, is_circumcenter O A B C ∧ is_orthocenter H A B C ∧ is_parallel (line_through O H) (internal_bisector A B C)) :
angle A B C = 120 := sorry

end euler_line_parallel_bisector_angle_l347_347724


namespace distinguish_ball_box_ways_l347_347354

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347354


namespace operation_commutative_operation_associative_l347_347152

def my_operation (a b : ℝ) : ℝ := a * b + a + b

theorem operation_commutative (a b : ℝ) : my_operation a b = my_operation b a := by
  sorry

theorem operation_associative (a b c : ℝ) : my_operation (my_operation a b) c = my_operation a (my_operation b c) := by
  sorry

end operation_commutative_operation_associative_l347_347152


namespace number_of_triangles_in_regular_decagon_l347_347123

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347123


namespace sum_of_six_angles_l347_347528

theorem sum_of_six_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle3 + angle5 = 180)
  (h2 : angle2 + angle4 + angle6 = 180) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 :=
by
  sorry

end sum_of_six_angles_l347_347528


namespace number_of_triangles_in_decagon_l347_347080

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347080


namespace ways_to_distribute_balls_l347_347446

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347446


namespace inequality_solution_l347_347985

open Real

theorem inequality_solution (x : ℝ) (x1 x2 : ℝ) (h1 : x1 = (-9 - sqrt 21) / 2) (h2 : x2 = (-9 + sqrt 21) / 2) :
  (x - 1) / (x + 3) > (4 * x + 5) / (3 * x + 8) ↔ (x ∈ Ioo (-3) x1 ∪ Ioi x2) := by
  sorry

end inequality_solution_l347_347985


namespace locus_of_points_theorem_l347_347882

variable {V : Type} [inner_product_space ℝ V] [finite_dimensional ℝ V]

def locus_of_points (A B M : V) (k l d : ℝ) : Prop :=
  k * ∥A - M∥^2 + l * ∥B - M∥^2 = d

theorem locus_of_points_theorem 
  (A B : V) (k l d : ℝ) (h : k + l ≠ 0) :
  ∃ c : set V, (∀ M : V, locus_of_points A B M k l d ↔ M ∈ c) ∧ 
               (c = ∅ ∨ ∃ center : V, ∃ r : ℝ, c = {M | ∥M - center∥ = r} ∨ (c = {A})) :=
sorry

end locus_of_points_theorem_l347_347882


namespace geometric_sequence_a7_l347_347544

variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

-- Given condition
axiom geom_seq_condition : a 4 * a 10 = 9

-- proving the required result
theorem geometric_sequence_a7 (h : is_geometric_sequence a r) : a 7 = 3 ∨ a 7 = -3 :=
by
  sorry

end geometric_sequence_a7_l347_347544


namespace number_of_triangles_in_decagon_l347_347079

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347079


namespace find_salary_May_l347_347993

-- Define the salaries for each month as variables
variables (J F M A May : ℝ)

-- Declare the conditions as hypotheses
def avg_salary_Jan_to_Apr := (J + F + M + A) / 4 = 8000
def avg_salary_Feb_to_May := (F + M + A + May) / 4 = 8100
def salary_Jan := J = 6100

-- The theorem stating the salary for the month of May
theorem find_salary_May (h1 : avg_salary_Jan_to_Apr J F M A) (h2 : avg_salary_Feb_to_May F M A May) (h3 : salary_Jan J) :
  May = 6500 :=
  sorry

end find_salary_May_l347_347993


namespace kiwi_count_l347_347739

theorem kiwi_count (s b o k : ℕ)
  (h1 : s + b + o + k = 340)
  (h2 : s = 3 * b)
  (h3 : o = 2 * k)
  (h4 : k = 5 * s) :
  k = 104 :=
sorry

end kiwi_count_l347_347739


namespace distinct_balls_boxes_l347_347391

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347391


namespace triangle_a_c_sin_A_minus_B_l347_347593

theorem triangle_a_c_sin_A_minus_B (a b c : ℝ) (A B C : ℝ):
  a + c = 6 → b = 2 → Real.cos B = 7/9 →
  a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27 :=
by
  intro h1 h2 h3
  sorry

end triangle_a_c_sin_A_minus_B_l347_347593


namespace num_ways_to_distribute_balls_into_boxes_l347_347260

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347260


namespace ways_to_place_balls_in_boxes_l347_347494

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347494


namespace probability_not_below_x_axis_of_parallelogram_l347_347614

structure Point where
  x : ℝ
  y : ℝ

def P : Point := {x := 4, y := 4}
def Q : Point := {x := -2, y := -2}
def R : Point := {x := -8, y := -2}
def S : Point := {x := -2, y := 4}

theorem probability_not_below_x_axis_of_parallelogram (P Q R S : Point) 
  (hP : P = {x := 4, y := 4})
  (hQ : Q = {x := -2, y := -2})
  (hR : R = {x := -8, y := -2})
  (hS : S = {x := -2, y := 4}) :
  probability_not_below_x_axis (parallelogram P Q R S) = 1 / 2 := by
  sorry

end probability_not_below_x_axis_of_parallelogram_l347_347614


namespace function_symmetry_f1_l347_347517

theorem function_symmetry_f1 {θ : ℝ} (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f(x) = 3 * Real.cos (Real.pi * x + θ))
  (h2 : ∀ x : ℝ, f(x) = f(2 - x)) :
  f 1 = 3 ∨ f 1 = -3 :=
by
  sorry

end function_symmetry_f1_l347_347517


namespace ria_number_is_2_l347_347890

theorem ria_number_is_2 
  (R S : ℕ) 
  (consecutive : R = S + 1 ∨ S = R + 1) 
  (R_positive : R > 0) 
  (S_positive : S > 0) 
  (R_not_1 : R ≠ 1) 
  (Sylvie_does_not_know : S ≠ 1) 
  (Ria_knows_after_Sylvie : ∃ (R_known : ℕ), R_known = R) :
  R = 2 :=
sorry

end ria_number_is_2_l347_347890


namespace max_x_solutions_l347_347821

theorem max_x_solutions :
  (∃ y : ℤ, 3^2 - 3 * y - 2 * y^2 = 9) ∧
  (∀ x : ℤ, ∃ y : ℤ, x^2 - x * y - 2 * y^2 = 9 → x ≤ 3) :=
begin
  sorry
end

end max_x_solutions_l347_347821


namespace balls_in_boxes_l347_347289

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347289


namespace ball_in_boxes_l347_347275

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347275


namespace triangles_from_decagon_l347_347103

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347103


namespace segment_parallel_to_x_axis_l347_347228

theorem segment_parallel_to_x_axis 
  (f : ℤ → ℤ) 
  (h_poly : ∃ n : ℕ, ∃ a : ℕ → ℤ, ∀ x : ℤ, f x = ∑ i in Finset.range (n + 1), a i * x^i)
  (c d : ℤ)
  (h_int_dist : ∃ (D : ℕ), ↑D = (int.natAbs (c - d)) + (int.natAbs (f c - f d))) :
  f c = f d :=
begin
  sorry,
end

end segment_parallel_to_x_axis_l347_347228


namespace balls_into_boxes_l347_347322

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347322


namespace problem1_problem2_l347_347847

-- Definitions from the problem's conditions
variables (x y : ℝ) (A B : Set ℝ)
def A_set := {x^2 + 2, -x, - (x + 1)} 
def B_set := {-y, - (y / 2), y + 1}

-- Condition x > 0 and y > 0
axiom x_pos : x > 0
axiom y_pos : y > 0

-- Problem 1: Prove that x^2 + y^2 = 5 given A = B
axiom A_eq_B : A_set x = B_set y

theorem problem1 : x^2 + y^2 = 5 :=
by sorry

-- Problem 2: Prove that A ∪ B = {-2, -3, -5, -5/2, 6} given A ∩ B = {6}
axiom A_inter_B : (A_set x ∩ B_set y) = {6}

theorem problem2 : (A_set x ∪ B_set y) = {-2, -3, -5, - (5 / 2), 6} :=
by sorry

end problem1_problem2_l347_347847


namespace relationship_among_a_b_c_l347_347720

def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f(x) = f(-x)

def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f(x) < f(y)

noncomputable def f : ℝ → ℝ := sorry

theorem relationship_among_a_b_c
  (h_even : is_even f)
  (h_increasing : is_increasing_on f (set.Iic 0)) :
  let a := f(real.log 7 / real.log 4)
      b := f(- real.log 3 / real.log (1/2))
      c := f(0.2 ^ 0.6)
  in c > a ∧ a > b :=
sorry

end relationship_among_a_b_c_l347_347720


namespace oliver_kept_stickers_l347_347602

theorem oliver_kept_stickers (initial_stickers : ℕ) 
(used_fraction : ℚ) 
(gave_fraction : ℚ) 
(h_initial : initial_stickers = 135) 
(h_used : used_fraction = 1/3) 
(h_gave : gave_fraction = 2/5) : 
∃ (kept_stickers : ℕ), kept_stickers = 54 := 
by 
-- Skip the steps and just assert the existence of the proof.
sory

end oliver_kept_stickers_l347_347602


namespace put_balls_in_boxes_l347_347313

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347313


namespace total_area_wdws_approx_l347_347729

-- Define constants based on conditions
def length_rect_pane := 12 -- inches
def width_rect_pane := 8 -- inches
def num_panes := 8
def base_triangle := 10 -- inches
def height_triangle := 12 -- inches
def diameter_half_circle := 14 -- inches
def pi := 3.14159 -- Approximate value of π

-- Define total area calculation
noncomputable def total_area_wdws (num_panes : ℕ) (length_rect_pane width_rect_pane base_triangle height_triangle radius_half_circle pi : ℝ) : ℝ :=
  let area_rect_pane := length_rect_pane * width_rect_pane  
  let area_rect_window := num_panes * area_rect_pane  
  let area_triangle := (base_triangle * height_triangle) / 2
  let area_half_circle := (pi * (radius_half_circle ^ 2)) / 2
  area_rect_window + area_triangle + area_half_circle

-- Radius calculation for half-circular window
def radius_half_circle := diameter_half_circle / 2 

theorem total_area_wdws_approx :
  total_area_wdws num_panes length_rect_pane width_rect_pane base_triangle height_triangle radius_half_circle pi ≈ 904.97 := sorry

end total_area_wdws_approx_l347_347729


namespace area_under_curve_distance_variable_speed_motion_work_done_by_variable_force_l347_347780

-- Let's represent the mathematical statements, converting them to Lean.
variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem area_under_curve (h : ∀ x ∈ set.Icc a b, f x < 0) : 
  ∀ a b, |∫ x in set.Icc a b, f x| = ∫ x in set.Icc a b, -f x :=
begin
  sorry
end

theorem distance_variable_speed_motion :
  ¬ ∀ v : ℝ → ℝ, (∀ t, v t ≥ 0) → ∀ (a b : ℝ), ∫ t in set.Icc a b, |v t| = ∫ t in set.Icc a b, v t :=
begin
  sorry
end

theorem work_done_by_variable_force (F : ℝ → ℝ) (d : ℝ → ℝ) :
  ¬ (∀ x, ∫ t in set.Icc 0 1, F (d t) = ∫ t in set.Icc 0 1, F (d t) * d' t) :=
begin
  sorry
end

end area_under_curve_distance_variable_speed_motion_work_done_by_variable_force_l347_347780


namespace ways_to_put_balls_in_boxes_l347_347484

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347484


namespace magnitude_of_c_l347_347885

noncomputable def vector_projection (u v: ℝ × ℝ): ℝ :=
  (u.1 * v.1 + u.2 * v.2) / (u.1 * u.1 + u.2 * u.2)

def magnitude (v: ℝ × ℝ): ℝ :=
  (v.1 ^ 2 + v.2 ^ 2).sqrt

theorem magnitude_of_c
  (a b c: ℝ × ℝ)
  (ha: a = (1,0))
  (hb: b = (1,2))
  (h_projection: vector_projection a c = 2)
  (h_parallel: ∃ k: ℝ, c = (k * b.1, k * b.2)):
  magnitude c = 2 * (5:ℝ).sqrt :=
sorry

end magnitude_of_c_l347_347885


namespace distinct_balls_boxes_l347_347458

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347458


namespace find_ellipse_equation_line_passes_through_fixed_point_l347_347855

-- The first part of the problem: finding the equation of the ellipse.
theorem find_ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = a * (sqrt 3 / 2))
(h4 : (1 / 2) * b * (2 * c) = sqrt 3) : (a = 2) ∧ (b = 1) ∧ (c = sqrt 3) ∧ (∀ x y, ((x^2 / 4) + y^2 = 1 ↔ (x^2 / a^2 + y^2 / b^2 = 1))) :=
sorry

-- The second part of the problem: verifying if the line passes through a fixed point.
theorem line_passes_through_fixed_point (l k m : ℝ) (A B P : ℝ × ℝ) (h1 : P = (0, 1)) (h2 : ∃ x1 x2, 
((4*k^2 + 1)*x1*x1 + 8*k*m*x1 + 4*(m*m - 1) = 0) ∧ ((4*k^2 + 1)*x2*x2 + 8*k*m*x2 + 4*(m*m - 1) = 0) ∧
((P.snd - 1)/P.fst + (A.snd - 1)/A.fst = 2) ∧ ((A.fst + B.fst) = (-8*m*k / (4*k*k+1)) ∧ (A.fst * B.fst = (4*(m*m-1) / (4*k*k+1)))) : 
∃ (M : ℝ × ℝ), M = (-1, -1) :=
sorry

end find_ellipse_equation_line_passes_through_fixed_point_l347_347855


namespace mary_remaining_money_after_shopping_l347_347969

def initial_balance := 200
def video_game_price := 60
def video_game_discount := 0.15
def goggles_spending_percentage := 0.20
def goggles_sales_tax := 0.08
def jacket_price := 80
def jacket_discount := 0.25
def book_spending_percentage := 0.10
def book_sales_tax := 0.05
def gift_card_amount := 20

noncomputable def remaining_money_after_shopping : ℝ :=
  let video_game_cost := video_game_price * (1 - video_game_discount)
  let remaining_after_video_game := initial_balance - video_game_cost
  let goggles_cost := (remaining_after_video_game * goggles_spending_percentage) * (1 + goggles_sales_tax)
  let remaining_after_goggles := remaining_after_video_game - goggles_cost
  let jacket_cost := jacket_price * (1 - jacket_discount)
  let remaining_after_jacket := remaining_after_goggles - jacket_cost
  let book_cost := (remaining_after_jacket * book_spending_percentage) * (1 + book_sales_tax)
  let remaining_after_book := remaining_after_jacket - book_cost
  let remaining_after_socks := remaining_after_book
  remaining_after_socks

theorem mary_remaining_money_after_shopping : remaining_money_after_shopping = 50.85 :=
by 
  sorry

end mary_remaining_money_after_shopping_l347_347969


namespace number_of_triangles_in_regular_decagon_l347_347142

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347142


namespace magnitude_of_2a_plus_b_l347_347214

open Real

variables (a b : ℝ × ℝ) (angle : ℝ)

-- Conditions
axiom angle_between_a_b (a b : ℝ × ℝ) : angle = π / 3 -- 60 degrees in radians
axiom norm_a_eq_1 (a : ℝ × ℝ) : ‖a‖ = 1
axiom b_eq (b : ℝ × ℝ) : b = (3, 0)

-- Theorem
theorem magnitude_of_2a_plus_b (h1 : angle = π / 3) (h2 : ‖a‖ = 1) (h3 : b = (3, 0)) :
  ‖2 • a + b‖ = sqrt 19 :=
sorry

end magnitude_of_2a_plus_b_l347_347214


namespace decagon_triangle_count_l347_347056

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347056


namespace parallel_lines_slope_l347_347649

theorem parallel_lines_slope (a : ℝ) : (∀ (x y : ℝ), (x + 2 * a * y - 5 = 0) → (a * x + 4 * y + 2 = 0)) → (a = real.sqrt(2) ∨ a = -real.sqrt(2)) :=
by
  sorry

end parallel_lines_slope_l347_347649


namespace sin_3_plus_exp_cos_eq_sin_3_plus_16_l347_347179

theorem sin_3_plus_exp_cos_eq_sin_3_plus_16 :
  sin 3 + 2^(8 - 3) * cos (Real.pi / 3) = sin 3 + 16 := 
by 
  sorry

end sin_3_plus_exp_cos_eq_sin_3_plus_16_l347_347179


namespace first_discount_percentage_l347_347731

theorem first_discount_percentage :
  ∃ x : ℝ, (9649.12 * (1 - x / 100) * 0.9 * 0.95 = 6600) ∧ (19.64 ≤ x ∧ x ≤ 19.66) :=
sorry

end first_discount_percentage_l347_347731


namespace no_such_function_exists_l347_347809

noncomputable def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ 
| 0, x     => x
| (n+1), x => f (iterate n x)

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, (∀ n : ℕ, iterate f n n = n + 1) := sorry

end no_such_function_exists_l347_347809


namespace percentage_of_first_to_second_l347_347712

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) :
  first = 0.06 * X →
  second = 0.30 * X →
  (first / second) * 100 = 20 :=
by
  intros h1 h2
  sorry

end percentage_of_first_to_second_l347_347712


namespace balls_into_boxes_l347_347339

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347339


namespace monotonicity_f_on_interval_l347_347653

def f (x : ℝ) : ℝ := |x + 2|

theorem monotonicity_f_on_interval :
  ∀ x1 x2 : ℝ, x1 < x2 → x1 < -4 → x2 < -4 → f x1 ≥ f x2 :=
by
  sorry

end monotonicity_f_on_interval_l347_347653


namespace decagon_triangle_count_l347_347065

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347065


namespace bh_length_is_3_l347_347916

noncomputable def length_bh (x : ℝ) (side_length : ℝ) (half_side_length : ℝ) : Prop :=
  let g_distance := half_side_length in
  let bg_length := side_length - x in
  (bg_length^2 = x^2 + g_distance^2)

theorem bh_length_is_3 :
  length_bh 3 8 4 :=
by
  unfold length_bh
  sorry

end bh_length_is_3_l347_347916


namespace distinct_balls_boxes_l347_347384

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347384


namespace angles_sum_to_90_degrees_l347_347762

/-- Given two right triangles ∆ABE and ∆CAD with right-angle vertices at A,
  show that ∠DAC + ∠BAE equals 90 degrees. -/
theorem angles_sum_to_90_degrees
  (A B C D E : Type) (A_right_angle : ∠BAE = 90)
  (lineaire_combined_angles : ∠DAC + ∠BAE = 180)
  (line_right_angle : ∠BAC + ∠CAD + ∠BAE = 180) :
  ∠DAC + ∠BAE = 90 :=
by
  sorry

end angles_sum_to_90_degrees_l347_347762


namespace ways_to_distribute_balls_l347_347369

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347369


namespace zero_of_composite_function_l347_347222

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then -2 * Real.exp(x) else Real.log(x)

theorem zero_of_composite_function :
  f (f (Real.exp 1)) = 0 :=
by
  sorry

end zero_of_composite_function_l347_347222


namespace cannot_determine_right_triangle_l347_347761

theorem cannot_determine_right_triangle (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = 180) →
  (\<^angle A = B + C) →
  (5 * a = 13 * c ∧ 12 * b = 13 * c) →
  (a^2 = (b+c) * (b-c)) →
  (\<^angle A = 3 * x ∧ \<^angle B = 4 * x ∧ \<^angle C = 5 * x) →
  (12 * x = 180) →
  (x ≠ 15 → ∃ (A B C : ℝ), A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90) :=
by sorry

end cannot_determine_right_triangle_l347_347761


namespace distinct_balls_boxes_l347_347472

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347472


namespace find_sum_of_sequence_l347_347949

theorem find_sum_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, S n = ∑ k in finset.range n, a (k + 1))
  (h3 : ∀ n, n * a (n + 1) + S (n + 1) = n * S n) :
  ∀ n, S n = 2^(n-1) / n :=
begin
  sorry
end

end find_sum_of_sequence_l347_347949


namespace polyhedron_volume_of_folded_shapes_l347_347924

-- Defining the conditions
def isosceles_right_triangle (a : ℝ) : Prop :=
  ∃ (A E F : Point), is_triangle A E F ∧ is_right_angle A E F ∧ 
  dist A E = a ∧ dist E F = a

def square (a : ℝ) : Prop :=
  ∃ (B C D : Point), is_square B C D ∧
  dist B C = a

def regular_hexagon (a : ℝ) : Prop :=
  ∃ (G : Point), is_regular_hexagon G ∧
  side_length G = a

-- Proving the desired volume calculation under these conditions
theorem polyhedron_volume_of_folded_shapes :
  (isosceles_right_triangle 2) ∧ (isosceles_right_triangle 2) ∧ (isosceles_right_triangle 2) ∧
  (square 2) ∧ (square 2) ∧ (square 2) ∧
  (regular_hexagon (real.sqrt 8)) →
  volume_of_folded_polyhedron = 47 / 6 :=
by
  sorry

end polyhedron_volume_of_folded_shapes_l347_347924


namespace max_value_f_l347_347651

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 - 3 * cos x + 2

theorem max_value_f : ∃ (x : ℝ), f x = 5 := 
sorry

end max_value_f_l347_347651


namespace neg_exists_eq_forall_l347_347881

theorem neg_exists_eq_forall (p : Prop) :
  (∀ x : ℝ, ¬(x^2 + 2*x = 3)) ↔ ¬(∃ x : ℝ, x^2 + 2*x = 3) := 
by
  sorry

end neg_exists_eq_forall_l347_347881


namespace put_balls_in_boxes_l347_347310

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347310


namespace curve_defined_by_theta_eq_pi_div_4_is_line_l347_347891

theorem curve_defined_by_theta_eq_pi_div_4_is_line :
  (∀ (r : ℝ), ∃ (x y : ℝ), (x, y) = (r * cos (π / 4), r * sin (π / 4)) → ∃ (a b : ℂ), a = (1 : ℂ) ∧ b = (1 : ℂ) ∧ line := 
  by
    sorry

end curve_defined_by_theta_eq_pi_div_4_is_line_l347_347891


namespace put_balls_in_boxes_l347_347303

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347303


namespace evaluate_floor_ceil_l347_347801

theorem evaluate_floor_ceil :
  (⌊1.999⌋ : ℤ) + (⌈3.001⌉ : ℤ) = 5 :=
by
  sorry

end evaluate_floor_ceil_l347_347801


namespace count_ordered_pairs_without_0_or_5_l347_347824

/-- 
There are 4105 ordered pairs of positive integers (a, b) such that:
1. a + b = 5000
2. a does not contain the digits 0 or 5
3. b does not contain the digits 0 or 5
-/
theorem count_ordered_pairs_without_0_or_5 : 
  (finset.range 4999).card 
    (λ a, (¬(0 ∈ a.digits 10 ∨ 5 ∈ a.digits 10) ∧ 
           ¬(0 ∈ (5000 - a).digits 10 ∨ 5 ∈ (5000 - a).digits 10)))
    = 4105 := 
  sorry

end count_ordered_pairs_without_0_or_5_l347_347824


namespace sum_f_geq_sqrt2p_sub2_l347_347566

open BigOperators
open Nat

-- Define the necessary variables
variables {ℤ : Type*} {f : ℕ → ℕ → ℤ}

-- Define the conditions based on the problem statement
def f_conditions (f : ℕ → ℕ → ℤ) : Prop :=
  (f 1 1 = 0) ∧ 
  (∀ a b : ℕ, coprime a b → (a ≠ 1 ∨ b ≠ 1) → f a b + f b a = 1) ∧
  (∀ a b : ℕ, coprime a b → f (a + b) b = f a b)

-- State the main theorem to be proven
theorem sum_f_geq_sqrt2p_sub2 (p : ℕ) [fact (p.prime)] (h_odd : odd p) :
  f_conditions f →
  ∑ n in finset.range p, f (n^2) p ≥ sqrt (2 * p) - 2 :=
sorry

end sum_f_geq_sqrt2p_sub2_l347_347566


namespace distinguish_ball_box_ways_l347_347355

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347355


namespace circle_tangency_problem_l347_347864

noncomputable def circle_center (h : ℝ → ℝ → ℝ) (x₀ y₀ : ℝ) : Prop :=
  ∀ x y, h x y = (x - x₀)^2 + (y - y₀)^2

noncomputable def internally_tangent (radius1 radius2 dist : ℝ) : Prop :=
  dist = radius2 - radius1

theorem circle_tangency_problem 
  (C1 : ℝ → ℝ → ℝ)
  (C2 : ℝ → ℝ → ℝ)
  (hC1 : circle_center C1 0 0)
  (hC2 : circle_center C2 3 4)
  (dist_centers : ℝ)
  (h_d : ∀ x y, dist_centers = Math.sqrt ((3 - 0)^2 + (4 - 0)^2))
  (radius_C1 : ℝ := 1)
  (radius_C2 : ℝ)
  (h_rC2 : ∀ n, C2 x y = (x - 3)^2 + (y - 4)^2 - (25 - n))
  (h_internally_tangent : internally_tangent radius_C1 radius_C2 dist_centers) :
  ∃ n, radius_C2 = Math.sqrt (25 - n) ∧ n = -11 :=
begin
  sorry
end

end circle_tangency_problem_l347_347864


namespace sin_neg_eq_neg_sin_l347_347832

variable (α : ℝ)

theorem sin_neg_eq_neg_sin : sin (-α) = - sin α :=
sorry

end sin_neg_eq_neg_sin_l347_347832


namespace balls_into_boxes_l347_347345

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347345


namespace distinct_balls_boxes_l347_347382

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347382


namespace cookies_with_chocolate_chips_l347_347964

-- Define the conditions given in the problem
def num_cookies_with_nuts_needed (total_nuts : ℕ) (nuts_per_cookie : ℕ) : ℕ :=
  total_nuts / nuts_per_cookie

def percent_of_cookies_with_nuts_wanted (total_cookies : ℕ) : ℚ :=
  1 / 4 * total_cookies

def num_cookies_with_nuts_and_chips (total_nuts : ℕ) (nuts_per_cookie : ℕ) (total_cookies : ℕ) : ℕ :=
  num_cookies_with_nuts_needed total_nuts nuts_per_cookie - percent_of_cookies_with_nuts_wanted total_cookies

def num_cookies_with_just_chips (total_cookies : ℕ) (total_nuts : ℕ) (nuts_per_cookie : ℕ) : ℕ :=
  total_cookies - (percent_of_cookies_with_nuts_wanted total_cookies) - (num_cookies_with_nuts_and_chips total_nuts nuts_per_cookie total_cookies)

def percent_cookies_with_just_chips (total_cookies : ℕ) (total_nuts : ℕ) (nuts_per_cookie : ℕ) : ℚ :=
  (num_cookies_with_just_chips total_cookies total_nuts nuts_per_cookie) / total_cookies * 100

-- Main theorem to prove
theorem cookies_with_chocolate_chips
  (total_cookies : ℕ) (total_nuts : ℕ) (nuts_per_cookie : ℕ) (h_total_cookies : total_cookies = 60) (h_total_nuts : total_nuts = 72) 
  (h_nuts_per_cookie : nuts_per_cookie = 2) :
  percent_cookies_with_just_chips total_cookies total_nuts nuts_per_cookie = 40 :=
by
  sorry

end cookies_with_chocolate_chips_l347_347964


namespace derivative_f1_derivative_f2_derivative_f3_derivative_f4_l347_347819

open Real

noncomputable def f1 (x : ℝ) := 4 * x + 1 / x
noncomputable def f2 (x : ℝ) := exp x * sin x
noncomputable def f3 (x : ℝ) := log x / x
noncomputable def f4 (x : ℝ) := cos (2 * x + 5)

theorem derivative_f1 : deriv f1 = λ x, 4 - 1 / x^2 :=
by
  -- proof of the derivative
  sorry

theorem derivative_f2 : deriv f2 = λ x, exp x * sin x + exp x * cos x :=
by
  -- proof of the derivative
  sorry

theorem derivative_f3 : deriv f3 = λ x, (1 - log x) / x^2 :=
by
  -- proof of the derivative
  sorry

theorem derivative_f4 : deriv f4 = λ x, -2 * sin (2 * x + 5) :=
by
  -- proof of the derivative
  sorry

end derivative_f1_derivative_f2_derivative_f3_derivative_f4_l347_347819


namespace balls_into_boxes_l347_347342

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347342


namespace simplify_expression_l347_347979

theorem simplify_expression (y : ℝ) : 
  (3 * y) ^ 3 - 2 * y * y ^ 2 + y ^ 4 = 25 * y ^ 3 + y ^ 4 :=
by
  sorry

end simplify_expression_l347_347979


namespace hyperbola_vector_norm_sum_is_2sqrt17_l347_347858

noncomputable def hyperbola : Type := {
  a : ℝ, -- semi-major axis
  b : ℝ, -- semi-minor axis
  c : ℝ, -- distance from the center to each focus
  f1 : (ℝ × ℝ), -- left focus
  f2 : (ℝ × ℝ), -- right focus
  P : (ℝ × ℝ), -- point on the hyperbola
  on_hyperbola : (P.1 ^ 2) / 9 - (P.2 ^ 2) / 4 = 1, -- point P is on the hyperbola
  orthogonal : (P.1 - f1.1) * (P.1 - f2.1) + (P.2 - f1.2) * (P.2 - f2.2) = 0 -- orthogonality condition
}

def vector_norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

theorem hyperbola_vector_norm_sum_is_2sqrt17 (H : hyperbola) :
  vector_norm (vector_add (H.P.1 - H.f1.1, H.P.2 - H.f1.2) (H.P.1 - H.f2.1, H.P.2 - H.f2.2)) = 2 * real.sqrt 17 := by
  sorry

end hyperbola_vector_norm_sum_is_2sqrt17_l347_347858


namespace cannot_determine_right_triangle_l347_347754

-- Definitions of conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (a b c : ℝ) : Prop := a/b = 5/12 ∧ b/c = 12/13
def condition_C (a b c : ℝ) : Prop := a^2 = (b + c) * (b - c)
def condition_D (A B C : ℝ) : Prop := A/B = 3/4 ∧ B/C = 4/5

-- The proof problem
theorem cannot_determine_right_triangle (a b c A B C : ℝ)
  (hD : condition_D A B C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end cannot_determine_right_triangle_l347_347754


namespace number_of_triangles_in_decagon_l347_347025

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347025


namespace mean_of_two_means_eq_l347_347652

theorem mean_of_two_means_eq (z : ℚ) (h : (5 + 10 + 20) / 3 = (15 + z) / 2) : z = 25 / 3 :=
by
  sorry

end mean_of_two_means_eq_l347_347652


namespace put_balls_in_boxes_l347_347308

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347308


namespace change_calculation_l347_347932

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple = 4.25) := by
  sorry

end change_calculation_l347_347932


namespace gcd_12m_18n_l347_347510

theorem gcd_12m_18n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_gcd_mn : m.gcd n = 10) : (12 * m).gcd (18 * n) = 60 := by
  sorry

end gcd_12m_18n_l347_347510


namespace sum_of_first_11_terms_l347_347923

variable (a : Nat → Int)

-- Given condition: a₆ = 1
axiom a6_eq_1 : a 6 = 1

-- Definition of the sum of the first 11 terms of the arithmetic sequence
def S₁₁ : Int := ∑ i in Finset.range 11, a i

-- Theorem to be proven
theorem sum_of_first_11_terms : S₁₁ = 11 := 
  by
    sorry

end sum_of_first_11_terms_l347_347923


namespace number_of_triangles_in_regular_decagon_l347_347126

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347126


namespace Riemann_property_a_Riemann_property_c_Riemann_property_d_l347_347989

def RiemannFunc (x : ℝ) : ℝ :=
  if h : ∃ (p q : ℕ) (hq : q ≠ 0), x = p / q ∧ Nat.coprime p q then
    let ⟨p, q, hq, hx, _⟩ := h
    1 / q
  else
    0

theorem Riemann_property_a : RiemannFunc (2 / 3) = 1 / 3 := sorry
theorem Riemann_property_c (x : ℝ) : x ∈ set.Icc 0 1 → RiemannFunc x = RiemannFunc (1 - x) := sorry
theorem Riemann_property_d (a b : ℝ) : a ∈ set.Icc 0 1 → b ∈ set.Icc 0 1 → RiemannFunc (a * b) ≥ RiemannFunc a * RiemannFunc b := sorry

end Riemann_property_a_Riemann_property_c_Riemann_property_d_l347_347989


namespace ways_to_distribute_balls_l347_347424

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347424


namespace num_triangles_from_decagon_l347_347113

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347113


namespace good_integers_bound_l347_347974

--Define a constant independent of p
constant c : ℝ

-- The statement of the theorem
theorem good_integers_bound (p : ℕ) (hp : Nat.Prime p ∧ p % 2 = 1) : 
  ∃ c > 0, ∀ n : ℕ, (∃ k < p, p ∣ k! + 1) → n ≤ (c * (p : ℝ)^(2/3)) := 
sorry

end good_integers_bound_l347_347974


namespace num_ways_to_distribute_balls_into_boxes_l347_347262

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347262


namespace balls_into_boxes_l347_347324

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347324


namespace profit_percentage_is_correct_l347_347711

noncomputable def cost_price (SP : ℝ) : ℝ := 0.81 * SP

noncomputable def profit (SP CP : ℝ) : ℝ := SP - CP

noncomputable def profit_percentage (profit CP : ℝ) : ℝ := (profit / CP) * 100

theorem profit_percentage_is_correct (SP : ℝ) (h : SP = 100) :
  profit_percentage (profit SP (cost_price SP)) (cost_price SP) = 23.46 :=
by
  sorry

end profit_percentage_is_correct_l347_347711


namespace ball_in_boxes_l347_347276

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347276


namespace tetrahedron_volume_l347_347206

variable {R : ℝ}
variable {S1 S2 S3 S4 : ℝ}
variable {V : ℝ}

theorem tetrahedron_volume (R : ℝ) (S1 S2 S3 S4 V : ℝ) :
  V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l347_347206


namespace num_triangles_from_decagon_l347_347005

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347005


namespace total_distance_is_59_4_l347_347977

-- Define the distances driven by each of the three men
def Renaldo_distance := 15
def Ernesto_distance := 1 / 3 * Renaldo_distance + 7
def total_RE_distance := Renaldo_distance + Ernesto_distance
def Marcos_distance := 1.2 * total_RE_distance

-- Define total distance driven by all three men
def total_distance := Renaldo_distance + Ernesto_distance + Marcos_distance

-- State that this distance must be 59.4
theorem total_distance_is_59_4 : total_distance = 59.4 := by sorry

end total_distance_is_59_4_l347_347977


namespace find_a_l347_347513

-- Define the problem conditions
variables (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a^b = b^a)
variable (h4 : b = 4 * a)

-- Define the theorem to prove
theorem find_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 4 * a) : 
  a = real.cbrt 4 := 
sorry

end find_a_l347_347513


namespace simplify_fraction_l347_347980

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by
  sorry

end simplify_fraction_l347_347980


namespace find_constants_eq_l347_347811

theorem find_constants_eq (P Q R : ℚ)
  (h : ∀ x, (x^2 - 5) = P * (x - 4) * (x - 6) + Q * (x - 1) * (x - 6) + R * (x - 1) * (x - 4)) :
  (P = -4 / 15) ∧ (Q = -11 / 6) ∧ (R = 31 / 10) :=
by
  sorry

end find_constants_eq_l347_347811


namespace number_of_triangles_in_regular_decagon_l347_347137

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347137


namespace ball_in_boxes_l347_347271

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347271


namespace inequality_solution_empty_l347_347521

theorem inequality_solution_empty {a : ℝ} :
  (∀ x : ℝ, ¬ (|x+2| + |x-1| < a)) ↔ a ≤ 3 :=
by
  sorry

end inequality_solution_empty_l347_347521


namespace number_of_triangles_in_regular_decagon_l347_347143

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347143


namespace number_of_ways_to_select_books_l347_347672

theorem number_of_ways_to_select_books :
  let bag1 := 4
  let bag2 := 5
  bag1 * bag2 = 20 :=
by
  sorry

end number_of_ways_to_select_books_l347_347672


namespace minimum_value_l347_347709

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℝ), (∃ (x : ℝ), x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) ∧ (a^2 + b^2 = 4 / 5)

-- This line states that the minimum possible value of a^2 + b^2, given the condition, is 4/5.
theorem minimum_value (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
  sorry

end minimum_value_l347_347709


namespace z_mul_conj_eq_two_l347_347859

def i : ℂ := complex.I
 
def z : ℂ := (1 - i) / (1 + i) ^ 2016 + i

def z_conj : ℂ := conj z

theorem z_mul_conj_eq_two : 
  z * z_conj = 2 := 
by sorry

end z_mul_conj_eq_two_l347_347859


namespace balls_in_boxes_l347_347295

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347295


namespace number_of_triangles_in_decagon_l347_347052

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347052


namespace ondra_homework_problems_l347_347661

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end ondra_homework_problems_l347_347661


namespace balls_into_boxes_l347_347344

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347344


namespace proof_part_a_proof_part_b_l347_347734

-- Define the daily work rates.
variable (x y z u : ℚ)

-- Conditions from the problem
axiom h1 : x + y + z = 1 / 12
axiom h2 : x + y + u = 1 / 15
axiom h3 : x + z + u = 1 / 18
axiom h4 : y + z + u = 1 / 20

-- Total time together without any absences
def total_work_time_no_absence : ℚ := x + y + z + u 

-- Each part of the problem
theorem proof_part_a : ∀ (a_absence_one_day : ℚ) (rem_work_three_workers : ℚ) (final_time_a : ℚ), 
  a_absence_one_day = 1 / 20 ∧
  rem_work_three_workers = 1 - a_absence_one_day ∧
  final_time_a = 12 + 7 / 46 →
  total_work_time_no_absence = 23 / 270 →
  (rem_work_three_workers / (3 * total_work_time_no_absence)) + 1 = final_time_a := 
sorry

theorem proof_part_b : ∀ (c_absence_last_day : ℚ) (rem_work_before_final_day : ℚ) (final_time_b : ℚ), 
  rem_work_before_final_day = 11 * total_work_time_no_absence ∧
  c_absence_last_day = (1 - rem_work_before_final_day) / (x + y + u) ∧
  final_time_b = 11 + 17 / 18 →
  total_work_time_no_absence = 23 / 270 →
  c_absence_last_day + 11 = final_time_b := 
sorry

end proof_part_a_proof_part_b_l347_347734


namespace number_of_triangles_in_regular_decagon_l347_347132

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347132


namespace num_triangles_from_decagon_l347_347105

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347105


namespace num_ways_to_distribute_balls_into_boxes_l347_347252

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347252


namespace decagon_triangle_count_l347_347058

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347058


namespace distinct_balls_boxes_l347_347387

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347387


namespace all_three_eq_have_common_root_l347_347968

noncomputable theory
open Classical

theorem all_three_eq_have_common_root (a b c : ℝ)
  (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
  (h3_1 : ∃ x, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0)
  (h3_2 : ∃ y, b * y^11 + c * y^4 + a = 0 ∧ c * y^11 + a * y^4 + b = 0)
  (h3_3 : ∃ z, c * z^11 + a * z^4 + b = 0 ∧ a * z^11 + b * z^4 + c = 0): 
  ∃ w, a * w^11 + b * w^4 + c = 0 ∧ b * w^11 + c * w^4 + a = 0 ∧ c * w^11 + a * w^4 + b = 0 :=
sorry

end all_three_eq_have_common_root_l347_347968


namespace length_CD_l347_347194

-- Defining the line l: x - sqrt(3) y + 6 = 0
def line (x y : ℝ) : Prop := x - (Real.sqrt 3) * y + 6 = 0

-- Defining the circle x^2 + y^2 = 12
def circle (x y : ℝ) : Prop := x^2 + y^2 = 12

-- Statement of the theorem
theorem length_CD : 
  ∃ A B C D : (ℝ × ℝ),
  circle A.1 A.2 ∧
  circle B.1 B.2 ∧
  line A.1 A.2 ∧
  line B.1 B.2 ∧
  (∃ Cx : ℝ, C = (Cx, 0) ∧ line A.1 A.2 ∧ ⟂ line Cx 0) ∧
  (∃ Dx : ℝ, D = (Dx, 0) ∧ line B.1 B.2 ∧ ⟂ line Dx 0) ∧
  dist C D = 4 :=
sorry

end length_CD_l347_347194


namespace concurrency_AX_BY_CZ_l347_347530

-- Define the necessary structures and conditions
variables (A B C M_a M_b M_c S X Y Z : Point)
variables (n : Triangle A B C)
variables (Mid_BC : midpoint n.BC M_a)
variables (Mid_CA : midpoint n.CA M_b)
variables (Mid_AB : midpoint n.AB M_c)
variables (Euler : Euler_line n S)
variables (Int_MaS : (line_through M_a S).second_intersection (nine_point_circle n) X)
variables (Int_MbS : (line_through M_b S).second_intersection (nine_point_circle n) Y)
variables (Int_McS : (line_through M_c S).second_intersection (nine_point_circle n) Z)

-- State the theorem
theorem concurrency_AX_BY_CZ 
  (hneq : ¬equilateral n)
  (hx : (line_through A X) ∪ (line_through B Y) ∪ (line_through C Z)) : 
  concurrency (line_through A X) (line_through B Y) (line_through C Z) :=
sorry

end concurrency_AX_BY_CZ_l347_347530


namespace balls_into_boxes_l347_347400

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347400


namespace convergent_a_l347_347151

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 2 else a (n - 1) ^ (1 + (n - 1) ^ (-3 / 2))

theorem convergent_a : ∃ L : ℝ, filter.tendsto (λ n, a n) filter.at_top (nhds L) :=
sorry

end convergent_a_l347_347151


namespace existence_of_correct_division_l347_347148

-- Define grid side length, total area, and figure area
def side_length : ℕ := 9
def total_area : ℕ := side_length * side_length
def area_per_figure : ℕ := total_area / 3

-- Define properties of the figures A, B, and C
variables (A B C : finset (fin side_length × fin side_length))

-- Define the conditions for the problem
def equal_area_condition : Prop := A.card = area_per_figure ∧ B.card = area_per_figure ∧ C.card = area_per_figure

def perimeter (s : finset (fin side_length × fin side_length)) : ℕ := 
  -- Perimeter calculation logic to be defined
  sorry

def perimeter_condition : Prop := perimeter C = perimeter A + perimeter B

theorem existence_of_correct_division : 
  ∃ (A B C : finset (fin side_length × fin side_length)), equal_area_condition A B C ∧ perimeter_condition A B C :=
by
  sorry

end existence_of_correct_division_l347_347148


namespace abundant_numbers_less_than_30_l347_347248

-- Define a function to calculate the sum of proper divisors of a number
def sum_proper_divisors (n : Nat) : Nat :=
  (List.range n).filter (fun d => d > 0 ∧ n % d = 0).sum

-- Define what it means to be an abundant number
def is_abundant (n : Nat) : Prop :=
  sum_proper_divisors n > n

-- Main theorem statement: proving the number of abundant numbers less than 30 is 4
theorem abundant_numbers_less_than_30 : (List.range 30).countp is_abundant = 4 := by
  sorry

end abundant_numbers_less_than_30_l347_347248


namespace perfect_square_divisor_sum_l347_347743

theorem perfect_square_divisor_sum : 
  (∃ m n : ℕ, gcd m n = 1 ∧ (m + n = 43) ∧ 
  (let num_divisors := 4032,
       num_perfect_square_divisors := 96 in 
   (m : ℚ) / n = num_perfect_square_divisors / num_divisors)) :=
sorry

end perfect_square_divisor_sum_l347_347743


namespace math_problem_l347_347707

-- Define the first part of the problem
def line_area_to_axes (line_eq : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  line_eq x y ∧ x = 4 ∧ y = -4

-- Define the second part of the problem
def line_through_fixed_point (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (m * x) + y + m = 0 ∧ x = -1 ∧ y = 0

-- Theorem combining both parts
theorem math_problem (line_eq : ℝ → ℝ → Prop) (m : ℝ) :
  (∃ x y, line_area_to_axes line_eq x y → 8 = (1 / 2) * 4 * 4) ∧ line_through_fixed_point m :=
sorry

end math_problem_l347_347707


namespace subtraction_of_negatives_l347_347718

theorem subtraction_of_negatives : (-1) - (-4) = 3 :=
by
  -- Proof goes here.
  sorry

end subtraction_of_negatives_l347_347718


namespace speed_including_stoppages_l347_347165

def train_speed_excluding_stoppages : ℝ := 45.0
def stoppage_time_per_hour : ℝ := 18.67

theorem speed_including_stoppages : 
  let running_time := (60.0 - stoppage_time_per_hour) / 60.0 in
  let distance_covered := train_speed_excluding_stoppages * running_time in
  distance_covered / 1 == 31.0 :=
by
  sorry

end speed_including_stoppages_l347_347165


namespace ways_to_distribute_balls_l347_347375

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347375


namespace distinguish_ball_box_ways_l347_347350

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347350


namespace example_problem_l347_347810

theorem example_problem (a b : ℕ) : a = 1 → a * (a + b) + 1 ∣ (a + b) * (b + 1) - 1 :=
by
  sorry

end example_problem_l347_347810


namespace curve_is_circle_l347_347818

noncomputable def curve_eq (θ : ℝ) : ℝ := 2 / (1 + Real.sin θ)

theorem curve_is_circle : ∀ (θ : ℝ), ∃ (x y : ℝ), 
  let r := curve_eq θ in
  r = Real.sqrt (x^2 + y^2) ∧ y = r * Real.sin θ ∧ r^2 + 2 * y * r + y^2 = 4 :=
begin
  sorry
end

end curve_is_circle_l347_347818


namespace calculate_polynomial_value_l347_347185

theorem calculate_polynomial_value (a a1 a2 a3 a4 a5 : ℝ) : 
  (∀ x : ℝ, (1 - x)^2 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) → 
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by 
  intro h
  sorry

end calculate_polynomial_value_l347_347185


namespace num_triangles_in_decagon_l347_347076

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347076


namespace balls_into_boxes_l347_347326

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347326


namespace distinct_balls_boxes_l347_347461

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347461


namespace balls_into_boxes_l347_347315

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347315


namespace solve_for_m_l347_347777

theorem solve_for_m (m : ℝ) (f g : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^2 - 2 * x + m) →
  (∀ x : ℝ, g x = x^2 - 2 * x + 9 * m) →
  f 2 = 2 * g 2 →
  m = 0 :=
  by
    intros hf hg hs
    sorry

end solve_for_m_l347_347777


namespace distinct_balls_boxes_l347_347468

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347468


namespace distinct_balls_boxes_l347_347388

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347388


namespace euler_line_l347_347620

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (A B C : triangle)

noncomputable def centroid : point A :=
sorry

noncomputable def orthocenter : point A :=
sorry

noncomputable def circumcenter : point A :=
sorry

theorem euler_line (triangle : Type) [metric_space triangle] (G H O : point triangle)
  (hG : G = centroid triangle)
  (hH : H = orthocenter triangle)
  (hO : O = circumcenter triangle) :
  collinear [O, G, H] :=
sorry

end euler_line_l347_347620


namespace power_calculation_l347_347691

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l347_347691


namespace base8_to_base10_sum_l347_347828

theorem base8_to_base10_sum (a b : ℕ) (h₁ : a = 1 * 8^3 + 4 * 8^2 + 5 * 8^1 + 3 * 8^0)
                            (h₂ : b = 5 * 8^2 + 6 * 8^1 + 7 * 8^0) :
                            ((a + b) = 2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0) →
                            (2 * 8^3 + 1 * 8^2 + 4 * 8^1 + 4 * 8^0 = 1124) :=
by {
  sorry
}

end base8_to_base10_sum_l347_347828


namespace fraction_of_emilys_coins_l347_347162

theorem fraction_of_emilys_coins {total_states : ℕ} (h1 : total_states = 30)
    {states_from_1790_to_1799 : ℕ} (h2 : states_from_1790_to_1799 = 9) :
    (states_from_1790_to_1799 / total_states : ℚ) = 3 / 10 := by
  sorry

end fraction_of_emilys_coins_l347_347162


namespace stewart_farm_horse_food_l347_347765

theorem stewart_farm_horse_food 
  (ratio : ℚ) (food_per_horse : ℤ) (num_sheep : ℤ) (num_horses : ℤ)
  (h1 : ratio = 5 / 7)
  (h2 : food_per_horse = 230)
  (h3 : num_sheep = 40)
  (h4 : ratio * num_horses = num_sheep) : 
  (num_horses * food_per_horse = 12880) := 
sorry

end stewart_farm_horse_food_l347_347765


namespace roberta_started_with_8_records_l347_347978

variable (R : ℕ)

def received_records := 12
def bought_records := 30
def total_received_and_bought := received_records + bought_records

theorem roberta_started_with_8_records (h : R + total_received_and_bought = 50) : R = 8 :=
by
  sorry

end roberta_started_with_8_records_l347_347978


namespace put_balls_in_boxes_l347_347299

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347299


namespace number_of_triangles_in_decagon_l347_347085

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347085


namespace gabe_playlist_count_l347_347843

open Nat

theorem gabe_playlist_count :
  let best_day := 3
  let raise_the_roof := 2
  let rap_battle := 3
  let total_playlist_time := best_day + raise_the_roof + rap_battle
  let total_ride_time := 40
  total_ride_time / total_playlist_time = 5 := by
  have h_playlist_time : total_playlist_time = best_day + raise_the_roof + rap_battle := rfl
  have h_total_playlist_time : total_playlist_time = 8 := by
    rw [h_playlist_time]
    norm_num
  have h_total_ride_time : total_ride_time = 40 := rfl
  rw [h_total_playlist_time, h_total_ride_time]
  norm_num
  sorry

end gabe_playlist_count_l347_347843


namespace ball_box_distribution_l347_347439

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347439


namespace find_k_max_product_l347_347221

theorem find_k_max_product : 
  (∃ k : ℝ, (3 : ℝ) * (x ^ 2) - 4 * x + k = 0 ∧ 16 - 12 * k ≥ 0 ∧ (∀ x1 x2 : ℝ, x1 * x2 = k / 3 → x1 + x2 = 4 / 3 → x1 * x2 ≤ (2 / 3) ^ 2)) →
  k = 4 / 3 :=
by 
  sorry

end find_k_max_product_l347_347221


namespace n_equal_three_l347_347920

variable (m n : ℝ)

-- Conditions
def in_second_quadrant (m n : ℝ) : Prop := m < 0 ∧ n > 0
def distance_to_x_axis_eq_three (n : ℝ) : Prop := abs n = 3

-- Proof problem statement
theorem n_equal_three 
  (h1 : in_second_quadrant m n) 
  (h2 : distance_to_x_axis_eq_three n) : 
  n = 3 := 
sorry

end n_equal_three_l347_347920


namespace find_x_given_y_and_ratio_l347_347870

variable (x y k : ℝ)

theorem find_x_given_y_and_ratio :
  (∀ x y, (5 * x - 6) / (2 * y + 20) = k) →
  (5 * 3 - 6) / (2 * 5 + 20) = k →
  y = 15 →
  x = 21 / 5 :=
by 
  intro h1 h2 hy
  -- proof steps would go here
  sorry

end find_x_given_y_and_ratio_l347_347870


namespace triangles_from_decagon_l347_347097

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347097


namespace number_of_solutions_l347_347175

noncomputable def question : Type :=
  {z : ℂ // complex.abs z = 1 ∧ complex.abs ((z / (complex.conj z)) + (complex.conj z / z) + 1) = 2}

theorem number_of_solutions : ∃ (finite_count : ℕ), finite_count = 4 ∧ 
  ∀ (z : question), ∃ θ ∈ {θ : ℝ // θ = real.pi / 6 ∨ θ = 5 * real.pi / 6 ∨ θ = 7 * real.pi /6 ∨ θ = 11 * real.pi / 6}, z.val = complex.exp (complex.I * θ)
sorry

end number_of_solutions_l347_347175


namespace number_of_triangles_in_regular_decagon_l347_347125

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347125


namespace expr_equals_l347_347792

def expr : ℝ :=
  2 / (1 / (real.sqrt 2 + real.root 8 (4 : ℝ) + 2) + 1 / (real.sqrt 2 - real.root 8 (4 : ℝ) + 2))

theorem expr_equals : expr = 4 - real.sqrt 2 :=
by
  sorry

end expr_equals_l347_347792


namespace solve_quadratic_using_completing_square_l347_347983

theorem solve_quadratic_using_completing_square :
  ∀ x : ℝ, 2 * x^2 - 8 * x + 3 = 0 ↔ (x = 2 + sqrt 10 / 2 ∨ x = 2 - sqrt 10 / 2) :=
by
  intro x
  split
  intro h
  sorry -- proof omitted
  intro h
  cases h
  sorry -- proof omitted

end solve_quadratic_using_completing_square_l347_347983


namespace ways_to_distribute_balls_l347_347413

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347413


namespace ways_to_distribute_balls_l347_347453

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347453


namespace correct_imaginary_part_l347_347846

noncomputable def z := (-2 / 5) + ((1 / 5) * Complex.i)
def eq1 : z * (2 + Complex.i) = Complex.i ^ 10 := by 
  sorry

theorem correct_imaginary_part 
  (h : z * (2 + Complex.i) = Complex.i ^ 10) : z = (-2 / 5) + ((1 / 5) * Complex.i) :=
sorry

end correct_imaginary_part_l347_347846


namespace fixed_point_of_f_l347_347215

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 1

-- Stating the theorem
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = 0 := 
by
  -- This is a placeholder for the actual proof.
  sorry

end fixed_point_of_f_l347_347215


namespace ways_to_place_balls_in_boxes_l347_347491

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347491


namespace number_of_triangles_in_decagon_l347_347048

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347048


namespace unique_solution_to_equation_l347_347170

theorem unique_solution_to_equation (a : ℝ) (h : ∀ x : ℝ, a * x^2 + Real.sin x ^ 2 = a^2 - a) : a = 1 :=
sorry

end unique_solution_to_equation_l347_347170


namespace balls_into_boxes_l347_347406

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347406


namespace quadratic_function_vertex_l347_347617

theorem quadratic_function_vertex (a : ℝ) (h k : ℝ) (hx : h = 2) (hk : k = 1) (ha : a > 0) :
  ∃ (f : ℝ → ℝ), f = λ x, a * (x - h) ^ 2 + k :=
by {
  use λ x, a * (x - 2) ^ 2 + 1,
  simp [hx, hk],
  sorry
}

end quadratic_function_vertex_l347_347617


namespace find_c_d_l347_347830

theorem find_c_d (c d : ℚ)
  (h : (⟨3, c, -5⟩ : ℚ × ℚ × ℚ) ×₃ (⟨6, 9, d⟩) = (0, 0, 0)) :
  c = 9 / 2 ∧ d = -10 := 
sorry

end find_c_d_l347_347830


namespace ball_box_distribution_l347_347430

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347430


namespace decagon_triangle_count_l347_347059

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347059


namespace determine_gizmos_l347_347738

theorem determine_gizmos (g d : ℝ)
  (h1 : 80 * (g * 160 + d * 240) = 80)
  (h2 : 100 * (3 * g * 900 + 3 * d * 600) = 100)
  (h3 : 70 * (5 * g * n + 5 * d * 1050) = 70 * 5 * (g + d) ) :
  n = 70 := sorry

end determine_gizmos_l347_347738


namespace expressionA_is_negative_l347_347648

-- Define the variables and their approximate values
def A : ℝ := -4.2
def B : ℝ := -0.8
def C : ℝ := 1.2
def D : ℝ := 2.4
def E : ℝ := 2.6

-- Theorem statement checking that A - B is a negative number
theorem expressionA_is_negative : (A - B) < 0 := 
by {
  have hA : A = -4.2 := rfl,
  have hB : B = -0.8 := rfl,
  -- This is the proof part where you would show this inequality holds
  sorry
}

end expressionA_is_negative_l347_347648


namespace range_of_a_l347_347227

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1
noncomputable def g (x a : ℝ) : ℝ := 2^x - a

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (0 : ℝ) (2 : ℝ), ∃ x2 ∈ set.Icc (0 : ℝ) (2 : ℝ), |f x1 - g x2 a| ≤ 2) ↔ 2 ≤ a ∧ a ≤ 5 :=
sorry

end range_of_a_l347_347227


namespace num_triangles_in_decagon_l347_347071

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347071


namespace find_m_value_l347_347829

-- Defining the problem conditions and the final equality to be proven.
theorem find_m_value (α : ℝ) :
  (sin α + csc α + tan α)^2 + (cos α + sec α + cot α)^2 = m + 2 * tan α ^ 2 + 2 * cot α ^ 2 → m = 11 :=
by
  intro h
  sorry

end find_m_value_l347_347829


namespace num_triangles_from_decagon_l347_347013

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347013


namespace ball_in_boxes_l347_347266

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347266


namespace product_fraction_identity_l347_347628

theorem product_fraction_identity :
  (∏ k in finset.range (2022 - 1) + 2, (1 - (1 / (k : ℚ)^2))) = (2023 : ℚ) / 4044 :=
by sorry

end product_fraction_identity_l347_347628


namespace ball_in_boxes_l347_347269

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347269


namespace find_d_squared_l347_347740

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * Complex.I) * z

theorem find_d_squared (c d : ℝ) (z : ℂ) (h1 : ∀ z : ℂ, Complex.abs (g z c d - z) = 2 * Complex.abs (g z c d)) (h2 : Complex.abs (c + d * Complex.I) = 6) : d^2 = 11305 / 4 := 
sorry

end find_d_squared_l347_347740


namespace stewart_farm_horse_food_l347_347766

theorem stewart_farm_horse_food 
  (ratio : ℚ) (food_per_horse : ℤ) (num_sheep : ℤ) (num_horses : ℤ)
  (h1 : ratio = 5 / 7)
  (h2 : food_per_horse = 230)
  (h3 : num_sheep = 40)
  (h4 : ratio * num_horses = num_sheep) : 
  (num_horses * food_per_horse = 12880) := 
sorry

end stewart_farm_horse_food_l347_347766


namespace other_root_exists_l347_347612

-- Given the condition
def given_condition : Prop :=
  ∃ z : ℂ, z^2 = -91 + 84i ∧ z = 7 + 12i

-- The proof problem to state the solution
theorem other_root_exists : given_condition → (∃ w : ℂ, w^2 = - 91 + 84i ∧ w = -(7 + 12i)) := by
  sorry

end other_root_exists_l347_347612


namespace Ondra_problems_conditions_l347_347664

-- Define the conditions as provided
variables {a b : ℤ}

-- Define the first condition where the subtraction is equal to the product.
def condition1 : Prop := a + b = a * b

-- Define the second condition involving the relationship with 182.
def condition2 : Prop := a * b * (a + b) = 182

-- The statement to be proved: Ondra's problems (a, b) are (2, 2) and (1, 13) or (13, 1)
theorem Ondra_problems_conditions {a b : ℤ} (h1 : condition1) (h2 : condition2) :
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
sorry

end Ondra_problems_conditions_l347_347664


namespace angle_between_given_vectors_l347_347887

noncomputable def angle_between_vectors
  {α : Type*} [inner_product_space ℝ α] (a b : α) : ℝ :=
  real.arccos ((inner_product_space.inner a b) / (∥a∥ * ∥b∥))

theorem angle_between_given_vectors :
  ∀ (a b : ℝ × ℝ),
  a + 2 • b = (2, -4) ∧ 3 • a - b = (-8, 16) →
  angle_between_vectors a b = real.pi :=
by
  intros a b h
  sorry

end angle_between_given_vectors_l347_347887


namespace decagon_triangle_count_l347_347054

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347054


namespace infinite_geometric_series_sum_l347_347163

theorem infinite_geometric_series_sum : 
  (∃ (a r : ℚ), a = 5/4 ∧ r = 1/3) → 
  ∑' n : ℕ, ((5/4 : ℚ) * (1/3 : ℚ) ^ n) = (15/8 : ℚ) :=
by
  sorry

end infinite_geometric_series_sum_l347_347163


namespace miles_per_walk_l347_347161

-- Definitions for the given conditions
def total_days_in_march : ℕ := 31
def days_not_walked : ℕ := 4
def total_miles_walked : ℕ := 108

-- Statement to be proved
theorem miles_per_walk : 
  let days_walked := total_days_in_march - days_not_walked in
  total_miles_walked / days_walked = 4 := by
  sorry

end miles_per_walk_l347_347161


namespace angle_A_sum_b_n_l347_347524
noncomputable section

-- Define the problem conditions
def triangle (A B C a b c : ℝ) := 
  ∀ (A B C a b c : ℝ), 
  (b ≠ a) ∧ (c ≠ 0) → 
  (sqrt 3 * Real.sin B - Real.sin C) / (b - a) = (Real.sin A + Real.sin B) / c 

-- Prove A = π / 6 given the conditions above
theorem angle_A (A B C a b c : ℝ) (h : triangle A B C a b c) : 
  A = π / 6 :=
sorry

-- Definitions for the sequence problem conditions:
variable {n : ℕ} (a : ℕ → ℝ) (b : ℕ → ℝ)
def a_sequence (d : ℝ) (h1 : d ≠ 0) : Prop :=
  ∀ n : ℕ, a n = 2 * n ∧ (a 2, a 4, a 8 form geometric sequence) ∧ a 1 * Real.sin (π / 6) = 1

-- Defining b_n in terms of a_n
def b_sequence : Prop :=
  ∀ n : ℕ, b n = 1 / (a n * a (n + 1))

-- Prove the sum of the first n terms of b_n is the given value
theorem sum_b_n (S : ℕ → ℝ) (h_a : a_sequence d h1) (h_b : b_sequence)
  : S n = n / (4 * (n + 1)) :=
sorry

end angle_A_sum_b_n_l347_347524


namespace part_a_part_b_part_c_part_d_l347_347926

-- Part a
theorem part_a (n : ℕ) (numbers : Finset ℕ) (H1 : ∀ a ∈ numbers, a ≤ n) 
  (H2 : numbers.card = 4)
  (H3 : ∀ a b ∈ numbers, is_connected a b → Nat.coprime (a - b) n)
  (H4 : ∀ a b ∈ numbers, is_not_connected a b → ∃ d > 1, d ∣ (a - b) ∧ d ∣ n) :
  n = 4 := 
  sorry

-- Part b
theorem part_b (n : ℕ) (numbers : Finset ℕ) 
  (Hn : n = 49)
  (H1 : ∀ a ∈ numbers, a ≤ n) 
  (H2 : numbers.card = 5)
  (H3 : ∀ a b ∈ numbers, is_connected a b → Nat.coprime (a - b) n)
  (H4 : ∀ a b ∈ numbers, is_not_connected a b → ∃ d > 1, d ∣ (a - b) ∧ d ∣ n) :
  False := 
  sorry

-- Part c
theorem part_c (n : ℕ) (numbers : Finset ℕ) 
  (Hn : n = 33)
  (H1 : ∀ a ∈ numbers, a ≤ n) 
  (H2 : numbers.card = 5)
  (H3 : ∀ a b ∈ numbers, is_connected a b → Nat.coprime (a - b) n)
  (H4 : ∀ a b ∈ numbers, is_not_connected a b → ∃ d > 1, d ∣ (a - b) ∧ d ∣ n) :
  False := 
  sorry

-- Part d
theorem part_d (n : ℕ) (numbers : Finset ℕ) 
  (H1 : ∀ a ∈ numbers, a ≤ n) 
  (H2 : numbers.card = 5)
  (H3 : ∀ a b ∈ numbers, is_connected a b → Nat.coprime (a - b) n)
  (H4 : ∀ a b ∈ numbers, is_not_connected a b → ∃ d > 1, d ∣ (a - b) ∧ d ∣ n) :
  n = 105 := 
  sorry

end part_a_part_b_part_c_part_d_l347_347926


namespace distinguish_ball_box_ways_l347_347352

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347352


namespace num_ways_to_distribute_balls_into_boxes_l347_347250

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347250


namespace cannot_determine_right_triangle_l347_347756

theorem cannot_determine_right_triangle (A B C : Type) (angle_A angle_B angle_C : A) (a b c : B) 
  (h1 : angle_A = angle_B + angle_C)
  (h2 : a / b = 5 / 12 ∧ b / c = 12 / 13)
  (h3 : a ^ 2 = (b + c) * (b - c)):
  ¬ (angle_A / angle_B = 3 / 4 ∧ angle_B / angle_C = 4 / 5) :=
sorry

end cannot_determine_right_triangle_l347_347756


namespace number_of_triangles_in_decagon_l347_347088

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347088


namespace distinct_balls_boxes_l347_347471

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347471


namespace find_x_in_average_l347_347992

theorem find_x_in_average (x : ℝ) :
  (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + x) / 9 = 207 → x = 217 :=
by
  intro h
  sorry

end find_x_in_average_l347_347992


namespace floor_ceil_sum_l347_347796

open Real

theorem floor_ceil_sum : (⌊1.999⌋ : ℤ) + (⌈3.001⌉ : ℤ) = 5 := by
  sorry

end floor_ceil_sum_l347_347796


namespace ways_to_distribute_balls_l347_347415

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347415


namespace balls_into_boxes_l347_347328

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347328


namespace interval_of_monotonicity_range_on_interval_l347_347877

noncomputable def f (x : ℝ) := x * Real.exp x + 5

theorem interval_of_monotonicity : 
  (∀ x < -1, (f' x) < 0) ∧ (∀ x > -1, (f' x) > 0) :=
sorry

theorem range_on_interval : 
  ∃ f_min f_max, f_min = f 0 ∧ f_max = f 1 ∧ set.range (f '' set.Icc 0 1) = set.Icc 5 (Real.exp 1 + 5) :=
sorry

end interval_of_monotonicity_range_on_interval_l347_347877


namespace pieces_of_gum_per_cousin_l347_347936

theorem pieces_of_gum_per_cousin (total_gum : ℕ) (num_cousins : ℕ) (h1 : total_gum = 20) (h2 : num_cousins = 4) : total_gum / num_cousins = 5 := by
  sorry

end pieces_of_gum_per_cousin_l347_347936


namespace binomial_expansion_fourth_term_l347_347155

noncomputable def binomial_fourth_term (a x : ℝ) : ℝ :=
  ∑ k in finset.range 8, (binomial 8 k) * ((2*a / x^0.5)^(8-k)) * ((-x^0.5 / (2*a^2))^k)

theorem binomial_expansion_fourth_term (a x : ℝ) : binomial_fourth_term a x = -4 / (a * x) := 
by
  sorry

end binomial_expansion_fourth_term_l347_347155


namespace find_value_l347_347507

-- Given conditions of the problem
axiom condition : ∀ (a : ℝ), a - 1/a = 1

-- The mathematical proof problem
theorem find_value (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 :=
by
  sorry

end find_value_l347_347507


namespace ways_to_put_balls_in_boxes_l347_347479

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347479


namespace ways_to_put_balls_in_boxes_l347_347483

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347483


namespace smallest_n_for_7_terms_l347_347634

def largest_square_leq (m : ℕ) : ℕ := (√m).toNat ^ 2

def sequence_length (n : ℕ) : ℕ :=
  let rec tailSequence (current : ℕ) (len : ℕ) : ℕ  :=
    if current = 0 then len 
    else tailSequence (current - largest_square_leq current) (len + 1)
  tailSequence n 1

theorem smallest_n_for_7_terms : ∃ n : ℕ, sequence_length n = 7 ∧ ∀ m < n, sequence_length m ≠ 7 :=
  sorry

end smallest_n_for_7_terms_l347_347634


namespace balls_into_boxes_l347_347335

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347335


namespace angle_neg_4_in_second_quadrant_l347_347515

def quadrant (α : ℝ) : String :=
  if 0 ≤ α ∧ α < Real.pi / 2 then "First Quadrant"
  else if Real.pi / 2 ≤ α ∧ α < Real.pi then "Second Quadrant"
  else if Real.pi ≤ α ∧ α < 3 * Real.pi / 2 then "Third Quadrant"
  else if -Real.pi ≤ α ∧ α < -Real.pi / 2 then "Third Quadrant"
  else if -Real.pi / 2 ≤ α ∧ α < 0 then "Fourth Quadrant"
  else if 3 * Real.pi / 2 ≤ α ∧ α < 2 * Real.pi then "Fourth Quadrant"
  else if 2 * Real.pi ≤ α ∧ α < 5 * Real.pi / 2 then "First Quadrant"
  else if 5 * Real.pi / 2 ≤ α ∧ α < 3 * Real.pi then "Second Quadrant"
  else "Second Quadrant"

theorem angle_neg_4_in_second_quadrant : 
  quadrant (-4) = "Second Quadrant" := 
  by
  sorry

end angle_neg_4_in_second_quadrant_l347_347515


namespace num_triangles_from_decagon_l347_347003

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347003


namespace ways_to_place_balls_in_boxes_l347_347492

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347492


namespace g_decreasing_interval_l347_347876

def f (ω x : ℝ) : ℝ := sin (ω * x) - sqrt 3 * cos (ω * x)

def g (x : ℝ) : ℝ := 2 * sin (2 * x)

-- condition that ω = 2 based on the intersection distance
def omega_eq_two : Prop := ∀ ω > 0, (∃ d > 0, (∀ x, f ω (x + d) = f ω x) ∧ d = π / 2) → ω = 2

-- statement of the theorem proving the interval where g(x) is decreasing
theorem g_decreasing_interval : omega_eq_two →
  ∀ x, (π / 4 < x ∧ x < π / 3) → (g x) < g (x - 1) → sorry

end g_decreasing_interval_l347_347876


namespace hundreds_digit_of_factorial_difference_l347_347692

theorem hundreds_digit_of_factorial_difference :
  (let f := λ n : ℕ, n! in (f 17 - f 12) % 1000) / 100 % 10 = 4 :=
by
  have h1 : 12! % 1000 = 600 := by sorry
  have h2 : 17! % 1000 = 0 := by sorry
  calc
    (17! - 12!) % 1000 = (0 - 600) % 1000 : by rw [h1, h2]
                  ... = 400 : by norm_num
    400 / 100 % 10 = 4 : by norm_num

end hundreds_digit_of_factorial_difference_l347_347692


namespace number_of_triangles_in_regular_decagon_l347_347140

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347140


namespace problem_statement_l347_347207

-- Definitions based on the conditions in a)
def p : Prop := ∃ x ∈ ℝ, sin x = (Real.sqrt 5) / 2
def q : Prop := ∀ x ∈ Ioo 0 (Real.pi / 2), x > sin x

-- Statement to prove the correct judgment
theorem problem_statement : ¬q = false := 
by 
  -- (Proof would go here, but we add sorry to indicate it's not provided) 
  sorry

end problem_statement_l347_347207


namespace minimum_distance_curve_C_to_line_l_l347_347722

noncomputable def parametric_curve_C (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos α, Real.sin α)

def line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + (Real.pi / 4)) = (3 * sqrt 2) / 2

def distance_between_point_and_line (α : ℝ) : ℝ :=
  (abs ((sqrt 3 * Real.cos α) + (Real.sin α) - 3)) / sqrt 2

theorem minimum_distance_curve_C_to_line_l :
  ∃ α : ℝ, distance_between_point_and_line α = sqrt 2 / 2 :=
begin
  use Real.pi / 6,
  sorry,
end

end minimum_distance_curve_C_to_line_l_l347_347722


namespace distinct_balls_boxes_l347_347378

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347378


namespace production_days_l347_347837

theorem production_days (n : ℕ) (P : ℕ) (h1: P = n * 50) 
    (h2: (P + 110) / (n + 1) = 55) : n = 11 :=
by
  sorry

end production_days_l347_347837


namespace balls_into_boxes_l347_347336

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347336


namespace mn_eq_one_l347_347892

theorem mn_eq_one
  (M N : ℝ)
  (h1 : log M (N^3) = log N (M^2))
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = 1 :=
sorry

end mn_eq_one_l347_347892


namespace find_angle_A_l347_347908
open Real

theorem find_angle_A
  (a b : ℝ)
  (A B : ℝ)
  (h1 : b = 2 * a)
  (h2 : B = A + 60) :
  A = 30 :=
by 
  sorry

end find_angle_A_l347_347908


namespace ways_to_distribute_balls_l347_347443

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347443


namespace num_ways_to_distribute_balls_into_boxes_l347_347263

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347263


namespace number_of_triangles_in_decagon_l347_347082

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347082


namespace no_integer_solutions_l347_347950

theorem no_integer_solutions (P Q : Polynomial ℤ) (a : ℤ) (hP1 : P.eval a = 0) 
  (hP2 : P.eval (a + 1997) = 0) (hQ : Q.eval 1998 = 2000) : 
  ¬ ∃ x : ℤ, Q.eval (P.eval x) = 1 := 
by
  sorry

end no_integer_solutions_l347_347950


namespace simplify_expression_l347_347580

variables {a b c x : ℝ}
hypothesis h₁ : a ≠ b
hypothesis h₂ : b ≠ c
hypothesis h₃ : c ≠ a

def p (x : ℝ) : ℝ :=
  (x + a)^4 / ((a - b) * (a - c)) + 
  (x + b)^4 / ((b - a) * (b - c)) + 
  (x + c)^4 / ((c - a) * (c - b))

theorem simplify_expression : p x = a + b + c + 3 * x^2 :=
by sorry

end simplify_expression_l347_347580


namespace num_ways_to_distribute_balls_into_boxes_l347_347264

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347264


namespace num_triangles_from_decagon_l347_347004

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347004


namespace right_triangle_acute_angles_l347_347618

theorem right_triangle_acute_angles 
  (A B C D : Point)
  (hABC : is_right_triangle A B C)
  (hD_midpoint : midpoint D A B)
  (hCircle_tangent : inscribed_circle_tangent_midpoint A C D C D) :
  acute_angle A B C = 30 ∧ acute_angle B A C = 60 :=
by
  sorry

end right_triangle_acute_angles_l347_347618


namespace election_total_votes_l347_347918

theorem election_total_votes
  (V : ℕ)
  (h1 : 0.60 * V = w)
  (h2 : 0.40 * V = l)
  (h3 : w - l = 280) :
  V = 1400 :=
by 
  sorry

end election_total_votes_l347_347918


namespace decagon_triangle_count_l347_347062

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347062


namespace probability_even_and_gt_15_l347_347629

-- Define the numbering of the balls
def balls : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define the condition for the product of two numbers to be even and greater than 15
def even_and_gt_15 (a b : ℕ) : Prop := (a * b) % 2 = 0 ∧ (a * b) > 15

-- The main statement to be proved
theorem probability_even_and_gt_15 : 
  (∑ i in balls, ∑ j in balls, if even_and_gt_15 i j then 1 else 0) / 36 = 1 / 9 :=
by
  sorry -- The proof is omitted

end probability_even_and_gt_15_l347_347629


namespace ways_to_distribute_balls_l347_347377

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347377


namespace ways_to_distribute_balls_l347_347442

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347442


namespace number_of_people_l347_347771

def totalCups : ℕ := 10
def cupsPerPerson : ℕ := 2

theorem number_of_people {n : ℕ} (h : n = totalCups / cupsPerPerson) : n = 5 := by
  sorry

end number_of_people_l347_347771


namespace balls_into_boxes_l347_347395

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347395


namespace evaluate_floor_ceil_l347_347800

theorem evaluate_floor_ceil :
  (⌊1.999⌋ : ℤ) + (⌈3.001⌉ : ℤ) = 5 :=
by
  sorry

end evaluate_floor_ceil_l347_347800


namespace put_balls_in_boxes_l347_347312

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347312


namespace sara_added_onions_l347_347625

theorem sara_added_onions
  (initial_onions X : ℤ) 
  (h : initial_onions + X - 5 + 9 = initial_onions + 8) :
  X = 4 :=
by
  sorry

end sara_added_onions_l347_347625


namespace range_g_in_interval_l347_347224

def f (x : Real) := 2 * Real.sin (2 * x + (Real.pi / 6))
def g (x : Real) := 2 * Real.sin (4 * x + (Real.pi / 3))

theorem range_g_in_interval :
  ∀ x, 0 ≤ x ∧ x ≤ (5 * Real.pi / 24) → -1 ≤ g x ∧ g x ≤ 2 := by
  sorry

end range_g_in_interval_l347_347224


namespace put_balls_in_boxes_l347_347300

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347300


namespace cannot_determine_right_triangle_l347_347759

theorem cannot_determine_right_triangle (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = 180) →
  (\<^angle A = B + C) →
  (5 * a = 13 * c ∧ 12 * b = 13 * c) →
  (a^2 = (b+c) * (b-c)) →
  (\<^angle A = 3 * x ∧ \<^angle B = 4 * x ∧ \<^angle C = 5 * x) →
  (12 * x = 180) →
  (x ≠ 15 → ∃ (A B C : ℝ), A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90) :=
by sorry

end cannot_determine_right_triangle_l347_347759


namespace min_value_inequality_l347_347945

theorem min_value_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 / 2 :=
sorry

end min_value_inequality_l347_347945


namespace arrangement_of_SUCCESS_l347_347790

theorem arrangement_of_SUCCESS : 
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  (fact total_letters) / (fact count_S * fact count_C * fact count_U * fact count_E) = 420 := 
by
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  exact sorry

end arrangement_of_SUCCESS_l347_347790


namespace number_of_triangles_in_decagon_l347_347049

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347049


namespace number_of_polynomials_l347_347569

def is_valid_polynomial (Q : ℂ[X]) : Prop :=
  Q.coeff 0 = 75 ∧ (∀ a b : ℤ, (Q.has_root (a + b * complex.I) ∨ Q.has_root (a - b * complex.I)))

noncomputable def count_valid_polynomials : ℕ :=
  sorry

theorem number_of_polynomials (N : ℕ) : count_valid_polynomials = N :=
  sorry

end number_of_polynomials_l347_347569


namespace number_of_triangles_in_decagon_l347_347051

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347051


namespace remainder_division_l347_347570

noncomputable def Q : ℤ[X] := sorry

theorem remainder_division :
  ∃ (a b : ℚ), (∀ (x : ℚ), Q.eval x = (x - 10) * (x - 7) * (R x) + a * x + b) ∧ a = 10 / 3 ∧ b = -40 / 3
  :=
begin
  sorry
end

end remainder_division_l347_347570


namespace find_b2097_l347_347946

noncomputable def sequence (b : ℕ → ℝ) := ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

noncomputable def b1 := 3 + 2 * Real.sqrt 5
noncomputable def b2046 := 17 + 2 * Real.sqrt 5

theorem find_b2097 (b : ℕ → ℝ) (h : sequence b) (hb1 : b 1 = b1) (hb2046 : b 2046 = b2046) :
  b 2097 = -1 / (3 + 2 * Real.sqrt 5) + 6 * Real.sqrt 5 / (3 + 2 * Real.sqrt 5) :=
sorry

end find_b2097_l347_347946


namespace distinct_balls_boxes_l347_347392

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347392


namespace sin_ratio_l347_347187

-- Define the conditions
variables {a b c : ℝ} -- sides of the triangle
variables {A B C : ℝ} -- angles of the triangle

-- Assume the given condition
axiom given_condition : 3 * sin B * cos C = sin C * (1 - 3 * cos B)

-- Prove the ratio of sine of angles based on the given condition
theorem sin_ratio (h : a = b) (h : b = c) : sin C = 3 * sin A :=
by
  sorry


end sin_ratio_l347_347187


namespace balls_in_boxes_l347_347291

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347291


namespace complement_A_l347_347232

def U := Set ℝ

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}

theorem complement_A :
  (U \ A) = {x | x < -1 ∨ x ≥ 2} :=
sorry

end complement_A_l347_347232


namespace prob_three_even_dice_l347_347769

noncomputable def fairTwelveSidedDie := ⟨Set.range (λ n, n + 1), sorry⟩

def isEven (n : ℕ) : Prop := n % 2 = 0

theorem prob_three_even_dice (n : ℕ) (hn : n = 6) (m : ℕ) (hm : m = 12) :
  let p_even : ℝ := 1 / 2 in
  let probability :=
    (Nat.choose 6 3) * (p_even ^ 3) * ((1 - p_even) ^ 3) in
  probability = 5 / 16 :=
by
  have h : @Finset.card ℕ _ (Finset.filter isEven (Finset.range (m))) = m / 2 := sorry
  have p_even_def : p_even = 1 / 2 := rfl
  have hc : Nat.choose 6 3 = 20 := sorry
  have h_power : (p_even ^ 3) = (1 / 2) ^ 3 := by rw [p_even_def]
  have h_power2 : ((1 - p_even) ^ 3) = (1 / 2) ^ 3 := sorry
  let probability := 20 * (1 / 64)
  show probability = 5 / 16
  sorry

end prob_three_even_dice_l347_347769


namespace num_triangles_from_decagon_l347_347109

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347109


namespace put_balls_in_boxes_l347_347298

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347298


namespace balls_into_boxes_l347_347398

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347398


namespace distinct_balls_boxes_l347_347393

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347393


namespace regular_decagon_triangle_count_l347_347032

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347032


namespace balls_into_boxes_l347_347317

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347317


namespace y_at_40_l347_347655

def y_at_x (x : ℤ) : ℤ :=
  3 * x + 4

theorem y_at_40 : y_at_x 40 = 124 :=
by {
  sorry
}

end y_at_40_l347_347655


namespace find_a2015_l347_347928

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (a n - 1) / (a n + 1)

theorem find_a2015 (a : ℕ → ℚ) (h : sequence a) : a 2015 = -1/2 :=
sorry

end find_a2015_l347_347928


namespace ball_in_boxes_l347_347268

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347268


namespace number_of_possible_integral_values_for_BC_l347_347954

-- Definitions
def Triangle (A B C : Type) := -- Definition of a triangle with vertices A, B, C

-- Conditions
variables {A B C D E F : Type}
variable [Triangle A B C]
variable (AB AC BC AD : ℝ)
variable (BC : ℤ)

-- Preconditions
axiom AB_eq_7 : AB = 7
axiom angle_bisector : AD bisects ∠BAC
axiom parallel_AD_EF : parallel AD (line E F)
axiom equal_area : divides_triangle_into_equal_areas AD E F

-- The problem's statement
theorem number_of_possible_integral_values_for_BC : 
  7 < BC ∧ BC < 21 → ∃! (n : ℕ), n = 13 := 
by
  sorry

end number_of_possible_integral_values_for_BC_l347_347954


namespace xiaoGong_walking_speed_l347_347642

-- Defining the parameters for the problem
def distance : ℕ := 1200
def daChengExtraSpeedPerMinute : ℕ := 20
def timeUntilMeetingForDaCheng : ℕ := 12
def timeUntilMeetingForXiaoGong : ℕ := 6 + timeUntilMeetingForDaCheng

-- The main statement to prove Xiao Gong's speed
theorem xiaoGong_walking_speed : ∃ v : ℕ, 12 * (v + daChengExtraSpeedPerMinute) + 18 * v = distance ∧ v = 32 :=
by
  sorry

end xiaoGong_walking_speed_l347_347642


namespace ball_box_distribution_l347_347437

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347437


namespace distinguish_ball_box_ways_l347_347351

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347351


namespace sum_of_positive_factors_of_40_l347_347695

theorem sum_of_positive_factors_of_40 : 
  (∑ d in (finset.range 41).filter (λ n, 40 % n = 0), n) = 90 := 
by
  sorry

end sum_of_positive_factors_of_40_l347_347695


namespace ways_to_place_balls_in_boxes_l347_347505

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347505


namespace volumeCircumscribedSphereOfTrianglePyramid_l347_347198

noncomputable def volumeCircumscribedSphere (baseEdge : ℝ) (height: ℝ) : ℝ :=
  let r_inscribed := (1 / 3) * sqrt ((baseEdge ^ 2) - ((baseEdge / 2) ^ 2))
  let r_circumcircle := (2 / 3) * sqrt ((baseEdge ^ 2) - ((baseEdge / 2) ^ 2))
  let h := height / 2
  let r_sphere := sqrt (r_inscribed^2 + (h ^ 2))
  (4 / 3) * Real.pi * r_sphere ^ 3

theorem volumeCircumscribedSphereOfTrianglePyramid : 
  volumeCircumscribedSphere (2 * sqrt 3) 3 = (20 * sqrt 5 * Real.pi) / 3 :=
by 
  sorry

end volumeCircumscribedSphereOfTrianglePyramid_l347_347198


namespace balls_into_boxes_l347_347319

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347319


namespace sum_of_digits_T_is_36_l347_347747

-- Definition of a six-digit palindrome
structure Palindrome :=
  (a b c : Nat)
  (a_pos : a ≠ 0)
  (is_palindrome : true)

-- Sum of all palindromes
def sum_all_palindromes : Nat :=
  450 * 110022

-- Sum of digits
def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

open Nat

/-- The sum of the digits of T, where T is the sum of all six-digit palindromes, equals 36. -/
theorem sum_of_digits_T_is_36 : sum_of_digits sum_all_palindromes = 36 := by sorry

end sum_of_digits_T_is_36_l347_347747


namespace transformed_triangle_area_l347_347637

/-
Suppose the function g is defined on the domain {x_1, x_2, x_3} such that the graph of y = g(x) consists of just three points. 
These three points form a triangle of area 45. Prove that the area of the triangle formed by the points on the graph of y = 3g(3x)
is also 45.
-/

theorem transformed_triangle_area (g : ℝ → ℝ) (x1 x2 x3 : ℝ) 
  (h_domain : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)
  (h_area : ∃ t : ℝ, t = 45) :
  let y1 := g(x1), y2 := g(x2), y3 := g(x3),
      p1 := (x1, y1), p2 := (x2, y2), p3 := (x3, y3),
      new_p1 := (x1/3, 3*y1), new_p2 := (x2/3, 3*y2), new_p3 := (x3/3, 3*y3),
      original_area := 45 in
  triangle_area new_p1 new_p2 new_p3 = original_area :=
by
  sorry

end transformed_triangle_area_l347_347637


namespace simplify_expression_l347_347578

theorem simplify_expression (a b c x : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  ( ( (x + a)^4 ) / ( (a - b) * (a - c) ) 
  + ( (x + b)^4 ) / ( (b - a) * (b - c) ) 
  + ( (x + c)^4 ) / ( (c - a) * (c - b) ) ) = a + b + c + 4 * x := 
by
  sorry

end simplify_expression_l347_347578


namespace triangles_from_decagon_l347_347098

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347098


namespace number_of_triangles_in_regular_decagon_l347_347139

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347139


namespace crosses_win_in_six_moves_l347_347958

-- Definition of the game and its rules
structure Game :=
(grid : ℕ × ℕ)
(x_mark : (ℕ × ℕ) → Prop)
(o_mark : (ℕ × ℕ) → Prop)

-- Define initial conditions
def initial_position_valid (g : Game) (p : ℕ × ℕ) : Prop :=
p.1 ≥ 7 ∧ p.2 ≥ 7 ∧ p.1 < g.grid.1 - 7 ∧ p.2 < g.grid.2 - 7

def adjacent (p q : ℕ × ℕ) : Prop :=
(abs (p.1 - q.1) ≤ 1) ∧ (abs (p.2 - q.2) ≤ 1)

-- The statement to prove
theorem crosses_win_in_six_moves (g : Game) 
  (initial_position : ℕ × ℕ) 
  (h_valid : initial_position_valid g initial_position) 
  (n_moves : (ℕ × ℕ) → ℕ) :
  ∃ winning_strategy : (ℕ × ℕ) → (ℕ × ℕ),
    ∀ (x_pos : ℕ × ℕ), n_moves x_pos ≤ 6 → 
    winning_strategy x_pos ∨ g.x_mark x_pos → ∃ four_in_a_row : list (ℕ × ℕ),
    all_in_a_row four_in_a_row g.x_mark := 
sorry

end crosses_win_in_six_moves_l347_347958


namespace polynomial_value_not_less_than_factorial_divided_l347_347204

theorem polynomial_value_not_less_than_factorial_divided :
  ∀ (n : ℕ) (P : ℤ → ℤ)
  (x : ℕ → ℤ)
  (hmonotone : ∀ i j : ℕ, i < j → x i < x j),
  (∀ k : ℕ, k ≤ n → P (x k) = ∑ j in finset.range (n + 1), (P (x j) * ∏ i in (finset.range (n + 1)).erase j, (x k - x i) / (x j - x i))) ∧
  ∀ j : ℕ, j ≤ n →
  (∀ i : ℕ, i ≠ j →  x j ≠ x i) → 
  ∃ j : ℕ, j ≤ n ∧ |P (x j)| ≥ nat.factorial n / 2^n := sorry

end polynomial_value_not_less_than_factorial_divided_l347_347204


namespace rhombus_perimeter_l347_347999

-- Define the conditions
def is_rhombus (d1 d2 : ℝ) : Prop :=
  d1 > 0 ∧ d2 > 0 ∧ (d1 / 2)^2 + (d2 / 2)^2 = (√((d1 / 2)^2 + (d2 / 2)^2))^2

-- The theorem statement that we need to prove
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : is_rhombus d1 d2) (hd1 : d1 = 8) (hd2 : d2 = 30) : 
  4 * √((d1 / 2)^2 + (d2 / 2)^2) = 4 * √241 := 
by
  sorry

end rhombus_perimeter_l347_347999


namespace problem_statement_l347_347145

def x : ℝ := 102
def y : ℝ := 98

theorem problem_statement :
  ( (x^2 - y^2) / (x + y)^3 - (x^3 + y^3) * Real.log (x * y) ) ≈ -18_446_424.7199 := by
  sorry

end problem_statement_l347_347145


namespace total_horse_food_l347_347763

theorem total_horse_food (ratio_sh_to_h : ℕ → ℕ → Prop) 
    (sheep : ℕ) 
    (ounce_per_horse : ℕ) 
    (total_ounces_per_day : ℕ) : 
    ratio_sh_to_h 5 7 → sheep = 40 → ounce_per_horse = 230 → total_ounces_per_day = 12880 :=
by
  intros h_ratio h_sheep h_ounce
  sorry

end total_horse_food_l347_347763


namespace balls_in_boxes_l347_347290

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347290


namespace balls_in_boxes_l347_347283

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347283


namespace ways_to_distribute_balls_l347_347374

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347374


namespace balls_into_boxes_l347_347320

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347320


namespace vector_properties_l347_347239

variables {V : Type*} [innerProductSpace ℝ V]

-- Define unit vectors and their dot product condition
variables (a b : V) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hab : ⟪a, b⟫ = 1 / 2)

-- Theorem stating both correct conclusions
theorem vector_properties :
  ∥a + b∥ = sqrt 3 ∧ proj b a = (1 / 2 : ℝ) • b :=
begin
  sorry
end

end vector_properties_l347_347239


namespace num_ways_to_distribute_balls_into_boxes_l347_347251

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347251


namespace ball_box_distribution_l347_347429

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347429


namespace landA_area_and_ratio_l347_347714

/-
  a = 3, b = 5, c = 6
  p = 1/2 * (a + b + c)
  S = sqrt(p * (p - a) * (p - b) * (p - c))
  S_A = 2 * sqrt(14)
  S_B = 3/2 * sqrt(14)
  S_A / S_B = 4 / 3
-/
theorem landA_area_and_ratio :
  let a := 3
  let b := 5
  let c := 6
  let p := (a + b + c) / 2
  let S_A := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let S_B := 3 / 2 * Real.sqrt 14
  S_A = 2 * Real.sqrt 14 ∧ S_A / S_B = 4 / 3 :=
by
  sorry

end landA_area_and_ratio_l347_347714


namespace ways_to_distribute_balls_l347_347364

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347364


namespace ways_to_put_balls_in_boxes_l347_347480

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347480


namespace room_length_l347_347822

theorem room_length (b h d : ℕ) (hl : b = 8) (hh : h = 9) (hd : d = 17) : 
  ∃ l : ℕ, sqrt (l^2 + b^2 + h^2) = d ∧ l = 12 :=
by
  -- Conditions of the problem
  have h1 : b = 8 := hl
  have h2 : h = 9 := hh
  have h3 : d = 17 := hd

  -- Set the length of the room to 12
  let l : ℕ := 12

  -- Show that the length satisfies the given equation
  have hl_eq : sqrt (l^2 + b^2 + h^2) = d := by
    sorry

  -- Conclude the proof
  exact ⟨l, hl_eq, rfl⟩

end room_length_l347_347822


namespace parabola_focus_distance_l347_347857

theorem parabola_focus_distance (p : ℝ) (h_pos : p > 0) (A : ℝ × ℝ)
  (h_A_on_parabola : A.2 = 5 ∧ A.1^2 = 2 * p * A.2)
  (h_AF : abs (A.2 - (p / 2)) = 8) : p = 6 :=
by
  sorry

end parabola_focus_distance_l347_347857


namespace triangles_from_decagon_l347_347000

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l347_347000


namespace pure_imaginary_a_value_l347_347902

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define z and the condition that z is pure imaginary
def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_a_value (a : ℝ) :
  isPureImaginary ((2 - i) * (a - i)) → a = 1 / 2 :=
by
  simp [isPureImaginary]
  sorry

end pure_imaginary_a_value_l347_347902


namespace number_of_triangles_in_regular_decagon_l347_347138

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347138


namespace coefficient_fourth_term_expansion_l347_347781

theorem coefficient_fourth_term_expansion (x : ℝ) :
  let term_coefficient := (nat.choose 7 3) * (2^3)
  term_coefficient = 280 :=
by
  sorry

end coefficient_fourth_term_expansion_l347_347781


namespace find_y_l347_347527

variable (P y : ℝ)

def initial_price := 100

def price_increase (P : ℝ) (r : ℝ) : ℝ := P * (1 + r / 100)
def price_decrease (P : ℝ) (r : ℝ) : ℝ := P * (1 - r / 100)

theorem find_y :
  let P0 := initial_price in
  let P1 := price_increase P0 15 in
  let P2 := price_decrease P1 10 in
  let P3 := price_increase P2 30 in
  let P4 := price_decrease P3 (y : ℝ) in
  let P5 := price_increase P4 10 in
  P5 = P0 → y = 32 :=
sorry

end find_y_l347_347527


namespace circles_internally_tangent_l347_347791

def circle_center_radius (a b c : ℝ) :=
  let h := -a / 2
  let k := -b / 2
  let r := real.sqrt ((a / 2)^2 + (b / 2)^2 - c)
  (h, k, r)

noncomputable def C_1_center_radius := circle_center_radius 0 0 (-4)
noncomputable def C_2_center_radius := circle_center_radius 6 (-8) 24

theorem circles_internally_tangent :
  let (h1, k1, r1) := C_1_center_radius
  let (h2, k2, r2) := C_2_center_radius
  real.sqrt ((h1 - h2)^2 + (k1 - k2)^2) = abs (r2 - r1) :=
  sorry

end circles_internally_tangent_l347_347791


namespace minimum_boxes_to_eliminate_to_have_1_3_chance_l347_347535

open Set

-- Define the boxes and their amounts
def box_amounts := {0.05, 10, 15, 20, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 2000, 6000, 12000, 30000, 60000, 90000, 120000, 250000, 350000, 450000, 600000, 800000, 1200000}

-- Define the criterion for the amount being at least $250,000
def at_least_250000 (x : ℝ) := x ≥ 250000

-- Count the boxes with at least $250,000
def num_boxes_at_least_250000 := card (filter at_least_250000 box_amounts)

-- Define the total number of boxes
def total_boxes := 30

-- Define the total number of boxes needed to be at most for a 1/3 chance
def required_boxes := 3 * num_boxes_at_least_250000

-- Statement of the problem in Lean 4
theorem minimum_boxes_to_eliminate_to_have_1_3_chance :
  {x : ℝ | x ∈ box_amounts} → total_boxes - required_boxes = 12 :=
by
  sorry

end minimum_boxes_to_eliminate_to_have_1_3_chance_l347_347535


namespace exists_divisible_by_3_on_circle_l347_347606

theorem exists_divisible_by_3_on_circle :
  ∃ a : ℕ → ℕ, (∀ i, a i ≥ 1) ∧
               (∀ i, i < 99 → (a (i + 1) < 99 → (a (i + 1) - a i = 1 ∨ a (i + 1) - a i = 2 ∨ a (i + 1) = 2 * a i))) ∧
               (∃ i, i < 99 ∧ a i % 3 = 0) := 
sorry

end exists_divisible_by_3_on_circle_l347_347606


namespace num_triangles_from_decagon_l347_347107

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347107


namespace bananas_bought_l347_347768

def cost_per_banana : ℝ := 5.00
def total_cost : ℝ := 20.00

theorem bananas_bought : total_cost / cost_per_banana = 4 :=
by {
   sorry
}

end bananas_bought_l347_347768


namespace balls_in_boxes_l347_347287

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347287


namespace gabe_playlist_plays_l347_347841

def song_length_1 : ℕ := 3
def song_length_2 : ℕ := 2
def song_length_3 : ℕ := 3
def ride_time : ℕ := 40

theorem gabe_playlist_plays :
  let playlist_length := song_length_1 + song_length_2 + song_length_3 in
  ride_time / playlist_length = 5 := 
by 
  let playlist_length := song_length_1 + song_length_2 + song_length_3
  have h : playlist_length = 8 := by sorry
  have h2 : ride_time / playlist_length = 40 / 8 := by sorry
  have h3 : 40 / 8 = 5 := by sorry
  exact h2.trans h3

end gabe_playlist_plays_l347_347841


namespace EG_length_valid_l347_347919

noncomputable def quadrilateral_EFGH :=
  ∃ (EG : ℤ), (7 + 12 > EG) ∧ (EG + 7 > 12) ∧ (EG + 7 > 15) ∧ (15 + 7 > EG)

-- Proving that EG can indeed be 12 among the other valid possibilities in the given range
theorem EG_length_valid:
  quadrilateral_EFGH :=
begin
  sorry
end

end EG_length_valid_l347_347919


namespace ordinary_equation_curve_C_trajectory_midpoint_l347_347229

-- Define the parametric equation of curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (6 * Real.cos θ, 4 * Real.sin θ)

-- Define the coordinate transformation
def transform (x y : ℝ) : ℝ × ℝ := (x / 3, y / 4)

-- State that the transformed parametric equations yield the curve C'
def curve_C' (θ : ℝ) : Prop :=
  let (x', y') := transform 6 (4 * Real.sin θ)
  x' = 2 * Real.cos θ ∧ y' = Real.sin θ

-- Prove the ordinary equation of curve C'
theorem ordinary_equation_curve_C' : ∀ (θ : ℝ),
  let (x', y') := (2 * Real.cos θ, Real.sin θ)
  (x' ^ 2 / 4 + y' ^ 2 = 1)

-- Define midpoint condition for the trajectory
def midpoint_condition (x y : ℝ) : ℝ × ℝ := ((2 * x - 1), (2 * y - 3))

-- Prove the trajectory equation of midpoint P
theorem trajectory_midpoint (x y : ℝ) :
  let (x_0, y_0) := midpoint_condition x y
  (2 * x - 1) ^ 2 + 4 * (2 * y - 3) ^ 2 = 4


end ordinary_equation_curve_C_trajectory_midpoint_l347_347229


namespace distribute_thermometers_l347_347677

theorem distribute_thermometers : 
  ∃ (n : ℕ), n = 90 ∧ 
    ∃ (f: Fin 10 → ℕ), 
      (∀ c, 2 ≤ f c) ∧ 
      (∑ c, f c = 22) :=
sorry

end distribute_thermometers_l347_347677


namespace sequence_limit_l347_347564

noncomputable def seq (c : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then c 0
  else if n = 1 then c 1
  else sqrt (c (n - 1)) + sqrt (c (n - 2))

theorem sequence_limit (c : ℕ → ℝ) (h0 : c 0 > 0) (h1 : c 1 > 0) 
  (h_rec : ∀ n ≥ 1, c (n + 1) = sqrt (c n) + sqrt (c (n - 1))) :
  ∃ L : ℝ, L = 4 ∧ filter.tendsto c filter.at_top (nhds L) :=
begin
  -- Proof would go here
  sorry
end

end sequence_limit_l347_347564


namespace ways_to_distribute_balls_l347_347445

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347445


namespace nearest_integer_to_S_is_5_l347_347939

noncomputable def S : ℝ :=
  ∑ x in {x : ℝ | 4^x = x^4}, x

theorem nearest_integer_to_S_is_5 : abs (S - 5) ≤ 0.5 := 
sorry

end nearest_integer_to_S_is_5_l347_347939


namespace not_all_sticks_same_length_l347_347203

variable (a b c d : ℝ)

def can_form_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x 

def equal_area_triangles (a b c d : ℝ) : Prop :=
  let s₁ := (a + b + d) / 2
  let area_1 := Real.sqrt (s₁ * (s₁ - a) * (s₁ - b) * (s₁ - d))
  let s₂ := (a + c + d) / 2
  let area_2 := Real.sqrt (s₂ * (s₂ - a) * (s₂ - c) * (s₂ - d))
  let s₃ := (b + c + d) / 2
  let area_3 := Real.sqrt (s₃ * (s₃ - b) * (s₃ - c) * (s₃ - d))
  let s₄ := (a + b + c) / 2
  let area_4 := Real.sqrt (s₄ * (s₄ - a) * (s₄ - b) * (s₄ - c))
  area_1 = area_2 ∧ area_2 = area_3 ∧ area_3 = area_4

theorem not_all_sticks_same_length (a b c d : ℝ) 
  (h₁ : can_form_triangle a b d)
  (h₂ : can_form_triangle a c d)
  (h₃ : can_form_triangle b c d)
  (h₄ : can_form_triangle a b c)
  (h₅ : equal_area_triangles a b c d) :
  ¬(a = b ∧ b = c ∧ c = d) :=
begin
  -- proof omitted
  sorry
end

end not_all_sticks_same_length_l347_347203


namespace frequency_sum_eq_total_l347_347914

/--
In a frequency distribution table, each group represents a subset of the total data set,
and the frequency of a group indicates how many data points fall into that group.
Given these conditions, the sum of the frequencies of each group is equal to the total number of data points.
-/
theorem frequency_sum_eq_total (groups : Finset (Finset α)) (freq : Finset α → ℕ) (total : ℕ) :
  (∀ g ∈ groups, ∀ x ∈ g, x ∈ ⋃₀ groups) →
  (∀ x ∈ ⋃₀ groups, ∃ g ∈ groups, x ∈ g) →
  total = ∑ g in groups, freq g :=
sorry

end frequency_sum_eq_total_l347_347914


namespace distinct_balls_boxes_l347_347470

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347470


namespace ways_to_distribute_balls_l347_347421

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347421


namespace regular_decagon_triangle_count_l347_347029

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347029


namespace non_sophomores_is_75_percent_l347_347915

def students_not_sophomores_percentage (total_students : ℕ) 
                                       (percent_juniors : ℚ)
                                       (num_seniors : ℕ)
                                       (freshmen_more_than_sophomores : ℕ) : ℚ :=
  let num_juniors := total_students * percent_juniors 
  let s := (total_students - num_juniors - num_seniors - freshmen_more_than_sophomores) / 2
  let f := s + freshmen_more_than_sophomores
  let non_sophomores := total_students - s
  (non_sophomores / total_students) * 100

theorem non_sophomores_is_75_percent : students_not_sophomores_percentage 800 0.28 160 16 = 75 := by
  sorry

end non_sophomores_is_75_percent_l347_347915


namespace trig_identity_l347_347793

theorem trig_identity : 
  sin (315 * (Real.pi / 180)) - cos (135 * (Real.pi / 180)) + 2 * sin (570 * (Real.pi / 180)) = -1 := 
by
  sorry

end trig_identity_l347_347793


namespace josie_total_animals_is_correct_l347_347556

noncomputable def totalAnimals : Nat :=
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  let giraffes := antelopes + 15
  let lions := leopards + giraffes
  let elephants := 3 * lions
  antelopes + rabbits + hyenas + wild_dogs + leopards + giraffes + lions + elephants

theorem josie_total_animals_is_correct : totalAnimals = 1308 := by
  sorry

end josie_total_animals_is_correct_l347_347556


namespace decagon_triangle_count_l347_347063

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347063


namespace ways_to_distribute_balls_l347_347363

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347363


namespace no_real_polynomials_l347_347181

noncomputable def f (x : ℝ) (n : ℕ) : ℝ := x^n + ∑ i in finset.range n, x^i

theorem no_real_polynomials (n : ℕ) :
  ¬ (∃ g h : polynomial ℝ, polynomial.degree g > 1 ∧ polynomial.degree h > 1 ∧ ∀ x, f x n = polynomial.eval (polynomial.eval x h) g) :=
by
  sorry

end no_real_polynomials_l347_347181


namespace expression_equality_l347_347772

theorem expression_equality :
  (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := 
by
  sorry

end expression_equality_l347_347772


namespace polynomial_roots_l347_347177

theorem polynomial_roots :
  ∀ x, (3 * x^4 + 16 * x^3 - 36 * x^2 + 8 * x = 0) ↔ 
       (x = 0 ∨ x = 1 / 3 ∨ x = -3 + 2 * Real.sqrt 17 ∨ x = -3 - 2 * Real.sqrt 17) :=
by
  sorry

end polynomial_roots_l347_347177


namespace distinct_balls_boxes_l347_347389

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347389


namespace sum_distinct_pairs_product_is_124_l347_347598

theorem sum_distinct_pairs_product_is_124 :
  let pairs := {p : ℕ × ℕ | p.1 * p.2 = 48 ∧ p.1 ≠ p.2 ∧ p.1 > 0 ∧ p.2 > 0}
  let sums := {s | ∃ (p : ℕ × ℕ) (hp : p ∈ pairs), s = p.1 + p.2}
  ∑ s in sums, s = 124 := by
sorry

end sum_distinct_pairs_product_is_124_l347_347598


namespace find_ratio_KM_LM_l347_347562

variables {A B C D M K L : Point}
variables {AB AC BD DC KM KL : ℝ}
variables [MetricSpace Point]

-- Conditions
def angle_A_eq_60 : angle A B C = 60 := sorry
def length_AB_eq_12 : dist A B = 12 := sorry
def length_AC_eq_14 : dist A C = 14 := sorry
def D_on_BC : B = D ∨ C = D ∨ collinear A B C := sorry
def BAD_eq_CAD : ∠ B A D = ∠ C A D := sorry
def AD_meets_circumcircle_at_M : is_on_circumcircle (triangle A B C) M := sorry
def circumcircle_BDM_intersects_AB_K : B ≠ K ∧ is_on_circumcircle (triangle B D M) K := sorry
def KM_crosses_circumcircle_CDM_at_L : M ≠ L ∧ is_on_circumcircle (triangle C D M) L := sorry

-- Question and Answer
theorem find_ratio_KM_LM (h : D_on_BC ∧ BAD_eq_CAD ∧ AD_meets_circumcircle_at_M ∧ circumcircle_BDM_intersects_AB_K ∧ KM_crosses_circumcircle_CDM_at_L): 
  (dist K M) / (dist L M) = (Real.sqrt 3 + 1) / 2 := sorry

end find_ratio_KM_LM_l347_347562


namespace triangles_from_decagon_l347_347092

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347092


namespace cannot_determine_right_triangle_l347_347753

-- Definitions of conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (a b c : ℝ) : Prop := a/b = 5/12 ∧ b/c = 12/13
def condition_C (a b c : ℝ) : Prop := a^2 = (b + c) * (b - c)
def condition_D (A B C : ℝ) : Prop := A/B = 3/4 ∧ B/C = 4/5

-- The proof problem
theorem cannot_determine_right_triangle (a b c A B C : ℝ)
  (hD : condition_D A B C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end cannot_determine_right_triangle_l347_347753


namespace inscribed_and_circumscribed_l347_347744

-- Given a cyclic quadrilateral ABCD
variables {A B C D O A' B' C' D' : Type*} [AddGroup O]

-- Conditions: ABCD is cyclic and its diagonals AC and BD intersect perpendicularly at O
def cyclic_quad_and_perpendicular_diagonals := 
  ∃ (circle : Ideal.Circle),
    circle.is_cyclic_quadrilateral (A, B, C, D) ∧ 
    (∃ (O : Ideal.Point),
      are_perpendicular (diagonal A C O) (diagonal B D O)) ∧
    (O ∈ diagonal A C O ∧ O ∈ diagonal B D O)

-- Conditions: Projections from O to the sides of ABCD form a new quadrilateral A'B'C'D'
def new_quadrilateral_formed_by_projections :=
  foot_of_perpendicular (O) (A, B) = A' ∧
  foot_of_perpendicular (O) (B, C) = B' ∧
  foot_of_perpendicular (O) (C, D) = C' ∧
  foot_of_perpendicular (O) (D, A) = D'

-- Proof we need to establish: A'B'C'D' has an inscribed and circumscribed circle
theorem inscribed_and_circumscribed (h : cyclic_quad_and_perpendicular_diagonals) :
  (∃ (circle1 : Ideal.Circle), circle1.is_inscribed_quadrilateral (A', B', C', D')) ∧ 
  (∃ (circle2 : Ideal.Circle), circle2.is_circumscribed_quadrilateral (A', B', C', D')) :=
sorry

end inscribed_and_circumscribed_l347_347744


namespace balls_into_boxes_l347_347402

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347402


namespace number_of_triangles_in_decagon_l347_347042

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347042


namespace modulus_of_z_l347_347591

theorem modulus_of_z (z : ℂ) (h : (im * z) = abs (2 + im) + 2 * im) : abs z = 3 :=
sorry

end modulus_of_z_l347_347591


namespace angle_condition_l347_347751

theorem angle_condition (A B C : Type) [linear_ordered_ring A] [ordered_ring A] [add_comm_group A]
[ordered_semimodule A] 
-- Conditions
(h_angle_A : angle A < 120) 
(h_angle_B : angle B < 120) 
(h_angle_C : angle C < 120) : 
-- Conclusion
∃ P : A, all_sides_angle120 A B C P 
:=
sorry

end angle_condition_l347_347751


namespace domain_of_sqrt_l347_347643

noncomputable def domain (f : ℝ → ℝ) (D : set ℝ) : Prop :=
∀ x : ℝ, x ∈ D ↔ ∃ y : ℝ, f y = x

theorem domain_of_sqrt :
  domain (λ x, sqrt (1 - 2 * x)) {x : ℝ | x ≤ 1 / 2} :=
by {
  sorry
}

end domain_of_sqrt_l347_347643


namespace num_triangles_in_decagon_l347_347067

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347067


namespace modulus_of_z_l347_347636

theorem modulus_of_z (z : ℂ) (h : z^2 = -48 + 64 * complex.i) : complex.abs z = 4 * real.sqrt 5 := 
by sorry

end modulus_of_z_l347_347636


namespace balls_into_boxes_l347_347337

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347337


namespace distinct_balls_boxes_l347_347459

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347459


namespace number_of_arrangements_of_SUCCESS_l347_347785

-- Definitions based on conditions
def word : String := "SUCCESS"
def total_letters : Nat := 7
def repetitions : List Nat := [3, 2, 1, 1]  -- Corresponding to S, C, U, E

-- Lean statement proving the number of arrangements
theorem number_of_arrangements_of_SUCCESS : 
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1) * (Nat.factorial 1)) = 420 := by
  sorry

end number_of_arrangements_of_SUCCESS_l347_347785


namespace distinct_balls_boxes_l347_347385

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347385


namespace min_value_b_div_a_l347_347893

theorem min_value_b_div_a (a b : ℝ) (h : ∀ x > -1, ln (x + 1) - 1 ≤ a * x + b) : 
  ∃ (a : ℝ), a = real.exp(-1) ∧ a > 0 ∧ (b / a) ≥ 1 - real.exp(1) :=
begin
  sorry
end

end min_value_b_div_a_l347_347893


namespace number_of_triangles_in_decagon_l347_347022

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347022


namespace sum_of_roots_correct_l347_347630

noncomputable def sum_of_roots : ℝ :=
  let equation := ∀ x : ℝ, cos(x)^2 + cos(8 * x)^2 = 2 * cos(x)^2 * cos(8 * x)^2
  let interval := [3 * real.pi, 6 * real.pi]
  let sum_roots := ∑ x in interval, if equation x then x else 0
  56.55

theorem sum_of_roots_correct : sum_of_roots = 56.55 := by
  sorry

end sum_of_roots_correct_l347_347630


namespace jessica_and_sibling_age_l347_347699

theorem jessica_and_sibling_age
  (J M S : ℕ)
  (h1 : J = M / 2)
  (h2 : M + 10 = 70)
  (h3 : S = J + ((70 - M) / 2)) :
  J = 40 ∧ S = 45 :=
by
  sorry

end jessica_and_sibling_age_l347_347699


namespace part_I_part_II_l347_347875

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 6) + cos (x - π / 3)
noncomputable def g (x : ℝ) : ℝ := 2 * sin (x / 2) ^ 2

theorem part_I (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : f α = 3 * (sqrt 3) / 5) : 
  g α = 1 / 5 := 
sorry

theorem part_II (x : ℝ) : 
  (f x ≥ g x) ↔ ∃ k : ℤ, 2 * k * π ≤ x ∧ x ≤ 2 * k * π + (2 * π / 3) := 
sorry

end part_I_part_II_l347_347875


namespace put_balls_in_boxes_l347_347302

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347302


namespace balls_into_boxes_l347_347330

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347330


namespace ball_box_distribution_l347_347431

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347431


namespace sequence_formula_inequality_proof_l347_347851

noncomputable def a_seq : ℕ → ℝ
| 0       := 0 -- not used, dummy value since sequence starts at n=1
| (n + 1) := 2 - (1 / 2^(n + 1))

noncomputable def sum_seq : ℕ → ℝ
| 0       := 0 -- not used, dummy value since sum starts at n=1
| (n + 1) := (n + 1) * (2 - (1 / 2^(n)))

theorem sequence_formula (n : ℕ) (h : n > 0):
  a_seq n = 2 - (1 / 2^n) :=
sorry

theorem inequality_proof :
  ∀ (n : ℕ) (h : n > 0),
  (∑ k in finset.range n, 1 / (2^(k + 1) * a_seq (k + 1) * a_seq (k + 2))) < 1 / 3 :=
sorry

end sequence_formula_inequality_proof_l347_347851


namespace fraction_zero_solution_l347_347522

theorem fraction_zero_solution (x : ℝ) (h : (x - 1) / (2 - x) = 0) : x = 1 :=
sorry

end fraction_zero_solution_l347_347522


namespace ways_to_place_balls_in_boxes_l347_347497

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347497


namespace ways_to_distribute_balls_l347_347365

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347365


namespace area_enclosed_by_circles_l347_347144

theorem area_enclosed_by_circles :
  let α := real.pi
  let sqrt3 := real.sqrt 3
  let r1 := 9 - 3 * sqrt3
  let r2 := 3 * sqrt3 - 3
  let r3 := 3 * sqrt3 + 3
  let area_triangle := 18 * sqrt3
  let area_a := (1/4) * α * (r1 ^ 2)
  let area_b := (1/6) * α * (r2 ^ 2)
  let area_c := (1/12) * α * (r3 ^ 2)
  let total_area_circles := area_a + area_b + area_c
  (area_triangle - total_area_circles) = 9 * sqrt3 * (2 + α) - 36 * α :=
sorry

end area_enclosed_by_circles_l347_347144


namespace neil_halloween_candy_l347_347245

-- Definitions based on the conditions
def maggie_collected : ℕ := 50
def percentage_increase_harper : ℝ := 0.30
def percentage_increase_neil : ℝ := 0.40

-- Define the extra candy Harper collected
def extra_candy_harper (m : ℕ) (p : ℝ) : ℕ := (p * m).nat_abs

-- Define the total candy Harper collected
def harper_collected (m : ℕ) (p : ℝ) : ℕ := m + extra_candy_harper m p

-- Define the extra candy Neil collected
def extra_candy_neil (h : ℕ) (p : ℝ) : ℕ := (p * h).nat_abs

-- Define the total candy Neil collected
def neil_collected (h : ℕ) (p : ℝ) : ℕ := h + extra_candy_neil h p

-- Problem statement
theorem neil_halloween_candy : neil_collected (harper_collected maggie_collected percentage_increase_harper) percentage_increase_neil = 91 :=
by
  sorry

end neil_halloween_candy_l347_347245


namespace tan_theta_value_l347_347219

noncomputable def theta : ℝ := sorry -- the angle θ, specifics not used for definition
axiom theta_in_second_quadrant : ∃ (θ : ℝ), 0 < θ ∧ θ < π -- θ is in the second quadrant
axiom sin_theta : Real.sin theta = sqrt 3 / 2 -- sin θ = √3 / 2

theorem tan_theta_value : Real.tan theta = -sqrt 3 := by
  sorry -- proof to be filled in...

end tan_theta_value_l347_347219


namespace smallest_n_l347_347624

theorem smallest_n {n : ℕ} (h1 : n ≡ 4 [MOD 6]) (h2 : n ≡ 3 [MOD 7]) (h3 : n > 10) : n = 52 :=
sorry

end smallest_n_l347_347624


namespace joseph_total_payment_l347_347935
-- Importing necessary libraries

-- Defining the variables and conditions
variables (W : ℝ) -- The cost for the water heater

-- Conditions
def condition1 := 3 * W -- The cost for the refrigerator
def condition2 := 2 * W = 500 -- The electric oven
def condition3 := 300 -- The cost for the air conditioner
def condition4 := 100 -- The cost for the washing machine

-- Calculate total cost
def total_cost := (3 * W) + W + 500 + 300 + 100

-- The theorem stating the total amount Joseph pays
theorem joseph_total_payment : total_cost = 1900 :=
by 
  have hW := condition2;
  sorry

end joseph_total_payment_l347_347935


namespace number_of_triangles_in_regular_decagon_l347_347127

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347127


namespace eval_expression_l347_347804

-- Definitions for the floor and ceiling functions
def floor (x : ℝ) : ℤ := Int.floor x
def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Problem statement to prove
theorem eval_expression : floor 1.999 + ceiling 3.001 = 5 :=
by
  sorry

end eval_expression_l347_347804


namespace put_balls_in_boxes_l347_347311

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347311


namespace boat_travel_distance_l347_347554

variable (v c d : ℝ) (c_eq_1 : c = 1)

theorem boat_travel_distance : 
  (∀ (v : ℝ), d = (v + c) * 4 → d = (v - c) * 6) → d = 24 := 
by
  intro H
  sorry

end boat_travel_distance_l347_347554


namespace first_player_wins_game_l347_347682

/-- Theorem: In a game with two piles of tokens and specific types of moves, 
the first player has a winning strategy.
-/
theorem first_player_wins_game :
  let initial_pile1 := 10000
  let initial_pile2 := 20000
  (∀ (x y : ℕ), (0 < x ∨ 0 < y) → ((0 < x ∧ 0 < y) → (x + y) % 2015 = 0) →
  ∀ (pile1 pile2 : ℕ), (pile1 = initial_pile1 ∧ pile2 = initial_pile2) →
  ∀ (move1 move2 : ℕ), ((pile1 - move1 = 0 ∧ pile2 - move2 ≥ 0) ∨
                        (pile1 - move1 ≥ 0 ∧ pile2 - move2 = 0) ∨
                        (pile1 - move1 > 0 ∧ pile2 - move2 > 0 ∧ (move1 + move2) % 2015 = 0)) →
  ∃ (winning_strategy : ∀ (pile1 pile2 : ℕ), pile1 ≠ 0 ∨ pile2 ≠ 0 →
                      ∃ (move1 move2 : ℕ), ((pile1 - move1 = 0 ∧ pile2 - move2 ≥ 0) ∨
                                           (pile1 - move1 ≥ 0 ∧ pile2 - move2 = 0) ∨
                                           (pile1 - move1 > 0 ∧ pile2 - move2 > 0 ∧ (move1 + move2) % 2015 = 0)) ∧
                      (∀ (next_pile1 next_pile2 : ℕ),
                        next_pile1 = pile1 - move1 →
                        next_pile2 = pile2 - move2 →
                        next_pile1 = next_pile2 ∨ ∃ (n : ℕ), n % 2015 = 0 ∧ next_pile1 + next_pile2 = 2 * n))) :=
sorry

end first_player_wins_game_l347_347682


namespace num_ways_to_distribute_balls_into_boxes_l347_347255

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347255


namespace balls_into_boxes_l347_347316

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347316


namespace ways_to_distribute_balls_l347_347372

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347372


namespace decagon_triangle_count_l347_347060

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347060


namespace simplify_expression_l347_347627

theorem simplify_expression (b c : ℝ) : 
  (2 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4 * 7 * c^2 = 5040 * b^10 * c^2) :=
by sorry

end simplify_expression_l347_347627


namespace balls_into_boxes_l347_347399

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347399


namespace num_triangles_in_decagon_l347_347069

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347069


namespace number_of_triangles_in_decagon_l347_347016

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347016


namespace problem_solution_l347_347848

-- Define the function f and the conditions
variable (f : ℝ → ℝ)
variable (D : ∀ x ∈ Ioo 0 (Real.pi / 2), HasDerivAt f (f' x) x)
variable (f' : ℝ → ℝ)
variable h_cond : ∀ x ∈ Ioo 0 (Real.pi / 2), f x < (f' x) * Real.tan x

-- The goal is to prove the statement
theorem problem_solution :
  sqrt 3 * f (Real.pi / 6) < f (Real.pi / 3) :=
sorry

end problem_solution_l347_347848


namespace rook_visit_twice_l347_347605

def same_color (pos1 pos2 : (ℕ × ℕ)) : Prop :=
  ((pos1.1 + pos1.2) % 2 = (pos2.1 + pos2.2) % 2)

theorem rook_visit_twice (pos1 pos2 : (ℕ × ℕ)) (h_color : same_color pos1 pos2) :
  ∃ path : list (ℕ × ℕ), path.head = pos1 ∧ path.last = some pos2 ∧ 
  (∀ sq : (ℕ × ℕ), sq ∈ path) ∧ (∀ sq : (ℕ × ℕ), sq = pos2 → count sq path = 2) ∧
  (∀ sq : (ℕ × ℕ), sq ≠ pos2 → count sq path = 1) :=
sorry

def count {α : Type*} (a : α) : list α → ℕ
| [] := 0
| (h :: t) := (if a = h then 1 else 0) + (count t)

end rook_visit_twice_l347_347605


namespace eval_expression_l347_347803

-- Definitions for the floor and ceiling functions
def floor (x : ℝ) : ℤ := Int.floor x
def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Problem statement to prove
theorem eval_expression : floor 1.999 + ceiling 3.001 = 5 :=
by
  sorry

end eval_expression_l347_347803


namespace evaluate_sum_of_powers_of_i_l347_347806

-- Definition of the imaginary unit i with property i^2 = -1.
def i : ℂ := Complex.I

lemma i_pow_2 : i^2 = -1 := by
  sorry

lemma i_pow_4n (n : ℤ) : i^(4 * n) = 1 := by
  sorry

-- Problem statement: Evaluate i^13 + i^18 + i^23 + i^28 + i^33 + i^38.
theorem evaluate_sum_of_powers_of_i : 
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by
  sorry

end evaluate_sum_of_powers_of_i_l347_347806


namespace balls_in_boxes_l347_347292

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347292


namespace balls_into_boxes_l347_347318

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347318


namespace trapezoid_inequality_l347_347609

noncomputable def isosceles_trapezoid (A B C D : Point)
  (AD_is_base : AD > BC)
  (isosceles: Triangle.isIsosceles ABCD)
  (DK_eq_CD : DK = CD)
  (AL_parallel_CK : Line.parallel AL CK) :
  Prop :=
  (AL + CL ≤ AD + BL)

theorem trapezoid_inequality (A B C D K L : Point)
  (AD_is_base : AD > BC)
  (is_isosceles_trapezoid : isosceles_trapezoid A B C D AD_is_base DK_eq_CD AL_parallel_CK)
  (DK_eq_CD : DK = CD)
  (AL_parallel_CK : Line.parallel AL CK) :
  AL + CL ≤ AD + BL := sorry

end trapezoid_inequality_l347_347609


namespace neil_halloween_candy_l347_347246

-- Definitions based on the conditions
def maggie_collected : ℕ := 50
def percentage_increase_harper : ℝ := 0.30
def percentage_increase_neil : ℝ := 0.40

-- Define the extra candy Harper collected
def extra_candy_harper (m : ℕ) (p : ℝ) : ℕ := (p * m).nat_abs

-- Define the total candy Harper collected
def harper_collected (m : ℕ) (p : ℝ) : ℕ := m + extra_candy_harper m p

-- Define the extra candy Neil collected
def extra_candy_neil (h : ℕ) (p : ℝ) : ℕ := (p * h).nat_abs

-- Define the total candy Neil collected
def neil_collected (h : ℕ) (p : ℝ) : ℕ := h + extra_candy_neil h p

-- Problem statement
theorem neil_halloween_candy : neil_collected (harper_collected maggie_collected percentage_increase_harper) percentage_increase_neil = 91 :=
by
  sorry

end neil_halloween_candy_l347_347246


namespace eccentricity_of_ellipse_l347_347200

theorem eccentricity_of_ellipse 
  (a b : ℝ) (e : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), x = 0 ∧ y > 0 ∧ (9 * b^2 = 16/7 * a^2)) :
  e = Real.sqrt (10) / 6 :=
sorry

end eccentricity_of_ellipse_l347_347200


namespace integral_x2_sub_x_l347_347164

noncomputable def F (x : ℝ) : ℝ := x^3 / 3 - x^2 / 2

theorem integral_x2_sub_x : ∫ x in 0..2, (x^2 - x) = 2 / 3 :=
by
  -- Apply the Fundamental Theorem of Calculus
  have h_derivative : ∀ x : ℝ, deriv F x = x^2 - x := by
    intros x
    simp [F]
    -- Need to show deriv (x^3/3 - x^2/2) = x^2 - x
    calc
      deriv (x^3 / 3 - x^2 / 2) = deriv (x^3 / 3) - deriv (x^2 / 2) : by apply deriv_sub
                         ... = 3 * x^2 / 3 - 2 * x / 2               : by simp [deriv]
                         ... = x^2 - x                             : by ring
  -- Evaluate the integral using the antiderivative F and the bounds 0 and 2
  have h_integral : ∫ x in 0..2, (x^2 - x) = F 2 - F 0 := by apply integral_deriv_of_le, exact h_derivative, linarith
  -- Calculate F 2 and F 0 and show the result is 2 / 3
  have : F 2 = 2^3 / 3 - 2^2 / 2 := rfl
  rw [this]
  have : F 2 = 8 / 3 - 2 := rfl
  have : F 0 = 0 := rfl
  rw [this, sub_zero, sub_eq_of_eq_add, add_zero, eq_div_iff],
  congr; ring

end integral_x2_sub_x_l347_347164


namespace smaller_hexagon_area_fraction_l347_347745

theorem smaller_hexagon_area_fraction (a : ℝ) :
  let area_ratio := (3 * real.sqrt 3 / 8 * a^2) / (3 * real.sqrt 3 / 2 * a^2)
  in area_ratio = 3 / 4 := 
by 
  sorry 

end smaller_hexagon_area_fraction_l347_347745


namespace induction_inequality_term_added_l347_347702

theorem induction_inequality_term_added (k : ℕ) (h : k > 0) :
  let termAdded := (1 / (2 * (k + 1) - 1 : ℝ)) + (1 / (2 * (k + 1) : ℝ)) - (1 / (k + 1 : ℝ))
  ∃ h : ℝ, termAdded = h :=
by
  sorry

end induction_inequality_term_added_l347_347702


namespace distinct_balls_boxes_l347_347379

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347379


namespace balls_into_boxes_l347_347409

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347409


namespace actual_distance_traveled_l347_347900

theorem actual_distance_traveled :
  ∀ (t : ℝ) (d1 d2 : ℝ),
  d1 = 15 * t →
  d2 = 30 * t →
  d2 = d1 + 45 →
  d1 = 45 := by
  intro t d1 d2 h1 h2 h3
  sorry

end actual_distance_traveled_l347_347900


namespace ball_in_boxes_l347_347280

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347280


namespace distinguish_ball_box_ways_l347_347358

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347358


namespace balls_into_boxes_l347_347401

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347401


namespace math_equivalence_proof_l347_347209

noncomputable def problem_statement : Prop :=
  ∃ (t : ℝ) (r p q : ℕ), 0 < r ∧ 0 < p ∧ 0 < q ∧ Nat.coprime p q ∧
  ((1 + Real.sin t) * (1 + Real.cos t) = 9/4) ∧
  ((1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r) ∧
  (r + p + q = 46)

theorem math_equivalence_proof : problem_statement :=
  sorry

end math_equivalence_proof_l347_347209


namespace ball_box_distribution_l347_347426

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347426


namespace number_of_proper_subsets_of_A_l347_347235

universe u

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 3}
def complement_U_A : Set ℕ := { x ∈ U | x ∉ A } := {2}

theorem number_of_proper_subsets_of_A : ∀ n, n ∈ A ↔ ¬ n ∈ complement_U_A → P where
  P := (∅ ⊆ A ∧ ∅ ≠ A) ∧
       ({1} ⊆ A ∧ {1} ≠ A) ∧
       ({3} ⊆ A ∧ {3} ≠ A) ∧
       (∀ B, B ⊆ A → B ≠ A → B = ∅ ∨ B = {1} ∨ B = {3})

#check number_of_proper_subsets_of_A

end number_of_proper_subsets_of_A_l347_347235


namespace complex_magnitude_add_reciprocals_l347_347590

open Complex

theorem complex_magnitude_add_reciprocals
  (z w : ℂ)
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hz_plus_w : Complex.abs (z + w) = 6) :
  Complex.abs (1 / z + 1 / w) = 3 / 4 := by
  sorry

end complex_magnitude_add_reciprocals_l347_347590


namespace cannot_determine_right_triangle_l347_347757

theorem cannot_determine_right_triangle (A B C : Type) (angle_A angle_B angle_C : A) (a b c : B) 
  (h1 : angle_A = angle_B + angle_C)
  (h2 : a / b = 5 / 12 ∧ b / c = 12 / 13)
  (h3 : a ^ 2 = (b + c) * (b - c)):
  ¬ (angle_A / angle_B = 3 / 4 ∧ angle_B / angle_C = 4 / 5) :=
sorry

end cannot_determine_right_triangle_l347_347757


namespace balls_in_boxes_l347_347288

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347288


namespace ball_box_distribution_l347_347428

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347428


namespace uniformity_comparison_l347_347623

theorem uniformity_comparison (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : S1^2 < S2^2 :=
by {
  sorry
}

end uniformity_comparison_l347_347623


namespace ball_box_distribution_l347_347438

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347438


namespace number_of_triangles_in_decagon_l347_347050

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347050


namespace correct_statements_l347_347836

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.tan (x + Real.pi / 6)

theorem correct_statements :
  (¬ (∀ x, f x = f (-x - Real.pi / 12))) ∧
  (∃ c, ∀ x, g (c - x) = -g x) ∧
  (¬ (∀ x ∈ set.Ioo (-(Real.pi/6)), x, g x ≥ 0 / g x < 0)) ∧
  (∀ x, f (x - Real.pi / 6) = 3 * Real.cos (2 * x)) ∧
  (∀ x1 x2, f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * (Real.pi / 2)) := sorry

end correct_statements_l347_347836


namespace ways_to_distribute_balls_l347_347452

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347452


namespace ball_box_distribution_l347_347435

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347435


namespace ways_to_put_balls_in_boxes_l347_347474

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347474


namespace distinguish_ball_box_ways_l347_347349

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347349


namespace two_x_plus_y_equals_7_l347_347898

noncomputable def proof_problem (x y A : ℝ) : ℝ :=
  if (2 * x + y = A ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) then A else 0

theorem two_x_plus_y_equals_7 (x y : ℝ) : 
  (2 * x + y = proof_problem x y 7) ↔
  (2 * x + y = 7 ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) :=
by sorry

end two_x_plus_y_equals_7_l347_347898


namespace ball_in_boxes_l347_347281

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347281


namespace minimum_a_l347_347188

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 1)
noncomputable def g (x a : ℝ) : ℝ := f x - a

theorem minimum_a (a : ℝ) : (∃ x : ℝ, g x a = 0) ↔ (a ≥ 1) :=
by sorry

end minimum_a_l347_347188


namespace min_function_value_l347_347212

theorem min_function_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  (1/3 * x^3 + y^2 + z) = 13/12 :=
sorry

end min_function_value_l347_347212


namespace num_triangles_in_decagon_l347_347075

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347075


namespace ways_to_distribute_balls_l347_347373

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347373


namespace simplify_expression_l347_347577

theorem simplify_expression (a b c x : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  ( ( (x + a)^4 ) / ( (a - b) * (a - c) ) 
  + ( (x + b)^4 ) / ( (b - a) * (b - c) ) 
  + ( (x + c)^4 ) / ( (c - a) * (c - b) ) ) = a + b + c + 4 * x := 
by
  sorry

end simplify_expression_l347_347577


namespace ball_box_distribution_l347_347440

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347440


namespace num_ways_to_distribute_balls_into_boxes_l347_347253

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347253


namespace sarah_speed_for_rest_of_trip_l347_347626

def initial_speed : ℝ := 15  -- miles per hour
def initial_time : ℝ := 1  -- hour
def total_distance : ℝ := 45  -- miles
def extra_time_if_same_speed : ℝ := 1  -- hour (late)
def arrival_early_time : ℝ := 0.5  -- hour (early)

theorem sarah_speed_for_rest_of_trip (remaining_distance remaining_time : ℝ) :
  remaining_distance = total_distance - initial_speed * initial_time →
  remaining_time = (remaining_distance / initial_speed - extra_time_if_same_speed) + arrival_early_time →
  remaining_distance / remaining_time = 20 :=
by
  intros h1 h2
  sorry

end sarah_speed_for_rest_of_trip_l347_347626


namespace find_area_of_S_l347_347585

structure Triangle (a b c : ℝ) :=
  (side_positivity: a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality: a + b > c ∧ a + c > b ∧ b + c > a)

def heron_area (a b c s: ℝ) : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def inradius (area s : ℝ) : ℝ := area / s

noncomputable def similar_area (area ratio: ℝ) : ℝ := area * ratio^2

theorem find_area_of_S :
  let T := Triangle.mk 26 51 73 (by linarith) (by linarith) in
  let s := (26 + 51 + 73) / 2 in
  let area_T := heron_area 26 51 73 s in
  let r := inradius area_T s in
  let r' := r - 5 in
  let ratio := r' / r in
  let area_S := similar_area area_T ratio in
  area_S = 135 / 28 :=
by
  sorry

end find_area_of_S_l347_347585


namespace ball_in_boxes_l347_347278

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347278


namespace number_of_triangles_in_decagon_l347_347047

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347047


namespace sum_of_coefficients_expansion_l347_347601

theorem sum_of_coefficients_expansion (x : ℝ) :
  let p := x^2 + x + 1 in
  (p^0 = 1) →
  (p^1 = x^2 + x + 1) →
  (p^2 = x^4 + 2 * x^3 + 3 * x^2 + 2 * x + 1) →
  (p^3 = x^6 + 3 * x^5 + 6 * x^4 + 7 * x^3 + 6 * x^2 + 3 * x + 1) →
  (coeff (expand (p^5)) 6 + coeff (expand (p^5)) 5 + coeff (expand (p^5)) 4 = 141)
:= by
  sorry

end sum_of_coefficients_expansion_l347_347601


namespace num_triangles_in_decagon_l347_347066

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347066


namespace ball_box_distribution_l347_347433

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347433


namespace sum_of_roots_eq_l347_347633

theorem sum_of_roots_eq :
  let π := Real.pi in
  ∑ k in (Finset.range 22).filter (λ k, 11 ≤ k ∧ k ≤ 21), (2 * k * π / 7) = 18 * π := 
by 
  sorry

end sum_of_roots_eq_l347_347633


namespace balls_into_boxes_l347_347396

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347396


namespace derivative_log_base_3_at_3_l347_347508

noncomputable def f (x : ℝ) : ℝ := real.log x / real.log 3

theorem derivative_log_base_3_at_3 : deriv f 3 = 1 / (3 * real.log 3) :=
  sorry

end derivative_log_base_3_at_3_l347_347508


namespace maximum_profit_l347_347730

noncomputable def C : ℝ → ℝ
| x => if 0 < x ∧ x < 80 then (1/3)*x^2 + 10*x else 51*x + 10000/x - 1450

noncomputable def L : ℝ → ℝ
| x => if 0 < x ∧ x < 80 then -(1/3)*x^2 + 40*x - 250 else 50*x - 10000/x + 1200

theorem maximum_profit : 
  ∃ x_max : ℝ, L x_max = 1000 ∧ (∀ x : ℝ, 0 < x → L x ≤ L x_max) :=
begin
  use 100,
  split,
  { -- Prove that L(100) = 1000
    sorry },
  { -- Prove that L(x) ≤ 1000 for all x > 0
    sorry }
end

end maximum_profit_l347_347730


namespace parallel_vectors_lambda_l347_347886

open Real

theorem parallel_vectors_lambda (λ : ℝ) : 
  let a := (1, -2)
  let b := (λ, 1)
  (a.1 * b.2 - a.2 * b.1 = 0) → λ = -1 / 2 :=
begin
  intros h,
  sorry
end

end parallel_vectors_lambda_l347_347886


namespace angle_B_in_trapezoid_l347_347929

theorem angle_B_in_trapezoid
  (ABCD_trapezoid : ∀ A B C D : ℝ, Trapezoid ABCD ∧ (AB ∥ CD))
  (A_eq_3D : ∀ A D : ℝ, A = 3 * D)
  (B_eq_2C : ∀ B C : ℝ, B = 2 * C) :
  ∀ B : ℝ, B = 120 :=
by
  sorry

end angle_B_in_trapezoid_l347_347929


namespace max_length_of_each_piece_l347_347597

theorem max_length_of_each_piece (a b c d : ℕ) (h1 : a = 48) (h2 : b = 72) (h3 : c = 108) (h4 : d = 120) : Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 12 := by
  sorry

end max_length_of_each_piece_l347_347597


namespace parallelogram_angle_ratio_l347_347959

theorem parallelogram_angle_ratio (A B C D E : Type)
  (h_parallelogram : is_parallelogram A B C D)
  (h_ext_AB : extends_through A B E)
  (h_ext_CD : extends_through C D E)
  (S : ℝ)
  (S' : ℝ)
  (h_S_def : S = ∠ AEB + ∠ CED)
  (h_S'_def : S' = ∠ DAC + ∠ DCB)
  (r : ℝ)
  (h_r_def : r = S / S') :
  r = 1 :=
sorry

end parallelogram_angle_ratio_l347_347959


namespace balls_into_boxes_l347_347327

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347327


namespace find_t_cards_with_known_numbers_l347_347970

noncomputable def largest_known t := 2013

theorem find_t_cards_with_known_numbers :
  ∀ (cards : Type) [fintype cards] (numbers : cards → ℕ), unique_numbers numbers →
  (∀ (choose10 : fin 10 → cards), ∃ n, n ∈ (set.range numbers) (choose10 ∘ fin.val)) →
  ∃ (t : ℕ), t = 1986 :=
begin
  intros cards fintype_cards numbers unique_numbers choose10 exists_n,
  let t := 1986,
  use t,
  sorry
end

end find_t_cards_with_known_numbers_l347_347970


namespace married_men_fraction_l347_347913

-- define the total number of women
def W : ℕ := 7

-- define the number of single women
def single_women (W : ℕ) : ℕ := 3

-- define the probability of picking a single woman
def P_s : ℚ := single_women W / W

-- define number of married women
def married_women (W : ℕ) : ℕ := W - single_women W

-- define number of married men
def married_men (W : ℕ) : ℕ := married_women W

-- define total number of people
def total_people (W : ℕ) : ℕ := W + married_men W

-- define fraction of married men
def married_men_ratio (W : ℕ) : ℚ := married_men W / total_people W

-- theorem to prove that the ratio is 4/11
theorem married_men_fraction : married_men_ratio W = 4 / 11 := 
by 
  sorry

end married_men_fraction_l347_347913


namespace triangles_from_decagon_l347_347096

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347096


namespace alex_total_marbles_l347_347596

theorem alex_total_marbles :
  ∀ (lorin_black marbles) (jimmy_yellow_marbles : ℕ), 
  lorin_black marbles = 4 ∧ jimmy_yellow_marbles = 22 →
  let alex_black_marbles := 2 * lorin_black marbles in
  let alex_yellow_marbles := jimmy_yellow_marbles / 2 in
  alex_black_marbles + alex_yellow_marbles = 19 :=
by
  intros lorin_black_marbles jimmy_yellow_marbles
  rintro ⟨hb, hy⟩
  let alex_black_marbles : ℕ := 2 * lorin_black_marbles
  let alex_yellow_marbles : ℕ := jimmy_yellow_marbles / 2
  have h_total : alex_black_marbles + alex_yellow_marbles = 19
  sorry

end alex_total_marbles_l347_347596


namespace ball_box_distribution_l347_347436

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347436


namespace num_triangles_from_decagon_l347_347111

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347111


namespace sum_of_areas_limit_l347_347735

noncomputable def sum_of_areas (a : ℝ) (n : ℕ) : ℝ :=
  let r1 := a / Real.sqrt 3
  let A1 := π * r1^2
  let r := 4 / 9
  (A1 * (1 - r^n)) / (1 - r)

theorem sum_of_areas_limit (a : ℝ) :
  (a > 0) → (Real.sqrt 3 > 0) →
  (tendsto (λ n, sum_of_areas a n) at_top (𝓝 (3 * π * a^2 / 5))) :=
begin
  intros h1 h2,
  sorry
end

end sum_of_areas_limit_l347_347735


namespace distinguish_ball_box_ways_l347_347359

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347359


namespace tangent_line_parabola_k_l347_347159

theorem tangent_line_parabola_k :
  ∃ (k : ℝ), (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → (28 ^ 2 = 4 * 1 * 4 * k)) → k = 49 :=
by
  sorry

end tangent_line_parabola_k_l347_347159


namespace extreme_points_sum_gt_two_l347_347223

noncomputable def f (x : ℝ) (b : ℝ) := x^2 / 2 + b * Real.exp x
noncomputable def f_prime (x : ℝ) (b : ℝ) := x + b * Real.exp x

theorem extreme_points_sum_gt_two
  (b : ℝ)
  (h_b : -1 / Real.exp 1 < b ∧ b < 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : f_prime x₁ b = 0)
  (h_x₂ : f_prime x₂ b = 0)
  (h_x₁_lt_x₂ : x₁ < x₂) :
  x₁ + x₂ > 2 := by
  sorry

end extreme_points_sum_gt_two_l347_347223


namespace ways_to_distribute_balls_l347_347420

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347420


namespace number_of_triangles_in_decagon_l347_347024

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347024


namespace distinct_balls_boxes_l347_347383

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347383


namespace num_ways_to_distribute_balls_into_boxes_l347_347265

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347265


namespace number_of_triangles_in_decagon_l347_347023

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347023


namespace number_of_triangles_in_decagon_l347_347081

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347081


namespace combined_capacity_eq_l347_347671

variable {x y z : ℚ}

-- Container A condition
def containerA_full (x : ℚ) := 0.75 * x
def containerA_initial (x : ℚ) := 0.30 * x
def containerA_diff (x : ℚ) := containerA_full x - containerA_initial x = 36

-- Container B condition
def containerB_full (y : ℚ) := 0.70 * y
def containerB_initial (y : ℚ) := 0.40 * y
def containerB_diff (y : ℚ) := containerB_full y - containerB_initial y = 20

-- Container C condition
def containerC_full (z : ℚ) := (2 / 3) * z
def containerC_initial (z : ℚ) := 0.50 * z
def containerC_diff (z : ℚ) := containerC_full z - containerC_initial z = 12

-- Theorem to prove the total capacity
theorem combined_capacity_eq : containerA_diff x → containerB_diff y → containerC_diff z → 
(218 + 2 / 3 = x + y + z) :=
by
  intros hA hB hC
  sorry

end combined_capacity_eq_l347_347671


namespace ways_to_distribute_balls_l347_347362

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347362


namespace perimeter_is_integer_l347_347925

-- Define the geometric properties provided.
variables (A B C D O : Point)
variables (AB BC CD AD : ℝ)
variables (tangent_point_circle : ℝ)
hypothesis (h1 : perp AB BC)
hypothesis (h2 : perp BC CD)
hypothesis (h3 : BC = tangent_point_circle)
hypothesis (h4 : AD = 2 * BC)
hypothesis (AB_values : Set [4, 6, 8, 10, 12])
hypothesis (CD_values : Set [2, 3, 4, 5, 6]) -- Since CD = BC

-- Define the perimeter calculation given the properties.
noncomputable def perimeter (AB CD : ℝ) : ℝ := AB + 5 * CD

-- Prove the perimeter is an integer for the given cases.
theorem perimeter_is_integer 
  (AB : ℝ) (CD : ℝ) 
  (hAB : AB ∈ AB_values) 
  (hCD : CD ∈ CD_values) : 
  ∃ (k : ℤ), (k : ℝ) = perimeter AB CD :=
by {
  sorry
}

end perimeter_is_integer_l347_347925


namespace problem_l347_347196

noncomputable def discriminant (p q : ℝ) : ℝ := p^2 - 4 * q
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (p q : ℝ) (hq : q = -2 * p - 5) :
  (quadratic 1 p (q + 1) 2 = 0) →
  q = -2 * p - 5 ∧
  discriminant p q > 0 ∧
  (discriminant p (q + 1) = 0 → 
    (p = -4 ∧ q = 3 ∧ ∀ x : ℝ, quadratic 1 p q x = 0 ↔ (x = 1 ∨ x = 3))) :=
by
  intro hroot_eq
  sorry

end problem_l347_347196


namespace modulus_of_z_l347_347588

open Complex

variable (r : ℝ) (z : ℂ)
variable (hr : abs r < 3)
variable (hz : z + 1 / z = r)

theorem modulus_of_z : abs z = 1 := by
  sorry

end modulus_of_z_l347_347588


namespace find_solutions_l347_347169

theorem find_solutions (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_eq : m + n^2 + (Nat.gcd m n)^3 = m * n * (Nat.gcd m n)) :
  (m, n) ∈ { (4, 2), (4, 6), (5, 2), (5, 3) } :=
by
  sorry

end find_solutions_l347_347169


namespace general_formula_l347_347905

noncomputable def seq : ℕ → ℝ
| 0 := 0 -- since sequence is indexed from 1, this could be anything
| 1 := 2
| n := if even n then 2^((n/2) - 1) else 2^((n+1)/2)

theorem general_formula (a : ℕ → ℝ) (h₁ : a 1 = 2) (h₂ : ∀ n ≥ 1, a n * a (n + 1) = 2^n) :
  ∀ n, a n = if even n then 2^((n / 2) - 1) else 2^((n + 1) / 2) :=
sorry

end general_formula_l347_347905


namespace find_A_find_B_l347_347899

-- First problem: Prove A = 10 given 100A = 35^2 - 15^2
theorem find_A (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) : A = 10 := by
  sorry

-- Second problem: Prove B = 4 given (A-1)^6 = 27^B and A = 10
theorem find_B (B : ℕ) (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) (h₂ : (A - 1) ^ 6 = 27 ^ B) : B = 4 := by
  have A_is_10 : A = 10 := by
    apply find_A
    assumption
  sorry

end find_A_find_B_l347_347899


namespace period_of_repeating_decimal_l347_347727

def is_100_digit_number_with_98_sevens (a : ℕ) : Prop :=
  ∃ (n : ℕ), n = 10^98 ∧ a = 1776 + 1777 * n

theorem period_of_repeating_decimal (a : ℕ) (h : is_100_digit_number_with_98_sevens a) : 
  (1:ℚ) / a == 1 / 99 := 
  sorry

end period_of_repeating_decimal_l347_347727


namespace ways_to_distribute_balls_l347_347422

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347422


namespace number_of_arrangements_of_SUCCESS_l347_347786

-- Definitions based on conditions
def word : String := "SUCCESS"
def total_letters : Nat := 7
def repetitions : List Nat := [3, 2, 1, 1]  -- Corresponding to S, C, U, E

-- Lean statement proving the number of arrangements
theorem number_of_arrangements_of_SUCCESS : 
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1) * (Nat.factorial 1)) = 420 := by
  sorry

end number_of_arrangements_of_SUCCESS_l347_347786


namespace num_triangles_from_decagon_l347_347110

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347110


namespace algebraic_expression_value_l347_347669

theorem algebraic_expression_value :
  sqrt (5 - 2 * sqrt 6) + sqrt (7 - 4 * sqrt 3) = 2 - sqrt 2 :=
sorry

end algebraic_expression_value_l347_347669


namespace find_number_l347_347710

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 64) : x = 160 :=
sorry

end find_number_l347_347710


namespace regular_decagon_triangle_count_l347_347035

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347035


namespace richard_ends_at_1_0_probability_l347_347987

-- Define the necessary concepts and variables
variables (n : ℕ)

-- Define the main proposition
def probability_Ending_at_1_0 (n : ℕ) : ℝ :=
  ((n + 0.5) ! ^ 2) / (Real.pi * ((n + 1) !) ^ 2)

-- The main statement of the proof
theorem richard_ends_at_1_0_probability (n : ℕ) :
  let total_steps := 2 * n + 1 in
  let possible_steps := 4 ^ total_steps in
  let valid_steps : ℝ := Σ' (W : Fin (n + 1)), (total_steps ! / (W ! * (W + 1) ! * ((n - W) ! ^ 2))) in
  (valid_steps / possible_steps) = probability_Ending_at_1_0 n := by
    sorry

end richard_ends_at_1_0_probability_l347_347987


namespace ways_to_distribute_balls_l347_347417

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347417


namespace number_of_triangles_in_decagon_l347_347019

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347019


namespace number_of_triangles_in_decagon_l347_347020

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347020


namespace part1_part2_l347_347546

open Real

-- Define the points M and N
def M : ℝ × ℝ := (0, -1)
def N : ℝ × ℝ := (2, 5)

-- Prove the equations of the two lines passing through vertex M when M and N are vertices on one side of the square
theorem part1 :
  ∃ (l₁ l₂ : ℝ × ℝ → Prop),
    (∀ p ∈ {M, N}, l₁ p) ∧ (∀ p ∈ {M, (0,0)}, l₂ p) ∧
    (l₁ (1, 2)) ∧ l₁ = (λ p, 3 * (p.1) - (p.2) - 1 = 0) ∧
    l₂ = (λ p, (p.1) + 3 * (p.2) + 3 = 0) :=
sorry

-- Prove the equation of the other diagonal line when M and N are vertices on the diagonal of the square
theorem part2 :
  ∃ (l : ℝ × ℝ → Prop), (∀ p ∈ {(1,2)}, l p) ∧ l = (λ p, (p.1) + 3 * (p.2) - 7 = 0) :=
sorry

end part1_part2_l347_347546


namespace ways_to_place_balls_in_boxes_l347_347499

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347499


namespace length_of_room_l347_347647

theorem length_of_room 
  (width : ℝ) (cost : ℝ) (rate : ℝ) (area : ℝ) (length : ℝ) 
  (h1 : width = 3.75) 
  (h2 : cost = 24750) 
  (h3 : rate = 1200) 
  (h4 : area = cost / rate) 
  (h5 : area = length * width) : 
  length = 5.5 :=
sorry

end length_of_room_l347_347647


namespace num_triangles_from_decagon_l347_347116

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347116


namespace homework_problem1_homework_problem2_l347_347657

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end homework_problem1_homework_problem2_l347_347657


namespace SUCCESS_arrangement_count_l347_347783

theorem SUCCESS_arrangement_count : 
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  ∃ (ways_to_arrange : ℕ), ways_to_arrange = Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_C) := 420 :=
by
  -- Problem Conditions
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  -- The Proof
  existsi 420
  sorry

end SUCCESS_arrangement_count_l347_347783


namespace decagon_triangle_count_l347_347057

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347057


namespace evaluate_expression_l347_347807

theorem evaluate_expression : 2^(3^2) + 3^(2^3) = 7073 := by
  sorry

end evaluate_expression_l347_347807


namespace num_triangles_from_decagon_l347_347009

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347009


namespace SUCCESS_arrangement_count_l347_347782

theorem SUCCESS_arrangement_count : 
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  ∃ (ways_to_arrange : ℕ), ways_to_arrange = Nat.factorial n / (Nat.factorial n_S * Nat.factorial n_C) := 420 :=
by
  -- Problem Conditions
  let n := 7
  let n_S := 3
  let n_C := 2
  let n_U := 1
  let n_E := 1
  -- The Proof
  existsi 420
  sorry

end SUCCESS_arrangement_count_l347_347782


namespace number_of_triangles_in_decagon_l347_347087

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347087


namespace problem_part1_problem_part2_l347_347208

variable (A B : Set ℝ)
def C_R (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem problem_part1 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  C_R (A ∩ B) = { x : ℝ | x < 3 ∨ x ≥ 6 } :=
by
  intros hA hB
  sorry

theorem problem_part2 :
  A = { x : ℝ | 3 ≤ x ∧ x < 6 } →
  B = { x : ℝ | 2 < x ∧ x < 9 } →
  (C_R B) ∪ A = { x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9 } :=
by
  intros hA hB
  sorry

end problem_part1_problem_part2_l347_347208


namespace general_term_b_sum_c_n_terms_l347_347871

-- Conditions
variable {aₙ : ℕ → ℤ} -- aₙ is an arithmetic sequence where ℤ denotes integers
lemma aₙ_nonzero (n : ℕ) : aₙ n ≠ 0 := sorry -- Given aₙ ≠ 0
lemma a₇_value : 32 * aₙ 3 + 32 * aₙ 11 = aₙ 7 * aₙ 7 := sorry -- 32a₃ + 32a₁₁ = a₇²

-- Sequence bₙ
def b (n : ℕ) : ℕ := 2 ^ (n - 1)

axiom b₇_value : b 7 = a₇

-- general term of bₙ
theorem general_term_b : ∀ n, b n = 2 ^ (n - 1) :=
by sorry

-- Define cₙ
def c (n : ℕ) := n * b n

-- Sum of first n terms of cₙ
def S (n : ℕ) : ℕ := ∑ i in finset.range n, c i

-- Proving the sum formula
theorem sum_c_n_terms (n : ℕ) : S n = (n - 1) * 2 ^ n + 1 :=
by sorry

end general_term_b_sum_c_n_terms_l347_347871


namespace dot_product_property_l347_347238

-- Definitions based on conditions
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

-- Required property
theorem dot_product_property : dot_product (vec_add (scalar_mult 2 vec_a) vec_b) vec_a = 6 :=
by sorry

end dot_product_property_l347_347238


namespace triangles_from_decagon_l347_347101

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347101


namespace num_ways_to_distribute_balls_into_boxes_l347_347257

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347257


namespace number_of_triangles_in_decagon_l347_347045

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347045


namespace find_polynomial_q_l347_347176

noncomputable def polynomial_q (x : ℝ) : ℝ := - (10 / 3) * x^2 + (20 / 3) * x + 10

theorem find_polynomial_q :
  ∃ (a : ℝ), (∀ x : ℝ, polynomial_q x = a * (x - 3) * (x + 1)) ∧ polynomial_q 2 = 10 :=
by
  use - (10 / 3)
  split
  {
    intro x,
    apply polynomial.ext,
    intro,
    simp only [polynomial_q], -- add further simplifficult use substitution
    sorry, -- need to fill in polynomial simplification here
  },
  {
    simp only [polynomial_q],
    norm_num
  }


end find_polynomial_q_l347_347176


namespace leo_remaining_words_l347_347557

variables (words_per_line : ℕ) (lines_per_page : ℕ) (pages_written : ℚ) (total_words_required : ℕ)

theorem leo_remaining_words :
  words_per_line = 10 →
  lines_per_page = 20 →
  pages_written = 1.5 →
  total_words_required = 400 →
  total_words_required - (words_per_line * lines_per_page * pages_written).to_nat = 100 :=
by
  intro h_words_per_line h_lines_per_page h_pages_written h_total_words_required
  rw [h_words_per_line, h_lines_per_page, h_pages_written, h_total_words_required]
  norm_num
  sorry

end leo_remaining_words_l347_347557


namespace max_value_of_function_within_interval_l347_347650

theorem max_value_of_function_within_interval :
  ∀ (x : ℝ), (-real.pi / 2 ≤ x ∧ x ≤ 0) →
  (∃ y, y = 3 * real.sin x + 5 ∧ y ≤ 5) :=
begin
  sorry
end

end max_value_of_function_within_interval_l347_347650


namespace ways_to_distribute_balls_l347_347416

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347416


namespace ways_to_put_balls_in_boxes_l347_347476

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347476


namespace ways_to_distribute_balls_l347_347449

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347449


namespace sin_cos_fourth_power_sum_correct_l347_347944

noncomputable def sin_cos_fourth_power_sum (θ : ℝ) : ℝ :=
  let cos2θ := 1/3 in
  let cos_sq := (cos2θ + 1) / 2 in
  let sin_sq := 1 - cos_sq in
  sin_sq^2 + cos_sq^2

theorem sin_cos_fourth_power_sum_correct {θ : ℝ} (h : cos (2 * θ) = 1 / 3) :
  sin_cos_fourth_power_sum θ = 5 / 9 :=
by
  -- Here, you would typically provide the proof steps. Since we only need the statement:
  sorry

end sin_cos_fourth_power_sum_correct_l347_347944


namespace balls_in_boxes_l347_347293

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347293


namespace find_values_of_a_and_b_l347_347681

theorem find_values_of_a_and_b 
  (a b d : ℤ) 
  (h1 : real.sqrt (12 - a) + real.sqrt (12 + b) = 7)
  (h2 : real.sqrt (13 + a) + real.sqrt (13 + d) = 7) : 
  a = 3 ∧ b = 4 := 
sorry

end find_values_of_a_and_b_l347_347681


namespace area_trapezoid_l347_347548

variables (VO VD VODY K : Type) [HasArea VD] [HasArea KD] [HasArea VK]
variable (area_KOV : ℝ)
variable (segment_ratio : ℝ → ℝ → Prop)
variable (base_property : Type -> Bool)
variable (triangle_area_property : Type -> ℝ -> Bool)
variable (area_of_trapezoid : Type -> ℝ)

axiom base_longer_base (VO VODY : Type) : base_property VO

axiom intersection_ratio (K VD : Type) : segment_ratio VD 3 2

axiom area_triangle_KOV_13_5 (K V O : Type) : triangle_area_property K 13.5

theorem area_trapezoid (VODY : Type) : area_of_trapezoid VODY = 37.5 :=
by
  sorry

end area_trapezoid_l347_347548


namespace solve_for_x_l347_347862

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 4 * x = 0) (h₁ : x ≠ 0) : x = 4 := 
by
  sorry

end solve_for_x_l347_347862


namespace num_triangles_in_decagon_l347_347073

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347073


namespace distinct_balls_boxes_l347_347386

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347386


namespace distinct_balls_boxes_l347_347463

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347463


namespace ways_to_put_balls_in_boxes_l347_347477

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347477


namespace distinct_balls_boxes_l347_347469

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347469


namespace sum_of_D_coordinates_l347_347537

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (10, 5)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def D : ℝ × ℝ := 
  let M := midpoint A C in
  let x := 2 * M.1 - B.1 in
  let y := 2 * M.2 - B.2 in
  (x, y)

theorem sum_of_D_coordinates : (D.1 + D.2) = 14 := by
  let sum := D.1 + D.2
  sorry

end sum_of_D_coordinates_l347_347537


namespace years_taught_third_grade_l347_347966

def total_years : ℕ := 26
def years_taught_second_grade : ℕ := 8

theorem years_taught_third_grade :
  total_years - years_taught_second_grade = 18 :=
by {
  -- Subtract the years taught second grade from the total years
  -- Exact the result
  sorry
}

end years_taught_third_grade_l347_347966


namespace age_difference_l347_347153

-- Denise's age in two years
def denise_age_in_two_years : ℕ := 25

-- Diane's age in six years
def diane_age_in_six_years : ℕ := 25

-- Proof problem: Calculate the difference in age between Diane and Denise
theorem age_difference (h1 : denise_age_in_two_years = 25) (h2 : diane_age_in_six_years = 25) : 
  let denise_age_now := denise_age_in_two_years - 2,
      diane_age_now := diane_age_in_six_years - 6 in
  denise_age_now - diane_age_now = 4 := 
by
  sorry

end age_difference_l347_347153


namespace min_value_inequality_l347_347581

theorem min_value_inequality (p q r s t u : ℝ) (h_sum : p + q + r + s + t + u = 11)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hu : 0 < u) :
  (3 / p) + (12 / q) + (27 / r) + (48 / s) + (75 / t) + (108 / u) ≥ (819 / 11) :=
begin
  sorry
end

end min_value_inequality_l347_347581


namespace find_phi_symmetric_sine_l347_347519

noncomputable def function_is_symmetric (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = f (a + x)

theorem find_phi_symmetric_sine :
  ∃ (φ : ℝ), 0 < φ ∧ φ < real.pi ∧ function_is_symmetric (λ x, real.sin (2 * x + φ)) (real.pi / 6) ∧ φ = real.pi / 6 :=
begin
  sorry
end

end find_phi_symmetric_sine_l347_347519


namespace ways_to_distribute_balls_l347_347376

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347376


namespace ways_to_place_balls_in_boxes_l347_347502

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347502


namespace john_work_days_l347_347933

theorem john_work_days (J : ℕ) (H1 : 1 / J + 1 / 480 = 1 / 192) : J = 320 :=
sorry

end john_work_days_l347_347933


namespace sum_of_elements_of_S_l347_347574

-- Lean definition of our problem

def isRepeatingDecimal (x : ℝ) (a b c d : ℕ) : Prop :=
  x = (1000 * a + 100 * b + 10 * c + d) / 9999

theorem sum_of_elements_of_S :
  ∃ S : Set ℝ, (∀ (x : ℝ), x ∈ S ↔ 
    ∃ a b c d : ℕ, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    isRepeatingDecimal x a b c d) → 
  (Finset.sum (S.to_finset) id = 2520) := 
by
  sorry

end sum_of_elements_of_S_l347_347574


namespace x_eq_y_suff_but_not_necc_l347_347717

theorem x_eq_y_suff_but_not_necc (x y : ℝ) : (x = y) → (|x| = |y|) ∧ (∃ x y, |x| = |y| ∧ x ≠ y) :=
by
  intros h1
  split
  exact congr_arg abs h1
  have h2 := abs_eq_of_sq_eq
  sorry

end x_eq_y_suff_but_not_necc_l347_347717


namespace mean_score_proof_l347_347967

noncomputable def mean_score_all_students : ℝ :=
let F := 90 in          -- Mean score of first class
let S := 75 in          -- Mean score of second class
let T := 65 in          -- Mean score of third class
let ratio_f_s := (2 : ℝ) / 3 in   -- Ratio of students in first class to second class
let ratio_s_t := (4 : ℝ) / 5 in   -- Ratio of students in second class to third class
let f := ratio_f_s * s in
let t := s / ratio_s_t in
let total_students := f + s + t in
let total_scores := F * f + S * s + T * t in
total_scores / total_students

theorem mean_score_proof :
  mean_score_all_students = 55.16 :=
sorry

end mean_score_proof_l347_347967


namespace find_hyperbola_eq_and_line_eq_l347_347863

noncomputable def ellipse_eq : (ℝ × ℝ) → Prop :=
  λ p, let (x, y) := p in (x^2 / 25) + (y^2 / 9) = 1

noncomputable def hyperbola_eq : (ℝ × ℝ) → Prop :=
  λ p, let (x, y) := p in (x^2 / 4) - (y^2 / 12) = 1

noncomputable def point_exists_on_line (P : ℝ × ℝ) (l : (ℝ × ℝ) → Prop) : Prop :=
  l P

theorem find_hyperbola_eq_and_line_eq (focus : ℝ) (p : ℝ × ℝ) 
  (common_focus : focus = 4) 
  (eccentricity_sum : (focus / 5 + focus / a = 14/5) ) 
  (line_pass_p : point_exists_on_line p (λ q, (q.2 = 9*q.1 - 26))) :
  (∀ q, hyperbola_eq q ↔ ((q.1^2 / 4) - (q.2^2 / 12) = 1))
  ∧ (∀ q, (λ q, q.2 = 9 * q.1 - 26) q) :=
by
  -- Sorry here to skip the proofs.
  sorry

end find_hyperbola_eq_and_line_eq_l347_347863


namespace balls_into_boxes_l347_347397

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347397


namespace minimum_polynomial_l347_347952

def polynomial (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range (2*n+1), (2*n+1-i) * (x ^ i)

theorem minimum_polynomial (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, polynomial n x ≥ (n + 1) ∧ polynomial n (-1) = (n + 1) :=
by
  sorry

end minimum_polynomial_l347_347952


namespace enhanced_computer_price_difference_l347_347667

noncomputable def price_of_basic_computer : ℝ := 2000
noncomputable def total_price_basic_computer_printer : ℝ := 2500
noncomputable def price_of_printer : ℝ := total_price_basic_computer_printer - price_of_basic_computer
noncomputable def price_of_enhanced_computer : ℝ :=
  6 * (price_of_printer - (1/6) * price_of_printer)

theorem enhanced_computer_price_difference :
  price_of_enhanced_computer - price_of_basic_computer = 500 :=
by
  have hP : price_of_printer = 500 := by
    rw [←sub_eq_iff_eq_add, sub_sub_cancel, sub_self, sub_zero]
  rw [price_of_enhanced_computer, hP, sub_mul, one_div, mul_div_cancel']
  linarith
  sorry

end enhanced_computer_price_difference_l347_347667


namespace balls_into_boxes_l347_347331

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347331


namespace ondra_homework_problems_l347_347660

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end ondra_homework_problems_l347_347660


namespace sum_of_squares_base_b_l347_347961

theorem sum_of_squares_base_b (b : ℕ) (h : (b + 4)^2 + (b + 8)^2 + (2 * b)^2 = 2 * b^3 + 8 * b^2 + 5 * b) :
  (4 * b + 12 : ℕ) = 62 :=
by
  sorry

end sum_of_squares_base_b_l347_347961


namespace equation_of_C_slope_of_l_l347_347201

variable (a b c : ℝ) (D : ℝ × ℝ) (A B F : ℝ × ℝ) (x y : ℝ)
variable (k : ℝ) (l : ℝ)

axiom ellipse : a > 0 ∧ b > 0 ∧ a > b

axiom right_focus : 2 * F.1 - F.2 - 2 = 0

axiom vertices : A = (-a, 0) ∧ B = (a, 0)

axiom distance_relation : |A.1 - F.1| = 3 * |B.1 - F.1|

axiom line_d : D = (4, 0)

axiom intersections : ∃ P Q : ℝ × ℝ, P ≠ Q ∧ 
  (P.2 = k * (P.1 - 4)) ∧ (Q.2 = k * (Q.1 - 4)) ∧ 
  (P.1 ^ 2 / a^2 + P.2 ^ 2 / b^2 = 1) ∧ 
  (Q.1 ^ 2 / a^2 + Q.2 ^ 2 / b^2 = 1)

axiom midpoint : ∃ N : ℝ × ℝ, N = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

axiom slope_an : (N.2 - A.2) / (N.1 - A.1) = 2 / 5

theorem equation_of_C : ∃ a b : ℝ, a = 2 ∧ b = √3 ∧ 
  (∀ x y, (x ^ 2 / (2:ℝ)^2 + y ^ 2 / (√3:ℝ)^2 = 1) ↔ 
  ∃ x y, (x ^ 2 / 4 + y ^ 2 / 3 = 1)) :=
sorry

theorem slope_of_l : l = -1 / 4 :=
sorry

end equation_of_C_slope_of_l_l347_347201


namespace cube_painted_faces_counts_l347_347737

theorem cube_painted_faces_counts (n : ℕ) :
  let total_cubes_with_4_painted_faces := 12
  let total_cubes_with_1_painted_face := 6
  let total_cubes_with_0_painted_faces := 1
  (num_cubes_with_4_faces n = total_cubes_with_4_painted_faces) ∧
  (num_cubes_with_1_face n = total_cubes_with_1_painted_face) ∧
  (num_cubes_with_0_faces n = total_cubes_with_0_painted_faces) :=
sorry

end cube_painted_faces_counts_l347_347737


namespace value_of_expression_l347_347668

theorem value_of_expression : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end value_of_expression_l347_347668


namespace num_triangles_from_decagon_l347_347007

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347007


namespace area_of_octagon_l347_347679

/--
Given:
- Two congruent squares share the same center O and have sides of length 2.
- The length of segment AB is 7/15.
- Points A, B, C, D, E, F, G, H are points of intersection of the inner rotated square with the outer square, arranged to maintain symmetry.

Prove that the area of octagon ABCDEFGH is 56/15.
-/
theorem area_of_octagon :
  let side_length := 2
  let center := (0, 0)
  let segment_AB := (7 : ℚ) / 15
  let area_octa := (56 : ℚ) / 15
  ∃ (a b c d e f g h : ℚ × ℚ),
    (a.1 = 0 ∧ a.2 = side_length / 2) ∧
    (b.1 = segment_AB / 2 ∧ b.2 = side_length / 2) ∧
    (area_of_octagon = area_octa) := by
      sorry

end area_of_octagon_l347_347679


namespace balls_into_boxes_l347_347394

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347394


namespace correct_choices_are_bd_l347_347705

def is_correct_statement_b : Prop := 
  let S := (1 / 2) * 4 * |(-4)| in
  S = 8

def is_correct_statement_d : Prop :=
  ∀ m : ℝ, ∃ p : ℝ × ℝ, p = (-1, 0) ∧
    (p.1 = -1) → (m * p.1 + p.2 + m = 0)

theorem correct_choices_are_bd : is_correct_statement_b ∧ is_correct_statement_d :=
by 
  sorry

end correct_choices_are_bd_l347_347705


namespace num_triangles_from_decagon_l347_347108

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347108


namespace floor_ceil_sum_l347_347798

open Real

theorem floor_ceil_sum : (⌊1.999⌋ : ℤ) + (⌈3.001⌉ : ℤ) = 5 := by
  sorry

end floor_ceil_sum_l347_347798


namespace a_1000_is_1501_l347_347911

noncomputable def sequence : ℕ → ℤ
| 1 := 502
| 2 := 503
| n := 
  if h : n ≥ 3 then
    let a := sequence (n - 2);
    let b := sequence (n - 1);
    let c := 3 * (n - 2) - a - 2 * b;
    c
  else 0

theorem a_1000_is_1501 :
  let a := sequence in a 1000 = 1501 :=
by sorry

end a_1000_is_1501_l347_347911


namespace homework_problem1_homework_problem2_l347_347659

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end homework_problem1_homework_problem2_l347_347659


namespace ball_box_distribution_l347_347432

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347432


namespace compute_integral_l347_347906

def x_value : ℝ := 3.87

-- Find the integral from 0 to 2 of x^2 dx
def integral_part : ℝ := ∫ x in 0..2, x^2

theorem compute_integral : integral_part = (8 / 3) := by
  sorry

end compute_integral_l347_347906


namespace parabola_focus_l347_347641

theorem parabola_focus :
  ∀ (x y : ℝ), x^2 = 4 * y → (0, 1) = (0, (2 / 2)) :=
by
  intros x y h
  sorry

end parabola_focus_l347_347641


namespace sqrt_ac_bd_le_sqrt_ef_l347_347957

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_ac_bd_le_sqrt_ef
  (a b c d e f : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f)
  (h1 : a + b ≤ e)
  (h2 : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) :=
by
  sorry

end sqrt_ac_bd_le_sqrt_ef_l347_347957


namespace num_triangles_in_decagon_l347_347068

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347068


namespace find_number_l347_347733

theorem find_number (x : ℕ) (h : x * 12 = 540) : x = 45 :=
by sorry

end find_number_l347_347733


namespace ways_to_put_balls_in_boxes_l347_347475

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347475


namespace num_ways_to_distribute_balls_into_boxes_l347_347256

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347256


namespace permutation_exists_l347_347561

-- Define the conditions
variables {n : ℕ} (x : Fin n → ℝ)

-- Condition 1: Sum of x_i equals 1 in absolute value
def sum_condition (x : Fin n → ℝ) : Prop :=
  |∑ i, x i| = 1

-- Condition 2: Absolute value of each x_i is less than or equal to (n + 1) / 2
def absv_condition (x : Fin n → ℝ) : Prop :=
  ∀ i, |x i| ≤ (n + 1) / 2

-- Statement of the theorem to be proven
theorem permutation_exists :
  ∀ (x : Fin n → ℝ),
    sum_condition x → absv_condition x → 
    ∃ (y : Fin n → ℝ) (σ : Equiv.Perm (Fin n)),
      (y = λ i, x (σ i)) ∧ |∑ i in Finset.range n, (i + 1 : ℕ) * y ⟨i, sorry⟩| ≤ (n + 1) / 2 :=
  sorry

end permutation_exists_l347_347561


namespace number_of_triangles_in_decagon_l347_347021

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347021


namespace distinct_balls_boxes_l347_347381

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347381


namespace sum_of_areas_of_cross_sections_l347_347197

-- Let a regular tetrahedron ABCD with edge length 2 be given
def regular_tetrahedron {A B C D : Type} (edge_length : ℝ) :=
  edge_length = 2 ∧
  ∀ (P Q : Type), P ≠ Q → dist P Q = edge_length

-- Proof statement: The sum of the areas of the cross-sections obtained by cutting the regular tetrahedron with a plane equidistant from its four vertices is sqrt(3) + 3
theorem sum_of_areas_of_cross_sections
  (A B C D : Type)
  (edge_length : ℝ)
  (h_tetrahedron : regular_tetrahedron edge_length)
  (plane_equidistant : ∀ (P : Type), dist P (plane_equidistant P) = dist A (plane_equidistant A)) :
  ∑ (sec_area : ℝ), sec_area = sqrt 3 + 3 :=
by
  sorry

end sum_of_areas_of_cross_sections_l347_347197


namespace segment_length_294_l347_347973

theorem segment_length_294
  (A B P Q : ℝ)   -- Define points A, B, P, Q on the real line
  (h1 : P = A + (3 / 8) * (B - A))   -- P divides AB in the ratio 3:5
  (h2 : Q = A + (4 / 11) * (B - A))  -- Q divides AB in the ratio 4:7
  (h3 : Q - P = 3)                   -- The length of PQ is 3
  : B - A = 294 := 
sorry

end segment_length_294_l347_347973


namespace number_of_triangles_in_decagon_l347_347083

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347083


namespace balls_into_boxes_l347_347321

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347321


namespace convert_polar_to_rectangular_l347_347147

def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  polar_to_rectangular 5 (3 * Real.pi / 4) = (- 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2) ∧ 
  polar_to_rectangular 6 (5 * Real.pi / 3) = (3, -3 * Real.sqrt 3) :=
by {
  sorry
}

end convert_polar_to_rectangular_l347_347147


namespace matrix_element_squares_sum_condition_l347_347572

variables {A : Matrix (Fin 2) (Fin 2) ℝ}
variables {B : Matrix (Fin 2) (Fin 2) ℝ}

-- Definition of orthogonality
def is_orthogonal (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  Mᵀ = M⁻¹

theorem matrix_element_squares_sum_condition
  (a b c d x y z w : ℝ)
  (A : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]])
  (B : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]])
  (hA : is_orthogonal A)
  (hB : is_orthogonal B)
  : x ^ 2 + y ^ 2 + z ^ 2 + w ^ 2 = 2 :=
sorry

end matrix_element_squares_sum_condition_l347_347572


namespace min_distance_PQ_l347_347213

-- Definitions of the functions
def func1 (x : ℝ) := -x^2 + 3 * Real.log x
def func2 (x : ℝ) := x + 2

-- Definition of the distance function
def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Points on the respective graphs
def P : ℝ × ℝ := (1, func1 1)
def Q : ℝ × ℝ := (c, func2 c) -- Q would be defined dependent on finding the minimum distance; here c should result in the minimum distance

-- The theorem to prove the minimum distance between P and Q is 2√2
theorem min_distance_PQ :
  distance P Q = 2 * Real.sqrt 2 :=
sorry

end min_distance_PQ_l347_347213


namespace fungi_growth_day_l347_347910

theorem fungi_growth_day (n : ℕ) (h : 2 * 3^n > 200) : n = 5 :=
begin
  sorry
end

end fungi_growth_day_l347_347910


namespace transformed_parabola_eq_l347_347921

noncomputable def initial_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3
def shift_left (h : ℝ) (c : ℝ): ℝ := h - c
def shift_down (k : ℝ) (d : ℝ): ℝ := k - d

theorem transformed_parabola_eq :
  ∃ (x : ℝ), (initial_parabola (shift_left x 2) - 1 = 2 * (x + 1)^2 + 2) :=
sorry

end transformed_parabola_eq_l347_347921


namespace hyperbola_asymptotes_equation_l347_347644

theorem hyperbola_asymptotes_equation :
  ∀ (x y : ℝ), x ^ 2 - y ^ 2 / 3 = 1 → y = sqrt 3 * x ∨ y = -sqrt 3 * x :=
by
  intros x y hyp_eq
  sorry

end hyperbola_asymptotes_equation_l347_347644


namespace ways_to_distribute_balls_l347_347371

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347371


namespace num_triangles_from_decagon_l347_347011

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347011


namespace num_ways_to_distribute_balls_into_boxes_l347_347254

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347254


namespace imaginary_part_eq_two_l347_347646

def imaginary_part_of_complex (z : ℂ) : ℂ :=
  let i := complex.I in
  (2 + i) * i

theorem imaginary_part_eq_two :
  let z := imaginary_part_of_complex (2 + complex.I) in
  z.im = 2 :=
by
  sorry

end imaginary_part_eq_two_l347_347646


namespace balls_into_boxes_l347_347408

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347408


namespace distinguish_ball_box_ways_l347_347357

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347357


namespace xy_value_l347_347896

theorem xy_value (x y : ℝ) (h : |x - 5| + |y + 3| = 0) : x * y = -15 := by
  sorry

end xy_value_l347_347896


namespace number_of_arrangements_of_SUCCESS_l347_347787

-- Definitions based on conditions
def word : String := "SUCCESS"
def total_letters : Nat := 7
def repetitions : List Nat := [3, 2, 1, 1]  -- Corresponding to S, C, U, E

-- Lean statement proving the number of arrangements
theorem number_of_arrangements_of_SUCCESS : 
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1) * (Nat.factorial 1)) = 420 := by
  sorry

end number_of_arrangements_of_SUCCESS_l347_347787


namespace eccentricity_of_ellipse_equation_of_ellipse_l347_347854

-- Define the conditions
def ellipse (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) ≤ 1
def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := ((-sqrt (a^2 - b^2), 0), (sqrt (a^2 - b^2), 0))
def line_l1 (a : ℝ) (c : ℝ) := ∀ y, ∃ x, (x, y) ∈ ellipse a (sqrt(a^2 - c^2)) ∧ (x = c ∨ x = -c)
def AF1_AF2_relation (A : ℝ × ℝ) (F1 F2 : ℝ × ℝ) := 
    dist A F1 = 7 * dist A F2

-- Statement 1: Proving eccentricity
theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0)
    (h3 : |AF1| = 7 * |AF2|)
    (h4 : ∀ A F1 F2 : ℝ × ℝ, line_l1 a (sqrt (a^2 - b^2)) A ∧ 
       AF1_AF2_relation A F1 F2) :
    (a^2 - b^2) / a^2 = 3 / 4 :=
sorry

-- Define the conditions for the second part
def line_through_F1 (F1 : ℝ × ℝ) (slope : ℝ) := ∀ Q : ℝ × ℝ, Q.2 = slope * Q.1 + F1.2

-- Statement 2: Proving the equation of the ellipse
theorem equation_of_ellipse (a b : ℝ) (h : a > b > 0)
    (h2: ∀ (f1 f2 : ℝ × ℝ), foci a b = (f1, f2))
    (h3: ∀ l, line_through_F1 (fst (foci a b)) 1 l) 
    (h4: triangle_area OMN = (2 * sqrt 6) / 5)
    (h5: ∃ M N : ℝ × ℝ, l M ∧ l N ∧ (M ≠ N)) :
    ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1 :=
sorry

end eccentricity_of_ellipse_equation_of_ellipse_l347_347854


namespace shape_is_cylinder_l347_347834

def positive_constant (c : ℝ) := c > 0

def is_cylinder (r θ z : ℝ) (c : ℝ) : Prop :=
  r = c

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) 
  (h_pos : positive_constant c) (h_eq : r = c) :
  is_cylinder r θ z c := by
  sorry

end shape_is_cylinder_l347_347834


namespace number_of_triangles_in_decagon_l347_347089

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347089


namespace function_domain_l347_347154

theorem function_domain :
  (∀ x, (x-3) / (x^2 + 4*x + 3) ≠ 0 ↔ x ∈ (-∞ : ℝ) ∪ (-3 : ℝ) ∪ (-1 : ℝ) ∪ (∞ : ℝ)) :=
by sorry

end function_domain_l347_347154


namespace increase_in_daily_mess_expenses_l347_347674

theorem increase_in_daily_mess_expenses (A X : ℝ)
  (h1 : 35 * A = 420)
  (h2 : 42 * (A - 1) = 420 + X) :
  X = 42 :=
by
  sorry

end increase_in_daily_mess_expenses_l347_347674


namespace average_score_is_correct_l347_347912

def frequency_intervals : List (ℝ × ℕ) :=
  [ (65, 3), (75, 16), (85, 24), (95, 7) ]

def total_people : ℕ := 50

def estimated_average_score (freqs : List (ℝ × ℕ)) (total : ℕ) : ℝ :=
  (freqs.map (fun (score, count) => score * count)).sum / total

theorem average_score_is_correct :
  estimated_average_score frequency_intervals total_people = 82 :=
by
  sorry

end average_score_is_correct_l347_347912


namespace minimal_valid_tiling_l347_347174

-- Definitions (conditions derived)
def valid_tiling (n : ℕ) : Prop :=
  ∃ (f : ℝ × ℝ → ℕ), (∀ x y, x ≠ y → f x = f y → (dist x y ≠ 1)) ∧ ∀ (m : ℕ), m > n → ∀ x, f x ≠ m

-- The statement of the problem
theorem minimal_valid_tiling :
  ∃ n, valid_tiling n ∧ ∀ k, k < n → ¬ valid_tiling k :=
  sorry

end minimal_valid_tiling_l347_347174


namespace abc_inequality_l347_347551

theorem abc_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l347_347551


namespace find_a_l347_347568

def A := { x : ℝ | x^2 + 4 * x = 0 }
def B (a : ℝ) := { x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 1) = 0 }

theorem find_a (a : ℝ) :
  (∀ x : ℝ, x ∈ (A ∩ B a) ↔ x ∈ B a) → (a = 1 ∨ a ≤ -1) :=
by 
  sorry

end find_a_l347_347568


namespace balls_into_boxes_l347_347334

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347334


namespace number_of_triangles_in_decagon_l347_347086

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347086


namespace arrangement_of_SUCCESS_l347_347789

theorem arrangement_of_SUCCESS : 
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  (fact total_letters) / (fact count_S * fact count_C * fact count_U * fact count_E) = 420 := 
by
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  exact sorry

end arrangement_of_SUCCESS_l347_347789


namespace find_angle_ABC_find_DC_length_l347_347523

-- Given definitions and conditions
variables 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AC BC AD : ℝ)
  (angle_BAC angle_ABC angle_C : ℝ)

-- Conditions
def AC_length : Prop := AC = 4
def BC_length : Prop := BC = 4 * real.sqrt 3
def angle_BAC_measure : Prop := angle_BAC = 2 * real.pi / 3

-- Additional conditions for question II
def AD_length : Prop := AD = real.sqrt 7
def angle_ABC_result : Prop := angle_ABC = real.pi / 6

-- Question I: Prove that the angle ABC = π/6
theorem find_angle_ABC 
  (h_AC : AC_length)
  (h_BC : BC_length)
  (h_angle_BAC : angle_BAC_measure)
  : angle_ABC = real.pi / 6 := by
  sorry

-- Additional condition for question II
def angle_C_measure : Prop := angle_C = real.pi / 6

-- Question II: Prove that length DC can be either 3√3 or √3
theorem find_DC_length
  (h_AC : AC_length)
  (h_BC : BC_length)
  (h_angle_BAC : angle_BAC_measure)
  (h_AD : AD_length)
  (h_angle_ABC : angle_ABC_result)
  (h_angle_C : angle_C_measure)
  : ∃ DC, DC = 3 * real.sqrt 3 ∨ DC = real.sqrt 3 := by
  sorry

end find_angle_ABC_find_DC_length_l347_347523


namespace total_horse_food_l347_347764

theorem total_horse_food (ratio_sh_to_h : ℕ → ℕ → Prop) 
    (sheep : ℕ) 
    (ounce_per_horse : ℕ) 
    (total_ounces_per_day : ℕ) : 
    ratio_sh_to_h 5 7 → sheep = 40 → ounce_per_horse = 230 → total_ounces_per_day = 12880 :=
by
  intros h_ratio h_sheep h_ounce
  sorry

end total_horse_food_l347_347764


namespace prime_mod4_eq3_has_x0_y0_solution_l347_347942

theorem prime_mod4_eq3_has_x0_y0_solution (p x0 y0 : ℕ) (h1 : Nat.Prime p) (h2 : p % 4 = 3)
    (h3 : (p + 2) * x0^2 - (p + 1) * y0^2 + p * x0 + (p + 2) * y0 = 1) :
    p ∣ x0 :=
sorry

end prime_mod4_eq3_has_x0_y0_solution_l347_347942


namespace find_omega_l347_347520

theorem find_omega : ∃ (ω : ℝ), (∀ x, sin (ω * x + ω * (π / 3)) = cos (ω * x)) → ω = 3 / 2 :=
begin
  sorry
end

end find_omega_l347_347520


namespace num_triangles_from_decagon_l347_347117

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347117


namespace regular_tetrahedron_dihedral_angle_l347_347996

noncomputable def tetrahedron_dihedral_angle : ℝ :=
  let ABCD := "regular tetrahedron"
  let O := "circumcenter of ABCD"
  let B := "vertex of ABCD"
  let C := "vertex of ABCD"
  let E := midpoint(B, C)
  dihedral_angle(A, B, O, E)

theorem regular_tetrahedron_dihedral_angle :
  tetrahedron_dihedral_angle = 2 * π / 3 := 
sorry

end regular_tetrahedron_dihedral_angle_l347_347996


namespace comprehensive_survey_option_l347_347708

def suitable_for_comprehensive_survey (survey : String) : Prop :=
  survey = "Survey on the components of the first large civil helicopter in China"

theorem comprehensive_survey_option (A B C D : String)
  (hA : A = "Survey on the number of waste batteries discarded in the city every day")
  (hB : B = "Survey on the quality of ice cream in the cold drink market")
  (hC : C = "Survey on the current mental health status of middle school students nationwide")
  (hD : D = "Survey on the components of the first large civil helicopter in China") :
  suitable_for_comprehensive_survey D :=
by
  sorry

end comprehensive_survey_option_l347_347708


namespace balls_in_boxes_l347_347284

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347284


namespace triangular_black_cells_l347_347715

theorem triangular_black_cells (n : ℕ) (T : ℕ → ℕ) :
  (∀ k, T k = k * (k + 1) / 2) →
  (∃ n, n % 8 = 0 ∧ T n = 120) :=
begin
  intros hT,
  use 15,
  split,
  { norm_num, },
  { rw hT,
    norm_num,
    ring, }
end

end triangular_black_cells_l347_347715


namespace binomial_expansion_judgments_l347_347835

theorem binomial_expansion_judgments :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r) ∧
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r + 3) :=
by
  sorry

end binomial_expansion_judgments_l347_347835


namespace find_sum_of_common_ratios_l347_347721

-- Definition of the problem conditions
def is_geometric_sequence (a b c : ℕ) (k : ℕ) (r : ℕ) : Prop :=
  b = k * r ∧ c = k * r * r

-- Main theorem statement
theorem find_sum_of_common_ratios (k p r a_2 a_3 b_2 b_3 : ℕ) 
  (hk : k ≠ 0)
  (hp_neq_r : p ≠ r)
  (hp_seq : is_geometric_sequence k a_2 a_3 k p)
  (hr_seq : is_geometric_sequence k b_2 b_3 k r)
  (h_eq : a_3 - b_3 = 3 * (a_2 - b_2)) :
  p + r = 3 :=
sorry

end find_sum_of_common_ratios_l347_347721


namespace vector_properties_l347_347240

variables {V : Type*} [innerProductSpace ℝ V]

-- Define unit vectors and their dot product condition
variables (a b : V) (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hab : ⟪a, b⟫ = 1 / 2)

-- Theorem stating both correct conclusions
theorem vector_properties :
  ∥a + b∥ = sqrt 3 ∧ proj b a = (1 / 2 : ℝ) • b :=
begin
  sorry
end

end vector_properties_l347_347240


namespace ways_to_put_balls_in_boxes_l347_347489

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347489


namespace john_sells_20_woodburnings_l347_347934

variable (x : ℕ)

theorem john_sells_20_woodburnings (price_per_woodburning cost profit : ℤ) 
  (h1 : price_per_woodburning = 15) (h2 : cost = 100) (h3 : profit = 200) :
  (profit = price_per_woodburning * x - cost) → 
  x = 20 :=
by
  intros h_profit
  rw [h1, h2, h3] at h_profit
  linarith

end john_sells_20_woodburnings_l347_347934


namespace ways_to_distribute_balls_l347_347410

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347410


namespace num_triangles_from_decagon_l347_347106

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347106


namespace distance_between_points_PQ_l347_347883

theorem distance_between_points_PQ :
  let P := (1 : ℝ, 3 : ℝ)
  let Q := (4 : ℝ, -1 : ℝ)
  dist P Q = 5 :=
by
  sorry

end distance_between_points_PQ_l347_347883


namespace kishore_savings_amount_l347_347750

def mr_kishore_savings (rent milk groceries education petrol medical utilities misc total_expense : ℝ) (s : ℝ) : Prop :=
  rent = 7000 ∧ milk = 1800 ∧ groceries = 6000 ∧ education = 3500 ∧ petrol = 2500 ∧ medical = 1500 ∧ utilities = 1200 ∧ misc = 800 ∧ 
  total_expense = rent + milk + groceries + education + petrol + medical + utilities + misc ∧  
  s - total_expense = 0.15 * s

theorem kishore_savings_amount :
  mr_kishore_savings 7000 1800 6000 3500 2500 1500 1200 800 24300 28588.2353 →
  (0.15 * 28588.2353 ≈ 4288.24) := 
by
  intro h
  sorry

end kishore_savings_amount_l347_347750


namespace cannot_determine_right_triangle_l347_347760

theorem cannot_determine_right_triangle (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = 180) →
  (\<^angle A = B + C) →
  (5 * a = 13 * c ∧ 12 * b = 13 * c) →
  (a^2 = (b+c) * (b-c)) →
  (\<^angle A = 3 * x ∧ \<^angle B = 4 * x ∧ \<^angle C = 5 * x) →
  (12 * x = 180) →
  (x ≠ 15 → ∃ (A B C : ℝ), A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90) :=
by sorry

end cannot_determine_right_triangle_l347_347760


namespace sufficient_but_not_necessary_l347_347997

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, (0 < x ∧ x < 2) → (x < 2)) ∧ ¬(∀ x : ℝ, (0 < x ∧ x < 2) ↔ (x < 2)) :=
sorry

end sufficient_but_not_necessary_l347_347997


namespace students_per_group_correct_l347_347726

def total_students : ℕ := 850
def number_of_teachers : ℕ := 23
def students_per_group : ℕ := total_students / number_of_teachers

theorem students_per_group_correct : students_per_group = 36 := sorry

end students_per_group_correct_l347_347726


namespace ways_to_distribute_balls_l347_347419

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347419


namespace num_triangles_from_decagon_l347_347010

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347010


namespace spacy_subsets_15_l347_347150

def spacy_subsets_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | (k + 5) => spacy_subsets_count (k + 4) + spacy_subsets_count k

theorem spacy_subsets_15 : spacy_subsets_count 15 = 181 :=
sorry

end spacy_subsets_15_l347_347150


namespace distinguish_ball_box_ways_l347_347347

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347347


namespace max_k_value_l347_347951

open Finset

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem max_k_value (k : ℕ) (A : Fin k (Finset ℕ)) :
  (∀ i : Fin k, A[i] ⊆ S) →
  (∀ i : Fin k, A[i].card = 5) →
  (∀ i j : Fin k, i ≠ j → (A[i] ∩ A[j]).card ≤ 2) →
  k ≤ 6 :=
sorry -- Proof goes here

-- The statement verifies that for any collection of sets A satisfying the above conditions,
-- the number of such sets k cannot exceed 6.

end max_k_value_l347_347951


namespace magnitude_of_2a_plus_b_l347_347884

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_2a_plus_b : 
  let a := (3, 2) in
  let b := (-1, 1) in
  magnitude ( (2 * a.1 + b.1, 2 * a.2 + b.2) ) = 5 * real.sqrt 2 :=
by
  let a := (3, 2)
  let b := (-1, 1)
  let sum := (2 * a.1 + b.1, 2 * a.2 + b.2)
  have eqn1 : sum = (5, 5) := by
    dsimp [a, b]
    rw [mul_add, mul_add]
    rw [mul_one, mul_one]
  rw [eqn1]
  dsimp [magnitude]
  sorry

end magnitude_of_2a_plus_b_l347_347884


namespace log_relationship_l347_347186

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem log_relationship :
  a > b ∧ b > c := by
  sorry

end log_relationship_l347_347186


namespace num_ways_to_distribute_balls_into_boxes_l347_347261

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347261


namespace ways_to_distribute_balls_l347_347455

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347455


namespace find_duplicate_page_l347_347752

theorem find_duplicate_page (n p : ℕ) (h : (n * (n + 1) / 2) + p = 3005) : p = 2 := 
sorry

end find_duplicate_page_l347_347752


namespace find_tan_of_angle_A_l347_347538

-- Define the problem as Lean assumptions and a goal theorem

variable (A B C : Type) [InnerProductSpace ℝ A] 

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem find_tan_of_angle_A (h_right : is_right_triangle 8 AC 17)
    (h_90 : ∠BAC = π/2) 
    (h_AB : AB = 8)
    (h_BC : BC = 17)
    (h_AC : AC = 15):
  Real.tan A = 15/8 :=
sorry

end find_tan_of_angle_A_l347_347538


namespace find_m_l347_347853

variable {a_n : ℕ → ℤ}
variable {S : ℕ → ℤ}

def isArithmeticSeq (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a_n (n + 1) = a_n n + d

def sumSeq (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n, S n = (n * (a_n 1 + a_n n)) / 2

theorem find_m
  (d : ℤ)
  (a_1 : ℤ)
  (a_n : ∀ n, ℤ)
  (S : ℕ → ℤ)
  (h_arith : isArithmeticSeq a_n d)
  (h_sum : sumSeq S a_n)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end find_m_l347_347853


namespace balls_into_boxes_l347_347340

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347340


namespace total_weight_all_bags_sold_l347_347545

theorem total_weight_all_bags_sold (morning_potatoes afternoon_potatoes morning_onions afternoon_onions morning_carrots afternoon_carrots : ℕ)
  (weight_potatoes weight_onions weight_carrots total_weight : ℕ)
  (h_morning_potatoes : morning_potatoes = 29)
  (h_afternoon_potatoes : afternoon_potatoes = 17)
  (h_morning_onions : morning_onions = 15)
  (h_afternoon_onions : afternoon_onions = 22)
  (h_morning_carrots : morning_carrots = 12)
  (h_afternoon_carrots : afternoon_carrots = 9)
  (h_weight_potatoes : weight_potatoes = 7)
  (h_weight_onions : weight_onions = 5)
  (h_weight_carrots : weight_carrots = 4)
  (h_total_weight : total_weight = 591) :
  morning_potatoes + afternoon_potatoes * weight_potatoes +
  morning_onions + afternoon_onions * weight_onions +
  morning_carrots + afternoon_carrots * weight_carrots = total_weight :=
by {
  sorry
}

end total_weight_all_bags_sold_l347_347545


namespace num_triangles_from_decagon_l347_347008

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347008


namespace ways_to_distribute_balls_l347_347368

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347368


namespace number_of_triangles_in_regular_decagon_l347_347141

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347141


namespace sum_of_roots_correct_l347_347631

noncomputable def sum_of_roots : ℝ :=
  let equation := ∀ x : ℝ, cos(x)^2 + cos(8 * x)^2 = 2 * cos(x)^2 * cos(8 * x)^2
  let interval := [3 * real.pi, 6 * real.pi]
  let sum_roots := ∑ x in interval, if equation x then x else 0
  56.55

theorem sum_of_roots_correct : sum_of_roots = 56.55 := by
  sorry

end sum_of_roots_correct_l347_347631


namespace min_value_expression_l347_347582

noncomputable def f (x y : ℝ) : ℝ := 
  (x + 1 / y) * (x + 1 / y - 2023) + (y + 1 / x) * (y + 1 / x - 2023)

theorem min_value_expression : ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ f x y = -2048113 :=
sorry

end min_value_expression_l347_347582


namespace num_ways_to_distribute_balls_into_boxes_l347_347258

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347258


namespace number_of_triangles_in_regular_decagon_l347_347131

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347131


namespace max_in_circle_eqn_l347_347927

theorem max_in_circle_eqn : 
  ∀ (x y : ℝ), (x ≥ 0) → (y ≥ 0) → (4 * x + 3 * y ≤ 12) → (x - 1)^2 + (y - 1)^2 = 1 :=
by
  intros x y hx hy hineq
  sorry

end max_in_circle_eqn_l347_347927


namespace num_ways_to_distribute_balls_into_boxes_l347_347259

theorem num_ways_to_distribute_balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls = 243) := 
by
  sorry

end num_ways_to_distribute_balls_into_boxes_l347_347259


namespace age_ratio_in_ten_years_l347_347540

-- Definitions of given conditions
variable (A : ℕ) (B : ℕ)
axiom age_condition : A = 20
axiom sum_of_ages : A + 10 + (B + 10) = 45

-- Theorem and proof skeleton for the ratio of ages in ten years.
theorem age_ratio_in_ten_years (A B : ℕ) (hA : A = 20) (hSum : A + 10 + (B + 10) = 45) :
  (A + 10) / (B + 10) = 2 := by
  sorry

end age_ratio_in_ten_years_l347_347540


namespace put_balls_in_boxes_l347_347307

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347307


namespace Ondra_problems_conditions_l347_347665

-- Define the conditions as provided
variables {a b : ℤ}

-- Define the first condition where the subtraction is equal to the product.
def condition1 : Prop := a + b = a * b

-- Define the second condition involving the relationship with 182.
def condition2 : Prop := a * b * (a + b) = 182

-- The statement to be proved: Ondra's problems (a, b) are (2, 2) and (1, 13) or (13, 1)
theorem Ondra_problems_conditions {a b : ℤ} (h1 : condition1) (h2 : condition2) :
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
sorry

end Ondra_problems_conditions_l347_347665


namespace region_Z_probability_l347_347748

variable (P : Type) [Field P]
variable (P_X P_Y P_W P_Z : P)

theorem region_Z_probability :
  P_X = 1 / 3 → P_Y = 1 / 4 → P_W = 1 / 6 → P_X + P_Y + P_Z + P_W = 1 → P_Z = 1 / 4 := by
  sorry

end region_Z_probability_l347_347748


namespace regular_decagon_triangle_count_l347_347027

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347027


namespace max_min_diff_triang_ABC_l347_347550

theorem max_min_diff_triang_ABC
  (BC : ℝ) (BD : ℝ) (DC : ℝ) (AD : ℝ)
  (hBC : BC = 10) (hBD : BD = 5) (hDC : DC = 5) (hAD : AD = 6)
  (hMidpoint : BD = DC)
  : let AB2 := λ x h : ℝ, x^2 + h
    let AC2 := λ x h : ℝ, (10 - x)^2 + h
    let diff := λ x : ℝ, AB2 x (36 - (5 - x)^2) - AC2 x (36 - (5 - x)^2) 
    ∃ x_max x_min : ℝ, x_max = 5 ∧ x_min = 0 ∧
      diff x_max = -10 ∧ diff x_min = -100
    ∧ (-10) - (-100) = 90 := 
by {
  let AB2 := λ x h : ℝ, x^2 + h,
  let AC2 := λ x h : ℝ, (10 - x)^2 + h,
  let diff := λ x : ℝ, AB2 x (36 - (5 - x)^2) - AC2 x (36 - (5 - x)^2),
  use [5, 0],
  split,
  { refl, },
  split,
  { refl, },
  split,
  { sorry, },
  split,
  { sorry, },
  sorry,
}

end max_min_diff_triang_ABC_l347_347550


namespace sum_ineq_l347_347960

noncomputable def seq (a1 : ℝ) : ℕ → ℝ 
| 0     := a1
| (n+1) := (seq n ^ 2 + 1) / (2 * seq n)

theorem sum_ineq (a1 : ℝ) (n : ℕ) (h : a1 > 1) :
  (∑ i in finset.range n, seq a1 i) < n + 2 * (a1 - 1) :=
sorry

end sum_ineq_l347_347960


namespace curve_descriptions_and_min_distance_l347_347220
noncomputable def C3 := λ t : ℝ, (3 + 2 * t, -2 + t)

def C1 := λ t : ℝ, (-4 + Real.cos t, 3 + Real.sin t)
def P := C1 (Real.pi / 2)

def C2 := λ θ : ℝ, (8 * Real.cos θ, 3 * Real.sin θ)
def M (θ : ℝ) := ((-2 + 4 * Real.cos θ), (2 + 3 / 2 * Real.sin θ))

def line_eq (x y : ℝ) := x - 2 * y - 7

def min_distance_to_line (θ : ℝ) : ℝ :=
  abs ((4 * Real.cos θ) - (3 * Real.sin θ) - 13) / Real.sqrt 5

theorem curve_descriptions_and_min_distance :
  (∀ t : ℝ, (C1 t).fst + 4) ^ 2 + ((C1 t).snd - 3) ^ 2 = 1 ∧
  (∀ θ : ℝ, ((C2 θ).fst ^ 2 / 64) + ((C2 θ).snd ^ 2 / 9) = 1) ∧
  (min_distance_to_line (Real.arccos (4 / 5)) = (abs (5 * (Real.sin (Real.arccos (4 / 5)) + 2 / 5)) / (Real.sqrt 5)) ∧
   min_distance_to_line (Real.arccos (4 / 5)) = 8 * Real.sqrt 5 / 5) := sorry

end curve_descriptions_and_min_distance_l347_347220


namespace horse_speed_l347_347639

def square_side_length (area : ℝ) : ℝ :=
  real.sqrt area

def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

def speed_of_horse (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem horse_speed (area : ℝ) (time : ℝ) (h1 : area = 625) (h2 : time = 4) :
  speed_of_horse (perimeter_of_square (square_side_length area)) time = 25 := 
by
  sorry

end horse_speed_l347_347639


namespace number_of_triangles_in_regular_decagon_l347_347128

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347128


namespace number_of_triangles_in_decagon_l347_347046

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347046


namespace product_of_YZ_values_l347_347676

noncomputable section

variables (X Y Z : ℝ)

def XY := 10
def XZ := 3

def YZ_1 := XY + XZ
def YZ_2 := XY - XZ

theorem product_of_YZ_values : YZ_1 * YZ_2 = 91 :=
by 
  have h1 : YZ_1 = 13 := by sorry
  have h2 : YZ_2 = 7 := by sorry
  rw [h1, h2]
  exact eq.refl 91

end product_of_YZ_values_l347_347676


namespace balls_into_boxes_l347_347407

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347407


namespace regular_decagon_triangle_count_l347_347039

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347039


namespace number_of_triangles_in_decagon_l347_347026

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347026


namespace seconds_in_3_hours_25_minutes_l347_347888

theorem seconds_in_3_hours_25_minutes:
  let hours := 3
  let minutesInAnHour := 60
  let additionalMinutes := 25
  let secondsInAMinute := 60
  (hours * minutesInAnHour + additionalMinutes) * secondsInAMinute = 12300 := 
by
  sorry

end seconds_in_3_hours_25_minutes_l347_347888


namespace find_g_inv_f_10_l347_347509

variable (f g : ℝ → ℝ)
variable (h₁ : ∀ x, f⁻¹(g(x)) = x^2 + 1)
variable (h₂ : Function.HasInverse g)

theorem find_g_inv_f_10 : g⁻¹(f 10) = 3 ∨ g⁻¹(f 10) = -3 := by
  sorry

end find_g_inv_f_10_l347_347509


namespace decagon_triangle_count_l347_347061

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347061


namespace g_monotonic_intervals_t_greater_than_constant_l347_347856

noncomputable theory

open Real

def f (x t : ℝ) : ℝ := (x - 1) * exp x - t / 2 * x ^ 2 - 2 * x
def g (x t : ℝ) : ℝ := exp x - 2 / x - t

-- Problem: Prove the monotonicity and conditions in Lean statement
theorem g_monotonic_intervals (t : ℝ) :
  ∀ {x : ℝ}, (x ∈ Ioo (-∞ : ℝ) 0 ∨ x ∈ Ioo 0 (∞ : ℝ)) → 0 < exp x + 2 / (x ^ 2) :=
sorry

theorem t_greater_than_constant {t x1 x2 : ℝ}
  (h1 : x1 < x2)
  (h2 : f x1 t + 5 / (2 * exp 1) - 1 < 0)
  (h3 : ∃ x1 x2, derivative (λ x, f x t) x1 = 0 ∧ derivative (λ x, f x t) x2 = 0) :
  t > 2 + 1 / exp 1 :=
sorry

end g_monotonic_intervals_t_greater_than_constant_l347_347856


namespace terminating_unique_configuration_l347_347608

-- Definitions
def move_a (a : ℤ → ℕ) (n : ℤ) : ℤ → ℕ :=
λ m, if m = n - 1 then a m - 1
     else if m = n then a m - 1
     else if m = n + 1 then a m + 1
     else a m

def move_b (a : ℤ → ℕ) (n : ℤ) : ℤ → ℕ :=
λ m, if m = n then a m - 2
     else if m = n + 1 then a m + 1
     else if m = n - 2 then a m + 1
     else a m

-- Main theorem statement
theorem terminating_unique_configuration (a : ℤ → ℕ) :
  ∃ b : ℤ → ℕ, (∀ n, (a = move_a b n ∨ a = move_b b n)) → (∀ n, a n ≤ 1) ∧ (∀ n, ¬(a n = 1 ∧ a (n+1) = 1)) :=
sorry

end terminating_unique_configuration_l347_347608


namespace number_of_triangles_in_decagon_l347_347014

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347014


namespace number_of_triangles_in_decagon_l347_347041

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347041


namespace decagon_triangle_count_l347_347055

theorem decagon_triangle_count : 
  ∀ (V : Finset ℕ), V.card = 10 → (∀ a b c ∈ V, a ≠ b → b ≠ c → a ≠ c) → (Finset.card { S : Finset (Fin ℕ) // S ⊆ V ∧ S.card = 3 } = 120) := 
by
  intros V hV hNCL
  sorry

end decagon_triangle_count_l347_347055


namespace D_144_l347_347584

def D (n : ℕ) : ℕ :=
  if n = 1 then 1 else sorry

theorem D_144 : D 144 = 51 := by
  sorry

end D_144_l347_347584


namespace num_triangles_in_decagon_l347_347070

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347070


namespace homework_problem1_homework_problem2_l347_347658

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end homework_problem1_homework_problem2_l347_347658


namespace num_odd_even_subsets_equal_total_capacity_odd_even_equal_total_capacity_odd_subsets_l347_347592

-- Definitions based on the conditions provided
def S_n (n : ℕ) : set ℕ := { i | 1 ≤ i ∧ i ≤ n }
def capacity (Z : set ℕ) : ℕ := Z.sum id
def is_odd_subset (n : ℕ) (Z : set ℕ) : Prop := capacity Z % 2 = 1
def is_even_subset (n : ℕ) (Z : set ℕ) : Prop := capacity Z % 2 = 0

-- 1. Prove the number of odd subsets equals the number of even subsets
theorem num_odd_even_subsets_equal (n : ℕ) : 
  (finset.filter (is_odd_subset n) (finset.powerset (S_n n))).card = 
  (finset.filter (is_even_subset n) (finset.powerset (S_n n))).card := sorry

-- 2. Prove that for n ≥ 3, the total capacity of all odd subsets equals the total capacity of all even subsets
theorem total_capacity_odd_even_equal (n : ℕ) (h : n ≥ 3) : 
  (finset.filter (is_odd_subset n) (finset.powerset (S_n n))).sum capacity = 
  (finset.filter (is_even_subset n) (finset.powerset (S_n n))).sum capacity := sorry

-- 3. For n ≥ 3, find the total capacity of all odd subsets
theorem total_capacity_odd_subsets (n : ℕ) (h : n ≥ 3) : 
  (finset.filter (is_odd_subset n) (finset.powerset (S_n n))).sum capacity = 
  2^(n - 3) * n * (n + 1) := sorry

end num_odd_even_subsets_equal_total_capacity_odd_even_equal_total_capacity_odd_subsets_l347_347592


namespace smallest_N_is_105_l347_347742

section
  def is_palindromic_in_base (n b : ℕ) : Prop := 
    let digits := nat.digits b n in 
    digits = digits.reverse

  def problem_statement := ∃ (N : ℕ), N > 20 ∧ is_palindromic_in_base N 14 ∧ is_palindromic_in_base N 20 ∧ N = 105

  theorem smallest_N_is_105 : problem_statement :=
  sorry
end

end smallest_N_is_105_l347_347742


namespace total_time_is_11_l347_347767

-- Define the times each person spent in the pool
def Jerry_time : Nat := 3
def Elaine_time : Nat := 2 * Jerry_time
def George_time : Nat := Elaine_time / 3
def Kramer_time : Nat := 0

-- Define the total time spent in the pool by all friends
def total_time : Nat := Jerry_time + Elaine_time + George_time + Kramer_time

-- Prove that the total time is 11 minutes
theorem total_time_is_11 : total_time = 11 := sorry

end total_time_is_11_l347_347767


namespace correct_choices_are_bd_l347_347704

def is_correct_statement_b : Prop := 
  let S := (1 / 2) * 4 * |(-4)| in
  S = 8

def is_correct_statement_d : Prop :=
  ∀ m : ℝ, ∃ p : ℝ × ℝ, p = (-1, 0) ∧
    (p.1 = -1) → (m * p.1 + p.2 + m = 0)

theorem correct_choices_are_bd : is_correct_statement_b ∧ is_correct_statement_d :=
by 
  sorry

end correct_choices_are_bd_l347_347704


namespace cover_with_radius_root3_over_3_l347_347868

-- Definitions based on given problem conditions
def diameter (s : set (ℝ × ℝ)) : ℝ :=
  Sup { dist p q | p ∈ s ∧ q ∈ s }

def can_cover_with_circle (s : set (ℝ × ℝ)) (r : ℝ) : Prop :=
  ∃ (c : ℝ × ℝ), ∀ p ∈ s, dist p c ≤ r

-- The main theorem to prove
theorem cover_with_radius_root3_over_3 (M : set (ℝ × ℝ)) (hdiam : diameter M = 1) :
  can_cover_with_circle M (real.sqrt 3 / 3) :=
by
  sorry  -- Proof omitted

end cover_with_radius_root3_over_3_l347_347868


namespace arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l347_347670

-- Definitions based on conditions in A)
def students : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def A : Char := 'A'
def B : Char := 'B'
def C : Char := 'C'
def D : Char := 'D'
def E : Char := 'E'
def F : Char := 'F'
def G : Char := 'G'

-- Holistic theorem statements for each question derived from the correct answers in B)
theorem arrangement_A_and_B_adjacent :
  ∃ (n : ℕ), n = 1440 := sorry

theorem arrangement_A_B_and_C_adjacent :
  ∃ (n : ℕ), n = 720 := sorry

theorem arrangement_A_and_B_adjacent_C_not_ends :
  ∃ (n : ℕ), n = 960 := sorry

theorem arrangement_ABC_and_DEFG_units :
  ∃ (n : ℕ), n = 288 := sorry

end arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l347_347670


namespace gabe_playlist_plays_l347_347840

def song_length_1 : ℕ := 3
def song_length_2 : ℕ := 2
def song_length_3 : ℕ := 3
def ride_time : ℕ := 40

theorem gabe_playlist_plays :
  let playlist_length := song_length_1 + song_length_2 + song_length_3 in
  ride_time / playlist_length = 5 := 
by 
  let playlist_length := song_length_1 + song_length_2 + song_length_3
  have h : playlist_length = 8 := by sorry
  have h2 : ride_time / playlist_length = 40 / 8 := by sorry
  have h3 : 40 / 8 = 5 := by sorry
  exact h2.trans h3

end gabe_playlist_plays_l347_347840


namespace eval_expression_l347_347802

-- Definitions for the floor and ceiling functions
def floor (x : ℝ) : ℤ := Int.floor x
def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Problem statement to prove
theorem eval_expression : floor 1.999 + ceiling 3.001 = 5 :=
by
  sorry

end eval_expression_l347_347802


namespace balls_in_boxes_l347_347282

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347282


namespace arithmetic_mean_is_correct_l347_347813

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_in_list : List ℕ := [31, 33, 35, 37, 39, 41]

def arithmetic_mean_of_primes (l : List ℕ) : ℚ :=
  let primes := l.filter is_prime
  let sum_primes := primes.foldl (· + ·) 0
  sum_primes / primes.length

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_primes primes_in_list = 109 / 3 :=
by
  sorry

end arithmetic_mean_is_correct_l347_347813


namespace ways_to_put_balls_in_boxes_l347_347485

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347485


namespace find_chemistry_marks_l347_347149

variable (English : ℕ) (Mathematics : ℕ) (Physics : ℕ) (Biology : ℕ) (Average : ℕ)

theorem find_chemistry_marks (h1 : English = 86) 
                            (h2 : Mathematics = 85) 
                            (h3 : Physics = 82) 
                            (h4 : Biology = 85) 
                            (h5 : Average = 85) 
                            (n_subjects : ℕ := 5) 
                            (total_marks := Average * n_subjects): 
                            total_marks - (English + Mathematics + Physics + Biology) = 87 := 
by {
  rw [h1, h2, h3, h4, h5],
  calc
    total_marks - (English + Mathematics + Physics + Biology)
        = 425 - (86 + 85 + 82 + 85) : by sorry
        ... = 87 : by sorry
}

end find_chemistry_marks_l347_347149


namespace range_of_alpha_range_of_x_plus_y_l347_347541

noncomputable def curve_cartesian := { p : ℝ × ℝ // (p.1^2 + p.2^2 - 6 * p.1 + 1 = 0) }
noncomputable def line_parametric (α : ℝ) (t : ℝ) := (-1 + t * Real.cos α, t * Real.sin α)

theorem range_of_alpha :
  ∀ α : ℝ, ∃ t : ℝ, line_parametric α t ∈ curve_cartesian → (0 ≤ α ∧ α < Real.pi) ∧ (Real.cos α ≥ Real.sqrt 2 / 2 ∨ Real.cos α ≤ -Real.sqrt 2 / 2) :=
sorry

theorem range_of_x_plus_y :
  ∀ (M : ℝ × ℝ), M ∈ curve_cartesian → (-1 ≤ M.1 + M.2 ∧ M.1 + M.2 ≤ 7) :=
sorry

end range_of_alpha_range_of_x_plus_y_l347_347541


namespace intersection_point_l347_347173

def line_eq (x y z : ℝ) : Prop :=
  (x - 1) / 1 = (y + 1) / 0 ∧ (y + 1) / 0 = (z - 1) / -1

def plane_eq (x y z : ℝ) : Prop :=
  3 * x - 2 * y - 4 * z - 8 = 0

theorem intersection_point : 
  ∃ (x y z : ℝ), line_eq x y z ∧ plane_eq x y z ∧ x = -6 ∧ y = -1 ∧ z = 8 :=
by 
  sorry

end intersection_point_l347_347173


namespace eccentricity_condition_l347_347866

noncomputable def ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  set ℝ :=
{e : ℝ | sqrt (2 - sqrt 2) ≤ e ∧ e < 1}

theorem eccentricity_condition
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (A B : ℝ × ℝ)
  (hAonEllipse : (A.1 ^ 2 / a ^ 2 + A.2 ^ 2 / b ^ 2 = 1))
  (hBonEllipse : (B.1 ^ 2 / a ^ 2 + B.2 ^ 2 / b ^ 2 = 1))
  (F1 F2 : ℝ × ℝ) 
  (hFocus : F1 = (c, 0) ∧ F2 = (-c, 0)) -- Assuming foci are on x-axis for simplicity, c = sqrt(a^2 - b^2)
  (hOrigin : (0, 0))
  (hPerp : A.1 * B.1 + A.2 * B.2 = 0)
  (hDotProduct : (A.1 - c, A.2) ∘ (A.1 + c, A.2) + (B.1 - c, B.2) ∘ (B.1 + c, B.2) = 0
  ) :
  (sqrt (2 - sqrt 2) ≤ c / a ∧ c / a < 1) :=
sorry

end eccentricity_condition_l347_347866


namespace ways_to_put_balls_in_boxes_l347_347478

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347478


namespace ways_to_distribute_balls_l347_347418

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347418


namespace number_of_possible_integral_values_for_BC_l347_347955

-- Definitions
def Triangle (A B C : Type) := -- Definition of a triangle with vertices A, B, C

-- Conditions
variables {A B C D E F : Type}
variable [Triangle A B C]
variable (AB AC BC AD : ℝ)
variable (BC : ℤ)

-- Preconditions
axiom AB_eq_7 : AB = 7
axiom angle_bisector : AD bisects ∠BAC
axiom parallel_AD_EF : parallel AD (line E F)
axiom equal_area : divides_triangle_into_equal_areas AD E F

-- The problem's statement
theorem number_of_possible_integral_values_for_BC : 
  7 < BC ∧ BC < 21 → ∃! (n : ℕ), n = 13 := 
by
  sorry

end number_of_possible_integral_values_for_BC_l347_347955


namespace ways_to_put_balls_in_boxes_l347_347481

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347481


namespace Ondra_problems_conditions_l347_347663

-- Define the conditions as provided
variables {a b : ℤ}

-- Define the first condition where the subtraction is equal to the product.
def condition1 : Prop := a + b = a * b

-- Define the second condition involving the relationship with 182.
def condition2 : Prop := a * b * (a + b) = 182

-- The statement to be proved: Ondra's problems (a, b) are (2, 2) and (1, 13) or (13, 1)
theorem Ondra_problems_conditions {a b : ℤ} (h1 : condition1) (h2 : condition2) :
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
sorry

end Ondra_problems_conditions_l347_347663


namespace distinguish_ball_box_ways_l347_347356

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347356


namespace number_of_triangles_in_regular_decagon_l347_347118

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347118


namespace tan_squared_sum_geq_three_over_eight_l347_347956

theorem tan_squared_sum_geq_three_over_eight 
  (α β γ : ℝ) 
  (hα : 0 ≤ α ∧ α < π / 2) 
  (hβ : 0 ≤ β ∧ β < π / 2) 
  (hγ : 0 ≤ γ ∧ γ < π / 2) 
  (h_sum : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3 / 8 := 
sorry

end tan_squared_sum_geq_three_over_eight_l347_347956


namespace maximum_value_fraction_sum_l347_347943

theorem maximum_value_fraction_sum :
  ∃ (A B C D : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  (∀ (A' B' C' D' : ℕ), A' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} → B' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} → C' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} → D' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} → 
    A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' → 
    A' / B' + C' / D' ≤ 13) :=
  (∃ (A B C D : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  A / B + C / D = 13)
  sorry

end maximum_value_fraction_sum_l347_347943


namespace parabola_intersection_x_diff_l347_347826

theorem parabola_intersection_x_diff :
  let f := λ x : ℝ, 3 * x^2 - 6 * x + 7,
      g := λ x : ℝ, -3 * x^2 + 4 * x + 5 in
  let x_coords := {x | f x = g x} in
  ∃ a c : ℝ, a ∈ x_coords ∧ c ∈ x_coords ∧ c ≥ a ∧ (c - a = 2 / 3) :=
by
  sorry

end parabola_intersection_x_diff_l347_347826


namespace problem_1_problem_2_l347_347874

noncomputable def f (a b x : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem problem_1 (a b : ℝ)
  (h1 : deriv (f a b) (-1) = 0)
  (h2 : f a b (-1) = 1) :
  a = 1 ∧ b = -1 := sorry

noncomputable def g (x : ℝ) : ℝ := (f 1 (-1) x) + Real.exp (2 * x - 1)

theorem problem_2 :
  let k := deriv g 1,
  let pt := (1 : ℝ, g 1)
  (k = 2 * Real.exp 1 ∧ pt = (1, Real.exp 1 - 3)) →
  ∃ a b : ℝ, (a, b) = (2 * Real.exp 1, -(Real.exp 1 + 3)) := sorry

end problem_1_problem_2_l347_347874


namespace intersection_of_circles_on_diameters_lies_on_BC_l347_347621

-- Assume given conditions.
variables {A B C : Type} [EuclideanGeometry A]
variables (triangle_ABC : Triangle A B C)
variables (circle_AB : Circle (Midpoint A B) (A.dist B / 2))
variables (circle_AC : Circle (Midpoint A C) (A.dist C / 2))

-- Define the theorem
theorem intersection_of_circles_on_diameters_lies_on_BC (H : A ≠ B ∧ A ≠ C ∧ B ≠ C) (P : A ≠ P) 
  (H1 : P ∈ circle_AB) (H2 : P ∈ circle_AC) : P ∈ line_through B C := 
begin
  sorry 
end

end intersection_of_circles_on_diameters_lies_on_BC_l347_347621


namespace floyd_infinite_jumps_l347_347180

def sum_of_digits (n: Nat) : Nat := 
  n.digits 10 |>.sum 

noncomputable def jumpable (a b: Nat) : Prop := 
  b > a ∧ b ≤ 2 * a 

theorem floyd_infinite_jumps :
  ∃ f : ℕ → ℕ, 
    (∀ n : ℕ, jumpable (f n) (f (n + 1))) ∧
    (∀ m n : ℕ, m ≠ n → sum_of_digits (f m) ≠ sum_of_digits (f n)) :=
sorry

end floyd_infinite_jumps_l347_347180


namespace measure_of_third_angle_l347_347680

-- Definitions based on given conditions
def angle_sum_of_triangle := 180
def angle1 := 30
def angle2 := 60

-- Problem Statement: Prove the third angle (angle3) in a triangle is 90 degrees
theorem measure_of_third_angle (angle_sum : ℕ := angle_sum_of_triangle) 
  (a1 : ℕ := angle1) (a2 : ℕ := angle2) : (angle_sum - (a1 + a2)) = 90 :=
by
  sorry

end measure_of_third_angle_l347_347680


namespace roots_reciprocal_l347_347697

theorem roots_reciprocal {a b c x y : ℝ} (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (a * x^2 + b * x + c = 0) ↔ (c * y^2 + b * y + a = 0) := by
sorry

end roots_reciprocal_l347_347697


namespace pq_root_of_quad_eq_l347_347947

noncomputable def solve_pq : ℝ × ℝ :=
  let p := 10
  let q := 2
  (p, q)

theorem pq_root_of_quad_eq :
  ∃ (p q : ℝ), solve_pq = (p, q) ∧
                (p + 5 * complex.I) * (q + 3 * complex.I) = 20 + 40 * complex.I ∧
                (p + 5 * complex.I) + (q + 3 * complex.I) = 12 + 8 * complex.I :=
by {
  use 10,
  use 2,
  split,
  { reflexivity, },
  { split,
    { field_simp [complex.I, complex.I_re, complex.I_im],
      ring },
    { field_simp [complex.I, complex.I_re, complex.I_im],
      ring } }
}

end pq_root_of_quad_eq_l347_347947


namespace distance_measured_is_approximately_976_32_miles_l347_347607

-- Define the conversions and known quantities
def inches_to_centimeters := 2.54
def centimeters_to_inches (cm : ℝ) : ℝ := cm / inches_to_centimeters
def inches_to_miles := 40 / 2.5

-- Define the total distance in centimeters
def distance_cm := 155
-- Convert the distance to inches
def distance_in_inches := centimeters_to_inches distance_cm
-- Convert the distance in inches to miles
def distance_in_miles := distance_in_inches * inches_to_miles

-- Theorem statement using the given conditions
theorem distance_measured_is_approximately_976_32_miles : 155 / 2.54 * (40 / 2.5) ≈ 976.32 :=
by
  sorry

end distance_measured_is_approximately_976_32_miles_l347_347607


namespace student_marks_problem_l347_347666

-- Define the variables
variables (M P C X : ℕ)

-- State the conditions
-- Condition 1: M + P = 70
def condition1 : Prop := M + P = 70

-- Condition 2: C = P + X
def condition2 : Prop := C = P + X

-- Condition 3: (M + C) / 2 = 45
def condition3 : Prop := (M + C) / 2 = 45

-- The theorem stating the problem
theorem student_marks_problem (h1 : condition1 M P) (h2 : condition2 C P X) (h3 : condition3 M C) : X = 20 :=
by sorry

end student_marks_problem_l347_347666


namespace number_of_triangles_in_regular_decagon_l347_347122

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347122


namespace row_column_product_sets_not_identical_l347_347909

theorem row_column_product_sets_not_identical :
  (∃ (grid : Matrix ℕ (Fin 9) (Fin 9)),
    (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 81) ∧
    (∀ n, ∃! (i, j), grid i j = n) ∧
    (set.univ.image (λ i, (Matrix.vec_prod i (λ j, grid i j))) =
     set.univ.image (λ j, (Matrix.vec_prod j (λ i, grid i j)))).false) :=
sorry

end row_column_product_sets_not_identical_l347_347909


namespace construct_triangle_l347_347778

theorem construct_triangle (a b : ℝ) (h : 0 < b) (hα : ∃ α β : ℝ, α = 3 * β ∧ α + β + ? = π) :
  b < a ∧ a < 3 * b ↔ ∃ (triangle : Type), (∃ A B C : triangle, ∃ sa sb sc : ℝ, 
    sa = a ∧ sb = b ∧ (sa * sa + sb * sb = sc * sc)) := sorry

end construct_triangle_l347_347778


namespace ondra_homework_problems_l347_347662

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end ondra_homework_problems_l347_347662


namespace initial_number_of_girls_is_31_l347_347183

-- Define initial number of boys and girls
variables (b g : ℕ)

-- Conditions
def first_condition (g b : ℕ) : Prop := b = 3 * (g - 18)
def second_condition (g b : ℕ) : Prop := 4 * (b - 36) = g - 18

-- Theorem statement
theorem initial_number_of_girls_is_31 (b g : ℕ) (h1 : first_condition g b) (h2 : second_condition g b) : g = 31 :=
by
  sorry

end initial_number_of_girls_is_31_l347_347183


namespace num_triangles_in_decagon_l347_347072

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347072


namespace ways_to_distribute_balls_l347_347367

theorem ways_to_distribute_balls (n_balls : ℕ) (n_boxes : ℕ) (h_n_balls : n_balls = 5) (h_n_boxes : n_boxes = 3) : 
  n_boxes ^ n_balls = 243 := 
by 
  rw [h_n_balls, h_n_boxes]
  exact pow_succ 3 4
  sorry

end ways_to_distribute_balls_l347_347367


namespace ways_to_distribute_balls_l347_347456

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347456


namespace figures_on_pages_l347_347532

theorem figures_on_pages (A B C D : Type) (pages : fin 6 → set (A ⊕ B ⊕ C ⊕ D))
  (fixed_order : ∀ (p : fin 6), ∀ x y : A ⊕ B ⊕ C ⊕ D, x ≠ y → x ∈ pages p → y ∈ pages p → false)
  (no_more_than_two : ∀ (p : fin 6), card (pages p) ≤ 2) :
  ∃ ways : ℕ, ways = 225 := 
begin
  have cases1 : ℕ := nat.choose 6 4,
  have cases2 : ℕ := nat.choose 6 3 * nat.choose 4 2,
  have cases3 : ℕ := nat.choose 6 2 * nat.choose 4 2,
  
  let ways := cases1 + cases2 + cases3,
  have h : ways = 225,
  { rw [cases1, cases2, cases3],
    simp [nat.choose],
    -- Prove each calculation step explicitly if needed, or just simplify directly
    exact sorry, -- Calculation details omitted
  },
  exact ⟨ways, h⟩,
end

end figures_on_pages_l347_347532


namespace number_of_triangles_in_regular_decagon_l347_347134

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347134


namespace john_haircut_length_l347_347555

noncomputable def haircut_growth (hair_growth : ℕ → ℝ) (months_between_haircuts : ℕ) : ℝ :=
  hair_growth months_between_haircuts

theorem john_haircut_length :
  let hair_growth_per_month := 1.5
  let haircut_down_to := 6
  let cost_per_haircut := 54
  let total_annual_cost := 324
  let number_of_haircuts_per_year := total_annual_cost / cost_per_haircut
  let months_between_haircuts := 12 / number_of_haircuts_per_year
  let hair_growth_between_haircuts := haircut_growth (λ n, hair_growth_per_month * n) months_between_haircuts
  haircut_down_to + hair_growth_between_haircuts = 9 :=
by
  sorry

end john_haircut_length_l347_347555


namespace largest_angle_of_triangle_l347_347867

-- Defining the conditions.
def altitudes (a b c : ℕ) := a = 9 ∧ b = 12 ∧ c = 18

-- Statement for the proof problem.
theorem largest_angle_of_triangle (a b c : ℕ) (h : altitudes a b c) : 
    largest_angle a b c = 104.5 :=
by 
  sorry

end largest_angle_of_triangle_l347_347867


namespace ball_box_distribution_l347_347427

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347427


namespace two_of_a_b_c_have_same_magnitude_l347_347941

open Complex

theorem two_of_a_b_c_have_same_magnitude
  (n : ℕ) (h : n > 1)
  (a b c : ℂ)
  (h1 : a + b + c = 0)
  (h2 : a ^ n + b ^ n + c ^ n = 0) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (x ≠ y) ∧ (|x| = |y|) :=
sorry

end two_of_a_b_c_have_same_magnitude_l347_347941


namespace range_of_f_l347_347878

noncomputable def f (x : ℝ) : ℝ := - (2 / (x - 1))

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, (0 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2) ∧ f x = y} = 
  {y : ℝ | y ≤ -2 ∨ 2 ≤ y} :=
by
  sorry

end range_of_f_l347_347878


namespace transformed_curve_l347_347547

theorem transformed_curve 
  (x x' y y' : ℝ)
  (h_x' : x' = 2 * x)
  (h_y' : y' = 3 * y)
  (h_curve : y = (1/3) * sin (2 * x)) : 
  y' = sin x' := 
by 
  sorry

end transformed_curve_l347_347547


namespace line_tangent_to_parabola_k_value_l347_347156

theorem line_tangent_to_parabola_k_value :
  ∃ k : ℝ, (∀ x y : ℝ, 4 * x + 7 * y + k = 0 → y ^ 2 = 16 * x) → k = 49 :=
by
  -- definitions of the line and parabola
  let line := (x y : ℝ) → 4 * x + 7 * y + k = 0
  let parabola := (y : ℝ) → y ^ 2 = 16 * (-7*y - k)/4
  -- proof to be filled
  sorry

end line_tangent_to_parabola_k_value_l347_347156


namespace ball_in_boxes_l347_347270

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347270


namespace cross_product_ratio_range_l347_347779

variable {R : Type*} [OrderedRing R]
variable (a b c : R^3)

/-- Define the cross product magnitude --/
def cross_product_magnitude (a b : R^3) : R :=
  (Vector3.norm a) * (Vector3.norm b) * Real.sin (Vector3.angle a b)

/-- Main theorem —/
theorem cross_product_ratio_range
  (h1 : a + b + c = Vector3.zero)
  (h2 : Vector3.norm a = Vector3.norm b)
  (h3 : Vector3.norm b = Vector3.norm c) :
  ∃ k ∈ (Set.Icc (0 : R) 2), 
  |(cross_product_magnitude a b + cross_product_magnitude b c + cross_product_magnitude c a) / 
  (Vector3.dot a b + Vector3.dot b c + Vector3.dot c a)| = k :=
sorry

end cross_product_ratio_range_l347_347779


namespace number_of_ways_to_fill_grid_with_conditions_l347_347616

theorem number_of_ways_to_fill_grid_with_conditions :
  ∃ (f : Fin 9 → Fin 9) (all_diff : ∀ i j, i ≠ j → f i ≠ f j)
    (ordered_rows : ∀ i j : Fin 3, i < j → f (i * 3 + k) < f (j * 3 + k))
    (ordered_cols : ∀ i j : Fin 3, i < j → f (i + k * 3) < f (j + k * 3))
    (center_condition : f 4 = 5),
  number_of_ways_to_fill_grid f = 18 := 
sorry

end number_of_ways_to_fill_grid_with_conditions_l347_347616


namespace log_base_8_of_2_l347_347805

theorem log_base_8_of_2 : log 8 2 = 1 / 3 := by
  -- Condition: 8 = 2^3
  have h1 : 8 = 2^3 := by norm_num
  -- Condition: definition of logarithm
  -- log_b a = c   iff   b^c = a
  have h2 : 8 ^ (1 / 3) = 2 := by rw [h1]; exact rfl
  -- Conclusion: log 8 2 = 1 / 3
  have h3 : log 8 2 = 1 / 3 := by rw [←log_eq_iff_pow_eq]; exact h2
  exact h3

end log_base_8_of_2_l347_347805


namespace balls_into_boxes_l347_347405

/-- 
Prove that the number of ways to put 5 distinguishable balls into 3 distinguishable boxes
is equal to 243.
-/
theorem balls_into_boxes : (3^5 = 243) :=
  by
    sorry

end balls_into_boxes_l347_347405


namespace triangle_angle_at_least_108_l347_347192

/-- Given 5 points in a plane, none of which are collinear, there is at least one triangle among the triangles determined by these points with an angle that is not smaller than 108 degrees. -/
theorem triangle_angle_at_least_108 (P1 P2 P3 P4 P5 : ℝ × ℝ)
  (h1 : ¬ collinear P1 P2 P3)
  (h2 : ¬ collinear P1 P2 P4)
  (h3 : ¬ collinear P1 P2 P5)
  (h4 : ¬ collinear P1 P3 P4)
  (h5 : ¬ collinear P1 P3 P5)
  (h6 : ¬ collinear P1 P4 P5)
  (h7 : ¬ collinear P2 P3 P4)
  (h8 : ¬ collinear P2 P3 P5)
  (h9 : ¬ collinear P2 P4 P5)
  (h10 : ¬ collinear P3 P4 P5) :
  ∃ (A B C : ℝ × ℝ), angle A B C ≥ 108 * (π / 180) :=
sorry

end triangle_angle_at_least_108_l347_347192


namespace midpoint_locus_is_circle_l347_347823

open Real EuclideanGeometry

noncomputable def locus_of_midpoints (O M : Point) (circle_radius : ℝ) : Set Point :=
  {P : Point | ∃ A B : Point, circle (center O) circle_radius ∧ midPoint A B = P ∧ onLineSegment M A B}

theorem midpoint_locus_is_circle (O M : Point) (circle_radius : ℝ) :
  O ≠ M →
  insideCircle M O circle_radius →
  locus_of_midpoints O M circle_radius = circle (midPoint O M) (dist O M / 2) := 
sorry

end midpoint_locus_is_circle_l347_347823


namespace flowchart_descriptions_correct_l347_347703

theorem flowchart_descriptions_correct :
  (∀ (desc1 desc2 desc3 desc4 : Prop),
    (desc1 ↔ ¬(∀ t, t > 0 → t = t + 1)) → -- condition ①: the loop in a flowchart cannot be infinite.
    (desc2 ↔ ∃ f g, f ≠ g ∧ ∀ h, (h = f ∨ h = g → True)) → -- condition ②: the flowchart for an algorithm is not unique.
    (desc3 ↔ ∃ s e, s = "start" ∧ e = "end") → -- condition ③: every flowchart must have a start and an end block.
    (desc4 ↔ ∃ ep, ep = "entry" ∧ ∃ ex1 ex2, ex1 ≠ ex2 ∧ (ex1 = "exit" ∨ ex2 = "exit"))) → -- condition ④: a flowchart has only one entry point and can have multiple exit points.
  (desc3)) := 
by sorry

end flowchart_descriptions_correct_l347_347703


namespace midpoint_example_l347_347814

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_example :
  midpoint (2, 9) (8, 3) = (5, 6) :=
by
  sorry

end midpoint_example_l347_347814


namespace intersection_S_T_eq_l347_347231

def S : Set ℝ := { x | (x - 2) * (x - 3) ≥ 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T_eq : (S ∩ T) = { x | (0 < x ∧ x ≤ 2) ∨ (x ≥ 3) } :=
by
  sorry

end intersection_S_T_eq_l347_347231


namespace find_PC_l347_347525

noncomputable def side_lengths : ℕ × ℕ × ℕ := (10, 8, 7)

noncomputable def similarity_ratios (PC PA : ℝ) := 
  PC / PA = 7 / 10 ∧ 
  PA / (PC + 8) = 7 / 10

theorem find_PC (PC : ℝ) (PA : ℝ) (AB BC CA : ℕ) (similar : similarity_ratios PC PA) :
  AB = 10 ∧ BC = 8 ∧ CA = 7 → 
  PC = 392 / 51 :=
by
  sorry

end find_PC_l347_347525


namespace number_of_triangles_in_decagon_l347_347090

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347090


namespace ways_to_place_balls_in_boxes_l347_347495

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347495


namespace fraction_decomposition_l347_347794

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -1 ∧ x ≠ 2  →
    7 * x - 18 = A * (3 * x + 1) + B * (x - 2))
  ↔ (A = -4 / 7 ∧ B = 61 / 7) :=
by
  sorry

end fraction_decomposition_l347_347794


namespace words_left_to_write_l347_347560

theorem words_left_to_write :
  ∀ (total_words : ℕ) (words_per_line : ℕ) (lines_per_page : ℕ) (pages_written : ℚ),
  total_words = 400 → words_per_line = 10 → lines_per_page = 20 → pages_written = 1.5 →
  (total_words - (pages_written * lines_per_page * words_per_line)) = 100 :=
by
  intros total_words words_per_line lines_per_page pages_written
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end words_left_to_write_l347_347560


namespace ways_to_distribute_balls_l347_347411

theorem ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) 
  (h1 : balls = 5) (h2 : boxes = 3) : 
  (boxes ^ balls) = 243 :=
by
  rw [h1, h2]
  exact pow_succ 3 4
  rw [pow_one, Nat.mul_succ, pow_zero, mul_one, Nat.mul_zero, add_zero]
  exact Nat.zero


end ways_to_distribute_balls_l347_347411


namespace triangles_from_decagon_l347_347094

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347094


namespace balls_in_boxes_l347_347296

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347296


namespace gauss_identity_l347_347849

-- Definitions of the conditions
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (a b c d : ℝ)
def u := (dist A D)^2
def v := (dist B D)^2
def w := (dist C D)^2
def U := (dist B D)^2 + (dist C D)^2 - (dist B C)^2
def V := (dist A D)^2 + (dist C D)^2 - (dist A C)^2
def W := (dist A D)^2 + (dist B D)^2 - (dist A B)^2

-- The theorem statement without proof
theorem gauss_identity : u * U^2 + v * V^2 + w * W^2 = U * V * W + 4 * u * v * w := by
  sorry

end gauss_identity_l347_347849


namespace arithmetic_mean_is_correct_l347_347812

variable (x a : ℝ)
variable (hx : x ≠ 0)

theorem arithmetic_mean_is_correct : 
  (1/2 * ((x + 2 * a) / x - 1 + (x - 3 * a) / x + 1)) = (1 - a / (2 * x)) := 
  sorry

end arithmetic_mean_is_correct_l347_347812


namespace number_of_triangles_in_decagon_l347_347040

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347040


namespace ways_to_put_balls_in_boxes_l347_347486

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347486


namespace regular_decagon_triangle_count_l347_347030

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347030


namespace survey_participants_l347_347533

-- Total percentage for option A and option B in bytes
def percent_A : ℝ := 0.50
def percent_B : ℝ := 0.30

-- Number of participants who chose option A
def participants_A : ℕ := 150

-- Target number of participants who chose option B (to be proved)
def participants_B : ℕ := 90

-- The theorem to prove the number of participants who chose option B
theorem survey_participants :
  (participants_B : ℝ) = participants_A * (percent_B / percent_A) :=
by
  sorry

end survey_participants_l347_347533


namespace minimum_value_l347_347576

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 3)

theorem minimum_value : 
  (1 / (3 * a + 5 * b)) + (1 / (3 * b + 5 * c)) + (1 / (3 * c + 5 * a)) ≥ 9 / 8 :=
by
  sorry

end minimum_value_l347_347576


namespace exists_fibonacci_divisible_by_2014_l347_347563

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Problem Statement
theorem exists_fibonacci_divisible_by_2014 :
  ∃ n ≥ 1, fibonacci n % 2014 = 0 := sorry

end exists_fibonacci_divisible_by_2014_l347_347563


namespace maximum_integer_value_of_fraction_is_12001_l347_347895

open Real

def max_fraction_value_12001 : Prop :=
  ∃ x : ℝ, (1 + 12 / (4 * x^2 + 12 * x + 8) : ℝ) = 12001

theorem maximum_integer_value_of_fraction_is_12001 :
  ∃ x : ℝ, 1 + (12 / (4 * x^2 + 12 * x + 8)) = 12001 :=
by
  -- Here you should provide the proof steps.
  sorry

end maximum_integer_value_of_fraction_is_12001_l347_347895


namespace ways_to_distribute_balls_l347_347454

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347454


namespace number_of_triangles_in_decagon_l347_347084

theorem number_of_triangles_in_decagon : 
  ∀ (v : Finset ℕ), v.card = 10 → (∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (v.powerset.card 3 = 120) :=
by admit

end number_of_triangles_in_decagon_l347_347084


namespace distances_not_all_rational_l347_347975

theorem distances_not_all_rational {r : ℝ} (h_r : r = 2) (P : ℝ × ℝ)
  (hP_on_circle: P.1^2 + P.2^2 = r^2)
  (AP BP CP DP : ℝ) :
  ¬ (rat.rat_of_real AP ∧ rat.rat_of_real BP ∧ rat.rat_of_real CP ∧ rat.rat_of_real DP) ↔
  (AP = dist P (1, 0) ∧ BP = dist P (0, 1) ∧ CP = dist P (-1, 0) ∧ DP = dist P (0, -1)) :=
sorry

end distances_not_all_rational_l347_347975


namespace product_cubes_identity_l347_347775

noncomputable def prod_cubes : ℚ :=
  ∏ n in (finset.range 6).map (finset.nat_emb (4 +·)), (n^3 - 1) / (n^3 + 1)

theorem product_cubes_identity :
  prod_cubes = 728 / 39 :=
sorry

end product_cubes_identity_l347_347775


namespace prob_xi_greater_4_minus_a_l347_347850

noncomputable def ξ : ℝ → ℝ := sorry

axiom xi_normal : ∀ σ : ℝ, ∃ ξ : ℝ → ℝ, ξ ∼ Normal 2 σ^2
axiom prob_xi_greater_a : ∀ a : ℝ, P(ξ > a = 0.3)

theorem prob_xi_greater_4_minus_a (σ a : ℝ) :
  P(ξ > 4 - a) = 0.7 :=
by {
  -- Proof omitted
  sorry
}

end prob_xi_greater_4_minus_a_l347_347850


namespace part1_increasing_part2_range_a_l347_347873

noncomputable def f (x a : ℝ) : ℝ := exp x - (1/2) * (x + a)^2

theorem part1_increasing (a : ℝ) (h1 : (exp 0 - (1/2) * (0 + a)^2) = 1) :
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 a < f x2 a :=
sorry

theorem part2_range_a (a : ℝ) : (∀ x : ℝ, x ≥ 0 → f x a ≥ 0) ↔ (-sqrt 2 ≤ a ∧ a ≤ 2 - log 2) :=
sorry

end part1_increasing_part2_range_a_l347_347873


namespace angle_between_vectors_is_90_degree_l347_347635

noncomputable theory

open_locale big_operators

variables {α : Type*} [inner_product_space ℝ α]

-- Define the vectors a and b as nonzero elements in inner product space
variables (a b : α) (ha : a ≠ 0) (hb : b ≠ 0)

-- Hypothesis based on the given condition
def hypothesis : Prop :=
  ‖a + 2 • b‖^2 = ‖a - 2 • b‖^2

-- The main statement we want to prove
theorem angle_between_vectors_is_90_degree (h : hypothesis a b) : ⟪a, b⟫ = 0 := 
sorry

end angle_between_vectors_is_90_degree_l347_347635


namespace elephant_entry_duration_l347_347688

theorem elephant_entry_duration
  (initial_elephants : ℕ)
  (exodus_duration : ℕ)
  (leaving_rate : ℕ)
  (entering_rate : ℕ)
  (final_elephants : ℕ)
  (h_initial : initial_elephants = 30000)
  (h_exodus_duration : exodus_duration = 4)
  (h_leaving_rate : leaving_rate = 2880)
  (h_entering_rate : entering_rate = 1500)
  (h_final : final_elephants = 28980) :
  (final_elephants - (initial_elephants - (exodus_duration * leaving_rate))) / entering_rate = 7 :=
by
  sorry

end elephant_entry_duration_l347_347688


namespace base_conversion_correct_l347_347167

def convert_base_9_to_10 (n : ℕ) : ℕ :=
  3 * 9^2 + 6 * 9^1 + 1 * 9^0

def convert_base_13_to_10 (n : ℕ) (C : ℕ) : ℕ :=
  4 * 13^2 + C * 13^1 + 5 * 13^0

theorem base_conversion_correct :
  convert_base_9_to_10 361 + convert_base_13_to_10 4 12 = 1135 :=
by
  sorry

end base_conversion_correct_l347_347167


namespace distinct_balls_boxes_l347_347464

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347464


namespace power_calculation_l347_347690

theorem power_calculation : (3^4)^2 = 6561 := by 
  sorry

end power_calculation_l347_347690


namespace symmetric_points_y_axis_l347_347865

theorem symmetric_points_y_axis (a b : ℤ) (hA : (a, 4)) (hB : (-2, b)) (symm : (a = -2) ∧ (b = 4)) : a + b = 2 := by
  sorry

end symmetric_points_y_axis_l347_347865


namespace initial_amount_of_money_l347_347962

-- Define the conditions
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20
def num_friends : ℕ := 2
def amount_left : ℝ := 114.85

-- Define the calculated amount given to friends
def total_given_to_friends : ℝ := given_to_each_friend * num_friends

-- State the theorem to prove the initial amount of money
theorem initial_amount_of_money :
  spent_on_sweets + total_given_to_friends + amount_left = 200.50 :=
by 
  -- proof goes here
  sorry

end initial_amount_of_money_l347_347962


namespace ways_to_distribute_balls_l347_347447

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347447


namespace arrangement_of_SUCCESS_l347_347788

theorem arrangement_of_SUCCESS : 
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  (fact total_letters) / (fact count_S * fact count_C * fact count_U * fact count_E) = 420 := 
by
  let total_letters := 7
  let count_S := 3
  let count_C := 2
  let count_U := 1
  let count_E := 1
  exact sorry

end arrangement_of_SUCCESS_l347_347788


namespace number_of_triangles_in_regular_decagon_l347_347136

-- Define a regular decagon with its properties
def regular_decagon := { vertices : Finset ℕ // vertices.card = 10 }

theorem number_of_triangles_in_regular_decagon (D : regular_decagon) : 
  (D.vertices.card.choose 3) = 120 :=
by sorry

end number_of_triangles_in_regular_decagon_l347_347136


namespace find_x2_minus_y2_l347_347948

def x : ℝ := 2023 ^ 1012 - 2023 ^ (-1012)
def y : ℝ := 2023 ^ 1012 + 2023 ^ (-1012)

theorem find_x2_minus_y2 : x^2 - y^2 = -4 :=
by
  -- The proof goes here, but we place sorry to skip the proof as instructed.
  sorry

end find_x2_minus_y2_l347_347948


namespace balls_into_boxes_l347_347332

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347332


namespace triangles_from_decagon_l347_347093

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347093


namespace ball_in_boxes_l347_347279

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347279


namespace number_of_triangles_in_regular_decagon_l347_347121

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347121


namespace speed_of_faster_train_is_correct_l347_347684

-- Definitions as per the conditions
constant length_train : ℝ := 55
constant speed_slower_train_kmph : ℝ := 36
constant time_to_pass_seconds : ℝ := 36

-- Conversion factor from km/hr to m/s
constant kmph_to_mps : ℝ := 5 / 18

-- Speed of the slower train in m/s
def speed_slower_train_mps : ℝ :=
  speed_slower_train_kmph * kmph_to_mps

-- Distance to be covered for the faster train to pass the slower train
def distance_to_pass : ℝ :=
  2 * length_train

-- Relational speed (difference in speed between faster and slower train)
from mathlib

-- Speed of the faster train in m/s
def speed_faster_train_mps : ℝ :=
  (distance_to_pass / time_to_pass_seconds) + speed_slower_train_mps

-- Conversion of the faster train speed from m/s to km/hr
def speed_faster_train_kmph : ℝ :=
  speed_faster_train_mps / kmph_to_mps

-- Theorem: Speed of the faster train in km/hr
theorem speed_of_faster_train_is_correct : speed_faster_train_kmph = 47 :=
by
  sorry

end speed_of_faster_train_is_correct_l347_347684


namespace horner_value_at_5_l347_347230

def f (x : ℝ) : ℝ := 2 * x ^ 5 - 5 * x ^ 4 - 4 * x ^ 3 + 3 * x ^ 2 - 6 * x + 7

def horner_method (x : ℝ) : ℝ :=
  let v := 2 in
  let v₁ := v * x - 5 in
  let v₂ := v₁ * x - 4 in
  let v₃ := v₂ * x + 3 in
  v₃

theorem horner_value_at_5 : horner_method 5 = 108 := by
  sorry

end horner_value_at_5_l347_347230


namespace total_points_team_l347_347526

def T : ℕ := 4
def J : ℕ := 2 * T + 6
def S : ℕ := J / 2
def R : ℕ := T + J - 3
def A : ℕ := S + R + 4

theorem total_points_team : T + J + S + R + A = 66 := by
  sorry

end total_points_team_l347_347526


namespace triangle_is_isosceles_l347_347610

-- Define the triangle and its median
variables {A B C M_3 P: Type*}
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space P] [metric_space M_3]

-- Conditions
variable is_median_CM3 : is_median CM_3 A B C
variable P_on_CM3 : lies_on P CM_3
variable parallel_MN_CA : ∀ (M N: Type*), parallel_segment M N P CA
variable parallel_KL_CB : ∀ (K L: Type*), parallel_segment K L P CB
variable segments_equal : equal_lengths_segment MN KL

-- Prove that triangle ABC is isosceles
theorem triangle_is_isosceles 
  (A B C M_3 P : Type*)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space P] [metric_space M_3]
  (is_median_CM3 : is_median CM_3 A B C)
  (P_on_CM3 : lies_on P CM_3)
  (parallel_MN_CA : ∀ (M N: Type*), parallel_segment M N P CA)
  (parallel_KL_CB : ∀ (K L: Type*), parallel_segment K L P CB)
  (segments_equal : equal_lengths_segment MN KL)
  : is_isosceles_triangle A B C := 
sorry

end triangle_is_isosceles_l347_347610


namespace boxes_with_no_items_l347_347773

-- Definitions of each condition as given in the problem
def total_boxes : Nat := 15
def pencil_boxes : Nat := 8
def pen_boxes : Nat := 5
def marker_boxes : Nat := 3
def pen_pencil_boxes : Nat := 4
def all_three_boxes : Nat := 1

-- The theorem to prove
theorem boxes_with_no_items : 
     (total_boxes - ((pen_pencil_boxes - all_three_boxes)
                     + (pencil_boxes - pen_pencil_boxes - all_three_boxes)
                     + (pen_boxes - pen_pencil_boxes - all_three_boxes)
                     + (marker_boxes - all_three_boxes)
                     + all_three_boxes)) = 5 := 
by 
  -- This is where the proof would go, but we'll use sorry to indicate it's skipped.
  sorry

end boxes_with_no_items_l347_347773


namespace ways_to_distribute_balls_l347_347448

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347448


namespace regular_decagon_triangle_count_l347_347034

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347034


namespace count_odd_numbers_with_3_between_200_and_499_l347_347249

def is_odd (n : ℕ) : Prop := n % 2 = 1

def contains_digit_3 (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.any (λ digit, digit = 3)

def in_range_200_499 (n : ℕ) : Prop := 200 ≤ n ∧ n < 500

theorem count_odd_numbers_with_3_between_200_and_499 :
  (finset.filter (λ n, is_odd n ∧ contains_digit_3 n ∧ in_range_200_499 n) 
                 (finset.range 500)).card = 87 :=
by sorry

end count_odd_numbers_with_3_between_200_and_499_l347_347249


namespace solve_inequality_l347_347984

theorem solve_inequality (x : ℝ) :
  x^2 - 3 * Real.sqrt (x^2 + 3) ≤ 1 ↔ -Real.sqrt 13 ≤ x ∧ x ≤ Real.sqrt 13 := by
sorry

end solve_inequality_l347_347984


namespace distinct_balls_boxes_l347_347473

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347473


namespace cannot_determine_right_triangle_l347_347755

-- Definitions of conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (a b c : ℝ) : Prop := a/b = 5/12 ∧ b/c = 12/13
def condition_C (a b c : ℝ) : Prop := a^2 = (b + c) * (b - c)
def condition_D (A B C : ℝ) : Prop := A/B = 3/4 ∧ B/C = 4/5

-- The proof problem
theorem cannot_determine_right_triangle (a b c A B C : ℝ)
  (hD : condition_D A B C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end cannot_determine_right_triangle_l347_347755


namespace number_of_triangles_in_regular_decagon_l347_347124

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347124


namespace probability_less_than_9_l347_347656

theorem probability_less_than_9 :
  let P_10 := 0.24;
      P_9 := 0.28;
      P_8 := 0.19 in
  1 - (P_10 + P_9) = 0.29 :=
by 
  intros P_10 P_9 P_8
  sorry

end probability_less_than_9_l347_347656


namespace ball_in_boxes_l347_347274

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347274


namespace game_no_loser_l347_347972

theorem game_no_loser (x : ℕ) (h_start : x = 2017) :
  ∀ y, (y = x ∨ ∀ n, (n = 2 * y ∨ n = y - 1000) → (n > 1000 ∧ n < 4000)) →
       (y > 1000 ∧ y < 4000) :=
sorry

end game_no_loser_l347_347972


namespace number_of_triangles_in_decagon_l347_347015

theorem number_of_triangles_in_decagon (h : ∀ (a b c : ℕ), a ≠ b → b ≠ c → a ≠ c) : 
  ∃ n : ℕ, n = 120 ∧ combinational 10 3 = n := by
  sorry

end number_of_triangles_in_decagon_l347_347015


namespace length_RS_l347_347542

variables (FD DR FR FS : ℝ)
variable (θ : ℝ) 
variable (RS : ℝ)

axiom angle_equality : ∠RFS = ∠FDR
axiom length_FD : FD = 5
axiom length_DR : DR = 7
axiom length_FR : FR = 6
axiom length_FS : FS = 9

theorem length_RS :
  RS = 7.59 :=
by
  have h1 : ∠RFS = ∠FDR := angle_equality
  have h2 : FD = 5 := length_FD
  have h3 : DR = 7 := length_DR
  have h4 : FR = 6 := length_FR
  have h5 : FS = 9 := length_FS
  have h6 : cos (∠FDR) = 19 / 35 := 
    calc cos (∠FDR) = (FD^2 + DR^2 - FR^2) / (2 * FD * DR) : by sorry
                      ... = (5^2 + 7^2 - 6^2) / (2 * 5 * 7) : by rw [h2, h3, h4]
                      ... = (25 + 49 - 36) / (2 * 5 * 7) : by norm_num
                      ... = 38 / 70 : by norm_num
                      ... = 19 / 35 : by norm_num
  have h7 : RS^2 = FR^2 + FS^2 - 2 * FR * FS * (19 / 35) := 
    calc RS^2 = FR^2 + FS^2 - 2 * FR * FS * cos (∠RFS) : by sorry
              ... = FR^2 + FS^2 - 2 * FR * FS * (19 / 35) : by rw [← h1, h6]
              ... = 6^2 + 9^2 - 2 * 6 * 9 * (19 / 35) : by rw [h4, h5]
              ... = 36 + 81 - 2 * 6 * 9 * (19 / 35) : by norm_num
              ... = 36 + 81 - 1224 / 35 : by norm_num
              ... = 117 - 1224 / 35 : by norm_num
              ... = 4095 / 35 - 2043 / 35 : by norm_num
              ... = 2043 / 35 : by norm_num
  show RS = 7.59 := by sorry

end length_RS_l347_347542


namespace barnyard_owls_calc_l347_347611

theorem barnyard_owls_calc (hoots_per_owl : ℕ) (total_hoots : ℕ) (hoots_heard : ℕ) :
  hoots_heard = total_hoots - 5 →
  hoots_per_owl = 5 →
  total_hoots = 20 →
  hoots_heard / hoots_per_owl = 3 :=
by
  intros h1 h2 h3
  rw [h3, h2] at h1
  have h : hoots_heard = 15 := by
    rw h1;
    norm_num
  rw [h]
  norm_num

end barnyard_owls_calc_l347_347611


namespace proposition_correctness_count_l347_347619

noncomputable def proposition1 (P : Point) (prism : RectangularPrism) : Prop :=
  ∃ P, ∀ v ∈ vertices prism, distance P v = distance P (next_vertex v prism)

noncomputable def proposition2 (Q : Point) (prism : RectangularPrism) : Prop :=
  ∃ Q, ∀ e ∈ edges prism, distance Q e = distance Q (next_edge e prism)

noncomputable def proposition3 (R : Point) (prism : RectangularPrism) : Prop :=
  ∃ R, ∀ f ∈ faces prism, distance R f = distance R (next_face f prism)

theorem proposition_correctness_count (prism : RectangularPrism) : 
  (proposition1 = True ∧ proposition2 = True ∧ proposition3 = True) ↔ 
    (proposition2 => True) ∧ (proposition3 => True) ∧ (proposition1 = prism.is_cube) :=
by
  sorry

end proposition_correctness_count_l347_347619


namespace line_tangent_to_parabola_k_value_l347_347157

theorem line_tangent_to_parabola_k_value :
  ∃ k : ℝ, (∀ x y : ℝ, 4 * x + 7 * y + k = 0 → y ^ 2 = 16 * x) → k = 49 :=
by
  -- definitions of the line and parabola
  let line := (x y : ℝ) → 4 * x + 7 * y + k = 0
  let parabola := (y : ℝ) → y ^ 2 = 16 * (-7*y - k)/4
  -- proof to be filled
  sorry

end line_tangent_to_parabola_k_value_l347_347157


namespace geometric_series_sum_l347_347160

theorem geometric_series_sum {a r : ℚ} (n : ℕ) (h_a : a = 3/4) (h_r : r = 3/4) (h_n : n = 8) : 
       a * (1 - r^n) / (1 - r) = 176925 / 65536 :=
by
  -- Utilizing the provided conditions
  have h_a := h_a
  have h_r := h_r
  have h_n := h_n
  -- Proving the theorem using sorry as a placeholder for the detailed steps
  sorry

end geometric_series_sum_l347_347160


namespace ways_to_place_balls_in_boxes_l347_347496

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347496


namespace trig_identity_l347_347191

-- Define the given condition
def tan_half (α : ℝ) : Prop := Real.tan (α / 2) = 2

-- The main statement we need to prove
theorem trig_identity (α : ℝ) (h : tan_half α) : (1 + Real.cos α) / (Real.sin α) = 1 / 2 :=
  by
  sorry

end trig_identity_l347_347191


namespace ways_to_put_balls_in_boxes_l347_347487

theorem ways_to_put_balls_in_boxes : 
  ∀ (balls boxes : ℕ), 
  balls = 5 → boxes = 3 → (boxes ^ balls) = 243 :=
begin
  intros balls boxes h_balls h_boxes,
  rw [h_balls, h_boxes],
  norm_num,
end

end ways_to_put_balls_in_boxes_l347_347487


namespace faucets_fill_time_l347_347831

theorem faucets_fill_time (rate_same : ∀ f g : ℕ, f > 0 → g > 0 → (rate f = rate g)) 
  (five_faucets_180_gallons_480_seconds : rate 5 * 480 = 180) : 
  fill_time 10 90 = 120 :=
by
  sorry

end faucets_fill_time_l347_347831


namespace abs_frac_sqrt_15_div_3_l347_347512

theorem abs_frac_sqrt_15_div_3 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 8 * a * b) : 
  abs ((a + b) / (a - b)) = sqrt 15 / 3 := 
by sorry

end abs_frac_sqrt_15_div_3_l347_347512


namespace range_f_compare_sizes_final_comparison_l347_347225

noncomputable def f (x : ℝ) := |2 * x - 1| + |x + 1|

theorem range_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = {y : ℝ | y ∈ Set.Ici (3 / 2)} :=
sorry

theorem compare_sizes (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
sorry

theorem final_comparison (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
by
  exact compare_sizes a ha

end range_f_compare_sizes_final_comparison_l347_347225


namespace ways_to_distribute_balls_l347_347451

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347451


namespace find_cos_alpha_l347_347845

theorem find_cos_alpha 
  (α : ℝ) 
  (h₁ : Real.tan (π - α) = 3/4) 
  (h₂ : α ∈ Set.Ioo (π/2) π) 
: Real.cos α = -4/5 :=
sorry

end find_cos_alpha_l347_347845


namespace triangles_from_decagon_l347_347095

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347095


namespace projection_of_projection_magnitude_l347_347573

noncomputable def projection (u v : ℝ^3) : ℝ^3 := (u ⬝ v / u ⬝ u) • u

theorem projection_of_projection_magnitude
  (v w : ℝ^3)
  (p := projection v w)
  (q := projection p v)
  (h : ∥p∥ / ∥v∥ = 3 / 4) :
  ∥q∥ / ∥v∥ = 3 / 4 := 
by
  sorry

end projection_of_projection_magnitude_l347_347573


namespace cans_per_dollars_l347_347988

variables (S Q D : ℕ)

theorem cans_per_dollars (S Q D : ℕ) : 
  let quarters_per_dollar := 5 in
  let total_quarters := D * quarters_per_dollar in
  (total_quarters * S) / Q = (5 * D * S) / Q := by
  let quarters_per_dollar := 5 in
  let total_quarters := D * quarters_per_dollar in
  have h1 : total_quarters = 5 * D := by rfl
  have h2 : (total_quarters * S) = 5 * D * S := by rw [h1]
  rw [h2]
  rfl

end cans_per_dollars_l347_347988


namespace no_polynomial_transformation_l347_347553

-- Define the problem conditions: initial and target sequences
def initial_seq : List ℤ := [-3, -1, 1, 3]
def target_seq : List ℤ := [-3, -1, -3, 3]

-- State the main theorem to be proved
theorem no_polynomial_transformation :
  ¬ (∃ (P : ℤ → ℤ), ∀ x ∈ initial_seq, P x ∈ target_seq) :=
  sorry

end no_polynomial_transformation_l347_347553


namespace possible_degrees_of_remainder_l347_347701

theorem possible_degrees_of_remainder (f g : Polynomial ℝ) (h : degree g = 7) :
  ∃ d, degree (f % g) = d ∧ d ∈ set.univ ∧ d < 7 :=
by
  sorry

end possible_degrees_of_remainder_l347_347701


namespace put_balls_in_boxes_l347_347305

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347305


namespace log2_3_value_l347_347211

variables (a b log2 log3 : ℝ)

-- Define the conditions
axiom h1 : a = log2 + log3
axiom h2 : b = 1 + log2

-- Define the logarithmic requirement to be proved
theorem log2_3_value : log2 * log3 = (a - b + 1) / (b - 1) :=
sorry

end log2_3_value_l347_347211


namespace balls_into_boxes_l347_347314

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347314


namespace ways_to_distribute_balls_l347_347457

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l347_347457


namespace range_of_m_l347_347190

theorem range_of_m (m : ℝ) : 
  (¬ (∀ x : ℝ, x^2 + m * x + 1 = 0 → x > 0) → m ≥ -2) :=
by
  sorry

end range_of_m_l347_347190


namespace investment_in_stocks_l347_347770

theorem investment_in_stocks (T b s : ℝ) (h1 : T = 200000) (h2 : s = 5 * b) (h3 : T = b + s) :
  s = 166666.65 :=
by sorry

end investment_in_stocks_l347_347770


namespace find_x1_plus_x2_l347_347189

def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem find_x1_plus_x2 (x1 x2 : ℝ) (hneq : x1 ≠ x2) (h1 : f x1 = 101) (h2 : f x2 = 101) : x1 + x2 = 2 := 
by 
  -- proof or sorry can be used; let's assume we use sorry to skip proof
  sorry

end find_x1_plus_x2_l347_347189


namespace quadrant_of_angle_l347_347210

theorem quadrant_of_angle (θ : ℝ) (h : sin θ * cos θ < 0) : 
  (π / 2 < θ ∧ θ < π) ∨ (3 * π / 2 < θ ∧ θ < 2 * π) :=
sorry

end quadrant_of_angle_l347_347210


namespace find_b_minus_d_squared_l347_347897

theorem find_b_minus_d_squared (a b c d : ℝ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 3) :
  (b - d) ^ 2 = 25 :=
sorry

end find_b_minus_d_squared_l347_347897


namespace vector_addition_l347_347237

variable (a : Vector)

theorem vector_addition (a : Vector) : a + (2:ℝ) • a = (3:ℝ) • a := 
by
  -- The proof goes here
  sorry

end vector_addition_l347_347237


namespace ways_to_place_balls_in_boxes_l347_347498

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347498


namespace ways_to_place_balls_in_boxes_l347_347501

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347501


namespace union_sets_l347_347567

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- State the theorem
theorem union_sets : A ∪ B = { x | -1 < x ∧ x ≤ 4 } := 
by
   sorry

end union_sets_l347_347567


namespace increasing_sequence_implies_lambda_gt_neg3_l347_347852

variable (λ : ℝ)

def a_n (n : ℕ) : ℝ := n^2 + λ * n

theorem increasing_sequence_implies_lambda_gt_neg3 :
  (∀ n : ℕ, a_n λ (n + 1) > a_n λ n) → λ > -3 := by
  sorry

end increasing_sequence_implies_lambda_gt_neg3_l347_347852


namespace fraction_zero_implies_x_zero_l347_347907

theorem fraction_zero_implies_x_zero (x : ℝ) (h : (x^2 - x) / (x - 1) = 0) (h₁ : x ≠ 1) : x = 0 := by
  sorry

end fraction_zero_implies_x_zero_l347_347907


namespace vector_magnitude_and_projection_l347_347241

variables (a b : Vector ℝ 2)
variables [unit_vector a] [unit_vector b] -- Assumes a and b are unit vectors
variables (h : a ⬝ b = 1 / 2) -- Inner product of a and b is 1/2

theorem vector_magnitude_and_projection:
  |a + b| = sqrt 3 ∧ (proj b a) = (1 / 2) • b := by
  sorry

end vector_magnitude_and_projection_l347_347241


namespace distinct_balls_boxes_l347_347465

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347465


namespace regular_decagon_triangle_count_l347_347037

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347037


namespace sum_of_squares_l347_347673

def b1 : ℚ := 10 / 32
def b2 : ℚ := 0
def b3 : ℚ := -5 / 32
def b4 : ℚ := 0
def b5 : ℚ := 1 / 32

theorem sum_of_squares : b1^2 + b2^2 + b3^2 + b4^2 + b5^2 = 63 / 512 :=
by
  sorry

end sum_of_squares_l347_347673


namespace remainder_div_7_l347_347713

theorem remainder_div_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k < 39) : k % 7 = 3 :=
sorry

end remainder_div_7_l347_347713


namespace mutually_exclusive_probability_zero_l347_347511

theorem mutually_exclusive_probability_zero {A B : Prop} (p1 p2 : ℝ) 
  (hA : 0 ≤ p1 ∧ p1 ≤ 1) 
  (hB : 0 ≤ p2 ∧ p2 ≤ 1) 
  (hAB : A ∧ B → False) : 
  (A ∧ B) = False :=
by
  sorry

end mutually_exclusive_probability_zero_l347_347511


namespace f_at_8_l347_347193

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem f_at_8 : f 8 = -1 := 
by
-- The following will be filled with the proof, hence sorry for now.
sorry

end f_at_8_l347_347193


namespace locus_of_incenters_l347_347723

open EuclideanGeometry

variable {A B C C₀ C₁ C₂ : Point}
variable {l : Line}
variable (ℓ : Line) {τ : Line}

-- Definitions of conditions
def feet_of_median_bisector_altitude_coincide_with_C₀_C₁_C₂ :
  Feet_of_median_bisector_altitude C l C₀ C₁ C₂ :=
sorry

theorem locus_of_incenters (h1 : A ∈ l) (h2 : B ∈ l) 
  (h3 : C₀ ∈ l) (h4 : C₁ ∈ l) (h5 : C₂ ∈ l) 
  (h6 : feet_of_median_bisector_altitude_coincide_with_C₀_C₁_C₂) :
  locus_of_incenter = τ → τ = perpendicular_line l through fixed_point X :=
sorry

end locus_of_incenters_l347_347723


namespace probability_of_drawing_white_ball_l347_347536

/--
In an opaque bag, there are 5 red balls, 2 white balls, and 3 black balls, all of which are identical except for their colors.
If one ball is randomly drawn from the bag, the probability of drawing a white ball is 1/5.
-/
theorem probability_of_drawing_white_ball :
  let total_balls := 5 + 2 + 3 in
  let white_balls := 2 in
  let probability := white_balls / total_balls.toRat in
  probability = 1 / 5 := 
by
  sorry

end probability_of_drawing_white_ball_l347_347536


namespace probability_y_intercept_gt_1_l347_347218

theorem probability_y_intercept_gt_1 (b : ℝ) (H : b ∈ set.Icc (-3 : ℝ) 2) : 
  (measure_theory.measure_univ {b : ℝ | b > 1} ∩ set.Icc (-3 : ℝ) 2)
     / (measure_theory.volume (set.Icc (-3 : ℝ) 2)) = 1 / 5 :=
by sorry

end probability_y_intercept_gt_1_l347_347218


namespace units_digit_of_sum_of_sequence_proof_l347_347696

def units_digit_of_sum_of_sequence : ℕ :=
  let sequence := [2!, 3!, 4!, 5!, 6!, 7!, 8!, 9!, 10!].zipWith (· + ·) (list.range 2 11)
  let units_digits := sequence.map (λ x => x % 10)
  (units_digits.sum % 10)

theorem units_digit_of_sum_of_sequence_proof : units_digit_of_sum_of_sequence = 1 :=
by
  sorry

end units_digit_of_sum_of_sequence_proof_l347_347696


namespace inequality_solution_l347_347986

theorem inequality_solution {x : ℝ} :
  ((x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x)) ↔
  ((x - 2) * (x - 3) * (x - 4) / ((x - 1) * (x - 5) * (x - 6)) > 0) := sorry

end inequality_solution_l347_347986


namespace math_proof_problem_l347_347827

theorem math_proof_problem :
  2018^2019^2020 % 11 = 5 :=
  by
    have fermat_lt : ∀ a p : ℕ, Nat.Prime p ∧ (a % p ≠ 0) → a ^ (p - 1) % p = 1 :=
      sorry -- Fermat's Little theorem
    have cond1 : 2018 % 11 = 5 := by norm_num
    have cond2 : 2019 % 10 = 9 := by norm_num  -- Since (-1 ≡ 9 mod 10)
    have exp_simpl : 2019^2020 % 10 = 1 := 
      by
        have p1 : 2019 % 10 = 9  := cond2
        have p2 : 9^2020 % 10 = 1 :=
          by
            sorry -- Use cyclicity of powers mod 10: (-1 for odd exponent, 1 for even)
        exact p2
    have base_simpl : 2018 ^ 2019 ^ 2020 % 11 = 5 :=
      by 
        have s1 : 2018 % 11 = 5 := cond1
        have e1 : 2019^2020 % 10 = 1 := exp_simpl
        have final_result : 5 ^ 1 % 11 = 5 := by norm_num
        exact final_result
    exact base_simpl

end math_proof_problem_l347_347827


namespace balls_into_boxes_l347_347343

theorem balls_into_boxes :
  (number_of_ways_to_put_balls_in_boxes 5 3) = 243 := by
  sorry

noncomputable def number_of_ways_to_put_balls_in_boxes (balls : ℕ) (boxes : ℕ) : ℕ :=
  boxes ^ balls

end balls_into_boxes_l347_347343


namespace num_triangles_from_decagon_l347_347006

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l347_347006


namespace ball_box_distribution_l347_347434

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347434


namespace regular_decagon_triangle_count_l347_347028

theorem regular_decagon_triangle_count (n : ℕ) (hn : n = 10) :
  choose n 3 = 120 :=
by
  sorry

end regular_decagon_triangle_count_l347_347028


namespace line_intersects_circle_l347_347202

axiom circle_eq (x y : ℝ) : x^2 + y^2 - 4 * x = 0

/-- Center of the circle -/
def center_c : ℝ × ℝ := (2, 0)

/-- Radius of the circle -/
def radius_c : ℝ := 2

/-- Point P on the line l -/
def point_p : ℝ × ℝ := (3, 0)

/-- Distance between point P and the center of the circle -/
def distance_p_to_center_c : ℝ := real.sqrt ((3 - 2)^2 + (0 - 0)^2)

theorem line_intersects_circle :
  distance_p_to_center_c < radius_c := by
  sorry

end line_intersects_circle_l347_347202


namespace derivative_of_cosine_over_x_l347_347998

noncomputable def cosine_over_x_derivative : Prop :=
  ∀ (x : ℝ), x ≠ 0 → deriv (λ x, (cos x) / x) x = -((x * sin x + cos x) / x^2)

-- Statement of the problem
theorem derivative_of_cosine_over_x (x : ℝ) (hx : x ≠ 0) : deriv (λ x, (cos x) / x) x = -((x * sin x + cos x) / x^2) :=
by
  sorry

end derivative_of_cosine_over_x_l347_347998


namespace distinguish_ball_box_ways_l347_347360

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347360


namespace average_temperature_week_l347_347995

theorem average_temperature_week :
  let d1 := 40
  let d2 := 40
  let d3 := 40
  let d4 := 80
  let d5 := 80
  let remaining_days_total := 140
  d1 + d2 + d3 + d4 + d5 + remaining_days_total = 420 ∧ 420 / 7 = 60 :=
by sorry

end average_temperature_week_l347_347995


namespace max_length_line_segment_l347_347776

noncomputable def max_line_segment_len (AB BC CA : ℝ) : ℝ :=
let s := (AB + BC + CA) / 2 in
let h := (2 * Real.sqrt (s * (s - AB) * (s - BC) * (s - CA))) / BC in
Real.sqrt (h^2 + CA^2)

theorem max_length_line_segment :
  max_line_segment_len 5 7 8 =  √((2  * Real.sqrt(10 * 5 * 3 * 2))/7)^2 + 8^2 := sorry

end max_length_line_segment_l347_347776


namespace triangle_collinear_l347_347937

variable {P : Type*} [EuclideanGeometry P]

open EuclideanGeometry

theorem triangle_collinear (A B C B1 C1 A1 : P) : 
  is_triangle A B C →
  interior_angle A B C (∡ A B C) = 60 →
  is_foot_of_bisector B1 B A C → 
  is_foot_of_bisector C1 C A B →
  A1 = sym_point A (line_join B1 C1) →
  collinear {A1, B, C} :=
by
  sorry

end triangle_collinear_l347_347937


namespace simplify_expression_l347_347579

variables {a b c x : ℝ}
hypothesis h₁ : a ≠ b
hypothesis h₂ : b ≠ c
hypothesis h₃ : c ≠ a

def p (x : ℝ) : ℝ :=
  (x + a)^4 / ((a - b) * (a - c)) + 
  (x + b)^4 / ((b - a) * (b - c)) + 
  (x + c)^4 / ((c - a) * (c - b))

theorem simplify_expression : p x = a + b + c + 3 * x^2 :=
by sorry

end simplify_expression_l347_347579


namespace words_left_to_write_l347_347559

theorem words_left_to_write :
  ∀ (total_words : ℕ) (words_per_line : ℕ) (lines_per_page : ℕ) (pages_written : ℚ),
  total_words = 400 → words_per_line = 10 → lines_per_page = 20 → pages_written = 1.5 →
  (total_words - (pages_written * lines_per_page * words_per_line)) = 100 :=
by
  intros total_words words_per_line lines_per_page pages_written
  assume h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end words_left_to_write_l347_347559


namespace lambda_mu_constant_l347_347879

-- Define the parabola and the conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the ellipse and the conditions
def ellipse (x y : ℝ) : Prop := (y^2) / 2 + x^2 = 1

-- Define the conditions on eccentricity and passing through a specific point.
def eccentricity (a b e : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ e = real.sqrt 2 / 2 ∧ b = 1 ∧ a^2 = 2

-- Define the linearity conditions and proof goal.
noncomputable def lambda_mu_sum_1 (x1 x2 : ℝ) : Prop :=
  ∀ λ μ : ℝ, 
  (λ = x1 / (1 - x1)) ∧ (μ = x2 / (1 - x2)) → 
  λ + μ = -1

-- Main theorem statement in Lean
theorem lambda_mu_constant :
  ∀ (x1 x2 : ℝ),
  (x1 * x2 = 1) → 
  lambda_mu_sum_1 x1 x2 :=
begin
  intros x1 x2 h,
  sorry -- Proof needs to be filled in.
end

end lambda_mu_constant_l347_347879


namespace put_balls_in_boxes_l347_347309

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347309


namespace triangles_from_decagon_l347_347104

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347104


namespace george_run_speed_last_half_mile_l347_347184

theorem george_run_speed_last_half_mile :
  ∀ (distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance)
    (today_total_time normal_total_time remaining_time : ℝ),
    distance_school = 2 →
    normal_speed = 4 →
    first_segment_distance = 3 / 4 →
    first_segment_speed = 3 →
    second_segment_distance = 3 / 4 →
    second_segment_speed = 4 →
    remaining_distance = 1 / 2 →
    normal_total_time = distance_school / normal_speed →
    today_total_time = (first_segment_distance / first_segment_speed) + (second_segment_distance / second_segment_speed) →
    normal_total_time = today_total_time + remaining_time →
    (remaining_distance / remaining_time) = 8 :=
by
  intros distance_school normal_speed first_segment_distance first_segment_speed second_segment_distance second_segment_speed remaining_distance today_total_time normal_total_time remaining_time h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end george_run_speed_last_half_mile_l347_347184


namespace ways_to_place_balls_in_boxes_l347_347504

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347504


namespace min_value_of_expression_l347_347861

theorem min_value_of_expression
  (x y : ℝ) 
  (h : x + y = 1) : 
  ∃ (m : ℝ), m = 2 * x^2 + 3 * y^2 ∧ m = 6 / 5 := 
sorry

end min_value_of_expression_l347_347861


namespace distinguish_ball_box_ways_l347_347346

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347346


namespace range_of_a_l347_347645

noncomputable def f (a x : ℝ) := log (1 / 2) (3 * x^2 - a * x + 5)

theorem range_of_a (a : ℝ) :
  (∀ x y ∈ set.Ici (-1 : ℝ), x < y → f a x > f a y) →
  (-8 ≤ a ∧ a ≤ -6) :=
by
  sorry

end range_of_a_l347_347645


namespace triangles_from_decagon_l347_347100

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347100


namespace distinct_balls_boxes_l347_347466

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347466


namespace isosceles_right_triangle_area_l347_347990

theorem isosceles_right_triangle_area (alt : ℝ) (h_alt : alt = real.sqrt 8) :
  let hyp := alt * real.sqrt 2 in
  let area := (1/2) * alt * alt in
  area = 4 :=
by
  sorry

end isosceles_right_triangle_area_l347_347990


namespace midpoint_example_l347_347815

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_example :
  midpoint (2, 9) (8, 3) = (5, 6) :=
by
  sorry

end midpoint_example_l347_347815


namespace triangles_from_decagon_l347_347099

theorem triangles_from_decagon : ∃ n : ℕ, n = 120 ∧ (n = nat.choose 10 3) := 
by {
  sorry 
}

end triangles_from_decagon_l347_347099


namespace distinct_balls_boxes_l347_347390

theorem distinct_balls_boxes :
  (3^5 = 243) :=
by sorry

end distinct_balls_boxes_l347_347390


namespace distinct_balls_boxes_l347_347462

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l347_347462


namespace tangent_line_equation_l347_347518

noncomputable def f (x : ℝ) : ℝ := - x ^ 3 + x ^ 2

theorem tangent_line_equation :
  let f' (x : ℝ) : ℝ := -3 * x ^ 2 + 2 * x,
      f_1 := f 1,
      f'_1 := f' 1
  in (f_1 = 0 ∧ f'_1 = -1) → ∀ x : ℝ, (y = -x + 1) :=
by
  intros
  sorry

end tangent_line_equation_l347_347518


namespace tangent_line_parabola_k_l347_347158

theorem tangent_line_parabola_k :
  ∃ (k : ℝ), (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → (28 ^ 2 = 4 * 1 * 4 * k)) → k = 49 :=
by
  sorry

end tangent_line_parabola_k_l347_347158


namespace put_balls_in_boxes_l347_347306

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l347_347306


namespace ball_in_boxes_l347_347267

theorem ball_in_boxes (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  k^n = 243 :=
by
  sorry

end ball_in_boxes_l347_347267


namespace num_triangles_in_decagon_l347_347078

theorem num_triangles_in_decagon : 
  let V := 10 -- number of vertices 
  in (V.choose 3) = 120 :=
by
  have V := 10
  exact Nat.choose_eq (10, 3)

end num_triangles_in_decagon_l347_347078


namespace ball_box_distribution_l347_347441

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l347_347441


namespace num_triangles_from_decagon_l347_347115

-- Given condition: a decagon has 10 vertices.
def vertices_of_decagon : ℕ := 10
def k : ℕ := 3

-- We are to prove the number of triangles (combinations of 3 vertices from 10) is 120.
theorem num_triangles_from_decagon : Nat.choose vertices_of_decagon k = 120 := by
  sorry

end num_triangles_from_decagon_l347_347115


namespace paint_for_small_balls_l347_347613

noncomputable def paint_required_for_large_ball := 2.4
def num_small_balls := 64

-- Definitions
def radius_large_ball (R : ℝ) := R
def surface_area_large_ball (R : ℝ) := 4 * Real.pi * R^2

lemma paint_for_large_ball (R : ℝ) : paint_required_for_large_ball = 2.4 := sorry

-- Volume of spheres
def volume_sphere (r : ℝ) := (4/3) * Real.pi * r^3

-- Given we melt the large ball into 64 small balls
def radius_small_ball (r R : ℝ) := r = R / 4

lemma volume_conservation (R r : ℝ) (h : radius_small_ball r R) :
  volume_sphere R = num_small_balls * volume_sphere r := sorry

-- Small ball surface areas
def surface_area_small_balls (r : ℝ) := num_small_balls * 4 * Real.pi * r^2

-- Total surface area using radius relation r = R / 4
def total_surface_area_small_balls (R : ℝ) := 4 * surface_area_large_ball R

-- Theorem statement
theorem paint_for_small_balls : ∀ (R : ℝ) (r : ℝ) (h : r = R / 4),
  4 * paint_required_for_large_ball = 9.6 := sorry

end paint_for_small_balls_l347_347613


namespace vector_magnitude_and_projection_l347_347242

variables (a b : Vector ℝ 2)
variables [unit_vector a] [unit_vector b] -- Assumes a and b are unit vectors
variables (h : a ⬝ b = 1 / 2) -- Inner product of a and b is 1/2

theorem vector_magnitude_and_projection:
  |a + b| = sqrt 3 ∧ (proj b a) = (1 / 2) • b := by
  sorry

end vector_magnitude_and_projection_l347_347242


namespace balls_into_boxes_l347_347323

theorem balls_into_boxes :
  let balls := 5
  let boxes := 3
  (boxes ^ balls) = 243 := by
  let balls := 5
  let boxes := 3
  calc
    (boxes ^ balls) = (3 ^ 5) : by rfl
    ... = 243 : by norm_num

end balls_into_boxes_l347_347323


namespace ways_to_place_balls_in_boxes_l347_347490

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347490


namespace num_solutions_1973_l347_347889

theorem num_solutions_1973 :
  ∃ (S : finset (ℤ × ℤ × ℤ)), 
  ((∀ (x y z : ℤ), (x, y, z) ∈ S ↔ 15 * x + 6 * y + 10 * z = 1973 ∧ x ≥ 13 ∧ y ≥ -4 ∧ z > -6) ∧
  S.card = 1953) := by
  sorry

end num_solutions_1973_l347_347889


namespace systematic_sampling_remove_l347_347839

theorem systematic_sampling_remove (total_people : ℕ) (sample_size : ℕ) (remove_count : ℕ): 
  total_people = 162 → sample_size = 16 → remove_count = 2 → 
  (total_people - 1) % sample_size = sample_size - 1 :=
by
  sorry

end systematic_sampling_remove_l347_347839


namespace balls_in_boxes_l347_347286

open Nat

theorem balls_in_boxes : (3 ^ 5 = 243) :=
by
  sorry

end balls_in_boxes_l347_347286


namespace union_A_B_union_complement_A_B_l347_347234

open Set

-- Definitions for sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {3, 5}

-- Statement 1: Prove that A ∪ B = {1, 3, 5, 7}
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by
  sorry

-- Definition for complement of A in U
def complement_A_U : Set ℕ := {x ∈ U | x ∉ A}

-- Statement 2: Prove that (complement of A in U) ∪ B = {2, 3, 4, 5, 6}
theorem union_complement_A_B : complement_A_U ∪ B = {2, 3, 4, 5, 6} := by
  sorry

end union_A_B_union_complement_A_B_l347_347234


namespace cannot_determine_right_triangle_l347_347758

theorem cannot_determine_right_triangle (A B C : Type) (angle_A angle_B angle_C : A) (a b c : B) 
  (h1 : angle_A = angle_B + angle_C)
  (h2 : a / b = 5 / 12 ∧ b / c = 12 / 13)
  (h3 : a ^ 2 = (b + c) * (b - c)):
  ¬ (angle_A / angle_B = 3 / 4 ∧ angle_B / angle_C = 4 / 5) :=
sorry

end cannot_determine_right_triangle_l347_347758


namespace distinguish_ball_box_ways_l347_347361

theorem distinguish_ball_box_ways :
  let num_ways := 3 ^ 5
  in num_ways = 243 :=
by
  let num_ways := 3 ^ 5
  have h : num_ways = 243 := by sorry
  exact h

end distinguish_ball_box_ways_l347_347361


namespace set_intersection_complement_l347_347233

def U : Set ℝ := Set.univ
def A : Set ℝ := { y | ∃ x, x > 0 ∧ y = 4 / x }
def B : Set ℝ := { y | ∃ x, x < 1 ∧ y = 2^x }
def comp_B : Set ℝ := { y | y ≤ 0 } ∪ { y | y ≥ 2 }
def intersection : Set ℝ := { y | y ≥ 2 }

theorem set_intersection_complement :
  A ∩ comp_B = intersection :=
by
  sorry

end set_intersection_complement_l347_347233


namespace sum_of_roots_eq_l347_347632

theorem sum_of_roots_eq :
  let π := Real.pi in
  ∑ k in (Finset.range 22).filter (λ k, 11 ≤ k ∧ k ≤ 21), (2 * k * π / 7) = 18 * π := 
by 
  sorry

end sum_of_roots_eq_l347_347632


namespace number_of_triangles_in_decagon_l347_347043

theorem number_of_triangles_in_decagon : (∃ n k : ℕ, n = 10 ∧ k = 3 ∧ ∀ (d : ℕ), (d = n.choose k) → d = 120) :=
by
  use 10
  use 3
  split
  . exact rfl
  split
  . exact rfl
  intro d h
  rw ←h
  exact Nat.choose_spec 10 3 120 sorry

end number_of_triangles_in_decagon_l347_347043


namespace number_of_triangles_in_regular_decagon_l347_347129

def number_of_triangles_from_decagon (vertices : Finset ℕ) (h : vertices.card = 10) : ℕ :=
  vertices.choose 3

theorem number_of_triangles_in_regular_decagon 
  (vertices : Finset ℕ)
  (h : vertices.card = 10)
  (h_no_three_collinear : True) : number_of_triangles_from_decagon vertices h = 120 :=
sorry

end number_of_triangles_in_regular_decagon_l347_347129


namespace share_expenses_l347_347247

theorem share_expenses (h l : ℕ) : 
  let henry_paid := 120
  let linda_paid := 150
  let jack_paid := 210
  let total_paid := henry_paid + linda_paid + jack_paid
  let each_should_pay := total_paid / 3
  let henry_owes := each_should_pay - henry_paid
  let linda_owes := each_should_pay - linda_paid
  (h = henry_owes) → 
  (l = linda_owes) → 
  h - l = 30 := by
  sorry

end share_expenses_l347_347247


namespace ways_to_place_balls_in_boxes_l347_347493

theorem ways_to_place_balls_in_boxes :
  ∃ ways : ℕ, ways = 3 ^ 5 ∧ ways = 243 :=
by
  use 3 ^ 5
  split
  · rfl
  · norm_num
  done

end ways_to_place_balls_in_boxes_l347_347493
