import Mathlib

namespace NUMINAMATH_GPT_equal_segments_l958_95852

-- Given a triangle ABC and D as the foot of the bisector from B
variables (A B C D E F : Point) (ABC : Triangle A B C) (Dfoot : BisectorFoot B A C D) 

-- Given that the circumcircles of triangles ABD and BCD intersect sides AB and BC at E and F respectively
variables (circABD : Circumcircle A B D) (circBCD : Circumcircle B C D)
variables (intersectAB : Intersect circABD A B E) (intersectBC : Intersect circBCD B C F)

-- The theorem to prove that AE = CF
theorem equal_segments : AE = CF :=
by
  sorry

end NUMINAMATH_GPT_equal_segments_l958_95852


namespace NUMINAMATH_GPT_base7_to_base10_l958_95855

open Nat

theorem base7_to_base10 : (3 * 7^2 + 5 * 7^1 + 1 * 7^0 = 183) :=
by
  sorry

end NUMINAMATH_GPT_base7_to_base10_l958_95855


namespace NUMINAMATH_GPT_division_of_fractions_l958_95890

theorem division_of_fractions : (1 / 6) / (1 / 3) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_division_of_fractions_l958_95890


namespace NUMINAMATH_GPT_sum_of_opposites_is_zero_l958_95898

theorem sum_of_opposites_is_zero (a b : ℚ) (h : a = -b) : a + b = 0 := 
by sorry

end NUMINAMATH_GPT_sum_of_opposites_is_zero_l958_95898


namespace NUMINAMATH_GPT_mixed_number_division_l958_95875

theorem mixed_number_division :
  (4 + 2 / 3 + 5 + 1 / 4) / (3 + 1 / 2 - 2 + 3 / 5) = 11 + 1 / 54 :=
by
  sorry

end NUMINAMATH_GPT_mixed_number_division_l958_95875


namespace NUMINAMATH_GPT_segment_lengths_l958_95816

theorem segment_lengths (AB BC CD DE EF : ℕ) 
  (h1 : AB > BC)
  (h2 : BC > CD)
  (h3 : CD > DE)
  (h4 : DE > EF)
  (h5 : AB = 2 * EF)
  (h6 : AB + BC + CD + DE + EF = 53) :
  (AB, BC, CD, DE, EF) = (14, 12, 11, 9, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 11, 8, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 10, 9, 7) :=
sorry

end NUMINAMATH_GPT_segment_lengths_l958_95816


namespace NUMINAMATH_GPT_find_fifth_day_sales_l958_95810

-- Define the variables and conditions
variables (x : ℝ)
variables (a : ℝ := 100) (b : ℝ := 92) (c : ℝ := 109) (d : ℝ := 96) (f : ℝ := 96) (g : ℝ := 105)
variables (mean : ℝ := 100.1)

-- Define the mean condition which leads to the proof of x
theorem find_fifth_day_sales : (a + b + c + d + x + f + g) / 7 = mean → x = 102.7 := by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_fifth_day_sales_l958_95810


namespace NUMINAMATH_GPT_quadratic_has_real_roots_range_l958_95877

noncomputable def has_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := 2
  let c := -1
  b^2 - 4 * a * c ≥ 0

theorem quadratic_has_real_roots_range (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 ∧ k ≠ 0 := by
sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_range_l958_95877


namespace NUMINAMATH_GPT_h_even_if_g_odd_l958_95825

structure odd_function (g : ℝ → ℝ) : Prop :=
(odd : ∀ x : ℝ, g (-x) = -g x)

def h (g : ℝ → ℝ) (x : ℝ) : ℝ := abs (g (x^5))

theorem h_even_if_g_odd (g : ℝ → ℝ) (hg : odd_function g) : ∀ x : ℝ, h g x = h g (-x) :=
by
  sorry

end NUMINAMATH_GPT_h_even_if_g_odd_l958_95825


namespace NUMINAMATH_GPT_maximum_distance_product_l958_95857

theorem maximum_distance_product (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let ρ1 := 4 * Real.cos α
  let ρ2 := 2 * Real.sin α
  |ρ1 * ρ2| ≤ 4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_maximum_distance_product_l958_95857


namespace NUMINAMATH_GPT_range_of_b_div_c_l958_95889

theorem range_of_b_div_c (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : b^2 = c^2 + a * c) :
  1 < b / c ∧ b / c < 2 := 
sorry

end NUMINAMATH_GPT_range_of_b_div_c_l958_95889


namespace NUMINAMATH_GPT_depth_of_well_l958_95870

theorem depth_of_well (d : ℝ) (t1 t2 : ℝ)
  (h1 : d = 15 * t1^2)
  (h2 : t2 = d / 1100)
  (h3 : t1 + t2 = 9.5) :
  d = 870.25 := 
sorry

end NUMINAMATH_GPT_depth_of_well_l958_95870


namespace NUMINAMATH_GPT_initial_quarters_l958_95854

-- Define the conditions
def quartersAfterLoss (x : ℕ) : ℕ := (4 * x) / 3
def quartersAfterThirdYear (x : ℕ) : ℕ := x - 4
def quartersAfterSecondYear (x : ℕ) : ℕ := x - 36
def quartersAfterFirstYear (x : ℕ) : ℕ := x * 2

-- The main theorem
theorem initial_quarters (x : ℕ) (h1 : quartersAfterLoss x = 140)
    (h2 : quartersAfterThirdYear 140 = 136)
    (h3 : quartersAfterSecondYear 136 = 100)
    (h4 : quartersAfterFirstYear 50 = 100) :
  x = 50 := by
  simp [quartersAfterFirstYear, quartersAfterSecondYear,
        quartersAfterThirdYear, quartersAfterLoss] at *
  sorry

end NUMINAMATH_GPT_initial_quarters_l958_95854


namespace NUMINAMATH_GPT_tile_D_is_IV_l958_95859

structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

def Tile_I : Tile := ⟨3, 1, 4, 2⟩
def Tile_II : Tile := ⟨2, 3, 1, 5⟩
def Tile_III : Tile := ⟨4, 0, 3, 1⟩
def Tile_IV : Tile := ⟨5, 4, 2, 0⟩

def is_tile_D (t : Tile) : Prop :=
  t.left = 0 ∧ t.top = 5

theorem tile_D_is_IV : is_tile_D Tile_IV :=
  by
    -- skip proof here
    sorry

end NUMINAMATH_GPT_tile_D_is_IV_l958_95859


namespace NUMINAMATH_GPT_simon_fraction_of_alvin_l958_95887

theorem simon_fraction_of_alvin (alvin_age simon_age : ℕ) (h_alvin : alvin_age = 30)
  (h_simon : simon_age = 10) (h_fraction : ∃ f : ℚ, simon_age + 5 = f * (alvin_age + 5)) :
  ∃ f : ℚ, f = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_simon_fraction_of_alvin_l958_95887


namespace NUMINAMATH_GPT_atomic_number_order_l958_95895

-- Define that elements A, B, C, D, and E are in the same period
variable (A B C D E : Type)

-- Define conditions based on the problem
def highest_valence_oxide_basic (x : Type) : Prop := sorry
def basicity_greater (x y : Type) : Prop := sorry
def gaseous_hydride_stability (x y : Type) : Prop := sorry
def smallest_ionic_radius (x : Type) : Prop := sorry

-- Assume conditions given in the problem
axiom basic_oxides : highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
axiom basicity_order : basicity_greater B A
axiom hydride_stabilities : gaseous_hydride_stability C D
axiom smallest_radius : smallest_ionic_radius E

-- Prove that the order of atomic numbers from smallest to largest is B, A, E, D, C
theorem atomic_number_order :
  ∃ (A B C D E : Type), highest_valence_oxide_basic A ∧ highest_valence_oxide_basic B
  ∧ basicity_greater B A ∧ gaseous_hydride_stability C D ∧ smallest_ionic_radius E
  ↔ B = B ∧ A = A ∧ E = E ∧ D = D ∧ C = C := sorry

end NUMINAMATH_GPT_atomic_number_order_l958_95895


namespace NUMINAMATH_GPT_length_AC_l958_95803

open Real

noncomputable def net_south_north (south north : ℝ) : ℝ := south - north
noncomputable def net_east_west (east west : ℝ) : ℝ := east - west
noncomputable def distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

theorem length_AC :
  let A : ℝ := 0
  let south := 30
  let north := 20
  let east := 40
  let west := 35
  let net_south := net_south_north south north
  let net_east := net_east_west east west
  distance net_south net_east = 5 * sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_length_AC_l958_95803


namespace NUMINAMATH_GPT_harry_change_l958_95851

theorem harry_change (a : ℕ) :
  (∃ k : ℕ, a = 50 * k + 2 ∧ a < 100) ∧ (∃ m : ℕ, a = 5 * m + 4 ∧ a < 100) →
  a = 52 :=
by sorry

end NUMINAMATH_GPT_harry_change_l958_95851


namespace NUMINAMATH_GPT_don_travel_time_to_hospital_l958_95893

noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def time_to_travel (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem don_travel_time_to_hospital :
  let speed_mary := 60
  let speed_don := 30
  let time_mary_minutes := 15
  let time_mary_hours := time_mary_minutes / 60
  let distance := distance_traveled speed_mary time_mary_hours
  let time_don_hours := time_to_travel distance speed_don
  time_don_hours * 60 = 30 :=
by
  sorry

end NUMINAMATH_GPT_don_travel_time_to_hospital_l958_95893


namespace NUMINAMATH_GPT_Greg_and_Earl_together_l958_95804

-- Conditions
def Earl_initial : ℕ := 90
def Fred_initial : ℕ := 48
def Greg_initial : ℕ := 36

def Earl_to_Fred : ℕ := 28
def Fred_to_Greg : ℕ := 32
def Greg_to_Earl : ℕ := 40

def Earl_final : ℕ := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final : ℕ := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final : ℕ := Greg_initial + Fred_to_Greg - Greg_to_Earl

-- Theorem statement
theorem Greg_and_Earl_together : Greg_final + Earl_final = 130 := by
  sorry

end NUMINAMATH_GPT_Greg_and_Earl_together_l958_95804


namespace NUMINAMATH_GPT_find_g_75_l958_95878

variable (g : ℝ → ℝ)

def prop_1 := ∀ x y : ℝ, x > 0 → y > 0 → g (x * y) = g x / y
def prop_2 := g 50 = 30

theorem find_g_75 (h1 : prop_1 g) (h2 : prop_2 g) : g 75 = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_g_75_l958_95878


namespace NUMINAMATH_GPT_algebraic_sum_of_coefficients_l958_95849

open Nat

theorem algebraic_sum_of_coefficients
  (u : ℕ → ℤ)
  (h1 : u 1 = 5)
  (hrec : ∀ n : ℕ, n > 0 → u (n + 1) - u n = 3 + 4 * (n - 1)) :
  (∃ P : ℕ → ℤ, (∀ n, u n = P n) ∧ (P 1 + P 0 = 5)) :=
sorry

end NUMINAMATH_GPT_algebraic_sum_of_coefficients_l958_95849


namespace NUMINAMATH_GPT_john_needs_to_add_empty_cans_l958_95865

theorem john_needs_to_add_empty_cans :
  ∀ (num_full_cans : ℕ) (weight_per_full_can total_weight weight_per_empty_can required_weight : ℕ),
  num_full_cans = 6 →
  weight_per_full_can = 14 →
  total_weight = 88 →
  weight_per_empty_can = 2 →
  required_weight = total_weight - (num_full_cans * weight_per_full_can) →
  required_weight / weight_per_empty_can = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_john_needs_to_add_empty_cans_l958_95865


namespace NUMINAMATH_GPT_distribution_scheme_count_l958_95835

noncomputable def NumberOfDistributionSchemes : Nat :=
  let plumbers := 5
  let residences := 4
  Nat.choose plumbers (residences - 1) * Nat.factorial residences

theorem distribution_scheme_count :
  NumberOfDistributionSchemes = 240 :=
by
  sorry

end NUMINAMATH_GPT_distribution_scheme_count_l958_95835


namespace NUMINAMATH_GPT_arrival_time_difference_l958_95897

-- Define the times in minutes, with 600 representing 10:00 AM.
def my_watch_time_planned := 600
def my_watch_fast := 5
def my_watch_slow := 10

def friend_watch_time_planned := 600
def friend_watch_fast := 5

-- Calculate actual arrival times.
def my_actual_arrival_time := my_watch_time_planned - my_watch_fast + my_watch_slow
def friend_actual_arrival_time := friend_watch_time_planned - friend_watch_fast

-- Prove the arrival times and difference.
theorem arrival_time_difference :
  friend_actual_arrival_time < my_actual_arrival_time ∧
  my_actual_arrival_time - friend_actual_arrival_time = 20 :=
by
  -- Proof terms can be filled in later.
  sorry

end NUMINAMATH_GPT_arrival_time_difference_l958_95897


namespace NUMINAMATH_GPT_pawns_left_l958_95833

-- Definitions of the initial conditions
def initial_pawns : ℕ := 8
def kennedy_lost_pawns : ℕ := 4
def riley_lost_pawns : ℕ := 1

-- Definition of the total pawns left function
def total_pawns_left (initial_pawns kennedy_lost_pawns riley_lost_pawns : ℕ) : ℕ :=
  (initial_pawns - kennedy_lost_pawns) + (initial_pawns - riley_lost_pawns)

-- The statement to prove
theorem pawns_left : total_pawns_left initial_pawns kennedy_lost_pawns riley_lost_pawns = 11 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pawns_left_l958_95833


namespace NUMINAMATH_GPT_mod_exp_l958_95830

theorem mod_exp (n : ℕ) : (5^303) % 11 = 4 :=
  by sorry

end NUMINAMATH_GPT_mod_exp_l958_95830


namespace NUMINAMATH_GPT_Peter_vacation_l958_95863

theorem Peter_vacation
  (A : ℕ) (S : ℕ) (M : ℕ) (T : ℕ)
  (hA : A = 5000)
  (hS : S = 2900)
  (hM : M = 700)
  (hT : T = (A - S) / M) : T = 3 :=
sorry

end NUMINAMATH_GPT_Peter_vacation_l958_95863


namespace NUMINAMATH_GPT_min_cubes_needed_proof_l958_95899

noncomputable def min_cubes_needed_to_form_30_digit_number : ℕ :=
  sorry

theorem min_cubes_needed_proof : min_cubes_needed_to_form_30_digit_number = 50 :=
  sorry

end NUMINAMATH_GPT_min_cubes_needed_proof_l958_95899


namespace NUMINAMATH_GPT_perfect_rectangle_squares_l958_95860

theorem perfect_rectangle_squares (squares : Finset ℕ) 
  (h₁ : 9 ∈ squares) 
  (h₂ : 2 ∈ squares) 
  (h₃ : squares.card = 9) 
  (h₄ : ∀ x ∈ squares, ∃ y ∈ squares, x ≠ y ∧ (gcd x y = 1)) :
  squares = {2, 5, 7, 9, 16, 25, 28, 33, 36} := 
sorry

end NUMINAMATH_GPT_perfect_rectangle_squares_l958_95860


namespace NUMINAMATH_GPT_simplify_expression_l958_95840

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l958_95840


namespace NUMINAMATH_GPT_bridge_length_proof_l958_95843

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 49.9960003199744
noncomputable def train_speed_kmph : ℝ := 18
noncomputable def conversion_factor : ℝ := 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * conversion_factor
noncomputable def total_distance : ℝ := train_speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_proof : bridge_length = 149.980001599872 := 
by 
  sorry

end NUMINAMATH_GPT_bridge_length_proof_l958_95843


namespace NUMINAMATH_GPT_interest_rate_l958_95896

-- Define the sum of money
def P : ℝ := 1800

-- Define the time period in years
def T : ℝ := 2

-- Define the difference in interests
def interest_difference : ℝ := 18

-- Define the relationship between simple interest, compound interest, and the interest rate
theorem interest_rate (R : ℝ) 
  (h1 : SI = P * R * T / 100)
  (h2 : CI = P * (1 + R/100)^2 - P)
  (h3 : CI - SI = interest_difference) :
  R = 10 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_l958_95896


namespace NUMINAMATH_GPT_train_number_of_cars_l958_95894

theorem train_number_of_cars (lena_cars : ℕ) (time_counted : ℕ) (total_time : ℕ) 
  (cars_in_train : ℕ)
  (h1 : lena_cars = 8) 
  (h2 : time_counted = 15)
  (h3 : total_time = 210)
  (h4 : (8 / 15 : ℚ) * 210 = 112)
  : cars_in_train = 112 :=
sorry

end NUMINAMATH_GPT_train_number_of_cars_l958_95894


namespace NUMINAMATH_GPT_percentage_seats_not_taken_l958_95827

theorem percentage_seats_not_taken
  (rows : ℕ) (seats_per_row : ℕ) 
  (ticket_price : ℕ)
  (earnings : ℕ)
  (H_rows : rows = 150)
  (H_seats_per_row : seats_per_row = 10) 
  (H_ticket_price : ticket_price = 10)
  (H_earnings : earnings = 12000) :
  (1500 - (12000 / 10)) / 1500 * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_percentage_seats_not_taken_l958_95827


namespace NUMINAMATH_GPT_common_roots_cubic_polynomials_l958_95871

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ (r^3 + a * r^2 + 17 * r + 10 = 0) ∧ (s^3 + a * s^2 + 17 * s + 10 = 0) ∧ 
               (r^3 + b * r^2 + 20 * r + 12 = 0) ∧ (s^3 + b * s^2 + 20 * s + 12 = 0)) →
  (a, b) = (-6, -7) :=
by sorry

end NUMINAMATH_GPT_common_roots_cubic_polynomials_l958_95871


namespace NUMINAMATH_GPT_arccos_cos_11_equals_4_717_l958_95809

noncomputable def arccos_cos_11 : Real :=
  let n : ℤ := Int.floor (11 / (2 * Real.pi))
  Real.arccos (Real.cos 11)

theorem arccos_cos_11_equals_4_717 :
  arccos_cos_11 = 4.717 := by
  sorry

end NUMINAMATH_GPT_arccos_cos_11_equals_4_717_l958_95809


namespace NUMINAMATH_GPT_ratio_volumes_tetrahedron_octahedron_l958_95824

theorem ratio_volumes_tetrahedron_octahedron (a b : ℝ) (h_eq_areas : a^2 * (Real.sqrt 3) = 2 * b^2 * (Real.sqrt 3)) :
  (a^3 * (Real.sqrt 2) / 12) / (b^3 * (Real.sqrt 2) / 3) = 1 / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_volumes_tetrahedron_octahedron_l958_95824


namespace NUMINAMATH_GPT_max_value_f_l958_95848

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

theorem max_value_f (x : ℝ) (h : -4 < x ∧ x < 1) : ∃ y, f y = -1 ∧ (∀ z, f z ≤ f y) :=
by 
  sorry

end NUMINAMATH_GPT_max_value_f_l958_95848


namespace NUMINAMATH_GPT_number_of_Sunzi_books_l958_95867

theorem number_of_Sunzi_books
    (num_books : ℕ) (total_cost : ℕ)
    (price_Zhuangzi price_Kongzi price_Mengzi price_Laozi price_Sunzi : ℕ)
    (num_Zhuangzi num_Kongzi num_Mengzi num_Laozi num_Sunzi : ℕ) :
  num_books = 300 →
  total_cost = 4500 →
  price_Zhuangzi = 10 →
  price_Kongzi = 20 →
  price_Mengzi = 15 →
  price_Laozi = 30 →
  price_Sunzi = 12 →
  num_Zhuangzi = num_Kongzi →
  num_Sunzi = 4 * num_Laozi + 15 →
  num_Zhuangzi + num_Kongzi + num_Mengzi + num_Laozi + num_Sunzi = num_books →
  price_Zhuangzi * num_Zhuangzi +
  price_Kongzi * num_Kongzi +
  price_Mengzi * num_Mengzi +
  price_Laozi * num_Laozi +
  price_Sunzi * num_Sunzi = total_cost →
  num_Sunzi = 75 :=
by
  intros h_nb h_tc h_pZ h_pK h_pM h_pL h_pS h_nZ h_nS h_books h_cost
  sorry

end NUMINAMATH_GPT_number_of_Sunzi_books_l958_95867


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l958_95881

theorem distance_between_foci_of_ellipse :
  let c := (5, 2)
  let a := 5
  let b := 2
  2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21 :=
by
  let c := (5, 2)
  let a := 5
  let b := 2
  show 2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 21
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l958_95881


namespace NUMINAMATH_GPT_range_of_a_if_monotonic_l958_95874

theorem range_of_a_if_monotonic :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 3 * a * x^2 - 2 * x + 1 ≥ 0) → a > 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_if_monotonic_l958_95874


namespace NUMINAMATH_GPT_unique_solution_for_log_problem_l958_95866

noncomputable def log_problem (x : ℝ) :=
  let a := Real.log (x / 2 - 1) / Real.log (x - 11 / 4).sqrt
  let b := 2 * Real.log (x - 11 / 4) / Real.log (x / 2 - 1 / 4)
  let c := Real.log (x / 2 - 1 / 4) / (2 * Real.log (x / 2 - 1))
  a * b * c = 2 ∧ (a = b ∧ c = a + 1)

theorem unique_solution_for_log_problem :
  ∃! x, log_problem x = true := sorry

end NUMINAMATH_GPT_unique_solution_for_log_problem_l958_95866


namespace NUMINAMATH_GPT_range_of_a_for_quadratic_eq_l958_95845

theorem range_of_a_for_quadratic_eq (a : ℝ) (h : ∀ x : ℝ, ax^2 = (x+1)*(x-1)) : a ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_quadratic_eq_l958_95845


namespace NUMINAMATH_GPT_determine_p_range_l958_95873

theorem determine_p_range :
  ∀ (p : ℝ), (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = (x + 9 / 8) * (x + 9 / 8) ∧ (f x) = (8*x^2 + 18*x + 4*p)/8 ) →
  2.5 < p ∧ p < 2.6 :=
by
  sorry

end NUMINAMATH_GPT_determine_p_range_l958_95873


namespace NUMINAMATH_GPT_standard_equation_of_circle_l958_95856

-- Definitions based on problem conditions
def center : ℝ × ℝ := (-1, 2)
def radius : ℝ := 2

-- Lean statement of the problem
theorem standard_equation_of_circle :
  ∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = radius ^ 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end NUMINAMATH_GPT_standard_equation_of_circle_l958_95856


namespace NUMINAMATH_GPT_sum_of_numbers_eq_l958_95838

theorem sum_of_numbers_eq (a b : ℕ) (h1 : a = 64) (h2 : b = 32) (h3 : a = 2 * b) : a + b = 96 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_eq_l958_95838


namespace NUMINAMATH_GPT_jerry_painting_hours_l958_95837

-- Define the variables and conditions
def time_painting (P : ℕ) : ℕ := P
def time_counter (P : ℕ) : ℕ := 3 * P
def time_lawn : ℕ := 6
def hourly_rate : ℕ := 15
def total_paid : ℕ := 570

-- Hypothesize that the total hours spent leads to the total payment
def total_hours (P : ℕ) : ℕ := time_painting P + time_counter P + time_lawn

-- Prove that the solution for P matches the conditions
theorem jerry_painting_hours (P : ℕ) 
  (h1 : hourly_rate * total_hours P = total_paid) : 
  P = 8 :=
by
  sorry

end NUMINAMATH_GPT_jerry_painting_hours_l958_95837


namespace NUMINAMATH_GPT_f_g_3_value_l958_95853

def f (x : ℝ) := x^3 + 1
def g (x : ℝ) := 3 * x + 2

theorem f_g_3_value : f (g 3) = 1332 := by
  sorry

end NUMINAMATH_GPT_f_g_3_value_l958_95853


namespace NUMINAMATH_GPT_problem_1_problem_2_l958_95823

variable (a : ℕ → ℝ)

variables (h1 : ∀ n, 0 < a n) (h2 : ∀ n, a (n + 1) + 1 / a n < 2)

-- Prove that: (1) a_{n+2} < a_{n+1} < 2 for n ∈ ℕ*
theorem problem_1 (n : ℕ) : a (n + 2) < a (n + 1) ∧ a (n + 1) < 2 := 
sorry

-- Prove that: (2) a_n > 1 for n ∈ ℕ*
theorem problem_2 (n : ℕ) : 1 < a n := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l958_95823


namespace NUMINAMATH_GPT_range_cos_A_l958_95876

theorem range_cos_A {A B C : ℚ} (h : 1 / (Real.tan B) + 1 / (Real.tan C) = 1 / (Real.tan A))
  (h_non_neg_A: 0 ≤ A) (h_less_pi_A: A ≤ π): 
  (Real.cos A ∈ Set.Ico (2 / 3) 1) :=
sorry

end NUMINAMATH_GPT_range_cos_A_l958_95876


namespace NUMINAMATH_GPT_total_time_assignment_l958_95882

-- Define the time taken for each part
def time_first_part : ℕ := 25
def time_second_part : ℕ := 2 * time_first_part
def time_third_part : ℕ := 45

-- Define the total time taken for the assignment
def total_time : ℕ := time_first_part + time_second_part + time_third_part

-- The theorem stating that the total time is 120 minutes
theorem total_time_assignment : total_time = 120 := by
  sorry

end NUMINAMATH_GPT_total_time_assignment_l958_95882


namespace NUMINAMATH_GPT_participants_initial_count_l958_95821

theorem participants_initial_count 
  (x : ℕ) 
  (p1 : x * (2 : ℚ) / 5 * 1 / 4 = 30) :
  x = 300 :=
by
  sorry

end NUMINAMATH_GPT_participants_initial_count_l958_95821


namespace NUMINAMATH_GPT_intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l958_95820

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

theorem intervals_of_monotonicity_a_eq_1 : 
  ∀ x : ℝ, (0 < x ∧ x < Real.sqrt 2) → 
  f x 1 < f (Real.sqrt 2) 1 ∧ 
  ∀ x : ℝ, (Real.sqrt 2 < x ∧ x < 2) → 
  f x 1 > f (Real.sqrt 2) 1 := 
sorry

theorem max_value_implies_a_half : 
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ∧ f 1 a = 1/2 → a = 1/2 := 
sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l958_95820


namespace NUMINAMATH_GPT_symmetric_point_y_axis_l958_95891

theorem symmetric_point_y_axis (A B : ℝ × ℝ) (hA : A = (2, 5)) (h_symm : B = (-A.1, A.2)) :
  B = (-2, 5) :=
sorry

end NUMINAMATH_GPT_symmetric_point_y_axis_l958_95891


namespace NUMINAMATH_GPT_neg_ex_proposition_l958_95836

open Classical

theorem neg_ex_proposition :
  ¬ (∃ n : ℕ, n^2 > 2^n) ↔ ∀ n : ℕ, n^2 ≤ 2^n :=
by sorry

end NUMINAMATH_GPT_neg_ex_proposition_l958_95836


namespace NUMINAMATH_GPT_acid_base_mixture_ratio_l958_95814

theorem acid_base_mixture_ratio (r s t : ℝ) (hr : r ≥ 0) (hs : s ≥ 0) (ht : t ≥ 0) :
  (r ≠ -1) → (s ≠ -1) → (t ≠ -1) →
  let acid_volume := (r/(r+1) + s/(s+1) + t/(t+1))
  let base_volume := (1/(r+1) + 1/(s+1) + 1/(t+1))
  acid_volume / base_volume = (rst + rt + rs + st) / (rs + rt + st + r + s + t + 3) := 
by {
  sorry
}

end NUMINAMATH_GPT_acid_base_mixture_ratio_l958_95814


namespace NUMINAMATH_GPT_perpendicular_condition_sufficient_but_not_necessary_l958_95872

theorem perpendicular_condition_sufficient_but_not_necessary (m : ℝ) (h : m = -1) :
  (∀ x y : ℝ, mx + (2 * m - 1) * y + 1 = 0 ∧ 3 * x + m * y + 2 = 0) → (m = 0 ∨ m = -1) → (m = 0 ∨ m = -1) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_perpendicular_condition_sufficient_but_not_necessary_l958_95872


namespace NUMINAMATH_GPT_total_distance_hiked_l958_95858

theorem total_distance_hiked
  (a b c d e : ℕ)
  (h1 : a + b + c = 34)
  (h2 : b + c = 24)
  (h3 : c + d + e = 40)
  (h4 : a + c + e = 38)
  (h5 : d = 14) :
  a + b + c + d + e = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_hiked_l958_95858


namespace NUMINAMATH_GPT_highest_elevation_l958_95847

-- Define the function for elevation as per the conditions
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2

-- Prove that the highest elevation reached is 500 meters
theorem highest_elevation : (exists t : ℝ, elevation t = 500) ∧ (∀ t : ℝ, elevation t ≤ 500) := sorry

end NUMINAMATH_GPT_highest_elevation_l958_95847


namespace NUMINAMATH_GPT_prob_one_AB_stuck_prob_at_least_two_stuck_l958_95868

-- Define the events and their probabilities.
def prob_traffic_I := 1 / 10
def prob_no_traffic_I := 9 / 10
def prob_traffic_II := 3 / 5
def prob_no_traffic_II := 2 / 5

-- Define the events
def event_A := prob_traffic_I
def not_event_A := prob_no_traffic_I
def event_B := prob_traffic_I
def not_event_B := prob_no_traffic_I
def event_C := prob_traffic_II
def not_event_C := prob_no_traffic_II

-- Define the probabilities as required in the problem
def prob_exactly_one_of_A_B_in_traffic :=
  event_A * not_event_B + not_event_A * event_B

def prob_at_least_two_in_traffic :=
  event_A * event_B * not_event_C +
  event_A * not_event_B * event_C +
  not_event_A * event_B * event_C +
  event_A * event_B * event_C

-- Proofs (statements only)
theorem prob_one_AB_stuck :
  prob_exactly_one_of_A_B_in_traffic = 9 / 50 := sorry

theorem prob_at_least_two_stuck :
  prob_at_least_two_in_traffic = 59 / 500 := sorry

end NUMINAMATH_GPT_prob_one_AB_stuck_prob_at_least_two_stuck_l958_95868


namespace NUMINAMATH_GPT_chocolate_per_friend_l958_95801

-- Definitions according to the conditions
def total_chocolate : ℚ := 60 / 7
def piles := 5
def friends := 3

-- Proof statement for the equivalent problem
theorem chocolate_per_friend :
  (total_chocolate / piles) * (piles - 1) / friends = 16 / 7 := by
  sorry

end NUMINAMATH_GPT_chocolate_per_friend_l958_95801


namespace NUMINAMATH_GPT_complex_point_quadrant_l958_95884

theorem complex_point_quadrant 
  (i : Complex) 
  (h_i_unit : i = Complex.I) : 
  (Complex.re ((i - 3) / (1 + i)) < 0) ∧ (Complex.im ((i - 3) / (1 + i)) > 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_complex_point_quadrant_l958_95884


namespace NUMINAMATH_GPT_four_digit_multiples_of_13_and_7_l958_95879

theorem four_digit_multiples_of_13_and_7 : 
  (∃ n : ℕ, 
    (∀ k : ℕ, 1000 ≤ k ∧ k < 10000 ∧ k % 91 = 0 → k = 1001 + 91 * (n - 11)) 
    ∧ n - 11 + 1 = 99) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_multiples_of_13_and_7_l958_95879


namespace NUMINAMATH_GPT_least_large_groups_l958_95817

theorem least_large_groups (total_members : ℕ) (members_large_group : ℕ) (members_small_group : ℕ) (L : ℕ) (S : ℕ)
  (H_total : total_members = 90)
  (H_large : members_large_group = 7)
  (H_small : members_small_group = 3)
  (H_eq : total_members = L * members_large_group + S * members_small_group) :
  L = 12 :=
by
  have h1 : total_members = 90 := by exact H_total
  have h2 : members_large_group = 7 := by exact H_large
  have h3 : members_small_group = 3 := by exact H_small
  rw [h1, h2, h3] at H_eq
  -- The proof is skipped here
  sorry

end NUMINAMATH_GPT_least_large_groups_l958_95817


namespace NUMINAMATH_GPT_no_real_roots_for_polynomial_l958_95808

theorem no_real_roots_for_polynomial :
  (∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + (5/2) ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_for_polynomial_l958_95808


namespace NUMINAMATH_GPT_hyperbola_product_slopes_constant_l958_95885

theorem hyperbola_product_slopes_constant (a b x0 y0 : ℝ) (h_a : a > 0) (h_b : b > 0) (hP : (x0 / a) ^ 2 - (y0 / b) ^ 2 = 1) (h_diff_a1_a2 : x0 ≠ a ∧ x0 ≠ -a) :
  (y0 / (x0 + a)) * (y0 / (x0 - a)) = b^2 / a^2 :=
by sorry

end NUMINAMATH_GPT_hyperbola_product_slopes_constant_l958_95885


namespace NUMINAMATH_GPT_prism_faces_l958_95834

-- Define the conditions of the problem
def prism (E : ℕ) : Prop :=
  ∃ (L : ℕ), 3 * L = E

-- Define the main proof statement
theorem prism_faces (E : ℕ) (hE : prism E) : E = 27 → 2 + E / 3 = 11 :=
by
  sorry -- Proof is not required

end NUMINAMATH_GPT_prism_faces_l958_95834


namespace NUMINAMATH_GPT_polynomial_divisibility_l958_95802

theorem polynomial_divisibility (P : Polynomial ℂ) (n : ℕ) 
  (h : ∃ Q : Polynomial ℂ, P.comp (X ^ n) = (X - 1) * Q) : 
  ∃ R : Polynomial ℂ, P.comp (X ^ n) = (X ^ n - 1) * R :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l958_95802


namespace NUMINAMATH_GPT_determine_ω_and_φ_l958_95883

noncomputable def f (x : ℝ) (ω φ : ℝ) := 2 * Real.sin (ω * x + φ)
def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) := (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∃ d > 0, d < T ∧ ∀ m n : ℤ, m ≠ n → f (m * d) ≠ f (n * d))

theorem determine_ω_and_φ :
  ∃ ω φ : ℝ,
    (0 < ω) ∧
    (|φ| < Real.pi / 2) ∧
    (smallest_positive_period (f ω φ) Real.pi) ∧
    (f 0 ω φ = Real.sqrt 3) ∧
    (ω = 2 ∧ φ = Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_determine_ω_and_φ_l958_95883


namespace NUMINAMATH_GPT_value_of_t5_l958_95828

noncomputable def t_5_value (t1 t2 : ℚ) (r : ℚ) (a : ℚ) : ℚ := a * r^4

theorem value_of_t5 
  (a r : ℚ)
  (h1 : a > 0)  -- condition: each term is positive
  (h2 : a + a * r = 15 / 2)  -- condition: sum of first two terms is 15/2
  (h3 : a^2 + (a * r)^2 = 153 / 4)  -- condition: sum of squares of first two terms is 153/4
  (h4 : r > 0)  -- ensuring positivity of r
  (h5 : r < 1)  -- ensuring t1 > t2
  : t_5_value a (a * r) r a = 3 / 128 :=
sorry

end NUMINAMATH_GPT_value_of_t5_l958_95828


namespace NUMINAMATH_GPT_video_game_map_width_l958_95850

theorem video_game_map_width (volume length height : ℝ) (h1 : volume = 50)
                            (h2 : length = 5) (h3 : height = 2) :
  ∃ width : ℝ, volume = length * width * height ∧ width = 5 :=
by
  sorry

end NUMINAMATH_GPT_video_game_map_width_l958_95850


namespace NUMINAMATH_GPT_ellipse_foci_distance_l958_95861

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_l958_95861


namespace NUMINAMATH_GPT_calvin_haircut_goal_percentage_l958_95841

theorem calvin_haircut_goal_percentage :
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  (completed_haircuts / total_haircuts_needed) * 100 = 80 :=
by
  let completed_haircuts := 8
  let total_haircuts_needed := 8 + 2
  show (completed_haircuts / total_haircuts_needed) * 100 = 80
  sorry

end NUMINAMATH_GPT_calvin_haircut_goal_percentage_l958_95841


namespace NUMINAMATH_GPT_triangle_number_placement_l958_95888

theorem triangle_number_placement
  (A B C D E F : ℕ)
  (h1 : A + B + C = 6)
  (h2 : D = 5)
  (h3 : E = 6)
  (h4 : D + E + F = 14)
  (h5 : B = 3) : 
  (A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_number_placement_l958_95888


namespace NUMINAMATH_GPT_num_people_watched_last_week_l958_95869

variable (s f t : ℕ)
variable (h1 : s = 80)
variable (h2 : f = s - 20)
variable (h3 : t = s + 15)
variable (total_last_week total_this_week : ℕ)
variable (h4 : total_this_week = f + s + t)
variable (h5 : total_this_week = total_last_week + 35)

theorem num_people_watched_last_week :
  total_last_week = 200 := sorry

end NUMINAMATH_GPT_num_people_watched_last_week_l958_95869


namespace NUMINAMATH_GPT_part1_part2_l958_95800

theorem part1 (x p : ℝ) (h : abs p ≤ 2) : (x^2 + p * x + 1 > 2 * x + p) ↔ (x < -1 ∨ 3 < x) := 
by 
  sorry

theorem part2 (x p : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : (x^2 + p * x + 1 > 2 * x + p) ↔ (-1 < p) := 
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l958_95800


namespace NUMINAMATH_GPT_opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l958_95811

theorem opposite_neg_abs_five_minus_six : -|5 - 6| = -1 := by
  sorry

theorem opposite_of_neg_abs (h : -|5 - 6| = -1) : -(-1) = 1 := by
  sorry

theorem math_problem_proof : -(-|5 - 6|) = 1 := by
  apply opposite_of_neg_abs
  apply opposite_neg_abs_five_minus_six

end NUMINAMATH_GPT_opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l958_95811


namespace NUMINAMATH_GPT_product_sequence_equals_8_l958_95822

theorem product_sequence_equals_8 :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := 
by
  sorry

end NUMINAMATH_GPT_product_sequence_equals_8_l958_95822


namespace NUMINAMATH_GPT_find_number_of_two_dollar_pairs_l958_95839

noncomputable def pairs_of_two_dollars (x y z : ℕ) : Prop :=
  x + y + z = 15 ∧ 2 * x + 4 * y + 5 * z = 38 ∧ x >= 1 ∧ y >= 1 ∧ z >= 1

theorem find_number_of_two_dollar_pairs (x y z : ℕ) 
  (h1 : x + y + z = 15) 
  (h2 : 2 * x + 4 * y + 5 * z = 38) 
  (hx : x >= 1) 
  (hy : y >= 1) 
  (hz : z >= 1) :
  pairs_of_two_dollars x y z → x = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_number_of_two_dollar_pairs_l958_95839


namespace NUMINAMATH_GPT_find_a_and_theta_find_sin_alpha_plus_pi_over_3_l958_95815

noncomputable def f (a θ x : ℝ) : ℝ :=
  (a + 2 * Real.cos x ^ 2) * Real.cos (2 * x + θ)

theorem find_a_and_theta (a θ : ℝ) (h1 : f a θ (Real.pi / 4) = 0)
  (h2 : ∀ x, f a θ (-x) = -f a θ x) :
  a = -1 ∧ θ = Real.pi / 2 :=
sorry

theorem find_sin_alpha_plus_pi_over_3 (α θ : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : f (-1) (Real.pi / 2) (α / 4) = -2 / 5) :
  Real.sin (α + Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end NUMINAMATH_GPT_find_a_and_theta_find_sin_alpha_plus_pi_over_3_l958_95815


namespace NUMINAMATH_GPT_tan_diff_l958_95842

variables {α β : ℝ}

theorem tan_diff (h1 : Real.tan α = -3/4) (h2 : Real.tan (Real.pi - β) = 1/2) :
  Real.tan (α - β) = -2/11 :=
by
  sorry

end NUMINAMATH_GPT_tan_diff_l958_95842


namespace NUMINAMATH_GPT_ratio_EG_GD_l958_95812

theorem ratio_EG_GD (a EG GD : ℝ)
  (h1 : EG = 4 * GD)
  (gcd_1 : Int.gcd 4 1 = 1) :
  4 + 1 = 5 := by
  sorry

end NUMINAMATH_GPT_ratio_EG_GD_l958_95812


namespace NUMINAMATH_GPT_algebra_expression_value_l958_95807

theorem algebra_expression_value (a b c : ℝ) (h1 : a - b = 3) (h2 : b + c = -5) : 
  ac - bc + a^2 - ab = -6 := by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l958_95807


namespace NUMINAMATH_GPT_alpha_value_l958_95846

-- Define the conditions in Lean
variables (α β γ k : ℝ)

-- Mathematically equivalent problem statements translated to Lean
theorem alpha_value :
  (∀ β γ, α = (k * γ) / β) → -- proportionality condition
  (α = 4) →
  (β = 27) →
  (γ = 3) →
  (∀ β γ, β = -81 → γ = 9 → α = -4) :=
by
  sorry

end NUMINAMATH_GPT_alpha_value_l958_95846


namespace NUMINAMATH_GPT_winning_candidate_votes_percentage_l958_95818

theorem winning_candidate_votes_percentage (P : ℝ) 
    (majority : P/100 * 6000 - (6000 - P/100 * 6000) = 1200) : 
    P = 60 := 
by 
  sorry

end NUMINAMATH_GPT_winning_candidate_votes_percentage_l958_95818


namespace NUMINAMATH_GPT_change_received_correct_l958_95819

-- Define the prices of items and the amount paid
def price_hamburger : ℕ := 4
def price_onion_rings : ℕ := 2
def price_smoothie : ℕ := 3
def amount_paid : ℕ := 20

-- Define the total cost and the change received
def total_cost : ℕ := price_hamburger + price_onion_rings + price_smoothie
def change_received : ℕ := amount_paid - total_cost

-- Theorem stating the change received
theorem change_received_correct : change_received = 11 := by
  sorry

end NUMINAMATH_GPT_change_received_correct_l958_95819


namespace NUMINAMATH_GPT_mandy_chocolate_l958_95829

theorem mandy_chocolate (total : ℕ) (h1 : total = 60)
  (michael : ℕ) (h2 : michael = total / 2)
  (paige : ℕ) (h3 : paige = (total - michael) / 2) :
  (total - michael - paige = 15) :=
by
  -- By hypothesis: total = 60, michael = 30, paige = 15
  sorry 

end NUMINAMATH_GPT_mandy_chocolate_l958_95829


namespace NUMINAMATH_GPT_floor_eq_l958_95832

theorem floor_eq (r : ℝ) (h : ⌊r⌋ + r = 12.4) : r = 6.4 := by
  sorry

end NUMINAMATH_GPT_floor_eq_l958_95832


namespace NUMINAMATH_GPT_common_difference_is_3_l958_95844

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop := 
  a 3 + a 11 = 24

def condition2 (a : ℕ → ℝ) : Prop := 
  a 4 = 3

theorem common_difference_is_3 (h_arith : is_arithmetic a d) (h1 : condition1 a) (h2 : condition2 a) : 
  d = 3 := 
sorry

end NUMINAMATH_GPT_common_difference_is_3_l958_95844


namespace NUMINAMATH_GPT_parallelogram_area_leq_half_triangle_area_l958_95864

-- Definition of a triangle and a parallelogram inside it.
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

structure Parallelogram (α : Type) [LinearOrderedField α] :=
(P Q R S : α × α)

-- Function to calculate the area of a triangle
def triangle_area {α : Type} [LinearOrderedField α] (T : Triangle α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Function to calculate the area of a parallelogram
def parallelogram_area {α : Type} [LinearOrderedField α] (P : Parallelogram α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Statement of the problem
theorem parallelogram_area_leq_half_triangle_area {α : Type} [LinearOrderedField α]
(T : Triangle α) (P : Parallelogram α) (inside : P.P.1 < T.A.1 ∧ P.P.2 < T.C.1) : 
  parallelogram_area P ≤ 1 / 2 * triangle_area T :=
sorry

end NUMINAMATH_GPT_parallelogram_area_leq_half_triangle_area_l958_95864


namespace NUMINAMATH_GPT_salary_net_change_l958_95806

variable {S : ℝ}

theorem salary_net_change (S : ℝ) : (1.4 * S - 0.4 * (1.4 * S)) - S = -0.16 * S :=
by
  sorry

end NUMINAMATH_GPT_salary_net_change_l958_95806


namespace NUMINAMATH_GPT_percent_decrease_in_square_area_l958_95805

theorem percent_decrease_in_square_area (A B C D : Type) 
  (side_length_AD side_length_AB side_length_CD : ℝ) 
  (area_square_original new_side_length new_area : ℝ) 
  (h1 : side_length_AD = side_length_AB) (h2 : side_length_AD = side_length_CD) 
  (h3 : area_square_original = side_length_AD^2)
  (h4 : new_side_length = side_length_AD * 0.8)
  (h5 : new_area = new_side_length^2)
  (h6 : side_length_AD = 9) : 
  (area_square_original - new_area) / area_square_original * 100 = 36 := 
  by 
    sorry

end NUMINAMATH_GPT_percent_decrease_in_square_area_l958_95805


namespace NUMINAMATH_GPT_exists_non_degenerate_triangle_l958_95880

theorem exists_non_degenerate_triangle
  (l : Fin 7 → ℝ)
  (h_ordered : ∀ i j, i ≤ j → l i ≤ l j)
  (h_bounds : ∀ i, 1 ≤ l i ∧ l i ≤ 12) :
  ∃ i j k : Fin 7, i < j ∧ j < k ∧ l i + l j > l k ∧ l j + l k > l i ∧ l k + l i > l j := 
sorry

end NUMINAMATH_GPT_exists_non_degenerate_triangle_l958_95880


namespace NUMINAMATH_GPT_white_marbles_multiple_of_8_l958_95892

-- Definitions based on conditions
def blue_marbles : ℕ := 16
def num_groups : ℕ := 8

-- Stating the problem
theorem white_marbles_multiple_of_8 (white_marbles : ℕ) :
  (blue_marbles + white_marbles) % num_groups = 0 → white_marbles % num_groups = 0 :=
by
  sorry

end NUMINAMATH_GPT_white_marbles_multiple_of_8_l958_95892


namespace NUMINAMATH_GPT_square_of_distance_is_82_l958_95862

noncomputable def square_distance_from_B_to_center (a b : ℝ) : ℝ := a^2 + b^2

theorem square_of_distance_is_82
  (a b : ℝ)
  (r : ℝ := 11)
  (ha : a^2 + (b + 7)^2 = r^2)
  (hc : (a + 3)^2 + b^2 = r^2) :
  square_distance_from_B_to_center a b = 82 := by
  -- Proof steps omitted
  sorry

end NUMINAMATH_GPT_square_of_distance_is_82_l958_95862


namespace NUMINAMATH_GPT_combined_work_days_l958_95826

-- Definitions for the conditions
def work_rate (days : ℕ) : ℚ := 1 / days
def combined_work_rate (days_a days_b : ℕ) : ℚ :=
  work_rate days_a + work_rate days_b

-- Theorem to prove
theorem combined_work_days (days_a days_b : ℕ) (ha : days_a = 15) (hb : days_b = 30) :
  1 / (combined_work_rate days_a days_b) = 10 :=
by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_combined_work_days_l958_95826


namespace NUMINAMATH_GPT_curve_is_ellipse_with_foci_on_y_axis_l958_95813

theorem curve_is_ellipse_with_foci_on_y_axis (α : ℝ) (hα : 0 < α ∧ α < 90) :
  ∃ a b : ℝ, (0 < a) ∧ (0 < b) ∧ (a < b) ∧ 
  (∀ x y : ℝ, x^2 + y^2 * (Real.cos α) = 1 ↔ (x/a)^2 + (y/b)^2 = 1) :=
sorry

end NUMINAMATH_GPT_curve_is_ellipse_with_foci_on_y_axis_l958_95813


namespace NUMINAMATH_GPT_hide_and_seek_l958_95886

variables (A B V G D : Prop)

-- Conditions
def condition1 : Prop := A → (B ∧ ¬V)
def condition2 : Prop := B → (G ∨ D)
def condition3 : Prop := ¬V → (¬B ∧ ¬D)
def condition4 : Prop := ¬A → (B ∧ ¬G)

-- Problem statement:
theorem hide_and_seek :
  condition1 A B V →
  condition2 B G D →
  condition3 V B D →
  condition4 A B G →
  (B ∧ V ∧ D) :=
by
  intros h1 h2 h3 h4
  -- Proof would normally go here
  sorry

end NUMINAMATH_GPT_hide_and_seek_l958_95886


namespace NUMINAMATH_GPT_greatest_product_sum_300_l958_95831

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end NUMINAMATH_GPT_greatest_product_sum_300_l958_95831
