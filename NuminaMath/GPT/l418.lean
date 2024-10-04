import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Conformal.Complex
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.IsROrC
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Independence
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Vector
import Mathlib.Data.Zmod.Basic
import Mathlib.FieldTheory.Finite.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclid.Basic
import Mathlib.GroupTheory.Pigeonhole
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Real
import Real.Trigonometry
import data.real.basic
import tactic

namespace brian_needs_2_liters_of_milk_l418_418888

def brian_milk_problem : Prop :=
  let number_of_people := 8 in -- Brian, his wife, his two kids, his parents, his wife's parents
  let servings_per_person := 2 in
  let total_servings := number_of_people * servings_per_person in
  let milk_per_serving_in_cups := 1 / 2 in
  let total_milk_in_cups := total_servings * milk_per_serving_in_cups in
  let cup_to_ml := 250 in
  let total_milk_in_ml := total_milk_in_cups * cup_to_ml in
  let ml_to_l := 1000 in
  let total_milk_in_liters := total_milk_in_ml / ml_to_l in
  total_milk_in_liters = 2

theorem brian_needs_2_liters_of_milk : brian_milk_problem :=
by
  sorry

end brian_needs_2_liters_of_milk_l418_418888


namespace copy_pages_15_dollars_l418_418821

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l418_418821


namespace prime_squares_mod_180_l418_418253

theorem prime_squares_mod_180 (p : ℕ) (hp : prime p) (hp_gt_5 : p > 5) :
  ∃ (r1 r2 : ℕ), 
  r1 ≠ r2 ∧ 
  ∀ r : ℕ, (∃ m : ℕ, p^2 = m * 180 + r) → (r = r1 ∨ r = r2) :=
sorry

end prime_squares_mod_180_l418_418253


namespace hyperbola_properties_l418_418374

noncomputable def length_imaginary_axis (C : Type) [hyperbola_eq C] := 
  (2 * Real.sqrt 6)

noncomputable def eccentricity (C : Type) [hyperbola_eq C] := 
  (Real.sqrt 15) / 3

axiom hyperbola_eq : ∀(x y : ℝ), (y^2 / 9) - (x^2 / 6) = 1

theorem hyperbola_properties : 
  length_imaginary_axis hyperbola_eq = 2 * Real.sqrt 6 ∧ 
  eccentricity hyperbola_eq = Real.sqrt 15 / 3 :=
by 
  simp [length_imaginary_axis, eccentricity]
  split
  . refl
  . refl

end hyperbola_properties_l418_418374


namespace probability_two_primes_is_one_tenth_l418_418565

def is_prime : ℕ → Prop := λ n, ∀ d, 1 < d ∧ d < n → n % d ≠ 0
def primes_up_to_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
def total_ways_to_choose_2 : ℕ := Nat.choose 30 2
def ways_to_choose_2_primes : ℕ := Nat.choose 10 2
def probability_both_prime : ℚ := ways_to_choose_2_primes / total_ways_to_choose_2

theorem probability_two_primes_is_one_tenth : probability_both_prime = (1 : ℚ) / 10 :=
  by
    sorry

end probability_two_primes_is_one_tenth_l418_418565


namespace reflex_angle_at_H_l418_418323

/-- Four points C, D, F, and M are on a straight line.
    Point H is not on the line such that ∠CDH = 130° and ∠HFM = 70°.
    If the reflex angle at H is y°, then prove y = 340°. -/
theorem reflex_angle_at_H
  (C D F M H : Point)
  (h_collinear : collinear ({C, D, F, M} : set Point))
  (H_not_on_line : H ∉ line_through C D)
  (angle_CDH : ∠(C, D, H) = 130)
  (angle_HFM : ∠(H, F, M) = 70) :
  reflex_angle_at H = 340 :=
sorry

end reflex_angle_at_H_l418_418323


namespace odd_times_even_is_even_l418_418633

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_times_even_is_even (a b : ℤ) (h₁ : is_odd a) (h₂ : is_even b) : is_even (a * b) :=
by sorry

end odd_times_even_is_even_l418_418633


namespace compute_f_2007_l418_418459

variable A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1}
variable f : A → ℝ
variable x : A

axiom functional_equation : ∀ x : ℚ, x ∈ A → f ⟨x, _⟩ + f ⟨1 - 1/x, _⟩ = Real.log (Real.abs x)

theorem compute_f_2007 : f ⟨2007, sorry⟩ = Real.log (2007 ^ 2 / 2006) :=
sorry

end compute_f_2007_l418_418459


namespace number_of_valid_ns_l418_418447

theorem number_of_valid_ns (N a : ℤ) (p : ℕ → ℤ) (b : ℕ → ℤ) (k : ℕ)
  (hN : N = 2^a * ∏ i in finset.range (k + 1), p i ^ b i) :
  let valid_n (n : ℤ) := (n * (n + 1) / 2 ≤ N) ∧ (N - n * (n + 1) / 2) % n = 0 in
  finset.card (finset.filter valid_n (finset.range N)) = ∏ i in finset.range (k + 1), (b i + 1) :=
sorry

end number_of_valid_ns_l418_418447


namespace total_amount_shared_l418_418240

theorem total_amount_shared (a b c : ℕ) (h_ratio : a = 3 * b / 5 ∧ c = 9 * b / 5) (h_b : b = 50) : a + b + c = 170 :=
by sorry

end total_amount_shared_l418_418240


namespace largest_two_digit_divisible_by_6_ending_in_4_l418_418926

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l418_418926


namespace smaller_angle_36_degrees_l418_418141

noncomputable def smaller_angle_measure (larger smaller : ℝ) : Prop :=
(larger + smaller = 180) ∧ (larger = 4 * smaller)

theorem smaller_angle_36_degrees : ∃ (smaller : ℝ), smaller_angle_measure (4 * smaller) smaller ∧ smaller = 36 :=
by
  sorry

end smaller_angle_36_degrees_l418_418141


namespace work_done_together_l418_418597

theorem work_done_together
    (A_time : ℝ) (B_time : ℝ) (A_rate : ℝ) (B_rate : ℝ) (work_done_together_rate : ℝ) :
    A_time = 18 ∧ B_time = A_time / 2 ∧ 
    A_rate = 1 / A_time ∧ B_rate = 1 / B_time ∧ 
    work_done_together_rate = A_rate + B_rate → work_done_together_rate = 1 / 6 :=
by
  intros h
  rw [h.1, h.2.1, h.2.2.1, h.2.2.2, h.2.2.2]
  have B_time_calc : B_time = 9 := by
    rw [h.1] at h.2.1
    exact h.2.1
  have A_rate_calc : A_rate = 1 / 18 := by
    rw [h.1] at h.2.2.1
    exact h.2.2.1
  have B_rate_calc : B_rate = 1 / 9 := by
    rw [B_time_calc] at h.2.2.2
    exact h.2.2.2
  have total_rate_calc : work_done_together_rate = 1 / 18 + 1 / 9 := by
    rw [A_rate_calc, B_rate_calc]
    exact h.2.2.2
  have common_denom : 1 / 18 + 1 / 9 = 1 / 6 := by
    rw [add_comm (1 / 18) (1 / 9)]
    norm_num
  rw [total_rate_calc, common_denom]
  exact rfl

end work_done_together_l418_418597


namespace convex_octagon_four_acute_angles_l418_418157

-- Definitions from conditions
def octagon_interior_angle_sum : ℝ := 1080
def is_acute (angle : ℝ) : Prop := angle < 90
def is_convex_angle (angle : ℝ) : Prop := angle < 180

-- Lean statement: Prove that a convex octagon cannot have more than four acute angles
theorem convex_octagon_four_acute_angles : 
  ∀ (angles : list ℝ), (angles.length = 8) → (∀ angle ∈ angles, is_convex_angle angle) → 
  (∑ angle in angles, angle = octagon_interior_angle_sum) → 
  (∃ (n : ℕ), n ≤ 4 ∧ (countp is_acute angles = n)) :=
by
  sorry

end convex_octagon_four_acute_angles_l418_418157


namespace sequence_exists_l418_418315

variable (n : ℕ)

theorem sequence_exists (k : ℕ) (hkn : k ∈ Set.range (λ x : ℕ, x + 1) n) :
  ∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i → i ≤ n → x (i+1) > x i) ∧ (∀ i, x i ∈ ℕ) :=
sorry

end sequence_exists_l418_418315


namespace sum_of_roots_of_quadratic_l418_418163

theorem sum_of_roots_of_quadratic : 
  (∀ x : ℝ, x^2 - 7 * x + 12 = 0) → 
  ∑ x in { x : ℝ | x^2 - 7 * x + 12 = 0 }, x = 7 :=
sorry

end sum_of_roots_of_quadratic_l418_418163


namespace greatest_value_of_squares_exists_max_value_of_squares_l418_418449

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
sorry

theorem exists_max_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 702 :=
sorry

end greatest_value_of_squares_exists_max_value_of_squares_l418_418449


namespace find_fourth_intersection_point_l418_418416

theorem find_fourth_intersection_point :
  ∃ (x y : ℝ), (x * y = 2) ∧
    ((x, y) = (3, 2 / 3) ∨ (x, y) = (-4, -1 / 2) ∨ (x, y) = (1 / 4, 8) ∨ (x, y) = (-2 / 3, -3)) ∧
    xy_elliptic_condition x y :=
sorry

/--
  The condition that (x, y) lies on the ellipse passing through the given three points.
-/
def xy_elliptic_condition (x y : ℝ) : Prop :=
  ∃ (h k a b : ℝ), 
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ∧
  ((3 - h)^2 / a^2 + (2 / 3 - k)^2 / b^2 = 1) ∧
  ((-4 - h)^2 / a^2 + (-1 / 2 - k)^2 / b^2 = 1) ∧
  ((1 / 4 - h)^2 / a^2 + (8 - k)^2 / b^2 = 1)

end find_fourth_intersection_point_l418_418416


namespace binom_18_10_l418_418654

theorem binom_18_10 :
  (nat.choose 18 10) = 45760 :=
by
  have h1 : (nat.choose 16 7) = 11440 := sorry
  have h2 : (nat.choose 16 9) = 11440 := sorry
  sorry

end binom_18_10_l418_418654


namespace eggs_at_park_l418_418759

-- Define the number of eggs found at different locations
def eggs_at_club_house : Nat := 40
def eggs_at_town_hall : Nat := 15
def total_eggs_found : Nat := 80

-- Prove that the number of eggs found at the park is 25
theorem eggs_at_park :
  ∃ P : Nat, eggs_at_club_house + P + eggs_at_town_hall = total_eggs_found ∧ P = 25 := 
by
  sorry

end eggs_at_park_l418_418759


namespace remainder_when_2x_div_8_is_1_l418_418579

theorem remainder_when_2x_div_8_is_1 (x y : ℤ) 
  (h1 : x = 11 * y + 4)
  (h2 : ∃ r : ℤ, 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) : ∃ r : ℤ, r = 1 :=
by
  sorry

end remainder_when_2x_div_8_is_1_l418_418579


namespace cylinder_surface_area_l418_418227

def height : ℝ := 8
def radius : ℝ := 3
def lateral_surface_area : ℝ := 2 * Real.pi * radius * height
def top_bottom_area : ℝ := 2 * Real.pi * radius^2
def total_surface_area : ℝ := lateral_surface_area + top_bottom_area

theorem cylinder_surface_area : total_surface_area = 66 * Real.pi := by
  sorry

end cylinder_surface_area_l418_418227


namespace smaller_angle_36_degrees_l418_418142

noncomputable def smaller_angle_measure (larger smaller : ℝ) : Prop :=
(larger + smaller = 180) ∧ (larger = 4 * smaller)

theorem smaller_angle_36_degrees : ∃ (smaller : ℝ), smaller_angle_measure (4 * smaller) smaller ∧ smaller = 36 :=
by
  sorry

end smaller_angle_36_degrees_l418_418142


namespace product_of_digits_largest_integer_l418_418997

theorem product_of_digits_largest_integer : 
  ∃ (n : ℕ), (sum_of_squares_of_digits n = 85 ∧ digits_in_order n) → product_of_digits (largest_integer_with_properties) = 48 :=
by 
  sorry

-- Definitions based on conditions
def sum_of_squares_of_digits (n : ℕ) : ℕ :=
    -- code for calculating sum of squares of digits
    
def digits_in_order (n : ℕ) : Prop :=
    -- code to check if digits are in increasing order
    
def largest_integer_with_properties : ℕ :=
    -- code to find the largest integer with the required properties

def product_of_digits (n : ℕ) : ℕ :=
    -- code for calculating the product of digits

end product_of_digits_largest_integer_l418_418997


namespace flight_time_l418_418690

variable (V₀ : ℝ) (g : ℝ)

def max_flight_time : ℝ :=
  let theta := Real.arcsin 0.96
  let alpha := (Real.pi - theta) / 2
  (2 * V₀ / g) * Real.cos(alpha)

theorem flight_time (hV₀ : V₀ = 10) (hg : g = 10) : 
  max_flight_time V₀ g = 1.6 :=
by
  sorry

end flight_time_l418_418690


namespace stock_price_percentage_increase_l418_418172

theorem stock_price_percentage_increase :
  ∀ (total higher lower : ℕ), 
    total = 1980 →
    higher = 1080 →
    higher > lower →
    lower = total - higher →
  ((higher - lower) / lower : ℚ) * 100 = 20 :=
by
  intros total higher lower total_eq higher_eq higher_gt lower_eq
  sorry

end stock_price_percentage_increase_l418_418172


namespace largestValidNumberIs84_l418_418901

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l418_418901


namespace sum_of_squares_const_centroid_does_not_move_midpoint_loci_concentric_l418_418629

-- Given definitions of the geometric entities and conditions
variables {α : Type*} [MetricSpace α]
  (O : α) (r R : ℝ) (P A B C : α)
  (h_center : ∀ x, dist O x = r ∨ dist O x = R)
  (h_P_on_smaller_circle : dist O P = r)
  (h_perpendicular_chords : ∀ Q, dist P Q = 0 → ∃ A B C, ∠ A P B = π/2 ∧ ∠ B P C = π/2)
  
-- The assertions to prove
theorem sum_of_squares_const : PA^2 + PB^2 + PC^2 = k :=
sorry

theorem centroid_does_not_move : ∀ G, centroid O A B C G → centroid_does_not_move :=
sorry

theorem midpoint_loci_concentric :
  ∃ T R S,
    (midpoint α A B T ∧ midpoint α B C R ∧ midpoint α C A S) →
    loci_on_concentric_circles :=
sorry

end sum_of_squares_const_centroid_does_not_move_midpoint_loci_concentric_l418_418629


namespace find_rate_of_current_l418_418548

-- The conditions given in natural language
def boat_speed : ℝ := 20
def distance_downstream : ℝ := 10
def time_downstream : ℝ := 24 / 60  -- convert minutes to hours

def rate_of_current (c : ℝ) : Prop :=
  distance_downstream = (boat_speed + c) * time_downstream

-- The proof problem, i.e., finding the rate of current
theorem find_rate_of_current : ∃ c, rate_of_current c :=
begin
  use 5,
  unfold rate_of_current,
  norm_num,
end

end find_rate_of_current_l418_418548


namespace train_length_l418_418624

-- The given conditions
def speed_kmh : ℝ := 96
def bridge_length : ℝ := 300
def time_seconds : ℝ := 15

-- Conversion factor from km/h to m/s
def speed_mps : ℝ := speed_kmh * (1000 / 3600)

-- The statement to prove
theorem train_length :
  ∃ (L : ℝ), L + bridge_length = speed_mps * time_seconds ∧ L = 100.05 :=
by
  use 100.05
  split
  -- First part
  show 100.05 + bridge_length = speed_mps * time_seconds
  sorry
  -- Second part is an equality assertion
  show 100.05 = 100.05
  rfl

end train_length_l418_418624


namespace sequence_product_consecutive_integers_l418_418491

theorem sequence_product_consecutive_integers (n : ℕ) :
  let term := (10^n - 1) / 3 * (10^n + 2) / 3
  in (term =
      let str_n_ones := (Nat.repeat '1' n). Nat.parseInt!
      let str_n_twos := (Nat.repeat '2' n). Nat.parseInt!
      str_n_ones * Nat.shiftLeft str_n_twos n
     ) :=
sorry

end sequence_product_consecutive_integers_l418_418491


namespace vowel_soup_sequences_l418_418238

theorem vowel_soup_sequences :
  let A := 8
  let E := 6
  let I := 7
  let O := 5
  let U := 4
  ∃ S : Finset (Fin 5), (∀ x ∈ S, ∃ (a e i o u : ℕ), a + e + i + o + u = 5 ∧ 
    a ≥ 1 ∧ e ≥ 1 ∧ i ≥ 1 ∧ o ≥ 1 ∧ u ≥ 1 ∧ a ≤ A ∧ e ≤ E ∧ i ≤ I ∧ o ≤ O ∧ u ≤ U),
  S.card = 120 := by
sorry

end vowel_soup_sequences_l418_418238


namespace find_rate_of_interest_l418_418621

-- Conditions
def SI : ℝ := 4016.25
def P : ℝ := 8925
def T : ℝ := 5

-- Given the formula for simple interest
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- The proof problem statement
theorem find_rate_of_interest (R : ℝ) (h : simple_interest P R T = SI) : R = 9 :=
by
  sorry

end find_rate_of_interest_l418_418621


namespace solve_equation_l418_418544

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) :
    (3 / (x + 2) - 1 / x = 0) → x = 1 :=
  by sorry

end solve_equation_l418_418544


namespace p_is_x_squared_plus_1_l418_418292

noncomputable def p : ℝ → ℝ := sorry

lemma polynomial_property (x y : ℝ) : p(x) * p(y) = p(x) + p(y) + p(x * y) - 3 := sorry

lemma p_at_3 : p 3 = 10 := sorry

theorem p_is_x_squared_plus_1 (x : ℝ) : p x = x^2 + 1 :=
sorry

end p_is_x_squared_plus_1_l418_418292


namespace mallory_travel_expenses_l418_418112

theorem mallory_travel_expenses (fuel_tank_cost : ℕ) (fuel_tank_miles : ℕ) (total_miles : ℕ) (food_ratio : ℚ)
  (h_fuel_tank_cost : fuel_tank_cost = 45)
  (h_fuel_tank_miles : fuel_tank_miles = 500)
  (h_total_miles : total_miles = 2000)
  (h_food_ratio : food_ratio = 3/5) :
  ∃ total_cost : ℕ, total_cost = 288 :=
by
  sorry

end mallory_travel_expenses_l418_418112


namespace calculate_correctly_l418_418427

theorem calculate_correctly (n : ℕ) (h1 : n - 21 = 52) : n - 40 = 33 := 
by 
  sorry

end calculate_correctly_l418_418427


namespace power_function_increasing_is_3_l418_418721

theorem power_function_increasing_is_3 (m : ℝ) :
  (∀ x : ℝ, x > 0 → (m^2 - m - 5) * (x^(m)) > 0) ∧ (m^2 - m - 5 = 1) → m = 3 :=
by
  sorry

end power_function_increasing_is_3_l418_418721


namespace axis_of_symmetry_l418_418737

noncomputable def function_y (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

theorem axis_of_symmetry :
  ∃ φ, |φ| < Real.pi / 2 ∧ function_y 0 φ = Real.sqrt 3 ∧ 
    (∀ k : ℤ, (λ x, function_y x φ) = (λ x, function_y (x + k * Real.pi / 2 + Real.pi / 12) φ) ∨ (λ x, function_y x φ) = (λ x, function_y (x - k * Real.pi / 2 - Real.pi / 12) φ)) :=
sorry

end axis_of_symmetry_l418_418737


namespace polynomial_value_at_minus_two_l418_418150

def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

theorem polynomial_value_at_minus_two : f (-2) = 12 := by 
  sorry

end polynomial_value_at_minus_two_l418_418150


namespace sue_nuts_count_l418_418261

theorem sue_nuts_count (B H S : ℕ) 
  (h1 : B = 6 * H) 
  (h2 : H = 2 * S) 
  (h3 : B + H = 672) : S = 48 := 
by
  sorry

end sue_nuts_count_l418_418261


namespace sum_distinct_negative_prime_roots_constant_term_l418_418635

theorem sum_distinct_negative_prime_roots_constant_term :
  let primes := [2, 3, 5, 7]
  let pairs := [(2, 3), (2, 5), (2, 7), (3, 5), (3, 7), (5, 7)]
  let products := pairs.map (λ pair, pair.1 * pair.2)
  products.sum = 101 :=
by 
  let primes := [2, 3, 5, 7]
  let pairs := [(2, 3), (2, 5), (2, 7), (3, 5), (3, 7), (5, 7)]
  let products := pairs.map (λ pair, pair.1 * pair.2)
  have products_eq : products = [6, 10, 14, 15, 21, 35] := by decide
  rw products_eq
  norm_num

end sum_distinct_negative_prime_roots_constant_term_l418_418635


namespace solution_set_f_inequality_l418_418519

noncomputable def f : ℝ → ℝ := sorry
lemma f_odd : ∀ x : ℝ, f (-x) = -f x := sorry
lemma f'_positive : ∀ x : ℝ, x < 0 → f' x > 0 := sorry
lemma f_neg_half_zero : f (-1 / 2) = 0 := sorry

theorem solution_set_f_inequality :
  {x : ℝ | f x < 0} = {x : ℝ | x < -1 / 2 ∨ (0 < x ∧ x < 1 / 2)} :=
by sorry

end solution_set_f_inequality_l418_418519


namespace total_journey_time_l418_418405

def river_speed := 5 -- km/hr
def upstream_distance := 80 -- km
def downstream_distance := 100 -- km
def initial_boat_speed := 12 -- km/hr
def obstacle_count := 3
def speed_reduction_per_obstacle := 0.5 -- km/hr

theorem total_journey_time :
  let speed_upstream := initial_boat_speed - river_speed in
  let time_upstream := upstream_distance / speed_upstream in
  let speed_downstream_initial := initial_boat_speed + river_speed in
  let total_speed_reduction := obstacle_count * speed_reduction_per_obstacle in
  let speed_downstream_final := speed_downstream_initial - total_speed_reduction in
  let time_downstream := downstream_distance / speed_downstream_final in
  time_upstream + time_downstream = 17.88 :=
by
  let speed_upstream := initial_boat_speed - river_speed
  let time_upstream := upstream_distance / speed_upstream
  let speed_downstream_initial := initial_boat_speed + river_speed
  let total_speed_reduction := obstacle_count * speed_reduction_per_obstacle
  let speed_downstream_final := speed_downstream_initial - total_speed_reduction
  let time_downstream := downstream_distance / speed_downstream_final
  have h1 : time_upstream = 80 / 7 := sorry
  have h2 : time_downstream = 100 / 15.5 := sorry
  have h3 : time_upstream + time_downstream = 11.43 + 6.45 := sorry
  have h4 : 11.43 + 6.45 = 17.88 := sorry
  exact h4

end total_journey_time_l418_418405


namespace base_seven_sum_of_digits_eq_eight_l418_418646

noncomputable def base_seven_to_base_ten (n : Nat) : Nat :=
  let digits : List Nat := 
    -- Decomposes base-7 number into its digits
    (List.unfoldr (λ x => if x = 0 then none else some (x % 10, x / 10))) n
  digits.enumFrom 0
         |>.foldl (λ acc ⟨i, d⟩ => acc + d * 7 ^ i) 0

noncomputable def base_ten_to_base_seven (n : Nat) : Nat :=
  -- Converts base-10 number to base-7 representation
  Nat.ofDigits 7 (Nat.digits 7 n)

noncomputable def sum_of_digits_base_seven_product : Nat :=
  let n1 := base_seven_to_base_ten 35
  let n2 := base_seven_to_base_ten 13
  let product := n1 * n2
  let base7_product := base_ten_to_base_seven product
  -- Sum the digits of base7_product
  (List.unfoldr
       (λ x => if x = 0 then none else some (x % 10, x / 10)) base7_product)
    |>.foldl (· + ·) 0

theorem base_seven_sum_of_digits_eq_eight :
  sum_of_digits_base_seven_product = 8 :=
sorry

end base_seven_sum_of_digits_eq_eight_l418_418646


namespace Fr_zero_all_r_l418_418033

variables {x y z A B C : ℝ}
variable {n : ℕ}

def F (r : ℕ) : ℝ := x ^ r * Real.sin (r * A) + y ^ r * Real.sin (r * B) + z ^ r * Real.sin (r * C)

theorem Fr_zero_all_r (h₁ : F 1 = 0) (h₂ : F 2 = 0) (hABC : ∃ k : ℤ, A + B + C = k * Real.pi) : ∀ r : ℕ, F r = 0 := 
sorry

end Fr_zero_all_r_l418_418033


namespace initial_percentage_liquid_X_l418_418619

theorem initial_percentage_liquid_X (P : ℝ) :
  let original_solution_kg := 8
  let evaporated_water_kg := 2
  let added_solution_kg := 2
  let remaining_solution_kg := original_solution_kg - evaporated_water_kg
  let new_solution_kg := remaining_solution_kg + added_solution_kg
  let new_solution_percentage := 0.25
  let initial_liquid_X_kg := (P / 100) * original_solution_kg
  let final_liquid_X_kg := initial_liquid_X_kg + (P / 100) * added_solution_kg
  let final_liquid_X_kg' := new_solution_percentage * new_solution_kg
  (final_liquid_X_kg = final_liquid_X_kg') → 
  P = 20 :=
by
  intros
  let original_solution_kg_p0 := 8
  let evaporated_water_kg_p1 := 2
  let added_solution_kg_p2 := 2
  let remaining_solution_kg_p3 := (original_solution_kg_p0 - evaporated_water_kg_p1)
  let new_solution_kg_p4 := (remaining_solution_kg_p3 + added_solution_kg_p2)
  let new_solution_percentage : ℝ := 0.25
  let initial_liquid_X_kg_p6 := ((P / 100) * original_solution_kg_p0)
  let final_liquid_X_kg_p7 := initial_liquid_X_kg_p6 + ((P / 100) * added_solution_kg_p2)
  let final_liquid_X_kg_p8 := (new_solution_percentage * new_solution_kg_p4)
  exact sorry

end initial_percentage_liquid_X_l418_418619


namespace last_integer_in_sequence_is_one_l418_418617

theorem last_integer_in_sequence_is_one :
  ∀ seq : ℕ → ℕ, (seq 0 = 37) ∧ (∀ n, seq (n + 1) = seq n / 2) → (∃ n, seq (n + 1) = 0 ∧ seq n = 1) :=
by
  sorry

end last_integer_in_sequence_is_one_l418_418617


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l418_418940

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l418_418940


namespace pages_can_be_copied_l418_418829

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l418_418829


namespace rate_of_change_of_alpha_l418_418632

noncomputable def d_alpha_dt (n : ℝ) : ℝ :=
  let theta := π / 6
  let t := Real.sqrt(theta / 6)
  let dtheta_dt := 2 * t
  let dtheta_dt_minus_dalpha_dt := (dtheta_dt - (Real.sqrt(3 * π / 12))) in
  dtheta_dt_minus_dalpha_dt

theorem rate_of_change_of_alpha :
  d_alpha_dt 1.50 ≈ 0.561 :=
sorry

end rate_of_change_of_alpha_l418_418632


namespace rowing_distance_upstream_l418_418218

theorem rowing_distance_upstream 
  (v : ℝ) (d : ℝ)
  (h1 : 75 = (v + 3) * 5)
  (h2 : d = (v - 3) * 5) :
  d = 45 :=
by {
  sorry
}

end rowing_distance_upstream_l418_418218


namespace years_passed_l418_418551

noncomputable def initial_value : ℝ := 40000
noncomputable def final_value : ℝ := 12656.25
noncomputable def depreciation_rate : ℝ := 3 / 4

theorem years_passed : 
  ∃ n : ℕ, final_value = initial_value * depreciation_rate ^ n ∧ n = 4 :=
by
  use 4
  field_simp
  norm_num
  apply pow_succ
  sorry

end years_passed_l418_418551


namespace data_instances_in_one_hour_l418_418437

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l418_418437


namespace john_fixes_8_computers_l418_418012

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end john_fixes_8_computers_l418_418012


namespace ellipse_chord_through_focus_l418_418631

theorem ellipse_chord_through_focus (x y : ℝ) (a b : ℝ := 6) (c : ℝ := 3 * Real.sqrt 3)
  (F : ℝ × ℝ := (3 * Real.sqrt 3, 0)) (AF BF : ℝ) :
  (x^2 / 36) + (y^2 / 9) = 1 ∧ ((x - 3 * Real.sqrt 3)^2 + y^2 = (3/2)^2) ∧
  (AF = 3 / 2) ∧ F.1 = 3 * Real.sqrt 3 ∧ F.2 = 0 →
  BF = 3 / 2 :=
sorry

end ellipse_chord_through_focus_l418_418631


namespace range_of_m_l418_418722

noncomputable def f (x m : ℝ) : ℝ := x ^ 2 - 2 * m * x + 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x < 1 → (2 * x - 2 * m < 0)) ↔ m ∈ set.Ici 1 :=
begin
  sorry
end

end range_of_m_l418_418722


namespace simplify_fraction_l418_418063

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l418_418063


namespace surface_area_l418_418230

theorem surface_area (r : ℝ) (π : ℝ) (V : ℝ) (S : ℝ) 
  (h1 : V = 48 * π) 
  (h2 : V = (4 / 3) * π * r^3) : 
  S = 4 * π * r^2 :=
  sorry

end surface_area_l418_418230


namespace mass_percentage_of_Ba_in_BaF2_l418_418673

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_F : ℝ := 19.00

def mass_percentage_Ba : ℝ :=
  let molar_mass_BaF2 := molar_mass_Ba + 2 * molar_mass_F in
  (molar_mass_Ba / molar_mass_BaF2) * 100

theorem mass_percentage_of_Ba_in_BaF2 : mass_percentage_Ba = 78.35 :=
by
  sorry

end mass_percentage_of_Ba_in_BaF2_l418_418673


namespace pipe_B_filling_time_l418_418567

-- Definitions and assumptions based on the conditions
def rate_A : ℝ := 1 / 36
def rate_combined : ℝ := 1 / 20.195121951219512

-- The required proof problem statement
theorem pipe_B_filling_time :
  (∃ T_B : ℝ, 1 / T_B = rate_combined - rate_A) →
  ∃ T_B : ℝ, abs(T_B - 46) < 1 :=
sorry

end pipe_B_filling_time_l418_418567


namespace find_remainder_l418_418204

-- Conditions
variables (O A B C D P E F : Point)
variables (r : ℝ) (chordAB chordCD d : ℝ)
variables (m n : ℕ)

-- Circle and Chords
def Circle := ∀ (x : Point), dist x O = r
def Chord (x y : Point) := dist x y
def Midpoint (x y midpoint : Point) := dist x midpoint = (dist x y) / 2 ∧ dist y midpoint = (dist x y) / 2

-- Conditions assumptions
axiom h1 : Circle O
axiom h2 : chordAB = 30
axiom h3 : chordCD = 14
axiom h4 : Midpoint A B E
axiom h5 : Midpoint C D F
axiom h6 : dist E F = 12
axiom h7 : dist O E = 20
axiom h8 : dist O F = 24

-- Proof statement
theorem find_remainder : 
  (∀ (OP EP FP : ℝ), OP ^ 2 = EP ^ 2 + 400 ∧ OP ^ 2 = FP ^ 2 + 576 ∧ (dist E F) ^ 2 = EP ^ 2 + FP ^ 2 - 2 * EP * FP * cos(∠ EPF) 
   → OP ^ 2 = 4050 / 7) → (4050 + 7) % 1000 = 57 :=
sorry

end find_remainder_l418_418204


namespace A_share_of_profit_eq_357_l418_418195

-- Conditions
variables (initial_investment_A initial_investment_B : ℝ)
variables (withdrawal_A advance_B : ℝ)
variables (months_before_change months_after_change : ℕ)
variables (total_profit : ℝ)

-- Initial conditions
def investment_A_before_change := initial_investment_A * months_before_change
def investment_B_before_change := initial_investment_B * months_before_change
def investment_A_after_change := (initial_investment_A - withdrawal_A) * months_after_change
def investment_B_after_change := (initial_investment_B + advance_B) * months_after_change

-- Total investments over the year
def total_investment_A := investment_A_before_change initial_investment_A initial_investment_B months_before_change + investment_A_after_change initial_investment_A withdrawal_A months_after_change
def total_investment_B := investment_B_before_change initial_investment_B months_before_change + investment_B_after_change initial_investment_B advance_B months_after_change

-- Ratio of investments
def ratio_A_B := total_investment_A initial_investment_A initial_investment_B months_before_change withdrawal_A months_after_change / 
                 total_investment_B initial_investment_B months_before_change advance_B months_after_change

-- A's share calculation
def A_share := (total_profit * (total_investment_A initial_investment_A initial_investment_B months_before_change withdrawal_A months_after_change / (total_investment_A initial_investment_A initial_investment_B months_before_change withdrawal_A months_after_change + total_investment_B initial_investment_B months_before_change advance_B months_after_change)))

-- Theorem
theorem A_share_of_profit_eq_357 
  (h1 : initial_investment_A = 6000)
  (h2 : initial_investment_B = 4000)
  (h3 : withdrawal_A = 1000)
  (h4 : advance_B = 1000)
  (h5 : months_before_change = 8)
  (h6 : months_after_change = 4)
  (h7 : total_profit = 630) :
  A_share initial_investment_A initial_investment_B withdrawal_A advance_B months_before_change months_after_change total_profit = 357 :=
by sorry

end A_share_of_profit_eq_357_l418_418195


namespace prime_square_remainder_l418_418251

theorem prime_square_remainder (p : ℕ) (hp : Nat.Prime p) (h5 : p > 5) : 
  ∃! r : ℕ, r < 180 ∧ (p^2 ≡ r [MOD 180]) := 
by
  sorry

end prime_square_remainder_l418_418251


namespace largestValidNumberIs84_l418_418899

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l418_418899


namespace egyptian_fraction_decomposition_l418_418787

theorem egyptian_fraction_decomposition (n : ℕ) (hn : 0 < n) : 
  (2 : ℚ) / (2 * n + 1) = (1 : ℚ) / (n + 1) + (1 : ℚ) / ((n + 1) * (2 * n + 1)) := 
by {
  sorry
}

end egyptian_fraction_decomposition_l418_418787


namespace sally_sours_total_l418_418495

theorem sally_sours_total (cherry_sours lemon_sours orange_sours total_sours : ℕ) 
    (h1 : cherry_sours = 32)
    (h2 : 5 * cherry_sours = 4 * lemon_sours)
    (h3 : orange_sours = total_sours / 4)
    (h4 : cherry_sours + lemon_sours + orange_sours = total_sours) : 
    total_sours = 96 :=
by
  rw [h1] at h2
  have h5 : lemon_sours = 40 := by linarith
  rw [h1, h5] at h4
  have h6 : orange_sours = total_sours / 4 := by assumption
  rw [h6] at h4
  have h7 : 72 + total_sours / 4 = total_sours := by linarith
  sorry

end sally_sours_total_l418_418495


namespace find_z_l418_418541

open Real

variable {z : ℝ}

def vec1 : ℝ × ℝ × ℝ := (4, -1, z)
def vec2 : ℝ × ℝ × ℝ := (6, -2, 3)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dp_uv := dot_product u v
  let dp_vv := dot_product v v
  let factor := dp_uv / dp_vv
  (factor * v.1, factor * v.2, factor * v.3)

theorem find_z (h : projection vec1 vec2 = (20 / 49 * 6, 20 / 49 * -2, 20 / 49 * 3)) : z = -2 := by
  sorry

end find_z_l418_418541


namespace exists_sequence_for_k_l418_418313

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end exists_sequence_for_k_l418_418313


namespace f_is_odd_and_periodic_l418_418980

noncomputable def f (x : ℝ) : ℝ :=
  sin (x - (Real.pi / 4)) * cos (x + (Real.pi / 4)) + 1 / 2

theorem f_is_odd_and_periodic :
  (∀ x, f (-x) = -f x) ∧ (∃ p, p > 0 ∧ ∀ x, f (x + p) = f x) ∧ (∀ k, k > 0 → ∀ x, f (x + k) = f x → k ≥ Real.pi) :=
by
  sorry

end f_is_odd_and_periodic_l418_418980


namespace solve_recurrence_l418_418956

noncomputable def recurrence_relation (a : ℕ → ℤ) : Prop :=
∀ (n : ℕ), n ≥ 4 → a n = 8 * a (n - 1) - 22 * a (n - 2) + 24 * a (n - 3) - 9 * a (n - 4)

lemma initial_conditions (a : ℕ → ℤ) : a 0 = -1 ∧ a 1 = -3 ∧ a 2 = -5 ∧ a 3 = 5 :=
  by sorry

theorem solve_recurrence (a : ℕ → ℤ) :
  recurrence_relation a ∧ initial_conditions a →
  ∀ n, a n = 2 + n - 3^(n + 1) + n * 3^n :=
by sorry

end solve_recurrence_l418_418956


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418933

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418933


namespace completing_the_square_l418_418149

theorem completing_the_square (x : ℝ) (h : x^2 - 6 * x + 7 = 0) : (x - 3)^2 - 2 = 0 := 
by sorry

end completing_the_square_l418_418149


namespace max_n_value_l418_418462

theorem max_n_value (k : ℕ) (h_k : k ≥ 2) (a : Fin k → ℕ) (h_a_pos : ∀ i, 0 < a i) 
  (h_n_int : (∑ i, a i) ^ 2 % (∏ i, a i) = 0) : 
  ∃ (n : ℕ), ∀ (a : Fin k → ℕ), (∑ i, a i) ^ 2 / (∏ i, a i) ≤ k^2 := sorry

end max_n_value_l418_418462


namespace exists_circle_through_vertex_of_triangle_l418_418220

-- Define a non-isosceles triangle ABC
noncomputable def triangle_non_isosceles (A B C : Point) : Prop :=
  ¬(A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ (AB = AC ∨ AB = BC ∨ AC = BC))

-- Define the incenter I of triangle ABC
def incenter (A B C I : Point) : Prop := 
  is_incenter I A B C

-- Define points D, E, F where the incircle touches BC, CA, AB respectively
def incircle_contact_points (A B C D E F : Point) (I : Point) : Prop :=
  is_incenter I A B C ∧ touches_incircle_at D BC ∧ touches_incircle_at E CA ∧ touches_incircle_at F AB

-- Define the excenter A'
def excenter_opposite_A (A B C A' : Point) : Prop :=
  is_excenter_opposite A' A B C

-- Define the incircle, circumcircle and A-excircle
def circles (A B C I A' : Point) (omega Omega omega_A : Circle) : Prop :=
  is_incircle omega A B C I ∧ is_circumcircle Omega A B C ∧ is_excircle_opposite omega_A A' A B C

-- Proof statement
theorem exists_circle_through_vertex_of_triangle 
  (A B C I D E F A' : Point) 
  (omega Omega omega_A : Circle) :
  triangle_non_isosceles A B C →
  incenter A B C I →
  incircle_contact_points A B C D E F I →
  excenter_opposite_A A B C A' →
  circles A B C I A' omega Omega omega_A →
  ∃ (gamma : Circle) (P : Point), 
    (circle_tangent_to gamma omega) ∧ 
    (circle_tangent_to gamma Omega) ∧ 
    (circle_tangent_to gamma omega_A) ∧ 
    (P = A ∨ P = B ∨ P = C) :=
sorry

end exists_circle_through_vertex_of_triangle_l418_418220


namespace simplify_fraction_l418_418078

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l418_418078


namespace tom_age_ratio_l418_418562

theorem tom_age_ratio (T N : ℕ) 
  (h1 : T = T)
  (h2 : T - N = 3 * (T - 5 * N)) : T / N = 7 :=
by sorry

end tom_age_ratio_l418_418562


namespace simplify_fraction_l418_418082

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l418_418082


namespace simplify_expression_l418_418575

theorem simplify_expression : 
  -2^2003 + (-2)^2004 + 2^2005 - 2^2006 = -3 * 2^2003 := 
  by
    sorry

end simplify_expression_l418_418575


namespace sanda_exercise_each_day_l418_418010

def exercise_problem (javier_exercise_daily sanda_exercise_total total_minutes : ℕ) (days_in_week : ℕ) :=
  javier_exercise_daily * days_in_week + sanda_exercise_total = total_minutes

theorem sanda_exercise_each_day 
  (javier_exercise_daily : ℕ := 50)
  (days_in_week : ℕ := 7)
  (total_minutes : ℕ := 620)
  (days_sanda_exercised : ℕ := 3): 
  ∃ (sanda_exercise_each_day : ℕ), exercise_problem javier_exercise_daily (sanda_exercise_each_day * days_sanda_exercised) total_minutes days_in_week → sanda_exercise_each_day = 90 :=
by 
  sorry

end sanda_exercise_each_day_l418_418010


namespace marbles_total_l418_418021

variable (original_marbles : ℕ)
variable (found_marbles : ℕ)

theorem marbles_total : original_marbles = 21 → found_marbles = 7 → (original_marbles + found_marbles) = 28 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end marbles_total_l418_418021


namespace muffin_combinations_l418_418198

theorem muffin_combinations (k : ℕ) (n : ℕ) (h_k : k = 4) (h_n : n = 4) :
  (Nat.choose ((n + k - 1) : ℕ) ((k - 1) : ℕ)) = 35 :=
by
  rw [h_k, h_n]
  -- Simplifying Nat.choose (4 + 4 - 1) (4 - 1) = Nat.choose 7 3
  sorry

end muffin_combinations_l418_418198


namespace angle_E_equals_135_l418_418422

theorem angle_E_equals_135 (EF GH : Line) (H E F G : Angle) 
  (parallel_EF_GH : EF ∥ GH) 
  (angle_E_eq_3H : E = 3 * H) 
  (angle_G_eq_2F : G = 2 * F) 
  (supplementary_EH : E + H = 180) 
  (supplementary_FG : F + G = 180) : 
  E = 135 := 
sorry

end angle_E_equals_135_l418_418422


namespace polygon_sides_20_diagonals_l418_418285

-- Define the number of diagonals formula
def diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Define the proof goal
theorem polygon_sides_20_diagonals : ∃ n : ℕ, diagonals n = 20 ∧ n = 8 :=
by 
  use 8
  have h : diagonals 8 = 20 := by
    simp [diagonals]
    norm_num
  exact ⟨h, rfl⟩

end polygon_sides_20_diagonals_l418_418285


namespace gcd_consecutive_triplets_l418_418155

theorem gcd_consecutive_triplets : ∀ i : ℕ, 1 ≤ i → gcd (i * (i + 1) * (i + 2)) 6 = 6 :=
by
  sorry

end gcd_consecutive_triplets_l418_418155


namespace simplify_fraction_l418_418091

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l418_418091


namespace total_num_birds_l418_418830

-- Definitions for conditions
def num_crows := 30
def percent_more_hawks := 0.60

-- Theorem to prove the total number of birds
theorem total_num_birds : num_crows + num_crows * percent_more_hawks + num_crows = 78 := 
sorry

end total_num_birds_l418_418830


namespace each_student_needs_1925_l418_418611

noncomputable theory

-- Declare the constants and conditions
def num_students : ℕ := 6
def misc_expenses : ℕ := 3000
def day1_fund : ℕ := 600
def day2_fund : ℕ := 900
def day3_fund : ℕ := 400
def additional_funds_needed_per_student : ℕ := 475

-- Summarize the calculations
def total_first_3_days : ℕ := day1_fund + day2_fund + day3_fund
def fund_per_day_next_4_days : ℕ := total_first_3_days / 2
def total_next_4_days : ℕ := fund_per_day_next_4_days * 4
def total_raised : ℕ := total_first_3_days + total_next_4_days
def total_with_misc : ℕ := total_raised + misc_expenses
def total_additional : ℕ := additional_funds_needed_per_student * num_students
def total_needed : ℕ := total_with_misc + total_additional
def individual_need : ℕ := total_needed / num_students

-- The theorem we want to prove
theorem each_student_needs_1925 :
  individual_need = 1925 :=
by
  sorry

end each_student_needs_1925_l418_418611


namespace arithmetic_sequence_a3_l418_418363

variable (a : ℕ → ℕ)
variable (S5 : ℕ)
variable (arithmetic_seq : Prop)

def is_arithmetic_seq (a : ℕ → ℕ) : Prop := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_a3 (h1 : is_arithmetic_seq a) (h2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 25) : a 3 = 5 :=
by
  sorry

end arithmetic_sequence_a3_l418_418363


namespace solve_sqrt_equation_l418_418096

theorem solve_sqrt_equation (x : ℝ) (h : sqrt x + sqrt (x + 4) = 12) : 
  x = 1225 / 36 := 
by 
  sorry

end solve_sqrt_equation_l418_418096


namespace exists_b_mod_5_l418_418027

theorem exists_b_mod_5 (p q r s : ℤ) (h1 : ¬ (s % 5 = 0)) (a : ℤ) (h2 : (p * a^3 + q * a^2 + r * a + s) % 5 = 0) : 
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end exists_b_mod_5_l418_418027


namespace tunnel_length_l418_418192

-- Definitions as per the conditions
def train_length : ℚ := 2  -- 2 miles
def train_speed : ℚ := 40  -- 40 miles per hour

def speed_in_miles_per_minute (speed_mph : ℚ) : ℚ :=
  speed_mph / 60  -- Convert speed from miles per hour to miles per minute

def time_travelled_in_minutes : ℚ := 5  -- 5 minutes

-- Theorem statement to prove the length of the tunnel
theorem tunnel_length (h1 : train_length = 2) (h2 : train_speed = 40) :
  (speed_in_miles_per_minute train_speed * time_travelled_in_minutes) - train_length = 4 / 3 :=
by
  sorry  -- Proof not included

end tunnel_length_l418_418192


namespace rate_of_grapes_per_kg_l418_418747

/-- Define variables -/
variable (G : ℕ) (total_cost_grapes : ℕ) (total_cost_mangoes : ℕ) (total_paid : ℕ)

/-- Define given conditions -/
def condition_grapes : total_cost_grapes = 8 * G := by sorry
def condition_mangoes : total_cost_mangoes = 9 * 65 := by sorry
def total_payment : total_paid = total_cost_grapes + total_cost_mangoes := by sorry

/-- Define the proof problem -/
theorem rate_of_grapes_per_kg :
  total_paid = 1145 → G = 70 :=
by
  intro h_total
  have h_grapes : total_cost_grapes = 8 * G := condition_grapes
  have h_mangoes : total_cost_mangoes = 9 * 65 := condition_mangoes
  have h_total_cost : total_paid = total_cost_grapes + total_cost_mangoes := total_payment
  rw [h_grapes, h_mangoes, h_total_cost] at h_total
  sorry

end rate_of_grapes_per_kg_l418_418747


namespace tunnel_length_l418_418190

-- Define relevant constants
def train_length : ℝ := 2
def exit_time_minutes : ℝ := 5
def train_speed_mph : ℝ := 40
def miles_per_hour_to_miles_per_minute (mph : ℝ) := mph / 60
def travel_distance (time_minutes : ℝ) (speed_mph : ℝ) := time_minutes * miles_per_hour_to_miles_per_minute speed_mph

-- The main theorem we want to prove
theorem tunnel_length : travel_distance exit_time_minutes train_speed_mph - train_length = 4 / 3 := sorry

end tunnel_length_l418_418190


namespace find_t_l418_418745

open Complex

-- Define the problem conditions
def vec_OA := (2, 1 : ℝ×ℝ)
def vec_OB (t : ℝ) := (t, -2)
def vec_OC (t : ℝ) := (1, 2 * t)

def vec_AB (t : ℝ) := (t - 2, -3)
def vec_AC (t : ℝ) := (-1, 2 * t - 1)

def distance (v : ℝ × ℝ) := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- 1. Magnitude of AB is 5
def magnitude_AB_five (t : ℝ) : Prop :=
  distance (vec_AB t) = 5

-- 2. Angle BOC = 90 degrees
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

def right_angle_BOC (t : ℝ) : Prop :=
  dot_product (vec_OB t) (vec_OC t) = 0

-- 3. Points A, B, and C are collinear
def collinear (t : ℝ) : Prop :=
  ∃ λ : ℝ, vec_AC t = (λ * vec_AB t).1

theorem find_t :
  ∃ t, (magnitude_AB_five t ∧ (t = 6 ∨ t = -2))
    ∨ (right_angle_BOC t ∧ t = 0)
    ∨ (collinear t ∧ (t = (3 - sqrt 13) / 2 ∨ t = (3 + sqrt 13) / 2)) :=
by
  sorry

end find_t_l418_418745


namespace brown_beads_initial_l418_418557

theorem brown_beads_initial (B : ℕ) 
  (h1 : 1 = 1) -- There is 1 green bead in the container.
  (h2 : 3 = 3) -- There are 3 red beads in the container.
  (h3 : 4 = 4) -- Tom left 4 beads in the container.
  (h4 : 2 = 2) -- Tom took out 2 beads.
  (h5 : 6 = 2 + 4) -- Total initial beads before Tom took any out.
  : B = 2 := sorry

end brown_beads_initial_l418_418557


namespace janet_counts_total_birds_l418_418832

theorem janet_counts_total_birds :
  let crows := 30
  let hawks := crows + (60 / 100) * crows
  hawks + crows = 78 :=
by
  sorry

end janet_counts_total_birds_l418_418832


namespace log_function_is_decreasing_l418_418736

noncomputable def decreasing_interval (a : ℝ) (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x > f y

theorem log_function_is_decreasing (a : ℝ) (h : a > 1) :
    decreasing_interval a (λ x, Real.log (x^2 + 2*x - 3)) :=
sorry

end log_function_is_decreasing_l418_418736


namespace pages_copied_l418_418810

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l418_418810


namespace largest_two_digit_divisible_by_6_ending_in_4_l418_418930

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l418_418930


namespace solve_inequality_l418_418097

theorem solve_inequality (x : ℝ) (h : 3 - (1 / (3 * x + 4)) < 5) : 
  x ∈ { x : ℝ | x < -11/6 } ∨ x ∈ { x : ℝ | x > -4/3 } :=
by
  sorry

end solve_inequality_l418_418097


namespace smaller_angle_measure_l418_418144

theorem smaller_angle_measure (x : ℝ) (a b : ℝ) (h_suppl : a + b = 180) (h_ratio : a = 4 * x ∧ b = x) :
  b = 36 :=
by
  sorry

end smaller_angle_measure_l418_418144


namespace smallest_a_l418_418298

theorem smallest_a (a : ℕ) (h₁ : a > 8) (h₂ : ∀ x : ℤ, ¬prime (x^4 + a^2)) : a = 12 := 
sorry

end smallest_a_l418_418298


namespace pages_copied_for_15_dollars_l418_418805

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l418_418805


namespace solve_equation_l418_418545

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) :
    (3 / (x + 2) - 1 / x = 0) → x = 1 :=
  by sorry

end solve_equation_l418_418545


namespace smallest_a_l418_418297

theorem smallest_a (a : ℕ) (h₁ : a > 8) (h₂ : ∀ x : ℤ, ¬prime (x^4 + a^2)) : a = 12 := 
sorry

end smallest_a_l418_418297


namespace face_value_stock_l418_418601

-- Given conditions
variables (F : ℝ) (yield quoted_price dividend_rate : ℝ)
variables (h_yield : yield = 20) (h_quoted_price : quoted_price = 125)
variables (h_dividend_rate : dividend_rate = 0.25)

--Theorem to prove the face value of the stock is 100
theorem face_value_stock : (dividend_rate * F / quoted_price) * 100 = yield ↔ F = 100 :=
by
  sorry

end face_value_stock_l418_418601


namespace Bernoulli_inequality_l418_418487

theorem Bernoulli_inequality (p : ℝ) (k : ℚ) (hp : 0 < p) (hk : 1 < k) : 
  (1 + p) ^ (k : ℝ) > 1 + p * (k : ℝ) := by
sorry

end Bernoulli_inequality_l418_418487


namespace floor_sequence_sum_2018_l418_418847

noncomputable def seq : ℕ → ℕ
| 0       => 1
| (n + 1) => seq n ^ 2 + seq n

def floor_sequence_sum : ℕ → ℤ :=
  λ n, ∑ i in finset.range n, rat.floor ( (seq i : ℚ) / (seq i + 1) )

theorem floor_sequence_sum_2018 :
  floor_sequence_sum 2018 = 2017 :=
by
  sorry

end floor_sequence_sum_2018_l418_418847


namespace sum_series_eq_4_l418_418268

theorem sum_series_eq_4 : 
  (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 4 := 
by
  sorry

end sum_series_eq_4_l418_418268


namespace fare_for_30km_is_55_l418_418124

def taxi_fare (x : ℝ) : ℝ :=
  if 0 < x ∧ x <= 4 then 10
  else if 4 < x ∧ x <= 18 then 1.5 * x + 4
  else if x > 18 then 2 * x - 5
  else 0 -- Edge case for x <= 0 not explicitly handled in problem

theorem fare_for_30km_is_55 : taxi_fare 30 = 55 :=
by
  unfold taxi_fare
  split_ifs
  · sorry    -- This will be replaced by the actual proof.

end fare_for_30km_is_55_l418_418124


namespace find_perimeter_l418_418969

-- Define the area function of an equilateral triangle
def area_equilateral (a : ℝ) : ℝ := (sqrt 3 / 4) * a^2

-- Define the given area
def given_area : ℝ := 50 * sqrt 12

-- Define the perimeter function
def perimeter_equilateral (a : ℝ) : ℝ := 3 * a

-- Proof statement
theorem find_perimeter (a : ℝ) (h : area_equilateral a = given_area) : perimeter_equilateral a = 60 := sorry

end find_perimeter_l418_418969


namespace exists_sequence_for_k_l418_418311

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end exists_sequence_for_k_l418_418311


namespace arithmetic_sequence_20th_term_l418_418659

theorem arithmetic_sequence_20th_term :
  let a := 2
  let d := 3
  let n := 20
  (a + (n - 1) * d) = 59 :=
by 
  sorry

end arithmetic_sequence_20th_term_l418_418659


namespace horner_value_at_5_l418_418740

def poly (x : ℝ) : ℝ :=
  2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 6 * x + 7

def horner (x : ℝ) : ℝ :=
  (((((2 * x - 5) * x - 4) * x + 3) * x - 6) * x + 7)

theorem horner_value_at_5 : horner 5 = 21 :=
by
  unfold horner
  norm_num
-- sorry

end horner_value_at_5_l418_418740


namespace simplify_fraction_l418_418086

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l418_418086


namespace average_percentage_decrease_l418_418952

-- Given definitions
def original_price : ℝ := 10000
def final_price : ℝ := 6400
def num_reductions : ℕ := 2

-- The goal is to prove the average percentage decrease per reduction
theorem average_percentage_decrease (x : ℝ) (h : (original_price * (1 - x)^num_reductions = final_price)) : x = 0.2 :=
sorry

end average_percentage_decrease_l418_418952


namespace range_of_a_for_positive_f_l418_418041

-- Let the function \(f(x) = ax^2 - 2x + 2\)
def f (a x : ℝ) := a * x^2 - 2 * x + 2

-- Theorem: The range of the real number \( a \) such that \( f(x) > 0 \) for all \( x \) in \( 1 < x < 4 \) is \((\dfrac{1}{2}, +\infty)\)
theorem range_of_a_for_positive_f :
  { a : ℝ | ∀ x : ℝ, 1 < x ∧ x < 4 → f a x > 0 } = { a : ℝ | a > 1/2 } :=
sorry

end range_of_a_for_positive_f_l418_418041


namespace min_tiles_for_square_l418_418492

theorem min_tiles_for_square (a b : ℕ) (ha : a = 6) (hb : b = 4) (harea_tile : a * b = 24)
  (h_lcm : Nat.lcm a b = 12) : 
  let area_square := (Nat.lcm a b) * (Nat.lcm a b) 
  let num_tiles_required := area_square / (a * b)
  num_tiles_required = 6 :=
by
  sorry

end min_tiles_for_square_l418_418492


namespace fraction_equality_l418_418664

def at_op (a b : ℕ) : ℕ := a * b - b^2 + b^3
def hash_op (a b : ℕ) : ℕ := a + b - a * b^2 + a * b^3

theorem fraction_equality : 
  ∀ (a b : ℕ), a = 7 → b = 3 → (at_op a b : ℚ) / (hash_op a b : ℚ) = 39 / 136 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  sorry

end fraction_equality_l418_418664


namespace simplify_fraction_l418_418075

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l418_418075


namespace solve_equation_l418_418547

theorem solve_equation : ∀ x : ℝ, x ≠ -2 → x ≠ 0 → (3 / (x + 2) - 1 / x = 0 ↔ x = 1) :=
by
  intro x h1 h2
  sorry

end solve_equation_l418_418547


namespace no_equal_digit_sequences_l418_418425

theorem no_equal_digit_sequences (k : ℕ) : ¬(∃ A B : Finset ℕ, A ∪ B = Finset.range (k + 1) ∧ A ∩ B = ∅ ∧ 
  (∃ fA fB : List ℕ, (A.val.sort = fA ∧ B.val.sort = fB) ∧ (fA.digits = fB.digits))) :=
by
  sorry

end no_equal_digit_sequences_l418_418425


namespace no_final_number_2_l418_418572

noncomputable def final_number_possible (S : Finset ℕ) : Prop := 
  ∃ x ∈ S, x = 2

theorem no_final_number_2 (S : Finset ℕ) (hS : S = {1, 2, 3, ..., 2013}) :
  ¬ final_number_possible S := 
sorry

end no_final_number_2_l418_418572


namespace cannot_form_triangle_l418_418235

theorem cannot_form_triangle (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 7) (h₃ : c = 14) : ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by {
  rw [h₁, h₂, h₃],
  sorry
}

end cannot_form_triangle_l418_418235


namespace total_pairs_of_shoes_l418_418281

theorem total_pairs_of_shoes : 
  let ellie_pairs := 8 
  let riley_pairs := ellie_pairs - 3
  let combined_pairs := ellie_pairs + riley_pairs
  let jordan_pairs := 1.5 * combined_pairs
  let total_pairs := ellie_pairs + riley_pairs + jordan_pairs.floor
  total_pairs = 32 := by
  let ellie := 8
  let riley := ellie - 3
  let combined := ellie + riley
  let jordan := 1.5 * combined
  let jordan_rounded := jordan.floor
  let total := ellie + riley + jordan_rounded
  have : total = 32 := by linarith
  exact this

end total_pairs_of_shoes_l418_418281


namespace combined_students_yellow_blue_l418_418480

theorem combined_students_yellow_blue {total_students blue_percent red_percent yellow_combined : ℕ} :
  total_students = 200 →
  blue_percent = 30 →
  red_percent = 40 →
  yellow_combined = (total_students * 3 / 10) + ((total_students - (total_students * 3 / 10)) * 6 / 10) →
  yellow_combined = 144 :=
by
  intros
  sorry

end combined_students_yellow_blue_l418_418480


namespace boy_running_time_l418_418586

-- Define the basic parameters
def side_length (s : ℕ) : ℕ := s
def speed_kmph (v : ℕ) : ℕ := v

-- Define a function to compute the perimeter of a square
def perimeter (side : ℕ) : ℕ := 4 * side

-- Define a function to convert speed from km/hr to m/s
def speed_mps (v : ℕ) : ℚ := (v * 1000) / 3600

-- Define a function to compute time taken to cover a distance given speed
def time_to_run (distance : ℕ) (speed : ℚ) : ℚ := distance / speed

-- Main theorem statement
theorem boy_running_time (s v : ℕ) (h1 : s = 40) (h2 : v = 12) : time_to_run (perimeter s) (speed_mps v) ≈ 48 :=
by
  have perimeter_eq := by rw [perimeter, h1]; simp
  have speed_eq := by rw [speed_mps, h2]; simp
  rw [time_to_run]
  sorry

end boy_running_time_l418_418586


namespace simplify_fraction_l418_418076

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l418_418076


namespace projectile_reaches_100_feet_l418_418976

-- Define the height function of the projectile
def height (t : ℝ) : ℝ := -16 * t^2 + 80 * t

-- Define the time at which the projectile reaches 100 feet
def timeAtHeight (h : ℝ) : Option ℝ :=
  let a : ℝ := -16
  let b : ℝ := 80
  let c : ℝ := -h
  if discriminant b a c < 0 then none else
  some ((-b + sqrt (discriminant b a c)) / (2 * a))

-- Help function to calculate the discriminant
def discriminant (b a c : ℝ) : ℝ :=
  b^2 - 4 * a * c

-- Lean 4 statement to prove timeAtHeight 100 = some 2.5
theorem projectile_reaches_100_feet : timeAtHeight 100 = some 2.5 :=
by
  sorry

end projectile_reaches_100_feet_l418_418976


namespace machine_produces_480_cans_in_8_hours_l418_418517

def cans_produced_in_interval : ℕ := 30
def interval_duration_minutes : ℕ := 30
def hours_worked : ℕ := 8
def minutes_in_hour : ℕ := 60

theorem machine_produces_480_cans_in_8_hours :
  (hours_worked * (minutes_in_hour / interval_duration_minutes) * cans_produced_in_interval) = 480 := by
  sorry

end machine_produces_480_cans_in_8_hours_l418_418517


namespace min_area_triangle_PMN_point_through_MN_l418_418350

-- Definitions related to the problem conditions
def point_P := (2 : ℝ, 0 : ℝ)
def parabola (y x : ℝ) : Prop := y^2 = 4 * x
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- (I) Minimum area of triangle PMN when slopes k1 and k2 are perpendicular
theorem min_area_triangle_PMN (k1 k2 : ℝ) (k1k2_perp : k1 * k2 = -1) :
  ∃ (M N : ℝ × ℝ), (∃ A B C D : ℝ × ℝ, 
  (parabola A.2 A.1 ∧ parabola B.2 B.1 ∧ parabola C.2 C.1 ∧ parabola D.2 D.1) ∧
  (midpoint A B = M ∧ midpoint C D = N) ∧
  (∃ t_pq x y : ℝ, t_pq = 2 * real.abs k1 * real.sqrt (k1^2 + 1) * real.sqrt (k2^2 + 1) / real.abs k2 ∧ 
  (t_pq / 2) * (real.abs k1 + real.abs (1 / k1)) = 4)) := sorry

-- (II) Line MN passes through the fixed point (2, 2) when k1 + k2 = 1
theorem point_through_MN (k1 k2 : ℝ) (k_sum : k1 + k2 = 1) :
  ∃ (M N : ℝ × ℝ), (∃ A B C D : ℝ × ℝ, 
  (parabola A.2 A.1 ∧ parabola B.2 B.1 ∧ parabola C.2 C.1 ∧ parabola D.2 D.1) ∧
  (midpoint A B = M ∧ midpoint C D = N) ∧
  (∀ (x y : ℝ), (x - 2) / (M.1 - 2) = (y - 2) / (M.2 - 2) → 
  (x = 2 ∧ y = 2))) := sorry

end min_area_triangle_PMN_point_through_MN_l418_418350


namespace white_surface_fraction_l418_418606

-- Definition of the problem conditions
def larger_cube_surface_area : ℕ := 54
def white_cubes : ℕ := 6
def white_surface_area_minimized : ℕ := 5

-- Theorem statement proving the fraction of white surface area
theorem white_surface_fraction : (white_surface_area_minimized / larger_cube_surface_area : ℚ) = 5 / 54 := 
by
  sorry

end white_surface_fraction_l418_418606


namespace simplify_expression_l418_418950

theorem simplify_expression :
  (360 / 24) * (10 / 240) * (6 / 3) * (9 / 18) = 5 / 8 := by
  sorry

end simplify_expression_l418_418950


namespace part_I_part_II_l418_418862

def f (x a : ℝ) : ℝ := abs (x - 1) + a * abs (x - 2)

theorem part_I (a : ℝ) : 
  (∃ x min, ∀ y, f y a ≥ min) ↔ -1 ≤ a ∧ a ≤ 1 := 
by sorry

theorem part_II (a : ℝ) : 
  (∀ x, f x a ≥ 1 / 2) ↔ a = 1 / 2 := 
by sorry

end part_I_part_II_l418_418862


namespace maria_min_score_fifth_term_l418_418139

theorem maria_min_score_fifth_term (score1 score2 score3 score4 : ℕ) (avg_required : ℕ) 
  (h1 : score1 = 84) (h2 : score2 = 80) (h3 : score3 = 82) (h4 : score4 = 78)
  (h_avg_required : avg_required = 85) :
  ∃ x : ℕ, x ≥ 101 :=
by
  sorry

end maria_min_score_fifth_term_l418_418139


namespace polynomial_remainder_distinct_l418_418300

open Nat

theorem polynomial_remainder_distinct (a b c p : ℕ) (hp : Nat.Prime p) (hp_ge5 : p ≥ 5)
  (ha : Nat.gcd a p = 1) (hb : b^2 ≡ 3 * a * c [MOD p]) (hp_mod3 : p ≡ 2 [MOD 3]) :
  ∀ m1 m2 : ℕ, m1 < p ∧ m2 < p → m1 ≠ m2 → (a * m1^3 + b * m1^2 + c * m1) % p ≠ (a * m2^3 + b * m2^2 + c * m2) % p := 
by
  sorry

end polynomial_remainder_distinct_l418_418300


namespace expression_simplification_l418_418270

def base_expr := (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) *
                (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64)

theorem expression_simplification :
  base_expr = 3^128 - 4^128 := by
  sorry

end expression_simplification_l418_418270


namespace boat_distance_along_stream_l418_418788

-- Define the conditions
def speed_of_boat_still_water : ℝ := 9
def distance_against_stream_per_hour : ℝ := 7

-- Define the speed of the stream
def speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour

-- Define the speed of the boat along the stream
def speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream

-- Theorem statement
theorem boat_distance_along_stream (speed_of_boat_still_water : ℝ)
                                    (distance_against_stream_per_hour : ℝ)
                                    (effective_speed_against_stream : ℝ := speed_of_boat_still_water - speed_of_stream)
                                    (speed_of_stream : ℝ := speed_of_boat_still_water - distance_against_stream_per_hour)
                                    (speed_of_boat_along_stream : ℝ := speed_of_boat_still_water + speed_of_stream)
                                    (one_hour : ℝ := 1) :
  speed_of_boat_along_stream = 11 := 
  by
    sorry

end boat_distance_along_stream_l418_418788


namespace bob_salary_at_end_of_march_l418_418642

-- Define the initial conditions and parameters
def initial_salary := 3500
def february_raise_percentage := 0.10
def march_pay_cut_percentage := 0.15

-- Compute the intermediate values for February and March salaries
def salary_february := initial_salary * (1 + february_raise_percentage)
def salary_march := salary_february * (1 - march_pay_cut_percentage)

-- Define the expected result
def expected_salary_march := 3273

-- The theorem to prove
theorem bob_salary_at_end_of_march : 
  round salary_march = expected_salary_march := 
by
  sorry

end bob_salary_at_end_of_march_l418_418642


namespace tan_y_eq_tan_x_plus_one_over_cos_x_l418_418035

theorem tan_y_eq_tan_x_plus_one_over_cos_x 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hxy : x < y) 
  (hy : y < π / 2) 
  (h_tan : Real.tan y = Real.tan x + (1 / Real.cos x)) 
  : y - (x / 2) = π / 6 :=
sorry

end tan_y_eq_tan_x_plus_one_over_cos_x_l418_418035


namespace certain_amount_l418_418430

theorem certain_amount (x : ℝ) (h1 : 2 * x = 86 - 54) (h2 : 8 + 3 * 8 = 24) (h3 : 86 - 54 + 32 = 86) : x = 43 := 
by {
  sorry
}

end certain_amount_l418_418430


namespace simplify_expression_l418_418719

theorem simplify_expression (α : ℝ) (h1 : π < α) (h2 : α < 3 * π / 2) :
  sqrt ((1 - Real.cos α) / (1 + Real.cos α)) + sqrt ((1 + Real.cos α) / (1 - Real.cos α)) = -2 / Real.sin α :=
sorry

end simplify_expression_l418_418719


namespace groups_of_friends_l418_418046

theorem groups_of_friends {n k : ℕ} (h_n : n = 7) (h_k : k = 4) : nat.choose n k = 35 :=
by
  rw [h_n, h_k]
  exact nat.choose_eq_factorial_div_factorial (by decide) (by decide)
  -- Note: detailed steps would typically follow here, but are omitted due to 'sorry'
  sorry

end groups_of_friends_l418_418046


namespace smallest_n_l418_418783

noncomputable def P (n : ℕ) : ℝ := sorry
-- P represents the probability function, the definition of which would be comprehensive based on the problem.

theorem smallest_n {n : ℕ} : 
  (∃ n, P(n) < 1 / 2023) → n = 51 :=
  sorry

end smallest_n_l418_418783


namespace roller_coaster_ticket_cost_l418_418248

def ferrisWheelCost : ℕ := 6
def logRideCost : ℕ := 7
def initialTickets : ℕ := 2
def ticketsToBuy : ℕ := 16

def totalTicketsNeeded : ℕ := initialTickets + ticketsToBuy
def ridesCost : ℕ := ferrisWheelCost + logRideCost
def rollerCoasterCost : ℕ := totalTicketsNeeded - ridesCost

theorem roller_coaster_ticket_cost :
  rollerCoasterCost = 5 :=
by
  sorry

end roller_coaster_ticket_cost_l418_418248


namespace sufficient_but_not_necessary_l418_418848

theorem sufficient_but_not_necessary (a b : ℝ) :
  ((a - b) ^ 3 * b ^ 2 > 0 → a > b) ∧ ¬(a > b → (a - b) ^ 3 * b ^ 2 > 0) :=
by
  sorry

end sufficient_but_not_necessary_l418_418848


namespace standard_equation_ellipse_existence_of_lines_y_eq_negx_plus_minus_sqrt3_div3_l418_418709

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
noncomputable def distance (x1 y1 x2 y2 : ℝ) := real.sqrt((x2 - x1)^2 + (y2 - y1)^2)
noncomputable def slope (x1 y1 x2 y2 : ℝ) := (y2 - y1) / (x2 - x1)

theorem standard_equation_ellipse :
  ∃ a b x y, (a > b ∧ b > 0) ∧ (c = 1) ∧ (e = 1 / 2) ∧ ellipse_eq 2 (real.sqrt 3) x y :=
sorry

theorem existence_of_lines_y_eq_negx_plus_minus_sqrt3_div3 :
  ∃ m x1 y1 x2 y2 (F₁ F₂ : ℝ × ℝ) (C D : ℝ × ℝ),
    (F₁ = (-1, 0)) ∧ (F₂ = (1, 0)) ∧
    distance 0 0 1 0 < 1 ∧ 
    ∃ m, (|CD| / |AB| = (8 * real.sqrt 3) / 7) ∧ 
    (y = -x + m) ∨ (y = -x - m) ∧ 
    m = (real.sqrt 3 / 3) :=
sorry

end standard_equation_ellipse_existence_of_lines_y_eq_negx_plus_minus_sqrt3_div3_l418_418709


namespace jail_time_ratio_l418_418148

def arrests (days : ℕ) (cities : ℕ) (arrests_per_day : ℕ) : ℕ := days * cities * arrests_per_day
def jail_days_before_trial (total_arrests : ℕ) (days_before_trial : ℕ) : ℕ := total_arrests * days_before_trial
def weeks_from_days (days : ℕ) : ℕ := days / 7
def time_after_trial (total_jail_time_weeks : ℕ) (weeks_before_trial : ℕ) : ℕ := total_jail_time_weeks - weeks_before_trial
def total_possible_jail_time (total_arrests : ℕ) (sentence_weeks : ℕ) : ℕ := total_arrests * sentence_weeks
def ratio (after_trial_weeks : ℕ) (total_possible_weeks : ℕ) : ℚ := after_trial_weeks / total_possible_weeks

theorem jail_time_ratio 
    (days : ℕ := 30) 
    (cities : ℕ := 21)
    (arrests_per_day : ℕ := 10)
    (days_before_trial : ℕ := 4)
    (total_jail_time_weeks : ℕ := 9900)
    (sentence_weeks : ℕ := 2) :
    ratio 
      (time_after_trial 
        total_jail_time_weeks 
        (weeks_from_days 
          (jail_days_before_trial 
            (arrests days cities arrests_per_day) 
            days_before_trial))) 
      (total_possible_jail_time 
        (arrests days cities arrests_per_day) 
        sentence_weeks) = 1/2 := 
by
  -- We leave the proof as an exercise
  sorry

end jail_time_ratio_l418_418148


namespace probability_Xiao_Jie_boards_high_comfort_l418_418638

open Classical

noncomputable def probabilities_of_boards_high_comfort : ℚ := 
let buses := ["high", "medium", "low"] in
let permutations := List.permutations buses in
let favorable := permutations.filter (λ p, 
  (p.nth_le 1 sorry = "high" ∧ p.nth_le 0 sorry ≠ "high") ∨
  (p.nth_le 2 sorry = "high" ∧ p.nth_le 0 sorry = "low") ∨
  (p.nth_le 2 sorry = "high" ∧ p.nth_le 1 sorry > p.nth_le 0 sorry)) in
favorable.length / permutations.length

theorem probability_Xiao_Jie_boards_high_comfort : probabilities_of_boards_high_comfort = 1 / 2 :=
sorry

end probability_Xiao_Jie_boards_high_comfort_l418_418638


namespace pages_copied_l418_418812

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l418_418812


namespace problem_1_problem_2_l418_418044

-- Definition of "J_k type" sequence
def J_type (k : ℕ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ K : ℕ, a[n + K] * a[n + 2 * K] = a[n + K] * a[n + 2 * K]

-- Problem 1: Prove the specific form of a_2n
theorem problem_1 (a : ℕ → ℝ) (hJ_2 : J_type 2 a) (h_a2 : a 2 = 8) (h_a8 : a 8 = 1) : 
  ∀ n : ℕ, a (2 * n) = 2 ^ (4 - n) := 
by 
  sorry

-- Problem 2: Prove that the sequence is a geometric sequence
theorem problem_2 (a : ℕ → ℝ) (hJ_3 : J_type 3 a) (hJ_4 : J_type 4 a) : 
  ∃ r : ℝ, ∀ n : ℕ, a(n + 1) = r * a(n) :=
by
  sorry

end problem_1_problem_2_l418_418044


namespace projection_is_one_l418_418151

variable (a b : Vector ℝ 3)
variable (angle : ℝ)

-- Conditions
axiom length_a : ‖a‖ = 2
axiom length_b : ‖b‖ = 2
axiom angle_ab : angle = Real.pi / 3 -- 60 degrees in radians

-- Definition of projection of vector b in the direction of vector a
noncomputable def projection (a b : Vector ℝ 3) (angle : ℝ) : ℝ :=
  ‖b‖ * Real.cos angle

-- Statement to prove the projection is 1
theorem projection_is_one : projection a b angle = 1 :=
by 
  rw [projection]
  rw [length_b, angle_ab]
  have : Real.cos (Real.pi / 3) = 1 / 2 := Real.cos_pi_div_three
  rw [this]
  norm_num
  sorry

end projection_is_one_l418_418151


namespace rhombus_area_l418_418973

theorem rhombus_area (d1 d2 : ℝ) (hd1 : d1 = 15) (hd2 : d2 = 21) : 
  (d1 * d2) / 2 = 157.5 :=
by {
  rw [hd1, hd2],
  norm_num,
  sorry
}

end rhombus_area_l418_418973


namespace find_possible_values_of_n_l418_418842

def is_equilateral_triangle (ABC : Triangle) : Prop :=
  ∀ (A B C : Point), dist A B = dist B C ∧ dist B C = dist C A

def law_of_reflection (α : ℝ) : ℝ :=
  180 - α

def valid_bounces (n : ℕ) : Prop :=
  n % 6 = 1 ∨ n % 6 = 5 ∧ n ≠ 5 ∧ n ≠ 17

theorem find_possible_values_of_n (ABC: Triangle) (A: Point) (ray: Ray) :
  is_equilateral_triangle ABC →
  (∀ n, valid_bounces n → n) :=
sorry

end find_possible_values_of_n_l418_418842


namespace problem_1_l418_418857

def coprime (a b : ℕ) := gcd a b = 1

theorem problem_1 (x₁ x₂ : ℕ) (h_coprime : coprime x₁ x₂) (h_pos_x₁ : x₁ > 0) (h_pos_x₂ : x₂ > 0) :
  ∀ (i : ℕ), i > 1 → ∃ (j : ℕ), j > i ∧ ∃ p : ℕ, p ∣ x_j ∧ p ∣ (x_j^(i + 1) - x_j) :=
begin
  sorry
end

end problem_1_l418_418857


namespace data_instances_in_one_hour_l418_418438

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l418_418438


namespace round_nearest_hundredth_l418_418493

noncomputable def repeating_decimal := real.of_rat 54 + real.of_rat 54 / 99

theorem round_nearest_hundredth :
  real.lt_repeat (54 + 54 / 99) to 0.01 (either_round (to_round (14 / 99)).
    54.5454 ≥ by 1 to add previous latest 4 + 1) + 1 but 
round_eq_real_one (0.01 by 1 and solution 44 4
  result == value next place is == 1) rounding with equality with thee.  by sorry
51.5455 rounded to real (to rounding by )
ori_eval_eq one repeat equality as  (54 100) implies divisionon
    that realE rounding ..


end round_nearest_hundredth_l418_418493


namespace find_a_l418_418716

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
sorry

end find_a_l418_418716


namespace maximize_profit_l418_418233

-- Constants and conditions
def fixed_cost : ℝ := 500
def selling_price_per_device : ℝ := 100

def production_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then
    1/2 * x^2 + 40 * x
  else if x ≥ 80 then
    101 * x + 8100 / x - 2180
  else
    0 -- Invalid case since x must be positive

-- Profit function
def profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then
    selling_price_per_device * x - (1/2 * x^2 + 40 * x) - fixed_cost
  else if x ≥ 80 then
    selling_price_per_device * x - (101 * x + 8100 / x - 2180) - fixed_cost
  else
    0 -- Invalid case since x must be positive

-- Statement to maximize profit
theorem maximize_profit :
  ∃ x : ℝ, profit x = 1500 ∧ x = 90 :=
sorry

end maximize_profit_l418_418233


namespace distinct_pawn_placements_l418_418394

theorem distinct_pawn_placements : 
  let n := 5 in ((n.factorial) * (n.factorial) = 14400) :=
by 
  let n := 5
  sorry

end distinct_pawn_placements_l418_418394


namespace number_of_correct_propositions_l418_418031

-- Assume the context of planes and lines in a 3D space with necessary properties
variables {Plane Line : Type}
variables [HasOrthogonal Plane Line] [HasParallel Plane Line]

-- Define planes and lines involved and the propositions
variables (a b : Plane) (l : Line)

-- Define the propositions as Lean predicates
def prop1 : Prop := (∀ l, InPlane l a → (∀ m, InPlane m b → Perpendicular l m)) → Perpendicular a b
def prop2 : Prop := (∀ l, InPlane l a → Parallel l b) → Parallel a b
def prop3 : Prop := Perpendicular a b → (InPlane l a → Perpendicular l b)
def prop4 : Prop := Parallel a b → (InPlane l a → Parallel l b)

-- State the main theorem
theorem number_of_correct_propositions : (∑ (i : ℕ) in [prop1, prop2, prop4], if i then 1 else 0) = 3 :=
sorry

end number_of_correct_propositions_l418_418031


namespace groups_partition_count_l418_418522

-- Definitions based on the conditions
def num_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 6
def group3_size : ℕ := 2

-- Given specific names for groups based on problem statement
def Fluffy_group_size : ℕ := group1_size
def Nipper_group_size : ℕ := group2_size

-- The total number of ways to form the groups given the conditions
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem groups_partition_count :
  total_ways 10 3 * total_ways 7 5 = 2520 := sorry

end groups_partition_count_l418_418522


namespace Laura_running_speed_l418_418441

noncomputable def running_speed (x : ℝ) : Prop :=
  (15 / (3 * x + 2)) + (4 / x) = 1.5 ∧ x > 0

theorem Laura_running_speed : ∃ (x : ℝ), running_speed x ∧ abs (x - 5.64) < 0.01 :=
by
  sorry

end Laura_running_speed_l418_418441


namespace Mark_charged_more_l418_418178

theorem Mark_charged_more (K P M : ℕ) 
  (h1 : P = 2 * K) 
  (h2 : P = M / 3)
  (h3 : K + P + M = 153) : M - K = 85 :=
by
  -- proof to be filled in later
  sorry

end Mark_charged_more_l418_418178


namespace dog_grouping_l418_418524

theorem dog_grouping : 
  let dogs := 12 in
  let group1_size := 4 in
  let group2_size := 6 in
  let group3_size := 2 in
  let Fluffy := "Fluffy" in
  let Nipper := "Nipper" in
  let remaining_dogs := dogs - 2 in
  nat.choose remaining_dogs (group1_size - 1) * nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1) = 2520 :=
by
  sorry

end dog_grouping_l418_418524


namespace length_of_side_x_l418_418665

theorem length_of_side_x 
  (A B C D : Type)
  [EuclideanGeometry A]
  (point_A point_B point_C point_D : A)
  (angle_BAC : angle point_B point_A point_C = 30)
  (angle_DAB : angle point_D point_A point_B = 45)
  (AD : distance point_A point_D = 10)
  (ABD_is_45_45_90 : is_45_45_90_triangle point_A point_B point_D)
  (ACD_is_30_60_90 : is_30_60_90_triangle point_A point_C point_D) :
  distance point_A point_C = 10 / sqrt 3 →
  distance point_A point_C * 2 = 20 * sqrt 3 / 3 :=
by
  sorry

end length_of_side_x_l418_418665


namespace minimum_n_sqrt_27n_is_integer_l418_418717

theorem minimum_n_sqrt_27n_is_integer (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, sqrt (27 * n) = k) : n = 3 := 
sorry

end minimum_n_sqrt_27n_is_integer_l418_418717


namespace stephen_round_trips_l418_418101

theorem stephen_round_trips :
  ∀ (h : ℕ) (r : ℚ) (d : ℕ),
  h = 40000 →
  r = 3 / 4 →
  d = 600000 →
  let single_trip_distance := (r * h) in
  let round_trip_distance := (2 * single_trip_distance) in
  d / round_trip_distance = 10 :=
by
  intros h r d h_eq r_eq d_eq
  let single_trip_distance := r * h
  let round_trip_distance := 2 * single_trip_distance
  have h1 : single_trip_distance = 30000 := by
    rw [h_eq, r_eq]
    norm_num
  have h2 : round_trip_distance = 60000 := by
    rw [h1]  
    norm_num
  rw [h2, d_eq]  
  norm_num
  sorry

end stephen_round_trips_l418_418101


namespace Dan_running_speed_l418_418140

theorem Dan_running_speed :
  ∃ R : ℝ, 
  let distance_running := 3 in
  let distance_swimming := 3 in
  let swimming_speed := 6 in
  let average_speed := 7.5 in
  let total_distance := distance_running + distance_swimming in
  let total_time := total_distance / average_speed in
  let time_to_swim := distance_swimming / swimming_speed in
  let time_to_run := distance_running / R in
  total_time = time_to_run + time_to_swim ∧ R = 10 :=
begin
  sorry
end

end Dan_running_speed_l418_418140


namespace Q1_Q2_Q3_l418_418770

def rapidlyIncreasingSequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n+2) - a (n+1) > a (n+1) - a n

theorem Q1 :
  rapidlyIncreasingSequence (λ n, 2^n) :=
sorry

theorem Q2 (a : ℕ → ℤ)
  (h_rapid : rapidlyIncreasingSequence a)
  (h_int : ∀ n, a n ∈ ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (hk : a k = 2023) :
  k = 63 :=
sorry

theorem Q3 (b : ℕ → ℝ) (k : ℕ)
  (h_rapid : rapidlyIncreasingSequence b)
  (h_terms : ∀ n < 2*k, b n ∈ ℝ)
  (h_sum : (finset.range (2*k)).sum b = k)
  (c : ℕ → ℝ) (h_c : ∀ n, c n = 2^(b n)) :
  c k * c (k+1) < 2 :=
sorry

end Q1_Q2_Q3_l418_418770


namespace minimal_stick_length_exists_l418_418152

theorem minimal_stick_length_exists (n : ℕ) : (n = 2012^2 - 2012 + 1) → 
  ∃ (sticks : list ℕ), (sticks.length = 2012 ∧ (sticks.sum = n) ∧ 
    (∀ k, k ∈ (list.range 1 2013) ↔ ∃ s ∈ sticks, s.k = k)) :=
begin
  intro h,
  sorry
end

end minimal_stick_length_exists_l418_418152


namespace dot_product_magnitude_l418_418379

open Real

variables {a b : ℝ^3}

-- Given conditions
def unit_vector (v : ℝ^3) : Prop :=
  ‖v‖ = 1

def given_condition (a b : ℝ^3) : Prop :=
  (2 • a - 3 • b) ⬝ (2 • a + b) = 3

-- To prove:
theorem dot_product (ha : unit_vector a) (hb : unit_vector b) (hc : given_condition a b) :
  a ⬝ b = -1 / 2 :=
sorry

theorem magnitude (ha : unit_vector a) (hb : unit_vector b) (hc : given_condition a b) :
  ‖2 • a - b‖ = sqrt 7 :=
sorry

end dot_product_magnitude_l418_418379


namespace correct_calculation_l418_418167

theorem correct_calculation (x y : ℝ) : 
  (sqrt (x^2) ≠ -3) ∧ 
  (sqrt 16 ≠ 4) ∧ 
  (¬ (sqrt (-25)).isReal) ∧ 
  (cbrt 8 = 2) :=
by
  sorry

end correct_calculation_l418_418167


namespace scarf_cost_is_10_l418_418499

-- Define the conditions as given in the problem statement
def initial_amount : ℕ := 53
def cost_per_toy_car : ℕ := 11
def num_toy_cars : ℕ := 2
def cost_of_beanie : ℕ := 14
def remaining_after_beanie : ℕ := 7

-- Calculate the cost of the toy cars
def total_cost_toy_cars : ℕ := num_toy_cars * cost_per_toy_car

-- Calculate the amount left after buying the toy cars
def amount_after_toys : ℕ := initial_amount - total_cost_toy_cars

-- Calculate the amount left after buying the beanie
def amount_after_beanie : ℕ := amount_after_toys - cost_of_beanie

-- Define the cost of the scarf
def cost_of_scarf : ℕ := amount_after_beanie - remaining_after_beanie

-- The theorem stating that cost_of_scarf is 10 dollars
theorem scarf_cost_is_10 : cost_of_scarf = 10 := by
  sorry

end scarf_cost_is_10_l418_418499


namespace range_of_lambda_l418_418723

namespace proof_problem

def f (x : ℝ) : ℝ := x^2 + 2 * x
def g (x : ℝ) : ℝ := -x^2 + 2 * x

noncomputable def h (x : ℝ) (λ : ℝ) : ℝ := g(x) - λ * f(x) + 1

theorem range_of_lambda {λ : ℝ} :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → deriv (h x λ) x ≥ 0) → (λ ≤ 0) :=
sorry

end proof_problem

end range_of_lambda_l418_418723


namespace sally_sours_total_l418_418496

theorem sally_sours_total (cherry_sours lemon_sours orange_sours total_sours : ℕ) 
    (h1 : cherry_sours = 32)
    (h2 : 5 * cherry_sours = 4 * lemon_sours)
    (h3 : orange_sours = total_sours / 4)
    (h4 : cherry_sours + lemon_sours + orange_sours = total_sours) : 
    total_sours = 96 :=
by
  rw [h1] at h2
  have h5 : lemon_sours = 40 := by linarith
  rw [h1, h5] at h4
  have h6 : orange_sours = total_sours / 4 := by assumption
  rw [h6] at h4
  have h7 : 72 + total_sours / 4 = total_sours := by linarith
  sorry

end sally_sours_total_l418_418496


namespace tangent_line_at_1_monotonicity_of_g_l418_418334

-- Given conditions and definitions
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * (Real.log x)
def g (a x : ℝ) : ℝ := f a x - 2 * x

-- Condition a > 0
variable (a : ℝ) (ha : a > 0)

-- Prove (1)
theorem tangent_line_at_1 : 
  f 1 x = x^2 - 2 * x + 2 * Real.log x → 
  let y := f 1 1 in 
  y = -1 → 
  ∀ x, ∀ y, 
  tangent (curve y) (point (1, f 1 1)) (line_eq y = 2 * x - 3 ) := 
sorry

-- Prove (2)
theorem monotonicity_of_g : 
  (0 < a ∧ a < 1 → (∀ x, (0 < x ∧ x < 1 → deriv (g a x) > 0) ∧ (∀ x, (1 < x ∧ x < 1 / a) → deriv (g a x) < 0) ∧ (∀ x, (1 / a < x → deriv (g a x) > 0))) ∧
  (a > 1 → (∀ x, (0 < x ∧ x < 1 / a → deriv (g a x) > 0) ∧ (∀ x, (1 / a < x ∧ x < 1) → deriv (g a x) < 0) ∧ (∀ x, (1 < x → deriv (g a x) > 0))) ∧
  (a = 1 → ∀ x, 0 < x → deriv (g a x) ≥ 0) := 
sorry

end tangent_line_at_1_monotonicity_of_g_l418_418334


namespace points_in_circle_l418_418489

theorem points_in_circle (A : Type) [metric_space A] {P : set A}
  (h : ∀ x ∈ P, ∃ c : A, dist c x ≤ 5 / 2)
  (hP : finite P) (hP_card : card P = 10) :
  ∃ x y ∈ P, x ≠ y ∧ dist x y < 2 :=
by sorry

end points_in_circle_l418_418489


namespace copy_pages_15_dollars_l418_418822

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l418_418822


namespace max_cos_sum_l418_418677

theorem max_cos_sum (x y : ℝ) (h : cos x - cos y = 1 / 4) : 
  cos (x + y) ≤ 31 / 32 :=
begin
  sorry
end

end max_cos_sum_l418_418677


namespace parabola_equation_l418_418130

-- Define the given conditions
def ParabolaPassingThroughPoint (x y : ℝ) (p : ℝ) (focus_on_x_axis focus_on_y_axis : Bool) : Bool :=
  if focus_on_x_axis then
    y^2 = 2 * p * x
  else if focus_on_y_axis then
    x^2 = 2 * p * y
  else
    false

-- Define the theorem to be proven
theorem parabola_equation (x y : ℝ) (hx : x = 1) (hy : y = 2) (focus_on_x_axis focus_on_y_axis : Bool) (p1 p2 : ℝ)
  (h_p1_pos : focus_on_x_axis → 0 < p1)
  (h_p2_pos : focus_on_y_axis → 0 < p2) :
  ParabolaPassingThroughPoint x y p1 focus_on_x_axis false ∨
  ParabolaPassingThroughPoint x y p2 false focus_on_y_axis :=
by
  sorry

end parabola_equation_l418_418130


namespace no_linear_term_implies_equal_l418_418419

theorem no_linear_term_implies_equal (m n : ℝ) (h : (x : ℝ) → (x + m) * (x - n) - x^2 - (- mn) = 0) : m = n :=
by
  sorry

end no_linear_term_implies_equal_l418_418419


namespace volume_increased_by_18_75_l418_418165

-- Define the original volume
def original_volume (π r h : ℝ) := π * r^2 * h

-- Define the new volume with increased radius and tripled height
def new_volume (π r h : ℝ) := π * (2.5 * r)^2 * (3 * h)

-- Define the factor by which the volume is increased
def volume_increase_factor (π r h : ℝ) := new_volume π r h / original_volume π r h

-- Theorem stating the volume increase factor
theorem volume_increased_by_18_75 (π r h : ℝ) (hπ : π ≠ 0) (hr : r ≠ 0) (hh : h ≠ 0):
  volume_increase_factor π r h = 18.75 :=
begin
  unfold volume_increase_factor,
  unfold new_volume,
  unfold original_volume,
  field_simp [hπ, hr, hh],
  norm_num,
  ring,
end

end volume_increased_by_18_75_l418_418165


namespace student_fraction_mistake_l418_418404

theorem student_fraction_mistake (n : ℕ) (h_n : n = 576) 
(h_mistake : ∃ r : ℚ, r * n = (5 / 16) * n + 300) : ∃ r : ℚ, r = 5 / 6 :=
by
  sorry

end student_fraction_mistake_l418_418404


namespace intersection_of_sets_l418_418039

theorem intersection_of_sets :
  (M ∩ N = {0}) :=
by
  let M := {-1, 0}
  let N := {0, 1}
  sorry

end intersection_of_sets_l418_418039


namespace sum_of_a_values_l418_418762

theorem sum_of_a_values :
  (∑ a in (Finset.filter (λ a : ℤ,
    let x := (a + 4) / 3 in
    x ≥ 0 ∧ x ≠ 2 ∧
    (1 - a) / 2 ≥ -2 ∧
    (a + 4) % 3 = 0), (Finset.Icc (-1000 : ℤ) 1000)), id) = 0 :=
by
  sorry

end sum_of_a_values_l418_418762


namespace max_element_sum_l418_418840

-- Definitions based on conditions
def S : Set ℚ :=
  {r | ∃ (p q : ℕ), r = p / q ∧ q ≤ 2009 ∧ p / q < 1257/2009}

-- Maximum element of S in reduced form
def max_element_S (r : ℚ) : Prop := r ∈ S ∧ ∀ s ∈ S, r ≥ s

-- Main statement to be proven
theorem max_element_sum : 
  ∃ p0 q0 : ℕ, max_element_S (p0 / q0) ∧ Nat.gcd p0 q0 = 1 ∧ p0 + q0 = 595 := 
sorry

end max_element_sum_l418_418840


namespace rain_total_duration_l418_418003

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end rain_total_duration_l418_418003


namespace road_repair_completion_time_l418_418228

theorem road_repair_completion_time (L R r : ℕ) (hL : L = 100) (hR : R = 64) (hr : r = 9) :
  (L - R) / r = 5 :=
by
  sorry

end road_repair_completion_time_l418_418228


namespace cos_equation_solution_count_l418_418795

theorem cos_equation_solution_count :
  let interval := Set.Icc 0 Real.pi in
  Set.Finite (Set {x | interval x ∧ Real.cos (7 * x) = Real.cos (5 * x) }).to_finset.card = 7 :=
by
  sorry

end cos_equation_solution_count_l418_418795


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418919

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418919


namespace area_ratio_acute_area_ratio_obtuse_l418_418183

variables {A B C : ℝ}

theorem area_ratio_acute (h_acute : ∀ (A B C : ℝ), 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧ A + B + C = π) :
  ∀ (K Δ : ℝ), K / Δ = 1 - (cos A) ^ 2 - (cos B) ^ 2 - (cos C) ^ 2 :=
sorry

theorem area_ratio_obtuse (h_obtuse : ∃ A B C : ℝ, (A > π/2 ∨ B > π/2 ∨ C > π/2) ∧ A + B + C = π) :
  ∀ (K Δ : ℝ), K / Δ = (cos A) ^ 2 + (cos B) ^ 2 + (cos C) ^ 2 - 1 :=
sorry

end area_ratio_acute_area_ratio_obtuse_l418_418183


namespace max_of_function_in_interval_l418_418330

noncomputable def max_value (x : ℝ) : ℝ := 2^(2*x-1) - 3 * 2^x + 5

theorem max_of_function_in_interval :
  ∃ x (h₀ : 0 ≤ x) (h₂ : x ≤ 2),  max_value x = 5/2 :=
sorry

end max_of_function_in_interval_l418_418330


namespace pages_copied_for_15_dollars_l418_418806

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l418_418806


namespace solve_arcsin_equation_l418_418508

noncomputable def solution (x : ℝ) : Prop :=
  arcsin x + arcsin (3 * x) = π / 4

theorem solve_arcsin_equation :
  solution (sqrt (2 / 51)) :=
by sorry

end solve_arcsin_equation_l418_418508


namespace g_half_l418_418663

noncomputable def g : ℝ → ℝ := sorry

axiom g0 : g 0 = 0
axiom g1 : g 1 = 1
axiom g_non_decreasing : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom g_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom g_fraction : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

theorem g_half : g (1 / 2) = 1 / 2 := sorry

end g_half_l418_418663


namespace seed_placement_count_l418_418325

theorem seed_placement_count :
  ∀ (S : Finset ℕ) (A B : ℕ),
    S.card = 10 → 
    A ∈ S → 
    B ∈ S → 
    (S.erase A).erase B.card = 8 →
    ∃ (C₁ C₅ : ℕ),
      C₁ = (Nat.choose 8 1) ∧
      C₅ = Nat.fact 9 / Nat.fact (9 - 5) →
      C₁ * C₅ = Nat.choose 8 1 * Nat.fact 9 / Nat.fact (9 - 5) := sorry

end seed_placement_count_l418_418325


namespace function_extremum_has_value_at_1_l418_418730

def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

def f_prime (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem function_extremum_has_value_at_1 :
  ∃ (a b : ℝ), f a b 1 = 10 ∧ f_prime a b 1 = 0 ∧ f a b = λ x, x^3 + 4 * x^2 - 11 * x + 16 :=
by
  sorry

end function_extremum_has_value_at_1_l418_418730


namespace Chinese_conversation_among_engineers_l418_418184

theorem Chinese_conversation_among_engineers :
  ∃ (s : Finset (Fin 2017)), s.card = 673 ∧ 
  ∀ (x y : Fin 2017), x ∈ s → y ∈ s → x ≠ y → converse_in_chinese x y :=
sorry

def converse_in_chinese (x y : Fin 2017) : Prop :=
  sorry

-- Conditions from the problem converted to Lean
constant converse : Fin 2017 → Fin 2017 → Bool → Prop

-- No two engineers converse with each other more than once
axiom no_duplicate_conversations :
  ∀ (x y : Fin 2017) (b1 b2 : Bool), converse x y b1 → converse x y b2 → b1 = b2

-- Within any four engineers, there is an even number of conversations
axiom even_conversations :
  ∀ (a b c d : Fin 2017), (converse a b True + converse a c True + converse a d True + converse b c True + converse b d True + converse c d True) % 2 = 0

-- At least one of these conversations is in Chinese
axiom at_least_one_chinese :
  ∀ (a b c d : Fin 2017),
    (converse a b True ∨ converse a c True ∨ converse a d True ∨ converse b c True ∨ converse b d True ∨ converse c d True)

-- Either no conversations are in English, or the number of English conversations is at least that of Chinese conversations
axiom english_conversations_bound :
  ∀ (a b c d : Fin 2017),
    (¬(∃ x y, converse x y False) 
     ∨ (∑ x y in (Finset.univ : Finset (Fin 2017)).product (Finset.univ), if converse x y False then 1 else 0) 
         ≥ (∑ x y in (Finset.univ : Finset (Fin 2017)).product (Finset.univ), if converse x y True then 1 else 0))

end Chinese_conversation_among_engineers_l418_418184


namespace octahedron_probability_sum_eq_85_l418_418539

/-- The numbers 1, 2, 3, 4, 5, 6, 7, and 8 are randomly written on the faces of a regular octahedron
so that each face contains a different number. Consider 1 and 8 to be consecutive. The probability 
that no two consecutive numbers are written on faces that share an edge is m/n, where m and n are 
relatively prime positive integers. Prove m + n = 85. -/
theorem octahedron_probability_sum_eq_85 : 
  let faces := {1, 2, 3, 4, 5, 6, 7, 8}
  let m, n be positive integers such that gcd m n = 1 
  (∃ m n : ℕ, gcd m n = 1 ∧ (prob_no_consecutive_shared_edge faces = m / n)) → 
  m + n = 85 :=
begin
  sorry
end

end octahedron_probability_sum_eq_85_l418_418539


namespace election_winner_l418_418561

theorem election_winner (vote_Joao : ℚ) (vote_Rosa : ℚ) (total_votes : ℚ) :
  vote_Joao = 2 / 7 → vote_Rosa = 2 / 5 → total_votes = 1 →
  let vote_Marcos := total_votes - (vote_Joao + vote_Rosa) in
  (vote_Rosa > vote_Joao) ∧ (vote_Rosa > vote_Marcos) :=
by
  intros hJ hR hT
  let vote_Marcos := total_votes - (vote_Joao + vote_Rosa)
  have h_vote_Joao : vote_Joao = 2 / 7 := hJ
  have h_vote_Rosa : vote_Rosa = 2 / 5 := hR
  have h_vote_Marcos : vote_Marcos = 1 - (2 / 7 + 2 / 5) := by sorry 
  have h_cmp1 : 2 / 5 > 2 / 7 := by sorry 
  have h_cmp2 : 2 / 5 > vote_Marcos := by sorry 
  exact ⟨h_cmp1, h_cmp2⟩

end election_winner_l418_418561


namespace razorback_tshirts_sold_l418_418965

variable (T : ℕ) -- Number of t-shirts sold
variable (price_per_tshirt : ℕ := 62) -- Price of each t-shirt
variable (total_revenue : ℕ := 11346) -- Total revenue from t-shirts

theorem razorback_tshirts_sold :
  (price_per_tshirt * T = total_revenue) → T = 183 :=
by
  sorry

end razorback_tshirts_sold_l418_418965


namespace number_of_possible_monograms_l418_418471

/-- 
Prove that the number of possible monograms, 
given the first initial is 'A', the middle and last initials 
are distinct letters from the remaining 25 letters of the alphabet,
and the initials are in alphabetical order, is 300.
--/
theorem number_of_possible_monograms : 
  (∃ X Y : Fin 26, X ≠ 0 ∧ Y ≠ 0 ∧ X ≠ Y ∧ X < Y ∧ X ≠ 0 ∧ Y ≠ 0) → Finset.card (Finset.filter (fun (p : Fin 26 × Fin 26) => p.1 < p.2) ((Finset.univ.product Finset.univ).filter (λ p, p.1 ≠ 0 ∧ p.2 ≠ 0))) = 300 := 
sorry

end number_of_possible_monograms_l418_418471


namespace sequence_exists_l418_418316

variable (n : ℕ)

theorem sequence_exists (k : ℕ) (hkn : k ∈ Set.range (λ x : ℕ, x + 1) n) :
  ∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i → i ≤ n → x (i+1) > x i) ∧ (∀ i, x i ∈ ℕ) :=
sorry

end sequence_exists_l418_418316


namespace prime_squares_mod_180_l418_418254

theorem prime_squares_mod_180 (p : ℕ) (hp : prime p) (hp_gt_5 : p > 5) :
  ∃ (r1 r2 : ℕ), 
  r1 ≠ r2 ∧ 
  ∀ r : ℕ, (∃ m : ℕ, p^2 = m * 180 + r) → (r = r1 ∨ r = r2) :=
sorry

end prime_squares_mod_180_l418_418254


namespace first_method_error_second_method_error_better_method_l418_418242

noncomputable def height_equilateral_triangle (r : ℝ) : ℝ := (Real.sqrt 3 / 2) * r

noncomputable def side_heptagon_exact (r : ℝ) : ℝ := 0.867768 * r

noncomputable def side_new_method (r : ℝ) : ℝ := 0.867806 * r

theorem first_method_error (r : ℝ) :
  let m := height_equilateral_triangle r
      exact_side := side_heptagon_exact r in
  (((exact_side - m) * 1000) / exact_side) = 2.01 :=
by
  let m := height_equilateral_triangle r
  let exact_side := side_heptagon_exact r
  have h1 : m = 0.866025 * r := by sorry
  have h2 : exact_side = 0.867768 * r := by sorry
  have h_diff : exact_side - m = 0.001743 * r := by sorry
  have permil_error := ((h_diff * 1000) / exact_side)
  show permil_error = 2.01 := by sorry

theorem second_method_error (r : ℝ) :
  let approx_side := side_new_method r
      exact_side := side_heptagon_exact r in
  (((approx_side - exact_side) * 1000) / exact_side) = 0.0438 :=
by
  let approx_side := side_new_method r
  let exact_side := side_heptagon_exact r
  have h1 : approx_side = 0.867806 * r := by sorry
  have h2 : exact_side = 0.867768 * r := by sorry
  have h_diff : approx_side - exact_side = 0.000038 * r := by sorry
  have permil_error := ((h_diff * 1000) / exact_side)
  show permil_error = 0.0438 := by sorry

theorem better_method (r : ℝ) :
  let error_old := 2.01
      error_new := 0.0438 in
  error_new < error_old :=
by
  have h := (0.0438 : ℝ) < (2.01 : ℝ)
  show h = true := by sorry

end first_method_error_second_method_error_better_method_l418_418242


namespace power_of_p_l418_418028

theorem power_of_p (x y p n k : ℕ) (hx : 0 < x) (hy : 0 < y) (hp : 0 < p) 
                   (hn : 0 < n) (hk : 0 < k) (h_eq : x ^ n + y ^ n = p ^ k)
                   (h_odd_n : odd n) (h_odd_p : Nat.Prime p ∧ odd p) :
  ∃ (m : ℕ), n = p ^ m := 
sorry

end power_of_p_l418_418028


namespace range_of_m_l418_418589

theorem range_of_m (m : ℝ) :
  ((-2 < m ∧ m ≤ -2 / 3) ∨ m = 0 ∨ (2 / 3 ≤ m ∧ m < 2)) ↔  
  let Δ := λ m, int_partitions (λ h, (ellipse h ∧ (lines_best_case h) ∧ (parts_coloring h)))
  Δ m = 720 :=
begin
  sorry
end

end range_of_m_l418_418589


namespace sum_two_primes_eq_91_prod_is_178_l418_418133

theorem sum_two_primes_eq_91_prod_is_178
  (p1 p2 : ℕ) 
  (hp1 : p1.Prime) 
  (hp2 : p2.Prime) 
  (h_sum : p1 + p2 = 91) :
  p1 * p2 = 178 := 
sorry

end sum_two_primes_eq_91_prod_is_178_l418_418133


namespace cubic_eq_solutions_l418_418521

theorem cubic_eq_solutions (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : ∀ x, x^3 + a * x^2 + b * x + c = 0 → (x = a ∨ x = -b ∨ x = c)) : (a, b, c) = (1, -1, -1) := 
by {
  -- Convert solution steps into a proof
  sorry
}

end cubic_eq_solutions_l418_418521


namespace solve_arcsin_equation_l418_418507

noncomputable def solution (x : ℝ) : Prop :=
  arcsin x + arcsin (3 * x) = π / 4

theorem solve_arcsin_equation :
  solution (sqrt (2 / 51)) :=
by sorry

end solve_arcsin_equation_l418_418507


namespace pooja_speed_l418_418897

theorem pooja_speed (v : ℝ) 
  (roja_speed : ℝ := 5)
  (distance : ℝ := 32)
  (time : ℝ := 4)
  (h : distance = (roja_speed + v) * time) : v = 3 :=
by
  sorry

end pooja_speed_l418_418897


namespace pages_copied_for_15_dollars_l418_418807

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l418_418807


namespace min_cost_of_planting_l418_418429

noncomputable def area (length width : ℝ) : ℝ := length * width

noncomputable def multiplyCost (costPerSquareMeter area : ℝ) : ℝ :=
  costPerSquareMeter * area

theorem min_cost_of_planting :
  let
    region1 := area 2 7,
    region2 := area 5 7,
    region3 := area 5 2,
    region4 := area 3 7,
    region5 := area 4 9,

    costs := [3.00, 2.50, 2.00, 1.50, 1.00],
    regions := [region1, region2, region3, region4, region5],

    sortedRegions := regions.qsort (· < ·),
    minCost := 
      multiplyCost costs[0] sortedRegions[0] +
      multiplyCost costs[1] sortedRegions[1] +
      multiplyCost costs[2] sortedRegions[2] +
      multiplyCost costs[3] sortedRegions[3] +
      multiplyCost costs[4] sortedRegions[4]

  in minCost = 195.50 :=
by
  sorry

end min_cost_of_planting_l418_418429


namespace largest_gcd_of_ten_numbers_with_sum_1001_is_91_l418_418853

theorem largest_gcd_of_ten_numbers_with_sum_1001_is_91 :
  ∃ (d : ℕ), (∀ a : ℕ, (∃ b : ℕ, a = d * b) → (∃ l : list ℕ, l.length = 10 ∧ list.sum l = 1001 ∧ ∀ x ∈ l, d ∣ x)) ∧ d = 91 := 
sorry

end largest_gcd_of_ten_numbers_with_sum_1001_is_91_l418_418853


namespace differenceInCents_l418_418882

variable (p n d : ℕ)

noncomputable def maxCoinsValue (p n d : ℕ) : ℕ := p + 5 * n + 10 * d 
noncomputable def minCoinsValue (p n d : ℕ) : ℕ := p + 5 * n + 10 * d

theorem differenceInCents (h1 : p + n + d = 3030) (h2 : p ≥ 1) (h3 : n ≥ 1) (h4 : d ≥ 1) : 
  ∃ p n d, 
    let maxValue := maxCoinsValue 1 1 (3030 - 1 - 1) 
    let minValue := minCoinsValue 3028 1 1 
    maxValue - minValue = 27243 :=
begin
  sorry
end

end differenceInCents_l418_418882


namespace mallory_total_expense_l418_418115

theorem mallory_total_expense
  (cost_per_refill : ℕ)
  (distance_per_refill : ℕ)
  (total_distance : ℕ)
  (food_ratio : ℚ)
  (refill_count : ℕ)
  (total_fuel_cost : ℕ)
  (total_food_cost : ℕ)
  (total_expense : ℕ)
  (h1 : cost_per_refill = 45)
  (h2 : distance_per_refill = 500)
  (h3 : total_distance = 2000)
  (h4 : food_ratio = 3 / 5)
  (h5 : refill_count = total_distance / distance_per_refill)
  (h6 : total_fuel_cost = refill_count * cost_per_refill)
  (h7 : total_food_cost = (food_ratio * ↑total_fuel_cost).to_nat)
  (h8 : total_expense = total_fuel_cost + total_food_cost) :
  total_expense = 288 := by
  sorry

end mallory_total_expense_l418_418115


namespace calculate_extra_fica_taxes_per_week_l418_418834

variable {weekly_hours : ℕ} (current_hourly : ℝ) (freelance_hourly : ℝ) (healthcare_premium_monthly : ℝ) (extra_income_monthly : ℝ) (weeks_per_month : ℕ) (fica_taxes_per_week : ℝ)

def weekly_income (hourly_wage : ℝ) (hours_worked : ℕ) : ℝ :=
  hourly_wage * hours_worked

def monthly_income (weekly_wage : ℝ) (weeks_per_month : ℕ) : ℝ :=
  weekly_wage * weeks_per_month

def required_monthly_fica_taxes (freelance_income : ℝ) (current_income : ℝ) (healthcare_premium : ℝ) (extra_income : ℝ) : ℝ :=
  freelance_income - current_income - healthcare_premium - extra_income

noncomputable def required_weekly_fica_taxes (monthly_fica_taxes : ℝ) (weeks_per_month : ℕ) : ℝ :=
  monthly_fica_taxes / weeks_per_month

theorem calculate_extra_fica_taxes_per_week (T : weekly_hours = 40) (C : current_hourly = 30) (F : freelance_hourly = 40)
    (H : healthcare_premium_monthly = 400) (E : extra_income_monthly = 1100) (W : weeks_per_month = 4) : 
  fica_taxes_per_week = 25 := 
  by
    -- Variables definitions
    let current_weekly_income := weekly_income current_hourly weekly_hours
    let freelance_weekly_income := weekly_income freelance_hourly weekly_hours
    let current_monthly_income := monthly_income current_weekly_income weeks_per_month
    let freelance_monthly_income := monthly_income freelance_weekly_income weeks_per_month
    let required_monthly_fica := required_monthly_fica_taxes freelance_monthly_income current_monthly_income healthcare_premium_monthly extra_income_monthly
    let required_weekly_fica := required_weekly_fica_taxes required_monthly_fica weeks_per_month
    show fica_taxes_per_week = 25 from sorry

end calculate_extra_fica_taxes_per_week_l418_418834


namespace min_x2_y2_l418_418351

theorem min_x2_y2 {x y : ℝ} (h : (x - 2)^2 + (y - 2)^2 = 1) : x^2 + y^2 ≥ 9 - 4 * Real.sqrt 2 :=
sorry

end min_x2_y2_l418_418351


namespace new_batting_average_l418_418595

def initial_runs (A : ℕ) := 16 * A
def additional_runs := 85
def increased_average := 3
def runs_in_5_innings := 100 + 120 + 45 + 75 + 65
def total_runs_17_innings (A : ℕ) := 17 * (A + increased_average)
def A : ℕ := 34
def total_runs_22_innings := total_runs_17_innings A + runs_in_5_innings
def number_of_innings := 22
def new_average := total_runs_22_innings / number_of_innings

theorem new_batting_average : new_average = 47 :=
by sorry

end new_batting_average_l418_418595


namespace triangle_area_l418_418534

noncomputable def area_of_triangle (BD DC : ℝ) (R : ℝ) : ℝ :=
  if angle_A = 60 then
    12 * Real.sqrt 3 
  else if angle_A = 120 then
    4 * Real.sqrt 3 
  else
    0

theorem triangle_area (BD DC R : ℝ) :
  BD = 3 → DC = 4 → R = 7 / Real.sqrt 3 →
  (area_of_triangle BD DC R = 12 * Real.sqrt 3 ∨
   area_of_triangle BD DC R = 4 * Real.sqrt 3) :=
by sorry

end triangle_area_l418_418534


namespace vector_sum_eq_l418_418743

variables (x y : ℝ)
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, 3)
def c : ℝ × ℝ := (7, 8)

theorem vector_sum_eq :
  ∃ (x y : ℝ), c = (x • a.1 + y • b.1, x • a.2 + y • b.2) ∧ x + y = 8 / 3 :=
by
  have h1 : 7 = 2 * x + 3 * y := sorry
  have h2 : 8 = 3 * x + 3 * y := sorry
  sorry

end vector_sum_eq_l418_418743


namespace initial_quantity_of_liquid_A_l418_418596

theorem initial_quantity_of_liquid_A
  (x : ℝ)
  (h_ratio : 7 * x > 0)
  (h_mixture_removed : (7/12) * 9 > 0) :
  let initial_liquid_A := 7 * x,
      initial_liquid_B := 5 * x,
      total_initial_mixture := 12 * x,
      liquid_A_removed := (7/12) * 9,
      liquid_B_removed := (5/12) * 9,
      remaining_liquid_A := 7 * x - (7/12) * 9,
      remaining_liquid_B := 5 * x - (5/12) * 9 + 9,
      final_ratio := remaining_liquid_A / remaining_liquid_B in
  final_ratio = 7 / 9 → 
  initial_liquid_A = 20.8125 :=
begin
  sorry
end

end initial_quantity_of_liquid_A_l418_418596


namespace cone_minimum_volume_l418_418207

noncomputable def minimumVolume (a : ℝ) : ℝ := (9 / 8) * π * a^3

theorem cone_minimum_volume (a : ℝ) : (∃ V : ℝ, 
V = minimumVolume a 
∧ ∀ h : ℝ, h > 0 → ∀ r : ℝ, r > 0 → 
∃ V₀ : ℝ, 
  V₀ = (1 / 3) * π * r^2 * h 
  ∧ (∃ h₀ : ℝ, h₀ = 3 * a → ∃ r₀ : ℝ, 
  r₀ = (a * h * sqrt(2)) / (2 * (h - a))
  ∧ V₀ = V)) :=
  sorry

end cone_minimum_volume_l418_418207


namespace passes_through_point_l418_418751

theorem passes_through_point (a : ℝ) (h : a > 0) (h2 : a ≠ 1) : 
  (2, 1) ∈ {p : ℝ × ℝ | ∃ a, p.snd = a * p.fst - 2} :=
sorry

end passes_through_point_l418_418751


namespace range_x_0_l418_418371

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem range_x_0 (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  f x > f (Real.pi / 6) ↔ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6) ∪ Set.Ioi (Real.pi / 6) ∩ Set.Icc (Real.pi / 6) (Real.pi / 2) := 
  by
    sorry

end range_x_0_l418_418371


namespace det_A_l418_418657

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![2, -4, 5],
  ![0, 6, -2],
  ![3, -1, 2]
]

theorem det_A : A.det = -46 := by
  sorry

end det_A_l418_418657


namespace base_length_of_parallelogram_l418_418967

theorem base_length_of_parallelogram (area : ℝ) (base altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base) :
  base = 7 :=
by
  sorry

end base_length_of_parallelogram_l418_418967


namespace john_fixed_computers_l418_418015

theorem john_fixed_computers (total_computers unfixable waiting_for_parts fixed_right_away : ℕ)
  (h1 : total_computers = 20)
  (h2 : unfixable = 0.20 * 20)
  (h3 : waiting_for_parts = 0.40 * 20)
  (h4 : fixed_right_away = total_computers - unfixable - waiting_for_parts) :
  fixed_right_away = 8 :=
by
  sorry

end john_fixed_computers_l418_418015


namespace shortest_distance_parabola_to_line_l418_418701

theorem shortest_distance_parabola_to_line :
  let d (x y : ℝ) : ℝ := abs (2 * x + 4 * y + 5) / real.sqrt (2 ^ 2 + 4 ^ 2)
  ∃ (x y : ℝ), x^2 = 4 * y ∧ d x y = (2 * real.sqrt 5) / 5 :=
begin
  sorry
end

end shortest_distance_parabola_to_line_l418_418701


namespace sequence_nat_nums_exists_l418_418308

theorem sequence_nat_nums_exists (n : ℕ) : { k : ℕ | ∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i, 1 ≤ i → i ≤ n → x i)} = { k | 1 ≤ k ∧ k ≤ n } :=
sorry

end sequence_nat_nums_exists_l418_418308


namespace cubical_pyramidal_segment_volume_and_area_l418_418209

noncomputable def volume_and_area_sum (a : ℝ) : ℝ :=
  (1/4 * (9 + 27 * Real.sqrt 13))

theorem cubical_pyramidal_segment_volume_and_area :
  ∀ a : ℝ, a = 3 → volume_and_area_sum a = (9/2 + 27 * Real.sqrt 13 / 8) := by
  intro a ha
  sorry

end cubical_pyramidal_segment_volume_and_area_l418_418209


namespace coefficient_x3_in_expansion_of_1_plus_x_pow_5_l418_418531

theorem coefficient_x3_in_expansion_of_1_plus_x_pow_5 :
  (∃ c : ℕ, coeff (mk (1 + X)^5) 3 = c) → c = 10 :=
begin
  sorry
end

end coefficient_x3_in_expansion_of_1_plus_x_pow_5_l418_418531


namespace angus_total_investment_l418_418247

variable (x T : ℝ)

theorem angus_total_investment (h1 : 0.03 * x + 0.05 * 6000 = 660) (h2 : T = x + 6000) : T = 18000 :=
by
  sorry

end angus_total_investment_l418_418247


namespace vector_magnitude_example_l418_418529

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_example :
  let a : ℝ × ℝ := (2, 0)
  let b : ℝ × ℝ := (cos (π / 3), sin (π / 3))  -- using the fact that the magnitude of b is 1 and angle is 60 degrees
  in magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = 2 * real.sqrt 3 := 
  sorry

end vector_magnitude_example_l418_418529


namespace max_prime_factors_c_l418_418106

theorem max_prime_factors_c
  (c d : ℕ)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_gcd : (Nat.gcd c d).prime_factors.length = 10)
  (h_lcm : (Nat.lcm c d).prime_factors.length = 40)
  (h_fewer_factors : c.prime_factors.length < d.prime_factors.length) :
  c.prime_factors.length ≤ 25 :=
by
  sorry

end max_prime_factors_c_l418_418106


namespace max_traffic_flow_rate_traffic_flow_rate_at_least_10_l418_418138

noncomputable def traffic_flow : ℝ → ℝ :=
  λ v, 920 * v / (v^2 + 3 * v + 1600)

theorem max_traffic_flow_rate :
  ∃ v : ℝ, v > 0 ∧ traffic_flow v = 920 / 83 :=
sorry

theorem traffic_flow_rate_at_least_10 :
  ∀ v : ℝ, traffic_flow v ≥ 10 ↔ 25 ≤ v ∧ v ≤ 64 :=
sorry

end max_traffic_flow_rate_traffic_flow_rate_at_least_10_l418_418138


namespace min_k_disjoint_pairs_l418_418571

open Finset

def friendly_pair {α : Type*} (rel : α → α → Prop) (x y : α) : Prop :=
rel x y

def non_friendly_pair {α : Type*} (rel : α → α → Prop) (x y : α) : Prop :=
¬rel x y

def exists_disjoint_pairs_of_size {α : Type*} (s : Finset α) (rel : α → α → Prop) (k : ℕ) : Prop :=
∃ (p : Finset (Finset α)),
  p.card = k ∧
  (∀ t ∈ p, t.card = 2 ∧ (∃ x y, t = {x, y} ∧ rel x y)) ∧
  pairwise_disjoint id p

theorem min_k_disjoint_pairs (m n : ℕ) : ∃ k : ℕ, ∀ (s : Finset (Σ i, Type) ) (rel : α → α → Prop), 
  s.card >= k → (exists_disjoint_pairs_of_size {a // a ∈ s} rel m ∨ exists_disjoint_pairs_of_size {a // a ∈ s} (non_friendly_pair rel) n) :=
begin
  use 2 * m + n - 1,
  sorry
end

end min_k_disjoint_pairs_l418_418571


namespace interchange_grades_l418_418134

theorem interchange_grades (a b : ℕ) (a_ne_zero : a ≠ 0) (a_ne_hundred : a ≠ 100) (b_ne_zero : b ≠ 0) (b_ne_hundred : b ≠ 100) :
  ∃ seq : list ℕ, valid_transformations a b seq := 
sorry

end interchange_grades_l418_418134


namespace region_R_area_l418_418440

-- Definition of the kite ABCD with given side lengths and angle
structure Kite :=
(AB AD BC CD : ℝ)
(angle_B : ℝ)

-- Condition definitions for kite ABCD
def kite_ABCD : Kite := 
{ AB := 3, AD := 3, BC := 5, CD := 5, angle_B := 100 }

-- Definition of region R within kite ABCD
def region_R (K : Kite) := 
{ p | dist p K.A < min (dist p K.B) (min (dist p K.C) (dist p K.D)) }

-- The theorem stating that the area of region R is 5.
theorem region_R_area {K : Kite} (h : K = kite_ABCD) : 
  area (region_R K) = 5 :=
sorry

end region_R_area_l418_418440


namespace average_of_first_n_multiples_of_8_is_88_l418_418156

theorem average_of_first_n_multiples_of_8_is_88 (n : ℕ) (h : (n / 2) * (8 + 8 * n) / n = 88) : n = 21 :=
sorry

end average_of_first_n_multiples_of_8_is_88_l418_418156


namespace investment_for_future_value_l418_418750

-- Define the parameters and expected output
noncomputable def present_value (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r)^n

theorem investment_for_future_value :
  present_value 600000 0.06 10 ≈ 335267.29 := by
  sorry

end investment_for_future_value_l418_418750


namespace planet_combination_proof_l418_418385

theorem planet_combination_proof :
  let earth_like := 7
  let mars_like := 6
  let total_colonization_units := 15
  (∑ a in (finset.range (total_colonization_units // 3 + 1)),
    if 3 * a ≤ total_colonization_units ∧ total_colonization_units - 3 * a <= mars_like then
      (nat.choose earth_like a) * (nat.choose mars_like (total_colonization_units - 3 * a))
    else 0) = 736 :=
by
  sorry

end planet_combination_proof_l418_418385


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418924

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418924


namespace sam_current_yellow_marbles_l418_418055

theorem sam_current_yellow_marbles (original_yellow : ℕ) (taken_yellow : ℕ) (current_yellow : ℕ) :
  original_yellow = 86 → 
  taken_yellow = 25 → 
  current_yellow = original_yellow - taken_yellow → 
  current_yellow = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_current_yellow_marbles_l418_418055


namespace incorrect_conclusion_l418_418169

theorem incorrect_conclusion : 
  (¬ (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → exp x ≥ 1) ∨ ¬ (∃ x : ℝ, x^2 + x + 1 < 0)) → B := sorry

end incorrect_conclusion_l418_418169


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418938

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418938


namespace solve_for_x_l418_418095

theorem solve_for_x : ∃ x : ℝ, (x - 5)^4 = (1 / 16)⁻² ∧ x = 9 :=
by
  sorry

end solve_for_x_l418_418095


namespace sequence_exists_for_all_k_l418_418303

theorem sequence_exists_for_all_k (n : ℕ) :
  ∀ k : ℕ, (k ∈ {1, 2, ..., n}) ↔ (∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i < n, x i > 0) ∧ (∃ i, x(i) = k)) :=
by
  sorry

end sequence_exists_for_all_k_l418_418303


namespace part1_part2_l418_418979

open Real

-- Define the function f(x) based on the given condition a ∈ ℝ
def f (x : ℝ) (a : ℝ) : ℝ := (x - 2) * exp x - a * x^2 + 2 * a * x

-- Part 1: Prove that when a = 0, f(x) + exp(1) ≥ 0 for all x ∈ ℝ
theorem part1 : ∀ x : ℝ, f x 0 + exp 1 ≥ 0 :=
  sorry

-- Part 2: Prove that if x = 1 is a local maximum point of f(x), then the range of a is (exp(1) / 2, +∞)
theorem part2 : (∃ a : ℝ, f (1 : ℝ) a = 0 ∧ ∀ x, (x < 1 → f x a < f 1 a) ∧ (x > 1 → f x a < f 1 a)) → (exp 1 / 2 < a) :=
  sorry

end part1_part2_l418_418979


namespace problem_l418_418718

variable (x y : ℝ)

-- Define the given condition
def condition : Prop := |x + 5| + (y - 4)^2 = 0

-- State the theorem we need to prove
theorem problem (h : condition x y) : (x + y)^99 = -1 := sorry

end problem_l418_418718


namespace eval_derivative_at_one_and_neg_one_l418_418042

def f (x : ℝ) : ℝ := x^4 + x - 1

theorem eval_derivative_at_one_and_neg_one : 
  (deriv f 1) + (deriv f (-1)) = 2 :=
by 
  -- proof to be filled in
  sorry

end eval_derivative_at_one_and_neg_one_l418_418042


namespace ming_first_round_probability_ming_more_likely_advance_l418_418203

-- Definitions for the problem conditions
def num_questions_A : ℕ := 5
def num_questions_B : ℕ := 5
def first_round_points_all_correct : ℕ := 40
def second_round_points_per_correct : ℕ := 30
def points_to_advance : ℕ := 60
def prob_correct_fang : ℚ := 0.5
def prob_correct_ming_B : ℚ := 0.4
def total_combinations_A : ℕ := 10
def ming_successful_combinations_A : ℕ := 6

-- Part 1 
theorem ming_first_round_probability : 
  (ming_successful_combinations_A : ℚ) / total_combinations_A = 3 / 5 := 
sorry

-- Definitions for Part 2 calculations
def ming_first_round_prob : ℚ := 3 / 5
def ming_first_round_fail_prob : ℚ := 2 / 5

-- Scenarios probability setup for second round
-- Scenario 1: Ming scores 40 points in the first round and enough in the second round
def ming_second_round_prob1 : ℚ := ming_first_round_prob * (prob_correct_ming_B * prob_correct_ming_B) * 2 -- redundant multiplication for combinatory, adjust as per exact need

-- Scenario 2: Adjust probabilities for new combinations, similar structure as described in text

-- Summarize both
def ming_advance_prob : ℚ := ming_second_round_prob1 + 
-- appropriately add other calculated probabilities according to the initial problem

-- Placeholder probability for Fang for similar steps
def fang_advance_prob : ℚ := 0.375 -- assume similar structured steps as shown in original computation

theorem ming_more_likely_advance :
  ming_advance_prob > fang_advance_prob := 
sorry

end ming_first_round_probability_ming_more_likely_advance_l418_418203


namespace molecular_weight_of_NH4Br_l418_418158

def atomic_weight (element : String) : Real :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | _ => 0.0

def molecular_weight (composition : List (String × Nat)) : Real :=
  composition.foldl (λ acc (elem, count) => acc + count * atomic_weight elem) 0

theorem molecular_weight_of_NH4Br :
  molecular_weight [("N", 1), ("H", 4), ("Br", 1)] = 97.95 :=
by
  sorry

end molecular_weight_of_NH4Br_l418_418158


namespace largest_divisible_by_6_ending_in_4_l418_418916

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l418_418916


namespace radii_proof_l418_418974

noncomputable def to_radians (d m : ℝ) : ℝ :=
  d * (π / 180) + m * (π / (180 * 60))

noncomputable def radius_R (c : ℝ) (alpha beta : ℝ) : ℝ :=
  c * real.sin ((beta + alpha) / 4) * real.cos ((beta - alpha) / 4)

noncomputable def radius_r (c : ℝ) (alpha beta : ℝ) : ℝ :=
  c * real.cos ((beta + alpha) / 4) * real.sin ((beta - alpha) / 4)

-- Given values
def c : ℝ := 714
def alpha_d : ℝ := 36
def alpha_m : ℝ := 8
def beta_d : ℝ := 104
def beta_m : ℝ := 12

-- Convert angles to radians
def alpha : ℝ := to_radians alpha_d alpha_m
def beta : ℝ := to_radians beta_d beta_m

-- Hypotheses
axiom h1 : R ≈ 392.42
axiom h2 : r ≈ 170.99

-- The result we want to prove
theorem radii_proof :
  radius_R c alpha beta ≈ 392.42 ∧
  radius_r c alpha beta ≈ 170.99 :=
begin
  -- Proof goes here
  sorry
end

end radii_proof_l418_418974


namespace quadrilateral_classification_l418_418052

theorem quadrilateral_classification
  (α β γ δ : ℝ)
  (h1 : cos α + cos β + cos γ + cos δ = 0)
  (h2 : α + β + γ + δ = 2 * π) :
  (∃ (ABCD : Type) [Is_parallelogram ABCD] , True) ∨
  (∃ (ABCD : Type) [Is_trapezoid ABCD], True) ∨
  (∃ (ABCD : Type) [Is_cyclic_quadrilateral ABCD], True) :=
sorry

end quadrilateral_classification_l418_418052


namespace anthony_painting_time_l418_418835

-- Define constants and conditions
def kathleen_rate := 1 / 3
def combined_rate := 2 / 3.428571428571429

-- Define the target value for Anthony's rate
def anthony_rate := 1 / 4

-- State the theorem
theorem anthony_painting_time : 
  ∃ A : ℝ, (kathleen_rate + (1 / A) = combined_rate) → A = 4 :=
by sorry

end anthony_painting_time_l418_418835


namespace solve_for_x_l418_418513

def equation (x : ℝ) (y : ℝ) : Prop := 5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)

def y_condition (x : ℝ) : ℝ := 3 * x

theorem solve_for_x (x : ℝ) :
  equation x (y_condition x) ↔ (x = 1/3 ∨ x = -2/9) := by
  sorry

end solve_for_x_l418_418513


namespace fixed_term_sequence_l418_418741

variable {b c : ℕ}
variable {a : ℕ → ℕ}

-- Conditions
def a_1 (b c : ℕ) := b * (b + 1) / 2 + c

def a_n_plus_1 (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if a n ≤ n then a n + n else a n - n

-- Proof of first fixed term
def first_fixed_term (b c : ℕ) (h_bc : c < b) : Prop :=
  b + 2 * c

-- General form of the fixed term sequence
def general_form (b c m : ℕ) : ℕ :=
  ((2 * b + 4 * c - 1) * 3 ^ (m - 1) + 1) / 2

theorem fixed_term_sequence {b c : ℕ} (h_bc : c < b):
  first_fixed_term b c h_bc = b + 2 * c ∧
  ∀ (m : ℕ) (hm : m > 0), general_form b c m = ((2 * b + 4 * c - 1) * 3 ^ (m - 1) + 1) / 2 :=
by
  sorry

end fixed_term_sequence_l418_418741


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l418_418909

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l418_418909


namespace smaller_fraction_l418_418550

variable (x y : ℚ)

theorem smaller_fraction (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 1 / 6 :=
by
  sorry

end smaller_fraction_l418_418550


namespace not_exists_18_consecutive_good_l418_418869

def is_good (n : ℕ) : Prop := 
  (∀ p : ℕ, nat.prime p → p ∣ n → nat.factor_multiset p n = 2) 

theorem not_exists_18_consecutive_good :
  ¬ ∃ (a : ℕ), ∀ (i : ℕ), i < 18 → is_good (a + i) :=
sorry

end not_exists_18_consecutive_good_l418_418869


namespace find_ratio_of_a_b_l418_418724

noncomputable def slope_of_tangent_to_curve_at_P := 3 * 1^2 + 1

noncomputable def perpendicular_slope (a b : ℝ) : Prop :=
  slope_of_tangent_to_curve_at_P * (a / b) = -1

noncomputable def line_slope_eq_slope_of_tangent (a b : ℝ) : Prop := 
  perpendicular_slope a b

theorem find_ratio_of_a_b (a b : ℝ) 
  (h1 : a - b * 2 = 0) 
  (h2 : line_slope_eq_slope_of_tangent a b) : 
  a / b = -1 / 4 :=
by
  sorry

end find_ratio_of_a_b_l418_418724


namespace distance_from_P_to_face_ABC_l418_418891

noncomputable def PA : ℝ := 15
noncomputable def PB : ℝ := 15
noncomputable def PC : ℝ := 9

theorem distance_from_P_to_face_ABC (A B C P : EuclideanSpace ℝ) 
  (h1 : dist P A = PA) 
  (h2 : dist P B = PB) 
  (h3 : dist P C = PC) 
  (h4 : ∀ (X Y Z : EuclideanSpace ℝ), ∠ (X - P) (Y - P) = 90 ∧ ∠ (Y - P) (Z - P) = 90 ∧ ∠ (Z - P) (X - P) = 90) :
  ∃ (d : ℝ), d = 3 * sqrt 3 :=
by sorry

end distance_from_P_to_face_ABC_l418_418891


namespace boxes_given_away_l418_418837

-- Given: She baked 53 lemon cupcakes.
-- Given: 2 lemon cupcakes were left at home.
-- Given: Each box contains 3 lemon cupcakes.
-- Prove: The number of boxes with 3 lemon cupcakes each given away is 17.

theorem boxes_given_away (baked_cupcakes left_at_home cupcakes_per_box : ℕ) 
  (h_baked : baked_cupcakes = 53) 
  (h_left : left_at_home = 2) 
  (h_per_box : cupcakes_per_box = 3) : 
  (baked_cupcakes - left_at_home) / cupcakes_per_box = 17 :=
by 
  rw [h_baked, h_left, h_per_box]
  norm_num

end boxes_given_away_l418_418837


namespace range_of_f_l418_418983

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) - cos x + 1

theorem range_of_f : 
  set.range (λ x, f x) = set.Icc (-1/2 : ℝ) (3/2 + real.sqrt 3) :=
sorry

end range_of_f_l418_418983


namespace richard_twice_as_old_l418_418599

noncomputable def future_years_needed (richard_age david_age scott_age : ℕ) :=
  ∃ (x : ℕ), (richard_age + x = 2 * (scott_age + x))

theorem richard_twice_as_old (david_age : ℕ) (richard_age : ℕ := david_age + 6) (scott_age : ℕ := david_age - 8) :
  david_age = 14 →
  ∃ (x : ℕ), (richard_age + x = 2 * (scott_age + x)) ∧ x = 8 :=
by
  intro h_david
  rw [<-h_david]
  sorry

end richard_twice_as_old_l418_418599


namespace polynomial_real_root_inequality_l418_418051

theorem polynomial_real_root_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a * x^3 + 2 * x^2 - b * x + 1 = 0) → (a^2 + b^2 ≥ 8) :=
sorry

end polynomial_real_root_inequality_l418_418051


namespace fraction_know_both_l418_418408

variable (U : Type) -- Universal set representing all students
variable (A B : Set U) -- Sets of students knowing Greek and Latin

-- Known conditions
variable (h1 : (A ∪ B).card = U.card) -- All students know at least one of the languages
variable (h2 : A.card = 85 / 100 * U.card) -- 85% of all students know Greek
variable (h3 : B.card = 75 / 100 * U.card) -- 75% of all students know Latin

-- Statement to be proven
theorem fraction_know_both : 
  (A ∩ B).card = 60 / 100 * U.card :=
by
  sorry

end fraction_know_both_l418_418408


namespace smallest_a_l418_418295

theorem smallest_a (a : ℕ) (h_a : a > 8) : (∀ x : ℤ, ¬ Prime (x^4 + a^2)) ↔ a = 9 :=
by
  sorry

end smallest_a_l418_418295


namespace number_of_lattice_points_l418_418384

def boundedBy (x y : ℝ) : Prop := y = 2 * abs x ∨ y = -x^2 + 4

def isLatticePoint (x y : ℤ) : Prop := ∃ r s : ℝ, x = ⌊r⌋ ∧ y = ⌊s⌋ ∧ boundedBy r s

theorem number_of_lattice_points : 
  (∃ f : Fin 15, ∀ i : Fin 15, isLatticePoint f.val (i : ℤ)) :=
sorry

end number_of_lattice_points_l418_418384


namespace tunnel_length_l418_418193

-- Definitions as per the conditions
def train_length : ℚ := 2  -- 2 miles
def train_speed : ℚ := 40  -- 40 miles per hour

def speed_in_miles_per_minute (speed_mph : ℚ) : ℚ :=
  speed_mph / 60  -- Convert speed from miles per hour to miles per minute

def time_travelled_in_minutes : ℚ := 5  -- 5 minutes

-- Theorem statement to prove the length of the tunnel
theorem tunnel_length (h1 : train_length = 2) (h2 : train_speed = 40) :
  (speed_in_miles_per_minute train_speed * time_travelled_in_minutes) - train_length = 4 / 3 :=
by
  sorry  -- Proof not included

end tunnel_length_l418_418193


namespace part1_part2_l418_418343

variable (a b c : ℝ)
variable (AB_eq_c BC_eq_a CA_eq_b : Prop)
variable (AD BE CF : Type)
variable (DE_eq_DF : Prop)
variable (ABC_is_triangle : Triangle ABC)

theorem part1 (h1 : AB_eq_c) (h2 : BC_eq_a) (h3 : CA_eq_b)
              (h4 : IsAngleBisector AD)
              (h5 : IsAngleBisector BE)
              (h6 : IsAngleBisector CF)
              (h7 : DE_eq_DF) :
  a / (b + c) = b / (c + a) + c / (a + b) := sorry

theorem part2 (h1 : AB_eq_c) (h2 : BC_eq_a) (h3 : CA_eq_b)
              (h4 : IsAngleBisector AD)
              (h5 : IsAngleBisector BE)
              (h6 : IsAngleBisector CF)
              (h7 : DE_eq_DF)
              (h8 : a / (b + c) = b / (c + a) + c / (a + b)) :
  ∠BAC > 90 := sorry

end part1_part2_l418_418343


namespace cells_surpass_10_pow_10_in_46_hours_l418_418555

noncomputable def cells_exceed_threshold_hours : ℕ := 46

theorem cells_surpass_10_pow_10_in_46_hours : 
  ∀ (n : ℕ), (100 * ((3 / 2 : ℝ) ^ n) > 10 ^ 10) ↔ n ≥ cells_exceed_threshold_hours := 
by
  sorry

end cells_surpass_10_pow_10_in_46_hours_l418_418555


namespace min_cylinder_surface_area_l418_418620

noncomputable def h := Real.sqrt (5^2 - 4^2)
noncomputable def V_cone := (1 / 3) * Real.pi * 4^2 * h
noncomputable def V_cylinder (r h': ℝ) := Real.pi * r^2 * h'
noncomputable def h' (r: ℝ) := 16 / r^2
noncomputable def S (r: ℝ) := 2 * Real.pi * r^2 + (32 * Real.pi) / r

theorem min_cylinder_surface_area : 
  ∃ r, r = 2 ∧ ∀ r', r' ≠ 2 → S r' > S 2 := sorry

end min_cylinder_surface_area_l418_418620


namespace simplify_fraction_l418_418088

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l418_418088


namespace carlos_meeting_percentage_l418_418266

-- Definitions for the given conditions
def work_day_minutes : ℕ := 10 * 60
def first_meeting_minutes : ℕ := 80
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def break_minutes : ℕ := 15
def total_meeting_and_break_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + break_minutes

-- Statement to prove
theorem carlos_meeting_percentage : 
  (total_meeting_and_break_minutes * 100 / work_day_minutes) = 56 := 
by
  sorry

end carlos_meeting_percentage_l418_418266


namespace total_time_in_timeouts_is_195_l418_418468

def running_timeouts : ℕ := 5
def throwing_food_timeouts : ℕ := 2 + 3 * running_timeouts
def swearing_timeouts : ℕ := Int.to_nat (Int.sqrt (3 * throwing_food_timeouts))
def talking_loudly_timeouts : ℕ := 2 * running_timeouts
def timeout_duration : ℕ := 5

def total_timeouts_in_minutes : ℕ := 
  running_timeouts * timeout_duration + 
  throwing_food_timeouts * timeout_duration + 
  swearing_timeouts * timeout_duration + 
  talking_loudly_timeouts * timeout_duration

theorem total_time_in_timeouts_is_195 : total_timeouts_in_minutes = 195 := by
  sorry

end total_time_in_timeouts_is_195_l418_418468


namespace surface_area_sphere_dihedral_l418_418111

open Real

theorem surface_area_sphere_dihedral (R a : ℝ) (hR : 0 < R) (haR : 0 < a ∧ a < R) (α : ℝ) :
  2 * R^2 * arccos ((R * cos α) / sqrt (R^2 - a^2 * sin α^2)) 
  - 2 * R * a * sin α * arccos ((a * cos α) / sqrt (R^2 - a^2 * sin α^2)) = sorry :=
sorry

end surface_area_sphere_dihedral_l418_418111


namespace f_odd_l418_418362

-- Define the function f as given in the problem
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x * (1 - x) else x * (1 + x)

-- Prove the function is odd − f(−x) = −f(x)
theorem f_odd (x : ℝ) : f (-x) = -f x := by
  sorry

-- State the monotonically increasing interval
def monotonically_increasing_interval : Set ℝ :=
  Set.Icc (-(1/2)) (1/2)

end f_odd_l418_418362


namespace polynomial_characterization_l418_418672

theorem polynomial_characterization (P : ℝ → ℝ) :
  (∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) →
  ∃ (α β : ℝ), ∀ x : ℝ, P x = α * x^4 + β * x^2 :=
by
  sorry

end polynomial_characterization_l418_418672


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418937

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418937


namespace problem_l418_418715

noncomputable def a_n (x : ℕ) (n : ℕ) := 
  let a : list ℕ := (list.range (n+1)).map (λ k, binomial n k * (x-1)^(n-k)) -- Define a list of coefficients a_0, a_1, ..., a_n
  a

def S_n (n : ℕ) : ℕ := 
  ((list.range (n+1)).tail.map (λ k, binomial n k * (1-1)^(n-k))).sum

theorem problem (n : ℕ) (h : n ≥ 4) :
  S_n n > (n-2) * 2^n + 2 * n^2 :=
by sorry

end problem_l418_418715


namespace no_18_consecutive_good_numbers_l418_418866

def is_good (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, (p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ n = p1 * p2)

theorem no_18_consecutive_good_numbers :
  ¬ ∃ (a : ℕ), ∀ i : ℕ, i < 18 → is_good (a + i) :=
sorry

end no_18_consecutive_good_numbers_l418_418866


namespace albert_investment_l418_418239

open Real

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem albert_investment :
  ∃ P : ℝ, compound_interest P 0.05 1 2 = 8820 ∧ P = 8000 :=
by
  use 8000
  split
  · sorry
  · rfl

end albert_investment_l418_418239


namespace tunnel_length_l418_418191

-- Define relevant constants
def train_length : ℝ := 2
def exit_time_minutes : ℝ := 5
def train_speed_mph : ℝ := 40
def miles_per_hour_to_miles_per_minute (mph : ℝ) := mph / 60
def travel_distance (time_minutes : ℝ) (speed_mph : ℝ) := time_minutes * miles_per_hour_to_miles_per_minute speed_mph

-- The main theorem we want to prove
theorem tunnel_length : travel_distance exit_time_minutes train_speed_mph - train_length = 4 / 3 := sorry

end tunnel_length_l418_418191


namespace range_of_f_l418_418293

noncomputable def f (x : ℝ) : ℝ := if x ≠ 2 then (3 * (x - 2) * (x + 4) / (x - 2)) else 0

theorem range_of_f :
  (set.range (λ x, if x ≠ 2 then 3 * (x + 4) else 0)) = set.Ioo (-∞) 18 ∪ set.Ioo 18 ∞ :=
by
  sorry

end range_of_f_l418_418293


namespace sequence_nat_nums_exists_l418_418307

theorem sequence_nat_nums_exists (n : ℕ) : { k : ℕ | ∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i, 1 ≤ i → i ≤ n → x i)} = { k | 1 ≤ k ∧ k ≤ n } :=
sorry

end sequence_nat_nums_exists_l418_418307


namespace total_distance_A_travels_l418_418174

/-- 
Given that A and B are initially 90 km apart. A moves at an initial speed of 10 km/h, 
and B at an initial speed of 5 km/h. Each hour, both A and B double their speeds. 
We want to prove that the total distance A travels until he meets B is 60 km.
-/
theorem total_distance_A_travels
  (initial_distance : ℝ) 
  (initial_speed_A initial_speed_B : ℝ) 
  (double_speed : ℝ → ℝ)
  (t : ℝ → ℝ → ℝ)
  (meet_distance : ℝ → ℝ → ℝ → ℝ) : 
  initial_distance = 90 → 
  initial_speed_A = 10 → 
  initial_speed_B = 5 → 
  (∀ s, double_speed s = s * 2) → 
  meet_distance initial_distance initial_speed_A initial_speed_B = 60 := 
begin
  sorry
end

end total_distance_A_travels_l418_418174


namespace no_square_divisible_by_six_in_range_l418_418283

theorem no_square_divisible_by_six_in_range : ¬ ∃ y : ℕ, (∃ k : ℕ, y = k * k) ∧ (6 ∣ y) ∧ (50 ≤ y ∧ y ≤ 120) :=
by
  sorry

end no_square_divisible_by_six_in_range_l418_418283


namespace trigonometric_inequality_l418_418894

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
    cos x < 1 - (x ^ 2) / 2 + (x ^ 4) / 16 :=
sorry

end trigonometric_inequality_l418_418894


namespace nine_people_nine_times_work_three_days_l418_418391

theorem nine_people_nine_times_work_three_days 
  (h : ∀ (people work days : ℕ), people = 3 → work = 3 → days = 3 → (people * days) / work = 3) : 
  ∀ (people work days : ℕ), people = 9 → work = 9 → (people * days) / work = 3 :=
by {
  intros people work days hp hw,
  have h1 : (3 * 3) / 3 = 3, from h 3 3 3 rfl rfl rfl,
  have h2 : people / 3 = 3, from nat.div_eq_of_eq_mul_left (by norm_num) (by rw [hp, nat.mul_comm, ←one_mul 3]; exact rfl),
  have h3 : work / 3 = 3, from nat.div_eq_of_eq_mul_left (by norm_num) (by rw [hw, nat.mul_comm, ←one_mul 3]; exact rfl),
  have h4 : (people * 3) / work = 3, by rw [←hp, nat.mul_comm people, nat.mul_div_cancel_left _ (by norm_num)],
  exact nat.div_eq_of_eq_mul_left (by norm_num) h4,
  sorry
}

end nine_people_nine_times_work_three_days_l418_418391


namespace john_can_fix_l418_418018

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end john_can_fix_l418_418018


namespace b_share_220_l418_418179

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B + A + C = 770) : B = 220 :=
by
  sorry

end b_share_220_l418_418179


namespace correct_scientific_notation_representation_l418_418627

-- Defining the given number of visitors in millions
def visitors_in_millions : Float := 8.0327
-- Converting this number to an integer and expressing in scientific notation
def rounded_scientific_notation (num : Float) : String :=
  if num == 8.0327 then "8.0 × 10^6" else "incorrect"

-- The mathematical proof statement
theorem correct_scientific_notation_representation :
  rounded_scientific_notation visitors_in_millions = "8.0 × 10^6" :=
by
  sorry

end correct_scientific_notation_representation_l418_418627


namespace number_of_songs_performed_l418_418322

def total_songs_performed (anna bea cili dora : ℕ) : ℕ :=
  sorry

theorem number_of_songs_performed (anna_sang : 8) (dora_sang : 5) (bea_sang cili_sang : ℕ) (bea_greater_dora : bea_sang > dora_sang) (cili_greater_dora : cili_sang > dora_sang) (bea_le_anna : bea_sang ≤ anna_sang) (cili_le_anna : cili_sang ≤ anna_sang) (songs_divisible_by_three : (anna_sang + bea_sang + cili_sang + dora_sang) % 3 = 0) : total_songs_performed 8 bea_sang cili_sang 5 = 9 :=
  sorry

end number_of_songs_performed_l418_418322


namespace value_of_x_plus_2y_l418_418752

theorem value_of_x_plus_2y 
  (x y : ℝ) 
  (h : (x + 5)^2 = -(|y - 2|)) : 
  x + 2 * y = -1 :=
sorry

end value_of_x_plus_2y_l418_418752


namespace range_f_triangle_area_proof_l418_418369

noncomputable def f (x : Real) : Real := 2 * Real.sqrt 3 * (Real.sin x) ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

theorem range_f : ∀ x ∈ Set.Icc (Real.pi / 3) (11 * Real.pi / 24), f x ∈ Set.Icc (Real.sqrt 3) 2 :=
sorry

noncomputable def triangle_area (a b r : Real) := (1 / 2) * a * b * (a / (2 * r) * √ 6 / 3)

theorem triangle_area_proof : triangle_area (Real.sqrt 3) 2 (3 * Real.sqrt 2 / 4) = √ 2 :=
sorry

end range_f_triangle_area_proof_l418_418369


namespace simplify_fraction_l418_418068

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l418_418068


namespace rate_percent_simple_interest_l418_418587

theorem rate_percent_simple_interest (SI P T R : ℝ) (h₁ : SI = 500) (h₂ : P = 2000) (h₃ : T = 2)
  (h₄ : SI = (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for the proof
  sorry

end rate_percent_simple_interest_l418_418587


namespace metal_waste_l418_418205

theorem metal_waste (s : ℝ) (h : s > 0) : 
  let area_square := s^2,
      radius := s / 2,
      area_circle := π * radius^2,
      area_rectangle := s * (s / 2),
      waste1 := area_square - area_circle,
      waste2 := area_circle - area_rectangle,
      total_waste := waste1 + waste2
  in total_waste = s^2 / 2 :=
by
  sorry

end metal_waste_l418_418205


namespace baseball_singles_percentage_l418_418666

theorem baseball_singles_percentage :
  let total_hits := 50
  let home_runs := 2
  let triples := 3
  let doubles := 8
  let non_single_hits := home_runs + triples + doubles
  let singles := total_hits - non_single_hits
  let singles_percentage := (singles / total_hits) * 100
  singles = 37 ∧ singles_percentage = 74 :=
by
  sorry

end baseball_singles_percentage_l418_418666


namespace prime_square_remainder_l418_418252

theorem prime_square_remainder (p : ℕ) (hp : Nat.Prime p) (h5 : p > 5) : 
  ∃! r : ℕ, r < 180 ∧ (p^2 ≡ r [MOD 180]) := 
by
  sorry

end prime_square_remainder_l418_418252


namespace sequence_nat_nums_exists_l418_418306

theorem sequence_nat_nums_exists (n : ℕ) : { k : ℕ | ∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i, 1 ≤ i → i ≤ n → x i)} = { k | 1 ≤ k ∧ k ≤ n } :=
sorry

end sequence_nat_nums_exists_l418_418306


namespace simplify_expression_l418_418849

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

noncomputable def x : ℝ := (b / c) * (c / b)
noncomputable def y : ℝ := (a / c) * (c / a)
noncomputable def z : ℝ := (a / b) * (b / a)

theorem simplify_expression : x^2 + y^2 + z^2 + x^2 * y^2 * z^2 = 4 := 
by {
  sorry
}

end simplify_expression_l418_418849


namespace find_explicit_formula_sum_of_sequence_terms_l418_418341

variable (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (n : ℕ)
variable (n_pos : n > 0) (n_nat : n ∈ ℕ) -- n is a positive natural number

-- Definitions
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x
def derivative (x : ℝ) : ℝ := 2 * a * x + b

-- Problem Conditions
axiom passes_through_point : quadratic_function a b (-4 * n) = 0
axiom derivative_at_zero : derivative a b 0 = 2 * n

-- Questions
def explicit_formula : Prop :=
  quadratic_function a b = (0.5 : ℝ)*x^2 + 2 * (n : ℝ) * x

def sequence_sum : Prop :=
  ∑ k in finset.range n, (derivative a b (-k) * 2^k) = (n - 1) * 2^(n + 1) + 2

-- Statements to prove
theorem find_explicit_formula :
  explicit_formula n a b :=
sorry

theorem sum_of_sequence_terms :
  sequence_sum n a b :=
sorry

end find_explicit_formula_sum_of_sequence_terms_l418_418341


namespace ball_placement_l418_418484

theorem ball_placement : 
  ∃ (x_1 x_2 x_3 : ℕ), (x_1 + x_2 + x_3 = 10) ∧ (x_1 ≥ 1) ∧ (x_2 ≥ 2) ∧ (x_3 ≥ 3) ∧ 
    A: nat.choose (3 + 4 - 1) 4 = 15 := 
begin
  sorry
end

end ball_placement_l418_418484


namespace oak_grove_books_total_l418_418996

theorem oak_grove_books_total :
  let public_library_books := 1986
  let school_library_books := 5106
  public_library_books + school_library_books = 7092 :=
by
  have h₁ : public_library_books = 1986 := rfl
  have h₂ : school_library_books = 5106 := rfl
  calc
    public_library_books + school_library_books
        = 1986 + 5106 : by rw [h₁, h₂]
    ... = 7092 : by norm_num

end oak_grove_books_total_l418_418996


namespace problem_1_problem_2_l418_418861

open Real

def f (ω x : ℝ) : ℝ :=
  sin (ω * x) * cos (ω * x) - sqrt 3 * (cos (ω * x))^2 + (sqrt 3) / 2

def g (φ x : ℝ) : ℝ := cos (2 * x - φ)

theorem problem_1 (ω : ℝ) (H_dist : ∃ x, (f ω x)' * (f ω x)' + 4 = π^2 + 4) : ω = 1 / 2 :=
  sorry

theorem problem_2 (φ : ℝ) (Hφ : φ = π / 3) :
  intervals_of_decrease (g φ) [0, 2 * π] = [ (π / 6, 2 * π / 3), (7 * π / 6, 5 * π / 3) ] :=
  sorry

end problem_1_problem_2_l418_418861


namespace largestValidNumberIs84_l418_418902

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l418_418902


namespace geometric_sequence_sum_first_five_terms_l418_418793

noncomputable def geometric_sequence_S5
  (a_1 a_2 a_3 a_4 : ℝ) (q : ℝ)
  (h1 : a_1 * q = a_2)
  (h2 : a_1 * q^2 = a_3)
  (h3 : a_2 * q = a_4)
  (h4 : a_1 + a_1 * q^2 = 10)
  (h5 : a_1 * q + a_1 * q^3 = 30) : ℝ :=
  ∑ i in range 5, a_1 * q^i

theorem geometric_sequence_sum_first_five_terms
  (a_1 a_2 a_3 a_4 : ℝ) (q : ℝ)
  (h1 : a_1 * q = a_2)
  (h2 : a_1 * q^2 = a_3)
  (h3 : a_2 * q = a_4)
  (h4 : a_1 + a_1 * q^2 = 10)
  (h5 : a_1 * q + a_1 * q^3 = 30) :
  geometric_sequence_S5 a_1 a_2 a_3 a_4 q h1 h2 h3 h4 h5 = 121 :=
by
  sorry

end geometric_sequence_sum_first_five_terms_l418_418793


namespace pages_can_be_copied_l418_418826

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l418_418826


namespace aaron_ends_up_with_24_cards_l418_418249

def initial_cards_aaron : Nat := 5
def found_cards_aaron : Nat := 62
def lost_cards_aaron : Nat := 15
def given_cards_to_arthur : Nat := 28

def final_cards_aaron (initial: Nat) (found: Nat) (lost: Nat) (given: Nat) : Nat :=
  initial + found - lost - given

theorem aaron_ends_up_with_24_cards :
  final_cards_aaron initial_cards_aaron found_cards_aaron lost_cards_aaron given_cards_to_arthur = 24 := by
  sorry

end aaron_ends_up_with_24_cards_l418_418249


namespace yogurt_combinations_l418_418625

theorem yogurt_combinations : 
  let yogurt_flavors := 6
      toppings := 8 in
  (yogurt_flavors * (Nat.choose toppings 2)) = 168 :=
by
  let yogurt_flavors := 6
  let toppings := 8
  show yogurt_flavors * (Nat.choose toppings 2) = 168
  sorry

end yogurt_combinations_l418_418625


namespace problem_inequality_l418_418036

theorem problem_inequality (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h: (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  a / b + b / c + c / a = 4.5 :=
by
  sorry

end problem_inequality_l418_418036


namespace smaller_square_side_length_l418_418516

variables {Point : Type} [euclidean_geometry Point]

-- Defining points and the square ABCD with side length 1.
def square (A B C D : Point) (len : ℝ) : Prop :=
  (dist A B = len) ∧ (dist B C = len) ∧ (dist C D = len) ∧ (dist D A = len)
  ∧ (dist A C = dist B D ∧ dist A C = len * Real.sqrt 2)

-- Defining the condition about the points E and F and the triangle AEF.
def right_isosceles_triangle (A E F : Point) : Prop :=
  ∠ A E F = 45 ∧ dist A E = dist A F

-- The side length of the smaller square can be derived.
theorem smaller_square_side_length (A B C D E F : Point)
  (h₁ : square A B C D 1) 
  (h₂ : E ∈ (Segment B C)) 
  (h₃ : F ∈ (Segment C D)) 
  (h₄ : right_isosceles_triangle A E F) :
  ∃ s : ℝ, s = (2 - Real.sqrt 2) / 2 := 
sorry

end smaller_square_side_length_l418_418516


namespace part1_part3_l418_418771

-- Part (1) Lean statement
theorem part1 (a : ℕ → ℝ) (h : ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n) :
  (∀ n, 2^((n : ℕ)) ∈ {a n}) := sorry

-- Part (2) Lean statement
noncomputable def part2 (a : ℕ → ℤ) (hn : ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n)
  (h1 : a 1 = 1) (h2 : a 2 = 3) (hk : ∃ (k : ℕ), a k = 2023) :
  ∃ (k : ℕ), k * (k + 1) / 2 ≤ 2023 := sorry

-- Part (3) Lean statement
theorem part3 (b : ℕ → ℤ) (k : ℕ) (h : 2 ≤ k) (hsum: (∑ i in range (2 * k), b i) = k)
  (hb : ∀ n, b (n + 2) - b (n + 1) > b (n + 1) - b n) :
  ∀ {c : ℕ → ℝ}, (∀ n, c n = 2^(b n)) → c k * c (k + 1) < 2 := sorry

end part1_part3_l418_418771


namespace jed_gives_2_cards_every_two_weeks_l418_418428

theorem jed_gives_2_cards_every_two_weeks
  (starting_cards : ℕ)
  (cards_per_week : ℕ)
  (cards_after_4_weeks : ℕ)
  (number_of_two_week_intervals : ℕ)
  (cards_given_away_each_two_weeks : ℕ):
  starting_cards = 20 →
  cards_per_week = 6 →
  cards_after_4_weeks = 40 →
  number_of_two_week_intervals = 2 →
  (starting_cards + 4 * cards_per_week - number_of_two_week_intervals * cards_given_away_each_two_weeks = cards_after_4_weeks) →
  cards_given_away_each_two_weeks = 2 := 
by
  intros h_start h_week h_4weeks h_intervals h_eq
  sorry

end jed_gives_2_cards_every_two_weeks_l418_418428


namespace students_answered_all_three_correctly_l418_418883

theorem students_answered_all_three_correctly
    (total_students : ℕ)
    (answered_1_correctly : ℕ)
    (answered_2_correctly : ℕ)
    (answered_3_correctly : ℕ)
    (students_not_took_test : ℕ)
    (h1 : total_students = 30)
    (h2 : answered_1_correctly = 25)
    (h3 : answered_2_correctly = 22)
    (h4 : answered_3_correctly = 18)
    (h5 : students_not_took_test = 5) :
    ∃ (students_answered_all_three : ℕ),
    students_answered_all_three = 18 :=
by
  use 18
  sorry

end students_answered_all_three_correctly_l418_418883


namespace polynomial_degree_is_one_l418_418703

noncomputable def deg (P : Polynomial ℝ) : ℕ := 
sorry  -- Assume we have a function to get the degree of polynomial

theorem polynomial_degree_is_one {P : Polynomial ℝ} 
  (hP_real_coeff : ∀ n : ℕ, P.coeff n ∈ ℝ)
  (a : ℕ → ℕ) 
  (h_seq : Function.Injective a
           ∧ (∀ n : ℕ, P.eval (a n) = if n = 0 then 0 else a (n - 1))) :
  deg P = 1 :=
sorry  -- Proof is omitted as per instructions.

end polynomial_degree_is_one_l418_418703


namespace find_p_l418_418223

noncomputable def parabola_focus (p : ℝ) (hp : 0 < p) : ℝ × ℝ :=
  (0, p / 2)

noncomputable def line_through_focus (p : ℝ) : (ℝ × ℝ) → Prop :=
  fun (x y) => y = x + p / 2

def parabola_eq (p : ℝ) : (ℝ × ℝ) → Prop :=
  fun (x y) => x^2 = 2 * p * y

axiom area_ABCD (p : ℝ) : ℝ :=
  12 * real.sqrt 2

theorem find_p (p : ℝ) (hp : 0 < p) :
  let A := parabola_focus p hp,
      B := parabola_focus p hp
  in area_ABCD p = 12 * real.sqrt 2 → p = 2 :=
begin
  sorry
end

end find_p_l418_418223


namespace largest_prime_factor_101_l418_418272

-- Define the sequence and its properties
def Sequence : Type := { l: List ℕ // ∀ a ∈ l, 1000 ≤ a ∧ a < 10000 }

-- Define the rotation property of the sequence
def rotation_property (seq: Sequence) : Prop :=
  ∀ (n: ℕ), n < seq.val.length → 
  let a := seq.val.nth_le n (by linarith) in
  let b := seq.val.nth_le ((n + 1) % seq.val.length) (by linarith) in
  let (d3, d2, d1, d0) := (a / 1000, (a / 100) % 10, (a / 10) % 10, a % 10) in
  let (c3, c2, c1, c0) := (b / 1000, (b / 100) % 10, (b / 10) % 10, b % 10) in
  c3 = d2 ∧ c2 = d1 ∧ c1 = d0

-- Define the sum of the sequence
def seq_sum (seq: Sequence) : ℕ :=
  seq.val.foldl (+) 0

-- The main theorem to be proven
theorem largest_prime_factor_101 (seq: Sequence) (h: rotation_property seq) : 
  101 ∣ seq_sum seq :=
sorry

end largest_prime_factor_101_l418_418272


namespace zero_of_f_in_interval_l418_418335

def f (x : ℝ) : ℝ := 3^x + 3*x - 8

axiom approx_3_pow_1_25 : 3^1.25 ≈ 3.9
axiom approx_3_pow_1_5 : 3^1.5 ≈ 5.2

theorem zero_of_f_in_interval : ∃ x : ℝ, (1.25 < x ∧ x < 1.5) ∧ f x = 0 :=
by
  sorry

end zero_of_f_in_interval_l418_418335


namespace find_number_l418_418201

-- Define the conditions
def number_times_x_eq_165 (number x : ℕ) : Prop :=
  number * x = 165

def x_eq_11 (x : ℕ) : Prop :=
  x = 11

-- The proof problem statement
theorem find_number (number x : ℕ) (h1 : number_times_x_eq_165 number x) (h2 : x_eq_11 x) : number = 15 :=
by
  sorry

end find_number_l418_418201


namespace sequence_exists_l418_418317

variable (n : ℕ)

theorem sequence_exists (k : ℕ) (hkn : k ∈ Set.range (λ x : ℕ, x + 1) n) :
  ∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i → i ≤ n → x (i+1) > x i) ∧ (∀ i, x i ∈ ℕ) :=
sorry

end sequence_exists_l418_418317


namespace average_value_of_set_l418_418460

theorem average_value_of_set 
  (T : Finset ℕ)
  (m : ℕ)
  (b : Fin (m+1) → ℕ)
  (hm : m > 1)
  (sum_without_bm : (∑ i in range m, b i) = 45 * m)
  (sum_without_b1_bm : (∑ i in range (m-1), b i.succ) = 50 * (m - 1))
  (sum_without_b1 : (∑ i in range m, b i.succ) = 55 * m)
  (bm_b1 : b ⟨m, lt_add_one m⟩ - b 0 = 50) :
  ((∑ i in Finset.range (m+1), b i) / (m+1) : ℝ) = 50 := sorry

end average_value_of_set_l418_418460


namespace probability_of_set_l418_418658

def is_multiple_of_63 (x : ℕ) : Prop := 63 ∣ x

def valid_pairs (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (λ ⟨a, b⟩ => a ≠ b ∧ is_multiple_of_63 (a * b))

def probability_multiple_of_63 (s : Finset ℕ) : ℚ :=
  (valid_pairs s).card / (s.card.choose 2)

theorem probability_of_set :
  probability_multiple_of_63 ({6, 14, 18, 27, 35, 42, 54} : Finset ℕ) =
    10 / 21 :=
by
  sorry

end probability_of_set_l418_418658


namespace pages_copied_l418_418814

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l418_418814


namespace correct_charcoal_calculation_l418_418126

def additional_charcoal_needed 
  (ratio_KNO3_S_C : ℕ × ℕ × ℕ) 
  (current_charcoal : ℕ) 
  (total_black_powder : ℕ) : ℕ :=
by
  let ⟨rK, rS, rC⟩ := ratio_KNO3_S_C,
  -- Ensure the ratio corresponds to 15:2:3
  have hRatio : rK = 15 ∧ rS = 2 ∧ rC = 3 := by exact ⟨rfl, rfl, rfl⟩,
  -- Ratio sum
  let tot_ratio := rK + rS + rC,
  -- Calculating required charcoal
  let required_charcoal := (total_black_powder * rC) / tot_ratio,
  -- Calculating additional charcoal needed
  let additional_needed := required_charcoal - current_charcoal,
  -- Verify that the additional_needed is correct
  have hCorrect : additional_needed = 100 := sorry,
  exact additional_needed

theorem correct_charcoal_calculation :
  additional_charcoal_needed (15, 2, 3) 50 1000 = 100 :=
by
  -- This uses the defined function and conditions to confirm the additional charcoal needed in proof problem context
  exact rfl

end correct_charcoal_calculation_l418_418126


namespace min_cars_needed_l418_418585

theorem min_cars_needed (h1 : ∀ d ∈ Finset.range 7, ∃ s : Finset ℕ, s.card = 2 ∧ (∃ n : ℕ, 7 * (n - 10) ≥ 2 * n)) : 
  ∃ n, n ≥ 14 :=
by
  sorry

end min_cars_needed_l418_418585


namespace greatest_value_of_NPMK_l418_418581

def is_digit (n : ℕ) : Prop := n < 10

theorem greatest_value_of_NPMK : 
  ∃ M K N P : ℕ, is_digit M ∧ is_digit K ∧ 
  M = K + 1 ∧ M = 9 ∧ K = 8 ∧ 
  1000 * N + 100 * P + 10 * M + K = 8010 ∧ 
  (100 * M + 10 * M + K) * M = 8010 := by
  sorry

end greatest_value_of_NPMK_l418_418581


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418921

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418921


namespace part1_part2_part3_l418_418734

noncomputable def f (x m : ℝ) : ℝ :=
  -x^2 + m*x - m

-- Part (1)
theorem part1 (m : ℝ) : (∀ x, f x m ≤ 0) → (m = 0 ∨ m = 4) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x, -1 ≤ x ∧ x ≤ 0 → f x m ≤ f (-1) m) → (m ≤ -2) :=
sorry

-- Part (3)
theorem part3 : ∃ (m : ℝ), (∀ x, 2 ≤ x ∧ x ≤ 3 → (2 ≤ f x m ∧ f x m ≤ 3)) → m = 6 :=
sorry

end part1_part2_part3_l418_418734


namespace Xiaoya_catchup_first_time_Xiaoya_catchup_second_time_l418_418502

theorem Xiaoya_catchup_first_time (track_length : ℕ) (Xiaoya_speed Xiaopang_speed : ℕ) (time_first_catchup : ℕ) 
    (distance_Xiaoya distance_Xiaopang : ℕ) 
    (h1 : track_length = 300)
    (h2 : Xiaoya_speed = 6)
    (h3 : Xiaopang_speed = 4)
    (h4 : time_first_catchup = 150)
    (h5 : distance_Xiaoya = Xiaoya_speed * time_first_catchup)
    (h6 : distance_Xiaopang = Xiaopang_speed * time_first_catchup) : 
    distance_Xiaoya = 900 ∧ distance_Xiaopang = 600 := 
begin
  sorry
end

theorem Xiaoya_catchup_second_time (track_length : ℕ) (laps_Xiaoya laps_Xiaopang : ℕ) 
    (h1 : track_length = 300)
    (h2 : laps_Xiaoya = 6)
    (h3 : laps_Xiaopang = 4) : 
    ((laps_Xiaopang * track_length + track_length) / laps_Xiaopang) = 300 ∧ ((laps_Xiaoya * track_length + track_length) / laps_Xiaoya) = 300 :=
begin
  sorry
end

end Xiaoya_catchup_first_time_Xiaoya_catchup_second_time_l418_418502


namespace find_smallest_n_l418_418753

def a (n : ℕ) : ℝ := 
  match n with
  | 0 => sin (Real.pi / 18) ^ 2
  | (n+1) => 4 * a n * (1 - a n)

theorem find_smallest_n (a : ℕ → ℝ) :
  (a 0 = sin (Real.pi / 18) ^ 2) ∧
  (∀ n, a (n+1) = 4 * a n * (1 - a n)) →
  (∃ n > 0, a n = a 1) ∧ 
  (∀ k, k > 0 → k < n → a k ≠ a 1) :=
by
  sorry

end find_smallest_n_l418_418753


namespace magnitude_difference_l418_418695

-- Definitions and given conditions
variables {a b : EuclideanSpace ℝ (Fin 3)}
def vector_a (a : EuclideanSpace ℝ (Fin 3)) := ∥a∥ = 2 * √10
def vector_b (b : EuclideanSpace ℝ (Fin 3)) := ∥b∥ = √10
def angle (a b : EuclideanSpace ℝ (Fin 3)) := real.arccos (inner a b / (∥a∥ * ∥b∥)) = π / 3

-- The theorem to prove
theorem magnitude_difference (a b : EuclideanSpace ℝ (Fin 3))
  (ha : vector_a a) (hb : vector_b b) (hab : angle a b) :
  ∥a - 2 • b∥ = 2 * √10 :=
sorry

end magnitude_difference_l418_418695


namespace razorback_tshirts_game_profit_correct_l418_418109

noncomputable def razorback_tshirts_game_profits (total_tshirts : ℕ) (arkansas_tshirts : ℕ) (texas_tech_tshirts : ℕ) (profit_per_tshirt : ℕ) 
(h1 : total_tshirts = 186) 
(h2 : arkansas_tshirts = 172) 
(h3 : texas_tech_tshirts = total_tshirts - arkansas_tshirts) 
(h4 : texas_tech_tshirts ≤ 50) 
(h5 : profit_per_tshirt = 78) : ℕ :=
let total_profit := texas_tech_tshirts * profit_per_tshirt in
total_profit

theorem razorback_tshirts_game_profit_correct : razorback_tshirts_game_profits 186 172 (186 - 172) 78 
  (by rfl) (by rfl) (by rfl) (by norm_num) (by rfl) = 1092 :=
by
  sorry

end razorback_tshirts_game_profit_correct_l418_418109


namespace range_of_p_l418_418742

def A (x : ℝ) : Prop := -2 < x ∧ x < 5
def B (p : ℝ) (x : ℝ) : Prop := p + 1 < x ∧ x < 2 * p - 1

theorem range_of_p (p : ℝ) :
  (∀ x, A x ∨ B p x → A x) ↔ p ≤ 3 :=
by
  sorry

end range_of_p_l418_418742


namespace mixed_number_division_l418_418264

theorem mixed_number_division :
  (5 + 1 / 2) / (2 / 11) = 121 / 4 :=
by sorry

end mixed_number_division_l418_418264


namespace binomial_coefficient_congruence_binomial_coefficient_congruence_eq_l418_418851

theorem binomial_coefficient_congruence (p : ℕ) (n : ℕ) (q : ℕ) [hp : Fact (Nat.Prime p)] (hn : 0 < n) (hq : 0 < q) :
  n ≠ q * (p - 1) → binomial n (p - 1) % p = 0 :=
sorry

theorem binomial_coefficient_congruence_eq (p : ℕ) (n : ℕ) (q : ℕ) [hp : Fact (Nat.Prime p)] (hn : 0 < n) (hq : 0 < q) :
  n = q * (p - 1) → binomial n (p - 1) % p = 1 :=
sorry

end binomial_coefficient_congruence_binomial_coefficient_congruence_eq_l418_418851


namespace interval_of_increase_increasing_on_positive_reals_minimum_value_a_l418_418593

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 + (a + 1) * x + 1

theorem interval_of_increase (a : ℝ) (h : a = -1) : 
  ∃ I : Set ℝ, IsInterval I ∧ ∀ x ∈ I, ∃ ε > 0, ∀ h, 0 < |h| ∧ |h| < ε → (f a (x + h) - f a x) / h > 0 :=
sorry

theorem increasing_on_positive_reals {a : ℝ} (h : ∀ x > 0, deriv (f a) x ≥ 0) : 
  a ≥ 0 :=
sorry

theorem minimum_value_a {a : ℝ} (h1 : a > 0) 
                         (h2 : ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → abs (f a x1 - f a x2) > 2 * abs (x1 - x2)) : 
  a ≥ 1 :=
sorry

end interval_of_increase_increasing_on_positive_reals_minimum_value_a_l418_418593


namespace chord_ratio_proof_l418_418145

-- Define the geometric configuration
variables {P Q R S T : Type} [TopologicalSpace P] [TopologicalSpace Q] [TopologicalSpace R] [TopologicalSpace S] [TopologicalSpace T]
variables (A B C D : P) (circ1 circ2 : set P) (chordAC chordAD : set P)

-- Topological circles intersecting at points A and B
axiom circles_intersect : circ1 ∩ circ2 = {A, B}

-- Chords AC and AD drawn through point A
axiom chords_through_A : A ∈ chordAC ∧ A ∈ chordAD

-- Chords AC and AD are tangent to circ1 and circ2 respectively
axiom chordAC_tangent_circ1 : chordAC ∈ (tangent circ1 A)
axiom chordAD_tangent_circ2 : chordAD ∈ (tangent circ2 A)

-- The theorem stating the required proof
theorem chord_ratio_proof :
  AC ≠ 0 ∧ AD ≠ 0 → 
  (AC^2 / AD^2) = (BC / BD) :=
by
  assume h : AC ≠ 0 ∧ AD ≠ 0,
  sorry -- Proof is not required

end chord_ratio_proof_l418_418145


namespace total_pies_l418_418600

theorem total_pies (P : ℕ) (h1 : 0.32 * P = 640) : P = 2000 :=
by
  sorry

end total_pies_l418_418600


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l418_418906

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l418_418906


namespace largest_two_digit_divisible_by_6_ending_in_4_l418_418928

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l418_418928


namespace range_of_a_l418_418727

theorem range_of_a (a : ℝ) : (1 < a) → 
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 
  (1 / (x1 + 2) = a * |x1| ∧ 1 / (x2 + 2) = a * |x2| ∧ 1 / (x3 + 2) = a * |x3|) :=
sorry

end range_of_a_l418_418727


namespace increase_by_ninety_percent_correct_l418_418164

-- Define initial number and percentage increase
def initial_number : ℝ := 75
def percentage_increase : ℝ := 90 / 100

-- Define the expected result
def expected_result : ℝ := 142.5

-- State the theorem
theorem increase_by_ninety_percent_correct :
  initial_number + (initial_number * percentage_increase) = expected_result :=
by
  -- Placeholder for actual proof
  sorry

end increase_by_ninety_percent_correct_l418_418164


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l418_418943

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l418_418943


namespace option_C_correct_l418_418713

variables {a c x1 x2 x3 y1 y2 y3 : ℝ}

def parabola (x : ℝ) : ℝ := -a / 6 * x^2 + a * x + c

-- Assume points lie on the parabola
axiom A_on_parabola : y1 = parabola x1
axiom B_on_parabola : y2 = parabola x2
axiom C_on_parabola : y3 = parabola x3

-- Given condition for y2
axiom y2_condition : y2 = 3 / 2 * a + c

theorem option_C_correct (h : y1 > y3 ∧ y3 > y2) : |x1 - x2| ≥ |x2 - x3| := sorry

end option_C_correct_l418_418713


namespace largestValidNumberIs84_l418_418904

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l418_418904


namespace chipmunk_acorns_l418_418301

theorem chipmunk_acorns :
  ∀ (x y : ℕ), (3 * x = 4 * y) → (y = x - 4) → (3 * x = 48) :=
by
  intros x y h1 h2
  sorry

end chipmunk_acorns_l418_418301


namespace truck_travel_time_l418_418128

def miles_to_feet (miles : ℝ) : ℝ := miles * 5280
def feet_per_second (miles_per_hour : ℝ) : ℝ := (miles_to_feet miles_per_hour) / 3600

theorem truck_travel_time 
    (truck_length tunnel_length speed_mph : ℝ)
    (h_truck_length : truck_length = 66)
    (h_tunnel_length : tunnel_length = 330)
    (h_speed_mph : speed_mph = 45) :
    let speed_fps := feet_per_second speed_mph in
    let total_distance := tunnel_length + truck_length in
    total_distance / speed_fps = 6 :=
by
  unfold miles_to_feet feet_per_second
  unfold_projs
  sorry

end truck_travel_time_l418_418128


namespace simplify_fraction_l418_418087

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l418_418087


namespace teachers_without_conditions_percentage_l418_418232

theorem teachers_without_conditions_percentage (
  total_teachers : ℕ,
  teachers_high_bp : ℕ,
  teachers_heart_trouble : ℕ,
  teachers_diabetes : ℕ,
  teachers_high_bp_heart_trouble : ℕ,
  teachers_heart_trouble_diabetes : ℕ,
  teachers_high_bp_diabetes : ℕ,
  teachers_all_three : ℕ
) : 
  total_teachers = 150 → 
  teachers_high_bp = 80 → 
  teachers_heart_trouble = 60 → 
  teachers_diabetes = 30 → 
  teachers_high_bp_heart_trouble = 20 → 
  teachers_heart_trouble_diabetes = 10 → 
  teachers_high_bp_diabetes = 15 → 
  teachers_all_three = 5 →
  (total_teachers - (teachers_high_bp - (teachers_high_bp_heart_trouble + teachers_high_bp_diabetes - teachers_all_three) + 
  teachers_heart_trouble - (teachers_high_bp_heart_trouble + teachers_heart_trouble_diabetes - teachers_all_three) + 
  teachers_diabetes - (teachers_high_bp_diabetes + teachers_heart_trouble_diabetes - teachers_all_three) + 
  (teachers_high_bp_heart_trouble - teachers_all_three) + 
  (teachers_heart_trouble_diabetes - teachers_all_three) + 
  (teachers_high_bp_diabetes - teachers_all_three) + 
  teachers_all_three)) / total_teachers.to_float * 100 = 13.33 :=
by
  intros
  sorry

end teachers_without_conditions_percentage_l418_418232


namespace range_of_a_l418_418720

-- Definition of an even function on ℝ that is decreasing on [0, +∞)
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def decreasing_on_nonneg_halfline (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ∈ [0,+∞) → f y ≤ f x 

-- Our main statement
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  is_even_function f →
  decreasing_on_nonneg_halfline f →
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f(x^3 - x^2 + a) + f(-x^3 + x^2 - a) ≥ 2 * f 1) →
  -23 / 27 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l418_418720


namespace amount_transferred_l418_418652

def original_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : original_balance - remaining_balance = 69 :=
by
  sorry

end amount_transferred_l418_418652


namespace smallest_region_area_l418_418288

-- Define the basic elements
def line (x : ℝ) : ℝ := x + 1
def circle_radius : ℝ := 3
def circle (x y : ℝ) : Prop := x^2 + y^2 = circle_radius^2

-- Define the function to find the area of the smaller region
def area_of_smaller_region : ℝ :=
  (9 / 2) * Real.arcsin(1 / 3) - 1.5

-- Define the proof problem
theorem smallest_region_area :
  ∃ A : ℝ, (A = area_of_smaller_region) ∧ (∀ x y : ℝ, circle x y ∧ y = line x → 
  A = (9 / 2) * Real.arcsin(1 / 3) - 1.5) :=
by
  exists (9 / 2 * Real.arcsin (1 / 3) - 1.5)
  split
  { rfl }
  { intros x y h
    unfold area_of_smaller_region
    cases h 
    sorry
  }

end smallest_region_area_l418_418288


namespace complex_numbers_equation_l418_418858

theorem complex_numbers_equation {a b : ℂ} (h : (a + b) / (a - b) - (a - b) / (a + b) = 0) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = 0 := 
by sorry

end complex_numbers_equation_l418_418858


namespace absolute_value_condition_necessary_non_sufficient_l418_418754

theorem absolute_value_condition_necessary_non_sufficient (x : ℝ) :
  (abs (x - 1) < 2 → x^2 < x) ∧ ¬ (x^2 < x → abs (x - 1) < 2) := sorry

end absolute_value_condition_necessary_non_sufficient_l418_418754


namespace largest_divisible_by_6_ending_in_4_l418_418917

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l418_418917


namespace smallest_term_divisible_by_two_million_l418_418860

-- Defining the first term and the second term of the geometric sequence
def first_term : ℚ := 5 / 8
def second_term : ℚ := 25

-- Define the target divisibility number
def target_divisibility : ℕ := 2000000

-- Function to calculate the nth term of the geometric sequence
def nth_term (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

-- The smallest n for which the nth term is divisible by 2,000,000
def smallest_n (a r : ℚ) (target : ℕ) : ℕ :=
  (inf_n : ℕ) where
  inf_n := sorry -- This will be solved in the proof

-- The theorem statement
theorem smallest_term_divisible_by_two_million :
  let r : ℚ := second_term / first_term
  ∃ (n : ℕ), nth_term first_term r n / 2 = 2000000 / 2 ∧ n = 7 :=
by {
  let r := second_term / first_term,
  use 7,
  sorry
}

end smallest_term_divisible_by_two_million_l418_418860


namespace factorial_sum_squares_l418_418514

-- Define the condition that 1! + 2! + ... + n! = m^2
def sum_factorials_eq_square (n m : ℕ) : Prop :=
  (Finset.range (n + 1)).sum (λ i, i!) = m^2

-- The statement we need to prove
theorem factorial_sum_squares :
  { (n, m) | sum_factorials_eq_square n m } = { (1, 1), (3, 3) } := 
sorry

end factorial_sum_squares_l418_418514


namespace allocation_methods_count_l418_418616

noncomputable def numAllocationMethods : ℕ :=
  let T := finset.range 8
  let teachers := {T0, T1, T2, T3, T4, T5, T6, T7}
  let A := T0
  let B := T1
  let C := T2
  let choose (k : ℕ) := (T.choose k).to_list
  let perm := list.permutations
  let validSelections := (list.filter (λ ts, ¬ (A ∈ ts ∧ B ∈ ts ∨ A ∈ ts ∧ C ∉ ts ∨ C ∈ ts ∧ A ∉ ts)) (choose 4)).bind perm
  validSelections.length

theorem allocation_methods_count : numAllocationMethods = 600 := by
  sorry

end allocation_methods_count_l418_418616


namespace constant_term_in_binomial_expansion_l418_418791

theorem constant_term_in_binomial_expansion : 
    let x : ℝ := sorry
    constant_term (expand (x - 1/x)^8) = 70 :=
sorry

end constant_term_in_binomial_expansion_l418_418791


namespace variance_scaled_l418_418552

-- Let V represent the variance of the set of data
def original_variance : ℝ := 3
def scale_factor : ℝ := 3

-- Prove that the new variance is 27 
theorem variance_scaled (V : ℝ) (s : ℝ) (hV : V = 3) (hs : s = 3) : s^2 * V = 27 := by
  sorry

end variance_scaled_l418_418552


namespace arcsin_neg_sqrt3_div_2_l418_418269

theorem arcsin_neg_sqrt3_div_2 : 
  Real.arcsin (- (Real.sqrt 3 / 2)) = - (Real.pi / 3) := 
by sorry

end arcsin_neg_sqrt3_div_2_l418_418269


namespace maximum_number_of_integers_with_sum_condition_l418_418136

theorem maximum_number_of_integers_with_sum_condition (n : ℕ) : 
    (∀ (a_i a_j a_k : ℕ), a_i ∈ S → a_j ∈ S → a_k ∈ S → (a_i + a_j + a_k) % 39 = 0) → 
    (∀ a ∈ S, a ≤ 2013) → 
    ∃ (S : Finset ℕ), S.card = 52 :=
by
  sorry

end maximum_number_of_integers_with_sum_condition_l418_418136


namespace sum_of_first_70_odd_integers_l418_418988

theorem sum_of_first_70_odd_integers : 
  let sum_even := 70 * (70 + 1)
  let sum_odd := 70 ^ 2
  let diff := sum_even - sum_odd
  diff = 70 → sum_odd = 4900 :=
by
  intros
  sorry

end sum_of_first_70_odd_integers_l418_418988


namespace precious_more_correct_l418_418875

theorem precious_more_correct (n : ℕ) (h₁ : n = 75)
    (h₂ : ∃ x, x = 0.20 * 75 ∧ x = 15)
    (h₃ : 12 < 75) :
    let correct_answers_Lyssa := n - 15,
        correct_answers_Precious := n - 12
    in correct_answers_Precious - correct_answers_Lyssa = 3 := by
  sorry

end precious_more_correct_l418_418875


namespace simplify_fraction_l418_418081

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l418_418081


namespace max_cos_sum_l418_418676

theorem max_cos_sum (x y : ℝ) (h : cos x - cos y = 1 / 4) : 
  ∃ M, M = 31 / 32 ∧ M = max (cos (x + y)) :=
sorry

end max_cos_sum_l418_418676


namespace contrapositive_equivalence_l418_418528

variable (Person : Type)
variable (Happy Have : Person → Prop)

theorem contrapositive_equivalence :
  (∀ (x : Person), Happy x → Have x) ↔ (∀ (x : Person), ¬Have x → ¬Happy x) :=
by
  sorry

end contrapositive_equivalence_l418_418528


namespace minimum_omega_l418_418732

theorem minimum_omega (f : ℝ → ℝ) (ω ϕ x0 : ℝ) (h_pos : ω > 0)
  (h_f : ∀ x : ℝ, f x = Real.sin (ω x + ϕ))
  (h_ineq : ∀ x : ℝ, f x0 ≤ f x ∧ f x ≤ f (x0 + 2016 * Real.pi)) :
  ω = 1 / 2016 :=
sorry

end minimum_omega_l418_418732


namespace original_price_per_kg_of_salt_l418_418615

variable {P X : ℝ}

theorem original_price_per_kg_of_salt (h1 : 400 / (0.8 * P) = X + 10)
    (h2 : 400 / P = X) : P = 10 :=
by
  sorry

end original_price_per_kg_of_salt_l418_418615


namespace value_of_4_inch_cube_is_approximately_711_l418_418607

-- Define the side lengths of the cubes
def side_length_3_inch_cube := 3
def side_length_4_inch_cube := 4

-- Define the volume of each cube
def volume_of_cube (side_length : ℕ) : ℕ := side_length ^ 3

-- Define the value of the 3-inch cube
def value_of_3_inch_cube := 300

-- Calculate the volume of the 3-inch and 4-inch cubes
def volume_3_inch_cube := volume_of_cube side_length_3_inch_cube
def volume_4_inch_cube := volume_of_cube side_length_4_inch_cube

-- Define the volume ratio
def volume_ratio := volume_4_inch_cube / volume_3_inch_cube

-- Define the expected value of the 4-inch cube
def expected_value_4_inch_cube := value_of_3_inch_cube * volume_ratio

-- Prove that the value of the 4-inch cube is approximately $711
theorem value_of_4_inch_cube_is_approximately_711
  (value_4_inch_cube : ℕ)
  (h : value_4_inch_cube = 711) :
  expected_value_4_inch_cube ≈ 711 := by
  sorry

end value_of_4_inch_cube_is_approximately_711_l418_418607


namespace domain_of_f_l418_418975

noncomputable def f (x : ℝ) : ℝ := real.sqrt (2^x - 1/4) + real.log (1 - x)

def domain_condition_1 (x : ℝ) : Prop := 2^x - 1/4 ≥ 0
def domain_condition_2 (x : ℝ) : Prop := 1 - x > 0

theorem domain_of_f :
  {x : ℝ | domain_condition_1 x ∧ domain_condition_2 x} = Icc (-2 : ℝ) 1 ∩ Iio 1 :=
by
  sorry

end domain_of_f_l418_418975


namespace curve_rectangular_coord_equation_reciprocal_sum_dist_l418_418418

noncomputable def parametric_curve_C (t : ℝ) : ℝ × ℝ :=
  (4 / (1 + t ^ 2), 4 * t / (1 + t ^ 2))

noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (1 + sqrt 3 * t, t)

noncomputable def point_M : ℝ × ℝ := (1, 0)

theorem curve_rectangular_coord_equation (t : ℝ) :
  let (x, y) := parametric_curve_C t in (x ≠ 0) → (x - 2) ^ 2 + y ^ 2 = 4 :=
sorry

theorem reciprocal_sum_dist (t1 t2 : ℝ) :
  let (A_x, A_y) := parametric_curve_C t1 in
  let (B_x, B_y) := parametric_curve_C t2 in
  let A_dist := sqrt ((A_x - point_M.1) ^ 2 + (A_y - point_M.2) ^ 2) in
  let B_dist := sqrt ((B_x - point_M.1) ^ 2 + (B_y - point_M.2) ^ 2) in
  (A_x, A_y) ≠ (B_x, B_y) →
  A_x = 1 + sqrt 3 * t1 ∧ A_y = t1 →
  B_x = 1 + sqrt 3 * t2 ∧ B_y = t2 →
  1 / A_dist + 1 / B_dist = sqrt 15 / 3 :=
sorry

end curve_rectangular_coord_equation_reciprocal_sum_dist_l418_418418


namespace parents_offered_to_chaperone_l418_418608

theorem parents_offered_to_chaperone (students_per_class teachers remaining_individuals : ℕ) :
  students_per_class = 10 →
  2 * students_per_class + teachers = 22 →
  teachers = 2 →
  remaining_individuals = 15 →
  remaining_individuals + 10 + 2 = 27 →
  5 = 27 - 22 :=
by
  intros h_students_per_class h_total_students_and_teachers h_teachers h_remaining_individuals h_total_individuals
  rw [←h_total_students_and_teachers, ←h_total_individuals]
  simp
  sorry

end parents_offered_to_chaperone_l418_418608


namespace sours_total_l418_418498

variable (c l o T : ℕ)

axiom cherry_sours : c = 32
axiom ratio_cherry_lemon : 4 * l = 5 * c
axiom orange_sours_ratio : o = 25 * T / 100
axiom total_sours : T = c + l + o

theorem sours_total :
  T = 96 :=
by
  sorry

end sours_total_l418_418498


namespace perpendicular_d_to_BC_l418_418744

def vector := (ℝ × ℝ)

noncomputable def AB : vector := (1, 1)
noncomputable def AC : vector := (2, 3)

noncomputable def BC : vector := (AC.1 - AB.1, AC.2 - AB.2)

def is_perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

noncomputable def d : vector := (-6, 3)

theorem perpendicular_d_to_BC : is_perpendicular d BC :=
by
  sorry

end perpendicular_d_to_BC_l418_418744


namespace time_to_clear_trains_l418_418147

def length_train_1 : ℝ := 305 -- meters
def length_train_2 : ℝ := 415 -- meters
def speed_train_1 : ℝ := 120 -- kmph
def speed_train_2 : ℝ := 150 -- kmph
def total_length := length_train_1 + length_train_2 -- meters
def relative_speed := speed_train_1 + speed_train_2 -- kmph
def relative_speed_mps := relative_speed * (5 / 18) -- m/s
def time_to_clear := total_length / relative_speed_mps -- seconds

theorem time_to_clear_trains :
  time_to_clear = 9.6 := by
  sorry

end time_to_clear_trains_l418_418147


namespace minimum_tangent_distance_l418_418037

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  let mo_sq := x^2 + y^2
  let mc_sq := (x - 2)^2 + (y - 2)^2
  let nc_sq := 1 -- radius squared of the circle
  have eq : mo_sq + 1 = mc_sq := by
    calc
      x^2 + y^2 + 1
        = (x - 2)^2 + (y - 2)^2 := sorry
  let l := {p : ℝ × ℝ // p.1 + p.2 = 7 / 4}
  let dist_to_origin := λ p : ℝ × ℝ, abs (p.1 + p.2 - 7 / 4) / real.sqrt (1^2 + 1^2)
  let min_dist_to_origin := real.sqrt 2 * 7 / 8
  min_dist_to_origin

theorem minimum_tangent_distance : minimum_value = real.sqrt 2 * 7 / 8 := 
  sorry

end minimum_tangent_distance_l418_418037


namespace largestValidNumberIs84_l418_418903

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l418_418903


namespace problem_statement_l418_418610

-- Define the function to be used
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

-- The minimum positive period G and the symmetry point for the given function
noncomputable def G := Real.pi
noncomputable def a := Real.pi / 3
noncomputable def f_max := 1

-- Statement to be proved
theorem problem_statement : 
  (∀ x, f (x + G) = f x) ∧ 
  (f a = f_max) := sorry

end problem_statement_l418_418610


namespace copy_pages_15_dollars_l418_418823

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l418_418823


namespace ratio_hexagon_triangle_area_l418_418226

theorem ratio_hexagon_triangle_area (s : ℝ) 
    (h1 : s > 0) 
    (hexagon_perimeter : 6 * (s / 2) = 3 * s) 
    (triangle_perimeter : 3 * s = 3 * s) : 
    (6 * ((sqrt 3 / 4) * (s / 2)^2)) / ((sqrt 3 / 4) * s^2) = 3 / 2 :=
by
  sorry

end ratio_hexagon_triangle_area_l418_418226


namespace boys_amount_per_person_l418_418494

theorem boys_amount_per_person (total_money : ℕ) (total_children : ℕ) (per_girl : ℕ) (number_of_boys : ℕ) (amount_per_boy : ℕ) : 
  total_money = 460 ∧
  total_children = 41 ∧
  per_girl = 8 ∧
  number_of_boys = 33 → 
  amount_per_boy = 12 :=
by sorry

end boys_amount_per_person_l418_418494


namespace circle_geometry_problem_l418_418402

theorem circle_geometry_problem
  (O : Point)
  (r : ℝ)
  (A B D C E : Point)
  (radius_OA : dist O A = r)
  (radius_OB : dist O B = r)
  (radius_OD : dist O D = r)
  (diameter_AB : collinear O A B)
  (chord_AD : D ≠ A ∧ collinear O A D ∧ ¬ collinear O A B)
  (tangent_BC : tangent  Line.B C O)
  (extension_ADC : collinear A D C)
  (AE_2DC : ∃ DC, dist A E = 2 * dist D C)
  (x y : ℝ)
  (x_def : x = dist E tangent (A))
  (y_def : y = dist E diameter (A B)) :
  y = x^2 / (4 * r) :=
sorry

end circle_geometry_problem_l418_418402


namespace lifespan_of_bat_l418_418118

variable (B H F T : ℝ)

theorem lifespan_of_bat (h₁ : H = B - 6)
                        (h₂ : F = 4 * H)
                        (h₃ : T = 2 * B)
                        (h₄ : B + H + F + T = 62) :
  B = 11.5 :=
by
  sorry

end lifespan_of_bat_l418_418118


namespace exists_four_digit_number_sum_digits_14_divisible_by_14_l418_418278

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100 % 10) % 10 + (n / 10 % 10) % 10 + (n % 10)

theorem exists_four_digit_number_sum_digits_14_divisible_by_14 :
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ sum_of_digits n = 14 ∧ n % 14 = 0 :=
sorry

end exists_four_digit_number_sum_digits_14_divisible_by_14_l418_418278


namespace last_score_is_71_l418_418047

theorem last_score_is_71 (scores : List ℕ) (h : scores = [71, 74, 79, 85, 88, 92]) (sum_eq: scores.sum = 489) :
  ∃ s : ℕ, s ∈ scores ∧ 
           (∃ avg : ℕ, avg = (scores.sum - s) / 5 ∧ 
           ∀ lst : List ℕ, lst = scores.erase s → (∀ n, n ∈ lst → lst.sum % (lst.length - 1) = 0)) :=
  sorry

end last_score_is_71_l418_418047


namespace device_records_720_instances_in_one_hour_l418_418436

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l418_418436


namespace second_set_parallel_lines_l418_418755

theorem second_set_parallel_lines (n : ℕ) (h : 7 * (n - 1) = 784) : n = 113 := 
by
  sorry

end second_set_parallel_lines_l418_418755


namespace find_a1_a2_and_t_l418_418543

-- Define the sequence a_n using the given recursive condition
def a_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) + 3^n - 1

-- Given initial condition a_3
def initial_condition (a : ℕ → ℝ) : Prop :=
  a 3 = 95

-- Define sequence b_n
def b_seq (a : ℕ → ℝ) (t : ℝ) (n : ℕ) : ℝ :=
  (a n + t) / 3^n

-- Define arithmetic sequence condition for b_seq
def arithmetic_seq (a : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → b_seq a t n - b_seq a t (n - 1) = b_seq a t (n - 1) - b_seq a t (n - 2)

-- Main theorem statement
theorem find_a1_a2_and_t :
  ∃ (a : ℕ → ℝ) (t : ℝ),
    initial_condition a ∧ a_seq a ∧ a 1 = 5 ∧ a 2 = 23 ∧
    arithmetic_seq a t ∧ t = -1/2 :=
  sorry

end find_a1_a2_and_t_l418_418543


namespace fraction_of_perimeter_grey_l418_418532

theorem fraction_of_perimeter_grey (s : ℝ) (h : s > 0) : 
  let side_length := s in
  let perimeter := 4 * side_length in
  let grey_length := (2 / 5) * side_length in
  (4 * grey_length) / perimeter = 2 / 5 := 
sorry

end fraction_of_perimeter_grey_l418_418532


namespace correct_option_is_A_l418_418170

-- Define the five statements
def statement1 : Prop := ∀ (L1 L2 : Plane) (h1 : L1 ≠ L2), ¬(intersect L1 L2) → parallel L1 L2
def statement2 : Prop := ∀ {α β : Angle} (h : α = β), ¬vertical α β → α = β
def statement3 : Prop := ∀ {L1 L2 : Line} {T : Transversal} (h : intersect T L1) (h' : intersect T L2), corresponding_angles L1 L2 T → L1.parallel_with L2
def statement4 : Prop := ∀ {L1 L2 : Line} {T : Transversal} (h : L1.parallel_with L2) (bisec : AlternateInteriorAngleBisectors L1 L2 T), bisec.parallel_with  
def statement5 : Prop := ∀ {L : Line} {P : Point} (h : P ∉ L), ∃! L', L'.parallel_with L ∧ P ∈ L'

-- Prove the correct option is selected
theorem correct_option_is_A : ¬(statement2) → True := sorry

end correct_option_is_A_l418_418170


namespace hockey_pads_cost_l418_418798

theorem hockey_pads_cost
  (initial_money : ℕ)
  (cost_hockey_skates : ℕ)
  (remaining_money : ℕ)
  (h : initial_money = 150)
  (h1 : cost_hockey_skates = initial_money / 2)
  (h2 : remaining_money = 25) :
  initial_money - cost_hockey_skates - 50 = remaining_money :=
by sorry

end hockey_pads_cost_l418_418798


namespace sum_c_d_is_11_l418_418452

noncomputable def P := (1, 3)
noncomputable def Q := (4, 7)
noncomputable def R := (8, 3)
noncomputable def S := (10, 1)

noncomputable def dist (A B : ℕ × ℕ) : ℝ :=
  Real.sqrt (((B.1 - A.1) ^ 2) + ((B.2 - A.2) ^ 2))

-- Distances
noncomputable def dPQ : ℝ := dist P Q
noncomputable def dQR : ℝ := dist Q R
noncomputable def dRS : ℝ := dist R S
noncomputable def dSP : ℝ := dist S P

-- Perimeter expression
noncomputable def perimeter : ℝ := dPQ + dQR + dRS + dSP

-- Lean statement for the problem
theorem sum_c_d_is_11 (c d : ℤ) (h : perimeter = c * Real.sqrt 2 + d * Real.sqrt 10) : c + d = 11 := 
sorry

end sum_c_d_is_11_l418_418452


namespace range_of_a3_l418_418153

open Real

def convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def sequence_condition (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, 1 ≤ n → n < 10 → abs (a n - b n) ≤ 20

def b (n : ℕ) : ℝ := n^2 - 6 * n + 10

theorem range_of_a3 (a : ℕ → ℝ) :
  convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  sequence_condition a b →
  7 ≤ a 3 ∧ a 3 ≤ 19 :=
sorry

end range_of_a3_l418_418153


namespace find_missing_number_l418_418121

theorem find_missing_number (x : ℕ) (h : (1 + x + 23 + 24 + 25 + 26 + 27 + 2) / 8 = 20) : x = 32 := 
by sorry

end find_missing_number_l418_418121


namespace count_muffin_combinations_l418_418197

theorem count_muffin_combinations (kinds total : ℕ) (at_least_one: ℕ) :
  kinds = 4 ∧ total = 8 ∧ at_least_one = 1 → 
  let valid_combinations := 23 in
  valid_combinations = 
    (1 -- case 1: 4 different kinds out of 4 remaining muffins
    + 4 -- case 2: all 4 remaining muffins of the same kind
    + (4 * 3) -- case 3: 3 of one kind, 1 of another (4 choices for 3 + 3 choices for 1)
    + (4 * 3 / 2)) -- case 4: 2 of one kind and 2 of another (combinations)
  sorry

end count_muffin_combinations_l418_418197


namespace no_18_consecutive_good_numbers_l418_418868

def is_good (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, (p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ n = p1 * p2)

theorem no_18_consecutive_good_numbers :
  ¬ ∃ (a : ℕ), ∀ i : ℕ, i < 18 → is_good (a + i) :=
sorry

end no_18_consecutive_good_numbers_l418_418868


namespace slope_angle_tangent_line_x1_eq_3pi_div_4_l418_418683

def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

def derivative (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  (differentiable_at ℝ f x).deriv

def theta (m : ℝ) : ℝ := Real.arctan m + Real.pi

theorem slope_angle_tangent_line_x1_eq_3pi_div_4 : 
  theta (derivative f 1) = 3 * Real.pi / 4 := by
  sorry

end slope_angle_tangent_line_x1_eq_3pi_div_4_l418_418683


namespace sum_inequality_l418_418353

variable {n : ℕ}
variable {k : ℝ} (hk : 1 ≤ k)
variable {x : Fin n → ℝ} (hx : ∀ i, 0 < x i)

theorem sum_inequality :
  (∑ i, 1 / (1 + x i)) * (∑ i, x i) ≤ (∑ i, (x i) ^ (k + 1) / (1 + x i)) * (∑ i, 1 / (x i) ^ k) :=
by
  sorry

end sum_inequality_l418_418353


namespace largest_divisible_by_6_ending_in_4_l418_418914

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l418_418914


namespace sphere_volume_l418_418991

theorem sphere_volume (S : ℝ) (HS : S = 256 * real.pi) : 
  ∃ V : ℝ, V = (4 / 3) * real.pi * (8 ^ 3) ∧ V = (2048 / 3) * real.pi := 
by
  sorry

end sphere_volume_l418_418991


namespace find_n_l418_418290

theorem find_n : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 4 ∧ n ≡ -1458 [MOD 5] ∧ n = 2 := by
  use 2
  split
  { norm_num }  -- 0 ≤ 2
  split
  { norm_num }  -- 2 ≤ 4
  split
  { norm_num }  -- 2 ≡ 2 [MOD 5], which implies -1458 ≡ 2 [MOD 5], since -1458 ≡ 2 [MOD 5]
  sorry         -- The proof step.

end find_n_l418_418290


namespace sum_of_three_digit_numbers_l418_418685

theorem sum_of_three_digit_numbers :
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  let Sum := n / 2 * (first_term + last_term)
  Sum = 494550 :=
by {
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  have n_def : n = 900 := by norm_num [n]
  let Sum := n / 2 * (first_term + last_term)
  have sum_def : Sum = 450 * (100 + 999) := by norm_num [Sum, first_term, last_term, n_def]
  have final_sum : Sum = 494550 := by norm_num [sum_def]
  exact final_sum
}

end sum_of_three_digit_numbers_l418_418685


namespace baseball_team_ratio_l418_418219

theorem baseball_team_ratio 
    (total_games : ℕ)
    (games_won : ℕ)
    (games_lost : ℕ)
    (h1 : total_games = 130)
    (h2 : games_won = games_lost + 14)
    (h3 : games_won = 101) :
    games_won : games_lost = 101 : 87 :=
by
  sorry

end baseball_team_ratio_l418_418219


namespace smallest_value_of_c_l418_418054

def bound_a (a b : ℝ) : Prop := 1 + a ≤ b
def bound_inv (a b c : ℝ) : Prop := (1 / a) + (1 / b) ≤ (1 / c)

theorem smallest_value_of_c (a b c : ℝ) (ha : 1 < a) (hb : a < b) 
  (hc : b < c) (h_ab : bound_a a b) (h_inv : bound_inv a b c) : 
  c ≥ (3 + Real.sqrt 5) / 2 := 
sorry

end smallest_value_of_c_l418_418054


namespace Q1_Q2_Q3_l418_418768

def rapidlyIncreasingSequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n+2) - a (n+1) > a (n+1) - a n

theorem Q1 :
  rapidlyIncreasingSequence (λ n, 2^n) :=
sorry

theorem Q2 (a : ℕ → ℤ)
  (h_rapid : rapidlyIncreasingSequence a)
  (h_int : ∀ n, a n ∈ ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (hk : a k = 2023) :
  k = 63 :=
sorry

theorem Q3 (b : ℕ → ℝ) (k : ℕ)
  (h_rapid : rapidlyIncreasingSequence b)
  (h_terms : ∀ n < 2*k, b n ∈ ℝ)
  (h_sum : (finset.range (2*k)).sum b = k)
  (c : ℕ → ℝ) (h_c : ∀ n, c n = 2^(b n)) :
  c k * c (k+1) < 2 :=
sorry

end Q1_Q2_Q3_l418_418768


namespace coeff_x4y2_in_expansion_l418_418792

theorem coeff_x4y2_in_expansion : 
  let f := (x - y) * (x + 2 * y) ^ 5 in
  (f.coeff (Fin 4)) * (f.coeff (Fin 2)) =
  30 :=
by
  sorry

end coeff_x4y2_in_expansion_l418_418792


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l418_418910

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l418_418910


namespace eccentricity_of_hyperbola_l418_418738

noncomputable def hyperbola_eccentricity (a b c : ℝ) (e : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ 
    ( ∀ x y, (x, y) = (c, 2 * c) → (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1) ) ∧ 
    (c^2 = a^2 * (e^2 - 1)) ∧ (b^2 = a^2 * e^2) ∧
    (e = Real.sqrt 2 + 1)

theorem eccentricity_of_hyperbola :
  ∀ (a b c : ℝ) (e : ℝ), hyperbola_eccentricity a b c e → e = Real.sqrt 2 + 1 :=
begin
  intros a b c e h,
  sorry
end

end eccentricity_of_hyperbola_l418_418738


namespace number_in_tenth_group_l418_418779

-- Number of students
def students : ℕ := 1000

-- Number of groups
def groups : ℕ := 100

-- Interval between groups
def interval : ℕ := students / groups

-- First number drawn
def first_number : ℕ := 6

-- Number drawn from n-th group given first_number and interval
def number_in_group (n : ℕ) : ℕ := first_number + interval * (n - 1)

-- Statement to prove
theorem number_in_tenth_group :
  number_in_group 10 = 96 :=
by
  sorry

end number_in_tenth_group_l418_418779


namespace largest_divisible_by_6_ending_in_4_l418_418912

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l418_418912


namespace triangle_area_sum_l418_418797

theorem triangle_area_sum (a b c : ℕ) (hA : ∠A = 60) (hB : ∠B = 45) (hT : bisector A intersects BC at T) (hAT : AT = 24) :
  (area_of_triangle_ABC = a + b * real.sqrt c) → (a + b + c = 291) := 
  sorry

end triangle_area_sum_l418_418797


namespace tenth_term_of_sequence_l418_418527

-- Define the first term and the common difference
def a1 : ℤ := 10
def d : ℤ := -2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) : ℤ := a1 + d * (n - 1)

-- State the theorem about the 10th term
theorem tenth_term_of_sequence : a_n 10 = -8 := by
  -- Skip the proof
  sorry

end tenth_term_of_sequence_l418_418527


namespace largest_two_digit_divisible_by_6_ending_in_4_l418_418932

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l418_418932


namespace match_view_with_option_D_l418_418958

-- Definitions based on conditions
def progress_of_science_and_technology_is_endless : Prop := 
  ∀ progress, progress ∈ science ∪ technology → progress ≠ end

def earth_capacity_supports_humanity_unlimited : Prop := 
  ∀ resources, ∃ capacity, capacity = unlimited & supports_humanity capacity resources

def no_need_to_doubt_natural_resources_potential : Prop := 
  ∀ resources, potential_power resources = undeniable

-- Composite definition for the view
def the_view : Prop :=
  progress_of_science_and_technology_is_endless ∧ 
  earth_capacity_supports_humanity_unlimited ∧ 
  no_need_to_doubt_natural_resources_potential

-- Statement of the problem
theorem match_view_with_option_D :
  the_view → (the_view == option_D) := sorry

end match_view_with_option_D_l418_418958


namespace integer_pairs_satisfying_equation_l418_418749

theorem integer_pairs_satisfying_equation : 
  {p : ℤ × ℤ // p.1 + p.2 = 2 * p.1 * p.2 }.card = 2 := 
sorry

end integer_pairs_satisfying_equation_l418_418749


namespace one_number_greater_than_one_l418_418540

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_prod: a * b * c = 1)
  (h_sum: a + b + c > 1 / a + 1 / b + 1 / c) :
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
by
  sorry

end one_number_greater_than_one_l418_418540


namespace total_rain_duration_l418_418006

theorem total_rain_duration :
  let day1 := 17 - 7 in
  let day2 := day1 + 2 in
  let day3 := day2 * 2 in
  day1 + day2 + day3 = 46 :=
by
  let day1 := 17 - 7
  let day2 := day1 + 2
  let day3 := day2 * 2
  calc
    day1 + day2 + day3 = 10 + 12 + 24 : by sorry
                     ... = 46 : by sorry

end total_rain_duration_l418_418006


namespace sum_of_roots_of_Q_eq_neg4_l418_418844

variable {a b c : ℝ}
def Q (x : ℝ) := a * x ^ 2 + b * x + c

theorem sum_of_roots_of_Q_eq_neg4
  (h : ∀ x : ℝ, Q (x^4 + x^2) ≥ Q (x^3 + 2))
  (ha : a ≠ 0) :
  (- b / a) = -4 := by
  sorry

end sum_of_roots_of_Q_eq_neg4_l418_418844


namespace alpha_div_3_range_l418_418697

theorem alpha_div_3_range (α : ℝ) (k : ℤ) 
  (h1 : Real.sin α > 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k : ℤ, (2 * k * Real.pi + Real.pi / 4 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi / 3) ∨ 
            (2 * k * Real.pi + 5 * Real.pi / 6 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi) :=
sorry

end alpha_div_3_range_l418_418697


namespace range_of_a_l418_418854

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) / 2
noncomputable def g (x : ℝ) : ℝ := (2^x + 2^(-x)) / 2

theorem range_of_a (a : ℝ) 
  (hx1 : ∀ x ∈ (set.Icc 1 2), a * f x + g (2 * x) ≥ 0) : a ≥ -17/6 :=
sorry

end range_of_a_l418_418854


namespace number_of_booklets_l418_418008

theorem number_of_booklets (pages_per_booklet : ℕ) (total_pages : ℕ) (h1 : pages_per_booklet = 9) (h2 : total_pages = 441) : total_pages / pages_per_booklet = 49 :=
by
  rw [h1, h2]
  norm_num

end number_of_booklets_l418_418008


namespace prime_square_remainder_l418_418250

theorem prime_square_remainder (p : ℕ) (hp : Nat.Prime p) (h5 : p > 5) : 
  ∃! r : ℕ, r < 180 ∧ (p^2 ≡ r [MOD 180]) := 
by
  sorry

end prime_square_remainder_l418_418250


namespace john_fixed_computers_l418_418016

theorem john_fixed_computers (total_computers unfixable waiting_for_parts fixed_right_away : ℕ)
  (h1 : total_computers = 20)
  (h2 : unfixable = 0.20 * 20)
  (h3 : waiting_for_parts = 0.40 * 20)
  (h4 : fixed_right_away = total_computers - unfixable - waiting_for_parts) :
  fixed_right_away = 8 :=
by
  sorry

end john_fixed_computers_l418_418016


namespace total_relay_schemes_l418_418964

-- Defining the problem conditions
def runners : Finset String := {"A", "B", "C"}
def segments : ℕ := 6

-- Proof statement
theorem total_relay_schemes :
  (∃ f : Fin segments → String, (f 0 ∈ {"A", "B", "C"}) ∧ (f (segments - 1) ∈ {"A", "B"})) :=
  sorry

end total_relay_schemes_l418_418964


namespace books_per_shelf_l418_418011

theorem books_per_shelf (total_books shelves : ℕ) (h_books : total_books = 12) (h_shelves : shelves = 3) :
  total_books / shelves = 4 :=
by
  rw [h_books, h_shelves]
  norm_num

end books_per_shelf_l418_418011


namespace daily_rate_is_three_l418_418215

theorem daily_rate_is_three (r : ℝ) : 
  (∀ (initial bedbugs : ℝ), initial = 30 ∧ 
  (∀ days later_bedbugs, days = 4 ∧ later_bedbugs = 810 →
  later_bedbugs = initial * r ^ days)) → r = 3 :=
by
  intros h
  sorry

end daily_rate_is_three_l418_418215


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418935

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418935


namespace copy_pages_l418_418815

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l418_418815


namespace find_a_l418_418360

theorem find_a (m c a b : ℝ) (h_m : m < 0) (h_radius : (m^2 + 3) = 4) 
  (h_c : c = 1 ∨ c = -3) (h_focus : c > 0) (h_ellipse : b^2 = 3) 
  (h_focus_eq : c^2 = a^2 - b^2) : a = 2 :=
by
  sorry

end find_a_l418_418360


namespace three_configuration_m_separable_l418_418057

theorem three_configuration_m_separable
  {n m : ℕ} (A : Finset (Fin n)) (h : m ≥ n / 2) :
  ∀ (C : Finset (Fin n)), C.card = 3 → ∃ B : Finset (Fin n), B.card = m ∧ (∀ c ∈ C, ∃ b ∈ B, c ≠ b) :=
by
  sorry

end three_configuration_m_separable_l418_418057


namespace tallest_of_shortest_vs_shortest_of_tallest_l418_418176

theorem tallest_of_shortest_vs_shortest_of_tallest 
  (people : Matrix (Fin 30) (Fin 10) ℝ)
  (A : ℝ := min (λ (r : Fin 30), max (λ (c : Fin 10), people r c)))
  (B : ℝ := max (λ (c : Fin 10), min (λ (r : Fin 30), people r c))) :
  A > B :=
by 
  -- Placeholder for the actual detailed proof.
  sorry

end tallest_of_shortest_vs_shortest_of_tallest_l418_418176


namespace probability_rectangle_not_include_shaded_l418_418188

theorem probability_rectangle_not_include_shaded :
  let num_rectangles := (1002.choose 2) * 2,
      num_rectangles_with_shaded := 501 * 501 * 2
  in (num_rectangles - num_rectangles_with_shaded) / num_rectangles = 500 / 1001 :=
by
  sorry

end probability_rectangle_not_include_shaded_l418_418188


namespace sequence_exists_for_all_k_l418_418302

theorem sequence_exists_for_all_k (n : ℕ) :
  ∀ k : ℕ, (k ∈ {1, 2, ..., n}) ↔ (∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i < n, x i > 0) ∧ (∃ i, x(i) = k)) :=
by
  sorry

end sequence_exists_for_all_k_l418_418302


namespace remainder_123456789012_mod_252_l418_418648

def M : ℕ := 123456789012
def n : ℕ := 252
def a₁ : ℕ := 4
def a₂ : ℕ := 9
def a₃ : ℕ := 7

theorem remainder_123456789012_mod_252 :
  let x := 84 in
  (M % a₁ = 0) ∧ (M % a₂ = 3) ∧ (M % a₃ = 0) → (M % n = x) := by
  sorry

end remainder_123456789012_mod_252_l418_418648


namespace angle_A_sixty_max_area_l418_418410

variables {A B C : ℝ} {a b c : ℝ}

/-- Prove that in an acute triangle ABC with sides a, b, c and given condition,
    angle A is equal to 60 degrees. -/
theorem angle_A_sixty (h1 : (b^2 + c^2 - a^2) * tan A = sqrt 3 * b * c)
  (h2 : 0 < A ∧ A < pi / 2) : A = pi / 3 :=
sorry

/-- Prove that if a = 2, the maximum value of area S of triangle ABC is sqrt 3,
    given that angle A is 60 degrees and satisfies the given condition. -/
theorem max_area (h1 : (b^2 + c^2 - a^2) * tan A = sqrt 3 * b * c)
  (h2 : 0 < A ∧ A < pi / 2) (ha : a = 2) : 
  ∃ (S : ℝ), S = sqrt 3 ∧ S ≤ sqrt 3 :=
sorry

end angle_A_sixty_max_area_l418_418410


namespace combined_yellow_blue_correct_l418_418475

-- Declare the number of students in the class
def total_students : ℕ := 200

-- Declare the percentage of students who like blue
def percent_like_blue : ℝ := 0.3

-- Declare the percentage of remaining students who like red
def percent_like_red : ℝ := 0.4

-- Function that calculates the number of students liking a certain color based on percentage
def students_like_color (total : ℕ) (percent : ℝ) : ℕ :=
  (percent * total).toInt

-- Calculate the number of students who like blue
def students_like_blue : ℕ :=
  students_like_color total_students percent_like_blue

-- Calculate the number of students who don't like blue
def students_not_like_blue : ℕ :=
  total_students - students_like_blue

-- Calculate the number of students who like red from those who don't like blue
def students_like_red : ℕ :=
  students_like_color students_not_like_blue percent_like_red

-- Calculate the number of students who like yellow (those who don't like blue or red)
def students_like_yellow : ℕ :=
  students_not_like_blue - students_like_red

-- The combined number of students who like yellow and blue
def combined_yellow_blue : ℕ :=
  students_like_blue + students_like_yellow

-- Theorem to prove that the combined number of students liking yellow and blue is 144
theorem combined_yellow_blue_correct : combined_yellow_blue = 144 := by
  sorry

end combined_yellow_blue_correct_l418_418475


namespace find_P_l418_418542

-- Define the variables A, B, C and their type
variables (A B C P : ℤ)

-- The main theorem statement according to the given conditions and question
theorem find_P (h1 : A = C + 1) (h2 : A + B = C + P) : P = 1 + B :=
by
  sorry

end find_P_l418_418542


namespace angle_transformation_proof_l418_418537

-- Definitions of angles and transformations
open Real

noncomputable def initial_angle_ACB : ℝ := 50
noncomputable def rotation_degrees : ℝ := 450
noncomputable def expected_new_angle_ACB : ℝ := 40

-- Mathematical statement to prove that the new measure of angle ACB is 40 degrees
theorem angle_transformation_proof :
  (rotate_and_reflect_angle (initial_angle_ACB) (rotation_degrees)) = expected_new_angle_ACB :=
sorry

-- Note: rotate_and_reflect_angle would be a function defined to encapsulate the rotation and reflection transformations

end angle_transformation_proof_l418_418537


namespace recipe_serves_correctly_l418_418889

theorem recipe_serves_correctly:
  ∀ (cream_fat_per_cup : ℝ) (cream_amount_cup : ℝ) (fat_per_serving : ℝ) (total_servings: ℝ),
    cream_fat_per_cup = 88 →
    cream_amount_cup = 0.5 →
    fat_per_serving = 11 →
    total_servings = (cream_amount_cup * cream_fat_per_cup) / fat_per_serving →
    total_servings = 4 :=
by
  intros cream_fat_per_cup cream_amount_cup fat_per_serving total_servings
  intros hcup hccup hfserv htserv
  sorry

end recipe_serves_correctly_l418_418889


namespace triangle_ABC_properties_l418_418472

def point := ℤ × ℤ

def distance (p1 p2 : point) : ℝ := real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def area (A B C : point) : ℝ :=
  (1 / 2) * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_ABC_properties :
  let A := (0, 0) in
  let B := (2, 3) in
  let C := (3, 5) in
  area A B C < 1 ∧ distance A B > 2 ∧ distance B C > 2 ∧ distance A C > 2 :=
by
  sorry

end triangle_ABC_properties_l418_418472


namespace binom_18_10_l418_418653

theorem binom_18_10 :
  (nat.choose 18 10) = 45760 :=
by
  have h1 : (nat.choose 16 7) = 11440 := sorry
  have h2 : (nat.choose 16 9) = 11440 := sorry
  sorry

end binom_18_10_l418_418653


namespace groups_partition_count_l418_418523

-- Definitions based on the conditions
def num_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 6
def group3_size : ℕ := 2

-- Given specific names for groups based on problem statement
def Fluffy_group_size : ℕ := group1_size
def Nipper_group_size : ℕ := group2_size

-- The total number of ways to form the groups given the conditions
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem groups_partition_count :
  total_ways 10 3 * total_ways 7 5 = 2520 := sorry

end groups_partition_count_l418_418523


namespace simplify_fraction_l418_418064

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l418_418064


namespace new_songs_added_l418_418048

-- Define the initial, deleted, and final total number of songs as constants
def initial_songs : ℕ := 8
def deleted_songs : ℕ := 5
def total_songs_now : ℕ := 33

-- Define and prove the number of new songs added
theorem new_songs_added : total_songs_now - (initial_songs - deleted_songs) = 30 :=
by
  sorry

end new_songs_added_l418_418048


namespace copy_pages_l418_418819

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l418_418819


namespace copy_pages_l418_418816

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l418_418816


namespace unique_split_sum_l418_418712

theorem unique_split_sum :
  ∀ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
  ∃! (S : set (set ℕ)), ∀ x ∈ S, (∃ (x1 x2 : ℕ) (y1 y2 y3 : ℕ), x = {x1, x2} ∧ S \ x = {y1, y2, y3} ∧ x1 + x2 = y1 + y2 + y3) → 
  ∀ x1 x2 y1 y2 y3, x1 + x2 = y1 + y2 + y3 → x1 ∈ {a, b, c, d, e} ∧ x2 ∈ {a, b, c, d, e} ∧ y1 ∈ {a, b, c, d, e} ∧ y2 ∈ {a, b, c, d, e} ∧ y3 ∈ {a, b, c, d, e} :=
by
  sorry

end unique_split_sum_l418_418712


namespace right_triangle_area_l418_418865

-- Define the conditions in the hypothesis
theorem right_triangle_area (X Y Z : ℝ × ℝ) 
  (hypotenuse_len : dist X Y = 50) 
  (right_angle_at_Z : angle Z X Y = π / 2)
  (median_X : ∀ P, P ∈ line (X, (0,0)) ↔ P.2 = P.1 + 5)
  (median_Y : ∀ P, P ∈ line (Y, (0,0)) ↔ P.2 = 3 * P.1 + 6)  :
  area (triangle XYZ) = 625 / 3 :=
sorry

end right_triangle_area_l418_418865


namespace simplify_fraction_l418_418092

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l418_418092


namespace inequality_range_of_a_l418_418535

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, sin x ^ 2 + a * cos x + a ^ 2 ≥ 1 + cos x) ↔ (a ≤ -2 ∨ a ≥ 1) :=
by
  sorry

end inequality_range_of_a_l418_418535


namespace relationship_of_magnitudes_l418_418696

noncomputable def is_ordered (x : ℝ) (A B C : ℝ) : Prop :=
  0 < x ∧ x < Real.pi / 4 ∧
  A = Real.cos (x ^ Real.sin (x ^ Real.sin x)) ∧
  B = Real.sin (x ^ Real.cos (x ^ Real.sin x)) ∧
  C = Real.cos (x ^ Real.sin (x * (x ^ Real.cos x))) ∧
  B < A ∧ A < C

theorem relationship_of_magnitudes (x A B C : ℝ) : 
  is_ordered x A B C := 
sorry

end relationship_of_magnitudes_l418_418696


namespace number_of_non_similar_isosceles_triangles_l418_418748

theorem number_of_non_similar_isosceles_triangles : 
  ∃ (n : ℕ), ∀ (n < 54 ∧ n > 45), (∃ (d : ℕ), d = 2 * n - 90 ∧ 
  ((n - d).nat_abs > 0 ∧ (n + d).nat_abs < 180 ∧ (n - d ≠ n + d))) → 
  card (set_of (λ n, n < 54 ∧ n > 45) = 8 :=
begin
  sorry

end number_of_non_similar_isosceles_triangles_l418_418748


namespace solve_for_x_l418_418577

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) (h : (5*x)^10 = (10*x)^5) : x = 2/5 :=
sorry

end solve_for_x_l418_418577


namespace max_x_y_z_c_geq_3_max_x_y_z_c_lt_3_l418_418841

variable (c x y z : ℝ) -- Define the variables as real numbers

-- Define the conditions as hypotheses
hypothesis (hc : c > 0) -- Given positive number c
hypothesis (h1 : x + c * y ≤ 36) -- First condition on x and y
hypothesis (h2 : 2 * x + 3 * z ≤ 72) -- Second condition on x and z

-- Proof for the first scenario: c ≥ 3
theorem max_x_y_z_c_geq_3 (hc_ge_3 : c ≥ 3) : (∃ x y z, x + y + z = 36) :=
  sorry

-- Proof for the second scenario: c < 3
theorem max_x_y_z_c_lt_3 (hc_lt_3 : c < 3) : (∃ x y z, x + y + z = 24 + 36 / c) :=
  sorry

end max_x_y_z_c_geq_3_max_x_y_z_c_lt_3_l418_418841


namespace triangles_no_obtuse_angles_l418_418953

variable {Point : Type} 

-- Assuming six points placed on three distinct equidistant arcs
variables {A1 A2 B1 B2 C1 C2 : Point}
variable {O : Point}
variable {arc1 arc2 arc3 : set Point}

-- Conditions
axiom equidistant_arcs : arc1 ∪ arc2 ∪ arc3 = {A1, A2, B1, B2, C1, C2}
axiom arc1_distinct : A1 ∈ arc1 ∧ A2 ∈ arc1
axiom arc2_distinct : B1 ∈ arc2 ∧ B2 ∈ arc2
axiom arc3_distinct : C1 ∈ arc3 ∧ C2 ∈ arc3
axiom arcs_equidistant : ∀ (P ∈ arc1) (Q ∈ arc2) (R ∈ arc3), ‖O - P‖ = ‖O - Q‖ ∧ ‖O - Q‖ = ‖O - R‖ ∧ ‖O - R‖ = ‖O - P‖

-- Prove that any triangle formed by these points does not contain an obtuse angle at the circle's center
theorem triangles_no_obtuse_angles :
  ( ∀ P ∈ {A1, A2}, ∀ Q ∈ {B1, B2}, ∀ R ∈ {C1, C2}, 
    let θ_POQ := ∠ P O Q, θ_QOR := ∠ Q O R, θ_ROP := ∠ R O P in 
    θ_POQ ≤ π / 2 ∧ θ_QOR ≤ π / 2 ∧ θ_ROP ≤ π / 2 ) := sorry

end triangles_no_obtuse_angles_l418_418953


namespace rationalize_denominator_l418_418896

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem rationalize_denominator :
  let a := cbrt 2
  let b := cbrt 27
  b = 3 -> ( 1 / (a + b)) = (cbrt 4 / (2 + 3 * cbrt 4))
:= by
  intro a
  intro b
  sorry

end rationalize_denominator_l418_418896


namespace sum_of_odd_integers_from_13_to_53_l418_418162

-- Definition of the arithmetic series summing from 13 to 53 with common difference 2
def sum_of_arithmetic_series (a l d : ℕ) (n : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Main theorem
theorem sum_of_odd_integers_from_13_to_53 :
  sum_of_arithmetic_series 13 53 2 21 = 693 := 
sorry

end sum_of_odd_integers_from_13_to_53_l418_418162


namespace solve_equation_l418_418546

theorem solve_equation : ∀ x : ℝ, x ≠ -2 → x ≠ 0 → (3 / (x + 2) - 1 / x = 0 ↔ x = 1) :=
by
  intro x h1 h2
  sorry

end solve_equation_l418_418546


namespace simplify_fraction_l418_418093

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l418_418093


namespace largest_divisible_by_6_ending_in_4_l418_418913

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l418_418913


namespace brendan_remaining_money_l418_418262

theorem brendan_remaining_money :
  let earnings := [1200, 1300, 1100, 1400];
      recharges := [0.60 * 1200, 0.60 * 1300, 0.40 * 1100, 0.30 * 1400];
      expenses := [200, 150, 250, 300];
      additional_costs := 500 + 1500;
      total_earnings := earnings.sum;
      total_recharged := recharges.sum;
      total_expenses := expenses.sum;
  total_earnings - (total_recharged + total_expenses + additional_costs) = -260 := 
by 
  sorry

end brendan_remaining_money_l418_418262


namespace arcsin_solution_l418_418506

theorem arcsin_solution (x : ℝ) (h_eq : arcsin x + arcsin (3 * x) = π / 4)
    (h_x_le_1 : |x| ≤ 1) (h_3x_le_1 : |3 * x| ≤ 1) : 
    x = sqrt 102 / 51 :=
begin
  sorry
end

end arcsin_solution_l418_418506


namespace john_fixes_8_computers_l418_418014

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end john_fixes_8_computers_l418_418014


namespace probability_rectangle_not_include_shaded_l418_418189

theorem probability_rectangle_not_include_shaded :
  let num_rectangles := (1002.choose 2) * 2,
      num_rectangles_with_shaded := 501 * 501 * 2
  in (num_rectangles - num_rectangles_with_shaded) / num_rectangles = 500 / 1001 :=
by
  sorry

end probability_rectangle_not_include_shaded_l418_418189


namespace problem_solution_l418_418692

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem problem_solution (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end problem_solution_l418_418692


namespace students_like_both_l418_418781

theorem students_like_both (total_students French_fries_likers burger_likers neither_likers : ℕ)
(H1 : total_students = 25)
(H2 : French_fries_likers = 15)
(H3 : burger_likers = 10)
(H4 : neither_likers = 6)
: (French_fries_likers + burger_likers + neither_likers - total_students) = 12 :=
by sorry

end students_like_both_l418_418781


namespace percentage_increase_after_additives_l418_418244

-- Define the initial conditions
def initial_time : ℕ := 25  -- initial maintenance check interval in days
def additive_A : ℝ := 0.10  -- 10% increase
def additive_B : ℝ := 0.15  -- 15% increase
def additive_C : ℝ := 0.05  -- 5% increase

-- Define the function to calculate the new time after applying one additive
def apply_additive (initial_time : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_time + (percentage_increase * initial_time)

-- Calculate the consecutive increases
def final_time : ℝ :=
  let time_after_A := apply_additive initial_time additive_A
  let time_after_B := apply_additive time_after_A additive_B
  apply_additive time_after_B additive_C

-- Calculate the percentage increase
def percentage_increase : ℝ :=
  (final_time - initial_time) / initial_time * 100

-- Given condition
theorem percentage_increase_after_additives :
  percentage_increase = 32.825 := by
  sorry

end percentage_increase_after_additives_l418_418244


namespace trigonometric_range_l418_418328

theorem trigonometric_range 
  (α t : ℝ) 
  (h1 : sin(α/2 + π/6) = t) 
  (h2 : 0 < t ∧ t ≤ 1) : 
  (cos(2*π/3 - α) / sin(α/2 + π/6)) ∈ set.Iic (1 : ℝ) :=
sorry

end trigonometric_range_l418_418328


namespace invalid_speed_against_stream_l418_418217

theorem invalid_speed_against_stream (rate_still_water speed_with_stream : ℝ) (h1 : rate_still_water = 6) (h2 : speed_with_stream = 20) :
  ∃ (v : ℝ), speed_with_stream = rate_still_water + v ∧ (rate_still_water - v < 0) → false :=
by
  sorry

end invalid_speed_against_stream_l418_418217


namespace distinct_pawn_placements_l418_418395

theorem distinct_pawn_placements : 
  let n := 5 in ((n.factorial) * (n.factorial) = 14400) :=
by 
  let n := 5
  sorry

end distinct_pawn_placements_l418_418395


namespace flash_catches_ace_distance_l418_418628

variable (v z k t_0 : ℝ)
variable (hz : z > 1)

def distance_flash_runs_to_catch_ace : ℝ :=
  z * (t_0 * v + k) / (z - 1)

theorem flash_catches_ace_distance :
  (v z k t_0 : ℝ) (hz : z > 1) :
  distance_flash_runs_to_catch_ace v z k t_0 = z * (t_0 * v + k) / (z - 1) := sorry

end flash_catches_ace_distance_l418_418628


namespace simplify_fraction_l418_418070

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l418_418070


namespace sum_of_reciprocals_l418_418131

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  (1/x) + (1/y) = 5 :=
by
  sorry

end sum_of_reciprocals_l418_418131


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l418_418908

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l418_418908


namespace friend_P_walked_35_km_l418_418566

noncomputable def distance_walked_by_friend_P : ℝ :=
  let trail_length := 60 in
  let rate_ratio := 1.4 in
  let d_Q := (trail_length / (1 + rate_ratio)) in
  let d_P := rate_ratio * d_Q in
  d_P

theorem friend_P_walked_35_km :
  (distance_walked_by_friend_P) = 35 :=
by
  -- lean proof verification skip
  sorry

end friend_P_walked_35_km_l418_418566


namespace serum_prevents_colds_l418_418613

noncomputable def hypothesis_preventive_effect (H : Prop) : Prop :=
  let K2 := 3.918
  let critical_value := 3.841
  let P_threshold := 0.05
  K2 >= critical_value ∧ P_threshold = 0.05 → H

theorem serum_prevents_colds (H : Prop) : hypothesis_preventive_effect H → H :=
by
  -- Proof will be added here
  sorry

end serum_prevents_colds_l418_418613


namespace smallest_identical_digit_divisible_by_18_l418_418294

theorem smallest_identical_digit_divisible_by_18 :
  ∃ n : Nat, (∀ d : Nat, d < n → ∃ a : Nat, (n = a * (10 ^ d - 1) / 9 + 1 ∧ (∃ k : Nat, n = 18 * k))) ∧ n = 666 :=
by
  sorry

end smallest_identical_digit_divisible_by_18_l418_418294


namespace laptop_selection_l418_418501

theorem laptop_selection :
  let A := 4
  let B := 5
  let n := 3
  A > 0 ∧ B > 0 → 
  (∃ x y : ℕ, x + y = n ∧ x ≥ 1 ∧ y ≥ 1 ∧ nat.choose A x * nat.choose B y = 70) :=
by
  sorry

end laptop_selection_l418_418501


namespace area_of_square_inscribed_circle_l418_418726

theorem area_of_square_inscribed_circle :
  let circle_eq : ℝ → ℝ → Prop := λ x y, 2 * x ^ 2 + 2 * y ^ 2 - 16 * x + 8 * y - 40 = 0
  in (∃ (side_length : ℝ), 
      (∀ (x y : ℝ), circle_eq x y → (side_length = 4 * Real.sqrt 10)) 
   → side_length^2 = 160) :=
by
  sorry

end area_of_square_inscribed_circle_l418_418726


namespace g_bound_l418_418687

noncomputable def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f(x) < f(y)

noncomputable def g (f : ℝ → ℝ) (x y : ℝ) : ℝ :=
  (f (x + y) - f x) / (f x - f (x - y))

theorem g_bound (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_half_two : ∀ x y, (x = 0 ∧ y > 0) ∨ (x ≠ 0 ∧ 0 < y ∧ y ≤ |x|) → 1 / 2 < g f x y ∧ g f x y < 2)
  (x y : ℝ)
  (hy_pos : y > 0) :
  1 / 14 < g f x y ∧ g f x y < 14 :=
sorry

end g_bound_l418_418687


namespace simplify_fraction_l418_418094

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l418_418094


namespace ben_overall_correct_percentage_is_89_l418_418649

def total_problems : ℝ := sorry
def chloe_correct_alone_percentage : ℝ := 0.70
def chloe_total_correct_percentage : ℝ := 0.84
def ben_correct_alone_percentage : ℝ := 0.80
def half (t : ℝ) : ℝ := t / 2

theorem ben_overall_correct_percentage_is_89 :
  let solo_correct_chloe := chloe_correct_alone_percentage * half total_problems in
  let total_correct_chloe := chloe_total_correct_percentage * total_problems in
  let together_correct := total_correct_chloe - solo_correct_chloe in
  let solo_correct_ben := ben_correct_alone_percentage * half total_problems in
  let total_correct_ben := solo_correct_ben + together_correct in
  (total_correct_ben / total_problems) * 100 = 89 :=
  sorry

end ben_overall_correct_percentage_is_89_l418_418649


namespace order_of_abc_l418_418691

theorem order_of_abc (a b c : ℝ) (h1 : a = 0.3 ^ 2) (h2 : b = Real.log 0.3 / Real.log 2) (h3 : c = 2 ^ 0.3) : b < a ∧ a < c :=
by
  have fact1 : 0 < a := by sorry -- Placeholder for intermediate proof that a > 0
  have fact2 : a < 1 := by sorry -- Placeholder for intermediate proof that a < 1
  have fact3 : b < 0 := by sorry -- Placeholder for intermediate proof that b < 0
  have fact4 : c > 1 := by sorry -- Placeholder for intermediate proof that c > 1
  exact ⟨fact3, fact4⟩ -- Combining the intermediate results to show b < a and a < c

end order_of_abc_l418_418691


namespace arithmetic_sequence_general_term_minimum_value_of_T_l418_418346

-- Part (I) Define the arithmetic sequence and prove the general term formula
-- Given conditions: a₁ = 3, S₃ = 15
-- Goal: Prove the formula aₙ = 2 * n + 1 is correct

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 3) ∧ (S 3 = 15) ∧ (∀ n, S n = n * (3 + a (n + 1) - a 1) / 2) →
  (∀ n, a n = 2 * n + 1) :=
by
  sorry

-- Part (II) Define the sum of inverses sequence Tₙ and prove its minimum value
-- Given conditions: Sₙ = n * (n + 2)
-- Goal: Prove the minimum value of Tₙ is 1/3

theorem minimum_value_of_T (T : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, S n = n * (n + 2)) ∧ (∀ n, T n = ∑ i in Finset.range n, 1 / S i) →
  (∀ n, T n = (1 / 2) * (3 / 2 - 1 / (n + 1) - 1 / (n + 2))) ∧
  (∃ m, T m = 1 / 3) :=
by
  sorry

end arithmetic_sequence_general_term_minimum_value_of_T_l418_418346


namespace part1_part3_l418_418773

-- Part (1) Lean statement
theorem part1 (a : ℕ → ℝ) (h : ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n) :
  (∀ n, 2^((n : ℕ)) ∈ {a n}) := sorry

-- Part (2) Lean statement
noncomputable def part2 (a : ℕ → ℤ) (hn : ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n)
  (h1 : a 1 = 1) (h2 : a 2 = 3) (hk : ∃ (k : ℕ), a k = 2023) :
  ∃ (k : ℕ), k * (k + 1) / 2 ≤ 2023 := sorry

-- Part (3) Lean statement
theorem part3 (b : ℕ → ℤ) (k : ℕ) (h : 2 ≤ k) (hsum: (∑ i in range (2 * k), b i) = k)
  (hb : ∀ n, b (n + 2) - b (n + 1) > b (n + 1) - b n) :
  ∀ {c : ℕ → ℝ}, (∀ n, c n = 2^(b n)) → c k * c (k + 1) < 2 := sorry

end part1_part3_l418_418773


namespace _l418_418396

open Real

noncomputable def log_expr (x : ℝ) := log (x + 5) + log (x - 2)
noncomputable def rhs_expr (x : ℝ) := log (x^2 - 7 * x + 10)
noncomputable def expr_eq : ℝ → Prop := λ x, log_expr x = rhs_expr x

noncomputable theorem no_real_solutions (x : ℝ) : ¬ expr_eq x :=
begin
  sorry
end

end _l418_418396


namespace find_y_l418_418342

noncomputable def transformation (b : ℕ → ℝ) (n : ℕ) : ℝ :=
if h : n > 0 then (3 * b (n - 1) + 2 * b n) / 5 else b n

noncomputable def iterate_transform (k : ℕ) (b : ℕ → ℝ) : ℕ → ℝ
| 0 => λ n => b n
| (m + 1) => transformation (iterate_transform m b)

theorem find_y (y : ℝ) (h : y > 0) :
  let T n := y ^ n
  iterate_transform 49 T 0 = 1 / 5 ^ 24.5 
  → y = (√5 - 3) / 2 :=
by
  intros
  sorry

end find_y_l418_418342


namespace find_initial_cards_l418_418045

-- Define the variables and conditions
def Marcus_initial_cards (total_cards now_cards_given : ℕ) : ℕ :=
  total_cards - now_cards_given

-- Define the theorem to prove the initial number of cards Marcus had
theorem find_initial_cards :
  ∀ (total_cards now_cards_given : ℕ), total_cards = 268 → now_cards_given = 58 → Marcus_initial_cards total_cards now_cards_given = 210 :=
by
  intros total_cards now_cards_given h_total h_given
  rw [Marcus_initial_cards]
  rw [h_total, h_given]
  exact rfl
  

end find_initial_cards_l418_418045


namespace simplify_fraction_l418_418071

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l418_418071


namespace arcsin_solution_l418_418504

theorem arcsin_solution (x : ℝ) (h_eq : arcsin x + arcsin (3 * x) = π / 4)
    (h_x_le_1 : |x| ≤ 1) (h_3x_le_1 : |3 * x| ≤ 1) : 
    x = sqrt 102 / 51 :=
begin
  sorry
end

end arcsin_solution_l418_418504


namespace length_of_walls_l418_418009

-- Definitions of the given conditions.
def wall_height : ℝ := 12
def third_wall_length : ℝ := 20
def third_wall_height : ℝ := 12
def total_area : ℝ := 960

-- The area of two walls with length L each and height 12 feet.
def two_walls_area (L : ℝ) : ℝ := 2 * L * wall_height

-- The area of the third wall.
def third_wall_area : ℝ := third_wall_length * third_wall_height

-- The proof statement
theorem length_of_walls (L : ℝ) (h1 : two_walls_area L + third_wall_area = total_area) : L = 30 :=
by
  sorry

end length_of_walls_l418_418009


namespace problem_1_problem_2_l418_418397

-- Definitions and conditions for proof problem 1
open Real
def strict_boundary_function (f g F : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, f x < F x ∧ F x < g x

-- Definitions of functions
def f1 (x : ℝ) := 1 + x
def g1 (x : ℝ) := 1 + x + x^2 / 2
def F1 (x : ℝ) := exp x
def D := Ioo (-1 : ℝ) 0

-- Proof problem 1 statement
theorem problem_1 : strict_boundary_function f1 g1 F1 D :=  
by 
  sorry

-- Definitions and conditions for proof problem 2
def h (x : ℝ) := 2 * exp x + (1 / (1 + x)) - 2
def M := 8 

-- Proof problem 2 statement
theorem problem_2 : ∀ x ∈ Ioo (-1 : ℝ) 0, h x > M / 10 := 
by 
  sorry

end problem_1_problem_2_l418_418397


namespace smallest_a_l418_418296

theorem smallest_a (a : ℕ) (h_a : a > 8) : (∀ x : ℤ, ¬ Prime (x^4 + a^2)) ↔ a = 9 :=
by
  sorry

end smallest_a_l418_418296


namespace min_expression_value_l418_418954

theorem min_expression_value :
  ∀ (a b c d e f : ℝ), 
    (0 ≤ a ∧ a ≤ 3) ∧ (0 ≤ b ∧ b ≤ 3) ∧ (0 ≤ c ∧ c ≤ 3) ∧ (0 ≤ d ∧ d ≤ 3) ∧ 
    (0 ≤ e ∧ e ≤ 3) ∧ (0 ≤ f ∧ f ≤ 3) ∧ 
    (a + b + c + d = 6) ∧ (e + f = 2) →
    \left(\sqrt{a^{2} + 4} + \sqrt{b^{2} + e^{2}} + \sqrt{c^{2} + f^{2}} + \sqrt{d^{2} + 4}\right)^{2} ≥ 72 :=
sorry

end min_expression_value_l418_418954


namespace total_rain_duration_l418_418000

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end total_rain_duration_l418_418000


namespace ratio_Delta_y_Delta_x_l418_418339

theorem ratio_Delta_y_Delta_x (Δx Δy : ℝ) : 
  (1, 2) ∈ set_of (λ p : ℝ × ℝ, p.2 = p.1^2 + 1) →
  (1 + Δx, 2 + Δy) ∈ set_of (λ p : ℝ × ℝ, p.2 = p.1^2 + 1) →
  Δy / Δx = Δx + 2 :=
by
  sorry

end ratio_Delta_y_Delta_x_l418_418339


namespace lambda_value_l418_418320

noncomputable def find_lambda (ω : ℂ) (λ : ℝ) : Prop :=
  abs(ω) = 3 ∧ λ > 1 ∧ 
  (let ω2 := ω^2
       ω3 := ω^3 
       λω2 := λ * ω2 in
   abs(ω2) = 9 ∧ abs(ω3) = 27 ∧ 
   abs(ω3 - ω2) = abs(λω2 - ω2) ∧ 
   abs(λω2 - ω3) = abs(ω3 - ω2))

theorem lambda_value (ω : ℂ) (λ : ℝ) (h : find_lambda ω λ) : λ = 3 :=
  sorry

end lambda_value_l418_418320


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l418_418942

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l418_418942


namespace eagles_win_at_least_three_out_of_five_l418_418108

noncomputable theory

def probability_of_eagles_winning_at_least_three_games (p : ℝ) (n : ℕ) : ℝ :=
  (nat.choose n 3 * p^3 * (1 - p)^(n - 3)) + 
  (nat.choose n 4 * p^4 * (1 - p)^(n - 4)) + 
  (nat.choose n 5 * p^5 * (1 - p)^(n - 5))

theorem eagles_win_at_least_three_out_of_five :
  probability_of_eagles_winning_at_least_three_games (1/2) 5 = 1/2 :=
by 
  sorry

end eagles_win_at_least_three_out_of_five_l418_418108


namespace tan_alpha_value_l418_418386

theorem tan_alpha_value 
(α : ℝ)
(h : (sin α - 2 * cos α) / (2 * sin α + 5 * cos α) = -5) : tan α = -23 / 11 := 
sorry

end tan_alpha_value_l418_418386


namespace sample_size_is_five_l418_418326

def population := 100
def sample (n : ℕ) := n ≤ population
def sample_size (n : ℕ) := n

theorem sample_size_is_five (n : ℕ) (h : sample 5) : sample_size 5 = 5 :=
by
  sorry

end sample_size_is_five_l418_418326


namespace probability_dot_product_condition_l418_418365

-- Define the ellipse with its equation
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the foci of the ellipse
def foci1 : ℝ × ℝ := (- sqrt 3, 0)
def foci2 : ℝ × ℝ := ( sqrt 3, 0)

-- Define that M is any point on the major axis A1A2
def on_major_axis (x : ℝ) : Prop :=
  -2 ≤ x ∧ x ≤ 2 ∧ ∀ y : ℝ, y = 0

-- Define point P on the ellipse, obtained by intersection of the line through M perpendicular to A1A2
def point_P (M : ℝ) : ℝ × ℝ × ℝ :=
  if -2 ≤ M ∧ M ≤ 2 then (M, sqrt (1 - M^2 / 4), sqrt (1 - M^2 / 4)) else (0, 0, 0)

-- Define the condition for the dot product of vectors PF1 and PF2 to be less than 0
def dot_product_less_than_zero (P : ℝ × ℝ × ℝ) : Prop :=
  let M := P.1,
  let P1 := (M + sqrt 3, sqrt (1 - M^2 / 4)),
  let P2 := (M - sqrt 3, sqrt (1 - M^2 / 4))
  in (P1.1 * P2.1 + P1.2 * P2.2) < 0

-- Probability calculation for dot product condition
def probability : ℝ :=
  (2 * sqrt 6 / 3) / 4

-- The Lean statement for the proof problem
theorem probability_dot_product_condition :
  ∀ M : ℝ, -2 ≤ M ∧ M ≤ 2 ∧ dot_product_less_than_zero (point_P M) ↔ (probability = sqrt 6 / 3) :=
by {
  intros,
  sorry
}

end probability_dot_product_condition_l418_418365


namespace total_four_digit_numbers_proof_even_four_digit_numbers_proof_greater_than_4301_numbers_proof_l418_418376

-- Definition for the total number of four-digit numbers
def total_four_digit_numbers : ℕ := 300

-- Definition for the number of four-digit even numbers
def even_four_digit_numbers : ℕ := 156

-- Definition for the number of four-digit numbers greater than 4301
def greater_than_4301_numbers : ℕ := 83

theorem total_four_digit_numbers_proof :
  (∃ digits : Finset ℕ, 
     digits.card = 6 ∧ ∀ x ∈ digits, x < 6) →
  (4 < 6) → 
  (factorial 5 * (factorial 5 / factorial 2)) / factorial 1 = total_four_digit_numbers := 
by sorry

theorem even_four_digit_numbers_proof :
  (∃ digits : Finset ℕ, 
     digits.card = 6 ∧ ∀ x ∈ digits, x < 6) →
  (4 < 6) →
  (factorial 5 * (factorial 5 / factorial 2 + 4 * 2 * (factorial 4 / factorial 2))) / factorial 1 = even_four_digit_numbers := 
by sorry

theorem greater_than_4301_numbers_proof :
  (∃ digits : Finset ℕ, 
     digits.card = 6 ∧ ∀ x ∈ digits, x < 6) →
  (4 < 6) →
  (factorial 5 * (factorial 5 / factorial 2) + factorial 4 / factorial 2 + (factorial 4 / factorial 2 - 1)) / factorial 1 = greater_than_4301_numbers := 
by sorry

end total_four_digit_numbers_proof_even_four_digit_numbers_proof_greater_than_4301_numbers_proof_l418_418376


namespace decagon_diagonals_l418_418380

-- Define the number of sides of a decagon
def n : ℕ := 10

-- Define the formula for the number of diagonals in an n-sided polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem decagon_diagonals : num_diagonals n = 35 := by
  sorry

end decagon_diagonals_l418_418380


namespace relationship_of_a_b_c_l418_418032

noncomputable def a : ℝ := log (1/2) / log (1/3)
noncomputable def b : ℝ := log (1/3) / log (1/2)
noncomputable def c : ℝ := log (4/3) / log 3

theorem relationship_of_a_b_c : b > a ∧ a > c := sorry

end relationship_of_a_b_c_l418_418032


namespace minimize_BC_with_fixed_angle_area_l418_418050

variable {α : ℝ} -- fixed angle α in radians
variable {S : ℝ} -- fixed area S

-- Definitions for a triangle with a given angle and area
def area_of_triangle (b c : ℝ) : ℝ := (1 / 2) * b * c * Real.sin α

theorem minimize_BC_with_fixed_angle_area (b c : ℝ) :
  area_of_triangle b c = S → (∀ b c, area_of_triangle b c = S → (∀ a : ℝ, a = Real.sqrt (b^2 + c^2 - 2*b*c * Real.cos α) 
  → ∀ a' : ℝ, a' = Real.sqrt (c^2 + c^2 - 2*c*c * Real.cos α) → a' ≤ a)) :=
by
  intros h_area b c h_area a h_a a' h_a'
  have h_iso : b = c := sorry
  have hay := Real.sqrt_nonneg (c^2 + c^2 - 2 * c * c * Real.cos α)
  have hay' := Real.sqrt_nonneg (b^2 + c^2 - 2 * b * c * Real.cos α)
  sorry

end minimize_BC_with_fixed_angle_area_l418_418050


namespace function_neither_even_nor_odd_l418_418660

noncomputable def f (x : ℝ) : ℝ := log (x + sqrt (9 + x^2))

theorem function_neither_even_nor_odd :
  ¬(∀ x, f (-x) = f x) ∧ ¬(∀ x, f (-x) = -f x) :=
sorry

end function_neither_even_nor_odd_l418_418660


namespace greenville_university_volume_l418_418578

-- Define the dimensions of the box
def box_length := 20
def box_width := 20
def box_height := 12

-- Define the cost per box
def cost_per_box := 0.50

-- Define the minimum amount the university must spend on boxes
def minimum_spent := 250.0

-- Calculate the volume of one box
def volume_of_one_box := box_length * box_width * box_height

-- Calculate the number of boxes the university can buy with $250
def number_of_boxes := minimum_spent / cost_per_box

-- Calculate the total volume needed for the collection
def total_volume : ℤ := volume_of_one_box * number_of_boxes

-- The theorem statement that needs to be proved
theorem greenville_university_volume : total_volume = 2400000 :=
by
  sorry

end greenville_university_volume_l418_418578


namespace decagon_diagonals_l418_418382

theorem decagon_diagonals : ∀ n : ℕ, n = 10 → (n * (n - 3) / 2) = 35 :=
by
  intros n hn
  rw [hn]
  norm_num
  sorry

end decagon_diagonals_l418_418382


namespace polynomial_solution_l418_418284

theorem polynomial_solution (P : ℝ → ℝ) (h : ∀ x : ℝ, x * P(x-1) = (x-2) * P(x)) :
  ∃ a : ℝ, ∀ x : ℝ, P(x) = a * x * (x-1) :=
by
  -- Proof would go here
  sorry

end polynomial_solution_l418_418284


namespace total_distance_to_run_l418_418879

theorem total_distance_to_run
  (track_length : ℕ)
  (initial_laps : ℕ)
  (additional_laps : ℕ)
  (total_laps := initial_laps + additional_laps) :
  track_length = 150 →
  initial_laps = 6 →
  additional_laps = 4 →
  total_laps * track_length = 1500 := by
  sorry

end total_distance_to_run_l418_418879


namespace sequence_exists_l418_418314

variable (n : ℕ)

theorem sequence_exists (k : ℕ) (hkn : k ∈ Set.range (λ x : ℕ, x + 1) n) :
  ∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i → i ≤ n → x (i+1) > x i) ∧ (∀ i, x i ∈ ℕ) :=
sorry

end sequence_exists_l418_418314


namespace Julio_age_is_10_l418_418986

-- Defining the given conditions
def Zipporah_age : ℕ := 7
def combined_age_Zipporah_Dina : ℕ := 51
def combined_age_Julio_Dina : ℕ := 54

-- Defining the problem to find Julio's age and proving it is correct
theorem Julio_age_is_10
    (Zipporah_age + Dina_age = combined_age_Zipporah_Dina)
    (Julio_age + Dina_age = combined_age_Julio_Dina) : 
    Julio_age = 10 :=
by
  sorry

end Julio_age_is_10_l418_418986


namespace cone_volume_and_height_l418_418212

theorem cone_volume_and_height
  (r_sector : ℝ) (sector_ratio : ℝ) (radius_circle : ℝ) (slant_height : ℝ) :
  r_sector = 5/6 →
  radius_circle = 6 →
  slant_height = radius_circle →
  2 * π * slant_height * r_sector = 2 * π * 5 →
  let h := real.sqrt 11 in
  let V := (1/3) * π * (5^2) * h in
  h = real.sqrt 11 ∧ V = (25/3) * π * (real.sqrt 11) :=
by
  sorry

end cone_volume_and_height_l418_418212


namespace work_done_together_l418_418598

theorem work_done_together
    (A_time : ℝ) (B_time : ℝ) (A_rate : ℝ) (B_rate : ℝ) (work_done_together_rate : ℝ) :
    A_time = 18 ∧ B_time = A_time / 2 ∧ 
    A_rate = 1 / A_time ∧ B_rate = 1 / B_time ∧ 
    work_done_together_rate = A_rate + B_rate → work_done_together_rate = 1 / 6 :=
by
  intros h
  rw [h.1, h.2.1, h.2.2.1, h.2.2.2, h.2.2.2]
  have B_time_calc : B_time = 9 := by
    rw [h.1] at h.2.1
    exact h.2.1
  have A_rate_calc : A_rate = 1 / 18 := by
    rw [h.1] at h.2.2.1
    exact h.2.2.1
  have B_rate_calc : B_rate = 1 / 9 := by
    rw [B_time_calc] at h.2.2.2
    exact h.2.2.2
  have total_rate_calc : work_done_together_rate = 1 / 18 + 1 / 9 := by
    rw [A_rate_calc, B_rate_calc]
    exact h.2.2.2
  have common_denom : 1 / 18 + 1 / 9 = 1 / 6 := by
    rw [add_comm (1 / 18) (1 / 9)]
    norm_num
  rw [total_rate_calc, common_denom]
  exact rfl

end work_done_together_l418_418598


namespace find_m_l418_418725

theorem find_m (y x m : ℝ) (h1 : 2 - 3 * (1 - y) = 2 * y) (h2 : y = x) (h3 : m * (x - 3) - 2 = -8) : m = 3 :=
sorry

end find_m_l418_418725


namespace part_i_part_ii_l418_418352

open Set

variable (a : ℝ) (x : ℝ)

def A (a : ℝ) : Set ℝ := { x : ℝ | (x - 2) * (x - (3 * a + 1)) < 0 }
def B (a : ℝ) : Set ℝ := { x : ℝ | 2 * a < x ∧ x < a^2 + 1 }

-- Part (i)
theorem part_i : A (-2) ∪ B (-2) = { x | -5 < x ∧ x < 5 } :=
sorry

-- Part (ii)
theorem part_ii : ∀ (x : ℝ), (∀ a, a ∈ ({x : ℝ | 1 ≤ x} ∩ {x : ℝ | x ≤ 3}) ∪ {-1}) → B a ⊆ A a :=
sorry

end part_i_part_ii_l418_418352


namespace find_a_b_l418_418461

theorem find_a_b :
  ∃ (a b : ℝ), 
  (∃ x : ℝ, x^3 + a * x^2 - 2 * x + b = 0) ∧ 
  (∃ y : ℂ, y ≠ x ∧ y = 2 - 3 * complex.I) ∧
  (polynomial.C (a : ℂ) * polynomial.X^2 + polynomial.C (-2 : ℂ) * polynomial.X + polynomial.C b).eval (2 - 3 * complex.I) = 0 ∧ 
  (polynomial.C (a : ℂ) * polynomial.X^2 + polynomial.C (-2 : ℂ) * polynomial.X + polynomial.C b).eval (2 + 3 * complex.I) = 0 ∧ 
  a = -1/4 ∧ b = 195/4 :=
sorry

end find_a_b_l418_418461


namespace find_original_number_l418_418236

theorem find_original_number (x y : ℕ) (h1 : x + y = 8) (h2 : 10 * y + x = 10 * x + y + 18) : 10 * x + y = 35 := 
sorry

end find_original_number_l418_418236


namespace kim_total_ounces_l418_418836

def quarts_to_ounces (q : ℚ) : ℚ := q * 32

def bottle_quarts : ℚ := 1.5
def can_ounces : ℚ := 12
def bottle_ounces : ℚ := quarts_to_ounces bottle_quarts

def total_ounces : ℚ := bottle_ounces + can_ounces

theorem kim_total_ounces : total_ounces = 60 :=
by
  -- Proof will go here
  sorry

end kim_total_ounces_l418_418836


namespace proof_l418_418421

noncomputable def parametric_line :=
  { x := λ t : ℝ, -2 + t, y := λ t : ℝ, -2 * t }

def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * y = 0

def polar_eq_line (ρ θ : ℝ) : Prop :=
  2 * ρ * Float.cos θ + ρ * Float.sin θ + 4 = 0

def min_distance (d r : ℝ) : ℝ :=
  Float.sqrt (d^2 - r^2)

theorem proof (
  -- Given parametric equations of line l
  (line_param : ∃ t : ℝ, parametric_line.x t = x ∧ parametric_line.y t = y),
  -- Given the standard equation of circle C
  (circle : circle_eq x y),
  -- Center of circle C at (0, 1) and radius 1
  (center : x = 0 ∧ y = 1)
  ) :
  -- Proof part 1: The polar equation of line l
  (∃ ρ θ : ℝ, polar_eq_line ρ θ) ∧
  -- Proof part 2: The minimum value of |MA| and the polar coordinates of point M
  (min_distance (Float.sqrt 5) 1 = 2) ∧
  (∃ ρ θ : ℝ, (ρ = 2 ∧ θ = Float.pi)) := by
  sorry

end proof_l418_418421


namespace math_proof_problem_l418_418650

noncomputable def problem_statement : Prop :=
  ∃ (A B C D P L T : Type)
    (circle : Circle center := O radius := 2.5)
    (chordAB chordCD : Segment)
    (lengthChords : chordAB.length = 4 ∧ chordCD.length = 4)
    (extensions : segmentBA extension beyond A ∧ segmentCD extension beyond D)
    (intersect : lineBA extension ∧ lineCD extension → intersect point P)
    (linePO : Line) 
    (intersectPO : linePO intersect segmentAC at L)
    (ratio : AL / LC = 2 / 3)
    (inscribedCenter : T = center inscribed circle in ∆ACP)
    (AP : P distance from A)
    (PT : T distance from P)
    (areaACP : Area of ∆ACP),
  AP = 8 ∧ PT = (sqrt 409 - 5) / 2 ∧ areaACP = 5760 / 409

theorem math_proof_problem : problem_statement :=
  by sorry

end math_proof_problem_l418_418650


namespace right_triangle_ratio_segments_l418_418785

theorem right_triangle_ratio_segments (a b c r s : ℝ) (h : a^2 + b^2 = c^2) (h_drop : r + s = c) (a_to_b_ratio : 2 * b = 5 * a) : r / s = 4 / 25 :=
sorry

end right_triangle_ratio_segments_l418_418785


namespace perimeter_of_square_l418_418968

theorem perimeter_of_square (s : ℝ) (area : s^2 = 468) : 4 * s = 24 * Real.sqrt 13 := 
by
  sorry

end perimeter_of_square_l418_418968


namespace find_a_l418_418962

theorem find_a (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : 1 / (a:nat) + 1 / (b:nat) + 1 / (c:nat) = 1) 
  (h4 : a + b + c = 11) :
  a = 2 :=
sorry

end find_a_l418_418962


namespace abc_inequality_l418_418982

theorem abc_inequality
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1)
  (hc : 0 < c ∧ c < 1) :
  a + b + c + 2 * a * b * c > a * b + b * c + c * a + 2 * real.sqrt (a * b * c) := by
sorry

end abc_inequality_l418_418982


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418925

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418925


namespace common_ratio_l418_418987

-- Define the geometric sequence and the sum of the first n terms
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def sum_geom_seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = a 0 * (1 - q ^ (n + 1)) / (1 - q)

-- Define the conditions
def a_3 := 4
def S_3 := 12

-- Main theorem statement
theorem common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  geom_seq a q → sum_geom_seq a S → a 2 = a_3 → S 2 = S_3 → (q = 1 ∨ q = -1/2) :=
by
  intros h1 h2 h3 h4
  sorry

end common_ratio_l418_418987


namespace find_complex_and_quadratic_solution_l418_418456

def complex_number_properties (z : ℂ) (a b : ℝ) (m n : ℝ) : Prop :=
  z = a + b * complex.I ∧
  a > 0 ∧
  abs z = 2 * real.sqrt 5 ∧
  ((1 + 2 * complex.I) * z).re = 0 ∧
  ∃ z_conjugate : ℂ, z_conjugate = complex.conj(z) ∧
                      z_conjugate = a - b * complex.I ∧
                      m = -((a + b * complex.I) + (a - b * complex.I)).re ∧
                      n = (a + b * complex.I) * (a - b * complex.I).re

theorem find_complex_and_quadratic_solution 
  (z : ℂ) (a b : ℝ) (m n : ℝ) 
  (H1 : complex_number_properties z a b m n) : 
  z = 4 + 2 * complex.I ∧ m = -8 ∧ n = 20 := 
by {
  sorry,
}

end find_complex_and_quadratic_solution_l418_418456


namespace interval_monotonicity_max_min_values_on_interval_l418_418368

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem interval_monotonicity :
  (∀ x, x < -1 → f (x + 1) > f x) ∧
  (∀ x, -1 < x ∧ x < 1 → f (x + 1) < f x) ∧
  (∀ x, x > 1 → f (x + 1) > f x) := sorry

theorem max_min_values_on_interval :
  (∀ x ∈ Icc (-3 : ℝ) 2, f x ≥ f (-3)) ∧
  (f (-3) = -18) ∧
  (∀ x ∈ Icc (-3 : ℝ) 2, f x ≤ 2) ∧
  (f (-1) = 2) ∧
  (f 2 = 2) := sorry

end interval_monotonicity_max_min_values_on_interval_l418_418368


namespace blake_score_guarantee_l418_418568

-- Define the conditions and outcome of the game scenario
theorem blake_score_guarantee : ∃ k : ℕ, k = 4 ∧ ∀ strategy_ruby, score_blake ≥ k := 
sorry

end blake_score_guarantee_l418_418568


namespace area_C_greater_than_sum_A_B_l418_418536

noncomputable def side_length_of_square_A := a : ℝ
noncomputable def side_length_of_square_B := 2 * a
noncomputable def side_length_of_square_C := 2.8 * a

noncomputable def area_square_A := a^2
noncomputable def area_square_B := (2 * a)^2
noncomputable def area_square_C := (2.8 * a)^2

theorem area_C_greater_than_sum_A_B : 
  100 * ((area_square_C - (area_square_A + area_square_B)) / (area_square_A + area_square_B)) = 56.8 :=
by
  sorry

end area_C_greater_than_sum_A_B_l418_418536


namespace largest_divisible_by_6_ending_in_4_l418_418918

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l418_418918


namespace cubic_sum_identity_l418_418520

   theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = -1) :
     a^3 + b^3 + c^3 = 12 :=
   by
     sorry
   
end cubic_sum_identity_l418_418520


namespace largest_two_digit_divisible_by_6_ending_in_4_l418_418931

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l418_418931


namespace find_r_l418_418843

-- Define the hexagon and points as described in the conditions
def hexagon : Type := sorry
constant A B C D E F : hexagon

constant AC CE : ℝ -- diagonals of the hexagon
constant side_length : ℝ -- side length of the hexagon
constant sqrt_3 : ℝ -- √3
axiom AC_length_eq_CE_length : AC = sqrt_3
axiom CE_length_eq_sqrt_3 : CE = sqrt_3

-- Define points M and N which divide AC and CE internally
constant M N : hexagon
constant r : ℝ -- the ratio

axiom AM_length : ∀ AC, ∀ r, r * AC = AM
axiom CN_length : ∀ CE, ∀ r, r * CE = CN

-- Collinearity constraint
axiom collinearity_B_M_N : sorry -- existence of a proof establishing B, M, and N collinear 

-- The main theorem to prove the value of r making B, M, and N collinear
theorem find_r {r : ℝ} (h : collinearity_B_M_N) :
  r = 1 / sqrt_3 := 
sorry

end find_r_l418_418843


namespace Bruce_total_payment_l418_418263

theorem Bruce_total_payment 
  (kg_grapes : ℕ) (kg_grapes_rate : ℕ) 
  (kg_mangoes : ℕ) (kg_mangoes_rate : ℕ) (eur_to_usd : ℝ) 
  (kg_oranges : ℕ) (kg_oranges_rate : ℕ) (pound_to_usd : ℝ) 
  (kg_apples : ℕ) (kg_apples_rate : ℕ) (yen_to_usd : ℝ)
  (discount_grapes : ℝ) (tax_mangoes : ℝ) (premium_oranges : ℝ) : 
  kg_grapes = 8 ∧ kg_grapes_rate = 70 ∧ 
  kg_mangoes = 8 ∧ kg_mangoes_rate = 55 ∧ eur_to_usd = 1.15 ∧ 
  kg_oranges = 5 ∧ kg_oranges_rate = 40 ∧ pound_to_usd = 1.25 ∧ 
  kg_apples = 10 ∧ kg_apples_rate = 3000 ∧ yen_to_usd = 0.009 ∧
  discount_grapes = 0.10 ∧ tax_mangoes = 0.05 ∧ premium_oranges = 0.03 → 
  let cost_grapes := kg_grapes * kg_grapes_rate in
  let cost_mangoes := kg_mangoes * kg_mangoes_rate * eur_to_usd in
  let cost_oranges := kg_oranges * kg_oranges_rate * pound_to_usd in
  let cost_apples := kg_apples * kg_apples_rate * yen_to_usd in
  let final_grapes := cost_grapes * (1 - discount_grapes) in
  let final_mangoes := cost_mangoes * (1 + tax_mangoes) in
  let final_oranges := cost_oranges * (1 + premium_oranges) in
  final_grapes + final_mangoes + final_oranges + cost_apples = 1563.10 :=
by
  intros
  sorry

end Bruce_total_payment_l418_418263


namespace probability_AB_meet_in_final_l418_418689

-- Define the teams
inductive Team
| A | B | C | D
deriving DecidableEq, Fintype

-- Define equal probability of winning, group formation, and final match
noncomputable def probability_meet_in_final : ℚ :=
  -- Probability that A and B are in the same group AND one of them reaches the final
  let P1 := 1 / 3 in
  -- Probability that A and B are in different groups AND both win their matches
  let P2 := (2 / 3) * (1 / 2) * (1 / 2) in
  -- Total probability
  P1 + P2

-- The statement to prove
theorem probability_AB_meet_in_final :
  probability_meet_in_final = 1 / 2 :=
by
  sorry

end probability_AB_meet_in_final_l418_418689


namespace total_rain_duration_l418_418001

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end total_rain_duration_l418_418001


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l418_418907

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l418_418907


namespace isosceles_triangle_vertex_angle_l418_418710

variables {x : ℝ}

def angle1 (x : ℝ) : ℝ := 2 * x - 2
def angle2 (x : ℝ) : ℝ := 3 * x - 5

theorem isosceles_triangle_vertex_angle :
  ∃ x : ℝ, angle1 x = 76 ∨ angle2 x = 46 ∨ angle1 x = angle2 x ∧ (180 - angle1 x - angle2 x = 172) := 
begin
  sorry -- Proof goes here
end

end isosceles_triangle_vertex_angle_l418_418710


namespace total_fertilizer_spread_l418_418213

-- Definitions derived from the conditions
variables (total_area small_area : ℕ)
variables (fertilizer_small : ℕ)
variables (even_spread : Prop)

-- Given Conditions:
def cond_area_field : total_area = 9600 := rfl
def cond_area_small : small_area = 3600 := rfl
def cond_fertilizer_small : fertilizer_small = 300 := rfl
def cond_even_spread : even_spread := true

-- Prove the total fertilizer spread
theorem total_fertilizer_spread 
  (h1 : total_area = 9600)
  (h2 : small_area = 3600)
  (h3 : fertilizer_small = 300)
  (h4 : even_spread)
  : ∃ (total_fertilizer : ℕ), total_fertilizer = 800 :=
begin
  use 800,
  sorry
end

end total_fertilizer_spread_l418_418213


namespace probability_of_A_winning_l418_418194

-- Define the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p  -- probability of losing a set

-- Formulate the probabilities for each win scenario
def P_WW : ℝ := p * p
def P_LWW : ℝ := q * p * p
def P_WLW : ℝ := p * q * p

-- Calculate the total probability of winning the match
def total_probability : ℝ := P_WW + P_LWW + P_WLW

-- Prove that the total probability of A winning the match is 0.648
theorem probability_of_A_winning : total_probability = 0.648 :=
by
    -- Provide the calculation details
    sorry  -- replace with the actual proof steps if needed, otherwise keep sorry to skip the proof

end probability_of_A_winning_l418_418194


namespace andrews_age_l418_418246

-- Define Andrew's age
variable (a g : ℚ)

-- Problem conditions
axiom condition1 : g = 10 * a
axiom condition2 : g - (a + 2) = 57

theorem andrews_age : a = 59 / 9 := 
by
  -- Set the proof steps aside for now
  sorry

end andrews_age_l418_418246


namespace concurrency_or_parallel_GE_FD_CB_l418_418445

theorem concurrency_or_parallel_GE_FD_CB
    (A B C D E F G : Point)
    (hD_in_triangle_ABC : D ∈ interior (triangle A B C))
    (h_angle_CDA_CBA : ∠ C D A + ∠ C B A = 180)
    (h_meets_circle_E : second_intersection (line_through C D) (circumcircle A B C) = E)
    (hG_on_circle_with_center_C : G ∈ circle_centered_at C (dist C D))
    (hG_on_arc_AC : G ∈ arc_AC_not_containing_B (circumcircle A B C))
    (h_circle_A_radius_AD : F ∈ second_intersection_circle_A_radius_AD (circumcircle B C D)) :
    concurrency_or_parallel (line_through G E) (line_through F D) (line_through C B) :=
sorry

end concurrency_or_parallel_GE_FD_CB_l418_418445


namespace recorded_instances_l418_418431

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l418_418431


namespace probability_red_bean_l418_418803

section ProbabilityRedBean

-- Initially, there are 5 red beans and 9 black beans in a bag.
def initial_red_beans : ℕ := 5
def initial_black_beans : ℕ := 9
def initial_total_beans : ℕ := initial_red_beans + initial_black_beans

-- Then, 3 red beans and 3 black beans are added to the bag.
def added_red_beans : ℕ := 3
def added_black_beans : ℕ := 3
def final_red_beans : ℕ := initial_red_beans + added_red_beans
def final_black_beans : ℕ := initial_black_beans + added_black_beans
def final_total_beans : ℕ := final_red_beans + final_black_beans

-- The probability of drawing a red bean should be 2/5
theorem probability_red_bean :
  (final_red_beans : ℚ) / final_total_beans = 2 / 5 := by
  sorry

end ProbabilityRedBean

end probability_red_bean_l418_418803


namespace trajectory_of_C_l418_418790

def point := (ℝ × ℝ)

noncomputable def A : point := (3, 1)
noncomputable def B : point := (-1, 3)

theorem trajectory_of_C : 
  ∃ (λ1 λ2 : ℝ) (C : point), 
    λ1 + λ2 = 1 ∧ 
    C = (λ1 * A.1 + λ2 * B.1, λ1 * A.2 + λ2 * B.2) ∧ 
    (C.1 + 2 * C.2 = 5) :=
begin
  sorry
end

end trajectory_of_C_l418_418790


namespace recorded_instances_l418_418433

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l418_418433


namespace integral_of_3x_plus_sin_x_l418_418669

-- Define the integrand function
def integrand (x : ℝ) := 3 * x + Real.sin x

-- State the theorem
theorem integral_of_3x_plus_sin_x :
  ∫ x in 0..(Real.pi / 2), integrand x = (3 / 8) * Real.pi ^ 2 + 1 := by
  sorry

end integral_of_3x_plus_sin_x_l418_418669


namespace unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l418_418731

noncomputable def f (x k : ℝ) : ℝ := (Real.log x) - k * x + k

theorem unique_solution_f_geq_0 {k : ℝ} :
  (∃! x : ℝ, 0 < x ∧ f x k ≥ 0) ↔ k = 1 :=
sorry

theorem inequality_hold_for_a_leq_1 {a x : ℝ} (h₀ : a ≤ 1) :
  x * (f x 1 + x - 1) < Real.exp x - a * x^2 - 1 :=
sorry

end unique_solution_f_geq_0_inequality_hold_for_a_leq_1_l418_418731


namespace perpendicular_bisector_b_value_l418_418119

theorem perpendicular_bisector_b_value (b : ℝ) :
  (∀ (P Q : ℝ × ℝ), P = (2, 1) ∧ Q = (8, 9) →
   ∃ (M : ℝ × ℝ), M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧
   ∃ (L : ℝ → ℝ → Prop), L = (λ x y, 2 * x + 3 * y = b) ∧
   L M.1 M.2) →
  b = 25 := 
by
  sorry

end perpendicular_bisector_b_value_l418_418119


namespace joe_eats_at_least_two_kinds_of_fruit_l418_418639

noncomputable def joe_probability_at_least_two_kinds_fruit : ℚ := 
  1 - (4 * (1 / 4) ^ 3)

theorem joe_eats_at_least_two_kinds_of_fruit : 
  joe_probability_at_least_two_kinds_fruit = 15 / 16 := by
  sorry

end joe_eats_at_least_two_kinds_of_fruit_l418_418639


namespace f_lt_2_l418_418698

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f (x + 2) = f (-x + 2)

axiom f_ge_2 (x : ℝ) (h : x ≥ 2) : f x = x^2 - 6 * x + 4

theorem f_lt_2 (x : ℝ) (h : x < 2) : f x = x^2 - 2 * x - 4 :=
by
  sorry

end f_lt_2_l418_418698


namespace PQ_parallel_to_MH_l418_418450

-- To define the problem setup in Lean
variables {A B C O M H P Q X Y : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] [MetricSpace M]
          [MetricSpace H] [MetricSpace P] [MetricSpace Q] [MetricSpace X] [MetricSpace Y]
          [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O] [Inhabited M]
          [Inhabited H] [Inhabited P] [Inhabited Q] [Inhabited X] [Inhabited Y]

-- Assumptions that match the conditions listed
variable (triangle_ABC : Triangle A B C)
variable (circumcircle_ABC : Circumcircle A B C O)
variable (orthocenter_H : Orthocenter A B C H)
variable (foot_of_altitude_B : FootOfAltitude B H X)
variable (foot_of_altitude_C : FootOfAltitude C H Y)
variable (midpoint_M : Midpoint B C M)
variable (P_on_AO_BC : LineIntersection (Line AO) (Segment B C) P)
variable (Q_on_XY_CH : LineIntersection (Segment X Y) (Line CH) Q)

-- The statement to prove
theorem PQ_parallel_to_MH :
  Parallel (LineSegment P Q) (LineSegment M H) :=
sorry

end PQ_parallel_to_MH_l418_418450


namespace sequence_exists_for_all_k_l418_418305

theorem sequence_exists_for_all_k (n : ℕ) :
  ∀ k : ℕ, (k ∈ {1, 2, ..., n}) ↔ (∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i < n, x i > 0) ∧ (∃ i, x(i) = k)) :=
by
  sorry

end sequence_exists_for_all_k_l418_418305


namespace product_of_roots_of_quadratic_l418_418681

theorem product_of_roots_of_quadratic :
  (∃ a b c : ℝ, a = 25 ∧ b = 60 ∧ c = -375 ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 → ∃ p q : ℝ, p * q = -15)) :=
by
  let a := 25
  let b := 60
  let c := -375
  have h : ∀ x : ℝ, a * x^2 + b * x + c = 0 → a * c / a = -15 
  { intros x h_eq,
    rw [h_eq, mul_div_right_comm, mul_assoc, mul_one],
    exact -15 },
  use [a, b, c],
  exact ⟨rfl, rfl, rfl, h⟩

end product_of_roots_of_quadratic_l418_418681


namespace cookies_baked_on_monday_is_32_l418_418173

-- Definitions for the problem.
variable (X : ℕ)

-- Conditions.
def cookies_baked_on_monday := X
def cookies_baked_on_tuesday := X / 2
def cookies_baked_on_wednesday := 3 * (X / 2) - 4

-- Total cookies at the end of three days.
def total_cookies := cookies_baked_on_monday X + cookies_baked_on_tuesday X + cookies_baked_on_wednesday X

-- Theorem statement to prove the number of cookies baked on Monday.
theorem cookies_baked_on_monday_is_32 : total_cookies X = 92 → cookies_baked_on_monday X = 32 :=
by
  -- We would add the proof steps here.
  sorry

end cookies_baked_on_monday_is_32_l418_418173


namespace g_definition_l418_418117

def f (x : ℝ) : ℝ :=
  if x < 0 then -2 - x
  else if x >= 0 ∧ x < 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x >= 2 ∧ x <= 3 then 2 * (x - 2)
  else 0

noncomputable def g (x : ℝ) : ℝ :=
  f (4 - x) + 3

theorem g_definition : 
  ∀ x : ℝ, g(x) = f (4 - x) + 3 :=
by
  intro x
  simp [g]
  sorry -- Proof is omitted as per instructions.

end g_definition_l418_418117


namespace age_difference_l418_418994

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : C + 11 = A :=
by {
  sorry
}

end age_difference_l418_418994


namespace midpoints_concyclic_l418_418025

open EuclideanGeometry

variables {A B C X Y D E K L M N : Point}
variables (Ω : Circle) {ABC : Triangle}

theorem midpoints_concyclic (hcircum : circumcircle ABC Ω)
  (hX : on_circle X Ω) (hY : on_circle Y Ω) (hD : intercircpoint XY AB D) (hE : intercircpoint XY AC E) :
  concyclic { midpoint BE, midpoint CD, midpoint DE, midpoint XY } :=
begin
  sorry
end

end midpoints_concyclic_l418_418025


namespace exists_sequence_for_k_l418_418312

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end exists_sequence_for_k_l418_418312


namespace distance_triangle_inequality_distance_strict_inequality_l418_418588

section
variable (Φ1 Φ2 Φ3 : Type)
variable (ρ : Φ1 → Φ2 → ℝ)

-- Part (a)
theorem distance_triangle_inequality :
  (ρ Φ1 Φ2) + (ρ Φ2 Φ3) ≥ (ρ Φ1 Φ3) :=
sorry

-- Part (b)
theorem distance_strict_inequality (cond1 : (ρ Φ1 Φ2 = 0)) (cond2 : (ρ Φ2 Φ3 = 0)) (h : ∃ c > 0, c = ρ Φ3 Φ1) :
  (ρ Φ1 Φ2) + (ρ Φ2 Φ3) < (ρ Φ3 Φ1) :=
sorry
end

end distance_triangle_inequality_distance_strict_inequality_l418_418588


namespace permutation_cosine_sum_zero_l418_418448

theorem permutation_cosine_sum_zero (n : ℕ) (h : n > 0) (hn : ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ p ∣ n ∧ q ∣ n) :
  ∃ σ : fin n → fin n, (∑ k in finset.range n, (k + 1 : ℂ) * complex.cos(2 * real.pi * (σ k : ℂ) / n)) = 0 := 
sorry

end permutation_cosine_sum_zero_l418_418448


namespace price_of_a_bag_of_cherries_is_5_l418_418767

theorem price_of_a_bag_of_cherries_is_5
  (olive_price : ℝ := 7)
  (num_bags_cherries : ℝ := 50)
  (num_bags_olives : ℝ := 50)
  (discount : ℝ := 0.9)
  (total_paid : ℝ := 540) :
  let C := (total_paid / discount - num_bags_olives * olive_price) / num_bags_cherries in
  C = 5 := 
by
  sorry

end price_of_a_bag_of_cherries_is_5_l418_418767


namespace value_of_x_l418_418591

theorem value_of_x (m n : ℝ) (h1 : ∀ y z, x = m * y^2) (h2 : ∀ z, y = n / z.sqrt) 
                   (h3 : x = 8) (h4 : z = 16) :
                   (h => x = 2) when (h5 : z = 64) :=
sorry

end value_of_x_l418_418591


namespace solve_arcsin_eq_l418_418511

theorem solve_arcsin_eq (x : ℝ) :
  arcsin x + arcsin (3 * x) = π / 4 → x = sqrt 102 / 51 ∨ x = -sqrt 102 / 51 :=
by 
  sorry

end solve_arcsin_eq_l418_418511


namespace largest_two_digit_divisible_by_6_ending_in_4_l418_418927

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l418_418927


namespace simple_interest_rate_l418_418682

theorem simple_interest_rate (P T SI : ℝ) (hP : P = 10000) (hT : T = 1) (hSI : SI = 400) :
    (SI = P * 0.04 * T) := by
  rw [hP, hT, hSI]
  sorry

end simple_interest_rate_l418_418682


namespace general_term_and_min_value_proof_l418_418345

-- Define the arithmetic sequence and conditions
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

-- State the given conditions
axiom arith_seq_conditions :
  S 7 = 0 ∧ (a 3 - 2 * a 2 = 12) ∧ (S n = n * (a 1 + a n) / 2)

-- Define the properties to be proven
theorem general_term_and_min_value_proof :
  (∀ n, a n = 4 * n - 16) ∧ (∃ n, n ∈ ℕ ∧ S n - 11 * n + 40 = -38) :=
begin
  sorry
end

end general_term_and_min_value_proof_l418_418345


namespace curve_eqns_l418_418420

theorem curve_eqns (α θ : ℝ) :
    (∀ α, x = 2 * Real.cos α ∧ y = Real.sqrt(2) * Real.sin α → 
          (x^2 / 4 + y^2 / 2 = 1) ∧ 
          ∀ θ ρ, ρ * Real.sin(θ + Real.pi / 4) = 3 * Real.sqrt(2) → 
                 (ρ * Real.cos θ + ρ * Real.sin θ = 6) ∧ 
                 (let P1 := (2 * Real.cos α, Real.sqrt(2) * Real.sin α) in
                  let P2 := (x, y) in
                  ∀ x y, x + y = 6 → 
                        ∃ θ, min_dist P1 P2 = 3 * Real.sqrt(2) - Real.sqrt(3))) :=
sorry

end curve_eqns_l418_418420


namespace product_roots_cos_pi_by_9_cos_2pi_by_9_l418_418961

theorem product_roots_cos_pi_by_9_cos_2pi_by_9 :
  ∀ (d e : ℝ), (∀ x, x^2 + d * x + e = (x - Real.cos (π / 9)) * (x - Real.cos (2 * π / 9))) → 
    d * e = -5 / 64 :=
by
  sorry

end product_roots_cos_pi_by_9_cos_2pi_by_9_l418_418961


namespace value_of_b_prod_l418_418533

-- Conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 2 ^ (n - 1)

-- The goal is to prove that b_{a_1} * b_{a_3} * b_{a_5} = 4096
theorem value_of_b_prod : b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end value_of_b_prod_l418_418533


namespace property_P_two_of_2019_l418_418845

open Set Function

noncomputable def P (S : Set ℕ) (F : Set (S → S)) (k : ℕ) : Prop :=
  ∀ x y ∈ S, ∃ f : Fin k → (S → S),
  (∀ i : Fin k, f i ∈ F) ∧
  (f k.succ "f⁻²(k)" (f k.pred (f 0 x))) = (f k.succ "f⁻²(k)" (f k.pred (f 0 y)))

theorem property_P_two_of_2019 {S : Set ℕ} (hS : card S = 35) 
  (F : Set (S → S)) (hF : P S F 2019) : P S F 2 := sorry

end property_P_two_of_2019_l418_418845


namespace determine_n_equals_dsquared_l418_418276

def num_divisors (n : ℕ) : ℕ :=
  finset.card (finset.filter (λ d, n % d = 0) (finset.range (n+1)))

theorem determine_n_equals_dsquared (n : ℕ) (h : n = (num_divisors n)^2) : n = 1 := 
by 
  sorry

end determine_n_equals_dsquared_l418_418276


namespace simplify_fraction_l418_418074

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l418_418074


namespace solve_arcsin_eq_l418_418510

theorem solve_arcsin_eq (x : ℝ) :
  arcsin x + arcsin (3 * x) = π / 4 → x = sqrt 102 / 51 ∨ x = -sqrt 102 / 51 :=
by 
  sorry

end solve_arcsin_eq_l418_418510


namespace pieces_of_green_candy_l418_418560

theorem pieces_of_green_candy (total_pieces red_pieces blue_pieces : ℝ)
  (h_total : total_pieces = 3409.7)
  (h_red : red_pieces = 145.5)
  (h_blue : blue_pieces = 785.2) :
  total_pieces - red_pieces - blue_pieces = 2479 := by
  sorry

end pieces_of_green_candy_l418_418560


namespace cost_of_paving_l418_418182

def length : ℝ := 5.5
def width : ℝ := 4
def rate_per_sq_meter : ℝ := 850

theorem cost_of_paving :
  rate_per_sq_meter * (length * width) = 18700 :=
by
  sorry

end cost_of_paving_l418_418182


namespace min_value_frac_a4b_plus_1a_l418_418399

theorem min_value_frac_a4b_plus_1a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ m : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (x / (4 * y) + 1 / x) ≥ m) ∧ m = 2 :=
by
  use 2
  intros a b h1 h2 h3
  have h₄ : 0 < a / (4 * b) + 1 / a, from sorry
  have h₅ : a / (4 * b) + 1 / a ≥ 2, from sorry
  exact ⟨h₃, h₅.sorry⟩

end min_value_frac_a4b_plus_1a_l418_418399


namespace num_positive_integers_l418_418319

theorem num_positive_integers (m : ℕ) : 
  (∃ n, m^2 - 2 = n ∧ n ∣ 2002) ↔ (m = 2 ∨ m = 3 ∨ m = 4) :=
by
  sorry

end num_positive_integers_l418_418319


namespace room_lamp_switch_l418_418775

theorem room_lamp_switch (k : ℕ) (h : k ≥ 1) 
  (rooms : fin (2 * k) → fin 6 → bool)
  (pairs : fin (3 * k) → (fin 6) × (fin 6)) 
  (switch_config : fin (3 * k) → bool)
  (h_lamps : ∀ i, ∃ a b c, [a, b, c] = rooms i)
/- We state that, given 2k rooms and 3k switches, each controlling a pair of lamps,
   there exists a configuration of switches such that each room has at least one lamp on and one lamp off. -/
  (h_switch: ∃ (switch_config: fin (3 * k) → bool), ∀ i, ∃ (l1 l2 l3 : bool), 
      [l1, l2, l3] = [rooms i 0 (switch_config (pairs 0).1), 
                      rooms i 1 (switch_config (pairs 1).2), 
                      rooms i 2 (switch_config (pairs 2).2)]
    ∧ (l1 ≠ l2 ∨ l1 ≠ l3 ∨ l2 ≠ l3)) : 
  ∃ (switch_config: fin (3 * k) → bool), ∀ i, ∃ (l1 l2 l3 : bool), 
      [l1, l2, l3] = [rooms i 0 (switch_config (pairs 0).1), 
                      rooms i 1 (switch_config (pairs 1).2), 
                      rooms i 2 (switch_config (pairs 2).2)]
    ∧ (l1 ≠ l2 ∨ l1 ≠ l3 ∨ l2 ≠ l3) := 
sorry

end room_lamp_switch_l418_418775


namespace derivative_value_at_3pi_over_4_l418_418370

noncomputable def f (x : ℝ) := Real.sin x - 2 * Real.cos x + 1

theorem derivative_value_at_3pi_over_4 : 
  Deriv (λ x, Real.sin x - 2 * Real.cos x + 1) (3 * Real.pi / 4) = (Real.sqrt 2) / 2 := 
by {
  sorry
}

end derivative_value_at_3pi_over_4_l418_418370


namespace curve_C_properties_l418_418662

---- a) Identifying questions and conditions in the given problem
---- Question: Identify all true conclusions among the given options.
---- Conditions: 
---- Curve C is the locus of points such that the sum of distances to points F₁(-1,0), F₂(1,0), and F₃(0,1) is 2√2.
---- The given conclusions are:
---- ① Curve C is symmetrical about both the x-axis and the y-axis.
---- ② There exists a point P on curve C such that |PF₃| = 2√2/3
---- ③ If point P is on curve C, then the maximum area of ∆F₁PF₂ is 1.
---- ④ The maximum area of ∆PF₂F₃ is √3/2.

---- b) Identifying solution steps and the correct answers in the given solution
---- Solution steps are not to be considered directly as we only use final conditions and correct answers. 
---- Correct answer(s): ③

---- c) Translate the (question, conditions, correct answer) tuple to an equivalent proof problem
---- Prove: The only correct conclusion among the given conclusions (①, ②, ③, ④) is ③
---- Given: Curve C is defined by the sum of distances to three fixed points F1(-1,0), F2(1,0), F3(0,1) is equal to 2√2.

---- d) Rewrite the problem in Lean 4 statement (lean only needs the statement, no proof)

theorem curve_C_properties :
  let F1 := (-1 : ℝ, 0 : ℝ)
  let F2 := (1 : ℝ, 0 : ℝ)
  let F3 := (0 : ℝ, 1 : ℝ)
  (∀ (x y : ℝ), (√((x + 1)^2 + y^2) + √((x - 1)^2 + y^2) + √(x^2 + (y - 1)^2) = 2 * √2) → 
  ((∃ P : ℝ × ℝ, (P ∈ set_of (λ (P : ℝ × ℝ), √(P.1^2 + (P.2 - 1)^2) = 2 * √2 / 3))) = false ∧
   (the maximum area of (triangle F1 P F2) = 1) = true ∧
   (curve C is symmetrical about both the x-axis and the y-axis) = false ∧
   (the maximum area of (triangle P F2 F3) = √3 / 2) = false) :=
begin
  sorry -- proof not required as per instructions
end

end curve_C_properties_l418_418662


namespace Q1_Q2_Q3_l418_418769

def rapidlyIncreasingSequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n+2) - a (n+1) > a (n+1) - a n

theorem Q1 :
  rapidlyIncreasingSequence (λ n, 2^n) :=
sorry

theorem Q2 (a : ℕ → ℤ)
  (h_rapid : rapidlyIncreasingSequence a)
  (h_int : ∀ n, a n ∈ ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (hk : a k = 2023) :
  k = 63 :=
sorry

theorem Q3 (b : ℕ → ℝ) (k : ℕ)
  (h_rapid : rapidlyIncreasingSequence b)
  (h_terms : ∀ n < 2*k, b n ∈ ℝ)
  (h_sum : (finset.range (2*k)).sum b = k)
  (c : ℕ → ℝ) (h_c : ∀ n, c n = 2^(b n)) :
  c k * c (k+1) < 2 :=
sorry

end Q1_Q2_Q3_l418_418769


namespace range_of_2a_minus_b_l418_418329

variable (a b : ℝ)
variable (h1 : -2 < a ∧ a < 2)
variable (h2 : 2 < b ∧ b < 3)

theorem range_of_2a_minus_b (a b : ℝ) (h1 : -2 < a ∧ a < 2) (h2 : 2 < b ∧ b < 3) :
  -7 < 2 * a - b ∧ 2 * a - b < 2 := sorry

end range_of_2a_minus_b_l418_418329


namespace john_fixes_8_computers_l418_418013

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end john_fixes_8_computers_l418_418013


namespace minimum_value_inequality_l418_418357

def minimum_value_inequality_problem : Prop :=
∀ (a b : ℝ), (0 < a) → (0 < b) → (a + 3 * b = 1) → (1 / a + 1 / (3 * b)) = 4

theorem minimum_value_inequality : minimum_value_inequality_problem :=
sorry

end minimum_value_inequality_l418_418357


namespace simplify_fraction_l418_418090

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l418_418090


namespace range_of_a_l418_418765

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -5/2 :=
by 
  sorry

end range_of_a_l418_418765


namespace find_second_interest_rate_l418_418623

variable (total_amount first_amount first_rate total_interest second_interest_rate : ℕ)
variable (investment_parts_total : total_amount = 4000)
variable (first_investment_amount : first_amount = 2800)
variable (first_investment_rate : first_rate = 3)
variable (total_investment_interest : total_interest = 144)

theorem find_second_interest_rate :
  (second_interest_rate = 5) →
  let second_amount := total_amount - first_amount,
  let first_interest := first_amount * first_rate / 100,
  let second_interest := total_interest - first_interest,
  let second_rate := second_interest * 100 / second_amount in
  second_rate = second_interest_rate :=
by
  -- proof goes here
  sorry

end find_second_interest_rate_l418_418623


namespace how_many_people_in_group_l418_418390

-- Definition of the conditions
def ratio_likes_football : ℚ := 24 / 60
def ratio_plays_football_given_likes : ℚ := 1 / 2
def expected_to_play_football : ℕ := 50

-- Combining the ratios to get the fraction of total people playing football
def ratio_plays_football : ℚ := ratio_likes_football * ratio_plays_football_given_likes

-- Total number of people in the group
def total_people_in_group : ℕ := 250

-- Proof statement
theorem how_many_people_in_group (expected_to_play_football : ℕ) : 
  ratio_plays_football * total_people_in_group = expected_to_play_football :=
by {
  -- Directly using our definitions
  sorry
}

end how_many_people_in_group_l418_418390


namespace line_equation_unique_l418_418558

theorem line_equation_unique (m b k : ℝ) (h_intersect_dist : |(k^2 + 6*k + 5) - (m*k + b)| = 7)
  (h_passing_point : 8 = 2*m + b) (hb_nonzero : b ≠ 0) :
  y = 10*x - 12 :=
by
  sorry

end line_equation_unique_l418_418558


namespace equations_not_equivalent_l418_418637

theorem equations_not_equivalent :
  ∀ x : ℝ, (x + 7 + 10 / (2 * x - 1) = 8 - x + 10 / (2 * x - 1)) ↔ false :=
by
  intro x
  sorry

end equations_not_equivalent_l418_418637


namespace complement_of_M_wrt_U_l418_418864

open Set

example : Set ℕ := {1, 2, 3, 4 }

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_of_M_wrt_U : compl U M = {2, 3} :=
by {
  sorry
}

end complement_of_M_wrt_U_l418_418864


namespace find_2009th_2010th_terms_l418_418318

noncomputable def seq (a : ℕ → ℕ) : Prop := ∀ n : ℕ, n > 0 → 
  (∑ i in Finset.range n, a (i + 1)) / n = n

theorem find_2009th_2010th_terms (a : ℕ → ℕ) (h : seq a) : 
  a 2009 = 4017 ∧ a 2010 = 4019 :=
by
  sorry

end find_2009th_2010th_terms_l418_418318


namespace total_rain_duration_l418_418007

theorem total_rain_duration :
  let day1 := 17 - 7 in
  let day2 := day1 + 2 in
  let day3 := day2 * 2 in
  day1 + day2 + day3 = 46 :=
by
  let day1 := 17 - 7
  let day2 := day1 + 2
  let day3 := day2 * 2
  calc
    day1 + day2 + day3 = 10 + 12 + 24 : by sorry
                     ... = 46 : by sorry

end total_rain_duration_l418_418007


namespace option_A_option_B_option_C_option_D_m_pos_option_D_m_neg_main_problem_l418_418331

theorem option_A (a b : ℝ) (h : a > b) : a - 1 > b - 1 :=
by exact sub_gt_sub_right h 1

theorem option_B (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by exact mul_lt_mul_of_neg_left h (by norm_num)

theorem option_C (a b : ℝ) (h : a > b) : (1 / 2) * a + 1 > (1 / 2) * b + 1 :=
by exact add_lt_add_right (mul_lt_mul_of_pos_left h (by norm_num)) 1

theorem option_D_m_pos (a b m : ℝ) (h : a > b) (hm : m > 0) : ¬ (m * a < m * b) :=
by exact not_lt.mpr (mul_lt_mul_of_pos_left h hm)

theorem option_D_m_neg (a b m : ℝ) (h : a > b) (hm : m < 0) : m * a < m * b :=
by exact mul_lt_mul_of_neg_left h hm

theorem main_problem (a b m : ℝ) (h : a > b) : 
(¬ (a - 1 > b - 1)) ∨ (¬ (-2 * a < -2 * b)) ∨ (¬ ((1 / 2) * a + 1 > (1 / 2) * b + 1)) ∨ ( ∃ (m : ℝ), (m > 0 ∧ ¬ (m * a < m * b)) ∨ (m < 0 ∧ (m * a < m * b))) :=
begin
  left,
  exact option_D_m_pos a b m h,
  right,
  exact option_D_m_neg a b m h,
  sorry -- rest of the proof is left out since it's not needed as per the instructions
end

end option_A_option_B_option_C_option_D_m_pos_option_D_m_neg_main_problem_l418_418331


namespace possible_values_of_m_l418_418348

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line equation
def line_eq (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for the minimum chord length being 2
def min_chord_length_condition (k m : ℝ) : Prop :=
  let r := 2 in
  let d := sqrt (4 - (2 / 2)^2) in
  d = sqrt 3 ∧ (| m | / sqrt (1 + k^2)) = sqrt 3

-- Problem statement to prove
theorem possible_values_of_m (k m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, circle_eq x y ∧ line_eq k m x y) →
  min_chord_length_condition k m →
  m = sqrt 3 ∨ m = -sqrt 3 :=
by sorry

end possible_values_of_m_l418_418348


namespace simplify_fraction_l418_418061

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l418_418061


namespace lines_parallel_l418_418794

open Real

/-- Given the following angle conditions
    in a triangle, prove that lines EC and FD are parallel:
    
    In △BCE: 
    - ∠CBE = 42°
    - ∠BCE = 48°
    
    In △AFD:
    - ∠FAD = 28°
    - ∠ADF = 62°

  --/
theorem lines_parallel
  (angle_CBE : ℝ) (angle_BCE : ℝ)
  (angle_FAD : ℝ) (angle_ADF : ℝ)
  (h1 : angle_CBE = 42) (h2 : angle_BCE = 48)
  (h3 : angle_FAD = 28) (h4 : angle_ADF = 62) :
  is_parallel (line_through EC) (line_through FD) :=
by
  sorry

end lines_parallel_l418_418794


namespace students_who_like_yellow_and_blue_l418_418477

/-- Problem conditions -/
def total_students : ℕ := 200
def percentage_blue : ℕ := 30
def percentage_red_among_not_blue : ℕ := 40

/-- We need to prove the following statement: -/
theorem students_who_like_yellow_and_blue :
  let num_blue := (percentage_blue * total_students) / 100 in
  let num_not_blue := total_students - num_blue in
  let num_red := (percentage_red_among_not_blue * num_not_blue) / 100 in
  let num_yellow := num_not_blue - num_red in
  num_yellow + num_blue = 144 :=
by
  sorry

end students_who_like_yellow_and_blue_l418_418477


namespace Marla_laps_per_hour_l418_418469

theorem Marla_laps_per_hour (M : ℝ) :
  (0.8 * M = 0.8 * 5 + 4) → M = 10 :=
by
  sorry

end Marla_laps_per_hour_l418_418469


namespace simplify_fraction_l418_418062

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l418_418062


namespace prime_square_remainders_l418_418256

theorem prime_square_remainders (p : ℕ) (hp : Prime p) (hpg : p > 5) : 
  ∃ n : ℕ, n = 2 ∧ ∀ r ∈ {r : ℕ | ∃ p : ℕ, Prime p ∧ p > 5 ∧ r = (p ^ 2 % 180)}, r ∈ {1, 64} :=
by sorry

end prime_square_remainders_l418_418256


namespace simplify_fraction_l418_418069

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l418_418069


namespace at_least_one_column_with_8_people_l418_418594

/-- In a grid with 17 columns, if 65 people start in the central column
    and at each step, each person takes a step either to the left or
    to the right randomly, then there is always at least one column
    that contains at least 8 people. -/
theorem at_least_one_column_with_8_people 
  (num_people : ℕ)
  (num_columns : ℕ)
  (central_column : ℕ)
  (people_positions : ℕ → ℕ → set ℕ)
  (h1 : num_people = 65)
  (h2 : num_columns = 17)
  (h3 : central_column = 8)
  (h4 : people_positions 0 central_column = set.univ)
  (h5 : ∀ t n, n ∈ people_positions t central_column → 
      n ∈ people_positions (t + 1) (central_column - 1) ∨ 
      n ∈ people_positions (t + 1) (central_column + 1)) :
  ∃ col, ∀ t, ∃ (s : set ℕ), s ⊆ people_positions t col ∧ s.count ≥ 8 :=
sorry

end at_least_one_column_with_8_people_l418_418594


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418920

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418920


namespace marbles_allocation_l418_418778

theorem marbles_allocation (total_marbles boys : ℕ) (h1 : total_marbles = 35) (h2 : boys = 5) : 
  (total_marbles / boys = 7) :=
by
  rw [h1, h2]
  norm_num

end marbles_allocation_l418_418778


namespace count_muffin_combinations_l418_418196

theorem count_muffin_combinations (kinds total : ℕ) (at_least_one: ℕ) :
  kinds = 4 ∧ total = 8 ∧ at_least_one = 1 → 
  let valid_combinations := 23 in
  valid_combinations = 
    (1 -- case 1: 4 different kinds out of 4 remaining muffins
    + 4 -- case 2: all 4 remaining muffins of the same kind
    + (4 * 3) -- case 3: 3 of one kind, 1 of another (4 choices for 3 + 3 choices for 1)
    + (4 * 3 / 2)) -- case 4: 2 of one kind and 2 of another (combinations)
  sorry

end count_muffin_combinations_l418_418196


namespace triangle_altitudes_inequality_l418_418451

theorem triangle_altitudes_inequality 
  {A B C D E : Type} [Real S]
  (hBD : is_altitude BD ABC)
  (hCE : is_altitude CE ABC)
  (hAB_geq_AC : AB >= AC)
  (area_S : ∀ {ABC : Type}, S = 1/2 * AC * BD ∧ S = 1/2  * AB * CE):
  AB + CE >= AC + BD : sorry

end triangle_altitudes_inequality_l418_418451


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418939

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418939


namespace lyssa_vs_precious_correct_answer_difference_l418_418877

/-- Definitions based on given conditions -/
def total_items : ℕ := 75
def lyssa_mistakes : ℕ := total_items * 20 / 100
def precious_mistakes : ℕ := 12

/-- Main theorem based on question and correct answer -/
theorem lyssa_vs_precious_correct_answer_difference :
  let lyssa_correct := total_items - lyssa_mistakes,
      precious_correct := total_items - precious_mistakes
  in lyssa_correct - precious_correct = 3 :=
by
  -- placeholder for proof
  sorry

end lyssa_vs_precious_correct_answer_difference_l418_418877


namespace cost_price_of_product_l418_418602

theorem cost_price_of_product (x y : ℝ)
  (h1 : 0.8 * y - x = 120)
  (h2 : 0.6 * y - x = -20) :
  x = 440 := sorry

end cost_price_of_product_l418_418602


namespace largestValidNumberIs84_l418_418898

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l418_418898


namespace parabola_standard_eq_hyperbola_standard_eq_l418_418277

section
variable {x y: ℝ}

-- Parabola Condition
def directrix_parabola (p: ℝ) := x = -p / 2

-- Hyperbola Conditions
def point_on_hyperbola (x y a b: ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def hyperbola_conditions (a b: ℝ) := 
  point_on_hyperbola 2 0 a b ∧
  point_on_hyperbola (2 * Real.sqrt 3) (Real.sqrt 6) a b

-- Correct Answers
def parabola_equation (p: ℝ) := y^2 = 2 * p * x
def hyperbola_equation (a b: ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

-- Proof Problems
theorem parabola_standard_eq {p : ℝ} (h: directrix_parabola 3) : parabola_equation 3 :=
sorry

theorem hyperbola_standard_eq (h: hyperbola_conditions 2 1.73205) : hyperbola_equation 2 (Real.sqrt 3) :=
sorry

end

end parabola_standard_eq_hyperbola_standard_eq_l418_418277


namespace total_num_birds_l418_418831

-- Definitions for conditions
def num_crows := 30
def percent_more_hawks := 0.60

-- Theorem to prove the total number of birds
theorem total_num_birds : num_crows + num_crows * percent_more_hawks + num_crows = 78 := 
sorry

end total_num_birds_l418_418831


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l418_418944

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l418_418944


namespace increasing_even_periodic_fn_l418_418241

-- Define the functions in the conditions
def tan_fn (x : ℝ) : ℝ := Real.tan x
def sin_fn (x : ℝ) : ℝ := Real.sin x
def cos_abs_fn (x : ℝ) : ℝ := abs (Real.cos (-x))
def sin_abs_fn (x : ℝ) : ℝ := abs (Real.sin (-x))

-- Proposition statement
noncomputable def trigonometric_function_property : Prop :=
  (∀ x : ℝ, (0 < x ∧ x < π / 2) → sin_abs_fn x > 0) ∧
  (∀ x : ℝ, (cos_abs_fn x) ≠ (cos_abs_fn (-x))) ∧
  (∀ x : ℝ, (tan_fn x) ≠ (tan_fn (-x))) ∧
  (∀ x : ℝ, (sin_fn x) ≠ (sin_fn (-x)))

-- Proof statement
theorem increasing_even_periodic_fn :
  trigonometric_function_property →
  (∀ x : ℝ, | -(sin x) | = | sin x | ∧ | sin x | = | sin x | ∧ (∀ x, abs (sin x) = abs (Real.sin (-x))) ∧ ∀ x : ℝ, 0 < x ∧ x < π / 2 → abs (Real.sin x) > abs (Real.sin 0)) :=
sorry

end increasing_even_periodic_fn_l418_418241


namespace friendship_count_l418_418403

noncomputable def friendshipConfigurations : Nat :=
  1008

theorem friendship_count :
  ∀ {A B C D E F G H : Type} (f : A → B → Prop)
    (h0 : (∃ a b c d e f g h, 
             ∀ x y, x ≠ y → (f x y ↔ y ∈ [a, b, c, d, e, f, g]))
             ∧ ∀ x y, x ≠ y → ¬f A B
             ∧ (∃! x, ∀ y, ¬f x y)
             ∧ ∀ x, ¬(∃ y, ∀ z, f z y)),

    A.functions (λ h1, let ⟨a0, b0, c0, d0, e0, f0, g0, h0, hf⟩ := h1 in
                        (hf ∧ Assume h1).bind_or_return.assume
        _sorry_proof___))
  → friendshipConfigurations = 1008
by
  sorry

end friendship_count_l418_418403


namespace min_number_students_l418_418777

-- Let's define the sets and the given counts
variables (A S P D : Set α) (students: Finset α) [Fintype α]

open Finset

noncomputable def number_students_in_class : ℕ :=
  card (students)

-- Given cardinalities of sets as conditions
axiom hA : card (A ∩ students) = 13
axiom hS : card (S ∩ students) = 11
axiom hP : card (P ∩ students) = 15
axiom hD : card (D ∩ students) = 6

-- Given conditions on the intersections
axiom h_plums_conditions : (S ∩ students) ⊆ ((A ∩ students) ∪ (P ∩ students))
axiom h_peaches_conditions : (P ∩ students) ⊆ ((S ∩ students) ∪ (D ∩ students))

-- The theorem stating the minimal number of students
theorem min_number_students : number_students_in_class = 22 :=
  sorry

end min_number_students_l418_418777


namespace largest_circle_radius_l418_418618

theorem largest_circle_radius 
  (h H : ℝ) (h_pos : h > 0) (H_pos : H > 0) :
  ∃ R, R = (h * H) / (h + H) :=
sorry

end largest_circle_radius_l418_418618


namespace geometric_sequence_sum_l418_418116

noncomputable def aₙ (n : ℕ) : ℝ := (2 / 3) ^ (n - 1)

noncomputable def Sₙ (n : ℕ) : ℝ := 3 * (1 - (2 / 3) ^ n)

theorem geometric_sequence_sum (n : ℕ) : Sₙ n = 3 - 2 * aₙ n := by
  sorry

end geometric_sequence_sum_l418_418116


namespace muffin_combinations_l418_418199

theorem muffin_combinations (k : ℕ) (n : ℕ) (h_k : k = 4) (h_n : n = 4) :
  (Nat.choose ((n + k - 1) : ℕ) ((k - 1) : ℕ)) = 35 :=
by
  rw [h_k, h_n]
  -- Simplifying Nat.choose (4 + 4 - 1) (4 - 1) = Nat.choose 7 3
  sorry

end muffin_combinations_l418_418199


namespace prime_sum_l418_418856

theorem prime_sum (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (h : 2 * p + 3 * q = 6 * r) : 
  p + q + r = 7 := 
sorry

end prime_sum_l418_418856


namespace simplify_fraction_l418_418072

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l418_418072


namespace pages_can_be_copied_l418_418827

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l418_418827


namespace total_trees_expression_total_trees_given_A_l418_418103

section
variables (x : ℝ) (total_trees : ℝ)

-- Define conditions
def num_trees_A (trees_B : ℝ) : ℝ := 1.2 * trees_B
def num_trees_C (trees_A : ℝ) : ℝ := trees_A - 2

-- Lean statement for the first question
theorem total_trees_expression (hx : x ≥ 0) :
  total_trees = x + num_trees_A x + num_trees_C (num_trees_A x) → 
  total_trees = 3.4 * x - 2 :=
by sorry

-- Lean statement for the second question with A planted 12 trees
def num_trees_B_given_A (trees_A : ℝ) : ℝ := trees_A / 1.2
def num_trees_C_given_A (trees_A : ℝ) : ℝ := trees_A - 2

theorem total_trees_given_A (hA : num_trees_A 10 = 12) (A_planted : ℝ) :
  A_planted = 12 →
  total_trees = A_planted + num_trees_B_given_A 12 + num_trees_C_given_A 12 → 
  total_trees = 32 :=
by sorry
end

end total_trees_expression_total_trees_given_A_l418_418103


namespace increasing_function_a_gt_2_5_g_ge_h_m_ge_8_minus_5_log_2_l418_418735

-- Define the functions g and h
def g (a x : ℝ) : ℝ := a * x - a / x - 5 * log x
def h (m x : ℝ) : ℝ := x^2 - m * x + 4

-- (I) Prove that if g(x) is increasing for x > 0, then a > 2.5
theorem increasing_function_a_gt_2_5 (a : ℝ) (h_inc : ∀ x > 0, (a * x - a / x - 5 * log x)' > 0) : a > 5 / 2 :=
  sorry

-- (II) Prove that if a = 2 and for x1 ∈ (0,1), x2 ∈ [1,2], g(x1) ≥ h(x2), then m >= 8 - 5 * log 2
theorem g_ge_h_m_ge_8_minus_5_log_2 (m : ℝ) (a : ℝ = 2) 
  (h_cond : ∀ x1 ∈ set.Ioo 0 1, ∀ x2 ∈ set.Icc 1 2, g 2 x1 ≥ h m x2) : m ≥ 8 - 5 * log 2 :=
  sorry

end increasing_function_a_gt_2_5_g_ge_h_m_ge_8_minus_5_log_2_l418_418735


namespace shopkeeper_loss_percent_l418_418175

noncomputable def loss_percentage (cost_price profit_percent theft_percent: ℝ) :=
  let selling_price := cost_price * (1 + profit_percent / 100)
  let value_lost := cost_price * (theft_percent / 100)
  let remaining_cost_price := cost_price * (1 - theft_percent / 100)
  (value_lost / remaining_cost_price) * 100

theorem shopkeeper_loss_percent
  (cost_price : ℝ)
  (profit_percent : ℝ := 10)
  (theft_percent : ℝ := 20)
  (expected_loss_percent : ℝ := 25)
  (h1 : profit_percent = 10) (h2 : theft_percent = 20) : 
  loss_percentage cost_price profit_percent theft_percent = expected_loss_percent := 
by
  sorry

end shopkeeper_loss_percent_l418_418175


namespace sqrt_meaningful_range_l418_418761

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x = Real.sqrt (a + 2)) ↔ a ≥ -2 := 
sorry

end sqrt_meaningful_range_l418_418761


namespace nora_muffin_price_l418_418886

theorem nora_muffin_price
  (cases : ℕ)
  (packs_per_case : ℕ)
  (muffins_per_pack : ℕ)
  (total_money : ℕ)
  (total_cases : ℕ)
  (h1 : total_money = 120)
  (h2 : packs_per_case = 3)
  (h3 : muffins_per_pack = 4)
  (h4 : total_cases = 5) :
  (total_money / (total_cases * packs_per_case * muffins_per_pack) = 2) :=
by
  sorry

end nora_muffin_price_l418_418886


namespace trig_expression_eq_zero_l418_418702

theorem trig_expression_eq_zero (α : ℝ) (h1 : Real.sin α = -2 / Real.sqrt 5) (h2 : Real.cos α = 1 / Real.sqrt 5) :
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 0 := by
  sorry

end trig_expression_eq_zero_l418_418702


namespace area_of_triangle_F1PF2_l418_418981

theorem area_of_triangle_F1PF2 (x y : ℝ) (a b c : ℝ)
  (ellipse_eq : x^2 / 16 + y^2 / 9 = 1)
  (f1 f2 : ℝ × ℝ)
  (P : ℝ × ℝ)
  (angle_F1PF2 : ∠ (ComplexArgument f1 P) (ComplexArgument P f2) = π / 6)
  (sum_dist_f1P_f2P : dist f1 P + dist P f2 = 8)
  (dist_f1f2 : dist f1 f2 = 2 * Real.sqrt 7) :
  (1 / 2) * (36 / (2 + Real.sqrt 3) * 1 / 2)  = 18 - 9 * Real.sqrt 3 := 
sorry

end area_of_triangle_F1PF2_l418_418981


namespace z12_real_cardinality_l418_418554

open Complex

noncomputable def num_real_z12_of_z48_eq_1 : ℕ :=
  let S : Set ℂ := { z | z ^ 48 = 1 }
  { z | z ^ 12 ∈ ℝ } . card

theorem z12_real_cardinality : num_real_z12_of_z48_eq_1 = 8 := by
  sorry

end z12_real_cardinality_l418_418554


namespace proper_subsets_count_l418_418985

theorem proper_subsets_count {α : Type} (s : set α) (hs : s = {1, 2, 3}) : 
  (fintype.card (set_of (λ t : set α, t ⊂ s))) = 7 :=
by
  sorry

end proper_subsets_count_l418_418985


namespace consecutive_product_plus_one_l418_418267

theorem consecutive_product_plus_one (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  sorry

end consecutive_product_plus_one_l418_418267


namespace decagon_diagonals_l418_418383

theorem decagon_diagonals : ∀ n : ℕ, n = 10 → (n * (n - 3) / 2) = 35 :=
by
  intros n hn
  rw [hn]
  norm_num
  sorry

end decagon_diagonals_l418_418383


namespace four_digit_number_l418_418609

theorem four_digit_number (x : ℕ) (hx : 100 ≤ x ∧ x < 1000) (unit_digit : ℕ) (hu : unit_digit = 2) :
    (10 * x + unit_digit) - (2000 + x) = 108 → 10 * x + unit_digit = 2342 :=
by
  intros h
  sorry


end four_digit_number_l418_418609


namespace max_constant_inequality_l418_418674

theorem max_constant_inequality (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha1 : a ≤ 1)
    (hb : 0 ≤ b) (hb1 : b ≤ 1)
    (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (hd : 0 ≤ d) (hd1 : d ≤ 1) 
    : a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3) :=
sorry

end max_constant_inequality_l418_418674


namespace find_m_l418_418340

def quadratic (m : ℝ) := λ x : ℝ, m*x^2 + 2*(m+1)*x + (m-1)

noncomputable def has_two_distinct_real_roots (m : ℝ) : Prop :=
  let discriminant := 4*(m+1)^2 - 4*m*(m-1) in
  discriminant > 0

noncomputable def roots_square_sum_eq_eight (m : ℝ) : Prop :=
  let sum_of_roots := -2*(m+1)/m
  let product_of_roots := (m-1)/m
  (sum_of_roots^2 - 2*product_of_roots = 8)

theorem find_m (m : ℝ) (h₁ : has_two_distinct_real_roots m) (h₂ : roots_square_sum_eq_eight m) : m = 2 :=
  sorry

end find_m_l418_418340


namespace pure_imaginary_condition_fourth_quadrant_condition_l418_418349

theorem pure_imaginary_condition (m : ℝ) (h1: m * (m - 1) = 0) (h2: m ≠ 1) : m = 0 :=
by
  sorry

theorem fourth_quadrant_condition (m : ℝ) (h3: m + 1 > 0) (h4: m^2 - 1 < 0) : -1 < m ∧ m < 1 :=
by
  sorry

end pure_imaginary_condition_fourth_quadrant_condition_l418_418349


namespace simplify_fraction_l418_418060

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l418_418060


namespace intersection_points_l418_418538

theorem intersection_points (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  (∃ x1 x2, 0 ≤ x1 ∧ x1 ≤ 2 * Real.pi ∧ 
   0 ≤ x2 ∧ x2 ≤ 2 * Real.pi ∧ 
   x1 ≠ x2 ∧ 
   1 + Real.sin x1 = 3 / 2 ∧ 
   1 + Real.sin x2 = 3 / 2 ) ∧ 
  (∀ x, (0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 1 + Real.sin x = 3 / 2) → 
   (x = x1 ∨ x = x2)) :=
sorry

end intersection_points_l418_418538


namespace pages_copied_l418_418813

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l418_418813


namespace HQ_tangent_to_circumcircle_FHK_l418_418446

noncomputable def orthocenter (A B C : Point) : Point := sorry -- calculating orthocenter
noncomputable def perpendicular_foot (X Y Z : Point) : Point := sorry -- foot of perpendicular from X to line YZ
noncomputable def circumcircle (A B C : Point) : Circle := sorry -- circumcircle of triangle
noncomputable def symmetric_point (P Q : Point) : Point := sorry -- symmetric point of P with respect to Q

variables {A B C H F K P Q : Point}

-- Given Conditions
axiom h1 : H = orthocenter A B C
axiom h2 : AB ≠ AC
axiom h3 : (F : circumcircle A B C) ∧ ∠AFH = 90 -- circumcircle condition and 90 degree angle
axiom h4 : K = symmetric_point H B
axiom h5 : ∠PHB = 90 ∧ ∠PBC = 90
axiom h6 : Q = perpendicular_foot B C P

-- Question to Prove
theorem HQ_tangent_to_circumcircle_FHK : tangent_to (segment H Q) (circumcircle F H K) :=
sorry

end HQ_tangent_to_circumcircle_FHK_l418_418446


namespace find_n_l418_418364

variable
  (S : ℕ → ℚ)   -- Sum of the first n terms of the sequence
  (a : ℕ → ℚ)   -- The nth term of the sequence

-- Conditions
def cond1 (n : ℕ) : Prop := S n = 18
def cond2 : Prop := S 3 = 1
def cond3 (n : ℕ) : Prop := a n + a (n-1) + a (n-2) = 3

-- The statement we want to prove
theorem find_n (n : ℕ) (h1 : cond1 n) (h2 : cond2) (h3 : cond3 n) : n = 27 :=
by
  sorry

end find_n_l418_418364


namespace minimum_value_of_m_l418_418040

open Real

-- Define the functions f and g.
noncomputable def f (x : ℝ) : ℝ := exp x - exp (-x)
noncomputable def g (m x : ℝ) : ℕ := log (m * x ^ 2 - x + 1 / 4)

-- Define the problem statement.
theorem minimum_value_of_m (m : ℝ) :
  (∀ x1 ∈ Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g m x2) → -1 / 3 ≤ m ∧ m ≤ 0 :=
sorry -- Proof placeholder.

end minimum_value_of_m_l418_418040


namespace opposite_of_one_l418_418387

theorem opposite_of_one (a : ℤ) (h : a = -1) : a = -1 := 
by 
  exact h

end opposite_of_one_l418_418387


namespace simplify_fraction_l418_418066

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l418_418066


namespace solve_for_x_l418_418389

theorem solve_for_x (x : ℝ) (h : (2 / 3 - 1 / 4) = 4 / x) : x = 48 / 5 :=
by sorry

end solve_for_x_l418_418389


namespace circumradius_of_A_l418_418626

open EuclideanGeometry

variables {A B C A' B' C' : Point}
variable {r : ℝ}

-- Conditions
variable (ABC_inradius : ∀ {ABC : Triangle}, inradius ABC = r)
variable (A_circle_orthogonal : orthogonal_circle_through A C)
variable (B_circle_orthogonal : orthogonal_circle_through A B)
variable (C_def : ∀ {A B C : Point}, ∃ A', circle A C ∩ circle A B = {A, A'})
variable (B'_def : ∀ {A B C : Point}, ∃ B', circle B A ∩ circle B C = {B, B'})
variable (C'_def : ∀ {A B C : Point}, ∃ C', circle C A ∩ circle C B = {C, C'})

-- Goal
theorem circumradius_of_A'B'C' (h1 : ABC_inradius) (h2 : A_circle_orthogonal)
  (h3 : B_circle_orthogonal) (h4 : C_def) (h5 : B'_def) (h6 : C'_def): circumradius (triangle A' B' C') = r / 2 := by
  sorry

end circumradius_of_A_l418_418626


namespace shifted_function_is_correct_l418_418398

-- Given conditions
def original_function (x : ℝ) : ℝ := -(x + 2) ^ 2 + 1

def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Resulting function after shifting 1 unit to the right
def shifted_function : ℝ → ℝ := shift_right original_function 1

-- Correct answer
def correct_function (x : ℝ) : ℝ := -(x + 1) ^ 2 + 1

-- Proof Statement
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = correct_function x := by
  sorry

end shifted_function_is_correct_l418_418398


namespace range_of_c_l418_418464

theorem range_of_c (x y c : ℝ) (h1 : x^2 + (y - 2)^2 = 1) (h2 : x^2 + y^2 + c ≤ 0) : c ≤ -9 :=
by
  -- Proof goes here
  sorry

end range_of_c_l418_418464


namespace sum_of_ab_l418_418686

theorem sum_of_ab (a b : ℕ) (h1 : a^2 + sqrt (2017 - b^2) ∈ ℕ) : 
  ∑ (a, b) in {(4, 44), (10, 9)}, (a + b) = 67 :=
by
  sorry

end sum_of_ab_l418_418686


namespace a6_is_32_l418_418708

namespace arithmetic_sequence

variables {a : ℕ → ℝ} -- {aₙ} is an arithmetic sequence with positive terms
variables (q : ℝ) -- Common ratio

-- Conditions as definitions
def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a1_is_1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2_times_a4_is_16 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 4 = 16

-- The ultimate goal is to prove a₆ = 32
theorem a6_is_32 (h_arith : is_arithmetic_sequence a q) 
  (h_a1 : a1_is_1 a) (h_product : a2_times_a4_is_16 a q) : 
  a 6 = 32 := 
sorry

end arithmetic_sequence

end a6_is_32_l418_418708


namespace prime_squares_mod_180_l418_418255

theorem prime_squares_mod_180 (p : ℕ) (hp : prime p) (hp_gt_5 : p > 5) :
  ∃ (r1 r2 : ℕ), 
  r1 ≠ r2 ∧ 
  ∀ r : ℕ, (∃ m : ℕ, p^2 = m * 180 + r) → (r = r1 ∨ r = r2) :=
sorry

end prime_squares_mod_180_l418_418255


namespace copy_pages_15_dollars_l418_418824

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l418_418824


namespace abs_of_neg_square_add_l418_418951

theorem abs_of_neg_square_add (a b : ℤ) : |-a^2 + b| = 10 :=
by
  sorry

end abs_of_neg_square_add_l418_418951


namespace number_of_correct_propositions_l418_418361

variable {α : Type*} [linear_ordered_field α] {a b M m p : α} (f : α → α)

theorem number_of_correct_propositions 
  (h_max : ∀ x ∈ set.Icc a b, f x ≤ M)
  (h_min : ∀ x ∈ set.Icc a b, m ≤ f x) : 
  (∃ p : α, 
    (∀ x ∈ set.Icc a b, p ≤ f x) ↔ p ≤ m ∧ 
    (∀ x ∈ set.Icc a b, p ≤ f x) ↔ p ≠ m ↔ 
    (∃ x ∈ set.Icc a b, f x = p) ↔ p ∈ set.Icc m M ∧ 
    (∃ x ∈ set.Icc a b, p ≤ f x) ↔ p ≤ m ↔ 
    (∃ x ∈ set.Icc a b, p ≤ f x) ↔ p ≤ M) = 3 :=
sorry

end number_of_correct_propositions_l418_418361


namespace simplify_fraction_l418_418067

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l418_418067


namespace total_operations_l418_418100

-- Define the process of iterative multiplication and division as described in the problem
def process (start : Nat) : Nat :=
  let m1 := 3 * start
  let m2 := 3 * m1
  let m3 := 3 * m2
  let m4 := 3 * m3
  let m5 := 3 * m4
  let d1 := m5 / 2
  let d2 := d1 / 2
  let d3 := d2 / 2
  let d4 := d3 / 2
  let d5 := d4 / 2
  let d6 := d5 / 2
  let d7 := d6 / 2
  d7

theorem total_operations : process 1 = 1 ∧ 5 + 7 = 12 :=
by
  sorry

end total_operations_l418_418100


namespace floor_decoration_angle_l418_418603

theorem floor_decoration_angle
  (N : ℕ) 
  (hN : N = 10)
  (central_angle : ℝ)
  (hcentral_angle : central_angle = 360 / N) 
  (east_angle : ℝ)
  (heast_angle : east_angle = 90)
  (south_angle : ℝ)
  (hsouth_angle : south_angle = 180) :
  let smaller_angle := ((south_angle - east_angle) / central_angle) * central_angle
  in  smaller_angle = 54 :=
by 
  have hcentral : central_angle = 36 := hcentral_angle
  have heast_half_ray_angle : east_angle = 2.5 * central_angle :=
    by norm_num [heast_angle, hcentral]
  have hsouth_ray_angle : south_angle = 5 * central_angle := 
    by norm_num [hsouth_angle, hcentral]
  have span_rays : south_angle - east_angle = 1.5 * central_angle := 
    by norm_num [heast_half_ray_angle, hsouth_ray_angle]
  have smaller_angle_def := (south_angle - east_angle) / central_angle * central_angle
  exact norm_num [span_rays]

end floor_decoration_angle_l418_418603


namespace pebbles_problem_l418_418559

theorem pebbles_problem :
  ∀ (initial_piles : List ℕ),
    initial_piles = [5, 49, 51] →
    (∀ piles, True) →
    ¬ (∃ piles, piles.length = 105 ∧ ∀ pile ∈ piles, pile = 1) :=
by
  intros initial_piles h1 h2
  sorry

end pebbles_problem_l418_418559


namespace lyssa_vs_precious_correct_answer_difference_l418_418878

/-- Definitions based on given conditions -/
def total_items : ℕ := 75
def lyssa_mistakes : ℕ := total_items * 20 / 100
def precious_mistakes : ℕ := 12

/-- Main theorem based on question and correct answer -/
theorem lyssa_vs_precious_correct_answer_difference :
  let lyssa_correct := total_items - lyssa_mistakes,
      precious_correct := total_items - precious_mistakes
  in lyssa_correct - precious_correct = 3 :=
by
  -- placeholder for proof
  sorry

end lyssa_vs_precious_correct_answer_difference_l418_418878


namespace number_of_ways_to_place_pawns_l418_418392

theorem number_of_ways_to_place_pawns : 
  let board := (Fin 5 → Fin 5)
  let pawns := Finset.univ : Finset (Fin 5)
  (∃ f : Fin 5 → Fin 5, Function.Injective f ∧ Finset.card (Finset.image f pawns) = Finset.card pawns) → 
  (Finset.product pawns pawns).card = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l418_418392


namespace pages_copied_l418_418811

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l418_418811


namespace unique_g_function_l418_418846

def T := { x : ℝ // 0 < x ∧ x ≠ 1 }

def g (x : T) : ℝ := sorry -- This will be defined in proof.

theorem unique_g_function :
  (∀ (x y : ℝ), g ⟨(1 / real.exp (x + y)), sorry⟩ = g ⟨(1 / real.exp x), sorry⟩ + g ⟨(1 / real.exp y), sorry⟩) ∧
  (∀ (x y : ℝ), (real.exp (x + y) - 1) * g ⟨(real.exp (x + y)), sorry⟩ =
   (real.exp x - 1) * (real.exp y - 1) * g ⟨(real.exp x), sorry⟩ * g ⟨(real.exp y), sorry⟩) ∧
  (g ⟨real.exp 1, sorry⟩ = 1) →
  ∃! f : T → ℝ, ∀ x, f x = 1 / x.val :=
begin
  sorry
end

end unique_g_function_l418_418846


namespace john_can_fix_l418_418019

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end john_can_fix_l418_418019


namespace combined_yellow_blue_correct_l418_418474

-- Declare the number of students in the class
def total_students : ℕ := 200

-- Declare the percentage of students who like blue
def percent_like_blue : ℝ := 0.3

-- Declare the percentage of remaining students who like red
def percent_like_red : ℝ := 0.4

-- Function that calculates the number of students liking a certain color based on percentage
def students_like_color (total : ℕ) (percent : ℝ) : ℕ :=
  (percent * total).toInt

-- Calculate the number of students who like blue
def students_like_blue : ℕ :=
  students_like_color total_students percent_like_blue

-- Calculate the number of students who don't like blue
def students_not_like_blue : ℕ :=
  total_students - students_like_blue

-- Calculate the number of students who like red from those who don't like blue
def students_like_red : ℕ :=
  students_like_color students_not_like_blue percent_like_red

-- Calculate the number of students who like yellow (those who don't like blue or red)
def students_like_yellow : ℕ :=
  students_not_like_blue - students_like_red

-- The combined number of students who like yellow and blue
def combined_yellow_blue : ℕ :=
  students_like_blue + students_like_yellow

-- Theorem to prove that the combined number of students liking yellow and blue is 144
theorem combined_yellow_blue_correct : combined_yellow_blue = 144 := by
  sorry

end combined_yellow_blue_correct_l418_418474


namespace largest_in_column_smallest_in_row_l418_418582

theorem largest_in_column_smallest_in_row :
  ∃ n, n = 1 ∧ 
    (∀ i, n ≥ (fun (a : Array (Array Int)) => a[3][i]) ⟨array![
       array![5, -2, 3, 7],
       array![8, 0, 2, -1],
       array![1, -3, 6, 0],
       array![9, 1, 4, 2]
     ]⟩) ∧ 
    (∀ j, n ≤ (fun (a : Array (Array Int)) => a[j][1]) ⟨array![
       array![5, -2, 3, 7],
       array![8, 0, 2, -1],
       array![1, -3, 6, 0],
       array![9, 1, 4, 2]
     ]⟩)
:= sorry

end largest_in_column_smallest_in_row_l418_418582


namespace probability_b_gt_a_l418_418895

theorem probability_b_gt_a : 
  (1:ℝ) / 5 = 
    let outcomes := (finset.product (finset.range 5) (finset.range 3))
    let favorable := (finset.filter (λ (x : ℕ × ℕ), x.snd > x.fst) outcomes)
    (favorable.card : ℝ) / (outcomes.card : ℝ) :=
begin
  let outcomes := (finset.product (finset.range 5) (finset.range 3)),
  let favorable := (finset.filter (λ (x : ℕ × ℕ), x.snd > x.fst) outcomes),
  have h1 : outcomes.card = 15, from sorry,
  have h2 : favorable.card = 3, from sorry,
  have h3 : (3 : ℝ) / 15 = (1 : ℝ) / 5, from sorry,
  exact h3,
end

end probability_b_gt_a_l418_418895


namespace non_seniors_play_music_l418_418407

theorem non_seniors_play_music:
    (s n : ℕ) 
    -- Conditions
    (total_students : s + n = 800)
    (seniors_play_instrument : 0.6 * s)
    (non_seniors_not_play_instrument : 0.25 * n)
    (total_not_play_instrument : 0.55 * 800 = 440)
    : (0.75 * n = 600) :=
by
  -- Sorry is used here as proof is not required
  sorry

end non_seniors_play_music_l418_418407


namespace no_fractional_linear_function_l418_418893

noncomputable def fractional_linear_function (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem no_fractional_linear_function (a b c d : ℝ) :
  ∀ x : ℝ, c ≠ 0 → 
  (fractional_linear_function a b c d x + fractional_linear_function b (-d) c (-a) x ≠ -2) :=
by
  sorry

end no_fractional_linear_function_l418_418893


namespace hyperbola_standard_eq1_hyperbola_standard_eq2_l418_418684

-- Define points and conditions
def P := (3 : ℝ, 15 / 4 : ℝ)
def Q := (-16 / 3 : ℝ, 5 : ℝ)
def c := sqrt 6
def additional_point := (-5 : ℝ, 2 : ℝ)

-- First standard equation
theorem hyperbola_standard_eq1 :
  (∃ m n : ℝ, m * n < 0 ∧
    9 * m + (225 / 16) * n = 1 ∧
    (256 / 9) * m + 25 * n = 1) →
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧
    (a * P.1^2 + b * P.2^2 = 1) ∧
    (a * Q.1^2 + b * Q.2^2 = 1) ∧
    x^2 / a - y^2 / b = 1 :=
begin
  sorry
end

-- Second standard equation focusing on foci and additional point
theorem hyperbola_standard_eq2 :
  (∃ λ : ℝ, 0 < λ ∧ λ < 6 ∧
    25 / λ - 4 / (6 - λ) = 1) →
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧
    (a * additional_point.1^2 - b * additional_point.2^2 = 1) ∧
    x^2 / a - y^2 = 1 :=
begin
  sorry
end

end hyperbola_standard_eq1_hyperbola_standard_eq2_l418_418684


namespace alternating_sum_l418_418299

theorem alternating_sum (a b : ℕ) (h1 : a = 1) (h2 : b = 10000) :
  let triangular := λ n : ℕ, n * (n + 1) / 2
  let change_sign_points := {n | ∃ k, n = triangular k}
  (∑ n in {1..b}, (-1) ^ (if (∑ k in change_sign_points.filter (λ k, k ≤ n), 1) % 2 = 0 then 1 else 0) * n) = 1722700 :=
by
  sorry

end alternating_sum_l418_418299


namespace find_chemistry_marks_l418_418274

theorem find_chemistry_marks :
  (let E := 96
   let M := 95
   let P := 82
   let B := 95
   let avg := 93
   let n := 5
   let Total := avg * n
   let Chemistry_marks := Total - (E + M + P + B)
   Chemistry_marks = 97) :=
by
  let E := 96
  let M := 95
  let P := 82
  let B := 95
  let avg := 93
  let n := 5
  let Total := avg * n
  have h_total : Total = 465 := by norm_num
  let Chemistry_marks := Total - (E + M + P + B)
  have h_chemistry_marks : Chemistry_marks = 97 := by norm_num
  exact h_chemistry_marks

end find_chemistry_marks_l418_418274


namespace guarantee_win_first_turn_take_1_l418_418995

-- Definitions from conditions
def initial_pieces : ℕ := 100
def can_take (n : ℕ) : Prop := n = 1 ∨ n = 2
def takes_turn (first_turn : bool) (num_pieces : ℕ) : ℕ :=
  if first_turn then num_pieces - 1 else num_pieces - 2

-- The main theorem to be proven
theorem guarantee_win_first_turn_take_1 :
  (∀ (num_pieces : ℕ), num_pieces ≥ 0 → (can_take 1 ∨ can_take 2) → ∃ (moves : list ℕ), last moves = 1 ∧
    (∀ (turn : ℕ), turn < num_pieces → turn % 3 = 0 → takes_turn true initial_pieces - turn ∈ moves)) :=
sorry

end guarantee_win_first_turn_take_1_l418_418995


namespace largest_integer_le_zero_of_f_l418_418186

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero_of_f :
  ∃ x₀ : ℝ, (f x₀ = 0) ∧ 2 ≤ x₀ ∧ x₀ < 3 ∧ (∀ k : ℤ, k ≤ x₀ → k = 2 ∨ k < 2) :=
by
  sorry

end largest_integer_le_zero_of_f_l418_418186


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l418_418911

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l418_418911


namespace sequence_nat_nums_exists_l418_418309

theorem sequence_nat_nums_exists (n : ℕ) : { k : ℕ | ∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i, 1 ≤ i → i ≤ n → x i)} = { k | 1 ≤ k ∧ k ≤ n } :=
sorry

end sequence_nat_nums_exists_l418_418309


namespace rain_total_duration_l418_418004

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end rain_total_duration_l418_418004


namespace part1_part2_l418_418728

noncomputable def f (x a : ℝ) : ℝ := real.exp x - a * x^2

theorem part1 (h : ∃ x ∈ set.Ioi (0 : ℝ), continuous_on (λ x, real.exp x - a * x^2) (set.Ioi 0) ∧ (∀ y ∈ set.Ioi (0 : ℝ), real.exp y - a * y^2 ≥ real.exp x - a * x^2) ∧ real.exp x - a * x^2 = 0) : a = real.exp 2 / 4 :=
sorry

theorem part2 (h : ∃ x1 x2 ∈ set.Ioi (0 : ℝ), real.exp x1 - a * x1^2 = 0 ∧ real.exp x2 - a * x2^2 = 0) : x1 + x2 > 4 :=
sorry

end part1_part2_l418_418728


namespace no_18_consecutive_good_numbers_l418_418874

def is_good (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p * q

theorem no_18_consecutive_good_numbers :
  ¬ ∃ a : ℕ, ∀ k : ℕ, k < 18 → is_good (a + k) :=
by
  sorry

end no_18_consecutive_good_numbers_l418_418874


namespace age_difference_is_18_l418_418123

def difference_in_ages (X Y Z : ℕ) : ℕ := (X + Y) - (Y + Z)
def younger_by_eighteen (X Z : ℕ) : Prop := Z = X - 18

theorem age_difference_is_18 (X Y Z : ℕ) (h : younger_by_eighteen X Z) : difference_in_ages X Y Z = 18 := by
  sorry

end age_difference_is_18_l418_418123


namespace triangle_shape_l418_418774

variables {α β γ : ℝ}

theorem triangle_shape (h1 : ∃ γ, α + β + γ = 180 ∧ 0 < α ∧ α < 180 ∧ 0 < β ∧ β < 180 ∧ 0 < γ ∧ γ < 180) 
        (h2 : sin α = cos β) : α + β = 90 ∧ γ = 90 :=
by
  sorry

end triangle_shape_l418_418774


namespace frank_initial_mushrooms_l418_418324

theorem frank_initial_mushrooms (pounds_eaten pounds_left initial_pounds : ℕ) 
  (h1 : pounds_eaten = 8) 
  (h2 : pounds_left = 7) 
  (h3 : initial_pounds = pounds_eaten + pounds_left) : 
  initial_pounds = 15 := 
by 
  sorry

end frank_initial_mushrooms_l418_418324


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l418_418946

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l418_418946


namespace josie_gift_money_l418_418023

-- Define the cost of each cassette tape
def tape_cost : ℕ := 9

-- Define the number of cassette tapes Josie plans to buy
def num_tapes : ℕ := 2

-- Define the cost of the headphone set
def headphone_cost : ℕ := 25

-- Define the amount of money Josie will have left after the purchases
def money_left : ℕ := 7

-- Define the total cost of tapes
def total_tape_cost := num_tapes * tape_cost

-- Define the total cost of both tapes and headphone set
def total_cost := total_tape_cost + headphone_cost

-- The total money Josie will have would be total_cost + money_left
theorem josie_gift_money : total_cost + money_left = 50 :=
by
  -- Proof will be provided here
  sorry

end josie_gift_money_l418_418023


namespace jordyn_total_payment_l418_418766

theorem jordyn_total_payment :
  let price_cherries := 5
  let price_olives := 7
  let price_grapes := 11
  let num_cherries := 50
  let num_olives := 75
  let num_grapes := 25
  let discount_cherries := 0.12
  let discount_olives := 0.08
  let discount_grapes := 0.15
  let sales_tax := 0.05
  let service_charge := 0.02
  let total_cherries := num_cherries * price_cherries
  let total_olives := num_olives * price_olives
  let total_grapes := num_grapes * price_grapes
  let discounted_cherries := total_cherries * (1 - discount_cherries)
  let discounted_olives := total_olives * (1 - discount_olives)
  let discounted_grapes := total_grapes * (1 - discount_grapes)
  let subtotal := discounted_cherries + discounted_olives + discounted_grapes
  let taxed_amount := subtotal * (1 + sales_tax)
  let final_amount := taxed_amount * (1 + service_charge)
  final_amount = 1002.32 :=
by
  sorry

end jordyn_total_payment_l418_418766


namespace at_least_one_not_less_than_two_l418_418454

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) :=
sorry

end at_least_one_not_less_than_two_l418_418454


namespace gcd_420_135_l418_418573

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end gcd_420_135_l418_418573


namespace compare_abc_l418_418333

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := (1 / 3 : ℝ) ^ (1 / 3 : ℝ)
noncomputable def c : ℝ := (3 : ℝ) ^ (-1 / 4 : ℝ)

theorem compare_abc : b < c ∧ c < a :=
by
  sorry

end compare_abc_l418_418333


namespace device_records_720_instances_in_one_hour_l418_418434

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l418_418434


namespace root_condition_l418_418366

variables {n : ℕ} (a : Fin n → ℝ)

noncomputable
def polynomial_eqn (x : ℂ) : ℂ :=
  x ^ n + ∑ i in Finset.range n, (a i) * x ^ (n - i - 1)

theorem root_condition (λ : ℂ) (h_coeffs : ∀ i, 0 < a i ∧ a i ≤ 1)
  (h_sorted : ∀ i j, i ≤ j → a i ≤ a j)
  (h_root : polynomial_eqn a λ = 0)
  (h_magnitude : |λ| ≥ 1) :
  λ^(n + 1) = 1 :=
sorry

end root_condition_l418_418366


namespace integral_eval_l418_418647

open scoped Real

theorem integral_eval :
  ∫ (x : ℝ) in filter.at_top, (x * cos x + sin x) / (x * sin x)^2 = -(1/(x * sin x)) + C :=
by
  sorry

end integral_eval_l418_418647


namespace find_ellipse_equation_l418_418347

noncomputable def ellipse_standard_equation (a b c : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a > b ∧ 2 * c = 4 ∧ c^2 = a^2 - b^2 ∧ 
  (b^2 / a^2 = 1 / 3)) →
  (∀ x y : ℝ, (x^2 / 6 + y^2 / 2 = 1) ↔ (C : x^2 / a^2 + y^2 / b^2 = 1))

theorem find_ellipse_equation (a b : ℝ) (C : ℝ → ℝ → Prop) :
  (a > 0 ∧ b > 0 ∧ a > b ∧ (4 : ℝ) = a^2 - b^2 ∧ (b^2 / a^2 = 1 / 3)) →
  (C x y ↔ (x^2 / 6 + y^2 / 2 = 1)) :=
begin
  intros h,
  sorry
end

end find_ellipse_equation_l418_418347


namespace _l418_418444

open Real    -- we need Real to define sqrt and other real number operations

-- Defining the square ABCD with points A, B, C, and D
structure Square where
  A B C D : Point
  side_length : ℝ
  square : (side_length = 1) ∧ (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A)

-- Defining the conditions of the problem
def condition (s : Square) := 
  (s.side_length = 1) -- side length is 1

-- Defining the variable point P on line CD
def P_cd (s : Square) := 
  ∃ (P : Point), on_line_segment P s.C s.D

-- Defining the point Q as the intersection of the angle bisector of ∠APB with AB
def Q (s : Square) (P : Point) := 
  ∃ (Q : Point), angle_bisector (∠(s.A s.P s.B)) (intersection_with s.A s.B)

noncomputable def main_theorem (s : Square) (P : Point) :=
  condition s →
  P_cd s →
  Q s P →
  length_segment Q = 3 - 2 * sqrt 2

#eval show_main_theorem

end _l418_418444


namespace polynomial_sum_l418_418034

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - 5 * x - 7
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 2 * x + 8

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -5 * x^3 + 11 * x^2 + x - 8 :=
  sorry

end polynomial_sum_l418_418034


namespace copy_pages_l418_418817

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l418_418817


namespace correct_calculation_l418_418168

theorem correct_calculation (x y : ℝ) : 
  (sqrt (x^2) ≠ -3) ∧ 
  (sqrt 16 ≠ 4) ∧ 
  (¬ (sqrt (-25)).isReal) ∧ 
  (cbrt 8 = 2) :=
by
  sorry

end correct_calculation_l418_418168


namespace regression_equation_l418_418580

-- Define the conditions specified in the problem.
variable (x : ℝ) (y : ℝ)
variable (k a : ℝ)
variable (hat_z : ℝ -> ℝ)

-- Assumptions based on the problem conditions.
def exponential_model (x : ℝ) (k a : ℝ) := exp (k * x + a)
def z_definition (y : ℝ) := Real.log y
def regression_line (x : ℝ) := 0.25 * x - 2.58

-- The final statement we need to prove.
theorem regression_equation (x y : ℝ) (k a : ℝ)
  (h1 : y = exponential_model x k a)
  (h2 : z_definition y = k * x + a)
  (hat_z : ℝ -> ℝ)
  (h3 : hat_z x = regression_line x) :
  y = exp (0.25 * x - 2.58) := sorry

end regression_equation_l418_418580


namespace complement_A_inter_B_l418_418465

def A : Set ℝ := {x | abs (x - 2) ≤ 2}

def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

def A_inter_B : Set ℝ := A ∩ B

def C_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

theorem complement_A_inter_B :
  C_R A_inter_B = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end complement_A_inter_B_l418_418465


namespace lines_coincide_l418_418406

-- definition of the problem conditions
def scalene_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = 180 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def line1 (A C : ℝ) (x y : ℝ) : ℝ := 
  x + (Real.sin A + Real.sin C) / Real.sqrt 3 * y + 1

def line2 (A C : ℝ) (x y : ℝ) : ℝ :=
  x * Real.tan (60 - C) + y * (Real.sin A - Real.sin C) - Real.tan ((C - A) / 2)

-- the theorem to prove
theorem lines_coincide (A B C : ℝ) (a b c : ℝ) :
  scalene_triangle A B C a b c →
  B = 60 →
  (∀ x y : ℝ, line1 A C x y = 0 → line2 A C x y = 0) :=
by
  intros h_triangle h_B
  sorry

end lines_coincide_l418_418406


namespace kathryn_more_pints_than_annie_l418_418667

-- Definitions for conditions
def annie_pints : ℕ := 8
def ben_pints (kathryn_pints : ℕ) : ℕ := kathryn_pints - 3
def total_pints (annie_pints kathryn_pints ben_pints : ℕ) : ℕ := annie_pints + kathryn_pints + ben_pints

-- The problem statement
theorem kathryn_more_pints_than_annie (k : ℕ) (h1 : total_pints annie_pints k (ben_pints k) = 25) : k - annie_pints = 2 :=
sorry

end kathryn_more_pints_than_annie_l418_418667


namespace positive_time_difference_l418_418999

-- Define the rates at which Tom and Linda are moving
def linda_rate : ℝ := 4  -- miles per hour
def tom_rate : ℝ := 9    -- miles per hour

-- Define the distance Linda covers in the first hour
def linda_distance_first_hour : ℝ := linda_rate * 1

-- Define the times it takes Tom to cover half and twice of Linda's distance in the first hour
def tom_time_half_linda : ℝ := (1/2 * linda_distance_first_hour) / tom_rate
def tom_time_twice_linda : ℝ := (2 * linda_distance_first_hour) / tom_rate

-- Convert these times to minutes
def tom_time_half_linda_minutes : ℝ := tom_time_half_linda * 60
def tom_time_twice_linda_minutes : ℝ := tom_time_twice_linda * 60

-- Define the positive difference in minutes
def time_difference_minutes : ℝ := abs (tom_time_twice_linda_minutes - tom_time_half_linda_minutes)

-- The theorem stating the desired result
theorem positive_time_difference : time_difference_minutes = 40 := by
  sorry

end positive_time_difference_l418_418999


namespace convert_50_to_base_3_l418_418661

-- Define a function to convert decimal to ternary (base-3)
def convert_to_ternary (n : ℕ) : ℕ := sorry

-- Main theorem statement
theorem convert_50_to_base_3 : convert_to_ternary 50 = 1212 :=
sorry

end convert_50_to_base_3_l418_418661


namespace mallory_total_expense_l418_418114

theorem mallory_total_expense
  (cost_per_refill : ℕ)
  (distance_per_refill : ℕ)
  (total_distance : ℕ)
  (food_ratio : ℚ)
  (refill_count : ℕ)
  (total_fuel_cost : ℕ)
  (total_food_cost : ℕ)
  (total_expense : ℕ)
  (h1 : cost_per_refill = 45)
  (h2 : distance_per_refill = 500)
  (h3 : total_distance = 2000)
  (h4 : food_ratio = 3 / 5)
  (h5 : refill_count = total_distance / distance_per_refill)
  (h6 : total_fuel_cost = refill_count * cost_per_refill)
  (h7 : total_food_cost = (food_ratio * ↑total_fuel_cost).to_nat)
  (h8 : total_expense = total_fuel_cost + total_food_cost) :
  total_expense = 288 := by
  sorry

end mallory_total_expense_l418_418114


namespace stooge_sort_alpha_243_l418_418526

-- Given conditions
def is_runtime_stooge_sort (f : ℕ → ℕ) (α : ℝ) :=
  ∃ C n₀, ∀ n ≥ n₀, f(n) ≤ C * n^α
-- Recurrence relation of Stooge sort
def stooge_sort_recurrence (f : ℕ → ℕ) :=
  ∀ n, f(n) = 3 * f(⌈2 * n / 3⌉)

-- Definition of α
noncomputable def α := (Real.log 3 / Real.log (3 / 2))

-- Problem Statement
theorem stooge_sort_alpha_243 (f : ℕ → ℕ) (h₁ : is_runtime_stooge_sort f α) (h₂ : stooge_sort_recurrence f) :
  (243 / 32 : ℝ) ^ α = 243 := by
  sorry

end stooge_sort_alpha_243_l418_418526


namespace equilateral_triangles_similar_l418_418166

theorem equilateral_triangles_similar :
  ∀ (T₁ T₂ : Triangle), (T₁.equilateral ∧ T₂.equilateral) → (Triangle.similar T₁ T₂) :=
by
  -- T₁ and T₂ are equilateral triangles
  intelligent sorry_here sorry

-- Definition of a triangle
structure Triangle :=
  (angle1 angle2 angle3 : ℕ)
  (side1 side2 side3 : ℕ)

-- An equilateral triangle is defined as having all angles of 60 degrees
def Triangle.equilateral (T : Triangle) : Prop :=
  T.angle1 = 60 ∧ T.angle2 = 60 ∧ T.angle3 = 60 ∧ T.side1 = T.side2 ∧ T.side2 = T.side3 ∧ T.side1 = T.side3

-- Definition for triangle similarity based on Angle-Angle (AA) criterion
def Triangle.similar (T₁ T₂ : Triangle) : Prop :=
  (T₁.angle1 = T₂.angle1 ∧ T₁.angle2 = T₂.angle2) ∨
  (T₁.angle1 = T₂.angle1 ∧ T₁.angle3 = T₂.angle3) ∨
  (T₁.angle2 = T₁.angle2 ∧ T₁.angle3 = T₁.angle3)

end equilateral_triangles_similar_l418_418166


namespace gcd_48_72_120_l418_418289

theorem gcd_48_72_120 : Nat.gcd (Nat.gcd 48 72) 120 = 24 :=
by
  sorry

end gcd_48_72_120_l418_418289


namespace smaller_angle_measure_l418_418143

theorem smaller_angle_measure (x : ℝ) (a b : ℝ) (h_suppl : a + b = 180) (h_ratio : a = 4 * x ∧ b = x) :
  b = 36 :=
by
  sorry

end smaller_angle_measure_l418_418143


namespace quadrilateral_area_l418_418287

theorem quadrilateral_area 
  (d : ℝ) (h₁ h₂ : ℝ) 
  (hd : d = 22) 
  (hh₁ : h₁ = 9) 
  (hh₂ : h₂ = 6) : 
  (1/2 * d * h₁ + 1/2 * d * h₂ = 165) :=
by
  sorry

end quadrilateral_area_l418_418287


namespace correct_graph_is_E_l418_418401

noncomputable def at_home_workforce_percent (year : ℕ) : ℝ :=
  if year = 1960 then 0.05 else
  if year = 1970 then 0.08 else
  if year = 1980 then 0.15 else
  if year = 1990 then 0.30 else 0

theorem correct_graph_is_E :
  ∀ {a b c d e : ℝ}, 
    (a = at_home_workforce_percent 1960 ∧ 
     b = at_home_workforce_percent 1970 ∧ 
     c = at_home_workforce_percent 1980 ∧ 
     d = at_home_workforce_percent 1990) → 
    graph_E_best_represents a b c d e :=
by
  intro a b c d e
  intro h
  sorry

end correct_graph_is_E_l418_418401


namespace exists_sequence_for_k_l418_418310

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end exists_sequence_for_k_l418_418310


namespace find_ab_pairs_l418_418671

open Set

-- Definitions
def f (a b x : ℝ) : ℝ := a * x + b

-- Main theorem
theorem find_ab_pairs (a b : ℝ) :
  (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → 
    f a b x * f a b y + f a b (x + y - x * y) ≤ 0) ↔ 
  (-1 ≤ b ∧ b ≤ 0 ∧ -(b + 1) ≤ a ∧ a ≤ -b) :=
by sorry

end find_ab_pairs_l418_418671


namespace simplify_fraction_l418_418085

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l418_418085


namespace angle_AMP_is_118_125_degrees_l418_418500

theorem angle_AMP_is_118_125_degrees
  (A B M N P : Type)
  (h1: M = midpoint A B)
  (h2: N = midpoint M B)
  (h3: semicircle_on_diameter A B)
  (h4: semicircle_on_diameter M B)
  (h5: semicircle_on_diameter N B)
  (h6: splits_region_into_equal_areas M P) :
  measure_angle A M P = 118.125 :=
by
  sorry

end angle_AMP_is_118_125_degrees_l418_418500


namespace rectangle_BM_length_l418_418705

open Set

-- Define the theorem according to the conditions
theorem rectangle_BM_length (a : ℝ) (h1 : ∀ {A B C D: Point}, is_rectangle A B C D)
  (h2 : ∀ {A B : Point}, length(A, B) = a)
  (h3 : ∀ {B C : Point}, length(B, C) = (2 * a) / 3)
  (h4 : ∀ {A D : Point}, midpoint(A, D) = K)
  (h5 : ∀ {C D : Point} (L : Point), length(C, L) = length(A, K))
  (h6 : ∀ {K L : Point}, length(K, L) = 4)
  : ∀ {B M : Point}, length(B, M) = 8 := sorry

end rectangle_BM_length_l418_418705


namespace minimum_value_at_one_over_e_l418_418729

def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_at_one_over_e (x : ℝ) (hx : 0 < x) : x = 1 / Real.exp 1 → f x = - 1 / Real.exp 1 := by
  sorry

end minimum_value_at_one_over_e_l418_418729


namespace parabolic_triangle_area_l418_418488

theorem parabolic_triangle_area (n : ℕ) : 
  ∃ (m : ℕ) (x1 y1 x2 y2 x3 y3: ℤ), 
  (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x1, y1) ≠ (x3, y3) ∧ 
  2 ∣ y1 - x1^2 ∧ 2 ∣ y2 - x2^2 ∧ 2 ∣ y3 - x3^2 ∧ 
  (∃ k : ℕ, k = (2^n * m)^2 ∧ 2 * k = abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))) :=
begin
  sorry
end

end parabolic_triangle_area_l418_418488


namespace bisector_intersects_varies_l418_418780

open Real EuclideanGeometry Circle Segment

-- Define the problem conditions
variables {O A B C D : Point} 
variables {r : ℝ}  -- radius

-- Fixed chord AB of the circle
axiom fixed_chord : Segment A B

-- Point C is any point on the circle with center O and radius r
axiom point_on_circle : Circle O r.contains C

-- Chord CD forms a 30º angle with chord AB
axiom angle_condition : ∃ D, Segment C D ∧ Angle (C, O, D) = 30°

-- Define quadrant traversal for point C
axiom quadrant_traversal : ∀ θ, (0 ≤ θ) ∧ (θ ≤ π/2) → Circle O r.radius_point θ = C

-- State the theorem to be proved
theorem bisector_intersects_varies :
  ∃ P, (∉ {A, B}) ∧ (is_bisector_point P (Circle O r) (Segment O C) (Segment C D)) → varies :=
sorry

end bisector_intersects_varies_l418_418780


namespace combined_students_yellow_blue_l418_418482

theorem combined_students_yellow_blue {total_students blue_percent red_percent yellow_combined : ℕ} :
  total_students = 200 →
  blue_percent = 30 →
  red_percent = 40 →
  yellow_combined = (total_students * 3 / 10) + ((total_students - (total_students * 3 / 10)) * 6 / 10) →
  yellow_combined = 144 :=
by
  intros
  sorry

end combined_students_yellow_blue_l418_418482


namespace pages_copied_for_15_dollars_l418_418808

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l418_418808


namespace base_conversion_subtraction_l418_418645

def base6_to_base10 (n : ℕ) : ℕ :=
3 * (6^2) + 2 * (6^1) + 5 * (6^0)

def base5_to_base10 (m : ℕ) : ℕ :=
2 * (5^2) + 3 * (5^1) + 1 * (5^0)

theorem base_conversion_subtraction : 
  base6_to_base10 325 - base5_to_base10 231 = 59 :=
by
  sorry

end base_conversion_subtraction_l418_418645


namespace number_of_cars_lifted_l418_418426

def total_cars_lifted : ℕ := 6

theorem number_of_cars_lifted : total_cars_lifted = 6 := by
  sorry

end number_of_cars_lifted_l418_418426


namespace cats_bought_l418_418885

def original_cats := 11.0
def current_cats := 54

theorem cats_bought : current_cats - original_cats = 43 :=
by
  sorry

end cats_bought_l418_418885


namespace largest_two_digit_number_divisible_by_6_ending_in_4_l418_418905

theorem largest_two_digit_number_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ n = 84 :=
by
  existsi 84
  sorry

end largest_two_digit_number_divisible_by_6_ending_in_4_l418_418905


namespace ellipse_problem_ellipse_slope_condition_l418_418354

-- Statement (conditions and goals) in Lean 4

-- Condition: Ellipse equation E and constraints on b
def ellipse_pred (x y b : ℝ) : Prop := x^2 + y^2 / b^2 = 1 ∧ 0 < b ∧ b < 1

-- foci F1 and F2 definitions and condition for line l intersecting E at points A and B
def foci (F1 F2 : ℝ × ℝ) (b : ℝ) : Prop := 
  F1 = (-sqrt(1 - b^2), 0) ∧ F2 = (sqrt(1 - b^2), 0)

-- Line l passing through F1 that intersects ellipse E at points A and B
def line_and_intersections (A B : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  -- line l has an arbitrary slope (not necessarily 1)
  ∃ l : ℝ → ℝ, l F1.1 = F1.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Distances forming an arithmetic sequence
def arithmetic_sequence (AF2 AB BF2 : ℝ) : Prop :=
  AF2 + AB + BF2 = 4 ∧ 2 * AB = AF2 + BF2

-- Given conditions for the second part, slope = 1 implies specific value for b
def slope_condition (l : ℝ → ℝ) (slope : ℝ) : Prop := ∀ x, l x = slope * x

-- The main theorem
theorem ellipse_problem (b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h_ellipse : ellipse_pred A.1 A.2 b)
  (h_foci : foci F1 F2 b)
  (h_intersections : line_and_intersections A B F1)
  (h_seq : arithmetic_sequence (dist A F2) (dist A B) (dist B F2)) :
  dist A B = 4 / 3 :=
sorry

-- Theorem for the specific slope condition
theorem ellipse_slope_condition (b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h_ellipse : ellipse_pred A.1 A.2 b)
  (h_foci : foci F1 F2 b)
  (h_intersections : line_and_intersections A B F1)
  (h_seq : arithmetic_sequence (dist A F2) (dist A B) (dist B F2))
  (h_slope : slope_condition (λ x, x + sqrt(1 - b^2)) 1) :
  b = sqrt(2) / 2 :=
sorry

end ellipse_problem_ellipse_slope_condition_l418_418354


namespace sum_sequence_l418_418271

noncomputable def sequence (n : ℕ) : ℕ → ℕ
| 1     := 2
| (k+1) := sequence k + k

theorem sum_sequence (n : ℕ) : 
  (∑ k in Finset.range n, sequence n k) = 2 * n + (n - 1) * n * (n + 1) / 6 :=
by
  sorry

end sum_sequence_l418_418271


namespace isosceles_triangle_ef_length_l418_418412

theorem isosceles_triangle_ef_length :
  ∀ (D E F G : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]
  (DE DF : ℝ) 
  (h₁ : DE = 10) 
  (h₂ : DF = 10)
  (h₃ : ∃ (G : E × F), 2 * Real.dist DE G = Real.dist G DF)
  (h₄ : ∃ (x : ℝ), EF = 3x) ,
  EF = 6 * Real.sqrt 5 :=
by
  sorry

end isosceles_triangle_ef_length_l418_418412


namespace solve_equation_l418_418955

theorem solve_equation (x : ℝ) (h : (x^2 + x + 1) / (x + 1) = x + 2) : x = -1/2 :=
by sorry

end solve_equation_l418_418955


namespace cylinder_volume_equal_l418_418146

theorem cylinder_volume_equal (x : Real) :
  let r₁ := 5  -- initial radius of both cylinders
  let h₁ := 7  -- initial height of both cylinders
  let new_r₁ := 2 * r₁  -- radius doubled for the first cylinder
  let new_h₂ := 3 * h₁  -- height tripled for the second cylinder
  let V₁ := (Math.pi * (new_r₁ + x)^2 * h₁) -- volume of the first modified cylinder
  let V₂ := (Math.pi * r₁^2 * (new_h₂ + x))  -- volume of the second modified cylinder
  
  (V₁ = V₂) → x = (5 + sqrt 9125) / 14 :=
begin
  sorry
end

end cylinder_volume_equal_l418_418146


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l418_418941

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l418_418941


namespace sequence_general_formula_l418_418863

theorem sequence_general_formula {a : ℕ → ℝ} (h : ∀ n : ℕ, n > 0 → (∑ i in finset.range n, 2^i * a (i + 1)) = n / 2) : 
  ∀ (n : ℕ), n > 0 → a n = 1 / (2 ^ n) :=
by
  -- Given condition: ∀ n > 0, ∑ i in finset.range n, 2^i * a (i + 1) = n / 2
  sorry

end sequence_general_formula_l418_418863


namespace copy_pages_l418_418818

theorem copy_pages (cost_per_page total_cents : ℕ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) : (total_cents / cost_per_page) = 500 :=
by {
  rw [h1, h2],
  norm_num,
}

end copy_pages_l418_418818


namespace john_fixed_computers_l418_418017

theorem john_fixed_computers (total_computers unfixable waiting_for_parts fixed_right_away : ℕ)
  (h1 : total_computers = 20)
  (h2 : unfixable = 0.20 * 20)
  (h3 : waiting_for_parts = 0.40 * 20)
  (h4 : fixed_right_away = total_computers - unfixable - waiting_for_parts) :
  fixed_right_away = 8 :=
by
  sorry

end john_fixed_computers_l418_418017


namespace find_apex_angle_l418_418998

noncomputable def apex_angle_of_identical_cones : ℝ :=
  2 * (Real.arcsin (1 / 4)) + π / 6

theorem find_apex_angle (a1 a2 a3 a4 : Cone)
  (h_touch : TouchExternally a1 a2 ∧ TouchExternally a2 a3 ∧ TouchExternally a1 a3)
  (h_identical : a1.apexAngle = a2.apexAngle)
  (h_third : a3.apexAngle = 2 * Real.arcsin (1 / 4))
  (h_fourth : InternalTouch a1 a4 ∧ InternalTouch a2 a4 ∧ InternalTouch a3 a4)
  (h_apexA : a4.apex = a1.apex ∧ a4.apex = a2.apex ∧ a4.apex = a3.apex)
  (half_apex : ∀ (x y : Cone), x.apexAngle = 2 * y.apexAngle → y.apexAngle = a4.apexAngle) :
  a1.apexAngle = apex_angle_of_identical_cones :=
sorry

end find_apex_angle_l418_418998


namespace simplify_fraction_l418_418065

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l418_418065


namespace quadratic_root_u_quartic_root_v_l418_418053

noncomputable def u : ℝ := Real.cot (22.5 * Real.pi / 180)
noncomputable def v : ℝ := 1 / Real.sin (22.5 * Real.pi / 180)

theorem quadratic_root_u : Polynomial.eval u (Polynomial.C (0:ℝ) + Polynomial.X^2 - Polynomial.C 2 * Polynomial.X - Polynomial.C 1) = 0 := sorry

theorem quartic_root_v : Polynomial.eval v (Polynomial.C (0:ℝ) + Polynomial.X^4 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 8) = 0 := sorry

end quadratic_root_u_quartic_root_v_l418_418053


namespace prime_cubic_condition_l418_418388

theorem prime_cubic_condition (p : ℕ) (hp : Nat.Prime p) (hp_prime : Nat.Prime (p^4 - 3 * p^2 + 9)) : p = 2 :=
sorry

end prime_cubic_condition_l418_418388


namespace kara_water_intake_l418_418024

-- Definitions based on the conditions
def daily_doses := 3
def week1_days := 7
def week2_days := 7
def forgot_doses_day := 2
def total_weeks := 2
def total_water := 160

-- The statement to prove
theorem kara_water_intake :
  let total_doses := (daily_doses * week1_days) + (daily_doses * week2_days - forgot_doses_day)
  ∃ (water_per_dose : ℕ), water_per_dose * total_doses = total_water ∧ water_per_dose = 4 :=
by
  sorry

end kara_water_intake_l418_418024


namespace inequality_am_gm_l418_418337

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (x + z) + (2 * z^2) / (x + y) ≥ x + y + z :=
by
  sorry

end inequality_am_gm_l418_418337


namespace total_trees_expr_total_trees_given_A_12_l418_418105

-- Definitions based on conditions
def trees_planted_by_B (x : ℕ) := x
def trees_planted_by_A (x : ℕ) := (1.2 * x : ℝ)
def trees_planted_by_C (x : ℕ) := (1.2 * x : ℝ) - 2

-- Proof statement: Part 1
theorem total_trees_expr (x : ℕ) : 
    trees_planted_by_B x + trees_planted_by_A x + trees_planted_by_C x = 3.4 * x - 2 :=
by sorry

-- Definitions based on given "A planted 12 trees"
def B_when_A_12 := 12 / 1.2
def C_when_A_12 := 12 - 2

-- Proof statement: Part 2
theorem total_trees_given_A_12 (hA : trees_planted_by_A 10 = 12) : 
    12 + B_when_A_12 + C_when_A_12 = 32 :=
by sorry

end total_trees_expr_total_trees_given_A_12_l418_418105


namespace solution_set_l418_418763

open Set

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^(x - 1) - 3 else - (2^(-x - 1) - 3)

lemma odd_function_f (x : ℝ) : f (-x) = - (f x) :=
by
  by_cases h : x > 0
  {
    -- x > 0 implies -x < 0
    have h_neg : -x < 0 := by linarith
    unfold f
    rw [if_neg h_neg, if_pos h]
    ring
  }
  {
    -- x <= 0 implies -x >= 0
    have h_neg : -x >= 0 := by linarith
    by_cases h0 : x = 0
    {
      have : -x = 0 := by linarith
      rw [h0, this]
      unfold f
      rw [if_neg h0, if_pos (by linarith)]
      norm_num
    }
    {
      have : x < 0 := by linarith
      have h_neg' : -x > 0 := by linarith
      unfold f
      rw [if_neg this, if_pos h_neg']
      ring
    }
  }

theorem solution_set (x : ℝ) : f x > 1 ↔ (x ∈ Ioo (-2 : ℝ) 0 ∪ Ioi 3) :=
by sorry

end solution_set_l418_418763


namespace students_who_like_yellow_and_blue_l418_418479

/-- Problem conditions -/
def total_students : ℕ := 200
def percentage_blue : ℕ := 30
def percentage_red_among_not_blue : ℕ := 40

/-- We need to prove the following statement: -/
theorem students_who_like_yellow_and_blue :
  let num_blue := (percentage_blue * total_students) / 100 in
  let num_not_blue := total_students - num_blue in
  let num_red := (percentage_red_among_not_blue * num_not_blue) / 100 in
  let num_yellow := num_not_blue - num_red in
  num_yellow + num_blue = 144 :=
by
  sorry

end students_who_like_yellow_and_blue_l418_418479


namespace total_trees_expression_total_trees_given_A_l418_418102

section
variables (x : ℝ) (total_trees : ℝ)

-- Define conditions
def num_trees_A (trees_B : ℝ) : ℝ := 1.2 * trees_B
def num_trees_C (trees_A : ℝ) : ℝ := trees_A - 2

-- Lean statement for the first question
theorem total_trees_expression (hx : x ≥ 0) :
  total_trees = x + num_trees_A x + num_trees_C (num_trees_A x) → 
  total_trees = 3.4 * x - 2 :=
by sorry

-- Lean statement for the second question with A planted 12 trees
def num_trees_B_given_A (trees_A : ℝ) : ℝ := trees_A / 1.2
def num_trees_C_given_A (trees_A : ℝ) : ℝ := trees_A - 2

theorem total_trees_given_A (hA : num_trees_A 10 = 12) (A_planted : ℝ) :
  A_planted = 12 →
  total_trees = A_planted + num_trees_B_given_A 12 + num_trees_C_given_A 12 → 
  total_trees = 32 :=
by sorry
end

end total_trees_expression_total_trees_given_A_l418_418102


namespace statement1_statement2_statement3_statement4_correct_statements_count_l418_418714

-- Definitions of polynomials M and N
def M (x : ℝ) : ℝ := 2 - 3x
def N (x : ℝ) : ℝ := 3x + 1

-- Statement 1: (2M + 3N = 4) iff (x = 1)
theorem statement1 (x : ℝ) : 2 * M x + 3 * N x = 4 ↔ x = 1 := sorry

-- Statement 2: Solution set of inequalities gives 5a + b = 3
theorem statement2 (a b : ℝ) : (0 < x ∧ x < 3) ↔ (5 * a + b = 3) := sorry

-- Statement 3: If x is an integer and (M + 4) / N is an integer, then x = 0 or x = 2.
theorem statement3 (x : ℤ) : (M x + 4) / N x ∈ ℤ ↔ x = 0 ∨ x = 2 := sorry

-- Statement 4: If (N - M) / 2 = y, then expression equals 2023.
theorem statement4 (x y : ℝ) : (N x - M x) / 2 = y → 2022 * x - 674 * y + 1686 = 2023 := sorry

-- Count of correct statements
theorem correct_statements_count (n : ℕ) : n = 3 := sorry

end statement1_statement2_statement3_statement4_correct_statements_count_l418_418714


namespace projection_sum_of_squares_invariant_l418_418443

variables {A B C D A' B' C' D' : ℝ × ℝ}
variables {M : ℝ × ℝ}
variables {ℓ : ℝ → ℝ}
variables {d : ℝ}

def is_square (A B C D : ℝ × ℝ) : Prop :=
  -- Define the conditions for points A, B, C, D to form a square of diagonal length 2 centered at M
  let diag_len : ℝ := 2 in
  let M := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = diag_len^2 ∧
  (B.1 - D.1)^2 + (B.2 - D.2)^2 = diag_len^2 ∧
  (A.1 - B.1) = (D.1 - C.1) ∧ (A.2 - B.2) = (D.2 - C.2)

def is_projection (P P' : ℝ × ℝ) (ℓ : ℝ → ℝ → Prop) : Prop :=
  -- Define the condition for P' to be orthogonal projection of P onto line ℓ
  ∃ a b c, ℓ = (λ x y, a * x + b * y + c = 0) ∧ 
  a * P'.1 + b * P'.2 + c = 0

theorem projection_sum_of_squares_invariant (A B C D A' B' C' D' : ℝ × ℝ) (ℓ : ℝ → ℝ → Prop) (d : ℝ) (M : ℝ × ℝ) 
  (h_square : is_square A B C D)
  (h_projection_A : is_projection A A' ℓ) 
  (h_projection_B : is_projection B B' ℓ)
  (h_projection_C : is_projection C C' ℓ)
  (h_projection_D : is_projection D D' ℓ)
  (h_dist : ∀ (x y : ℝ), ℓ x y → (x - M.1)^2 + (y - M.2)^2 > 1):
  (A.1 - A'.1)^2 + (A.2 - A'.2)^2 + 
  (B.1 - B'.1)^2 + (B.2 - B'.2)^2 + 
  (C.1 - C'.1)^2 + (C.2 - C'.2)^2 + 
  (D.1 - D'.1)^2 + (D.2 - D'.2)^2 = 
  4 * ((M.1 - A'.1)^2 + (M.2 - A'.2)^2) + 2 := 
sorry

end projection_sum_of_squares_invariant_l418_418443


namespace number_of_boys_l418_418127

theorem number_of_boys {total_students : ℕ} (h1 : total_students = 49)
  (ratio_girls_boys : ℕ → ℕ → Prop)
  (h2 : ratio_girls_boys 4 3) :
  ∃ boys : ℕ, boys = 21 := by
  sorry

end number_of_boys_l418_418127


namespace flea_death_ensured_l418_418409

-- Definitions of vectors and conditions
variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Assume u1, u2, u3 are vectors in ℝ^2
structure Vector2 (ℝ : Type*) :=
(x : ℝ)
(y : ℝ)

variables (u1 u2 u3 : Vector2 ℝ)

-- Conditions that the vectors do not lie in the same half-plane
def not_in_same_halfplane (u1 u2 u3 : Vector2 ℝ) : Prop :=
  ∃ (a1 a2 a3 : ℝ), (a1 ≠ 0 ∨ a2 ≠ 0 ∨ a3 ≠ 0) ∧
    a1 * u1.x + a2 * u2.x + a3 * u3.x = 0 ∧
    a1 * u1.y + a2 * u2.y + a3 * u3.y = 0

-- Main theorem statement
theorem flea_death_ensured 
  (u1 u2 u3 : Vector2 ℝ) 
  (hne : not_in_same_halfplane u1 u2 u3) : 
  ∃ (finite_paths : ℕ), ∀ (flea_path : ℕ → Vector2 ℝ), 
    flea_path 0 = ⟨0, 0⟩ ∧
    (∀ n, flea_path (n + 1) = flea_path n + u1 ∨
        flea_path (n + 1) = flea_path n + u2 ∨
        flea_path (n + 1) = flea_path n + u3 ∨
        flea_path (n + 1) = flea_path n) →
    ∃ (t : ℕ), flea_path t = flea_path (t + 1) -- flea jumps to or lands on a poisoned point (silenced)
  :=
sorry

end flea_death_ensured_l418_418409


namespace can_reach_4_white_l418_418802

/-
We define the possible states and operations on the urn as described.
-/

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def operation1 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation2 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation3 (u : Urn) : Urn :=
  { white := u.white - 1, black := u.black - 1 }

def operation4 (u : Urn) : Urn :=
  { white := u.white - 2, black := u.black + 1 }

theorem can_reach_4_white : ∃ (u : Urn), u.white = 4 ∧ u.black > 0 :=
  sorry

end can_reach_4_white_l418_418802


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418922

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418922


namespace probability_of_even_number_l418_418211

def is_even (n : ℕ) : Prop := n % 2 = 0

def probability_of_event (n : ℕ) : ℚ := 1 / n

theorem probability_of_even_number : probability_of_event 6 * (if is_even 2 then 1 else 0 + if is_even 4 then 1 else 0 + if is_even 6 then 1 else 0) = 1 / 2 :=
sorry

end probability_of_even_number_l418_418211


namespace three_x4_plus_two_x5_l418_418275

theorem three_x4_plus_two_x5 (x1 x2 x3 x4 x5 : ℤ)
  (h1 : 2 * x1 + x2 + x3 + x4 + x5 = 6)
  (h2 : x1 + 2 * x2 + x3 + x4 + x5 = 12)
  (h3 : x1 + x2 + 2 * x3 + x4 + x5 = 24)
  (h4 : x1 + x2 + x3 + 2 * x4 + x5 = 48)
  (h5 : x1 + x2 + x3 + x4 + 2 * x5 = 96) : 
  3 * x4 + 2 * x5 = 181 := 
sorry

end three_x4_plus_two_x5_l418_418275


namespace club_population_after_four_years_l418_418605

theorem club_population_after_four_years:
  let b : ℕ → ℕ,
      b 0 := 20,
      b (n+1) := 4 * (b n - 4) + 4 in
  b 4 = 4100 :=
by
  let b : ℕ → ℕ
  exact fun n =>
    match n with
    | 0    => 20
    | n+1  => 4 * (b n - 4) + 4
  sorry

end club_population_after_four_years_l418_418605


namespace tables_in_convention_center_l418_418782

theorem tables_in_convention_center : ∃ t : ℕ, 
  (let c := 8 * t in 
   let c_extra := 10 in 
   4 * c + 5 * t + 4 * c_extra = 1010) 
  ∧ t = 26 := 
by
  -- Define variables
  let c := 8 * 26
  let c_extra := 10
  -- Ensure conditions
  have H1 : 4 * c + 5 * 26 + 4 * c_extra = 1010 := by sorry
  -- Prove the number of tables
  existsi 26
  split
  exact H1
  refl

end tables_in_convention_center_l418_418782


namespace sum_of_powers_l418_418859

noncomputable def z : ℂ := (-1 + complex.I) / real.sqrt 2

theorem sum_of_powers (z : ℂ) (h : z = (-1 + complex.I) / real.sqrt 2) :
  (∑ k in finset.range 1 11, z^(k^2)) * (∑ k in finset.range 1 11, (1 / z)^(k^2)) = 25 :=
by
  sorry

end sum_of_powers_l418_418859


namespace classroom_activity_solution_l418_418280

theorem classroom_activity_solution 
  (x y : ℕ) 
  (h1 : x - y = 6) 
  (h2 : x * y = 45) : 
  x = 11 ∧ y = 5 :=
by
  sorry

end classroom_activity_solution_l418_418280


namespace problem_statement_l418_418171

/-- 
Define a statement that checks if a given statement from the provided 
list is correct based on geometric and algebraic properties.
-/
def correctStatement : Prop :=
  (∀ (a : ℝ), ¬ (sqrt a = quadratic radical)) ∧
  (∀ (R : Type) [EuclideanGeometry R] (r s : R) (rect : Rectangle r s), 
    adjacentSidesEqual rect → isSquare rect) ∧ 
  (∀ (Q : Type) (diag1 diag2 : Q) (quad : Quadrilateral diag1 diag2), 
    equalDiagonals quad → ¬ isRectangle quad) ∧
  (∀ (ABC : Type) [RightTriangle ABC] (a b c : ℝ), 
    sideRelation ABC a b c → a^2 + b^2 = c^2)

theorem problem_statement : correctStatement :=
by sorry

end problem_statement_l418_418171


namespace number_of_lines_satisfying_conditions_l418_418378

def Point := ℝ × ℝ

def distance_to_line (P : Point) (l : ℝ × ℝ × ℝ) : ℝ :=
  let (a, b, c) := l in
  (abs (a * P.1 + b * P.2 + c)) / (sqrt (a^2 + b^2))

def PointA : Point := (1, 2)
def PointB : Point := (4, -2)

/-- Given points A and B, the number of lines \( l \) such that the distance from A to \( l \) is 1 and the distance from B to \( l \) is 4 is 3 -/
theorem number_of_lines_satisfying_conditions : 
  ∃ l₁ l₂ l₃ : ℝ × ℝ × ℝ, 
    (distance_to_line PointA l₁ = 1 ∧ distance_to_line PointB l₁ = 4) ∧
    (distance_to_line PointA l₂ = 1 ∧ distance_to_line PointB l₂ = 4) ∧
    (distance_to_line PointA l₃ = 1 ∧ distance_to_line PointB l₃ = 4) ∧
    ∀ l, ((distance_to_line PointA l = 1 ∧ distance_to_line PointB l = 4) → l = l₁ ∨ l = l₂ ∨ l = l₃) :=
sorry

end number_of_lines_satisfying_conditions_l418_418378


namespace smallest_value_in_geometric_progression_l418_418229

open Real

theorem smallest_value_in_geometric_progression 
  (d : ℝ) : 
  (∀ a b c d : ℝ, 
    a = 5 ∧ b = 5 + d ∧ c = 5 + 2 * d ∧ d = 5 + 3 * d ∧ 
    ∀ a' b' c' d' : ℝ, 
      a' = 5 ∧ b' = 6 + d ∧ c' = 15 + 2 * d ∧ d' = 3 * d ∧ 
      (b' / a' = c' / b' ∧ c' / b' = d' / c')) → 
  (d = (-1 + 4 * sqrt 10) ∨ d = (-1 - 4 * sqrt 10)) → 
  (min (3 * (-1 + 4 * sqrt 10)) (3 * (-1 - 4 * sqrt 10)) = -3 - 12 * sqrt 10) :=
by
  intros ha hd
  sorry

end smallest_value_in_geometric_progression_l418_418229


namespace two_pow_start_digits_l418_418892

theorem two_pow_start_digits (A : ℕ) : 
  ∃ (m n : ℕ), 10^m * A < 2^n ∧ 2^n < 10^m * (A + 1) :=
  sorry

end two_pow_start_digits_l418_418892


namespace simplify_fraction_l418_418079

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l418_418079


namespace mallory_travel_expenses_l418_418113

theorem mallory_travel_expenses (fuel_tank_cost : ℕ) (fuel_tank_miles : ℕ) (total_miles : ℕ) (food_ratio : ℚ)
  (h_fuel_tank_cost : fuel_tank_cost = 45)
  (h_fuel_tank_miles : fuel_tank_miles = 500)
  (h_total_miles : total_miles = 2000)
  (h_food_ratio : food_ratio = 3/5) :
  ∃ total_cost : ℕ, total_cost = 288 :=
by
  sorry

end mallory_travel_expenses_l418_418113


namespace line_parallel_or_contained_l418_418758

variables {α β : Type*} [linear_ordered_field α]

-- Definitions of line and plane
structure Line (β : Type*) :=
  (points : set β)
  (is_line : ∃ p q : β, p ≠ q ∧ p ∈ points ∧ q ∈ points)

structure Plane (β : Type*) :=
  (points : set β)
  (is_plane : ∃ p q r : β, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p ∈ points ∧ q ∈ points ∧ r ∈ points)

-- Definitions for parallel lines and parallel line to a plane
def parallel_lines (a b : Line β) : Prop :=
  ∀ p ∈ a.points, ∀ q ∈ b.points, p ≠ q → (∃ s : set β, is_line s ∧ p ∈ s ∧ q ∈ s)

def parallel_line_plane (a : Line β) (α : Plane β) : Prop :=
  ∃ a' : Line β, parallel_lines a a' ∧ (∀ p ∈ a'.points, p ∈ α.points)

-- Positional relationship between line b and line a
theorem line_parallel_or_contained (a b : Line β) (α : Plane β) 
  (h1 : parallel_lines a b) 
  (h2 : parallel_line_plane a α) :
  parallel_lines b a ∨ ∀ p ∈ b.points, p ∈ α.points :=
sorry

end line_parallel_or_contained_l418_418758


namespace sum_of_divisors_of_common_numbers_l418_418556

theorem sum_of_divisors_of_common_numbers :
  let nums := [48, 144, 24, 192, 216, 120] in
  let common_divisors := {d ∣ 48 ∧ d ∣ 144 ∧ d ∣ 24 ∧ d ∣ 192 ∧ d ∣ 216 ∧ d ∣ 120 | d : ℕ} in
  ∑ d in common_divisors, d = 16 :=
begin
  sorry
end

end sum_of_divisors_of_common_numbers_l418_418556


namespace paint_budget_exceeds_l418_418641

theorem paint_budget_exceeds : 
  let bedrooms := 5
  let bathrooms := 10
  let kitchen := 1
  let living_rooms := 2
  let dining_room := 1
  let study_room := 1
  let gallons_bedroom := 3
  let gallons_bathroom := 2
  let gallons_kitchen := 4
  let gallons_living_room := 6
  let gallons_dining_room := 4
  let gallons_study_room := 3
  let cost_colored_per_gallon := 18
  let cost_white_per_can := 40

  let total_gallons_bedroom := bedrooms * gallons_bedroom
  let total_gallons_bathroom := bathrooms * gallons_bathroom
  let total_gallons_kitchen := kitchen * gallons_kitchen
  let total_gallons_living_room := living_rooms * gallons_living_room
  let total_gallons_dining_room := dining_room * gallons_dining_room
  let total_gallons_study_room := study_room * gallons_study_room

  let total_white_gallons := total_gallons_bathroom
  let total_colored_gallons := total_gallons_bedroom + total_gallons_kitchen + total_gallons_living_room + total_gallons_dining_room + total_gallons_study_room

  let white_cans := (total_white_gallons + 2) / 3  -- ceiling division
  let colored_gallons := total_colored_gallons

  let total_cost := white_cans * cost_white_per_can + colored_gallons * cost_colored_per_gallon

  total_cost > 500 :=
by
  let total_gallons_bedroom := 15
  let total_gallons_bathroom := 20
  let total_gallons_kitchen := 4
  let total_gallons_living_room := 12
  let total_gallons_dining_room := 4
  let total_gallons_study_room := 3

  let total_white_gallons := total_gallons_bathroom
  let total_colored_gallons := total_gallons_bedroom + total_gallons_kitchen + total_gallons_living_room + total_gallons_dining_room + total_gallons_study_room

  let white_cans := Nat.ceilDiv total_white_gallons 3
  let colored_gallons := total_colored_gallons

  let total_cost := white_cans * cost_white_per_can + colored_gallons * cost_colored_per_gallon

  have h : total_cost = 964 := sorry
  show total_cost > 500
  rw h
  exact by decide,
  sorry

end paint_budget_exceeds_l418_418641


namespace proof_problem_l418_418651

variable {O A B C D P L T : Type _}

-- Conditions
variable (length_AB : ℝ) (length_CD : ℝ)
variable (exts_BA_CD_int_P : Prop)
variable (PO_int_AC_L : Prop)
variable (ratio_AL_LC : ℝ)
variable (radius_O : ℝ)
variable (center_inscribed_T : Prop)

-- Assign values to known parameters
def conditions : Prop :=
  length_AB = 5 ∧ length_CD = 5 ∧
  exts_BA_CD_int_P ∧
  PO_int_AC_L ∧
  ratio_AL_LC = 1 / 2 ∧
  radius_O = 6.5 ∧
  center_inscribed_T

-- Goals
def to_prove : Prop :=
  (5 = 5 ∧ PT = (3 * Real.sqrt 41 - 13) / 2 ∧ 
   (10 * 5 / (2 * 41)) = 1000 / 41)

-- Main theorem statement
theorem proof_problem (h : conditions) : to_prove := sorry

end proof_problem_l418_418651


namespace valid_P_set_l418_418644

def unit_circle (C : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 = 1

def d (L : set (ℝ × ℝ)) (C : set (ℝ × ℝ)) : ℝ :=
  if ∃ p1 p2 : ℝ × ℝ, p1 ∈ L ∧ p2 ∈ L ∧ p1 ∈ C ∧ p2 ∈ C ∧ p1 ≠ p2
  then dist p1 p2
  else 0

def f (P : ℝ × ℝ) (C : set (ℝ × ℝ)) : ℝ :=
  let L := {L | ∃ a b : ℝ, L = {q : ℝ × ℝ | a*q.1 + b*q.2 = 0} ∧ (P.1*a + P.2*b) = 0 } in
  ⨆ (L L' : set (ℝ × ℝ)), L ∈ {L} ∧ L' ∈ {L} ∧ L ≠ L' ∧ 
                          inner_product_LP_everywhere_L' ∧ ∀ x ∈ L, ∃ y ∈ L', d L C + d L' C

def valid_points (C : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
  {P | f(P, C) > 2} 

theorem valid_P_set (O : set (ℝ × ℝ)) {P : (ℝ × ℝ)} (C : set (ℝ × ℝ)) (hC : unit_circle C) :
  (C = {p : ℝ × ℝ | norm p < sqrt(3/2)}) ↔ P ∈ valid_points C :=
sorry

end valid_P_set_l418_418644


namespace not_exists_18_consecutive_good_l418_418870

def is_good (n : ℕ) : Prop := 
  (∀ p : ℕ, nat.prime p → p ∣ n → nat.factor_multiset p n = 2) 

theorem not_exists_18_consecutive_good :
  ¬ ∃ (a : ℕ), ∀ (i : ℕ), i < 18 → is_good (a + i) :=
sorry

end not_exists_18_consecutive_good_l418_418870


namespace simplify_fraction_l418_418084

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l418_418084


namespace scientific_notation_142000_l418_418612

theorem scientific_notation_142000 : (142000 : ℝ) = 1.42 * 10^5 := sorry

end scientific_notation_142000_l418_418612


namespace new_average_weight_l418_418970

theorem new_average_weight (avg_weight_19_students : ℝ) (new_student_weight : ℝ) (num_students_initial : ℕ) : 
  avg_weight_19_students = 15 → new_student_weight = 7 → num_students_initial = 19 → 
  let total_weight_with_new_student := (avg_weight_19_students * num_students_initial + new_student_weight) 
  let new_num_students := num_students_initial + 1 
  let new_avg_weight := total_weight_with_new_student / new_num_students 
  new_avg_weight = 14.6 :=
by
  intros h1 h2 h3
  let total_weight := avg_weight_19_students * num_students_initial
  let total_weight_with_new_student := total_weight + new_student_weight
  let new_num_students := num_students_initial + 1
  let new_avg_weight := total_weight_with_new_student / new_num_students
  have h4 : total_weight = 285 := by sorry
  have h5 : total_weight_with_new_student = 292 := by sorry
  have h6 : new_num_students = 20 := by sorry
  have h7 : new_avg_weight = 292 / 20 := by sorry
  have h8 : new_avg_weight = 14.6 := by sorry
  exact h8

end new_average_weight_l418_418970


namespace number_of_teachers_l418_418202

theorem number_of_teachers (total_people total_sampled sampled_students : ℕ) (total_2400 : total_people = 2400)
  (sample_160 : total_sampled = 160) (sampled_students_150 : sampled_students = 150)
  (stratified_sampling_ratio: (sampled_students / total_sampled) = (students / total_people)) :
  let students := 2250 in total_people - students = 150 :=
by
  -- assumptions
  have h1 : total_people = 2400 := total_2400
  have h2 : total_sampled = 160 := sample_160
  have h3 : sampled_students = 150 := sampled_students_150
  -- assertions and proof
  let students := 2250
  have stratified_ratio : (students / 2400) = (150 / 160) := stratified_sampling_ratio
  sorry

end number_of_teachers_l418_418202


namespace initial_volume_mixture_l418_418137

theorem initial_volume_mixture (V : ℝ) (h1 : 0.84 * V = 0.6 * (V + 24)) : V = 60 :=
by
  sorry

end initial_volume_mixture_l418_418137


namespace value_of_a_l418_418764

theorem value_of_a (a : ℝ) (h : ∀ x > 0, (a * x - 9) * log (2 * a / x) ≤ 0) : a = 3 * Real.sqrt 2 / 2 := 
by
  sorry

end value_of_a_l418_418764


namespace probability_letter_in_mathematics_l418_418756

/-- 
Given that Lisa picks one letter randomly from the alphabet, 
prove that the probability that Lisa picks a letter in "MATHEMATICS" is 4/13.
-/
theorem probability_letter_in_mathematics :
  (8 : ℚ) / 26 = 4 / 13 :=
by
  sorry

end probability_letter_in_mathematics_l418_418756


namespace pages_copied_for_15_dollars_l418_418809

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end pages_copied_for_15_dollars_l418_418809


namespace sin_cos_sum_to_sin2theta_l418_418356

theorem sin_cos_sum_to_sin2theta (theta : ℝ) (h : sin theta + cos theta = 1 / 3) : sin (2 * theta) = - 8 / 9 := by
  sorry

end sin_cos_sum_to_sin2theta_l418_418356


namespace grid_total_area_l418_418214

theorem grid_total_area 
  (num_squares : ℕ) 
  (diagonal_length : ℝ) 
  (num_congruent_squares : ℕ)
  (congruent_squares_diagonal : ℝ)
  (congruent_squares_area : ℝ)
  (total_area : ℝ)  :
  num_squares = 20 →
  diagonal_length = 10 →
  let large_square_area := (diagonal_length^2) / 2 in
  let small_square_area := large_square_area / 16 in
  let calculated_total_area := num_squares * small_square_area in
  total_area = calculated_total_area →
  total_area = 62.5 :=
by
  intros ns_eq dl_eq --
  sorry

end grid_total_area_l418_418214


namespace archie_touchdown_passes_l418_418636

-- Definitions based on the conditions
def richard_avg_first_14_games : ℕ := 6
def richard_avg_last_2_games : ℕ := 3
def richard_games_first : ℕ := 14
def richard_games_last : ℕ := 2

-- Total touchdowns Richard made in the first 14 games
def touchdowns_first_14 := richard_games_first * richard_avg_first_14_games

-- Total touchdowns Richard needs in the final 2 games
def touchdowns_last_2 := richard_games_last * richard_avg_last_2_games

-- Total touchdowns Richard made in the season
def richard_touchdowns_season := touchdowns_first_14 + touchdowns_last_2

-- Archie's record is one less than Richard's total touchdowns for the season
def archie_record := richard_touchdowns_season - 1

-- Proposition to prove Archie's touchdown passes in a season
theorem archie_touchdown_passes : archie_record = 89 := by
  sorry

end archie_touchdown_passes_l418_418636


namespace jerry_showers_l418_418279

-- Define constants and values according to the conditions provided.
def water_limit : ℕ := 1000
def drinking_cooking : ℕ := 100
def water_per_shower : ℕ := 20
def pool_length : ℕ := 10
def pool_width : ℕ := 10
def pool_depth : ℕ := 6
def gallons_per_cubic_foot : ℕ := 1
def leakage_odd_days : ℕ := 5
def leakage_even_days : ℕ := 8
def evaporation_rate : ℕ := 2
def total_days : ℕ := 31
def odd_days : ℕ := 16
def even_days : ℕ := 15

-- Statement of the problem in Lean 4
theorem jerry_showers : 
  let pool_gallons := pool_length * pool_width * pool_depth * gallons_per_cubic_foot in
  let total_leakage := (leakage_odd_days * odd_days) + (leakage_even_days * even_days) in
  let total_evaporation := evaporation_rate * total_days in
  let total_water_loss := total_leakage + total_evaporation in
  let total_pool_use := pool_gallons + total_water_loss in
  let total_usage := total_pool_use + drinking_cooking in
  let remaining_water := water_limit - total_usage in
  let possible_showers := remaining_water / water_per_shower in
  possible_showers = 1 :=
by {
  sorry 
}

end jerry_showers_l418_418279


namespace ellipse_contains_circles_area_min_k_l418_418243

theorem ellipse_contains_circles_area_min_k :
  ∃ (a b : ℝ), (∀ (x y : ℝ), (x - 2)^2 + y^2 = 4 → x^2 / a^2 + y^2 / b^2 = 1) ∧ 
               (∀ (x y : ℝ), (x + 2)^2 + y^2 = 4 → x^2 / a^2 + y^2 / b^2 = 1) ∧ 
               (∀ (a b : ℝ), a > 0 ∧ b > 0 → a * b = sqrt 3) :=
begin
  -- Proof is omitted
  sorry
end

end ellipse_contains_circles_area_min_k_l418_418243


namespace andrew_current_age_l418_418634

-- Definitions based on conditions.
def initial_age := 11  -- Andrew started donating at age 11
def donation_per_year := 7  -- Andrew donates 7k each year on his birthday
def total_donation := 133  -- Andrew has donated a total of 133k till now

-- The theorem stating the problem and the conclusion.
theorem andrew_current_age : 
  ∃ (A : ℕ), donation_per_year * (A - initial_age) = total_donation :=
by {
  sorry
}

end andrew_current_age_l418_418634


namespace part1_part3_l418_418772

-- Part (1) Lean statement
theorem part1 (a : ℕ → ℝ) (h : ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n) :
  (∀ n, 2^((n : ℕ)) ∈ {a n}) := sorry

-- Part (2) Lean statement
noncomputable def part2 (a : ℕ → ℤ) (hn : ∀ n, a (n + 2) - a (n + 1) > a (n + 1) - a n)
  (h1 : a 1 = 1) (h2 : a 2 = 3) (hk : ∃ (k : ℕ), a k = 2023) :
  ∃ (k : ℕ), k * (k + 1) / 2 ≤ 2023 := sorry

-- Part (3) Lean statement
theorem part3 (b : ℕ → ℤ) (k : ℕ) (h : 2 ≤ k) (hsum: (∑ i in range (2 * k), b i) = k)
  (hb : ∀ n, b (n + 2) - b (n + 1) > b (n + 1) - b n) :
  ∀ {c : ℕ → ℝ}, (∀ n, c n = 2^(b n)) → c k * c (k + 1) < 2 := sorry

end part1_part3_l418_418772


namespace comb_18_10_proof_l418_418655

theorem comb_18_10_proof (comb : ℕ → ℕ → ℕ) : comb 18 10 = 47190 :=
by
  assume h1 : comb 16 7 = 11440,
  assume h2 : comb 16 9 = 11440,
  have pascal : ∀ n k, comb (n + 1) (k + 1) = comb n (k + 1) + comb n k := sorry,
  have symmetry : ∀ n k, comb n k = comb n (n - k) := sorry,
  let h3 : comb 16 8 = comb 16 8 := by sorry, -- by property and known value
  let h4 : comb 16 8 = 12870 := by sorry, -- complementary counting
  let h5 : comb 17 10 = comb 16 9 + comb 16 8 := by apply pascal,
  let h6 : comb 17 9 = comb 16 8 + comb 16 7 := by apply pascal,
  let h7 : comb 17 10 = 11440 + comb 16 8 := by rw [h2, h3],
  let h8 : comb 17 9 = 11440 + comb 16 8 := by rw [h1, h3],
  let h9 : comb 18 10 = comb 17 10 + comb 17 9 := by apply pascal,
  rw [h7, h8, h4] at h9,
  exact h9

end comb_18_10_proof_l418_418655


namespace max_cos_sum_l418_418675

theorem max_cos_sum (x y : ℝ) (h : cos x - cos y = 1 / 4) : 
  ∃ M, M = 31 / 32 ∧ M = max (cos (x + y)) :=
sorry

end max_cos_sum_l418_418675


namespace prism_height_unique_l418_418187

theorem prism_height_unique (k : ℝ) : 
  (∃ b : ℕ × ℕ × ℝ, b = (6, 6, k) ∧ 
   ∃ r1 r2 : ℝ, 
     r1 = 3 ∧ 
     r2 = 1 ∧ 
     ∃ P : EuclideanSpace ℝ (Fin 3), 
       P = ![(1, 1, 1), ...(positions of other small spheres made symmetric)] ∧ 
       ∀ (small_sphere_center : EuclideanSpace ℝ (Fin 3)), 
         ∥small_sphere_center - large_sphere_center∥ = r1 + r2) 
  → k = 6 + 4 * Real.sqrt 2 := 
by
  sorry

end prism_height_unique_l418_418187


namespace reduce_to_single_digit_l418_418338

theorem reduce_to_single_digit (n : ℕ) : ∃ k ≤ 10, (λ f, ∀ m, n = f^[m] 0 → m ≤ 10) :=
begin
  sorry
end

end reduce_to_single_digit_l418_418338


namespace area_formula_l418_418373

-- Given heights m₁, m₂, and m₃, let's define them as non-negative real numbers
variables (m1 m2 m3 : ℝ)

-- We impose the condition that all m₁, m₂, and m₃ are positive
axiom h_m1 : 0 < m1
axiom h_m2 : 0 < m2
axiom h_m3 : 0 < m3

noncomputable def triangle_area (m1 m2 m3 : ℝ) : ℝ :=
  (m1 * m2 * m3) ^ 2 / Real.sqrt (
    (m1 * m2 + m1 * m3 + m2 * m3) *
    (m1 * m3 + m1 * m2 - m2 * m3) *
    (m2 * m3 + m1 * m2 - m1 * m3) *
    (m2 * m3 + m1 * m3 - m1 * m2))

theorem area_formula :
  ∃ t, t = triangle_area m1 m2 m3 :=
begin
  sorry
end

end area_formula_l418_418373


namespace parallelogram_perimeter_l418_418160

theorem parallelogram_perimeter 
  (EF FG EH : ℝ)
  (hEF : EF = 40) (hFG : FG = 30) (hEH : EH = 50) : 
  2 * (EF + FG) = 140 := 
by 
  rw [hEF, hFG]
  norm_num

end parallelogram_perimeter_l418_418160


namespace sum_intervals_total_length_l418_418059

theorem sum_intervals_total_length :
  let S := { x : ℝ | (∑ k in Finset.range 70, k / (x - k)) ≥ 5 / 4 }
  ∃ I : set (set ℝ), (∀ s ∈ I, ∃ a b, s = set.Icc a b) ∧ 
                      (⋃₀ I = S) ∧ 
                      (set.Union I).measure = 1988 := 
sorry

end sum_intervals_total_length_l418_418059


namespace sours_total_l418_418497

variable (c l o T : ℕ)

axiom cherry_sours : c = 32
axiom ratio_cherry_lemon : 4 * l = 5 * c
axiom orange_sours_ratio : o = 25 * T / 100
axiom total_sours : T = c + l + o

theorem sours_total :
  T = 96 :=
by
  sorry

end sours_total_l418_418497


namespace inequality_holds_l418_418043

section Proof

-- Definition of the real sequence
variable {x : ℕ → ℝ}

-- Assume x₁ = 1
def x_one : Prop := x 1 = 1

-- Main theorem statement
theorem inequality_holds (x_one : x_one) (n : ℕ) (hn : n ≥ 2) :
  (∑ i in divisors n, ∑ j in divisors n, (x i * x j) / nat.lcm i j) 
  ≥ (∏ p in nat.factors n, (1 - (1 / (p : ℝ)))) :=
sorry

end Proof

end inequality_holds_l418_418043


namespace divide_horseshoe_into_7_parts_with_holes_l418_418273

-- Definitions for the problem
constant paper_horseshoe : Type
constant has_hole : paper_horseshoe → Prop
constant is_cut : paper_horseshoe → paper_horseshoe → Prop
constant move_and_stack : paper_horseshoe → paper_horseshoe → Prop

-- The Lean statement for the proof problem
theorem divide_horseshoe_into_7_parts_with_holes :
  ∃ (pieces : list paper_horseshoe), 
    (∀ p ∈ pieces, has_hole p) ∧ 
    length pieces = 7 ∧ 
    ∃ (first_cut second_cut : paper_horseshoe), 
      (is_cut paper_horseshoe first_cut ∧ 
      move_and_stack first_cut second_cut ∧
      is_cut second_cut paper_horseshoe) := 
sorry

end divide_horseshoe_into_7_parts_with_holes_l418_418273


namespace trader_gain_percentage_l418_418643

theorem trader_gain_percentage 
  (C : ℝ) -- cost of each pen
  (h1 : 250 * C ≠ 0) -- ensure the cost of 250 pens is non-zero
  (h2 : 65 * C > 0) -- ensure the gain is positive
  (h3 : 250 * C + 65 * C > 0) -- ensure the selling price is positive
  : (65 / 250) * 100 = 26 := 
sorry

end trader_gain_percentage_l418_418643


namespace no_real_roots_f_f_eq_x_l418_418375

-- Defining the conditions
variables {a b c : ℝ}
variable (a_ne_zero : a ≠ 0)
variable (no_real_roots_f_eq_x : (b - 1) ^ 2 - 4 * a * c < 0)

-- Define the quadratic function f
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- The main theorem to be proved
theorem no_real_roots_f_f_eq_x : ∀ x : ℝ, f (f x) ≠ x :=
by
  sorry

end no_real_roots_f_f_eq_x_l418_418375


namespace xy_comm_n_minus_1_l418_418026

variable {R : Type*} [ring R]

-- The condition that x^n = x for all x in R where n > 2
def xn_eq_x (x : R) (n : ℕ) : Prop := x ^ n = x

theorem xy_comm_n_minus_1 (R : Type*) [ring R] (n : ℕ) (h : ∀ x : R, xn_eq_x x n) :
  ∀ x y : R, x * y ^ (n - 1) = y ^ (n - 1) * x :=
begin
  -- Conditions
  sorry
end

end xy_comm_n_minus_1_l418_418026


namespace mean_of_remaining_students_l418_418604

theorem mean_of_remaining_students (k : ℕ) (h_k : k > 15) 
  (h_mean_class : (∑ i in finset.range k, i) / k = 8) 
  (h_mean_fifteen : (∑ i in finset.Icc 1 15, 16) / 15 = 16) : 
  (∑ i in finset.range (k-15), i) / (k - 15) = (8 * k - 240) / (k - 15) :=
by
  sorry

end mean_of_remaining_students_l418_418604


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418936

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418936


namespace rain_total_duration_l418_418002

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end rain_total_duration_l418_418002


namespace hotel_room_charge_difference_l418_418972

-- Definitions based on the given conditions
variable (R G P : ℝ)

-- Standard single room charges:
-- P is 50% less than R
axiom h1 : P = 0.5 * R

-- P is 10% less than G
axiom h2 : P = 0.9 * G

-- Weekend discounts
-- P on weekend
noncomputable def P_weekend : ℝ := 0.93 * P

-- R on weekend
noncomputable def R_weekend : ℝ := 0.9 * R

-- Problem statement
theorem hotel_room_charge_difference :
  R = 1.15 * (P_weekend) :=
by sorry

end hotel_room_charge_difference_l418_418972


namespace pages_can_be_copied_l418_418828

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l418_418828


namespace parallelogram_contains_points_in_L_l418_418838

def L : Set (ℤ × ℤ) :=
  {p | ∃ x y : ℤ, p = (41 * x + 2 * y, 59 * x + 15 * y)}

theorem parallelogram_contains_points_in_L {P : Set (ℝ × ℝ)}
  (hP₁ : ∀ p ∈ P, (-p.1, -p.2) ∈ P)
  (hP₂ : ∀ (x y : ℝ), (x, y) ∈ P → (-x, -y) ∈ P)
  (hP₃ : convex ℝ P)
  (hP₄ : (0, 0) ∈ P)
  (hP₅ : area P = 1990) :
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ∈ P ∧ p₂ ∈ P ∧ p₁ ≠ p₂ ∧ p₁ ∈ L ∧ p₂ ∈ L :=
sorry

end parallelogram_contains_points_in_L_l418_418838


namespace intersecting_circles_through_midpoint_l418_418563

open EuclideanGeometry

/-- Given a circle \(S\) and a segment \(AB\) with \(C\) as the midpoint of the remaining arc of \(AB\) on \(S\),
two circles \(S_1\) and \(S_2\) inscribed in the segment \(AB\) intersect at points \(M\) and \(N\). Prove that
the line \(MN\) passes through the point \(C\). -/
theorem intersecting_circles_through_midpoint
  (S : Circle) (A B C : Point) (M N : Point)
  (inscribed_in_segment_AB_S1 : Circle) (inscribed_in_segment_AB_S2 : Circle)
  (H_midpoint_C : midpoint C A B)
  (H_inscribed_S1 : inscribed_in_segment S inscribed_in_segment_AB_S1 A B)
  (H_inscribed_S2 : inscribed_in_segment S inscribed_in_segment_AB_S2 A B)
  (H_intersect_MN : intersect_circles inscribed_in_segment_AB_S1 inscribed_in_segment_AB_S2 M N) :
  collinear C M N := sorry

end intersecting_circles_through_midpoint_l418_418563


namespace min_value_expr_min_value_achieved_l418_418711

variable {n : ℕ} {x : Fin n → ℝ} (h_pos : ∀ i, x i > 0) (h_cond : ∑ i, 1 / (1 + (x i)^2) = 1)

theorem min_value_expr : 
  (∑ i, x i) / (∑ i, 1 / (x i)) ≥ n - 1 := 
by sorry

theorem min_value_achieved :
  (∑ i, x i) / (∑ i, 1 / (x i)) = n - 1 ↔ ∀ i, x i = Real.sqrt (n - 1) :=
by sorry

end min_value_expr_min_value_achieved_l418_418711


namespace triangle_angle_B_sine_ratio_l418_418400

variables {A B C a b c : ℝ}

-- Proof Problem 1: B = π/3 under given conditions
theorem triangle_angle_B (h1 : ∠B = B) (h2 : ∠C = C) (h3 : side A = a) (h4 : side B = b) (h5 : side C = c)
  (h6 : (2 * a - c) / b = cos C / cos B) : B = π/3 := 
  sorry

-- Proof Problem 2: sin C / sin A = 3 / 2 under given conditions
theorem sine_ratio (h1 : ∠A = A) (h2 : ∠B = B) (h3 : ∠C = C) (h4 : side A = a) (h5 : side B = b) (h6 : side C = c)
  (h7 : midpoint M B C) (h8 : AM = AC) : (sin C / sin A) = 3 / 2 := 
  sorry

end triangle_angle_B_sine_ratio_l418_418400


namespace remainder_of_power_mod_remainder_17_pow_53_l418_418161

theorem remainder_of_power_mod :
  ∀ (n : ℕ), (17 ^ n) % 5 = (2 ^ n) % 5 :=
begin
  intros n,
  have h1: 17 % 5 = 2 % 5, from rfl,
  have h2: (17 ^ n) % 5 = ((2 + 15) ^ n) % 5, by rw nat.mod_eq_of_lt h1,
  have h3: (2 ^ n) % 5 = ((2 + 15) ^ n) % 5, by rw nat.mod_eq_of_lt h1,
  have h4: ∀ (x y n : ℕ), ((x + y) ^ n) % 5 = (x ^ n) % 5,
  intros x y n,
  induction n with n hn,
  { simp },
  { rw [nat.add_pow_succ],
    apply nat.add_mod,
    apply hn },
  have h5 : ((2 + 15) ^ n) % 5 = (2 ^ n) % 5,
  { apply h4 },
  exact h5,
end

theorem remainder_17_pow_53 : (17 ^ 53) % 5 = 2 :=
by {
  rw remainder_of_power_mod,
  have h6: 53 % 4 = 1, from by norm_num,
  have h7: (2 ^ 53) % 5 = 2, from by sorry,
  exact h7
}

end remainder_of_power_mod_remainder_17_pow_53_l418_418161


namespace cost_per_litre_mixed_fruit_juice_l418_418120

theorem cost_per_litre_mixed_fruit_juice :
  let cost_superfruit := 1399.45
  let cost_acai := 3104.35
  let volume_mixed_fruit := 34.0
  let volume_acai := 22.666666666666668
  let total_volume := volume_mixed_fruit + volume_acai
  let total_cost_acai := volume_acai * cost_acai
  let total_cost := total_volume * cost_superfruit
  let C := (total_cost - total_cost_acai) / volume_mixed_fruit in
  C = 264.18 :=
by
  sorry

end cost_per_litre_mixed_fruit_juice_l418_418120


namespace compute_H_iterated_l418_418614

noncomputable def H : ℝ → ℝ := sorry  -- This definition will only ensure the type but no explicit definition
axiom H_of_2 : H 2 = 2

theorem compute_H_iterated :
  H (H (H (H (H 2)))) = 2 :=
by
  rw [H_of_2, H_of_2, H_of_2, H_of_2, H_of_2]
  sorry

end compute_H_iterated_l418_418614


namespace min_sum_value_l418_418839

noncomputable def min_sum (P : ℝ[X]) (hP : P ≠ 0) (coeff_nonneg : ∀ n, 0 ≤ P.coeff n) (N : ℕ) (hN : 0 < N) 
  (σ : Equiv.Perm (Fin N)) : ℝ := 
  ∑ i in Finset.range N, 
    (P.eval (i : ℝ) ^ 2) / (P.eval ((i : ℝ) * (σ i : ℝ)))

theorem min_sum_value {P : ℝ[X]} (hP : P ≠ 0) (coeff_nonneg : ∀ n, 0 ≤ P.coeff n) (N : ℕ) (hN : 0 < N) 
  (σ : Equiv.Perm (Fin N)) : 
  (∀ x : Fin N → ℝ, (0 < x) → min_sum P hP coeff_nonneg N hN σ ≥ N) :=
begin
  sorry
end

end min_sum_value_l418_418839


namespace proof_problem_l418_418129

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

noncomputable def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem proof_problem 
  (a b S T : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_arithmetic : arithmetic_sequence b)
  (h_a1 : a 1 = 1 / 2)
  (h_a3_a2 : 1 / a 3 = 1 / a 2 + 4)
  (h_a3_b4_b6 : a 3 = 1 / (b 4 + b 6))
  (h_a4_b5_b7 : a 4 = 1 / (b 5 + 2 * b 7))
  (h_S : ∀ n, S n = (a 1 * (1 - (geometric_sequence.q) ^ n) / (1 - geometric_sequence.q)))
  (h_T : ∀ n, T n = n - 1 + 1 / (2^n)) :
  (∀ n, a n = 1 / (2^n)) ∧ (∀ n, b n = n - 1) ∧ 
  ∀ n, ∑ i in finset.range n, ((T (i + 1) - b (i + 1)) * b (i + 3)) / (b (i + 1) * b (i + 2)) < 1 / 2 :=
sorry

end proof_problem_l418_418129


namespace complement_intersection_l418_418377

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3, 4}
def A_complement : Set ℕ := U \ A

theorem complement_intersection :
  (A_complement ∩ B) = {2, 4} :=
by 
  sorry

end complement_intersection_l418_418377


namespace simple_sampling_incorrect_l418_418630

-- Definitions based on the conditions
def simple_sampling (s : Set ℕ) : Prop :=
  ∀ x ∈ s, ∃ p : ℕ → ℝ, p x = 1 / (s.card : ℝ)

def systematic_sampling (s : Set ℕ) : Prop :=
  ∀ x ∈ s, ∃ p : ℕ → ℝ, p x = 1 / (s.card : ℝ)

def stratified_sampling (s : Set ℕ) : Prop :=
  ∀ x ∈ s, ∃ p : ℕ → ℝ, ∀ stratum ∈ s, p stratum = 1 / (s.card : ℝ)

def principle_of_sampling (s : Set ℕ) : Prop :=
  ∀ x ∈ s, ∃ p : ℕ → ℝ, p x = 1 / (s.card : ℝ)

-- We are to prove that the description about simple sampling in option A is incorrect
theorem simple_sampling_incorrect (s : Set ℕ) :
  simple_sampling s ≠ systematic_sampling s :=
sor.types.bool

end simple_sampling_incorrect_l418_418630


namespace prime_square_remainders_l418_418257

theorem prime_square_remainders (p : ℕ) (hp : Prime p) (hpg : p > 5) : 
  ∃ n : ℕ, n = 2 ∧ ∀ r ∈ {r : ℕ | ∃ p : ℕ, Prime p ∧ p > 5 ∧ r = (p ^ 2 % 180)}, r ∈ {1, 64} :=
by sorry

end prime_square_remainders_l418_418257


namespace john_can_fix_l418_418020

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end john_can_fix_l418_418020


namespace polyomino_count_5_l418_418569

-- Definition of distinct polyomino counts for n = 2, 3, and 4.
def polyomino_count (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 5
  else 0

-- Theorem stating the distinct polyomino count for n = 5
theorem polyomino_count_5 : polyomino_count 5 = 12 :=
by {
  -- Proof steps would go here, but for now we use sorry.
  sorry
}

end polyomino_count_5_l418_418569


namespace planned_pencils_is_49_l418_418890

def pencils_planned (x : ℕ) : ℕ := x
def pencils_bought (x : ℕ) : ℕ := x + 12
def total_pencils_bought (x : ℕ) : ℕ := 61

theorem planned_pencils_is_49 (x : ℕ) :
  pencils_bought (pencils_planned x) = total_pencils_bought x → x = 49 :=
sorry

end planned_pencils_is_49_l418_418890


namespace cost_of_balls_max_basketball_count_l418_418789

-- Define the prices of basketball and soccer ball
variables (x y : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := 2 * x + 3 * y = 310
def condition2 : Prop := 5 * x + 2 * y = 500

-- Proving the cost of each basketball and soccer ball
theorem cost_of_balls (h1 : condition1 x y) (h2 : condition2 x y) : x = 80 ∧ y = 50 :=
sorry

-- Define the total number of balls and the inequality constraint
variable (m : ℕ)
def total_balls_condition : Prop := m + (60 - m) = 60
def cost_constraint : Prop := 80 * m + 50 * (60 - m) ≤ 4000

-- Proving the maximum number of basketballs
theorem max_basketball_count (hc : cost_constraint m) (ht : total_balls_condition m) : m ≤ 33 :=
sorry

end cost_of_balls_max_basketball_count_l418_418789


namespace find_fe_value_l418_418694

noncomputable def f (x : ℝ) := (deriv f 1) + x * Real.log x

theorem find_fe_value : f(Real.exp 1) = 1 + Real.exp 1 := by
  sorry

end find_fe_value_l418_418694


namespace shortest_closed_broken_lines_exist_l418_418259

open Set

variables {A B C D X Y Z T : Type*}

-- Definitions of points and angles
variables (points_on_edges : ∀ {l m: Type*}, l = m)
            (acute_faces : ∀ {face : Type*}, acute face)
            (angle_DAB angle_BCD angle_ABC angle_CDA angle_BAC angle_CAD : ℝ)
            (alpha := angle_BAC + angle_CAD + angle_DAB)
            
-- Define the condition given in the statement
variables (condition : angle_DAB + angle_BCD = angle_ABC + angle_CDA)

-- Define AC being a segment length
variables (length_AC : ℝ)

-- Define length
def closed_polygonal_chain_length := 2 * length_AC * Real.sin (alpha / 2)

-- The equivalent proof problem
theorem shortest_closed_broken_lines_exist : 
  ∃∞ (p : Set ℝ), 
    (closed_polygonal_chain_length points_on_edges acute_faces condition length_AC) :=
sorry

end shortest_closed_broken_lines_exist_l418_418259


namespace simplify_fraction_l418_418089

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l418_418089


namespace joshua_needs_2_5_more_usd_l418_418022

theorem joshua_needs_2_5_more_usd (usd_to_chf : ℝ)
                                  (eur_to_chf : ℝ)
                                  (initial_usd : ℝ)
                                  (initial_eur : ℝ)
                                  (pen_cost_chf : ℝ)
                                  (converted_eur_to_chf : initial_eur * eur_to_chf = 15.75)
                                  (needed_chf : pen_cost_chf - (initial_eur * eur_to_chf) = 2.25)
                                  (usd_needed : (pen_cost_chf - (initial_eur * eur_to_chf)) / usd_to_chf = 2.5)
                                  :
                                  usd_needed = 2.5 :=
by
  sorry

-- Define the initial data
def initial_usd := 20 : ℝ
def initial_eur := 15 : ℝ
def pen_cost_chf := 18 : ℝ
def usd_to_chf := 0.9 : ℝ
def eur_to_chf := 1.05 : ℝ

-- Create concrete values for the calculated conversions based on conditions
def converted_eur_to_chf := eq.refl 15.75
def needed_chf := eq.refl 2.25
def usd_needed := eq.refl 2.5

noncomputable example : joshua_needs_2_5_more_usd usd_to_chf eur_to_chf initial_usd initial_eur pen_cost_chf converted_eur_to_chf needed_chf usd_needed = (2.5 : ℝ) :=
by
  sorry

end joshua_needs_2_5_more_usd_l418_418022


namespace count_squares_containing_A_l418_418977

-- Given conditions
def figure_with_squares : Prop := ∃ n : ℕ, n = 20

-- The goal is to prove that the number of squares containing A is 13
theorem count_squares_containing_A (h : figure_with_squares) : ∃ k : ℕ, k = 13 :=
by 
  sorry

end count_squares_containing_A_l418_418977


namespace minimum_area_of_triangle_AOC_is_28_over_3_l418_418485

noncomputable def minimum_area_triangle_AOC : ℝ :=
  let A (x₁ : ℝ) := (x₁, 3 * x₁) in
  let C (k : ℝ) := (2 / k + 3, 0) in 
  let area (k : ℝ) := (3 * (3 * k - 2) * (3 * k - 10) / (k * (k - 3))) / 2 in
  let valid_k := λ k, 2 / 3 < k ∧ k < 3 in
  Sup (set.image area {k | valid_k k})

theorem minimum_area_of_triangle_AOC_is_28_over_3 :
  minimum_area_triangle_AOC = 28 / 3 :=
sorry

end minimum_area_of_triangle_AOC_is_28_over_3_l418_418485


namespace normal_distribution_symmetry_l418_418358

noncomputable def a_value (σ : ℝ) (a : ℝ) : Prop :=
  ∀ (ξ : ℝ), (ξ ∼ Normal 1 σ) → (P(ξ > a) = 0.5) → (a = 1)

theorem normal_distribution_symmetry (σ : ℝ) (ξ : ℝ) :
  (ξ ∼ Normal 1 σ) → (P(ξ > a) = 0.5) → (a = 1) :=
by
  sorry

end normal_distribution_symmetry_l418_418358


namespace pages_can_be_copied_l418_418825

theorem pages_can_be_copied (dollars : ℕ) (cost_per_page_cents : ℕ) (conversion_rate : ℕ):
  dollars = 15 → cost_per_page_cents = 3 → conversion_rate = 100 → 
  ((dollars * conversion_rate) / cost_per_page_cents = 500) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  exact rfl

end pages_can_be_copied_l418_418825


namespace simplify_fraction_l418_418077

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l418_418077


namespace find_100d_l418_418850

open Real

noncomputable def b_seq : ℕ → ℝ
| 0     := 7 / 17
| (n+1) := 2 * (b_seq n)^2 - 1

def sequence_condition (n : ℕ) : Prop :=
  abs (∏ i in Finset.range n, b_seq i) ≤ 17 / (3^n * sqrt 210)

theorem find_100d :
  (∃ d, ∀ n, abs (∏ i in Finset.range n, b_seq i) ≤ d / 3^n) → 100 * 17 / sqrt 210 = 56 :=
by
  intro h
  have d := 17 / sqrt 210
  use d
  sorry

end find_100d_l418_418850


namespace susan_gave_sean_8_apples_l418_418947

theorem susan_gave_sean_8_apples (initial_apples total_apples apples_given : ℕ) 
  (h1 : initial_apples = 9)
  (h2 : total_apples = 17)
  (h3 : apples_given = total_apples - initial_apples) : 
  apples_given = 8 :=
by
  sorry

end susan_gave_sean_8_apples_l418_418947


namespace locus_of_midpoints_of_right_angled_triangles_l418_418231

theorem locus_of_midpoints_of_right_angled_triangles 
  (A B C D : Point)
  (h_square : isSquare A B C D)
  (X Y Z : Point)
  (hX : onSide X A B)
  (hY : onSide Y A D)
  (hZ : onSide Z B C)
  (h_need_condition : ∃ (X Y Z : Point), X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z): 
  ∃ O, isMidpoint O X Y Z ∧ lies_in_curvilinear_octagon O :=
sorry

end locus_of_midpoints_of_right_angled_triangles_l418_418231


namespace solve_system_of_equations_l418_418515

theorem solve_system_of_equations (a1 a2 a3 a4 : ℝ) (h_distinct: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :
  ∃ x1 x2 x3 x4 : ℝ, 
    (|a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1) ∧
    (|a2 - a1| * x1 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1) ∧
    (|a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a4| * x4 = 1) ∧
    (|a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 = 1) ∧
    (x1 = 1 / |a1 - a4| ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / |a1 - a4|) :=
by {
  use 1 / |a1 - a4|, 0, 0, 1 / |a1 - a4|,
  split, 
  sorry,
}

end solve_system_of_equations_l418_418515


namespace sum_m_squared_multiples_l418_418518

noncomputable theory

def sum_of_arithmetic_series (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def sum_first_m_multiples (m n : ℕ) : ℕ :=
  sum_of_arithmetic_series n n m

def sum_first_m_cubed_multiples (m n : ℕ) : ℕ :=
  sum_of_arithmetic_series (n^3) (n^3) (m^3)

def sum_first_m_squared_multiples (m n : ℕ) : ℕ :=
  sum_of_arithmetic_series (n^2) (n^2) (m^2)

theorem sum_m_squared_multiples (m n : ℕ) (h1 : sum_first_m_multiples m n = 120) 
  (h2 : sum_first_m_cubed_multiples m n = 4032000) :
  sum_first_m_squared_multiples m n = 20800 := by
  sorry

end sum_m_squared_multiples_l418_418518


namespace smaller_angle_at_3_45_l418_418260

def minute_hand_angle : ℝ := 270
def hour_hand_angle : ℝ := 90 + 0.75 * 30

theorem smaller_angle_at_3_45 :
  min (|minute_hand_angle - hour_hand_angle|) (360 - |minute_hand_angle - hour_hand_angle|) = 202.5 := 
by
  sorry

end smaller_angle_at_3_45_l418_418260


namespace a_alone_can_finish_job_l418_418584

def work_in_one_day (A B : ℕ) : Prop := 1/A + 1/B = 1/40

theorem a_alone_can_finish_job (A B : ℕ)
  (work_rate : work_in_one_day A B) 
  (together_10_days : 10 * (1/A + 1/B) = 1/4) 
  (a_21_days : 21 * (1/A) = 3/4) : 
  A = 28 := 
sorry

end a_alone_can_finish_job_l418_418584


namespace assignments_divisible_by_17_l418_418221

/-- Given 17 workers and each worker must be part of some brigades where each brigade is a
    contiguous group of at least 2 workers, we need to assign leaders to brigades such
    that each worker is a leader of some brigades and the number of assignments for each worker 
    is divisible by 4. Prove that the number of possible ways to do this is divisible by 17. -/
theorem assignments_divisible_by_17 :
  ∃ (assignments : Fin 17 → Fin 4 → Fin 17), 
    (∀ i : Fin 17, (∑ j : Fin 4, 1) % 4 = 0) →
    (∃ k : Fin 17, assignments k 0 + assignments k 1 + assignments k 2 + assignments k 3 = 0) ∧
    (∃ n : ℕ, n % 17 = 0) :=
by
  sorry

end assignments_divisible_by_17_l418_418221


namespace complex_properties_l418_418457

def z := Complex
def a := Real
def b := Real
def m := Real
def n := Real

theorem complex_properties (z : Complex)
  (h1 : |z| = 2 * Real.sqrt 5)
  (h2 : (1 + 2 * Complex.i) * z).im = 0
  (h3 : 0 < z.re)
  (h4 : ∃ a b, z = a + b * Complex.i ∧ a^2 + b^2 = 20 ∧ a - 2 * b = 0)
  (root_of_eq : polynomial.aeval z (polynomial.C0 + polynomial.Cn * X + X^2) = 0)
  : z = 4 + 2 * Complex.i ∧ ∃ m n, root_of_eq ∧ m = -8 ∧ n = 20 :=
  sorry

end complex_properties_l418_418457


namespace solve_arcsin_equation_l418_418509

noncomputable def solution (x : ℝ) : Prop :=
  arcsin x + arcsin (3 * x) = π / 4

theorem solve_arcsin_equation :
  solution (sqrt (2 / 51)) :=
by sorry

end solve_arcsin_equation_l418_418509


namespace redesigned_survey_response_l418_418622

theorem redesigned_survey_response :
  let original_responded := 7 in
  let original_sent := 70 in
  let redesigned_sent := 63 in
  let original_response_rate := (original_responded / original_sent : ℝ) * 100 in
  let response_increase := 4 in
  let redesigned_response_rate := original_response_rate + response_increase in
  let expected_responded := (redesigned_response_rate / 100) * redesigned_sent in
  Int.round expected_responded = 9 :=
by
  let original_responded := 7
  let original_sent := 70
  let redesigned_sent := 63
  let original_response_rate := (original_responded / original_sent : ℝ) * 100
  let response_increase := 4
  let redesigned_response_rate := original_response_rate + response_increase
  let expected_responded := (redesigned_response_rate / 100) * redesigned_sent
  show Int.round expected_responded = 9
  sorry

end redesigned_survey_response_l418_418622


namespace probability_heads_odd_l418_418245

theorem probability_heads_odd (n : ℕ) (p : ℚ) (Q : ℕ → ℚ) (h : p = 3/4) (h_rec : ∀ n, Q (n + 1) = p * (1 - Q n) + (1 - p) * Q n) :
  Q 40 = 1/2 * (1 - 1/4^40) := 
sorry

end probability_heads_odd_l418_418245


namespace prop_converse_inverse_contrapositive_correct_statements_l418_418122

-- Defining the proposition and its types
def prop (x : ℕ) : Prop := x > 0 → x^2 ≥ 0
def converse (x : ℕ) : Prop := x^2 ≥ 0 → x > 0
def inverse (x : ℕ) : Prop := ¬ (x > 0) → x^2 < 0
def contrapositive (x : ℕ) : Prop := x^2 < 0 → ¬ (x > 0)

-- The proof problem
theorem prop_converse_inverse_contrapositive_correct_statements :
  (∃! (p : Prop), p = (∀ x : ℕ, converse x) ∨ p = (∀ x : ℕ, inverse x) ∨ p = (∀ x : ℕ, contrapositive x) ∧ p = True) :=
sorry

end prop_converse_inverse_contrapositive_correct_statements_l418_418122


namespace monotonic_intervals_max_min_values_l418_418185

noncomputable def f : ℝ → ℝ := λ x, (1 / 3) ^ x
noncomputable def f_inv : ℝ → ℝ := λ y, Real.log y / Real.log (1 / 3)
noncomputable def g : ℝ → ℝ := λ x, f_inv (x^2 + 2 * x - 3)

theorem monotonic_intervals :
  (∀ x, x > 1 → StrictMono g) ∧ (∀ x, x < -3 → StrictAnti g) :=
by sorry

theorem max_min_values :
  ∀ x, x ∈ Icc (-1 : ℝ) 1 →
  2 ≤ (f x) ^ 2 - 2 * (f x) + 3 ∧ (f x) ^ 2 - 2 * (f x) + 3 ≤ 6 :=
by sorry

end monotonic_intervals_max_min_values_l418_418185


namespace combined_yellow_blue_correct_l418_418476

-- Declare the number of students in the class
def total_students : ℕ := 200

-- Declare the percentage of students who like blue
def percent_like_blue : ℝ := 0.3

-- Declare the percentage of remaining students who like red
def percent_like_red : ℝ := 0.4

-- Function that calculates the number of students liking a certain color based on percentage
def students_like_color (total : ℕ) (percent : ℝ) : ℕ :=
  (percent * total).toInt

-- Calculate the number of students who like blue
def students_like_blue : ℕ :=
  students_like_color total_students percent_like_blue

-- Calculate the number of students who don't like blue
def students_not_like_blue : ℕ :=
  total_students - students_like_blue

-- Calculate the number of students who like red from those who don't like blue
def students_like_red : ℕ :=
  students_like_color students_not_like_blue percent_like_red

-- Calculate the number of students who like yellow (those who don't like blue or red)
def students_like_yellow : ℕ :=
  students_not_like_blue - students_like_red

-- The combined number of students who like yellow and blue
def combined_yellow_blue : ℕ :=
  students_like_blue + students_like_yellow

-- Theorem to prove that the combined number of students liking yellow and blue is 144
theorem combined_yellow_blue_correct : combined_yellow_blue = 144 := by
  sorry

end combined_yellow_blue_correct_l418_418476


namespace precious_more_correct_l418_418876

theorem precious_more_correct (n : ℕ) (h₁ : n = 75)
    (h₂ : ∃ x, x = 0.20 * 75 ∧ x = 15)
    (h₃ : 12 < 75) :
    let correct_answers_Lyssa := n - 15,
        correct_answers_Precious := n - 12
    in correct_answers_Precious - correct_answers_Lyssa = 3 := by
  sorry

end precious_more_correct_l418_418876


namespace solve_arcsin_eq_l418_418512

theorem solve_arcsin_eq (x : ℝ) :
  arcsin x + arcsin (3 * x) = π / 4 → x = sqrt 102 / 51 ∨ x = -sqrt 102 / 51 :=
by 
  sorry

end solve_arcsin_eq_l418_418512


namespace maxwells_walking_speed_l418_418470

theorem maxwells_walking_speed 
    (brad_speed : ℕ) 
    (distance_between_homes : ℕ) 
    (maxwell_distance : ℕ)
    (meeting : maxwell_distance = 12)
    (brad_speed_condition : brad_speed = 6)
    (distance_between_homes_condition: distance_between_homes = 36) : 
    (maxwell_distance / (distance_between_homes - maxwell_distance) * brad_speed ) = 3 := by
  sorry

end maxwells_walking_speed_l418_418470


namespace Sophie_saves_by_using_wool_dryer_balls_l418_418098

-- Definitions for the given conditions
def weekly_loads := [4, 5, 6, 7]
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def wool_dryer_balls_cost : ℝ := 15
def yearly_increase_rate : ℝ := 2.5 / 100 
def period_years : ℕ := 2

-- Calculate monthly loads
def monthly_loads : ℕ := weekly_loads.sum

-- Calculate yearly loads
def yearly_loads : ℕ := monthly_loads * 12

-- Calculate boxes required per year (rounded up)
def boxes_per_year : ℕ := (yearly_loads + sheets_per_box - 1) / sheets_per_box

-- Calculate cost of dryer sheets in the first and second years
def first_year_cost : ℝ := boxes_per_year * cost_per_box
def second_year_cost : ℝ := boxes_per_year * (cost_per_box * (1 + yearly_increase_rate))

-- Total cost of dryer sheets over two years
def total_cost_dryer_sheets : ℝ := first_year_cost + second_year_cost

-- Calculate total savings
def total_savings : ℝ := total_cost_dryer_sheets - wool_dryer_balls_cost

-- The proof statement in Lean 4
theorem Sophie_saves_by_using_wool_dryer_balls :
  total_savings = 18.4125 := by
  sorry

end Sophie_saves_by_using_wool_dryer_balls_l418_418098


namespace palindromic_clock_l418_418222

def is_palindromic_time (a b c d m n : ℕ) : Prop :=
  a = n ∧ b = m ∧ c = d

/-- The total number of palindromic sequences in a 24-hour day (from 00:00:00 to 23:59:59) is 144. -/
theorem palindromic_clock (h: ∀ a b c d m n, is_palindromic_time a b c d m n → 
                             (a = 0 ∨ a = 1 ∨ (a = 2 ∧ b ≤ 3)) ∧ 
                             (b ≤ 9) ∧ 
                             (c ≤ 5) ∧ 
                             (d ≤ 9) ∧ 
                             (m ≤ 5) ∧ 
                             (n ≤ 9)) 
    : ∃ count : ℕ, count = 144 := 
begin
  sorry
end

end palindromic_clock_l418_418222


namespace largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418934

theorem largest_two_digit_number_divisible_by_6_and_ends_in_4 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n % 6 = 0 ∧ n % 10 = 4 ∧ n = 84 :=
by
  sorry

end largest_two_digit_number_divisible_by_6_and_ends_in_4_l418_418934


namespace hockey_pads_cost_l418_418799

theorem hockey_pads_cost
  (initial_money : ℕ)
  (cost_hockey_skates : ℕ)
  (remaining_money : ℕ)
  (h : initial_money = 150)
  (h1 : cost_hockey_skates = initial_money / 2)
  (h2 : remaining_money = 25) :
  initial_money - cost_hockey_skates - 50 = remaining_money :=
by sorry

end hockey_pads_cost_l418_418799


namespace Vincent_sells_8_literature_books_per_day_l418_418570

theorem Vincent_sells_8_literature_books_per_day
  (fantasy_book_cost : ℕ)
  (literature_book_cost : ℕ)
  (fantasy_books_sold_per_day : ℕ)
  (total_earnings_5_days : ℕ)
  (H_fantasy_book_cost : fantasy_book_cost = 4)
  (H_literature_book_cost : literature_book_cost = 2)
  (H_fantasy_books_sold_per_day : fantasy_books_sold_per_day = 5)
  (H_total_earnings_5_days : total_earnings_5_days = 180) :
  ∃ L : ℕ, L = 8 :=
by
  sorry

end Vincent_sells_8_literature_books_per_day_l418_418570


namespace not_exists_18_consecutive_good_l418_418871

def is_good (n : ℕ) : Prop := 
  (∀ p : ℕ, nat.prime p → p ∣ n → nat.factor_multiset p n = 2) 

theorem not_exists_18_consecutive_good :
  ¬ ∃ (a : ℕ), ∀ (i : ℕ), i < 18 → is_good (a + i) :=
sorry

end not_exists_18_consecutive_good_l418_418871


namespace smallest_prime_factor_1547_l418_418576

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, prime p ∧ p ∣ n then Nat.find h else n

theorem smallest_prime_factor_1547 : smallest_prime_factor 1547 = 7 :=
by
  sorry

end smallest_prime_factor_1547_l418_418576


namespace no_18_consecutive_good_numbers_l418_418867

def is_good (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, (p1.prime ∧ p2.prime ∧ p1 ≠ p2 ∧ n = p1 * p2)

theorem no_18_consecutive_good_numbers :
  ¬ ∃ (a : ℕ), ∀ i : ℕ, i < 18 → is_good (a + i) :=
sorry

end no_18_consecutive_good_numbers_l418_418867


namespace f_range_l418_418757

def f (x : ℝ) : ℝ := x^2 + real.sqrt (1 - x^2)

theorem f_range :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → 1 ≤ f x ∧ f x ≤ 5 / 4 :=
by
  sorry

end f_range_l418_418757


namespace smallest_possible_value_of_c_l418_418640

theorem smallest_possible_value_of_c
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (H : ∀ x : ℝ, (a * Real.sin (b * x + c)) ≤ (a * Real.sin (b * 0 + c))) :
  c = Real.pi / 2 :=
by
  sorry

end smallest_possible_value_of_c_l418_418640


namespace expression_for_f_general_formula_a_n_sum_S_n_l418_418704

-- Definitions for conditions
def f (x : ℝ) : ℝ := x^2 + x

-- Given conditions
axiom f_zero : f 0 = 0
axiom f_recurrence : ∀ x : ℝ, f (x + 1) - f x = x + 1

-- Statements to prove
theorem expression_for_f (x : ℝ) : f x = x^2 + x := 
sorry

theorem general_formula_a_n (t : ℝ) (n : ℕ) (H : 0 < t) : 
    ∃ a_n : ℕ → ℝ, a_n n = t^n := 
sorry

theorem sum_S_n (t : ℝ) (n : ℕ) (H : 0 < t) :
    ∃ S_n : ℕ → ℝ, (S_n n = if t = 1 then ↑n else (t * (t^n - 1)) / (t - 1)) := 
sorry

end expression_for_f_general_formula_a_n_sum_S_n_l418_418704


namespace sum_of_odd_terms_l418_418706

noncomputable def S (n : ℕ) : ℕ := n^2 + 2*n - 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem sum_of_odd_terms : (Finset.range 13).sum (λ i, a (2*i + 1)) = 350 :=
by
  -- The body of the proof is intentionally left as sorry 
  -- since the problem specifies that no solution steps should be included.
  sorry

end sum_of_odd_terms_l418_418706


namespace sum_S_18_34_51_l418_418852

def S (n : ℕ) : ℤ :=
  if even n then -n else n + 1

theorem sum_S_18_34_51 : S 18 + S 34 + S 51 = 0 :=
by
  unfold S
  simp [nat.even, nat.odd]
  sorry

end sum_S_18_34_51_l418_418852


namespace fraction_power_identity_l418_418503

-- Definitions for the given conditions.
variables (a b c : ℝ) (n : ℕ)

-- Hypothesis representing the given condition.
def condition1 : Prop := 1/a + 1/b + 1/c = 1/(a + b + c)
def condition2 : Prop := odd n  ∧ n > 0

-- Statement to be proved.
theorem fraction_power_identity (h1 : condition1 a b c) (h2 : condition2 n) :
  1/(a^n) + 1/(b^n) + 1/(c^n) = 1/(a^n + b^n + c^n) :=
by sorry

end fraction_power_identity_l418_418503


namespace max_pairs_300_grid_l418_418466

noncomputable def max_pairs (n : ℕ) (k : ℕ) (remaining_squares : ℕ) [Fintype (Fin n × Fin n)] : ℕ :=
  sorry

theorem max_pairs_300_grid :
  max_pairs 300 100 50000 = 49998 :=
by
  -- problem conditions
  let grid_size := 300
  let corner_size := 100
  let remaining_squares := 50000
  let no_checkerboard (squares : Fin grid_size × Fin grid_size → Prop) : Prop :=
    ∀ i j, ¬(squares (i, j) ∧ squares (i + 1, j) ∧ squares (i, j + 1) ∧ squares (i + 1, j + 1))
  -- statement of the bound
  have max_pairs := max_pairs grid_size corner_size remaining_squares
  exact sorry

end max_pairs_300_grid_l418_418466


namespace derangement_formula_l418_418224

open Nat

noncomputable def derangement (n : ℕ) : ℕ :=
  n! * (∑ i in finset.range (n + 1), (-1) ^ i / i!)

theorem derangement_formula (n : ℕ) :
  derangement n = n! * (∑ i in finset.range (n + 1), (-1) ^ i / i!) :=
sorry

end derangement_formula_l418_418224


namespace difference_in_profit_l418_418483

def records := 300
def price_sammy := 4
def price_bryan_two_thirds := 6
def price_bryan_one_third := 1
def price_christine_thirty := 10
def price_christine_remaining := 3

def profit_sammy := records * price_sammy
def profit_bryan := ((records * 2 / 3) * price_bryan_two_thirds) + ((records * 1 / 3) * price_bryan_one_third)
def profit_christine := (30 * price_christine_thirty) + ((records - 30) * price_christine_remaining)

theorem difference_in_profit : 
  max profit_sammy (max profit_bryan profit_christine) - min profit_sammy (min profit_bryan profit_christine) = 190 :=
by
  sorry

end difference_in_profit_l418_418483


namespace division_of_powers_l418_418583

variable {a : ℝ}

theorem division_of_powers (ha : a ≠ 0) : a^5 / a^3 = a^2 :=
by sorry

end division_of_powers_l418_418583


namespace male_students_stratified_sampling_l418_418216

theorem male_students_stratified_sampling :
  ∀ (total_students total_female_students sample_size : ℕ)
  (sampling_ratio : ℚ)
  (total_male_students : ℕ),
  total_students = 900 →
  total_female_students = 400 →
  sample_size = 45 →
  sampling_ratio = sample_size /. total_students →
  total_male_students = total_students - total_female_students →
  (total_male_students /. 20) = 25 :=
by
  intros total_students total_female_students sample_size sampling_ratio total_male_students
  assume h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end male_students_stratified_sampling_l418_418216


namespace shaded_area_fraction_l418_418417

theorem shaded_area_fraction (A B C D E F P : Type)
  (equilateral_triangle : ∀ {ABC : Triangle Type}, is_equilateral ABC)
  (midpoints : midpoint D A B ∧ midpoint E B C ∧ midpoint F C A)
  (center_P : is_center P D E F A B C) :
  let S := area ABC in
  shaded_area S P D E F =
  (5 / 24) * S :=
sorry

end shaded_area_fraction_l418_418417


namespace eccentricity_condition_l418_418549

theorem eccentricity_condition (m : ℝ) (h : 0 < m) : 
  (m < (4 / 3) ∨ m > (3 / 4)) ↔ ((1 - m) > (1 / 4) ∨ ((m - 1) / m) > (1 / 4)) :=
by
  sorry

end eccentricity_condition_l418_418549


namespace hockey_pads_cost_l418_418800

variable (x h p remaining : ℝ)

theorem hockey_pads_cost :
  x = 150 ∧ h = x / 2 ∧ remaining = x - h - p ∧ remaining = 25 → p = 50 :=
by
  intro h₁
  cases h₁ with hx hh
  cases hh with hh₁ hh₂
  cases hh₂ with hr hr₁
  sorry

end hockey_pads_cost_l418_418800


namespace max_range_of_temps_l418_418181

noncomputable def max_temp_range (T1 T2 T3 T4 T5 : ℝ) : ℝ := 
  max (max (max (max T1 T2) T3) T4) T5 - min (min (min (min T1 T2) T3) T4) T5

theorem max_range_of_temps :
  ∀ (T1 T2 T3 T4 T5 : ℝ), 
  (T1 + T2 + T3 + T4 + T5) / 5 = 60 →
  T1 = 40 →
  (max_temp_range T1 T2 T3 T4 T5) = 100 :=
by
  intros T1 T2 T3 T4 T5 Havg Hlowest
  sorry

end max_range_of_temps_l418_418181


namespace real_part_complex_inv_l418_418463

theorem real_part_complex_inv (θ : ℝ) (z : ℂ) (hz : z = complex.exp (complex.I * θ)) (hz_mod : complex.abs z = 1) :
  complex.re (1 / (2 - z)) = (2 - real.cos θ) / (5 - 4 * real.cos θ) :=
sorry

end real_part_complex_inv_l418_418463


namespace num_permutations_l418_418234

noncomputable def max_permutations_tree : ℕ :=
  2^128

def ten_level_tree (X Y : Type) (P : X → Y) : Prop := 
∃ (A₁ : X) (B C : X) (J : Y), 
  (B = 2 ∧ C = 4 ∧ J = 512) ∧ 
  (∀ (k : ℕ), k ∈ range 1 11 →
    (∃ l : ℕ, l = 2^(k - 1)) ∧
    (∀ (i : ℕ), i < l →
      ∃ m : ℕ, m = 2i + 1 ∧ 
      P (i, k) = (m, k+1) ∧ P (i, k) = (m+1, k+1)))

def is_permutation {X : Type} (f : X → X) (P : X → Prop) : Prop := 
  ∀ (x y : X), P x → P y → ((x ≠ y) ∧ (f x = f y → x = y) ∧ (f y = y → x = y)) ∧ (f x = x ∨ f x ≠ x)

theorem num_permutations : 
  ∃ (M : ℕ), M = max_permutations_tree ∧ 
  ∀ (f : ℕ → ℕ), is_permutation f (λ n, n < 512) ∧
  ten_level_tree ℕ ℕ (λ k, 512) →
  M = 2 ^ 128 := 
begin
  use 2^128,
  sorry
end

end num_permutations_l418_418234


namespace geometric_sequence_ratio_l418_418700

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

-- Definitions based on given conditions
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement
theorem geometric_sequence_ratio :
  is_geometric_seq a q →
  q = -1/3 →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by
  intros
  sorry

end geometric_sequence_ratio_l418_418700


namespace arcsin_solution_l418_418505

theorem arcsin_solution (x : ℝ) (h_eq : arcsin x + arcsin (3 * x) = π / 4)
    (h_x_le_1 : |x| ≤ 1) (h_3x_le_1 : |3 * x| ≤ 1) : 
    x = sqrt 102 / 51 :=
begin
  sorry
end

end arcsin_solution_l418_418505


namespace largest_two_digit_divisible_by_6_and_ends_in_4_l418_418945

-- Define what it means to be a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define what it means to be divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Define what it means to end in 4
def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

-- Final theorem statement
theorem largest_two_digit_divisible_by_6_and_ends_in_4 : 
  ∀ n, is_two_digit n ∧ divisible_by_6 n ∧ ends_in_4 n → n ≤ 84 :=
by
  -- sorry is used here as we are not providing the proof
  sorry

end largest_two_digit_divisible_by_6_and_ends_in_4_l418_418945


namespace no_rational_solution_l418_418978

-- Definition of the fractional part
def fractional_part (x : ℝ) : ℝ := x - floor x

-- The problem statement in Lean
theorem no_rational_solution 
    (x : ℝ) (hx_pos : x > 0) :
    (fractional_part x + fractional_part (1 / x) = 1) 
    → ¬ ∃ q : ℚ, (q : ℝ) = x :=
sorry

end no_rational_solution_l418_418978


namespace number_of_ways_to_place_pawns_l418_418393

theorem number_of_ways_to_place_pawns : 
  let board := (Fin 5 → Fin 5)
  let pawns := Finset.univ : Finset (Fin 5)
  (∃ f : Fin 5 → Fin 5, Function.Injective f ∧ Finset.card (Finset.image f pawns) = Finset.card pawns) → 
  (Finset.product pawns pawns).card = 14400 :=
by
  sorry

end number_of_ways_to_place_pawns_l418_418393


namespace cubes_with_one_painted_side_l418_418208

theorem cubes_with_one_painted_side (side_length : ℕ) (one_cm_cubes : ℕ) : 
  side_length = 5 → one_cm_cubes = 54 :=
by 
  intro h 
  sorry

end cubes_with_one_painted_side_l418_418208


namespace simplify_fraction_l418_418083

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l418_418083


namespace find_a2_l418_418948

variable {a : ℕ → ℝ}

def sequence_properties (a : ℕ → ℝ) : Prop :=
  a 1 = 19 ∧
  a 9 = 99 ∧
  (∀ n, n ≥ 3 → a n = (∑ i in finset.range (n-1), a i) / (n-1))

theorem find_a2 (h : sequence_properties a) : a 2 = 179 :=
by
  sorry

end find_a2_l418_418948


namespace F_eq_0_or_1_or_F_eq_1_iff_perfect_square_l418_418590

noncomputable def f : ℕ → ℤ
| 1       := 1
| n@(m+1) := if n > 1 then (-1) ^ ((nat.factors n).to_finset.card) else 1

noncomputable def F (n : ℕ) : ℤ :=
∑ d in (finset.divisors n), f d

theorem F_eq_0_or_1_or (n : ℕ) : 
  F(n) = 0 ∨ F(n) = 1 := sorry

theorem F_eq_1_iff_perfect_square (n : ℕ) :
  F(n) = 1 ↔ ∃ k : ℕ, n = k * k := sorry

end F_eq_0_or_1_or_F_eq_1_iff_perfect_square_l418_418590


namespace quadrilateral_AEGH_area_l418_418413

structure Point :=
(x : ℝ)
(y : ℝ)

structure Rectangle :=
(A B C D : Point)
(AB_length : ℝ)
(BC_length : ℝ)
(AB_CD_parallel : A.x = D.x ∧ B.x = C.x)
(AD_BC_parallel : A.y = B.y ∧ D.y = C.y)
(AB_ABCD : AB_length = dist A B)
(BC_ABCD : BC_length = dist B C)

def midpoint (P Q : Point) : Point :=
{ x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def dist (P Q : Point) : ℝ :=
real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def rectangleABCD : Rectangle :=
{ A := {x := 0, y := 0},
  B := { x := 2, y := 0},
  C := { x := 2, y := 4 },
  D := { x := 0, y := 4 },
  AB_length := 2,
  BC_length := 4,
  AB_CD_parallel := by simp,
  AD_BC_parallel := by simp,
  AB_ABCD := by simp [dist],
  BC_ABCD := by simp [dist] }

def E := midpoint rectangleABCD.B rectangleABCD.C
def F := midpoint rectangleABCD.C rectangleABCD.D
def G := midpoint rectangleABCD.A rectangleABCD.D
def H := midpoint G F

def area_of_triangle (P Q R : Point) : ℝ :=
| (Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y) | / 2

def area_AEGH : ℝ :=
let A := rectangleABCD.A in
let E_h := E in
let G_h := G in
let H_h := H in
area_of_triangle A E_h G_h + area_of_triangle E_h G_h H_h

theorem quadrilateral_AEGH_area : area_AEGH = 2.75 := 
sorry

end quadrilateral_AEGH_area_l418_418413


namespace data_instances_in_one_hour_l418_418439

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l418_418439


namespace recorded_instances_l418_418432

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l418_418432


namespace device_records_720_instances_in_one_hour_l418_418435

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l418_418435


namespace range_of_a_l418_418699

def f (a x : ℝ) : ℝ :=
  abs ((2^(x + 1) / (2^x + 2^(-x))) - 1 - a)

theorem range_of_a :
  { a : ℝ |
    ((∃ (n : ℕ) (x : ℕ → ℝ), n = 8 ∧ (finset.range (n-1)).sum (λ i, f a (x i)) = f a (x n)) →
    a ∈ (-4/3, -9/7] ∪ [9/7, 4/3))
  }
  sorry

end range_of_a_l418_418699


namespace hemisphere_surface_area_l418_418110

theorem hemisphere_surface_area (base_area : ℝ) (r : ℝ) (total_surface_area : ℝ) 
(h1: base_area = 64 * Real.pi) 
(h2: r^2 = 64)
(h3: total_surface_area = base_area + 2 * Real.pi * r^2) : 
total_surface_area = 192 * Real.pi := 
sorry

end hemisphere_surface_area_l418_418110


namespace multiple_less_than_k4_with_four_digits_l418_418058

theorem multiple_less_than_k4_with_four_digits (k : ℕ) (hk : k > 1) : 
  ∃ m : ℕ, (m < k^4) ∧ (∃ d : List ℕ, d.length ≤ 4 ∧ ∀ x ∈ d, x ∈ [0, 1, 8, 9] ∧ decimal_digits m = d) := sorry

end multiple_less_than_k4_with_four_digits_l418_418058


namespace angle_bisector_inequality_l418_418804

theorem angle_bisector_inequality {a b c fa fb fc : ℝ} 
  (h_triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (1 / fa + 1 / fb + 1 / fc > 1 / a + 1 / b + 1 / c) :=
by
  sorry

end angle_bisector_inequality_l418_418804


namespace point_above_parabola_probability_l418_418960

theorem point_above_parabola_probability :
  (∃ (n : ℕ), n = 8 ∧
    ∃ (count : ℕ), count = 2 ∧
    ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 4 ∧
    ((a = 1 ∧ b ∈ {3, 4}) ∨
    (a = 2 ∨ a = 3 ∨ a = 4 → b > a^3 + a^2) ∨
    (a = 2 ∨ a = 3 ∨ a = 4 → b > 12 ∨ b > 36 ∨ b > 80))) →
  (finset.univ.sum (λ (_ : finset (ℕ × ℕ)), if ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 4 ∧ (a = 1 ∧ b ∈ {3, 4})) then 1 else 0) = 2 / 16) :=
begin
  sorry
end

end point_above_parabola_probability_l418_418960


namespace distance_between_BC_more_than_AB_l418_418225

variables (A B C D : Type) 

def distance : Type := ℝ

variables (AB : distance) (BC : distance) (CD : distance)

-- Given conditions
axiom distance_AB : AB = 100
axiom distance_BC_is_x : ∃ x : distance, BC = x
axiom distance_CD : ∃ x : distance, CD = 2 * x
axiom total_distance_AD : ∃ x : distance, 100 + x + 2 * x = 550

-- Prove the distance between city B and city C is 50 miles more than the distance between city A and city B.
theorem distance_between_BC_more_than_AB :
  ∃ x : distance, BC = x ∧ (x - 100 = 50) :=
by
  sorry

end distance_between_BC_more_than_AB_l418_418225


namespace find_ratio_l418_418030

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : pure_imaginary ((3 - 5*ℂ.i) * (complex.of_real a + complex.of_real b * ℂ.i) * (1 + 2*ℂ.i))) :
  a / b = -1 / 7 :=
by
  sorry

end find_ratio_l418_418030


namespace comb_18_10_proof_l418_418656

theorem comb_18_10_proof (comb : ℕ → ℕ → ℕ) : comb 18 10 = 47190 :=
by
  assume h1 : comb 16 7 = 11440,
  assume h2 : comb 16 9 = 11440,
  have pascal : ∀ n k, comb (n + 1) (k + 1) = comb n (k + 1) + comb n k := sorry,
  have symmetry : ∀ n k, comb n k = comb n (n - k) := sorry,
  let h3 : comb 16 8 = comb 16 8 := by sorry, -- by property and known value
  let h4 : comb 16 8 = 12870 := by sorry, -- complementary counting
  let h5 : comb 17 10 = comb 16 9 + comb 16 8 := by apply pascal,
  let h6 : comb 17 9 = comb 16 8 + comb 16 7 := by apply pascal,
  let h7 : comb 17 10 = 11440 + comb 16 8 := by rw [h2, h3],
  let h8 : comb 17 9 = 11440 + comb 16 8 := by rw [h1, h3],
  let h9 : comb 18 10 = comb 17 10 + comb 17 9 := by apply pascal,
  rw [h7, h8, h4] at h9,
  exact h9

end comb_18_10_proof_l418_418656


namespace tetrahedron_volume_ratio_l418_418135

open Real

theorem tetrahedron_volume_ratio (ABCD : ℝ) (k : ℝ) : 
  let V_ABCD := ABCD in
  let V1 := (k^3 + 3 * k^2) / (k + 1)^3 * V_ABCD in
  let V2 := V_ABCD - V1 in
  V1 / V2 = (k^2 * (k + 3)) / (3 * k + 1) := 
by
  sorry

end tetrahedron_volume_ratio_l418_418135


namespace no_integer_solutions_l418_418490

theorem no_integer_solutions (x y : ℤ) : ¬ (3 * x^2 + 2 = y^2) :=
sorry

end no_integer_solutions_l418_418490


namespace cos_sin_sequence_rational_l418_418355

variable (α : ℝ) (h₁ : ∃ r : ℚ, r = (Real.sin α + Real.cos α))

theorem cos_sin_sequence_rational :
    (∀ n : ℕ, n > 0 → ∃ r : ℚ, r = (Real.cos α)^n + (Real.sin α)^n) :=
by
  sorry

end cos_sin_sequence_rational_l418_418355


namespace area_of_isosceles_trapezoid_l418_418180

def isIsoscelesTrapezoid (a b c h : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2

theorem area_of_isosceles_trapezoid :
  ∀ (a b c : ℝ), 
    a = 8 → b = 14 → c = 5 →
    ∃ h: ℝ, isIsoscelesTrapezoid a b c h ∧ ((a + b) / 2 * h = 44) :=
by
  intros a b c ha hb hc
  sorry

end area_of_isosceles_trapezoid_l418_418180


namespace sphere_volume_l418_418992

theorem sphere_volume (surface_area : ℝ) :
  surface_area = 256 * real.pi → ∃ (v : ℝ), v = (2048 / 3) * real.pi :=
by
  assume h : surface_area = 256 * real.pi
  let r := real.sqrt (surface_area / (4 * real.pi))
  have : r = 8 := by sorry  -- calculation assuming it is correct
  let volume := (4 / 3) * real.pi * r^3
  use volume
  show volume = (2048/3) * real.pi, from sorry  -- calculation assuming it is correct

end sphere_volume_l418_418992


namespace combined_students_yellow_blue_l418_418481

theorem combined_students_yellow_blue {total_students blue_percent red_percent yellow_combined : ℕ} :
  total_students = 200 →
  blue_percent = 30 →
  red_percent = 40 →
  yellow_combined = (total_students * 3 / 10) + ((total_students - (total_students * 3 / 10)) * 6 / 10) →
  yellow_combined = 144 :=
by
  intros
  sorry

end combined_students_yellow_blue_l418_418481


namespace slices_per_pizza_l418_418884

def num_pizzas : ℕ := 2
def total_slices : ℕ := 16

theorem slices_per_pizza : total_slices / num_pizzas = 8 := by
  sorry

end slices_per_pizza_l418_418884


namespace quadrilateral_diagonal_and_sides_proof_l418_418971

noncomputable def quadrilateral_diagonal_and_sides (d AC : ℝ) (P A : ℝ) (x y : ℝ) (alpha : ℝ) :=
  AC = 5 ∧
  P = 14 ∧
  A = 12 ∧
  x + y = 7 ∧
  x * y * Real.sin alpha = 12 ∧
  x^2 + y^2 - 2 * x * y * Real.cos alpha = 25 ∧
  1 + Real.cos alpha = Real.sin alpha ∧
  (x = 3 ∧ y = 4 ∨ x = 4 ∧ y = 3) ∧
  d = 4.8

theorem quadrilateral_diagonal_and_sides_proof :
  ∀ (AC P A x y alpha : ℝ),
    quadrilateral_diagonal_and_sides 4.8 AC P A x y alpha →
    ∃ bd : ℝ × list ℕ,
      bd = (4.8, [3, 4, 3, 4]) :=
by
  intro AC P A x y alpha h
  sorry

end quadrilateral_diagonal_and_sides_proof_l418_418971


namespace points_lie_on_line_l418_418688

theorem points_lie_on_line : ∀ (u : ℝ), (∃ (x y : ℝ), x = Real.cos (2 * u)^2 ∧ y = Real.sin (2 * u)^2 ∧ x + y = 1) :=
by
  intro u
  use Real.cos (2 * u) ^ 2
  use Real.sin (2 * u) ^ 2
  split
  { exact rfl }
  split
  { exact rfl }
  { sorry }

end points_lie_on_line_l418_418688


namespace length_of_XY_l418_418049

theorem length_of_XY (X Y G H I J : Point) (h1 : midpoint G X Y) 
                     (h2 : midpoint H X G) (h3 : midpoint I X H) 
                     (h4 : midpoint J X I) (h5 : dist X J = 5) :
    dist X Y = 40 :=
sorry

end length_of_XY_l418_418049


namespace matrix_sum_correct_l418_418670

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 0],
  ![1, 2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-5, -7],
  ![4, -9]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, -7],
  ![5, -7]
]

theorem matrix_sum_correct : A + B = C := by 
  sorry

end matrix_sum_correct_l418_418670


namespace fair_2m_digits_exists_l418_418154

def is_even (n : Nat) : Prop := n % 2 = 0

def even_positions (digits : List Nat) : List Nat :=
  List.filteri (fun i d => is_even (i + 1)) digits

def odd_positions (digits : List Nat) : List Nat :=
  List.filteri (fun i d => ¬ is_even (i + 1)) digits

def count_even_digits (digits : List Nat) : Nat :=
  List.length (List.filter is_even digits)

def is_fair (digits : List Nat) : Prop :=
  count_even_digits (even_positions digits) = count_even_digits (odd_positions digits)

theorem fair_2m_digits_exists (m : Nat) (digits : List Nat) (h_len : List.length digits = 2 * m + 1) :
  ∃ digits', List.length digits' = 2 * m ∧ is_fair digits' :=
by
  sorry

end fair_2m_digits_exists_l418_418154


namespace problem_statement_l418_418177

variable (a b c : ℝ)

theorem problem_statement
  (h1 : a + b = 100)
  (h2 : b + c = 140) :
  c - a = 40 :=
sorry

end problem_statement_l418_418177


namespace investment_rate_l418_418237

theorem investment_rate (total_investment income1_rate income2_rate income_total remaining_investment expected_income : ℝ)
  (h1 : total_investment = 12000)
  (h2 : income1_rate = 0.03)
  (h3 : income2_rate = 0.045)
  (h4 : expected_income = 600)
  (h5 : (5000 * income1_rate + 4000 * income2_rate) = 330)
  (h6 : remaining_investment = total_investment - 5000 - 4000) :
  (remaining_investment * 0.09 = expected_income - (5000 * income1_rate + 4000 * income2_rate)) :=
by
  sorry

end investment_rate_l418_418237


namespace triangle_ABC_properties_l418_418473

def point := ℤ × ℤ

def distance (p1 p2 : point) : ℝ := real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def area (A B C : point) : ℝ :=
  (1 / 2) * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_ABC_properties :
  let A := (0, 0) in
  let B := (2, 3) in
  let C := (3, 5) in
  area A B C < 1 ∧ distance A B > 2 ∧ distance B C > 2 ∧ distance A C > 2 :=
by
  sorry

end triangle_ABC_properties_l418_418473


namespace plantable_area_l418_418206

noncomputable def flowerbed_r := 10
noncomputable def path_w := 4
noncomputable def full_area := 100 * Real.pi
noncomputable def segment_area := 20.67 * Real.pi * 2 -- each path affects two segments

theorem plantable_area :
  full_area - segment_area = 58.66 * Real.pi := 
by sorry

end plantable_area_l418_418206


namespace angle_B_is_pi_div_3_area_DEF_range_l418_418423

-- Helper condition
axiom two_a_minus_c_eq_two_b_cos_C 
  (a b c : ℝ) (C : ℝ) : 2 * a - c = 2 * b * real.cos C

-- Proof Problem 1: Prove that angle B is π / 3
theorem angle_B_is_pi_div_3 (a b c C B : ℝ)
  (h : two_a_minus_c_eq_two_b_cos_C a b c C) : 
  B = π / 3 := 
sorry

-- Helper condition for Proof problem 2
axiom α_in_range (α : ℝ) : π / 6 ≤ α ∧ α ≤ π / 2

-- Proof Problem 2: Prove the area of triangle DEF as a function of α 
-- ranges within [sqrt(3)/4, 3 * sqrt(3)/8]
theorem area_DEF_range (α S : ℝ)
  (h1 : b = 2) (h2 : c = 2) 
  (h3 : α_in_range α)
  (h4 : ∃ a : ℝ, 2 * a - c = 2 * b * (real.cos (π / 3))) :
  S ∈ set.Icc ((√3) / 4 : ℝ) ((3 * √3) / 8 : ℝ) :=
sorry

end angle_B_is_pi_div_3_area_DEF_range_l418_418423


namespace gcf_palindromes_multiple_of_3_eq_3_l418_418574

-- Defining a condition that expresses a three-digit palindrome in the form 101a + 10b + a
def is_palindrome (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Defining a condition that n is a multiple of 3
def is_multiple_of_3 (n : ℕ) : Prop :=
n % 3 = 0

-- The Lean statement to prove the greatest common factor of all three-digit palindromes that are multiples of 3
theorem gcf_palindromes_multiple_of_3_eq_3 :
  ∃ gcf : ℕ, gcf = 3 ∧ ∀ n : ℕ, (is_palindrome n ∧ is_multiple_of_3 n) → gcf ∣ n :=
by
  sorry

end gcf_palindromes_multiple_of_3_eq_3_l418_418574


namespace molecular_weight_K2Cr2O7_l418_418159

def molecular_weight_one_mole (total_weight : ℕ) (num_moles : ℕ) : ℕ :=
  total_weight / num_moles

theorem molecular_weight_K2Cr2O7 :
  ∀ (total_weight : ℕ) (num_moles : ℕ),
  total_weight = 1184 → num_moles = 4 → molecular_weight_one_mole total_weight num_moles = 296 :=
by
  intros total_weight num_moles h1 h2
  rw [h1, h2]
  simp [molecular_weight_one_mole]
  norm_num

end molecular_weight_K2Cr2O7_l418_418159


namespace number_of_regions_eight_lines_l418_418668

theorem number_of_regions_eight_lines (h1 : ∀ i j, i ≠ j → ¬ ∥ (ℓ i) ∥ = ∥ (ℓ j) ∥) 
    (h2 : ∀ i j k, (i ≠ j ∧ j ≠ k ∧ i ≠ k) → ¬ (ℓ i) ∥ (ℓ j) ∧ ¬ (ℓ j) ∥ (ℓ k) ∧ ¬ (ℓ k) ∥ (ℓ i))
    (ℓ : Fin 8 → Line) : 
    (1 + 8 + Nat.choose 8 2) = 37 :=
by
  -- The proof is to be filled in later
  sorry

end number_of_regions_eight_lines_l418_418668


namespace sum_and_round_to_nearest_ten_l418_418467

/-- A function to round a number to the nearest ten -/
def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + 10 - n % 10

/-- The sum of 54 and 29 rounded to the nearest ten is 80 -/
theorem sum_and_round_to_nearest_ten : round_to_nearest_ten (54 + 29) = 80 :=
by
  sorry

end sum_and_round_to_nearest_ten_l418_418467


namespace number_of_machines_in_first_group_l418_418957

-- Define the initial conditions
def first_group_production_rate (x : ℕ) : ℚ :=
  20 / (x * 10)

def second_group_production_rate : ℚ :=
  180 / (20 * 22.5)

-- The theorem we aim to prove
theorem number_of_machines_in_first_group (x : ℕ) (h1 : first_group_production_rate x = second_group_production_rate) :
  x = 5 :=
by
  -- Placeholder for the proof steps
  sorry

end number_of_machines_in_first_group_l418_418957


namespace total_trees_expr_total_trees_given_A_12_l418_418104

-- Definitions based on conditions
def trees_planted_by_B (x : ℕ) := x
def trees_planted_by_A (x : ℕ) := (1.2 * x : ℝ)
def trees_planted_by_C (x : ℕ) := (1.2 * x : ℝ) - 2

-- Proof statement: Part 1
theorem total_trees_expr (x : ℕ) : 
    trees_planted_by_B x + trees_planted_by_A x + trees_planted_by_C x = 3.4 * x - 2 :=
by sorry

-- Definitions based on given "A planted 12 trees"
def B_when_A_12 := 12 / 1.2
def C_when_A_12 := 12 - 2

-- Proof statement: Part 2
theorem total_trees_given_A_12 (hA : trees_planted_by_A 10 = 12) : 
    12 + B_when_A_12 + C_when_A_12 = 32 :=
by sorry

end total_trees_expr_total_trees_given_A_12_l418_418104


namespace problem_a_problem_b_l418_418564

-- Definitions for the conditions
def numCrocodiles := 10
def probCrocInEgg := 0.1
def firstCollectionComplete := true

def p : ℕ → ℝ := sorry -- Assume a function p for probabilities

-- Proof problem (a)
theorem problem_a (p1 : ℝ) (p2 : ℝ) :
  p 1 = p1 →
  p 2 = p2 →
  p1 = p2 :=
by sorry

-- Proof problem (b)
theorem problem_b (p2 : ℝ) (p3 : ℝ) (p4 : ℝ) (p5 : ℝ) (p6 : ℝ) (p7 : ℝ) (p8 : ℝ) (p9 : ℝ) (p10 : ℝ) :
  p 2 = p2 →
  p 3 = p3 →
  p 4 = p4 →
  p 5 = p5 →
  p 6 = p6 →
  p 7 = p7 →
  p 8 = p8 →
  p 9 = p9 →
  p 10 = p10 →
  p2 > p3 →
  p3 > p4 →
  p4 > p5 →
  p5 > p6 →
  p6 > p7 →
  p7 > p8 →
  p8 > p9 →
  p9 > p10 :=
by sorry

end problem_a_problem_b_l418_418564


namespace seq_term_150_l418_418286

theorem seq_term_150 :
  ∃ n : ℕ, (n = 150) ∧
  let seq := {x | ∃ (S : Finset ℕ), (∀ i ∈ S, 3^i ∣ x) ∧ (∀ i j ∈ S, i ≠ j → 3^i ≠ 3^j)} in
  ∃ m ∈ seq, (m.sum = 2280) ∧ (seq.to_list.nth 149 = some m) :=
sorry

end seq_term_150_l418_418286


namespace point_in_fourth_quadrant_l418_418359

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : -4 * a < 0) (h2 : 2 + b < 0) : 
  (a > 0) ∧ (b < -2) → (a > 0) ∧ (b < 0) := 
by
  sorry

end point_in_fourth_quadrant_l418_418359


namespace better_performance_image_l418_418963

-- Heights and variances
variable (N : ℕ) (xA xB : ℝ) (sA2 sB2 : ℝ)

-- Conditions
def team_conditions (N : ℕ) (xA xB sA2 sB2 : ℝ) : Prop :=
  N = 20 /\
  xA = 160 /\
  xB = 160 /\
  sA2 = 10.5 /\
  sB2 = 1.2

-- Theorem statement
theorem better_performance_image : 
  team_conditions 20 160 160 10.5 1.2 → sB2 < sA2 :=
by
  intro h
  have h_variances : sA2 = 10.5 ∧ sB2 = 1.2 := by
    cases h with h1 h2
    cases h2 with h3 h4
    cases h4 with h5 h6
    exact h
  sorry

end better_performance_image_l418_418963


namespace find_complex_and_quadratic_solution_l418_418455

def complex_number_properties (z : ℂ) (a b : ℝ) (m n : ℝ) : Prop :=
  z = a + b * complex.I ∧
  a > 0 ∧
  abs z = 2 * real.sqrt 5 ∧
  ((1 + 2 * complex.I) * z).re = 0 ∧
  ∃ z_conjugate : ℂ, z_conjugate = complex.conj(z) ∧
                      z_conjugate = a - b * complex.I ∧
                      m = -((a + b * complex.I) + (a - b * complex.I)).re ∧
                      n = (a + b * complex.I) * (a - b * complex.I).re

theorem find_complex_and_quadratic_solution 
  (z : ℂ) (a b : ℝ) (m n : ℝ) 
  (H1 : complex_number_properties z a b m n) : 
  z = 4 + 2 * complex.I ∧ m = -8 ∧ n = 20 := 
by {
  sorry,
}

end find_complex_and_quadratic_solution_l418_418455


namespace arithmetic_sequence_correct_geometric_sequence_sum_correct_l418_418344

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  (a 4 = 6) ∧ (a 6 = 10)

noncomputable def general_formula_valid (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 2

theorem arithmetic_sequence_correct :
  ∃ a : ℕ → ℤ, arithmetic_sequence a ∧ general_formula_valid a :=
by
  let a := λ n : ℕ, 2 * n - 2
  use a
  split
  · -- Proof that a 4 = 6 and a 6 = 10
    sorry
  · -- Proof that a n = 2n - 2
    sorry

noncomputable def geometric_sequence (b : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  (∀ n, b n > 0) ∧ (b 1 = 1) ∧ (b 3 = a 3)

noncomputable def sum_of_terms (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n, T n = 2^n - 1

theorem geometric_sequence_sum_correct (a : ℕ → ℤ) :
  ∃ b : ℕ → ℤ, ∃ T : ℕ → ℕ, geometric_sequence b a ∧ sum_of_terms b T :=
by
  let a := λ n : ℕ, 2 * n - 2
  let b := λ n : ℕ, 2^(n-1)
  let T := λ n : ℕ, 2^n - 1
  use b, T
  split
  · -- Proof that ∀ n, b n > 0 ∧ b 1 = 1 ∧ b 3 = a 3
    sorry
  · -- Proof that T n = 2^n - 1
    sorry

end arithmetic_sequence_correct_geometric_sequence_sum_correct_l418_418344


namespace daughters_meet_days_count_l418_418415

noncomputable def days_elder_returns := 5
noncomputable def days_second_returns := 4
noncomputable def days_youngest_returns := 3

noncomputable def total_days := 100

-- Defining the count of individual and combined visits
noncomputable def count_individual_visits (period : ℕ) : ℕ := total_days / period
noncomputable def count_combined_visits (period1 : ℕ) (period2 : ℕ) : ℕ := total_days / Nat.lcm period1 period2
noncomputable def count_all_together_visits (periods : List ℕ) : ℕ := total_days / periods.foldr Nat.lcm 1

-- Specific counts
noncomputable def count_youngest_visits : ℕ := count_individual_visits days_youngest_returns
noncomputable def count_second_visits : ℕ := count_individual_visits days_second_returns
noncomputable def count_elder_visits : ℕ := count_individual_visits days_elder_returns

noncomputable def count_youngest_and_second : ℕ := count_combined_visits days_youngest_returns days_second_returns
noncomputable def count_youngest_and_elder : ℕ := count_combined_visits days_youngest_returns days_elder_returns
noncomputable def count_second_and_elder : ℕ := count_combined_visits days_second_returns days_elder_returns

noncomputable def count_all_three : ℕ := count_all_together_visits [days_youngest_returns, days_second_returns, days_elder_returns]

-- Final Inclusion-Exclusion principle application
noncomputable def days_at_least_one_returns : ℕ := 
  count_youngest_visits + count_second_visits + count_elder_visits
  - count_youngest_and_second
  - count_youngest_and_elder
  - count_second_and_elder
  + count_all_three

theorem daughters_meet_days_count : days_at_least_one_returns = 60 := by
  sorry

end daughters_meet_days_count_l418_418415


namespace seating_arrangements_valid_l418_418321

-- Define individual children and sibling pairs
structure SiblingPair (α : Type) :=
(a : α)
(b : α)

variables {α : Type} [DecidableEq α]

def rows := fin 2 → fin 4 → α

-- Define the constraint: siblings must either sit directly behind each other or not adjacent but on the same row
def valid_arrangement (pairs : list (SiblingPair α)) (seating : rows) : Prop :=
  ∀ (p : SiblingPair α), p ∈ pairs →
    ((∃ (i : fin 4), seating 0 i = p.a ∧ seating 1 i = p.b) ∨
     (∃ (i j: fin 4), seating 0 i = p.a ∧ seating 1 j = p.b ∧ abs (i.val - j.val) > 1))

-- Define the problem: calculating the number of valid seating arrangements
def number_of_valid_arrangements (pairs : list (SiblingPair α)) : ℕ :=
  -- Leaving the proof to be filled in
  sorry

theorem seating_arrangements_valid (pairs : list (SiblingPair α)) (h : length pairs = 4) :
  number_of_valid_arrangements pairs = 576 := sorry

end seating_arrangements_valid_l418_418321


namespace largest_divisible_by_6_ending_in_4_l418_418915

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l418_418915


namespace max_cos_sum_l418_418678

theorem max_cos_sum (x y : ℝ) (h : cos x - cos y = 1 / 4) : 
  cos (x + y) ≤ 31 / 32 :=
begin
  sorry
end

end max_cos_sum_l418_418678


namespace part1_part2_l418_418693

noncomputable def f (x : ℝ) : ℝ :=
  Math.cos x * (Real.sqrt 3 * Math.sin x - Math.cos x) + 3 / 2

theorem part1 : 
  ∀ x : ℝ, x ∈ Set.Icc 0 Real.pi → (∃ I : Set ℝ, I = Set.Icc (Real.pi / 3) (5 * Real.pi / 6) ∧ (∀ x ∈ I, f x' ≤ f x)) :=
sorry

theorem part2 (α : ℝ) (hα1 : α ∈ Set.Ioo (Real.pi / 3) (5 * Real.pi / 6)) (hα2 : f α = 2 / 5) : 
  Real.sin (2 * α) = (-(3 * Real.sqrt 3) - 4) / 10 :=
sorry

end part1_part2_l418_418693


namespace ratio_adults_children_is_one_l418_418966

theorem ratio_adults_children_is_one (a c : ℕ) (ha : a ≥ 1) (hc : c ≥ 1) (h : 30 * a + 15 * c = 2475) : a / c = 1 :=
by
  sorry

end ratio_adults_children_is_one_l418_418966


namespace prime_square_remainders_l418_418258

theorem prime_square_remainders (p : ℕ) (hp : Prime p) (hpg : p > 5) : 
  ∃ n : ℕ, n = 2 ∧ ∀ r ∈ {r : ℕ | ∃ p : ℕ, Prime p ∧ p > 5 ∧ r = (p ^ 2 % 180)}, r ∈ {1, 64} :=
by sorry

end prime_square_remainders_l418_418258


namespace find_p_q_r_sum_l418_418099

theorem find_p_q_r_sum :
  let p := 300, q := 100, r := 7 in
  -- Given conditions
  let AB_length := 1200,
      GH_length := 600,
      angle_GOH := 60 in
  let AB := (0, 0) to (AB_length, 0),
      G := (g, 0),
      H := (h, 0),
      O := (AB_length / 2, AB_length / 2) in
  -- Geometrical constraints
  G.1 < H.1 ∧ G.1 ∈ AB ∧ H.1 ∈ AB ∧ angle O G H = angle_GOH ∧ GH_length = G.1 - H.1
  -- Prove that $BH$ and $GH$ are as described in the solution
  ∧ h + q * Real.sqrt r = AB_length / 2
  ∧ p = 300 ∧ q = 100 ∧ r = 7 → p + q + r = 407 :=
begin
  sorry
end

end find_p_q_r_sum_l418_418099


namespace workers_in_workshop_l418_418530

theorem workers_in_workshop
    (W : ℕ) 
    (average_salary_all_workers : W * 8000) 
    (average_salary_technicians : 7 * 16000) 
    (average_salary_non_technicians : (W - 7) * 6000) 
    (total_salary : average_salary_all_workers = average_salary_technicians + average_salary_non_technicians) : 
    W = 35 :=
by {
  sorry
}

end workers_in_workshop_l418_418530


namespace non_empty_subsets_of_complement_intersection_l418_418038

-- Definitions for sets A, B, and universal set U
def A : Set ℕ := {4, 5, 7, 9}
def B : Set ℕ := {3, 4, 7, 8, 9}
def U : Set ℕ := A ∪ B

-- Define the complement of the intersection of A and B in the universal set U
def complement_intersection : Set ℕ := U \ (A ∩ B)

-- Prove that the number of non-empty subsets is 7
theorem non_empty_subsets_of_complement_intersection : 
  Fintype.card (Set (complement_intersection : Set ℕ)) - 1 = 7 := 
by 
  -- The proof is omitted.
  sorry

end non_empty_subsets_of_complement_intersection_l418_418038


namespace simplify_fraction_l418_418073

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l418_418073


namespace line_through_fixed_point_bisected_line_through_M_l418_418739

theorem line_through_fixed_point (m : ℝ) : 
    ∃ M : ℝ × ℝ, M = (-1, -2) ∧ (2 + m) * M.1 + (1 - 2 * m) * M.2 + (4 - 3 * m) = 0 := sorry

theorem bisected_line_through_M : 
    ∃ l1 : ℝ × ℝ → ℝ, (∀ M : ℝ × ℝ, M = (-1, -2) → l1 M = 0) → 
    (∃ k : ℝ, ∀ x y : ℝ, l1 (x, y) = y + 2 - k * (x + 1) ∧ 
        (2 / k - 1, k - 2) is the midpoint of (M.1, 0) and (0, M.2)) ∧ 
    l1 = λ (x, y), 2 * x + y + 4 := sorry

end line_through_fixed_point_bisected_line_through_M_l418_418739


namespace tangents_from_P_to_k_l418_418442

noncomputable def problem : Prop :=
  ∀ (k : Circle) (A B C D O P Q M N : Point),
    is_quadrilateral_inscribed A B C D k →
    is_intersection (line_through A C) (line_through B D) O →
    is_intersection (line_through A D) (line_through B C) P →
    is_intersection (line_through A B) (line_through C D) Q →
    is_intersection_with_circle (line_through Q O) k M N →
    is_tangent P M k ∧ is_tangent P N k

theorem tangents_from_P_to_k :
  problem :=
sorry

end tangents_from_P_to_k_l418_418442


namespace suki_bags_l418_418959

theorem suki_bags (bag_weight_suki : ℕ) (bag_weight_jimmy : ℕ) (containers : ℕ) 
  (container_weight : ℕ) (num_bags_jimmy : ℝ) (num_containers : ℕ)
  (h1 : bag_weight_suki = 22) 
  (h2 : bag_weight_jimmy = 18) 
  (h3 : container_weight = 8) 
  (h4 : num_bags_jimmy = 4.5)
  (h5 : num_containers = 28) : 
  6 = ⌊(num_containers * container_weight - num_bags_jimmy * bag_weight_jimmy) / bag_weight_suki⌋ :=
by
  sorry

end suki_bags_l418_418959


namespace median_is_5_5_l418_418327

def possible_sums : Finset ℕ := {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15}

def a : ℕ := (possible_sums.filter (λ x, x % 2 = 0)).card
def b : ℕ := (possible_sums.filter (λ x, x % 3 = 0)).card

def data_set : List ℕ := [6, a, b, 9]

def median (lst : List ℕ) : ℝ :=
  let sorted := lst.qsort (≤)
  if sorted.length % 2 = 0 then
    (sorted.nth (sorted.length / 2 - 1) + sorted.nth (sorted.length / 2)) / 2
  else
    sorted.nth (sorted.length / 2)

theorem median_is_5_5 : median data_set = 5.5 :=
  by
  sorry

end median_is_5_5_l418_418327


namespace no_18_consecutive_good_numbers_l418_418873

def is_good (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p * q

theorem no_18_consecutive_good_numbers :
  ¬ ∃ a : ℕ, ∀ k : ℕ, k < 18 → is_good (a + k) :=
by
  sorry

end no_18_consecutive_good_numbers_l418_418873


namespace hockey_pads_cost_l418_418801

variable (x h p remaining : ℝ)

theorem hockey_pads_cost :
  x = 150 ∧ h = x / 2 ∧ remaining = x - h - p ∧ remaining = 25 → p = 50 :=
by
  intro h₁
  cases h₁ with hx hh
  cases hh with hh₁ hh₂
  cases hh₂ with hr hr₁
  sorry

end hockey_pads_cost_l418_418801


namespace dog_grouping_l418_418525

theorem dog_grouping : 
  let dogs := 12 in
  let group1_size := 4 in
  let group2_size := 6 in
  let group3_size := 2 in
  let Fluffy := "Fluffy" in
  let Nipper := "Nipper" in
  let remaining_dogs := dogs - 2 in
  nat.choose remaining_dogs (group1_size - 1) * nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1) = 2520 :=
by
  sorry

end dog_grouping_l418_418525


namespace day110_of_year_N_minus_2_is_Wednesday_l418_418424

-- Definitions for the days of the week
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Given conditions
def day290_of_year_N (N : ℕ) : DayOfWeek := Wednesday
def day210_of_year_N_plus_2 (N : ℕ) : DayOfWeek := Wednesday

-- Proof problem statement
theorem day110_of_year_N_minus_2_is_Wednesday (N : ℕ) :
  day110_of_year_N_minus_2 N = Wednesday :=
sorry

end day110_of_year_N_minus_2_is_Wednesday_l418_418424


namespace sequence_properties_l418_418984

open Nat

-- Lean statement
theorem sequence_properties :
  (∀ n : ℕ, n > 0 → a (n + 1) - a n = 2) ∧ a 1 = 1 →
  (∀ n : ℕ, n > 0 → a n = 2 * n - 1) ∧ 
  (∃ n : ℕ, (sum_first_n_terms n) < 100 ∧ ∀ m : ℕ, (sum_first_n_terms m) < 100 → m ≤ n) :=
sorry

-- Definition of the sequence
def a : ℕ → ℤ
| 0     := 0  -- by default, we set a(0) as 0 since n > 0 in the problem
| (n+1) := if n = 0 then 1 else a n + 2
  
-- Definition of the sum of the first n terms
def sum_first_n_terms (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

end sequence_properties_l418_418984


namespace product_8_40_product_5_1_6_sum_6_instances_500_l418_418125

-- The product of 8 and 40 is 320
theorem product_8_40 : 8 * 40 = 320 := sorry

-- 5 times 1/6 is 5/6
theorem product_5_1_6 : 5 * (1 / 6) = 5 / 6 := sorry

-- The sum of 6 instances of 500 ends with 3 zeros and the sum is 3000
theorem sum_6_instances_500 :
  (500 * 6 = 3000) ∧ ((3000 % 1000) = 0) := sorry

end product_8_40_product_5_1_6_sum_6_instances_500_l418_418125


namespace complex_properties_l418_418458

def z := Complex
def a := Real
def b := Real
def m := Real
def n := Real

theorem complex_properties (z : Complex)
  (h1 : |z| = 2 * Real.sqrt 5)
  (h2 : (1 + 2 * Complex.i) * z).im = 0
  (h3 : 0 < z.re)
  (h4 : ∃ a b, z = a + b * Complex.i ∧ a^2 + b^2 = 20 ∧ a - 2 * b = 0)
  (root_of_eq : polynomial.aeval z (polynomial.C0 + polynomial.Cn * X + X^2) = 0)
  : z = 4 + 2 * Complex.i ∧ ∃ m n, root_of_eq ∧ m = -8 ∧ n = 20 :=
  sorry

end complex_properties_l418_418458


namespace problem1_problem2_l418_418733

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * log (x + 1)

theorem problem1 (a : ℝ) : (∀ x : ℝ, (1 ≤ x → (2 * x^2 + 2 * x + a) / (x + 1) ≥ 0)) ↔ a ≥ -4 :=
sorry

theorem problem2 (a : ℝ) (x1 x2 : ℝ) (h0 : 0 < a) (h1 : a < 1 / 2) (h2 : x1 < x2)
  (h3 : ∃ x : ℝ, 2 * x ^ 2 + 2 * x + a = 0) :
  0 < (f x2 a) / x1 ∧ (f x2 a) / x1 < -1 / 2 + log 2 :=
sorry

end problem1_problem2_l418_418733


namespace sum_real_imaginary_l418_418989

noncomputable def complex_div (a b c d : ℝ) : ℂ :=
  ((a * c + b * d) + (b * c - a * d) * complex.I) / (c^2 + d^2)

theorem sum_real_imaginary (a b c d : ℝ) (h : a = 1 ∧ b = 4 ∧ c = 2 ∧ d = -4) :
  let z := complex_div a b c d in
  (z.re + z.im) = -1 / 10 :=
by
  sorry

end sum_real_imaginary_l418_418989


namespace range_of_b_over_a_l418_418411

-- Define the problem conditions and conclusion
theorem range_of_b_over_a 
  (a b c : ℝ) (A B C : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) 
  (h_sum_angles : A + B + C = π) 
  (h_sides_relation : ∀ x, (x^2 + c^2 - a^2 - ab = 0 ↔ x = 0)) : 
  1 < b / a ∧ b / a < 2 := 
sorry

end range_of_b_over_a_l418_418411


namespace simplify_fraction_l418_418080

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l418_418080


namespace sphere_volume_l418_418990

theorem sphere_volume (S : ℝ) (HS : S = 256 * real.pi) : 
  ∃ V : ℝ, V = (4 / 3) * real.pi * (8 ^ 3) ∧ V = (2048 / 3) * real.pi := 
by
  sorry

end sphere_volume_l418_418990


namespace value_of_expression_l418_418332

theorem value_of_expression
  (a b : ℝ)
  (h₁ : a = 2 + Real.sqrt 3)
  (h₂ : b = 2 - Real.sqrt 3) :
  a^2 + 2 * a * b - b * (3 * a - b) = 13 :=
by
  sorry

end value_of_expression_l418_418332


namespace ratio_of_cans_l418_418881

theorem ratio_of_cans (martha_cans : ℕ) (total_required : ℕ) (remaining_cans : ℕ) (diego_cans : ℕ) (ratio : ℚ) 
  (h1 : martha_cans = 90) 
  (h2 : total_required = 150) 
  (h3 : remaining_cans = 5) 
  (h4 : martha_cans + diego_cans = total_required - remaining_cans) 
  (h5 : ratio = (diego_cans : ℚ) / martha_cans) : 
  ratio = 11 / 18 := 
by
  sorry

end ratio_of_cans_l418_418881


namespace sunset_time_l418_418887

theorem sunset_time (length_of_daylight : Nat := 11 * 60 + 18) -- length of daylight in minutes
    (sunrise : Nat := 6 * 60 + 32) -- sunrise time in minutes after midnight
    : (sunrise + length_of_daylight) % (24 * 60) = 17 * 60 + 50 := -- sunset time calculation
by
  sorry

end sunset_time_l418_418887


namespace sphere_volume_l418_418993

theorem sphere_volume (surface_area : ℝ) :
  surface_area = 256 * real.pi → ∃ (v : ℝ), v = (2048 / 3) * real.pi :=
by
  assume h : surface_area = 256 * real.pi
  let r := real.sqrt (surface_area / (4 * real.pi))
  have : r = 8 := by sorry  -- calculation assuming it is correct
  let volume := (4 / 3) * real.pi * r^3
  use volume
  show volume = (2048/3) * real.pi, from sorry  -- calculation assuming it is correct

end sphere_volume_l418_418993


namespace no_18_consecutive_good_numbers_l418_418872

def is_good (n : ℕ) : Prop :=
  ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ n = p * q

theorem no_18_consecutive_good_numbers :
  ¬ ∃ a : ℕ, ∀ k : ℕ, k < 18 → is_good (a + k) :=
by
  sorry

end no_18_consecutive_good_numbers_l418_418872


namespace largestValidNumberIs84_l418_418900

-- Define the set of two-digit numbers
def isTwoDigitNumber (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

-- Define the predicate for a number being divisible by 6
def isDivisibleBy6 (n : ℕ) : Prop := n % 6 = 0

-- Define the predicate for a number ending in 4
def endsIn4 (n : ℕ) : Prop := n % 10 = 4

-- Define the set of numbers which are two-digit, divisible by 6, and end in 4
def validNumbers : List ℕ := (List.range 100).filter (λ n, isTwoDigitNumber n ∧ isDivisibleBy6 n ∧ endsIn4 n)

-- State that the largest number in validNumbers is 84
theorem largestValidNumberIs84 : 
    ∃ n, n ∈ validNumbers ∧ (∀ m, m ∈ validNumbers → m ≤ n) ∧ n = 84 :=
by
    sorry

end largestValidNumberIs84_l418_418900


namespace sequence_exists_for_all_k_l418_418304

theorem sequence_exists_for_all_k (n : ℕ) :
  ∀ k : ℕ, (k ∈ {1, 2, ..., n}) ↔ (∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i < n, x i > 0) ∧ (∃ i, x(i) = k)) :=
by
  sorry

end sequence_exists_for_all_k_l418_418304


namespace arrangement_schemes_exists_and_lowest_cost_l418_418107

def arrangements_possible (A B : ℕ) : Prop :=
  A + B = 50 ∧ 80 * A + 50 * B ≤ 3490 ∧ 40 * A + 90 * B ≤ 2950

def arrangement_cost (A B : ℕ) : ℕ :=
  800 * A + 960 * B

theorem arrangement_schemes_exists_and_lowest_cost :
  (∃ A B : ℕ, arrangements_possible A B ∧ A ∈ {31, 32, 33} ∧ B ∈ {17, 18, 19})
  ∧ ∃ A B : ℕ, arrangements_possible A B ∧ arrangement_cost A B = 42720 :=
sorry

end arrangement_schemes_exists_and_lowest_cost_l418_418107


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418923

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l418_418923


namespace copy_pages_15_dollars_l418_418820

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l418_418820


namespace tan_phi_of_right_triangle_l418_418784

-- Definition of the problem conditions and the goal
theorem tan_phi_of_right_triangle (β φ : ℝ)
  (h1 : tan (β / 2) = 1 / sqrt 3)
  (h2 : φ = angle_between_median_and_bisector β) :
  tan φ = (3 * sqrt 3 + 2) / 4 := 
sorry

end tan_phi_of_right_triangle_l418_418784


namespace blanket_thickness_after_foldings_l418_418282

theorem blanket_thickness_after_foldings (initial_thickness : ℕ) (folds : ℕ) (h1 : initial_thickness = 3) (h2 : folds = 4) :
  (initial_thickness * 2^folds) = 48 :=
by
  -- start with definitions as per the conditions
  rw [h1, h2]
  -- proof would follow
  sorry

end blanket_thickness_after_foldings_l418_418282


namespace triangle_side_a_triangle_angle_B_l418_418707

-- Part I: Prove that a = sqrt(2)
theorem triangle_side_a (b : ℝ) (A C : ℝ) (a : ℝ) 
    (h1 : b = sqrt 3) (h2 : A = 45) (h3 : C = 75) : a = sqrt 2 := 
by
  -- Lean proof would go here
  sorry

-- Part II: Prove that B = 135 degrees
theorem triangle_angle_B (a b c : ℝ) (B : ℝ) 
    (h : b^2 = a^2 + c^2 + sqrt 2 * a * c) : B = 135 := 
by
  -- Lean proof would go here
  sorry

end triangle_side_a_triangle_angle_B_l418_418707


namespace count_elements_starting_with_one_l418_418453

theorem count_elements_starting_with_one :
  let T := {x : ℕ | ∃ k : ℤ, 0 ≤ k ∧ k ≤ 1000 ∧ x = 3^k}
  in (∃ n : ℕ, 3^1000 = n ∧ n.toString.length = 477) →
     (∀ k : ℤ, 0 ≤ k ∧ k ≤ 1000 → ∃ d : Int, d ∈ T ∧ d.toString.head! = '1') →
     T.card = 524 :=
by
  sorry

end count_elements_starting_with_one_l418_418453


namespace condition1_condition2_l418_418414

-- Definition for the coordinates of point P based on given m
def P (m : ℝ) : ℝ × ℝ := (3 * m - 6, m + 1)

-- Condition 1: Point P lies on the x-axis
theorem condition1 (m : ℝ) (hx : P m = (3 * m - 6, 0)) : P m = (-9, 0) := 
by {
  -- Show that if y-coordinate is zero, then m + 1 = 0, hence m = -1
  sorry
}

-- Condition 2: Point A is (-1, 2) and AP is parallel to the y-axis
theorem condition2 (m : ℝ) (A : ℝ × ℝ := (-1, 2)) (hy : (3 * m - 6 = -1)) : P m = (-1, 8/3) :=
by {
  -- Show that if the x-coordinates of A and P are equal, then 3m-6 = -1, hence m = 5/3
  sorry
}

end condition1_condition2_l418_418414


namespace janet_counts_total_birds_l418_418833

theorem janet_counts_total_birds :
  let crows := 30
  let hawks := crows + (60 / 100) * crows
  hawks + crows = 78 :=
by
  sorry

end janet_counts_total_birds_l418_418833


namespace decagon_diagonals_l418_418381

-- Define the number of sides of a decagon
def n : ℕ := 10

-- Define the formula for the number of diagonals in an n-sided polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem decagon_diagonals : num_diagonals n = 35 := by
  sorry

end decagon_diagonals_l418_418381


namespace find_f1_l418_418367

noncomputable def f (x a b : ℝ) : ℝ := a * Real.sin x - b * Real.tan x + 4 * Real.cos (Real.pi / 3)

theorem find_f1 (a b : ℝ) (h : f (-1) a b = 1) : f 1 a b = 3 :=
by {
  sorry
}

end find_f1_l418_418367


namespace find_m_sum_terms_l418_418592

theorem find_m (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) 
  (h2 : a 3 + a 6 + a 10 + a 13 = 32) (hm : a m = 8) : m = 8 :=
sorry

theorem sum_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) (hS3 : S 3 = 9) (hS6 : S 6 = 36) 
  (a_def : ∀ n, S n = n * (a 1 + a n) / 2) : a 7 + a 8 + a 9 = 45 :=
sorry

end find_m_sum_terms_l418_418592


namespace selling_price_of_book_l418_418200

theorem selling_price_of_book (bought_price profit_percentage : ℝ) (hb : bought_price = 60) (hp : profit_percentage = 30) : 
  let profit := (profit_percentage / 100) * bought_price in
  let selling_price := bought_price + profit in
  selling_price = 78 :=
by
  sorry

end selling_price_of_book_l418_418200


namespace find_f_prime_at_2_l418_418372

noncomputable def f (f'2 : ℝ) : ℝ → ℝ := λ x, x^2 * f'2 + 5 * x

theorem find_f_prime_at_2 :
  ∃ f'2 : ℝ, (∃ f x, f x = x^2 * f'2 + 5 * x) ∧ deriv (f f'2) 2 = -5/3 :=
  sorry

end find_f_prime_at_2_l418_418372


namespace max_sum_of_two_integers_l418_418132

theorem max_sum_of_two_integers (x : ℕ) (h : x + 2 * x < 100) : x + 2 * x = 99 :=
sorry

end max_sum_of_two_integers_l418_418132


namespace PQ_parallel_AB_l418_418796

variable (A B C D E F P Q : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C]
variable (ABC : Triangle A B C)
variable (D_on_BC : PointOnLine D (Line B C))
variable (E_on_CA : PointOnLine E (Line C A))
variable (F_on_AB : PointOnLine F (Line A B))
variable (FE_parallel_BC : Parallel (Line F E) (Line B C))
variable (DF_parallel_CA : Parallel (Line D F) (Line C A))
variable (P_inter_BED : Intersects P (Line B E) (Line D F))
variable (Q_inter_FEA : Intersects Q (Line F E) (Line A D))

theorem PQ_parallel_AB (h : Triangle A B C)
  (hD : PointOnLine D (Line B C))
  (hE : PointOnLine E (Line C A))
  (hF : PointOnLine F (Line A B))
  (h1 : Parallel (Line F E) (Line B C))
  (h2 : Parallel (Line D F) (Line C A))
  (h3 : Intersects P (Line B E) (Line D F))
  (h4 : Intersects Q (Line F E) (Line A D)) :
  Parallel (Line P Q) (Line A B) :=
sorry

end PQ_parallel_AB_l418_418796


namespace melon_count_l418_418553

theorem melon_count (watermelons : ℕ) (apples : ℕ) (melons : ℕ) 
  (h_w : watermelons = 3) (h_a : apples = 7)
  (h_m : melons = 2 * (watermelons + apples)) :
  melons = 20 :=
by
  rw [h_w, h_a] at h_m
  exact h_m

# You can also add statements to check the equivalence
# Check the correct computation of the sum of fruits
example (watermelons : ℕ) (apples : ℕ) (h_w : watermelons = 3) (h_a : apples = 7) :
  watermelons + apples = 10 :=
by
  rw [h_w, h_a]
  rfl

# Verify the correct computation of the number of melons
example (sum_fruits : ℕ) (h_sf : sum_fruits = 10) :
  2 * sum_fruits = 20 :=
by
  rw [h_sf]
  rfl

end melon_count_l418_418553


namespace min_area_bounded_l418_418679

noncomputable def intersection_points (a : ℝ) (ha : 0 < a) : set ℝ :=
    { x | a^3 * x^2 - a^4 * x = x }

noncomputable def area_bounded (a : ℝ) (ha : 0 < a) : ℝ :=
    ∫ x in 0..(a^4 + 1) / a^3, x - (a^3 * x^2 - a^4 * x)

theorem min_area_bounded (a : ℝ) (ha : 0 < a) : area_bounded a ha = 4 / 3 :=
begin
    sorry
end

end min_area_bounded_l418_418679


namespace Cornelia_current_age_l418_418776

theorem Cornelia_current_age (K : ℕ) (C : ℕ) (h1 : K = 20) (h2 : C + 10 = 3 * (K + 10)) : C = 80 :=
by
  sorry

end Cornelia_current_age_l418_418776


namespace sandy_initial_books_l418_418056

-- Define the initial conditions as given.
def books_tim : ℕ := 33
def books_lost : ℕ := 24
def books_after_loss : ℕ := 19

-- Define the equation for the total books before Benny's loss and solve for Sandy's books.
def books_total_before_loss : ℕ := books_after_loss + books_lost
def books_sandy_initial : ℕ := books_total_before_loss - books_tim

-- Assert the proof statement:
def proof_sandy_books : Prop :=
  books_sandy_initial = 10

theorem sandy_initial_books : proof_sandy_books := by
  -- Placeholder for the actual proof.
  sorry

end sandy_initial_books_l418_418056


namespace simplify_expression_l418_418949

-- Define the given expressions
def numerator : ℕ := 5^5 + 5^3 + 5
def denominator : ℕ := 5^4 - 2 * 5^2 + 5

-- Define the simplified fraction
def simplified_fraction : ℚ := numerator / denominator

-- Prove that the simplified fraction is equivalent to 651 / 116
theorem simplify_expression : simplified_fraction = 651 / 116 := by
  sorry

end simplify_expression_l418_418949


namespace inequality_solution_set_solution_set_real_numbers_l418_418336

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + |2 * x - 4| + a

theorem inequality_solution_set (x : ℝ) : 
  {x | f x (-3) > x^2 + |x|} = {x | x < 1/3 ∨ 7 < x} :=
by
  sorry

theorem solution_set_real_numbers (a : ℝ) :
  ({ x : ℝ | f x a ≥ 0 } = set.univ) ↔ a ∈ set.Ici (-3) :=
by
  sorry

end inequality_solution_set_solution_set_real_numbers_l418_418336


namespace find_N_l418_418680

-- Define the problem conditions
variables (p : ℕ) (h_prime : nat.prime p) (h_not3 : p ≠ 3)
def N : ℕ := 3 * p^2

-- The hypothesis that the sum of the divisors of N is 124
def sum_divisors (n : ℕ) := n.divisors.sum
axiom h_sum_divisors : sum_divisors N = 124

-- The theorem that we need to prove
theorem find_N : N = 75 :=
by sorry

end find_N_l418_418680


namespace students_who_like_yellow_and_blue_l418_418478

/-- Problem conditions -/
def total_students : ℕ := 200
def percentage_blue : ℕ := 30
def percentage_red_among_not_blue : ℕ := 40

/-- We need to prove the following statement: -/
theorem students_who_like_yellow_and_blue :
  let num_blue := (percentage_blue * total_students) / 100 in
  let num_not_blue := total_students - num_blue in
  let num_red := (percentage_red_among_not_blue * num_not_blue) / 100 in
  let num_yellow := num_not_blue - num_red in
  num_yellow + num_blue = 144 :=
by
  sorry

end students_who_like_yellow_and_blue_l418_418478


namespace equilateral_triangle_side_length_l418_418486

theorem equilateral_triangle_side_length
  (P Q R S A B C : Type)
  [is_point P] [is_point Q] [is_point R] [is_point S] [is_point A] [is_point B] [is_point C]
  (hABC : is_equilateral_triangle A B C)
  (hP_inside : is_inside_triangle P A B C)
  (hQ_perp : is_perpendicular P Q A B)
  (hR_perp : is_perpendicular P R B C)
  (hS_perp : is_perpendicular P S C A)
  (PQ_len : length P Q = 3)
  (PR_len : length P R = 4)
  (PS_len : length P S = 5) :
  length A B = 8 * real.sqrt 3 :=
sorry

end equilateral_triangle_side_length_l418_418486


namespace mass_percentage_of_O_in_C6H8O6_l418_418291

def molar_mass (C H O : Nat) : Float :=
  (C * 12.01) + (H * 1.01) + (O * 16.00)

def mass_percentage (mass_element mass_total : Float) : Float :=
  (mass_element / mass_total) * 100

theorem mass_percentage_of_O_in_C6H8O6 :
  let C := 6
  let H := 8
  let O := 6
  let molar_mass_C6H8O6 := molar_mass C H O
  let mass_percentage_H := 4.55
  let mass_percentage_C := mass_percentage (C * 12.01) molar_mass_C6H8O6
  mass_percentage (O * 16.00) molar_mass_C6H8O6 = 54.54 :=
by
  sorry

end mass_percentage_of_O_in_C6H8O6_l418_418291


namespace cross_section_area_l418_418786

theorem cross_section_area (p q α : ℝ) (A B C D: α) 
  (area_ABC : triangle_area A B C = p) 
  (area_ABD : triangle_area A B D = q) 
  (angle_ABC_ABD : angle_between_faces A B C D = α) :
  cross_section_area A B (inscribed_sphere_center A B C D) = (2 * p * q * cos (α / 2)) / (p + q) :=
begin
  sorry
end

end cross_section_area_l418_418786


namespace largest_two_digit_divisible_by_6_ending_in_4_l418_418929

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def ends_in_4 (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_6_ending_in_4 :
  ∃ n : ℕ, is_two_digit n ∧ is_divisible_by_6 n ∧ ends_in_4 n ∧
  ∀ m : ℕ, is_two_digit m ∧ is_divisible_by_6 m ∧ ends_in_4 m → m ≤ n :=
  sorry

end largest_two_digit_divisible_by_6_ending_in_4_l418_418929


namespace total_rain_duration_l418_418005

theorem total_rain_duration :
  let day1 := 17 - 7 in
  let day2 := day1 + 2 in
  let day3 := day2 * 2 in
  day1 + day2 + day3 = 46 :=
by
  let day1 := 17 - 7
  let day2 := day1 + 2
  let day3 := day2 * 2
  calc
    day1 + day2 + day3 = 10 + 12 + 24 : by sorry
                     ... = 46 : by sorry

end total_rain_duration_l418_418005


namespace single_common_point_implies_a_eq_1_max_area_triangle_OAB_l418_418210

-- Definition of the curve C with parametric equations
def curve_C (a β : ℝ) : ℝ × ℝ :=
  (a + cos β, a * sin β)

-- Definition of the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop :=
  ρ * cos (θ - π / 3) = 3 / 2

-- Given conditions
variable (a : ℝ) (h_a_pos : a > 0)

-- Proof that curve C and line l have only one common point implies a = 1
theorem single_common_point_implies_a_eq_1 :
  (∃ β ρ θ, curve_C a β = (ρ * cos θ, ρ * sin θ) ∧ line_l ρ θ) ∧ 
  (∀ β₁ β₂ ρ θ, curve_C a β₁ = (ρ * cos θ, ρ * sin θ) ∧ line_l ρ θ → β₁ = β₂) →
  a = 1 :=
  sorry

-- Maximum area of triangle OAB on curve C given angle AOB = π / 3
theorem max_area_triangle_OAB :
  (∀ θ : ℝ, A B : curve_C a θ ∧ curve_C a (θ + π / 3)) →
  ∃ S, S = (3 * sqrt 3 * a^2 / 4) :=
  sorry

end single_common_point_implies_a_eq_1_max_area_triangle_OAB_l418_418210


namespace lambda_value_l418_418746

noncomputable def vector1 (λ : ℝ) : ℝ × ℝ := (λ + 1, 1)
noncomputable def vector2 (λ : ℝ) : ℝ × ℝ := (λ + 2, 2)
noncomputable def vector_sub (λ : ℝ) : ℝ × ℝ := (vector1 λ).fst - (vector2 λ).fst, (vector1 λ).snd - (vector2 λ).snd
noncomputable def vector_add (λ : ℝ) : ℝ × ℝ := (vector1 λ).fst + (vector2 λ).fst, (vector1 λ).snd + (vector2 λ).snd

theorem lambda_value (λ : ℝ) (h : (vector_sub λ).fst * (vector_add λ).fst + (vector_sub λ).snd * (vector_add λ).snd = 0) :
  λ = -3 :=
by sorry

end lambda_value_l418_418746


namespace optimal_strategy_loss_l418_418855

-- Define the game conditions
namespace game

variables {n : ℕ} [fact (0 < n)]

/-- The set of integers from 1 to n -/
def set_A : set ℕ := { k | k ≤ n }

-- Define properties of the game
-- No player can choose a number that has already been chosen in any previous round
def valid_choice (chosen : set ℕ) (k : ℕ) : Prop :=
  k ∈ set_A ∧ k ∉ chosen

-- No player can choose any number that is adjacent to any number they have chosen in any previous round
def valid_adjacent (chosen : set ℕ) (k : ℕ) : Prop :=
  ∀ t ∈ chosen, abs (t - k) ≠ 1

-- If all numbers are chosen, the game is a draw
def is_draw (chosen : set ℕ) : Prop :=
  ∀ k ∈ set_A, k ∈ chosen

-- The player who cannot choose a number loses the game
def cannot_choose (chosen : set ℕ) (player_turn : ℕ) : Prop :=
  ∀ k, valid_choice chosen k → valid_adjacent chosen k → false

-- Main theorem
theorem optimal_strategy_loss : n = 3024 :=
sorry

end game

end optimal_strategy_loss_l418_418855


namespace percent_profit_l418_418760

theorem percent_profit (C S : ℝ) (h : 60 * C = 50 * S):
  (((S - C) / C) * 100) = 20 :=
by 
  sorry

end percent_profit_l418_418760


namespace mixing_paint_l418_418880

theorem mixing_paint (total_parts : ℕ) (blue_parts : ℕ) (red_parts : ℕ) (white_parts : ℕ) (blue_ounces : ℕ) (max_mixture : ℕ) (ounces_per_part : ℕ) :
  total_parts = blue_parts + red_parts + white_parts →
  blue_parts = 7 →
  red_parts = 2 →
  white_parts = 1 →
  blue_ounces = 140 →
  max_mixture = 180 →
  ounces_per_part = blue_ounces / blue_parts →
  max_mixture / ounces_per_part = 9 →
  white_ounces = white_parts * ounces_per_part →
  white_ounces = 20 :=
sorry

end mixing_paint_l418_418880


namespace right_triangle_BC_val_l418_418029

noncomputable def right_triangle_BC : ℝ :=
  let AP : ℝ := 15
  let CQ : ℝ := 17
  let x : ℝ := AP
  let y : ℝ := CQ / 2
  let AB := 2 * x
  let AC := 2 * y
  let BC := Math.sqrt (4 * x^2 + 4 * y^2)
  BC

theorem right_triangle_BC_val :
  right_triangle_BC = Real.sqrt 514 :=
by 
  -- Proof to be provided
  sorry

end right_triangle_BC_val_l418_418029


namespace power_of_2_in_exp_sum_l418_418265

theorem power_of_2_in_exp_sum (p : ℝ) (e : ℝ → ℝ) :
  (∀ k, k ∈ finset.range(1, 9) → p = p + k * real.log k) →
  (e p = (∏ k in (finset.range(1, 9)), (k ^ k))) →
  (∃ n : ℕ, 2^n ∣ e ^ p ∧ n ≥ 40) :=
begin
  intro h,
  sorry
end

end power_of_2_in_exp_sum_l418_418265
