import Mathlib

namespace NUMINAMATH_GPT_apple_bags_l1077_107709

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end NUMINAMATH_GPT_apple_bags_l1077_107709


namespace NUMINAMATH_GPT_zarnin_staffing_l1077_107744

open Finset

theorem zarnin_staffing :
  let total_resumes := 30
  let unsuitable_resumes := total_resumes / 3
  let suitable_resumes := total_resumes - unsuitable_resumes
  let positions := 5
  suitable_resumes = 20 → 
  positions = 5 → 
  Nat.factorial suitable_resumes / Nat.factorial (suitable_resumes - positions) = 930240 := by
  intro total_resumes unsuitable_resumes suitable_resumes positions h1 h2
  have hs : suitable_resumes = 20 := h1
  have hp : positions = 5 := h2
  sorry

end NUMINAMATH_GPT_zarnin_staffing_l1077_107744


namespace NUMINAMATH_GPT_quilt_cost_l1077_107751

theorem quilt_cost :
  let length := 7
  let width := 8
  let cost_per_sq_ft := 40
  let area := length * width
  let total_cost := area * cost_per_sq_ft
  total_cost = 2240 :=
by
  sorry

end NUMINAMATH_GPT_quilt_cost_l1077_107751


namespace NUMINAMATH_GPT_ants_total_l1077_107727

namespace Ants

-- Defining the number of ants each child finds based on the given conditions
def Abe_ants := 4
def Beth_ants := Abe_ants + Abe_ants
def CeCe_ants := 3 * Abe_ants
def Duke_ants := Abe_ants / 2
def Emily_ants := Abe_ants + (3 * Abe_ants / 4)
def Frances_ants := 2 * CeCe_ants

-- The total number of ants found by the six children
def total_ants := Abe_ants + Beth_ants + CeCe_ants + Duke_ants + Emily_ants + Frances_ants

-- The statement to prove
theorem ants_total: total_ants = 57 := by
  sorry

end Ants

end NUMINAMATH_GPT_ants_total_l1077_107727


namespace NUMINAMATH_GPT_trig_identity_l1077_107746

theorem trig_identity 
  (α : ℝ) 
  (h : Real.tan α = 1 / 3) : 
  Real.cos α ^ 2 + Real.cos (π / 2 + 2 * α) = 3 / 10 :=
sorry

end NUMINAMATH_GPT_trig_identity_l1077_107746


namespace NUMINAMATH_GPT_sum_first_9_terms_l1077_107722

noncomputable def sum_of_first_n_terms (a1 d : Int) (n : Int) : Int :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_9_terms (a1 d : ℤ) 
  (h1 : a1 + (a1 + 3 * d) + (a1 + 6 * d) = 39)
  (h2 : (a1 + 2 * d) + (a1 + 5 * d) + (a1 + 8 * d) = 27) :
  sum_of_first_n_terms a1 d 9 = 99 := by
  sorry

end NUMINAMATH_GPT_sum_first_9_terms_l1077_107722


namespace NUMINAMATH_GPT_infinite_sum_eq_3_over_8_l1077_107770

theorem infinite_sum_eq_3_over_8 :
  ∑' n : ℕ, (n : ℝ) / (n^4 + 4) = 3 / 8 :=
sorry

end NUMINAMATH_GPT_infinite_sum_eq_3_over_8_l1077_107770


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l1077_107738

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 217) : 
  x^2 + y^2 ≥ 505 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l1077_107738


namespace NUMINAMATH_GPT_total_weight_of_fish_l1077_107772

theorem total_weight_of_fish (fry : ℕ) (survival_rate : ℚ) 
  (first_catch : ℕ) (first_avg_weight : ℚ) 
  (second_catch : ℕ) (second_avg_weight : ℚ)
  (third_catch : ℕ) (third_avg_weight : ℚ)
  (total_weight : ℚ) :
  fry = 100000 ∧ 
  survival_rate = 0.95 ∧ 
  first_catch = 40 ∧ 
  first_avg_weight = 2.5 ∧ 
  second_catch = 25 ∧ 
  second_avg_weight = 2.2 ∧ 
  third_catch = 35 ∧ 
  third_avg_weight = 2.8 ∧ 
  total_weight = fry * survival_rate * 
    ((first_catch * first_avg_weight + 
      second_catch * second_avg_weight + 
      third_catch * third_avg_weight) / 100) / 10000 →
  total_weight = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_fish_l1077_107772


namespace NUMINAMATH_GPT_arithmetic_seq_solution_l1077_107708

variables (a : ℕ → ℤ) (d : ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) - a n = d

def seq_cond (a : ℕ → ℤ) (d : ℤ) : Prop :=
is_arithmetic_sequence a d ∧ (a 2 + a 6 = a 8)

noncomputable def sum_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem arithmetic_seq_solution :
  ∀ (a : ℕ → ℤ) (d : ℤ), seq_cond a d → (a 2 - a 1 ≠ 0) → 
    (sum_first_n a 5 / a 5) = 3 :=
by
  intros a d h_cond h_d_ne_zero
  sorry

end NUMINAMATH_GPT_arithmetic_seq_solution_l1077_107708


namespace NUMINAMATH_GPT_compute_c_minus_d_cubed_l1077_107720

-- define c as the number of positive multiples of 12 less than 60
def c : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

-- define d as the number of positive integers less than 60 and a multiple of both 3 and 4
def d : ℕ := Finset.card (Finset.filter (λ n => 12 ∣ n) (Finset.range 60))

theorem compute_c_minus_d_cubed : (c - d)^3 = 0 := by
  -- since c and d are computed the same way, (c - d) = 0
  -- hence, (c - d)^3 = 0^3 = 0
  sorry

end NUMINAMATH_GPT_compute_c_minus_d_cubed_l1077_107720


namespace NUMINAMATH_GPT_similar_triangle_perimeter_l1077_107767

theorem similar_triangle_perimeter :
  ∀ (a b c : ℝ), a = 7 ∧ b = 7 ∧ c = 12 →
  ∀ (d : ℝ), d = 30 →
  ∃ (p : ℝ), p = 65 ∧ 
  (∃ a' b' c' : ℝ, (a' = 17.5 ∧ b' = 17.5 ∧ c' = d) ∧ p = a' + b' + c') :=
by sorry

end NUMINAMATH_GPT_similar_triangle_perimeter_l1077_107767


namespace NUMINAMATH_GPT_house_prices_and_yields_l1077_107711

theorem house_prices_and_yields :
  ∃ x y : ℝ, 
    (425 = (y / 100) * x) ∧ 
    (459 = ((y - 0.5) / 100) * (6/5) * x) ∧ 
    (x = 8500) ∧ 
    (y = 5) ∧ 
    ((6/5) * x = 10200) ∧ 
    (y - 0.5 = 4.5) :=
by
  sorry

end NUMINAMATH_GPT_house_prices_and_yields_l1077_107711


namespace NUMINAMATH_GPT_A_gt_B_and_C_lt_A_l1077_107701

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def canBePlacedInside (K P : Box) :=
  (K.x ≤ P.x ∧ K.y ≤ P.y ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.x ∧ K.y ≤ P.z ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.x ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.z ∧ K.z ≤ P.x) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.x ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.y ∧ K.z ≤ P.x)

theorem A_gt_B_and_C_lt_A :
  let A := Box.mk 6 5 3
  let B := Box.mk 5 4 1
  let C := Box.mk 3 2 2
  (canBePlacedInside B A ∧ ¬ canBePlacedInside A B) ∧
  (canBePlacedInside C A ∧ ¬ canBePlacedInside A C) :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_A_gt_B_and_C_lt_A_l1077_107701


namespace NUMINAMATH_GPT_prime_division_l1077_107731

-- Definitions used in conditions
variables {p q : ℕ}

-- We assume p and q are prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def divides (a b : ℕ) : Prop := ∃ k, b = k * a

-- The problem states
theorem prime_division 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hdiv : divides q (3^p - 2^p)) 
  : p ∣ (q - 1) :=
sorry

end NUMINAMATH_GPT_prime_division_l1077_107731


namespace NUMINAMATH_GPT_distance_from_sphere_center_to_triangle_plane_l1077_107739

theorem distance_from_sphere_center_to_triangle_plane :
  ∀ (O : ℝ × ℝ × ℝ) (r : ℝ) (a b c : ℝ), 
  r = 9 →
  a = 13 →
  b = 13 →
  c = 10 →
  (∀ (d : ℝ), d = distance_from_O_to_plane) →
  d = 8.36 :=
by
  intro O r a b c hr ha hb hc hd
  sorry

end NUMINAMATH_GPT_distance_from_sphere_center_to_triangle_plane_l1077_107739


namespace NUMINAMATH_GPT_max_value_of_x2_plus_y2_l1077_107765

open Real

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x - 2 * y + 2) : 
  x^2 + y^2 ≤ 6 + 4 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_x2_plus_y2_l1077_107765


namespace NUMINAMATH_GPT_ryan_hours_difference_l1077_107788

theorem ryan_hours_difference :
  let hours_english := 6
  let hours_chinese := 7
  hours_chinese - hours_english = 1 := 
by
  -- this is where the proof steps would go
  sorry

end NUMINAMATH_GPT_ryan_hours_difference_l1077_107788


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1077_107706

-- Defining our sets M and N based on the conditions provided
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x^2 < 4 }

-- The statement we want to prove
theorem intersection_of_M_and_N :
  M ∩ N = { x | -2 < x ∧ x < 1 } :=
sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1077_107706


namespace NUMINAMATH_GPT_rectangle_perimeter_l1077_107783

theorem rectangle_perimeter (s : ℕ) (ABCD_area : 4 * s * s = 400) :
  2 * (2 * s + 2 * s) = 80 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1077_107783


namespace NUMINAMATH_GPT_customer_paid_l1077_107795

theorem customer_paid (cost_price : ℕ) (markup_percent : ℕ) (selling_price : ℕ) : 
  cost_price = 6672 → markup_percent = 25 → selling_price = cost_price + (markup_percent * cost_price / 100) → selling_price = 8340 :=
by
  intros h_cost_price h_markup_percent h_selling_price
  rw [h_cost_price, h_markup_percent] at h_selling_price
  exact h_selling_price

end NUMINAMATH_GPT_customer_paid_l1077_107795


namespace NUMINAMATH_GPT_balance_balls_l1077_107747

variables (R B O P : ℝ)

-- Conditions
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 6 * B
axiom h3 : 8 * B = 6 * P

-- Proof problem
theorem balance_balls : 5 * R + 3 * O + 3 * P = 20 * B :=
by sorry

end NUMINAMATH_GPT_balance_balls_l1077_107747


namespace NUMINAMATH_GPT_largest_value_fraction_l1077_107724

noncomputable def largest_value (x y : ℝ) : ℝ := (x + y) / x

theorem largest_value_fraction
  (x y : ℝ)
  (hx1 : -5 ≤ x)
  (hx2 : x ≤ -3)
  (hy1 : 3 ≤ y)
  (hy2 : y ≤ 5)
  (hy_odd : ∃ k : ℤ, y = 2 * k + 1) :
  largest_value x y = 0.4 :=
sorry

end NUMINAMATH_GPT_largest_value_fraction_l1077_107724


namespace NUMINAMATH_GPT_angle_measure_of_three_times_complementary_l1077_107726

def is_complementary (α β : ℝ) : Prop := α + β = 90

def three_times_complement (α : ℝ) : Prop := 
  ∃ β : ℝ, is_complementary α β ∧ α = 3 * β

theorem angle_measure_of_three_times_complementary :
  ∀ α : ℝ, three_times_complement α → α = 67.5 :=
by sorry

end NUMINAMATH_GPT_angle_measure_of_three_times_complementary_l1077_107726


namespace NUMINAMATH_GPT_bag_cost_is_10_l1077_107777

def timothy_initial_money : ℝ := 50
def tshirt_cost : ℝ := 8
def keychain_cost : ℝ := 2
def keychains_per_set : ℝ := 3
def number_of_tshirts : ℝ := 2
def number_of_bags : ℝ := 2
def number_of_keychains : ℝ := 21

noncomputable def cost_of_each_bag : ℝ :=
  let cost_of_tshirts := number_of_tshirts * tshirt_cost
  let remaining_money_after_tshirts := timothy_initial_money - cost_of_tshirts
  let cost_of_keychains := (number_of_keychains / keychains_per_set) * keychain_cost
  let remaining_money_after_keychains := remaining_money_after_tshirts - cost_of_keychains
  remaining_money_after_keychains / number_of_bags

theorem bag_cost_is_10 :
  cost_of_each_bag = 10 := by
  sorry

end NUMINAMATH_GPT_bag_cost_is_10_l1077_107777


namespace NUMINAMATH_GPT_max_obtuse_in_convex_quadrilateral_l1077_107758

-- Definition and problem statement
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

def convex_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c + d = 360 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

theorem max_obtuse_in_convex_quadrilateral (a b c d : ℝ) :
  convex_quadrilateral a b c d →
  (is_obtuse a → (is_obtuse b → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse b → (is_obtuse a → ¬ (is_obtuse c ∧ is_obtuse d))) →
  (is_obtuse c → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse d))) →
  (is_obtuse d → (is_obtuse a → ¬ (is_obtuse b ∧ is_obtuse c))) :=
by
  intros h_convex h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_max_obtuse_in_convex_quadrilateral_l1077_107758


namespace NUMINAMATH_GPT_geometric_sequence_third_term_l1077_107729

theorem geometric_sequence_third_term (a : ℕ → ℝ) (r : ℝ)
  (h : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_third_term_l1077_107729


namespace NUMINAMATH_GPT_exists_infinite_arith_prog_exceeding_M_l1077_107743

def sum_of_digits(n : ℕ) : ℕ :=
n.digits 10 |> List.sum

theorem exists_infinite_arith_prog_exceeding_M (M : ℝ) :
  ∃ (a d : ℕ), ¬ (10 ∣ d) ∧ (∀ n : ℕ, a + n * d > 0) ∧ (∀ n : ℕ, sum_of_digits (a + n * d) > M) := by
sorry

end NUMINAMATH_GPT_exists_infinite_arith_prog_exceeding_M_l1077_107743


namespace NUMINAMATH_GPT_average_speed_of_car_l1077_107791

theorem average_speed_of_car 
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (total_time : ℕ)
  (h1 : speed_first_hour = 90)
  (h2 : speed_second_hour = 40)
  (h3 : total_time = 2) : 
  (speed_first_hour + speed_second_hour) / total_time = 65 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_of_car_l1077_107791


namespace NUMINAMATH_GPT_cities_with_fewer_than_500000_residents_l1077_107716

theorem cities_with_fewer_than_500000_residents (P Q R : ℕ) 
  (h1 : P + Q + R = 100) 
  (h2 : P = 40) 
  (h3 : Q = 35) 
  (h4 : R = 25) : P + Q = 75 :=
by 
  sorry

end NUMINAMATH_GPT_cities_with_fewer_than_500000_residents_l1077_107716


namespace NUMINAMATH_GPT_square_area_l1077_107714

theorem square_area (perimeter : ℝ) (h : perimeter = 32) : 
  ∃ (side area : ℝ), side = perimeter / 4 ∧ area = side * side ∧ area = 64 := 
by
  sorry

end NUMINAMATH_GPT_square_area_l1077_107714


namespace NUMINAMATH_GPT_quadratic_increasing_for_x_geq_3_l1077_107779

theorem quadratic_increasing_for_x_geq_3 (x : ℝ) : 
  x ≥ 3 → y = 2 * (x - 3)^2 - 1 → ∃ d > 0, ∀ p ≥ x, y ≤ 2 * (p - 3)^2 - 1 := sorry

end NUMINAMATH_GPT_quadratic_increasing_for_x_geq_3_l1077_107779


namespace NUMINAMATH_GPT_series_sum_l1077_107707

noncomputable def S (n : ℕ) : ℝ := 2^(n + 1) + n - 2

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then S 1 else S n - S (n - 1)

theorem series_sum : 
  ∑' i, a i / 4^i = 4 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_series_sum_l1077_107707


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_length_l1077_107732

theorem right_triangle_hypotenuse_length (a b c : ℝ) (h₀ : a = 7) (h₁ : b = 24) (h₂ : a^2 + b^2 = c^2) : c = 25 :=
by
  rw [h₀, h₁] at h₂
  -- This step will simplify the problem
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_length_l1077_107732


namespace NUMINAMATH_GPT_add_multiply_round_l1077_107782

theorem add_multiply_round :
  let a := 73.5891
  let b := 24.376
  let c := (a + b) * 2
  (Float.round (c * 100) / 100) = 195.93 :=
by
  sorry

end NUMINAMATH_GPT_add_multiply_round_l1077_107782


namespace NUMINAMATH_GPT_contradiction_example_l1077_107736

theorem contradiction_example (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2) : ¬ (a < 0 ∧ b < 0) :=
by
  -- The proof goes here, but we just need the statement
  sorry

end NUMINAMATH_GPT_contradiction_example_l1077_107736


namespace NUMINAMATH_GPT_total_pears_l1077_107733

theorem total_pears (Alyssa_picked Nancy_picked : ℕ) (h₁ : Alyssa_picked = 42) (h₂ : Nancy_picked = 17) : Alyssa_picked + Nancy_picked = 59 :=
by
  sorry

end NUMINAMATH_GPT_total_pears_l1077_107733


namespace NUMINAMATH_GPT_rate_of_stream_l1077_107764

theorem rate_of_stream (v : ℝ) (h : 126 = (16 + v) * 6) : v = 5 :=
by 
  sorry

end NUMINAMATH_GPT_rate_of_stream_l1077_107764


namespace NUMINAMATH_GPT_total_reduction_500_l1077_107725

noncomputable def total_price_reduction (P : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : ℝ :=
  let first_reduction := P * first_reduction_percent / 100
  let intermediate_price := P - first_reduction
  let second_reduction := intermediate_price * second_reduction_percent / 100
  let final_price := intermediate_price - second_reduction
  P - final_price

theorem total_reduction_500 (P : ℝ) (first_reduction_percent : ℝ)  (second_reduction_percent: ℝ) (h₁ : P = 500) (h₂ : first_reduction_percent = 5) (h₃ : second_reduction_percent = 4):
  total_price_reduction P first_reduction_percent second_reduction_percent = 44 := 
by
  sorry

end NUMINAMATH_GPT_total_reduction_500_l1077_107725


namespace NUMINAMATH_GPT_cylinder_surface_area_l1077_107789

noncomputable def surface_area_of_cylinder (r l : ℝ) : ℝ :=
  2 * Real.pi * r * (r + l)

theorem cylinder_surface_area (r : ℝ) (h_radius : r = 1) (l : ℝ) (h_length : l = 2 * r) :
  surface_area_of_cylinder r l = 6 * Real.pi := by
  -- Using the given conditions and definition, we need to prove the surface area is 6π
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l1077_107789


namespace NUMINAMATH_GPT_ratio_of_other_triangle_to_square_l1077_107754

noncomputable def ratio_of_triangle_areas (m : ℝ) : ℝ :=
  let side_of_square := 2
  let area_of_square := side_of_square ^ 2
  let area_of_smaller_triangle := m * area_of_square
  let r := area_of_smaller_triangle / (side_of_square / 2)
  let s := side_of_square * side_of_square / r
  let area_of_other_triangle := side_of_square * s / 2
  area_of_other_triangle / area_of_square

theorem ratio_of_other_triangle_to_square (m : ℝ) (h : m > 0) :
  ratio_of_triangle_areas m = 1 / (4 * m) :=
sorry

end NUMINAMATH_GPT_ratio_of_other_triangle_to_square_l1077_107754


namespace NUMINAMATH_GPT_triangle_relation_l1077_107717

theorem triangle_relation (A B C a b : ℝ) (h : 4 * A = B ∧ B = C) (hABC : A + B + C = 180) : 
  a^3 + b^3 = 3 * a * b^2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_relation_l1077_107717


namespace NUMINAMATH_GPT_oranges_per_child_l1077_107785

theorem oranges_per_child (children oranges : ℕ) (h1 : children = 4) (h2 : oranges = 12) : oranges / children = 3 := by
  sorry

end NUMINAMATH_GPT_oranges_per_child_l1077_107785


namespace NUMINAMATH_GPT_num_bicycles_eq_20_l1077_107794

-- Definitions based on conditions
def num_cars : ℕ := 10
def num_motorcycles : ℕ := 5
def total_wheels : ℕ := 90
def wheels_per_bicycle : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_motorcycle : ℕ := 2

-- Statement to prove
theorem num_bicycles_eq_20 (B : ℕ) 
  (h_wheels_from_bicycles : wheels_per_bicycle * B = 2 * B)
  (h_wheels_from_cars : num_cars * wheels_per_car = 40)
  (h_wheels_from_motorcycles : num_motorcycles * wheels_per_motorcycle = 10)
  (h_total_wheels : wheels_per_bicycle * B + 40 + 10 = total_wheels) :
  B = 20 :=
sorry

end NUMINAMATH_GPT_num_bicycles_eq_20_l1077_107794


namespace NUMINAMATH_GPT_tan_alpha_l1077_107781

theorem tan_alpha (α : ℝ) (h1 : Real.sin (Real.pi - α) = 1 / 3) (h2 : Real.sin (2 * α) > 0) : 
  Real.tan α = Real.sqrt 2 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_l1077_107781


namespace NUMINAMATH_GPT_third_consecutive_odd_integers_is_fifteen_l1077_107721

theorem third_consecutive_odd_integers_is_fifteen :
  ∃ x : ℤ, (x % 2 = 1 ∧ (x + 2) % 2 = 1 ∧ (x + 4) % 2 = 1) ∧ (x + 2 + (x + 4) = x + 17) → (x + 4 = 15) :=
by
  sorry

end NUMINAMATH_GPT_third_consecutive_odd_integers_is_fifteen_l1077_107721


namespace NUMINAMATH_GPT_exists_almost_square_divides_2010_l1077_107793

noncomputable def almost_square (a b : ℕ) : Prop :=
  (a = b + 1 ∨ b = a + 1) ∧ a * b = 2010

theorem exists_almost_square_divides_2010 :
  ∃ (a b : ℕ), almost_square a b :=
sorry

end NUMINAMATH_GPT_exists_almost_square_divides_2010_l1077_107793


namespace NUMINAMATH_GPT_factor_polynomial_l1077_107774

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1077_107774


namespace NUMINAMATH_GPT_total_cats_l1077_107786

def initial_siamese_cats : Float := 13.0
def initial_house_cats : Float := 5.0
def added_cats : Float := 10.0

theorem total_cats : initial_siamese_cats + initial_house_cats + added_cats = 28.0 := by
  sorry

end NUMINAMATH_GPT_total_cats_l1077_107786


namespace NUMINAMATH_GPT_toppings_combination_l1077_107790

-- Define the combination function
def combination (n k : ℕ) : ℕ := n.choose k

theorem toppings_combination :
  combination 9 3 = 84 := by
  sorry

end NUMINAMATH_GPT_toppings_combination_l1077_107790


namespace NUMINAMATH_GPT_bus_speed_incl_stoppages_l1077_107768

theorem bus_speed_incl_stoppages (v_excl : ℝ) (minutes_stopped : ℝ) :
  v_excl = 64 → minutes_stopped = 13.125 →
  v_excl - (v_excl * (minutes_stopped / 60)) = 50 :=
by
  intro v_excl_eq minutes_stopped_eq
  rw [v_excl_eq, minutes_stopped_eq]
  have hours_stopped : ℝ := 13.125 / 60
  have distance_lost : ℝ := 64 * hours_stopped
  have v_incl := 64 - distance_lost
  sorry

end NUMINAMATH_GPT_bus_speed_incl_stoppages_l1077_107768


namespace NUMINAMATH_GPT_not_prime_for_all_n_ge_2_l1077_107728

theorem not_prime_for_all_n_ge_2 (n : ℕ) (hn : n ≥ 2) : ¬ Prime (2 * (n^3 + n + 1)) := 
by
  sorry

end NUMINAMATH_GPT_not_prime_for_all_n_ge_2_l1077_107728


namespace NUMINAMATH_GPT_max_c_val_l1077_107756

theorem max_c_val (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : 2 * a * b = 2 * a + b) 
  (h2 : a * b * c = 2 * a + b + c) :
  c ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_c_val_l1077_107756


namespace NUMINAMATH_GPT_simplify_fraction_l1077_107780

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) : 
  (2 * a / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4)) := 
by 
  sorry 

end NUMINAMATH_GPT_simplify_fraction_l1077_107780


namespace NUMINAMATH_GPT_trigonometric_identity_l1077_107784

noncomputable def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem trigonometric_identity :
  special_operation (Real.sin (Real.pi / 12)) (Real.cos (Real.pi / 12))
  = - (1 + 2 * Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1077_107784


namespace NUMINAMATH_GPT_polynomial_simplification_l1077_107749

theorem polynomial_simplification (x : ℝ) :
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l1077_107749


namespace NUMINAMATH_GPT_speed_of_faster_train_l1077_107766

theorem speed_of_faster_train
  (length_each_train : ℕ)
  (length_in_meters : length_each_train = 50)
  (speed_slower_train_kmh : ℝ)
  (speed_slower : speed_slower_train_kmh = 36)
  (pass_time_seconds : ℕ)
  (pass_time : pass_time_seconds = 36) :
  ∃ speed_faster_train_kmh, speed_faster_train_kmh = 46 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_faster_train_l1077_107766


namespace NUMINAMATH_GPT_find_X_l1077_107712

theorem find_X (k : ℝ) (R1 R2 X1 X2 Y1 Y2 : ℝ) (h1 : R1 = k * (X1 / Y1)) (h2 : R1 = 10) (h3 : X1 = 2) (h4 : Y1 = 4) (h5 : R2 = 8) (h6 : Y2 = 5) : X2 = 2 :=
sorry

end NUMINAMATH_GPT_find_X_l1077_107712


namespace NUMINAMATH_GPT_correct_order_l1077_107730

noncomputable def f : ℝ → ℝ := sorry

axiom periodic : ∀ x : ℝ, f (x + 4) = f x
axiom increasing : ∀ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ x₁ < 2) → (0 ≤ x₂ ∧ x₂ ≤ 2) → x₁ < x₂ → f x₁ < f x₂
axiom symmetric : ∀ x : ℝ, f (x + 2) = f (2 - x)

theorem correct_order : f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  sorry

end NUMINAMATH_GPT_correct_order_l1077_107730


namespace NUMINAMATH_GPT_more_uniform_team_l1077_107762

-- Define the parameters and the variances
def average_height := 1.85
def variance_team_A := 0.32
def variance_team_B := 0.26

-- Main theorem statement
theorem more_uniform_team : variance_team_B < variance_team_A → "Team B" = "Team with more uniform heights" :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_more_uniform_team_l1077_107762


namespace NUMINAMATH_GPT_triangle_max_area_proof_l1077_107745

open Real

noncomputable def triangle_max_area (A B C : ℝ) (AB : ℝ) (tanA tanB : ℝ) : Prop :=
  AB = 4 ∧ tanA * tanB = 3 / 4 → ∃ S : ℝ, S = 2 * sqrt 3

theorem triangle_max_area_proof (A B C : ℝ) (tanA tanB : ℝ) (AB : ℝ) : 
  triangle_max_area A B C AB tanA tanB :=
by
  sorry

end NUMINAMATH_GPT_triangle_max_area_proof_l1077_107745


namespace NUMINAMATH_GPT_time_for_A_l1077_107705

-- Given rates of pipes A, B, and C filling the tank
variable (A B C : ℝ)

-- Condition 1: Tank filled by all three pipes in 8 hours
def combined_rate := (A + B + C = 1/8)

-- Condition 2: Pipe C is twice as fast as B
def rate_C := (C = 2 * B)

-- Condition 3: Pipe B is twice as fast as A
def rate_B := (B = 2 * A)

-- Question: To prove that pipe A alone will take 56 hours to fill the tank
theorem time_for_A (h₁ : combined_rate A B C) (h₂ : rate_C B C) (h₃ : rate_B A B) : 
  1 / A = 56 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_for_A_l1077_107705


namespace NUMINAMATH_GPT_max_sum_cos_isosceles_triangle_l1077_107741

theorem max_sum_cos_isosceles_triangle :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ (2 * Real.cos α + Real.cos (π - 2 * α)) ≤ 1.5 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_cos_isosceles_triangle_l1077_107741


namespace NUMINAMATH_GPT_max_x_minus_y_l1077_107710

theorem max_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x^2 + y) :
  x - y ≤ 1 / Real.sqrt 24 :=
sorry

end NUMINAMATH_GPT_max_x_minus_y_l1077_107710


namespace NUMINAMATH_GPT_largest_y_coordinate_on_graph_l1077_107742

theorem largest_y_coordinate_on_graph :
  ∀ x y : ℝ, (x / 7) ^ 2 + ((y - 3) / 5) ^ 2 = 0 → y ≤ 3 := 
by
  intro x y h
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_on_graph_l1077_107742


namespace NUMINAMATH_GPT_edge_length_of_box_l1077_107723

noncomputable def edge_length_cubical_box (num_cubes : ℕ) (edge_length_cube : ℝ) : ℝ :=
  if num_cubes = 8 ∧ edge_length_cube = 0.5 then -- 50 cm in meters
    1 -- The edge length of the cubical box in meters
  else
    0 -- Placeholder for other cases

theorem edge_length_of_box :
  edge_length_cubical_box 8 0.5 = 1 :=
sorry

end NUMINAMATH_GPT_edge_length_of_box_l1077_107723


namespace NUMINAMATH_GPT_problem_condition_l1077_107757

variable (x y z : ℝ)

theorem problem_condition (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end NUMINAMATH_GPT_problem_condition_l1077_107757


namespace NUMINAMATH_GPT_correct_power_functions_l1077_107752

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), ∀ x, x ≠ 0 → f x = k * x^n

def f1 (x : ℝ) : ℝ := x^2 + 2
def f2 (x : ℝ) : ℝ := x^(1 / 2)
def f3 (x : ℝ) : ℝ := 2 * x^3
def f4 (x : ℝ) : ℝ := x^(3 / 4)
def f5 (x : ℝ) : ℝ := x^(1 / 3) + 1

theorem correct_power_functions :
  {f2, f4} = {f : ℝ → ℝ | is_power_function f} ∩ {f2, f4, f1, f3, f5} :=
by
  sorry

end NUMINAMATH_GPT_correct_power_functions_l1077_107752


namespace NUMINAMATH_GPT_complement_union_example_l1077_107775

open Set

variable (I : Set ℕ) (A : Set ℕ) (B : Set ℕ)

noncomputable def complement (U : Set ℕ) (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem complement_union_example
    (hI : I = {0, 1, 2, 3, 4})
    (hA : A = {0, 1, 2, 3})
    (hB : B = {2, 3, 4}) :
    (complement I A) ∪ (complement I B) = {0, 1, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_example_l1077_107775


namespace NUMINAMATH_GPT_product_of_roots_l1077_107748

theorem product_of_roots (a b c : ℤ) (h_eqn : a = 12 ∧ b = 60 ∧ c = -720) :
  (c : ℚ) / a = -60 :=
by sorry

end NUMINAMATH_GPT_product_of_roots_l1077_107748


namespace NUMINAMATH_GPT_cos_240_is_neg_half_l1077_107763

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end NUMINAMATH_GPT_cos_240_is_neg_half_l1077_107763


namespace NUMINAMATH_GPT_ted_age_l1077_107702

variable (t s : ℕ)

theorem ted_age (h1 : t = 3 * s - 10) (h2 : t + s = 65) : t = 46 := by
  sorry

end NUMINAMATH_GPT_ted_age_l1077_107702


namespace NUMINAMATH_GPT_min_area_triangle_l1077_107700

-- Conditions
def point_on_curve (x y : ℝ) : Prop :=
  y^2 = 2 * x

def incircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem min_area_triangle (x₀ y₀ b c : ℝ) (h_curve : point_on_curve x₀ y₀) 
  (h_bc_yaxis : b ≠ c) (h_incircle : incircle x₀ y₀) :
  ∃ P : ℝ × ℝ, 
    ∃ B C : ℝ × ℝ, 
    ∃ S : ℝ,
    point_on_curve P.1 P.2 ∧
    B = (0, b) ∧
    C = (0, c) ∧
    incircle P.1 P.2 ∧
    S = (x₀ - 2) + (4 / (x₀ - 2)) + 4 ∧
    S = 8 :=
sorry

end NUMINAMATH_GPT_min_area_triangle_l1077_107700


namespace NUMINAMATH_GPT_xy_proposition_l1077_107718

theorem xy_proposition (x y : ℝ) : (x + y ≥ 5) → (x ≥ 3 ∨ y ≥ 2) :=
sorry

end NUMINAMATH_GPT_xy_proposition_l1077_107718


namespace NUMINAMATH_GPT_even_function_symmetric_y_axis_l1077_107761

theorem even_function_symmetric_y_axis (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = f (-x) := by
  sorry

end NUMINAMATH_GPT_even_function_symmetric_y_axis_l1077_107761


namespace NUMINAMATH_GPT_sea_creatures_lost_l1077_107799

theorem sea_creatures_lost (sea_stars : ℕ) (seashells : ℕ) (snails : ℕ) (items_left : ℕ)
  (h1 : sea_stars = 34) (h2 : seashells = 21) (h3 : snails = 29) (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end NUMINAMATH_GPT_sea_creatures_lost_l1077_107799


namespace NUMINAMATH_GPT_number_of_pies_is_correct_l1077_107797

def weight_of_apples : ℕ := 120
def weight_for_applesauce (w : ℕ) : ℕ := w / 2
def weight_for_pies (w wholly_app : ℕ) : ℕ := w - wholly_app
def pies (weight_per_pie total_weight : ℕ) : ℕ := total_weight / weight_per_pie

theorem number_of_pies_is_correct :
  pies 4 (weight_for_pies weight_of_apples (weight_for_applesauce weight_of_apples)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pies_is_correct_l1077_107797


namespace NUMINAMATH_GPT_positive_solution_sqrt_eq_l1077_107798

theorem positive_solution_sqrt_eq (y : ℝ) (hy_pos : 0 < y) : 
    (∃ a, a = y ∧ a^2 = y * a) ∧ (∃ b, b = y ∧ b^2 = y + b) ∧ y = 2 :=
by 
  sorry

end NUMINAMATH_GPT_positive_solution_sqrt_eq_l1077_107798


namespace NUMINAMATH_GPT_smallest_yummy_is_minus_2013_l1077_107703

-- Define a yummy integer
def is_yummy (A : ℤ) : Prop :=
  ∃ (k : ℕ), ∃ (a : ℤ), (a <= A) ∧ (a + k = A) ∧ ((k + 1) * A - k*(k + 1)/2 = 2014)

-- Define the smallest yummy integer
def smallest_yummy : ℤ :=
  -2013

-- The Lean theorem to state the proof problem
theorem smallest_yummy_is_minus_2013 : ∀ A : ℤ, is_yummy A → (-2013 ≤ A) :=
by
  sorry

end NUMINAMATH_GPT_smallest_yummy_is_minus_2013_l1077_107703


namespace NUMINAMATH_GPT_solve_equation_l1077_107759

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) ↔ x = -2 ∨ x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1077_107759


namespace NUMINAMATH_GPT_factorize1_factorize2_l1077_107787

-- Part 1: Prove the factorization of xy - 1 - x + y
theorem factorize1 (x y : ℝ) : (x * y - 1 - x + y) = (y - 1) * (x + 1) :=
  sorry

-- Part 2: Prove the factorization of (a^2 + b^2)^2 - 4a^2b^2
theorem factorize2 (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
  sorry

end NUMINAMATH_GPT_factorize1_factorize2_l1077_107787


namespace NUMINAMATH_GPT_find_set_C_l1077_107776

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def C : Set ℝ := {a | B a ⊆ A}

theorem find_set_C : C = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_find_set_C_l1077_107776


namespace NUMINAMATH_GPT_smallest_nonnegative_a_l1077_107778

open Real

theorem smallest_nonnegative_a (a b : ℝ) (h_b : b = π / 4)
(sin_eq : ∀ (x : ℤ), sin (a * x + b) = sin (17 * x)) : 
a = 17 - π / 4 := by 
  sorry

end NUMINAMATH_GPT_smallest_nonnegative_a_l1077_107778


namespace NUMINAMATH_GPT_area_of_shaded_region_l1077_107755

theorem area_of_shaded_region :
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  shaded_area = 22 :=
by
  let ABCD_area := 36
  let EFGH_area := 1 * 3
  let IJKL_area := 2 * 4
  let MNOP_area := 3 * 1
  let shaded_area := ABCD_area - (EFGH_area + IJKL_area + MNOP_area)
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1077_107755


namespace NUMINAMATH_GPT_average_of_last_four_numbers_l1077_107735

theorem average_of_last_four_numbers
  (avg_seven : ℝ) (avg_first_three : ℝ) (avg_last_four : ℝ)
  (h1 : avg_seven = 62) (h2 : avg_first_three = 55) :
  avg_last_four = 67.25 := 
by
  sorry

end NUMINAMATH_GPT_average_of_last_four_numbers_l1077_107735


namespace NUMINAMATH_GPT_factory_production_l1077_107719

theorem factory_production (y x : ℝ) (h1 : y + 40 * x = 1.2 * y) (h2 : y + 0.6 * y * x = 2.5 * y) 
  (hx : x = 2.5) : y = 500 ∧ 1 + x = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_factory_production_l1077_107719


namespace NUMINAMATH_GPT_find_TU_square_l1077_107771

-- Definitions
variables (P Q R S T U : ℝ × ℝ)
variable (side : ℝ)
variable (QT RU PT SU PQ : ℝ)

-- Setting the conditions
variables (side_eq_10 : side = 10)
variables (QT_eq_7 : QT = 7)
variables (RU_eq_7 : RU = 7)
variables (PT_eq_24 : PT = 24)
variables (SU_eq_24 : SU = 24)
variables (PQ_eq_10 : PQ = 10)

-- The theorem statement
theorem find_TU_square : TU^2 = 1150 :=
by
  -- Proof to be done here.
  sorry

end NUMINAMATH_GPT_find_TU_square_l1077_107771


namespace NUMINAMATH_GPT_total_points_correct_l1077_107769

-- Define the number of teams
def num_teams : ℕ := 16

-- Define the number of draws
def num_draws : ℕ := 30

-- Define the scoring system
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def loss_deduction_threshold : ℕ := 3
def points_deduction_per_threshold : ℕ := 1

-- Define the total number of games
def total_games : ℕ := num_teams * (num_teams - 1) / 2

-- Define the number of wins (non-draw games)
def num_wins : ℕ := total_games - num_draws

-- Define the total points from wins
def total_points_from_wins : ℕ := num_wins * points_for_win

-- Define the total points from draws
def total_points_from_draws : ℕ := num_draws * points_for_draw * 2

-- Define the total points (as no team lost more than twice, no deductions apply)
def total_points : ℕ := total_points_from_wins + total_points_from_draws

theorem total_points_correct :
  total_points = 330 := by
  sorry

end NUMINAMATH_GPT_total_points_correct_l1077_107769


namespace NUMINAMATH_GPT_closest_vector_l1077_107715

theorem closest_vector 
  (s : ℝ)
  (u b d : ℝ × ℝ × ℝ)
  (h₁ : u = (3, -2, 4) + s • (6, 4, 2))
  (h₂ : b = (1, 7, 6))
  (hdir : d = (6, 4, 2))
  (h₃ : (u - b) = (2 + 6 * s, -9 + 4 * s, -2 + 2 * s)) :
  ((2 + 6 * s) * 6 + (-9 + 4 * s) * 4 + (-2 + 2 * s) * 2) = 0 →
  s = 1 / 2 :=
by
  -- Skipping the proof, adding sorry
  sorry

end NUMINAMATH_GPT_closest_vector_l1077_107715


namespace NUMINAMATH_GPT_functional_equation_solution_form_l1077_107796

noncomputable def functional_equation_problem (f : ℝ → ℝ) :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem functional_equation_solution_form :
  (∀ f : ℝ → ℝ, (functional_equation_problem f) → (∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ 2 + b * x)) :=
by 
  sorry

end NUMINAMATH_GPT_functional_equation_solution_form_l1077_107796


namespace NUMINAMATH_GPT_num_valid_n_l1077_107792

theorem num_valid_n (n q r : ℤ) (h₁ : 10000 ≤ n) (h₂ : n ≤ 99999)
  (h₃ : n = 50 * q + r) (h₄ : 200 ≤ q) (h₅ : q ≤ 1999)
  (h₆ : 0 ≤ r) (h₇ : r < 50) :
  (∃ (count : ℤ), count = 14400) := by
  sorry

end NUMINAMATH_GPT_num_valid_n_l1077_107792


namespace NUMINAMATH_GPT_other_solution_of_quadratic_l1077_107737

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop :=
  65 * x^2 - 104 * x + 31 = 0

-- Main theorem statement
theorem other_solution_of_quadratic :
  quadratic_eq (6 / 5) → quadratic_eq (5 / 13) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_other_solution_of_quadratic_l1077_107737


namespace NUMINAMATH_GPT_prob_score_at_most_7_l1077_107773

-- Definitions based on the conditions
def prob_10_ring : ℝ := 0.15
def prob_9_ring : ℝ := 0.35
def prob_8_ring : ℝ := 0.2
def prob_7_ring : ℝ := 0.1

-- Define the event of scoring no more than 7
def score_at_most_7 := prob_7_ring

-- Theorem statement
theorem prob_score_at_most_7 : score_at_most_7 = 0.1 := by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_prob_score_at_most_7_l1077_107773


namespace NUMINAMATH_GPT_elephants_ratio_l1077_107740

theorem elephants_ratio (x : ℝ) (w : ℝ) (g : ℝ) (total : ℝ) :
  w = 70 →
  total = 280 →
  g = x * w →
  w + g = total →
  x = 3 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_elephants_ratio_l1077_107740


namespace NUMINAMATH_GPT_initial_observations_l1077_107704

theorem initial_observations (n : ℕ) (S : ℕ) (new_obs : ℕ) :
  (S = 12 * n) → (new_obs = 5) → (S + new_obs = 11 * (n + 1)) → n = 6 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_initial_observations_l1077_107704


namespace NUMINAMATH_GPT_mom_approach_is_sampling_survey_l1077_107753

def is_sampling_survey (action : String) : Prop :=
  action = "tasting a little bit"

def is_census (action : String) : Prop :=
  action = "tasting the entire dish"

theorem mom_approach_is_sampling_survey :
  is_sampling_survey "tasting a little bit" :=
by {
  -- This follows from the given conditions directly.
  sorry
}

end NUMINAMATH_GPT_mom_approach_is_sampling_survey_l1077_107753


namespace NUMINAMATH_GPT_range_of_m_l1077_107750

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : y1 = x1^2 - 4*x1 + 3)
  (h2 : y2 = x2^2 - 4*x2 + 3) (h3 : -1 < x1) (h4 : x1 < 1)
  (h5 : m > 0) (h6 : m-1 < x2) (h7 : x2 < m) (h8 : y1 ≠ y2) :
  (2 ≤ m ∧ m ≤ 3) ∨ (m ≥ 6) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1077_107750


namespace NUMINAMATH_GPT_sum_of_digits_l1077_107760

theorem sum_of_digits (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h3 : 34 * a + 42 * b = 142) : a + b = 4 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_l1077_107760


namespace NUMINAMATH_GPT_malcolm_walked_uphill_l1077_107713

-- Define the conditions as variables and parameters
variables (x : ℕ)

-- Define the conditions given in the problem
def first_route_time := x + 2 * x + x
def second_route_time := 14 + 28
def time_difference := 18

-- Theorem statement - proving that Malcolm walked uphill for 6 minutes in the first route
theorem malcolm_walked_uphill : first_route_time - second_route_time = time_difference → x = 6 := by
  sorry

end NUMINAMATH_GPT_malcolm_walked_uphill_l1077_107713


namespace NUMINAMATH_GPT_average_of_all_digits_l1077_107734

theorem average_of_all_digits {a b : ℕ} (n : ℕ) (x y : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : n = 10) (h4 : x = 58) (h5 : y = 113) :
  ((a * x + b * y) / n = 80) :=
  sorry

end NUMINAMATH_GPT_average_of_all_digits_l1077_107734
